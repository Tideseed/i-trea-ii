"""The self-adaptive repair strategy (`EAConfig.strategy`, `itrea.ea.engine`).

Every individual carries a strategy = (repair row scale, LP guidance); children inherit parent A's, redraw it with
`mutation_rate`, every strategy keeps `min_share` of the offspring. One mutate call per generation (children without
LP guidance never draw toward_lp) and one local-search call per repair, with a per-individual index into the search's
(R, m_aug) row-scale table. The survival's V keeps the one scale `ranks.viol_norm`.

The first tests are the regression: with `strategy.enabled: false` the engine must reproduce, column by column, a short
run recorded with the engine as it was BEFORE the feature (`tests/_engine_reference.py`, fixture written at dd31d67).

Toy models only; GPU tests (marker `gpu`) use a 1 GB CuPy pool; the HiGHS LP worker test is marked `solver`.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
import scipy.sparse as sps

import _engine_reference as REF
from itrea.benchmarks import toy
from itrea.config import EAConfig, StrategyCfg
from itrea.ea import engine as E
from itrea.ea import mutation as M
from itrea.ea.engine import Engine, plan_memory, strategy_ls_bytes
from itrea.ea.evaluate import row_scale_table, row_scale_vector, to_device
from itrea.ea.population import ORIGIN_INJECTED, ORIGIN_M_TOWARD_LP, ORIGIN_SEED_LP, gene_dtype_for
from itrea.ea.repair import LocalSearch
from itrea.exchange.mailbox import Message
from itrea.problem import from_arrays
from itrea.trace import Trace, read_events
from test_engine import check_invariants, gpu_or_skip, knapsack_optimum_x, run_checked

SMALL = {"population.size": 48, "population.offspring_per_gen": 48, "gpu_mem_gb": 1.0}
ON = {"strategy.enabled": True}


def cfg_for(backend: str = "numpy", **over) -> EAConfig:
    d = dict(SMALL, backend=backend, **ON)
    d.update(over)
    return EAConfig().with_overrides(d)


def prepared(p, cfg, seed=0, mailbox=None, trace=None, warmup=False) -> Engine:
    eng = Engine(p, cfg, seed, mailbox=mailbox, trace=trace, warmup=warmup)
    eng.prepare()
    return eng


def with_lp(p) -> REF.LPMailbox:
    lp = REF._lp_point(p)
    return REF.LPMailbox(lp, float(p.objective(lp)))


def scaled_cover(n: int = 30, m: int = 24, seed: int = 0):
    """Covering rows a_i·x >= b_i (x in {0..3}) whose coefficient magnitudes (row factors 1..1000) and right-hand sides
    (5 % to 60 % of the row's maximum activity, independently drawn) vary independently, so the l2 and the rhs row
    scales weigh the rows very differently — the two local searches make different moves on it."""
    rs = np.random.default_rng(seed)
    A = (rs.random((m, n)) < 0.25).astype(float)
    A[np.arange(m), rs.integers(n, size=m)] = 1.0
    f = 10.0 ** rs.uniform(0, 3, size=m)
    A = A * np.round(f)[:, None]
    b = np.maximum(1.0, np.round(rs.uniform(0.05, 0.6, size=m) * 3.0 * A.sum(axis=1)))
    c = rs.integers(1, 10, size=n).astype(float)
    return from_arrays(sps.csr_matrix(A), b, np.full(m, np.inf), np.zeros(n), np.full(n, 3.0), c, np.ones(n, bool),
                       names=[f"x{j}" for j in range(n)], path="<toy:scaled_cover>")


# ------------------------------------------------------------------------------------------- regression
def _compare_with_reference(backend: str) -> None:
    ref = REF.load_reference()
    for name in REF.TOYS:
        got = REF.capture(name, backend)
        for k, v in got.items():
            r = ref[f"{backend}/{name}/{k}"]
            assert r.shape == v.shape, (backend, name, k, r.shape, v.shape)
            assert np.array_equal(r, v, equal_nan=True), (backend, name, k)


def test_disabled_matches_the_reference_numpy():
    """strategy.enabled false: the same genealogy, final population, counters and best key as before the feature."""
    assert not REF.config("numpy").strategy.enabled
    _compare_with_reference("numpy")


@pytest.mark.gpu
def test_disabled_matches_the_reference_cupy():
    gpu_or_skip()
    _compare_with_reference("cupy")


def test_disabled_records_no_strategy_fields(tmp_path):
    """Off: no strategy record in the start-up / gen events / summary; the genealogy's strat column is all 0."""
    p = toy.knapsack(30)
    eng = prepared(p, cfg_for(**{"strategy.enabled": False, "log_every": 1}), seed=2, trace=Trace(tmp_path, "ea"))
    assert "strategies" not in eng.startup and "local_search_strategies" not in eng.plan.terms
    run_checked(eng, 6)
    out = eng.finish()
    assert "strategy" not in out
    assert np.all(eng.genealogy()["strat"] == 0)
    eng.trace.close()
    ev = read_events(tmp_path / "ea.events.jsonl")
    assert not any(k.startswith("strat") for e in ev if e["kind"] in ("gen", "best") for k in e)


# ------------------------------------------------------------------------------------------- configuration
def test_strategy_config_validation():
    base = EAConfig()
    assert not base.strategy.enabled                                  # default off
    assert base.strategy.strategies() == [("l2", False), ("l2", True), ("rhs", False), ("rhs", True)]
    assert StrategyCfg(row_scales=["count"], lp_guidance=[True]).strategies() == [("count", True)]
    for bad in ({"strategy.row_scales": []}, {"strategy.row_scales": ["l2", "l2"]}, {"strategy.row_scales": ["l1"]},
                {"strategy.row_scales": "l2"}, {"strategy.lp_guidance": []}, {"strategy.lp_guidance": [True, True]},
                {"strategy.lp_guidance": [0, 1]}, {"strategy.mutation_rate": -0.1}, {"strategy.mutation_rate": 1.1},
                {"strategy.mutation_rate": True}, {"strategy.min_share": -0.01},
                {"strategy.min_share": 0.26},                          # 4 strategies x 0.26 > 1
                {"strategy.enabled": 1}):
        with pytest.raises(ValueError, match="strategy"):
            base.with_overrides(bad)
    with pytest.raises(KeyError):
        base.with_overrides({"strategy.share": 0.1})                  # unknown key
    ok = base.with_overrides({"strategy.min_share": 0.25, "strategy.mutation_rate": 1.0,
                              "strategy.row_scales": ["rhs", "count", "l2"], "strategy.lp_guidance": [True]})
    assert len(ok.strategy.strategies()) == 3


# ------------------------------------------------------------------------------------------- row-scale table
def test_row_scale_table():
    from itrea.backend import get_backend
    p = scaled_cover()
    dp = to_device(p, get_backend("numpy"), viol_norm="l2")
    modes, own = row_scale_table(dp)
    assert modes == ["l2"] and own.shape == (1, p.m)
    np.testing.assert_array_equal(own[0], np.asarray(dp.viol_norm_vec, np.float64))     # exactly V's vector
    modes, tab = row_scale_table(dp, ["rhs", "l2", "count"])
    assert modes == ["rhs", "l2", "count"] and tab.flags.c_contiguous
    np.testing.assert_array_equal(tab[0], row_scale_vector(p, "rhs"))
    np.testing.assert_array_equal(tab[1], own[0])
    np.testing.assert_array_equal(tab[2], np.ones(p.m))
    assert not np.allclose(tab[0], tab[1])                            # the toy discriminates the two scales
    for bad in (["l2", "l2"], ["max"], []):
        with pytest.raises(ValueError):
            row_scale_table(dp, bad)


def _ls(backend: str, method: str, scales, seed: int = 5):
    from itrea.backend import get_backend
    from itrea.config import RepairCfg
    p = scaled_cover()
    bk = get_backend(backend, gpu_mem_gb=1.0)
    dp = to_device(p, bk, viol_norm="l2")
    rp = RepairCfg(method=method, max_steps=60, stall_steps=20, steps=8)
    return p, bk, LocalSearch(bk, dp, rp, 1e-9, 64 << 20, seed=seed, row_scales=scales)


def _small_budget(impl, rows: int = 7) -> None:
    """Shrink a search's memory budget so a call works in chunks of `rows` individuals (FJGpu, DenseLS); FJCpu searches
    one individual at a time and is sliced by `should_stop` instead."""
    if hasattr(impl, "d_rs"):                                         # FJGpu, X float64 C-contiguous: no work copy
        impl.mem_budget_bytes = rows * (24 * impl.m_aug + 80)
        assert impl.chunk_size(40, True) == rows
    elif hasattr(impl, "per_individual_bytes"):                       # DenseLS
        impl.mem_budget_bytes = rows * impl.per_individual_bytes
        assert impl.chunk_size(40) == rows


def _check_mixed_call(backend: str, method: str) -> None:
    """ONE call with a per-row scale index makes, row by row, exactly the moves of a call in which every row has that
    scale (rows are searched independently: FJ seeds each row by its index, the dense baseline masks finished rows);
    scale 0 of the table is the plain search's scale, a one-scale rhs search equals the rhs row; l2 and rhs differ.
    Also with the cutoff row armed, and in chunks of 7 rows (the per-chunk slice of the scale index) / in `should_stop`
    slices (FJCpu)."""
    X0 = np.random.default_rng(1).integers(0, 2, size=(40, 30)).astype(np.float64)
    sidx = np.random.default_rng(2).integers(0, 2, size=40)
    assert 5 < sidx.sum() < 35
    for small in (False, True):
        p, bk, ls = _ls(backend, method, ["l2", "rhs"])
        _, _, plain = _ls(backend, method, None)
        _, _, rhs_only = _ls(backend, method, ["rhs"])
        assert ls.row_scales == ["l2", "rhs"] and plain.row_scales == ["l2"] and rhs_only.row_scales == ["rhs"]
        stop = None
        if small:
            for s in (ls, plain, rhs_only):
                _small_budget(s.impl)
            stop = lambda: False                                       # noqa: E731 — FJCpu: the sliced path
        for cut in (None, 40.0):
            outs = {}
            for name, s, idx in (("l2", ls, np.zeros(40, int)), ("rhs", ls, np.ones(40, int)), ("mix", ls, sidx),
                                 ("plain", plain, None), ("rhs_only", rhs_only, None)):
                s.set_cutoff(cut)
                s.impl._calls = 7                                     # one random stream for every call
                X = bk.xp.asarray(X0.copy())
                st = s.run(X, should_stop=stop, scale_idx=idx)
                assert st["finished_feasible"] + st["finished_stall"] + st["finished_budget"] == 40
                assert st["interrupted"] == 0
                if small and "chunk" in st:
                    assert st["chunk"] == 7
                outs[name] = bk.asnumpy(X)
            assert np.array_equal(outs["l2"], outs["plain"])
            assert np.array_equal(outs["rhs"], outs["rhs_only"])
            np.testing.assert_array_equal(outs["mix"], np.where(sidx[:, None] == 1, outs["rhs"], outs["l2"]))
            assert not np.array_equal(outs["l2"], outs["rhs"]), "the toy does not discriminate the two scales"


@pytest.mark.parametrize("method", ["fj", "dense"])
def test_one_call_repairs_every_row_with_its_own_scale_numpy(method):
    _check_mixed_call("numpy", method)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["fj", "dense"])
def test_one_call_repairs_every_row_with_its_own_scale_cupy(method):
    gpu_or_skip()
    _check_mixed_call("cupy", method)


def _cut_entries(ls):
    """(cutoff row's divisor per scale, cutoff row's upper side) of a LocalSearch's implementation."""
    impl = ls.impl
    if hasattr(impl, "d_rs"):                                         # FJGpu
        tab, ru = impl.dp.bk.asnumpy(impl.d_rs), impl.dp.bk.asnumpy(impl.d_ru)
    elif hasattr(impl, "rs_tab"):                                     # DenseLS
        tab, ru = impl.bk.asnumpy(impl.rs_tab), impl.bk.asnumpy(impl.ru)
    else:                                                             # FJCpu
        tab, ru = impl.rows.rs_tab, impl.rows.ru
    m = impl.m
    return tab[:, m], float(ru[m])


def _check_cutoff_table(backend: str, method: str) -> None:
    p, bk, ls = _ls(backend, method, ["l2", "rhs", "count"])
    cut_l2 = float(np.linalg.norm(p.cmin))
    d, ru = _cut_entries(ls)
    np.testing.assert_allclose(d, [cut_l2, 1.0, 1.0])
    assert math.isinf(ru)
    ls.set_cutoff(40.0)
    d, ru = _cut_entries(ls)
    assert math.isfinite(ru) and ru == ls.impl.cut_rhs
    np.testing.assert_allclose(d, [cut_l2, max(1.0, abs(ru)), 1.0])  # only the rhs row follows the side
    ls.set_cutoff(None)
    d, ru = _cut_entries(ls)
    assert math.isinf(ru)
    np.testing.assert_allclose(d, [cut_l2, 1.0, 1.0])
    with pytest.raises(ValueError):
        ls.run(bk.xp.zeros((3, p.n)), scale_idx=np.array([0, 1]))     # wrong length
    with pytest.raises(ValueError):
        ls.run(bk.xp.zeros((2, p.n)), scale_idx=np.array([0, 3]))     # outside the table


@pytest.mark.parametrize("method", ["fj", "dense"])
def test_cutoff_writes_every_scale_row_numpy(method):
    _check_cutoff_table("numpy", method)


@pytest.mark.gpu
@pytest.mark.parametrize("method", ["fj", "dense"])
def test_cutoff_writes_every_scale_row_cupy(method):
    gpu_or_skip()
    _check_cutoff_table("cupy", method)


# ------------------------------------------------------------------------------------------- strategy draws
def _begun(over=None, p=None, seed=0) -> Engine:
    eng = prepared(p if p is not None else toy.knapsack(30), cfg_for(**(over or {})), seed=seed)
    eng.begin(time.monotonic(), 3600.0)
    return eng


def test_initial_population_is_round_robin():
    """The seeded population's strategies are i mod S (the repair at t0 sees them so)."""
    eng = prepared(toy.knapsack(30), cfg_for(), seed=0)
    seen = []
    real = eng._repair

    def spy(X, strat=None):
        seen.append(None if strat is None else np.array(strat))
        return real(X, strat)
    eng._repair = spy
    eng.begin(time.monotonic(), 3600.0)
    assert eng.S == 4
    np.testing.assert_array_equal(seen[0], np.arange(eng.P) % 4)
    assert np.all(eng.pop.strat < 4)


def test_inheritance_without_mutation_and_min_share():
    eng = _begun({"strategy.mutation_rate": 0.0, "strategy.min_share": 0.0})
    parent = np.array([0, 0, 1, 3, 3, 3, 2] * 7, np.uint8)
    np.testing.assert_array_equal(eng._child_strategies(parent), parent)
    assert eng.c_strat == {"strategy_redrawn": 0, "strategy_reassigned": 0}


def test_redraw_rate_and_uniformity():
    eng = _begun({"strategy.mutation_rate": 0.3, "strategy.min_share": 0.0})
    parent = np.zeros(20000, np.uint8)
    cs = eng._child_strategies(parent)
    r = eng.c_strat["strategy_redrawn"]
    assert abs(r / parent.size - 0.3) < 0.02                          # binomial sd ~0.003
    changed = int(np.sum(cs != 0))
    assert abs(changed / r - 0.75) < 0.03                              # a redraw is uniform over the 4 strategies
    cnt = np.bincount(cs[cs != 0], minlength=4)[1:]
    assert np.all(np.abs(cnt / cnt.sum() - 1 / 3) < 0.03)
    one = _begun({"strategy.mutation_rate": 1.0, "strategy.min_share": 0.0})
    cs1 = one._child_strategies(np.full(8000, 2, np.uint8))
    assert np.all(np.abs(np.bincount(cs1, minlength=4) / 8000 - 0.25) < 0.03)


def test_min_share_brings_every_strategy_back():
    eng = _begun({"strategy.mutation_rate": 0.0, "strategy.min_share": 0.1})
    O = 48
    q = math.ceil(0.1 * O)                                             # 5
    cs = eng._child_strategies(np.zeros(O, np.uint8))                  # the population lost strategies 1..3
    cnt = np.bincount(cs, minlength=4)
    assert np.all(cnt[1:] == q) and cnt[0] == O - 3 * q
    assert eng.c_strat["strategy_reassigned"] == 3 * q
    # donors never drop below the floor; the floor is capped at O // S
    mixed = np.array([0] * 30 + [1] * 6 + [2] * 12, np.uint8)
    cnt = np.bincount(eng._child_strategies(mixed), minlength=4)
    assert np.all(cnt >= q) and cnt.sum() == O
    tiny = _begun({"strategy.mutation_rate": 0.0, "strategy.min_share": 0.25})
    cnt = np.bincount(tiny._child_strategies(np.zeros(6, np.uint8)), minlength=4)
    assert np.all(cnt[1:] == 1) and cnt[0] == 3                        # ceil(1.5) = 2 capped at 6 // 4 = 1


def test_children_inherit_parent_a_in_a_run():
    """mutation_rate 0, min_share 0: every bred child's strategy is its first parent's (genealogy pa -> strat)."""
    eng = prepared(toy.set_cover(40, 25), cfg_for(**{"strategy.mutation_rate": 0.0, "strategy.min_share": 0.0}),
                   seed=4)
    run_checked(eng, 8)
    g = eng.genealogy()
    of = dict(zip(g["id"].tolist(), g["strat"].tolist()))
    bred = np.flatnonzero(g["pa"] >= 0)
    assert bred.size > 50
    assert all(g["strat"][i] == of[int(g["pa"][i])] for i in bred)


def test_min_share_holds_every_generation(tmp_path):
    eng = prepared(toy.knapsack(30), cfg_for(**{"strategy.min_share": 0.1, "log_every": 1}), seed=5,
                   trace=Trace(tmp_path, "ea"))
    run_checked(eng, 10)
    eng.finish()
    eng.trace.close()
    q = min(math.ceil(0.1 * eng.O), eng.O // 4)
    gens = [e for e in read_events(tmp_path / "ea.events.jsonl") if e["kind"] == "gen" and e["gen"] >= 1]
    assert len(gens) >= 10
    for e in gens:
        assert sum(e["strat_children"]) == eng.O and min(e["strat_children"]) >= q, e["strat_children"]


# ------------------------------------------------------------------------------------------- one repair call
def test_one_local_search_call_per_repair():
    """Instrumented: every `_repair` makes exactly ONE local-search call, on the rows as given (no grouping, no
    gather), with each row's row-scale index — children, the initial population, immigrants and the LP batch."""
    p = scaled_cover()
    eng = prepared(p, cfg_for(**{"stagnation.gens": 2, "seeding.lp_source": "solver"}), seed=6, mailbox=with_lp(p))
    assert eng.ls.row_scales == ["l2", "rhs"]
    log: list = []
    orig = eng.ls.run

    def spy(X, should_stop=None, scale_idx=None):
        log.append(("ls", None if scale_idx is None else np.array(scale_idx), np.array(X, copy=True)))
        return orig(X, should_stop=should_stop, scale_idx=scale_idx)
    eng.ls.run = spy
    orig_repair = eng._repair

    def repair_spy(X, strat=None):
        log.append(("repair", None if strat is None else np.array(strat), np.array(X, copy=True)))
        return orig_repair(X, strat)
    eng._repair = repair_spy
    eng.begin(time.monotonic(), 3600.0)
    for _ in range(6):
        eng.step()
    assert eng.c["lp_immigrants"] >= 1
    reps = [e for e in log if e[0] == "repair"]
    assert len(reps) >= 8 and len(log) == 2 * len(reps) and eng.c["ls_calls"] == len(reps)
    mixed = 0
    for (k1, strat, Xin), (k2, sidx, Xls) in zip(log[0::2], log[1::2]):
        assert k1 == "repair" and k2 == "ls" and strat is not None
        np.testing.assert_array_equal(sidx, eng.strat_scale[strat])
        np.testing.assert_array_equal(Xls, Xin)                       # the rows as given, in order
        mixed += int(np.unique(sidx).size == 2)
    assert mixed >= 4
    assert int(eng.sc["repaired"].sum()) == sum(Xin.shape[0] for _, _, Xin in reps)


def test_engine_repair_equals_one_scale_searches():
    """The engine's `_repair` on rows of mixed strategies equals, row by row, a stand-alone one-scale LocalSearch of
    that row's scale from the same random stream (seed, call counter, row index) — the strategy's row scale is what
    the repair uses — and the two scales differ on this toy."""
    p = scaled_cover()
    eng = prepared(p, cfg_for(), seed=7)
    eng.begin(time.monotonic(), 3600.0)
    X0 = np.random.default_rng(3).integers(0, 2, size=(32, p.n)).astype(np.float64)
    strat = np.random.default_rng(4).integers(0, eng.S, size=32).astype(np.uint8)
    eng.ls.impl.seed, eng.ls.impl._calls = 99, 100
    X = X0.copy()
    eng._repair(X, strat)
    outs = {}
    for sc in ("l2", "rhs"):
        ref = LocalSearch(eng.bk, eng.dp, eng.cfg.repair, eng.cfg.ranks.imp_tol, eng.budgets["local_search"],
                          seed=99, row_scales=[sc])
        ref.impl._calls = 100
        ref.set_cutoff(eng.cutoff)
        Xr = X0.copy()
        ref.run(Xr)
        outs[sc] = Xr
    is_rhs = np.array([eng.strat_list[t][0] == "rhs" for t in strat])
    assert 4 < is_rhs.sum() < 28
    np.testing.assert_array_equal(X, np.where(is_rhs[:, None], outs["rhs"], outs["l2"]))
    assert not np.array_equal(outs["l2"], outs["rhs"])


# ------------------------------------------------------------------------------------------- LP guidance
def test_lp_guidance_off_means_no_toward_lp(monkeypatch):
    """Mutation weights toward_lp only: ONE mutate call per generation, `lp_rows` = the children whose strategy has LP
    guidance; those get toward_lp, the others never (no ORIGIN_M_TOWARD_LP bit)."""
    p = toy.mixed()
    calls: list = []
    real = E.mutate

    def spy(bk, dp, C, cfg, rng_dev, rng_host, lp_x=None, lp_rows=None, **kw):
        o = real(bk, dp, C, cfg, rng_dev, rng_host, lp_x=lp_x, lp_rows=lp_rows, **kw)
        calls.append((int(C.shape[0]), lp_x is not None, None if lp_rows is None else np.array(lp_rows), o.copy()))
        return o
    monkeypatch.setattr(E, "mutate", spy)
    eng = prepared(p, cfg_for(**{"mutation.weights": {"toward_lp": 1.0}, "seeding.lp_source": "solver"}), seed=8,
                   mailbox=with_lp(p))
    css: list = []
    real_cs = eng._child_strategies

    def cs_spy(parent):
        cs = real_cs(parent)
        css.append(cs.copy())
        return cs
    eng._child_strategies = cs_spy
    eng.begin(time.monotonic(), 3600.0)
    assert eng.lp_x is not None                                        # the LP point arrived at t0
    for _ in range(5):
        eng.step()
    assert len(calls) == len(css) == 5                                 # one call per generation
    seen_lp = seen_off = 0
    for (k, has_lp, lp_rows, o), cs in zip(calls, css):
        assert has_lp and k == cs.size == eng.O
        np.testing.assert_array_equal(lp_rows, eng.strat_lp[cs])
        assert not np.any(o[~lp_rows] & ORIGIN_M_TOWARD_LP)
        seen_lp += int(np.sum((o[lp_rows] & ORIGIN_M_TOWARD_LP) != 0))
        seen_off += int((~lp_rows).sum())
    assert seen_lp > 0 and seen_off > 0


def test_mutate_lp_rows_mask():
    """`mutate(lp_rows=...)`: barred children never get toward_lp — also not when violation_guided re-draws a child
    without a violated row; all-True or None is the call without the argument, draw for draw."""
    from itrea.backend import get_backend
    from itrea.config import MutationCfg
    p = toy.set_cover(40, 25)
    bk = get_backend("numpy")
    dp = to_device(p, bk)
    lp = REF._lp_point(p)
    feasible = np.ones((200, p.n))                                     # every child feasible: violation_guided re-draws
    assert p.check(feasible[0]).feasible
    rs = np.random.default_rng(0)
    mixed = np.vstack([feasible[:100], rs.integers(0, 2, size=(100, p.n)).astype(float)])
    barred = np.arange(200) % 3 == 0
    for w in ({"toward_lp": 1.0}, {"toward_lp": 1.0, "violation_guided": 1.0},
              {"violation_guided": 1.0, "toward_lp": 3.0, "flip": 0.5}):
        cfg = MutationCfg(weights=w)
        for X0 in (feasible, mixed):
            C = X0.copy()
            o = M.mutate(bk, dp, C, cfg, None, np.random.default_rng(1), lp_x=lp, lp_rows=~barred)
            assert not np.any(o[barred] & ORIGIN_M_TOWARD_LP)
            assert np.any(o[~barred] & ORIGIN_M_TOWARD_LP)
            assert np.all(o[barred] != 0)                              # barred rows still mutate (reset / flip)
            ref = []
            for lr in (None, np.ones(200, bool), "absent"):
                C = X0.copy()
                kw = {} if isinstance(lr, str) else {"lp_rows": lr}
                ref.append((M.mutate(bk, dp, C, cfg, None, np.random.default_rng(1), lp_x=lp, **kw), C))
            for oo, CC in ref[:2]:
                np.testing.assert_array_equal(oo, ref[2][0])
                np.testing.assert_array_equal(CC, ref[2][1])
    with pytest.raises(ValueError):
        M.mutate(bk, dp, feasible.copy(), MutationCfg(weights={"toward_lp": 1.0}), None, np.random.default_rng(1),
                 lp_x=lp, lp_rows=np.ones(3, bool))


def test_lp_roundings_only_for_lp_guided_strategies():
    p = toy.mixed()
    eng = prepared(p, cfg_for(**{"seeding.lp_source": "solver"}), seed=9, mailbox=with_lp(p))
    eng.begin(time.monotonic(), 3600.0)
    assert eng.c["lp_immigrants"] >= 1                                 # the batch made when the LP point arrived
    lp_guided = np.flatnonzero(eng.strat_lp)
    for _ in range(3):                                                 # immigrant batches
        Xn, on, st = eng._immigrants(200, lp_share=0.5)
        is_lp = (on & ORIGIN_SEED_LP) != 0
        assert is_lp.any() and np.all(np.isin(st[is_lp], lp_guided))
        n_on = int(np.isin(st, lp_guided).sum())
        assert int(is_lp.sum()) == round(0.5 * n_on)                   # lp_share of the LP-guided immigrants
        assert set(np.unique(st)) == set(range(eng.S))                 # strategies drawn uniformly
    fr = np.asarray(eng.pop.X[:3])                                     # frontier rows first, with their own draws
    Xn, on, st = eng._immigrants(10, frontier_rows=fr, lp_share=1.0)
    assert Xn.shape[0] == st.size == 10 and not np.any(on[:3] & ORIGIN_SEED_LP)
    assert np.all(eng.strat_lp[st[3:][(on[3:] & ORIGIN_SEED_LP) != 0]])
    # no LP-guided strategy at all: no LP roundings anywhere
    off = prepared(p, cfg_for(**{"seeding.lp_source": "solver", "strategy.lp_guidance": [False]}), seed=9,
                   mailbox=with_lp(p))
    run_checked(off, 5)
    assert off.lp_x is not None and off.c["lp_immigrants"] == 0
    Xn, on, st = off._immigrants(100, lp_share=1.0)
    assert not np.any(on & ORIGIN_SEED_LP)
    assert not np.any(off.genealogy()["origin"] & ORIGIN_SEED_LP)


def test_pending_lp_batch_carries_lp_guided_strategies():
    p = toy.mixed()
    eng = prepared(p, cfg_for(**{"seeding.lp_source": "solver"}), seed=10)
    eng.begin(time.monotonic(), 3600.0)
    eng._lp_arrived(REF._lp_point(p), source="solver")
    (b,) = eng.pending
    assert b.src == "lp" and b.strat.size == b.X.shape[0] >= 1
    assert np.all(eng.strat_lp[b.strat])
    eng.step()
    assert not eng.pending
    draws = eng._random_strategies(4000, lp_only=True)                 # uniform over the LP-guided strategies
    lp_guided = np.flatnonzero(eng.strat_lp)
    assert set(np.unique(draws)) == set(lp_guided)
    assert np.all(np.abs(np.bincount(draws, minlength=eng.S)[lp_guided] / 4000 - 1 / lp_guided.size) < 0.04)


# ------------------------------------------------------------------------------------------- injected solver rows
def test_injected_rows_get_independent_uniform_strategies():
    """Review finding 2026-10-06: the injected rows' strategies were drawn sorted while `band_select` returns the rows
    best first, so the BEST injected row got the minimum of k draws (strategy 0 about 90 % of the time with k = 8) and
    children of the solver's solution inherited it. Now every row draws independently: the best row's strategy is
    uniform."""
    p = toy.knapsack(30)
    eng = prepared(p, cfg_for(**{"exchange.inject_band": 1.0, "exchange.inject_k": 8}), seed=1)
    eng.begin(time.monotonic(), 3600.0)
    rs = np.random.default_rng(0)
    best, every = [], []
    n_msg = 2000
    for s_ in range(n_msg):
        X = np.zeros((8, p.n))
        X[:, :3] = rs.integers(0, 2, size=(8, 3))
        obj = np.array([float(p.objective(x)) for x in X]) + rs.random(8) * 1e-3     # distinct, all in the band
        eng.pending = []
        eng._receive(Message(X=X, obj=obj, seq=s_ + 1, t=0.0))
        (b,) = eng.pending
        assert b.strat.size == 8
        best.append(int(b.strat[0]))
        every.extend(b.strat.tolist())
    f_best = np.bincount(best, minlength=4) / n_msg                    # sd ~0.0097 per share
    f_all = np.bincount(every, minlength=4) / len(every)
    assert np.all(np.abs(f_best - 0.25) < 0.045), f_best
    assert np.all(np.abs(f_all - 0.25) < 0.02), f_all


def test_a_solver_incumbent_is_not_credited_to_a_strategy(tmp_path):
    """Review finding 2026-10-06: a solver solution injected into an enabled run was counted in the per-strategy
    `incumbents` / `best_events` of the strategy the injected row was given. Now such a find (ORIGIN_INJECTED and no
    EA operator bit — the incumbent's source "solver") goes to `incumbents_solver` / `best_events_solver`, its `best`
    event says `strat_credited: false`, and `strat_best` in the gen events leaves it out."""
    p = toy.knapsack(30)
    x_opt = knapsack_optimum_x(p)
    mb = with_lp(p)
    mb.lp.q = []                                                       # no LP point: only the injection
    rows = np.repeat(x_opt[None, :], 4, axis=0)
    ones = np.flatnonzero(x_opt > 0.5)
    for r in range(1, 4):
        rows[r, ones[r % ones.size]] = 0.0                             # near-copies, worse than the optimum
    mb.to_ea.q = [Message(X=rows, obj=np.array([float(p.objective(x)) for x in rows]), seq=1, t=0.0)]
    eng = prepared(p, cfg_for(**{"exchange.inject_band": 1.0, "exchange.poll_s": 1e-6, "log_every": 1}), seed=2,
                   mailbox=mb, trace=Trace(tmp_path, "ea"))
    eng.begin(time.monotonic(), 3600.0)
    for _ in range(3):
        eng.step()
    out = eng.finish()
    eng.trace.close()
    sg = out["strategy"]
    assert eng.best_obj == pytest.approx(p.toy_optimum)
    assert sum(sg["incumbents"]) == 0 and sum(sg["incumbents_solver"]) == out["n_incumbents"] == 1
    ev = read_events(tmp_path / "ea.events.jsonl")
    inc = [e for e in ev if e["kind"] == "incumbent"]
    assert [e["source"] for e in inc] == ["solver"]
    bests = [e for e in ev if e["kind"] == "best"]
    solver_bests = [e for e in bests if e["origin"] == ORIGIN_INJECTED]
    assert solver_bests and all(e["strat_credited"] is False for e in solver_bests)
    assert all(e["strat_credited"] is True for e in bests if e["origin"] != ORIGIN_INJECTED)
    assert sum(sg["best_events_solver"]) == len(solver_bests)
    assert sum(sg["best_events"]) == len(bests) - len(solver_bests)
    gens = [e for e in ev if e["kind"] == "gen"]
    assert sum(sum(e["strat_best"]) for e in gens) == sum(sg["best_events"])


# ------------------------------------------------------------------------------------------- records
def test_genealogy_events_and_summary(tmp_path):
    p = toy.knapsack(30)
    eng = prepared(p, cfg_for(**{"log_every": 1}), seed=11, trace=Trace(tmp_path, "ea"))
    st = eng.startup
    assert [s["name"] for s in st["strategies"]] == ["l2/nolp", "l2/lp", "rhs/nolp", "rhs/lp"]
    assert st["local_search_row_scales"] == ["l2", "rhs"] and "local_searches" not in st
    run_checked(eng, 12)
    out = eng.finish()
    eng.write_outputs(tmp_path)
    eng.trace.close(**{k: v for k, v in out.items() if k not in ("best_x", "frontier")})
    S = eng.S
    sg = out["strategy"]
    assert sg["names"] == [s["name"] for s in st["strategies"]] and len(sg["children"]) == S
    assert sum(sg["children"]) == out["children"] and sum(sg["final_pop"]) == out["final_pop"]
    assert sum(sg["incumbents"]) + sum(sg["incumbents_solver"]) == out["n_incumbents"] >= 1
    assert sum(sg["incumbents_solver"]) == 0                           # no partner: every find is the EA's
    assert sg["strategy_redrawn"] > 0 and all(0 <= f <= r for f, r in zip(sg["repaired_feasible"], sg["repaired"]))
    assert sum(sg["repaired_feasible"]) > 0
    assert out["ls_calls"] >= out["generations"] + 1                  # one per repair (exact: the spy test above)
    from itrea.analysis.lineage import load_genealogy
    g = load_genealogy(tmp_path / "ea.genealogy.npz")
    assert "strat" in g.columns and g["strat"].between(0, S - 1).all()
    ids_now = dict(zip(eng.pop.ids.tolist(), eng.pop.strat.tolist()))
    gs = dict(zip(g["id"].tolist(), g["strat"].tolist()))
    assert all(gs[i] == s for i, s in ids_now.items())                 # survivors keep their strategy
    ev = read_events(tmp_path / "ea.events.jsonl")
    gens = [e for e in ev if e["kind"] == "gen"]
    for e in gens:
        assert len(e["strat_pop"]) == S and sum(e["strat_pop"]) == e["pop"]
        assert len(e["strat_best"]) == S and len(e["strat_children"]) == S
    bests = [e for e in ev if e["kind"] == "best"]
    assert bests and all(0 <= e["strat"] < S and e["strat_credited"] for e in bests)
    assert sum(sum(e["strat_best"]) for e in gens) == len(bests) == sum(sg["best_events"])


# ------------------------------------------------------------------------------------------- end to end
def _end_to_end(backend: str, name: str) -> None:
    p = {"knapsack": lambda: toy.knapsack(30), "set_cover": lambda: toy.set_cover(40, 25), "mixed": toy.mixed}[name]()
    eng = prepared(p, cfg_for(backend, **{"stagnation.gens": 3, "frontier.update_every": 2,
                                          "frontier.reinject": 0.5, "seeding.lp_source": "solver"}),
                   seed=12, mailbox=with_lp(p), warmup=True)
    assert eng.startup["warmup"]["many"]["generations"] >= 4          # the warm-up runs the strategy paths too
    if name == "knapsack":
        run_checked(eng, 100000, t_limit=20.0, stop=lambda e: e.best_obj == p.toy_optimum)
        assert eng.best_obj == pytest.approx(p.toy_optimum)
    else:
        run_checked(eng, 30)
        if name == "mixed":
            assert eng.best_obj == pytest.approx(4.0)
    out = eng.finish()
    assert out["feasible_found"] and p.check(out["best_x"]).feasible
    assert out["stagnation_events"] >= 1 or name == "knapsack"
    assert len(out["strategy"]["children"]) == 4 and min(out["strategy"]["children"]) > 0
    assert out["ls_calls"] >= out["generations"]                       # one call per repair, not one per scale
    check_invariants(eng)
    assert eng.ls.row_scales == ["l2", "rhs"]
    if backend == "cupy":
        assert out["peak_gpu_pool_mb"] <= 1024
        assert eng.startup["local_search"] == "fj_gpu" and tuple(eng.ls.impl.d_rs.shape) == (2, eng.ls.impl.m_aug)


@pytest.mark.parametrize("name", ["knapsack", "set_cover", "mixed"])
def test_end_to_end_numpy(name):
    _end_to_end("numpy", name)


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["knapsack", "set_cover", "mixed"])
def test_end_to_end_cupy(name):
    gpu_or_skip()
    _end_to_end("cupy", name)


@pytest.mark.parametrize("over", [{"strategy.row_scales": ["rhs"], "strategy.lp_guidance": [True]},
                                  {"strategy.row_scales": ["count", "rhs"]},
                                  {"ranks.viol_norm": "rhs"},
                                  {"repair.method": "dense"}, {"repair.method": "none"}])
def test_other_strategy_spaces_and_repairs(over):
    """One strategy only, a row scale list without ranks.viol_norm, V on rhs, the dense baseline and no repair."""
    p = toy.set_cover(40, 25)
    eng = prepared(p, cfg_for(**over), seed=13)
    none = eng.cfg.repair.method == "none"
    assert eng.ls.row_scales == ([None] if none else list(eng.cfg.strategy.row_scales))
    run_checked(eng, 6)
    out = eng.finish()
    assert sum(out["strategy"]["children"]) == out["children"]
    if none:
        assert np.all(eng.strat_scale == 0) and sum(out["strategy"]["repaired_feasible"]) == 0   # not tracked
        assert sum(out["strategy"]["repaired"]) > 0


# ------------------------------------------------------------------------------------------- memory plan
def test_memory_plan_counts_the_extra_row_scales():
    n, m = 40000, 100
    A = sps.random(m, n, density=0.001, random_state=0, format="csr")
    A.data[:] = 1.0
    p = from_arrays(A, np.ones(m), np.full(m, np.inf), np.zeros(n), np.ones(n), np.ones(n), np.ones(n, bool))
    gdt = gene_dtype_for(p.lb, p.ub, p.isint)
    fixed = {"population.size_rule": "fixed", "population.size": 64, "population.offspring_per_gen": 64,
             "gpu_mem_gb": 1.0}
    off = plan_memory(p, EAConfig().with_overrides(fixed), gdt)
    on_cfg = EAConfig().with_overrides(dict(fixed, **ON))
    on = plan_memory(p, on_cfg, gdt)
    extra = strategy_ls_bytes(p, on_cfg)
    assert "local_search_strategies" not in off.terms and on.terms["local_search_strategies"] == extra
    assert extra == 1 * 8 * (m + 1)                                    # one more row of the scale table
    assert on.total_bytes - off.total_bytes == extra
    assert extra < 0.01 * off.terms["local_search"]
    three = EAConfig().with_overrides(dict(fixed, **ON, **{"strategy.row_scales": ["rhs", "count", "l2"]}))
    assert strategy_ls_bytes(p, three) == 2 * 8 * (m + 1)
    assert strategy_ls_bytes(p, three.with_overrides({"repair.method": "none"})) == 0
    # the memory rule: P with the strategy is at most P without it
    mem = EAConfig().with_overrides({"gpu_mem_gb": 0.2})
    assert plan_memory(p, mem.with_overrides(ON), gdt).P <= plan_memory(p, mem, gdt).P


@pytest.mark.gpu
def test_memory_plan_term_covers_the_device_arrays():
    """On the GPU the strategy's local search owns exactly the plan's term more than the disabled one (CuPy
    allocations: the extra rows of the scale table), and there is one search."""
    gpu_or_skip()
    p = scaled_cover(n=60, m=40)
    on = prepared(p, cfg_for("cupy"), seed=14)
    off = prepared(p, cfg_for("cupy", **{"strategy.enabled": False}), seed=14)
    extra = on.ls.impl.static_bytes - off.ls.impl.static_bytes
    assert extra == 8 * on.ls.impl.m_aug <= on.plan.terms["local_search_strategies"]
    assert on.startup["local_search_static_bytes"] == on.ls.impl.static_bytes


# ------------------------------------------------------------------------------------------- the config file
@pytest.mark.solver
def test_adaptive_config_runs_with_its_own_lp_point(tmp_path):
    """configs/itrea2-adaptive.yaml on a toy MPS: the EA alone gets its LP point from the HiGHS worker after t0."""
    from pathlib import Path
    from itrea.problem import load
    cfg = EAConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs" / "itrea2-adaptive.yaml")
    cfg = cfg.with_overrides(dict(SMALL, backend="numpy", **{"seeding.lp_time_limit": 10.0}))
    p = load(toy.write_mps(toy.mixed(), tmp_path / "mixed.mps"))
    eng = prepared(p, cfg, seed=15)
    eng.begin(time.monotonic(), 3600.0)
    end = time.monotonic() + 20.0
    while (eng.lp_x is None or eng.gen < 6) and time.monotonic() < end:
        eng.step()
    out = eng.finish()
    assert eng.lp_x is not None and eng.lp_info["source"] == "highs_async"
    assert out["lp_immigrants"] >= 1 and out["feasible_found"]
    assert out["strategy"]["names"] == ["l2/nolp", "l2/lp", "rhs/nolp", "rhs/lp"]
