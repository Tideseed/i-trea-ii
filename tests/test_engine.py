"""The EA engine (`itrea.ea.engine`): the generation loop on toy models, the survival invariants every generation, the
exchange directions (a spy mailbox proves the one-way arm never reads the solver's channels), injections and echoes,
solver rejections, the LP point on the clock, the memory plan, the warm-up's synthetic problem, the output files.

Toy models only (`itrea.benchmarks.toy`). GPU tests (marker `gpu`) need CuPy and use a 1 GB pool; the solver test
(marker `solver`) runs HiGHS's LP relaxation in the worker process on a toy MPS.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from itrea.benchmarks import toy
from itrea.config import EAConfig
from itrea.ea import engine as E
from itrea.ea.engine import DoesNotFit, Engine, plan_memory, synthetic_problem
from itrea.ea.population import ORIGIN_INJECTED, R0, R1, gene_dtype_for
from itrea.exchange.mailbox import Message
from itrea.solvers.common import ORIGIN_EA_ECHO, ORIGIN_SOLVER
from itrea.trace import Trace, load_incumbents, read_events

SMALL = {"population.size": 48, "population.offspring_per_gen": 48, "gpu_mem_gb": 1.0}


def cfg_for(backend: str = "numpy", **over) -> EAConfig:
    d = dict(SMALL, backend=backend)
    d.update(over)
    return EAConfig().with_overrides(d)


def gpu_or_skip():
    pytest.importorskip("cupy")
    from itrea.backend import get_backend
    try:
        get_backend("cupy", gpu_mem_gb=1.0)
    except Exception as e:  # pragma: no cover - no device
        pytest.skip(f"no CuPy device: {e}")


def knapsack_optimum_x(p) -> np.ndarray:
    """An optimal vector of toy.knapsack by DP with backtracking."""
    w = np.asarray(p.A.toarray()[0], int)
    v = np.asarray(p.c, float)
    W = int(p.ru[0])
    n = w.size
    best = np.zeros((n + 1, W + 1))
    for j in range(n):
        best[j + 1] = best[j]
        best[j + 1, w[j]:] = np.maximum(best[j, w[j]:], best[j, :W + 1 - w[j]] + v[j])
    x, cap = np.zeros(n), W
    for j in range(n - 1, -1, -1):
        if best[j + 1, cap] != best[j, cap]:
            x[j] = 1
            cap -= w[j]
    assert p.objective(x) == pytest.approx(p.toy_optimum)
    return x


def prepared(p, cfg, seed=0, mailbox=None, trace=None) -> Engine:
    eng = Engine(p, cfg, seed, mailbox=mailbox, trace=trace)
    eng.prepare()
    return eng


# ----------------------------------------------------------------------------------------------- invariants
def check_invariants(eng: Engine) -> None:
    """The survival invariants of the design note, on the population after a generation."""
    pop, cfg = eng.pop, eng.cfg
    rk = cfg.ranks
    assert pop.size <= eng.P
    assert len(set(pop.ids.tolist())) == pop.size                       # no individual twice
    hk, hv, hf = pop.host_scores(eng.bk)
    np.testing.assert_array_equal(hf, eng.h_feas)                        # host mirrors in sync with the device
    np.testing.assert_allclose(hk, eng.h_key, rtol=0, atol=1e-9)
    # Rank 1 = feasible, Rank 0 = infeasible
    assert np.all(hf[pop.rank == R1])
    assert not np.any(hf[pop.rank == R0])
    surv = eng.last_surv
    inc = surv.inc
    cap = math.floor(rk.x * eng.P + 1e-12)
    assert int(np.sum(pop.rank == R0)) <= cap
    if math.isfinite(inc):
        thr = inc - rk.imp_tol * max(1.0, abs(inc))
        # the manipulation check: super-optimal infeasible survivors never exceed the x quota (a TOTAL cap)
        assert int(np.sum(~hf & (hk < thr))) <= cap
        assert np.all(hk[pop.rank == R0] < thr)


def run_checked(eng: Engine, gens: int, t_limit: float = 60.0, stop=None, elitist: bool = True) -> list[dict]:
    """begin + `gens` steps, the invariants after each, Rank 1 survival checked between generations (not with a
    partner that rejects posts: a rejected incumbent is demoted, so the best feasible may be lost on purpose)."""
    eng.begin(time.monotonic(), 3600.0)
    check_invariants(eng)
    hist = []
    end = time.monotonic() + t_limit
    for _ in range(gens):
        before = {int(i): float(k) for i, k, r in zip(eng.pop.ids, eng.h_key, eng.pop.rank) if r == R1}
        best_before = float(eng.h_key[eng.h_feas].min()) if eng.h_feas.any() else math.inf
        rej_before = eng.c["solver_rejections"]
        eng.step()
        check_invariants(eng)
        ids_now = set(int(i) for i in eng.pop.ids)
        fk = eng.h_key[eng.h_feas]
        best_now = float(fk.min()) if fk.size else math.inf
        if elitist:
            assert best_now <= best_before + 1e-9                       # the best feasible is never lost
            # a Rank 1 member of generation g is in g+1 unless displaced: at least as many better (or equal-key
            # older) feasible individuals as Rank 1 had now hold Rank 1 (the ratchet, ranks.three_rank_survival)
            n1_prev = len(before)
            r1_keys = eng.h_key[eng.pop.rank == R1]
            for iid, k in before.items():
                if iid not in ids_now:
                    assert int(np.sum(r1_keys <= k + 1e-9)) >= n1_prev, (iid, k, n1_prev, np.sort(r1_keys))
            if eng.c["solver_rejections"] == rej_before:               # only a verdict can demote a Rank 1 member
                assert r1_keys.size >= n1_prev, (r1_keys.size, n1_prev)  # Rank 1 never shrinks
        hist.append(dict(gen=eng.gen, best=best_now))
        if stop is not None and stop(eng):
            break
        if time.monotonic() > end:
            break
    return hist


# ------------------------------------------------------------------------------------- end to end, NumPy
def test_knapsack_optimum_numpy(tmp_path):
    p = toy.knapsack(30)
    tr = Trace(tmp_path, "ea")
    eng = prepared(p, cfg_for(), seed=1, trace=tr)
    assert eng.startup["backend"] == "numpy" and eng.startup["gene_dtype"] == "int8"
    t0 = time.monotonic()
    run_checked(eng, 100000, t_limit=20.0, stop=lambda e: e.best_obj == p.toy_optimum)
    assert eng.best_obj == pytest.approx(p.toy_optimum), f"not found within 20 s: {eng.best_obj} vs {p.toy_optimum}"
    out = eng.finish()
    assert out["feasible_found"] and out["best_obj"] == pytest.approx(p.toy_optimum)
    assert p.check(out["best_x"]).feasible
    assert time.monotonic() - t0 < 25.0
    # files
    paths = eng.write_outputs(tmp_path)
    tr.close(**{k: v for k, v in out.items() if k not in ("best_x", "frontier")})
    names, recs = load_incumbents(paths["incumbents"])
    assert names == p.names and len(recs) == out["n_incumbents"] >= 1
    objs = [r[1] for r in recs]
    assert objs == sorted(objs) and objs[-1] == pytest.approx(p.toy_optimum)       # maximisation: improving
    for _, o, src, x in recs:
        assert src == "self" and p.check(x).feasible and p.objective(x) == pytest.approx(o)
    from itrea.analysis.lineage import load_genealogy, run_tables
    g = load_genealogy(paths["genealogy"])
    assert len(g) == out["genealogy_rows"] > 0
    assert g["id"].is_unique
    assert g["is_new_best"].sum() == out["n_incumbents"]
    bred = g[g["pa"] >= 0]
    assert len(bred) > 0 and set(bred["rank_pa"].unique()) <= {0, 1, 2}
    mo = bred[bred["pa"] == bred["pb"]]                                # mutation-only: same parent twice
    assert (mo["rank_pa"] == mo["rank_pb"]).all()
    rt = run_tables(tmp_path)                                           # the lineage analysis reads it
    assert rt["rows"] == len(g) and not rt["derived_flags"]
    fz = np.load(paths["frontier"])
    assert fz["X"].shape[0] == fz["keys"].shape[0] >= 1
    for x in fz["X"]:
        assert p.check(p.round_integers(x)).feasible
    ev = read_events(tmp_path / "ea.events.jsonl")
    kinds = {e["kind"] for e in ev}
    assert {"gen", "best", "incumbent", "frontier"} <= kinds
    gen = [e for e in ev if e["kind"] == "gen"]
    for k in ("superopt_survivors", "shortfall", "n0", "n1", "n2", "inc", "seconds", "children_per_s",
              "rank0_candidates", "rank0_demoted", "rank0_over_quota", "feasibility_phase"):
        assert k in gen[-1]
    assert all(e["superopt_survivors"] <= math.floor(0.1 * eng.P) for e in gen)


@pytest.mark.parametrize("name", ["set_cover", "mixed"])
def test_invariants_numpy(name):
    p = {"set_cover": lambda: toy.set_cover(40, 25), "mixed": toy.mixed}[name]()
    eng = prepared(p, cfg_for(**{"stagnation.gens": 3, "frontier.update_every": 2, "frontier.reinject": 0.5}), seed=3)
    hist = run_checked(eng, 40)
    assert eng.best_x is not None and p.check(eng.best_x).feasible
    assert eng.c["stagnation_events"] >= 1                              # immigrants exercised under the invariants
    if name == "mixed":
        assert eng.best_obj == pytest.approx(4.0)                       # the optimum (hand-checked)
        assert eng.gene_dtype == np.float64
    assert hist[-1]["best"] < math.inf


def test_singleton_slacks_are_decoded():
    """An equality row with slack columns (markshare-like): the decoder makes every individual feasible, and the
    search drives the slack cost down; the planted solution has cost 0."""
    import scipy.sparse as sps
    from itrea.problem import from_arrays
    rs = np.random.default_rng(0)
    n = 14
    a = rs.integers(5, 40, size=n).astype(float)
    planted = (rs.random(n) < 0.5).astype(float)
    b = float(a @ planted)
    A = sps.csr_matrix(np.r_[a, 1.0, -1.0][None, :])
    p = from_arrays(A, [b], [b], np.zeros(n + 2), np.r_[np.ones(n), np.inf, np.inf], np.r_[np.zeros(n), 1.0, 1.0],
                    np.r_[np.ones(n, bool), False, False], names=[f"x{j}" for j in range(n)] + ["sp", "sm"])
    eng = prepared(p, cfg_for(), seed=11)
    assert eng.decoder is not None and eng.startup["decoded_columns"] == 2
    run_checked(eng, 30)
    assert eng.h_feas.all()                                             # slacks absorb every residual
    assert eng.best_obj is not None and eng.best_obj < 0.5 * float(np.abs(a).sum())


def test_rank0_cap_holds_with_x_zero():
    """x = 0 (the H6 ablation): no infeasible individual better than the incumbent ever survives."""
    p = toy.knapsack(30)
    eng = prepared(p, cfg_for(**{"ranks.x": 0.0}), seed=2)
    run_checked(eng, 30)
    assert int(np.sum(eng.pop.rank == R0)) == 0


def test_backend_auto_and_bad_dtype_refused():
    p = toy.knapsack(10)
    with pytest.raises(ValueError, match="auto"):
        Engine(p, EAConfig().with_overrides({"backend": "auto"}), 0).prepare()
    with pytest.raises(ValueError, match="continuous"):
        Engine(toy.mixed(), cfg_for(**{"population.gene_dtype": "int32"}), 0).prepare()


# ----------------------------------------------------------------------------------------------- exchange
class SpyMailbox:
    """Records every attribute access; reading a solver channel raises (the one-way arm must not)."""
    FORBIDDEN = ("to_ea", "info", "lp", "verdicts")

    def __init__(self):
        object.__setattr__(self, "accessed", [])
        object.__setattr__(self, "posts", [])

        class _To:
            def post(_s, X, obj, origin=None):
                self.posts.append((np.asarray(X), np.asarray(obj), origin))
                return len(self.posts)
        object.__setattr__(self, "_to_solver", _To())

    def __getattr__(self, name):
        self.accessed.append(name)
        if name in SpyMailbox.FORBIDDEN:
            raise AssertionError(f"the one-way arm read the solver channel {name!r}")
        if name == "to_solver":
            return self._to_solver
        if name == "stopped":
            return lambda: False
        raise AttributeError(name)


def test_one_way_arm_never_reads_solver_channels():
    p = toy.knapsack(30)
    mb = SpyMailbox()
    cfg = cfg_for(**{"exchange.from_solver": False, "exchange.use_dual_bound": False,
                     "exchange.use_root_lp": False})
    eng = prepared(p, cfg, seed=4, mailbox=mb)
    run_checked(eng, 20)
    out = eng.finish()
    assert set(mb.accessed) <= {"to_solver", "stopped"}
    assert not set(mb.accessed) & set(SpyMailbox.FORBIDDEN)
    assert mb.posts and out["posts"] == len(mb.posts)                  # it does post (EA -> solver)
    assert out["inc_external"] == math.inf and out["dual_key"] == -math.inf
    for X, obj, origin in mb.posts:
        assert p.check(X[0]).feasible and p.objective(X[0]) == pytest.approx(obj[0])


class FakeMailbox:
    """A solver partner in memory: queued to_ea / info / lp messages, recorded posts, scripted verdicts."""

    def __init__(self, to_ea=(), info=(), lp=(), verdict=None):
        self._to_ea, self._info, self._lp = list(to_ea), list(info), list(lp)
        self.posts: list = []
        self._verdicts: list[dict] = []
        self.verdict = verdict                   # callable(seq, X, obj) -> accepted, or None (no verdicts)
        outer = self

        class _Q:
            def __init__(self, q):
                self.q = q

            def poll(self):
                return self.q.pop(0) if self.q else None

        class _To:
            def post(_s, X, obj, origin=None):
                outer.posts.append((np.asarray(X), np.asarray(obj), origin))
                seq = len(outer.posts)
                if outer.verdict is not None:
                    for r in range(len(obj)):
                        outer._verdicts.append(dict(seq=seq, t=0.0, obj=float(obj[r]),
                                                    accepted=bool(outer.verdict(seq, X[r], obj[r])),
                                                    note=f'{{"row": {r}}}'))
                return seq

        self.to_ea, self.info, self.lp, self.to_solver = _Q(self._to_ea), _Q(self._info), _Q(self._lp), _To()

    def stopped(self):
        return False

    def verdicts(self):
        return list(self._verdicts)


def test_injection_echo_and_solver_incumbent():
    """The solver's optimum enters the population (counted, provenance ORIGIN_INJECTED, source 'solver'), an echo row
    is skipped, the solver's incumbent and bound are read as keys, and nothing worse than it is posted."""
    p = toy.knapsack(30)
    x_opt = knapsack_optimum_x(p)
    worse = x_opt.copy()
    worse[np.flatnonzero(x_opt)[0]] = 0.0
    msg = Message(X=np.vstack([x_opt, worse, x_opt]), obj=np.array([p.objective(x_opt), p.objective(worse),
                                                                    p.objective(x_opt)]), seq=1, t=0.0,
                  origin=np.array([ORIGIN_SOLVER, ORIGIN_SOLVER, ORIGIN_EA_ECHO]))
    info = dict(incumbent=p.toy_optimum, dual_bound=p.toy_optimum + 0.5, status="solving")
    mb = FakeMailbox(to_ea=[msg], info=[info])
    eng = prepared(p, cfg_for(**{"exchange.inject_band": 0.5}), seed=5, mailbox=mb)
    run_checked(eng, 5)
    out = eng.finish()
    assert out["inject_msgs"] == 1 and out["inject_rows"] == 3 and out["echoes"] == 1
    assert out["injected"] == 2 and out["injected_infeasible"] == 0
    assert eng.inc_external == pytest.approx(-p.toy_optimum)            # key = sense * objective (maximisation)
    assert eng.dual_key == pytest.approx(-(p.toy_optimum + 0.5))
    assert eng.best_obj == pytest.approx(p.toy_optimum)
    assert eng.records[0][2] == "solver" or any(r[2] == "solver" for r in eng.records)
    i = int(np.argmin(np.where(eng.h_feas, eng.h_key, np.inf)))
    assert eng.pop.origin[i] & ORIGIN_INJECTED
    for X, obj, _ in mb.posts:                                          # nothing posted that is not better
        assert p.key_of_objective(obj[0]) < -p.toy_optimum


def test_solver_rejection_demotes():
    p = toy.knapsack(30)
    mb = FakeMailbox(verdict=lambda seq, x, obj: False)
    eng = prepared(p, cfg_for(**{"exchange.poll_s": 1e-6}), seed=6, mailbox=mb)
    run_checked(eng, 6, elitist=False)
    eng._exchange_in()                           # the verdict on the last generation's post (read at the next tick)
    out = eng.finish()
    assert out["posts"] >= 1 and out["solver_rejections"] == out["posts"]
    # every incumbent was rejected: the incumbent fell back to none and the final best_x is no rejected vector
    assert out["incumbents_reverted"] >= 1 and out["best_x"] is None and not out["feasible_found"]
    rejected = {iid for seq, ids in eng.posted.items() for iid in ids}
    assert rejected <= eng.rejected_ids and not rejected & eng.r1_ids
    for iid in rejected:
        pos = np.flatnonzero(eng.pop.ids == iid)
        if pos.size:
            assert not eng.h_feas[pos[0]] and eng.pop.rank[pos[0]] != R1


def test_accepted_posts_marked_in_genealogy(tmp_path):
    p = toy.knapsack(30)
    mb = FakeMailbox(verdict=lambda seq, x, obj: True)
    eng = prepared(p, cfg_for(**{"exchange.poll_s": 1e-6}), seed=7, mailbox=mb)
    run_checked(eng, 8)
    out = eng.finish()
    assert out["solver_accepted"] >= 1
    g = eng.genealogy()
    assert g["is_accepted_post"].sum() >= 1
    assert np.all(g["is_new_best"][g["is_accepted_post"]])              # posts are new incumbents


def test_lp_point_from_solver_channel():
    p = toy.mixed()
    lp = np.array([0.0, 4.5, 0.0, 5.5])                                 # an LP-like point (x1 fractional)
    mb = FakeMailbox(lp=[Message(X=lp[None, :], obj=np.array([p.objective(lp)]), seq=1, t=0.0)])
    eng = prepared(p, cfg_for(**{"seeding.lp_source": "solver"}), seed=8, mailbox=mb)
    run_checked(eng, 4)
    assert eng.lp_x is not None and np.allclose(eng.lp_x, lp)
    assert eng.c["lp_immigrants"] >= 1 and eng.lp_info["state"] == "done"


@pytest.mark.solver
def test_lp_point_from_highs_worker(tmp_path):
    p_file = toy.write_mps(toy.mixed(), tmp_path / "mixed.mps")
    from itrea.problem import load
    p = load(p_file)
    eng = prepared(p, cfg_for(**{"seeding.lp_source": "highs_async", "seeding.lp_time_limit": 10.0}), seed=9)
    eng.begin(time.monotonic(), 3600.0)
    end = time.monotonic() + 20.0
    while eng.lp_x is None and time.monotonic() < end:
        eng.step()
    eng.finish()
    assert eng.lp_x is not None, eng.lp_info
    assert eng.lp_info["source"] == "highs_async" and eng.lp_info["state"] == "done"
    assert eng.lp_worker is None or not eng.lp_worker.running


# ------------------------------------------------------------------------------------------- memory plan
def test_memory_plan_rules():
    import scipy.sparse as sps
    from itrea.problem import from_arrays
    n, m = 40000, 100
    A = sps.random(m, n, density=0.001, random_state=0, format="csr")
    A.data[:] = 1.0
    p = from_arrays(A, np.ones(m), np.full(m, np.inf), np.zeros(n), np.ones(n), np.ones(n), np.ones(n, bool))
    gdt = gene_dtype_for(p.lb, p.ub, p.isint)
    big = plan_memory(p, EAConfig().with_overrides({"gpu_mem_gb": 4.0}), gdt)
    assert big.P == 256 and big.fits                                    # fits: P = size
    # a small budget forces a smaller P (and offspring in proportion), the same on any backend
    tiny = EAConfig().with_overrides({"gpu_mem_gb": 0.2})
    a, b = plan_memory(p, tiny, gdt), plan_memory(p, tiny.with_overrides({"backend": "numpy"}), gdt)
    assert a.P == b.P < 256 and a.offspring == b.offspring == a.P
    # the switches that differ between the arms of a block do not change P (A3 / A4 / A5 / A5x0 run the same P)
    for ov in ({"exchange.enabled": False}, {"frontier.enabled": False}, {"ranks.x": 0.0},
               {"exchange.enabled": False, "frontier.enabled": False, "ranks.x": 0.0}):
        c = plan_memory(p, tiny.with_overrides(ov), gdt)
        assert (c.P, c.offspring) == (a.P, a.offspring), ov
    assert a.total_bytes <= 0.9 * a.limit_bytes
    over = plan_memory(p, tiny.with_overrides({"population.size": a.P + 1, "population.offspring_per_gen": a.P + 1,
                                               "population.size_rule": "fixed"}), gdt)
    assert over.P == a.P + 1 and not over.fits                          # a.P is the largest P that fits
    # fixed size that does not fit: refused by prepare
    with pytest.raises(DoesNotFit):
        Engine(p, tiny.with_overrides({"backend": "numpy", "population.size_rule": "fixed"}), 0).prepare()
    # nothing fits at all
    with pytest.raises(DoesNotFit):
        Engine(p, EAConfig().with_overrides({"backend": "numpy", "gpu_mem_gb": 0.01}), 0).prepare()


def test_synthetic_problem_mirrors_classes():
    for p in (toy.knapsack(30), toy.set_cover(40, 25), toy.mixed()):
        s = synthetic_problem(p)
        assert (s.bins.size > 0) == (p.bins.size > 0)
        assert (s.gints.size > 0) == (p.gints.size > 0)
        assert (s.conts.size > 0) == (p.conts.size > 0)
        assert s.m == min(p.m, s.m) and (s.m == 1) == (p.m == 1)
        assert s.sense == p.sense
        assert s.check(s.toy_point).feasible
        assert gene_dtype_for(s.lb, s.ub, s.isint).kind == gene_dtype_for(p.lb, p.ub, p.isint).kind
    assert np.any(synthetic_problem(toy.mixed()).rl == synthetic_problem(toy.mixed()).ru)   # equality row kept


def test_incumbent_log_recovers_after_a_kill(tmp_path):
    """The .part file holds every incumbent while the run goes on (a SIGKILL leaves it behind); the solvers'
    `recover_incumbents` turns it into the same .npz; `write_outputs` removes it once the .npz exists."""
    from itrea.solvers.common import recover_incumbents
    p = toy.knapsack(30)
    eng = prepared(p, cfg_for(), seed=12)
    eng.incumbent_log(tmp_path, "ea")
    run_checked(eng, 30)
    assert (tmp_path / "ea.incumbents.part").exists() and (tmp_path / "ea.names.json").exists()
    rec = recover_incumbents(tmp_path, "ea")                           # what build_runs does for a crashed run
    names, recs = load_incumbents(rec)
    assert names == p.names and len(recs) == eng.c["n_incumbents"] >= 1
    for (t, o, s, x), (t2, o2, s2, r) in zip(recs, eng.records):
        assert (t, o, s) == (t2, o2, s2) and np.array_equal(x, r.dense())
    rec.unlink()
    eng.finish()
    paths = eng.write_outputs(tmp_path)
    assert not (tmp_path / "ea.incumbents.part").exists() and not (tmp_path / "ea.names.json").exists()
    assert len(load_incumbents(paths["incumbents"])[1]) == len(recs)


def test_warmup_does_not_touch_the_instance():
    """prepare() evaluates nothing of the instance: no population exists and no instance evaluation is counted."""
    p = toy.knapsack(30)
    eng = prepared(p, cfg_for(), seed=10)
    assert not hasattr(eng, "pop") and not hasattr(eng, "c")
    w = eng.startup["warmup"]
    assert w["many"]["generations"] >= 4 and w["one"]["generations"] >= 4


# ------------------------------------------------------------------------------------------------- GPU
@pytest.mark.gpu
def test_knapsack_optimum_cupy(tmp_path):
    gpu_or_skip()
    p = toy.knapsack(30)
    eng = prepared(p, cfg_for("cupy"), seed=1, trace=Trace(tmp_path, "ea"))
    assert eng.startup["backend"] == "cupy" and eng.startup["pool_limit_bytes"] == 1 << 30
    assert eng.startup["local_search"] == "fj_gpu"
    run_checked(eng, 100000, t_limit=20.0, stop=lambda e: e.best_obj == p.toy_optimum)
    assert eng.best_obj == pytest.approx(p.toy_optimum)
    out = eng.finish()
    assert out["peak_gpu_pool_mb"] <= 1024
    paths = eng.write_outputs(tmp_path)
    _, recs = load_incumbents(paths["incumbents"])
    assert all(p.check(x).feasible for _, _, _, x in recs)


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["set_cover", "mixed"])
def test_invariants_cupy(name):
    gpu_or_skip()
    p = {"set_cover": lambda: toy.set_cover(40, 25), "mixed": toy.mixed}[name]()
    eng = prepared(p, cfg_for("cupy", **{"stagnation.gens": 3, "frontier.update_every": 2}), seed=3)
    run_checked(eng, 25)
    assert eng.best_x is not None and p.check(eng.best_x).feasible
    if name == "mixed":
        assert eng.best_obj == pytest.approx(4.0)


@pytest.mark.gpu
def test_one_way_arm_cupy():
    gpu_or_skip()
    p = toy.knapsack(30)
    mb = SpyMailbox()
    eng = prepared(p, cfg_for("cupy", **{"exchange.from_solver": False}), seed=4, mailbox=mb)
    run_checked(eng, 10)
    assert mb.posts and not set(mb.accessed) & set(SpyMailbox.FORBIDDEN)


def test_internal_constants_documented():
    """The engine's own knobs that are not in EAConfig stay visible (they appear in the start-up record)."""
    assert 0 < E.HEADROOM < 1 and E.FRONTIER_OFFER >= 1 and E.MAX_DEMOTIONS >= 1 and 0 < E.STOP_POLL_S <= 1
