"""Rank 1 as a distinct elite — cuOpt's clearing rule (vault D-016, `ea/elite.py`, `ranks.rank1_mode`)."""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pytest

import _engine_reference as REF
from itrea.backend import get_backend
from itrea.benchmarks import toy
from itrea.config import EAConfig, RankCfg
from itrea.ea.elite import clearing_pick, elite_spread, min_diff_auto
from itrea.ea.engine import Engine
from itrea.ea.population import R1
from itrea.ea.ranks import three_rank_survival
from itrea.trace import Trace, read_events

BK = get_backend("numpy")
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("n_int,k", [(0, 1), (1, 1), (49, 1), (50, 2), (79, 2), (80, 4), (199, 4), (200, 8),
                                     (999, 8), (1000, 10), (10 ** 6, 10)])
def test_min_diff_auto_is_cuopts_rule(n_int, k):
    assert min_diff_auto(n_int) == k


def _diff(a, b) -> int:
    return int(np.sum(np.rint(a) != np.rint(b)))


@pytest.mark.parametrize("dtype", [np.int8, np.float64])
@pytest.mark.parametrize("k", [1, 3, 6])
def test_clearing_pick_is_greedy_clearing(dtype, k):
    rng = np.random.default_rng(11)
    n, m = 12, 40
    X = rng.integers(0, 2, size=(m, n)).astype(dtype)
    X[5] = X[3]
    X[7, :] = X[3, :]
    X[7, 0] = 1 - X[7, 0]                                     # one integer away from row 3
    if dtype == np.float64:
        X = X + rng.uniform(-1e-7, 1e-7, size=X.shape)       # integral within tolerance
    ints = np.arange(n - 2)                                   # the last two columns are "continuous": ignored
    order = rng.permutation(m)
    ref = None
    for cap, block, budget in ((1, 32, 1 << 26), (4, 32, 1 << 26), (m, 32, 1 << 26), (m, 1, 1), (m, 3, 50),
                               (m, m, 1 << 30), (m, 9, 1)):
        st = {}
        got = clearing_pick(BK, X, order, ints, k, cap, stats=st, block=block, budget=budget)
        if cap == m:                                          # blocks and chunks never change the result
            ref = got if ref is None else ref
            assert np.array_equal(got, ref)
            assert st["cleared"] == m - got.size              # the walk ended at the last row: all others cleared
        assert got.size <= cap and got[0] == order[0]
        pos = {int(r): i for i, r in enumerate(order)}
        assert all(pos[int(a)] < pos[int(b)] for a, b in zip(got, got[1:]))          # best first, a subsequence
        for a, b in itertools.combinations(got, 2):                                    # pairwise distinct by >= k
            assert _diff(X[a, ints], X[b, ints]) >= k
        if got.size < cap:                                    # every skipped row is within k of an earlier member
            for r in order:
                if r in got:
                    continue
                earlier = [g for g in got if pos[int(g)] < pos[int(r)]]
                assert any(_diff(X[g, ints], X[r, ints]) < k for g in earlier), (r, earlier)


def test_clearing_pick_one_row_or_cap_one_needs_no_distance():
    X = np.zeros((4, 3), np.int8)
    st = {}
    assert np.array_equal(clearing_pick(BK, X, np.array([2]), np.arange(3), 1, 5, stats=st), [2]) and st["cleared"] == 0
    assert np.array_equal(clearing_pick(BK, X, np.array([3, 1, 0]), np.arange(3), 1, 1, stats=st), [3])
    assert st["cleared"] == 0


def test_min_diff_above_n_int_is_refused():
    p = toy.knapsack(30)
    with pytest.raises(ValueError, match="rank1_min_diff"):
        Engine(p, REF.config("numpy", **{"ranks.rank1_mode": "clearing", "ranks.rank1_min_diff": 31}), seed=1,
               warmup=False).prepare()


def test_clearing_pick_without_integers_is_top():
    X = np.zeros((5, 3))
    order = np.array([4, 2, 0, 1, 3])
    assert np.array_equal(clearing_pick(BK, X, order, np.empty(0, np.int64), 10, 3), [4, 2, 0])
    assert clearing_pick(BK, X, order, np.arange(3), 1, 0).size == 0
    assert clearing_pick(BK, X, np.empty(0, np.int64), np.arange(3), 1, 3).size == 0


def test_elite_spread():
    X = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int8)
    out = elite_spread(BK, X, np.array([0, 1, 2]), np.array([5.0, 1.0, 9.0]), np.arange(4))
    # best = row 1; rows 0 and 2 differ from it in 1 and 3 of 4; worst gap (9 - 1) / max(1, 1)
    assert out == dict(r1_n=3, r1_dmax=0.75, r1_dmed=0.5, r1_gap=8.0)
    out = elite_spread(BK, X, np.array([0, 1, 2]), np.array([1.04, 1.0, 9.0]), np.arange(4), band=0.05)
    assert out["r1_dmax_band"] == 0.25 and out["r1_dmax"] == 0.75   # only row 0 is within 5 % of the best
    out = elite_spread(BK, X, np.array([0, 1, 2]), np.array([3.0, 1.0, 9.0]), np.arange(4), band=0.05)
    assert out["r1_dmax_band"] == 0.0                                 # no other member inside the band
    assert elite_spread(BK, X, np.array([0]), np.array([1.0]), np.arange(4), band=0.05) == dict(
        r1_n=1, r1_dmax=None, r1_dmed=None, r1_gap=None, r1_dmax_band=None)


# ------------------------------------------------------------------------------------------- the survival step
def _cands():
    key = np.array([5.0, 1.0, 1.0, 3.0, 2.0, 9.0, 0.5])
    viol = np.array([0, 0, 0, 0, 0, 0, 2.0])
    feas = viol == 0
    return key, viol, feas


def test_clearing_needs_a_picker():
    key, viol, feas = _cands()
    with pytest.raises(ValueError, match="rank1_pick"):
        three_rank_survival(key, viol, feas, 10, RankCfg(rank1_mode="clearing"))


def test_clearing_uses_the_picker_and_its_cap():
    key, viol, feas = _cands()
    seen = {}

    def pick(F, cap):
        seen["F"], seen["cap"] = F.copy(), cap
        return F[[0, 2]]                                      # the best and the third: e.g. the second was cleared

    s = three_rank_survival(key, viol, feas, 30, RankCfg(rank1_mode="clearing", y=0.1), rank1_pick=pick,
                            prev_r1=np.array([1, 1, 1, 1, 1, 1, 0], bool))
    assert list(seen["F"]) == [1, 2, 4, 3, 0, 5]              # feasible, (key, age) order, both equal-key twins
    assert seen["cap"] == 3                                   # ceil(0.1 * 30), the ratchet does not apply
    assert list(s.idx[s.rank == R1]) == [1, 4] and s.n1 == 2
    s = three_rank_survival(key, viol, feas, 30, RankCfg(rank1_mode="clearing", rank1_size=5), rank1_pick=pick)
    assert seen["cap"] == 5
    with pytest.raises(RuntimeError):
        three_rank_survival(key, viol, feas, 30, RankCfg(rank1_mode="clearing"), rank1_pick=lambda F, c: np.array([6]))


def test_clearing_keeps_distinct_equal_key_members():
    key, viol, feas = _cands()
    X = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]],
                 dtype=np.int8)
    s = three_rank_survival(key, viol, feas, 30, RankCfg(rank1_mode="clearing", rank1_size=4),
                            rank1_pick=lambda F, cap: clearing_pick(BK, X, F, np.arange(4), 2, cap))
    # order 1, 2, 4, 3, 0, 5: 1 joins; 2 (same key, 4 apart) joins; 4 is 2 from 1 and 2 -> joins; 3 is 1 from 1 -> cleared;
    # 0 is 2 from 1 and 2 -> joins (cap 4 reached)
    assert list(s.idx[s.rank == R1]) == [1, 2, 4, 0]


def test_top_mode_ignores_the_picker():
    key, viol, feas = _cands()
    a = three_rank_survival(key, viol, feas, 30, RankCfg())
    b = three_rank_survival(key, viol, feas, 30, RankCfg(), rank1_pick=lambda F, c: F[:0])
    assert np.array_equal(a.idx, b.idx) and np.array_equal(a.rank, b.rank)


# ------------------------------------------------------------------------------------------------- the config
def test_defaults_keep_every_recorded_hash():
    assert EAConfig().hash() == "9ac4a93bf93b"
    assert EAConfig.from_yaml(ROOT / "configs/itrea2-dev-rc-3way.yaml").hash() == "1d12ad490dd0"
    on = EAConfig().with_overrides({"ranks.rank1_mode": "clearing"})
    assert on.hash() != "9ac4a93bf93b"
    with pytest.raises(Exception):
        EAConfig().with_overrides({"ranks.rank1_mode": "nearest"}).validate()
    with pytest.raises(Exception):
        EAConfig().with_overrides({"ranks.rank1_size": -1}).validate()


# ------------------------------------------------------------------------------------------------- the engine
@pytest.mark.parametrize("name", REF.TOYS)
def test_engine_rank1_is_a_distinct_elite(name, tmp_path):
    p = REF._problem(name)
    lp = REF._lp_point(p)
    cfg = REF.config("numpy", log_every=1, **{"ranks.rank1_mode": "clearing", "ranks.rank1_min_diff": 2})
    eng = Engine(p, cfg, seed=17, mailbox=REF.LPMailbox(lp, float(p.objective(lp))), warmup=False,
                 trace=Trace(tmp_path, "ea"))
    eng.prepare()
    assert eng.startup["rank1"] == dict(mode="clearing", min_diff=2, n_int=int(p.isint.sum()),
                                        size=int(np.ceil(0.1 * eng.P)))
    eng.begin(time.monotonic(), 3600.0)
    for _ in range(REF.GENS):
        eng.step()
        r1 = np.flatnonzero(eng.pop.rank == R1)
        assert r1.size <= np.ceil(0.1 * eng.P) and eng.h_feas[r1].all()
        Xi = np.asarray(eng.pop.X)[:, np.asarray(eng.dp.ints)]
        for a, b in itertools.combinations(r1, 2):
            assert _diff(Xi[a], Xi[b]) >= 2
    out = eng.finish()
    eng.trace.close()
    assert out["final_r1_n"] == out["final_n1"] and "rank1_cleared" in out
    gens = [e for e in read_events(tmp_path / "ea.events.jsonl") if e["kind"] == "gen"]
    assert gens and all(k in e for e in gens for k in ("r1_n", "r1_dmax", "r1_dmax_band", "r1_gap", "r1_cleared"))


def test_clearing_without_integers_falls_back_to_top():
    from itrea.problem import from_arrays
    import scipy.sparse as sps
    p = from_arrays(sps.csr_matrix(np.array([[1.0, 1.0, 1.0]])), np.array([1.0]), np.array([2.0]), np.zeros(3),
                    np.ones(3), np.array([1.0, 2.0, 3.0]), np.zeros(3, bool))
    on = Engine(p, REF.config("numpy", **{"ranks.rank1_mode": "clearing"}), seed=1, warmup=False)
    off = Engine(p, REF.config("numpy"), seed=1, warmup=False)
    for e in (on, off):
        e.prepare()
        e.begin(time.monotonic(), 3600.0)
        for _ in range(3):
            e.step()
    assert on.startup["rank1"]["mode"].startswith("top") and on.r1_k is None
    assert np.array_equal(on.pop.ids, off.pop.ids) and np.array_equal(on.pop.rank, off.pop.rank)


def test_top_mode_logs_the_spread_but_no_clearing(tmp_path):
    p = toy.knapsack(30)
    eng = Engine(p, REF.config("numpy", log_every=1), seed=3, trace=Trace(tmp_path, "ea"), warmup=False)
    eng.prepare()
    assert eng.startup["rank1"] == dict(mode="top", min_diff=None, n_int=30, size=None)
    eng.begin(time.monotonic(), 3600.0)
    for _ in range(3):
        eng.step()
    out = eng.finish()
    eng.trace.close()
    assert "rank1_cleared" not in out and "final_r1_dmax" in out
    gens = [e for e in read_events(tmp_path / "ea.events.jsonl") if e["kind"] == "gen"]
    assert all("r1_n" in e and "r1_cleared" not in e for e in gens)


def test_top_mode_run_equals_the_reference_capture():
    """The default path draws nothing new and selects the same Rank 1: identical to the stored reference."""
    ref = REF.load_reference()
    for name in REF.TOYS:
        cur = REF.capture(name, "numpy")
        for k, v in cur.items():
            assert np.array_equal(ref[f"numpy/{name}/{k}"], v, equal_nan=True), (name, k)


# ---------------------------------------------------------------------------------------------------- CuPy path
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.float64])
def test_clearing_on_cupy_equals_numpy(dtype):
    from test_engine import gpu_or_skip
    gpu_or_skip()
    gb = get_backend("cupy", gpu_mem_gb=1.0)
    rng = np.random.default_rng(5)
    X = rng.integers(0, 3, size=(60, 30)).astype(dtype)
    ints = np.arange(25)
    order = rng.permutation(60)
    for k in (1, 4, 9):
        for block, budget in ((32, 1 << 26), (1, 1), (3, 100), (9, 1), (32, 1)):
            want = clearing_pick(BK, X, order, ints, k, 20)
            got = clearing_pick(gb, gb.xp.asarray(X), order, gb.xp.asarray(ints), k, 20, block=block, budget=budget)
            assert np.array_equal(want, got), (k, block)
    assert np.array_equal(clearing_pick(gb, gb.xp.asarray(X), order[:1], gb.xp.asarray(ints), 1, 5), order[:1])
    keys = rng.normal(size=60)
    assert elite_spread(BK, X, order[:7], keys[order[:7]], ints, band=0.5) == elite_spread(
        gb, gb.xp.asarray(X), order[:7], keys[order[:7]], gb.xp.asarray(ints), band=0.5)


@pytest.mark.gpu
def test_engine_clearing_runs_on_cupy():
    """The engine runs with clearing on the CuPy backend (reference toys) and keeps a Rank 1."""
    from test_engine import gpu_or_skip
    gpu_or_skip()
    for name in REF.TOYS:
        cur = REF.capture(name, "cupy", cfg=REF.config("cupy", **{"ranks.rank1_mode": "clearing",
                                                                   "ranks.rank1_min_diff": 2}))
        ranks = cur["pop_rank"]
        assert (ranks == R1).sum() >= 1


@pytest.mark.parametrize("name", REF.TOYS)
def test_warm_elite_runs_on_the_warmup_population(name):
    p = REF._problem(name)
    eng = Engine(p, REF.config("numpy"), seed=2, warmup=False)
    eng.prepare()
    eng.begin(time.monotonic(), 3600.0)
    eng.step()
    eng._warm_elite()                                        # both modes call it in the warm-up; must not raise


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["top", "clearing"])
def test_cupy_cold_cache_compiles_the_elite_kernels_before_the_clock(mode, tmp_path):
    """D-016 review: `elite_spread` runs at every gen record in BOTH modes and `clearing_pick` in clearing mode; their
    comparison kernels must be compiled by the warm-up (before t0) on every gene dtype — int8 (binaries), int32
    (general integers beyond ±127), float64 (continuous columns) — or a run sets compiled_on_clock and its block is
    void (D-002). Each run starts from an EMPTY CuPy cache."""
    import subprocess
    import scipy.sparse as sps
    from test_ea_run import MODULE, PY, SMALL, _env, _gpu_ok, _summary
    from itrea.problem import from_arrays
    if not _gpu_ok():
        pytest.skip("no CuPy device")
    rs = np.random.default_rng(4)
    n, m = 60, 20
    A = sps.csr_matrix(rs.integers(1, 9, size=(m, n)).astype(float) * (rs.random((m, n)) < 0.3))
    gen = from_arrays(A, np.zeros(m), np.full(m, 4000.0), np.zeros(n), np.full(n, 400.0),
                      -rs.integers(1, 50, n).astype(float), np.ones(n, bool), sense=1,
                      names=[f"g{j}" for j in range(n)])
    insts = {"int8": toy.write_mps(toy.knapsack(30), tmp_path / "knap.mps"),
             "int32": toy.write_mps(gen, tmp_path / "gen.mps"),
             "float64": toy.write_mps(toy.mixed(), tmp_path / "mixed.mps")}
    for dt, inst in insts.items():
        rd = tmp_path / f"run_{dt}"
        env = {k: v for k, v in _env(CUPY_CACHE_DIR=str(tmp_path / f"cache_{dt}")).items()
               if k != "CUPY_ACCELERATORS"}
        # y = 0.5 makes Rank 1 large on these small runs, so the timed run really reaches the spread's comparison
        # and the clearing's distance fetches after t0 (with the default y the earlier version of this test passed
        # without ever reaching them — 2026-10-08)
        pr = subprocess.run([PY, "-m", MODULE, "--instance", str(inst), "--seed", "0", "--budget", "4", "--run-dir",
                             str(rd), "--no-barrier", "--backend", "cupy", "--set", f"ranks.rank1_mode={mode}",
                             "--set", "ranks.y=0.5", "--set", "log_every=1", *SMALL], env=env, capture_output=True,
                            timeout=300)
        assert pr.returncode == 0, (pr.stdout + pr.stderr).decode(errors="replace")
        s = _summary(rd)
        assert s["gene_dtype"] == dt, (dt, s["gene_dtype"])
        assert all(s["startup"]["warmup"][p]["elite_warmed"] >= 1 for p in ("many", "one")), s["startup"]["warmup"]
        gens = [e for e in read_events(rd / "ea.events.jsonl") if e["kind"] == "gen"]
        assert any((e.get("r1_n") or 0) >= 3 and e.get("r1_dmax") is not None for e in gens), (dt, "never reached")
        assert s["compiled_on_clock"] is False and s["cupy_cache"]["new_after_t0"] == [], (dt, s["cupy_cache"])
