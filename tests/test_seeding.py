"""Seeding: bounds, integrality and gene dtype of the initial population (no LP), LP randomised rounding, immigrants,
and the LP relaxation worker process on toy MPS files."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps
from hypothesis import given, settings, strategies as st
from scipy.optimize import linprog

from itrea.backend import get_backend
from itrea.benchmarks import toy
from itrea.config import SeedingCfg
from itrea.ea.evaluate import to_device
from itrea.ea.population import (ORIGIN_IMMIGRANT, ORIGIN_M_RESET, ORIGIN_SEED_LP, ORIGIN_SEED_RANDOM,
                                 gene_dtype_for)
from itrea.ea.seeding import LPWorker, immigrants, lp_relaxation, lp_round, seed_population
from itrea.problem import from_arrays


def _has_cupy() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


HAS_CUPY = _has_cupy()
GPU = [pytest.mark.gpu, pytest.mark.skipif(not HAS_CUPY, reason="CuPy/GPU unavailable")]
BACKENDS = [pytest.param("numpy", id="numpy"), pytest.param("cupy", id="cupy", marks=GPU)]
_BK: dict = {}


def bk_of(name):
    if name not in _BK:
        _BK[name] = get_backend(name, gpu_mem_gb=1.0 if name == "cupy" else None)
    return _BK[name]


def _rngs(bk, seed):
    return bk.rng(seed), np.random.default_rng(seed)


def random_problem(rs, n, m, all_int=False):
    isint = np.ones(n, bool) if all_int else rs.random(n) < 0.6
    kind = rs.integers(0, 5, n)          # 0 box, 1 [lb, inf), 2 (-inf, ub], 3 free, 4 fixed
    lb = rs.integers(-5, 3, n).astype(float)
    ub = lb + rs.integers(1, 6, n)
    lb[kind == 2] = -np.inf
    ub[kind == 1] = np.inf
    lb[kind == 3], ub[kind == 3] = -np.inf, np.inf
    ub[kind == 4] = lb[kind == 4]
    bins = rs.random(n) < 0.3
    lb[bins & isint], ub[bins & isint] = 0.0, 1.0
    if all_int:
        lb, ub = np.where(np.isfinite(lb), lb, -7.0), np.where(np.isfinite(ub), ub, 7.0)
        ub = np.maximum(ub, lb)
    A = (rs.random((m, n)) < 0.5) * rs.integers(-3, 4, (m, n))
    rl = np.where(rs.random(m) < 0.5, -np.inf, -1.0)
    ru = np.where(np.isfinite(rl), np.inf, 3.0)
    return from_arrays(sps.csr_matrix(A.astype(float)), rl, ru, lb, ub, rs.normal(size=n), isint)


def _window(prob):
    lo = np.where(np.isfinite(prob.lb), prob.lb, np.where(np.isfinite(prob.ub), prob.ub - 10, -10))
    hi = np.where(np.isfinite(prob.ub), prob.ub, np.where(np.isfinite(prob.lb), prob.lb + 10, 10))
    return lo, hi


def _check_rows(prob, Xh, dt):
    assert Xh.dtype == np.dtype(dt)
    X = Xh.astype(np.float64)
    assert np.all(np.isfinite(X))
    xi = X[:, prob.isint]
    assert np.all(xi == np.round(xi))
    assert np.all(X >= prob.lb) and np.all(X <= prob.ub)


# ---------------------------------------------------------------------------------------------- seed_population
def _seed_property(bname, seed, n, m, P, share, dtype_req):
    bk = bk_of(bname)
    rs = np.random.default_rng(seed)
    prob = random_problem(rs, n, m, all_int=(dtype_req != "float64"))
    dt = gene_dtype_for(prob.lb, prob.ub, prob.isint, dtype_req)
    dp = to_device(prob, bk)
    X, origin = seed_population(bk, dp, P, SeedingCfg(random=share), *_rngs(bk, seed), gene_dtype=dtype_req,
                                mem_budget_bytes=int(rs.integers(1, 1 << 14)))
    Xh = bk.asnumpy(X)
    assert Xh.shape == (P, n) and origin.shape == (P,) and origin.dtype == np.uint16
    _check_rows(prob, Xh, dt)
    n_rand = int(round(share * P))
    assert np.all(origin & ORIGIN_SEED_RANDOM) and not np.any(origin & ORIGIN_SEED_LP)
    lo, hi = _window(prob)
    Xf = Xh.astype(np.float64)
    assert np.all(Xf[:n_rand] >= lo - 1e-9) and np.all(Xf[:n_rand] <= hi + 1e-9)
    assert np.all(origin[:n_rand] == ORIGIN_SEED_RANDOM)
    if P > n_rand:
        zero = np.clip(0.0, np.where(prob.isint, np.ceil(prob.lb), prob.lb),
                       np.where(prob.isint, np.floor(prob.ub), prob.ub))
        assert np.array_equal(Xf[n_rand], zero)                     # the exact zero-ish point
        assert np.all((origin[n_rand + 1:] & ~np.uint16(ORIGIN_M_RESET)) == ORIGIN_SEED_RANDOM)


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(0, 2**31), n=st.integers(1, 12), m=st.integers(0, 5), P=st.integers(1, 30),
       share=st.floats(0, 1), dtype_req=st.sampled_from(["auto", "float64", "int32", "int8"]))
def test_seed_population_property_numpy(seed, n, m, P, share, dtype_req):
    _seed_property("numpy", seed, n, m, P, share, dtype_req)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUPY, reason="CuPy/GPU unavailable")
@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 2**31), n=st.integers(1, 12), m=st.integers(0, 5), P=st.integers(1, 30),
       share=st.floats(0, 1), dtype_req=st.sampled_from(["auto", "float64", "int32", "int8"]))
def test_seed_population_property_cupy(seed, n, m, P, share, dtype_req):
    _seed_property("cupy", seed, n, m, P, share, dtype_req)


@pytest.mark.parametrize("bname", BACKENDS)
def test_seed_uniform_covers_integer_domain(bname):
    bk = bk_of(bname)
    n = 3
    prob = from_arrays(sps.csr_matrix((0, n)), np.empty(0), np.empty(0), np.array([0.0, -2.0, 3.0]),
                       np.array([1.0, 2.0, 3.0]), np.zeros(n), np.ones(n, bool))
    dp = to_device(prob, bk)
    X, _ = seed_population(bk, dp, 5000, SeedingCfg(random=1.0), *_rngs(bk, 1))
    Xh = bk.asnumpy(X)
    assert X.dtype == np.int8
    for j, vals in enumerate([[0, 1], [-2, -1, 0, 1, 2], [3]]):
        freq = np.array([np.mean(Xh[:, j] == v) for v in vals])
        assert np.all(np.abs(freq - 1 / len(vals)) < 0.03)


@pytest.mark.parametrize("bname", BACKENDS)
def test_zeroish_seeds_are_distinct_and_mostly_feasible_on_knapsack(bname):
    bk = bk_of(bname)
    prob = toy.knapsack(30)
    dp = to_device(prob, bk)
    X, origin = seed_population(bk, dp, 40, SeedingCfg(random=0.0), *_rngs(bk, 2))
    Xh = bk.asnumpy(X)
    assert np.all(Xh[0] == 0)
    assert len({r.tobytes() for r in Xh}) > 35
    assert np.all(origin[1:] == ORIGIN_SEED_RANDOM | ORIGIN_M_RESET)
    assert np.mean([prob.check(r.astype(float)).feasible for r in Xh]) > 0.9


# ----------------------------------------------------------------------------------------------------- lp_round
@pytest.mark.parametrize("bname", BACKENDS)
def test_lp_round_floor_or_ceil_with_fractional_probability(bname):
    bk = bk_of(bname)
    prob = toy.mixed()                      # x0, x1 integer [0, 5]; y0 >= 0; y1 free
    dp = to_device(prob, bk)
    lp = np.array([2.25, 4.9, -1.0, 7.5])
    X, origin = lp_round(bk, dp, bk.asarray(lp), 8000, *_rngs(bk, 3))
    Xh = bk.asnumpy(X)
    assert X.dtype == np.float64 and np.all(origin == ORIGIN_SEED_LP)
    assert set(np.unique(Xh[:, 0])) == {2.0, 3.0} and abs(np.mean(Xh[:, 0] == 3) - 0.25) < 0.02
    assert set(np.unique(Xh[:, 1])) == {4.0, 5.0} and abs(np.mean(Xh[:, 1] == 5) - 0.9) < 0.02
    assert np.all(Xh[:, 2] == 0.0) and np.all(Xh[:, 3] == 7.5)       # continuous: LP value clipped to bounds


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**31), n=st.integers(1, 12), k=st.integers(0, 20),
       dtype_req=st.sampled_from(["auto", "int8", "float64"]))
def test_lp_round_property(seed, n, k, dtype_req):
    bk = bk_of("numpy")
    rs = np.random.default_rng(seed)
    prob = random_problem(rs, n, 2, all_int=(dtype_req == "int8"))
    dt = gene_dtype_for(prob.lb, prob.ub, prob.isint, dtype_req)
    dp = to_device(prob, bk)
    lp = rs.normal(size=n) * 8
    lp[rs.random(n) < 0.1] = np.nan
    X, origin = lp_round(bk, dp, list(lp), k, *_rngs(bk, seed), gene_dtype=dtype_req)
    Xh = bk.asnumpy(X)
    assert Xh.shape == (k, n) and origin.shape == (k,)
    _check_rows(prob, Xh, dt)


# --------------------------------------------------------------------------------------------------- immigrants
@pytest.mark.parametrize("bname", BACKENDS)
def test_immigrants_composition(bname):
    bk = bk_of(bname)
    prob = toy.knapsack(20)
    dp = to_device(prob, bk)
    front = np.zeros((3, prob.n))
    front[:, :3] = np.eye(3)
    X, origin = immigrants(bk, dp, 20, *_rngs(bk, 4), lp_x=np.full(prob.n, 0.5), frontier_rows=bk.asarray(front),
                           lp_share=0.5)
    Xh = bk.asnumpy(X)
    assert X.dtype == np.int8 and Xh.shape == (20, prob.n)
    assert np.all(origin & ORIGIN_IMMIGRANT)
    assert np.array_equal(Xh[:3], front) and np.all(origin[:3] == ORIGIN_IMMIGRANT)
    assert np.sum(origin == (ORIGIN_IMMIGRANT | ORIGIN_SEED_LP)) == round(0.5 * 17)          # = 8
    assert np.sum(origin == (ORIGIN_IMMIGRANT | ORIGIN_SEED_RANDOM)) == 17 - round(0.5 * 17)
    _check_rows(prob, Xh, np.int8)


def test_immigrants_without_lp_or_frontier_are_random():
    bk = bk_of("numpy")
    prob = toy.set_cover()
    dp = to_device(prob, bk)
    X, origin = immigrants(bk, dp, 12, *_rngs(bk, 5))
    assert np.all(origin == (ORIGIN_IMMIGRANT | ORIGIN_SEED_RANDOM)) and X.shape == (12, prob.n)
    X, origin = immigrants(bk, dp, 0, *_rngs(bk, 5))
    assert X.shape == (0, prob.n) and origin.size == 0


def test_immigrants_more_frontier_rows_than_k():
    bk = bk_of("numpy")
    prob = toy.mixed()
    dp = to_device(prob, bk)
    front = np.tile(np.array([1.0, 4.0, 0.0, 4.0]), (5, 1))
    X, origin = immigrants(bk, dp, 2, *_rngs(bk, 6), frontier_rows=front)
    assert np.array_equal(X, front[:2]) and np.all(origin == ORIGIN_IMMIGRANT)


# --------------------------------------------------------------------------------------------------- LP relaxation
def _fractional_knapsack(prob):
    w = prob.A.toarray().ravel()
    v = prob.c
    W = prob.ru[0]
    order = np.argsort(-v / w, kind="stable")
    tot, cap = 0.0, W
    for j in order:
        take = min(1.0, cap / w[j])
        tot += take * v[j]
        cap -= take * w[j]
        if cap <= 0:
            break
    return tot


def _linprog_opt(prob):
    res = linprog(prob.sense * prob.c, A_ub=np.vstack([prob.A.toarray()[np.isfinite(prob.ru)],
                                                     -prob.A.toarray()[np.isfinite(prob.rl)]]),
                  b_ub=np.r_[prob.ru[np.isfinite(prob.ru)], -prob.rl[np.isfinite(prob.rl)]],
                  bounds=list(zip(np.where(np.isfinite(prob.lb), prob.lb, None),
                                  np.where(np.isfinite(prob.ub), prob.ub, None))), method="highs")
    assert res.status == 0
    return prob.sense * res.fun + prob.offset


@pytest.mark.solver
def test_lp_relaxation_in_process_matches_linprog():
    prob = toy.mixed()
    x, info = lp_relaxation(prob, time_limit=10.0)
    assert info["state"] == "done" and info["status"] == "Optimal"
    assert abs(info["objective"] - _linprog_opt(prob)) < 1e-7
    assert abs(prob.objective(x) - info["objective"]) < 1e-7


@pytest.mark.solver
def test_lp_worker_returns_lp_optimum_knapsack(tmp_path):
    prob = toy.knapsack(30)
    path = toy.write_mps(prob, tmp_path / "knap.mps")
    with LPWorker(path, time_limit=10.0, names=prob.names) as w:
        assert w.poll()[1]["state"] == "not_started"
        w.start()
        x, info = _wait(w)
    assert info["state"] == "done", info
    assert abs(info["objective"] - _fractional_knapsack(prob)) < 1e-6
    assert abs(prob.objective(x) - info["objective"]) < 1e-6
    assert np.all(x >= -1e-9) and np.all(x <= 1 + 1e-9)
    assert np.sum((x > 1e-9) & (x < 1 - 1e-9)) <= 1                  # one fractional item at the LP optimum
    assert prob.A @ x <= prob.ru[0] + 1e-6


@pytest.mark.solver
def test_lp_worker_mixed_and_name_order(tmp_path):
    prob = toy.mixed()
    path = toy.write_mps(prob, tmp_path / "mixed.mps")
    ref = _linprog_opt(prob)
    w = LPWorker(path, time_limit=10.0, names=prob.names).start()
    try:
        x, info = _wait(w)
    finally:
        w.stop()
    assert info["state"] == "done" and abs(info["objective"] - ref) < 1e-7
    assert np.allclose(x, [0.0, 4.5, 0.0, 5.5])
    w2 = LPWorker(path, time_limit=10.0, names=list(reversed(prob.names))).start()
    try:
        x2, info2 = _wait(w2)
    finally:
        w2.stop()
    assert info2.get("reordered") and np.allclose(x2, x[::-1])


@pytest.mark.solver
def test_lp_worker_affinity_and_failure(tmp_path):
    import os
    prob = toy.set_cover()
    path = toy.write_mps(prob, tmp_path / "sc.mps")
    cpu = sorted(os.sched_getaffinity(0))[0]
    w = LPWorker(path, time_limit=10.0, cpus=[cpu]).start()
    try:
        x, info = _wait(w)
    finally:
        w.stop()
    assert info["state"] == "done" and info["cpus"] == [cpu] and x.shape == (prob.n,)
    assert info["omp_num_threads"] == "1"                              # set for the child only
    bad = LPWorker(tmp_path / "missing.mps", time_limit=5.0).start()
    try:
        x, info = _wait(bad)
    finally:
        bad.stop()
    assert x is None and info["state"] == "failed" and "could not read" in info["error"]
    assert bad.poll()[1]["state"] == "failed"                          # final state is sticky


@pytest.mark.solver
def test_lp_worker_stop_before_result(tmp_path):
    prob = toy.knapsack(30)
    path = toy.write_mps(prob, tmp_path / "knap.mps")
    w = LPWorker(path, time_limit=10.0).start()
    w.stop()
    x, info = w.poll()
    assert info["state"] in ("stopped", "done")                        # it may have finished already
    assert not w.running
    w.stop()                                                           # idempotent


def _wait(w, timeout=20.0):
    import time
    t = time.monotonic()
    while time.monotonic() - t < timeout:
        x, info = w.poll()
        if info["state"] not in ("running", "not_started"):
            return x, info
        time.sleep(0.02)
    raise AssertionError("LP worker did not finish in time")
