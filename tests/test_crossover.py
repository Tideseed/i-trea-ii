"""Crossover: dtype and bounds preservation, agreement keeps agreeing genes, block takes whole rows (row closure),
k_point cuts, operator frequencies — on NumPy and (marked gpu) on CuPy."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps
from hypothesis import given, settings, strategies as st

from itrea.backend import get_backend
from itrea.benchmarks import toy
from itrea.config import CrossoverCfg
from itrea.ea.crossover import ORIGIN_X_AGREE, crossover
from itrea.ea.evaluate import to_device
from itrea.ea.population import ORIGIN_X_BLOCK, ORIGIN_X_KPOINT, ORIGIN_X_UNIFORM, gene_dtype_for
from itrea.problem import from_arrays


def _has_cupy() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


HAS_CUPY = _has_cupy()
BACKENDS = [pytest.param("numpy", id="numpy"),
            pytest.param("cupy", id="cupy", marks=[pytest.mark.gpu,
                                                   pytest.mark.skipif(not HAS_CUPY, reason="CuPy/GPU unavailable")])]
_BK: dict = {}


def bk_of(name):
    if name not in _BK:
        _BK[name] = get_backend(name, gpu_mem_gb=1.0 if name == "cupy" else None)
    return _BK[name]


def _cfg(**kw) -> CrossoverCfg:
    base = dict(pc=1.0, weights={"agreement": 1.0})
    base.update(kw)
    return CrossoverCfg(**base)


def _rngs(bk, seed):
    return bk.rng(seed), np.random.default_rng(seed)


def binary_problem(n=40, m=12, density=0.15, seed=0):
    rs = np.random.default_rng(seed)
    A = (rs.random((m, n)) < density).astype(float)
    for i in range(m):
        if A[i].sum() == 0:
            A[i, rs.integers(n)] = 1.0
    return from_arrays(sps.csr_matrix(A), np.ones(m), np.full(m, np.inf), np.zeros(n), np.ones(n),
                       rs.integers(1, 9, n).astype(float), np.ones(n, bool))


def block_diagonal_problem(blocks=6, width=5):
    n, m = blocks * width, blocks
    A = np.zeros((m, n))
    for i in range(m):
        A[i, i * width:(i + 1) * width] = 1.0
    return from_arrays(sps.csr_matrix(A), np.ones(m), np.ones(m), np.zeros(n), np.ones(n), np.ones(n),
                       np.ones(n, bool))


def _parents(bk, prob, k, seed, agree_share=0.5):
    rs = np.random.default_rng(seed)
    dt = gene_dtype_for(prob.lb, prob.ub, prob.isint)
    A = rs.integers(0, 2, size=(k, prob.n))
    B = 1 - A
    same = rs.random((k, prob.n)) < agree_share
    B[same] = A[same]
    return bk.asarray(A.astype(dt)), bk.asarray(B.astype(dt))


# ---------------------------------------------------------------------------------------------------- agreement
@pytest.mark.parametrize("bname", BACKENDS)
def test_agreement_keeps_agreeing_genes(bname):
    bk = bk_of(bname)
    prob = binary_problem()
    dp = to_device(prob, bk)
    XA, XB = _parents(bk, prob, 400, 1)
    A0, B0 = bk.asnumpy(XA).copy(), bk.asnumpy(XB).copy()
    C, origin, touched = crossover(bk, dp, XA, XB, _cfg(), *_rngs(bk, 2))
    A, B, Ch, T = bk.asnumpy(XA), bk.asnumpy(XB), bk.asnumpy(C), bk.asnumpy(touched)
    assert np.array_equal(A, A0) and np.array_equal(B, B0)            # parents untouched
    assert C.dtype == XA.dtype
    agree = A == B
    assert np.array_equal(Ch[agree], A[agree])
    assert np.all((Ch == A) | (Ch == B))
    assert np.array_equal(T, ~agree)                                  # touched = disagreement set
    assert np.all(origin == ORIGIN_X_AGREE)
    share_b = np.mean(Ch[~agree] == B[~agree])
    assert abs(share_b - 0.5) < 0.02


@pytest.mark.parametrize("bname", BACKENDS)
def test_uniform_takes_p_swap_from_b(bname):
    bk = bk_of(bname)
    prob = binary_problem()
    dp = to_device(prob, bk)
    XA, XB = _parents(bk, prob, 400, 3, agree_share=0.0)
    C, origin, touched = crossover(bk, dp, XA, XB, _cfg(weights={"uniform": 1.0}, p_swap=0.3), *_rngs(bk, 4))
    A, B, Ch, T = (bk.asnumpy(v) for v in (XA, XB, C, touched))
    assert abs(np.mean(Ch == B) - 0.3) < 0.02
    assert np.array_equal(T, Ch != A)
    assert np.all(origin == ORIGIN_X_UNIFORM)


@pytest.mark.parametrize("bname", BACKENDS)
@pytest.mark.parametrize("k", [1, 3])
def test_k_point_segments(bname, k):
    bk = bk_of(bname)
    prob = binary_problem(n=25)
    dp = to_device(prob, bk)
    n, kids = prob.n, 300
    XA = bk.asarray(np.zeros((kids, n), np.int8))
    XB = bk.asarray(np.ones((kids, n), np.int8))
    C, origin, touched = crossover(bk, dp, XA, XB, _cfg(weights={"k_point": 1.0}, k=k), *_rngs(bk, 5))
    Ch = bk.asnumpy(C)
    switches = np.abs(np.diff(Ch.astype(int), axis=1)).sum(axis=1)
    assert np.all(switches == k)                                      # k distinct cuts, alternating segments
    assert np.all(Ch[:, 0] == 0)                                      # starts with parent A
    assert np.all(origin == ORIGIN_X_KPOINT)
    assert np.array_equal(bk.asnumpy(touched), Ch == 1)


# ------------------------------------------------------------------------------------------------------- block
@pytest.mark.parametrize("bname", BACKENDS)
def test_block_takes_whole_rows_block_diagonal(bname):
    bk = bk_of(bname)
    prob = block_diagonal_problem(blocks=6, width=5)
    dp = to_device(prob, bk)
    XA, XB = _parents(bk, prob, 300, 6, agree_share=0.7)
    br = 2
    C, origin, touched = crossover(bk, dp, XA, XB, _cfg(weights={"block": 1.0}, block_rows=br), *_rngs(bk, 7))
    A, B, Ch, T = (bk.asnumpy(v) for v in (XA, XB, C, touched))
    assert np.all(origin == ORIGIN_X_BLOCK)
    assert np.array_equal(Ch[T], B[T]) and np.array_equal(Ch[~T], A[~T])
    Tb = T.reshape(T.shape[0], 6, 5)
    full = Tb.all(axis=2)
    assert np.array_equal(Tb.any(axis=2), full)                       # a row is taken entirely or not at all
    dis_rows = (A != B).reshape(T.shape[0], 6, 5).any(axis=2)
    assert np.all(~full | dis_rows)                                   # only rows where the parents differ
    assert np.array_equal(full.sum(axis=1), np.minimum(br, dis_rows.sum(axis=1)))


@pytest.mark.parametrize("bname", BACKENDS)
def test_block_row_closure_general(bname):
    bk = bk_of(bname)
    prob = binary_problem(n=30, m=10, density=0.2, seed=11)
    dp = to_device(prob, bk)
    XA, XB = _parents(bk, prob, 200, 12, agree_share=0.5)
    C, _, touched = crossover(bk, dp, XA, XB, _cfg(weights={"block": 1.0}, block_rows=3), *_rngs(bk, 13))
    T = bk.asnumpy(touched)
    Ad = prob.A.toarray() != 0
    for t in T:
        inside = [i for i in range(prob.m) if Ad[i].any() and np.all(t[Ad[i]])]   # rows fully taken
        cover = np.zeros(prob.n, bool)
        for i in inside:
            cover |= Ad[i]
        assert np.array_equal(cover | ~t, np.ones(prob.n, bool))      # every touched gene lies in a fully taken row


@pytest.mark.parametrize("bname", BACKENDS)
def test_block_identical_parents_copy_of_a(bname):
    bk = bk_of(bname)
    prob = binary_problem()
    dp = to_device(prob, bk)
    XA, _ = _parents(bk, prob, 20, 14)
    C, origin, touched = crossover(bk, dp, XA, XA.copy(), _cfg(weights={"block": 1.0}), *_rngs(bk, 15))
    assert np.array_equal(bk.asnumpy(C), bk.asnumpy(XA))
    assert not bk.asnumpy(touched).any()


def test_block_not_applicable_without_rows():
    bk = bk_of("numpy")
    n = 6
    prob = from_arrays(sps.csr_matrix((0, n)), np.empty(0), np.empty(0), np.zeros(n), np.ones(n), np.ones(n),
                       np.ones(n, bool))
    dp = to_device(prob, bk)
    XA, XB = np.zeros((50, n), np.int8), np.ones((50, n), np.int8)
    C, origin, touched = crossover(bk, dp, XA, XB, _cfg(weights={"block": 1.0}), *_rngs(bk, 16))
    assert touched is None and np.all(origin == 0) and np.array_equal(C, XA)
    C, origin, _ = crossover(bk, dp, XA, XB, _cfg(weights={"block": 0.5, "uniform": 0.5}), *_rngs(bk, 16))
    assert np.all(origin == ORIGIN_X_UNIFORM)                         # block's weight renormalised away


# -------------------------------------------------------------------------------------------- operator choice
@pytest.mark.parametrize("bname", BACKENDS)
def test_operator_frequencies(bname):
    bk = bk_of(bname)
    prob = binary_problem()
    dp = to_device(prob, bk)
    k = 6000
    XA, XB = _parents(bk, prob, k, 17)
    cfg = _cfg(pc=0.8, weights={"agreement": 0.5, "uniform": 0.3, "block": 0.2})
    _, origin, _ = crossover(bk, dp, XA, XB, cfg, *_rngs(bk, 18))
    assert abs(np.mean(origin == 0) - 0.2) < 0.02
    assert abs(np.mean(origin == ORIGIN_X_BLOCK) - 0.16) < 0.02
    if ORIGIN_X_AGREE == ORIGIN_X_UNIFORM:
        assert abs(np.mean(origin == ORIGIN_X_UNIFORM) - 0.64) < 0.02
    else:
        assert abs(np.mean(origin == ORIGIN_X_AGREE) - 0.4) < 0.02
        assert abs(np.mean(origin == ORIGIN_X_UNIFORM) - 0.24) < 0.02


@pytest.mark.parametrize("bname", BACKENDS)
def test_pc_zero_copies_a(bname):
    bk = bk_of(bname)
    prob = binary_problem()
    dp = to_device(prob, bk)
    XA, XB = _parents(bk, prob, 30, 19)
    C, origin, touched = crossover(bk, dp, XA, XB, _cfg(pc=0.0), *_rngs(bk, 20))
    assert touched is None and np.all(origin == 0)
    assert C is not XA and np.array_equal(bk.asnumpy(C), bk.asnumpy(XA))


def test_dtype_mismatch_raises():
    bk = bk_of("numpy")
    dp = to_device(binary_problem(), bk)
    with pytest.raises(TypeError):
        crossover(bk, dp, np.zeros((2, 40), np.int8), np.zeros((2, 40), np.float64), _cfg(), *_rngs(bk, 0))


# ------------------------------------------------------------------------------------------ property: any problem
def _random_problem(rs, n, m):
    isint = rs.random(n) < 0.7
    lb = np.where(rs.random(n) < 0.2, -np.inf, rs.integers(-5, 3, n).astype(float))
    ub = np.where(rs.random(n) < 0.2, np.inf, lb + rs.integers(0, 6, n))
    ub = np.where(np.isinf(lb) & np.isinf(ub), np.inf, np.where(np.isinf(lb), rs.integers(-3, 4, n), ub))
    A = (rs.random((m, n)) < 0.5) * rs.integers(-3, 4, (m, n))
    rl = np.where(rs.random(m) < 0.5, -np.inf, -2.0)
    ru = np.where(np.isinf(rl), 4.0, np.inf)
    return from_arrays(sps.csr_matrix(A.astype(float)), rl, ru, lb, ub, rs.normal(size=n), isint)


def _in_bounds_sample(rs, prob, k, dt):
    lo = np.where(np.isfinite(prob.lb), prob.lb, np.where(np.isfinite(prob.ub), prob.ub - 10, -10))
    hi = np.where(np.isfinite(prob.ub), prob.ub, lo + 10)
    X = lo + rs.random((k, prob.n)) * (hi - lo)
    X[:, prob.isint] = np.clip(np.round(X[:, prob.isint]), np.ceil(lo[prob.isint]), np.floor(hi[prob.isint]))
    return X.astype(dt)


def _property_check(bname, seed, n, m, dtype_req, ops):
    bk = bk_of(bname)
    rs = np.random.default_rng(seed)
    prob = _random_problem(rs, n, m)
    if dtype_req != "float64":
        prob = from_arrays(prob.A, prob.rl, prob.ru, np.clip(np.nan_to_num(prob.lb, neginf=-7), -7, 7),
                           np.clip(np.nan_to_num(prob.ub, posinf=7), -7, 7), prob.c, np.ones(n, bool))
        prob.ub[:] = np.maximum(prob.ub, prob.lb)
    dt = gene_dtype_for(prob.lb, prob.ub, prob.isint, dtype_req)
    dp = to_device(prob, bk)
    k = int(rs.integers(1, 12))
    A, B = _in_bounds_sample(rs, prob, k, dt), _in_bounds_sample(rs, prob, k, dt)
    XA, XB = bk.asarray(A), bk.asarray(B)
    w = {o: float(rs.random() + 0.01) for o in ops}
    cfg = CrossoverCfg(pc=float(rs.random()), weights=w, p_swap=float(rs.random()), k=int(rs.integers(1, 4)),
                       block_rows=int(rs.integers(1, 4)))
    C, origin, touched = crossover(bk, dp, XA, XB, cfg, *_rngs(bk, seed), mem_budget_bytes=int(rs.integers(1, 4096)))
    Ch = bk.asnumpy(C)
    assert C.dtype == np.dtype(dt) and Ch.shape == (k, n)
    assert np.array_equal(bk.asnumpy(XA), A) and np.array_equal(bk.asnumpy(XB), B)
    assert np.all((Ch == A) | (Ch == B))
    assert np.all(Ch >= prob.lb - 1e-12) and np.all(Ch <= prob.ub + 1e-12)
    assert origin.dtype == np.uint16 and origin.shape == (k,)
    if touched is not None:
        T = bk.asnumpy(touched)
        assert T.dtype == bool and T.shape == (k, n)
        assert np.all(T | (Ch == A))                                   # every gene that changed is touched
        assert not T[origin == 0].any()


OPS = st.lists(st.sampled_from(["agreement", "uniform", "k_point", "block", "none"]), min_size=1, max_size=5,
               unique=True)


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 2**31), n=st.integers(1, 12), m=st.integers(0, 6),
       dtype_req=st.sampled_from(["float64", "int32", "int8"]), ops=OPS)
def test_property_numpy(seed, n, m, dtype_req, ops):
    _property_check("numpy", seed, n, m, dtype_req, ops)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUPY, reason="CuPy/GPU unavailable")
@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 2**31), n=st.integers(1, 12), m=st.integers(0, 6),
       dtype_req=st.sampled_from(["float64", "int32", "int8"]), ops=OPS)
def test_property_cupy(seed, n, m, dtype_req, ops):
    _property_check("cupy", seed, n, m, dtype_req, ops)


def test_toy_knapsack_runs_both_dtypes():
    bk = bk_of("numpy")
    prob = toy.knapsack(20)
    dp = to_device(prob, bk)
    XA, XB = _parents(bk, prob, 8, 21)
    for dt in (np.int8, np.float64):
        C, _, _ = crossover(bk, dp, XA.astype(dt), XB.astype(dt), _cfg(weights={"block": 1.0}), *_rngs(bk, 22))
        assert C.dtype == dt
