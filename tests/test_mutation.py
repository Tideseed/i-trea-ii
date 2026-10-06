"""Mutation: integrality, bounds and dtype after every call (property tests on random problems, both backends),
gene-count distribution, operator frequencies, and the direction of each guided operator."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps
from hypothesis import given, settings, strategies as st

from itrea.backend import get_backend
from itrea.benchmarks import toy
from itrea.config import MutationCfg
from itrea.ea.evaluate import to_device
from itrea.ea.mutation import OPERATORS, mutate
from itrea.ea.population import (ORIGIN_M_FLIP, ORIGIN_M_OBJ, ORIGIN_M_RESET, ORIGIN_M_STEP, ORIGIN_M_TOWARD_LP,
                                 ORIGIN_M_VIOL, gene_dtype_for)
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


def _cfg(weights, **kw) -> MutationCfg:
    return MutationCfg(weights=weights, **kw)


def _rngs(bk, seed):
    return bk.rng(seed), np.random.default_rng(seed)


def binary_problem(n, cost=None, rows=None):
    A = sps.csr_matrix(rows if rows is not None else np.ones((1, n)))
    m = A.shape[0]
    c = np.ones(n) if cost is None else np.asarray(cost, float)
    return from_arrays(A, np.full(m, -np.inf), np.full(m, float(n)), np.zeros(n), np.ones(n), c, np.ones(n, bool))


def general_int_problem(n=200, lo=0, hi=10):
    return from_arrays(sps.csr_matrix(np.ones((1, n))), [-np.inf], [1e9], np.full(n, lo, float), np.full(n, hi, float),
                       np.ones(n), np.ones(n, bool))


# ------------------------------------------------------------------------------------------- random problems
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
    if all_int:                           # int8/int32 storage needs finite bounds
        lb, ub = np.where(np.isfinite(lb), lb, -7.0), np.where(np.isfinite(ub), ub, 7.0)
        ub = np.maximum(ub, lb)
    A = (rs.random((m, n)) < 0.5) * rs.integers(-3, 4, (m, n))
    rl = np.where(rs.random(m) < 0.5, -np.inf, rs.integers(-4, 3, m).astype(float))
    ru = np.where(np.isfinite(rl) & (rs.random(m) < 0.5), np.inf, rs.integers(0, 6, m).astype(float))
    ru = np.maximum(ru, np.where(np.isfinite(rl), rl, -np.inf))
    c = rs.normal(size=n) * (rs.random(n) < 0.8)
    return from_arrays(sps.csr_matrix(A.astype(float)), rl, ru, lb, ub, c, isint)


def sample_in_bounds(rs, prob, k, dt):
    lo = np.where(np.isfinite(prob.lb), prob.lb, np.where(np.isfinite(prob.ub), prob.ub - 10, -10))
    hi = np.where(np.isfinite(prob.ub), prob.ub, lo + 10)
    X = lo + rs.random((k, prob.n)) * (hi - lo)
    X[:, prob.isint] = np.clip(np.round(X[:, prob.isint]), np.ceil(lo[prob.isint]), np.floor(hi[prob.isint]))
    return X.astype(dt)


def _property_check(bname, seed, n, m, dtype_req, ops, use_act, use_lp, rate):
    bk = bk_of(bname)
    rs = np.random.default_rng(seed)
    prob = random_problem(rs, n, m, all_int=(dtype_req != "float64"))
    dt = gene_dtype_for(prob.lb, prob.ub, prob.isint, dtype_req)
    dp = to_device(prob, bk)
    k = int(rs.integers(1, 15))
    X0 = sample_in_bounds(rs, prob, k, dt)
    C = bk.asarray(X0.copy())
    lp_x = None
    if use_lp:
        lp_x = np.clip(rs.normal(size=n) * 3, np.nan_to_num(prob.lb, neginf=-50), np.nan_to_num(prob.ub, posinf=50))
        if bname == "cupy" and rs.random() < 0.5:
            lp_x = bk.asarray(lp_x)
    act = dp.A @ bk.asarray(X0.astype(np.float64)).T if use_act else None
    cfg = MutationCfg(genes=float(rs.random() * 6), rate=(float(rs.random()) if rate else None),
                      min_genes=int(rs.integers(0, 3)), weights={o: float(rs.random() + 0.01) for o in ops},
                      step_max=int(rs.integers(1, 4)), reset_window=float(rs.integers(1, 20)))
    stats: dict = {}
    origin = mutate(bk, dp, C, cfg, *_rngs(bk, seed), lp_x=lp_x, act=act,
                    mem_budget_bytes=int(rs.integers(1, 1 << 16)), stats=stats)
    Xh = bk.asnumpy(C)
    assert C.dtype == np.dtype(dt) and Xh.shape == (k, n)
    assert origin.dtype == np.uint16 and origin.shape == (k,)
    xi = Xh[:, prob.isint].astype(np.float64)
    assert np.all(xi == np.round(xi))
    assert np.all(Xh >= prob.lb) and np.all(Xh <= prob.ub)
    fixed = prob.ub == prob.lb
    assert np.array_equal(Xh[:, fixed], X0[:, fixed])                     # fixed variables never move
    cont = ~prob.isint
    not_reset = (origin & ORIGIN_M_RESET) == 0
    assert np.array_equal(Xh[not_reset][:, cont], X0[not_reset][:, cont])  # continuous genes move only by reset
    changed = np.any(Xh != X0, axis=1)
    assert np.all(changed == (origin != 0))                               # a bit exactly when something changed
    assert all(bin(int(o)).count("1") <= 1 for o in origin)               # one operator per child
    if "toward_lp" not in ops or lp_x is None:
        assert not np.any(origin & ORIGIN_M_TOWARD_LP)


OPS = st.lists(st.sampled_from(list(OPERATORS)), min_size=1, max_size=6, unique=True)


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 2**31), n=st.integers(1, 12), m=st.integers(0, 6),
       dtype_req=st.sampled_from(["float64", "int32", "int8"]), ops=OPS, use_act=st.booleans(),
       use_lp=st.booleans(), rate=st.booleans())
def test_property_numpy(seed, n, m, dtype_req, ops, use_act, use_lp, rate):
    _property_check("numpy", seed, n, m, dtype_req, ops, use_act, use_lp, rate)


@pytest.mark.gpu
@pytest.mark.skipif(not HAS_CUPY, reason="CuPy/GPU unavailable")
@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31), n=st.integers(1, 12), m=st.integers(0, 6),
       dtype_req=st.sampled_from(["float64", "int32", "int8"]), ops=OPS, use_act=st.booleans(),
       use_lp=st.booleans(), rate=st.booleans())
def test_property_cupy(seed, n, m, dtype_req, ops, use_act, use_lp, rate):
    _property_check("cupy", seed, n, m, dtype_req, ops, use_act, use_lp, rate)


# ------------------------------------------------------------------------------------------- gene counts
@pytest.mark.parametrize("bname", BACKENDS)
def test_gene_count_is_poisson(bname):
    bk = bk_of(bname)
    n, k = 2000, 4000
    prob = binary_problem(n)
    dp = to_device(prob, bk)
    X0 = np.random.default_rng(0).integers(0, 2, (k, n)).astype(np.int8)
    C = bk.asarray(X0.copy())
    mutate(bk, dp, C, _cfg({"flip": 1.0}, genes=4.0, min_genes=0), *_rngs(bk, 1))
    g = (bk.asnumpy(C) != X0).sum(axis=1)
    assert abs(g.mean() - 4.0) < 0.12 and abs(g.var() - 4.0) < 0.4


def test_gene_count_min_genes_and_rate():
    bk = bk_of("numpy")
    n, k = 1000, 3000
    prob = binary_problem(n)
    dp = to_device(prob, bk)
    X0 = np.zeros((k, n), np.int8)
    C = X0.copy()
    mutate(bk, dp, C, _cfg({"flip": 1.0}, genes=0.0, min_genes=2), *_rngs(bk, 2))
    assert np.all((C != X0).sum(axis=1) == 2)
    C = X0.copy()
    mutate(bk, dp, C, _cfg({"flip": 1.0}, rate=0.01, min_genes=0), *_rngs(bk, 3))
    g = (C != X0).sum(axis=1)
    assert abs(g.mean() - 10.0) < 0.25                                    # Binomial(1000, 0.01)


def test_gene_count_capped_by_candidates():
    bk = bk_of("numpy")
    prob = binary_problem(3)
    dp = to_device(prob, bk)
    C = np.zeros((50, 3), np.int8)
    mutate(bk, dp, C, _cfg({"flip": 1.0}, genes=0.0, min_genes=10), *_rngs(bk, 4))
    assert np.all(C == 1)


# ------------------------------------------------------------------------------------------ operator choice
@pytest.mark.parametrize("bname", BACKENDS)
def test_operator_frequencies(bname):
    bk = bk_of(bname)
    nb, ng = 100, 100
    n = nb + ng
    ub = np.r_[np.ones(nb), np.full(ng, 10.0)]
    prob = from_arrays(sps.csr_matrix(np.ones((1, n))), [-np.inf], [1e9], np.zeros(n), ub, np.ones(n),
                       np.ones(n, bool))
    dp = to_device(prob, bk)
    k = 6000
    C = bk.asarray(np.full((k, n), 0, np.int8))
    origin = mutate(bk, dp, C, _cfg({"flip": 0.6, "step": 0.4, "toward_lp": 5.0}), *_rngs(bk, 5))
    # toward_lp has no LP point: dropped and the rest renormalised
    assert abs(np.mean(origin == ORIGIN_M_FLIP) - 0.6) < 0.02
    assert abs(np.mean(origin == ORIGIN_M_STEP) - 0.4) < 0.02
    assert not np.any(origin & ORIGIN_M_TOWARD_LP)


def test_nothing_applicable_falls_back_to_reset():
    bk = bk_of("numpy")
    prob = general_int_problem(50)
    dp = to_device(prob, bk)
    C = np.full((100, 50), 5, np.int8)
    origin = mutate(bk, dp, C, _cfg({"flip": 1.0}), *_rngs(bk, 6))      # no binaries -> flip impossible
    assert np.all(origin == ORIGIN_M_RESET)


# -------------------------------------------------------------------------------------------- operator moves
@pytest.mark.parametrize("bname", BACKENDS)
def test_step_moves_at_most_step_max(bname):
    bk = bk_of(bname)
    prob = general_int_problem(300, 0, 10)
    dp = to_device(prob, bk)
    X0 = np.random.default_rng(7).integers(0, 11, (500, 300)).astype(np.int8)
    C = bk.asarray(X0.copy())
    mutate(bk, dp, C, _cfg({"step": 1.0}, step_max=3), *_rngs(bk, 8))
    d = np.abs(bk.asnumpy(C).astype(int) - X0)
    assert d.max() <= 3 and (d > 0).any()


@pytest.mark.parametrize("bname", BACKENDS)
def test_objective_guided_direction_and_weights(bname):
    bk = bk_of(bname)
    n = 30
    cost = -np.r_[np.full(10, 1.0), np.full(10, 2.0), np.full(10, 7.0)]   # maximise: cmin = -v < 0 -> up
    prob = from_arrays(sps.csr_matrix(np.ones((1, n))), [-np.inf], [1e9], np.zeros(n), np.ones(n), cost,
                       np.ones(n, bool))
    dp = to_device(prob, bk)
    k = 8000
    X0 = np.zeros((k, n), np.int8)
    C = bk.asarray(X0.copy())
    origin = mutate(bk, dp, C, _cfg({"objective_guided": 1.0}, genes=0.0, min_genes=1), *_rngs(bk, 9))
    Xh = bk.asnumpy(C)
    assert np.all(origin == ORIGIN_M_OBJ)
    assert np.all((Xh - X0) >= 0) and np.all((Xh != X0).sum(axis=1) == 1)
    share = np.array([Xh[:, g * 10:(g + 1) * 10].sum() for g in range(3)]) / k
    assert np.all(np.abs(share - np.array([1, 2, 7]) / 10) < 0.02)       # ∝ |cmin|


def test_objective_guided_minimisation_moves_down():
    bk = bk_of("numpy")
    prob = toy.set_cover(40, 25)
    dp = to_device(prob, bk)
    X0 = np.ones((300, prob.n), np.int8)
    C = X0.copy()
    mutate(bk, dp, C, _cfg({"objective_guided": 1.0}), *_rngs(bk, 10))
    assert np.all(C <= X0) and (C < X0).any()


@pytest.mark.parametrize("bname", BACKENDS)
@pytest.mark.parametrize("use_act", [False, True])
def test_violation_guided_reduces_row_violation(bname, use_act):
    bk = bk_of(bname)
    prob = toy.knapsack(30)                       # one <= row; all ones is overweight
    dp = to_device(prob, bk)
    k = 400
    X0 = np.ones((k, prob.n), np.int8)
    C = bk.asarray(X0.copy())
    act = dp.A @ bk.asarray(X0.astype(np.float64)).T if use_act else None
    origin = mutate(bk, dp, C, _cfg({"violation_guided": 1.0}), *_rngs(bk, 11), act=act)
    Xh = bk.asnumpy(C)
    assert np.all(origin == ORIGIN_M_VIOL)
    assert np.all(Xh <= X0) and np.all((Xh != X0).sum(axis=1) >= 1)       # weight row too high: genes go 1 -> 0


@pytest.mark.parametrize("bname", BACKENDS)
def test_violation_guided_uses_violated_rows_only(bname):
    bk = bk_of(bname)
    prob = toy.set_cover(40, 25, seed=3)          # covering rows >= 1
    dp = to_device(prob, bk)
    rs = np.random.default_rng(12)
    k = 300
    X0 = (rs.random((k, prob.n)) < 0.15).astype(np.int8)
    C = bk.asarray(X0.copy())
    stats: dict = {}
    origin = mutate(bk, dp, C, _cfg({"violation_guided": 1.0, "flip": 0.0}), *_rngs(bk, 13), stats=stats)
    Xh = bk.asnumpy(C)
    Ad = prob.A.toarray() != 0
    act = X0 @ prob.A.T.toarray()
    for c in range(k):
        moved = np.flatnonzero(Xh[c] != X0[c])
        viol_rows = act[c] < 1
        if not viol_rows.any():
            assert origin[c] != ORIGIN_M_VIOL                            # feasible child: another operator
            continue
        assert origin[c] == ORIGIN_M_VIOL
        in_viol = Ad[viol_rows].any(axis=0)
        assert np.all(in_viol[moved]) and np.all(Xh[c, moved] == 1)      # under-covered: genes go 0 -> 1
    feasible = int(np.sum(np.all(act >= 1, axis=1)))
    assert stats["violation_guided_no_violation"] == feasible


def test_violation_guided_feasible_children_redrawn():
    bk = bk_of("numpy")
    prob = toy.knapsack(30)
    dp = to_device(prob, bk)
    C = np.zeros((200, prob.n), np.int8)          # weight 0: feasible
    stats: dict = {}
    origin = mutate(bk, dp, C, _cfg({"violation_guided": 0.5, "flip": 0.5}), *_rngs(bk, 14), stats=stats)
    assert not np.any(origin & ORIGIN_M_VIOL) and np.all(origin == ORIGIN_M_FLIP)
    assert stats["violation_guided_no_violation"] > 0


@pytest.mark.parametrize("bname", BACKENDS)
def test_toward_lp_randomised_rounding(bname):
    bk = bk_of(bname)
    n = 50
    prob = general_int_problem(n, 0, 10)
    dp = to_device(prob, bk)
    lp = np.full(n, 3.3)
    k = 4000
    X0 = np.full((k, n), 8, np.int8)
    C = bk.asarray(X0.copy())
    origin = mutate(bk, dp, C, _cfg({"toward_lp": 1.0}, genes=0.0, min_genes=1), *_rngs(bk, 15),
                    lp_x=bk.asarray(lp) if bname == "cupy" else lp)
    Xh = bk.asnumpy(C)
    moved = Xh[Xh != X0]
    assert np.all(origin == ORIGIN_M_TOWARD_LP)
    assert set(np.unique(moved)) <= {3, 4} and abs(np.mean(moved == 4) - 0.3) < 0.03


def test_reset_continuous_window_and_bounds():
    bk = bk_of("numpy")
    n = 4
    prob = from_arrays(sps.csr_matrix((0, n)), np.empty(0), np.empty(0), np.array([-np.inf, 0, -np.inf, 2.0]),
                       np.array([np.inf, np.inf, 5.0, 3.0]), np.zeros(n), np.zeros(n, bool))
    dp = to_device(prob, bk)
    X0 = np.tile(np.array([100.0, 1.0, -40.0, 2.5]), (3000, 1))
    C = X0.copy()
    origin = mutate(bk, dp, C, _cfg({"reset": 1.0}, reset_window=2.0), *_rngs(bk, 16))
    assert np.all(origin == ORIGIN_M_RESET)
    assert np.all(np.abs(C[:, 0] - 100) <= 2) and np.all((C[:, 1] >= 0) & (C[:, 1] <= 3))
    # (-inf, 5] at -40: only the infinite side is replaced by the window -> [-42, 5]
    assert np.all((C[:, 2] >= -42) & (C[:, 2] <= 5)) and C[:, 2].max() > -38
    assert np.all((C[:, 3] >= 2) & (C[:, 3] <= 3))


@pytest.mark.parametrize("bname", BACKENDS)
def test_integers_normalised_even_without_mutation(bname):
    bk = bk_of(bname)
    prob = toy.mixed()                            # x0, x1 integer in [0, 5]; y0, y1 continuous
    dp = to_device(prob, bk)
    X0 = np.array([[0.9999999, 7.2, 1.25, -3.5], [-0.4, 2.0000001, 0.0, 9.75]])
    C = bk.asarray(X0.copy())
    origin = mutate(bk, dp, C, _cfg({"step": 1.0}, genes=0.0, min_genes=0), *_rngs(bk, 17))
    Xh = bk.asnumpy(C)
    assert np.all(origin == 0)
    assert np.array_equal(Xh[:, :2], np.array([[1.0, 5.0], [0.0, 2.0]]))
    assert np.array_equal(Xh[:, 2:], X0[:, 2:])
