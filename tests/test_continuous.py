"""Closed-form decoding of singleton continuous columns (`itrea.ea.continuous.SingletonDecoder`).

The reference is an LP per row: with the non-singleton activity fixed, (1) minimise the row's violation over the
singleton columns, then (2) minimise their cost subject to that violation (scipy.optimize.linprog, HiGHS).
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps
from scipy.optimize import linprog

from itrea.backend import get_backend
from itrea.benchmarks import toy
from itrea.ea.continuous import SingletonDecoder
from itrea.ea.evaluate import to_device
from itrea.problem import from_arrays

BK = get_backend("numpy")


def slack_model(m: int = 12, n_int: int = 10, seed: int = 0):
    """Rows over a few integers plus 1-3 singleton continuous columns each (slack-like, mixed bounds and costs)."""
    rs = np.random.default_rng(seed)
    rows, cols, vals = [], [], []
    lb, ub, c, isint = [], [], [], []
    for j in range(n_int):
        lb.append(0.0), ub.append(5.0), c.append(float(rs.integers(-3, 4))), isint.append(True)
    rl, ru = [], []
    nxt = n_int
    for i in range(m):
        for j in rs.choice(n_int, size=3, replace=False):
            rows.append(i), cols.append(int(j)), vals.append(float(rs.integers(-4, 5) or 1))
        kind = i % 4
        b = float(rs.integers(-5, 15))
        rl.append(b if kind in (0, 1, 3) else -np.inf)
        ru.append(b if kind == 0 else (b + 6 if kind == 3 else (np.inf if kind == 1 else b)))
        for _ in range(int(rs.integers(1, 4))):
            a = float(rs.choice([-2.0, -1.0, 0.5, 1.0, 3.0]))
            shape = int(rs.integers(0, 5))
            if shape == 0:
                lo, hi, cost = 0.0, np.inf, float(rs.integers(0, 4))          # slack, non-negative cost
            elif shape == 1:
                lo, hi, cost = 0.0, float(rs.integers(1, 6)), float(rs.integers(-3, 4))
            elif shape == 2:
                lo, hi, cost = -np.inf, 0.0, -float(rs.integers(0, 3))        # cost <= 0 on a column going down
            elif shape == 3:
                lo, hi, cost = -2.0, 2.0, 0.0                                  # zero cost, bounded
            else:
                lo, hi, cost = float(rs.integers(-4, 0)), float(rs.integers(0, 4)), float(rs.normal())
            rows.append(i), cols.append(nxt), vals.append(a)
            lb.append(lo), ub.append(hi), c.append(cost), isint.append(False)
            nxt += 1
    A = sps.csr_matrix((vals, (rows, cols)), shape=(m, nxt))
    return from_arrays(A, rl, ru, lb, ub, c, isint, sense=1, path=f"<test:slack{seed}>")


def row_lp_reference(prob, i: int, x: np.ndarray, S: np.ndarray):
    """(min violation, min cost at that violation) of row i over its singleton columns S, others fixed at x."""
    A = prob.A.tocsr()
    row = A.getrow(i).toarray().ravel()
    ap = float(row @ x - row[S] @ x[S])
    a = row[S]
    cmin = prob.cmin[S]
    bnds = [(None if not np.isfinite(lo) else lo, None if not np.isfinite(hi) else hi)
            for lo, hi in zip(prob.lb[S], prob.ub[S])]
    k = S.size
    rl, ru = prob.rl[i] - ap, prob.ru[i] - ap
    # variables: x_S, e_lo, e_hi >= 0
    A_ub, b_ub = [], []
    if np.isfinite(rl):
        A_ub.append(np.concatenate([-a, [-1.0, 0.0]])), b_ub.append(-rl)
    if np.isfinite(ru):
        A_ub.append(np.concatenate([a, [0.0, -1.0]])), b_ub.append(ru)
    A_ub, b_ub = np.array(A_ub), np.array(b_ub)
    b2 = bnds + [(0, None), (0, None)]
    r1 = linprog(np.concatenate([np.zeros(k), [1.0, 1.0]]), A_ub=A_ub, b_ub=b_ub, bounds=b2, method="highs")
    assert r1.status == 0
    v = r1.fun
    A2 = np.vstack([A_ub, np.concatenate([np.zeros(k), [1.0, 1.0]])])
    r2 = linprog(np.concatenate([cmin, [0.0, 0.0]]), A_ub=A2, b_ub=np.append(b_ub, v + 1e-9), bounds=b2,
                 method="highs")
    if r2.status == 3:          # unbounded in cost: the decoder skips such rows
        return v, None
    assert r2.status == 0
    return v, r2.fun


def row_violation(prob, i, x):
    act = float((prob.A.getrow(i) @ x)[0])
    return max(prob.rl[i] - act, 0.0) + max(act - prob.ru[i], 0.0)


def test_no_singletons_on_toy_mixed():
    p = toy.mixed()                     # y0 and y1 each appear in two rows
    dec = SingletonDecoder(to_device(p, BK))
    assert len(dec) == 0 and dec.columns.size == 0
    X = np.ones((3, p.n))
    X0 = X.copy()
    assert dec.apply(X)["columns"] == 0
    assert np.array_equal(X, X0)


def test_detection():
    p = slack_model(seed=1)
    dec = SingletonDecoder(to_device(p, BK))
    C = p.A.tocsc()
    deg = np.diff(C.indptr)
    expect = np.flatnonzero(~p.isint & (deg == 1) & (p.lb < p.ub))
    assert np.array_equal(np.sort(np.concatenate([dec.columns, _skipped_cols(p, dec)])), expect)


def _skipped_cols(p, dec):
    C = p.A.tocsc()
    deg = np.diff(C.indptr)
    allc = np.flatnonzero(~p.isint & (deg == 1) & (p.lb < p.ub))
    return np.setdiff1d(allc, dec.columns)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_decoding_matches_row_lp(seed):
    p = slack_model(seed=seed)
    dp = to_device(p, BK)
    dec = SingletonDecoder(dp)
    rs = np.random.default_rng(100 + seed)
    k = 12
    X = np.zeros((k, p.n))
    X[:, p.ints] = rs.integers(0, 6, size=(k, p.ints.size))
    X[:, p.conts] = rs.normal(0, 3, size=(k, p.conts.size))          # arbitrary (even out-of-bound) start
    X0 = X.copy()
    dec.apply(X)
    assert np.array_equal(X[:, p.ints], X0[:, p.ints])               # only singleton columns change
    assert np.array_equal(X[:, _skipped_cols(p, dec)], X0[:, _skipped_cols(p, dec)])
    C = p.A.tocsc()
    row_of = {int(j): int(C.indices[C.indptr[j]]) for j in dec.columns}
    lbS, ubS = p.lb[dec.columns], p.ub[dec.columns]
    assert np.all(X[:, dec.columns] >= lbS - 1e-12) and np.all(X[:, dec.columns] <= ubS + 1e-12)
    for i in dec.rows:
        S = np.array(sorted(j for j, r in row_of.items() if r == i))
        for b in range(k):
            v_ref, c_ref = row_lp_reference(p, int(i), X0[b], S)
            v = row_violation(p, int(i), X[b])
            assert v <= v_ref + 1e-7 * max(1.0, abs(v_ref)), (i, b, v, v_ref)
            if c_ref is not None:
                cost = float(p.cmin[S] @ X[b, S])
                assert cost <= c_ref + 1e-7 * max(1.0, abs(c_ref)), (i, b, cost, c_ref)


def test_markshare_style_pair_of_slacks():
    # 3 x0 + 5 x1 + s_plus - s_minus = 7, both slacks >= 0 with cost 1 (markshare's form): |residual| goes to
    # exactly one slack and the row is satisfied exactly
    A = sps.csr_matrix(np.array([[3.0, 5.0, 1.0, -1.0]]))
    p = from_arrays(A, [7.0], [7.0], [0, 0, 0, 0], [1, 1, np.inf, np.inf], [0, 0, 1, 1], [True, True, False, False])
    dec = SingletonDecoder(to_device(p, BK))
    assert sorted(dec.columns.tolist()) == [2, 3]
    X = np.array([[0, 0, 9, 9], [1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 4, 0]], dtype=float)
    dec.apply(X)
    assert np.allclose(X[:, 2:], [[7, 0], [4, 0], [0, 1], [2, 0]])
    for x in X:
        assert p.check(x).feasible


def test_infinite_bounds_zero_cost_and_least_violation():
    # x0 + z <= 3 with z free and cost 0 -> z = min(0, 3 - x0); 3 y >= 4 + x0 with y in [0,1] -> y = 1 (least violation)
    A = sps.csr_matrix(np.array([[1.0, 1.0, 0.0], [-1.0, 0.0, 3.0]]))
    p = from_arrays(A, [-np.inf, 4.0], [3.0, np.inf], [0, -np.inf, 0], [10, np.inf, 1], [0, 0, 1.0],
                    [True, False, False])
    dec = SingletonDecoder(to_device(p, BK))
    X = np.array([[0, 5, 0], [9, 5, 0], [1, -3, 0.2]], dtype=float)
    dec.apply(X)
    assert np.allclose(X[:, 1], [0.0, -6.0, 0.0])
    assert np.allclose(X[:, 2], [1.0, 1.0, 1.0])


def test_cost_unbounded_row_is_skipped():
    # x1 + w1 + 2 w2 <= 5 with w1 <= 2 (cost 1) and w2 >= 0 (cost -1): w2 up / w1 down is an unbounded ray
    A = sps.csr_matrix(np.array([[1.0, 1.0, 2.0]]))
    p = from_arrays(A, [-np.inf], [5.0], [0, -np.inf, 0], [10, 2, np.inf], [0, 1.0, -1.0], [True, False, False])
    dec = SingletonDecoder(to_device(p, BK))
    assert len(dec) == 0 and dec.skipped_rows == 1
    X = np.array([[1.0, 0.5, 0.5]])
    dec.apply(X)
    assert np.array_equal(X, [[1.0, 0.5, 0.5]])


def test_idempotent_and_chunked():
    p = slack_model(seed=2)
    dp = to_device(p, BK)
    rs = np.random.default_rng(0)
    X = rs.normal(0, 2, size=(40, p.n))
    X[:, p.ints] = rs.integers(0, 6, size=(40, p.ints.size))
    Y = X.copy()
    SingletonDecoder(dp).apply(X)
    small = SingletonDecoder(dp, mem_budget_bytes=1)              # one individual per pass
    out = small.apply(Y)
    assert out["chunk"] == 1
    assert np.array_equal(X, Y)
    Z = X.copy()
    SingletonDecoder(dp).apply(Z)
    assert np.allclose(Z, X, rtol=0, atol=1e-12)


def test_integer_dtype_refused():
    p = slack_model(seed=0)
    dec = SingletonDecoder(to_device(p, BK))
    with pytest.raises(TypeError):
        dec.apply(np.zeros((2, p.n), dtype=np.int32))


@pytest.mark.gpu
def test_gpu_matches_cpu():
    cp = pytest.importorskip("cupy")
    try:
        bk = get_backend("cupy", gpu_mem_gb=1.0)
    except Exception as e:  # pragma: no cover - no device
        pytest.skip(f"no CuPy device: {e}")
    p = slack_model(seed=3)
    rs = np.random.default_rng(1)
    X = rs.normal(0, 2, size=(32, p.n))
    X[:, p.ints] = rs.integers(0, 6, size=(32, p.ints.size))
    Xg = cp.asarray(X)
    SingletonDecoder(to_device(p, BK)).apply(X)
    SingletonDecoder(to_device(p, bk)).apply(Xg)
    assert np.allclose(cp.asnumpy(Xg), X, rtol=0, atol=1e-12)
