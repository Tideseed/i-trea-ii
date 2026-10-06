"""Small deterministic MIP models for tests and smoke runs (seconds, megabytes).

Each generator returns an `itrea.problem.Problem`; `write_mps` writes it with HiGHS so the solver processes can read
the same model from a file. Known optimal objectives are given where they are easy to state, so tests can assert
"the EA/solver found the optimum" rather than only "it ran".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sps

from itrea.problem import Problem, from_arrays


def knapsack(n: int = 30, seed: int = 0) -> Problem:
    """max v·x s.t. w·x <= W, x binary. Optimum computed by DP (stored in `toy_optimum`)."""
    rs = np.random.default_rng(seed)
    w = rs.integers(5, 30, size=n)
    v = w + rs.integers(0, 10, size=n)
    W = int(w.sum() // 2)
    p = from_arrays(sps.csr_matrix(w.reshape(1, -1)), [-np.inf], [W], np.zeros(n), np.ones(n), v, np.ones(n, bool),
                    sense=-1, names=[f"k{j}" for j in range(n)], path=f"<toy:knapsack{n}s{seed}>")
    best = np.zeros(W + 1)
    for j in range(n):
        best[w[j]:] = np.maximum(best[w[j]:], best[:W + 1 - w[j]] + v[j])
    p.toy_optimum = float(best[W])  # type: ignore[attr-defined]
    return p


def set_cover(n: int = 40, m: int = 25, density: float = 0.15, seed: int = 0) -> Problem:
    """min c·x s.t. every row covered at least once, x binary. (Optimum not stated.)"""
    rs = np.random.default_rng(seed)
    A = (rs.random((m, n)) < density).astype(float)
    for i in range(m):                      # every row coverable
        if A[i].sum() == 0:
            A[i, rs.integers(n)] = 1.0
    c = rs.integers(1, 10, size=n).astype(float)
    return from_arrays(sps.csr_matrix(A), np.ones(m), np.full(m, np.inf), np.zeros(n), np.ones(n), c,
                       np.ones(n, bool), sense=1, names=[f"s{j}" for j in range(n)], path=f"<toy:setcover{n}x{m}s{seed}>")


def mixed(seed: int = 0) -> Problem:
    """A small mixed problem with general integers, continuous variables, an equality row and a free variable.

    min  x0 + 2 x1 + 3 y0 - y1
    s.t. x0 + x1 + y0          >= 4.5
         x0 - x1      + y1      = 1
         2 x0 + y0 - y1        <= 8
         x0, x1 integer in [0, 5]; y0 >= 0 continuous; y1 free continuous
    """
    A = sps.csr_matrix(np.array([[1, 1, 1, 0], [1, -1, 0, 1], [2, 0, 1, -1]], dtype=float))
    rl = np.array([4.5, 1.0, -np.inf])
    ru = np.array([np.inf, 1.0, 8.0])
    lb = np.array([0.0, 0.0, 0.0, -np.inf])
    ub = np.array([5.0, 5.0, np.inf, np.inf])
    c = np.array([1.0, 2.0, 3.0, -1.0])
    return from_arrays(A, rl, ru, lb, ub, c, np.array([True, True, False, False]), sense=1,
                       names=["x0", "x1", "y0", "y1"], path=f"<toy:mixed s{seed}>")


def write_mps(prob: Problem, path: str | Path) -> Path:
    """Write the problem as MPS with HiGHS (variable names preserved)."""
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = prob.n, prob.m
    inf = highspy.kHighsInf

    def hv(v):
        v = np.asarray(v, dtype=np.float64).copy()
        v[v == np.inf] = inf
        v[v == -np.inf] = -inf
        return v

    lp.col_cost_ = np.asarray(prob.c, dtype=np.float64)
    lp.offset_ = float(prob.offset)
    lp.col_lower_, lp.col_upper_ = hv(prob.lb), hv(prob.ub)
    lp.row_lower_, lp.row_upper_ = hv(prob.rl), hv(prob.ru)
    csc = prob.A.tocsc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = csc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = csc.indices.astype(np.int32)
    lp.a_matrix_.value_ = csc.data.astype(np.float64)
    lp.sense_ = highspy.ObjSense.kMaximize if prob.sense == -1 else highspy.ObjSense.kMinimize
    lp.integrality_ = [highspy.HighsVarType.kInteger if t else highspy.HighsVarType.kContinuous for t in prob.isint]
    lp.col_names_ = list(prob.names)
    lp.row_names_ = [f"r{i}" for i in range(prob.m)]
    h.passModel(lp)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h.writeModel(str(path))
    return path
