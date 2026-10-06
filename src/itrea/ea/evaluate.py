"""The problem on the EA's device, and batched evaluation of a population.

`DeviceProblem` holds the constraint matrix and vectors on the backend (CuPy or NumPy) in the search precision; the
objective is ALWAYS float64 (a float32 key carries ~6e-8 relative error, far above the improvement tolerances).
`evaluate` returns, per individual, the key (lower is better), the total violation V and the feasibility flag of the
per-row test under `tol_mode` — the partner solver's rule (vault `10-design/I-TREA II algorithm.md`). Work is chunked
over the population so the (m x chunk) and (chunk x n) temporaries stay inside a memory budget. A vector with a
non-finite entry gets V = +inf and is never feasible.

Feasibility here is the SEARCH test, in `eval_dtype`. Anything posted to a solver is re-checked on the CPU in float64
(`Problem.check`), and reported results are re-checked by the independent exact checker.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np

from itrea.backend import Backend
from itrea.problem import Problem


@dataclass
class DeviceProblem:
    bk: Backend
    prob: Problem
    dtype: str
    A: object            # m x n CSR on the device
    rl: object
    ru: object
    lb: object
    ub: object
    cmin: object         # minimisation-form costs (sense * c), float64 always
    key_offset: float    # sense * offset
    row_norm: object
    scale_l: object      # max(1, |rl|) where finite, else 1
    scale_u: object
    scale_lb: object
    scale_ub: object
    isint: object        # bool mask
    ints: object         # integer indices
    bins: object
    gints: object
    conts: object
    # COO view of A (for the local search's per-non-zero work), sorted by row
    nz_row: object
    nz_col: object
    nz_val: object
    tol_mode: str = "scip"
    viol_norm_vec: object = None   # per-row divisor of the violation in V (l2 row norm or max(1,|rhs|))
    count_rows: bool = False       # viol_norm == "count": V counts violated rows

    @property
    def n(self) -> int:
        return self.prob.n

    @property
    def m(self) -> int:
        return self.prob.m

    @property
    def nnz(self) -> int:
        return self.prob.nnz


def to_device(prob: Problem, bk: Backend, dtype: str = "float64", tol_mode: str = "scip",
              viol_norm: str = "l2") -> DeviceProblem:
    xp = bk.xp
    dt = np.dtype(dtype)

    def dev(a, d=dt):
        return xp.asarray(np.asarray(a, dtype=d))

    def scale(v):
        v = np.asarray(v, dtype=np.float64)
        return np.where(np.isfinite(v), np.maximum(1.0, np.abs(v)), 1.0)

    coo = prob.A.tocoo()
    order = np.lexsort((coo.col, coo.row))
    rhs_scale = np.maximum(scale(prob.rl), scale(prob.ru))
    vn = prob.row_norm if viol_norm == "l2" else rhs_scale
    return DeviceProblem(
        bk=bk, prob=prob, dtype=dtype, A=bk.to_device_sparse(prob.A, dtype=dt),
        rl=dev(prob.rl), ru=dev(prob.ru), lb=dev(prob.lb), ub=dev(prob.ub), cmin=dev(prob.cmin, np.float64),
        key_offset=float(prob.sense * prob.offset), row_norm=dev(prob.row_norm),
        scale_l=dev(scale(prob.rl)), scale_u=dev(scale(prob.ru)), scale_lb=dev(scale(prob.lb)),
        scale_ub=dev(scale(prob.ub)), isint=xp.asarray(prob.isint), ints=xp.asarray(prob.ints),
        bins=xp.asarray(prob.bins), gints=xp.asarray(prob.gints), conts=xp.asarray(prob.conts),
        nz_row=xp.asarray(coo.row[order].astype(np.int32)), nz_col=xp.asarray(coo.col[order].astype(np.int32)),
        nz_val=dev(coo.data[order]), tol_mode=tol_mode, viol_norm_vec=dev(vn), count_rows=(viol_norm == "count"))


@dataclass
class Evaluation:
    key: object        # (P,) float64 — lower is better
    viol: object       # (P,) float64 — total violation V
    feasible: object   # (P,) bool — per-row search test
    max_rel: object    # (P,) float64 — largest relative row/bound violation (diagnostics)


def activities(dp: DeviceProblem, X):
    """A @ X^T for a block of individuals: (m, k)."""
    return dp.A @ X.T


def row_violation(dp: DeviceProblem, act):
    """Per-row absolute violation (m, k) of activities (m, k). Infinite sides contribute 0."""
    xp = dp.bk.xp
    lo = xp.maximum(dp.rl[:, None] - act, 0)
    hi = xp.maximum(act - dp.ru[:, None], 0)
    return xp.nan_to_num(lo, nan=0.0, posinf=0.0) + xp.nan_to_num(hi, nan=0.0, posinf=0.0)


def evaluate(dp: DeviceProblem, X, feas_tol: float = 1e-6, int_tol: float = 1e-6,
             chunk: int | None = None, mem_budget_bytes: int = 512 * 1024**2) -> Evaluation:
    """Evaluate every row of X (P x n, any numeric dtype). `chunk` individuals at a time; derived from
    `mem_budget_bytes` if None, counting both the (m, chunk) activity temporaries and the (chunk, n) bound ones."""
    xp = dp.bk.xp
    P = X.shape[0]
    item = np.dtype(dp.dtype).itemsize
    if chunk is None:
        per_ind = item * (6 * max(dp.m, 1) + 6 * dp.n) + 8 * dp.n
        chunk = max(1, int(mem_budget_bytes // per_ind))
    key = xp.empty(P, dtype=xp.float64)
    viol = xp.empty(P, dtype=xp.float64)
    feas = xp.empty(P, dtype=bool)
    mrel = xp.empty(P, dtype=xp.float64)
    ints = dp.ints
    dt = np.dtype(dp.dtype)
    quiet = contextlib.nullcontext() if dp.bk.is_gpu else np.errstate(invalid="ignore", divide="ignore")
    with quiet:
        _evaluate_chunks(dp, X, P, chunk, dt, ints, key, viol, feas, mrel, feas_tol, int_tol)
    return Evaluation(key=key, viol=viol, feasible=feas, max_rel=mrel)


def _evaluate_chunks(dp, X, P, chunk, dt, ints, key, viol, feas, mrel, feas_tol, int_tol):
    xp = dp.bk.xp
    for s in range(0, P, chunk):
        Xc = X[s:s + chunk].astype(dt, copy=False)
        k = Xc.shape[0]
        finite = xp.all(xp.isfinite(Xc), axis=1)
        act = activities(dp, Xc)                                  # (m, k)
        lo = xp.maximum(dp.rl[:, None] - act, 0)                  # -inf side -> 0
        hi = xp.maximum(act - dp.ru[:, None], 0)                  # +inf side -> 0
        lo = xp.where(xp.isnan(lo), 0, lo)
        hi = xp.where(xp.isnan(hi), 0, hi)
        if dp.m:
            if dp.tol_mode == "highs":
                rel_rows = xp.maximum(lo, hi).max(axis=0)
            elif dp.tol_mode == "scip":
                aa = xp.abs(act)
                rel_rows = xp.maximum(lo / xp.maximum(xp.maximum(aa, dp.scale_l[:, None]), 1.0),
                                      hi / xp.maximum(xp.maximum(aa, dp.scale_u[:, None]), 1.0)).max(axis=0)
            else:
                rel_rows = xp.maximum(lo / dp.scale_l[:, None], hi / dp.scale_u[:, None]).max(axis=0)
            if dp.count_rows:
                rv = ((lo + hi) > 0).sum(axis=0).astype(xp.float64)
            else:
                rv = ((lo + hi) / dp.viol_norm_vec[:, None]).sum(axis=0)
        else:
            rel_rows = rv = xp.zeros(k)
        del act, lo, hi
        blo = xp.maximum(dp.lb[None, :] - Xc, 0)
        bhi = xp.maximum(Xc - dp.ub[None, :], 0)
        blo = xp.where(xp.isnan(blo), 0, blo)
        bhi = xp.where(xp.isnan(bhi), 0, bhi)
        bv = (blo + bhi).sum(axis=1)
        if dp.tol_mode == "highs":
            rel_b = xp.maximum(blo, bhi).max(axis=1)
        else:
            rel_b = xp.maximum(blo / dp.scale_lb[None, :], bhi / dp.scale_ub[None, :]).max(axis=1)
        del blo, bhi
        if ints.size:
            xi = Xc[:, ints]
            fr = xp.abs(xi - xp.rint(xi))
            iv, imax = fr.sum(axis=1), fr.max(axis=1)
            del xi, fr
        else:
            iv = imax = xp.zeros(k, dtype=xp.float64)
        sl = slice(s, s + k)
        v = (rv + bv + iv).astype(xp.float64)
        viol[sl] = xp.where(finite, v, xp.inf)
        mr = xp.maximum(rel_rows, rel_b).astype(xp.float64)
        mrel[sl] = xp.where(finite, mr, xp.inf)
        feas[sl] = (mr <= feas_tol) & (imax <= int_tol) & finite
        kk = X[s:s + chunk].astype(xp.float64, copy=False) @ dp.cmin
        key[sl] = xp.where(finite, kk + dp.key_offset, xp.nan)
