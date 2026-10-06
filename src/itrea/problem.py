"""A MIP instance as arrays: read once with HiGHS (CPU), copied to the EA's device.

Form: min  key(x) = sense * (c·x + offset)  s.t.  rl <= A x <= ru,  lb <= x <= ub,  x_j integer for j in `ints`.
`sense` is +1 for minimisation and -1 for maximisation, so a lower key is always better; objective values exchanged
with the solvers are in the user's sense (`Problem.objective`).

Canonical variable order = HiGHS's column order of the MPS file. Solver adapters map their own order onto it by
variable name (`Problem.names`).

`Problem.check` is the float64, per-row feasibility test every candidate passes before it is posted to a solver; its
`tol_mode` must match the partner ("scip": |act - side| / max(|act|, |side|, 1); "highs": absolute; HiGHS silently
drops a user solution outside its absolute tolerance — measured 2026-10-05). It is NOT the independent checker used for
reported results (`itrea.analysis.check_solutions`, exact arithmetic, another library).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sps

from itrea.paths import resolve


@dataclass
class CheckResult:
    feasible: bool
    max_row_viol: float      # largest row violation under tol_mode
    max_bound_viol: float    # largest bound violation (relative to max(1, |bound|); absolute for "highs")
    max_int_viol: float      # largest distance to the nearest integer over integer variables
    objective: float         # user-sense objective


@dataclass
class Problem:
    path: str
    names: list[str]
    A: sps.csr_matrix        # m x n
    rl: np.ndarray
    ru: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    c: np.ndarray            # user-sense objective coefficients
    offset: float
    sense: int               # +1 minimise, -1 maximise
    isint: np.ndarray        # bool, n
    row_norm: np.ndarray = field(init=False)
    ints: np.ndarray = field(init=False)
    conts: np.ndarray = field(init=False)
    bins: np.ndarray = field(init=False)
    gints: np.ndarray = field(init=False)   # general (non-binary) integers

    def __post_init__(self):
        self.A = sps.csr_matrix(self.A, dtype=np.float64)
        self.A.sum_duplicates()
        self.A.eliminate_zeros()
        rn = np.sqrt(np.asarray(self.A.multiply(self.A).sum(axis=1)).ravel())
        self.row_norm = np.where(rn > 0, rn, 1.0)
        self.ints = np.flatnonzero(self.isint)
        self.conts = np.flatnonzero(~self.isint)
        isbin = self.isint & (np.ceil(self.lb - 1e-9) >= 0) & (np.floor(self.ub + 1e-9) <= 1)
        self.bins = np.flatnonzero(isbin)
        self.gints = np.flatnonzero(self.isint & ~isbin)

    # ------------------------------------------------------------------ sizes
    @property
    def n(self) -> int:
        return self.A.shape[1]

    @property
    def m(self) -> int:
        return self.A.shape[0]

    @property
    def nnz(self) -> int:
        return int(self.A.nnz)

    @property
    def cmin(self) -> np.ndarray:
        """Minimisation-form costs: key(x) = cmin @ x + sense * offset."""
        return self.sense * self.c

    # -------------------------------------------------------------- objective
    def objective(self, x: np.ndarray) -> float:
        return float(self.c @ x + self.offset)

    def key(self, x: np.ndarray) -> float:
        return self.sense * self.objective(x)

    def key_of_objective(self, obj: float) -> float:
        return self.sense * float(obj)

    def objective_of_key(self, key: float) -> float:
        return self.sense * float(key)

    # ------------------------------------------------------------------ check
    def check(self, x: np.ndarray, feas_tol: float = 1e-6, int_tol: float = 1e-6, tol_mode: str = "scip"
              ) -> CheckResult:
        """Per-row float64 feasibility test. tol_mode: "scip" — |act - side| / max(|act|, |side|, 1) (SCIP's linear
        constraint check); "highs" — absolute residual (HiGHS' mip_feasibility_tolerance); "rhs" — relative to
        max(1, |side|). Bounds: relative to max(1, |bound|) except "highs" (absolute). Integers within int_tol."""
        x = np.asarray(x, dtype=np.float64)
        act = self.A @ x

        def rel(side, val, lower, act_scale):
            fin = np.isfinite(side)
            out = np.zeros(val.shape, dtype=np.float64)
            d = (side[fin] - val[fin]) if lower else (val[fin] - side[fin])
            if tol_mode == "highs":
                sc = 1.0
            elif tol_mode == "scip" and act_scale:
                sc = np.maximum(np.maximum(np.abs(val[fin]), np.abs(side[fin])), 1.0)
            else:
                sc = np.maximum(1.0, np.abs(side[fin]))
            out[fin] = d / sc
            return out

        rv = float(max(0.0, rel(self.rl, act, True, True).max(initial=0.0),
                       rel(self.ru, act, False, True).max(initial=0.0)))
        bv = float(max(0.0, rel(self.lb, x, True, False).max(initial=0.0),
                       rel(self.ub, x, False, False).max(initial=0.0)))
        xi = x[self.ints]
        iv = float(np.abs(xi - np.round(xi)).max(initial=0.0))
        ok = bool(np.all(np.isfinite(x)) and rv <= feas_tol and bv <= feas_tol and iv <= int_tol)
        return CheckResult(ok, rv, bv, iv, self.objective(x))

    def round_integers(self, x: np.ndarray) -> np.ndarray:
        """Copy of x with integer coordinates rounded and every coordinate clipped to its bounds (before checks/posts)."""
        y = np.array(x, dtype=np.float64, copy=True)
        y[self.ints] = np.round(y[self.ints])
        return np.clip(y, self.lb, self.ub)

    def violation(self, x: np.ndarray) -> float:
        """Total violation V(x) (the population's infeasibility measure, CPU float64 reference implementation)."""
        x = np.asarray(x, dtype=np.float64)
        act = self.A @ x
        r = np.maximum(np.nan_to_num(self.rl - act, nan=0.0, neginf=0.0), 0) + \
            np.maximum(np.nan_to_num(act - self.ru, nan=0.0, neginf=0.0), 0)
        b = np.maximum(np.nan_to_num(self.lb - x, neginf=0.0), 0) + np.maximum(np.nan_to_num(x - self.ub, neginf=0.0), 0)
        xi = x[self.ints]
        return float((r / self.row_norm).sum() + b.sum() + np.abs(xi - np.round(xi)).sum())

    # ------------------------------------------------------------------- misc
    def name_index(self) -> dict[str, int]:
        return {nm: j for j, nm in enumerate(self.names)}

    def summary(self) -> dict:
        return dict(path=self.path, n=self.n, m=self.m, nnz=self.nnz, ints=len(self.ints), bins=len(self.bins),
                    gints=len(self.gints), conts=len(self.conts), sense=self.sense,
                    unbounded_int=int(np.sum(~np.isfinite(self.lb[self.ints]) | ~np.isfinite(self.ub[self.ints]))))


def load(path: str | Path) -> Problem:
    """Read an (optionally gzipped) MPS/LP file with HiGHS."""
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    st = h.readModel(str(resolve(path)))
    if st == highspy.HighsStatus.kError:
        raise IOError(f"HiGHS could not read {path}")
    lp = h.getLp()
    n, m = lp.num_col_, lp.num_row_
    a = lp.a_matrix_
    start, index, value = np.asarray(a.start_), np.asarray(a.index_), np.asarray(a.value_, dtype=np.float64)
    if a.format_ == highspy.MatrixFormat.kColwise:
        A = sps.csc_matrix((value, index, start), shape=(m, n)).tocsr()
    else:
        A = sps.csr_matrix((value, index, start), shape=(m, n))
    vt = highspy.HighsVarType
    integ = list(lp.integrality_) if len(lp.integrality_) else [vt.kContinuous] * n
    if any(t in (vt.kSemiContinuous, vt.kSemiInteger) for t in integ):
        raise NotImplementedError(f"{path}: semi-continuous / semi-integer variables are not supported")
    isint = np.array([t in (vt.kInteger, vt.kImplicitInteger) for t in integ], dtype=bool)

    def fin(v):
        v = np.asarray(v, dtype=np.float64)
        v[v >= 1e30] = np.inf
        v[v <= -1e30] = -np.inf
        return v

    names = list(lp.col_names_) if len(lp.col_names_) == n else [f"x{j}" for j in range(n)]
    sense = -1 if lp.sense_ == highspy.ObjSense.kMaximize else 1
    return Problem(path=str(path), names=names, A=A, rl=fin(lp.row_lower_), ru=fin(lp.row_upper_),
                   lb=fin(lp.col_lower_), ub=fin(lp.col_upper_), c=np.asarray(lp.col_cost_, dtype=np.float64),
                   offset=float(lp.offset_), sense=sense, isint=isint)


def from_arrays(A, rl, ru, lb, ub, c, isint, sense=1, offset=0.0, names=None, path="<arrays>") -> Problem:
    """Build a Problem directly (tests, toy models)."""
    A = sps.csr_matrix(A, dtype=np.float64)
    n = A.shape[1]
    return Problem(path=path, names=list(names) if names is not None else [f"x{j}" for j in range(n)], A=A,
                   rl=np.asarray(rl, float), ru=np.asarray(ru, float), lb=np.asarray(lb, float),
                   ub=np.asarray(ub, float), c=np.asarray(c, float), offset=float(offset), sense=int(sense),
                   isint=np.asarray(isint, bool))
