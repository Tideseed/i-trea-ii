"""Exact aggregation of two-variable equality rows — a presolve-lite for the EA's genome (development, 2026-10-06).

Why. SCIP and cuOpt run their heuristics on a presolved model; the EA sees the raw one. On equality-chained instances
(radiationm18-12-05: cont = int, cont = bin, b1 + b2 = 1; rail01: 20,788 rows x_a = x_b) a single-variable feasibility
jump cannot keep a chain satisfied — moving one member breaks the row that ties it to the next. Substituting the chain
away makes the whole chain ONE gene, so a jump moves all its members at once.

What. Every equality row with exactly two non-zeros of equal magnitude, a·x_j ± a·x_k = r, ties x_k = ±x_j + b. A
weighted union-find composes the ties, so every variable v becomes x_v = s_v · y_root(v) + o_v with s_v = ±1. A tie is
used only if the integrality of every member is kept: the offset between the two components must be integral unless
both components are all-continuous (then x_member = ±y + o with integral o, so an integer member forces an integral y:
the root is integer iff any member is). The rows used for ties are dropped (satisfied identically); a tie that closes a
cycle is NOT used — its row stays and becomes a row on the root (a fixing, or 0 = r).

The reduced problem: columns = the roots; bounds = the intersection of every member's bounds mapped through its
(s, o), rounded inward for integer roots; A' = A[kept] · T (T[v, root(v)] = s_v), row sides shifted by A[kept] · o;
c' = Tᵀ c, objective offset + c · o (user sense, so the reduced key equals the original key exactly). Rows emptied by
cancellation are dropped after checking 0 ∈ [rl', ru']. Empty bound intervals or violated empty rows mean the original
problem is infeasible — `aggregate` then returns None and the caller runs on the original.

Maps: `lift(Y)` = s · Y[:, rep] + o (reduced -> original, exact); `project(X)` = X[:, root] (original -> reduced; for a
vector that satisfies the dropped rows, lift(project(x)) == x). Both accept one vector or a (k, n) batch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps

from itrea.problem import Problem, from_arrays

INT_TOL = 1e-9      # an offset closer than this to an integer counts as integral
EQ_TOL = 1e-12      # relative tolerance for "rl == ru" and for "|a_j| == |a_k|"


@dataclass
class Aggregation:
    original: Problem
    reduced: Problem
    rep: np.ndarray          # (n,) reduced column of every original variable
    sign: np.ndarray         # (n,) +-1.0
    offset: np.ndarray       # (n,) x_v = sign_v * y_rep(v) + offset_v
    root: np.ndarray         # (n_red,) original index of each reduced column's root (sign +1, offset 0)
    rows_kept: np.ndarray    # original row indices that remain (in the reduced row order)
    ties: int                # rows used as ties (dropped)
    cycles: int              # tie rows that closed a cycle (kept as ordinary rows)
    refused: int             # two-variable equalities not used (integrality would be lost)
    emptied: int             # kept rows that became empty and were dropped

    def lift(self, Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y, dtype=np.float64)
        return self.sign * Y[..., self.rep] + self.offset

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X[..., self.root].copy()

    def summary(self) -> dict:
        return dict(n=self.original.n, n_reduced=self.reduced.n, m=self.original.m, m_reduced=self.reduced.m,
                    ties=self.ties, cycles=self.cycles, refused=self.refused, emptied=self.emptied)


def _integral(v: float) -> bool:
    return abs(v - round(v)) <= INT_TOL * max(1.0, abs(v))


def aggregate(p: Problem) -> Aggregation | None:
    """The aggregation of `p` (module docstring), or None if nothing can be aggregated or `p` is found infeasible."""
    A = p.A.tocsr()
    n, m = p.n, p.m
    rl, ru = np.asarray(p.rl, float), np.asarray(p.ru, float)
    isint = np.asarray(p.isint, bool)
    nnz = np.diff(A.indptr)
    eq = np.isfinite(rl) & (np.abs(ru - rl) <= EQ_TOL * np.maximum(1.0, np.abs(rl)))

    parent = np.arange(n)
    s_par = np.ones(n)            # x_v = s_par[v] * x_parent + o_par[v]
    o_par = np.zeros(n)
    size = np.ones(n, np.int64)
    has_int = isint.copy()        # per root: does the component hold an integer variable

    def find(v: int):
        """(root, s, o) with x_v = s * x_root + o; compresses the path."""
        path = []
        while parent[v] != v:
            path.append(v)
            v = parent[v]
        root = v
        # compose from the node just below the root down to the start
        s_acc, o_acc = 1.0, 0.0
        for u in reversed(path):
            # x_u = s_par[u] * x_parent(u) + o_par[u], and x_parent(u) = s_acc * x_root + o_acc
            s_acc, o_acc = s_par[u] * s_acc, s_par[u] * o_acc + o_par[u]
            parent[u], s_par[u], o_par[u] = root, s_acc, o_acc
        return (root, s_par[path[0]], o_par[path[0]]) if path else (root, 1.0, 0.0)

    used = np.zeros(m, bool)
    cycles = refused = 0
    for i in np.flatnonzero(eq & (nnz == 2)):
        s0 = A.indptr[i]
        j, k = int(A.indices[s0]), int(A.indices[s0 + 1])
        aj, ak = float(A.data[s0]), float(A.data[s0 + 1])
        if abs(abs(aj) - abs(ak)) > EQ_TOL * max(abs(aj), abs(ak)) or aj == 0.0:
            refused += 1
            continue
        r, b = -aj / ak, float(rl[i]) / ak             # x_k = r * x_j + b, r = +-1
        rj, sj, oj = find(j)
        rk, sk, ok = find(k)
        if rj == rk:
            cycles += 1
            continue
        # y_rk = S * y_rj + O
        S = r * sj * sk
        O = (r * oj + b - ok) * sk
        if has_int[rj] or has_int[rk]:
            if not _integral(O):
                refused += 1
                continue
            O = float(round(O))
        # attach the smaller component under the larger one (inverse relation if needed)
        if size[rk] > size[rj]:
            parent[rj], s_par[rj], o_par[rj] = rk, S, -O * S       # y_rj = S * y_rk - S * O  (S = +-1)
            size[rk] += size[rj]
            has_int[rk] |= has_int[rj]
        else:
            parent[rk], s_par[rk], o_par[rk] = rj, S, O
            size[rj] += size[rk]
            has_int[rj] |= has_int[rk]
        used[i] = True
    ties = int(used.sum())
    if ties == 0:
        return None

    roots = np.empty(n, np.int64)
    sign = np.empty(n)
    off = np.empty(n)
    for v in range(n):
        rt, s, o = find(v)
        roots[v], sign[v], off[v] = rt, s, o
    root = np.unique(roots)                              # sorted original indices: deterministic column order
    col_of_root = np.full(n, -1, np.int64)
    col_of_root[root] = np.arange(root.size)
    rep = col_of_root[roots]
    nr = root.size

    # bounds of y: every member's bounds mapped through (s, o), intersected
    lb, ub = np.asarray(p.lb, float), np.asarray(p.ub, float)
    lo_m = np.where(sign > 0, lb - off, off - ub)
    hi_m = np.where(sign > 0, ub - off, off - lb)
    ylb = np.full(nr, -np.inf)
    yub = np.full(nr, np.inf)
    np.maximum.at(ylb, rep, lo_m)
    np.minimum.at(yub, rep, hi_m)
    yint = np.zeros(nr, bool)
    np.logical_or.at(yint, rep, isint)
    fl = np.isfinite(ylb) & yint
    fu = np.isfinite(yub) & yint
    ylb[fl] = np.ceil(ylb[fl] - INT_TOL * np.maximum(1.0, np.abs(ylb[fl])))
    yub[fu] = np.floor(yub[fu] + INT_TOL * np.maximum(1.0, np.abs(yub[fu])))
    if np.any(ylb > yub + 1e-9 * np.maximum(1.0, np.abs(ylb))):
        return None                                      # an empty interval: the original is infeasible

    # rows: the ties go; everything else is mapped through T
    keep = np.flatnonzero(~used)
    T = sps.csr_matrix((sign, (np.arange(n), rep)), shape=(n, nr))
    Ak = A[keep]
    Ar = (Ak @ T).tocsr()
    Ar.sum_duplicates()
    if Ar.nnz:
        scale = np.repeat(np.asarray(abs(Ak).max(axis=1).toarray()).ravel(), np.diff(Ar.indptr))
        Ar.data[np.abs(Ar.data) <= EQ_TOL * np.maximum(1.0, scale)] = 0.0
    Ar.eliminate_zeros()
    shift = Ak @ off
    rlr = np.where(np.isfinite(rl[keep]), rl[keep] - shift, -np.inf)
    rur = np.where(np.isfinite(ru[keep]), ru[keep] - shift, np.inf)
    empty = np.diff(Ar.indptr) == 0
    if empty.any():
        tol = 1e-9 * np.maximum(1.0, np.abs(np.where(np.isfinite(rlr), rlr, 0.0)))
        if np.any(empty & ((rlr > tol) | (rur < -tol))):
            return None                                  # 0 outside [rl', ru']: infeasible
        nonempty = ~empty
        Ar, rlr, rur, keep = Ar[nonempty], rlr[nonempty], rur[nonempty], keep[nonempty]
    c = np.asarray(p.c, float)
    cr = np.asarray(T.T @ c).ravel()
    red = from_arrays(Ar, rlr, rur, ylb, yub, cr, yint, sense=p.sense, offset=float(p.offset + c @ off),
                      names=[p.names[int(r)] for r in root], path=f"{p.path}#agg")
    return Aggregation(original=p, reduced=red, rep=rep, sign=sign, offset=off, root=root, rows_kept=keep,
                       ties=ties, cycles=cycles, refused=refused, emptied=int(empty.sum()))
