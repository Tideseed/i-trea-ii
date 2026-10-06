"""Closed-form decoding of singleton continuous columns (`repair.continuous: auto`).

Why: slack and penalty columns — a continuous variable that appears in exactly one row — are fully determined by the
rest of the solution: given the activity a' of the row's other variables, the best value of the row's singleton
columns is the solution of a one-row LP. Searching for them as genes wastes moves and, on equality rows (markshare's
s+/s- slacks, unit-commitment load shedding), almost never lands exactly on the row (design critique 2026-10-05).
Decoding them in closed form before every evaluation removes them from the search: an equality row with slack
columns is then always satisfied when the slacks' bounds allow it, and the objective sees the cheapest slack.

The rule, per row i with singleton columns S_i (each x_j in [lb_j, ub_j], coefficient a_j, minimisation cost c_j):
  1. violation first: the row's achievable singleton activity t = Σ a_j x_j lies in [tmin, tmax]; the target interval
     is [rl_i - a', ru_i - a']; when they intersect, t is chosen inside the intersection, otherwise t is the end of
     [tmin, tmax] nearest the row (the least violation — the row stays violated by the smallest possible amount);
  2. objective second (ties): min Σ c_j x_j subject to Σ a_j x_j = t — a continuous knapsack, solved exactly by
     ordering the columns by cost per unit of activity r_j = c_j / a_j: columns below a threshold sit at their
     activity-maximising bound, columns above it at their activity-minimising bound, one column in between. Within
     the feasible interval t is the cost-optimal activity t0 (each column at its own cheapest bound, a zero-cost
     column at the value nearest 0) clipped to the interval.
For one singleton per row this is "the value that makes the row tight at its violated side, clipped to the bounds";
with several (s+ and s- of an equality row) it coordinates them so that the cheap one moves.

Rows whose singleton LP is unbounded in cost (a column that can grow without limit cheaper than another can shrink
without limit — only possible in an unbounded MIP or with two free zero-cost singletons in one row) are left alone
and counted in `skipped_rows`; their columns stay ordinary genes. Fixed columns (lb == ub) are not decoded.

Works on NumPy or CuPy arrays (the backend of the array passed in), in float64 regardless of the search precision,
chunked over individuals to stay inside `mem_budget_bytes`. Integer-typed gene matrices cannot hold continuous values
and are refused.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps

from itrea.ea.evaluate import DeviceProblem


class SingletonDecoder:
    """Detects singleton continuous columns of `dp.prob` and decodes them in closed form: `apply(X)` (in place)."""

    def __init__(self, dp: DeviceProblem, mem_budget_bytes: int = 256 << 20):
        self.dp = dp
        self.mem_budget_bytes = int(mem_budget_bytes)
        prob = dp.prob
        n = prob.n
        C = sps.csc_matrix(prob.A, dtype=np.float64)
        C.sort_indices()
        deg = np.diff(C.indptr)
        lb = np.asarray(prob.lb, dtype=np.float64)
        ub = np.asarray(prob.ub, dtype=np.float64)
        cand = np.flatnonzero(~np.asarray(prob.isint, bool) & (deg == 1) & (lb < ub))
        self.skipped_rows = 0
        self._dev: dict = {}
        self._host: dict | None = None
        self._cols = np.zeros(0, np.int64)
        self._rows = np.zeros(0, np.int64)
        if cand.size == 0:
            return
        row = C.indices[C.indptr[cand]].astype(np.int64)
        a = C.data[C.indptr[cand]]
        cmin = np.asarray(prob.cmin, dtype=np.float64)[cand]
        lbs, ubs = lb[cand], ub[cand]
        pos = a > 0
        with np.errstate(invalid="ignore"):
            ulo = np.where(pos, a * lbs, a * ubs)          # activity range of a_j x_j (never +inf / NaN)
            uhi = np.where(pos, a * ubs, a * lbs)          # (never -inf / NaN)
        x_lo = np.where(pos, lbs, ubs)                     # the value of x_j at its activity-minimising bound
        x_hi = np.where(pos, ubs, lbs)
        r = cmin / a                                       # cost per unit of activity
        u0 = np.where(r > 0, ulo, np.where(r < 0, uhi, np.clip(0.0, ulo, uhi)))
        # order: by row, then ratio; ties put columns that can reach -inf first and +inf last (keeps sums NaN-free)
        cls = np.where(np.isinf(ulo) & np.isfinite(uhi), 0,
                       np.where(np.isfinite(ulo) & np.isfinite(uhi), 1, np.where(np.isinf(ulo), 2, 3)))
        order = np.lexsort((cls, r, row))
        row, a, ulo, uhi, x_lo, x_hi, u0, cols = (v[order] for v in (row, a, ulo, uhi, x_lo, x_hi, u0, cand))
        rows_u, start = np.unique(row, return_index=True)
        R, S = rows_u.size, cols.size
        bounds = np.append(start, S)
        pos_in_row = np.arange(S) - np.repeat(start, np.diff(bounds))
        seg = np.repeat(np.arange(R), np.diff(bounds))
        shi_excl = np.empty(S)
        slo_after = np.empty(S)
        bhi = np.empty(S)
        tmin, tmax, t0 = np.empty(R), np.empty(R), np.empty(R)
        bad = np.zeros(R, bool)
        with np.errstate(invalid="ignore"):
            for g in range(R):
                s0, s1 = bounds[g], bounds[g + 1]
                hi_g, lo_g = uhi[s0:s1], ulo[s0:s1]
                ce = np.concatenate(([0.0], np.cumsum(hi_g)[:-1]))                # Σ_{q'<q} uhi
                sa = np.concatenate((np.cumsum(lo_g[::-1])[::-1][1:], [0.0]))    # Σ_{q'>q} ulo
                shi_excl[s0:s1], slo_after[s0:s1] = ce, sa
                bhi[s0:s1] = ce + hi_g + sa                                       # top of the range where q is free
                tmin[g], tmax[g], t0[g] = lo_g.sum(), hi_g.sum(), u0[s0:s1].sum()
                bad[g] = bool(np.any(np.isnan(bhi[s0:s1])) or np.isnan(tmin[g]) or np.isnan(tmax[g]))
        if bad.any():
            keep_r = ~bad
            self.skipped_rows = int(bad.sum())
            keep = keep_r[seg]
            remap = np.cumsum(keep_r) - 1
            row, a, ulo, uhi, x_lo, x_hi, cols = (v[keep] for v in (row, a, ulo, uhi, x_lo, x_hi, cols))
            shi_excl, slo_after, bhi, pos_in_row = shi_excl[keep], slo_after[keep], bhi[keep], pos_in_row[keep]
            seg = remap[seg[keep]]
            rows_u, tmin, tmax, t0 = rows_u[keep_r], tmin[keep_r], tmax[keep_r], t0[keep_r]
            R, S = rows_u.size, cols.size
            start = np.flatnonzero(np.concatenate(([True], seg[1:] != seg[:-1]))) if S else np.zeros(0, np.int64)
            if R == 0:
                return
        self._cols, self._rows = cols.astype(np.int64), rows_u.astype(np.int64)
        # the rows' activity without the singleton columns: A[rows] with those columns zeroed
        Ar = sps.csr_matrix(prob.A, dtype=np.float64)[rows_u]
        mask = np.ones(n)
        mask[cols] = 0.0
        Ar = sps.csr_matrix(Ar.multiply(mask.reshape(1, -1)))
        Ar.eliminate_zeros()
        G = sps.csr_matrix((np.ones(S), (seg, np.arange(S))), shape=(R, S))     # entry -> row indicator
        self._host = dict(
            Ar=Ar, G=G, cols=self._cols, seg=seg, pos=pos_in_row.astype(np.float64), start=start.astype(np.int64),
            a=a, ulo=ulo, uhi=uhi, x_lo=x_lo, x_hi=x_hi, shi_excl=shi_excl, slo_after=slo_after, bhi=bhi,
            rl=np.asarray(prob.rl, dtype=np.float64)[rows_u], ru=np.asarray(prob.ru, dtype=np.float64)[rows_u],
            tmin=tmin, tmax=tmax, t0=t0, lb=lb[cols], ub=ub[cols])

    # ----------------------------------------------------------------------- api
    @property
    def columns(self) -> np.ndarray:
        """Indices of the decoded (singleton continuous) columns, host int64, in decoding order."""
        return self._cols

    @property
    def rows(self) -> np.ndarray:
        """Rows that hold at least one decoded column."""
        return self._rows

    def __len__(self) -> int:
        return int(self._cols.size)

    def _arrays(self, gpu: bool) -> dict:
        if not gpu:
            return self._host
        if not self._dev:
            import cupy as cp
            import cupyx.scipy.sparse as cps
            h = self._host
            self._dev = {k: (cps.csr_matrix(v) if sps.issparse(v) else cp.asarray(v)) for k, v in h.items()}
        return self._dev

    def apply(self, X) -> dict:
        """Decode the singleton columns of every row of X (k, n; NumPy or CuPy, float dtype) in place.
        Returns {"columns": S, "rows": R, "chunk": individuals per pass, "seconds"}."""
        import time

        t0 = time.perf_counter()
        S = int(self._cols.size)
        out = {"columns": S, "rows": int(self._rows.size), "chunk": 0, "seconds": 0.0}
        if S == 0 or X.shape[0] == 0:
            return out
        if np.issubdtype(np.dtype(X.dtype), np.integer):
            raise TypeError("SingletonDecoder needs a float gene dtype (the problem has continuous columns)")
        if X.shape[1] != self.dp.prob.n:
            raise ValueError(f"X has {X.shape[1]} columns, the problem has {self.dp.prob.n}")
        gpu = not isinstance(X, np.ndarray)
        if gpu:
            import cupy as xp
        else:
            xp = np
        d = self._arrays(gpu)
        R = int(self._rows.size)
        k = X.shape[0]
        per_ind = 8 * (X.shape[1] + 8 * R + 10 * S) + 64
        chunk = max(1, min(k, self.mem_budget_bytes // per_ind))
        out["chunk"] = int(chunk)
        seg = d["seg"]
        for s in range(0, k, chunk):
            Xc = X[s:s + chunk]
            kc = Xc.shape[0]
            XT = xp.ascontiguousarray(Xc.T.astype(xp.float64))
            ap = d["Ar"] @ XT                                           # (R, kc) activity without the singletons
            del XT
            L = d["rl"][:, None] - ap
            U = d["ru"][:, None] - ap
            tmin, tmax = d["tmin"][:, None], d["tmax"][:, None]
            lo = xp.maximum(L, tmin)
            hi = xp.minimum(U, tmax)
            meet = lo <= hi
            t = xp.where(meet, xp.clip(xp.broadcast_to(d["t0"][:, None], lo.shape), lo, hi),
                         xp.where(tmax < L, tmax, tmin))
            del L, U, lo, hi, meet, ap
            ok = xp.isfinite(t)                                         # (R, kc)
            t = xp.where(ok, t, 0.0)
            te = t[seg]                                                 # (S, kc) the row's target per entry
            below = (d["bhi"][:, None] < te).astype(xp.float64)
            p = (d["G"] @ below).astype(xp.int64)                       # (R, kc) the free column's position
            del below
            pe = p[seg]
            pos = d["pos"][:, None]
            gi = d["start"][seg][:, None] + pe                           # index of the row's free column
            umid = te - d["shi_excl"][gi] - d["slo_after"][gi]
            xmid = umid / d["a"][:, None]
            val = xp.where(pos < pe, d["x_hi"][:, None], xp.where(pos > pe, d["x_lo"][:, None], xmid))
            val = xp.clip(val, d["lb"][:, None], d["ub"][:, None])
            keep = ok[seg]
            cur = Xc[:, d["cols"]].T.astype(xp.float64)
            Xc[:, d["cols"]] = xp.where(keep, val, cur).T.astype(X.dtype)
            del te, pe, gi, umid, xmid, val, keep, cur, t, p
        out["seconds"] = time.perf_counter() - t0
        return out
