"""The frontier — an archive of diverse near-optimal feasible solutions (vault D-005).

I-TREA (part I) was bi-objective: its Rank 1 was a Pareto front the decision maker chose from. I-TREA II starts with
one objective (the MIP's), where the "front" of objectives collapses to the optimum. Berk (2026-10-05): "we can start
single objective but we shall also define a frontier of diverse solutions". The frontier makes diversity the second
axis:

    pareto_incumbent (default)  feasible solutions inside the quality band (key <= inc + band * max(1, |inc|)) that
                                are non-dominated in (key ↓, distance to the incumbent ↑): for each level of quality,
                                the solution farthest from the incumbent; thinned by crowding distance to max_size
                                (the incumbent and the farthest point are always kept);
    maxmin                      greedy max-min diverse set inside the band, starting from the incumbent.

Distance: normalised Hamming distance over integer genes (share of integer variables that differ), or normalised L1
over all genes. With several objectives (later phases) the same archive holds the Pareto front of the objectives —
`nondominated_fronts_k` is the general k-objective sort.

The archive is an output (diverse alternatives, reported with its size, spread and a 2-D hypervolume in
(relative gap, distance)), a source of immigrants (`reinject`) and, optionally, of solutions posted to the solver
(`exchange.post_frontier`: SCIP keeps non-improving solutions in its solution storage for its crossover-type heuristics).
"""
from __future__ import annotations

import numpy as np

from itrea.backend import Backend
from itrea.config import FrontierCfg
from itrea.ea.evaluate import DeviceProblem
from itrea.ea.population import DupHasher
from itrea.ea.ranks import crowding_distance, nondominated_fronts


def nondominated_fronts_k(F: np.ndarray) -> np.ndarray:
    """Front index of every row of F (N, k), all objectives minimised; O(N^2 k) — for small sets and k > 2."""
    F = np.asarray(F, dtype=np.float64)
    N = F.shape[0]
    if F.ndim == 1 or F.shape[1] == 1:
        f = F.reshape(N)
        u, inv = np.unique(f, return_inverse=True)
        return inv.astype(np.int64)
    if F.shape[1] == 2:
        return nondominated_fronts(F[:, 0], F[:, 1])
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dom = le & lt                                   # dom[i, j]: i dominates j
    count = dom.sum(axis=0)
    front = np.full(N, -1, dtype=np.int64)
    k = 0
    cur = np.flatnonzero(count == 0)
    while cur.size:
        front[cur] = k
        count = count - dom[cur].sum(axis=0)
        count[front >= 0] = -1
        cur = np.flatnonzero(count == 0)
        k += 1
    return front


def hypervolume_2d(f1: np.ndarray, f2: np.ndarray, ref: tuple[float, float]) -> float:
    """Area dominated by the points w.r.t. `ref`, both objectives minimised (points outside ref are ignored)."""
    f1, f2 = np.asarray(f1, float), np.asarray(f2, float)
    keep = (f1 < ref[0]) & (f2 < ref[1])
    a, b = f1[keep], f2[keep]
    if a.size == 0:
        return 0.0
    o = np.lexsort((b, a))
    nd, best_b = [], np.inf
    for x, y in zip(a[o], b[o]):           # sweep in f1 order; keep the staircase of non-dominated points
        if y < best_b:
            nd.append((x, y))
            best_b = y
    hv = 0.0
    for i, (x, y) in enumerate(nd):
        x_next = nd[i + 1][0] if i + 1 < len(nd) else ref[0]
        hv += (x_next - x) * (ref[1] - y)
    return float(hv)


class Frontier:
    def __init__(self, bk: Backend, dp: DeviceProblem, cfg: FrontierCfg):
        self.bk, self.dp, self.cfg = bk, dp, cfg
        xp = bk.xp
        self.X = xp.empty((0, dp.n), dtype=np.dtype(dp.dtype))
        self.key = np.empty(0)
        self.ids = np.empty(0, dtype=np.int64)
        self.dist = np.empty(0)
        self.updates = 0
        self._dup = DupHasher(bk, dp.n)
        ints = bk.asnumpy(dp.ints)
        self._ints = dp.ints if ints.size else None
        if cfg.distance == "l1" or ints.size == 0:
            lb, ub = dp.prob.lb, dp.prob.ub
            rngv = np.where(np.isfinite(ub - lb) & (ub > lb), ub - lb, 1.0)
            self._range = xp.asarray(rngv)
        else:
            self._range = None

    # --------------------------------------------------------------------------------------------- distance
    def distance_to(self, C, ref) -> np.ndarray:
        """Normalised distance of every row of C (k, n) to the vector ref (n,): host float array."""
        xp = self.bk.xp
        if C.shape[0] == 0:
            return np.empty(0)
        if self._range is None:
            Ci = C[:, self._ints]
            ri = ref[self._ints]
            d = (xp.abs(xp.rint(Ci) - xp.rint(ri)[None, :]) > 0.5).sum(axis=1) / float(Ci.shape[1])
        else:
            d = (xp.abs(C - ref[None, :]) / self._range[None, :]).mean(axis=1)
        return self.bk.asnumpy(d).astype(np.float64)

    def pairwise_mean(self) -> float:
        k = self.X.shape[0]
        if k < 2:
            return 0.0
        tot = 0.0
        for i in range(k):
            tot += float(self.distance_to(self.X, self.X[i]).sum())
        return tot / (k * (k - 1))

    # ------------------------------------------------------------------------------------------------ update
    def update(self, Xc, key: np.ndarray, feasible: np.ndarray, ids: np.ndarray, inc_x, inc_key: float) -> dict:
        """Merge feasible candidates (rows of Xc) into the archive. inc_x / inc_key: the incumbent (device vector, key)."""
        xp, cfg = self.bk.xp, self.cfg
        self.updates += 1
        if inc_x is None or not np.isfinite(inc_key):
            return self.stats(inc_key)
        lim = inc_key + cfg.band * max(1.0, abs(inc_key))
        sel = np.flatnonzero(np.asarray(feasible, bool) & (np.asarray(key) <= lim))
        CX = xp.concatenate([inc_x[None, :].astype(self.X.dtype), self.X, Xc[xp.asarray(sel)]])
        ck = np.concatenate([[inc_key], self.key, np.asarray(key)[sel]])
        cid = np.concatenate([[-1], self.ids, np.asarray(ids)[sel]])
        inb = ck <= lim
        CX, ck, cid = CX[xp.asarray(np.flatnonzero(inb))], ck[inb], cid[inb]
        dup = self._dup(CX)[0]                            # the incumbent comes first, so it is kept
        keep = np.flatnonzero(~dup)
        CX, ck, cid = CX[xp.asarray(keep)], ck[keep], cid[keep]
        d = self.distance_to(CX, inc_x.astype(CX.dtype))
        if cfg.mode == "pareto_incumbent":
            fr = nondominated_fronts(ck, -d)
            idx = np.flatnonzero(fr == 0)
            if idx.size > cfg.max_size:
                cd = crowding_distance(ck[idx], -d[idx])
                cd[np.argmin(ck[idx])] = np.inf          # incumbent (best key)
                cd[np.argmax(d[idx])] = np.inf           # farthest
                idx = idx[np.argsort(-cd, kind="stable")[:cfg.max_size]]
        else:  # maxmin
            order = list(np.lexsort((-d, ck)))
            chosen = [order[0]]
            mind = self.distance_to(CX, CX[order[0]])
            while len(chosen) < min(cfg.max_size, len(order)):
                mind_c = mind.copy()
                mind_c[chosen] = -1.0
                j = int(np.argmax(mind_c))
                if mind_c[j] <= 0:
                    break
                chosen.append(j)
                mind = np.minimum(mind, self.distance_to(CX, CX[j]))
            idx = np.asarray(chosen, dtype=np.int64)
        idx = idx[np.lexsort((-d[idx], ck[idx]))]
        self.X, self.key, self.ids, self.dist = CX[xp.asarray(idx)], ck[idx], cid[idx], d[idx]
        return self.stats(inc_key)

    # ------------------------------------------------------------------------------------------------- report
    def stats(self, inc_key: float) -> dict:
        k = int(self.key.size)
        if k == 0 or not np.isfinite(inc_key):
            return dict(size=k)
        scale = max(1.0, abs(inc_key))
        gap = (self.key - inc_key) / scale
        return dict(size=k, best_key=float(self.key.min()), worst_key=float(self.key.max()),
                    max_dist=float(self.dist.max()), mean_dist_to_inc=float(self.dist.mean()),
                    mean_pairwise=float(self.pairwise_mean()) if k <= 64 else None,
                    hv=hypervolume_2d(gap, -self.dist, (self.cfg.band + 1e-12, 0.0)))

    def host_solutions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(X float64 canonical order, keys, distances) — what the run stores at the end."""
        return self.bk.asnumpy(self.X).astype(np.float64), self.key.copy(), self.dist.copy()

    def sample(self, k: int, rng: np.random.Generator):
        """k archive rows (device) for immigration; empty if the archive is empty."""
        if self.key.size == 0 or k <= 0:
            return self.X[:0]
        pick = rng.integers(0, self.key.size, size=k)
        return self.X[self.bk.xp.asarray(pick)]
