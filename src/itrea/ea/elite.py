"""Rank 1 as a distinct elite — cuOpt's clearing rule (vault D-016, issue #25).

NVIDIA cuOpt's MIP heuristics keep a population in which any two members differ in at least k integer variables
(`population_t::add_solution`, `get_max_var_threshold`; NVIDIA/cuopt dfbbcc5a, Apache-2.0): a new solution that is
better than its most similar member evicts every similar worse member, a worse one is refused, and the best feasible
solution is always kept. Applied best-first to a candidate set, that rule is the classic clearing procedure: walk the
candidates from best to worst and accept one when it is at least k integer variables away from every member already
accepted. `clearing_pick` does that on the device; `three_rank_survival` (ranks.py) calls it through a picker when
`ranks.rank1_mode == "clearing"`.

Distance: the number of integer variables whose values differ (values rounded to the nearest integer first, as in
`frontier.Frontier.distance_to`; Rank 1 candidates are feasible, so their integer genes are integral within int_tol).
Continuous variables do not count (cuOpt's MIP similarity counts integers only).
"""
from __future__ import annotations

import numpy as np

from itrea.backend import Backend


def min_diff_auto(n_int: int) -> int:
    """cuOpt's distinctness threshold as a number of differing integer variables: its population treats two solutions
    as similar when more than `get_max_var_threshold(n_int)` integers are equal, i.e. when fewer than k differ, with
    k = 1 (n_int < 50; max(1, n_int - 1) equal), 2 (< 80), 4 (< 200), 8 (< 1000), 10 (otherwise)."""
    if n_int < 50:
        return 1
    if n_int < 80:
        return 2
    if n_int < 200:
        return 4
    if n_int < 1000:
        return 8
    return 10


def _int_genes(bk: Backend, X, rows, ints):
    """The integer columns of the given rows of X, gathered in one step, as an integer-valued device array (rounded if
    the gene dtype is floating)."""
    xp = bk.xp
    Xi = X[xp.asarray(rows)[:, None], ints[None, :]]
    if np.dtype(Xi.dtype).kind == "f":
        Xi = xp.rint(Xi)
    return Xi


DEFAULT_BLOCK = 32                     # rows whose distances are fetched per device->host copy
DEFAULT_BUDGET = 64 * 1024 ** 2        # bytes of the (chunk, m, n_int) comparison temporaries


def _distances(bk: Backend, Xi, a: int, b: int, budget: int) -> np.ndarray:
    """Host (b - a, m) array: the number of differing integer genes between rows a..b-1 of Xi and every row of Xi.
    Computed in chunks of rows so that the comparison temporaries stay within `budget` bytes; one copy to the host."""
    xp = bk.xp
    m, n_int = int(Xi.shape[0]), int(Xi.shape[1])
    c = max(1, budget // max(1, m * n_int))
    parts = [(Xi[r:min(r + c, b), None, :] != Xi[None, :, :]).sum(axis=2) for r in range(a, b, c)]
    return bk.asnumpy(parts[0] if len(parts) == 1 else xp.concatenate(parts))


def clearing_pick(bk: Backend, X, order: np.ndarray, ints, k: int, cap: int, stats: dict | None = None,
                  block: int = DEFAULT_BLOCK, budget: int = DEFAULT_BUDGET) -> np.ndarray:
    """Clearing over the rows `order` of X (host int array, best first): returns the accepted rows (a subsequence of
    `order`, best first), at most `cap`. A row is accepted when it differs from every row accepted before it in at
    least k integer variables; the first row is always accepted. `ints`: device array of integer column indices (empty:
    no distance exists, so the first `cap` rows are returned). `stats` (optional dict) receives `cleared`: the rows the
    distance rule skipped before the walk ended (the mechanism D-016 checks).

    The distances of an accepted row to every candidate are fetched in blocks of `block` rows (one device->host copy
    per block, usually one or two per call), not one synchronisation per accepted row."""
    order = np.asarray(order, dtype=np.int64)
    if stats is not None:
        stats["cleared"] = 0
    if cap <= 0 or order.size == 0:
        return np.empty(0, dtype=np.int64)
    n_int = int(ints.shape[0])
    if n_int == 0 or k <= 0:
        return order[:cap]
    m = order.size
    Xi = _int_genes(bk, X, order, ints)                       # (m, n_int), rows in `order`
    D = np.zeros((m, m), dtype=np.int64)                      # rows filled lazily, block by block
    have = 0                                                  # rows [0, have) of D are filled
    chosen: list[int] = []
    cleared = 0
    for j in range(m):
        if any(D[a, j] < k for a in chosen):
            cleared += 1
            continue
        chosen.append(j)
        if len(chosen) == cap:
            break
        if j >= have:                                         # this row's distances are needed from now on
            b = min(m, j + max(1, block))
            D[j:b] = _distances(bk, Xi, j, b, budget)
            have = b
    if stats is not None:
        stats["cleared"] = cleared
    return order[np.asarray(chosen, dtype=np.int64)]


def elite_spread(bk: Backend, X, rows: np.ndarray, keys: np.ndarray, ints, band: float | None = None) -> dict:
    """Spread of an elite (logged in both Rank 1 modes): its size `r1_n`; the share of integer variables in which each
    member differs from the elite's best member — max `r1_dmax` and median `r1_dmed` over the other members; the
    relative objective gap of the worst member `r1_gap` = (worst - best) / max(1, |best|); and, with `band`, `r1_dmax_band`
    = r1_dmax over the members within (worst - best) <= band * max(1, |best|) (the frontier archive's quality band, so the
    two spreads compare like for like). Distances are None with fewer than two members (or no integer variable)."""
    rows = np.asarray(rows, dtype=np.int64)
    keys = np.asarray(keys, dtype=np.float64)
    out = dict(r1_n=int(rows.size), r1_dmax=None, r1_dmed=None, r1_gap=None)
    if band is not None:
        out["r1_dmax_band"] = None
    n_int = int(ints.shape[0])
    if rows.size < 2:
        return out
    b = int(np.argmin(keys))
    scale = max(1.0, abs(float(keys[b])))
    out["r1_gap"] = round(float((keys.max() - keys[b]) / scale), 9)
    if n_int == 0:
        return out
    Xi = _int_genes(bk, X, rows, ints)
    d = bk.asnumpy((Xi != Xi[b][None, :]).sum(axis=1)).astype(np.float64) / n_int
    others = np.ones(rows.size, bool)
    others[b] = False
    out.update(r1_dmax=round(float(d[others].max()), 6), r1_dmed=round(float(np.median(d[others])), 6))
    if band is not None:
        inb = others & ((keys - keys[b]) <= band * scale)
        out["r1_dmax_band"] = round(float(d[inb].max()), 6) if inb.any() else 0.0
    return out
