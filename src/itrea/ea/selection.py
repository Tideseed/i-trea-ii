"""Parent selection: tournaments over the ranked population, on the host.

The population has at most a few thousand members and selection needs only (rank, key, V, feasible), so this is plain
NumPy on the host for both backends; the engine gathers the chosen rows of X on the device afterwards.

select_parents(pop_rank, key, viol, feasible, n_pairs, cfg, rng) -> (ia, ib)
    Two int64 arrays of length n_pairs: the first and second parent of every child (indices into the population).

Tournament. Each parent is the winner of a tournament of `cfg.tournament_k` entrants drawn uniformly WITH replacement
from its pool (as in DEAP's `selTournament`; without replacement only matters when k approaches the pool size).
Consequence worth knowing when reading frequencies: with k = 2 and two individuals, the better one wins 3/4 of the
draws, because a quarter of the tournaments draw the worse individual twice.

Pools. `cfg.parent_rank_weights = None` — one pool, the whole population (I-TREA: every member can enter the mating
pool). `[w_rank0, w_rank1, w_rank2]` — each draw first picks a pool with these probabilities (empty pools get weight 0
and the rest are renormalised; if every weighted pool is empty, the whole population). Unranked members (NO_RANK)
belong to no rank pool.

Comparators.
    "itrea"       I-TREA's tournament: Rank 1 < Rank 0 < Rank 2 < NO_RANK (better rank wins), then lower key, then
                  lower V, then a fair coin. A total order, so a k-way tournament is an argmin.
    "deb"         Deb (2000): feasible beats infeasible; two feasible by key (then V); two infeasible by V (then key);
                  then a fair coin. A total order.
    "stochastic"  Stochastic ranking (Runarsson & Yao 2000): a pair is compared by key if both are feasible OR with
                  probability `cfg.p_f`; otherwise by Deb's rule. This is R&Y's rule — p_f applies whenever at least
                  one side is infeasible (both-infeasible pairs included), which is broader than the contract stub's
                  "exactly one infeasible" wording. Not a total order: a k-way tournament is a chain of k - 1 pairwise
                  comparisons (winner so far vs the next entrant), each with a fresh coin.
A NaN key (a non-finite vector) or NaN V counts as +inf, i.e. worst.

ia and ib differ whenever the population has more than one member: a colliding second parent is redrawn from its pool
a few times, and if it still collides (a one-member pool), drawn by a tournament over the whole population minus ia.
"""
from __future__ import annotations

import numpy as np

from itrea.config import SelectionCfg
from itrea.ea.population import NO_RANK, R0, R1, R2

# Tournament position of each rank label for comparator "itrea" (lower wins): Rank 1 < Rank 0 < Rank 2 < unranked.
_ITREA_RANK_ORDER = {R1: 0, R0: 1, R2: 2, NO_RANK: 3}
_REDRAWS = 8          # attempts to draw a second parent different from the first before the fallback


def _clean(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return np.where(np.isnan(v), np.inf, v)


def _dense_rank(*keys_low_to_high: np.ndarray) -> np.ndarray:
    """Dense rank (0 = best) of the lexicographic order given by the keys, the LAST key being the primary one (the
    np.lexsort convention). Exactly tied individuals share a rank."""
    order = np.lexsort(keys_low_to_high)
    N = order.size
    out = np.empty(N, dtype=np.int64)
    if N == 0:
        return out
    stacked = np.stack([k[order] for k in keys_low_to_high], axis=0)
    new = np.ones(N, dtype=bool)
    new[1:] = np.any(stacked[:, 1:] != stacked[:, :-1], axis=0)
    out[order] = np.cumsum(new) - 1
    return out


def order_position(comparator: str, pop_rank: np.ndarray, key: np.ndarray, viol: np.ndarray,
                   feasible: np.ndarray) -> np.ndarray:
    """Dense tournament position of every individual under a TOTAL-order comparator ("itrea" or "deb"):
    a lower position wins; equal positions are exact ties (broken by a fair coin in the tournament)."""
    key, viol = _clean(key), _clean(viol)
    feasible = np.asarray(feasible, dtype=bool)
    if comparator == "itrea":
        r = np.asarray(pop_rank, dtype=np.int64)
        ro = np.full(r.shape, _ITREA_RANK_ORDER[NO_RANK], dtype=np.int64)
        for lab, pos in _ITREA_RANK_ORDER.items():
            ro[r == lab] = pos
        return _dense_rank(viol, key, ro)
    if comparator == "deb":
        # feasible first by (key, V); infeasible after them by (V, key)
        prim = np.where(feasible, key, viol)
        sec = np.where(feasible, viol, key)
        return _dense_rank(sec, prim, (~feasible).astype(np.int64))
    raise ValueError(f"comparator {comparator!r} is not a total order (use 'itrea' or 'deb')")


class _Selector:
    """One call's state: pools, positions and the draw machinery shared by the first and second parents."""

    def __init__(self, pop_rank, key, viol, feasible, cfg: SelectionCfg, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.N = int(np.asarray(key).shape[0])
        self.k = int(cfg.tournament_k)
        self.feasible = np.asarray(feasible, dtype=bool)
        if cfg.comparator in ("itrea", "deb"):
            self.pos = order_position(cfg.comparator, pop_rank, key, viol, feasible)
        elif cfg.comparator == "stochastic":
            # R&Y needs two orders: by key (ties by Deb) and Deb's total order
            deb = order_position("deb", pop_rank, key, viol, feasible)
            self.deb = deb
            self.by_key = _dense_rank(deb, _clean(key))
        else:
            raise ValueError(f"unknown comparator {cfg.comparator!r}")
        # pools
        self.pools: list[np.ndarray] | None = None
        self.pool_p: np.ndarray | None = None
        if cfg.parent_rank_weights is not None:
            r = np.asarray(pop_rank, dtype=np.int64)
            pools = [np.flatnonzero(r == lab) for lab in (R0, R1, R2)]
            w = np.array([float(x) if p.size else 0.0 for x, p in zip(cfg.parent_rank_weights, pools)])
            if w.sum() > 0:
                self.pools, self.pool_p = pools, w / w.sum()

    # ------------------------------------------------------------------------------------------- entrants
    def entrants(self, n: int) -> np.ndarray:
        """(n, k) entrant indices: each row is one tournament, drawn from that tournament's pool."""
        rng, k = self.rng, self.k
        if self.pools is None:
            return rng.integers(0, self.N, size=(n, k))
        which = rng.choice(3, size=n, p=self.pool_p)
        out = np.empty((n, k), dtype=np.int64)
        for q in range(3):
            rows = np.flatnonzero(which == q)
            if rows.size:
                pool = self.pools[q]
                out[rows] = pool[rng.integers(0, pool.size, size=(rows.size, k))]
        return out

    # ------------------------------------------------------------------------------------------- winners
    def winners(self, E: np.ndarray) -> np.ndarray:
        """Winner of every tournament (rows of E)."""
        n = E.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.int64)
        rng = self.rng
        if self.cfg.comparator != "stochastic":
            # integer positions + a uniform [0, 1) coin: exact ties are broken fairly, strict order never flips
            score = self.pos[E] + rng.random(E.shape)
            return E[np.arange(n), np.argmin(score, axis=1)]
        w = E[:, 0].copy()
        for j in range(1, E.shape[1]):
            c = E[:, j]
            both_feas = self.feasible[w] & self.feasible[c]
            by_key = both_feas | (rng.random(n) < self.cfg.p_f)
            sw = np.where(by_key, self.by_key[w], self.deb[w]) + rng.random(n)
            sc = np.where(by_key, self.by_key[c], self.deb[c]) + rng.random(n)
            w = np.where(sc < sw, c, w)
        return w

    def draw(self, n: int) -> np.ndarray:
        return self.winners(self.entrants(n))

    def draw_excluding(self, ex: np.ndarray) -> np.ndarray:
        """Tournaments over the whole population minus ex[i] (one excluded index per tournament; N >= 2)."""
        n = ex.size
        E = self.rng.integers(0, self.N - 1, size=(n, self.k))
        E += (E >= ex[:, None])                       # skip the excluded index
        return self.winners(E)


def select_parents(pop_rank: np.ndarray, key: np.ndarray, viol: np.ndarray, feasible: np.ndarray, n_pairs: int,
                   cfg: SelectionCfg, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """First and second parent (int64 indices into the population) of `n_pairs` children. See the module docstring."""
    N = int(np.asarray(key).shape[0])
    if N == 0:
        raise ValueError("select_parents: empty population")
    n_pairs = int(n_pairs)
    if n_pairs <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if N == 1:
        z = np.zeros(n_pairs, dtype=np.int64)
        return z, z.copy()
    s = _Selector(pop_rank, key, viol, feasible, cfg, rng)
    ia = s.draw(n_pairs).astype(np.int64)
    ib = s.draw(n_pairs).astype(np.int64)
    for _ in range(_REDRAWS):
        same = np.flatnonzero(ia == ib)
        if same.size == 0:
            break
        ib[same] = s.draw(same.size)
    same = np.flatnonzero(ia == ib)
    if same.size:
        ib[same] = s.draw_excluding(ia[same])
    return ia, ib
