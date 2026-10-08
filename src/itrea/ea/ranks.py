"""Three-rank survival — the core rule of I-TREA, carried from the unpublished manuscript to single-objective MIP.

I-TREA (Orbay 2011 M.S. thesis, manuscript unpublished, 2015) ranks a population into
    Rank 0  infeasible good   infeasible solutions not dominated by any feasible solution; a quota of the population is
                              reserved for them;
    Rank 1  feasible good     feasible non-dominated solutions (not rejected by the decision maker);
    Rank 2  rejects           solutions dominated by feasible ones, or rejected by the decision maker.
With one objective (the MIP's), "not dominated by any feasible solution" means "objective better than the best feasible
one known", and Berk's brief (2026-10-05) fixes the sizes: Rank 0 at most x of the population, Rank 1 the protected best
y of the feasible solutions. The solver plays the decision maker's part: a Rank 0 point whose key beats the solver's
PROVEN bound can never become feasible at that value and is eliminated (`rank0_bound_demote`; part I moved the
decision maker's rejections to Rank 2 — here they would sit on Rank 2's first front by a proven-impossible objective).

Survival from a candidate set (population ∪ offspring ∪ injected) to at most P, following I-TREA's elimination order:
  0. duplicates (identical vectors) and non-finite candidates are removed first (caller passes `dup`);
  1. Rank 1: the best n1 feasible candidates by key, n1 = max(ceil(y * basis), #previous Rank 1 members still among
     the Rank 1 candidates) (`prev_r1`), basis = min(#feasible, P) ("feasible", default) or P; equal-key twins leave
     Rank 1 and compete as Rank 2 (`rank1_twins="drop"`, I-TREA's first Rank 1 elimination). Protected: always
     survives. The ratchet keeps the protection when #feasible falls between generations (generational Rank 2 can
     shrink it): a previous member leaves Rank 1 only when at least n1 feasible candidates precede it in (key, age),
     i.e. it is displaced by strictly better ones (or by an older equal-key one with `rank1_twins="keep"`). So under
     "feasible" n1 never shrinks below the previous Rank 1 still present and grows towards ceil(y * min(#feasible, P));
     once enough feasible solutions have appeared it behaves like "population". With y = 0, n1 = 0.
     `rank1_mode="clearing"` (D-016, cuOpt's population rule) replaces this step: the feasible candidates in (key, age)
     order are passed to a picker (`elite.clearing_pick`) that accepts one only when it differs from every member
     already accepted in at least `rank1_min_diff` integer variables, up to `rank1_size` (0: ceil(y * P)) members.
     The best feasible candidate always joins; no twin rule, no ratchet (the cap is fixed, so a member leaves only when
     a better candidate lies within the distance or enough better distinct ones exist).
  2. Rank 0: infeasible candidates with key < inc − imp_tol·max(1,|inc|) (inc = best feasible key known, the
     population's or the solver's); bound-demoted ones are eliminated. At most floor(x * P). Over the quota,
     `rank0_select`: "itrea" — non-dominated fronts in (key, V) first (I-TREA's first Rank 0 elimination), the
     overflowing front thinned by repeatedly removing the member nearest to another in normalised (key, V) space, ties
     → worse key removed (second Rank 0 elimination); "violation" — ascending V; "rank_prob" — rank-weighted sampling
     without replacement. Over-quota infeasible-good candidates are ELIMINATED, never kept as Rank 2: x is a total cap
     on super-optimal infeasible survivors (in part I they are Rank 0 or nothing — they are not dominated by a
     feasible solution, so they cannot be "rejects").
  3. Rank 2: the rest fill the remaining slots — with `rank2_replacement="generational"` (default) only this
     generation's NEW candidates (offspring, injected, immigrants: `is_new`) are eligible, so the old Rank 2 dies and
     the protection of Rank 1 / reservation of Rank 0 has a counterfactual; "plus" (part I) lets parents compete.
     Not in the feasibility phase (no incumbent): there is no Rank 0 or Rank 1 to protect anyone, so ALL candidates,
     parents included, compete by (V, key) — a (mu + lambda) by violation until the first feasible solution, the same
     in every arm.
     `rank2_select`: "itrea" — non-dominated sorting in (key, V) with crowding distance on the overflowing front
     (I-TREA's two Rank 2 eliminations; NSGA-II's crowding), ties → newer kept (drift across plateaus);
     "key_then_violation" — feasible by key, then infeasible by V.
  4. If fewer than P qualify, the survival returns fewer; the engine fills the shortfall with immigrants.
Before any feasible solution is known (`no_incumbent_policy`): "violation" (default) — feasibility phase, all candidates
(old and new, whatever `rank2_replacement`) by (V, key), labelled Rank 2; "itrea" — everyone is infeasible good, Rank 0
selection over all of them with no quota.

Pure host-side NumPy: the candidate set has a few thousand entries at most, and only (key, V, feasible, id) are needed.
"""
from __future__ import annotations

import bisect
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from itrea.config import RankCfg
from itrea.ea.population import R0, R1, R2


@dataclass
class Survival:
    idx: np.ndarray       # indices into the candidate set, in order Rank 1, Rank 0, Rank 2
    rank: np.ndarray      # (len(idx),) int8 rank labels
    inc: float            # incumbent key used (inf if none)
    n0: int
    n1: int
    n2: int
    feasibility_phase: bool
    rank0_candidates: int  # infeasible-good candidates before the quota
    rank0_demoted: int     # infeasible-good candidates eliminated by the solver's proven bound
    rank0_over_quota: int  # infeasible-good candidates eliminated by the quota
    shortfall: int         # P - len(idx): slots the engine fills with immigrants


# ----------------------------------------------------------------------------------------------- 2-D Pareto tools
def nondominated_fronts(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Front index (0 = non-dominated) of every point, both objectives minimised; O(N log N) (Jensen 2003).

    Identical points share a front (neither dominates the other)."""
    N = f1.shape[0]
    front = np.empty(N, dtype=np.int64)
    if N == 0:
        return front
    pairs = np.stack([f1, f2], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)   # sorted by f1 then f2
    inv = inv.ravel()
    mins: list[float] = []          # f2 of the last point placed in each front (non-decreasing over fronts)
    uf = np.empty(uniq.shape[0], dtype=np.int64)
    for u, (_, b) in enumerate(uniq):
        k = bisect.bisect_right(mins, b)
        if k == len(mins):
            mins.append(b)
        else:
            mins[k] = b
        uf[u] = k
    front[:] = uf[inv]
    return front


def crowding_distance(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """NSGA-II crowding distance of points in ONE front (boundary points get inf)."""
    N = f1.shape[0]
    d = np.zeros(N)
    if N <= 2:
        d[:] = np.inf
        return d
    for f in (f1, f2):
        o = np.argsort(f, kind="stable")
        rng = f[o[-1]] - f[o[0]]
        d[o[0]] = d[o[-1]] = np.inf
        if rng > 0 and np.isfinite(rng):
            d[o[1:-1]] += (f[o[2:]] - f[o[:-2]]) / rng
    return d


def _normalise(v: np.ndarray) -> np.ndarray:
    lo, hi = np.min(v), np.max(v)
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def nearest_thinning(f1: np.ndarray, f2: np.ndarray, keep: int) -> np.ndarray:
    """I-TREA's second Rank 0 elimination: repeatedly drop the point with the smallest distance to its nearest
    neighbour in normalised (f1, f2) space; of the two points of that closest pair, drop the one with the larger f1
    (worse key — I-TREA drops the one with the lower estimated utility). Returns the kept positions (sorted)."""
    N = f1.shape[0]
    if keep >= N:
        return np.arange(N)
    if keep <= 0:
        return np.empty(0, dtype=np.int64)
    a, b = _normalise(f1), _normalise(f2)
    alive = np.ones(N, dtype=bool)
    D = np.sqrt((a[:, None] - a[None, :]) ** 2 + (b[:, None] - b[None, :]) ** 2)
    np.fill_diagonal(D, np.inf)
    for _ in range(N - keep):
        i, j = np.unravel_index(np.argmin(D), D.shape)
        drop = i if (f1[i], -i) > (f1[j], -j) else j     # worse key; ties: the later index (younger) stays out
        alive[drop] = False
        D[drop, :] = np.inf
        D[:, drop] = np.inf
    return np.flatnonzero(alive)


def _pareto_select(idx: np.ndarray, key: np.ndarray, viol: np.ndarray, ids: np.ndarray, cap: int,
                   last_front: str) -> np.ndarray:
    """Choose `cap` of idx by fronts in (key, V); the overflowing front by `last_front` ("crowding" | "nearest").
    Exact ties prefer the NEWER individual (higher id), so the population can drift across objective plateaus."""
    if cap <= 0 or idx.size == 0:
        return np.empty(0, dtype=np.int64)
    if idx.size <= cap:
        return idx[np.lexsort((-ids[idx], viol[idx], key[idx]))]
    fr = nondominated_fronts(key[idx], viol[idx])
    out: list[np.ndarray] = []
    taken = 0
    for f in range(int(fr.max()) + 1):
        members = idx[fr == f]
        if taken + members.size <= cap:
            out.append(members[np.lexsort((-ids[members], viol[members], key[members]))])
            taken += members.size
            if taken == cap:
                break
            continue
        room = cap - taken
        if last_front == "crowding":
            cd = crowding_distance(key[members], viol[members])
            # largest crowding first; ties: newer (higher id) first so that older ones are eliminated (I-TREA)
            order = np.lexsort((-ids[members], -cd))
            out.append(members[order[:room]])
        else:
            kept = nearest_thinning(key[members], viol[members], room)
            sel = members[kept]
            out.append(sel[np.lexsort((-ids[sel], viol[sel], key[sel]))])
        break
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def _by_violation(idx: np.ndarray, key: np.ndarray, viol: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return idx[np.lexsort((-ids[idx], key[idx], viol[idx]))]


def rank1_cap(cfg: RankCfg, P: int) -> int:
    """Rank 1's size cap under `rank1_mode="clearing"` (D-016): `rank1_size`, or ceil(y * P) when it is 0; never above P."""
    return min(P, cfg.rank1_size if cfg.rank1_size > 0 else math.ceil(cfg.y * P - 1e-12))


# ------------------------------------------------------------------------------------------------------- survival
def three_rank_survival(key: np.ndarray, viol: np.ndarray, feasible: np.ndarray, P: int, cfg: RankCfg,
                        inc_external: float = np.inf, dual_bound: float = -np.inf,
                        rng: np.random.Generator | None = None, dup: np.ndarray | None = None,
                        ids: np.ndarray | None = None, is_new: np.ndarray | None = None,
                        prev_r1: np.ndarray | None = None, viol_alt: np.ndarray | None = None,
                        rank1_pick: Callable[[np.ndarray, int], np.ndarray] | None = None) -> Survival:
    """Choose at most P survivors and their ranks.

    key/viol/feasible: host arrays over the candidate set (key lower is better). dup: True = drop (duplicate vector).
    inc_external: the solver's incumbent key known to the EA (inf if none). dual_bound: the solver's proven bound as a
    key (-inf if unknown). ids: individual ids (age tie-breaks; lower = older); default = position.
    is_new: True for this generation's offspring / injected / immigrants (generational Rank 2); default all True.
    prev_r1: True for the members of the previous generation's Rank 1 (the Rank 1 ratchet, step 1); default none.
    viol_alt: V of every candidate on a second row scale (`ranks.feas_alt_scale`); used only in the feasibility phase
    with the "violation" policy — non-dominated fronts over (V, viol_alt), crowding on the overflowing front, so each
    scale's least-violated candidate survives; ignored otherwise.
    rank1_pick: required when `cfg.rank1_mode == "clearing"` (D-016): called as rank1_pick(F, cap) with the feasible
    alive candidates in (key, age) order and the size cap; returns Rank 1 (a subsequence of F, best first), e.g.
    `elite.clearing_pick` bound to the candidates' vectors. The twin rule and the ratchet do not apply then."""
    key = np.asarray(key, dtype=np.float64)
    viol = np.asarray(viol, dtype=np.float64)
    feasible = np.asarray(feasible, dtype=bool)
    N = key.shape[0]
    ids = np.arange(N, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
    alive = np.ones(N, dtype=bool) if dup is None else ~np.asarray(dup, dtype=bool)
    alive &= np.isfinite(key) & np.isfinite(viol)
    new = np.ones(N, dtype=bool) if is_new is None else np.asarray(is_new, dtype=bool)

    F = np.flatnonzero(feasible & alive)
    F = F[np.lexsort((ids[F], key[F]))]          # Rank 1 boundary: oldest first (displaced only by strictly better)
    I = np.flatnonzero(~feasible & alive)
    best_feas = key[F[0]] if F.size else np.inf
    inc = float(min(best_feas, inc_external))
    feas_phase = not np.isfinite(inc)

    in_r1 = np.zeros(N, dtype=bool)
    in_r0 = np.zeros(N, dtype=bool)

    # ---- Rank 1: protected feasible elite (twins removed first if asked)
    r1 = np.empty(0, dtype=np.int64)
    if cfg.rank1_mode == "clearing":
        if rank1_pick is None:
            raise ValueError("ranks.rank1_mode='clearing' needs rank1_pick (the engine binds elite.clearing_pick)")
        cap = rank1_cap(cfg, P)
        if F.size and cap > 0:
            r1 = np.asarray(rank1_pick(F, cap), dtype=np.int64)
            if r1.size > cap or not np.all(np.isin(r1, F)):
                raise RuntimeError("rank1_pick returned rows outside the feasible candidates or over the cap")
            in_r1[r1] = True
    elif F.size and cfg.y > 0:
        cand = F
        if cfg.rank1_twins == "drop":
            _, first = np.unique(key[F], return_index=True)   # F is key-sorted, so `first` keeps the oldest twin
            cand = F[np.sort(first)]
        basis = min(F.size, P) if cfg.elite_basis == "feasible" else P
        # ratchet: never fewer slots than the previous Rank 1 members still among the candidates, so a member leaves
        # only when n1 better (key, age) feasible candidates precede it — never because #feasible fell
        n_keep = int(np.asarray(prev_r1, dtype=bool)[cand].sum()) if prev_r1 is not None else 0
        n1 = min(cand.size, P, max(1, math.ceil(cfg.y * basis - 1e-12), n_keep))
        r1 = cand[:n1]
        in_r1[r1] = True

    # ---- Rank 0: infeasible good
    r0 = np.empty(0, dtype=np.int64)
    n_good = n_demoted = 0
    eliminated = np.zeros(N, dtype=bool)         # infeasible good that did not make Rank 0 (quota, bound): gone
    if feas_phase:
        if cfg.no_incumbent_policy == "itrea":
            n_good = int(I.size)
            r0 = _pareto_select(I, key, viol, ids, P - r1.size, "nearest")
    else:
        thr = inc - cfg.imp_tol * max(1.0, abs(inc))
        good = I[key[I] < thr]
        eliminated[good] = True
        if cfg.rank0_bound_demote and np.isfinite(dual_bound):
            ok = key[good] >= dual_bound - cfg.imp_tol * max(1.0, abs(dual_bound))
            n_demoted = int((~ok).sum())
            good = good[ok]
        n_good = int(good.size)
        cap = min(math.floor(cfg.x * P + 1e-12), P - r1.size)
        if cap > 0 and good.size:
            if cfg.rank0_select == "itrea":
                r0 = _pareto_select(good, key, viol, ids, cap, "nearest")
            elif cfg.rank0_select == "violation":
                r0 = _by_violation(good, key, viol, ids)[:cap]
            elif cfg.rank0_select == "rank_prob":
                ordered = _by_violation(good, key, viol, ids)
                if ordered.size > cap:
                    if rng is None:
                        raise ValueError("rank_prob needs an rng")
                    w = np.exp(-np.arange(ordered.size) / max(1.0, cap / 2.0))
                    pick = rng.choice(ordered.size, size=cap, replace=False, p=w / w.sum())
                    ordered = ordered[np.sort(pick)]
                r0 = ordered[:cap]
            else:
                raise ValueError(f"unknown rank0_select {cfg.rank0_select!r}")
    in_r0[r0] = True
    n_over = int(eliminated.sum()) - int(r0.size) - n_demoted if not feas_phase else 0

    # ---- Rank 2: rejects fill the rest (never an eliminated infeasible-good candidate)
    slots = P - r1.size - r0.size
    pool2 = alive & ~in_r1 & ~in_r0 & ~eliminated
    if cfg.rank2_replacement == "generational" and not feas_phase:
        pool2 &= new                             # feasibility phase: everyone competes by (V, key) (docstring)
    rest = np.flatnonzero(pool2)
    if slots <= 0 or rest.size == 0:
        r2 = np.empty(0, dtype=np.int64)
    elif feas_phase and cfg.no_incumbent_policy == "violation" and viol_alt is not None:
        va = np.asarray(viol_alt, dtype=np.float64)
        rest = rest[np.isfinite(va[rest])]
        r2 = _pareto_select(rest, viol, va, ids, slots, "crowding")
    elif feas_phase and cfg.no_incumbent_policy == "violation":
        r2 = _by_violation(rest, key, viol, ids)[:slots]
    elif cfg.rank2_select == "itrea":
        r2 = _pareto_select(rest, key, viol, ids, slots, "crowding")
    elif cfg.rank2_select == "key_then_violation":
        rf = rest[feasible[rest]]
        ri = rest[~feasible[rest]]
        r2 = np.concatenate([rf[np.lexsort((ids[rf], key[rf]))], _by_violation(ri, key, viol, ids)])[:slots]
    else:
        raise ValueError(f"unknown rank2_select {cfg.rank2_select!r}")

    idx = np.concatenate([r1, r0, r2]).astype(np.int64)
    rank = np.concatenate([np.full(r1.size, R1), np.full(r0.size, R0), np.full(r2.size, R2)]).astype(np.int8)
    return Survival(idx=idx, rank=rank, inc=inc, n0=int(r0.size), n1=int(r1.size), n2=int(r2.size),
                    feasibility_phase=bool(feas_phase), rank0_candidates=n_good, rank0_demoted=n_demoted,
                    rank0_over_quota=max(0, n_over), shortfall=int(P - idx.size))
