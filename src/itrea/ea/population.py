"""The population: individuals on the device plus their bookkeeping (ranks, ids, provenance, ancestry).

Provenance is kept as fixed-size bit masks so ancestry can be tested directly (H4: does Rank 0 — infeasible good —
contribute to new incumbents?) without an unbounded lineage store:
- `origin`  (P,) uint16 — how this individual was made (one bit per operator / source, `ORIGIN_*`);
- `anc_rank` (P, depth) uint8 — bit d-1 row: the ranks (`RANK_BIT`) and sources of its ancestors d generations back;
- `anc_ops`  (P, depth) uint16 — the operators that made its ancestors d generations back.
A child's row d (d >= 2) is the OR of its parents' rows d-1; row 1 holds the parents' own rank bits at mating time
and their origin bits.

`strat` (P,) uint8 — the individual's repair strategy, an index into `StrategyCfg.strategies()` (the self-adaptive
repair strategy; 0 for everyone while `strategy.enabled` is false). Carried through `take` / `concat` like the ids.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from itrea.backend import Backend

# rank labels — I-TREA's names (Orbay 2011 M.S. thesis, `orbay2011itrea`)
R0, R1, R2 = 0, 1, 2      # Rank 0 infeasible good · Rank 1 feasible good (protected) · Rank 2 rejects
NO_RANK = -1              # not yet ranked (new children before survival)
RANK_NAMES = {R0: "rank0_infeasible_good", R1: "rank1_feasible_good", R2: "rank2_rejects", NO_RANK: "unranked"}
# bits of anc_rank
RANK_BIT = {R0: 1, R1: 2, R2: 4}
SRC_INJECTED = 8      # came from the solver
SRC_SEED_RANDOM = 16
SRC_SEED_LP = 32
SRC_IMMIGRANT = 64

# bits of origin / anc_ops
ORIGIN_SEED_RANDOM = 1 << 0
ORIGIN_SEED_LP = 1 << 1
ORIGIN_INJECTED = 1 << 2
ORIGIN_IMMIGRANT = 1 << 3
ORIGIN_X_UNIFORM = 1 << 4
ORIGIN_X_BLOCK = 1 << 5
ORIGIN_X_KPOINT = 1 << 6
ORIGIN_M_FLIP = 1 << 7
ORIGIN_M_STEP = 1 << 8
ORIGIN_M_RESET = 1 << 9
ORIGIN_M_TOWARD_LP = 1 << 10
ORIGIN_M_VIOL = 1 << 11
ORIGIN_REPAIR = 1 << 12
ORIGIN_LP_DECODE = 1 << 13
ORIGIN_M_OBJ = 1 << 14
ORIGIN_X_AGREE = 1 << 15          # crossover "agreement" (last free bit of the uint16 origin)

ORIGIN_NAMES = {
    ORIGIN_SEED_RANDOM: "seed_random", ORIGIN_SEED_LP: "seed_lp", ORIGIN_INJECTED: "injected",
    ORIGIN_IMMIGRANT: "immigrant", ORIGIN_X_UNIFORM: "x_uniform", ORIGIN_X_BLOCK: "x_block",
    ORIGIN_X_KPOINT: "x_kpoint", ORIGIN_M_FLIP: "m_flip", ORIGIN_M_STEP: "m_step", ORIGIN_M_RESET: "m_reset",
    ORIGIN_M_TOWARD_LP: "m_toward_lp", ORIGIN_M_VIOL: "m_violation_guided", ORIGIN_REPAIR: "repair",
    ORIGIN_LP_DECODE: "lp_decode", ORIGIN_M_OBJ: "m_objective_guided", ORIGIN_X_AGREE: "x_agreement",
}


def origin_names(bits: int) -> list[str]:
    return [nm for b, nm in ORIGIN_NAMES.items() if bits & b]


@dataclass
class Population:
    X: object          # (P, n) on the device, eval dtype
    key: object        # (P,) float64 device
    viol: object       # (P,) float64 device
    feasible: object   # (P,) bool device
    rank: np.ndarray   # (P,) int8 host — NO_RANK (-1) until the first survival
    ids: np.ndarray    # (P,) int64 host
    parents: np.ndarray  # (P, 2) int64 host, -1 = none
    origin: np.ndarray   # (P,) uint16 host
    birth: np.ndarray    # (P,) int32 host — generation of birth
    anc_rank: np.ndarray  # (P, depth) uint8 host
    anc_ops: np.ndarray   # (P, depth) uint16 host
    strat: np.ndarray | None = None  # (P,) uint8 host — repair strategy index (None = all 0)

    def __post_init__(self):
        if self.strat is None:
            self.strat = np.zeros(self.ids.shape[0], dtype=np.uint8)
        else:
            self.strat = np.asarray(self.strat, dtype=np.uint8)
            if self.strat.shape != self.ids.shape:
                raise ValueError(f"strat has shape {self.strat.shape}, ids {self.ids.shape}")

    @property
    def size(self) -> int:
        return int(self.ids.shape[0])

    def take(self, bk: Backend, idx: np.ndarray, rank: np.ndarray | None = None) -> "Population":
        """Subset in the given order (idx: host int array); `rank` overrides the rank labels (after survival)."""
        di = bk.xp.asarray(idx)
        return Population(X=self.X[di], key=self.key[di], viol=self.viol[di], feasible=self.feasible[di],
                          rank=(self.rank[idx] if rank is None else np.asarray(rank, dtype=np.int8)),
                          ids=self.ids[idx], parents=self.parents[idx],
                          origin=self.origin[idx], birth=self.birth[idx], anc_rank=self.anc_rank[idx],
                          anc_ops=self.anc_ops[idx], strat=self.strat[idx])

    @staticmethod
    def concat(bk: Backend, parts: list["Population"]) -> "Population":
        xp = bk.xp
        parts = [p for p in parts if p is not None and p.size > 0]
        return Population(
            X=xp.concatenate([p.X for p in parts]), key=xp.concatenate([p.key for p in parts]),
            viol=xp.concatenate([p.viol for p in parts]), feasible=xp.concatenate([p.feasible for p in parts]),
            rank=np.concatenate([p.rank for p in parts]), ids=np.concatenate([p.ids for p in parts]),
            parents=np.concatenate([p.parents for p in parts]), origin=np.concatenate([p.origin for p in parts]),
            birth=np.concatenate([p.birth for p in parts]), anc_rank=np.concatenate([p.anc_rank for p in parts]),
            anc_ops=np.concatenate([p.anc_ops for p in parts]), strat=np.concatenate([p.strat for p in parts]))

    def host_scores(self, bk: Backend) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(key, viol, feasible) on the host — what survival needs; one stacked device-to-host copy."""
        xp = bk.xp
        h = bk.asnumpy(xp.stack([self.key, self.viol, self.feasible.astype(xp.float64)]))
        return h[0], h[1], h[2] > 0.5


class IdCounter:
    def __init__(self, start: int = 0):
        self.next = start

    def take(self, k: int) -> np.ndarray:
        out = np.arange(self.next, self.next + k, dtype=np.int64)
        self.next += k
        return out


_RB = np.zeros(4, dtype=np.uint8)          # index rank + 1: NO_RANK(-1) -> 0, R0 -> 1, R1 -> 2, R2 -> 4
_RB[R0 + 1], _RB[R1 + 1], _RB[R2 + 1] = RANK_BIT[R0], RANK_BIT[R1], RANK_BIT[R2]


def _rank_bits(rank: np.ndarray) -> np.ndarray:
    return _RB[np.asarray(rank, dtype=np.int64) + 1]


def gene_dtype_for(lb: np.ndarray, ub: np.ndarray, isint: np.ndarray, requested: str = "auto") -> np.dtype:
    """Storage dtype of X: int8 if every variable is integer with bounds in [-127, 127], int32 if integer within int32,
    else float64 (s100, 364k binaries, P = 256: 93 MB as int8 vs 746 MB as float64)."""
    if requested != "auto":
        return np.dtype(requested)
    if isint.size and isint.all() and np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)):
        if lb.min() >= -127 and ub.max() <= 127:
            return np.dtype(np.int8)
        if lb.min() >= -(2**31 - 1) and ub.max() <= 2**31 - 1:
            return np.dtype(np.int32)
    return np.dtype(np.float64)


def child_ancestry(depth: int, pa_rank: np.ndarray, pb_rank: np.ndarray, pa_origin: np.ndarray,
                   pb_origin: np.ndarray, pa_anc_rank: np.ndarray, pb_anc_rank: np.ndarray,
                   pa_anc_ops: np.ndarray, pb_anc_ops: np.ndarray, pa_src: np.ndarray | None = None,
                   pb_src: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Ancestry rows for k children from their two parents (use the same parent twice for mutation-only children).

    pa_rank/pb_rank: (k,) the parents' rank labels at mating time; *_src: optional extra source bits (SRC_*) of the
    parents themselves (e.g. a parent that came from the solver)."""
    k = pa_rank.shape[0]
    anc_r = np.zeros((k, depth), dtype=np.uint8)
    anc_o = np.zeros((k, depth), dtype=np.uint16)
    if depth == 0:
        return anc_r, anc_o
    anc_r[:, 0] = _rank_bits(pa_rank) | _rank_bits(pb_rank)
    if pa_src is not None:
        anc_r[:, 0] |= pa_src.astype(np.uint8)
    if pb_src is not None:
        anc_r[:, 0] |= pb_src.astype(np.uint8)
    anc_o[:, 0] = pa_origin | pb_origin
    if depth > 1:
        anc_r[:, 1:] = pa_anc_rank[:, :-1] | pb_anc_rank[:, :-1]
        anc_o[:, 1:] = pa_anc_ops[:, :-1] | pb_anc_ops[:, :-1]
    return anc_r, anc_o


def src_bits_of_origin(origin: np.ndarray) -> np.ndarray:
    """SRC_* bits implied by an individual's own origin (so a solver-injected parent marks its children)."""
    out = np.zeros(origin.shape, dtype=np.uint8)
    out[(origin & ORIGIN_INJECTED) != 0] |= SRC_INJECTED
    out[(origin & ORIGIN_SEED_RANDOM) != 0] |= SRC_SEED_RANDOM
    out[(origin & ORIGIN_SEED_LP) != 0] |= SRC_SEED_LP
    out[(origin & ORIGIN_IMMIGRANT) != 0] |= SRC_IMMIGRANT
    return out


class DupHasher:
    """Duplicate detection for rows of X: a random projection buckets rows, an exact comparison confirms.

    Integer coordinates are compared exactly (after rounding), continuous ones at an absolute resolution of 1e-9.
    The projection matrix is built once per run on the device."""

    def __init__(self, bk: Backend, n: int, salt: int = 12345):
        self.bk = bk
        rs = np.random.default_rng(salt)
        self.R = bk.xp.asarray(rs.uniform(1.0, 2.0, size=(n, 2)))

    def __call__(self, X) -> tuple[np.ndarray, np.ndarray]:
        """(dup, keep_of): dup[i] True if row i equals an earlier row; keep_of[i] = index of the kept equal row
        (i itself if kept). Earlier rows win, so callers order candidates population-first."""
        xp, bk = self.bk.xp, self.bk
        N = X.shape[0]
        Xq = X if X.dtype.kind in "iu" else xp.rint(X.astype(xp.float64, copy=False) * 1e9)
        step = max(1, (64 << 20) // max(1, 8 * X.shape[1]))     # rows per 64 MiB float64 chunk (s100: C x n is ~1.5 GB)
        h = (np.concatenate([bk.asnumpy(Xq[s:s + step].astype(xp.float64, copy=False) @ self.R)
                             for s in range(0, N, step)]) if N else np.zeros((0, 2)))
        order = np.lexsort((np.arange(N), h[:, 1], h[:, 0]))
        hs = h[order]
        scale = np.maximum(1.0, np.abs(hs))
        same = np.all(np.abs(np.diff(hs, axis=0)) <= 1e-9 * scale[1:], axis=1)
        dup = np.zeros(N, dtype=bool)
        keep_of = np.arange(N)
        i = 0
        while i < N:
            j = i
            while j + 1 < N and same[j]:
                j += 1
            if j > i:
                grp = np.sort(order[i:j + 1])
                rows = Xq[xp.asarray(grp)]
                # exact confirmation: assign each member to the first earlier member it equals
                for a_pos in range(1, grp.size):
                    if dup[grp[a_pos]]:
                        continue
                    eq = bk.asnumpy(xp.all(rows[a_pos + 1:] == rows[a_pos], axis=1)) if a_pos + 1 < grp.size else []
                    for off, e in enumerate(eq):
                        if e:
                            b = grp[a_pos + 1 + off]
                            if not dup[b]:
                                dup[b] = True
                                keep_of[b] = grp[a_pos]
                eq0 = bk.asnumpy(xp.all(rows[1:] == rows[0], axis=1))
                for off, e in enumerate(eq0):
                    b = grp[1 + off]
                    if e:
                        dup[b] = True
                        keep_of[b] = grp[0]
            i = j + 1
        # chains: point every duplicate at its root
        for b in np.flatnonzero(dup):
            r = keep_of[b]
            while dup[r]:
                r = keep_of[r]
            keep_of[b] = r
        return dup, keep_of


def duplicate_mask(bk: Backend, X, salt: int = 12345) -> np.ndarray:
    """Host bool mask: True for every row that equals an earlier row (exactly; see DupHasher)."""
    return DupHasher(bk, X.shape[1], salt)(X)[0]
