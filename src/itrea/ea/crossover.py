"""Crossover on the device: k children from k (parent A, parent B) pairs, batched per operator.

crossover(bk, dp, XA, XB, cfg, rng_dev, rng_host, mem_budget_bytes=256 MiB) -> (C, origin, touched)
    XA, XB   (k, n) device arrays, the same gene dtype (int8 / int32 / float64, `population.gene_dtype_for`).
    C        (k, n) NEW device array in that dtype (parents untouched); every gene is A's or B's, so children are
             inside the bounds whenever both parents are. Nothing is rounded, clipped or repaired here.
    origin   (k,) host uint16: the ORIGIN_X_* bit of the operator applied to each child, 0 for a plain copy of A.
    touched  (k, n) device bool — the genes the repair should focus on — or None when no child was crossed.
             A row that is all False means "not crossed, or parents identical": the repair must then treat that child
             as UNRESTRICTED, not as "nothing to do".

Why these operators. On structured MIPs (set partitioning, assignment, unit commitment) a child that takes each
disagreeing gene at random from either parent is close to random on the disagreement set, and almost every row through
that set breaks (design critique 2026-10-05). Established MIP recombination keeps what the parents agree on and
re-optimises the rest (Rothberg 2007, CPLEX polishing), so `agreement` is the default and reports its disagreement set
for the repair to work on.

Each child is crossed with probability `cfg.pc`; its operator is drawn from `cfg.weights` renormalised over the
operators that apply to the problem (`block` needs a row with a non-zero, `k_point` needs n >= 2):
    agreement  genes where A and B agree are kept; each disagreeing gene comes from A or B with probability 1/2.
               touched = the disagreement set. Origin: ORIGIN_X_AGREE once `itrea.ea.population` defines it — until
               then ORIGIN_X_UNIFORM, so agreement and uniform children share a bit (the constant below switches by
               itself when the core adds the bit).
    uniform    each gene from B with probability `cfg.p_swap`. touched = genes taken from B that differ from A.
    k_point    `cfg.k` distinct cut points over the variable order (k = 1 is part I's single-point crossover), segments
               alternate A, B, A, ... touched = genes taken from B that differ from A. The MPS column order carries no
               structure, so this is the part-I-faithful option, not a default.
    block      constraint-aware: `cfg.block_rows` rows per child, drawn uniformly WITHOUT replacement among the rows
               that contain at least one disagreeing variable (a row where the parents agree would make the operator a
               no-op); every variable of those rows (the row closure) comes from B. touched = those variables, which is
               what `cfg.block_closure` asks the repair to focus on. A pair that agrees everywhere gives a copy of A
               with an empty touched row. Masks come from two sparse products with the 0/1 pattern of A
               ((m, n) @ (n, kc) and (n, m) @ (m, kc)) — never a dense (kc, m) indicator times a dense matrix.
    none       copy of A.

Agreement of float genes means |a - b| <= 1e-9 (the resolution of the duplicate hash); integer dtypes compare exactly.
Children are processed per operator in chunks of children so the (chunk, n) and (m, chunk) temporaries stay inside
`mem_budget_bytes`; the outputs C and touched are outside that budget.
"""
from __future__ import annotations

import weakref

import numpy as np
import scipy.sparse as sps

from itrea.backend import Backend
from itrea.config import CrossoverCfg
from itrea.ea import population as _population
from itrea.ea.evaluate import DeviceProblem
from itrea.ea.population import ORIGIN_X_BLOCK, ORIGIN_X_KPOINT, ORIGIN_X_UNIFORM

ORIGIN_X_AGREE: int = getattr(_population, "ORIGIN_X_AGREE", ORIGIN_X_UNIFORM)
OPERATORS = ("agreement", "uniform", "k_point", "block", "none")
ORIGIN_OF = {"agreement": ORIGIN_X_AGREE, "uniform": ORIGIN_X_UNIFORM, "k_point": ORIGIN_X_KPOINT,
             "block": ORIGIN_X_BLOCK, "none": 0}
AGREE_TOL = 1e-9
DEFAULT_MEM_BUDGET = 256 * 1024**2


# ------------------------------------------------------------------------------------------- per-problem cache
_CACHE: dict[int, tuple[weakref.ref, dict]] = {}


def dp_cache(dp: DeviceProblem) -> dict:
    """A dict of derived structures (sparsity patterns, candidate lists) tied to one DeviceProblem and freed with it.

    Keyed by id() with a weak reference, because DeviceProblem is an unhashable dataclass owned by another module."""
    ent = _CACHE.get(id(dp))
    if ent is not None and ent[0]() is dp:
        return ent[1]
    key = id(dp)
    d: dict = {}
    _CACHE[key] = (weakref.ref(dp, lambda _r, k=key: _CACHE.pop(k, None)), d)
    return d


def _patterns(bk: Backend, dp: DeviceProblem):
    """(pattern (m, n), pattern^T (n, m)) of A as float32 0/1 CSR on the device."""
    c = dp_cache(dp)
    if "x_pattern" not in c:
        A = dp.prob.A
        pat = sps.csr_matrix((np.ones(A.nnz, dtype=np.float32), A.indices, A.indptr), shape=A.shape)
        c["x_pattern"] = bk.to_device_sparse(pat, dtype=np.float32)
        c["x_patternT"] = bk.to_device_sparse(pat.T.tocsr(), dtype=np.float32)
    return c["x_pattern"], c["x_patternT"]


# ------------------------------------------------------------------------------------------------- helpers
def _differ(xp, A, B):
    if A.dtype.kind in "iub":
        return A != B
    return ~(xp.abs(A - B) <= AGREE_TOL)          # NaN counts as a disagreement


def _applicable(name: str, dp: DeviceProblem) -> bool:
    if name == "block":
        return dp.m > 0 and dp.nnz > 0
    if name == "k_point":
        return dp.n >= 2
    return True


def _draw_operators(k: int, cfg: CrossoverCfg, dp: DeviceProblem, rng: np.random.Generator) -> np.ndarray:
    """Operator index (into OPERATORS) per child; "none" for children that are not crossed."""
    none = OPERATORS.index("none")
    ops = np.full(k, none, dtype=np.int64)
    names = [nm for nm in OPERATORS if cfg.weights.get(nm, 0.0) > 0 and _applicable(nm, dp)]
    if not names or k == 0:
        return ops
    w = np.array([cfg.weights[nm] for nm in names], dtype=np.float64)
    crossed = np.flatnonzero(rng.random(k) < cfg.pc)
    if crossed.size:
        pick = rng.choice(len(names), size=crossed.size, p=w / w.sum())
        ops[crossed] = np.array([OPERATORS.index(nm) for nm in names])[pick]
    return ops


def _chunk_size(name: str, n: int, m: int, itemsize: int, budget: int) -> int:
    """Children per chunk so that the temporaries of one chunk stay near `budget` (an estimate of every live
    temporary, checked with the CuPy pool at n = 120k, m = 47k)."""
    per = n * (3 * itemsize + 4 + 4)              # gathered A, B, result; float32 coin; bool masks
    if name == "k_point":
        per += n * 12                             # int32 cut marks, their int32 cumulative sum, the parity
    if name == "block":
        per += n * (4 + 4 + 1 + 1) + m * (4 + 4 + 4 + 1 + 1)   # D^T, closure product, masks; row counts, coins, picks
    return max(1, int(budget // max(per, 1)))


# ----------------------------------------------------------------------------------------------- operators
def _agreement(bk, dp, A, B, cfg, rng_dev, rng_host):
    xp = bk.xp
    D = _differ(xp, A, B)
    take_b = D & (rng_dev.random(A.shape, dtype=xp.float32) < 0.5)
    return xp.where(take_b, B, A), D


def _uniform(bk, dp, A, B, cfg, rng_dev, rng_host):
    xp = bk.xp
    take_b = rng_dev.random(A.shape, dtype=xp.float32) < cfg.p_swap
    return xp.where(take_b, B, A), take_b & _differ(xp, A, B)


def _distinct_sorted_cuts(rng: np.random.Generator, rows: int, kk: int, n: int) -> np.ndarray:
    """(rows, kk) sorted DISTINCT cut positions in 1..n-1 (a gene j is after a cut c when j >= c)."""
    cuts = np.sort(rng.integers(1, n, size=(rows, kk)), axis=1)
    if kk == 1:
        return cuts
    for _ in range(20):
        bad = np.flatnonzero(np.any(np.diff(cuts, axis=1) == 0, axis=1))
        if bad.size == 0:
            return cuts
        cuts[bad] = np.sort(rng.integers(1, n, size=(bad.size, kk)), axis=1)
    for r in np.flatnonzero(np.any(np.diff(cuts, axis=1) == 0, axis=1)):
        cuts[r] = np.sort(rng.choice(np.arange(1, n), size=kk, replace=False))
    return cuts


def _k_point(bk, dp, A, B, cfg, rng_dev, rng_host):
    xp = bk.xp
    kc, n = A.shape
    kk = min(int(cfg.k), n - 1)
    cuts = _distinct_sorted_cuts(rng_host, kc, kk, n)
    rows = np.repeat(np.arange(kc), kk)
    delta = xp.zeros((kc, n), dtype=xp.int32)
    delta[xp.asarray(rows), xp.asarray(cuts.ravel())] = 1          # cuts are distinct per row: no accumulation
    from_b = (xp.cumsum(delta, axis=1, dtype=xp.int32) & 1).astype(bool)
    del delta
    return xp.where(from_b, B, A), from_b & _differ(xp, A, B)


def _block(bk, dp, A, B, cfg, rng_dev, rng_host):
    xp = bk.xp
    kc = A.shape[0]
    pat, patT = _patterns(bk, dp)
    D = _differ(xp, A, B)
    rows_d = pat @ xp.ascontiguousarray(D.T, dtype=xp.float32)        # (m, kc): disagreeing variables per row
    # coins >= 1 on eligible rows, 0 elsewhere; take the block_rows largest per child (without replacement)
    score = xp.where(rows_d > 0, rng_dev.random(rows_d.shape, dtype=xp.float32) + 1.0, 0.0).astype(xp.float32)
    del rows_d
    R = xp.zeros(score.shape, dtype=xp.float32)
    cols = xp.arange(kc)
    for _ in range(int(cfg.block_rows)):
        j = xp.argmax(score, axis=0)
        ok = score[j, cols] > 0.5
        R[j[ok], cols[ok]] = 1.0
        score[j, cols] = 0.0
    del score
    T = (patT @ R) > 0                                                  # (n, kc): row closure of the chosen rows
    T = xp.ascontiguousarray(T.T)
    return xp.where(T, B, A), T


_OPS = {"agreement": _agreement, "uniform": _uniform, "k_point": _k_point, "block": _block}


# -------------------------------------------------------------------------------------------------- entry
def crossover(bk: Backend, dp: DeviceProblem, XA, XB, cfg: CrossoverCfg, rng_dev, rng_host: np.random.Generator,
              mem_budget_bytes: int = DEFAULT_MEM_BUDGET):
    """Children of the (XA[i], XB[i]) pairs; see the module docstring. Returns (C, origin, touched)."""
    xp = bk.xp
    if XA.ndim != 2 or XA.shape != XB.shape:
        raise ValueError(f"crossover: parents must be (k, n) arrays of one shape, got {XA.shape} and {XB.shape}")
    if XA.dtype != XB.dtype:
        raise TypeError(f"crossover: parents have different dtypes {XA.dtype} and {XB.dtype}")
    k, n = XA.shape
    C = XA.copy()
    origin = np.zeros(k, dtype=np.uint16)
    ops = _draw_operators(k, cfg, dp, rng_host)
    none = OPERATORS.index("none")
    if np.all(ops == none):
        return C, origin, None
    touched = xp.zeros((k, n), dtype=bool)
    for code, name in enumerate(OPERATORS):
        if name == "none":
            continue
        idx = np.flatnonzero(ops == code)
        if idx.size == 0:
            continue
        origin[idx] = ORIGIN_OF[name]
        chunk = _chunk_size(name, n, dp.m, XA.dtype.itemsize, mem_budget_bytes)
        for s in range(0, idx.size, chunk):
            di = xp.asarray(idx[s:s + chunk])
            Cc, Tc = _OPS[name](bk, dp, XA[di], XB[di], cfg, rng_dev, rng_host)
            C[di] = Cc
            touched[di] = Tc
            del Cc, Tc
    return C, origin, touched
