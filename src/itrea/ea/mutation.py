"""Mutation on the device, batched over children: one operator per child, a few genes each.

mutate(bk, dp, C, cfg, rng_dev, rng_host, lp_x=None, act=None, mem_budget_bytes=256 MiB, stats=None) -> origin
    C       (k, n) device array of children, mutated IN PLACE; its dtype (int8 / int32 / float64) is preserved.
    origin  (k,) host uint16: the ORIGIN_M_* bit of each child whose genes were changed by its operator (0 = none).
    lp_x    optional (n,) LP-relaxation point (host or device), canonical order — enables "toward_lp".
    act     optional (m, k) activities A C^T of the children as they are on entry — saves "violation_guided" an SpMM.
    stats   optional dict, filled with counters: children and genes per operator, violation_guided children that had
            no violated row (and were re-drawn).

Genes per child. Poisson(`cfg.genes`) — an ABSOLUTE number, so the same setting means the same disturbance on a
60-variable and on a 117k-variable instance (design critique 2026-10-05: a rate of 1 % meant ~1,175 genes on rail01
against ≤ 50 repair moves) — or Binomial(|ints|, `cfg.rate`) when `rate` is set; then at least `cfg.min_genes` and at
most the number of variables the operator can move. Fixed variables (lb == ub) are never chosen.

Operators (`cfg.weights`, renormalised over the operators that apply to this problem and call; none left -> "reset"):
    flip              binaries: x -> 1 - x.                                                   needs a free binary
    step              general integers: x -> x ± U{1..step_max}; the other direction if that leaves the bounds,
                      then clipped.                                                           needs a general integer
    reset             any free variable, integer or continuous: uniform in [lb, ub]; an infinite side is replaced by
                      the current value ∓ `cfg.reset_window`. The only operator that moves continuous genes.
    toward_lp         integers: randomised rounding of the LP value (floor, or ceil with probability = its
                      fractional part).                                                       needs lp_x
    violation_guided  I-TREA's Second Chance analogue: a violated row of THIS child is drawn uniformly, then an
                      integer variable of that row, which moves one unit in the direction that reduces the row's
                      violation (sign of the coefficient; up for a row below its lower side). A row counts as violated
                      above 1e-9 · max(1, |side|). A child with no violated row is re-drawn among the other operators
                      by weight (reset if no other has weight) — a feasible child would otherwise come out unchanged.
                      Needs a row with a non-zero and an integer.
    objective_guided  I-TREA's Forced Closure analogue: integer variables with cmin_j != 0, drawn with probability
                      ∝ |cmin_j|, move one unit in the improving direction -sign(cmin_j) (a binary goes to 0 if
                      cmin_j > 0, to 1 if cmin_j < 0); feasibility is left to the repair.     needs a costed integer

`rng_dev` is accepted for interface symmetry and not used: every draw is O(k · genes) and made on the host.

Every draw is (child, variable): a draw is rejected if the move would change nothing or, for violation_guided /
objective_guided, could not be made inside the bounds; the first `genes` distinct accepted variables of each child are
applied. With a large candidate set a child gets 4 draws (with replacement) per gene; with a small one (<= 256
variables, or genes >= half of it) a full random permutation of the set — weighted for objective_guided — so the draw
is exact sampling without replacement among the possible moves. In the first case a child whose possible moves are
rare (e.g. objective_guided when almost every variable already sits at its improving bound) may get fewer genes than
drawn; violation_guided, whose candidates depend on the child, always uses the 4-draw rule.

Batching. Draws, rejection and targets are host NumPy on O(k · genes) numbers; the device sees one gather of the
current values and one scatter of the new ones per operator. Only violation_guided touches (m, k) data on the device
(activities, the violated-row mask and its cumulative sum, used to draw "the r-th violated row of child c" with one
searchsorted), chunked over children inside `mem_budget_bytes`.

Afterwards EVERY child (mutated or not) has its integer genes rounded and clipped to their bounds, so mutation's output
is integral and in bounds whatever the parents were (a solver vector can carry 0.9999999); continuous genes are left as
they are except where "reset" moved them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from itrea.backend import Backend
from itrea.config import MutationCfg
from itrea.ea.crossover import dp_cache
from itrea.ea.evaluate import DeviceProblem
from itrea.ea.population import (ORIGIN_M_FLIP, ORIGIN_M_OBJ, ORIGIN_M_RESET, ORIGIN_M_STEP, ORIGIN_M_TOWARD_LP,
                                 ORIGIN_M_VIOL)

OPERATORS = ("flip", "step", "reset", "toward_lp", "violation_guided", "objective_guided")
ORIGIN_OF = {"flip": ORIGIN_M_FLIP, "step": ORIGIN_M_STEP, "reset": ORIGIN_M_RESET, "toward_lp": ORIGIN_M_TOWARD_LP,
             "violation_guided": ORIGIN_M_VIOL, "objective_guided": ORIGIN_M_OBJ}
TRIES = 4                       # tries per gene before a child gets fewer genes than drawn
PERM_MAX = 256                  # candidate sets up to this size are drawn as a full permutation per child
VIOL_TOL = 1e-9                 # violation_guided: row violated above VIOL_TOL * max(1, |side|)
DEFAULT_MEM_BUDGET = 256 * 1024**2


@dataclass
class _Static:
    """Host-side per-problem candidate sets (cached per DeviceProblem)."""
    n: int
    lo: np.ndarray        # effective bounds: integers ceil(lb), floor(ub); continuous as given
    hi: np.ndarray
    isint: np.ndarray
    movable: np.ndarray   # not fixed: hi > lo
    bins: np.ndarray      # movable binaries
    gints: np.ndarray     # movable general integers
    ints: np.ndarray      # movable integers
    free: np.ndarray      # movable variables (any type)
    obj: np.ndarray       # movable integers with cmin != 0
    obj_p: np.ndarray     # their draw probabilities ∝ |cmin|
    cmin: np.ndarray
    n_ints: int           # all integer variables (for the Binomial rate)


def _static(dp: DeviceProblem) -> _Static:
    c = dp_cache(dp)
    if "m_static" in c:
        return c["m_static"]
    p = dp.prob
    isint = np.asarray(p.isint, dtype=bool)
    lo = np.where(isint, np.ceil(p.lb - 1e-9), p.lb)
    hi = np.where(isint, np.floor(p.ub + 1e-9), p.ub)
    movable = hi > lo
    isbin = np.zeros(p.n, dtype=bool)
    isbin[p.bins] = True
    bins = np.flatnonzero(isbin & movable & (lo == 0) & (hi == 1))
    gints = np.flatnonzero(isint & ~isbin & movable)
    ints = np.flatnonzero(isint & movable)
    cmin = np.asarray(p.cmin, dtype=np.float64)
    obj = ints[cmin[ints] != 0]
    w = np.abs(cmin[obj])
    s = _Static(n=p.n, lo=lo, hi=hi, isint=isint, movable=movable, bins=bins, gints=gints, ints=ints,
                free=np.flatnonzero(movable), obj=obj, obj_p=(w / w.sum()) if w.size else w, cmin=cmin,
                n_ints=int(isint.sum()))
    c["m_static"] = s
    return s


def _candidates(op: str, S: _Static) -> np.ndarray:
    return {"flip": S.bins, "step": S.gints, "reset": S.free, "toward_lp": S.ints, "violation_guided": S.ints,
            "objective_guided": S.obj}[op]


def _applicable(op: str, S: _Static, dp: DeviceProblem, lp_x) -> bool:
    if op == "toward_lp" and lp_x is None:
        return False
    if op == "violation_guided" and (dp.m == 0 or dp.nnz == 0):
        return False
    return _candidates(op, S).size > 0


def _take(bk: Backend, v, idx: np.ndarray) -> np.ndarray:
    """v[idx] on the host as float64, for a host or device vector v."""
    if isinstance(v, np.ndarray):
        return v[idx].astype(np.float64)
    return bk.asnumpy(v[bk.xp.asarray(idx)]).astype(np.float64)


def _gather(bk: Backend, C, cc: np.ndarray, jj: np.ndarray) -> np.ndarray:
    if cc.size == 0:
        return np.empty(0)
    xp = bk.xp
    return bk.asnumpy(C[xp.asarray(cc), xp.asarray(jj)]).astype(np.float64)


def _keep_first(cc: np.ndarray, jj: np.ndarray, valid: np.ndarray, G: np.ndarray, n: int) -> np.ndarray:
    """Positions of the first G[c] DISTINCT valid variables of each child c, in draw order (cc grouped ascending)."""
    pos = np.flatnonzero(valid)
    if pos.size == 0:
        return pos
    _, first = np.unique(cc[pos].astype(np.int64) * n + jj[pos], return_index=True)
    pos = pos[np.sort(first)]
    c = cc[pos]
    rank = np.arange(pos.size) - np.searchsorted(c, c, side="left")
    return pos[rank < G[c]]


def _draw_static(op: str, children: np.ndarray, Gc: np.ndarray, S: _Static, rng: np.random.Generator
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(child, variable) draw sequences for an operator with a fixed candidate set, grouped by child ascending.

    Large set, few genes: Gc * TRIES draws with replacement (duplicates are dropped later). Small set (<= PERM_MAX)
    or genes >= half the set: a full random permutation of the set per child — weighted by |cmin| for
    objective_guided via exponential keys (Efraimidis & Spirakis 2006) — so every candidate can be reached and
    "first G valid" is exact sampling without replacement."""
    cand = _candidates(op, S)
    p = S.obj_p if op == "objective_guided" else None
    nc = cand.size
    perm = (nc <= PERM_MAX) | (2 * Gc[children] >= nc)
    ch_r, ch_p = children[~perm], children[perm]
    cc_r = np.repeat(ch_r, Gc[ch_r] * TRIES)
    jj_r = cand[rng.choice(nc, size=cc_r.size, p=p) if p is not None else rng.integers(0, nc, size=cc_r.size)]
    if ch_p.size == 0:
        return cc_r, jj_r
    keys = rng.random((ch_p.size, nc))
    if p is not None:
        keys = np.log(np.maximum(keys, 1e-300)) / p[None, :]        # larger = earlier; = -Exp(1) / w
        order = np.argsort(-keys, axis=1, kind="stable")
    else:
        order = np.argsort(keys, axis=1)
    cc = np.concatenate([cc_r, np.repeat(ch_p, nc)])
    jj = np.concatenate([jj_r, cand[order].ravel()])
    srt = np.argsort(cc, kind="stable")
    return cc[srt], jj[srt]


def _targets(op: str, x: np.ndarray, jj: np.ndarray, S: _Static, cfg: MutationCfg, rng: np.random.Generator,
             lp_vals: np.ndarray | None = None, direction: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """New values and validity for draws (current values x of variables jj)."""
    lo, hi, isint = S.lo[jj], S.hi[jj], S.isint[jj]
    base = np.where(np.isfinite(x), x, 0.0)
    xr = np.where(isint, np.rint(base), base)
    L = x.size
    strict = None                                   # move must be possible inside the bounds without clipping
    if op == "flip":
        t = 1.0 - xr
    elif op == "step":
        s = rng.integers(1, cfg.step_max + 1, size=L).astype(np.float64)
        sg = np.where(rng.random(L) < 0.5, -1.0, 1.0)
        t = xr + sg * s
        t = np.where((t < lo) | (t > hi), xr - sg * s, t)
    elif op == "reset":
        w = float(cfg.reset_window)
        a = np.where(np.isfinite(lo), lo, xr - w)
        b = np.where(np.isfinite(hi), hi, xr + w)
        u = rng.random(L)
        ai, bi = np.ceil(a), np.floor(b)
        ti = np.minimum(ai + np.floor(u * (bi - ai + 1.0)), bi)
        t = np.where(isint, ti, a + u * (b - a))
    elif op == "toward_lp":
        f = np.floor(lp_vals)
        t = f + (rng.random(L) < (lp_vals - f))
    elif op == "objective_guided":
        t = xr - np.sign(S.cmin[jj])
        strict = t
    elif op == "violation_guided":
        t = xr + direction
        strict = t
    else:
        raise ValueError(op)
    tc = np.clip(t, lo, hi)
    valid = np.isfinite(tc) & (tc != x)
    if strict is not None:
        valid &= tc == strict
    return tc, valid


# ------------------------------------------------------------------------------------------- violation_guided
def _violation_draws(bk: Backend, dp: DeviceProblem, C, children: np.ndarray, G: np.ndarray, act, S: _Static,
                     rng: np.random.Generator, budget: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(cc, jj, direction, no_violation_children) for violation_guided children; draws grouped by child ascending."""
    xp = bk.xp
    m, n = dp.m, dp.n
    per = m * (8 + 8 + 4) + n * (8 + 2 * C.dtype.itemsize + 8)   # act, cumsum, masks; gathered C, its float cast
    chunk = max(1, int(budget // per))
    indptr, indices, data = dp.A.indptr, dp.A.indices, dp.A.data
    out_c, out_j, out_d, none = [], [], [], []
    for s in range(0, children.size, chunk):
        ii = children[s:s + chunk]
        di = xp.asarray(ii)
        kc = ii.size
        if act is not None:
            a = act[:, di]
        else:
            a = dp.A @ xp.ascontiguousarray(C[di].T).astype(np.dtype(dp.dtype), copy=False)
        below = (dp.rl[:, None] - a) > VIOL_TOL * dp.scale_l[:, None]
        above = (a - dp.ru[:, None]) > VIOL_TOL * dp.scale_u[:, None]
        del a
        Vt = xp.ascontiguousarray((below | above).T)                # (kc, m)
        counts = bk.asnumpy(Vt.sum(axis=1)).astype(np.int64)
        has = counts > 0
        none.append(ii[~has])
        reps = np.where(has, G[ii] * TRIES, 0)
        loc = np.repeat(np.arange(kc), reps)
        if loc.size == 0:
            continue
        base = np.concatenate([[0], np.cumsum(counts)[:-1]])
        r = np.minimum(np.floor(rng.random(loc.size) * counts[loc]).astype(np.int64), counts[loc] - 1)
        target = base[loc] + r + 1                                  # the (r+1)-th violated row of child loc
        cs = xp.cumsum(Vt.reshape(-1), dtype=xp.int64)
        del Vt
        flat = xp.searchsorted(cs, xp.asarray(target), side="left")
        del cs
        dloc = xp.asarray(loc)
        row = flat - dloc * m
        is_below = below[row, dloc]
        start = indptr[row].astype(xp.int64)
        length = indptr[row + 1].astype(xp.int64) - start
        e = start + xp.floor(xp.asarray(rng.random(loc.size)) * length).astype(xp.int64)
        e = xp.minimum(e, max(int(indices.shape[0]) - 1, 0))
        ok = length > 0
        j = xp.where(ok, indices[e], 0).astype(xp.int64)
        sgn = xp.where(ok, xp.sign(data[e]), 0)
        d = xp.where(is_below, sgn, -sgn)
        out_c.append(ii[loc])
        out_j.append(bk.asnumpy(j))
        out_d.append(bk.asnumpy(d).astype(np.float64))
    cat = (lambda parts, dt: np.concatenate(parts).astype(dt) if parts else np.empty(0, dt))
    return cat(out_c, np.int64), cat(out_j, np.int64), cat(out_d, np.float64), cat(none, np.int64)


# ------------------------------------------------------------------------------------------------- normalise
def _normalise_integers(bk: Backend, dp: DeviceProblem, C, S: _Static, budget: int) -> None:
    """Round every integer gene and clip it to its bounds, in place (all children)."""
    xp = bk.xp
    if S.n_ints == 0 or C.shape[0] == 0:
        return
    c = dp_cache(dp)
    if C.dtype.kind in "iu":
        key = ("m_int_bounds", C.dtype.str)
        if key not in c:
            info = np.iinfo(C.dtype)
            lo = np.clip(np.where(np.isfinite(S.lo), S.lo, info.min), info.min, info.max)
            hi = np.clip(np.where(np.isfinite(S.hi), S.hi, info.max), info.min, info.max)
            lo = np.where(S.isint, lo, info.min)                      # a forced int dtype on a continuous column:
            hi = np.where(S.isint, hi, info.max)                      # leave it (only integer genes are normalised)
            c[key] = (xp.asarray(lo.astype(C.dtype)), xp.asarray(hi.astype(C.dtype)))
        lo, hi = c[key]
        xp.clip(C, lo, hi, out=C)
        return
    if "m_float_bounds" not in c:
        c["m_float_bounds"] = (xp.asarray(np.where(S.isint, S.lo, -np.inf)),
                               xp.asarray(np.where(S.isint, S.hi, np.inf)), xp.asarray(S.isint))
    lo, hi, isint = c["m_float_bounds"]
    all_int = S.n_ints == S.n
    chunk = max(1, int(budget // max(1, 32 * S.n)))
    for s in range(0, C.shape[0], chunk):
        X = C[s:s + chunk]                                            # a view: writes go to C
        R = xp.clip(xp.rint(X), lo, hi)
        X[...] = R if all_int else xp.where(isint, R, X)


# ------------------------------------------------------------------------------------------------------ entry
def mutate(bk: Backend, dp: DeviceProblem, C, cfg: MutationCfg, rng_dev, rng_host: np.random.Generator,
           lp_x=None, act=None, mem_budget_bytes: int = DEFAULT_MEM_BUDGET, stats: dict | None = None) -> np.ndarray:
    """Mutate the children C in place; returns their ORIGIN_M_* bits. See the module docstring."""
    xp = bk.xp
    if C.ndim != 2 or C.shape[1] != dp.n:
        raise ValueError(f"mutate: C must be (k, {dp.n}), got {C.shape}")
    k = int(C.shape[0])
    origin = np.zeros(k, dtype=np.uint16)
    S = _static(dp)
    st = stats if stats is not None else {}
    if k == 0:
        return origin
    if lp_x is not None and not hasattr(lp_x, "shape"):
        lp_x = np.asarray(lp_x, dtype=np.float64)
    avail = [op for op in OPERATORS if _applicable(op, S, dp, lp_x)]
    weights = {op: float(cfg.weights.get(op, 0.0)) for op in avail}
    if sum(weights.values()) <= 0:
        weights = {"reset": 1.0} if "reset" in avail else {}
    ops = np.full(k, -1, dtype=np.int64)                               # -1: no operator (nothing applicable)
    if weights:
        names = list(weights)
        w = np.array([weights[o] for o in names])
        ops = np.array([OPERATORS.index(o) for o in names])[rng_host.choice(len(names), size=k, p=w / w.sum())]
    # genes per child
    if cfg.rate is not None:
        G = rng_host.binomial(S.n_ints if S.n_ints else S.n, cfg.rate, size=k)
    else:
        G = rng_host.poisson(cfg.genes, size=k)
    G = np.maximum(G, cfg.min_genes).astype(np.int64)
    lp_host = None
    if lp_x is not None:
        lp_host = lp_x if isinstance(lp_x, np.ndarray) else None

    vg = OPERATORS.index("violation_guided")
    order = [vg] + [i for i in range(len(OPERATORS)) if i != vg]
    for code in order:
        op = OPERATORS[code]
        children = np.flatnonzero(ops == code)
        if children.size == 0:
            continue
        Gc = np.zeros(k, dtype=np.int64)
        Gc[children] = np.minimum(G[children], _candidates(op, S).size)
        if op == "violation_guided":
            cc, jj, direction, no_viol = _violation_draws(bk, dp, C, children, Gc, act, S, rng_host,
                                                          mem_budget_bytes)
            st["violation_guided_no_violation"] = st.get("violation_guided_no_violation", 0) + int(no_viol.size)
            if no_viol.size:                                            # feasible children: another operator
                others = {o: wt for o, wt in weights.items() if o != op and wt > 0}
                if not others and "reset" in avail:
                    others = {"reset": 1.0}
                if others:
                    nm = list(others)
                    p = np.array([others[o] for o in nm])
                    ops[no_viol] = np.array([OPERATORS.index(o) for o in nm])[
                        rng_host.choice(len(nm), size=no_viol.size, p=p / p.sum())]
                else:
                    ops[no_viol] = -1
            x = _gather(bk, C, cc, jj)
            valid_var = S.isint[jj] & S.movable[jj] & (direction != 0)
            t, valid = _targets(op, x, jj, S, cfg, rng_host, direction=direction)
            valid &= valid_var
        else:
            cc, jj = _draw_static(op, children, Gc, S, rng_host)
            x = _gather(bk, C, cc, jj)
            lp_vals = None
            if op == "toward_lp":
                lp_vals = lp_host[jj].astype(np.float64) if lp_host is not None else _take(bk, lp_x, jj)
            t, valid = _targets(op, x, jj, S, cfg, rng_host, lp_vals=lp_vals)
        keep = _keep_first(cc, jj, valid, Gc, S.n)
        if keep.size:
            ck, jk = cc[keep], jj[keep]
            C[xp.asarray(ck), xp.asarray(jk)] = xp.asarray(t[keep].astype(C.dtype))
            changed = np.unique(ck)
            origin[changed] |= np.uint16(ORIGIN_OF[op])
            st[f"genes_{op}"] = st.get(f"genes_{op}", 0) + int(keep.size)
        st[f"children_{op}"] = st.get(f"children_{op}", 0) + int(
            children.size - (no_viol.size if op == "violation_guided" else 0))
    _normalise_integers(bk, dp, C, S, mem_budget_bytes)
    return origin
