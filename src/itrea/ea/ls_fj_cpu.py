"""Feasibility-jump local search on one CPU core (Numba) — the CPU reference implementation.

Spec: vault `10-design/Feasibility jump local search.md`. Two jobs:
- arm A4's repair operator (the EA on the CPU, the control that isolates the GPU's share of any effect), and
- the single-core throughput reference that the GPU kernel (`ls_fj_gpu.FJGpu`) is measured against (D-002 E1 gate (b)).
Both jobs need the SAME move rule as the GPU kernel, so this file mirrors `ls_fj_gpu` decision for decision (listed
below); only the execution model differs — here one individual after another, each a plain sequential loop.

Per individual: working copy x (float64), activities act = A_aug x, row weights w (1.0 at the start of every call —
the individuals are new children each generation), the violated-row set as an index list + position array, and the
total weighted violation F (for the stall rule). A_aug is A plus, under `objective_mode: cutoff_row`, the objective row
`cmin·x <= rhs` (row m; its upper side is +inf while no incumbent is known, so it never counts as violated), which is
therefore selected, weighted, bumped and updated exactly like a constraint row.

One step (rules shared with the GPU kernel):
  0. stop if no row is violated (feasible), else if `max_steps` moves were made (budget), else if `stall_steps`
     consecutive steps did not lower the best F of this call by more than 1e-12 relative (stall), else if the call's
     lane-step horizon is reached (`run(horizon=h)`, the repair clock: status 4 = ST_HORIZON, the current x is kept —
     checked last, as the GPU kernel's end-of-slice re-check does before the host cuts the remaining lanes);
  1. pick a violated row: all of them when there are <= 32, else 32 sampled uniformly with replacement; then one of the
     sample with probability proportional to w_i·v_i (the spec's "bounded sample" option, exact when nviol <= 32);
  2. score every variable of that row (optionally a random cyclic window of `row_scan_cap` of them — off by default,
     the GPU kernel scans the whole row): candidate values are the flip for binaries; x±1, the finite bounds and the
     rounded (floor and ceil) tightening values for general integers; the finite bounds and the tightening values for
     continuous variables. Tightening values come from the column's violated rows — the chosen row first, then the
     others by largest w_i·v_i — at most (max_candidates-4)/2 rows for integers and max_candidates-2 for continuous
     variables. Candidates are clipped to the bounds; a candidate equal to the current value is not a move. Score of
     a value t: sum over the column's rows of w_i·[v_i(act_i + a_ij (t - x_j)) - v_i(act_i)] (+ obj_weight·cmin_j·Δ /
     max|cmin| in `weighted` mode). Best value per variable (first strict minimum in candidate order), then best
     variable (ties by (j + rot) mod n with a random rotation per step);
  3. improving (score < -1e-12): apply it — x_j, the column's activities, violated-set membership, F;
  4. otherwise a local minimum: every violated row's weight grows by `weight_bump`; with probability `p_random_walk`
     (default 0, a knob not yet in RepairCfg) the least-bad move is applied anyway.
Row violation v_i(a) = (max(rl_i - a, 0) + max(a - ru_i, 0)) / s_i with s_i the `viol_norm` divisor of the population's
V (row 2-norm or max(1,|rhs|); 1 for "count") — or, with `row_scales` (the self-adaptive repair strategy), the row of
the individual's own scale in `AugmentedRows.rs_tab`. A row is in the violated set when it misses a side by more than
min(viol_tol·max(1, |side|), viol_cap) = min(1e-9·max(1,|side|), 1e-7): far inside the 1e-6 test of every partner
mode (SCIP relative, HiGHS absolute), so "finished feasible" here implies `evaluate`-feasible.

Objective cutoff (`set_cutoff`): rhs = inc - key_offset - delta, delta = max(imp_tol·max(1,|inc|), g), g = 1 when the
objective is integral (every cost on an integer variable with an integer coefficient; the side is then rounded down
with a 1e-6 allowance for solver-side noise). For a NON-integral objective the side is lowered by a further
2·min(1e-9·max(1,|b|), 1e-7) (b = inc - key_offset): the violated-set tolerance would otherwise accept a point at
the incumbent's own value as "satisfied", and a feasible finish would not be strictly better (`ls_fj_gpu` uses the
same rule).

Integers stay integral and inside [ceil(lb), floor(ub)]; every gene is first projected (non-finite -> 0, integers
rounded half-to-even, clipped to the bounds), so a feasible, integral, in-bounds input with an inactive cutoff leaves
the call bit-for-bit unchanged. Each individual's activities are recomputed from scratch at the end and the largest
relative drift is returned (`act_drift`) — the test of the incremental bookkeeping.

Randomness: numba's generator, re-seeded per individual from (seed, call counter, global index), so a result does not
depend on the batch composition and a fixed seed reproduces a run exactly.
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sps
from numba import njit

from itrea.config import RepairCfg
from itrea.ea.evaluate import DeviceProblem, own_row_scale, row_scale_table

# variable classes (same codes as ls_fj_gpu)
VT_CONT, VT_BIN, VT_INT, VT_FIXED = 0, 1, 2, 3
# finishing status per individual (same codes as ls_fj_gpu; 0 = still active, never returned)
ST_FEASIBLE, ST_STALL, ST_BUDGET, ST_HORIZON = 1, 2, 3, 4
NO_HORIZON = 1 << 62   # max_lane_steps without a horizon (never reached)

MAX_CAND = 32          # capacity of the candidate arrays (runtime max_candidates is clipped to [4, MAX_CAND])
MAX_K = MAX_CAND - 2   # capacity of the tightening-row list
SAMPLE = 32            # violated rows sampled per step
IMPROVING = -1e-12     # a move is improving if its score is below this
VIOL_TOL = 1e-9        # violated-set membership: miss > min(VIOL_TOL * max(1, |side|), VIOL_CAP)
VIOL_CAP = 1e-7

_clock = time.perf_counter   # the call deadline's clock (`run(cap_s=...)`; tests replace it)


# ---------------------------------------------------------------------------------------------------- host side
def violation_tolerance(side: float) -> float:
    """The violated-set tolerance of a row side (the same rule in every implementation)."""
    return min(VIOL_TOL * max(1.0, abs(side)), VIOL_CAP)


def var_types(prob) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vtype, lb, ub) with integer bounds tightened to ceil(lb) / floor(ub); binaries = integers with bounds {0, 1};
    fixed = lb >= ub (including integers whose rounded bounds meet)."""
    isint = np.asarray(prob.isint, dtype=bool)
    lb = np.asarray(prob.lb, dtype=np.float64).copy()
    ub = np.asarray(prob.ub, dtype=np.float64).copy()
    with np.errstate(invalid="ignore"):
        lb[isint] = np.ceil(lb[isint] - 1e-9)
        ub[isint] = np.floor(ub[isint] + 1e-9)
    vt = np.full(prob.n, VT_CONT, dtype=np.int8)
    vt[isint] = VT_INT
    vt[isint & (lb == 0) & (ub == 1)] = VT_BIN
    vt[lb >= ub] = VT_FIXED
    return vt, lb, ub


def viol_norm_mode(dp: DeviceProblem) -> tuple[str, np.ndarray]:
    """("l2" | "rhs" | "count", per-row divisor s_i) as used by the population's V (`evaluate.own_row_scale`)."""
    return own_row_scale(dp)


def objective_is_integral(prob) -> bool:
    """g = 1 in the cutoff: every non-zero cost sits on an integer variable with an integer coefficient."""
    cmin = np.asarray(prob.cmin, dtype=np.float64)
    ci = cmin != 0
    return bool(np.any(ci) and np.all(prob.isint[ci]) and np.all(cmin[ci] == np.round(cmin[ci])))


def cutoff_side(inc_key: float | None, key_offset: float, imp_tol: float, integral: bool) -> float:
    """Upper side of the cutoff row cmin·x <= side for the incumbent key `inc_key` (+inf = inactive). See the module
    docstring for the rule; shared by FJCpu and DenseLS."""
    if inc_key is None or not math.isfinite(float(inc_key)):
        return math.inf
    inc = float(inc_key)
    b = inc - float(key_offset)
    if integral:
        delta = max(imp_tol * max(1.0, abs(inc)), 1.0)
        return float(math.floor(b - delta + 1e-6))
    delta = imp_tol * max(1.0, abs(inc))
    return b - delta - 2.0 * violation_tolerance(b)


class AugmentedRows:
    """A with the objective cutoff row appended (row m) when `objective_mode == "cutoff_row"` and some cost is
    non-zero; host arrays (CSR and CSC), row sides and the row divisors s_i. `set_cutoff_side` rewrites row m.

    Row scales (`scales`, default: `dp`'s own — the self-adaptive repair strategy passes several): `rs_tab` (R, m_aug)
    holds one row of divisors per scale, in the order of `scales` (`evaluate.row_scale_table`; the cutoff row's entry
    is ||cmin||_2 for "l2", max(1, |side|) for "rhs" once armed, 1 otherwise); `rs` is `rs_tab[0]` (a view) and
    `viol_norm` its mode. The sides `rl` / `ru` are the same for every scale."""

    def __init__(self, dp: DeviceProblem, cfg: RepairCfg, scales=None):
        prob = dp.prob
        cmin = np.asarray(prob.cmin, dtype=np.float64)
        self.m = prob.m
        self.has_cut = cfg.objective_mode == "cutoff_row" and bool(np.any(cmin != 0))
        self.scales, tab = row_scale_table(dp, scales)
        self.viol_norm = self.scales[0]
        A = sps.csr_matrix(prob.A, dtype=np.float64)
        rl = np.asarray(prob.rl, dtype=np.float64).copy()
        ru = np.asarray(prob.ru, dtype=np.float64).copy()
        if self.has_cut:
            A = sps.vstack([A, sps.csr_matrix(cmin.reshape(1, -1))], format="csr")
            rl = np.append(rl, -np.inf)
            ru = np.append(ru, np.inf)
            cn = float(np.linalg.norm(cmin))
            self.cut_l2 = cn if cn > 0 else 1.0
            cut = np.array([[self.cut_l2 if md == "l2" else 1.0] for md in self.scales])
            tab = np.hstack([tab, cut])
        A.sort_indices()
        C = A.tocsc()
        C.sort_indices()
        self.A, self.C = A, C
        self.rl, self.ru = rl, ru
        self.rs_tab = np.ascontiguousarray(tab, dtype=np.float64)
        self.rs = self.rs_tab[0]
        self.m_aug = A.shape[0]
        self.integral = objective_is_integral(prob) if self.has_cut else False

    def set_cutoff_side(self, side: float) -> None:
        if not self.has_cut:
            return
        self.ru[self.m] = side
        for r, md in enumerate(self.scales):
            if md == "rhs":
                self.rs_tab[r, self.m] = max(1.0, abs(side)) if math.isfinite(side) else 1.0


def weighted_objective(prob, cfg: RepairCfg) -> tuple[np.ndarray, int]:
    """Per-variable score term under `objective_mode: weighted`: obj_weight·cmin_j / max|cmin| (else zeros)."""
    cmin = np.asarray(prob.cmin, dtype=np.float64)
    if cfg.objective_mode == "weighted":
        cmax = float(np.max(np.abs(cmin))) if cmin.size else 0.0
        if cmax > 0:
            return cfg.obj_weight * cmin / cmax, 1
    return np.zeros(max(prob.n, 1)), 0


# ------------------------------------------------------------------------------------------------- numba kernel
@njit(cache=True)
def _rowviol(a, lo, hi, sc):
    v = 0.0
    if a < lo:
        v += lo - a
    if a > hi:
        v += a - hi
    return v / sc


@njit(cache=True)
def _isviol(a, lo, hi, vtol, vcap):
    if a < lo and lo - a > min(vtol * max(1.0, abs(lo)), vcap):
        return True
    if a > hi and a - hi > min(vtol * max(1.0, abs(hi)), vcap):
        return True
    return False


@njit(cache=True)
def _eval_var(j, r, x, col_ptr, row_idx, col_val, rl, ru, rs, lb, ub, vtype, cobj, use_obj, act, w, max_cand,
              vtol, vcap, kw, kt, cand, sc):
    """Best candidate value of variable j (chosen row r): (score incl. objective term, violation part, value).
    score = +inf if j cannot move."""
    inf = np.inf
    ty = vtype[j]
    if ty == VT_FIXED:
        return inf, 0.0, 0.0
    xj = x[j]
    lbj = lb[j]
    ubj = ub[j]
    c0 = col_ptr[j]
    c1 = col_ptr[j + 1]
    nc = 0
    if ty == VT_BIN:
        cand[0] = 1.0 - xj
        nc = 1
    else:
        K = (max_cand - 4) // 2 if ty == VT_INT else max_cand - 2
        if K < 1:
            K = 1
        if K > MAX_K:
            K = MAX_K
        nk = 0
        for p in range(c0, c1):
            i = row_idx[p]
            a = act[i]
            lo = rl[i]
            hi = ru[i]
            if not _isviol(a, lo, hi, vtol, vcap):
                continue
            side = lo if a < lo else hi
            t = xj + (side - a) / col_val[p]
            wv = inf if i == r else w[i] * _rowviol(a, lo, hi, rs[i])
            if nk < K:
                q = nk
                nk += 1
            elif wv > kw[K - 1]:
                q = K - 1
            else:
                continue
            while q > 0 and kw[q - 1] < wv:
                kw[q] = kw[q - 1]
                kt[q] = kt[q - 1]
                q -= 1
            kw[q] = wv
            kt[q] = t
        if ty == VT_INT:
            cand[nc] = xj - 1.0
            nc += 1
            cand[nc] = xj + 1.0
            nc += 1
            if lbj > -inf:
                cand[nc] = lbj
                nc += 1
            if ubj < inf:
                cand[nc] = ubj
                nc += 1
            for q in range(nk):
                if nc + 2 > MAX_CAND:
                    break
                cand[nc] = math.floor(kt[q] + 1e-9)
                nc += 1
                cand[nc] = math.ceil(kt[q] - 1e-9)
                nc += 1
        else:
            if lbj > -inf:
                cand[nc] = lbj
                nc += 1
            if ubj < inf:
                cand[nc] = ubj
                nc += 1
            for q in range(nk):
                if nc >= MAX_CAND:
                    break
                cand[nc] = kt[q]
                nc += 1
    # clip, drop no-moves; from here cand[c] holds the target value and sc[c] its score
    for c in range(nc):
        t = min(max(cand[c], lbj), ubj)
        cand[c] = t
        sc[c] = inf if t == xj else 0.0
    for p in range(c0, c1):
        i = row_idx[p]
        a = act[i]
        av = col_val[p]
        lo = rl[i]
        hi = ru[i]
        s = rs[i]
        wi = w[i]
        v0 = _rowviol(a, lo, hi, s)
        for c in range(nc):
            if sc[c] < inf:
                sc[c] += wi * (_rowviol(a + av * (cand[c] - xj), lo, hi, s) - v0)
    cj = cobj[j] if use_obj else 0.0
    bs = inf
    bdv = 0.0
    bt = 0.0
    for c in range(nc):
        if not (sc[c] < inf):
            continue
        tot = sc[c] + cj * (cand[c] - xj)
        if tot < bs:
            bs = tot
            bdv = sc[c]
            bt = cand[c]
    return bs, bdv, bt


@njit(cache=True)
def _move_dviol(j, d, col_ptr, row_idx, col_val, rl, ru, rs, act, w):
    """Change of the weighted violation if x_j moves by d (current weights)."""
    s = 0.0
    for p in range(col_ptr[j], col_ptr[j + 1]):
        i = row_idx[p]
        a = act[i]
        s += w[i] * (_rowviol(a + col_val[p] * d, rl[i], ru[i], rs[i]) - _rowviol(a, rl[i], ru[i], rs[i]))
    return s


@njit(cache=True)
def _fj_batch(X, seeds, row_ptr, col_idx, row_val, col_ptr, row_idx, col_val, rl, ru, rs, lb, ub, isint, vtype,
              cobj, use_obj, max_moves, stall_steps, max_cand, bump, p_rw, vtol, vcap, row_cap, max_lane_steps,
              status, lane_moves, lane_steps, lane_lmin, lane_drift, keep, ACT, WGT, NVIOL):
    k, n = X.shape
    m = rl.shape[0]
    inf = np.inf
    x = np.empty(n)
    act = np.empty(m)
    w = np.empty(m)
    vlist = np.empty(m, np.int64)
    vpos = np.empty(m, np.int64)
    kw = np.empty(MAX_K)
    kt = np.empty(MAX_K)
    cand = np.empty(MAX_CAND)
    sc = np.empty(MAX_CAND)
    srow = np.empty(SAMPLE, np.int64)
    spre = np.empty(SAMPLE)
    for b in range(k):
        np.random.seed(seeds[b])
        # projection: non-finite -> 0, integers rounded half-to-even, clip to the (integer-tightened) bounds
        for j in range(n):
            v = float(X[b, j])
            if not np.isfinite(v):
                v = 0.0
            if isint[j]:
                v = np.rint(v)
            v = min(max(v, lb[j]), ub[j])
            x[j] = v
        F = 0.0
        for i in range(m):
            a = 0.0
            for p in range(row_ptr[i], row_ptr[i + 1]):
                a += row_val[p] * x[col_idx[p]]
            act[i] = a
            w[i] = 1.0
            F += _rowviol(a, rl[i], ru[i], rs[i])
        nv = 0
        for i in range(m):
            if _isviol(act[i], rl[i], ru[i], vtol, vcap):
                vpos[i] = nv
                vlist[nv] = i
                nv += 1
            else:
                vpos[i] = -1
        Fbest = F
        moves = 0
        steps = 0
        lmin = 0
        since = 0
        stt = 0
        while True:
            # 0. stop rules
            if nv == 0:
                stt = ST_FEASIBLE
                break
            if moves >= max_moves:
                stt = ST_BUDGET
                break
            if since >= stall_steps:
                stt = ST_STALL
                break
            if steps >= max_lane_steps:
                stt = ST_HORIZON
                break
            # 1. pick a violated row (sample <= SAMPLE, proportional to w * v)
            S = nv if nv < SAMPLE else SAMPLE
            tot = 0.0
            for q in range(S):
                pos = q if nv <= SAMPLE else np.random.randint(0, nv)
                row = vlist[pos]
                srow[q] = row
                tot += w[row] * _rowviol(act[row], rl[row], ru[row], rs[row])
                spre[q] = tot
            u = np.random.random() * tot
            r = srow[S - 1]
            for q in range(S):
                if spre[q] > u:
                    r = srow[q]
                    break
            # 2. score the row's variables
            p0 = row_ptr[r]
            L = row_ptr[r + 1] - p0
            rot = np.random.randint(0, n)
            cnt = L
            start = 0
            if row_cap > 0 and L > row_cap:
                cnt = row_cap
                start = np.random.randint(0, L)
            bs = inf
            bkey = n
            bj = -1
            bt = 0.0
            bdv = 0.0
            for q in range(cnt):
                j = col_idx[p0 + (start + q) % L]
                s, dv, t = _eval_var(j, r, x, col_ptr, row_idx, col_val, rl, ru, rs, lb, ub, vtype, cobj, use_obj,
                                     act, w, max_cand, vtol, vcap, kw, kt, cand, sc)
                if s < inf:
                    key = (j + rot) % n
                    if s < bs or (s == bs and key < bkey):
                        bs = s
                        bkey = key
                        bj = j
                        bt = t
                        bdv = dv
            domove = bs < IMPROVING
            # 4. local minimum: bump every violated row; maybe a random-walk move
            if not domove:
                lv = 0.0
                for q in range(nv):
                    i = vlist[q]
                    lv += _rowviol(act[i], rl[i], ru[i], rs[i])
                    w[i] += bump
                F += bump * lv
                lmin += 1
                if p_rw > 0.0 and bj >= 0 and np.random.random() < p_rw:
                    domove = True
                    bdv = _move_dviol(bj, bt - x[bj], col_ptr, row_idx, col_val, rl, ru, rs, act, w)
            # 3. apply the move
            if domove:
                d = bt - x[bj]
                for p in range(col_ptr[bj], col_ptr[bj + 1]):
                    i = row_idx[p]
                    a = act[i] + col_val[p] * d
                    act[i] = a
                    now = _isviol(a, rl[i], ru[i], vtol, vcap)
                    was = vpos[i] >= 0
                    if now and not was:
                        vpos[i] = nv
                        vlist[nv] = i
                        nv += 1
                    elif was and not now:
                        pi = vpos[i]
                        last = vlist[nv - 1]
                        vlist[pi] = last
                        vpos[last] = pi
                        vpos[i] = -1
                        nv -= 1
                x[bj] = bt
                moves += 1
                F += bdv
            steps += 1
            if F < Fbest - 1e-12 * max(1.0, abs(Fbest)):
                Fbest = F
                since = 0
            else:
                since += 1
        # incremental bookkeeping check: activities from scratch
        drift = 0.0
        for i in range(m):
            a = 0.0
            for p in range(row_ptr[i], row_ptr[i + 1]):
                a += row_val[p] * x[col_idx[p]]
            e = abs(a - act[i]) / max(1.0, abs(a))
            if e > drift:
                drift = e
        for j in range(n):
            X[b, j] = x[j]
        status[b] = stt
        lane_moves[b] = moves
        lane_steps[b] = steps
        lane_lmin[b] = lmin
        lane_drift[b] = drift
        if keep:
            for i in range(m):
                ACT[b, i] = act[i]
                WGT[b, i] = w[i]
            NVIOL[b] = nv


# ------------------------------------------------------------------------------------------------------- class
class FJCpu:
    """Feasibility-jump local search for a population on one CPU core (Numba).

    `run(X)` improves every row of X (k, n; NumPy, any numeric dtype — int8 / int32 / float32 / float64) in place and
    returns counters {moves, local_minima, steps, finished_feasible, finished_stall, finished_budget, interrupted,
    seconds, act_drift, state_bytes}. `set_cutoff(inc_key)` arms the objective cutoff row (`objective_mode: cutoff_row`); call
    it whenever the incumbent key improves. `last_status` / `last_moves` / `last_steps` hold per-individual results of
    the last call (ST_* codes); `last_profile` = {kind: "sequential", seconds, H, cap_s, cap_hit, cut_rows} (the repair
    clock reads a call's time at horizon h as seconds · Σ min(steps, h) / Σ steps).

    Keyword knobs (not in RepairCfg yet; read from it when present): `max_candidates` (16), `p_random_walk` (0.0);
    `row_scan_cap` (0 = score the whole chosen row, as the GPU kernel does; > 0 = a random cyclic window of that many
    variables — a speed knob for very long rows such as the cutoff row on large n). `mem_budget_bytes` is checked
    against the per-call state (x, act, w, violated list: O(n + m) — one individual at a time).

    `row_scales` (default: `dp`'s own scale) = the row scales s_i one call can use (`AugmentedRows.rs_tab`, the
    self-adaptive repair strategy); `run(X, scale_idx=...)` gives each row its index into that list (None: index 0
    for every row). The Numba kernel takes ONE scale vector, so a call is split into runs of rows with equal index —
    no cost on one core, where individuals are searched one after another anyway, and no recompile (a row of the
    C-contiguous table has the signature of a vector).
    """

    def __init__(self, dp: DeviceProblem, cfg: RepairCfg, imp_tol: float = 1e-9, mem_budget_bytes: int = 256 << 20,
                 seed: int = 0, *, max_candidates: int | None = None, p_random_walk: float | None = None,
                 row_scan_cap: int = 0, slice_ms: float = 50.0, row_scales=None):
        self.dp, self.cfg = dp, cfg
        self.slice_ms = float(slice_ms)   # with should_stop: target seconds per Numba call (one slice of rows)
        self.imp_tol = float(imp_tol)
        self.mem_budget_bytes = int(mem_budget_bytes)
        self.seed = int(seed)
        mc = getattr(cfg, "max_candidates", 16) if max_candidates is None else max_candidates
        self.max_candidates = int(min(max(int(mc), 4), MAX_CAND))
        pr = getattr(cfg, "p_random_walk", 0.0) if p_random_walk is None else p_random_walk
        self.p_random_walk = float(pr)
        self.row_scan_cap = int(row_scan_cap)
        self._calls = 0
        self.last_status: np.ndarray | None = None
        self.last_moves: np.ndarray | None = None
        self.last_steps: np.ndarray | None = None
        self.last_profile: dict | None = None
        self.state: dict | None = None

        prob = dp.prob
        self.n, self.m = prob.n, prob.m
        self.key_offset = float(dp.key_offset)
        self.rows = AugmentedRows(dp, cfg, scales=row_scales)
        self.row_scales = list(self.rows.scales)
        self.m_aug = self.rows.m_aug
        self.vtype, self.lb, self.ub = var_types(prob)
        self.isint = np.asarray(prob.isint, dtype=np.bool_)
        self.has_cont = bool(np.any(~self.isint))
        self.cobj, self.use_obj = weighted_objective(prob, cfg)
        A, C = self.rows.A, self.rows.C
        i64 = np.int64
        self._row_ptr, self._col_idx, self._row_val = A.indptr.astype(i64), A.indices.astype(i64), A.data
        self._col_ptr, self._row_idx, self._col_val = C.indptr.astype(i64), C.indices.astype(i64), C.data
        self.cut_active = False
        self.cut_rhs = math.inf
        self.state_bytes = int(8 * self.n + 32 * self.m_aug + 1024)
        if self.state_bytes > self.mem_budget_bytes:
            raise MemoryError(f"FJCpu state {self.state_bytes} B exceeds mem_budget_bytes {self.mem_budget_bytes}")

    @property
    def obj_integral(self) -> bool:
        return self.rows.integral

    def set_cutoff(self, inc_key: float | None) -> None:
        """Arm (finite key) or disarm (None / inf) the cutoff row. No-op unless objective_mode == "cutoff_row" and some
        cost is non-zero (with no objective every feasible point is optimal, so no cutoff can be met)."""
        if not self.rows.has_cut:
            self.cut_active, self.cut_rhs = False, math.inf
            return
        rhs = cutoff_side(inc_key, self.key_offset, self.imp_tol, self.rows.integral)
        self.cut_active, self.cut_rhs = math.isfinite(rhs), rhs
        self.rows.set_cutoff_side(rhs)

    def _scale_index(self, scale_idx, k: int):
        """The validated (k,) int64 scale index, or None when every row uses scale 0 (also with one scale)."""
        if scale_idx is None:
            return None
        sidx = np.asarray(scale_idx).astype(np.int64, copy=False).reshape(-1)
        if sidx.shape[0] != k:
            raise ValueError(f"scale_idx has {sidx.shape[0]} entries for {k} rows")
        if k and (sidx.min() < 0 or sidx.max() >= len(self.row_scales)):
            raise ValueError(f"scale_idx outside [0, {len(self.row_scales)})")
        return sidx if len(self.row_scales) > 1 else None

    def _seeds(self, k: int) -> np.ndarray:
        base = int(np.random.SeedSequence([self.seed & 0xFFFFFFFF, self._calls]).generate_state(1)[0])
        return ((base + np.arange(k, dtype=np.uint64) * np.uint64(2654435761)) % np.uint64(1 << 32)).astype(np.uint32)

    def run(self, X, keep_state: bool = False, should_stop=None, scale_idx=None, horizon: int | None = None,
            cap_s: float | None = None, pace=None) -> dict:
        """Improve every row of X in place; see the class docstring for the counters. keep_state=True also stores the
        final per-individual activities, weights and violated-row counts in `self.state` (tests). `scale_idx` (k,):
        each row's index into `row_scales` (None: 0).

        should_stop: optional callable. A Numba call cannot be interrupted, so with it the rows are processed in
        slices (1 row first, doubling while a call takes less than half of `slice_ms`) and `should_stop()` is polled
        before every slice; when it returns True the remaining rows are left untouched (counted in `finished_budget`
        and `interrupted`). Every row has its own seed, drawn for the whole call, so the result of a row does not
        depend on the slicing. Without it: one call over all rows.

        The repair clock (both None: the call above, unchanged). `horizon` = h: a row still searching after h steps
        stops with ST_HORIZON and keeps its current x (the stop rules feasible / budget / stall are checked first, so
        a row that would stop at step s <= h stops exactly as without the horizon — a horizon-h call is a prefix of
        the unbounded one). `cap_s`: a deadline for the whole call; the rows are processed in slices as with
        should_stop, and the rows NOT STARTED when it passes are only projected (horizon 0: ST_HORIZON with 0 steps,
        or ST_FEASIBLE if the projection is feasible) — a row in progress cannot be cut on one core. This is the
        sequential meaning of the GPU's call deadline (where the chunk in progress is cut at an equal step count).
        The counters then add `finished_horizon` (rows with ST_HORIZON, also counted in `finished_budget`, so the
        three finish counters still add up to k) and `cap_hits` (1 if the deadline cut the call).

        `pace` (the EA's GPU back-off, D-018, wired here for CPU tests of the engine; None: the call above, unchanged):
        a callable that may sleep and returns the seconds it slept, called before every slice (before the stop poll,
        so a stop during a sleep is seen at once). With it the rows are processed in slices as with should_stop, the
        deadline compares BUSY time (sleeps subtracted), `seconds` and the profile's seconds are busy time, and the
        counters add `paused_s` and `wall_s`."""
        t0 = time.perf_counter()
        paused = 0.0                                    # seconds slept in `pace` (0.0 without it)
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise TypeError("FJCpu.run needs a 2-D NumPy array (use FJGpu for device arrays)")
        k, n = X.shape
        if n != self.n:
            raise ValueError(f"X has {n} columns, the problem has {self.n}")
        if np.issubdtype(X.dtype, np.integer) and self.has_cont:
            raise TypeError("integer gene dtype with continuous variables in the problem")
        if horizon is not None and int(horizon) < 0:
            raise ValueError("horizon must be >= 0")
        if cap_s is not None and not float(cap_s) > 0:
            raise ValueError("cap_s must be > 0")
        clock = horizon is not None or cap_s is not None
        mls = NO_HORIZON if horizon is None else int(horizon)
        self._calls += 1
        out = dict(moves=0, local_minima=0, steps=0, finished_feasible=0, finished_stall=0, finished_budget=0,
                   interrupted=0, seconds=0.0, act_drift=0.0, state_bytes=self.state_bytes)
        cut_rows = 0
        status = np.zeros(k, np.int8)                   # 0 = not finished (a row a stop left unreached)
        lmoves = np.zeros(k, np.int64)
        lsteps = np.zeros(k, np.int64)
        llmin = np.zeros(k, np.int64)
        ldrift = np.zeros(k, np.float64)
        ma = max(self.m_aug, 1)
        if keep_state:
            ACT, WGT, NVIOL = np.zeros((k, ma)), np.zeros((k, ma)), np.zeros(k, np.int64)
        else:
            ACT, WGT, NVIOL = np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1, np.int64)
        sidx = self._scale_index(scale_idx, k)
        if k > 0:
            c = self.cfg
            r = self.rows
            seeds = self._seeds(k)

            def batch(a: int, b: int, rs, h: int) -> None:
                # row slices of C-contiguous arrays keep the Numba signature of the whole call (no new compile)
                kept = (ACT[a:b], WGT[a:b], NVIOL[a:b]) if keep_state else (ACT, WGT, NVIOL)
                _fj_batch(X[a:b], seeds[a:b], self._row_ptr, self._col_idx, self._row_val, self._col_ptr,
                          self._row_idx, self._col_val, r.rl, r.ru, rs, self.lb, self.ub, self.isint, self.vtype,
                          self.cobj, self.use_obj, int(c.max_steps), int(c.stall_steps), self.max_candidates,
                          float(c.weight_bump), self.p_random_walk, VIOL_TOL, VIOL_CAP, self.row_scan_cap, h,
                          status[a:b], lmoves[a:b], lsteps[a:b], llmin[a:b], ldrift[a:b], keep_state, *kept)

            def call(a: int, b: int, h: int = mls) -> None:
                if sidx is None:
                    batch(a, b, r.rs, h)
                    return
                cut = a + 1 + np.flatnonzero(sidx[a + 1:b] != sidx[a:b - 1])     # runs of equal scale index
                for u, v in zip(np.r_[a, cut], np.r_[cut, b]):
                    batch(int(u), int(v), r.rs_tab[int(sidx[u])], h)

            if should_stop is None and cap_s is None and pace is None:
                call(0, k)
            else:
                deadline = None if cap_s is None else _clock() + float(cap_s)
                a, size, target = 0, 1, self.slice_ms / 1000.0
                while a < k:
                    if pace is not None:
                        paused += float(pace())
                    if should_stop is not None and should_stop():
                        break
                    if deadline is not None and _clock() - paused >= deadline:
                        call(a, k, 0)                       # rows not started: projected only (horizon 0)
                        cut_rows = k - a
                        break
                    b = min(k, a + size)
                    t = time.perf_counter()
                    call(a, b)
                    if time.perf_counter() - t < 0.5 * target:
                        size *= 2
                    a = b
        out["moves"] = int(lmoves.sum())
        out["local_minima"] = int(llmin.sum())
        out["steps"] = int(lsteps.sum())
        out["finished_feasible"] = int(np.sum(status == ST_FEASIBLE))
        out["finished_stall"] = int(np.sum(status == ST_STALL))
        out["interrupted"] = int(np.sum(status == 0))
        out["finished_budget"] = int(np.sum(status == ST_BUDGET)) + out["interrupted"]
        if clock:
            out["finished_horizon"] = int(np.sum(status == ST_HORIZON))
            out["finished_budget"] += out["finished_horizon"]
            out["cap_hits"] = int(cut_rows > 0)
        out["act_drift"] = float(ldrift.max(initial=0.0))
        wall = time.perf_counter() - t0
        out["seconds"] = wall - paused
        if pace is not None:
            out["paused_s"], out["wall_s"] = paused, wall
        self.last_status, self.last_moves, self.last_steps = status, lmoves, lsteps
        self.last_profile = dict(kind="sequential", seconds=out["seconds"], H=None if horizon is None else int(horizon),
                                 cap_s=cap_s, cap_hit=cut_rows > 0, cut_rows=int(cut_rows))
        if keep_state:
            self.state = {"act": ACT[:, :self.m_aug], "w": WGT[:, :self.m_aug], "nviol": NVIOL}
        return out
