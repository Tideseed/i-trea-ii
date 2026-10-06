"""Feasibility-jump local search on the GPU: one CUDA thread block per individual (CuPy `RawKernel`).

Spec: vault `10-design/Feasibility jump local search.md`. This is the GPU EA's repair operator; `ls_fj_cpu.py` is the
single-core reference with the same move rule, and `repair.LocalSearch` dispatches between them.

WHY a hand-written kernel. The vectorised baseline (`repair: dense`) rescores every non-zero for every individual to
apply ONE move per individual — O(nnz·P) per move, which the design critique measured at single-digit generations per
300 s on s100-sized instances. A feasibility-jump move only touches the moved column's rows and the chosen row's
columns (~coldeg × rowdeg non-zeros), so each individual's search is a sequential chain of cheap, data-dependent steps:
the natural mapping is one thread block per individual, with the block's threads cooperating inside one step.

One step of one block (individual):
  1. thread 0 checks the stop rules (no violated row → feasible; `max_steps` moves → budget; `stall_steps` steps without
     a decrease of the best total weighted violation → stall);
  2. warp 0 samples up to 32 rows of the violated list (all of them when there are ≤ 32) and picks one with probability
     ∝ w_i·v_i inside the sample — the spec's "bounded sample" option, exact when nviol ≤ 32;
  3. the block's threads stride over the chosen row's variables; each thread builds its variable's candidate values
     (binary flip; ±1, bounds and rounded tightening values for general integers; bounds and tightening values for
     continuous variables — the chosen row's own tightening value always first, then the other violated rows of the
     column by largest weighted violation, at most `max_candidates`) and scores all of them in one walk of its CSC
     column; a block argmin picks the best move (ties broken by a per-step random rotation of the variable index).
     Optional, off by default (spec behaviour): `row_scan_cap` > 0 scores a row longer than that — in practice the
     cutoff row, whose support is the whole objective, so a full scan costs ~nnz per step — on a random cyclic window
     of that many entries (the same knob and rule as `ls_fj_cpu`); `row_scan_filter` also skips the window's
     variables that cannot move in the direction reducing the row's violation;
  4. improving (score < −1e-12): the moved column's rows are updated in tiles of blockDim — every thread updates one
     row's activity (rows of a column are distinct, so there is no write race) and flags a change of violated-set
     membership; a block exclusive scan packs the changed rows into shared memory in a deterministic order, and thread
     0 applies the appends / swap-with-last removals serially. Only rows whose status changed take the serial path;
  5. otherwise (local minimum): every violated row's weight grows by `weight_bump` (FJ's weighting); with probability
     `p_random_walk` (default 0) the least-bad move is applied anyway.

The objective cutoff (`objective_mode: cutoff_row`) is an extra row m of the augmented matrix — `cmin·x ≤ inc − δ` —
so it is selected, weighted, bumped and kept incrementally exactly like any other row; `set_cutoff` only rewrites its
upper side (+inf = inactive). `weighted` adds `obj_weight·cmin_j·Δx_j / max|cmin|` to every move's score; `none` adds
nothing. Continuous variables are moved by jump (tightening) values under every `continuous` mode; a singleton column's
tightening value is its closed-form decode.

No host synchronisation per step, but the kernel is NOT run to completion in one launch: this GB10 reports
`kernelExecTimeoutEnabled = 1` (the GPU also drives a desktop), so a kernel that loops for seconds can be killed. The
loop state of every individual (counters, RNG, weighted violation) lives in global memory, `fj_steps` runs at most
`iters` steps per launch, and the host relaunches until every individual is finished — one tiny device→host read per
launch. `iters` adapts so that a launch takes about `launch_ms`.

Repair clock (`run(horizon=..., cap_s=...)`, host side only — the kernel source is unchanged). An active block runs
exactly `iters` steps per launch or stops on its own rule, so after every launch all still-active blocks have the same
step count (the sum of the launches' `iters`). A horizon h is therefore enforced between launches: the host truncates
the next launch to min(iters, h − steps so far) and stops relaunching at h; blocks still active get ST_HORIZON on the
host (never written to STI) and keep their current x. A per-call deadline `cap_s`, split over the chunks (chunk c of C
gets t0 + cap·(c+1)/C), stops relaunching the same way, at the equal step count reached (H_eff); a chunk reached after
its deadline is still projected (prep, fj_init and a launch of 0 steps, which only applies the stop rules) instead of
being left untouched. Every call records its per-launch (steps, seconds) profile (`last_profile`) and each block's
step count (`last_steps`), whatever the arguments.

Determinism: the per-block RNG is seeded from (seed, global individual index, call counter), list updates use scans
instead of atomics, and reductions run in a fixed tree order — so a run is reproducible, and an individual's result
does not depend on how the population is chunked.

Memory. Per individual in global memory: working x (n float64; skipped when X is already float64 and C-contiguous —
then the kernel works on X in place), activities, weights, violated list and positions (m_aug each: 8+8+4+4 bytes),
plus 80 bytes of loop state. Individuals are processed in chunks sized from `mem_budget_bytes`. Not in the CuPy pool
(known, unaccounted): NVRTC's compile-time memory, the CUDA context, and the driver's local-memory reservation for the
per-thread candidate arrays (~1 KB × resident threads, ≤ ~75 MB on 48 SMs). Static: the device copy of A_aug (CSR and
CSC, ~24·nnz_aug bytes) and the row / column vectors, among them the row-scale table (R × m_aug float64, R = 1 unless
`row_scales` lists several).

Row scales (the self-adaptive repair strategy, `StrategyCfg`). `row_scales` = the scales s_i one call can use; the
kernels read row `sidx[b]` of the stacked (R, m_aug) table for block b — the only difference between scales — so
individuals of every scale are repaired in ONE call: their blocks run side by side, the slowest individual's tail is
paid once per call, and the launch ramp (`ITERS_START`) restarts once. One call per scale would serialise those tails
(measured 2026-10-06: +13–39 % per generation with two scales). Without a scale index (`use_sidx` = 0) every block
reads row 0, the kernel's arithmetic is unchanged.
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sps

from itrea.config import RepairCfg
from itrea.ea.evaluate import DeviceProblem, row_scale_table

# column (variable) classes in the kernel
VT_CONT, VT_BIN, VT_INT, VT_FIXED = 0, 1, 2, 3
# per-individual integer state (int64 columns of STI)
S_NVIOL, S_MOVES, S_STEPS, S_LMIN, S_SINCE, S_STATUS, S_RNG = 0, 1, 2, 3, 4, 5, 6
N_SI = 8
# per-individual float state (float64 columns of STD)
D_F, D_FBEST = 0, 1
N_SD = 2
# finishing status (ST_HORIZON: set on the host only, for blocks still active at the repair clock's horizon or deadline)
ST_ACTIVE, ST_FEASIBLE, ST_STALL, ST_BUDGET, ST_HORIZON = 0, 1, 2, 3, 4

ITERS_START = 8        # steps per block in the first launch of every call (then x4 per launch towards launch_ms)
LAUNCH_MS_MIN = 5.0    # with a call deadline: launches target clamp(cap / 8, LAUNCH_MS_MIN, launch_ms)
MAX_CAND = 32          # compile-time capacity of the per-thread candidate array (runtime max_candidates <= this)
SAMPLE = 32            # violated rows sampled per step (one warp)
LONG_ROW = 1024        # rows longer than this get their initial activity from the whole block (not one thread)
DEFAULT_THREADS = 128  # threads per block: equal to 32/64 on short rows, 1.5x faster than 64 on long (cutoff) rows

_KERNEL_SRC = r"""
#define FULL 0xffffffffu
#define NW (BD / 32)
#define MAXK (MAXC - 2)
#define FJ_INF __longlong_as_double(0x7ff0000000000000LL)
#define VT_CONT 0
#define VT_BIN 1
#define VT_INT 2
#define VT_FIXED 3
#define S_NVIOL 0
#define S_MOVES 1
#define S_STEPS 2
#define S_LMIN 3
#define S_SINCE 4
#define S_STATUS 5
#define S_RNG 6
#define N_SI 8
#define D_F 0
#define D_FBEST 1
#define N_SD 2

typedef unsigned long long u64;

__device__ __forceinline__ u64 splitmix64(u64 z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__device__ __forceinline__ u64 xorshift64(u64* s) {
    u64 x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}

__device__ __forceinline__ double u01(u64 r) { return (double)(r >> 11) * (1.0 / 9007199254740992.0); }

// normalised violation of a row with activity a; comparisons (not max(lo - a, 0)) so infinite sides give 0, never NaN
__device__ __forceinline__ double rowviol(double a, double lo, double hi, double sc) {
    double v = 0.0;
    if (a < lo) v += lo - a;
    if (a > hi) v += a - hi;
    return v / sc;
}

// violated-set membership: beyond min(tol * max(1, |side|), cap) — far inside every partner's feasibility test
__device__ __forceinline__ bool isviol(double a, double lo, double hi, double tol, double cap) {
    if (a < lo && lo - a > fmin(tol * fmax(1.0, fabs(lo)), cap)) return true;
    if (a > hi && a - hi > fmin(tol * fmax(1.0, fabs(hi)), cap)) return true;
    return false;
}

__device__ __forceinline__ bool better(double s1, u64 k1, double s2, u64 k2) {
    return (s1 < s2) || (s1 == s2 && k1 < k2);
}

// block-wide sum in a fixed tree order (deterministic for a given BD); every thread must call; sh has NW+1 slots
__device__ double block_sum(double v, double* sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(FULL, v, o);
#if BD == 32
    return __shfl_sync(FULL, v, 0);          // one warp: no shared memory, no barrier
#endif
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        double s = (lane < NW) ? sh[lane] : 0.0;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(FULL, s, o);
        if (lane == 0) sh[NW] = s;
    }
    __syncthreads();
    const double r = sh[NW];
    __syncthreads();
    return r;
}

// block-wide exclusive scan of int flags; every thread must call; sh has NW+1 slots; *total = sum of all flags
__device__ int block_excl_scan(int v, int* total, int* sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(FULL, x, o);
        if (lane >= o) x += y;
    }
#if BD == 32
    *total = __shfl_sync(FULL, x, 31);       // one warp: no shared memory, no barrier
    return x - v;
#endif
    if (lane == 31) sh[wid] = x;
    __syncthreads();
    if (wid == 0) {
        int s = (lane < NW) ? sh[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(FULL, s, o);
            if (lane >= o) s += y;
        }
        if (lane < NW) sh[lane] = s;
    }
    __syncthreads();
    const int off = (wid > 0) ? sh[wid - 1] : 0;
    *total = sh[NW - 1];
    __syncthreads();
    return off + x - v;
}

// best candidate value of variable j (chosen row r) and its score; bs = +inf if j cannot move
__device__ void eval_var(const int j, const int r, const double* __restrict__ x,
                         const int* __restrict__ col_ptr, const int* __restrict__ row_idx,
                         const double* __restrict__ col_val,
                         const double* __restrict__ rl, const double* __restrict__ ru, const double* __restrict__ rs,
                         const double* __restrict__ lb, const double* __restrict__ ub,
                         const signed char* __restrict__ vtype, const double* __restrict__ cobj, const int use_obj,
                         const double* act, const double* w, const int max_cand,
                         const double vtol, const double vcap,
                         double* bs, double* bdv, double* bt) {
    *bs = FJ_INF; *bdv = 0.0; *bt = 0.0;
    const int ty = vtype[j];
    if (ty == VT_FIXED) return;
    const double xj = x[j];
    const double lbj = lb[j], ubj = ub[j];
    const int c0 = col_ptr[j], c1 = col_ptr[j + 1];
    double cand[MAXC];
    int nc = 0;
    if (ty == VT_BIN) {
        cand[nc++] = 1.0 - xj;
    } else {
        // tightening values of the column's violated rows: chosen row first, then by largest w_i * v_i
        int K = (ty == VT_INT) ? (max_cand - 4) / 2 : (max_cand - 2);
        if (K < 1) K = 1;
        if (K > MAXK) K = MAXK;
        double kw[MAXK], kt[MAXK];
        int nk = 0;
        for (int p = c0; p < c1; ++p) {
            const int i = row_idx[p];
            const double a = act[i], lo = rl[i], hi = ru[i];
            if (!isviol(a, lo, hi, vtol, vcap)) continue;
            const double side = (a < lo) ? lo : hi;
            const double t = xj + (side - a) / col_val[p];
            const double wv = (i == r) ? FJ_INF : w[i] * rowviol(a, lo, hi, rs[i]);
            int q;
            if (nk < K) q = nk++;
            else if (wv > kw[K - 1]) q = K - 1;
            else continue;
            while (q > 0 && kw[q - 1] < wv) { kw[q] = kw[q - 1]; kt[q] = kt[q - 1]; --q; }
            kw[q] = wv; kt[q] = t;
        }
        if (ty == VT_INT) {
            cand[nc++] = xj - 1.0;
            cand[nc++] = xj + 1.0;
            if (lbj > -FJ_INF) cand[nc++] = lbj;
            if (ubj < FJ_INF) cand[nc++] = ubj;
            for (int q = 0; q < nk && nc + 2 <= MAXC; ++q) {
                cand[nc++] = floor(kt[q] + 1e-9);
                cand[nc++] = ceil(kt[q] - 1e-9);
            }
        } else {
            if (lbj > -FJ_INF) cand[nc++] = lbj;
            if (ubj < FJ_INF) cand[nc++] = ubj;
            for (int q = 0; q < nk && nc < MAXC; ++q) cand[nc++] = kt[q];
        }
    }
    double sc[MAXC];
    for (int c = 0; c < nc; ++c) {
        const double t = fmin(fmax(cand[c], lbj), ubj);
        cand[c] = t - xj;                       // from here on: the step d
        sc[c] = (t == xj) ? FJ_INF : 0.0;
    }
    for (int p = c0; p < c1; ++p) {
        const int i = row_idx[p];
        const double a = act[i], av = col_val[p], lo = rl[i], hi = ru[i], s = rs[i], wi = w[i];
        const double v0 = rowviol(a, lo, hi, s);
        for (int c = 0; c < nc; ++c) sc[c] += wi * (rowviol(a + av * cand[c], lo, hi, s) - v0);
    }
    const double cj = use_obj ? cobj[j] : 0.0;
    for (int c = 0; c < nc; ++c) {
        if (!(sc[c] < FJ_INF)) continue;
        const double tot = sc[c] + cj * cand[c];
        if (tot < *bs) { *bs = tot; *bdv = sc[c]; *bt = xj + cand[c]; }
    }
}

extern "C" __global__ void __launch_bounds__(BD)
fj_init(const int n, const int m,
        const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const double* __restrict__ row_val,
        const double* __restrict__ rl, const double* __restrict__ ru, const double* __restrict__ rs_all,
        const int* __restrict__ sidx, const int use_sidx,
        const int* __restrict__ long_rows, const int n_long, const int long_len,
        const double* X, double* ACT, double* W, int* VLIST, int* VPOS, long long* STI, double* STD,
        const unsigned long long seed, const long long idx0, const unsigned long long call,
        const double vtol, const double vcap) {
    __shared__ double sh_d[NW + 1];
    __shared__ int sh_i[NW + 1];
    const int b = blockIdx.x, tid = threadIdx.x;
    // this individual's row scale: row sidx[b] of the (R, m) table (row 0 without a scale index)
    const double* __restrict__ rs = rs_all + (use_sidx ? (long long)sidx[b] * m : 0LL);
    const double* x = X + (long long)b * n;
    double* act = ACT + (long long)b * m;
    double* w = W + (long long)b * m;
    int* vlist = VLIST + (long long)b * m;
    int* vpos = VPOS + (long long)b * m;
    double fl = 0.0;
    for (int i = tid; i < m; i += BD) {
        w[i] = 1.0;
        const int p0 = row_ptr[i], p1 = row_ptr[i + 1];
        if (p1 - p0 > long_len) continue;
        double a = 0.0;
        for (int p = p0; p < p1; ++p) a += row_val[p] * x[col_idx[p]];
        act[i] = a;
        fl += rowviol(a, rl[i], ru[i], rs[i]);
    }
    for (int q = 0; q < n_long; ++q) {           // long rows (e.g. the cutoff row): the whole block, one reduction
        const int i = long_rows[q];
        double part = 0.0;
        for (int p = row_ptr[i] + tid; p < row_ptr[i + 1]; p += BD) part += row_val[p] * x[col_idx[p]];
        const double a = block_sum(part, sh_d);
        if (tid == 0) { act[i] = a; fl += rowviol(a, rl[i], ru[i], rs[i]); }
    }
    __syncthreads();
    const double F = block_sum(fl, sh_d);
    int nv = 0;
    for (int base = 0; base < m; base += BD) {
        const int i = base + tid;
        const int flag = (i < m) ? (isviol(act[i], rl[i], ru[i], vtol, vcap) ? 1 : 0) : 0;
        int total;
        const int pos = block_excl_scan(flag, &total, sh_i);
        if (i < m) {
            if (flag) { vlist[nv + pos] = i; vpos[i] = nv + pos; }
            else vpos[i] = -1;
        }
        nv += total;
    }
    if (tid == 0) {
        long long* st = STI + (long long)b * N_SI;
        u64 r = splitmix64(seed ^ splitmix64((u64)(idx0 + b) ^ splitmix64(call ^ 0xA0761D6478BD642FULL)));
        if (r == 0) r = 0x9E3779B97F4A7C15ULL;
        st[S_NVIOL] = nv; st[S_MOVES] = 0; st[S_STEPS] = 0; st[S_LMIN] = 0; st[S_SINCE] = 0; st[S_STATUS] = 0;
        st[S_RNG] = (long long)r;
        double* sd = STD + (long long)b * N_SD;
        sd[D_F] = F; sd[D_FBEST] = F;
    }
}

extern "C" __global__ void __launch_bounds__(BD)
fj_steps(const int n, const int m,
         const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const double* __restrict__ row_val,
         const int* __restrict__ col_ptr, const int* __restrict__ row_idx, const double* __restrict__ col_val,
         const double* __restrict__ rl, const double* __restrict__ ru, const double* __restrict__ rs_all,
         const int* __restrict__ sidx, const int use_sidx,
         const double* __restrict__ lb, const double* __restrict__ ub, const signed char* __restrict__ vtype,
         const double* __restrict__ cobj, const int use_obj,
         double* X, double* ACT, double* W, int* VLIST, int* VPOS, long long* STI, double* STD,
         const long long max_moves, const long long stall_steps, const int max_cand, const double bump,
         const double p_rw, const double vtol, const double vcap, const int row_cap, const int dir_filter,
         const int iters) {
    const int b = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    long long* st = STI + (long long)b * N_SI;
    if (st[S_STATUS] != 0) return;               // finished in an earlier launch (same value for every thread)
    const double* __restrict__ rs = rs_all + (use_sidx ? (long long)sidx[b] * m : 0LL);   // this individual's scale
    double* x = X + (long long)b * n;
    double* act = ACT + (long long)b * m;
    double* w = W + (long long)b * m;
    int* vlist = VLIST + (long long)b * m;
    int* vpos = VPOS + (long long)b * m;
    double* sd = STD + (long long)b * N_SD;

    __shared__ int s_nviol, s_status, s_row, s_bj, s_domove;
    __shared__ long long s_moves, s_steps, s_lmin, s_since;
    __shared__ u64 s_rng, s_rstep;
    __shared__ double s_F, s_Fbest, s_bs, s_bdv, s_bt, s_d;
    __shared__ int s_chg[BD];
    __shared__ double sh_d[NW + 1];
    __shared__ int sh_i[NW + 1];
    __shared__ double r_s[NW], r_t[NW], r_dv[NW];
    __shared__ u64 r_k[NW];
    __shared__ int r_j[NW];

    if (tid == 0) {
        s_nviol = (int)st[S_NVIOL]; s_moves = st[S_MOVES]; s_steps = st[S_STEPS]; s_lmin = st[S_LMIN];
        s_since = st[S_SINCE]; s_rng = (u64)st[S_RNG]; s_F = sd[D_F]; s_Fbest = sd[D_FBEST]; s_status = 0;
    }
    __syncthreads();

    for (int it = 0; it < iters; ++it) {
        // 1. stop rules
        if (tid == 0) {
            int stt = 0;
            if (s_nviol == 0) stt = 1;
            else if (s_moves >= max_moves) stt = 3;
            else if (s_since >= stall_steps) stt = 2;
            s_status = stt;
            if (!stt) s_rstep = xorshift64(&s_rng);
        }
        __syncthreads();
        if (s_status) break;
        const u64 rstep = s_rstep;

        // 2. pick a violated row: warp 0 samples <= 32, picks proportional to w * v inside the sample
        if (wid == 0) {
            const int nv = s_nviol;
            const int S = nv < 32 ? nv : 32;
            double wv = 0.0;
            int row = -1;
            if (lane < S) {
                const int pos = (nv <= 32) ? lane : (int)(splitmix64(rstep + (u64)lane + 1ULL) % (u64)nv);
                row = vlist[pos];
                wv = w[row] * rowviol(act[row], rl[row], ru[row], rs[row]);
            }
            double pre = wv;
            for (int o = 1; o < 32; o <<= 1) {
                const double y = __shfl_up_sync(FULL, pre, o);
                if (lane >= o) pre += y;
            }
            const double tot = __shfl_sync(FULL, pre, 31);
            const double u = u01(splitmix64(rstep ^ 0x5BD1E9955BD1E995ULL)) * tot;
            const unsigned msk = __ballot_sync(FULL, lane < S && pre > u);
            const int sel = msk ? (__ffs(msk) - 1) : (S - 1);
            const int r = __shfl_sync(FULL, row, sel);
            if (lane == 0) s_row = r;
        }
        __syncthreads();
        const int r = s_row;

        // 3. score every variable of row r; block argmin with a random rotation as the tie-break.
        //    Optional (row_scan_cap > 0): a row longer than that (e.g. the cutoff row, whose support is the whole
        //    objective) is scored on a random cyclic window of row_scan_cap entries — the same rule as ls_fj_cpu —
        //    and, with dir_filter, only the window's variables that can move in the direction reducing row r's
        //    violation are scored.
        double bs = FJ_INF, bdv = 0.0, bt = 0.0;
        int bj = -1;
        u64 bk = ~0ULL;
        const u64 nn = (u64)(n > 0 ? n : 1);
        const u64 rot = splitmix64(rstep ^ 0x2545F4914F6CDD1DULL) % nn;
        const int p0 = row_ptr[r], L = row_ptr[r + 1] - p0;
        const bool lng = row_cap > 0 && L > row_cap;
        const int Wn = lng ? row_cap : L;
        const int off = lng ? (int)(splitmix64(rstep ^ 0x94D049BB133111EBULL) % (u64)L) : 0;
        const bool r_low = act[r] < rl[r];          // row r is violated below its lower side (else above its upper)
        for (int q = tid; q < Wn; q += BD) {
            int p = p0 + q;
            if (lng) {
                int o = off + q;
                if (o >= L) o -= L;
                p = p0 + o;
            }
            const int j = col_idx[p];
            if (lng && dir_filter) {
                const bool up = (row_val[p] > 0.0) == r_low;   // helpful direction of x_j for row r
                const double xv = x[j];
                if (up ? !(xv < ub[j]) : !(xv > lb[j])) continue;
            }
            double s, dv, t;
            eval_var(j, r, x, col_ptr, row_idx, col_val, rl, ru, rs, lb, ub, vtype, cobj, use_obj, act, w,
                     max_cand, vtol, vcap, &s, &dv, &t);
            if (s < FJ_INF) {
                const u64 key = ((u64)j + rot) % nn;
                if (better(s, key, bs, bk)) { bs = s; bk = key; bj = j; bt = t; bdv = dv; }
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const double s2 = __shfl_down_sync(FULL, bs, o);
            const u64 k2 = __shfl_down_sync(FULL, bk, o);
            const int j2 = __shfl_down_sync(FULL, bj, o);
            const double t2 = __shfl_down_sync(FULL, bt, o);
            const double dv2 = __shfl_down_sync(FULL, bdv, o);
            if (better(s2, k2, bs, bk)) { bs = s2; bk = k2; bj = j2; bt = t2; bdv = dv2; }
        }
        if (lane == 0) { r_s[wid] = bs; r_k[wid] = bk; r_j[wid] = bj; r_t[wid] = bt; r_dv[wid] = bdv; }
        __syncthreads();
        if (tid == 0) {
            for (int q = 1; q < NW; ++q)
                if (better(r_s[q], r_k[q], bs, bk)) { bs = r_s[q]; bk = r_k[q]; bj = r_j[q]; bt = r_t[q]; bdv = r_dv[q]; }
            s_bs = bs; s_bj = bj; s_bt = bt; s_bdv = bdv;
            s_domove = (bs < -1e-12) ? 1 : 0;
            s_d = (bj >= 0) ? (bt - x[bj]) : 0.0;
        }
        __syncthreads();

        // 5. local minimum: bump the weights of every violated row (and maybe take the least-bad move)
        if (!s_domove) {
            const int nv = s_nviol;
            double lv = 0.0;
            for (int q = tid; q < nv; q += BD) {
                const int i = vlist[q];
                lv += rowviol(act[i], rl[i], ru[i], rs[i]);
                w[i] += bump;
            }
            const double tot = block_sum(lv, sh_d);
            if (tid == 0) {
                s_F += bump * tot;
                s_lmin += 1;
                if (p_rw > 0.0 && s_bj >= 0 && u01(splitmix64(rstep ^ 0x8BB84B93962EACC9ULL)) < p_rw) s_domove = 1;
            }
            __syncthreads();
        }

        // 4. apply the move: activities of column j's rows, then violated-set changes (scan + serial apply)
        if (s_domove) {
            const int j = s_bj;
            const double d = s_d;
            const int c0 = col_ptr[j], c1 = col_ptr[j + 1];
            for (int base = c0; base < c1; base += BD) {
                const int p = base + tid;
                int flag = 0, code = 0;
                if (p < c1) {
                    const int i = row_idx[p];
                    const double a = act[i] + col_val[p] * d;
                    act[i] = a;
                    const bool now = isviol(a, rl[i], ru[i], vtol, vcap);
                    const bool was = vpos[i] >= 0;
                    if (now != was) { flag = 1; code = now ? i : ~i; }
                }
                int total;
                const int pos = block_excl_scan(flag, &total, sh_i);
                if (flag) s_chg[pos] = code;
                __syncthreads();
                if (tid == 0) {
                    int nv = s_nviol;
                    for (int q = 0; q < total; ++q) {
                        const int c = s_chg[q];
                        if (c >= 0) {
                            vpos[c] = nv; vlist[nv] = c; ++nv;
                        } else {
                            const int i = ~c;
                            const int pi = vpos[i];
                            const int last = vlist[nv - 1];
                            vlist[pi] = last; vpos[last] = pi; vpos[i] = -1; --nv;
                        }
                    }
                    s_nviol = nv;
                }
                __syncthreads();
            }
            if (tid == 0) { x[j] = s_bt; s_moves += 1; s_F += s_bdv; }
        }

        if (tid == 0) {
            s_steps += 1;
            if (s_F < s_Fbest - 1e-12 * fmax(1.0, fabs(s_Fbest))) { s_Fbest = s_F; s_since = 0; }
            else s_since += 1;
        }
        __syncthreads();
    }

    if (tid == 0) {
        int stt = s_status;
        if (!stt) {                              // re-check at the end of the slice so finished lanes are not relaunched
            if (s_nviol == 0) stt = 1;
            else if (s_moves >= max_moves) stt = 3;
            else if (s_since >= stall_steps) stt = 2;
        }
        st[S_NVIOL] = s_nviol; st[S_MOVES] = s_moves; st[S_STEPS] = s_steps; st[S_LMIN] = s_lmin;
        st[S_SINCE] = s_since; st[S_STATUS] = stt; st[S_RNG] = (long long)s_rng;
        sd[D_F] = s_F; sd[D_FBEST] = s_Fbest;
    }
}
"""

_MODULES: dict[int, object] = {}
_EW: dict[str, object] = {}


def _module(threads: int):
    """The compiled RawModule for a block size (NVRTC, in-process; CuPy caches the binary on disk)."""
    import cupy as cp

    mod = _MODULES.get(threads)
    if mod is None:
        mod = cp.RawModule(code=_KERNEL_SRC, options=("--std=c++14", f"-DBD={threads}", f"-DMAXC={MAX_CAND}"))
        _MODULES[threads] = mod
    return mod


def _elementwise():
    """Round/clip pre-pass and integer write-back without (k, n) temporaries."""
    import cupy as cp

    if not _EW:
        _EW["prep"] = cp.ElementwiseKernel(
            "float64 lb, float64 ub, bool isint", "float64 x",
            "double v = x; if (!isfinite(v)) v = 0.0; if (isint) v = rint(v); x = fmin(fmax(v, lb), ub);",
            "itrea_fj_prep")
        _EW["wb_int"] = cp.ElementwiseKernel("float64 x", "T y", "y = (T)rint(x);", "itrea_fj_wb_int")
        _EW["wb_float"] = cp.ElementwiseKernel("float64 x", "T y", "y = (T)x;", "itrea_fj_wb_float")
    return _EW


class FJGpu:
    """Feasibility-jump local search for a population on the GPU (one thread block per individual).

    `run(X)` improves the rows of X (k, n) in place and returns counters. Row weights start at 1 in every call (the
    individuals are new children each generation). `set_cutoff(inc_key)` arms the objective cutoff row
    (`objective_mode: cutoff_row`); call it whenever the incumbent key improves.

    Keyword knobs (not in `RepairCfg` yet; read from it when present, a keyword overrides): `max_candidates` (16),
    `p_random_walk` (0.0), `row_scan_cap` (0), `row_scan_filter` (False).
    `threads` = block size (32/64/128/256); `viol_tol` / `viol_cap` define the violated-set membership
    `excess > min(viol_tol * max(1, |side|), viol_cap)`; `row_scan_cap` (0 = score the whole chosen row, as the spec
    states; > 0 = rows longer than this are scored on a random cyclic window of this many entries — the same knob and
    rule as `ls_fj_cpu`); `row_scan_filter` (GPU only, default off) additionally skips window variables that cannot
    move in the direction reducing the row's violation; `launch_ms` = target time per kernel launch.

    `row_scales` (default: `dp`'s own scale; the self-adaptive repair strategy passes several) = the row scales one
    call can use: `d_rs` is the stacked (R, m_aug) table (`evaluate.row_scale_table` plus the cutoff row's entry), and
    `run(X, scale_idx=...)` gives each individual its index (None: 0) — one call for every scale (module docstring).
    `set_cutoff` writes the cutoff row's upper side once and its divisor in every "rhs" row of the table.
    `static_bytes` counts the device arrays this instance owns.
    """

    def __init__(self, dp: DeviceProblem, cfg: RepairCfg, imp_tol: float = 1e-9, mem_budget_bytes: int = 256 << 20,
                 seed: int = 0, *, threads: int = DEFAULT_THREADS, max_candidates: int | None = None,
                 p_random_walk: float | None = None, viol_tol: float = 1e-9, viol_cap: float = 1e-7,
                 row_scan_cap: int | None = None, row_scan_filter: bool | None = None, launch_ms: float = 50.0,
                 row_scales=None):
        import cupy as cp

        if not dp.bk.is_gpu:
            raise ValueError("FJGpu needs the CuPy backend (use ls_fj_cpu on the CPU)")
        if threads not in (32, 64, 128, 256):
            raise ValueError("threads must be 32, 64, 128 or 256")
        self.dp, self.cfg = dp, cfg
        self.imp_tol = float(imp_tol)
        self.mem_budget_bytes = int(mem_budget_bytes)
        self.seed = int(seed) & ((1 << 64) - 1)
        self.threads = int(threads)
        mc = getattr(cfg, "max_candidates", 16) if max_candidates is None else max_candidates
        self.max_candidates = int(min(max(int(mc), 4), MAX_CAND))
        pr = getattr(cfg, "p_random_walk", 0.0) if p_random_walk is None else p_random_walk
        self.p_random_walk = float(pr)
        self.viol_tol, self.viol_cap = float(viol_tol), float(viol_cap)
        rc = getattr(cfg, "row_scan_cap", 0) if row_scan_cap is None else row_scan_cap
        self.row_scan_cap = int(rc)
        rf = getattr(cfg, "row_scan_filter", False) if row_scan_filter is None else row_scan_filter
        self.row_scan_filter = bool(rf)
        self.launch_ms = float(launch_ms)
        self._calls = 0
        self._iters = ITERS_START
        self.last_status: np.ndarray | None = None      # (k,) int8 — ST_* per individual of the last run
        self.last_moves: np.ndarray | None = None       # (k,) int64
        self.last_steps: np.ndarray | None = None       # (k,) int64 — steps per individual (S_STEPS)
        self.last_profile: dict | None = None           # per-chunk launch profile of the last run (`run`)
        self.state: dict | None = None                  # filled by run(keep_state=True)

        prob = dp.prob
        n, m = prob.n, prob.m
        cmin = np.asarray(prob.cmin, dtype=np.float64)
        self.key_offset = float(dp.key_offset)
        mode = cfg.objective_mode
        self.has_cut = mode == "cutoff_row" and bool(np.any(cmin != 0))
        rl = np.asarray(prob.rl, dtype=np.float64)
        ru = np.asarray(prob.ru, dtype=np.float64)
        # row scales s_i: one row of divisors per scale (the viol_norm divisor; 1 for "count"); row 0 = dp's own
        # scale unless `row_scales` says otherwise
        self.row_scales, rs = row_scale_table(dp, row_scales)
        self._viol_norm = self.row_scales[0]
        A = sps.csr_matrix(prob.A, dtype=np.float64)
        if self.has_cut:
            A = sps.vstack([A, sps.csr_matrix(cmin.reshape(1, -1))], format="csr")
            rl = np.append(rl, -np.inf)
            ru = np.append(ru, np.inf)
            cn = float(np.linalg.norm(cmin))
            self._cut_l2 = cn if cn > 0 else 1.0
            rs = np.hstack([rs, np.array([[self._cut_l2 if md == "l2" else 1.0] for md in self.row_scales])])
            ci = cmin != 0
            # g = 1: every cost on an integer variable with an integer coefficient -> the objective is integral
            self.obj_integral = bool(np.all(prob.isint[ci]) and np.all(cmin[ci] == np.round(cmin[ci])))
        else:
            self.obj_integral = False
        A.sort_indices()
        self.n, self.m, self.m_aug = n, m, A.shape[0]
        self.nnz_aug = int(A.nnz)
        C = A.tocsc()
        C.sort_indices()
        self.cut_active = False
        self.cut_rhs = math.inf

        isint = np.asarray(prob.isint, dtype=bool)
        lb = np.asarray(prob.lb, dtype=np.float64).copy()
        ub = np.asarray(prob.ub, dtype=np.float64).copy()
        with np.errstate(invalid="ignore"):
            lb[isint] = np.ceil(lb[isint] - 1e-9)
            ub[isint] = np.floor(ub[isint] + 1e-9)
        vt = np.full(n, VT_CONT, dtype=np.int8)
        vt[isint] = VT_INT
        vt[isint & (lb == 0) & (ub == 1)] = VT_BIN
        vt[lb >= ub] = VT_FIXED
        self.vtype_host = vt
        if mode == "weighted":
            cmax = float(np.max(np.abs(cmin))) if n else 0.0
            cobj = cfg.obj_weight * cmin / cmax if cmax > 0 else np.zeros(n)
            self.use_obj = 1 if cmax > 0 else 0
        else:
            cobj, self.use_obj = np.zeros(max(n, 1)), 0
        rowlen = np.diff(A.indptr)
        long_rows = np.flatnonzero(rowlen > LONG_ROW).astype(np.int32)

        i32, f64 = np.int32, np.float64
        self.d_row_ptr = cp.asarray(A.indptr.astype(i32))
        self.d_col_idx = cp.asarray(A.indices.astype(i32))
        self.d_row_val = cp.asarray(A.data.astype(f64))
        self.d_col_ptr = cp.asarray(C.indptr.astype(i32))
        self.d_row_idx = cp.asarray(C.indices.astype(i32))
        self.d_col_val = cp.asarray(C.data.astype(f64))
        self.d_rl = cp.asarray(rl)
        self.d_ru = cp.asarray(ru)
        self.d_rs = cp.asarray(np.ascontiguousarray(rs, dtype=f64))       # (R, m_aug)
        self.d_sidx0 = cp.zeros(1, dtype=i32)                            # placeholder scale index (use_sidx = 0)
        self.d_lb = cp.asarray(lb)
        self.d_ub = cp.asarray(ub)
        self.d_isint = cp.asarray(isint)
        self.d_vtype = cp.asarray(vt)
        self.d_cobj = cp.asarray(np.asarray(cobj, dtype=f64))
        self.d_long = cp.asarray(long_rows if long_rows.size else np.zeros(1, i32))
        self.n_long = int(long_rows.size)
        self.static_bytes = int(sum(a.nbytes for a in (
            self.d_row_ptr, self.d_col_idx, self.d_row_val, self.d_col_ptr, self.d_row_idx, self.d_col_val,
            self.d_rl, self.d_ru, self.d_rs, self.d_lb, self.d_ub, self.d_isint, self.d_vtype, self.d_cobj,
            self.d_long)))
        mod = _module(self.threads)
        self._k_init = mod.get_function("fj_init")
        self._k_steps = mod.get_function("fj_steps")

    # ---------------------------------------------------------------- cutoff
    def set_cutoff(self, inc_key: float | None) -> None:
        """Arm (finite key) or disarm (None / inf) the cutoff row cmin·x <= inc - key_offset - delta, with
        delta = max(imp_tol * max(1, |inc|), g) and g = 1 when the objective is integral (then the side is rounded
        down with a 1e-6 allowance, so solver-side noise such as 104.99999999 does not cost a whole unit). A
        non-integral side is lowered by a further 2 * min(viol_tol * max(1, |b|), viol_cap), b = inc - key_offset, so
        a feasible finish is strictly better than the incumbent despite the violated-set tolerance (the same rule as
        `ls_fj_cpu.cutoff_side`). No-op unless cutoff_row mode."""
        if not self.has_cut:
            self.cut_active, self.cut_rhs = False, math.inf
            return
        if inc_key is None or not math.isfinite(float(inc_key)):
            rhs = math.inf
        else:
            inc = float(inc_key)
            b = inc - self.key_offset
            if self.obj_integral:
                delta = max(self.imp_tol * max(1.0, abs(inc)), 1.0)
                rhs = math.floor(b - delta + 1e-6)
            else:
                # lowered by a further 2x the violated-set tolerance of the side: otherwise a point AT the incumbent's
                # value could count as satisfying the row and finish "feasible" without being strictly better
                delta = self.imp_tol * max(1.0, abs(inc))
                rhs = b - delta - 2.0 * min(self.viol_tol * max(1.0, abs(b)), self.viol_cap)
        self.cut_active = math.isfinite(rhs)
        self.cut_rhs = rhs
        self.d_ru[self.m] = rhs
        for r, md in enumerate(self.row_scales):
            if md == "rhs":
                self.d_rs[r, self.m] = max(1.0, abs(rhs)) if math.isfinite(rhs) else 1.0

    # ------------------------------------------------------------------- run
    def chunk_size(self, k: int, X_is_work: bool) -> int:
        per_ind = (0 if X_is_work else 8 * self.n) + 24 * self.m_aug + 8 * (N_SI + N_SD)
        return max(1, min(k, int(self.mem_budget_bytes // max(per_ind, 1))))

    def run(self, X, keep_state: bool = False, should_stop=None, scale_idx=None, horizon: int | None = None,
            cap_s: float | None = None) -> dict:
        """Improve every row of X (k, n; int8 / int32 / float32 / float64) in place. Integers stay integral and inside
        [ceil(lb), floor(ub)]; every gene is clipped to its bounds first. `scale_idx` (k,): each row's index into
        `row_scales` (None: 0 for every row; one scale: ignored). Returns the counters
        {moves, local_minima, steps, finished_feasible, finished_stall, finished_budget, interrupted, seconds, chunk,
        peak_bytes, launches, max_launch_s}. keep_state=True also stores per-individual act / w / violated list in
        `self.state` (tests). `should_stop()` (optional) is polled before every chunk and after every launch; when it
        returns True the search ends: the chunk in progress is written back as it stands (its unfinished rows keep
        ST_ACTIVE), later chunks are left untouched; unfinished rows count in `finished_budget` and `interrupted`.

        The repair clock (module docstring; both None: the loop above, unchanged). `horizon` = h lane steps: no block
        runs more than h steps, blocks still active at h get ST_HORIZON. `cap_s`: the call's deadline, split over the
        chunks; a chunk cut by it gets ST_HORIZON at the equal step count reached, a chunk reached after it is
        projected only. The counters then add `finished_horizon` (rows with ST_HORIZON, also counted in
        `finished_budget`, so the three finish counters still add up to k) and `cap_hits` (chunks cut by the deadline).
        `last_steps` (k,) and `last_profile` = {kind: "launches", seconds, H, cap_s, chunks: [{k, cum, t, end, H_eff,
        done, cut, cap_hit}]} — per chunk the cumulative steps after every launch (starting at 0) with the seconds
        since the call started (`t[0]` = the chunk's start), the chunk's end, the step count it ended at, whether
        every block stopped by itself (`done`) and whether the horizon or the deadline cut it — are kept for every
        call."""
        import cupy as cp

        t0 = time.perf_counter()
        if not isinstance(X, cp.ndarray) or X.ndim != 2:
            raise TypeError("X must be a 2-D CuPy array (k, n)")
        k, n = int(X.shape[0]), int(X.shape[1])
        if n != self.n:
            raise ValueError(f"X has {n} columns, the problem has {self.n}")
        if horizon is not None and int(horizon) < 0:
            raise ValueError("horizon must be >= 0")
        if cap_s is not None and not float(cap_s) > 0:
            raise ValueError("cap_s must be > 0")
        clock = horizon is not None or cap_s is not None
        hz = None if horizon is None else int(horizon)
        out = dict(moves=0, local_minima=0, steps=0, finished_feasible=0, finished_stall=0, finished_budget=0,
                   interrupted=0, seconds=0.0, chunk=0, peak_bytes=0, launches=0, max_launch_s=0.0)
        d_sidx = None
        if scale_idx is not None:
            sidx = np.asarray(scale_idx).astype(np.int64, copy=False).reshape(-1)
            if sidx.shape[0] != k or (k and (sidx.min() < 0 or sidx.max() >= len(self.row_scales))):
                raise ValueError(f"scale_idx must hold {k} indices in [0, {len(self.row_scales)})")
            if len(self.row_scales) > 1 and k:
                d_sidx = cp.asarray(sidx.astype(np.int32))
        self._calls += 1
        prof_chunks: list[dict] = []
        self.last_profile = dict(kind="launches", seconds=0.0, H=hz, cap_s=cap_s, chunks=prof_chunks)
        if k == 0:
            self.last_status = np.zeros(0, np.int8)
            self.last_moves = np.zeros(0, np.int64)
            self.last_steps = np.zeros(0, np.int64)
            if clock:
                out.update(finished_horizon=0, cap_hits=0)
            return out
        ew = _elementwise()
        direct = X.dtype == np.float64 and X.flags.c_contiguous
        chunk = self.chunk_size(k, direct)
        out["chunk"] = chunk
        pool = cp.get_default_memory_pool()
        used0 = pool.used_bytes()
        ma = max(self.m_aug, 1)
        Xw = None if direct else cp.empty((chunk, n), dtype=cp.float64)
        ACT = cp.empty((chunk, ma), dtype=cp.float64)
        W = cp.empty((chunk, ma), dtype=cp.float64)
        VLIST = cp.empty((chunk, ma), dtype=cp.int32)
        VPOS = cp.empty((chunk, ma), dtype=cp.int32)
        STI = cp.zeros((chunk, N_SI), dtype=cp.int64)
        STD = cp.zeros((chunk, N_SD), dtype=cp.float64)
        peak = pool.used_bytes() - used0
        status_all = np.zeros(k, np.int8)                # ST_ACTIVE (0) for rows a stop leaves unfinished / unreached
        moves_all = np.zeros(k, np.int64)
        steps_all = np.zeros(k, np.int64)
        kept: dict[str, list] = {"act": [], "w": [], "vlist": [], "vpos": [], "nviol": [], "F": []}
        bd = self.threads
        is_int_dtype = np.issubdtype(X.dtype, np.integer)
        launches = 0
        max_launch = 0.0
        cap_hits = 0
        n_chunks = -(-k // chunk)
        # with a deadline, launches are kept to about 1/8 of it, so the overshoot past a deadline is bounded likewise
        launch_ms = self.launch_ms if cap_s is None else min(self.launch_ms, max(float(cap_s) * 125.0, LAUNCH_MS_MIN))
        # every call starts with a short slice: the step cost can change by orders of magnitude between calls (an armed
        # cutoff row spans the whole objective), and a slice calibrated on cheap steps could outlive the watchdog
        self._iters = min(self._iters, ITERS_START)
        stopped = False
        for ci, s in enumerate(range(0, k, chunk)):
            if should_stop is not None and should_stop():
                stopped = True
                break
            kc = min(chunk, k - s)
            tc = time.perf_counter()
            deadline = None if cap_s is None else t0 + float(cap_s) * (ci + 1) / n_chunks
            late = deadline is not None and tc >= deadline       # reached after its deadline: projected only
            prof = dict(k=kc, cum=[0], t=[tc - t0], end=0.0, H_eff=0, done=False, cut=False, cap_hit=False)
            prof_chunks.append(prof)
            if direct:
                Xc = X[s:s + kc]
            else:
                Xc = Xw[:kc]
                cp.copyto(Xc, X[s:s + kc], casting="unsafe")
            ew["prep"](self.d_lb, self.d_ub, self.d_isint, Xc)
            if d_sidx is None:
                sx, use_sx = self.d_sidx0, np.int32(0)
            else:
                sx, use_sx = d_sidx[s:s + kc], np.int32(1)
            if self.m_aug == 0:
                st = np.zeros((kc, N_SI), np.int64)
                st[:, S_STATUS] = ST_FEASIBLE
                prof["done"] = True
            else:
                self._k_init((kc,), (bd,), (
                    np.int32(n), np.int32(self.m_aug), self.d_row_ptr, self.d_col_idx, self.d_row_val,
                    self.d_rl, self.d_ru, self.d_rs, sx, use_sx, self.d_long, np.int32(self.n_long), np.int32(LONG_ROW),
                    Xc, ACT, W, VLIST, VPOS, STI, STD, np.uint64(self.seed), np.int64(s), np.uint64(self._calls),
                    np.float64(self.viol_tol), np.float64(self.viol_cap)))
                cum = 0
                cut = False
                while True:
                    tl = time.perf_counter()
                    iters = int(self._iters)
                    if hz is not None:
                        iters = min(iters, hz - cum)                 # never past the horizon (0: stop rules only)
                    if late:
                        iters = 0
                    self._k_steps((kc,), (bd,), (
                        np.int32(n), np.int32(self.m_aug), self.d_row_ptr, self.d_col_idx, self.d_row_val,
                        self.d_col_ptr, self.d_row_idx, self.d_col_val, self.d_rl, self.d_ru, self.d_rs,
                        sx, use_sx, self.d_lb, self.d_ub, self.d_vtype, self.d_cobj, np.int32(self.use_obj),
                        Xc, ACT, W, VLIST, VPOS, STI, STD,
                        np.int64(self.cfg.max_steps), np.int64(self.cfg.stall_steps), np.int32(self.max_candidates),
                        np.float64(self.cfg.weight_bump), np.float64(self.p_random_walk),
                        np.float64(self.viol_tol), np.float64(self.viol_cap), np.int32(self.row_scan_cap),
                        np.int32(1 if self.row_scan_filter else 0), np.int32(iters)))
                    active = int(cp.count_nonzero(STI[:kc, S_STATUS] == ST_ACTIVE))
                    launches += 1
                    cum += iters
                    tn = time.perf_counter()
                    dt = tn - tl
                    max_launch = max(max_launch, dt)
                    prof["cum"].append(cum)
                    prof["t"].append(tn - t0)
                    if active == 0:
                        prof["done"] = True
                        break
                    if hz is not None and cum >= hz:                 # the horizon: every active block has hz steps
                        cut = True
                        break
                    if late or (deadline is not None and tn >= deadline):
                        cut = True                                   # the deadline: cut at the equal count reached
                        prof["cap_hit"] = True
                        cap_hits += 1
                        break
                    if should_stop is not None and should_stop():
                        stopped = True
                        break
                    # aim each launch at ~launch_ms (watchdog); grow at most 4x per launch
                    target = launch_ms / 1000.0
                    scale = target / dt if dt > 0 else 4.0
                    self._iters = int(min(max(iters * min(scale, 4.0), 1), 1 << 20))
                st = cp.asnumpy(STI[:kc])
                if cut:
                    stc = st[:, S_STATUS]
                    stc[stc == ST_ACTIVE] = ST_HORIZON               # host-side status only (STI is not rewritten)
                prof["cut"] = cut
                prof["H_eff"] = cum
            if not direct:
                (ew["wb_int"] if is_int_dtype else ew["wb_float"])(Xc, X[s:s + kc])
            status_all[s:s + kc] = st[:, S_STATUS]
            moves_all[s:s + kc] = st[:, S_MOVES]
            steps_all[s:s + kc] = st[:, S_STEPS]
            out["moves"] += int(st[:, S_MOVES].sum())
            out["local_minima"] += int(st[:, S_LMIN].sum())
            out["steps"] += int(st[:, S_STEPS].sum())
            if keep_state and self.m_aug:
                kept["act"].append(cp.asnumpy(ACT[:kc]))
                kept["w"].append(cp.asnumpy(W[:kc]))
                kept["vlist"].append(cp.asnumpy(VLIST[:kc]))
                kept["vpos"].append(cp.asnumpy(VPOS[:kc]))
                kept["nviol"].append(st[:, S_NVIOL].copy())
                kept["F"].append(cp.asnumpy(STD[:kc, D_F]))
            peak = max(peak, pool.used_bytes() - used0)
            prof["end"] = time.perf_counter() - t0
            if stopped:
                break
        cp.cuda.Stream.null.synchronize()
        del Xw, ACT, W, VLIST, VPOS, STI, STD, d_sidx
        out["finished_feasible"] = int(np.sum(status_all == ST_FEASIBLE))
        out["finished_stall"] = int(np.sum(status_all == ST_STALL))
        out["interrupted"] = int(np.sum(status_all == ST_ACTIVE))
        out["finished_budget"] = int(np.sum(status_all == ST_BUDGET)) + out["interrupted"]
        if clock:
            out["finished_horizon"] = int(np.sum(status_all == ST_HORIZON))
            out["finished_budget"] += out["finished_horizon"]
            out["cap_hits"] = cap_hits
        out["launches"] = launches
        out["max_launch_s"] = max_launch
        out["peak_bytes"] = int(peak)
        out["seconds"] = time.perf_counter() - t0
        self.last_status, self.last_moves, self.last_steps = status_all, moves_all, steps_all
        self.last_profile["seconds"] = out["seconds"]
        if keep_state:
            self.state = {kk: (np.concatenate(v) if v else None) for kk, v in kept.items()}
        return out
