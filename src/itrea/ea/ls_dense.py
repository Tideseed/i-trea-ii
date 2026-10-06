"""Vectorised local search with a full rescore per step (`repair.method: dense`) — the BASELINE, not the method.

Why it exists: it is the obvious array-programming formulation (one SpMM-sized pass per step, no custom kernel), so
it shows what the incremental feasibility jump (`ls_fj_cpu`, `ls_fj_gpu`) buys. The design critique measured this
formulation at O(nnz·P) work per single move; keep it for that comparison (D-002 E1, `tools/bench_ls.py`).

It shares the move rule's ingredients with feasibility jump — the same augmented rows (the objective cutoff row
`cmin·x <= rhs` under `objective_mode: cutoff_row`, via `ls_fj_cpu.AugmentedRows` and `cutoff_side`), per-individual
row weights starting at 1.0 every call and bumped by `weight_bump` on every violated row at a local minimum, the same
row violation v_i and violated-set tolerance, the same stop rules (feasible / `max_steps` moves / `stall_steps` steps
without a lower best weighted violation) — but it scores differently: every step rescans EVERY non-zero for every
individual and applies that individual's best improving moves among all variables that appear in a violated row.

Candidate value per variable (one or a few, not FJ's list):
  binary            the flip;
  general integer   x+1, x-1, and the tightening value of the column's violated row with the largest w_i·v_i,
                    rounded in the direction of the move; the best of the three;
  continuous        that tightening value, clipped to the bounds.
Score = Σ_{i in col j} w_i [v_i(act_i + a_ij d) - v_i(act_i)] (+ obj_weight·cmin_j·d / max|cmin| in `weighted` mode),
reduced per column with a sparse indicator SpMM. `moves_per_step` (default 1) moves per individual per step on
DISJOINT row sets: after each chosen move, every variable sharing a row with it is masked (two indicator SpMMs), so the
scores of the moves applied together add up exactly. With an active cutoff row, every costed variable shares that row,
so at most one costed move is applied per step. `steps` bounds the steps per call.

Memory: the work is chunked over individuals; `per_individual_bytes` counts the (entries, k) temporaries of a scoring
pass, the (m, k) row state and the (n, k) column state, and `chunk = mem_budget_bytes // per_individual_bytes`.
Works on NumPy and CuPy (`dp.bk`); X lives on the same backend and is modified in place.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sps

from itrea.config import RepairCfg
from itrea.ea.evaluate import DeviceProblem
from itrea.ea.ls_fj_cpu import (IMPROVING, ST_BUDGET, ST_FEASIBLE, ST_STALL, VIOL_CAP, VIOL_TOL, VT_BIN, VT_CONT,
                                VT_INT, AugmentedRows, cutoff_side, var_types, weighted_objective)


def _tol(side: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        t = np.minimum(VIOL_TOL * np.maximum(1.0, np.abs(side)), VIOL_CAP)
    return np.where(np.isfinite(side), t, np.inf)


class _ClassBlock:
    """The non-zeros of A_aug whose column belongs to one variable class, with a column-reduce indicator."""

    def __init__(self, bk, A_aug: sps.csr_matrix, cols: np.ndarray):
        coo = A_aug.tocoo()
        local = np.full(A_aug.shape[1], -1, dtype=np.int64)
        local[cols] = np.arange(cols.size)
        keep = local[coo.col] >= 0
        rows, cc, vals = coo.row[keep], coo.col[keep], coo.data[keep]
        self.size = int(rows.size)
        self.n_sub = int(cols.size)
        xp = bk.xp
        self.cols = xp.asarray(cols.astype(np.int64))
        self.rows_e = xp.asarray(rows.astype(np.int64))
        self.cols_e = xp.asarray(cc.astype(np.int64))
        self.lcol_e = xp.asarray(local[cc])
        self.vals_e = xp.asarray(vals.astype(np.float64))
        R = sps.csr_matrix((np.ones(self.size), (local[cc], np.arange(self.size))), shape=(self.n_sub, self.size))
        self.R = bk.to_device_sparse(R, dtype=np.float64)


class DenseLS:
    """Vectorised full-rescore local search (baseline). Same constructor, `set_cutoff` and `run` as FJCpu / FJGpu.

    `run(X)` returns {moves, local_minima, steps, finished_feasible, finished_stall, finished_budget, seconds, chunk,
    peak_bytes}. Keyword knob `p_random_walk` (default from cfg if present, else 0): at a local minimum apply the
    least-bad move with this probability."""

    def __init__(self, dp: DeviceProblem, cfg: RepairCfg, imp_tol: float = 1e-9, mem_budget_bytes: int = 256 << 20,
                 seed: int = 0, *, p_random_walk: float | None = None):
        bk = dp.bk
        xp = bk.xp
        self.dp, self.cfg, self.bk = dp, cfg, bk
        self.imp_tol = float(imp_tol)
        self.mem_budget_bytes = int(mem_budget_bytes)
        self.seed = int(seed)
        pr = getattr(cfg, "p_random_walk", 0.0) if p_random_walk is None else p_random_walk
        self.p_random_walk = float(pr)
        self._calls = 0
        self.last_status: np.ndarray | None = None
        prob = dp.prob
        self.n, self.m = prob.n, prob.m
        self.key_offset = float(dp.key_offset)
        self.rows = AugmentedRows(dp, cfg)
        self.m_aug = self.rows.m_aug
        vt, lb, ub = var_types(prob)
        self.vtype = vt
        self.has_cont = bool(np.any(~np.asarray(prob.isint, bool)))
        cobj, self.use_obj = weighted_objective(prob, cfg)
        A = self.rows.A
        Ai = A.copy()
        Ai.data = np.ones_like(Ai.data)
        self.A = bk.to_device_sparse(A, dtype=np.float64)
        self.Ai = bk.to_device_sparse(Ai, dtype=np.float64)
        self.AiT = bk.to_device_sparse(Ai.T.tocsr(), dtype=np.float64)
        self.rl = xp.asarray(self.rows.rl)
        self.ru = xp.asarray(self.rows.ru)
        self.rs = xp.asarray(self.rows.rs)
        self.tl = xp.asarray(_tol(self.rows.rl))
        self.tu = xp.asarray(_tol(self.rows.ru))
        self.lb = xp.asarray(lb)
        self.ub = xp.asarray(ub)
        self.isint = xp.asarray(np.asarray(prob.isint, bool))
        self.cobj = xp.asarray(np.asarray(cobj[:self.n], dtype=np.float64)) if self.use_obj else None
        self.blocks = {}
        for cls in (VT_BIN, VT_INT, VT_CONT):
            cols = np.flatnonzero(vt == cls)
            if cols.size:
                blk = _ClassBlock(bk, A, cols)
                if blk.size:
                    self.blocks[cls] = blk
        self.max_entries = max([b.size for b in self.blocks.values()], default=0)
        self.cut_active = False
        self.cut_rhs = float("inf")
        self._rng = np.random.default_rng(self.seed)

    # ---------------------------------------------------------------- cutoff
    def set_cutoff(self, inc_key: float | None) -> None:
        if not self.rows.has_cut:
            self.cut_active, self.cut_rhs = False, float("inf")
            return
        rhs = cutoff_side(inc_key, self.key_offset, self.imp_tol, self.rows.integral)
        self.cut_active, self.cut_rhs = bool(np.isfinite(rhs)), rhs
        self.rows.set_cutoff_side(rhs)
        m = self.m
        self.ru[m] = rhs
        self.rs[m] = float(self.rows.rs[m])
        self.tu[m] = float(_tol(np.array([rhs]))[0])

    # ---------------------------------------------------------------- memory
    @property
    def per_individual_bytes(self) -> int:
        """Bytes of temporaries per individual in a chunk: 7 float64 (entries, k) arrays in a scoring pass, 8 (m, k)
        row-state arrays, 12 (n, k) column-state arrays."""
        return int(8 * (7 * self.max_entries + 8 * max(self.m_aug, 1) + 12 * self.n) + 64)

    def chunk_size(self, k: int) -> int:
        return max(1, min(k, int(self.mem_budget_bytes // self.per_individual_bytes)))

    # ------------------------------------------------------------------- run
    def run(self, X, should_stop=None) -> dict:
        """Improve every row of X in place. `should_stop()` (optional) is polled before every chunk and every step;
        when it returns True the chunk in progress is written back as it stands (its active rows count as
        `finished_budget`), later chunks are left untouched; both kinds count in `interrupted`."""
        bk, xp = self.bk, self.bk.xp
        t0 = time.perf_counter()
        k, n = int(X.shape[0]), int(X.shape[1])
        if n != self.n:
            raise ValueError(f"X has {n} columns, the problem has {self.n}")
        is_int_dtype = np.issubdtype(np.dtype(X.dtype), np.integer)
        if is_int_dtype and self.has_cont:
            raise TypeError("integer gene dtype with continuous variables in the problem")
        self._calls += 1
        out = dict(moves=0, local_minima=0, steps=0, finished_feasible=0, finished_stall=0, finished_budget=0,
                   interrupted=0, seconds=0.0, chunk=0, peak_bytes=0)
        status_all = np.zeros(k, np.int8)               # 0 = not reached (a stop before this row's chunk)
        if k == 0:
            self.last_status = status_all
            return out
        chunk = self.chunk_size(k)
        out["chunk"] = chunk
        used0 = bk.mem_used_bytes()
        peak = 0
        n_cut = 0
        for s in range(0, k, chunk):
            if should_stop is not None and should_stop():
                break
            kc = min(chunk, k - s)
            XT = xp.ascontiguousarray(X[s:s + kc].astype(xp.float64).T)          # (n, kc)
            XT = xp.where(xp.isfinite(XT), XT, 0.0)
            XT = xp.where(self.isint[:, None], xp.rint(XT), XT)
            XT = xp.clip(XT, self.lb[:, None], self.ub[:, None])
            st, cnt, pk = self._run_chunk(XT, kc, should_stop)
            n_cut += cnt.pop("interrupted", 0)
            peak = max(peak, pk - used0)
            if is_int_dtype:
                X[s:s + kc] = xp.rint(XT.T).astype(X.dtype)
            else:
                X[s:s + kc] = XT.T.astype(X.dtype)
            status_all[s:s + kc] = st
            for key in ("moves", "local_minima", "steps"):
                out[key] += cnt[key]
            del XT
        bk.synchronize()
        out["finished_feasible"] = int(np.sum(status_all == ST_FEASIBLE))
        out["finished_stall"] = int(np.sum(status_all == ST_STALL))
        out["interrupted"] = int(np.sum(status_all == 0)) + n_cut
        out["finished_budget"] = int(np.sum(status_all == ST_BUDGET)) + int(np.sum(status_all == 0))
        out["peak_bytes"] = int(max(peak, 0))
        out["seconds"] = time.perf_counter() - t0
        self.last_status = status_all
        return out

    # -------------------------------------------------------------- internals
    def _measure(self, act, W):
        xp = self.bk.xp
        rl, ru = self.rl[:, None], self.ru[:, None]
        below, above = act < rl, act > ru
        raw = xp.where(below, rl - act, 0.0) + xp.where(above, act - ru, 0.0)
        violated = (below & (rl - act > self.tl[:, None])) | (above & (act - ru > self.tu[:, None]))
        F = (W * (raw / self.rs[:, None])).sum(axis=0)
        return violated, raw, F

    def _scatter_max(self, out, idx, vals):
        self.bk.xp.maximum.at(out, idx, vals)

    def _rowviol(self, a, rl, ru, rs):
        xp = self.bk.xp
        return (xp.where(a < rl, rl - a, 0.0) + xp.where(a > ru, a - ru, 0.0)) / rs

    def _pass(self, blk: _ClassBlock, act, W, D):
        """Δ weighted violation (n_sub, k) of moving each class column by D (n_sub, k)."""
        rl = self.rl[blk.rows_e][:, None]
        ru = self.ru[blk.rows_e][:, None]
        rs = self.rs[blk.rows_e][:, None]
        a0 = act[blk.rows_e]
        v0 = self._rowviol(a0, rl, ru, rs)
        a1 = a0 + blk.vals_e[:, None] * D[blk.lcol_e]
        dv = W[blk.rows_e] * (self._rowviol(a1, rl, ru, rs) - v0)
        del a0, a1, v0
        return blk.R @ dv

    def _jump(self, blk: _ClassBlock, XT, act, W, violated, raw, kc):
        """Tightening value (n_sub, k) of each class column for its violated row with the largest w·v (NaN if none).

        The segment argmax runs on int64 keys only: cupyx.scatter_max on float64 loses precision (measured
        2026-10-06: 5.78665516 came back as 5.78665495), so the priority is compared through its IEEE bit pattern
        (order-preserving for non-negative doubles) and the winning entry is found by a second, index-valued pass."""
        xp = self.bk.xp
        re = blk.rows_e
        pr = xp.where(violated[re], W[re] * raw[re] / self.rs[re][:, None], 0.0)
        bits = pr.view(xp.int64)
        colmax = xp.zeros((blk.n_sub, kc), dtype=xp.int64)
        self._scatter_max(colmax, blk.lcol_e, bits)
        sel = (pr > 0) & (bits == colmax[blk.lcol_e])
        eidx = xp.where(sel, xp.arange(blk.size, dtype=xp.int64)[:, None], -1)
        del pr, bits, sel
        win = xp.full((blk.n_sub, kc), -1, dtype=xp.int64)
        self._scatter_max(win, blk.lcol_e, eidx)
        del eidx
        has = win >= 0
        e = xp.where(has, win, 0)                                   # (n_sub, k) winning entry per column
        ar = xp.arange(kc)[None, :]
        r = blk.rows_e[e]
        a = act[r, ar]
        rl, ru = self.rl[r], self.ru[r]
        side = xp.where(a < rl, rl, ru)
        t = XT[blk.cols_e[e], ar] + (side - a) / blk.vals_e[e]
        return xp.where(has, t, xp.nan)

    def _run_chunk(self, XT, kc, should_stop=None):
        bk, xp, cfg = self.bk, self.bk.xp, self.cfg
        inf = float("inf")
        ar = xp.arange(kc)
        act = self.A @ XT                                       # (m_aug, kc)
        W = xp.ones((self.m_aug, kc))
        violated, raw, F = self._measure(act, W)
        Fbest = F.copy()
        since = xp.zeros(kc, dtype=xp.int64)
        moves_i = xp.zeros(kc, dtype=xp.int64)
        status = xp.zeros(kc, dtype=xp.int8)
        active = xp.ones(kc, dtype=bool)
        cnt = dict(moves=0, local_minima=0, steps=0)
        peak = bk.mem_used_bytes()
        mps = max(1, int(cfg.moves_per_step))
        for _ in range(int(cfg.steps)):
            done = ~violated.any(axis=0) if self.m_aug else xp.ones(kc, dtype=bool)
            fin = active & done
            status = xp.where(fin, ST_FEASIBLE, status).astype(xp.int8)
            active &= ~done
            bud = active & (moves_i >= cfg.max_steps)
            status = xp.where(bud, ST_BUDGET, status).astype(xp.int8)
            active &= ~bud
            stl = active & (since >= cfg.stall_steps)
            status = xp.where(stl, ST_STALL, status).astype(xp.int8)
            active &= ~stl
            if not bool(active.any()):
                break
            if should_stop is not None and should_stop():
                cnt["interrupted"] = int(active.sum())       # these end as ST_BUDGET or ST_FEASIBLE below
                break
            score = xp.full((self.n, kc), inf)
            delta = xp.zeros((self.n, kc))
            for cls, blk in self.blocks.items():
                x = XT[blk.cols]
                lbc, ubc = self.lb[blk.cols][:, None], self.ub[blk.cols][:, None]
                cands = []
                if cls == VT_BIN:
                    cands.append(1.0 - 2.0 * x)
                else:
                    jt = self._jump(blk, XT, act, W, violated, raw, kc)
                    if cls == VT_INT:
                        cands.append(xp.ones_like(x))
                        cands.append(-xp.ones_like(x))
                        jt = xp.where(jt > x, xp.ceil(jt - 1e-9), xp.floor(jt + 1e-9))
                    cands.append(xp.where(xp.isnan(jt), 0.0, xp.clip(jt, lbc, ubc) - x))
                best_s = xp.full((blk.n_sub, kc), inf)
                best_d = xp.zeros((blk.n_sub, kc))
                for D in cands:
                    ok = (D != 0) & (x + D >= lbc) & (x + D <= ubc)
                    D = xp.where(ok, D, 0.0)
                    sc = xp.where(ok, self._pass(blk, act, W, D), inf)
                    better = sc < best_s
                    best_s = xp.where(better, sc, best_s)
                    best_d = xp.where(better, D, best_d)
                    del sc, better, ok
                score[blk.cols] = best_s
                delta[blk.cols] = best_d
                del cands, best_s, best_d, x
            if self.use_obj:
                score = score + self.cobj[:, None] * delta
            inviol = (self.AiT @ violated.astype(xp.float64)) > 0         # variables in a violated row
            score = xp.where(inviol & active[None, :], score, inf)
            peak = max(peak, bk.mem_used_bytes())
            Dlt = xp.zeros((self.n, kc))
            j1 = xp.argmin(score, axis=0)
            s1 = score[j1, ar]
            improving = active & (s1 < IMPROVING)
            lm = active & ~improving
            n_lm = int(lm.sum())
            if n_lm:
                W = W + cfg.weight_bump * (violated & lm[None, :])
                if self.p_random_walk > 0:
                    u = xp.asarray(self._rng.random(kc))
                    walk = lm & xp.isfinite(s1) & (u < self.p_random_walk)
                else:
                    walk = xp.zeros(kc, dtype=bool)
            else:
                walk = xp.zeros(kc, dtype=bool)
            take = improving | walk
            Dlt[j1, ar] = xp.where(take, delta[j1, ar], 0.0)
            n_moves = take.astype(xp.int64)
            if mps > 1:
                chosen = take.copy()
                for _ in range(mps - 1):
                    mark = (self.Ai @ (Dlt != 0).astype(xp.float64)) > 0      # rows touched so far
                    if self.rows.has_cut and not self.cut_active:
                        mark[self.m] = False                                   # an inactive cutoff row is no row
                    conflict = (self.AiT @ mark.astype(xp.float64)) > 0
                    score = xp.where(conflict, inf, score)
                    j2 = xp.argmin(score, axis=0)
                    s2 = score[j2, ar]
                    ok = chosen & (s2 < IMPROVING)
                    if not bool(ok.any()):
                        break
                    Dlt[j2, ar] = xp.where(ok, delta[j2, ar], Dlt[j2, ar])
                    n_moves += ok.astype(xp.int64)
                    chosen = ok
            XT += Dlt
            act = act + self.A @ Dlt
            moves_i += n_moves
            cnt["moves"] += int(n_moves.sum())
            cnt["local_minima"] += n_lm
            cnt["steps"] += int(active.sum())
            del score, delta, Dlt, inviol
            violated, raw, F = self._measure(act, W)
            imp = active & (F < Fbest - 1e-12 * xp.maximum(1.0, xp.abs(Fbest)))
            Fbest = xp.where(imp, F, Fbest)
            since = xp.where(active, xp.where(imp, 0, since + 1), since)
        # steps exhausted: whoever is still active is feasible now or out of budget
        done = ~violated.any(axis=0) if self.m_aug else xp.ones(kc, dtype=bool)
        status = xp.where(active & done, ST_FEASIBLE, status)
        status = xp.where(active & ~done, ST_BUDGET, status)
        return bk.asnumpy(status).astype(np.int8), cnt, peak
