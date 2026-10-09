"""Batched local search / repair of the children — the dispatcher (`repair.method`).

    LocalSearch(bk, dp, cfg: RepairCfg, imp_tol, mem_budget_bytes, seed, row_scales=None)
        set_cutoff(inc_key)   arm / refresh / disarm the objective cutoff row (`objective_mode: cutoff_row`); the
                              engine calls it whenever the incumbent key improves (None = no incumbent yet)
        run(X, should_stop=None, scale_idx=None, horizon=None, cap_s=None) -> dict
                              improve every row of X (k, n) in place; returns counters. `should_stop()` (optional) is
                              polled between kernel launches / row slices / steps; when it returns True the search
                              returns early (rows not reached stay as they were, rows in progress keep their last move).
                              `scale_idx` (k,): each row's index into `row_scales` (None: index 0). `horizon` /
                              `cap_s`: the repair clock's lane-step horizon and per-call deadline ("fj" only; passed
                              on only when given, so a call without them is exactly the call before they existed)

Methods (vault `10-design/Feasibility jump local search.md`):
    "fj"     feasibility jump. On the CuPy backend: `ls_fj_gpu.FJGpu` (one thread block per individual). On NumPy:
             `ls_fj_cpu.FJCpu` (Numba, one core — arm A4's repair and the GPU kernel's reference). If the backend is
             the GPU and the GPU module cannot be imported, this is an ImportError: a run labelled as a GPU arm must
             never silently repair on the CPU (the same rule as `get_backend`, design critique 2026-10-05).
    "dense"  `ls_dense.DenseLS`, the vectorised full-rescore baseline (NumPy or CuPy).
    "none"   only the projection every method starts with: non-finite genes -> 0, integer genes rounded
             half-to-even, every gene clipped to its bounds (integer bounds tightened to ceil(lb) / floor(ub)).

Every method returns at least {moves, local_minima, steps, finished_feasible, finished_stall, finished_budget,
interrupted, seconds} plus "method"; the implementation may add more (chunk, peak_bytes, act_drift, …). Rows cut short
by `should_stop` count in `finished_budget` (so the three finish counters still add up to k) and in `interrupted`.
Integer genes stay integral and inside their bounds in every row the search reached (a row skipped by a stop is left
untouched).

Row scales (the self-adaptive repair strategy, `StrategyCfg`). The row scale s_i of the search is `dp`'s own
(`dp.viol_norm_vec` / `count_rows` / `viol_norm`) unless `row_scales` lists several: "fj" and "dense" then keep one row
of divisors per scale (an (R, m_aug) table, `evaluate.row_scale_table`; everything else is shared — the scales differ
only in s_i) and `run(X, scale_idx=...)` repairs individuals of every scale in ONE call, each with its own row. On the
GPU that matters: the blocks of all scales run side by side, so the slowest individual's tail is paid once per call
(`ls_fj_gpu` module docstring). "none" has no row scale and ignores both (`row_scales` = [None]).

Continuous variables: the singleton columns are decoded in closed form by `continuous.SingletonDecoder`, which the
ENGINE applies (cfg.continuous == "auto"); this dispatcher does not decode, so `run` keeps the invariant "a feasible
input with an inactive cutoff is returned unchanged". Recommended engine order: decode -> LocalSearch.run -> decode
-> evaluate (the first decode gives the search the best slack values to start from, the second re-optimises them for
the final integer part). Every other continuous variable is moved by feasibility-jump values.
"""
from __future__ import annotations

import time

import numpy as np

from itrea.backend import Backend
from itrea.config import RepairCfg
from itrea.ea.evaluate import DeviceProblem

COUNTER_KEYS = ("moves", "local_minima", "steps", "finished_feasible", "finished_stall", "finished_budget")


class _Project:
    """repair.method == "none": projection only (chunked over individuals: one (chunk, n) temporary)."""

    def __init__(self, bk: Backend, dp: DeviceProblem, mem_budget_bytes: int):
        from itrea.ea.ls_fj_cpu import var_types

        xp = bk.xp
        self.bk = bk
        _, lb, ub = var_types(dp.prob)
        self.lb, self.ub = xp.asarray(lb), xp.asarray(ub)
        self.isint = xp.asarray(np.asarray(dp.prob.isint, bool))
        self.has_cont = bool(np.any(~np.asarray(dp.prob.isint, bool)))
        self.mem_budget_bytes = int(mem_budget_bytes)

    def set_cutoff(self, inc_key: float | None) -> None:
        pass

    def run(self, X, should_stop=None) -> dict:          # noqa: ARG002 — one fast pass, nothing to interrupt
        xp = self.bk.xp
        t0 = time.perf_counter()
        k, n = int(X.shape[0]), int(X.shape[1])
        is_int = np.issubdtype(np.dtype(X.dtype), np.integer)
        if is_int and self.has_cont:
            raise TypeError("integer gene dtype with continuous variables in the problem")
        chunk = max(1, min(max(k, 1), self.mem_budget_bytes // max(1, 16 * n)))
        for s in range(0, k, chunk):
            Xc = X[s:s + chunk].astype(xp.float64)
            Xc = xp.where(xp.isfinite(Xc), Xc, 0.0)
            Xc = xp.where(self.isint[None, :], xp.rint(Xc), Xc)
            Xc = xp.clip(Xc, self.lb[None, :], self.ub[None, :])
            X[s:s + chunk] = Xc.astype(X.dtype)
            del Xc
        out = {key: 0 for key in COUNTER_KEYS}
        out["finished_budget"] = k
        out["interrupted"] = 0
        out["seconds"] = time.perf_counter() - t0
        out["chunk"] = chunk
        return out


class LocalSearch:
    """Dispatches `cfg.method` to the implementation for this backend (see the module docstring)."""

    def __init__(self, bk: Backend, dp: DeviceProblem, cfg: RepairCfg, imp_tol: float = 1e-9,
                 mem_budget_bytes: int = 256 << 20, seed: int = 0, row_scales=None, **kw):
        self.bk, self.dp, self.cfg = bk, dp, cfg
        self.method = cfg.method
        if cfg.method in ("fj", "dense") and row_scales is not None:
            kw = dict(kw, row_scales=list(row_scales))
        if cfg.method == "fj":
            if bk.is_gpu:
                try:
                    from itrea.ea.ls_fj_gpu import FJGpu
                except ImportError as err:
                    raise ImportError(
                        "repair.method='fj' on the CuPy backend needs itrea.ea.ls_fj_gpu.FJGpu, which could not be "
                        f"imported ({err}); refusing to fall back to the CPU implementation silently — use "
                        "backend='numpy' (arm A4) or repair.method='dense' explicitly") from err
                self.impl = FJGpu(dp, cfg, imp_tol, mem_budget_bytes, seed, **kw)
                self.impl_name = "fj_gpu"
            else:
                from itrea.ea.ls_fj_cpu import FJCpu
                self.impl = FJCpu(dp, cfg, imp_tol, mem_budget_bytes, seed, **kw)
                self.impl_name = "fj_cpu"
        elif cfg.method == "dense":
            from itrea.ea.ls_dense import DenseLS
            self.impl = DenseLS(dp, cfg, imp_tol, mem_budget_bytes, seed, **kw)
            self.impl_name = "dense_gpu" if bk.is_gpu else "dense_cpu"
        elif cfg.method == "none":
            if kw:
                raise TypeError(f"repair.method='none' takes no options, got {sorted(kw)}")
            self.impl = _Project(bk, dp, mem_budget_bytes)
            self.impl_name = "none"
        else:
            raise ValueError(f"unknown repair.method {cfg.method!r}")
        self.row_scales = list(getattr(self.impl, "row_scales", [None]))

    def set_cutoff(self, inc_key: float | None) -> None:
        self.impl.set_cutoff(inc_key)

    def run(self, X, should_stop=None, scale_idx=None, horizon=None, cap_s=None, pace=None) -> dict:
        """Repair X in place. `horizon` / `cap_s` (the repair clock) and `pace` (the EA's GPU back-off, D-018: a
        callable that may sleep between launches / slices and returns the seconds slept) reach the feasibility jump
        only; any of them with another method is an error."""
        clock = {k: v for k, v in (("horizon", horizon), ("cap_s", cap_s), ("pace", pace)) if v is not None}
        if clock and self.method != "fj":
            raise ValueError(f"horizon / cap_s / pace need repair.method 'fj', not {self.method!r}")
        if self.method == "none":
            out = self.impl.run(X, should_stop=should_stop)
        else:
            out = self.impl.run(X, should_stop=should_stop, scale_idx=scale_idx, **clock)
        out["method"] = self.impl_name
        return out
