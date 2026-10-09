"""The LP decode (`repair.continuous: lp_decode`, vault D-017, issue #27): a HiGHS worker process fixes the integer
part of a few candidates per generation and solves the LP over the continuous variables.

Why. The closed-form decoder (`ea.continuous`, mode "auto") sets only singleton continuous columns. Every other
continuous variable — an "active" column: continuous, lb < ub, in two or more rows (`active_columns`) — is moved by the
feasibility jump's single-variable moves, which rarely land on the LP optimum of the continuous part (polish) and cannot
coordinate several continuous variables to complete an infeasible point (completion). With the integers fixed the rest
is an LP, which HiGHS solves warm-started from the previous basis.

Mode. "lp_decode" means "auto" plus the LP decode: the singleton decoder is built in both modes, so the treatment differs
from its "auto" control by the LP decode alone. An instance without an active column is inert: no worker, no counter, no
event; the start-up record says `lp_decode = {state: "inert", reason}`. `cont_obj` = some continuous column has a
non-zero cost; without it no LP can improve a feasible point's key, so only completion rows are offered.

The engine side (`ea.engine`, module docstring "LP decode"): every generation, after the survival, over the whole
candidate set, it polls the worker and, if the worker is idle, sends ONE batch chosen by `select_offers` (no random
draw): F (polish) = feasible rows by (key, id), at most N = `lp_decode_per_gen`; I (completion) = infeasible rows better
than the incumbent by imp_tol (all infeasible rows in the feasibility phase; with no continuous cost also not beyond the
solver's proven bound), by (V, key, id), at most OFFER_FACTOR * N. Rows that are duplicates, non-finite, unchanged
solver finds, decoded copies or already offered are never offered; a row counts as offered once a reply has screened or
solved it (rows the worker left NOT_SOLVED or AFTER_DEADLINE stay eligible). Only integer parts travel, in the smallest
of int8 / int16 / int32 that holds every integer column's bounds when they are finite integers (`int_part_dtype`; ca,
all binary: int8), else float64; the reply carries the continuous values (float64) of the rows solved with a
primal-feasible point.

The worker (`LPDecoder`, a "spawn" process started at t0 — the read of the instance and every solve are on the clock,
D-002): one persistent `highspy.Highs` (output off, 1 thread, `solve_relaxation`, presolve at its default), niced by
NICE; `OMP/OPENBLAS/MKL_NUM_THREADS=1` and `CUDA_VISIBLE_DEVICES=""` in its environment. It reads the instance file
itself and refuses to work (state "failed") unless the digest of its column names and integrality equals the EA's
(`model_digest`), so a column-order mismatch can never produce wrong vectors. Until it replies "ready" it counts as busy.
Per batch (`_DecodeSolver.solve`):
  1. cache: a row whose integer part was already solved (blake2b of its bytes; also a repeat inside the batch) is CACHED;
  2. range screen, I rows only: every row's activity range over the continuous columns' bounds, plus the fixed integer
     activity, must meet [rl - tol, ru + tol], tol = feas_tol * max(1, |side|) — a necessary condition only; a failing
     row is RANGE_FAIL. I rows are screened in priority order until enough have passed;
  3. budget (`split_budget`): s_I = min(n_I', max(N - n_F', N // 2)), s_F = min(n_F', N - s_I) — two and two with N = 4
     when both kinds exist, a kind that is short passes its share to the other; F rows are solved first, then I rows;
     rows beyond the budget are NOT_SOLVED;
  4. each solve: `changeColsBounds(integer columns, x, x)`; if the previous solve was not optimal and an optimal basis
     exists, `setBasis(that anchor)`; `time_limit = getRunTime() + L` (HiGHS's limit is cumulative over runs) with
     L = cold_limit while no anchor exists (priming), else lp_time_limit for F rows and I_LIMIT_FACTOR * lp_time_limit for
     I rows; `run()`; the row is OK iff HiGHS holds a primal-FEASIBLE point (an optimum, or a feasible point at the time
     limit — counted as a time-limit hit too), else INFEASIBLE / TIME_LIMIT / OTHER; on "Optimal" the basis becomes the
     anchor. No solve starts at or after the deadline (t0 + budget, `time.monotonic()`, system-wide): AFTER_DEADLINE.

Statistics. Every reply carries the worker's cumulative statistics: requests, rows, solves, optimal, ok, simplex
iterations, LP seconds (`solve_s_total` / `solve_s_max`, wall) and CPU seconds (`solve_cpu_s_total`), cache hits,
screen passes / fails, basis restores (and errors), after-deadline rows, HiGHS run time, maxrss, process CPU seconds;
and, for D-017's M2, the priming / warm split: `priming_solves`, `priming_s`, `priming_first_status` and
`priming_first_s` (the FIRST cold solve, set once), `priming_last_status`, `priming_time_limit_hits`, `warm_solves`,
`warm_time_limit_hits` (warm = an anchor basis existed), `time_limit_hits` (all) and `time_limit_hits_starved` (a hit
whose solve got less than STARVED_CPU_SHARE of a core: the worker was starved, not the LP slow — HiGHS's limit is wall
time). Per row the reply has the wall seconds `secs`, the CPU seconds `cpu` and the limit. The worker stamps its own
clock (`time.monotonic()`, system-wide) as `t_sent` on "ready" and on every reply, so the EA records when the worker
was ready and how long a finished reply waited before it was read (D-017 M0).

IPC. Pipe EOF, OSError or a broken pipe end the worker quietly; any other exception is sent back as ("failed", repr,
traceback). At most one request is outstanding. `offer()` refuses unless idle and returns once the pipe has taken the
request (on ca the worker reads most of it meanwhile). `poll()` never waits for the worker to answer — it reads only
when a message has begun to arrive — but the recv() of a large reply (≤ N × n_cont float64, about 5 MB on ca) waits
while the worker writes the rest. Both are timed on the EA side (`LPDecoder.ipc`: wall seconds and payload bytes of
every send and read) and count in the engine's `lpd_host_s`. Shared memory is deferred until that measurement asks for
it (D-017 M1). A failure (`_fail`: digest mismatch, a worker exception, a dead worker, a failed send) never waits
either: the worker gets SIGTERM, the pipe is closed, the process is reaped later without blocking; only `stop()` joins
(so the PID is gone) — it is idempotent and runs after the clock.

Constants (echoed in the start-up record; any of them must become a `_LATE_DEFAULTS` configuration field before it is
changed): OFFER_FACTOR = 3, I_LIMIT_FACTOR = 0.5, NICE = 10, CACHE_CAP = 65,536 integer parts; STARVED_CPU_SHARE = 0.5
(a diagnostic threshold only).

This module imports no other `itrea` module (the spawned worker unpickles its target from here) and never CuPy; highspy
is imported inside the worker.
"""
from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
import time
import traceback

import numpy as np
import scipy.sparse as sps

OFFER_FACTOR = 3              # at most OFFER_FACTOR * N completion (I) rows offered per batch
I_LIMIT_FACTOR = 0.5          # an I row's warm LP gets this share of lp_time_limit (cuOpt: 2 s / 1 s)
NICE = 10                     # os.nice of the worker
CACHE_CAP = 65_536            # integer parts remembered as solved (oldest forgotten first)
STARVED_CPU_SHARE = 0.5       # diagnostic only: a time-limit hit whose solve got < this share of a core was starved

KIND_F, KIND_I = 0, 1         # polish (feasible source) / completion (infeasible source)
OK, INFEASIBLE, TIME_LIMIT, OTHER, RANGE_FAIL, CACHED, NOT_SOLVED, AFTER_DEADLINE = range(8)
CODE_NAMES = ("ok", "infeasible", "time_limit", "other", "range_fail", "cached", "not_solved", "after_deadline")
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def constants() -> dict:
    """The module's constants, for the start-up record and the summary."""
    return dict(offer_factor=OFFER_FACTOR, i_limit_factor=I_LIMIT_FACTOR, nice=NICE, cache_cap=CACHE_CAP,
                starved_cpu_share=STARVED_CPU_SHARE)


# ------------------------------------------------------------------------------------------------ pure functions
def active_columns(prob) -> np.ndarray:
    """Columns the LP decode acts on: continuous, lb < ub, in at least two rows (the singleton decoder owns the
    degree-1 columns). Used by both `plan_memory` and `Engine.prepare`."""
    n = int(prob.A.shape[1])
    if n == 0:
        return np.zeros(0, np.int64)
    deg = np.diff(sps.csc_matrix(prob.A).indptr)
    lb, ub = np.asarray(prob.lb, float), np.asarray(prob.ub, float)
    return np.flatnonzero(~np.asarray(prob.isint, bool) & (lb < ub) & (deg >= 2)).astype(np.int64)


def continuous_cost(prob) -> bool:
    """`cont_obj`: some continuous column has a non-zero objective coefficient."""
    isint = np.asarray(prob.isint, bool)
    return bool(np.any(np.asarray(prob.c, float)[~isint] != 0.0))


def model_digest(names, isint) -> str:
    """Digest of the column names (in order) and integrality — what the worker checks against its own read."""
    h = hashlib.blake2b(digest_size=16)
    names = [str(v) for v in names]
    h.update(str(len(names)).encode())
    for nm in names:
        h.update(nm.encode("utf-8"))
        h.update(b"\0")
    h.update(np.asarray(isint, bool).astype(np.uint8).tobytes())
    return h.hexdigest()


def int_part_dtype(lb: np.ndarray, ub: np.ndarray) -> np.dtype:
    """The payload dtype of the integer parts: the smallest of int8 / int16 / int32 that holds every integer column's
    bounds when they are all finite integers (so a rounded, clipped value is exact — an all-binary instance such as ca
    sends int8, a quarter of int32's bytes), else float64."""
    lb, ub = np.asarray(lb, float), np.asarray(ub, float)
    if lb.size == 0:
        return np.dtype(np.int8)
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)) and np.all(lb == np.round(lb))
            and np.all(ub == np.round(ub))):
        return np.dtype(np.float64)
    lo, hi = float(lb.min()), float(ub.max())
    for dt in (np.int8, np.int16, np.int32):
        info = np.iinfo(dt)
        if lo >= info.min and hi <= info.max:
            return np.dtype(dt)
    return np.dtype(np.float64)


def select_offers(key, viol, feasible, *, ids, dup, solver_find, decoded, offered, inc: float, dual_key: float,
                  imp_tol: float, per_gen: int, cont_obj: bool, infeasible: bool) -> tuple[np.ndarray, np.ndarray]:
    """The rows of one batch (indices into the candidate set): (F, I) in priority order. Host arrays over the candidate
    set: key, viol, feasible (after the survival's demotions), ids, and the exclusion masks dup (an exact duplicate of
    an earlier candidate), solver_find (an unchanged solver solution), decoded (carries ORIGIN_LP_DECODE), offered
    (offered before). inc = the incumbent key the survival used (+inf in the feasibility phase), dual_key = the solver's
    proven bound as a key (-inf if unknown). No random draw."""
    key = np.asarray(key, np.float64)
    viol = np.asarray(viol, np.float64)
    feasible = np.asarray(feasible, bool)
    ids = np.asarray(ids, np.int64)
    ok = np.isfinite(key) & np.isfinite(viol)
    for m in (dup, solver_find, decoded, offered):
        ok &= ~np.asarray(m, bool)
    N = int(per_gen)
    F = np.zeros(0, np.int64)
    if cont_obj:
        f = np.flatnonzero(ok & feasible)
        F = f[np.lexsort((ids[f], key[f]))][:N].astype(np.int64)
    I = np.zeros(0, np.int64)
    if infeasible:
        i = np.flatnonzero(ok & ~feasible)
        if math.isfinite(inc):
            i = i[key[i] < inc - imp_tol * max(1.0, abs(inc))]
        if not cont_obj and math.isfinite(dual_key):
            # the key then depends on the integers only: no completion can beat the proven bound
            i = i[key[i] >= dual_key - imp_tol * max(1.0, abs(dual_key))]
        I = i[np.lexsort((ids[i], key[i], viol[i]))][:OFFER_FACTOR * N].astype(np.int64)
    return F, I


def split_budget(n_f: int, n_i: int, per_gen: int) -> tuple[int, int]:
    """(s_F, s_I): LPs for polish and completion rows in one batch of budget N = per_gen."""
    N = int(per_gen)
    s_i = min(int(n_i), max(N - int(n_f), N // 2))
    s_f = min(int(n_f), N - s_i)
    return s_f, s_i


def continuous_activity_range(A, isint, lb, ub) -> tuple[np.ndarray, np.ndarray]:
    """(cmin, cmax) per row: the range of the row's continuous activity over the continuous columns' bounds (±inf when
    a bound is infinite)."""
    A = sps.csc_matrix(A, dtype=np.float64)
    m = A.shape[0]
    cont = np.flatnonzero(~np.asarray(isint, bool))
    if cont.size == 0 or m == 0:
        return np.zeros(m), np.zeros(m)
    C = A[:, cont].tocoo()
    a = C.data
    lo_b, hi_b = np.asarray(lb, float)[cont][C.col], np.asarray(ub, float)[cont][C.col]
    with np.errstate(invalid="ignore"):
        lo = np.where(a > 0, a * lo_b, a * hi_b)
        hi = np.where(a > 0, a * hi_b, a * lo_b)
    lo = np.where(np.isnan(lo), -np.inf, lo)
    hi = np.where(np.isnan(hi), np.inf, hi)
    with np.errstate(invalid="ignore"):
        cmin = np.bincount(C.row, weights=lo, minlength=m)
        cmax = np.bincount(C.row, weights=hi, minlength=m)
    return np.where(np.isnan(cmin), -np.inf, cmin), np.where(np.isnan(cmax), np.inf, cmax)


def range_screen(act_int: np.ndarray, cmin: np.ndarray, cmax: np.ndarray, rl: np.ndarray, ru: np.ndarray,
                 feas_tol: float) -> bool:
    """True if every row's reachable activity [act_int + cmin, act_int + cmax] meets [rl - tol, ru + tol],
    tol = feas_tol * max(1, |side|). A necessary condition for the fixed-integer LP to be feasible."""
    rl, ru = np.asarray(rl, float), np.asarray(ru, float)
    with np.errstate(invalid="ignore"):
        lo = act_int + cmin
        hi = act_int + cmax
        tl = feas_tol * np.maximum(1.0, np.abs(np.where(np.isfinite(rl), rl, 0.0)))
        tu = feas_tol * np.maximum(1.0, np.abs(np.where(np.isfinite(ru), ru, 0.0)))
        ok = (lo <= ru + tu) & (hi >= rl - tl)
    return bool(np.all(ok | np.isnan(lo) | np.isnan(hi)))


# ----------------------------------------------------------------------------------------------------- the worker
class _DecodeSolver:
    """The worker's state: one persistent HiGHS object over the instance, the screen's arrays, the cache, the anchor
    basis and the cumulative statistics. Built inside the worker process (module docstring)."""

    def __init__(self, path: str, digest: str, int_dtype: str, time_limit: float, cold_limit: float, feas_tol: float,
                 deadline: float, cache_cap: int = CACHE_CAP, hooks: dict | None = None):
        import highspy

        self.hs = highspy
        t = time.monotonic()
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        h.setOptionValue("solve_relaxation", True)
        if h.readModel(str(path)) == highspy.HighsStatus.kError:
            raise RuntimeError(f"HiGHS could not read {path}")
        self.read_s = time.monotonic() - t
        lp = h.getLp()
        n, m = int(lp.num_col_), int(lp.num_row_)
        vt = highspy.HighsVarType
        integ = list(lp.integrality_) if len(lp.integrality_) else [vt.kContinuous] * n
        isint = np.array([v in (vt.kInteger, vt.kImplicitInteger) for v in integ], dtype=bool)
        names = list(lp.col_names_) if len(lp.col_names_) == n else [f"x{j}" for j in range(n)]
        self.digest_ok = model_digest(names, isint) == digest
        self.h, self.n, self.m = h, n, m
        if not self.digest_ok:
            return
        a = lp.a_matrix_
        start, index = np.asarray(a.start_), np.asarray(a.index_)
        value = np.asarray(a.value_, dtype=np.float64)
        if a.format_ == highspy.MatrixFormat.kColwise:
            A = sps.csc_matrix((value, index, start), shape=(m, n))
        else:
            A = sps.csr_matrix((value, index, start), shape=(m, n)).tocsc()

        def fin(v):
            v = np.asarray(v, dtype=np.float64).copy()
            v[v >= 1e30] = np.inf
            v[v <= -1e30] = -np.inf
            return v

        lb, ub, rl, ru = fin(lp.col_lower_), fin(lp.col_upper_), fin(lp.row_lower_), fin(lp.row_upper_)
        self.int_idx = np.flatnonzero(isint).astype(np.int32)
        self.cont_idx = np.flatnonzero(~isint).astype(np.int64)
        self.A_int = A[:, self.int_idx].tocsr()
        self.cmin, self.cmax = continuous_activity_range(A, isint, lb, ub)
        self.rl, self.ru = rl, ru
        self.int_dtype = np.dtype(int_dtype)
        self.time_limit, self.cold_limit = float(time_limit), float(cold_limit)
        self.feas_tol, self.deadline = float(feas_tol), float(deadline)
        self.cache_cap = int(cache_cap)
        self.cache: dict[bytes, None] = {}            # insertion-ordered: the oldest is forgotten first
        self.anchor = None                            # basis of the last optimal solve
        self.last_optimal = False
        self.hooks = dict(hooks or {})
        self.cold_overrides = list(self.hooks.get("cold_limits", ()))
        # cumulative statistics (module docstring "Statistics"); every reply carries a copy
        self.stats = dict(requests=0, rows=0, solves=0, priming_solves=0, priming_s=0.0, priming_first_status=None,
                          priming_first_s=None, priming_last_status=None, priming_time_limit_hits=0, warm_solves=0,
                          warm_time_limit_hits=0, solve_s_total=0.0, solve_s_max=0.0, solve_cpu_s_total=0.0,
                          iterations=0, time_limit_hits=0, time_limit_hits_starved=0, optimal=0, ok=0, cache_hits=0,
                          screen_pass=0, screen_fail=0, restores=0, restore_errors=0, after_deadline=0, run_time=0.0,
                          ru_maxrss_mb=None, cpu_s=None)

    def ready_info(self) -> dict:
        return dict(pid=os.getpid(), read_s=round(self.read_s, 6), n=self.n, m=self.m,
                    n_int=int(self.int_idx.size), n_cont=int(self.cont_idx.size), nice=os.getpriority(os.PRIO_PROCESS, 0),
                    omp_num_threads=os.environ.get("OMP_NUM_THREADS"), cpus=sorted(os.sched_getaffinity(0)))

    # ------------------------------------------------------------------------------------------------- a batch
    def solve(self, rows: np.ndarray, kinds: np.ndarray, per_gen: int) -> dict:
        rows = np.asarray(rows)
        kinds = np.asarray(kinds, np.uint8)
        k = int(rows.shape[0])
        st = self.stats
        st["requests"] += 1
        st["rows"] += k
        codes = np.full(k, NOT_SOLVED, np.uint8)
        obj = np.full(k, np.nan)
        iters = np.zeros(k, np.int64)
        secs = np.zeros(k)
        cpu = np.zeros(k)
        limit = np.full(k, np.nan)
        status = [""] * k
        hashes = [hashlib.blake2b(np.ascontiguousarray(rows[r]).tobytes(), digest_size=16).digest()
                  for r in range(k)]
        seen: set[bytes] = set()
        for r in range(k):                                # 1. cache (and repeats inside the batch)
            if hashes[r] in self.cache or hashes[r] in seen:
                codes[r] = CACHED
                st["cache_hits"] += 1
            seen.add(hashes[r])
        f_rows = [r for r in range(k) if kinds[r] == KIND_F and codes[r] != CACHED]
        need_i = max(int(per_gen) - len(f_rows), int(per_gen) // 2)
        i_rows: list[int] = []
        for r in range(k):                                # 2. range screen, I rows in priority order
            if kinds[r] != KIND_I or codes[r] == CACHED or len(i_rows) >= need_i:
                continue
            x = rows[r].astype(np.float64)
            if range_screen(self.A_int @ x, self.cmin, self.cmax, self.rl, self.ru, self.feas_tol):
                i_rows.append(r)
                st["screen_pass"] += 1
            else:
                codes[r] = RANGE_FAIL
                st["screen_fail"] += 1
        s_f, s_i = split_budget(len(f_rows), len(i_rows), per_gen)   # 3. budget
        todo = [(r, KIND_F) for r in f_rows[:s_f]] + [(r, KIND_I) for r in i_rows[:s_i]]
        conts, ok_rows = [], []
        for r, kind in todo:                              # 4. solves, F first
            if time.monotonic() >= self.deadline:
                codes[r] = AFTER_DEADLINE
                st["after_deadline"] += 1
                continue
            code, o, it, dt, dc, L, s, vals = self._solve_one(rows[r], kind)
            self._remember(hashes[r])
            codes[r], obj[r], iters[r], secs[r], cpu[r], limit[r], status[r] = code, o, it, dt, dc, L, s
            if code == OK:
                ok_rows.append(r)
                conts.append(vals)
        st["run_time"] = float(self.h.getRunTime())
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF)
            st["ru_maxrss_mb"] = round(ru.ru_maxrss / 1024, 1)
            st["cpu_s"] = round(ru.ru_utime + ru.ru_stime, 3)
        except (ImportError, OSError):
            pass
        n_c = int(self.cont_idx.size)
        return dict(codes=codes, ok_rows=np.asarray(ok_rows, np.int64),
                    conts=np.vstack(conts) if conts else np.zeros((0, n_c)), obj=obj, iters=iters, secs=secs,
                    cpu=cpu, limit=limit, status=status, stats=dict(st))

    def _remember(self, hsh: bytes) -> None:
        if self.cache_cap <= 0:
            return
        self.cache[hsh] = None
        while len(self.cache) > self.cache_cap:
            del self.cache[next(iter(self.cache))]

    def _solve_one(self, row: np.ndarray, kind: int):
        hs, h, st = self.hs, self.h, self.stats
        sleep = self.hooks.get("sleep_s")
        if sleep:
            time.sleep(float(sleep))                      # tests only: a slow batch
        x = np.asarray(row, dtype=np.float64)
        h.changeColsBounds(int(self.int_idx.size), self.int_idx, x, x)
        if not self.last_optimal and self.anchor is not None and not self.hooks.get("no_restore"):
            if h.setBasis(self.anchor) == hs.HighsStatus.kError:   # ("no_restore": tests only, a negative control)
                st["restore_errors"] += 1
            else:
                st["restores"] += 1
        priming = self.anchor is None
        if priming:
            L = float(self.cold_overrides.pop(0)) if self.cold_overrides else self.cold_limit
        else:
            L = self.time_limit if kind == KIND_F else I_LIMIT_FACTOR * self.time_limit
        h.setOptionValue("time_limit", float(h.getRunTime()) + max(0.0, L))
        t, c = time.perf_counter(), time.process_time()
        h.run()
        dt, dc = time.perf_counter() - t, time.process_time() - c
        ms = h.getModelStatus()
        info = h.getInfo()
        it = int(info.simplex_iteration_count)
        feasible = int(info.primal_solution_status) == int(hs.SolutionStatus.kSolutionStatusFeasible)
        MS = hs.HighsModelStatus
        if feasible:
            code = OK
        elif ms == MS.kInfeasible:
            code = INFEASIBLE
        elif ms == MS.kTimeLimit:
            code = TIME_LIMIT
        else:
            code = OTHER
        hit = ms == MS.kTimeLimit
        if hit:
            st["time_limit_hits"] += 1
            if dc < STARVED_CPU_SHARE * dt:               # stopped by the wall clock while barely running
                st["time_limit_hits_starved"] += 1
        optimal = ms == MS.kOptimal
        if optimal:
            self.anchor = h.getBasis()                    # a copy (checked with highspy 1.15.1)
            st["optimal"] += 1
        self.last_optimal = optimal
        st["solves"] += 1
        st["iterations"] += it
        st["solve_s_total"] += dt
        st["solve_s_max"] = max(st["solve_s_max"], dt)
        st["solve_cpu_s_total"] += dc
        s = h.modelStatusToString(ms)
        if priming:
            st["priming_solves"] += 1
            st["priming_s"] += dt
            st["priming_last_status"] = s
            st["priming_time_limit_hits"] += int(hit)
            if st["priming_first_status"] is None:        # the first priming solve, kept (D-017 M2)
                st["priming_first_status"], st["priming_first_s"] = s, dt
        else:
            st["warm_solves"] += 1
            st["warm_time_limit_hits"] += int(hit)
        vals = None
        o = math.nan
        if code == OK:
            st["ok"] += 1
            vals = np.asarray(h.getSolution().col_value, dtype=np.float64)[self.cont_idx]
            o = float(info.objective_function_value)
        return code, o, it, dt, dc, L, s, vals


def _decoder_main(conn, path: str, digest: str, int_dtype: str, time_limit: float, cold_limit: float,
                  feas_tol: float, deadline: float, cache_cap: int = CACHE_CAP, hooks: dict | None = None) -> None:
    """Entry point of the spawned worker (module docstring): read, check the digest, "ready", then one reply per
    request until the pipe closes."""
    try:
        try:
            os.nice(NICE)
        except OSError:
            pass
        solver = _DecodeSolver(path, digest, int_dtype, time_limit, cold_limit, feas_tol, deadline, cache_cap, hooks)
        if not solver.digest_ok:
            conn.send(("failed", "model digest mismatch: the worker's column names / integrality differ from the "
                                 "EA's", ""))
            return
        # the worker's own clock (time.monotonic(), system-wide): when it was ready and when each reply left it
        conn.send(("ready", dict(solver.ready_info(), t_sent=time.monotonic())))
        while True:
            msg = conn.recv()
            if not isinstance(msg, tuple) or not msg or msg[0] == "stop":
                return
            if msg[0] == "solve":
                _, req_id, rows, kinds, per_gen = msg
                rep = solver.solve(rows, kinds, per_gen)
                rep["t_sent"] = time.monotonic()
                conn.send(("result", int(req_id), rep))
    except (EOFError, BrokenPipeError, ConnectionError):
        return                                            # the EA closed the pipe (or died): exit quietly
    except BaseException as e:  # noqa: BLE001 — anything else goes back to the EA as a failure
        try:
            conn.send(("failed", repr(e)[:1000], traceback.format_exc()[-8000:]))
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


class LPDecoder:
    """The LP decode worker as seen from the EA (module docstring).

        dec = LPDecoder(path, model_digest(prob.names, prob.isint), "int8", per_gen=4, time_limit=2.0,
                        cold_limit=10.0, feas_tol=1e-6, deadline=t0 + budget).start()      # AFTER t0
        msgs = dec.poll()    # never waits for an answer: [("ready", info) | ("result", req_id, reply) | ("failed", err,
                             # tb)]; reading a large reply waits for the rest of its transfer (module docstring "IPC")
        dec.offer(req_id, rows, kinds)   # False unless idle
        dec.stop()                   # True if a request was outstanding (lost); idempotent; joins the worker

    States: "not_started", "starting" (until "ready"; counts as busy), "idle", "busy" (one request outstanding),
    "failed" (digest mismatch, an exception in the worker, the worker died, or a send failed; the worker was signalled
    and the pipe closed, `error` says why), "stopped". A result's reply dict gains `t_read` (the EA's monotonic read
    time) and `read_s` (seconds in recv()); `ipc` holds the EA-side send / read seconds and bytes. `cache_cap` and
    `hooks` ({"sleep_s": s before every solve, "cold_limits": [L, ...] for the first cold solves, "no_restore": skip the
    basis restore}) exist for tests; a run uses the defaults."""

    def __init__(self, path, digest: str, int_dtype, per_gen: int, time_limit: float, cold_limit: float,
                 feas_tol: float, deadline: float = math.inf, *, cache_cap: int = CACHE_CAP, hooks: dict | None = None):
        self.path = str(path)
        self.digest = str(digest)
        self.int_dtype = np.dtype(int_dtype)
        self.per_gen = int(per_gen)
        self.time_limit, self.cold_limit = float(time_limit), float(cold_limit)
        self.feas_tol, self.deadline = float(feas_tol), float(deadline)
        self.cache_cap = int(cache_cap)
        self.hooks = dict(hooks) if hooks else None
        self.state = "not_started"
        self.pid: int | None = None
        self.exitcode: int | None = None
        self.error: str | None = None
        self.outstanding: int | None = None
        self.ready_info: dict | None = None
        # the EA thread's side of the pipe (module docstring "IPC"): wall seconds inside send() of a request and inside
        # recv() of a message that poll() found waiting, and the payload bytes (integer parts / continuous values)
        self.ipc = dict(offers=0, offer_s_total=0.0, offer_s_max=0.0, offer_bytes_max=0, reads=0, read_s_total=0.0,
                        read_s_max=0.0, read_bytes_max=0)
        self._proc = None
        self._conn = None
        self._t_start: float | None = None

    # ---------------------------------------------------------------------------------------------- lifecycle
    def start(self) -> "LPDecoder":
        if self._proc is not None:
            raise RuntimeError("LPDecoder already started")
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe(duplex=True)
        proc = ctx.Process(target=_decoder_main, name="itrea-lp-decoder", daemon=True,
                           args=(child, self.path, self.digest, self.int_dtype.name, self.time_limit, self.cold_limit,
                                 self.feas_tol, self.deadline, self.cache_cap, self.hooks))
        env = {v: "1" for v in _THREAD_VARS}
        env["CUDA_VISIBLE_DEVICES"] = ""                  # the worker must never create a CUDA context
        saved = {v: os.environ.get(v) for v in env}
        try:
            os.environ.update(env)                        # spawn copies the parent's environment at launch
            proc.start()
        finally:
            for v, old in saved.items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old
        child.close()                                     # the parent keeps only its end: EOF on either side's death
        self._proc, self._conn, self._t_start = proc, parent, time.monotonic()
        self.pid = proc.pid
        self.state = "starting"
        return self

    def elapsed(self) -> float:
        return time.monotonic() - self._t_start if self._t_start is not None else 0.0

    def offer(self, req_id: int, rows: np.ndarray, kinds: np.ndarray, per_gen: int | None = None) -> bool:
        """Send one batch (integer parts in the worker's integer-column order, kinds KIND_F / KIND_I). Refused (False)
        unless idle."""
        if self.state != "idle":
            return False
        rows = np.ascontiguousarray(np.asarray(rows).astype(self.int_dtype, copy=False))
        kinds = np.asarray(kinds, np.uint8)
        t = time.perf_counter()
        try:
            self._conn.send(("solve", int(req_id), rows, kinds, int(per_gen if per_gen is not None else self.per_gen)))
        except (OSError, EOFError) as e:
            self._fail(f"send failed: {e!r}")
            return False
        dt = time.perf_counter() - t
        ipc = self.ipc
        ipc["offers"] += 1
        ipc["offer_s_total"] += dt
        ipc["offer_s_max"] = max(ipc["offer_s_max"], dt)
        ipc["offer_bytes_max"] = max(ipc["offer_bytes_max"], int(rows.nbytes))
        self.state = "busy"
        self.outstanding = int(req_id)
        return True

    def poll(self) -> list[tuple]:
        """The messages waiting (see the class docstring); a dead worker becomes ("failed", ...). It never waits for the
        worker to answer: it reads only when `conn.poll()` says a message has begun to arrive — the recv() of a large
        reply then waits while the worker writes the rest of it (timed in `ipc`)."""
        out: list[tuple] = []
        if self.state == "failed":
            self._reap()
        if self.state in ("not_started", "failed", "stopped") or self._conn is None:
            return out
        try:
            while self._conn is not None and self.state not in ("failed", "stopped") and self._conn.poll():
                out.append(self._handle(self._recv()))
        except (EOFError, OSError):
            pass
        if self.state not in ("failed", "stopped") and not self._proc.is_alive():   # is_alive() reaps it
            try:                                          # it may have sent and exited between the two checks
                while self._conn is not None and self.state not in ("failed", "stopped") and self._conn.poll():
                    out.append(self._handle(self._recv()))
            except (EOFError, OSError):
                pass
            if self.state not in ("failed", "stopped"):
                self.exitcode = self._proc.exitcode
                self._fail(f"the decoder worker exited (exit code {self.exitcode})")
                out.append(("failed", self.error, ""))
        return out

    def _recv(self):
        """One waiting message; a result's reply dict gets the EA's read time `t_read` (time.monotonic()) and the
        seconds recv() took `read_s`."""
        t = time.perf_counter()
        msg = self._conn.recv()
        t_read = time.monotonic()
        dt = time.perf_counter() - t
        if isinstance(msg, tuple) and len(msg) > 2 and msg[0] == "result" and isinstance(msg[2], dict):
            rep = msg[2]
            rep["t_read"], rep["read_s"] = t_read, dt
            ipc = self.ipc
            ipc["reads"] += 1
            ipc["read_s_total"] += dt
            ipc["read_s_max"] = max(ipc["read_s_max"], dt)
            c = rep.get("conts")
            ipc["read_bytes_max"] = max(ipc["read_bytes_max"], int(getattr(c, "nbytes", 0)))
        elif isinstance(msg, tuple) and len(msg) > 1 and msg[0] == "ready" and isinstance(msg[1], dict):
            msg[1]["t_read"] = t_read
        return msg

    def _handle(self, msg) -> tuple:
        kind = msg[0] if isinstance(msg, tuple) and msg else None
        if kind == "ready":
            info = dict(msg[1], ready_after_s=round(self.elapsed(), 6))   # read by the EA, seconds after start()
            if self._t_start is not None and isinstance(info.get("t_sent"), (int, float)):
                info["ready_after_s_worker"] = round(float(info["t_sent"]) - self._t_start, 6)   # sent by the worker
            self.ready_info = info
            self.state = "idle"
            return ("ready", self.ready_info)
        if kind == "result":
            if self.outstanding is not None and int(msg[1]) == self.outstanding:
                self.outstanding = None
                self.state = "idle"
            return msg
        if kind == "failed":
            self._fail(str(msg[1]))
            return ("failed", self.error, msg[2] if len(msg) > 2 else "")
        self._fail(f"unexpected message from the decoder worker: {str(msg)[:200]}")
        return ("failed", self.error, "")

    def _fail(self, why: str) -> None:
        """The decoder fails for the rest of the run. Runs on the EA thread during the run, so it never waits: SIGTERM
        to the worker and our end of the pipe closed (a worker blocked in send() then gets EPIPE); the process is
        reaped later without blocking (`poll()`), and joined only by `stop()`, after the clock."""
        self.error = why
        p = self._proc
        if p is not None:
            try:
                if p.is_alive():                          # is_alive() reaps a worker that already exited
                    p.terminate()
                self.exitcode = p.exitcode
            except (OSError, ValueError, AssertionError):
                pass
        self._close()
        self.state = "failed"

    def _reap(self) -> None:
        """Without blocking: record the exit code of a failed worker once it has exited."""
        p = self._proc
        if p is not None and self.exitcode is None:
            try:
                if not p.is_alive():
                    self.exitcode = p.exitcode
            except (OSError, ValueError, AssertionError):
                pass

    def _close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None

    def _terminate(self) -> None:
        """`stop()` only (after the clock): terminate, join, kill, join — so the PID is gone."""
        p = self._proc
        if p is not None:
            if p.is_alive():
                p.terminate()                             # terminate before join: a child blocked in send never exits
                p.join(timeout=2.0)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2.0)
            else:
                p.join(timeout=2.0)                       # reap it, so its PID is gone
            self.exitcode = p.exitcode
        self._close()

    def stop(self) -> bool:
        """Terminate and join the worker (idempotent). Returns True if a request was outstanding (lost)."""
        lost = self.outstanding is not None and self.state == "busy"
        self.outstanding = None
        self._terminate()
        if self.state != "failed":
            self.state = "stopped" if self._proc is not None else self.state
        return bool(lost)

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def __enter__(self) -> "LPDecoder":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
