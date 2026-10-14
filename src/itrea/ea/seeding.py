"""Initial population, immigrants, LP-rounded individuals, and the LP relaxation solved in a worker process.

The clock rule (vault D-002, design critique 2026-10-05). Before the shared start t0 only cheap, instance-blind
generation is allowed: solving an LP is search work that the SCIP and cuOpt arms pay for on the clock. So:
    seed_population   uses NO LP and solves nothing — random points in the bounds and "zero-ish" points. Safe before t0.
    LPWorker          solves the LP relaxation with HiGHS in a separate PROCESS, started AFTER t0; the engine polls it
                      and turns the point into individuals with `lp_round` when it arrives (`seeding.lp_source =
                      "highs_async"`). With `lp_source = "solver"` the point comes from the solver's root LP through the
                      mailbox instead, and this module only rounds it.
    lp_relaxation     the same solve in-process. It is what the worker runs; calling it before t0 in a timed run breaks
                      the clock rule.

seed_population(bk, dp, P, cfg, rng_dev, rng_host, gene_dtype="auto", mem_budget_bytes=256 MiB) -> (X, origin)
    X (P, n) device array in the gene dtype (`population.gene_dtype_for(lb, ub, isint, gene_dtype)`), origin (P,) host
    uint16. The first round(cfg.random · P) rows are uniform in the bounds: integers uniform over the integer points,
    continuous uniform in the interval; an infinite side is replaced by the finite bound ∓ 10 (free variables: [-10, 10]).
    Origin ORIGIN_SEED_RANDOM. The rest are "zero-ish": every variable at the point of its bounds nearest 0 (bound-
    feasible, and often row-feasible on covering/packing models); the first of them exactly, the others with a few
    random genes reset (Poisson(4), at least 1, the mutation module's "reset", window 10 on infinite sides) so they are
    distinct. Origin ORIGIN_SEED_RANDOM, plus ORIGIN_M_RESET on the perturbed ones (there is no free origin bit for a
    "zero-ish" seed; SEED_RANDOM alone = uniform or the exact zero-ish point).
lp_round(bk, dp, lp_x, k, rng_dev, rng_host, gene_dtype="auto") -> (X, origin)
    k randomised roundings of an LP point: an integer variable takes floor(x) or ceil(x) with probability = the
    fractional part, then its bounds; continuous variables take the LP value clipped to their bounds. A non-finite LP
    entry is replaced by the zero-ish value. Origin ORIGIN_SEED_LP.
immigrants(bk, dp, k, rng_dev, rng_host, lp_x=None, frontier_rows=None, lp_share=0.5, gene_dtype="auto",
           lp_count=None) -> (X, origin)
    Up to k rows of `frontier_rows` first (rows of the frontier archive, D-005; integer genes rounded, cast to the gene
    dtype), then of the rest round(lp_share · rest) LP roundings when lp_x is given (exactly `lp_count` if set), and
    uniform random points for the remainder. Every row carries ORIGIN_IMMIGRANT, plus SEED_LP / SEED_RANDOM for the
    generated ones (frontier rows: IMMIGRANT alone).

All generation is batched on the device in chunks of rows under `mem_budget_bytes`.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import resource
import time
import traceback

import numpy as np

from itrea.backend import Backend
from itrea.config import MutationCfg, SeedingCfg
from itrea.ea.crossover import dp_cache
from itrea.ea.evaluate import DeviceProblem
from itrea.ea.mutation import mutate
from itrea.ea.population import ORIGIN_IMMIGRANT, ORIGIN_SEED_LP, ORIGIN_SEED_RANDOM, gene_dtype_for
from itrea.paths import resolve
from itrea.problem import Problem

SEED_WINDOW = 10.0          # an infinite bound is replaced by the finite one (or 0) -/+ this for random points
ZEROISH_GENES = 4.0         # mean number of genes reset in each perturbed zero-ish seed
DEFAULT_MEM_BUDGET = 256 * 1024**2
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


# ----------------------------------------------------------------------------------------------- shared pieces
def _gene_dtype(dp: DeviceProblem, gene_dtype: str) -> np.dtype:
    p = dp.prob
    return gene_dtype_for(np.asarray(p.lb), np.asarray(p.ub), np.asarray(p.isint, bool), gene_dtype)


def _ranges(bk: Backend, dp: DeviceProblem):
    """Device (lo, hi, int_lo, int_hi, zero, isint): sampling window per variable and the zero-ish point."""
    c = dp_cache(dp)
    if "s_ranges" not in c:
        p = dp.prob
        lb, ub = np.asarray(p.lb, np.float64), np.asarray(p.ub, np.float64)
        isint = np.asarray(p.isint, bool)
        lo = np.where(np.isfinite(lb), lb, np.where(np.isfinite(ub), ub - SEED_WINDOW, -SEED_WINDOW))
        hi = np.where(np.isfinite(ub), ub, np.where(np.isfinite(lb), lb + SEED_WINDOW, SEED_WINDOW))
        ilo, ihi = np.ceil(lb - 1e-9), np.floor(ub + 1e-9)          # integer bounds (may be infinite)
        zero = np.clip(0.0, np.where(isint, ilo, lb), np.where(isint, ihi, ub))
        xp = bk.xp
        c["s_ranges"] = tuple(xp.asarray(v) for v in (lo, hi, np.where(isint, ilo, lb), np.where(isint, ihi, ub),
                                                      zero, isint))
    return c["s_ranges"]


def _rows_per_chunk(n: int, budget: int) -> int:
    """Rows per chunk: about ten float64 temporaries per generated row are alive at once."""
    return max(1, int(budget // max(1, 88 * n)))


def _cast(bk: Backend, X, dt: np.dtype):
    """float64 rows -> gene dtype; integer columns are integral already, an int dtype is reached by rint first."""
    xp = bk.xp
    if dt.kind in "iu":
        return xp.rint(X).astype(dt)
    return X.astype(dt, copy=False)


def _uniform(bk: Backend, dp: DeviceProblem, k: int, rng_dev, dt: np.dtype, budget: int):
    xp = bk.xp
    lo, hi, blo, bhi, _, isint = _ranges(bk, dp)
    out = xp.empty((k, dp.n), dtype=dt)
    ilo, ihi = xp.ceil(lo), xp.floor(hi)
    width = ihi - ilo + 1.0
    step = _rows_per_chunk(dp.n, budget)
    for s in range(0, k, step):
        e = min(k, s + step)
        U = rng_dev.random((e - s, dp.n))
        Xi = xp.minimum(ilo + xp.floor(U * width), ihi)
        Xc = lo + U * (hi - lo)
        X = xp.where(isint, Xi, Xc)
        X = xp.clip(X, blo, bhi)                     # only matters for empty integer domains (infeasible bounds)
        out[s:e] = _cast(bk, X, dt)
    return out


def _zeroish(bk: Backend, dp: DeviceProblem, k: int, rng_dev, rng_host, dt: np.dtype, budget: int):
    """(X, origin): the zero-ish point, then k - 1 copies with a few genes reset."""
    xp = bk.xp
    zero = _ranges(bk, dp)[4]
    X = xp.empty((k, dp.n), dtype=dt)
    X[...] = _cast(bk, zero[None, :], dt)
    origin = np.full(k, ORIGIN_SEED_RANDOM, dtype=np.uint16)
    if k > 1:
        rest = X[1:].copy()
        cfg = MutationCfg(genes=ZEROISH_GENES, min_genes=1, weights={"reset": 1.0}, reset_window=SEED_WINDOW)
        origin[1:] |= mutate(bk, dp, rest, cfg, rng_dev, rng_host, mem_budget_bytes=budget)
        X[1:] = rest
    return X, origin


# ---------------------------------------------------------------------------------------------- population seeds
def seed_population(bk: Backend, dp: DeviceProblem, P: int, cfg: SeedingCfg, rng_dev, rng_host: np.random.Generator,
                    gene_dtype: str = "auto", mem_budget_bytes: int = DEFAULT_MEM_BUDGET):
    """Initial population without any LP or instance-specific solve; see the module docstring."""
    P = int(P)
    dt = _gene_dtype(dp, gene_dtype)
    n_rand = int(round(cfg.random * P))
    n_zero = P - n_rand
    parts, origins = [], []
    if n_rand:
        parts.append(_uniform(bk, dp, n_rand, rng_dev, dt, mem_budget_bytes))
        origins.append(np.full(n_rand, ORIGIN_SEED_RANDOM, dtype=np.uint16))
    if n_zero:
        Xz, oz = _zeroish(bk, dp, n_zero, rng_dev, rng_host, dt, mem_budget_bytes)
        parts.append(Xz)
        origins.append(oz)
    xp = bk.xp
    if not parts:
        return xp.empty((0, dp.n), dtype=dt), np.empty(0, dtype=np.uint16)
    X = parts[0] if len(parts) == 1 else xp.concatenate(parts)
    return X, np.concatenate(origins)


def lp_round(bk: Backend, dp: DeviceProblem, lp_x, k: int, rng_dev, rng_host: np.random.Generator,
             gene_dtype: str = "auto", mem_budget_bytes: int = DEFAULT_MEM_BUDGET):
    """k randomised roundings of the LP point lp_x (host or device, canonical order); see the module docstring."""
    xp = bk.xp
    k = int(k)
    dt = _gene_dtype(dp, gene_dtype)
    _, _, blo, bhi, zero, isint = _ranges(bk, dp)
    lp = xp.asarray(lp_x, dtype=xp.float64).reshape(-1)
    if lp.shape[0] != dp.n:
        raise ValueError(f"lp_round: lp_x has {lp.shape[0]} entries, the problem {dp.n} variables")
    lp = xp.where(xp.isfinite(lp), lp, zero)
    f = xp.floor(lp)
    frac = lp - f
    out = xp.empty((k, dp.n), dtype=dt)
    step = _rows_per_chunk(dp.n, mem_budget_bytes)
    for s in range(0, k, step):
        e = min(k, s + step)
        U = rng_dev.random((e - s, dp.n))
        X = xp.where(isint, f + (U < frac), lp)
        out[s:e] = _cast(bk, xp.clip(X, blo, bhi), dt)
    return out, np.full(k, ORIGIN_SEED_LP, dtype=np.uint16)


def immigrants(bk: Backend, dp: DeviceProblem, k: int, rng_dev, rng_host: np.random.Generator, lp_x=None,
               frontier_rows=None, lp_share: float = 0.5, gene_dtype: str = "auto",
               mem_budget_bytes: int = DEFAULT_MEM_BUDGET, lp_count: int | None = None):
    """k immigrants: frontier rows first, then LP roundings (lp_share of the rest, if lp_x), then uniform random.
    `lp_count` (if given, with lp_x) replaces round(lp_share · rest) by that exact number, clipped to the rest — the
    self-adaptive repair strategy gives LP roundings only to immigrants whose strategy has LP guidance."""
    xp = bk.xp
    k = int(k)
    dt = _gene_dtype(dp, gene_dtype)
    parts, origins = [], []
    if frontier_rows is not None and k > 0 and frontier_rows.shape[0] > 0:
        R = xp.asarray(frontier_rows[:k], dtype=xp.float64)
        _, _, blo, bhi, _, isint = _ranges(bk, dp)
        R = xp.where(isint, xp.clip(xp.rint(R), blo, bhi), R)
        parts.append(_cast(bk, R, dt))
        origins.append(np.full(R.shape[0], ORIGIN_IMMIGRANT, dtype=np.uint16))
    rest = k - sum(p.shape[0] for p in parts)
    if lp_x is None:
        k_lp = 0
    elif lp_count is not None:
        k_lp = max(0, min(int(lp_count), rest))
    else:
        k_lp = int(round(lp_share * rest))
    if k_lp:
        X, o = lp_round(bk, dp, lp_x, k_lp, rng_dev, rng_host, gene_dtype, mem_budget_bytes)
        parts.append(X)
        origins.append(o | ORIGIN_IMMIGRANT)
    k_rand = rest - k_lp
    if k_rand:
        parts.append(_uniform(bk, dp, k_rand, rng_dev, dt, mem_budget_bytes))
        origins.append(np.full(k_rand, ORIGIN_IMMIGRANT | ORIGIN_SEED_RANDOM, dtype=np.uint16))
    if not parts:
        return xp.empty((0, dp.n), dtype=dt), np.empty(0, dtype=np.uint16)
    X = parts[0] if len(parts) == 1 else xp.concatenate(parts)
    return X, np.concatenate(origins)


# ------------------------------------------------------------------------------------------------ LP relaxation
def _highs_from_problem(h, prob: Problem) -> None:
    """Pass a Problem built from arrays to HiGHS (adapted from itrea.benchmarks.toy.write_mps)."""
    import highspy

    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = prob.n, prob.m
    inf = highspy.kHighsInf

    def hv(v):
        v = np.asarray(v, dtype=np.float64).copy()
        v[v == np.inf] = inf
        v[v == -np.inf] = -inf
        return v

    lp.col_cost_ = np.asarray(prob.c, dtype=np.float64)
    lp.offset_ = float(prob.offset)
    lp.col_lower_, lp.col_upper_ = hv(prob.lb), hv(prob.ub)
    lp.row_lower_, lp.row_upper_ = hv(prob.rl), hv(prob.ru)
    csc = prob.A.tocsc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = csc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = csc.indices.astype(np.int32)
    lp.a_matrix_.value_ = csc.data.astype(np.float64)
    lp.sense_ = highspy.ObjSense.kMaximize if prob.sense == -1 else highspy.ObjSense.kMinimize
    lp.integrality_ = [highspy.HighsVarType.kInteger if t else highspy.HighsVarType.kContinuous for t in prob.isint]
    lp.col_names_ = list(prob.names)
    h.passModel(lp)


def lp_relaxation(source: str | os.PathLike | Problem, time_limit: float = 60.0, names: list[str] | None = None,
                  threads: int = 1) -> tuple[np.ndarray | None, dict]:
    """Solve the LP relaxation with HiGHS (`solve_relaxation`, 1 thread, time limit) IN THIS PROCESS.

    source: an MPS/LP path (repository-relative or absolute) or a Problem (passed as arrays). Returns (x, info): x in
    canonical order — HiGHS's column order, or `names`' order if given and different — when HiGHS holds a primal-
    FEASIBLE point (an optimum, or a feasible point at the time limit), else None. info: state ("done" | "no_solution"),
    status, primal_solution_status, objective (user sense), simplex_iterations, read_seconds, solve_seconds.
    In a timed run this is on the clock: call it only after t0, normally through LPWorker."""
    import highspy

    t_read = time.monotonic()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("solve_relaxation", True)
    h.setOptionValue("time_limit", float(time_limit))
    if isinstance(source, Problem):
        _highs_from_problem(h, source)
    else:
        if h.readModel(str(resolve(source))) == highspy.HighsStatus.kError:
            raise IOError(f"HiGHS could not read {source}")
    t_solve = time.monotonic()
    h.run()
    t_end = time.monotonic()
    st = h.getModelStatus()
    hi = h.getInfo()
    feasible = int(hi.primal_solution_status) == int(highspy.SolutionStatus.kSolutionStatusFeasible)
    info = dict(state="done" if feasible else "no_solution", status=h.modelStatusToString(st),
                primal_solution_status=int(hi.primal_solution_status),
                objective=float(hi.objective_function_value) if feasible else None,
                simplex_iterations=int(hi.simplex_iteration_count),
                read_seconds=t_solve - t_read, solve_seconds=t_end - t_solve)
    if not feasible:
        return None, info
    x = np.asarray(h.getSolution().col_value, dtype=np.float64)
    lp = h.getLp()
    info["n"] = int(x.size)
    if names is not None:
        col = list(lp.col_names_)
        if len(col) == x.size and col != list(names):
            pos = {nm: j for j, nm in enumerate(col)}
            missing = [nm for nm in names if nm not in pos]
            if missing or len(names) != x.size:
                raise ValueError(f"lp_relaxation: variable names differ from the expected order ({missing[:3]} ...)")
            x = x[np.fromiter((pos[nm] for nm in names), dtype=np.int64, count=len(names))]
            info["reordered"] = True
    return x, info


def _cpu_self_s() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return float(ru.ru_utime + ru.ru_stime)


def _lp_worker_main(conn, path: str, time_limit: float, cpus: list[int] | None, names: list[str] | None) -> None:
    """Entry point of the spawned worker: pin, solve, send (x, info) once, exit. The worker stamps its own clock
    (`time.monotonic()`, system-wide) and CPU seconds (RUSAGE_SELF: the spawned interpreter's start-up included) when
    it enters here (`t_entry`, `cpu_entry_s`) and just before it sends (`t_done`, `cpu_done_s`), so the EA records
    when the LP was really finished and how much of a core it had, apart from when the EA polled it (R1)."""
    t_entry, cpu_entry = time.monotonic(), _cpu_self_s()
    stamps = dict(t_entry=t_entry, cpu_entry_s=cpu_entry, nice=os.getpriority(os.PRIO_PROCESS, 0))
    try:
        if cpus:
            os.sched_setaffinity(0, cpus)
        x, info = lp_relaxation(path, time_limit=time_limit, names=names)
        info["pid"] = os.getpid()
        info["cpus"] = sorted(os.sched_getaffinity(0))
        info["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS")
        info.update(stamps, cpu_done_s=_cpu_self_s(), t_done=time.monotonic())
        conn.send((x, info))
    except BaseException as e:  # noqa: BLE001 — anything the solve raises goes back to the EA as a failure
        try:
            conn.send((None, dict(state="failed", error=repr(e), traceback=traceback.format_exc(), **stamps,
                                  cpu_done_s=_cpu_self_s(), t_done=time.monotonic())))
        except Exception:
            pass
    finally:
        conn.close()


class LPWorker:
    """The LP relaxation of an instance file, solved by HiGHS in a separate process (multiprocessing "spawn").

        w = LPWorker(path, time_limit=60.0, cpus=[3], names=prob.names)
        w.start()                       # AFTER t0 — the solve is on the clock (D-002)
        ...
        x, info = w.poll()              # non-blocking: (None, {"state": "running", ...}) until it is done
        w.stop()                        # always; also a context manager

    The worker runs HiGHS with 1 thread, `OMP/OPENBLAS/MKL_NUM_THREADS=1`, and — if `cpus` is given — pinned to those
    cores; otherwise it inherits the EA's affinity (the EA's own core, `seeding.lp_source = "highs_async"`). It is a
    daemon process: it dies with the EA. Its memory is charged to the EA's systemd scope (same cgroup).
    poll() states: "not_started", "running", "done" (x is the canonical-order LP point), "no_solution" (time limit or
    infeasible LP without a feasible point), "failed" (exception or the process died; `error`, `exitcode`),
    "stopped" (stop() before a result). Once a final state is reached poll() keeps returning it. A result the worker
    sent carries its own stamps (`t_entry`, `t_done`: time.monotonic(), system-wide; `cpu_entry_s`, `cpu_done_s`:
    RUSAGE_SELF; `nice`), so the EA records when the LP was finished apart from `wall_seconds` (when it was read)."""

    def __init__(self, instance: str | os.PathLike, time_limit: float = 60.0, cpus: list[int] | None = None,
                 names: list[str] | None = None):
        self.path = str(resolve(instance))
        self.time_limit = float(time_limit)
        self.cpus = list(cpus) if cpus else None
        self.names = list(names) if names is not None else None
        self._proc = None
        self._conn = None
        self._t_start: float | None = None
        self._result: tuple[np.ndarray | None, dict] | None = None

    # ---------------------------------------------------------------------------------------------- lifecycle
    def start(self) -> "LPWorker":
        if self._proc is not None:
            raise RuntimeError("LPWorker already started")
        ctx = mp.get_context("spawn")
        recv, send = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_lp_worker_main, args=(send, self.path, self.time_limit, self.cpus, self.names),
                           name="itrea-lp-worker", daemon=True)
        saved = {v: os.environ.get(v) for v in _THREAD_VARS}
        try:
            for v in _THREAD_VARS:
                os.environ[v] = "1"                     # spawn copies the parent's environment at launch
            proc.start()
        finally:
            for v, old in saved.items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old
        send.close()                                    # the parent keeps only the receiving end (EOF on child death)
        self._proc, self._conn, self._t_start = proc, recv, time.monotonic()
        return self

    def _elapsed(self) -> float:
        return time.monotonic() - self._t_start if self._t_start is not None else 0.0

    def _receive(self) -> bool:
        """Take the result if one is waiting. Receive BEFORE joining: the child blocks in send() until we read."""
        try:
            if not self._conn.poll():
                return False
            x, info = self._conn.recv()
        except (EOFError, OSError):
            return False
        info = dict(info)
        info["wall_seconds"] = self._elapsed()
        self._result = (x, info)
        self._proc.join(timeout=5.0)
        self._close_conn()
        return True

    def poll(self) -> tuple[np.ndarray | None, dict]:
        """(x or None, info) without blocking; see the class docstring for the states."""
        if self._result is not None:
            return self._result
        if self._proc is None:
            return None, dict(state="not_started")
        if self._receive():
            return self._result
        if not self._proc.is_alive():
            if self._receive():                         # it sent and exited between the two checks
                return self._result
            self._proc.join(timeout=1.0)
            self._result = (None, dict(state="failed", error="LP worker exited without a result",
                                       exitcode=self._proc.exitcode, wall_seconds=self._elapsed()))
            self._close_conn()
            return self._result
        return None, dict(state="running", wall_seconds=self._elapsed(), pid=self._proc.pid)

    def stop(self) -> None:
        """Terminate the worker if it is still running (idempotent)."""
        if self._proc is not None and self._result is None and self._conn is not None:
            self._receive()
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()                      # terminate before join: a child blocked in send() never exits
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=2.0)
        self._close_conn()
        if self._proc is not None and self._result is None:
            self._result = (None, dict(state="stopped", wall_seconds=self._elapsed()))

    def _close_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._result is None and self._proc.is_alive()

    def __enter__(self) -> "LPWorker":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
