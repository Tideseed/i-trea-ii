"""NVIDIA cuOpt process (control arm and race partner): `python -m itrea.solvers.cuopt_run --instance P --seed S
--budget T --run-dir D [--mailbox M] [--run-id R] [--role-name N] [--no-barrier] [--sol-out F] [--cpu-threads N]
[--gpu-mem-gb G] [--mip-gap-zero] [--send] [--no-warmup] [--arm A]`.

cuOpt 26.08 (installed in this venv; `--cuopt-python` in the runner can point at another interpreter). cuOpt NEVER
receives solutions: registering a `SetSolutionCallback` disables cuOpt's presolve (optopt X1, and cuOpt's own
`set_mip_callback` documentation), which would turn the "as-shipped cuOpt" control (A2/A6) into something else. It
only reports incumbents, through a `GetSolutionCallback`: every one is recorded (vector + objective + bound), and with
--send also posted to the EA on cuOpt's OWN channels `to_ea_cuopt` (the vector, ORIGIN_SOLVER; cuOpt never sees an EA
vector, so there are no echoes) and `info_cuopt` (incumbent, bound, status) — the three-way arm A9 (D-015), where SCIP
writes `to_ea` / `info` beside it and the channels keep only their latest message. The runner passes --send only in
A9; the controls A2 and A6 never do.

Before the barrier (start-up, recorded, never on the clock):
- parse the instance (`ParseMps`, timed as `read_s`);
- set an RMM `StatisticsResourceAdaptor(LimitingResourceAdaptor(CudaMemoryResource(), --gpu-mem-gb))` as the current
  device resource: GPU memory is system RAM on GB10 and a systemd MemoryMax does not charge it (D-003), so this limit
  is cuOpt's only cap; the statistics layer gives the measured peak (`rmm_peak_bytes`) the run-set memory gate budgets;
- warm the CUDA context and cuOpt's kernels with a tiny synthetic knapsack solve (1 s limit) — the clock rule allows
  context creation and warm-up on a synthetic problem only. If the warm-up fails (for example the RMM limit is too
  small) the process exits 1 BEFORE reporting ready, so the runner sees a dead process, not a stuck barrier.
After the barrier: `Solve` with time_limit = remaining budget, num_cpu_threads = --cpu-threads, random_seed = seed,
and with --mip-gap-zero mip_absolute_gap = mip_relative_gap = 0 (GAMS's MIPfeas settings, arm A6m, D-020; otherwise
cuOpt's default gaps, relative 1e-4). Those parameters are recorded in the summary as `settings` (`solver_params`).
--cpu-threads has no upper limit: GAMS ran num_cpu_threads 24 on the 20-core GB10, and A6m does the same.

Measured 2026-10-06 (this venv, cuOpt 26.08.00, a 400-variable set-cover toy): cuOpt's MIP solve requests two fixed
2.4 GB device buffers (2,399,998,904 bytes each) whatever the instance size, so it needs more than about 4.8 GB under
any limit; with 0.5 GB or 4 GB it ends with error status 6 "Memory allocation failed" and termination status 0. It
also refuses `num_cpu_threads < 2` (error status 6). An error status is a failed run (exit 1, error and RMM statistics
in the summary). Incumbents its CPU feasibility jump reported through the callback before the failure are real
vectors and stay recorded (a crashed run is kept, truncated at the crash: D-002 §6); cuOpt's placeholder objective 0.0
on such statuses is never recorded (the optopt dano3_3 trap, 2026-09-23).

Stop file / SIGTERM: cuOpt has no interrupt API; both are checked only when an incumbent arrives (control plane), and
otherwise the time limit ends the solve. The runner's grace period and exit-code check cover the rest.

Adapted in part from berkorbay/optopt@4dd48395, MIT (`src/optopt/solvers/cuopt.py`: status table, the placeholder
objective rule).
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from itrea.solvers.common import (ORIGIN_SOLVER, BadArguments, base_parser, finite_or_none, improves, mem_available_gb,
                                  run_main, validate_common, versions)

MODULE = "itrea.solvers.cuopt_run"

# cuOpt 26.08 MILPTerminationStatus (optopt, verified 2026-09-23)
STATUS = {0: "NoTermination", 1: "Optimal", 2: "Infeasible", 3: "Unbounded", 5: "TimeLimit(no solution)",
          8: "FeasibleFound", 11: "UnboundedOrInfeasible"}
_EXIT = {"Optimal": "optimal", "FeasibleFound": "budget", "TimeLimit(no solution)": "budget",
         "Infeasible": "infeasible", "Unbounded": "unbounded", "UnboundedOrInfeasible": "infeasible_or_unbounded"}


class CuoptError(RuntimeError):
    """cuOpt returned an error status. `summary` carries what the run record needs (status, RMM statistics)."""

    def __init__(self, msg: str, summary: dict):
        super().__init__(msg)
        self.summary = summary


def build_parser():
    ap = base_parser(f"python -m {MODULE}", "NVIDIA cuOpt process (incumbents only; never receives)", "cuopt",
                     exchange=False)
    ap.add_argument("--cpu-threads", type=int, default=3, help="cuOpt num_cpu_threads (>= 2; partner footprint)")
    ap.add_argument("--gpu-mem-gb", type=float, default=4.0, help="RMM LimitingResourceAdaptor limit (GB)")
    ap.add_argument("--mip-gap-zero", action="store_true",
                    help="mip_absolute_gap = mip_relative_gap = 0, as GAMS ran cuOpt for MIPfeas (arm A6m, D-020); "
                         "default: cuOpt's own gaps")
    ap.add_argument("--send", action="store_true",
                    help="post every new incumbent (and bound) to the EA on to_ea_cuopt / info_cuopt (arm A9)")
    ap.add_argument("--no-warmup", action="store_true", help="skip the synthetic warm-up solve (tests only)")
    return ap


def validate(args) -> None:
    validate_common(args, exchange=False)
    if args.cpu_threads < 2:
        raise BadArguments("--cpu-threads must be >= 2 (cuOpt 26.08 refuses fewer: error status 6)")
    if not (args.gpu_mem_gb > 0):
        raise BadArguments("--gpu-mem-gb must be positive")
    if args.send and args.mailbox is None:
        raise BadArguments("--send needs --mailbox")
    if args.seed < 0:
        raise BadArguments("--seed must be >= 0")


def _rmm_setup(limit_gb: float):
    import rmm.mr as mr
    lim = mr.LimitingResourceAdaptor(mr.CudaMemoryResource(), int(limit_gb * 1024**3))
    stats = mr.StatisticsResourceAdaptor(lim)
    mr.set_current_device_resource(stats)
    return lim, stats


def _rmm_stats(lim, stats) -> dict:
    out = {"rmm_limit_bytes": int(lim.get_allocation_limit())}
    try:
        c = stats.allocation_counts
        out.update(rmm_peak_bytes=int(c.peak_bytes), rmm_total_bytes=int(c.total_bytes),
                   rmm_current_bytes=int(c.current_bytes), rmm_peak_count=int(c.peak_count))
    except Exception as e:  # noqa: BLE001 — statistics are a side product
        out["rmm_stats_error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def solver_params(time_limit: float, threads: int, seed: int, gap_zero: bool = False) -> dict:
    """cuOpt's parameters for one solve (the summary records them as `settings`); every other parameter keeps cuOpt's
    default. gap_zero: GAMS's mip_absolute_gap = mip_relative_gap = 0 (D-020)."""
    d = dict(time_limit=float(time_limit), log_to_console=False, num_cpu_threads=int(threads), random_seed=int(seed))
    if gap_zero:
        d.update(mip_absolute_gap=0.0, mip_relative_gap=0.0)
    return d


def _settings(params: dict):
    from cuopt.linear_programming import SolverSettings
    s = SolverSettings()
    for k, v in params.items():
        s.set_parameter(k, v)
    return s


def _warmup(threads: int, seed: int) -> dict:
    """A tiny synthetic knapsack solve: CUDA context, cuOpt's kernels and handles before the clock."""
    from cuopt.linear_programming import ParseMps, Solve

    from itrea.benchmarks.toy import knapsack, write_mps
    with tempfile.TemporaryDirectory(prefix="itrea-cuopt-warm-") as td:
        p = write_mps(knapsack(12, seed=0), Path(td) / "warm.mps")
        dm = ParseMps(str(p))
        sol = Solve(dm, _settings(solver_params(1.0, threads, seed)))
    err = int(sol.get_error_status())
    st = int(sol.get_termination_status())
    return dict(error_status=err, error_message=str(sol.get_error_message()) if err else "",
                termination=STATUS.get(st, str(st)))


def body(proc):
    args = proc.args
    proc.timing["mem_available_gb_before_cuda"] = mem_available_gb()
    tc = time.monotonic()
    lim, stats = _rmm_setup(args.gpu_mem_gb)
    import cuopt
    from cuopt.linear_programming import ParseMps, Solve
    from cuopt.linear_programming.internals import GetSolutionCallback
    proc.timing["import_rmm_s"] = time.monotonic() - tc
    from itrea.paths import resolve
    tr0 = time.monotonic()
    dm = ParseMps(str(resolve(args.instance)))
    proc.timing["read_s"] = time.monotonic() - tr0
    names = [str(nm) for nm in dm.get_variable_names()]
    sense = -1 if dm.get_sense() else 1
    proc.set_names(names, sense)
    n = len(names)
    if args.send:
        proc.map_order()
    proc.trace.set(versions=versions(cuopt=cuopt.__version__), n=n, sense=sense, cpu_threads=args.cpu_threads,
                   gpu_mem_gb=args.gpu_mem_gb, mip_gap_zero=bool(args.mip_gap_zero))
    if not args.no_warmup:
        tw = time.monotonic()
        w = _warmup(args.cpu_threads, args.seed)
        proc.timing["warmup_s"] = time.monotonic() - tw
        proc.trace.set(warmup=w)
        if w["error_status"]:
            raise CuoptError(f"cuOpt warm-up failed before the barrier: error {w['error_status']} "
                             f"{w['error_message']}", dict(error_status=w["error_status"], status="warmup_failed",
                                                          **_rmm_stats(lim, stats)))
    proc.timing["mem_available_gb_after_cuda"] = mem_available_gb()
    proc.trace.set(rmm_after_warmup=_rmm_stats(lim, stats))

    class Recorder(GetSolutionCallback):
        """Every incumbent cuOpt reports: recorded, and posted with --send. Runs in cuOpt's thread (GIL held)."""

        def __init__(self):
            super().__init__()
            self.calls = 0
            self.errors = 0
            self.last_bound = None

        def get_solution(self, solution, solution_cost, solution_bound, user_data):
            try:
                self._on(solution, solution_cost, solution_bound)
            except Exception as e:  # noqa: BLE001 — never raise into cuOpt
                self.errors += 1
                proc.trace.event("note", msg=f"callback error: {type(e).__name__}: {e}"[:500])

        def _on(self, solution, solution_cost, solution_bound):
            self.calls += 1
            obj = finite_or_none(np.asarray(solution_cost).ravel()[0])
            bound = finite_or_none(np.asarray(solution_bound).ravel()[0])
            if bound is not None and bound != self.last_bound:
                self.last_bound = bound
                proc.trace.event("bound", bound=bound)
            if obj is None:
                return
            x = np.array(np.asarray(solution).ravel()[:n], dtype=np.float64, copy=True)
            if x.size != n or not improves(obj, proc.best_obj, sense):
                return
            proc.record_incumbent(obj, x, "self", bound=bound)
            if args.send:
                tp = time.perf_counter()
                xc = proc.namemap.to_canonical(x)
                seq = proc.mailbox.to_ea_cuopt.post(xc[None, :], np.array([obj]),
                                                    origin=np.array([ORIGIN_SOLVER], dtype=np.int64))
                proc.mailbox.info_cuopt.post(incumbent=obj, dual_bound=bound, status="solving", solver="cuopt")
                proc.stats.add("posts_out")
                proc.stats.add("rows_out")
                proc.stats.add("info_posts")
                proc.stats.bytes_out += int(xc.nbytes + 8)
                proc.stats.seconds += time.perf_counter() - tp
                proc.trace.event("send", seq=seq, k=1, best_obj=obj, echo=0)
            if proc.should_stop():
                proc.trace.event("note", msg="stop requested; cuOpt has no interrupt API, running to its time limit")

    t0 = proc.barrier()
    if proc.should_stop():
        return "stop", dict(status="not started", **_rmm_stats(lim, stats))
    limit = max(0.001, proc.remaining())
    params = solver_params(limit, args.cpu_threads, args.seed, args.mip_gap_zero)
    proc.trace.set(settings=params)
    s = _settings(params)
    cb = Recorder()
    s.set_mip_callback(cb, None)                     # GetSolutionCallback only — never a SetSolutionCallback
    proc.timing.update(time_limit_s=limit, optimize_at_s=time.monotonic() - t0)
    sol = Solve(dm, s)
    t_end = time.monotonic() - t0
    err = int(sol.get_error_status())
    st = int(sol.get_termination_status())
    status = STATUS.get(st, str(st))
    rs = _rmm_stats(lim, stats)
    if err:
        raise CuoptError(f"cuOpt error status {err}: {sol.get_error_message()}",
                         dict(error_status=err, status=status, wall_s=round(t_end, 6), **rs))
    ms = sol.get_milp_stats() or {}
    if status in ("Optimal", "FeasibleFound"):       # only these statuses carry a solution
        fo = finite_or_none(sol.get_primal_objective())
        if fo is not None and improves(fo, proc.best_obj, sense):
            proc.record_incumbent(fo, np.asarray(sol.get_primal_solution(), dtype=np.float64), "self", t=t_end,
                                  late=True)
    reason = _EXIT.get(status, status)
    if proc.sigterm:
        reason = "sigterm"
    if args.send:                                    # final status and bound, as scip_run / highs_run do
        proc.mailbox.info_cuopt.post(incumbent=proc.best_obj, dual_bound=finite_or_none(ms.get("solution_bound")),
                                     status=status, solver="cuopt")
        proc.stats.add("info_posts")
    return reason, dict(status=status, termination_status=st, wall_s=round(t_end, 6),
                        overrun_s=round(max(0.0, t_end - args.budget), 6),
                        final_dual=finite_or_none(ms.get("solution_bound")), nodes=ms.get("num_nodes"),
                        lp_iterations=ms.get("num_simplex_iterations"), presolve_s=ms.get("presolve_time"),
                        max_constraint_violation=ms.get("max_constraint_violation"),
                        callbacks=dict(calls=cb.calls, errors=cb.errors), **rs)


def main(argv=None) -> int:
    return run_main(argv, build_parser(), validate, body, "cuopt", MODULE)


if __name__ == "__main__":
    raise SystemExit(main())
