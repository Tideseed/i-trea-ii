"""cuOpt 26.08 MIP-start + stopping check (< 2 min on the GPU). NOT RUN by the research subagent (GPU forbidden there).

Run only inside a box-window lease held by i-trea-ii, below 80 C, with nothing else on the GPU lane; from the
repository root:
    OMP_NUM_THREADS=1 taskset -c 1-3 .venv/bin/python tools/cuopt_verify_mipstart.py
GPU: RMM limit 12 GB (cuOpt asks for ~5.6-6 GB peak on glass4/warm-up, traces/raw/dev/glass4__A6__*).
Wall: ~8 solves x 3 s + ~5 s start-up, all in ONE process (one CUDA context).

Instance glass4 (322 vars, presolve 0.4 s). From traces/raw/dev/glass4__A6__s0__120s__6e35f122f659: cuOpt alone has
best 2.1000173e9 at t <= 3 s and 1.6000131e9 at 120 s. So a 3-s solve that returns <= 1.6000131e9 used the start.

Claims checked (expected outcome in brackets):
 V0 control, no start, 3 s                         [obj ~2.1e9 or worse; log shows Papilo presolve]
 V1 full start = cuOpt's 120-s final vector, 3 s    [obj <= 1.6000131e9; a callback with obj == start obj early;
                                                     log shows Papilo presolve, NOT "Presolve is disabled"]
 V2 restart in-process (same DataModel) again, 3 s [same as V1; second Solve's time-to-first-callback = restart overhead]
 V3 partial start (NaN on half the vars), 3 s       ["Error cannot add the provided initial solution" or no effect;
                                                     obj like V0]
 V4 start with one var 1.0 above its upper bound   [error_status != 0, "Initial solution violates variable bounds."]
 V5 SIGTERM to self at 1 s of a 3-s solve          [Solve returns at ~3 s; the Python handler runs only after]
 V6 heuristics-only + start, 3 s                    [obj <= 1.6000131e9]
 V7 deterministic mode + start, 3 s                [source: start never added (diversity_manager.cu:461-481)
                                                     -> obj > 1.6000131e9 and no echo]
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]   # the repository root (this file is tools/cuopt_verify_mipstart.py)
INST = ROOT / "data/miplib/inst/glass4.mps.gz"
SRC = ROOT / "traces/raw/dev/glass4__A6__s0__120s__6e35f122f659/cuopt.incumbents.npz"
OUT = Path("/tmp/itrea-cuopt/verify_out")
OUT.mkdir(parents=True, exist_ok=True)

import rmm.mr as mr  # noqa: E402

lim = mr.LimitingResourceAdaptor(mr.CudaMemoryResource(), int(12 * 1024**3))
stats = mr.StatisticsResourceAdaptor(lim)
mr.set_current_device_resource(stats)

from cuopt.linear_programming import ParseMps, Solve, SolverSettings  # noqa: E402
from cuopt.linear_programming.internals import GetSolutionCallback  # noqa: E402

t_parse = time.monotonic()
dm = ParseMps(str(INST))
t_parse = time.monotonic() - t_parse
names = [str(n) for n in dm.get_variable_names()]
lb = np.asarray(dm.get_variable_lower_bounds(), dtype=np.float64)
ub = np.asarray(dm.get_variable_upper_bounds(), dtype=np.float64)

z = np.load(SRC)
k = int(np.argmin(z["obj"]))
x_src = np.zeros(len(z["names"]))
a, b = z["ptr"][k], z["ptr"][k + 1]
x_src[z["idx"][a:b]] = z["val"][a:b]
pos = {str(nm): j for j, nm in enumerate(z["names"])}
start = np.array([x_src[pos[nm]] for nm in names], dtype=np.float64)
start_obj = float(z["obj"][k])


class Rec(GetSolutionCallback):
    def __init__(self, t_call):
        super().__init__()
        self.t_call = t_call
        self.hits = []

    def get_solution(self, solution, solution_cost, solution_bound, user_data):
        self.hits.append((round(time.monotonic() - self.t_call, 3), float(np.asarray(solution_cost).ravel()[0])))


def settings(tag, tl=3.0, heur_only=False, determ=False):
    s = SolverSettings()
    s.set_parameter("time_limit", tl)
    s.set_parameter("log_to_console", False)
    s.set_parameter("log_file", str(OUT / f"{tag}.log"))
    s.set_parameter("num_cpu_threads", 3)
    s.set_parameter("random_seed", 0)
    if heur_only:
        s.set_parameter("mip_heuristics_only", True)
    if determ:
        s.set_parameter("mip_determinism_mode", 1)
    return s


def run(tag, x0=None, heur_only=False, sigterm_at=None, determ=False):
    if x0 is None:
        dm.set_initial_primal_solution(np.array([], dtype=np.float64))   # clear (data_model_wrapper.pyx:449)
    else:
        dm.set_initial_primal_solution(np.asarray(x0, dtype=np.float64))
    s = settings(tag, heur_only=heur_only, determ=determ)
    t_call = time.monotonic()
    cb = Rec(t_call)
    s.set_mip_callback(cb, None)
    fired = {}
    if sigterm_at is not None:
        signal.signal(signal.SIGTERM, lambda *_: fired.setdefault("handler_at", round(time.monotonic() - t_call, 3)))
        threading.Timer(sigterm_at, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    exc = None
    try:
        sol = Solve(dm, s)      # @catch_cuopt_exception (solver.py:13) may raise instead of returning an error status
    except Exception as e:  # noqa: BLE001 — record and continue with the next variant
        sol, exc = None, repr(e)[:400]
    wall = time.monotonic() - t_call
    err = int(sol.get_error_status()) if sol is not None else -1
    rec = dict(tag=tag, wall_s=round(wall, 3), error_status=err, exception=exc,
               error_message=str(sol.get_error_message()) if (sol is not None and err) else "",
               termination=int(sol.get_termination_status()) if sol is not None else None,
               obj=None if (sol is None or err) else float(sol.get_primal_objective()),
               first_cb=cb.hits[:3], n_cb=len(cb.hits),
               start_obj_echoed=any(abs(o - start_obj) <= 1e-6 * abs(start_obj) for _, o in cb.hits),
               rmm_peak=int(stats.allocation_counts.peak_bytes), **fired)
    log = OUT / f"{tag}.log"
    txt = log.read_text(errors="replace") if log.exists() else ""
    rec["log_papilo"] = "Papilo presolve time" in txt
    rec["log_presolve_disabled"] = "Presolve is disabled" in txt
    rec["log_cannot_add"] = "cannot add the provided initial solution" in txt
    rec["log_mip_start"] = [ln for ln in txt.splitlines() if "MIP start" in ln or "initial solution" in ln][:5]
    print(json.dumps(rec), flush=True)
    return rec


print(json.dumps(dict(parse_s=round(t_parse, 3), n=len(names), start_obj=start_obj,
                      start_in_bounds=bool(np.all((start >= lb - 1e-9) & (start <= ub + 1e-9))))), flush=True)
run("warm", None)                       # absorbs lazy kernel loading; not a measurement
r0 = run("V0_control", None)
r1 = run("V1_start", start)
r2 = run("V2_restart_same_dm", start)
part = start.copy()
part[::2] = np.nan
r3 = run("V3_partial_nan", part)
bad = start.copy()
j = int(np.argmax(np.isfinite(ub)))
bad[j] = ub[j] + 1.0
r4 = run("V4_out_of_bounds", bad)
r5 = run("V5_sigterm_at_1s", None, sigterm_at=1.0)
r6 = run("V6_heur_only_start", start, heur_only=True)
r7 = run("V7_deterministic_start", start, determ=True)
print("VERDICT start_used:", (r1["obj"] or 9e99) <= start_obj * (1 + 1e-9) and (r0["obj"] or 9e99) > start_obj,
      "| presolve_kept:", r1["log_papilo"] and not r1["log_presolve_disabled"],
      "| oob_rejected:", r4["error_status"] != 0 or r4["exception"] is not None,
      "| sigterm_deferred:", r5["wall_s"] >= 2.5 and r5.get("handler_at", 0) >= 2.5,
      "| determ_ignores_start:", not r7["start_obj_echoed"] and (r7["obj"] or 9e99) > start_obj, flush=True)
