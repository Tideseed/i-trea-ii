"""Probe cuOpt's SetSolutionCallback with presolve (NVIDIA/cuopt#2073, fixed by PR #2098 in 26.10): does a solution
injected in ORIGINAL variable order survive presolve, is an infeasible one rejected, and what does merely registering
a set-callback cost (PR #2098 turns Papilo's dual reductions off while one is registered)?

Runs under any interpreter with cuOpt + NumPy (our 26.08 venv, or ~/venvs/cuopt-2610n for the 26.10 nightly); no itrea
import, so it runs outside the repository's venv. One solve per call:

    python tools/probe_cuopt_setsol.py --instance data/miplib/inst/glass4.mps.gz --sol /tmp/setsol/glass4.json \
        --mode inject --time 30 --out /tmp/setsol/out/glass4__inject__2610.json

Modes: none (GET callback only), noop (GET + a SET callback that never offers a solution — the cost of registering),
inject (the SET callback offers `--sol`'s vector once, at its first call at or after --inject-after seconds), bad (it
offers an infeasible variant: one integer variable moved to a fractional value, then the vector with every value +1)
and raw (it offers `--sol`'s vector as given, for an edge case such as a point beating the reference).
`--sol` JSON: {names, x, obj} (as written by the export step of 2026-10-13, journal). Output JSON: versions, the cuOpt
log's presolve lines, every GET record (t, obj; vectors in an .npz beside the JSON, original variable order), every SET
call (t, offered, objective offered), and the final status / objective. Development probe (D-010), not a timed run.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
import time
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True)
    ap.add_argument("--sol", help="JSON with names, x, obj (original variable names)")
    ap.add_argument("--mode", choices=("none", "noop", "inject", "bad", "raw"), required=True)
    ap.add_argument("--time", type=float, default=30.0)
    ap.add_argument("--inject-after", type=float, default=3.0)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--rmm-gb", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import rmm
    import rmm.mr as mr
    lim = mr.LimitingResourceAdaptor(mr.CudaMemoryResource(), int(a.rmm_gb * 2**30))
    mr.set_current_device_resource(lim)
    import cuopt
    from cuopt.linear_programming import ParseMps, Solve, SolverSettings
    from cuopt.linear_programming.internals import GetSolutionCallback, SetSolutionCallback

    dm = ParseMps(str(a.instance))
    names = [str(n) for n in dm.get_variable_names()]
    n = len(names)
    x_inj = None
    obj_inj = math.inf
    if a.sol:
        s = json.loads(Path(a.sol).read_text())
        pos = {nm: i for i, nm in enumerate(s["names"])}
        missing = [nm for nm in names if nm not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} cuOpt variables not in --sol (first {missing[:3]})")
        x_inj = np.array([s["x"][pos[nm]] for nm in names], dtype=np.float64)
        obj_inj = float(s["obj"])
    vtypes = [str(t) for t in dm.get_variable_types()] if hasattr(dm, "get_variable_types") else []
    c = np.asarray(dm.get_objective_coefficients(), dtype=np.float64)
    off = float(dm.get_objective_offset()) if hasattr(dm, "get_objective_offset") else 0.0

    offers: list[np.ndarray] = []
    if a.mode in ("inject", "raw"):
        offers = [x_inj]
    elif a.mode == "bad":
        xb = x_inj.copy()
        ints = [i for i, t in enumerate(vtypes) if t.upper().startswith("I")]
        if ints:
            xb[ints[0]] = xb[ints[0]] + 0.5
        offers = [xb, x_inj + 1.0]

    t0 = time.monotonic()
    gets: list[tuple[float, float, np.ndarray]] = []
    sets: list[dict] = []

    class Get(GetSolutionCallback):
        def get_solution(self, solution, solution_cost, solution_bound, user_data):
            try:
                gets.append((time.monotonic() - t0, float(np.asarray(solution_cost).ravel()[0]),
                             np.array(np.asarray(solution).ravel()[:n], dtype=np.float64, copy=True)))
            except Exception:  # noqa: BLE001 — never raise into cuOpt
                pass

    class Set(SetSolutionCallback):
        def set_solution(self, solution, solution_cost, solution_bound, user_data):
            t = time.monotonic() - t0
            rec = dict(t=t, offered=False, obj=None)
            try:
                if offers and t >= a.inject_after:
                    x = offers.pop(0)
                    solution[:] = x
                    o = float(c @ x + off) if a.mode == "bad" else obj_inj
                    solution_cost[0] = o
                    rec.update(offered=True, obj=o)
            finally:
                sets.append(rec)

    log = Path(tempfile.mkdtemp(prefix="cuopt-setsol-")) / "cuopt.log"
    st = SolverSettings()
    for k, v in dict(time_limit=float(a.time), log_to_console=False, num_cpu_threads=int(a.threads),
                     random_seed=int(a.seed), log_file=str(log)).items():
        st.set_parameter(k, v)
    get_cb = Get()
    st.set_mip_callback(get_cb, None)
    if a.mode != "none":
        set_cb = Set()
        st.set_mip_callback(set_cb, None)
    ts = time.monotonic()
    sol = Solve(dm, st)
    wall = time.monotonic() - ts
    text = log.read_text(errors="replace") if log.exists() else ""
    pres = [ln for ln in text.splitlines() if re.search(r"[Pp]resolv|Papilo|after presolve|dual", ln)][:40]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"), names=np.array(names),
                        t=np.array([g[0] for g in gets]), obj=np.array([g[1] for g in gets]),
                        X=np.array([g[2] for g in gets]) if gets else np.zeros((0, n)))
    try:
        import importlib.metadata as md
        ver = {d: md.version(d) for d in ("cuopt-cu13", "libcuopt-cu13")}
    except Exception:  # noqa: BLE001
        ver = {"cuopt": getattr(cuopt, "__version__", "?")}
    res = dict(instance=a.instance, mode=a.mode, versions=ver, time_limit=a.time, wall_s=wall,
               injected_obj=obj_inj if a.mode in ("inject", "raw") else None,
               termination=int(sol.get_termination_status()), error=int(sol.get_error_status()),
               final_obj=float(sol.get_primal_objective()) if hasattr(sol, "get_primal_objective") else None,
               presolve_log=pres, n_get=len(gets), gets=[(round(g[0], 3), g[1]) for g in gets],
               n_set=len(sets), set_offers=[s for s in sets if s["offered"]], set_call_t=[round(s["t"], 3) for s in sets],
               rmm_peak_gb=None, log_path=str(log))
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("mode", "versions", "termination", "final_obj", "n_get", "n_set",
                                          "set_offers")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
