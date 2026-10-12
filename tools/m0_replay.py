"""D-021's M0 (offline, no box): replay R12's stop at the proven optimum on every recorded run and on GAMS's 3,495
traces, and A9's shape-only memory plan over MIPfeas's 233.

1. **Our runs** (`--runs` / `--inc`: a run table and its checked-incumbent table, read only): for every run whose
   instance MIPLIB lists as `=opt=` (R12's eligibility, `pair.miplib_stop_ref`) the stop is applied offline — the
   first stored incumbent before T, over every process, whose objective (the exact checker's recomputed one; the
   reported one if it was never checked) meets `pair.ref_reached` against MIPLIB's z*. If that incumbent failed the
   exact check, the live run would have been void (`ref_rejected`) and rerun without the stop: counted, nothing saved.
   Otherwise the run's checked portfolio is cut at the stop and its last incumbent held to T, and `pt_gams`, Berthold's
   P(T), `pt_mf` (T = the run's budget) and `pt_gams` on the launch clock are compared with the uncut run: D-021 asks
   `pt_gams` unchanged to 1e-12 and P, `pt_mf` within 5e-7. Hours saved = the run's end (run.json `wall_s`, read from
   `--root`, capped at T; T when the record is unreadable) minus the stop, per set and arm.
2. **GAMS's traces** (`--gams-dir`, all 3,495): the same stop on each trace's best-so-far steps against GAMS's z*
   (T = 600), the same comparisons, hours as run (each trace's E record, capped at 600 s) and with the stop, per
   machine and solver.
3. **A9's shape-only memory plan** (`--plan`): per instance of the 233, in a child process each (a parse that hits the
   analysis's memory cap loses one instance, recorded, not the run): the instance read as the EA reads it (HiGHS),
   A9's effective configuration (`configs/itrea2-dev-rc-3way.yaml` + exchange.from_cuopt), the gene dtype, and the
   EA's own refusals before the clock — a gene dtype that cannot hold the instance, a population that does not fit
   gpu_mem_gb (`plan_memory`), no Rank 0 slot — plus the child's peak RSS beside the EA's 6-GB host cap. No outcome
   is read (R29: an instance A9 cannot start on is an A9 outcome, never on the both-arms list).

CPU only, single-threaded, outside any window; run it pinned and capped:

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 \
        taskset -c 15-19 python tools/m0_replay.py --runs ../i-trea-ii/datasets/runs.parquet \
        --inc ../i-trea-ii/datasets/incumbents_checked.parquet --root ../i-trea-ii --plan [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from itrea import paths
from itrea.analysis import metrics as M
from itrea.analysis import mipfeas as MF
from itrea.experiments import pair

TOL_GAMS, TOL_OTHER = 1e-12, 5e-7              # D-021 M0: pt_gams unchanged to 1e-12; P and pt_mf within 5e-7
A9_CONFIG = "configs/itrea2-dev-rc-3way.yaml"
EA_HOST_GB = 6.0


# ------------------------------------------------------------------------------------------------- the replay
def stop_time(points, z: float, sense: int, T: float) -> tuple[float | None, int | None]:
    """The first point (t, obj) before T, in time order, that meets R12's test: (t, its index) or (None, None)."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], i))
    for i in order:
        t, v = points[i]
        if t < T and v is not None and math.isfinite(v) and pair.ref_reached(v, z, sense)[0]:
            return float(t), i
    return None, None


def metrics_of(steps, z: float, sense: int, T: float, offset: float = math.nan) -> dict:
    out = dict(pt_gams=M.gams_integral(steps, z, sense, T).value, P=M.primal_integral(steps, z, sense, T).value,
               pt_mf=M.mipfeas_integral(steps, z, sense, T))
    if math.isfinite(offset):
        out["pt_gams_launch"] = M.gams_integral([(t + offset, v) for t, v in steps], z, sense, T).value
    return out


def compare(steps, t_stop: float, z: float, sense: int, T: float, offset: float = math.nan) -> dict:
    """Full vs cut-at-the-stop (the last incumbent held to T): the metrics and their absolute differences."""
    full = metrics_of(steps, z, sense, T, offset)
    cut = metrics_of([(t, v) for t, v in steps if t <= t_stop], z, sense, T, offset)
    return {**{k: full[k] for k in full}, **{f"d_{k}": abs(cut[k] - full[k]) for k in full}}


def _wall(root: Path | None, run_dir: str) -> float:
    if root is None or not run_dir:
        return math.nan
    try:
        return float(json.loads((root / run_dir / "run.json").read_text()).get("wall_s") or math.nan)
    except (OSError, ValueError, TypeError):
        return math.nan


def replay_runs(runs: pd.DataFrame, inc: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    """Part 1 (module docstring): one row per run."""
    r = runs[~runs["synthetic"].astype(bool)] if "synthetic" in runs.columns else runs
    inc = inc.assign(obj_used=np.where(inc["checked"].astype(bool), inc["obj_recomputed"], inc["obj_reported"]))
    by_run = {k: g for k, g in inc.groupby("run_id", sort=False)}
    rows = []
    for _, row in r.iterrows():
        rid, T, s = str(row["run_id"]), float(row["budget"]), int(row["sense"])
        ref, why = pair.miplib_stop_ref(dict(instance=str(row.get("instance_path") or "")))
        end = _wall(root, str(row.get("run_dir") or ""))
        end = min(T, end) if math.isfinite(end) else T
        out = dict(run_id=rid, set=str(row["set"]), arm=str(row["arm"]), instance=str(row["instance"]), budget=T,
                   eligible=ref is not None, end_s=end, t_stop=math.nan, would_void=False, ref_mismatch=False)
        if ref is None:
            rows.append(out)
            continue
        g = by_run.get(rid)
        if g is None or not len(g):
            rows.append(out)
            continue
        out["ref_mismatch"] = bool(math.isfinite(float(row["ref"])) and float(row["ref"]) != ref.value)
        pts = list(zip(g["t"].astype(float), g["obj_used"].astype(float)))
        t_stop, i = stop_time(pts, ref.value, s, T)
        if t_stop is None:
            rows.append(out)
            continue
        out["t_stop"] = t_stop
        if not bool(g["ok"].iloc[i]):
            out["would_void"] = True                 # live: void (ref_rejected), rerun without the stop
            rows.append(out)
            continue
        ok = g[g["ok"].astype(bool)]
        roles = [M.best_steps(zip(x["t"], x["obj_recomputed"]), s) for _, x in ok.groupby("role")]
        steps = M.merge_steps(*roles, sense=s)
        out.update(compare(steps, t_stop, ref.value, s, T, float(row.get("startup_offset_s", math.nan))))
        out["parquet_pt_gams_diff"] = abs(out["pt_gams"] - float(row["pt_gams"])) if "pt_gams" in row else math.nan
        rows.append(out)
    return pd.DataFrame(rows)


def replay_gams(gams_dir: Path, solu_path: Path | None = None) -> pd.DataFrame:
    """Part 2 (module docstring): one row per GAMS trace."""
    z = MF.read_z(gams_dir)
    rows = []
    for p in sorted((Path(gams_dir) / "trc_files").glob("*.trc")):
        name, machine, solver = p.name[:-4].rsplit("_", 2)
        trc = MF.read_gams_trc(p)
        e = trc[trc["seriesID"].astype(str).str.strip() == "E"]
        end = min(600.0, float(e["seconds"].iloc[-1])) if len(e) else 600.0
        ref, why = pair.miplib_stop_ref(dict(instance=f"data/miplib/inst/{name}.mps.gz"), solu_path)
        out = dict(trace=p.name, instance=name, run=f"{machine}_{solver}", end_s=end, eligible=ref is not None,
                   t_stop=math.nan)
        if ref is not None and name in z:
            steps = M.best_steps(MF.trc_points(trc), 1)
            t_stop, _ = stop_time(steps, z[name], 1, 600.0)
            if t_stop is not None:
                out["t_stop"] = t_stop
                out.update(compare(steps, t_stop, z[name], 1, 600.0))
        rows.append(out)
    return pd.DataFrame(rows)


def hours(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    d = df.assign(with_stop=np.where(df["t_stop"].notna() & ~df.get("would_void", False),
                                     np.minimum(df["end_s"], df["t_stop"]), df["end_s"]),
                  stopped=df["t_stop"].notna() & ~df.get("would_void", False))
    g = d.groupby(by, sort=True)
    out = g.agg(runs=("end_s", "size"), eligible=("eligible", "sum"), stopped=("stopped", "sum"),
                hours_as_run=("end_s", lambda x: x.sum() / 3600), hours_with_stop=("with_stop", lambda x: x.sum() / 3600))
    out["hours_saved"] = out["hours_as_run"] - out["hours_with_stop"]
    for k in ("d_pt_gams", "d_P", "d_pt_mf", "d_pt_gams_launch"):
        if k in d.columns:
            out[f"max_{k}"] = g[k].max()
    return out.reset_index()


# ------------------------------------------------------------------------------------------------ memory plan
def plan_one(name: str, data: Path) -> dict:
    """Part 3 for one instance (runs in a child process)."""
    from itrea.ea.engine import plan_memory
    from itrea.ea.population import gene_dtype_for
    from itrea.problem import load
    inst = f"data/miplib/inst/{name}.mps.gz"
    out: dict = dict(name=name, refused="", reason="")
    t = time.monotonic()
    prob = load(str(Path(data) / "miplib" / "inst" / f"{name}.mps.gz"))
    out.update(load_s=round(time.monotonic() - t, 3), n=prob.n, m=prob.m, nnz=prob.nnz, conts=int(prob.conts.size))
    cfg, _ = pair.effective_config(dict(arm="A9", config=A9_CONFIG, instance=inst, seed=0, budget=600, set="m0"))
    gdt = gene_dtype_for(np.asarray(prob.lb), np.asarray(prob.ub), np.asarray(prob.isint, bool),
                         cfg.population.gene_dtype)
    out["gene_dtype"] = np.dtype(gdt).name
    if np.dtype(gdt).kind in "iu":                      # the EA's own two refusals (ea.engine prepare)
        info = np.iinfo(gdt)
        lb, ub = np.asarray(prob.lb), np.asarray(prob.ub)
        if prob.conts.size:
            out.update(refused="gene_dtype", reason=f"{gdt} cannot hold {prob.conts.size} continuous variables")
        elif prob.n and not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)) and lb.min() >= info.min
                             and ub.max() <= info.max):
            out.update(refused="gene_dtype", reason=f"{gdt} cannot hold the integer bounds")
    plan = plan_memory(prob, cfg, np.dtype(gdt))
    out.update(P=plan.P, offspring=plan.offspring, total_mb=round(plan.total_bytes / 2**20, 1),
               limit_mb=round(plan.limit_bytes / 2**20, 1), usable_mb=round(0.9 * plan.limit_bytes / 2**20, 1),
               fits=bool(plan.fits))
    if not out["refused"] and not plan.fits:
        out.update(refused="does_not_fit", reason=f"P={plan.P}: {out['total_mb']} MB > 0.9 x {out['limit_mb']} MB")
    if not out["refused"] and cfg.ranks.x > 0 and math.floor(cfg.ranks.x * plan.P + 1e-12) < 1:
        out.update(refused="no_rank0_slot", reason=f"P={plan.P} leaves no Rank 0 slot for x={cfg.ranks.x}")
    out["maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    return out


def memory_plan(names, data: Path, timeout_s: float = 900.0, log=print) -> pd.DataFrame:
    rows = []
    for k, n in enumerate(names, 1):
        try:
            pr = subprocess.run([sys.executable, __file__, "--plan-one", n, "--data", str(data)], capture_output=True,
                                timeout=timeout_s, env=dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=""))
            if pr.returncode == 0:
                rows.append(json.loads(pr.stdout.decode().strip().splitlines()[-1]))
            else:
                rows.append(dict(name=n, refused="analysis_error", reason=f"child rc={pr.returncode}: "
                                 + pr.stderr.decode(errors="replace").strip().splitlines()[-1:][0][:200]
                                 if pr.stderr.strip() else f"child rc={pr.returncode} (killed: the analysis cap?)"))
        except subprocess.TimeoutExpired:
            rows.append(dict(name=n, refused="analysis_error", reason=f"no answer in {timeout_s:g} s"))
        if k % 25 == 0:
            log(f"memory plan: {k} of {len(names)}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------- main
def _fmt(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}g}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", help="run table (datasets/runs.parquet)")
    ap.add_argument("--inc", help="checked-incumbent table (datasets/incumbents_checked.parquet)")
    ap.add_argument("--root", type=Path, help="the checkout the run table's run_dir paths are relative to (wall_s)")
    ap.add_argument("--gams-dir", type=Path, default=MF.GAMS_DIR)
    ap.add_argument("--data", type=Path, default=paths.DATA)
    ap.add_argument("--plan", action="store_true", help="also A9's shape-only memory plan over the 233")
    ap.add_argument("--plan-one", help=argparse.SUPPRESS)
    ap.add_argument("--json", type=Path, help="write every table as JSON here")
    a = ap.parse_args(argv)
    if a.plan_one:
        print(json.dumps(plan_one(a.plan_one, a.data)))
        return 0
    res: dict = {}
    ok = True
    if a.runs and a.inc:
        runs, inc = pd.read_parquet(a.runs), pd.read_parquet(a.inc)
        rr = replay_runs(runs, inc, a.root)
        st = rr[rr["t_stop"].notna() & ~rr["would_void"]]
        h = hours(rr, ["set", "arm"])
        print(f"## Our runs ({a.runs}): {len(rr)} runs, {int(rr['eligible'].sum())} R12-eligible, {len(st)} would "
              f"stop, {int(rr['would_void'].sum())} would be void (the triggering incumbent failed the check)")
        for k, tol in (("d_pt_gams", TOL_GAMS), ("d_P", TOL_OTHER), ("d_pt_mf", TOL_OTHER),
                       ("d_pt_gams_launch", TOL_GAMS)):
            m = float(st[k].max()) if k in st and len(st) and st[k].notna().any() else 0.0
            good = m <= tol
            ok &= good
            print(f"max |{k[2:]} cut - uncut| = {m:.3e} (limit {tol:g}): {'PASS' if good else 'FAIL'}")
        if "parquet_pt_gams_diff" in st and len(st):
            print(f"replayed uncut pt_gams vs the table's: max |diff| {float(st['parquet_pt_gams_diff'].max()):.3e}")
        print(f"reference check: {int(rr['ref_mismatch'].sum())} run(s) whose table reference differs from MIPLIB's z*")
        print(f"hours as run {h['hours_as_run'].sum():.2f}, with the stop {h['hours_with_stop'].sum():.2f}, saved "
              f"{h['hours_saved'].sum():.2f}")
        print(h.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        res["runs"], res["runs_by_arm"] = rr, h
    gd = Path(a.gams_dir)
    if (gd / "trc_files").is_dir():
        gr = replay_gams(gd, a.data / "miplib" / "solu.txt")
        st = gr[gr["t_stop"].notna()]
        print(f"\n## GAMS's traces ({gd}): {len(gr)} traces, {int(gr['eligible'].sum())} R12-eligible, {len(st)} "
              f"reach the proven optimum before 600 s")
        for k, tol in (("d_pt_gams", TOL_GAMS), ("d_P", TOL_OTHER), ("d_pt_mf", TOL_OTHER)):
            m = float(st[k].max()) if len(st) else 0.0
            good = m <= tol
            ok &= good
            print(f"max |{k[2:]} cut - uncut| = {m:.3e} (limit {tol:g}): {'PASS' if good else 'FAIL'}")
        h = hours(gr.assign(would_void=False), ["run"])
        print(h.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        res["gams"], res["gams_by_run"] = gr, h
    if a.plan:
        names = sorted(pd.read_csv(paths.DATASETS / "mipfeas_strata.csv")["name"])
        mp = memory_plan(names, a.data)
        ref = mp[mp["refused"].astype(str) != ""]
        print(f"\n## A9's shape-only memory plan ({A9_CONFIG}, gpu_mem_gb from it): {len(mp)} instances, "
              f"{len(ref)} refused or not analysed")
        for _, r in ref.iterrows():
            print(f"- {r['name']}: {r['refused']} — {r['reason']}")
        if "maxrss_mb" in mp:
            big = mp[mp["maxrss_mb"].astype(float) >= 0.5 * EA_HOST_GB * 1024]
            print(f"parse peak RSS >= half the EA's {EA_HOST_GB:g}-GB host cap (risk, not a refusal): "
                  + (", ".join(f"{r['name']} {r['maxrss_mb']:.0f} MB" for _, r in big.iterrows()) or "none"))
            print(f"P: median {mp['P'].median():g}, min {mp['P'].min():g} ({mp.loc[mp['P'].idxmin(), 'name']})")
        res["memory_plan"] = mp
    if a.json:
        a.json.write_text(json.dumps({k: json.loads(v.to_json(orient="records")) for k, v in res.items()}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
