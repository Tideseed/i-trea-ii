"""D-022 report: A9 under the external harness — cuOpt's MIP gaps 0 and its thread variables unset (D-021 R13 + R14),
development test (D-010) — and C1's interim A9 result on the 17 instances (D-019 addendum). The rules are D-022's,
fixed before any run; nothing is chosen by eye.

Usage (repository root, after `itrea build-runs`; untimed analysis: OMP_NUM_THREADS=1, never beside timed runs with
default thread counts):

    OMP_NUM_THREADS=1 .venv/bin/python tools/d022_report.py [--set itx-r13] [--failed both|exclude|keep] [--json out]
    itrea d022-report …                       (the same, through the entry point)

Reads `datasets/runs.parquet` (P(120) = `pt`, `final_gap`, `status`, `attempt`, `run_dir`, …) and, through each run's
`run_dir` (relative to `--root`), its `run.json` (the cuOpt process record, the compile-on-the-clock flags, the
harness), `cuopt.summary.json` (the solve's `settings`, `exit_reason`) and `ea.summary.json` (`P`, `gene_dtype`), and
each set's void log `<raw>/<set>/void.jsonl` (the pool's record of voided runs, moved aside and not in the table). The
checked-incumbents table is not needed: every number here is a run-table column or a run-directory field.

Arms (`jobs/itx-r13.jsonl`, set itx-r13, 120 s, seeds 0–1, the 17 development + class-probe instances): control A9@r13
(A9 under the harness {ext, cuopt_gap_zero false, cuopt_env_unset false}), treatment A9@r13x (A9 under ext), same-day
A6F@r13x (cuOpt alone on F, ext). A0 for the C1 block: the runs of sets `dev` / `classprobe`, 120 s, seeds 0–1.

Validity (must hold; else the verdict is INVALID):
  - no compile on the clock: run.json `compiled_on_clock_cuda` (the project's CUDA cache, every run) and, on the arms
    with an EA GPU process (A9), the CuPy cache (run.json `cupy_cache.compiled_on_clock`, the EA summary's
    `compiled_on_clock`) — the runner's own rule (`pair.void_checks`), so A6F is not charged with a CuPy flag the
    runner never applies to it;
  - every run completed or a recorded void rerun: every void in a set's `void.jsonl` has its rerun (the same run id)
    completed in the table; a missing run makes the verdict INCOMPLETE; a FAILED run breaks this leg by the letter —
    reading `keep` counts it as D-002 §6 keeps crashes (truncated at the crash, as build-runs scores it), a stated
    deviation that the report names;
  - equal P and gene dtype per block (the two A9 runs of an instance x seed, from their EA summaries);
  - cuOpt's gaps 0 in every treatment solve and default in every control solve: the cuOpt summary's `settings`
    carries mip_absolute_gap = mip_relative_gap = 0 for A9@r13x and A6F@r13x and neither key for A9@r13 (cuOpt's
    default, relative 1e-4); a run without cuOpt settings fails the leg.
  R14 is reported beside: the cuOpt process's `env_unset` (OMP / OPENBLAS / MKL) under ext, empty in the control.

Mechanism (reported): per A9 arm, the runs ended by cuOpt's "optimal" — the cuOpt summary's `exit_reason` optimal and
cuOpt's process the first to exit (its `exit_t` <= every other process's; so a cuOpt that finished inside the kill
window after SCIP's proof does not count) — and, of those, the ones whose run ends with a final gap > 1e-9 against
the reference (`final_gap`, the portfolio's; 1e-9 = the relative tolerance used for objective equality elsewhere,
so floating-point noise of an optimal value is not counted); cuOpt CPU-s per wall-s per arm = (cpu_user_s +
cpu_sys_s) of the cuOpt process (run.json `processes`) / its `exit_t` (seconds from t0 to its exit) — D-021 M1's
reading (A6F 5.02 / 4.99 / 5.03); start-up CPU is in the numerator, so it reads ~1–3 % high.

Outcome (bounded d of the treatment vs the control on seed means, positive = treatment better; per seed
d_s = effect_d(P_ctrl,s, P_treat,s); band_i = max(0.05, the larger seed-0 vs seed-1 bounded difference |a − b| /
max(a, b, 1e-3) of the control or of A6F@r13x); the mean over the 17 counts every instance, both-P = 1 ones at d = 0):
  - REJECT (A9 keeps the default gaps; D-021's A9 rows run under {ext, gap_zero false, env_unset false}) if the mean
    d <= −0.05 or any instance has d_s < −band_i on both seeds;
  - ADOPT R13 + R14 for A9 (the C1 freeze carries harness ext) if the mean d >= −0.02 and no instance is worse than
    −band_i on both seeds;
  - otherwise INCONCLUSIVE (−0.05 < mean d < −0.02, no instance worse on both seeds) -> seeds 2–3 on the instances
    worse than −band_i on exactly one seed, same rules (the report names them, or says that there are none).

C1's interim block (reported, not criteria; D-013 terms): A9@r13x and A9@r13 against A6F@r13x (equal footprint,
same day) and against SCIP alone (A0 of `dev` / `classprobe`) — better / worse, mean d with its 95 % bootstrap
interval and the Wilcoxon test (`itrea.analysis.stats.compare`, which leaves out instances with P = 1 in both arms on
every seed); and each solver's weak set (A6F@r13x P >= 0.5, A0 P >= 0.5, seed means): improved, mean P before ->
after (and its relative reduction), mean d with interval, Wilcoxon, worst P after coupling. Caveats printed with it:
cuOpt's weak set is selected on the same runs it is compared with; the A0 runs are from earlier days.

Failed runs: both readings (`--failed both`, the default) — `exclude` (the letter: a failed run is missing, the
verdict INCOMPLETE) and `keep` (D-002 §6). Run identity: a job run again in another set keeps its run id and the run
table qualifies the later copy as `<run_id>#<set>` (build_runs.qualify_reused_ids). Runs are therefore selected by the
`set` column and keyed by (instance, arm, seed) — the run id is never parsed —, the directory is `run_dir`; when one
key occurs in more than one selected set the bare id is kept (then the highest attempt) and the others are listed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.analysis.backoff import _jsonable, bounded
from itrea.analysis.stats import compare, effect_d
from itrea.paths import DATASETS, ROOT

CTRL, TREAT, A6F, A0 = "A9@r13", "A9@r13x", "A6F@r13x", "A0"
ARMS = (CTRL, TREAT, A6F)
A9_ARMS = (CTRL, TREAT)
GAP_ZERO_ARMS = (TREAT, A6F)              # every cuOpt solve under the ext harness
SET, SEEDS, BUDGET = "itx-r13", (0, 1), 120.0
A0_SETS = ("dev", "classprobe")
BAND_MIN, ADOPT_MEAN, REJECT_MEAN = 0.05, -0.02, -0.05
GAP_TOL = 1e-9
WEAK_P = 0.5
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
REUSE_SEP = "#"


def _fin(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _read(p: Path) -> dict:
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def development_instances() -> list[str]:
    from itrea.experiments.make_jobs import development_names
    return development_names()


# ----------------------------------------------------------------------------------------------- selection
def select(runs: pd.DataFrame, sets, arms, seeds=SEEDS, budget: float = BUDGET, keep_failed: bool = False
           ) -> tuple[pd.DataFrame, list[dict]]:
    """The runs of `sets` x `arms` x `seeds` at `budget`, completed (and failed with `keep_failed`), not synthetic, not
    in a voided block; one row per (instance, arm, seed): the bare run id before a `#<set>` copy, then the highest
    attempt. Returns (rows, the duplicates left out)."""
    df = runs[runs["set"].isin(list(sets)) & runs["arm"].isin(list(arms)) & runs["seed"].isin(list(seeds))]
    if "budget" in df:
        df = df[np.isclose(df["budget"].astype(float), float(budget))]
    for col in ("synthetic", "block_void"):
        if col in df:
            df = df[~df[col].fillna(False).astype(bool)]
    ok = {"completed", "failed"} if keep_failed else {"completed"}
    df = df[df["status"].isin(ok)].copy()
    df["_reused"] = df["run_id"].astype(str).str.contains(REUSE_SEP, regex=False)
    df["_attempt"] = df["attempt"].fillna(0).astype(int) if "attempt" in df else 0
    df = df.sort_values(["instance", "arm", "seed", "_reused", "_attempt"], ascending=[True, True, True, True, False],
                        kind="mergesort")
    dup = df.duplicated(["instance", "arm", "seed"], keep="first")
    dropped = [dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"]), set=r["set"], run_id=r["run_id"])
               for r in df[dup].to_dict("records")]
    return df[~dup].drop(columns=["_reused", "_attempt"]), dropped


# ----------------------------------------------------------------------------------------------- one run's files
def run_info(row: dict, root: Path) -> dict:
    """What validity and the mechanism read from one run's directory."""
    p = Path(str(row.get("run_dir") or ""))
    d = p if p.is_absolute() else root / p
    rj, cs = _read(d / "run.json"), _read(d / "cuopt.summary.json")
    base = str(rj.get("base_arm") or str(row["arm"]).split("@")[0].split("~")[0])
    es = _read(d / "ea.summary.json") if base == "A9" else {}
    procs = rj.get("processes") if isinstance(rj.get("processes"), list) else []
    cu = next((x for x in procs if isinstance(x, dict) and x.get("role") == "cuopt"), {})
    cuda = rj.get("compiled_on_clock_cuda")
    cupy = (rj.get("cupy_cache") or {}).get("compiled_on_clock") if isinstance(rj.get("cupy_cache"), dict) else None
    compiled = []
    if cuda:
        compiled.append("CUDA cache")
    if base == "A9" and (cupy or es.get("compiled_on_clock")):
        compiled.append("CuPy cache")
    st = cs.get("settings") if isinstance(cs.get("settings"), dict) else None
    gaps = None if st is None else (st.get("mip_absolute_gap"), st.get("mip_relative_gap"))
    exits = {x.get("role"): x.get("exit_t") for x in procs if isinstance(x, dict)}
    t_cu = exits.get("cuopt")
    others = [t for r, t in exits.items() if r != "cuopt" and _fin(t)]
    cpu = (float(cu["cpu_user_s"]) + float(cu.get("cpu_sys_s") or 0.0)) if _fin(cu.get("cpu_user_s")) else math.nan
    harness = (rj.get("config") or {}).get("harness") if isinstance(rj.get("config"), dict) else None
    return dict(dir=str(d), base_arm=base, compiled=compiled, settings=st, gaps=gaps,
                gap_zero=bool(gaps is not None and gaps[0] == 0.0 and gaps[1] == 0.0),
                gap_default=bool(st is not None and "mip_absolute_gap" not in st and "mip_relative_gap" not in st),
                env_unset=sorted(cu.get("env_unset") or []), cuopt_exit=cs.get("exit_reason"), cuopt_exit_t=t_cu,
                cuopt_first=bool(_fin(t_cu) and all(float(t_cu) <= float(t) for t in others)),
                cpu_per_wall=cpu / float(t_cu) if _fin(cpu) and _fin(t_cu) and float(t_cu) > 0 else math.nan,
                P=es.get("P"), gene_dtype=es.get("gene_dtype"), harness=harness, run_json=bool(rj))


def void_log(df: pd.DataFrame, sets, root: Path) -> list[dict]:
    """The pool's void records (`<raw>/<set>/void.jsonl`) of the selected sets: the directories next to the selected
    runs' directories and `<root>/traces/raw/<set>`."""
    out, seen = [], set()
    for s in sets:
        dirs = {root / "traces" / "raw" / s}
        for rd in df.loc[df["set"] == s, "run_dir"].astype(str):
            p = Path(rd)
            dirs.add((p if p.is_absolute() else root / p).parent)
        for d in sorted(dirs):
            f = d / "void.jsonl"
            if f in seen or not f.exists():
                continue
            seen.add(f)
            for line in f.read_text().splitlines():
                if line.strip():
                    try:
                        out.append(dict(json.loads(line), set=s))
                    except ValueError:
                        out.append(dict(set=s, unreadable=line[:200]))
    return out


# ----------------------------------------------------------------------------------------------- the rules
def validity(df: pd.DataFrame, info: dict, instances, voids: list[dict], failed: list[dict]) -> dict:
    """D-022's validity legs (module docstring). `info`: {run_id: run_info}; `failed`: the failed runs of the set
    (counted under the reading `keep`, missing under `exclude`)."""
    have = {(r["instance"], r["arm"], int(r["seed"])): r for r in df.to_dict("records")}
    missing = [(i, a, s) for i in instances for a in ARMS for s in SEEDS if (i, a, s) not in have]
    compiled = [dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"]), what=info[r["run_id"]]["compiled"])
                for r in have.values() if info[r["run_id"]]["compiled"]]
    no_record = [dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"])) for r in have.values()
                 if not info[r["run_id"]]["run_json"]]
    completed_ids = {str(r["run_id"]).split(REUSE_SEP)[0] for r in have.values() if r["status"] == "completed"}
    void_rows = [dict(v, rerun_completed=str(v.get("run_id", "")) in completed_ids) for v in voids]
    block = []
    for i in instances:
        for s in SEEDS:
            a, b = have.get((i, CTRL, s)), have.get((i, TREAT, s))
            if a is None or b is None:
                continue
            ia, ib = info[a["run_id"]], info[b["run_id"]]
            if ia["P"] is None or ib["P"] is None or ia["P"] != ib["P"] or ia["gene_dtype"] != ib["gene_dtype"]:
                block.append(dict(instance=i, seed=s, P=(ia["P"], ib["P"]), gene_dtype=(ia["gene_dtype"],
                                                                                     ib["gene_dtype"])))
    gaps = []
    for r in have.values():
        x = info[r["run_id"]]
        want = "0" if r["arm"] in GAP_ZERO_ARMS else "default"
        ok = x["gap_zero"] if want == "0" else x["gap_default"]
        if not ok:
            gaps.append(dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"]), want=want,
                             settings=x["settings"]))
    env = []
    for r in have.values():
        x = info[r["run_id"]]
        want_unset = r["arm"] in GAP_ZERO_ARMS
        if (set(THREAD_VARS) <= set(x["env_unset"])) != want_unset or (not want_unset and x["env_unset"]):
            env.append(dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"]), env_unset=x["env_unset"]))
    voids_open = [v for v in void_rows if not v["rerun_completed"]]
    hard_ok = not compiled and not block and not gaps and not no_record
    return dict(ok=bool(hard_ok and not missing and not failed and not voids_open), hard_ok=bool(hard_ok),
                missing=missing, failed=failed, compiled=compiled, no_run_record=no_record, voids=void_rows,
                voids_without_rerun=voids_open, block_mismatch=block, gaps=gaps, r14_env=env)


def mechanism(df: pd.DataFrame, info: dict) -> dict:
    """Runs ended by cuOpt's "optimal" (cuOpt first), those with a final gap > GAP_TOL, and cuOpt CPU-s per wall-s,
    per arm."""
    out = {}
    for arm in ARMS:
        g = df[df["arm"] == arm]
        ended, gap = [], []
        cpw = []
        for r in g.to_dict("records"):
            x = info[r["run_id"]]
            if _fin(x["cpu_per_wall"]):
                cpw.append(x["cpu_per_wall"])
            if x["cuopt_exit"] == "optimal" and x["cuopt_first"]:
                fg = float(r["final_gap"]) if _fin(r.get("final_gap")) else math.nan
                row = dict(instance=r["instance"], seed=int(r["seed"]), cuopt_exit_t=x["cuopt_exit_t"], final_gap=fg)
                ended.append(row)
                if _fin(fg) and fg > GAP_TOL:
                    gap.append(row)
        out[arm] = dict(n=int(len(g)), ended_by_cuopt_optimal=ended, with_final_gap=gap,
                        cpu_per_wall=dict(n=len(cpw), median=float(np.median(cpw)) if cpw else math.nan,
                                          min=min(cpw) if cpw else math.nan, max=max(cpw) if cpw else math.nan))
    return out


def outcome(df: pd.DataFrame, instances) -> dict:
    """D-022's outcome rule (module docstring)."""
    P = {(r["instance"], r["arm"], int(r["seed"])): float(r["pt"]) for r in df.to_dict("records") if _fin(r["pt"])}
    rows, missing = [], []
    for i in instances:
        pc = [P.get((i, CTRL, s)) for s in SEEDS]
        pt = [P.get((i, TREAT, s)) for s in SEEDS]
        pa = [P.get((i, A6F, s)) for s in SEEDS]
        if any(v is None for v in pc + pt):
            missing.append(i)
            rows.append(dict(instance=i, d=math.nan))
            continue
        ds = [effect_d(c, t) for c, t in zip(pc, pt)]
        spreads = [bounded(pc[0], pc[1])] + ([bounded(pa[0], pa[1])] if None not in pa else [])
        band = max([BAND_MIN] + [v for v in spreads if _fin(v)])
        worse = [d < -band for d in ds]
        rows.append(dict(instance=i, p_ctrl=float(np.mean(pc)), p_treat=float(np.mean(pt)),
                         p_a6f=float(np.mean(pa)) if None not in pa else math.nan,
                         d=effect_d(float(np.mean(pc)), float(np.mean(pt))), d_seed=ds, band=band,
                         worse_both=bool(all(worse)), worse_one=bool(sum(worse) == 1)))
    if missing:
        return dict(verdict=f"INCOMPLETE: no seed pair of {CTRL} and {TREAT} on {missing}", rows=rows,
                    mean_d=math.nan, worse_both=[], worse_one=[], missing=missing)
    mean_d = float(np.mean([r["d"] for r in rows]))
    wb = [r["instance"] for r in rows if r["worse_both"]]
    wo = [r["instance"] for r in rows if r["worse_one"]]
    if mean_d <= REJECT_MEAN or wb:
        why = ([f"mean d {mean_d:+.3f} <= {REJECT_MEAN}"] if mean_d <= REJECT_MEAN else []) + \
              ([f"worse than −band on both seeds: {wb}"] if wb else [])
        verdict = ("REJECT R13 + R14 for A9 (" + "; ".join(why) + "): A9 keeps cuOpt's default gaps; D-021's A9 rows "
                   "run under {ext, cuopt_gap_zero false, cuopt_env_unset false}")
    elif mean_d >= ADOPT_MEAN:
        verdict = (f"ADOPT R13 + R14 for A9 (mean d {mean_d:+.3f} >= {ADOPT_MEAN}, no instance worse than −band on "
                   "both seeds): the C1 freeze carries harness ext")
    else:
        verdict = (f"INCONCLUSIVE (mean d {mean_d:+.3f} in ({REJECT_MEAN}, {ADOPT_MEAN}), no instance worse on both "
                   "seeds): seeds 2–3 on the instances worse than −band on one seed, same rules — "
                   + (", ".join(wo) if wo else "none is (the rule names no instance; the record must say what runs)"))
    return dict(verdict=verdict, rows=rows, mean_d=mean_d, worse_both=wb, worse_one=wo, missing=[])


def _weak_block(long: pd.DataFrame, arm: str, base: str, weak: list[str]) -> dict:
    """D-013 terms on a weak set: improved, mean P before -> after, mean d with its bootstrap interval, Wilcoxon,
    worst P after coupling."""
    sub = long[long["instance"].isin(weak)]
    c = compare(sub, arm, base) if len(weak) else None
    if c is None or not c["n_instances"] + c["n_excluded"]:
        return dict(n=0)
    pa = sub[sub["arm"] == arm].groupby("instance")["pt"].mean()
    red = (c["mean_p_ctrl"] - c["mean_p_arm"]) / c["mean_p_ctrl"] if c["mean_p_ctrl"] else math.nan
    return dict(n=c["n_instances"], better=c["better"], worse=c["worse"], mean_P_before=c["mean_p_ctrl"],
                mean_P_after=c["mean_p_arm"], P_reduction=red, mean_d=c["mean_d"], ci=(c["ci_lo"], c["ci_hi"]),
                p=c["p"], worst_P_after=float(pa.max()) if len(pa) else math.nan)


def c1_interim(df: pd.DataFrame, a0: pd.DataFrame, instances) -> dict:
    """C1's interim A9 result (reported, not criteria): both A9 arms against A6F@r13x and against A0."""
    cols = ["instance", "seed", "arm", "pt"]
    long = pd.concat([df[cols], a0[cols]], ignore_index=True)
    long = long[long["instance"].isin(list(instances)) & long["pt"].notna()]
    means = long.groupby(["arm", "instance"])["pt"].mean()
    out = {}
    for base, label in ((A6F, "cuOpt alone (A6F@r13x, same day)"), (A0, "SCIP alone (A0, dev / classprobe)")):
        weak = sorted(i for i in instances if (base, i) in means.index and means[(base, i)] >= WEAK_P)
        res = dict(label=label, weak_set=weak, n_baseline=int(sum(1 for i in instances if (base, i) in means.index)))
        for arm in A9_ARMS:
            c = compare(long, arm, base)
            res[arm] = dict(all=dict(n=c["n_instances"], excluded=c["excluded"], better=c["better"], worse=c["worse"],
                                     mean_d=c["mean_d"], ci=(c["ci_lo"], c["ci_hi"]), p=c["p"], method=c["method"],
                                     mean_P_before=c["mean_p_ctrl"], mean_P_after=c["mean_p_arm"]),
                            weak=_weak_block(long, arm, base, weak))
        out[base] = res
    return out


# ----------------------------------------------------------------------------------------------- the report
def reading(runs: pd.DataFrame, root: Path, sets, instances, keep_failed: bool, info_cache: dict) -> dict:
    df, dropped = select(runs, sets, ARMS, keep_failed=keep_failed)
    fr = runs[runs["set"].isin(list(sets)) & runs["arm"].isin(list(ARMS)) & (runs["status"] == "failed")]
    failed = [dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"]), run_id=r["run_id"])
              for r in fr.to_dict("records")]
    for r in df.to_dict("records"):
        if r["run_id"] not in info_cache:
            info_cache[r["run_id"]] = run_info(r, root)
    a0, a0_dropped = select(runs, A0_SETS, (A0,))
    # failed runs: under `exclude` they are missing (and listed); under `keep` they are scored, and validity's
    # "completed" leg fails by the letter (a stated deviation, named in the verdict)
    val = validity(df, info_cache, instances, void_log(df, sets, root), failed)
    out = outcome(df, instances)
    if not val["hard_ok"]:
        legs = [k for k in ("compiled", "block_mismatch", "gaps", "no_run_record") if val[k]]
        verdict = "INVALID (validity fails: " + ", ".join(legs) + "); the outcome below is not decided"
    elif val["missing"] or out["missing"]:
        verdict = f"INCOMPLETE: {len(val['missing'])} missing runs (first: {val['missing'][:3]})"
    elif val["voids_without_rerun"]:
        verdict = f"INCOMPLETE: voids without a completed rerun {val['voids_without_rerun'][:2]}"
    else:
        verdict = out["verdict"]
        if keep_failed and failed:
            verdict += (f" [{len(failed)} failed runs kept as D-002 §6 keeps crashes — validity's 'completed' leg "
                        "fails by the letter; a stated deviation]")
    return dict(reading="keep" if keep_failed else "exclude", verdict=verdict, validity=val,
                mechanism=mechanism(df, info_cache), outcome=out, c1=c1_interim(df, a0, instances),
                duplicates_left_out=dropped + a0_dropped, n_runs=int(len(df)), n_a0=int(len(a0)),
                failed_runs=failed)


def report(runs: pd.DataFrame, root: Path = ROOT, sets=(SET,), instances=None, failed: str = "both") -> dict:
    """D-022 under the readings of failed runs: "exclude", "keep" or "both" (both only when the sets have failed
    runs; otherwise the readings coincide and one is returned)."""
    instances = list(instances) if instances is not None else development_instances()
    cache: dict = {}
    has_failed = bool(((runs["set"].isin(list(sets))) & runs["arm"].isin(list(ARMS))
                       & (runs["status"] == "failed")).any())
    modes = {"exclude": [False], "keep": [True], "both": [False, True] if has_failed else [False]}[failed]
    return dict(sets=list(sets), instances=instances, readings=[reading(runs, root, sets, instances, k, cache)
                                                                for k in modes])


# ----------------------------------------------------------------------------------------------- printing
def _fmt(v, nd=3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (float, np.floating)):
        return "—" if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def print_report(res: dict, out=sys.stdout) -> None:
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"D-022 — A9 under the external harness (R13 + R14): {TREAT} (ext) vs {CTRL} (ext without R13 / R14), "
      f"same-day {A6F}; sets {res['sets']}, {len(res['instances'])} instances, seeds {list(SEEDS)}, {BUDGET:g} s")
    for rd in res["readings"]:
        v = rd["validity"]
        p("")
        p(f"=== reading '{rd['reading']}' (failed runs " + ("kept as D-002 §6 keeps crashes" if rd["reading"] == "keep"
                                                            else "missing") + f"): {rd['n_runs']} runs")
        p(f"validity: {'holds' if v['ok'] else 'FAILS'} — missing {len(v['missing'])}"
          + (f" {v['missing'][:4]}" if v["missing"] else "") + f"; failed {len(v['failed'])}"
          + (f" {[(f['instance'], f['arm'], f['seed']) for f in v['failed']]}" if v["failed"] else "")
          + f"; compile on the clock {len(v['compiled'])}" + (f" {v['compiled'][:3]}" if v["compiled"] else "")
          + f"; P / gene dtype mismatches {len(v['block_mismatch'])}"
          + (f" {v['block_mismatch'][:3]}" if v["block_mismatch"] else "")
          + f"; gap settings wrong {len(v['gaps'])}"
          + (f" {[(g['instance'], g['arm'], g['seed'], g['want']) for g in v['gaps'][:4]]}" if v["gaps"] else "")
          + f"; voids {len(v['voids'])} (without a completed rerun {len(v['voids_without_rerun'])})")
        p(f"R14 (reported): thread variables not as the harness says on {len(v['r14_env'])} runs"
          + (f" {[(e['instance'], e['arm'], e['seed']) for e in v['r14_env'][:4]]}" if v["r14_env"] else ""))
        for arm, m in rd["mechanism"].items():
            c = m["cpu_per_wall"]
            p(f"mechanism {arm:<9}: {len(m['ended_by_cuopt_optimal'])} of {m['n']} runs ended by cuOpt's 'optimal' "
              f"(cuOpt first), {len(m['with_final_gap'])} of them with a final gap > {GAP_TOL:g} "
              f"{[(x['instance'], x['seed'], _fmt(x['final_gap'], 5)) for x in m['with_final_gap']]}; cuOpt CPU-s "
              f"per wall-s median {_fmt(c['median'], 2)} [{_fmt(c['min'], 2)}, {_fmt(c['max'], 2)}] (n {c['n']})")
        o = rd["outcome"]
        p("instance                       P ctrl  P treat P A6F   d       d s0    d s1    band  worse")
        for r in o["rows"]:
            if not _fin(r.get("d")):
                p(f"{r['instance']:<30} (missing)")
                continue
            p(f"{r['instance']:<30} {_fmt(r['p_ctrl']):<7} {_fmt(r['p_treat']):<7} {_fmt(r['p_a6f']):<7} "
              f"{_fmt(r['d']):<7} {_fmt(r['d_seed'][0]):<7} {_fmt(r['d_seed'][1]):<7} {_fmt(r['band'], 2):<5} "
              + ("both seeds" if r["worse_both"] else ("one seed" if r["worse_one"] else "")))
        p(f"mean d over the {len(o['rows'])}: {_fmt(o['mean_d'])} (adopt >= {ADOPT_MEAN}, reject <= {REJECT_MEAN})")
        p("C1 interim (reported, not criteria; D-013 terms):")
        for base, c in rd["c1"].items():
            p(f"  vs {c['label']}: weak set (P >= {WEAK_P}) {c['weak_set']} ({len(c['weak_set'])} of "
              f"{c['n_baseline']} with a baseline)")
            for arm in A9_ARMS:
                a, w = c[arm]["all"], c[arm]["weak"]
                p(f"    {arm:<8} all: {a['better']} better / {a['worse']} worse of {a['n']}, mean d "
                  f"{_fmt(a['mean_d'])} [{_fmt(a['ci'][0])}, {_fmt(a['ci'][1])}], Wilcoxon p {_fmt(a['p'], 4)}"
                  + (f" (both P = 1, left out: {a['excluded']})" if a["excluded"] else ""))
                if w.get("n"):
                    p(f"    {'':<8} weak set: improved {w['better']} of {w['n']}, mean P {_fmt(w['mean_P_before'])} -> "
                      f"{_fmt(w['mean_P_after'])} ({_fmt(-100 * w['P_reduction'], 0)} %), mean d {_fmt(w['mean_d'])} "
                      f"[{_fmt(w['ci'][0])}, {_fmt(w['ci'][1])}], Wilcoxon p {_fmt(w['p'], 4)}, worst P after "
                      f"{_fmt(w['worst_P_after'])}")
        p("  caveats (D-022): cuOpt's weak set is selected on the same runs it is compared with; the A0 runs are from "
          "earlier days (sets dev / classprobe), not same-day")
        if rd["duplicates_left_out"]:
            p(f"note: the same (instance, arm, seed) in more than one selected set — kept the bare run id, left out "
              f"{[(d['instance'], d['arm'], d['seed'], d['run_id']) for d in rd['duplicates_left_out'][:4]]}")
        p(f"VERDICT ({rd['reading']}): {rd['verdict']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools/d022_report.py", description="D-022: A9 under the external harness "
                                 "(R13 + R14), the rules applied; C1's interim A9 result")
    ap.add_argument("--set", dest="sets", default=SET, help="comma-separated run sets (default itx-r13)")
    ap.add_argument("--runs", default=str(DATASETS / "runs.parquet"))
    ap.add_argument("--root", default=str(ROOT), help="root of the run_dir paths in the run table")
    ap.add_argument("--failed", choices=("both", "exclude", "keep"), default="both",
                    help="failed runs: both readings (default), missing (the letter) or kept as D-002 §6 crashes")
    ap.add_argument("--json", default=None, help="also write the result as JSON here")
    a = ap.parse_args(argv)
    if not Path(a.runs).exists():
        print(f"{a.runs} does not exist (run `itrea build-runs` first)", file=sys.stderr)
        return 2
    res = report(pd.read_parquet(a.runs), Path(a.root), tuple(s for s in a.sets.split(",") if s), failed=a.failed)
    print_report(res)
    if a.json:
        Path(a.json).write_text(json.dumps(_jsonable(res), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
