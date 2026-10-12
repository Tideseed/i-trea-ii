"""D-022 report: A9 under the external harness — cuOpt's MIP gaps 0 and its thread variables unset (D-021 R13 + R14),
development test (D-010) — and C1's interim A9 result on the 17 instances (D-019 addendum). The rules are D-022's,
fixed before any run (with its clarifications of 2026-10-12, also before any run); nothing is chosen by eye.

Usage (repository root, after `itrea build-runs`; untimed analysis: OMP_NUM_THREADS=1, never beside timed runs with
default thread counts):

    OMP_NUM_THREADS=1 .venv/bin/python tools/d022_report.py [--set itx-r13] [--failed both|exclude|keep]
                                                            [--followup [itx-r13f]] [--json out]
    itrea d022-report …                       (the same, through the entry point)

Reads `datasets/runs.parquet` (P(120) = `pt`, `final_gap`, `status`, `attempt`, `run_dir`, …) and, through each run's
`run_dir` (relative to `--root`), its `run.json` (the cuOpt process record, the compile-on-the-clock flags, the
resolved harness `config.harness`; the run table's `harness` column when run.json has none), `cuopt.summary.json` (the
solve's `settings`, `exit_reason`) and `ea.summary.json` (`P`, `gene_dtype`), and each set's void log
`<raw>/<set>/void.jsonl` (the pool's record of voided runs, moved aside and not in the table). The checked-incumbents
table is not needed: every number here is a run-table column or a run-directory field.

Arms (`jobs/itx-r13.jsonl`, set itx-r13, 120 s, seeds 0–1, the 17 development + class-probe instances): control A9@r13
(A9 under the harness {ext, cuopt_gap_zero false, cuopt_env_unset false}), treatment A9@r13x (A9 under ext), same-day
A6F@r13x (cuOpt alone on F, ext). A0 for the C1 block: the runs of sets `dev` / `classprobe`, 120 s, seeds 0–1.

A reading reads only its own runs: the 17 instances x the three arms x its seeds (0–1; the follow-up's 2–3) x 120 s,
not synthetic, not in a voided block. Its failed runs and its voids are of those keys only (a void's key comes from the
run table's row of its run id — the rerun —, else from the voided directory's own run.json, written at the run's
start; a void whose key cannot be read is counted).

Validity (must hold; else the verdict is INVALID):
  - no compile on the clock: run.json `compiled_on_clock_cuda` (the project's CUDA cache, every run) and, on the arms
    with an EA GPU process (A9), the CuPy cache (run.json `cupy_cache.compiled_on_clock`, the EA summary's
    `compiled_on_clock`) — the runner's own rule (`pair.void_checks`), so A6F is not charged with a CuPy flag the
    runner never applies to it;
  - the harness: every run's resolved harness is its arm's — `pair.resolve_harness("ext")` for A9@r13x and A6F@r13x,
    `resolve_harness(make_jobs.ITX_R13_CONTROL)` for A9@r13 (RMM 24 / host 24, the project CUDA cache and D-021's
    voids shared; only R13 + R14 differ) — a run without one fails the leg;
  - every run completed or a recorded void rerun: every void of the reading's keys has its rerun (the same run id)
    completed in the table; a missing run makes the verdict INCOMPLETE; a FAILED run breaks this leg by the letter —
    reading `keep` counts it as D-002 §6 keeps crashes (truncated at the crash, as build-runs scores it), a stated
    deviation that the report names; under `keep` a void whose rerun failed is closed by that kept crash;
  - equal P and gene dtype per block (the two A9 runs of an instance x seed, from their EA summaries);
  - cuOpt's gaps 0 in every treatment solve and default in every control solve: the cuOpt summary's `settings`
    carries mip_absolute_gap = mip_relative_gap = 0 for A9@r13x and A6F@r13x and neither key for A9@r13 (cuOpt's
    default, relative 1e-4); a run without cuOpt settings fails the leg.
  Under `keep`, a kept crash that died before writing its EA or cuOpt summary (a SIGKILLed process writes none) is
  listed as "crashed before the record" instead of failing the P / gene-dtype or the gap leg; its harness is checked.
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
max(a, b, 1e-3) of the control or of A6F@r13x); the mean over the 17 counts every instance, both-P = 1 ones at d = 0;
a run without P(120) on any of the three arms makes the verdict INCOMPLETE and is named):
  - REJECT (A9 keeps the default gaps; D-021's A9 rows run under {ext, gap_zero false, env_unset false}) if the mean
    d <= −0.05 or any instance has d_s < −band_i on both seeds;
  - ADOPT R13 + R14 for A9 (the C1 freeze carries harness ext) if the mean d >= −0.02 and no instance is worse than
    −band_i on both seeds;
  - otherwise INCONCLUSIVE (−0.05 < mean d < −0.02, no instance worse on both seeds) -> seeds 2–3 on the instances
    worse than −band_i on exactly one seed (all 17 when none is), same rules (the report names them).

The follow-up (`--followup [SET]`, D-022's clarification of 2026-10-12, fixed before any run): applies only when the
seeds 0–1 verdict is INCONCLUSIVE. Runs: the named instances (or all 17) x seeds 2–3 x the same three arms and
harnesses, 120 s, in their own set `itx-r13f` (`itrea make-jobs itx-r13f`). Validity as above, on those runs. band_i
stays seeds 0–1's; a rerun instance's d is the bounded d on its means over seeds 0–3 and it is "worse on both seeds"
when d_s < −band_i on both seeds 2 and 3; the others keep their seeds-0–1 d; the mean is over the 17. REJECT if the
mean d <= −0.05 or a rerun instance is worse on both seeds 2 and 3; ADOPT if the mean d >= −0.02; otherwise NOT
ADOPTED — non-inferiority not shown, the reject consequence, no further seeds (the terminal rule the record left
open, stated before any run and open to Berk).

C1's interim block (reported, not criteria; D-013 terms): A9@r13x and A9@r13 against A6F@r13x (equal footprint,
same day) and against SCIP alone (A0 of `dev` / `classprobe`) — all instances: better / worse, mean d with its 95 %
bootstrap interval and the Wilcoxon test (`itrea.analysis.stats.compare`, which leaves out instances with P = 1 in
both arms on every seed; the left-out ones are printed); each solver's weak set (A6F@r13x P >= 0.5, A0 P >= 0.5, seed
means), read as `backoff.d013_report` and D-013's own table read it — over EVERY weak instance with both arms' runs,
both-P = 1 ones included (d = 0): improved k of |W_B|, mean P before -> after (and its relative reduction), mean d with
its bootstrap interval (`stats.bootstrap_mean` on the same d's) and the Wilcoxon test (zeros dropped), worst P after
coupling, and the weak instances `compare` would leave out. Caveats printed with it: cuOpt's weak set is selected on
the same runs it is compared with; the A0 runs are from earlier days.

Failed runs: both readings (`--failed both`, the default; produced when the readings' own runs include a failed one)
— `exclude` (the letter: a failed run is missing, the verdict INCOMPLETE) and `keep` (D-002 §6). Run identity: a job
run again in another set keeps its run id and the run table qualifies the later copy as `<run_id>#<set>`
(build_runs.qualify_reused_ids). Runs are therefore selected by the `set` column and keyed by (instance, arm, seed) —
the run id is never parsed —, the directory is `run_dir`; when one key occurs in more than one selected set the bare
id is kept (then the highest attempt) and the others are listed.
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
from itrea.analysis.stats import bootstrap_mean, compare, effect_d, wilcoxon_d
from itrea.paths import DATASETS, ROOT

CTRL, TREAT, A6F, A0 = "A9@r13", "A9@r13x", "A6F@r13x", "A0"
ARMS = (CTRL, TREAT, A6F)
A9_ARMS = (CTRL, TREAT)
GAP_ZERO_ARMS = (TREAT, A6F)              # every cuOpt solve under the ext harness
SET, SEEDS, BUDGET = "itx-r13", (0, 1), 120.0
FOLLOWUP_SET, FOLLOWUP_SEEDS = "itx-r13f", (2, 3)   # D-022's INCONCLUSIVE follow-up (clarification 2026-10-12)
A0_SETS = ("dev", "classprobe")
BAND_MIN, ADOPT_MEAN, REJECT_MEAN = 0.05, -0.02, -0.05
GAP_TOL = 1e-9
WEAK_P = 0.5
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
REUSE_SEP = "#"
REJECT_TAIL = ("A9 keeps cuOpt's default gaps; D-021's A9 rows run under {ext, cuopt_gap_zero false, "
               "cuopt_env_unset false}")


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


def expected_harness() -> dict:
    """{arm: the resolved harness its runs must carry} (D-022): ext for the treatment and A6F; {ext, cuopt_gap_zero
    false, cuopt_env_unset false} for the control — `pair.resolve_harness` of make_jobs' own specs, so the report and
    the job file cannot drift apart."""
    from itrea.experiments.make_jobs import ITX_EXT, ITX_R13_CONTROL
    from itrea.experiments.pair import resolve_harness
    ext = resolve_harness(ITX_EXT)
    return {CTRL: resolve_harness(ITX_R13_CONTROL), TREAT: ext, A6F: ext}


# ----------------------------------------------------------------------------------------------- selection
def _candidates(runs: pd.DataFrame, sets, arms, seeds, budget: float, instances=None) -> pd.DataFrame:
    """Every row of `sets` x `arms` x `seeds` at `budget` (and of `instances`), not synthetic, not in a voided block,
    whatever its status — the one filter `select` and the failed-run list share."""
    df = runs[runs["set"].isin(list(sets)) & runs["arm"].isin(list(arms)) & runs["seed"].isin(list(seeds))]
    if instances is not None:
        df = df[df["instance"].isin(list(instances))]
    if "budget" in df:
        df = df[np.isclose(df["budget"].astype(float), float(budget))]
    for col in ("synthetic", "block_void"):
        if col in df:
            df = df[~df[col].fillna(False).astype(bool)]
    return df


def select(runs: pd.DataFrame, sets, arms, seeds=SEEDS, budget: float = BUDGET, keep_failed: bool = False,
           instances=None) -> tuple[pd.DataFrame, list[dict]]:
    """The runs of `sets` x `arms` x `seeds` at `budget` (of `instances`, when given), completed (and failed with
    `keep_failed`), not synthetic, not in a voided block; one row per (instance, arm, seed): the bare run id before a
    `#<set>` copy, then the highest attempt. Returns (rows, the duplicates left out)."""
    df = _candidates(runs, sets, arms, seeds, budget, instances)
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


def failed_runs(runs: pd.DataFrame, sets, seeds, instances, budget: float = BUDGET) -> list[dict]:
    """The reading's failed runs: of the rows `select` takes under `keep` — the same sets, arms, seeds, budget and
    instances, not synthetic, not in a voided block, one per key — those whose status is failed."""
    df, _ = select(runs, sets, ARMS, seeds, budget, keep_failed=True, instances=instances)
    return [dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"]), run_id=r["run_id"], set=r["set"])
            for r in df[df["status"] == "failed"].to_dict("records")]


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
    if harness is None and isinstance(row.get("harness"), str) and row["harness"]:
        try:                                         # the run table's copy (JSON; "" = no harness)
            harness = json.loads(row["harness"])
        except ValueError:
            harness = None
    return dict(dir=str(d), base_arm=base, compiled=compiled, settings=st, gaps=gaps,
                gap_zero=bool(gaps is not None and gaps[0] == 0.0 and gaps[1] == 0.0),
                gap_default=bool(st is not None and "mip_absolute_gap" not in st and "mip_relative_gap" not in st),
                env_unset=sorted(cu.get("env_unset") or []), cuopt_exit=cs.get("exit_reason"), cuopt_exit_t=t_cu,
                cuopt_first=bool(_fin(t_cu) and all(float(t_cu) <= float(t) for t in others)),
                cpu_per_wall=cpu / float(t_cu) if _fin(cpu) and _fin(t_cu) and float(t_cu) > 0 else math.nan,
                P=es.get("P"), gene_dtype=es.get("gene_dtype"), harness=harness, run_json=bool(rj))


def _fill_info(df: pd.DataFrame, root: Path, cache: dict) -> None:
    for r in df.to_dict("records"):
        if r["run_id"] not in cache:
            cache[r["run_id"]] = run_info(r, root)


def void_log(runs: pd.DataFrame, sets, root: Path) -> list[dict]:
    """The pool's void records (`<raw>/<set>/void.jsonl`) of `sets` — the directories next to the sets' run
    directories and `<root>/traces/raw/<set>` —, each with `key` = (instance, arm, seed, budget) of its run: from the
    run table's row of the same run id in the set (the rerun), else from the voided directory's own run.json
    (`moved_to`; the runner writes it at the run's start: name, arm label, seed, budget); None when neither is
    readable (the reading then counts the void: it cannot show that the void is not its own)."""
    out, seen = [], set()
    for s in sets:
        rows = runs[runs["set"] == s]
        by_id: dict = {}
        for r in rows.to_dict("records"):
            by_id.setdefault(str(r["run_id"]).split(REUSE_SEP)[0], r)
        dirs = {root / "traces" / "raw" / s}
        for rd in rows["run_dir"].astype(str):
            if rd and rd not in ("nan", "None"):
                p = Path(rd)
                dirs.add((p if p.is_absolute() else root / p).parent)
        for d in sorted(dirs):
            f = d / "void.jsonl"
            if f in seen or not f.exists():
                continue
            seen.add(f)
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    v = dict(json.loads(line), set=s)
                except ValueError:
                    out.append(dict(set=s, unreadable=line[:200], key=None))
                    continue
                r = by_id.get(str(v.get("run_id", "")))
                if r is not None:
                    b = r.get("budget")
                    v["key"] = (r["instance"], r["arm"], int(r["seed"]), float(b) if _fin(b) else math.nan)
                else:
                    rj = _read(d / str(v["moved_to"]) / "run.json") if v.get("moved_to") else {}
                    ok = all(rj.get(k) is not None for k in ("name", "arm", "seed")) and _fin(rj.get("budget"))
                    v["key"] = (str(rj["name"]), str(rj["arm"]), int(rj["seed"]), float(rj["budget"])) if ok else None
                out.append(v)
    return out


# ----------------------------------------------------------------------------------------------- the rules
def validity(df: pd.DataFrame, info: dict, instances, voids: list[dict], failed: list[dict], seeds=SEEDS,
             keep_failed: bool = False, budget: float = BUDGET) -> dict:
    """D-022's validity legs (module docstring) on the reading's runs `df` (instances x ARMS x `seeds`). `info`:
    {run_id: run_info}; `voids`: `void_log`'s records (only those of the reading's keys count); `failed`: the
    reading's failed runs (counted under the reading `keep`, missing under `exclude`)."""
    have = {(r["instance"], r["arm"], int(r["seed"])): r for r in df.to_dict("records")}
    want = {(i, a, int(s)) for i in instances for a in ARMS for s in seeds}
    missing = [(i, a, s) for i in instances for a in ARMS for s in seeds if (i, a, s) not in have]
    compiled = [dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"]), what=info[r["run_id"]]["compiled"])
                for r in have.values() if info[r["run_id"]]["compiled"]]
    no_record = [dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"])) for r in have.values()
                 if not info[r["run_id"]]["run_json"]]
    crashed: list[dict] = []          # kept crashes that died before their summary (reading `keep` only)

    def kept_crash(r) -> bool:
        return keep_failed and str(r["status"]) == "failed"

    # voids: the reading's own keys; closed by a completed rerun (and, under keep, by a kept failed one)
    closing = {"completed", "failed"} if keep_failed else {"completed"}
    status = {str(r["run_id"]).split(REUSE_SEP)[0]: str(r["status"]) for r in have.values()}
    void_rows, outside = [], 0
    for v in voids:
        k = v.get("key")
        if k is not None:
            mine = (k[0], k[1], int(k[2])) in want and (not _fin(k[3]) or math.isclose(float(k[3]), float(budget)))
            if not mine:
                outside += 1
                continue
        st = status.get(str(v.get("run_id", "")))
        void_rows.append(dict(v, rerun_status=st, rerun_completed=st == "completed", closed=st in closing))
    voids_open = [v for v in void_rows if not v["closed"]]
    voids_crash = [v for v in void_rows if v["closed"] and v["rerun_status"] == "failed"]
    block = []
    for i in instances:
        for s in seeds:
            a, b = have.get((i, CTRL, s)), have.get((i, TREAT, s))
            if a is None or b is None:
                continue
            ia, ib = info[a["run_id"]], info[b["run_id"]]
            unrecorded = [(r, x) for r, x in ((a, ia), (b, ib)) if x["P"] is None or x["gene_dtype"] is None]
            if unrecorded and all(kept_crash(r) for r, _ in unrecorded):
                crashed += [dict(instance=i, arm=r["arm"], seed=s, missing="EA summary (P, gene dtype)")
                            for r, _ in unrecorded]
                continue                      # the block cannot be compared: listed, not a mismatch
            if ia["P"] is None or ib["P"] is None or ia["P"] != ib["P"] or ia["gene_dtype"] != ib["gene_dtype"]:
                block.append(dict(instance=i, seed=s, P=(ia["P"], ib["P"]), gene_dtype=(ia["gene_dtype"],
                                                                                     ib["gene_dtype"])))
    gaps = []
    for r in have.values():
        x = info[r["run_id"]]
        if x["settings"] is None and kept_crash(r):
            crashed.append(dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"]),
                                missing="cuOpt summary (settings)"))
            continue
        want_gap = "0" if r["arm"] in GAP_ZERO_ARMS else "default"
        ok = x["gap_zero"] if want_gap == "0" else x["gap_default"]
        if not ok:
            gaps.append(dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"]), want=want_gap,
                             settings=x["settings"]))
    exp = expected_harness()
    harness = [dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"]), harness=info[r["run_id"]]["harness"])
               for r in have.values() if info[r["run_id"]]["harness"] != exp[r["arm"]]]
    env = []
    for r in have.values():
        x = info[r["run_id"]]
        want_unset = r["arm"] in GAP_ZERO_ARMS
        if (set(THREAD_VARS) <= set(x["env_unset"])) != want_unset or (not want_unset and x["env_unset"]):
            env.append(dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"]), env_unset=x["env_unset"]))
    hard_ok = not compiled and not block and not gaps and not harness and not no_record
    return dict(ok=bool(hard_ok and not missing and not failed and not voids_open), hard_ok=bool(hard_ok),
                missing=missing, failed=failed, compiled=compiled, no_run_record=no_record, voids=void_rows,
                voids_without_rerun=voids_open, voids_closed_by_kept_crash=voids_crash, voids_outside_reading=outside,
                block_mismatch=block, gaps=gaps, harness=harness, crashed_before_record=crashed, r14_env=env)


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


def _p_table(df: pd.DataFrame) -> tuple[dict, list]:
    """({(instance, arm, seed): P(120)} of the rows with a finite pt, [(instance, arm, seed) of the rows without])."""
    P, no_p = {}, []
    for r in df.to_dict("records"):
        k = (r["instance"], r["arm"], int(r["seed"]))
        if _fin(r.get("pt")):
            P[k] = float(r["pt"])
        else:
            no_p.append(k)
    return P, sorted(no_p)


def outcome(df: pd.DataFrame, instances, seeds=SEEDS) -> dict:
    """D-022's outcome rule (module docstring) on the seed pair `seeds`."""
    P, no_p = _p_table(df)
    rows, missing = [], []
    for i in instances:
        pc = [P.get((i, CTRL, s)) for s in seeds]
        pt = [P.get((i, TREAT, s)) for s in seeds]
        pa = [P.get((i, A6F, s)) for s in seeds]
        if any(v is None for v in pc + pt + pa):          # band_i needs A6F's pair as well
            missing.append(i)
            rows.append(dict(instance=i, d=math.nan))
            continue
        ds = [effect_d(c, t) for c, t in zip(pc, pt)]
        band = max([BAND_MIN] + [v for v in (bounded(pc[0], pc[1]), bounded(pa[0], pa[1])) if _fin(v)])
        worse = [d < -band for d in ds]
        rows.append(dict(instance=i, p_ctrl=float(np.mean(pc)), p_treat=float(np.mean(pt)), p_a6f=float(np.mean(pa)),
                         pc_seeds=pc, pt_seeds=pt, pa_seeds=pa,
                         d=effect_d(float(np.mean(pc)), float(np.mean(pt))), d_seed=ds, band=band,
                         worse_both=bool(all(worse)), worse_one=bool(sum(worse) == 1)))
    if missing:
        return dict(kind="INCOMPLETE", verdict=f"INCOMPLETE: no seed pair of {CTRL}, {TREAT} and {A6F} on {missing}",
                    rows=rows, mean_d=math.nan, worse_both=[], worse_one=[], missing=missing, no_p=no_p)
    mean_d = float(np.mean([r["d"] for r in rows]))
    wb = [r["instance"] for r in rows if r["worse_both"]]
    wo = [r["instance"] for r in rows if r["worse_one"]]
    if mean_d <= REJECT_MEAN or wb:
        why = ([f"mean d {mean_d:+.3f} <= {REJECT_MEAN}"] if mean_d <= REJECT_MEAN else []) + \
              ([f"worse than −band on both seeds: {wb}"] if wb else [])
        kind, verdict = "REJECT", "REJECT R13 + R14 for A9 (" + "; ".join(why) + "): " + REJECT_TAIL
    elif mean_d >= ADOPT_MEAN:
        kind, verdict = "ADOPT", (f"ADOPT R13 + R14 for A9 (mean d {mean_d:+.3f} >= {ADOPT_MEAN}, no instance worse "
                                  "than −band on both seeds): the C1 freeze carries harness ext")
    else:
        kind = "INCONCLUSIVE"
        verdict = (f"INCONCLUSIVE (mean d {mean_d:+.3f} in ({REJECT_MEAN}, {ADOPT_MEAN}), no instance worse on both "
                   f"seeds): seeds 2–3 in set {FOLLOWUP_SET}, same rules, on the instances worse than −band on one "
                   "seed — " + (", ".join(wo) if wo else "none is, so all 17 (D-022's clarification of 2026-10-12)"))
    return dict(kind=kind, verdict=verdict, rows=rows, mean_d=mean_d, worse_both=wb, worse_one=wo, missing=[],
                no_p=no_p)


def followup_instances(base: dict, instances) -> list[str]:
    """The instances the follow-up reruns: those worse than −band_i on exactly one seed, or all when none is."""
    return list(base["worse_one"]) or list(instances)


def outcome_followup(df: pd.DataFrame, base: dict, rerun) -> dict:
    """The follow-up reading (D-022's clarification of 2026-10-12; module docstring): `df` holds the rerun instances'
    seeds 2–3, `base` is the seeds 0–1 outcome (INCONCLUSIVE)."""
    P, no_p = _p_table(df)
    rows, missing = [], []
    for r in base["rows"]:
        i = r["instance"]
        if i not in rerun:
            rows.append(dict(r, rerun=False))
            continue
        pc = [P.get((i, CTRL, s)) for s in FOLLOWUP_SEEDS]
        pt = [P.get((i, TREAT, s)) for s in FOLLOWUP_SEEDS]
        pa = [P.get((i, A6F, s)) for s in FOLLOWUP_SEEDS]
        if any(v is None for v in pc + pt + pa):
            missing.append(i)
            rows.append(dict(instance=i, d=math.nan, rerun=True))
            continue
        ds = [effect_d(c, t) for c, t in zip(pc, pt)]
        worse = [d < -r["band"] for d in ds]
        mc, mt = float(np.mean(r["pc_seeds"] + pc)), float(np.mean(r["pt_seeds"] + pt))
        rows.append(dict(instance=i, rerun=True, p_ctrl=mc, p_treat=mt, p_a6f=float(np.mean(r["pa_seeds"] + pa)),
                         d=effect_d(mc, mt), d_seed=list(r["d_seed"]) + ds, band=r["band"],
                         worse_both=bool(all(worse)), worse_one=bool(sum(worse) == 1)))
    if missing:
        return dict(kind="INCOMPLETE", verdict=f"INCOMPLETE: no seeds 2–3 of the three arms on {missing}", rows=rows,
                    mean_d=math.nan, worse_both=[], missing=missing, no_p=no_p)
    mean_d = float(np.mean([r["d"] for r in rows]))
    wb = [r["instance"] for r in rows if r["rerun"] and r["worse_both"]]
    if mean_d <= REJECT_MEAN or wb:
        why = ([f"mean d {mean_d:+.3f} <= {REJECT_MEAN}"] if mean_d <= REJECT_MEAN else []) + \
              ([f"worse than −band on both seeds 2 and 3: {wb}"] if wb else [])
        kind, verdict = "REJECT", "REJECT R13 + R14 for A9 after seeds 2–3 (" + "; ".join(why) + "): " + REJECT_TAIL
    elif mean_d >= ADOPT_MEAN:
        kind, verdict = "ADOPT", (f"ADOPT R13 + R14 for A9 after seeds 2–3 (mean d {mean_d:+.3f} >= {ADOPT_MEAN}, no "
                                  "rerun instance worse than −band on both seeds 2 and 3): the C1 freeze carries "
                                  "harness ext")
    else:
        kind, verdict = "NOT ADOPTED", (f"NOT ADOPTED: still inconclusive after seeds 2–3 (mean d {mean_d:+.3f} in "
                                        f"({REJECT_MEAN}, {ADOPT_MEAN})) — non-inferiority not shown, the reject "
                                        "consequence (D-022's terminal rule of 2026-10-12, open to Berk): " +
                                        REJECT_TAIL)
    return dict(kind=kind, verdict=verdict, rows=rows, mean_d=mean_d, worse_both=wb, missing=[], no_p=no_p)


def _weak_block(long: pd.DataFrame, arm: str, base: str, weak: list[str]) -> dict:
    """D-013 terms on the weak set W_B, read as `backoff.d013_report` and D-013's own table read it: over EVERY weak
    instance with runs of both the arm and the baseline — both-P = 1 ones included (d = 0; D-013 counts rail01
    1.0 -> 1.0) — improved k of |W_B|, mean P before -> after and its relative reduction, mean d with its 95 %
    bootstrap interval (`stats.bootstrap_mean` on the same d's, compare's resampling) and the Wilcoxon test (zeros
    dropped, as in compare), worst P after coupling; `left_out_by_compare` = the weak instances `stats.compare` would
    drop (both P = 1 on every seed)."""
    sub = long[long["instance"].isin(weak)]
    pa = sub[sub["arm"] == arm].groupby("instance")["pt"].mean()
    pb = sub[sub["arm"] == base].groupby("instance")["pt"].mean()
    w = [i for i in weak if i in pa.index and i in pb.index]
    if not w:
        return dict(n=0, n_weak=len(weak))
    ds = np.array([effect_d(float(pb[i]), float(pa[i])) for i in w])
    b, wx = bootstrap_mean(ds), wilcoxon_d(ds)
    before, after = float(np.mean([pb[i] for i in w])), float(np.mean([pa[i] for i in w]))
    return dict(n=len(w), n_weak=len(weak), better=int(sum(pa[i] < pb[i] for i in w)),
                worse=int(sum(pa[i] > pb[i] for i in w)), mean_P_before=before, mean_P_after=after,
                P_reduction=(before - after) / before if before else math.nan, mean_d=b["mean"], ci=(b["lo"], b["hi"]),
                p=wx["p"], method=wx["method"], worst_P_after=float(max(pa[i] for i in w)),
                left_out_by_compare=compare(sub, arm, base)["excluded"])


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
def _verdict(val: dict, out: dict, keep_failed: bool, failed: list[dict]) -> str:
    """INVALID before INCOMPLETE before the outcome; under `keep`, what was kept is named."""
    if not val["hard_ok"]:
        legs = [k for k in ("compiled", "harness", "block_mismatch", "gaps", "no_run_record") if val[k]]
        return "INVALID (validity fails: " + ", ".join(legs) + "); the outcome below is not decided"
    if val["missing"]:
        return f"INCOMPLETE: {len(val['missing'])} missing runs (first: {val['missing'][:3]})"
    if out["missing"]:
        return (f"INCOMPLETE: no P(120) for {len(out['no_p'])} runs {out['no_p'][:4]} (instances {out['missing']}; the "
                "run table has them, without a finite pt)")
    if val["voids_without_rerun"]:
        return "INCOMPLETE: voids without a rerun in the reading " + str(
            [(v.get("run_id"), v.get("reason")) for v in val["voids_without_rerun"][:2]])
    verdict = out["verdict"]
    if keep_failed:
        notes = []
        if failed:
            notes.append(f"{len(failed)} failed runs kept as D-002 §6 keeps crashes — validity's 'completed' leg fails "
                         "by the letter; a stated deviation")
        if val["voids_closed_by_kept_crash"]:
            notes.append(f"{len(val['voids_closed_by_kept_crash'])} voids closed by a rerun that failed (kept)")
        if val["crashed_before_record"]:
            notes.append(f"{len(val['crashed_before_record'])} kept crashes died before their record (not read for "
                         "P / gene dtype or gaps): " + str([(c["instance"], c["arm"], c["seed"])
                                                           for c in val["crashed_before_record"][:4]]))
        if notes:
            verdict += " [" + "; ".join(notes) + "]"
    return verdict


def followup_reading(runs: pd.DataFrame, root: Path, sets, instances, keep_failed: bool, info_cache: dict,
                     base: dict, base_verdict: str) -> dict:
    """D-022's follow-up (module docstring) under one reading of failed runs; it applies only when the seeds 0–1
    verdict of the same reading is INCONCLUSIVE."""
    if not base_verdict.startswith("INCONCLUSIVE"):
        return dict(sets=list(sets), applies=False,
                    verdict="not applicable: the seeds 0–1 verdict is not INCONCLUSIVE")
    rerun = followup_instances(base, instances)
    df, dropped = select(runs, sets, ARMS, FOLLOWUP_SEEDS, keep_failed=keep_failed, instances=rerun)
    failed = failed_runs(runs, sets, FOLLOWUP_SEEDS, rerun)
    _fill_info(df, root, info_cache)
    val = validity(df, info_cache, rerun, void_log(runs, sets, root), failed, FOLLOWUP_SEEDS, keep_failed)
    out = outcome_followup(df, base, rerun)
    return dict(sets=list(sets), applies=True, rerun=rerun, rerun_all=not base["worse_one"],
                seeds=list(FOLLOWUP_SEEDS), validity=val, outcome=out, verdict=_verdict(val, out, keep_failed, failed),
                n_runs=int(len(df)), duplicates_left_out=dropped, failed_runs=failed)


def reading(runs: pd.DataFrame, root: Path, sets, instances, keep_failed: bool, info_cache: dict,
            followup=None) -> dict:
    df, dropped = select(runs, sets, ARMS, SEEDS, keep_failed=keep_failed, instances=instances)
    # failed runs: under `exclude` they are missing (and listed); under `keep` they are scored, and validity's
    # "completed" leg fails by the letter (a stated deviation, named in the verdict)
    failed = failed_runs(runs, sets, SEEDS, instances)
    _fill_info(df, root, info_cache)
    a0, a0_dropped = select(runs, A0_SETS, (A0,))
    val = validity(df, info_cache, instances, void_log(runs, sets, root), failed, SEEDS, keep_failed)
    out = outcome(df, instances)
    verdict = _verdict(val, out, keep_failed, failed)
    res = dict(reading="keep" if keep_failed else "exclude", verdict=verdict, validity=val,
               mechanism=mechanism(df, info_cache), outcome=out, c1=c1_interim(df, a0, instances),
               duplicates_left_out=dropped + a0_dropped, n_runs=int(len(df)), n_a0=int(len(a0)), failed_runs=failed)
    if followup:
        res["followup"] = followup_reading(runs, root, followup, instances, keep_failed, info_cache, out, verdict)
    return res


def report(runs: pd.DataFrame, root: Path = ROOT, sets=(SET,), instances=None, failed: str = "both",
           followup=None) -> dict:
    """D-022 under the readings of failed runs: "exclude", "keep" or "both" (both only when the readings' own runs —
    seeds 0–1 and, with `followup` (its sets), seeds 2–3 — include a failed one; otherwise the readings coincide and
    one is returned)."""
    instances = list(instances) if instances is not None else development_instances()
    followup = tuple(followup) if followup else None
    cache: dict = {}
    has_failed = bool(failed_runs(runs, sets, SEEDS, instances)) or bool(
        followup and failed_runs(runs, followup, FOLLOWUP_SEEDS, instances))
    modes = {"exclude": [False], "keep": [True], "both": [False, True] if has_failed else [False]}[failed]
    return dict(sets=list(sets), followup_sets=list(followup) if followup else None, instances=instances,
                readings=[reading(runs, root, sets, instances, k, cache, followup) for k in modes])


# ----------------------------------------------------------------------------------------------- printing
def _fmt(v, nd=3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (float, np.floating)):
        return "—" if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def _validity_line(v: dict) -> str:
    return (f"validity: {'holds' if v['ok'] else 'FAILS'} — missing {len(v['missing'])}"
            + (f" {v['missing'][:4]}" if v["missing"] else "") + f"; failed {len(v['failed'])}"
            + (f" {[(f['instance'], f['arm'], f['seed']) for f in v['failed']]}" if v["failed"] else "")
            + f"; compile on the clock {len(v['compiled'])}" + (f" {v['compiled'][:3]}" if v["compiled"] else "")
            + f"; harness not the arm's {len(v['harness'])}"
            + (f" {[(h['instance'], h['arm'], h['seed']) for h in v['harness'][:4]]}" if v["harness"] else "")
            + f"; P / gene dtype mismatches {len(v['block_mismatch'])}"
            + (f" {v['block_mismatch'][:3]}" if v["block_mismatch"] else "")
            + f"; gap settings wrong {len(v['gaps'])}"
            + (f" {[(g['instance'], g['arm'], g['seed'], g['want']) for g in v['gaps'][:4]]}" if v["gaps"] else "")
            + f"; voids {len(v['voids'])} (without a rerun {len(v['voids_without_rerun'])}"
            + (f", closed by a kept crash {len(v['voids_closed_by_kept_crash'])}"
               if v["voids_closed_by_kept_crash"] else "")
            + (f"; {v['voids_outside_reading']} other voids of the set are not this reading's" if
               v["voids_outside_reading"] else "") + ")"
            + (f"; crashed before the record (kept) {[(c['instance'], c['arm'], c['seed'], c['missing']) for c in v['crashed_before_record'][:4]]}"  # noqa: E501
               if v["crashed_before_record"] else ""))


def print_report(res: dict, out=sys.stdout) -> None:
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"D-022 — A9 under the external harness (R13 + R14): {TREAT} (ext) vs {CTRL} (ext without R13 / R14), "
      f"same-day {A6F}; sets {res['sets']}, {len(res['instances'])} instances, seeds {list(SEEDS)}, {BUDGET:g} s"
      + (f"; follow-up sets {res['followup_sets']}, seeds {list(FOLLOWUP_SEEDS)}" if res.get("followup_sets") else ""))
    for rd in res["readings"]:
        v = rd["validity"]
        p("")
        p(f"=== reading '{rd['reading']}' (failed runs " + ("kept as D-002 §6 keeps crashes" if rd["reading"] == "keep"
                                                            else "missing") + f"): {rd['n_runs']} runs")
        p(_validity_line(v))
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
                    p(f"    {'':<8} weak set: improved {w['better']} of {w['n']}"
                      + (f" ({w['n_weak'] - w['n']} weak without the arm's runs)" if w["n_weak"] > w["n"] else "")
                      + f", mean P {_fmt(w['mean_P_before'])} -> {_fmt(w['mean_P_after'])} "
                      f"({_fmt(-100 * w['P_reduction'], 0)} %), mean d {_fmt(w['mean_d'])} [{_fmt(w['ci'][0])}, "
                      f"{_fmt(w['ci'][1])}], Wilcoxon p {_fmt(w['p'], 4)}, worst P after {_fmt(w['worst_P_after'])}"
                      + (f"; both P = 1 (counted at d = 0; stats.compare would leave out): {w['left_out_by_compare']}"
                         if w["left_out_by_compare"] else ""))
        p("  caveats (D-022): cuOpt's weak set is selected on the same runs it is compared with; the A0 runs are from "
          "earlier days (sets dev / classprobe), not same-day")
        if rd["duplicates_left_out"]:
            p(f"note: the same (instance, arm, seed) in more than one selected set — kept the bare run id, left out "
              f"{[(d['instance'], d['arm'], d['seed'], d['run_id']) for d in rd['duplicates_left_out'][:4]]}")
        p(f"VERDICT ({rd['reading']}): {rd['verdict']}")
        f = rd.get("followup")
        if f:
            if not f["applies"]:
                p(f"FOLLOW-UP ({rd['reading']}, sets {f['sets']}): {f['verdict']}")
                continue
            p(f"--- follow-up ({rd['reading']}): seeds {f['seeds']} of sets {f['sets']} on "
              + ("all instances (the seeds 0–1 reading named none)" if f["rerun_all"] else f"{f['rerun']}")
              + f": {f['n_runs']} runs")
            p("    " + _validity_line(f["validity"]))
            fo = f["outcome"]
            p("    instance                       P ctrl  P treat d (0–3)  d s2    d s3    band  worse on s2 and s3")
            for r in fo["rows"]:
                if not r.get("rerun"):
                    continue
                if not _fin(r.get("d")):
                    p(f"    {r['instance']:<30} (missing)")
                    continue
                p(f"    {r['instance']:<30} {_fmt(r['p_ctrl']):<7} {_fmt(r['p_treat']):<7} {_fmt(r['d']):<8} "
                  f"{_fmt(r['d_seed'][2]):<7} {_fmt(r['d_seed'][3]):<7} {_fmt(r['band'], 2):<5} "
                  + ("yes" if r["worse_both"] else ""))
            p(f"    mean d over the {len(fo['rows'])} (rerun instances on seeds 0–3): {_fmt(fo['mean_d'])}")
            p(f"FOLLOW-UP VERDICT ({rd['reading']}): {f['verdict']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools/d022_report.py", description="D-022: A9 under the external harness "
                                 "(R13 + R14), the rules applied; C1's interim A9 result")
    ap.add_argument("--set", dest="sets", default=SET, help="comma-separated run sets (default itx-r13)")
    ap.add_argument("--runs", default=str(DATASETS / "runs.parquet"))
    ap.add_argument("--root", default=str(ROOT), help="root of the run_dir paths in the run table")
    ap.add_argument("--failed", choices=("both", "exclude", "keep"), default="both",
                    help="failed runs: both readings (default), missing (the letter) or kept as D-002 §6 crashes")
    ap.add_argument("--followup", nargs="?", const=FOLLOWUP_SET, default=None, metavar="SETS",
                    help=f"also read D-022's INCONCLUSIVE follow-up: seeds 2-3 of these comma-separated sets (default "
                         f"{FOLLOWUP_SET}) on the instances the seeds 0-1 reading names (all 17 when none)")
    ap.add_argument("--json", default=None, help="also write the result as JSON here")
    a = ap.parse_args(argv)
    if not Path(a.runs).exists():
        print(f"{a.runs} does not exist (run `itrea build-runs` first)", file=sys.stderr)
        return 2
    fu = tuple(s for s in a.followup.split(",") if s) if a.followup else None
    res = report(pd.read_parquet(a.runs), Path(a.root), tuple(s for s in a.sets.split(",") if s), failed=a.failed,
                 followup=fu)
    print_report(res)
    if a.json:
        Path(a.json).write_text(json.dumps(_jsonable(res), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
