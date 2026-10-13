"""D-022 report: A9 under the external harness — cuOpt's MIP gaps 0 and its thread variables unset (D-021 R13 + R14),
development test (D-010) — and C1's interim A9 result (D-019 addendum), as D-022's two amendments of 2026-10-13
(before any run / before any reading; Berk's "targeted and surgical tests", D-024, then an adversarial check) define
them. The rules are D-022's, fixed before any reading (with its clarifications of 2026-10-12); nothing is chosen by
eye.

Usage (repository root, after `itrea build-runs`; untimed analysis: OMP_NUM_THREADS=1, never beside timed runs with
default thread counts):

    OMP_NUM_THREADS=1 .venv/bin/python tools/d022_report.py [--set itx-r13] [--failed both|exclude|keep]
                                                            [--instances A,B,C] [--c1-instances A,B,…]
                                                            [--guard-instances A,B,…]
                                                            [--followup [itx-r13f]] [--json out]
    itrea d022-report …                       (the same, through the entry point)

Reads `datasets/runs.parquet` (P(120) = `pt`, `final_gap`, `status`, `attempt`, `run_dir`, …) and, through each run's
`run_dir` (relative to `--root`), its `run.json` (the cuOpt process record, the compile-on-the-clock flags, the
resolved harness `config.harness`; the run table's `harness` column when run.json has none), `cuopt.summary.json` (the
solve's `settings`, `exit_reason`) and `ea.summary.json` (`P`, `gene_dtype`), and each set's void log
`<raw>/<set>/void.jsonl` (the pool's record of voided runs, moved aside and not in the table). The checked-incumbents
table is not needed: every number here is a run-table column or a run-directory field.

Arms (set itx-r13, 120 s, seeds 0–1): control A9@r13 (A9 under the harness {ext, cuopt_gap_zero false,
cuopt_env_unset false}), treatment A9@r13x (A9 under ext), same-day A6F@r13x (cuOpt alone on F, ext). A0 for the C1
block: the runs of sets `dev` / `classprobe`, 120 s, seeds 0–1. Set itx-r13 holds (D-022 amendments 2026-10-13):
  (a) the RULE SET (`--instances`, default RULE_INSTANCES): the three instances where cuOpt stops on its own 1e-4
      "optimal" in our recorded runs — fhnw-binpack4-48, neos-3004026-krka, neos-960392 — x the three arms (18 runs).
      Only these are read by D-022's rule (validity, mechanism, no harm, verdict, follow-up). The other 14 development
      + class-probe instances are not run — the stated assumption (ASSUMPTION, printed): cuOpt uses the MIP gaps only
      as a termination test, not as a pruning limit (not verified from its source or documentation); C1's guard over
      the 17 carries the check.
  (b) C1's runs: {A9@r13x, A6F@r13x} on the fixed 11 (`--c1-instances`, default C1_INSTANCES: cuOpt's weak
      development instances, cuOpt-alone P >= 0.5 at 120 s in earlier 3-core A6 runs; 44 runs, `jobs/itx-r13.jsonl`'s
      58 with (a)) and on the four the first amendment dropped (GUARD_ADDED, +16 runs queued after the 58), so the
      guard reads all 17 (`--guard-instances`, default the 17; the rule set's A9@r13x / A6F@r13x runs are C1's too).
      Read only by C1's block, which has its own completeness line (17 x 2 arms x 2 seeds = 68 runs) and never
      changes D-022's verdict. A9@r13 is not expected there (it runs on (a) only).

The rule's reading reads only its own runs: the rule set x the three arms x its seeds (0–1; the follow-up's 2–3) x
120 s, not synthetic, not in a voided block. Its failed runs and its voids are of those keys only (a void's key comes
from the run table's row of its run id — the rerun —, else from the voided directory's own run.json, written at the
run's start; a void whose key cannot be read is counted). C1's runs off the rule set, failed or voided, are not
its runs.

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

Mechanism (per A9 arm, reported; leg (2) of the rule for A9@r13x): the runs ended by cuOpt's "optimal" — the cuOpt
summary's `exit_reason` optimal and cuOpt's process the first to exit (its `exit_t` <= every other process's; so a
cuOpt that finished inside the kill window after SCIP's proof does not count) — and, of those, the ones whose run ends
with a final gap > 1e-9 against the reference (`final_gap`, the portfolio's; 1e-9 = the relative tolerance used for
objective equality elsewhere); cuOpt CPU-s per wall-s per arm = (cpu_user_s + cpu_sys_s) of the cuOpt process
(run.json `processes`) / its `exit_t` — D-021 M1's reading (A6F 5.02 / 4.99 / 5.03); start-up CPU is in the
numerator, so it reads ~1–3 % high.

The R13 + R14 rule (D-022 second amendment; replaces the bounded-d mean, which read noise at floor-level P; no mean
over the instances is taken). A run without P(120) on any of the three arms makes the verdict INCOMPLETE and is named.
  (1) validity holds (above);
  (2) the mechanism shows: no A9@r13x run is ended first by cuOpt's "optimal" with a final gap > 1e-9;
  (3) no harm: on FLOOR_INSTANCES (fhnw-binpack4-48, neos-3004026-krka) |ΔP| <= 0.005 on seed means, ΔP = P(A9@r13x)
      − P(A9@r13) — one-sided since D-022's clarification of 16:0x (only a loss beyond 0.005 fails it; a gain never does); on neos-960392 (and
      any other rule-set instance) the bounded d of the treatment vs the control on seed means >= −band_i, band_i =
      max(0.05, the larger seed-0 vs seed-1 bounded difference |a − b| / max(a, b, 1e-3) of the control or A6F@r13x).
  - REJECT (A9 keeps the default gaps; D-021's A9 rows run under {ext, gap_zero false, env_unset false}) if (3) fails
    on both seeds of an instance (|ΔP_s| > 0.005 on both seeds; d_s < −band_i on both seeds) — checked first, so an
    instance at |ΔP_s| > 0.005 on both seeds with opposite signs rejects although its seed-mean ΔP may hold;
  - ADOPT R13 + R14 for A9 (the C1 freeze carries harness ext) if (1), (2) and (3) hold;
  - otherwise INCONCLUSIVE -> seeds 2–3 on the instances where (3) fails on seed means or (2) fails (named; `itrea
    make-jobs itx-r13f --instances <names>` — its `all` means the 17).

The follow-up (`--followup [SET]`, D-022's clarification of 2026-10-12, same rules): applies only when the seeds 0–1
verdict is INCONCLUSIVE. Runs: the named instances x seeds 2–3 x the same three arms and harnesses, 120 s, in their own
set `itx-r13f`. Validity on those runs. A rerun instance: (3) on its seed means over seeds 0–3 (band_i stays seeds
0–1's), failing "on both seeds" when it fails on seeds 2 and 3; (2) over its A9@r13x runs of seeds 0–3 (a reading:
the mechanism leg is evidence about R13, not seed noise, so a seeds-0–1 failure is not washed out — an INCONCLUSIVE
from (2) alone can only end NOT ADOPTED, and the verdict says so). The other instances keep their seeds-0–1 result.
REJECT if a rerun instance fails (3) on both seeds 2 and 3; ADOPT if (3) and (2) hold on every instance; otherwise NOT
ADOPTED — the reject consequence, no further seeds (the terminal rule of 2026-10-12, open to Berk).

C1's interim block (D-022 second amendment; inputs to C1's go / no-go, which is Berk's; never D-022's verdict):
  - HINT present iff the ratio of means of P(120), A9@r13x / A6F@r13x, over the FIXED 11 (seed means, both seeds of
    both arms; not re-selected on these runs) <= 0.95; failed runs count as the arm's outcomes under `keep` (D-002
    §6) and are missing under `exclude` (both readings printed); INCOMPLETE while a run of the 11 is missing. No
    "improved > worse" limb.
  - GUARD holds iff the mean bounded d of A9@r13x vs A6F@r13x on seed means over all 17 (both-P = 1 at d = 0) >=
    −0.05; INCOMPLETE (no partial mean) while a run of the 17 is missing — the four GUARD_ADDED instances' 16 runs
    are queued after the 58.
  - Reported, not criteria (D-013 terms, read as `backoff.d013_report` and D-013's own table read them — every
    instance with both arms' runs, both-P = 1 ones at d = 0): A9@r13x vs A6F@r13x and vs A0 on the fixed 11, and on
    SCIP's weak set among them (A0 P >= 0.5): improved k of n and worse, mean P before -> after (relative
    reduction), mean d with its 95 % bootstrap interval (`stats.bootstrap_mean`), the Wilcoxon test
    (`stats.wilcoxon_d`, zeros dropped), worst P after, and the instances `stats.compare` would leave out.
  Its completeness line: the runs of the 17 (and the 11) x {A9@r13x, A6F@r13x} x seeds 0–1 present, missing,
  failed, without P(120), and A0's runs on the 11. Caveats printed: the 11 were chosen on earlier 3-core A6 runs; the
  A0 runs are from earlier days.

Failed runs: both readings (`--failed both`, the default; produced when the readings' own runs — the rule set's, C1's,
the follow-up's — include a failed one) — `exclude` (the letter: a failed run is missing, the verdict
INCOMPLETE) and `keep` (D-002 §6). Run identity: a job run again in another set keeps its run id and the run table
qualifies the later copy as `<run_id>#<set>` (build_runs.qualify_reused_ids). Runs are therefore selected by the `set`
column and keyed by (instance, arm, seed) — the run id is never parsed —, the directory is `run_dir`; when one key
occurs in more than one selected set the bare id is kept (then the highest attempt) and the others are listed.
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
GAP_ZERO_ARMS = (TREAT, A6F)              # every cuOpt solve under the ext harness
SET, SEEDS, BUDGET = "itx-r13", (0, 1), 120.0
FOLLOWUP_SET, FOLLOWUP_SEEDS = "itx-r13f", (2, 3)   # D-022's INCONCLUSIVE follow-up (clarification 2026-10-12)
A0_SETS = ("dev", "classprobe")
BAND_MIN = 0.05
GAP_TOL = 1e-9
WEAK_P = 0.5
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
REUSE_SEP = "#"
# D-022 amendment before any run (2026-10-13; D-024): (a) the rule set — where cuOpt stops on its own 1e-4 "optimal"
# (fhnw 9/9, krka 10/10, neos-960392 3/4 of the 92 recorded cuOpt runs at 120 s on the 17); (b) C1's interim set —
# cuOpt's weak development instances (cuOpt-alone P >= 0.5 at 120 s in earlier 3-core A6 runs; comp21 at 0.506).
# `jobs/itx-r13.jsonl` = (a) x ARMS + (b) x C1_ARMS x seeds 0–1 = 58 runs (neos-960392 in both).
RULE_INSTANCES = ("fhnw-binpack4-48", "neos-3004026-krka", "neos-960392")
C1_INSTANCES = ("neos-2746589-doon", "cryptanalysiskb128n5obj16", "rail01", "neos-2657525-crna", "neos-3024952-loue",
                "sing44", "ca__Scenario400_reserves_1", "item_placement_9903", "neos-960392", "neos-4763324-toguru",
                "comp21-2idx")
C1_ARMS = (TREAT, A6F)
# D-022 second amendment before any reading (2026-10-13 15:3x): the rule's no-harm leg — D-018's floor value, one-sided (harm only; clarification 16:0x)
# on the two floor-level instances, the band on the third; C1's hint and guard (inputs to Berk's go / no-go).
FLOOR_INSTANCES, FLOOR_DP = ("fhnw-binpack4-48", "neos-3004026-krka"), 0.005
HINT_RATIO, GUARD_MEAN = 0.95, -0.05
GUARD_ADDED = ("chromaticindex1024-7", "neos-3656078-kumeu", "glass4", "radiationm18-12-05")   # +16 runs for the guard
ASSUMPTION = ("the 14 other development + class-probe instances are not run because cuOpt never stopped early there; "
              "this assumes cuOpt uses the MIP gaps only as a termination test (not as a pruning limit), not verified "
              "from cuOpt's source or documentation — the C1 guard over the 17 carries the check")
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


def failed_runs(runs: pd.DataFrame, sets, seeds, instances, budget: float = BUDGET, arms=ARMS) -> list[dict]:
    """The reading's failed runs: of the rows `select` takes under `keep` — the same sets, arms, seeds, budget and
    instances, not synthetic, not in a voided block, one per key — those whose status is failed."""
    df, _ = select(runs, sets, arms, seeds, budget, keep_failed=True, instances=instances)
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


def _mech_fail(mech: dict) -> list[tuple[str, int]]:
    """Leg (2)'s failures: the A9@r13x runs ended first by cuOpt's "optimal" with a final gap > GAP_TOL."""
    return sorted((x["instance"], int(x["seed"])) for x in mech[TREAT]["with_final_gap"])


def _harm_row(i: str, pc: list, pt: list, pa: list, band: float | None = None) -> dict:
    """The no-harm leg (3) on one instance, from its per-seed P of the control, the treatment and A6F: the floor limb
    (FLOOR_INSTANCES: |ΔP| <= FLOOR_DP on seed means, ΔP = P(A9@r13x) − P(A9@r13), one-sided: harm only, D-022 clarification 16:0x)
    or the band limb (every other instance: bounded d on seed means >= −band_i); `fail_seed` = the limb failing per
    seed (|ΔP_s| > FLOOR_DP, or d_s < −band_i). band_i = max(BAND_MIN, the control's or A6F's seed-0 vs seed-1 spread)
    unless given (the follow-up keeps seeds 0–1's)."""
    floor = i in FLOOR_INSTANCES
    mc, mt, ma = float(np.mean(pc)), float(np.mean(pt)), float(np.mean(pa))
    if band is None:
        band = max([BAND_MIN] + [v for v in (bounded(pc[0], pc[1]), bounded(pa[0], pa[1])) if _fin(v)])
    dp_s = [t - c for c, t in zip(pc, pt)]
    d_s = [effect_d(c, t) for c, t in zip(pc, pt)]
    if floor:
        value, value_s, limit = mt - mc, dp_s, FLOOR_DP
        no_harm, fail_s = value <= FLOOR_DP, [v > FLOOR_DP for v in dp_s]   # one-sided: harm only (D-022 clarification 16:0x)
    else:
        value, value_s, limit = effect_d(mc, mt), d_s, band
        no_harm, fail_s = value >= -band, [v < -band for v in d_s]
    return dict(instance=i, kind="floor" if floor else "band", p_ctrl=mc, p_treat=mt, p_a6f=ma, pc_seeds=list(pc),
                pt_seeds=list(pt), pa_seeds=list(pa), dP=mt - mc, dP_seed=dp_s, d=effect_d(mc, mt), d_seed=d_s,
                band=band, value=value, value_seed=value_s, limit=limit, no_harm=bool(no_harm), fail_seed=fail_s)


def _limb(r: dict) -> str:
    return (f"ΔP <= +{FLOOR_DP:g}" if r["kind"] == "floor" else f"bounded d >= −band ({r['band']:.2f})")


def outcome(df: pd.DataFrame, instances, seeds=SEEDS, mech_fail=()) -> dict:
    """D-022's R13 + R14 rule after its second amendment (module docstring) on the seed pair `seeds`, given validity
    (read apart) and leg (2)'s failures `mech_fail` [(instance, seed)]: REJECT if leg (3) fails on both seeds of an
    instance (before ADOPT); ADOPT if leg (2) shows and leg (3) holds on every instance; otherwise INCONCLUSIVE ->
    seeds 2–3 on the instances where (3) fails on seed means or (2) fails."""
    P, no_p = _p_table(df)
    rows, missing = [], []
    for i in instances:
        pc = [P.get((i, CTRL, s)) for s in seeds]
        pt = [P.get((i, TREAT, s)) for s in seeds]
        pa = [P.get((i, A6F, s)) for s in seeds]
        if any(v is None for v in pc + pt + pa):          # band_i needs A6F's pair as well
            missing.append(i)
            rows.append(dict(instance=i, kind="floor" if i in FLOOR_INSTANCES else "band", value=math.nan))
            continue
        r = _harm_row(i, pc, pt, pa)
        r.update(fails_both=bool(all(r["fail_seed"])), mechanism_fail=sorted(s for j, s in mech_fail if j == i))
        rows.append(r)
    base = dict(rows=rows, missing=missing, no_p=no_p, mechanism_fail=list(mech_fail))
    if missing:
        return dict(base, kind="INCOMPLETE", rejected=[], rerun=[],
                    verdict=f"INCOMPLETE: no seed pair of {CTRL}, {TREAT} and {A6F} on {missing}")
    rej = [r for r in rows if r["fails_both"]]
    rerun = [r["instance"] for r in rows if not r["no_harm"] or r["mechanism_fail"]]
    if rej:
        why = "; ".join(f"{r['instance']} ({_limb(r)} fails on both seeds: "
                        + ", ".join(f"{v:+.4f}" for v in r["value_seed"]) + ")" for r in rej)
        return dict(base, kind="REJECT", rejected=[r["instance"] for r in rej], rerun=[],
                    verdict="REJECT R13 + R14 for A9 (no-harm leg fails on both seeds: " + why + "): " + REJECT_TAIL)
    if not rerun:
        return dict(base, kind="ADOPT", rejected=[], rerun=[],
                    verdict="ADOPT R13 + R14 for A9 (validity holds; mechanism shows: no A9@r13x run ended first by "
                            f"cuOpt's 'optimal' with a final gap > {GAP_TOL:g}; no harm on the {len(rows)}): the C1 "
                            "freeze carries harness ext")
    harm = [r["instance"] for r in rows if not r["no_harm"]]
    mech = [(r["instance"], r["mechanism_fail"]) for r in rows if r["mechanism_fail"]]
    why = ([f"no harm not shown on seed means on {harm}"] if harm else []) + \
          ([f"mechanism fails (A9@r13x ended first by cuOpt's 'optimal' with a final gap) on {mech}"] if mech else [])
    only_mech = not harm
    return dict(base, kind="INCONCLUSIVE", rejected=[], rerun=rerun,
                verdict=f"INCONCLUSIVE ({'; '.join(why)}): seeds 2–3 in set {FOLLOWUP_SET}, same rules, on "
                        + ", ".join(rerun) + " (pass these names to `itrea make-jobs itx-r13f --instances`; its 'all' "
                        "is the 17)" + ("; the mechanism leg reads every A9@r13x run of seeds 0–3, so a seeds 0–1 "
                                        "failure ends NOT ADOPTED whatever seeds 2–3 show (the rerun can be declined)"
                                        if only_mech else ""))


def followup_instances(base: dict, instances=None) -> list[str]:
    """The instances the follow-up reruns: those the seeds 0–1 reading names (leg (3) fails on seed means without
    failing on both seeds, or leg (2) fails)."""
    return list(base["rerun"])


def outcome_followup(df: pd.DataFrame, base: dict, rerun, mech_fail=()) -> dict:
    """The follow-up reading (module docstring): `df` holds the rerun instances' seeds 2–3, `base` is the seeds 0–1
    outcome (INCONCLUSIVE), `mech_fail` = leg (2)'s failures on the follow-up's A9@r13x runs. A rerun instance: leg (3)
    on its seed means over seeds 0–3 (band_i kept), failing "on both seeds" when it fails on seeds 2 and 3; leg (2)
    over its A9@r13x runs of seeds 0–3. REJECT if a rerun instance fails (3) on both seeds 2 and 3; ADOPT if (3) holds
    and (2) shows on every instance; otherwise NOT ADOPTED."""
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
            rows.append(dict(instance=i, kind=r["kind"], value=math.nan, rerun=True))
            continue
        x = _harm_row(i, r["pc_seeds"] + pc, r["pt_seeds"] + pt, r["pa_seeds"] + pa, band=r["band"])
        x.update(rerun=True, fails_both=bool(all(x["fail_seed"][2:])),
                 mechanism_fail=sorted(r["mechanism_fail"] + [s for j, s in mech_fail if j == i]))
        rows.append(x)
    base_out = dict(rows=rows, missing=missing, no_p=no_p, mechanism_fail=list(mech_fail))
    if missing:
        return dict(base_out, kind="INCOMPLETE", rejected=[],
                    verdict=f"INCOMPLETE: no seeds 2–3 of the three arms on {missing}")
    rej = [x for x in rows if x["rerun"] and x["fails_both"]]
    if rej:
        why = "; ".join(f"{x['instance']} ({_limb(x)} fails on seeds 2 and 3: "
                        + ", ".join(f"{v:+.4f}" for v in x["value_seed"][2:]) + ")" for x in rej)
        return dict(base_out, kind="REJECT", rejected=[x["instance"] for x in rej],
                    verdict="REJECT R13 + R14 for A9 after seeds 2–3 (" + why + "): " + REJECT_TAIL)
    if all(x["no_harm"] and not x["mechanism_fail"] for x in rows):
        return dict(base_out, kind="ADOPT", rejected=[],
                    verdict="ADOPT R13 + R14 for A9 after seeds 2–3 (no harm on seeds 0–3 of the rerun instances, "
                            "mechanism shows on their A9@r13x runs of seeds 0–3): the C1 freeze carries harness ext")
    open_ = [x["instance"] for x in rows if not x["no_harm"] or x["mechanism_fail"]]
    return dict(base_out, kind="NOT ADOPTED", rejected=[],
                verdict=f"NOT ADOPTED: still inconclusive after seeds 2–3 on {open_} — no harm or the mechanism not "
                        "shown, the reject consequence (D-022's terminal rule of 2026-10-12, open to Berk): "
                        + REJECT_TAIL)


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


def _c1_missing(long: pd.DataFrame, instances) -> list[tuple]:
    have = {(r["instance"], r["arm"], int(r["seed"])) for r in long.to_dict("records")}
    return [(i, a, s) for i in instances for a in C1_ARMS for s in SEEDS if (i, a, s) not in have]


def c1_hint(long: pd.DataFrame, instances) -> dict:
    """C1's interim hint (D-022 second amendment): the ratio of means of P(120), A9@r13x / A6F@r13x, over the FIXED
    weak set `instances` (seed means, both seeds of both arms; failed runs are A9 / A6F outcomes under `keep`, missing
    under `exclude`): present iff <= HINT_RATIO; INCOMPLETE while a run is missing."""
    miss = _c1_missing(long, instances)
    out = dict(n=len(instances), threshold=HINT_RATIO, missing=miss)
    if miss or not instances:
        return dict(out, state="INCOMPLETE", ratio=math.nan, mean_P_a9=math.nan, mean_P_a6f=math.nan)
    m = long.groupby(["arm", "instance"])["pt"].mean()
    a9 = float(np.mean([m[(TREAT, i)] for i in instances]))
    a6 = float(np.mean([m[(A6F, i)] for i in instances]))
    ratio = a9 / a6 if a6 > 0 else (math.inf if a9 > 0 else math.nan)
    return dict(out, state="present" if _fin(ratio) and ratio <= HINT_RATIO else "absent", ratio=ratio,
                mean_P_a9=a9, mean_P_a6f=a6)


def c1_guard(long: pd.DataFrame, instances) -> dict:
    """C1's guard (D-022 second amendment): the mean bounded d of A9@r13x vs A6F@r13x on seed means over `instances`
    (all 17; both-P = 1 at d = 0) >= GUARD_MEAN; INCOMPLETE while a run of either arm is missing (no partial mean)."""
    miss = _c1_missing(long, instances)
    out = dict(n=len(instances), threshold=GUARD_MEAN, missing=miss)
    if miss or not instances:
        return dict(out, state="INCOMPLETE", mean_d=math.nan, rows=[])
    m = long.groupby(["arm", "instance"])["pt"].mean()
    rows = [dict(instance=i, p_a6f=float(m[(A6F, i)]), p_a9=float(m[(TREAT, i)]),
                 d=effect_d(float(m[(A6F, i)]), float(m[(TREAT, i)]))) for i in instances]
    mean_d = float(np.mean([r["d"] for r in rows]))
    return dict(out, state="holds" if mean_d >= GUARD_MEAN else "fails", mean_d=mean_d, rows=rows)


def c1_interim(df: pd.DataFrame, a0: pd.DataFrame, instances, guard_instances) -> dict:
    """C1's interim A9 result (inputs to C1's go / no-go, which is Berk's): the hint on the fixed 11 (`instances`), the
    guard over the 17 (`guard_instances`), and reported in D-013 terms: A9@r13x vs A6F@r13x on the fixed 11 (not
    re-selected on these runs) and vs A0 on the fixed 11 with SCIP's weak set among them (A0 P >= WEAK_P)."""
    cols = ["instance", "seed", "arm", "pt"]
    long = pd.concat([df[cols], a0[cols]], ignore_index=True)
    long = long[long["pt"].notna()]
    fixed = long[long["instance"].isin(list(instances))]
    means = fixed.groupby(["arm", "instance"])["pt"].mean()
    weak0 = sorted(i for i in instances if (A0, i) in means.index and means[(A0, i)] >= WEAK_P)
    return {"hint": c1_hint(long, list(instances)), "guard": c1_guard(long, list(guard_instances)),
            A6F: {"label": "cuOpt alone (A6F@r13x, same day, equal footprint)",
                  TREAT: dict(all=_weak_block(fixed, TREAT, A6F, list(instances)))},
            A0: {"label": "SCIP alone (A0, dev / classprobe)", "weak_set": weak0,
                 "n_baseline": int(sum(1 for i in instances if (A0, i) in means.index)),
                 TREAT: dict(all=_weak_block(fixed, TREAT, A0, list(instances)),
                             weak=_weak_block(fixed, TREAT, A0, weak0))}}


def c1_reading(runs: pd.DataFrame, sets, instances, keep_failed: bool, guard_instances=()) -> dict:
    """C1's interim block under one reading of failed runs, with its own completeness line over the guard's 17 and
    the fixed 11 x C1_ARMS x seeds 0–1 (a failed run is missing under `exclude`, kept under `keep`), failed runs,
    runs without P(120), and A0's runs on the fixed 11. Nothing here enters D-022's verdict."""
    every = list(dict.fromkeys(list(guard_instances) + list(instances)))
    df, dropped = select(runs, sets, C1_ARMS, SEEDS, keep_failed=keep_failed, instances=every)
    failed = failed_runs(runs, sets, SEEDS, every, arms=C1_ARMS)
    a0, a0_dropped = select(runs, A0_SETS, (A0,), instances=instances)
    have = {(r["instance"], r["arm"], int(r["seed"])) for r in df.to_dict("records")}
    missing = [(i, a, s) for i in every for a in C1_ARMS for s in SEEDS if (i, a, s) not in have]
    _, no_p = _p_table(df)
    a0_have = {(r["instance"], int(r["seed"])) for r in a0.to_dict("records")}
    a0_missing = [(i, s) for i in instances for s in SEEDS if (i, s) not in a0_have]
    return dict(blocks=c1_interim(df, a0, instances, guard_instances), duplicates_left_out=dropped + a0_dropped,
                completeness=dict(instances=every, arms=list(C1_ARMS), seeds=list(SEEDS),
                                  n_expected=len(every) * len(C1_ARMS) * len(SEEDS), n_runs=int(len(df)),
                                  missing=missing, failed=failed, no_p=no_p, n_a0=int(len(a0)),
                                  n_a0_expected=len(instances) * len(SEEDS), a0_missing=a0_missing,
                                  complete=bool(not missing and not no_p)))


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
    mech = mechanism(df, info_cache)
    out = outcome_followup(df, base, rerun, _mech_fail(mech))
    return dict(sets=list(sets), applies=True, rerun=rerun, seeds=list(FOLLOWUP_SEEDS), validity=val, mechanism=mech,
                outcome=out, verdict=_verdict(val, out, keep_failed, failed), n_runs=int(len(df)),
                duplicates_left_out=dropped, failed_runs=failed)


def reading(runs: pd.DataFrame, root: Path, sets, instances, keep_failed: bool, info_cache: dict,
            followup=None, c1_instances=C1_INSTANCES, guard_instances=()) -> dict:
    """One reading of failed runs: D-022's rule on the rule set `instances` and, apart from it, C1's interim block
    (the hint on `c1_instances`, the guard on `guard_instances`); C1's runs, missing or failed, never enter the rule's
    validity or verdict."""
    df, dropped = select(runs, sets, ARMS, SEEDS, keep_failed=keep_failed, instances=instances)
    # failed runs: under `exclude` they are missing (and listed); under `keep` they are scored, and validity's
    # "completed" leg fails by the letter (a stated deviation, named in the verdict)
    failed = failed_runs(runs, sets, SEEDS, instances)
    _fill_info(df, root, info_cache)
    val = validity(df, info_cache, instances, void_log(runs, sets, root), failed, SEEDS, keep_failed)
    mech = mechanism(df, info_cache)
    out = outcome(df, instances, mech_fail=_mech_fail(mech))
    verdict = _verdict(val, out, keep_failed, failed)
    c1 = c1_reading(runs, sets, list(c1_instances), keep_failed, list(guard_instances))
    dup, seen = [], set()
    for d in dropped + c1["duplicates_left_out"]:      # the rule set's A9@r13x / A6F@r13x are C1's too
        k = (d["instance"], d["arm"], d["seed"], d["run_id"])
        if k not in seen:
            seen.add(k)
            dup.append(d)
    res = dict(reading="keep" if keep_failed else "exclude", verdict=verdict, validity=val, mechanism=mech,
               outcome=out, c1=c1["blocks"], c1_completeness=c1["completeness"], duplicates_left_out=dup,
               n_runs=int(len(df)), n_a0=c1["completeness"]["n_a0"], failed_runs=failed)
    if followup:
        res["followup"] = followup_reading(runs, root, followup, instances, keep_failed, info_cache, out, verdict)
    return res


def report(runs: pd.DataFrame, root: Path = ROOT, sets=(SET,), instances=None, failed: str = "both",
           followup=None, c1_instances=None, guard_instances=None) -> dict:
    """D-022 under the readings of failed runs: "exclude", "keep" or "both" (both only when the readings' own runs —
    the rule set's seeds 0–1, C1's (the guard's 17 and the fixed 11 x {A9@r13x, A6F@r13x}) and, with `followup` (its
    sets), seeds 2–3 — include a failed one; otherwise the readings coincide and one is returned). `instances` = the
    rule set (default RULE_INSTANCES), `c1_instances` = C1's fixed weak set (default C1_INSTANCES), `guard_instances`
    = the guard's (default the 17 development + class-probe instances)."""
    instances = list(instances) if instances is not None else list(RULE_INSTANCES)
    c1_instances = list(c1_instances) if c1_instances is not None else list(C1_INSTANCES)
    dev = development_instances()
    guard_instances = list(guard_instances) if guard_instances is not None else list(dev)
    followup = tuple(followup) if followup else None
    cache: dict = {}
    every_c1 = list(dict.fromkeys(guard_instances + c1_instances))
    has_failed = bool(failed_runs(runs, sets, SEEDS, instances)) or bool(
        failed_runs(runs, sets, SEEDS, every_c1, arms=C1_ARMS)) or bool(
        followup and failed_runs(runs, followup, FOLLOWUP_SEEDS, instances))
    modes = {"exclude": [False], "keep": [True], "both": [False, True] if has_failed else [False]}[failed]
    return dict(sets=list(sets), followup_sets=list(followup) if followup else None, instances=instances,
                c1_instances=c1_instances, guard_instances=guard_instances,
                rule_set_is_the_amendments=sorted(instances) == sorted(RULE_INSTANCES),
                not_in_rule_set=[i for i in dev if i not in instances],
                readings=[reading(runs, root, sets, instances, k, cache, followup, c1_instances, guard_instances)
                          for k in modes])


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


def _d013_line(w: dict, label: str, of: str) -> str:
    """One D-013-terms line of the C1 block (`_weak_block`'s numbers)."""
    if not w.get("n"):
        return f"{label}: no instance of the {w.get('n_weak', 0)} {of} has both arms' runs"
    return (f"{label}: improved {w['better']} of {w['n']}"
            + (f" ({w['n_weak'] - w['n']} {of} without both arms' runs)" if w["n_weak"] > w["n"] else "")
            + f", mean P {_fmt(w['mean_P_before'])} -> {_fmt(w['mean_P_after'])} ({_fmt(-100 * w['P_reduction'], 0)} %)"
            f", worse {w['worse']}, mean d {_fmt(w['mean_d'])} [{_fmt(w['ci'][0])}, {_fmt(w['ci'][1])}], Wilcoxon p "
            f"{_fmt(w['p'], 4)}, worst P after {_fmt(w['worst_P_after'])}"
            + (f"; both P = 1 (counted at d = 0; stats.compare would leave out): {w['left_out_by_compare']}"
               if w["left_out_by_compare"] else ""))


def _c1_completeness_line(c: dict, keep: bool) -> str:
    n_inst = len(c["instances"])
    return (f"C1 completeness (its own; not D-022's): {c['n_runs']} of {c['n_expected']} runs ({n_inst} instances x "
            f"{' / '.join(c['arms'])} x seeds {c['seeds']}) — missing {len(c['missing'])}"
            + (f" {c['missing'][:6]}" if c["missing"] else "")
            + (f" (of them {sum(1 for m in c['missing'] if m[0] in GUARD_ADDED)} on the four instances added for the "
               "guard, queued after the 58)" if any(m[0] in GUARD_ADDED for m in c["missing"]) else "")
            + f"; failed {len(c['failed'])}"
            + (f" {[(f['instance'], f['arm'], f['seed']) for f in c['failed'][:6]]} "
               + ("(kept as D-002 §6 crashes)" if keep else "(missing)") if c["failed"] else "")
            + f"; without P(120) {len(c['no_p'])}" + (f" {c['no_p'][:6]}" if c["no_p"] else "")
            + f"; A0 (dev / classprobe, the fixed 11) {c['n_a0']} of {c['n_a0_expected']}"
            + (f", missing {c['a0_missing'][:6]}" if c["a0_missing"] else "")
            + (" — complete" if c["complete"] else " — INCOMPLETE (C1 reads what is there; D-022's verdict is not "
               "affected)"))


def _harm_cells(r: dict, seeds: tuple[int, ...]) -> str:
    """One row of leg (3): limb, P on seed means, the limb's value on seed means and on the last two seeds (0–1, or
    the follow-up's 2–3), the limit, the leg, the seeds where it fails (index = seed) and leg (2)'s failures."""
    fail = [k for k, f in enumerate(r["fail_seed"]) if f]
    return (f"{r['kind']:<6}{_fmt(r['p_ctrl']):<8}{_fmt(r['p_treat']):<8}{_fmt(r['p_a6f']):<8}{r['value']:+.4f}     "
            + "".join(f"{v:+.4f}  " for v in r["value_seed"][-2:]) + f"{_limb(r):<28}"
            + ("holds   " if r["no_harm"] else "FAILS   ")
            + (f"both seeds {list(seeds)}" if r["fails_both"] else (f"seeds {fail}" if fail else "-"))
            + (f"; mechanism fails on seeds {r['mechanism_fail']}" if r.get("mechanism_fail") else ""))


def _harm_header(indent: str, seeds: tuple[int, ...]) -> str:
    return (indent + "instance".ljust(29) + "limb".ljust(6) + "P ctrl".ljust(8) + "P treat".ljust(8) + "P A6F".ljust(8)
            + "seed means".ljust(12) + "".join(f"s{s}".ljust(9) for s in seeds[-2:]) + "limit".ljust(28)
            + "leg".ljust(8) + "fails on")


def print_report(res: dict, out=sys.stdout) -> None:
    p = lambda *a: print(*a, file=out)  # noqa: E731
    n, n_c1, other = len(res["instances"]), len(res["c1_instances"]), res.get("not_in_rule_set") or []
    n_g = len(res.get("guard_instances") or [])
    p(f"D-022 — A9 under the external harness (R13 + R14): {TREAT} (ext) vs {CTRL} (ext without R13 / R14), "
      f"same-day {A6F}; sets {res['sets']}, seeds {list(SEEDS)}, {BUDGET:g} s"
      + (f"; follow-up sets {res['followup_sets']}, seeds {list(FOLLOWUP_SEEDS)}" if res.get("followup_sets") else ""))
    if res.get("rule_set_is_the_amendments"):
        p(f"(a) rule set — the only instances D-022's rule reads (validity, mechanism, no harm, verdict, follow-up): "
          f"{n} {res['instances']}, where cuOpt stops on its own 1e-4 'optimal' (D-022 amendments of 2026-10-13); the "
          f"other {len(other)} development + class-probe instances are not run")
    else:
        p(f"(a) rule set (--instances, overriding D-022's amendments of 2026-10-13): {n} {res['instances']}; "
          f"{len(other)} development + class-probe instances are not read by the rule")
    p(f"    rule (second amendment): ADOPT iff validity holds, the mechanism shows (no {TREAT} run ended first by "
      f"cuOpt's 'optimal' with a final gap > {GAP_TOL:g}) and no harm (floor limb on {list(FLOOR_INSTANCES)}: |ΔP| <= "
      f"{FLOOR_DP:g} on seed means, ΔP = P({TREAT}) − P({CTRL}), one-sided: only a loss fails (clarification 16:0x); band limb on the "
      "others: bounded d >= −band_i); REJECT iff the no-harm leg fails on both seeds of an instance (checked first); "
      "otherwise INCONCLUSIVE -> seeds 2–3 on those instances; still inconclusive -> NOT ADOPTED. No mean over the "
      "instances.")
    p(f"    stated assumption: {ASSUMPTION}")
    p(f"(b) C1 interim: hint on the fixed {n_c1} {res['c1_instances']} (not re-selected on these runs), guard over the "
      f"{n_g} — {TREAT} vs {A6F} (and vs A0, reported); {CTRL} is not expected there; C1 never enters D-022's verdict")
    for rd in res["readings"]:
        v = rd["validity"]
        p("")
        p(f"=== reading '{rd['reading']}' (failed runs " + ("kept as D-002 §6 keeps crashes" if rd["reading"] == "keep"
                                                            else "missing") + f"): {rd['n_runs']} runs of the rule set")
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
        mf = o.get("mechanism_fail") or []
        p(f"mechanism leg (2): {'shows' if not mf else 'FAILS on ' + str(mf)} ({TREAT} runs ended first by cuOpt's "
          f"'optimal' with a final gap > {GAP_TOL:g}: {len(mf)})")
        p(f"no-harm leg (3) (floor limb: ΔP = P({TREAT}) − P({CTRL}); band limb: bounded d, + = treatment better):")
        p(_harm_header("  ", SEEDS))
        for r in o["rows"]:
            if not _fin(r.get("value")):
                p(f"  {r['instance']:<28} (missing)")
                continue
            p(f"  {r['instance']:<28} " + _harm_cells(r, SEEDS))
        p("C1 interim (D-019 addendum; inputs to C1's go / no-go, which is Berk's — not D-022's rule):")
        p("  " + _c1_completeness_line(rd["c1_completeness"], rd["reading"] == "keep"))
        h, g = rd["c1"]["hint"], rd["c1"]["guard"]
        if h["state"] == "INCOMPLETE":
            p(f"  HINT INCOMPLETE: {len(h['missing'])} runs of the fixed {h['n']} missing {h['missing'][:4]} (needs "
              "both seeds of both arms)")
        else:
            p(f"  HINT {h['state']}: ratio of means of P(120) {TREAT} / {A6F} over the fixed {h['n']} = "
              f"{_fmt(h['ratio'])} ({_fmt(h['mean_P_a9'])} / {_fmt(h['mean_P_a6f'])}; present iff <= "
              f"{h['threshold']:g}; seed means, both seeds)")
        if g["state"] == "INCOMPLETE":
            p(f"  GUARD INCOMPLETE: {len(g['missing'])} runs of the {g['n']} missing {g['missing'][:4]} (no partial "
              "mean)")
        else:
            p(f"  GUARD {g['state']}: mean bounded d of {TREAT} vs {A6F} over the {g['n']} = {_fmt(g['mean_d'])} "
              f"(holds iff >= {g['threshold']:g}; seed means, both-P = 1 at d = 0)")
        p("  go / no-go on C1 is Berk's (D-019 addendum); the hint and the guard are its inputs")
        a = rd["c1"][A6F][TREAT]["all"]
        p(f"  reported, not criteria (D-013 terms) — vs {rd['c1'][A6F]['label']}: "
          + _d013_line(a, f"the fixed {n_c1}", "instances"))
        c = rd["c1"][A0]
        p(f"  reported, not criteria (D-013 terms) — vs {c['label']}: "
          + _d013_line(c[TREAT]["all"], f"the fixed {n_c1}", "instances"))
        p(f"    SCIP's weak set among them (A0 P >= {WEAK_P}) {c['weak_set']} ({len(c['weak_set'])} of "
          f"{c['n_baseline']} with A0 runs)" + (": " + _d013_line(c[TREAT]["weak"], "weak set", "weak")
                                              if c["weak_set"] else ""))
        p("  caveats (D-022): the fixed 11 were chosen on earlier 3-core A6 runs (cuOpt-alone P >= 0.5; comp21-2idx "
          "borderline at 0.506), not re-selected here; the A0 runs are from earlier days (sets dev / classprobe), not "
          "same-day")
        if rd["duplicates_left_out"]:
            p(f"note: the same (instance, arm, seed) in more than one selected set — kept the bare run id, left out "
              f"{[(d['instance'], d['arm'], d['seed'], d['run_id']) for d in rd['duplicates_left_out'][:4]]}")
        p(f"VERDICT ({rd['reading']}): {rd['verdict']}")
        f = rd.get("followup")
        if f:
            if not f["applies"]:
                p(f"FOLLOW-UP ({rd['reading']}, sets {f['sets']}): {f['verdict']}")
                continue
            p(f"--- follow-up ({rd['reading']}): seeds {f['seeds']} of sets {f['sets']} on {f['rerun']}: "
              f"{f['n_runs']} runs")
            p("    " + _validity_line(f["validity"]))
            fo = f["outcome"]
            p("    no-harm leg (3) on seeds 0–3 (fails on both = seeds 2 and 3); mechanism leg (2) over seeds 0–3:")
            p(_harm_header("    ", FOLLOWUP_SEEDS))
            for r in fo["rows"]:
                if not r.get("rerun"):
                    continue
                if not _fin(r.get("value")):
                    p(f"    {r['instance']:<28} (missing)")
                    continue
                p(f"    {r['instance']:<28} " + _harm_cells(r, FOLLOWUP_SEEDS))
            p(f"FOLLOW-UP VERDICT ({rd['reading']}): {f['verdict']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tools/d022_report.py", description="D-022: A9 under the external harness "
                                 "(R13 + R14), the rules applied; C1's interim A9 result")
    ap.add_argument("--set", dest="sets", default=SET, help="comma-separated run sets (default itx-r13)")
    ap.add_argument("--runs", default=str(DATASETS / "runs.parquet"))
    ap.add_argument("--root", default=str(ROOT), help="root of the run_dir paths in the run table")
    ap.add_argument("--failed", choices=("both", "exclude", "keep"), default="both",
                    help="failed runs: both readings (default), missing (the letter) or kept as D-002 §6 crashes")
    ap.add_argument("--instances", default=",".join(RULE_INSTANCES),
                    help="comma-separated rule set: the only instances D-022's rule reads (default: the amendment's "
                         "three, " + ",".join(RULE_INSTANCES) + ")")
    ap.add_argument("--c1-instances", default=",".join(C1_INSTANCES),
                    help="comma-separated fixed weak set of C1's hint (default: cuOpt's 11 weak development "
                         "instances, not re-selected)")
    ap.add_argument("--guard-instances", default=None,
                    help="comma-separated instances of C1's guard (default: the 17 development + class-probe ones)")
    ap.add_argument("--followup", nargs="?", const=FOLLOWUP_SET, default=None, metavar="SETS",
                    help=f"also read D-022's INCONCLUSIVE follow-up: seeds 2-3 of these comma-separated sets (default "
                         f"{FOLLOWUP_SET}) on the instances the seeds 0-1 reading names (`make-jobs itx-r13f "
                         "--instances all` is the 17, so pass the names)")
    ap.add_argument("--json", default=None, help="also write the result as JSON here")
    a = ap.parse_args(argv)
    if not Path(a.runs).exists():
        print(f"{a.runs} does not exist (run `itrea build-runs` first)", file=sys.stderr)
        return 2
    fu = tuple(s for s in a.followup.split(",") if s) if a.followup else None
    res = report(pd.read_parquet(a.runs), Path(a.root), tuple(s for s in a.sets.split(",") if s), failed=a.failed,
                 followup=fu, instances=[x for x in a.instances.split(",") if x],
                 c1_instances=[x for x in a.c1_instances.split(",") if x],
                 guard_instances=[x for x in a.guard_instances.split(",") if x] if a.guard_instances else None)
    print_report(res)
    if a.json:
        Path(a.json).write_text(json.dumps(_jsonable(res), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
