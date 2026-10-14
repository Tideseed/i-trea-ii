"""R3 report: the same-window diagnostic of `vault/30-results/2026-10-14 C1 diagnosis and improvement candidates.md`
§5 R3 — cuOpt's thread count (A6 = 3 threads on A725 1–3 vs A6F = 5 threads on F) and SCIP's presence beside cuOpt
(A2 = SCIP on X925 5 + the same 3-thread cuOpt, no EA, X925 6 idle), all under the ext harness. A diagnostic, not a
lever: no hint, no verdict on A9; three reading rules, fixed in that note before any run, applied here as written.

Usage (repository root, after `itrea build-runs`; untimed analysis: OMP_NUM_THREADS=1, never beside timed runs with
default thread counts):

    OMP_NUM_THREADS=1 .venv/bin/python tools/r3_report.py [--runs datasets/runs.parquet] [--root .] [--set r3-diag]
                                                          [--failed both|exclude|keep] [--json out]
    itrea r3-report …                         (the same, through the entry point)

Runs (`make-jobs r3-diag` -> jobs/r3-diag.jsonl, set r3-diag, 120 s, seeds 0–1, 30 runs): R3_INSTANCES (comp21-2idx,
neos-4763324-toguru, neos-2746589-doon, sing44, item_placement_9903, ca__Scenario400_reserves_1) x {A6@r3, A6F@r3} and
R3_A2_INSTANCES (comp21-2idx, toguru, doon) x A2@r3. Instances and labels come from `make_jobs`, so the report and the
job file cannot drift apart. Reads `datasets/runs.parquet` — `pt` (P(120) of the run), `cuopt_pt` (P(120) of cuOpt's
own checked incumbents), `cuopt_best_obj` (the checked objective cuOpt holds at T), `sense`, `status`, `harness`,
`cuda_compiled_on_clock`, `run_dir` — and each run's `run.json` (`processes[*].role / cores / cmd`).

The reading rules (the note, §5 R3, "fixed before the runs"):
  1. Thread effect on instance i is present iff mean P(A6@ext) − mean P(A6F@ext) > max(0.05, the larger seed-0 vs
     seed-1 spread of either arm), on each of the six. Present on toguru or doon → the layout question (a cuOpt thread
     on X925 6) goes to Berk with these numbers; otherwise the layout line is closed for v1. These same-day numbers
     replace §2's indicative footprint / slowdown split.
  2. comp21's part 2, from cuOpt's final objective in A2@ext: >= 146 on both seeds → SCIP beside cuOpt is the cause;
     <= 133 on both seeds → the EA's CUDA context remains, and the MPS question (a host change, Berk's) becomes
     concrete; mixed → inconclusive at n = 2. Precondition: A6@ext's own cuOpt must reach <= 133 on comp21; if it ends
     >= 146 too, the stall is the 3-thread footprint under ext and rule 2 does not apply.
  3. SCIP's presence on toguru and doon: P of cuOpt in A2@ext vs A6@ext, with the same band as rule 1.

How this report reads them (written with the code, 2026-10-14, before any R3 run; each point is a reading of the note's
words, not a new rule):
  - "spread" = the raw |P(seed 0) − P(seed 1)| of an arm — the note's other rules use the word for raw P (R2: "ca's
    spread is 0.005", from 0.391 / 0.396) and rule 1 compares it with a raw difference of mean P — not D-018's bounded
    difference. band_i = max(0.05, spread of A6@r3, spread of A6F@r3) (rule 3: of A2@r3's and A6@r3's cuOpt P).
  - Rules 1 and 3 are one-sided as written: "present" only when the 3-thread / SCIP-beside arm is WORSE (higher P) by
    more than the band. A difference the other way is printed as a number, never called present.
  - Rule 1 on `pt` (A6 and A6F have one process, so `pt` = `cuopt_pt`); rule 3 on `cuopt_pt` for both arms (A2's `pt`
    is the race's portfolio, not cuOpt's).
  - Rule 1's consequence: QUESTION as soon as toguru or doon reads "present"; CLOSED only when both are read and
    neither is; otherwise INCOMPLETE.
  - Rule 2's precondition is read at the rule's own "both seeds" standard: A6@r3's cuOpt <= 133 on both seeds → the
    rule is read on A2@r3; >= 146 on both seeds → "the stall is the 3-thread footprint under ext", rule 2 does not
    apply; anything else (mixed, or between 133 and 146) → the precondition is not met at n = 2 and rule 2 does not
    apply. "Mixed" in the rule itself covers every A2 pattern that is neither >= 146 on both seeds nor <= 133 on both
    (e.g. 140 / 140). comp21-2idx is a minimisation (sense +1, checked; another sense makes rule 2 invalid); a
    completed run whose cuOpt holds no checked incumbent at T has final objective +inf (worse than any threshold).
    Thresholds compared with a 1e-6 tolerance (comp21's objective values are integers).

Validity (per run; a run that fails it is not read, and the rule on its instance says INVALID with the reason) — this
is a thread-count diagnostic, so the runner's own record, not the job, must show the layout:
  - the process layout in run.json: A6@r3 = one cuOpt process on cores [1, 2, 3] with `--cpu-threads 3`; A6F@r3 = one
    cuOpt on [1, 2, 3, 5, 6] with `--cpu-threads 5`; A2@r3 = SCIP (role solver) on [5] + cuOpt on [1, 2, 3] with
    `--cpu-threads 3` (so nothing on X925 6); no other process;
  - the harness: the run's resolved harness (run.json `config.harness`, else the run table's `harness`) equals
    `pair.resolve_harness("ext")`;
  - no compile on the clock (run.json `compiled_on_clock_cuda` / the table's `cuda_compiled_on_clock`; the runner voids
    such a run under a harness, so one in the table is a fault).

Selection: rows of `--set` x the three labels x seeds 0–1 x 120 s, not synthetic, not in a voided block; one row per
(instance, arm, seed): the bare run id before a `<run_id>#<set>` copy (build_runs.qualify_reused_ids), then the highest
attempt. Other rows of the set are listed, never read. Failed runs (`--failed`, as `tools/d022_report.py`): `exclude`
(a failed run is missing: its instance's reading is INCOMPLETE), `keep` (D-002 §6: a crash is kept and read truncated,
as build-runs scores it), `both` (default: both readings when the reading's runs include a failed one, else one). A
completed run without P(120) is missing. Interim: n = 2 seeds per cell (the note's risk line); doon is a ceiling case.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.experiments.make_jobs import R3_A2_INSTANCES, R3_BUDGET, R3_INSTANCES, R3_SEEDS, R3_SET, R3_VARIANT
from itrea.paths import DATASETS, ROOT

A6, A6F, A2 = (f"{a}@{R3_VARIANT}" for a in ("A6", "A6F", "A2"))
ARMS = (A6, A6F, A2)
SET, SEEDS, BUDGET = R3_SET, tuple(R3_SEEDS), float(R3_BUDGET)
INSTANCES, A2_INSTANCES = tuple(R3_INSTANCES), tuple(R3_A2_INSTANCES)
BAND_MIN = 0.05
LAYOUT_INSTANCES = ("neos-4763324-toguru", "neos-2746589-doon")     # rule 1's consequence
SCIP_INSTANCES = ("neos-4763324-toguru", "neos-2746589-doon")       # rule 3
COMP21, STALL_OBJ, ALONE_OBJ, OBJ_TOL = "comp21-2idx", 146.0, 133.0, 1e-6
# the note's layout for each arm: role -> (cores, cuOpt's --cpu-threads or None)
LAYOUT = {A6: {"cuopt": ([1, 2, 3], 3)},
          A6F: {"cuopt": ([1, 2, 3, 5, 6], 5)},
          A2: {"solver": ([5], None), "cuopt": ([1, 2, 3], 3)}}
REUSE_SEP = "#"


def expected_keys() -> list[tuple[str, str, int]]:
    """(instance, label, seed) of the 30 runs the rules read."""
    out = []
    for n in INSTANCES:
        for arm in (A6, A6F) + ((A2,) if n in A2_INSTANCES else ()):
            out += [(n, arm, s) for s in SEEDS]
    return out


def _fin(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _flag(v) -> bool:
    """A true flag; None and NaN (a column the run lacks) are false."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    return bool(v)


def _read(p: Path) -> dict:
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _ext() -> dict:
    from itrea.experiments.make_jobs import ITX_EXT
    from itrea.experiments.pair import resolve_harness
    return resolve_harness(ITX_EXT)


# ----------------------------------------------------------------------------------------------- selection
def candidates(runs: pd.DataFrame, set_: str = SET) -> pd.DataFrame:
    """Every row of `set_` x ARMS x SEEDS at BUDGET, not synthetic, not in a voided block, whatever its status."""
    df = runs[(runs["set"] == set_) & runs["arm"].isin(ARMS) & runs["seed"].isin(SEEDS)]
    if "budget" in df:
        df = df[np.isclose(df["budget"].astype(float), BUDGET)]
    for col in ("synthetic", "block_void"):
        if col in df:
            df = df[~df[col].fillna(False).astype(bool)]
    return df


def select(runs: pd.DataFrame, set_: str = SET, keep_failed: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """One row per (instance, arm, seed) of `candidates`, completed (and failed with `keep_failed`): the bare run id
    before a `#<set>` copy, then the highest attempt. Returns (rows, the duplicates left out)."""
    df = candidates(runs, set_)
    df = df[df["status"].isin({"completed", "failed"} if keep_failed else {"completed"})].copy()
    df["_reused"] = df["run_id"].astype(str).str.contains(REUSE_SEP, regex=False)
    df["_attempt"] = df["attempt"].fillna(0).astype(int) if "attempt" in df else 0
    df = df.sort_values(["instance", "arm", "seed", "_reused", "_attempt"], ascending=[True, True, True, True, False],
                        kind="mergesort")
    dup = df.duplicated(["instance", "arm", "seed"], keep="first")
    dropped = [dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"]), run_id=r["run_id"])
               for r in df[dup].to_dict("records")]
    return df[~dup].drop(columns=["_reused", "_attempt"]), dropped


def others(runs: pd.DataFrame, set_: str = SET) -> list[str]:
    """Rows of the set that no rule reads (other labels, seeds, budgets, or a voided block)."""
    keep = set(candidates(runs, set_)["run_id"])
    return sorted(str(r) for r in runs[(runs["set"] == set_) & ~runs["run_id"].isin(keep)]["run_id"])


# ----------------------------------------------------------------------------------------------- validity
def _threads(cmd) -> int | None:
    if isinstance(cmd, list) and "--cpu-threads" in cmd:
        i = cmd.index("--cpu-threads")
        try:
            return int(cmd[i + 1])
        except (IndexError, ValueError):
            return None
    return None


def run_problems(row: dict, root: Path, ext: dict | None = None) -> list[str]:
    """Why one run may not be read (empty = valid): its layout, its harness, a compile on the clock."""
    ext = ext if ext is not None else _ext()
    p = Path(str(row.get("run_dir") or ""))
    d = p if p.is_absolute() else root / p
    rj = _read(d / "run.json")
    if not rj:
        return [f"no run.json under {d}"]
    out = []
    want = LAYOUT[row["arm"]]
    procs = [x for x in (rj.get("processes") or []) if isinstance(x, dict)]
    roles = sorted(str(x.get("role")) for x in procs)
    if roles != sorted(want):
        out.append(f"processes {roles}, expected {sorted(want)}")
    for x in procs:
        role = str(x.get("role"))
        if role not in want:
            continue
        cores, thr = want[role]
        got = sorted(int(c) for c in (x.get("cores") or []))
        if got != cores:
            out.append(f"{role} on cores {got}, expected {cores}")
        if thr is not None and _threads(x.get("cmd")) != thr:
            out.append(f"{role} --cpu-threads {_threads(x.get('cmd'))}, expected {thr}")
    h = (rj.get("config") or {}).get("harness") if isinstance(rj.get("config"), dict) else None
    if h is None and isinstance(row.get("harness"), str) and row["harness"]:
        try:
            h = json.loads(row["harness"])
        except ValueError:
            h = None
    if h != ext:
        out.append(f"harness {h!r} is not ext {ext!r}")
    if _flag(rj.get("compiled_on_clock_cuda")) or _flag(row.get("cuda_compiled_on_clock")):
        out.append("compiled on the clock (CUDA cache)")
    return out


# ----------------------------------------------------------------------------------------------- the rules
def _pair(cells: dict, inst: str, arm: str, col: str) -> tuple[list[float] | None, str | None]:
    """The two seeds' values of `col` for (inst, arm), or (None, why)."""
    vals = []
    for s in SEEDS:
        c = cells.get((inst, arm, s))
        if c is None:
            return None, f"{arm} seed {s} missing"
        if c["problems"]:
            return None, f"{arm} seed {s} invalid: {'; '.join(c['problems'])}"
        v = c["row"].get(col)
        if not _fin(v):
            return None, f"{arm} seed {s} has no {col}"
        vals.append(float(v))
    return vals, None


def banded(a: list[float], b: list[float]) -> dict:
    """Rules 1 / 3: diff = mean(a) − mean(b), band = max(0.05, |a0 − a1|, |b0 − b1|), present iff diff > band."""
    diff = float(np.mean(a) - np.mean(b))
    band = max(BAND_MIN, abs(a[0] - a[1]), abs(b[0] - b[1]))
    return dict(a=a, b=b, mean_a=float(np.mean(a)), mean_b=float(np.mean(b)), diff=diff, band=band,
                present=bool(diff > band))


def rule1(cells: dict) -> dict:
    per = {}
    for n in INSTANCES:
        a, why = _pair(cells, n, A6, "pt")
        b, why_b = _pair(cells, n, A6F, "pt")
        why = why or why_b
        per[n] = dict(status="unread", why=why) if why else dict(status="read", **banded(a, b))
    lay = [per[n] for n in LAYOUT_INSTANCES]
    if any(x["status"] == "read" and x["present"] for x in lay):
        cons = ("QUESTION", "thread effect present on " + ", ".join(
            n for n in LAYOUT_INSTANCES if per[n]["status"] == "read" and per[n]["present"]) +
            ": the layout question (a cuOpt thread on X925 6) goes to Berk with these numbers")
    elif all(x["status"] == "read" for x in lay):
        cons = ("CLOSED", "no thread effect on toguru or doon: the layout line is closed for v1")
    else:
        cons = ("INCOMPLETE", "not read on " + ", ".join(n for n in LAYOUT_INSTANCES if per[n]["status"] != "read"))
    return dict(per=per, consequence=cons[0], text=cons[1])


def _objs(cells: dict, arm: str) -> tuple[list[float] | None, str | None]:
    """cuOpt's final objective on comp21 per seed (+inf for a run whose cuOpt holds no checked incumbent at T)."""
    vals = []
    for s in SEEDS:
        c = cells.get((COMP21, arm, s))
        if c is None:
            return None, f"{arm} seed {s} missing"
        if c["problems"]:
            return None, f"{arm} seed {s} invalid: {'; '.join(c['problems'])}"
        sense = c["row"].get("sense")
        if not _fin(sense) or int(sense) != 1:
            return None, f"{arm} seed {s}: objective sense {sense!r} is not minimisation (+1)"
        v = c["row"].get("cuopt_best_obj")
        vals.append(float(v) if _fin(v) else math.inf)
    return vals, None


def _band_of(v: list[float]) -> str:
    if all(x >= STALL_OBJ - OBJ_TOL for x in v):
        return "stall"
    if all(x <= ALONE_OBJ + OBJ_TOL for x in v):
        return "alone"
    return "mixed"


def rule2(cells: dict) -> dict:
    a6, why = _objs(cells, A6)
    out = dict(a6=a6, a2=None, a6f=_objs(cells, A6F)[0])
    if why:
        return dict(out, status="INCOMPLETE", text=f"precondition not read: {why}")
    pre = _band_of(a6)
    if pre == "stall":
        return dict(out, status="NOT APPLIED", text="A6@r3's own cuOpt ends >= 146 on both seeds: the stall is the "
                                                    "3-thread footprint under ext; rule 2 does not apply")
    if pre == "mixed":
        return dict(out, status="NOT APPLIED", text="precondition not met: A6@r3's own cuOpt did not reach <= 133 on "
                                                    "both seeds (nor end >= 146 on both); rule 2 does not apply "
                                                    "(inconclusive at n = 2)")
    a2, why = _objs(cells, A2)
    out["a2"] = a2
    if why:
        return dict(out, status="INCOMPLETE", text=f"precondition met; A2@r3 not read: {why}")
    v = _band_of(a2)
    if v == "stall":
        return dict(out, status="SCIP", text="cuOpt in A2@r3 >= 146 on both seeds: SCIP beside cuOpt is the cause")
    if v == "alone":
        return dict(out, status="EA CONTEXT", text="cuOpt in A2@r3 <= 133 on both seeds: the EA's CUDA context "
                                                   "remains, and the MPS question (a host change, Berk's) becomes "
                                                   "concrete")
    return dict(out, status="MIXED", text="cuOpt in A2@r3 mixed: inconclusive at n = 2")


def rule3(cells: dict) -> dict:
    per = {}
    for n in SCIP_INSTANCES:
        a, why = _pair(cells, n, A2, "cuopt_pt")
        b, why_b = _pair(cells, n, A6, "cuopt_pt")
        why = why or why_b
        per[n] = dict(status="unread", why=why) if why else dict(status="read", **banded(a, b))
    return dict(per=per)


def reading(runs: pd.DataFrame, root: Path, set_: str = SET, keep_failed: bool = False, ext: dict | None = None) -> dict:
    sel, dropped = select(runs, set_, keep_failed)
    ext = ext if ext is not None else _ext()
    cells, missing = {}, []
    by = {(r["instance"], r["arm"], int(r["seed"])): r for r in sel.to_dict("records")}
    for k in expected_keys():
        r = by.get(k)
        if r is None:
            missing.append(k)
            continue
        cells[k] = dict(row=r, problems=run_problems(r, root, ext))
    invalid = {f"{k[0]} {k[1]} s{k[2]}": c["problems"] for k, c in cells.items() if c["problems"]}
    return dict(keep_failed=keep_failed, n_expected=len(expected_keys()), n_present=len(cells),
                missing=[f"{i} {a} s{s}" for i, a, s in missing], invalid=invalid, duplicates=dropped,
                rule1=rule1(cells), rule2=rule2(cells), rule3=rule3(cells), cells=cells)


def report(runs: pd.DataFrame, root: Path = ROOT, set_: str = SET, failed: str = "both") -> dict:
    cand = candidates(runs, set_)
    fails = sorted(str(r) for r in cand[cand["status"] == "failed"]["run_id"])
    modes = {"exclude": [False], "keep": [True], "both": [False, True] if fails else [False]}[failed]
    ext = _ext()
    return dict(set=set_, failed_runs=fails, other_rows=others(runs, set_),
                readings=[reading(runs, root, set_, k, ext) for k in modes])


# ----------------------------------------------------------------------------------------------- output
def _f(v, nd=3) -> str:
    return "inf" if v == math.inf else (f"{v:.{nd}f}" if _fin(v) else "–")


def _obj(v) -> str:
    return "none" if v is None else " / ".join("no incumbent" if x == math.inf else f"{x:g}" for x in v)


def render(rep: dict) -> str:
    L = [f"R3 report (C1 diagnosis note §5 R3), set {rep['set']}: rules fixed in the note before any run; interim, "
         f"n = 2 seeds per cell"]
    if rep["failed_runs"]:
        L.append(f"failed runs in the reading's keys: {', '.join(rep['failed_runs'])}")
    if rep["other_rows"]:
        L.append(f"rows of the set no rule reads: {', '.join(rep['other_rows'])}")
    for rd in rep["readings"]:
        L.append("")
        L.append(f"== reading: failed runs {'kept (D-002 §6)' if rd['keep_failed'] else 'excluded (missing)'} ==")
        L.append(f"completeness: {rd['n_present']} of {rd['n_expected']} runs present"
                 + (f"; missing: {', '.join(rd['missing'])}" if rd["missing"] else ""))
        for k, v in rd["invalid"].items():
            L.append(f"INVALID run {k}: {'; '.join(v)}")
        for d in rd["duplicates"]:
            L.append(f"duplicate left out: {d['run_id']}")
        L.append("per run (P(120) = pt; cuOpt P = cuopt_pt; cuOpt final objective = cuopt_best_obj):")
        for k, c in sorted(rd["cells"].items()):
            r = c["row"]
            L.append(f"  {k[0]:<28} {k[1]:<7} s{k[2]}  P {_f(r.get('pt'))}  cuOpt P {_f(r.get('cuopt_pt'))}  "
                     f"cuOpt obj {_f(r.get('cuopt_best_obj'), 4)}  {r.get('status')}")
        r1 = rd["rule1"]
        L.append("Rule 1 (thread effect: mean P(A6@r3) − mean P(A6F@r3) > max(0.05, larger seed spread)):")
        for n, x in r1["per"].items():
            if x["status"] != "read":
                L.append(f"  {n}: not read ({x['why']})")
                continue
            word = "PRESENT" if x["present"] else ("absent (A6@r3 lower)" if x["diff"] < -x["band"] else "absent")
            L.append(f"  {n}: A6 {_f(x['a'][0])} / {_f(x['a'][1])} (mean {_f(x['mean_a'])}), A6F {_f(x['b'][0])} / "
                     f"{_f(x['b'][1])} (mean {_f(x['mean_b'])}); diff {x['diff']:+.3f}, band {x['band']:.3f} -> {word}")
        L.append(f"  -> {r1['consequence']}: {r1['text']}")
        r2 = rd["rule2"]
        L.append(f"Rule 2 (comp21 part 2; cuOpt's final objective; stall >= {STALL_OBJ:g}, alone <= {ALONE_OBJ:g}):")
        L.append(f"  A6@r3 {_obj(r2['a6'])}; A2@r3 {_obj(r2['a2'])}; A6F@r3 (reported) {_obj(r2['a6f'])}")
        L.append(f"  -> {r2['status']}: {r2['text']}")
        L.append("Rule 3 (SCIP's presence: mean cuOpt P(A2@r3) − mean cuOpt P(A6@r3) > max(0.05, larger seed spread)):")
        for n, x in rd["rule3"]["per"].items():
            if x["status"] != "read":
                L.append(f"  {n}: not read ({x['why']})")
                continue
            word = "PRESENT" if x["present"] else ("absent (A2's cuOpt lower)" if x["diff"] < -x["band"] else "absent")
            L.append(f"  {n}: A2 cuOpt {_f(x['a'][0])} / {_f(x['a'][1])}, A6 {_f(x['b'][0])} / {_f(x['b'][1])}; "
                     f"diff {x['diff']:+.3f}, band {x['band']:.3f} -> {word}")
    return "\n".join(L)


def _jsonable(o):
    if isinstance(o, dict):
        return {(" | ".join(map(str, k)) if isinstance(k, tuple) else str(k)): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None if math.isnan(o) else ("inf" if o > 0 else "-inf")
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return _jsonable(o.item())
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="r3_report", description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=DATASETS / "runs.parquet")
    ap.add_argument("--root", type=Path, default=ROOT, help="the run directories' root (run_dir is relative to it)")
    ap.add_argument("--set", default=SET)
    ap.add_argument("--failed", choices=("both", "exclude", "keep"), default="both")
    ap.add_argument("--json", type=Path, help="write the report as JSON here too")
    a = ap.parse_args(argv)
    try:
        runs = pd.read_parquet(a.runs)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    rep = report(runs, a.root, a.set, a.failed)
    print(render(rep))
    if a.json:
        out = dict(rep, readings=[{k: v for k, v in rd.items() if k != "cells"} | dict(
            runs={f"{i} {arm} s{s}": dict(run_id=c["row"]["run_id"], pt=c["row"].get("pt"),
                                          cuopt_pt=c["row"].get("cuopt_pt"),
                                          cuopt_best_obj=c["row"].get("cuopt_best_obj"), status=c["row"].get("status"),
                                          problems=c["problems"])
                  for (i, arm, s), c in rd["cells"].items()}) for rd in rep["readings"]])
        a.json.write_text(json.dumps(_jsonable(out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
