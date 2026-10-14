"""D-027 report: the same-day diagnostic of cuOpt's thread count and SCIP's presence (R3 of `vault/30-results/2026-10-14
C1 diagnosis and improvement candidates.md` §5) — cuOpt alone with 3 threads on A725 1–3 (A6@ext) against cuOpt alone
on F with 5 threads (A6F@ext), and SCIP on X925 5 beside the same 3-thread cuOpt (A2@ext, no EA, X925 6 idle), all
under the ext harness. A diagnostic, not a lever: no hint and no adoption. Its validity and three reading rules are
D-027's, fixed before any run (the note's rules, made exact there); nothing is chosen by eye.

Usage (repository root, after `itrea build-runs`; untimed analysis: OMP_NUM_THREADS=1, never beside timed runs with
default thread counts):

    OMP_NUM_THREADS=1 .venv/bin/python tools/d027_report.py [--runs datasets/runs.parquet] [--root .] [--set r3-diag]
                                                            [--failed both|exclude|keep] [--json out]
    itrea d027-report …                       (the same, through the entry point)

Runs (`make-jobs r3-diag` -> jobs/r3-diag.jsonl, set r3-diag, 120 s, seeds 0–1, 16 runs — only what a rule reads,
D-027's amendment of 2026-10-14 before any run, D-024 rule 4): R3_INSTANCES (comp21-2idx, neos-4763324-toguru,
neos-2746589-doon) x A6@r3, R3_A6F_INSTANCES (toguru, doon) x A6F@r3 and R3_A2_INSTANCES (comp21-2idx, toguru, doon) x
A2@r3; instances and labels come from `make_jobs`, so the report and the job file cannot drift apart. Reads `datasets/runs.parquet` — `pt` (P(120) on the call clock), `cuopt_pt` (P(120) of
cuOpt's own checked incumbents), `cuopt_best_obj` (the objective of cuOpt's own last checked incumbent at 120 s),
`sense`, `status`, `harness`, `cuda_compiled_on_clock`, `run_dir` — and each run's `run.json` (`processes[*].role /
cores / cmd`, `config.harness`, `compiled_on_clock_cuda`).

Validity (D-027: must hold before any rule is read; otherwise INVALID, and the report names the leg):
  (a) complete: the 16 runs present and completed, each with P(120) (a void is moved aside by the pool and its rerun is
      the run read; a block skipped after its void leaves runs missing);
  (b) the harness: every run's resolved harness (run.json `config.harness`, else the run table's `harness`) equals
      `pair.resolve_harness("ext")`;
  (c) the layout, from each run's run.json: cuOpt with `--cpu-threads 3` on cores [1, 2, 3] in A6@r3 and A2@r3, with
      `--cpu-threads 5` on [1, 2, 3, 5, 6] in A6F@r3; SCIP (role solver) on [5] in A2@r3; no process on core 6 in A6@r3
      or A2@r3; no EA process (and no other process) in any run;
  (d) no compile on the clock (run.json `compiled_on_clock_cuda`, the table's `cuda_compiled_on_clock`).
Failed runs (`--failed`): `exclude` is the letter — a failed run is not completed, so (a) fails and the reading is
INVALID; `keep` reads a failed run as D-002 §6 keeps a crash (truncated at the crash, as build-runs scores it), a
stated deviation from (a) that the report names; `both` (default) prints both readings when the set holds a failed run
among the 16, else one.

Quantities (D-027): P = P(120) on the call clock; in A6@ext and A6F@ext the arm's P is cuOpt's (`pt`); in A2@ext "P of
cuOpt" is `cuopt_pt`, not the portfolio's; a seed spread is |P_s0 − P_s1| in raw P.

Reading rules (D-027, fixed before the runs):
  1. Thread effect on instance i is present iff P̄(A6@ext) − P̄(A6F@ext) > max(0.05, the larger seed spread of either
     arm), on toguru and doon (the amendment runs A6F only there). Present on toguru or doon → the layout question (a
     cuOpt thread on X925 6) goes to Berk with these numbers; otherwise the layout line is closed for v1.
  2. comp21's part 2, from cuOpt's final objective in A2@ext: >= 146 on both seeds → SCIP beside cuOpt is the cause (no
     question for Berk; the MPS question is not raised); <= 133 on both seeds → the EA's presence remains (D-027's
     clarification of 2026-10-14: its CUDA context, residual GPU work, the send callback, host CPU and memory traffic on
     core 6, SCIP's inputs from the EA — A2 sends nothing), and the MPS question (a host change, Berk's) goes to Berk
     with that wording, MPS being one candidate, not an attribution; anything else (mixed, or between 133 and 146) →
     inconclusive at n = 2. 146 is the lower of the two ext reference runs inside A9 (A9@r13x 149 / 146), so the rule
     leans towards the "<= 133" and "inconclusive" outcomes (stated in the clarification). Precondition: A6@ext's own
     cuOpt reaches <= 133 on both seeds; >= 146 on both seeds → the stall is the 3-thread footprint under ext, rule 2
     does not apply (quoted beside the layout question if rule 1 sends it); anything else → the precondition is not
     met, rule 2 is not read, comp21's part 2
     stays inconclusive at n = 2.
  3. SCIP's presence on toguru and doon: present iff P̄_cuOpt(A2@ext) − P̄(A6@ext) > max(0.05, the larger seed spread of
     either arm). Present → recorded as a cause of cuOpt's slowdown on that instance; absent → ruled out there at n = 2.
     Either way it names no lever and asks Berk nothing.
Reported beside, read by no rule: A6F@r3 against A6F@r13x of 10-13 (set itx-r13, the same job and run hash) on toguru
and doon — a day-to-day replicate of cuOpt alone on F. What goes to Berk: only rule 1 present on toguru or doon (the layout
exception) and rule 2's "<= 133 on both seeds" with its precondition met (CUDA MPS); every other outcome asks nothing.

Readings this report adds where D-027 is silent (written with the code, 2026-10-14, before any run): rules 1 and 3 are
one-sided as written ("present" only when the 3-thread / SCIP-beside arm is worse, i.e. higher P, by more than the
band; the other direction is printed as a number); rule 1's "goes to Berk" fires as soon as toguru or doon is present;
comp21-2idx is a minimisation (sense +1, checked; another sense leaves rule 2 not read); a run whose cuOpt holds no
checked incumbent at 120 s has final objective +inf; the thresholds are compared with a 1e-6 tolerance (comp21's
objective values are integers). Selection: rows of `--set` x the three labels x seeds 0–1 x 120 s, not synthetic, not
in a voided block; one row per (instance, arm, seed): the bare run id before a `<run_id>#<set>` copy
(build_runs.qualify_reused_ids), then the highest attempt; other rows of the set are listed, never read. Interim: n = 2
seeds per cell; doon is a ceiling case.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.experiments.make_jobs import (R3_A2_INSTANCES, R3_A6F_INSTANCES, R3_BUDGET, R3_INSTANCES, R3_SEEDS,
                                         R3_SET, R3_VARIANT)
from itrea.paths import DATASETS, ROOT

A6, A6F, A2 = (f"{a}@{R3_VARIANT}" for a in ("A6", "A6F", "A2"))
ARMS = (A6, A6F, A2)
SET, SEEDS, BUDGET = R3_SET, tuple(R3_SEEDS), float(R3_BUDGET)
INSTANCES, A6F_INSTANCES, A2_INSTANCES = tuple(R3_INSTANCES), tuple(R3_A6F_INSTANCES), tuple(R3_A2_INSTANCES)
REPLICATE_SET, REPLICATE_ARM = "itx-r13", "A6F@r13x"       # reported beside A6F@r3, read by no rule
BAND_MIN = 0.05
LAYOUT_INSTANCES = ("neos-4763324-toguru", "neos-2746589-doon")     # rule 1's consequence
SCIP_INSTANCES = ("neos-4763324-toguru", "neos-2746589-doon")       # rule 3
COMP21, STALL_OBJ, ALONE_OBJ, OBJ_TOL = "comp21-2idx", 146.0, 133.0, 1e-6
# D-027's layout per arm: role -> (cores, cuOpt's --cpu-threads or None)
LAYOUT = {A6: {"cuopt": ([1, 2, 3], 3)},
          A6F: {"cuopt": ([1, 2, 3, 5, 6], 5)},
          A2: {"solver": ([5], None), "cuopt": ([1, 2, 3], 3)}}
IDLE_CORE, IDLE_ARMS = 6, (A6, A2)                                   # X925 6 stays idle in A6@ext and A2@ext
LEGS = ("complete", "harness", "layout", "compile")
REUSE_SEP = "#"


def expected_keys() -> list[tuple[str, str, int]]:
    """(instance, label, seed) of the 16 runs the rules read."""
    out = []
    for n in INSTANCES:
        for arm in (A6,) + ((A6F,) if n in A6F_INSTANCES else ()) + ((A2,) if n in A2_INSTANCES else ()):
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
def candidates(runs: pd.DataFrame, set_: str = SET, arms=ARMS) -> pd.DataFrame:
    """Every row of `set_` x `arms` x SEEDS at BUDGET, not synthetic, not in a voided block, whatever its status."""
    df = runs[(runs["set"] == set_) & runs["arm"].isin(list(arms)) & runs["seed"].isin(SEEDS)]
    if "budget" in df:
        df = df[np.isclose(df["budget"].astype(float), BUDGET)]
    for col in ("synthetic", "block_void"):
        if col in df:
            df = df[~df[col].fillna(False).astype(bool)]
    return df


def select(runs: pd.DataFrame, set_: str = SET, keep_failed: bool = False,
           arms=ARMS) -> tuple[pd.DataFrame, list[dict]]:
    """One row per (instance, arm, seed) of `candidates`, completed (and failed with `keep_failed`): the bare run id
    before a `#<set>` copy, then the highest attempt. Returns (rows, the duplicates left out)."""
    df = candidates(runs, set_, arms)
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


def run_problems(row: dict, root: Path, ext: dict | None = None) -> dict[str, list[str]]:
    """{leg: problems} of one run (legs harness, layout, compile; empty = valid), read from its run.json."""
    ext = ext if ext is not None else _ext()
    p = Path(str(row.get("run_dir") or ""))
    d = p if p.is_absolute() else root / p
    rj = _read(d / "run.json")
    if not rj:
        return {"layout": [f"no run.json under {d}"]}
    out: dict[str, list[str]] = {}
    want = LAYOUT[row["arm"]]
    procs = [x for x in (rj.get("processes") or []) if isinstance(x, dict)]
    roles = sorted(str(x.get("role")) for x in procs)
    lay = []
    if roles != sorted(want):
        lay.append(f"processes {roles}, expected {sorted(want)}" + (" (an EA process)" if "ea" in roles else ""))
    for x in procs:
        role = str(x.get("role"))
        got = sorted(int(c) for c in (x.get("cores") or []))
        if row["arm"] in IDLE_ARMS and IDLE_CORE in got:
            lay.append(f"{role} on core {IDLE_CORE}, which stays idle in {row['arm']}")
        if role not in want:
            continue
        cores, thr = want[role]
        if got != cores:
            lay.append(f"{role} on cores {got}, expected {cores}")
        if thr is not None and _threads(x.get("cmd")) != thr:
            lay.append(f"{role} --cpu-threads {_threads(x.get('cmd'))}, expected {thr}")
    if lay:
        out["layout"] = lay
    h = (rj.get("config") or {}).get("harness") if isinstance(rj.get("config"), dict) else None
    if h is None and isinstance(row.get("harness"), str) and row["harness"]:
        try:
            h = json.loads(row["harness"])
        except ValueError:
            h = None
    if h != ext:
        out["harness"] = [f"harness {h!r} is not ext"]
    if _flag(rj.get("compiled_on_clock_cuda")) or _flag(row.get("cuda_compiled_on_clock")):
        out["compile"] = ["compiled on the clock (CUDA cache)"]
    return out


def validity(cells: dict, missing: list, no_p: list, kept_failed: list, failed=()) -> dict:
    """D-027's validity over the 16 runs: {leg: [problems]} and the verdict VALID / INVALID. `failed`: the missing keys
    whose run failed (excluded by the letter)."""
    legs: dict[str, list[str]] = {k: [] for k in LEGS}
    legs["complete"] = [f"{'failed (not completed)' if k in failed else 'missing'} {k[0]} {k[1]} s{k[2]}"
                        for k in missing] + [f"no P(120): {i} {a} s{s}" for i, a, s in no_p]
    for (i, a, s), c in cells.items():
        for leg, probs in c["problems"].items():
            legs[leg] += [f"{i} {a} s{s}: {x}" for x in probs]
    bad = [k for k in LEGS if legs[k]]
    return dict(verdict="INVALID" if bad else "VALID", failing=bad, legs=legs,
                deviation=[f"{i} {a} s{s} failed and is read as D-002 §6 keeps a crash (a stated deviation from leg "
                           f"(a): 16 of 16 completed)" for i, a, s in kept_failed])


# ----------------------------------------------------------------------------------------------- the rules
def _two(cells: dict, inst: str, arm: str, col: str) -> list[float]:
    return [float(cells[(inst, arm, s)]["row"][col]) for s in SEEDS]


def banded(a: list[float], b: list[float]) -> dict:
    """Rules 1 / 3: diff = mean(a) − mean(b), band = max(0.05, |a0 − a1|, |b0 − b1|), present iff diff > band."""
    diff = float(np.mean(a) - np.mean(b))
    band = max(BAND_MIN, abs(a[0] - a[1]), abs(b[0] - b[1]))
    return dict(a=a, b=b, mean_a=float(np.mean(a)), mean_b=float(np.mean(b)), diff=diff, band=band,
                present=bool(diff > band))


def rule1(cells: dict) -> dict:
    per = {n: banded(_two(cells, n, A6, "pt"), _two(cells, n, A6F, "pt")) for n in A6F_INSTANCES}
    hit = [n for n in LAYOUT_INSTANCES if per[n]["present"]]
    if hit:
        return dict(per=per, consequence="QUESTION", text=f"thread effect present on {', '.join(hit)}: the layout "
                    "question (a cuOpt thread on X925 6) goes to Berk with these numbers")
    return dict(per=per, consequence="CLOSED", text="no thread effect on toguru or doon: the layout line is closed for "
                "v1")


def _objs(cells: dict, arm: str) -> list[float]:
    """cuOpt's final objective on comp21 per seed (+inf for a run whose cuOpt holds no checked incumbent at 120 s)."""
    return [float(v) if _fin(v) else math.inf for v in (cells[(COMP21, arm, s)]["row"].get("cuopt_best_obj")
                                                         for s in SEEDS)]


def _band_of(v: list[float]) -> str:
    if all(x >= STALL_OBJ - OBJ_TOL for x in v):
        return "stall"
    if all(x <= ALONE_OBJ + OBJ_TOL for x in v):
        return "alone"
    return "mixed"


def rule2(cells: dict) -> dict:
    a6 = _objs(cells, A6)
    out = dict(a6=a6, a2=None)
    senses = {(a, s): cells[(COMP21, a, s)]["row"].get("sense") for a in (A6, A2) for s in SEEDS}
    wrong = {k: v for k, v in senses.items() if not _fin(v) or int(v) != 1}
    if wrong:
        return dict(out, status="NOT READ", text=f"comp21's objective sense is not minimisation (+1) in "
                                                 f"{sorted(f'{a} s{s}' for a, s in wrong)}")
    pre = _band_of(a6)
    if pre == "stall":
        return dict(out, status="FOOTPRINT", text="A6@r3's own cuOpt ends >= 146 on both seeds: the stall is the "
                                                  "3-thread footprint under ext; rule 2 does not apply (quoted beside "
                                                  "the layout question if rule 1 sends it)")
    if pre == "mixed":
        return dict(out, status="PRECONDITION NOT MET", text="A6@r3's own cuOpt did not reach <= 133 on both seeds "
                                                             "(nor end >= 146 on both): rule 2 is not read; comp21's "
                                                             "part 2 stays inconclusive at n = 2")
    a2 = _objs(cells, A2)
    out["a2"] = a2
    v = _band_of(a2)
    if v == "stall":
        return dict(out, status="SCIP", text="cuOpt in A2@r3 >= 146 on both seeds: SCIP beside cuOpt is the cause (no "
                                             "question for Berk; the MPS question is not raised)")
    if v == "alone":
        return dict(out, status="EA PRESENCE", text="cuOpt in A2@r3 <= 133 on both seeds: the EA's presence remains "
                                                    "(its CUDA context, residual GPU work, the send callback, host CPU "
                                                    "and memory traffic on core 6, SCIP's inputs from the EA); the MPS "
                                                    "question (a host change, Berk's) goes to Berk with these numbers, "
                                                    "MPS being one candidate, not an attribution")
    return dict(out, status="INCONCLUSIVE", text="cuOpt in A2@r3 neither >= 146 nor <= 133 on both seeds: "
                                                 "inconclusive at n = 2")


def rule3(cells: dict) -> dict:
    per = {}
    for n in SCIP_INSTANCES:
        x = banded(_two(cells, n, A2, "cuopt_pt"), _two(cells, n, A6, "pt"))
        x["text"] = (f"SCIP's presence recorded as a cause of cuOpt's slowdown on {n}" if x["present"]
                     else f"SCIP's presence ruled out on {n} at n = 2")
        per[n] = x
    return dict(per=per)


def to_berk(r1: dict, r2: dict) -> list[str]:
    """D-027: only the layout exception (rule 1 on toguru or doon) and CUDA MPS (rule 2 <= 133, precondition met)."""
    out = []
    if r1["consequence"] == "QUESTION":
        out.append("the layout question (a cuOpt thread on X925 6): " + r1["text"].split(":")[0]
                   + ("; with rule 2's footprint reading on comp21" if r2["status"] == "FOOTPRINT" else ""))
    if r2["status"] == "EA PRESENCE":
        out.append("the MPS question (a host change): cuOpt in A2@r3 <= 133 on both seeds on comp21")
    return out


def replicate(runs: pd.DataFrame, cells: dict) -> dict:
    """A6F@r3 against A6F@r13x of 10-13 on toguru and doon (reported beside, read by no rule): {instance: P per
    seed}."""
    old, _ = select(runs, REPLICATE_SET, False, (REPLICATE_ARM,))
    by = {(r["instance"], int(r["seed"])): r.get("pt") for r in old.to_dict("records")}
    out = {}
    for n in A6F_INSTANCES:
        now = [cells[(n, A6F, s)]["row"].get("pt") if (n, A6F, s) in cells else None for s in SEEDS]
        out[n] = dict(r3=now, r13x=[by.get((n, s)) for s in SEEDS])
    return out


def reading(runs: pd.DataFrame, root: Path, set_: str = SET, keep_failed: bool = False,
            ext: dict | None = None) -> dict:
    sel, dropped = select(runs, set_, keep_failed)
    ext = ext if ext is not None else _ext()
    by = {(r["instance"], r["arm"], int(r["seed"])): r for r in sel.to_dict("records")}
    cells, missing, no_p, kept = {}, [], [], []
    for k in expected_keys():
        r = by.get(k)
        if r is None:
            missing.append(k)
            continue
        if not _fin(r.get("pt")) or (k[1] == A2 and not _fin(r.get("cuopt_pt"))):
            no_p.append(k)
        if r.get("status") == "failed":
            kept.append(k)
        cells[k] = dict(row=r, problems=run_problems(r, root, ext))
    cand = candidates(runs, set_)
    failed = {(r["instance"], r["arm"], int(r["seed"])) for r in cand[cand["status"] == "failed"].to_dict("records")}
    val = validity(cells, missing, no_p, kept, failed)
    out = dict(keep_failed=keep_failed, n_expected=len(expected_keys()), n_present=len(cells), validity=val,
               duplicates=dropped, replicate=replicate(runs, cells), cells=cells, rule1=None, rule2=None, rule3=None,
               to_berk=None)
    if val["verdict"] == "VALID":
        out.update(rule1=rule1(cells), rule2=rule2(cells), rule3=rule3(cells))
        out["to_berk"] = to_berk(out["rule1"], out["rule2"])
    return out


def report(runs: pd.DataFrame, root: Path = ROOT, set_: str = SET, failed: str = "both") -> dict:
    cand = candidates(runs, set_)
    fails = sorted(str(r) for r in cand[cand["status"] == "failed"]["run_id"])
    modes = {"exclude": [False], "keep": [True], "both": [False, True] if fails else [False]}[failed]
    ext = _ext()
    return dict(set=set_, failed_runs=fails, other_rows=others(runs, set_),
                readings=[reading(runs, root, set_, k, ext) for k in modes])


# ----------------------------------------------------------------------------------------------- output
def _f(v, nd=3) -> str:
    return "inf" if v == math.inf else (f"{float(v):.{nd}f}" if _fin(v) else "–")


def _obj(v) -> str:
    return "–" if v is None else " / ".join("no incumbent" if x == math.inf else f"{x:g}" for x in v)


def render(rep: dict) -> str:
    L = [f"D-027 report (C1 diagnosis R3), set {rep['set']}: rules fixed in D-027 before any run; interim, n = 2 seeds "
         f"per cell; a diagnostic — no hint, no adoption"]
    if rep["failed_runs"]:
        L.append(f"failed runs among the 16: {', '.join(rep['failed_runs'])}")
    if rep["other_rows"]:
        L.append(f"rows of the set no rule reads: {', '.join(rep['other_rows'])}")
    for rd in rep["readings"]:
        L.append("")
        L.append(f"== reading: failed runs {'kept (D-002 §6)' if rd['keep_failed'] else 'excluded (the letter)'} ==")
        v = rd["validity"]
        L.append(f"VALIDITY {v['verdict']}: {rd['n_present']} of {rd['n_expected']} runs present"
                 + (f"; failing leg(s): {', '.join(v['failing'])}" if v["failing"] else ""))
        for leg in LEGS:
            for x in v["legs"][leg]:
                L.append(f"  ({leg}) {x}")
        for x in v["deviation"]:
            L.append(f"  stated deviation: {x}")
        for d in rd["duplicates"]:
            L.append(f"  duplicate left out: {d['run_id']}")
        L.append("per run (P(120) = pt; cuOpt P = cuopt_pt; cuOpt obj = cuopt_best_obj, cuOpt's last incumbent at "
                 "120 s):")
        for k, c in sorted(rd["cells"].items()):
            r = c["row"]
            L.append(f"  {k[0]:<28} {k[1]:<7} s{k[2]}  P {_f(r.get('pt'))}  cuOpt P {_f(r.get('cuopt_pt'))}  "
                     f"cuOpt obj {_f(r.get('cuopt_best_obj'), 4)}  {r.get('status')}")
        if v["verdict"] != "VALID":
            L.append("RULES NOT READ: validity fails (D-027: it must hold before any rule is read)")
        else:
            r1 = rd["rule1"]
            L.append("Rule 1 (thread effect: P̄(A6@r3) − P̄(A6F@r3) > max(0.05, the larger raw seed spread)):")
            for n, x in r1["per"].items():
                word = "PRESENT" if x["present"] else ("absent (A6@r3 lower)" if x["diff"] < -x["band"] else "absent")
                L.append(f"  {n}: A6 {_f(x['a'][0])} / {_f(x['a'][1])} (mean {_f(x['mean_a'])}), A6F {_f(x['b'][0])} "
                         f"/ {_f(x['b'][1])} (mean {_f(x['mean_b'])}); diff {x['diff']:+.3f}, band {x['band']:.3f} "
                         f"-> {word}")
            L.append(f"  -> {r1['consequence']}: {r1['text']}")
            r2 = rd["rule2"]
            L.append(f"Rule 2 (comp21's part 2; cuOpt's final objective; >= {STALL_OBJ:g} / <= {ALONE_OBJ:g} on both "
                     f"seeds):")
            L.append(f"  A6@r3 {_obj(r2['a6'])}; A2@r3 {_obj(r2['a2'])}")
            L.append(f"  -> {r2['status']}: {r2['text']}")
            L.append("Rule 3 (SCIP's presence: P̄_cuOpt(A2@r3) − P̄(A6@r3) > max(0.05, the larger raw seed spread)):")
            for n, x in rd["rule3"]["per"].items():
                word = "PRESENT" if x["present"] else ("absent (A2's cuOpt lower)" if x["diff"] < -x["band"]
                                                       else "absent")
                L.append(f"  {n}: A2 cuOpt {_f(x['a'][0])} / {_f(x['a'][1])}, A6 {_f(x['b'][0])} / {_f(x['b'][1])}; "
                         f"diff {x['diff']:+.3f}, band {x['band']:.3f} -> {word}: {x['text']}")
            L.append("To Berk: " + ("; ".join(rd["to_berk"]) if rd["to_berk"] else "nothing (every outcome is recorded "
                                                                                    "in the Development log and the "
                                                                                    "note's §7)"))
        L.append(f"Reported beside, read by no rule — A6F@r3 against {REPLICATE_ARM} (set {REPLICATE_SET}, 10-13):")
        for n, x in rd["replicate"].items():
            L.append(f"  {n}: r3 {' / '.join(_f(p) for p in x['r3'])}, r13x {' / '.join(_f(p) for p in x['r13x'])}")
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
    ap = argparse.ArgumentParser(prog="d027_report", description=__doc__.split("\n\n")[0])
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
