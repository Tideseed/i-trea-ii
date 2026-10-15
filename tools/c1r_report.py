"""C1 re-read report (D-030): C1 read again on the frozen-v2 candidate — D-022's C1 runs (the 17 development +
class-probe instances x {A9, A6F} x seeds 0-1, 120 s, harness ext) re-run in one set `c1r-<a9>` with A9 on the
candidate's configuration (D-026 ADOPT: `ld0`; or `ysync-ld0` if D-029 adopts ysync and Berk approves the
combination) and A6F re-run in the same window (Berk 2026-10-14, D-025's open question 1: "Re-run C1 on frozen hash").
The rules are D-022's (C1's hint and guard, its second amendment of 2026-10-13) and D-025 amendment 3's (the harm leg),
fixed before any run; D-030 records them. Nothing is chosen by eye, and nothing here decides C1: the go / no-go is
Berk's (D-019 addendum).

Usage (repository root, after `itrea build-runs`; untimed analysis: OMP_NUM_THREADS=1, never beside timed runs with
default thread counts):

    OMP_NUM_THREADS=1 .venv/bin/python tools/c1r_report.py --a9 ld0|ysync-ld0 [--set c1r-<a9>]
                                                           [--failed both|exclude|keep] [--json out]
    itrea c1r-report …                        (the same, through the entry point)

`--a9` (and `--set`) are the arguments the job file was written with (`make-jobs c1r-rerun --a9 …`): the expected runs,
labels and run ids come from `make_jobs.c1_rerun`, whose run hashes are pinned (A9@c1r 341267810eed for ld0,
f345de0ab936 for ysync-ld0; A6F@c1r 82a371305f88 = A6F@r13x's), so the report and the job file cannot drift apart.

Reads `datasets/runs.parquet` (`pt` = P(120), `status`, `attempt`, `run_dir`, `harness`, `cuda_compiled_on_clock`,
`window_id`, `block_void`), each run's `run.json` (`config.harness`, `compiled_on_clock_cuda`, `cupy_cache`) and, for
A9, `ea.summary.json` (`compiled_on_clock`; beside: `cuda_sched`, `phase_events_block`, `lp_decode`, `generations`).
Also from the run table: D-022's runs of set `itx-r13` (A9@r13x, A6F@r13x: the 10-13 reading and the harm leg's
reference) and A0's runs of sets `dev` / `classprobe` (seeds 0-1, 120 s; vs SCIP alone, reported).

Validity (gating; INVALID names the leg and no rule is read; a missing run alone -> INCOMPLETE, the rules read what is
there, each INCOMPLETE where it needs a missing run):
  - complete: 68 of 68 (17 x {A9@c1r, A6F@c1r} x seeds 0-1), each with P(120);
  - harness: every run's resolved harness (run.json `config.harness`, else the table's) is `pair.resolve_harness("ext")`;
  - config: each run's id is its job's (the run hash carries the EA configuration hash, the harness and the
    footprint) — the configurations as the job file;
  - compile: no compile on the clock — the project's CUDA cache on every run (run.json `compiled_on_clock_cuda`, the
    table's `cuda_compiled_on_clock`) and, on A9 (the arm with an EA GPU process), the CuPy cache (run.json
    `cupy_cache.compiled_on_clock`, the EA summary's `compiled_on_clock`); A6F is never charged with a CuPy flag
    (the runner's own rule, as D-022's report reads it);
  - window: the two runs of each block (instance x seed) ran in one window (the same `window_id`); the run may split
    across windows by whole blocks.
  Reported beside, not gating: each A9 run's CUDA scheduling flag (ld0: CUDA's default read back; ysync-ld0: yield set,
  verified and held to the end, timing events not blocking — a mismatch is printed as NOT AS WRITTEN) and its LP
  decode gate (open / why / when), generations.

Rules (D-030 cites them; the hint and the guard are computed by D-022's own functions, `tools/d022_report.py`
`c1_hint` / `c1_guard` / `c1_interim`, on these runs relabelled A9@c1r -> A9@r13x, A6F@c1r -> A6F@r13x):
  - HINT present iff the ratio of means of P(120) A9 / A6F over the FIXED 11 (D-022's C1_INSTANCES: doon,
    cryptanalysiskb128n5obj16, rail01, crna, loue, sing44, ca, item_placement, neos-960392, toguru, comp21; seed means,
    both seeds of both arms; not re-selected) is <= 0.95; INCOMPLETE while a run of the 11 is missing;
  - GUARD holds iff the mean bounded d of A9 vs A6F on seed means over all 17 (both-P = 1 at d = 0) is >= -0.05;
    INCOMPLETE (no partial mean) while a run of the 17 is missing;
  - HARM (D-025 amendment 3, labelled cross-day): per development instance of the 17, the new A9 (A9@c1r) against
    A9@r13x of 10-13 (set itx-r13): d_i = (P(A9@r13x) - P(new)) / max(both, 1e-3) on seed means (`stats.effect_d`),
    band_i = max(0.05, the larger seed-0 vs seed-1 bounded difference |a - b| / max(a, b, 1e-3) of either arm — A9@r13x
    or the new A9); harm iff d_i < -band_i. Any harm goes to Berk with the numbers beside the go / no-go, and the
    frozen-v2 candidate is not presumed: he decides which configuration freezes.
  Reported, not criteria (D-013 terms, as `tools/d022_report.py` reports them): A9@c1r vs A6F@c1r (same window, equal
  footprint) on the fixed 11, and vs A0 (SCIP alone; sets dev / classprobe, earlier days) on the fixed 11 and on
  SCIP's weak set among them (A0 P >= 0.5): improved k of n, worse, mean P before -> after, mean d with its 95 %
  bootstrap interval, the Wilcoxon test, worst P after. Beside: the 10-13 reading — the same functions on set itx-r13's
  A9@r13x / A6F@r13x (D-022's Result recorded hint 0.827, guard +0.063) — per instance and in total.

Failed runs: outcomes of their arm (D-002 §6, `keep`, the reading the output leads with: truncated at the crash, as
build-runs scores it); the letter (`exclude`: a failed run is missing, so what needs it is INCOMPLETE) printed beside
when a run failed. Output: HINT present / absent, GUARD holds / fails, the HARM list, and "go / no-go is Berk's".
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.analysis import c1_devtests as C
from itrea.analysis.backoff import bounded
from itrea.analysis.stats import effect_d
from itrea.experiments import make_jobs as mj
from itrea.experiments import pair
from itrea.paths import DATASETS, ROOT

_spec = importlib.util.spec_from_file_location("d022_report_for_c1r",
                                               Path(__file__).resolve().parent / "d022_report.py")
D22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D22)

A9, A6F = f"A9@{mj.C1R_VARIANT}", f"A6F@{mj.C1R_VARIANT}"
LABELS = (A9, A6F)
SEEDS, BUDGET = tuple(mj.C1R_SEEDS), float(mj.C1R_BUDGET)
OLD_SET, OLD_A9, OLD_A6F = D22.SET, D22.TREAT, D22.A6F          # itx-r13, A9@r13x, A6F@r13x (10-13)
TO_D022 = {A9: OLD_A9, A6F: OLD_A6F}                            # D-022's functions read these labels
FROM_D022 = {v: k for k, v in TO_D022.items()}
C1_INSTANCES = tuple(D22.C1_INSTANCES)                          # the fixed 11 (D-022 second amendment)
HARM_BAND_MIN = 0.05                                            # D-025 amendment 3 (D-023's band_i)
RECORDED_1013 = dict(hint=0.827, guard=0.063)                   # D-022's Result, 2026-10-13 22:2x (as recorded)
LEGS = ("complete", "harness", "config", "compile", "window")
FLAG = {"ld0": "auto", "ysync-ld0": "yield"}                    # each candidate's CUDA scheduling flag as written
BERK = ("go / no-go on C1 is Berk's (D-019 addendum): the hint and the guard are its inputs; any harm goes to Berk "
        "with the numbers beside the go / no-go, and the frozen-v2 candidate is not presumed — he decides which "
        "configuration freezes (D-025 amendment 3)")


def development_instances() -> list[str]:
    return list(mj.development_names())


def expected(a9: str, set_: str) -> dict:
    """{(instance, label, seed): run id} of the set's jobs (`make_jobs.c1_rerun`, pinned run hashes)."""
    return {(j["name"], pair.arm_label(j), int(j["seed"])): pair.run_id_for(j) for j in mj.c1_rerun(a9, set_=set_)}


# ----------------------------------------------------------------------------------------------- one run
def cupy_compiled(rj: dict) -> bool:
    cc = rj.get("cupy_cache")
    return bool(isinstance(cc, dict) and C.flag(cc.get("compiled_on_clock")))


def flag_state(a9: str, row: dict, summ: dict) -> dict:
    """Beside, not gating: the A9 run's CUDA scheduling flag against the candidate's file (ld0: CUDA's default read
    back; ysync-ld0: yield set, verified and held, timing events not blocking)."""
    if not summ:
        return dict(ok=None, text="no ea.summary.json" + (" (failed run)" if row.get("status") == "failed" else ""))
    cs = summ.get("cuda_sched") or {}
    want = FLAG[a9]
    if want == "auto":
        ok = cs.get("state") == "read" and cs.get("sched_active") == "auto"
    else:
        ok = cs.get("state") == "verified" and cs.get("sched_active") == want and cs.get("sched_at_end") == want
    ok = bool(ok and not C.flag(summ.get("phase_events_block")))
    return dict(ok=ok, want=want, state=cs.get("state"), active=cs.get("sched_active"), at_end=cs.get("sched_at_end"),
                events_block=summ.get("phase_events_block"),
                text=f"{cs.get('state')}/{cs.get('sched_active')}" + ("" if ok else f" NOT AS WRITTEN (want {want})"))


def decoder_state(summ: dict) -> dict:
    ld = summ.get("lp_decode") if isinstance(summ.get("lp_decode"), dict) else {}
    return dict(gate=ld.get("gate"), gate_why=ld.get("gate_why"), gate_t=ld.get("gate_t"),
                generations=summ.get("generations"))


# ----------------------------------------------------------------------------------------------- the rules
def _p_by(rows) -> dict:
    """{(instance, label, seed): P(120)} of the rows with a finite pt."""
    return {(r["instance"], r["arm"], int(r["seed"])): float(r["pt"]) for r in rows if C.fin(r.get("pt"))}


def harm_leg(new: dict, old: dict, instances) -> dict:
    """D-025 amendment 3 on each instance: `new` {(instance, seed): P of the new A9}, `old` the same of A9@r13x
    (10-13). d_i = effect_d(P̄ old, P̄ new) (positive = the new A9 better); band_i = max(0.05, bounded(old s0, s1),
    bounded(new s0, s1)); harm iff d_i < -band_i. An instance without both seeds of both arms is unread (INCOMPLETE)."""
    rows, harm, unread = [], [], []
    for i in instances:
        pn = [new.get((i, s)) for s in SEEDS]
        po = [old.get((i, s)) for s in SEEDS]
        if any(v is None for v in pn + po):
            unread.append(i)
            rows.append(dict(instance=i, read=False, p_new_seeds=pn, p_old_seeds=po))
            continue
        mo, mn = float(np.mean(po)), float(np.mean(pn))
        spreads = [x for x in (bounded(po[0], po[1]), bounded(pn[0], pn[1])) if C.fin(x)]
        band = max([HARM_BAND_MIN] + spreads)
        d = effect_d(mo, mn)
        h = bool(d < -band)
        rows.append(dict(instance=i, read=True, p_old=mo, p_new=mn, p_old_seeds=po, p_new_seeds=pn, d=d, band=band,
                         spread_old=bounded(po[0], po[1]), spread_new=bounded(pn[0], pn[1]), harm=h))
        if h:
            harm.append(i)
    state = "HARM" if harm else ("INCOMPLETE" if unread else "none")
    return dict(rows=rows, harm=harm, unread=unread, state=state, band_min=HARM_BAND_MIN)


def per_instance(p: dict, a9: str, a6f: str, instances) -> dict:
    """{instance: P̄ A6F, P̄ A9 and the bounded d of A9 vs A6F on seed means} where both seeds of both arms are present
    (the table printed beside the rules; the guard's own rows are D-022's)."""
    out = {}
    for i in instances:
        pa = [p.get((i, a6f, s)) for s in SEEDS]
        p9 = [p.get((i, a9, s)) for s in SEEDS]
        if any(v is None for v in pa + p9):
            continue
        ma, m9 = float(np.mean(pa)), float(np.mean(p9))
        out[i] = dict(p_a6f=ma, p_a9=m9, d=effect_d(ma, m9))
    return out


def _relabel(rows) -> pd.DataFrame:
    df = pd.DataFrame([dict(instance=r["instance"], seed=int(r["seed"]), arm=TO_D022[r["arm"]], pt=r.get("pt"))
                       for r in rows], columns=["instance", "seed", "arm", "pt"])
    df["pt"] = pd.to_numeric(df["pt"], errors="coerce")
    return df


def _ours(x: dict) -> dict:
    """A D-022 hint / guard result with its missing runs named by this set's labels (A9@c1r / A6F@c1r)."""
    return dict(x, missing=[(i, FROM_D022.get(a, a), s) for i, a, s in x.get("missing", [])])


def rules(rows, runs: pd.DataFrame, instances) -> dict:
    """C1's hint and guard (D-022's functions on the relabelled rows), the D-013 blocks vs A6F and vs A0, the harm leg
    against A9@r13x of 10-13, and the 10-13 reading beside."""
    a0, _ = D22.select(runs, D22.A0_SETS, (D22.A0,), instances=list(C1_INSTANCES))
    c1 = D22.c1_interim(_relabel(rows), a0, list(C1_INSTANCES), list(instances))
    old = C.select(runs, OLD_SET, (OLD_A9, OLD_A6F), SEEDS, BUDGET, keep_failed=True)[0]
    old = old[old["instance"].isin(list(instances))]
    then = D22.c1_interim(old[["instance", "seed", "arm", "pt"]].copy(), a0, list(C1_INSTANCES), list(instances))
    p_new = {(i, s): v for (i, a, s), v in _p_by(rows).items() if a == A9}
    p_old = {(i, s): v for (i, a, s), v in _p_by(old.to_dict("records")).items() if a == OLD_A9}
    harm = harm_leg(p_new, p_old, instances)
    return dict(hint=_ours(c1["hint"]), guard=_ours(c1["guard"]), vs_a6f=c1[D22.A6F][D22.TREAT]["all"],
                vs_a0=c1[D22.A0], harm=harm, table=per_instance(_p_by(rows), A9, A6F, instances),
                then=dict(hint=then["hint"], guard=then["guard"], n_runs=int(len(old)), recorded=RECORDED_1013,
                          table=per_instance(_p_by(old.to_dict("records")), OLD_A9, OLD_A6F, instances)))


# ----------------------------------------------------------------------------------------------- one reading
def reading(runs: pd.DataFrame, root: Path, a9: str, set_: str, keep_failed: bool, ext: dict | None = None) -> dict:
    exp = expected(a9, set_)
    ext = ext if ext is not None else C.ext_harness()
    instances = development_instances()
    sel, dropped = C.select(runs, set_, LABELS, SEEDS, BUDGET, keep_failed)
    by = {(r["instance"], r["arm"], int(r["seed"])): r for r in sel.to_dict("records")}
    outside = sorted(str(r["run_id"]) for k, r in by.items() if k not in exp)
    failed = C.failed_keys(runs, set_, LABELS, SEEDS, BUDGET)
    legs: dict[str, list[str]] = {k: [] for k in LEGS}
    legs["complete"], present = C.completeness(list(exp), by, failed)
    cells, kept = {}, []
    for k in present:
        r = by[k]
        rj, summ = C.run_files(r, root)
        if k[1] != A9:
            summ = {}                                    # A6F has no EA: never charged with an EA / CuPy record
        name = f"{k[0]} {k[1]} s{k[2]}"
        for leg, probs in C.base_problems(r, rj, summ, exp[k], ext).items():
            legs[leg] += [f"{name}: {x}" for x in probs]
        if k[1] == A9 and cupy_compiled(rj) and not any(x.startswith(name) for x in legs["compile"]):
            legs["compile"].append(f"{name}: compiled on the clock (CuPy cache)")
        if not C.fin(r.get("pt")):
            legs["complete"].append(f"no P(120): {name}")
        if r.get("status") == "failed":
            kept.append(name)
        cells[k] = dict(row=r, summ=summ, flag=flag_state(a9, r, summ) if k[1] == A9 else None,
                        decoder=decoder_state(summ) if k[1] == A9 else None)
    for i in instances:
        for s in SEEDS:
            have = [cells[(i, lab, s)] for lab in LABELS if (i, lab, s) in cells]
            if len(have) < 2:
                continue
            ws = [h["row"].get("window_id") for h in have]
            if not all(isinstance(w, str) and w for w in ws):
                legs["window"].append(f"{i} s{s}: a window id is missing")
            elif len(set(ws)) > 1:
                legs["window"].append(f"{i} s{s}: the two runs ran in different windows {sorted(set(ws))}")
    bad = [k for k in LEGS if legs[k]]
    verdict = "INVALID" if any(k != "complete" for k in bad) else ("INCOMPLETE" if bad else "VALID")
    val = dict(verdict=verdict, failing=bad, legs=legs,
               deviation=[f"{x} failed and is read as D-002 §6 keeps a crash (an outcome of its arm)" for x in kept])
    out = dict(keep_failed=keep_failed, a9=a9, set=set_, n_expected=len(exp), n_present=len(cells), validity=val,
               duplicates=dropped, outside=outside, cells=cells, rules=None, summary=None)
    if verdict != "INVALID":
        out["rules"] = rules([c["row"] for c in cells.values()], runs, instances)
        out["summary"] = summary(out["rules"], verdict)
    else:
        out["summary"] = dict(validity="INVALID", hint="not read", guard="not read", harm="not read", berk=BERK)
    return out


def summary(ru: dict, verdict: str) -> dict:
    h, g, hm = ru["hint"], ru["guard"], ru["harm"]
    return dict(validity=verdict, hint=h["state"], hint_ratio=h.get("ratio"), guard=g["state"],
                guard_mean_d=g.get("mean_d"), harm=hm["state"], harm_list=list(hm["harm"]),
                harm_unread=list(hm["unread"]), berk=BERK)


def report(runs: pd.DataFrame, root: Path = ROOT, a9: str = "ld0", set_: str | None = None,
           failed: str = "both") -> dict:
    if a9 not in mj.C1R_A9:
        raise ValueError(f"--a9: one of {', '.join(mj.C1R_A9)} (as the job file was written)")
    set_ = set_ or mj.C1R_SET_PREFIX + a9
    cand = C.candidates(runs, set_, LABELS, SEEDS, BUDGET)
    fails = sorted(str(r) for r in cand[cand["status"] == "failed"]["run_id"])
    modes = {"exclude": [False], "keep": [True], "both": [True, False] if fails else [True]}[failed]
    ext = C.ext_harness()
    return dict(set=set_, a9=a9, config=mj.C1R_A9[a9][0], run_hash=mj.C1R_A9[a9][1], failed_runs=fails,
                other_rows=C.others(runs, set_, LABELS, SEEDS, BUDGET),
                readings=[reading(runs, root, a9, set_, k, ext) for k in modes])


# ----------------------------------------------------------------------------------------------- output
def _hint_line(h: dict, label: str) -> str:
    if h["state"] == "INCOMPLETE":
        return f"HINT INCOMPLETE ({label}): {len(h['missing'])} runs of the fixed {h['n']} missing {h['missing'][:4]}"
    return (f"HINT {h['state']} ({label}): ratio of means of P(120) A9 / A6F over the fixed {h['n']} = "
            f"{C.f(h['ratio'])} ({C.f(h['mean_P_a9'])} / {C.f(h['mean_P_a6f'])}; present iff <= {h['threshold']:g})")


def _guard_line(g: dict, label: str) -> str:
    if g["state"] == "INCOMPLETE":
        return f"GUARD INCOMPLETE ({label}): {len(g['missing'])} runs of the {g['n']} missing {g['missing'][:4]}"
    return (f"GUARD {g['state']} ({label}): mean bounded d of A9 vs A6F over the {g['n']} = {g['mean_d']:+.3f} "
            f"(holds iff >= {g['threshold']:g}; seed means, both-P = 1 at d = 0)")


def render(rep: dict) -> str:
    L = [f"C1 re-read (D-030), set {rep['set']}: {A9} on {rep['config']} (run hash {rep['run_hash']}) vs {A6F} (A6F@r13x's "
         f"job, same window), 17 instances x seeds {list(SEEDS)}, {BUDGET:g} s, ext; rules fixed before any run "
         f"(D-022's hint / guard, D-025 amendment 3's harm leg); development, descriptive (D-010)"]
    if rep["failed_runs"]:
        L.append(f"failed runs: {', '.join(rep['failed_runs'])}")
    if rep["other_rows"]:
        L.append(f"rows of the set no rule reads: {', '.join(rep['other_rows'])}")
    for rd in rep["readings"]:
        L.append("")
        L.append(f"== reading: failed runs {'kept (D-002 §6; the reading that leads)' if rd['keep_failed'] else 'excluded (the letter)'} ==")
        v = rd["validity"]
        L.append(f"VALIDITY {v['verdict']}: {rd['n_present']} of {rd['n_expected']} runs present"
                 + (f"; failing leg(s): {', '.join(v['failing'])}" if v["failing"] else ""))
        for leg in LEGS:
            for x in v["legs"][leg][:12]:
                L.append(f"  ({leg}) {x}")
            if len(v["legs"][leg]) > 12:
                L.append(f"  ({leg}) … {len(v['legs'][leg]) - 12} more")
        for x in v["deviation"]:
            L.append(f"  kept: {x}")
        for d in rd["duplicates"]:
            L.append(f"  duplicate left out: {d['run_id']}")
        for x in rd["outside"]:
            L.append(f"  not one of the 68 (ignored): {x}")
        a9_cells = {k: c for k, c in rd["cells"].items() if k[1] == A9}
        if a9_cells:
            off = [f"{k[0]} s{k[2]} {c['flag']['text']}" for k, c in sorted(a9_cells.items()) if c["flag"]["ok"] is False]
            L.append(f"beside (not gating): {A9}'s CUDA flag as written ({FLAG[rd['a9']]}) on "
                     f"{sum(1 for c in a9_cells.values() if c['flag']['ok'])} of {len(a9_cells)} runs"
                     + (f"; NOT AS WRITTEN: {off}" if off else "") + "; LP decode gate open on "
                     f"{sum(1 for c in a9_cells.values() if c['decoder']['gate'] == 'open')} of {len(a9_cells)}")
        if rd["rules"] is None:
            L.append("RULES NOT READ: validity fails -> INVALID")
            L.append(f"SUMMARY: INVALID — {BERK}")
            continue
        ru = rd["rules"]
        g_rows, t_rows = ru["table"], ru["then"]["table"]
        L.append("per instance (seed means; d vs A6F: + = A9 better; harm: d_i = bounded d of the new A9 vs A9@r13x "
                 "of 10-13, cross-day, harm iff d_i < -band_i):")
        L.append("  " + "instance".ljust(28) + "P A6F".ljust(8) + "P A9".ljust(8) + "d".ljust(8) + "| 10-13: P A6F".ljust(15)
                 + "P A9".ljust(8) + "d".ljust(8) + "| harm d_i".ljust(11) + "band_i".ljust(8) + "harm")
        hm = {r["instance"]: r for r in ru["harm"]["rows"]}
        for i in development_instances():
            g, t, h = g_rows.get(i), t_rows.get(i), hm.get(i, {})
            fixed = "*" if i in C1_INSTANCES else " "
            L.append(f"  {fixed}{i:<27}" + (f"{C.f(g['p_a6f']):<8}{C.f(g['p_a9']):<8}{g['d']:+.3f}  " if g else "–".ljust(24))
                     + (f"| {C.f(t['p_a6f']):<13}{C.f(t['p_a9']):<8}{t['d']:+.3f}  " if t else "| –".ljust(31))
                     + (f"| {h['d']:+.3f}    {h['band']:.3f}   {'HARM' if h['harm'] else '-'}" if h.get("read")
                        else "| unread"))
        L.append("  (* = the fixed 11 of the hint)")
        L.append(_hint_line(ru["hint"], "this set"))
        L.append(f"  beside, 10-13 (set {OLD_SET}, {ru['then']['n_runs']} runs): " + _hint_line(ru["then"]["hint"], "10-13")
                 + f"; recorded in D-022's Result: {RECORDED_1013['hint']:.3f}")
        L.append(_guard_line(ru["guard"], "this set"))
        L.append(f"  beside, 10-13: " + _guard_line(ru["then"]["guard"], "10-13")
                 + f"; recorded in D-022's Result: {RECORDED_1013['guard']:+.3f}")
        h = ru["harm"]
        if h["harm"]:
            L.append("HARM (D-025 amendment 3, cross-day vs A9@r13x of 10-13) on " + ", ".join(
                f"{r['instance']} (d_i {r['d']:+.3f} < -{r['band']:.3f}; P {C.f(r['p_old'])} -> {C.f(r['p_new'])}; seeds "
                f"{' / '.join(C.f(x) for x in r['p_old_seeds'])} -> {' / '.join(C.f(x) for x in r['p_new_seeds'])})"
                for r in h["rows"] if r.get("read") and r["harm"]) + " -> to Berk with these numbers")
        else:
            L.append("HARM (D-025 amendment 3, cross-day vs A9@r13x of 10-13): none"
                     + (f" on the {17 - len(h['unread'])} read" if h["unread"] else " on the 17"))
        if h["unread"]:
            L.append(f"  harm leg INCOMPLETE: unread (a seed of either arm missing) {h['unread']}")
        L.append("reported, not criteria (D-013 terms) — vs cuOpt alone (A6F@c1r, same window, equal footprint): "
                 + D22._d013_line(ru["vs_a6f"], "the fixed 11", "instances"))
        c = ru["vs_a0"]
        L.append(f"reported, not criteria (D-013 terms) — vs {c['label']}: "
                 + D22._d013_line(c[D22.TREAT]["all"], "the fixed 11", "instances"))
        L.append(f"  SCIP's weak set among them (A0 P >= {D22.WEAK_P}) {c['weak_set']} ({len(c['weak_set'])} of "
                 f"{c['n_baseline']} with A0 runs)" + (": " + D22._d013_line(c[D22.TREAT]["weak"], "weak set", "weak")
                                                       if c["weak_set"] else ""))
        s = rd["summary"]
        L.append(f"SUMMARY ({'keep' if rd['keep_failed'] else 'exclude'}): validity {s['validity']}; HINT {s['hint']}"
                 f" ({C.f(s['hint_ratio'])}); GUARD {s['guard']} ({C.f(s['guard_mean_d'])}); HARM "
                 + (", ".join(s["harm_list"]) if s["harm_list"] else s["harm"])
                 + (f" (unread {s['harm_unread']})" if s["harm_unread"] else ""))
        L.append(f"  {BERK}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="c1r_report", description=__doc__.split("\n\n")[0])
    ap.add_argument("--a9", required=True, choices=tuple(mj.C1R_A9),
                    help="the candidate, as the job file was written (make-jobs c1r-rerun --a9 …)")
    ap.add_argument("--set", default=None, help="default c1r-<a9>")
    ap.add_argument("--runs", type=Path, default=DATASETS / "runs.parquet")
    ap.add_argument("--root", type=Path, default=ROOT, help="the run directories' root (run_dir is relative to it)")
    ap.add_argument("--failed", choices=("both", "exclude", "keep"), default="both")
    ap.add_argument("--json", type=Path, help="write the report as JSON here too")
    a = ap.parse_args(argv)
    try:
        runs = pd.read_parquet(a.runs)
        rep = report(runs, a.root, a.a9, a.set, a.failed)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(render(rep))
    if a.json:
        out = dict(rep, readings=[{k: v for k, v in rd.items() if k != "cells"} | dict(
            runs={f"{i} {lab} s{s}": dict(run_id=c["row"]["run_id"], pt=c["row"].get("pt"),
                                          status=c["row"].get("status"), window_id=c["row"].get("window_id"),
                                          flag=c["flag"], decoder=c["decoder"])
                  for (i, lab, s), c in rd["cells"].items()}) for rd in rep["readings"]])
        a.json.write_text(json.dumps(C.jsonable(out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
