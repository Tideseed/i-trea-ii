"""D-025 report: R1 of the C1 diagnosis — the EA's CUDA context at blocking sync so its HiGHS LP worker gets X925 core 6
(vault `50-decisions/D-025 …`, from `30-results/2026-10-14 C1 diagnosis and improvement candidates.md` §5), and its
pre-stated fallbacks (a) `bnice` (renice +10 relative to the LP worker, no context flag) and (b) `ysync`
(CU_CTX_SCHED_YIELD). The rules are D-025's, fixed before any run (with its dated clarification and amendment of
2026-10-14, also before any run); nothing is chosen by eye.

Usage (repository root, after `itrea build-runs`; untimed analysis: OMP_NUM_THREADS=1, never beside timed runs with
default thread counts):

    OMP_NUM_THREADS=1 .venv/bin/python tools/d025_report.py [--set r1-bsync] [--control r13x|boY]
                                                            [--failed both|exclude|keep] [--json out]
    itrea d025-report …                       (the same, through the entry point)

The set names the variant (`r1-bsync` → bsync, `r1-bnice` → bnice, `r1-ysync` → ysync; `--variant` for a rerun set of
another name). `--control` is D-023's verdict as the job file was written (`make-jobs r1-bsync --control …`); without it
the report takes the one treatment label the set holds (A9@<variant> → r13x, A9@boY-<variant> → boY). The expected
runs, labels and run ids come from `make_jobs.r1_bsync`, so the report and the job file cannot drift apart: 5
instances x seeds 0–1 x {A9@<variant> or A9@boY-<variant>, A9@r1ctl (A9@r1ctl-<variant> for a fallback set)} = 20 runs.

Reads `datasets/runs.parquet` (`pt` = P(120) on the call clock, `status`, `attempt`, `run_dir`, `harness`,
`cuda_compiled_on_clock`) and each run's `run.json` (`config.harness`, `compiled_on_clock_cuda`) and `ea.summary.json`
(`P`, `gene_dtype`, `compiled_on_clock`, `cuda_sched` — the flags set and verified, or under "auto" read back, at the
start and at the end —, `phase_events_block`, `host_renice`, `generations`, `wall_s`, `ls_launches`, `cpu_clock`, `lp`
with the LP worker's own stamps `worker_done_t`, `worker_wall_s`, `worker_cpu_share`).

Validity (D-025; must hold, else INVALID and the report names the leg): complete (20 of 20; a void is moved aside by the
pool and its rerun is the run read); harness ext on every run; config (each run's id is its job's: the EA configuration,
harness and footprint hash as written); no compile on the clock; equal P and gene dtype per block; the flag state —
treatment bsync: blocking sync set, verified on the active context and still held at the end, the timing events
blocking; ysync: yield, the same way; bnice: CUDA's default read back (no flag) and `host_renice.gap` = 10 (the clarification:
"the variant's flag"); control: CUDA's default read back, no renice, non-blocking events. Failed runs (`--failed`): D-025
reads them as outcomes of their arm (D-002 §6, `keep`) — the verdict's reading — and the letter (`exclude`: a failed
run is not completed, so the complete leg fails) is printed beside (`both`, the default, when the set holds a failed
run).

Rules (gating unless marked):
  - no-harm on item_placement_9903 and comp21-2idx: generations on seed means, Ḡ(T) / Ḡ(C) >= 1 − max(0.05, the
    control's relative seed spread |G_C,s0 − G_C,s1| / Ḡ(C)) (D-025's amendment of 2026-10-14, before any run: the
    record's "paired by seed, >= 0.95" assumed item's 4 % within-day spread; comp21's is 9–17 %, so the paired rule
    would fail by noise in about 60 % of comp21 tests and fire fallback (b) on noise); P on seed means,
    P̄(T) − P̄(C) <= max(0.05, |P_C,s0 − P_C,s1|) (one-sided, raw P);
  - no-harm on the hint instances ca__Scenario400_reserves_1, sing44, neos-960392 (D-025's amendment of 2026-10-14,
    before any run: the per-instance band that was "reported beside" now gates, so a gain on ca cannot carry a loss
    elsewhere through the pooled hint): the same P band;
  - the hint (futility-only, D-024 rule 2): ΣP(T) / ΣP(C) over the three hint instances x seeds 0–1 <= 0.95, per
    instance beside;
  - mechanism (reported, not gating): the EA's self-CPU per wall-second, the LP worker's own finish time and CPU share,
    the LP poll time, generations and FJ launches per second against the control on the hint instances.
Verdict: ADOPT iff validity holds, both no-harm legs on item_placement and comp21, the band on the three hint instances
and the hint; NOT ADOPTED otherwise (the failing legs named); INVALID when a validity leg fails. For set r1-bsync the
report also prints D-025's fallback triggers: (a) bnice iff there is no hint and ca's LP worker got < 0.8 of the core
(its CPU / its own solve wall) on both seeds; (b) ysync iff the generation guard failed on item_placement or comp21;
both firing → (a) first (it carries no wake latency); no trigger → the R1 line ends for C1. A fallback set's
NOT ADOPTED ends the R1 line (no fallback of a fallback). The C1 re-read's harm leg (the amendment) is read at the
re-read, not here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from itrea.analysis import c1_devtests as C
from itrea.experiments import pair
from itrea.experiments.make_jobs import (R1_BUDGET, R1_CONTROL, R1_CONTROLS, R1_INSTANCES, R1_SEEDS, R1_SET_PREFIX,
                                         R1_VARIANTS, r1_bsync)
from itrea.paths import DATASETS, ROOT

SEEDS, BUDGET = tuple(R1_SEEDS), float(R1_BUDGET)
CA = "ca__Scenario400_reserves_1"
HINT_INSTANCES = (CA, "sing44", "neos-960392")
GUARD_INSTANCES = ("item_placement_9903", "comp21-2idx")
G_FLOOR, P_FLOOR, HINT_MAX, LP_SHARE_MIN = 0.05, 0.05, 0.95, 0.8
FLAG_MODE = {"bsync": "blocking_sync", "bnice": "auto", "ysync": "yield"}
RENICE_GAP = 10
LEGS = C.BASE_LEGS + ("flag",)


def labels(control: str, variant: str) -> tuple[str, str]:
    """(treatment, control) labels as the writer makes them."""
    t = "A9@" + R1_CONTROLS[control][1] + variant
    c = "A9@" + (R1_CONTROL if variant == "bsync" else f"{R1_CONTROL}-{variant}")
    return t, c


def variant_of(set_: str) -> str | None:
    v = set_.removeprefix(R1_SET_PREFIX)
    return v if set_.startswith(R1_SET_PREFIX) and v in R1_VARIANTS else None


def detect_control(runs: pd.DataFrame, set_: str, variant: str) -> str | None:
    """The control (D-023's verdict) the set was written for: the one treatment label it holds."""
    have = set(runs[runs["set"] == set_]["arm"].astype(str))
    hits = [c for c in R1_CONTROLS if labels(c, variant)[0] in have]
    return hits[0] if len(hits) == 1 else None


def expected(control: str, variant: str, set_: str) -> dict:
    """{(instance, label, seed): run id} of the set's jobs (`make_jobs.r1_bsync`)."""
    return {(j["name"], pair.arm_label(j), int(j["seed"])): pair.run_id_for(j)
            for j in r1_bsync(control, variant, set_=set_)}


# ----------------------------------------------------------------------------------------------- per run
def flag_problems(kind: str, variant: str, row: dict, summ: dict) -> tuple[list[str], str | None]:
    """D-025's flag leg for one run (`kind` treatment / control): (problems, note). A failed run without an EA summary
    is noted, not charged (an outcome of its arm under `keep`)."""
    if not summ:
        if row.get("status") == "failed":
            return [], "failed run without ea.summary.json: flag unread"
        return ["no ea.summary.json"], None
    cs = summ.get("cuda_sched") or {}
    hr = summ.get("host_renice")
    probs = []
    want = FLAG_MODE[variant] if kind == "treatment" else "auto"
    if want == "auto":
        if cs.get("state") != "read" or cs.get("sched_active") != "auto":
            probs.append(f"CUDA's default not read back (cuda_sched state {cs.get('state')!r}, active "
                         f"{cs.get('sched_active')!r})")
    elif not (cs.get("state") == "verified" and cs.get("sched_active") == want and cs.get("sched_at_end") == want):
        probs.append(f"{want} not set and held (state {cs.get('state')!r}, active {cs.get('sched_active')!r}, at the "
                     f"end {cs.get('sched_at_end')!r})")
    blocking = want == "blocking_sync"
    if bool(summ.get("phase_events_block")) != blocking:
        probs.append(f"timing events blocking = {summ.get('phase_events_block')!r}, expected {blocking}")
    if kind == "treatment" and variant == "bnice":
        if not isinstance(hr, dict) or hr.get("gap") != RENICE_GAP:
            probs.append(f"renice gap {None if not isinstance(hr, dict) else hr.get('gap')!r}, expected "
                         f"{RENICE_GAP} (run the pool without nice)")
    elif hr is not None:
        probs.append("a renice in a run whose configuration has none")
    return probs, None


def mechanism(summ: dict) -> dict:
    """Reported per run (D-025, not gating)."""
    wall = summ.get("wall_s")
    g, ls = summ.get("generations"), summ.get("ls_launches")
    per = (lambda x: float(x) / float(wall) if C.fin(x) and C.fin(wall) and float(wall) > 0 else None)
    return dict(self_per_wall=C.dig(summ, "cpu_clock", "self_per_wall_s"),
                main_thread_per_wall=C.dig(summ, "cpu_clock", "main_thread_per_wall_s"),
                lp_state=C.dig(summ, "lp", "state"), lp_poll_t=C.dig(summ, "lp", "t"),
                lp_done_t=C.dig(summ, "lp", "worker_done_t"), lp_wall=C.dig(summ, "lp", "worker_wall_s"),
                lp_share=C.dig(summ, "lp", "worker_cpu_share"), generations=g, gens_per_s=per(g),
                launches=ls, launches_per_s=per(ls))


# ----------------------------------------------------------------------------------------------- the rules
def _pts(cells: dict, inst: str, lab: str) -> list:
    return [cells[(inst, lab, s)]["row"].get("pt") if (inst, lab, s) in cells else None for s in SEEDS]


def gen_guard(cells: dict, inst: str, t: str, c: str) -> dict:
    """The generation guard (D-025 as amended 2026-10-14): Ḡ(T) / Ḡ(C) >= 1 − max(0.05, |G_C,s0 − G_C,s1| / Ḡ(C)),
    seed means; a missing G (a failed run without a summary) fails it. The per-seed ratios are printed beside."""
    gt = [C.dig(cells.get((inst, t, s), {}), "summ", "generations") for s in SEEDS]
    gc = [C.dig(cells.get((inst, c, s), {}), "summ", "generations") for s in SEEDS]
    per = {s: dict(t=a, c=b, ratio=(float(a) / float(b) if C.fin(a) and C.fin(b) and float(b) > 0 else None))
           for s, a, b in zip(SEEDS, gt, gc)}
    if not all(C.fin(x) for x in gt + gc) or min(float(x) for x in gc) <= 0:
        return dict(per=per, ok=False, why="a generation count is missing", ratio=None, threshold=None)
    mt, mc = sum(map(float, gt)) / len(gt), sum(map(float, gc)) / len(gc)
    rel = (max(map(float, gc)) - min(map(float, gc))) / mc
    thr = 1.0 - max(G_FLOOR, rel)
    return dict(per=per, mean_t=mt, mean_c=mc, ratio=mt / mc, rel_spread=rel, threshold=thr,
                ok=bool(mt / mc >= thr - 1e-12))


def rules(cells: dict, t: str, c: str, variant: str, set_: str) -> dict:
    guard_g = {i: gen_guard(cells, i, t, c) for i in GUARD_INSTANCES}
    guard_p = {i: C.band_leg(_pts(cells, i, t), _pts(cells, i, c), P_FLOOR) for i in GUARD_INSTANCES}
    hint_band = {i: C.band_leg(_pts(cells, i, t), _pts(cells, i, c), P_FLOOR) for i in HINT_INSTANCES}
    hint = C.ratio_leg({i: _pts(cells, i, t) for i in HINT_INSTANCES}, {i: _pts(cells, i, c) for i in HINT_INSTANCES},
                       HINT_MAX)
    fails = []
    if not all(x["ok"] for x in guard_g.values()):
        fails.append("generation guard (item_placement / comp21)")
    if not all(x["ok"] for x in guard_p.values()):
        fails.append("P no-harm (item_placement / comp21)")
    if not all(x["ok"] for x in hint_band.values()):
        fails.append("P no-harm on the hint instances (amendment)")
    if not hint["present"]:
        fails.append("no hint")
    out = dict(guard_g=guard_g, guard_p=guard_p, hint_band=hint_band, hint=hint, fails=fails,
               verdict="ADOPT" if not fails else "NOT ADOPTED")
    shares = [C.dig(cells.get((CA, t, s), {}), "mech", "lp_share") for s in SEEDS]
    trig_b = not all(x["ok"] for x in guard_g.values())
    trig_a = (not hint["present"]) and all(C.fin(x) and float(x) < LP_SHARE_MIN for x in shares)
    out["ca_lp_share"] = shares
    if variant != "bsync":
        out["next"] = ("the R1 line widens (D-025: the variant becomes A9's and A3x's development default)"
                       if not fails else "the R1 line ends for C1 (a fallback's NOT ADOPTED; no fallback of a "
                                         "fallback)")
        out["triggers"] = None
    else:
        out["triggers"] = dict(a=bool(trig_a), b=bool(trig_b))
        if not fails:
            out["next"] = "the R1 line widens (D-025: the variant becomes A9's and A3x's development default)"
        elif trig_a:
            out["next"] = ("fallback (a) bnice: no hint and ca's LP worker got < 0.8 of the core on both seeds"
                           + ("; (b)'s trigger fired too — (a) first, no chaining" if trig_b else ""))
        elif trig_b:
            out["next"] = "fallback (b) ysync: the generation guard failed on item_placement or comp21 (wake latency)"
        else:
            out["next"] = "no fallback trigger: the R1 line ends for C1"
    return out


# ----------------------------------------------------------------------------------------------- reading
def reading(runs: pd.DataFrame, root: Path, set_: str, control: str, variant: str, keep_failed: bool,
            ext: dict | None = None) -> dict:
    t, c = labels(control, variant)
    exp = expected(control, variant, set_)
    ext = ext if ext is not None else C.ext_harness()
    sel, dropped = C.select(runs, set_, (t, c), SEEDS, BUDGET, keep_failed)
    by = {(r["instance"], r["arm"], int(r["seed"])): r for r in sel.to_dict("records")}
    failed = C.failed_keys(runs, set_, (t, c), SEEDS, BUDGET)
    legs: dict[str, list[str]] = {k: [] for k in LEGS}
    legs["complete"], present = C.completeness(list(exp), by, failed)
    cells, notes, kept = {}, [], []
    for k in present:
        r = by[k]
        rj, summ = C.run_files(r, root)
        kind = "treatment" if k[1] == t else "control"
        for leg, probs in C.base_problems(r, rj, summ, exp[k], ext).items():
            legs[leg] += [f"{k[0]} {k[1]} s{k[2]}: {x}" for x in probs]
        fp, note = flag_problems(kind, variant, r, summ)
        legs["flag"] += [f"{k[0]} {k[1]} s{k[2]}: {x}" for x in fp]
        if note:
            notes.append(f"{k[0]} {k[1]} s{k[2]}: {note}")
        if not C.fin(r.get("pt")):
            legs["complete"].append(f"no P(120): {k[0]} {k[1]} s{k[2]}")
        if r.get("status") == "failed":
            kept.append(k)
        cells[k] = dict(row=r, summ=summ, kind=kind, mech=mechanism(summ))
    legs["block"] = C.block_problems(cells, [(i, s, [(i, t, s), (i, c, s)]) for i in R1_INSTANCES for s in SEEDS])
    bad = [k for k in LEGS if legs[k]]
    val = dict(verdict="INVALID" if bad else "VALID", failing=bad, legs=legs, notes=notes,
               deviation=[f"{i} {a} s{s} failed and is read as D-002 §6 keeps a crash (D-025: an outcome of its arm)"
                          for i, a, s in kept])
    out = dict(keep_failed=keep_failed, control=control, variant=variant, treatment_label=t, control_label=c,
               n_expected=len(exp), n_present=len(cells), validity=val, duplicates=dropped, cells=cells, rules=None,
               verdict="INVALID")
    if val["verdict"] == "VALID":
        out["rules"] = rules(cells, t, c, variant, set_)
        out["verdict"] = out["rules"]["verdict"]
    return out


def report(runs: pd.DataFrame, root: Path = ROOT, set_: str = "r1-bsync", control: str | None = None,
           variant: str | None = None, failed: str = "both") -> dict:
    variant = variant or variant_of(set_)
    if variant not in R1_VARIANTS:
        raise ValueError(f"set {set_!r} names no R1 variant ({', '.join(R1_VARIANTS)}): pass --variant")
    control = control or detect_control(runs, set_, variant)
    if control not in R1_CONTROLS:
        raise ValueError(f"cannot tell D-023's control for set {set_!r} from its rows: pass --control r13x|boY")
    t, c = labels(control, variant)
    cand = C.candidates(runs, set_, (t, c), SEEDS, BUDGET)
    fails = sorted(str(r) for r in cand[cand["status"] == "failed"]["run_id"])
    modes = {"exclude": [False], "keep": [True], "both": [True, False] if fails else [True]}[failed]
    ext = C.ext_harness()
    return dict(set=set_, control=control, variant=variant, failed_runs=fails,
                other_rows=C.others(runs, set_, (t, c), SEEDS, BUDGET),
                readings=[reading(runs, root, set_, control, variant, k, ext) for k in modes])


# ----------------------------------------------------------------------------------------------- output
def _band_line(i: str, x: dict) -> str:
    if x.get("diff") is None:
        return f"  {i}: {x.get('why')} -> FAIL"
    return (f"  {i}: T {' / '.join(C.f(v) for v in x['t'])} (mean {C.f(x['mean_t'])}), C {' / '.join(C.f(v) for v in x['c'])}"
            f" (mean {C.f(x['mean_c'])}); diff {x['diff']:+.3f}, band {x['band']:.3f} -> {'ok' if x['ok'] else 'FAIL'}")


def render(rep: dict) -> str:
    L = [f"D-025 report (C1 diagnosis R1), set {rep['set']}, variant {rep['variant']}, control {rep['control']} "
         f"(D-023's verdict as written): rules fixed in D-025 before any run; interim, n = 2 seeds; a hint never "
         f"declares success (D-024)"]
    if rep["failed_runs"]:
        L.append(f"failed runs: {', '.join(rep['failed_runs'])}")
    if rep["other_rows"]:
        L.append(f"rows of the set no rule reads: {', '.join(rep['other_rows'])}")
    for rd in rep["readings"]:
        t, c = rd["treatment_label"], rd["control_label"]
        L.append("")
        L.append(f"== reading: failed runs {'kept (D-002 §6; the verdict reading)' if rd['keep_failed'] else 'excluded (the letter)'} ==")
        v = rd["validity"]
        L.append(f"VALIDITY {v['verdict']}: {rd['n_present']} of {rd['n_expected']} runs present"
                 + (f"; failing leg(s): {', '.join(v['failing'])}" if v["failing"] else ""))
        for leg in LEGS:
            for x in v["legs"][leg]:
                L.append(f"  ({leg}) {x}")
        for x in v["notes"]:
            L.append(f"  note: {x}")
        for x in v["deviation"]:
            L.append(f"  kept: {x}")
        for d in rd["duplicates"]:
            L.append(f"  duplicate left out: {d['run_id']}")
        L.append(f"per run (T = {t}, C = {c}; P = pt; mechanism reported, not gating):")
        for k, cl in sorted(rd["cells"].items()):
            m, r = cl["mech"], cl["row"]
            L.append(f"  {k[0]:<28} {k[1]:<16} s{k[2]}  P {C.f(r.get('pt'))}  G {m['generations']}  gen/s "
                     f"{C.f(m['gens_per_s'], 2)}  launch/s {C.f(m['launches_per_s'], 2)}  EA self-CPU/wall "
                     f"{C.f(m['self_per_wall'])}  LP done {C.f(m['lp_done_t'], 2)} s (polled {C.f(m['lp_poll_t'], 2)})"
                     f"  LP CPU share {C.f(m['lp_share'])}  {r.get('status')}")
        if v["verdict"] != "VALID":
            L.append("RULES NOT READ: validity fails (D-025) -> INVALID; a wrong flag state means wrong code: fixed "
                     "and re-checked before any rerun, with a dated note in D-025")
            continue
        ru = rd["rules"]
        L.append(f"No-harm, generations on seed means (Ḡ(T) / Ḡ(C) >= 1 − max({G_FLOOR}, the control's relative seed "
                 f"spread); amendment 2026-10-14):")
        for i, x in ru["guard_g"].items():
            L.append(f"  {i}: " + "; ".join(f"s{s} {y['t']} vs {y['c']} ({C.f(y['ratio'])})" for s, y in x["per"].items())
                     + (f"; means ratio {C.f(x['ratio'])} vs threshold {C.f(x['threshold'])}" if x.get("ratio") is not None
                        else f"; {x.get('why')}") + f" -> {'ok' if x['ok'] else 'FAIL'}")
        L.append(f"No-harm, P on seed means (P̄(T) − P̄(C) <= max({P_FLOOR}, the control's seed spread)):")
        for i, x in ru["guard_p"].items():
            L.append(_band_line(i, x))
        L.append("No-harm on the hint instances (amendment 2026-10-14, gating):")
        for i, x in ru["hint_band"].items():
            L.append(_band_line(i, x))
        h = ru["hint"]
        if h["ratio"] is None:
            L.append(f"Hint: {h.get('why')} -> no hint")
        else:
            L.append(f"Hint (futility-only): ΣP(T) / ΣP(C) over {', '.join(HINT_INSTANCES)} x seeds 0–1 = "
                     f"{h['sum_t']:.4f} / {h['sum_c']:.4f} = {h['ratio']:.4f} (<= {HINT_MAX}) -> "
                     f"{'PRESENT' if h['present'] else 'absent'}; per instance "
                     + ", ".join(f"{i} {C.f(x['ratio'])}" for i, x in h["per"].items()))
        L.append(f"VERDICT: {ru['verdict']}" + (f" — failing: {', '.join(ru['fails'])}" if ru["fails"] else ""))
        if ru["triggers"] is not None:
            L.append(f"  fallback triggers: (a) {'FIRES' if ru['triggers']['a'] else 'no'} (ca LP CPU share "
                     f"{' / '.join(C.f(x) for x in ru['ca_lp_share'])}, < {LP_SHARE_MIN} on both seeds with no hint); "
                     f"(b) {'FIRES' if ru['triggers']['b'] else 'no'} (generation guard)")
        L.append(f"  next: {ru['next']}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="d025_report", description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=DATASETS / "runs.parquet")
    ap.add_argument("--root", type=Path, default=ROOT, help="the run directories' root (run_dir is relative to it)")
    ap.add_argument("--set", default="r1-bsync")
    ap.add_argument("--control", choices=tuple(R1_CONTROLS), default=None,
                    help="D-023's verdict as the job file was written (default: read from the set's labels)")
    ap.add_argument("--variant", choices=R1_VARIANTS, default=None, help="default: from the set's name")
    ap.add_argument("--failed", choices=("both", "exclude", "keep"), default="both")
    ap.add_argument("--json", type=Path, help="write the report as JSON here too")
    a = ap.parse_args(argv)
    try:
        runs = pd.read_parquet(a.runs)
        rep = report(runs, a.root, a.set, a.control, a.variant, a.failed)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(render(rep))
    if a.json:
        out = dict(rep, readings=[{k: v for k, v in rd.items() if k != "cells"} | dict(
            runs={f"{i} {arm} s{s}": dict(run_id=cl["row"]["run_id"], pt=cl["row"].get("pt"),
                                          status=cl["row"].get("status"), mechanism=cl["mech"])
                  for (i, arm, s), cl in rd["cells"].items()}) for rd in rep["readings"]])
        a.json.write_text(json.dumps(C.jsonable(out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
