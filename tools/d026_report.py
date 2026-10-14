"""D-026 report: R2 of the C1 diagnosis — lp_decode inside A9 on the EA's core, its worker started only after the EA's
HiGHS LP worker reached a final state (vault `50-decisions/D-026 …`, from `30-results/2026-10-14 C1 diagnosis and
improvement candidates.md` §5). The rules are D-026's, fixed before any run (with its dated clarification and amendment
of 2026-10-14, also before any run); nothing is chosen by eye.

Usage (repository root, after `itrea build-runs`; untimed analysis: OMP_NUM_THREADS=1, never beside timed runs with
default thread counts):

    OMP_NUM_THREADS=1 .venv/bin/python tools/d026_report.py --base bsync|boY-bsync --nice 10|0 [--set r2-ld]
                                                            [--failed both|exclude|keep] [--json out]
    OMP_NUM_THREADS=1 .venv/bin/python tools/d026_report.py --base r13x|boY [--set r2-ld] …   (R1 not adopted)
    itrea d026-report …                       (the same, through the entry point)

`--base` and `--nice` are the arguments the job file was written with (`make-jobs r2-ld --base … --nice …`): the
expected runs, labels and run ids come from `make_jobs.r2_ld`, so the report and the job file cannot drift apart:
{A9@r2ld, A9@r2ctl} x item_placement_9903 and sing44 at seeds 0–3 + ca__Scenario400_reserves_1, glass4 and
neos-3656078-kumeu at seeds 0–1 = 28 runs.

Reads `datasets/runs.parquet` (`pt`, `status`, `attempt`, `run_dir`, `harness`, `cuda_compiled_on_clock`, the decoder's
columns `lpd_incumbents`, `lpd_incumbents_I`, `lpd_wait_s_max`, `lpd_host_s`, `lpd_worker_solve_cpu_s` — and D-017's
A3@ld1 rows of set `dev` beside) and each run's `run.json` and `ea.summary.json` (`P`, `gene_dtype`,
`compiled_on_clock`, `cuda_sched`, `lp` with the LP worker's own stamps `worker_done_t` / `worker_wall_s`, `lp_decode`
with `gate`, `gate_t`, `gate_why`, `ready_t_worker`, `nice`, and `concat_parts_max`).

Validity (D-026; must hold, else INVALID and the report names the leg): complete (28 of 28); harness ext; config (each
run's id is its job's) and the base's flag state in both arms (blocking sync verified for a bsync base, CUDA's default
read back otherwise — the configuration as written); no compile on the clock (the > 8-part concatenate included: any
compile after t0); equal P and gene dtype per block; the decoder ran in every treatment run (its gate opened and its
worker started) and in no control run (no `lp_decode` block). Failed runs: outcomes of their arm (D-002 §6, `keep`, the
verdict's reading), the letter's reading (`exclude`) printed beside.

Rules (gating unless marked):
  - mechanism: (1) the decoder ready <= 10 s after its gate opened (`ready_t_worker` − `gate_t`, D-017's M0
    re-anchored) in every treatment run; (2) >= 1 decoded incumbent (`lpd_incumbents`) in each of the 8 treatment runs
    on item_placement and sing44; (3) the gate costs ca's LP nothing: the LP worker's own solve wall (`worker_wall_s`)
    in the treatment <= 1.10 x the control's, per seed;
  - no-harm (one-sided, seed means, raw P): ca P̄(T) − P̄(C) <= max(0.02, the control's seed spread); glass4 and kumeu
    <= max(0.05, spread); item_placement and sing44 (D-026's amendment of 2026-10-14, before any run: a gain on one
    cannot carry a loss on the other through the pooled hint) <= max(0.05, the control's range over its four seeds);
  - the hint (futility-only): ΣP(T) / ΣP(C) over item_placement and sing44 x seeds 0–3 <= 0.90, per instance beside;
  - reported: `lpd_wait_s_max`, `lpd_host_s` against A3@ld1 (D-017), decoder CPU, decoded incumbents (from completion
    beside), the LP worker's finish time, the gate's lag behind it (`gate_t` − `worker_done_t`: the gate opens when the
    EA reads the LP worker's final state at a generation boundary), `concat_parts_max`.
Verdict: ADOPT iff validity, the three mechanism checks, every no-harm leg and the hint hold; NOT ADOPTED otherwise
(the failing legs named); INVALID when a validity leg fails. The C1 re-read's harm leg (the amendment) is read at the
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
from itrea.experiments.make_jobs import (R2_BASES, R2_BUDGET, R2_CONTROL, R2_GUARD_INSTANCES, R2_GUARD_SEEDS,
                                         R2_HINT_INSTANCES, R2_HINT_SEEDS, R2_NICE, R2_SET, R2_TREATMENT, r2_ld)
from itrea.paths import DATASETS, ROOT

T, CTL = f"A9@{R2_TREATMENT}", f"A9@{R2_CONTROL}"
BUDGET = float(R2_BUDGET)
CA = "ca__Scenario400_reserves_1"
SEEDS = {**{i: tuple(R2_HINT_SEEDS) for i in R2_HINT_INSTANCES}, **{i: tuple(R2_GUARD_SEEDS) for i in R2_GUARD_INSTANCES}}
ALL_SEEDS = tuple(sorted(set(R2_HINT_SEEDS) | set(R2_GUARD_SEEDS)))
READY_MAX_S, LP_RATIO_MAX, HINT_MAX = 10.0, 1.10, 0.90
FLOORS = {CA: 0.02, "glass4": 0.05, "neos-3656078-kumeu": 0.05, "item_placement_9903": 0.05, "sing44": 0.05}
A3_SET, A3_ARM = "dev", "A3@ld1"                     # D-017's treatment runs, reported beside
LEGS = C.BASE_LEGS + ("flag", "decoder")


def base_mode(base: str) -> str:
    return "blocking_sync" if "bsync" in base else "auto"


def expected(base: str, nice: int | None, set_: str) -> dict:
    """{(instance, label, seed): run id} of the set's jobs (`make_jobs.r2_ld`)."""
    return {(j["name"], pair.arm_label(j), int(j["seed"])): pair.run_id_for(j) for j in r2_ld(base, nice, set_=set_)}


def flag_problems(mode: str, row: dict, summ: dict) -> tuple[list[str], str | None]:
    if not summ:
        if row.get("status") == "failed":
            return [], "failed run without ea.summary.json: flag unread"
        return ["no ea.summary.json"], None
    cs = summ.get("cuda_sched") or {}
    if mode == "auto":
        ok = cs.get("state") == "read" and cs.get("sched_active") == "auto"
    else:
        ok = cs.get("state") == "verified" and cs.get("sched_active") == mode and cs.get("sched_at_end") == mode
    return ([] if ok else [f"the base's flag ({mode}) not as written (cuda_sched state {cs.get('state')!r}, active "
                           f"{cs.get('sched_active')!r})"]), None


def decoder_problems(kind: str, row: dict, summ: dict) -> list[str]:
    if not summ:
        return []                                    # a failed run without a summary: an outcome (flag note)
    ld = summ.get("lp_decode")
    if kind == "control":
        return ["the decoder ran in a control run"] if isinstance(ld, dict) else []
    if not isinstance(ld, dict):
        return ["no lp_decode block: the decoder never ran"]
    if ld.get("gate") != "open":
        return [f"the gate never opened (gate {ld.get('gate')!r}): the decoder never ran"]
    return []


def mechanism(row: dict, summ: dict) -> dict:
    ld = summ.get("lp_decode") if isinstance(summ.get("lp_decode"), dict) else {}
    gate_t, ready_w = ld.get("gate_t"), ld.get("ready_t_worker")
    done_t = C.dig(summ, "lp", "worker_done_t")
    val = (lambda k: summ.get(k) if summ.get(k) is not None else row.get(k))
    return dict(gate=ld.get("gate"), gate_why=ld.get("gate_why"), gate_t=gate_t, ready_t_worker=ready_w,
                ready_after_gate=(float(ready_w) - float(gate_t) if C.fin(ready_w) and C.fin(gate_t) else None),
                lp_done_t=done_t, lp_wall=C.dig(summ, "lp", "worker_wall_s"),
                gate_lag=(float(gate_t) - float(done_t) if C.fin(gate_t) and C.fin(done_t) else None),
                incumbents=val("lpd_incumbents"), incumbents_I=val("lpd_incumbents_I"),
                wait_s_max=ld.get("wait_s_max", row.get("lpd_wait_s_max")),
                host_s=ld.get("host_s", row.get("lpd_host_s")), solve_cpu_s=row.get("lpd_worker_solve_cpu_s"),
                nice=ld.get("nice"), concat_parts_max=summ.get("concat_parts_max"))


def _pts(cells: dict, inst: str, lab: str) -> list:
    return [cells[(inst, lab, s)]["row"].get("pt") if (inst, lab, s) in cells else None for s in SEEDS[inst]]


def rules(cells: dict) -> dict:
    ready = {k: c["mech"]["ready_after_gate"] for k, c in cells.items() if k[1] == T}
    ready_ok = {k: C.fin(v) and float(v) <= READY_MAX_S for k, v in ready.items()}
    inc = {(i, s): C.dig(cells.get((i, T, s), {}), "mech", "incumbents") for i in R2_HINT_INSTANCES for s in SEEDS[i]}
    inc_ok = {k: C.fin(v) and float(v) >= 1 for k, v in inc.items()}
    lp = {}
    for s in SEEDS[CA]:
        wt = C.dig(cells.get((CA, T, s), {}), "mech", "lp_wall")
        wc = C.dig(cells.get((CA, CTL, s), {}), "mech", "lp_wall")
        lp[s] = dict(t=wt, c=wc, ratio=(float(wt) / float(wc) if C.fin(wt) and C.fin(wc) and float(wc) > 0 else None))
        lp[s]["ok"] = bool(lp[s]["ratio"] is not None and lp[s]["ratio"] <= LP_RATIO_MAX + 1e-12)
    harm = {i: C.band_leg(_pts(cells, i, T), _pts(cells, i, CTL), FLOORS[i]) for i in
            tuple(R2_GUARD_INSTANCES) + tuple(R2_HINT_INSTANCES)}
    hint = C.ratio_leg({i: _pts(cells, i, T) for i in R2_HINT_INSTANCES},
                       {i: _pts(cells, i, CTL) for i in R2_HINT_INSTANCES}, HINT_MAX)
    fails = []
    if not all(ready_ok.values()) or len(ready_ok) == 0:
        fails.append("mechanism 1: decoder ready <= 10 s after the gate")
    if not all(inc_ok.values()):
        fails.append("mechanism 2: >= 1 decoded incumbent per seed on item_placement and sing44")
    if not all(x["ok"] for x in lp.values()):
        fails.append("mechanism 3: ca's LP solve time <= 1.10 x the control's")
    if not all(harm[i]["ok"] for i in R2_GUARD_INSTANCES):
        fails.append("no-harm (ca / glass4 / kumeu)")
    if not all(harm[i]["ok"] for i in R2_HINT_INSTANCES):
        fails.append("no-harm on the hint instances (amendment)")
    if not hint["present"]:
        fails.append("no hint")
    return dict(ready=ready, ready_ok=ready_ok, incumbents=inc, incumbents_ok=inc_ok, lp=lp, harm=harm, hint=hint,
                fails=fails, verdict="ADOPT" if not fails else "NOT ADOPTED")


def a3_beside(runs: pd.DataFrame) -> dict:
    """D-017's A3@ld1 rows (set dev): lpd_wait_s_max and lpd_host_s per instance and seed (reported, not gating)."""
    df = runs[(runs["set"] == A3_SET) & (runs["arm"] == A3_ARM) & runs["instance"].isin(list(SEEDS))]
    out = {}
    for r in df.to_dict("records"):
        out[(r["instance"], int(r["seed"]))] = dict(wait_s_max=r.get("lpd_wait_s_max"), host_s=r.get("lpd_host_s"))
    return out


def reading(runs: pd.DataFrame, root: Path, set_: str, base: str, nice: int | None, keep_failed: bool,
            ext: dict | None = None) -> dict:
    exp = expected(base, nice, set_)
    ext = ext if ext is not None else C.ext_harness()
    sel, dropped = C.select(runs, set_, (T, CTL), ALL_SEEDS, BUDGET, keep_failed)
    sel = sel[[s in SEEDS.get(i, ()) for i, s in zip(sel["instance"], sel["seed"])]] if len(sel) else sel
    by = {(r["instance"], r["arm"], int(r["seed"])): r for r in sel.to_dict("records")}
    failed = C.failed_keys(runs, set_, (T, CTL), ALL_SEEDS, BUDGET)
    legs: dict[str, list[str]] = {k: [] for k in LEGS}
    legs["complete"], present = C.completeness(list(exp), by, failed)
    cells, notes, kept = {}, [], []
    mode = base_mode(base)
    for k in present:
        r = by[k]
        rj, summ = C.run_files(r, root)
        kind = "treatment" if k[1] == T else "control"
        for leg, probs in C.base_problems(r, rj, summ, exp[k], ext).items():
            legs[leg] += [f"{k[0]} {k[1]} s{k[2]}: {x}" for x in probs]
        fp, note = flag_problems(mode, r, summ)
        legs["flag"] += [f"{k[0]} {k[1]} s{k[2]}: {x}" for x in fp]
        legs["decoder"] += [f"{k[0]} {k[1]} s{k[2]}: {x}" for x in decoder_problems(kind, r, summ)]
        if note:
            notes.append(f"{k[0]} {k[1]} s{k[2]}: {note}")
        if not C.fin(r.get("pt")):
            legs["complete"].append(f"no P(120): {k[0]} {k[1]} s{k[2]}")
        if r.get("status") == "failed":
            kept.append(k)
        cells[k] = dict(row=r, summ=summ, kind=kind, mech=mechanism(r, summ))
    legs["block"] = C.block_problems(cells, [(i, s, [(i, T, s), (i, CTL, s)]) for i in SEEDS for s in SEEDS[i]])
    bad = [k for k in LEGS if legs[k]]
    val = dict(verdict="INVALID" if bad else "VALID", failing=bad, legs=legs, notes=notes,
               deviation=[f"{i} {a} s{s} failed and is read as D-002 §6 keeps a crash (D-026: an outcome of its arm)"
                          for i, a, s in kept])
    out = dict(keep_failed=keep_failed, base=base, nice=nice, n_expected=len(exp), n_present=len(cells), validity=val,
               duplicates=dropped, cells=cells, rules=None, verdict="INVALID", a3=a3_beside(runs))
    if val["verdict"] == "VALID":
        out["rules"] = rules(cells)
        out["verdict"] = out["rules"]["verdict"]
    return out


def report(runs: pd.DataFrame, root: Path = ROOT, set_: str = R2_SET, base: str = "bsync", nice: int | None = None,
           failed: str = "both") -> dict:
    if base not in R2_BASES:
        raise ValueError(f"--base: one of {', '.join(R2_BASES)}")
    if "bsync" in base and nice is None:
        raise ValueError("--nice is required with a bsync base (the value the job file was written with)")
    nice = 0 if nice is None else nice
    cand = C.candidates(runs, set_, (T, CTL), ALL_SEEDS, BUDGET)
    fails = sorted(str(r) for r in cand[cand["status"] == "failed"]["run_id"])
    modes = {"exclude": [False], "keep": [True], "both": [True, False] if fails else [True]}[failed]
    ext = C.ext_harness()
    return dict(set=set_, base=base, nice=nice, failed_runs=fails,
                other_rows=C.others(runs, set_, (T, CTL), ALL_SEEDS, BUDGET),
                readings=[reading(runs, root, set_, base, nice, k, ext) for k in modes])


def _band_line(i: str, x: dict) -> str:
    if x.get("diff") is None:
        return f"  {i}: {x.get('why')} -> FAIL"
    return (f"  {i}: T {' / '.join(C.f(v) for v in x['t'])} (mean {C.f(x['mean_t'])}), C {' / '.join(C.f(v) for v in x['c'])}"
            f" (mean {C.f(x['mean_c'])}); diff {x['diff']:+.3f}, band {x['band']:.3f} -> {'ok' if x['ok'] else 'FAIL'}")


def render(rep: dict) -> str:
    L = [f"D-026 report (C1 diagnosis R2), set {rep['set']}, base {rep['base']}, decoder nice +{rep['nice']}: rules "
         f"fixed in D-026 before any run; interim; a hint never declares success (D-024)"]
    if rep["failed_runs"]:
        L.append(f"failed runs: {', '.join(rep['failed_runs'])}")
    if rep["other_rows"]:
        L.append(f"rows of the set no rule reads: {', '.join(rep['other_rows'])}")
    for rd in rep["readings"]:
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
        L.append(f"per run (T = {T}, C = {CTL}; reported):")
        for k, cl in sorted(rd["cells"].items()):
            m, r = cl["mech"], cl["row"]
            a3 = rd["a3"].get((k[0], k[2]), {})
            L.append(f"  {k[0]:<28} {k[1]:<9} s{k[2]}  P {C.f(r.get('pt'))}  LP done {C.f(m['lp_done_t'], 2)} s "
                     f"(wall {C.f(m['lp_wall'], 2)})  gate {m['gate']} at {C.f(m['gate_t'], 2)} (lag "
                     f"{C.f(m['gate_lag'], 2)})  ready +{C.f(m['ready_after_gate'], 2)} s  decoded inc {m['incumbents']}"
                     f" (I {m['incumbents_I']})  wait max {C.f(m['wait_s_max'], 4)} host {C.f(m['host_s'], 3)} s "
                     f"(A3@ld1 {C.f(a3.get('wait_s_max'), 4)} / {C.f(a3.get('host_s'), 3)})  decoder CPU "
                     f"{C.f(m['solve_cpu_s'], 2)} s  parts max {m['concat_parts_max']}  {r.get('status')}")
        if v["verdict"] != "VALID":
            L.append("RULES NOT READ: validity fails (D-026) -> INVALID")
            continue
        ru = rd["rules"]
        L.append(f"Mechanism 1 (ready <= {READY_MAX_S:g} s after the gate): " + ", ".join(
            f"{i} s{s} {C.f(x, 2)}{'' if ru['ready_ok'][(i, a, s)] else ' FAIL'}" for (i, a, s), x in
            sorted(ru["ready"].items())))
        L.append("Mechanism 2 (>= 1 decoded incumbent per seed): " + ", ".join(
            f"{i} s{s} {x}{'' if ru['incumbents_ok'][(i, s)] else ' FAIL'}" for (i, s), x in
            sorted(ru["incumbents"].items())))
        L.append(f"Mechanism 3 (ca's LP worker solve wall, T <= {LP_RATIO_MAX} x C): " + "; ".join(
            f"s{s} {C.f(x['t'], 2)} vs {C.f(x['c'], 2)} (ratio {C.f(x['ratio'])}) {'ok' if x['ok'] else 'FAIL'}"
            for s, x in ru["lp"].items()))
        L.append("No-harm (one-sided, seed means; floors ca 0.02, others 0.05; item / sing44 by the amendment):")
        for i, x in ru["harm"].items():
            L.append(_band_line(i, x))
        h = ru["hint"]
        if h["ratio"] is None:
            L.append(f"Hint: {h.get('why')} -> no hint")
        else:
            L.append(f"Hint (futility-only): ΣP(T) / ΣP(C) over {', '.join(R2_HINT_INSTANCES)} x seeds 0–3 = "
                     f"{h['sum_t']:.4f} / {h['sum_c']:.4f} = {h['ratio']:.4f} (<= {HINT_MAX}) -> "
                     f"{'PRESENT' if h['present'] else 'absent'}; per instance "
                     + ", ".join(f"{i} {C.f(x['ratio'])}" for i, x in h["per"].items()))
        L.append(f"VERDICT: {ru['verdict']}" + (f" — failing: {', '.join(ru['fails'])}" if ru["fails"] else ""))
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="d026_report", description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=DATASETS / "runs.parquet")
    ap.add_argument("--root", type=Path, default=ROOT, help="the run directories' root (run_dir is relative to it)")
    ap.add_argument("--set", default=R2_SET)
    ap.add_argument("--base", required=True, choices=tuple(R2_BASES),
                    help="as the job file was written (make-jobs r2-ld --base …)")
    ap.add_argument("--nice", type=int, choices=R2_NICE, default=None,
                    help="as the job file was written (required with a bsync base)")
    ap.add_argument("--failed", choices=("both", "exclude", "keep"), default="both")
    ap.add_argument("--json", type=Path, help="write the report as JSON here too")
    a = ap.parse_args(argv)
    try:
        runs = pd.read_parquet(a.runs)
        rep = report(runs, a.root, a.set, a.base, a.nice, a.failed)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(render(rep))
    if a.json:
        out = dict(rep, readings=[{k: v for k, v in rd.items() if k not in ("cells", "a3")} | dict(
            runs={f"{i} {arm} s{s}": dict(run_id=cl["row"]["run_id"], pt=cl["row"].get("pt"),
                                          status=cl["row"].get("status"), mechanism=cl["mech"])
                  for (i, arm, s), cl in rd["cells"].items()}) for rd in rep["readings"]])
        a.json.write_text(json.dumps(C.jsonable(out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
