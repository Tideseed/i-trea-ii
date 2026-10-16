"""Tables and figures of the phase-1 interim report (`paper/generated/*.tex`, `paper/figures/*.pdf`).

Deterministic: the same inputs give byte-identical files. Every number is either computed here from the inputs below
or encoded as data with a `source:` comment naming the record it is quoted from; where a quoted number can also be
recomputed from the run table, it is, and the two must agree (the script stops otherwise). Nothing here decides
anything: the tables restate decision records and saved reports (vault/50-decisions D-011 ... D-033) for the paper.

Inputs (read only; the run table is never rebuilt here):
  - datasets/runs.parquet                 (`itrea build-runs`; the 2026-10-16 13:01 build, D-031's set included)
  - datasets/incumbents_checked.parquet   (the checked incumbents of every run; fig_pi)
  - traces/raw/<set>/<run>/<role>.incumbents.npz  (fig_pi: each file's sha256 must match the checked rows)
  - logs/d030-reading.json                (`itrea c1r-report --a9 ysync-ld0 --json`, D-030's saved reading)
  - logs/d031-reading.json                (`tools/d031_report.py --d030-json logs/d030-reading.json --json`, D-031's
                                           saved four-seed reading: readings[0] keeps the crashed runs, readings[1]
                                           leaves them out)
  - tools/d022_report.py                  (D-022's selection and C1 functions, as c1r_report.py uses them)
  - itrea.analysis.mipfeas / itx_report   (D-021's M4 and M6 readings, recomputed for list_calib)
Outputs:
  paper/generated/table_devrecord.tex  (tab:devrecord)   the development tests D-011 ... D-032, numbered 1-18
  paper/generated/table_readings.tex   (tab:readings)    the main text's checkpoint readings (tabular only; the float
                                                         and its caption are in paper/sections/results.tex)
  paper/generated/table_d031.tex       (tab:d031)        Appendix B: the 17 instances, D-031 per instance on four seeds
                                                         (A0, A6F, A9, d, the halves, the harm leg)
  paper/generated/list_calib.tex       (a list)          D-021's harness measurements M1-M4 and M6
  paper/generated/table_chromatic.tex  (tab:chromatic)   cuOpt's first incumbent on chromaticindex1024-7, every run, in
                                                         the five groups of the main text's Table tab:loss
  paper/figures/fig_pi.pdf             (fig:pi)          primal gap curves on one C1 instance (rule below)
The per-instance two-seed readings (10-13, D-030) and the D-013 lines are still computed and checked (they feed
table_readings and table_d031) but no longer written as tables of their own (appendix rewrite, 2026-10-16).

Usage (repository root; untimed analysis, pinned, never beside timed runs with default thread counts):

    OMP_NUM_THREADS=1 nice -n 19 taskset -c 10-14 .venv/bin/python tools/paper_tables.py           # write
    OMP_NUM_THREADS=1 nice -n 19 taskset -c 10-14 .venv/bin/python tools/paper_tables.py --check   # compare only

`--check` regenerates every file in memory and compares it byte for byte with the file on disk (exit 1 on any
difference). Both modes stop (exit 1, nothing written) if a generated file would hold a `\\pending` marker, and the
script stops when D-031's set (`c1r-ysync-ld0-s23`) is missing from the run table or its saved reading is not the
four-seed VALID reading of record.

fig_pi's instance rule (fixed before plotting): among the fixed 11 (D-022's C1_INSTANCES) with no failed run of either
arm in D-030's set, the instance whose A9@c1r seed-mean P(120) is closest to the fixed-11 mean A9 P (D-030's hint
`mean_P_a9`); ties by the smaller distance, then by name. Both seeds are drawn (one panel each), so no seed is chosen.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from itrea.analysis import metrics as M  # noqa: E402
from itrea.analysis.stats import effect_d  # noqa: E402
from itrea.experiments import make_jobs as mj  # noqa: E402
from itrea.paths import DATASETS, ROOT  # noqa: E402

_spec = importlib.util.spec_from_file_location("d022_report_for_paper",
                                               Path(__file__).resolve().parent / "d022_report.py")
D22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D22)

GEN = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"
D030_JSON = ROOT / "logs" / "d030-reading.json"
D031_JSON = ROOT / "logs" / "d031-reading.json"
D031_SET = "c1r-ysync-ld0-s23"
C1R_SET, C1R_A9, C1R_A6F = "c1r-ysync-ld0", "A9@c1r", "A6F@c1r"
OLD_SET, OLD_A9, OLD_A6F = D22.SET, D22.TREAT, D22.A6F            # itx-r13, A9@r13x, A6F@r13x (2026-10-13)
A0, A0_SETS = D22.A0, D22.A0_SETS                                 # SCIP alone, sets dev / classprobe (10-06 / 10-07)
FIXED11 = tuple(D22.C1_INSTANCES)
CA = "ca__Scenario400_reserves_1"     # provisional reference (best of study): re-scored by build-runs
SEEDS = (0, 1)
SEEDS23 = (2, 3)
SEEDS4 = (0, 1, 2, 3)
T = 120.0
TOL = 1e-9
HEADER = "% Generated by tools/paper_tables.py -- do not edit by hand; rerun the script.\n"


# ----------------------------------------------------------------------------------------------- formatting
def f3(x: float) -> str:
    return f"{x:.3f}"


def sd(x: float, nd: int = 3) -> str:
    """A signed number in math mode; an exact zero prints as 0 (a tie at P = 1 on both sides)."""
    if x == 0:
        return "0"
    s = f"{x:+.{nd}f}"
    return f"${s}$"


def ci(lo: float, hi: float, nd: int = 3) -> str:
    return f"[{sd(lo, nd)}, {sd(hi, nd)}]"


def tt(name: str) -> str:
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def ascii_only(name: str, text: str) -> str:
    bad = sorted({c for c in text if ord(c) > 126})
    if bad:
        raise SystemExit(f"{name}: non-ASCII characters in generated LaTeX: {bad!r}")
    return text


class Mismatch(SystemExit):
    pass


def agree(label: str, got: float, want: float, tol: float = TOL) -> None:
    if not (math.isfinite(got) and math.isfinite(want) and abs(got - want) <= tol):
        raise Mismatch(f"MISMATCH {label}: computed {got!r} vs source {want!r} (tol {tol:g})")


def agree_at(label: str, got: float, want: float, nd: int) -> None:
    """Agreement at the printed precision (a quoted record against a recomputation)."""
    if round(got, nd) != round(want, nd):
        raise Mismatch(f"MISMATCH {label}: computed {got:.{nd + 3}f} vs recorded {want:.{nd}f}")


# ----------------------------------------------------------------------------------------------- inputs
def load() -> dict:
    runs = pd.read_parquet(DATASETS / "runs.parquet")
    d030 = json.loads(D030_JSON.read_text())
    if not D031_JSON.exists():
        raise SystemExit(f"{D031_JSON.relative_to(ROOT)} is missing: run tools/d031_report.py --d030-json "
                         "logs/d030-reading.json --json logs/d031-reading.json after itrea build-runs")
    if not bool((runs["set"] == D031_SET).any()):
        raise SystemExit(f"D-031's set {D031_SET} is not in the run table: run itrea build-runs first")
    d031 = json.loads(D031_JSON.read_text())
    return dict(runs=runs, d030=d030, d031=d031, dev17=list(mj.development_names()))


def seed_means(runs: pd.DataFrame, sets, arms, instances, keep_failed: bool = True,
               seeds: tuple[int, ...] = SEEDS) -> tuple[dict, pd.DataFrame]:
    """{(arm, instance): seed-mean P(120)} over `seeds` (default 0-1) with D-022's selection (failed runs kept as
    outcomes); every (arm, instance) must have every seed."""
    df, dropped = D22.select(runs, sets, arms, seeds, T, keep_failed=keep_failed, instances=instances)
    if dropped:
        raise SystemExit(f"duplicates in {sets} {arms}: {dropped}")
    n = df.groupby(["arm", "instance"])["seed"].nunique()
    short = n[n != len(seeds)]
    if len(short) or len(n) != len(arms) * len(instances):
        raise SystemExit(f"{sets} {arms}: not every seed of {seeds} for {list(short.index)} ({len(n)} arm-instances)")
    return df.groupby(["arm", "instance"])["pt"].mean().to_dict(), df


# ----------------------------------------------------------------------------------------------- c1_data
def c1_data(inp: dict) -> dict:
    runs, d030, dev17 = inp["runs"], inp["d030"], inp["dev17"]
    r0 = d030["readings"][0]
    if not r0["keep_failed"] or r0["set"] != C1R_SET or r0["a9"] != "ysync-ld0":
        raise SystemExit("logs/d030-reading.json: readings[0] is not D-030's kept reading")
    rules = r0["rules"]
    new, _ = seed_means(runs, (C1R_SET,), (C1R_A9, C1R_A6F), dev17)
    old, _ = seed_means(runs, (OLD_SET,), (OLD_A9, OLD_A6F), dev17)
    a0, _ = seed_means(runs, A0_SETS, (A0,), dev17, keep_failed=False)
    rows, drift = {}, []
    for i in dev17:
        jn, jo = rules["table"][i], rules["then"]["table"][i]
        pairs = ((f"D-030 A6F {i}", new[(C1R_A6F, i)], jn["p_a6f"]), (f"D-030 A9 {i}", new[(C1R_A9, i)], jn["p_a9"]),
                 (f"D-030 d {i}", effect_d(new[(C1R_A6F, i)], new[(C1R_A9, i)]), jn["d"]),
                 (f"10-13 A6F {i}", old[(OLD_A6F, i)], jo["p_a6f"]), (f"10-13 A9 {i}", old[(OLD_A9, i)], jo["p_a9"]),
                 (f"10-13 d {i}", effect_d(old[(OLD_A6F, i)], old[(OLD_A9, i)]), jo["d"]))
        for lab, got, want in pairs:
            if i == CA and abs(got - want) > TOL:
                # ca's reference is provisional (best of study, D-031 § Rules): any build-runs after a better ca
                # solution re-scores every ca run. The saved D-030 reading stays the printed source; the drift is
                # reported in the summary, not fatal.
                drift.append((lab, got, want))
            else:
                agree(lab, got, want)
        rows[i] = dict(old_a6f=jo["p_a6f"], old_a9=jo["p_a9"], old_d=jo["d"], new_a6f=jn["p_a6f"], new_a9=jn["p_a9"],
                       new_d=jn["d"], a0=a0[(A0, i)])
    # the markers, from the saved reading's runs and the run table
    failed = sorted((v["run_id"].split("__")[0], int(k.split(" s")[-1])) for k, v in r0["runs"].items()
                    if v["status"] == "failed")
    if failed != [("comp21-2idx", 0), ("comp21-2idx", 1)]:
        raise SystemExit(f"D-030 failed runs changed: {failed}")
    crash = {k.split(" s")[-1]: v["pt"] for k, v in r0["runs"].items() if v["status"] == "failed"}
    agree("comp21 A6F s0 (kept, truncated)", crash["0"], 0.7761086824722132)
    agree("comp21 A6F s1 (kept, truncated)", crash["1"], 0.7816434471011805)
    rail = runs[(runs["set"] == C1R_SET) & (runs["instance"] == "rail01")].set_index(["arm", "seed"])["pt"]
    if not (rail[(C1R_A6F, 0)] == 1.0 and rail[(C1R_A6F, 1)] < 1.0 and rail[(C1R_A9, 0)] == 1.0
            and rail[(C1R_A9, 1)] == 1.0):
        raise SystemExit("rail01's D-030 pattern changed (note b)")
    floor = {i for i, r in rows.items() if max(round(r[k], 3) for k in ("old_a6f", "old_a9", "new_a6f", "new_a9"))
             <= 0.011}
    if floor != {"radiationm18-12-05", "fhnw-binpack4-48", "neos-3004026-krka"}:
        raise SystemExit(f"floor set changed: {sorted(floor)}")
    # hint and guard: the saved reading's, recomputed from its per-instance seed means (themselves checked against
    # the run table above), and against D-022's / D-030's recorded values at three decimals
    h, g = rules["hint"], rules["guard"]
    ho, go = rules["then"]["hint"], rules["then"]["guard"]
    for lab, hh, pre in (("D-030", h, "new"), ("10-13", ho, "old")):
        ma9 = float(np.mean([rows[i][f"{pre}_a9"] for i in FIXED11]))
        ma6 = float(np.mean([rows[i][f"{pre}_a6f"] for i in FIXED11]))
        agree(f"{lab} hint mean A9", ma9, hh["mean_P_a9"])
        agree(f"{lab} hint mean A6F", ma6, hh["mean_P_a6f"])
        agree(f"{lab} hint ratio", ma9 / ma6, hh["ratio"])
    agree("D-030 guard", float(np.mean([rows[i]["new_d"] for i in dev17])), g["mean_d"])
    agree("10-13 guard", float(np.mean([rows[i]["old_d"] for i in dev17])), go["mean_d"])
    agree_at("10-13 hint vs D-022 Result", ho["ratio"], rules["then"]["recorded"]["hint"], 3)
    agree_at("10-13 guard vs D-022 Result", go["mean_d"], rules["then"]["recorded"]["guard"], 3)
    agree_at("D-030 hint vs D-030 Result", h["ratio"], 0.809, 3)
    agree_at("D-030 guard vs D-030 Result", g["mean_d"], 0.050, 3)
    agree_at("ca 10-13 A6F (D-022 Result 0.586; re-scored)", rows["ca__Scenario400_reserves_1"]["old_a6f"], 0.587, 3)
    return dict(rows=rows, floor=floor, hint=h, guard=g, hint_old=ho, guard_old=go, crash=crash, drift=drift)


# ----------------------------------------------------------------------------------------------- d013_data
# source: D-022 § Result (2026-10-13 22:2x), "Reported, not criteria (D-013 terms)" — the fixed-11 lines of the 10-13
# reading; logs/d030-reading.json does not hold them (its `then` block has only hint, guard, vs_a0_17 and table).
D022_RECORDED = {
    "A6F": dict(n=11, better=6, worse=3, before=0.707, after=0.585, red=0.17, d=0.185, lo=0.032, hi=0.368, p=0.074,
                worst=1.000),
    "A0": dict(n=11, better=8, worse=0, before=0.880, after=0.585, red=0.34, d=0.331, lo=0.172, hi=0.500, p=0.0078,
               worst=1.000),
}


def d013_data(inp: dict) -> dict:
    runs, d030, dev17 = inp["runs"], inp["d030"], inp["dev17"]
    rules0, rules1 = d030["readings"][0]["rules"], d030["readings"][1]["rules"]
    if d030["readings"][1]["keep_failed"]:
        raise SystemExit("readings[1] is not the exclude reading")
    rec = D22.c1_reading(runs, (OLD_SET,), FIXED11, keep_failed=True, guard_instances=dev17)["blocks"]
    re_a6f, re_a0 = rec[OLD_A6F][OLD_A9]["all"], rec[A0][OLD_A9]["all"]
    checks, drift = [], []
    for key, got in (("A6F", re_a6f), ("A0", re_a0)):
        want = D022_RECORDED[key]
        for a, b in (("better", "better"), ("worse", "worse")):
            if int(got[a]) != int(want[b]):
                raise Mismatch(f"MISMATCH 10-13 vs {key} {a}: {got[a]} vs recorded {want[b]} (D-022 Result)")
        # The continuous fields must agree at the printed precision, with one stated exception: the reference of
        # ca__Scenario400_reserves_1 is provisional (best of study) and every build-runs after a better ca solution
        # re-scores every ca run (D-031 § Rules). D-022's Result read ca at 0.586 / +0.329, the 08:57 table at
        # 0.587 / +0.328. A difference of at most 0.005 is recorded as drift and printed in the summary beside the
        # quoted (recorded) value; anything larger stops the script.
        fields = (("mean_P_before", float(got["mean_P_before"]), want["before"], 3, "mean $P$ alone"),
                  ("mean_P_after", float(got["mean_P_after"]), want["after"], 3, "mean $P$ of A9"),
                  ("mean_d", float(got["mean_d"]), want["d"], 3, "mean $d$"),
                  ("ci_lo", float(got["ci"][0]), want["lo"], 3, "interval low end"),
                  ("ci_hi", float(got["ci"][1]), want["hi"], 3, "interval high end"),
                  ("worst_P_after", float(got["worst_P_after"]), want["worst"], 3, "worst $P$ after"),
                  ("p", float(got["p"]), want["p"], 3 if want["p"] >= 0.01 else 4, "$p$"))
        for name, g_, w_, nd, words in fields:
            if round(g_, nd) != round(w_, nd):
                if abs(g_ - w_) > 0.005:
                    raise Mismatch(f"MISMATCH 10-13 vs {key} {name}: {g_:.6f} vs recorded {w_} (D-022 Result)")
                drift.append((key, name, words, g_, w_, nd))
        checks.append((key, got))
    lines = dict(
        a6f_old=D022_RECORDED["A6F"],
        a6f_new=rules0["vs_a6f"], a6f_new_excl=rules1["vs_a6f"],
        a0_old=D022_RECORDED["A0"], a0_new=rules0["vs_a0"][OLD_A9]["all"],
        a0w_old=rules0["then"]["vs_a0_17"]["block"], a0w_new=rules0["vs_a0_17"]["block"],
        weak15=rules0["vs_a0_17"]["weak_set"], weak15_old=rules0["then"]["vs_a0_17"]["weak_set"],
        guard_old=rules0["then"]["guard"]["mean_d"], guard_new=rules0["guard"]["mean_d"])
    if lines["weak15"] != lines["weak15_old"] or len(lines["weak15"]) != 15:
        raise SystemExit("SCIP's weak set among the 17 differs between the readings")
    for k in ("a6f_new", "a6f_new_excl", "a0_new", "a0w_old", "a0w_new"):
        if lines[k]["worst_P_after"] != 1.0:
            raise SystemExit(f"{k}: worst P after is not 1.000")
    # D-030 Result / the phase-1 note quote these at three decimals; the saved report is the source
    agree_at("D-030 vs A6F d", lines["a6f_new"]["mean_d"], 0.235, 3)
    agree_at("D-030 vs A6F p", lines["a6f_new"]["p"], 0.049, 3)
    agree_at("D-030 vs A0 weak-15 d", lines["a0w_new"]["mean_d"], 0.502, 3)
    agree_at("10-13 vs A0 weak-15 after", lines["a0w_old"]["mean_P_after"], 0.473, 3)
    # the both-P = 1 ties each row's d's include (stats.compare's `excluded`), named in the caption
    ties = {"a6f_old": list(re_a6f["left_out_by_compare"]), "a6f_new": lines["a6f_new"]["left_out_by_compare"],
            "a6f_new_excl": lines["a6f_new_excl"]["left_out_by_compare"],
            "a0_old": list(re_a0["left_out_by_compare"]), "a0_new": lines["a0_new"]["left_out_by_compare"],
            "a0w_old": lines["a0w_old"]["left_out_by_compare"], "a0w_new": lines["a0w_new"]["left_out_by_compare"]}

    ca_old = rules0["then"]["table"]["ca__Scenario400_reserves_1"]
    agree_at("ca 10-13 d now (D-022 Result +0.329)", ca_old["d"], 0.328, 3)
    return dict(lines=lines, recomputed=checks, drift=drift, ties_raw=ties)


def d013_add_d031(c: dict, d31: dict) -> None:
    """Add D-031's four-seed lines (crashed runs kept, and left out) to the D-013 lines, and check that every
    line's ties (n - better - worse) are exactly its instances at P = 1 on both sides."""
    four, excl = d31["four"], d31["excl"]["four"]
    c["lines"].update(a6f_4=four["vs_a6f"], a6f_4_excl=excl["vs_a6f"], a0_4=four["vs_a0"][OLD_A9]["all"],
                      a0w_4=four["vs_a0_17"]["block"], guard_4=four["guard"]["mean_d"])
    for k in ("a6f_4", "a6f_4_excl", "a0_4", "a0w_4"):
        c["ties_raw"][k] = list(c["lines"][k]["left_out_by_compare"])
        if c["lines"][k]["worst_P_after"] != 1.0:
            raise SystemExit(f"{k}: worst P after is not 1.000")
    ties = c["ties_raw"]
    # every row's ties (n - better - worse) are exactly its both-P = 1 instances, including the quoted 10-13 lines
    for k in ("a6f_old", "a6f_new", "a6f_new_excl", "a6f_4", "a6f_4_excl", "a0_old", "a0_new", "a0w_old", "a0w_new",
              "a0_4", "a0w_4"):
        b = c["lines"][k]
        if b["n"] - b["better"] - b["worse"] != len(ties[k]):
            raise SystemExit(f"D-013 line {k}: {b['n']} - {b['better']} - {b['worse']} ties but {ties[k]} at P = 1")


# ----------------------------------------------------------------------------------------------- list_calib
def calib_data(inp: dict) -> dict:
    from itrea.analysis import itx_report, mipfeas
    runs = inp["runs"]
    m4 = mipfeas.report(runs, arm="A6F")
    agree_at("M4 R call", m4["call"]["R"], 1.313, 3)
    agree_at("M4 R call lo", m4["call"]["lo"], 1.091, 3)
    agree_at("M4 R call hi", m4["call"]["hi"], 1.618, 3)
    agree_at("M4 R launch", m4["launch"]["R"], 1.963, 3)
    agree_at("M4 R launch lo", m4["launch"]["lo"], 1.606, 3)
    agree_at("M4 R launch hi", m4["launch"]["hi"], 2.408, 3)
    lc = m4["late_clock"]["pooled"]
    agree_at("M4 late clock median", lc["median"], -0.119, 3)
    if (m4["n_runs"], m4["feasible"]["ours"], m4["feasible"]["gams_sample"], lc["n"], m4["verdict"]) != \
            (30, 27, 27, 10, "OUTSIDE"):
        raise SystemExit("M4 reading changed")
    if (mipfeas.R_LO, mipfeas.R_HI) != (0.870, 1.150):
        raise SystemExit("M4 band changed")
    # M4's checker rejections: incumbents of cuOpt alone (the only role of an A6F run) that failed our exact checker
    # and so never enter pt_gams (build_runs.role_steps keeps checked incumbents only), while GAMS's scoring reads
    # cuOpt's own trace. Counted from the run table; their effect on R is not computed anywhere.
    m4r = runs[(runs["set"] == "mipfeas") & (runs["arm"] == "A6F")]
    m4_rej = (int(m4r["n_inc_failed"].sum()), int((m4r["n_inc_failed"] > 0).sum()), int(len(m4r)))
    if m4_rej != (15, 5, 30):
        raise SystemExit(f"M4 checker rejections changed: {m4_rej} (the text says 15 in 5 of 30 runs)")
    m6 = itx_report.m6_report(runs)
    agree_at("M6 mean slowdown", m6["mean_slowdown"], 0.132, 3)
    agree_at("M6 hottest", m6["hottest_c"], 78.6, 1)
    if m6["lanes"] != 3:
        raise SystemExit("M6 lanes changed")
    counts = {s: runs[runs["set"] == s]["status"].value_counts().to_dict() for s in
              ("itx-m1", "itx-m2", "itx-m3", "mipfeas", "itx-m6", "itx-m6a")}
    want = {"itx-m1": {"completed": 12}, "itx-m2": {"completed": 4}, "itx-m3": {"completed": 6, "failed": 2},
            "mipfeas": {"completed": 30}, "itx-m6": {"completed": 17}, "itx-m6a": {"completed": 3}}
    if counts != want:
        raise SystemExit(f"harness run counts changed: {counts}")
    st = pd.read_csv(DATASETS / "mipfeas_strata.csv")
    nofit = sorted(st.loc[st["does_not_fit"].astype(bool), "name"])
    unseen = st["unseen"].astype(bool)
    hcu = st["H_cu"].astype(bool)
    fit = ~st["does_not_fit"].astype(bool)
    if (nofit, int(unseen.sum()), int((unseen & fit).sum()), int((unseen & hcu).sum()),
            int((unseen & hcu & fit).sum())) != (["eilA101-2", "nw04", "supportcase6"], 218, 215, 65, 64):
        raise SystemExit("M3 list or decision-set sizes changed")
    return dict(m4=m4, m6=m6, m4_rej=m4_rej)


def list_calib(c: dict) -> str:
    """Appendix C: D-021's harness measurements M1-M4 and M6 as a compact list (an itemize, not a float)."""
    m4, m6 = c["m4"], c["m6"]
    lc = m4["late_clock"]["pooled"]
    items = [
        # source: vault/30-results/2026-10-11 D-021 measurements M1-M3 and the interim calibration.md, Verdict and
        # § M1 (12 / 12 runs; R14 inert: cuOpt CPU-s per wall-s 5.02 / 4.99 / 5.03 unset vs 5.02 / 4.98 / 5.03 set);
        # run count checked against runs.parquet set itx-m1
        "\\textbf{M1 smoke} (12 runs of 120\\,s): every harness setting acts as specified; unsetting cuOpt's thread "
        "variables is inert (cuOpt CPU-s per wall-s 5.02 / 4.99 / 5.03 unset, 5.02 / 4.98 / 5.03 set). Passed.",
        # source: same note, § M2 and Verdict (R23: p95 <= 85 C, no 5-s sample >= 90 C, no clock cap, SM >= 2,350 MHz)
        "\\textbf{M2 soak} (4 runs of 600\\,s on $F$): steady-state p95 66--73\\,$^{\\circ}$C, maximum 76\\,$^{\\circ}$C, "
        "no sample $\\ge$ 90\\,$^{\\circ}$C, no clock cap, busy SM clock 2,392\\,MHz. Passed; $F$ accepted.",
        # source: same note, § M3; the list and the set sizes are checked against datasets/mipfeas_strata.csv
        "\\textbf{M3 memory} (8 instances, cuOpt alone at 40-GiB caps, 600\\,s): \\texttt{eilA101-2} and \\texttt{nw04} "
        "(cap hits) and \\texttt{supportcase6} (peak $>$ 0.8 $\\times$ 24\\,GiB) do not fit $F$; the unseen test "
        "set goes from 218 to 215 instances.",
        # source: same note, Addendum 2026-10-15 (M4 complete) and its 2026-10-16 correction; every number is
        # recomputed here with itrea.analysis.mipfeas.report(runs, arm="A6F") and equal at three decimals
        "\\textbf{M4 calibration} (D-020's 30-instance sample, cuOpt alone on $F$, 600\\,s), against GAMS's published "
        "cuOpt 26.08 run on a GB10: late-clock check passed (median first incumbent, ours $-$ GAMS, "
        f"{sd(lc['median'])}\\,s, $n$ {lc['n']}); $R$ = {f3(m4['call']['R'])} [{f3(m4['call']['lo'])}, "
        f"{f3(m4['call']['hi'])}] on the call clock (launch clock {f3(m4['launch']['R'])} [{f3(m4['launch']['lo'])}, "
        f"{f3(m4['launch']['hi'])}]), OUTSIDE the band [{f3(0.870)}, {f3(1.150)}]; feasible {m4['feasible']['ours']} vs "
        f"GAMS {m4['feasible']['gams_sample']}. $R$: ratio of shifted geometric means (shift 0.001) of GAMS's MIPfeas "
        "primal integral, ours over GAMS's, with a 95\\% bootstrap interval. Our checker rejected "
        f"{c['m4_rej'][0]} of cuOpt's incumbents in {c['m4_rej'][1]} of the {c['m4_rej'][2]} runs; GAMS's scoring "
        "would count them, and their effect on $R$ has not been computed.",
        # source: Development log "External harness in phase 1's last week" (M6; journal 2026-10-13 22:2x); recomputed
        # with itrea.analysis.itx_report.m6_report(runs)
        f"\\textbf{{M6 lanes}} (20 runs): SCIP alone's nodes/s are {100 * m6['mean_slowdown']:.1f}\\% lower beside 3 "
        f"busy lanes (rule: at most 3\\%); hottest zone {m6['hottest_c']:.1f}\\,$^{{\\circ}}$C. Verdict: 3 lanes.",
    ]
    s = HEADER
    s += ("% Inputs: vault/30-results/2026-10-11 D-021 measurements M1-M3 and the interim calibration.md (M1-M3, M4\n"
          "% addendum 2026-10-15); M4 and M6 recomputed from datasets/runs.parquet (itrea.analysis.mipfeas.report,\n"
          "% itx_report.m6_report); M3's list from datasets/mipfeas_strata.csv. A list, not a float (Appendix C).\n")
    s += "\\begin{itemize}\\setlength{\\itemsep}{0pt}\\setlength{\\parskip}{0pt}\n"
    s += "".join(f"\\item {it}\n" for it in items)
    s += "\\end{itemize}\n"
    return s


# ----------------------------------------------------------------------------------------------- table_chromatic
CHROM = "chromaticindex1024-7"
NONE, FULL, LOW = "no EA", "EA at full duty", "EA duty reduced"
# (set, arm) -> (what shares the GPU with cuOpt, group, record). Every cuOpt-carrying run of chromaticindex1024-7 in
# the run table must have a row here and every row a run (checked), so nothing is left out or added.
# source: run labels and meanings from D-014 (classprobe A6), D-015 (a9pilot), D-018 stage 0 table (bo0 full duty,
# bof50 / bof25 / bof12 forced duty 1/2, 1/4, 1/8) and stage 1 (bo1 active, bo1s = the same values in shadow mode,
# full duty), D-022 (itx-r13), D-023 (boY = phase Y: duty 1/2 until cuOpt's first incumbent, 30-s expiry), D-027
# (A6@r3 three threads, A2@r3 beside SCIP, A6F@r3), D-028 (boY under ext with R13 + R14 off), D-030 (c1r, seeds
# 0-1) and D-031 (the same labels at seeds 2-3, set c1r-ysync-ld0-s23)
CHROM_ARMS = {
    ("classprobe", "A6"): ("cuOpt alone, 3 threads", NONE, "D-014"),
    ("a9pilot", "A2"): ("SCIP and cuOpt race", NONE, "D-015"),
    ("backoff0", "A6"): ("cuOpt alone, 3 threads", NONE, "D-018"),
    ("backoff1", "A6"): ("cuOpt alone, 3 threads", NONE, "D-018"),
    ("itx-r13", "A6F@r13x"): ("cuOpt alone on F", NONE, "D-022"),
    ("backoff2", "A6F"): ("cuOpt alone on F", NONE, "D-023"),
    ("r3-diag", "A6F@r3"): ("cuOpt alone on F", NONE, "D-027"),
    ("r3-diag", "A6@r3"): ("cuOpt alone, 3 threads", NONE, "D-027"),
    ("r3-diag", "A2@r3"): ("cuOpt, 3 threads, beside SCIP", NONE, "D-027"),
    ("d028", "A6F"): ("cuOpt alone on F", NONE, "D-028"),
    ("c1r-ysync-ld0", "A6F@c1r"): ("cuOpt alone on F", NONE, "D-030"),
    ("c1r-ysync-ld0-s23", "A6F@c1r"): ("cuOpt alone on F", NONE, "D-031"),
    ("a9pilot", "A9@a9p"): ("A9, pilot", FULL, "D-015"),
    ("backoff0", "A9@bo0"): ("A9, back-off shadow", FULL, "D-018"),
    ("backoff1", "A9@bo1s"): ("A9, back-off shadow", FULL, "D-018"),
    ("itx-r13", "A9@r13x"): ("A9, old default", FULL, "D-022"),
    ("backoff2", "A9@bo1s"): ("A9, back-off shadow", FULL, "D-023"),
    ("c1r-ysync-ld0", "A9@c1r"): ("A9, frozen-v2 candidate", FULL, "D-030"),
    ("c1r-ysync-ld0-s23", "A9@c1r"): ("A9, frozen-v2 candidate", FULL, "D-031"),
    ("backoff0", "A9@bof50"): ("A9, EA duty forced to 1/2", LOW, "D-018"),
    ("backoff0", "A9@bof25"): ("A9, EA duty forced to 1/4", LOW, "D-018"),
    ("backoff0", "A9@bof12"): ("A9, EA duty forced to 1/8", LOW, "D-018"),
    ("backoff1", "A9@bo1"): ("A9, back-off active", LOW, "D-018"),
    ("backoff2", "A9@boY"): ("A9, phase Y only", LOW, "D-023"),
    ("d028", "A9@boY"): ("A9, phase Y only", LOW, "D-028"),
}


def harness_label(h: str) -> str:
    if not isinstance(h, str) or not h.strip():
        return "old"
    j = json.loads(h)
    if j.get("cuopt_gap_zero") and j.get("cuopt_env_unset"):
        return "ext"
    if not j.get("cuopt_gap_zero") and not j.get("cuopt_env_unset"):
        return "ext, no R13/R14"
    raise SystemExit(f"unknown harness {h}")


def chrom_data(inp: dict) -> list[dict]:
    runs = inp["runs"]
    c = runs[(runs["instance"] == CHROM) & runs["arm"].astype(str).str.match(r"^(A2|A6|A6F|A9)(@|$)")]
    c = c[(c["budget"] == T) & ~c["synthetic"].fillna(False).astype(bool) & ~c["block_void"].fillna(False).astype(bool)]
    if set(c["status"]) != {"completed"} or c["cuopt_t_first"].isna().any() or not c["cuopt_t_first_event"].all():
        raise SystemExit("chromaticindex: a cuOpt run failed or found no incumbent")
    keys = set(zip(c["set"], c["arm"]))
    if keys != set(CHROM_ARMS):
        raise SystemExit(f"chromaticindex arms changed: extra {keys - set(CHROM_ARMS)}, missing "
                         f"{set(CHROM_ARMS) - keys}")
    out = []
    for (set_, arm), (what, group, rec) in CHROM_ARMS.items():
        g = c[(c["set"] == set_) & (c["arm"] == arm)].sort_values("seed")
        harness = {harness_label(h) for h in g["harness"]}
        days = {str(w)[:8] for w in g["window_id"]}
        if len(harness) != 1 or len(days) != 1:
            raise SystemExit(f"{set_} {arm}: mixed harness or days")
        day = days.pop()
        out.append(dict(set=set_, arm=arm, what=what, group=group, rec=rec, harness=harness.pop(),
                        date=f"{day[4:6]}-{day[6:8]}", seeds=[int(s) for s in g["seed"]],
                        t=[float(x) for x in g["cuopt_t_first"]], p=[float(x) for x in g["pt"]]))
    order = {NONE: 0, FULL: 1, LOW: 2}
    out.sort(key=lambda r: (order[r["group"]], r["date"], r["set"], r["arm"]))
    # D-027 rule 4 as its amendment states it: every A9 run at 62-66 s except bo1, bof25, bof50 (old harness)
    early = {(r["set"], r["arm"]) for r in out if r["arm"].startswith("A9") and min(r["t"]) < 62.0}
    if early != {("backoff1", "A9@bo1"), ("backoff0", "A9@bof25"), ("backoff0", "A9@bof50")}:
        raise SystemExit(f"D-027's exceptions changed: {early}")
    late = [x for r in out if r["arm"].startswith("A9") and (r["set"], r["arm"]) not in early for x in r["t"]]
    if not (min(late) >= 62.0 and max(late) <= 66.0):
        raise SystemExit("A9's late first incumbents are not within 62-66 s")
    get = {(r["set"], r["arm"]): r for r in out}
    # source: D-027 Result (rule 4): 9.80 / 7.33, 9.24 / 8.21, 6.53 / 9.65 s; D-028 Result: A6F 7.45 / 9.91 -> L 19.82,
    # A9@boY r13-off 63.25 / 63.24 s; D-023 Result: boY 63.47 / 63.41 s, A6F 10.87 / 10.52 s; D-018 bo1 9.38 / 12.35 s
    for key, want in ((("r3-diag", "A6F@r3"), (9.80, 7.33)), (("r3-diag", "A6@r3"), (9.24, 8.21)),
                      (("r3-diag", "A2@r3"), (6.53, 9.65)), (("d028", "A6F"), (7.45, 9.91)),
                      (("d028", "A9@boY"), (63.25, 63.24)), (("backoff2", "A9@boY"), (63.47, 63.41)),
                      (("backoff2", "A6F"), (10.87, 10.52)), (("backoff1", "A9@bo1"), (9.38, 12.35)),
                      (("c1r-ysync-ld0", "A9@c1r"), (65.24, 65.22)), (("c1r-ysync-ld0", "A6F@c1r"), (7.94, 10.31))):
        for got, w in zip(get[key]["t"], want):
            agree_at(f"chromatic t {key}", got, w, 2)
    agree_at("D-028 L", 2 * max(get[("d028", "A6F")]["t"]), 19.82, 2)
    return out


# The five groups of the main text's Table tab:loss (paper/sections/results.tex) and their run counts there, in its
# words (review fix 2026-10-16: the same labels as Table tab:loss -- earlier / final harness, the coupled system, first
# solution -- and the last group split in two: phase Y (D-023, D-028) and the forced 1/8 duty (D-018 bof12)).
LOSS_GROUPS = (
    ("nothing, or SCIP only (no EA)", 22),
    ("the EA using the GPU freely (the coupled system, any configuration)", 12),
    ("the EA on the GPU $\\frac14$ or $\\frac12$ of the time, or backing off while it adds nothing (earlier harness)",
     4),
    ("the EA backing off until cuOpt's first solution (final harness, with or without its two cuOpt settings)", 4),
    ("the EA on the GPU $\\frac18$ of the time (earlier harness)", 1),
)
LATE_LOW_ONE_EIGHTH = ("backoff0", "A9@bof12")   # source: D-018 stage 0 (bof12 = forced duty 1/8, old harness)


def table_chromatic(rows: list[dict]) -> str:
    """Every run's first-incumbent time, sorted, in the main text's five groups (Table tab:loss): no EA; full duty;
    reduced duty with cuOpt back early (< 62 s, D-027's three exceptions); phase Y with cuOpt still late (D-023,
    D-028); the forced 1/8 duty with cuOpt still late (D-018)."""
    runs_: list[list[tuple[str, float]]] = [[], [], [], [], []]
    for r in rows:              # chrom_data's order: group, date, set, arm; seeds in order within a row
        for t in r["t"]:
            if r["group"] == NONE:
                k = 0
            elif r["group"] == FULL:
                k = 1
            elif t < 62.0:
                k = 2
            elif (r["set"], r["arm"]) == LATE_LOW_ONE_EIGHTH:
                k = 4
            else:
                k = 3
            runs_[k].append((r["rec"], t))
    if {rec for rec, _ in runs_[3]} != {"D-023", "D-028"} or [rec for rec, _ in runs_[4]] != ["D-018"]:
        raise SystemExit("chromaticindex: the late back-off groups are not D-023 / D-028 and D-018's 1/8 duty")
    for (what, n), g in zip(LOSS_GROUPS, runs_):
        if len(g) != n:
            raise SystemExit(f"chromaticindex group '{what}': {len(g)} runs, the main text's Table tab:loss says {n}")
    s = HEADER
    s += ("% Input: datasets/runs.parquet, instance chromaticindex1024-7, every run with cuOpt (arms A2, A6, A6F, A9),\n"
          "% 120 s: cuopt_t_first (cuOpt's first checked incumbent, call clock), harness, window_id. Spot values\n"
          "% checked against D-018, D-023, D-027, D-028 and D-030 in chrom_data; groups and counts as Table tab:loss.\n")
    s += "\\begin{table}[!tb]\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
    s += ("\\begin{tabular}{>{\\raggedright}p{5.7cm}r>{\\raggedright\\arraybackslash}p{9.3cm}}\n\\toprule\n"
          "what shares the GPU with cuOpt & runs & cuOpt's first solution (s), by decision record \\\\\n"
          "\\midrule\n")
    for (what, n), g in zip(LOSS_GROUPS, runs_):
        by_rec: dict[str, list[float]] = {}
        for rec, t in g:
            by_rec.setdefault(rec, []).append(t)
        vals = "; ".join(f"{rec}: " + ", ".join(f"{t:.2f}" for t in ts)
                         for rec, ts in sorted(by_rec.items()))
        s += f"{what} & {n} & {vals} \\\\\n"
    s += "\\bottomrule\n\\end{tabular}\n"
    s += ("\\caption{cuOpt's first checked solution (seconds since the solver call) on \\texttt{chromaticindex1024-7} "
          "in every 120-s run with cuOpt, 7--16 October, grouped as in Table~\\ref{tab:loss}.}\\label{tab:chromatic}\n")
    s += "\\end{table}\n"
    return s


# ----------------------------------------------------------------------------------------------- table_d031
# source: D-031 § Result (2026-10-16 13:1x), the printed values every recomputation below must equal at their printed
# precision: four seeds HINT 0.798 (P 0.550 / 0.690), GUARD +0.045; seeds 2-3 alone 0.787 / +0.034; seeds 0-1 (= D-030)
# 0.809 / +0.050; harm glass4 d -0.114 (band 0.057), comp21-2idx d -0.051 (band 0.050); vs cuOpt alone, fixed 11: 7 / 3,
# 0.690 -> 0.550 (-20 %), d +0.249 [+0.078, +0.441], p 0.027 (seeds 2-3 alone 7 / 1, -21 %); vs SCIP alone, fixed 11
# 8 / 0, -38 %; SCIP-weak 15: 12 / 0, 0.889 -> 0.448 (-50 %), d +0.499 [+0.323, +0.676], p 0.0005; 67 completed, 1 failed;
# VALID 136 of 136 (kept); exclude reading INCOMPLETE.
D031_RECORDED = dict(hint4=0.798, a9_4=0.550, a6f_4=0.690, guard4=0.045, hint23=0.787, guard23=0.034, hint01=0.809,
                     guard01=0.050, harm={"glass4": (-0.114, 0.057), "comp21-2idx": (-0.051, 0.050)},
                     vs_a6f=dict(better=7, worse=3, red=0.20, d=0.249, lo=0.078, hi=0.441, p=0.027),
                     vs_a6f23=dict(better=7, worse=1, red=0.21),
                     vs_a0_11=dict(better=8, worse=0, red=0.38),
                     vs_a0_15=dict(better=12, worse=0, before=0.889, after=0.448, red=0.50, d=0.499, lo=0.323,
                                   hi=0.676, p=0.0005))
COMP21 = "comp21-2idx"


def _days(windows: dict) -> list[int]:
    """The October days of a set's windows (window ids start YYYYMMDD)."""
    days = sorted({str(w)[:8] for w in windows})
    if any(not d.startswith("202610") for d in days):
        raise SystemExit(f"windows outside October 2026: {days}")
    return [int(d[6:8]) for d in days]


def _ties_ok(label: str, block: dict, means: dict, a_key, b_key, instances) -> None:
    """Every instance a D-013 block leaves out (n - better - worse of them) has P = 1 on both sides."""
    left = list(block["left_out_by_compare"])
    if block["n"] - block["better"] - block["worse"] != len(left):
        raise SystemExit(f"{label}: ties {block['n'] - block['better'] - block['worse']} but left out {left}")
    for i in left:
        if i not in instances or means[(a_key, i)] != 1.0 or means[(b_key, i)] != 1.0:
            raise SystemExit(f"{label}: {i} left out but not at P = 1 on both sides")


def d031_data(inp: dict, c1: dict) -> dict:
    runs, d031, dev17 = inp["runs"], inp["d031"], inp["dev17"]
    if (d031.get("a9"), d031.get("run_hash"), d031.get("a6f_hash"), tuple(d031.get("sets", ()))) != \
            ("ysync-ld0", "f345de0ab936", "82a371305f88", (C1R_SET, D031_SET)):
        raise SystemExit("logs/d031-reading.json: not D-031's reading of the frozen-v2 candidate")
    keep, excl = d031["readings"]
    if not keep["keep_failed"] or excl["keep_failed"]:
        raise SystemExit("logs/d031-reading.json: readings[0] must keep the failed runs, readings[1] leave them out")
    v, ve = keep["validity"], excl["validity"]
    if (v["verdict"], v["n_present"], v["n_expected"]) != ("VALID", 136, 136):
        raise SystemExit(f"D-031 kept reading is not VALID 136 of 136: {v}")
    if (ve["verdict"], ve["n_present"], ve["n_expected"]) != ("INCOMPLETE", 133, 136):
        raise SystemExit(f"D-031 exclude reading is not INCOMPLETE 133 of 136: {ve}")
    if not (keep["vs_saved_d030"]["ok"] and keep["same_as_d030"]["ok"]):
        raise SystemExit("D-031's seeds 0-1 differ from D-030's saved reading")
    four, s01, s23 = keep["four"], keep["s01"], keep["s23"]
    if (tuple(four["seeds"]), four["n_runs"], tuple(s23["seeds"])) != (SEEDS4, 136, SEEDS23):
        raise SystemExit("D-031's readings are not on seeds 0-3 / 2-3")
    ph = keep["phase1"]
    if tuple(ph["seeds"]) != SEEDS4 or ph["hint"] != four["hint"] or ph["guard"] != four["guard"]:
        raise SystemExit("D-031's phase-1 result is not the four-seed reading")
    if four["hint"]["state"] != "present" or four["guard"]["state"] != "holds":
        raise SystemExit("D-031: hint absent or guard broken")
    want_failed = [f"{COMP21}__{C1R_A6F}__s{k}__120s__82a371305f88" for k in (0, 1, 2)]
    if sorted(d031["failed_runs"]) != want_failed:
        raise SystemExit(f"D-031 failed runs changed: {d031['failed_runs']}")
    if d031["voids"][D031_SET]["n"] != 0 or d031["n_old"] != 68:
        raise SystemExit("D-031: voids at seeds 2-3 or D-030's run count changed")
    # the run table: D-031's set, its statuses, the crash and seed 3 on comp21
    g = runs[runs["set"] == D031_SET]
    status = {k: int(n) for k, n in g["status"].value_counts().items()}
    if status != {"completed": 67, "failed": 1}:
        raise SystemExit(f"D-031's set statuses changed: {status}")
    cp = g[(g["instance"] == COMP21) & (g["arm"] == C1R_A6F)].set_index("seed")
    if list(cp.loc[[2, 3], "status"]) != ["failed", "completed"]:
        raise SystemExit("comp21-2idx A6F at seeds 2-3: expected failed / completed")
    crash2, comp21_s3 = float(cp.loc[2, "pt"]), float(cp.loc[3, "pt"])
    agree("comp21 A6F s2 (kept, truncated)", crash2, keep["d031"]["runs"][f"{COMP21} {C1R_A6F} s2"]["pt"])
    # per-instance seed means, recomputed from the run table: four seeds over both sets, and seeds 2-3 alone
    m4, _ = seed_means(runs, (C1R_SET, D031_SET), (C1R_A9, C1R_A6F), dev17, seeds=SEEDS4)
    m23, _ = seed_means(runs, (D031_SET,), (C1R_A9, C1R_A6F), dev17, seeds=SEEDS23)
    drift = []
    for lab, m, ru in (("four", m4, four), ("seeds 2-3", m23, s23)):
        for i in dev17:
            want = ru["table"][i]
            for k, got in (("p_a6f", m[(C1R_A6F, i)]), ("p_a9", m[(C1R_A9, i)]),
                           ("d", effect_d(m[(C1R_A6F, i)], m[(C1R_A9, i)]))):
                if i == CA and abs(got - want[k]) > TOL:
                    drift.append((f"{lab} {k} {i}", got, want[k]))     # provisional reference, as in c1_data
                else:
                    agree(f"D-031 {lab} {k} {i}", got, want[k])
        ma9 = float(np.mean([m[(C1R_A9, i)] for i in FIXED11]))
        ma6 = float(np.mean([m[(C1R_A6F, i)] for i in FIXED11]))
        agree(f"D-031 {lab} hint mean A9", ma9, ru["hint"]["mean_P_a9"])
        agree(f"D-031 {lab} hint mean A6F", ma6, ru["hint"]["mean_P_a6f"])
        agree(f"D-031 {lab} hint ratio", ma9 / ma6, ru["hint"]["ratio"])
        agree(f"D-031 {lab} guard", float(np.mean([effect_d(m[(C1R_A6F, i)], m[(C1R_A9, i)]) for i in dev17])),
              ru["guard"]["mean_d"])
    # seeds 0-1 inside D-031's reading = D-030's saved reading (itself checked against the run table in c1_data)
    agree("D-031 s01 hint = D-030", s01["hint"]["ratio"], c1["hint"]["ratio"])
    agree("D-031 s01 guard = D-030", s01["guard"]["mean_d"], c1["guard"]["mean_d"])
    R = D031_RECORDED
    for lab, got, want in (("hint4", four["hint"]["ratio"], R["hint4"]), ("A9 4", four["hint"]["mean_P_a9"], R["a9_4"]),
                           ("A6F 4", four["hint"]["mean_P_a6f"], R["a6f_4"]),
                           ("guard4", four["guard"]["mean_d"], R["guard4"]),
                           ("hint23", s23["hint"]["ratio"], R["hint23"]),
                           ("guard23", s23["guard"]["mean_d"], R["guard23"]),
                           ("hint01", s01["hint"]["ratio"], R["hint01"]),
                           ("guard01", s01["guard"]["mean_d"], R["guard01"])):
        agree_at(f"D-031 Result {lab}", got, want, 3)
    # D-013 lines: counts recomputed from the per-instance d, the rest quoted from the saved report and checked
    # against the Result at its printed precision
    va, va23, vx = four["vs_a6f"], s23["vs_a6f"], excl["four"]["vs_a6f"]
    for lab, blk, m in (("four", va, m4), ("seeds 2-3", va23, m23)):
        ds = [effect_d(m[(C1R_A6F, i)], m[(C1R_A9, i)]) for i in FIXED11]
        if (blk["n"], blk["better"], blk["worse"]) != (11, sum(x > 0 for x in ds), sum(x < 0 for x in ds)):
            raise Mismatch(f"MISMATCH D-031 {lab} vs A6F counts: {blk['better']} / {blk['worse']}")
        _ties_ok(f"D-031 {lab} vs A6F", blk, m, C1R_A6F, C1R_A9, FIXED11)
    for k in ("better", "worse"):
        if va[k] != R["vs_a6f"][k] or va23[k] != R["vs_a6f23"][k]:
            raise Mismatch(f"MISMATCH D-031 vs A6F {k}")
    agree_at("D-031 vs A6F reduction", va["P_reduction"], R["vs_a6f"]["red"], 2)
    agree_at("D-031 seeds 2-3 vs A6F reduction", va23["P_reduction"], R["vs_a6f23"]["red"], 2)
    for k in ("d", "lo", "hi", "p"):
        got = {"d": va["mean_d"], "lo": va["ci"][0], "hi": va["ci"][1], "p": va["p"]}[k]
        agree_at(f"D-031 vs A6F {k}", got, R["vs_a6f"][k], 3)
    if (vx["n"], vx["better"], vx["worse"], tuple(vx["left_out_by_compare"])) != (10, 6, 3, ("cryptanalysiskb128n5obj16",)):
        raise SystemExit(f"D-031 exclude reading vs A6F changed: {vx}")
    a0 = {i: c1["rows"][i]["a0"] for i in dev17}
    w11, w15 = four["vs_a0"]["A9@r13x"]["all"], four["vs_a0_17"]["block"]
    weak15 = four["vs_a0_17"]["weak_set"]
    if sorted(weak15) != sorted(inp["d030"]["readings"][0]["rules"]["vs_a0_17"]["weak_set"]) or len(weak15) != 15:
        raise SystemExit("D-031: SCIP's weak set among the 17 differs from D-030's")
    for lab, blk, inst, rec in (("fixed 11", w11, FIXED11, R["vs_a0_11"]), ("SCIP-weak 15", w15, weak15, R["vs_a0_15"])):
        ds = [effect_d(a0[i], m4[(C1R_A9, i)]) for i in inst]
        agree(f"D-031 vs A0 {lab} before", float(np.mean([a0[i] for i in inst])), blk["mean_P_before"])
        agree(f"D-031 vs A0 {lab} after", float(np.mean([m4[(C1R_A9, i)] for i in inst])), blk["mean_P_after"])
        agree(f"D-031 vs A0 {lab} d", float(np.mean(ds)), blk["mean_d"])
        if (blk["better"], blk["worse"]) != (sum(x > 0 for x in ds), sum(x < 0 for x in ds)) or \
                (blk["better"], blk["worse"]) != (rec["better"], rec["worse"]):
            raise Mismatch(f"MISMATCH D-031 vs A0 {lab} counts")
        agree_at(f"D-031 vs A0 {lab} reduction", blk["P_reduction"], rec["red"], 2)
        left = list(blk["left_out_by_compare"])
        if blk["n"] - blk["better"] - blk["worse"] != len(left) or \
                any(a0[i] != 1.0 or m4[(C1R_A9, i)] != 1.0 for i in left):
            raise SystemExit(f"D-031 vs A0 {lab}: ties are not both at P = 1")
    for k, got in (("before", w15["mean_P_before"]), ("after", w15["mean_P_after"]), ("d", w15["mean_d"]),
                   ("lo", w15["ci"][0]), ("hi", w15["ci"][1])):
        agree_at(f"D-031 vs A0 15 {k}", got, R["vs_a0_15"][k], 3)
    agree_at("D-031 vs A0 15 p", w15["p"], R["vs_a0_15"]["p"], 4)
    # the harm leg: d_i of the new A9 on four seeds against A9@r13x's seeds 0-1 (10-13), bands as D-030's
    hrows = {r["instance"]: r for r in four["harm"]["rows"]}
    bands030 = {r["instance"]: r["band"] for r in inp["d030"]["readings"][0]["rules"]["harm"]["rows"]}
    for i in dev17:
        h = hrows[i]
        agree(f"D-031 harm p_old {i}", h["p_old"], c1["rows"][i]["old_a9"])
        if i == CA and abs(h["p_new"] - m4[(C1R_A9, i)]) > TOL:
            drift.append((f"harm p_new {i}", m4[(C1R_A9, i)], h["p_new"]))
        else:
            agree(f"D-031 harm p_new {i}", h["p_new"], m4[(C1R_A9, i)])
            agree(f"D-031 harm d {i}", effect_d(h["p_old"], h["p_new"]), h["d"])
        agree(f"D-031 harm band {i} = D-030's", h["band"], bands030[i])
    if four["harm"]["harm"] != list(R["harm"]) or sorted(i for i, h in hrows.items() if h["harm"]) != sorted(R["harm"]):
        raise SystemExit(f"D-031 harm list changed: {four['harm']['harm']}")
    for i, (dd, band) in R["harm"].items():
        agree_at(f"D-031 harm d {i}", hrows[i]["d"], dd, 3)
        agree_at(f"D-031 harm band {i}", hrows[i]["band"], band, 3)
    cannot = sorted(four["harm"]["cannot_fire"])
    if cannot != sorted(["rail01", "cryptanalysiskb128n5obj16", "neos-2657525-crna", "neos-2746589-doon"]):
        raise SystemExit(f"D-031: instances where the harm leg cannot fire changed: {cannot}")
    # the halves: per-instance d at seeds 0-1 and 2-3, sign agreement
    hv = keep["halves"]
    if hv["count"] != {"same": 15, "opposite": 0, "a zero": 2} or not (hv["hint_agree"] and hv["guard_agree"]):
        raise SystemExit(f"D-031 halves changed: {hv['count']}")
    d01 = {r["instance"]: r["d01"] for r in hv["rows"]}
    d23 = {r["instance"]: r["d23"] for r in hv["rows"]}
    for i in dev17:
        agree(f"D-031 halves d01 {i}", d01[i], c1["rows"][i]["new_d"])
        if i != CA:
            agree(f"D-031 halves d23 {i}", d23[i], effect_d(m23[(C1R_A6F, i)], m23[(C1R_A9, i)]))
    # results.tex: "Over all 17 instances, P was lower on 8, higher on 8 and equal on 1" (four seeds)
    d17 = [effect_d(m4[(C1R_A6F, i)], m4[(C1R_A9, i)]) for i in dev17]
    if (sum(x > 0 for x in d17), sum(x < 0 for x in d17), sum(x == 0 for x in d17)) != (8, 8, 1):
        raise SystemExit("four seeds over the 17: not 8 lower / 8 higher / 1 equal (results.tex says so)")
    floor4 = {i for i in dev17 if max(round(m4[(C1R_A6F, i)], 3), round(m4[(C1R_A9, i)], 3)) <= 0.011}
    if floor4 != c1["floor"]:
        raise SystemExit(f"floor set at four seeds differs: {sorted(floor4)}")
    return dict(m4=m4, m23=m23, four=four, s01=s01, s23=s23, excl=excl, harm=hrows, d01=d01, d23=d23,
                halves=hv["count"], cannot=cannot, crash=(c1["crash"]["0"], c1["crash"]["1"], crash2),
                comp21_s3=comp21_s3, status=status, floor=floor4, drift=drift,
                days01=_days(d031["windows"][C1R_SET]), days23=_days(d031["windows"][D031_SET]))


def instance_sets() -> tuple[dict[str, str], dict[str, str]]:
    """{instance: 'dev' | 'probe'} and {instance: source library} for the 17, from datasets/dev_set.csv and
    datasets/classprobe_set.csv (the source is the library directory of the instance path)."""
    lib = {"miplib": "MIPLIB", "pglib_uc": "PGLib-UC", "ml4co": "ML4CO"}
    sets, src = {}, {}
    for f, label in (("dev_set.csv", "dev"), ("classprobe_set.csv", "probe")):
        t = pd.read_csv(DATASETS / f)
        for path, name in zip(t["instance"], t["name"]):
            sets[name] = label
            src[name] = lib[Path(path).parts[1]]
    return sets, src


def table_d031(c: dict, c1: dict, dev17: list[str]) -> str:
    """Appendix B: the 17 instances and the phase-1 interim result per instance (D-031, four seeds), with SCIP alone
    (seeds 0-1) beside. The caveats are bullets in paper/sections/app-c1.tex."""
    four, hr = c["four"], c["harm"]
    sets, src = instance_sets()
    if sorted(sets) != sorted(dev17) or sum(v == "dev" for v in sets.values()) != 9:
        raise SystemExit("datasets/dev_set.csv + classprobe_set.csv are not the 9 + 8 development instances")
    other_src = sorted((i, s) for i, s in src.items() if s != "MIPLIB")
    if other_src != [(CA, "PGLib-UC"), ("item_placement_9903", "ML4CO")]:
        raise SystemExit(f"non-MIPLIB instances changed: {other_src}")
    d4 = {i: four["table"][i]["d"] for i in hr}
    fixed = sorted(FIXED11, key=lambda i: (-d4[i], i))
    other = sorted((i for i in hr if i not in FIXED11), key=lambda i: (-d4[i], i))

    def line(i: str) -> str:
        r = four["table"][i]
        name = tt(i) + ("$^{\\dagger}$" if i in c["floor"] else "")
        a6 = f3(r["p_a6f"]) + ("$^{a}$" if i == COMP21 else "")
        h = hr[i]
        harm = ("--" if h["cannot_fire"] else f"{sd(h['d'])} / {f3(h['band'])}") + ("$^{*}$" if h["harm"] else "")
        # source: logs/d031-reading.json readings[0].four.table[i], .halves.rows[i], .four.harm.rows[i]; the four-seed
        # and seeds 2-3 means recomputed from datasets/runs.parquet in d031_data; A0 = c1_data (runs.parquet, arm A0,
        # sets dev / classprobe, seeds 0-1)
        return (f"{name} & {sets[i]} & {f3(c1['rows'][i]['a0'])} & {a6} & {f3(r['p_a9'])} & {sd(r['d'])} & "
                f"{sd(c['d01'][i])} & {sd(c['d23'][i])} & {harm} \\\\\n")

    s23, s01, h4 = c["s23"], c["s01"], four["hint"]
    s = HEADER
    s += ("% Inputs: logs/d031-reading.json readings[0] (four, s01, s23, halves, four.harm; crashed runs kept);\n"
          "% per-instance means recomputed from datasets/runs.parquet (sets c1r-ysync-ld0 and c1r-ysync-ld0-s23) with\n"
          "% tools/d022_report.py's select(); A0 from sets dev / classprobe; set and source from datasets/dev_set.csv\n"
          "% and datasets/classprobe_set.csv; printed values checked against D-031's Result.\n")
    s += "\\begin{table}[!tb]\\centering\\scriptsize\n\\setlength{\\tabcolsep}{3.5pt}\n"
    s += "\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
    s += (" & & \\multicolumn{3}{c}{$P(120)$, seed means} & & \\multicolumn{2}{c}{$d$ by half} & harm check \\\\\n"
          "\\cmidrule(lr){3-5}\\cmidrule(lr){7-8}\\cmidrule(lr){9-9}\n"
          "instance & set & A0 & A6F & A9 & $d$ & seeds 0--1 & seeds 2--3 & $d_i$ / band \\\\\n\\midrule\n")
    s += "\\multicolumn{9}{l}{\\emph{cuOpt's hard set, the fixed 11 (the hint reads these)}} \\\\\n"
    s += "".join(line(i) for i in fixed)
    s += "\\midrule\n\\multicolumn{9}{l}{\\emph{the other 6 (the guard reads all 17)}} \\\\\n"
    s += "".join(line(i) for i in other)
    s += "\\midrule\n"
    s += (f"\\multicolumn{{3}}{{l}}{{hint: mean $P$, fixed 11; ratio}} & {f3(h4['mean_P_a6f'])} & "
          f"{f3(h4['mean_P_a9'])} & {f3(h4['ratio'])} & {f3(s01['hint']['ratio'])} & {f3(s23['hint']['ratio'])} & "
          "\\\\\n")
    s += (f"\\multicolumn{{5}}{{l}}{{guard: mean $d$ over the 17}} & {sd(four['guard']['mean_d'])} & "
          f"{sd(s01['guard']['mean_d'])} & {sd(s23['guard']['mean_d'])} & \\\\\n")
    s += "\\bottomrule\n\\end{tabular}\n"
    days = sorted(set(c["days01"]) | set(c["days23"]))
    s += ("\\caption{The 17 instances and the phase-1 result per instance: seed means of $P(120)$. A0: SCIP alone, "
          "seeds 0--1 (6--7 October). A6F: cuOpt alone; A9: the coupled system; same cores, GPU and window, seeds 0--3 "
          f"({days[0]}--{days[-1]} October). $d$: A9 against A6F, positive if A9 is lower. Harm check: against A9 of 13 "
          "October; $^{*}$ flagged, -- cannot fire. Set: development set or class probe; all MIPLIB 2017 except "
          f"{tt(CA)} (PGLib-UC) and {tt('item_placement_9903')} (ML4CO). $^{{a}}$Crashed on seeds 0--2 ($P$ "
          f"{f3(c['crash'][0])}, {f3(c['crash'][1])}, {f3(c['crash'][2])}, kept); {f3(c['comp21_s3'])} on seed 3. "
          "$^{\\dagger}$Both arms at $P \\le 0.011$.}\\label{tab:d031}\n")
    s += "\\end{table}\n"
    return s


# ----------------------------------------------------------------------------------------------- table_readings
def table_readings(c1: dict, d13: dict, d31: dict) -> str:
    """The main text's Table tab:readings: the tabular only (the float and its caption are in sections/results.tex,
    which uses no internal codes). Rows (2026-10-16 review: the two halves are given in the text, reading 1 only in
    Appendix A): the final configuration on four seeds (D-031, the phase-1 interim result) against cuOpt alone, the
    same with cuOpt alone's crashed runs left out, and against SCIP alone (A0, seeds 0-1) on SCIP's hard set.
    c1 and d13 are still computed and checked (their numbers feed the text and Appendix A) but print no row here."""
    four, excl = d31["four"], d31["excl"]["four"]["vs_a6f"]
    ds = sorted(set(d31["days01"]) | set(d31["days23"]))
    days = f"{ds[0]}--{ds[-1]} Oct." if len(ds) > 1 else f"{ds[0]} Oct."

    def row(label: str, b: dict, hint: str, guard: str, of: str = "") -> str:
        bt, wo, before, after, d, lo, hi = (b["better"], b["worse"], b["mean_P_before"], b["mean_P_after"],
                                            b["mean_d"], b["ci"][0], b["ci"][1])
        return (f"{label} & {bt} / {wo}{of} & {f3(before)} $\\to$ {f3(after)} & {sd(d)} {ci(lo, hi)} & {hint} & "
                f"{guard} \\\\\n")

    if excl["n"] != 10:
        raise SystemExit("the crashes-left-out row expects 10 instances")
    s = HEADER
    s += ("% Inputs: logs/d031-reading.json readings[0].four (vs_a6f, hint, guard, vs_a0_17.block; crashed runs kept)\n"
          "% and readings[1].four.vs_a6f (crashes left out); every value checked against D-031's Result in d031_data.\n"
          "% The float and caption are in paper/sections/results.tex (tab:readings).\n")
    s += "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lccccc}\n\\toprule\n"
    s += "runs & lower / higher & mean $P$ & mean $d$ [95\\,\\% CI] & hint & guard \\\\\n\\midrule\n"
    s += "\\multicolumn{6}{l}{\\emph{against cuOpt alone, on cuOpt's hard set (11 instances)}} \\\\\n"
    s += row(f"\\textbf{{seeds 0--3 ({days})}}", four["vs_a6f"], f"\\textbf{{{f3(four['hint']['ratio'])}}}",
             "$\\mathbf{" + f"{four['guard']['mean_d']:+.3f}" + "}$")
    s += row("seeds 0--3, crashes left out", excl, "--", "--", of=" of 10")
    s += "\\multicolumn{6}{l}{\\emph{against SCIP alone, on SCIP's hard set (15 instances)}} \\\\\n"
    s += row("seeds 0--3 (SCIP alone: 0--1)", four["vs_a0_17"]["block"], "", "")
    s += "\\midrule\nrule & & & & $\\le 0.95$ & $\\ge -0.05$ \\\\\n\\bottomrule\n\\end{tabular}\n"
    return s


# ----------------------------------------------------------------------------------------------- table_devrecord
# Development tests, diagnostics and C1 readings (kinds T, D, R); the framework records D-010, D-013, D-019, D-020,
# D-021 and D-024 are not tests and stay in the text. Each row: (record, kind, question, runs, verdict, key numbers).
# Run counts of dedicated sets are checked against runs.parquet (DEV_COUNTS).
DEVRECORD = [
    # Compact wording (2026-10-16, appendix rewrite): the same numbers as the records, fewer words. runs: instances x
    # seeds (runs over all arms; read = runs the reading used).
    # source: Development log "LP stream" (2026-10-06 16:03-16:35); D-011 § 5 (a measurement plan, no adoption rule
    # fixed before the runs: kind M)
    ("D-011", "M", "SCIP streams good infeasible node-LP points to the EA (A3)",
     "9 $\\times$ 2 (36)", "kept on, not credited", "$d$ $+0.05$; 3 / 1"),
    # source: D-012 addendum 2 (2026-10-07 18:1x); Development log "Repair clock, Stage 2 -- complete"
    ("D-012", "T", "Repair depth set per generation: the repair clock (A3)",
     "9 $\\times$ 2--4 (44)", "default; 2 rules failed by the letter",
     "\\texttt{item\\_placement\\_9903} $+0.50$ (4 of 4 seeds); all 5 / 2, $+0.12$ [$+0.00$, $+0.26$]"),
    # source: D-014 Result (2026-10-07 15:13); Development log "Class probe -- D-014 result"
    ("D-014", "T", "Does A3 hold on 8 new instances (predictions fixed first)?",
     "8 $\\times$ 2 (48)", "INCONCLUSIVE (3 of 5)", "vs cuOpt alone 1 / 5, $-0.51$ [$-0.84$, $-0.12$]"),
    # source: D-015 "Pilot result" (2026-10-07 20:41); Development log "Three-way arm A9 -- pilot"
    ("D-015", "pilot", "Pass cuOpt's solutions to the EA: arm A9",
     "7 $\\times$ 1 (21)", "built",
     "vs SCIP alone 7 / 0, $+0.68$ [$+0.42$, $+0.90$]; vs cuOpt alone 3 / 4, $-0.07$ [$-0.42$, $+0.25$]"),
    # source: D-016 Final (2026-10-09 14:5x); Development log "D-016 final"
    ("D-016", "T", "Rank 1 by cuOpt's clearing rule (A3)",
     "9 $\\times$ 2, 8 $\\times$ 2 (102)", "NOT ADOPTED", "$d$ $-0.033$ [$-0.122$, $+0.050$]; 4 / 4"),
    # source: D-017 Result (2026-10-09 14:49-16:07); Development log "LP decode -- D-017 result"
    ("D-017", "T", "LP decode of candidates' continuous variables (A3)",
     "6 $\\times$ 2 (24)", "ADOPTED (A3); 1 rule failed by the letter",
     "$d$ $+0.219$ [$+0.041$, $+0.429$]; 5 / 1; \\texttt{item\\_placement\\_9903} 0.311 $\\to$ 0.118"),
    # source: D-018 "Stage 1 result" (2026-10-11); Development log "D-018 stage 1"
    ("D-018", "T", "The EA backs off the GPU while it does not contribute (A9)",
     "14 $\\times$ 2 (84)", "NOT ADOPTED: 3 mechanism rules failed",
     "outcome rules held; \\texttt{chromaticindex1024-7} $d$ $+0.834$; all 9 / 5, $+0.083$ [$-0.054$, $+0.232$]"),
    # source: D-022 Result (2026-10-13 22:2x)
    ("D-022", "T, R", "A9 under the final harness; C1's first reading",
     "17 $\\times$ 2 (74)", "ADOPT (mechanism vacuous)",
     "no harm where cuOpt can stop itself (3); old default: hint 0.827, guard $+0.063$"),
    # source: D-023 Result (2026-10-14 18:1x)
    ("D-023", "T", "Phase Y only: the EA yields until cuOpt's first incumbent",
     "11 $\\times$ 2 (66)", "NOT ADOPTED: 4 rules failed",
     "\\texttt{chromaticindex1024-7} $d$ $+0.030$ (needed $\\ge +0.30$); cuOpt first at 63.47 / 63.41\\,s"),
    # source: D-025 "Result of the bsync test" (2026-10-15 08:1x)
    ("D-025", "T", "R1: the EA's CUDA context blocks instead of spinning",
     "5 $\\times$ 2 (20)", "NOT ADOPTED",
     "hint 0.765; generations 0.824, 0.818 (bars 0.921, 0.950); \\texttt{comp21-2idx} $\\Delta P$ $+0.060$ (band "
     "0.050)"),
    # source: D-025 "Result of fallback (b) ysync" (2026-10-15 11:2x)
    ("D-025", "T", "R1 fallback: the CUDA context yields instead of spinning",
     "5 $\\times$ 2 (20)", "NOT ADOPTED",
     "hint 0.834; \\texttt{item\\_placement\\_9903} generations 0.759 (bar 0.921): 678 vs 1,480 on seed 1"),
    # source: D-026 Result (2026-10-15 13:2x)
    ("D-026", "T", "R2: LP decode inside A9, after the EA's LP worker",
     "2 $\\times$ 4, 3 $\\times$ 2 (28)", "ADOPT",
     "hint 0.758 (bar 0.90); \\texttt{item\\_placement\\_9903} 0.330 $\\to$ 0.198, \\texttt{sing44} 0.473 $\\to$ "
     "0.411, all 4 seeds"),
    # source: D-027 Result (2026-10-15 09:2x) and its amendment 18:3x
    ("D-027", "D", "Why A9 loses on 4 instances: cuOpt's threads or SCIP?",
     "4 $\\times$ 2 (22)", "the EA's presence",
     "cuOpt's first incumbent, \\texttt{chromaticindex1024-7}: 6.5--9.8\\,s without the EA, 62--66\\,s in A9"),
    # source: D-028 Result (2026-10-15 11:4x)
    ("D-028", "D", "Do R13 and R14 keep phase Y from helping cuOpt?",
     "1 $\\times$ 2 (4)", "not R13 and R14", "63.25 / 63.24\\,s against a limit of 19.82\\,s"),
    # source: D-029 Result (2026-10-15 15:0x); its amendment B1-B2 and § Forking paths
    ("D-029", "T", "The yield flag again, on seeds 0--5 with D-025's failing seed",
     "3 $\\times$ 2--4 (20; 28 read)", "ADOPT; a bar loosened after the failure",
     "generations 1.020, 1.152 (bar 0.921 $\\to$ 0.90); $\\Delta P$ $+0.016$, $+0.015$ (bar $+0.05$); "
     "\\texttt{ca\\_\\_Scenario400\\_reserves\\_1} $P$ ratio, seeds 2--3: 0.675 (bar 0.95)"),
    # source: D-030 Result (2026-10-16 08:2x) and its correction 10:0x
    ("D-030", "R", "C1 on the frozen-v2 candidate, cuOpt alone in the same window",
     "17 $\\times$ 2 (68)", "VALID; harm flag: \\texttt{glass4}",
     "hint 0.809, guard $+0.050$; cuOpt alone crashed on \\texttt{comp21-2idx} (kept)"),
    # source: D-031 Result (2026-10-16 13:1x); the numbers checked against logs/d031-reading.json in devrecord_check
    ("D-031", "R", "The phase-1 result: C1 at seeds 2--3, read with D-030's 0--1",
     "17 $\\times$ 2 (68; 136 read)", "VALID; harm flags: \\texttt{glass4}, \\texttt{comp21-2idx}",
     "hint 0.798, guard $+0.045$ (seeds 2--3: 0.787, $+0.034$); vs cuOpt alone 7 / 3, $+0.249$ "
     "[$+0.078$, $+0.441$]"),
    # source: D-032 Result (2026-10-16 08:5x)
    ("D-032", "T", "\\texttt{glass4}'s harm flag: real, or day-to-day drift?",
     "1 $\\times$ 4 (12)", "DRIFT: candidate freezes",
     "mean $P$ 0.2176 / 0.2233 / 0.2177 (3 configurations); $d$ $-0.025$ (bar $-0.057$)"),
]
# record -> (set(s), arms or None, expected run count)
DEV_COUNTS = [
    ("D-011", ("dev",), ("A3@lps1", "A3@ad1"), 36), ("D-014", ("classprobe",), None, 48),
    ("D-015", ("a9pilot",), None, 21), ("D-016", ("dev",), ("A3@rc1d", "A3@cl1", "A3@tp1"), 102),
    ("D-017", ("dev",), ("A3@rc1e", "A3@ld1"), 24), ("D-018", ("backoff1",), None, 84),
    ("D-022", ("itx-r13",), None, 74), ("D-023", ("backoff2",), None, 66), ("D-025 bsync", ("r1-bsync",), None, 20),
    ("D-025 ysync", ("r1-ysync",), None, 20), ("D-026", ("r2-ld",), None, 28), ("D-027", ("r3-diag",), None, 22),
    ("D-028", ("d028",), None, 4), ("D-029", ("r1f-ysync",), None, 20), ("D-029 warm-up", ("r1f-ysync-warmup",), None, 1),
    ("D-030", ("c1r-ysync-ld0",), None, 68), ("D-031", (D031_SET,), None, 68), ("D-032", ("d032",), None, 12),
]
# The main text numbers the tests 1-18 in DEVRECORD's row order (paper/sections/results.tex, app-devrecord.tex); the
# table prints that number, and a reordered or added row stops the script.
DEV_NUMBERS = ("D-011", "D-012", "D-014", "D-015", "D-016", "D-017", "D-018", "D-022", "D-023", "D-025 blocking",
               "D-025 yield", "D-026", "D-027", "D-028", "D-029", "D-030", "D-031", "D-032")


def devrecord_check(inp: dict) -> None:
    runs = inp["runs"]
    for rec, sets, arms, n in DEV_COUNTS:
        g = runs[runs["set"].isin(sets)]
        if arms is not None:
            g = g[g["arm"].isin(arms)]
        if len(g) != n:
            raise SystemExit(f"{rec}: {len(g)} runs in the table, the record says {n}")
    m = runs[runs["set"] == "d032"].groupby("arm")["pt"].mean()
    # source: D-032 Result: old 0.2176, candidate 0.2233, lp_decode 0.2177; d -0.025
    agree_at("D-032 old", m["A9@g4old"], 0.2176, 4)
    agree_at("D-032 candidate", m["A9@g4yl"], 0.2233, 4)
    agree_at("D-032 ld", m["A9@g4ld"], 0.2177, 4)
    agree_at("D-032 d", effect_d(m["A9@g4old"], m["A9@g4yl"]), -0.025, 3)
    # D-030 / D-022 hint and guard as printed in their rows (also in table_readings)
    r = inp["d030"]["readings"][0]["rules"]
    agree_at("D-030 hint (devrecord)", r["hint"]["ratio"], 0.809, 3)
    agree_at("D-022 hint (devrecord)", r["then"]["hint"]["ratio"], 0.827, 3)
    # D-031's row (also in table_d031 / table_readings)
    k = inp["d031"]["readings"][0]
    agree_at("D-031 hint (devrecord)", k["four"]["hint"]["ratio"], 0.798, 3)
    agree_at("D-031 guard (devrecord)", k["four"]["guard"]["mean_d"], 0.045, 3)
    agree_at("D-031 s23 hint (devrecord)", k["s23"]["hint"]["ratio"], 0.787, 3)
    agree_at("D-031 s23 guard (devrecord)", k["s23"]["guard"]["mean_d"], 0.034, 3)
    va = k["four"]["vs_a6f"]
    if (va["better"], va["worse"]) != (7, 3):
        raise Mismatch("MISMATCH D-031 vs A6F counts (devrecord)")
    for lab, got, want in (("d", va["mean_d"], 0.249), ("lo", va["ci"][0], 0.078), ("hi", va["ci"][1], 0.441)):
        agree_at(f"D-031 vs A6F {lab} (devrecord)", got, want, 3)
    if k["four"]["harm"]["harm"] != ["glass4", "comp21-2idx"]:
        raise SystemExit("D-031's harm list changed (devrecord)")
    names = []
    for rec, _kind, q, *_ in DEVRECORD:
        names.append(rec if rec != "D-025" else ("D-025 blocking" if "blocks" in q else "D-025 yield"))
    if tuple(names) != DEV_NUMBERS:
        raise SystemExit(f"DEVRECORD's row order is not the main text's 1-18 numbering: {names}")


def table_devrecord() -> str:
    s = HEADER
    s += ("% Inputs: the decision records vault/50-decisions/D-011 ... D-032 (Result / Status lines) and\n"
          "% vault/30-results/Development log.md, encoded as data in tools/paper_tables.py (DEVRECORD, one source\n"
          "% comment per row); dedicated sets' run counts checked against datasets/runs.parquet.\n")
    s += "\\begin{table}[!tb]\\centering\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\\setlength{\\abovecaptionskip}{4pt}\n"
    s += ("\\begin{tabular}{r>{\\raggedright}p{0.95cm}>{\\raggedright}p{4.5cm}>{\\raggedright}p{1.7cm}"
          ">{\\raggedright\\arraybackslash}p{8.25cm}}\n\\toprule\n")
    s += ("no. & record & question & inst. $\\times$ seeds (runs) & \\textbf{verdict}: key numbers \\\\\n"
          "\\midrule\n")
    for no, (rec, kind, q, runs_, verdict, nums) in enumerate(DEVRECORD, start=1):
        s += f"{no} & {rec} ({kind}) & {q} & {runs_} & \\textbf{{{verdict}}}: {nums} \\\\\n"
    s += "\\bottomrule\n\\end{tabular}\n"
    s += ("\\caption{The development record of phase 1. no.: the test number of the main text. Kinds: T, a test with "
          "rules fixed before any of its runs was read (exceptions in the text); M, measured without such a rule; D, a "
          "diagnostic; R, a C1 reading. $a$ / $b$: instances better / worse; $d$: mean bounded difference (positive: "
          "the change or A9 better) [95\\% bootstrap interval]; hint: ratio of mean $P(120)$; generations: treatment "
          "over control. 120-s runs, 1--6 seeds: development data, descriptive.}\\label{tab:devrecord}\n")
    s += "\\end{table}\n"
    return s


# ----------------------------------------------------------------------------------------------- figures
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (validated all-pairs, light mode)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RC = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
      "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
      "text.color": INK, "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.hashsalt": "itrea",
      "font.family": "DejaVu Sans"}
PDF_META = {"CreationDate": None, "Producer": None, "Creator": None}


def _pdf_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", metadata=PDF_META, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return buf.getvalue()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_curve(row: pd.Series, inc: pd.DataFrame) -> dict:
    """The run's portfolio trajectory from its run directory's incumbent files and their checked rows: per role, the
    npz's sha256 and row count must match `incumbents_checked`; steps = best-so-far of the passing incumbents'
    recomputed objectives (build_runs.role_steps), merged over roles; P(120) recomputed must equal the table's `pt`."""
    rid, run_dir, s, ref = row["run_id"], ROOT / row["run_dir"], int(row["sense"]), float(row["ref"])
    g = inc[inc["run_id"] == rid]
    roles = sorted(p.name.split(".")[0] for p in run_dir.glob("*.incumbents.npz"))
    if not roles:
        raise SystemExit(f"{rid}: no incumbent file in {run_dir}")
    steps = {}
    for role in roles:
        npz = run_dir / f"{role}.incumbents.npz"
        with np.load(npz, allow_pickle=False) as z:
            n = int(z["t"].shape[0])
        gr = g[g["role"] == role]
        if len(gr) != n:
            raise SystemExit(f"{rid} {role}: {n} incumbents in the npz, {len(gr)} checked rows")
        if n and set(gr["npz_sha256"]) != {_sha(npz)}:
            raise SystemExit(f"{rid} {role}: the npz's sha256 differs from the checked rows'")
        ok = gr[gr["ok"]]
        steps[role] = M.best_steps(zip(ok["t"].to_numpy(), ok["obj_recomputed"].to_numpy()), s)
    if set(g["role"]) - set(roles):
        raise SystemExit(f"{rid}: checked rows for roles without an npz")
    port = M.merge_steps(*steps.values(), sense=s)
    P = M.primal_integral(port, ref, s, T).value
    agree(f"P(120) {rid}", P, float(row["pt"]))
    xs, ys = [0.0], [1.0]
    for t, v in port:
        if t >= T:
            break
        xs.append(float(t))
        ys.append(M.primal_gap(v, ref, s)[0])
    xs.append(T)
    ys.append(ys[-1])
    return dict(x=xs, y=ys, P=P, n=sum(len(v) for v in steps.values()))


def pi_choice(inp: dict) -> tuple[str, list]:
    runs, d030 = inp["runs"], inp["d030"]
    target = d030["readings"][0]["rules"]["hint"]["mean_P_a9"]
    df, _ = D22.select(runs, (C1R_SET,), (C1R_A9, C1R_A6F), SEEDS, T, keep_failed=True, instances=FIXED11)
    failed = set(df.loc[df["status"] == "failed", "instance"])
    m = df[df["arm"] == C1R_A9].groupby("instance")["pt"].mean()
    rank = sorted(((abs(float(m[i]) - target), i, float(m[i])) for i in FIXED11 if i not in failed))
    return rank[0][1], [dict(instance=i, a9=p, dist=d) for d, i, p in rank] + \
        [dict(instance=i, excluded="a failed run in D-030") for i in sorted(failed)]


def fig_pi(inp: dict) -> tuple[bytes, dict]:
    runs = inp["runs"]
    inst, ranking = pi_choice(inp)
    inc = pd.read_parquet(DATASETS / "incumbents_checked.parquet")
    a9, _ = D22.select(runs, (C1R_SET,), (C1R_A9, C1R_A6F), SEEDS, T, keep_failed=True, instances=[inst])
    a0, _ = D22.select(runs, A0_SETS, (A0,), SEEDS, T, instances=[inst])
    sel = pd.concat([a9, a0]).set_index(["arm", "seed"])
    # Plain-word legend: the figure sits in the main text, which uses no arm codes (A9 / A6F / A0).
    arms = ((C1R_A9, "coupled system", BLUE, "-"), (C1R_A6F, "cuOpt alone, same cores and GPU", ORANGE, "--"),
            (A0, "SCIP alone, one core", AQUA, ":"))
    info = dict(instance=inst, ranking=ranking, ref=None, ref_kind=None, curves={})
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.35), sharey=True)  # short: the figure sits in the main text
        for ax, seed in zip(axes, SEEDS):
            for arm, label, col, ls in arms:
                row = sel.loc[(arm, seed)]
                cur = run_curve(row, inc)
                info["ref"], info["ref_kind"] = float(row["ref"]), str(row["ref_kind"])
                info["curves"][f"{arm} s{seed}"] = dict(P=cur["P"], n_inc=cur["n"], run_id=row["run_id"],
                                                        window=str(row["window_id"]))
                # P values go in the caption (written from this function's info / the summary line).
                ax.step(cur["x"], cur["y"], where="post", color=col, ls=ls, lw=1.6, label=label)
            ax.set_xlim(0, T)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xlabel("time since the solver call (s)")
            ax.set_title(f"{inst}, seed {seed}", color=INK)
            ax.grid(True, color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0].set_ylabel("primal gap")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False,
                   handlelength=2.6)
        fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.93))
        return _pdf_bytes(fig), info


# ----------------------------------------------------------------------------------------------- main
def build(inp: dict) -> tuple[dict[Path, bytes], dict]:
    out: dict[Path, bytes] = {}
    meta: dict = {}
    devrecord_check(inp)
    out[GEN / "table_devrecord.tex"] = ascii_only("table_devrecord", table_devrecord()).encode()
    c1 = c1_data(inp)
    d13 = d013_data(inp)
    d31 = d031_data(inp, c1)
    d013_add_d031(d13, d31)
    out[GEN / "table_d031.tex"] = ascii_only("table_d031", table_d031(d31, c1, inp["dev17"])).encode()
    out[GEN / "table_readings.tex"] = ascii_only("table_readings", table_readings(c1, d13, d31)).encode()
    cal = calib_data(inp)
    out[GEN / "list_calib.tex"] = ascii_only("list_calib", list_calib(cal)).encode()
    chrom = chrom_data(inp)
    out[GEN / "table_chromatic.tex"] = ascii_only("table_chromatic", table_chromatic(chrom)).encode()
    pi_pdf, pi_info = fig_pi(inp)
    out[FIG / "fig_pi.pdf"] = pi_pdf
    for p, b in out.items():
        if p.suffix == ".tex" and b"\\pending" in b:
            raise SystemExit(f"{p.relative_to(ROOT)} would hold a \\pending marker")
    meta.update(c1=c1, d013=d13, d031=d31, calib=cal, chrom=chrom, pi=pi_info)
    return out, meta


def summary(meta: dict) -> str:
    c1, d13, cal, pi, d31 = meta["c1"], meta["d013"], meta["calib"], meta["pi"], meta["d031"]
    L = [
        "paper_tables: every computed number agreed with its source (the script stops on the first mismatch).",
        "table_devrecord: 18 rows numbered 1-18 as the main text (D-011 ... D-032, T / M / D / R / pilot), encoded "
        "from the records' Results (one source comment per row); 18 run counts checked against the run table; D-032's "
        "means 0.2176 / 0.2233 / 0.2177 and d -0.025 recomputed from set d032; D-031's row checked against its JSON.",
        f"c1_data (no table): 17 instances; 10-13 hint {c1['hint_old']['ratio']:.3f} guard {c1['guard_old']['mean_d']:+.3f} "
        f"(D-022 Result 0.827 / +0.063); D-030 hint {c1['hint']['ratio']:.3f} guard {c1['guard']['mean_d']:+.3f} "
        "(D-030 Result 0.809 / +0.050); every seed mean equals logs/d030-reading.json and the run table (1e-9); "
        f"floor instances {sorted(c1['floor'])}; comp21 A6F crashes kept at P {c1['crash']['0']:.3f} / "
        f"{c1['crash']['1']:.3f}; ca drift against the saved reading: "
        + (", ".join(f"{lab} {g:.4f} vs {w:.4f}" for lab, g, w in c1["drift"]) or "none (0 of 6 values)") + ".",
        "d013_data (no table): 10-13 fixed-11 lines quoted from D-022's Result; recomputed with d022_report.c1_reading on set "
        "itx-r13 -> "
        + "; ".join(f"vs {k}: {g['better']}/{g['worse']}, {g['mean_P_before']:.3f} -> {g['mean_P_after']:.3f}, "
                    f"d {g['mean_d']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}], p {g['p']:.4f}"
                    for k, g in d13["recomputed"])
        + "; equal at the printed precision except "
        + (", ".join(f"vs {k} {name} {g:+.4f} vs recorded {w:+.{nd}f} (ca re-scoring: provisional reference)"
                     for k, name, _, g, w, nd in d13["drift"]) or "none")
        + "; D-030 lines from the saved JSON.",
        f"list_calib: M4 recomputed: R {cal['m4']['call']['R']:.3f} [{cal['m4']['call']['lo']:.3f}, "
        f"{cal['m4']['call']['hi']:.3f}], launch {cal['m4']['launch']['R']:.3f}, feasible "
        f"{cal['m4']['feasible']['ours']}/{cal['m4']['feasible']['gams_sample']}, late clock "
        f"{cal['m4']['late_clock']['pooled']['median']:+.3f} s (n {cal['m4']['late_clock']['pooled']['n']}), "
        f"{cal['m4']['verdict']}; M6 {100 * cal['m6']['mean_slowdown']:.1f} %, {cal['m6']['hottest_c']} C, "
        f"{cal['m6']['lanes']} lanes; M1-M3 quoted from the D-021 measurements note (run counts and M3's list checked).",
        f"table_chromatic: {len(meta['chrom'])} (set, arm) rows = every cuOpt run on chromaticindex1024-7; D-027's "
        "62-66 s and its three exceptions (bo1, bof25, bof50) and D-028's L = 19.82 s re-derived.",
        f"table_d031 / table_readings: D-031 VALID 136/136; four seeds hint {d31['four']['hint']['ratio']:.3f} "
        f"({d31['four']['hint']['mean_P_a9']:.3f} / {d31['four']['hint']['mean_P_a6f']:.3f}), guard "
        f"{d31['four']['guard']['mean_d']:+.3f}; seeds 2-3 {d31['s23']['hint']['ratio']:.3f} / "
        f"{d31['s23']['guard']['mean_d']:+.3f}; harm {d31['four']['harm']['harm']}; halves {d31['halves']}; every "
        "four-seed and seeds 2-3 seed mean recomputed from the run table (1e-9) and every printed value equal to "
        "D-031's Result; ca drift against the saved reading: "
        + (", ".join(f"{lab} {g:.4f} vs {w:.4f}" for lab, g, w in d31["drift"]) or "none") + ".",
        f"fig_pi: instance {pi['instance']} (ref {pi['ref']} [{pi['ref_kind']}]); rule = fixed-11 instance with no "
        "failed D-030 run whose A9@c1r seed-mean P is closest to the fixed-11 mean A9 P; ranking: "
        + ", ".join(f"{r['instance']} {r['a9']:.3f} (distance {r['dist']:.3f})" if "a9" in r else
                    f"{r['instance']} excluded ({r['excluded']})" for r in pi["ranking"]) + ".",
        "fig_pi curves (P recomputed from the run directories' checked incumbents = run table pt, 1e-9): "
        + "; ".join(f"{k} P {v['P']:.3f} ({v['n_inc']} incumbent records, window {v['window']})"
                    for k, v in pi["curves"].items()) + ".",
    ]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the files on disk; write nothing")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    inp = load()
    try:
        out, meta = build(inp)
    except Mismatch as e:
        print(e, file=sys.stderr)
        return 2
    rc = 0
    if a.check:
        for p, b in out.items():
            if not p.exists():
                print(f"MISSING {p.relative_to(ROOT)}")
                rc = 1
            elif p.read_bytes() != b:
                print(f"STALE   {p.relative_to(ROOT)}")
                rc = 1
            else:
                print(f"ok      {p.relative_to(ROOT)}")
    else:
        GEN.mkdir(parents=True, exist_ok=True)
        FIG.mkdir(parents=True, exist_ok=True)
        for p, b in out.items():
            p.write_bytes(b)
            print(f"wrote   {p.relative_to(ROOT)} ({len(b)} bytes)")
    if not a.quiet:
        print(summary(meta))
    return rc


if __name__ == "__main__":
    sys.exit(main())
