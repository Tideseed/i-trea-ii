"""`itrea itx-report` (D-021, ITX-1) on SYNTHETIC run tables over the committed instance sets: the ε-floored effect,
the ratio-of-means bootstrap, run selection (void / failed / duplicates / integrity), the target, the minimum, the
stratified estimates, the computed race and the secondaries, the geo-means, D-019's literal readings, both clocks
(R10) and M6's lane reading."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from itrea.analysis import itx_report as X
from itrea.analysis import metrics as M
from itrea.analysis.itx_strata import read_strata

LAB = X.labels("26.10")
INTACT = dict(incumbents_missing="", n_inc_unchecked=0, inc_count_mismatch="", pre_t0=False, n_inc_bad_t=0,
              n_inc_failed=0, n_obj_mismatch=0)


def _run(rid, inst, label, seed, t_hit, z=10.0, worse=None, off=1.0, status="completed", **kw):
    """One run row and its checked incumbents: `worse` (objective) at 1 s if given, z at t_hit (None: never)."""
    row = dict(run_id=rid, instance=inst, arm=label, seed=seed, budget=600.0, synthetic=False, status=status,
               block_void=False, ref=z, sense=1, startup_offset_s=off, stop_kind="", set="x", **INTACT)
    row.update(kw)
    inc = []
    if worse is not None:
        inc.append(dict(run_id=rid, role="solver", t=1.0, obj_recomputed=worse, ok=True))
    if t_hit is not None:
        inc.append(dict(run_id=rid, role="solver", t=float(t_hit), obj_recomputed=z, ok=True))
    return row, inc


def _table(spec):
    rows, inc = [], []
    for k, args in enumerate(spec):
        r, i = _run(f"r{k}", *args[:4], **(args[4] if len(args) > 4 else {}))
        rows.append(r)
        inc += i
    return pd.DataFrame(rows), pd.DataFrame(inc, columns=["run_id", "role", "t", "obj_recomputed", "ok"])


def test_d_eps_and_ratio_of_means():
    assert X.eps_for(600) == 0.1 and X.eps_for(300) == 0.2
    d = X.d_eps([0.2, 0.0, 0.01, 2.0], [0.1, 0.0, 0.02, 2.0], 0.1)
    assert d.tolist() == pytest.approx([0.5, 0.0, -0.1, 0.0])                # the floor: 0.01 vs 0.02 -> -0.1
    r = X.ratio_of_means([1.0, 1.0, 2.0], [2.0, 2.0, 4.0])
    assert r["ratio"] == 0.5 and r["lo"] == r["hi"] == 0.5 and r["n"] == 3
    r = X.ratio_of_means(np.linspace(0.1, 1, 30), np.linspace(0.2, 1.2, 30), bca=True)
    assert r["lo"] < r["ratio"] < r["hi"] and r["hi95"] <= r["hi"] and r["bca_lo"] < r["bca_hi"]
    assert X.sign_test(10, 0) < 0.01 and X.sign_test(0, 0) == 1.0


def test_select_runs():
    runs, _ = _table([("a", LAB["A9"], 2000, 30), ("b", LAB["A9"], 2000, 30, dict(status="void")),
                      ("c", LAB["A9"], 2000, None, dict(status="failed")),
                      ("d", LAB["A9"], 2000, 30), ("d", LAB["A9"], 2000, 31),
                      ("e", LAB["A9"], 2000, 30, dict(n_inc_unchecked=2)), ("f", LAB["A9"], 2001, 30)])
    ch, why = X.select(runs, LAB["A9"], 2000)
    assert sorted(ch) == ["a", "c"]                                     # failed counts (an arm outcome); void not
    assert "status void" in why["b"][0] and "refused" in why["d"][0] and "not intact" in why["e"][0]


def _synthetic_report(gap_launch=0.0):
    """Every instance that runs (the 233 minus the list): A6F reaches z* at 60 s, A9 at 30 s on H_cu-unseen and at
    60 s elsewhere; A0 at 90 s; on W_SCIP-unseen A0 at 300 s. The launch offsets differ by `gap_launch`."""
    st = read_strata()
    spec = []
    for _, r in st[~st["does_not_fit"]].iterrows():
        n = r["name"]
        h = bool(r["decide"] and r["H_cu"])
        spec.append((n, LAB["A9"], 2000, 30.0 if h else 60.0, dict(startup_offset_s=1.0 + gap_launch)))
        spec.append((n, LAB["A6F"], 2000, 60.0, dict(startup_offset_s=1.0)))
        spec.append((n, LAB["A0"], 2000, 300.0 if (r["decide"] and r["W_SCIP"]) else 90.0,
                     dict(startup_offset_s=0.2)))
        if r["panel"]:
            spec.append((n, LAB["A9"], 2001, 36.0 if h else 60.0))
            spec.append((n, LAB["A6F"], 2001, 60.0))
    runs, inc = _table(spec)
    return X.report(runs, inc, st), st


def test_report_on_synthetic_runs():
    res, st = _synthetic_report()
    ag = res["aggregates"]
    c = ag["call"]
    assert ag["counts"] == dict(decide=215, H_cu_unseen=64, W_SCIP_unseen=62, run_set=230)
    t = c["target"]
    assert (t["n"], t["missing"]) == (64, 0) and t["ratio"] == pytest.approx(0.5) and t["met"] and t["better_shown"]
    assert (t["improved"], t["worse"]) == (64, 0) and t["sign_p"] < 1e-15
    g = c["guard"]
    # d = 0.5 on the 64 H_cu-unseen, 0 on the other 151 of the decision set
    assert g["n"] == 215 and g["mean_d"] == pytest.approx(64 * 0.5 / 215) and g["passes"] and not g["harm_shown"]
    assert c["solutions"]["decide"] == dict(n=215, A9=215, A6F=215) and c["solutions"]["passes"] and c["minimum"]
    assert c["solutions"]["run_set"]["n"] == 230
    sg = c["stratified"]["guard"]
    assert sg["complete"] and sg["estimate"] == pytest.approx(64 * 0.5 / 215) and sg["total"] == 215
    assert c["stratified"]["solutions_diff"]["count_estimate"] == 0.0
    # the race: A0 at 90 / A6F at 60 -> the race reaches z* at 60, A9 at 30 on H_cu: d = 0.5; on W_SCIP A0 at 300
    sec = c["secondary"]
    assert sec["A9_vs_race_Hcu"]["mean_d"] == pytest.approx(0.5) and sec["A9_vs_race_Hcu"]["n"] == 64
    w = sec["A9_vs_A0_Wscip"]
    assert w["n"] == 62 and w["mean_d"] > 0.5 and w["holm_p"] >= w["wilcoxon_p"]
    # geo-means and D-019's literal readings
    gm = c["geomeans"]["A9"]["run_set"]
    assert gm["n"] == 230 and gm["shift_0.001"] > 0 and gm["shift_0.0333333"] > gm["shift_0.001"] - 1
    lit = c["d019_literal"]
    assert lit["n"] == 230 and lit["minimum"] and lit["hard_n"] == 230                 # every A6F pt_gams = 0.2
    assert lit["hard_ratio"] == pytest.approx((64 * 0.1 + 166 * 0.2) / (230 * 0.2))
    assert ag["reading"] == dict(target="met", minimum="met")
    assert ag["panel_noise"]["A9"]["n"] == 20 and ag["panel_noise"]["A6F"]["p90_abs_d"] == 0.0
    inst = {r["instance"]: r for r in res["instances"]}
    hn = st[st["decide"] & st["H_cu"]]["name"].iloc[0]
    assert inst[hn]["A9_pt_gams_call"] == pytest.approx(2 * 30 / 600)
    assert inst[hn]["A9_pt_gams_launch"] == pytest.approx(2 * 31 / 600)
    assert inst[hn]["A9_P300_call"] == pytest.approx(30 / 300) and inst[hn]["race_pt_gams_call"] == pytest.approx(0.2)
    assert inst[hn]["race_pt_gams_launch"] == pytest.approx(2 * 61 / 600)          # A6F at 60 + its offset 1
    assert res["tiers"] == dict(T1=dict(instances=64, pairs=64), T2=dict(instances=57, pairs=57),
                                T3=dict(instances=109, pairs=109))


def test_better_shown_reads_the_ratio_interval():
    """The target's "better shown" reads the RATIO's bootstrap upper end, not the d_ε interval's: A9 twice as fast on
    half of H_cu-unseen and 1.5 x slower on the other half -> mean ratio 1 (upper end above 1: not shown) while the
    d_ε interval lies below 1 (+0.5 / −0.33)."""
    st = read_strata()
    hcu = list(st[st["decide"] & st["H_cu"]]["name"])
    spec = []
    for k, n in enumerate(hcu):
        spec.append((n, LAB["A9"], 2000, 30.0 if k % 2 == 0 else 90.0))
        spec.append((n, LAB["A6F"], 2000, 60.0))
    runs, inc = _table(spec)
    t = X.report(runs, inc, st)["aggregates"]["call"]["target"]
    assert t["n"] == t["n_d"] == 64 and t["ratio"] == pytest.approx(1.0)
    assert t["lo"] < 1.0 < t["hi"] and not t["better_shown"] and not t["met"]
    assert t["d_hi"] < 1.0 and t["d_mean"] == pytest.approx((0.5 - 1 / 3) / 2)
    assert (t["improved"], t["worse"]) == (32, 32)


def test_both_clocks_and_incomplete_t3():
    """A9 started 25 s late on the launch clock only: the target holds on the call clock only (R10). A missing pair
    makes T3 incomplete and the stratified estimate the one to read."""
    res, st = _synthetic_report(gap_launch=25.0)
    ag = res["aggregates"]
    assert ag["call"]["target"]["met"] and not ag["launch"]["target"]["met"]
    assert ag["reading"]["target"] == "met on the call clock only"
    # drop one T3 instance's A9 run
    st_ = read_strata()
    t3 = st_[(st_["tier"] == "T3") & st_["unseen"] & ~st_["e_sample"]]["name"].iloc[0]
    runs = pd.DataFrame([r for r in res["instances"]])            # (just to reuse names)
    assert t3 in set(runs["instance"])
    spec_runs, inc = [], []
    for _, r in st_[~st_["does_not_fit"]].iterrows():
        n = r["name"]
        if n != t3:
            spec_runs.append((n, LAB["A9"], 2000, 60.0))
        spec_runs.append((n, LAB["A6F"], 2000, 60.0))
    rr, ii = _table(spec_runs)
    out = X.report(rr, ii, st_)["aggregates"]["call"]
    assert not out["t3_complete"] and out["guard"]["missing"] == 1 and out["stratified"]["guard"]["complete"]
    assert out["stratified"]["guard"]["estimate"] == 0.0


def test_m6_report():
    rows = [dict(arm="A0@m6a", instance=n, solver_nodes_per_s=v, set="itx-m6a", acpi_during_max_c=60.0)
            for n, v in (("a", 1000.0), ("b", 500.0), ("c", 800.0))]
    rows += [dict(arm="A0@m6b", instance=n, solver_nodes_per_s=v, set="itx-m6", acpi_during_max_c=79.0)
             for n, v in (("a", 990.0), ("b", 495.0), ("c", 790.0))]
    rows += [dict(arm="A0@m6L", instance="d", solver_nodes_per_s=1.0, set="itx-m6", acpi_during_max_c=83.5)]
    r = X.m6_report(pd.DataFrame(rows))
    assert r["lanes"] == 4 and r["hottest_c"] == 83.5 and r["mean_slowdown"] == pytest.approx(0.01, abs=2e-3)
    rows[3]["solver_nodes_per_s"] = 900.0                         # 10 % slower on a
    assert X.m6_report(pd.DataFrame(rows))["lanes"] == 3
    rows[3]["solver_nodes_per_s"], rows[-1]["acpi_during_max_c"] = 990.0, 86.0
    assert X.m6_report(pd.DataFrame(rows))["lanes"] == 3
