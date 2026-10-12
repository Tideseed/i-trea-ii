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
    assert ag["counts"] == dict(decide=215, H_cu_unseen=64, W_SCIP_unseen=62, run_set=230, exposed_decide=31)
    t = c["target"]
    assert (t["n"], t["N"], t["missing"]) == (64, 64, 0) and t["ratio"] == pytest.approx(0.5) and t["met"] and \
        t["better_shown"] and t["complete"]
    assert (t["improved"], t["worse"]) == (64, 0) and t["sign_p"] < 1e-15
    g = c["guard"]
    # d = 0.5 on the 64 H_cu-unseen, 0 on the other 151 of the decision set
    assert g["n"] == 215 and g["mean_d"] == pytest.approx(64 * 0.5 / 215) and g["passes_on_available"] and \
        not g["harm_shown_on_available"]
    assert c["solutions"]["decide"] == dict(n=215, A9=215, A6F=215) and c["solutions"]["passes_on_available"]
    assert c["minimum"] and c["minimum_basis"] == "full" and c["completeness"]["full_complete"]
    assert c["minimum_detail"]["guard"] == pytest.approx(64 * 0.5 / 215)
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
    assert out["census_complete"] and out["minimum_basis"] == "stratified" and out["minimum"]


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


# ----------------------------------------------------------------------------------- review fixes (2026-10-12)
def _runs(st, t9, t6=None, rows=None, t0=None, ta2f=None, off9=None):
    """A synthetic run table over the run set (or the rows `rows` accepts): A9 at t9(row) (None: no solution, False:
    no run), A6F at t6(row) (default 60 s), A0 at t0(row) if given, live A2F at ta2f(row) on the panel (t5)."""
    spec = []
    for _, r in st[~st["does_not_fit"]].iterrows():
        if rows is not None and not rows(r):
            continue
        n = r["name"]
        x = t9(r)
        if x is not False:
            spec.append((n, LAB["A9"], 2000, x, dict(startup_offset_s=1.0 if off9 is None else off9(r))))
        spec.append((n, LAB["A6F"], 2000, 60.0 if t6 is None else t6(r), dict(startup_offset_s=1.0)))
        if t0 is not None:
            spec.append((n, LAB["A0"], 2000, t0(r), dict(startup_offset_s=0.2)))
        if ta2f is not None and r["t5"]:
            spec.append((n, LAB["A2F"], 2000, ta2f(r), dict(startup_offset_s=1.5)))
    return _table(spec)


def test_minimum_reads_the_stratified_estimates_while_t3_is_incomplete():
    """Review blocker (R31): T1 and T2 complete, T3 not run — A9 +0.5 on the 64 H_cu-unseen, 0 on the 17 W-only,
    −0.7 on the 40 E-sample instances (A9 at 200 s against A6F at 60 s). The plain guard over the 121 pairs present
    (+0.033) would pass; R31's stratified guard over the 215 is −0.287 with its interval below −0.05: the minimum is
    read on the stratified estimates — not met, harm shown. Then (b): equal counts on the pairs present but the
    stratified count of solutions found below 0 also fails the minimum."""
    st = read_strata()
    t12 = lambda r: r["tier"] in ("T1", "T2")                              # noqa: E731

    def t9(r):
        return 30.0 if r["tier"] == "T1" else 200.0 if r["e_sample"] else 60.0
    runs, inc = _runs(st, t9, rows=t12)
    res = X.report(runs, inc, st)
    ag = res["aggregates"]
    c = ag["call"]
    cm = c["completeness"]
    assert cm["census"] == dict(n=121, N=121) and cm["t3"] == dict(n=0, N=94) and cm["decision_set"] == dict(n=121,
                                                                                                            N=215)
    assert c["census_complete"] and not c["t3_complete"] and not cm["full_complete"]
    g = c["guard"]
    assert g["n"] == 121 and g["mean_d"] == pytest.approx((32 - 28) / 121) and g["passes_on_available"]
    sg = c["stratified"]["guard"]
    assert sg["estimate"] == pytest.approx((64 * 0.5 - 134 * 0.7) / 215) and sg["hi"] < X.GUARD
    md = c["minimum_detail"]
    assert c["minimum_basis"] == "stratified" and md["guard_passes"] is False and md["harm_shown"] is True
    assert md["solutions_pass"] is True and c["minimum"] is False and md["guard"] == pytest.approx(sg["estimate"])
    assert c["target"]["met"] and ag["reading"] == dict(target="met", minimum="not met")
    text = X.markdown(res)
    assert "read on: stratified" in text and "minimum not met" in text and "T3 INCOMPLETE" not in text
    # (b): A9 finds a solution on one W-only instance where A6F finds none, and none on one E-sample instance of
    # quartile 1 where A6F does: 0 on the pairs present; the stratified count 1 − N_1 / 10 < 0
    w1 = sorted(st[st["decide"] & st["W_only"]]["name"])[0]
    e1 = sorted(st[st["decide"] & st["e_sample"] & (st["e_quartile"] == 1)]["name"])[0]
    runs, inc = _runs(st, lambda r: None if r["name"] == e1 else 60.0,
                      t6=lambda r: None if r["name"] == w1 else 60.0, rows=t12)
    c = X.report(runs, inc, st)["aggregates"]["call"]
    s = c["solutions"]["decide"]
    assert (s["A9"], s["A6F"]) == (120, 120) and c["solutions"]["passes_on_available"]
    n1 = int((st["decide"] & st["E"] & (st["e_quartile"] == 1)).sum())
    assert c["stratified"]["solutions_diff"]["count_estimate"] == pytest.approx(1 - n1 / 10)
    assert c["minimum_basis"] == "stratified" and c["minimum_detail"]["solutions_pass"] is False and \
        c["minimum"] is False


def test_readings_are_incomplete_on_a_partial_census():
    """Review major (R10 / R31): 8 of the 64 H_cu-unseen pairs and nothing else — the target and the minimum read
    "incomplete (n of N)", never "met" (T1 and T2 are never cut, so no estimator reads them partial); the markdown no
    longer calls a gap in T1 a T3 gap. A gap in T1 alone (T3 complete) leaves both incomplete too."""
    st = read_strata()
    hcu = sorted(st[st["decide"] & st["H_cu"]]["name"])
    runs, inc = _runs(st, lambda r: 30.0, rows=lambda r: r["name"] in set(hcu[:8]))
    res = X.report(runs, inc, st)
    ag = res["aggregates"]
    c = ag["call"]
    t = c["target"]
    assert (t["n"], t["N"], t["missing"]) == (8, 64, 56) and t["met"] is None and t["better_shown"] is None
    assert t["met_on_available"] and not t["complete"]
    assert c["minimum"] is None and c["minimum_basis"] == "incomplete" and not c["census_complete"]
    assert ag["reading"] == dict(target="incomplete (8 of 64 H_cu-unseen pairs)",
                                 minimum="incomplete (census + E-sample 8 of 121 pairs, decision set 8 of 215)")
    text = X.markdown(res)
    assert "T3 INCOMPLETE" not in text and "Reading (R10): target incomplete (8 of 64" in text
    keep = set(st["name"]) - set(hcu[8:])
    runs, inc = _runs(st, lambda r: 60.0, rows=lambda r: r["name"] in keep)
    c = X.report(runs, inc, st)["aggregates"]["call"]
    assert c["t3_complete"] and not c["census_complete"] and c["minimum"] is None and c["target"]["met"] is None


def test_decisions_with_and_without_the_exposed_instances():
    """R33: every decision is also read without the 31 decision instances exposed before prereg-v2 (calibration
    sample, M3's probe). A9 far better on the 8 exposed H_cu-unseen instances only: met with them (ratio 0.706), not
    without (0.8 on 56)."""
    st = read_strata()
    ex = set(st[st["exposed"]]["name"])

    def t9(r):
        if r["decide"] and r["H_cu"]:
            return 3.0 if r["name"] in ex else 48.0
        return 60.0
    res = X.report(*_runs(st, t9), st)
    ag = res["aggregates"]
    wo = ag["without_exposed"]
    assert set(wo["excluded"]) == ex and wo["n_excluded_decide"] == 31 and wo["call"]["N_decide"] == 184
    assert ag["call"]["target"]["ratio"] == pytest.approx((8 * 0.01 + 56 * 0.16) / (64 * 0.2))
    assert ag["call"]["target"]["met"] and ag["reading"]["target"] == "met"
    t = wo["call"]["target"]
    assert (t["n"], t["N"]) == (56, 56) and t["ratio"] == pytest.approx(0.8) and t["met"] is False
    assert wo["reading"]["target"] == "not met" and wo["reading"]["minimum"] == "met"
    assert wo["call"]["stratified"]["guard"]["total"] == 184 and wo["call"]["guard"]["n"] == 184
    assert wo["call"]["minimum_basis"] == "full" and wo["call"]["completeness"]["census"]["N"] == 121 - 8 - 11
    assert "Without the exposed instances" in X.markdown(res)


def test_bca_for_every_mean_d_eps():
    """R21: BCa as a sensitivity for the ratio AND for every mean d_ε (target, guard, secondaries): `--bca`."""
    d = np.linspace(-0.4, 0.9, 50) ** 3
    e = X.effect_summary(d, 0.1, bca=True)
    assert e["bca_lo"] < e["mean_d"] < e["bca_hi"] and "bca_lo" not in X.effect_summary(d, 0.1)
    assert X.bca_mean([0.2] * 5) == dict(bca_lo=0.2, bca_hi=0.2)
    assert math.isnan(X.bca_mean([0.1, 0.2])["bca_lo"])
    st = read_strata()
    res = X.report(*_runs(st, lambda r: 30.0 if (r["decide"] and r["H_cu"] and len(r["name"]) % 3) else 60.0,
                          t0=lambda r: 90.0), st, bca=True)
    c = res["aggregates"]["call"]
    assert c["target"]["d_bca_lo"] <= c["target"]["d_mean"] <= c["target"]["d_bca_hi"]
    assert c["guard"]["bca_lo"] <= c["guard"]["mean_d"] <= c["guard"]["bca_hi"]
    assert all(math.isfinite(v["bca_lo"]) for v in c["secondary"].values())
    assert "BCa [" in X.markdown(res)


def test_call_clock_only_reading_carries_the_startup_cost():
    """R10: "met on the call clock only" comes with the start-up cost — each arm's median / p90 start-up on the set —
    and the launch clock's figure beside it."""
    res, _ = _synthetic_report(gap_launch=25.0)
    ag = res["aggregates"]
    assert ag["reading"]["target"] == "met on the call clock only"
    det = ag["reading_detail"]["target"]
    assert det["startup"]["A9"]["median"] == pytest.approx(26.0) and det["startup"]["A6F"]["median"] == 1.0
    assert det["launch_ratio"] == pytest.approx(ag["launch"]["target"]["ratio"]) and det["launch_ratio"] > 0.75
    assert ag["startup"]["A0"]["decision_set"]["median"] == pytest.approx(0.2)
    inst = {r["instance"]: r for r in res["instances"]}
    assert all(r["A9_startup_s"] == 26.0 for r in inst.values() if r["A9_run"])
    assert "A9 26.00 / 26.00 s" in X.markdown(res)


def test_hcu_p05_and_live_a2f_against_the_race():
    """R3: the target's figures on H_cu at P >= 0.5 (30 decision instances) are reported beside; R8: live A2F against
    the computed race on the panel (A2F at 75 s, the race of A0 at 90 / A6F at 60 at 60 s: ratio 1.25, d_ε −0.2)."""
    st = read_strata()
    res = X.report(*_runs(st, lambda r: 30.0, t0=lambda r: 90.0, ta2f=lambda r: 75.0), st)
    c = res["aggregates"]["call"]
    tp = c["target_p05"]
    assert tp["N"] == int((st["decide"] & st["H_cu_p05"]).sum()) == 30 and tp["n"] == 30
    assert tp["ratio"] == pytest.approx(0.5)
    ar = c["A2F_vs_race_panel"]
    assert (ar["n"], ar["N"]) == (20, 20) and ar["ratio"] == pytest.approx(1.25) and ar["d_mean"] == pytest.approx(-0.2)
    assert res["aggregates"]["launch"]["A2F_vs_race_panel"]["n"] == 20
    assert "live A2F vs the computed race" in X.markdown(res)


def test_registered_settings_decide_which_runs_count():
    """Review finding (sets): with `registered` (the CLI always passes it) a v1 run counts only under the registered
    itx1 harness — for A0 only its SCIP-side part (R33) — and A9 only on the configuration registered with prereg-v2;
    no registered file, no A9 run."""
    import json
    from itrea.config import EAConfig
    from itrea.experiments import pair
    itx1 = pair.resolve_harness("itx1")
    h = lambda d: json.dumps(d, sort_keys=True)                            # noqa: E731
    reg = dict(harness=itx1, frozen="configs/frozen-v2.yaml", A9_base_hash="aaaa")
    spec = [("a", LAB["A9"], 2000, 30, dict(harness=h(itx1), base_hash="aaaa")),
            ("b", LAB["A9"], 2000, 30, dict(harness=h(itx1), base_hash="bbbb")),
            ("c", LAB["A9"], 2000, 30, dict(harness=h(pair.resolve_harness("ext")), base_hash="aaaa")),
            ("d", LAB["A9"], 2000, 30, dict(harness="", base_hash="aaaa")),
            ("a", LAB["A0"], 2000, 30, dict(harness=h(dict(itx1, cuopt_rmm_gb=30.0)))),
            ("b", LAB["A0"], 2000, 30, dict(harness=h(dict(itx1, cuda_cache=False)))),
            ("a", LAB["A6F"], 2000, 30, dict(harness=h(dict(itx1, cuopt_rmm_gb=30.0)))),
            ("b", LAB["A6F"], 2000, 30, dict(harness=h(itx1)))]
    runs, _ = _table(spec)
    ch, why = X.select(runs, LAB["A9"], 2000, registered=reg)
    assert sorted(ch) == ["a"] and "not the registered aaaa" in why["b"][0]
    assert "not the registered itx1" in why["c"][0] and "no harness recorded" in why["d"][0]
    ch, why = X.select(runs, LAB["A0"], 2000, registered=reg)
    assert sorted(ch) == ["a"] and "not the registered itx1" in why["b"][0]           # A0: SCIP-side keys only
    ch, why = X.select(runs, LAB["A6F"], 2000, registered=reg)
    assert sorted(ch) == ["b"] and "a" in why                                          # A6F: every key
    assert sorted(X.select(runs, LAB["A9"], 2000)[0]) == ["a", "b", "c", "d"]         # no check without it
    ch, why = X.select(runs, LAB["A9"], 2000, registered=dict(reg, A9_base_hash=None))
    assert not ch and "does not exist" in why["a"][0]
    r = X.registered_values("configs/itrea2-dev-rc-3way.yaml")
    assert r["A9_base_hash"] == EAConfig.from_yaml("configs/itrea2-dev-rc-3way.yaml").hash() and r["harness"] == itx1
    assert X.registered_values("configs/nothere.yaml")["A9_base_hash"] is None
