"""`itrea backoff-report` (`itrea.analysis.backoff`, D-018): the stage-0 rules (sanity, D_Y with its stop rule and the
INVALID verdict, the floor with its late-window sing44 limb against same-day A2 and its shortfall, yield_until_s and both
patiences inside their clamp ranges and on each clamp side, censored cuOpt times) and the stage-1 criteria (M1–M5, T, G
with the noise rule, the guard-set mean and "still beats A6", the ladder, F, the mean guard — each adoption gate with a
case that fails it —, the ca fallback classes, M6 on due times), on synthetic run tables and run directories — no
solver, no EA, no GPU. Also the stretch statistics, the stage-0 job file and the `bo_*` columns of build_runs."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itrea.analysis import backoff as BR

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------- synthetic runs
def _write_events(path: Path, events) -> None:
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def make_run(root: Path, set_: str, inst: str, arm: str, seed: int, pt: float, *, cuopt_t_first=math.nan,
             cuopt_pt=None, bo: dict | None = None, ea_events=(), solver_events=(), cuopt_events=None,
             gpu_util=80.0, summary_extra=None, budget=120.0, run_extra=None, best_obj=math.nan, sense=1,
             censored=False, **row_extra) -> dict:
    """One synthetic run (its directory and its runs.parquet row). `censored`: cuOpt found no incumbent — as
    build_runs records it, cuopt_t_first = the budget with cuopt_t_first_event false."""
    if censored:
        cuopt_t_first = budget
    rid = f"{inst}__{arm}__s{seed}__{budget:g}s__synthetic"
    d = root / set_ / rid
    d.mkdir(parents=True, exist_ok=True)
    rj = dict(run_id=rid, gpu_health=dict(verdict="ok", mean_util=gpu_util), stop=None)
    rj.update(run_extra or {})
    (d / "run.json").write_text(json.dumps(rj))
    if arm not in (BR.A6, BR.A2, BR.A6F):
        s = dict(exit_reason="budget", wall_s=budget, ls_calls=100, ls_cap_hits=0)
        if bo is not None:
            s["backoff"] = bo
        s.update(summary_extra or {})
        (d / "ea.summary.json").write_text(json.dumps(s))
        _write_events(d / "ea.events.jsonl", ea_events)
    _write_events(d / "solver.events.jsonl", solver_events)
    if cuopt_events is None:
        cuopt_events = ([dict(t=cuopt_t_first, kind="incumbent", obj=1.0), dict(t=cuopt_t_first, kind="send", k=1)]
                        if math.isfinite(cuopt_t_first) and not censored else [])
    _write_events(d / "cuopt.events.jsonl", cuopt_events)
    row = dict(run_id=rid, set=set_, instance=inst, arm=arm, seed=seed, budget=budget, status="completed",
               block_void=False, synthetic=False, run_dir=str(d), pt=pt, cuopt_pt=pt if cuopt_pt is None else cuopt_pt,
               cuopt_t_first=cuopt_t_first, cuopt_t_first_event=(not censored) if math.isfinite(cuopt_t_first) else None,
               n_inc_unchecked=0, n_inc_failed=0, gpu_health="ok", gpu_void=False, attempt=0, best_obj=best_obj,
               sense=sense)
    row.update(row_extra)
    return row


def bo_block(mode="shadow", busy_share=1.0, ls_steps=120_000.0, wall=120.0, first_lead=None, own=None,
             exit_t=0.4, reason="peek", levels=2, yield_level=2, exit_busy=None, **kw) -> dict:
    py = dict(length_s=exit_t, exit_t=exit_t, reason=reason)
    if exit_busy is not None:                                      # the governor's busy seconds at the exit (1y's M4)
        py["exit_busy_s"] = exit_busy
    b = dict(mode=mode, busy_share=busy_share, ls_steps=ls_steps, wall_s=wall, busy_s=busy_share * wall,
             phase_y=py, levels=levels, yield_level=yield_level,
             first_lead=None if first_lead is None else dict(t=first_lead, busy_s=first_lead),
             own_feasible=None if own is None else dict(t=own, busy_s=own), max_launch_s=0.05, patience_own_s=30.0)
    b.update(kw)
    return b


def traj_rows(run_id: str, pts) -> list[dict]:
    return [dict(run_id=run_id, role="cuopt", t=float(t), obj_recomputed=float(v), ok=True) for t, v in pts]


# ================================================================================================ stage 0
A6_SING = [(0.5, 100.0), (2.0, 90.0), (4.0, 80.0), (6.0, 70.0), (8.0, 60.0), (10.0, 50.0), (12.0, 40.0)]
# sing44 A2 (SCIP || cuOpt, zero EA duty): most values in cuOpt's first seconds, then v* = 60 at 40 s (its best by
# 60 s) and a later 50 (after 60 s: not v*) — the early / late split of the pilot (53 of 55 shared values in 5 s)
A2_SING = [(0.5, 100.0), (1.0, 95.0), (1.5, 92.0), (2.0, 90.0), (3.0, 85.0), (4.0, 80.0), (5.0, 75.0), (40.0, 60.0),
           (90.0, 50.0)]
LATE = {"A9@bo0": 2.0, "A9@bof50": 1.3, "A9@bof25": 1.10, "A9@bof12": 1.05}


def sing44_arm_traj(early: float, late: float | None) -> list[tuple[float, float]]:
    """An arm's sing44 cuOpt trajectory against A2_SING: the early values stretched by `early`, v* (60) by `late`
    (None: never reached — nor anything better — within 120 s)."""
    pts = [(t * early, v) for t, v in A2_SING if t <= 5.0]
    if late is not None:
        pts += [(40.0 * late, 60.0), (min(90.0 * late, 119.0), 50.0)]
    return pts


def stage0_world(tmp_path, *, s_first=None, late=None, early=1.05, comp_p=None, duty_err=None, ca_lead=50.0,
                 ca_inc=24.0, ca_send=25.0, glass4_lead=47.7, glass4_own=10.0, sing44_lead=None, sing44_own=None,
                 a2_traj=None, censored=()):
    """A6 / bo0 / bof50 / bof25 / bof12 on the three duty instances, A2 on sing44, ca and glass4 bo0, and two older A6
    comp21 runs (the spread); `censored` = {(instance, arm)} whose cuOpt found nothing. Returns (runs, incumbents)."""
    s_first = s_first or {"A9@bo0": 7.5, "A9@bof50": 3.75, "A9@bof25": 1.875, "A9@bof12": 1.25}
    late = dict(LATE, **(late or {}))
    comp_p = comp_p or {"A9@bo0": 0.70, "A9@bof50": 0.60, "A9@bof25": 0.53, "A9@bof12": 0.51}
    duty_err = duty_err or {}
    rows, inc = [], []
    S = "backoff0"
    t_a6 = 8.0
    rows.append(make_run(tmp_path, S, BR.CHROMATIC, BR.A6, 2, 0.07, cuopt_t_first=t_a6,
                         censored=(BR.CHROMATIC, BR.A6) in censored))
    a6s = make_run(tmp_path, S, BR.SING44, BR.A6, 2, 0.70, cuopt_t_first=0.5)
    rows.append(a6s)
    inc += traj_rows(a6s["run_id"], A6_SING)
    a2s = make_run(tmp_path, S, BR.SING44, BR.A2, 2, 0.60, cuopt_t_first=0.5)
    rows.append(a2s)
    inc += traj_rows(a2s["run_id"], A2_SING if a2_traj is None else a2_traj)
    rows.append(make_run(tmp_path, S, BR.COMP21, BR.A6, 2, 0.50, cuopt_t_first=1.0))
    for seed, p in ((0, 0.49), (1, 0.53)):                                     # the classprobe A6 seeds: spread 0.04
        rows.append(make_run(tmp_path, "classprobe", BR.COMP21, BR.A6, seed, p, cuopt_t_first=1.0))
    for arm, d in [("A9@bo0", 1.0)] + list(BR.FORCED.items()):
        for inst in (BR.CHROMATIC, BR.SING44, BR.COMP21):
            duty = d + duty_err.get((arm, inst), 0.0)
            kw_bo = {}
            if inst == BR.SING44 and arm == "A9@bo0" and sing44_lead is not None:
                kw_bo = dict(first_lead=sing44_lead, own=sing44_own)
            b = bo_block(mode="shadow" if d == 1.0 else "forced", busy_share=duty, ls_steps=120_000.0 * d, **kw_bo)
            kw = dict(bo=b, cuopt_t_first=s_first[arm] * t_a6 if inst == BR.CHROMATIC else 0.6,
                      censored=(inst, arm) in censored)
            if inst == BR.COMP21:
                kw["cuopt_pt"] = comp_p[arm]
            r = make_run(tmp_path, S, inst, arm, 2, 0.5, **kw)
            rows.append(r)
            if inst == BR.SING44:
                inc += traj_rows(r["run_id"], sing44_arm_traj(early, late[arm]))
    ca = bo_block(first_lead=ca_lead)
    rows.append(make_run(tmp_path, S, BR.CA, "A9@bo0", 2, 0.3, bo=ca, cuopt_t_first=ca_inc,
                         cuopt_events=[dict(t=ca_inc, kind="incumbent", obj=5.0), dict(t=ca_send, kind="send", k=1)]))
    rows.append(make_run(tmp_path, S, BR.GLASS4, "A9@bo0", 2, 0.3,
                         bo=bo_block(first_lead=glass4_lead, own=glass4_own), cuopt_t_first=0.1))
    return pd.DataFrame(rows), pd.DataFrame(inc)


def test_stretch_definitions():
    ref = [(0.5, 10.0), (1.0, 9.0), (2.0, 8.0), (3.0, 7.0), (4.0, 6.0), (5.0, 5.0), (6.0, 4.0)]
    arm = [(t * 2.0, v) for t, v in ref]
    assert BR.stretch(arm, ref) == (2.0, 6)                     # t_ref = 0.5 s is left out
    assert BR.stretch(arm[:4], ref) == (None, 3)                 # fewer than 5 shared values: n/a
    assert BR.stretch([(1.0, 9.0 + 1e-3)], ref)[1] == 0          # values must match (relative 1e-9)
    # the late stretch: v* = the reference's best by 60 s (not its later 50), first reached at 40 s
    a = BR.late_stretch(sing44_arm_traj(1.05, 1.4), A2_SING)
    assert a["v_star"] == 60.0 and a["t_ref"] == 40.0 and a["reached"] and a["S_late"] == pytest.approx(1.4)
    # reached by a BETTER value first (the arm jumps past v*): the first time at least as good counts
    b = BR.late_stretch([(1.0, 100.0), (44.0, 55.0)], A2_SING)
    assert b["t_arm"] == 44.0 and b["S_late"] == pytest.approx(1.1)
    never = BR.late_stretch(sing44_arm_traj(1.05, None), A2_SING)
    assert not never["reached"] and never["S_late"] == pytest.approx(120.0 / 40.0)
    late_only = BR.late_stretch([(130.0, 60.0)], A2_SING, budget=120.0)              # past the arm's budget
    assert not late_only["reached"]
    assert BR.late_stretch(A2_SING, [(61.0, 60.0)])["S_late"] is None             # no reference incumbent by 60 s
    mx = BR.late_stretch([(8.0, 7.0), (20.0, 9.0)], [(5.0, 7.0), (10.0, 9.0), (70.0, 12.0)], sense=-1)
    assert mx["v_star"] == 9.0 and mx["S_late"] == pytest.approx(2.0)              # maximisation


def test_stage0_proceeds_with_the_rules_values(tmp_path):
    runs, inc = stage0_world(tmp_path)
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path)
    ru = res["rules"]
    assert not res["missing"] and all(v["ok"] for v in res["sanity"].values())
    assert ru["a"]["D_Y"] == 0.25 and ru["a"]["qualifying"] == [0.25, 0.125]
    assert res["readouts"]["A9@bof25"]["sing44_S_late"] == pytest.approx(1.10)
    assert res["readouts"][BR.A2]["v_star"] == 60.0 and res["readouts"][BR.A2]["t_v_star"] == 40.0
    assert ru["b"]["D_min"] == 0.25 and ru["b"]["L"] == 2 and not ru["b"]["shortfall"]
    assert ru["b"]["limbs"][0.5]["sing44"] == "fail" and ru["b"]["limbs"][0.5]["comp21"] == "fail"
    assert ru["b"]["decided_by"].startswith("d = 0.5: sing44 limb fails")
    assert ru["b"]["comp21_limit"] == pytest.approx(0.50 + 0.04)
    assert ru["c"]["yield_until_s"] == 30                        # max(30, ceil(2 x 15))
    assert ru["d"]["patience_search_s"] == 40                    # ceil(1.25 x (50 − 25)) = 32 -> clamp 40
    assert ru["e"]["patience_own_s"] == 40 and ru["e"]["lags"]["glass4"] == pytest.approx(37.7)
    assert res["bo1"] == dict(enabled=True, mode="active", yield_level=2, levels=2, yield_until_s=30.0,
                              patience_search_s=40.0, patience_own_s=40.0)
    assert res["verdict"].startswith("PROCEED")
    BR.print_stage0(res, out=open(tmp_path / "out.txt", "w"))
    assert "VERDICT: PROCEED" in (tmp_path / "out.txt").read_text()


def test_stage0_stop_rule_and_shortfall(tmp_path):
    runs, inc = stage0_world(tmp_path, s_first={"A9@bo0": 7.5, "A9@bof50": 5.0, "A9@bof25": 3.0, "A9@bof12": 2.5},
                             late={"A9@bo0": 1.5, "A9@bof50": 1.4, "A9@bof25": 1.3, "A9@bof12": 1.2},
                             ca_lead=math.nan)
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path)
    assert res["rules"]["a"]["D_Y"] is None and res["verdict"].startswith("STOP") and res["bo1"] is None
    assert not res["rules"]["a"]["not_counted"]                    # every forced run counted: a real STOP
    assert res["rules"]["b"]["shortfall"] and res["rules"]["b"]["D_min"] == 0.125 and res["rules"]["b"]["L"] == 3
    assert res["rules"]["b"]["decided_by"].startswith("shortfall")
    assert res["rules"]["d"]["patience_search_s"] == 40           # no material lead on ca


def test_stage0_invalid_when_sanity_failed_and_nothing_qualified(tmp_path):
    """No duty qualifies for rule (a), but bof25 — whose S_first would have qualified — did not count: the
    measurement failed, not the mechanism — INVALID, not STOP (D-018 amendment)."""
    runs, inc = stage0_world(tmp_path, s_first={"A9@bo0": 7.5, "A9@bof50": 5.0, "A9@bof25": 1.5, "A9@bof12": 2.5},
                             duty_err={("A9@bof25", BR.CHROMATIC): 0.10})
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path)
    assert res["rules"]["a"]["D_Y"] is None and res["rules"]["a"]["not_counted"] == [0.25]
    assert res["verdict"].startswith("INVALID") and "STOP" not in res["verdict"].split(":")[0] and res["bo1"] is None
    assert "d = 0.25" in res["verdict"] and "S_first 1.50" in res["verdict"]
    BR.print_stage0(res, out=open(tmp_path / "out.txt", "w"))
    assert "VERDICT: INVALID" in (tmp_path / "out.txt").read_text()


def test_stage0_rule_b_reads_the_late_window_against_a2(tmp_path):
    """The early / late split of the pilot: every arm's early values are only 5 % late against A2, so a median over the
    shared values would pass every duty (it would pick D_min = 1/2); v* (A2's best by 60 s) is reached 40 % late at
    1/2, so the late limb fails there and the floor is 1/4 — decided by the sing44 limb (comp21 passes everywhere)."""
    runs, inc = stage0_world(tmp_path, late={"A9@bof50": 1.4, "A9@bof25": 1.12, "A9@bof12": 1.05},
                             comp_p={"A9@bo0": 0.70, "A9@bof50": 0.52, "A9@bof25": 0.52, "A9@bof12": 0.51})
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path)
    b = res["rules"]["b"]
    assert b["D_min"] == 0.25 and b["qualifying"] == [0.25, 0.125] and not b["shortfall"]
    assert b["limbs"][0.5] == dict(sing44="fail", S_late=pytest.approx(1.4), reached=True, comp21="pass",
                                   comp21_P=0.52, counted=True)
    assert b["decided_by"] == "d = 0.5: sing44 limb fails (S_late 1.400)"
    # the statistic a median-of-shared-values reading would use passes 1/2 on the same trajectories
    rid = runs[(runs["instance"] == BR.SING44) & (runs["arm"] == "A9@bof50")]["run_id"].iloc[0]
    a2 = runs[(runs["instance"] == BR.SING44) & (runs["arm"] == BR.A2)]["run_id"].iloc[0]
    med, n = BR.stretch(BR.cuopt_traj(inc, rid, 120.0), BR.cuopt_traj(inc, a2, 120.0), min_t_ref=0.0)
    assert n >= 7 and med <= BR.S_SING_MAX
    # an arm that never reaches v* fails the limb (120 / 40 = 3)
    runs2, inc2 = stage0_world(tmp_path / "never", late={"A9@bof50": None},
                               comp_p={"A9@bo0": 0.70, "A9@bof50": 0.52, "A9@bof25": 0.52, "A9@bof12": 0.51})
    b2 = BR.stage0(runs2, inc2, "backoff0", root=tmp_path / "never")["rules"]["b"]
    assert b2["limbs"][0.5]["sing44"] == "fail" and not b2["limbs"][0.5]["reached"]
    assert b2["limbs"][0.5]["S_late"] == pytest.approx(3.0) and b2["D_min"] == 0.25
    assert "v* never reached" in b2["decided_by"]


def test_stage0_rule_b_sing44_limb_na_without_an_a2_incumbent_by_60s(tmp_path):
    """A2 without a cuOpt incumbent by 60 s: the sing44 limb is n/a at every duty and the floor rests on comp21."""
    runs, inc = stage0_world(tmp_path, a2_traj=[(61.0, 60.0), (90.0, 50.0)],
                             late={"A9@bof50": 3.0, "A9@bof25": 3.0, "A9@bof12": 3.0})
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path)
    b = res["rules"]["b"]
    assert all(lb["sing44"] == "n/a" for lb in b["limbs"].values())
    assert b["D_min"] == 0.25 and b["decided_by"].startswith("d = 0.5: comp21 limb fails")      # comp21 at 1/2: 0.60
    assert "rests on comp21" in b["decided_by"] and any("no cuOpt incumbent by 60 s" in n for n in res["notes"])
    # and the A2 run is required: without it stage 0 is incomplete
    res2 = BR.stage0(runs[runs["arm"] != BR.A2], inc, "backoff0", root=tmp_path)
    assert res2["verdict"].startswith("INCOMPLETE") and (BR.SING44, BR.A2) in res2["missing"]


def test_stage0_censored_cuopt_times(tmp_path):
    """A cuOpt first incumbent censored at the budget is missing / ∞, never 120 s: an arm that found nothing has
    S_first = ∞ (not 120 / t_A6); an A6 that found nothing leaves rule (a) without a reference — INVALID."""
    rv = lambda **row: BR.RunView(row, tmp_path)                                         # noqa: E731
    assert math.isinf(rv(cuopt_t_first=120.0, cuopt_t_first_event=False).cuopt_t_first())
    assert rv(cuopt_t_first=8.5, cuopt_t_first_event=True).cuopt_t_first() == 8.5
    assert math.isnan(rv(cuopt_t_first=math.nan, cuopt_t_first_event=False).cuopt_t_first())   # no cuOpt role
    runs, inc = stage0_world(tmp_path, censored={(BR.CHROMATIC, "A9@bof50")})
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path)
    assert math.isinf(res["readouts"]["A9@bof50"]["S_first"]) and res["rules"]["a"]["D_Y"] == 0.25
    # the A6 reference and every arm censored: 120 / 120 = 1.0 would pass rule (a) — it must not
    every = {(BR.CHROMATIC, a) for a in (BR.A6, "A9@bo0", *BR.FORCED)}
    runs2, inc2 = stage0_world(tmp_path / "all", censored=every)
    res2 = BR.stage0(runs2, inc2, "backoff0", root=tmp_path / "all")
    assert res2["rules"]["a"]["D_Y"] is None and not res2["rules"]["a"]["qualifying"]
    assert all(math.isnan(res2["readouts"][a]["S_first"]) for a in BR.FORCED)
    assert res2["verdict"].startswith("INVALID") and "censored" in res2["verdict"] and res2["bo1"] is None


def test_stage0_a_run_that_fails_sanity_does_not_qualify(tmp_path):
    runs, inc = stage0_world(tmp_path, duty_err={("A9@bof25", BR.CHROMATIC): 0.10})
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path)
    assert not res["sanity"][f"A9@bof25 {BR.CHROMATIC}"]["ok"]
    assert res["rules"]["a"]["D_Y"] == 0.125 and 0.25 in res["rules"]["a"]["not_counted"]
    assert res["bo1"]["yield_level"] == 3 and res["bo1"]["yield_until_s"] == 30.0


@pytest.mark.parametrize("case, kw, want", [
    # (c) yield_until_s = max(30, ceil(2 t)): bof25 S_first 1.95 -> t = 15.6 -> 31.2 -> 32 (round would give 31)
    ("c inside", dict(s_first={"A9@bo0": 7.5, "A9@bof50": 3.75, "A9@bof25": 1.95, "A9@bof12": 1.25}),
     dict(yield_until_s=32)),
    ("c floor", {}, dict(yield_until_s=30)),                                  # 2 x 15 = 30
    # (d) clamp(ceil(1.25 (lead - first SEND)), 40, 90): 1.25 x (59.5 - 25) = 43.125 -> 44 (round: 43; the incumbent
    # event at 24 instead of the send: 44.375 -> 45)
    ("d inside", dict(ca_lead=59.5), dict(patience_search_s=44, t_first_partner=25.0)),
    ("d floor", dict(ca_lead=50.0), dict(patience_search_s=40)),              # ceil(31.25) = 32 -> 40
    ("d ceiling", dict(ca_lead=105.0), dict(patience_search_s=90)),           # 1.25 x 80 = 100 -> 90
    # (e) clamp(ceil(1.25 x the LARGER lag over sing44 and glass4), 20, 40): glass4 20.2 -> 25.25 -> 26 (round: 25)
    ("e inside", dict(glass4_lead=30.2, glass4_own=10.0, sing44_lead=15.0, sing44_own=5.0),
     dict(patience_own_s=26)),
    ("e floor", dict(glass4_lead=18.0, glass4_own=10.0), dict(patience_own_s=20)),     # 1.25 x 8 = 10 -> 20
    ("e ceiling", {}, dict(patience_own_s=40)),                               # 1.25 x 37.7 -> 48 -> 40
    ("e larger lag", dict(glass4_lead=12.0, glass4_own=10.0, sing44_lead=25.0, sing44_own=5.0),
     dict(patience_own_s=25)),                                                # sing44's 20 s, not glass4's 2 s
])
def test_stage0_rules_c_d_e_values(tmp_path, case, kw, want):
    runs, inc = stage0_world(tmp_path, **kw)
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path)
    ru = res["rules"]
    got = dict(yield_until_s=ru["c"]["yield_until_s"], patience_search_s=ru["d"]["patience_search_s"],
               t_first_partner=ru["d"].get("t_first_partner"), patience_own_s=ru["e"]["patience_own_s"])
    for k, v in want.items():
        assert got[k] == v, (case, k, got)
    assert res["verdict"].startswith("PROCEED")
    assert res["bo1"]["yield_until_s"] == float(got["yield_until_s"])
    assert res["bo1"]["patience_search_s"] == float(got["patience_search_s"])
    assert res["bo1"]["patience_own_s"] == float(got["patience_own_s"])


def test_stage0_steps_sanity_and_missing_runs(tmp_path):
    runs, inc = stage0_world(tmp_path)
    i = runs.index[(runs["arm"] == "A9@bof50") & (runs["instance"] == BR.SING44)][0]
    p = Path(runs.loc[i, "run_dir"]) / "ea.summary.json"
    s = json.loads(p.read_text())
    s["backoff"]["ls_steps"] = 120_000.0                           # duty 1/2 but the work of full duty
    p.write_text(json.dumps(s))
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path, steps_gate=True)       # the rule as first written
    assert not res["sanity"][f"A9@bof50 {BR.SING44}"]["steps_ok"] and 0.5 in res["rules"]["b"]["not_counted"]
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path)                         # D-018 deviation: duty gates, steps reported
    sn = res["sanity"][f"A9@bof50 {BR.SING44}"]
    assert not sn["steps_ok"] and sn["ok"] and 0.5 not in res["rules"]["b"]["not_counted"]
    i2 = runs.index[(runs["arm"] == "A9@bof25") & (runs["instance"] == BR.SING44)][0]
    p2 = Path(runs.loc[i2, "run_dir"]) / "ea.summary.json"
    s2 = json.loads(p2.read_text())
    s2["backoff"]["busy_share"] = 0.40                              # the duty limb still gates
    p2.write_text(json.dumps(s2))
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path)
    assert not res["sanity"][f"A9@bof25 {BR.SING44}"]["duty_ok"] and 0.25 in res["rules"]["b"]["not_counted"]
    res2 = BR.stage0(runs[runs["instance"] != BR.GLASS4], inc, "backoff0", root=tmp_path)
    assert res2["verdict"].startswith("INCOMPLETE") and (BR.GLASS4, BR.BO0) in res2["missing"]


# ================================================================================================ stage 1
def _gen_events(busy_at_60, busy_end, steps_rate, wall=120.0):
    ev = []
    for t in np.arange(10.0, wall + 1e-9, 10.0):
        busy = busy_at_60 * t / 60.0 if t <= 60 else busy_at_60 + (busy_end - busy_at_60) * (t - 60) / (wall - 60)
        ev.append(dict(t=float(t), kind="gen", bo_busy_s=busy, ls_steps=steps_rate * 10.0))
    return ev


def _rclock(n=3, paused=0.5):
    return [dict(t=10.0 * k, kind="rclock", gen=k, gen_s=2.0, bo_paused_s=paused, bo_wall_s=2.0 + paused,
                 o_s=0.5, calls=[dict(s=1.5, cf=True)]) for k in range(1, n + 1)]


def stage1_world(tmp_path, *, p_bo1=None, p_bo0=None, ca_escalates=False, ca_phase_y=0.4, glass4_noise=False,
                 leads_bo1=None, bo1_util=None, flat=(), bo0_events=None, bo1_events=None, bo1_summary=None,
                 bo1_row=None, yield_level=2, censored=()):
    """14 instances x {A6, A9@bo1s (control), A9@bo1} x seeds 0-1 where every criterion holds unless changed: `p_bo0` = {instance:
    (seed 0, seed 1)} (A6 follows it), `leads_bo1` = {instance: bo1's lead times}, `bo1_util` = {instance: bo1's GPU
    utilisation}, `flat` = M4 instances where bo1 works like the control (no step down), `bo0_events` / `bo1_events`
    = {instance: extra EA events}, `bo1_summary` / `bo1_row` = {instance: extra EA-summary / run-table fields of bo1},
    `censored` = {(instance, arm, seed)} whose cuOpt found nothing."""
    S = "backoff1"
    rows = []
    base = {i: 0.4 for i in BR.STAGE1}
    base.update({BR.CHROMATIC: 0.55, "neos-960392": 0.03, "neos-3004026-krka": 0.005, "fhnw-binpack4-48": 0.004,
                 "radiationm18-12-05": 0.01})
    p_bo1, p_bo0, leads_bo1 = p_bo1 or {}, p_bo0 or {}, leads_bo1 or {}
    bo1_util, bo0_events, bo1_events = bo1_util or {}, bo0_events or {}, bo1_events or {}
    bo1_summary, bo1_row = bo1_summary or {}, bo1_row or {}
    for i in BR.STAGE1:
        for s in BR.STAGE1_SEEDS:
            p0 = p_bo0[i][s] if i in p_bo0 else base[i]
            p1 = p_bo1.get(i, 0.11 if i == BR.CHROMATIC else p0)
            a6 = min(1.0, p0 + 0.3) if i not in BR.FLOOR else p0
            send = [dict(t=0.3, kind="incumbent", obj=1.0), dict(t=0.3, kind="send", k=1)]
            rows.append(make_run(tmp_path, S, i, BR.A6, s, a6, cuopt_t_first=8.0 if i == BR.CHROMATIC else 0.3,
                                 cuopt_events=send, gpu_util=75.0, best_obj=10.0,
                                 censored=(i, BR.A6, s) in censored))
            leads = [5.0, 12.0, 20.0] if i in BR.M3_SET else []
            if i == BR.CA:
                leads = [40.0]
            contrib = [dict(t=t, kind="contrib", what="lead", material=True, dg=0.1) for t in leads]
            ctrl_gen = _gen_events(60.0, 120.0, 1000.0)
            b0 = bo_block(first_lead=leads[0] if leads else None, exit_t=0.4)
            esc0 = [dict(t=0.0, kind="backoff", phase="Y", level=2, reason="start"),
                    dict(t=0.4, kind="backoff", phase="P", level=0, reason="y_peek")] + list(bo0_events.get(i, []))
            rows.append(make_run(tmp_path, S, i, BR.CTRL1, s, p0, cuopt_t_first=60.0 if i == BR.CHROMATIC else 0.3,
                                 cuopt_events=send, bo=b0, ea_events=esc0 + contrib + ctrl_gen, gpu_util=95.0,
                                 best_obj=10.0, censored=(i, BR.CTRL1, s) in censored))
            m4 = i in BR.M4_SET and i not in flat
            gen1 = _gen_events(30.0, 45.0, 300.0) if m4 else ctrl_gen
            yl = ca_phase_y if i == BR.CA else (0.4 if not (glass4_noise and i == BR.GLASS4) else 0.4)
            ev1 = [dict(t=0.0, kind="backoff", phase="Y", level=2, reason="start"),
                   dict(t=yl, kind="backoff", phase="P", level=0, reason="y_peek")]
            if i == BR.CA and ca_escalates:
                ev1.append(dict(t=30.0, kind="backoff", phase="P", level=1, reason="escalate", busy_s=30.0))
                leads1 = [60.0]
            else:
                leads1 = leads_bo1.get(i, leads)
            if m4:
                ev1.append(dict(t=50.0, kind="backoff", phase="P", level=1, reason="escalate", busy_s=40.0))
                ev1.append(dict(t=60.0, kind="backoff", phase="P", level=2, reason="escalate", busy_s=44.0))
            ev1 += list(bo1_events.get(i, []))
            c1 = [dict(t=t, kind="contrib", what="lead", material=True, dg=0.1) for t in leads1]
            b1 = bo_block(mode="active", busy_share=45.0 / 120.0 if m4 else 1.0,
                          ls_steps=300.0 * 120.0 if m4 else 120_000.0,          # as the gen events add up
                          first_lead=leads1[0] if leads1 else None, exit_t=yl, yield_level=yield_level)
            rows.append(make_run(tmp_path, S, i, BR.BO1, s, p1, cuopt_t_first=12.0 if i == BR.CHROMATIC else 0.3,
                                 cuopt_events=send, bo=b1, ea_events=ev1 + c1 + gen1 + _rclock(),
                                 gpu_util=bo1_util.get(i, 60.0), best_obj=10.0, summary_extra=bo1_summary.get(i),
                                 censored=(i, BR.BO1, s) in censored, **bo1_row.get(i, {})))
    return pd.DataFrame(rows)


GATES = ("M1", "M2", "M3", "M4", "M5", "T", "G", "F", "mean_guard")


def failed_gates(res: dict) -> set[str]:
    m, o = res["mechanism"], res["outcome"]
    return {k for k in GATES if not (m[k]["ok"] if k in m else o[k]["ok"])}


def test_stage1_adopt_when_everything_holds(tmp_path):
    runs = stage1_world(tmp_path)
    res = BR.stage1(runs, None, "backoff1", root=tmp_path, stage0_set=None)
    m, o = res["mechanism"], res["outcome"]
    assert not res["missing"]
    for k in ("M1", "M2", "M3", "M4", "M5"):
        assert m[k]["ok"], (k, m[k])
    assert m["M4"]["util_lower"] == 4 and m["M4"]["D_min"] == 0.25
    assert o["T"]["ok"] and o["T"]["d"] == pytest.approx(0.8)
    assert o["G"]["ok"] and o["ladder"]["ok"] and o["F"]["ok"] and o["mean_guard"]["ok"]
    assert res["verdict"].startswith("ADOPT")
    assert m["M3"]["rows"][0]["l0_share"] == pytest.approx((20.0 - 0.4) / 20.0)
    with open(tmp_path / "out.txt", "w") as f:
        BR.print_stage1(res, out=f)
    assert "VERDICT: ADOPT" in (tmp_path / "out.txt").read_text()


def test_stage1_m1_and_m2_failures(tmp_path):
    runs = stage1_world(tmp_path)
    i = runs.index[(runs["arm"] == BR.BO1) & (runs["instance"] == BR.CHROMATIC) & (runs["seed"] == 0)][0]
    runs.loc[i, "cuopt_t_first"] = 30.0                              # > 2 x A6's 8 s
    p = Path(runs.loc[i, "run_dir"]) / "ea.summary.json"
    s = json.loads(p.read_text())
    s["backoff"]["phase_y"]["exit_t"] = 1.2                         # partner posted at 0.3 s
    p.write_text(json.dumps(s))
    res = BR.stage1(runs, None, "backoff1", root=tmp_path, stage0_set=None)
    assert not res["mechanism"]["M1"]["ok"] and not res["mechanism"]["M2"]["ok"]
    assert res["verdict"].startswith("DO NOT ADOPT") and "M1" in res["verdict"] and "M2" in res["verdict"]


def test_stage1_ca_guard_failure_class_p(tmp_path):
    runs = stage1_world(tmp_path, p_bo1={BR.CA: 0.70}, ca_escalates=True)
    res = BR.stage1(runs, None, "backoff1", root=tmp_path, stage0_set=None)
    g = {r["instance"]: r for r in res["outcome"]["G"]["rows"]}
    assert not g[BR.CA]["ok"] and not res["outcome"]["G"]["ok"]
    cls = " ".join(res["classification"])
    assert "class (P)" in cls and "patience_search_s = 1.25 x the control's lag = 49.625" in cls   # 1.25 x (40 − 0.3)


def test_stage1_ca_guard_failure_class_y(tmp_path):
    runs = stage1_world(tmp_path, p_bo1={BR.CA: 0.70}, ca_phase_y=24.0)
    res = BR.stage1(runs, None, "backoff1", root=tmp_path, stage0_set=None)
    assert "class (Y)" in " ".join(res["classification"])
    assert not res["mechanism"]["M1"]["ok"]                          # phase Y outlived the partner's post


def test_stage1_noise_rule(tmp_path):
    """A guard that fails by the letter while bo1 never left level 0 and phase Y lasted < 1 s is noise: reported, not
    counted."""
    runs = stage1_world(tmp_path, p_bo1={BR.GLASS4: 0.60})
    res = BR.stage1(runs, None, "backoff1", root=tmp_path, stage0_set=None)
    g = res["outcome"]["G"]
    assert BR.GLASS4 in g["noise"] and not next(r for r in g["rows"] if r["instance"] == BR.GLASS4)["ok"]
    assert not next(b for b in g["still_beats_A6"] if b["instance"] == BR.GLASS4)["ok"]     # also exempt
    assert g["mean_d"] >= BR.GUARD_MEAN and g["ok"]
    # the same failure with a step down on glass4 is counted
    runs2 = stage1_world(tmp_path / "b", p_bo1={BR.GLASS4: 0.60})
    for i in runs2.index[(runs2["arm"] == BR.BO1) & (runs2["instance"] == BR.GLASS4)]:
        f = Path(runs2.loc[i, "run_dir"]) / "ea.events.jsonl"
        f.write_text(f.read_text() + json.dumps(dict(t=50.0, kind="backoff", phase="P", level=1,
                                                     reason="escalate", busy_s=45.0)) + "\n")
    g2 = BR.stage1(runs2, None, "backoff1", root=tmp_path / "b", stage0_set=None)["outcome"]["G"]
    assert not g2["noise"] and not g2["ok"]


def test_stage1_floor_and_guard_960(tmp_path):
    runs = stage1_world(tmp_path, p_bo1={"neos-3004026-krka": 0.02, "neos-960392": 0.06})
    res = BR.stage1(runs, None, "backoff1", root=tmp_path, stage0_set=None)
    assert not res["outcome"]["F"]["ok"] and not res["outcome"]["G"]["neos960"]["ok"]
    assert res["outcome"]["G"]["neos960"]["limit"] == pytest.approx(-0.01)


_ESC = dict(kind="backoff", phase="P", reason="escalate")
_GUARDS6 = ("item_placement_9903", BR.CA, "neos-3024952-loue", "neos-2657525-crna", BR.GLASS4, BR.SING44)
_LADDER = ("neos-3656078-kumeu", BR.COMP21, "neos-4763324-toguru")


@pytest.mark.parametrize("case, kw, gate", [
    # M3: fewer leads on item_placement (1 < 0.8 x 3) / a late first lead on loue / item_placement mostly below level 0
    ("M3 leads", dict(leads_bo1={"item_placement_9903": [5.0]}), "M3"),
    ("M3 first lead", dict(leads_bo1={"neos-3024952-loue": [15.0, 20.0, 25.0]}), "M3"),
    ("M3 level-0 share", dict(bo1_events={"item_placement_9903": [dict(_ESC, t=3.0, level=1, busy_s=3.0)]}), "M3"),
    # M4: kumeu's bo1 never backs off (busy share 1, the control's step rate) / no GPU utilisation drop
    ("M4 share and steps", dict(flat={"neos-3656078-kumeu"}), "M4"),
    ("M4 utilisation", dict(bo1_util={i: 99.0 for i in BR.M4_SET}), "M4"),
    # M5: each leg on its own
    ("M5 accounting", dict(bo1_events={BR.COMP21: [dict(t=95.0, kind="rclock", gen=9, gen_s=2.0, bo_paused_s=0.5,
                                                        bo_wall_s=2.8, o_s=0.5, calls=[dict(s=1.5, cf=True)])]}),
     "M5"),
    ("M5 o_s", dict(bo1_events={BR.COMP21: [dict(t=95.0, kind="rclock", gen=9, gen_s=2.0, bo_paused_s=0.5,
                                                 bo_wall_s=2.5, o_s=0.0, calls=[dict(s=2.5, cf=True)])]}), "M5"),
    ("M5 cap hits", dict(bo1_summary={BR.COMP21: dict(ls_cap_hits=10)}), "M5"),
    ("M5 overrun", dict(bo1_summary={BR.GLASS4: dict(wall_s=121.0)}), "M5"),
    ("M5 unchecked incumbents", dict(bo1_row={BR.SING44: dict(n_inc_unchecked=1)}), "M5"),
    ("M5 gpu health", dict(bo1_row={BR.SING44: dict(gpu_health="void")}), "M5"),
    # T: d >= 0.30 but P > 0.30 / P <= 0.30 but d < 0.30
    ("T P", dict(p_bo1={BR.CHROMATIC: 0.35}), "T"),
    ("T d", dict(p_bo0={BR.CHROMATIC: (0.40, 0.40)}, p_bo1={BR.CHROMATIC: 0.30}), "T"),
    # G, the guard-set mean: every binding guard at d = -0.06, each inside its own (widened) band, mean < -0.05
    ("G mean", dict(p_bo0={i: (0.37, 0.43) for i in _GUARDS6}, p_bo1={i: 0.4 / 0.94 for i in _GUARDS6}), "G"),
    # G, still beating A6: loue's bo0 beats A6 by 0.43, bo1 by only 0.14 (inside its band; a step down: not noise)
    ("G beats A6", dict(p_bo0={"neos-3024952-loue": (0.2, 0.6)}, p_bo1={"neos-3024952-loue": 0.6},
                        bo1_events={"neos-3024952-loue": [dict(_ESC, t=50.0, level=1, busy_s=50.0)]}), "G"),
    # the mean over the 11 non-floor instances < 0 (the ladder at d = -0.5) while T and G hold
    ("mean guard", dict(p_bo1={i: 0.8 for i in _LADDER}), "mean_guard"),
])
def test_stage1_each_adoption_gate_fails_on_its_own(tmp_path, case, kw, gate):
    """Every adoption gate has a case where it alone fails and the verdict says DO NOT ADOPT naming it (a gate made
    vacuous — ok always true — would pass the rest of the suite otherwise)."""
    runs = stage1_world(tmp_path, **kw)
    res = BR.stage1(runs, None, "backoff1", root=tmp_path, stage0_set=None)
    assert failed_gates(res) == {gate}, (case, failed_gates(res))
    assert res["verdict"].startswith("DO NOT ADOPT") and gate in res["verdict"]
    if case == "G mean":
        g = res["outcome"]["G"]
        assert all(r["ok"] for r in g["rows"]) and g["mean_d"] < BR.GUARD_MEAN and not g["noise"]
    if case == "G beats A6":
        g = res["outcome"]["G"]
        assert all(r["ok"] for r in g["rows"]) and g["mean_d"] >= BR.GUARD_MEAN
        assert [b["instance"] for b in g["still_beats_A6"] if not b["ok"]] == ["neos-3024952-loue"]


def test_stage1_the_ladder_is_reported_not_a_gate(tmp_path):
    """A ladder target outside its band fails the ladder row, but adoption does not require the ladder."""
    runs = stage1_world(tmp_path, p_bo1={BR.COMP21: 0.6})
    res = BR.stage1(runs, None, "backoff1", root=tmp_path, stage0_set=None)
    lad = res["outcome"]["ladder"]
    assert not lad["ok"] and [r["instance"] for r in lad["rows"] if not r["ok"]] == [BR.COMP21]
    assert not failed_gates(res) and res["verdict"].startswith("ADOPT")


@pytest.mark.parametrize("cens, ok", [
    ({(BR.CHROMATIC, BR.BO1, 0)}, False),                                  # bo1 found nothing, A6 did: fails
    ({(BR.CHROMATIC, BR.BO1, 0), (BR.CHROMATIC, BR.A6, 0)}, True),          # both censored: reported, not failed
    ({(BR.CHROMATIC, BR.A6, 0)}, True),                                     # bo1 found one where A6 did not
])
def test_stage1_m2_censored_first_incumbents(tmp_path, cens, ok):
    """A censored cuOpt first incumbent is ∞, never the budget: 120 <= 2 x 120 must not pass M2."""
    runs = stage1_world(tmp_path, censored=cens)
    res = BR.stage1(runs, None, "backoff1", root=tmp_path, stage0_set=None)
    m2 = res["mechanism"]["M2"]
    assert m2["ok"] is ok
    row = next(r for r in m2["rows"] if r["seed"] == 0)
    if len(cens) == 2:
        assert "both censored" in row["why"]
        with open(tmp_path / "out.txt", "w") as f:
            BR.print_stage1(res, out=f)
        assert "M2: holds (reported: seed 0: both censored" in (tmp_path / "out.txt").read_text()
    if not ok:
        assert row["why"] == "bo1 censored" and "M2" in res["verdict"]


def test_stage1_floor_seconds_of_a_censored_run_are_none(tmp_path):
    runs = stage1_world(tmp_path, censored={("neos-3004026-krka", BR.BO1, 1)})
    res = BR.stage1(runs, None, "backoff1", root=tmp_path, stage0_set=None)
    sec = next(r for r in res["outcome"]["F"]["rows"] if r["instance"] == "neos-3004026-krka")["seconds"]
    assert sec[f"{BR.BO1} s1"] == "none" and sec[f"{BR.BO1} s0"] == pytest.approx(0.3)


def test_stage1_m6_reads_the_due_time_of_a_step_down(tmp_path):
    """The control logs a would-be step down at a generation boundary (t = 50) that fell due at 30, before its first
    lead at 40: M6 counts it as before the lead (the logging time would not)."""
    late_log = dict(_ESC, t=50.0, t_due=30.0, level=1, busy_s=30.0)
    runs = stage1_world(tmp_path, bo0_events={BR.CA: [late_log]})
    res = BR.stage1(runs, None, "backoff1", root=tmp_path, stage0_set=None)
    rows = [r for r in res["mechanism"]["M6"]["rows"] if r["instance"] == BR.CA]
    assert rows and all(r["before_lead"] and r["would_escalate_t"] == 30.0 for r in rows)


@pytest.mark.parametrize("yield_level, s_first, accepted, sf", [
    (1, {}, False, "7.5"),                                                 # duty 1 = stage 0's bo0: 7.5 > 3
    (1, {"A9@bo0": 2.5}, True, "2.50"),
    (2, {}, False, "3.75"),                                                # duty 1/2 = bof50: 3.75 > 3
    (2, {"A9@bof50": 2.5}, True, "2.50"),
])
def test_stage1_ca_class_y_step_up_reads_stage0(tmp_path, yield_level, s_first, accepted, sf):
    """The ca class-(Y) fallback: D_Y one step up is accepted only if stage 0 measured S_first <= 3 at that duty —
    at duty 1 that is stage 0's bo0 run (shadow = full duty), never an invented 1.0."""
    sf0 = dict({"A9@bo0": 7.5, "A9@bof50": 3.75, "A9@bof25": 1.875, "A9@bof12": 1.25}, **s_first)
    runs0, _ = stage0_world(tmp_path / "s0", s_first=sf0)
    runs1 = stage1_world(tmp_path / "s1", p_bo1={BR.CA: 0.70}, ca_phase_y=24.0, yield_level=yield_level)
    runs = pd.concat([runs1, runs0], ignore_index=True)
    res = BR.stage1(runs, None, "backoff1", root=tmp_path, stage0_set="backoff0")
    cls = " ".join(res["classification"])
    assert "class (Y)" in cls
    d_up = 2.0 ** -(yield_level - 1)
    if accepted:
        assert f"Next variable: D_Y one step up ({d_up}; stage-0 S_first {sf} <= 3)" in cls
    else:
        assert f"D_Y one step up ({d_up}) is NOT accepted (stage-0 S_first {sf}" in cls


# ======================================================================================== jobs and columns
def test_stage0_job_file():
    from itrea.experiments import make_jobs, pair
    jobs = make_jobs.backoff_stage0()
    assert len(jobs) == 18
    labels = {(j["name"], pair.arm_label(j)) for j in jobs}
    for n in BR.CHROMATIC, BR.SING44, BR.COMP21:
        assert {(n, a) for a in ("A6", "A9@bo0", "A9@bof50", "A9@bof25", "A9@bof12")} <= labels
    assert {(BR.CA, "A9@bo0"), (BR.GLASS4, "A9@bo0")} <= labels
    a2 = [j for j in jobs if j["arm"] == "A2"]                     # rule (b)'s zero-duty reference (D-018 amendment)
    assert len(a2) == 1 and a2[0]["name"] == BR.SING44 and a2[0]["config"] is None and "variant" not in a2[0]
    s44 = next(j for j in jobs if j["name"] == BR.SING44 and j["arm"] == "A6")
    assert {k: a2[0][k] for k in ("instance", "set", "seed", "budget", "block", "stratum", "split")} == \
        {k: s44[k] for k in ("instance", "set", "seed", "budget", "block", "stratum", "split")}
    assert len(labels) == 18
    assert {j["seed"] for j in jobs} == {2} and {j["budget"] for j in jobs} == {120.0}
    assert all(j["config"] is None for j in jobs if j["arm"] in ("A6", "A2"))
    for j in jobs:
        if j["arm"] == "A9":
            cfg, ov = pair.effective_config(j)
            assert cfg.backoff.enabled and cfg.exchange.from_cuopt and ov["exchange.from_cuopt"] is True
            assert (ROOT / j["config"]).exists()
    assert len({j["block"] for j in jobs}) == 5
    on_disk = [json.loads(line) for line in (ROOT / "jobs" / "dev-backoff0.jsonl").read_text().splitlines()]
    assert on_disk == jobs                                           # the committed file is the writer's output


def test_build_runs_backoff_columns(tmp_path):
    from itrea.analysis import build_runs as BRN
    b = bo_block(mode="active", busy_share=0.4, first_lead=12.0, own=3.0, reason="tick",
                 seconds_per_level={"Y": 1.0, "0": 50.0, "2": 69.0}, n_lead=4, n_assist=1, escalations=3,
                 first_escalation=dict(t=40.0, busy_s=33.0), wakes=2)
    r = make_run(tmp_path, "x", "toy", "A9@bo1", 0, 0.1, bo=b)
    run = BRN.load_run(Path(r["run_dir"]))
    cols = BRN.backoff_stats(run)
    assert cols["bo_mode"] == "active" and cols["bo_busy_share"] == 0.4 and cols["bo_phase_y_exit"] == "tick"
    assert (cols["bo_level0_s"], cols["bo_level1_s"], cols["bo_level2_s"]) == (50.0, 0.0, 69.0)
    assert cols["bo_first_lead_busy_s"] == 12.0 and cols["bo_first_escalation_s"] == 33.0 and cols["bo_n_lead"] == 4
    assert set(cols) <= set(BRN.RUN_COLUMNS)
    plain = make_run(tmp_path, "x", "toy", "A3", 0, 0.1)
    assert BRN.backoff_stats(BRN.load_run(Path(plain["run_dir"]))) == {}


def test_stage1_band_uses_the_control_spread():
    """D-018: band_i = max(0.05, the larger seed-0 vs seed-1 bounded difference of the control or of A6); the control is
    CTRL1 (A9@bo1s), not A9@bo0 (bug found 2026-10-11: the band read bo0, which stage 1 never runs)."""
    from itrea.analysis import backoff as B
    R = {("x", B.CTRL1, 0): {"pt": 0.698}, ("x", B.CTRL1, 1): {"pt": 0.090},
         ("x", B.A6, 0): {"pt": 0.758}, ("x", B.A6, 1): {"pt": 0.783}}
    assert abs(B._band(R, "x") - B.bounded(0.698, 0.090)) < 1e-12
    R2 = {("y", B.A6, 0): {"pt": 0.5}, ("y", B.A6, 1): {"pt": 0.5}}
    assert B._band(R2, "y") == B.BAND_MIN


# ======================================================================================== stage 1y (D-023)
def _resolved(spec):
    from itrea.experiments import pair
    return pair.resolve_harness(spec)


def stage1y_world(tmp_path, *, p_ctrl=None, p_boy=None, leads_boy=None, leads_ctrl=None, phase_y=None,
                  boy_events=None, boy_summary=None, ctrl_summary=None, boy_row=None, censored=(), drop=(),
                  failed=(), harness=None, extra=(), extra_seeds=None):
    """The 11 instances of D-023's amendment (BR.STAGE1Y) x {A6F, A9@bo1s (control), A9@boY} x seeds 0-1 in set backoff2
    where every D-023 rule holds unless changed, plus the instances `extra` (outside the amended set: run in the set,
    read by no rule): `p_ctrl` = {instance: (seed 0, seed 1)} (A6F follows it), `p_boy` = {instance: P}, `leads_boy` /
    `leads_ctrl` = {instance: {seed: lead times}}, `phase_y` = {(instance, seed): (exit_t, exit_busy_s)} of boY (default
    0.4 s / 0.2 s, the partner posts at 0.3 s), `boy_events` = {instance: extra EA events}, `boy_summary` /
    `ctrl_summary` = {instance: extra EA-summary fields}, `boy_row` = {instance: extra run-table fields}, `censored` =
    {(instance, arm, seed)} whose cuOpt found nothing, `drop` = {(instance, arm, seed)} not run, `failed` = same, status
    failed; `harness` = {(instance, arm, seed): a run.json config.harness (a resolved dict, or None)} — every run
    carries the resolved ext harness by default (D-022 adopted); `extra_seeds` = {instance: seeds beyond 0-1} run in
    the set (outside the amended set's seeds: read by no rule)."""
    from itrea.experiments import make_jobs as MJ
    S = "backoff2"
    ext = _resolved(MJ.ITX_EXT)
    harness = harness or {}
    rows = []
    base = {i: 0.4 for i in BR.STAGE1}
    base.update({BR.CHROMATIC: 0.55, "neos-960392": 0.03, "neos-3004026-krka": 0.005, "fhnw-binpack4-48": 0.004,
                 "radiationm18-12-05": 0.01})
    p_ctrl, p_boy, phase_y = p_ctrl or {}, p_boy or {}, phase_y or {}
    leads_boy, leads_ctrl, boy_events = leads_boy or {}, leads_ctrl or {}, boy_events or {}
    boy_summary, ctrl_summary, boy_row = boy_summary or {}, ctrl_summary or {}, boy_row or {}
    extra_seeds = extra_seeds or {}
    send = [dict(t=0.3, kind="incumbent", obj=1.0), dict(t=0.3, kind="send", k=1)]
    for i in BR.STAGE1Y + tuple(extra):
        for s in BR.STAGE1_SEEDS + tuple(extra_seeds.get(i, ())):
            p0 = p_ctrl[i][s] if i in p_ctrl else base[i]
            p1 = p_boy.get(i, 0.11 if i == BR.CHROMATIC else p0)
            a6 = min(1.0, p0 + 0.3) if i not in BR.FLOOR else p0
            default_leads = [5.0, 12.0, 20.0] if i in BR.M3_SET else ([40.0] if i == BR.CA else [])
            lc = leads_ctrl.get(i, {}).get(s, default_leads)
            lb = leads_boy.get(i, {}).get(s, lc)
            gen = _gen_events(60.0, 120.0, 1000.0)

            def add(arm, p, **kw):
                if (i, arm, s) in drop:
                    return
                st = "failed" if (i, arm, s) in failed else "completed"
                hz = harness.get((i, arm, s), ext)
                rows.append(make_run(tmp_path, S, i, arm, s, p, gpu_util=80.0, best_obj=10.0, cuopt_events=send,
                                     censored=(i, arm, s) in censored, status=st,
                                     run_extra=dict(config=dict(harness=hz)), **kw))
            add(BR.A6F, a6, cuopt_t_first=8.0 if i == BR.CHROMATIC else 0.3)
            c_ev = [dict(t=0.0, kind="backoff", phase="Y", level=1, reason="start"),
                    dict(t=0.4, kind="backoff", phase="P", level=0, reason="y_peek")]
            c_ev += [dict(t=t, kind="contrib", what="lead", material=True, dg=0.1) for t in lc]
            add(BR.CTRL1, p0, cuopt_t_first=60.0 if i == BR.CHROMATIC else 0.3,
                bo=bo_block(first_lead=lc[0] if lc else None, exit_t=0.4, exit_busy=0.4, yield_level=1, levels=3),
                ea_events=c_ev + gen, summary_extra=dict(dict(ls_calls=100, ls_cap_hits=10), **ctrl_summary.get(i, {})))
            ty, by = phase_y.get((i, s), (0.4, 0.2))
            b_ev = [dict(t=0.0, kind="backoff", phase="Y", level=1, reason="start"),
                    dict(t=ty, kind="backoff", phase="P", level=0, reason="y_peek")] + list(boy_events.get(i, []))
            b_ev += [dict(t=t, kind="contrib", what="lead", material=True, dg=0.1) for t in lb]
            row_kw = dict(dict(cuopt_t_first=12.0 if i == BR.CHROMATIC else 0.3), **boy_row.get(i, {}))
            add(BR.BOY, p1,
                bo=bo_block(mode="active", busy_share=(120.0 - (ty - by)) / 120.0, first_lead=lb[0] if lb else None,
                            exit_t=ty, exit_busy=by, yield_level=1, levels=3),
                ea_events=b_ev + gen + _rclock(), summary_extra=dict(dict(ls_calls=100, ls_cap_hits=10),
                                                                     **boy_summary.get(i, {})),
                **row_kw)
    return pd.DataFrame(rows)


GATES_Y = ("M1", "M2", "M3", "M4", "M5", "T", "G", "ladder", "F", "mean_guard")
_LADDER_Y = ("neos-3656078-kumeu", "neos-4763324-toguru")          # the ladder targets of the amended set


def failed_gates_y(res: dict) -> set[str]:
    m, o = res["mechanism"], res["outcome"]
    return {k for k in GATES_Y if not (m[k]["ok"] if k in m else o[k]["ok"])}


def test_stage1y_adopt_when_everything_holds(tmp_path):
    runs = stage1y_world(tmp_path)
    res = BR.stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None)
    assert not res["missing"] and not failed_gates_y(res), failed_gates_y(res)
    assert res["expected"] == 66 and res["instances"] == list(BR.STAGE1Y) and not res["outside"]
    m = res["mechanism"]
    assert len(m["M1"]["rows"]) == 22 and not m["M1"]["n_a"] and m["M1"]["limit_s"] == 1.0
    assert m["M4"]["busy_share"] == pytest.approx(0.5) and len(m["M4"]["rows"]) == 22
    assert all(r["leads_boY"] == r["leads_ctrl"] == 6 for r in m["M3"]["rows"])
    assert res["outcome"]["T"]["d"] == pytest.approx(0.8)
    assert res["verdict"].startswith("ADOPT boY")
    assert set(res["mechanism_d018"]) == {"M1", "M3", "M5"} and res["mechanism_d018"]["M1"]["limit_s"] == 0.5
    assert res["effects"][BR.CHROMATIC]["d_bo0_vs_A6"] == pytest.approx((0.85 - 0.55) / 0.85)   # vs A6F's 0.85
    with open(tmp_path / "out.txt", "w") as f:
        BR.print_stage1y(res, out=f)
    txt = (tmp_path / "out.txt").read_text()
    assert "VERDICT: ADOPT boY" in txt and "D-018's own rules on these runs (reported, not gating)" in txt
    assert "ladder targets (a gate in 1y): hold" in txt and "runs: 66 of 66" in txt


@pytest.mark.parametrize("case, kw, gate", [
    # M1: phase Y ends 1.2 s after the partner's first post (0.3 s) on one run
    ("M1", dict(phase_y={(BR.SING44, 0): (1.5, 0.75)}), "M1"),
    # M2: chromaticindex cuOpt first incumbent 30 s inside boY against A6F's 8 s (> 2x)
    ("M2", dict(boy_row={BR.CHROMATIC: dict(cuopt_t_first=30.0)}), "M2"),
    # M3: item_placement 1 + 1 leads under boY against 3 + 3 (2 < 0.8 x 6)
    ("M3", dict(leads_boy={"item_placement_9903": {0: [5.0], 1: [5.0]}}), "M3"),
    # M4: phase Y busy 70 % on every boY run (pooled 0.7 > 0.6)
    ("M4", dict(phase_y={(i, s): (0.4, 0.28) for i in BR.STAGE1 for s in (0, 1)}), "M4"),
    # M5: an accounting breach / cap hits per call 0.25 against the control's 0.10 (+0.15)
    # M5 on a floor instance of the amended set: the mechanism rules read all 11
    ("M5 accounting", dict(boy_events={"fhnw-binpack4-48": [dict(t=95.0, kind="rclock", gen=9, gen_s=2.0,
                                                                 bo_paused_s=0.5, bo_wall_s=2.8, o_s=0.5,
                                                                 calls=[dict(s=1.5, cf=True)])]}), "M5"),
    ("M5 cap hits", dict(boy_summary={"neos-4763324-toguru": dict(ls_cap_hits=25)}), "M5"),
    ("T", dict(p_boy={BR.CHROMATIC: 0.35}), "T"),
    # G: sing44 worse beyond its band with phase Y >= 1 s (the noise rule does not exempt it)
    ("G", dict(p_boy={BR.SING44: 0.60}, phase_y={(BR.SING44, 0): (1.2, 0.6), (BR.SING44, 1): (1.2, 0.6)}), "G"),
    # the ladder targets gate in 1y: kumeu d = -0.33 beyond its band 0.05 (the mean over the nine stays >= 0)
    ("ladder", dict(p_boy={"neos-3656078-kumeu": 0.6}), "ladder"),
    # F, one-sided: krka's boY loses 0.015 > 0.005
    ("F", dict(p_boy={"neos-3004026-krka": 0.02}), "F"),
    # the mean over the nine < 0 with both ladder targets inside their wide band (seed spread 0.75): chromaticindex
    # +0.8 and kumeu, toguru -0.444 each
    ("mean guard", dict(p_ctrl={i: (0.2, 0.8) for i in _LADDER_Y}, p_boy={i: 0.9 for i in _LADDER_Y}),
     "mean_guard"),
])
def test_stage1y_each_adoption_gate_fails_on_its_own(tmp_path, case, kw, gate):
    """Every gate of 1y — D-023's M1–M5 and D-018 stage 1's outcome rules, the ladder included — has a case where it
    alone fails and the verdict names it."""
    runs = stage1y_world(tmp_path, **kw)
    res = BR.stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None)
    assert failed_gates_y(res) == {gate}, (case, failed_gates_y(res))
    assert res["verdict"].startswith("DO NOT ADOPT boY") and gate in res["verdict"]


def test_stage1y_d023_limits_differ_from_d018s(tmp_path):
    """D-023's own limits: an exit 0.8 s after the partner's post passes M1 (1.0 s) where D-018's 0.5 s fails; leads
    summed over seeds (1 + 5 against 3 + 3) pass where D-018's per-seed M3 fails; cap hits +0.10 pass where D-018's +0
    fails; D-018's rules are reported, not gating."""
    runs = stage1y_world(tmp_path, phase_y={("neos-4763324-toguru", 0): (1.1, 0.55)},
                         leads_boy={"neos-3024952-loue": {0: [5.0], 1: [5.0, 6.0, 7.0, 8.0, 9.0]}},
                         boy_summary={BR.SING44: dict(ls_cap_hits=20)})            # 0.20 = the control's 0.10 + 0.10
    res = BR.stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None)
    assert not failed_gates_y(res) and res["verdict"].startswith("ADOPT boY")
    d18 = res["mechanism_d018"]
    assert not d18["M1"]["ok"] and [(r["instance"], r["seed"]) for r in d18["M1"]["rows"] if not r["ok"]] == \
        [("neos-4763324-toguru", 0)]
    assert not d18["M3"]["ok"] and ("neos-3024952-loue", 0) in [(r["instance"], r["seed"]) for r in
                                                                 d18["M3"]["rows"] if not r["ok"]]
    assert not d18["M5"]["ok"] and d18["M5"]["cap_hits"]
    loue = next(r for r in res["mechanism"]["M3"]["rows"] if r["instance"] == "neos-3024952-loue")
    assert (loue["leads_boY"], loue["leads_ctrl"]) == (6, 6)
    # 1 + 3 = 4 leads against 6: 4 < 0.8 x 6 = 4.8 fails
    runs2 = stage1y_world(tmp_path / "b", leads_boy={"neos-3024952-loue": {0: [5.0], 1: [5.0, 6.0, 7.0]}})
    assert not BR.stage1y(runs2, None, "backoff2", root=tmp_path / "b", stage0_set=None)["mechanism"]["M3"]["ok"]


def test_stage1y_m4_is_pooled_not_per_run(tmp_path):
    """Short phase Y runs have share 1.0 (no sleep falls due before the bucket's burst is spent): the arm's pooled
    share over the 11 decides — 18 runs at 0.3 s busy throughout, chromaticindex 10 s and ca 30 s at half duty: 45.4 /
    85.4 = 0.53 <= 0.6 — and the per-run shares are listed."""
    py = {(i, s): (0.3, 0.3) for i in BR.STAGE1Y for s in (0, 1)}              # every short phase Y at share 1.0
    py.update({(BR.CHROMATIC, 0): (10.0, 5.0), (BR.CHROMATIC, 1): (10.0, 5.0), (BR.CA, 0): (30.0, 15.0),
               (BR.CA, 1): (30.0, 15.0)})
    runs = stage1y_world(tmp_path, phase_y=py)
    m4 = BR.stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None)["mechanism"]["M4"]
    want = (18 * 0.3 + 2 * 5.0 + 2 * 15.0) / (18 * 0.3 + 2 * 10.0 + 2 * 30.0)
    assert m4["busy_share"] == pytest.approx(want) and m4["ok"]
    assert sum(1 for r in m4["rows"] if r["share"] == pytest.approx(1.0)) == 18
    # a phase Y that never ended is read from length_s and busy_per_level["Y"]
    rv = BR.RunView(dict(), tmp_path)
    rv._cache[("summary", "ea")] = dict(backoff=dict(phase_y=dict(length_s=120.0, exit_t=None, reason="never"),
                                                     busy_per_level={"Y": 60.0}))
    assert BR._phase_y_share(rv) == (120.0, 60.0, "never")


def test_stage1y_missing_runs_and_both_readings_of_failed_runs(tmp_path):
    runs = stage1y_world(tmp_path, failed={(BR.CA, BR.BOY, 1), (BR.CA, BR.A6F, 1)})
    res = BR.run_stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None)
    assert res["kept_failed"] and len(res["kept_failed"]) == 2
    assert res["verdict"].startswith("ADOPT boY")                               # kept as D-002 §6 crashes
    assert res["exclude_reading"]["verdict"].startswith("INCOMPLETE") and len(res["exclude_reading"]["missing"]) == 2
    with open(tmp_path / "out.txt", "w") as f:
        BR.print_stage1y(res, out=f)
    txt = (tmp_path / "out.txt").read_text()
    assert "kept as failed" in txt and "VERDICT with the failed runs excluded" in txt
    ex = BR.run_stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None, failed="exclude")
    assert ex["verdict"].startswith("INCOMPLETE") and "exclude_reading" not in ex
    runs2 = stage1y_world(tmp_path / "b", drop={(BR.CHROMATIC, BR.A6F, 0)})
    res2 = BR.stage1y(runs2, None, "backoff2", root=tmp_path / "b", stage0_set=None)
    assert res2["verdict"].startswith("INCOMPLETE") and (BR.CHROMATIC, BR.A6F, 0) in res2["missing"]


def test_stage1y_m1_counts_runs_without_a_partner_post_as_na(tmp_path):
    runs = stage1y_world(tmp_path)
    i = runs.index[(runs["arm"] == BR.BOY) & (runs["instance"] == BR.SING44) & (runs["seed"] == 1)][0]
    d = Path(runs.loc[i, "run_dir"])
    for role in ("solver", "cuopt"):
        (d / f"{role}.events.jsonl").write_text("")
    m1 = BR.stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None)["mechanism"]["M1"]
    assert m1["ok"] and len(m1["rows"]) == 21 and [(r["instance"], r["seed"]) for r in m1["n_a"]] == [(BR.SING44, 1)]


def test_stage1y_cli(tmp_path):
    runs = stage1y_world(tmp_path)
    runs.to_parquet(tmp_path / "runs.parquet")
    out = tmp_path / "r.json"
    rc = BR.main(["--stage", "1y", "--runs", str(tmp_path / "runs.parquet"), "--incumbents", str(tmp_path / "none"),
                  "--root", str(tmp_path), "--json", str(out), "--stage0-set", "none"])
    assert rc == 0
    js = json.loads(out.read_text())
    assert js["stage"] == "1y" and js["verdict"].startswith("ADOPT boY") and js["kept_failed"] == []
    assert BR.main(["--stage", "1", "--failed", "both", "--runs", str(tmp_path / "runs.parquet")]) == 2
    assert BR.main(["--stage", "2", "--runs", str(tmp_path / "runs.parquet")]) == 2


def test_stage1y_harness_leg(tmp_path):
    """D-023 runs under D-022's harness outcome: the A9 arms share one harness, A6F runs under ext, and --harness checks
    the A9 harness against the outcome; a mix — seed 0 under ext, seed 1 under r13-off — is INVALID, not a verdict."""
    from itrea.experiments import make_jobs as MJ
    off = _resolved(MJ.ITX_R13_CONTROL)
    a9 = (BR.BOY, BR.CTRL1)
    # r13-off on both A9 arms, A6F ext: valid; named in the header; --harness r13-off agrees, --harness ext does not
    runs = stage1y_world(tmp_path, harness={(i, a, s): off for i in BR.STAGE1Y for a in a9 for s in (0, 1)})
    res = BR.stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None)
    assert res["harness"]["ok"] and res["harness"]["a9"] == "r13-off" and res["verdict"].startswith("ADOPT boY")
    assert any("not checked against D-022's recorded outcome" in n for n in res["notes"])
    assert BR.stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None, harness="r13-off")["verdict"] \
        .startswith("ADOPT boY")
    bad = BR.stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None, harness="ext")
    assert bad["verdict"].startswith("INVALID") and "not D-022's outcome ext" in bad["verdict"]
    with open(tmp_path / "out.txt", "w") as f:
        BR.print_stage1y(bad, out=f)
    assert "harness: A9 arms r13-off, A6F ext (D-022's outcome given: ext) — FAILS" in (tmp_path / "out.txt").read_text()
    # seed 0 under ext, seed 1 under r13-off (every A9 run): INVALID with both groups named
    mix = stage1y_world(tmp_path / "m", harness={(i, a, 1): off for i in BR.STAGE1Y for a in a9})
    r = BR.stage1y(mix, None, "backoff2", root=tmp_path / "m", stage0_set=None)
    assert r["verdict"].startswith("INVALID") and "2 harnesses" in r["verdict"] and not r["harness"]["ok"]
    assert {g["name"] for g in r["harness"]["a9_groups"]} == {"ext", "r13-off"}
    # A6F under r13-off (D-021 R13 / R6: A6F is cuOpt alone at gaps 0 in either outcome): INVALID
    a6 = stage1y_world(tmp_path / "a", harness={(BR.CA, BR.A6F, 0): off})
    r = BR.stage1y(a6, None, "backoff2", root=tmp_path / "a", stage0_set=None)
    assert r["verdict"].startswith("INVALID") and r["harness"]["a6f_not_ext"] == [dict(instance=BR.CA, seed=0,
                                                                                       harness=off)]
    # a run without a harness (none in run.json, none in the table): INVALID
    nh = stage1y_world(tmp_path / "n", harness={(BR.SING44, BR.BOY, 1): None})
    assert BR.stage1y(nh, None, "backoff2", root=tmp_path / "n", stage0_set=None)["verdict"].startswith("INVALID")
    # the run table's `harness` column stands in when run.json has none
    nh["harness"] = json.dumps(_resolved(MJ.ITX_EXT), sort_keys=True)
    assert BR.stage1y(nh, None, "backoff2", root=tmp_path / "n", stage0_set=None)["verdict"].startswith("ADOPT boY")
    # a second attempt of one key under the other harness is caught although `load` keeps only the latest attempt
    dup = stage1y_world(tmp_path / "d")
    row = dict(dup[(dup["instance"] == BR.SING44) & (dup["arm"] == BR.BOY) & (dup["seed"] == 0)].iloc[0])
    d2 = Path(row["run_dir"] + "_a1")
    d2.mkdir()
    for f in Path(row["run_dir"]).iterdir():
        (d2 / f.name).write_bytes(f.read_bytes())
    rj = json.loads((d2 / "run.json").read_text())
    rj["config"] = dict(harness=off)
    (d2 / "run.json").write_text(json.dumps(rj))
    row.update(run_dir=str(d2), attempt=1, run_id=row["run_id"] + "_a1")
    dup = pd.concat([dup, pd.DataFrame([row])], ignore_index=True)
    assert BR.stage1y(dup, None, "backoff2", root=tmp_path / "d", stage0_set=None)["verdict"].startswith("INVALID")
    # the CLI: --harness is a 1y option
    runs.to_parquet(tmp_path / "runs.parquet")
    assert BR.main(["--stage", "1y", "--harness", "r13-off", "--runs", str(tmp_path / "runs.parquet"), "--incumbents",
                    str(tmp_path / "none"), "--root", str(tmp_path), "--stage0-set", "none"]) == 0
    assert BR.main(["--stage", "1", "--harness", "ext", "--runs", str(tmp_path / "runs.parquet")]) == 2
    with pytest.raises(ValueError):
        BR.stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None, harness="nope")


def test_stage1y_reads_the_amended_set(tmp_path):
    """D-023's amendment of 2026-10-13 (D-024 rule 5), fixed before any backoff2 run: 11 instances / 66 runs; T, G,
    the ladder and the mean guard read the nine, F (one-sided) reads krka and fhnw, the mechanism rules all 11; comp21
    (ladder target), glass4 (guard) and radiationm18 (floor) are not run and print as such."""
    assert BR.STAGE1Y == ("chromaticindex1024-7", "neos-3656078-kumeu", "neos-4763324-toguru", "item_placement_9903",
                          BR.CA, "neos-3024952-loue", "neos-2657525-crna", "neos-960392", BR.SING44,
                          "neos-3004026-krka", "fhnw-binpack4-48")
    assert BR.Y_OUTCOME == BR.STAGE1Y[:9] and BR.Y_FLOOR == ("neos-3004026-krka", "fhnw-binpack4-48")
    assert set(BR.STAGE1Y_NOT_RUN) == {BR.COMP21, BR.GLASS4, "radiationm18-12-05"}
    assert set(BR.STAGE1Y) | set(BR.STAGE1Y_NOT_RUN) == set(BR.STAGE1)
    runs = stage1y_world(tmp_path)
    res = BR.stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None)
    o = res["outcome"]
    assert [r["instance"] for r in o["G"]["rows"]] == ["item_placement_9903", BR.CA, "neos-3024952-loue",
                                                       "neos-2657525-crna", BR.SING44]
    assert o["G"]["not_run"] == [BR.GLASS4]
    # the guard-set mean: D-018's 7 GUARDS (glass4 included) -> the 6 in the nine
    assert len(BR.GUARDS) == 7 and o["G"]["mean_over"] == ["item_placement_9903", BR.CA, "neos-3024952-loue",
                                                           "neos-2657525-crna", "neos-960392", BR.SING44]
    assert [r["instance"] for r in o["ladder"]["rows"]] == list(_LADDER_Y) and o["ladder"]["not_run"] == [BR.COMP21]
    assert [r["instance"] for r in o["F"]["rows"]] == list(BR.Y_FLOOR) and o["F"]["one_sided"]
    assert o["F"]["not_run"] == ["radiationm18-12-05"] and o["mean_guard"]["n"] == 9
    assert set(res["not_run"]) == set(BR.STAGE1Y_NOT_RUN)
    assert res["reports"]["d013"]["cuOpt"]["n_with_baseline"] == 9        # the nine, not the floor
    with open(tmp_path / "out.txt", "w") as f:
        BR.print_stage1y(res, out=f)
    txt = (tmp_path / "out.txt").read_text()
    for i in BR.STAGE1Y_NOT_RUN:
        assert f"not run (D-023 amendment 2026-10-13): {i}" in txt
    assert "F (one-sided, P boY − P ctrl <= +0.005" in txt and "mean guard (the nine, 9 with a d" in txt
    assert "ladder targets (a gate in 1y): hold" in txt and "not run (D-023 amendment 2026-10-13): comp21-2idx)" in txt
    assert "(F and mechanism only: its d and band are read by no rule)" in txt
    # F is one-sided: fhnw's boY better by 0.016 holds (D-018's two-sided |dP| <= 0.005 would fail it) …
    better = stage1y_world(tmp_path / "b", p_ctrl={"fhnw-binpack4-48": (0.02, 0.02)},
                           p_boy={"fhnw-binpack4-48": 0.004})
    rb = BR.stage1y(better, None, "backoff2", root=tmp_path / "b", stage0_set=None)
    assert rb["outcome"]["F"]["ok"] and rb["verdict"].startswith("ADOPT boY")
    R = {k: v for k, v in BR.load(better, "backoff2", tmp_path / "b").items() if k[0] in BR.STAGE1Y}
    eff = BR._effects(R, BR.BOY, BR.CTRL1, BR.A6F, BR.STAGE1Y)
    assert not BR._outcome(R, None, eff, BR.BOY, BR.A6F, BR.STAGE1Y)[0]["F"]["ok"]       # two-sided: fails
    # … while krka's boY worse by 0.004 holds and by 0.006 fails
    for worse, ok in ((0.009, True), (0.011, False)):
        w = stage1y_world(tmp_path / f"w{worse}", p_boy={"neos-3004026-krka": worse})
        rw = BR.stage1y(w, None, "backoff2", root=tmp_path / f"w{worse}", stage0_set=None)
        assert rw["outcome"]["F"]["ok"] is ok and failed_gates_y(rw) == (set() if ok else {"F"})
    # completeness: n of 66
    d = stage1y_world(tmp_path / "d", drop={("fhnw-binpack4-48", BR.A6F, 1)})
    rd = BR.stage1y(d, None, "backoff2", root=tmp_path / "d", stage0_set=None)
    assert rd["verdict"].startswith("INCOMPLETE: 65 of 66 runs") and rd["missing"] == [("fhnw-binpack4-48", BR.A6F, 1)]


def test_stage1y_outside_runs_are_named_and_read_by_no_rule(tmp_path):
    """Runs of instances outside the amended set (here comp21, glass4 and radiationm18, each set to fail its old rule,
    one under another harness, one failed) are named under `outside` and change nothing: no rule row, no harness
    problem, no kept failed run."""
    from itrea.experiments import make_jobs as MJ
    off = _resolved(MJ.ITX_R13_CONTROL)
    extra = (BR.COMP21, BR.GLASS4, "radiationm18-12-05")
    runs = stage1y_world(tmp_path, extra=extra, p_boy={BR.COMP21: 0.9, BR.GLASS4: 0.9, "radiationm18-12-05": 0.5},
                         harness={(BR.GLASS4, BR.BOY, 0): off}, failed={(BR.COMP21, BR.BOY, 1)})
    res = BR.run_stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None)
    assert res["verdict"].startswith("ADOPT boY") and res["kept_failed"] == [] and res["harness"]["ok"]
    assert len(res["outside"]) == 18 and {r["instance"] for r in res["outside"]} == set(extra)
    assert dict(instance=BR.COMP21, arm=BR.BOY, seed=1, status="failed") in res["outside"]
    assert set(res["effects"]) == set(BR.STAGE1Y) and not res["missing"]
    m = res["mechanism"]
    seen = {r["instance"] for k in ("M1", "M4") for r in m[k]["rows"]} | \
        {r["instance"] for r in res["mechanism_d018"]["M1"]["rows"]}
    assert seen == set(BR.STAGE1Y) and len(m["M4"]["rows"]) == 22
    o = res["outcome"]
    assert not {r["instance"] for k in ("G", "ladder", "F") for r in o[k]["rows"]} & set(extra)
    assert any("lie outside the amended set and are read by no rule" in n for n in res["notes"])
    with open(tmp_path / "out.txt", "w") as f:
        BR.print_stage1y(res, out=f)
    assert "outside the amended set, read by no rule (18 runs): comp21-2idx A6F s0 (completed)" in \
        (tmp_path / "out.txt").read_text()


def test_stage1y_still_beats_binds_a_noise_guard(tmp_path):
    """D-023 (13:5x clarification, the amendment and the 23:3x clarification): a guard under the noise conditions is
    exempt from its own band only — "still beats A6F" binds it. item_placement: the governor slept 0 s in phase Y on
    both seeds, its own band holds (seed spread 0.1 / 0.7: band 0.86), the guard-set mean holds (loue improved), but
    boY beats A6F by d 0.114 < 0.15 where the control beat it by 0.43: G fails in 1y. The same inputs under D-018
    stage 1's reading (noise test on length, "still beats" exempt too) hold."""
    ip = "item_placement_9903"
    kw = dict(p_ctrl={ip: (0.1, 0.7)}, p_boy={ip: 0.62, "neos-3024952-loue": 0.35},
              phase_y={(ip, 0): (0.3, 0.3), (ip, 1): (0.3, 0.3)})
    runs = stage1y_world(tmp_path, **kw)
    res = BR.stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None)
    g = res["outcome"]["G"]
    assert failed_gates_y(res) == {"G"} and res["verdict"] == "DO NOT ADOPT boY: failed G"
    assert next(r for r in g["rows"] if r["instance"] == ip)["ok"] and g["mean_d"] >= BR.GUARD_MEAN
    b = next(x for x in g["still_beats_A6"] if x["instance"] == ip)
    assert not b["ok"] and b["d_bo1_vs_A6"] == pytest.approx(0.08 / 0.7) and b["d_bo0_vs_A6"] >= BR.BEAT_A6
    assert g["noise"] == [] and g["noise_test"] == "slept" and g["still_beats_noise_exempt"] is False
    with open(tmp_path / "out.txt", "w") as f:
        BR.print_stage1y(res, out=f)
    assert "still beating A6F (binds every guard)" in (tmp_path / "out.txt").read_text()
    # D-018 stage 1's reading of the same runs: item_placement is noise and its "still beats" failure is exempt
    R = {k: v for k, v in BR.load(runs, "backoff2", tmp_path).items() if k[0] in BR.STAGE1Y}
    eff = BR._effects(R, BR.BOY, BR.CTRL1, BR.A6F, BR.STAGE1Y)
    g18 = BR._outcome(R, None, eff, BR.BOY, BR.A6F, BR.STAGE1Y, f_one_sided=True)[0]["G"]
    assert g18["ok"] and g18["noise"] == [ip] and "noise_test" not in g18
    with pytest.raises(ValueError):
        BR._outcome(R, None, eff, BR.BOY, BR.A6F, BR.STAGE1Y, noise_test="nope")


def test_stage1y_noise_test_is_no_sleep_in_phase_y(tmp_path):
    """1y's noise test (D-023 clarification 2026-10-13 23:3x): the governor slept 0 s in phase Y on every boY run, not
    D-018's "phase Y < 1 s". crna fails its own band (boY 0.3 vs 0.1, band 0.05) while still beating A6F (0.4): slept
    0.25 s inside a 0.8-s phase Y on both seeds -> not noise, G fails (D-018's length test would exempt it); slept 0 s
    in the same 0.8 s -> noise, G holds."""
    crna = "neos-2657525-crna"
    base = dict(p_ctrl={crna: (0.1, 0.1)}, p_boy={crna: 0.3, "neos-3024952-loue": 0.1})
    slept = stage1y_world(tmp_path, **base, phase_y={(crna, 0): (0.8, 0.55), (crna, 1): (0.8, 0.55)})
    res = BR.stage1y(slept, None, "backoff2", root=tmp_path, stage0_set=None)
    g = res["outcome"]["G"]
    assert failed_gates_y(res) == {"G"} and g["noise"] == [] and g["mean_d"] >= BR.GUARD_MEAN
    assert not next(r for r in g["rows"] if r["instance"] == crna)["ok"]
    assert next(b for b in g["still_beats_A6"] if b["instance"] == crna)["ok"]
    R = {k: v for k, v in BR.load(slept, "backoff2", tmp_path).items() if k[0] in BR.STAGE1Y}
    assert BR._y_slept(R[(crna, BR.BOY, 0)]) == pytest.approx(0.25)
    eff = BR._effects(R, BR.BOY, BR.CTRL1, BR.A6F, BR.STAGE1Y)
    g18 = BR._outcome(R, None, eff, BR.BOY, BR.A6F, BR.STAGE1Y, f_one_sided=True)[0]["G"]
    assert g18["ok"] and g18["noise"] == [crna]                       # D-018's length test: 0.8 s < 1 s, exempt
    awake = stage1y_world(tmp_path / "a", **base, phase_y={(crna, 0): (0.8, 0.8), (crna, 1): (0.8, 0.8)})
    ra = BR.stage1y(awake, None, "backoff2", root=tmp_path / "a", stage0_set=None)
    assert not failed_gates_y(ra) and ra["outcome"]["G"]["noise"] == [crna] and ra["verdict"].startswith("ADOPT boY")
    # a phase Y that never ended is not noise (no exit record to read the sleep from)
    rv = BR.RunView(dict(), tmp_path)
    rv._cache[("summary", "ea")] = dict(backoff=dict(phase_y=dict(length_s=120.0, exit_t=None, reason="never")))
    assert math.isnan(BR._y_slept(rv))


def test_stage1y_other_seeds_are_outside(tmp_path):
    """Runs of the 11 at a seed other than 0-1 are named under `outside` and read by no rule (D-023 clarification
    2026-10-13 23:3x): ca at seed 2 with boY's phase Y never yielding (100 s, M1 and M4 would fail), its control under
    r13-off (the harness leg would fail) and its A6F failed (no kept failed run) change nothing."""
    from itrea.experiments import make_jobs as MJ
    off = _resolved(MJ.ITX_R13_CONTROL)
    runs = stage1y_world(tmp_path, extra_seeds={BR.CA: (2,)}, phase_y={(BR.CA, 2): (100.0, 100.0)},
                         harness={(BR.CA, BR.CTRL1, 2): off}, failed={(BR.CA, BR.A6F, 2)})
    res = BR.run_stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None)
    assert res["verdict"].startswith("ADOPT boY") and res["kept_failed"] == [] and res["harness"]["ok"]
    assert {(r["instance"], r["arm"], r["seed"], r["status"]) for r in res["outside"]} == {
        (BR.CA, BR.A6F, 2, "failed"), (BR.CA, BR.CTRL1, 2, "completed"), (BR.CA, BR.BOY, 2, "completed")}
    m = res["mechanism"]
    assert len(m["M4"]["rows"]) == 22 and len(m["M1"]["rows"]) == 22 and m["M4"]["busy_share"] == pytest.approx(0.5)
    assert res["expected"] == 66 and not res["missing"]


def test_stage1y_f_one_sided_at_exactly_the_limit(tmp_path):
    """F's one-sided limit compares with REL_TOL: krka's boY worse by exactly 0.005 (0.012 -> 0.017, a float
    difference of -0.005000000000000001) holds; by 0.0051 it fails."""
    for p_boy, ok in ((0.017, True), (0.0171, False)):
        w = stage1y_world(tmp_path / str(p_boy), p_ctrl={"neos-3004026-krka": (0.012, 0.012)},
                          p_boy={"neos-3004026-krka": p_boy})
        r = BR.stage1y(w, None, "backoff2", root=tmp_path / str(p_boy), stage0_set=None)
        assert r["outcome"]["F"]["ok"] is ok and failed_gates_y(r) == (set() if ok else {"F"})
    assert 0.012 - 0.017 < -BR.FLOOR_DP                                       # the float edge the tolerance absorbs


def test_stage1y_t_failure_inside_its_band_is_not_adopted(tmp_path):
    """D-023: a T failure inside chromaticindex's band is NOT ADOPTED (clarification 2026-10-13 23:3x); D-018's
    "inside the band: seeds 3 and 4" class is printed as information, not as a pending verdict. Control 0.3 / 0.8
    (band 0.625), boY 0.6: d −0.083; kumeu improved so the mean over the nine holds."""
    runs = stage1y_world(tmp_path, p_ctrl={BR.CHROMATIC: (0.3, 0.8)},
                         p_boy={BR.CHROMATIC: 0.6, "neos-3656078-kumeu": 0.2})
    res = BR.stage1y(runs, None, "backoff2", root=tmp_path, stage0_set=None)
    assert failed_gates_y(res) == {"T"} and res["verdict"] == "DO NOT ADOPT boY: failed T"
    inside = [c for c in res["classification"] if c.startswith("inside the band")]
    assert len(inside) == 1 and BR.CHROMATIC in inside[0] and "NOT ADOPTED" in inside[0]
    assert "not pre-registered" in inside[0]


# ======================================================================================== D-023 config and jobs
def test_boy_config_file():
    """configs/itrea2-dev-rc-3way-boY.yaml (D-023) = bo1's back-off values in active mode with the four patiences at
    100000 s (the ladder never steps down), on the A9 development default; its hash differs from bo1's."""
    from itrea.config import EAConfig
    root = ROOT / "configs"
    boy = EAConfig.from_yaml(root / "itrea2-dev-rc-3way-boY.yaml")
    bo1 = EAConfig.from_yaml(root / "itrea2-dev-rc-3way-bo1.yaml")
    assert boy.backoff.enabled and boy.backoff.mode == "active" and not boy.exchange.from_cuopt
    pat = ("patience_search_s", "patience_own_s", "patience_min_s", "patience_max_s")
    assert all(getattr(boy.backoff, k) == 100000.0 for k in pat)
    a, b = boy.to_dict()["backoff"], bo1.to_dict()["backoff"]
    assert {k for k in a if a[k] != b[k]} == set(pat)
    assert (boy.backoff.yield_level, boy.backoff.levels, boy.backoff.yield_until_s) == (1, 3, 30.0)
    assert boy.hash() != bo1.hash()
    assert boy.with_overrides({"backoff.enabled": False}).hash() == "1d12ad490dd0"     # the A9 development default
    d0, d1 = boy.to_dict(), EAConfig.from_yaml(root / "itrea2-dev-rc-3way.yaml").to_dict()
    for k in ("backoff", "name"):
        d0.pop(k), d1.pop(k)
    assert d0 == d1


def test_backoff2_jobs(tmp_path):
    """D-023's job file, as amended on 2026-10-13: the 11 instances of STAGE1Y x seeds x {A9@boY, A9@bo1s, A6F},
    120 s, set backoff2, one block per instance x seed (66 runs), the harness of D-022's outcome on both A9 arms
    (required: no default) and ext on A6F in either outcome (D-021 R13 / R6; D-022's reject clause covers only A9's
    rows); `instances` "stage1" = D-018 stage 1's 14 (84), or named instances."""
    from itrea.experiments import make_jobs as MJ
    from itrea.experiments import pair
    with pytest.raises(MJ.JobError, match="harness"):
        MJ.backoff2(seeds=(0,))
    jobs = MJ.backoff2(seeds=(0, 1), harness="ext")
    assert len(jobs) == len(BR.STAGE1Y) * 2 * 3 == 66
    assert {pair.arm_label(j) for j in jobs} == {BR.BOY, BR.CTRL1, BR.A6F}
    assert all(j["budget"] == 120.0 and j["set"] == "backoff2" and j["split"] == "dev" and j["harness"] == "ext"
               for j in jobs)
    blocks: dict = {}
    for j in jobs:
        blocks.setdefault(j["block"], set()).add(pair.arm_label(j))
    assert len(blocks) == 22 and all(len(v) == 3 for v in blocks.values())
    assert {j["name"] for j in jobs} == set(BR.STAGE1Y)
    assert MJ.backoff2(seeds=(0, 1), harness="ext", instances="amended") == jobs
    old = MJ.backoff2(seeds=(0, 1), harness="ext", instances="stage1")
    assert len(old) == 84 and {j["name"] for j in old} == set(BR.STAGE1)
    assert all(j in old for j in jobs)                       # the same 66 jobs (so run ids) as in the 84
    two = MJ.backoff2(seeds=(0,), harness="ext", instances=[BR.SING44, BR.CA, BR.SING44])
    assert len(two) == 6 and {j["name"] for j in two} == {BR.SING44, BR.CA}
    with pytest.raises(MJ.JobError, match="not in"):
        MJ.backoff2(seeds=(0,), harness="ext", instances=["nothere"])
    with pytest.raises(MJ.JobError):
        MJ.backoff2(seeds=(0,), harness="ext", instances="nine")
    for j in jobs:
        if j["arm"] == "A9":
            cfg, ov = pair.effective_config(j)
            assert cfg.exchange.from_cuopt and cfg.backoff.enabled and (ROOT / j["config"]).exists()
            assert cfg.backoff.mode == ("active" if j["variant"] == "boY" else "shadow")
        else:
            assert j["config"] is None and "variant" not in j
    off = MJ.backoff2(seeds=(0,), harness="r13-off")
    assert len(off) == 33
    assert all(j["harness"] == (MJ.ITX_R13_CONTROL if j["arm"] == "A9" else "ext") for j in off)
    a9_off = next(j for j in off if j["arm"] == "A9")
    a6_off = next(j for j in off if j["arm"] == "A6F")
    assert pair.resolve_harness(a9_off["harness"])["cuopt_gap_zero"] is False
    assert pair.resolve_harness(a6_off["harness"]) == pair.resolve_harness("ext")       # gaps 0, env unset
    # no run id of backoff2 is one of D-018 stage 1's (the harness enters the hash); the A9 ids differ between the
    # harness outcomes, A6F's are the same run under both
    b1 = {pair.run_id_for(j) for j in MJ.backoff_stage1(seeds=(0,))}
    ext0 = {pair.run_id_for(j) for j in jobs if j["seed"] == 0}
    assert not b1 & ext0 and len(ext0) == 33
    ids = {h: {(j["arm"], pair.run_id_for(j)) for j in js if j["seed"] == 0} for h, js in (("ext", jobs), ("off", off))}
    a6 = {h: {i for a, i in v if a == "A6F"} for h, v in ids.items()}
    a9 = {h: {i for a, i in v if a == "A9"} for h, v in ids.items()}
    assert a6["ext"] == a6["off"] and len(a6["ext"]) == 11 and not a9["ext"] & a9["off"]
    with pytest.raises(MJ.JobError):
        MJ.backoff2(seeds=(0,), harness="nope")
    # the CLI: --harness required; a file in a temporary directory, never jobs/
    out = tmp_path / "b2.jsonl"
    assert MJ.main(["backoff2", "--harness", "ext", "--seeds", "1", "--out", str(out)]) == 0
    on_disk = [json.loads(line) for line in out.read_text().splitlines()]
    assert on_disk == MJ.backoff2(seeds=(1,), harness="ext")
    with pytest.raises(SystemExit):
        MJ.main(["backoff2", "--seeds", "0", "--out", str(tmp_path / "x.jsonl")])
    assert not (tmp_path / "x.jsonl").exists()
    assert MJ.main(["backoff2", "--harness", "ext", "--instances", "stage1", "--out", str(out)]) == 0
    assert len(out.read_text().splitlines()) == 84
    assert MJ.main(["backoff2", "--harness", "ext", "--instances", f"{BR.CA},{BR.SING44}", "--seeds", "0", "--out",
                    str(out)]) == 0
    assert len(out.read_text().splitlines()) == 6
    assert MJ.main(["backoff2", "--harness", "ext", "--instances", "nothere", "--out", str(out)]) == 2


def test_backoff2_job_file_is_the_writers_output(tmp_path):
    """jobs/backoff2.jsonl (D-023 as amended on 2026-10-13, D-022's outcome ext) is exactly `make-jobs backoff2
    --harness ext`: 66 runs, every A9 configuration file present."""
    from itrea.experiments import make_jobs as MJ
    out = tmp_path / "b2.jsonl"
    assert MJ.main(["backoff2", "--harness", "ext", "--out", str(out)]) == 0
    assert out.read_bytes() == (ROOT / "jobs" / "backoff2.jsonl").read_bytes()
    jobs = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(jobs) == 66 and all((ROOT / j["config"]).is_file() for j in jobs if j["arm"] == "A9")
