"""`itrea backoff-report` (`itrea.analysis.backoff`, D-018): the stage-0 rules (sanity, D_Y with its stop rule, the
floor with its shortfall, yield_until_s and both patiences) and the stage-1 criteria (M1–M5, T, G with the noise rule, the
ladder, F, the mean guard, the ca fallback classes), on synthetic run tables and run directories — no solver, no EA,
no GPU. Also the shared-prefix stretch, the stage-0 job file and the `bo_*` columns of build_runs."""
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
             gpu_util=80.0, summary_extra=None, budget=120.0, run_extra=None, best_obj=math.nan, sense=1) -> dict:
    rid = f"{inst}__{arm}__s{seed}__{budget:g}s__synthetic"
    d = root / set_ / rid
    d.mkdir(parents=True, exist_ok=True)
    rj = dict(run_id=rid, gpu_health=dict(verdict="ok", mean_util=gpu_util), stop=None)
    rj.update(run_extra or {})
    (d / "run.json").write_text(json.dumps(rj))
    if arm != BR.A6:
        s = dict(exit_reason="budget", wall_s=budget, ls_calls=100, ls_cap_hits=0)
        if bo is not None:
            s["backoff"] = bo
        s.update(summary_extra or {})
        (d / "ea.summary.json").write_text(json.dumps(s))
        _write_events(d / "ea.events.jsonl", ea_events)
    _write_events(d / "solver.events.jsonl", solver_events)
    if cuopt_events is None:
        cuopt_events = ([dict(t=cuopt_t_first, kind="incumbent", obj=1.0), dict(t=cuopt_t_first, kind="send", k=1)]
                        if math.isfinite(cuopt_t_first) else [])
    _write_events(d / "cuopt.events.jsonl", cuopt_events)
    return dict(run_id=rid, set=set_, instance=inst, arm=arm, seed=seed, budget=budget, status="completed",
                block_void=False, synthetic=False, run_dir=str(d), pt=pt, cuopt_pt=pt if cuopt_pt is None else cuopt_pt,
                cuopt_t_first=cuopt_t_first, n_inc_unchecked=0, n_inc_failed=0, gpu_health="ok", gpu_void=False,
                attempt=0, best_obj=best_obj, sense=sense)


def bo_block(mode="shadow", busy_share=1.0, ls_steps=120_000.0, wall=120.0, first_lead=None, own=None,
             exit_t=0.4, reason="peek", levels=2, **kw) -> dict:
    b = dict(mode=mode, busy_share=busy_share, ls_steps=ls_steps, wall_s=wall, busy_s=busy_share * wall,
             phase_y=dict(length_s=exit_t, exit_t=exit_t, reason=reason), levels=levels, yield_level=2,
             first_lead=None if first_lead is None else dict(t=first_lead, busy_s=first_lead),
             own_feasible=None if own is None else dict(t=own, busy_s=own), max_launch_s=0.05, patience_own_s=30.0)
    b.update(kw)
    return b


def traj_rows(run_id: str, pts) -> list[dict]:
    return [dict(run_id=run_id, role="cuopt", t=float(t), obj_recomputed=float(v), ok=True) for t, v in pts]


# ================================================================================================ stage 0
A6_SING = [(0.5, 100.0), (2.0, 90.0), (4.0, 80.0), (6.0, 70.0), (8.0, 60.0), (10.0, 50.0), (12.0, 40.0)]


def stage0_world(tmp_path, *, s_first=None, stretch=None, comp_p=None, duty_err=None, ca_lead=50.0):
    """A6 / bo0 / bof50 / bof25 / bof12 on the three duty instances, ca and glass4 bo0, and two older A6 comp21 runs
    (the spread); returns (runs, incumbents)."""
    s_first = s_first or {"A9@bo0": 7.5, "A9@bof50": 3.75, "A9@bof25": 1.875, "A9@bof12": 1.25}
    stretch = stretch or {"A9@bo0": 1.5, "A9@bof50": 1.3, "A9@bof25": 1.10, "A9@bof12": 1.05}
    comp_p = comp_p or {"A9@bo0": 0.70, "A9@bof50": 0.60, "A9@bof25": 0.53, "A9@bof12": 0.51}
    duty_err = duty_err or {}
    rows, inc = [], []
    S = "backoff0"
    t_a6 = 8.0
    rows.append(make_run(tmp_path, S, BR.CHROMATIC, BR.A6, 2, 0.07, cuopt_t_first=t_a6))
    a6s = make_run(tmp_path, S, BR.SING44, BR.A6, 2, 0.70, cuopt_t_first=0.5)
    rows.append(a6s)
    inc += traj_rows(a6s["run_id"], A6_SING)
    rows.append(make_run(tmp_path, S, BR.COMP21, BR.A6, 2, 0.50, cuopt_t_first=1.0))
    for seed, p in ((0, 0.49), (1, 0.53)):                                     # the classprobe A6 seeds: spread 0.04
        rows.append(make_run(tmp_path, "classprobe", BR.COMP21, BR.A6, seed, p, cuopt_t_first=1.0))
    for arm, d in [("A9@bo0", 1.0)] + list(BR.FORCED.items()):
        for inst in (BR.CHROMATIC, BR.SING44, BR.COMP21):
            duty = d + duty_err.get((arm, inst), 0.0)
            b = bo_block(mode="shadow" if d == 1.0 else "forced", busy_share=duty, ls_steps=120_000.0 * d)
            kw = dict(bo=b, cuopt_t_first=s_first[arm] * t_a6 if inst == BR.CHROMATIC else 0.6)
            if inst == BR.COMP21:
                kw["cuopt_pt"] = comp_p[arm]
            r = make_run(tmp_path, S, inst, arm, 2, 0.5, **kw)
            rows.append(r)
            if inst == BR.SING44:
                inc += traj_rows(r["run_id"], [(t * stretch[arm], v) for t, v in A6_SING])
    ca = bo_block(first_lead=ca_lead)
    rows.append(make_run(tmp_path, S, BR.CA, "A9@bo0", 2, 0.3, bo=ca, cuopt_t_first=25.0,
                         cuopt_events=[dict(t=24.0, kind="incumbent", obj=5.0), dict(t=25.0, kind="send", k=1)]))
    rows.append(make_run(tmp_path, S, BR.GLASS4, "A9@bo0", 2, 0.3, bo=bo_block(first_lead=47.7, own=10.0),
                         cuopt_t_first=0.1))
    return pd.DataFrame(rows), pd.DataFrame(inc)


def test_stretch_definition():
    ref = [(0.5, 10.0), (1.0, 9.0), (2.0, 8.0), (3.0, 7.0), (4.0, 6.0), (5.0, 5.0), (6.0, 4.0)]
    arm = [(t * 2.0, v) for t, v in ref]
    assert BR.stretch(arm, ref) == (2.0, 6)                     # t_ref = 0.5 s is left out
    assert BR.stretch(arm[:4], ref) == (None, 3)                 # fewer than 5 shared values: n/a
    assert BR.stretch([(1.0, 9.0 + 1e-3)], ref)[1] == 0          # values must match (relative 1e-9)


def test_stage0_proceeds_with_the_rules_values(tmp_path):
    runs, inc = stage0_world(tmp_path)
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path)
    ru = res["rules"]
    assert not res["missing"] and all(v["ok"] for v in res["sanity"].values())
    assert ru["a"]["D_Y"] == 0.25 and ru["a"]["qualifying"] == [0.25, 0.125]
    assert res["readouts"]["A9@bof25"]["sing44_S"] == pytest.approx(1.10)
    assert ru["b"]["D_min"] == 0.25 and ru["b"]["L"] == 2 and not ru["b"]["shortfall"]
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
                             stretch={"A9@bo0": 1.5, "A9@bof50": 1.4, "A9@bof25": 1.3, "A9@bof12": 1.2},
                             ca_lead=math.nan)
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path)
    assert res["rules"]["a"]["D_Y"] is None and res["verdict"].startswith("STOP") and res["bo1"] is None
    assert res["rules"]["b"]["shortfall"] and res["rules"]["b"]["D_min"] == 0.125 and res["rules"]["b"]["L"] == 3
    assert res["rules"]["d"]["patience_search_s"] == 40           # no material lead on ca


def test_stage0_a_run_that_fails_sanity_does_not_qualify(tmp_path):
    runs, inc = stage0_world(tmp_path, duty_err={("A9@bof25", BR.CHROMATIC): 0.10})
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path)
    assert not res["sanity"][f"A9@bof25 {BR.CHROMATIC}"]["ok"]
    assert res["rules"]["a"]["D_Y"] == 0.125 and 0.25 in res["rules"]["a"]["not_counted"]
    assert res["bo1"]["yield_level"] == 3 and res["bo1"]["yield_until_s"] == 30.0


def test_stage0_steps_sanity_and_missing_runs(tmp_path):
    runs, inc = stage0_world(tmp_path)
    i = runs.index[(runs["arm"] == "A9@bof50") & (runs["instance"] == BR.SING44)][0]
    p = Path(runs.loc[i, "run_dir"]) / "ea.summary.json"
    s = json.loads(p.read_text())
    s["backoff"]["ls_steps"] = 120_000.0                           # duty 1/2 but the work of full duty
    p.write_text(json.dumps(s))
    res = BR.stage0(runs, inc, "backoff0", root=tmp_path)
    assert not res["sanity"][f"A9@bof50 {BR.SING44}"]["steps_ok"] and 0.5 in res["rules"]["b"]["not_counted"]
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


def stage1_world(tmp_path, *, p_bo1=None, ca_escalates=False, ca_phase_y=0.4, glass4_noise=False):
    """14 instances x {A6, A9@bo0, A9@bo1} x seeds 0-1 where every criterion holds unless changed."""
    S = "backoff1"
    rows = []
    base = {i: 0.4 for i in BR.STAGE1}
    base.update({BR.CHROMATIC: 0.55, "neos-960392": 0.03, "neos-3004026-krka": 0.005, "fhnw-binpack4-48": 0.004,
                 "radiationm18-12-05": 0.01})
    p_bo1 = p_bo1 or {}
    for i in BR.STAGE1:
        for s in BR.STAGE1_SEEDS:
            p0 = base[i]
            p1 = p_bo1.get(i, 0.11 if i == BR.CHROMATIC else p0)
            a6 = min(1.0, p0 + 0.3) if i not in BR.FLOOR else p0
            send = [dict(t=0.3, kind="incumbent", obj=1.0), dict(t=0.3, kind="send", k=1)]
            rows.append(make_run(tmp_path, S, i, BR.A6, s, a6, cuopt_t_first=8.0 if i == BR.CHROMATIC else 0.3,
                                 cuopt_events=send, gpu_util=75.0, best_obj=10.0))
            leads = [5.0, 12.0, 20.0] if i in BR.M3_SET else []
            if i == BR.CA:
                leads = [40.0]
            contrib = [dict(t=t, kind="contrib", what="lead", material=True, dg=0.1) for t in leads]
            ctrl_gen = _gen_events(60.0, 120.0, 1000.0)
            b0 = bo_block(first_lead=leads[0] if leads else None, exit_t=0.4)
            esc0 = [dict(t=0.0, kind="backoff", phase="Y", level=2, reason="start"),
                    dict(t=0.4, kind="backoff", phase="P", level=0, reason="y_peek")]
            rows.append(make_run(tmp_path, S, i, BR.BO0, s, p0, cuopt_t_first=60.0 if i == BR.CHROMATIC else 0.3,
                                 cuopt_events=send, bo=b0, ea_events=esc0 + contrib + ctrl_gen, gpu_util=95.0,
                                 best_obj=10.0))
            m4 = i in BR.M4_SET
            gen1 = _gen_events(30.0, 45.0, 300.0) if m4 else ctrl_gen
            yl = ca_phase_y if i == BR.CA else (0.4 if not (glass4_noise and i == BR.GLASS4) else 0.4)
            ev1 = [dict(t=0.0, kind="backoff", phase="Y", level=2, reason="start"),
                   dict(t=yl, kind="backoff", phase="P", level=0, reason="y_peek")]
            if i == BR.CA and ca_escalates:
                ev1.append(dict(t=30.0, kind="backoff", phase="P", level=1, reason="escalate", busy_s=30.0))
                leads1 = [60.0]
            else:
                leads1 = leads
            if m4:
                ev1.append(dict(t=50.0, kind="backoff", phase="P", level=1, reason="escalate", busy_s=40.0))
                ev1.append(dict(t=60.0, kind="backoff", phase="P", level=2, reason="escalate", busy_s=44.0))
            c1 = [dict(t=t, kind="contrib", what="lead", material=True, dg=0.1) for t in leads1]
            b1 = bo_block(mode="active", busy_share=45.0 / 120.0 if m4 else 1.0,
                          ls_steps=300.0 * 120.0 if m4 else 120_000.0,          # as the gen events add up
                          first_lead=leads1[0] if leads1 else None, exit_t=yl)
            rows.append(make_run(tmp_path, S, i, BR.BO1, s, p1, cuopt_t_first=12.0 if i == BR.CHROMATIC else 0.3,
                                 cuopt_events=send, bo=b1, ea_events=ev1 + c1 + gen1 + _rclock(), gpu_util=60.0,
                                 best_obj=10.0))
    return pd.DataFrame(rows)


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


# ======================================================================================== jobs and columns
def test_stage0_job_file():
    from itrea.experiments import make_jobs, pair
    jobs = make_jobs.backoff_stage0()
    assert len(jobs) == 17
    labels = {(j["name"], pair.arm_label(j)) for j in jobs}
    for n in BR.CHROMATIC, BR.SING44, BR.COMP21:
        assert {(n, a) for a in ("A6", "A9@bo0", "A9@bof50", "A9@bof25", "A9@bof12")} <= labels
    assert {(BR.CA, "A9@bo0"), (BR.GLASS4, "A9@bo0")} <= labels
    assert {j["seed"] for j in jobs} == {2} and {j["budget"] for j in jobs} == {120.0}
    assert all(j["config"] is None for j in jobs if j["arm"] == "A6")
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
