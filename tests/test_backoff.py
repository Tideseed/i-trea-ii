"""The EA's GPU back-off beside cuOpt (D-018; `itrea.ea.backoff`, `BackoffCfg`): the governor on a fake clock (long-run
duty per level, escalation times per state, the own-feasible reset, the κ/λ̂ clamp, phase-Y exit and expiry — dated at
yield_until_s however late it is detected —, one forgiven wake per wake_min_s in active mode and none in forced mode
(its dose stays fixed under partner traffic), steps due before a lead / own-feasible reset logged at their due time,
the max_sleep cap, forced and shadow modes), the configuration (hash out while off, validation), the mailboxes'
non-consuming `peek`, `FJCpu.run(pace=...)`, and the engine on NumPy: inert while off or without `exchange.from_cuopt`
(no governor built; equal to the recorded references), shadow mode with the genealogy of the run without the governor,
and active / forced mode under scripted cuOpt traffic (sleeps, a forgiven wake ends a between-generation sleep and a
non-forgiven one does not, the stop file within one slice, the deadline never overslept, the repair clock's gen_s /
o_s without the pauses and its decisions on busy time, leads and assists credited once per improvement, an unchanged
cuOpt row never an assist).

Toy models only; NumPy backend; nothing on the GPU (the GPU pace tests are in tests/test_fj_gpu.py, marker `gpu`)."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

import _engine_reference as REF
from itrea.benchmarks import toy
from itrea.config import BackoffCfg, EAConfig
from itrea.ea.backoff import Backoff, gap_unit
from itrea.exchange.mailbox import Message
from itrea.solvers.common import ORIGIN_SOLVER, POST_CUOPT
from itrea.trace import Trace, read_events
from test_engine import cfg_for, knapsack_optimum_x, prepared
from test_three_way import _CuoptLPMailbox, _TrafficCuoptMailbox

ROOT = Path(__file__).resolve().parents[1]


def cfg(**kw) -> BackoffCfg:
    return BackoffCfg(**dict(dict(enabled=True), **kw))


class Sim:
    """A fake clock: the EA works in launches of `work` busy seconds; at every pace point it sleeps what is due, in
    slices of slice_s (as the engine does)."""

    def __init__(self, c: BackoffCfg, t0: float = 1000.0):
        self.c = c
        self.now = t0
        self.bo = Backoff(c, t0, clock=lambda: self.now)
        self.recs: list[dict] = []

    def pace(self) -> float:
        self.recs += self.bo.update(self.now)
        s = self.bo.sleep_due(self.now)
        slept = 0.0
        while s - slept > 1e-12:
            sl = min(self.c.slice_s, s - slept)
            self.now += sl
            self.bo.slept(sl, "call")
            slept += sl
        return slept

    def work(self, seconds: float, step: float = 0.05) -> None:
        end = self.bo.busy(self.now) + seconds
        while self.bo.busy(self.now) < end - 1e-12:
            self.now += min(step, end - self.bo.busy(self.now))
            self.pace()

    def wall(self) -> float:
        return self.now - self.bo.t0


# ------------------------------------------------------------------------------------------------- the rule
def test_gap_unit():
    assert gap_unit(math.inf, 5.0) == 1.0
    assert gap_unit(10.0, 9.0) == pytest.approx(0.1)
    assert gap_unit(10.0, 9.0, dual_key=-100.0) == pytest.approx(0.01)
    assert gap_unit(-1.0, -3.0) == pytest.approx(2.0 / 3.0)
    assert gap_unit(1.0, -1.0) == 1.0                       # opposite signs: capped at 1
    assert gap_unit(5.0, 6.0) < 0                            # worse


@pytest.mark.parametrize("d", [1.0, 0.5, 0.25, 0.125])
def test_long_run_duty_forced(d):
    s = Sim(cfg(mode="forced", forced_duty=d))
    s.work(60.0)
    w = s.wall()
    assert s.bo.busy(s.now) / w == pytest.approx(d, abs=0.02)
    assert s.bo.phase == "forced" and s.bo.level is None and not s.recs
    if d < 1:
        assert s.bo.paused_site["call"] > 0
    else:
        assert s.bo.paused == 0.0


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_long_run_duty_in_phase_y_per_level(level):
    s = Sim(cfg(mode="active", yield_level=level, levels=3, yield_until_s=1e6))
    s.work(40.0)
    assert s.bo.in_phase_y and s.bo.level == level
    assert s.bo.busy(s.now) / s.wall() == pytest.approx(2.0 ** -level, abs=0.02)


@pytest.mark.parametrize("d, run, sleep", [(0.25, 0.333, 1.0), (0.125, 0.286, 2.0)])
def test_bucket_cycle_lengths(d, run, sleep):
    """At 1/4 about 0.33 s of work and 1 s asleep; at 1/8 0.29 s and 2 s (burst_s = 0.25)."""
    s = Sim(cfg(mode="forced", forced_duty=d))
    t = 0.0
    while s.bo.sleep_due(s.now) == 0.0:
        s.now += 0.001
        t += 0.001
    assert t == pytest.approx(run, abs=0.005)
    assert s.bo.sleep_due(s.now) == pytest.approx(sleep, rel=0.02)


def test_max_sleep_cap():
    s = Sim(cfg(mode="forced", forced_duty=0.25))
    s.now += 10.0                                           # ten busy seconds without a pace point
    assert s.bo.sleep_due(s.now) == 4.0                    # min(7.5 / 0.25, max_sleep_s)
    s2 = Sim(cfg(mode="forced", forced_duty=0.25, max_sleep_s=50.0))
    s2.now += 10.0
    assert s2.bo.sleep_due(s2.now) == pytest.approx(30.0)


def test_phase_y_exit_and_expiry():
    s = Sim(cfg(yield_level=2, yield_until_s=30.0))
    s.work(2.0)
    assert s.bo.in_phase_y and s.bo.paused > 0
    r = s.bo.phase_y_exit(s.now, "peek")
    assert r["reason"] == "y_peek" and r["phase"] == "P" and r["level"] == 0 and s.bo.y_exit["reason"] == "peek"
    assert s.bo.sleep_due(s.now) == 0.0                    # full duty at once
    assert s.bo.phase_y_exit(s.now + 1, "tick") is None    # never twice
    # expiry: on WALL time since t0, with the sleeps counted
    e = Sim(cfg(yield_level=3, yield_until_s=30.0, patience_search_s=1e3))
    e.work(3.0)
    assert e.bo.in_phase_y
    while e.bo.in_phase_y:
        e.work(0.05)
    assert e.bo.y_exit["reason"] == "expiry" and e.bo.y_exit["t"] == pytest.approx(30.0, abs=0.06)
    assert e.bo.busy(e.now) < 10.0                         # mostly asleep at 1/8
    assert any(x["reason"] == "y_expiry" for x in e.recs)
    e.work(5.0)
    assert e.bo.paused == pytest.approx(e.bo.paused_site["call"]) and e.bo.level == 0


def _escalations(recs):
    return [(r["level"], r["busy_s"]) for r in recs if r["reason"] == "escalate"]


def test_escalation_search_state():
    """No own feasible member: first step after patience_search_s busy seconds from the phase-P start, then step_s per
    level, down to L; the dated busy times do not depend on how often update runs."""
    for step in (0.05, 7.0):                                # an update per launch, or one per 7-s generation
        s = Sim(cfg(levels=3, patience_search_s=40.0, step_s=4.0))
        s.bo.phase_y_exit(s.now, "tick")
        b0 = s.bo.busy(s.now)
        s.work(80.0, step=step)
        esc = _escalations(s.recs)
        assert [lv for lv, _ in esc] == [1, 2, 3]
        assert [b - b0 for _, b in esc] == pytest.approx([40.0, 44.0, 48.0], abs=1e-9)
        assert s.bo.level == 3 and s.bo.duty == 0.125


def test_escalation_own_state_and_reset():
    """own_feasible restarts the patience (search -> own: patience_own_s from there), also at a lower level."""
    s = Sim(cfg(levels=2, patience_search_s=75.0, patience_own_s=30.0, step_s=4.0))
    s.bo.phase_y_exit(s.now, "tick")
    b0 = s.bo.busy(s.now)
    s.work(50.0)
    r = s.bo.own_feasible(s.now)[-1]
    assert r["reason"] == "own_feasible" and r["state"] == "own" and s.bo.state == "own"
    assert s.bo.own_feasible(s.now + 1) is None                       # first transition only
    s.work(40.0)
    esc = _escalations(s.recs)
    assert esc[0][1] - b0 == pytest.approx(80.0, abs=0.06)             # 50 + 30, not 75
    # at a lower level: the reset gives step_s from the transition
    t = Sim(cfg(levels=3, patience_search_s=10.0, step_s=4.0))
    t.bo.phase_y_exit(t.now, "tick")
    t.work(11.0)
    assert t.bo.level == 1
    bo = t.bo.busy(t.now)
    t.bo.own_feasible(t.now)
    t.work(5.0)
    esc = _escalations(t.recs)
    assert esc[1][0] == 2 and esc[1][1] == pytest.approx(bo + 4.0, abs=1e-9)


def test_lead_and_assist_restore_full_duty():
    s = Sim(cfg(levels=2, patience_search_s=5.0, step_s=2.0))
    s.bo.phase_y_exit(s.now, "tick")
    s.work(10.0)
    assert s.bo.level == 2
    m, r = s.bo.contribution(s.now, 5e-4, "lead", ref=1.0)              # below materiality: nothing
    assert not m and r == [] and s.bo.level == 2 and s.bo.state == "search"
    m, (r,) = s.bo.contribution(s.now, 0.01, "lead", ref=1.0)
    assert m and r["reason"] == "lead" and r["level"] == 0 and r["state"] == "led" and s.bo.sleep_due(s.now) == 0
    assert s.bo.first_lead is not None and s.bo.own is not None
    s.work(60.0)
    assert s.bo.level == 2
    m, (r,) = s.bo.contribution(s.now, 0.5, "assist")
    assert m and r["reason"] == "assist" and s.bo.level == 0
    assert s.bo.n_material == {"lead": 1, "assist": 1} and s.bo.n_all == {"lead": 2, "assist": 1}
    with pytest.raises(ValueError):
        s.bo.contribution(s.now, 1.0, "gift")


def test_led_patience_clamp_and_decay():
    """led: τ = clamp(κ / λ̂, min, max) with λ̂ decayed in busy time; a burst of leads gives min, silence gives max; the
    step down falls when b − b_last reaches the τ of that moment."""
    c = cfg(levels=2, patience_kappa=3.0, patience_min_s=2.0, patience_max_s=40.0, half_life_s=20.0)
    s = Sim(c)
    s.bo.phase_y_exit(s.now, "tick")
    for _ in range(20):                                      # 20 leads in 2 busy seconds: λ̂ ~ 10/s
        s.work(0.1)
        s.bo.contribution(s.now, 0.01, "lead")
    assert s.bo.lam() > 1.5 and s.bo.patience() == 2.0
    b_lead = s.bo.busy(s.now)
    s.work(10.0)
    esc = _escalations(s.recs)
    assert esc and esc[0][1] - b_lead >= 2.0 - 1e-9
    # self-consistent: at the dated step the patience of that moment has elapsed (within one update)
    one = Sim(c)
    one.bo.phase_y_exit(one.now, "tick")
    one.work(10.0)
    one.bo.contribution(one.now, 0.2, "lead")
    bl = one.bo.busy(one.now)
    one.work(60.0, step=0.01)
    (lv, be), = _escalations(one.recs)[:1]
    probe = Backoff(c, 0.0)                                  # the patience at be, from the same event history
    probe.phase_y_exit(0.0, "tick")
    probe.E, probe.T, probe.lam_b = 0.0, 0.0, 0.0
    probe._lam_to(10.0)
    probe.E += 1.0
    probe.state = "led"
    probe._lam_to(be)
    assert be - bl == pytest.approx(probe.patience(), abs=0.02)
    # silence: λ̂ decays and τ reaches the max
    late = Sim(c)
    late.bo.phase_y_exit(late.now, "tick")
    late.bo.contribution(late.now + 0.5, 0.2, "lead")
    late.now += 0.5
    late.bo._lam_to(late.bo.busy(late.now) + 400.0)
    assert late.bo.patience() == 40.0


def test_one_forgiven_wake_per_window():
    """Active mode at a reduced duty (phase Y at level 2): a wake forgives the credit at most once per wake_min_s and
    leaves the level alone."""
    s = Sim(cfg(mode="active", yield_level=2, yield_until_s=1e6, wake_min_s=5.0))
    s.now += 2.0
    assert s.bo.sleep_due(s.now) > 0
    assert s.bo.wake(s.now) is True and s.bo.c == 0.0 and s.bo.sleep_due(s.now) == 0.0
    s.now += 2.0
    debt = s.bo.sleep_due(s.now)
    assert debt > 0
    assert s.bo.wake(s.now) is False and s.bo.sleep_due(s.now) == pytest.approx(debt)   # not forgiven within 5 s
    s.now += 3.1
    assert s.bo.wake(s.now) is True
    assert (s.bo.wakes, s.bo.forgiven) == (3, 2) and s.bo.level == 2 and s.bo.in_phase_y


def test_forced_mode_counts_wakes_but_never_forgives():
    """Forced mode (stage 0): a wake is counted but the debt stays, so the measured duty is the set dose however often
    partners post (D-018 amendment) — here a wake at every between-generation pause."""
    s = Sim(cfg(mode="forced", forced_duty=0.25, wake_min_s=5.0))
    s.now += 2.0
    debt = s.bo.sleep_due(s.now)
    c = s.bo.c
    assert debt > 0 and s.bo.wake(s.now) is False and s.bo.c == c and s.bo.sleep_due(s.now) == pytest.approx(debt)
    s.now += 10.0
    assert s.bo.wake(s.now) is False and (s.bo.wakes, s.bo.forgiven) == (2, 0)
    for d in (0.5, 0.25, 0.125):
        f = Sim(cfg(mode="forced", forced_duty=d, wake_min_s=5.0))
        while f.wall() < 120.0:
            f.work(0.3)                                      # a generation of paced FJ launches
            f.bo.wake(f.now)                                 # a partner improvement at every pause
            f.pace()
        assert f.bo.wakes >= 40 and f.bo.forgiven == 0
        assert f.bo.busy(f.now) / f.wall() == pytest.approx(d, abs=0.01)


def test_shadow_never_sleeps_and_logs_would_be_steps():
    s = Sim(cfg(mode="shadow", levels=2, patience_search_s=10.0, step_s=4.0))
    s.work(3.0)
    assert s.bo.paused == 0.0 and s.bo.sleep_due(s.now) == 0.0
    s.bo.phase_y_exit(s.now, "tick")
    b0 = s.bo.busy(s.now)
    s.work(30.0, step=9.0)                                   # sparse updates (once per generation)
    esc = _escalations(s.recs)
    assert [b - b0 for _, b in esc] == pytest.approx([10.0, 14.0])
    assert s.bo.paused == 0.0 and s.bo.busy(s.now) == pytest.approx(s.wall())
    rec = s.bo.record(s.now)
    assert rec["mode"] == "shadow" and rec["first_escalation"]["busy_s"] == pytest.approx(b0 + 10.0)
    assert rec["busy_share"] == pytest.approx(1.0)


def test_shadow_logs_a_step_due_before_a_lead_resets_it():
    """Shadow mode updates once per generation: a step down that falls due inside a generation in which a material
    lead (or the first own feasible member) then resets the ladder is still logged, dated at its due time — before
    the reset — and the would-be first escalation precedes the first lead."""
    s = Sim(cfg(mode="shadow", levels=2, patience_search_s=75.0, patience_own_s=30.0, step_s=4.0))
    t = lambda w: s.bo.t0 + w                                                   # noqa: E731
    s.bo.phase_y_exit(t(0.3), "tick")
    s.recs += s.bo.own_feasible(t(10.0))                                       # own: the step falls due at busy 40
    s.recs += s.bo.update(t(15.0)) + s.bo.update(t(30.0))                      # generation boundaries
    m, recs = s.bo.contribution(t(43.0), 0.01, "lead")                         # inside the next generation
    assert m and [r["reason"] for r in recs] == ["escalate", "lead"]
    esc = recs[0]
    assert esc["busy_s"] == pytest.approx(40.0) and esc["t_due"] == pytest.approx(40.0) and esc["level"] == 1
    assert s.bo.first_escalation == dict(t=40.0, busy_s=40.0, level=1, state="own")
    assert s.bo.level == 0 and s.bo.first_lead["t"] == pytest.approx(43.0)
    rec = s.bo.record(t(60.0))
    assert rec["escalations"] == 1 and rec["busy_per_level"]["1"] == pytest.approx(3.0)
    assert [x[0] for x in rec["timeline"]] == sorted(x[0] for x in rec["timeline"])   # the timeline is in due order
    # the first own feasible member resets too: a search-state step due before it is logged first
    o = Sim(cfg(mode="shadow", levels=2, patience_search_s=10.0, step_s=4.0))
    o.bo.phase_y_exit(o.bo.t0, "tick")
    recs = o.bo.own_feasible(o.bo.t0 + 15.0)
    assert [r["reason"] for r in recs] == ["escalate", "escalate", "own_feasible"]
    assert [r["t_due"] for r in recs[:2]] == pytest.approx([10.0, 14.0]) and o.bo.b_last == pytest.approx(15.0)


def test_phase_y_expiry_is_dated_at_yield_until_s():
    """An expiry detected late (shadow: at the first generation boundary after it) is dated at yield_until_s — phase P
    and the would-be ladder start there —, and an exit seen by a tick, a peek or the EA's own incumbent after
    yield_until_s is the expiry too."""
    c = cfg(mode="shadow", yield_until_s=30.0, patience_search_s=75.0, levels=2)
    s = Sim(c)
    recs = s.bo.update(s.bo.t0 + 48.0)                                         # chromaticindex's generation 0
    assert [r["reason"] for r in recs] == ["y_expiry"] and recs[0]["t_due"] == pytest.approx(30.0)
    assert s.bo.y_exit == dict(t=30.0, busy_s=30.0, reason="expiry") and s.bo.b_last == pytest.approx(30.0)
    assert s.bo.update(s.bo.t0 + 96.0) == []
    esc = s.bo.update(s.bo.t0 + 119.0)
    assert [(r["reason"], r["busy_s"], r["t_due"]) for r in esc] == [("escalate", 105.0, 105.0),
                                                                     ("escalate", 109.0, 109.0)]
    rec = s.bo.record(s.bo.t0 + 120.0)
    assert rec["phase_y"]["length_s"] == pytest.approx(30.0) and rec["seconds_per_level"]["Y"] == pytest.approx(30.0)
    for why in ("tick", "peek", "own"):
        late = Sim(c)
        r = late.bo.phase_y_exit(late.bo.t0 + 48.0, why)
        assert r["reason"] == "y_expiry" and late.bo.y_exit == dict(t=30.0, busy_s=30.0, reason="expiry")
    early = Sim(c)
    assert early.bo.phase_y_exit(early.bo.t0 + 29.0, "tick")["reason"] == "y_tick"
    # active at 1/8: the expiry's busy time counts the sleeps before it, not the work after it
    a = Sim(cfg(yield_level=3, yield_until_s=30.0, patience_search_s=1e3))
    a.now += 2.0
    a.pace()                                                                   # 2 s of work, then a 4-s sleep
    w1, b1 = a.wall(), a.bo.busy(a.now)
    assert a.bo.paused == pytest.approx(4.0) and a.bo.in_phase_y
    a.now = a.bo.t0 + 31.0                                                     # work from 6 s to 31 s, no pace point
    r = a.bo.update(a.now)[0]
    assert r["reason"] == "y_expiry" and a.bo.y_exit["t"] == 30.0
    assert a.bo.y_exit["busy_s"] == pytest.approx(b1 + (30.0 - w1)) == pytest.approx(26.0)


def test_record_and_gen_fields():
    s = Sim(cfg(yield_level=1, levels=2, patience_search_s=5.0, step_s=1.0))
    s.work(4.0)
    s.bo.phase_y_exit(s.now, "peek")
    s.work(10.0)
    g = s.bo.gen_fields(s.now)
    assert g["bo_phase"] == "P" and g["bo_level"] == 2 and g["bo_duty"] == 0.25 and g["bo_paused_call_s"] > 0
    rec = s.bo.record(s.now)
    assert rec["phase_y"]["reason"] == "peek" and rec["phase_y"]["length_s"] == pytest.approx(s.bo.y_exit["t"])
    assert set(rec["seconds_per_level"]) == {"Y", "0", "1", "2"}
    assert sum(rec["seconds_per_level"].values()) == pytest.approx(s.wall(), abs=1e-5)
    assert sum(rec["busy_per_level"].values()) == pytest.approx(s.bo.busy(s.now), abs=1e-5)
    assert rec["busy_share"] == pytest.approx(s.bo.busy(s.now) / s.wall(), abs=1e-5)
    assert rec["timeline"][0][4] == "y_peek" and rec["escalations"] == 2
    json.dumps(rec)                                           # serialisable


# --------------------------------------------------------------------------------------------- configuration
def test_config_hash_out_while_disabled_and_validation():
    base = EAConfig.from_yaml(ROOT / "configs" / "itrea2-dev-rc-3way.yaml")
    assert EAConfig().hash() == "9ac4a93bf93b" and base.hash() == "1d12ad490dd0"
    assert base.with_overrides({"exchange.from_cuopt": True}).hash() == "0f011a2094a2"
    assert base.with_overrides({"ranks.rank1_mode": "clearing"}).hash() == "8d01ac13f25f"
    assert base.with_overrides({"ranks.elite_basis": "population"}).hash() == "e0483418b66d"
    assert base.with_overrides({"backoff.mode": "shadow", "backoff.levels": 3}).hash() == "1d12ad490dd0"   # still off
    on = base.with_overrides({"backoff.enabled": True})
    assert on.hash() != base.hash()
    assert on.with_overrides({"backoff.mode": "shadow"}).hash() != on.hash()
    for bad in ({"backoff.mode": "sleepy"}, {"backoff.levels": 0}, {"backoff.levels": 4}, {"backoff.yield_level": 4},
                {"backoff.forced_duty": 0.0}, {"backoff.forced_duty": 1.5}, {"backoff.step_s": 0.0},
                {"backoff.patience_min_s": 50.0}, {"backoff.slice_s": -1.0}, {"backoff.patience_kappa": 0.0}):
        with pytest.raises(ValueError):
            base.with_overrides(dict(bad, **{"backoff.enabled": True}))
    with pytest.raises(ValueError):
        EAConfig().with_overrides({"backoff.enabled": True, "exchange.enabled": False})
    with pytest.raises(ValueError):
        EAConfig().with_overrides({"backoff.enabled": True, "repair.method": "dense"})
    # a phase-Y duty below the floor stays expressible (stage-0 rules (a) and (b) are separate)
    assert EAConfig().with_overrides({"backoff.enabled": True, "backoff.yield_level": 3, "backoff.levels": 2})


def test_stage0_config_files():
    """The stage-0 arms: the development base plus a `backoff` section (bo0 shadow, bof* forced); with the section
    switched off each is exactly the base (hash 1d12ad490dd0); none sets exchange.from_cuopt (the runner does, for
    A9)."""
    root = ROOT / "configs"
    want = {"bo0": ("shadow", 1.0), "bof50": ("forced", 0.5), "bof25": ("forced", 0.25), "bof12": ("forced", 0.125)}
    for v, (mode, d) in want.items():
        c = EAConfig.from_yaml(root / f"itrea2-dev-rc-3way-{v}.yaml")
        assert c.backoff.enabled and c.backoff.mode == mode and not c.exchange.from_cuopt
        if mode == "forced":
            assert c.backoff.forced_duty == d
        assert c.with_overrides({"backoff.enabled": False}).hash() == "1d12ad490dd0"
    bo1 = root / "itrea2-dev-rc-3way-bo1.yaml"                     # written after stage 0 (D-018), from its values
    if bo1.exists():
        c = EAConfig.from_yaml(bo1)
        assert c.backoff.enabled and c.backoff.mode == "active" and not c.exchange.from_cuopt
        assert c.with_overrides({"backoff.enabled": False}).hash() == "1d12ad490dd0"


# ------------------------------------------------------------------------------------------------- mailboxes
def test_file_info_peek_does_not_consume(tmp_path):
    from itrea.exchange.mailbox import Mailbox
    Mailbox(tmp_path / "mb", "runner", create=True, run_id="mb")
    w, r = Mailbox(tmp_path / "mb", "cuopt", run_id="mb"), Mailbox(tmp_path / "mb", "ea", run_id="mb")
    assert r.info_cuopt.peek() is None and r.info.peek() is None
    w.info_cuopt.post(incumbent=7.0, dual_bound=3.0, status="solving", solver="cuopt")
    assert r.info_cuopt.peek()["incumbent"] == 7.0
    assert r.info_cuopt.peek()["incumbent"] == 7.0
    assert r.info_cuopt.poll()["incumbent"] == 7.0              # still there for the poll
    assert r.info_cuopt.poll() is None and r.info_cuopt.peek()["incumbent"] == 7.0


def test_redis_info_peek_does_not_consume():
    import uuid
    from itrea.experiments import exchange_ctl
    if not exchange_ctl.ping():
        pytest.skip("itrea-redis is not answering on 127.0.0.1:6390")
    from itrea.exchange.redis_mailbox import RedisMailbox
    rid = f"pytest-bo-{uuid.uuid4().hex[:10]}"
    url = f"redis://127.0.0.1:6390/0?run={rid}"
    runner = RedisMailbox(url, run_id=rid, role="runner", create=True)
    try:
        w, r = RedisMailbox(url, run_id=rid, role="cuopt"), RedisMailbox(url, run_id=rid, role="ea")
        assert r.info_cuopt.peek() is None
        w.info_cuopt.post(incumbent=5.0, dual_bound=1.0, status="solving", solver="cuopt")
        assert r.info_cuopt.peek()["incumbent"] == 5.0 and r.info_cuopt.peek()["incumbent"] == 5.0
        assert r.info_cuopt.poll()["incumbent"] == 5.0 and r.info_cuopt.poll() is None
    finally:
        runner.cleanup()


# ------------------------------------------------------------------------------------- FJCpu.run(pace=...)
def _fj_cpu(prob, seed=5):
    from itrea.backend import get_backend
    from itrea.config import RepairCfg
    from itrea.ea.evaluate import to_device
    from itrea.ea.ls_fj_cpu import FJCpu
    return FJCpu(to_device(prob, get_backend("numpy")), RepairCfg(max_steps=2000, stall_steps=100), 1e-9, 64 << 20,
                 seed)


def _starts(prob, k, seed):
    rs = np.random.default_rng(seed)
    X = (rs.random((k, prob.n)) < 0.3).astype(float)
    return np.clip(X, prob.lb, prob.ub)


class _Pace:
    def __init__(self, s: float = 0.0, on_call=None):
        self.s, self.calls, self.on_call = s, 0, on_call

    def __call__(self) -> float:
        self.calls += 1
        if self.on_call is not None:
            self.on_call(self)
        if self.s > 0:
            time.sleep(self.s)
        return self.s


def test_fj_cpu_pace_changes_nothing_but_the_clock():
    prob = toy.set_cover(60, 40, seed=1)
    X0 = _starts(prob, 24, 2)
    ref = _fj_cpu(prob)
    Xr = X0.copy()
    out0 = ref.run(Xr)
    assert "paused_s" not in out0 and "wall_s" not in out0
    for s in (0.0, 0.002):
        fj = _fj_cpu(prob)
        X = X0.copy()
        p = _Pace(s)
        out = fj.run(X, pace=p)
        np.testing.assert_array_equal(X, Xr)
        np.testing.assert_array_equal(fj.last_status, ref.last_status)
        np.testing.assert_array_equal(fj.last_steps, ref.last_steps)
        assert p.calls >= 1 and out["paused_s"] == pytest.approx(p.calls * s)
        assert out["seconds"] == pytest.approx(out["wall_s"] - out["paused_s"])
        assert fj.last_profile["seconds"] == out["seconds"]


def test_fj_cpu_deadline_on_busy_time(monkeypatch):
    """cap_s compares BUSY time: a pace that 'sleeps' 10 s per slice on a fake clock does not trigger a 5-s deadline."""
    from itrea.ea import ls_fj_cpu
    prob = toy.set_cover(60, 40, seed=1)
    X0 = _starts(prob, 16, 3)
    clock = [100.0]
    monkeypatch.setattr(ls_fj_cpu, "_clock", lambda: clock[0])

    def sleep10():
        clock[0] += 10.0
        return 10.0
    fj = _fj_cpu(prob)
    X = X0.copy()
    out = fj.run(X, cap_s=5.0, pace=sleep10)
    assert out["cap_hits"] == 0 and out["finished_horizon"] == 0
    ref = _fj_cpu(prob)
    Xr = X0.copy()
    ref.run(Xr)
    np.testing.assert_array_equal(X, Xr)
    # the same call where the pace's seconds were NOT sleeps: wall time passes the deadline
    fj2 = _fj_cpu(prob)

    def busy10():
        clock[0] += 10.0
        return 0.0
    out2 = fj2.run(X0.copy(), cap_s=5.0, pace=busy10)
    assert out2["cap_hits"] == 1


def test_fj_cpu_stop_during_a_sleep():
    prob = toy.set_cover(60, 40, seed=1)
    X0 = _starts(prob, 16, 4)
    flag = {"stop": False}

    def on_call(p):
        if p.calls == 2:
            flag["stop"] = True                              # the stop arrives while asleep
    fj = _fj_cpu(prob)
    p = _Pace(0.001, on_call)
    out = fj.run(X0.copy(), should_stop=lambda: flag["stop"], pace=p)
    # pace 1, a one-row slice, pace 2 (the stop arrives while asleep), seen by the poll right after it: one row ran
    assert p.calls == 2 and out["interrupted"] == 15 and int(np.sum(fj.last_status != 0)) == 1


def test_local_search_pace_needs_fj():
    from itrea.backend import get_backend
    from itrea.config import RepairCfg
    from itrea.ea.evaluate import to_device
    from itrea.ea.repair import LocalSearch
    prob = toy.set_cover(20, 10, seed=1)
    dp = to_device(prob, get_backend("numpy"))
    ls = LocalSearch(get_backend("numpy"), dp, RepairCfg(method="dense"), 1e-9, 64 << 20, 0)
    with pytest.raises(ValueError):
        ls.run(_starts(prob, 2, 0), pace=lambda: 0.0)


# ================================================================================================ the engine


class _Latest:
    """A latest-wins info channel: poll returns a new post once, peek always (non-consuming)."""

    def __init__(self):
        self.d, self.seq, self.seen = None, 0, 0

    def post(self, **d):
        self.seq += 1
        self.d = dict(d, seq=self.seq)

    def poll(self):
        if self.d is None or self.seq <= self.seen:
            return None
        self.seen = self.seq
        return dict(self.d)

    def peek(self):
        return None if self.d is None else dict(self.d)


class _Queue:
    def __init__(self, items=()):
        self.items = list(items)

    def poll(self):
        return self.items.pop(0) if self.items else None


class Partners:
    """SCIP and cuOpt in memory (arm A9): queued solution messages, latest-wins info channels with `peek`, posts to SCIP
    answered by a verdict per row whose note carries SCIP's primal_before / primal_after / improved (an EA row better
    than SCIP's best becomes SCIP's incumbent) and, as scip_run does, `cuopt` for a row whose origin has POST_CUOPT
    (`mark_cuopt`), and a stop flag that turns on at `stop_at` (monotonic)."""

    def __init__(self, prob, scip_inc: float | None = None, mark_cuopt: bool = True):
        self.prob = prob
        self.to_ea, self.lp, self.to_ea_cuopt = _Queue(), _Queue(), _Queue()
        self.info, self.info_cuopt = _Latest(), _Latest()
        self.best = scip_inc
        if scip_inc is not None:
            self.info.post(incumbent=scip_inc, dual_bound=None, status="solving")
        self.posts: list = []
        self._verdicts: list[dict] = []
        self.stop_at: float | None = None
        outer = self

        class _To:
            def post(_s, X, obj, origin=None):
                outer.posts.append((np.asarray(X), np.asarray(obj)))
                seq = len(outer.posts)
                for r, o in enumerate(np.asarray(obj, float)):
                    before = outer.best
                    better = before is None or prob.key_of_objective(o) < prob.key_of_objective(before)
                    if better:
                        outer.best = float(o)
                    note = dict(row=r, primal_before=before, primal_after=outer.best, improved=bool(better),
                                stored=bool(better), feasible=True, dup=False)
                    if mark_cuopt and origin is not None and int(np.asarray(origin)[r]) & POST_CUOPT:
                        note["cuopt"] = True
                    outer._verdicts.append(dict(seq=seq, t=0.0, obj=float(o), accepted=True, note=json.dumps(note)))
                return seq
        self.to_solver = _To()

    def stopped(self) -> bool:
        return self.stop_at is not None and time.monotonic() >= self.stop_at

    def verdicts(self) -> list:
        return list(self._verdicts)


def _msg(x, obj, seq=1):
    return Message(X=np.atleast_2d(x), obj=np.atleast_1d(np.asarray(obj, float)), seq=seq, t=0.0,
                   origin=np.full(np.atleast_2d(x).shape[0], ORIGIN_SOLVER, np.int64))


def bo_cfg(**over):
    d = {"exchange.from_cuopt": True, "backoff.enabled": True, "exchange.poll_s": 1e-6, "log_every": 1}
    d.update(over)
    return cfg_for(**d)


FAST = {"backoff.burst_s": 0.02, "backoff.slice_s": 0.02, "backoff.max_sleep_s": 0.2}


@pytest.mark.parametrize("mode", ["active", "forced", "shadow"])
def test_inert_without_from_cuopt_equals_the_references(mode, monkeypatch):
    """Enabled but without exchange.from_cuopt (A3 / A3x): no governor at all (building one fails the test) — the
    recorded references of the engine before the feature (plain toys, and the A3 path under solver traffic) are
    reproduced exactly."""
    from itrea.ea import engine as E

    def no_governor(*a, **k):
        pytest.fail("a back-off governor was built without exchange.from_cuopt")
    monkeypatch.setattr(E, "Backoff", no_governor)
    over = {"backoff.enabled": True, "backoff.mode": mode, "backoff.forced_duty": 0.125, "backoff.burst_s": 1e-3}
    ref = REF.load_reference()
    for name in REF.TOYS:
        got = REF.capture(name, "numpy", cfg=REF.config("numpy", **over))
        for k, v in got.items():
            r = ref[f"numpy/{name}/{k}"]
            assert r.shape == v.shape and np.array_equal(r, v, equal_nan=True), (mode, name, k)
    tref = REF.load_traffic_reference()
    for name in REF.TRAFFIC_TOYS:
        got = REF.capture_traffic(name, cfg=REF.traffic_config(name, **over))
        for k, v in got.items():
            r = tref[f"{name}/{k}"]
            assert r.shape == v.shape and np.array_equal(r, v, equal_nan=True), (mode, name, k)


def test_shadow_equals_the_references_beside_a_silent_cuopt(monkeypatch):
    """Shadow mode beside cuOpt (from_cuopt on, cuOpt silent): bookkeeping only, so genealogy, population, counters,
    posts and keys equal the recorded references (plain toys and solver traffic) — with steps down due at once."""
    over = {"exchange.from_cuopt": True, "backoff.enabled": True, "backoff.mode": "shadow",
            "backoff.patience_search_s": 1e-3, "backoff.patience_own_s": 1e-3, "backoff.step_s": 1e-3,
            "backoff.patience_min_s": 1e-3, "backoff.patience_max_s": 1e-3, "backoff.yield_until_s": 1e-3}
    monkeypatch.setattr(REF, "LPMailbox", _CuoptLPMailbox)
    ref = REF.load_reference()
    for name in REF.TOYS:
        got = REF.capture(name, "numpy", cfg=REF.config("numpy", **over))
        for k, v in got.items():
            r = ref[f"numpy/{name}/{k}"]
            assert r.shape == v.shape and np.array_equal(r, v, equal_nan=True), (name, k)
    tref = REF.load_traffic_reference()
    for name in REF.TRAFFIC_TOYS:
        got = REF.capture_traffic(name, cfg=REF.traffic_config(name, **over), mailbox_cls=_TrafficCuoptMailbox)
        for k, v in got.items():
            r = tref[f"{name}/{k}"]
            assert r.shape == v.shape and np.array_equal(r, v, equal_nan=True), (name, k)


def _three_way_partners(p):
    """SCIP with a poor incumbent, cuOpt posting a mediocre row and incumbent: the EA can lead and assist."""
    x_opt = knapsack_optimum_x(p)
    mid = x_opt.copy()
    mid[np.flatnonzero(x_opt)[:3]] = 0.0
    mb = Partners(p, scip_inc=1.0)
    mb.to_ea_cuopt.items.append(_msg(mid, p.objective(mid)))
    mb.info_cuopt.post(incumbent=float(p.objective(mid)), dual_bound=None, status="solving", solver="cuopt")
    return mb


def _genealogy(eng):
    g = eng.genealogy()
    return {k: np.asarray(v) for k, v in g.items() if k != "t_first"}


def test_shadow_under_cuopt_traffic_logs_and_keeps_the_genealogy(tmp_path):
    """Shadow mode with talking partners: the would-be steps down, leads and assists are logged, nothing sleeps, no
    pace reaches the local search, and the genealogy equals the run with the section off."""
    p = toy.knapsack(30)
    shadow = {"backoff.mode": "shadow", "backoff.patience_search_s": 1e-3, "backoff.patience_own_s": 1e-3,
              "backoff.step_s": 1e-3, "backoff.patience_min_s": 1e-3, "backoff.patience_max_s": 1e-3}
    gens = {}
    for label, over in (("off", {"backoff.enabled": False}), ("shadow", shadow)):
        tr = Trace(tmp_path / label, "ea")
        eng = prepared(p, bo_cfg(**over), seed=11, mailbox=_three_way_partners(p), trace=tr)
        eng.begin(time.monotonic(), 3600.0)
        for _ in range(12):
            eng.step()
        eng._exchange_in(force=True)
        out = eng.finish()
        tr.close()
        gens[label] = (_genealogy(eng), out, read_events(tmp_path / label / "ea.events.jsonl"))
    (g0, o0, e0), (g1, o1, e1) = gens["off"], gens["shadow"]
    assert set(g0) == set(g1)
    for k in g0:
        assert np.array_equal(g0[k], g1[k], equal_nan=True), k
    assert "backoff" not in o0 and not any(e["kind"] in ("backoff", "contrib") for e in e0)
    bo = o1["backoff"]
    assert bo["mode"] == "shadow" and bo["paused_s"] == 0.0 and bo["busy_share"] == pytest.approx(1.0)
    assert bo["phase_y"]["reason"] == "tick" and bo["n_lead"] >= 1 and bo["n_assist"] + bo["n_assist_dup"] >= 1
    assert bo["escalations"] >= 1 and bo["first_escalation"] is not None
    reasons = [e["reason"] for e in e1 if e["kind"] == "backoff"]
    assert reasons[0] == "start" and "y_tick" in reasons and "escalate" in reasons
    assert all("paused" not in e["seconds"] for e in e1 if e["kind"] == "gen")
    assert all("bo_level" in e and "ls_steps" in e for e in e1 if e["kind"] == "gen")
    contrib = [e for e in e1 if e["kind"] == "contrib"]
    assert {c["what"] for c in contrib} == {"lead", "assist"} and any(c["material"] for c in contrib)


def test_finish_logs_a_step_due_after_the_last_generation_boundary(tmp_path):
    """Shadow mode updates at generation boundaries; a step that falls due after the last one is logged by `finish`
    (dated at its due time, before the end)."""
    p = toy.knapsack(30)
    tr = Trace(tmp_path, "ea")
    eng = prepared(p, bo_cfg(**{"backoff.mode": "shadow"}), seed=11, mailbox=_three_way_partners(p), trace=tr)
    eng.begin(time.monotonic(), 3600.0)
    for _ in range(2):
        eng.step()
    bo = eng.bo
    n0 = bo.escalations
    assert bo.level == 0 and bo.phase == "P"
    bo.b_last -= 1000.0                                      # the patience ran out after the last boundary
    out = eng.finish()
    tr.close()
    assert out["backoff"]["escalations"] == n0 + bo.L
    ev = read_events(tmp_path / "ea.events.jsonl")
    last_gen = max(e["t"] for e in ev if e["kind"] == "gen")
    esc = [e for e in ev if e["kind"] == "backoff" and e["reason"] == "escalate" and e["t"] >= last_gen]
    assert len(esc) == bo.L and all(e["t_due"] <= e["t"] for e in esc)


def test_forced_duty_paces_and_the_repair_clock_sees_busy_time(tmp_path):
    """Forced 1/4 with the repair clock active: sleeps between generations and inside calls, the measured busy share
    near 1/4, the deadline never overslept, and per generation the repair clock's gen_s + the pause = the wall time,
    with o_s >= 0 (its calls' seconds are busy seconds too)."""
    p = toy.knapsack(30)
    cfg = bo_cfg(**dict(FAST, **{"backoff.burst_s": 0.002, "backoff.mode": "forced", "backoff.forced_duty": 0.25,
                                 "exchange.poll_s": 0.01, "repair_clock.enabled": True,
                                 "repair_clock.mode": "active"}))
    tr = Trace(tmp_path, "ea")
    eng = prepared(p, cfg, seed=3, mailbox=_three_way_partners(p), trace=tr)
    budget = 1.5
    t0 = time.monotonic()
    out = eng.run(budget, t0)
    t_end = time.monotonic()
    tr.close()
    bo = out["backoff"]
    assert bo["mode"] == "forced" and bo["paused_gen_s"] > 0 and bo["paused_call_s"] > 0
    assert bo["busy_share"] == pytest.approx(0.25, abs=0.08)
    assert t_end - t0 <= budget + 0.1                       # one slice and the finish at most
    ev = read_events(tmp_path / "ea.events.jsonl")
    rc = [e for e in ev if e["kind"] == "rclock"]
    assert len(rc) >= 3
    for e in rc:                                             # exact by construction (one clock reading), up to rounding
        assert e["gen_s"] + e["bo_paused_s"] == pytest.approx(e["bo_wall_s"], abs=1e-5)
        assert e["o_s"] >= 0 and e["gen_s"] - sum(c["s"] for c in e["calls"] if c["cf"]) >= -1e-3
    # the repair clock decides (and decays, probes) on BUSY time: at 1/4 its decision time is about a quarter of wall
    assert rc[-1]["t"] > 0.8 and rc[-1]["t_dec"] <= 0.5 * rc[-1]["t"]
    nxt = {}                                                 # each rclock event's generation record (written after it)
    for j, e in enumerate(ev):
        if e["kind"] == "rclock":
            nxt[j] = next((g for g in ev[j + 1:] if g["kind"] == "gen"), None)
    pairs = [(ev[j]["t_dec"], g["bo_busy_s"]) for j, g in nxt.items() if g is not None]
    assert len(pairs) >= 3 and all(td <= b + 1e-6 for td, b in pairs)     # decided at a busy time, not a wall time
    assert any(e["bo_paused_s"] > 0 for e in rc)            # in-call pauses, outside gen_s
    assert sum(e["gen_s"] for e in rc) <= bo["busy_s"] + 1e-3
    gen = [e for e in ev if e["kind"] == "gen"]
    assert any("paused" in e["seconds"] for e in gen)
    assert sum(e["ls_steps"] for e in gen) == bo["ls_steps"] > 0


def test_active_phase_y_ends_at_a_peek_inside_a_call(tmp_path):
    """cuOpt's incumbent is on its info channel at t0: the first pace point of the initial repair peeks (without
    consuming it) and ends phase Y; the initial exchange tick still reads it."""
    p = toy.knapsack(30)
    mb = Partners(p)
    mb.info_cuopt.post(incumbent=5.0, dual_bound=None, status="solving", solver="cuopt")
    tr = Trace(tmp_path, "ea")
    eng = prepared(p, bo_cfg(**FAST, **{"backoff.yield_level": 3}), seed=4, mailbox=mb, trace=tr)
    eng.begin(time.monotonic(), 3600.0)
    assert not eng.bo.in_phase_y and eng.bo.y_exit["reason"] == "peek"
    assert eng.inc_cuopt == pytest.approx(p.key_of_objective(5.0))        # not consumed by the peek
    for _ in range(3):
        eng.step()
    out = eng.finish()
    tr.close()
    reasons = [e["reason"] for e in read_events(tmp_path / "ea.events.jsonl") if e["kind"] == "backoff"]
    assert reasons[:2] == ["start", "y_peek"] and "y_tick" not in reasons
    assert out["backoff"]["phase_y"]["reason"] == "peek"


def test_active_leads_and_assists_are_credited(tmp_path):
    p = toy.knapsack(30)
    tr = Trace(tmp_path, "ea")
    eng = prepared(p, bo_cfg(**FAST), seed=5, mailbox=_three_way_partners(p), trace=tr)
    out = eng.run(60.0, time.monotonic(), max_gens=10)
    tr.close()
    bo = out["backoff"]
    assert bo["n_lead"] >= 1 and bo["n_assist_dup"] >= 1 and bo["first_lead"] is not None
    assert bo["phase_y"]["reason"] == "peek"                  # partners' incumbents on the channels at t0
    con = [e for e in read_events(tmp_path / "ea.events.jsonl") if e["kind"] == "contrib"]
    leads = [e for e in con if e["what"] == "lead" and e["material"]]
    assists = [e for e in con if e["what"] == "assist" and e["material"]]
    dups = [e for e in con if e["what"] == "assist" and e.get("dup_lead")]
    assert len(leads) == bo["n_lead"] and len(assists) == bo["n_assist"] and len(dups) == bo["n_assist_dup"]
    assert all(0 < e["dg"] <= 1 and e["key"] < e["ref"] for e in leads + assists)
    # one improvement enters λ̂ once: SCIP's confirmation of a material lead is not credited again
    assert {e["id"] for e in leads}.isdisjoint({e["id"] for e in assists})
    assert dups and {e["id"] for e in dups} <= {e["id"] for e in leads} and not any(e["material"] for e in dups)


def test_an_assist_is_credited_when_the_post_was_not_a_material_lead(tmp_path):
    """cuOpt's incumbent (360 of the optimum 370) is known on its info channel but its vector never arrives: EA rows
    worse than it are no leads, yet SCIP (far behind) takes them as improvements — those assists are credited; the
    EA's rows past 360 are material leads, and SCIP's confirmations of those are not credited again."""
    p = toy.knapsack(30)
    mb = Partners(p, scip_inc=1.0)
    mb.info_cuopt.post(incumbent=360.0, dual_bound=None, status="solving", solver="cuopt")
    tr = Trace(tmp_path, "ea")
    eng = prepared(p, bo_cfg(**FAST), seed=5, mailbox=mb, trace=tr)
    out = eng.run(60.0, time.monotonic(), max_gens=10)
    tr.close()
    bo = out["backoff"]
    con = [e for e in read_events(tmp_path / "ea.events.jsonl") if e["kind"] == "contrib"]
    lead_ids = {e["id"] for e in con if e["what"] == "lead" and e["material"]}
    assist_ids = {e["id"] for e in con if e["what"] == "assist" and e["material"]}
    assert bo["n_assist"] >= 1 and len(assist_ids) == bo["n_assist"] and lead_ids.isdisjoint(assist_ids)
    assert bo["n_lead"] == len(lead_ids) >= 1 and bo["n_assist_dup"] >= 1


class _NeverPosted(set):
    """`cuopt_posted` that never answers yes: the engine's other guard (SCIP's note `cuopt`) must hold alone."""

    def __contains__(self, item):
        return False


@pytest.mark.parametrize("guard", ["note", "cuopt_posted"])
def test_an_unchanged_cuopt_row_is_never_an_assist(tmp_path, guard):
    """cuOpt delivers the optimum; the EA relays that unchanged vector to SCIP, which takes it as an improvement. It is
    cuOpt's, not the EA's: no assist (each of the engine's two guards on its own — SCIP's note `cuopt`, the EA's own
    record `cuopt_posted`)."""
    p = toy.knapsack(30)
    x = knapsack_optimum_x(p)
    mb = Partners(p, scip_inc=1.0, mark_cuopt=(guard == "note"))
    mb.to_ea_cuopt.items.append(_msg(x, p.toy_optimum))
    mb.info_cuopt.post(incumbent=p.toy_optimum, dual_bound=None, status="solving", solver="cuopt")
    tr = Trace(tmp_path, "ea")
    eng = prepared(p, bo_cfg(**FAST), seed=5, mailbox=mb, trace=tr)
    eng.begin(time.monotonic(), 3600.0)
    posted = set(eng.cuopt_posted)
    if guard == "note":
        eng.cuopt_posted = _NeverPosted(eng.cuopt_posted)
    for _ in range(4):
        eng.step()
    eng._exchange_in(force=True)
    out = eng.finish()
    tr.close()
    posted |= set(eng.cuopt_posted)
    assert posted, "the cuOpt row was not relayed to SCIP"
    assert out["cuopt_rows_accepted"] >= 1 or guard == "note"
    con = [e for e in read_events(tmp_path / "ea.events.jsonl") if e["kind"] == "contrib" and e["what"] == "assist"]
    assert not any(e["id"] in posted for e in con)
    assert out["backoff"]["n_assist_all"] == len(con) - out["backoff"]["n_assist_dup"]


def test_stop_file_is_honoured_within_one_slice():
    p = toy.knapsack(30)
    mb = _three_way_partners(p)
    cfg = bo_cfg(**{"backoff.mode": "forced", "backoff.forced_duty": 0.125, "backoff.burst_s": 0.05,
                    "backoff.max_sleep_s": 2.0, "backoff.slice_s": 0.25, "exchange.poll_s": 0.25})
    eng = prepared(p, cfg, seed=6, mailbox=mb)
    t0 = time.monotonic()
    mb.stop_at = t0 + 0.7
    out = eng.run(30.0, t0)
    seen = time.monotonic()
    assert out["exit_reason"] == "stop" and out["backoff"]["paused_s"] > 0.3
    assert seen - mb.stop_at <= 0.25 + 0.15


def _paused_at_one_eighth(p, mb, seed=7):
    """An active engine in phase P at level 3 (duty 1/8) with begin's verdicts drained (an assist there would restore
    full duty and end any sleep for another reason)."""
    cfg = bo_cfg(**{"backoff.levels": 3, "backoff.max_sleep_s": 3.0, "backoff.slice_s": 0.05, "exchange.poll_s": 0.2})
    eng = prepared(p, cfg, seed=seed, mailbox=mb)
    eng.begin(time.monotonic(), 3600.0)
    eng._exchange_in(force=True)
    bo = eng.bo
    assert not bo.in_phase_y
    bo.level = 3
    return eng, bo


def test_between_generation_sleep_wakes_on_a_partner_improvement_and_stops_at_the_deadline():
    p = toy.knapsack(30)
    mb = _three_way_partners(p)
    eng, bo = _paused_at_one_eighth(p, mb)
    x = knapsack_optimum_x(p)
    mb.to_ea_cuopt.items.append(_msg(x, p.toy_optimum, seq=2))           # arrives at the next tick, 0.2 s away
    mb.info_cuopt.post(incumbent=p.toy_optimum, dual_bound=None, status="solving", solver="cuopt")
    bo.c = -2.0                                                         # a debt: a 3-s sleep is due
    eng.next_poll = time.monotonic() + 0.2
    t = time.monotonic()
    eng._bo_between()
    dt = time.monotonic() - t
    assert 0.15 <= dt <= 0.45 and (bo.wakes, bo.forgiven) == (1, 1) and bo.c == 0.0 and bo.level == 3
    assert any(b.src == "cuopt" for b in eng.pending) and eng.inc_cuopt == pytest.approx(-p.toy_optimum)
    # a wake within wake_min_s of the last forgiveness forgives nothing and does NOT end the sleep: the ticks go on,
    # and a later wake outside the window forgives and ends it (one batch per tick: two queued batches, two ticks)
    mb.to_ea_cuopt.items += [_msg(x, p.toy_optimum, seq=3), _msg(x, p.toy_optimum, seq=4)]
    bo.c = -2.0
    t = time.monotonic()
    bo.last_forgive = t - 4.75                                          # the window closes 0.25 s from now
    eng.next_poll = t + 0.1
    eng._bo_between()
    dt = time.monotonic() - t
    assert (bo.wakes, bo.forgiven) == (3, 2) and bo.c == 0.0
    assert 0.25 <= dt <= 0.6                                            # slept through the first wake (at 0.1 s)
    # a non-forgiven wake with the deadline near: asleep to the deadline, never past it, the debt kept
    mb.to_ea_cuopt.items.append(_msg(x, p.toy_optimum, seq=5))
    bo.c = -2.0
    t = time.monotonic()
    eng.next_poll = t + 0.1
    eng.deadline = t + 0.4
    eng._bo_between()
    dt = time.monotonic() - t
    assert (bo.wakes, bo.forgiven) == (4, 2) and bo.c < -1.5
    assert 0.35 <= dt <= 0.4 + 0.06                                     # never past the deadline
    assert eng._should_stop() and eng.stop_reason == "budget"           # (step() returns without a generation)


def test_forced_mode_sleeps_through_wakes():
    """Forced mode never forgives, so a partner improvement never ends its between-generation sleep (the stage-0 dose
    stays fixed); the wake is counted and the rows wait in `pending` for the next generation."""
    p = toy.knapsack(30)
    mb = _three_way_partners(p)
    cfg = bo_cfg(**{"backoff.mode": "forced", "backoff.forced_duty": 0.125, "backoff.max_sleep_s": 0.5,
                    "backoff.slice_s": 0.05, "exchange.poll_s": 0.1})
    eng = prepared(p, cfg, seed=7, mailbox=mb)
    eng.begin(time.monotonic(), 3600.0)
    eng._exchange_in(force=True)
    bo = eng.bo
    x = knapsack_optimum_x(p)
    mb.to_ea_cuopt.items.append(_msg(x, p.toy_optimum, seq=2))
    mb.info_cuopt.post(incumbent=p.toy_optimum, dual_bound=None, status="solving", solver="cuopt")
    bo.c = -2.0                                                         # min(16, 0.5) = a 0.5-s sleep
    eng.next_poll = time.monotonic() + 0.1
    t = time.monotonic()
    eng._bo_between()
    dt = time.monotonic() - t
    assert (bo.wakes, bo.forgiven) == (1, 0) and 0.45 <= dt <= 0.65 and bo.c < -1.5
    assert any(b.src == "cuopt" for b in eng.pending)
