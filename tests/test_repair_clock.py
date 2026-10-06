"""The repair clock (`EAConfig.repair_clock`, `itrea.ea.repair_clock`, the horizon of `ls_fj_cpu`, `itrea.ea.engine`).

Design: vault `80-reviews/2026-10-06 Repair-depth design (workflow).md`. What is tested here, CPU only:
- the horizon of the CPU feasibility jump: ST_HORIZON, steps never above h, the stop rules first (a lane that stops by
  stall at step h stops by stall), horizon 0 = projection only, the call deadline on rows not started (fake clock), and
  the call without the new arguments unchanged bit for bit;
- the counterfactual identity the controller rests on: with the objective cutoff armed, a horizon-h call equals the
  unbounded call on every lane that stopped within h steps (x, status, steps, moves) — the rest stop at h;
- the controller on synthetic calls: a finite solver incumbent alone does not end the pre-feasible phase, one material
  gain never switches, the materiality floor, the hysteresis, the decay, the probe share, the cap (and its rise when the
  calls slow down), the pinned LP batch, the calls after the survival left out of the gain, the gain unit bounded to
  [0, 1], shadow mode accumulating only what its decision would reveal (so it probes), the call-time interpolation and
  what a cut call reveals;
- the engine: shadow mode reproduces the run with the section off (genealogy, population, best key, counters), also
  with the controller held at a shallow level (no argument reaches a call), the section off records nothing new, active
  mode honours the horizon, the LP batch is pinned, the pre-feasible phase holds under a finite solver incumbent and
  under an injected feasible solver row, and the one pairing helper maps every call's lanes onto the evaluated rows at
  every call site (children, each pending batch, `_admit`'s seeded subset) and marks the calls after the survival.
The GPU wrapper's tests (marker `gpu`) are in `tests/test_fj_gpu.py`. Toy models only.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest
import scipy.sparse as sps

import _engine_reference as REF
from itrea.backend import get_backend
from itrea.benchmarks import toy
from itrea.config import EAConfig, RepairCfg, RepairClockCfg
from itrea.ea import ls_fj_cpu
from itrea.ea.engine import Engine, _is_solver_find
from itrea.ea.evaluate import evaluate, to_device
from itrea.ea.ls_fj_cpu import ST_BUDGET, ST_FEASIBLE, ST_HORIZON, ST_STALL, FJCpu
from itrea.ea.repair import LocalSearch
from itrea.ea.repair_clock import RepairClock, _chunk_time
from itrea.exchange.mailbox import Message
from itrea.problem import from_arrays
from itrea.trace import Trace, read_events
from test_fj_cpu import random_start

BK = get_backend("numpy")
DEEP = dict(max_steps=20000, stall_steps=300)


def fj_for(p, seed: int = 5, **cfg) -> FJCpu:
    return FJCpu(to_device(p, BK), RepairCfg(**dict(DEEP, **cfg)), 1e-9, 64 << 20, seed)


def armed_case(name: str):
    """(problem, starts, cutoff key) with a cutoff that some lanes beat quickly and others never (varied steps)."""
    if name == "knapsack":
        p = toy.knapsack(30)
        return p, random_start(p, 64, seed=4, density=0.3), -(p.toy_optimum - 25.0)
    if name == "set_cover":                                # 58 of 64 lanes beat 30, at 9 to ~100 steps
        p = toy.set_cover(40, 25)
        return p, random_start(p, 64, seed=4, density=0.3), 30.0
    p = toy.mixed()
    X0 = random_start(p, 64, seed=4, density=0.3)
    X = X0.copy()
    fj_for(p, seed=3).run(X)                               # an incumbent from an unarmed search
    ev = evaluate(to_device(p, BK), X)
    keys = np.asarray(ev.key)[np.asarray(ev.feasible)]
    return p, X0, float(np.quantile(keys, 0.05))


def run_fj(p, X0, cut, seed=5, **kw):
    fj = fj_for(p, seed=seed)
    fj.set_cutoff(cut)
    X = X0.copy()
    out = fj.run(X, **kw)
    return X, fj, out


# ============================================================================== the CPU feasibility jump's horizon
@pytest.mark.parametrize("name", ["mixed", "knapsack", "set_cover"])
def test_counterfactual_identity_with_cutoff_armed(name):
    """A horizon-h call is a prefix of the unbounded call: every lane that stopped within h steps in the unbounded
    call is identical in the horizon-h call (x, status, steps, moves); every other lane stops at h with ST_HORIZON."""
    p, X0, cut = armed_case(name)
    XU, fu, _ = run_fj(p, X0, cut)
    su, stu = fu.last_steps.copy(), fu.last_status.copy()
    assert fu.cut_active and np.any(stu == ST_FEASIBLE) and np.any(stu != ST_FEASIBLE)
    hs = sorted({0, 1, 3, 8, int(np.median(su)), int(su.max()) - 1, int(su.max())} - {-1})
    both = 0
    for h in hs:
        XH, fh, out = run_fj(p, X0, cut, horizon=h)
        within = su <= h
        both += int(within.any() and (~within).any())
        np.testing.assert_array_equal(XH[within], XU[within])
        np.testing.assert_array_equal(fh.last_status[within], stu[within])
        np.testing.assert_array_equal(fh.last_steps[within], su[within])
        np.testing.assert_array_equal(fh.last_moves[within], fu.last_moves[within])
        assert np.all(fh.last_status[~within] == ST_HORIZON) and np.all(fh.last_steps[~within] == h)
        assert np.all(fh.last_steps <= h)
        assert out["finished_horizon"] == int((~within).sum()) and out["cap_hits"] == 0
        assert out["finished_feasible"] + out["finished_stall"] + out["finished_budget"] == X0.shape[0]
        # the gain signal: the FEASIBLE lanes of the horizon call are exactly the unbounded call's within h
        np.testing.assert_array_equal(fh.last_status == ST_FEASIBLE, (stu == ST_FEASIBLE) & within)
    assert both >= 2                                       # the identity was tested on both sides of h


def test_unset_arguments_unchanged():
    """Without horizon / cap_s the call is the one before the clock existed: no new counter, and the same rows as with
    a horizon that is never reached (both through one Numba call and through the should_stop slicing)."""
    p, X0, cut = armed_case("mixed")
    XA, fa, oa = run_fj(p, X0, cut)
    assert "finished_horizon" not in oa and "cap_hits" not in oa
    assert fa.last_profile["kind"] == "sequential" and fa.last_profile["H"] is None
    for kw in (dict(horizon=1 << 40), dict(should_stop=lambda: False), dict(cap_s=3600.0)):
        XB, fb, ob = run_fj(p, X0, cut, **kw)
        np.testing.assert_array_equal(XB, XA)
        for attr in ("last_status", "last_steps", "last_moves"):
            np.testing.assert_array_equal(getattr(fb, attr), getattr(fa, attr))
        assert ob["moves"] == oa["moves"] and ob["finished_budget"] == oa["finished_budget"]


def test_horizon_zero_projects_only():
    """horizon 0: projection and the stop rules only — the rows equal a max_steps=0 call's; no step, no move; an
    already feasible projection finishes feasible, every other row stops with ST_HORIZON."""
    p = toy.mixed()
    X0 = random_start(p, 16, seed=2, wild=True)
    XA = X0.copy()
    fa = fj_for(p, max_steps=0)
    fa.run(XA)
    XB = X0.copy()
    fb = fj_for(p)
    out = fb.run(XB, horizon=0)
    np.testing.assert_array_equal(XB, XA)
    assert out["moves"] == 0 and np.all(fb.last_steps == 0)
    np.testing.assert_array_equal(fb.last_status == ST_FEASIBLE, fa.last_status == ST_FEASIBLE)
    assert np.all(fb.last_status[fa.last_status == ST_BUDGET] == ST_HORIZON)
    assert out["finished_horizon"] == int(np.sum(fa.last_status == ST_BUDGET))


def test_stop_rules_win_ties_at_the_horizon():
    """The horizon is checked after feasible / budget / stall: a lane that stops by stall exactly at step h stops by
    stall under horizon h, and by the horizon under h - 1."""
    p = from_arrays(sps.csr_matrix(np.array([[1.0, 1.0]])), [3.0], [np.inf], [0, 0], [1, 1], [1.0, 1.0],
                    [True, True])                       # x0 + x1 >= 3 on binaries: infeasible, every lane stalls
    X0 = np.zeros((4, 2))
    fu = fj_for(p, stall_steps=50)
    fu.run(X0.copy())
    assert np.all(fu.last_status == ST_STALL)
    for b in range(4):
        s = int(fu.last_steps[b])
        for h, want, steps in ((s, ST_STALL, s), (s - 1, ST_HORIZON, s - 1)):
            f = fj_for(p, stall_steps=50)
            x = X0.copy()
            f.run(x, horizon=h)
            assert f.last_status[b] == want and f.last_steps[b] == steps


def test_cpu_cap_projects_the_rows_not_started(monkeypatch):
    """With cap_s the rows are processed in slices; the rows not started when the deadline passes are only projected
    (ST_HORIZON, 0 steps); the rows before them are exactly the unbounded call's. A fake clock passes the deadline
    after two slices."""
    p = toy.set_cover(40, 25)
    X0 = np.zeros((24, p.n))                              # infeasible after projection: every searched row moves
    XU, fu, _ = run_fj(p, X0, None)
    ticks = iter([0.0, 0.0, 0.0] + [100.0] * 50)          # deadline set at 0 + cap; two slices start before it
    monkeypatch.setattr(ls_fj_cpu, "_clock", lambda: next(ticks))
    XC, fc, out = run_fj(p, X0, None, cap_s=1.0)
    done = fc.last_steps > 0
    nd = int(done.sum())
    assert 1 <= nd < 24 and np.all(done[:nd])             # a prefix was searched
    np.testing.assert_array_equal(XC[:nd], XU[:nd])
    np.testing.assert_array_equal(fc.last_status[:nd], fu.last_status[:nd])
    assert np.all(fc.last_status[nd:] == ST_HORIZON) and np.all(XC[nd:] == 0)
    assert out["cap_hits"] == 1 and fc.last_profile["cap_hit"] and fc.last_profile["cut_rows"] == 24 - nd
    assert out["finished_horizon"] == 24 - nd


def test_dispatcher_passes_the_clock_arguments_only_when_given():
    p = toy.knapsack(30)
    dp = to_device(p, BK)
    X0 = random_start(p, 8, seed=1)
    ls = LocalSearch(BK, dp, RepairCfg(), 1e-9, 64 << 20, 0)
    seen = []
    orig = ls.impl.run

    def spy(X, **kw):
        seen.append(sorted(kw))
        return orig(X, **kw)
    ls.impl.run = spy
    ls.run(X0.copy())
    ls.run(X0.copy(), horizon=4)
    ls.run(X0.copy(), cap_s=1.0)
    assert seen == [["scale_idx", "should_stop"], ["horizon", "scale_idx", "should_stop"],
                    ["cap_s", "scale_idx", "should_stop"]]
    for method in ("dense", "none"):
        other = LocalSearch(BK, dp, RepairCfg(method=method), 1e-9, 64 << 20, 0)
        other.run(X0.copy())
        with pytest.raises(ValueError, match="horizon"):
            other.run(X0.copy(), horizon=4)


# ============================================================================== the controller (synthetic calls)
def rc_cfg(**kw) -> RepairClockCfg:
    return RepairClockCfg(**dict(dict(enabled=True, mode="active", level_min_log2=7, level_max_log2=16), **kw))


def seq_call(lanes, seconds=1.0, cutoff=100.0, dual=-math.inf, site="children", H=None, pinned=False):
    """A synthetic sequential call: `lanes` = [(steps, status, dg)] with dg the lane's gain (key = cutoff - dg·D,
    D = |cutoff|); status 1 lanes are evaluated feasible."""
    steps = np.array([s for s, _, _ in lanes], np.int64)
    status = np.array([st for _, st, _ in lanes], np.int64)
    D = max(abs(cutoff), abs(dual) if math.isfinite(dual) else 0.0, 1e-9)
    keys = np.array([cutoff - g * D if st == ST_FEASIBLE else cutoff + 1.0 for _, st, g in lanes])
    prof = dict(kind="sequential", seconds=seconds, H=H, cap_s=None, cap_hit=False, cut_rows=0)
    return dict(steps=steps, status=status, keys=keys, feasible=status == ST_FEASIBLE, profile=prof, cutoff=cutoff,
                dual_key=dual, site=site, pinned=pinned, horizon=H)


def one_gen(rc, now, has_feasible, calls, gen_s=1.0):
    dec = rc.decide(now, has_feasible)
    for c in calls:
        rc.add_call(**c)
    return dec, rc.end_generation(now + gen_s, gen_s)


SHALLOW = [(100, ST_FEASIBLE, 0.01), (40000, ST_STALL, 0.0), (40000, ST_STALL, 0.0)]   # a shallow gain, a deep tail
DEEP_GAIN = [(30000, ST_FEASIBLE, 0.5), (30000, ST_STALL, 0.0)]                        # a gain only deep repair finds
NOTHING = [(40000, ST_STALL, 0.0), (40000, ST_STALL, 0.0)]


def test_controller_moves_to_a_shallow_level_on_evidence():
    rc = RepairClock(rc_cfg())
    now, ev = 0.0, None
    for g in range(4):
        dec, ev = one_gen(rc, now, True, [seq_call(SHALLOW)])
        now += 1.0
        assert dec.horizon is None and dec.phase == "post" and dec.cap_s == 10.0
    assert rc.cur == rc.NONE                              # 4 observed generations < min_gens
    dec, ev = one_gen(rc, now, True, [seq_call(SHALLOW)])
    assert ev["next"] == 128 and ev["switched"] and rc.switches == 1
    dec = rc.decide(now + 1.0, True)
    assert dec.horizon == 128 and dec.level == 0
    t128 = 1.0 * (100 + 128 + 128) / (100 + 80000)        # sequential: seconds · Σ min(steps, h) / Σ steps
    assert dec.cap_s == pytest.approx(min(10.0, 3.0 * t128))
    assert ev["gain"][0] == pytest.approx(0.01) and ev["T"][0] == pytest.approx(t128)
    assert ev["T"][-1] == pytest.approx(1.0) and ev["rate"][0] > 1.25 * ev["rate"][-1]


def test_controller_stays_at_none_without_material_gains():
    """No gain, or only gains below the materiality floor: G stays 0, nothing is eligible, the level stays "none"."""
    for g in (0.0, 5e-4):
        rc = RepairClock(rc_cfg())
        lanes = [(100, ST_FEASIBLE if g else ST_STALL, g), (40000, ST_STALL, 0.0)]
        for i in range(30):
            dec, ev = one_gen(rc, float(i), True, [seq_call(lanes)])
            assert dec.horizon is None
        assert rc.cur == rc.NONE and np.all(rc.G == 0) and np.all(rc.events == 0)
    rc = RepairClock(rc_cfg())
    for i in range(6):                                    # just above the floor: a switch
        one_gen(rc, float(i), True, [seq_call([(100, ST_FEASIBLE, 2e-3), (40000, ST_STALL, 0.0)])])
    assert rc.cur == 0


def test_a_single_gain_never_switches():
    rc = RepairClock(rc_cfg())
    one_gen(rc, 0.0, True, [seq_call(SHALLOW)])           # one material gain at a shallow step count ...
    for i in range(1, 40):
        one_gen(rc, float(i), True, [seq_call(NOTHING)])  # ... and nothing else
    assert rc.events[0] == 1 and rc.G[0] > 0
    assert rc.cur == rc.NONE and rc.switches == 0


def test_a_finite_solver_incumbent_alone_keeps_the_pre_feasible_phase():
    """has_feasible False (no feasible member — whatever the cutoff, here armed by a solver incumbent): "none", no cap,
    nothing accumulated, even with gains in every call; the phase ends with the first feasible member, for good."""
    rc = RepairClock(rc_cfg())
    for i in range(20):
        dec, ev = one_gen(rc, float(i), False, [seq_call(SHALLOW, cutoff=42.0)])
        assert dec.phase == "pre" and dec.horizon is None and dec.cap_s is None and ev["phase"] == "pre"
        assert ev["gain"][0] == pytest.approx(0.01)       # logged, but not accumulated
    assert np.all(rc.n == 0) and rc.cur == rc.NONE and rc.gens_pre == 20 and not rc.post
    dec, _ = one_gen(rc, 20.0, True, [seq_call(SHALLOW)])
    assert dec.phase == "post" and rc.post and rc.t_post == 20.0
    dec, _ = one_gen(rc, 21.0, False, [seq_call(SHALLOW)])   # one-way: a lost feasible member does not reopen it
    assert dec.phase == "post"


def test_shadow_mode_plays_none_and_logs_the_choice():
    rc = RepairClock(rc_cfg(mode="shadow"))
    for i in range(6):
        one_gen(rc, float(i), True, [seq_call(SHALLOW)])
    dec, ev = one_gen(rc, 6.0, True, [seq_call(SHALLOW)])
    assert dec.horizon is None and dec.cap_s is None and dec.level == rc.NONE
    assert ev["level"] == "none" and ev["would"] == 128 and ev["cap"] is None and ev["would_cap"] < 10.0
    rec = rc.record()
    assert rec["time_share_played"][-1] == pytest.approx(1.0) and rec["time_share_would"][0] > 0


def test_decay_lets_the_level_follow_a_phase_change():
    """Phase 1: large gains only deep repair finds; phase 2: small gains at a shallow step count. With the 20 s half-life
    the old deep evidence fades and the controller moves shallow within 40 generations; without decay it does not."""
    def run(half_life):
        rc = RepairClock(rc_cfg(half_life_s=half_life))
        now = 0.0
        for _ in range(40):
            one_gen(rc, now, True, [seq_call(DEEP_GAIN)])
            now += 1.0
        assert rc.cur == rc.NONE
        for i in range(40):
            one_gen(rc, now, True, [seq_call([(100, ST_FEASIBLE, 0.002), (30000, ST_STALL, 0.0)])])
            now += 1.0
            if rc.cur != rc.NONE:
                return i
        return None
    assert run(20.0) is not None
    assert run(1e12) is None


def test_probes_re_measure_none_within_the_probe_share():
    """Away from "none" a probe plays "none" when "none" was not observed for a half-life, and probe generations take
    at most `probe_share` of the post-feasible time (plus the one probe that crosses it)."""
    cfg = rc_cfg(half_life_s=5.0)
    rc = RepairClock(cfg)
    o = 0.04                                              # the generation's time outside the repair call
    t128 = (100 + 128 + 128) / (100 + 80000)              # the unbounded call (1 s) cut at 128 steps
    now = 0.0
    for _ in range(6):
        one_gen(rc, now, True, [seq_call(SHALLOW)], gen_s=o + 1.0)
        now += o + 1.0
    assert rc.cur == 0
    probes, t_probe = 0, 0.0
    for _ in range(400):
        dec = rc.decide(now, True)
        if dec.level == rc.NONE:                          # the probe: the unbounded call, long
            assert dec.probe and dec.horizon is None and dec.cap_s == cfg.cap_s
            rc.add_call(**seq_call(SHALLOW))
            gs = o + 1.0
            probes += 1
            t_probe += gs
        else:                                             # the shallow level: the same call cut at 128, short
            assert dec.horizon == 128 and not dec.probe
            rc.add_call(**seq_call([(100, ST_FEASIBLE, 0.01), (128, ST_HORIZON, 0.0), (128, ST_HORIZON, 0.0)],
                                   seconds=t128, H=128))
            gs = o + t128
        rc.end_generation(now + gs, gs)
        now += gs
        assert rc.probe_s <= cfg.probe_share * (now - rc.t_post) + o + 1.0
    assert probes >= 2 and rc.probe_s == pytest.approx(t_probe)
    assert rc.cur == 0                                    # the probes confirmed the shallow level


def test_cap_rule():
    """Post phase: "none" gets cap_s; a level gets min(cap_s, cap_factor x the median of its recent children-call
    times); other call sites do not enter the median. Pre phase: no cap."""
    rc = RepairClock(rc_cfg(cap_s=10.0, cap_factor=3.0))
    assert rc.decide(0.0, False).cap_s is None
    rc.end_generation(0.1, 0.1)
    for i, s in enumerate((1.0, 2.0, 3.0)):
        rc.decide(float(i + 1), True)
        rc.add_call(**seq_call(SHALLOW, seconds=s))
        rc.add_call(**seq_call(SHALLOW, seconds=100.0, site="lps"))
        rc.end_generation(float(i + 1) + s, s)
    frac = (100 + 128 + 128) / (100 + 80000)
    assert rc._cap(rc.NONE) == 10.0
    assert rc._cap(0) == pytest.approx(min(10.0, 3.0 * 2.0 * frac))
    assert rc._cap(rc.NONE - 1) == pytest.approx(6.0)     # 2^16 >= every lane's steps: 3 x median(1, 2, 3)
    rc2 = RepairClock(rc_cfg(cap_s=5.0))
    rc2.decide(0.0, True)
    rc2.add_call(**seq_call(SHALLOW, seconds=1.0))
    rc2.end_generation(1.0, 1.0)
    assert rc2._cap(rc2.NONE - 1) == pytest.approx(3.0)


def test_cap_rises_when_the_children_calls_slow_down():
    """Active, settled at 128: the children calls then become 5x slower per step, so the cap (3 x the median of the
    level's call times) cuts them below 128. A cut call adds a censored time (max(seconds, cap)) to the level's window,
    so the median and the cap rise until the calls finish again; without it the cut calls never enter the window and
    every later call is cut (reviewer simulation 2026-10-06: 5,835 of 5,843)."""
    rc = RepairClock(rc_cfg())
    now, cuts, caps = 0.0, [], []
    for g in range(400):
        d = rc.decide(now, True, gen=g)
        H, slow = d.horizon, g >= 100
        steps, status = np.array([100, 2500, 2500, 2500]), np.array([1, 2, 2, 2])
        if H is not None:
            status, steps = np.where(steps > H, ST_HORIZON, status), np.minimum(steps, H)
        secs = 1e-5 * (5 if slow else 1) * steps.max() * 4 + 0.001
        cut_rows = 0
        if d.cap_s is not None and secs > d.cap_s:        # the deadline cuts the call (CPU: rows not started projected)
            cut_rows, secs = 1, d.cap_s
        prof = dict(kind="sequential", seconds=secs, H=H, cap_s=d.cap_s, cap_hit=cut_rows > 0, cut_rows=cut_rows)
        rc.add_call(steps=steps, status=status, keys=np.array([90.0, 100, 100, 100]), feasible=np.ones(4, bool),
                    profile=prof, cutoff=100.0, site="children", horizon=H, cap_s=d.cap_s, cap_hits=cut_rows)
        now += secs + 0.01
        rc.end_generation(now, secs + 0.01)
        if slow and d.level == 0:
            cuts.append(cut_rows)
            caps.append(d.cap_s)
    t128 = 5e-5 * 128 * 4 + 0.001                         # the slow call's time at 128
    assert rc.cur == 0 and len(cuts) > 200
    assert 0 < sum(cuts) <= 40 and sum(cuts[-100:]) == 0  # the cap caught up and the calls finish again
    assert caps[0] < t128 < caps[-1] <= 3.0 * t128 * (1 + 1e-9)


def test_calls_after_the_survival_are_left_out_of_the_gain():
    """A call marked counterfactual=False (an immigrant call after the survival): its gain is not in Δ but reported as
    `gain_approx`, its seconds count in o_g, and its cut does not make levels unobserved."""
    rc = RepairClock(rc_cfg(level_min_log2=7, level_max_log2=9))   # 128, 256, 512, none
    rc.decide(0.0, True)
    rc.add_call(**seq_call(NOTHING, seconds=0.5))
    imm = seq_call([(100, ST_FEASIBLE, 0.3), (128, ST_HORIZON, 0.0)], seconds=0.2, site="admit", H=128)
    rc.add_call(counterfactual=False, **imm)
    ev = rc.end_generation(1.0, 1.0)
    assert ev["gain"] == [0.0] * rc.L and ev["gain_approx"] == pytest.approx(0.3)
    assert ev["o_s"] == pytest.approx(0.5) and all(ev["used"])
    assert ev["T"][-1] == pytest.approx(1.0) and ev["best"] is None
    assert [c["cf"] for c in ev["calls"]] == [True, False]
    assert np.all(rc.n == 1) and np.all(rc.G == 0)


def test_shadow_mode_accumulates_what_the_decision_would_reveal():
    """Shadow, the would-level at 128 while every call runs unbounded: only the levels the would-call reveals (<= 128,
    its lanes beyond 128 would have been cut) are accumulated, so "none" goes unobserved and probes fire as in active
    mode, within the probe share; the event keeps the full-information gain and time and marks what was used."""
    def run(mode):
        rc = RepairClock(rc_cfg(mode=mode))
        now, probes, used = 0.0, 0, []
        for g in range(200):
            rc.decide(now, True, gen=g)
            H = rc.g["play"].horizon
            steps, status = np.array([100, 6000, 6000, 6000]), np.array([1, 2, 2, 2])
            if H is not None:                             # active at a level: lanes beyond H are cut there
                status, steps = np.where(steps > H, ST_HORIZON, status), np.minimum(steps, H)
            prof = dict(kind="sequential", seconds=0.5 * float(steps.sum()) / 18100, H=H, cap_s=None, cap_hit=False,
                        cut_rows=0)
            rc.add_call(steps=steps, status=status, keys=np.array([90.0, 100, 100, 100]), feasible=np.ones(4, bool),
                        profile=prof, cutoff=100.0, site="children")
            now += 0.6
            ev = rc.end_generation(now, 0.6)
            probes += int(ev["probe"])
            used.append((ev["would"], ev["used"], ev["T"]))
        return rc, probes, used
    rs, ps, us = run("shadow")
    ra, pa, _ = run("active")
    assert rs.cur == ra.cur == 0 and ps == pa >= 2
    share = rs.record()["probe_time_share"]
    assert share == pytest.approx(ra.record()["probe_time_share"]) and 0 < share <= 0.1 + 0.6 / 120
    at128 = [(u, T) for w, u, T in us if w == 128]
    assert at128 and all(u == [True] + [False] * (rs.L - 1) for u, _ in at128)
    assert all(T[-1] is not None for _, T in at128)       # shadow's calls still showed every level (logged)


def test_pinned_call_counts_for_every_level():
    """The one-shot LP batch runs at "none" under every level: its best gain and its whole time count for every
    level, and it does not make a level unobserved."""
    rc = RepairClock(rc_cfg())
    rc.decide(0.0, True)
    rc.add_call(**seq_call(NOTHING, seconds=0.5))
    rc.add_call(**seq_call([(30000, ST_FEASIBLE, 0.2)], seconds=2.0, site="lp", pinned=True))
    ev = rc.end_generation(3.0, 3.0)
    assert ev["gain"] == pytest.approx([0.2] * rc.L)
    t_child = np.array([0.5 * min(h, 40000.0) / 40000.0 for h in rc.hz])
    np.testing.assert_allclose(ev["T"], 0.5 + 2.0 + t_child)   # o = 3 - 2.5; the pinned 2 s at every level
    assert [c["pinned"] for c in ev["calls"]] == [False, True]


def test_call_times_and_what_a_cut_call_reveals():
    rc = RepairClock(rc_cfg(level_min_log2=5, level_max_log2=9))   # levels 32, 64, 128, 256, 512, none
    chunk = dict(k=4, cum=[0, 8, 40, 168, 680], t=[0.0, 0.01, 0.02, 0.06, 0.16], end=0.165, H_eff=680, done=True,
                 cut=False, cap_hit=False)
    assert _chunk_time(chunk, 128) == pytest.approx(0.02 + (128 - 40) / 128 * 0.04 + 0.005)
    assert _chunk_time(chunk, 1e9) == pytest.approx(0.165)
    steps = np.array([600, 680, 30, 5])
    prof = dict(kind="launches", seconds=0.17, H=None, cap_s=None, chunks=[chunk])
    t, obs = rc._call_times(prof, steps, np.array([2, 2, 1, 1]))
    assert obs.all()                                      # every lane stopped by itself: every level observed
    assert t[-1] == pytest.approx(0.17) and t[2] == pytest.approx(0.005 + _chunk_time(chunk, 128))
    cutc = dict(chunk, cum=[0, 8, 40, 168, 256], t=[0.0, 0.01, 0.02, 0.06, 0.08], end=0.085, H_eff=256, done=False,
                cut=True)
    t, obs = rc._call_times(dict(prof, H=256, seconds=0.09, chunks=[cutc]), np.array([256, 256, 30, 5]),
                            np.array([4, 4, 1, 1]))
    assert obs.tolist() == [True, True, True, True, False, False]   # a call cut at 256 reveals the levels <= 256
    assert np.isnan(t[4]) and t[3] == pytest.approx(0.09)
    _, obs = rc._call_times(prof, np.array([600, 680, 30, 5, 0, 0]), np.array([2, 2, 1, 1, 0, 0]))
    assert not obs.any()                                  # a run-level stop left a chunk unreached: nothing observed
    # a sequential call: cut rows hide everything; a horizon hides the levels above it
    st = np.array([1, 2, 4])
    _, obs = rc._call_times(dict(kind="sequential", seconds=1.0, H=None, cut_rows=1), np.array([3, 9, 0]), st)
    assert not obs.any()
    _, obs = rc._call_times(dict(kind="sequential", seconds=1.0, H=64, cut_rows=0), np.array([3, 9, 64]), st)
    assert obs.tolist() == [True, True, False, False, False, False]
    # ... but a call with a horizon none of whose lanes was cut reveals every level, as on the GPU
    t, obs = rc._call_times(dict(kind="sequential", seconds=1.0, H=64, cut_rows=0), np.array([3, 9, 64]),
                            np.array([1, 2, 2]))
    assert obs.all() and t[-1] == pytest.approx(1.0) and t[0] == pytest.approx((3 + 9 + 32) / 76)


def test_gain_unit_and_lane_filters():
    """dg = min((cutoff - key) / max(|cutoff|, |key|, |dual| if finite, 1e-9), 1); Δ counts the lanes FEASIBLE in the
    search AND feasible in the evaluation (after the decoder); a lane above its level's horizon does not count there.
    A lane that beats the cutoff under the evaluation but stalled is logged apart (`beat_stopped`), not counted."""
    rc = RepairClock(rc_cfg())
    rc.decide(0.0, True)
    steps = np.array([100, 200, 300, 5000])
    status = np.array([1, 1, 2, 1])
    keys = np.array([90.0, 80.0, 50.0, 60.0])
    feas = np.array([True, False, True, True])            # lane 1 failed the evaluation; lane 2 stalled
    prof = dict(kind="sequential", seconds=1.0, H=None, cap_s=None, cap_hit=False, cut_rows=0)
    rc.add_call(steps=steps, status=status, keys=keys, feasible=feas, profile=prof, cutoff=100.0, dual_key=-200.0)
    ev = rc.end_generation(1.0, 1.0)
    D = 200.0
    assert ev["gain"][0] == pytest.approx(10 / D)                     # 128: lane 0 only
    assert ev["gain"][rc.hz.tolist().index(8192)] == pytest.approx(40 / D)   # lane 3 counts from 8192 up
    assert ev["best"] == dict(site="", steps=5000, dg=pytest.approx(40 / D))
    c = ev["calls"][0]
    assert c["cutoff"] == 100.0 and c["dual_key"] == -200.0 and c["cf"]
    assert c["beat_stopped"] == [1, pytest.approx(50 / D)] and c["beat_cut"] == [0, None]
    assert [g[2] for g in c["lane_gains"]] == [2, 1, 1]                 # [steps, dg, status], largest dg first
    assert max(v for v in ev["gain"] if v is not None) == pytest.approx(40 / D)   # the stalled lane's 50 is not in Δ


@pytest.mark.parametrize("cutoff, dual, key, want", [
    (0.0, -math.inf, -0.5, 1.0),          # incumbent 0, no dual bound: was 5e8
    (0.0, -math.inf, -90.0, 1.0),         # neos-960392's first step: was 9e10
    (1000.0, -math.inf, 995.0, 0.005),
    (10.0, -math.inf, -10.0, 1.0),        # opposite signs: 1 (Berthold), not 2
    (5.0, -math.inf, 2.0, 0.6),
    (-100.0, -200.0, -110.0, 0.05),       # the dual bound's size in the denominator
    (-100.0, -math.inf, -400.0, 0.75),    # the point's own size
])
def test_gain_unit_is_bounded_like_the_primal_gap(cutoff, dual, key, want):
    """dg is in [0, 1] whatever the cutoff (P(T)'s gap unit): the point itself is in the denominator and a sign change
    gives 1, so one improvement over an incumbent near 0 without a dual bound cannot dominate the rates."""
    rc = RepairClock(rc_cfg())
    rc.decide(0.0, True)
    prof = dict(kind="sequential", seconds=1.0, H=None, cap_s=None, cap_hit=False, cut_rows=0)
    rc.add_call(steps=np.array([100]), status=np.array([ST_FEASIBLE]), keys=np.array([key]),
                feasible=np.array([True]), profile=prof, cutoff=cutoff, dual_key=dual)
    ev = rc.end_generation(1.0, 1.0)
    assert ev["gain"] == pytest.approx([want] * rc.L)


# ============================================================================== the engine
def _summary_cols(out: dict) -> dict:
    return {k: out[k] for k in REF.SUMMARY_KEYS}


def test_shadow_mode_equals_the_disabled_run():
    """Shadow mode plays "none" with no cap and draws no random number: genealogy, final population, counters and best
    key equal the run with the section off, on every reference toy (LP batch, stagnation and frontier immigrants)."""
    for name in REF.TOYS:
        off = REF.capture(name, "numpy")
        on = REF.capture(name, "numpy", cfg=REF.config("numpy", **{"repair_clock.enabled": True}))
        assert off.keys() == on.keys()
        for k in off:
            assert np.array_equal(off[k], on[k], equal_nan=True), (name, k)


def test_disabled_records_nothing_new(tmp_path):
    p = toy.knapsack(30)
    cfg = REF.config("numpy", log_every=1)
    eng = Engine(p, cfg, seed=3, trace=Trace(tmp_path, "ea"), warmup=False)
    eng.prepare()
    assert "repair_clock" not in eng.startup
    eng.begin(time.monotonic(), 3600.0)
    for _ in range(4):
        eng.step()
    out = eng.finish()
    eng.trace.close()
    assert eng.rclock is None and "repair_clock" not in out
    assert "ls_finished_horizon" not in out and "ls_cap_hits" not in out
    ev = read_events(tmp_path / "ea.events.jsonl")
    assert not any(e["kind"] == "rclock" for e in ev)
    assert not any(k.startswith("rc_") for e in ev if e["kind"] == "gen" for k in e)


def _force_level(eng: Engine, level: int) -> None:
    """Tests: keep the controller at `level` after the first post-feasible generation (no probe, no switch)."""
    rc = eng.rclock
    rc._choose = lambda: level
    rc._probe_due = lambda now: False


def test_active_mode_honours_the_horizon(tmp_path):
    """Active, the controller held at h = 8 lane steps after the first post-feasible generation: every repair call of
    those generations (children and immigrants) runs at most 8 steps per lane, lanes cut there have ST_HORIZON, and
    the records carry the level; generation 0 (no feasible member yet) plays "none"."""
    p = toy.set_cover(40, 25)
    cfg = REF.config("numpy", log_every=1, **{"repair_clock.enabled": True, "repair_clock.mode": "active",
                                              "repair_clock.level_min_log2": 3, "repair.stall_steps": 400})
    eng = Engine(p, cfg, seed=17, trace=Trace(tmp_path, "ea"), warmup=False)
    eng.prepare()
    assert eng.startup["repair_clock"]["levels"][0] == 8
    eng.begin(time.monotonic(), 3600.0)
    _force_level(eng, 0)
    for _ in range(6):
        eng.step()
    out = eng.finish()
    eng.trace.close()
    ev = read_events(tmp_path / "ea.events.jsonl")
    rcl = [e for e in ev if e["kind"] == "rclock"]
    assert [e["gen"] for e in rcl] == list(range(7))
    assert rcl[0]["phase"] == "pre" and rcl[0]["level"] == "none"
    played = [e for e in rcl if e["level"] == 8]
    assert len(played) >= 4
    cut = 0
    for e in played:
        assert e["cap"] is not None and e["cap"] <= 10.0
        for c in e["calls"]:
            assert c["H"] == 8 and max(c["steps"]) <= 8
            cut += c["status"].count("4")
    assert cut > 0 and out["ls_finished_horizon"] == cut
    assert out["repair_clock"]["time_share_played"][0] > 0.5
    gens = [e for e in ev if e["kind"] == "gen"]
    assert gens[-1]["rc_level"] == 8 and gens[-1]["rc_mode"] == "active" and gens[-1]["rc_H_eff"][1] <= 8


def test_shadow_mode_plays_none_while_its_choice_differs(tmp_path):
    """Shadow, the controller held at h = 8 lane steps (as in the active test): the choice differs from what is played,
    yet no local-search call gets an argument, no lane is cut, and genealogy and population equal the run with the
    section off. (Unforced, the reference toys never leave "none", so this is the test that sees a leak.)"""
    p = toy.set_cover(40, 25)

    def run(over, tdir):
        cfg = REF.config("numpy", log_every=1, **dict({"repair.stall_steps": 400}, **over))
        eng = Engine(p, cfg, seed=17, trace=Trace(tdir, "ea"), warmup=False)
        eng.prepare()
        log = _spy_ls(eng)
        eng.begin(time.monotonic(), 3600.0)
        if eng.rclock is not None:
            _force_level(eng, 0)
        for _ in range(6):
            eng.step()
        out = eng.finish()
        eng.trace.close()
        g = eng.genealogy()
        cols = {c: np.asarray(g[c]) for c in REF.GEN_COLS}
        cols.update(pop_ids=np.asarray(eng.pop.ids), pop_key=np.asarray(eng.h_key, np.float64))
        return out, log, cols
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    out, log, on = run({"repair_clock.enabled": True, "repair_clock.mode": "shadow",
                        "repair_clock.level_min_log2": 3}, tmp_path / "on")
    _, _, off = run({}, tmp_path / "off")
    assert log and all(kw == {} for _, _, kw in log)
    assert out["ls_finished_horizon"] == 0 and out["ls_cap_hits"] == 0
    rcl = [e for e in read_events(tmp_path / "on" / "ea.events.jsonl") if e["kind"] == "rclock"]
    differ = [e for e in rcl if e["would"] == 8 and e["level"] == "none"]
    assert len(differ) >= 4 and all(e["cap"] is None and e["would_cap"] is not None for e in differ)
    assert all("4" not in c["status"] and c["H"] is None and c["cap"] is None for e in rcl for c in e["calls"])
    for k in off:
        assert np.array_equal(off[k], on[k], equal_nan=True), k


class _Chan:
    def __init__(self, msgs=None, delay: int = 0):
        self.msgs, self.delay, self.polls = list(msgs or []), delay, 0

    def poll(self):
        self.polls += 1
        if self.polls <= self.delay or not self.msgs:
            return None
        return self.msgs.pop(0)


class _Mailbox:
    """A partner with an optional info message (solver incumbent) and an optional LP point delivered at poll `delay`."""

    def __init__(self, info=None, lp=None, lp_delay: int = 0):
        self.to_ea = _Chan()
        self.info = _Chan([info] if info else [])
        self.lp = _Chan([lp] if lp is not None else [], delay=lp_delay)
        self.posted = []

        outer = self

        class _To:
            def post(_s, X, obj, origin=None):
                outer.posted.append(np.asarray(X))
                return len(outer.posted)
        self.to_solver = _To()

    def stopped(self) -> bool:
        return False

    def verdicts(self) -> list:
        return []


def _spy_ls(eng: Engine) -> list:
    log = []
    orig = eng.ls.run

    def spy(X, **kw):
        log.append((eng.rc_site, eng.gen, {k: v for k, v in kw.items() if k in ("horizon", "cap_s")}))
        return orig(X, **kw)
    eng.ls.run = spy
    return log


def test_pre_feasible_phase_holds_under_a_finite_solver_incumbent():
    """An infeasible toy (no feasible member ever) with a solver incumbent from generation 0: the cutoff is armed, but
    every call is played at "none" without a cap (active mode)."""
    n = 20
    rs = np.random.default_rng(0)
    p = from_arrays(sps.csr_matrix(np.ones((1, n))), [n + 5.0], [np.inf], np.zeros(n), np.ones(n),
                    rs.integers(1, 9, n).astype(float), np.ones(n, bool), names=[f"x{j}" for j in range(n)])
    cfg = REF.config("numpy", **{"repair_clock.enabled": True, "repair_clock.mode": "active",
                                 "exchange.poll_s": 1e-6})
    eng = Engine(p, cfg, seed=2, mailbox=_Mailbox(info=dict(incumbent=30.0, dual_bound=0.0)), warmup=False)
    eng.prepare()
    log = _spy_ls(eng)
    eng.begin(time.monotonic(), 3600.0)
    for _ in range(6):
        eng.step()
    assert eng.cutoff == pytest.approx(30.0) and math.isfinite(eng.inc_external)
    assert not eng.h_feas.any() and not eng.rclock.post and eng.rclock.gens_pre == 7
    assert log and all(kw == {} for _, _, kw in log)


def test_an_injected_feasible_solver_row_keeps_the_pre_feasible_phase():
    """A subset-sum toy the EA cannot solve in a step (max_steps = stall_steps = 1) with the solution injected by the
    solver at generation 0: the population holds a feasible member, but not one of the EA's own, so every call stays at
    "none" without a cap (active mode); the phase ends one generation after the first feasible member of the EA's own
    (a member that is not an unchanged solver solution)."""
    n = 40
    rs = np.random.default_rng(3)
    a = rs.integers(100, 1000, n).astype(float)
    xs = (rs.random(n) < 0.5).astype(float)
    b = float(a @ xs)
    p = from_arrays(sps.csr_matrix(a[None, :]), [b], [b], np.zeros(n), np.ones(n), rs.integers(1, 9, n).astype(float),
                    np.ones(n, bool), names=[f"x{j}" for j in range(n)])
    cfg = REF.config("numpy", **{"repair_clock.enabled": True, "repair_clock.mode": "active",
                                 "exchange.poll_s": 1e-6, "repair.max_steps": 1, "repair.stall_steps": 1})
    mb = _Mailbox()
    mb.to_ea = _Chan([Message(X=xs[None, :], obj=np.array([float(p.objective(xs))]), seq=1, t=0.0)])
    eng = Engine(p, cfg, seed=2, mailbox=mb, warmup=False)
    eng.prepare()
    log = _spy_ls(eng)
    eng.begin(time.monotonic(), 3600.0)
    assert eng.h_feas.any() and not eng.rclock.post       # the solver's row, feasible from generation 0
    g_own = None
    for _ in range(10):
        eng.step()
        own = [i for i in np.flatnonzero(eng.h_feas) if not _is_solver_find(int(eng.pop.origin[i]))]
        if own:
            g_own = eng.gen
            break
        assert eng.h_feas.any() and not eng.rclock.post and eng.rc_dec.phase == "pre"
    assert g_own is not None and g_own >= 2               # at least one generation with the solver's row alone
    assert all(kw == {} for _, g, kw in log if g <= g_own)
    eng.step()                                            # the next decision sees the EA's own feasible member
    assert eng.rclock.post and eng.rc_dec.phase == "post"
    assert all(kw.get("cap_s") == 10.0 for _, g, kw in log if g == g_own + 1)


def test_lp_batch_is_pinned_at_none():
    """Active, held at a shallow level: the LP batch that arrives after the first feasible member is repaired with no
    horizon and no cap, the children of the same generation with the level's horizon."""
    p = toy.knapsack(30)
    lp = REF._lp_point(p)
    msg = Message(X=lp[None, :], obj=np.array([float(p.objective(lp))]), seq=1, t=0.0)
    cfg = REF.config("numpy", **{"repair_clock.enabled": True, "repair_clock.mode": "active",
                                 "exchange.poll_s": 1e-6})
    eng = Engine(p, cfg, seed=4, mailbox=_Mailbox(lp=msg, lp_delay=4), warmup=False)
    eng.prepare()
    log = _spy_ls(eng)
    eng.begin(time.monotonic(), 3600.0)
    _force_level(eng, 2)                                  # 2^9 = 512
    for _ in range(6):
        eng.step()
    lp_calls = [(g, kw) for site, g, kw in log if site == "lp"]
    assert len(lp_calls) == 1 and lp_calls[0][1] == {}
    g_lp = lp_calls[0][0]
    assert g_lp >= 2 and eng.rclock.post
    child = [kw for site, g, kw in log if site == "children" and g == g_lp]
    assert child and child[0]["horizon"] == 512 and child[0]["cap_s"] > 0
    eng.rc_site = "lps"
    assert not eng._rc_pinned() and eng._rc_call_args()["horizon"] == 512   # LP-stream batches follow the level


@pytest.mark.parametrize("name", ["mixed", "set_cover"])
def test_pairing_helper_maps_lanes_onto_the_evaluated_rows(monkeypatch, name):
    """At every call site — initial population, children, the LP batch, `_admit` (whose call repairs only its seeded
    rows; frontier rows are not repaired) — the keys and feasibility handed to the clock are those of the call's own
    rows after the repair (and the decoder: toy.mixed has continuous variables), lane for lane."""
    p = {"mixed": toy.mixed, "set_cover": lambda: toy.set_cover(40, 25)}[name]()
    lp = REF._lp_point(p)
    cfg = REF.config("numpy", **{"repair_clock.enabled": True})
    eng = Engine(p, cfg, seed=17, mailbox=REF.LPMailbox(lp, float(p.objective(lp))), warmup=False)
    eng.prepare()
    repaired, added, rows_seen = [], [], []
    orig_repair, orig_pair, orig_add = eng._repair, eng._rc_pair, RepairClock.add_call

    def repair_spy(X, strat=None):
        moved = orig_repair(X, strat)
        repaired.append((eng.rc_site, np.array(eng.bk.asnumpy(X), np.float64, copy=True)))
        return moved

    def pair_spy(key, feasible, rows=None):
        rows_seen.append((eng.rc_site, None if rows is None else len(rows), len(np.asarray(key))))
        return orig_pair(key, feasible, rows)

    def add_spy(self, **kw):
        added.append(kw)
        return orig_add(self, **kw)
    eng._repair, eng._rc_pair = repair_spy, pair_spy
    monkeypatch.setattr(RepairClock, "add_call", add_spy)
    eng.begin(time.monotonic(), 3600.0)
    for _ in range(12):
        eng.step()
    assert len(added) == len(repaired) and len(added) >= 14
    for (site, X), kw in zip(repaired, added):
        assert kw["site"] == site and kw["steps"].shape[0] == X.shape[0] == kw["keys"].shape[0]
        ev = evaluate(eng.dp, X)
        np.testing.assert_array_equal(kw["keys"], np.asarray(ev.key, np.float64))
        np.testing.assert_array_equal(kw["feasible"], np.asarray(ev.feasible, bool))
    sites = [s for s, _ in repaired]
    assert sites[0] == "init" and {"children", "lp", "admit"} <= set(sites)
    # the immigrant calls after the survival are marked: the clock leaves them out of the gain
    assert all(kw["counterfactual"] == (kw["site"] != "admit") for kw in added)
    if name == "set_cover":                               # stagnation immigrants with frontier rows: a strict subset
        assert any(s == "admit" and r is not None and r < k for s, r, k in rows_seen)


def test_rc_pair_refuses_a_mismatch():
    p = toy.knapsack(30)
    eng = Engine(p, REF.config("numpy", **{"repair_clock.enabled": True}), seed=1, warmup=False)
    eng.prepare()
    eng.begin(time.monotonic(), 3600.0)
    eng._rc_last = dict(steps=np.zeros(3, np.int64), status=np.ones(3, np.int64), profile=None, site="children",
                        pinned=False, horizon=None, cap_s=None, cap_hits=0)
    with pytest.raises(RuntimeError, match="lanes"):
        eng._rc_pair(np.zeros(4), np.ones(4, bool))


# ============================================================================== configuration
def test_repair_clock_validation():
    base = EAConfig()
    assert not base.repair_clock.enabled and base.repair_clock.mode == "shadow"
    for bad in ({"repair_clock.mode": "on"}, {"repair_clock.level_min_log2": 9, "repair_clock.level_max_log2": 8},
                {"repair_clock.level_max_log2": 21}, {"repair_clock.level_min_log2": -1},
                {"repair_clock.min_gens": 0}, {"repair_clock.min_events": 0}, {"repair_clock.probe_share": 0.0},
                {"repair_clock.probe_share": 1.0}, {"repair_clock.cap_s": 0.0}, {"repair_clock.cap_factor": 0.0},
                {"repair_clock.half_life_s": 0.0}, {"repair_clock.materiality": -1e-3},
                {"repair_clock.hysteresis": -0.1}, {"repair_clock.enabled": 1},
                {"repair_clock.enabled": True, "repair.method": "dense"},
                {"repair_clock.enabled": True, "repair.objective_mode": "weighted"}):
        with pytest.raises(ValueError, match="repair_clock"):
            base.with_overrides(bad)
    ok = base.with_overrides({"repair_clock.enabled": True, "repair_clock.mode": "active"})
    assert ok.repair_clock.enabled


# ============================================================================== GPU (marker gpu; run in a GPU window)
@pytest.mark.gpu
def test_shadow_mode_equals_the_disabled_run_cupy():
    from test_engine import gpu_or_skip
    gpu_or_skip()
    for name in REF.TOYS:
        off = REF.capture(name, "cupy")
        on = REF.capture(name, "cupy", cfg=REF.config("cupy", **{"repair_clock.enabled": True}))
        for k in off:
            assert np.array_equal(off[k], on[k], equal_nan=True), (name, k)


@pytest.mark.gpu
def test_active_mode_honours_the_horizon_cupy(tmp_path):
    """The engine on the GPU with the controller held at h = 8: every call of those generations stops its lanes at 8
    steps at most, cut lanes have ST_HORIZON, the launch profile reaches the clock."""
    from test_engine import gpu_or_skip
    gpu_or_skip()
    p = toy.set_cover(40, 25)
    cfg = REF.config("cupy", log_every=1, **{"repair_clock.enabled": True, "repair_clock.mode": "active",
                                             "repair_clock.level_min_log2": 3, "repair.stall_steps": 400})
    eng = Engine(p, cfg, seed=17, trace=Trace(tmp_path, "ea"), warmup=False)
    eng.prepare()
    eng.begin(time.monotonic(), 3600.0)
    _force_level(eng, 0)
    for _ in range(6):
        eng.step()
    out = eng.finish()
    eng.trace.close()
    rcl = [e for e in read_events(tmp_path / "ea.events.jsonl") if e["kind"] == "rclock"]
    played = [e for e in rcl if e["level"] == 8]
    assert len(played) >= 4 and out["ls_finished_horizon"] > 0
    for e in played:
        for c in e["calls"]:
            assert c["H"] == 8 and max(c["steps"]) <= 8 and c["prof"]["kind"] == "launches"
            assert all(ch["cum"][-1] <= 8 for ch in c["prof"]["chunks"])
