"""The repair clock: one lane-step horizon per generation for every repair call, chosen from what the calls showed.

Design: vault `80-reviews/2026-10-06 Repair-depth design (workflow).md` ("Plan (synthesis, verbatim)"), configuration
`RepairClockCfg` (`repair_clock`, off by default). Host-side bookkeeping only: no backend, no device array, no random
number — the engine (`ea.engine`) asks for a level, passes it to the local search and hands back each call's per-lane
results paired with the evaluated keys.

    rc = RepairClock(cfg)
    dec = rc.decide(now, has_feasible)         # after the exchange tick: the level of this generation (played)
    rc.add_call(steps=..., status=..., keys=..., feasible=..., profile=..., cutoff=..., dual_key=..., site=...,
                counterfactual=...)
    ev = rc.end_generation(now, gen_seconds)   # the generation's per-level gain and time; the next level; the event
    rc.record()                                # the summary block

Levels. h_j = 2^j lane steps, j = level_min_log2 .. level_max_log2, plus "none" (no horizon: the repair as configured).
A level is in force for a whole generation and applies to every repair call of it (children, pending batches,
immigrants) except the one-shot LP batch (`pin_lp`: always "none", uncapped).

Phase. Until the population holds a feasible member of the EA's own (`has_feasible`: the engine counts a feasible
member unless it is an unchanged solver solution — origin INJECTED and no other bit, the rule of `_is_solver_find`;
one-way) every call is played at "none" with no cap — the run is the one without the clock. Neither a finite solver
incumbent nor an injected feasible solver row ends this phase (the cutoff may be armed by them; glass4 has one at
generation 0). Nothing is accumulated in this phase.

Gain of level h in generation g (the counterfactual — exact for the children, the pending batches and an admit before
the children, because lanes are independent, the cutoff is constant over a generation's calls and the stop rules are
checked before the horizon, so a horizon-h call is a prefix of the unbounded one): Δ_g(h) = max dg over the lanes that
ended FEASIBLE with S_STEPS <= h and are feasible under the evaluation (after the decoder), dg = min((cutoff − key) /
max(|cutoff|, |key|, |dual_key| if finite, 1e-9), 1) — the primal-gap unit of P(T), in [0, 1] like Berthold's gap
(the point itself in the denominator; a cutoff and a key of opposite signs give 1). This bound is a deliberate change
from the synthesis plan's max(|cutoff|, |dual_key|, 1e-9), which gave one improvement over a cutoff near 0 without a
dual bound a weight of ~1e9. A Δ below `materiality` counts as zero. Lanes that beat the cutoff under the evaluation
without ending FEASIBLE in the search (stopped by stall / budget, or cut) do not count; each call logs their number
and best dg. The immigrant calls after the survival (shortfall fill, stagnation: `counterfactual=False`) exist and
have their inputs only because of the generation's earlier, level-dependent calls: they are left out of Δ and of what
is observed, their seconds count in o_g, and the event reports their best gain apart (`gain_approx`).
Level h is observed in g if no counterfactual call of g was cut below h (a call cut at H_eff, by the horizon or the
deadline, reveals the levels <= H_eff only; a call none of whose lanes was cut reveals every level).
Time of level h: T_g(h) = o_g + Σ_calls t_call(h), o_g = generation seconds − the counterfactual calls' measured
seconds; t_call(h) from the call's profile — GPU (`ls_fj_gpu`, kind "launches"): the (cumulative steps, seconds) pair
after every launch, interpolated linearly inside a launch, per chunk; CPU (`ls_fj_cpu`, kind "sequential"): seconds ·
Σ min(steps, h) / Σ steps. A pinned call (the LP batch) runs at "none" under every level: its gain and its full time
count for every level.

Choice (post phase, after each generation). Per level the decayed sums G = Σ w_g Δ_g, T = Σ w_g T_g (w halves every
`half_life_s` seconds), n = observed generations, events = generations with a material gain; rate = G / T. Eligible:
n >= min_gens, events >= min_events. No eligible level: "none". Else the eligible level of highest rate (the deeper
one on a tie) replaces the current one only if its rate is above (1 + hysteresis) × the current level's rate.
Probes: away from "none", while the probe generations took less than `probe_share` of the post-phase time and "none"
was not observed within one half-life, the next generation plays "none" — one unbounded call re-measures every level.
Cap (post phase only; the pinned LP batch has none): min(cap_s, cap_factor × the median of the level's recent
children-call times t(h)); "none": cap_s. The children call is the reference because one median over calls of every
size (4-row stream batches beside 256-row children calls) would cut the children call at a small batch's pace. A
children call the played cap cut below its level adds a censored time, max(its seconds, the cap), so the median can
rise (by up to cap_factor per window of calls) when the calls become slower; otherwise cut calls would never enter it.

Modes. `active` plays the decision; `shadow` plays "none" with no cap — no argument reaches the local search — and
accumulates and logs the same decisions. Shadow mode accumulates only what the decision would have revealed: at a
level h, the levels <= h, or every level if no lane of a call ran beyond h; so its probes fire as in active mode (its
probe time is the real time of the generations it would have probed). The event's `gain` / `T` keep everything the
played calls showed; `used` marks the levels accumulated.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from itrea.config import RepairClockCfg

ST_FEASIBLE, ST_STALL, ST_BUDGET, ST_HORIZON = 1, 2, 3, 4   # the local searches' status codes; 0 = interrupted
CALL_WINDOW = 32                   # recent children-call times per level kept for the cap's median
LANE_GAINS = 64                    # per call, at most this many [steps, dg, status] of improving lanes go to the event


@dataclass(frozen=True)
class Decision:
    level: int                     # index into RepairClock.H; RepairClock.NONE = "none"
    horizon: int | None            # lane steps (None: no horizon)
    cap_s: float | None            # per-call deadline in seconds (None: no cap)
    probe: bool                    # a probe of "none"
    phase: str                     # "pre" (before the first feasible member) | "post"


class RepairClock:
    """The controller of the module docstring (one per run)."""

    def __init__(self, cfg: RepairClockCfg):
        self.cfg = cfg
        self.H = [1 << j for j in range(int(cfg.level_min_log2), int(cfg.level_max_log2) + 1)]
        self.L = len(self.H) + 1
        self.NONE = self.L - 1
        self.hz = np.array([float(h) for h in self.H] + [math.inf])
        self.active = cfg.mode == "active"
        L = self.L
        self.G = np.zeros(L)                          # decayed gains
        self.T = np.zeros(L)                          # decayed seconds
        self.n = np.zeros(L, np.int64)                # observed post-phase generations
        self.events = np.zeros(L, np.int64)           # ... with a material gain
        self.last_obs = np.full(L, -math.inf)         # when each level was last observed (post phase)
        self.child_t = [deque(maxlen=CALL_WINDOW) for _ in range(L)]
        self.t_decay: float | None = None
        self.cur = self.NONE
        self.post = False
        self.t_post: float | None = None
        self.probe_s = 0.0
        self.switches = 0
        self.time_played = np.zeros(L)
        self.time_would = np.zeros(L)
        self.gens_pre = 0
        self.gens_post = 0
        self.cap_hits = 0                             # calls the played cap cut
        self.cap_hits_would = 0                       # generations whose decision's cap a call would have exceeded
        self.heff = np.zeros(L, np.int64)             # calls by H_eff (largest lane step count): smallest level >= it
        self.g: dict | None = None
        self._since = self._fresh_since()

    # ------------------------------------------------------------------------------------------------- labels
    def label(self, level: int):
        return "none" if level == self.NONE else int(self.H[level])

    def labels(self) -> list:
        return [self.label(j) for j in range(self.L)]

    def describe(self) -> dict:
        c = self.cfg
        return dict(mode=c.mode, levels=self.labels(), min_gens=c.min_gens, min_events=c.min_events,
                    materiality=c.materiality, half_life_s=c.half_life_s, hysteresis=c.hysteresis,
                    probe_share=c.probe_share, cap_s=c.cap_s, cap_factor=c.cap_factor, pin_lp=c.pin_lp)

    # ------------------------------------------------------------------------------------------------- decide
    def decide(self, now: float, has_feasible: bool, gen: int | None = None) -> Decision:
        """The level of the generation that starts now (seconds since t0); returns the decision to PLAY (shadow:
        always "none" without a cap). An unfinished previous generation (cut short) is discarded."""
        if has_feasible and not self.post:
            self.post, self.t_post = True, float(now)
        if not self.post:
            would = Decision(self.NONE, None, None, False, "pre")
        else:
            lvl, probe = self.cur, False
            if self._probe_due(now):
                lvl, probe = self.NONE, True
            would = Decision(lvl, None if lvl == self.NONE else int(self.H[lvl]), self._cap(lvl), probe, "post")
        play = would if self.active else Decision(self.NONE, None, None, False, would.phase)
        self.g = dict(gen=gen, t=float(now), would=would, play=play, calls=[])
        return play

    def _probe_due(self, now: float) -> bool:
        c = self.cfg
        if self.cur == self.NONE:
            return False
        if self.probe_s >= c.probe_share * max(now - self.t_post, 0.0):
            return False
        return now - self.last_obs[self.NONE] > c.half_life_s

    def _cap(self, level: int) -> float:
        c = self.cfg
        ct = self.child_t[level]
        if level == self.NONE or not ct:
            return float(c.cap_s)
        return float(min(c.cap_s, c.cap_factor * float(np.median(np.asarray(ct)))))

    # ------------------------------------------------------------------------------------------------- calls
    def add_call(self, *, steps, status, keys, feasible, profile: dict | None, cutoff: float | None,
                 dual_key: float = -math.inf, site: str = "", pinned: bool = False, horizon: int | None = None,
                 cap_s: float | None = None, cap_hits: int = 0, counterfactual: bool = True) -> None:
        """One repair call of the current generation, its lanes paired with the evaluated rows (lane i = row i):
        `steps`, `status` from the local search (`last_steps`, `last_status`), `keys`, `feasible` from the evaluation
        after the decoder, `profile` = the search's `last_profile`; `cutoff` / `dual_key` = the incumbent key the
        call's cutoff row was armed with and the solver's proven bound (keys); `counterfactual` False for a call made
        after the survival (module docstring: left out of Δ and of what is observed)."""
        if self.g is None:
            return
        steps = np.asarray(steps, np.int64).reshape(-1)
        status = np.asarray(status, np.int64).reshape(-1)
        keys = np.asarray(keys, np.float64).reshape(-1)
        feasible = np.asarray(feasible, bool).reshape(-1)
        k = steps.size
        if not (status.size == keys.size == feasible.size == k):
            raise ValueError(f"repair clock: {k} lanes but {status.size} statuses, {keys.size} keys, "
                             f"{feasible.size} feasibility flags")
        dg = np.full(k, np.nan)
        inc = float(cutoff) if cutoff is not None and math.isfinite(float(cutoff)) else None
        dual = float(dual_key)
        if inc is not None:                 # every lane that beats the cutoff under the evaluation (any stop but 0)
            ok = (status != 0) & feasible & np.isfinite(keys) & (keys < inc)
            base = max(abs(inc), abs(dual) if math.isfinite(dual) else 0.0, 1e-9)
            dg[ok] = np.minimum((inc - keys[ok]) / np.maximum(base, np.abs(keys[ok])), 1.0)
        beat = np.isfinite(dg) & (dg > 0)
        L = self.L
        gain = np.zeros(L)
        good = np.flatnonzero(beat & (status == ST_FEASIBLE))   # the lanes Δ counts
        if good.size:
            for j in range(L):
                sel = dg[good] if pinned else dg[good][steps[good] <= self.hz[j]]
                gain[j] = float(sel.max()) if sel.size else 0.0
        seconds = float(profile.get("seconds", 0.0)) if profile else 0.0
        if pinned:
            t = np.full(L, seconds)
            obs = np.ones(L, bool)
        else:
            t, obs = self._call_times(profile, steps, status)
        heff = int(steps.max()) if k else 0
        cut = bool(k and np.any(status == ST_HORIZON))
        top = None if not good.size else [int(steps[good[np.argmax(dg[good])]]), float(dg[good].max())]
        lanes = np.flatnonzero(beat)
        order = lanes[np.argsort(-dg[lanes], kind="stable")][:LANE_GAINS]

        def beat_by(m):                     # [lanes, best dg] of the improving lanes not ended FEASIBLE
            m = beat & m
            return [int(m.sum()), float(dg[m].max()) if m.any() else None]
        self.g["calls"].append(dict(
            site=site, k=int(k), pinned=bool(pinned), cf=bool(counterfactual), H=horizon, cap=cap_s,
            cap_hits=int(cap_hits), H_eff=heff, cut=cut, s=round(seconds, 6), gain=gain, t=t, obs=obs, top=top,
            cutoff=inc, dual_key=dual, steps=steps.tolist(), status="".join(str(int(v)) for v in status),
            lane_gains=[[int(steps[i]), float(dg[i]), int(status[i])] for i in order],
            beat_stopped=beat_by((status == ST_STALL) | (status == ST_BUDGET)),
            beat_cut=beat_by(status == ST_HORIZON), prof=_compact_profile(profile)))
        self.cap_hits += int(cap_hits)
        self.heff[int(np.searchsorted(self.hz, heff, side="left"))] += 1
        sc = self._since
        sc["H_min"] = heff if sc["H_min"] is None else min(sc["H_min"], heff)
        sc["H_max"] = heff if sc["H_max"] is None else max(sc["H_max"], heff)
        sc["cap_hits"] += int(cap_hits)

    def _call_times(self, profile: dict | None, steps: np.ndarray, status: np.ndarray):
        """(t_call(h), observed(h)) per level from a call's profile (module docstring)."""
        L = self.L
        t = np.full(L, np.nan)
        obs = np.zeros(L, bool)
        if not profile:
            return t, obs
        if profile.get("kind") == "sequential":
            seconds = float(profile.get("seconds", 0.0))
            H = profile.get("H")
            if profile.get("cut_rows", 0) or np.any(status == 0):   # rows only projected, or left by a stop
                return t, obs
            tot = float(steps.sum())
            cut = bool(np.any(status == ST_HORIZON))           # no lane cut: every lane stopped by itself
            for j in range(L):
                h = self.hz[j]
                if H is not None and h > H and cut:
                    continue
                obs[j] = True
                t[j] = seconds * (float(np.minimum(steps, h).sum()) / tot if tot > 0 else 1.0)
            return t, obs
        chunks = profile.get("chunks") or []
        if sum(int(c["k"]) for c in chunks) != steps.size:     # a run-level stop left chunks unreached
            return t, obs
        seconds = float(profile.get("seconds", 0.0))
        spans = sum(float(c["end"]) - float(c["t"][0]) for c in chunks)
        over = max(seconds - spans, 0.0)                     # outside the chunks: allocation, the final sync
        for j in range(L):
            h = self.hz[j]
            tot, ok = over, True
            for c in chunks:
                v = _chunk_time(c, h)
                if v is None:
                    ok = False
                    break
                tot += v
            if ok:
                obs[j], t[j] = True, tot
        return t, obs

    # ------------------------------------------------------------------------------------------------- generation
    def end_generation(self, now: float, gen_seconds: float) -> dict | None:
        """Close the current generation: its per-level gain Δ and time T, the accumulators (post phase) and the next
        level. Returns the `rclock` trace event's fields (None if no generation is open)."""
        g, self.g = self.g, None
        if g is None:
            return None
        c, L = self.cfg, self.L
        calls = g["calls"]
        cf = [cl for cl in calls if cl["cf"]]               # the counterfactual calls (module docstring)
        gain = np.zeros(L)
        tsum = np.zeros(L)
        obs = np.ones(L, bool)
        child = None
        call_s = 0.0
        for cl in cf:
            gain = np.maximum(gain, cl["gain"])
            tsum = tsum + np.where(cl["obs"], cl["t"], 0.0)
            call_s += cl["s"]
            if not cl["pinned"]:
                obs &= cl["obs"]
            if cl["site"] == "children" and child is None:
                child = cl
        o = max(float(gen_seconds) - call_s, 0.0)
        T = np.where(obs, o + tsum, np.nan)
        gain = np.where(obs, np.where(gain >= c.materiality, gain, 0.0), np.nan)
        gain_approx = max((float(cl["gain"].max()) for cl in calls if not cl["cf"]), default=None)
        would, play = g["would"], g["play"]
        use = obs if self.active else obs & self._would_reveal(cf, would.horizon)
        would_cap_hit = False
        if would.cap_s is not None:
            for cl in calls:
                tw = cl["t"][would.level]
                if not cl["pinned"] and np.isfinite(tw) and tw > would.cap_s:
                    would_cap_hit = True
        switched = False
        if would.phase == "post":
            self._decay(now)
            m = use
            self.G[m] += gain[m]
            self.T[m] += T[m]
            self.n[m] += 1
            self.events[m] += gain[m] > 0
            self.last_obs[m] = now
            if child is not None:
                ct = child["t"]
                for j in np.flatnonzero(m & np.isfinite(ct)):
                    self.child_t[j].append(float(ct[j]))
                if (play.cap_s is not None and play.level != self.NONE and child["cap_hits"]
                        and not np.isfinite(ct[play.level])):   # cut below its level by the cap: a censored time
                    self.child_t[play.level].append(max(float(child["s"]), float(play.cap_s)))
            self.time_played[play.level] += gen_seconds
            self.time_would[would.level] += gen_seconds
            if would.probe:
                self.probe_s += gen_seconds
                self._since["probes"] += 1
            self.cap_hits_would += int(would_cap_hit)
            self.gens_post += 1
            prev = self.cur
            self.cur = self._choose()
            switched = self.cur != prev
            self.switches += int(switched)
        else:
            self.gens_pre += 1
        rate = self.rates()
        self._since.update(level=self.label(play.level), would=self.label(would.level), phase=would.phase)
        best = None
        for cl in cf:
            if cl["top"] is not None and (best is None or cl["top"][1] > best["dg"]):
                best = dict(site=cl["site"], steps=cl["top"][0], dg=cl["top"][1])
        return dict(
            gen=g["gen"], t_dec=round(g["t"], 6), phase=would.phase, mode=c.mode, level=self.label(play.level),
            cap=play.cap_s, would=self.label(would.level), would_cap=would.cap_s, probe=would.probe,
            would_cap_hit=would_cap_hit, next=self.label(self.cur), switched=switched,
            gen_s=round(float(gen_seconds), 6), o_s=round(o, 6), gain=_floats(gain), T=_floats(T),
            used=use.tolist(), gain_approx=gain_approx, t_children=None if child is None else _floats(child["t"]),
            rate=_floats(rate), n=self.n.tolist(), events=self.events.tolist(), best=best,
            calls=[{k: (_floats(v) if k == "t" else v) for k, v in cl.items() if k not in ("gain", "obs", "top")}
                   for cl in calls])

    def _would_reveal(self, calls: list, H: int | None) -> np.ndarray:
        """Shadow mode: the levels a call played at horizon H would have revealed — every level if no lane of a
        (non-pinned) call ran beyond H, else the levels <= H. The deadline's cut is not modelled (`would_cap_hit`)."""
        m = np.ones(self.L, bool)
        if H is None:
            return m
        for cl in calls:
            if not cl["pinned"] and cl["k"] and max(cl["steps"]) > H:
                m &= self.hz <= H
        return m

    def _decay(self, now: float) -> None:
        if self.t_decay is not None:
            f = 2.0 ** (-max(now - self.t_decay, 0.0) / self.cfg.half_life_s)
            self.G *= f
            self.T *= f
        self.t_decay = float(now)

    def rates(self) -> np.ndarray:
        return np.where(self.T > 0, self.G / np.where(self.T > 0, self.T, 1.0), 0.0)

    def _choose(self) -> int:
        c = self.cfg
        rate = self.rates()
        elig = (self.n >= c.min_gens) & (self.events >= c.min_events) & (self.G > 0)
        if not elig.any():
            return self.NONE                                 # no evidence: the configured repair
        best = float(rate[elig].max())
        j = int(np.flatnonzero(elig & (rate == best)).max())   # the deeper level on a tie
        if j != self.cur and rate[j] > (1.0 + c.hysteresis) * rate[self.cur]:
            return j
        return self.cur

    # ------------------------------------------------------------------------------------------------- records
    @staticmethod
    def _fresh_since() -> dict:
        return dict(level=None, would=None, phase=None, probes=0, H_min=None, H_max=None, cap_hits=0)

    def gen_fields(self) -> dict:
        """The `rc_*` fields of a `gen` event: the level played and the one the controller chose for the last
        generation, its phase, and since the previous `gen` event the probe generations, the smallest and largest
        H_eff of a call and the calls the played cap cut; the cumulative switches."""
        s, self._since = self._since, self._fresh_since()
        return dict(rc_mode=self.cfg.mode, rc_phase=s["phase"], rc_level=s["level"], rc_would=s["would"],
                    rc_next=self.label(self.cur), rc_probe=s["probes"], rc_H_eff=[s["H_min"], s["H_max"]],
                    rc_cap_hits=s["cap_hits"], rc_switches=self.switches)

    def record(self) -> dict:
        """The summary block `repair_clock`."""
        post = float(self.time_played.sum())

        def share(a):
            return (a / post).round(6).tolist() if post > 0 else [0.0] * self.L
        return dict(
            self.describe(), generations_pre=self.gens_pre, generations_post=self.gens_post,
            t_post=self.t_post, post_s=round(post, 6), time_share_played=share(self.time_played),
            time_share_would=share(self.time_would), probe_s=round(self.probe_s, 6),
            probe_time_share=round(self.probe_s / post, 6) if post > 0 else 0.0, switches=self.switches,
            cap_hits=self.cap_hits, cap_hits_would=self.cap_hits_would, H_eff_calls=self.heff.tolist(),
            final_level=self.label(self.cur), rate=_floats(self.rates()), G=_floats(self.G), T=_floats(self.T),
            n=self.n.tolist(), events=self.events.tolist())


# ------------------------------------------------------------------------------------------------------ helpers
def _chunk_time(c: dict, h: float) -> float | None:
    """Seconds one chunk of a GPU call would have taken at horizon h (None: not observed — the chunk was cut below
    h). From the chunk's start: the (cumulative steps, seconds) points interpolated linearly inside a launch, plus the
    chunk's tail after its last launch (copy-back)."""
    cum, t = c["cum"], c["t"]
    start, end = float(t[0]), float(c["end"])
    last = int(cum[-1])
    if h >= last:
        if not (c.get("done") or h == last):
            return None
        return end - start
    i = int(np.searchsorted(np.asarray(cum), h, side="left"))
    if i == 0:
        v = float(t[0])
    else:
        c0, c1 = float(cum[i - 1]), float(cum[i])
        f = (h - c0) / (c1 - c0) if c1 > c0 else 1.0
        v = float(t[i - 1]) + f * (float(t[i]) - float(t[i - 1]))
    return v + (end - float(t[-1])) - start


def _compact_profile(profile: dict | None):
    """The profile as it goes into the `rclock` event: GPU chunks as {k, cum, t, end, H_eff, done, cut, cap_hit}
    (seconds rounded to 1e-6), the CPU's sequential record as it is."""
    if not profile:
        return None
    if profile.get("kind") != "launches":
        return dict(profile)
    return dict(kind="launches", seconds=round(float(profile.get("seconds", 0.0)), 6), H=profile.get("H"),
                cap_s=profile.get("cap_s"),
                chunks=[dict(c, t=[round(float(v), 6) for v in c["t"]], end=round(float(c["end"]), 6))
                        for c in profile.get("chunks", [])])


def _floats(a) -> list:
    return [None if not np.isfinite(v) else round(float(v), 9) for v in np.asarray(a, np.float64)]
