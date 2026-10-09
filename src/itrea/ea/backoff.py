"""The EA's GPU back-off beside cuOpt in arm A9: a contribution-gated duty cycle (D-018).

Design: vault `80-reviews/2026-10-08 A9 GPU back-off design (workflow).md`, section 1 (the rule) and section 4 (this
module); configuration `BackoffCfg` (`backoff`, off by default and out of the hash while off). Host-side bookkeeping
only, like `ea.repair_clock`: no backend, no device array, no random number. Every method takes `now` — seconds on the
monotonic clock the engine runs on — so tests drive it with a fake clock (`clock` is used only where no time is given).
The engine (`ea.engine`) decides WHERE it may sleep (between generations after a sync, between FJ launches) and does
the sleeping; this class decides HOW LONG and keeps the accounting.

    bo = Backoff(cfg, t0)
    bo.update(now)                -> [change records]: phase-Y expiry, steps down the ladder
    bo.sleep_due(now)             -> seconds to sleep at this pace point (0 in shadow mode and at full duty)
    bo.slept(s, "gen" | "call")   -> the credit and the pause accounting of a sleep (measured seconds)
    bo.wake(now)                  -> a partner improvement during a between-generation sleep: forgive the credit?
    bo.phase_y_exit(now, why)     -> the exit record (None if phase Y is over)
    bo.own_feasible(now)          -> [records] or None;  bo.contribution(now, dg, kind, ref) -> (material, [records])
    bo.busy(now), bo.gen_fields(now), bo.record(now)

Clocks. Wall time w = now − t0; busy time b = w − (seconds slept by the governor, between generations and inside
calls). Every patience is counted in busy seconds, so the same EA work is needed at every duty (and the engine runs
the repair clock on the same busy clock, so the pause is not a second variable).

Duty. Level ℓ gives duty D = 2^−ℓ. Phase Y (from t0, while no incumbent exists anywhere): ℓ = `yield_level`. It ends
(`phase_y_exit`) at the first partner incumbent (reason "peek": a non-consuming read of SCIP's / cuOpt's info channel
inside a repair call; "tick": an exchange tick made the external incumbent finite), at the EA's first incumbent
("own"), or when `yield_until_s` of WALL time have passed ("expiry"); then phase P starts at ℓ = 0 and phase Y never
returns. Whatever detects the exit, an exit detected at or after `yield_until_s` is the expiry, dated at
`yield_until_s` (its busy time = busy now − the wall seconds since; no governor sleep falls between the two, because a
phase-Y sleep ends at the expiry and shadow mode never sleeps), so a late update — shadow mode updates once per
generation — does not move phase P's start. Forced mode: D = `forced_duty` from t0 to the end — no phases, no ladder,
no forgiven credit (stage-0 calibration only: the dose stays fixed, D-018 amendment).

Ladder (phase P). b_last = the busy time of the last reset: phase-P start, every level change, every material
contribution, the EA's first own feasible member. The first step down ℓ0 → ℓ1 comes when b − b_last ≥ τ, each further
one after `step_s` more busy seconds, down to ℓ = L (`levels`). τ by the EA's state:
  - "search" (no feasible member of its own; the engine's `_rc_own_feasible` test): `patience_search_s`;
  - "own" (an own feasible member, no material lead yet): `patience_own_s`;
  - "led" (at least one material lead): clamp(κ / λ̂, `patience_min_s`, `patience_max_s`), λ̂ = material events
    (leads and assists) per busy second since phase P started, both sums decayed with half-life `half_life_s` busy
    seconds (λ̂ = E / T, E = Σ 2^−(b − b_i)/h over the events, T = ∫ 2^−(b − u)/h du over phase P).
A step down is dated at the busy time it fell due (b_last + τ), so an update made late — shadow mode updates once per
generation — still logs when it would have happened: `busy_s` of the record, and `t_due` = its wall time (now − t0 −
the busy seconds since; exact in shadow mode, where busy = wall, and within one pace interval in active mode). In the
led state τ is taken at the update (λ̂ decayed to the update's busy time), which can date a late shadow step somewhat
later than a per-pace update would (λ̂ only falls between events, so never earlier). Any material contribution (gain ≥
`materiality` in the repair clock's gap unit, `gap_unit`) sets ℓ = 0 at once; `contribution` and `own_feasible` first
run `update` to their `now`, so a step that fell due before the reset is logged (dated at its due time), not erased.

Token bucket. Credit c in seconds: busy time lowers it, c −= Δb·(1 − D); sleep raises it, c += s·D. When c < −burst_s
a sleep of min(−c / D, max_sleep_s) is due (the engine cuts it into slices of `slice_s`, each of which keeps the stop
poll and, between generations, the exchange ticks alive). Long-run duty = D: at ¼ about 0.33 s of work and 1 s
asleep, at ⅛ 0.29 s and 2 s. At full duty no sleep is due and the credit is reset (the debt of a reduced duty is
forgiven when full duty returns). A partner improvement during a between-generation sleep (`wake`) forgives the
credit at most once per `wake_min_s` and never in forced mode; the level is not changed. The engine ends the sleep
only on a forgiven wake; otherwise the sleep goes on with its ticks (ending it would only move the same debt into the
children's first FJ chunk, where no tick runs, before the pending rows are injected).

Modes. "active": the rule. "shadow": the same bookkeeping, but `sleep_due` is always 0 — nothing paces, the busy clock
is the wall clock, and the records show what the rule would have done (the stage-1 control). "forced": fixed duty.
"""
from __future__ import annotations

import math
import time

from itrea.config import BackoffCfg

TIMELINE_MAX = 4000                # change records kept in the summary's timeline (all are in the event stream)
_LN2 = math.log(2.0)


def gap_unit(ref: float, key: float, dual_key: float = -math.inf) -> float:
    """The repair clock's gap unit of the improvement ref -> key (keys: lower is better): 1 if ref is not finite, else
    min((ref − key) / max(|ref|, |key|, |dual_key| if finite, 1e-9), 1) — negative when key is worse than ref."""
    ref, key = float(ref), float(key)
    if not math.isfinite(ref):
        return 1.0
    if not math.isfinite(key):
        return 0.0
    base = max(abs(ref), abs(key), abs(dual_key) if math.isfinite(dual_key) else 0.0, 1e-9)
    return min((ref - key) / base, 1.0)


def _r(v, nd: int = 6):
    return None if v is None or not math.isfinite(v) else round(float(v), nd)


class Backoff:
    """The governor of the module docstring (one per run)."""

    def __init__(self, cfg: BackoffCfg, t0: float, clock=time.monotonic):
        self.cfg, self.t0, self.clock = cfg, float(t0), clock
        self.mode = cfg.mode
        self.forced = self.mode == "forced"
        self.shadow = self.mode == "shadow"
        self.L = int(cfg.levels)
        self.phase = "forced" if self.forced else "Y"
        self.level: int | None = None if self.forced else int(cfg.yield_level)
        self.state = "search"
        # pauses
        self.paused = 0.0
        self.paused_site = {"gen": 0.0, "call": 0.0}
        self.sleep_calls = 0                       # slept() calls with s > 0 (slices)
        # token bucket
        self.c = 0.0
        self.b_cred = 0.0                          # busy time up to which the credit is charged
        # ladder
        self.b_last = 0.0
        self.E, self.T, self.lam_b = 0.0, 0.0, None   # decayed material events / busy seconds since phase P
        self.escalations = 0
        self.first_escalation: dict | None = None
        # milestones and counts
        self.y_exit: dict | None = None
        self.own: dict | None = None
        self.first_lead: dict | None = None
        self.n_material = {"lead": 0, "assist": 0}
        self.n_all = {"lead": 0, "assist": 0}
        self.wakes = 0
        self.forgiven = 0
        self.last_forgive = -math.inf
        # seconds per level / phase (wall and busy)
        self.sec_w: dict[str, float] = {}
        self.sec_b: dict[str, float] = {}
        self.w_seen = 0.0
        self.b_seen = 0.0
        self.timeline: list[list] = []

    # ------------------------------------------------------------------------------------------------- state
    @property
    def duty(self) -> float:
        return float(self.cfg.forced_duty) if self.forced else 2.0 ** -int(self.level)

    @property
    def in_phase_y(self) -> bool:
        return self.phase == "Y"

    def busy(self, now: float | None = None) -> float:
        now = self.clock() if now is None else float(now)
        return (now - self.t0) - self.paused

    def lam(self) -> float:
        """λ̂: decayed material events per decayed busy second since phase P started (0 before any event)."""
        if self.E <= 0:
            return 0.0
        return self.E / self.T if self.T > 0 else math.inf

    def patience(self) -> float:
        """τ of the EA's current state (module docstring)."""
        c = self.cfg
        if self.state == "search":
            return float(c.patience_search_s)
        if self.state == "own":
            return float(c.patience_own_s)
        lam = self.lam()
        tau = c.patience_kappa / lam if lam > 0 else math.inf
        return float(min(max(tau, c.patience_min_s), c.patience_max_s))

    def _threshold(self) -> float:
        return self.patience() if self.level == 0 else float(self.cfg.step_s)

    def _key(self) -> str:
        return "forced" if self.forced else ("Y" if self.phase == "Y" else str(self.level))

    # ------------------------------------------------------------------------------------------ accounting
    def _acc(self, w: float, b: float) -> None:
        """Seconds per level / phase: the wall and busy time since the last call go to the current one."""
        k = self._key()
        if w > self.w_seen:
            self.sec_w[k] = self.sec_w.get(k, 0.0) + (w - self.w_seen)
            self.w_seen = w
        if b > self.b_seen:
            self.sec_b[k] = self.sec_b.get(k, 0.0) + (b - self.b_seen)
            self.b_seen = b

    def _acc_to_busy(self, now: float, be: float) -> None:
        """`_acc` up to the busy time `be` (< busy(now)): the wall time is split in the same proportion."""
        w, b = now - self.t0, self.busy(now)
        f = 1.0 if b <= self.b_seen else min(max((be - self.b_seen) / (b - self.b_seen), 0.0), 1.0)
        self._acc(self.w_seen + f * max(w - self.w_seen, 0.0), be)

    def _charge(self, b: float) -> None:
        """Busy time up to b lowers the credit at the current duty."""
        if b > self.b_cred:
            self.c -= (b - self.b_cred) * (1.0 - self.duty)
            self.b_cred = b

    def _lam_to(self, b: float) -> None:
        if self.lam_b is None or b <= self.lam_b:
            return
        h = float(self.cfg.half_life_s)
        f = 2.0 ** (-(b - self.lam_b) / h)
        self.E *= f
        self.T = self.T * f + (h / _LN2) * (1.0 - f)
        self.lam_b = b

    def _full_duty(self, b: float) -> None:
        """ℓ = 0: the debt is forgiven, the patience restarts."""
        self.level = 0
        self.c = 0.0
        self.b_cred = max(self.b_cred, b)
        self.b_last = b

    # ------------------------------------------------------------------------------------------ transitions
    def update(self, now: float) -> list[dict]:
        """Phase-Y expiry and every step down due by now (dated at its busy time). Returns the change records."""
        now = float(now)
        out: list[dict] = []
        b = self.busy(now)
        if self.forced:
            self._acc(now - self.t0, b)
            self._charge(b)
            return out
        if self.phase == "Y" and now - self.t0 >= self.cfg.yield_until_s:
            out.append(self.phase_y_exit(now, "expiry"))
        if self.phase == "P":
            self._lam_to(b)
            while self.level < self.L:
                thr = self._threshold()
                if b - self.b_last < thr:
                    break
                be = self.b_last + thr
                self._acc_to_busy(now, be)
                self._charge(be)
                self.level += 1
                self.b_last = be
                self.escalations += 1
                rec = self._rec(now, "escalate", busy=be)
                if self.first_escalation is None:
                    self.first_escalation = dict(t=rec["t_due"], busy_s=_r(be), level=self.level, state=self.state)
                out.append(rec)
        self._acc(now - self.t0, b)
        return out

    def phase_y_exit(self, now: float, why: str) -> dict | None:
        """End phase Y (why: "peek" | "tick" | "own" | "expiry"); ℓ = 0. None if phase Y is already over. Detected at
        or after `yield_until_s` of wall time, the exit is the expiry, dated at `yield_until_s` (module docstring)."""
        if self.phase != "Y":
            return None
        now = float(now)
        w, b = now - self.t0, self.busy(now)
        yu = float(self.cfg.yield_until_s)
        if w >= yu:                                          # the expiry came first: dated at yield_until_s
            why, t_exit = "expiry", yu
            be = min(max(b - (w - yu), 0.0), b)
        else:
            t_exit, be = w, b
        self._acc_to_busy(now, be)                           # phase Y gets the seconds up to the exit
        self._charge(be)
        self.phase = "P"
        self._full_duty(be)
        self.lam_b, self.E, self.T = be, 0.0, 0.0
        self.y_exit = dict(t=_r(t_exit), busy_s=_r(be), reason=why)
        return self._rec(now, f"y_{why}", busy=be, t_due=t_exit)

    def own_feasible(self, now: float) -> list[dict] | None:
        """The population first holds a feasible member of the EA's own: state search -> own, patience restarts. The
        steps (and the phase-Y expiry) due before the restart are logged first. Returns the records, None if none."""
        if self.own is not None:
            return None
        now = float(now)
        out = self.update(now)
        b = self.busy(now)
        self.own = dict(t=_r(now - self.t0), busy_s=_r(b))
        if self.state == "search":
            self.state = "own"
            if self.phase == "P":
                self.b_last = b
            out.append(self._rec(now, "own_feasible"))
        return out or None

    def contribution(self, now: float, dg: float, kind: str, ref: float | None = None) -> tuple[bool, list[dict]]:
        """A lead (an EA-made incumbent better than min(EA best, external incumbent) by dg) or an assist (SCIP took an
        EA post as an improvement by dg). Returns (material, change records); a material one resets ℓ to 0 — after
        the steps (and the phase-Y expiry) due before it are logged, dated at their due time."""
        if kind not in self.n_all:
            raise ValueError(f"unknown contribution kind {kind!r}")
        self.n_all[kind] += 1
        if not (dg >= self.cfg.materiality):
            return False, []
        now = float(now)
        out = self.update(now)
        b = self.busy(now)
        self.n_material[kind] += 1
        changed = False
        if kind == "lead":
            if self.first_lead is None:
                self.first_lead = dict(t=_r(now - self.t0), busy_s=_r(b), dg=_r(dg, 9))
            if self.own is None:
                self.own = dict(t=_r(now - self.t0), busy_s=_r(b))
            changed = self.state != "led"
            self.state = "led"
        if self.phase != "P":
            return True, out + ([self._rec(now, kind)] if changed else [])
        self._lam_to(b)
        self.E += 1.0
        self._acc(now - self.t0, b)
        self._charge(b)
        changed = changed or self.level != 0
        self._full_duty(b)
        return True, out + ([self._rec(now, kind)] if changed else [])

    # ------------------------------------------------------------------------------------------ the bucket
    def sleep_due(self, now: float) -> float:
        """Seconds to sleep at this pace point: min(−c / D, max_sleep_s) once c < −burst_s (in phase Y no further than
        its expiry at yield_until_s); 0 in shadow mode and at full duty."""
        if self.shadow:
            return 0.0
        D = self.duty
        if D >= 1.0:
            return 0.0
        now = float(now)
        self._charge(self.busy(now))
        if self.c < -self.cfg.burst_s:
            s = min(-self.c / D, self.cfg.max_sleep_s)
            if self.phase == "Y":                            # never asleep past phase Y's expiry
                s = min(s, max(self.t0 + self.cfg.yield_until_s - now, 0.0))
            return float(s)
        return 0.0

    def slept(self, s: float, site: str) -> None:
        """A sleep of s measured seconds at `site` ("gen": between generations, "call": inside a repair call)."""
        s = max(float(s), 0.0)
        if s <= 0.0:
            return
        self.paused += s
        self.paused_site[site] += s
        self.sleep_calls += 1
        D = self.duty
        if not self.shadow and D < 1.0:
            self.c += s * D

    def wake(self, now: float) -> bool:
        """A partner improvement during a between-generation sleep (counted); the credit is forgiven at most once per
        wake_min_s, never in forced mode (its dose stays fixed). Returns True if it was forgiven (the engine then ends
        the sleep)."""
        now = float(now)
        self.wakes += 1
        if self.forced:
            return False
        if now - self.last_forgive >= self.cfg.wake_min_s:
            self.last_forgive = now
            self.forgiven += 1
            self.c = 0.0
            self.b_cred = max(self.b_cred, self.busy(now))
            return True
        return False

    # ------------------------------------------------------------------------------------------ records
    def _rec(self, now: float, reason: str, busy: float | None = None, t_due: float | None = None) -> dict:
        """A change record dated at its due time: busy `busy` (default: now) and wall `t_due` (default: now − t0 − the
        busy seconds since `busy`)."""
        now = float(now)
        bn = self.busy(now)
        b = bn if busy is None else float(busy)
        td = (now - self.t0) - (bn - b) if t_due is None else float(t_due)
        lam = self.lam()
        rec = dict(mode=self.mode, phase=self.phase, state=self.state, level=self.level, duty=_r(self.duty),
                   reason=reason, busy_s=_r(b), t_due=_r(td), paused_s=_r(self.paused), lam=_r(lam),
                   patience_s=_r(self.patience()))
        if len(self.timeline) < TIMELINE_MAX:
            self.timeline.append([_r(td), _r(b), self.phase, self.level, reason])
        return rec

    def snapshot(self, now: float, reason: str) -> dict:
        """A record of the current state (the start record, wakes)."""
        return self._rec(float(now), reason)

    def describe(self) -> dict:
        return describe(self.cfg)

    def gen_fields(self, now: float) -> dict:
        """The `bo_*` fields of a `gen` event (cumulative since t0)."""
        b = self.busy(now)
        return dict(bo_mode=self.mode, bo_phase=self.phase, bo_state=self.state, bo_level=self.level,
                    bo_duty=_r(self.duty), bo_busy_s=_r(b), bo_paused_gen_s=_r(self.paused_site["gen"]),
                    bo_paused_call_s=_r(self.paused_site["call"]), bo_wakes=self.wakes,
                    bo_contrib=sum(self.n_material.values()), bo_lam=_r(self.lam()), bo_patience_s=_r(self.patience()))

    def record(self, now: float) -> dict:
        """The summary block `backoff`."""
        now = float(now)
        self._acc(now - self.t0, self.busy(now))
        w, b = now - self.t0, self.busy(now)
        y_len = (self.y_exit["t"] if self.y_exit is not None else (w if self.phase == "Y" else None))
        return dict(
            describe(self.cfg), final_phase=self.phase, final_state=self.state, final_level=self.level,
            final_duty=_r(self.duty), wall_s=_r(w), busy_s=_r(b), busy_share=_r(b / w) if w > 0 else None,
            paused_s=_r(self.paused), paused_gen_s=_r(self.paused_site["gen"]),
            paused_call_s=_r(self.paused_site["call"]), sleeps=self.sleep_calls,
            phase_y=dict(length_s=_r(y_len) if y_len is not None else None,
                         exit_t=self.y_exit["t"] if self.y_exit else None,
                         exit_busy_s=self.y_exit["busy_s"] if self.y_exit else None,
                         reason=self.y_exit["reason"] if self.y_exit else (None if self.forced else "never")),
            seconds_per_level={k: _r(v) for k, v in sorted(self.sec_w.items())},
            busy_per_level={k: _r(v) for k, v in sorted(self.sec_b.items())},
            own_feasible=self.own, first_lead=self.first_lead, n_lead=self.n_material["lead"],
            n_assist=self.n_material["assist"], n_lead_all=self.n_all["lead"], n_assist_all=self.n_all["assist"],
            escalations=self.escalations, first_escalation=self.first_escalation, wakes=self.wakes,
            wakes_forgiven=self.forgiven, lam=_r(self.lam()), timeline=list(self.timeline),
            timeline_truncated=len(self.timeline) >= TIMELINE_MAX)


def describe(cfg: BackoffCfg) -> dict:
    """The configuration as recorded in the start-up record and the summary."""
    return dict(mode=cfg.mode, forced_duty=cfg.forced_duty, yield_level=cfg.yield_level, levels=cfg.levels,
                yield_until_s=cfg.yield_until_s, patience_search_s=cfg.patience_search_s,
                patience_own_s=cfg.patience_own_s, materiality=cfg.materiality, patience_kappa=cfg.patience_kappa,
                patience_min_s=cfg.patience_min_s, patience_max_s=cfg.patience_max_s, half_life_s=cfg.half_life_s,
                step_s=cfg.step_s, burst_s=cfg.burst_s, max_sleep_s=cfg.max_sleep_s, slice_s=cfg.slice_s,
                wake_min_s=cfg.wake_min_s, assist=cfg.assist)
