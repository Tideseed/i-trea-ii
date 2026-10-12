"""`itrea backoff-report`: the stage-0 calibration and the stage-1 test of A9's GPU back-off (D-018), applied by the
rules the design fixed before any run (vault `80-reviews/2026-10-08 A9 GPU back-off design (workflow).md`, section 5) —
nothing is chosen by eye. Development only (D-010).

Inputs: `datasets/runs.parquet` and `datasets/incumbents_checked.parquet` (`itrea build-runs`) for P(T) (`pt`, cuOpt's
own `cuopt_pt`), cuOpt's checked first incumbent (`cuopt_t_first`; a time censored at the budget —
`cuopt_t_first_event` false, cuOpt found nothing — counts as missing / ∞, never as the budget) and the checked cuOpt
trajectories; every run's
directory (`run_dir`, relative to `--root`) for the EA summary's `backoff` block, the EA's `backoff` / `contrib` / `gen`
/ `rclock` / `lp` events, the partners' `send` / `incumbent` events and the runner's `gpu_health` and `stop` records.
Runs are taken from one set (`--set`), status completed and not voided; a duplicate (instance, arm, seed) keeps the
highest attempt.

    itrea backoff-report --stage 0 [--set backoff0] [--runs datasets/runs.parquet] [--json out.json]
    itrea backoff-report --stage 1 [--set backoff1] [--stage0-set backoff0] [--json out.json]
    itrea backoff-report --stage 1y [--set backoff2] [--failed both|keep|exclude] [--harness ext|r13-off] [--json out]

Stage 0 (seed 2; chromaticindex1024-7, sing44, comp21-2idx x {A6, A9@bo0, A9@bof50, A9@bof25, A9@bof12}; ca and
glass4 x A9@bo0; sing44 x A2, the zero-duty reference of rule (b), D-018 amendment: 18 runs). Readouts: S_first =
cuOpt's first chromaticindex incumbent inside the arm / same-day A6's (∞ when the arm's cuOpt found none; undefined
when A6's found none); sing44's late stretch S_late = t_arm(v*) / t_A2(v*), v* = the best cuOpt objective same-day A2
reaches by 60 s and t(v) the first time cuOpt inside the run reaches an objective at least as good as v (an arm that
never does within its budget: budget / t_A2(v*), and the limb fails); n/a when A2 has no cuOpt incumbent by 60 s;
comp21's cuOpt-inside P against A6; the measured EA duty (busy share) and EA lane steps per wall second / bo0's;
gpu_health mean_util; the fair-share model S = 1 / (1 − d/2) (1.33 / 1.14 / 1.07); reported only, not a rule: the
shared-prefix stretch against A6 (median t_arm(v) / t_A6(v), t_A6(v) >= 1 s, at least 5 values — the statistic the
amendment replaced: it sits in cuOpt's first seconds and barely responds to the duty). Rules:
  - sanity: a forced run counts only if its measured duty is within ±0.03 of d and its lane steps per wall second lie
    in [0.75, 1.5]·d × bo0's on the same instance; a run that does not count cannot qualify its d;
  - (a) D_Y = the largest d in {1/2, 1/4, 1/8} with S_first <= 2.0 — if none and every forced chromaticindex run
    counted: STOP, no stage 1 (the pacing mechanism cannot protect cuOpt's first incumbent; MPS is next and needs
    Berk); if none and a run did not count (or A6 found no incumbent): INVALID — the measurement failed, not the
    mechanism; fix and rerun stage 0;
  - (b) D_min = the largest d whose sing44 limb (S_late <= 1.15, or n/a) and comp21 limb (cuOpt-inside P <= A6's +
    max(0.02, A6's two-seed spread) — the spread from A6 runs of comp21 at the same budget outside this set, seeds 0
    and 1 when present) both pass; if none qualifies D_min = 1/8, a shortfall; L = log2(1/D_min); the report states
    which limb decided;
  - (c) yield_until_s = max(30, ceil(2 x chromaticindex's first cuOpt incumbent at D_Y));
  - (d) patience_search_s = clamp(ceil(1.25 x (t_first_lead − t_first_partner)), 40, 90) from ca's bo0 run (shadow:
    busy = wall), the partner's first POSTED incumbent (first `send` of SCIP or cuOpt); 40 if no material lead;
  - (e) patience_own_s = clamp(ceil(1.25 x the larger own-feasible -> first-lead lag), 20, 40) over the sing44 and
    glass4 bo0 runs with a material lead; 30 if neither leads.
The verdict prints the bo1 values (configs/itrea2-dev-rc-3way-bo1.yaml is written from them).

Stage 1 (A9@bo0 shadow control vs A9@bo1 active, same-day A6, seeds 0–1, 14 instances): bounded d of bo1 vs bo0 on
seed means (positive = bo1 better); band_i = max(0.05, the larger seed-0 vs seed-1 bounded difference of bo0 or of
A6). Mechanism M1–M5 (M2: a censored bo1 first incumbent fails unless A6's is censored too — reported; M6 reported,
on the governor's DUE times: `t_due` of the step-down records) and outcome T, G, ladder, F and the mean guard exactly
as the design states; the
noise rule (a guard failing by the letter while bo1 never left level 0 and phase Y lasted < 1 s), the ca / glass4 /
sing44 fallbacks, "T fails with M2 failing" and "inside the band" are classified and printed. Adopt only if T, G, F,
the mean guard and M1–M5 hold. The D-013 weak-set report, the relay latency (cuOpt send -> SCIP incumbent made from
it) and the HiGHS LP-point arrival are reported, not judged.

Stage 1y (D-023, phase Y only; 2026-10-12, fixed before any run): A9@boY (bo1's values, active, the ladder disabled by
patience 100000 s) vs the control A9@bo1s, same-day A6F (cuOpt alone on A9's footprint, D-021) in A6's place, set
backoff2, the 14 instances x seeds 0–1, under D-022's harness. Outcome: D-018 stage 1's rules unchanged — T, G (noise
rule, guard-set mean, "still beats A6F"), the ladder targets, F, the mean guard; band_i on the control or A6F — the
same code as stage 1 with the arms as parameters. Mechanism, D-023 exactly:
  - M1: phase Y ends within 1.0 s of the first posted partner incumbent on every boY run where a partner posted (the
    runs without a post are listed as n/a, as in D-018);
  - M2: chromaticindex's cuOpt first incumbent inside boY <= 2 x A6F's on both seeds (censoring as in D-018's M2);
  - M3: the EA's material leads under boY >= 0.8 x the control's on item_placement, loue, neos-960392 and crna,
    each SUMMED over the two seeds (no first-lead or level-0 limb);
  - M4: boY's busy share in phase Y <= 0.6, read POOLED over the arm: the sum of the summaries' `phase_y.exit_busy_s`
    over the sum of `phase_y.exit_t` (wall) over every boY run (a phase Y that never ended: `length_s` and
    `busy_per_level["Y"]`). Not per run: no sleep falls due before the token bucket's credit reaches −burst_s, so a
    phase Y shorter than ~0.5 s has share 1.0 by construction (bo1's stage-1 runs: 9 of 14 at seed 0); the per-run
    shares are printed;
  - M5: the busy / paused accounting of every boY `rclock` event within 2 % (D-018's accounting leg) and local-search
    cap hits per call within +0.10 of the same-seed control's.
D-018's own M1 (0.5 s), M3 (per seed, with the first-lead and level-0 limbs) and M5 (every leg, cap hits +0) are
computed on the same runs and printed beside (`mechanism_d018`, not gating). Adopt boY only if M1–M5, T, G, the
ladder, F and the mean guard all hold: D-023 asks every outcome rule to hold and lists the ladder targets among them,
whereas D-018 stage 1 (design §5) reported the ladder without gating — stage 1 keeps that. Failed runs: both readings
(`--failed both`, the default for 1y): kept as D-002 §6 keeps crashes in full, the excluded reading's verdict beside.
In 1y's result the effect keys keep stage 1's names (`*_bo1` = boY, `*_bo0` = the control, `*_A6` = A6F).
Harness (validity of 1y; D-023 runs under D-022's harness outcome): every run of the set's three arms (completed, and
failed under the kept reading; not voided, not synthetic; every attempt, not only the one `load` keeps) carries a
resolved harness — run.json `config.harness`, else the run table's `harness` column —; the two A9 arms share ONE
harness (so every block is equal), every A6F run is `pair.resolve_harness("ext")` (D-021 R13 / R6: D-022's outcome
sets only A9's harness), and with `--harness ext|r13-off` the A9 harness must be that outcome's. Otherwise the
verdict is INVALID with the runs listed; the header names the A9 harness found.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.analysis.stats import D_FLOOR, effect_d
from itrea.paths import DATASETS, ROOT
from itrea.trace import read_events

# ----------------------------------------------------------------------------------------------- the design's names
A6, A2, BO0, BO1 = "A6", "A2", "A9@bo0", "A9@bo1"
CTRL1 = "A9@bo1s"                         # stage-1 control: bo1's values in shadow mode (2026-10-09; its would-be timeline is bo1's)
BOY, A6F = "A9@boY", "A6F"                # stage 1y (D-023): the treatment (phase Y only); cuOpt alone on A9's F
FORCED = {"A9@bof50": 0.5, "A9@bof25": 0.25, "A9@bof12": 0.125}
DUTIES = (0.5, 0.25, 0.125)
STAGE0_SEED = 2
CHROMATIC, SING44, COMP21 = "chromaticindex1024-7", "sing44", "comp21-2idx"
CA, GLASS4 = "ca__Scenario400_reserves_1", "glass4"
TARGET = (CHROMATIC,)
LADDER = ("neos-3656078-kumeu", COMP21, "neos-4763324-toguru")
GUARDS_D = ("item_placement_9903", CA, "neos-3024952-loue", "neos-2657525-crna", GLASS4, SING44)
GUARD_960 = "neos-960392"
GUARDS = GUARDS_D[:4] + (GUARD_960,) + GUARDS_D[4:]
FLOOR = ("neos-3004026-krka", "fhnw-binpack4-48", "radiationm18-12-05")
STAGE1 = TARGET + LADDER + GUARDS + FLOOR
M3_SET = ("item_placement_9903", "neos-3024952-loue", GUARD_960, "neos-2657525-crna")
M4_SET = ("neos-3656078-kumeu", COMP21, "neos-4763324-toguru", "radiationm18-12-05")
STAGE1_SEEDS = (0, 1)
# stage-0 thresholds
S_FIRST_MAX, S_SING_MAX, P_COMP_SLACK = 2.0, 1.15, 0.02
V_STAR_S = 60.0                           # rule (b): v* = A2's best cuOpt objective by this time
DUTY_TOL, STEPS_LO, STEPS_HI = 0.03, 0.75, 1.5
# D-018 deviation (2026-10-09, after stage 0's first 13 runs, stated in the record): the lane-steps limb of the sanity
# rule assumed EA work proportional to the duty; contention refutes it in both directions (sing44: MORE steps per busy
# second at low duty — less contention from cuOpt; chromaticindex: fewer — launches restart short). The duty limb gates
# whether a forced run counts; the steps ratio is reported. `steps_gate=True` reproduces the rule as first written.
STEPS_GATE = False
MIN_SHARED, MIN_T_REF = 5, 1.0
# stage-1 thresholds
T_D, T_P = 0.30, 0.30
BAND_MIN, GUARD_MEAN, BEAT_A6 = 0.05, -0.05, 0.15
P960_MIN, FLOOR_DP = 0.01, 0.005
M1_S, M2_X, M3_LEADS, M3_FIRST_S, M3_FIRST_F, M3_L0 = 0.5, 2.0, 0.8, 2.0, 0.10, 0.90
M4_LATE, M4_FROM, M4_SHARE, M4_STEPS, M4_UTIL = 30.0, 60.0, 0.1, 0.5, 3
M5_ACC, M5_STOP = 0.02, 0.25
# stage-1y thresholds (D-023, fixed before any run): M1's phase-Y exit latency relaxed to 1.0 s; M3 on leads summed
# over the seeds; M4 the arm's busy share in phase Y; M5 cap hits per call within +0.10 of the control's
Y_M1_S, Y_M3_LEADS, Y_M4_SHARE, Y_M5_CAP = 1.0, 0.8, 0.6, 0.10
Y_HARNESS = ("ext", "r13-off")             # D-022's outcomes -> make_jobs.ITX_EXT / ITX_R13_CONTROL (A9 arms only)
REL_TOL = 1e-9


def _fin(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _f(v) -> float:
    return float(v) if _fin(v) else math.nan


def _due(e: dict) -> float:
    """A `backoff` record's due wall time: `t_due` when present, else its logging time `t`."""
    return _f(e.get("t_due")) if _fin(e.get("t_due")) else _f(e.get("t"))


def bounded(a: float, b: float) -> float:
    """|a − b| / max(a, b, 1e-3): the seed-to-seed difference in the unit of d."""
    if not (_fin(a) and _fin(b)):
        return math.nan
    return abs(a - b) / max(a, b, D_FLOOR)


# ------------------------------------------------------------------------------------------------- one run
@dataclass
class RunView:
    """One run: its runs.parquet row and lazily read files of its directory."""
    row: dict
    path: Path
    _cache: dict = field(default_factory=dict)

    def get(self, k, default=None):
        v = self.row.get(k, default)
        return default if v is None or (isinstance(v, float) and math.isnan(v)) else v

    def summary(self, role: str = "ea") -> dict:
        key = ("summary", role)
        if key not in self._cache:
            p = self.path / f"{role}.summary.json"
            try:
                self._cache[key] = json.loads(p.read_text())
            except (OSError, ValueError):
                self._cache[key] = {}
        return self._cache[key]

    def events(self, role: str = "ea") -> list[dict]:
        key = ("events", role)
        if key not in self._cache:
            p = self.path / f"{role}.events.jsonl"
            self._cache[key] = read_events(p) if p.exists() else []
        return self._cache[key]

    def run_json(self) -> dict:
        if "run" not in self._cache:
            try:
                self._cache["run"] = json.loads((self.path / "run.json").read_text())
            except (OSError, ValueError):
                self._cache["run"] = {}
        return self._cache["run"]

    @property
    def bo(self) -> dict:
        b = self.summary("ea").get("backoff")
        return b if isinstance(b, dict) else {}

    def kind(self, kind: str, role: str = "ea") -> list[dict]:
        return [e for e in self.events(role) if e.get("kind") == kind]

    # ---------------------------------------------------------------------------------------------- readouts
    def cuopt_t_first(self) -> float:
        """cuOpt's first checked incumbent (wall seconds); ∞ when censored at the budget (a time with
        `cuopt_t_first_event` false: cuOpt found nothing); NaN when the run has no cuOpt time at all (build_runs then
        stores NaN with the flag false)."""
        t = _f(self.get("cuopt_t_first"))
        ev = self.row.get("cuopt_t_first_event")
        if _fin(t) and ev is not None and not (isinstance(ev, float) and math.isnan(ev)) and not bool(ev):
            return math.inf
        return t

    def measured_duty(self) -> float:
        return _f(self.bo.get("busy_share"))

    def steps_per_wall_s(self) -> float:
        b = self.bo
        w = _f(b.get("wall_s"))
        return _f(b.get("ls_steps")) / w if w > 0 else math.nan

    def mean_util(self) -> float:
        return _f((self.run_json().get("gpu_health") or {}).get("mean_util"))

    def first_post(self) -> float:
        """The first posted partner incumbent (wall seconds since t0): the earliest first `send` of SCIP or cuOpt with
        rows, else (no send events) their first `incumbent` event; NaN if neither partner posted."""
        ts = []
        for role in ("solver", "cuopt"):
            sends = [_f(e.get("t")) for e in self.kind("send", role) if int(e.get("k", 1) or 0) > 0]
            if not sends and not self.kind("send", role):
                sends = [_f(e.get("t")) for e in self.kind("incumbent", role)]
            sends = [t for t in sends if _fin(t)]
            if sends:
                ts.append(min(sends))
        return min(ts) if ts else math.nan

    def leads(self, material: bool = True) -> list[float]:
        """Wall times of the EA's leads (`contrib` events, what = lead)."""
        return sorted(_f(e.get("t")) for e in self.kind("contrib")
                      if e.get("what") == "lead" and (bool(e.get("material")) or not material))

    def first_lead(self) -> float:
        fl = self.bo.get("first_lead")
        if isinstance(fl, dict) and _fin(fl.get("t")):
            return float(fl["t"])
        ls = self.leads()
        return ls[0] if ls else math.nan

    def own_feasible(self) -> float:
        o = self.bo.get("own_feasible")
        return _f(o.get("t")) if isinstance(o, dict) else math.nan

    def phase_y(self) -> tuple[float, str]:
        py = self.bo.get("phase_y") or {}
        return _f(py.get("exit_t")), str(py.get("reason") or "")

    def first_escalation(self) -> tuple[float, float]:
        """(wall t, busy seconds) of the first step down (shadow: the would-be one), at its DUE time — the record's
        `t_due`, not the time it was logged (a shadow update comes once per generation); NaN if none."""
        evs = [e for e in self.kind("backoff") if e.get("reason") == "escalate"]
        if evs:
            e = min(evs, key=lambda x: _due(x))
            return _due(e), _f(e.get("busy_s"))
        fe = self.bo.get("first_escalation")
        if isinstance(fe, dict):
            return _f(fe.get("t")), _f(fe.get("busy_s"))
        return math.nan, math.nan

    def level_timeline(self) -> list[tuple[float, str, int | None]]:
        """(t, phase, level) after every change, from the `backoff` events (the start record first), each at its due
        time (`t_due`; the logging time `t` for records without one)."""
        return [(_due(e), str(e.get("phase")), e.get("level")) for e in self.kind("backoff")]

    def full_duty_share(self, t_end: float) -> float:
        """Share of the wall time in [0, t_end] spent in phase P at level 0."""
        if not (t_end > 0):
            return math.nan
        tl = [x for x in self.level_timeline() if _fin(x[0])]
        if not tl:
            return math.nan
        full, prev_t, prev = 0.0, 0.0, (tl[0][1], tl[0][2])
        for t, ph, lv in tl[1:] + [(t_end, None, None)]:
            t = min(t, t_end)
            if t > prev_t and prev[0] == "P" and prev[1] == 0:
                full += t - prev_t
            prev_t = max(prev_t, t)
            prev = (ph, lv)
            if prev_t >= t_end:
                break
        return full / t_end

    def max_level_p(self) -> int:
        lv = [x[2] for x in self.level_timeline() if x[1] == "P" and isinstance(x[2], int)]
        return max(lv) if lv else 0

    def _cum(self, field_: str, cumulative: bool) -> list[tuple[float, float]]:
        pts = [(0.0, 0.0)]
        acc = 0.0
        for e in self.kind("gen"):
            if field_ not in e or not _fin(e.get("t")):
                continue
            v = _f(e[field_])
            acc = acc + v if cumulative else v
            pts.append((_f(e["t"]), acc))
        return pts

    def window_rates(self, t_from: float, t_to: float | None = None) -> tuple[float, float]:
        """(busy share, EA lane steps per wall second) over [t_from, t_to] (t_to: the EA's end), interpolated
        linearly between `gen` events (bo_busy_s cumulative, ls_steps per record)."""
        b = self.bo
        end = _f(b.get("wall_s"))
        t_to = end if t_to is None else min(t_to, end) if _fin(end) else t_to
        busy = self._cum("bo_busy_s", cumulative=False)
        steps = self._cum("ls_steps", cumulative=True)
        if _fin(end):
            busy.append((end, _f(b.get("busy_s"))))
            steps.append((end, _f(b.get("ls_steps"))))
        if not (_fin(t_to) and t_to > t_from) or len(busy) < 2:
            return math.nan, math.nan

        def at(pts, t):
            ts, vs = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            o = np.argsort(ts, kind="stable")
            return float(np.interp(t, ts[o], vs[o]))
        span = t_to - t_from
        return (at(busy, t_to) - at(busy, t_from)) / span, (at(steps, t_to) - at(steps, t_from)) / span


# ------------------------------------------------------------------------------------------------- loading
def load(runs: pd.DataFrame, set_name: str | None, root: Path) -> dict[tuple[str, str, int], RunView]:
    """{(instance, arm, seed): RunView} of the completed, not voided runs of the set (latest attempt)."""
    df = runs if set_name is None else runs[runs["set"] == set_name]

    def flag(col):
        return df[col].fillna(False).astype(bool) if col in df else pd.Series(False, index=df.index)
    df = df[(df["status"] == "completed") & ~flag("block_void") & ~flag("synthetic")]
    out: dict[tuple[str, str, int], RunView] = {}
    for r in df.sort_values("attempt" if "attempt" in df else "run_id", kind="mergesort").to_dict("records"):
        p = Path(str(r.get("run_dir") or ""))
        out[(str(r["instance"]), str(r["arm"]), int(r["seed"]))] = RunView(r, p if p.is_absolute() else root / p)
    return out


def cuopt_traj(inc: pd.DataFrame | None, run_id: str, T: float) -> list[tuple[float, float]]:
    """cuOpt's checked incumbents of one run: (t, recomputed objective), by time."""
    if inc is None or not len(inc):
        return []
    g = inc[(inc["run_id"] == run_id) & (inc["role"] == "cuopt") & inc["ok"] & (inc["t"] <= T)]
    return sorted(zip(g["t"].astype(float), g["obj_recomputed"].astype(float)))


def stretch(arm: list[tuple[float, float]], ref: list[tuple[float, float]], min_shared: int = MIN_SHARED,
            min_t_ref: float = MIN_T_REF) -> tuple[float | None, int]:
    """The shared-prefix stretch: median t_arm(v) / t_ref(v) over the objectives v both trajectories reached (first
    time each), t_ref(v) >= min_t_ref; (None, n) with fewer than min_shared such values."""
    def firsts(traj):
        out: list[tuple[float, float]] = []
        for t, v in traj:
            if not any(abs(v - u) <= REL_TOL * max(1.0, abs(u)) for u, _ in out):
                out.append((v, t))
        return out
    fa = firsts(arm)
    ratios = []
    for v, tr in firsts(ref):
        if tr < min_t_ref:
            continue
        ta = next((t for u, t in fa if abs(u - v) <= REL_TOL * max(1.0, abs(v))), None)
        if ta is not None:
            ratios.append(ta / tr)
    return (float(np.median(ratios)) if len(ratios) >= min_shared else None), len(ratios)


def first_reach(traj: list[tuple[float, float]], v: float, sense: int = 1, T: float = math.inf) -> float:
    """The first time (<= T) a trajectory reaches an objective at least as good as v (sense +1 minimises, −1
    maximises; relative tolerance 1e-9); ∞ if it never does."""
    tol = REL_TOL * max(1.0, abs(v))
    for t, o in traj:
        if t <= T and sense * (o - v) <= tol:
            return float(t)
    return math.inf


def late_stretch(arm: list[tuple[float, float]], ref: list[tuple[float, float]], sense: int = 1,
                 t_ref: float = V_STAR_S, budget: float = 120.0) -> dict:
    """Rule (b)'s sing44 limb (D-018 amendment): v* = the best objective of the reference (same-day A2, zero EA duty)
    by t_ref; S_late = t_arm(v*) / t_ref(v*). An arm that never reaches v* within `budget` gets budget / t_ref(v*) and
    `reached` false (the limb fails). {S_late: None} ("n/a") when the reference has no incumbent by t_ref."""
    pts = [(t, o) for t, o in ref if t <= t_ref]
    if not pts:
        return dict(v_star=None, t_ref=None, t_arm=None, S_late=None, reached=None)
    v = min(pts, key=lambda x: (sense * x[1], x[0]))[1]
    tr = first_reach(ref, v, sense)
    ta = first_reach(arm, v, sense, budget)
    if not (tr > 0):
        return dict(v_star=v, t_ref=tr, t_arm=ta, S_late=None, reached=None)
    reached = _fin(ta)
    return dict(v_star=v, t_ref=tr, t_arm=ta if reached else None, S_late=(ta if reached else budget) / tr,
                reached=bool(reached))


def _ceil(x: float) -> int:
    return int(math.ceil(x - 1e-9))


def _clamp(x, lo, hi):
    return min(max(x, lo), hi)


# ================================================================================================= stage 0
def stage0(runs: pd.DataFrame, inc: pd.DataFrame | None, set_name: str = "backoff0", root: Path = ROOT,
           seed: int = STAGE0_SEED, steps_gate: bool = STEPS_GATE) -> dict:
    """The stage-0 rules (module docstring). Returns {readouts, sanity, rules, bo1, verdict, notes}."""
    R = load(runs, set_name, root)
    notes: list[str] = []
    if inc is None or not len(inc):
        notes.append("no checked incumbents table: the sing44 limb is n/a for every duty (rule b then rests on "
                     "comp21 alone)")

    def run(inst, arm) -> RunView | None:
        return R.get((inst, arm, seed))

    missing = [(i, a) for i in (CHROMATIC, SING44, COMP21) for a in (A6, BO0, *FORCED) if run(i, a) is None]
    missing += [(i, BO0) for i in (CA, GLASS4) if run(i, BO0) is None]
    missing += [(SING44, A2)] if run(SING44, A2) is None else []
    arms = [(BO0, 1.0)] + list(FORCED.items())
    rd: dict = {}
    # ------------------------------------------------------------------------ readouts per duty
    a6c, a6s, a6p, a2s = run(CHROMATIC, A6), run(SING44, A6), run(COMP21, A6), run(SING44, A2)
    t_a6 = a6c.cuopt_t_first() if a6c else math.nan
    a6_censored = a6c is not None and math.isinf(t_a6)
    if a6_censored:
        notes.append("chromaticindex A6: cuOpt found no incumbent within the budget (censored): S_first has no "
                     "reference")
    traj_a6s = cuopt_traj(inc, a6s.get("run_id"), _f(a6s.get("budget"))) if a6s else []
    traj_a2s = cuopt_traj(inc, a2s.get("run_id"), _f(a2s.get("budget"))) if a2s else []
    sense_s = int(a2s.get("sense", 1) or 1) if a2s else 1
    if a2s is not None and inc is not None and len(inc) and not any(t <= V_STAR_S for t, _ in traj_a2s):
        notes.append(f"sing44 A2: no cuOpt incumbent by {V_STAR_S:g} s: the sing44 limb is n/a for every duty (rule "
                     "b rests on comp21 alone)")
    p_a6 = _f(a6p.get("cuopt_pt", a6p.get("pt"))) if a6p else math.nan
    if a6p is not None and not _fin(p_a6):
        p_a6 = _f(a6p.get("pt"))
    spread, spread_seeds = a6_spread(runs, COMP21, set_name, _f(a6p.get("budget")) if a6p else 120.0)
    if spread_seeds < 2:
        notes.append(f"comp21: fewer than two A6 seeds outside the set for the spread ({spread_seeds}); spread = 0")
    for arm, d in arms:
        c, s, p = run(CHROMATIC, arm), run(SING44, arm), run(COMP21, arm)
        t_first = c.cuopt_t_first() if c else math.nan          # ∞: the arm's cuOpt found nothing (censored)
        if not (_fin(t_a6) and t_a6 > 0):
            s_first = math.nan                                    # no reference (A6 missing or censored)
        elif math.isinf(t_first):
            s_first = math.inf
        else:
            s_first = t_first / t_a6 if _fin(t_first) else (math.inf if c is not None else math.nan)
        traj_s = cuopt_traj(inc, s.get("run_id"), _f(s.get("budget"))) if s else []
        S, n_sh = stretch(traj_s, traj_a6s) if s and a6s else (None, 0)
        late = (late_stretch(traj_s, traj_a2s, sense_s, V_STAR_S, _f(s.get("budget")) if _fin(s.get("budget"))
                             else 120.0) if s is not None and a2s is not None
                else dict(v_star=None, t_ref=None, t_arm=None, S_late=None, reached=None))
        pc = _f(p.get("cuopt_pt")) if p else math.nan
        rd[arm] = dict(d=d, t_first_chromatic=t_first, S_first=s_first, sing44_S_late=late["S_late"],
                       sing44_late=late, sing44_S_prefix_A6=S, sing44_shared=n_sh,
                       comp21_cuopt_P=pc, model_S=1.0 / (1.0 - d / 2.0) if d < 1 else 1.0,
                       duty={i: (run(i, arm).measured_duty() if run(i, arm) else math.nan)
                             for i in (CHROMATIC, SING44, COMP21)},
                       steps_per_s={i: (run(i, arm).steps_per_wall_s() if run(i, arm) else math.nan)
                                    for i in (CHROMATIC, SING44, COMP21)},
                       mean_util={i: (run(i, arm).mean_util() if run(i, arm) else math.nan)
                                  for i in (CHROMATIC, SING44, COMP21)})
    rd[A6] = dict(t_first_chromatic=t_a6, comp21_P=p_a6, comp21_spread=spread,
                  mean_util={i: (run(i, A6).mean_util() if run(i, A6) else math.nan)
                             for i in (CHROMATIC, SING44, COMP21)})
    v_pts = [(t, o) for t, o in traj_a2s if t <= V_STAR_S]
    v_star = min(v_pts, key=lambda x: (sense_s * x[1], x[0]))[1] if v_pts else None
    rd[A2] = dict(v_star=v_star, t_v_star=first_reach(traj_a2s, v_star, sense_s) if v_star is not None else None,
                  mean_util=run(SING44, A2).mean_util() if a2s else math.nan)
    # ------------------------------------------------------------------------ sanity
    sanity: dict = {}
    for arm, d in FORCED.items():
        for i in (CHROMATIC, SING44, COMP21):
            r, ref = run(i, arm), run(i, BO0)
            if r is None or ref is None:
                sanity[(arm, i)] = dict(ok=False, why="missing run")
                continue
            duty, sps, sps0 = r.measured_duty(), r.steps_per_wall_s(), ref.steps_per_wall_s()
            ratio = sps / sps0 if _fin(sps) and sps0 > 0 else math.nan
            ok_d = _fin(duty) and abs(duty - d) <= DUTY_TOL
            ok_s = _fin(ratio) and STEPS_LO * d <= ratio <= STEPS_HI * d
            sanity[(arm, i)] = dict(ok=bool(ok_d and (ok_s or not steps_gate)), duty=duty, steps_ratio=ratio,
                                    duty_ok=bool(ok_d), steps_ok=bool(ok_s), steps_gate=bool(steps_gate))

    def counts(arm, inst) -> bool:
        return bool(sanity.get((arm, inst), {}).get("ok"))
    arm_of = {d: a for a, d in FORCED.items()}
    # ------------------------------------------------------------------------ (a) D_Y
    rules: dict = {}
    qual_a = [d for d in DUTIES if counts(arm_of[d], CHROMATIC) and _fin(rd[arm_of[d]]["S_first"])
              and rd[arm_of[d]]["S_first"] <= S_FIRST_MAX]
    d_y = max(qual_a) if qual_a else None
    rules["a"] = dict(D_Y=d_y, qualifying=qual_a,
                      S_first={d: rd[arm_of[d]]["S_first"] for d in DUTIES},
                      not_counted=[d for d in DUTIES if not counts(arm_of[d], CHROMATIC)])
    # ------------------------------------------------------------------------ (b) D_min
    lim = p_a6 + max(P_COMP_SLACK, spread) if _fin(p_a6) else math.nan
    qual_b, limbs = [], {}
    for d in DUTIES:
        a = arm_of[d]
        late = rd[a]["sing44_late"]
        S = late["S_late"]
        sing = "n/a" if S is None else ("pass" if late["reached"] and S <= S_SING_MAX else "fail")
        ok_p = _fin(rd[a]["comp21_cuopt_P"]) and _fin(lim) and rd[a]["comp21_cuopt_P"] <= lim
        counted = counts(a, SING44) and counts(a, COMP21)
        limbs[d] = dict(sing44=sing, S_late=S, reached=late["reached"], comp21="pass" if ok_p else "fail",
                        comp21_P=rd[a]["comp21_cuopt_P"], counted=bool(counted))
        if counted and sing != "fail" and ok_p:
            qual_b.append(d)
    shortfall = not qual_b
    d_min = max(qual_b) if qual_b else 0.125
    L = int(round(math.log2(1.0 / d_min)))
    rules["b"] = dict(D_min=d_min, L=L, qualifying=qual_b, shortfall=shortfall, comp21_limit=lim, limbs=limbs,
                      v_star=v_star, decided_by=_floor_decision(limbs, d_min, shortfall),
                      not_counted=[d for d in DUTIES if not (counts(arm_of[d], SING44) and counts(arm_of[d], COMP21))])
    # ------------------------------------------------------------------------ (c) yield_until_s
    yu = None
    if d_y is not None:
        t_dy = rd[arm_of[d_y]]["t_first_chromatic"]
        yu = max(30, _ceil(2.0 * t_dy)) if _fin(t_dy) else None
    rules["c"] = dict(yield_until_s=yu)
    # ------------------------------------------------------------------------ (d) patience_search_s
    ca = run(CA, BO0)
    ps, d_detail = None, {}
    if ca is not None:
        tl, tp = ca.first_lead(), ca.first_post()
        if not _fin(tl):
            ps = 40
        else:
            if not _fin(tp):
                notes.append("ca bo0: no partner posted an incumbent; the lag is taken from t0")
                tp = 0.0
            ps = int(_clamp(_ceil(1.25 * (tl - tp)), 40, 90))
        d_detail = dict(t_first_lead=tl, t_first_partner=tp)
    rules["d"] = dict(patience_search_s=ps, **d_detail)
    # ------------------------------------------------------------------------ (e) patience_own_s
    lags = {}
    for i in (SING44, GLASS4):
        r = run(i, BO0)
        if r is None:
            continue
        tl, to = r.first_lead(), r.own_feasible()
        if _fin(tl):
            lags[i] = max(tl - (to if _fin(to) else tl), 0.0)
    po = int(_clamp(_ceil(1.25 * max(lags.values())), 20, 40)) if lags else 30
    rules["e"] = dict(patience_own_s=po, lags=lags)
    # ------------------------------------------------------------------------ verdict
    bo1 = None
    nc_a = rules["a"]["not_counted"]
    if missing:
        verdict = "INCOMPLETE: missing runs " + ", ".join(f"{i} {a}" for i, a in missing)
    elif a6_censored:
        verdict = ("INVALID: same-day A6 found no cuOpt incumbent on chromaticindex1024-7 within the budget "
                   "(censored), so rule (a) has no reference — rerun stage 0; not a STOP.")
    elif d_y is None and nc_a:
        bad = "; ".join(f"d = {d}: duty {_fmt(sanity[(arm_of[d], CHROMATIC)].get('duty'), 3)}, steps ratio "
                        f"{_fmt(sanity[(arm_of[d], CHROMATIC)].get('steps_ratio'), 2)}, S_first "
                        f"{_fmt(rd[arm_of[d]]['S_first'], 2)}" for d in nc_a)
        verdict = ("INVALID: no duty qualifies for rule (a), but the sanity check failed on chromaticindex for "
                   f"{bad} — the measurement failed, not the mechanism; fix it and rerun stage 0 (not a STOP).")
    elif d_y is None:
        verdict = ("STOP: no duty in {1/2, 1/4, 1/8} brings cuOpt's first chromaticindex incumbent within 2.0x of "
                   "A6's (rule a; every forced run counted) — no stage 1; the pacing mechanism cannot protect cuOpt's "
                   "first incumbent. MPS is the next candidate and needs Berk.")
    elif yu is None or ps is None:
        verdict = "INCOMPLETE: rule (c) or (d) has no value"
    else:
        bo1 = dict(enabled=True, mode="active", yield_level=int(round(math.log2(1.0 / d_y))), levels=L,
                   yield_until_s=float(yu), patience_search_s=float(ps), patience_own_s=float(po))
        verdict = "PROCEED to stage 1 with bo1 = " + json.dumps(bo1)
        if shortfall:
            nc_b = rules["b"]["not_counted"]
            verdict += (" (rule b shortfall: no duty met the floor's tests; D_min = 1/8" +
                        (f"; the sanity check failed on sing44 / comp21 for d in {nc_b}, so the shortfall rests on "
                         "runs that did not count" if nc_b else "") + ")")
    return dict(stage=0, set=set_name, seed=seed, readouts=rd, sanity={f"{a} {i}": v for (a, i), v in sanity.items()},
                rules=rules, bo1=bo1, verdict=verdict, notes=notes, missing=missing)


def _floor_decision(limbs: dict, d_min: float, shortfall: bool) -> str:
    """Which limb of rule (b) decided the floor: the limbs at the next larger duty (the first one that did not
    qualify), or at every duty for a shortfall."""
    def why(d):
        lb = limbs[d]
        out = []
        if not lb["counted"]:
            out.append("sanity failed")
        if lb["sing44"] == "fail":
            out.append("sing44 limb fails (S_late " + (_fmt(lb["S_late"], 3) if lb["reached"] else
                                                      f"{_fmt(lb['S_late'], 3)}, v* never reached") + ")")
        if lb["comp21"] == "fail":
            out.append(f"comp21 limb fails (P {_fmt(lb['comp21_P'], 3)})")
        return ", ".join(out) or "qualifies"
    sing_na = all(lb["sing44"] == "n/a" for lb in limbs.values())
    tail = " (sing44 limb n/a: the floor rests on comp21)" if sing_na else ""
    if shortfall:
        return "shortfall — " + "; ".join(f"d = {d}: {why(d)}" for d in DUTIES) + tail
    larger = [d for d in DUTIES if d > d_min]
    if not larger:
        return f"d = {d_min} (the largest duty) qualifies" + tail
    return f"d = {min(larger)}: {why(min(larger))}" + tail


def a6_spread(runs: pd.DataFrame, inst: str, set_name: str | None, budget: float) -> tuple[float, int]:
    """A6's seed spread of P on an instance from runs OUTSIDE the given set at the same budget: max − min over seeds 0
    and 1 when both exist, else over every seed present. (spread, number of seeds)."""
    df = runs[(runs["instance"] == inst) & (runs["arm"] == A6) & (runs["status"] == "completed")]
    if set_name is not None:
        df = df[df["set"] != set_name]
    if _fin(budget) and "budget" in df:
        df = df[np.isclose(df["budget"].astype(float), budget)]
    p = df.groupby("seed")["pt"].mean()
    if {0, 1} <= set(p.index):
        p = p.loc[[0, 1]]
    return (float(p.max() - p.min()) if len(p) >= 2 else 0.0), int(len(p))


# ================================================================================================= stage 1
def _pmeans(R, inst: str, arm: str, seeds=STAGE1_SEEDS, col: str = "pt") -> dict[int, float]:
    out = {}
    for s in seeds:
        r = R.get((inst, arm, s))
        if r is not None and _fin(r.get(col)):
            out[s] = float(r.get(col))
    return out


def _d(R, inst, arm, ctrl, col="pt") -> tuple[float, float, float]:
    """(d, mean P arm, mean P ctrl) on the seeds both have (positive = arm better)."""
    a, c = _pmeans(R, inst, arm, col=col), _pmeans(R, inst, ctrl, col=col)
    common = sorted(set(a) & set(c))
    if not common:
        return math.nan, math.nan, math.nan
    pa, pc = float(np.mean([a[s] for s in common])), float(np.mean([c[s] for s in common]))
    return effect_d(pc, pa), pa, pc


def _band(R, inst, ctrl: str = CTRL1, a6: str = A6) -> float:
    """band_i = max(0.05, the larger seed-0 vs seed-1 bounded difference of the CONTROL or of A6) (D-018). The control
    is CTRL1 since 2026-10-09; until 2026-10-11 this read BO0, which stage 1 never runs, so every band fell back to A6's
    spread (or 0.05) — fixed before the stage-1 verdict was recorded (journal 2026-10-11). Stage 1y: the same control,
    A6F in A6's place."""
    vals = []
    for arm in (ctrl, a6):
        p = _pmeans(R, inst, arm)
        if 0 in p and 1 in p:
            vals.append(bounded(p[0], p[1]))
    vals = [v for v in vals if _fin(v)]
    return max([BAND_MIN] + vals)


def _all_inc(inc, run_id, T) -> list[tuple[float, float]]:
    if inc is None or not len(inc):
        return []
    g = inc[(inc["run_id"] == run_id) & inc["ok"] & (inc["t"] <= T)]
    return sorted(zip(g["t"].astype(float), g["obj_recomputed"].astype(float)))


def _effects(R, treat: str = BO1, ctrl: str = CTRL1, a6: str = A6) -> dict:
    """Per instance of STAGE1: d of the treatment vs the control (seed means), both against the cuOpt-alone arm, the
    band. The keys keep stage 1's names: `*_bo1` = the treatment, `*_bo0` = the control, `*_A6` = the cuOpt-alone arm
    (A6F in stage 1y)."""
    eff = {}
    for i in STAGE1:
        d, pa, pc = _d(R, i, treat, ctrl)
        d0a6, _, _ = _d(R, i, ctrl, a6)
        d1a6, _, _ = _d(R, i, treat, a6)
        p0 = _pmeans(R, i, ctrl)
        eff[i] = dict(d=d, p_bo1=pa, p_bo0=pc, dP=pc - pa if _fin(pa) and _fin(pc) else math.nan,
                      band=_band(R, i, ctrl, a6), d_bo0_vs_A6=d0a6, d_bo1_vs_A6=d1a6,
                      bo0_spread=abs(p0[0] - p0[1]) if 0 in p0 and 1 in p0 else math.nan)
    return eff


def _m1_rows(R, treat: str = BO1, limit_s: float = M1_S) -> tuple[list[dict], list[dict]]:
    """M1's rows: phase Y of every treatment run ends within `limit_s` of the first posted partner incumbent; and the
    runs where no partner posted (the rule does not apply: n/a)."""
    m1, na = [], []
    for i in STAGE1:
        for s in STAGE1_SEEDS:
            r = R.get((i, treat, s))
            if r is None:
                continue
            tp = r.first_post()
            if not _fin(tp):
                na.append(dict(instance=i, seed=s, exit_t=r.phase_y()[0], reason=r.phase_y()[1]))
                continue
            tx, why = r.phase_y()
            ok = _fin(tx) and tx <= tp + limit_s
            m1.append(dict(instance=i, seed=s, partner_t=tp, exit_t=tx, reason=why, ok=bool(ok)))
    return m1, na


def _m2(R, treat: str = BO1, a6: str = A6, tlabel: str = "bo1", alabel: str = "A6") -> dict:
    """M2: chromaticindex's cuOpt first incumbent inside the treatment <= M2_X x the cuOpt-alone arm's, both seeds (a
    censored treatment time fails unless the reference's is censored too — reported). Row keys keep stage 1's names
    (`t_bo1` = the treatment, `t_A6` = the cuOpt-alone arm)."""
    m2 = []
    for s in STAGE1_SEEDS:
        r, a = R.get((CHROMATIC, treat, s)), R.get((CHROMATIC, a6, s))
        if r is None or a is None:
            m2.append(dict(seed=s, ok=False, why="missing run"))
            continue
        tb, ta = r.cuopt_t_first(), a.cuopt_t_first()          # ∞: censored at the budget (cuOpt found nothing)
        cb, ca = math.isinf(tb), math.isinf(ta)
        if math.isnan(tb) or math.isnan(ta):
            m2.append(dict(seed=s, t_bo1=tb, t_A6=ta, ok=False, why="no cuOpt time"))
        elif cb and ca:
            m2.append(dict(seed=s, t_bo1=tb, t_A6=ta, ok=True,
                           why=f"both censored: cuOpt found nothing in {tlabel} nor in {alabel} (reported, not "
                               "failed)"))
        else:
            m2.append(dict(seed=s, t_bo1=tb, t_A6=ta, ok=bool(not cb and (ca or tb <= M2_X * ta)),
                           why=f"{tlabel} censored" if cb else None))
    return dict(ok=all(x["ok"] for x in m2), rows=m2)


def _m3_d018(R, treat: str = BO1, ctrl: str = CTRL1) -> dict:
    """D-018's M3: per seed, the treatment's material leads >= M3_LEADS x the control's, its first lead not later than
    the control's + max(M3_FIRST_S, M3_FIRST_F x it), and on item_placement level 0 for >= M3_L0 of the time up to the
    control's last lead."""
    m3 = []
    for i in M3_SET:
        for s in STAGE1_SEEDS:
            r, c = R.get((i, treat, s)), R.get((i, ctrl, s))
            if r is None or c is None:
                m3.append(dict(instance=i, seed=s, ok=False, why="missing run"))
                continue
            nr, nc = len(r.leads()), len(c.leads())
            fr, fc = r.first_lead(), c.first_lead()
            ok_n = nr >= M3_LEADS * nc
            ok_f = (not _fin(fc)) or (_fin(fr) and fr <= fc + max(M3_FIRST_S, M3_FIRST_F * fc))
            row = dict(instance=i, seed=s, leads_bo1=nr, leads_bo0=nc, first_bo1=fr, first_bo0=fc,
                       leads_ok=bool(ok_n), first_ok=bool(ok_f))
            ok = ok_n and ok_f
            if i == "item_placement_9903":
                cl = c.leads()
                t_last = cl[-1] if cl else math.nan
                share = r.full_duty_share(t_last) if _fin(t_last) else math.nan
                ok_l0 = (not _fin(t_last)) or (_fin(share) and share >= M3_L0)
                row.update(control_last_lead=t_last, l0_share=share, l0_ok=bool(ok_l0))
                ok = ok and ok_l0
            row["ok"] = bool(ok)
            m3.append(row)
    return dict(ok=all(x["ok"] for x in m3), rows=m3)


def _cap_per_call(v: RunView) -> float:
    """Local-search cap hits per call of a run's EA (0 without calls)."""
    sm = v.summary()
    n = _f(sm.get("ls_calls"))
    return _f(sm.get("ls_cap_hits", 0)) / n if n > 0 else 0.0


def _accounting_rows(r: RunView, i: str, s: int) -> tuple[list[dict], list[dict]]:
    """M5's accounting legs on one treatment run's `rclock` events: generation seconds + paused seconds = the
    governor's wall seconds within max(M5_ACC x wall, 1 ms), and the repair clock's overhead o = gen − capped-call
    seconds never negative."""
    acc, o_s = [], []
    for e in r.kind("rclock"):
        if "bo_wall_s" not in e:
            continue
        w, g, p = _f(e.get("bo_wall_s")), _f(e.get("gen_s")), _f(e.get("bo_paused_s"))
        if _fin(w) and abs(g + p - w) > max(M5_ACC * w, 1e-3):
            acc.append(dict(instance=i, seed=s, gen=e.get("gen"), wall=w, gen_s=g, paused=p))
        cf = sum(_f(c.get("s")) for c in (e.get("calls") or []) if c.get("cf"))
        if _fin(g) and g - cf < -1e-3:
            o_s.append(dict(instance=i, seed=s, gen=e.get("gen"), o_raw=g - cf))
    return acc, o_s


def _m5_d018(R, treat: str = BO1, ctrl: str = CTRL1, a6: str = A6) -> dict:
    """D-018's M5: accounting, o >= 0, cap hits per call not above the control's, no overrun past the stop, every
    incumbent checked, GPU health ok."""
    m5 = dict(accounting=[], o_s=[], cap_hits=[], overrun=[], checked=[], gpu=[])
    for (i, a, s), r in sorted(R.items()):
        if a not in (ctrl, treat, a6):
            continue
        if a in (ctrl, treat):
            if int(r.get("n_inc_unchecked", 0)) > 0 or int(r.get("n_inc_failed", 0)) > 0:
                m5["checked"].append(dict(instance=i, arm=a, seed=s, unchecked=r.get("n_inc_unchecked"),
                                          failed=r.get("n_inc_failed")))
        if str(r.get("gpu_health", "ok") or "ok") not in ("ok",) or bool(r.get("gpu_void", False)):
            m5["gpu"].append(dict(instance=i, arm=a, seed=s, gpu_health=r.get("gpu_health")))
        if a != treat:
            continue
        acc, o_s = _accounting_rows(r, i, s)
        m5["accounting"] += acc
        m5["o_s"] += o_s
        c = R.get((i, ctrl, s))
        if c is not None:
            if _cap_per_call(r) > _cap_per_call(c) + 1e-12:
                m5["cap_hits"].append(dict(instance=i, seed=s, bo1=_cap_per_call(r), bo0=_cap_per_call(c)))
        sm = r.summary()
        wall, budget = _f(sm.get("wall_s")), _f(r.get("budget"))
        stop = r.run_json().get("stop") or {}
        end_ref = budget
        if isinstance(stop, dict) and _fin(stop.get("t")) and stop.get("kind") not in ("budget",):
            end_ref = min(budget, float(stop["t"]))
        slack = _f(r.bo.get("max_launch_s")) + M5_STOP
        if _fin(wall) and _fin(end_ref) and wall - end_ref > (slack if _fin(slack) else M5_STOP):
            m5["overrun"].append(dict(instance=i, seed=s, wall=wall, end_ref=end_ref, slack=slack))
    return dict(ok=not any(m5.values()), **m5)


def _outcome(R, inc, eff: dict, treat: str = BO1, a6: str = A6) -> tuple[dict, list, list, list]:
    """D-018 stage 1's outcome rules: T, G (with the noise rule, the guard-set mean and "still beats A6"), the ladder
    targets, F and the mean guard over the 11 non-floor instances. Returns (out, guard rows, ladder rows, noise)."""
    out: dict = {}
    e = eff[CHROMATIC]
    out["T"] = dict(ok=bool(_fin(e["d"]) and e["d"] >= T_D and _fin(e["p_bo1"]) and e["p_bo1"] <= T_P),
                    d=e["d"], p_bo1=e["p_bo1"])

    def noise(i) -> bool:
        rs = [R.get((i, treat, s)) for s in STAGE1_SEEDS]
        rs = [r for r in rs if r is not None]
        return bool(rs) and all(r.max_level_p() == 0 and _fin(r.phase_y()[0]) and r.phase_y()[0] < 1.0 for r in rs)
    g_rows, noise_list = [], []
    for i in GUARDS_D:
        e = eff[i]
        ok = _fin(e["d"]) and e["d"] >= -e["band"]
        if not ok and noise(i):
            noise_list.append(i)
        g_rows.append(dict(instance=i, d=e["d"], band=e["band"], ok=bool(ok)))
    e = eff[GUARD_960]
    lim960 = -max(P960_MIN, e["bo0_spread"] if _fin(e["bo0_spread"]) else 0.0)
    ok960 = _fin(e["dP"]) and e["dP"] >= lim960
    if not ok960 and noise(GUARD_960):
        noise_list.append(GUARD_960)
    gd = [eff[i]["d"] for i in GUARDS if _fin(eff[i]["d"])]
    gmean = float(np.mean(gd)) if gd else math.nan
    beat = [dict(instance=i, d_bo0_vs_A6=eff[i]["d_bo0_vs_A6"], d_bo1_vs_A6=eff[i]["d_bo1_vs_A6"],
                 ok=bool(_fin(eff[i]["d_bo1_vs_A6"]) and eff[i]["d_bo1_vs_A6"] >= BEAT_A6))
            for i in GUARDS if _fin(eff[i]["d_bo0_vs_A6"]) and eff[i]["d_bo0_vs_A6"] >= BEAT_A6]
    for b in beat:
        if not b["ok"] and b["instance"] not in noise_list and noise(b["instance"]):
            noise_list.append(b["instance"])
    # the noise rule: a per-instance guard failure is reported but not counted; the guard-set mean is not exempt
    counted = [r for r in g_rows if r["instance"] not in noise_list]
    g_ok = (all(r["ok"] for r in counted) and (ok960 or GUARD_960 in noise_list)
            and _fin(gmean) and gmean >= GUARD_MEAN and all(b["ok"] for b in beat if b["instance"] not in noise_list))
    out["G"] = dict(ok=bool(g_ok), rows=g_rows, neos960=dict(dP=e["dP"], limit=lim960, ok=bool(ok960)),
                    mean_d=gmean, still_beats_A6=beat, noise=noise_list)
    l_rows = [dict(instance=i, d=eff[i]["d"], band=eff[i]["band"],
                   ok=bool(_fin(eff[i]["d"]) and eff[i]["d"] >= -eff[i]["band"])) for i in LADDER]
    ld = [r["d"] for r in l_rows if _fin(r["d"])]
    out["ladder"] = dict(ok=all(r["ok"] for r in l_rows), rows=l_rows, mean_d=float(np.mean(ld)) if ld else math.nan)
    f_rows = []
    for i in FLOOR:
        e = eff[i]
        row = dict(instance=i, dP=e["dP"], ok=bool(_fin(e["dP"]) and abs(e["dP"]) <= FLOOR_DP))
        row["seconds"] = floor_seconds(R, inc, i, treat=treat, a6=a6)
        f_rows.append(row)
    out["F"] = dict(ok=all(r["ok"] for r in f_rows), rows=f_rows)
    nf = [eff[i]["d"] for i in STAGE1 if i not in FLOOR and _fin(eff[i]["d"])]
    out["mean_guard"] = dict(ok=bool(nf and float(np.mean(nf)) >= 0.0), mean_d=float(np.mean(nf)) if nf else math.nan,
                             n=len(nf))
    return out, g_rows, l_rows, noise_list


def _classify(R, runs: pd.DataFrame, eff: dict, out: dict, m2_ok: bool, g_rows, l_rows, noise_list,
              stage0_set: str | None, root: Path, treat: str = BO1, ctrl: str = CTRL1, tlabel: str = "bo1") -> list:
    """The failure classes D-018 names (the ca / glass4 / sing44 fallbacks, "T fails with M2 failing", "inside the
    band")."""
    cls: list[str] = []
    ca_row = next(r for r in g_rows if r["instance"] == CA)
    if not ca_row["ok"] and CA not in noise_list:
        cls.append(classify_ca(R, runs, stage0_set, root, treat=treat, ctrl=ctrl, tlabel=tlabel))
    for i in (GLASS4, SING44):
        row = next(r for r in g_rows if r["instance"] == i)
        if not row["ok"] and i not in noise_list:
            if any(_esc_before_lead(R.get((i, treat, s))) for s in STAGE1_SEEDS):
                po = None
                for s in STAGE1_SEEDS:
                    r = R.get((i, treat, s))
                    if r is not None and r.bo:
                        po = _f(r.bo.get("patience_own_s"))
                nxt = min(po + 10.0, 50.0) if _fin(po) else None
                cls.append(f"{i} guard failed with a step down before its first lead: next test patience_own_s = "
                           f"{nxt} (+10, <= 50)")
            else:
                cls.append(f"{i} guard failed without a step down before its first lead (no pre-stated fallback)")
    if not out["T"]["ok"] and not m2_ok:
        cls.append("T fails with M2 failing: duty does not restore cuOpt at run level although stage 0 said it "
                   "would. STOP and report; MPS needs Berk.")
    inside = [i for i in TARGET + LADDER if _fin(eff[i]["d"]) and abs(eff[i]["d"]) <= eff[i]["band"]
              and not (out["T"]["ok"] if i in TARGET else next(r for r in l_rows if r["instance"] == i)["ok"])]
    if inside:
        cls.append("inside the band: seeds 3 and 4 on " + ", ".join(inside) + ", same rules")
    return cls


def stage1(runs: pd.DataFrame, inc: pd.DataFrame | None, set_name: str = "backoff1", root: Path = ROOT,
           stage0_set: str | None = "backoff0") -> dict:
    """The stage-1 criteria (module docstring). Returns {mechanism, outcome, classification, reports, verdict}."""
    R = load(runs, set_name, root)
    notes: list[str] = []
    missing = [(i, a, s) for i in STAGE1 for a in (CTRL1, BO1, A6) for s in STAGE1_SEEDS if (i, a, s) not in R]
    d_min = None
    for (i, a, s), r in R.items():
        if a == BO1 and r.bo:
            d_min = 2.0 ** -int(r.bo.get("levels", 2))
            break
    # ------------------------------------------------------------------------ effects
    eff = _effects(R, BO1, CTRL1, A6)
    # ------------------------------------------------------------------------ mechanism
    mech: dict = {}
    # M1
    m1, _ = _m1_rows(R, BO1, M1_S)
    mech["M1"] = dict(ok=bool(m1) and all(x["ok"] for x in m1), rows=m1)
    # M2
    mech["M2"] = _m2(R, BO1, A6)
    # M3
    mech["M3"] = _m3_d018(R, BO1, CTRL1)
    # M4
    m4, util = [], {}
    for i in M4_SET:
        u1, u0 = [], []
        for s in STAGE1_SEEDS:
            r, c = R.get((i, BO1, s)), R.get((i, CTRL1, s))
            if r is None or c is None:
                continue
            if any(t > M4_LATE for t in c.leads()):
                m4.append(dict(instance=i, seed=s, applies=False))
                continue
            sh1, st1 = r.window_rates(M4_FROM)
            _, st0 = c.window_rates(M4_FROM)
            ok_sh = _fin(sh1) and d_min is not None and sh1 <= d_min + M4_SHARE
            ok_st = _fin(st1) and _fin(st0) and st1 <= M4_STEPS * st0
            m4.append(dict(instance=i, seed=s, applies=True, busy_share=sh1, steps_bo1=st1, steps_bo0=st0,
                           share_ok=bool(ok_sh), steps_ok=bool(ok_st), ok=bool(ok_sh and ok_st)))
            u1.append(r.mean_util())
            u0.append(c.mean_util())
        if u1:
            util[i] = dict(bo1=float(np.nanmean(u1)), bo0=float(np.nanmean(u0)),
                           lower=bool(np.nanmean(u1) < np.nanmean(u0)))
    app = [x for x in m4 if x.get("applies")]
    need = min(M4_UTIL, _ceil(0.75 * len(util))) if util else 0
    n_lower = sum(1 for v in util.values() if v["lower"])
    mech["M4"] = dict(ok=all(x["ok"] for x in app) and n_lower >= need, rows=m4, util=util,
                      util_lower=n_lower, util_needed=need, D_min=d_min)
    # M5
    mech["M5"] = _m5_d018(R, BO1, CTRL1, A6)
    # M6 (reported): the control's would-be first step down against its first lead, guard instances
    m6 = []
    for i in GUARDS:
        for s in STAGE1_SEEDS:
            c = R.get((i, CTRL1, s))
            if c is None:
                continue
            te, be = c.first_escalation()
            m6.append(dict(instance=i, seed=s, would_escalate_t=te, would_escalate_busy=be, first_lead=c.first_lead(),
                           before_lead=bool(_fin(te) and (not _fin(c.first_lead()) or te < c.first_lead()))))
    mech["M6"] = dict(rows=m6)
    mech_ok = all(mech[k]["ok"] for k in ("M1", "M2", "M3", "M4", "M5"))
    # ------------------------------------------------------------------------ outcome
    out, g_rows, l_rows, noise_list = _outcome(R, inc, eff, BO1, A6)
    # ------------------------------------------------------------------------ classification of failures
    cls = _classify(R, runs, eff, out, mech["M2"]["ok"], g_rows, l_rows, noise_list, stage0_set, root, BO1, CTRL1)
    # ------------------------------------------------------------------------ reports
    reports = dict(d013=d013_report(R, runs, STAGE1), relay=relay_latency(R), lp_arrival=lp_arrival(R))
    adopt = mech_ok and out["T"]["ok"] and out["G"]["ok"] and out["F"]["ok"] and out["mean_guard"]["ok"]
    if missing:
        verdict = f"INCOMPLETE: {len(missing)} missing runs (first: {missing[:3]})"
    elif adopt:
        verdict = "ADOPT bo1 as A9's development default: T, G, F, the mean guard and M1–M5 hold"
    else:
        failed = [k for k in ("M1", "M2", "M3", "M4", "M5") if not mech[k]["ok"]] + \
                 [k for k in ("T", "G", "F", "mean_guard") if not out[k]["ok"]]
        verdict = "DO NOT ADOPT: failed " + ", ".join(failed)
    return dict(stage=1, set=set_name, effects=eff, mechanism=mech, outcome=out, classification=cls, reports=reports,
                verdict=verdict, notes=notes, missing=missing)


# ================================================================================================= stage 1y (D-023)
def _phase_y_share(r: RunView) -> tuple[float, float, str]:
    """(wall seconds, busy seconds) of a run's phase Y and its exit reason, from the EA summary's `phase_y` block: the
    exit's wall time `exit_t` and busy time `exit_busy_s` (the governor's own record of the exit). A phase Y that
    never ended (reason "never") is read from `length_s` and `busy_per_level["Y"]`. NaN when the run has no record."""
    py = r.bo.get("phase_y") if isinstance(r.bo.get("phase_y"), dict) else {}
    w, b = _f(py.get("exit_t")), _f(py.get("exit_busy_s"))
    if not _fin(w):
        w = _f(py.get("length_s"))
        b = _f((r.bo.get("busy_per_level") or {}).get("Y"))
    return w, b, str(py.get("reason") or "")


def _run_harness(row: dict, root: Path) -> dict | None:
    """A run's resolved harness: run.json `config.harness` (the runner's record), else the run table's `harness`
    column (JSON; "" = none); None without one."""
    p = Path(str(row.get("run_dir") or ""))
    d = p if p.is_absolute() else root / p
    try:
        rj = json.loads((d / "run.json").read_text())
    except (OSError, ValueError):
        rj = {}
    cfg = rj.get("config") if isinstance(rj, dict) and isinstance(rj.get("config"), dict) else {}
    h = cfg.get("harness")
    if h is None and isinstance(row.get("harness"), str) and row["harness"]:
        try:
            h = json.loads(row["harness"])
        except ValueError:
            h = None
    return h if isinstance(h, dict) else None


def harness_check(runs: pd.DataFrame, set_name: str | None, root: Path, expect: str | None = None) -> dict:
    """Stage 1y's harness leg (module docstring): over every completed, not voided, not synthetic run of the set's
    three arms — every attempt — the A9 arms (boY and the control) share one resolved harness, A6F runs under ext, and
    the A9 harness is `expect`'s ("ext" / "r13-off", D-022's outcome) when given."""
    from itrea.experiments.make_jobs import ITX_EXT, ITX_R13_CONTROL
    from itrea.experiments.pair import resolve_harness
    named = {"ext": resolve_harness(ITX_EXT), "r13-off": resolve_harness(ITX_R13_CONTROL)}
    df = runs if set_name is None else runs[runs["set"] == set_name]

    def flag(col):
        return df[col].fillna(False).astype(bool) if col in df else pd.Series(False, index=df.index)
    df = df[(df["status"] == "completed") & ~flag("block_void") & ~flag("synthetic") & df["arm"].isin([BOY, CTRL1,
                                                                                                         A6F])]
    a9: dict[str, list] = {}
    a6_bad = []
    for r in df.sort_values(["instance", "arm", "seed"], kind="mergesort").to_dict("records"):
        h = _run_harness(r, root)
        k = (str(r["instance"]), str(r["arm"]), int(r["seed"]))
        if r["arm"] == A6F:
            if h != named["ext"]:
                a6_bad.append(dict(instance=k[0], seed=k[2], harness=h))
        else:
            a9.setdefault(json.dumps(h, sort_keys=True), []).append(k)
    groups = [dict(harness=json.loads(j), name=next((n for n, v in named.items() if v == json.loads(j)), None),
                   n=len(ks), runs=ks[:6]) for j, ks in sorted(a9.items(), key=lambda x: -len(x[1]))]
    one = groups[0] if len(groups) == 1 else None
    problems = []
    if len(groups) > 1:
        problems.append("the A9 arms run under " + str(len(groups)) + " harnesses: " + "; ".join(
            f"{g['name'] or g['harness']} ({g['n']} runs, e.g. {g['runs'][:2]})" for g in groups))
    if one is not None and one["harness"] is None:
        problems.append("the A9 runs carry no harness")
    if one is not None and one["harness"] is not None and one["name"] is None:
        problems.append(f"the A9 harness is neither D-022 outcome's: {one['harness']}")
    if expect is not None and one is not None and one["name"] is not None and one["name"] != expect:
        problems.append(f"the A9 harness is {one['name']}, not D-022's outcome {expect}")
    if a6_bad:
        problems.append(f"{len(a6_bad)} A6F runs not under ext (D-021 R13 / R6): "
                        f"{[(x['instance'], x['seed']) for x in a6_bad[:4]]}")
    return dict(ok=not problems, a9=one["name"] if one else None, a9_harness=one["harness"] if one else None,
                a9_groups=groups, a6f_not_ext=a6_bad, expect=expect, problems=problems)


def stage1y(runs: pd.DataFrame, inc: pd.DataFrame | None, set_name: str = "backoff2", root: Path = ROOT,
            stage0_set: str | None = "backoff0", harness: str | None = None) -> dict:
    """D-023: A9@boY (phase Y only) vs the control A9@bo1s, same-day A6F; D-018 stage 1's outcome rules unchanged and
    D-023's mechanism rules M1–M5 (module docstring). D-018's own M1 / M3 / M5 on the same runs are returned under
    `mechanism_d018` (reported, not gating). `harness`: D-022's outcome ("ext" / "r13-off") the A9 arms must carry,
    or None (the harness found is reported and must still be one for both A9 arms, ext for A6F). Returns the stage-1
    keys plus `mechanism_d018` and `harness`."""
    if harness is not None and harness not in Y_HARNESS:
        raise ValueError(f"harness must be one of {Y_HARNESS}, not {harness!r}")
    R = load(runs, set_name, root)
    hz = harness_check(runs, set_name, root, harness)
    notes: list[str] = []
    missing = [(i, a, s) for i in STAGE1 for a in (CTRL1, BOY, A6F) for s in STAGE1_SEEDS if (i, a, s) not in R]
    eff = _effects(R, BOY, CTRL1, A6F)
    mech: dict = {}
    # M1: phase Y ends within 1.0 s of the first partner incumbent, every boY run where a partner posted
    m1, m1_na = _m1_rows(R, BOY, Y_M1_S)
    mech["M1"] = dict(ok=bool(m1) and all(x["ok"] for x in m1), rows=m1, n_a=m1_na, limit_s=Y_M1_S)
    # M2: chromaticindex's cuOpt first incumbent inside boY <= 2 x A6F's, both seeds
    mech["M2"] = _m2(R, BOY, A6F, "boY", "A6F")
    # M3: material leads under boY >= 0.8 x the control's, summed over the seeds, on the four lead instances
    m3 = []
    for i in M3_SET:
        nr = nc = 0
        miss = []
        for s in STAGE1_SEEDS:
            r, c = R.get((i, BOY, s)), R.get((i, CTRL1, s))
            if r is None or c is None:
                miss.append(s)
                continue
            nr += len(r.leads())
            nc += len(c.leads())
        ok = not miss and nr >= Y_M3_LEADS * nc
        m3.append(dict(instance=i, leads_boY=nr, leads_ctrl=nc, ratio=nr / nc if nc else math.nan,
                       missing_seeds=miss, ok=bool(ok)))
    mech["M3"] = dict(ok=all(x["ok"] for x in m3), rows=m3)
    # M4: boY's busy share in phase Y <= 0.6, pooled over the arm's runs (per run reported)
    m4, no_rec = [], []
    for (i, a, s), r in sorted(R.items()):
        if a != BOY:
            continue
        w, b, why = _phase_y_share(r)
        if not (_fin(w) and _fin(b)):
            no_rec.append(dict(instance=i, seed=s))
            continue
        m4.append(dict(instance=i, seed=s, wall_s=w, busy_s=b, share=b / w if w > 0 else math.nan, reason=why))
    W, B = sum(x["wall_s"] for x in m4), sum(x["busy_s"] for x in m4)
    share = B / W if W > 0 else math.nan
    mech["M4"] = dict(ok=bool(_fin(share) and share <= Y_M4_SHARE), busy_share=share, wall_s=W, busy_s=B,
                      limit=Y_M4_SHARE, rows=m4, no_record=no_rec)
    # M5: the busy / paused accounting within 2 %, and cap hits per call within +0.10 of the control's
    acc, cap = [], []
    for (i, a, s), r in sorted(R.items()):
        if a != BOY:
            continue
        acc += _accounting_rows(r, i, s)[0]
        c = R.get((i, CTRL1, s))
        if c is not None and _cap_per_call(r) > _cap_per_call(c) + Y_M5_CAP + 1e-9:
            cap.append(dict(instance=i, seed=s, boY=_cap_per_call(r), ctrl=_cap_per_call(c)))
    mech["M5"] = dict(ok=not acc and not cap, accounting=acc, cap_hits=cap, cap_slack=Y_M5_CAP)
    mech_ok = all(mech[k]["ok"] for k in ("M1", "M2", "M3", "M4", "M5"))
    # D-018's own M1 / M3 / M5 on the same runs, for comparison (not gating)
    d1, _ = _m1_rows(R, BOY, M1_S)
    d018 = dict(M1=dict(ok=bool(d1) and all(x["ok"] for x in d1), rows=d1, limit_s=M1_S),
                M3=_m3_d018(R, BOY, CTRL1), M5=_m5_d018(R, BOY, CTRL1, A6F))
    # outcome: D-018 stage 1's rules unchanged; the ladder targets gate here (D-023: every outcome rule must hold)
    out, g_rows, l_rows, noise_list = _outcome(R, inc, eff, BOY, A6F)
    cls = _classify(R, runs, eff, out, mech["M2"]["ok"], g_rows, l_rows, noise_list, stage0_set, root, BOY, CTRL1,
                    "boY")
    reports = dict(d013=d013_report(R, runs, STAGE1, a6=A6F, arms=(CTRL1, BOY)),
                   relay=relay_latency(R, (CTRL1, BOY)), lp_arrival=lp_arrival(R, (CTRL1, BOY)))
    gates = ("T", "G", "ladder", "F", "mean_guard")
    adopt = mech_ok and all(out[k]["ok"] for k in gates)
    if not hz["ok"]:
        verdict = "INVALID (harness: " + "; ".join(hz["problems"]) + "); the rules below are not decided"
    elif missing:
        verdict = f"INCOMPLETE: {len(missing)} missing runs (first: {missing[:3]})"
    elif adopt:
        verdict = ("ADOPT boY as A9's development default: M1–M5 (D-023), T, G, the ladder targets, F and the mean "
                   "guard hold")
    else:
        failed = [k for k in ("M1", "M2", "M3", "M4", "M5") if not mech[k]["ok"]] + \
                 [k for k in gates if not out[k]["ok"]]
        verdict = "DO NOT ADOPT boY: failed " + ", ".join(failed)
    notes.append("the ladder targets gate the verdict here; D-018 stage 1 reported them without gating (design §5: "
                 "adopt if T, G, F, the mean guard and M1–M5 hold) — D-023 asks every outcome rule to hold")
    notes.append(f"M4 reads the arm's pooled phase-Y busy share (sum of exit_busy_s / sum of exit_t over every boY "
                 f"run); a phase Y shorter than ~2 x burst_s ends before the first sleep falls due, so a per-run share "
                 f"is 1.0 by construction there (per run listed)")
    if harness is None and hz["ok"]:
        notes.append(f"the A9 harness ({hz['a9']}) was not checked against D-022's recorded outcome (--harness)")
    return dict(stage="1y", set=set_name, effects=eff, mechanism=mech, mechanism_d018=d018, outcome=out,
                classification=cls, reports=reports, verdict=verdict, notes=notes, missing=missing, harness=hz)


def run_stage1y(runs: pd.DataFrame, inc: pd.DataFrame | None, set_name: str = "backoff2", root: Path = ROOT,
                stage0_set: str | None = "backoff0", failed: str = "both", harness: str | None = None) -> dict:
    """Stage 1y under a reading of the set's failed runs (D-023: "read as D-002 §6 keeps crashes, reported both
    ways"): "keep" (kept, truncated at the crash, as build-runs scores them), "exclude" (missing: the verdict is then
    INCOMPLETE) or "both" (the kept reading in full, the excluded reading's verdict and missing runs under
    `exclude_reading`). Without failed runs the readings coincide and one is returned."""
    sel = (runs["set"] == set_name) & (runs["status"] == "failed")
    failed_ids = sorted(runs.loc[sel, "run_id"].astype(str))
    if failed == "exclude" or not failed_ids:
        res = stage1y(runs, inc, set_name, root, stage0_set, harness)
        res["kept_failed"] = []
        if failed_ids:
            res["notes"].append(f"{len(failed_ids)} failed runs excluded (reading 'exclude'): {failed_ids}")
        return res
    kept = runs.copy()
    kept.loc[sel, "status"] = "completed"
    res = stage1y(kept, inc, set_name, root, stage0_set, harness)
    res["kept_failed"] = failed_ids
    if failed == "both":
        ex = stage1y(runs, inc, set_name, root, stage0_set, harness)
        res["exclude_reading"] = dict(verdict=ex["verdict"], missing=ex["missing"])
    return res


def _esc_before_lead(r: RunView | None) -> bool:
    if r is None:
        return False
    te, _ = r.first_escalation()
    tl = r.first_lead()
    return bool(_fin(te) and (not _fin(tl) or te < tl))


def classify_ca(R, runs: pd.DataFrame, stage0_set: str | None, root: Path = ROOT, treat: str = BO1,
                ctrl: str = CTRL1, tlabel: str = "bo1") -> str:
    """The design's ca diagnosis from the events: (P) the treatment stepped down before the EA's first ca lead -> next
    test patience_search_s = 1.25 x the control's lag; (Y) no such step and phase Y >= 10 s -> D_Y one step up,
    accepted only if stage 0 showed S_first <= 3 at that duty (duty 1 = stage 0's bo0, the shadow run); otherwise
    unclassified."""
    rs = [R.get((CA, treat, s)) for s in STAGE1_SEEDS]
    cs = [R.get((CA, ctrl, s)) for s in STAGE1_SEEDS]
    if any(_esc_before_lead(r) for r in rs):
        lags = [c.first_lead() - c.first_post() for c in cs if c is not None and _fin(c.first_lead())
                and _fin(c.first_post())]
        nxt = 1.25 * max(lags) if lags else None
        return (f"ca guard failed, class (P): {tlabel} stepped down before the EA's first ca lead — next test "
                f"patience_search_s = 1.25 x the control's lag = {nxt}, nothing else changed")
    ylen = [r.phase_y()[0] for r in rs if r is not None and _fin(r.phase_y()[0])]
    if ylen and max(ylen) >= 10.0:
        msg = "ca guard failed, class (Y): no step down before the first lead and phase Y >= 10 s — the phase-Y trade."
        s0 = None
        if stage0_set:
            try:
                s0 = stage0(runs, None, stage0_set, root)
            except Exception:  # noqa: BLE001 — the stage-0 runs may not be in this table
                s0 = None
        level = None
        for r in rs:
            if r is not None and r.bo:
                level = int(r.bo.get("yield_level", 2))
        if s0 is not None and level is not None and level >= 1:
            d_up = 2.0 ** -(level - 1)
            arm = BO0 if d_up >= 1 else {v: k for k, v in FORCED.items()}.get(d_up)
            sf = s0["readouts"].get(arm, {}).get("S_first") if arm else None
            if sf is not None and _fin(sf) and sf <= 3.0:
                return msg + f" Next variable: D_Y one step up ({d_up}; stage-0 S_first {sf:.2f} <= 3)."
            return msg + (f" D_Y one step up ({d_up}) is NOT accepted (stage-0 S_first {sf}); recorded as 'no single "
                          "phase-Y duty serves both' — the deferred progress exit is the next candidate.")
        return msg + " Stage-0 S_first at the next duty is needed to decide (stage-0 set not found)."
    return "ca guard failed, unclassified (neither a step down before the first lead nor phase Y >= 10 s)"


def floor_seconds(R, inc, inst: str, treat: str = BO1, ctrl: str = CTRL1, a6: str = A6) -> dict:
    """fhnw / krka: seconds to cuOpt's first incumbent (checked); radiationm: seconds to within 0.1 % of same-seed A6's
    final best (any process of the run). Stage 1y: A6F in A6's place."""
    out = {}
    for s in STAGE1_SEEDS:
        for a in (ctrl, treat, a6):
            r = R.get((inst, a, s))
            if r is None:
                continue
            if inst != "radiationm18-12-05":
                t = r.cuopt_t_first()
                out[f"{a} s{s}"] = "none" if math.isinf(t) else t      # censored: cuOpt found nothing
                continue
            ref = R.get((inst, a6, s))
            best = _f(ref.get("cuopt_best_obj", ref.get("best_obj"))) if ref is not None else math.nan
            if not _fin(best):
                out[f"{a} s{s}"] = math.nan
                continue
            tol = 1e-3 * max(abs(best), 1e-9)
            sense = int(r.get("sense", 1) or 1)
            hit = [t for t, v in _all_inc(inc, r.get("run_id"), _f(r.get("budget"))) if sense * (v - best) <= tol]
            out[f"{a} s{s}"] = min(hit) if hit else math.nan
    return out


def d013_report(R, runs: pd.DataFrame, insts, a6: str = A6, arms=(CTRL1, BO1)) -> dict:
    """D-013: for SCIP (A0, from any set at the same budget) and cuOpt (same-day A6; A6F in stage 1y), the weak set
    W_B (P_B >= 0.5, seed mean) and both A9 arms on it: share improved, mean P, mean d, worst P; and the mean d over all
    (guard)."""
    out = {}
    budgets = {float(r.get("budget")) for r in R.values() if _fin(r.get("budget"))}
    for B, label in (("A0", "SCIP"), (a6, "cuOpt")):
        pb = {}
        for i in insts:
            if B == a6:
                p = _pmeans(R, i, a6)
            else:
                df = runs[(runs["instance"] == i) & (runs["arm"] == "A0") & (runs["status"] == "completed")]
                if budgets and "budget" in df:
                    df = df[df["budget"].astype(float).isin(budgets)]
                p = {int(s): float(v) for s, v in df.groupby("seed")["pt"].mean().items()}
            if p:
                pb[i] = float(np.mean(list(p.values())))
        weak = [i for i, v in pb.items() if v >= 0.5]
        res = dict(weak_set=weak, n_with_baseline=len(pb))
        for arm in arms:
            pa = {i: float(np.mean(list(_pmeans(R, i, arm).values()))) for i in pb if _pmeans(R, i, arm)}
            ds = {i: effect_d(pb[i], pa[i]) for i in pa}
            w = [i for i in weak if i in pa]
            res[arm] = dict(improved=f"{sum(1 for i in w if pa[i] < pb[i])} of {len(w)}",
                            mean_P_B=float(np.mean([pb[i] for i in w])) if w else math.nan,
                            mean_P_arm=float(np.mean([pa[i] for i in w])) if w else math.nan,
                            mean_d_weak=float(np.mean([ds[i] for i in w])) if w else math.nan,
                            worst_P=max((pa[i] for i in w), default=math.nan),
                            mean_d_all=float(np.mean(list(ds.values()))) if ds else math.nan)
        out[label] = res
    return out


def relay_latency(R, arms=(CTRL1, BO1)) -> dict:
    """Median seconds from a cuOpt `send` to the SCIP incumbent recorded with source "cuopt" for the same objective,
    per A9 arm."""
    out = {}
    for arm in arms:
        lat = []
        for (i, a, s), r in R.items():
            if a != arm:
                continue
            sends = [(_f(e.get("t")), _f(e.get("best_obj"))) for e in r.kind("send", "cuopt")]
            for e in r.kind("incumbent", "solver"):
                if e.get("source") != "cuopt":
                    continue
                t, v = _f(e.get("t")), _f(e.get("obj"))
                prior = [ts for ts, o in sends if _fin(o) and abs(o - v) <= 1e-6 * max(1.0, abs(v)) and ts <= t]
                if prior:
                    lat.append(t - max(prior))
        out[arm] = dict(n=len(lat), median_s=float(np.median(lat)) if lat else math.nan)
    return out


def lp_arrival(R, arms=(CTRL1, BO1)) -> dict:
    """Median arrival time of the HiGHS LP point in the EA (`lp` events, state done), per A9 arm (a confound: a sleeping
    EA frees its core for the LP worker)."""
    out = {}
    for arm in arms:
        ts = [_f(e.get("t")) for (i, a, s), r in R.items() if a == arm for e in r.kind("lp")
              if e.get("state") == "done"]
        out[arm] = dict(n=len(ts), median_s=float(np.median(ts)) if ts else math.nan)
    return out


# ================================================================================================= printing
def _fmt(v, nd=3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        if math.isinf(v):
            return "inf"
        return f"{v:.{nd}f}"
    return str(v)


def print_stage0(res: dict, out=sys.stdout) -> None:
    p = lambda *a: print(*a, file=out)  # noqa: E731
    rd = res["readouts"]
    p(f"Stage 0 — A9 GPU back-off calibration (set {res['set']}, seed {res['seed']})")
    p(f"A6: chromaticindex first cuOpt incumbent {_fmt(rd[A6]['t_first_chromatic'], 2)} s; comp21 P "
      f"{_fmt(rd[A6]['comp21_P'])} (two-seed spread {_fmt(rd[A6]['comp21_spread'])})")
    a2 = rd.get(A2, {})
    p(f"A2 (sing44, zero EA duty): v* = {_fmt(a2.get('v_star'), 6)} (best cuOpt objective by {V_STAR_S:g} s), "
      f"reached at {_fmt(a2.get('t_v_star'), 2)} s")
    p("arm        d      S_first  S_late (model)  S prefix|A6 (n, reported)  comp21 cuOpt P  duty c/s/p          "
      "steps/s ratio c/s/p")
    for arm in (BO0, *FORCED):
        r = rd[arm]
        duty = "/".join(_fmt(r["duty"][i], 2) for i in (CHROMATIC, SING44, COMP21))
        ratio = "/".join(_fmt(res["sanity"].get(f"{arm} {i}", {}).get("steps_ratio"), 2)
                         for i in (CHROMATIC, SING44, COMP21)) if arm != BO0 else "1/1/1"
        late = r["sing44_late"]
        sl = _fmt(r["sing44_S_late"], 2) + ("!" if late.get("reached") is False else "")
        p(f"{arm:<10} {r['d']:<6} {_fmt(r['S_first'], 2):<8} {sl:>6} ({_fmt(r['model_S'], 2)})"
          f"     {_fmt(r['sing44_S_prefix_A6'], 2):>6} ({r['sing44_shared']:>3})"
          f"                {_fmt(r['comp21_cuopt_P']):<15} {duty:<19} {ratio}")
    p("  (S_late! = v* never reached within the budget: budget / t_A2(v*), the limb fails)")
    bad = {k: v for k, v in res["sanity"].items() if not v.get("ok")}
    why = {k: (", " + v["why"]) if "why" in v else "" for k, v in bad.items()}
    p("sanity: " + ("every forced run counts" if not bad else
                    "NOT counted: " + "; ".join(f"{k} (duty {_fmt(v.get('duty'), 3)}, steps ratio "
                                                f"{_fmt(v.get('steps_ratio'), 2)}{why[k]})" for k, v in bad.items())))
    ru = res["rules"]
    p(f"(a) D_Y = {_fmt(ru['a']['D_Y'])}  (qualifying {ru['a']['qualifying']}; S_first <= {S_FIRST_MAX})")
    p(f"(b) D_min = {ru['b']['D_min']} -> levels L = {ru['b']['L']}  (qualifying {ru['b']['qualifying']}; sing44 "
      f"S_late <= {S_SING_MAX}; comp21 limit {_fmt(ru['b']['comp21_limit'])}"
      f"{'; SHORTFALL' if ru['b']['shortfall'] else ''})")
    for d, lb in ru["b"]["limbs"].items():
        p(f"    d = {d}: sing44 limb {lb['sing44']} (S_late {_fmt(lb['S_late'], 3)}), comp21 limb {lb['comp21']} "
          f"(P {_fmt(lb['comp21_P'], 3)}){'' if lb['counted'] else ', NOT counted (sanity)'}")
    p(f"    decided by: {ru['b']['decided_by']}")
    p(f"(c) yield_until_s = {_fmt(ru['c']['yield_until_s'])}")
    p(f"(d) patience_search_s = {_fmt(ru['d'].get('patience_search_s'))}  (ca bo0 first lead "
      f"{_fmt(ru['d'].get('t_first_lead'), 2)} s, first partner incumbent {_fmt(ru['d'].get('t_first_partner'), 2)} s)")
    p(f"(e) patience_own_s = {ru['e']['patience_own_s']}  (lags {ru['e']['lags']})")
    for n in res["notes"]:
        p("note: " + n)
    p("VERDICT: " + res["verdict"])
    if res["bo1"]:
        p("configs/itrea2-dev-rc-3way-bo1.yaml = configs/itrea2-dev-rc-3way-bo0.yaml with backoff:")
        for k, v in res["bo1"].items():
            p(f"  {k}: {str(v).lower() if isinstance(v, bool) else v}")


def print_stage1(res: dict, out=sys.stdout) -> None:
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"Stage 1 — {BO1} (active) vs {CTRL1} (shadow control), set {res['set']}")
    for n in res.get("kept_failed", []):
        p(f"   kept as failed (D-002 §6: kept, truncated at the crash): {n}")
    p("instance                       P ctrl  P bo1   d       band   d ctrl|A6 d bo1|A6")
    for i, e in res["effects"].items():
        p(f"{i:<30} {_fmt(e['p_bo0']):<7} {_fmt(e['p_bo1']):<7} {_fmt(e['d']):<7} {_fmt(e['band'], 2):<6} "
          f"{_fmt(e['d_bo0_vs_A6']):<9} {_fmt(e['d_bo1_vs_A6'])}")
    m = res["mechanism"]
    for k in ("M1", "M2", "M3", "M4", "M5"):
        bad = []
        if k in ("M1", "M2", "M3"):
            bad = [r for r in m[k]["rows"] if not r.get("ok")]
        elif k == "M4":
            bad = [r for r in m[k]["rows"] if r.get("applies") and not r.get("ok")]
        else:
            bad = [f"{kk}: {len(v)}" for kk, v in m[k].items() if isinstance(v, list) and v]
        noted = [f"seed {r['seed']}: {r['why']}" for r in m[k]["rows"] if r.get("ok") and r.get("why")] \
            if k == "M2" else []
        p(f"{k}: {'holds' if m[k]['ok'] else 'FAILS'}" + (f" — {bad[:4]}" if bad else "")
          + (f" (reported: {'; '.join(noted)})" if noted else ""))
    p(f"M4 gpu util lower on {m['M4']['util_lower']} of {len(m['M4']['util'])} (needed {m['M4']['util_needed']})")
    esc = [r for r in m["M6"]["rows"] if r["before_lead"]]
    p(f"M6 (reported): the control would have stepped down before its first lead on {len(esc)} guard runs "
      f"{[(r['instance'], r['seed']) for r in esc]}")
    o = res["outcome"]
    p(f"T: {'holds' if o['T']['ok'] else 'FAILS'} (chromaticindex d {_fmt(o['T']['d'])}, P {_fmt(o['T']['p_bo1'])})")
    p(f"G: {'holds' if o['G']['ok'] else 'FAILS'} (guard mean d {_fmt(o['G']['mean_d'])}; neos-960392 dP "
      f"{_fmt(o['G']['neos960']['dP'])} >= {_fmt(o['G']['neos960']['limit'])}; failing "
      f"{[r['instance'] for r in o['G']['rows'] if not r['ok']]}; noise {o['G']['noise']}; still beating A6 "
      f"{[(b['instance'], b['ok']) for b in o['G']['still_beats_A6']]})")
    p(f"ladder targets: {'hold' if o['ladder']['ok'] else 'FAIL'} (mean d {_fmt(o['ladder']['mean_d'])})")
    p(f"F: {'holds' if o['F']['ok'] else 'FAILS'} ({[(r['instance'], _fmt(r['dP'], 4)) for r in o['F']['rows']]})")
    for r in o["F"]["rows"]:
        p(f"   {r['instance']} seconds: " + ", ".join(f"{k} {_fmt(v, 2)}" for k, v in r["seconds"].items()))
    p(f"mean guard (11 non-floor): {'holds' if o['mean_guard']['ok'] else 'FAILS'} (mean d "
      f"{_fmt(o['mean_guard']['mean_d'])})")
    rep = res["reports"]
    for b, v in rep["d013"].items():
        p(f"D-013 {b}: weak set {v['weak_set']}; " + "; ".join(
            f"{arm}: improved {v[arm]['improved']}, P {_fmt(v[arm]['mean_P_B'])} -> {_fmt(v[arm]['mean_P_arm'])}, "
            f"mean d {_fmt(v[arm]['mean_d_weak'])}, worst {_fmt(v[arm]['worst_P'])}, all-mean d "
            f"{_fmt(v[arm]['mean_d_all'])}" for arm in (CTRL1, BO1)))
    p("relay latency (cuOpt send -> SCIP incumbent): " + ", ".join(
        f"{a} median {_fmt(v['median_s'], 2)} s (n {v['n']})" for a, v in rep["relay"].items()))
    p("HiGHS LP point arrival: " + ", ".join(f"{a} median {_fmt(v['median_s'], 2)} s (n {v['n']})"
                                             for a, v in rep["lp_arrival"].items()))
    for c in res["classification"]:
        p("classification: " + c)
    for n in res["notes"]:
        p("note: " + n)
    p("VERDICT: " + res["verdict"])


def print_stage1y(res: dict, out=sys.stdout) -> None:
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"Stage 1y (D-023, phase Y only) — {BOY} (active, no ladder) vs {CTRL1} (shadow control), same-day {A6F}, "
      f"set {res['set']}")
    hz = res["harness"]
    p(f"harness: A9 arms {hz['a9'] or [g['harness'] for g in hz['a9_groups']]}, A6F ext"
      + (f" (D-022's outcome given: {hz['expect']})" if hz["expect"] else " (not checked against D-022's outcome)")
      + ("" if hz["ok"] else " — FAILS: " + "; ".join(hz["problems"])))
    for n in res.get("kept_failed", []):
        p(f"   kept as failed (D-002 §6: kept, truncated at the crash): {n}")
    p("instance                       P ctrl  P boY   d       band   d ctrl|A6F d boY|A6F")
    for i, e in res["effects"].items():
        p(f"{i:<30} {_fmt(e['p_bo0']):<7} {_fmt(e['p_bo1']):<7} {_fmt(e['d']):<7} {_fmt(e['band'], 2):<6} "
          f"{_fmt(e['d_bo0_vs_A6']):<10} {_fmt(e['d_bo1_vs_A6'])}")
    m, d18 = res["mechanism"], res["mechanism_d018"]
    hold = lambda ok: "holds" if ok else "FAILS"  # noqa: E731
    bad1 = [(r["instance"], r["seed"], _fmt(r["exit_t"] - r["partner_t"] if _fin(r["exit_t"]) else math.nan, 2))
            for r in m["M1"]["rows"] if not r["ok"]]
    lat = [r["exit_t"] - r["partner_t"] for r in m["M1"]["rows"] if _fin(r["exit_t"])]
    p(f"M1 (exit within {m['M1']['limit_s']:g} s of the first partner incumbent): {hold(m['M1']['ok'])} — "
      f"{len(m['M1']['rows'])} runs, max latency {_fmt(max(lat) if lat else math.nan, 2)} s"
      + (f", failing {bad1}" if bad1 else "") + f"; n/a (no partner post) {len(m['M1']['n_a'])}"
      + (f" {[(r['instance'], r['seed']) for r in m['M1']['n_a']]}" if m["M1"]["n_a"] else ""))
    noted = [f"seed {r['seed']}: {r['why']}" for r in m["M2"]["rows"] if r.get("ok") and r.get("why")]
    p(f"M2 (chromaticindex cuOpt first incumbent <= {M2_X:g} x A6F's): {hold(m['M2']['ok'])} — "
      + ", ".join(f"s{r['seed']} {_fmt(r.get('t_bo1'), 2)} vs {_fmt(r.get('t_A6'), 2)} s"
                  + (f" ({r['why']})" if r.get("why") and not r.get("ok") else "") for r in m["M2"]["rows"])
      + (f" (reported: {'; '.join(noted)})" if noted else ""))
    p(f"M3 (leads summed over seeds >= {Y_M3_LEADS:g} x the control's): {hold(m['M3']['ok'])} — "
      + ", ".join(f"{r['instance']} {r['leads_boY']} vs {r['leads_ctrl']}"
                  + (f" (missing seeds {r['missing_seeds']})" if r["missing_seeds"] else "") for r in m["M3"]["rows"]))
    hi = [(r["instance"], r["seed"], _fmt(r["share"], 2), _fmt(r["wall_s"], 2)) for r in m["M4"]["rows"]
          if _fin(r["share"]) and r["share"] > Y_M4_SHARE]
    p(f"M4 (boY's pooled busy share in phase Y <= {Y_M4_SHARE:g}): {hold(m['M4']['ok'])} — "
      f"{_fmt(m['M4']['busy_s'], 2)} busy / {_fmt(m['M4']['wall_s'], 2)} s = {_fmt(m['M4']['busy_share'], 3)} over "
      f"{len(m['M4']['rows'])} runs; per run above {Y_M4_SHARE:g} (instance, seed, share, phase-Y s): {hi}"
      + (f"; no phase-Y record {[(r['instance'], r['seed']) for r in m['M4']['no_record']]}"
         if m["M4"]["no_record"] else ""))
    p(f"M5 (accounting within {M5_ACC:.0%}; cap hits per call <= the control's + {Y_M5_CAP:g}): {hold(m['M5']['ok'])}"
      f" — accounting breaches {len(m['M5']['accounting'])}, cap-hit breaches "
      f"{[(r['instance'], r['seed'], _fmt(r['boY'], 2), _fmt(r['ctrl'], 2)) for r in m['M5']['cap_hits']]}")
    b3 = [(r["instance"], r["seed"]) for r in d18["M3"]["rows"] if not r["ok"]]
    b5 = [f"{k}: {len(v)}" for k, v in d18["M5"].items() if isinstance(v, list) and v]
    lat18 = [(r["instance"], r["seed"]) for r in d18["M1"]["rows"] if not r["ok"]]
    p(f"D-018's own rules on these runs (reported, not gating): M1 at {M1_S:g} s {hold(d18['M1']['ok'])} {lat18}; "
      f"M3 per seed (leads, first lead, level-0 share) {hold(d18['M3']['ok'])} {b3}; M5 (every leg, cap hits +0) "
      f"{hold(d18['M5']['ok'])} {b5}")
    o = res["outcome"]
    p(f"T: {hold(o['T']['ok'])} (chromaticindex d {_fmt(o['T']['d'])}, P {_fmt(o['T']['p_bo1'])})")
    p(f"G: {hold(o['G']['ok'])} (guard mean d {_fmt(o['G']['mean_d'])}; neos-960392 dP "
      f"{_fmt(o['G']['neos960']['dP'])} >= {_fmt(o['G']['neos960']['limit'])}; failing "
      f"{[r['instance'] for r in o['G']['rows'] if not r['ok']]}; noise {o['G']['noise']}; still beating A6F "
      f"{[(b['instance'], b['ok']) for b in o['G']['still_beats_A6']]})")
    p(f"ladder targets (a gate in 1y): {'hold' if o['ladder']['ok'] else 'FAIL'} "
      f"(mean d {_fmt(o['ladder']['mean_d'])}; failing {[r['instance'] for r in o['ladder']['rows'] if not r['ok']]})")
    p(f"F: {hold(o['F']['ok'])} ({[(r['instance'], _fmt(r['dP'], 4)) for r in o['F']['rows']]})")
    for r in o["F"]["rows"]:
        p(f"   {r['instance']} seconds: " + ", ".join(f"{k} {_fmt(v, 2)}" for k, v in r["seconds"].items()))
    p(f"mean guard (11 non-floor): {hold(o['mean_guard']['ok'])} (mean d {_fmt(o['mean_guard']['mean_d'])})")
    rep = res["reports"]
    for b, v in rep["d013"].items():
        p(f"D-013 {b}: weak set {v['weak_set']}; " + "; ".join(
            f"{arm}: improved {v[arm]['improved']}, P {_fmt(v[arm]['mean_P_B'])} -> {_fmt(v[arm]['mean_P_arm'])}, "
            f"mean d {_fmt(v[arm]['mean_d_weak'])}, worst {_fmt(v[arm]['worst_P'])}, all-mean d "
            f"{_fmt(v[arm]['mean_d_all'])}" for arm in (CTRL1, BOY)))
    p("relay latency (cuOpt send -> SCIP incumbent): " + ", ".join(
        f"{a} median {_fmt(v['median_s'], 2)} s (n {v['n']})" for a, v in rep["relay"].items()))
    p("HiGHS LP point arrival: " + ", ".join(f"{a} median {_fmt(v['median_s'], 2)} s (n {v['n']})"
                                             for a, v in rep["lp_arrival"].items()))
    for c in res["classification"]:
        p("classification: " + c)
    for n in res["notes"]:
        p("note: " + n)
    p("VERDICT: " + res["verdict"])
    ex = res.get("exclude_reading")
    if ex:
        p(f"VERDICT with the failed runs excluded (the other reading, D-023): {ex['verdict']}")


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else ("inf" if math.isinf(f) and f > 0 else ("-inf" if math.isinf(f) else f))
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="itrea backoff-report", description="A9 GPU back-off: stage-0 / stage-1 rules "
                                 "of D-018 and stage 1y (D-023, phase Y only) applied to the run table and the run "
                                 "directories")
    ap.add_argument("--stage", choices=("0", "1", "1y"), required=True,
                    help="0 / 1: D-018's calibration and test; 1y: D-023 (A9@boY vs A9@bo1s + A6F, set backoff2)")
    ap.add_argument("--set", dest="set_", default=None, help="run set (default backoff0 / backoff1 / backoff2)")
    ap.add_argument("--stage0-set", default="backoff0", help="stage 1 / 1y: the stage-0 set (ca's class-(Y) fallback)")
    ap.add_argument("--runs", default=str(DATASETS / "runs.parquet"))
    ap.add_argument("--incumbents", default=str(DATASETS / "incumbents_checked.parquet"))
    ap.add_argument("--root", default=str(ROOT), help="root of the run_dir paths in the run table")
    ap.add_argument("--json", default=None, help="also write the result as JSON here")
    ap.add_argument("--failed", choices=("exclude", "keep", "both"), default=None,
                    help="stage 1: failed runs are excluded (default: the verdict needs every run completed) or kept "
                         "as D-002 §6 keeps crashes — truncated at the crash, as build-runs already scores them "
                         "(stated deviation of 2026-10-11: cuOpt's RMM limit hit on ca seed 1); stage 1y: default "
                         "both — the kept reading in full (D-023) and the excluded reading's verdict beside it")
    ap.add_argument("--harness", choices=Y_HARNESS, default=None,
                    help="stage 1y: D-022's outcome — the harness both A9 arms must carry (A6F: ext); without it the "
                         "harness found is reported and must still be one for the A9 arms")
    ap.add_argument("--steps-gate", action="store_true",
                    help="stage 0: let the lane-steps limb gate the sanity check (the rule as first written, before "
                         "the D-018 deviation of 2026-10-09)")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    if not Path(a.runs).exists():
        print(f"{a.runs} does not exist (run `itrea build-runs` first)", file=sys.stderr)
        return 2
    runs = pd.read_parquet(a.runs)
    inc = pd.read_parquet(a.incumbents) if Path(a.incumbents).exists() else None
    if a.harness is not None and a.stage != "1y":
        print("--harness is a stage-1y option (D-023 under D-022's outcome)", file=sys.stderr)
        return 2
    if a.stage == "0":
        res = stage0(runs, inc, a.set_ or "backoff0", Path(a.root), steps_gate=a.steps_gate)
        print_stage0(res)
    elif a.stage == "1":
        if a.failed == "both":
            print("--failed both is a stage-1y reading; stage 1 takes exclude or keep", file=sys.stderr)
            return 2
        kept = []
        if a.failed == "keep":
            sel = (runs["set"] == (a.set_ or "backoff1")) & (runs["status"] == "failed")
            kept = sorted(runs.loc[sel, "run_id"].astype(str))
            runs = runs.copy()
            runs.loc[sel, "status"] = "completed"
        res = stage1(runs, inc, a.set_ or "backoff1", Path(a.root), stage0_set=a.stage0_set)
        res["kept_failed"] = kept
        print_stage1(res)
    else:
        res = run_stage1y(runs, inc, a.set_ or "backoff2", Path(a.root), a.stage0_set, a.failed or "both", a.harness)
        print_stage1y(res)
    if a.json:
        Path(a.json).write_text(json.dumps(_jsonable(res), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
