"""Trajectory metrics: primal gap, primal integral P(T), primal-dual integral, times to target (vault D-002 §4,
`10-design/Trace format and metrics.md`).

Adapted from berkorbay/optopt@4dd48395, MIT (`src/optopt/analysis/metrics.py`): the gap, the integral on exact event
times (optopt's 2026-09-24 PDI correction — never round times before the lookup) and the time-to-target logic. What
changed for I-TREA II, and why:

- **Input** is a list of `(t, obj)` incumbent points — t in seconds since the shared start t0, obj in the user's sense
  — built by `itrea.analysis.build_runs` from CHECKED incumbents only, with the objective the exact checker
  recomputed. optopt read a trace dict with `events[*].primal`; here the trace format is `itrea.trace`.
- **Beating the reference** gives gap 0 and a `beats_ref` flag (D-002 v2+: the reference is frozen before the
  confirmatory runs, so a checked incumbent that beats it is a real, logged event). optopt instead scored an incumbent
  that beat a published optimum by more than 1e-4 as invalid (gap 1) because its incumbents were not all checked.
- **Degenerate gaps are flagged, not hidden.** Berthold's gap is 1 whenever the incumbent and the reference differ in
  sign, and 1 for every value other than 0 when the reference is 0, so P(T) silently turns into "time to the exact
  optimum". Every result carries the flags `ref_zero` and `sign_crossing`; D-002 analyses such instances on time to
  target only.
- **A sign crossing must persist to count** (review 2026-10-06, F3). The step flag SIGN_CROSSING marks an incumbent
  with v·r < 0 strictly (v = 0 against r ≠ 0 is Berthold's exact gap 1, not a degeneracy). A transient crossing —
  SCIP's trivial all-zero or a first heuristic solution of the wrong sign, replaced within a second — only marks an
  early, legitimately large gap, so the per-run flag `sign_crossing` (what makes an instance P-degenerate) requires
  the held incumbent to cross for at least SIGN_CROSSING_MIN_SHARE = 5 % of [0, T] (`sign_crossing_share`);
  `sign_crossing_any` keeps the old "any step crossed" reading as information.
- **MIPfeas-style integral** (D-002 §4 secondary metric, Bussieck & Dirkse 2026): gap 2 with no incumbent, otherwise
  |v − r| / max(|v|, |r|, 1) with no sign clamp (so it lies in [0, 2]), 0 when v equals or beats r; aggregated with
  the shifted geometric mean, s = 0.001 (`shifted_geomean` takes the shift as a REQUIRED keyword, so the
  conventional s = 1 cannot be used by accident).
- **GAMS's MIPfeas integral, exactly** (D-020; `gams_integral`, column `pt_gams`): a port of
  `calculate_primal_integral` from GAMS's published scoring script (`create_primalintegral.py`, 2026-09-15, in
  data/mipfeas/gams-2026-09-16/). It differs from `pt_mf` on purpose: γ = 1 for opposite signs, the ratio rounded to 6
  decimals, T = 600 s, every time capped at T. Their script assumes minimisation (it raises on an incumbent better
  than z*); here the difference is taken in the objective's sense, so a maximisation instance scores as its
  minimisation mirror (none of the 233 MIPfeas instances maximises: no MPS file has an OBJSENSE section). An
  incumbent that beats z* by their own rounded test gets γ = 0 and the flag `beats_ref` (theirs raises).
- **Censoring is explicit.** A time to target is a `Censored(t, event)` pair with t = T when the target was not reached
  by T (Type I censoring at a common T), never `inf`, so tables and the censored-data tests downstream stay typed.

Definitions (Berthold 2013, "Measuring the impact of primal heuristics"):

    γ(v, r) = 1                          no incumbent, or v·r < 0 (signs differ)
            = 0                          v = r (including both 0), or v beats r (flag beats_ref)
            = |v − r| / max(|v|, |r|)    otherwise (≤ 1)
    P(T)    = (1/T) ∫₀ᵀ γ(t) dt ∈ [0, 1],  γ(t) of the best incumbent held at t, every trajectory truncated at T.
"""
from __future__ import annotations

import math
from typing import Iterable, NamedTuple, Sequence

# flags attached to gaps and integrals
REF_ZERO = "ref_zero"            # reference is exactly 0: gap is 1 for every other value
SIGN_CROSSING = "sign_crossing"  # an incumbent with v·r < 0 strictly (gap pinned at 1 by the sign alone)
BEATS_REF = "beats_ref"          # a checked incumbent better than the reference (gap 0, logged)
NO_REF = "no_ref"                # no reference value: integral undefined (NaN)
PRE_T0 = "pre_t0"                # an incumbent time < 0 (clamped to 0) — should not happen under the clock rule

BEATS_TOL = 1e-9                 # relative margin below which "better than the reference" is equality (numerics)
TARGETS = (0.05, 0.01)           # D-002 §4: time to gap <= 5 % and <= 1 %
IMPROVE_TOL = 1e-12              # a new step needs a relative improvement above this (ties are not new incumbents)
SIGN_CROSSING_MIN_SHARE = 0.05   # a run is sign-crossing (P-degenerate) if it held a crossing incumbent >= 5 % of T
MIPFEAS_SHIFT = 0.001            # D-002 §4: shifted geometric mean of the MIPfeas-style integral, s = 0.001
MIPFEAS_NONE = 2.0               # MIPfeas-style gap with no incumbent
GAMS_T = 600.0                   # D-020: MIPfeas's time limit (GAMS's max_time)
GAMS_DIGITS = 6                  # GAMS rounds the relative gap to 6 decimals


class Censored(NamedTuple):
    t: float       # event time, or T when censored
    event: bool    # True = the target was reached at t <= T


Step = tuple[float, float]       # (t since t0, objective in the user's sense)


def _fl(v) -> float | None:
    """Float from a trace value: numbers, None, and the strings "inf"/"-inf" that itrea.trace writes for infinities."""
    if v is None:
        return None
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


# --------------------------------------------------------------------------------------------------------- gap
def gap(a: float | None, b: float | None) -> float:
    """Berthold's symmetric relative gap between two values (no sense, no beats-reference rule)."""
    if a is None or b is None:
        return 1.0
    if a == b:
        return 0.0
    if a * b < 0:
        return 1.0
    return min(1.0, abs(a - b) / max(abs(a), abs(b)))


def primal_gap(v: float | None, ref: float | None, sense: int = 1) -> tuple[float, frozenset[str]]:
    """Gap of incumbent value v against the reference, with flags. sense: +1 minimise, -1 maximise.
    Returns (gap, flags); gap is NaN only when there is an incumbent but no reference (flag no_ref)."""
    if ref is None:
        return (1.0 if v is None else math.nan), frozenset({NO_REF})
    flags = {REF_ZERO} if ref == 0.0 else set()
    if v is None:
        return 1.0, frozenset(flags)
    d = sense * (v - ref)
    if d < 0:
        if -d > BEATS_TOL * max(1.0, abs(ref)):
            flags.add(BEATS_REF)
        return 0.0, frozenset(flags)
    if v * ref < 0:                     # strictly opposite signs (v = 0 against r != 0 is an exact gap of 1)
        flags.add(SIGN_CROSSING)
    return gap(v, ref), frozenset(flags)


def mipfeas_gap(v: float | None, ref: float | None, sense: int = 1) -> float:
    """MIPfeas-style gap (D-002 §4): 2 with no incumbent; 0 when v equals or beats the reference (same rule as
    `primal_gap`); otherwise |v − r| / max(|v|, |r|, 1), in [0, 2], with no sign-crossing clamp. NaN when there is an
    incumbent but no reference."""
    if v is None:
        return MIPFEAS_NONE
    if ref is None:
        return math.nan
    if sense * (v - ref) <= 0:
        return 0.0
    return abs(v - ref) / max(abs(v), abs(ref), 1.0)


# ----------------------------------------------------------------------------------------------------- steps
def best_steps(points: Iterable[Sequence], sense: int = 1) -> list[Step]:
    """Monotone best-so-far steps from incumbent points (t, obj): only strict improvements (relative > 1e-12) make a
    step. Points with obj None/NaN are ignored; times < 0 are clamped to 0 (see `clamped`)."""
    pts = []
    for p in points:
        t, v = _fl(p[0]), _fl(p[1])
        if t is None or v is None or not math.isfinite(v):
            continue
        pts.append((max(0.0, t), v))
    pts.sort(key=lambda x: (x[0], sense * x[1]))
    steps: list[Step] = []
    best = None
    for t, v in pts:
        if best is None or sense * v < sense * best - IMPROVE_TOL * max(1.0, abs(best)):
            best = v
            steps.append((t, v))
    return steps


def clamped(points: Iterable[Sequence]) -> bool:
    """True if any point has t < 0 (it was clamped to 0 by best_steps)."""
    return any((_fl(p[0]) or 0.0) < 0 for p in points)


def merge_steps(*step_lists: Sequence[Step], sense: int = 1) -> list[Step]:
    """Pointwise best of several trajectories (the portfolio: best held by any process at every moment)."""
    return best_steps([s for lst in step_lists for s in lst], sense)


def truncate(steps: Sequence[Step], T: float) -> list[Step]:
    return [(t, v) for t, v in steps if t <= T]


def value_at(steps: Sequence[Step], t: float) -> float | None:
    v = None
    for s, x in steps:
        if s <= t:
            v = x
        else:
            break
    return v


def dual_steps(points: Iterable[Sequence], sense: int = 1) -> list[Step]:
    """Monotone dual bound steps (t, bound): for minimisation the bound rises, for maximisation it falls. Infinite
    bounds ("-inf" in the trace) are skipped (no bound yet)."""
    pts = []
    for p in points:
        t, d = _fl(p[0]), _fl(p[1])
        if t is None or d is None or not math.isfinite(d):
            continue
        pts.append((max(0.0, t), d))
    pts.sort(key=lambda x: x[0])
    steps, best = [], None
    for t, d in pts:
        if best is None or sense * d > sense * best:
            best = d
            steps.append((t, d))
    return steps


# ------------------------------------------------------------------------------------------------- integrals
class Integral(NamedTuple):
    value: float                  # normalised integral in [0, 1] (NaN if no reference and an incumbent exists)
    flags: frozenset[str]
    beats_ref_t: float | None     # first time a checked incumbent beat the reference


def primal_integral(steps: Sequence[Step], ref: float | None, sense: int, T: float) -> Integral:
    """P(T)/T ∈ [0, 1] on exact event times; steps after T are ignored; gap 1 before the first incumbent."""
    if T <= 0:
        raise ValueError("T must be positive")
    st = [(t, v) for t, v in steps if t < T]
    flags: set[str] = set()
    if ref is None:
        return Integral(1.0 if not st else math.nan, frozenset({NO_REF}), None)
    if ref == 0.0:
        flags.add(REF_ZERO)
    area, t_prev, g_prev, beats_t = 0.0, 0.0, 1.0, None
    for t, v in st:
        area += g_prev * (t - t_prev)
        g, f = primal_gap(v, ref, sense)
        flags |= f
        if BEATS_REF in f and beats_t is None:
            beats_t = t
        t_prev, g_prev = t, g
    area += g_prev * (T - t_prev)
    return Integral(area / T, frozenset(flags), beats_t)


def sign_crossing_share(steps: Sequence[Step], ref: float | None, sense: int, T: float) -> float:
    """Share of [0, T] during which the HELD incumbent crosses the reference's sign (its gap carries SIGN_CROSSING).
    Before the first incumbent nothing crosses. NaN without a reference."""
    if T <= 0:
        raise ValueError("T must be positive")
    if ref is None:
        return math.nan
    st = [(t, v) for t, v in steps if t < T]
    span, t_prev, crossing = 0.0, 0.0, False
    for t, v in st:
        if crossing:
            span += t - t_prev
        crossing = SIGN_CROSSING in primal_gap(v, ref, sense)[1]
        t_prev = t
    if crossing:
        span += T - t_prev
    return span / T


def mipfeas_integral(steps: Sequence[Step], ref: float | None, sense: int, T: float) -> float:
    """MIPfeas-style integral in [0, 2]: `mipfeas_gap` of the held incumbent integrated over [0, T] on exact event
    times and divided by T; gap 2 before the first incumbent; steps at or after T are ignored. Without a reference:
    2 if there is no incumbent before T, else NaN."""
    if T <= 0:
        raise ValueError("T must be positive")
    st = [(t, v) for t, v in steps if t < T]
    if ref is None:
        return MIPFEAS_NONE if not st else math.nan
    area, t_prev, g_prev = 0.0, 0.0, MIPFEAS_NONE
    for t, v in st:
        area += g_prev * (t - t_prev)
        t_prev, g_prev = t, mipfeas_gap(v, ref, sense)
    area += g_prev * (T - t_prev)
    return area / T


class GamsIntegral(NamedTuple):
    value: float                  # GAMS's MIPfeas integral in [0, 2] (NaN with an incumbent but no reference)
    beats_ref: bool               # an incumbent beat the reference by GAMS's rounded test (theirs raises)
    beats_ref_t: float | None     # its first time (capped at T)


def gams_gap(v: float, ref: float, sense: int = 1) -> tuple[float, bool]:
    """γ of one incumbent in GAMS's `calculate_primal_integral` (D-020), in their order of tests: (0, True) if the
    rounded relative difference is negative (v beats z*: theirs raises), 1 for opposite signs (v·z* < 0), else
    round((v − z*)/max(|v|, |z*|, 1), 6). The difference is taken in the objective's sense (`sense` −1: maximise);
    Python's round on Python floats, as in their loop (pandas' iterrows hands them Python floats)."""
    v, ref = float(v), float(ref)
    rel = round(sense * (v - ref) / max(abs(v), abs(ref), 1), GAMS_DIGITS)
    if rel < 0:
        return 0.0, True
    if v * ref < 0:
        return 1.0, False
    return rel, False


def gams_integral(steps: Sequence[Step], ref: float | None, sense: int = 1, T: float = GAMS_T) -> GamsIntegral:
    """GAMS's MIPfeas primal integral (D-020) on best-so-far steps (t, obj in the user's sense): γ = 2 before the first
    incumbent, `gams_gap` after it; every time capped at T (a step at or after T adds nothing), the last γ held to T;
    the area divided by T. Their trace rows that do not improve the incumbent change nothing in their sum, so the
    best-so-far steps give the same value (tested on every GB10 26.08 trace of their archive). Without a reference:
    2 if there is no incumbent before T, else NaN. Times below 0 are clamped to 0."""
    if T <= 0:
        raise ValueError("T must be positive")
    if ref is None:
        return GamsIntegral(MIPFEAS_NONE if not any(t < T for t, _ in steps) else math.nan, False, None)
    area, t_last, g, beats_t = 0.0, 0.0, MIPFEAS_NONE, None
    for t, v in steps:
        t = min(max(0.0, float(t)), T)
        if t < t_last:
            raise ValueError(f"steps not in time order: {t} after {t_last}")
        area += (t - t_last) * g
        t_last = t
        g, beats = gams_gap(v, ref, sense)
        if beats and beats_t is None:
            beats_t = t
    area += (T - t_last) * g
    return GamsIntegral(area / T, beats_t is not None, beats_t)


def primal_dual_integral(primal: Sequence[Step], dual: Sequence[Step], T: float) -> float:
    """Normalised primal-dual integral in [0, 1]: Berthold's gap between the held incumbent and the held dual bound,
    integrated over [0, T]. Primal and dual changes at the same instant are applied together (optopt fix)."""
    changes: dict[float, dict] = {}
    for t, v in primal:
        if t < T:
            changes.setdefault(t, {})["p"] = v
    for t, v in dual:
        if t < T:
            changes.setdefault(t, {})["d"] = v
    p = d = None
    area, t_prev = 0.0, 0.0
    for t in sorted(changes):
        area += gap(p, d) * (t - t_prev)
        c = changes[t]
        p, d = c.get("p", p), c.get("d", d)
        t_prev = t
    area += gap(p, d) * (T - t_prev)
    return area / T


# ------------------------------------------------------------------------------------------ times to target
def time_to_first(steps: Sequence[Step], T: float) -> Censored:
    """Time of the first (checked) incumbent, censored at T."""
    for t, _ in steps:
        if t <= T:
            return Censored(t, True)
        break
    return Censored(float(T), False)


def time_to_gap(steps: Sequence[Step], ref: float | None, sense: int, T: float, tol: float) -> Censored:
    """First time the held incumbent's gap is <= tol, censored at T. No reference: censored."""
    if ref is None:
        return Censored(float(T), False)
    for t, v in steps:
        if t > T:
            break
        if primal_gap(v, ref, sense)[0] <= tol:
            return Censored(t, True)
    return Censored(float(T), False)


def gap_at(steps: Sequence[Step], ref: float | None, sense: int, t: float) -> float:
    return primal_gap(value_at(steps, t), ref, sense)[0]


# --------------------------------------------------------------------------------------------------- summary
def summarize(steps: Sequence[Step], ref: float | None, sense: int, T: float,
              dual: Sequence[Step] | None = None, prefix: str = "") -> dict:
    """Every per-trajectory metric of D-002 §4 in one flat dict (keys prefixed by `prefix`)."""
    P = primal_integral(steps, ref, sense, T)
    first = time_to_first(steps, T)
    held = truncate(steps, T)
    best = value_at(held, T)
    share = sign_crossing_share(steps, ref, sense, T)
    out = {
        "pt": P.value,
        "ref_zero": REF_ZERO in P.flags,
        "sign_crossing": bool(share >= SIGN_CROSSING_MIN_SHARE),     # persistent crossing (False when NaN)
        "sign_crossing_any": SIGN_CROSSING in P.flags,               # any held incumbent crossed (information)
        "sign_crossing_share": share,
        "beats_ref": BEATS_REF in P.flags,
        "beats_ref_t": math.nan if P.beats_ref_t is None else P.beats_ref_t,
        "pt_mf": mipfeas_integral(steps, ref, sense, T),
        "pt_gams": gams_integral(steps, ref, sense, T).value,            # D-020, with T = the run's budget
        "t_first": first.t, "t_first_event": first.event,
        "final_gap": gap_at(held, ref, sense, T),
        "best_obj": math.nan if best is None else best,
        "n_steps": len(held),
    }
    for tol in TARGETS:
        c = time_to_gap(steps, ref, sense, T, tol)
        tag = f"t_gap{int(round(tol * 100))}"
        out[tag], out[tag + "_event"] = c.t, c.event
    if dual is not None:
        out["pdi"] = primal_dual_integral(steps, dual, T)
    return {prefix + k: v for k, v in out.items()}


def shifted_geomean(xs: Iterable[float], *, shift: float) -> float:
    """exp(mean(log(x + s))) − s over the non-missing values (NaN if none). The shift has no default on purpose: the
    MIPfeas-style integral is registered with s = MIPFEAS_SHIFT = 0.001 (D-002 §4), not the conventional 1."""
    if not shift > 0:
        raise ValueError("shift must be positive")
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return math.nan
    return math.exp(sum(math.log(x + shift) for x in xs) / len(xs)) - shift
