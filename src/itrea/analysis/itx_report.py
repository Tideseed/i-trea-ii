"""`itrea itx-report`: D-021's (ITX-1) analysis of the v1 external runs — per instance and the aggregates of R17–R22,
D-019's literal readings beside, the published rows re-scored on exactly our completed subset (R36) — and M6's lane
reading (`--m6`). The rules were fixed before any run (D-021, Berk's "Accept all", 2026-10-11); this module applies
them and decides nothing else.

Inputs: `datasets/runs.parquet` and `datasets/incumbents_checked.parquet` (`itrea build-runs`), the committed instance
sets `datasets/mipfeas_strata.csv` (`itrea.analysis.itx_strata`) and GAMS's archive (published rows). Arms (labels as
the v1 job writers make them, `make-jobs itx-v1 | itx-a0`): A9 = `A9@v1c<tag>`, A6F = `A6F@v1c<tag>` (seed 2000; the
panel's seed 2001 too), A0 = `A0@v1`, live A2F = `A2F@v1c<tag>`; T = 600 s.

Runs: per (instance, arm, seed) the one run at budget T, status completed or failed (an arm's failure is its outcome,
R29: it counts against that arm), not void and not in a voided block, not synthetic, carrying the registered settings
(R33, prereg-v2; `registered_values` / `registered_fault`: the `itx1` harness as recorded in run.json — for an arm
without cuOpt only its SCIP-side part, `pair.hashed_harness` — and, for A9, the base configuration hash of
`configs/frozen-v2.yaml`, the C1 freeze registered with the tag; `--frozen` names another file for a development
reading; while no registered file exists no A9 run is taken), intact (`mipfeas.integrity_faults`: no incumbent file
missing, none unchecked, stored = traced, none before t0). Two or more such runs: refused, never one picked. Every
run's trajectory is its checked portfolio (`gams_trc.run_steps`). The full run hash (which also folds in the lane's
footprint) is not re-derived here.

Per instance (`instances`), every arm, both clocks (call = since t0, launch = + the run's `startup_offset_s`):
`pt_gams` at T = 600 (GAMS's integral, `metrics.gams_integral`) and at 300, Berthold's P(300), the time to the first
incumbent, the run's start-up (`<arm>_startup_s` = launch − call clock); the computed race (R8: at every moment the
better of A0's and A6F's checked trajectories, same instance and seed — `metrics.merge_steps`; on the launch clock each
run with its own offset); the set flags (decide, H_cu, H_cu_p05, W_SCIP, W_only, E, e_sample, panel / t5, exposed …).

Aggregates (each names its clock, set, tier completion and seed count; `decisions` per clock):
- **Completeness:** A9 / A6F pairs present of expected on the decision set (`full_complete`), on the census +
  E-sample basis — H_cu-unseen, W-only and the E-sample (`census_complete`; T1 + T2, never cut, R32) — and on T3's
  decision instances (`t3_complete`). They decide how the target and the minimum are read.
- **Target (R19):** the ratio of mean `pt_gams` A9 / A6F on H_cu-unseen (after M3's list, 64) with a 10,000-resample
  paired percentile bootstrap (resampling instances, seed `stats.BOOT_SEED`): met iff the point estimate <= 0.75;
  "better shown" iff the interval's upper end (the two-sided 95 % interval's, i.e. a one-sided 97.5 % bound; the
  one-sided 95 % bound is reported beside) < 1; improved / worse counts (d_ε >= +0.10 / <= −0.10, R20) with a two-sided
  sign test; the worst case. Read only on the complete set: with a pair missing, `met` / `better_shown` are None and
  the reading is "incomplete (n of N)" (`met_on_available` keeps the figure on the pairs present). Beside it (R3): the
  same figures on H_cu at P >= 0.5 (`target_p05`, 30 decision instances).
- **Minimum (R19):** (a) the ε-floored guard d_ε = (P_ctrl − P_arm) / max(P_ctrl, P_arm, ε), ε = 60 γ-seconds / T =
  0.1 at T = 600 (R18), mean over the decision set (215 = unseen minus the list) A9 vs A6F with its bootstrap interval
  and Wilcoxon p: passes iff the mean >= −0.05; harm "shown" only if the whole interval lies below −0.05; (b) solutions
  found (`pt_gams` < 2) by A9 >= A6F's on the decision set (on the 233 beside). Minimum = (a) and (b), read on a basis
  (`minimum_basis`): "full" when every decision pair is present (the plain mean and the counts); "stratified" when T3
  is incomplete but the census and the E-sample are complete — (a) on the stratified guard (harm shown iff its
  interval's upper end < −0.05), (b) on the stratified count of solutions found A9 − A6F >= 0 (R31, Berk's "Accept
  all"); "incomplete" otherwise (None, never "met"). The plain figures on the pairs present are kept as
  `passes_on_available` / `harm_shown_on_available`.
- **Stratified estimates (R31)** of the guard and solutions found on the decision set from the census of H_cu-unseen
  and W-only and the E-sample (each E quartile weighted N_q / 10): reported always, decisive while T3 is incomplete.
- **Without the exposed instances (R33):** every decisive aggregate above (completeness, target, guard, solutions,
  stratified estimates, minimum and its R10 reading) again on the decision set minus the instances whose A6F 26.08
  outcome was seen before prereg-v2 (`exposed` in the strata file: the calibration sample and M3's probe; 31 of the
  215), under `without_exposed`.
- **Secondary (R21, Holm over the two):** A9 vs the computed race on H_cu-unseen; A9 vs A0 on W_SCIP-unseen (62):
  mean d_ε, bootstrap interval, Wilcoxon p, Holm-adjusted p. `--bca`: BCa intervals beside every percentile interval
  of the target ratio and of each mean d_ε (target, guard, secondaries; R21's sensitivity).
- **R8, live A2F vs the computed race** on the panel (T5, 20): ratio of means A2F / race and mean d_ε (negative: the
  computed race flatters the live one).
- **Reported, not decisive (R17):** the shifted geometric means of `pt_gams` at shift 0.001 and 1/30, both clocks, per
  arm, over the instances run (the 233 minus the list) and the decision set; the panel's 90th percentile of |d_ε|
  between its two seeds per arm (R20's per-instance noise scale).
- **D-019's literal readings** (beside): A9's geo-mean (0.001) <= A6F's over the instances run; the hard part by OUR
  A6F (`pt_gams` >= 0.1 or no solution): mean ratio <= 0.75; geo-mean ratio <= 0.9; stretch A9's geo-mean <= 0.0274.
- **Published rows (R36)** re-scored by the port from GAMS's own traces on exactly our completed subset (instances with
  both A9 and A6F at seed 2000), with footprint, clock, version and machine named; descriptive only (no "beats").
Claims are written "met" only when met on both clocks (R10); otherwise "met on the call clock only", with the
start-up cost beside it (each arm's median / p90 start-up on the set, `startup`) and the launch clock's figure; or
"not met", or "incomplete (n of N)".

M6 (`--m6`): SCIP's nodes/s alone (A0@m6a) vs beside 3 busy lanes (A0@m6b) per instance — more than 3 % slower on
the mean ratio -> fewer lanes — and the hottest ACPI zone over the heat phase's runs (set itx-m6): >= 85 °C -> 3 lanes.

    itrea itx-report [--cuopt 26.10] [--runs datasets/runs.parquet] [--inc datasets/incumbents_checked.parquet]
                     [--strata datasets/mipfeas_strata.csv] [--gams-dir …] [--frozen configs/frozen-v2.yaml]
                     [--json F] [--md F] [--bca] [--m6]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.analysis import metrics as M
from itrea.analysis import stats as S
from itrea.paths import DATASETS

T_EXT = 600.0
EPS_GAMMA_S = 60.0                       # R18: ε = 60 γ-seconds / T
IMPROVED = 0.10                          # R20
GUARD = -0.05                            # R19 (a)
TARGET_RATIO = 0.75                      # R19 target (D-019's −25 %)
STRETCH_SGM = 0.0274                     # D-019's stretch (SCIPConc-11, published)
LIT_GEO_RATIO = 0.90                     # D-019's geo-mean −10 %
SHIFTS = (0.001, 1.0 / 30.0)             # R17
M6_SLOWER, M6_HOT_C = 0.03, 85.0         # D-021 M6 (D-002 §7's 3 %; the governor's 85 °C line)
CLOCKS = ("call", "launch")
VALID_STATUS = ("completed", "failed")


def eps_for(T: float) -> float:
    return EPS_GAMMA_S / float(T)


def d_eps(p_ctrl, p_arm, eps: float) -> np.ndarray:
    """R18's ε-floored effect, positive = the arm better: (P_ctrl − P_arm) / max(P_ctrl, P_arm, ε)."""
    c, a = np.asarray(p_ctrl, dtype=float), np.asarray(p_arm, dtype=float)
    return (c - a) / np.maximum(np.maximum(c, a), eps)


def labels(cuopt: str = "26.10") -> dict[str, str]:
    from itrea.experiments.make_jobs import ITX_A0_VARIANT, cuopt_tag
    v = f"v1c{cuopt_tag(cuopt)}"
    return dict(A9=f"A9@{v}", A6F=f"A6F@{v}", A2F=f"A2F@{v}", A0=f"A0@{ITX_A0_VARIANT}")


# ------------------------------------------------------------------------------------------------------ runs
def registered_values(frozen: str | Path | None = None) -> dict:
    """What every v1 run must carry (R33, prereg-v2): the `itx1` harness as this code resolves it and, for A9, the
    base configuration hash of the file registered with the tag (`configs/frozen-v2.yaml`, the C1 freeze; another file
    only by `--frozen` for a development reading). `A9_base_hash` is None while the file does not exist: then no A9 run
    is taken."""
    from itrea import paths
    from itrea.config import EAConfig
    from itrea.experiments import pair
    from itrea.experiments.pool import FROZEN_V2_CONFIG
    name = str(frozen or FROZEN_V2_CONFIG)
    fp = paths.resolve(name)
    return dict(harness=pair.resolve_harness("itx1"), frozen=name,
                A9_base_hash=EAConfig.from_yaml(fp).hash() if fp.exists() else None)


def registered_fault(row, arm: str, reg: dict) -> str:
    """'' if a run carries the registered settings, else why not: its recorded harness (`harness`, JSON) against the
    registered itx1 — for an arm without cuOpt only the part that acts on it (`pair.hashed_harness`; R33: A0 stands
    if its SCIP-side harness is unchanged at the tag) — and, for A9, its `base_hash` against the registered
    configuration's. The full run hash (it also folds in the lane's footprint) is not re-derived here."""
    from itrea.experiments import pair
    spec = pair.arm_spec({"arm": arm})
    raw = str(row.get("harness") or "")
    try:
        got = json.loads(raw) if raw else None
    except ValueError:
        got = None
    if not isinstance(got, dict):
        return "no harness recorded (v1 runs carry itx1)"
    if pair.hashed_harness(spec, got) != pair.hashed_harness(spec, reg["harness"]):
        return f"harness {json.dumps(got, sort_keys=True)} is not the registered itx1"
    if arm == "A9":
        if reg.get("A9_base_hash") is None:
            return f"A9's registered configuration {reg.get('frozen')} does not exist"
        bh = str(row.get("base_hash") or "")
        if bh != reg["A9_base_hash"]:
            return f"config {bh or '(none)'} is not the registered {reg['A9_base_hash']} ({reg.get('frozen')})"
    return ""


def select(runs: pd.DataFrame, label: str, seed: int, budget: float = T_EXT,
           registered: dict | None = None) -> tuple[dict, dict]:
    """The one valid run per instance of an arm label at a seed (module docstring): ({instance: row}, {instance: [why
    the other runs were not taken]}). With `registered` (`registered_values`; the CLI always passes it) a run that does
    not carry the registered settings (`registered_fault`) is not taken either."""
    from itrea.analysis.mipfeas import integrity_faults
    r = runs[(runs["arm"].astype(str) == label) & (runs["seed"].astype(int) == int(seed)) &
             np.isclose(runs["budget"].astype(float), float(budget))]
    if "synthetic" in r.columns:
        r = r[~r["synthetic"].astype(bool)]
    chosen: dict = {}
    why: dict = {}
    valid: dict = {}
    for _, row in r.sort_values("run_id", kind="mergesort").iterrows():
        n = str(row["instance"])
        if str(row["status"]) not in VALID_STATUS or bool(row.get("block_void", False)):
            why.setdefault(n, []).append(f"{row['run_id']}: status {row['status']}"
                                         + (", block void" if bool(row.get("block_void", False)) else ""))
            continue
        if registered is not None:
            rf = registered_fault(row, label.split("@")[0], registered)
            if rf:
                why.setdefault(n, []).append(f"{row['run_id']}: {rf}")
                continue
        f = integrity_faults(row)
        if f:
            why.setdefault(n, []).append(f"{row['run_id']}: not intact ({'; '.join(f)})")
            continue
        valid.setdefault(n, []).append(row)
    for n, rows in valid.items():
        if len(rows) == 1:
            chosen[n] = rows[0]
        else:
            why.setdefault(n, []).insert(0, f"{len(rows)} valid runs: refused, never one picked")
    return chosen, why


def steps_by_run(inc: pd.DataFrame, run_ids, senses: dict[str, int]) -> dict[str, list]:
    """The checked portfolio's best-so-far steps per run (as build_runs builds them)."""
    out: dict[str, list] = {}
    want = set(run_ids)
    g = inc[inc["ok"].astype(bool) & inc["run_id"].isin(want)]
    for rid, gr in g.groupby("run_id", sort=False):
        s = senses.get(rid, 1)
        roles = [M.best_steps(zip(x["t"], x["obj_recomputed"]), s) for _, x in gr.groupby("role", sort=True)]
        out[rid] = M.merge_steps(*roles, sense=s)
    for rid in want:
        out.setdefault(rid, [])
    return out


def _shift(steps, off: float) -> list:
    return [(t + off, v) for t, v in steps]


def run_metrics(steps, z: float, sense: int, T: float, off: float) -> dict:
    """pt_gams at T and 300, Berthold P(300), t_first, on the call and launch clocks."""
    out = {}
    for clock, st in (("call", steps), ("launch", _shift(steps, off) if math.isfinite(off) else None)):
        if st is None:
            out.update({f"{k}_{clock}": math.nan for k in ("pt_gams", "pt_gams300", "P300", "t_first")})
            continue
        out[f"pt_gams_{clock}"] = M.gams_integral(st, z, sense, T).value
        out[f"pt_gams300_{clock}"] = M.gams_integral(st, z, sense, 300.0).value if T >= 300 else math.nan
        out[f"P300_{clock}"] = M.primal_integral(st, z, sense, 300.0).value if T >= 300 else math.nan
        tf = M.time_to_first(st, T)
        out[f"t_first_{clock}"] = tf.t if tf.event else math.nan
    return out


# ------------------------------------------------------------------------------------------------- statistics
def ratio_of_means(a, b, n_boot: int = S.N_BOOT, seed: int = S.BOOT_SEED, bca: bool = False) -> dict:
    """mean(a) / mean(b) on paired values with a percentile bootstrap over the pairs (same resamples as
    `stats.bootstrap_mean`); `hi` = the two-sided 95 % interval's upper end, `hi95` = the one-sided 95 % bound."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    out = dict(n=int(a.size), mean_arm=float(a.mean()) if a.size else math.nan,
               mean_ctrl=float(b.mean()) if b.size else math.nan, n_boot=n_boot, seed=seed)
    out["ratio"] = out["mean_arm"] / out["mean_ctrl"] if a.size and out["mean_ctrl"] > 0 else math.nan
    if a.size == 0 or not math.isfinite(out["ratio"]):
        out.update(lo=math.nan, hi=math.nan, hi95=math.nan)
        return out
    rng = np.random.default_rng(seed)
    rs = np.empty(n_boot)
    for s in range(0, n_boot, 1000):
        e = min(n_boot, s + 1000)
        idx = rng.integers(0, a.size, size=(e - s, a.size))
        mb = b[idx].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs[s:e] = np.where(mb > 0, a[idx].mean(axis=1) / mb, np.inf)
    lo, hi, hi95 = np.percentile(rs, [2.5, 97.5, 95.0])
    out.update(lo=float(lo), hi=float(hi), hi95=float(hi95))
    if bca and a.size > 2:
        from scipy import stats as st
        try:
            r = st.bootstrap((a, b), lambda x, y, axis=-1: x.mean(axis=axis) / y.mean(axis=axis), paired=True,
                             vectorized=True, n_resamples=n_boot, method="BCa", random_state=seed)
            out.update(bca_lo=float(r.confidence_interval.low), bca_hi=float(r.confidence_interval.high))
        except Exception as e:  # noqa: BLE001 — a sensitivity only
            out["bca_error"] = repr(e)[:200]
    return out


def sign_test(improved: int, worse: int) -> float:
    from scipy import stats as st
    n = improved + worse
    return float(st.binomtest(improved, n, 0.5).pvalue) if n else 1.0


def bca_mean(d, n_boot: int = S.N_BOOT, seed: int = S.BOOT_SEED) -> dict:
    """R21's sensitivity for a mean d_ε: the BCa 95 % interval (`scipy.stats.bootstrap`, the resample count and seed
    of the percentile interval). Fewer than 3 values: none; all values equal: the degenerate interval [d, d]."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return dict(bca_lo=math.nan, bca_hi=math.nan)
    if np.ptp(d) == 0:
        return dict(bca_lo=float(d[0]), bca_hi=float(d[0]))
    import warnings
    from scipy import stats as st
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = st.bootstrap((d,), np.mean, n_resamples=n_boot, method="BCa", random_state=seed)
        return dict(bca_lo=float(r.confidence_interval.low), bca_hi=float(r.confidence_interval.high))
    except Exception as e:  # noqa: BLE001 — a sensitivity only
        return dict(bca_lo=math.nan, bca_hi=math.nan, bca_error=repr(e)[:200])


def effect_summary(d, eps: float, names=None, bca: bool = False) -> dict:
    """mean d_ε with its bootstrap interval (with `bca`, also the BCa interval: R21's sensitivity), Wilcoxon, improved
    / worse (±0.10) with the sign test, the worst case."""
    d = np.asarray(d, dtype=float)
    ok = np.isfinite(d)
    b = S.bootstrap_mean(d[ok])
    w = S.wilcoxon_d(d[ok])
    imp, wor = int((d[ok] >= IMPROVED).sum()), int((d[ok] <= -IMPROVED).sum())
    out = dict(n=int(ok.sum()), eps=eps, mean_d=b["mean"], lo=b["lo"], hi=b["hi"], wilcoxon_p=w["p"],
               improved=imp, worse=wor, sign_p=sign_test(imp, wor))
    if bca:
        out.update(bca_mean(d[ok]))
    if ok.any():
        k = int(np.nanargmin(np.where(ok, d, np.inf)))
        out.update(worst_d=float(d[k]), worst=(None if names is None else str(list(names)[k])))
    return out


def shifted_geomeans(p) -> dict:
    p = [x for x in p if x is not None and math.isfinite(x)]
    return {f"shift_{s:.6g}": M.shifted_geomean(p, shift=s) if p else math.nan for s in SHIFTS}


def stratified(df: pd.DataFrame, value: str, strata: pd.DataFrame, exclude=()) -> dict:
    """R31's stratified estimate over the decision set of a per-instance value (`value` column of `df`, indexed by
    instance): census of H_cu-unseen and W-only, the E-sample per quartile weighted N_q / n_q; with a stratified
    bootstrap (resampling within each stratum). `exclude`: names left out of the decision set, its strata and their
    sizes (R33's reading without the exposed instances). Returns {estimate, lo, hi, complete, n_by_stratum}."""
    st = strata.set_index("name")
    dec = st[st["decide"] & ~st.index.isin(set(exclude))]
    groups: dict[str, tuple[list[str], int]] = {
        "H_cu": (list(dec.index[dec["H_cu"]]), int(dec["H_cu"].sum())),
        "W_only": (list(dec.index[dec["W_only"]]), int(dec["W_only"].sum()))}
    for q in (1, 2, 3, 4):
        inq = dec[dec["E"] & (dec["e_quartile"] == q)]
        groups[f"E_q{q}"] = (list(inq.index[inq["e_sample"]]), int(len(inq)))
    total = sum(N for _, N in groups.values())
    vals: dict[str, np.ndarray] = {}
    complete = True
    for g, (names, N) in groups.items():
        v = np.array([float(df.loc[n, value]) if n in df.index else math.nan for n in names], dtype=float)
        complete &= bool(np.isfinite(v).all()) and len(v) > 0
        vals[g] = v[np.isfinite(v)]
    if any(v.size == 0 for v in vals.values()):
        return dict(estimate=math.nan, lo=math.nan, hi=math.nan, complete=False,
                    n_by_stratum={g: int(v.size) for g, v in vals.items()})
    w = {g: groups[g][1] / total for g in groups}
    est = float(sum(w[g] * vals[g].mean() for g in groups))
    rng = np.random.default_rng(S.BOOT_SEED)
    boots = np.zeros(S.N_BOOT)
    for g in groups:
        v = vals[g]
        idx = rng.integers(0, v.size, size=(S.N_BOOT, v.size))
        boots += w[g] * v[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(estimate=est, lo=float(lo), hi=float(hi), complete=complete, total=total,
                n_by_stratum={g: int(v.size) for g, v in vals.items()},
                N_by_stratum={g: groups[g][1] for g in groups})


# ---------------------------------------------------------------------------------------------------- report
PER_INSTANCE_FLAGS = ("decide", "H_cu", "H_cu_p05", "W_SCIP", "W_only", "E", "e_sample", "panel", "t5", "unseen",
                      "does_not_fit", "exposed")


def per_instance(runs: pd.DataFrame, inc: pd.DataFrame, strata: pd.DataFrame, lab: dict,
                 registered: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """The per-instance table (module docstring) and the selection notes."""
    want = [("A9", 2000), ("A6F", 2000), ("A0", 2000), ("A2F", 2000), ("A9", 2001), ("A6F", 2001)]
    chosen, notes = {}, {}
    for arm, seed in want:
        c, w = select(runs, lab[arm], seed, registered=registered)
        chosen[(arm, seed)] = c
        if w:
            notes[f"{arm}@s{seed}"] = w
    rids = {str(r["run_id"]): int(r["sense"]) for c in chosen.values() for r in c.values()}
    steps = steps_by_run(inc, rids.keys(), rids)
    rows = []
    for _, srow in strata.iterrows():
        n = str(srow["name"])
        out = dict(instance=n, tier=srow["tier"], **{k: bool(srow[k]) for k in PER_INSTANCE_FLAGS},
                   p_gams_cuopt_2608=float(srow["p_cuopt"]))
        race_parts = {}
        for (arm, seed), c in chosen.items():
            r = c.get(n)
            key = f"{arm}" + ("" if seed == 2000 else f"_s{seed}")
            if r is None:
                out[f"{key}_run"] = ""
                continue
            st = steps[str(r["run_id"])]
            off = float(r.get("startup_offset_s", math.nan))
            m = run_metrics(st, float(r["ref"]), int(r["sense"]), T_EXT, off)
            out[f"{key}_run"] = str(r["run_id"])
            out[f"{key}_status"] = str(r["status"])
            out[f"{key}_stop"] = str(r.get("stop_kind", "") or "")
            out[f"{key}_startup_s"] = off                 # launch clock − call clock (R10's start-up cost)
            out.update({f"{key}_{k}": v for k, v in m.items()})
            if seed == 2000 and arm in ("A0", "A6F"):
                race_parts[arm] = (st, off, float(r["ref"]), int(r["sense"]))
        if len(race_parts) == 2:                         # R8: the computed race of A0 and A6F, same seed
            (s0, o0, z, s), (s6, o6, _, _) = race_parts["A0"], race_parts["A6F"]
            m = run_metrics(M.merge_steps(s0, s6, sense=s), z, s, T_EXT, 0.0)
            ml = M.gams_integral(M.merge_steps(_shift(s0, o0), _shift(s6, o6), sense=s), z, s, T_EXT).value \
                if math.isfinite(o0) and math.isfinite(o6) else math.nan
            out.update(race_pt_gams_call=m["pt_gams_call"], race_pt_gams_launch=ml)
        rows.append(out)
    return pd.DataFrame(rows).set_index("instance"), notes


def _pairs(df: pd.DataFrame, mask, a: str, b: str):
    """The paired values of columns a and b on the instances of `mask` where both exist (empty if a column is absent:
    no run of that arm)."""
    if a not in df or b not in df:
        return np.zeros(0), np.zeros(0), []
    m = mask & df[a].notna() & df[b].notna()
    return df.loc[m, a].to_numpy(float), df.loc[m, b].to_numpy(float), list(df.index[m])


def _have_pair(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    if a not in df or b not in df:
        return pd.Series(False, index=df.index)
    return df[a].notna() & df[b].notna()


def _arm_vs_ctrl(df: pd.DataFrame, mask, arm_col: str, ctrl_col: str, eps: float, bca: bool) -> dict:
    """Ratio of means arm / control with its paired bootstrap, and the d_ε side (prefixed d_: n / lo / hi stay the
    ratio's), on the instances of `mask` where both exist; N = the instances of `mask`, missing = N − n."""
    x, y, names = _pairs(df, mask, arm_col, ctrl_col)
    e = effect_summary(d_eps(y, x, eps), eps, names, bca=bca)
    out = dict(ratio_of_means(x, y, bca=bca), d_mean=e["mean_d"], d_lo=e["lo"], d_hi=e["hi"], n_d=e["n"],
               wilcoxon_p=e["wilcoxon_p"], improved=e["improved"], worse=e["worse"], sign_p=e["sign_p"],
               worst_d=e.get("worst_d", math.nan), worst=e.get("worst"), N=int(mask.sum()),
               missing=int(mask.sum()) - len(names))
    if bca:
        out.update(d_bca_lo=e.get("bca_lo", math.nan), d_bca_hi=e.get("bca_hi", math.nan))
    return out


def decisions(df: pd.DataFrame, strata: pd.DataFrame, clock: str, exclude=(), bca: bool = False) -> dict:
    """The decisive aggregates of one clock (R19, R31) on the decision set minus `exclude` (R33's reading without the
    exposed instances), with the completeness that decides how they are read:
    - `completeness`: pairs present / expected on the decision set (`full_complete`), on the census + E-sample basis —
      H_cu-unseen, W-only and the E-sample (`census_complete`) — and on T3's decision instances (`t3_complete`);
    - **target** (R19): the ratio of mean pt_gams A9 / A6F on H_cu-unseen. `met` / `better_shown` are read only on the
      complete set (T1 is never cut, R32): with any pair missing they are None and the reading says "incomplete (n of
      N)"; `met_on_available` / `better_shown_on_available` keep the figures on the pairs present;
    - **guard** and **solutions found** on the pairs present (`passes_on_available`, `harm_shown_on_available`);
    - **minimum** (R19 (a) and (b)), read on the full decision set when every pair is present (basis "full"), else —
      T3 incomplete, the census and E-sample complete — on R31's stratified estimates (basis "stratified": (a) the
      stratified guard >= −0.05, harm shown iff its interval's upper end < −0.05; (b) the stratified count of
      solutions found A9 − A6F >= 0), else not read (basis "incomplete", None)."""
    eps = eps_for(T_EXT)
    ex = df.index.isin(set(exclude))
    dec = df["decide"] & ~ex
    hcu = dec & df["H_cu"]
    a9, a6 = f"A9_pt_gams_{clock}", f"A6F_pt_gams_{clock}"
    have = _have_pair(df, a9, a6)
    census = dec & (df["H_cu"] | df["W_only"] | df["e_sample"])
    t3 = dec & (df["tier"] == "T3")
    comp = {k: dict(n=int((m & have).sum()), N=int(m.sum())) for k, m in
            (("decision_set", dec), ("census", census), ("t3", t3), ("target", hcu))}
    comp["full_complete"] = comp["decision_set"]["n"] == comp["decision_set"]["N"]
    comp["census_complete"] = comp["census"]["n"] == comp["census"]["N"]
    comp["t3_complete"] = comp["t3"]["n"] == comp["t3"]["N"]
    c: dict = dict(N_decide=int(dec.sum()), completeness=comp)
    t = _arm_vs_ctrl(df, hcu, a9, a6, eps, bca)
    t["complete"] = bool(t["missing"] == 0 and t["N"] > 0)
    fin = math.isfinite(t["ratio"])
    t["met_on_available"] = bool(t["ratio"] <= TARGET_RATIO) if fin else None
    t["better_shown_on_available"] = bool(t["hi"] < 1) if math.isfinite(t["hi"]) else None
    t["met"] = t["met_on_available"] if t["complete"] else None
    t["better_shown"] = t["better_shown_on_available"] if t["complete"] else None
    c["target"] = t
    x, y, names = _pairs(df, dec, a9, a6)
    g = effect_summary(d_eps(y, x, eps), eps, names, bca=bca)
    g["N"], g["missing"] = int(dec.sum()), int(dec.sum()) - len(names)
    g["passes_on_available"] = bool(g["mean_d"] >= GUARD) if math.isfinite(g["mean_d"]) else None
    g["harm_shown_on_available"] = bool(g["hi"] < GUARD) if math.isfinite(g["hi"]) else None
    c["guard"] = g
    sol = {}
    for scope, mask in (("decide", dec), ("run_set", ~df["does_not_fit"] & ~ex)):
        xs, ys, nm = _pairs(df, mask, a9, a6)
        sol[scope] = dict(n=len(nm), A9=int((xs < M.MIPFEAS_NONE).sum()), A6F=int((ys < M.MIPFEAS_NONE).sum()))
    sol["passes_on_available"] = sol["decide"]["A9"] >= sol["decide"]["A6F"] if sol["decide"]["n"] else None
    c["solutions"] = sol
    # R31: stratified estimates of the guard and solutions found (always reported; decisive while T3 is incomplete)
    tmp = pd.DataFrame(index=df.index, data=dict(d=np.nan, found_diff=np.nan))
    if a9 in df and a6 in df:
        pa, pc = df[a9].astype(float), df[a6].astype(float)
        tmp["d"] = np.where(have, d_eps(pc, pa, eps), np.nan)
        tmp["found_diff"] = np.where(have, (pa < M.MIPFEAS_NONE).astype(float) - (pc < M.MIPFEAS_NONE).astype(float),
                                     np.nan)
    sg = stratified(tmp, "d", strata, exclude)
    ss = stratified(tmp, "found_diff", strata, exclude)
    if ss.get("total"):
        ss["count_estimate"] = ss["estimate"] * ss["total"]
    if bca and sg.get("complete"):
        sg["note"] = "BCa is not computed for the stratified estimate (a percentile interval within strata only)"
    c["stratified"] = dict(guard=sg, solutions_diff=ss)
    # the minimum, on the basis the completeness allows
    if comp["full_complete"] and comp["decision_set"]["N"] > 0:
        basis, a, harm = "full", g["passes_on_available"], g["harm_shown_on_available"]
        b = sol["passes_on_available"]
    elif comp["census_complete"] and math.isfinite(sg["estimate"]) and math.isfinite(ss.get("count_estimate",
                                                                                             math.nan)):
        basis = "stratified"
        a, harm = bool(sg["estimate"] >= GUARD), bool(sg["hi"] < GUARD)
        b = bool(ss["count_estimate"] >= -1e-9)          # float noise of the weighted sum, not a solution
    else:
        basis, a, b, harm = "incomplete", None, None, None
    c["minimum_basis"] = basis
    c["minimum_detail"] = dict(basis=basis, guard_passes=a, harm_shown=harm, solutions_pass=b,
                               guard=(g["mean_d"] if basis == "full" else sg["estimate"] if basis == "stratified"
                                      else math.nan),
                               solutions_diff=(sol["decide"]["A9"] - sol["decide"]["A6F"] if basis == "full" else
                                               ss.get("count_estimate", math.nan) if basis == "stratified"
                                               else math.nan))
    c["minimum"] = bool(a and b) if basis != "incomplete" else None
    c["t3_complete"] = comp["t3_complete"]
    c["census_complete"] = comp["census_complete"]
    return c


def startup_summary(df: pd.DataFrame, masks: dict) -> dict:
    """R10's start-up cost: per arm, the median and 90th percentile of `startup_offset_s` (launch clock − call clock)
    over the runs taken on each set of `masks`."""
    out: dict = {}
    for arm in ("A9", "A6F", "A0", "A2F"):
        col = f"{arm}_startup_s"
        if col not in df:
            continue
        out[arm] = {}
        for name, m in masks.items():
            v = df.loc[m & df[col].notna(), col].astype(float).to_numpy()
            v = v[np.isfinite(v)]
            out[arm][name] = dict(n=int(v.size), median=float(np.median(v)) if v.size else math.nan,
                                  p90=float(np.percentile(v, 90)) if v.size else math.nan)
    return out


def _reading(call, launch, kind: str) -> str:
    """R10's words for one decision: "met" only if met on both clocks, "met on the call clock only", "not met", or
    "incomplete (n of N)" when the call clock's set is not complete enough to read it."""
    if kind == "target":
        a, b = call["target"]["met"], launch["target"]["met"]
        if a is None:
            ct = call["completeness"]["target"]
            return f"incomplete ({ct['n']} of {ct['N']} H_cu-unseen pairs)"
    else:
        a, b = call["minimum"], launch["minimum"]
        if a is None:
            cc, cd = call["completeness"]["census"], call["completeness"]["decision_set"]
            return (f"incomplete (census + E-sample {cc['n']} of {cc['N']} pairs, decision set {cd['n']} of "
                    f"{cd['N']})")
    if a and b:
        return "met"
    if a:
        return "met on the call clock only"
    return "not met"


def readings(out: dict, df: pd.DataFrame, hcu, dec) -> tuple[dict, dict]:
    """R10 for the target and the minimum, and beside each its detail: the basis, and — when it holds on the call
    clock only — the start-up cost (each arm's median / p90 start-up on the set) and the launch clock's figure."""
    rd, det = {}, {}
    st = startup_summary(df, dict(target=hcu, decision_set=dec))
    for k in ("target", "minimum"):
        rd[k] = _reading(out["call"], out["launch"], k)
        d: dict = {}
        if k == "target":
            d.update(call_ratio=out["call"]["target"]["ratio"], launch_ratio=out["launch"]["target"]["ratio"],
                     n=out["call"]["target"]["n"], N=out["call"]["target"]["N"])
        else:
            d.update(basis=out["call"]["minimum_basis"], call=out["call"]["minimum_detail"],
                     launch=out["launch"]["minimum_detail"])
        if rd[k] == "met on the call clock only":
            sset = "target" if k == "target" else "decision_set"
            d["startup"] = {arm: v[sset] for arm, v in st.items()}
        det[k] = d
    return rd, det


def aggregates(df: pd.DataFrame, strata: pd.DataFrame, bca: bool = False) -> dict:
    eps = eps_for(T_EXT)
    dec = df["decide"]
    hcu = dec & df["H_cu"]
    wsc = dec & df["W_SCIP"]
    ran = ~df["does_not_fit"]
    exposed = sorted(df.index[df["exposed"]]) if "exposed" in df else []
    out: dict = dict(T=T_EXT, eps=eps, counts=dict(decide=int(dec.sum()), H_cu_unseen=int(hcu.sum()),
                                                    W_SCIP_unseen=int(wsc.sum()), run_set=int(ran.sum()),
                                                    exposed_decide=int((dec & df.index.isin(exposed)).sum())))
    for clock in CLOCKS:
        a9, a6 = f"A9_pt_gams_{clock}", f"A6F_pt_gams_{clock}"
        c = decisions(df, strata, clock, bca=bca)
        # R3: H_cu at P >= 0.5 (GAMS's cuOpt row), reported beside the target
        c["target_p05"] = _arm_vs_ctrl(df, dec & df["H_cu_p05"], a9, a6, eps, bca)
        # secondary (Holm): A9 vs the computed race on H_cu-unseen, A9 vs A0 on W_SCIP-unseen
        sec = {}
        x, y, names = _pairs(df, hcu, a9, f"race_pt_gams_{clock}")
        sec["A9_vs_race_Hcu"] = effect_summary(d_eps(y, x, eps), eps, names, bca=bca)
        x, y, names = _pairs(df, wsc, a9, f"A0_pt_gams_{clock}")
        sec["A9_vs_A0_Wscip"] = effect_summary(d_eps(y, x, eps), eps, names, bca=bca)
        p = [sec[k]["wilcoxon_p"] for k in sec]
        adj, rej = S.holm([v if math.isfinite(v) else 1.0 for v in p])
        for k, a_, r_ in zip(sec, adj, rej):
            sec[k].update(holm_p=float(a_), holm_reject=bool(r_))
        c["secondary"] = sec
        # R8: live A2F against the computed race on the panel (T5) — how much the computed race flatters a real one:
        # d_ε of A2F against the race (negative = the live race worse), the ratio of means A2F / race
        c["A2F_vs_race_panel"] = _arm_vs_ctrl(df, df["t5"], f"A2F_pt_gams_{clock}", f"race_pt_gams_{clock}", eps, bca)
        # reported: geo-means per arm, over the run set and the decision set
        geo = {}
        for arm in ("A9", "A6F", "A0", "race", "A2F"):
            col = f"{arm}_pt_gams_{clock}"
            if col in df:
                geo[arm] = {scope: dict(n=int((mask & df[col].notna()).sum()), **shifted_geomeans(df.loc[mask, col]))
                            for scope, mask in (("run_set", ran), ("decide", dec))}
        c["geomeans"] = geo
        # D-019's literal readings (beside)
        x, y, names = _pairs(df, ran, a9, a6)
        lit: dict = dict(n=len(names))
        if names:
            ga, gb = M.shifted_geomean(x, shift=SHIFTS[0]), M.shifted_geomean(y, shift=SHIFTS[0])
            hard = (y >= 0.1)
            lit.update(geo_A9=ga, geo_A6F=gb, minimum=bool(ga <= gb), geo_ratio=(ga + 0.0) / gb if gb else math.nan,
                       target_geo=bool(gb and ga / gb <= LIT_GEO_RATIO), stretch=bool(ga <= STRETCH_SGM),
                       hard_n=int(hard.sum()),
                       hard_ratio=float(x[hard].mean() / y[hard].mean()) if hard.any() and y[hard].mean() > 0
                       else math.nan)
            lit["target_hard"] = bool(lit["hard_ratio"] <= TARGET_RATIO) if math.isfinite(lit["hard_ratio"]) else None
        c["d019_literal"] = lit
        out[clock] = c
    # R10: met on both clocks, or on the call clock only (with the start-up cost), or incomplete
    out["reading"], out["reading_detail"] = readings(out, df, hcu, dec)
    out["startup"] = startup_summary(df, dict(target=hcu, decision_set=dec))
    # R33: every decision also without the instances exposed before prereg-v2 (calibration sample, M3's probe)
    wo: dict = dict(excluded=exposed, n_excluded_decide=out["counts"]["exposed_decide"])
    for clock in CLOCKS:
        wo[clock] = decisions(df, strata, clock, exclude=exposed, bca=bca)
    ex_m = df.index.isin(exposed)
    wo["reading"], wo["reading_detail"] = readings(wo, df, hcu & ~ex_m, dec & ~ex_m)
    out["without_exposed"] = wo
    # R20: the panel's per-instance noise scale
    noise = {}
    for arm in ("A9", "A6F"):
        a, b = f"{arm}_pt_gams_call", f"{arm}_s2001_pt_gams_call"
        if a in df and b in df:
            m = df[a].notna() & df[b].notna()
            dd = np.abs(d_eps(df.loc[m, a], df.loc[m, b], eps))
            noise[arm] = dict(n=int(m.sum()), p90_abs_d=float(np.percentile(dd, 90)) if dd.size else math.nan)
    out["panel_noise"] = noise
    return out


def published_rows(df: pd.DataFrame, gams_dir: str | Path) -> dict:
    """R36: GAMS's own traces re-scored by the port on exactly our completed subset (A9 and A6F at seed 2000)."""
    from itrea.analysis import mipfeas as MF
    gd = Path(gams_dir)
    if "A9_pt_gams_call" not in df or "A6F_pt_gams_call" not in df or not (gd / "trc_files").is_dir():
        return {}
    sub = sorted(df.index[df["A9_pt_gams_call"].notna() & df["A6F_pt_gams_call"].notna()])
    if not sub:
        return {}
    z = MF.read_z(gd)
    runs = sorted({p.name[:-4].rsplit("_", 2)[1] + "_" + p.name[:-4].rsplit("_", 2)[2]
                   for p in (gd / "trc_files").glob("*.trc")})
    out = dict(subset=len(sub), rows={})
    for run in runs:
        ps = [MF.trc_integral(MF.trc_path(gd, n, run), z[n]).value for n in sub
              if MF.trc_path(gd, n, run).exists() and n in z]
        machine, solver = run.split("_", 1)
        out["rows"][run] = dict(machine=machine, solver=solver, n=len(ps),
                                mean=float(np.mean(ps)) if ps else math.nan,
                                sgm=M.shifted_geomean(ps, shift=SHIFTS[0]) if ps else math.nan,
                                feasible=int(sum(p < M.MIPFEAS_NONE for p in ps)),
                                footprint=("GB10, 20 cores / 24 threads + GPU" if machine == "gb10" else
                                           f"{machine}, as GAMS ran it"), clock="GAMS's call clock")
    for arm in ("A9", "A6F"):
        for clock in CLOCKS:
            p = df.loc[sub, f"{arm}_pt_gams_{clock}"].astype(float)
            out[f"ours_{arm}_{clock}"] = dict(n=len(sub), mean=float(p.mean()), sgm=M.shifted_geomean(p, shift=0.001),
                                              footprint="F: X925 5-6 + A725 1-3 + the GPU", clock=clock)
    return out


def report(runs: pd.DataFrame, inc: pd.DataFrame, strata: pd.DataFrame, cuopt: str = "26.10",
           gams_dir: str | Path | None = None, bca: bool = False, registered: dict | None = None) -> dict:
    """The whole reading. `registered` (`registered_values`): only runs carrying the registered itx1 harness (and, for
    A9, the registered configuration) are taken; None (synthetic tables in tests) checks nothing."""
    lab = labels(cuopt)
    df, notes = per_instance(runs, inc, strata, lab, registered)
    out = dict(labels=lab, cuopt=cuopt, notes=notes, aggregates=aggregates(df, strata, bca),
               registered=(None if registered is None else
                           dict(frozen=registered["frozen"], A9_base_hash=registered["A9_base_hash"],
                                harness=registered["harness"])),
               tiers={t: dict(instances=int((df["tier"] == t).sum()),
                              pairs=int(((df["tier"] == t) & df.get("A9_pt_gams_call", pd.Series(np.nan, index=df.index))
                                         .notna() & df.get("A6F_pt_gams_call", pd.Series(np.nan, index=df.index))
                                         .notna()).sum())) for t in ("T1", "T2", "T3")})
    if gams_dir is not None:
        out["published"] = published_rows(df, gams_dir)
    out["instances"] = df.reset_index().to_dict("records")
    return out


# --------------------------------------------------------------------------------------------------------- M6
def m6_report(runs: pd.DataFrame) -> dict:
    """D-021 M6: nodes/s beside 3 busy lanes vs alone, per instance (A0@m6b vs A0@m6a), and the heat phase's hottest
    ACPI zone (set itx-m6). Lanes: 4 if the mean slowdown <= 3 % and the hottest zone < 85 °C, else 3."""
    a = runs[runs["arm"].astype(str) == "A0@m6a"].set_index("instance")["solver_nodes_per_s"]
    b = runs[runs["arm"].astype(str) == "A0@m6b"].set_index("instance")["solver_nodes_per_s"]
    common = sorted(set(a.index) & set(b.index))
    per = {n: dict(alone=float(a[n]), beside=float(b[n]), slowdown=1.0 - float(b[n]) / float(a[n])
                   if float(a[n]) > 0 else math.nan) for n in common}
    sl = [v["slowdown"] for v in per.values() if math.isfinite(v["slowdown"])]
    heat = runs[runs["set"].astype(str) == "itx-m6"]
    hot = float(heat["acpi_during_max_c"].max()) if "acpi_during_max_c" in heat and len(heat) else math.nan
    mean_sl = float(np.mean(sl)) if sl else math.nan
    lanes = None
    if sl and math.isfinite(hot):
        lanes = 4 if (mean_sl <= M6_SLOWER and hot < M6_HOT_C) else 3
    return dict(instances=per, mean_slowdown=mean_sl, hottest_c=hot, lanes=lanes,
                rule=f"4 lanes iff the mean nodes/s slowdown beside 3 lanes <= {M6_SLOWER:.0%} and the hottest zone "
                     f"< {M6_HOT_C:g} °C over the heat phase, else 3")


# ----------------------------------------------------------------------------------------------------- output
def _f(x, nd=4) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _bca(d: dict, pre: str = "") -> str:
    lo, hi = d.get(f"{pre}bca_lo"), d.get(f"{pre}bca_hi")
    return "" if lo is None and hi is None else f", BCa [{_f(lo, 3)}, {_f(hi, 3)}]"


def _decision_lines(c: dict) -> list[str]:
    """The decisive aggregates of one clock (`decisions`) as markdown lines."""
    t, g, s, cm = c["target"], c["guard"], c["solutions"], c["completeness"]
    sg, ss, md = c["stratified"]["guard"], c["stratified"]["solutions_diff"], c["minimum_detail"]
    return [
        f"- Completeness: decision set {cm['decision_set']['n']} of {cm['decision_set']['N']} pairs; census + "
        f"E-sample (H_cu-unseen, W-only, E-sample) {cm['census']['n']} of {cm['census']['N']}; T3 (decision) "
        f"{cm['t3']['n']} of {cm['t3']['N']} — the minimum is read on: {md['basis']}",
        f"- Target (H_cu-unseen, n {t['n']} of {t['N']}): mean pt_gams ratio A9/A6F "
        f"{_f(t['ratio'], 3)} [{_f(t['lo'], 3)}, {_f(t['hi'], 3)}]{_bca(t)} (one-sided 95 % bound {_f(t['hi95'], 3)}); "
        f"met (<= {TARGET_RATIO}) {t['met']}, better shown {t['better_shown']}"
        + ("" if t["complete"] else f" (incomplete; on the pairs present: met {t['met_on_available']}, better "
                                    f"shown {t['better_shown_on_available']})")
        + f"; mean d_ε {_f(t['d_mean'], 3)} [{_f(t['d_lo'], 3)}, {_f(t['d_hi'], 3)}]{_bca(t, 'd_')}, Wilcoxon p "
          f"{_f(t['wilcoxon_p'], 3)}; improved {t['improved']} / worse {t['worse']} (sign p {_f(t['sign_p'], 3)}); "
          f"worst d_ε {_f(t.get('worst_d'), 3)} ({t.get('worst')})",
        f"- Guard on the pairs present (decision set, n {g['n']} of {g['N']}): mean d_ε {_f(g['mean_d'], 3)} "
        f"[{_f(g['lo'], 3)}, {_f(g['hi'], 3)}]{_bca(g)}, Wilcoxon p {_f(g['wilcoxon_p'], 3)}; >= {GUARD} "
        f"{g['passes_on_available']}, harm shown {g['harm_shown_on_available']}",
        f"- Solutions found (pt_gams < 2) on the pairs present: decision set A9 {s['decide']['A9']} vs A6F "
        f"{s['decide']['A6F']} (n {s['decide']['n']}); the run set A9 {s['run_set']['A9']} vs A6F "
        f"{s['run_set']['A6F']}",
        f"- Stratified (R31; {'decisive: T3 incomplete' if md['basis'] == 'stratified' else 'reported'}): guard "
        f"{_f(sg['estimate'], 3)} [{_f(sg['lo'], 3)}, {_f(sg['hi'], 3)}]; solutions A9 − A6F "
        f"{_f(ss.get('count_estimate'), 1)}",
        f"- Minimum (R19 (a) and (b), basis {md['basis']}): guard {_f(md['guard'], 3)} >= {GUARD} "
        f"{md['guard_passes']} (harm shown {md['harm_shown']}); solutions A9 − A6F {_f(md['solutions_diff'], 1)} "
        f">= 0 {md['solutions_pass']} -> {c['minimum']}"]


def _reading_line(rd: dict, det: dict) -> str:
    def startup(d):
        st = d.get("startup")
        if not st:
            return ""
        return " (start-up, launch − call, median / p90: " + ", ".join(
            f"{a} {_f(v['median'], 2)} / {_f(v['p90'], 2)} s" for a, v in st.items() if v["n"]) + ")"
    t, m = det["target"], det["minimum"]
    return (f"target {rd['target']}{startup(t)} (ratio: call clock {_f(t['call_ratio'], 3)}, launch clock "
            f"{_f(t['launch_ratio'], 3)}); minimum {rd['minimum']}{startup(m)} (basis {m['basis']}; guard: call "
            f"{_f(m['call']['guard'], 3)}, launch {_f(m['launch']['guard'], 3)})")


def markdown(res: dict) -> str:
    ag = res["aggregates"]
    lab = res["labels"]
    reg = res.get("registered")
    L = [f"<!-- generated by itrea.analysis.itx_report (D-021, ITX-1), cuOpt {res['cuopt']} -->",
         f"ITX-1 v1 report: {lab['A9']} vs {lab['A6F']} (A0 = {lab['A0']}), T = {ag['T']:g} s, ε = {ag['eps']:g}; "
         f"tiers (instances / pairs): " + ", ".join(f"{t} {v['instances']} / {v['pairs']}" for t, v in
                                                    res["tiers"].items())]
    if reg is not None:
        L.append(f"Runs taken only with the registered settings: harness itx1, A9 on {reg['frozen']} "
                 f"({reg['A9_base_hash'] or 'NOT REGISTERED: no A9 run taken'}).")
    L.append("")
    for clock in CLOCKS:
        c = ag[clock]
        L += [f"### {clock} clock"] + _decision_lines(c)
        tp = c["target_p05"]
        L.append(f"- Beside (R3): H_cu at P >= 0.5, decision set (n {tp['n']} of {tp['N']}): ratio A9/A6F "
                 f"{_f(tp['ratio'], 3)} [{_f(tp['lo'], 3)}, {_f(tp['hi'], 3)}], mean d_ε {_f(tp['d_mean'], 3)}")
        for k, v in c["secondary"].items():
            L.append(f"- {k} (n {v['n']}): mean d_ε {_f(v['mean_d'], 3)} [{_f(v['lo'], 3)}, {_f(v['hi'], 3)}]"
                     f"{_bca(v)}, Wilcoxon p {_f(v['wilcoxon_p'], 3)}, Holm p {_f(v.get('holm_p'), 3)}")
        ar = c["A2F_vs_race_panel"]
        L.append(f"- R8, live A2F vs the computed race on the panel (n {ar['n']} of {ar['N']}): ratio of means A2F / "
                 f"race {_f(ar['ratio'], 3)} [{_f(ar['lo'], 3)}, {_f(ar['hi'], 3)}], mean d_ε {_f(ar['d_mean'], 3)} "
                 f"[{_f(ar['d_lo'], 3)}, {_f(ar['d_hi'], 3)}] (negative: the computed race flatters the live one)")
        for arm, gm in c["geomeans"].items():
            L.append(f"- geo-mean {arm}: " + "; ".join(
                f"{scope} (n {d['n']}) " + ", ".join(f"{k} {_f(v)}" for k, v in d.items() if k != "n")
                for scope, d in gm.items()))
        lit = c["d019_literal"]
        if lit.get("n"):
            L.append(f"- D-019 literal (n {lit['n']}): geo A9 {_f(lit['geo_A9'])} vs A6F {_f(lit['geo_A6F'])} "
                     f"(minimum {lit['minimum']}, ratio {_f(lit['geo_ratio'], 3)}, −10 % {lit['target_geo']}, "
                     f"stretch <= {STRETCH_SGM} {lit['stretch']}); hard part by our A6F (n {lit['hard_n']}) ratio "
                     f"{_f(lit['hard_ratio'], 3)} (target {lit['target_hard']})")
        L.append("")
    L.append("Reading (R10): " + _reading_line(ag["reading"], ag["reading_detail"]) + ".")
    if ag.get("startup"):
        L.append("Start-up (launch − call clock, median / p90 s; R10): " + "; ".join(
            f"{a} " + ", ".join(f"{k} {_f(v['median'], 2)} / {_f(v['p90'], 2)} (n {v['n']})" for k, v in d.items())
            for a, d in ag["startup"].items()))
    wo = ag["without_exposed"]
    L += ["", f"### Without the exposed instances (R33: calibration sample + M3's probe; {wo['n_excluded_decide']} of "
              f"the decision set out)"]
    for clock in CLOCKS:
        L += [f"#### {clock} clock"] + _decision_lines(wo[clock])
    L.append("Reading without the exposed (R10): " + _reading_line(wo["reading"], wo["reading_detail"]) + ".")
    L.append("")
    if ag["panel_noise"]:
        L.append("Panel noise (R20): " + "; ".join(f"{a} p90 |d_ε| {_f(v['p90_abs_d'], 3)} (n {v['n']})"
                                                   for a, v in ag["panel_noise"].items()))
    pub = res.get("published") or {}
    if pub:
        L.append(f"Published rows re-scored on our completed subset ({pub['subset']} instances; descriptive):")
        for run, r in pub["rows"].items():
            L.append(f"- GAMS {r['solver']} on {r['machine']} ({r['footprint']}, {r['clock']}): mean {_f(r['mean'])},"
                     f" geo-mean {_f(r['sgm'])}, feasible {r['feasible']}")
        for k in sorted(x for x in pub if x.startswith("ours_")):
            r = pub[k]
            L.append(f"- {k[5:]} ({r['footprint']}): mean {_f(r['mean'])}, geo-mean {_f(r['sgm'])}")
    if res["notes"]:
        L.append("Runs not taken: " + "; ".join(f"{k}: {len(v)} instance(s)" for k, v in res["notes"].items()))
    return "\n".join(L) + "\n"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv: list[str] | None = None) -> int:
    from itrea.analysis.itx_strata import STRATA_CSV, read_strata
    from itrea.analysis.mipfeas import GAMS_DIR
    ap = argparse.ArgumentParser(prog="itrea itx-report", description="D-021 (ITX-1): the v1 external analysis (target, "
                                 "minimum, stratified estimates, secondaries, geo-means, D-019's literal readings, "
                                 "published rows on our subset) or M6's lane reading (--m6)")
    ap.add_argument("--cuopt", default="26.10", help="the cuOpt version of the runs (their variant v1c<tag>)")
    ap.add_argument("--runs", default=str(DATASETS / "runs.parquet"))
    ap.add_argument("--inc", default=str(DATASETS / "incumbents_checked.parquet"))
    ap.add_argument("--strata", default=str(STRATA_CSV))
    ap.add_argument("--gams-dir", default=str(GAMS_DIR))
    ap.add_argument("--bca", action="store_true", help="BCa intervals beside the percentile ones (a sensitivity, R21): "
                                                       "the target ratio, every mean d_ε (target, guard, secondaries)")
    ap.add_argument("--frozen", default=None, help="A9's registered configuration (default configs/frozen-v2.yaml, "
                                                   "the C1 freeze registered with prereg-v2); only A9 runs on it count")
    ap.add_argument("--m6", action="store_true", help="M6's lane reading instead")
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    if not Path(a.runs).exists():
        print(f"{a.runs} does not exist (run `itrea build-runs` first)", file=sys.stderr)
        return 2
    runs = pd.read_parquet(a.runs)
    if a.m6:
        res = m6_report(runs)
        print(json.dumps(_jsonable(res), indent=1))
        if a.json:
            Path(a.json).write_text(json.dumps(_jsonable(res), indent=1) + "\n")
        return 0
    reg = registered_values(a.frozen)
    if reg["A9_base_hash"] is None:
        print(f"warning: {reg['frozen']} does not exist — A9's registered configuration is unknown, so no A9 run is "
              f"taken", file=sys.stderr)
    res = report(runs, pd.read_parquet(a.inc), read_strata(a.strata), a.cuopt,
                 a.gams_dir if Path(a.gams_dir).exists() else None, a.bca, registered=reg)
    md = markdown(res)
    print(md)
    if a.json:
        Path(a.json).write_text(json.dumps(_jsonable(res), indent=1, sort_keys=True) + "\n")
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
