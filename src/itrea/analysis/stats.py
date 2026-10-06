"""Pre-registered statistics (vault D-002 v3 §6): the bounded per-instance effect, its tests, the decision rule, and
the censored-data and lineage tests.

Why these, briefly (the critique's methodology lens, 2026-10-05):
- **d_i = (P_ctrl − P_arm) / max(P_ctrl, P_arm, 10⁻³)** lies in [−1, 1], so a near-zero control cannot blow up a
  ratio (30 of optopt's 80 MIPLIB instances had SCIP P(300) < 0.02); positive = the arm is better. Seeds are first
  averaged per arm on the seeds both arms share (common random numbers: a block is instance × seed), then d is
  formed. An instance leaves a comparison only if both arms have P = 1 on all seeds.
- **Wilcoxon signed-rank on d_i**, two-sided, `zero_method="wilcox"` (zeros dropped), exact when the non-zero count
  is ≤ 50 and |d| has no ties, else the normal approximation without continuity correction. The method is decided
  here, not left to SciPy, which would switch silently.
- **Mean d_i with a 95 % percentile bootstrap** (10,000 resamples of instances, fixed seed → deterministic).
- **Holm** step-down within a family; H1a and H1b are an intersection–union pair (each at α, no adjustment).
- **Decision rule** (per comparison): "arm worse" if the whole interval lies below 0; "supported" if p < α and the
  interval lies above 0 ("excludes 0" on the arm's side — the other side is "arm worse"); "negative" if the whole
  interval lies below the +0.05 non-inferiority margin; "inconclusive" otherwise.
- **Censored times** (time to first feasible, to 5 % and 1 % gap; Type I censoring at T): the paired
  Prentice–Wilcoxon test of O'Brien & Fleming (1987, Biometrics 43:169–180) on per-instance seed medians, with a
  sign test as the robustness check. PPW scores come from Prentice's modified Kaplan–Meier estimate on the pooled 2n
  observations, S̃(t) = Π_{t_j ≤ t} (n_j + 1 − d_j)/(n_j + 1): an event at t scores 1 − 2·S̃(t), a censored value
  1 − S̃(t); with s_i the arm-minus-control score difference, Z = Σ s_i / √(Σ s_i²) is approximately N(0, 1).
  Lower time is better, so a positive mean score difference means the arm is SLOWER; `z` is reported with the
  sign flipped so that positive = arm better, like d.
- **Mantel–Haenszel** pooled odds ratio over 2 × 2 tables (lineage, D-002 §4), with the Robins–Breslow–Greenland
  variance of log OR.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as _st

ALPHA = 0.05
MARGIN = 0.05             # non-inferiority margin on d (D-002 §6)
D_FLOOR = 1e-3            # denominator floor of d_i
N_BOOT = 10_000
BOOT_SEED = 20261005      # the pre-registration's master seed (D-002 §6 run order uses the same number)
EXACT_MAX_N = 50

SUPPORTED, NEGATIVE, ARM_WORSE, INCONCLUSIVE = "supported", "negative", "arm worse", "inconclusive"


# ------------------------------------------------------------------------------------------------- effect d
def effect_d(p_ctrl: float, p_arm: float, floor: float = D_FLOOR) -> float:
    """Bounded per-instance effect; positive = the arm (lower P) is better."""
    return (p_ctrl - p_arm) / max(p_ctrl, p_arm, floor)


def instance_effects(df: pd.DataFrame, arm: str, ctrl: str, value: str = "pt", instance: str = "instance",
                     seed: str = "seed", arm_col: str = "arm", floor: float = D_FLOOR) -> pd.DataFrame:
    """Per-instance seed-averaged P of both arms on their COMMON seeds, d_i, and the exclusion flag.

    df: one row per run (instance, seed, arm, value). Duplicate (instance, seed, arm) rows are an error (the caller
    resolves reruns before). Returns columns instance, n_seeds, p_arm, p_ctrl, d, excluded (both P = 1 on all
    seeds), sorted by instance."""
    sub = df[df[arm_col].isin([arm, ctrl])][[instance, seed, arm_col, value]]
    if sub.duplicated([instance, seed, arm_col]).any():
        dup = sub[sub.duplicated([instance, seed, arm_col], keep=False)]
        raise ValueError(f"duplicate runs for {dup[[instance, seed, arm_col]].drop_duplicates().values[:3].tolist()}")
    wide = sub.pivot_table(index=[instance, seed], columns=arm_col, values=value, aggfunc="first")
    if arm not in wide.columns or ctrl not in wide.columns:
        return pd.DataFrame(columns=[instance, "n_seeds", "p_arm", "p_ctrl", "d", "excluded"])
    wide = wide[[arm, ctrl]].dropna()
    rows = []
    for inst, g in wide.groupby(level=0, sort=True):
        pa, pc = float(g[arm].mean()), float(g[ctrl].mean())
        excl = bool((g[arm] >= 1.0).all() and (g[ctrl] >= 1.0).all())
        rows.append({instance: inst, "n_seeds": int(len(g)), "p_arm": pa, "p_ctrl": pc,
                     "d": effect_d(pc, pa, floor), "excluded": excl})
    return pd.DataFrame(rows, columns=[instance, "n_seeds", "p_arm", "p_ctrl", "d", "excluded"])


# ------------------------------------------------------------------------------------------------- Wilcoxon
def _has_ties(a: np.ndarray) -> bool:
    a = np.sort(np.abs(a))
    return bool(a.size > 1 and np.any(np.diff(a) == 0.0))


def wilcoxon_d(d, alpha: float = ALPHA) -> dict:
    """Two-sided Wilcoxon signed-rank test on d (zero_method wilcox). method = exact iff n_nonzero <= 50 and no ties
    in |d|, else asymptotic (normal, no continuity correction)."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    nz = d[d != 0.0]
    n, k = int(d.size), int(nz.size)
    if k == 0:
        return dict(n=n, n_nonzero=0, ties=False, method="none", statistic=math.nan, p=1.0)
    ties = _has_ties(nz)
    method = "exact" if (k <= EXACT_MAX_N and not ties) else "asymptotic"
    r = _st.wilcoxon(nz, zero_method="wilcox", correction=False, alternative="two-sided", method=method)
    return dict(n=n, n_nonzero=k, ties=ties, method=method, statistic=float(r.statistic), p=float(r.pvalue))


# ------------------------------------------------------------------------------------------------ bootstrap
def bootstrap_mean(d, n_boot: int = N_BOOT, seed: int = BOOT_SEED, level: float = 0.95) -> dict:
    """Mean of d with a percentile bootstrap interval (resampling instances with replacement). Deterministic."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(mean=math.nan, lo=math.nan, hi=math.nan, n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = 1000
    for s in range(0, n_boot, chunk):
        e = min(n_boot, s + chunk)
        idx = rng.integers(0, d.size, size=(e - s, d.size))
        means[s:e] = d[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    lo, hi = np.percentile(means, [100 * a, 100 * (1 - a)])
    return dict(mean=float(d.mean()), lo=float(lo), hi=float(hi), n_boot=n_boot, seed=seed)


# ----------------------------------------------------------------------------------------------- multiplicity
def holm(pvals, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down: (adjusted p-values, reject flags) in the input order."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    if m == 0:
        return p.copy(), np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="mergesort")
    adj_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj, adj <= alpha


def intersection_union(decisions: list[str]) -> str:
    """H1 = H1a AND H1b: supported only if every part is supported."""
    return SUPPORTED if decisions and all(x == SUPPORTED for x in decisions) else "not supported"


# --------------------------------------------------------------------------------------------- decision rule
def decide(p: float, lo: float, hi: float, alpha: float = ALPHA, margin: float = MARGIN) -> str:
    """D-002 §6, per comparison (d positive = arm better). Order: arm worse, supported, negative, inconclusive."""
    if any(map(lambda v: v is None or (isinstance(v, float) and math.isnan(v)), (p, lo, hi))):
        return INCONCLUSIVE
    if hi < 0:
        return ARM_WORSE
    if p < alpha and lo > 0:
        return SUPPORTED
    if hi < margin:
        return NEGATIVE
    return INCONCLUSIVE


def compare(df: pd.DataFrame, arm: str, ctrl: str, value: str = "pt", alpha: float = ALPHA,
            margin: float = MARGIN, n_boot: int = N_BOOT, seed: int = BOOT_SEED, **kw) -> dict:
    """One pre-registered comparison: effects, Wilcoxon, bootstrap, decision, ratio of means (secondary)."""
    eff = instance_effects(df, arm, ctrl, value=value, **kw)
    eff["excluded"] = eff["excluded"].astype(bool)     # an empty frame (an arm absent) has object dtype: `~` would
    used = eff[~eff["excluded"]]                      # turn the mask into a column selection
    w = wilcoxon_d(used["d"].to_numpy())
    b = bootstrap_mean(used["d"].to_numpy(), n_boot=n_boot, seed=seed)
    pc, pa = float(used["p_ctrl"].mean()) if len(used) else math.nan, \
        float(used["p_arm"].mean()) if len(used) else math.nan
    ratio_gain = (pc - pa) / pc if pc and not math.isnan(pc) else math.nan
    return dict(arm=arm, control=ctrl, value=value, n_instances=int(len(used)), n_excluded=int(eff["excluded"].sum()),
                excluded=eff.loc[eff["excluded"], eff.columns[0]].tolist(),
                better=int((used["d"] > 0).sum()), worse=int((used["d"] < 0).sum()), ties=int((used["d"] == 0).sum()),
                mean_d=b["mean"], ci_lo=b["lo"], ci_hi=b["hi"], n_boot=b["n_boot"], boot_seed=b["seed"],
                wilcoxon_stat=w["statistic"], p=w["p"], method=w["method"], n_nonzero=w["n_nonzero"],
                p_ties=w["ties"], mean_p_ctrl=pc, mean_p_arm=pa, ratio_of_means_gain=ratio_gain,
                decision=decide(w["p"], b["lo"], b["hi"], alpha, margin))


# ------------------------------------------------------------------------------------------ censored times
def censored_median(times, events) -> tuple[float, bool]:
    """Median of a few right-censored times (Type I at a common T: censored values sit at T, the maximum).
    The median is the usual sample median of the observed values; it is censored if any value entering it is
    censored — then it is a lower bound (with 2 seeds, one censored: (t + T)/2, censored)."""
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(events, dtype=bool)
    if t.size == 0:
        return math.nan, False
    o = np.lexsort((~e, t))            # ties: events before censored values
    t, e = t[o], e[o]
    n = t.size
    if n % 2:
        k = n // 2
        return float(t[k]), bool(e[k])
    k = n // 2
    return float((t[k - 1] + t[k]) / 2.0), bool(e[k - 1] and e[k])


def prentice_scores(times, events) -> np.ndarray:
    """Prentice–Wilcoxon scores on a pooled sample (O'Brien & Fleming 1987): event 1 − 2·S̃(t), censored 1 − S̃(t),
    S̃(t) = Π_{t_j ≤ t} (n_j + 1 − d_j)/(n_j + 1) over distinct event times; censored values at an event time are
    still at risk there."""
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(events, dtype=bool)
    ev_times = np.unique(t[e])
    S_at = {}
    s = 1.0
    for tj in ev_times:
        n_j = int(np.sum(t >= tj))
        d_j = int(np.sum((t == tj) & e))
        s *= (n_j + 1.0 - d_j) / (n_j + 1.0)
        S_at[tj] = s
    out = np.empty(t.size)
    for i in range(t.size):
        prior = [S_at[tj] for tj in ev_times if tj <= t[i]]
        st = prior[-1] if prior else 1.0
        out[i] = 1.0 - 2.0 * st if e[i] else 1.0 - st
    return out


def paired_prentice_wilcoxon(t_arm, e_arm, t_ctrl, e_ctrl) -> dict:
    """Paired Prentice–Wilcoxon test (O'Brien & Fleming 1987) on per-instance censored times, two-sided.
    z > 0 means the ARM is faster (sign flipped from the score difference so it reads like d)."""
    t1, e1 = np.asarray(t_arm, float), np.asarray(e_arm, bool)
    t2, e2 = np.asarray(t_ctrl, float), np.asarray(e_ctrl, bool)
    n = t1.size
    if n == 0 or t2.size != n:
        raise ValueError("paired_prentice_wilcoxon: arm and control need the same, non-zero length")
    sc = prentice_scores(np.concatenate([t1, t2]), np.concatenate([e1, e2]))
    s = sc[:n] - sc[n:]                # positive: the arm's time ranks later (slower)
    ss = float(np.sum(s * s))
    if ss == 0.0:
        return dict(n=n, z=0.0, p=1.0, mean_score_diff=0.0)
    z = -float(np.sum(s)) / math.sqrt(ss)
    return dict(n=n, z=z, p=float(2.0 * _st.norm.sf(abs(z))), mean_score_diff=float(np.mean(s)))


def censored_sign_test(t_arm, e_arm, t_ctrl, e_ctrl) -> dict:
    """Sign test on paired censored times. A pair counts only when its order is certain: both events (by time), or an
    event before the other's censoring time; both censored, equal times, or an event after the other's censoring
    point are indeterminate and dropped."""
    better = worse = 0
    for a, ea, c, ec in zip(t_arm, e_arm, t_ctrl, e_ctrl):
        if ea and ec:
            better += a < c
            worse += a > c
        elif ea and not ec:
            better += a < c
        elif ec and not ea:
            worse += c < a
    n = better + worse
    p = float(_st.binomtest(better, n, 0.5).pvalue) if n else 1.0
    return dict(n=n, arm_faster=better, arm_slower=worse, p=p)


def compare_censored(df: pd.DataFrame, arm: str, ctrl: str, time: str, event: str, instance: str = "instance",
                     seed: str = "seed", arm_col: str = "arm") -> dict:
    """Seed medians per instance on the common seeds, then the paired Prentice–Wilcoxon test and the sign test."""
    sub = df[df[arm_col].isin([arm, ctrl])]
    rows = []
    for inst, g in sub.groupby(instance, sort=True):
        ga, gc = g[g[arm_col] == arm], g[g[arm_col] == ctrl]
        common = sorted(set(ga[seed]) & set(gc[seed]))
        if not common:
            continue
        ga, gc = ga[ga[seed].isin(common)], gc[gc[seed].isin(common)]
        ta, ea = censored_median(ga[time], ga[event])
        tc, ec = censored_median(gc[time], gc[event])
        rows.append((inst, ta, ea, tc, ec))
    if not rows:
        return dict(arm=arm, control=ctrl, time=time, n_instances=0, ppw=None, sign=None)
    _, ta, ea, tc, ec = map(list, zip(*rows))
    return dict(arm=arm, control=ctrl, time=time, n_instances=len(rows),
                ppw=paired_prentice_wilcoxon(ta, ea, tc, ec), sign=censored_sign_test(ta, ea, tc, ec),
                instances=[r[0] for r in rows])


# ------------------------------------------------------------------------------------------ Mantel–Haenszel
@dataclass
class Table2x2:
    a: int    # exposed cases
    b: int    # exposed references
    c: int    # unexposed cases
    d: int    # unexposed references

    @property
    def n(self) -> int:
        return self.a + self.b + self.c + self.d


def mantel_haenszel(tables, level: float = 0.95) -> dict:
    """Pooled odds ratio over strata with the Robins–Breslow–Greenland confidence interval. Strata with n = 0 are
    skipped; strata without a case or without a reference contribute 0 to both sums."""
    tabs = [t if isinstance(t, Table2x2) else Table2x2(*map(int, t)) for t in tables]
    tabs = [t for t in tabs if t.n > 0]
    R = S = 0.0
    PR = PS_QR = QS = 0.0
    for t in tabs:
        n = float(t.n)
        r, s = t.a * t.d / n, t.b * t.c / n
        p, q = (t.a + t.d) / n, (t.b + t.c) / n
        R += r
        S += s
        PR += p * r
        PS_QR += p * s + q * r
        QS += q * s
    out = dict(n_strata=len(tabs), n_strata_informative=int(sum(1 for t in tabs if (t.a + t.c) and (t.b + t.d))),
               cases_exposed=int(sum(t.a for t in tabs)), cases_unexposed=int(sum(t.c for t in tabs)),
               refs_exposed=int(sum(t.b for t in tabs)), refs_unexposed=int(sum(t.d for t in tabs)))
    if R == 0 or S == 0:
        out.update(or_mh=(math.inf if R > 0 else (0.0 if S > 0 else math.nan)), ci_lo=math.nan, ci_hi=math.nan,
                   se_log=math.nan)
        return out
    or_mh = R / S
    var = PR / (2 * R * R) + PS_QR / (2 * R * S) + QS / (2 * S * S)
    se = math.sqrt(var)
    zc = float(_st.norm.ppf(1 - (1 - level) / 2))
    out.update(or_mh=or_mh, se_log=se, ci_lo=math.exp(math.log(or_mh) - zc * se),
               ci_hi=math.exp(math.log(or_mh) + zc * se))
    return out
