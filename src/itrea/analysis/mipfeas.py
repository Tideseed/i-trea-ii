"""`itrea mipfeas-report`: the paired calibration of the MIPfeas harness against GAMS's published cuOpt 26.08 run on a
GB10 (vault D-020), by the rules D-020 fixed before any calibration run — R, the feasible counts and the verdict —,
and since D-021 (ITX-1) the same report for **A6F** (cuOpt alone at A9's footprint F, the calibration that replaced
A6m's: `--arm A6F`) with D-021's late-clock check and reading. Development only (D-010); the clock and the metric of
the v1 runs are fixed in prereg-v2, not here.

Inputs: `datasets/runs.parquet` (`itrea build-runs`: `pt_gams` on the call clock, `pt_gams_launch` on the launch
clock, `startup_offset_s`, `t_first` / `t_first_event` — the portfolio's first checked incumbent on the call clock —
and `ref`) and GAMS's archive `data/mipfeas/gams-2026-09-16/` (Berk's yes, 2026-10-11; gitignored like all of data/):
`primalintegral.csv` (their per-instance integral; column `cuopt-gb10` = cuOpt 26.08 on the GB10),
`optimal_objective.csv` (their z*) and the traces `trc_files/<name>_<machine>_<solver>.trc`.

Runs: the arm (A6m) and set (mipfeas), seed 0, budget 600, status completed, not synthetic, not block-void, and
**intact** — no incumbent file missing (`incumbents_missing`), none unchecked (`n_inc_unchecked`), the stored count
equal to the trace's (`inc_count_mismatch`), none stamped before t0 (`pre_t0`, `n_inc_bad_t`): a run without them
would be scored as if its incumbents never existed (a missing incumbent file gives P = 2). Incumbents that failed the
check (`n_inc_failed`) or whose reported objective differs from the recomputed one (`n_obj_mismatch`) keep the run —
only checked incumbents count anyway (D-002) — and are printed as flags. An instance with two or more such runs (a
rerun under another configuration hash, say) is refused, never resolved by picking one. An instance without exactly
one valid run is listed with every run it has and why each was not taken; the verdict needs all 30.

Statistics (D-020 § Calibration):
- per instance P_ours on both clocks and P_GAMS (`primalintegral.csv`, `cuopt-gb10`);
- R = exp(mean_i log((P_ours,i + 0.001)/(P_GAMS,i + 0.001))) = (SGM_ours + 0.001)/(SGM_GAMS + 0.001), where SGM is
  the shifted geometric mean exp(mean log(P + 0.001)) − 0.001 on the sample — the ratio of the two geometric means of
  P + 0.001, not SGM_ours / SGM_GAMS — with a 95 % percentile bootstrap interval over instances
  (`stats.bootstrap_mean` on the log ratios: 10,000 resamples, seed `stats.BOOT_SEED`); on the call clock (the pass
  rule) and the launch clock (reported);
- both shifted geometric means (s = 0.001), the feasible counts (P < 2), R per stratum (D-020's six strata);
- first incumbent: ours (call clock; launch clock = + `startup_offset_s`) minus the first `I` record of GAMS's GB10
  26.08 trace — median and quartiles over the instances where both found one;
- GAMS's own GB10 cuOpt 26.04 and B200 cuOpt 26.08 traces against GB10 26.08 on the sample, scored with the port
  (`metrics.gams_integral`, their z*): the size of a version or machine difference, for scale;
- the reference check: max |ref − z*| / max(1, |z*|) over the sample (D-020 found ≤ 1e-6 on all 233), since our
  `pt_gams` is scored against our reference; and the instances where a checked incumbent beat that reference
  (`beats_ref`: γ = 0 in the port, where GAMS's script raises; supportcase22's reference is only the best known, 110).
Late-clock check (D-021 § Calibration, reported for every arm): the median first-incumbent offset ours − GAMS on the
call clock over the instances of strata 0–1 where both found one; it passes when that one median exists and is at most
+0.3 s (one-sided: a late clock is the fault it looks for); the median of every stratum is reported beside (stratum 2
and the others are thread-sensitive and only reported).
Calibration settings (D-021 reading only): every chosen run must carry the ext harness exactly (MIP gaps 0, thread
variables unset, RMM 24 / host 24 GB, the CUDA cache, cap hits voiding) and cuOpt's thread count |F| (and, when the
run table records it, F's cores), else it is listed under missing with the reason; the configuration hash(es) of the
chosen runs are printed.
Verdict for A6m (D-020's rule): PASS iff every instance of the sample has a run, 0.870 ≤ R ≤ 1.150 on the call clock
and our feasible count is within ±1 of GAMS's on the sample (27 for D-020's sample, computed here); FAIL otherwise;
INCOMPLETE (no verdict) while an instance is missing.
Verdict for every other arm (A6F, D-021's reading): INCOMPLETE while an instance is missing; CLOCK-FAIL when the
late-clock check fails (the calibration's reading stops: the barrier and the clock are diagnosed before any v1 run);
else INSIDE (R within the band and feasible within ±1: no footprint or harness difference larger than about 30 % is
likely — the sample's interval is ×/÷ 1.28, 8–14 % differences are not resolved — and the published row may be cited
as a bridge) or OUTSIDE (with the clock check passed the difference is the footprint, and the machine, stated as such;
only then could the heat ladder, R9 with Berk's yes, be asked for). v1's targets are anchored on our own A6F in either
case.

The module also reads the archive for the rest of the code: `read_gams_trc` (a trace, with their exact `read_csv`),
`read_z`, `read_pi`, and D-020's sample (`STRATA`, re-derived by a test). The port scored on `read_gams_trc` +
`metrics.best_steps` equals GAMS's `calculate_primal_integral` (their `create_primalintegral.py`, in the archive) on
every one of the archive's 3,495 traces (2026-10-11; the 233 GB10 cuOpt 26.08 ones are a test).

    itrea mipfeas-report [--arm A6m | A6F] [--set mipfeas] [--gams-dir data/mipfeas/gams-2026-09-16]
                         [--runs datasets/runs.parquet] [--json out.json] [--md out.md]
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
from itrea.paths import DATA, DATASETS

GAMS_DIR = DATA / "mipfeas" / "gams-2026-09-16"
GAMS_COL = "cuopt-gb10"                    # primalintegral.csv: cuOpt 26.08 on the GB10 (their published row)
GB10_2608 = "gb10_cuopt-26-08"             # the same run's traces
OTHER_RUNS = (("GB10 cuOpt 26.04", "gb10_cuopt-26-04"), ("B200 cuOpt 26.08", "b200_cuopt-26-08"))
TRC_COLUMNS = ["lineNum", "seriesID", "node", "seconds", "bestFound", "bestBound"]
ARM, SET, SEED, BUDGET = "A6m", "mipfeas", 0, 600.0
SHIFT = M.MIPFEAS_SHIFT
R_LO, R_HI = 0.870, 1.150                  # D-019's ±15 %, paired (D-020)
FEASIBLE_TOL = 1                           # our feasible count within ±1 of GAMS's
PASS, FAIL, INCOMPLETE = "PASS", "FAIL", "INCOMPLETE"
INSIDE, OUTSIDE, CLOCK_FAIL = "INSIDE", "OUTSIDE", "CLOCK-FAIL"     # D-021's reading (A6F and any arm but A6m)
D020_ARMS = ("A6m",)                       # arms read by D-020's PASS / FAIL rule
LATE_CLOCK_STRATA, LATE_CLOCK_MAX_S = (0, 1), 0.3   # D-021: median first incumbent ours − GAMS <= +0.3 s there
# D-020's sample: the 233 sorted by GAMS's GB10 26.08 P (stable sort), numpy array_split into six strata, five per
# stratum by numpy default_rng(20261011).choice(stratum, 5, replace=False), each pick sorted (tests re-derive it)
SAMPLE_SEED = 20261011
STRATA: tuple[tuple[str, ...], ...] = (
    ("assign1-5-8", "ic97_potential", "neos-1171448", "nu25-pr12", "p200x1188c"),
    ("mas74", "neos-4338804-snowy", "proteindesign122trx11p8", "seymour", "swath3"),
    ("CMS750_4", "chromaticindex512-7", "milo-v12-6-r2-40-1", "neos-1122047", "neos-5093327-huahum"),
    ("blp-ar98", "csched008", "dano3_3", "map10", "neos-4387871-tavua"),
    ("irish-electricity", "neos-4300652-rahue", "neos-5107597-kakapo", "neos-933966", "sing44"),
    ("cryptanalysiskb128n5obj16", "neos-2746589-doon", "neos-3216931-puriri", "satellites2-60-fs", "supportcase22"),
)
SAMPLE: tuple[str, ...] = tuple(n for s in STRATA for n in s)


def stratum_of(name: str) -> int:
    for k, s in enumerate(STRATA):
        if name in s:
            return k
    raise KeyError(name)


# ------------------------------------------------------------------------------------------------ GAMS's archive
def read_gams_trc(path: str | Path) -> pd.DataFrame:
    """A GAMS MIP trace (`.trc`): one row per record — lineNum, seriesID (S start, I incumbent, E end), node, seconds,
    bestFound, bestBound — read with the exact `pandas.read_csv` call of GAMS's `calculate_primal_integral`, so the
    floats are the ones their script scored ('na' and 'inf' are NaN)."""
    return pd.read_csv(path, comment="*", header=None, na_values=["na", "inf"], skipinitialspace=True,
                       names=TRC_COLUMNS)


def trc_points(trc: pd.DataFrame) -> list[tuple[float, float]]:
    """(seconds, bestFound) of every record that carries an objective, whatever its series: GAMS's integral counts the
    E record's objective too (an incumbent reported only at the end)."""
    g = trc[trc["bestFound"].notna()]
    return [(float(t), float(v)) for t, v in zip(g["seconds"], g["bestFound"])]


def trc_first_incumbent(trc: pd.DataFrame) -> float | None:
    """Seconds of the first `I` record with an objective (None if the run found none)."""
    g = trc[(trc["seriesID"].astype(str).str.strip() == "I") & trc["bestFound"].notna()]
    return float(g["seconds"].iloc[0]) if len(g) else None


def trc_path(gams_dir: str | Path, name: str, run: str = GB10_2608) -> Path:
    return Path(gams_dir) / "trc_files" / f"{name}_{run}.trc"


def trc_integral(path: str | Path, z: float, sense: int = 1, T: float = M.GAMS_T) -> M.GamsIntegral:
    """GAMS's integral of one trace, by the port (`metrics.gams_integral` on the trace's best-so-far steps)."""
    return M.gams_integral(M.best_steps(trc_points(read_gams_trc(path)), sense), z, sense, T)


def read_z(gams_dir: str | Path = GAMS_DIR) -> dict[str, float]:
    """GAMS's reference values z* (`optimal_objective.csv`: instance -> objective; its 7 'Infeasible' rows, the
    instances outside the 233, are left out)."""
    df = pd.read_csv(Path(gams_dir) / "optimal_objective.csv")
    out = {}
    for n, x in zip(df["InstanceInst."], df["ObjectiveObje."]):
        try:
            out[str(n)] = float(x)          # Python's float of the cell, as their script's float(z_map[name])
        except ValueError:
            continue
    return out


def read_pi(gams_dir: str | Path = GAMS_DIR) -> pd.DataFrame:
    """GAMS's per-instance integrals (`primalintegral.csv`), indexed by instance name."""
    return pd.read_csv(Path(gams_dir) / "primalintegral.csv").set_index("mps_name")


# -------------------------------------------------------------------------------------------------- statistics
def log_ratios(p_ours, p_ref, shift: float = SHIFT) -> np.ndarray:
    return np.log((np.asarray(p_ours, dtype=float) + shift) / (np.asarray(p_ref, dtype=float) + shift))


def ratio(p_ours, p_ref, shift: float = SHIFT, boot: bool = True) -> dict:
    """R = exp(mean log ratio) with its 95 % percentile bootstrap interval (resampling instances)."""
    lr = log_ratios(p_ours, p_ref, shift)
    lr = lr[np.isfinite(lr)]
    if lr.size == 0:
        return dict(n=0, R=math.nan, lo=math.nan, hi=math.nan, sd_log=math.nan)
    out = dict(n=int(lr.size), R=float(np.exp(lr.mean())), sd_log=float(lr.std(ddof=1)) if lr.size > 1 else math.nan)
    if boot:
        b = S.bootstrap_mean(lr)
        out.update(lo=float(np.exp(b["lo"])), hi=float(np.exp(b["hi"])), n_boot=b["n_boot"], boot_seed=b["seed"])
    return out


def _quartiles(xs) -> dict:
    x = np.asarray([v for v in xs if v is not None and math.isfinite(v)], dtype=float)
    if x.size == 0:
        return dict(n=0, median=math.nan, q25=math.nan, q75=math.nan)
    q25, med, q75 = np.percentile(x, [25, 50, 75])
    return dict(n=int(x.size), median=float(med), q25=float(q25), q75=float(q75))


INTEGRITY_COLUMNS = ("incumbents_missing", "n_inc_unchecked", "inc_count_mismatch", "pre_t0", "n_inc_bad_t",
                     "n_inc_failed", "n_obj_mismatch")


def integrity_faults(row) -> list[str]:
    """Why a run is not a valid calibration run (module docstring); [] if it is."""
    out = []
    if str(row["incumbents_missing"] or ""):
        out.append(f"incumbent file missing for {row['incumbents_missing']}")
    if int(row["n_inc_unchecked"]) > 0:
        out.append(f"{int(row['n_inc_unchecked'])} incumbents unchecked")
    if str(row["inc_count_mismatch"] or ""):
        out.append(f"stored incumbents != trace incumbents for {row['inc_count_mismatch']}")
    if bool(row["pre_t0"]) or int(row["n_inc_bad_t"]) > 0:
        out.append(f"{int(row['n_inc_bad_t'])} incumbents stamped before t0 or without a time")
    return out


def integrity_flags(row) -> list[str]:
    """What a valid run carries but does not lose its validity for (only checked incumbents count, D-002)."""
    out = []
    if int(row["n_inc_failed"]) > 0:
        out.append(f"{int(row['n_inc_failed'])} incumbents failed the check")
    if int(row["n_obj_mismatch"]) > 0:
        out.append(f"{int(row['n_obj_mismatch'])} reported objectives differ from the recomputed ones")
    return out


def calibration_faults(row) -> list[str]:
    """Why a run is not D-021's calibration configuration (A6F at F under the ext harness): its harness differs from
    the ext preset, cuOpt's thread count is not |F|, or (when recorded) F's cores differ. [] if it is."""
    from itrea.experiments.pair import GPU_LANE, resolve_harness
    want = resolve_harness("ext")
    try:
        got = json.loads(row["harness"]) if str(row.get("harness") or "") else None
    except (TypeError, ValueError):
        got = None
    try:
        ss = json.loads(row["solver_settings"]) if str(row.get("solver_settings") or "") else {}
    except (TypeError, ValueError):
        ss = {}
    out = []
    if got != want:
        out.append(f"harness {json.dumps(got, sort_keys=True) if got else 'none'} is not the ext preset")
    nf = len(GPU_LANE.x925) + len(GPU_LANE.a725)
    if ss.get("cuopt_threads") != nf:
        out.append(f"cuOpt threads {ss.get('cuopt_threads')} != |F| = {nf}")
    cores = (ss.get("footprint") or {}).get("cores")
    want_cores = dict(x925=sorted(GPU_LANE.x925), a725=sorted(GPU_LANE.a725))
    if cores is not None and cores != want_cores:
        out.append(f"cores {cores} are not F {want_cores}")
    return out


def select_runs(runs: pd.DataFrame, arm: str = ARM, set_: str = SET, seed: int = SEED,
                budget: float = BUDGET, settings=None) -> tuple[dict[str, pd.Series], dict[str, list[str]]]:
    """The calibration's run per instance (module docstring) and, per instance, why every other run of the arm in the
    set was not taken (status, seed, budget, void, integrity) — or, for an instance with several valid runs, that it
    was refused. The run table needs build_runs' integrity columns (INTEGRITY_COLUMNS); without them nothing can be
    shown intact, so it raises."""
    absent = [c for c in INTEGRITY_COLUMNS if c not in runs.columns]
    if absent:
        raise ValueError(f"the run table lacks the integrity columns {absent} (rebuild it with `itrea build-runs`)")
    r = runs[(runs["arm"].astype(str) == arm) & (runs["set"].astype(str) == set_)]
    if "synthetic" in r.columns:
        r = r[~r["synthetic"].astype(bool)]
    void = r["block_void"].astype(bool) if "block_void" in r.columns else pd.Series(False, index=r.index)
    ok = (r["status"].astype(str) == "completed") & ~void & (r["seed"].astype(int) == int(seed)) & \
        np.isclose(r["budget"].astype(float), float(budget))
    why_not: dict[str, list[str]] = {}
    valid: dict[str, list[pd.Series]] = {}
    for idx, row in r.sort_values("run_id", kind="mergesort").iterrows():
        n = str(row["instance"])
        if not ok.loc[idx]:
            why_not.setdefault(n, []).append(f"{row['run_id']}: status {row.get('status') or '?'}, seed {row['seed']}, "
                                             f"budget {row['budget']:g}" + (", block void" if void.loc[idx] else ""))
            continue
        faults = integrity_faults(row)
        wrong = settings(row) if settings is not None else []
        if faults:
            why_not.setdefault(n, []).append(f"{row['run_id']}: not intact ({'; '.join(faults)})")
        elif wrong:
            why_not.setdefault(n, []).append(f"{row['run_id']}: not D-021's calibration settings ({'; '.join(wrong)})")
        else:
            valid.setdefault(n, []).append(row)
    chosen = {}
    for n, rows in valid.items():
        if len(rows) == 1:
            chosen[n] = rows[0]
        else:
            why_not.setdefault(n, []).insert(0, f"{len(rows)} valid runs ({', '.join(str(x['run_id']) for x in rows)}): "
                                                "refused, never one picked")
    return chosen, why_not


def late_clock(have: pd.DataFrame, checked=LATE_CLOCK_STRATA, limit_s: float = LATE_CLOCK_MAX_S) -> dict:
    """D-021's late-clock check on the valid runs (`t_first`, `t_first_gams`, `stratum`): the median of ours − GAMS
    over the instances of strata 0–1 (`checked`) where both found a first incumbent DECIDES — `ok` iff there is at
    least one and that median is at most +limit_s (one median, as D-021 states it); the per-stratum quartiles of every
    stratum are reported beside (`checked` marks the strata the median pools; the others are thread-sensitive and only
    reported)."""
    both = have[have["t_first"].notna() & have["t_first_gams"].notna()]
    d = both["t_first"] - both["t_first_gams"]
    strata = sorted({int(k) for k in have["stratum"]} | set(checked))
    per = [dict(stratum=k, checked=k in checked, **_quartiles(d[both["stratum"] == k])) for k in strata]
    pooled = _quartiles(d[both["stratum"].isin(list(checked))])
    ok = pooled["n"] > 0 and pooled["median"] <= limit_s
    return dict(limit_s=limit_s, checked=list(checked), strata=per, pooled=pooled, ok=bool(ok),
                rule=f"median first incumbent ours − GAMS over strata {'–'.join(map(str, checked))} <= +{limit_s:g} s "
                     f"(call clock; per stratum reported)")


def report(runs: pd.DataFrame, gams_dir: str | Path = GAMS_DIR, arm: str = ARM, set_: str = SET, seed: int = SEED,
           budget: float = BUDGET, strata=STRATA) -> dict:
    """The calibration's numbers and verdict (module docstring)."""
    gams_dir = Path(gams_dir)
    sample = [n for s in strata for n in s]
    pi, z = read_pi(gams_dir), read_z(gams_dir)
    absent = [n for n in sample if n not in pi.index or n not in z]
    if absent:
        raise ValueError(f"{gams_dir}: no published P or z* for {absent}")
    chosen, others = select_runs(runs, arm, set_, seed, budget,
                                 settings=None if arm in D020_ARMS else calibration_faults)
    rows, missing = [], []
    for k, s in enumerate(strata):
        for n in s:
            gtrc = read_gams_trc(trc_path(gams_dir, n))
            row = dict(stratum=k, instance=n, p_gams=float(pi.loc[n, GAMS_COL]), z=z[n],
                       t_first_gams=trc_first_incumbent(gtrc))
            r = chosen.get(n)
            pc = float(r["pt_gams"]) if r is not None and "pt_gams" in r else math.nan
            if r is None or not math.isfinite(pc):
                why = "; ".join(others.get(n, [])) or "no run" if r is None else \
                    f"{r['run_id']}: no pt_gams (no reference, or a run table built before D-020)"
                missing.append(dict(instance=n, stratum=k, why=why))
                rows.append(dict(row, run_id=None, p_call=math.nan, p_launch=math.nan, t_first=math.nan,
                                 startup_offset_s=math.nan, ref=math.nan, ref_rel_diff=math.nan, flags="",
                                 beats_ref=False, config_hash="", run_hash=""))
                continue
            off = float(r.get("startup_offset_s", math.nan))
            t_first = float(r["t_first"]) if bool(r.get("t_first_event", False)) else math.nan
            ref = float(r.get("ref", math.nan))
            rows.append(dict(row, run_id=str(r["run_id"]), p_call=pc, p_launch=float(r.get("pt_gams_launch", math.nan)),
                             t_first=t_first, startup_offset_s=off, ref=ref,
                             ref_rel_diff=abs(ref - z[n]) / max(1.0, abs(z[n])) if math.isfinite(ref) else math.nan,
                             flags="; ".join(integrity_flags(r)), beats_ref=bool(r.get("beats_ref", False)),
                             config_hash=str(r.get("config_hash", "") or ""), run_hash=str(r.get("run_hash", "") or "")))
    df = pd.DataFrame(rows)
    have = df[df["run_id"].notna()]
    feas = lambda p: int((np.asarray(p, dtype=float) < M.MIPFEAS_NONE).sum())      # noqa: E731
    out: dict = dict(arm=arm, set=set_, seed=seed, budget=budget, gams_dir=str(gams_dir), n_sample=len(sample),
                     n_runs=int(len(have)), missing=missing, instances=df.to_dict("records"))
    out["call"] = ratio(have["p_call"], have["p_gams"])
    out["launch"] = ratio(have["p_launch"], have["p_gams"])
    out["sgm"] = dict(ours_call=M.shifted_geomean(have["p_call"], shift=SHIFT),
                      ours_launch=M.shifted_geomean(have["p_launch"], shift=SHIFT),
                      gams_on_runs=M.shifted_geomean(have["p_gams"], shift=SHIFT),
                      gams_sample=M.shifted_geomean(df["p_gams"], shift=SHIFT))
    out["feasible"] = dict(ours=feas(have["p_call"]), gams_on_runs=feas(have["p_gams"]),
                           gams_sample=feas(df["p_gams"]))
    out["strata"] = [dict(stratum=k, **ratio(g["p_call"], g["p_gams"], boot=False),
                          launch_R=ratio(g["p_launch"], g["p_gams"], boot=False)["R"])
                     for k, g in have.groupby("stratum", sort=True)]
    both = have[have["t_first"].notna() & have["t_first_gams"].notna()]
    out["first_incumbent"] = dict(
        call=_quartiles(both["t_first"] - both["t_first_gams"]),
        launch=_quartiles(both["t_first"] + both["startup_offset_s"] - both["t_first_gams"]),
        only_ours=int((have["t_first"].notna() & have["t_first_gams"].isna()).sum()),
        only_gams=int((have["t_first"].isna() & have["t_first_gams"].notna()).sum()),
        startup_offset_s=_quartiles(have["startup_offset_s"]))
    others_rows = {}
    for label, run in OTHER_RUNS:
        p = [trc_integral(trc_path(gams_dir, n, run), z[n]).value for n in df["instance"]]
        others_rows[run] = dict(label=label, **ratio(p, df["p_gams"]), feasible=feas(p),
                                sgm=M.shifted_geomean(p, shift=SHIFT))
    out["gams_rows"] = others_rows
    out["ref_rel_diff_max"] = float(have["ref_rel_diff"].max()) if len(have) else math.nan
    out["flags"] = {str(x["instance"]): x["flags"] for _, x in have.iterrows() if x["flags"]}
    out["beats_ref"] = [dict(instance=str(x["instance"]), ref=float(x["ref"]), z=float(x["z"]))
                        for _, x in have.iterrows() if bool(x["beats_ref"])]
    lc = out["late_clock"] = late_clock(have)
    d020 = arm in D020_ARMS
    hcol = "run_hash" if len(have) and have["run_hash"].astype(str).ne("").all() else "config_hash"
    out["config_hashes"] = sorted({str(x) for x in have[hcol] if str(x)}) if len(have) else []
    out["rule"] = dict(R_lo=R_LO, R_hi=R_HI, feasible_tol=FEASIBLE_TOL, clock="call", decision="D-020" if d020
                       else "D-021", late_clock=lc["rule"])
    if missing:
        out["verdict"], out["verdict_reason"] = INCOMPLETE, (f"{len(missing)} of {len(sample)} instances have no "
                                                             f"valid run")
        return out
    R = out["call"]["R"]
    r_ok = R_LO <= R <= R_HI
    f_ok = abs(out["feasible"]["ours"] - out["feasible"]["gams_sample"]) <= FEASIBLE_TOL
    band = (f"R {R:.3f} {'within' if r_ok else 'outside'} [{R_LO:.3f}, {R_HI:.3f}]; feasible "
            f"{out['feasible']['ours']} vs GAMS {out['feasible']['gams_sample']} "
            f"({'within' if f_ok else 'outside'} ±{FEASIBLE_TOL})")
    if d020:
        out["verdict"], out["verdict_reason"] = (PASS if (r_ok and f_ok) else FAIL), band
    elif not lc["ok"]:
        pq = lc["pooled"]
        bad = ("no paired first incumbent in strata " + "–".join(map(str, lc["checked"])) if not pq["n"] else
               f"median {pq['median']:+.3f} s over strata {'–'.join(map(str, lc['checked']))}, n {pq['n']}")
        out["verdict"] = CLOCK_FAIL
        out["verdict_reason"] = (f"late-clock check failed ({bad}; the limit is +{lc['limit_s']:g} s): the "
                                 f"calibration's reading stops — the barrier and the clock are diagnosed before any v1 "
                                 f"run (D-021); {band}")
    elif r_ok and f_ok:
        out["verdict"] = INSIDE
        out["verdict_reason"] = (f"{band}; late-clock check passed: no footprint or harness difference larger than about "
                                 f"30 % is likely (the sample's interval is ×/÷ 1.28; 8–14 % differences are not "
                                 f"resolved), and the published row may be cited as a bridge; v1's targets are anchored "
                                 f"on our own {arm} (D-021)")
    else:
        out["verdict"] = OUTSIDE
        out["verdict_reason"] = (f"{band}; late-clock check passed: the difference is the footprint (and the machine), "
                                 f"stated as such; only now could the heat ladder (R9, Berk's yes) be asked for; v1's "
                                 f"targets are anchored on our own {arm} (D-021)")
    return out


# ----------------------------------------------------------------------------------------------------- output
def _f(x, nd=4) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def markdown(res: dict) -> str:
    c, la, sg, fe = res["call"], res["launch"], res["sgm"], res["feasible"]
    which = "D-020" if res.get("rule", {}).get("decision", "D-020") == "D-020" else "D-021"
    lines = [f"<!-- generated by itrea.analysis.mipfeas from datasets/runs.parquet and {res['gams_dir']} -->",
             f"MIPfeas calibration ({which}): {res['arm']} in set {res['set']}, seed {res['seed']}, "
             f"{res['budget']:g} s, {res['n_runs']} of {res['n_sample']} instances",
             "",
             "| stratum | instance | P ours (call) | P ours (launch) | P GAMS | log ratio (call) | first inc. ours | "
             "first inc. GAMS |",
             "|---|---|---|---|---|---|---|---|"]
    for r in res["instances"]:
        lr = log_ratios([r["p_call"]], [r["p_gams"]])[0]
        lines.append(f"| {r['stratum']} | {r['instance']} | {_f(r['p_call'])} | {_f(r['p_launch'])} | "
                     f"{_f(r['p_gams'])} | {_f(float(lr), 3)} | {_f(r['t_first'], 3)} | {_f(r['t_first_gams'], 3)} |")
    lines += ["",
              f"R = (SGM_ours + 0.001) / (SGM_GAMS + 0.001) = exp(mean log((P_ours + 0.001) / (P_GAMS + 0.001))): call "
              f"clock {_f(c['R'], 3)} [{_f(c.get('lo'), 3)}, {_f(c.get('hi'), 3)}] over {c['n']} instances; launch "
              f"clock {_f(la['R'], 3)} [{_f(la.get('lo'), 3)}, {_f(la.get('hi'), 3)}] (95 % bootstrap intervals).",
              f"Shifted geometric means (s = 0.001): ours {_f(sg['ours_call'])} (call) / {_f(sg['ours_launch'])} "
              f"(launch); GAMS {_f(sg['gams_on_runs'])} on the same instances, {_f(sg['gams_sample'])} on the "
              f"whole sample.",
              f"Feasible (P < 2): ours {fe['ours']}, GAMS {fe['gams_on_runs']} on the same instances, "
              f"{fe['gams_sample']} on the whole sample.",
              "R per stratum (call / launch): " + "; ".join(f"{s['stratum']}: {_f(s['R'], 3)} / {_f(s['launch_R'], 3)}"
                                                           f" (n {s['n']})" for s in res["strata"]) + "."]
    fi = res["first_incumbent"]
    lines.append(f"First incumbent, ours − GAMS (s): call clock median {_f(fi['call']['median'], 3)} "
                 f"[{_f(fi['call']['q25'], 3)}, {_f(fi['call']['q75'], 3)}], launch clock median "
                 f"{_f(fi['launch']['median'], 3)} [{_f(fi['launch']['q25'], 3)}, {_f(fi['launch']['q75'], 3)}] "
                 f"over {fi['call']['n']} instances (only ours {fi['only_ours']}, only GAMS {fi['only_gams']}); "
                 f"start-up offset median {_f(fi['startup_offset_s']['median'], 3)} s.")
    lc = res.get("late_clock")
    if lc:
        per = "; ".join(f"{p['stratum']}{'*' if p['checked'] else ''}: {_f(p['median'], 3)} (n {p['n']})"
                        for p in lc["strata"])
        lines.append(f"Late-clock check (D-021: {lc['rule']}): median ours − GAMS over the strata marked * "
                     f"{_f(lc['pooled']['median'], 3)} s (n {lc['pooled']['n']}) — {'passed' if lc['ok'] else 'FAILED'};"
                     f" per stratum (s) {per}.")
    if res.get("config_hashes"):
        hs = res["config_hashes"]
        lines.append("Configuration hash of the chosen runs: " + (f"{hs[0]} (all {res['n_runs']})" if len(hs) == 1
                                                                 else ", ".join(hs) + " (NOT one configuration)") + ".")
    for run, g in res["gams_rows"].items():
        lines.append(f"GAMS {g['label']} vs GB10 26.08 on the sample (port): R {_f(g['R'], 3)} "
                     f"[{_f(g.get('lo'), 3)}, {_f(g.get('hi'), 3)}], feasible {g['feasible']}, "
                     f"geo-mean {_f(g['sgm'])}.")
    lines.append(f"Reference check: max |ref − z*| / max(1, |z*|) = {res['ref_rel_diff_max']:.2e}.")
    if res["beats_ref"]:
        lines.append("Beats the reference (a checked incumbent better than z*: γ = 0 here, GAMS's script would raise): "
                     + "; ".join(f"{b['instance']} (ref {b['ref']:g}, z* {b['z']:g})" for b in res["beats_ref"]) + ".")
    if res["flags"]:
        lines.append("Integrity flags (runs kept; only checked incumbents count): " +
                     "; ".join(f"{n}: {f}" for n, f in res["flags"].items()) + ".")
    if res["missing"]:
        lines.append("")
        lines.append("Missing (no valid completed seed-0 run at the budget): " +
                     "; ".join(f"{m['instance']} ({m['why']})" for m in res["missing"]) + ".")
    if res.get("rule", {}).get("decision") == "D-021":
        rule = (f"D-021's reading: the late-clock check first, then {R_LO:.3f} ≤ R ≤ {R_HI:.3f} on the call clock and "
                f"feasible within ±{FEASIBLE_TOL} = INSIDE the band, else OUTSIDE; all {res['n_sample']} instances run")
    else:
        rule = (f"pass: {R_LO:.3f} ≤ R ≤ {R_HI:.3f} on the call clock and feasible within ±{FEASIBLE_TOL}, all "
                f"{res['n_sample']} instances run")
    lines += ["", f"**Verdict: {res['verdict']}** — {res['verdict_reason']} ({rule})."]
    return "\n".join(lines) + "\n"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="itrea mipfeas-report", description="MIPfeas harness: the paired calibration "
                                 "against GAMS's published cuOpt 26.08 GB10 run (R, feasible counts, late-clock check, "
                                 "verdict) — D-021's M4 with --arm A6F, D-020's with A6m")
    ap.add_argument("--arm", default=ARM, help=f"arm label in the run table ({ARM}: D-020's PASS / FAIL; A6F: D-021's "
                                               f"calibration, verdict INSIDE / OUTSIDE / CLOCK-FAIL)")
    ap.add_argument("--set", dest="set_", metavar="SET", default=SET, help=f"run set ({SET})")
    ap.add_argument("--seed", type=int, default=SEED, help=f"D-020: one run per instance, seed {SEED}")
    ap.add_argument("--budget", type=float, default=BUDGET, help=f"seconds ({BUDGET:g}, MIPfeas's limit)")
    ap.add_argument("--gams-dir", default=str(GAMS_DIR), help="GAMS's unpacked archive (D-020)")
    ap.add_argument("--runs", default=str(DATASETS / "runs.parquet"))
    ap.add_argument("--json", default=None, help="also write the result as JSON here")
    ap.add_argument("--md", default=None, help="also write the markdown here")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    for p, what in ((a.runs, "run `itrea build-runs` first"), (a.gams_dir, "GAMS's archive, D-020")):
        if not Path(p).exists():
            print(f"{p} does not exist ({what})", file=sys.stderr)
            return 2
    res = report(pd.read_parquet(a.runs), a.gams_dir, a.arm, a.set_, a.seed, a.budget)
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
