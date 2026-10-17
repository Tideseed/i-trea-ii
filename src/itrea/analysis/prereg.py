"""The pre-registered analysis, exactly as vault D-002 v3 §5–6 states it: H1 (= H1a AND H1b), the secondary family
H2–H5 (Holm), the mechanism hypothesis H6, the feasibility-stratum and P-degenerate companions, from
`datasets/runs.parquet`, the frozen pool `datasets/hard_pool.csv` and the frozen configuration.

    H1a  A3 (SCIP ⇄ GPU EA)  <  A0 (SCIP alone, harness-matched)          primary, intersection–union with H1b
    H1b  A3                  <  A5 (EA alone, GPU)                         primary
    H2   A3                  <  A0+A5 (uncoupled portfolio, computed)      secondary, Holm over H2–H5
    H3   A3                  <  A2 (SCIP ‖ cuOpt)                          secondary
    H4   A3                  <  A1 (SCIP ‖ SCIP)                           secondary
    H5   A3                  <  A0* (tuned SCIP alone; arm label A0star)   secondary
    H6   A5 (x = 10 %)       <  A5x0 (x = 0), on the EA's own P(300)       mechanism

(The computed task that built this module listed v2's H1/H2/H2*/H3/H4/H5; D-002 is v3 since 2026-10-06 00:0x
(D-007), and D-002 is what this module encodes.)

Why it refuses: D-002 §1 makes the pre-registration binding at the tag `prereg-v1` and the confirmatory runs valid only
under the configuration frozen at `frozen-e2`. A confirmatory number computed from a run with another configuration, a
provisional reference, or code older than `prereg-v1` would not be the registered analysis. So in confirmatory mode
every EA run must carry the expected configuration hash (`configs/frozen-e2.yaml` with the arm's registered override,
`ARM_OVERRIDES`), every non-EA run the hash of its frozen solver-side settings (`configs/frozen-e2-solver.json`, see
below; review 2026-10-06 F4), every test instance must have a frozen reference, no run may hold an incumbent stamped
before t0 (clock rule), and (optionally checked here, always checked by the runner) every run's recorded git revision
must descend from `prereg-v1` (a revision from before the history rewrite of 2026-10-17 is translated to its new
commit through `COMMIT_MAP.tsv` first). Otherwise `PreregError`. `--exploratory` computes the same numbers labelled
exploratory.

The frozen solver-side settings: `configs/frozen-e2-solver.json` = {"emphasis": <A0*'s static SCIP emphasis>,
"cuopt_threads": <A2's num_cpu_threads, 3>, "gpu_mem_gb": <A2's RMM limit; must equal frozen-e2's gpu_mem_gb>}, written
with frozen-e2. A non-EA run's config_hash is `pair.run_hash` of {arm, emphasis (A0* only), and for cuOpt arms the
RMM limit and threads}; the expected values are computed with that same function (`expected_hashes`).

Data rules (D-002 §6): seeds 1000 and 1001, T = 300 s, test side of the frozen pool; blocks voided by GPU health are
dropped (their rerun is used: the highest `attempt` of a non-void run); runs whose EA backend does not match the arm
are dropped and counted (the computed A0+A5 row inherits its A5 parent's backend, so it goes with it); crashed runs are
kept (truncated at the crash). The P tests leave out the feasibility stratum (time to first feasible, paired
Prentice–Wilcoxon + sign test) and P-degenerate instances. P-degenerate = the pool's screening-time flag, plus every
test instance whose FROZEN reference is 0 (`p_degenerate_added_from_reference`; the pool flag predates reference.csv).
Their companion endpoints: reference 0 → time to first feasible and to the reference (t_gap1; t_gap5 is identical
there, Berthold's gap being 1 for every other value); sign crossing → time to gap ≤ 1 % and ≤ 5 %.

Companions (all censored, paired Prentice–Wilcoxon + sign test) run for EVERY comparison with the time columns of its
value column (pt → t_first/t_gap1/t_gap5, ea_pt → ea_t_first/ea_t_gap1/ea_t_gap5), in the families of the main tests
(H1a, H1b unadjusted; Holm within the secondary family and within the mechanism family): the feasibility stratum, the
P-degenerate instances, and the instances a comparison excluded because both arms had P = 1 on every seed (D-002 §6:
"reported under time to first feasible").

Descriptive only (no test, outside every family): H2's live check (A3r against the computed A0+A5 and A3 against A3r on
the A3r instances, D-002 §5) and the MIPfeas-style integral (D-002 §4 secondary; shifted geometric mean, s = 0.001) per
arm over the P instances and over all test instances, plus the SCREEN arm (SCIP alone, seed 900, T = 300 s) over the
screened MIPLIB candidates — the study's only number near the published 233-instance GB10 baseline, which is at 600 s
with another solver.

    python -m itrea.analysis.prereg [--exploratory] [--no-git-check] [--frozen-solver configs/frozen-e2-solver.json]
        -> datasets/prereg_results.json, paper/generated/prereg_results.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.analysis import commit_map as CM
from itrea.analysis import metrics as M
from itrea.analysis import stats as S
from itrea.benchmarks import references as R
from itrea.paths import CONFIGS, DATASETS, ROOT
from itrea.trace import _clean

PREREG = "D-002 v3 (2026-10-06)"
SEEDS = (1000, 1001)
T = 300.0
COMPARISONS = (  # id, arm, control, family, value column
    ("H1a", "A3", "A0", "primary", "pt"),
    ("H1b", "A3", "A5", "primary", "pt"),
    ("H2", "A3", "A0+A5", "secondary", "pt"),
    ("H3", "A3", "A2", "secondary", "pt"),
    ("H4", "A3", "A1", "secondary", "pt"),
    ("H5", "A3", "A0star", "secondary", "pt"),
    ("H6", "A5", "A5x0", "mechanism", "ea_pt"),
)
# The registered configuration of every EA arm = frozen-e2 with the arm's implied changes, exactly as the runner
# applies them (itrea.experiments.pair.ARMS: A5/A5x0 run without exchange; every GPU EA with backend cupy); the run's
# config_hash is the hash of that effective configuration. A0+A5 (computed) carries A5's hash.
ARM_OVERRIDES: dict[str, dict] = {
    "A3": {"backend": "cupy"},
    "A3r": {"exchange.enabled": False, "backend": "cupy"},     # H2's live check (descriptive), same EA as A5
    "A5": {"exchange.enabled": False, "backend": "cupy"},
    "A5x0": {"exchange.enabled": False, "ranks.x": 0.0, "backend": "cupy"},
    "A0+A5": {"exchange.enabled": False, "backend": "cupy"},
}
SOLVER_ARMS = ("A0", "A0star", "A1", "A2")       # non-EA arms: hash of pair.solver_settings (frozen-e2-solver.json)
SOLVER_KEYS = ("emphasis", "cuopt_threads", "gpu_mem_gb")
DESCRIPTIVE_ARMS = ("A3r",)                      # selected and reported, never tested
TIME_COLS = {   # companion endpoints per comparison value column
    "pt": {"t_first": "t_first", "t_gap1": "t_gap1", "t_gap5": "t_gap5"},
    "ea_pt": {"t_first": "ea_t_first", "t_gap1": "ea_t_gap1", "t_gap5": "ea_t_gap5"},
}
MF_COLS = {"pt": "pt_mf", "ea_pt": "ea_pt_mf"}   # the MIPfeas-style integral beside each value column
SCREEN_ARM, SCREEN_SEED = "SCREEN", 900
TAG = "prereg-v1"


class PreregError(RuntimeError):
    """The inputs are not the registered ones: no confirmatory result is computed."""


def sha256_file(p: str | os.PathLike) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_frozen_solver(path: str | os.PathLike) -> dict:
    """The frozen solver-side settings (module docstring). Unknown keys or a missing emphasis raise."""
    d = json.loads(Path(path).read_text())
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected a JSON object")
    unknown = sorted(set(d) - set(SOLVER_KEYS) - {"name", "note"})
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown} (known: {SOLVER_KEYS})")
    if not d.get("emphasis"):
        raise ValueError(f"{path}: 'emphasis' (A0*'s frozen static emphasis) is required")
    return d


def expected_hashes(frozen_yaml: str | os.PathLike, overrides: dict[str, dict] = ARM_OVERRIDES,
                    solver: dict | None = None) -> dict[str, str]:
    """Expected config_hash per registered arm. EA arms: frozen-e2 with the arm's override. With `solver` (the frozen
    solver-side settings) also A0, A0*, A1, A2, computed by the runner's own `pair.run_hash` so the two cannot
    drift apart."""
    from itrea.config import EAConfig

    base = EAConfig.from_yaml(frozen_yaml)
    out = {arm: base.with_overrides(ov).hash() if ov else base.hash() for arm, ov in overrides.items()} | \
        {"__base__": base.hash()}
    if solver is not None:
        from itrea.experiments import pair

        g = solver.get("gpu_mem_gb")
        if g is not None and float(g) != float(base.gpu_mem_gb):
            raise ValueError(f"frozen solver settings: gpu_mem_gb {g} differs from frozen-e2's {base.gpu_mem_gb} "
                             "(A2's RMM limit must equal the EA's pool, D-002 §3)")
        opts = pair.PairOptions(cuopt_threads=solver.get("cuopt_threads"),
                                gpu_mem_gb=None if g is None else float(g))
        for arm in SOLVER_ARMS:
            job = dict(arm=arm, set="e2", instance="placeholder.mps", seed=SEEDS[0], budget=T,
                       config=str(frozen_yaml))
            if arm == "A0star":
                job["emphasis"] = str(solver["emphasis"])
            out[arm] = pair.run_hash(job, opts)
    return out


def _git_descends(rev: str, tag: str = TAG, cwd: Path = ROOT) -> bool | None:
    """True/False if git can tell whether `rev` descends from `tag`; None if git or the tag is unavailable. A hash from
    before the history rewrite of 2026-10-17 is first translated to its new commit (`COMMIT_MAP.tsv`,
    `itrea.analysis.commit_map`), so a `runs.parquet` built from old run records still passes the check."""
    if not rev:
        return False
    rev = CM.translate(rev)
    try:
        r = subprocess.run(["git", "merge-base", "--is-ancestor", tag, rev], cwd=cwd, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode in (0, 1):
        return r.returncode == 0
    return None


def select_runs(runs: pd.DataFrame, pool: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Confirmatory rows: test instances, seeds 1000/1001, T = 300, the registered arms, void blocks and backend
    mismatches dropped, one row per (instance, arm, seed) (the highest attempt)."""
    test = pool[pool["side"] == "test"]
    arms = sorted({a for _, a, c, _, _ in COMPARISONS for a in (a, c)})
    r = runs[runs["instance"].isin(test["instance"]) & runs["seed"].isin(SEEDS) &
             runs["arm"].isin(arms + list(DESCRIPTIVE_ARMS)) & np.isclose(runs["budget"].astype(float), T)].copy()
    if "split" in r.columns:                     # a smoke or tuning run of a test instance is not a confirmatory run
        r = r[(r["split"].fillna("") == "") | (r["split"] == "test")]    # A0+A5 carries its parents' split
    info = dict(void_runs=int(r["block_void"].astype(bool).sum()),
                backend_mismatch=int(r["backend_mismatch"].astype(bool).sum()))
    r = r[~r["block_void"].astype(bool) & ~r["backend_mismatch"].astype(bool)]
    r = r.sort_values(["instance", "arm", "seed", "attempt", "run_id"], kind="mergesort")
    dup = r[r.duplicated(["instance", "arm", "seed"], keep=False) &
            ~r.duplicated(["instance", "arm", "seed", "attempt"], keep=False)]
    amb = r[r.duplicated(["instance", "arm", "seed", "attempt"], keep=False)]
    info["ambiguous"] = sorted({f"{i}|{a}|s{s}" for i, a, s in zip(amb["instance"], amb["arm"], amb["seed"])})
    info["reruns_used"] = int(dup.drop_duplicates(["instance", "arm", "seed"]).shape[0])
    r = r.drop_duplicates(["instance", "arm", "seed"], keep="last")
    have = set(zip(r["instance"], r["arm"], r["seed"]))
    info["missing"] = sorted(f"{i}|{a}|s{s}" for i in sorted(test["instance"]) for a in arms for s in SEEDS
                             if (i, a, s) not in have)
    info["crashed_by_arm"] = {a: int(g["crashed"].astype(bool).sum()) for a, g in r.groupby("arm", sort=True)}
    info["runs_by_arm"] = {a: int(len(g)) for a, g in r.groupby("arm", sort=True)}
    mism = []
    if "parent_git_revs" in r.columns:           # computed rows whose two parents ran different code (both checked)
        for rid, js in zip(r["run_id"], r["parent_git_revs"].fillna("")):
            if js and len(set(json.loads(js).values())) > 1:
                mism.append(rid)
    info["portfolio_parent_rev_mismatch"] = sorted(mism)
    r = r.reset_index(drop=True)
    r.attrs["ambiguous"] = info["ambiguous"]
    return r, info


def check_registered(r: pd.DataFrame, pool: pd.DataFrame, hashes: dict[str, str] | None, git_check: bool) -> list[str]:
    """Every reason the selected runs are not the registered analysis (empty = registered)."""
    problems = []
    if info_ambiguous := r.attrs.get("ambiguous"):
        problems.append(f"{len(info_ambiguous)} (instance, arm, seed) with several non-void runs of the same attempt: "
                        f"{info_ambiguous[:3]}")
    if hashes is None:
        problems.append("no frozen configuration (configs/frozen-e2.yaml) — the confirmatory analysis needs it")
    else:
        base = hashes.get("__base__")
        if base and "base_hash" in r.columns:
            ea = r[r["arm"].isin([a for a in hashes if a != "__base__"]) & (r["base_hash"] != "") &
                   (r["base_hash"] != base)]
            if len(ea):
                problems.append(f"{len(ea)} run(s) started from a configuration file other than frozen-e2 "
                                f"({base}): {sorted(set(ea['base_hash']))[:3]}")
        nosolver = [a for a in SOLVER_ARMS if a not in hashes]
        if nosolver:
            problems.append(f"no frozen solver-side settings (configs/frozen-e2-solver.json): the configuration of "
                            f"{', '.join(nosolver)} (A0*'s emphasis, A2's cuOpt limits) cannot be checked")
        # a run with a D-021 harness is identified by its run_hash (an EA arm's config_hash is the EA configuration's
        # alone, the same with and without a harness); prereg-v1 registered no harness, so such a run cannot match
        if "run_hash" in r.columns and "harness" in r.columns:
            got = r["config_hash"].where(r["harness"].fillna("").astype(str) == "", r["run_hash"])
        else:
            got = r["config_hash"]
        for arm, h in sorted(hashes.items()):
            if arm == "__base__":
                continue
            bad = r[(r["arm"] == arm) & (got != h)]
            if len(bad):
                problems.append(f"{len(bad)} {arm} run(s) with a configuration hash other than the frozen {h}: "
                                f"{sorted(set(got[bad.index]))[:3]}")
    if "pre_t0" in r.columns:
        bad = r[~r["synthetic"].astype(bool) & r["pre_t0"].astype(bool)]
        if len(bad):
            problems.append(f"{len(bad)} run(s) with an incumbent stamped before t0 or before the trace clock started "
                            f"(clock rule): {sorted(bad['run_id'])[:3]}")
    nf = r[~r["ref_frozen"].astype(bool)]
    if len(nf):
        problems.append(f"{nf['instance'].nunique()} test instance(s) without a frozen reference "
                        f"(datasets/reference.csv): {sorted(set(nf['instance']))[:5]}")
    if git_check:
        real = r[~r["synthetic"].astype(bool)]
        for rev in sorted(set(real["git_rev"])):
            ok = _git_descends(rev)
            if ok is not True:
                problems.append(f"git revision {rev or '(none)'} is not known to descend from {TAG}"
                                + (" (git or the tag unavailable)" if ok is None else ""))
    return problems


def run(runs: pd.DataFrame, pool: pd.DataFrame, hashes: dict[str, str] | None, exploratory: bool = False,
        git_check: bool = True, inputs: dict | None = None) -> dict:
    """The whole registered analysis as one JSON-ready dict (deterministic: no time stamps)."""
    r, info = select_runs(runs, pool)
    problems = check_registered(r, pool, hashes, git_check)
    if problems and not exploratory:
        raise PreregError("refusing to compute confirmatory results:\n  - " + "\n  - ".join(problems))
    test = pool[pool["side"] == "test"]
    test_inst = set(test["instance"])
    feas = sorted(test.loc[test["stratum"] == "miplib_feas", "instance"])
    pool_degen = set(test.loc[test["p_degenerate"].astype(bool) & (test["stratum"] != "miplib_feas"), "instance"])
    # the pool flag is the screening-time value; re-evaluated against the FROZEN reference (review F5): a test instance
    # whose frozen reference is 0 is P-degenerate whatever screening saw (arm-independent, so no outcome selection)
    ref0 = set(r.loc[r["ref_zero_instance"].astype(bool), "instance"]) if "ref_zero_instance" in r.columns else set()
    added = sorted((ref0 & test_inst) - pool_degen - set(feas))
    degen = sorted(pool_degen | set(added))
    pool_ref0 = set(test.loc[test["ref_zero"].fillna(False).astype(bool), "instance"]) if "ref_zero" in test.columns \
        else set()
    degen_ref0 = sorted(set(degen) & (pool_ref0 | ref0))
    degen_sign = sorted(set(degen) - set(degen_ref0))
    p_inst = sorted(test_inst - set(feas) - set(degen))
    rp = r[r["instance"].isin(p_inst)]
    comps = {}
    for hid, arm, ctrl, fam, col in COMPARISONS:
        c = S.compare(rp, arm, ctrl, value=col)
        c.update(family=fam)
        comps[hid] = c
    for fam in ("secondary", "mechanism"):
        ids = [h for h, *_ in COMPARISONS if comps[h]["family"] == fam]
        adj, _ = S.holm([comps[h]["p"] for h in ids])
        for h, pa in zip(ids, adj):
            comps[h]["p_holm"] = float(pa)
            comps[h]["decision"] = S.decide(float(pa), comps[h]["ci_lo"], comps[h]["ci_hi"])
    for h in ("H1a", "H1b"):
        comps[h]["p_holm"] = comps[h]["p"]       # intersection–union: each at alpha, no adjustment
    out = dict(
        prereg=PREREG, mode="exploratory" if exploratory else "confirmatory",
        problems=problems, inputs=inputs or {}, seeds=list(SEEDS), T=T,
        alpha=S.ALPHA, margin=S.MARGIN, n_boot=S.N_BOOT, boot_seed=S.BOOT_SEED,
        test_instances=int(len(test)), p_instances=p_inst, feasibility_stratum=feas, p_degenerate=degen,
        p_degenerate_ref_zero=degen_ref0, p_degenerate_sign_crossing=degen_sign,
        p_degenerate_added_from_reference=added,
        comparisons=comps, H1=S.intersection_union([comps["H1a"]["decision"], comps["H1b"]["decision"]]),
        reading=_reading(comps), runs=info)
    rf = r[r["instance"].isin(feas)]
    out["feasibility_companion"] = censored_family(lambda h: rf, "t_first")
    out["p_degenerate_companion"] = {
        "ref_zero": {tt: censored_family(lambda h: r[r["instance"].isin(degen_ref0)], tt)
                     for tt in ("t_first", "t_gap1")},
        "sign_crossing": {tt: censored_family(lambda h: r[r["instance"].isin(degen_sign)], tt)
                          for tt in ("t_gap1", "t_gap5")}}
    out["excluded_companion"] = censored_family(lambda h: rp[rp["instance"].isin(comps[h]["excluded"])], "t_first")
    out["descriptive"] = dict(H2_live_check=h2_live_check(rp), mipfeas=mipfeas_report(r, runs, p_inst, test_inst))
    return out


def censored_family(df_for, tt: str) -> dict:
    """The censored companion of every comparison on the endpoint `tt` ("t_first" | "t_gap1" | "t_gap5", mapped to
    the comparison's own time column by TIME_COLS), with the main tests' family structure: H1a/H1b unadjusted, Holm
    on the Prentice–Wilcoxon p within the secondary and within the mechanism family. df_for(hid) -> rows to use."""
    out = {}
    for hid, arm, ctrl, fam, col in COMPARISONS:
        t = TIME_COLS[col][tt]
        df = df_for(hid)
        if len(df) and (t not in df.columns or t + "_event" not in df.columns):
            c = dict(arm=arm, control=ctrl, time=t, n_instances=0, ppw=None, sign=None, note=f"no column {t}")
        else:
            c = S.compare_censored(df, arm, ctrl, t, t + "_event")
        c["family"] = fam
        out[hid] = c
    for fam in ("secondary", "mechanism"):
        ids = [h for h in out if out[h]["family"] == fam and out[h].get("ppw")]
        if ids:
            adj, _ = S.holm([out[h]["ppw"]["p"] for h in ids])
            for h, pa in zip(ids, adj):
                out[h]["p_holm"] = float(pa)
    for h, *_ in COMPARISONS:
        if out[h]["family"] == "primary" and out[h].get("ppw"):
            out[h]["p_holm"] = out[h]["ppw"]["p"]
    return out


def _descriptive(df: pd.DataFrame, arm: str, ctrl: str, value: str = "pt") -> dict:
    c = S.compare(df, arm, ctrl, value=value)
    c.pop("decision", None)                       # descriptive: no decision is drawn
    c["descriptive"] = True
    return c


def h2_live_check(rp: pd.DataFrame) -> dict:
    """D-002 §5 H2: the computed uncoupled portfolio is checked against the live A3r subset (descriptive)."""
    inst = sorted(set(rp.loc[rp["arm"] == "A3r", "instance"]))
    live = rp[rp["instance"].isin(inst)]
    return {"instances": inst, "note": "descriptive, outside Holm; P instances with A3r runs",
            "A3r_vs_A0+A5": _descriptive(live, "A3r", "A0+A5"), "A3_vs_A3r": _descriptive(live, "A3", "A3r")}


def _sgm(per_instance: pd.Series) -> dict:
    v = per_instance.dropna()
    return dict(n=int(len(v)), sgm=M.shifted_geomean(v.tolist(), shift=M.MIPFEAS_SHIFT))


def mipfeas_report(r: pd.DataFrame, runs: pd.DataFrame, p_inst: list[str], test_inst: set[str]) -> dict:
    """MIPfeas-style integral (D-002 §4, secondary, descriptive): per arm, the seed mean per instance, then the shifted
    geometric mean (s = 0.001) over the P instances and over all test instances; plus the SCREEN arm's value over the
    screened MIPLIB candidates."""
    out: dict = dict(shift=M.MIPFEAS_SHIFT, by_arm={}, ea_by_arm={},
                     definition="gap 2 with no incumbent, |z - z*| / max(|z|, |z*|, 1) otherwise, 0 at or beyond "
                                "z*; integrated over [0, T] and divided by T; shifted geometric mean, s = 0.001")
    arms = sorted({a for _, a, c, _, _ in COMPARISONS for a in (a, c)} | set(DESCRIPTIVE_ARMS))
    ea_arms = sorted({a for _, a, c, _, col in COMPARISONS if col == "ea_pt" for a in (a, c)} | {"A3"})
    for col, key, arm_list in ((MF_COLS["pt"], "by_arm", arms), (MF_COLS["ea_pt"], "ea_by_arm", ea_arms)):
        if col not in r.columns:
            out[key] = None
            continue
        for a in arm_list:
            per = r[r["arm"] == a].groupby("instance")[col].mean()
            if len(per) == 0:
                continue
            out[key][a] = dict(p_instances=_sgm(per[per.index.isin(p_inst)]),
                               test_instances=_sgm(per[per.index.isin(test_inst)]))
    out["screen_miplib"] = screen_mipfeas(runs)
    return out


def screen_mipfeas(runs: pd.DataFrame) -> dict | None:
    """The SCREEN arm (SCIP alone, default) over the screened MIPLIB candidates, seed 900 (stage 1 covers every
    candidate; seed 901 only some), T = 300 s, reference .solu — the only MIPfeas-style number of the study near the
    published 233-instance baseline, which differs in T (600 s) and solver."""
    need = {"arm", "seed", "budget", "pt_mf", "set", "instance_path", "instance"}
    if not need <= set(runs.columns):
        return None
    sc = runs[(runs["arm"] == SCREEN_ARM) & (runs["seed"] == SCREEN_SEED) &
              np.isclose(runs["budget"].astype(float), T)]
    if "synthetic" in sc.columns:
        sc = sc[~sc["synthetic"].astype(bool)]
    if "block_void" in sc.columns:
        sc = sc[~sc["block_void"].astype(bool)]
    sc = sc[[R.is_miplib(s, p) for s, p in zip(sc["set"].astype(str), sc["instance_path"].astype(str))]]
    if "attempt" in sc.columns:
        sc = sc.sort_values(["instance", "attempt", "run_id"], kind="mergesort")
    sc = sc.drop_duplicates("instance", keep="last")
    return dict(_sgm(sc.set_index("instance")["pt_mf"]), arm=SCREEN_ARM, seed=SCREEN_SEED, T=T,
                label="SCIP alone (default settings), seed 900, T = 300 s, reference .solu, screened MIPLIB 2017 "
                      "benchmark candidates; the published GB10 cuOpt baseline (0.0393) is at T = 600 s on 233 "
                      "instances with another solver, so the two are not directly comparable")


def _reading(c: dict) -> str:
    """D-002 §6's reading of H1 with H2 (no claim beyond it)."""
    h1 = c["H1a"]["decision"] == S.SUPPORTED and c["H1b"]["decision"] == S.SUPPORTED
    if not h1:
        return "H1 not supported: the main claim is not shown"
    if c["H2"]["decision"] == S.SUPPORTED:
        return "H1 and H2 supported: the coupling creates value beyond running both"
    return "H1 supported, H2 not: running both helps; the exchange is not shown to"


def markdown(res: dict) -> str:
    """The results table for the vault and the paper (deterministic)."""
    def f(x, nd=3):
        return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"

    lines = [f"<!-- generated by itrea.analysis.prereg from datasets/runs.parquet; {res['prereg']}; "
             f"mode {res['mode']} -->",
             "| hypothesis | arm | control | n | mean d [95 % CI] | Wilcoxon p (method) | Holm p | decision |",
             "|---|---|---|---|---|---|---|---|"]
    for hid, c in res["comparisons"].items():
        lines.append(f"| {hid} | {c['arm']} | {c['control']} | {c['n_instances']} | {f(c['mean_d'])} "
                     f"[{f(c['ci_lo'])}, {f(c['ci_hi'])}] | {f(c['p'], 4)} ({c['method']}) | {f(c.get('p_holm'), 4)} "
                     f"| {c['decision']} |")
    lines.append("")
    lines.append(f"H1 (H1a AND H1b): **{res['H1']}**. {res['reading']}.")
    d = res.get("descriptive") or {}
    mf = (d.get("mipfeas") or {}).get("by_arm") or {}
    if mf:
        lines.append("")
        lines.append("Descriptive (no test): MIPfeas-style integral, shifted geometric mean (s = 0.001), P instances / "
                     "all test instances: " + "; ".join(
                         f"{a} {f(v['p_instances']['sgm'], 4)} / {f(v['test_instances']['sgm'], 4)}"
                         for a, v in sorted(mf.items())) + ".")
    sm = (d.get("mipfeas") or {}).get("screen_miplib")
    if sm and sm.get("n"):
        lines.append(f"SCREEN arm over {sm['n']} screened MIPLIB instances: {f(sm['sgm'], 4)} ({sm['label']}).")
    live = d.get("H2_live_check") or {}
    if live.get("instances"):
        c = live["A3r_vs_A0+A5"]
        lines.append(f"H2 live check on {len(live['instances'])} instances: A3r vs A0+A5 mean d {f(c['mean_d'])} "
                     f"[{f(c['ci_lo'])}, {f(c['ci_hi'])}] (descriptive).")
    if res["mode"] != "confirmatory":
        lines.append("")
        lines.append("**Exploratory** — not the registered analysis: " + "; ".join(res["problems"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the pre-registered analysis of D-002 v3")
    ap.add_argument("--runs", default=str(DATASETS / "runs.parquet"))
    ap.add_argument("--pool", default=str(DATASETS / "hard_pool.csv"))
    ap.add_argument("--frozen", default=str(CONFIGS / "frozen-e2.yaml"))
    ap.add_argument("--frozen-solver", default=str(CONFIGS / "frozen-e2-solver.json"),
                    help="frozen solver-side settings of the non-EA arms (A0* emphasis, A2 cuOpt limits)")
    ap.add_argument("--reference", default=str(DATASETS / "reference.csv"))
    ap.add_argument("--out", default=str(DATASETS / "prereg_results.json"))
    ap.add_argument("--md", default=str(ROOT / "paper" / "generated" / "prereg_results.md"))
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--no-git-check", action="store_true")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    for p in (a.runs, a.pool):
        if not Path(p).exists():
            print(f"{p} does not exist")
            return 2
    solver = read_frozen_solver(a.frozen_solver) if Path(a.frozen_solver).exists() else None
    hashes = expected_hashes(a.frozen, solver=solver) if Path(a.frozen).exists() else None
    from itrea.analysis.build_runs import frame_digest

    runs = pd.read_parquet(a.runs)
    inputs = dict(runs_content_sha256=frame_digest(runs), pool_sha256=sha256_file(a.pool),
                  reference_sha256=sha256_file(a.reference) if Path(a.reference).exists() else "",
                  frozen_config=os.path.relpath(a.frozen, ROOT), frozen_hashes=hashes or {},
                  frozen_solver=os.path.relpath(a.frozen_solver, ROOT) if solver is not None else "",
                  frozen_solver_sha256=sha256_file(a.frozen_solver) if solver is not None else "")
    try:
        res = run(runs, pd.read_csv(a.pool), hashes, a.exploratory, not a.no_git_check, inputs)
    except PreregError as e:
        print(e)
        return 3
    Path(a.out).write_text(json.dumps(_clean(res), indent=1, sort_keys=True) + "\n")
    Path(a.md).parent.mkdir(parents=True, exist_ok=True)
    Path(a.md).write_text(markdown(res))
    print(markdown(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
