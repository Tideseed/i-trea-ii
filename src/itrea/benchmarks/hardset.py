"""The hard instance pool: candidates, the two-stage SCIP screening rule, strata, the deterministic every-k-th
selection, the group split and the memory-fit replacement (vault D-002 v3 §1–2, `10-design/Hard instance pool.md`).

Why each rule exists:
- **Hardness from SCIP alone, from a fresh screening pass** (arm SCREEN, seeds 900/901, 300 s). optopt's grid mixes
  cuOpt into "best strategy", so selecting on it would remove the instances where cuOpt is strong and decide the
  SCIP ‖ cuOpt comparison through the selection. This module therefore reads ONLY `datasets/runs.parquet` rows with
  arm SCREEN, never `datasets/hardness_from_optopt.csv`.
- **Two stages:** seed 901 runs only where seed 900 gave P(300) ≥ 0.05 or no solution (`stage2_needed`); an instance
  below that on seed 900 cannot reach a stratum, so it is excluded with that reason.
- **Strata with fixed quotas** (test / tuning): MIPLIB optimisation-hard 16/8 (feasible on both seeds, mean
  P(300) ≥ 0.10), MIPLIB feasibility-hard 4/2 (no feasible solution on ≥ 1 seed), PGLib-UC 6/3 (mean P(300) ≥ 0.10 or
  no solution on a seed; `rts_gmlc` never in tuning — the unseen-family check), ML4CO item placement 6/3 (top half by
  mean P(300)).
- **Every k-th, deterministically.** Within a stratum the eligible instances are sorted hardest first (mean P(300)
  descending; the feasibility stratum by mean time to first feasible, censored at T, descending), ties by name, and
  for each side cut into Q (that side's quota) consecutive buckets (bucket b = positions ⌊b·N/Q⌋ … ⌊(b+1)·N/Q⌋ − 1).
  The first member of every bucket is the "every k-th" pick, with k = N/Q allowed to be fractional so the picks
  spread over the whole range; when a pick cannot be taken, the NEXT instance of its bucket replaces it (D-002's
  "next instance in its stratum"). Slots still open after the buckets are filled from the rest in sorted order.
- **Sides by group, never split.** MIPLIB: family = the leading letters of the name ("neos-1122047" and "neos859080" →
  neos, "supportcase10" → supportcase, "comp07-2idx" → comp), merged (union-find) with MIPLIB's own `Group` column of
  `tag_benchmark.html` where it names one — a family is on one side under both definitions. PGLib: group = the base
  CASE (`ca__2014-09-01_reserves_{0,1,3,5}` → `ca__2014-09-01`; each `rts_gmlc` day is its own case), family = the
  system (`ca`, `rts_gmlc`): `ca` days may land on both sides (a within-family split, as `10-design/Datasets.md`
  "Generators" expects), `rts_gmlc` only on test (the unseen-family check). D-002 §2 and the pool note still say
  "PGLib by base system"; that wording is to be amended to "by base case" before `prereg-v1` (review 2026-10-06,
  F6). Under the literal by-system reading `ca` would be one group, tuning PGLib is `ca` only, so test PGLib would be
  `rts_gmlc` only. ML4CO: group = the instance index, family = item_placement (one generator).
  Phase 1 (`assign_sides`) puts every group on one side before any pick, largest group first, on the side whose
  remaining need (its quota's share — 2/3 test, 1/3 tuning — of each stratum's supply, counted within that side's
  family cap) it covers better; phase 2 takes every k-th instance per (stratum, side), so both sides are spread over
  difficulty.
- **No family supplies more than 20 % of the test set** (⌊0.2 · 32⌋ = 6), **nor more than 20 % of the tuning set**
  (⌊0.2 · 16⌋ = 3). Without the tuning cap the largest family (`neos`, 55 candidates) went to tuning wholesale and
  filled 5–10 of the 10 MIPLIB tuning slots (20 synthetic screening draws on the real candidate table), so E4 would
  tune on mostly one family (review 2026-10-06, F12; the tuning cap is not yet in D-002 §2).
- **Memory fit:** an instance fits when the EA's own memory plan (`itrea.ea.engine.plan_memory` on the instance's n, m,
  nnz and gene dtype, `EAConfig()` defaults, budget `gpu_mem_gb`) fits with P ≥ 32 — the same test `Engine.prepare`
  applies (at P ≥ 32 and x = 0.1 Rank 0 always has a slot). The plan counts the whole device footprint (population,
  offspring, candidate set and dedup temporaries, frontier, device problem, local-search matrices, 10 % headroom);
  D-002's "P × n × bytes at P = 32" (kept as `pop_bytes_p32`) counts only the population and can never exclude a
  candidate (review 2026-10-06, F10). An unfit instance is skipped and replaced by the next one of its bucket; both
  are listed (`replaces` in the pool, `replacements` / `unfit` in the report). The fit therefore depends on the
  whole default `EAConfig()` and on `plan_memory`'s constants, not only on `gpu_mem_gb`: the report records the
  config hash (`fit_config_hash`) and the engine constants (`fit_rule`), and a later change to either changes the
  fit columns of a re-derived pool file.
- **Flags:** P-degenerate (reference 0 or a sign crossing in screening), feasibility stratum (time-to-first-feasible
  endpoint), "no exchange window" (first SOLVING-stage SCIP callback after T/2 on a screening seed), and
  `solver_limit_stop`: a screening run where SCIP stopped at its own memory or node limit (exit 0, so not a crash)
  and held its last incumbent to T — the P(300) then partly measures the limit, not hardness. Flagged and listed in
  the report (`solver_limit_stops`), not excluded: exclusion would change the pre-registered selection rule (review
  2026-10-06, F13).

A greedy pick-by-pick split (tried first) stranded quota: a family put on one side early left the other side short
later. Sides are therefore fixed for all strata at once from the supply, and any remaining shortfall is reported
(`report["shortfall"]`, written to `hard_pool_report.json` with the replacements, unfit instances and group sides),
never filled by breaking a rule.

    python -m itrea.benchmarks.hardset candidates                                   # stage-1 list (CSV)
    python -m itrea.benchmarks.hardset stage2 [--runs datasets/runs.parquet]       # seed-901 list
    python -m itrea.benchmarks.hardset select [--runs …] [--out datasets/hard_pool.csv] [--shapes …]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.benchmarks import references as R
from itrea.paths import DATA, DATASETS, ROOT

SCREEN_ARM = "SCREEN"
SEED1, SEED2 = 900, 901
T_SCREEN = 300.0
STAGE2_P = 0.05
HARD_P = 0.10
E5_ONLY = ("supportcase22",)
PGLIB_SYSTEMS = ("ca", "rts_gmlc")          # ferc: exploratory only
ML4CO_VALID = tuple(range(9900, 9925))      # the 25 validation instances optopt used (first 25 of valid/)
ML4CO_TRAIN = tuple(range(0, 35))
FAMILY_CAP_SHARE = 0.20
P_FIT = 32
SOLVER_LIMIT_REASONS = ("memory_limit", "node_limit")     # scip_run._EXIT: SCIP's own limits, exit code 0
GENE_DTYPE_BY_BYTES = {1: np.dtype(np.int8), 4: np.dtype(np.int32), 8: np.dtype(np.float64)}   # gene_dtype_for


@dataclass(frozen=True)
class Stratum:
    name: str
    test: int
    tuning: int


STRATA = (Stratum("miplib_feas", 4, 2), Stratum("pglib", 6, 3), Stratum("ml4co", 6, 3),
          Stratum("miplib_opt", 16, 8))
TEST_TOTAL = sum(s.test for s in STRATA)

POOL_COLUMNS = ["instance", "set", "path", "stratum", "side", "group", "family", "order", "bucket",
                "p_s900", "p_s901", "p_mean", "n_seeds", "feasible_seeds", "t_first_mean", "solved_optimal",
                "p_degenerate", "ref_zero", "sign_crossing", "no_exchange_window", "feasibility_endpoint",
                "exit_s900", "exit_s901", "solver_limit_stop",
                "n", "m", "nnz", "gene_bytes", "pop_bytes_p32", "plan_p", "plan_bytes", "fits", "replaces"]


# -------------------------------------------------------------------------------------------- candidates
def relative(p: str | os.PathLike) -> str:
    """Repository-relative path WITHOUT resolving symlinks (data/ holds symlinks into the predecessor's checkout; the
    pool file must say data/miplib/inst/x.mps.gz, not the link target)."""
    a = Path(os.path.abspath(p))
    try:
        return str(a.relative_to(Path(os.path.abspath(ROOT))))
    except ValueError:
        return str(a)


def _html_table(path: Path) -> list[list[str]]:
    s = path.read_text()
    body = s[s.find("<tbody>"): s.find("</tbody>")]
    rows = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)
        rows.append([re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", t)).strip() for t in tds])
    return rows


def miplib_candidates(data: Path = DATA) -> pd.DataFrame:
    """The MIPLIB 2017 benchmark list from tag_benchmark.html minus =inf= (solu.txt) and E5-only instances.
    Cross-checked against benchmark-v2.test and inst/; any disagreement raises."""
    d = data / "miplib"
    rows = _html_table(d / "tag_benchmark.html")
    names = [r[0] for r in rows]
    groups = {r[0]: (r[9] if len(r) > 9 else "") for r in rows}
    objs = {r[0]: (r[10] if len(r) > 10 else "") for r in rows}
    test_list = [ln.strip() for ln in (d / "benchmark-v2.test").read_text().splitlines() if ln.strip()]
    test_names = sorted(R.stem(x) for x in test_list)
    on_disk = sorted(R.stem(p.name) for p in (d / "inst").glob("*.mps.gz"))
    if sorted(names) != test_names or sorted(names) != on_disk:
        raise ValueError(f"MIPLIB benchmark lists disagree: html {len(names)}, benchmark-v2.test {len(test_names)}, "
                         f"inst/ {len(on_disk)}; e.g. {sorted(set(names) ^ set(on_disk))[:5]}")
    solu = R.read_solu(d / "solu.txt")
    inf = sorted(n for n in names if (solu.get(n) and solu[n].kind == "infeasible"))
    html_inf = sorted(n for n in names if objs.get(n, "").lower() == "infeasible")
    if inf != html_inf:
        raise ValueError(f"=inf= in solu.txt {inf} differs from 'Infeasible' in tag_benchmark.html {html_inf}")
    keep = [n for n in sorted(names) if n not in inf and n not in E5_ONLY]
    return pd.DataFrame(dict(instance=keep, set="miplib", path=[relative(d / "inst" / f"{n}.mps.gz") for n in keep],
                             miplib_group=[groups.get(n, "") for n in keep]))


def pglib_candidates(data: Path = DATA) -> pd.DataFrame:
    files = sorted((data / "pglib_uc" / "mps").glob("*.mps.gz"))
    keep = [p for p in files if pglib_family(R.stem(p.name)) in PGLIB_SYSTEMS]
    return pd.DataFrame(dict(instance=[R.stem(p.name) for p in keep], set="pglib_uc",
                             path=[relative(p) for p in keep], miplib_group=""))


def ml4co_candidates(data: Path = DATA) -> pd.DataFrame:
    base = data / "ml4co" / "instances" / "1_item_placement"
    paths = [base / "valid" / f"item_placement_{i}.mps.gz" for i in ML4CO_VALID] + \
            [base / "train" / f"item_placement_{i}.mps.gz" for i in ML4CO_TRAIN]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"ML4CO candidates missing: {missing[:3]}")
    return pd.DataFrame(dict(instance=[R.stem(p.name) for p in paths], set="ml4co_item_placement",
                             path=[relative(p) for p in paths], miplib_group=""))


def candidates(data: Path = DATA) -> pd.DataFrame:
    """Every screening candidate (D-002 §2) with set, path, group and family."""
    df = pd.concat([miplib_candidates(data), pglib_candidates(data), ml4co_candidates(data)], ignore_index=True)
    return assign_groups(df)


# -------------------------------------------------------------------------------------- groups and families
def miplib_prefix(name: str) -> str:
    """Leading letters of the name, lower case ("neos-1122047", "neos859080" -> "neos"); a name that starts with a
    digit keeps its first token ("30n20b8" -> "30n20b8", "50v-10" -> "50v")."""
    n = name.lower()
    m = re.match(r"[a-z]+", n)
    if m:
        return m.group(0)
    return re.split(r"[-_.]", n)[0]


def pglib_family(name: str) -> str:
    return name.split("__")[0]


def pglib_group(name: str) -> str:
    """Base case: the reserve variants of one ca day form a group; every rts_gmlc day is its own case."""
    return re.sub(r"_reserves_\d+$", "", name)


def ml4co_group(name: str) -> str:
    return "item_placement:" + name.rsplit("_", 1)[-1]


def assign_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Columns `group` (the split unit) and `family` (the 20 % cap unit). MIPLIB: union-find of prefix and the
    official Group (when not '–'); group = family = the component's smallest prefix."""
    df = df.copy()
    parent: dict[str, str] = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    mip = df["set"] == "miplib"
    for r in df[mip].itertuples(index=False):
        k = "p:" + miplib_prefix(r.instance)
        find(k)
        g = str(r.miplib_group or "").strip()
        if g and g not in ("–", "-", "—"):
            union(k, "g:" + g)
    comp: dict[str, list[str]] = {}
    for k in list(parent):
        comp.setdefault(find(k), []).append(k)
    label = {root: min(x[2:] for x in ks if x.startswith("p:")) for root, ks in comp.items()
             if any(x.startswith("p:") for x in ks)}
    groups, fams = [], []
    for r in df.itertuples(index=False):
        if r.set == "miplib":
            g = "miplib:" + label[find("p:" + miplib_prefix(r.instance))]
            groups.append(g)
            fams.append(g)
        elif r.set == "pglib_uc":
            groups.append("pglib:" + pglib_group(r.instance))
            fams.append("pglib:" + pglib_family(r.instance))
        else:
            groups.append(ml4co_group(r.instance))
            fams.append("ml4co:item_placement")
    df["group"], df["family"] = groups, fams
    return df


# ---------------------------------------------------------------------------------------------- screening
def screening_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """The SCREEN rows at T = 300 s, seeds 900/901, real runs only. P and the degeneracy flags are taken against the
    SCREENING reference (`pt_screen`, `ref_zero_screen`, `sign_crossing_screen` of build_runs: .solu, else the best
    SCREEN run), so the pool does not move when reference.csv is frozen after it (D-002 steps 3–4)."""
    mask = (runs["arm"] == SCREEN_ARM) & runs["seed"].isin([SEED1, SEED2]) & \
        np.isclose(runs["budget"].astype(float), T_SCREEN)
    if "synthetic" in runs.columns:
        mask &= ~runs["synthetic"].astype(bool)
    out = runs[mask].copy()
    for col, src in (("pt", "pt_screen"), ("ref_zero_instance", "ref_zero_screen"),
                     ("sign_crossing", "sign_crossing_screen")):
        if src in out.columns:
            out[col] = out[src]
    return out


def stage2_needed(runs: pd.DataFrame) -> list[str]:
    """Instances that need the seed-901 run, not yet run (D-002 v3.1 §1.2):
    - MIPLIB: seed-900 P(300) >= 0.05 (against the published .solu value) or no solution;
    - PGLib-UC and ML4CO: always — their stage-1 reference is the best SCREEN run, i.e. the run's own final value, so
      a P rule on stage 1 alone would measure time to its own result (review 2026-10-06, finding screening-F5)."""
    s = screening_runs(runs)
    s1 = s[s["seed"] == SEED1]
    done = set(s.loc[s["seed"] == SEED2, "instance"])
    path = s1["instance_path"] if "instance_path" in s1.columns else pd.Series("", index=s1.index)
    sets = s1["set"] if "set" in s1.columns else pd.Series("", index=s1.index)
    mip = [R.is_miplib(st, ip) for st, ip in zip(sets, path)]
    mip = pd.Series(mip, index=s1.index, dtype=bool)
    rule = (s1["pt"] >= STAGE2_P) | (~s1["t_first_event"].astype(bool))
    need = s1[(mip & rule) | ~mip]["instance"]
    return sorted(set(need) - done)


def screening_table(runs: pd.DataFrame, T: float = T_SCREEN) -> pd.DataFrame:
    """One row per screened instance: P per seed, mean P, feasible seeds, mean time to first feasible (censored at T),
    optimality, degeneracy flags, the exchange-window flag, SCIP's exit reason per seed (`exit_s900`, `exit_s901`;
    `solver_limit_stop` when a seed stopped at SCIP's own memory or node limit), and whether the screen is
    complete/clean."""
    s = screening_runs(runs)
    rows = []

    def exit_reason(r) -> str:
        if r is None:
            return ""
        v = getattr(r, "solver_exit_reason", "")
        return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)

    for name, g in s.groupby("instance", sort=True):
        g = g.sort_values("seed")
        by = {int(r.seed): r for r in g.itertuples(index=False)}
        r1, r2 = by.get(SEED1), by.get(SEED2)
        ps = [float(r.pt) for r in (r1, r2) if r is not None]
        feas = [bool(r.t_first_event) for r in (r1, r2) if r is not None]
        tf = [float(r.t_first) if bool(r.t_first_event) else T for r in (r1, r2) if r is not None]
        crashed = any(bool(getattr(r, "crashed", False)) for r in (r1, r2) if r is not None)
        ex1, ex2 = exit_reason(r1), exit_reason(r2)
        rows.append(dict(
            instance=name, set=str(g["set"].iloc[0]), p_s900=float(r1.pt) if r1 is not None else math.nan,
            p_s901=float(r2.pt) if r2 is not None else math.nan, p_mean=float(np.mean(ps)) if ps else math.nan,
            n_seeds=len(ps), feasible_seeds=int(sum(feas)),
            feasible_s900=bool(r1.t_first_event) if r1 is not None else False,
            t_first_mean=float(np.mean(tf)) if tf else math.nan,
            solved_optimal=bool(any(bool(getattr(r, "solved_optimal", False)) for r in (r1, r2) if r is not None)),
            ref_zero=bool(g["ref_zero_instance"].astype(bool).any()) if "ref_zero_instance" in g else False,
            sign_crossing=bool(g["sign_crossing"].astype(bool).any()) if "sign_crossing" in g else False,
            no_exchange_window=bool(g["no_exchange_window"].astype(bool).any())
            if "no_exchange_window" in g else False,
            exit_s900=ex1, exit_s901=ex2, solver_limit_stop=bool({ex1, ex2} & set(SOLVER_LIMIT_REASONS)),
            crashed=crashed, has_s900=r1 is not None, has_s901=r2 is not None))
    return pd.DataFrame(rows)


def classify(cands: pd.DataFrame, screen: pd.DataFrame) -> pd.DataFrame:
    """Join candidates with screening and assign a stratum or an exclusion reason."""
    df = cands.merge(screen.drop(columns=["set"]), on="instance", how="left")
    # P-degenerate (D-002 v3.1 §2): reference 0, or the held incumbent across 0 from the reference for >= 5 % of [0, T]
    # in screening. They stay out of the P-scored strata (they would leave the primary test); the feasibility
    # stratum may still take them.
    df["p_degenerate"] = df["ref_zero"].fillna(False).astype(bool) | df["sign_crossing"].fillna(False).astype(bool) \
        if "ref_zero" in df.columns else False
    strata, reasons = [], []
    for r in df.itertuples(index=False):
        st, why = "", ""
        if pd.isna(r.has_s900) or not bool(r.has_s900):
            why = "not screened (no seed-900 run)"
        elif bool(r.crashed):
            why = "screening run crashed"
        else:
            # stage 2 always for PGLib-UC / ML4CO (their stage-1 reference is the run's own best, D-002 v3.1 §1.2)
            need2 = (r.p_s900 >= STAGE2_P) or not bool(r.feasible_s900) or r.set != "miplib"
            if need2 and not bool(r.has_s901):
                why = "stage 2 (seed 901) missing"
            elif not need2:
                why = f"stage 1 P(300) < {STAGE2_P} with a solution: cannot reach a stratum"
            elif r.set == "miplib":
                if r.feasible_seeds < r.n_seeds:
                    st = "miplib_feas"
                elif bool(r.p_degenerate):
                    why = "P-degenerate: excluded from the P-scored strata"
                elif r.p_mean >= HARD_P:
                    st = "miplib_opt"
                else:
                    why = f"MIPLIB, feasible on both seeds, mean P(300) < {HARD_P}"
            elif r.set == "pglib_uc":
                if r.feasible_seeds < r.n_seeds:
                    st = "pglib"
                elif bool(r.p_degenerate):
                    why = "P-degenerate: excluded from the P-scored strata"
                elif r.p_mean >= HARD_P:
                    st = "pglib"
                else:
                    why = f"PGLib, mean P(300) < {HARD_P} with solutions on both seeds"
            elif bool(r.p_degenerate):
                why = "P-degenerate: excluded from the P-scored strata"
            else:
                st = "ml4co?"       # decided below: top half by mean P(300)
        strata.append(st)
        reasons.append(why)
    df["stratum"], df["excluded_reason"] = strata, reasons
    # ML4CO: top half (by mean P(300), ties by name) of the 60 candidates; unscreened ones count as not in the top half
    ml = df[(df["set"] == "ml4co_item_placement") & ~df["p_degenerate"].astype(bool)].copy()
    half = len(ml) // 2
    ranked = ml.assign(_p=ml["p_mean"].fillna(-1.0)).sort_values(["_p", "instance"], ascending=[False, True])
    top = set(ranked["instance"].head(half))
    for i in df.index[df["stratum"] == "ml4co?"]:
        if df.at[i, "instance"] in top:
            df.at[i, "stratum"] = "ml4co"
        else:
            df.at[i, "stratum"] = ""
            df.at[i, "excluded_reason"] = "ML4CO, not in the top half by mean P(300)"
    for c in ("exit_s900", "exit_s901"):
        df[c] = df[c].fillna("").astype(str) if c in df.columns else ""
    df["solver_limit_stop"] = df["solver_limit_stop"].fillna(False).astype(bool) \
        if "solver_limit_stop" in df.columns else False
    return df


# ----------------------------------------------------------------------------------------------- memory fit
def instance_shape(path: str | os.PathLike) -> dict:
    """n, m, nnz and the gene dtype's byte size of one instance (a parse with HiGHS; no solve)."""
    from itrea.ea.population import gene_dtype_for
    from itrea.problem import load

    p = load(path)
    return dict(n=p.n, m=p.m, nnz=p.nnz, gene_bytes=int(gene_dtype_for(p.lb, p.ub, p.isint, "auto").itemsize))


def shapes_for(paths: dict[str, str], cache: str | os.PathLike | None = None) -> pd.DataFrame:
    """Shapes of the given instances (name -> path), from `cache` (CSV: instance, n, m, nnz, gene_bytes) where
    present; parsed instances are appended to the cache."""
    known = pd.read_csv(cache) if cache and Path(cache).exists() else \
        pd.DataFrame(columns=["instance", "n", "m", "nnz", "gene_bytes"])
    have = set(known["instance"])
    new = [dict(instance=k, **instance_shape(v)) for k, v in sorted(paths.items()) if k not in have]
    out = pd.concat([known, pd.DataFrame(new)], ignore_index=True) if new else known
    out = out.sort_values("instance", kind="mergesort").reset_index(drop=True)
    if cache and new:
        out.to_csv(cache, index=False, lineterminator="\n")
    return out


def gpu_budget_bytes(gpu_mem_gb: float | None = None) -> int:
    from itrea.config import EAConfig

    return int((gpu_mem_gb if gpu_mem_gb is not None else EAConfig().gpu_mem_gb) * 1024 ** 3)


def fit_config(budget_bytes: int | None = None):
    """`EAConfig()` (the defaults: size_rule "memory", P ≤ 256) with gpu_mem_gb = the budget."""
    from itrea.config import EAConfig

    cfg = EAConfig()
    if budget_bytes is not None:
        cfg = cfg.with_overrides({"gpu_mem_gb": budget_bytes / 1024 ** 3})
    return cfg


def memory_fit(n: int, m: int, nnz: int, gene_bytes: int, cfg, p_fit: int = P_FIT) -> tuple[int, int, bool]:
    """(P, bytes, fits) of the EA's own memory plan for an instance of this shape: `engine.plan_memory` on a
    shape-only stub (it reads only n, m, nnz). fits = the plan fits AND P >= p_fit — `Engine.prepare` refuses a plan
    that does not fit, and the pool asks for P >= 32 on top (D-002 §2)."""
    from types import SimpleNamespace

    from itrea.ea.engine import plan_memory

    gb = int(gene_bytes)
    if gb not in GENE_DTYPE_BY_BYTES:
        raise ValueError(f"gene_bytes={gene_bytes}: not a gene dtype of population.gene_dtype_for (1, 4 or 8)")
    plan = plan_memory(SimpleNamespace(n=int(n), m=int(m), nnz=int(nnz)), cfg, GENE_DTYPE_BY_BYTES[gb])
    fits = bool(plan.fits and plan.P >= p_fit)
    if fits and cfg.ranks.x > 0 and math.floor(cfg.ranks.x * plan.P + 1e-12) < 1:
        fits = False                                             # Engine.prepare's Rank 0 slot check (moot at P >= 32)
    return int(plan.P), int(plan.total_bytes), fits


# ------------------------------------------------------------------------------------------------- selection
def _sort_key(df: pd.DataFrame, stratum: str) -> pd.DataFrame:
    if stratum == "miplib_feas":
        return df.sort_values(["t_first_mean", "instance"], ascending=[False, True], kind="mergesort")
    return df.sort_values(["p_mean", "instance"], ascending=[False, True], kind="mergesort")


def family_caps(strata=STRATA, family_cap: int | None = None, tuning_family_cap: int | None = None) -> dict:
    """{"test": ⌊0.2 · test total⌋, "tuning": ⌊0.2 · tuning total⌋} unless given (6 and 3 for the D-002 quotas)."""
    def share(k: int) -> int:
        return int(math.floor(FAMILY_CAP_SHARE * k + 1e-9))

    return dict(test=share(sum(s.test for s in strata)) if family_cap is None else int(family_cap),
                tuning=share(sum(s.tuning for s in strata)) if tuning_family_cap is None else int(tuning_family_cap))


def assign_sides(t: pd.DataFrame, strata=STRATA, cap: int = 6, tuning_cap: int | None = None) -> dict[str, str]:
    """Side of every group that has eligible, fitting instances (phase 1 of the selection).

    Groups that hold an `rts_gmlc` case go to test (never tuning). The others are taken largest first (eligible
    instances, then name) and each goes to the side whose remaining need it covers better: a side's target in a
    stratum is its quota's share (test 2/3, tuning 1/3) of that stratum's usable supply, and the need a group covers
    is Σ_strata min(usable instances, target − usable assigned) / quota. Ties go to the side that is less filled by
    ALL its assigned instances.

    "Usable" applies the side's family cap (`cap` on test, `tuning_cap` on tuning; None = ⌊0.2 · tuning total⌋): a
    family counts at most up to the cap on a side, shared over the strata in `strata` order (phase 2's pick order),
    but only where the cap can bind — when the family's strata quotas on that side sum to more than the cap (MIPLIB
    families; not `ca` on tuning, whose PGLib quota 3 is the cap, nor ML4CO). So a family larger than a cap neither
    fills a side's target with instances it can never pick (with no tuning cap the 55-instance `neos` group went to
    tuning and took 5–10 of its 10 MIPLIB slots in synthetic screening draws on the real candidate table) nor wins a
    side it can use only part of; and the targets are shares
    of the usable supply per side, or a capped side would draw too many families and starve the other.
    Deterministic, and it keeps both sides supplied in every stratum."""
    caps = dict(test=cap, tuning=family_caps(strata)["tuning"] if tuning_cap is None else tuning_cap)
    el = t[t["stratum"] != ""]
    fit = el[el["fits"].astype(bool)]
    names = [s.name for s in strata]
    quota = {"test": {s.name: s.test for s in strata}, "tuning": {s.name: s.tuning for s in strata}}

    def usable(side: str, counts: dict, used: int = 0) -> dict[str, int]:
        """Instances of one family's `counts` (stratum -> fitting instances) that the side's cap leaves pickable."""
        binds = sum(quota[side][n] for n in names if counts.get(n, 0)) > caps[side]
        room = max(0, caps[side] - used) if binds else math.inf
        out = {}
        for n in names:
            u = min(int(counts.get(n, 0)), room)
            if u:
                out[n] = int(u)
                room -= u
        return out

    supply = {side: {n: 0 for n in names} for side in quota}
    for _, ff in fit.groupby("family", sort=True):
        fc = ff["stratum"].value_counts().to_dict()
        for side in quota:
            for n, u in usable(side, fc).items():
                supply[side][n] += u
    target = {side: {n: quota[side][n] / max(1, quota["test"][n] + quota["tuning"][n]) * supply[side][n]
                     for n in names} for side in quota}
    cnt = {side: {n: 0 for n in names} for side in quota}         # usable instances assigned
    raw = {side: {n: 0 for n in names} for side in quota}         # all fitting instances assigned (the tie-break)
    fam_on: dict[str, dict[str, int]] = {"test": {}, "tuning": {}}  # usable instances per family
    groups = []
    for g, gg in el.groupby("group", sort=True):            # a group without a fitting instance still gets a side,
        counts = gg.loc[gg["fits"].astype(bool), "stratum"].value_counts().to_dict()   # so its skips are listed once
        forced = bool(gg["instance"].map(lambda x: pglib_family(str(x)) == "rts_gmlc").any())
        groups.append((not forced, -sum(counts.values()), g, str(gg["family"].iloc[0]), counts))
    side_of: dict[str, str] = {}
    for not_forced, _, g, fam, counts in sorted(groups):
        use = {side: usable(side, counts, fam_on[side].get(fam, 0)) for side in quota}
        if not not_forced:
            side = "test"
        else:
            def cover(side: str) -> float:
                return sum(min(u, max(0.0, target[side][n] - cnt[side][n])) / max(1, quota[side][n])
                           for n, u in use[side].items())

            def fill(side: str) -> float:
                return sum(raw[side][n] / target[side][n] for n in names if target[side][n] > 0)

            ct, cu = cover("test"), cover("tuning")
            side = "test" if ct > cu else "tuning" if cu > ct else ("test" if fill("test") <= fill("tuning")
                                                                     else "tuning")
        side_of[g] = side
        for n, u in use[side].items():
            cnt[side][n] += u
            fam_on[side][fam] = fam_on[side].get(fam, 0) + u
        for n, e in counts.items():
            raw[side][n] += e
    return side_of


def select_pool(table: pd.DataFrame, strata=STRATA, budget_bytes: int | None = None, p_fit: int = P_FIT,
                family_cap: int | None = None, tuning_family_cap: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Deterministic pool selection (module docstring). Phase 1 puts every group on one side (`assign_sides`);
    phase 2 takes every k-th instance of each (stratum, side) list, sorted hardest first, with the next instance of the
    same bucket as the replacement for a pick that does not fit in GPU memory or would break its side's family cap.
    `table`: classify() output plus shapes (n, m, nnz, gene_bytes — all four: the fit is the EA's memory plan, and a
    table without m / nnz is refused rather than judged on the population alone). Returns (pool rows, report)."""
    missing = [c for c in ("n", "m", "nnz", "gene_bytes") if c not in table.columns]
    if missing:
        raise ValueError(f"select_pool needs the instance shapes {missing} (instance_shape / shapes_for)")
    from itrea.ea import engine as E

    cfg = fit_config(budget_bytes)
    caps = family_caps(strata, family_cap, tuning_family_cap)
    cap = caps["test"]
    t = table.copy()
    t["pop_bytes_p32"] = (p_fit * t["n"].astype(float) * t["gene_bytes"].astype(float))
    plans = [(math.nan, math.nan, False) if any(pd.isna(v) for v in r)   # no shape (not eligible): not fitted
             else memory_fit(r.n, r.m, r.nnz, r.gene_bytes, cfg, p_fit)
             for r in t[["n", "m", "nnz", "gene_bytes"]].itertuples(index=False)]
    t["plan_p"] = [x[0] for x in plans]
    t["plan_bytes"] = [x[1] for x in plans]
    t["fits"] = pd.Series([bool(x[2]) for x in plans], index=t.index, dtype=bool)
    side_of = assign_sides(t, strata, cap, caps["tuning"])
    t["side"] = t["group"].map(side_of).fillna("")
    fam_on: dict[str, dict[str, int]] = {"test": {}, "tuning": {}}
    chosen, report = [], dict(
        shortfall={}, replacements=[], unfit=[], family_cap=cap, tuning_family_cap=caps["tuning"],
        sides=dict(sorted(side_of.items())),
        fit_config_hash=cfg.hash(),
        fit_rule=(f"engine.plan_memory(n, m, nnz, gene dtype; EAConfig() defaults, config hash {cfg.hash()}, "
                  f"gpu_mem_gb = {cfg.gpu_mem_gb:g} GiB, size_rule {cfg.population.size_rule}, "
                  f"P <= {cfg.population.size}; engine HEADROOM {E.HEADROOM:g}, FRONTIER_OFFER {E.FRONTIER_OFFER}) "
                  f"fits with P >= {p_fit}; pop_bytes_p32 = {p_fit} x n x gene bytes is informative only"))
    for st in strata:
        report["unfit"] += sorted(str(x) for x in t.loc[(t["stratum"] == st.name) & ~t["fits"], "instance"])
        for side, Q in (("test", st.test), ("tuning", st.tuning)):
            el = _sort_key(t[(t["stratum"] == st.name) & (t["side"] == side)], st.name).reset_index(drop=True)
            N, room = len(el), Q
            taken: set[int] = set()

            def try_take(i: int, bucket: int, replaces: str) -> bool:
                nonlocal room
                r = el.iloc[i]
                if not bool(r["fits"]):
                    return False
                if fam_on[side].get(r["family"], 0) >= caps[side]:
                    return False
                room -= 1
                taken.add(i)
                fam_on[side][r["family"]] = fam_on[side].get(r["family"], 0) + 1
                chosen.append(dict(r.to_dict(), side=side, order=len(taken), bucket=bucket, replaces=replaces))
                return True

            for b in range(min(Q, N)):
                lo, hi = (b * N) // Q, max(((b + 1) * N) // Q, (b * N) // Q + 1)
                first = str(el.iloc[lo]["instance"])
                for i in range(lo, hi):
                    if i not in taken and try_take(i, b, "" if i == lo else first):
                        if i != lo:
                            report["replacements"].append(dict(stratum=st.name, side=side, skipped=first,
                                                               taken=str(el.iloc[i]["instance"])))
                        break
            for i in range(N):                                # slots still open: the rest in sorted order
                if room <= 0:
                    break
                if i not in taken:
                    try_take(i, -1, "")
            if room > 0:
                report["shortfall"].setdefault(st.name, {})[side] = room
    pool = pd.DataFrame(chosen)
    if len(pool):
        pool["feasibility_endpoint"] = pool["stratum"] == "miplib_feas"
        for c in POOL_COLUMNS:
            if c not in pool.columns:
                pool[c] = np.nan
        strat_order = {s.name: k for k, s in enumerate(strata)}
        pool = pool[POOL_COLUMNS].assign(_s=pool["stratum"].map(strat_order)) \
            .sort_values(["_s", "side", "order"], kind="mergesort").drop(columns="_s").reset_index(drop=True)
    report["family_counts"] = {side: dict(sorted(v.items())) for side, v in fam_on.items()}
    report["family_test_counts"] = report["family_counts"]["test"]
    report["solver_limit_stops"] = [] if not len(pool) else [
        dict(instance=str(r["instance"]), side=str(r["side"]), stratum=str(r["stratum"]),
             exit_s900=str(r["exit_s900"]), exit_s901=str(r["exit_s901"]))
        for _, r in pool[pool["solver_limit_stop"].fillna(False).astype(bool)].iterrows()]
    return pool, report


def write_pool(pool: pd.DataFrame, table: pd.DataFrame, out: str | os.PathLike,
               candidates_out: str | os.PathLike | None = None) -> str:
    """Write hard_pool.csv (and the candidate table with every exclusion reason). Returns the pool's sha256."""
    out = Path(out)
    pool.to_csv(out, index=False, float_format="%.12g", lineterminator="\n")
    if candidates_out:
        cols = [c for c in ("instance", "set", "group", "family", "stratum", "excluded_reason", "p_s900", "p_s901",
                            "p_mean", "n_seeds", "feasible_seeds", "t_first_mean", "p_degenerate", "exit_s900",
                            "exit_s901", "solver_limit_stop", "n", "m", "nnz", "gene_bytes", "fits")
                if c in table.columns]
        table.sort_values(["set", "instance"], kind="mergesort")[cols].to_csv(
            candidates_out, index=False, float_format="%.12g", lineterminator="\n")
    return hashlib.sha256(out.read_bytes()).hexdigest()


def freeze(runs: pd.DataFrame, data: Path = DATA, shapes: pd.DataFrame | None = None,
           shapes_cache: str | os.PathLike | None = None, cands: pd.DataFrame | None = None,
           budget_bytes: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """candidates -> screening table -> strata -> shapes -> selection. Returns (pool, candidate table, report)."""
    cands = candidates(data) if cands is None else cands
    table = classify(cands, screening_table(runs))
    if shapes is None:
        elig = table[table["stratum"] != ""]
        shapes = shapes_for(dict(zip(elig["instance"], elig["path"])), shapes_cache)
    table = table.merge(shapes[["instance", "n", "m", "nnz", "gene_bytes"]], on="instance", how="left")
    pool, report = select_pool(table, budget_bytes=budget_bytes)
    return pool, table, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="hard pool: screening stage 2 list, or the pool freeze (D-002 §2)")
    ap.add_argument("cmd", choices=["candidates", "stage2", "select"])
    ap.add_argument("--runs", default=str(DATASETS / "runs.parquet"))
    ap.add_argument("--out", default=str(DATASETS / "hard_pool.csv"))
    ap.add_argument("--candidates-out", default=str(DATASETS / "hard_pool_candidates.csv"))
    ap.add_argument("--shapes", default=str(DATASETS / "instance_shapes.csv"))
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    if a.cmd == "candidates":                           # the stage-1 list (CSV on stdout) for the job builder
        candidates()[["instance", "set", "path", "group", "family"]].to_csv(sys.stdout, index=False,
                                                                            lineterminator="\n")
        return 0
    if not Path(a.runs).exists():
        print(f"{a.runs} does not exist: run the screening pass and itrea.analysis.build_runs first")
        return 2
    runs = pd.read_parquet(a.runs)
    if a.cmd == "stage2":
        for n in stage2_needed(runs):
            print(n)
        return 0
    pool, table, rep = freeze(runs, shapes_cache=a.shapes)
    sha = write_pool(pool, table, a.out, a.candidates_out)
    rep_path = Path(a.out).with_name(Path(a.out).stem + "_report.json")
    rep_path.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    print(f"{len(pool)} instances -> {a.out} (sha256 {sha[:12]}); shortfall {rep['shortfall'] or 'none'}; "
          f"replacements {len(rep['replacements'])}; unfit {len(rep['unfit'])}; "
          f"solver-limit stops {len(rep['solver_limit_stops'])}; report {rep_path.name}")
    if rep["shortfall"]:
        print("WARNING: quota shortfall (fewer eligible instances than slots) — record it in the pool-freeze note")
    return 0


if __name__ == "__main__":
    sys.exit(main())
