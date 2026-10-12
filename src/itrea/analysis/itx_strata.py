"""D-021 (ITX-1): the instance sets of the external benchmark, fixed before any v1 run in
`datasets/mipfeas_strata.csv` (hashed in prereg-v2) — derived by `tools/derive_mipfeas_strata.py`, read here by the job
writers (`make-jobs itx-v1 | itx-a0`) and the analysis (`itrea itx-report`).

Per instance of MIPfeas's 233 (`make_jobs.mipfeas_233`: MIPLIB 2017's benchmark-v2.test minus its 7 =inf=):
- `dev` / `unseen`: the development (D-010) and class-probe (D-014) instances among the 233 (15 of the 17 in
  datasets/dev_set.csv and datasets/classprobe_set.csv) are not unseen (R2: run and reported, never decided on);
- `p_cuopt`: GAMS's published `pt_gams` of cuOpt 26.08 on the GB10 (`primalintegral.csv`, column cuopt-gb10; 2 = no
  solution); **H_cu** = p_cuopt >= 0.1 or no solution (R3), `H_cu_p05` = p_cuopt >= 0.5 or none (reported beside);
- `p300_scip`: Berthold's P(300) of our screening (screen1: arm SCREEN, seed 900, 300 s, one X925 core; the one
  completed run per instance in datasets/runs.parquet), `w_source` "screen1"; the two instances without a completed
  screening run — satellites2-40 (its screening run failed) and supportcase22 (never a candidate) — are classed by
  GAMS's SCIP 10 row instead: Berthold P(300) of GAMS's SCIP 10 trace (Intel 285K), against GAMS's z*, `w_source`
  "gams-scip10"; **W_SCIP** = p300_scip >= 0.5 (D-013); `p_scip10` = GAMS's published `pt_gams` of SCIP 10 (context);
- `W_only` = W_SCIP and not H_cu; `E` = neither (both flags are set for every instance; the decision sets read them on
  the unseen ones);
- `does_not_fit`: M3's fixed list of instances cuOpt cannot run at F's caps (eilA101-2, nw04, supportcase6; excluded
  from both arms); `decide` = unseen and not on the list (215);
- `ref_kind` (MIPLIB solu: opt | best_known), `z_star` (GAMS's optimal_objective.csv) and `r12` (R12's stop applies:
  ref_kind opt);
- the fixed samples (R4), one `numpy.default_rng(20261012)` stream in this order: the **E-sample** — E unseen and not on
  the list, in primalintegral.csv's row order (alphabetical), stable-sorted by p_cuopt and cut into 4 quartiles with
  `numpy.array_split` (`e_quartile` 1–4), 10 per quartile by `rng.choice(quartile, 10, replace=False)`, each pick
  sorted (D-020's sample pattern) — then the **repeatability panel**: H_cu unseen not on the list, in the same order,
  stable-sorted by p_cuopt; 10 of those with p_cuopt >= 0.5 or none (`panel_group` "p05"), then 10 of those with
  0.1 <= p_cuopt < 0.5 ("p01"), each by `rng.choice(group, 10, replace=False)`, sorted;
- `tier` (R31): T1 = H_cu unseen; T2 = W-only unseen + the E-sample; T3 = the rest of E unseen + the development
  instances; "" = on the does-not-fit list; `t4` / `t5` = the panel (T4: A9 and A6F at seed 2001; T5: live A2F).

D-021's counts (before M3's list: H_cu 75 / 65 unseen, H_cu_p05 34 / 30, W_SCIP 74 / 62 unseen, W-only 17, E 136;
after it H_cu-unseen 64, the decision set 215) are checked by the tool, which reports any difference instead of
forcing it. The CSV is deterministic (sorted by name, fixed columns, %.17g floats, booleans as 0 / 1) and its sha256 is
printed and named in prereg-v2.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.paths import DATASETS

STRATA_CSV = DATASETS / "mipfeas_strata.csv"
STRATA_SEED = 20261012
DOES_NOT_FIT = ("eilA101-2", "nw04", "supportcase6")      # D-021 M3 (2026-10-11): cuOpt's cap at F, both arms out
H_CU_MIN, H_CU_P05, W_MIN = 0.1, 0.5, 0.5
E_PER_QUARTILE, PANEL_PER_GROUP = 10, 10
NO_SOLUTION = 2.0
SCIP10_RUN = "intel-core-ultra-9-285k_scip-soplex-10-0-1"
TIERS = ("T1", "T2", "T3")
COLUMNS = ("name", "instance", "dev", "unseen", "ref_kind", "z_star", "r12", "p_cuopt", "H_cu", "H_cu_p05",
           "p300_scip", "w_source", "p_scip10", "W_SCIP", "W_only", "E", "does_not_fit", "decide", "e_quartile",
           "e_sample", "panel", "panel_group", "tier", "t4", "t5")
BOOL_COLUMNS = ("dev", "unseen", "r12", "H_cu", "H_cu_p05", "W_SCIP", "W_only", "E", "does_not_fit", "decide",
                "e_sample", "panel", "t4", "t5")
# D-021's stated counts (R3, R31, M3 addendum) — the tool compares the derived ones with these
D021_COUNTS = {"the 233": 233, "unseen": 218, "H_cu": 75, "H_cu unseen": 65, "H_cu_p05": 34, "H_cu_p05 unseen": 30,
               "W_SCIP": 74, "W_SCIP unseen": 62, "W-only unseen": 17, "E unseen": 136,
               "H_cu unseen after the list": 64, "decide (unseen after the list)": 215}


def screening_p300(runs: pd.DataFrame, names) -> dict[str, float]:
    """Berthold's P(300) of the screening run per instance (screen1, SCREEN, seed 900, 300 s, completed and not
    block-void); an instance with two such runs is an error (never one picked)."""
    r = runs[(runs["set"].astype(str) == "screen1") & (runs["arm"].astype(str) == "SCREEN") &
             (runs["seed"].astype(int) == 900) & np.isclose(runs["budget"].astype(float), 300.0) &
             (runs["status"].astype(str) == "completed")]
    if "block_void" in r.columns:
        r = r[~r["block_void"].astype(bool)]
    if "synthetic" in r.columns:
        r = r[~r["synthetic"].astype(bool)]
    r = r[r["instance"].astype(str).isin(set(names))]
    dup = r["instance"][r["instance"].duplicated()].tolist()
    if dup:
        raise ValueError(f"several completed screening runs for {sorted(set(dup))[:5]}")
    return {str(n): float(p) for n, p in zip(r["instance"], r["pt"])}


def scip10_p300(gams_dir: str | Path, name: str, z: float) -> float:
    """Berthold's P(300) of GAMS's SCIP 10 trace (Intel 285K) of one instance, against GAMS's z* (minimisation, as
    every MIPfeas instance)."""
    from itrea.analysis import metrics as M
    from itrea.analysis.mipfeas import read_gams_trc, trc_path, trc_points
    steps = M.best_steps(trc_points(read_gams_trc(trc_path(gams_dir, name, SCIP10_RUN))), 1)
    return float(M.primal_integral(steps, z, 1, 300.0).value)


def derive(names233, dev_names, pi: pd.DataFrame, z: dict[str, float], p300: dict[str, float],
           p300_fallback: dict[str, float], solu: dict, instance_paths: dict[str, str] | None = None,
           does_not_fit=DOES_NOT_FIT, seed: int = STRATA_SEED) -> pd.DataFrame:
    """The strata table (module docstring). `pi`: primalintegral.csv indexed by name, in the file's row order (columns
    cuopt-gb10, scip); `p300`: the screening's P(300); `p300_fallback`: GAMS SCIP 10's P(300) for instances without a
    screening run (every instance must have one of the two, else ValueError); `solu`: MIPLIB's solu entries."""
    names233 = list(names233)
    the233 = set(names233)
    if len(the233) != 233:
        raise ValueError(f"{len(the233)} names, not MIPfeas's 233")
    missing = sorted(the233 - set(pi.index))
    if missing:
        raise ValueError(f"no published P for {missing[:5]}")
    order = [n for n in pi.index if n in the233]           # primalintegral.csv's row order
    devs = set(dev_names) & the233
    rows = {}
    for n in order:
        pc = float(pi.loc[n, "cuopt-gb10"])
        if n in p300:
            pw, src = p300[n], "screen1"
        elif n in p300_fallback:
            pw, src = p300_fallback[n], "gams-scip10"
        else:
            raise ValueError(f"{n}: no screening run and no SCIP 10 fallback")
        e = solu.get(n)
        kind = getattr(e, "kind", "") if e is not None else ""
        h = pc >= H_CU_MIN or pc >= NO_SOLUTION
        w = pw >= W_MIN
        rows[n] = dict(name=n, instance=(instance_paths or {}).get(n, f"data/miplib/inst/{n}.mps.gz"),
                       dev=n in devs, unseen=n not in devs, ref_kind=kind, z_star=float(z[n]) if n in z else math.nan,
                       r12=kind == "opt", p_cuopt=pc, H_cu=h, H_cu_p05=pc >= H_CU_P05, p300_scip=pw, w_source=src,
                       p_scip10=float(pi.loc[n, "scip"]) if "scip" in pi.columns else math.nan, W_SCIP=w,
                       W_only=w and not h, E=not (w or h), does_not_fit=n in set(does_not_fit),
                       decide=(n not in devs) and n not in set(does_not_fit), e_quartile=0, e_sample=False,
                       panel=False, panel_group="", tier="", t4=False, t5=False)
    rng = np.random.default_rng(seed)

    def by_p(sel) -> np.ndarray:
        """The selected names in the file's order, stable-sorted by GAMS's cuOpt P."""
        lst = [n for n in order if sel(rows[n])]
        return np.array(sorted(lst, key=lambda n: rows[n]["p_cuopt"]), dtype=object) if lst else np.array([], object)
    e_pool = by_p(lambda r: r["E"] and r["decide"])
    for q, part in enumerate(np.array_split(e_pool, 4), start=1):
        for n in part:
            rows[n]["e_quartile"] = q
        if len(part) < E_PER_QUARTILE:
            raise ValueError(f"E quartile {q} has {len(part)} instances, fewer than {E_PER_QUARTILE}")
        for n in sorted(rng.choice(part, E_PER_QUARTILE, replace=False).tolist()):
            rows[n]["e_sample"] = True
    h_pool = by_p(lambda r: r["H_cu"] and r["decide"])
    groups = (("p05", lambda n: rows[n]["p_cuopt"] >= H_CU_P05),
              ("p01", lambda n: H_CU_MIN <= rows[n]["p_cuopt"] < H_CU_P05))
    for g, sel in groups:
        grp = np.array([n for n in h_pool if sel(n)], dtype=object)
        if len(grp) < PANEL_PER_GROUP:
            raise ValueError(f"panel group {g} has {len(grp)} instances, fewer than {PANEL_PER_GROUP}")
        for n in sorted(rng.choice(grp, PANEL_PER_GROUP, replace=False).tolist()):
            rows[n].update(panel=True, panel_group=g, t4=True, t5=True)
    for r in rows.values():
        if r["does_not_fit"]:
            continue
        if r["dev"]:
            r["tier"] = "T3"
        elif r["H_cu"]:
            r["tier"] = "T1"
        elif r["W_only"] or r["e_sample"]:
            r["tier"] = "T2"
        else:
            r["tier"] = "T3"
    df = pd.DataFrame([rows[n] for n in sorted(rows)], columns=list(COLUMNS))
    return df


def counts(df: pd.DataFrame) -> dict[str, int]:
    """The set sizes D-021 states (D021_COUNTS' keys) and the tiers', from a strata table."""
    b = {c: df[c].astype(bool) for c in BOOL_COLUMNS}
    un, fit = b["unseen"], ~b["does_not_fit"]
    out = {"the 233": len(df), "unseen": int(un.sum()), "H_cu": int(b["H_cu"].sum()),
           "H_cu unseen": int((b["H_cu"] & un).sum()), "H_cu_p05": int(b["H_cu_p05"].sum()),
           "H_cu_p05 unseen": int((b["H_cu_p05"] & un).sum()), "W_SCIP": int(b["W_SCIP"].sum()),
           "W_SCIP unseen": int((b["W_SCIP"] & un).sum()), "W-only unseen": int((b["W_only"] & un).sum()),
           "E unseen": int((b["E"] & un).sum()), "H_cu unseen after the list": int((b["H_cu"] & un & fit).sum()),
           "decide (unseen after the list)": int(b["decide"].sum()),
           "E unseen after the list": int((b["E"] & un & fit).sum()), "E-sample": int(b["e_sample"].sum()),
           "panel": int(b["panel"].sum()), "does not fit": int(b["does_not_fit"].sum()),
           "development among the 233": int(b["dev"].sum()), "R12-eligible (=opt=)": int(b["r12"].sum()),
           "W_SCIP from GAMS's SCIP 10": int(((df["w_source"] == "gams-scip10")).sum())}
    for t in TIERS:
        out[t] = int((df["tier"] == t).sum())
    for g in ("p05", "p01"):
        out[f"panel {g}"] = int((df["panel_group"] == g).sum())
    return out


def write_strata(df: pd.DataFrame, path: str | os.PathLike = STRATA_CSV) -> str:
    """Write the table deterministically (booleans 0 / 1, %.17g floats); returns the file's sha256."""
    out = df[list(COLUMNS)].copy()
    for c in BOOL_COLUMNS:
        out[c] = out[c].astype(bool).astype(int)
    out = out.sort_values("name", kind="mergesort").reset_index(drop=True)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    out.to_csv(tmp, index=False, float_format="%.17g", lineterminator="\n")
    os.replace(tmp, p)
    return file_sha256(p)


def file_sha256(path: str | os.PathLike) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_strata(path: str | os.PathLike = STRATA_CSV) -> pd.DataFrame:
    """The committed strata table (column `name`), booleans as bool, floats exactly as written (round trip)."""
    df = pd.read_csv(path, dtype={"name": str, "instance": str, "ref_kind": str, "w_source": str,
                                  "panel_group": str, "tier": str}, keep_default_na=False,
                     float_precision="round_trip",
                     na_values={c: ["", "nan"] for c in ("z_star", "p_cuopt", "p300_scip", "p_scip10")})
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for c in BOOL_COLUMNS:
        df[c] = df[c].astype(int).astype(bool)
    df["e_quartile"] = df["e_quartile"].astype(int)
    return df[list(COLUMNS)]


def names_where(df: pd.DataFrame, **flags) -> list[str]:
    """Names (sorted) whose boolean columns equal the given values, e.g. names_where(df, decide=True, H_cu=True)."""
    m = pd.Series(True, index=df.index)
    for k, v in flags.items():
        m &= (df[k] == v)
    return sorted(df.loc[m, "name"].tolist())
