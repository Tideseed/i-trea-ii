"""GAMS's MIPfeas archive (vault D-020): the trace reader, the published per-instance integrals and reference values,
and D-020's calibration sample. GAMS's integral itself is ported in `itrea.analysis.metrics.gams_integral`.

The archive `data/mipfeas/gams-2026-09-16/` (Berk's yes, 2026-10-11; gitignored like all of data/) holds GAMS's
scoring script `create_primalintegral.py`, `optimal_objective.csv` (their z*), `primalintegral.csv` (their integral
per instance and solver; column `cuopt-gb10` = cuOpt 26.08 on the GB10) and a trace per instance, machine and solver,
`trc_files/<name>_<machine>_<solver>.trc`. The port scored on `read_gams_trc` + `metrics.best_steps` equals their
`calculate_primal_integral` on every one of the archive's 3,495 traces (2026-10-11; the GB10 26.08 ones are a test).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from itrea.analysis import metrics as M
from itrea.paths import DATA

GAMS_DIR = DATA / "mipfeas" / "gams-2026-09-16"
GAMS_COL = "cuopt-gb10"                    # primalintegral.csv: cuOpt 26.08 on the GB10 (their published row)
GB10_2608 = "gb10_cuopt-26-08"             # the same run's traces
TRC_COLUMNS = ["lineNum", "seriesID", "node", "seconds", "bestFound", "bestBound"]
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
