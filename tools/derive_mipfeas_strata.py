"""Derive D-021's instance sets (ITX-1) and write `datasets/mipfeas_strata.csv` — the file prereg-v2 hashes.

Inputs (read only): MIPfeas's 233 (data/miplib/benchmark-v2.test minus its =inf=), the development and class-probe
tables (datasets/dev_set.csv, datasets/classprobe_set.csv), GAMS's archive (data/mipfeas/gams-2026-09-16:
primalintegral.csv, optimal_objective.csv, the SCIP 10 traces of the instances without a screening run), MIPLIB's solu
file and the run table with the screening (`--runs`: the main checkout's datasets/runs.parquet; the screening's set
screen1). The rules are in `itrea.analysis.itx_strata`. Prints every count beside D-021's and each difference
(reported, never forced), and the file's sha256. CPU only, seconds; run it pinned and single-threaded:

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 taskset -c 15-19 python tools/derive_mipfeas_strata.py \
        --runs ../i-trea-ii/datasets/runs.parquet [--out datasets/mipfeas_strata.csv] [--check]

`--check` re-derives and compares with the existing file (exit 1 if it differs), writing nothing.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import pandas as pd

from itrea import paths
from itrea.analysis import itx_strata as S
from itrea.analysis.mipfeas import GAMS_DIR, read_pi, read_z
from itrea.benchmarks import references as R
from itrea.experiments import make_jobs as MJ


def build(runs_path: str | Path, gams_dir: str | Path = GAMS_DIR, data: Path | None = None) -> pd.DataFrame:
    data = Path(data or paths.DATA)
    names = MJ.mipfeas_233(data)
    pi, z = read_pi(gams_dir), read_z(gams_dir)
    p300 = S.screening_p300(pd.read_parquet(runs_path), names)
    fallback = {n: S.scip10_p300(gams_dir, n, z[n]) for n in names if n not in p300}
    solu = R.read_solu(data / "miplib" / "solu.txt")
    return S.derive(names, MJ.development_names(), pi, z, p300, fallback, solu)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True, help="the run table holding the screening (set screen1)")
    ap.add_argument("--gams-dir", default=str(GAMS_DIR))
    ap.add_argument("--data", type=Path, default=paths.DATA)
    ap.add_argument("--out", type=Path, default=S.STRATA_CSV)
    ap.add_argument("--check", action="store_true", help="compare with --out instead of writing it")
    a = ap.parse_args(argv)
    df = build(a.runs, a.gams_dir, a.data)
    got = S.counts(df)
    print("count                               derived  D-021")
    diffs = []
    for k, v in got.items():
        want = S.D021_COUNTS.get(k)
        mark = "" if want is None or want == v else "   <- differs"
        if mark:
            diffs.append(f"{k}: derived {v}, D-021 {want}")
        print(f"{k:<36}{v:>7}  {'' if want is None else want}{mark}")
    fb = df[df["w_source"] == "gams-scip10"]
    for _, r in fb.iterrows():
        print(f"W_SCIP from GAMS's SCIP 10 trace: {r['name']} P(300) {r['p300_scip']:.4f} -> "
              f"{'in' if r['W_SCIP'] else 'not in'} W_SCIP")
    print("panel: " + ", ".join(f"{r['name']} ({r['panel_group']})" for _, r in df[df['panel']].iterrows()))
    print("E-sample: " + ", ".join(f"{r['name']} (q{r['e_quartile']})" for _, r in df[df['e_sample']].iterrows()))
    if a.check:
        with tempfile.TemporaryDirectory() as td:
            sha = S.write_strata(df, Path(td) / "strata.csv")
            same = a.out.exists() and (Path(td) / "strata.csv").read_bytes() == a.out.read_bytes()
        print(f"re-derived sha256 {sha}; {a.out} {'is identical' if same else 'DIFFERS'}")
        return 0 if same else 1
    sha = S.write_strata(df, a.out)
    print(f"{a.out}: {len(df)} rows, sha256 {sha}")
    if diffs:
        print("differences from D-021's counts (reported, not forced): " + "; ".join(diffs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
