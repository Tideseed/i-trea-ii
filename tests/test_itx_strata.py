"""D-021's instance sets (`itrea.analysis.itx_strata`, `tools/derive_mipfeas_strata.py`): the rules on synthetic
inputs, the committed `datasets/mipfeas_strata.csv` (its counts against D-021, its sha256, a round trip) and — where the
screening's run table and GAMS's archive are on disk — a re-derivation byte for byte."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itrea import paths
from itrea.analysis import itx_strata as S
from itrea.benchmarks.references import SoluEntry

STRATA_SHA256 = "d2ad90ee040e4b6a046cda9abe67c6add86b48f64547288c2edc292a0c203842"   # 2026-10-12, D-021 phase 2


def _synthetic(n=233, seed=1):
    """233 fake instances: GAMS cuOpt P spread over [0, 2], screening P(300), two without a screening run, five
    development instances, one =best= reference."""
    rs = np.random.default_rng(seed)
    names = [f"i{k:03d}" for k in range(n)]
    pc = np.round(rs.choice([0.001, 0.01, 0.05, 0.08, 0.2, 0.3, 0.6, 2.0], size=n), 6) + rs.random(n) * 1e-4
    pc = np.minimum(pc, 2.0)
    pi = pd.DataFrame({"cuopt-gb10": pc, "scip": rs.random(n)}, index=pd.Index(names, name="mps_name"))
    pw = rs.random(n)
    p300 = {nm: float(p) for nm, p in zip(names, pw) if nm not in ("i007", "i100")}
    fallback = {"i007": 0.9, "i100": 0.1}
    solu = {nm: SoluEntry(nm, "best_known" if nm == "i100" else "opt", 1.0) for nm in names}
    return names, ["i001", "i002", "i050", "x_not_in_233"], pi, {nm: 1.0 for nm in names}, p300, fallback, solu


def test_rules_on_synthetic_inputs():
    names, dev, pi, z, p300, fb, solu = _synthetic()
    df = S.derive(names, dev, pi, z, p300, fb, solu, does_not_fit=("i003", "i004"))
    r = df.set_index("name")
    assert list(df.columns) == list(S.COLUMNS) and list(df["name"]) == sorted(names)
    assert r["H_cu"].equals(r["p_cuopt"] >= 0.1) and r["H_cu_p05"].equals(r["p_cuopt"] >= 0.5)
    assert r.loc["i007", "w_source"] == "gams-scip10" and r.loc["i007", "W_SCIP"]
    assert r.loc["i100", "w_source"] == "gams-scip10" and not r.loc["i100", "W_SCIP"] and not r.loc["i100", "r12"]
    assert (r["W_only"] == (r["W_SCIP"] & ~r["H_cu"])).all() and (r["E"] == ~(r["W_SCIP"] | r["H_cu"])).all()
    assert set(r.index[r["dev"]]) == {"i001", "i002", "i050"} and (r["unseen"] == ~r["dev"]).all()
    assert (r["decide"] == (r["unseen"] & ~r.index.isin(["i003", "i004"]))).all()
    # the samples: 10 per E quartile among E-decide, 10 + 10 panel among H_cu-decide, all disjoint from the list
    e = r[r["e_sample"]]
    assert len(e) == 40 and (e["E"] & e["decide"]).all() and sorted(e["e_quartile"].value_counts()) == [10] * 4
    pan = r[r["panel"]]
    assert len(pan) == 20 and (pan["H_cu"] & pan["decide"]).all() and (pan["t4"] & pan["t5"]).all()
    assert (pan.loc[pan["panel_group"] == "p05", "p_cuopt"] >= 0.5).all() and \
        pan.loc[pan["panel_group"] == "p01", "p_cuopt"].between(0.1, 0.5, inclusive="left").all()
    # quartiles: stable sort by P over E-decide, array_split in 4 -> each quartile's P range below the next one's
    qs = [r.loc[(r["e_quartile"] == q), "p_cuopt"] for q in range(1, 5)]
    assert all(qs[k].max() <= qs[k + 1].min() for k in range(3))
    # tiers
    t = r["tier"]
    assert (t[r["does_not_fit"]] == "").all() and (t[r["dev"] & ~r["does_not_fit"]] == "T3").all()
    assert (t[r["H_cu"] & r["decide"]] == "T1").all()
    assert set(r.index[t == "T2"]) == set(r.index[r["decide"] & (r["W_only"] | r["e_sample"])])
    assert (t != "").sum() == 233 - 2
    # deterministic: the same inputs give the same table; another seed another sample
    assert S.derive(names, dev, pi, z, p300, fb, solu, does_not_fit=("i003", "i004")).equals(df)
    other = S.derive(names, dev, pi, z, p300, fb, solu, does_not_fit=("i003", "i004"), seed=7)
    assert set(other.loc[other["e_sample"], "name"]) != set(e.index)
    with pytest.raises(ValueError, match="no screening run"):
        S.derive(names, dev, pi, z, p300, {}, solu)
    with pytest.raises(ValueError, match="233"):
        S.derive(names[:-1], dev, pi, z, p300, fb, solu)


def test_write_read_round_trip(tmp_path):
    names, dev, pi, z, p300, fb, solu = _synthetic()
    df = S.derive(names, dev, pi, z, p300, fb, solu)
    sha = S.write_strata(df, tmp_path / "s.csv")
    assert sha == S.file_sha256(tmp_path / "s.csv") and sha == S.write_strata(df, tmp_path / "t.csv")
    back = S.read_strata(tmp_path / "s.csv")
    for c in S.BOOL_COLUMNS:
        assert back[c].tolist() == df[c].tolist()
    assert np.allclose(back["p_cuopt"], df["p_cuopt"], rtol=0, atol=0)          # %.17g: exact
    assert back["tier"].tolist() == df["tier"].tolist()


def test_committed_strata_file():
    """datasets/mipfeas_strata.csv (committed 2026-10-12): its sha256, D-021's counts, the tiers (T1 64, T2 17 + 40,
    T3 94 + 15), the panel 10 + 10, the does-not-fit list off every tier, R12 on every instance but supportcase22."""
    p = paths.DATASETS / "mipfeas_strata.csv"
    assert S.file_sha256(p) == STRATA_SHA256
    df = S.read_strata(p)
    c = S.counts(df)
    for k, v in S.D021_COUNTS.items():
        assert c[k] == v, k
    assert (c["T1"], c["T2"], c["T3"], c["panel p05"], c["panel p01"]) == (64, 57, 109, 10, 10)
    assert S.names_where(df, does_not_fit=True) == sorted(S.DOES_NOT_FIT)
    assert set(df.loc[df["does_not_fit"], "tier"]) == {""}
    assert S.names_where(df, r12=False) == ["supportcase22"]
    assert sorted(df.loc[df["w_source"] == "gams-scip10", "name"]) == ["satellites2-40", "supportcase22"]


def test_rederive_equals_the_committed_file(tmp_path):
    """Skipped unless the screening's run table (ITREA_STRATA_RUNS or datasets/runs.parquet) and GAMS's archive are on
    disk: the tool re-derives the committed file byte for byte."""
    from itrea.analysis.mipfeas import GAMS_DIR
    runs = Path(os.environ.get("ITREA_STRATA_RUNS", paths.DATASETS / "runs.parquet"))
    if not runs.exists() or not (GAMS_DIR / "primalintegral.csv").exists() or \
            not (paths.DATA / "miplib" / "benchmark-v2.test").exists():
        pytest.skip("the screening's run table or GAMS's archive is not on disk")
    spec = importlib.util.spec_from_file_location("derive_strata", paths.ROOT / "tools" / "derive_mipfeas_strata.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--runs", str(runs), "--check"]) == 0
