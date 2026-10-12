"""GAMS `.trc` export (D-021 R16): GAMS's own `calculate_primal_integral` (their create_primalintegral.py, in the
archive) on our exported traces reproduces our `pt_gams` to 1e-12 — on synthetic trajectories, on runs built by
build_runs (both clocks) and on all 233 of GAMS's own GB10 cuOpt 26.08 traces re-exported. Skipped without the archive
(their script is the referee)."""
from __future__ import annotations

import importlib.util
import math

import pytest

from itrea.analysis import gams_trc as G
from itrea.analysis import metrics as M
from itrea.analysis import mipfeas as MF

needs_archive = pytest.mark.skipif(not (MF.GAMS_DIR / "create_primalintegral.py").exists(),
                                   reason="GAMS's archive (data/mipfeas/gams-2026-09-16) is not on disk")


@pytest.fixture(scope="module")
def gams():
    spec = importlib.util.spec_from_file_location("gams_create_primalintegral",
                                                  MF.GAMS_DIR / "create_primalintegral.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.calculate_primal_integral


CASES = {
    "none": ([], 302.0, None),
    "one": ([(12.5, 400.0)], 302.0, None),
    "several": ([(0.118999555706978, 2700023800.0), (3.0, 2350019600.0), (77.25, 1200012600.0)], 1200012599.972384,
                None),
    "reaches_z_early_end": ([(1.0, 10.0), (4.0, 5.0)], 5.0, 4.2),
    "reaches_z_held_to_T": ([(1.0, 10.0), (4.0, 5.0)], 5.0, None),
    "z_zero": ([(2.0, 3.0), (9.0, 1e-7), (30.0, 0.0)], 0.0, 30.5),
    "z_negative_crossing": ([(0.5, 12.0), (2.0, -3.0), (100.0, -9.999)], -10.0, None),
    "after_T_dropped": ([(1.0, 8.0), (650.0, 5.0)], 5.0, None),
    "near_zero_times": ([(1e-9, 100.0), (2e-9, 99.0), (599.999999, 98.0)], 50.0, None),
}


def test_trc_lines_shape_and_refusals(tmp_path):
    lines = G.trc_lines([(1.0, 8.0), (650.0, 5.0)], 600.0, 5.0)
    assert lines[0].startswith("* mip_trace_ file") and lines[1] == f"* fields are {G.FIELDS}"
    assert lines[2:] == ["1, S, 0, 0, na, na", "2, I, 0, 1.0, 8.0, na", "3, E, 0, 600.0, 8.0, na"]
    assert G.trc_lines([], 600.0, 5.0)[-1] == "2, E, 0, 600.0, na, na"
    assert G.trc_lines([(1.0, 5.0)], 600.0, 5.0, end_t=3.0)[-1] == "3, E, 0, 3.0, 5.0, na"   # gamma 0: its end
    assert G.trc_lines([(1.0, 6.0)], 600.0, 5.0, end_t=3.0)[-1] == "3, E, 0, 600.0, 6.0, na"  # gamma > 0: held to T
    assert "beats_ref" in G.trc_lines([(1.0, 4.0)], 600.0, 5.0)[0]
    with pytest.raises(ValueError, match="minimisation"):
        G.trc_lines([(1.0, 4.0)], 600.0, 5.0, sense=-1)
    assert G.trc_name("ic97_potential", "A9@v1c2610", 2000, "call") == "ic97_potential_gb10-f-call_A9@v1c2610-s2000.trc"
    assert G.trc_name("x", "A0", 1, "launch").rsplit("_", 2) == ["x", "gb10-f-launch", "A0-s1.trc"]


def test_floats_read_back_round_trip_exactly_and_by_gams_default_parser_to_1e12(tmp_path):
    """Review finding (analysis): the export writes repr floats. Read with float_precision="round_trip" every double
    comes back bit for bit; GAMS's own call (`mipfeas.read_gams_trc`, pandas' default parser) is not round-trip exact
    on small magnitudes — it agrees to 1e-12 relative, the tolerance the docstring now states."""
    import numpy as np
    import pandas as pd
    rs = np.random.default_rng(7)
    t = np.sort(rs.uniform(0.0, 599.0, 2000))
    obj = np.exp(rs.uniform(np.log(1e-5), np.log(1e6), 2000))
    steps = list(zip(t.tolist(), obj.tolist()))
    p = G.write_trc(tmp_path / "x_gb10-f-call_A9-s0.trc", steps, 600.0, None)
    exact = pd.read_csv(p, comment="*", header=None, na_values=["na", "inf"], skipinitialspace=True,
                        names=MF.TRC_COLUMNS, float_precision="round_trip")
    rows = exact[exact["seriesID"] == "I"]
    assert rows["bestFound"].tolist() == obj.tolist() and rows["seconds"].tolist() == t.tolist()
    theirs = MF.read_gams_trc(p)
    got = theirs[theirs["seriesID"] == "I"]["bestFound"].to_numpy(float)
    rel = np.abs(got - obj) / np.abs(obj)
    assert rel.max() <= 1e-12
    assert np.abs(theirs[theirs["seriesID"] == "I"]["seconds"].to_numpy(float) - t).max() <= 1e-12


@needs_archive
@pytest.mark.parametrize("case", sorted(CASES))
def test_gams_script_reproduces_pt_gams_synthetic(gams, tmp_path, case):
    steps, z, end = CASES[case]
    p = G.write_trc(tmp_path / f"{case}_gb10-f-call_A9-s0.trc", steps, 600.0, z, end)
    theirs, _ = gams(str(p), z, 600)
    ours = M.gams_integral(steps, z, 1, 600.0).value
    assert abs(theirs - ours) <= 1e-12, (case, theirs, ours)


@needs_archive
def test_gams_script_reproduces_their_own_233_reexported(gams, tmp_path):
    """Every GB10 cuOpt 26.08 trace of GAMS's archive, read, reduced to its best-so-far steps, re-exported with its own
    E time, and scored by their script: equal to their published primalintegral.csv value to 1e-12."""
    pi, z = MF.read_pi(), MF.read_z()
    worst = 0.0
    for n in pi.index:
        trc = MF.read_gams_trc(MF.trc_path(MF.GAMS_DIR, n))
        steps = M.best_steps(MF.trc_points(trc), 1)
        e = trc[trc["seriesID"].astype(str).str.strip() == "E"]
        end = float(e["seconds"].iloc[-1]) if len(e) else None
        p = G.write_trc(tmp_path / f"{n}_gb10-f-call_A6F-s0.trc", steps, 600.0, z[n], end)
        theirs, _ = gams(str(p), z[n], 600)
        worst = max(worst, abs(theirs - float(pi.loc[n, MF.GAMS_COL])))
        assert abs(theirs - M.gams_integral(steps, z[n], 1, 600.0).value) <= 1e-12, n
    assert worst <= 1e-12


@needs_archive
def test_exported_runs_score_to_pt_gams_on_both_clocks(gams, tmp_path):
    """Runs built by build_runs (toy, min) exported per clock: GAMS's script gives `pt_gams` (call) and
    `pt_gams_launch` (launch) to 1e-12."""
    from test_build_runs import X_70, X_85, make_run, toy_problem
    from itrea.analysis import build_runs as B
    p, mps = toy_problem(tmp_path)
    raw, out = tmp_path / "raw", tmp_path / "ds"
    out.mkdir()
    procs = [dict(role="solver", launch_epoch=1000.0)]
    make_run(raw, p, mps, "A0", 0, 600, {"solver": [(2.0, X_85, "self"), (6.0, X_70, "self")]},
             run_json_extra=dict(t0_epoch=1001.25, processes=procs))
    make_run(raw, p, mps, "A6F", 0, 600, {"cuopt": [(590.0, X_85, "self")]},
             run_json_extra=dict(t0_epoch=1003.5, processes=procs))
    df, inc = B.build(raw, out, log=lambda *a: None)
    assert set(df["ref"]) == {7.0}                      # best of study: reached, never beaten (their script raises then)
    done = G.export_runs(df, inc, tmp_path / "trc")
    assert sum(len(d["files"]) for d in done) == 4
    for d, (_, row) in zip(done, df.iterrows()):
        assert d["run_id"] == row["run_id"]
        for f in d["files"]:
            theirs, _ = gams(f, 7.0, 600)
            want = row["pt_gams"] if "-call_" in f else row["pt_gams_launch"]
            assert math.isfinite(want) and abs(theirs - want) <= 1e-12, (f, theirs, want)
    assert {f.split("_gb10-f-")[1].split("_")[0] for d in done for f in d["files"]} == {"call", "launch"}
    assert df["pt_gams"].nunique() == 2 and (df["pt_gams"] != df["pt_gams_launch"]).all()
