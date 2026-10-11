"""GAMS's MIPfeas archive and D-020's calibration (`itrea.analysis.mipfeas`): the acceptance test of the port of GAMS's
integral on their own traces, the trace reader on a hand-made trace, and the re-derivation of D-020's sample. The
archive tests skip when data/mipfeas/gams-2026-09-16/ is absent (it is gitignored, like all of data/)."""
from __future__ import annotations

import importlib.util
import math

import numpy as np
import pandas as pd
import pytest

from itrea.analysis import metrics as M
from itrea.analysis import mipfeas as MF

ARCHIVE = MF.GAMS_DIR
needs_archive = pytest.mark.skipif(not (ARCHIVE / "create_primalintegral.py").exists(),
                                   reason=f"GAMS's MIPfeas archive is not at {ARCHIVE}")


def _theirs():
    """GAMS's scoring script, imported as a module (its main block does not run)."""
    spec = importlib.util.spec_from_file_location("gams_create_primalintegral", ARCHIVE / "create_primalintegral.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TRC = """* mip_trace_ file x.trc: ID = cuOpt.333 Instance = mf
* fields are lineNum, seriesID, node, seconds, bestFound, bestBound
1, S, 0, 0, na, na
2, I, 0, 0.5, 30, -1e+20
3, I, 0, 0.5, 20, -1e+20
4, I, 0, 2.25, 20, 8
5, I, 0, 10, 12.5, 9
6, E, 0, 601.5, 11, 9.5
"""


def test_trace_reader_and_port_on_a_hand_made_trace(tmp_path):
    p = tmp_path / "x_gb10_cuopt-26-08.trc"
    p.write_text(TRC)
    t = MF.read_gams_trc(p)
    assert list(t.columns) == MF.TRC_COLUMNS and len(t) == 6 and math.isnan(t.loc[0, "bestFound"])
    assert MF.trc_points(t) == [(0.5, 30.0), (0.5, 20.0), (2.25, 20.0), (10.0, 12.5), (601.5, 11.0)]
    assert MF.trc_first_incumbent(t) == 0.5
    # z* = 10: γ 2 to 0.5 s, 0.5 (20) to 10 s, round(2.5/12.5, 6) = 0.2 to 600 s; the E record's 11 lies past T
    r = MF.trc_integral(p, 10.0)
    assert r.value == pytest.approx((0.5 * 2 + 9.5 * 0.5 + 590 * 0.2) / 600, abs=1e-15)
    if (ARCHIVE / "create_primalintegral.py").exists():
        assert abs(r.value - _theirs().calculate_primal_integral(str(p), 10.0, 600)[0]) <= 1e-12
    (tmp_path / "none.trc").write_text("* header\n* fields\n1, S, 0, 0, na, na\n2, E, 0, 600, na, na\n")
    none = MF.read_gams_trc(tmp_path / "none.trc")
    assert MF.trc_first_incumbent(none) is None and MF.trc_integral(tmp_path / "none.trc", 10.0).value == 2.0


@needs_archive
def test_port_equals_gams_on_every_gb10_cuopt_2608_trace():
    """The acceptance test of the port (D-020): for every GB10 cuOpt 26.08 trace of their archive, GAMS's own
    `calculate_primal_integral` and `metrics.gams_integral` on our reading of the trace agree to 1e-12, and both equal
    the published `primalintegral.csv` value."""
    theirs, z = _theirs(), MF.read_z(ARCHIVE)
    pi = MF.read_pi(ARCHIVE)
    files = sorted((ARCHIVE / "trc_files").glob(f"*_{MF.GB10_2608}.trc"))
    assert len(files) == 233
    worst = 0.0
    for f in files:
        name = f.name[: -len(f"_{MF.GB10_2608}.trc")]
        ref = theirs.calculate_primal_integral(str(f), z[name], 600)[0]
        ours = MF.trc_integral(f, z[name])
        worst = max(worst, abs(ours.value - ref), abs(ours.value - pi.loc[name, MF.GAMS_COL]))
        assert abs(ours.value - ref) <= 1e-12, (name, ours.value, ref)
        assert not ours.beats_ref
    assert worst <= 1e-12
    feasible = int((pi[MF.GAMS_COL] < 2).sum())
    assert M.shifted_geomean(pi[MF.GAMS_COL], shift=M.MIPFEAS_SHIFT) == pytest.approx(0.0393, abs=5e-5) and feasible


@needs_archive
def test_sample_is_d020s_derivation():
    """D-020's sample, re-derived: the 233 sorted by `cuopt-gb10` (stable), numpy array_split into 6 strata,
    default_rng(20261011).choice(stratum, 5, replace=False) per stratum in order, each pick sorted."""
    pi = pd.read_csv(ARCHIVE / "primalintegral.csv")
    assert len(pi) == 233
    names = pi.sort_values(MF.GAMS_COL, kind="stable")["mps_name"].to_numpy()
    rng = np.random.default_rng(MF.SAMPLE_SEED)
    derived = tuple(tuple(sorted(rng.choice(st, 5, replace=False).tolist())) for st in np.array_split(names, 6))
    assert derived == MF.STRATA and len(set(MF.SAMPLE)) == 30
    p = pi.set_index("mps_name").loc[list(MF.SAMPLE), MF.GAMS_COL]
    assert M.shifted_geomean(p, shift=M.MIPFEAS_SHIFT) == pytest.approx(0.0393, abs=5e-5)    # D-020: = the 233's
    assert int((p < 2).sum()) == 27
    assert [MF.stratum_of(n) for n in MF.SAMPLE] == [k for k in range(6) for _ in range(5)]
