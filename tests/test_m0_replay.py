"""D-021's M0 replay (`tools/m0_replay.py`) on synthetic data: the offline stop, the cut-vs-uncut comparison, the
run-table replay (eligibility, a would-be void, hours) and the memory plan of one toy instance in a child process."""
from __future__ import annotations

import importlib.util
import math

import pandas as pd
import pytest

from itrea import paths
from itrea.analysis import metrics as M
from itrea.experiments import pair


@pytest.fixture(scope="module")
def m0():
    spec = importlib.util.spec_from_file_location("m0_replay", paths.ROOT / "tools" / "m0_replay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stop_time_and_compare(m0):
    pts = [(5.0, 100.0), (1.0, 120.0), (9.0, 100.0 * (1 + 4e-7)), (12.0, 100.0), (700.0, 100.0)]
    assert m0.stop_time(pts, 100.0, 1, 600.0) == (5.0, 0)                     # time order, not list order
    assert m0.stop_time([(1.0, 150.0)], 100.0, 1, 600.0) == (None, None)
    assert m0.stop_time([(700.0, 100.0)], 100.0, 1, 600.0) == (None, None)    # after T: never
    steps = [(1.0, 120.0), (5.0, 100.0 * (1 + 3e-7)), (9.0, 100.0)]            # gamma rounds to 0 at 5 s, gap 3e-7
    t, _ = m0.stop_time(steps, 100.0, 1, 600.0)
    assert t == 9.0                                                            # Berthold's 1e-9 holds it back
    c = m0.compare(steps, t, 100.0, 1, 600.0, offset=2.5)
    assert c["d_pt_gams"] == 0.0 and c["d_P"] == 0.0 and c["d_pt_gams_launch"] == 0.0
    # a stop that keeps a residual gap: pt_gams unchanged, P and pt_mf within the gap
    s2 = [(1.0, 120.0), (5.0, 100.0 * (1 + 5e-10)), (9.0, 100.0)]
    c2 = m0.compare(s2, 5.0, 100.0, 1, 600.0)
    assert c2["d_pt_gams"] <= 1e-12 and 0 < c2["d_P"] <= 5e-7 and c2["d_pt_mf"] <= 5e-7


def test_replay_runs(m0, monkeypatch):
    """Eligibility from MIPLIB's =opt= (stubbed), the stop over every role, a would-be void when the triggering
    incumbent failed the check, hours to the stop from the run's end."""
    monkeypatch.setattr(pair, "miplib_stop_ref", lambda job, solu_path=None: (
        (pair.StopRef(10.0, "opt", "t"), "") if "miplib" in job["instance"] else (None, "not MIPLIB")))
    runs = pd.DataFrame([
        dict(run_id="a", set="s", arm="A9", instance="x", instance_path="data/miplib/inst/x.mps.gz", budget=600.0,
             sense=1, ref=10.0, pt_gams=math.nan, startup_offset_s=2.0, run_dir="", synthetic=False),
        dict(run_id="b", set="s", arm="A6F", instance="x", instance_path="data/miplib/inst/x.mps.gz", budget=600.0,
             sense=1, ref=10.0, pt_gams=math.nan, startup_offset_s=2.0, run_dir="", synthetic=False),
        dict(run_id="c", set="s", arm="A6F", instance="y", instance_path="data/pglib_uc/mps/y.mps.gz", budget=600.0,
             sense=1, ref=10.0, pt_gams=math.nan, startup_offset_s=2.0, run_dir="", synthetic=False)])
    inc = pd.DataFrame([
        dict(run_id="a", role="ea", t=3.0, obj_recomputed=12.0, obj_reported=12.0, checked=True, ok=True),
        dict(run_id="a", role="cuopt", t=20.0, obj_recomputed=10.0, obj_reported=10.0, checked=True, ok=True),
        dict(run_id="a", role="solver", t=30.0, obj_recomputed=10.0, obj_reported=10.0, checked=True, ok=True),
        dict(run_id="b", role="cuopt", t=8.0, obj_recomputed=9.0, obj_reported=10.0, checked=True, ok=False),
        dict(run_id="b", role="cuopt", t=50.0, obj_recomputed=10.0, obj_reported=10.0, checked=True, ok=True),
        dict(run_id="c", role="cuopt", t=1.0, obj_recomputed=10.0, obj_reported=10.0, checked=True, ok=True)])
    rr = m0.replay_runs(runs, inc).set_index("run_id")
    assert rr.loc["a", "t_stop"] == 20.0 and rr.loc["a", "d_pt_gams"] == 0.0 and not rr.loc["a", "would_void"]
    want = M.gams_integral([(3.0, 12.0), (20.0, 10.0)], 10.0, 1, 600.0).value
    assert rr.loc["a", "pt_gams"] == pytest.approx(want, abs=1e-15)
    assert rr.loc["b", "would_void"] and rr.loc["b", "t_stop"] == 8.0         # a failed incumbent triggers: void
    assert not rr.loc["c", "eligible"] and math.isnan(rr.loc["c", "t_stop"])
    h = m0.hours(rr.reset_index(), ["arm"]).set_index("arm")
    assert h.loc["A9", "hours_saved"] == pytest.approx((600.0 - 20.0) / 3600)
    assert h.loc["A6F", "hours_saved"] == 0.0 and h.loc["A6F", "stopped"] == 0


def test_plan_one_on_a_toy(m0, tmp_path):
    """The shape-only plan reads the instance as the EA does and applies the EA's refusals (a toy fits)."""
    from itrea.benchmarks import toy
    inst = tmp_path / "miplib" / "inst"
    inst.mkdir(parents=True)
    toy.write_mps(toy.set_cover(30, 20, seed=1), inst / "toyx.mps.gz")
    out = m0.plan_one("toyx", tmp_path)
    assert out["fits"] and out["refused"] == "" and out["n"] == 30 and out["gene_dtype"] == "int8" and out["P"] > 0
