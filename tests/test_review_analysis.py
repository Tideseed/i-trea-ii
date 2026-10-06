"""Regression tests for the analysis findings of the 2026-10-06 review (F1–F11 on src/itrea/analysis). Toy models and
synthetic run directories only; the helpers come from tests/test_build_runs.py."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from itrea.analysis import build_runs as B
from itrea.analysis import lineage as L
from itrea.analysis import metrics as M
from itrea.analysis import prereg as PR
from itrea.analysis import reproduce as RP
from itrea.analysis.check_solutions import CHECKER_VERSION, file_sha256
from itrea.benchmarks import hardset as H
from itrea.benchmarks import toy
from test_build_runs import FROZEN_SOLVER, X_70, X_85, _fake_runs, _frozen, make_run, toy_problem

QUIET = dict(log=lambda *a: None)


def _miplib_toy(tmp_path, stem: str, solu_line: str):
    """The mixed toy written as data/miplib/inst/<stem>.mps (a MIPLIB-like path) and a .solu file for it."""
    inst = tmp_path / "data" / "miplib" / "inst"
    inst.mkdir(parents=True)
    p = toy.mixed()
    mps = toy.write_mps(p, inst / f"{stem}.mps")
    solu = tmp_path / "solu.txt"
    solu.write_text(solu_line + "\n")
    return p, mps, solu


# ------------------------------------------------------------------------------------------- F1: MIPLIB .solu
def test_screening_run_of_a_miplib_instance_uses_the_solu_reference(tmp_path):
    stem = "chromaticindex512-7"
    p, mps, solu = _miplib_toy(tmp_path, stem, f"=opt= {stem} 4")
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    make_run(raw, p, mps, "SCREEN", 900, 300, {"solver": [(3.0, X_85, "self"), (6.0, X_70, "self")]},
             set_name="screen1", stem=stem)
    df, _ = B.build(raw, out, solu_path=solu, **QUIET)
    r = df.iloc[0]
    assert r["set"] == "screen1" and r["instance"] == stem
    assert r["ref_kind"] == "opt" and r["ref"] == 4.0 and r["ref_screen"] == 4.0
    pt = (3 * 1 + 3 * (4.5 / 8.5) + 294 * (3 / 7)) / 300
    assert r["pt_screen"] == pytest.approx(pt) and r["pt"] == pytest.approx(pt)
    assert r["pt_mf"] == pytest.approx((3 * 2 + 3 * (4.5 / 8.5) + 294 * (3 / 7)) / 300)   # F3 MIPfeas, gap 2 first
    assert H.stage2_needed(df) == [stem]          # with the run's own best as reference P was 0.0118 and it was pruned


def test_reference_dataset_is_per_instance_and_order_free(tmp_path):
    stem = "air05"
    p, mps, solu = _miplib_toy(tmp_path, stem, f"=opt= {stem} 4")
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    make_run(raw, p, mps, "SCREEN", 900, 300, {"solver": [(3.0, X_85, "self")]}, set_name="screen1", stem=stem)
    make_run(raw, p, mps, "A0", 1000, 300, {"solver": [(2.0, X_70, "self")]}, set_name="e2", stem=stem)
    # a non-MIPLIB instance under two job sets is normal (the set name plays no part in its reference)
    p2, mps2 = toy_problem(tmp_path)
    make_run(raw, p2, mps2, "SCREEN", 900, 300, {"solver": [(3.0, X_85, "self")]}, set_name="screen1")
    make_run(raw, p2, mps2, "A0", 1000, 300, {"solver": [(2.0, X_70, "self")]}, set_name="e2")
    df, _ = B.build(raw, out, solu_path=solu, **QUIET)
    a = df[df["instance"] == stem]
    assert set(a["ref_kind"]) == {"opt"} and set(a["ref"]) == {4.0} and set(a["ref_screen"]) == {4.0}
    m = df[df["instance"] == "mixed"]
    assert set(m["ref_kind"]) == {"screening_best_provisional"} and set(m["ref_screen"]) == {8.5}


def test_reference_dataset_name_collision_raises(tmp_path):
    stem = "mixed"
    p, mps, solu = _miplib_toy(tmp_path, stem, f"=opt= {stem} 4")
    other = tmp_path / "elsewhere"
    other.mkdir()
    p2, mps2 = toy_problem(other)                 # same stem, not a MIPLIB path
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    make_run(raw, p, mps, "SCREEN", 900, 300, {"solver": [(3.0, X_85, "self")]}, set_name="screen1")
    make_run(raw, p2, mps2, "A0", 1000, 300, {"solver": [(3.0, X_85, "self")]}, set_name="e2")
    with pytest.raises(ValueError, match="MIPLIB instance in some runs"):
        B.build(raw, out, solu_path=solu, **QUIET)


def test_persistent_sign_crossing_on_the_screening_reference(tmp_path):
    stem = "negopt"
    p, mps, solu = _miplib_toy(tmp_path, stem, f"=opt= {stem} -4")
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    make_run(raw, p, mps, "SCREEN", 900, 300, {"solver": [(3.0, X_85, "self"), (6.0, X_70, "self")]},
             set_name="screen1", stem=stem)
    df, _ = B.build(raw, out, solu_path=solu, **QUIET)
    r = df.iloc[0]
    assert r["sign_crossing_screen_share"] == pytest.approx(297 / 300)
    assert bool(r["sign_crossing_screen"]) and bool(r["sign_crossing_screen_any"]) and bool(r["sign_crossing"])


# --------------------------------------------------------------------------------- F3: sign crossing, MIPfeas
def test_zero_incumbent_is_not_a_sign_crossing():
    st = M.best_steps([(0.01, 0.0), (0.5, -99.0), (2.0, -100.0)], 1)
    P = M.primal_integral(st, -100.0, 1, 300.0)
    assert M.SIGN_CROSSING not in P.flags and P.value == pytest.approx(0.00172, abs=1e-5)
    s = M.summarize(st, -100.0, 1, 300.0)
    assert not s["sign_crossing"] and not s["sign_crossing_any"] and s["sign_crossing_share"] == 0.0


def test_transient_and_persistent_sign_crossings():
    transient = M.best_steps([(0.01, 5.0), (0.5, -99.0), (2.0, -100.0)], 1)
    s = M.summarize(transient, -100.0, 1, 300.0)
    assert s["sign_crossing_any"] and not s["sign_crossing"]
    assert s["sign_crossing_share"] == pytest.approx(0.49 / 300)
    persistent = M.best_steps([(1.0, 5.0), (100.0, -99.0)], 1)
    s = M.summarize(persistent, -100.0, 1, 300.0)
    assert s["sign_crossing"] and s["sign_crossing_share"] == pytest.approx(99 / 300)
    assert math.isnan(M.sign_crossing_share(persistent, None, 1, 300.0))
    # beating the reference with the other sign is gap 0, not a crossing
    assert M.sign_crossing_share(M.best_steps([(1.0, -2.0)], 1), 5.0, 1, 10.0) == 0.0


def test_mipfeas_gap_and_integral():
    assert M.mipfeas_gap(None, 5.0) == 2.0
    assert M.mipfeas_gap(5.0, 5.0) == 0.0 and M.mipfeas_gap(4.0, 5.0, 1) == 0.0      # equal / beats -> 0
    assert M.mipfeas_gap(3.0, 0.0) == pytest.approx(1.0)                              # ref 0: finite, |v|/max(|v|,1)
    assert M.mipfeas_gap(0.5, 0.0) == pytest.approx(0.5)
    assert M.mipfeas_gap(-2.0, 5.0, -1) == pytest.approx(7 / 5)                       # no sign clamp: in [0, 2]
    assert math.isnan(M.mipfeas_gap(3.0, None))
    st = M.best_steps([(2.0, 20.0), (6.0, 10.0)], 1)
    assert M.mipfeas_integral(st, 10.0, 1, 10.0) == pytest.approx((2 * 2 + 4 * 0.5 + 4 * 0) / 10)
    assert M.mipfeas_integral([], 10.0, 1, 10.0) == 2.0
    assert M.mipfeas_integral([], None, 1, 10.0) == 2.0 and math.isnan(M.mipfeas_integral(st, None, 1, 10.0))
    assert M.mipfeas_integral(M.best_steps([(12.0, 10.0)], 1), 10.0, 1, 10.0) == 2.0       # after T: ignored


def test_shifted_geomean_needs_its_shift():
    with pytest.raises(TypeError):
        M.shifted_geomean([0.0, 0.01])                     # no silent s = 1
    assert M.shifted_geomean([0.0, 0.01], shift=M.MIPFEAS_SHIFT) == pytest.approx(0.002317, abs=1e-6)
    assert math.isnan(M.shifted_geomean([], shift=0.001))


# ------------------------------------------------------------------------------ F2, F6: the computed portfolio
def test_portfolio_row_inherits_backend_and_is_dropped_with_its_parent(tmp_path):
    p, mps = toy_problem(tmp_path)
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    make_run(raw, p, mps, "A0", 1000, 300, {"solver": [(2.0, X_85, "self")]},
             run_json_extra=dict(status="completed", env=dict(git=dict(revision="rev0"))))
    make_run(raw, p, mps, "A5", 1000, 300, {"ea": [(1.0, X_70, "self")]}, summary_extra={"ea": {"backend": "numpy"}},
             run_json_extra=dict(status="completed", env=dict(git=dict(revision="rev5"))))
    df, _ = B.build(raw, out, **QUIET)
    r = df.set_index("arm")
    assert r.loc["A5", "backend"] == "numpy" and bool(r.loc["A5", "backend_mismatch"])
    assert r.loc["A0+A5", "backend"] == "numpy" and bool(r.loc["A0+A5", "backend_mismatch"])
    assert r.loc["A0+A5", "git_rev"] == "rev5" and r.loc["A0+A5", "status"] == "completed"
    assert json.loads(r.loc["A0+A5", "parent_git_revs"]) == {"A0": "rev0", "A5": "rev5"}
    pool = pd.DataFrame(dict(instance=["mixed"], side="test", stratum="miplib_opt", p_degenerate=False))
    sel, info = PR.select_runs(df, pool)
    assert set(sel["arm"]) == {"A0"} and info["backend_mismatch"] == 2


def test_portfolio_row_status_split_and_revision_mismatch(tmp_path):
    p, mps = toy_problem(tmp_path)
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    make_run(raw, p, mps, "A0", 1000, 300, {"solver": [(2.0, X_85, "self")]},
             run_json_extra=dict(status="completed", split="test", env=dict(git=dict(revision="rev0"))))
    make_run(raw, p, mps, "A5", 1000, 300, {"ea": [(1.0, X_70, "self")]}, summary_extra={"ea": {"backend": "cupy"}},
             run_json_extra=dict(status="failed", split="test", env=dict(git=dict(revision="rev5"))))
    df, _ = B.build(raw, out, **QUIET)
    syn = df[df["synthetic"]].iloc[0]
    assert syn["status"] == "failed" and syn["split"] == "test" and not bool(syn["backend_mismatch"])
    pool = pd.DataFrame(dict(instance=["mixed"], side="test", stratum="miplib_opt", p_degenerate=False))
    sel, info = PR.select_runs(df, pool)
    assert "A0+A5" in set(sel["arm"]) and info["portfolio_parent_rev_mismatch"] == [syn["run_id"]]
    # a computed row of smoke parents is not a confirmatory row
    df2 = df.copy()
    df2["split"] = "smoke"
    sel2, _ = PR.select_runs(df2, pool)
    assert len(sel2) == 0


def test_incumbent_before_t0_is_flagged_and_refused(tmp_path):
    p, mps = toy_problem(tmp_path)
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    make_run(raw, p, mps, "A0", 1000, 300, {"solver": [(2.0, X_85, "self")]})
    make_run(raw, p, mps, "A5", 1000, 300, {"ea": [(-1.0, X_70, "self"), (float("nan"), X_85, "self")]},
             summary_extra={"ea": {"backend": "cupy"}})
    df, inc = B.build(raw, out, **QUIET)
    r = df.set_index("arm")
    assert r.loc["A5", "n_inc_bad_t"] == 2 and bool(r.loc["A5", "pre_t0"])
    assert not bool(r.loc["A0", "pre_t0"]) and bool(r.loc["A0+A5", "pre_t0"])            # propagated
    _, hashes = _frozen(tmp_path)
    runs, pool = _fake_runs(hashes, n_inst=6)
    runs["pre_t0"] = False
    runs.loc[(runs["instance"] == "i00") & (runs["arm"] == "A3") & (runs["seed"] == 1000), "pre_t0"] = True
    with pytest.raises(PR.PreregError, match="before t0"):
        PR.run(runs, pool, hashes, git_check=False)


# ------------------------------------------------------------------------------- F4: frozen non-EA settings
def test_expected_hashes_of_the_non_ea_arms(tmp_path):
    f, hashes = _frozen(tmp_path)

    def h(d):
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    assert hashes["A0"] == h({"arm": "A0", "emphasis": None})
    assert hashes["A1"] == h({"arm": "A1", "emphasis": None})
    assert hashes["A0star"] == h({"arm": "A0star", "emphasis": "optimality"})
    assert hashes["A2"] == h({"arm": "A2", "emphasis": None, "gpu_mem_gb": 12.0, "cuopt_threads": 3})
    other = PR.expected_hashes(f, solver=dict(FROZEN_SOLVER, emphasis="feasibility"))
    assert other["A0star"] != hashes["A0star"] and other["A0"] == hashes["A0"]
    assert PR.expected_hashes(f, solver=dict(FROZEN_SOLVER, cuopt_threads=4))["A2"] != hashes["A2"]
    with pytest.raises(ValueError, match="RMM limit"):
        PR.expected_hashes(f, solver=dict(FROZEN_SOLVER, gpu_mem_gb=8))
    (tmp_path / "bad.json").write_text(json.dumps({"emphasis": "x", "threads": 3}))
    with pytest.raises(ValueError, match="unknown keys"):
        PR.read_frozen_solver(tmp_path / "bad.json")


def test_non_ea_arm_with_another_setting_is_refused(tmp_path):
    f, hashes = _frozen(tmp_path)
    runs, pool = _fake_runs(hashes, n_inst=6)
    wrong = PR.expected_hashes(f, solver=dict(FROZEN_SOLVER, emphasis="feasibility"))["A0star"]
    bad = runs.copy()
    bad.loc[bad["arm"] == "A0star", "config_hash"] = wrong
    with pytest.raises(PR.PreregError, match="A0star run"):
        PR.run(bad, pool, hashes, git_check=False)
    ea_only = PR.expected_hashes(f)                                        # no frozen solver settings at all
    with pytest.raises(PR.PreregError, match="no frozen solver-side settings"):
        PR.run(runs, pool, ea_only, git_check=False)


def test_identity_exports_emphasis_and_solver_settings(tmp_path):
    p, mps = toy_problem(tmp_path)
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    ss = {"arm": "A0star", "emphasis": "optimality"}
    make_run(raw, p, mps, "A0star", 1000, 300, {"solver": [(2.0, X_85, "self")]},
             run_json_extra=dict(config=dict(run_hash="h0", solver_settings=ss), job=dict(emphasis="optimality")))
    df, _ = B.build(raw, out, **QUIET)
    r = df.iloc[0]
    assert r["emphasis"] == "optimality" and json.loads(r["solver_settings"]) == ss


def test_a3r_override_matches_the_runner(tmp_path):
    pair = pytest.importorskip("itrea.experiments.pair")
    f, hashes = _frozen(tmp_path)
    job = dict(instance="x.mps", set="s", arm="A3r", seed=1000, budget=300, config=str(f))
    cfg, _ = pair.effective_config(job, pair.PairOptions())
    assert cfg.hash() == hashes["A3r"] == hashes["A5"]
    for arm in PR.SOLVER_ARMS:
        j = dict(job, arm=arm, emphasis=FROZEN_SOLVER["emphasis"] if arm == "A0star" else None)
        assert pair.run_hash(j, pair.PairOptions(cuopt_threads=3)) == hashes[arm], arm


# ------------------------------------------------------------------- F5, F10, F11: degeneracy and companions
def _runs_with_times(hashes, n_inst=8):
    runs, pool = _fake_runs(hashes, n_inst=n_inst)
    for c in ("t_first", "t_gap1", "t_gap5"):
        runs["ea_" + c], runs["ea_" + c + "_event"] = runs[c], runs[c + "_event"]
    runs["ref_zero_instance"] = False
    return runs, pool


def test_frozen_reference_zero_makes_an_instance_p_degenerate(tmp_path):
    _, hashes = _frozen(tmp_path)
    runs, pool = _runs_with_times(hashes)
    runs.loc[runs["instance"] == "i03", "ref_zero_instance"] = True
    pool["ref_zero"] = False
    pool.loc[pool["instance"] == "i05", ["p_degenerate"]] = True             # a screening-time sign crossing
    res = PR.run(runs, pool, hashes, git_check=False)
    assert res["p_degenerate_added_from_reference"] == ["i03"]
    assert res["p_degenerate"] == ["i03", "i05"] and "i03" not in res["p_instances"]
    assert res["p_degenerate_ref_zero"] == ["i03"] and res["p_degenerate_sign_crossing"] == ["i05"]
    comp = res["p_degenerate_companion"]
    assert sorted(comp["ref_zero"]) == ["t_first", "t_gap1"] and sorted(comp["sign_crossing"]) == ["t_gap1", "t_gap5"]
    assert comp["ref_zero"]["t_first"]["H1a"]["n_instances"] == 1 and comp["ref_zero"]["t_first"]["H1a"]["time"] \
        == "t_first"
    assert comp["sign_crossing"]["t_gap5"]["H6"]["time"] == "ea_t_gap5"                     # H6: the EA's own


def test_companions_cover_every_comparison_and_the_excluded(tmp_path):
    _, hashes = _frozen(tmp_path)
    runs, pool = _runs_with_times(hashes, n_inst=10)
    pool.loc[pool["instance"].isin(["i00", "i01"]), "stratum"] = "miplib_feas"
    both_one = runs["instance"] == "i09"                          # both arms P = 1 on all seeds -> excluded
    runs.loc[both_one, ["pt", "ea_pt"]] = 1.0
    runs.loc[both_one, ["t_first_event", "ea_t_first_event"]] = False
    runs.loc[both_one, ["t_first", "ea_t_first"]] = 300.0
    res = PR.run(runs, pool, hashes, git_check=False)
    fc = res["feasibility_companion"]
    assert sorted(fc) == sorted(h for h, *_ in PR.COMPARISONS)
    assert fc["H6"]["time"] == "ea_t_first" and fc["H2"]["n_instances"] == 2
    assert all(fc[h]["p_holm"] >= fc[h]["ppw"]["p"] for h in ("H2", "H3", "H4", "H5"))
    assert fc["H1a"]["p_holm"] == fc["H1a"]["ppw"]["p"]
    assert res["comparisons"]["H1a"]["excluded"] == ["i09"]
    ex = res["excluded_companion"]
    assert ex["H1a"]["n_instances"] == 1 and ex["H1a"]["instances"] == ["i09"]


def test_h2_live_check_and_mipfeas_are_descriptive(tmp_path):
    _, hashes = _frozen(tmp_path)
    runs, pool = _runs_with_times(hashes, n_inst=8)
    a3r = runs[(runs["arm"] == "A3") & runs["instance"].isin(["i00", "i01", "i02"])].assign(
        arm="A3r", config_hash=hashes["A3r"], run_id=lambda d: d["run_id"] + "r")
    runs = pd.concat([runs, a3r], ignore_index=True)
    runs["pt_mf"] = runs["pt"] / 2
    runs["ea_pt_mf"] = runs["ea_pt"] / 2
    screen = pd.DataFrame(dict(run_id=["s1", "s2", "s3"], instance=["m1", "m2", "p1"], arm="SCREEN", seed=900,
                               budget=300.0, set="screen1", synthetic=False, block_void=False, attempt=0,
                               instance_path=["data/miplib/inst/m1.mps.gz", "data/miplib/inst/m2.mps.gz",
                                              "data/pglib_uc/mps/p1.mps.gz"], pt_mf=[0.0, 0.01, 0.5]))
    res = PR.run(pd.concat([runs, screen], ignore_index=True), pool, hashes, git_check=False)
    assert "i00|A3r|s1000" not in res["runs"]["missing"] and res["runs"]["runs_by_arm"]["A3r"] == 6
    live = res["descriptive"]["H2_live_check"]
    assert live["instances"] == ["i00", "i01", "i02"] and live["A3r_vs_A0+A5"]["n_instances"] == 3
    assert "decision" not in live["A3r_vs_A0+A5"] and "A3r" not in res["comparisons"]
    mf = res["descriptive"]["mipfeas"]
    per = runs[runs["arm"] == "A3"].groupby("instance")["pt_mf"].mean()
    assert mf["by_arm"]["A3"]["p_instances"]["sgm"] == pytest.approx(M.shifted_geomean(per.tolist(), shift=0.001))
    assert mf["by_arm"]["A3"]["p_instances"]["n"] == 8 and "A5x0" in mf["ea_by_arm"]
    sm = mf["screen_miplib"]
    assert sm["n"] == 2 and sm["sgm"] == pytest.approx(0.002317, abs=1e-6)            # the PGLib run is not MIPLIB
    assert "MIPfeas-style integral" in PR.markdown(res)


# ------------------------------------------------------------------------------------ F7: lineage selection
def test_lineage_pools_only_live_confirmatory_runs(tmp_path):
    from test_stats import _genealogy

    raw = tmp_path / "raw"

    def run_dir(rel, **rj):
        d = raw / rel
        d.mkdir(parents=True)
        _genealogy(d / "ea.genealogy.npz")
        (d / "run.json").write_text(json.dumps(rj))
        return d

    keep = run_dir("e2/x__A5__s1000__300s__c", set="e2", split="test", arm="A5")
    run_dir("e2/void/x__A5__s1001__300s__c__20261006T0100", set="e2", split="test", arm="A5")
    run_dir("e1/x__A5__s0__120s__c", set="e1", split="tuning", arm="A5")
    run_dir("smoke/x__A5__s7__30s__c", set="smoke", split="smoke", arm="A5")
    run_dir("e2/x__A3__s1000__300s__c", set="e2", split="test", arm="A3")
    dropped: dict = {}
    assert L.find_genealogies(raw, ["A5"], ["e2"], ["test"], dropped) == [keep]
    assert dropped == dict(void=1, set=2, split=0, arm=1)
    assert len(L.find_genealogies(raw)) == 4                                       # void is skipped even unfiltered
    out = tmp_path / "lineage.json"
    assert L.main([str(raw), "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["A5"]["all"]["runs"] == 1 and res["A5"]["selection"]["dropped"]["void"] == 1


# --------------------------------------------------------------------------------- F9: checker cache version
def test_cache_is_keyed_on_the_checker_version(tmp_path, monkeypatch):
    p, mps = toy_problem(tmp_path)
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    make_run(raw, p, mps, "A0", 1000, 10, {"solver": [(1.0, X_70, "self")]})
    _, inc1 = B.build(raw, out, **QUIET)
    assert set(inc1["checker_version"]) == {CHECKER_VERSION} and set(inc1["instance_sha256"]) == {file_sha256(mps)}
    monkeypatch.setattr(B, "CHECKER_VERSION", "next")
    _, inc2 = B.build(raw, out, **QUIET)                              # a new checker re-checks: instance present
    assert inc2["checked"].all() and set(inc2["checker_version"]) == {"next"}
    monkeypatch.setattr(B, "CHECKER_VERSION", "after-next")
    data = mps.read_bytes()
    mps.unlink()
    _, inc3 = B.build(raw, out, **QUIET)                              # ... and reports stale when it is gone
    assert not inc3["checked"].any() and inc3["reason"].str.contains("stale").all()
    mps.write_bytes(data)
    assert RP.checker_versions(tmp_path) == {"after-next": 1}       # what reproduce prints (out = tmp/datasets)


def test_verify_instances_rechecks_a_changed_instance(tmp_path):
    p, mps = toy_problem(tmp_path)
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    make_run(raw, p, mps, "A0", 1000, 10, {"solver": [(1.0, X_70, "self")]})
    _, inc1 = B.build(raw, out, **QUIET)
    with open(mps, "a") as fh:
        fh.write("* a comment changes the file, not the model\n")
    msgs: list[str] = []
    _, inc2 = B.build(raw, out, log=msgs.append)                       # default: the cache answers
    assert not any("changed" in m for m in msgs) and list(inc2["instance_sha256"]) == list(inc1["instance_sha256"])
    _, inc3 = B.build(raw, out, log=msgs.append, verify_instances=True)
    assert any("instance file changed" in m for m in msgs)
    assert set(inc3["instance_sha256"]) == {file_sha256(mps)} and inc3["ok"].all()
    root = tmp_path / "repo"
    (root / "datasets").mkdir(parents=True)
    inc3.to_parquet(root / "datasets" / "incumbents_checked.parquet", index=False)
    assert RP.checker_versions(root) == {CHECKER_VERSION: len(inc3)}


def test_stats_compare_with_an_absent_arm():
    from itrea.analysis import stats as S

    df = pd.DataFrame(dict(instance=["a", "a"], seed=[1, 1], arm=["A0+A5", "A3"], pt=[0.5, 0.4]))
    assert S.compare(df, "A3r", "A0+A5")["n_instances"] == 0
    assert np.isnan(S.compare(df, "A3r", "A0+A5")["mean_d"])
