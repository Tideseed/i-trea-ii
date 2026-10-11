"""GAMS's MIPfeas archive and D-020's calibration (`itrea.analysis.mipfeas`): the acceptance test of the port of GAMS's
integral on their own traces, the trace reader on a hand-made trace, the re-derivation of D-020's sample, and the
calibration report on a synthetic archive and run table (no GPU, no real run). The archive tests skip when
data/mipfeas/gams-2026-09-16/ is absent (it is gitignored, like all of data/)."""
from __future__ import annotations

import importlib.util
import json
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


# ------------------------------------------------------------------------------------- the report, on synthetic data
STRATA_T = (("a", "b"), ("c", "d"))
FIRST = {"a": 0.5, "b": 2.0, "c": None, "d": 30.0}            # GAMS's first incumbent (c: none in 600 s)


def _trace(first, z=10.0, late=0.0) -> str:
    lines = ["* mip_trace_ file: synthetic", "* fields are lineNum, seriesID, node, seconds, bestFound, bestBound",
             "1, S, 0, 0, na, na"]
    if first is not None:
        lines += [f"2, I, 0, {first + late}, {2 * z}, -1e+20", f"3, I, 0, {first + late + 100}, {z}, {z}"]
    lines.append(f"{len(lines) + 1 - 2}, E, 0, 600.2, {'na' if first is None else z}, na")
    return "\n".join(lines) + "\n"


def _archive(root, first=FIRST):
    """A GAMS-shaped archive: GB10 26.08 and 26.04 traces identical, B200 traces 5 s later; the published P of GB10
    26.08 is the port's value of its trace (so the file is self-consistent)."""
    d = root / "gams"
    (d / "trc_files").mkdir(parents=True)
    for n, f in first.items():
        for run, late in (("gb10_cuopt-26-08", 0.0), ("gb10_cuopt-26-04", 0.0), ("b200_cuopt-26-08", 5.0)):
            (d / "trc_files" / f"{n}_{run}.trc").write_text(_trace(f, late=late))
    pd.DataFrame({"InstanceInst.": list(first) + ["x_inf"], "ObjectiveObje.": ["10.0"] * len(first) + ["Infeasible"]}) \
        .to_csv(d / "optimal_objective.csv", index=False)
    pg = {n: MF.trc_integral(MF.trc_path(d, n), 10.0).value for n in first}
    pd.DataFrame({"mps_name": list(pg), "cuopt-gb10": list(pg.values())}).to_csv(d / "primalintegral.csv", index=False)
    return d, pg


def _runs(pg, scale=1.0, drop=(), extra=(), first=FIRST):
    rows = []
    for n, g in pg.items():
        if n in drop:
            continue
        ours = 2.0 if g >= 2 else g * scale
        tf = first[n]
        rows.append(dict(run_id=f"{n}__A6m__s0__600s__h", instance=n, arm="A6m", set="mipfeas", seed=0, budget=600.0,
                         status="completed", synthetic=False, block_void=False, attempt=0, pt_gams=ours,
                         pt_gams_launch=min(2.0, ours * 1.1) if ours < 2 else 2.0, startup_offset_s=3.0,
                         t_first=600.0 if tf is None else tf + 0.25, t_first_event=tf is not None, ref=10.0,
                         **INTACT))
    return pd.DataFrame(rows + list(extra))


INTACT = dict(incumbents_missing="", n_inc_unchecked=0, inc_count_mismatch="", pre_t0=False, n_inc_bad_t=0,
              n_inc_failed=0, n_obj_mismatch=0, beats_ref=False)


def test_report_pass_fail_incomplete(tmp_path):
    d, pg = _archive(tmp_path)
    res = MF.report(_runs(pg), d, strata=STRATA_T)
    assert res["verdict"] == MF.PASS and res["call"]["R"] == pytest.approx(1.0) and res["call"]["n"] == 4
    assert not res["flags"] and not res["beats_ref"]
    assert res["feasible"] == dict(ours=3, gams_on_runs=3, gams_sample=3)
    assert res["call"]["lo"] <= 1.0 <= res["call"]["hi"]
    lr = np.log((np.array([min(2.0, g * 1.1) if g < 2 else 2.0 for g in pg.values()]) + 0.001)
                / (np.array(list(pg.values())) + 0.001))
    assert res["launch"]["R"] == pytest.approx(float(np.exp(lr.mean())))
    fi = res["first_incumbent"]
    assert fi["call"]["n"] == 3 and fi["call"]["median"] == pytest.approx(0.25)          # c: neither found one
    assert fi["launch"]["median"] == pytest.approx(3.25) and fi["only_ours"] == fi["only_gams"] == 0
    assert [s["stratum"] for s in res["strata"]] == [0, 1] and res["ref_rel_diff_max"] == 0.0
    g26 = res["gams_rows"]["gb10_cuopt-26-04"]
    assert g26["R"] == pytest.approx(1.0) and g26["feasible"] == 3
    assert res["gams_rows"]["b200_cuopt-26-08"]["R"] > 1.0                                # 5 s later on every trace
    md = MF.markdown(res)
    assert "**Verdict: PASS**" in md and "| 1 | d |" in md
    # R outside the band (every P 1.3 x GAMS's) -> FAIL; the feasible count off by 2 -> FAIL
    worse = MF.report(_runs(pg, scale=1.3), d, strata=STRATA_T)
    assert worse["verdict"] == MF.FAIL and worse["call"]["R"] > MF.R_HI
    lost = _runs(pg)
    lost.loc[lost["instance"].isin(["a", "b"]), "pt_gams"] = 2.0
    assert MF.report(lost, d, strata=STRATA_T)["verdict"] == MF.FAIL
    # a missing instance: listed with what exists for it, never dropped; no verdict
    failed = dict(run_id="d__A6m__s0__600s__h", instance="d", arm="A6m", set="mipfeas", seed=0, budget=600.0,
                  status="failed", synthetic=False, block_void=False, attempt=0, pt_gams=0.3)
    inc = MF.report(_runs(pg, drop=("d",), extra=[failed]), d, strata=STRATA_T)
    assert inc["verdict"] == MF.INCOMPLETE and [m["instance"] for m in inc["missing"]] == ["d"]
    assert "status failed" in inc["missing"][0]["why"] and inc["n_runs"] == 3
    assert "Missing" in MF.markdown(inc) and "INCOMPLETE" in MF.markdown(inc)
    # two valid runs of one instance (a rerun under another configuration, say): refused, never one picked; other arms
    # are ignored
    rerun = dict(_runs(pg).iloc[0].to_dict(), run_id="a__A6m__s0__600s__other", attempt=1, pt_gams=pg["a"] * 1.05)
    other = dict(_runs(pg).iloc[1].to_dict(), run_id="b__A6", arm="A6", pt_gams=0.9)
    res2 = MF.report(_runs(pg, extra=[rerun, other]), d, strata=STRATA_T)
    assert res2["verdict"] == MF.INCOMPLETE and [m["instance"] for m in res2["missing"]] == ["a"]
    assert "2 valid runs" in res2["missing"][0]["why"] and "a__A6m__s0__600s__other" in res2["missing"][0]["why"]


@pytest.mark.parametrize("fault, text", [(dict(incumbents_missing="cuopt"), "incumbent file missing for cuopt"),
                                         (dict(n_inc_unchecked=3), "3 incumbents unchecked"),
                                         (dict(inc_count_mismatch="cuopt"), "stored incumbents != trace"),
                                         (dict(pre_t0=True, n_inc_bad_t=1), "before t0"),
                                         (dict(n_inc_bad_t=2), "before t0")])
def test_report_refuses_a_run_that_is_not_intact(tmp_path, fault, text):
    """A completed run whose incumbents are not all stored, checked and stamped after t0 is no calibration run: a
    missing incumbent file would score P = 2 and move R with no cause shown (review 2026-10-11)."""
    d, pg = _archive(tmp_path)
    runs = _runs(pg)
    for k, v in fault.items():
        runs.loc[runs["instance"] == "b", k] = v
    res = MF.report(runs, d, strata=STRATA_T)
    assert res["verdict"] == MF.INCOMPLETE and [m["instance"] for m in res["missing"]] == ["b"]
    assert text in res["missing"][0]["why"] and "not intact" in res["missing"][0]["why"]
    assert text in MF.markdown(res)


def test_report_flags_kept_runs_and_beats_and_the_ratio(tmp_path):
    """Failed checks and objective mismatches keep the run (only checked incumbents count) and are printed; a checked
    incumbent better than z* is printed; R is (SGM_ours + 0.001)/(SGM_GAMS + 0.001), not the ratio of the SGMs."""
    d, pg = _archive(tmp_path)
    runs = _runs(pg, scale=1.2)
    runs.loc[runs["instance"] == "a", ["n_inc_failed", "n_obj_mismatch"]] = [2, 1]
    runs.loc[runs["instance"] == "d", "beats_ref"] = True
    res = MF.report(runs, d, strata=STRATA_T)
    assert res["verdict"] in (MF.PASS, MF.FAIL) and not res["missing"]
    assert "2 incumbents failed the check" in res["flags"]["a"] and "1 reported objectives" in res["flags"]["a"]
    assert [b["instance"] for b in res["beats_ref"]] == ["d"]
    md = MF.markdown(res)
    assert "Integrity flags" in md and "Beats the reference" in md and "(SGM_ours + 0.001) / (SGM_GAMS + 0.001)" in md
    sg = res["sgm"]
    assert res["call"]["R"] == pytest.approx((sg["ours_call"] + 0.001) / (sg["gams_on_runs"] + 0.001))
    assert res["call"]["R"] != pytest.approx(sg["ours_call"] / sg["gams_on_runs"])
    with pytest.raises(ValueError, match="integrity columns"):
        MF.report(runs.drop(columns=["n_inc_unchecked"]), d, strata=STRATA_T)


def test_report_cli_writes_json_and_markdown(tmp_path, capsys):
    """The CLI on D-020's 30 names (a synthetic archive and run table): JSON without NaN, markdown, exit codes."""
    first = {n: (None if k % 10 == 9 else 0.5 + k) for k, n in enumerate(MF.SAMPLE)}
    d, pg = _archive(tmp_path, first)
    _runs(pg, first=first).to_parquet(tmp_path / "runs.parquet", index=False)
    args = ["--runs", str(tmp_path / "runs.parquet"), "--gams-dir", str(d), "--json", str(tmp_path / "r.json"),
            "--md", str(tmp_path / "md" / "r.md")]
    assert MF.main(args) == 0
    res = json.loads((tmp_path / "r.json").read_text())
    assert res["verdict"] == "PASS" and res["n_runs"] == 30 and res["feasible"]["gams_sample"] == 27
    assert "NaN" not in (tmp_path / "r.json").read_text()
    assert "Verdict: PASS" in capsys.readouterr().out and (tmp_path / "md" / "r.md").exists()
    assert MF.main(["--runs", str(tmp_path / "none.parquet"), "--gams-dir", str(d)]) == 2


EXT = json.dumps(dict(cuopt_gap_zero=True, cuopt_env_unset=True, cuopt_rmm_gb=24.0, cuopt_host_gb=24.0,
                     cuda_cache=True, cuopt_memory_void=True), sort_keys=True)
F_SETTINGS = json.dumps(dict(arm="A6F", cuopt_threads=5, footprint=dict(cores=dict(x925=[5, 6], a725=[1, 2, 3]))),
                        sort_keys=True)


def _a6f(runs: pd.DataFrame, late: float = 0.0) -> pd.DataFrame:
    """The synthetic runs as A6F under the ext harness at F (what M4's jobs produce)."""
    r = runs.copy()
    r["arm"] = "A6F"
    r["run_id"] = r["run_id"].str.replace("__A6m__", "__A6F__", regex=False)
    r["t_first"] = r["t_first"] + late
    r["harness"], r["solver_settings"] = EXT, F_SETTINGS
    r["config_hash"] = r["run_hash"] = "82a371305f88"
    return r


def test_report_for_a6f_reads_by_d021(tmp_path):
    """D-021's calibration (M4: A6F at F): the late-clock check first (ONE median of first incumbent ours − GAMS over
    strata 0–1, ≤ +0.3 s; per stratum reported), then the band: INSIDE (the published row may be cited as a bridge)
    or OUTSIDE (the difference is the footprint); a failed clock check stops the reading (CLOCK-FAIL) whatever R is.
    A6m keeps D-020's PASS / FAIL, with the clock check reported. The configuration hash of the chosen runs is
    printed."""
    d, pg = _archive(tmp_path)
    res = MF.report(_a6f(_runs(pg)), d, arm="A6F", strata=STRATA_T)
    lc = res["late_clock"]
    assert res["verdict"] == MF.INSIDE and lc["ok"] and res["rule"]["decision"] == "D-021"
    assert [(p["stratum"], p["checked"], p["n"]) for p in lc["strata"]] == [(0, True, 2), (1, True, 1)]
    assert lc["strata"][0]["median"] == pytest.approx(0.25) and lc["pooled"]["n"] == 3
    assert lc["pooled"]["median"] == pytest.approx(0.25) and "ok" not in lc["strata"][0]
    assert "bridge" in res["verdict_reason"] and "anchored on our own A6F" in res["verdict_reason"]
    assert res["config_hashes"] == ["82a371305f88"]
    md = MF.markdown(res)
    assert "MIPfeas calibration (D-021): A6F" in md and "Late-clock check" in md and "**Verdict: INSIDE**" in md
    assert "Configuration hash of the chosen runs: 82a371305f88 (all 4)" in md
    out = MF.report(_a6f(_runs(pg, scale=1.3)), d, arm="A6F", strata=STRATA_T)
    assert out["verdict"] == MF.OUTSIDE and "footprint" in out["verdict_reason"]
    late = MF.report(_a6f(_runs(pg), late=0.2), d, arm="A6F", strata=STRATA_T)       # +0.45 s: a late clock
    assert late["verdict"] == MF.CLOCK_FAIL and not late["late_clock"]["ok"]
    assert "median +0.450 s over strata 0–1, n 3" in late["verdict_reason"] and "diagnosed" in late["verdict_reason"]
    a6m = MF.report(_runs(pg), d, strata=STRATA_T)                                    # D-020's arm: unchanged rule
    assert a6m["verdict"] == MF.PASS and a6m["rule"]["decision"] == "D-020" and a6m["late_clock"]["ok"]
    assert MF.report(_a6f(_runs(pg)), d, strata=STRATA_T)["verdict"] == MF.INCOMPLETE   # --arm A6m finds no A6m run


def test_late_clock_check_decides_on_the_pooled_median():
    """D-021: one median over strata 0–1 decides; one slow stratum does not fail it when the pooled median holds;
    stratum 2 and up are only reported; no pair in strata 0–1 fails it."""
    have = pd.DataFrame(dict(stratum=[0, 0, 0, 1, 1, 2, 2], t_first=[1.0, 1.0, 1.0, 1.5, 1.6, 9.0, np.nan],
                             t_first_gams=[1.1, 1.05, 1.0, 1.0, 1.0, 1.0, 2.0]))
    lc = MF.late_clock(have)
    assert lc["strata"][1]["median"] == pytest.approx(0.55)                          # stratum 1 alone: late
    assert lc["ok"] and lc["pooled"]["n"] == 5 and lc["pooled"]["median"] == pytest.approx(0.0)
    assert [p["checked"] for p in lc["strata"]] == [True, True, False]
    assert lc["strata"][2]["median"] == pytest.approx(8.0)
    assert not MF.late_clock(have[have["stratum"] == 2])["ok"]                         # nothing paired in 0–1


def test_a6f_runs_must_carry_the_calibration_settings(tmp_path):
    """D-021's reading takes only runs under the ext harness at F: a run without it, with M3's 40/40 override or with
    another thread count is listed under missing with the reason, never scored; A6m asks for no harness."""
    d, pg = _archive(tmp_path)
    runs = _a6f(_runs(pg))
    m3 = json.loads(EXT)
    m3.update(cuopt_rmm_gb=40.0, cuopt_host_gb=40.0, cuopt_memory_void=False)
    runs.loc[runs["instance"] == "a", "harness"] = ""
    runs.loc[runs["instance"] == "b", "harness"] = json.dumps(m3, sort_keys=True)
    runs.loc[runs["instance"] == "c", "solver_settings"] = json.dumps(dict(arm="A6F", cuopt_threads=3))
    res = MF.report(runs, d, arm="A6F", strata=STRATA_T)
    assert res["verdict"] == MF.INCOMPLETE and [m["instance"] for m in res["missing"]] == ["a", "b", "c"]
    why = {m["instance"]: m["why"] for m in res["missing"]}
    assert "harness none is not the ext preset" in why["a"] and "is not the ext preset" in why["b"]
    assert "cuOpt threads 3 != |F| = 5" in why["c"] and "not D-021's calibration settings" in why["c"]
    assert MF.report(_runs(pg), d, strata=STRATA_T)["verdict"] == MF.PASS            # A6m: no harness asked for
