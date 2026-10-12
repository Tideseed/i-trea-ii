"""`tools/d022_report.py` (D-022: A9 under the external harness, R13 + R14; C1's interim A9 result) on synthetic run
tables and run directories — no solver, no EA, no GPU: the outcome rule (ADOPT / REJECT by the mean or by an
instance worse on both seeds / INCONCLUSIVE with and without named instances), the band from the control's or A6F's
seed spread, every validity leg (compile on the clock by the runner's own rule, gap settings, P and gene dtype per
block, missing runs, void reruns), the mechanism counts and cuOpt CPU-s per wall-s, both readings of failed runs, the
reused-run-id qualification (`<run_id>#<set>`) and the C1 interim block."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("d022_report", ROOT / "tools" / "d022_report.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

INST = ["i1", "i2", "i3", "i4", "i5"]
EXT = dict(cuopt_gap_zero=True, cuopt_env_unset=True, cuopt_rmm_gb=24.0, cuopt_host_gb=24.0, cuda_cache=True,
           cuopt_memory_void=True)
R13OFF = dict(EXT, cuopt_gap_zero=False, cuopt_env_unset=False)


def make(root: Path, set_: str, inst: str, arm: str, seed: int, pt: float, *, status="completed", gaps=None,
         compiled_cuda=False, compiled_cupy=False, P=256, dtype="float64", cuopt_exit="budget", cuopt_exit_t=120.4,
         solver_exit_t=120.2, final_gap=0.01, cpu=None, env_unset=None, attempt=0, run_id=None) -> dict:
    """One synthetic run of set itx-r13 (or any set): its directory and its runs.parquet row. `gaps`: None = the arm's
    own (0 under ext, absent in the control); "default" / "zero" force one."""
    rid = run_id or f"{inst}__{arm}__s{seed}__120s__h{abs(hash((set_, arm))) % 1000:03d}"
    d = root / set_ / rid.split("#")[0]
    d.mkdir(parents=True, exist_ok=True)
    base = arm.split("@")[0]
    ext = arm in (D.TREAT, D.A6F)
    zero = (gaps == "zero") or (gaps is None and ext)
    settings = dict(time_limit=120.0, log_to_console=False, num_cpu_threads=5 if base == "A6F" else 3, random_seed=seed)
    if zero:
        settings.update(mip_absolute_gap=0.0, mip_relative_gap=0.0)
    unset = list(D.THREAD_VARS) if (env_unset is None and ext) else (env_unset or [])
    n_thr = settings["num_cpu_threads"]
    procs = [dict(role="cuopt", exit_t=cuopt_exit_t, cpu_user_s=(cpu if cpu is not None else n_thr * cuopt_exit_t),
                  cpu_sys_s=0.0, env_unset=unset)]
    if base == "A9":
        procs += [dict(role="solver", exit_t=solver_exit_t, cpu_user_s=120.0, cpu_sys_s=0.1, env_unset=[]),
                  dict(role="ea", exit_t=solver_exit_t + 0.1, cpu_user_s=120.0, cpu_sys_s=0.1, env_unset=[])]
    rj = dict(run_id=rid.split("#")[0], arm=arm, base_arm=base, status=status, compiled_on_clock_cuda=compiled_cuda,
              cupy_cache=dict(files_at_t0=10, files_at_end=11 if compiled_cupy else 10,
                              compiled_on_clock=compiled_cupy),
              processes=procs, config=dict(harness=EXT if ext else R13OFF))
    (d / "run.json").write_text(json.dumps(rj))
    (d / "cuopt.summary.json").write_text(json.dumps(dict(settings=settings, exit_reason=cuopt_exit)))
    if base == "A9":
        (d / "ea.summary.json").write_text(json.dumps(dict(P=P, gene_dtype=dtype, compiled_on_clock=False)))
    return dict(run_id=rid, run_id_base=rid.split("#")[0], set=set_, instance=inst, arm=arm, seed=seed, budget=120.0,
                status=status, attempt=attempt, run_dir=str(d), pt=pt, final_gap=final_gap, synthetic=False,
                block_void=False)


def world(root: Path, *, p=None, extra=(), drop=(), a0=None, **per_run) -> pd.DataFrame:
    """INST x seeds 0-1 x {A9@r13, A9@r13x, A6F@r13x} in set itx-r13 (P: control 0.4, treatment 0.4, A6F 0.6 unless
    `p` = {(instance, arm): (seed 0, seed 1)}), A0 of sets dev / classprobe (0.9 unless `a0` = {instance: P});
    `per_run` = {name: {(instance, arm, seed): value}} keyword overrides of `make`; `drop` = (instance, arm, seed) not
    run; `extra` = more rows."""
    p = p or {}
    a0 = a0 or {}
    rows = []
    for i in INST:
        for s in D.SEEDS:
            for arm, dflt in ((D.CTRL, 0.4), (D.TREAT, 0.4), (D.A6F, 0.6)):
                if (i, arm, s) in drop:
                    continue
                kw = {k: v[(i, arm, s)] for k, v in per_run.items() if (i, arm, s) in v}
                rows.append(make(root, D.SET, i, arm, s, p.get((i, arm), (dflt, dflt))[s], **kw))
            st = "dev" if i in INST[:3] else "classprobe"
            rows.append(dict(run_id=f"{i}__A0__s{s}__120s__27f3171fac78", set=st, instance=i, arm="A0", seed=s,
                             budget=120.0, status="completed", attempt=0, run_dir="", pt=a0.get(i, 0.9),
                             final_gap=0.0, synthetic=False, block_void=False))
    return pd.DataFrame(rows + list(extra))


def rep(root, runs, **kw):
    return D.report(runs, root, instances=INST, **kw)


def verdict(res, k=0):
    return res["readings"][k]["verdict"]


# ================================================================================================= outcome
def test_adopt_when_nothing_changes(tmp_path):
    res = rep(tmp_path, world(tmp_path))
    rd = res["readings"]
    assert len(rd) == 1 and rd[0]["reading"] == "exclude"
    assert rd[0]["validity"]["ok"] and rd[0]["outcome"]["mean_d"] == 0.0
    assert verdict(res).startswith("ADOPT R13 + R14")
    with open(tmp_path / "out.txt", "w") as f:
        D.print_report(res, out=f)
    txt = (tmp_path / "out.txt").read_text()
    assert "VERDICT (exclude): ADOPT" in txt and "selected on the same runs" in txt and "earlier days" in txt


def test_reject_by_the_mean(tmp_path):
    """Every instance at d = -0.06 inside a wide band (the control's seeds 0.3 / 0.5: band 0.4): rejected by the
    mean alone (<= -0.05), no instance worse on both seeds."""
    pc = {(i, D.CTRL): (0.3, 0.5) for i in INST}
    pt = {(i, D.TREAT): (0.4 / 0.94, 0.4 / 0.94) for i in INST}
    res = rep(tmp_path, world(tmp_path, p={**pc, **pt}))
    o = res["readings"][0]["outcome"]
    assert o["mean_d"] == pytest.approx(-0.06) and not o["worse_both"]
    assert all(r["band"] == pytest.approx(0.4) for r in o["rows"])
    assert verdict(res).startswith("REJECT") and "mean d -0.060 <= -0.05" in verdict(res)


def test_reject_by_one_instance_worse_on_both_seeds(tmp_path):
    """i1 at d -0.5 on both seeds (band 0.05) while the mean is +0.06: rejected by the instance."""
    p = {("i1", D.TREAT): (0.8, 0.8)}
    p.update({(i, D.TREAT): (0.32, 0.32) for i in INST[1:]})                    # d = +0.2 each
    res = rep(tmp_path, world(tmp_path, p=p))
    o = res["readings"][0]["outcome"]
    assert o["mean_d"] == pytest.approx((-0.5 + 4 * 0.2) / 5) and o["worse_both"] == ["i1"]
    assert verdict(res).startswith("REJECT") and "both seeds: ['i1']" in verdict(res)


def test_inconclusive_names_the_instances_worse_on_one_seed(tmp_path):
    """i1 worse on seed 0 only (d_s0 -0.333 < -0.05, d_s1 0): mean d -0.04 -> INCONCLUSIVE, seeds 2-3 on i1."""
    res = rep(tmp_path, world(tmp_path, p={("i1", D.TREAT): (0.6, 0.4)}))
    o = res["readings"][0]["outcome"]
    assert o["mean_d"] == pytest.approx(-0.2 / 5) and o["worse_one"] == ["i1"] and not o["worse_both"]
    assert verdict(res).startswith("INCONCLUSIVE") and verdict(res).endswith("— i1")
    # the same mean with no instance outside its band on any seed: INCONCLUSIVE, and the report says no one is named
    res2 = rep(tmp_path / "b", world(tmp_path / "b", p={(i, D.TREAT): (0.4 / 0.97, 0.4 / 0.97) for i in INST}))
    assert res2["readings"][0]["outcome"]["mean_d"] == pytest.approx(-0.03)
    assert verdict(res2).startswith("INCONCLUSIVE") and "none is" in verdict(res2)


def test_band_takes_the_larger_spread_of_the_control_or_a6f(tmp_path):
    """i1 at d -0.33 on both seeds: inside A6F's seed spread (0.2 / 0.8: band 0.75) it is not 'worse', and with the
    others at +0.1 the mean is >= -0.02 -> ADOPT; with A6F's seeds equal the band is 0.05 -> REJECT."""
    p = {("i1", D.TREAT): (0.6, 0.6)}
    p.update({(i, D.TREAT): (0.36, 0.36) for i in INST[1:]})
    res = rep(tmp_path, world(tmp_path, p={**p, ("i1", D.A6F): (0.2, 0.8)}))
    row = res["readings"][0]["outcome"]["rows"][0]
    assert row["band"] == pytest.approx(0.75) and not row["worse_both"] and verdict(res).startswith("ADOPT")
    res2 = rep(tmp_path / "b", world(tmp_path / "b", p=p))
    assert res2["readings"][0]["outcome"]["rows"][0]["band"] == 0.05 and verdict(res2).startswith("REJECT")


def test_mean_counts_instances_with_p_one_in_both_arms(tmp_path):
    """The mean over the instances includes those where both arms have P = 1 (d = 0), which stats.compare leaves
    out: four at P = 1 and i1 at d -0.1 give -0.02 (ADOPT), not -0.1 (REJECT)."""
    p = {(i, a): (1.0, 1.0) for i in INST[1:] for a in (D.CTRL, D.TREAT)}
    p[("i1", D.CTRL)] = (0.3, 0.5)                                                   # band 0.4
    p[("i1", D.TREAT)] = (0.4 / 0.9, 0.4 / 0.9)
    res = rep(tmp_path, world(tmp_path, p=p))
    assert res["readings"][0]["outcome"]["mean_d"] == pytest.approx(-0.02) and verdict(res).startswith("ADOPT")


# ================================================================================================= validity
@pytest.mark.parametrize("case, kw, leg", [
    ("CUDA compile, A6F", dict(compiled_cuda={("i2", D.A6F, 1): True}), "compiled"),
    ("CuPy compile, A9", dict(compiled_cupy={("i2", D.TREAT, 0): True}), "compiled"),
    ("treatment at default gaps", dict(gaps={("i3", D.TREAT, 0): "default"}), "gaps"),
    ("A6F at default gaps", dict(gaps={("i3", D.A6F, 1): "default"}), "gaps"),
    ("control at gaps 0", dict(gaps={("i3", D.CTRL, 1): "zero"}), "gaps"),
    ("P differs", dict(P={("i4", D.TREAT, 0): 128}), "block_mismatch"),
    ("gene dtype differs", dict(dtype={("i4", D.CTRL, 1): "int16"}), "block_mismatch"),
])
def test_validity_legs_make_the_verdict_invalid(tmp_path, case, kw, leg):
    res = rep(tmp_path, world(tmp_path, **kw))
    v = res["readings"][0]["validity"]
    assert not v["ok"] and v[leg] and verdict(res).startswith("INVALID") and leg in verdict(res), case


def test_cupy_flag_on_a6f_is_not_a_compile(tmp_path):
    """The runner checks the CuPy cache only for arms with an EA GPU process; A6F has none."""
    res = rep(tmp_path, world(tmp_path, compiled_cupy={("i2", D.A6F, 0): True}))
    assert res["readings"][0]["validity"]["ok"] and verdict(res).startswith("ADOPT")


def test_missing_runs_and_void_reruns(tmp_path):
    runs = world(tmp_path, drop={("i5", D.A6F, 1)})
    res = rep(tmp_path, runs)
    v = res["readings"][0]["validity"]
    assert v["missing"] == [("i5", D.A6F, 1)] and verdict(res).startswith("INCOMPLETE")
    # a void whose block was rerun: recorded, valid; a void whose rerun is not in the table: open
    full = world(tmp_path / "v")
    rid = full[(full["instance"] == "i2") & (full["arm"] == D.TREAT) & (full["seed"] == 0)]["run_id"].iloc[0]
    (tmp_path / "v" / D.SET / "void.jsonl").write_text(json.dumps(dict(run_id=rid, block="i2__s0__itx-r13",
                                                                       arm=D.TREAT, reason="gpu_cap")) + "\n")
    v2 = rep(tmp_path / "v", full)["readings"][0]["validity"]
    assert v2["ok"] and len(v2["voids"]) == 1 and v2["voids"][0]["rerun_completed"]
    gone = full[~((full["instance"] == "i2") & (full["arm"] == D.TREAT) & (full["seed"] == 0))]
    r3 = rep(tmp_path / "v", gone)
    assert r3["readings"][0]["validity"]["voids_without_rerun"] and verdict(r3).startswith("INCOMPLETE")


def test_r14_thread_variables_are_reported(tmp_path):
    res = rep(tmp_path, world(tmp_path, env_unset={("i1", D.TREAT, 0): ["OMP_NUM_THREADS"]}))
    v = res["readings"][0]["validity"]
    assert [(e["instance"], e["arm"], e["seed"]) for e in v["r14_env"]] == [("i1", D.TREAT, 0)]
    assert v["ok"] and verdict(res).startswith("ADOPT")                             # reported, not a validity leg


# ================================================================================================= mechanism
def test_mechanism_counts_runs_ended_by_cuopts_optimal(tmp_path):
    """Control runs ended by cuOpt's 'optimal' with cuOpt first: i1 s0 at a final gap 1e-4 (counted), i2 s0 at gap 0
    (ended, not counted), i3 s0 where SCIP exited first (not cuOpt first). cuOpt CPU-s per wall-s = cpu / exit_t."""
    kw = dict(cuopt_exit={("i1", D.CTRL, 0): "optimal", ("i2", D.CTRL, 0): "optimal", ("i3", D.CTRL, 0): "optimal",
                          ("i4", D.TREAT, 1): "optimal"},
              cuopt_exit_t={("i1", D.CTRL, 0): 30.0, ("i2", D.CTRL, 0): 50.0, ("i3", D.CTRL, 0): 40.0,
                            ("i4", D.TREAT, 1): 60.0},
              solver_exit_t={("i1", D.CTRL, 0): 30.5, ("i2", D.CTRL, 0): 50.2, ("i3", D.CTRL, 0): 35.0,
                             ("i4", D.TREAT, 1): 60.3},
              final_gap={("i1", D.CTRL, 0): 1e-4, ("i2", D.CTRL, 0): 1e-12, ("i4", D.TREAT, 1): 0.0},
              cpu={("i1", D.A6F, 0): 500.0})
    res = rep(tmp_path, world(tmp_path, **kw))
    m = res["readings"][0]["mechanism"]
    assert [(x["instance"], x["seed"]) for x in m[D.CTRL]["ended_by_cuopt_optimal"]] == [("i1", 0), ("i2", 0)]
    assert [(x["instance"], x["final_gap"]) for x in m[D.CTRL]["with_final_gap"]] == [("i1", 1e-4)]
    assert len(m[D.TREAT]["ended_by_cuopt_optimal"]) == 1 and not m[D.TREAT]["with_final_gap"]
    assert m[D.CTRL]["cpu_per_wall"]["median"] == pytest.approx(3.0)
    c = m[D.A6F]["cpu_per_wall"]
    assert c["median"] == pytest.approx(5.0) and c["min"] == pytest.approx(500.0 / 120.4) and c["n"] == 10


# ================================================================================================= failed runs
def test_failed_runs_both_readings(tmp_path):
    runs = world(tmp_path, status={("i3", D.TREAT, 1): "failed"})
    res = rep(tmp_path, runs)
    by = {r["reading"]: r for r in res["readings"]}
    assert set(by) == {"exclude", "keep"}
    assert by["exclude"]["verdict"].startswith("INCOMPLETE")
    assert ("i3", D.TREAT, 1) in by["exclude"]["validity"]["missing"]
    assert by["keep"]["verdict"].startswith("ADOPT") and "1 failed runs kept as D-002 §6" in by["keep"]["verdict"]
    assert not by["keep"]["validity"]["ok"] and by["keep"]["validity"]["hard_ok"]
    assert [r["reading"] for r in rep(tmp_path, runs, failed="keep")["readings"]] == ["keep"]
    with open(tmp_path / "out.txt", "w") as f:
        D.print_report(res, out=f)
    txt = (tmp_path / "out.txt").read_text()
    assert "VERDICT (exclude): INCOMPLETE" in txt and "VERDICT (keep): ADOPT" in txt


# ================================================================================================= run identity
def test_reused_run_ids_are_selected_by_set(tmp_path):
    """A job's id reused in another set is `<id>#<set>` in the table. The A0 baseline comes from dev / classprobe only
    (a `#later` copy with P 0.1 is ignored); a treatment run of itx-r13 whose id an earlier set used keeps its
    `#itx-r13` row (the earlier set's bare copy, P 0.9, is not read); with two selected sets holding one key, the bare
    id is kept and the other listed."""
    runs = world(tmp_path)
    i = runs.index[(runs["instance"] == "i1") & (runs["arm"] == D.TREAT) & (runs["seed"] == 0)][0]
    bare = runs.loc[i, "run_id"]
    runs.loc[i, "run_id"] = bare + "#" + D.SET                              # itx-r13's copy, P 0.4
    earlier = make(tmp_path, "other", "i1", D.TREAT, 0, 0.9, run_id=bare)    # the earlier set's bare copy
    late_a0 = dict(runs[(runs["arm"] == "A0") & (runs["instance"] == "i1") & (runs["seed"] == 0)].iloc[0])
    late_a0.update(run_id=late_a0["run_id"] + "#later", set="later", pt=0.1)
    runs = pd.concat([runs, pd.DataFrame([earlier, late_a0])], ignore_index=True)
    res = rep(tmp_path, runs)
    rd = res["readings"][0]
    assert rd["outcome"]["rows"][0]["p_treat"] == pytest.approx(0.4) and verdict(res).startswith("ADOPT")
    assert rd["c1"]["A0"]["A9@r13x"]["all"]["mean_P_before"] == pytest.approx(0.9) and not rd["duplicates_left_out"]
    # two selected sets with the same key: the bare id wins, the qualified copy is listed
    copy = make(tmp_path, "itx-r13b", "i2", D.TREAT, 1, 0.95,
                run_id=runs[(runs["instance"] == "i2") & (runs["arm"] == D.TREAT) & (runs["seed"] == 1)]["run_id"]
                .iloc[0] + "#itx-r13b")
    runs2 = pd.concat([runs, pd.DataFrame([copy])], ignore_index=True)
    rd2 = D.report(runs2, tmp_path, sets=(D.SET, "itx-r13b"), instances=INST)["readings"][0]
    assert [d["set"] for d in rd2["duplicates_left_out"]] == ["itx-r13b"]
    assert rd2["outcome"]["rows"][1]["p_treat"] == pytest.approx(0.4)


# ================================================================================================= C1 interim
def test_c1_interim_against_cuopt_and_scip(tmp_path):
    """A6F weak set (P >= 0.5): i1, i2 (0.8); the treatment improves both (0.4) and ties the rest; SCIP's weak set
    (A0 >= 0.5): every instance but i5 (0.3)."""
    p = {("i1", D.A6F): (0.8, 0.8), ("i2", D.A6F): (0.8, 0.8)}
    p.update({(i, D.A6F): (0.4, 0.4) for i in INST[2:]})
    res = rep(tmp_path, world(tmp_path, p=p, a0={"i5": 0.3}))
    c1 = res["readings"][0]["c1"]
    cu = c1[D.A6F]
    assert cu["weak_set"] == ["i1", "i2"] and cu["n_baseline"] == 5
    a, w = cu[D.TREAT]["all"], cu[D.TREAT]["weak"]
    assert (a["better"], a["worse"], a["n"]) == (2, 0, 5) and a["mean_d"] == pytest.approx(2 * 0.5 / 5)
    assert w["n"] == 2 and w["better"] == 2 and w["mean_P_before"] == pytest.approx(0.8)
    assert w["mean_P_after"] == pytest.approx(0.4) and w["P_reduction"] == pytest.approx(0.5)
    assert w["mean_d"] == pytest.approx(0.5) and w["worst_P_after"] == pytest.approx(0.4)
    sc = c1["A0"]
    assert sc["weak_set"] == ["i1", "i2", "i3", "i4"]
    assert sc[D.CTRL]["all"]["worse"] == 1 and sc[D.CTRL]["weak"]["better"] == 4      # i5: 0.4 vs 0.3


def test_cli_writes_json(tmp_path, monkeypatch):
    runs = world(tmp_path)
    runs.to_parquet(tmp_path / "runs.parquet")
    monkeypatch.setattr(D, "development_instances", lambda: INST)           # the 17 names live in datasets/
    out = tmp_path / "r.json"
    assert D.main(["--runs", str(tmp_path / "runs.parquet"), "--root", str(tmp_path), "--json", str(out)]) == 0
    js = json.loads(out.read_text())
    assert js["readings"][0]["verdict"].startswith("ADOPT") and js["instances"] == INST
    assert D.main(["--runs", str(tmp_path / "nope.parquet")]) == 2


def test_default_instances_are_the_17_development_and_class_probe_instances():
    names = D.development_instances()
    assert len(names) == len(set(names)) == 17 and "chromaticindex1024-7" in names
