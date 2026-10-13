"""`tools/d022_report.py` (D-022: A9 under the external harness, R13 + R14; C1's interim A9 result) on synthetic run
tables and run directories — no solver, no EA, no GPU: the outcome rule (ADOPT / REJECT by the mean or by an
instance worse on both seeds / INCONCLUSIVE with and without named instances), the band from the control's or A6F's
seed spread, every validity leg (compile on the clock by the runner's own rule, gap settings, P and gene dtype per
block, missing runs, void reruns, the harness), the mechanism counts and cuOpt CPU-s per wall-s, both readings of
failed runs (a void whose rerun failed, a crash before the record), the reading's own keys (other seeds, budgets and
voided blocks of the set), a run without P(120), the reused-run-id qualification (`<run_id>#<set>`), the C1 interim
block with the weak sets in D-013 terms, the INCONCLUSIVE follow-up (seeds 2–3, set itx-r13f) and its job file; and
D-022's amendment of 2026-10-13: the defaults are `jobs/itx-r13.jsonl`'s two sets (the rule set of 3, C1's 11), the
58-run layout gives a complete verdict and a complete C1 block, the mean is over the rule set, a missing / failed /
voided C1 run never changes D-022's verdict while a missing or failed rule-set run does."""
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


ARM_HARNESS = "the arm's"


def make(root: Path, set_: str, inst: str, arm: str, seed: int, pt: float, *, status="completed", gaps=None,
         compiled_cuda=False, compiled_cupy=False, P=256, dtype="float64", cuopt_exit="budget", cuopt_exit_t=120.4,
         solver_exit_t=120.2, final_gap=0.01, cpu=None, env_unset=None, attempt=0, run_id=None,
         harness=ARM_HARNESS, ea_summary=True, cuopt_summary=True) -> dict:
    """One synthetic run of set itx-r13 (or any set): its directory and its runs.parquet row. `gaps`: None = the arm's
    own (0 under ext, absent in the control); "default" / "zero" force one. `harness`: run.json config.harness (the
    arm's resolved harness by default). `ea_summary` / `cuopt_summary` False: the process died without its summary."""
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
                              compiled_on_clock=compiled_cupy), name=inst, seed=seed, budget=120.0, set=set_,
              processes=procs, config=dict(harness=(EXT if ext else R13OFF) if harness == ARM_HARNESS else harness))
    (d / "run.json").write_text(json.dumps(rj))
    if cuopt_summary:
        (d / "cuopt.summary.json").write_text(json.dumps(dict(settings=settings, exit_reason=cuopt_exit)))
    if base == "A9" and ea_summary:
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
    """The report on the synthetic world: INST is both the rule set and C1's set (b) (the defaults are the real
    names of D-022's amendment; `world58` builds their layout)."""
    return D.report(runs, root, instances=INST, c1_instances=INST, **kw)


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
    rd2 = rep(tmp_path, runs2, sets=(D.SET, "itx-r13b"))["readings"][0]          # the rule and C1 share the key
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
    assert sc[D.TREAT]["all"]["worse"] == 1 and sc[D.TREAT]["weak"]["better"] == 4    # i5: 0.4 vs 0.3
    assert D.CTRL not in cu and D.CTRL not in sc                  # A9@r13 runs on the rule set only (amendment)


def test_cli_writes_json(tmp_path, monkeypatch):
    runs = world(tmp_path)
    runs.to_parquet(tmp_path / "runs.parquet")
    monkeypatch.setattr(D, "development_instances", lambda: INST)           # the 17 names live in datasets/
    out = tmp_path / "r.json"
    sets = ["--instances", ",".join(INST), "--c1-instances", ",".join(INST[1:])]
    assert D.main(["--runs", str(tmp_path / "runs.parquet"), "--root", str(tmp_path), "--json", str(out)] + sets) == 0
    js = json.loads(out.read_text())
    assert js["readings"][0]["verdict"].startswith("ADOPT") and js["instances"] == INST
    assert js["c1_instances"] == INST[1:] and js["readings"][0]["c1_completeness"]["n_expected"] == 4 * 2 * 2
    assert D.main(["--runs", str(tmp_path / "nope.parquet")]) == 2


def test_default_instances_are_the_17_development_and_class_probe_instances():
    names = D.development_instances()
    assert len(names) == len(set(names)) == 17 and "chromaticindex1024-7" in names


# ================================================================================================= harness (validity)
@pytest.mark.parametrize("arm, h", [(D.CTRL, None), (D.CTRL, "ext"), (D.TREAT, "r13off"), (D.A6F, "r13off"),
                                    (D.TREAT, "m3"), (D.A6F, None)])
def test_the_harness_leg(tmp_path, arm, h):
    """Only R13 + R14 may differ between the arms: a control without a harness (no R29 caps, no R15 CUDA cache) or
    under ext, a treatment or A6F under the control's harness or the memory probe's caps — each INVALID, the run named."""
    hz = {None: None, "ext": EXT, "r13off": R13OFF, "m3": dict(EXT, cuopt_rmm_gb=40.0, cuopt_host_gb=40.0)}[h]
    res = rep(tmp_path, world(tmp_path, harness={("i2", arm, 1): hz}))
    v = res["readings"][0]["validity"]
    assert not v["ok"] and [(x["instance"], x["arm"], x["seed"]) for x in v["harness"]] == [("i2", arm, 1)]
    assert verdict(res).startswith("INVALID") and "harness" in verdict(res)


def test_every_control_without_a_harness_is_invalid_and_the_table_column_stands_in(tmp_path):
    """The review's probe: every A9@r13 run.json with config.harness None — the gap leg (default gaps) and R14 (env
    kept) cannot tell it from the control; the harness leg does. The run table's `harness` column (JSON) is read when
    run.json has none."""
    runs = world(tmp_path, harness={(i, D.CTRL, s): None for i in INST for s in D.SEEDS})
    res = rep(tmp_path, runs)
    assert verdict(res).startswith("INVALID") and len(res["readings"][0]["validity"]["harness"]) == 10
    runs["harness"] = [json.dumps(R13OFF if a == D.CTRL else (EXT if a in D.ARMS else None), sort_keys=True)
                       if a in D.ARMS else "" for a in runs["arm"]]
    assert verdict(rep(tmp_path, runs)).startswith("ADOPT")
    assert D.expected_harness() == {D.CTRL: R13OFF, D.TREAT: EXT, D.A6F: EXT}


# ================================================================================================= kept crashes
def _void(root: Path, run_id: str, **kw) -> None:
    with open(root / D.SET / "void.jsonl", "a") as f:
        f.write(json.dumps(dict(dict(run_id=run_id, block="b", arm="?", reason="gpu_cap"), **kw)) + "\n")


def _rid(runs, i, arm, s) -> str:
    return runs[(runs["instance"] == i) & (runs["arm"] == arm) & (runs["seed"] == s)]["run_id"].iloc[0]


def test_keep_closes_a_void_whose_rerun_failed(tmp_path):
    """i2 / s0 voided (gpu_cap), its rerun of A9@r13x then failed: `exclude` reads it missing (INCOMPLETE); `keep`
    closes the void with the kept crash and gives the outcome verdict, naming both."""
    runs = world(tmp_path, status={("i2", D.TREAT, 0): "failed"})
    _void(tmp_path, _rid(runs, "i2", D.TREAT, 0))
    by = {r["reading"]: r for r in rep(tmp_path, runs)["readings"]}
    assert by["exclude"]["verdict"].startswith("INCOMPLETE")
    k = by["keep"]
    assert k["verdict"].startswith("ADOPT") and "1 voids closed by a rerun that failed (kept)" in k["verdict"]
    v = k["validity"]
    assert not v["voids_without_rerun"] and v["voids"][0]["rerun_status"] == "failed" and v["voids"][0]["closed"]
    assert not v["voids"][0]["rerun_completed"] and v["voids_closed_by_kept_crash"]


def test_keep_lists_a_crash_before_the_record(tmp_path):
    """A kept A9@r13x crash whose EA was SIGKILLed (no ea.summary.json: no P, no gene dtype — D-021 R29 keeps an EA
    killed at its host cap as an A9 outcome) and a kept A9@r13 crash whose cuOpt left no summary: listed as crashed
    before the record, not INVALID; the same holes in COMPLETED runs stay INVALID."""
    runs = world(tmp_path, status={("i3", D.TREAT, 1): "failed", ("i4", D.CTRL, 0): "failed"},
                 ea_summary={("i3", D.TREAT, 1): False}, cuopt_summary={("i4", D.CTRL, 0): False})
    k = {r["reading"]: r for r in rep(tmp_path, runs)["readings"]}["keep"]
    v = k["validity"]
    assert v["hard_ok"] and not v["block_mismatch"] and not v["gaps"]
    assert sorted((c["instance"], c["arm"], c["seed"]) for c in v["crashed_before_record"]) == \
        [("i3", D.TREAT, 1), ("i4", D.CTRL, 0)]
    assert k["verdict"].startswith("ADOPT") and "2 kept crashes died before their record" in k["verdict"]
    with open(tmp_path / "out.txt", "w") as f:
        D.print_report(rep(tmp_path, runs), out=f)
    assert "crashed before the record (kept)" in (tmp_path / "out.txt").read_text()
    r2 = rep(tmp_path / "c", world(tmp_path / "c", ea_summary={("i3", D.TREAT, 1): False}))
    assert verdict(r2).startswith("INVALID") and "block_mismatch" in verdict(r2)
    r3 = rep(tmp_path / "g", world(tmp_path / "g", cuopt_summary={("i4", D.CTRL, 0): False}))
    assert verdict(r3).startswith("INVALID") and "gaps" in verdict(r3)


# ================================================================================================= the reading's keys
def test_failed_runs_outside_the_reading_are_not_counted(tmp_path):
    """A failed A9@r13x at seed 2 of set itx-r13, one at 300 s and one in a voided block are not the reading's runs:
    one reading, its verdict without a 'kept' note, validity holding."""
    def extra(root):
        return [make(root, D.SET, "i1", D.TREAT, 2, 0.9, status="failed"),
                dict(make(root, D.SET, "i2", D.TREAT, 0, 0.9, status="failed", run_id="i2__x__s0__300s__b300"),
                     budget=300.0),
                dict(make(root, D.SET, "i3", D.CTRL, 1, 0.9, status="failed", run_id="i3__x__s1__120s__bv"),
                     block_void=True)]
    res = rep(tmp_path, world(tmp_path, extra=extra(tmp_path)))
    assert len(res["readings"]) == 1 and verdict(res).startswith("ADOPT") and "kept" not in verdict(res)
    assert res["readings"][0]["validity"]["ok"] and not res["readings"][0]["failed_runs"]
    k = rep(tmp_path / "k", world(tmp_path / "k", extra=extra(tmp_path / "k")), failed="keep")
    assert verdict(k).startswith("ADOPT") and "kept" not in verdict(k)


def test_voids_of_other_seeds_do_not_change_the_seeds_0_1_verdict(tmp_path):
    """A void at seed 2 of set itx-r13 whose rerun is in the table (the review's probe flipped ADOPT to INCOMPLETE),
    and one whose rerun is not (its key read from the voided directory's run.json): neither is the reading's. A void
    whose key cannot be read is counted (it cannot be shown not to be the reading's)."""
    s2 = make(tmp_path, D.SET, "i1", D.TREAT, 2, 0.4)
    runs = world(tmp_path, extra=[s2])
    _void(tmp_path, s2["run_id"])
    gone = make(tmp_path, D.SET, "i2", D.CTRL, 3, 0.4)
    moved = tmp_path / D.SET / "void" / (gone["run_id"] + "__20261012120000")
    moved.parent.mkdir(parents=True, exist_ok=True)
    Path(gone["run_dir"]).rename(moved)
    _void(tmp_path, gone["run_id"], moved_to=str(moved.relative_to(tmp_path / D.SET)))
    rd = rep(tmp_path, runs)["readings"][0]
    assert verdict(rep(tmp_path, runs)).startswith("ADOPT") and not rd["validity"]["voids"]
    assert rd["validity"]["voids_outside_reading"] == 2
    _void(tmp_path, "unknown__run__s0__120s__h000")
    rd2 = rep(tmp_path, runs)["readings"][0]
    assert rd2["verdict"].startswith("INCOMPLETE: voids without a rerun") and len(rd2["validity"]["voids"]) == 1


def test_a_run_without_p_is_named(tmp_path):
    """A completed run whose pt is not finite: 'INCOMPLETE: no P(120) for ...' naming it (the review's probe printed
    '0 missing runs (first: [])'); an A6F run without P is named too — band_i needs its seed pair."""
    runs = world(tmp_path)
    runs.loc[(runs["instance"] == "i1") & (runs["arm"] == D.CTRL) & (runs["seed"] == 0), "pt"] = float("nan")
    v = verdict(rep(tmp_path, runs))
    assert v.startswith("INCOMPLETE: no P(120) for 1 runs [('i1', 'A9@r13', 0)]") and "['i1']" in v
    runs2 = world(tmp_path / "b")
    runs2.loc[(runs2["instance"] == "i4") & (runs2["arm"] == D.A6F) & (runs2["seed"] == 1), "pt"] = float("nan")
    assert verdict(rep(tmp_path / "b", runs2)).startswith("INCOMPLETE: no P(120) for 1 runs [('i4', 'A6F@r13x', 1)]")


# ================================================================================================= C1: weak sets
def test_c1_weak_set_counts_instances_with_p_one_in_both_arms(tmp_path):
    """The review's probe: A0 P = 1 on all five; the treatment P = 1 on i1 and 0.4 elsewhere. D-013 terms (and
    backoff.d013_report's): improved 4 of 5, mean P 1.0 -> 0.52, i1 at d = 0 in the mean and in its interval;
    stats.compare would leave i1 out (named). With every weak instance at P = 1 in both arms the block still reads:
    improved 0 of 5, not an empty block."""
    res = rep(tmp_path, world(tmp_path, p={("i1", D.TREAT): (1.0, 1.0)}, a0={i: 1.0 for i in INST}))
    w = res["readings"][0]["c1"]["A0"][D.TREAT]["weak"]
    assert (w["better"], w["worse"], w["n"], w["n_weak"]) == (4, 0, 5, 5)
    assert w["mean_P_before"] == pytest.approx(1.0) and w["mean_P_after"] == pytest.approx(0.52)
    assert w["P_reduction"] == pytest.approx(0.48) and w["mean_d"] == pytest.approx(4 * 0.6 / 5)
    assert w["ci"][0] <= w["mean_d"] <= w["ci"][1] and w["worst_P_after"] == pytest.approx(1.0)
    assert w["left_out_by_compare"] == ["i1"]
    with open(tmp_path / "out.txt", "w") as f:
        D.print_report(res, out=f)
    txt = (tmp_path / "out.txt").read_text()
    assert "weak set: improved 4 of 5, mean P 1.000 -> 0.520 (-48 %)" in txt and "would leave out): ['i1']" in txt
    p2 = {(i, a): (1.0, 1.0) for i in INST for a in (D.CTRL, D.TREAT)}
    res2 = rep(tmp_path / "b", world(tmp_path / "b", p=p2, a0={i: 1.0 for i in INST}))
    w2 = res2["readings"][0]["c1"]["A0"][D.TREAT]["weak"]
    assert (w2["n"], w2["better"], w2["mean_d"], w2["p"]) == (5, 0, 0.0, 1.0) and len(w2["left_out_by_compare"]) == 5


# ================================================================================================= the follow-up
def followup_world(root: Path, p_f=None, inst=("i1",), drop=(), base_p=None, set_=D.FOLLOWUP_SET, **per_run):
    """The INCONCLUSIVE world (i1 worse on seed 0 only: control 0.4 / 0.4, treatment 0.6 / 0.4; mean d -0.04) unless
    `base_p`, plus seeds 2-3 of `inst` x the three arms in `set_` (control 0.4, treatment 0.4, A6F 0.6 unless `p_f` =
    {(instance, arm): (seed 2, seed 3)}); `drop` = (instance, arm, seed) not run; `per_run` as in `world`, for the
    follow-up runs."""
    runs = world(root, p=base_p if base_p is not None else {("i1", D.TREAT): (0.6, 0.4)})
    extra = []
    for i in inst:
        for s in D.FOLLOWUP_SEEDS:
            for arm, dflt in ((D.CTRL, 0.4), (D.TREAT, 0.4), (D.A6F, 0.6)):
                if (i, arm, s) in drop:
                    continue
                kw = {k: v[(i, arm, s)] for k, v in per_run.items() if (i, arm, s) in v}
                extra.append(make(root, set_, i, arm, s, (p_f or {}).get((i, arm), (dflt, dflt))[s - 2], **kw))
    return pd.concat([runs, pd.DataFrame(extra)], ignore_index=True)


def fu(res, k=0):
    return res["readings"][k]["followup"]


@pytest.mark.parametrize("treat, want", [
    ((0.35, 0.35), "ADOPT"),                      # i1 over seeds 0-3: 0.4 vs 0.425 -> d -0.059, mean -0.012
    ((0.6, 0.6), "REJECT"),                       # worse than -band on both seeds 2 and 3
    ((0.4, 0.4), "NOT ADOPTED"),                  # 0.4 vs 0.45 -> d -0.111, mean -0.022: still between
    ((0.6, 0.4), "NOT ADOPTED"),                  # worse on one of seeds 2-3 only: no reject; mean -0.044
])
def test_followup_reading(tmp_path, treat, want):
    runs = followup_world(tmp_path, p_f={("i1", D.TREAT): treat})
    res = rep(tmp_path, runs, followup=(D.FOLLOWUP_SET,))
    assert verdict(res).startswith("INCONCLUSIVE") and verdict(res).endswith("— i1")       # seeds 0-1 unchanged
    f = fu(res)
    assert f["applies"] and f["rerun"] == ["i1"] and not f["rerun_all"] and f["validity"]["ok"]
    assert f["verdict"].startswith(want), f["verdict"]
    row = next(r for r in f["outcome"]["rows"] if r["instance"] == "i1")
    assert row["band"] == 0.05 and len(row["d_seed"]) == 4
    assert row["p_treat"] == pytest.approx((0.6 + 0.4 + treat[0] + treat[1]) / 4)
    if want == "NOT ADOPTED":
        assert "non-inferiority not shown" in f["verdict"] and "open to Berk" in f["verdict"]
    with open(tmp_path / "out.txt", "w") as fh:
        D.print_report(res, out=fh)
    assert f"FOLLOW-UP VERDICT (exclude): {want}" in (tmp_path / "out.txt").read_text()


def test_followup_reruns_all_when_none_is_named_and_needs_every_run(tmp_path):
    """Every instance at d -0.03 inside its band: INCONCLUSIVE naming none -> all 17 (here 5) rerun; with their
    seeds 2-3 equal the 4-seed mean is -0.015 -> ADOPT. A missing follow-up run makes it INCOMPLETE."""
    base = {(i, D.TREAT): (0.4 / 0.97, 0.4 / 0.97) for i in INST}
    runs = followup_world(tmp_path, inst=INST, base_p=base)
    f = fu(rep(tmp_path, runs, followup=(D.FOLLOWUP_SET,)))
    assert f["rerun_all"] and f["rerun"] == INST and f["verdict"].startswith("ADOPT")
    p_treat = (2 * 0.4 / 0.97 + 2 * 0.4) / 4                                        # seeds 0-3
    assert f["outcome"]["mean_d"] == pytest.approx((0.4 - p_treat) / p_treat) and f["outcome"]["mean_d"] > -0.02
    runs2 = followup_world(tmp_path / "b", inst=INST, base_p=base, drop={("i3", D.A6F, 3)})
    f2 = fu(rep(tmp_path / "b", runs2, followup=(D.FOLLOWUP_SET,)))
    assert f2["verdict"].startswith("INCOMPLETE: 1 missing runs") and f2["validity"]["missing"] == [("i3", D.A6F, 3)]


def test_followup_does_not_apply_and_keeps_its_own_validity(tmp_path):
    """Seeds 0-1 ADOPT: the follow-up is 'not applicable'. Its runs carry the same validity legs (a treatment at
    default gaps: INVALID), and a failed follow-up run gives both readings."""
    res = rep(tmp_path, followup_world(tmp_path, base_p={}), followup=(D.FOLLOWUP_SET,))
    assert verdict(res).startswith("ADOPT") and not fu(res)["applies"] and "not applicable" in fu(res)["verdict"]
    bad = rep(tmp_path / "g", followup_world(tmp_path / "g", gaps={("i1", D.TREAT, 3): "default"}),
              followup=(D.FOLLOWUP_SET,))
    assert fu(bad)["verdict"].startswith("INVALID") and "gaps" in fu(bad)["verdict"]
    fl = rep(tmp_path / "f", followup_world(tmp_path / "f", status={("i1", D.CTRL, 2): "failed"}),
             followup=(D.FOLLOWUP_SET,))
    by = {r["reading"]: r for r in fl["readings"]}
    assert set(by) == {"exclude", "keep"} and by["exclude"]["followup"]["verdict"].startswith("INCOMPLETE")
    assert "1 failed runs kept" in by["keep"]["followup"]["verdict"]
    # without --followup the failed seed-2 run is not the reading's: one reading
    assert len(rep(tmp_path / "f", followup_world(tmp_path / "f", status={("i1", D.CTRL, 2): "failed"}))
               ["readings"]) == 1


def test_followup_runs_in_the_main_set_do_not_change_seeds_0_1(tmp_path, monkeypatch):
    """Seeds 2-3 put into set itx-r13 itself: the seeds 0-1 verdict is unchanged and `--followup itx-r13` reads them;
    the CLI writes the follow-up into the JSON."""
    runs = followup_world(tmp_path, p_f={("i1", D.TREAT): (0.35, 0.35)}, set_=D.SET)
    res = rep(tmp_path, runs, followup=(D.SET,))
    assert verdict(res).startswith("INCONCLUSIVE") and fu(res)["verdict"].startswith("ADOPT")
    runs.to_parquet(tmp_path / "runs.parquet")
    monkeypatch.setattr(D, "development_instances", lambda: INST)
    out = tmp_path / "r.json"
    assert D.main(["--runs", str(tmp_path / "runs.parquet"), "--root", str(tmp_path), "--followup", D.SET,
                   "--json", str(out), "--instances", ",".join(INST), "--c1-instances", ",".join(INST)]) == 0
    js = json.loads(out.read_text())
    assert js["followup_sets"] == [D.SET] and js["readings"][0]["followup"]["verdict"].startswith("ADOPT")


def test_followup_job_file(tmp_path):
    """`itrea make-jobs itx-r13f`: the named instances x seeds 2-3 x itx-r13's three arms and harnesses, set itx-r13f —
    itx_r13's jobs apart from the seed, the set and the block; 'all' = the 17; unknown names refused."""
    from itrea.experiments import make_jobs as MJ
    from itrea.experiments import pair
    assert (MJ.ITX_R13F_SET, MJ.ITX_R13F_SEEDS) == (D.FOLLOWUP_SET, D.FOLLOWUP_SEEDS)
    jobs = MJ.itx_r13_followup(["glass4", "sing44"])
    assert len(jobs) == 2 * 2 * 3 and {j["seed"] for j in jobs} == {2, 3} and {j["set"] for j in jobs} == {"itx-r13f"}
    assert {pair.arm_label(j): json.dumps(j["harness"], sort_keys=True) for j in jobs} == \
        {D.CTRL: json.dumps(MJ.ITX_R13_CONTROL, sort_keys=True), D.TREAT: '"ext"', D.A6F: '"ext"'}
    assert len({j["block"] for j in jobs}) == 4
    base = {(j["name"], pair.arm_label(j)): j for j in MJ.itx_r13(seeds=(2,)) if j["name"] in ("glass4", "sing44")}
    strip = ("set", "block", "stratum")
    for j in (x for x in jobs if x["seed"] == 2):
        b = base[(j["name"], pair.arm_label(j))]
        assert {k: v for k, v in j.items() if k not in strip} == {k: v for k, v in b.items() if k not in strip}
    assert len(MJ.itx_r13_followup("all")) == len(MJ.itx_r13_followup(["all"])) == 17 * 2 * 3
    for bad in (["nope"], []):
        with pytest.raises(MJ.JobError):
            MJ.itx_r13_followup(bad)
    out = tmp_path / "f.jsonl"
    assert MJ.main(["itx-r13f", "--instances", "glass4,sing44", "--out", str(out)]) == 0
    assert [json.loads(x) for x in out.read_text().splitlines()] == jobs
    assert MJ.main(["itx-r13f", "--instances", "nope", "--out", str(tmp_path / "x.jsonl")]) == 2
    assert not (tmp_path / "x.jsonl").exists()


# ================================================================================================= the amendment's sets
def job_layout() -> list[tuple[str, str, int]]:
    """(instance, arm label, seed) of every job in `jobs/itx-r13.jsonl` (D-022 amendment 2026-10-13: 58 runs)."""
    from itrea.experiments import pair
    jobs = [json.loads(x) for x in (ROOT / "jobs" / "itx-r13.jsonl").read_text().splitlines() if x.strip()]
    assert {j["set"] for j in jobs} == {D.SET} and {float(j["budget"]) for j in jobs} == {D.BUDGET}
    return [(j["name"], pair.arm_label(j), int(j["seed"])) for j in jobs]


def world58(root: Path, *, p=None, drop=(), a0=None, extra=(), **per_run) -> pd.DataFrame:
    """The 58-run layout of `jobs/itx-r13.jsonl` in set itx-r13 (P: control 0.4, treatment 0.4, A6F 0.6 unless `p` =
    {(instance, arm): (seed 0, seed 1)}) and A0 of set dev on C1's 11 (0.9 unless `a0` = {instance: P}); `per_run`,
    `drop`, `extra` as in `world`."""
    p, a0 = p or {}, a0 or {}
    dflt = {D.CTRL: 0.4, D.TREAT: 0.4, D.A6F: 0.6}
    rows = []
    for i, arm, s in job_layout():
        if (i, arm, s) in drop:
            continue
        kw = {k: v[(i, arm, s)] for k, v in per_run.items() if (i, arm, s) in v}
        rows.append(make(root, D.SET, i, arm, s, p.get((i, arm), (dflt[arm], dflt[arm]))[s], **kw))
    for i in D.C1_INSTANCES:
        for s in D.SEEDS:
            rows.append(dict(run_id=f"{i}__A0__s{s}__120s__27f3171fac78", set="dev", instance=i, arm="A0", seed=s,
                             budget=120.0, status="completed", attempt=0, run_dir="", pt=a0.get(i, 0.9),
                             final_gap=0.0, synthetic=False, block_void=False))
    return pd.DataFrame(rows + list(extra))


def text(res) -> str:
    import io
    buf = io.StringIO()
    D.print_report(res, out=buf)
    return buf.getvalue()


def test_default_sets_are_the_job_files():
    """The report's defaults are `jobs/itx-r13.jsonl`'s two sets: (a) the 3 x the three arms, (b) the 11 x {A9@r13x,
    A6F@r13x}, seeds 0-1, 58 runs, neos-960392 in both; both within the 17 development + class-probe instances."""
    lay = job_layout()
    assert len(lay) == len(set(lay)) == 58 and {s for _, _, s in lay} == set(D.SEEDS)
    rule = {(i, a) for i in D.RULE_INSTANCES for a in D.ARMS}
    c1 = {(i, a) for i in D.C1_INSTANCES for a in D.C1_ARMS}
    assert {(i, a) for i, a, _ in lay} == rule | c1 and len(rule | c1) * 2 == 58
    assert set(D.RULE_INSTANCES) & set(D.C1_INSTANCES) == {"neos-960392"}
    assert len(D.RULE_INSTANCES) == 3 and len(D.C1_INSTANCES) == len(set(D.C1_INSTANCES)) == 11
    dev = set(D.development_instances())
    assert set(D.RULE_INSTANCES) <= dev and set(D.C1_INSTANCES) <= dev


def test_58_run_layout_gives_a_complete_verdict_and_a_complete_c1_block(tmp_path):
    """The defaults on the 58-run layout: the rule reads the 18 runs of the 3 (ADOPT, validity holding, mean over the
    3; the header says the other 14 are not run), C1 reads its 44 (complete, A0 22 of 22): A9@r13x 0.4 vs A6F 0.6 on
    all 11 and A0 0.9 on all 11 — improved 11 of 11 against both; A6F's weak set on these runs = all 11."""
    runs = world58(tmp_path)
    res = D.report(runs, tmp_path)
    assert res["instances"] == list(D.RULE_INSTANCES) and res["c1_instances"] == list(D.C1_INSTANCES)
    assert res["rule_set_is_the_amendments"] and len(res["not_in_rule_set"]) == 14
    assert set(res["not_in_rule_set"]).isdisjoint(D.RULE_INSTANCES)
    rd = res["readings"]
    assert len(rd) == 1 and rd[0]["n_runs"] == 18 and rd[0]["validity"]["ok"]
    assert [r["instance"] for r in rd[0]["outcome"]["rows"]] == list(D.RULE_INSTANCES)
    assert verdict(res).startswith("ADOPT R13 + R14") and rd[0]["validity"]["voids_outside_reading"] == 0
    c = rd[0]["c1_completeness"]
    assert c["complete"] and (c["n_runs"], c["n_expected"], c["n_a0"], c["n_a0_expected"]) == (44, 44, 22, 22)
    assert not c["missing"] and not c["failed"] and not c["no_p"] and not c["a0_missing"]
    cu, sc = rd[0]["c1"][D.A6F], rd[0]["c1"]["A0"]
    assert cu["weak_set"] == sorted(D.C1_INSTANCES) and cu["n_baseline"] == 11
    for blk, before, d in ((cu, 0.6, 1 / 3), (sc, 0.9, 5 / 9)):
        a = blk[D.TREAT]["all"]
        assert (a["n"], a["better"], a["worse"]) == (11, 11, 0) and a["mean_P_before"] == pytest.approx(before)
        assert a["mean_P_after"] == pytest.approx(0.4) and a["worst_P_after"] == pytest.approx(0.4)
        assert a["mean_d"] == pytest.approx(d) and a["ci"][0] == pytest.approx(d) and a["p"] < 0.01
    txt = text(res)
    assert "The other 14 development + class-probe instances are not run: R13 cannot act there" in txt
    assert "mean d over the 3 of the rule set" in txt and "44 of 44 runs" in txt and "A0 (dev / classprobe) 22 of 22" \
        in txt and "— set (b) complete" in txt and "VERDICT (exclude): ADOPT" in txt
    assert "all 11: improved 11 of 11, mean P 0.600 -> 0.400 (-33 %), worse 0" in txt
    # the CLI's defaults are the same two sets; --instances overrides the rule set and the header says so
    runs.to_parquet(tmp_path / "runs.parquet")
    out = tmp_path / "r.json"
    assert D.main(["--runs", str(tmp_path / "runs.parquet"), "--root", str(tmp_path), "--json", str(out)]) == 0
    js = json.loads(out.read_text())
    assert js["instances"] == list(D.RULE_INSTANCES) and js["c1_instances"] == list(D.C1_INSTANCES)
    assert js["readings"][0]["verdict"] == verdict(res)
    two = D.report(runs, tmp_path, instances=list(D.RULE_INSTANCES[:2]))
    assert len(two["readings"][0]["outcome"]["rows"]) == 2 and "overriding D-022's amendment" in text(two)


def test_the_mean_is_over_the_rule_set(tmp_path):
    """D-022's amendment and its clarification: the mean d is over the three — the 14 not run do not enter it. Each
    of the three at d = -0.06 inside a wide band (control 0.3 / 0.5): REJECT by the mean, -0.06; a mean over the 17
    with the 14 at d = 0 would read -0.011 and adopt."""
    p = {(i, D.CTRL): (0.3, 0.5) for i in D.RULE_INSTANCES}
    p.update({(i, D.TREAT): (0.4 / 0.94, 0.4 / 0.94) for i in D.RULE_INSTANCES})
    res = D.report(world58(tmp_path, p=p), tmp_path)
    o = res["readings"][0]["outcome"]
    assert len(o["rows"]) == 3 and o["mean_d"] == pytest.approx(-0.06) and not o["worse_both"]
    assert verdict(res).startswith("REJECT") and "mean d -0.060 <= -0.05" in verdict(res)
    assert -0.06 * 3 / 17 > D.ADOPT_MEAN                                  # the other reading would have adopted


def test_inconclusive_naming_none_names_the_three_for_the_followup(tmp_path):
    """Every rule instance at d -0.03 inside its band: INCONCLUSIVE naming none -> the follow-up reruns all three of
    the rule set; the verdict names them, since `make-jobs itx-r13f --instances all` would make the 17."""
    p = {(i, D.TREAT): (0.4 / 0.97, 0.4 / 0.97) for i in D.RULE_INSTANCES}
    runs = world58(tmp_path, p=p)
    res = D.report(runs, tmp_path, followup=(D.FOLLOWUP_SET,))
    v = verdict(res)
    assert v.startswith("INCONCLUSIVE") and "none is, so all 3 of the rule set: " + ", ".join(D.RULE_INSTANCES) in v
    assert "its 'all' is the 17" in v
    f = res["readings"][0]["followup"]
    assert f["applies"] and f["rerun_all"] and f["rerun"] == list(D.RULE_INSTANCES)
    assert len(f["validity"]["missing"]) == 3 * 3 * 2 and f["verdict"].startswith("INCOMPLETE")


def test_a_missing_or_failed_c1_run_does_not_change_the_d022_verdict(tmp_path):
    """Set (b)'s own runs (sing44, rail01 — not in the rule set): missing, failed, voided without a rerun, or under a
    wrong harness, none of them changes D-022's verdict or its validity; the C1 completeness line names them. A failed
    one gives both readings, with byte-identical D-022 verdicts; `keep` scores it in the C1 block."""
    full = D.report(world58(tmp_path / "full"), tmp_path / "full")
    v0 = verdict(full)
    assert v0.startswith("ADOPT")
    gone = D.report(world58(tmp_path / "m", drop={("sing44", D.TREAT, 1)}), tmp_path / "m")
    rd = gone["readings"][0]
    assert verdict(gone) == v0 and rd["validity"]["ok"] and rd["validity"] == full["readings"][0]["validity"]
    c = rd["c1_completeness"]
    assert not c["complete"] and c["missing"] == [("sing44", D.TREAT, 1)] and c["n_runs"] == 43
    assert rd["c1"][D.A6F][D.TREAT]["all"]["n"] == 11                     # sing44 still has seed 0 of both arms
    txt = text(gone)
    assert "missing 1 [('sing44', 'A9@r13x', 1)]" in txt and "set (b) INCOMPLETE (the C1 block reads what is" in txt
    assert f"VERDICT (exclude): {v0}" in txt
    fl = D.report(world58(tmp_path / "f", status={("rail01", D.A6F, 0): "failed"}), tmp_path / "f")
    by = {r["reading"]: r for r in fl["readings"]}
    assert set(by) == {"exclude", "keep"} and by["exclude"]["verdict"] == by["keep"]["verdict"] == v0
    assert not by["exclude"]["failed_runs"] and not by["keep"]["failed_runs"]       # not the rule's runs
    ce, ck = by["exclude"]["c1_completeness"], by["keep"]["c1_completeness"]
    assert ce["missing"] == [("rail01", D.A6F, 0)] and not ce["complete"]
    assert ck["complete"] and ck["n_runs"] == 44 and [(f["instance"], f["arm"]) for f in ck["failed"]] == \
        [("rail01", D.A6F)]
    assert "(kept as D-002 §6 crashes)" in text(fl)
    # a void of a set-(b) run whose rerun is not in the table (its key read from the voided directory's run.json):
    # outside the rule's reading, missing in C1's
    runs = world58(tmp_path / "v")
    row = runs[(runs["instance"] == "sing44") & (runs["arm"] == D.A6F) & (runs["seed"] == 1)].iloc[0]
    moved = tmp_path / "v" / D.SET / "void" / (row["run_id"] + "__20261013160000")
    moved.parent.mkdir(parents=True, exist_ok=True)
    Path(row["run_dir"]).rename(moved)
    _void(tmp_path / "v", row["run_id"], moved_to=str(moved.relative_to(tmp_path / "v" / D.SET)))
    vd = D.report(runs.drop(index=row.name), tmp_path / "v")["readings"][0]
    assert vd["verdict"] == v0 and vd["validity"]["voids_outside_reading"] == 1 and not vd["validity"]["voids"]
    assert vd["c1_completeness"]["missing"] == [("sing44", D.A6F, 1)]
    # a set-(b) run under a wrong harness or compiling on the clock is not the rule's run either
    bad = D.report(world58(tmp_path / "h", harness={("rail01", D.TREAT, 0): None},
                           compiled_cuda={("sing44", D.A6F, 0): True}), tmp_path / "h")
    assert verdict(bad) == v0


def test_a_missing_or_failed_rule_set_run_changes_the_d022_verdict(tmp_path):
    """A rule-set run missing (fhnw's control; neos-960392's A9@r13x, which C1 reads too) or failed (krka's A6F):
    INCOMPLETE under `exclude`; `keep` scores the crash and names it. C1 stays complete unless the run is shared."""
    r1 = D.report(world58(tmp_path / "a", drop={("fhnw-binpack4-48", D.CTRL, 0)}), tmp_path / "a")
    assert verdict(r1).startswith("INCOMPLETE: 1 missing runs") and r1["readings"][0]["c1_completeness"]["complete"]
    r2 = D.report(world58(tmp_path / "b", drop={("neos-960392", D.TREAT, 0)}), tmp_path / "b")
    rd2 = r2["readings"][0]
    assert verdict(r2).startswith("INCOMPLETE") and rd2["validity"]["missing"] == [("neos-960392", D.TREAT, 0)]
    assert rd2["c1_completeness"]["missing"] == [("neos-960392", D.TREAT, 0)]
    r3 = D.report(world58(tmp_path / "c", status={("neos-3004026-krka", D.A6F, 1): "failed"}), tmp_path / "c")
    by = {r["reading"]: r for r in r3["readings"]}
    assert by["exclude"]["verdict"].startswith("INCOMPLETE")
    assert by["keep"]["verdict"].startswith("ADOPT") and "1 failed runs kept as D-002 §6" in by["keep"]["verdict"]
    assert by["exclude"]["c1_completeness"]["complete"] and by["keep"]["c1_completeness"]["complete"]
