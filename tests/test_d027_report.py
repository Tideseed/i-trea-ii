"""`tools/d027_report.py` (D-027: the same-day diagnostic of cuOpt's thread count and SCIP's presence, R3 of the C1
diagnosis note) on synthetic run tables and run directories — no solver, no GPU. Validity as one gate before any rule
(the 16 runs complete with P(120), the ext harness, the layout read from run.json — threads, cores, X925 6 idle, no EA
— and no compile on the clock; INVALID names the leg); failed runs under both readings (`exclude` = INVALID by the
letter, `keep` = read with a stated deviation). Rule 1's band (max(0.05, the larger raw seed spread), strict,
one-sided) and its consequence on toguru / doon; rule 2's precondition (A6@r3's cuOpt <= 133 on both seeds; >= 146 on
both = the footprint; otherwise not met) and its three outcomes on A2@r3 (inclusive thresholds, no incumbent = +inf,
minimisation checked); rule 3 on cuOpt's own P in A2 against A6's P; what goes to Berk; the A6F@r13x replicate beside;
the selection (reused ids, other sets, seeds, budgets, voided blocks); the JSON output; and the report's constants
against the job writer and the runner's own plan."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from itrea.experiments import make_jobs as mj
from itrea.experiments import pair

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("d027_report", ROOT / "tools" / "d027_report.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

COMP21, TOGURU, DOON, SING = "comp21-2idx", "neos-4763324-toguru", "neos-2746589-doon", "sing44"
EXT = pair.resolve_harness("ext")
LAYOUT = {R.A6: {"cuopt": ([1, 2, 3], 3)}, R.A6F: {"cuopt": ([1, 2, 3, 5, 6], 5)},
          R.A2: {"solver": ([5], None), "cuopt": ([1, 2, 3], 3)}, "A6F@r13x": {"cuopt": ([1, 2, 3, 5, 6], 5)}}


def make(root: Path, inst: str, arm: str, seed: int, pt: float, *, cuopt_pt=None, obj=None, status="completed",
         layout=None, harness=EXT, compiled=False, sense=1, set_=R.SET, run_id=None, attempt=0, budget=120.0,
         run_json=True, block_void=False) -> dict:
    """One synthetic run: its directory (run.json with the processes' roles, cores and cuOpt's --cpu-threads) and its
    runs.parquet row. `layout`: role -> (cores, threads), the arm's own by default."""
    rid = run_id or f"{inst}__{arm}__s{seed}__{budget:g}s__{abs(hash((set_, arm))) % 1000:03d}"
    d = root / "traces" / "raw" / set_ / rid.split("#")[0]
    d.mkdir(parents=True, exist_ok=True)
    procs = []
    for role, (cores, thr) in (layout or LAYOUT[arm]).items():
        cmd = ["taskset", "-c", ",".join(map(str, cores)), "python", "-m", f"itrea.solvers.{role}_run"]
        if thr is not None:
            cmd += ["--cpu-threads", str(thr), "--gpu-mem-gb", "24.0", "--mip-gap-zero"]
        procs.append(dict(role=role, cores=cores, cmd=cmd, exit_code=0))
    if run_json:
        (d / "run.json").write_text(json.dumps(dict(run_id=rid.split("#")[0], arm=arm, status=status, processes=procs,
                                                    compiled_on_clock_cuda=compiled, config=dict(harness=harness))))
    return dict(run_id=rid, set=set_, instance=inst, arm=arm, seed=seed, budget=budget, status=status,
                attempt=attempt, run_dir=str(d.relative_to(root)), pt=pt,
                cuopt_pt=pt if cuopt_pt is None else cuopt_pt, cuopt_best_obj=math.nan if obj is None else obj,
                sense=sense, harness=json.dumps(harness, sort_keys=True) if harness is not None else "",
                cuda_compiled_on_clock=compiled, synthetic=False, block_void=block_void)


# defaults: no thread effect, comp21 alone-level for A6 / A6F, the stall for A2 (rule 2 -> SCIP), no SCIP effect
P_DEF = {R.A6: (0.5, 0.5), R.A6F: (0.5, 0.5), R.A2: (0.4, 0.4)}
OBJ_DEF = {R.A6: (120.0, 125.0), R.A6F: (117.0, 133.0), R.A2: (150.0, 160.0)}


def world(root: Path, *, p=None, cpt=None, obj=None, drop=(), extra=(), per=None) -> pd.DataFrame:
    """The 16 runs (R.expected_keys) with P(120) from `p[(inst, arm)]` (else P_DEF), cuOpt's P from `cpt` (else = P
    for A6 / A6F and 0.5 for A2), comp21's cuOpt objective from `obj[arm]` (else OBJ_DEF; None = no incumbent);
    `drop` = keys left out, `per[key]` = make() keyword overrides (`pt` included), `extra` = more rows."""
    p, cpt, obj, per = p or {}, cpt or {}, obj or {}, per or {}
    rows = []
    for k in R.expected_keys():
        if k in drop:
            continue
        inst, arm, s = k
        pv = p.get((inst, arm), P_DEF[arm])[s]
        cv = cpt.get((inst, arm), (0.5, 0.5) if arm == R.A2 else None)
        o = obj.get(arm, OBJ_DEF[arm])[s] if inst == COMP21 else 2000.0
        kw = dict(cuopt_pt=None if cv is None else cv[s], obj=o)
        kw.update(per.get(k, {}))
        rows.append(make(root, inst, arm, s, kw.pop("pt", pv), **kw))
    return pd.DataFrame(rows + list(extra))


def one(root, runs, **kw) -> dict:
    rep = R.report(runs, root, **kw)
    assert len(rep["readings"]) == 1
    return rep["readings"][0]


def test_constants_follow_the_job_writer_and_the_runner(tmp_path):
    """The report reads exactly the writer's 16 keys, and its layout table is what the runner plans for those jobs
    (no process on X925 6 in A6 / A2)."""
    jobs = mj.r3_diag()
    assert sorted(R.expected_keys()) == sorted((j["name"], pair.arm_label(j), j["seed"]) for j in jobs)
    assert (R.SET, R.SEEDS, R.BUDGET) == ("r3-diag", (0, 1), 120.0)
    o = pair.PairOptions(traces_dir=tmp_path)
    for j in jobs:
        procs = pair.plan(j, o, tmp_path / pair.run_id_for(j, o), "S", pair.effective_config(j, o)[0])
        got = {p.role: (p.cores, int(p.argv[p.argv.index("--cpu-threads") + 1]) if p.spec.kind == "cuopt" else None)
               for p in procs}
        assert got == R.LAYOUT[pair.arm_label(j)], j
        if pair.arm_label(j) in R.IDLE_ARMS:
            assert R.IDLE_CORE not in {c for p in procs for c in p.cores}


def test_complete_defaults_valid_closed_scip(tmp_path):
    rd = one(tmp_path, world(tmp_path))
    assert (rd["n_present"], rd["n_expected"]) == (16, 16)
    assert rd["validity"]["verdict"] == "VALID" and rd["validity"]["failing"] == []
    assert not any(x["present"] for x in rd["rule1"]["per"].values()) and rd["rule1"]["consequence"] == "CLOSED"
    assert rd["rule2"]["status"] == "SCIP" and rd["rule2"]["a2"] == [150.0, 160.0]
    assert not any(x["present"] for x in rd["rule3"]["per"].values())
    assert "ruled out on" in rd["rule3"]["per"][TOGURU]["text"] and rd["to_berk"] == []
    txt = R.render(R.report(world(tmp_path / "b"), tmp_path / "b"))
    assert "VALIDITY VALID: 16 of 16 runs present" in txt and "-> CLOSED" in txt and "-> SCIP: " in txt
    assert "To Berk: nothing" in txt


def test_rule1_band_one_sided_and_question(tmp_path):
    p = {(TOGURU, R.A6): (0.70, 0.70), (TOGURU, R.A6F): (0.60, 0.62),       # diff 0.09 > band 0.05: present
         (DOON, R.A6): (0.60, 0.80), (DOON, R.A6F): (0.50, 0.52)}           # diff 0.19 < band 0.20 (A6's spread)
    rd = one(tmp_path, world(tmp_path, p=p))
    per = rd["rule1"]["per"]
    assert set(per) == {TOGURU, DOON}                                      # A6F runs only there (the amendment)
    assert per[TOGURU]["present"] and per[TOGURU]["diff"] == pytest.approx(0.09) and per[TOGURU]["band"] == 0.05
    assert not per[DOON]["present"] and per[DOON]["band"] == pytest.approx(0.20)
    assert rd["rule1"]["consequence"] == "QUESTION" and TOGURU in rd["rule1"]["text"]
    assert len(rd["to_berk"]) == 1 and rd["to_berk"][0].startswith("the layout question")
    txt = R.render(R.report(world(tmp_path / "b", p=p), tmp_path / "b"))
    assert "-> PRESENT" in txt and "-> QUESTION" in txt


@pytest.mark.parametrize("a6, a6f, present, word", [
    ((0.75, 0.25), (0.0, 0.0), False, "-> absent"),                         # diff 0.5 == band 0.5: strict
    ((0.30, 0.30), (0.60, 0.60), False, "absent (A6@r3 lower)")])            # reverse: never present
def test_rule1_is_strict_and_one_sided(tmp_path, a6, a6f, present, word):
    rd = one(tmp_path, world(tmp_path, p={(DOON, R.A6): a6, (DOON, R.A6F): a6f}))
    assert rd["rule1"]["per"][DOON]["present"] is present and rd["rule1"]["consequence"] == "CLOSED"
    assert word in R.render(R.report(world(tmp_path / "b", p={(DOON, R.A6): a6, (DOON, R.A6F): a6f}), tmp_path / "b"))


@pytest.mark.parametrize("a6, a2, status, words", [
    ((120.0, 125.0), (150.0, 160.0), "SCIP", "SCIP beside cuOpt is the cause"),
    ((133.0, 133.0), (146.0, 146.0), "SCIP", "SCIP beside cuOpt"),                     # both thresholds inclusive
    ((120.0, 125.0), (None, 150.0), "SCIP", "SCIP beside cuOpt"),                      # no incumbent = +inf
    ((120.0, 125.0), (120.0, 130.0), "EA PRESENCE", "MPS question"),
    ((120.0, 125.0), (133.0, 133.0), "EA PRESENCE", "the EA's presence remains"),
    ((120.0, 125.0), (140.0, 140.0), "INCONCLUSIVE", "inconclusive at n = 2"),          # between
    ((120.0, 125.0), (130.0, 150.0), "INCONCLUSIVE", "inconclusive at n = 2"),          # mixed
    ((150.0, 160.0), (120.0, 120.0), "FOOTPRINT", "3-thread footprint under ext"),      # the precondition's stall
    ((None, 150.0), (120.0, 120.0), "FOOTPRINT", "3-thread footprint under ext"),
    ((130.0, 140.0), (150.0, 150.0), "PRECONDITION NOT MET", "stays inconclusive at n = 2"),
    ((120.0, 150.0), (150.0, 150.0), "PRECONDITION NOT MET", "rule 2 is not read"),
])
def test_rule2_precondition_and_outcomes(tmp_path, a6, a2, status, words):
    rd = one(tmp_path, world(tmp_path, obj={R.A6: a6, R.A2: a2}))
    assert rd["rule2"]["status"] == status and words in rd["rule2"]["text"]
    assert rd["rule2"]["a6"] == [math.inf if x is None else x for x in a6]
    read_a2 = status in ("SCIP", "EA PRESENCE", "INCONCLUSIVE")
    assert rd["rule2"]["a2"] == ([math.inf if x is None else x for x in a2] if read_a2 else None)
    assert "a6f" not in rd["rule2"]                                   # A6F is not run on comp21 (the amendment)
    assert (rd["to_berk"] != []) == (status == "EA PRESENCE")


def test_rule2_sense_and_the_footprint_beside_the_layout_question(tmp_path):
    rd = one(tmp_path, world(tmp_path, per={(COMP21, R.A6, 0): dict(sense=-1)}))
    assert rd["rule2"]["status"] == "NOT READ" and "not minimisation" in rd["rule2"]["text"]
    both = one(tmp_path / "b", world(tmp_path / "b", obj={R.A6: (150.0, 150.0), R.A2: (120.0, 120.0)},
                                     p={(DOON, R.A6): (1.0, 1.0), (DOON, R.A6F): (0.79, 0.80)}))
    assert both["rule2"]["status"] == "FOOTPRINT" and both["rule1"]["consequence"] == "QUESTION"
    assert both["to_berk"] == ["the layout question (a cuOpt thread on X925 6): thread effect present on "
                               f"{DOON}; with rule 2's footprint reading on comp21"]
    mps = one(tmp_path / "c", world(tmp_path / "c", obj={R.A2: (120.0, 120.0)},
                                    p={(TOGURU, R.A6): (0.9, 0.9), (TOGURU, R.A6F): (0.5, 0.5)}))
    assert len(mps["to_berk"]) == 2 and mps["to_berk"][1].startswith("the MPS question")


def test_rule3_reads_cuopts_own_p_in_a2(tmp_path):
    """Rule 3 compares cuOpt's own P in A2@r3 (not the race's portfolio P) with A6@r3's P, in rule 1's band."""
    cpt = {(TOGURU, R.A2): (0.70, 0.70), (DOON, R.A2): (0.52, 0.56)}
    p = {(TOGURU, R.A2): (0.10, 0.10), (TOGURU, R.A6): (0.60, 0.60), (TOGURU, R.A6F): (0.60, 0.60),
         (DOON, R.A6): (0.50, 0.50)}                                       # rule 1 absent on both
    rd = one(tmp_path, world(tmp_path, p=p, cpt=cpt))
    t, d = rd["rule3"]["per"][TOGURU], rd["rule3"]["per"][DOON]
    assert t["present"] and t["diff"] == pytest.approx(0.10) and t["band"] == 0.05
    assert t["text"] == f"SCIP's presence recorded as a cause of cuOpt's slowdown on {TOGURU}"
    assert not d["present"] and d["diff"] == pytest.approx(0.04) and "ruled out" in d["text"]
    assert set(rd["rule3"]["per"]) == {TOGURU, DOON} and rd["to_berk"] == []      # rule 3 asks Berk nothing


@pytest.mark.parametrize("key, kw, leg, why", [
    ((TOGURU, R.A6, 0), dict(layout={"cuopt": ([1, 2, 3], 5)}), "layout", "cuopt --cpu-threads 5, expected 3"),
    ((TOGURU, R.A6F, 1), dict(layout={"cuopt": ([1, 2, 3], 3)}), "layout",
     "cuopt on cores [1, 2, 3], expected [1, 2, 3, 5, 6]"),
    ((TOGURU, R.A2, 0), dict(layout={"solver": ([6], None), "cuopt": ([1, 2, 3], 3)}), "layout",
     "solver on core 6, which stays idle"),
    ((TOGURU, R.A6, 0), dict(layout={"cuopt": ([1, 2, 3, 6], 3)}), "layout", "cuopt on core 6, which stays idle"),
    ((TOGURU, R.A2, 0), dict(layout={"solver": ([5], None), "cuopt": ([1, 2, 3], 3), "ea": ([4], None)}), "layout",
     "(an EA process)"),
    ((TOGURU, R.A6, 1), dict(harness=dict(EXT, cuopt_gap_zero=False)), "harness", "is not ext"),
    ((TOGURU, R.A6, 1), dict(harness=None), "harness", "harness None"),
    ((TOGURU, R.A6, 1), dict(compiled=True), "compile", "compiled on the clock"),
    ((TOGURU, R.A6, 1), dict(run_json=False), "layout", "no run.json"),
    ((DOON, R.A6F, 0), dict(pt=math.nan), "complete", "no P(120)"),
])
def test_validity_is_one_gate_and_names_the_leg(tmp_path, key, kw, leg, why):
    """Any run off the note's layout, off the ext harness, compiled on the clock or without P(120) makes the whole
    reading INVALID (D-027: validity holds before any rule is read); the report names the leg and the run."""
    rd = one(tmp_path, world(tmp_path, per={key: kw}))
    v = rd["validity"]
    assert v["verdict"] == "INVALID" and v["failing"] == [leg]
    assert any(why in x and f"{key[0]} {key[1]} s{key[2]}" in x for x in v["legs"][leg]), v["legs"][leg]
    assert rd["rule1"] is rd["rule2"] is rd["rule3"] is rd["to_berk"] is None
    txt = R.render(R.report(world(tmp_path / "b", per={key: kw}), tmp_path / "b"))
    assert f"VALIDITY INVALID" in txt and f"failing leg(s): {leg}" in txt and "RULES NOT READ" in txt


def test_a_missing_run_is_invalid(tmp_path):
    rd = one(tmp_path, world(tmp_path, drop={(TOGURU, R.A6, 1)}))
    assert rd["validity"]["verdict"] == "INVALID"
    assert rd["validity"]["legs"]["complete"] == [f"missing {TOGURU} A6@r3 s1"]
    assert rd["n_present"] == 15 and rd["rule1"] is None


def test_harness_from_the_table_when_run_json_has_none(tmp_path):
    runs = world(tmp_path)
    for rd_ in runs["run_dir"]:
        f = tmp_path / rd_ / "run.json"
        rj = json.loads(f.read_text())
        rj["config"] = {}
        f.write_text(json.dumps(rj))
    assert one(tmp_path, runs)["validity"]["verdict"] == "VALID"


def test_failed_runs_both_readings(tmp_path):
    """A failed run: `exclude` (the letter) makes the reading INVALID, `keep` reads it (D-002 §6) and names the
    deviation; `both` prints both readings."""
    runs = world(tmp_path, per={(TOGURU, R.A6, 0): dict(status="failed")},
                 p={(TOGURU, R.A6): (0.9, 0.9), (TOGURU, R.A6F): (0.5, 0.5)})
    rep = R.report(runs, tmp_path)
    assert [rd["keep_failed"] for rd in rep["readings"]] == [False, True] and len(rep["failed_runs"]) == 1
    ex, kp = rep["readings"]
    assert ex["validity"]["verdict"] == "INVALID"
    assert ex["validity"]["legs"]["complete"] == [f"failed (not completed) {TOGURU} A6@r3 s0"]
    assert kp["validity"]["verdict"] == "VALID" and len(kp["validity"]["deviation"]) == 1
    assert kp["rule1"]["consequence"] == "QUESTION"
    assert [rd["keep_failed"] for rd in R.report(runs, tmp_path, failed="exclude")["readings"]] == [False]
    assert [rd["keep_failed"] for rd in R.report(runs, tmp_path, failed="keep")["readings"]] == [True]
    assert [rd["keep_failed"] for rd in R.report(world(tmp_path / "b"), tmp_path / "b")["readings"]] == [False]
    txt = R.render(rep)
    assert "failed runs excluded (the letter)" in txt and "failed runs kept (D-002 §6)" in txt
    assert "stated deviation: " in txt


def test_replicate_beside_and_selection(tmp_path):
    """The A6F@r13x replicate (set itx-r13) is reported beside A6F@r3 and read by no rule. Only the set's three labels
    x seeds 0-1 x 120 s, not voided, are read: a `#<set>` copy loses to the bare id, the highest attempt wins, other
    sets / seeds / budgets / voided blocks never are (the set's own are listed)."""
    extra = [make(tmp_path, TOGURU, R.A6, 0, 0.99, run_id=f"{TOGURU}__A6@r3__s0__120s__x#other"),
             make(tmp_path, TOGURU, R.A6, 0, 0.98, set_="itx-r13"),
             make(tmp_path, TOGURU, R.A6, 2, 0.97),
             make(tmp_path, TOGURU, R.A6, 1, 0.96, budget=600.0),
             make(tmp_path, TOGURU, R.A6, 1, 0.95, block_void=True, run_id="voided"),
             make(tmp_path, TOGURU, "A6F@r13x", 0, 0.539, set_="itx-r13", run_id="r13x-0"),
             make(tmp_path, TOGURU, "A6F@r13x", 1, 0.559, set_="itx-r13", run_id="r13x-1")]
    runs = world(tmp_path, extra=extra)
    rd = one(tmp_path, runs)
    assert rd["validity"]["verdict"] == "VALID" and rd["rule1"]["per"][TOGURU]["a"] == [0.5, 0.5]
    assert [d["run_id"] for d in rd["duplicates"]] == [f"{TOGURU}__A6@r3__s0__120s__x#other"]
    assert rd["replicate"][TOGURU] == dict(r3=[0.5, 0.5], r13x=[0.539, 0.559])
    assert rd["replicate"][DOON] == dict(r3=[0.5, 0.5], r13x=[None, None]) and set(rd["replicate"]) == {TOGURU, DOON}
    others = R.report(runs, tmp_path)["other_rows"]
    assert len(others) == 3 and "voided" in others and "r13x-0" not in others     # the #other copy: a duplicate
    assert "r13x 0.539 / 0.559" in R.render(R.report(runs, tmp_path))
    again = world(tmp_path / "b", per={(DOON, R.A6, 0): dict(attempt=0)},
                  extra=[make(tmp_path / "b", DOON, R.A6, 0, 0.9, attempt=1, run_id="second")])
    assert one(tmp_path / "b", again)["rule1"]["per"][DOON]["a"] == [0.9, 0.5]


def test_main_prints_and_writes_json(tmp_path, capsys):
    runs = world(tmp_path, p={(DOON, R.A6): (1.0, 1.0), (DOON, R.A6F): (0.79, 0.80)},
                 obj={R.A2: (None, 147.0)})
    f = tmp_path / "runs.parquet"
    runs.to_parquet(f)
    assert R.main(["--runs", str(f), "--root", str(tmp_path), "--json", str(tmp_path / "d027.json")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("D-027 report") and "A2@r3 no incumbent / 147" in out
    js = json.loads((tmp_path / "d027.json").read_text())
    (rd,) = js["readings"]
    assert rd["validity"]["verdict"] == "VALID" and rd["rule1"]["consequence"] == "QUESTION"
    assert rd["rule2"]["status"] == "SCIP" and rd["rule2"]["a2"] == ["inf", 147.0] and len(rd["runs"]) == 16
    assert rd["runs"][f"{DOON} A6@r3 s0"]["pt"] == 1.0 and len(rd["to_berk"]) == 1
    assert R.main(["--runs", str(tmp_path / "nothing.parquet")]) == 2
