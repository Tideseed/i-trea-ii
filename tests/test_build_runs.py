"""build_runs on SYNTHETIC run directories (toy MPS, hand-made traces), the prereg refusal rules and the reproduce
chain. The helper `make_run` is imported by the other analysis tests (tests/ is on sys.path under pytest)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from itrea.analysis import build_runs as B
from itrea.analysis import prereg as PR
from itrea.benchmarks import toy
from itrea.config import EAConfig
from itrea.trace import Trace, save_incumbents

# mixed toy (min x0 + 2 x1 + 3 y0 - y1 + offset): feasible points and their objectives (offset 0)
X_85 = [2.0, 1.0, 1.5, 0.0]     # obj 8.5
X_70 = [3.0, 2.0, 0.0, 0.0]     # obj 7.0
X_BAD = [4.0, 0.5, 0.0, -3.0]   # fractional x1, r1 violated


def toy_problem(tmp: Path):
    p = toy.mixed()
    mps = toy.write_mps(p, tmp / "mixed.mps")
    return p, mps


def make_run(raw: Path, prob, mps: Path, arm: str, seed: int, budget: float, roles: dict, set_name: str = "toyset",
             posts=(), verdicts=(), bounds=(), gpu_health=None, exit_codes=None, cfg: str = "cfg000000000",
             stem: str = "mixed", run_json_extra=None, summary_extra=None, write_npz: bool = True) -> Path:
    """roles: {role: [(t, x, source), ...]}. posts: [(seq, t, obj)] on the EA; verdicts: [(seq, t, obj, accepted)] on
    the solver; bounds: [(t, bound)] on the solver."""
    rid = f"{stem}__{arm}__s{seed}__{budget:g}s__{cfg}"
    d = raw / set_name / rid
    for role, incs in roles.items():
        tr = Trace(d, role, instance=str(mps), arm=arm, seed=seed, budget=budget, config_hash=cfg,
                   **(summary_extra or {}).get(role, {}))
        recs = []
        for t, x, src in incs:
            x = np.asarray(x, dtype=float)
            obj = prob.objective(x)
            tr.incumbent(obj, src, t=t)
            recs.append((t, obj, src, x))
        if role == "ea":
            for s, t, o in posts:
                tr.event("post", t=t, seq=s, obj=o, gain=0.01)
        if role == "solver":
            for s, t, o, acc in verdicts:
                tr.event("verdict", t=t, seq=s, obj=o, accepted=acc)
            for t, b in bounds:
                tr.event("bound", t=t, bound=b)
        if write_npz:
            save_incumbents(d / f"{role}.incumbents.npz", prob.names, recs)
        tr.close(exit_reason="budget")
    rj = dict(run_id=rid, arm=arm, seed=seed, budget=budget, instance=str(mps), set=set_name, config_hash=cfg,
              exit_codes=exit_codes if exit_codes is not None else {r: 0 for r in roles})
    if gpu_health is not None:
        rj["gpu_health"] = gpu_health
    rj.update(run_json_extra or {})
    (d / "run.json").write_text(json.dumps(rj))
    return d


@pytest.fixture
def toyruns(tmp_path):
    p, mps = toy_problem(tmp_path)
    raw = tmp_path / "raw"
    make_run(raw, p, mps, "A0", 1000, 10, {"solver": [(2.0, X_85, "self"), (6.0, X_70, "self")]},
             bounds=[(0.5, float("-inf")), (1.0, 5.0)])
    make_run(raw, p, mps, "A5", 1000, 10, {"ea": [(1.0, X_85, "self"), (3.0, X_BAD, "self")]},
             gpu_health={"verdict": "void", "median_sm_mhz": 800.0, "foreign_share": 0.0})
    make_run(raw, p, mps, "A3", 1000, 10,
             {"solver": [(4.0, X_85, "self"), (5.0, X_70, "ea")], "ea": [(4.5, X_70, "self")]},
             posts=[(1, 4.5, 7.0), (2, 4.6, 7.0), (3, 9.0, 7.0), (4, 11.0, 7.0)],
             verdicts=[(2, 5.0, 7.0, True), (9, 6.0, 1.0, False)])
    out = tmp_path / "datasets"
    out.mkdir()
    return p, mps, raw, out


def test_parse_run_id_from_the_right():
    r = B.parse_run_id("ca__2014-09-01_reserves_0__A3__s1000__300s__abcdef012345")
    assert r == dict(stem="ca__2014-09-01_reserves_0", arm="A3", seed=1000, budget=300.0, config_hash="abcdef012345")
    assert B.parse_run_id("not-a-run") == {}


def test_build_hand_computed(toyruns):
    p, mps, raw, out = toyruns
    df, inc = B.build(raw, out, log=lambda *a: None)
    assert (out / "runs.parquet").exists() and (out / "incumbents_checked.parquet").exists()
    # every incumbent checked; the fractional one fails; EA vectors read by SCIP, SCIP's by HiGHS
    assert len(inc) == 7 and int((~inc["ok"]).sum()) == 1
    bad = inc[~inc["ok"]].iloc[0]
    assert "integrality" in bad["reason"] and bad["role"] == "ea"
    assert set(inc.loc[inc["role"] == "ea", "reader"]) == {"scip"}
    assert set(inc.loc[inc["role"] == "solver", "reader"]) == {"highs"}
    r = df.set_index("arm")
    g85 = 1.5 / 8.5                                         # Berthold gap of 8.5 against the reference 7.0
    assert r.loc["A0", "ref"] == 7.0 and r.loc["A0", "ref_kind"] == "best_of_study"
    assert r.loc["A0", "pt"] == pytest.approx((2 * 1 + 4 * g85) / 10)
    assert r.loc["A5", "pt"] == pytest.approx((1 + 9 * g85) / 10)       # the failed incumbent does not count
    assert r.loc["A5", "n_inc_failed"] == 1
    assert r.loc["A0+A5", "pt"] == pytest.approx((1 + 5 * g85) / 10)    # uncoupled portfolio: best of A0 and A5
    assert bool(r.loc["A0+A5", "synthetic"])
    assert r.loc["A3", "pt"] == pytest.approx((4 * 1 + 0.5 * g85) / 10)   # the EA's 7.0 at 4.5 counts for the pair
    assert r.loc["A3", "solver_pt"] == pytest.approx((4 + 1 * g85) / 10)
    assert r.loc["A3", "ea_pt"] == pytest.approx((4.5 + 0) / 10)
    assert np.isnan(r.loc["A5", "solver_pt"]) and np.isnan(r.loc["A0", "ea_pt"])
    assert r.loc["A3", "t_first"] == 4.0 and bool(r.loc["A3", "t_first_event"])
    assert r.loc["A3", "t_gap1"] == 4.5 and bool(r.loc["A3", "t_gap1_event"])
    # posts: seq1 superseded by seq2, seq2 accepted, seq3 pending at T, seq4 after T; verdict 9 has no post
    assert (r.loc["A3", "posts"], r.loc["A3", "posts_accepted"], r.loc["A3", "posts_superseded"],
            r.loc["A3", "posts_pending"], r.loc["A3", "posts_after_T"], r.loc["A3", "verdicts_unmatched"]) == \
        (3, 1, 1, 1, 1, 1)
    assert r.loc["A3", "solver_inc_from_ea"] == 1
    # GPU-health void of A5 voids the whole block (instance x seed x budget)
    assert bool(r.loc["A5", "gpu_void"]) and not bool(r.loc["A0", "gpu_void"])
    assert df["block_void"].all()
    # primal-dual integral of A0: bound 5 from t = 1 ("-inf" ignored), incumbents 8.5 at 2 and 7 at 6
    pdi = (1 * 1 + 1 * 1 + 4 * (3.5 / 8.5) + 4 * (2 / 7)) / 10
    assert r.loc["A0", "pdi"] == pytest.approx(pdi)


def test_build_is_deterministic_and_uses_the_cache(toyruns):
    p, mps, raw, out = toyruns
    df1, inc1 = B.build(raw, out, log=lambda *a: None)
    mps.unlink()                                            # without the instance only the cache can answer
    df2, inc2 = B.build(raw, out, log=lambda *a: None)
    pd.testing.assert_frame_equal(df1, df2)
    pd.testing.assert_frame_equal(inc1, inc2)
    _, inc3 = B.build(raw, out, recheck=True, log=lambda *a: None)
    assert not inc3["checked"].any() and inc3["reason"].str.startswith("unchecked").all()


def test_crash_reference_and_verdict_file(tmp_path):
    p, mps = toy_problem(tmp_path)
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    d = make_run(raw, p, mps, "A3", 1001, 10, {"solver": [(1.0, X_70, "self")], "ea": [(0.5, X_85, "self")]},
                 posts=[(1, 2.0, 7.0)], exit_codes={"solver": -11, "ea": 0}, run_json_extra={"t0": 100.0})
    # the verdict only in the mailbox file, on the monotonic clock (t0 = 100)
    (d / "verdicts.jsonl").write_text(json.dumps({"seq": 1, "t": 103.0, "obj": 7.0, "accepted": False}) + "\n")
    # a frozen reference for the instance
    pd.DataFrame([dict(instance="mixed", set="toyset", reference=6.0, sense=1, kind="reference_run",
                       ref_zero=False, source="test")]).to_csv(out / "reference.csv", index=False)
    df, _ = B.build(raw, out, log=lambda *a: None)
    r = df.iloc[0]
    assert bool(r["crashed"]) and r["crashed_roles"] == "solver"
    assert r["ref"] == 6.0 and bool(r["ref_frozen"]) and r["ref_kind"] == "reference_run"
    assert r["posts_rejected"] == 1 and r["verdict_source"] == "verdicts.jsonl"


def test_missing_incumbent_file_is_flagged(tmp_path):
    p, mps = toy_problem(tmp_path)
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    make_run(raw, p, mps, "A0", 1000, 10, {"solver": [(1.0, X_70, "self")]}, write_npz=False,
             exit_codes={"solver": 1})
    df, inc = B.build(raw, out, log=lambda *a: None)
    r = df.iloc[0]
    assert r["incumbents_missing"] == "solver" and r["pt"] == 1.0 and not bool(r["t_first_event"])
    assert bool(r["crashed"])


# ---------------------------------------------------------------------------------------------------- prereg
FROZEN_SOLVER = {"emphasis": "optimality", "cuopt_threads": 3}


def _frozen(tmp_path) -> tuple[Path, dict]:
    """frozen-e2.yaml and frozen-e2-solver.json (the non-EA arms' frozen settings, review F4) in tmp_path."""
    f = tmp_path / "frozen-e2.yaml"
    d = EAConfig().to_dict()
    d["name"] = "frozen-e2"
    f.write_text(yaml.safe_dump(d, sort_keys=False))
    (tmp_path / "frozen-e2-solver.json").write_text(json.dumps(FROZEN_SOLVER))
    return f, PR.expected_hashes(f, solver=FROZEN_SOLVER)


def _fake_runs(hashes: dict, n_inst: int = 12, seed: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    rs = np.random.default_rng(seed)
    pool = pd.DataFrame(dict(instance=[f"i{k:02d}" for k in range(n_inst)], side="test", stratum="miplib_opt",
                             p_degenerate=False))
    rows = []
    for inst in pool["instance"]:
        base = rs.uniform(0.2, 0.8)
        for s in PR.SEEDS:
            for arm, eff in (("A0", 0.0), ("A3", -0.1), ("A5", 0.05), ("A0+A5", -0.03), ("A2", -0.05),
                             ("A1", 0.0), ("A0star", -0.02), ("A5x0", 0.1)):
                pt = float(np.clip(base + eff + rs.normal(0, 0.01), 0, 1))
                rows.append(dict(run_id=f"{inst}__{arm}__s{s}", instance=inst, arm=arm, seed=s, budget=300.0,
                                 pt=pt, ea_pt=pt, block_void=False, backend_mismatch=False, attempt=0,
                                 config_hash=hashes.get(arm, "solver-only"), base_hash=hashes["__base__"]
                                 if arm in hashes else "", ref_frozen=True, crashed=False,
                                 synthetic=arm == "A0+A5", git_rev="", t_first=1.0, t_first_event=True,
                                 t_gap1=10.0, t_gap1_event=True, t_gap5=5.0, t_gap5_event=True))
    return pd.DataFrame(rows), pool


def test_prereg_confirmatory_and_refusal(tmp_path):
    f, hashes = _frozen(tmp_path)
    assert len({hashes["A5x0"], hashes["A3"], hashes["A5"]}) == 3 and hashes["A0+A5"] == hashes["A5"]
    runs, pool = _fake_runs(hashes)
    res = PR.run(runs, pool, hashes, git_check=False)
    assert res["mode"] == "confirmatory" and not res["problems"]
    c = res["comparisons"]
    assert c["H1a"]["decision"] == "supported" and c["H1a"]["n_instances"] == 12
    assert c["H1b"]["decision"] == "supported" and res["H1"] == "supported"
    assert c["H6"]["decision"] == "supported"                    # A5 better than A5x0 by construction
    assert "p_holm" in c["H3"] and c["H3"]["p_holm"] >= c["H3"]["p"]
    md = PR.markdown(res)
    assert "| H1a | A3 | A0 | 12 |" in md
    # a run under another configuration -> refused, unless exploratory
    bad = runs.copy()
    bad.loc[bad["arm"] == "A3", "config_hash"] = "deadbeef0000"
    with pytest.raises(PR.PreregError):
        PR.run(bad, pool, hashes, git_check=False)
    ex = PR.run(bad, pool, hashes, exploratory=True, git_check=False)
    assert ex["mode"] == "exploratory" and ex["problems"]
    # a provisional reference -> refused
    nf = runs.copy()
    nf["ref_frozen"] = False
    with pytest.raises(PR.PreregError):
        PR.run(nf, pool, hashes, git_check=False)
    # no frozen configuration at all -> refused
    with pytest.raises(PR.PreregError):
        PR.run(runs, pool, None, git_check=False)


def test_prereg_void_blocks_and_reruns(tmp_path):
    _, hashes = _frozen(tmp_path)
    runs, pool = _fake_runs(hashes, n_inst=6)
    void = runs[(runs["instance"] == "i00") & (runs["arm"] == "A3") & (runs["seed"] == 1000)].copy()
    runs.loc[void.index, "block_void"] = True
    rerun = void.assign(run_id="rerun", block_void=False, attempt=1, pt=0.0)
    r, info = PR.select_runs(pd.concat([runs, rerun], ignore_index=True), pool)
    sel = r[(r["instance"] == "i00") & (r["arm"] == "A3") & (r["seed"] == 1000)]
    assert len(sel) == 1 and sel.iloc[0]["run_id"] == "rerun" and info["void_runs"] == 1
    r2, info2 = PR.select_runs(runs[~((runs["instance"] == "i01") & (runs["arm"] == "A2"))], pool)
    assert "i01|A2|s1000" in info2["missing"]


# ------------------------------------------------------------------------------------------------- reproduce
def test_reproduce_chain(tmp_path):
    p, mps = toy_problem(tmp_path)
    root = tmp_path / "repo"
    raw = root / "traces" / "raw"
    (root / "datasets").mkdir(parents=True)
    env = dict(os.environ, ITREA_ROOT=str(root), OMP_NUM_THREADS="1")

    def py(*args):
        return subprocess.run([sys.executable, "-m", *args], cwd=root, env=env, capture_output=True, text=True,
                              timeout=600)

    r = py("itrea.analysis.reproduce")
    assert r.returncode == 2, r.stdout + r.stderr                            # no traces
    make_run(raw, p, mps, "A0", 1000, 10, {"solver": [(2.0, X_85, "self"), (6.0, X_70, "self")]})
    assert py("itrea.analysis.build_runs").returncode == 0
    r = py("itrea.analysis.reproduce")
    assert r.returncode == 0, r.stdout + r.stderr
    # change a trace: the committed runs.parquet no longer follows from the traces
    make_run(raw, p, mps, "A0", 1001, 10, {"solver": [(3.0, X_70, "self")]})
    r = py("itrea.analysis.reproduce")
    assert r.returncode == 1 and "changed" in r.stdout


def test_registered_overrides_match_the_runner(tmp_path):
    pair = pytest.importorskip("itrea.experiments.pair")
    f, hashes = _frozen(tmp_path)
    for arm in ("A3", "A5", "A5x0"):
        job = dict(instance="x.mps", set="s", arm=arm, seed=1000, budget=300, config=str(f))
        cfg, _ = pair.effective_config(job, pair.PairOptions())
        assert cfg.hash() == hashes[arm], arm
        assert pair.base_config_hash(job) == hashes["__base__"]


def test_runner_record_fields(tmp_path):
    """run.json as itrea.experiments.pair writes it: config.ea_hash, status void, processes, env.git; the SCIP
    summary's callbacks and exchange; verdict rows with stored; a void run voids its block."""
    p, mps = toy_problem(tmp_path)
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    rj = dict(config=dict(ea_hash="eahash000000", base_hash="basehash0000", overrides={"backend": "cupy"}),
              status="completed", base_arm="A3", window="w1", position=2, block="b7",
              processes=[dict(role="solver", exit_code=0, startup_s=1.5), dict(role="ea", exit_code=0, startup_s=4.0)],
              env=dict(git=dict(revision="abc123", prereg_tag="def456")))
    summ = {"solver": dict(solver="scip", status="optimal", exit_reason="optimal", nodes_per_s=10.0,
                           exchange=dict(seconds=0.25, bytes_out=100, bytes_in=50),
                           callbacks=dict(t_first_callback=0.1, t_first_solving_callback=6.0, max_gap_s=0.5))}
    make_run(raw, p, mps, "A3", 1000, 10, {"solver": [(1.0, X_70, "self")], "ea": [(2.0, X_70, "self")]},
             posts=[(1, 2.0, 7.0)], verdicts=[(1, 2.5, 7.0, True)], run_json_extra=rj, summary_extra=summ,
             cfg="eahash000000")
    make_run(raw, p, mps, "A0", 1000, 10, {"solver": [(1.0, X_70, "self")]},
             run_json_extra=dict(status="void", void_cause="abort"))
    df, _ = B.build(raw, out, log=lambda *a: None)
    r = df.set_index("arm")
    a3 = r.loc["A3"]
    assert a3["config_hash"] == "eahash000000" and a3["base_hash"] == "basehash0000"
    assert (a3["window_id"], a3["williams_pos"], a3["block"], a3["git_rev"]) == ("w1", 2, "b7", "abc123")
    assert a3["solver_startup_s"] == 1.5 and a3["ea_startup_s"] == 4.0 and a3["startup_s"] == 4.0
    assert a3["exchange_s"] == 0.25 and a3["exchange_bytes"] == 150 and bool(a3["solved_optimal"])
    assert a3["cb_first_solving_s"] == 6.0 and bool(a3["no_exchange_window"])         # 6 s > T/2 = 5 s
    assert a3["posts_accepted"] == 1
    assert r.loc["A0", "status"] == "void" and df["block_void"].all()


def test_prereg_cli_outputs_are_deterministic(tmp_path):
    f, hashes = _frozen(tmp_path)
    runs, pool = _fake_runs(hashes, n_inst=8)
    runs.to_parquet(tmp_path / "runs.parquet", index=False)
    pool.to_csv(tmp_path / "pool.csv", index=False)
    args = ["--runs", str(tmp_path / "runs.parquet"), "--pool", str(tmp_path / "pool.csv"), "--frozen", str(f),
            "--reference", str(tmp_path / "none.csv"), "--out", str(tmp_path / "res.json"),
            "--md", str(tmp_path / "gen" / "res.md"), "--no-git-check",
            "--frozen-solver", str(tmp_path / "frozen-e2-solver.json")]
    assert PR.main(args) == 0
    first = (tmp_path / "res.json").read_bytes(), (tmp_path / "gen" / "res.md").read_bytes()
    assert PR.main(args) == 0
    assert first == ((tmp_path / "res.json").read_bytes(), (tmp_path / "gen" / "res.md").read_bytes())
    res = json.loads(first[0])
    assert res["mode"] == "confirmatory" and res["H1"] == "supported" and "NaN" not in first[0].decode()
    # a wrong configuration: the CLI refuses (exit 3) and writes nothing new
    runs.assign(config_hash="0" * 12).to_parquet(tmp_path / "runs.parquet", index=False)
    (tmp_path / "res.json").unlink()
    assert PR.main(args) == 3 and not (tmp_path / "res.json").exists()


def test_screening_p_is_stable_under_the_reference_freeze(tmp_path):
    """The pool ranks on pt_screen (best SCREEN run as reference); a later reference.csv moves pt, not pt_screen."""
    p, mps = toy_problem(tmp_path)
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    make_run(raw, p, mps, "SCREEN", 900, 10, {"solver": [(2.0, X_85, "self")]}, set_name="pglib_uc")
    make_run(raw, p, mps, "REF", 1, 10, {"solver": [(1.0, X_70, "self")]}, set_name="pglib_uc")
    df1, _ = B.build(raw, out, log=lambda *a: None)
    s1 = df1[df1["arm"] == "SCREEN"].iloc[0]
    assert s1["ref_screen"] == 8.5 and s1["pt_screen"] == pytest.approx(0.2)    # gap 1 for 2 s, then 0
    assert s1["ref"] == 7.0 and s1["ref_kind"] == "screening_best_provisional"  # SCREEN + REF best
    pd.DataFrame([dict(instance="mixed", set="pglib_uc", reference=6.5, sense=1, kind="reference_run",
                       ref_zero=False, source="REF 1 h")]).to_csv(out / "reference.csv", index=False)
    df2, _ = B.build(raw, out, log=lambda *a: None)
    s2 = df2[df2["arm"] == "SCREEN"].iloc[0]
    assert s2["pt"] != s1["pt"] and s2["pt_screen"] == s1["pt_screen"]


def test_unchecked_rows_are_not_cached(tmp_path):
    p, mps = toy_problem(tmp_path)
    raw, out = tmp_path / "raw", tmp_path / "datasets"
    out.mkdir()
    make_run(raw, p, mps, "A0", 1000, 10, {"solver": [(1.0, X_70, "self")]})
    data = mps.read_bytes()
    mps.unlink()
    _, inc1 = B.build(raw, out, log=lambda *a: None)
    assert not inc1["checked"].any()
    mps.write_bytes(data)                                    # the instance is back: the next build checks it
    _, inc2 = B.build(raw, out, log=lambda *a: None)
    assert inc2["checked"].all() and inc2["ok"].all()


def test_prereg_ignores_smoke_runs_and_flags_ambiguity(tmp_path):
    _, hashes = _frozen(tmp_path)
    runs, pool = _fake_runs(hashes, n_inst=4)
    runs["split"] = np.where(runs["synthetic"], "", "test")
    smoke = runs[(runs["instance"] == "i00") & (runs["arm"] == "A3")].assign(split="smoke", run_id="smoke", pt=0.0)
    r, info = PR.select_runs(pd.concat([runs, smoke], ignore_index=True), pool)
    assert "smoke" not in set(r["run_id"]) and not info["ambiguous"]
    twin = runs[(runs["instance"] == "i01") & (runs["arm"] == "A2")].assign(run_id="twin")
    with pytest.raises(PR.PreregError, match="several non-void runs"):
        PR.run(pd.concat([runs, twin], ignore_index=True), pool, hashes, git_check=False)
