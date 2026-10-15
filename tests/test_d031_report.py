"""`tools/d031_report.py` (D-031: the phase-1 interim result — C1 on the frozen-v2 candidate ysync + ld0 read on four
seeds, D-030's set `c1r-ysync-ld0` at seeds 0-1 + D-031's set `c1r-ysync-ld0-s23` at seeds 2-3) on synthetic run tables
and run directories — no solver, no GPU. The job file: one call of D-030's writer at seeds 2-3, every field D-030's but
seed / set / stratum / block / position, pinned run hashes, new run ids, sha256 pinned. Validity over 136 runs (D-030's
own six legs + the same six on D-031's set); C1's hint and guard exactly as D-022's functions compute them on four-seed
means (each seed weighted equally); the harm leg cross-day against A9@r13x's seeds 0-1 with D-030's bands; seeds 0-1
alone equal to D-030's reading and seeds 2-3 alone beside, with whether the halves agree (no threshold); THE phase-1
interim result = the four-seed hint and guard when 136 of 136 read VALID, else D-030's reading; failed runs kept, the
letter beside; the CLI."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itrea.analysis.stats import effect_d
from itrea.experiments import make_jobs as mj
from itrea.experiments import pair
from itrea.experiments.pool import make_blocks

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("d031_report", ROOT / "tools" / "d031_report.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
R, D22 = M.R, M.D22

EXT = pair.resolve_harness("ext")
DEV = M.development_instances()
ELEVEN = list(M.C1_INSTANCES)
OTHERS = [i for i in DEV if i not in ELEVEN]
A9, A6F = M.A9, M.A6F


class World:
    """Both sets complete and valid (every P 0.4 unless set; seeds 0-1 in D-030's set, 2-3 in D-031's), D-022's 68
    C1 runs of 10-13 (set itx-r13, all 0.4) and A0's runs on the 17 (set dev, 0.8)."""

    def __init__(self, root: Path):
        self.root = root
        self.exp = {**{(k, M.SET_01): v for k, v in R.expected(M.A9C, M.SET_01).items()},
                    **{(k, M.SET_23): v for k, v in M.expected_23().items()}}
        self.p = {k: 0.4 for k, _ in self.exp}
        self.old = {(i, lab, s): 0.4 for i in DEV for lab in (R.OLD_A9, R.OLD_A6F) for s in (0, 1)}
        self.a0 = {(i, s): 0.8 for i in DEV for s in (0, 1)}
        self.row: dict = {}
        self.rj: dict = {}
        self.summ: dict = {}
        self.window: dict = {}
        self.drop: set = set()

    def set_p(self, inst: str, label: str, seeds_p, seeds=(0, 1, 2, 3)) -> None:
        for s, v in zip(seeds, seeds_p):
            self.p[(inst, label, s)] = v

    def set_old(self, inst: str, label: str, seeds_p) -> None:
        for s, v in zip((0, 1), seeds_p):
            self.old[(inst, label, s)] = v

    def build(self) -> pd.DataFrame:
        rows = []
        for (k, set_), rid in self.exp.items():
            if k in self.drop:
                continue
            inst, lab, s = k
            rw = dict(run_id=rid, set=set_, instance=inst, arm=lab, seed=s, budget=120.0, status="completed",
                      attempt=0, pt=self.p[k], harness=json.dumps(EXT, sort_keys=True), cuda_compiled_on_clock=False,
                      synthetic=False, block_void=False, window_id=self.window.get(k, f"W-{set_}"))
            rw.update(self.row.get(k, {}))
            d = self.root / "traces" / "raw" / set_ / str(rw["run_id"]).split("#")[0]
            d.mkdir(parents=True, exist_ok=True)
            rj = dict(run_id=rid, config=dict(harness=EXT), compiled_on_clock_cuda=False,
                      cupy_cache=dict(compiled_on_clock=False))
            rj.update(self.rj.get(k, {}))
            (d / "run.json").write_text(json.dumps(rj))
            if lab == A9 and not (k in self.summ and self.summ[k] is None):
                cs = dict(state="verified", sched_active="yield", sched_at_end="yield", active=True)
                sm = dict(P=48, gene_dtype="float64", compiled_on_clock=False, cuda_sched=cs, phase_events_block=False,
                          generations=100 + s, lp_decode=dict(gate="open", gate_why="lp_done", gate_t=20.0))
                sm.update(self.summ.get(k) or {})
                (d / "ea.summary.json").write_text(json.dumps(sm))
            rw["run_dir"] = str(d.relative_to(self.root))
            rows.append(rw)
        for (i, lab, s), v in self.old.items():
            rows.append(dict(run_id=f"{i}__{lab}__s{s}__120s__old", set="itx-r13", instance=i, arm=lab, seed=s,
                             budget=120.0, status="completed", attempt=0, pt=v, synthetic=False, block_void=False,
                             window_id="W0", run_dir=""))
        for (i, s), v in self.a0.items():
            rows.append(dict(run_id=f"{i}__A0__s{s}__120s__a0", set="dev", instance=i, arm="A0", seed=s, budget=120.0,
                             status="completed", attempt=0, pt=v, synthetic=False, block_void=False, run_dir=""))
        return pd.DataFrame(rows)

    def report(self, failed: str = "both") -> dict:
        return M.report(self.build(), self.root, failed)


def _first(rep) -> dict:
    return rep["readings"][0]


# ============================================================================================ the job file
def _file_jobs() -> list[dict]:
    return [json.loads(x) for x in (ROOT / M.JOBS_OUT).read_text().splitlines()]


def test_job_file_is_the_writers_output_and_pinned(tmp_path):
    """jobs/d031.jsonl is exactly what `--write-jobs` writes (one call of make_jobs.c1_rerun at seeds 2-3), its sha256
    is the one D-031 records, and the report's expected ids are its ids."""
    out = tmp_path / "jobs" / "d031.jsonl"
    p, h = M.write(out, tmp_path, None)
    assert p.read_bytes() == (ROOT / M.JOBS_OUT).read_bytes()
    assert h == M.sha256(ROOT / M.JOBS_OUT) == M.JOBS_SHA256
    jobs = _file_jobs()
    assert len(jobs) == 68 and {j["set"] for j in jobs} == {j["stratum"] for j in jobs} == {M.SET_23}
    assert {pair.arm_label(j) for j in jobs} == {A9, A6F} and {j["seed"] for j in jobs} == {2, 3}
    assert {j["name"] for j in jobs} == set(DEV) and {j["harness"] for j in jobs} == {"ext"}
    assert {j["budget"] for j in jobs} == {120.0} and {j["split"] for j in jobs} == {"dev"}
    assert {(j["config"], pair.run_hash(j)) for j in jobs if j["arm"] == "A9"} == {(M.CONFIG, M.A9_HASH)}
    assert {(j["config"], pair.run_hash(j)) for j in jobs if j["arm"] == "A6F"} == {(None, M.A6F_HASH)}
    assert set(M.expected_23().values()) == {pair.run_id_for(j) for j in jobs}
    assert M.job_file_state(ROOT)["state"] == "same"


def test_d031_jobs_are_d030s_at_seeds_2_3():
    """Every D-031 job equals D-030's job of the same instance and arm but for seed / set / stratum / block /
    position (the same configuration file, harness and run hash); the arms alternate by seed as in D-030's file —
    A6F first at even seeds — so each instance runs both orders over seeds 2-3 as over 0-1."""
    d030 = {(j["name"], j["arm"]): j for j in map(json.loads, (ROOT / "jobs" / "c1r-ysync-ld0.jsonl").read_text()
                                                  .splitlines()) if j["seed"] == 0}
    place = {"seed", "set", "stratum", "block", "position"}
    for j in _file_jobs():
        old = d030[(j["name"], j["arm"])]
        assert {k: v for k, v in j.items() if k not in place} == {k: v for k, v in old.items() if k not in place}
        assert pair.run_hash(j) == pair.run_hash(old)
        assert j["block"] == f"{j['name']}__s{j['seed']}__{M.SET_23}"
    for b in make_blocks(_file_jobs()):
        order = [r.label for r in sorted(b.runs, key=lambda r: r.job["position"])]
        assert order == ([A6F, A9] if b.seed % 2 == 0 else [A9, A6F]), (b.id, order)


def test_run_ids_are_new():
    """No D-031 run id is the id of a job in any other file under jobs/, nor of a row of another set in the run table
    (when this checkout has one), nor one of D-030's; each is new by its seed."""
    ids = {pair.run_id_for(j) for j in _file_jobs()}
    assert len(ids) == 68 and all("__s2__" in i or "__s3__" in i for i in ids)
    taken = M.taken_ids(ROOT, None, skip=ROOT / M.JOBS_OUT)
    assert not ids & taken["jobs"]
    assert not ids & set(R.expected(M.A9C, M.SET_01).values())
    runs = ROOT / "datasets" / "runs.parquet"
    if runs.exists():
        df = pd.read_parquet(runs, columns=["run_id", "set"])
        df = df[df["set"].astype(str) != M.SET_23]
        assert not ids & {str(x).split("#")[0] for x in df["run_id"]}


def test_writer_refuses_moved_pins_and_used_ids(tmp_path, monkeypatch):
    out = tmp_path / "jobs" / "d031.jsonl"
    rid = next(iter(M.expected_23().values()))
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs" / "other.jsonl").write_text(json.dumps(M.jobs_23()[0]) + "\n")
    with pytest.raises(ValueError, match="already used"):
        M.write(out, tmp_path, None)
    (tmp_path / "jobs" / "other.jsonl").unlink()
    with pytest.raises(ValueError, match="already used"):
        M.write(out, tmp_path, pd.DataFrame(dict(run_id=[rid + "#x"])))
    (tmp_path / "traces" / "raw" / M.SET_23 / rid).mkdir(parents=True)
    with pytest.raises(ValueError, match="already used"):
        M.write(out, tmp_path, None)
    (tmp_path / "traces" / "raw" / M.SET_23 / rid).rmdir()
    assert M.write(out, tmp_path, None)[1] == M.JOBS_SHA256           # rewriting its own target is allowed
    assert M.write(out, tmp_path, None)[1] == M.JOBS_SHA256
    monkeypatch.setitem(mj.C1R_A9, M.A9C, (M.CONFIG, "000000000000"))
    with pytest.raises(ValueError, match="moved"):
        M.jobs_23()
    monkeypatch.setitem(mj.C1R_A9, M.A9C, (M.CONFIG, M.A9_HASH))
    monkeypatch.setattr(mj, "C1R_A6F_HASH", "000000000000")
    with pytest.raises(ValueError, match="moved"):
        M.expected_23()


# ============================================================================================ the rules
def test_valid_four_seeds_is_the_phase1_result(tmp_path):
    """A9 0.3 vs A6F 0.4 on the fixed 11 at every seed, equal elsewhere: 136 of 136 VALID; four-seed hint 0.75
    present, guard 11 x 0.25 / 17 holds, no harm (the new A9 at or below A9@r13x's 0.4); the phase-1 result is the
    four-seed reading; the halves agree."""
    w = World(tmp_path)
    for i in ELEVEN:
        w.set_p(i, A9, (0.3,) * 4)
    rep = w.report()
    assert len(rep["readings"]) == 1 and not rep["failed_runs"]
    rd = _first(rep)
    assert rd["validity"]["verdict"] == "VALID" and rd["validity"]["n_present"] == 136
    four = rd["four"]
    assert four["seeds"] == [0, 1, 2, 3] and four["n_runs"] == 136
    assert four["hint"]["state"] == "present" and four["hint"]["ratio"] == pytest.approx(0.75)
    assert four["guard"]["state"] == "holds" and four["guard"]["mean_d"] == pytest.approx(11 * 0.25 / 17)
    assert four["harm"]["state"] == "none" and not four["harm"]["unread"]
    p1 = rd["phase1"]
    assert not p1["fallback"] and p1["seeds"] == [0, 1, 2, 3] and p1["hint"] is four["hint"]
    assert rd["same_as_d030"]["ok"] and rd["halves"]["hint_agree"] and rd["halves"]["guard_agree"]
    assert rd["halves"]["count"] == {"same": 17, "opposite": 0, "a zero": 0}          # both d = 0 is the same sign
    text = M.render(rep)
    assert "VALIDITY VALID (both sets): 136 of 136" in text
    assert "PHASE-1 INTERIM RESULT (four seeds 0-3 (D-030 + D-031, 136 runs)): HINT present (ratio 0.750" in text
    assert "go / no-go on C1 is Berk's" in text


def test_four_seed_means_are_d022s_functions_on_equal_seed_weights(tmp_path):
    """On irregular P values the four-seed hint and guard equal (a) D-022's c1_hint / c1_guard called directly on all
    four seeds under D-022's labels and (b) the definitions by hand: P̄ = (s0 + s1 + s2 + s3) / 4 = the mean of the two
    halves' seed means; ratio of the means of P̄ over the 11; mean of effect_d(P̄ A6F, P̄ A9) over the 17. The halves
    alone equal the same definitions on their two seeds."""
    rng = np.random.default_rng(31)
    w = World(tmp_path)
    for i in DEV:
        w.set_p(i, A9, rng.uniform(0.0, 1.0, 4))
        w.set_p(i, A6F, rng.uniform(0.0, 1.0, 4))
    for lab in (A9, A6F):                                         # both P = 1: d = 0
        w.set_p("rail01", lab, (1.0,) * 4)
    rd = _first(w.report())
    four, s01, s23 = rd["four"], rd["s01"], rd["s23"]
    long = pd.DataFrame([dict(instance=i, arm=R.TO_D022[lab], seed=s, pt=v) for (i, lab, s), v in w.p.items()])
    assert four["hint"]["ratio"] == pytest.approx(D22.c1_hint(long, ELEVEN)["ratio"])
    assert four["guard"]["mean_d"] == pytest.approx(D22.c1_guard(long, DEV)["mean_d"])

    def m(i, lab, seeds):
        return float(np.mean([w.p[(i, lab, s)] for s in seeds]))
    for seeds, ru in (((0, 1, 2, 3), four), ((0, 1), s01), ((2, 3), s23)):
        assert ru["hint"]["ratio"] == pytest.approx(np.mean([m(i, A9, seeds) for i in ELEVEN])
                                                    / np.mean([m(i, A6F, seeds) for i in ELEVEN]))
        assert ru["guard"]["mean_d"] == pytest.approx(np.mean([effect_d(m(i, A6F, seeds), m(i, A9, seeds))
                                                               for i in DEV]))
    for i in DEV:
        assert m(i, A9, (0, 1, 2, 3)) == pytest.approx((m(i, A9, (0, 1)) + m(i, A9, (2, 3))) / 2)
        assert four["table"][i]["p_a9"] == pytest.approx(m(i, A9, (0, 1, 2, 3)))
    assert four["table"]["rail01"]["d"] == 0.0
    assert rd["same_as_d030"]["ok"]                               # seeds 0-1 = c1r_report's D-030 reading
    ru30 = rd["d030"]["rules"]
    assert s01["hint"]["ratio"] == pytest.approx(ru30["hint"]["ratio"])
    assert s01["guard"]["mean_d"] == pytest.approx(ru30["guard"]["mean_d"])


def test_renumbered_seeds_change_nothing_but_the_name():
    rows = [dict(instance="glass4", arm=lab, seed=s, pt=0.1 * s) for lab in (A9, A6F) for s in (0, 1, 2, 3)]
    assert sorted(set(M.renumbered(rows, (2, 3))["seed"])) == [0, 1]
    assert sorted(set(M.renumbered(rows, (0, 1, 2, 3))["seed"])) == [0, 1, 2, 3]
    assert M.renumbered(rows, (2, 3))["pt"].tolist() == pytest.approx([0.2, 0.3, 0.2, 0.3])


def test_halves_disagreeing_are_shown_not_judged(tmp_path):
    """Seeds 0-1 with a hint (A9 0.2 on the 11), seeds 2-3 without (A9 0.6): each half is read and printed, the sign
    of d vs A6F flips on the 11, and the four-seed reading is what leads (ratio (0.2 + 0.6) / 2 / 0.4 = 1.0)."""
    w = World(tmp_path)
    for i in ELEVEN:
        w.set_p(i, A9, (0.2, 0.2, 0.6, 0.6))
    rd = _first(w.report())
    hv = rd["halves"]
    assert hv["hint"] == ("present", "absent") and not hv["hint_agree"]
    assert hv["count"]["opposite"] == 11 and set(hv["opposite"]) == set(ELEVEN)
    assert rd["four"]["hint"]["state"] == "absent" and rd["four"]["hint"]["ratio"] == pytest.approx(1.0)
    assert rd["phase1"]["hint"]["state"] == "absent" and not rd["phase1"]["fallback"]
    assert "DO THE HALVES AGREE? (reported, no threshold): hint present / absent" in M.render(w.report())


def test_harm_on_four_seeds_uses_d030s_bands_and_r13xs_seeds_0_1(tmp_path):
    """The harm leg on four seeds: d_i = effect_d(P̄ A9@r13x over its seeds 0-1, P̄ new over seeds 0-3), band_i and the
    threshold c1r_report.harm_bands' (10-13's runs only), harm iff d_i < -band_i. A loss at seeds 0-1 that seeds 2-3
    take back does not fire on four seeds; the seeds 0-1 leg equals c1r_report.harm_leg on the same runs."""
    w = World(tmp_path)
    w.set_old("glass4", R.OLD_A9, (0.3, 0.3))
    w.set_p("glass4", A9, (0.4, 0.4, 0.3, 0.3))        # 0-1: d -0.25 fires; 0-3: P̄ 0.35, d -0.143 fires too
    w.set_old("sing44", R.OLD_A9, (0.3, 0.3))
    w.set_p("sing44", A9, (0.4, 0.4, 0.2, 0.2))        # 0-1: fires; 0-3: P̄ 0.3, d 0 -> no harm
    w.set_old("comp21-2idx", R.OLD_A9, (0.3, 0.3))
    w.set_p("comp21-2idx", A9, (0.3, 0.3, 0.5, 0.5))   # 0-1: none; 0-3: P̄ 0.4, d -0.25 fires
    w.set_old("rail01", R.OLD_A9, (1.0, 1.0))           # cannot fire
    w.set_p("rail01", A9, (1.0,) * 4)
    rd = _first(w.report())
    four, s01, s23 = rd["four"]["harm"], rd["s01"]["harm"], rd["s23"]["harm"]
    rows = {r["instance"]: r for r in four["rows"]}
    assert four["harm"] == [i for i in DEV if i in ("glass4", "comp21-2idx")]
    assert rows["glass4"]["d"] == pytest.approx(effect_d(0.3, 0.35)) and rows["glass4"]["band"] == pytest.approx(0.05)
    assert rows["sing44"]["d"] == pytest.approx(0.0) and not rows["sing44"]["harm"]
    assert s01["harm"] == [i for i in DEV if i in ("glass4", "sing44")]
    assert s23["harm"] == ["comp21-2idx"]
    assert "rail01" in four["cannot_fire"] and rows["rail01"]["threshold"] > 1
    ctx = M.context(w.build())
    assert ctx["bands"] == R.harm_bands(ctx["p_old"], ctx["a6_old"], DEV)
    new01 = {(i, s): w.p[(i, A9, s)] for i in DEV for s in (0, 1)}
    ref = R.harm_leg(new01, ctx["p_old"], DEV, ctx["a6_old"])
    mine = M.cross_day(new01, ctx["p_old"], ctx["bands"], DEV, (0, 1))
    assert mine["harm"] == ref["harm"]
    for a, b in zip(mine["rows"], ref["rows"]):
        assert (a["instance"], a["harm"], a["band"], a["threshold"]) == (b["instance"], b["harm"], b["band"],
                                                                          b["threshold"])
        assert a["d"] == pytest.approx(b["d"])
    assert rd["same_as_d030"]["ok"]


def test_the_bands_are_d030s_table_on_10_13s_values(tmp_path):
    """With 10-13's recorded seeds (D-030's amendment table) the thresholds are that table's."""
    w = World(tmp_path)
    table = {"glass4": ((0.2212, 0.2108), (0.3064, 0.2890), 0.2290), "sing44": ((0.4412, 0.4490), (0.5241, 0.4570),
                                                                                  0.5105),
             "neos-960392": ((0.0654, 0.0502), (0.5260, 0.8932), 0.0981),
             "neos-2657525-crna": ((0.7518, 0.1704), (0.7272, 0.7859), None)}
    for i, (a9, a6, _) in table.items():
        w.set_old(i, R.OLD_A9, a9)
        w.set_old(i, R.OLD_A6F, a6)
    bands = M.context(w.build())["bands"]
    for i, (_, _, thr) in table.items():
        if thr is None:
            assert bands[i]["cannot_fire"]
        else:
            assert bands[i]["threshold"] == pytest.approx(thr, abs=1e-4)


# ============================================================================================ validity
@pytest.mark.parametrize("what", ["none", "harness", "config", "cupy_a9", "window", "flag", "drop"])
def test_read_set_is_c1r_reports_validity_on_d030s_set(tmp_path, what):
    """The six legs `read_set` applies to D-031's set give, on D-030's set, exactly c1r_report.reading's legs."""
    w = World(tmp_path)
    k9, k6 = ("sing44", A9, 1), ("sing44", A6F, 1)
    if what == "harness":
        w.rj[k6] = dict(config=dict(harness={}))
    elif what == "config":
        w.row[k9] = dict(run_id=w.exp[(k9, M.SET_01)].rsplit("__", 1)[0] + "__28e5a5e9c725")
    elif what == "cupy_a9":
        w.rj[k9] = dict(cupy_cache=dict(compiled_on_clock=True))
    elif what == "window":
        w.window[k9] = "W-other"
    elif what == "flag":
        w.summ[k9] = dict(cuda_sched=dict(state="read", sched_active="auto"))
    elif what == "drop":
        w.drop.add(k6)
    runs, ext = w.build(), M.C.ext_harness()
    for keep in (True, False):
        ref = R.reading(runs, tmp_path, M.A9C, M.SET_01, keep, ext)
        mine = M.read_set(runs, tmp_path, M.SET_01, (0, 1), R.expected(M.A9C, M.SET_01), keep, ext)
        assert mine["validity"] == ref["validity"] and mine["n_present"] == ref["n_present"]


@pytest.mark.parametrize("what, leg", [("harness", "harness"), ("config", "config"), ("cuda_a6f", "compile"),
                                       ("cupy_a9", "compile"), ("window", "window"), ("flag", "flag")])
def test_a_failing_leg_on_d031s_set_is_invalid_and_d030_leads(tmp_path, what, leg):
    """A failing leg among D-031's runs: INVALID with the set and leg named, no four-seed rule read, seeds 2-3 not
    read; seeds 0-1 (D-030's reading) still read, and it is the phase-1 result, the fallback stated."""
    w = World(tmp_path)
    k9, k6 = ("comp21-2idx", A9, 3), ("comp21-2idx", A6F, 3)
    if what == "harness":
        w.rj[k6] = dict(config=dict(harness={}))
    elif what == "config":
        w.row[k9] = dict(run_id=w.exp[(k9, M.SET_23)].rsplit("__", 1)[0] + "__341267810eed")   # ld0 alone
    elif what == "cuda_a6f":
        w.rj[k6] = dict(compiled_on_clock_cuda=True)
    elif what == "cupy_a9":
        w.rj[k9] = dict(cupy_cache=dict(compiled_on_clock=True))
    elif what == "window":
        w.window[k9] = "W-other"
    else:
        w.summ[k9] = dict(cuda_sched=dict(state="verified", sched_active="yield", sched_at_end="auto"))
    rd = _first(w.report())
    assert rd["validity"]["verdict"] == "INVALID" and rd["validity"]["invalid"] == [(M.SET_23, [leg])]
    assert rd["four"] is None and rd["s23"] is None and rd["s01"] is not None
    p1 = rd["phase1"]
    assert p1["fallback"] and p1["seeds"] == [0, 1] and f"{M.SET_23} INVALID ({leg})" in p1["source"]
    text = M.render(w.report())
    assert "FOUR-SEED RULES NOT READ" in text and "SEEDS 2-3 ALONE (D-031'S RUNS): not read" in text


def test_d030_invalid_leaves_no_phase1_result(tmp_path):
    w = World(tmp_path)
    w.summ[("glass4", A9, 0)] = dict(phase_events_block=True)
    rd = _first(w.report())
    assert rd["validity"]["invalid"] == [(M.SET_01, ["flag"])] and rd["s01"] is None and rd["s23"] is not None
    assert rd["phase1"]["hint"] is None and "INVALID" in rd["phase1"]["source"]


def test_d031_incomplete_falls_back_to_d030s_reading(tmp_path):
    """A D-031 run missing outside the 11 (glass4): INCOMPLETE 135 of 136; the four-seed hint reads (its 88 runs are
    in) but the guard has no partial mean, so the phase-1 result is D-030's reading (seeds 0-1), the four-seed hint
    beside; a missing run of the 11 leaves the four-seed hint INCOMPLETE too. Whole blocks in other windows are fine."""
    w = World(tmp_path)
    for i in ELEVEN:
        w.set_p(i, A9, (0.3,) * 4)
    for lab in (A9, A6F):
        w.window[("rail01", lab, 2)] = "W-friday-2"                   # a whole block in a second window
    w.drop.add(("glass4", A9, 3))
    rd = _first(w.report())
    assert rd["validity"]["verdict"] == "INCOMPLETE" and rd["validity"]["n_present"] == 135
    assert rd["validity"]["incomplete"] == [M.SET_23]
    assert rd["four"]["hint"]["state"] == "present" and rd["four"]["guard"]["state"] == "INCOMPLETE"
    assert rd["four"]["guard"]["missing"] == [("glass4", A9, 3)] and rd["four"]["harm"]["unread"] == ["glass4"]
    p1 = rd["phase1"]
    assert p1["fallback"] and p1["seeds"] == [0, 1] and p1["guard"]["state"] == "holds"
    assert p1["guard"] is rd["s01"]["guard"] and "INCOMPLETE" in p1["source"]
    w.drop = {("sing44", A6F, 2)}
    rd = _first(w.report())
    assert rd["four"]["hint"]["state"] == "INCOMPLETE" and rd["four"]["hint"]["missing"] == [("sing44", A6F, 2)]
    assert rd["s23"]["hint"]["state"] == "INCOMPLETE" and rd["s01"]["hint"]["state"] == "present"
    w.drop = set(k for k, s in w.exp if s == M.SET_23)               # D-031 never ran: D-030's reading leads
    rd = _first(w.report())
    assert rd["validity"]["n_present"] == 68 and rd["phase1"]["fallback"] and rd["phase1"]["hint"]["state"] == "present"


def test_failed_runs_are_kept_and_the_letter_is_beside(tmp_path):
    """A failed A9 run in D-031's set (P truncated at the crash, 0.9, no EA summary) is an outcome of its arm in the
    reading that leads (keep: VALID, counted in the four-seed means); the letter (exclude) is beside: INCOMPLETE, and
    the phase-1 result falls back there."""
    w = World(tmp_path)
    k = ("sing44", A9, 2)
    w.row[k] = dict(status="failed")
    w.p[k] = 0.9
    w.summ[k] = None
    rep = w.report()
    assert rep["failed_runs"] == [w.exp[(k, M.SET_23)]]
    keep, excl = rep["readings"]
    assert keep["keep_failed"] and not excl["keep_failed"]
    assert keep["validity"]["verdict"] == "VALID" and keep["d031"]["validity"]["deviation"]
    assert keep["four"]["hint"]["mean_P_a9"] == pytest.approx((10 * 0.4 + (0.4 * 3 + 0.9) / 4) / 11)
    assert not keep["phase1"]["fallback"]
    assert excl["validity"]["verdict"] == "INCOMPLETE" and excl["phase1"]["fallback"]
    assert len(w.report("keep")["readings"]) == 1 and len(w.report("exclude")["readings"]) == 1
    lines = M.render(rep).splitlines()          # one THE line: the kept reading's; the letter's is labelled beside
    assert [x for x in lines if x.startswith("PHASE-1 INTERIM RESULT")] == \
        [x for x in lines if x.startswith("PHASE-1 INTERIM RESULT (four seeds 0-3")]
    assert len([x for x in lines if x.startswith("PHASE-1 INTERIM RESULT")]) == 1
    assert len([x for x in lines if x.startswith("the letter (failed runs excluded), beside — not the phase-1")]) == 1
    only = M.render(w.report("exclude")).splitlines()                 # the letter alone: still no THE line
    assert not [x for x in only if x.startswith("PHASE-1 INTERIM RESULT")]


def test_partial_d031_d013_lines_read_only_full_instances_and_are_labelled(tmp_path):
    """D-031's runs missing on 6 of the fixed 11 (both seeds, both arms; 112 of 136): the D-013 blocks read only the
    instances with every seed of both arms at the reading's seeds — never a mix of 2- and 4-seed means — and say so;
    the four-seed block is printed as beside, not the result, and the phase-1 line carries no selection check."""
    w = World(tmp_path)
    for i in ELEVEN:
        w.set_p(i, A9, (0.2, 0.2, 0.6, 0.6))
    gone = ELEVEN[:6]
    for i in gone:
        for lab in (A9, A6F):
            for s in (2, 3):
                w.drop.add((i, lab, s))
    rep = w.report()
    rd = _first(rep)
    assert rd["validity"]["verdict"] == "INCOMPLETE" and rd["validity"]["n_present"] == 112
    four, s01, s23 = rd["four"], rd["s01"], rd["s23"]
    assert four["d013_full"]["n11"] == 5 and four["d013_full"]["n17"] == 11
    assert set(four["d013_full"]["instances"]) == set(DEV) - set(gone)
    v = four["vs_a6f"]
    assert v["n"] == 5 and v["mean_P_after"] == pytest.approx(0.4) and v["mean_P_before"] == pytest.approx(0.4)
    blk = four["vs_a0_17"]["block"]                                    # A0 0.8 everywhere: all 17 are SCIP-weak
    assert blk["n"] == 11 and blk["mean_P_after"] == pytest.approx(0.4)
    assert s01["vs_a6f"]["n"] == 11 and s01["vs_a6f"]["mean_P_after"] == pytest.approx(0.2)
    assert s23["vs_a6f"]["n"] == 5 and s23["vs_a6f"]["mean_P_after"] == pytest.approx(0.6)
    text = M.render(rep)
    lines = text.splitlines()
    tag = "beside, not the result (INCOMPLETE, 112 of 136 runs)"
    assert f"SEEDS 0-3 (the four-seed reading — {tag}):" in lines
    assert any(x.startswith(f"{tag}: reported, not criteria (D-013 terms, seeds 0-3) — vs cuOpt alone") and
               "the fixed 11 [partial: 5 of 11 instances have every seed 0-3 of both arms; only they are read]: "
               "improved 0 of 5 (6 instances without every seed 0-3 of both arms, left out)" in x for x in lines)
    assert "weak set [partial: 11 of 17 instances have every seed 0-3 of both arms" in text
    p1 = [x for x in lines if x.startswith("PHASE-1 INTERIM RESULT")]
    assert len(p1) == 1 and "D-030's reading, seeds 0-1" in p1[0] and "selection check" not in p1[0]


def test_halves_windows_are_named_and_a_shared_one_is_shown(tmp_path):
    w = World(tmp_path)
    for lab in (A9, A6F):
        w.window[("rail01", lab, 2)] = f"W-{M.SET_01}"                 # D-031's block in D-030's last window
    text = M.render(w.report())
    assert (f"the halves are not day-separated (descriptive): seeds 0-1 ran in window(s) ['W-{M.SET_01}'], seeds 2-3 "
            f"in ['W-{M.SET_01}', 'W-{M.SET_23}']; shared ['W-{M.SET_01}']") in text


def test_harm_lines_carry_the_replicates_beside(tmp_path):
    """A four-seed harm on glass4 prints its threshold and, beside (never changing the list), the same-configuration
    A9 replicates' d of 10-15 on the instance against the same fixed A9@r13x reference and cuOpt alone's d."""
    w = World(tmp_path)
    w.set_p("glass4", A9, (0.5,) * 4)                                  # vs A9@r13x 0.4: d -0.2 < -0.05
    df = w.build()
    rep_rows = [dict(run_id=f"glass4__A9@r2ctl__s{s}__120s__{R.OLD_A9_HASH}", set="r2-ld", instance="glass4",
                     arm="A9@r2ctl", seed=s, budget=120.0, status="completed", attempt=0, pt=0.45, synthetic=False,
                     block_void=False, window_id="W-r2", run_dir="") for s in (0, 1)]
    df = pd.concat([df, pd.DataFrame(rep_rows)], ignore_index=True)
    rd = _first(M.report(df, tmp_path))
    h = rd["four"]["harm"]
    assert h["harm"] == ["glass4"]
    notes = h["notes"]["glass4"]
    assert any(x.startswith("same-configuration replicates here: A9@r2ctl d -0.111") for x in notes), notes
    assert any(x.startswith("cuOpt alone 10-13 -> this window d +0.000") for x in notes), notes
    line = [x for x in M.render(M.report(df, tmp_path)).splitlines() if x.startswith("HARM (seeds 0-3")][0]
    assert "glass4 (d_i -0.200 < -0.050; P̄ 0.400 -> 0.500, fires > 0.4211" in line and "[beside: same-config" in line


def test_seeds_0_1_compared_with_the_saved_d030_reading(tmp_path, capsys):
    """`--d030-json F`: seeds 0-1 here against the JSON c1r_report saved when D-030 was read — equal on the same
    table; a moved threshold (ca's best_of_study reference is re-scored by every build-runs) or a moved hint ratio is
    printed with both numbers; another set's JSON is not compared."""
    w = World(tmp_path)
    for i in ELEVEN:
        w.set_p(i, A9, (0.3,) * 4)
    runs = tmp_path / "runs.parquet"
    w.build().to_parquet(runs)
    saved = tmp_path / "d030.json"
    assert R.main(["--a9", M.A9C, "--runs", str(runs), "--root", str(tmp_path), "--json", str(saved)]) == 0
    capsys.readouterr()
    assert M.main(["--runs", str(runs), "--root", str(tmp_path), "--d030-json", str(saved)]) == 0
    assert f"seeds 0-1 here vs the D-030 reading saved when it was read: equal to the reading saved in {saved}" \
        in capsys.readouterr().out
    js = json.loads(saved.read_text())
    rd = js["readings"][0]
    ca = "ca__Scenario400_reserves_1"
    row = next(r for r in rd["rules"]["harm"]["rows"] if r["instance"] == ca)
    row["threshold"] = 0.4142
    rd["rules"]["hint"]["ratio"] = 0.76
    moved = tmp_path / "moved.json"
    moved.write_text(json.dumps(js))
    vs = _first(M.report(w.build(), tmp_path, "both", moved))["vs_saved_d030"]
    assert vs["ok"] is False and len(vs["diffs"]) == 2
    assert vs["diffs"] == ["hint present 0.76 (saved) vs present 0.75 (now)",
                           f"{ca} threshold 0.4142 (saved) vs 0.421053 (now)"]
    assert "best_of_study" in vs["text"]
    js["set"] = "c1r-ld0"
    moved.write_text(json.dumps(js))
    vs = _first(M.report(w.build(), tmp_path, "both", moved))["vs_saved_d030"]
    assert vs["ok"] is False and "not D-030's reading" in vs["text"]


# ============================================================================================ output and CLI
def test_main_render_and_json(tmp_path, capsys):
    w = World(tmp_path)
    for i in ELEVEN:
        w.set_p(i, A9, (0.3,) * 4)
    runs = tmp_path / "runs.parquet"
    w.build().to_parquet(runs)
    out = tmp_path / "d031.json"
    assert M.main(["--runs", str(runs), "--root", str(tmp_path), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    for s in ("VALIDITY VALID (both sets): 136 of 136", "HINT present (seeds 0-3)", "GUARD holds (seeds 0-3)",
              "HINT present (seeds 0-1 alone)", "HINT present (seeds 2-3 alone)",
              "seeds 0-1 here vs c1r_report's D-030 reading on this run table: equal (hint, guard, harm list)",
              "seeds 0-1 here vs the D-030 reading saved when it was read: not compared (no --d030-json given)",
              "HARM (seeds 0-3; cross-day vs A9@r13x's seeds 0-1 of 10-13): none on the 17",
              "SEEDS 0-3 (the four-seed reading — the phase-1 interim result's source):",
              "reported, not criteria (D-013 terms, seeds 0-3) — vs cuOpt alone",
              "the halves are not day-separated (descriptive): seeds 0-1 ran in window(s) ['W-c1r-ysync-ld0'], seeds "
              "2-3 in ['W-c1r-ysync-ld0-s23'] — a difference",
              "SCIP's weak set among all 17", "PHASE-1 INTERIM RESULT (four seeds 0-3",
              "present iff <= 0.95; futility-only, D-024 rule 2: a hint never declares success)",
              "selection check (D-024 rule 2, the stage-2-only figure): seeds 2-3 alone HINT present (ratio 0.750)",
              "go / no-go on C1 is Berk's"):
        assert s in text, s
    assert "partial:" not in text and "beside, not the result" not in text
    js = json.loads(out.read_text())
    rd = js["readings"][0]
    assert rd["phase1"]["hint"]["state"] == "present" and len(rd["d030"]["runs"]) == len(rd["d031"]["runs"]) == 68
    assert M.main(["--runs", str(tmp_path / "none.parquet")]) == 2
