"""`tools/d029_report.py` (D-029 as amended 2026-10-15 12:2x: the last R1 test for v1 — ysync at seeds 2–5, its guards
pooled over seeds 0–5 with D-025's failing seed kept) on synthetic run tables and run directories — no solver, no GPU.
The job writer (20 read runs in 10 blocks at priority 1, new run ids, alternated positions, pinned run hashes; the
warm-up block first at priority 0; byte-for-byte regeneration of jobs/d029.jsonl; rerun sets); the 28 expected runs (8
of r1-ysync, 20 of r1f-ysync) and the 4 read beside; validity (INVALID dominates INCOMPLETE; D-025's legs, the window
leg and the flag state); failed runs (a treatment's an outcome, a control's never decides a rule) and the letter; rule 1
(Ḡ(T) / Ḡ(C) >= 0.90 over seeds 0–5), rule 2 (P̄(T) − P̄(C) <= 0.05, the bootstrap bound beside and not gating),
rule 3 (ca ΣP(T) / ΣP(C) <= 0.95 at seeds 2–3 alone, seeds 0–3 pooled beside); every verdict branch; the pooling of the
observed seeds 0–1; rerun sets read in place of their blocks; the pool's first run; JSON and --write-jobs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from itrea.experiments import make_jobs as mj
from itrea.experiments import pair
from itrea.experiments.pool import make_blocks, order_blocks

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("d029_report", ROOT / "tools" / "d029_report.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

EXT = pair.resolve_harness("ext")
ITEM, COMP, CA = R.ITEM, R.COMP, R.CA
FLAGS = {"yield": 2, "auto": 0}


def summary(role: str, *, gens=100, P=48, dtype="int8", **over) -> dict:
    """An EA summary as the engine writes it: T = yield set and held, C = CUDA's default read back."""
    mode = "yield" if role == "T" else "auto"
    cs = dict(mode=mode, requested=FLAGS[mode], state="read" if mode == "auto" else "verified", active=True,
              sched_active=mode, flags_active=FLAGS[mode], sched_at_end=mode, active_at_end=True)
    s = dict(P=P, gene_dtype=dtype, compiled_on_clock=False, cuda_sched=cs, phase_events_block=False,
             generations=gens, wall_s=120.0, ls_launches=10 * gens,
             cpu_clock=dict(self_per_wall_s=1.0, main_thread_per_wall_s=0.9),
             lp=dict(state="done", t=20.0, worker_done_t=18.0, worker_wall_s=17.0, worker_cpu_share=0.9))
    s.update(over)
    return s


class World:
    """The 28 runs D-029's rules read and the 4 ca runs read beside: P and G per (instance, role, seed), every file
    written under `root`. Default: no effect on item / comp21 (T = C), a ca gain (T 0.25 vs C 0.39), so the default
    world ADOPTs."""

    def __init__(self, root: Path):
        self.root = root
        self.exp = R.expected()
        self.bes = R.beside_keys()
        self.p, self.g = {}, {}
        for (set_, inst, lab, s) in list(self.exp) + list(self.bes):
            k = (inst, R.role(lab), s)
            self.p[k] = {ITEM: 0.36, COMP: 0.55, CA: 0.39 if k[1] == "C" else 0.25}[inst]
            self.g[k] = {ITEM: 1500, COMP: 280, CA: 15}[inst]
        self.summ: dict = {}
        self.row: dict = {}
        self.extra: list = []
        self.drop: set = set()
        self.created: dict = {}

    def key(self, inst, role, s):
        """The key (set, instance, label, seed) of (instance, role, seed), expected or beside."""
        hits = [k for k in list(self.exp) + list(self.bes) if k[1] == inst and R.role(k[2]) == role and k[3] == s]
        assert len(hits) == 1
        return hits[0]

    def _write(self, set_, rid, inst, lab, s, created, row_over=None, summ_over=None, p=None, g=None) -> dict:
        c = (inst, R.role(lab), s)
        rw = dict(run_id=rid, set=set_, instance=inst, arm=lab, seed=s, budget=120.0, status="completed",
                  attempt=0, pt=self.p[c] if p is None else p, harness=json.dumps(EXT, sort_keys=True),
                  cuda_compiled_on_clock=False, synthetic=False, block_void=False, window_id=f"w-{set_}")
        rw.update(row_over or {})
        d = self.root / "traces" / "raw" / set_ / rw["run_id"].split("#")[0]
        d.mkdir(parents=True, exist_ok=True)
        rj = dict(run_id=rid, config=dict(harness=rw.pop("_rj_harness", EXT)),
                  compiled_on_clock_cuda=rw.pop("_rj_compiled", False), created_epoch=created,
                  ended_epoch=created + 130.0, gates_before=dict(loadavg_1m=1.25))
        (d / "run.json").write_text(json.dumps(rj))
        (d / "ea.summary.json").write_text(json.dumps(summary(R.role(lab), **dict(
            dict(gens=self.g[c] if g is None else g), **(summ_over or {})))))
        rw["run_dir"] = str(d.relative_to(self.root))
        return rw

    def build(self) -> pd.DataFrame:
        rows = []
        for i, (k, rid) in enumerate(sorted(list(self.exp.items()) + list(self.bes.items()))):
            if k in self.drop:
                continue
            set_, inst, lab, s = k
            sm = self.summ.get(k, {})
            rw = self._write(set_, rid, inst, lab, s, self.created.get(k, 1000.0 + i), self.row.get(k), sm or {})
            if sm is None:                                  # no EA summary (a run that died without one)
                (self.root / rw["run_dir"] / "ea.summary.json").unlink()
            rows.append(rw)
        return pd.DataFrame(rows + self.extra)

    def rerun(self, set_: str, blocks, created: float = 5000.0, row_over=None, p=None):
        """Both arms of each (instance, seed) block again in rerun set `set_`; ids qualified `#<set_>` (build-runs:
        the earliest run keeps the bare id)."""
        for j, (inst, s) in enumerate(blocks):
            for lab in (R.C_NEW, R.T_LABEL):
                rid = R.run_id(inst, lab, s)
                self.extra.append(self._write(set_, f"{rid}#{set_}", inst, lab, s, created + j, row_over,
                                              {}, p=(p or {}).get((inst, R.role(lab), s))))

    def report(self, failed="both"):
        return R.report(self.build(), self.root, failed)


def _one(rep):
    assert len(rep["readings"]) == 1
    return rep["readings"][0]


# ----------------------------------------------------------------------------------------------- jobs
def test_the_job_writer_makes_20_new_runs_in_10_blocks():
    jobs = R.d029_jobs()
    assert len(jobs) == 20 and len({j["block"] for j in jobs}) == 10
    assert {j["set"] for j in jobs} == {"r1f-ysync"} and {j["harness"] for j in jobs} == {"ext"}
    assert {j["budget"] for j in jobs} == {120.0} and {j["split"] for j in jobs} == {"dev"}
    assert {j["priority"] for j in jobs} == {1}
    assert sorted({(j["name"], j["seed"]) for j in jobs}) == sorted(
        [(ITEM, s) for s in (2, 3, 4, 5)] + [(COMP, s) for s in (2, 3, 4, 5)] + [(CA, s) for s in (2, 3)])
    assert {pair.arm_label(j): j["config"] for j in jobs} == {
        "A9@ysync": "configs/itrea2-dev-rc-3way-ysync.yaml", "A9@r1fctl": "configs/itrea2-dev-rc-3way.yaml"}
    for j in jobs:
        assert (ROOT / j["config"]).exists()
        ctl_first = j["seed"] % 2 == 0                     # make_jobs._alternated: the control first at even seeds
        assert j["position"] == (0 if (j["variant"] == "r1fctl") == ctl_first else 1)
    ids = [pair.run_id_for(j) for j in jobs]
    assert len(set(ids)) == 20
    old = {pair.run_id_for(j) for j in mj.r1_bsync("r13x", "ysync")}
    assert not set(ids) & old
    # the treatment's run hash is r1-ysync's treatment's, the control's is the development default's (A9@r13x's)
    assert {i.rsplit("__", 1)[1] for i in ids if "@ysync" in i} == {i.rsplit("__", 1)[1] for i in old if "@ysync" in i}
    assert {i.rsplit("__", 1)[1] for i in ids if "@r1fctl" in i} == {"28e5a5e9c725"}


def test_the_job_file_runs_the_warmup_first_and_keeps_the_order():
    """The amendment: one warm-up (the control's configuration, item_placement seed 6, A9@r1fwarm, its own set) at
    priority 0, the 20 read runs at priority 1; the pool runs the warm-up first and the 10 blocks in the order they
    had without it; the warm-up's id is new and never read."""
    jobs = R.job_file()
    warm = [j for j in jobs if j["set"] == "r1f-ysync-warmup"]
    assert len(jobs) == 21 and len(warm) == 1
    w = warm[0]
    assert (w["name"], w["seed"], pair.arm_label(w), w["config"], w["priority"]) == (
        ITEM, 6, "A9@r1fwarm", "configs/itrea2-dev-rc-3way.yaml", 0)
    assert {j["priority"] for j in jobs if j is not w} == {1}
    order = [b.id for b in order_blocks(make_blocks(jobs))]
    plain = [b.id for b in order_blocks(make_blocks([{k: v for k, v in j.items() if k != "priority"}
                                                     for j in R.d029_jobs()]))]
    assert order[0] == w["block"] and order[1:] == plain
    wid = pair.run_id_for(w)
    assert wid not in set(R.expected().values()) | set(R.beside_keys().values())
    assert wid not in {pair.run_id_for(j) for j in mj.r1_bsync("r13x", "ysync")}
    assert not R.RERUN_RE.match(w["set"]) and not R.RERUN_RE.match("r1f-ysync-2-warmup")


def test_jobs_file_regenerates_byte_for_byte(tmp_path):
    out = tmp_path / "d029.jsonl"
    assert R.main(["--write-jobs", str(out)]) == 0
    assert [json.loads(x) for x in out.read_text().splitlines()] == R.job_file()
    committed = ROOT / R.JOBS_OUT
    if committed.exists():
        assert committed.read_bytes() == out.read_bytes()


def test_run_ids_are_pinned(tmp_path, monkeypatch):
    """The expected ids are pinned literals and equal today's `run_id_for` of the jobs (new) and of D-025's writer
    (old); `--write-jobs` refuses a job whose id is not the pinned one."""
    exp = R.expected()
    new = {pair.run_id_for(j) for j in R.d029_jobs()}
    assert {v for k, v in exp.items() if k[0] == "r1f-ysync"} == new
    old = {v for k, v in exp.items() if k[0] == "r1-ysync"} | set(R.beside_keys().values())
    committed = ROOT / "jobs" / "r1-ysync.jsonl"           # the file r1-ysync ran from (D-025's writer, 08:1x)
    if committed.exists():
        assert old <= {pair.run_id_for(json.loads(x)) for x in committed.read_text().splitlines()}
    assert R.pinned_problems(R.job_file()) == []
    assert R.CONTROL_CONFIG == "configs/itrea2-dev-rc-3way.yaml"
    monkeypatch.setitem(R.RUN_HASH, "A9@ysync", "ffffffffffff")
    out = tmp_path / "x.jsonl"
    assert R.main(["--write-jobs", str(out)]) == 2 and not out.exists()


def test_rerun_job_files(tmp_path):
    out = tmp_path / "d029-2.jsonl"
    assert R.main(["--write-jobs", str(out), "--set", "r1f-ysync-2", "--blocks", f"{ITEM}:s3,{CA}:s2"]) == 0
    jobs = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(jobs) == 5 and {j["set"] for j in jobs} == {"r1f-ysync-2", "r1f-ysync-2-warmup"}
    read = [j for j in jobs if j["set"] == "r1f-ysync-2"]
    assert sorted({(j["name"], j["seed"]) for j in read}) == [(CA, 2), (ITEM, 3)]
    assert {pair.run_id_for(j) for j in read} == {R.run_id(i, lab, s) for i, s in ((ITEM, 3), (CA, 2))
                                                  for lab in (R.T_LABEL, R.C_NEW)}
    for bad in (["--set", "r1f-ysync-1", "--blocks", f"{ITEM}:s3"], ["--set", "r1f-ysync-2"],
                ["--set", "r1f-ysync-2", "--blocks", f"{ITEM}:s1"], ["--blocks", f"{ITEM}:s3"],
                ["--set", "r1-ysync", "--blocks", f"{ITEM}:s3"]):
        assert R.main(["--write-jobs", str(tmp_path / "bad.jsonl")] + bad) == 2
    assert not (tmp_path / "bad.jsonl").exists()


def test_expected_is_28_runs_8_of_r1_ysync_and_20_new_and_4_beside():
    exp = R.expected()
    assert len(exp) == 28
    old = {k: v for k, v in exp.items() if k[0] == "r1-ysync"}
    assert len(old) == 8 and {k[2] for k in old} == {"A9@ysync", "A9@r1ctl-ysync"} and {k[3] for k in old} == {0, 1}
    assert {k[1] for k in old} == {ITEM, COMP}
    new = {k for k in exp if k[0] == "r1f-ysync"}
    assert len(new) == 20 and {k[2] for k in new} == {"A9@ysync", "A9@r1fctl"}
    bes = R.beside_keys()
    assert len(bes) == 4 and {(k[0], k[1]) for k in bes} == {("r1-ysync", CA)} and {k[3] for k in bes} == {0, 1}


# ----------------------------------------------------------------------------------------------- verdicts
def test_no_harm_and_a_ca_gain_adopt(tmp_path):
    w = World(tmp_path)
    rd = _one(w.report())
    assert rd["validity"]["verdict"] == "VALID", rd["validity"]
    ru = rd["rules"]
    assert ru["verdict"] == "ADOPT" and ru["fails"] == [] and "re-run on the frozen hash" in ru["next"]
    assert ru["gen"][ITEM]["ratio"] == pytest.approx(1.0) and ru["p"][COMP]["diff"] == pytest.approx(0.0)
    assert ru["ca"]["ratio"] == pytest.approx(0.25 / 0.39) and ru["ca"]["pooled03"] == pytest.approx(0.25 / 0.39)
    txt = R.render(w.report())
    assert "VERDICT: ADOPT" in txt and "Rule 3" in txt and "seeds 2–3 alone" in txt


def test_no_ca_gain_at_seeds_2_3_is_not_adopted_and_final(tmp_path):
    w = World(tmp_path)
    for s in (2, 3):
        w.p[(CA, "T", s)] = 0.39
    ru = _one(w.report())["rules"]
    assert ru["verdict"] == "NOT ADOPTED" and ru["fails"] == ["ca gain replication"]
    assert "final for v1" in ru["next"] and "v1.1" in ru["next"]
    assert ru["ca"]["ratio"] == pytest.approx(1.0) and ru["ca"]["pooled03"] == pytest.approx((0.5 + 0.78) / 1.56)


@pytest.mark.parametrize("t_mean, ok", [(1350.0, True), (1349.0, False)])
def test_generation_guard_ratio_of_means_at_090(tmp_path, t_mean, ok):
    """Ḡ(T) / Ḡ(C) >= 0.90 over seeds 0–5: 1350 / 1500 = 0.90 holds, 1349 / 1500 fails."""
    w = World(tmp_path)
    for s in range(6):
        w.g[(ITEM, "T", s)] = t_mean
    ru = _one(w.report())["rules"]
    g = ru["gen"][ITEM]
    assert g["ok"] is ok and g["ratio"] == pytest.approx(t_mean / 1500) and g["threshold"] == 0.90
    assert (ru["verdict"] == "ADOPT") is ok
    if not ok:
        assert ru["fails"] == ["generation guard (item_placement_9903)"]


def test_generation_guard_pools_the_observed_failing_seed(tmp_path):
    """D-025's ysync seeds 0–1 on item (T 1483 / 678, C 1367 / 1480) stay in: with T = C = 1500 at seeds 2–5 the pooled
    ratio is (2161 + 6000) / (2847 + 6000) = 0.922 (holds); with T at 0.95 x C at seeds 2–5 it is 0.888 (fails) —
    although seeds 2–5 alone (0.95) would pass."""
    w = World(tmp_path)
    w.g[(ITEM, "T", 0)], w.g[(ITEM, "T", 1)], w.g[(ITEM, "C", 0)], w.g[(ITEM, "C", 1)] = 1483, 678, 1367, 1480
    g = _one(w.report())["rules"]["gen"][ITEM]
    assert g["ok"] and g["ratio"] == pytest.approx(8161 / 8847)
    assert g["seeds01"] == pytest.approx(2161 / 2847) and g["seeds25"] == pytest.approx(1.0)
    assert g["per"][1]["ratio"] == pytest.approx(678 / 1480) and g["median_ratio"] == pytest.approx(1.0)
    w2 = World(tmp_path / "w2")
    w2.g[(ITEM, "T", 0)], w2.g[(ITEM, "T", 1)], w2.g[(ITEM, "C", 0)], w2.g[(ITEM, "C", 1)] = 1483, 678, 1367, 1480
    for s in (2, 3, 4, 5):
        w2.g[(ITEM, "T", s)] = 1425
    g2 = _one(w2.report())["rules"]["gen"][ITEM]
    assert g2["ok"] is False and g2["ratio"] == pytest.approx((2161 + 5700) / 8847)
    assert g2["seeds25"] == pytest.approx(0.95)


@pytest.mark.parametrize("delta, ok", [(0.05, True), (0.051, False), (-0.3, True)])
def test_p_noharm_is_one_sided_at_005(tmp_path, delta, ok):
    w = World(tmp_path)
    for s in range(6):
        w.p[(COMP, "T", s)] = 0.55 + delta
    ru = _one(w.report())["rules"]
    x = ru["p"][COMP]
    assert x["ok"] is ok and x["diff"] == pytest.approx(delta) and x["threshold"] == 0.05
    assert (ru["verdict"] == "ADOPT") is ok


def test_bootstrap_bound_is_reported_not_gating(tmp_path):
    """A noisy item P with a mean difference inside 0.05 passes even when the bootstrap's upper bound is above it."""
    w = World(tmp_path)
    for s, d in zip(range(6), (0.25, -0.20, 0.20, -0.15, 0.10, -0.08)):    # mean diff +0.02
        w.p[(ITEM, "T", s)] = 0.36 + d
    x = _one(w.report())["rules"]["p"][ITEM]
    assert x["diff"] == pytest.approx(0.02) and x["ok"]
    assert x["boot_upper"] > 0.05 and x["boot_upper"] == pytest.approx(R.boot_upper(
        [0.25, -0.20, 0.20, -0.15, 0.10, -0.08]))
    assert R.boot_upper([0.0] * 6) == pytest.approx(0.0)
    assert x["seeds01"] == pytest.approx(0.025) and x["seeds25"] == pytest.approx(0.0175)


@pytest.mark.parametrize("t, ok", [(0.95 * 0.39, True), (0.951 * 0.39, False)])
def test_ca_gain_ratio_at_095_on_seeds_2_3(tmp_path, t, ok):
    w = World(tmp_path)
    for s in (2, 3):
        w.p[(CA, "T", s)] = t
    ca = _one(w.report())["rules"]["ca"]
    assert ca["ok"] is ok and ca["ratio"] == pytest.approx(t / 0.39) and ca["threshold"] == 0.95


def test_ca_reads_seeds_2_3_alone_and_pools_0_to_3_beside(tmp_path):
    """The amendment: a gain at seeds 0–1 and none at 2–3 FAILS (seeds 2–3 alone 1.0; pooled 0.82 beside); a ca harm
    at seeds 2–3 (T 0.45) fails; seeds 0–1 do not gate (no gain there and a gain at 2–3 passes)."""
    w = World(tmp_path)
    for s in (2, 3):
        w.p[(CA, "T", s)] = 0.39
    ca = _one(w.report())["rules"]["ca"]
    assert ca["ok"] is False and ca["ratio"] == pytest.approx(1.0)
    assert ca["pooled03"] == pytest.approx((0.5 + 0.78) / 1.56) and set(ca["per"]) == {0, 1, 2, 3}
    w2 = World(tmp_path / "w2")
    for s in (2, 3):
        w2.p[(CA, "T", s)] = 0.45
    assert _one(w2.report())["rules"]["ca"]["ok"] is False
    w3 = World(tmp_path / "w3")
    for s in (0, 1):
        w3.p[(CA, "T", s)] = 0.50
    ru = _one(w3.report())["rules"]
    assert ru["ca"]["ok"] and ru["verdict"] == "ADOPT" and ru["ca"]["pooled03"] > 0.95


def test_several_failing_rules_are_all_named(tmp_path):
    w = World(tmp_path)
    for s in range(6):
        w.g[(COMP, "T", s)] = 200
        w.p[(ITEM, "T", s)] = 0.50
    ru = _one(w.report())["rules"]
    assert ru["fails"] == ["generation guard (comp21-2idx)", "P no-harm (item_placement_9903)"]


# ----------------------------------------------------------------------------------------------- validity
def test_a_missing_run_is_incomplete_and_rules_are_not_read(tmp_path):
    w = World(tmp_path)
    w.drop.add(w.key(ITEM, "T", 4))
    rd = _one(w.report())
    assert rd["validity"]["verdict"] == "INCOMPLETE" and rd["validity"]["failing"] == ["complete"]
    assert rd["rules"] is None and rd["verdict"] == "INCOMPLETE" and rd["n_present"] == 27
    assert "RULES NOT READ: INCOMPLETE" in R.render(w.report())


def test_an_empty_table_is_incomplete_with_28_missing(tmp_path):
    cols = ["run_id", "set", "instance", "arm", "seed", "budget", "status", "attempt", "pt", "run_dir", "window_id"]
    rd = _one(R.report(pd.DataFrame(columns=cols), tmp_path))
    assert rd["n_present"] == 0 and rd["validity"]["verdict"] == "INCOMPLETE"
    assert len(rd["validity"]["legs"]["complete"]) == 28


def test_invalid_dominates_incomplete(tmp_path):
    w = World(tmp_path)
    w.drop.add(w.key(ITEM, "T", 4))
    w.row[w.key(COMP, "C", 2)] = dict(_rj_harness={"cuda_cache": True})
    rd = _one(w.report())
    assert rd["validity"]["verdict"] == "INVALID" and set(rd["validity"]["failing"]) == {"complete", "harness"}


@pytest.mark.parametrize("case, leg", [
    ("treatment_not_yield", "flag"), ("control_flag", "flag"), ("control_renice", "flag"), ("events", "flag"),
    ("harness", "harness"), ("compile_rj", "compile"), ("compile_summ", "compile"), ("config", "config"),
    ("block_P", "block"), ("no_summary", "flag"), ("window", "window"), ("no_window", "window")])
def test_validity_legs(tmp_path, case, leg):
    w = World(tmp_path)
    k_t, k_c = w.key(ITEM, "T", 3), w.key(ITEM, "C", 3)
    if case == "treatment_not_yield":
        w.summ[k_t] = dict(cuda_sched=dict(mode="yield", state="verified", sched_active="auto", sched_at_end="auto"))
    elif case == "control_flag":
        w.summ[k_c] = dict(cuda_sched=dict(mode="auto", state="read_error", read_error="x"))
    elif case == "control_renice":
        w.summ[k_c] = dict(host_renice=dict(gap=10))
    elif case == "events":
        w.summ[k_t] = dict(phase_events_block=True)
    elif case == "harness":
        w.row[k_t] = dict(_rj_harness={"cuda_cache": True})
    elif case == "compile_rj":
        w.row[k_t] = dict(_rj_compiled=True)
    elif case == "compile_summ":
        w.summ[k_t] = dict(compiled_on_clock=True)
    elif case == "config":
        w.row[k_t] = dict(run_id=w.exp[k_t].rsplit("__", 1)[0] + "__ffffffffffff")
    elif case == "block_P":
        w.summ[k_t] = dict(P=64)
    elif case == "no_summary":
        w.summ[k_t] = None
    elif case == "window":                              # the two runs of a block in different pools
        w.row[k_t] = dict(window_id="w-another-pool")
    elif case == "no_window":
        w.row[k_c] = dict(window_id=None)
    rd = _one(w.report())
    assert rd["validity"]["verdict"] == "INVALID" and leg in rd["validity"]["failing"], rd["validity"]
    assert rd["rules"] is None and rd["verdict"] == "INVALID"


def test_the_old_sets_runs_are_checked_too_and_the_beside_runs_are_not(tmp_path):
    w = World(tmp_path)
    w.summ[w.key(COMP, "C", 1)] = dict(compiled_on_clock=True)      # an r1-ysync run the rules read
    rd = _one(w.report())
    assert rd["validity"]["verdict"] == "INVALID" and any("r1-ysync" in x for x in rd["validity"]["legs"]["compile"])
    w2 = World(tmp_path / "w2")
    w2.summ[w2.key(CA, "C", 1)] = dict(compiled_on_clock=True)     # an r1-ysync ca run read beside rule 3 only
    w2.drop.add(w2.key(CA, "T", 0))
    rd2 = _one(w2.report())
    assert rd2["validity"]["verdict"] == "VALID" and rd2["verdict"] == "ADOPT"
    assert rd2["rules"]["ca"]["pooled03"] is None and len(rd2["beside"]) == 3


# ----------------------------------------------------------------------------------------------- failed runs
def test_a_failed_treatment_run_is_kept_for_the_verdict_and_excluded_beside(tmp_path):
    w = World(tmp_path)
    k = w.key(COMP, "T", 5)
    w.row[k] = dict(status="failed")                    # the run crashed late: its summary exists
    rep = w.report()
    keep, excl = rep["readings"]
    assert keep["keep_failed"] and not excl["keep_failed"] and rep["failed_runs"] == [w.exp[k]]
    assert keep["validity"]["verdict"] == "VALID" and keep["validity"]["deviation"] and keep["verdict"] == "ADOPT"
    assert excl["validity"]["verdict"] == "INCOMPLETE" and excl["validity"]["failing"] == ["complete"]
    assert any("failed (not completed)" in x for x in excl["validity"]["legs"]["complete"])
    assert _one(w.report(failed="keep"))["keep_failed"] and not _one(w.report(failed="exclude"))["keep_failed"]


def test_a_failed_treatment_run_without_a_summary_fails_rule_1(tmp_path):
    w = World(tmp_path)
    k = w.key(ITEM, "T", 2)
    w.row[k] = dict(status="failed")
    w.summ[k] = None
    rd = _one(w.report(failed="keep"))
    assert rd["validity"]["verdict"] == "VALID" and any("flag unread" in x for x in rd["validity"]["notes"])
    ru = rd["rules"]
    assert ru["gen"][ITEM]["ok"] is False and "treatment generation count is missing" in ru["gen"][ITEM]["why"]
    assert ru["verdict"] == "NOT ADOPTED" and ru["p"][ITEM]["ok"]    # its P (the table's, truncated) still reads


def test_a_failed_control_run_without_a_summary_leaves_the_rules_unread(tmp_path):
    """The amendment: a control failure never decides a rule — no G -> rule 1 unread; its (truncated) P passes rule 2
    only by looking worse -> unread; verdict INCOMPLETE, the block named for a rerun set."""
    w = World(tmp_path)
    k = w.key(ITEM, "C", 2)
    w.row[k] = dict(status="failed")
    w.summ[k] = None
    rd = _one(w.report(failed="keep"))
    assert rd["validity"]["verdict"] == "VALID"
    ru = rd["rules"]
    assert ru["gen"][ITEM]["ok"] is None and ru["p"][ITEM]["ok"] is None and ru["fails"] == []
    assert ru["unread"] == ["generation guard (item_placement_9903)", "P no-harm (item_placement_9903)"]
    assert ru["verdict"] == "INCOMPLETE" and rd["verdict"] == "INCOMPLETE" and ru["rerun"] == [(ITEM, 2)]
    assert "r1f-ysync-<N>" in ru["next"] and f"{ITEM}:s2" in ru["next"]
    assert "VERDICT: INCOMPLETE" in R.render(w.report(failed="keep"))


def test_a_failed_control_run_with_a_summary_never_passes_a_rule(tmp_path):
    w = World(tmp_path)
    w.row[w.key(CA, "C", 3)] = dict(status="failed")
    ru = _one(w.report(failed="keep"))["rules"]
    assert ru["ca"]["ok"] is None and ru["verdict"] == "INCOMPLETE" and ru["rerun"] == [(CA, 3)]


def test_a_rule_failing_with_a_failed_control_still_fails(tmp_path):
    """A failed control only looks worse (fewer generations, a higher P), so a rule that fails with it in would fail
    without the failure too: NOT ADOPTED stands."""
    w = World(tmp_path)
    w.row[w.key(COMP, "C", 3)] = dict(status="failed")
    for s in range(6):
        w.g[(COMP, "T", s)] = 200
    ru = _one(w.report(failed="keep"))["rules"]
    assert ru["gen"][COMP]["ok"] is False and ru["p"][COMP]["ok"] is None
    assert ru["verdict"] == "NOT ADOPTED" and ru["fails"] == ["generation guard (comp21-2idx)"]


# ----------------------------------------------------------------------------------------------- rerun sets
def test_a_rerun_set_is_read_in_place_of_its_blocks(tmp_path):
    """An unread block (its control failed) rerun as r1f-ysync-2, both arms, ids qualified `#r1f-ysync-2`: the rerun's
    rows are read, the original block's rows are rows no rule reads; a later rerun set (-3) wins."""
    w = World(tmp_path)
    k = w.key(ITEM, "C", 2)
    w.row[k] = dict(status="failed")
    w.summ[k] = None
    w.rerun("r1f-ysync-2", [(ITEM, 2)])
    rep = w.report(failed="keep")
    rd = _one(rep)
    assert rd["validity"]["verdict"] == "VALID" and rd["verdict"] == "ADOPT" and rd["reruns"] == ["r1f-ysync-2"]
    assert ("r1f-ysync-2", ITEM, R.C_NEW, 2) in rd["runs"] and ("r1f-ysync", ITEM, R.C_NEW, 2) not in rd["runs"]
    assert set(rep["other_rows"]) >= {w.exp[k], w.exp[w.key(ITEM, "T", 2)]}
    assert rep["failed_replaced"] == [w.exp[k]] and "replaced (reported, not read)" in R.render(rep)
    assert "r1f-ysync-2" in rep["sets"] and "blocks read from rerun set(s) r1f-ysync-2" in R.render(rep)
    w2 = World(tmp_path / "w2")
    w2.rerun("r1f-ysync-2", [(COMP, 4)], p={(COMP, "T", 4): 0.95})
    w2.rerun("r1f-ysync-3", [(COMP, 4)], created=6000.0)
    rd2 = _one(w2.report())
    assert rd2["reruns"] == ["r1f-ysync-3"] and rd2["verdict"] == "ADOPT"
    assert rd2["cells"][(COMP, "T", 4)]["row"]["set"] == "r1f-ysync-3"


def test_an_invalid_block_rerun_makes_the_reading_valid(tmp_path):
    w = World(tmp_path)
    w.row[w.key(COMP, "T", 5)] = dict(_rj_harness={"cuda_cache": True})
    assert _one(w.report())["validity"]["verdict"] == "INVALID"
    w.rerun("r1f-ysync-2", [(COMP, 5)])
    rd = _one(w.report())
    assert rd["validity"]["verdict"] == "VALID" and rd["verdict"] == "ADOPT"


def test_rerun_sets_ignore_warmups_and_other_names():
    df = pd.DataFrame(dict(set=["r1f-ysync", "r1f-ysync-warmup", "r1f-ysync-2-warmup", "r1f-ysync-1", "r1f-ysync-10",
                                "r1f-ysync-2", "r1-ysync"]))
    assert R.rerun_sets(df) == ["r1f-ysync-2", "r1f-ysync-10"]


# ----------------------------------------------------------------------------------------------- reported beside
def test_the_pools_first_run_is_flagged_not_acted_on(tmp_path):
    w = World(tmp_path)
    k = w.key(ITEM, "T", 3)
    w.created[k] = 1.0                                  # the earliest read run of the r1f-ysync pool
    rd = _one(w.report())
    allruns = list(rd["runs"].values()) + list(rd["beside"].values())
    assert rd["runs"][k]["pool_first"] and sum(v["pool_first"] for v in allruns) == 2               # one per pool
    ps = rd["pool_starts"]["w-r1f-ysync"]
    assert ps["run_id"] == w.exp[k] and ps["read"] and ps["prev_window"] == "w-r1-ysync" and rd["verdict"] == "ADOPT"
    assert "[pool first]" in R.render(w.report())


def test_the_warmup_is_the_pools_first_run_and_is_not_read(tmp_path):
    w = World(tmp_path)
    wj = R.warmup_job()
    w.extra.append(dict(w._write("r1f-ysync-warmup", pair.run_id_for(wj), ITEM, R.C_NEW, 6, 0.5,
                                 dict(window_id="w-r1f-ysync"), {}, p=0.36, g=1500), arm=R.WARMUP_LABEL))
    rep = w.report()
    rd = _one(rep)
    ps = rd["pool_starts"]["w-r1f-ysync"]
    assert ps["run_id"] == pair.run_id_for(wj) and ps["read"] is False and ps["load"] == pytest.approx(1.25)
    assert not any(v["pool_first"] for k, v in rd["runs"].items() if k[0] == "r1f-ysync")
    assert rep["warmup_rows"] == [pair.run_id_for(wj)] and rd["verdict"] == "ADOPT" and rd["n_present"] == 28


def test_pool_starts_skip_voided_rows(tmp_path):
    w = World(tmp_path)
    k = w.key(COMP, "T", 4)
    w.created[k] = 2.0
    w.extra.append(w._write("r1f-ysync", "comp21-2idx__A9@r1fctl__s9__120s__28e5a5e9c725", COMP, R.C_NEW, 9, 1.0,
                            dict(block_void=True), {}, p=0.55, g=280))
    rd = _one(w.report())
    assert rd["pool_starts"]["w-r1f-ysync"]["run_id"] == w.exp[k]


def test_reused_ids_and_other_rows(tmp_path):
    w = World(tmp_path)
    df = w.build()
    k = w.key(ITEM, "C", 0)
    dup = df[df["run_id"] == w.exp[k]].iloc[0].to_dict()
    dup.update(run_id=dup["run_id"] + "#r1-ysync", pt=0.99)
    other = dict(dup, run_id="sing44__A9@ysync__s0__120s__a08221cca653", instance="sing44", pt=0.4)
    rep = R.report(pd.concat([df, pd.DataFrame([dup, other])]), tmp_path)
    rd = _one(rep)
    assert rd["cells"][(ITEM, "C", 0)]["row"]["pt"] == pytest.approx(0.36)
    assert [d["run_id"] for d in rd["duplicates"]] == [dup["run_id"]]
    assert rep["other_rows"] == ["sing44__A9@ysync__s0__120s__a08221cca653"]


def test_render_prints_generations_per_second_and_the_load(tmp_path):
    txt = R.render(World(tmp_path).report())
    assert "gen/s 12.50" in txt and "load 1.25" in txt and "[beside]" in txt


def test_main_writes_json(tmp_path):
    w = World(tmp_path)
    p = tmp_path / "runs.parquet"
    w.build().to_parquet(p)
    out = tmp_path / "r.json"
    assert R.main(["--runs", str(p), "--root", str(tmp_path), "--json", str(out)]) == 0
    j = json.loads(out.read_text())
    assert j["readings"][0]["verdict"] == "ADOPT" and len(j["readings"][0]["runs"]) == 28
    assert len(j["readings"][0]["beside"]) == 4
    assert R.main(["--runs", str(tmp_path / "missing.parquet")]) == 2
