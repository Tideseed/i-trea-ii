"""`tools/d025_report.py` (D-025: R1 of the C1 diagnosis — the EA's blocking sync — and its fallbacks bnice / ysync) on
synthetic run tables and run directories — no solver, no GPU. The expected runs and ids come from the job writer;
validity as one gate (complete, harness, config = the job's run id, compile, equal P / gene dtype per block, the flag
state per arm: blocking sync set and held, yield, or CUDA's default read back with the +10 renice for bnice and none
in the control); failed runs kept (the verdict's reading) and excluded (the letter); the no-harm legs (generations
on seed means against 1 − max(0.05, the control's relative seed spread) — the amendment —, the one-sided P band on item_placement and comp21 and — the amendment — on the three hint
instances); the hint (ratio of sums <= 0.95); the verdict; the fallback triggers and their precedence; the control's
detection; the JSON output."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from itrea.experiments import make_jobs as mj
from itrea.experiments import pair

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("d025_report", ROOT / "tools" / "d025_report.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

EXT = pair.resolve_harness("ext")
CA, SING, NEOS, ITEM, COMP = R.CA, "sing44", "neos-960392", "item_placement_9903", "comp21-2idx"
FLAGS = {"blocking_sync": 4, "yield": 2, "auto": 0}


def summary(kind: str, variant: str, *, gens=100, lp_share=0.9, renice_gap=10, P=48, dtype="int8", **over) -> dict:
    """An EA summary as the engine writes it for this arm (the flag state of `kind` / `variant`)."""
    mode = R.FLAG_MODE[variant] if kind == "treatment" else "auto"
    cs = dict(mode=mode, requested=FLAGS[mode], state="read" if mode == "auto" else "verified", active=True,
              sched_active=mode, flags_active=FLAGS[mode], sched_at_end=mode, active_at_end=True)
    s = dict(P=P, gene_dtype=dtype, compiled_on_clock=False, cuda_sched=cs, phase_events_block=mode == "blocking_sync",
             generations=gens, wall_s=120.0, ls_launches=10 * gens,
             cpu_clock=dict(self_per_wall_s=0.2 if mode == "blocking_sync" else 1.0, main_thread_per_wall_s=0.2),
             lp=dict(state="done", t=20.0, worker_done_t=18.0, worker_wall_s=17.0, worker_cpu_share=lp_share))
    if kind == "treatment" and variant == "bnice":
        s["host_renice"] = dict(value=10, before=0, target=renice_gap, gap=renice_gap, set=5, kept=0, threads=5)
    s.update(over)
    return s


class World:
    """The set's 20 runs: P per (instance, label) and seed, every file written under `root`."""

    def __init__(self, root: Path, control="r13x", variant="bsync", set_=None):
        self.root, self.control, self.variant = root, control, variant
        self.set = set_ or f"r1-{variant}"
        self.t, self.c = R.labels(control, variant)
        self.exp = R.expected(control, variant, self.set)
        self.p = {k: 0.5 for k in self.exp}
        for s in (0, 1):                                  # a realistic control on the hint instances
            self.p[(CA, self.c, s)] = self.p[(CA, self.t, s)] = 0.39
            self.p[(SING, self.c, s)] = self.p[(SING, self.t, s)] = 0.44
            self.p[(NEOS, self.c, s)] = self.p[(NEOS, self.t, s)] = 0.06
        self.summ: dict = {}
        self.row: dict = {}
        self.extra: list = []
        self.drop: set = set()

    def build(self) -> pd.DataFrame:
        rows = []
        for k, rid in self.exp.items():
            if k in self.drop:
                continue
            inst, lab, s = k
            kind = "treatment" if lab == self.t else "control"
            rw = dict(run_id=rid, set=self.set, instance=inst, arm=lab, seed=s, budget=120.0, status="completed",
                      attempt=0, pt=self.p[k], harness=json.dumps(EXT, sort_keys=True), cuda_compiled_on_clock=False,
                      synthetic=False, block_void=False)
            rw.update(self.row.get(k, {}))
            d = self.root / "traces" / "raw" / self.set / rw["run_id"].split("#")[0]
            d.mkdir(parents=True, exist_ok=True)
            rj = dict(run_id=rid, config=dict(harness=rw.pop("_rj_harness", EXT)),
                      compiled_on_clock_cuda=rw.pop("_rj_compiled", False))
            (d / "run.json").write_text(json.dumps(rj))
            sm = self.summ.get(k, {})
            if sm is not None:
                (d / "ea.summary.json").write_text(json.dumps(summary(kind, self.variant, **sm)))
            rw["run_dir"] = str(d.relative_to(self.root))
            rows.append(rw)
        return pd.DataFrame(rows + self.extra)

    def report(self, failed="both", **kw):
        return R.report(self.build(), self.root, self.set, kw.pop("control", None), kw.pop("variant", None), failed)


def _one(rep):
    assert len(rep["readings"]) == 1
    return rep["readings"][0]


def test_expected_runs_are_the_writers():
    exp = R.expected("r13x", "bsync", "r1-bsync")
    jobs = mj.r1_bsync("r13x", "bsync")
    assert len(exp) == 20 and set(exp.values()) == {pair.run_id_for(j) for j in jobs}
    assert R.labels("r13x", "bsync") == ("A9@bsync", "A9@r1ctl")
    assert R.labels("boY", "bnice") == ("A9@boY-bnice", "A9@r1ctl-bnice")
    assert R.variant_of("r1-ysync") == "ysync" and R.variant_of("r1-bsync-2") is None
    assert set(R.HINT_INSTANCES) | set(R.GUARD_INSTANCES) == set(mj.R1_INSTANCES)


def test_no_effect_is_valid_and_not_adopted_with_no_trigger(tmp_path):
    w = World(tmp_path)
    rd = _one(w.report())
    assert rd["validity"]["verdict"] == "VALID", rd["validity"]
    ru = rd["rules"]
    assert ru["verdict"] == "NOT ADOPTED" and ru["fails"] == ["no hint"] and ru["hint"]["ratio"] == pytest.approx(1.0)
    assert ru["triggers"] == dict(a=False, b=False) and "ends" in ru["next"]


def test_a_ca_gain_gives_the_hint_and_adopt(tmp_path):
    w = World(tmp_path)
    for s in (0, 1):
        w.p[(CA, w.t, s)] = 0.27
    rd = _one(w.report())
    h = rd["rules"]["hint"]
    assert h["present"] and h["ratio"] == pytest.approx((0.27 * 2 + 0.44 * 2 + 0.06 * 2) / (0.39 * 2 + 0.44 * 2 + 0.12))
    assert rd["verdict"] == "ADOPT" and "widens" in rd["rules"]["next"]
    txt = R.render(w.report())
    assert "VERDICT: ADOPT" in txt and "PRESENT" in txt


def test_the_amendment_band_on_the_hint_instances_gates(tmp_path):
    """The finding's case: ca −0.15 with sing44 +0.06 opens the pooled hint, but sing44's band (0.05) fails."""
    w = World(tmp_path)
    for s in (0, 1):
        w.p[(CA, w.t, s)] = 0.24
        w.p[(SING, w.t, s)] = 0.50
    rd = _one(w.report())
    ru = rd["rules"]
    assert ru["hint"]["present"] and not ru["hint_band"][SING]["ok"] and ru["hint_band"][CA]["ok"]
    assert ru["verdict"] == "NOT ADOPTED" and ru["fails"] == ["P no-harm on the hint instances (amendment)"]
    assert ru["triggers"] == dict(a=False, b=False)


@pytest.mark.parametrize("gt, gc, ok", [((100, 90), (100, 100), True), ((100, 89), (100, 100), False),
                                        ((100, 60), (90, 110), True), ((99, 60), (90, 110), False)])
def test_generation_guard_on_seed_means_with_the_controls_spread_and_trigger_b(tmp_path, gt, gc, ok):
    """D-025 as amended: Ḡ(T) / Ḡ(C) >= 1 − max(0.05, |G_C,s0 − G_C,s1| / Ḡ(C)): 190 / 200 = 0.95 holds at the floor,
    189 / 200 fails; a control at 90 / 110 (relative spread 0.2) sets 0.8: 160 / 200 holds, 159 / 200 fails."""
    w = World(tmp_path)
    for s in (0, 1):
        w.p[(CA, w.t, s)] = 0.27                          # a hint, so only the guard decides
        w.summ[(ITEM, w.c, s)], w.summ[(ITEM, w.t, s)] = dict(gens=gc[s]), dict(gens=gt[s])
    rd = _one(w.report())
    g = rd["rules"]["guard_g"][ITEM]
    assert g["ok"] is ok and g["threshold"] == pytest.approx(1 - max(0.05, abs(gc[0] - gc[1]) / (sum(gc) / 2)))
    assert (rd["rules"]["verdict"] == "ADOPT") is ok and rd["rules"]["triggers"]["b"] is (not ok)
    if not ok:
        assert rd["rules"]["next"].startswith("fallback (b) ysync")


@pytest.mark.parametrize("delta, spread, ok", [(0.05, 0.0, True), (0.051, 0.0, False), (0.07, 0.08, True),
                                               (-0.3, 0.0, True)])
def test_p_band_is_one_sided_on_seed_means(tmp_path, delta, spread, ok):
    w = World(tmp_path)
    for s in (0, 1):
        w.p[(CA, w.t, s)] = 0.27
    w.p[(COMP, w.c, 0)], w.p[(COMP, w.c, 1)] = 0.55 - spread / 2, 0.55 + spread / 2
    w.p[(COMP, w.t, 0)] = w.p[(COMP, w.t, 1)] = 0.55 + delta
    ru = _one(w.report())["rules"]
    assert ru["guard_p"][COMP]["ok"] is ok and ru["guard_p"][COMP]["band"] == pytest.approx(max(0.05, spread))


def test_trigger_a_needs_no_hint_and_ca_lp_share_below_08_on_both_seeds_and_comes_first(tmp_path):
    w = World(tmp_path)
    for s in (0, 1):
        w.summ[(CA, w.t, s)] = dict(lp_share=0.6)
    ru = _one(w.report())["rules"]
    assert ru["triggers"]["a"] and ru["next"].startswith("fallback (a) bnice")
    w.summ[(ITEM, w.t, 0)] = dict(gens=50)                 # (b) fires too: (a) first, no chaining
    ru = _one(w.report())["rules"]
    assert ru["triggers"] == dict(a=True, b=True) and "(a) first" in ru["next"]
    w2 = World(tmp_path / "w2")
    w2.summ[(CA, w2.t, 0)] = dict(lp_share=0.6)            # one seed only: no trigger
    assert not _one(w2.report())["rules"]["triggers"]["a"]


@pytest.mark.parametrize("variant", ["bnice", "ysync"])
def test_a_fallback_set_reads_the_same_rules_and_never_chains(tmp_path, variant):
    w = World(tmp_path, variant=variant)
    rd = _one(w.report())
    assert rd["validity"]["verdict"] == "VALID", rd["validity"]["legs"]
    assert rd["treatment_label"] == f"A9@{variant}" and rd["control_label"] == f"A9@r1ctl-{variant}"
    assert rd["rules"]["triggers"] is None and "no fallback of a fallback" in rd["rules"]["next"]


@pytest.mark.parametrize("case, leg", [
    ("treatment_spins", "flag"), ("control_flag", "flag"), ("control_renice", "flag"), ("events", "flag"),
    ("harness", "harness"), ("compile_rj", "compile"), ("compile_summ", "compile"), ("config", "config"),
    ("block_P", "block"), ("missing", "complete"), ("no_summary", "flag")])
def test_validity_legs(tmp_path, case, leg):
    w = World(tmp_path)
    k_t, k_c = (ITEM, w.t, 0), (ITEM, w.c, 0)
    if case == "treatment_spins":
        w.summ[k_t] = dict(cuda_sched=dict(mode="blocking_sync", state="verified", sched_active="auto",
                                           sched_at_end="auto"))
    elif case == "control_flag":
        w.summ[k_c] = dict(cuda_sched=dict(mode="auto", state="read_error", read_error="x"))
    elif case == "control_renice":
        w.summ[k_c] = dict(host_renice=dict(gap=10))
    elif case == "events":
        w.summ[k_t] = dict(phase_events_block=False)
    elif case == "harness":
        w.row[k_t] = dict(_rj_harness={"cuda_cache": True})
    elif case == "compile_rj":
        w.row[k_t] = dict(_rj_compiled=True)
    elif case == "compile_summ":
        w.summ[k_t] = dict(compiled_on_clock=True)
    elif case == "config":
        w.row[k_t] = dict(run_id=w.exp[k_t].replace("__330a10a77556", "__ffffffffffff"))
    elif case == "block_P":
        w.summ[k_t] = dict(P=64)
    elif case == "missing":
        w.drop.add(k_t)
    elif case == "no_summary":
        w.summ[k_t] = None
    rd = _one(w.report())
    assert rd["validity"]["verdict"] == "INVALID" and leg in rd["validity"]["failing"], rd["validity"]
    assert rd["rules"] is None and rd["verdict"] == "INVALID"
    assert "RULES NOT READ" in R.render(w.report())


def test_bnice_needs_the_renice_gap_of_ten(tmp_path):
    w = World(tmp_path, variant="bnice")
    w.summ[(CA, w.t, 1)] = dict(renice_gap=9)
    rd = _one(w.report())
    assert rd["validity"]["failing"] == ["flag"] and "gap 9" in rd["validity"]["legs"]["flag"][0]


def test_failed_runs_are_kept_for_the_verdict_and_excluded_beside(tmp_path):
    w = World(tmp_path)
    k = (SING, w.t, 1)
    w.row[k] = dict(status="failed")
    w.summ[k] = None                                       # the EA died: no summary
    rep = w.report()
    keep, excl = rep["readings"]
    assert keep["keep_failed"] and not excl["keep_failed"] and rep["failed_runs"] == [w.exp[k]]
    assert keep["validity"]["verdict"] == "VALID" and keep["validity"]["deviation"]
    assert any("flag unread" in x for x in keep["validity"]["notes"])
    assert excl["validity"]["verdict"] == "INVALID" and excl["validity"]["failing"] == ["complete"]
    assert any("failed (not completed)" in x for x in excl["validity"]["legs"]["complete"])
    assert _one(w.report(failed="keep"))["keep_failed"]


def test_a_failed_guard_run_without_a_summary_fails_its_guard(tmp_path):
    w = World(tmp_path)
    for s in (0, 1):
        w.p[(CA, w.t, s)] = 0.27
    k = (COMP, w.t, 0)
    w.row[k] = dict(status="failed")
    w.summ[k] = None
    ru = _one(w.report(failed="keep"))["rules"]
    assert not ru["guard_g"][COMP]["ok"] and ru["verdict"] == "NOT ADOPTED"


def test_control_detection_reused_ids_and_other_rows(tmp_path):
    w = World(tmp_path, control="boY")
    df = w.build()
    assert R.detect_control(df, "r1-bsync", "bsync") == "boY"
    k = (CA, w.c, 0)
    dup = df[(df["instance"] == CA) & (df["arm"] == w.c) & (df["seed"] == 0)].iloc[0].to_dict()
    dup.update(run_id=dup["run_id"] + "#r1-bsync", pt=0.99)
    other = dict(dup, run_id="x__A9@other__s0__120s__abc", arm="A9@other", pt=0.1)
    rep = R.report(pd.concat([df, pd.DataFrame([dup, other])]), tmp_path, "r1-bsync")
    rd = _one(rep)
    assert rep["control"] == "boY" and rd["cells"][k]["row"]["pt"] == 0.39
    assert [d["run_id"] for d in rd["duplicates"]] == [dup["run_id"]]
    assert rep["other_rows"] == ["x__A9@other__s0__120s__abc"]
    with pytest.raises(ValueError, match="--control"):
        R.report(df[df["arm"] == w.c], tmp_path, "r1-bsync")


def test_main_writes_json_and_refuses_an_unknown_set(tmp_path):
    w = World(tmp_path)
    p = tmp_path / "runs.parquet"
    w.build().to_parquet(p)
    out = tmp_path / "r.json"
    assert R.main(["--runs", str(p), "--root", str(tmp_path), "--json", str(out)]) == 0
    j = json.loads(out.read_text())
    assert j["readings"][0]["verdict"] == "NOT ADOPTED" and len(j["readings"][0]["runs"]) == 20
    assert R.main(["--runs", str(p), "--root", str(tmp_path), "--set", "r1-x"]) == 2


def test_an_empty_table_reads_invalid_with_nothing_present(tmp_path):
    cols = ["run_id", "set", "instance", "arm", "seed", "budget", "status", "attempt", "pt", "run_dir"]
    rep = R.report(pd.DataFrame(columns=cols), tmp_path, "r1-bsync", control="r13x")
    rd = _one(rep)
    assert rd["n_present"] == 0 and rd["validity"]["verdict"] == "INVALID" and len(rd["validity"]["legs"]["complete"]) == 20
