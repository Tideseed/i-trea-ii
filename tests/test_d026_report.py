"""`tools/d026_report.py` (D-026: R2 of the C1 diagnosis — lp_decode inside A9 after the EA's LP worker) on synthetic
run tables and run directories — no solver, no GPU. The expected runs and ids come from the job writer; validity as one
gate (complete, harness, config, compile, block, the base's flag state, the decoder in every treatment run and in no
control run); the three mechanism checks (ready <= 10 s after the gate, >= 1 decoded incumbent per seed on
item_placement and sing44, ca's LP solve wall <= 1.10 x the control's); the one-sided no-harm bands (ca 0.02, glass4 /
kumeu 0.05, and — the amendment — item_placement / sing44 by the control's four-seed range); the hint (<= 0.90); the
verdict; failed runs; the CLI."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from itrea.experiments import make_jobs as mj
from itrea.experiments import pair

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("d026_report", ROOT / "tools" / "d026_report.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

EXT = pair.resolve_harness("ext")
CA, ITEM, SING, GLASS, KUMEU = R.CA, "item_placement_9903", "sing44", "glass4", "neos-3656078-kumeu"
FLAGS = {"blocking_sync": 4, "auto": 0}


def summary(kind: str, mode: str, *, gate="open", gate_t=20.0, ready_w=24.0, lp_wall=17.0, inc=3, P=48, **over) -> dict:
    cs = dict(mode=mode, state="read" if mode == "auto" else "verified", active=True, sched_active=mode,
              flags_active=FLAGS[mode], sched_at_end=mode, active_at_end=True)
    s = dict(P=P, gene_dtype="float64", compiled_on_clock=False, cuda_sched=cs, phase_events_block=mode != "auto",
             generations=100, wall_s=120.0, concat_parts_max=4,
             lp=dict(state="done", t=19.5, worker_done_t=18.0, worker_wall_s=lp_wall, worker_cpu_share=0.9))
    if kind == "treatment":
        s["lp_decode"] = dict(gate=gate, gate_t=gate_t if gate == "open" else None, gate_why="lp_done",
                              ready_t_worker=ready_w, ready_t=ready_w + 0.5, wait_s_max=0.01, host_s=1.2, nice=10)
        s["lpd_incumbents"], s["lpd_incumbents_I"] = inc, 1
    s.update(over)
    return s


class World:
    def __init__(self, root: Path, base="bsync", nice=10, set_="r2-ld"):
        self.root, self.base, self.nice, self.set = root, base, nice, set_
        self.mode = R.base_mode(base)
        self.exp = R.expected(base, nice, set_)
        self.p = {k: 0.4 for k in self.exp}
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
            kind = "treatment" if lab == R.T else "control"
            rw = dict(run_id=rid, set=self.set, instance=inst, arm=lab, seed=s, budget=120.0, status="completed",
                      attempt=0, pt=self.p[k], harness=json.dumps(EXT, sort_keys=True), cuda_compiled_on_clock=False,
                      synthetic=False, block_void=False, lpd_worker_solve_cpu_s=2.5 if kind == "treatment" else None)
            rw.update(self.row.get(k, {}))
            d = self.root / "traces" / "raw" / self.set / rw["run_id"].split("#")[0]
            d.mkdir(parents=True, exist_ok=True)
            (d / "run.json").write_text(json.dumps(dict(run_id=rid, config=dict(harness=EXT),
                                                        compiled_on_clock_cuda=False)))
            sm = self.summ.get(k, {})
            if sm is not None:
                (d / "ea.summary.json").write_text(json.dumps(summary(kind, self.mode, **sm)))
            rw["run_dir"] = str(d.relative_to(self.root))
            rows.append(rw)
        return pd.DataFrame(rows + self.extra)

    def report(self, failed="both"):
        return R.report(self.build(), self.root, self.set, self.base, self.nice, failed)


def _one(rep):
    assert len(rep["readings"]) == 1
    return rep["readings"][0]


def _hint(w: World, t=0.3, c=0.4):
    for i in (ITEM, SING):
        for s in range(4):
            w.p[(i, R.T, s)], w.p[(i, R.CTL, s)] = t, c


def test_expected_runs_are_the_writers():
    exp = R.expected("bsync", 10, "r2-ld")
    assert len(exp) == 28 and set(exp.values()) == {pair.run_id_for(j) for j in mj.r2_ld("bsync", 10)}
    assert {k[1] for k in exp} == {"A9@r2ld", "A9@r2ctl"}
    assert R.SEEDS[ITEM] == (0, 1, 2, 3) and R.SEEDS[CA] == (0, 1)
    with pytest.raises(ValueError, match="--nice"):
        R.report(pd.DataFrame(columns=["run_id", "set", "arm", "seed", "status", "instance"]), Path("."),
                 "r2-ld", "bsync", None)


def test_no_effect_is_not_adopted_and_a_gain_with_every_mechanism_adopts(tmp_path):
    w = World(tmp_path)
    rd = _one(w.report())
    assert rd["validity"]["verdict"] == "VALID", rd["validity"]
    assert rd["rules"]["fails"] == ["no hint"] and rd["verdict"] == "NOT ADOPTED"
    _hint(w)
    rd = _one(w.report())
    assert rd["rules"]["hint"]["ratio"] == pytest.approx(0.75) and rd["verdict"] == "ADOPT"
    assert "VERDICT: ADOPT" in R.render(w.report())


@pytest.mark.parametrize("over, fail", [
    ({"ready_w": 30.01}, "mechanism 1"),                   # 10.01 s after the gate
    ({"inc": 0}, "mechanism 2")])
def test_mechanism_1_and_2(tmp_path, over, fail):
    w = World(tmp_path)
    _hint(w)
    w.summ[(SING, R.T, 3)] = over
    ru = _one(w.report())["rules"]
    assert ru["verdict"] == "NOT ADOPTED" and any(x.startswith(fail) for x in ru["fails"]), ru["fails"]


def test_ready_exactly_ten_seconds_after_the_gate_passes(tmp_path):
    w = World(tmp_path)
    _hint(w)
    w.summ[(ITEM, R.T, 0)] = dict(ready_w=30.0)
    assert _one(w.report())["verdict"] == "ADOPT"


@pytest.mark.parametrize("lp_t, ok", [(18.7, True), (18.71, False)])
def test_mechanism_3_ca_lp_wall_per_seed(tmp_path, lp_t, ok):
    w = World(tmp_path)
    _hint(w)
    w.summ[(CA, R.T, 1)] = dict(lp_wall=lp_t)               # control 17.0: 1.10 x = 18.7
    ru = _one(w.report())["rules"]
    assert ru["lp"][1]["ok"] is ok and ru["lp"][0]["ok"] and (ru["verdict"] == "ADOPT") is ok


@pytest.mark.parametrize("inst, delta, ok", [(CA, 0.02, True), (CA, 0.021, False), (GLASS, 0.05, True),
                                             (KUMEU, 0.051, False)])
def test_no_harm_floors(tmp_path, inst, delta, ok):
    w = World(tmp_path)
    _hint(w)
    for s in (0, 1):
        w.p[(inst, R.T, s)] = 0.4 + delta
    ru = _one(w.report())["rules"]
    assert ru["harm"][inst]["ok"] is ok and (ru["verdict"] == "ADOPT") is ok


def test_the_amendment_band_on_item_and_sing44_uses_the_four_seed_range(tmp_path):
    w = World(tmp_path)
    for s in range(4):
        w.p[(ITEM, R.T, s)], w.p[(ITEM, R.CTL, s)] = 0.10, 0.40      # a large item gain ...
        w.p[(SING, R.T, s)], w.p[(SING, R.CTL, s)] = 0.46, 0.40      # ... carries a sing44 loss through the hint
    ru = _one(w.report())["rules"]
    assert ru["hint"]["present"] and not ru["harm"][SING]["ok"]
    assert ru["fails"] == ["no-harm on the hint instances (amendment)"]
    w.p[(SING, R.CTL, 0)], w.p[(SING, R.CTL, 3)] = 0.36, 0.44          # range 0.08 > the loss 0.06
    ru = _one(w.report())["rules"]
    assert ru["harm"][SING]["band"] == pytest.approx(0.08) and ru["harm"][SING]["ok"] and ru["verdict"] == "ADOPT"


@pytest.mark.parametrize("case, leg", [("gate_closed", "decoder"), ("no_block", "decoder"),
                                       ("decoder_in_control", "decoder"), ("flag", "flag"), ("missing", "complete")])
def test_validity_legs(tmp_path, case, leg):
    w = World(tmp_path)
    if case == "gate_closed":
        w.summ[(GLASS, R.T, 0)] = dict(gate="waiting")
    elif case == "no_block":
        w.summ[(GLASS, R.T, 0)] = dict(lp_decode=None)
    elif case == "decoder_in_control":
        w.summ[(GLASS, R.CTL, 0)] = dict(lp_decode=dict(gate="none"))
    elif case == "flag":
        w.summ[(GLASS, R.CTL, 0)] = dict(cuda_sched=dict(mode="auto", state="read", sched_active="auto"))
    elif case == "missing":
        w.drop.add((GLASS, R.T, 0))
    rd = _one(w.report())
    assert rd["verdict"] == "INVALID" and leg in rd["validity"]["failing"], rd["validity"]


def test_a_base_without_r1_reads_cudas_default(tmp_path):
    w = World(tmp_path, base="r13x", nice=None)
    _hint(w)
    rep = w.report()
    assert rep["nice"] == 0 and _one(rep)["validity"]["verdict"] == "VALID" and _one(rep)["verdict"] == "ADOPT"


def test_failed_runs_kept_and_excluded(tmp_path):
    w = World(tmp_path)
    _hint(w)
    k = (KUMEU, R.CTL, 1)
    w.row[k] = dict(status="failed")
    rep = w.report()
    keep, excl = rep["readings"]
    assert keep["validity"]["verdict"] == "VALID" and keep["validity"]["deviation"]
    assert excl["validity"]["failing"] == ["complete"]


def test_gate_lag_and_a3_beside_are_reported(tmp_path):
    w = World(tmp_path)
    w.extra.append(dict(run_id="item_placement_9903__A3@ld1__s0__120s__x", set="dev", instance=ITEM, arm="A3@ld1",
                        seed=0, budget=120.0, status="completed", pt=0.1, lpd_wait_s_max=0.02, lpd_host_s=2.0,
                        run_dir="nowhere"))
    rd = _one(w.report())
    c = rd["cells"][(ITEM, R.T, 0)]["mech"]
    assert c["gate_lag"] == pytest.approx(2.0) and c["ready_after_gate"] == pytest.approx(4.0)
    assert rd["a3"][(ITEM, 0)] == dict(wait_s_max=0.02, host_s=2.0)
    assert "A3@ld1 0.0200 / 2.000" in R.render(w.report())


def test_main_cli(tmp_path):
    w = World(tmp_path, nice=0)
    _hint(w)
    p = tmp_path / "runs.parquet"
    w.build().to_parquet(p)
    out = tmp_path / "r.json"
    assert R.main(["--runs", str(p), "--root", str(tmp_path), "--base", "bsync", "--nice", "0", "--json",
                   str(out)]) == 0
    assert json.loads(out.read_text())["readings"][0]["verdict"] == "ADOPT"
    assert R.main(["--runs", str(p), "--root", str(tmp_path), "--base", "bsync"]) == 2           # --nice missing
    rep = R.report(pd.read_parquet(p), tmp_path, "r2-ld", "bsync", 10)                        # written at nice 0
    assert _one(rep)["validity"]["failing"] == ["config"]
