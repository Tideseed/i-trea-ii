"""`tools/c1r_report.py` (D-030: the C1 re-read on the frozen-v2 candidate, set c1r-<a9>) on synthetic run tables and
run directories — no solver, no GPU. The expected runs and ids come from the job writer (pinned hashes); validity as one
gate (68 of 68, harness ext, configs as the job file, no compile on the clock — CuPy only on A9 —, the two runs of a
block in one window, the CUDA flag as written — gating since D-030's amendment of 2026-10-15); C1's hint (ratio of
means over the fixed 11 <= 0.95) and guard (mean bounded d over the 17 >= -0.05) exactly as D-022's own functions
compute them; D-025 amendment 3's harm leg against A9@r13x of 10-13 (cross-day; band_i from A9@r13x and A6F@r13x as
D-030's amendment reads D-023's band_i — never from the treatment; the threshold and the cannot-fire instances fixed by
10-13's runs); beside: the other readings' bands, the floor marker, the null lines, the cannot-fire instances' seeds and
generations, vs SCIP alone on SCIP's weak set among all 17, voids, frozen-v2; the 10-13 reading beside; failed runs
kept, the letter beside; the CLI."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itrea.analysis.stats import effect_d
from itrea.experiments import pair

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("c1r_report", ROOT / "tools" / "c1r_report.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
D22 = R.D22

EXT = pair.resolve_harness("ext")
DEV = R.development_instances()
ELEVEN = list(R.C1_INSTANCES)
OTHERS = [i for i in DEV if i not in ELEVEN]


class World:
    """A complete, valid c1r set (every P 0.4 unless set), D-022's 68 C1 runs of 10-13 (set itx-r13) and A0's runs on
    the fixed 11 (set dev)."""

    def __init__(self, root: Path, a9: str = "ld0", set_: str | None = None):
        self.root, self.a9 = root, a9
        self.set = set_ or f"c1r-{a9}"
        self.exp = R.expected(a9, self.set)
        self.p = {k: 0.4 for k in self.exp}
        self.old = {(i, lab, s): 0.4 for i in DEV for lab in (R.OLD_A9, R.OLD_A6F) for s in (0, 1)}
        self.a0 = {(i, s): 0.8 for i in ELEVEN for s in (0, 1)}
        self.row: dict = {}
        self.rj: dict = {}
        self.summ: dict = {}
        self.window: dict = {}
        self.drop: set = set()
        self.extra: list = []
        self.old_gens: dict = {}

    def set_p(self, inst: str, label: str, seeds_p) -> None:
        for s, v in zip((0, 1), seeds_p):
            self.p[(inst, label, s)] = v

    def set_old(self, inst: str, label: str, seeds_p) -> None:
        for s, v in zip((0, 1), seeds_p):
            self.old[(inst, label, s)] = v

    def ea_summary(self, k) -> dict:
        mode = R.FLAG[self.a9]
        cs = dict(state="read" if mode == "auto" else "verified", sched_active=mode, sched_at_end=mode, active=True)
        s = dict(P=48, gene_dtype="float64", compiled_on_clock=False, cuda_sched=cs, phase_events_block=False,
                 generations=100, lp_decode=dict(gate="open", gate_why="lp_done", gate_t=20.0))
        s.update(self.summ.get(k) or {})
        return s

    def build(self) -> pd.DataFrame:
        rows = []
        for k, rid in self.exp.items():
            if k in self.drop:
                continue
            inst, lab, s = k
            rw = dict(run_id=rid, set=self.set, instance=inst, arm=lab, seed=s, budget=120.0, status="completed",
                      attempt=0, pt=self.p[k], harness=json.dumps(EXT, sort_keys=True), cuda_compiled_on_clock=False,
                      synthetic=False, block_void=False, window_id=self.window.get(k, "W1"))
            rw.update(self.row.get(k, {}))
            d = self.root / "traces" / "raw" / self.set / str(rw["run_id"]).split("#")[0]
            d.mkdir(parents=True, exist_ok=True)
            rj = dict(run_id=rid, config=dict(harness=EXT), compiled_on_clock_cuda=False,
                      cupy_cache=dict(compiled_on_clock=False))
            rj.update(self.rj.get(k, {}))
            (d / "run.json").write_text(json.dumps(rj))
            if lab == R.A9 and not (k in self.summ and self.summ[k] is None):
                (d / "ea.summary.json").write_text(json.dumps(self.ea_summary(k)))
            rw["run_dir"] = str(d.relative_to(self.root))
            rows.append(rw)
        for (i, lab, s), v in self.old.items():
            rid, rdir = f"{i}__{lab}__s{s}__120s__old", ""
            if lab == R.OLD_A9 and (i, s) in self.old_gens:
                d = self.root / "traces" / "raw" / "itx-r13" / rid
                d.mkdir(parents=True, exist_ok=True)
                (d / "ea.summary.json").write_text(json.dumps(dict(generations=self.old_gens[(i, s)])))
                rdir = str(d.relative_to(self.root))
            rows.append(dict(run_id=rid, set="itx-r13", instance=i, arm=lab, seed=s,
                             budget=120.0, status="completed", attempt=0, pt=v, synthetic=False, block_void=False,
                             window_id="W0", run_dir=rdir))
        for (i, s), v in self.a0.items():
            rows.append(dict(run_id=f"{i}__A0__s{s}__120s__a0", set="dev", instance=i, arm="A0", seed=s, budget=120.0,
                             status="completed", attempt=0, pt=v, synthetic=False, block_void=False, run_dir=""))
        return pd.DataFrame(rows + self.extra)

    def report(self, failed: str = "both") -> dict:
        return R.report(self.build(), self.root, self.a9, self.set, failed)


def _first(rep) -> dict:
    return rep["readings"][0]


# ============================================================================================ the expected runs
@pytest.mark.parametrize("a9", ["ld0", "ysync-ld0"])
def test_expected_runs_are_the_committed_job_file(a9):
    """The report's 68 expected run ids are exactly the ids of jobs/c1r-<a9>.jsonl (the writer's pinned hashes)."""
    jobs = [json.loads(x) for x in (ROOT / "jobs" / f"c1r-{a9}.jsonl").read_text().splitlines()]
    exp = R.expected(a9, f"c1r-{a9}")
    assert len(exp) == 68 and set(exp.values()) == {pair.run_id_for(j) for j in jobs}
    assert {k[1] for k in exp} == {"A9@c1r", "A6F@c1r"} and {k[0] for k in exp} == set(DEV)
    assert {v.rsplit("__", 1)[1] for k, v in exp.items() if k[1] == "A6F@c1r"} == {"82a371305f88"}
    assert {v.rsplit("__", 1)[1] for k, v in exp.items() if k[1] == "A9@c1r"} == {
        {"ld0": "341267810eed", "ysync-ld0": "f345de0ab936"}[a9]}


def test_unknown_candidate_is_refused(tmp_path):
    with pytest.raises(ValueError, match="--a9"):
        R.report(pd.DataFrame(), tmp_path, "ysync")


# ============================================================================================ the rules
def test_valid_set_reads_hint_guard_and_harm(tmp_path):
    """A9 0.3 vs A6F 0.4 on the fixed 11, equal on the other 6: hint 0.75 present, guard 11 x 0.25 / 17 holds, the new
    A9 better than A9@r13x (0.4) everywhere it moved: no harm; the 10-13 reading beside (all 0.4: ratio 1, d 0)."""
    w = World(tmp_path)
    for i in ELEVEN:
        w.set_p(i, R.A9, (0.3, 0.3))
    rep = w.report()
    assert len(rep["readings"]) == 1 and not rep["failed_runs"]
    rd = _first(rep)
    assert rd["validity"]["verdict"] == "VALID" and rd["n_present"] == 68
    ru = rd["rules"]
    assert ru["hint"]["state"] == "present" and ru["hint"]["ratio"] == pytest.approx(0.75)
    assert ru["guard"]["state"] == "holds" and ru["guard"]["mean_d"] == pytest.approx(11 * 0.25 / 17)
    assert ru["harm"]["state"] == "none" and ru["harm"]["harm"] == [] and not ru["harm"]["unread"]
    assert ru["then"]["hint"]["ratio"] == pytest.approx(1.0) and ru["then"]["guard"]["mean_d"] == pytest.approx(0.0)
    assert ru["then"]["n_runs"] == 68 and ru["then"]["recorded"] == dict(hint=0.827, guard=0.063)
    s = rd["summary"]
    assert (s["validity"], s["hint"], s["guard"], s["harm"], s["harm_list"]) == ("VALID", "present", "holds", "none", [])
    assert "go / no-go on C1 is Berk's" in s["berk"]


def test_hint_absent_and_guard_fails(tmp_path):
    w = World(tmp_path)
    for i in DEV:
        w.set_p(i, R.A9, (0.6, 0.6))
    ru = _first(w.report())["rules"]
    assert ru["hint"]["state"] == "absent" and ru["hint"]["ratio"] == pytest.approx(1.5)
    assert ru["guard"]["state"] == "fails" and ru["guard"]["mean_d"] == pytest.approx(-1 / 3)
    assert ru["harm"]["state"] == "HARM" and ru["harm"]["harm"] == DEV          # 0.4 -> 0.6: d -1/3 < -0.05


@pytest.mark.parametrize("a9_p, state", [(0.94 * 0.4, "present"), (0.96 * 0.4, "absent")])
def test_hint_threshold_is_095_on_the_fixed_11_only(tmp_path, a9_p, state):
    """The ratio reads the fixed 11 only: the other six can be anything."""
    w = World(tmp_path)
    for i in ELEVEN:
        w.set_p(i, R.A9, (a9_p, a9_p))
    for i in OTHERS:
        w.set_p(i, R.A9, (0.01, 0.9))
    assert _first(w.report())["rules"]["hint"]["state"] == state


def test_hint_and_guard_are_d022s_own_computation(tmp_path):
    """On irregular P values the report's hint and guard equal (a) D-022's c1_hint / c1_guard called directly on the
    same values under D-022's labels and (b) the definitions by hand: ratio of the means of seed means over the 11;
    mean of effect_d(seed-mean A6F, seed-mean A9) over the 17 (both P = 1 at d = 0). The 10-13 reading beside, on the
    same values, gives the same numbers."""
    rng = np.random.default_rng(30)
    w = World(tmp_path)
    for i in DEV:
        a, b = rng.uniform(0.0, 1.0, 2), rng.uniform(0.0, 1.0, 2)
        w.set_p(i, R.A9, a)
        w.set_p(i, R.A6F, b)
        w.set_old(i, R.OLD_A9, a)
        w.set_old(i, R.OLD_A6F, b)
    for lab, old in ((R.A9, R.OLD_A9), (R.A6F, R.OLD_A6F)):    # both P = 1: d = 0 (D-013 counts rail01 1.0 -> 1.0)
        w.set_p("rail01", lab, (1.0, 1.0))
        w.set_old("rail01", old, (1.0, 1.0))
    ru = _first(w.report())["rules"]
    long = pd.DataFrame([dict(instance=i, arm=R.TO_D022[lab], seed=s, pt=v) for (i, lab, s), v in w.p.items()])
    h, g = D22.c1_hint(long, ELEVEN), D22.c1_guard(long, DEV)
    assert ru["hint"]["ratio"] == pytest.approx(h["ratio"]) and ru["guard"]["mean_d"] == pytest.approx(g["mean_d"])
    m = {(i, lab): np.mean([w.p[(i, lab, s)] for s in (0, 1)]) for i in DEV for lab in (R.A9, R.A6F)}
    assert ru["hint"]["ratio"] == pytest.approx(np.mean([m[(i, R.A9)] for i in ELEVEN])
                                                / np.mean([m[(i, R.A6F)] for i in ELEVEN]))
    assert ru["guard"]["mean_d"] == pytest.approx(np.mean([effect_d(m[(i, R.A6F)], m[(i, R.A9)]) for i in DEV]))
    assert ru["then"]["hint"]["ratio"] == pytest.approx(ru["hint"]["ratio"])
    assert ru["then"]["guard"]["mean_d"] == pytest.approx(ru["guard"]["mean_d"])
    assert ru["table"]["rail01"]["d"] == 0.0


def test_harm_leg_is_d025_amendment_3_with_d023s_band(tmp_path):
    """Per instance, the new A9 vs A9@r13x of 10-13 on seed means: d_i = effect_d(P̄ r13x, P̄ new), band_i = max(0.05,
    the bounded seed spread of A9@r13x (D-023's control), that of A6F@r13x (D-023's A6F)) — D-030's amendment of
    2026-10-15; harm iff d_i < -band_i. The treatment's own spread never widens its band (D-030's first reading did:
    a noisy new A9 hid its own loss), nor does A6F@c1r's; both are printed beside."""
    w = World(tmp_path)
    x, y, z, v, u, t = "glass4", "sing44", "rail01", "comp21-2idx", "neos-960392", "item_placement_9903"
    w.set_old(x, R.OLD_A9, (0.3, 0.3))
    w.set_p(x, R.A9, (0.4, 0.4))                     # d -0.25, band 0.05 -> harm
    w.set_old(y, R.OLD_A9, (0.3, 0.3))
    w.set_p(y, R.A9, (0.2, 0.6))                     # d -0.25; the new arm's spread 0.667 no longer masks it -> harm
    w.set_old(z, R.OLD_A9, (0.1, 0.5))
    w.set_p(z, R.A9, (0.4, 0.4))                     # d -0.25, A9@r13x's spread 0.8 -> no harm
    w.set_old(v, R.OLD_A9, (0.3, 0.3))
    w.set_p(v, R.A9, (0.4, 0.4))
    w.set_old(v, R.OLD_A6F, (0.1, 0.9))              # A6F@r13x's spread 0.889 widens the band -> no harm
    w.set_old(t, R.OLD_A9, (0.3, 0.3))
    w.set_p(t, R.A9, (0.4, 0.4))
    w.set_p(t, R.A6F, (0.1, 0.9))                    # A6F@c1r's spread is beside only -> harm
    w.set_old(u, R.OLD_A9, (0.40, 0.40))
    w.set_p(u, R.A9, (0.4166, 0.4166))               # d -0.0398 within the floor 0.05 -> no harm
    rd = _first(w.report())
    h = rd["rules"]["harm"]
    rows = {r["instance"]: r for r in h["rows"]}
    assert h["state"] == "HARM" and h["harm"] == [i for i in DEV if i in (x, y, t)]
    assert rows[x]["d"] == pytest.approx(-0.25) and rows[x]["band"] == pytest.approx(0.05)
    assert rows[y]["band"] == pytest.approx(0.05) and rows[y]["harm"]
    assert rows[y]["band_first"] == pytest.approx(0.4 / 0.6)             # D-030's first reading, beside
    assert rows[z]["band"] == pytest.approx(0.8) and not rows[z]["harm"]
    assert rows[v]["band"] == pytest.approx(0.8 / 0.9) and not rows[v]["harm"]
    assert rows[t]["band"] == pytest.approx(0.05) and rows[t]["harm"]
    assert rows[t]["band_a6_new"] == pytest.approx(0.8 / 0.9)            # A6F@c1r in A6F@r13x's place, beside
    assert rows[u]["d"] == pytest.approx(-0.0166 / 0.4166) and not rows[u]["harm"]
    assert rows[x]["dp"] == pytest.approx(0.1) and not rows[x]["floor_dp"]
    assert rd["summary"]["harm_list"] == h["harm"]
    text = R.render(w.report())
    assert "HARM (D-025 amendment 3, cross-day vs A9@r13x of 10-13) on glass4" in text and "to Berk" in text


def test_threshold_and_cannot_fire_are_fixed_by_10_13_alone(tmp_path):
    """The threshold (the smallest P̄ of the new A9 at which harm fires) depends on A9@r13x and A6F@r13x only:
    harm fires just above it and not at it; where it is > 1 (A9@r13x at P = 1, or a band so wide that P would have to
    exceed 1 — crna on 10-13: 0.752 / 0.170) harm CANNOT fire, whatever the new A9 does; their seeds and generations
    are printed beside."""
    assert R.p_threshold(0.4, 0.05) == pytest.approx(0.4 / 0.95) and R.p_threshold(1.0, 0.05) > 1
    assert R.p_threshold(0.0, 0.05) == pytest.approx(0.05e-3) and R.p_threshold(0.5, 1.0) == float("inf")
    for p_new, fires in ((0.4 / 0.95 + 1e-6, True), (0.4 / 0.95 - 1e-6, False)):
        assert (R.effect_d(0.4, p_new) < -0.05) is fires
    w = World(tmp_path)
    w.set_old("rail01", R.OLD_A9, (1.0, 1.0))
    w.set_old("neos-2657525-crna", R.OLD_A9, (0.7518, 0.1704))
    w.set_p("neos-2657525-crna", R.A9, (1.0, 1.0))      # the worst possible: still no harm
    w.old_gens = {("neos-2657525-crna", 0): 855, ("neos-2657525-crna", 1): 299}
    w.summ[("neos-2657525-crna", R.A9, 0)] = dict(generations=12)
    rd = _first(w.report())
    h = rd["rules"]["harm"]
    assert h["cannot_fire"] == [i for i in DEV if i in ("rail01", "neos-2657525-crna")]
    assert "neos-2657525-crna" not in h["harm"]
    cf = {r["instance"]: r for r in h["cannot_fire_rows"]}
    assert cf["neos-2657525-crna"]["gens_old"] == [855, 299] and cf["neos-2657525-crna"]["gens_new"] == [12, 100]
    assert cf["neos-2657525-crna"]["band"] == pytest.approx(R.bounded(0.7518, 0.1704))
    assert rd["summary"]["harm_cannot_fire"] == h["cannot_fire"]
    text = R.render(w.report())
    assert "harm CANNOT fire on 2 of the 17" in text and "generations 855 / 299 -> 12 / 100" in text


def test_null_lines_and_harm_notes_are_beside(tmp_path):
    """The same d_i / band_i on pairs with no change — A6F@c1r vs A6F@r13x, and the same-configuration A9 replicates of
    10-15 (A9@r13x's run hash) vs A9@r13x — counted, printed, and named beside each harm; a replicate under another
    run hash is left out (stated). A harm with |ΔP| <= 0.005 carries D-022's floor marker. Nothing changes the list."""
    w = World(tmp_path)
    w.set_old("radiationm18-12-05", R.OLD_A9, (0.0109, 0.0111))
    w.set_p("radiationm18-12-05", R.A9, (0.0118, 0.0118))          # d -0.068 < -0.05 with ΔP +0.0008
    w.set_old("glass4", R.OLD_A9, (0.3, 0.3))
    w.set_p("glass4", R.A9, (0.4, 0.4))
    w.set_p("glass4", R.A6F, (0.5, 0.5))                           # cuOpt alone 0.4 -> 0.5 across the days: fires
    for s, v in ((0, 0.42), (1, 0.42)):                            # a same-configuration replicate as low as -0.286
        w.extra.append(dict(run_id=f"glass4__A9@r2ctl__s{s}__120s__{R.OLD_A9_HASH}", set="r2-ld", instance="glass4",
                            arm="A9@r2ctl", seed=s, budget=120.0, status="completed", attempt=0, pt=v,
                            synthetic=False, block_void=False, run_dir=""))
    w.extra.append(dict(run_id="sing44__A9@r1ctl__s0__120s__ffffffffffff", set="r1-bsync", instance="sing44",
                        arm="A9@r1ctl", seed=0, budget=120.0, status="completed", attempt=0, pt=0.9, synthetic=False,
                        block_void=False, run_dir=""))
    rd = _first(w.report())
    ru, s = rd["rules"], rd["summary"]
    assert ru["harm"]["harm"] == [i for i in DEV if i in ("glass4", "radiationm18-12-05")]
    nl = ru["nulls"]
    assert nl["cuopt"]["fired"] == ["glass4"] and nl["cuopt"]["n"] == 17
    assert [(x["label"], x["fired"], x["n"]) for x in nl["same_config"]] == [("A9@r2ctl", ["glass4"], 1)]
    assert (nl["k"], nl["n"]) == (1, 1) and s["null_rate"] == (1, 1)
    assert nl["skipped"] == ["sing44__A9@r1ctl__s0__120s__ffffffffffff"]
    assert any("D-022's floor" in x for x in s["harm_notes"]["radiationm18-12-05"])
    assert any("within the null range" in x for x in s["harm_notes"]["glass4"])
    assert any("no same-configuration replicate" in x for x in s["harm_notes"]["radiationm18-12-05"])
    text = R.render(w.report())
    assert "HARM (floor)" in text and "fire on 1 of 1 instance reads" in text
    assert "A9@r2ctl (r2-ld): glass4 -0.286 FIRES" in text and "fires on 1 of 17 ['glass4']" in text


def test_vs_scip_alone_on_scips_weak_set_among_all_17_is_beside(tmp_path):
    """D-019's C1 asks for the 17: SCIP's weak set among all of them (A0 P̄ >= 0.5) in D-013 terms, beside the
    fixed-11 line (D-022's reading); here A0 runs on all 17 (0.8 on 15, 0.2 on two) and A9 0.4 everywhere."""
    w = World(tmp_path)
    w.a0 = {(i, s): (0.2 if i in ("glass4", "radiationm18-12-05") else 0.8) for i in DEV for s in (0, 1)}
    ru = _first(w.report())["rules"]
    v = ru["vs_a0_17"]
    assert v["n_baseline"] == 17 and len(v["weak_set"]) == 15 and "glass4" not in v["weak_set"]
    assert v["block"]["n"] == 15 and v["block"]["better"] == 15 and v["block"]["mean_d"] == pytest.approx(0.5)
    assert len(ru["vs_a0"]["weak_set"]) == 11                     # the fixed-11 line unchanged
    assert "SCIP's weak set among all 17" in R.render(w.report())


def test_voids_and_frozen_v2_are_beside(tmp_path):
    """The set's void.jsonl is counted with its reasons (the pool reruns a voided block whole; the rerun is read);
    frozen-v2.yaml's state is printed (absent under a bare root); neither gates."""
    w = World(tmp_path)
    runs = w.build()
    vd = tmp_path / "traces" / "raw" / w.set
    (vd / "void.jsonl").write_text("\n".join(json.dumps(dict(run_id=w.exp[("sing44", lab, 1)], reason="split block "
                                                           "(resume)", moved_to="void/x")) for lab in (R.A9, R.A6F)))
    rep = R.report(runs, tmp_path, "ld0", w.set)
    assert rep["voids"]["n"] == 2 and rep["voids"]["reasons"] == {"split block": 2}
    assert rep["frozen_v2"]["state"] == "absent" and _first(rep)["validity"]["verdict"] == "VALID"
    text = R.render(rep)
    assert "voids in c1r-ld0/void.jsonl: 2 (split block 2;" in text and "frozen-v2.yaml not written yet" in text


# ============================================================================================ validity
@pytest.mark.parametrize("what, leg", [
    ("harness", "harness"), ("config", "config"), ("cuda_a6f", "compile"), ("cupy_a9", "compile"),
    ("ea_compiled", "compile"), ("window", "window"), ("window_missing", "window"), ("flag", "flag")])
def test_each_validity_leg_makes_it_invalid(tmp_path, what, leg):
    w = World(tmp_path)
    k9, k6 = ("sing44", R.A9, 1), ("sing44", R.A6F, 1)
    if what == "harness":
        w.rj[k6] = dict(config=dict(harness={}))
    elif what == "config":
        w.row[k9] = dict(run_id=w.exp[k9].rsplit("__", 1)[0] + "__28e5a5e9c725")     # the old default's hash
    elif what == "cuda_a6f":
        w.rj[k6] = dict(compiled_on_clock_cuda=True)
    elif what == "cupy_a9":
        w.rj[k9] = dict(cupy_cache=dict(compiled_on_clock=True))
    elif what == "ea_compiled":
        w.summ[k9] = dict(compiled_on_clock=True)
    elif what == "window":
        w.window[k9] = "W2"
    elif what == "flag":
        w.summ[k9] = dict(cuda_sched=dict(state="verified", sched_active="blocking_sync", sched_at_end="blocking_sync"))
    else:
        w.window[k6] = None
    rd = _first(w.report())
    assert rd["validity"]["verdict"] == "INVALID" and rd["validity"]["failing"] == [leg], rd["validity"]
    assert rd["rules"] is None and rd["summary"]["hint"] == "not read"
    assert "RULES NOT READ" in R.render(w.report())


def test_cupy_flag_is_never_charged_to_a6f_and_whole_blocks_may_split_across_windows(tmp_path):
    w = World(tmp_path)
    w.rj[("sing44", R.A6F, 0)] = dict(cupy_cache=dict(compiled_on_clock=True))
    for lab in (R.A9, R.A6F):                         # a whole block in a second window
        w.window[("glass4", lab, 1)] = "W2"
        w.window[("rail01", lab, 0)] = "W3"
    assert _first(w.report())["validity"]["verdict"] == "VALID"


def test_a_missing_run_is_incomplete_and_the_rules_read_the_rest(tmp_path):
    """glass4 is outside the fixed 11: the hint reads, the guard has no partial mean, the harm leg leaves glass4
    unread and reads the other 16."""
    w = World(tmp_path)
    w.drop.add(("glass4", R.A9, 1))
    rd = _first(w.report())
    assert rd["validity"]["verdict"] == "INCOMPLETE" and rd["validity"]["failing"] == ["complete"]
    ru = rd["rules"]
    assert ru["hint"]["state"] == "absent" and ru["hint"]["ratio"] == pytest.approx(1.0)      # read: all 0.4
    assert ru["guard"]["state"] == "INCOMPLETE" and ru["guard"]["missing"] == [("glass4", R.A9, 1)]
    assert ru["harm"]["unread"] == ["glass4"] and ru["harm"]["state"] == "INCOMPLETE"
    assert sum(1 for r in ru["harm"]["rows"] if r["read"]) == 16
    w.drop = {("sing44", R.A6F, 0)}
    ru = _first(w.report())["rules"]
    assert ru["hint"]["state"] == "INCOMPLETE" and ru["hint"]["missing"] == [("sing44", R.A6F, 0)]


def test_failed_runs_are_kept_and_the_letter_is_beside(tmp_path):
    """A failed A9 run (P truncated at the crash, 0.9) is an outcome of its arm in the reading that leads (keep);
    the letter (exclude) is printed beside: the run is missing, the hint on the 11 INCOMPLETE."""
    w = World(tmp_path)
    k = ("sing44", R.A9, 0)
    w.row[k] = dict(status="failed")
    w.p[k] = 0.9
    w.summ[k] = None                                  # died before its summary
    rep = w.report()
    assert rep["failed_runs"] == [w.exp[k]] and [rd["keep_failed"] for rd in rep["readings"]] == [True, False]
    keep, excl = rep["readings"]
    assert keep["validity"]["verdict"] == "VALID" and keep["validity"]["deviation"]
    assert keep["rules"]["hint"]["mean_P_a9"] == pytest.approx((10 * 0.4 + 0.65) / 11)
    assert keep["cells"][k]["flag"]["ok"] is None and "flag unread" in keep["cells"][k]["flag_note"]
    assert not keep["validity"]["legs"]["flag"]
    assert excl["validity"]["verdict"] == "INCOMPLETE" and excl["rules"]["hint"]["state"] == "INCOMPLETE"
    assert len(w.report("keep")["readings"]) == 1 and len(w.report("exclude")["readings"]) == 1


@pytest.mark.parametrize("a9", ["ld0", "ysync-ld0"])
def test_flag_state_gates(tmp_path, a9):
    """The flag leg (D-030's amendment of 2026-10-15, as D-025 / D-026 gate it): ld0 = CUDA's default read back,
    ysync-ld0 = yield verified and held; timing events never blocking. A wrong flag or a completed A9 run without an EA
    summary -> INVALID (flag); a failed A9 run without one is noted, not charged (test_failed_runs_…)."""
    w = World(tmp_path, a9=a9)
    rd = _first(w.report())
    assert rd["validity"]["verdict"] == "VALID" and all(c["flag"]["ok"] for k, c in rd["cells"].items() if k[1] == R.A9)
    wrong = dict(state="verified", sched_active="yield", sched_at_end="yield") if a9 == "ld0" else \
        dict(state="read", sched_active="auto")
    w.summ[("comp21-2idx", R.A9, 1)] = dict(cuda_sched=wrong)
    rd = _first(w.report())
    assert rd["validity"]["verdict"] == "INVALID" and rd["validity"]["failing"] == ["flag"]
    assert rd["rules"] is None and "comp21-2idx A9@c1r s1: CUDA flag not as written" in R.render(w.report())
    w.summ = {("glass4", R.A9, 0): dict(phase_events_block=True)}
    assert _first(w.report())["validity"]["failing"] == ["flag"]
    w = World(tmp_path / "nosumm", a9=a9)                       # a fresh tree: completed, no summary
    w.summ = {("glass4", R.A9, 0): None}
    rd = _first(w.report())
    assert rd["validity"]["failing"] == ["flag"] and "no ea.summary.json" in rd["validity"]["legs"]["flag"][0]
    if a9 == "ysync-ld0":
        w.summ = {("glass4", R.A9, 0): dict(cuda_sched=dict(state="verified", sched_active="yield",
                                                            sched_at_end="auto"))}
        assert _first(w.report())["validity"]["failing"] == ["flag"]          # not held to the end


def test_rows_of_the_set_no_rule_reads_are_listed(tmp_path):
    w = World(tmp_path)
    w.extra.append(dict(run_id="glass4__A9@other__s0__120s__x", set=w.set, instance="glass4", arm="A9@other", seed=0,
                        budget=120.0, status="completed", attempt=0, pt=0.1, synthetic=False, block_void=False,
                        run_dir=""))
    rep = w.report()
    assert rep["other_rows"] == ["glass4__A9@other__s0__120s__x"]
    assert _first(rep)["validity"]["verdict"] == "VALID"


# ============================================================================================ output and CLI
def test_render_main_json_and_cli(tmp_path, capsys):
    from itrea import cli
    w = World(tmp_path)
    for i in ELEVEN:
        w.set_p(i, R.A9, (0.3, 0.3))
    runs = tmp_path / "runs.parquet"
    w.build().to_parquet(runs)
    out = tmp_path / "c1r.json"
    assert R.main(["--a9", "ld0", "--runs", str(runs), "--root", str(tmp_path), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    for s in ("VALIDITY VALID: 68 of 68", "HINT present (this set)", "GUARD holds (this set)",
              "HARM (D-025 amendment 3, cross-day vs A9@r13x of 10-13): none on the 17",
              "recorded in D-022's Result: 0.827", "recorded in D-022's Result: +0.063",
              "go / no-go on C1 is Berk's", "reported, not criteria (D-013 terms) — vs cuOpt alone"):
        assert s in text, s
    js = json.loads(out.read_text())
    assert js["readings"][0]["summary"]["hint"] == "present" and len(js["readings"][0]["runs"]) == 68
    assert R.main(["--a9", "ld0", "--runs", str(tmp_path / "none.parquet")]) == 2
    assert callable(cli.resolve_main("c1r-report")) and "c1r-report" in cli.COMMANDS
