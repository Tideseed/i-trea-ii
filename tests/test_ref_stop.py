"""D-021 phase 2 in the runner: R12's stop at the proven optimum (`ref_stop`) and cuOpt's immediate SIGKILL after a
definitive outcome (`cuopt_kill_on_end`), both optional harness switches that leave every earlier harness — and run id —
as it was.

The runs use a FAKE role process that writes real incumbent vectors of a toy model in the solver processes' crash-safe
format (`<role>.names.json` + `<role>.incumbents.part`, then the `.npz` at its end), so the runner's monitor reads
them exactly as it reads SCIP's, the EA's and cuOpt's; the toy's optimum is known by enumeration and handed to the
runner through `PairOptions.ref_lookup` (in production: MIPLIB's =opt= lines only)."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from itrea.benchmarks.toy import knapsack, set_cover, write_mps
from itrea.experiments import pair

FAKE_REC = r'''
"""Fake role process writing real incumbent vectors (solvers' crash-safe format); behaviour from FAKE_PLAN_<ROLE>."""
import argparse, json, os, struct, sys, time
import numpy as np
from itrea.exchange.redis_mailbox import open_mailbox
from itrea.trace import Trace, save_incumbents
ap = argparse.ArgumentParser()
for a in ("--instance", "--seed", "--budget", "--run-dir", "--mailbox", "--sol-out", "--arm", "--mem-mb",
          "--role-name", "--config", "--cpu-threads", "--inject-band", "--inject-k", "--emphasis", "--run-id",
          "--gpu-mem-gb"):
    ap.add_argument(a)
for f in ("--recv", "--send", "--inert", "--no-barrier", "--mip-gap-zero"):
    ap.add_argument(f, action="store_true")
a, _ = ap.parse_known_args()
role = a.role_name or os.environ.get("ITREA_ROLE", "solver")
plan = json.loads(os.environ.get("FAKE_PLAN_" + role.upper(), "{}"))
run_dir = a.run_dir
mb = open_mailbox(a.mailbox, role, run_id=a.run_id)
tr = Trace(run_dir, role, arm=a.arm, seed=int(a.seed))
names = plan.get("names", [])
c = np.array(plan.get("c", []), dtype=float)
with open(os.path.join(run_dir, role + ".names.json"), "w") as fh:
    json.dump(names, fh)
part = open(os.path.join(run_dir, role + ".incumbents.part"), "ab", buffering=0)
mb.ready()
try:
    t0 = mb.wait_start(timeout=60)
except RuntimeError:
    sys.exit(0)
tr.start(t0)
ignore = bool(plan.get("ignore_stop"))
budget = float(a.budget)
stopped = lambda: (not ignore) and mb.stopped()
recs = []
for t_rec, x in plan.get("recs", []):
    while time.monotonic() - t0 < t_rec and not stopped():
        time.sleep(0.02)
    if stopped():
        break
    x = np.asarray(x, dtype=float)
    obj = float(c @ x + plan.get("offset", 0.0))
    t = time.monotonic() - t0
    tr.incumbent(obj, t=t)
    nz = np.flatnonzero(x)
    part.write(struct.pack("<ddBi", t, obj, 0, nz.size) + nz.astype("<i4").tobytes() + x[nz].astype("<f8").tobytes())
    recs.append((t, obj, "self", x))
end_after = plan.get("end_after")
while time.monotonic() - t0 < budget and not stopped():
    if end_after is not None and time.monotonic() - t0 >= end_after:
        break
    time.sleep(0.02)
part.close()
save_incumbents(os.path.join(run_dir, role + ".incumbents.npz"), names, recs)
for side in (role + ".incumbents.part", role + ".names.json"):
    os.unlink(os.path.join(run_dir, side))
if plan.get("exit_code"):
    tr.close(exit_reason="exception")
    sys.exit(int(plan["exit_code"]))
tr.close(exit_reason=plan.get("exit_reason") or ("stop" if mb.stopped() else "budget"))
sys.exit(0)
'''


def _best(prob):
    """The optimum of a small binary toy by enumeration: (x, objective)."""
    best, bx = None, None
    A, rl, ru = prob.A.toarray(), prob.rl, prob.ru
    for bits in itertools.product((0.0, 1.0), repeat=prob.n):
        x = np.array(bits)
        act = A @ x
        if np.all(act >= rl - 1e-9) and np.all(act <= ru + 1e-9):
            v = float(prob.c @ x + prob.offset)
            if best is None or prob.sense * v < prob.sense * best:
                best, bx = v, x
    return bx, best


@pytest.fixture(scope="module")
def ks(tmp_path_factory):
    """A 12-item knapsack (MAXIMISATION, so the sense is exercised), its optimum, a worse feasible point and an
    infeasible point that beats z*."""
    prob = knapsack(12, seed=3)
    path = write_mps(prob, tmp_path_factory.mktemp("ks") / "ks12.mps")
    x_opt, z = _best(prob)
    assert z == pytest.approx(prob.toy_optimum)
    x_poor = np.zeros(prob.n)
    x_poor[int(np.argmax(prob.c))] = 1.0                       # one item: feasible, far from z*
    x_bad = np.ones(prob.n)                                    # every item: value > z*, weight row violated
    return dict(prob=prob, path=path, x_opt=x_opt, z=z, x_poor=x_poor, x_bad=x_bad)


@pytest.fixture
def fake(tmp_path) -> list[str]:
    p = tmp_path / "fake_rec.py"
    p.write_text(FAKE_REC)
    return [sys.executable, str(p)]


def _opts(tmp_path, fake, z, **kw) -> pair.PairOptions:
    d = dict(traces_dir=tmp_path / "traces", mailbox_root=tmp_path / "mb", use_systemd=False, pin=False, grace_s=2.0,
             kill_after_s=10.0, gpu_health=False, record_snapshots=False, solver_mem_gb=2.0, ea_mem_gb=2.0,
             cuopt_mem_gb=3.0, cupy_cache_dir=tmp_path / "cupy", cuda_cache_dir=tmp_path / "cuda",
             module_override={k: fake for k in ("scip", "highs", "cuopt", "ea")}, log=lambda m: None,
             ref_lookup=(lambda job: (pair.StopRef(z, "opt", "test: enumeration"), "")) if z is not None
             else (lambda job: (None, "test: no proven optimum")), ref_poll_s=0.1)
    d.update(kw)
    return pair.PairOptions(**d)


def _job(ks, arm="A0", harness="itx1", budget=20.0, **kw) -> dict:
    j = dict(instance=str(ks["path"]), set="pytest", arm=arm, seed=0, budget=budget, split="smoke", **kw)
    if harness is not None:
        j["harness"] = harness
    return j


def _plan(ks, recs, reverse=False, **kw) -> str:
    """FAKE_PLAN: the process's own variable order (reversed: the runner must map by name) and its records."""
    p = ks["prob"]
    names, c = list(p.names), np.asarray(p.c, float)
    perm = np.arange(p.n)[::-1] if reverse else np.arange(p.n)
    return json.dumps(dict(names=[names[i] for i in perm], c=c[perm].tolist(), offset=float(p.offset),
                           recs=[[t, np.asarray(x, float)[perm].tolist()] for t, x in recs], **kw))


# --------------------------------------------------------------------------------------------- unit level
def test_ref_reached_rule():
    """GAMS's rounded gamma <= 0 AND Berthold's gap <= 1e-9: z* = 0 or a gap below GAMS's rounding never stops; an
    objective equal to or better than z* (either sense) does."""
    assert pair.ref_reached(100.0, 100.0)[0] and pair.ref_reached(99.0, 100.0)[0]        # equal, better (min)
    assert pair.ref_reached(101.0, 100.0, sense=-1)[0] and not pair.ref_reached(99.0, 100.0, sense=-1)[0]
    ok, g, b = pair.ref_reached(1e-7, 0.0)                     # GAMS: round(1e-7 / 1, 6) = 0, Berthold: 1
    assert (ok, g, b) == (False, 0.0, 1.0)
    ok, g, b = pair.ref_reached(1e6 * (1 + 4e-7), 1e6)         # gamma rounds to 0, gap 4e-7 > 1e-9
    assert not ok and g == 0.0 and b == pytest.approx(4e-7, rel=1e-6)
    assert pair.ref_reached(1e6 * (1 + 5e-10), 1e6)[0]         # within Berthold's 1e-9
    assert not pair.ref_reached(-5.0, 5.0, sense=-1)[0]        # opposite signs, worse: gamma 1


def test_miplib_stop_ref(tmp_path):
    """=opt= lines only: supportcase22's =best= and any instance outside MIPLIB are never eligible."""
    solu = tmp_path / "solu.txt"
    solu.write_text("=opt=  glass4  1200012600\n=best=  supportcase22  110\n=inf=  x\n")
    ref, why = pair.miplib_stop_ref(dict(instance="data/miplib/inst/glass4.mps.gz"), solu)
    assert ref == pair.StopRef(1200012600.0, "opt", "MIPLIB 2017 solu (=opt=)") and why == ""
    ref, why = pair.miplib_stop_ref(dict(instance="data/miplib/inst/supportcase22.mps.gz"), solu)
    assert ref is None and "best_known" in why
    ref, why = pair.miplib_stop_ref(dict(instance="data/pglib_uc/mps/ca__x.mps.gz"), solu)
    assert ref is None and "not a MIPLIB" in why
    assert pair.miplib_stop_ref(dict(instance="data/miplib/inst/nothere.mps.gz"), solu)[0] is None


def test_optional_harness_switches_keep_every_earlier_harness():
    """ref_stop / cuopt_kill_on_end are absent unless on: ext, and ext with them off, resolve and hash as before;
    itx1 = ext + both; a wrong type is refused."""
    ext = pair.resolve_harness("ext")
    assert set(ext) == set(pair.HARNESS_KEYS)
    assert pair.resolve_harness({"preset": "ext", "ref_stop": False, "cuopt_kill_on_end": False}) == ext
    itx = pair.resolve_harness("itx1")
    assert itx == dict(ext, ref_stop=True, cuopt_kill_on_end=True)
    assert pair.resolve_harness({"preset": "ext", "ref_stop": True, "cuopt_kill_on_end": True}) == itx
    assert pair.resolve_harness({"preset": "itx1", "ref_stop": False}) == dict(ext, cuopt_kill_on_end=True)
    with pytest.raises(pair.RunError, match="true or false"):
        pair.resolve_harness({"preset": "ext", "ref_stop": 1})
    j = dict(instance="data/miplib/inst/glass4.mps.gz", set="x", arm="A6F", seed=0, budget=600)
    assert pair.run_hash(dict(j, harness={"preset": "ext", "ref_stop": False})) == pair.run_hash(dict(j, harness="ext")) \
        == "82a371305f88"                                      # the M2 / M4 id (test_pair pins the rest)
    a9 = dict(j, arm="A9", config="configs/itrea2-dev-rc-3way.yaml")
    assert pair.run_hash(dict(a9, harness="ext")) == "28e5a5e9c725"
    assert len({pair.run_hash(dict(j, harness=h)) for h in ("ext", "itx1", {"preset": "ext", "ref_stop": True})}) == 3
    assert pair.run_hash(dict(a9, harness="itx1")) not in ("28e5a5e9c725", "0f011a2094a2")


# --------------------------------------------------------------------------------------------- runner level
def test_ref_reached_ends_the_run_and_is_confirmed(ks, tmp_path, fake, monkeypatch):
    """The fake SCIP records a poor point, then the optimum (in its own, reversed variable order): the runner stops
    the run at the optimum (stop kind ref_reached, long before the budget), the exact checker confirms it, nothing is a
    crash, the run is completed; the stop never reaches the process as z*."""
    monkeypatch.setenv("FAKE_PLAN_SOLVER", _plan(ks, [(0.1, ks["x_poor"]), (0.4, ks["x_opt"]), (6.0, ks["x_poor"])],
                                                 reverse=True))
    o = _opts(tmp_path, fake, ks["z"])
    rec = pair.run_pair(_job(ks), o)
    assert rec["status"] == "completed", rec["reasons"]
    assert rec["stop"]["kind"] == "ref_reached" and rec["wall_s"] < 5.0
    rs = rec["ref_stop"]
    assert rs["enabled"] and rs["z"] == ks["z"] and rs["sense"] == -1
    tr = rs["trigger"]
    assert (tr["role"], tr["k"]) == ("solver", 1) and tr["obj"] == pytest.approx(ks["z"]) and tr["gamma"] == 0.0
    assert 0.3 < tr["t"] < 1.5 and tr["detected_t"] >= tr["t"]
    assert rs["check"]["ok"] and rs["check"]["obj_recomputed"] == pytest.approx(ks["z"]) and rs["check"]["reader"]
    assert rs["roles"]["solver"]["records"] == 2 and rs["roles"]["solver"]["error"] is None
    assert all(p["exit_code"] == 0 and not p["killed"] for p in rec["processes"])
    assert "void_cause" not in rec or rec.get("void_cause") is None
    assert all(str(ks["z"]) not in " ".join(p["cmd"]) for p in rec["processes"])        # z* never on a command line


def test_cuopt_is_killed_at_once_and_its_records_still_count(ks, tmp_path, fake, monkeypatch):
    """A6F: cuOpt ignores the stop (no interrupt API). Under itx1 the runner SIGKILLs it as soon as its own record
    reaches z* (stopped_by_runner, not a crash); its .part file — read through the descriptor the runner kept, then
    recovered into the .npz — confirms the stop. Without cuopt_kill_on_end it would run on until kill_after_s."""
    monkeypatch.setenv("FAKE_PLAN_CUOPT", _plan(ks, [(0.1, ks["x_poor"]), (0.3, ks["x_opt"])], ignore_stop=True))
    o = _opts(tmp_path, fake, ks["z"], kill_after_s=30.0)
    rec = pair.run_pair(_job(ks, "A6F"), o)
    assert rec["status"] == "completed", rec["reasons"]
    (c,) = rec["processes"]
    assert c["killed"] == "SIGKILL (definitive outcome)" and c["stopped_by_runner"] and c["exit_code"] == -9
    assert rec["wall_s"] < 5.0 and rec["ref_stop"]["check"]["ok"]
    rd = Path(o.traces_dir) / "raw" / "pytest" / rec["run_id"]
    assert (rd / "cuopt.incumbents.npz").exists()                                     # recovered from the .part
    from itrea.analysis.build_runs import crash_flags, load_run
    assert crash_flags(load_run(rd))["crashed"] is False


def test_an_unconfirmed_stop_voids_the_run(ks, tmp_path, fake, monkeypatch):
    """A vector that beats z* but violates the weight row triggers the stop; the exact checker rejects it, so the run
    is void (ref_rejected) — the pool reruns its block with the stop off."""
    monkeypatch.setenv("FAKE_PLAN_SOLVER", _plan(ks, [(0.1, ks["x_bad"])]))
    rec = pair.run_pair(_job(ks), _opts(tmp_path, fake, ks["z"]))
    assert rec["stop"]["kind"] == "ref_reached"
    assert rec["status"] == "void" and rec["void_cause"] == "ref_rejected"
    assert "checker rejects" in rec["ref_stop"]["check"]["reason"] and not rec["ref_stop"]["check"]["feasible"]


def test_stop_off_ineligible_and_unmapped_roles_change_nothing(ks, tmp_path, fake, monkeypatch):
    """The pool's ref_stop_off (same run id), an instance without a proven optimum and a process whose names the
    runner cannot map all leave the run to its budget, recorded; a harness without ref_stop records None."""
    monkeypatch.setenv("FAKE_PLAN_SOLVER", _plan(ks, [(0.1, ks["x_opt"])]))
    j = _job(ks, budget=1.0)
    off = dict(j, ref_stop_off="rerun after ref_rejected in x")
    assert pair.run_id_for(off) == pair.run_id_for(j)
    rec = pair.run_pair(off, _opts(tmp_path / "a", fake, ks["z"]))
    assert rec["status"] == "completed" and (rec["stop"] or {}).get("kind") != "ref_reached"
    assert rec["ref_stop"] == dict(enabled=False, reason="off for this attempt: rerun after ref_rejected in x")
    rec = pair.run_pair(j, _opts(tmp_path / "b", fake, None))
    assert not rec["ref_stop"]["enabled"] and rec["ref_stop"]["reason"].startswith("not eligible: test")
    rec = pair.run_pair(_job(ks, harness="ext", budget=1.0), _opts(tmp_path / "c", fake, ks["z"]))
    assert rec["ref_stop"] is None and (rec["stop"] or {}).get("kind") != "ref_reached"
    plan = json.loads(_plan(ks, [(0.1, ks["x_opt"])]))
    plan["names"][0] = "not_a_variable"
    monkeypatch.setenv("FAKE_PLAN_SOLVER", json.dumps(plan))
    rec = pair.run_pair(j, _opts(tmp_path / "d", fake, ks["z"]))
    assert rec["status"] == "completed" and (rec["stop"] or {}).get("kind") != "ref_reached"
    assert "unknown to the runner" in rec["ref_stop"]["roles"]["solver"]["error"] and rec["ref_stop"]["trigger"] is None


def test_a_record_after_the_budget_does_not_stop(ks, tmp_path, fake, monkeypatch):
    """Only records stamped before T count (nothing after T enters a metric)."""
    monkeypatch.setenv("FAKE_PLAN_SOLVER", _plan(ks, [(0.1, ks["x_poor"]), (1.2, ks["x_opt"])]))
    rec = pair.run_pair(_job(ks, budget=1.0), _opts(tmp_path, fake, ks["z"], grace_s=2.0))
    assert rec["ref_stop"]["trigger"] is None and (rec["stop"] or {}).get("kind") != "ref_reached"


@pytest.mark.parametrize("switch", [False, True])
def test_partner_proof_kills_cuopt_only_under_the_switch(ks, tmp_path, fake, monkeypatch, switch):
    """A2F: SCIP proves optimality (exit 0, exit reason optimal) while cuOpt ignores the stop. With
    cuopt_kill_on_end the runner SIGKILLs cuOpt at once; without it (every earlier harness) cuOpt runs on until
    kill_after_s, then SIGTERM — the old behaviour."""
    monkeypatch.setenv("FAKE_PLAN_SOLVER", _plan(ks, [(0.1, ks["x_poor"])], end_after=0.4, exit_reason="optimal"))
    monkeypatch.setenv("FAKE_PLAN_CUOPT", _plan(ks, [(0.1, ks["x_poor"])], ignore_stop=True))
    hz = {"preset": "ext", "cuopt_kill_on_end": True} if switch else "ext"
    rec = pair.run_pair(_job(ks, "A2F", harness=hz), _opts(tmp_path, fake, ks["z"], kill_after_s=2.5))
    assert rec["stop"]["kind"] == "partner_end" and rec["stop"]["proven"] is True
    cu = next(p for p in rec["processes"] if p["role"] == "cuopt")
    assert cu["stopped_by_runner"] and rec["status"] == "completed"
    if switch:
        assert cu["killed"] == "SIGKILL (definitive outcome)" and cu["exit_t"] < 2.0
    else:
        assert cu["killed"] in ("SIGTERM after stop", "SIGKILL") and cu["exit_t"] > 2.5


def test_a_partner_crash_is_not_definitive(ks, tmp_path, fake, monkeypatch):
    """A crash of SCIP (exit 1) ends the race but decides nothing: cuOpt is not killed at once."""
    monkeypatch.setenv("FAKE_PLAN_SOLVER", _plan(ks, [], end_after=0.3, exit_code=1))
    monkeypatch.setenv("FAKE_PLAN_CUOPT", _plan(ks, [(0.1, ks["x_poor"])], ignore_stop=True))
    rec = pair.run_pair(_job(ks, "A2F"), _opts(tmp_path, fake, ks["z"], kill_after_s=1.5))
    assert rec["stop"]["kind"] == "partner_end" and not rec["stop"]["proven"]
    assert not pair.definitive_stop(rec["stop"]) and pair.definitive_stop(dict(kind="ref_reached"))
    cu = next(p for p in rec["processes"] if p["role"] == "cuopt")
    assert cu["killed"] != "SIGKILL (definitive outcome)" and rec["status"] == "failed"


def test_minimisation_and_three_roles(tmp_path, fake, monkeypatch):
    """A9's three processes on a set-cover toy (minimisation): the EA's record reaches z* first; the stop is credited
    to it and confirmed with the EA's reader (SCIP's parse)."""
    prob = set_cover(10, 6, seed=2)
    path = write_mps(prob, tmp_path / "sc.mps")
    x_opt, z = _best(prob)
    tiny = dict(path=path, prob=prob)
    worse = np.ones(prob.n)
    monkeypatch.setenv("FAKE_PLAN_SOLVER", _plan(tiny, [(0.1, worse)]))
    monkeypatch.setenv("FAKE_PLAN_EA", _plan(tiny, [(0.3, x_opt)], reverse=True))
    monkeypatch.setenv("FAKE_PLAN_CUOPT", _plan(tiny, [(0.2, worse)], ignore_stop=True))
    j = dict(instance=str(path), set="pytest", arm="A9", seed=0, budget=20.0, split="smoke", harness="itx1",
             config="configs/itrea2-dev-rc-3way.yaml")
    rec = pair.run_pair(j, _opts(tmp_path, fake, z, kill_after_s=20.0))
    assert rec["status"] == "completed", rec["reasons"]
    tr, chk = rec["ref_stop"]["trigger"], rec["ref_stop"]["check"]
    assert tr["role"] == "ea" and chk["ok"] and chk["producer"] == "ea" and chk["reader"] == "scip"
    assert rec["wall_s"] < 5.0
    cu = next(p for p in rec["processes"] if p["role"] == "cuopt")
    assert cu["killed"] == "SIGKILL (definitive outcome)" and cu["stopped_by_runner"]


def test_build_runs_reads_a_ref_reached_run_like_a_run_to_T(ks, tmp_path, fake, monkeypatch):
    """The runner's own output of an R12 stop (A6F, cuOpt SIGKILLed after its record reached z*) through build_runs:
    completed, no crash, stop_kind ref_reached, the stop confirmed, and the last checked incumbent held to T in P and
    pt_gams (gamma 0 from the stop to T)."""
    from itrea.analysis import build_runs as B
    from itrea.analysis import metrics as M
    monkeypatch.setenv("FAKE_PLAN_CUOPT", _plan(ks, [(0.1, ks["x_poor"]), (0.3, ks["x_opt"])], ignore_stop=True))
    o = _opts(tmp_path, fake, ks["z"], kill_after_s=30.0)
    rec = pair.run_pair(_job(ks, "A6F", budget=20.0), o)
    df, inc = B.build(Path(o.traces_dir) / "raw", tmp_path / "ds", log=lambda *a: None)
    (r,) = df.to_dict("records")
    assert (r["status"], r["crashed"], r["stop_kind"], r["ref_stop_check"]) == ("completed", False, "ref_reached",
                                                                                 "confirmed")
    assert r["ref"] == pytest.approx(ks["z"]) and r["n_inc_ok"] == 2 and r["sense"] == -1
    ok = inc[inc["ok"]].sort_values("t")
    steps = M.best_steps(zip(ok["t"], ok["obj_recomputed"]), -1)
    assert steps[-1][1] == pytest.approx(ks["z"]) and steps[-1][0] < 1.5
    assert r["pt_gams"] == pytest.approx(M.gams_integral(steps, r["ref"], -1, 20.0).value, abs=1e-15)
    assert r["final_gap"] == 0.0 and r["t_gap1_event"] and rec["wall_s"] < 5.0


def test_monitor_reads_partial_files_across_polls(tmp_path):
    """RefMonitor on its own: a half-written names.json (the EA writes it at t0) is read again later, not an error; a
    record split across two polls is parsed once complete; records at or after t_max never trigger; the trigger is the
    first record that meets the rule, in the process's own variable order mapped by name."""
    import struct
    names = ["a", "b", "c"]
    c = np.array([1.0, 2.0, 3.0])
    ref = pair.StopRef(5.0, "opt", "test")
    mon = pair.RefMonitor(tmp_path, ("ea",), names, c, 0.0, 1, ref, t_max=10.0)
    (tmp_path / "ea.names.json").write_text('["c", "b"')                 # half written, no .part yet
    assert mon.poll() is None and mon.st["ea"]["error"] is None
    (tmp_path / "ea.names.json").write_text(json.dumps(["c", "b", "a"]))

    def rec(t, x):                                                        # x in the process's order (c, b, a)
        x = np.asarray(x, float)
        nz = np.flatnonzero(x)
        obj = float(np.dot(c[[2, 1, 0]], x))
        return struct.pack("<ddBi", t, obj, 1, nz.size) + nz.astype("<i4").tobytes() + x[nz].astype("<f8").tobytes()
    part = open(tmp_path / "ea.incumbents.part", "ab", buffering=0)
    r1, r2, r3 = rec(1.0, [0, 0, 7]), rec(11.0, [1, 1, 0]), rec(12.0, [1, 1, 0])   # 7; 5 but after t_max
    r4 = rec(3.0, [1, 1, 0])                                                       # c + b = 3 + 2 = 5: z*
    part.write(r1 + r2[:10])
    assert mon.poll() is None and mon.st["ea"]["k"] == 1
    part.write(r2[10:] + r3)
    assert mon.poll() is None and mon.st["ea"]["k"] == 3                 # 11 s and 12 s: after t_max, no trigger
    part.write(r4)
    tr = mon.poll()
    assert tr is not None and (tr["role"], tr["k"], tr["t"], tr["obj"]) == ("ea", 3, 3.0, 5.0)
    part.close()
    (tmp_path / "ea.incumbents.part").unlink()                             # the process removed it: nothing breaks
    assert mon.poll() is tr and mon.summary()["roles"]["ea"]["records"] == 4
    mon.close()
