"""SCIP solver process (`itrea.solvers.scip_run`) on toy models: the exchange round trip through the file mailbox and
through Redis, --send (incumbents, pool, info, root LP in the canonical order), --inert overhead against no plugin,
the stop file, bad arguments, and the shared helpers in `itrea.solvers.common`.

The mock EA is deterministic: the test (acting as the runner) posts the EA's message BEFORE releasing the barrier, so
SCIP's first plugin call (BEFOREPRESOL) finds it — a toy solves in about a second, a racing mock would be flaky.
"""
from __future__ import annotations

import gzip
import json
import os
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from itrea.benchmarks import toy
from itrea.exchange.mailbox import Mailbox
from itrea.problem import load
from itrea.solvers.common import (ORIGIN_EA_ECHO, ORIGIN_SOLVER, SOURCE_CODES, band_select, recover_incumbents,
                                  vector_hash)
from itrea.trace import load_incumbents, read_events

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MODULE = "itrea.solvers.scip_run"


# ------------------------------------------------------------------------------------------------------ helpers
def _env():
    return dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))


def _launch(module, args, mailbox=None, role="solver", timeout=60.0, stop_after=None):
    """Start a process, release the barrier once it is ready (if a mailbox is given), wait for its exit code."""
    pr = subprocess.Popen([PY, "-m", module, *args], env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        if mailbox is not None and "--no-barrier" not in args:
            end = time.monotonic() + timeout
            while not mailbox.all_ready(roles=(role,)):
                if pr.poll() is not None:
                    break
                assert time.monotonic() < end, "process never reported ready"
                time.sleep(0.005)
            else:
                mailbox.release()
        if stop_after is not None:
            time.sleep(stop_after)
            mailbox.stop("test")
        out, _ = pr.communicate(timeout=timeout)
    finally:
        if pr.poll() is None:
            pr.kill()
    return pr.returncode, out.decode(errors="replace")


def _summary(run_dir, role="solver"):
    return json.loads((Path(run_dir) / f"{role}.summary.json").read_text())


def _scip_optimum(path):
    """Optimal vector of a toy in the CANONICAL (HiGHS) order, solved in-process by SCIP (1 thread, 20 s)."""
    from pyscipopt import Model
    p = load(path)
    m = Model()
    m.hideOutput()
    m.setParam("parallel/maxnthreads", 1)
    m.setParam("limits/time", 20.0)
    m.readProblem(str(path))
    m.optimize()
    assert m.getStatus() == "optimal"
    sol = m.getBestSol()
    vals = {v.name: m.getSolVal(sol, v) for v in m.getVars(transformed=False)}
    x = p.round_integers(np.array([vals[nm] for nm in p.names]))
    assert p.check(x).feasible
    return p, x


def _infeasible_variant(p, x):
    """x with one chosen set removed so that some row is uncovered (SCIP must reject it)."""
    for j in np.flatnonzero(x > 0.5):
        y = x.copy()
        y[j] = 0.0
        if not p.check(y).feasible:
            return y
    raise AssertionError("no infeasible variant found")


@pytest.fixture(scope="module")
def cover(tmp_path_factory):
    d = tmp_path_factory.mktemp("scip_toys")
    path = toy.write_mps(toy.set_cover(400, 250, 0.03, seed=2), d / "cover.mps")
    p, x = _scip_optimum(path)
    return path, p, x


def _new_mailbox(tmp_path, names, run_id):
    mb = Mailbox(tmp_path / "mb", "runner", create=True, run_id=run_id)
    mb.write_order(names)
    return mb


def _check_round_trip(mb, run_dir, p, x_opt, rc, out):
    assert rc == 0, out
    s = _summary(run_dir)
    assert s["exit_reason"] == "optimal", s
    # verdicts: the optimum accepted and stored, the infeasible variant rejected by SCIP's own check
    v = mb.verdicts()
    assert len(v) == 2, v
    by_row = {json.loads(r["note"])["row"]: r for r in v}
    good, bad = by_row[0], by_row[1]
    assert good["accepted"] is True and json.loads(good["note"])["stored"] is True
    assert good["obj"] == pytest.approx(p.objective(x_opt))
    assert json.loads(good["note"])["primal_after"] == pytest.approx(p.objective(x_opt))
    assert bad["accepted"] is False
    nb = json.loads(bad["note"])
    assert nb["stored"] is False and nb["feasible"] is False
    # the EA's solution became SCIP's incumbent, labelled as such, with its vector stored
    ev = read_events(Path(run_dir) / "solver.events.jsonl")
    inc = [e for e in ev if e["kind"] == "incumbent"]
    assert inc and inc[0]["source"] == "ea" and inc[0]["obj"] == pytest.approx(p.objective(x_opt))
    assert sum(1 for e in ev if e["kind"] == "verdict") == 2
    names, recs = load_incumbents(Path(run_dir) / "solver.incumbents.npz")
    pos = p.name_index()
    xs = np.zeros(p.n)
    xs[[pos[nm] for nm in names]] = recs[0][3]
    assert recs[0][2] == "ea" and np.allclose(xs, x_opt)
    sol = json.loads(gzip.open(Path(run_dir) / "solver.sol.json.gz", "rt").read())
    assert sum(sol.values()) == pytest.approx(x_opt.sum())
    # echo marking: SCIP's incumbent is the EA's own vector, so it goes back with ORIGIN_EA_ECHO
    msg = mb.to_ea.poll()
    assert msg is not None and msg.origin is not None
    hashes = {vector_hash(x_opt)}
    for row, org in zip(msg.X, msg.origin):
        assert org == (ORIGIN_EA_ECHO if vector_hash(row) in hashes else ORIGIN_SOLVER)
    assert ORIGIN_EA_ECHO in set(msg.origin.tolist())
    assert s["exchange"]["rows_tried"] == 2 and s["exchange"]["rows_stored"] == 1
    assert s["exchange"]["rows_rejected"] == 1 and s["exchange"]["echo_rows_out"] >= 1
    assert not (Path(run_dir) / "solver.incumbents.part").exists()


# -------------------------------------------------------------------------------------------------------- tests
@pytest.mark.solver
def test_scip_round_trip_file_mailbox(tmp_path, cover):
    path, p, x_opt = cover
    x_bad = _infeasible_variant(p, x_opt)
    mb = _new_mailbox(tmp_path, p.names, "rt-file")
    mb.to_solver.post(np.vstack([x_opt, x_bad]), [p.objective(x_opt), p.objective(x_bad)])
    run = tmp_path / "run"
    rc, out = _launch(MODULE, ["--instance", str(path), "--seed", "3", "--budget", "20", "--run-dir", str(run),
                               "--mailbox", str(tmp_path / "mb"), "--run-id", "rt-file", "--recv", "--send",
                               "--mem-mb", "1000"], mb)
    _check_round_trip(mb, run, p, x_opt, rc, out)


@pytest.mark.solver
def test_scip_credits_a_cuopt_row_to_cuopt(tmp_path, cover):
    """Arm A9 (D-015): an EA row that is an unchanged cuOpt vector carries POST_CUOPT in its origin; the incumbent SCIP
    makes from it is recorded with source "cuopt" (not "ea") and its verdict note says so, while a feasible row
    without the bit, tried just before it in the same message, keeps the "ea" label (the A3 path); the infeasible
    unmarked row is rejected without the note."""
    from itrea.ea.population import ORIGIN_INJECTED
    from itrea.solvers.common import POST_CUOPT
    path, p, x_opt = cover
    x_bad = _infeasible_variant(p, x_opt)
    x_worse = x_opt.copy()
    x_worse[np.flatnonzero(x_opt < 0.5)[0]] = 1.0                       # one more set: feasible, costlier
    assert p.check(x_worse).feasible and p.objective(x_worse) > p.objective(x_opt)
    mb = _new_mailbox(tmp_path, p.names, "rt-cuopt")
    mb.to_solver.post(np.vstack([x_worse, x_opt, x_bad]),
                      [p.objective(x_worse), p.objective(x_opt), p.objective(x_bad)],
                      origin=np.array([ORIGIN_INJECTED, ORIGIN_INJECTED | POST_CUOPT, 0], dtype=np.int64))
    run = tmp_path / "run"
    rc, out = _launch(MODULE, ["--instance", str(path), "--seed", "3", "--budget", "20", "--run-dir", str(run),
                               "--mailbox", str(tmp_path / "mb"), "--run-id", "rt-cuopt", "--recv", "--send",
                               "--mem-mb", "1000"], mb)
    assert rc == 0, out
    ev = read_events(Path(run) / "solver.events.jsonl")
    inc = [e for e in ev if e["kind"] == "incumbent"]
    # one trySol may record two incumbents (BESTSOLFOUND twice), so positions, not counts, are checked
    i_ea = [i for i, e in enumerate(inc) if e["source"] == "ea"]
    i_cu = [i for i, e in enumerate(inc) if e["source"] == "cuopt"]
    assert i_ea and i_cu and max(i_ea) < min(i_cu)
    assert all(inc[i]["obj"] == pytest.approx(p.objective(x_worse)) for i in i_ea)
    assert all(inc[i]["obj"] == pytest.approx(p.objective(x_opt)) for i in i_cu)
    names, recs = load_incumbents(Path(run) / "solver.incumbents.npz")
    assert [r[2] for r in recs if r[2] != "self"] == [inc[i]["source"] for i in sorted(i_ea + i_cu)]
    ver = {e["row"]: e for e in ev if e["kind"] == "verdict"}
    assert ver[0]["accepted"] and "cuopt" not in ver[0]
    assert ver[1]["accepted"] and ver[1]["cuopt"] is True
    assert not ver[2]["accepted"] and "cuopt" not in ver[2]
    assert _summary(run)["exchange"]["rows_in_cuopt"] == 1
    assert SOURCE_CODES["cuopt"] == 3                    # the crash-safe .part record has a code for it


def _redis_or_skip():
    try:
        import redis
        r = redis.Redis(host="127.0.0.1", port=6390, socket_timeout=2)
        r.ping()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"itrea-redis on 127.0.0.1:6390 not reachable: {e}")


@pytest.mark.solver
def test_scip_round_trip_redis_mailbox(tmp_path, cover):
    _redis_or_skip()
    from itrea.exchange.redis_mailbox import RedisMailbox
    path, p, x_opt = cover
    x_bad = _infeasible_variant(p, x_opt)
    run_id = f"pytest-scip-{uuid.uuid4().hex[:10]}"
    url = f"redis://127.0.0.1:6390/0?run={run_id}"
    mb = RedisMailbox(url, run_id=run_id, role="runner", create=True)
    try:
        mb.write_order(p.names)
        mb.to_solver.post(np.vstack([x_opt, x_bad]), [p.objective(x_opt), p.objective(x_bad)])
        run = tmp_path / "run"
        rc, out = _launch(MODULE, ["--instance", str(path), "--seed", "3", "--budget", "20", "--run-dir", str(run),
                                   "--mailbox", url, "--run-id", run_id, "--recv", "--send", "--mem-mb", "1000"], mb)
        _check_round_trip(mb, run, p, x_opt, rc, out)
        assert _summary(run)["mailbox_kind"] == "redis"
    finally:
        mb.cleanup()


@pytest.mark.solver
def test_scip_send_posts_incumbents_info_and_root_lp(tmp_path, cover):
    path, p, x_opt = cover
    mb = _new_mailbox(tmp_path, p.names, "send")
    run = tmp_path / "run"
    rc, out = _launch(MODULE, ["--instance", str(path), "--seed", "5", "--budget", "20", "--run-dir", str(run),
                               "--mailbox", str(tmp_path / "mb"), "--send", "--inject-band", "0.5",
                               "--inject-k", "4", "--poll-s", "0.05"], mb)
    assert rc == 0, out
    s = _summary(run)
    ex = s["exchange"]
    assert ex["posts_out"] >= 1 and ex["info_posts"] >= 1 and ex["lp_posts"] == 1 and ex["bytes_out"] > 0
    assert "polls" not in ex                                  # --send alone never reads to_solver
    # the latest to_ea message: canonical order (objective recomputed by the HiGHS-ordered Problem), band, origin
    msg = mb.to_ea.poll()
    assert msg is not None and 1 <= msg.X.shape[0] <= 4
    assert set(msg.origin.tolist()) == {ORIGIN_SOLVER}
    for row, obj in zip(msg.X, msg.obj):
        assert p.objective(row) == pytest.approx(obj)
        assert p.check(row).feasible
    best = min(msg.obj)
    assert all(o <= best + 0.5 * max(1.0, abs(best)) + 1e-9 for o in msg.obj)
    # info: incumbent and dual bound of a minimisation, bound <= incumbent
    info = mb.info.poll()                                     # the final flush after optimize()
    assert info is not None and info["incumbent"] == pytest.approx(s["final_obj"])
    assert info["status"] == "optimal"
    assert info["dual_bound"] is None or info["dual_bound"] <= info["incumbent"] + 1e-6
    # root LP in the original space, canonical order, objective consistent with its vector
    lp = mb.lp.poll()
    assert lp is not None and lp.X.shape == (1, p.n)
    assert p.objective(lp.X[0]) == pytest.approx(float(lp.obj[0]), rel=1e-9, abs=1e-9)
    assert float(lp.obj[0]) <= s["final_obj"] + 1e-6
    assert np.all(lp.X[0] >= -1e-9) and np.all(lp.X[0] <= 1 + 1e-9)
    ev = read_events(run / "solver.events.jsonl")
    kinds = {e["kind"] for e in ev}
    assert {"incumbent", "send", "lp", "callback"} <= kinds
    cb = s["callbacks"]
    assert cb["calls"] > 0 and cb["t_first_callback"] is not None and cb["t_first_solving_callback"] is not None
    assert cb["max_gap_s"] > 0 and cb["callbacks_per_s"] > 0
    assert s["versions"]["pyscipopt"] == "6.2.1"
    for k in ("read_s", "startup_s", "barrier_wait_s", "time_limit_s"):
        assert k in s["timing"]
    assert s["timing"]["time_limit_s"] <= 20.0


@pytest.mark.solver
def test_scip_inert_plugin_costs_under_5_percent(tmp_path):
    """Harness-matched A0: the inert plugin must not change the search (same nodes, same optimum) and must cost
    < 5 % (or < 100 ms) of a 3-4 s toy solve. Medians of 3 alternating runs each."""
    path = toy.write_mps(toy.set_cover(800, 300, 0.02, seed=3), tmp_path / "cover800.mps")
    walls = {"--no-plugin": [], "--inert": []}
    nodes = {"--no-plugin": set(), "--inert": set()}
    objs = set()
    for rep in range(3):
        for flag in ("--no-plugin", "--inert"):
            run = tmp_path / f"run{rep}{flag}"
            rc, out = _launch(MODULE, ["--instance", str(path), "--seed", "0", "--budget", "20", "--run-dir", str(run),
                                       "--no-barrier", flag])
            assert rc == 0, out
            s = _summary(run)
            assert s["exit_reason"] == "optimal" and s["plugin"] == ("none" if flag == "--no-plugin" else "inert")
            walls[flag].append(s["wall_s"])
            nodes[flag].add(s["nodes"])
            objs.add(round(s["final_obj"], 6))
    assert nodes["--no-plugin"] == nodes["--inert"] and len(objs) == 1
    base, inert = float(np.median(walls["--no-plugin"])), float(np.median(walls["--inert"]))
    assert inert - base < max(0.05 * base, 0.1), (walls, base, inert)


@pytest.mark.solver
def test_scip_stop_file_ends_the_run(tmp_path):
    path = toy.write_mps(toy.set_cover(600, 400, 0.02, seed=1), tmp_path / "cover600.mps")
    p = load(path)
    mb = _new_mailbox(tmp_path, p.names, "stop")
    run = tmp_path / "run"
    t = time.monotonic()
    rc, out = _launch(MODULE, ["--instance", str(path), "--seed", "0", "--budget", "20", "--run-dir", str(run),
                               "--mailbox", str(tmp_path / "mb"), "--inert"], mb, stop_after=1.0)
    assert rc == 0, out
    s = _summary(run)
    assert s["exit_reason"] == "stop" and s["wall_s"] < 4.0, s
    assert time.monotonic() - t < 15


@pytest.mark.solver
def test_scip_sigterm_is_a_normal_end(tmp_path):
    import signal
    path = toy.write_mps(toy.set_cover(600, 400, 0.02, seed=1), tmp_path / "cover600.mps")
    run = tmp_path / "run"
    pr = subprocess.Popen([PY, "-m", MODULE, "--instance", str(path), "--seed", "0", "--budget", "20", "--run-dir",
                           str(run), "--no-barrier", "--inert"], env=_env(), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    try:
        end = time.monotonic() + 30
        while not (run / "solver.names.json").exists():          # written after the instance is read
            assert pr.poll() is None and time.monotonic() < end
            time.sleep(0.01)
        time.sleep(0.5)
        pr.send_signal(signal.SIGTERM)
        pr.communicate(timeout=30)
    finally:
        if pr.poll() is None:
            pr.kill()
    assert pr.returncode == 0
    s = _summary(run)
    assert s["exit_reason"] == "sigterm" and s["wall_s"] < 4.0, s


@pytest.mark.solver
@pytest.mark.parametrize("emphasis", ["feasibility", "heuristics_aggressive"])
def test_scip_emphasis_a0star(tmp_path, cover, emphasis):
    """A0*: the runner passes the tuned static emphasis; the run must still be 1 thread, seeded, and solve the toy."""
    path, p, x_opt = cover
    run = tmp_path / "run"
    rc, out = _launch(MODULE, ["--instance", str(path), "--seed", "4", "--budget", "20", "--run-dir", str(run),
                               "--no-barrier", "--inert", "--emphasis", emphasis])
    assert rc == 0, out
    s = _summary(run)
    assert s["emphasis"] == emphasis and s["exit_reason"] == "optimal"
    assert s["final_obj"] == pytest.approx(p.objective(x_opt))


@pytest.mark.solver
@pytest.mark.parametrize("extra", [["--inert", "--recv"], ["--recv"], ["--no-plugin", "--send"], ["--budget", "0"],
                                   ["--emphasis", "turbo"]])
def test_scip_bad_arguments_exit_2(tmp_path, cover, extra):
    path, p, _ = cover
    args = ["--instance", str(path), "--seed", "0", "--budget", "5", "--run-dir", str(tmp_path / "run"),
            "--no-barrier"] + extra
    rc, out = _launch(MODULE, args)
    assert rc == 2, out


@pytest.mark.solver
def test_scip_missing_instance_and_unknown_flag_exit_2(tmp_path):
    rc, _ = _launch(MODULE, ["--instance", str(tmp_path / "nope.mps"), "--seed", "0", "--budget", "5",
                             "--run-dir", str(tmp_path / "run"), "--no-barrier"])
    assert rc == 2
    rc, _ = _launch(MODULE, ["--bogus"])
    assert rc == 2


# ----------------------------------------------------------------------------------------------- shared helpers
def test_vector_hash_folds_noise_and_negative_zero():
    x = np.array([0.0, 1.0, 2.5, 3.0])
    assert vector_hash(x) == vector_hash(x + 1e-9)
    assert vector_hash(x) == vector_hash(np.array([-0.0, 1.0, 2.5, 3.0]))
    assert vector_hash(x) != vector_hash(np.array([0.0, 1.0, 2.5, 4.0]))


def test_band_select_minimise_and_maximise():
    objs = [10.0, 10.05, 10.2, 9.99, 50.0]
    assert band_select(objs, +1, 0.01, 8) == [3, 0, 1]              # within 1 % of 9.99, best first
    assert band_select(objs, +1, 0.001, 8) == [3]
    assert band_select(objs, +1, 0.1, 2) == [3, 0]                  # at most k
    assert band_select([5.0, 4.96, 1.0], -1, 0.01, 8) == [0, 1]     # maximisation: best is the largest
    assert band_select([], 1, 0.1, 3) == []


def test_recover_incumbents_from_part_file(tmp_path):
    names = ["a", "b", "c"]
    (tmp_path / "solver.names.json").write_text(json.dumps(names))
    recs = [(0.5, 3.0, "self", np.array([1.0, 0.0, 2.0])), (1.25, 2.0, "ea", np.array([0.0, 1.0, 1.0]))]
    with open(tmp_path / "solver.incumbents.part", "wb") as f:
        for t, o, s, x in recs:
            nz = np.flatnonzero(x)
            f.write(struct.pack("<ddBi", t, o, SOURCE_CODES[s], nz.size) + nz.astype("<i4").tobytes()
                    + x[nz].astype("<f8").tobytes())
        f.write(struct.pack("<ddBi", 2.0, 1.0, 0, 3)[:10])          # a truncated record (crash mid-write)
    out = recover_incumbents(tmp_path, "solver")
    got_names, got = load_incumbents(out)
    assert got_names == names and len(got) == 2
    for (t, o, s, x), (gt, go, gs, gx) in zip(recs, got):
        assert (t, o, s) == (gt, go, gs) and np.array_equal(x, gx)
