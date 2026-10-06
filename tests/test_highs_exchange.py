"""HiGHS solver process (`itrea.solvers.highs_run`) on toy models: the user-solution round trip (D-001: no HiGHS arm
runs before this passes), HiGHS's ABSOLUTE feasibility tolerance (a 5e-7 violation is accepted, a 5e-4 one is
silently dropped and must be reported as "dropped"), non-improving rows (never tried, never "rejected"), --send on a
maximisation, and bad arguments.

No `Highs` object is `run()` in this test process: HiGHS's global scheduler is initialised by the first run with
that run's thread count, so a later threads=1 run in the same process fails (measured). Reference optima come from
SCIP in-process; HiGHS runs only inside the solver subprocesses.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps

from itrea.benchmarks import toy
from itrea.exchange.mailbox import Mailbox
from itrea.problem import from_arrays, load
from itrea.solvers.common import ORIGIN_EA_ECHO, ORIGIN_SOLVER, vector_hash
from itrea.trace import load_incumbents, read_events

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MODULE = "itrea.solvers.highs_run"


def _env():
    return dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))


def _launch(args, mailbox=None, role="solver", timeout=60.0):
    pr = subprocess.Popen([PY, "-m", MODULE, *args], env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
        out, _ = pr.communicate(timeout=timeout)
    finally:
        if pr.poll() is None:
            pr.kill()
    return pr.returncode, out.decode(errors="replace")


def _summary(run_dir):
    return json.loads((Path(run_dir) / "solver.summary.json").read_text())


def _scip_optimum(path):
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
    return p, x


def _mailbox(tmp_path, names, run_id="highs"):
    mb = Mailbox(tmp_path / "mb", "runner", create=True, run_id=run_id)
    mb.write_order(names)
    return mb


def _verdicts_by_row(mb):
    out = {}
    for v in mb.verdicts():
        note = json.loads(v["note"])
        out[note["row"]] = (v, note)
    return out


@pytest.fixture(scope="module")
def cover(tmp_path_factory):
    d = tmp_path_factory.mktemp("highs_toys")
    path = toy.write_mps(toy.set_cover(400, 250, 0.03, seed=2), d / "cover.mps")
    p, x = _scip_optimum(path)
    assert p.check(x).feasible
    return path, p, x


@pytest.fixture(scope="module")
def tol_toy(tmp_path_factory):
    """min -x - y - v.k  s.t.  x + y <= 10000.5,  w.k <= W;  x integer in [0, 20000], y in [0, 1], k binary.
    The optimum has x = 10000, y = 0.5; a post with y = 0.5 + d violates row 0 by d ABSOLUTE (d / 1e4 relative)."""
    d = tmp_path_factory.mktemp("highs_tol")
    kp = toy.knapsack(60, seed=3)
    nk = kp.n
    A = sps.lil_matrix((2, 2 + nk))
    A[0, 0] = 1.0
    A[0, 1] = 1.0
    A[1, 2:] = kp.A.toarray().ravel()
    p = from_arrays(A.tocsr(), [-np.inf, -np.inf], [10000.5, kp.ru[0]], np.zeros(2 + nk),
                    np.r_[20000.0, 1.0, np.ones(nk)], np.r_[-1.0, -1.0, -kp.c], np.r_[True, False, np.ones(nk, bool)],
                    sense=1, names=["x", "y"] + [f"k{j}" for j in range(nk)])
    path = toy.write_mps(p, d / "tol.mps")
    p, x = _scip_optimum(path)
    assert x[0] == 10000.0 and x[1] == pytest.approx(0.5)
    return path, p, x


# -------------------------------------------------------------------------------------------------------- tests
@pytest.mark.solver
def test_highs_round_trip_user_solution(tmp_path, cover):
    """The optimum posted before the start is injected at the first user-solution call, comes back through
    cbMipSolution (verdict accepted) and becomes HiGHS's incumbent with source "ea"; a feasible but worse row is never
    tried and never reported as rejected; the incumbent goes back to the EA marked as an echo."""
    path, p, x_opt = cover
    x_worse = np.ones(p.n)                                       # every set: feasible, far worse
    assert p.check(x_worse).feasible
    mb = _mailbox(tmp_path, p.names)
    mb.to_solver.post(np.vstack([x_opt, x_worse]), [p.objective(x_opt), p.objective(x_worse)])
    run = tmp_path / "run"
    rc, out = _launch(["--instance", str(path), "--seed", "1", "--budget", "20", "--run-dir", str(run),
                       "--mailbox", str(tmp_path / "mb"), "--run-id", "highs", "--recv", "--send"], mb)
    assert rc == 0, out
    s = _summary(run)
    assert s["exit_reason"] == "optimal" and s["final_obj"] == pytest.approx(p.objective(x_opt), abs=1e-6)
    v = _verdicts_by_row(mb)
    assert set(v) == {0, 1}
    good, gnote = v[0]
    assert good["accepted"] is True and gnote["outcome"] in ("solution_callback", "primal_bound") and gnote["tried"]
    worse, wnote = v[1]
    assert worse["accepted"] is True and wnote["outcome"] == "not_improving" and wnote["tried"] is False
    ev = read_events(run / "solver.events.jsonl")
    ea_inc = [e for e in ev if e["kind"] == "incumbent" and e["source"] == "ea"]
    assert ea_inc and ea_inc[-1]["obj"] == pytest.approx(p.objective(x_opt), abs=1e-6)
    names, recs = load_incumbents(run / "solver.incumbents.npz")
    assert names == p.names                                      # HiGHS order = canonical order
    assert any(src == "ea" and np.allclose(x, x_opt) for _, _, src, x in recs)
    for _, obj, _, x in recs:                                    # every stored vector matches its objective
        assert p.objective(x) == pytest.approx(obj, abs=1e-6)
    msg = mb.to_ea.poll()
    assert msg is not None
    for row, org in zip(msg.X, msg.origin):
        assert org == (ORIGIN_EA_ECHO if vector_hash(row) == vector_hash(x_opt) else ORIGIN_SOLVER)
    assert ORIGIN_EA_ECHO in set(msg.origin.tolist())
    info = mb.info.poll()
    assert info["incumbent"] == pytest.approx(p.objective(x_opt), abs=1e-6) and info["status"] == "Optimal"
    assert s["exchange"]["rows_accepted"] == 1 and s["exchange"]["rows_not_improving"] == 1


@pytest.mark.solver
@pytest.mark.parametrize("viol, accepted", [(5e-7, True), (5e-4, False)])
def test_highs_absolute_tolerance_verdicts(tmp_path, tol_toy, viol, accepted):
    """HiGHS keeps a user solution within its ABSOLUTE 1e-6 and silently drops one 5e-4 outside it (5e-8 relative:
    feasible for SCIP and for Problem.check in 'scip' mode). The process must report the second as dropped."""
    path, p, x_opt = tol_toy
    x = x_opt.copy()
    x[1] = 0.5 + viol
    assert p.check(x, tol_mode="scip").feasible
    assert p.check(x, tol_mode="highs").feasible is accepted
    mb = _mailbox(tmp_path, p.names)
    mb.to_solver.post(x[None, :], [p.objective(x)])
    run = tmp_path / "run"
    rc, out = _launch(["--instance", str(path), "--seed", "0", "--budget", "20", "--run-dir", str(run),
                       "--mailbox", str(tmp_path / "mb"), "--recv"], mb)
    assert rc == 0, out
    v = _verdicts_by_row(mb)
    verdict, note = v[0]
    assert verdict["accepted"] is accepted, note
    assert note["tried"] is True
    if accepted:
        assert note["outcome"] in ("solution_callback", "primal_bound")
        assert _summary(run)["final_obj"] == pytest.approx(p.objective(x), abs=1e-9)
    else:
        assert note["outcome"] in ("dropped", "dropped_final")
        s = _summary(run)
        assert s["exchange"]["rows_dropped"] == 1
        assert s["final_obj"] == pytest.approx(p.objective(x_opt), abs=1e-6)   # HiGHS's own optimum, not the post
        ev = read_events(run / "solver.events.jsonl")
        assert not any(e["kind"] == "incumbent" and e["source"] == "ea" for e in ev)


@pytest.mark.solver
def test_highs_send_on_a_maximisation(tmp_path):
    path = toy.write_mps(toy.knapsack(40, seed=1), tmp_path / "knap.mps")
    p = load(path)
    mb = _mailbox(tmp_path, p.names)
    run = tmp_path / "run"
    rc, out = _launch(["--instance", str(path), "--seed", "2", "--budget", "20", "--run-dir", str(run),
                       "--mailbox", str(tmp_path / "mb"), "--send", "--inject-band", "0.2", "--inject-k", "3",
                       "--poll-s", "0.01"], mb)
    assert rc == 0, out
    s = _summary(run)
    assert s["sense"] == -1 and s["exit_reason"] == "optimal"
    assert s["final_obj"] == pytest.approx(toy.knapsack(40, seed=1).toy_optimum)
    msg = mb.to_ea.poll()
    assert msg is not None and 1 <= msg.X.shape[0] <= 3 and set(msg.origin.tolist()) == {ORIGIN_SOLVER}
    for row, obj in zip(msg.X, msg.obj):
        assert p.objective(row) == pytest.approx(obj) and p.check(row).feasible
    best = max(msg.obj)                                              # maximisation: band below the best
    assert all(o >= best - 0.2 * max(1.0, abs(best)) - 1e-9 for o in msg.obj)
    info = mb.info.poll()
    assert info["incumbent"] == pytest.approx(s["final_obj"])
    assert info["dual_bound"] is None or info["dual_bound"] >= info["incumbent"] - 1e-6
    assert "polls" not in s["exchange"] and s["exchange"]["posts_out"] >= 1
    assert s["versions"]["highspy"].startswith("1.15")
    assert s["highs_options"]["mip_feasibility_tolerance"] == pytest.approx(1e-6)


@pytest.mark.solver
@pytest.mark.parametrize("extra", [["--inert", "--send"], ["--send"], ["--inject-k", "0"]])
def test_highs_bad_arguments_exit_2(tmp_path, cover, extra):
    path, _, _ = cover
    rc, out = _launch(["--instance", str(path), "--seed", "0", "--budget", "5", "--run-dir", str(tmp_path / "run"),
                       "--no-barrier"] + extra)
    assert rc == 2, out


@pytest.mark.solver
def test_highs_inert_runs_alone(tmp_path, cover):
    """A0-style HiGHS run without a mailbox: no exchange counters, the callback statistics are recorded."""
    path, p, x_opt = cover
    run = tmp_path / "run"
    rc, out = _launch(["--instance", str(path), "--seed", "0", "--budget", "20", "--run-dir", str(run),
                       "--no-barrier", "--inert"])
    assert rc == 0, out
    s = _summary(run)
    assert s["plugin"] == "inert" and s["exit_reason"] == "optimal"
    assert s["final_obj"] == pytest.approx(p.objective(x_opt), abs=1e-6)
    assert s["callbacks"]["user_calls"] > 0 and s["exchange"]["bytes_out"] == 0
    assert (run / "solver.sol.json.gz").exists() and (run / "solver.incumbents.npz").exists()
