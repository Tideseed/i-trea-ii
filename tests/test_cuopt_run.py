"""cuOpt process (`itrea.solvers.cuopt_run`) on a toy model.

Measured 2026-10-06 (cuOpt 26.08.00): the MIP solve requests two fixed 2.4 GB device buffers whatever the instance,
so it cannot solve anything under an RMM limit below about 4.8 GB. The default tests therefore run with the 0.5 GB
limit the brief asks for and check the FAILURE path is surfaced, not swallowed (optopt's dano3_3 trap: termination
status 0 with a placeholder objective 0.0 read as a result):
- with the warm-up, the process dies BEFORE reporting ready (exit 1), so the runner sees a dead process instead of a
  stuck barrier;
- without the warm-up, cuOpt's CPU feasibility-jump incumbents that arrive before the GPU allocation fails are
  recorded and stored (and posted with --send), every one consistent with its objective, and the run ends with exit
  1 and cuOpt's error status in the summary.
The solving path runs only when ITREA_TEST_CUOPT_GPU_GB (>= 5) is set by an operator holding the box (the per-process
budget of these tests is about 4 GB).
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

from itrea.benchmarks import toy
from itrea.exchange.mailbox import Mailbox
from itrea.problem import load
from itrea.solvers.common import ORIGIN_SOLVER
from itrea.trace import load_incumbents, read_events

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MODULE = "itrea.solvers.cuopt_run"


def _cuopt_available() -> bool:
    """cuOpt and RMM importable (checked without importing them here) and a CUDA device visible."""
    import importlib.util
    if any(importlib.util.find_spec(m) is None for m in ("cuopt", "rmm", "cupy")):
        return False
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


needs_cuopt = pytest.mark.skipif(not _cuopt_available(), reason="cuOpt / RMM / a CUDA device not available")


def _env():
    return dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))


def _launch(args, mailbox=None, timeout=120.0):
    """Returns (exit code, output, whether the process reported ready)."""
    pr = subprocess.Popen([PY, "-m", MODULE, *args], env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ready = False
    try:
        if mailbox is not None:
            end = time.monotonic() + timeout
            while not mailbox.all_ready(roles=("cuopt",)):
                if pr.poll() is not None:
                    break
                assert time.monotonic() < end, "process never reported ready"
                time.sleep(0.005)
            else:
                ready = True
                mailbox.release()
        out, _ = pr.communicate(timeout=timeout)
    finally:
        if pr.poll() is None:
            pr.kill()
    return pr.returncode, out.decode(errors="replace"), ready


@pytest.fixture(scope="module")
def cover(tmp_path_factory):
    d = tmp_path_factory.mktemp("cuopt_toys")
    path = toy.write_mps(toy.set_cover(400, 250, 0.03, seed=2), d / "cover.mps")
    return path, load(path)


def _mailbox(tmp_path, names):
    mb = Mailbox(tmp_path / "mb", "runner", create=True, run_id="cuopt")
    mb.write_order(names)
    return mb


@pytest.mark.solver
@pytest.mark.gpu
@needs_cuopt
def test_cuopt_warmup_failure_exits_before_ready(tmp_path, cover):
    path, p = cover
    mb = _mailbox(tmp_path, p.names)
    run = tmp_path / "run"
    rc, out, ready = _launch(["--instance", str(path), "--seed", "1", "--budget", "20", "--run-dir", str(run),
                              "--mailbox", str(tmp_path / "mb"), "--run-id", "cuopt", "--gpu-mem-gb", "0.5",
                              "--cpu-threads", "2"], mb)
    assert rc == 1, out
    assert not ready and not (tmp_path / "mb" / "ready.cuopt").exists()
    s = json.loads((run / "cuopt.summary.json").read_text())
    assert s["exit_reason"] == "exception" and s["status"] == "warmup_failed" and s["error_status"] == 6
    assert "Memory allocation failed" in s["error"]
    assert s["rmm_limit_bytes"] == int(0.5 * 1024**3) and s["rmm_peak_bytes"] <= s["rmm_limit_bytes"]
    assert s["n_incumbents"] == 0


@pytest.mark.solver
@pytest.mark.gpu
@needs_cuopt
def test_cuopt_rmm_limit_run_writes_trace_and_incumbents(tmp_path, cover):
    """0.5 GB RMM limit, 20 s budget, no warm-up: trace, summary and incumbent vectors are written; the GPU part
    fails under the limit and the run says so (exit 1, cuOpt error status), never a placeholder objective."""
    path, p = cover
    mb = _mailbox(tmp_path, p.names)
    run = tmp_path / "run"
    rc, out, ready = _launch(["--instance", str(path), "--seed", "1", "--budget", "20", "--run-dir", str(run),
                              "--mailbox", str(tmp_path / "mb"), "--run-id", "cuopt", "--gpu-mem-gb", "0.5",
                              "--cpu-threads", "2", "--no-warmup", "--send"], mb)
    assert ready
    s = json.loads((run / "cuopt.summary.json").read_text())
    assert (run / "cuopt.events.jsonl").exists() and (run / "cuopt.incumbents.npz").exists()
    assert s["rmm_limit_bytes"] == int(0.5 * 1024**3) and 0 < s["rmm_peak_bytes"] <= s["rmm_limit_bytes"]
    assert s["versions"]["cuopt"].startswith("26.08")
    assert s["cpu_threads"] == 2 and s["timing"]["time_limit_s"] <= 20.0
    if rc == 1:                     # the measured behaviour under 0.5 GB
        assert s["exit_reason"] == "exception" and s["error_status"] == 6, s.get("error")
    else:                           # a cuOpt that fits in 0.5 GB would be news: then it must have solved properly
        assert rc == 0 and s["exit_reason"] in ("optimal", "budget"), out
    names, recs = load_incumbents(run / "cuopt.incumbents.npz")
    assert len(recs) == s["n_incumbents"]
    pos = p.name_index()
    perm = np.array([pos[nm] for nm in names])
    keys = []
    for t, obj, src, x in recs:     # every stored incumbent is a real, feasible vector with its own objective
        xc = np.zeros(p.n)
        xc[perm] = x
        assert src == "self" and t >= 0
        assert p.objective(xc) == pytest.approx(obj, abs=1e-6) and p.check(xc).feasible
        keys.append(obj)
    assert keys == sorted(keys, reverse=True)                    # minimisation: strictly improving record
    ev = read_events(run / "cuopt.events.jsonl")
    assert sum(e["kind"] == "incumbent" for e in ev) == len(recs)
    if recs:                        # --send: the last incumbent went to the EA in canonical order, on cuOpt's own
        assert mb.to_ea.poll() is None and mb.info.poll() is None      # channels (A9) — never the solver's
        msg = mb.to_ea_cuopt.poll()
        assert msg is not None and set(msg.origin.tolist()) == {ORIGIN_SOLVER}
        info = mb.info_cuopt.poll()
        assert info is not None and info["solver"] == "cuopt"
        assert msg.obj[0] == pytest.approx(recs[-1][1])
        assert p.objective(msg.X[0]) == pytest.approx(msg.obj[0], abs=1e-6)
        assert s["exchange"]["posts_out"] == len(recs)


@pytest.mark.solver
@pytest.mark.gpu
@needs_cuopt
def test_cuopt_solves_with_enough_gpu_memory(tmp_path, cover):
    gb = float(os.environ.get("ITREA_TEST_CUOPT_GPU_GB", "0") or 0)
    if gb < 5.0:
        pytest.skip("cuOpt 26.08 needs > ~4.8 GB of device memory (two fixed 2.4 GB buffers, measured 2026-10-06); "
                    "set ITREA_TEST_CUOPT_GPU_GB >= 5 under a box-window lease to run this")
    path, p = cover
    mb = _mailbox(tmp_path, p.names)
    run = tmp_path / "run"
    rc, out, ready = _launch(["--instance", str(path), "--seed", "1", "--budget", "20", "--run-dir", str(run),
                              "--mailbox", str(tmp_path / "mb"), "--run-id", "cuopt", "--gpu-mem-gb", str(gb),
                              "--cpu-threads", "2", "--send"], mb)
    assert rc == 0 and ready, out
    s = json.loads((run / "cuopt.summary.json").read_text())
    assert s["exit_reason"] in ("optimal", "budget") and s["n_incumbents"] >= 1
    assert s["rmm_peak_bytes"] <= s["rmm_limit_bytes"]
    assert s["warmup"]["error_status"] == 0 and s["timing"]["warmup_s"] > 0
    assert (run / "cuopt.sol.json.gz").exists()
    names, recs = load_incumbents(run / "cuopt.incumbents.npz")
    pos = p.name_index()
    xc = np.zeros(p.n)
    xc[[pos[nm] for nm in names]] = recs[-1][3]
    assert p.check(xc).feasible and p.objective(xc) == pytest.approx(s["final_obj"], abs=1e-6)


@pytest.mark.solver
@pytest.mark.parametrize("extra", [["--cpu-threads", "1"], ["--gpu-mem-gb", "0"], ["--send"]])
def test_cuopt_bad_arguments_exit_2(tmp_path, cover, extra):
    """Validated before cuOpt or CUDA is touched (no GPU needed)."""
    path, _ = cover
    rc, out, _ = _launch(["--instance", str(path), "--seed", "0", "--budget", "5", "--run-dir", str(tmp_path / "run"),
                          "--no-barrier"] + extra)
    assert rc == 2, out
