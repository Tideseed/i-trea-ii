"""The EA process (`python -m itrea.ea.run`): files, summary fields, exit codes, the start barrier and the stop file,
`compiled_on_clock` with a cold CuPy kernel cache, and an end-to-end pair with SCIP (`scip_run --recv --send`)
through the file mailbox and through Redis.

Toy models only, written as MPS in tmp_path. The pair test needs a toy on which SCIP is still solving when the EA's
first post lands (SCIP polls its mailbox at its first callback and then every 0.25 s; a toy SCIP solves in
milliseconds ends before it ever sees a post). Measured 2026-10-06: on `set_cover(2000, 800, 0.01, seed=5)` with SCIP's
heuristics off, SCIP alone runs the whole 10 s budget and finds its first incumbent only after ~0.5 s, while the EA's
first feasible cover is posted ~0.02 s after t0 — so the EA's solutions become SCIP's incumbents. The test asserts
the timing precondition itself, so a slower box fails with a clear message rather than a missing verdict.
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from itrea.benchmarks import toy
from itrea.config import EAConfig
from itrea.exchange.mailbox import Mailbox
from itrea.problem import load
from itrea.trace import load_incumbents, read_events

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MODULE = "itrea.ea.run"
SMALL = ["--set", "population.size=48", "--set", "population.offspring_per_gen=48", "--gpu-mem-gb", "1"]


def _env(**extra):
    return dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), **extra)


def _run(args, timeout=120.0, **env):
    pr = subprocess.run([PY, "-m", MODULE, *args], env=_env(**env), capture_output=True, timeout=timeout)
    return pr.returncode, (pr.stdout + pr.stderr).decode(errors="replace")


def _summary(run_dir, role="ea"):
    return json.loads((Path(run_dir) / f"{role}.summary.json").read_text())


@pytest.fixture(scope="module")
def knap(tmp_path_factory):
    d = tmp_path_factory.mktemp("ea_run")
    p = toy.knapsack(30)
    return toy.write_mps(p, d / "knap30.mps"), p.toy_optimum


def _gpu_ok() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------- single process
def test_no_barrier_numpy_files_and_summary(knap, tmp_path):
    path, opt = knap
    rd = tmp_path / "run"
    rc, out = _run(["--instance", str(path), "--seed", "0", "--budget", "3", "--run-dir", str(rd), "--no-barrier",
                    "--backend", "numpy", "--sol-out", str(tmp_path / "copy.sol.json.gz"), "--arm", "A5", *SMALL])
    assert rc == 0, out
    s = _summary(rd)
    assert s["exit_reason"] == "budget" and s["arm"] == "A5"
    assert s["backend"] == "numpy" and s["device"] == "cpu" and s["pool_limit_bytes"] is None
    assert s["P"] == 48 and s["offspring"] == 48 and s["gene_dtype"] == "int8"
    cfg = EAConfig().with_overrides({"population.size": 48, "population.offspring_per_gen": 48, "gpu_mem_gb": 1.0,
                                     "backend": "numpy"})
    assert s["config_hash"] == cfg.hash() and s["config"]["backend"] == "numpy"
    assert s["sense"] == -1 and s["compiled_on_clock"] is None
    assert s["startup_s"] > 0 and s["timing"]["startup_s"] == s["startup_s"] and "warmup_s" in s["startup"]
    assert s["versions"]["numpy"] and s["versions"]["highspy"]
    assert s["generations"] > 0 and s["best_obj"] == pytest.approx(opt)
    for f in ("ea.events.jsonl", "ea.incumbents.npz", "ea.genealogy.npz", "ea.frontier.npz", "ea.sol.json.gz"):
        assert (rd / f).exists(), f
    assert not (rd / "ea.incumbents.part").exists() and not (rd / "ea.names.json").exists()   # crash log removed
    p = load(path)
    for sol_path in (rd / "ea.sol.json.gz", tmp_path / "copy.sol.json.gz"):
        sol = json.loads(gzip.open(sol_path, "rt").read())
        x = np.array([sol.get(nm, 0.0) for nm in p.names])
        assert p.check(x).feasible and p.objective(x) == pytest.approx(s["best_obj"])
    names, recs = load_incumbents(rd / "ea.incumbents.npz")
    assert names == p.names and len(recs) == s["n_incumbents"] == len(s["incumbents"])
    assert all(p.check(x).feasible for _, _, _, x in recs)
    ev = read_events(rd / "ea.events.jsonl")
    assert sum(e["kind"] == "incumbent" for e in ev) == s["n_incumbents"]
    assert all(e["t"] >= 0 for e in ev)


BAD = {
    "budget": lambda P, R: ["--instance", P, "--seed", "0", "--budget", "0", "--run-dir", R, "--no-barrier"],
    "unknown": lambda P, R: ["--instance", P, "--seed", "0", "--budget", "2", "--run-dir", R, "--no-barrier",
                             "--set", "ranks.nope=1"],
    "KEY=VALUE": lambda P, R: ["--instance", P, "--seed", "0", "--budget", "2", "--run-dir", R, "--no-barrier",
                               "--set", "ranksx"],
    "no such file": lambda P, R: ["--instance", "/nonexistent.mps", "--seed", "0", "--budget", "2", "--run-dir", R,
                                  "--no-barrier"],
    "mailbox": lambda P, R: ["--instance", P, "--seed", "0", "--budget", "2", "--run-dir", R],
    "invalid EA configuration": lambda P, R: ["--instance", P, "--seed", "0", "--budget", "2", "--run-dir", R,
                                              "--no-barrier", "--set", "ranks.x=2.0"],
}


@pytest.mark.parametrize("what", sorted(BAD))
def test_bad_arguments_exit_2(knap, tmp_path, what):
    path, _ = knap
    rd = tmp_path / "r"
    rc, out = _run(BAD[what](str(path), str(rd)) + ["--backend", "numpy"])
    assert rc == 2, out
    assert what in out
    assert _summary(rd)["exit_reason"] == "bad_arguments"


def test_barrier_and_stop_file(knap, tmp_path):
    path, _ = knap
    mb = Mailbox(tmp_path / "mb", "runner", create=True, run_id="r1")
    mb.write_order(load(path).names)
    rd = tmp_path / "run"
    pr = subprocess.Popen([PY, "-m", MODULE, "--instance", str(path), "--seed", "1", "--budget", "60", "--run-dir",
                           str(rd), "--mailbox", str(tmp_path / "mb"), "--run-id", "r1", "--backend", "numpy", *SMALL],
                          env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        end = time.monotonic() + 60
        while not mb.all_ready(roles=("ea",)):
            assert pr.poll() is None, pr.stdout.read().decode()
            assert time.monotonic() < end
            time.sleep(0.01)
        t0 = mb.release()
        time.sleep(1.5)
        mb.stop("test")
        rc = pr.wait(timeout=30)
    finally:
        if pr.poll() is None:
            pr.kill()
    assert rc == 0, pr.stdout.read().decode()
    s = _summary(rd)
    assert s["exit_reason"] == "stop" and s["t0"] == pytest.approx(t0)
    assert s["timing"]["barrier_wait_s"] >= 0 and s["wall_s"] < 10


def test_stopped_before_start(knap, tmp_path):
    path, _ = knap
    mb = Mailbox(tmp_path / "mb", "runner", create=True, run_id="r2")
    mb.write_order(load(path).names)
    rd = tmp_path / "run"
    pr = subprocess.Popen([PY, "-m", MODULE, "--instance", str(path), "--seed", "1", "--budget", "5", "--run-dir",
                           str(rd), "--mailbox", str(tmp_path / "mb"), "--run-id", "r2", "--backend", "numpy", *SMALL],
                          env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    end = time.monotonic() + 60
    while not mb.all_ready(roles=("ea",)) and time.monotonic() < end and pr.poll() is None:
        time.sleep(0.01)
    mb.stop("partner died during start-up")
    assert pr.wait(timeout=30) == 0
    assert _summary(rd)["exit_reason"] == "stopped_before_start"


def test_wrong_run_id_is_an_exception(knap, tmp_path):
    path, _ = knap
    Mailbox(tmp_path / "mb", "runner", create=True, run_id="right")
    rc, out = _run(["--instance", str(path), "--seed", "0", "--budget", "2", "--run-dir", str(tmp_path / "r"),
                    "--mailbox", str(tmp_path / "mb"), "--run-id", "wrong", "--backend", "numpy", *SMALL])
    assert rc == 1 and "another run" in out
    assert _summary(tmp_path / "r")["exit_reason"] == "exception"


def _size_classes(d: Path) -> list[Path]:
    """Two synthetic instances of other size classes than the warm-up's (review F2, 2026-10-06): a 250 x 15
    multi-knapsack, and a 2048 x 600 set cover whose crossover rows (2048 genes) and mutation scans (P * 600 rows)
    are past the 2^10-element class of CuPy's scan, which the synthetic warm-up problem never reaches by itself."""
    import scipy.sparse as sps
    from itrea.problem import from_arrays

    rs = np.random.default_rng(7)
    n, m = 250, 15
    W = rs.integers(5, 60, size=(m, n)).astype(float)
    v = (W.mean(0) + rs.integers(0, 15, n)).astype(float)
    mkp = from_arrays(sps.csr_matrix(W), np.full(m, -np.inf), W.sum(1) * 0.25, np.zeros(n), np.ones(n), v,
                      np.ones(n, bool), sense=-1, names=[f"x{j}" for j in range(n)])
    n, m = 2048, 600
    rows = rs.integers(0, m, size=3 * n)
    A = sps.csr_matrix((np.ones(rows.size), (rows, np.tile(np.arange(n), 3))), shape=(m, n))
    A.sum_duplicates()
    A.data[:] = 1.0
    cover = from_arrays(A, np.ones(m), np.full(m, np.inf), np.zeros(n), np.ones(n),
                        rs.integers(1, 100, n).astype(float), np.ones(n, bool), sense=1,
                        names=[f"y{j}" for j in range(n)])
    return [toy.write_mps(mkp, d / "mkp250.mps"), toy.write_mps(cover, d / "cover2048.mps")]


@pytest.mark.gpu
def test_cupy_cold_cache_compiles_before_the_clock(knap, tmp_path):
    """With an empty CuPy kernel cache, every kernel the run needs is compiled by the warm-up (before t0) — on the
    knapsack AND then, with the same cache, on instances of other size classes (CUB off by `ea.run`'s default, CuPy's
    scan size classes warmed by `Engine._warm_scans`). CUPY_ACCELERATORS is deliberately NOT set here: the test checks
    the process's own default (with CUB on, the 250 x 15 and 2048 x 600 runs compile on the clock)."""
    if not _gpu_ok():
        pytest.skip("no CuPy device")
    path, opt = knap
    cache = tmp_path / "cupy_cache"
    env = {k: v for k, v in _env(CUPY_CACHE_DIR=str(cache)).items() if k != "CUPY_ACCELERATORS"}
    for i, inst in enumerate([path, *_size_classes(tmp_path)]):
        rd = tmp_path / f"run{i}"
        pr = subprocess.run([PY, "-m", MODULE, "--instance", str(inst), "--seed", "0", "--budget", "3", "--run-dir",
                             str(rd), "--no-barrier", "--backend", "cupy", *SMALL],
                            env=env, capture_output=True, timeout=300)
        assert pr.returncode == 0, (pr.stdout + pr.stderr).decode(errors="replace")
        s = _summary(rd)
        assert s["backend"] == "cupy" and s["device"] and s["pool_limit_bytes"] == 1 << 30
        assert s["local_search"] == "fj_gpu"
        acc = s["startup"]["cupy_accelerators"]
        assert acc["env"] == "" and acc["routine"] == [] and acc["reduction"] == []
        cc = s["cupy_cache"]
        if i == 0:
            assert cc["files_before_prepare"] == 0 and cc["files_at_t0"] > 0
            assert s["best_obj"] == pytest.approx(opt)
        assert s["compiled_on_clock"] is False and cc["new_after_t0"] == [], (inst, cc)
        assert s["feasible_found"] and s["generations"] >= 1


# -------------------------------------------------------------------------------------------- pair with SCIP
@pytest.fixture(scope="module")
def cover(tmp_path_factory):
    d = tmp_path_factory.mktemp("pair")
    return toy.write_mps(toy.set_cover(2000, 800, 0.01, seed=5), d / "cover2000.mps")


def _redis_ok() -> bool:
    try:
        import redis
        return bool(redis.Redis(host="127.0.0.1", port=6390, socket_timeout=1).ping())
    except Exception:  # noqa: BLE001
        return False


def _pair(inst: Path, run_dir: Path, spec: str, run_id: str, mb, budget: float = 10.0):
    """The runner's part: both processes, the barrier, the stop file, the exit codes."""
    p = load(inst)
    mb.write_order(p.names)
    run_dir.mkdir(parents=True, exist_ok=True)
    common = ["--instance", str(inst), "--seed", "0", "--budget", str(budget), "--run-dir", str(run_dir),
              "--mailbox", spec, "--run-id", run_id]
    scip = subprocess.Popen([PY, "-m", "itrea.solvers.scip_run", *common, "--recv", "--send",
                             "--emphasis", "heuristics_off"],
                            env=_env(), stdout=open(run_dir / "solver.log", "w"), stderr=subprocess.STDOUT)
    ea = subprocess.Popen([PY, "-m", MODULE, *common, "--backend", "numpy", "--arm", "A4", *SMALL],
                          env=_env(), stdout=open(run_dir / "ea.log", "w"), stderr=subprocess.STDOUT)
    try:
        end = time.monotonic() + 120
        while not mb.all_ready(roles=("solver", "ea")):
            assert scip.poll() is None and ea.poll() is None, "a process died before reporting ready"
            assert time.monotonic() < end, "never ready"
            time.sleep(0.01)
        mb.release()
        time.sleep(budget + 1.0)
        mb.stop("runner: budget")
        rcs = (scip.wait(timeout=60), ea.wait(timeout=60))
    finally:
        for pr in (scip, ea):
            if pr.poll() is None:
                pr.kill()
    return p, rcs


def _check_pair(p, run_dir: Path, rcs, verdicts):
    assert rcs == (0, 0), (rcs, (run_dir / "solver.log").read_text()[-2000:], (run_dir / "ea.log").read_text()[-2000:])
    ss, es = _summary(run_dir, "solver"), _summary(run_dir, "ea")
    assert ss["exit_reason"] in ("budget", "stop", "optimal") and es["exit_reason"] in ("budget", "stop")
    ev_ea = read_events(run_dir / "ea.events.jsonl")
    posts = [e for e in ev_ea if e["kind"] == "post"]
    assert posts, "the EA never posted"
    # the precondition of the test: SCIP was still solving when the EA's first post was made
    assert ss["wall_s"] > posts[0]["t"] + 0.5, f"SCIP ended at {ss['wall_s']} s, the EA posted at {posts[0]['t']} s"
    # verdicts: recorded for EA posts, at least one accepted by SCIP's own check
    seqs = {e["seq"] for e in posts}
    assert verdicts and {v["seq"] for v in verdicts} <= seqs
    acc = [v for v in verdicts if v["accepted"]]
    assert acc, verdicts
    ev_s = read_events(run_dir / "solver.events.jsonl")
    assert sum(e["kind"] == "verdict" for e in ev_s) == len(verdicts)
    # an EA solution became SCIP's incumbent (SCIP's heuristics are off, so the EA's covers arrive first)
    from_ea = [e for e in ev_s if e["kind"] == "incumbent" and e["source"] == "ea"]
    assert from_ea, "no EA solution became SCIP's incumbent"
    post_objs = {round(e["obj"], 6) for e in posts}
    assert all(round(e["obj"], 6) in post_objs for e in from_ea)
    # SCIP sent them back marked as echoes; the EA counted and skipped them; the EA saw verdicts
    assert es["echoes"] >= 1 and es["inject_msgs"] >= 1 and es["solver_accepted"] >= 1
    assert es["solver_rejections"] == 0
    # every EA incumbent vector is feasible; every accepted post was an EA incumbent
    names, recs = load_incumbents(run_dir / "ea.incumbents.npz")
    assert names == p.names and all(p.check(x).feasible for _, _, _, x in recs)
    inc_objs = {round(o, 6) for _, o, _, _ in recs}
    assert all(round(v["obj"], 6) in inc_objs for v in acc)
    g = np.load(run_dir / "ea.genealogy.npz")
    assert g["is_accepted_post"].sum() >= 1
    return es, ss


@pytest.mark.solver
def test_pair_with_scip_file_mailbox(cover, tmp_path):
    run_id = f"pairf-{uuid.uuid4().hex[:8]}"
    mb = Mailbox(tmp_path / "mb", "runner", create=True, run_id=run_id)
    p, rcs = _pair(cover, tmp_path / "run", str(tmp_path / "mb"), run_id, mb)
    _check_pair(p, tmp_path / "run", rcs, mb.verdicts())


@pytest.mark.solver
def test_pair_with_scip_redis(cover, tmp_path):
    if not _redis_ok():
        pytest.skip("itrea-redis is not running on 127.0.0.1:6390 (itrea exchange up)")
    from itrea.exchange.redis_mailbox import open_mailbox
    run_id = f"pytest-pair-{uuid.uuid4().hex[:8]}"
    spec = f"redis://127.0.0.1:6390/0?run={run_id}"
    mb = open_mailbox(spec, "runner", run_id=run_id, create=True)
    try:
        p, rcs = _pair(cover, tmp_path / "run", spec, run_id, mb)
        es, _ = _check_pair(p, tmp_path / "run", rcs, mb.verdicts())
        assert es["mailbox_kind"] == "redis"
    finally:
        mb.cleanup()
        assert not list(mb.r.scan_iter(match=f"itrea:{run_id}:*"))
