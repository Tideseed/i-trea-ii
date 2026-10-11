"""The pair runner (vault `10-design/Process interface.md`): process plans and footprints, run ids, the barrier, exit-code
handling, stale-state refusal, both mailbox transports, and — where the role processes exist — real toy runs.

Most tests use a FAKE role process (written to tmp_path) that speaks the barrier protocol, so the runner is tested on
its own; the real solver/EA processes are tested only on toy models, with `--no-systemd` (no scopes)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest

from itrea.benchmarks.toy import knapsack, set_cover, write_mps
from itrea.config import EAConfig
from itrea.experiments import gates, pair

FAKE_PROC = r'''
"""Fake role process: barrier protocol, a few incumbents, exit codes from FAKE_* environment variables."""
import argparse, os, sys, time
from itrea.exchange.redis_mailbox import open_mailbox
from itrea.trace import Trace
ap = argparse.ArgumentParser()
for a in ("--instance", "--seed", "--budget", "--run-dir", "--mailbox", "--sol-out", "--arm", "--mem-mb",
          "--role-name", "--config", "--cpu-threads", "--inject-band", "--inject-k", "--emphasis", "--run-id",
          "--gpu-mem-gb"):
    ap.add_argument(a)
for f in ("--recv", "--send", "--inert", "--no-barrier", "--mip-gap-zero"):
    ap.add_argument(f, action="store_true")
a = ap.parse_args()
role = a.role_name or os.environ.get("ITREA_ROLE", "solver")
R = role.upper()
if os.environ.get("FAKE_ENV_OUT"):                      # what the process inherited (A6m: no thread variables)
    import json
    with open(os.environ["FAKE_ENV_OUT"] + "." + role, "w") as fh:
        json.dump({k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                  "ITREA_ROLE", "CUPY_GPU_MEMORY_LIMIT", "PYTHONUNBUFFERED",
                                                  "CUDA_CACHE_PATH", "CUDA_CACHE_MAXSIZE")}, fh)
mb = open_mailbox(a.mailbox, role, run_id=a.run_id)
tr = Trace(a.run_dir, role, arm=a.arm, seed=int(a.seed))
if os.environ.get("FAKE_EXIT_EARLY_" + R):
    if os.environ.get("FAKE_SUMMARY_EARLY_" + R):          # the arm's own code failed: its summary says why
        tr.close(exit_reason="exception")
    sys.exit(int(os.environ["FAKE_EXIT_EARLY_" + R]))
mb.ready()
try:
    t0 = mb.wait_start(timeout=60)
except RuntimeError:
    sys.exit(0)                       # stopped before start: a normal end
tr.start(t0)
base = float(os.environ.get("FAKE_BASE_" + R, "100"))
k = 0
while time.monotonic() - t0 < float(a.budget) and not mb.stopped():
    tr.incumbent(base - k, source="self")
    k += 1
    time.sleep(0.1)
    if os.environ.get("FAKE_CRASH_" + R) and k == 3:
        sys.exit(int(os.environ["FAKE_CRASH_" + R]))
    if os.environ.get("FAKE_END_AFTER_" + R) and time.monotonic() - t0 > float(os.environ["FAKE_END_AFTER_" + R]):
        tr.close(exit_reason=os.environ.get("FAKE_REASON_" + R, "node_limit"))
        sys.exit(0)
tr.close(exit_reason="stop" if mb.stopped() else "budget")
sys.exit(int(os.environ.get("FAKE_EXIT_" + R, "0")))
'''


@pytest.fixture
def toy(tmp_path) -> Path:
    return write_mps(set_cover(30, 20, seed=1), tmp_path / "toy_sc.mps")


@pytest.fixture
def fake(tmp_path) -> list[str]:
    p = tmp_path / "fake_proc.py"
    p.write_text(FAKE_PROC)
    return [sys.executable, str(p)]


def opts_for(tmp_path, fake=None, **kw) -> pair.PairOptions:
    d = dict(traces_dir=tmp_path / "traces", mailbox_root=tmp_path / "mb", use_systemd=False, pin=False, grace_s=2.0,
             kill_after_s=10.0, gpu_health=False, record_snapshots=False, solver_mem_gb=2.0, ea_mem_gb=2.0,
             cuopt_mem_gb=3.0, cupy_cache_dir=tmp_path / "cupy-cache", log=lambda m: None)
    if fake:
        d["module_override"] = {k: fake for k in ("scip", "highs", "cuopt", "ea")}
    d.update(kw)
    return pair.PairOptions(**d)


def job(instance, arm="A0", seed=1, budget=1.0, **kw) -> dict:
    return dict(instance=str(instance), set="pytest", arm=arm, seed=seed, budget=budget, split="smoke", **kw)


def _implemented(module: str) -> bool:
    """A role module counts as implemented when `python -m <module> --help` exits 0 (the contract stubs raise)."""
    try:
        r = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, timeout=120,
                           env=dict(os.environ, OMP_NUM_THREADS="1"))
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


def _cupy_ok() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------------------------------- identity
def test_run_id_format(toy, tmp_path):
    o = opts_for(tmp_path)
    rid = pair.run_id_for(job(toy, "A0", seed=7, budget=300), o)
    name, arm, seed, budget, h = rid.split("__")
    assert (name, arm, seed, budget) == ("toy_sc", "A0", "s7", "300s") and len(h) == 12
    # arms with an EA carry the effective configuration hash (it equals the EA summary's config_hash)
    cfg, ov = pair.effective_config(job(toy, "A5x0"), o)
    assert ov["ranks.x"] == 0.0 and ov["exchange.enabled"] is False and ov["backend"] == "cupy"
    assert pair.run_id_for(job(toy, "A5x0"), o).endswith(cfg.hash())
    assert pair.run_hash(job(toy, "A5"), o) != pair.run_hash(job(toy, "A5x0"), o)
    assert "__A3~one_way_ea_to_solver__" in pair.run_id_for(job(toy, "A3", ablation="one_way_ea_to_solver"), o)
    assert "__A5@c01__" in pair.run_id_for(job(toy, "A5", variant="c01"), o)
    with pytest.raises(pair.RunError, match="unknown arm"):
        pair.run_id_for(job(toy, "A99"), o)


# ------------------------------------------------------------------------------------------- process plans
def _plan(toy, tmp_path, arm, **kw):
    o = opts_for(tmp_path, use_systemd=True, pin=True, solver_mem_gb=9.0, ea_mem_gb=6.0, cuopt_mem_gb=12.0)
    j = job(toy, arm, seed=1000, budget=300, **kw)
    cfg, _ = pair.effective_config(j, o)
    rd = tmp_path / "traces" / "raw" / "pytest" / pair.run_id_for(j, o)
    return {p.role: p for p in pair.plan(j, o, rd, "SPEC", cfg)}, o, rd


def test_plan_footprint_a0_a1(toy, tmp_path):
    procs, o, rd = _plan(toy, tmp_path, "A0")
    s = procs["solver"]
    assert s.cores == [5] and s.mem_gb == 9.0 and s.gpu_gb is None
    assert s.cmd[:3] == ["systemd-run", "--user", "--scope"]
    assert "MemoryMax=9216M" in s.cmd and "MemorySwapMax=0" in s.cmd
    assert s.cmd[s.cmd.index("taskset"):s.cmd.index("taskset") + 3] == ["taskset", "-c", "5"]
    assert "--inert" in s.argv and "--recv" not in s.argv and s.argv[s.argv.index("--mem-mb") + 1] == "8192"
    assert s.argv[s.argv.index("--run-id") + 1] == rd.name
    assert s.env_extra["OMP_NUM_THREADS"] == s.env_extra["OPENBLAS_NUM_THREADS"] == s.env_extra["MKL_NUM_THREADS"] == "1"
    assert s.env_extra["CUDA_VISIBLE_DEVICES"] == ""
    procs, _, _ = _plan(toy, tmp_path, "A1")
    assert procs["solver"].cores == [5] and procs["solver2"].cores == [6]
    s2 = procs["solver2"].argv
    assert s2[s2.index("--seed") + 1] == str(1000 + pair.A1_SEED_OFFSET) and s2[s2.index("--role-name") + 1] == "solver2"


def test_plan_footprint_partner_arms(toy, tmp_path):
    procs, _, rd = _plan(toy, tmp_path, "A2")
    c = procs["cuopt"]
    g = EAConfig().gpu_mem_gb                     # the EA pool = cuOpt's RMM limit (12 GB since 2026-10-06, D-003)
    assert c.cores == [1, 2, 3] and c.gpu_gb == g
    assert c.argv[c.argv.index("--cpu-threads") + 1] == "3" and c.argv[c.argv.index("--gpu-mem-gb") + 1] == str(g)
    assert c.env_extra["CUPY_GPU_MEMORY_LIMIT"] == str(int(g * 1024**3)) and "CUDA_VISIBLE_DEVICES" not in c.env_extra
    procs, _, rd = _plan(toy, tmp_path, "A3")
    s, e = procs["solver"].argv, procs["ea"]
    assert "--recv" in s and "--send" in s and "--inert" not in s
    assert s[s.index("--inject-band") + 1] == "0.01" and s[s.index("--inject-k") + 1] == "8"
    assert e.cores == [1, 2, 3] and e.argv[e.argv.index("--config") + 1] == str(rd / "config.yaml")
    assert e.env_extra["CUPY_GPU_MEMORY_LIMIT"] == str(int(g * 1024**3)) and "--run-id" not in e.argv
    procs, _, _ = _plan(toy, tmp_path, "A4")
    assert procs["ea"].gpu_gb is None and procs["ea"].env_extra["CUDA_VISIBLE_DEVICES"] == ""
    assert pair.effective_config(job(toy, "A4"))[0].backend == "numpy"
    procs, _, _ = _plan(toy, tmp_path, "A3r")
    assert "--inert" in procs["solver"].argv
    assert pair.effective_config(job(toy, "A3r"))[0].exchange.enabled is False


def test_ablation_files_set_matching_solver_flags(toy, tmp_path):
    procs, _, _ = _plan(toy, tmp_path, "A3", ablation="one_way_ea_to_solver")
    s = procs["solver"].argv
    assert "--recv" in s and "--send" not in s           # nothing flows from the solver to the EA
    procs, _, _ = _plan(toy, tmp_path, "A3", ablation="one_way_solver_to_ea")
    s = procs["solver"].argv
    assert "--send" in s and "--recv" not in s
    with pytest.raises(pair.RunError, match="unknown ablation"):
        pair.effective_config(job(toy, "A3", ablation="no_such_ablation"))


def test_a0star_needs_emphasis(toy, tmp_path):
    procs, _, _ = _plan(toy, tmp_path, "A0star", emphasis="aggressive")
    s = procs["solver"].argv
    assert s[s.index("--emphasis") + 1] == "aggressive"
    with pytest.raises(pair.RunError, match="emphasis"):
        _plan(toy, tmp_path, "A0star")


def test_job_budget_follows_d003(toy, tmp_path):
    o = opts_for(tmp_path, solver_mem_gb=9.0, ea_mem_gb=6.0, cuopt_mem_gb=12.0)
    b = pair.job_budget(job(toy, "A3"), o)
    g = EAConfig().gpu_mem_gb
    assert (b.host_caps_gb, b.gpu_caps_gb, b.cuda_procs, b.redis_gb) == (15.0, g, 1, 0.0)
    assert b.need_gb == pytest.approx(15 + g + 1 + gates.RESERVE_GB)
    o.mailbox = "redis"
    assert pair.job_budget(job(toy, "A2"), o).need_gb == pytest.approx(9 + 12 + g + 1 + 2 + gates.RESERVE_GB)
    o.cuopt_gpu_peak_gb = 2.5                            # the measured RMM peak replaces the limit
    assert pair.job_budget(job(toy, "A6"), o).gpu_caps_gb == 2.5
    assert pair.job_budget(job(toy, "A4"), o).cuda_procs == 0


def test_lane_must_fit(toy, tmp_path):
    o = opts_for(tmp_path, lane=pair.Lane("x5", (5,)))
    with pytest.raises(pair.RunError, match="cannot run A1"):
        pair.job_budget(job(toy, "A1"), o)
    with pytest.raises(pair.RunError, match="cannot run A3"):
        pair.job_budget(job(toy, "A3"), o)


def test_existing_run_hashes_are_pinned():
    """Adding an arm must not move any existing run id (run directories, the Experiment index and D-018's records
    name these hashes)."""
    o = pair.PairOptions()

    def h(arm, config=None, variant=None):
        j = dict(instance="data/miplib/inst/glass4.mps.gz", set="x", arm=arm, seed=0, budget=120, config=config)
        if variant:
            j["variant"] = variant
        return pair.run_hash(j, o)
    assert h("A6") == h("A6", "configs/itrea2-default.yaml") == "6e35f122f659"
    assert h("A2") == "125d69c76695" and h("A0") == "27f3171fac78"
    assert h("A9", "configs/itrea2-dev-rc-3way-bo1.yaml", "bo1") == "f67354be775d"
    assert h("A9", "configs/itrea2-dev-rc-3way-bo1s.yaml", "bo1s") == "62bf82b236f6"
    assert h("A3x", "configs/itrea2-dev-rc-3way.yaml") == "1d12ad490dd0"
    assert h("A9", "configs/itrea2-dev-rc-3way.yaml") == "0f011a2094a2"       # the A9 development default
    # A6m (D-020) under the full lane's settings
    full = pair.PairOptions(lane=pair.FULL_LANE, gpu_mem_gb=36.0, cuopt_mem_gb=24.0)
    a6m = dict(instance="data/miplib/inst/glass4.mps.gz", set="x", arm="A6m", seed=0, budget=600, config=None)
    assert pair.run_hash(a6m, full) == "2bdfb1d95b46"


def test_d021_run_hashes_are_pinned():
    """The new arms and the harness (D-021) get these ids; the harness enters the hash only when the job has one."""
    o = pair.PairOptions()

    def h(arm, harness=None, config=None):
        j = dict(instance="data/miplib/inst/glass4.mps.gz", set="x", arm=arm, seed=0, budget=600, config=config)
        if harness is not None:
            j["harness"] = harness
        return pair.run_hash(j, o)
    a9 = "configs/itrea2-dev-rc-3way.yaml"
    pinned = {("A6F", None): "21b799abb61f", ("A2F", None): "ba509eeea380", ("A6F", "ext"): "acc1ab3ad730",
              ("A2F", "ext"): "e0fe33e2f7be", ("A9", "ext"): "37577482704d", ("A6", "ext"): "66ccaffa7a42",
              ("A2", "ext"): "abf37c7fd0d3"}
    got = {k: h(k[0], k[1], a9 if k[0] == "A9" else None) for k in pinned}
    assert got == pinned, got
    m3 = {"preset": "ext", "cuopt_rmm_gb": 40, "cuopt_host_gb": 40}
    assert h("A6F", m3) == h("A6F", {"preset": "ext", "cuopt_rmm_gb": 40.0, "cuopt_host_gb": 40.0}) == "0069fefa1a21"
    assert h("A6F", {"preset": "ext", "cuopt_env_unset": False}) == "6d2a04a700a0"
    assert h("A6F", {"preset": "ext"}) == h("A6F", "ext") != h("A6F")
    assert h("A9", "ext", a9) != h("A9", None, a9) == "0f011a2094a2"


def test_a6m_identity_settings_and_lane(toy, tmp_path):
    """A6m (D-020): cuOpt as GAMS ran it — 24 threads and zero MIP gaps whatever the lane, every core of the whole-box
    lane, its own run hash (encoding both), refused on any other lane; the whole-box lane refuses every other arm."""
    o = opts_for(tmp_path, use_systemd=True, pin=True, lane=pair.FULL_LANE, cuopt_mem_gb=24.0, gpu_mem_gb=36.0)
    j = job(toy, "A6m", seed=0, budget=600)
    ss = pair.solver_settings(j, o, pair.effective_config(j, o)[0])
    assert ss == dict(arm="A6m", emphasis=None, gpu_mem_gb=36.0, cuopt_threads=24, mip_gap_zero=True)
    a6 = dict(j, arm="A6")
    assert "mip_gap_zero" not in pair.solver_settings(a6, o, pair.effective_config(a6, o)[0])
    assert pair.run_hash(j, o) != pair.run_hash(a6, o)
    assert pair.run_hash(j, o) != pair.run_hash(j, opts_for(tmp_path, lane=pair.FULL_LANE, gpu_mem_gb=12.0))
    assert pair.run_hash(j, o) == pair.run_hash(j, opts_for(tmp_path, lane=pair.FULL_LANE, gpu_mem_gb=36.0,
                                                            cuopt_threads=24))
    cfg, _ = pair.effective_config(j, o)
    rd = tmp_path / "traces" / "raw" / "pytest" / pair.run_id_for(j, o)
    (c,) = pair.plan(j, o, rd, "SPEC", cfg)
    assert c.cores == list(range(20)) and c.mem_gb == 24.0 and c.gpu_gb == 36.0
    assert c.argv[c.argv.index("--cpu-threads") + 1] == "24" and "--mip-gap-zero" in c.argv
    assert c.argv[c.argv.index("--gpu-mem-gb") + 1] == "36.0" and "--send" not in c.argv
    assert c.cmd[c.cmd.index("taskset") + 2] == ",".join(map(str, range(20)))
    b = pair.job_budget(j, o)
    assert (b.host_caps_gb, b.gpu_caps_gb, b.cuda_procs) == (24.0, 36.0, 1)
    assert b.need_gb == pytest.approx(24 + 36 + 1 + gates.RESERVE_GB) == 81.0       # D-020's 81 GB (file mailbox)
    with pytest.raises(pair.RunError, match="whole-box lane"):
        pair.job_budget(j, opts_for(tmp_path, gpu_mem_gb=36.0))                     # the GPU lane g0
    with pytest.raises(pair.RunError, match="runs only A6m"):
        pair.job_budget(job(toy, "A6"), o)
    with pytest.raises(pair.RunError, match="fixed by the arm"):
        pair.run_hash(j, opts_for(tmp_path, lane=pair.FULL_LANE, cuopt_threads=3))
    assert not pair.GPU_LANE.fits(pair.ARMS["A6m"]) and pair.FULL_LANE.fits(pair.ARMS["A6m"])
    # appended after A3x, and A6F / A2F (D-021) after it: the canonical order of the earlier arms is unchanged
    assert pair.ARM_ORDER[pair.ARM_ORDER.index("A3x") + 1:] == ("A6m", "A6F", "A2F")


def _launch_env_before_a6m(p, base: dict) -> dict:
    """The child environment as the runner built it before A6m (530916c): the runner's environment without CuPy's
    pool limit, then the process's env_extra."""
    env = dict(base)
    env.pop("CUPY_GPU_MEMORY_LIMIT", None)
    env.update(p.env_extra)
    return env


def test_process_environments_unchanged_but_a6m(toy, tmp_path):
    """Every arm but A6m launches with the environment it had before A6m, byte for byte (the three thread variables
    = 1); A6m's cuOpt (the whole-lane process) gets them REMOVED, although the runner's own environment has them
    (review 2026-10-11: GAMS ran cuOpt with them unset; libcuopt links libomp)."""
    base = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                CUPY_GPU_MEMORY_LIMIT="123", PATH="/usr/bin")
    o = opts_for(tmp_path, use_systemd=True, pin=True)
    for arm in pair.ARMS:
        if arm == "A6m":
            continue
        j = job(toy, arm, seed=1000, budget=300, emphasis="aggressive" if arm == "A0star" else None)
        cfg, _ = pair.effective_config(j, o)
        for p in pair.plan(j, o, tmp_path / "rd" / pair.run_id_for(j, o), "SPEC", cfg):
            assert p.env_unset == () and pair.launch_env(p, base) == _launch_env_before_a6m(p, base), (arm, p.role)
            assert all(p.env_extra[k] == "1" for k in pair.THREAD_VARS)
    a2 = job(toy, "A2", seed=1000, budget=300)
    cu = {p.role: p for p in pair.plan(a2, o, tmp_path / "rd" / "x", "SPEC", pair.effective_config(a2, o)[0])}["cuopt"]
    assert cu.env_extra == {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                            "PYTHONUNBUFFERED": "1", "ITREA_RUN_ID": "x", "ITREA_ROLE": "cuopt",
                            "CUPY_CACHE_DIR": str(tmp_path / "cupy-cache"), "CUPY_GPU_MEMORY_LIMIT": str(12 * 1024**3),
                            "ITREA_GPU_MEM_GB": "12.0"}
    fo = opts_for(tmp_path, lane=pair.FULL_LANE, gpu_mem_gb=36.0, cuopt_mem_gb=24.0)
    m = job(toy, "A6m", seed=0, budget=600)
    (c,) = pair.plan(m, fo, tmp_path / "rd" / "x", "SPEC", pair.effective_config(m, fo)[0])
    assert c.env_unset == pair.THREAD_VARS and not set(pair.THREAD_VARS) & set(c.env_extra)
    env = pair.launch_env(c, base)
    assert not set(pair.THREAD_VARS) & set(env) and env["CUPY_GPU_MEMORY_LIMIT"] == str(36 * 1024**3)
    assert env["PATH"] == "/usr/bin" and env["ITREA_ROLE"] == "cuopt" and env["PYTHONUNBUFFERED"] == "1"


def test_a6m_child_inherits_no_thread_variables(toy, fake, tmp_path, monkeypatch):
    """End to end with the fake process: the runner's own environment has the thread variables (set at import), A6m's
    child does not, A6's child does; run.json records env_unset."""
    monkeypatch.setenv("FAKE_ENV_OUT", str(tmp_path / "env"))
    for k in pair.THREAD_VARS:
        monkeypatch.setenv(k, "1")
    rec = pair.run_pair(job(toy, "A6m", seed=0, budget=0.5),
                        opts_for(tmp_path, fake, lane=pair.FULL_LANE, gpu_mem_gb=36.0, cuopt_mem_gb=24.0))
    assert rec["status"] == "completed", rec["reasons"]
    seen = json.loads((tmp_path / "env.cuopt").read_text())
    assert all(seen[k] is None for k in pair.THREAD_VARS) and seen["CUPY_GPU_MEMORY_LIMIT"] == str(36 * 1024**3)
    assert rec["processes"][0]["env_unset"] == list(pair.THREAD_VARS)
    rec6 = pair.run_pair(job(toy, "A6", seed=0, budget=0.5), opts_for(tmp_path, fake))
    assert rec6["status"] == "completed" and rec6["processes"][0]["env_unset"] == []
    assert all(json.loads((tmp_path / "env.cuopt").read_text())[k] == "1" for k in pair.THREAD_VARS)
    assert not pair.FULL_LANE.fits(pair.ARMS["A2"]) and pair.GPU_LANE.fits(pair.ARMS["A6"])


# ------------------------------------------------------------------------------------------- fake processes
def test_fake_pair_completes_and_merges(toy, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BASE_SOLVER2", "90")      # minimisation: solver2 is better from the start
    rec = pair.run_pair(job(toy, "A1", budget=1.0, position=2, block="b1"), opts_for(tmp_path, fake))
    assert rec["status"] == "completed", rec["reasons"]
    assert rec["exit_codes"] == {"solver": 0, "solver2": 0}
    rd = tmp_path / "traces" / "raw" / "pytest" / rec["run_id"]
    on_disk = json.loads((rd / "run.json").read_text())
    assert on_disk["status"] == "completed" and on_disk["position"] == 2 and on_disk["block"] == "b1"
    assert json.loads((rd / "order.json").read_text()) == [f"s{j}" for j in range(30)]
    tr = rec["trajectories"]
    assert tr["sense"] == 1 and tr["portfolio"][-1][2] == "solver2"
    first2 = next(i for i, p in enumerate(tr["portfolio"]) if p[2] == "solver2")
    assert all(p[2] == "solver2" for p in tr["portfolio"][first2:])               # solver never catches up
    objs = [p[1] for p in tr["portfolio"]]
    assert objs == sorted(objs, reverse=True) and len(set(objs)) == len(objs)      # strictly improving
    assert all(p[2] == "solver" for p in tr["solver"])
    assert {p["role"]: p["cores"] for p in rec["processes"]} == {"solver": [5], "solver2": [6]}
    assert all(p["startup_s"] is not None and p["cpu_user_s"] is not None for p in rec["processes"])
    assert rec["env"]["versions"]["python"] and "git" in rec["env"]
    assert not (tmp_path / "mb" / rec["run_id"]).exists()                        # mailbox removed


def test_exit_code_1_marks_the_run_failed(toy, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT_SOLVER", "1")
    rec = pair.run_pair(job(toy, "A0", budget=0.5), opts_for(tmp_path, fake))
    assert rec["status"] == "failed" and rec["exit_codes"] == {"solver": 1}
    assert "solver rc=1" in rec["reasons"]


def test_crash_mid_run_stops_the_partner(toy, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CRASH_SOLVER2", "1")
    rec = pair.run_pair(job(toy, "A1", budget=5.0), opts_for(tmp_path, fake))
    assert rec["status"] == "failed" and rec["exit_codes"] == {"solver": 0, "solver2": 1}
    assert rec["stop"]["reason"].startswith("solver2 ended rc=1") and rec["wall_s"] < 4.0
    crashed = next(p for p in rec["processes"] if p["role"] == "solver2")
    assert crashed["exit_t"] is not None and crashed["exit_t"] < 2.0               # truncation point on file


def test_early_normal_end_does_not_stop_the_partner(toy, fake, tmp_path, monkeypatch):
    # a partner that ends at its own limit (exit 0, not a proven outcome) must not truncate the other's run
    monkeypatch.setenv("FAKE_END_AFTER_SOLVER2", "0.3")
    rec = pair.run_pair(job(toy, "A1", budget=2.0), opts_for(tmp_path, fake))
    assert rec["status"] == "completed" and rec["stop"] is None
    ex = {p["role"]: p["exit_t"] for p in rec["processes"]}
    assert ex["solver2"] < 1.0 and ex["solver"] >= 1.9


def test_proven_optimum_stops_the_partner(toy, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_END_AFTER_SOLVER2", "0.3")
    monkeypatch.setenv("FAKE_REASON_SOLVER2", "optimal")
    rec = pair.run_pair(job(toy, "A1", budget=5.0), opts_for(tmp_path, fake))
    assert rec["status"] == "completed" and rec["stop"]["reason"] == "solver2 ended rc=0 (optimal)"
    assert rec["wall_s"] < 3.0


def test_run_id_does_not_depend_on_the_lane(toy, tmp_path):
    a = opts_for(tmp_path)
    b = opts_for(tmp_path, lane=pair.Lane("other", (9, 15), (10, 11, 12, 13), gpu=True))
    for arm in ("A2", "A6", "A3", "A0"):
        assert pair.run_id_for(job(toy, arm), a) == pair.run_id_for(job(toy, arm), b)


def test_exit_before_ready_fails_without_release(toy, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT_EARLY_SOLVER2", "3")
    monkeypatch.setenv("FAKE_SUMMARY_EARLY_SOLVER2", "1")      # an arm failure (summary written), not the launcher
    rec = pair.run_pair(job(toy, "A1", budget=5.0), opts_for(tmp_path, fake))
    assert rec["status"] == "failed" and "t0" not in rec
    assert any("solver2 exited rc=3 before ready" in r for r in rec["reasons"])
    assert rec["exit_codes"]["solver"] == 0                                         # stopped at the barrier


def test_abort_voids_the_run(toy, fake, tmp_path):
    ev = threading.Event()
    o = opts_for(tmp_path, fake, abort=ev, abort_reason=lambda: "thermal: 93 °C")
    threading.Timer(0.8, ev.set).start()
    rec = pair.run_pair(job(toy, "A0", budget=10.0), o)
    assert rec["status"] == "void" and rec["void_cause"] == "abort" and rec["wall_s"] < 8.0
    assert rec["exit_codes"] == {"solver": 0}


def test_test_split_only_through_the_pool(toy, fake, tmp_path):
    j = dict(job(toy, "A0"), split="test")
    with pytest.raises(pair.RunError, match="only through the pool"):
        pair.run_pair(j, opts_for(tmp_path, fake))
    assert not pair.run_dir_for(j, opts_for(tmp_path, fake)).exists()


def test_stale_mailbox_is_refused(toy, fake, tmp_path):
    o = opts_for(tmp_path, fake)
    j = job(toy, "A0")
    stale = tmp_path / "mb" / pair.run_id_for(j, o)
    stale.mkdir(parents=True)
    (stale / "start").write_text('{"t0": 1.0}')
    with pytest.raises(pair.StaleMailboxError):
        pair.run_pair(j, o)
    assert not pair.run_dir_for(j, o).exists()                                      # nothing launched or written


def test_stale_run_dir_is_refused(toy, fake, tmp_path):
    o = opts_for(tmp_path, fake)
    j = job(toy, "A0", budget=0.3)
    assert pair.run_pair(j, o)["status"] == "completed"
    with pytest.raises(pair.StaleRunError):
        pair.run_pair(j, o)


def _redis_ok() -> bool:
    from itrea.experiments import exchange_ctl
    return exchange_ctl.ping()


@pytest.mark.skipif(not _redis_ok(), reason="itrea-redis is not answering on 127.0.0.1:6390")
def test_redis_mailbox_run_and_cleanup(toy, fake, tmp_path):
    from itrea.experiments import exchange_ctl
    o = opts_for(tmp_path, fake, mailbox="redis")
    j = job(toy, "A1", seed=int(uuid.uuid4().int % 10**6), budget=0.8)
    rec = pair.run_pair(j, o)
    assert rec["status"] == "completed", rec["reasons"]
    assert rec["mailbox"]["spec"].startswith("redis://127.0.0.1:6390/0?run=")
    assert rec["run_id"] not in exchange_ctl.run_ids()                              # keys deleted at the end
    assert rec["memory_budget"]["redis_gb"] == gates.REDIS_CAP_GB


@pytest.mark.skipif(not _redis_ok(), reason="itrea-redis is not answering on 127.0.0.1:6390")
def test_redis_stale_keys_are_refused(toy, fake, tmp_path):
    from itrea.experiments import exchange_ctl
    o = opts_for(tmp_path, fake, mailbox="redis")
    j = job(toy, "A0", seed=int(uuid.uuid4().int % 10**6))
    rid = pair.run_id_for(j, o)
    c = exchange_ctl.client()
    c.set(f"itrea:{rid}:start", '{"t0": 1.0}', ex=300)
    try:
        with pytest.raises(pair.StaleMailboxError):
            pair.run_pair(j, o)
        assert not pair.run_dir_for(j, o).exists()
        done = pair.clear_stale(j, o, wait_s=1.0)                                    # the pool's cleanup path
        assert done["mailbox_removed"] and rid not in exchange_ctl.run_ids()
    finally:
        exchange_ctl.flush(rid)


# ------------------------------------------------------------------------------------------- real processes
_SCIP = None


def scip_implemented() -> bool:
    global _SCIP
    if _SCIP is None:
        _SCIP = _implemented("itrea.solvers.scip_run")
    return _SCIP


@pytest.mark.solver
@pytest.mark.parametrize("arm", ["A0", "A1", "SCREEN"])
def test_real_scip_arms_on_a_toy(arm, tmp_path):
    if not scip_implemented():
        pytest.skip("itrea.solvers.scip_run is not implemented yet")
    p = knapsack(20, seed=3)
    inst = write_mps(p, tmp_path / "knap20.mps")
    rec = pair.run_pair(job(inst, arm, seed=900, budget=10.0), opts_for(tmp_path, solver_mem_gb=2.0, grace_s=10.0))
    assert rec["status"] == "completed", (rec["reasons"], rec["exit_codes"])
    assert set(rec["exit_codes"].values()) == {0}
    best = rec["trajectories"]["portfolio"][-1][1]
    assert best == pytest.approx(p.toy_optimum)                                    # maximisation, DP optimum
    rd = tmp_path / "traces" / "raw" / "pytest" / rec["run_id"]
    for role in rec["exit_codes"]:
        assert (rd / f"{role}.summary.json").exists() and (rd / f"{role}.events.jsonl").exists()


@pytest.mark.gpu
@pytest.mark.solver
def test_real_cuopt_alone_a6(tmp_path):
    if not _cupy_ok():
        pytest.skip("CuPy / CUDA device not available")
    if not _implemented("itrea.solvers.cuopt_run"):
        pytest.skip("itrea.solvers.cuopt_run is not implemented yet")
    inst = write_mps(knapsack(20, seed=3), tmp_path / "knap20.mps")
    o = opts_for(tmp_path, gpu_mem_gb=1.0, cuopt_mem_gb=3.5, grace_s=10.0, gpu_health=True)
    rec = pair.run_pair(job(inst, "A6", seed=1000, budget=10.0), o)
    log = (tmp_path / "traces" / "raw" / "pytest" / rec["run_id"] / "cuopt.log").read_text()
    if rec["status"] == "failed" and "Memory allocation failed" in log:
        pytest.skip("cuOpt needs more than the 1 GB GPU cap allowed in tests (RMM: Memory allocation failed)")
    assert rec["status"] in ("completed", "void"), (rec["reasons"], log[-500:])
    assert rec["gpu_arm"] and rec["gpu_health"]["n"] >= 0
    p = rec["processes"][0]
    assert p["gpu_limit_gb"] == 1.0 and p["env"]["CUPY_GPU_MEMORY_LIMIT"] == str(1024**3)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.solver
def test_real_a3_pair(tmp_path):
    if not _cupy_ok():
        pytest.skip("CuPy / CUDA device not available")
    if not (scip_implemented() and _implemented("itrea.ea.run")):
        pytest.skip("itrea.ea.run (or itrea.solvers.scip_run) is not implemented yet")
    inst = write_mps(knapsack(20, seed=3), tmp_path / "knap20.mps")
    o = opts_for(tmp_path, gpu_mem_gb=1.0, grace_s=15.0, gpu_health=True, ready_timeout_s=300.0)
    rec = pair.run_pair(job(inst, "A3", seed=1000, budget=8.0), o)
    assert rec["status"] in ("completed", "void"), (rec["reasons"], rec["exit_codes"])
    assert set(rec["exit_codes"]) == {"solver", "ea"}
    rd = tmp_path / "traces" / "raw" / "pytest" / rec["run_id"]
    assert EAConfig.from_yaml(rd / "config.yaml").hash() == rec["config"]["ea_hash"]


def test_run_record_has_the_box_load_and_heat(toy, fake, tmp_path, monkeypatch):
    """run.json: the 1-min load average in gates_before / gates_after (gates.snapshot) and the hottest ACPI zone seen
    while the clock ran (`acpi_during`, read every ACPI_SAMPLE_S) — what A6m's whole-box runs are judged against."""
    from itrea.experiments import gpu_health
    temps = iter([71.5, 74.0, 73.0] + [72.0] * 100)
    monkeypatch.setattr(gates, "acpi_max_c", lambda *a, **k: next(temps))
    monkeypatch.setattr(gates, "lease", lambda *a, **k: None)
    monkeypatch.setattr(gates, "serving_state", lambda: {})
    monkeypatch.setattr(gates, "co_running", lambda **k: [])
    monkeypatch.setattr(gpu_health, "compute_apps", lambda *a, **k: [])          # no nvidia-smi in a CPU test
    monkeypatch.setattr(pair, "ACPI_SAMPLE_S", 0.2)
    rec = pair.run_pair(job(toy, "A0", budget=1.0), opts_for(tmp_path, fake, record_snapshots=True))
    assert rec["status"] == "completed"
    assert "loadavg_1m" in rec["gates_before"] and "loadavg_1m" in rec["gates_after"]
    a = rec["acpi_during"]
    assert a["samples"] >= 2 and a["max_c"] >= 73.0 and a["period_s"] == 0.2
    assert pair.run_pair(job(toy, "A0", seed=2, budget=0.3), opts_for(tmp_path, fake))["acpi_during"] is None
    assert gates.loadavg_1m() is None or gates.loadavg_1m() >= 0


# ================================================================================================ D-021 (ITX-1)
def test_f_arms_take_the_gpu_lanes_cores(toy, tmp_path):
    """A6F = cuOpt alone on all of F (the GPU lane's cores: X925 5–6 + A725 1–3), num_cpu_threads = |F|; A2F = SCIP on
    F's first X925 core and cuOpt on the other |F| - 1 cores with |F| - 1 threads, no exchange. F comes from the lane,
    so another lane gives another F (and another run id)."""
    procs, o, rd = _plan(toy, tmp_path, "A6F")
    (c,) = procs.values()
    F = sorted(pair.GPU_LANE.x925 + pair.GPU_LANE.a725)
    assert c.cores == F == [1, 2, 3, 5, 6] and c.argv[c.argv.index("--cpu-threads") + 1] == str(len(F))
    assert c.mem_gb == 12.0 and c.gpu_gb == EAConfig().gpu_mem_gb and "--mip-gap-zero" not in c.argv
    assert "--send" not in c.argv and c.env_unset == () and c.env_extra["OMP_NUM_THREADS"] == "1"
    assert c.cmd[c.cmd.index("taskset") + 2] == "1,2,3,5,6"
    procs, _, _ = _plan(toy, tmp_path, "A2F")
    s, c = procs["solver"], procs["cuopt"]
    assert s.cores == [5] and "--inert" in s.argv and c.cores == [1, 2, 3, 6]
    assert c.argv[c.argv.index("--cpu-threads") + 1] == "4" and "--send" not in c.argv
    wide = pair.Lane("w", (5, 6, 7), (1, 2, 3, 4), gpu=True)
    j = job(toy, "A6F", seed=0, budget=600)
    assert pair.run_hash(j, opts_for(tmp_path, lane=wide)) != pair.run_hash(j, opts_for(tmp_path))
    cfg, _ = pair.effective_config(j, opts_for(tmp_path, lane=wide))
    (cw,) = pair.plan(j, opts_for(tmp_path, lane=wide), tmp_path / "x", "S", cfg)
    assert cw.cores == [1, 2, 3, 4, 5, 6, 7] and cw.argv[cw.argv.index("--cpu-threads") + 1] == "7"
    with pytest.raises(pair.RunError, match="contradicts"):
        pair.run_hash(j, opts_for(tmp_path, cuopt_threads=3))
    assert not pair.Lane("c1", (7, 8)).fits(pair.ARMS["A6F"]) and not pair.FULL_LANE.fits(pair.ARMS["A2F"])
    assert pair.ARMS["A6F"].lane == pair.ARMS["A2F"].lane == "partner"


def test_harness_presets_overrides_and_refusals(toy):
    ext = pair.resolve_harness("ext")
    assert ext == dict(cuopt_gap_zero=True, cuopt_env_unset=True, cuopt_rmm_gb=24.0, cuopt_host_gb=24.0,
                       cuda_cache=True)
    assert pair.resolve_harness(None) is None and pair.resolve_harness({}) == ext
    assert pair.resolve_harness({"preset": "ext", "cuopt_rmm_gb": 40, "cuopt_host_gb": 40})["cuopt_rmm_gb"] == 40.0
    assert pair.resolve_harness({"cuopt_env_unset": False})["cuopt_env_unset"] is False
    for bad, text in (("int", "unknown harness"), ({"preset": "nope"}, "unknown harness"),
                      ({"cuopt_rmm": 40}, "unknown harness keys"), ({"cuopt_rmm_gb": 0}, "positive"),
                      ({"cuopt_rmm_gb": True}, "positive"), ({"cuda_cache": "yes"}, "true or false"), (3, "preset")):
        with pytest.raises(pair.RunError, match=text):
            pair.resolve_harness(bad)
    with pytest.raises(pair.RunError, match="D-020's own settings"):
        pair.job_harness(job(toy, "A6m", harness="ext"))


def test_harness_ext_plans_every_cuopt_process(toy, tmp_path):
    """Under the ext harness (D-021 R13, R14, R15, R29): cuOpt gets gaps 0, no thread variables, its own RMM limit
    (24 GB, separate from the EA's 12-GB pool) and host MemoryMax 24 GB; every process gets the project's CUDA JIT
    cache. Without the harness nothing changes."""
    base = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    o = opts_for(tmp_path, use_systemd=True, pin=True, solver_mem_gb=9.0, ea_mem_gb=6.0, cuopt_mem_gb=12.0,
                 cuda_cache_dir=tmp_path / "cuda")
    for arm in ("A9", "A6F", "A2F", "A6", "A2"):
        cfgf = "configs/itrea2-dev-rc-3way.yaml" if arm == "A9" else None
        j = job(toy, arm, seed=0, budget=600, config=cfgf, harness="ext")
        cfg, _ = pair.effective_config(j, o)
        procs = {p.role: p for p in pair.plan(j, o, tmp_path / "rd" / pair.run_id_for(j, o), "S", cfg)}
        c = procs["cuopt"]
        assert c.mem_gb == 24.0 and c.gpu_gb == 24.0 and "--mip-gap-zero" in c.argv, arm
        assert c.argv[c.argv.index("--gpu-mem-gb") + 1] == "24.0" and "MemoryMax=24576M" in c.cmd
        assert c.env_unset == pair.THREAD_VARS and not set(pair.THREAD_VARS) & set(pair.launch_env(c, base))
        assert c.env_extra["CUPY_GPU_MEMORY_LIMIT"] == str(24 * 1024**3)
        for p in procs.values():
            assert p.env_extra["CUDA_CACHE_PATH"] == str(tmp_path / "cuda"), (arm, p.role)
            assert p.env_extra["CUDA_CACHE_MAXSIZE"] == "4294967296"
            if p.spec.kind != "cuopt":
                assert p.env_unset == () and p.env_extra["OMP_NUM_THREADS"] == "1"
        if arm == "A9":
            ea = procs["ea"]
            assert ea.gpu_gb == 12.0 and ea.mem_gb == 6.0 and "--send" in c.argv     # the EA keeps its own pool
        j0 = dict(j)
        j0.pop("harness")
        p0 = {p.role: p for p in pair.plan(j0, o, tmp_path / "rd" / "x", "S", pair.effective_config(j0, o)[0])}
        assert "CUDA_CACHE_PATH" not in p0["cuopt"].env_extra and p0["cuopt"].env_unset == ()
    kept = job(toy, "A6F", seed=0, budget=120, harness={"preset": "ext", "cuopt_env_unset": False})
    (c,) = pair.plan(kept, o, tmp_path / "x", "S", pair.effective_config(kept, o)[0])
    assert c.env_unset == () and c.env_extra["OMP_NUM_THREADS"] == "1" and "--mip-gap-zero" in c.argv
    with pytest.raises(pair.RunError, match="measured cuOpt peak"):
        pair.job_budget(job(toy, "A6F", harness="ext"), opts_for(tmp_path, cuopt_gpu_peak_gb=10.5))


def test_harness_memory_gate_follows_d003(toy, tmp_path):
    """D-021 R29's gate arithmetic, from D-003's formula and the harness's caps (file mailbox; Redis adds 2 GB):
    A9 ≈ 97 (99), A2F ≈ 78 (80), A6F ≈ 69 (71) GB, M3's A6F at RMM 40 / host 40 = 101 GB."""
    o = opts_for(tmp_path, solver_mem_gb=9.0, ea_mem_gb=6.0, cuopt_mem_gb=12.0)
    r = opts_for(tmp_path, solver_mem_gb=9.0, ea_mem_gb=6.0, cuopt_mem_gb=12.0, mailbox="redis")

    def need(arm, harness, opts):
        return pair.job_budget(job(toy, arm, seed=0, budget=600, harness=harness,
                                   config="configs/itrea2-dev-rc-3way.yaml" if arm == "A9" else None), opts).need_gb
    assert (need("A9", "ext", o), need("A9", "ext", r)) == (97.0, 99.0)
    assert (need("A2F", "ext", o), need("A2F", "ext", r)) == (78.0, 80.0)
    assert (need("A6F", "ext", o), need("A6F", "ext", r)) == (69.0, 71.0)
    assert need("A6F", {"preset": "ext", "cuopt_rmm_gb": 40, "cuopt_host_gb": 40}, o) == 101.0
    assert need("A9", None, o) == 9 + 6 + 12 + 12 + 12 + 2 + gates.RESERVE_GB         # unchanged without a harness


def test_named_scopes(toy, tmp_path):
    """D-021 R27: every process runs in a scope named itrea-run-<run-hash>-<role>-<epoch> (never in argv, never in a
    hash), so host tools such as thermal-watch can find the project's processes."""
    assert pair.scope_unit("glass4__A9__s0__600s__37577482704d", "cuopt", 1791716000) == \
        "itrea-run-37577482704d-cuopt-1791716000"
    o = opts_for(tmp_path, use_systemd=True, pin=True)
    j = job(toy, "A9", seed=0, budget=600, config="configs/itrea2-dev-rc-3way.yaml")
    rid = pair.run_id_for(j, o)
    procs = pair.plan(j, o, tmp_path / "rd" / rid, "S", pair.effective_config(j, o)[0], epoch=1791716000)
    h = rid.rsplit("__", 1)[-1]
    for p in procs:
        assert p.scope_unit == f"itrea-run-{h}-{p.role}-1791716000"
        assert p.cmd[:5] == ["systemd-run", "--user", "--scope", "--quiet", f"--unit={p.scope_unit}"]
        assert not any("--unit" in a for a in p.argv)
    assert len({p.scope_unit for p in procs}) == 3
    assert all(p.scope_unit is None for p in pair.plan(j, opts_for(tmp_path), tmp_path / "x", "S",
                                                       pair.effective_config(j, o)[0]))
