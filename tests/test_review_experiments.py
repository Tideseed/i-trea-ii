"""Regression tests for the 2026-10-06 review of `itrea.experiments` (screening and safety lenses): the memory gate's
admission (waiting lanes never count, admitted use credited, requeue instead of halt), the nine screening lanes, every
gate re-checked after a wait, the freeze and harness checks for screening and test jobs, toy-only exemptions, orphan
cleanup on resume, runner-stopped processes, the abort kill, fail-closed serving and GPU-health gates, the stage-2
table, and the never-split estimate.

Toy models and fake role processes only; no lease file, no systemd scope, no real solver; the runner lock is always a
tmp file."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from itrea.benchmarks.toy import set_cover, write_mps
from itrea.experiments import gates, gpu_health, pair, pool
from itrea.experiments import make_jobs as mj

# a role process speaking the barrier protocol; per-role knobs: FAKE_REASON_<R> + FAKE_END_AFTER_<R> (end with that
# exit reason), FAKE_IGNORE_STOP_<R> (run to the budget whatever the mailbox says), FAKE_IGNORE_TERM_<R> (SIGTERM only
# sets a flag, as in cuOpt), FAKE_CRASH_<R> (exit code at the 3rd incumbent), FAKE_EXIT_EARLY_<R> (exit before ready,
# FAKE_SUMMARY_EARLY_<R>: after writing a summary), FAKE_SIGKILL_EARLY_<R> (die by a signal before ready),
# FAKE_OVERRUN_<R> (sleep this long after its own budget: a process the runner must kill)
FAKE_PROC = r'''
import argparse, os, signal, sys, time
from itrea.exchange.redis_mailbox import open_mailbox
from itrea.trace import Trace
ap = argparse.ArgumentParser()
for a in ("--instance", "--seed", "--budget", "--run-dir", "--mailbox", "--sol-out", "--arm", "--mem-mb",
          "--role-name", "--config", "--cpu-threads", "--inject-band", "--inject-k", "--emphasis", "--run-id",
          "--gpu-mem-gb"):
    ap.add_argument(a)
for f in ("--recv", "--send", "--inert", "--no-barrier"):
    ap.add_argument(f, action="store_true")
a = ap.parse_args()
role = a.role_name or os.environ.get("ITREA_ROLE", "solver")
R = role.upper()
E = lambda k: os.environ.get(f"FAKE_{k}_{R}")
if E("IGNORE_TERM"):
    signal.signal(signal.SIGTERM, lambda *x: None)
mb = open_mailbox(a.mailbox, role, run_id=a.run_id)
tr = Trace(a.run_dir, role, arm=a.arm, seed=int(a.seed))
if E("SIGKILL_EARLY"):
    os.kill(os.getpid(), signal.SIGKILL)
if E("EXIT_EARLY"):
    if E("SUMMARY_EARLY"):
        tr.close(exit_reason="exception")
    sys.exit(int(E("EXIT_EARLY")))
mb.ready()
try:
    t0 = mb.wait_start(timeout=60)
except RuntimeError:
    sys.exit(0)
tr.start(t0)
k = 0
while time.monotonic() - t0 < float(a.budget) and (E("IGNORE_STOP") or not mb.stopped()):
    tr.incumbent(100.0 - k, source="self")
    k += 1
    time.sleep(0.05)
    if E("CRASH") and k == 3:
        sys.exit(int(E("CRASH")))
    if E("END_AFTER") and time.monotonic() - t0 > float(E("END_AFTER")):
        tr.close(exit_reason=E("REASON") or "node_limit")
        sys.exit(0)
if E("OVERRUN"):
    time.sleep(float(E("OVERRUN")))
tr.close(exit_reason="stop" if mb.stopped() else "budget")
sys.exit(0)
'''

# an orphan of a dead runner: ignores SIGTERM and the mailbox, never gets ready
ORPHAN = r'''
import signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(300)
'''

RUNNER = r'''
import json, sys
from pathlib import Path
from itrea.experiments import pair
job, tmp, fake = json.loads(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])
o = pair.PairOptions(traces_dir=tmp / "traces", mailbox_root=tmp / "mb", use_systemd=False, pin=False,
                     gpu_health=False, record_snapshots=False, solver_mem_gb=1.0, cupy_cache_dir=tmp / "cupy",
                     ready_timeout_s=300.0, module_override={"scip": fake})
pair.run_pair(job, o)
'''


@pytest.fixture
def toy(tmp_path) -> Path:
    return write_mps(set_cover(20, 12, seed=1), tmp_path / "toyrev.mps")


@pytest.fixture
def fake(tmp_path) -> list[str]:
    p = tmp_path / "fake_proc.py"
    p.write_text(FAKE_PROC)
    return [sys.executable, str(p)]


def popts(tmp_path, fake=None, **kw) -> pair.PairOptions:
    d = dict(traces_dir=tmp_path / "traces", mailbox_root=tmp_path / "mb", use_systemd=False, pin=False, grace_s=1.0,
             kill_after_s=10.0, gpu_health=False, record_snapshots=False, solver_mem_gb=1.0, ea_mem_gb=1.0,
             cuopt_mem_gb=1.0, cupy_cache_dir=tmp_path / "cupy", log=lambda m: None)
    if fake:
        d["module_override"] = {k: fake for k in ("scip", "highs", "cuopt", "ea")}
    d.update(kw)
    return pair.PairOptions(**d)


def poolopts(tmp_path, fake=None, po=None, **kw) -> pool.PoolOptions:
    d = dict(require_lease=False, logs_dir=tmp_path / "logs", lock_path=tmp_path / "runner.lock",
             thermal_start_c=500.0, watchdog=False, root=tmp_path, pair_opts=po or popts(tmp_path, fake),
             mem_poll_s=0.05)
    d.update(kw)
    return pool.PoolOptions(**d)


def _jobs(instances, arms, seeds=(1,), budget=0.3, split="smoke"):
    out = []
    for inst in instances:
        for s in seeds:
            for a in arms:
                n = Path(inst).stem
                out.append(dict(instance=str(inst), set="pytest", name=n, arm=a, seed=s, budget=budget, config=None,
                                overrides={}, split=split, stratum="toy", block=f"{n}__s{s}"))
    return out


def _toys(tmp_path, k):
    return [write_mps(set_cover(20, 12, seed=i), tmp_path / f"toyq{i}.mps") for i in range(k)]


def _stub_run_pair(duration, log):
    """A stand-in for pair.run_pair that only sleeps: records (job block, start, end) and the concurrency."""
    lock = threading.Lock()
    state = dict(now=0, max=0)

    def run_pair(job, opts):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        t = time.monotonic()
        rd = pair.run_dir_for(job, opts)
        rd.mkdir(parents=True)
        rec = dict(run_id=rd.name, status="completed", reasons=[], arm=job["arm"], processes=[])
        (rd / "run.json").write_text(json.dumps(rec))
        time.sleep(duration(job) if callable(duration) else duration)
        with lock:
            state["now"] -= 1
        log.append((job["block"], t, time.monotonic()))
        return rec
    return run_pair, state


# ================================================================ F2 (screening): admission, lanes, requeue
def test_screen_layout_has_nine_lanes_that_fit_the_gate(tmp_path, toy):
    assert len(pool.SCREEN_LANES) == 9
    assert all(len(l.x925) == 1 and l.x925[0] in (5, 6, 7, 8, 9, 15, 16, 17, 18, 19) for l in pool.SCREEN_LANES)
    assert len({l.x925[0] for l in pool.SCREEN_LANES}) == 9 and not any(l.gpu for l in pool.SCREEN_LANES)
    one = pair.job_budget(dict(instance=str(toy), set="s", arm="SCREEN", seed=900, budget=300),
                          pair.PairOptions(lane=pool.SCREEN_LANES[0]))
    total = one
    for _ in range(8):
        total = total + one
    assert one.host_caps_gb == 9.0 and total.need_gb == pytest.approx(9 * 9 + gates.RESERVE_GB)   # 101 GB < ~108
    p = pool.Pool(_jobs([toy], ["SCREEN"]), poolopts(tmp_path, layout="screen", cpu_lanes=10))
    with pytest.raises(pool.PoolError, match="9 screening lanes"):
        p._lanes()


def test_waiting_lanes_never_count_toward_each_other(tmp_path, monkeypatch):
    # three screen lanes, each run needs 1 GB caps + 20 GB reserve; MemAvailable holds two at once, never three
    monkeypatch.setattr(gates, "mem_available_gb", lambda *a: 22.5)
    log: list = []
    run_pair, state = _stub_run_pair(0.6, log)
    monkeypatch.setattr(pair, "run_pair", run_pair)
    p = pool.Pool(_jobs(_toys(tmp_path, 6), ["SCREEN"]), poolopts(tmp_path, layout="screen", cpu_lanes=3,
                                                                    mem_wait_s=60.0))
    assert p.run() == 0 and p.halt_reason is None
    assert p.counts["completed"] == 6 and p.counts["blocks_void"] == 0
    assert state["max"] == 2                           # two admitted at once; the third waited without blocking them


def test_admitted_runs_anonymous_use_is_credited(tmp_path):
    run_id = "pytest-credit-" + str(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "import sys, time; x = bytearray(150 * 2**20); "
                              "x[::4096] = b'1' * len(x[::4096]); print('ok', flush=True); time.sleep(60)", run_id],
                             stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ok"
        rd = tmp_path / "rd"
        rd.mkdir()
        rec = dict(run_id=run_id, processes=[dict(pid=child.pid, memory_max_gb=9.0, exit_code=None)])
        (rd / "run.json").write_text(json.dumps(rec))
        p = pool.Pool([], poolopts(tmp_path))
        r = pool.Run(job={}, block=None, run_dir=rd)
        use = p._run_use_gb(r)
        assert 0.12 < use < 0.5                                         # ~150 MB of anonymous memory
        rec["processes"][0]["memory_max_gb"] = 0.05                     # never more than the process's cap
        (rd / "run.json").write_text(json.dumps(rec))
        assert p._run_use_gb(r) == pytest.approx(0.05)
        rec["run_id"] = "another-run"                                   # a reused pid is not credited
        (rd / "run.json").write_text(json.dumps(rec))
        assert p._run_use_gb(r) == 0.0
    finally:
        child.kill()
        child.wait()


def test_memory_timeout_requeues_while_others_run_and_halts_only_when_idle(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "mem_available_gb", lambda *a: 21.5)       # exactly one run at a time
    log: list = []
    run_pair, state = _stub_run_pair(1.5, log)
    monkeypatch.setattr(pair, "run_pair", run_pair)
    p = pool.Pool(_jobs(_toys(tmp_path, 2), ["SCREEN"]), poolopts(tmp_path, layout="screen", cpu_lanes=2,
                                                                    mem_wait_s=0.4))
    assert p.run() == 0 and p.halt_reason is None                     # the waiting lane requeued, never halted
    assert p.counts["completed"] == 2 and p.counts["blocks_void"] >= 1 and state["max"] == 1
    assert "requeued" in p.log_path.read_text()
    # nothing in flight and still no room: that is a real shortage -> halt
    monkeypatch.setattr(gates, "mem_available_gb", lambda *a: 10.0)
    p = pool.Pool(_jobs(_toys(tmp_path, 1), ["SCREEN"], seeds=(7,)),
                  poolopts(tmp_path, layout="screen", cpu_lanes=2, mem_wait_s=0.3))
    assert p.run() == 3 and "memory gate" in p.halt_reason


def test_dry_run_prints_memory_arithmetic_and_preflight(tmp_path, capsys):
    toys = _toys(tmp_path, 2)
    assert pool.Pool(_jobs(toys, ["SCREEN"]), poolopts(tmp_path, layout="screen", dry_run=True)).run() == 0
    out = capsys.readouterr().out
    assert "memory: 9 lane(s) at once need up to" in out and "MemAvailable now" in out
    assert "preflight: passes" in out
    o = poolopts(tmp_path, layout="screen", dry_run=True, require_lease=True)
    assert pool.Pool(_jobs(toys, ["SCREEN"], split="screen"), o).run() == 0
    assert "preflight: would refuse" in capsys.readouterr().out
    assert not (tmp_path / "traces").exists()


# ===================================================== F2 (safety): every gate re-checked after a wait
def test_lease_expiring_during_a_thermal_wait_halts_before_launch(tmp_path, monkeypatch):
    from itrea.experiments import record_env
    record_env.static_env()                    # warm the cached record (lscpu, nvidia-smi, SCIP) before the clock
    lease = tmp_path / "lease.json"
    until = datetime.now().astimezone() + timedelta(seconds=4.0)
    lease.write_text(json.dumps(dict(who=pool.WHO, until=until.isoformat())))
    monkeypatch.setattr(gates, "LEASE", lease)
    t_hot = time.monotonic() + 7.0
    monkeypatch.setattr(gates, "acpi_max_c", lambda *a: 85.0 if time.monotonic() < t_hot else 60.0)
    calls: list = []
    run_pair, _ = _stub_run_pair(0.1, calls)
    monkeypatch.setattr(pair, "run_pair", run_pair)
    o = poolopts(tmp_path, require_lease=True, thermal_start_c=80.0, thermal_start_c_cpu=80.0, thermal_poll_s=0.1, est_overhead_s=0.0,
                 open_margin_s=0.0)
    p = pool.Pool(_jobs(_toys(tmp_path, 1), ["A0"]), o)
    assert p.run() == 3 and "lease" in p.halt_reason
    assert calls == []                                                   # nothing launched after the lease ended


# ======================================================== F4 (screening) / F11: freeze, serving, --who
@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
    (r / "configs").mkdir()
    (r / "src" / "x.py").write_text("x = 1\n")
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "init"], ["tag", "prereg-v1"]):
        subprocess.run(["git", "-C", str(r), "-c", "user.email=pytest@example.invalid", "-c", "user.name=pytest",
                        *args], check=True, capture_output=True)
    return r


def test_screening_needs_prereg_and_a_clean_tree(repo, monkeypatch):
    sj = dict(instance="/x/i0.mps", set="screen1", arm="SCREEN", seed=900, budget=300, split="screen")
    out = pool.check_frozen([sj], root=repo)
    assert out["prereg"] and out["screen_jobs"] == 1
    (repo / "src" / "x.py").write_text("x = 2\n")
    with pytest.raises(gates.GateError, match="clean"):
        pool.check_frozen([sj], allow_unfrozen=True, root=repo)
    # a job on a screening candidate needs the tag whatever its split says and whatever --allow-unfrozen says
    monkeypatch.setattr(mj, "candidate_names", lambda data=None: {"air05"})
    cj = dict(instance="/x/air05.mps.gz", set="s", arm="A0", seed=1, budget=1, split="smoke")
    pool.check_frozen([cj], allow_unfrozen=True, root=repo)              # tagged: fine
    with pytest.raises(gates.GateError, match="prereg-v1 does not exist"):
        pool.check_frozen([cj], allow_unfrozen=True, root=repo.parent)    # no tag there


def test_allow_serving_and_who_are_refused(tmp_path, toy, monkeypatch):
    p = pool.Pool(_jobs([toy], ["SCREEN"], split="screen"), poolopts(tmp_path, allow_serving=True, require_lease=True))
    p.lanes, p.blocks = p._lanes(), pool.make_blocks(p.jobs)
    with pytest.raises(gates.GateError, match="--allow-serving"):
        p.preflight()
    jf = tmp_path / "j.jsonl"
    jf.write_text("".join(json.dumps(j) + "\n" for j in _jobs([toy], ["A0"])))
    assert pool.main([str(jf), "--who", "someone else", "--dry-run"]) == 2
    assert pool.main([str(jf), "--who", pool.WHO, "--dry-run", "--no-lease", "--logs-dir", str(tmp_path / "l"),
                      "--traces-dir", str(tmp_path / "t"), "--no-pin"]) == 0


# ============================================================== F4 (safety): registered harness settings
def test_test_and_screen_jobs_need_the_registered_harness(toy):
    tj = [dict(instance=str(toy), set="e2", arm="A3", seed=1000, budget=300, split="test")]
    pool.check_registered_harness(tj, pair.PairOptions())
    for k, v in (("gpu_mem_gb", 6.0), ("pin", False), ("solver_mem_gb", 8.0), ("cuopt_threads", 2),
                 ("use_systemd", False)):
        with pytest.raises(gates.GateError, match=k):
            pool.check_registered_harness(tj, pair.PairOptions(**{k: v}))
    sj = [dict(instance=str(toy), set="s", arm="SCREEN", seed=900, budget=300, split="screen")]
    pool.check_registered_harness(sj, pair.PairOptions(gpu_mem_gb=6.0))   # no GPU in screening
    with pytest.raises(gates.GateError, match="solver_mem_gb"):
        pool.check_registered_harness(sj, pair.PairOptions(solver_mem_gb=8.0))
    pool.check_registered_harness([dict(sj[0], split="tuning")], pair.PairOptions(solver_mem_gb=8.0))


# ================================================================== F5: toy-only exemptions
def test_is_toy_follows_data_symlinks_and_candidate_names(tmp_path, monkeypatch):
    real = tmp_path / "elsewhere" / "pglib_uc"
    real.mkdir(parents=True)
    (real / "ferc__x.mps.gz").write_text("")
    data = tmp_path / "data"
    data.mkdir()
    (data / "pglib_uc").symlink_to(real)
    roots = mj.data_roots(data)
    names = {"air05"}
    assert not mj.is_toy(dict(instance=str(data / "pglib_uc" / "ferc__x.mps.gz")), names, roots)
    assert not mj.is_toy(dict(instance=str(real / "ferc__x.mps.gz")), names, roots)     # the link target too
    assert not mj.is_toy(dict(instance=str(tmp_path / "air05.mps.gz")), names, roots)   # a candidate copied out
    assert mj.is_toy(dict(instance=str(tmp_path / "toy.mps")), names, roots)
    assert mj.is_candidate(dict(instance="/x/whatever.mps", name="air05"), names)


def test_smoke_label_on_a_real_instance_gets_no_exemption(tmp_path, toy, monkeypatch):
    monkeypatch.setattr(mj, "data_roots", lambda data=None: {toy.parent.resolve()})     # the toy now lies in data/
    p = pool.Pool(_jobs([toy], ["A0"]), poolopts(tmp_path))
    p.lanes, p.blocks = p._lanes(), pool.make_blocks(p.jobs)
    with pytest.raises(gates.GateError, match="--no-lease is for toy"):
        p.preflight()
    jf = tmp_path / "j.jsonl"
    jf.write_text(json.dumps(_jobs([toy], ["A0"])[0]) + "\n")
    assert pool.main([str(jf), "--no-systemd", "--no-lease", "--dry-run"]) == 2


def test_pair_cli_refuses_a_candidate_before_the_tag_and_always_takes_the_lock(tmp_path, toy, monkeypatch):
    lock = tmp_path / "runner.lock"
    real_lock = gates.RunLock
    monkeypatch.setattr(gates, "RunLock", lambda *a, **k: real_lock(lock))   # never the real /tmp lock
    monkeypatch.setattr(mj, "candidate_names", lambda data=None: {toy.stem})
    monkeypatch.setattr(pool.paths, "ROOT", tmp_path)                   # not a git checkout: no prereg-v1 tag
    args = ["--instance", str(toy), "--traces-dir", str(tmp_path / "t"), "--no-pin"]
    assert pair.main(args) == 2                                        # candidate, no tag
    assert pair.main(args + ["--no-systemd"]) == 2                     # not a toy: scopes required
    assert not (tmp_path / "t").exists()
    monkeypatch.setattr(mj, "candidate_names", lambda data=None: set())
    with real_lock(lock):                                              # a pool holds the lock
        assert pair.main(args + ["--no-systemd", "--budget", "0.2"]) == 2
    assert not (tmp_path / "t").exists()


# =========================================================== F1: orphans of a dead runner, signals
def _alive(pid: int, run_id: str) -> bool:
    return pair._pid_runs(pid, run_id)


def test_resume_kills_the_orphans_of_a_dead_runner(tmp_path, toy):
    orphan = tmp_path / "orphan.py"
    orphan.write_text(ORPHAN)
    fake = [sys.executable, str(orphan)]
    job = _jobs([toy], ["A0"], budget=60.0)[0]
    runner = subprocess.Popen([sys.executable, "-c", RUNNER, json.dumps(job), str(tmp_path), json.dumps(fake)],
                              start_new_session=True)
    o = poolopts(tmp_path, po=popts(tmp_path, fake=fake), stale_wait_s=0.5)
    rd = pair.run_dir_for(dict(job), o.pair_opts)
    child = None
    try:
        end = time.monotonic() + 60
        while time.monotonic() < end:                      # wait on the record, never on a process name
            rec = pair.read_record(rd) or {}
            pids = [p["pid"] for p in rec.get("processes", []) if p.get("pid")]
            if pids:
                child = pids[0]
                break
            time.sleep(0.1)
        assert child is not None
        os.kill(runner.pid, signal.SIGKILL)                # the runner dies without running any finally block
        runner.wait()
        assert _alive(child, rd.name)                      # its child lives on (own session)
        p = pool.Pool([job], o)
        p.lanes, p.blocks = p._lanes(), pool.order_blocks(pool.make_blocks([job]))
        p.prepare()
        end = time.monotonic() + 10
        while time.monotonic() < end and _alive(child, rd.name):
            time.sleep(0.1)
        assert not _alive(child, rd.name)                  # killed (SIGTERM ignored -> SIGKILL to its group)
        void = [json.loads(x) for x in (rd.parent / "void.jsonl").read_text().splitlines()]
        assert void[-1]["reason"] == "stale run (resume)" and not rd.exists()
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait()
        if child and _alive(child, rd.name):               # only our own orphan, checked by its command line
            os.killpg(child, signal.SIGKILL)


def test_clear_stale_refuses_when_a_process_survives(tmp_path, toy, monkeypatch):
    job = _jobs([toy], ["A0"])[0]
    o = popts(tmp_path)
    rd = pair.run_dir_for(job, o)
    rd.mkdir(parents=True)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", rd.name], start_new_session=True)
    real_killpg = os.killpg
    try:
        (rd / "run.json").write_text(json.dumps(dict(run_id=rd.name, processes=[dict(pid=child.pid)])))
        monkeypatch.setattr(pair.os, "killpg", lambda *a: None)          # a process no signal reaches
        monkeypatch.setattr(pair.os, "kill", lambda *a: None)
        with pytest.raises(pair.RunError, match="survive SIGKILL"):
            pair.clear_stale(job, o, wait_s=0.2)
    finally:
        monkeypatch.undo()
        real_killpg(child.pid, signal.SIGKILL)
        child.wait()


def test_sigterm_becomes_one_keyboard_interrupt():
    prev = pair.install_term_handlers()
    try:
        assert signal.getsignal(signal.SIGTERM) is pair._term
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(2)
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN        # a repeat cannot break the cleanup
    finally:
        pair.restore_handlers(prev)
    assert signal.getsignal(signal.SIGTERM) == prev[signal.SIGTERM]


# ========================================================= F3 / F9 (pair) / F13: process outcomes
def test_partner_killed_after_a_proof_is_not_a_crash(toy, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_END_AFTER_SOLVER", "0.3")
    monkeypatch.setenv("FAKE_REASON_SOLVER", "optimal")
    monkeypatch.setenv("FAKE_IGNORE_STOP_CUOPT", "1")                   # cuOpt: no interrupt API
    rec = pair.run_pair(_jobs([toy], ["A2"], budget=20.0)[0], popts(tmp_path, fake, kill_after_s=0.5))
    assert rec["status"] == "completed", rec["reasons"]
    cu = next(p for p in rec["processes"] if p["role"] == "cuopt")
    assert cu["killed"] and cu["exit_code"] != 0 and cu["stopped_by_runner"] is True
    assert rec["stop"]["kind"] == "partner_end" and rec["wall_s"] < 10.0
    # a crash still counts, and the partner it stopped does not
    monkeypatch.delenv("FAKE_END_AFTER_SOLVER")
    monkeypatch.setenv("FAKE_CRASH_SOLVER", "1")
    rec = pair.run_pair(_jobs([toy], ["A2"], seeds=(2,), budget=20.0)[0], popts(tmp_path, fake, kill_after_s=0.5))
    assert rec["status"] == "failed" and "solver rc=1" in rec["reasons"]
    assert not any(r.startswith("cuopt rc=") for r in rec["reasons"])
    # a process killed at the budget is still a failure (it ignored stop)
    monkeypatch.delenv("FAKE_CRASH_SOLVER")
    monkeypatch.setenv("FAKE_OVERRUN_CUOPT", "30")
    rec = pair.run_pair(_jobs([toy], ["A2"], seeds=(3,), budget=0.5)[0],
                        popts(tmp_path, fake, kill_after_s=0.3, grace_s=0.2))
    assert rec["status"] == "failed" and rec["stop"]["kind"] == "budget"
    assert not any(p["stopped_by_runner"] for p in rec["processes"])


def test_abort_kills_cuopt_at_once(toy, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_IGNORE_STOP_CUOPT", "1")
    monkeypatch.setenv("FAKE_IGNORE_TERM_CUOPT", "1")
    ev = threading.Event()
    o = popts(tmp_path, fake, kill_after_s=60.0, abort=ev, abort_reason=lambda: "memory floor")
    threading.Timer(2.0, ev.set).start()
    t = time.monotonic()
    rec = pair.run_pair(_jobs([toy], ["A2"], budget=30.0)[0], o)
    assert rec["status"] == "void" and rec["void_cause"] == "abort"
    cu = next(p for p in rec["processes"] if p["role"] == "cuopt")
    assert cu["killed"] == "SIGKILL (abort)" and cu["exit_code"] == -9
    assert time.monotonic() - t < 15.0                                  # not kill_after_s (60 s) + 10 + 5


def test_launcher_failure_before_ready_is_void_and_halts_the_pool(toy, fake, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_EXIT_EARLY_SOLVER", "1")                   # no summary: its own code never ran
    rec = pair.run_pair(_jobs([toy], ["A0"])[0], popts(tmp_path, fake))
    assert rec["status"] == "void" and rec["void_cause"] == "runner_error"
    monkeypatch.delenv("FAKE_EXIT_EARLY_SOLVER")
    monkeypatch.setenv("FAKE_SIGKILL_EARLY_SOLVER", "1")                # a signal (OOM kill, segfault): the arm's
    rec = pair.run_pair(_jobs([toy], ["A0"], seeds=(2,))[0], popts(tmp_path, fake))
    assert rec["status"] == "failed"
    monkeypatch.delenv("FAKE_SIGKILL_EARLY_SOLVER")
    monkeypatch.setenv("FAKE_EXIT_EARLY_SOLVER", "1")
    p = pool.Pool(_jobs(_toys(tmp_path, 2), ["A0"]), poolopts(tmp_path, po=popts(tmp_path / "p", fake)))
    assert p.run() == 3 and "runner error" in p.halt_reason and p.opened == 1


# ======================================================================== F12: runner process, sampler
def test_runner_env_and_pinning_in_a_fresh_process():
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                              "MKL_NUM_THREADS")}
    code = ("import os; from itrea.experiments import pool, pair; "
            "print(os.environ['OMP_NUM_THREADS'], os.environ['OPENBLAS_NUM_THREADS']); "
            "print(sorted(pair.pin_runner() or []), sorted(os.sched_getaffinity(0)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[0] == "1 1"
    pinned = set(json.loads(lines[1].split("] ")[0] + "]"))
    assert pinned and pinned <= set(pair.RUNNER_CORES)


def test_run_pair_uses_and_never_stops_a_shared_sampler(toy, fake, tmp_path):
    class Shared:
        stopped = False
        windows: list = []

        def summary(self, t_from=None, t_to=None, gpu_run=True):
            self.windows.append((t_from, t_to, gpu_run))
            return dict(verdict="ok", reasons=[], n=0)

        def stop(self, join=True):
            self.stopped = True
    s = Shared()
    rec = pair.run_pair(_jobs([toy], ["A0"])[0], popts(tmp_path, fake, gpu_health=True, shared_sampler=s))
    assert rec["status"] == "completed" and s.windows and not s.stopped
    assert s.windows[0][0] == rec["t0"] and s.windows[0][2] is False


# ===================================================================== F7 (safety): serving gate fails closed
def test_serving_gate_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(gates, "LEASE", tmp_path / "no-lease.json")   # snapshot() reads it: never the real one

    def fake_run(out, rc=0):
        return lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], rc, stdout=out, stderr="")
    monkeypatch.setattr(gates.subprocess, "run", fake_run("", 1))       # no user bus: "Failed to connect to bus"
    st = gates.serving_state()
    assert st == {u: None for u in gates.SERVING_UNITS}
    with pytest.raises(gates.GateError, match="unreadable or not stopped"):
        gates.check_serving_stopped()
    assert gates.snapshot()["serving"] == st                            # the record never raises
    for s in ("activating", "deactivating", "reloading", "active"):
        monkeypatch.setattr(gates.subprocess, "run", fake_run(s + "\n", 0))
        with pytest.raises(gates.GateError):
            gates.check_serving_stopped()
    for s in ("inactive", "failed"):
        monkeypatch.setattr(gates.subprocess, "run", fake_run(s + "\n", 3))
        assert gates.check_serving_stopped()
    assert gates.check_serving_stopped({"ornith-vllm": False, "llama-qwen38": "inactive"})


# ===================================================================== F8 (safety): GPU health fails closed
def _samples(n, t0=0.0, clock=2450.0, util=95.0):
    return [dict(t=t0 + i, sm_mhz=clock, temp_c=70.0, util=util, apps=[], foreign=[]) for i in range(n)]


def test_gpu_health_coverage_voids_an_unverifiable_gpu_run():
    s = gpu_health.summarize([], gpu_run=True, t_from=0.0, t_to=300.0)
    assert s["verdict"] == "void" and "unverifiable" in s["reasons"][0]
    s = gpu_health.summarize(_samples(100), gpu_run=True, t_from=0.0, t_to=300.0)
    assert s["verdict"] == "void" and s["expected_samples"] == 300.0
    assert gpu_health.summarize(_samples(290), gpu_run=True, t_from=0.0, t_to=300.0)["verdict"] == "ok"
    assert gpu_health.summarize([], gpu_run=False, t_from=0.0, t_to=300.0)["verdict"] == "no-data"
    assert gpu_health.summarize([], gpu_run=True, t_from=0.0, t_to=2.0)["verdict"] == "no-data"     # toy window
    idle = gpu_health.summarize(_samples(290, util=0.0), gpu_run=True, t_from=0.0, t_to=300.0)
    assert idle["verdict"] == "ok" and idle["warnings"]

    def broken(cmd):
        raise RuntimeError("nvidia-smi gone")
    smp = gpu_health.Sampler(query=broken)
    smp.sample()
    assert smp.summary(t_from=time.monotonic() - 100, t_to=time.monotonic())["verdict"] == "void"


def test_preflight_refuses_gpu_jobs_when_nvidia_smi_fails(tmp_path, toy, monkeypatch):
    monkeypatch.setattr(gpu_health.Sampler, "sample", lambda self: None)
    p = pool.Pool(_jobs([toy], ["A2"]), poolopts(tmp_path, po=popts(tmp_path, gpu_health=True)))
    p.lanes, p.blocks = p._lanes(), pool.make_blocks(p.jobs)
    with pytest.raises(gates.GateError, match="nvidia-smi"):
        p.preflight()
    p = pool.Pool(_jobs([toy], ["A0"]), poolopts(tmp_path, po=popts(tmp_path, gpu_health=True)))
    p.lanes, p.blocks = p._lanes(), pool.make_blocks(p.jobs)
    p.preflight()                                                       # a CPU-only run set does not need it


# ============================================================ F7 (screening) / F10: the never-split estimate
def test_estimate_never_below_one_run_and_margin(tmp_path, toy):
    jobs = _jobs(_toys(tmp_path, 3), ["SCREEN"], budget=300.0)
    p = pool.Pool(jobs, poolopts(tmp_path, layout="screen", est_overhead_s=90.0))
    p.lanes, p.blocks = p._lanes(), pool.make_blocks(jobs)
    assert p._est_backlog(p.blocks[0]) == pytest.approx(390.0)         # was 390 / 9 = 43 s
    q = pool.Pool(jobs, poolopts(tmp_path, layout="screen"))
    q.lanes, q.blocks = q._lanes(), pool.make_blocks(jobs)
    po = q.o.pair_opts
    assert q._est_run(q.blocks[0].runs[0]) == pytest.approx(300 + po.grace_s + po.kill_after_s + 15 + 30)
    gpu = pool.Run(job=dict(arm="A3", budget=300.0), block=None)
    assert q._est_run(gpu) == pytest.approx(300 + po.grace_s + po.kill_after_s + 15 + 120)
    need = q._est_backlog(q.blocks[0])
    q.o.deadline = datetime.now().astimezone() + timedelta(seconds=need + 300)   # inside the 600 s margin
    assert "margin" in q._may_open(q.blocks[0])
    q.o.deadline = datetime.now().astimezone() + timedelta(seconds=need + 900)
    assert q._may_open(q.blocks[0]) is None


# ========================================================================= F8 (screening): stage 2
def test_stage2_table_fails_closed():
    ok = [dict(name="a", p300="0.01", feasible="1"), dict(name="b", p300="", feasible="0")]
    assert mj.stage2_selected(ok) == ["b"]
    with pytest.raises(mj.JobError, match="lacks"):
        mj.stage2_selected([dict(name="air05", pt_screen="0.001", t_first_event="True")])
    with pytest.raises(mj.JobError, match="blank or NaN"):
        mj.stage2_selected([dict(name="air05", p300="nan", feasible="True")])
    with pytest.raises(mj.JobError, match="blank or NaN"):
        mj.stage2_selected([dict(name="air05", p300="", feasible="1")])
    with pytest.raises(mj.JobError, match="empty"):
        mj.stage2_selected([])


def test_screen2_from_the_run_table(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "miplib" / "inst").mkdir(parents=True)
    for n in ("a1", "b1"):
        (data / "miplib" / "inst" / f"{n}.mps.gz").write_text("")
    (data / "miplib" / "solu.txt").write_text("")
    seen = {}

    def needed(path):
        seen["path"] = str(path)
        return ["b1"]
    monkeypatch.setattr(mj, "stage2_from_runs", needed)
    out = tmp_path / "s2.jsonl"
    assert mj.main(["screen2", "--runs", str(tmp_path / "runs.parquet"), "--data", str(data), "--out", str(out)]) == 0
    jobs = [json.loads(x) for x in out.read_text().splitlines()]
    assert [j["name"] for j in jobs] == ["b1"] and jobs[0]["seed"] == 901 and jobs[0]["split"] == "screen"
    assert seen["path"].endswith("runs.parquet")
    bad = tmp_path / "bad.csv"
    bad.write_text("name,pt_screen\na1,0.5\n")
    assert mj.main(["screen2", "--results", str(bad), "--data", str(data), "--out", str(out)]) == 2


def test_thermal_start_gate_depends_on_gpu_use():
    """D-008: CPU-only arms start below 88 °C, GPU arms below 80 °C."""
    from itrea.experiments import gates, pool
    assert gates.THERMAL_WAIT_C == 80.0 and gates.THERMAL_WAIT_C_CPU == 88.0
    assert pool._arm_uses_gpu({"arm": "SCREEN"}) is False
    assert pool._arm_uses_gpu({"arm": "A0"}) is False
    assert pool._arm_uses_gpu({"arm": "A3"}) is True
    assert pool._arm_uses_gpu({"arm": "A2"}) is True
    assert pool._arm_uses_gpu({"arm": "nope"}) is True


def test_rest_flag_holds_launches_until_it_clears(tmp_path, monkeypatch):
    """D-009: while fan-governor's rest flag exists no run starts; once it is gone the run launches."""
    from itrea.experiments import record_env
    record_env.static_env()
    flag = tmp_path / "rest"
    flag.write_text("rest_seconds=600\n")
    monkeypatch.setattr(gates, "REST_FLAG", flag)
    monkeypatch.setattr(gates, "acpi_max_c", lambda *a: 50.0)
    t_clear = time.monotonic() + 1.5

    def due(f=None):                      # reads the flag file itself (conftest stubs gates.rest_due to False)
        if time.monotonic() >= t_clear and flag.exists():
            flag.unlink()
        return flag.exists()
    monkeypatch.setattr(gates, "rest_due", due)
    calls: list = []
    run_pair, _ = _stub_run_pair(0.1, calls)
    monkeypatch.setattr(pair, "run_pair", run_pair)
    o = poolopts(tmp_path, require_lease=False, thermal_poll_s=0.1, est_overhead_s=0.0, open_margin_s=0.0)
    t0 = time.monotonic()
    p = pool.Pool(_jobs(_toys(tmp_path, 1), ["A0"]), o)
    assert p.run() == 0
    assert len(calls) == 1 and time.monotonic() - t0 >= 1.4          # launched only after the flag cleared
