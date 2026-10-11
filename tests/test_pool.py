"""The pool runner: Williams design balance, block order, the freeze check on a throw-away git repository, and
end-to-end runs of toy job files with a FAKE role process (resume, split-block voiding, kept crashes, stops, lanes,
GPU-health halt, thermal requeue). No lease, no systemd scopes, no real solver: split "smoke" only."""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from itertools import combinations, permutations
from pathlib import Path

import pytest
import yaml

from itrea.benchmarks.toy import set_cover, write_mps
from itrea.config import EAConfig
from itrea.experiments import gates, pair, pool

FAKE_PROC = r'''
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
mb = open_mailbox(a.mailbox, role, run_id=a.run_id)
tr = Trace(a.run_dir, role, arm=a.arm, seed=int(a.seed))
mb.ready()
try:
    t0 = mb.wait_start(timeout=60)
except RuntimeError:
    sys.exit(0)
tr.start(t0)
k = 0
while time.monotonic() - t0 < float(a.budget) and not mb.stopped():
    tr.incumbent(100.0 - k)
    k += 1
    time.sleep(0.05)
tr.close()
sys.exit(int(os.environ.get("FAKE_EXIT_" + role.upper(), "0")))
'''


# ---------------------------------------------------------------------------------------------------- Williams
@pytest.mark.parametrize("k", range(1, 10))
def test_williams_rows_are_balanced(k):
    rows = pool.williams_rows(k)
    n = len(rows)
    assert n == (k if k % 2 == 0 or k == 1 else 2 * k)
    assert all(sorted(r) == list(range(k)) for r in rows)              # every row is a permutation
    for pos in range(k):                                               # each treatment equally often per position
        assert Counter(r[pos] for r in rows) == {t: n // k for t in range(k)}
    if k > 1:                                                          # first-order carryover balance
        pairs = Counter((r[i], r[i + 1]) for r in rows for i in range(k - 1))
        assert set(pairs) == set(permutations(range(k), 2))
        assert len(set(pairs.values())) == 1


def test_williams_order_is_deterministic_and_spread():
    arms = ["A5x0", "A0", "A3", "A2", "A5", "A1", "A0star"]
    a = pool.williams_order(arms, "air05", 1000)
    assert sorted(a) == sorted(arms) and a == pool.williams_order(list(reversed(arms)), "air05", 1000)
    assert pool.block_hash("air05", 1000) == pool.block_hash("air05", 1000)
    rows = Counter(tuple(pool.williams_order(arms, f"inst{i}", s)) for i in range(200) for s in (1000, 1001))
    assert len(rows) == 14                                             # 7 arms (odd): all 14 rows are used
    assert max(rows.values()) < 3 * min(rows.values())
    assert pool.canonical_arms(["A3~x0", "A0", "A3", "A5@c1"]) == ["A0", "A3", "A3~x0", "A5@c1"]


def _jobs(instances, arms, seeds=(1,), budget=0.3, split="smoke", stratum=None):
    out = []
    for i, inst in enumerate(instances):
        for s in seeds:
            for a in arms:
                out.append(dict(instance=str(inst), set="pytest", name=Path(inst).stem.split(".")[0], arm=a, seed=s,
                                budget=budget, config=None, overrides={}, split=split,
                                stratum=stratum or f"st{i % 3}", block=f"{Path(inst).stem}__s{s}"))
    return out


def test_make_blocks_orders_and_rejects_duplicates():
    jobs = _jobs([f"/x/i{i}.mps" for i in range(4)], ["A0", "A1", "A2"])
    blocks = pool.make_blocks(jobs)
    assert len(blocks) == 4
    for b in blocks:
        assert [r.label for r in b.runs] == pool.williams_order(["A0", "A1", "A2"], b.name, b.seed)
        assert [r.job["position"] for r in b.runs] == [0, 1, 2]
    with pytest.raises(pool.PoolError, match="twice"):
        pool.make_blocks(jobs + jobs[:1])
    bad = dict(jobs[0], instance="/x/other.mps", name="other")
    with pytest.raises(pool.PoolError, match="mixes"):
        pool.make_blocks(jobs + [bad])


def test_block_order_interleaves_strata():
    jobs = []
    for s, n in (("a", 12), ("b", 6), ("c", 6)):
        jobs += _jobs([f"/x/{s}{i}.mps" for i in range(n)], ["A0"], stratum=s)
    order = pool.order_blocks(pool.make_blocks(jobs), seed=7)
    assert [b.key for b in order] == [b.key for b in pool.order_blocks(pool.make_blocks(jobs), seed=7)]
    assert [b.key for b in order] != [b.key for b in pool.order_blocks(pool.make_blocks(jobs), seed=8)]
    first = Counter(b.stratum for b in order[:12])
    assert abs(first["a"] - 6) <= 2 and abs(first["b"] - 3) <= 2 and abs(first["c"] - 3) <= 2
    longest = max(len(list(g)) for _, g in __import__("itertools").groupby(b.stratum for b in order))
    assert longest <= 4


# ------------------------------------------------------------------------------------------------ freeze check
def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=pytest@example.invalid", "-c", "user.name=pytest",
                    *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "configs").mkdir(parents=True)
    (r / "src").mkdir()
    (r / "src" / "x.py").write_text("x = 1\n")
    cfg = EAConfig().to_dict()
    (r / "configs" / "frozen-e2.yaml").write_text(yaml.safe_dump(cfg))
    other = EAConfig().with_overrides({"population.size": 64}).to_dict()
    (r / "configs" / "other.yaml").write_text(yaml.safe_dump(other))
    _git(r, "init", "-q")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "init")
    return r


def _tjob(repo, split="test", config="frozen-e2.yaml", **kw):
    return dict(instance="/x/i0.mps", set="e2", arm="A3", seed=1000, budget=300, split=split,
                config=str(repo / "configs" / config), **kw)


def test_freeze_check_refuses_without_prereg_tag(repo):
    pool.check_frozen([dict(_tjob(repo), split="smoke")], root=repo)               # smoke: no requirement
    with pytest.raises(gates.GateError, match="prereg-v1"):
        pool.check_frozen([_tjob(repo, split="screen")], root=repo)
    with pytest.raises(gates.GateError, match="prereg-v1") as ei:                  # never bypassed for screening
        pool.check_frozen([_tjob(repo, split="screen")], allow_unfrozen=True, root=repo)
    assert "--allow-unfrozen" not in str(ei.value)                                 # and the error does not offer it
    pool.check_frozen([_tjob(repo, split="tuning")], allow_unfrozen=True, root=repo)   # non-candidate tuning: may
    with pytest.raises(gates.GateError, match="prereg-v1"):                        # never bypassed for test jobs
        pool.check_frozen([_tjob(repo)], allow_unfrozen=True, root=repo)


def test_freeze_check_with_tag(repo):
    _git(repo, "tag", "prereg-v1")
    out = pool.check_frozen([_tjob(repo), _tjob(repo, split="tuning", config="other.yaml")], root=repo)
    assert out["frozen_hash"] == EAConfig().hash() and out["prereg"]
    with pytest.raises(gates.GateError, match="config hash"):
        pool.check_frozen([_tjob(repo, config="other.yaml")], root=repo)
    with pytest.raises(gates.GateError, match="free overrides"):
        pool.check_frozen([_tjob(repo, overrides={"ranks.x": 0.0})], root=repo)
    (repo / "src" / "x.py").write_text("x = 2\n")                                    # uncommitted change
    with pytest.raises(gates.GateError, match="clean"):
        pool.check_frozen([_tjob(repo)], root=repo)


def test_freeze_check_tag_must_be_an_ancestor(repo):
    _git(repo, "checkout", "-q", "-b", "side")
    (repo / "src" / "y.py").write_text("y = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "side")
    _git(repo, "tag", "prereg-v1")
    _git(repo, "checkout", "-q", "-")
    with pytest.raises(gates.GateError, match="not an ancestor"):
        pool.check_frozen([_tjob(repo)], root=repo)


def test_freeze_check_needs_frozen_file(repo):
    _git(repo, "rm", "-q", "configs/frozen-e2.yaml")
    _git(repo, "commit", "-q", "-m", "drop")
    _git(repo, "tag", "prereg-v1")
    with pytest.raises(gates.GateError, match="frozen-e2.yaml does not exist"):
        pool.check_frozen([_tjob(repo, config="other.yaml")], root=repo)


# ------------------------------------------------------------------------------------------------- end to end
@pytest.fixture
def env(tmp_path, monkeypatch):
    # The pool's gates must not read the real box (free memory changes when the model servers are up; a real rest flag
    # or a hot box would stall the tests): plenty of memory, no fan-governor rest.
    monkeypatch.setattr(gates, "mem_available_gb", lambda *a, **k: 500.0)
    monkeypatch.setattr(gates, "rest_due", lambda *a, **k: False)
    fake = tmp_path / "fake_proc.py"
    fake.write_text(FAKE_PROC)
    insts = [write_mps(set_cover(20, 12, seed=i), tmp_path / f"toy{i}.mps") for i in range(2)]
    return tmp_path, [sys.executable, str(fake)], insts


def popts(tmp_path, fake, **kw) -> pool.PoolOptions:
    po = pair.PairOptions(traces_dir=tmp_path / "traces", mailbox_root=tmp_path / "mb", use_systemd=False, pin=False,
                          grace_s=2.0, kill_after_s=10.0, gpu_health=False, record_snapshots=False, solver_mem_gb=1.0,
                          ea_mem_gb=1.0, cuopt_mem_gb=1.0, cupy_cache_dir=tmp_path / "cupy",
                          module_override={k: fake for k in ("scip", "highs", "cuopt", "ea")})
    d = dict(require_lease=False, logs_dir=tmp_path / "logs", lock_path=tmp_path / "runner.lock",
             thermal_start_c=500.0, thermal_start_c_cpu=500.0, watchdog=False, root=tmp_path, pair_opts=po)
    d.update(kw)
    return pool.PoolOptions(**d)


def _records(tmp_path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((tmp_path / "traces" / "raw").glob("*/*/run.json"))]


def test_pool_runs_resumes_and_skips(env):
    tmp_path, fake, insts = env
    jobs = _jobs(insts, ["A0", "A1"])
    assert pool.Pool(jobs, popts(tmp_path, fake)).run() == 0
    recs = _records(tmp_path)
    assert len(recs) == 4 and {r["status"] for r in recs} == {"completed"}
    for r in recs:
        order = pool.williams_order(["A0", "A1"], r["name"], r["seed"])
        assert r["position"] == order.index(r["arm"]) and r["window"]
    status = json.loads((tmp_path / "logs" / "pool-status.json").read_text())
    assert status["state"] == "done" and status["counts"]["completed"] == 4
    assert list((tmp_path / "logs").glob("pool-*.log")) and list((tmp_path / "traces" / "resources").glob("pool-*.json"))
    p2 = pool.Pool(jobs, popts(tmp_path, fake))
    assert p2.run() == 0 and p2.counts["skipped"] == 4 and p2.counts["completed"] == 0


def test_split_block_is_voided_and_rerun_whole(env):
    tmp_path, fake, insts = env
    jobs = _jobs(insts, ["A0", "A1"])
    assert pool.Pool(jobs, popts(tmp_path, fake)).run() == 0
    o = popts(tmp_path, fake)
    one = pair.run_dir_for(jobs[0], o.pair_opts)
    import shutil
    shutil.rmtree(one)                                    # as if the pool died after the first run of this block
    p = pool.Pool(jobs, popts(tmp_path, fake))
    assert p.run() == 0
    assert p.counts["completed"] == 2 and p.counts["skipped"] == 2              # the whole block, not one run
    void = [json.loads(line) for line in (one.parent / "void.jsonl").read_text().splitlines()]
    assert len(void) == 1 and void[0]["reason"] == "split block (resume)" and void[0]["status_before"] == "completed"
    assert (one.parent / "void" / void[0]["moved_to"].split("/", 1)[1]).is_dir()
    assert {r["status"] for r in _records(tmp_path)} == {"completed"}


def test_failed_runs_are_kept_not_rerun(env, monkeypatch):
    tmp_path, fake, insts = env
    jobs = _jobs(insts[:1], ["A0"])
    monkeypatch.setenv("FAKE_EXIT_SOLVER", "1")
    p = pool.Pool(jobs, popts(tmp_path, fake))
    assert p.run() == 0 and p.counts["failed"] == 1
    monkeypatch.delenv("FAKE_EXIT_SOLVER")
    p2 = pool.Pool(jobs, popts(tmp_path, fake))
    assert p2.run() == 0 and p2.counts["skipped"] == 1 and p2.counts["completed"] == 0
    p3 = pool.Pool(jobs, popts(tmp_path, fake, rerun_failed=True))               # engineering reruns, non-test only
    assert p3.run() == 0 and p3.counts["completed"] == 1


def test_stops_never_open_a_block(env):
    tmp_path, fake, insts = env
    jobs = _jobs(insts, ["A0"])
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "pool.stop").write_text("")
    p = pool.Pool(jobs, popts(tmp_path, fake))
    assert p.run() == 0 and p.opened == 0 and "stop file" in p.stopping
    (tmp_path / "logs" / "pool.stop").unlink()
    past = datetime.now().astimezone() - timedelta(minutes=1)
    p = pool.Pool(jobs, popts(tmp_path, fake, deadline=past))
    assert p.run() == 0 and p.opened == 0 and "deadline" in p.stopping
    p = pool.Pool(jobs, popts(tmp_path, fake, max_blocks=1))
    assert p.run() == 0 and p.opened == 1 and p.counts["completed"] == 1
    assert json.loads((tmp_path / "logs" / "pool-status.json").read_text())["state"] == "stopped"


def test_refusals(env):
    tmp_path, fake, insts = env
    screen = _jobs(insts[:1], ["SCREEN"], split="screen")
    assert pool.Pool(screen, popts(tmp_path, fake)).run() == 2                    # --no-lease only for smoke
    test = _jobs(insts[:1], ["A0"], split="test")
    assert pool.Pool(test, popts(tmp_path, fake, require_lease=True, who="nobody")).run() == 2
    smoke = _jobs(insts[:1], ["A0"])
    with gates.RunLock(tmp_path / "runner.lock"):
        assert pool.Pool(smoke, popts(tmp_path, fake)).run() == 2                 # one pool at a time
    tune = _jobs(insts[:1], ["A0"], split="tuning")
    o = popts(tmp_path, fake, require_lease=True, cpu_lanes=2, allow_unfrozen=True)
    p = pool.Pool(tune, o)
    p.lanes, p.blocks = p._lanes(), pool.make_blocks(tune)
    with pytest.raises(gates.GateError, match="E0 calibration"):
        p.preflight()
    assert not (tmp_path / "traces" / "raw").exists()


def test_lanes_split_partner_and_solver_arms(env):
    tmp_path, fake, insts = env
    jobs = _jobs(insts, ["A0", "A2"])
    p = pool.Pool(jobs, popts(tmp_path, fake, cpu_lanes=1))                      # smoke: no E0 flag needed
    assert p.run() == 0 and p.counts["completed"] == 4
    for r in _records(tmp_path):
        cores = {pr["role"]: pr["cores"] for pr in r["processes"]}
        if r["arm"] == "A0":
            assert cores == {"solver": [7]} and r["lane"]["name"] == "c1"
        else:
            assert cores == {"solver": [5], "cuopt": [1, 2, 3]} and r["lane"]["name"] == "g0"


def test_screen_layout_rejects_partner_arms(env):
    tmp_path, fake, insts = env
    p = pool.Pool(_jobs(insts[:1], ["A2"]), popts(tmp_path, fake, layout="screen", cpu_lanes=2))
    assert p.run() == 2


def _full(tmp_path, fake, rmm=36.0, host=24.0, **kw) -> pool.PoolOptions:
    o = popts(tmp_path, fake, layout="full", **kw)
    o.pair_opts.cuopt_mem_gb, o.pair_opts.gpu_mem_gb = host, rmm
    return o


def test_full_layout_runs_a6m_on_every_core(env, monkeypatch):
    """--layout full (D-020): one lane with all 20 cores; A6m pinned to all of them with 24 threads and zero gaps, the
    memory record at D-020's 81 GB; the serving check is part of the layout (stubbed here)."""
    tmp_path, fake, insts = env
    seen = []
    monkeypatch.setattr(gates, "check_serving_stopped", lambda *a, **k: seen.append(1))
    p = pool.Pool(_jobs(insts[:1], ["A6m"]), _full(tmp_path, fake))
    assert p.run() == 0 and p.counts["completed"] == 1 and seen        # preflight and the per-run gate
    (r,) = _records(tmp_path)
    assert r["lane"]["name"] == "full" and r["lane"]["whole_box"] and "runner_cores" in r
    (pr,) = r["processes"]
    assert pr["cores"] == list(range(20)) and pr["memory_max_gb"] == 24.0 and pr["gpu_limit_gb"] == 36.0
    assert "--mip-gap-zero" in pr["cmd"] and pr["cmd"][pr["cmd"].index("--cpu-threads") + 1] == "24"
    assert r["memory_budget"]["need_gb"] == 81.0


def test_full_layout_refusals(env, monkeypatch):
    tmp_path, fake, insts = env
    monkeypatch.setattr(gates, "check_serving_stopped", lambda *a, **k: None)
    assert pool.Pool(_jobs(insts[:1], ["A6"]), _full(tmp_path, fake)).run() == 2          # another arm on the box
    o = popts(tmp_path, fake)
    o.pair_opts.cuopt_mem_gb, o.pair_opts.gpu_mem_gb = 24.0, 36.0
    assert pool.Pool(_jobs(insts[:1], ["A6m"]), o).run() == 2                             # A6m on the pair layout
    assert pool.Pool(_jobs(insts[:1], ["A6m"]), _full(tmp_path, fake, allow_serving=True)).run() == 2
    assert pool.Pool(_jobs(insts[:1], ["A6m"]), _full(tmp_path, fake, rmm=12.0)).run() == 2   # D-020's memory only
    assert pool.Pool(_jobs(insts[:1], ["A6m"]), _full(tmp_path, fake, host=12.0)).run() == 2
    with pytest.raises(pool.PoolError, match="only under --layout full"):
        pool.check_layout(_jobs(insts[:1], ["A6m"]), "pair")
    with pytest.raises(pool.PoolError, match="runs only A6m"):
        pool.check_layout(_jobs(insts[:1], ["A6m", "A6"]), "full")
    with pytest.raises(pool.PoolError, match="cpu-lanes"):
        pool.Pool([], _full(tmp_path, fake, cpu_lanes=1))._lanes()

    def up(*a, **k):
        raise gates.GateError("ornith-vllm is active")
    monkeypatch.setattr(gates, "check_serving_stopped", up)                              # servers up: refused
    assert pool.Pool(_jobs(insts[:1], ["A6m"]), _full(tmp_path, fake)).run() == 2
    assert not (tmp_path / "traces" / "raw").exists()


def test_full_layout_dry_run_prints_the_need(env, capsys, monkeypatch):
    tmp_path, fake, insts = env
    monkeypatch.setattr(gates, "check_serving_stopped", lambda *a, **k: None)
    assert pool.Pool(_jobs(insts, ["A6m"]), _full(tmp_path, fake, dry_run=True)).run() == 0
    out = capsys.readouterr().out
    assert "2 blocks, 2 runs" in out and "full(x925=(5, 6, 7, 8, 9, 15, 16, 17, 18, 19)" in out
    assert "need up to 81.0 GB" in out and "preflight: passes" in out
    assert not (tmp_path / "traces").exists()


def _fake_run_pair(tmp_path, script):
    """A stand-in for pair.run_pair: writes a run directory and returns a record whose status the script chooses."""
    calls = []

    def run_pair(job, opts):
        rd = pair.run_dir_for(job, opts)
        rd.mkdir(parents=True)
        st, cause, side = script(len(calls), job)
        if side:
            side()
        rec = dict(run_id=rd.name, status=st, void_cause=cause, reasons=[f"scripted {st}"], arm=job["arm"])
        (rd / "run.json").write_text(json.dumps(rec))
        calls.append((job["block"], job["arm"], st))
        return rec
    return run_pair, calls


def test_gpu_health_void_halts_and_voids_the_block(env, monkeypatch):
    tmp_path, fake, insts = env
    jobs = _jobs(insts, ["A0", "A2"])
    run_pair, calls = _fake_run_pair(tmp_path, lambda i, j: ("void", "gpu_health", None) if j["arm"] == "A2"
                                     else ("completed", None, None))
    monkeypatch.setattr(pair, "run_pair", run_pair)
    p = pool.Pool(jobs, popts(tmp_path, fake))
    assert p.run() == 3 and "GPU health" in p.halt_reason
    assert p.opened == 1 and p.counts["blocks_void"] == 1
    voided = (tmp_path / "traces" / "raw" / "pytest" / "void.jsonl").read_text().splitlines()
    assert len(voided) == len(calls)                       # every run of the block moved aside
    assert json.loads((tmp_path / "logs" / "pool-status.json").read_text())["state"] == "halted"


def test_thermal_pause_requeues_the_block(env, monkeypatch):
    tmp_path, fake, insts = env
    jobs = _jobs(insts, ["A0", "A1"])
    holder = {}

    def script(i, job):
        if i == 0:      # the watchdog trips during the very first run
            return "void", "abort", lambda: holder["pool"]._on_breach("thermal", "thermal: 93 °C >= 92 °C")
        return "completed", None, None
    run_pair, calls = _fake_run_pair(tmp_path, script)
    monkeypatch.setattr(pair, "run_pair", run_pair)
    p = pool.Pool(jobs, popts(tmp_path, fake))
    holder["pool"] = p
    assert p.run() == 0 and p.halt_reason is None
    assert p.counts["void"] == 1 and p.counts["blocks_void"] == 1 and p.counts["completed"] == 4
    first_block = calls[0][0]
    assert [c for c in calls if c[0] == first_block][1:] and calls[1][0] == first_block    # rerun first, whole
    assert {r["status"] for r in _records(tmp_path)} == {"completed"}


def _redis_ok() -> bool:
    from itrea.experiments import exchange_ctl
    return exchange_ctl.ping()


@pytest.mark.skipif(not _redis_ok(), reason="itrea-redis is not answering on 127.0.0.1:6390")
def test_pool_over_redis(env):
    from itrea.experiments import exchange_ctl
    tmp_path, fake, insts = env
    jobs = _jobs(insts[:1], ["A0", "A1"], seeds=(int(__import__("uuid").uuid4().int % 10**6),))
    o = popts(tmp_path, fake)
    o.pair_opts.mailbox = "redis"
    p = pool.Pool(jobs, o)
    assert p.run() == 0 and p.counts["completed"] == 2
    recs = _records(tmp_path)
    assert all(r["mailbox"]["kind"] == "redis" for r in recs)
    assert not set(exchange_ctl.run_ids()) & {r["run_id"] for r in recs}          # every run's keys deleted


def test_dry_run_prints_the_plan(env, capsys):
    tmp_path, fake, insts = env
    assert pool.Pool(_jobs(insts, ["A0", "A1", "A2"]), popts(tmp_path, fake, dry_run=True)).run() == 0
    out = capsys.readouterr().out
    assert "2 blocks, 6 runs" in out and "g0" in out
    assert not (tmp_path / "traces").exists()
