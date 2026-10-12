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


def test_external_a9_jobs_are_sealed_until_prereg_v2(repo):
    """D-021 R33: an external job (split ext) of an arm with the EA needs the tag prereg-v2 as an ancestor of HEAD; A0
    and A6F's external jobs need prereg-v1 only."""
    _git(repo, "tag", "prereg-v1")
    a9 = dict(_tjob(repo, split="ext"), arm="A9")
    a6f, a0 = dict(a9, arm="A6F", config=None), dict(a9, arm="A0", config=None)
    assert "prereg_v2" not in pool.check_frozen([a6f, a0], root=repo)
    with pytest.raises(gates.GateError, match="R33.*prereg-v2"):
        pool.check_frozen([a6f, a9], root=repo)
    with pytest.raises(gates.GateError, match="prereg-v2"):
        pool.check_frozen([a9], allow_unfrozen=True, root=repo)                 # never bypassed
    _git(repo, "tag", "prereg-v2")
    assert pool.check_frozen([a6f, a9], root=repo)["prereg_v2"]


def test_block_priority_groups():
    """D-021 phase 2: when every job carries `priority`, the groups run in increasing priority, each in the usual
    seeded order with strata interleaved; mixed (some with, some without) and two priorities in one block are refused;
    files without the field keep their order."""
    jobs = []
    for s, n, pr in (("a", 6, 1), ("b", 6, 0), ("c", 4, 2)):
        jobs += [dict(j, priority=pr) for j in _jobs([f"/x/{s}{i}.mps" for i in range(n)], ["A0"], stratum=s)]
    order = pool.order_blocks(pool.make_blocks(jobs), seed=7)
    assert [b.priority for b in order] == [0] * 6 + [1] * 6 + [2] * 4
    assert [b.key for b in order[:6]] == [b.key for b in pool.order_blocks(pool.make_blocks(
        [j for j in jobs if j["priority"] == 0]), seed=7)]
    plain = [{k: v for k, v in j.items() if k != "priority"} for j in jobs]
    assert [b.key for b in pool.order_blocks(pool.make_blocks(plain), seed=7)] == \
        [b.key for b in pool._order_group(pool.make_blocks(plain), 7)]
    with pytest.raises(pool.PoolError, match="some blocks carry a priority"):
        pool.order_blocks(pool.make_blocks(plain[:3] + jobs[3:]), seed=7)
    two = _jobs(["/x/z.mps"], ["A0", "A1"])
    two[1]["priority"] = 1
    with pytest.raises(pool.PoolError, match="different priorities"):
        pool.make_blocks(two)
    with pytest.raises(pool.PoolError, match="integer"):
        pool.make_blocks([dict(jobs[0], priority="1")])


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


def test_full_layout_refuses_redis_and_a_measured_peak(env, monkeypatch):
    """Under --layout full: no Redis mailbox (its container's core 0 is inside the lane) and no --cuopt-gpu-peak-gb
    (it would admit A6m below D-020's 81 GB) — refused before anything runs (review 2026-10-11)."""
    tmp_path, fake, insts = env
    monkeypatch.setattr(gates, "check_serving_stopped", lambda *a, **k: None)
    o = _full(tmp_path, fake)
    o.pair_opts.mailbox = "redis"
    assert pool.Pool(_jobs(insts[:1], ["A6m"]), o).run() == 2
    o = _full(tmp_path, fake)
    o.pair_opts.cuopt_gpu_peak_gb = 20.0
    assert pool.Pool(_jobs(insts[:1], ["A6m"]), o).run() == 2
    with pytest.raises(gates.GateError, match="redis"):
        pool.check_full_options(pair.PairOptions(mailbox="redis"))
    with pytest.raises(gates.GateError, match="peak"):
        pool.check_full_options(pair.PairOptions(cuopt_gpu_peak_gb=20.0))
    with pytest.raises(pair.RunError, match="Redis"):
        pair.job_budget(_jobs(insts[:1], ["A6m"])[0], pair.PairOptions(lane=pair.FULL_LANE, mailbox="redis"))
    assert not (tmp_path / "traces" / "raw").exists()


def test_thermal_requeue_is_capped(env, monkeypatch):
    """A block voided by the thermal pause is rerun first, but only --max-thermal-requeues times (default 1): a further
    thermal void skips it — recorded with its run ids — and the pool moves on (review 2026-10-11; there was no cap)."""
    tmp_path, fake, insts = env
    jobs = _jobs(insts, ["A0"])
    holder, first = {}, {}

    def script(i, job):
        first.setdefault("block", job["block"])
        if job["block"] == first["block"]:      # this block overheats the box every time
            return "void", "abort", lambda: holder["pool"]._on_breach("thermal", "thermal: 93 °C >= 92 °C")
        return "completed", None, None
    run_pair, calls = _fake_run_pair(tmp_path, script)
    monkeypatch.setattr(pair, "run_pair", run_pair)
    p = pool.Pool(jobs, popts(tmp_path, fake, max_heat_voids=0))               # R26's breaker off: the cap alone
    holder["pool"] = p
    assert p.run() == 0 and p.halt_reason is None
    assert [c[0] for c in calls].count(first["block"]) == 2                     # two attempts, then skipped
    assert p.counts["blocks_skipped_thermal"] == 1 and p.counts["completed"] == 1 and p.counts["blocks_void"] == 2
    (sk,) = json.loads((tmp_path / "logs" / "pool-status.json").read_text())["skipped"]
    assert sk["block"] == first["block"] and sk["reason"] == "thermal" and sk["thermal_voids"] == 2
    assert len(sk["run_ids"]) == 1 and "__A0__" in sk["run_ids"][0]
    first.clear()
    import shutil
    shutil.rmtree(tmp_path / "traces")
    run_pair, calls = _fake_run_pair(tmp_path, script)
    monkeypatch.setattr(pair, "run_pair", run_pair)
    p = pool.Pool(jobs, popts(tmp_path, fake, max_thermal_requeues=0, max_heat_voids=0))   # no rerun at all
    holder["pool"] = p
    assert p.run() == 0 and [c[0] for c in calls].count(first["block"]) == 1
    assert p.counts["blocks_skipped_thermal"] == 1


def test_a_first_run_too_late_after_its_gate_wait_does_not_start(env, monkeypatch):
    """A gate wait (here the fan-governor's rest flag, 40 min on a fake clock) can push a run past the deadline: the
    check just before it starts keeps it from starting, and since it is its block's first run the block counts as
    never opened and the pool stops as at a deadline (review 2026-10-11)."""
    tmp_path, fake, insts = env
    base = datetime.now().astimezone()
    clock = [base]
    monkeypatch.setattr(pool.Pool, "_now", lambda self: clock[0])

    def rest_40_min():
        clock[0] += timedelta(minutes=40)
        return False
    monkeypatch.setattr(gates, "rest_due", rest_40_min)
    p = pool.Pool(_jobs(insts, ["A0"]), popts(tmp_path, fake, deadline=base + timedelta(minutes=30)))
    assert p.run() == 0 and p.opened == 0 and "deadline" in p.stopping and p.counts["completed"] == 0
    assert all(b.state == "unstarted" for b in p.blocks) and not (tmp_path / "traces" / "raw").exists()
    assert json.loads((tmp_path / "logs" / "pool-status.json").read_text())["state"] == "stopped"


def test_a_partly_run_block_keeps_running_past_a_late_check(env, monkeypatch):
    """The same check on the second run of an open block does not split the block: it runs (a stop only prevents new
    blocks), and the next block is not opened."""
    tmp_path, fake, insts = env
    base = datetime.now().astimezone()
    clock, n = [base], [0]
    monkeypatch.setattr(pool.Pool, "_now", lambda self: clock[0])

    def rest():
        n[0] += 1
        if n[0] == 2:                       # the block's second run waits 40 min at its gate
            clock[0] += timedelta(minutes=40)
        return False
    monkeypatch.setattr(gates, "rest_due", rest)
    p = pool.Pool(_jobs(insts, ["A0", "A1"]), popts(tmp_path, fake, deadline=base + timedelta(minutes=30)))
    assert p.run() == 0 and p.opened == 1 and p.counts["completed"] == 2 and "deadline" in p.stopping
    assert len(_records(tmp_path)) == 2 and {r["status"] for r in _records(tmp_path)} == {"completed"}


# ================================================================================================ D-021 (ITX-1)
def _state(cap=0, state="auto", age=1.0):
    return dict(gates.parse_fan_state(f"2026-10-11T13:30:19+02:00 state={state} for=60s zone=50 gpu=5W rest=no "
                                      f"cap={cap}"), raw=f"state={state} cap={cap}", age_s=age)


def test_governor_gate_holds_gpu_runs_only(env, monkeypatch):
    """D-021 R24: a GPU run starts only with the fan governor alive (state < 30 s old) and its cap off; a CPU-only run
    does not wait for it; a governor that stays gone halts the run set; --no-governor skips the check."""
    tmp_path, fake, insts = env
    states = iter([_state(cap=1), _state(cap=1), _state()] + [_state()] * 50)
    monkeypatch.setattr(gates, "fan_state", lambda *a, **k: next(states))
    p = pool.Pool(_jobs(insts[:1], ["A0", "A6F"]), popts(tmp_path, fake, thermal_poll_s=0.01))
    assert p.run() == 0 and p.counts["completed"] == 2
    monkeypatch.setattr(gates, "fan_state", lambda *a, **k: _state(age=45.0))          # stale: the governor is gone
    p = pool.Pool(_jobs(insts[1:], ["A6F"]), popts(tmp_path, fake, thermal_poll_s=0.01, governor_timeout_s=0.05))
    assert p.run() == 3 and "not running" in p.halt_reason
    p = pool.Pool(_jobs(insts[1:], ["A0"]), popts(tmp_path, fake, governor_timeout_s=0.05))
    assert p.run() == 0                                                               # CPU-only: no governor check
    p = pool.Pool(_jobs(insts[1:], ["A6F"]), popts(tmp_path, fake, governor=False))
    assert p.run() == 0 and p.counts["completed"] == 1


@pytest.mark.parametrize("cause", ["gpu_cap", "compiled_on_clock", "cuopt_memory", "external_kill"])
def test_d021_voids_requeue_once_then_skip(env, monkeypatch, cause):
    """A block voided by gpu_cap / compiled_on_clock / cuopt_memory / external_kill is rerun once (first, whole), then
    skipped and listed; the reruns are counted by cause; after a cap void the rerun waits for the fan on AUTO."""
    tmp_path, fake, insts = env
    jobs = _jobs(insts, ["A6F", "A9"])
    first, auto_waits = {}, []

    def script(i, job):
        first.setdefault("block", job["block"])
        if job["block"] == first["block"] and job["arm"] == "A6F":
            return "void", cause, None
        return "completed", None, None
    run_pair, calls = _fake_run_pair(tmp_path, script)
    monkeypatch.setattr(pair, "run_pair", run_pair)
    real_block = gates.governor_block
    monkeypatch.setattr(gates, "governor_block", lambda st, max_age_s=30.0, need_auto=False:
                        (auto_waits.append(need_auto), real_block(st, max_age_s, need_auto))[1])
    p = pool.Pool(jobs, popts(tmp_path, fake, max_heat_voids=0))               # R26's breaker: its own test
    assert p.run() == 0 and p.halt_reason is None
    assert [c[0] for c in calls if c[1] == "A6F"].count(first["block"]) == 2           # two attempts, then skipped
    assert p.counts["requeues"] == {cause: 1} and p.counts["blocks_skipped_cause"] == {cause: 1}
    (sk,) = json.loads((tmp_path / "logs" / "pool-status.json").read_text())["skipped"]
    assert sk["reason"] == cause and sk["voids"] == 2 and len(sk["run_ids"]) == 2
    assert any(auto_waits) == (cause == "gpu_cap")


def test_ref_rejected_reruns_the_block_once_with_the_stop_off(env, monkeypatch):
    """D-021 R12: a block voided by ref_rejected (the stop's incumbent not confirmed) is rerun first and whole with
    `ref_stop_off` in every job of the rerun — the same run ids —, once; a resumed pool finds the earlier void in the
    set's void.jsonl and keeps the stop off."""
    tmp_path, fake, insts = env
    jobs = [dict(j, harness="itx1") for j in _jobs(insts, ["A6F", "A9"])]
    first: dict = {}
    seen = []

    def script(i, job):
        first.setdefault("block", job["block"])
        seen.append((job["block"], job["arm"], job.get("ref_stop_off")))
        if job["block"] == first["block"] and job["arm"] == "A6F" and not job.get("ref_stop_off"):
            return "void", "ref_rejected", None
        return "completed", None, None
    run_pair, calls = _fake_run_pair(tmp_path, script)
    monkeypatch.setattr(pair, "run_pair", run_pair)
    p = pool.Pool(jobs, popts(tmp_path, fake, governor=False, max_heat_voids=0))
    monkeypatch.setattr(gates, "check_serving_stopped", lambda *a, **k: None)
    assert p.run() == 0 and p.halt_reason is None
    assert p.counts["requeues"] == {"ref_rejected": 1} and not p.counts["blocks_skipped_cause"]
    reruns = [s for s in seen if s[0] == first["block"] and s[2]]
    assert len(reruns) == 2 and all("rerun after ref_rejected" in s[2] for s in reruns)       # the whole block
    assert all(s[2] is None for s in seen if s[0] != first["block"])
    po = popts(tmp_path, fake).pair_opts
    j0 = next(j for j in jobs if j["block"] == first["block"])
    assert pair.run_id_for(dict(j0, ref_stop_off="x"), po) == pair.run_id_for(j0, po)
    # resume: the rerun's directories are gone (interrupted), the void.jsonl line remains
    import shutil
    for d in (tmp_path / "traces" / "raw" / "pytest").iterdir():
        if d.is_dir() and d.name != "void" and d.name.startswith(Path(j0["instance"]).stem):
            shutil.rmtree(d)
    seen.clear()
    p = pool.Pool(jobs, popts(tmp_path, fake, governor=False, max_heat_voids=0))
    assert p.run() == 0
    assert seen and all(s[2] and "void.jsonl, resume" in s[2] for s in seen if s[0] == first["block"])


def test_thermal_watch_void_requeues_as_thermal(env, monkeypatch):
    tmp_path, fake, insts = env
    run_pair, calls = _fake_run_pair(tmp_path, lambda i, j: ("void", "thermal_watch", None) if i == 0
                                     else ("completed", None, None))
    monkeypatch.setattr(pair, "run_pair", run_pair)
    p = pool.Pool(_jobs(insts, ["A6F"]), popts(tmp_path, fake))
    assert p.run() == 0 and p.counts["requeues"] == {"thermal": 1} and p.counts["completed"] == 2
    assert calls[1][0] == calls[0][0]                                                 # rerun first


def test_pause_c_bounds_and_the_watchdog(env, monkeypatch):
    """--pause-c (D-021 M2: 90 for the soak) is handed to the watchdog; never above 92 °C, never at or below 80."""
    tmp_path, fake, insts = env
    seen = {}
    real = gates.Watchdog

    def wd(*a, **k):
        seen.update(k)
        return real(*a, **k)
    monkeypatch.setattr(gates, "Watchdog", wd)
    assert pool.Pool(_jobs(insts[:1], ["A0"]), popts(tmp_path, fake, watchdog=True, pause_c=90.0)).run() == 0
    assert seen["pause_c"] == 90.0
    for bad in (93.0, 80.0):
        assert pool.Pool(_jobs(insts[1:], ["A0"]), popts(tmp_path, fake, pause_c=bad)).run() == 2
    assert pool.main([str(tmp_path / "none.jsonl"), "--pause-c", "90"]) == 2            # parses; the file is missing


def test_explicit_positions_fix_a_blocks_order():
    """A block whose every job carries `position` keeps that order (M2's AB / BA); partial or invalid positions are
    refused; blocks without positions keep the Williams order."""
    jobs = _jobs(["/x/a.mps", "/x/b.mps"], ["A9", "A6F"])
    for j in jobs:
        j["position"] = (0 if j["arm"] == "A9" else 1) if j["name"] == "a" else (1 if j["arm"] == "A9" else 0)
    got = {b.name: [r.label for r in b.runs] for b in pool.make_blocks(jobs)}
    assert got == {"a": ["A9", "A6F"], "b": ["A6F", "A9"]}
    jobs[0].pop("position")
    with pytest.raises(pool.PoolError, match="positions"):
        pool.make_blocks(jobs)
    jobs[0]["position"] = 5
    with pytest.raises(pool.PoolError, match="permutation"):
        pool.make_blocks(jobs)
    plain = _jobs(["/x/a.mps"], ["A9", "A6F"])
    (b,) = pool.make_blocks(plain)
    assert [r.label for r in b.runs] == pool.williams_order(["A9", "A6F"], "a", 1)


def test_harness_refusals_and_dry_run_arithmetic(env, capsys):
    """A bad harness is refused before anything runs; a harness on a test or screening job is refused (a prereg-v2
    item); --dry-run prints D-003's arithmetic per treatment, the thermal lines and the governor's state."""
    tmp_path, fake, insts = env
    bad = _jobs(insts[:1], ["A6F"])
    bad[0]["harness"] = "nope"
    with pytest.raises(pool.PoolError, match="unknown harness"):
        pool.make_blocks(bad)
    test = _jobs(insts[:1], ["A6F"], split="test")
    test[0]["harness"] = "ext"
    with pytest.raises(gates.GateError, match="prereg-v2"):
        pool.check_registered_harness(test, pair.PairOptions())
    jobs = _jobs(insts[:1], ["A6F", "A9"])
    for j in jobs:
        j["harness"] = "ext"
        j["config"] = "configs/itrea2-dev-rc-3way.yaml" if j["arm"] == "A9" else None
    o = popts(tmp_path, fake, dry_run=True)
    o.pair_opts.solver_mem_gb, o.pair_opts.ea_mem_gb, o.pair_opts.cuopt_mem_gb = 9.0, 6.0, 12.0
    assert pool.Pool(jobs, o).run() == 0
    out = capsys.readouterr().out
    assert "memory per run: A9 on g0 (harness \"ext\"): host [solver 9 + ea 6 + cuopt 24] + GPU [ea 12 + cuopt 24] + " \
           "2 CUDA x 1 + Redis 0 + reserve 20 = 97.0 GB" in out
    assert "memory per run: A6F on g0 (harness \"ext\"): host [cuopt 24] + GPU [cuopt 24] + 1 CUDA x 1" in out
    assert "pause / void at 92" in out and "fan governor: (test)" in out


def test_r26_breaker_after_two_heat_voids(env, monkeypatch):
    """D-021 R26: after 2 thermal or cap voids in a run set no new block opens and the pool ends `halted` with the
    reason; one thermal pause counts once; the breaker can be switched off (0)."""
    tmp_path, fake, insts = env
    insts = insts + [write_mps(set_cover(20, 12, seed=9 + i), tmp_path / f"more{i}.mps") for i in range(2)]
    jobs = _jobs(insts, ["A6F"])
    run_pair, calls = _fake_run_pair(tmp_path, lambda i, j: ("void", "gpu_cap", None) if i < 2
                                     else ("completed", None, None))
    monkeypatch.setattr(pair, "run_pair", run_pair)
    p = pool.Pool(jobs, popts(tmp_path, fake))
    assert p.run() == 3 and p.halt_reason is None and "R26" in p.breaker
    assert len(calls) == 2 and p.counts["completed"] == 0                        # nothing opened after the 2nd
    st = json.loads((tmp_path / "logs" / "pool-status.json").read_text())
    assert st["state"] == "halted" and "R26" in st["halt"]
    holder = {}

    def pause(i, job):                                  # two pauses, two episodes
        if i in (0, 1):
            return "void", "abort", lambda: holder["p"]._on_breach("thermal", "thermal: 93 °C >= 92 °C")
        return "completed", None, None
    import shutil
    shutil.rmtree(tmp_path / "traces")
    run_pair, calls = _fake_run_pair(tmp_path, pause)
    monkeypatch.setattr(pair, "run_pair", run_pair)
    p = pool.Pool(jobs, popts(tmp_path, fake))
    holder["p"] = p
    assert p.run() == 3 and p.thermal_episode == 2 and len(p.heat_voids) == 2
    shutil.rmtree(tmp_path / "traces")
    run_pair, calls = _fake_run_pair(tmp_path, lambda i, j: ("void", "gpu_cap", None) if i < 2
                                     else ("completed", None, None))
    monkeypatch.setattr(pair, "run_pair", run_pair)
    p = pool.Pool(jobs, popts(tmp_path, fake, max_heat_voids=0))                 # off: the first block is voided
    assert p.run() == 0 and p.breaker is None and p.counts["completed"] == 3      # twice and skipped, 3 others run
    assert p.counts["blocks_skipped_cause"] == {"gpu_cap": 1}


def test_unknown_clock_lock_halts_at_once(env, monkeypatch):
    """A cap void whose fan states read cap=0 before and after and that overlaps no cap interval is a lock the
    governor does not know about (a restart while capped): the pool halts at once."""
    tmp_path, fake, insts = env
    calls = []

    def run_pair(job, opts):
        rd = pair.run_dir_for(job, opts)
        rd.mkdir(parents=True)
        calls.append(job["block"])
        rec = dict(run_id=rd.name, status="void", void_cause="gpu_cap", reasons=["gpu_cap: 2200 < 2300"],
                   void_checks={"gpu_cap": "busy SM median 2200 MHz < 2300 MHz"},
                   void_info={"gpu_cap": dict(unknown_lock=True)}, arm=job["arm"])
        (rd / "run.json").write_text(json.dumps(rec))
        return rec
    monkeypatch.setattr(pair, "run_pair", run_pair)
    p = pool.Pool(_jobs(insts, ["A6F"]), popts(tmp_path, fake))
    assert p.run() == 3 and "unknown GPU clock lock" in p.halt_reason and len(calls) == 1


def test_rest_wait_halts_when_the_governor_is_gone(env, monkeypatch):
    """Only the governor removes its rest flag: a rest flag with a stale governor state for governor_timeout_s halts
    instead of idling until the lease ends; a live governor keeps the wait going."""
    tmp_path, fake, insts = env
    monkeypatch.setattr(gates, "rest_due", lambda *a, **k: True)
    monkeypatch.setattr(gates, "fan_state", lambda *a, **k: _state(age=300.0))
    p = pool.Pool(_jobs(insts[:1], ["A0"]), popts(tmp_path, fake, thermal_poll_s=0.01, governor_timeout_s=0.05))
    assert p.run() == 3 and "rest flag present but the fan governor is not running" in p.halt_reason
    n = []
    monkeypatch.setattr(gates, "rest_due", lambda *a, **k: (n.append(1), len(n) < 4)[1])
    monkeypatch.setattr(gates, "fan_state", lambda *a, **k: _state())
    p = pool.Pool(_jobs(insts[1:], ["A0"]), popts(tmp_path, fake, thermal_poll_s=0.01, governor_timeout_s=0.05))
    assert p.run() == 0 and p.counts["completed"] == 1


def test_harness_jobs_are_quiet(env, monkeypatch):
    """Jobs with a harness need both model servers stopped (R26): --allow-serving is refused and the serving check
    runs in preflight and before every run."""
    tmp_path, fake, insts = env
    jobs = _jobs(insts, ["A6F"])
    for j in jobs:
        j["harness"] = "ext"
    assert pool.is_quiet(jobs[0]) and not pool.is_quiet(_jobs(insts[:1], ["A6F"])[0])
    assert pool.Pool(jobs, popts(tmp_path, fake, allow_serving=True)).run() == 2
    seen = []
    monkeypatch.setattr(gates, "check_serving_stopped", lambda *a, **k: seen.append(1))
    run_pair, calls = _fake_run_pair(tmp_path, lambda i, j: ("completed", None, None))
    monkeypatch.setattr(pair, "run_pair", run_pair)
    assert pool.Pool(jobs, popts(tmp_path, fake)).run() == 0 and len(seen) >= 3      # preflight + each of 2 runs
