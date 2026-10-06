"""Gates (vault D-003), GPU-health parsing and verdicts, and the Redis control's key-prefix discipline.

Everything here reads fake files (lease, sysfs, /proc/meminfo) or fake nvidia-smi output; the only live resource is
the itrea-redis container (skipped if it does not answer), touched under a test-only run-id prefix that is deleted."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from itrea.experiments import exchange_ctl, gates, gpu_health


# ------------------------------------------------------------------------------------------------ memory formula
def test_budget_formula_counts_every_term():
    b = gates.Budget(host_caps_gb=9.0 + 6.0, gpu_caps_gb=4.0, reserve_gb=20.0, cuda_procs=1, gpu_overhead_gb=1.0,
                     redis_gb=2.0)
    # D-003: Σ MemoryMax + Σ pool limits + 1 GB per CUDA process + Redis cap + 20 GB reserve
    assert b.need_gb == pytest.approx(15 + 4 + 1 + 2 + 20)
    assert gates.check_memory(b, avail_gb=42.0) == 42.0
    with pytest.raises(gates.GateError, match="memory gate"):
        gates.check_memory(b, avail_gb=41.9)


def test_budget_sum_counts_reserve_and_redis_once():
    a = gates.Budget(9.0, 0.0, cuda_procs=0, redis_gb=2.0)
    b = gates.Budget(6.0, 4.0, cuda_procs=1, redis_gb=2.0)
    s = a + b
    assert s.host_caps_gb == 15.0 and s.gpu_caps_gb == 4.0 and s.cuda_procs == 1
    assert s.redis_gb == 2.0 and s.reserve_gb == gates.RESERVE_GB
    assert s.need_gb == pytest.approx(15 + 4 + 1 + 2 + gates.RESERVE_GB)


def test_mem_available_reads_meminfo(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal:       131072000 kB\nMemFree:  1000 kB\nMemAvailable:   52428800 kB\n")
    assert gates.mem_available_gb(p) == pytest.approx(50.0)


# ------------------------------------------------------------------------------------------------------- lease
def _lease(tmp_path, **d) -> Path:
    p = tmp_path / "lease.json"
    p.write_text(json.dumps(d))
    return p


def test_lease_valid_expired_and_holder(tmp_path):
    now = datetime(2026, 10, 6, 1, 0).astimezone()
    p = _lease(tmp_path, who="i-trea-ii (Claude Code)", why="x", until=(now + timedelta(hours=2)).isoformat())
    assert gates.lease(p, now)["who"] == "i-trea-ii (Claude Code)"
    assert gates.check_lease("i-trea-ii (Claude Code)", p, now)
    assert gates.lease_minutes_left(p, now) == pytest.approx(120.0)
    with pytest.raises(gates.GateError, match="held by"):
        gates.check_lease("someone else", p, now)
    later = now + timedelta(hours=3)
    assert gates.lease(p, later) is None
    with pytest.raises(gates.GateError, match="no box-window lease"):
        gates.check_lease("i-trea-ii (Claude Code)", p, later)
    assert gates.lease_minutes_left(p, later) == 0.0


def test_lease_malformed_or_missing(tmp_path):
    now = datetime.now().astimezone()
    assert gates.lease(tmp_path / "missing.json", now) is None
    (tmp_path / "bad.json").write_text("{not json")
    assert gates.lease(tmp_path / "bad.json", now) is None
    assert gates.lease(_lease(tmp_path, who="x"), now) is None                      # no `until`
    assert gates.lease(_lease(tmp_path, who="x", until="tomorrow"), now) is None    # unparsable
    (tmp_path / "list.json").write_text("[1, 2]")
    assert gates.lease(tmp_path / "list.json", now) is None


def test_lease_naive_until_is_local_time(tmp_path):
    now = datetime.now().astimezone()
    naive = (now + timedelta(minutes=30)).replace(tzinfo=None).isoformat()
    assert gates.lease(_lease(tmp_path, who="x", until=naive), now) is not None


# ----------------------------------------------------------------------------------------------------- thermal
def _zone(root: Path, i: int, typ: str, milli: str) -> None:
    z = root / f"thermal_zone{i}"
    z.mkdir(parents=True)
    (z / "type").write_text(typ + "\n")
    (z / "temp").write_text(milli + "\n")


def test_acpi_max_reads_only_acpitz_zones(tmp_path):
    _zone(tmp_path, 0, "acpitz", "61000")
    _zone(tmp_path, 1, "acpitz", "78500")
    _zone(tmp_path, 2, "cpu-thermal", "99000")      # not an ACPI zone: ignored
    _zone(tmp_path, 3, "acpitz", "garbage")         # unreadable: skipped
    assert gates.acpi_temps_c(tmp_path) == [61.0, 78.5]
    assert gates.acpi_max_c(tmp_path) == 78.5


def test_acpi_max_without_sensors(tmp_path):
    assert gates.acpi_max_c(tmp_path) is None
    assert gates.wait_cool(80.0, root=tmp_path, log=lambda m: None) is None


def test_wait_cool_start_gate(tmp_path):
    _zone(tmp_path, 0, "acpitz", "79900")
    assert gates.wait_cool(80.0, root=tmp_path, log=lambda m: None) == pytest.approx(79.9)
    (tmp_path / "thermal_zone0" / "temp").write_text("80000\n")      # 80 °C is NOT below 80 °C
    msgs = []
    with pytest.raises(gates.GateError, match="thermal gate"):
        gates.wait_cool(80.0, timeout_s=0.05, poll_s=0.02, root=tmp_path, log=msgs.append)
    assert msgs and "waiting" in msgs[0]
    assert gates.THERMAL_WAIT_C == 80.0           # D-003: start only below 80 °C


# ---------------------------------------------------------------------------------------------------- watchdog
def test_watchdog_breaches_once_per_episode():
    mem = iter([50.0, 10.0, 9.0, 30.0, 12.0])
    temp = iter([70.0, 93.0, 95.0, 85.0, 79.0])
    seen = []
    wd = gates.Watchdog(lambda kind, reason: seen.append(kind), mem_fn=lambda: next(mem), temp_fn=lambda: next(temp))
    wd.check_once()                       # fine
    assert seen == [] and not wd.paused.is_set()
    wd.check_once()                       # memory low + hot
    assert seen == ["memory", "thermal"] and wd.paused.is_set()
    wd.check_once()                       # still low, still hot: no new breach
    assert seen == ["memory", "thermal"]
    wd.check_once()                       # memory recovers; 85 °C keeps the pause (resume below 80)
    assert wd.paused.is_set()
    wd.check_once()                       # memory low again = a new episode; 79 °C resumes
    assert seen == ["memory", "thermal", "memory"] and not wd.paused.is_set()
    assert wd.min_avail_gb == 9.0 and wd.max_temp_c == 95.0


def test_watchdog_accepts_one_argument_callbacks():
    seen = []
    wd = gates.Watchdog(seen.append, mem_fn=lambda: 1.0, temp_fn=lambda: None)
    wd.check_once()
    assert len(seen) == 1 and seen[0].startswith("memory floor")


def test_watchdog_thread_stops():
    wd = gates.Watchdog(lambda *a: None, mem_fn=lambda: 100.0, temp_fn=lambda: None, poll_s=0.01)
    wd.start()
    wd.stop()
    wd.join(timeout=2)
    assert not wd.is_alive()


# ------------------------------------------------------------------------------------------- lock and serving
def test_runlock_is_exclusive(tmp_path):
    p = tmp_path / "runner.lock"
    with gates.RunLock(p):
        with pytest.raises(gates.GateError, match="another runner"):
            gates.RunLock(p).__enter__()
    with gates.RunLock(p):                 # released
        pass


def test_serving_gate():
    assert gates.check_serving_stopped({"ornith-vllm": False, "llama-qwen38": False})
    with pytest.raises(gates.GateError, match="llama-qwen38"):
        gates.check_serving_stopped({"ornith-vllm": False, "llama-qwen38": True})


def test_co_running_reads_fake_proc(tmp_path):
    for pid, name, kb in ((101, "big", 3 * 1048576), (102, "small", 1000), (103, "mine", 5 * 1048576)):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "status").write_text(f"Name:\t{name}\nUid:\t1000\t1000\t1000\t1000\nVmRSS:\t{kb} kB\n")
    (tmp_path / "self").mkdir()
    out = gates.co_running(1.0, exclude_pids={103}, proc=str(tmp_path))
    assert [d["name"] for d in out] == ["big"] and out[0]["rss_gb"] == 3.0


# ------------------------------------------------------------------------------------------------ GPU health
def test_parse_nvidia_smi_lines():
    assert gpu_health.parse_gpu("2405, 71, 96\n") == dict(sm_mhz=2405.0, temp_c=71.0, util=96.0)
    assert gpu_health.parse_gpu("[N/A], 40, [N/A]\n") == dict(sm_mhz=None, temp_c=40.0, util=None)
    assert gpu_health.parse_gpu("") is None
    # GB10 reports used_memory as [N/A] (unified memory)
    assert gpu_health.parse_apps("1234, [N/A]\n5678, 512\n\n") == [(1234, None), (5678, 512.0)]
    assert gpu_health.parse_apps("No running processes found\n") == []


def _samples(clock, util, foreign_every=0, n=100):
    return [dict(t=float(i), sm_mhz=clock, temp_c=70.0, util=util, apps=[],
                 foreign=[999] if foreign_every and i % foreign_every == 0 else []) for i in range(n)]


def test_summary_flags_stuck_low_clock():
    s = gpu_health.summarize(_samples(810.0, 96.0))       # the GB10 fault: ~800 MHz at 96 % utilisation
    assert s["verdict"] == "void" and "stuck-low-clock" in s["reasons"][0]
    assert gpu_health.summarize(_samples(2450.0, 96.0))["verdict"] == "ok"
    # a CPU arm is never voided by the GPU's state
    assert gpu_health.summarize(_samples(810.0, 96.0), gpu_run=False)["verdict"] == "ok"


def test_summary_ignores_idle_clock():
    # an idle GPU drops to ~290 MHz: busy samples only decide the clock verdict
    ss = _samples(286.0, 0.0, n=60) + _samples(2450.0, 95.0, n=40)
    s = gpu_health.summarize(ss)
    assert s["verdict"] == "ok" and s["median_sm_mhz"] == 2450.0 and s["median_sm_mhz_all"] == 286.0


def test_summary_foreign_share_threshold():
    assert gpu_health.summarize(_samples(2450.0, 95.0, foreign_every=50))["verdict"] == "ok"      # 2 %: at the limit
    s = gpu_health.summarize(_samples(2450.0, 95.0, foreign_every=25))                            # 4 %
    assert s["verdict"] == "void" and s["foreign_pids"] == [999] and s["foreign_share"] == 0.04
    assert gpu_health.summarize([])["verdict"] == "no-data"


def test_sampler_with_fake_query_and_own_pids(tmp_path):
    # fake /proc: 300 is our process, 301 its child (a helper on the GPU), 400 someone else's
    for pid, ppid in ((300, 1), (301, 300), (400, 1)):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "stat").write_text(f"{pid} (py thon) S {ppid} 1 1 0\n")

    def query(cmd):
        if "--query-gpu=clocks.sm,temperature.gpu,utilization.gpu" in cmd:
            return "2400, 65, 90\n"
        return "300, [N/A]\n301, [N/A]\n400, [N/A]\n"

    s = gpu_health.Sampler(own_pids=lambda: {300}, query=query, proc=str(tmp_path))
    smp = s.sample()
    assert smp["foreign"] == [400]
    s.sample()
    summ = s.summary()
    assert summ["n"] == 2 and summ["foreign_share"] == 1.0 and summ["verdict"] == "void"


def test_sampler_records_query_errors():
    def query(cmd):
        raise RuntimeError("nvidia-smi gone")
    s = gpu_health.Sampler(query=query)
    assert s.sample() is None
    assert s.summary()["verdict"] == "no-data" and s.errors


# --------------------------------------------------------------------------------------------- exchange control
def test_flush_refuses_patterns():
    for bad in ("", "*", "a*", "x?y", "a[b]", "run:1", " lead"):
        with pytest.raises(exchange_ctl.ExchangeError):
            exchange_ctl.check_run_id(bad)
    assert exchange_ctl.check_run_id("air05__A3~no_bound_demote__s1000__300s__0123456789ab")
    assert exchange_ctl.check_run_id("k__A5@e4c01__s0__120s__abc")


def test_flush_deletes_only_its_run():
    if not exchange_ctl.ping():
        pytest.skip("itrea-redis is not answering on 127.0.0.1:6390")
    c = exchange_ctl.client()
    mine, other = f"pytest-gates-{uuid.uuid4().hex[:8]}", f"pytest-gates-{uuid.uuid4().hex[:8]}"
    try:
        c.set(f"itrea:{mine}:a", "1", ex=300)
        c.set(f"itrea:{mine}:b", "1", ex=300)
        c.set(f"itrea:{mine}x:c", "1", ex=300)        # a run id that merely starts with ours
        c.set(f"itrea:{other}:a", "1", ex=300)
        assert exchange_ctl.flush(mine, c) == 2
        assert c.exists(f"itrea:{mine}x:c") and c.exists(f"itrea:{other}:a")
        assert exchange_ctl.run_ids(c).get(other) == 1
    finally:
        c.delete(f"itrea:{mine}:a", f"itrea:{mine}:b", f"itrea:{mine}x:c", f"itrea:{other}:a")


def test_redis_container_args_match_d006():
    a = exchange_ctl.RUN_ARGS
    assert "--pull" in a and a[a.index("--pull") + 1] == "never"
    assert a[a.index("--cpuset-cpus") + 1] == "0" and a[a.index("--memory") + 1] == "2g"
    assert a[a.index("-p") + 1] == "127.0.0.1:6390:6379" and a[a.index("--restart") + 1] == "no"
    assert "noeviction" in a and "1536mb" in a and exchange_ctl.IMAGE == "redis:7-alpine"
