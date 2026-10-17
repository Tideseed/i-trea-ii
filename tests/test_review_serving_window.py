"""Regression tests for tools/serving-window.sh (review finding F6).

The script runs with HOME pointed at a temp dir and stub commands in $HOME/.local/bin (the script puts that directory
first on PATH), so no real systemctl, box-window, hermes, curl or systemd-run is ever reached. Each stub appends its
argv to calls.log.
"""
from __future__ import annotations

import fcntl
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "serving-window.sh"
STUBS = ("systemctl", "curl", "hermes", "box-window", "systemd-run")


@pytest.fixture
def home(tmp_path):
    bin_dir = tmp_path / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    calls = tmp_path / "calls.log"
    for name in STUBS:
        body = f'#!/bin/bash\necho "{name} $*" >> "{calls}"\n'
        if name == "curl":       # health endpoints answer only when STUB_HEALTH=ok
            body += '[ "${STUB_HEALTH:-fail}" = ok ] || exit 22\n'
        if name == "systemctl":  # is-active / is-enabled print something, like the real one
            body += 'case "$2" in is-active) echo active;; is-enabled) echo enabled;; esac\n'
        p = bin_dir / name
        p.write_text(body + "exit 0\n")
        p.chmod(0o755)
    return tmp_path


def run(home: Path, *args: str, **env_extra: str) -> subprocess.CompletedProcess:
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
           "ITREA_RUNNER_LOCK": str(home / "runner.lock"), "ITREA_LOCK_WAIT_S": "1",
           "ITREA_HEALTH_TRIES": "2", "ITREA_HEALTH_SLEEP": "0", **env_extra}
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60)


def calls(home: Path) -> list[str]:
    f = home / "calls.log"
    return f.read_text().splitlines() if f.exists() else []


def log_text(home: Path) -> str:
    return (home / ".local/state/itrea/serving-window.log").read_text()


def write_hold(home: Path) -> Path:
    hold = home / ".config/cronrestore/hold-units"
    hold.parent.mkdir(parents=True)
    hold.write_text("other-unit until 2026-10-06T09:00:00+02:00\n"
                    "ornith-vllm until 2026-10-06T07:45:00+02:00  # i-trea-ii window\n")
    return hold


def test_restore_failed_health_exits_1_after_cleanup(home):
    hold = write_hold(home)
    r = run(home, "restore", STUB_HEALTH="fail")
    assert r.returncode == 1, r.stdout + r.stderr
    log = log_text(home)
    assert "RESTORE FAILED: vllm=0 llamacpp=0" in log
    c = calls(home)
    # the remaining steps still ran: canaries resumed, hold lines dropped, lease released, state lines logged
    assert any(x.startswith("hermes cron resume") for x in c)
    assert any(x.startswith("box-window release --who i-trea-ii (Claude Code)") for x in c)
    assert hold.read_text() == "other-unit until 2026-10-06T09:00:00+02:00\n"
    assert "ornith-vllm is-active=active is-enabled=enabled" in log


def test_restore_healthy_exits_0_and_renews_lease_first(home):
    r = run(home, "restore", STUB_HEALTH="ok")
    assert r.returncode == 0, r.stdout + r.stderr
    c = calls(home)
    i_claim = next(i for i, x in enumerate(c) if x.startswith("box-window claim --who i-trea-ii (Claude Code)"))
    i_start = next(i for i, x in enumerate(c) if x.startswith("systemctl --user start"))
    i_release = next(i for i, x in enumerate(c) if x.startswith("box-window release"))
    assert i_claim < i_start < i_release       # the lease covers the model load, released only at the end
    until = c[i_claim].split("--until ")[1].split()[0]
    left = datetime.fromisoformat(until) - datetime.now().astimezone()
    assert timedelta(minutes=40) < left <= timedelta(minutes=46)
    assert "RESTORE FAILED" not in log_text(home)
    assert "runner lock free" in log_text(home)


def test_restore_waits_for_runner_lock_then_starts_anyway(home):
    lock = home / "runner.lock"
    with open(lock, "a+") as fh:               # what gates.RunLock does while a run set is in flight
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = run(home, "restore", STUB_HEALTH="ok")
    assert r.returncode == 0, r.stdout + r.stderr
    log = log_text(home)
    assert "still held after 1s; starting the servers anyway" in log
    assert log.index("still held") < log.index("started ornith-vllm")
    assert any(x.startswith("systemctl --user start ornith-vllm") for x in calls(home))


def test_schedule_sets_onfailure_alert(home):
    r = run(home, "schedule", "2026-10-07 07:30:00")
    assert r.returncode == 0, r.stdout + r.stderr
    sr = [x for x in calls(home) if x.startswith("systemd-run ")]
    assert len(sr) == 1
    s = sr[0]
    assert "--user --unit itrea-restore-serving --on-calendar 2026-10-07 07:30:00" in s
    assert "-p OnFailure=agent-failure-alert@itrea-restore-serving.service.service" in s
    assert s.endswith(f"{SCRIPT} restore")
    assert run(home, "schedule").returncode == 2   # the time is required


def test_stop_holds_until_the_next_0745(home):
    r = run(home, "stop")
    assert r.returncode == 0, r.stdout + r.stderr
    lines = (home / ".config/cronrestore/hold-units").read_text().splitlines()
    assert len(lines) == 2 and all(l.endswith("# i-trea-ii window") for l in lines)
    until = datetime.fromisoformat(lines[0].split()[2])
    now = datetime.now().astimezone()
    assert (until.hour, until.minute, until.second) == (7, 45, 0)
    assert now < until <= now + timedelta(hours=24)   # never the 07:45 of the day after next
    assert [x for x in calls(home) if x.startswith("systemctl")] == \
        ["systemctl --user stop ornith-vllm", "systemctl --user stop llama-qwen38"]


def test_stop_respects_explicit_hold_until(home):
    r = run(home, "stop", ITREA_HOLD_UNTIL="2026-10-07T08:00:00+02:00")
    assert r.returncode == 0
    assert "until 2026-10-07T08:00:00+02:00" in (home / ".config/cronrestore/hold-units").read_text()
