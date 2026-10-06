"""Safety gates every timed run passes (vault D-003): box-window lease, run-set memory budget, thermal state, the local
model servers' state, one runner at a time, and a live watchdog.

These encode the host's incidents: memwatch sheds on 2026-09-23/24/25/27/28 (per-process caps were not a budget), the
thermal shutdown of 2026-10-01 (acpitz near the 104 °C trip under sustained GPU load), and the fact that a systemd
`MemoryMax` scope does NOT charge GPU allocations on GB10 (measured 2026-10-05) — so GPU pool limits, the CUDA
context's own memory and the Redis container are counted here explicitly.

The memory formula (D-003):
    Σ host MemoryMax + Σ CuPy pool limits (+ cuOpt's RMM limit / measured peak) + Σ non-pool GPU overhead per CUDA
    process (1 GB each until measured) + the Redis container cap (2 GB, when Redis carries the exchange) + 20 GB reserve
    ≤ MemAvailable at launch.
Every reader takes an optional path / sysfs root so the tests can feed it fake files.
"""
from __future__ import annotations

import fcntl
import glob
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

LEASE = Path.home() / ".local/state/box-window/lease.json"
THERMAL_ROOT = Path("/sys/class/thermal")
RESERVE_GB = 20.0          # keep this much MemAvailable on top of every cap (memwatch soft floor ~12, hard ~7)
GPU_OVERHEAD_GB = 1.0      # non-pool GPU memory per CUDA process (context, cuSPARSE/cuBLAS handles, NVRTC) until measured
REDIS_CAP_GB = 2.0         # the itrea-redis container's memory cap (D-006)
WATCHDOG_FLOOR_GB = 18.0   # live: below this, stop the run set
THERMAL_WAIT_C = 80.0      # do not START a run at or above this (D-003: start only below 80 °C)
THERMAL_WAIT_C_CPU = 88.0  # CPU-only runs (no GPU process) may start below this (D-008: the X925 cores ran at their full
                           # 3.9 GHz at 88–89 °C with 8 SCIP lanes, 2026-10-06 02:54; the 80 °C start rule protects the
                           # GPU arms' clock state and only idled CPU lanes). Pause/resume (92/80 °C) are unchanged.
THERMAL_PAUSE_C = 92.0     # pause the run set at or above this ...
THERMAL_RESUME_C = 80.0    # ... until below this
SERVING_UNITS = ("ornith-vllm", "llama-qwen38")
# fan-governor's rest flag (host convention since 2026-10-02, Berk: "allow the spark to recover some … fans should not
# go max for prolonged periods"): written after 30 min of continuous fan MAX, removed once the fan is back on AUTO (idle
# and cool) and the rest has lasted its length (escalating 10–30 min). Launchers start nothing new while it exists.
REST_FLAG = Path.home() / ".local/state/thermal/rest"


class GateError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------------- memory
def mem_available_gb(meminfo: str | os.PathLike = "/proc/meminfo") -> float:
    for line in Path(meminfo).read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576
    return 0.0


@dataclass
class Budget:
    """What a run (or a set of concurrent runs) may take from MemAvailable, in GB."""
    host_caps_gb: float           # sum of systemd MemoryMax caps of the processes about to start (or in flight)
    gpu_caps_gb: float            # sum of CuPy pool limits + cuOpt's RMM limit / measured peak (NOT charged to a cgroup)
    reserve_gb: float = RESERVE_GB
    cuda_procs: int = 0           # processes that create a CUDA context (EA on CuPy, cuOpt)
    gpu_overhead_gb: float = GPU_OVERHEAD_GB   # per CUDA process, outside the pool
    redis_gb: float = 0.0         # REDIS_CAP_GB when the run uses the Redis exchange

    @property
    def need_gb(self) -> float:
        return (self.host_caps_gb + self.gpu_caps_gb + self.cuda_procs * self.gpu_overhead_gb + self.redis_gb
                + self.reserve_gb)

    def __add__(self, other: "Budget") -> "Budget":
        """Two run sets together; the reserve and the Redis container are counted once."""
        return Budget(self.host_caps_gb + other.host_caps_gb, self.gpu_caps_gb + other.gpu_caps_gb,
                      max(self.reserve_gb, other.reserve_gb), self.cuda_procs + other.cuda_procs,
                      max(self.gpu_overhead_gb, other.gpu_overhead_gb), max(self.redis_gb, other.redis_gb))

    def describe(self) -> str:
        return (f"host caps {self.host_caps_gb:.1f} + GPU caps {self.gpu_caps_gb:.1f} + {self.cuda_procs} CUDA x "
                f"{self.gpu_overhead_gb:.1f} + Redis {self.redis_gb:.1f} + reserve {self.reserve_gb:.1f} "
                f"= {self.need_gb:.1f} GB")


def check_memory(budget: Budget, avail_gb: float | None = None) -> float:
    """Raise GateError unless MemAvailable covers every cap plus the reserve. Returns MemAvailable."""
    avail = mem_available_gb() if avail_gb is None else avail_gb
    if avail < budget.need_gb:
        raise GateError(f"memory gate: MemAvailable {avail:.1f} GB < {budget.describe()}")
    return avail


# ---------------------------------------------------------------------------------------------------- thermal
def acpi_temps_c(root: str | os.PathLike | None = None) -> list[float]:
    """Every ACPI thermal zone (type `acpitz`) in °C. Unreadable zones are skipped."""
    root = Path(root) if root is not None else THERMAL_ROOT
    vals = []
    for z in sorted(glob.glob(str(root / "thermal_zone*"))):
        try:
            if Path(z, "type").read_text().strip() != "acpitz":
                continue
            vals.append(int(Path(z, "temp").read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return vals


def acpi_max_c(root: str | os.PathLike | None = None) -> float | None:
    """Hottest ACPI thermal zone in °C (None if unreadable)."""
    vals = acpi_temps_c(root)
    return max(vals) if vals else None


def rest_due(flag: str | os.PathLike | None = None) -> bool:
    """True while fan-governor's rest flag exists (D-009): start no new run; in-flight runs finish."""
    return Path(flag if flag is not None else REST_FLAG).exists()


def wait_cool(start_below: float = THERMAL_WAIT_C, timeout_s: float = 3600.0, poll_s: float = 30.0,
              log=print, root: str | os.PathLike | None = None) -> float | None:
    """Block until the hottest ACPI zone is below `start_below` (or no sensor). Returns the temperature."""
    end = time.monotonic() + timeout_s
    t = acpi_max_c(root)
    while t is not None and t >= start_below:
        if time.monotonic() > end:
            raise GateError(f"thermal gate: still {t:.1f} °C after {timeout_s:.0f} s")
        log(f"thermal gate: {t:.1f} °C >= {start_below:.0f} °C, waiting")
        time.sleep(poll_s)
        t = acpi_max_c(root)
    return t


# ------------------------------------------------------------------------------------------------- box-window
def lease(path: str | os.PathLike | None = None, now: datetime | None = None) -> dict | None:
    """The current box-window lease, or None if there is none, it is unreadable, or it has expired."""
    p = Path(path) if path is not None else LEASE
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    now = now or datetime.now().astimezone()
    try:
        until = datetime.fromisoformat(d["until"])
        if until.tzinfo is None:
            until = until.astimezone()
        if until <= now:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return d


def check_lease(who: str, path: str | os.PathLike | None = None, now: datetime | None = None) -> dict:
    """The run set must hold the current box-window lease (`box-window claim --who <who> …`)."""
    d = lease(path, now)
    if d is None:
        raise GateError(f"no box-window lease: run `box-window claim --who \"{who}\" --why … --until HH:MM` first")
    if d.get("who") != who:
        raise GateError(f"box-window lease is held by {d.get('who')!r}, not {who!r}")
    return d


def lease_minutes_left(path: str | os.PathLike | None = None, now: datetime | None = None) -> float:
    d = lease(path, now)
    if d is None:
        return 0.0
    now = now or datetime.now().astimezone()
    until = datetime.fromisoformat(d["until"])
    if until.tzinfo is None:
        until = until.astimezone()
    return (until - now).total_seconds() / 60.0


# ----------------------------------------------------------------------------------------------- serving units
STOPPED_STATES = ("inactive", "failed")   # every other state (active, activating, deactivating, reloading, unreadable)
                                          # counts as NOT stopped: the gate fails closed


def unit_state(unit: str) -> str | None:
    """`systemctl --user is-active <unit>` as text, or None when it cannot be read (no user bus — sudo -u, cron, a
    stripped environment —, OSError, timeout). Empty stdout is a bus/connection failure, not a state."""
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    s = r.stdout.strip()
    return s or None


def unit_active(unit: str) -> bool:
    return unit_state(unit) == "active"


def serving_state() -> dict[str, str | None]:
    """{unit: state or None}. Never raises: `snapshot()` (run records) uses it."""
    return {u: unit_state(u) for u in SERVING_UNITS}


def check_serving_stopped(state: dict | None = None) -> dict:
    """Confirmatory and screening windows run with both local model servers stopped (D-003, Berk's yes 2026-10-05).
    Only `inactive` / `failed` (or False in a bool dict) pass; an unreadable or transitional state refuses."""
    state = serving_state() if state is None else state
    bad = {u: s for u, s in state.items() if not (s is False or s in STOPPED_STATES)}
    if bad:
        raise GateError(f"serving gate: {bad} (unreadable or not stopped; stop them for the window: "
                        f"tools/serving-window.sh stop)")
    return state


# --------------------------------------------------------------------------------------------- co-running work
def co_running(min_gb: float = 1.0, exclude_pids: set[int] | None = None, proc: str = "/proc") -> list[dict]:
    """Processes above `min_gb` RSS other than ours: what else is on the box (names only, no command lines)."""
    exclude = exclude_pids or set()
    out = []
    for p in Path(proc).iterdir():
        if not p.name.isdigit() or int(p.name) in exclude:
            continue
        try:
            st = (p / "status").read_text()
        except OSError:
            continue
        rss = name = uid = None
        for line in st.splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1048576
            elif line.startswith("Name:"):
                name = line.split(None, 1)[1].strip() if len(line.split()) > 1 else ""
            elif line.startswith("Uid:"):
                uid = int(line.split()[1])
        if rss is not None and rss >= min_gb:
            out.append(dict(pid=int(p.name), name=name, uid=uid, rss_gb=round(rss, 2)))
    return sorted(out, key=lambda d: -d["rss_gb"])


# ------------------------------------------------------------------------------------------------------- lock
class RunLock:
    """Single-instance lock for a run set (flock on a file; released when the process dies)."""

    def __init__(self, path: str | os.PathLike = "/tmp/itrea-runner.lock"):
        self.path = Path(path)
        self.fh = None

    def __enter__(self):
        self.fh = open(self.path, "a+")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.fh.close()
            self.fh = None
            raise GateError(f"another runner holds {self.path}")
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(str(os.getpid()))
        self.fh.flush()
        return self

    def __exit__(self, *exc):
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None


# --------------------------------------------------------------------------------------------------- watchdog
class Watchdog(threading.Thread):
    """Background check while a run set is active: memory floor and thermal pause. Calls `on_breach(kind, reason)`
    once per breach episode (kind = "memory" | "thermal"); the runner writes the stop file of every run in flight and
    refuses new launches. `paused` is set from >= pause_c until the temperature falls below resume_c."""

    def __init__(self, on_breach, floor_gb: float = WATCHDOG_FLOOR_GB, pause_c: float = THERMAL_PAUSE_C,
                 resume_c: float = THERMAL_RESUME_C, poll_s: float = 5.0, mem_fn=None, temp_fn=None):
        super().__init__(daemon=True, name="itrea-watchdog")
        self.on_breach, self.floor_gb, self.pause_c, self.resume_c, self.poll_s = \
            on_breach, floor_gb, pause_c, resume_c, poll_s
        self.mem_fn = mem_fn or mem_available_gb
        self.temp_fn = temp_fn or acpi_max_c
        self.paused = threading.Event()
        self.stop_flag = threading.Event()
        self.min_avail_gb = float("inf")
        self.max_temp_c = float("-inf")
        self.breaches: list[dict] = []
        self._mem_low = False
        try:                                   # on_breach(kind, reason); older callbacks take only the reason
            import inspect
            self._two_args = len(inspect.signature(on_breach).parameters) >= 2
        except (TypeError, ValueError):
            self._two_args = True

    def check_once(self) -> None:
        avail = self.mem_fn()
        self.min_avail_gb = min(self.min_avail_gb, avail)
        if avail < self.floor_gb:
            if not self._mem_low:
                self._mem_low = True
                self._breach("memory", f"memory floor: MemAvailable {avail:.1f} GB < {self.floor_gb} GB")
        else:
            self._mem_low = False
        t = self.temp_fn()
        if t is not None:
            self.max_temp_c = max(self.max_temp_c, t)
            if t >= self.pause_c and not self.paused.is_set():
                self.paused.set()
                self._breach("thermal", f"thermal: {t:.1f} °C >= {self.pause_c} °C")
            elif t < self.resume_c and self.paused.is_set():
                self.paused.clear()

    def _breach(self, kind: str, reason: str) -> None:
        self.breaches.append(dict(t=time.time(), kind=kind, reason=reason))
        if self._two_args:
            self.on_breach(kind, reason)
        else:
            self.on_breach(reason)

    def run(self):
        while not self.stop_flag.is_set():
            try:
                self.check_once()
            except Exception as e:  # noqa: BLE001 — a sensor read error must not kill the watchdog
                self.breaches.append(dict(t=time.time(), kind="error", reason=repr(e)))
            self.stop_flag.wait(self.poll_s)

    def stop(self):
        self.stop_flag.set()


def rss_anon_gb(pid: int, proc: str = "/proc") -> float:
    """Anonymous resident memory of a process in GB (0 if gone). File-backed pages are left out on purpose: they sit
    in the page cache, which MemAvailable already counts as reclaimable."""
    try:
        for line in Path(proc, str(pid), "status").read_text().splitlines():
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 1048576
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def snapshot(exclude_pids: set[int] | None = None) -> dict:
    """What the run record stores about the box at a moment: MemAvailable, the hottest ACPI zone, the box-window
    lease, the local model servers' state and other large processes."""
    return dict(t=time.time(), mem_available_gb=round(mem_available_gb(), 2), acpi_max_c=acpi_max_c(),
                lease=lease(), serving=serving_state(), co_running=co_running(exclude_pids=exclude_pids)[:12])
