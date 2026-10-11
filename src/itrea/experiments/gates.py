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

D-021 adds readers of the host's thermal tools: fan-governor's state file (`fan_state`, `governor_block`: a GPU run
starts only with the governor alive and its clock cap off), its log's clock-cap intervals (`cap_intervals`,
`cap_overlaps`: a run overlapping one is a cap void) and thermal-watch's 98 °C stop lines (`thermal_watch_stops`).
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
from datetime import datetime, timedelta
from pathlib import Path

LEASE = Path.home() / ".local/state/box-window/lease.json"
THERMAL_ROOT = Path("/sys/class/thermal")
RESERVE_GB = 20.0          # keep this much MemAvailable on top of every cap (memwatch soft floor ~12, hard ~7)
GPU_OVERHEAD_GB = 1.0      # non-pool GPU memory per CUDA process (context, cuSPARSE/cuBLAS handles, NVRTC) until measured
REDIS_CAP_GB = 2.0         # the itrea-redis container's memory cap (D-006)
WATCHDOG_FLOOR_GB = 18.0   # live: below this, stop the run set
THERMAL_WAIT_C = 80.0      # do not START a run at or above this (D-003: start only below 80 °C)
THERMAL_WAIT_C_CPU = 86.0  # CPU-only runs (no GPU process) may start below this (D-008: the X925 cores ran at their full
                           # 3.9 GHz at 88–89 °C with 8 SCIP lanes, 2026-10-06 02:54; the 80 °C start rule protects the
                           # GPU arms' clock state and only idled CPU lanes). 88 until 2026-10-11; 86 since D-008's
                           # 10-06 addendum (88 gave 8 pauses in 24 min) and D-021 R24. Pause/resume (92/80 °C) unchanged.
THERMAL_PAUSE_C = 92.0     # pause the run set at or above this ...
THERMAL_RESUME_C = 80.0    # ... until below this
SERVING_UNITS = ("ornith-vllm", "llama-qwen38")
# fan-governor's rest flag (host convention since 2026-10-02, Berk: "allow the spark to recover some … fans should not
# go max for prolonged periods"): written after 30 min of continuous fan MAX, removed once the fan is back on AUTO (idle
# and cool) and the rest has lasted its length (escalating 10–30 min). Launchers start nothing new while it exists.
THERMAL_DIR = Path.home() / ".local/state/thermal"   # the host's thermal tools ([other repository] tools/bin)
REST_FLAG = THERMAL_DIR / "rest"
# fan-governor's state file (rewritten every 5 s: `<iso> state=<auto|max> for=<s>s zone=<C> gpu=<W>W rest=<yes|no>
# cap=<0|1>`) and its log ("fan -> max", "REST requested", "clock cap 2200 MHz on (…)", "clock cap off (…)"); its
# stage 2 locks the GPU clock at 2,200 MHz after a zone stays >= 85 °C for 20 s at fan MAX and releases it at AUTO.
FAN_STATE = THERMAL_DIR / "fan-state"
FAN_LOG = THERMAL_DIR / "fan-governor.log"
GOVERNOR_MAX_AGE_S = 30.0  # D-021 R24: the governor is alive when its state file is younger than this
# thermal-watch (every 30 s) appends `<iso> HOT <n>C stopped:<units>` to THERMAL_DIR/<YYYY-MM-DD>.log after stopping
# experiment units at >= 98 °C (D-021 R27: our named scopes `itrea-run-*` once its pattern lists them)
THERMAL_WATCH_C = 98.0


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


def _iso_epoch(s: str) -> float | None:
    try:
        t = datetime.fromisoformat(s.strip())
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.astimezone()
    return t.timestamp()


def parse_fan_state(line: str) -> dict:
    """One fan-state line -> {iso, epoch, state, for_s, zone_c, gpu_w, rest, cap}; fields that do not parse are None."""
    line = (line or "").strip()
    parts = line.split()
    kv = dict(p.split("=", 1) for p in parts[1:] if "=" in p)

    def num(v, suffix=""):
        try:
            return float(v[: -len(suffix)] if suffix and v.endswith(suffix) else v)
        except (TypeError, ValueError):
            return None
    cap = kv.get("cap")
    return dict(iso=parts[0] if parts else None, epoch=_iso_epoch(parts[0]) if parts else None, state=kv.get("state"),
                for_s=num(kv.get("for"), "s"), zone_c=num(kv.get("zone")), gpu_w=num(kv.get("gpu"), "W"),
                rest=(kv.get("rest") == "yes") if "rest" in kv else None,
                cap=int(cap) if cap is not None and cap.isdigit() else None)


def fan_state(path: str | os.PathLike | None = None, now: float | None = None) -> dict | None:
    """fan-governor's state now: the parsed line plus `raw` and `age_s` (seconds since the file was written), or None
    when the file is missing or unreadable."""
    p = Path(path) if path is not None else FAN_STATE
    try:
        raw = p.read_text().strip().splitlines()[-1]
        mtime = p.stat().st_mtime
    except (OSError, IndexError):
        return None
    d = parse_fan_state(raw)
    d.update(raw=raw, age_s=round((time.time() if now is None else now) - mtime, 1))
    return d


def governor_block(state: dict | None, max_age_s: float = GOVERNOR_MAX_AGE_S, need_auto: bool = False) -> str | None:
    """D-021 R24: why a GPU run may not start now by the fan governor's state (None = it may): no state file, a state
    older than max_age_s (the governor is not running), the clock cap on, or — for a block requeued after a cap void —
    the fan not yet back on AUTO."""
    if state is None:
        return "fan governor: no state file (not running?)"
    if state.get("age_s") is None or state["age_s"] >= max_age_s:
        return f"fan governor: state file {state.get('age_s')} s old (>= {max_age_s:g} s: not running?)"
    if state.get("cap") != 0:
        return f"fan governor: clock cap on (cap={state.get('cap')})"
    if need_auto and state.get("state") != "auto":
        return f"fan governor: fan {state.get('state')} (a cap-voided block waits for AUTO)"
    return None


def cap_intervals(log: str | os.PathLike | None = None) -> list[tuple[float, float | None]]:
    """The governor's clock-cap intervals from its log: (on, off) epochs; off = None while still on. A "start:" line (a
    governor restart, which forgets its cap) ends an open interval — a lock left behind is caught by the SM floor."""
    p = Path(log) if log is not None else FAN_LOG
    try:
        lines = p.read_text(errors="replace").splitlines()
    except OSError:
        return []
    out: list[tuple[float, float | None]] = []
    on: float | None = None
    for line in lines:
        iso, _, msg = line.partition(" ")
        t = _iso_epoch(iso)
        if t is None:
            continue
        if msg.startswith("clock cap") and msg.split("(")[0].rstrip().endswith(" on"):
            if on is None:
                on = t
        elif msg.startswith("clock cap off") or msg.startswith("start:"):
            if on is not None:
                out.append((on, t))
                on = None
    if on is not None:
        out.append((on, None))
    return out


def cap_overlaps(t_from: float, t_to: float, log: str | os.PathLike | None = None) -> list[dict]:
    """Governor clock-cap intervals that overlap [t_from, t_to] (epochs): [{on, off}] (off None = still on)."""
    return [dict(on=a, off=b) for a, b in cap_intervals(log) if a <= t_to and (b is None or b >= t_from)]


def thermal_watch_stops(t_from: float, t_to: float, root: str | os.PathLike | None = None) -> list[str]:
    """thermal-watch's 98 °C stop lines (`<iso> HOT <n>C stopped:<units>`) stamped in [t_from, t_to] (epochs), read
    from the daily logs of every date the window touches."""
    d = Path(root) if root is not None else THERMAL_DIR
    days, day, last = [], datetime.fromtimestamp(float(t_from)).date(), datetime.fromtimestamp(float(t_to)).date()
    while day <= last:
        days.append(day.isoformat())
        day += timedelta(days=1)
    out = []
    for day in days:
        try:
            lines = (d / f"{day}.log").read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            iso, _, msg = line.partition(" ")
            if not msg.startswith("HOT "):
                continue
            ts = _iso_epoch(iso)
            if ts is not None and t_from <= ts <= t_to:
                out.append(line)
    return out


def parse_hot_line(line: str) -> dict:
    """thermal-watch's stop line `<iso> HOT <n>C stopped:<units>` -> {epoch, c, units} (unit names without `.scope`;
    `stopped: none` gives no units)."""
    iso, _, msg = (line or "").partition(" ")
    c, units = None, []
    if msg.startswith("HOT "):
        head, _, rest = msg.partition("stopped:")
        try:
            c = float(head.split()[1].rstrip("C"))
        except (IndexError, ValueError):
            c = None
        units = [u[: -len(".scope")] if u.endswith(".scope") else u for u in rest.split() if u != "none"]
    return dict(epoch=_iso_epoch(iso), c=c, units=units)


def scope_result(unit: str) -> str | None:
    """The `Result` of a run's transient scope after its process ended (`oom-kill` for a kill at MemoryMax), or None
    when it cannot be read; a failed scope is reset afterwards so it does not linger in the user manager."""
    name = unit if unit.endswith(".scope") else unit + ".scope"
    try:
        r = subprocess.run(["systemctl", "--user", "show", "-p", "Result", "--value", name], capture_output=True,
                           text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    res = r.stdout.strip() or None
    if res and res != "success":
        try:
            subprocess.run(["systemctl", "--user", "reset-failed", name], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return res


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


def loadavg_1m() -> float | None:
    try:
        return round(os.getloadavg()[0], 2)
    except OSError:
        return None


def snapshot(exclude_pids: set[int] | None = None) -> dict:
    """What the run record stores about the box at a moment: MemAvailable, the hottest ACPI zone, the 1-min load
    average, the box-window lease, the local model servers' state and other large processes."""
    return dict(t=time.time(), mem_available_gb=round(mem_available_gb(), 2), acpi_max_c=acpi_max_c(),
                loadavg_1m=loadavg_1m(), lease=lease(), serving=serving_state(),
                co_running=co_running(exclude_pids=exclude_pids)[:12])
