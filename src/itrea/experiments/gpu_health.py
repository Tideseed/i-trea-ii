"""GPU health during a run (vault D-003, D-002 §6): SM clock, temperature, utilisation and foreign GPU processes,
sampled every second by a background thread.

Two faults void a GPU run's whole block (all arms of that instance × seed) and halt the run set for a look:
- **the stuck-low-clock fault** of the DGX Spark: the SM clock pinned near 800 MHz while the GPU reports P0, ~96 %
  utilisation and no throttle reason. It costs about 40 % of throughput and looks healthy. Healthy under load here:
  2,400–2,500 MHz. Gate: median SM clock below 2,000 MHz in a GPU run;
- **foreign GPU activity**: another process on the GPU (an LLM server, another session's job) for more than 2 % of the
  run's samples. optopt measured continuous co-inference doubling cuOpt's P(30) with the SM clock unchanged, so a
  foreign process is a confounder even when the clock is fine.

The clock median is taken over BUSY samples (utilisation >= `busy_util`): an idle GPU drops to ~290 MHz, and a run
whose partner spends part of the budget on the CPU (cuOpt's presolve, the EA's host loop) would otherwise read as the
fault. The stuck-clock fault itself shows ~96 % utilisation, so busy samples still catch it. The all-sample median is
recorded too.

GB10 reports `used_memory` of compute apps as `[N/A]` (unified memory); that is parsed to None.

The check fails CLOSED for a GPU run: when the timed window is known and at least `COVERAGE_MIN_WINDOW` periods long, a
run with fewer than `COVERAGE_MIN` of the expected samples (nvidia-smi errors, timeouts, no samples at all) is voided as
"GPU health unverifiable", so a night with a broken nvidia-smi halts the run set instead of passing unchecked.
"""
from __future__ import annotations

import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

GPU_QUERY = ["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"]
APPS_QUERY = ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
CLOCK_FLOOR_MHZ = 2000.0
FOREIGN_MAX_SHARE = 0.02
BUSY_UTIL = 30.0
COVERAGE_MIN = 0.5            # a GPU run needs at least half of the expected samples over its timed window ...
COVERAGE_MIN_WINDOW = 5.0     # ... once the window spans this many sampling periods (shorter toy windows: no rule)


def _num(s: str) -> float | None:
    s = s.strip()
    if not s or s.startswith("[") or s.upper() in ("N/A", "NA"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_gpu(text: str) -> dict | None:
    """First GPU line of GPU_QUERY's output -> {sm_mhz, temp_c, util}; None if unparsable."""
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        return dict(sm_mhz=_num(parts[0]), temp_c=_num(parts[1]), util=_num(parts[2]))
    return None


def parse_apps(text: str) -> list[tuple[int, float | None]]:
    """APPS_QUERY's output -> [(pid, used_memory MiB or None)]. Lines that are not "pid, mem" are skipped."""
    out = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        out.append((int(parts[0]), _num(parts[1]) if len(parts) > 1 else None))
    return out


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} rc={r.returncode}: {r.stderr.strip()[:200]}")
    return r.stdout


def ppid(pid: int, proc: str = "/proc") -> int | None:
    try:
        stat = Path(proc, str(pid), "stat").read_text()
    except OSError:
        return None
    # the command name may contain spaces and parentheses: the fields after the LAST ')' are fixed
    rest = stat[stat.rfind(")") + 2:].split()
    try:
        return int(rest[1])
    except (IndexError, ValueError):
        return None


def descends_from(pid: int, roots: set[int], proc: str = "/proc", max_depth: int = 64) -> bool:
    """True if `pid` is one of `roots` or a descendant of one (walks the parent chain)."""
    cur, depth = pid, 0
    while cur and cur > 1 and depth < max_depth:
        if cur in roots:
            return True
        cur = ppid(cur, proc)
        depth += 1
    return False


class Sampler(threading.Thread):
    """Samples the GPU every `period_s` until stop(). `own_pids()` returns the run's process ids (descendants are
    recognised by walking the parent chain); every other compute process is foreign.

    `query(cmd) -> str` is injectable for tests (default: run nvidia-smi)."""

    def __init__(self, own_pids: Callable[[], set[int]] | None = None, period_s: float = 1.0,
                 query: Callable[[list[str]], str] | None = None, clock_floor_mhz: float = CLOCK_FLOOR_MHZ,
                 foreign_max_share: float = FOREIGN_MAX_SHARE, busy_util: float = BUSY_UTIL, proc: str = "/proc"):
        super().__init__(daemon=True, name="itrea-gpu-health")
        self.own_pids = own_pids or (lambda: set())
        self.period_s = period_s
        self.query = query or _run
        self.clock_floor_mhz, self.foreign_max_share, self.busy_util = clock_floor_mhz, foreign_max_share, busy_util
        self.proc = proc
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self._halt = threading.Event()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ sampling
    def sample(self) -> dict | None:
        t = time.monotonic()
        try:
            g = parse_gpu(self.query(GPU_QUERY))
            apps = parse_apps(self.query(APPS_QUERY))
        except Exception as e:  # noqa: BLE001 — recorded; a missing sample is not a fault by itself
            if len(self.errors) < 20:
                self.errors.append(repr(e)[:200])
            return None
        if g is None:
            return None
        own = set(self.own_pids())
        foreign = [pid for pid, _ in apps if not descends_from(pid, own, self.proc)]
        s = dict(t=t, sm_mhz=g["sm_mhz"], temp_c=g["temp_c"], util=g["util"],
                 apps=[[pid, mem] for pid, mem in apps], foreign=foreign)
        with self._lock:
            self.samples.append(s)
        return s

    def run(self):
        while not self._halt.is_set():
            self.sample()
            self._halt.wait(self.period_s)

    def stop(self, join: bool = True) -> None:
        self._halt.set()
        if join and self.is_alive():
            self.join(timeout=10)

    # ------------------------------------------------------------------- verdict
    def summary(self, t_from: float | None = None, t_to: float | None = None, gpu_run: bool = True) -> dict:
        """Health over [t_from, t_to] (monotonic seconds). verdict: "ok" | "void" | "no-data". A GPU run with a known
        window of at least COVERAGE_MIN_WINDOW periods and too few samples is "void" (unverifiable); "no-data" remains
        only for CPU runs, an unknown window, or a window too short for a coverage rule. Only a GPU run (gpu_run=True)
        can be voided."""
        with self._lock:
            ss = [s for s in self.samples if (t_from is None or s["t"] >= t_from) and (t_to is None or s["t"] <= t_to)]
            errors = list(self.errors)
        return summarize(ss, gpu_run=gpu_run, clock_floor_mhz=self.clock_floor_mhz,
                         foreign_max_share=self.foreign_max_share, busy_util=self.busy_util,
                         errors=errors, period_s=self.period_s, t_from=t_from, t_to=t_to)


def summarize(samples: list[dict], gpu_run: bool = True, clock_floor_mhz: float = CLOCK_FLOOR_MHZ,
              foreign_max_share: float = FOREIGN_MAX_SHARE, busy_util: float = BUSY_UTIL,
              errors: list[str] | None = None, period_s: float = 1.0, t_from: float | None = None,
              t_to: float | None = None) -> dict:
    n = len(samples)
    expected = None
    if t_from is not None and t_to is not None and period_s > 0:
        expected = max(1.0, (float(t_to) - float(t_from)) / period_s)
    coverage_rule = bool(gpu_run and expected is not None and expected >= COVERAGE_MIN_WINDOW)
    out: dict = dict(n=n, period_s=period_s, gpu_run=gpu_run, clock_floor_mhz=clock_floor_mhz,
                     foreign_max_share=foreign_max_share, busy_util=busy_util, errors=list(errors or [])[:5],
                     expected_samples=round(expected, 1) if expected is not None else None, warnings=[])
    unverifiable = (f"GPU health unverifiable: {n}/{expected:.0f} samples (< {COVERAGE_MIN:.0%} of the timed window)"
                    if coverage_rule and n < COVERAGE_MIN * expected else None)
    if n == 0:
        if unverifiable:
            out.update(verdict="void", reasons=[unverifiable])
        else:
            out.update(verdict="no-data", reasons=["no GPU samples"])
        return out
    clocks = [s["sm_mhz"] for s in samples if s["sm_mhz"] is not None]
    busy = [s["sm_mhz"] for s in samples if s["sm_mhz"] is not None and (s["util"] or 0) >= busy_util]
    temps = [s["temp_c"] for s in samples if s["temp_c"] is not None]
    utils = [s["util"] for s in samples if s["util"] is not None]
    n_foreign = sum(1 for s in samples if s["foreign"])
    foreign_pids = sorted({p for s in samples for p in s["foreign"]})
    out.update(
        median_sm_mhz=statistics.median(busy) if busy else None,
        median_sm_mhz_all=statistics.median(clocks) if clocks else None,
        min_sm_mhz_busy=min(busy) if busy else None,
        n_busy=len(busy),
        max_temp_c=max(temps) if temps else None,
        mean_util=round(sum(utils) / len(utils), 1) if utils else None,
        gpu_busy_s=round(sum(utils) / 100.0 * period_s, 2) if utils else None,   # ~ NVML GPU-seconds used
        foreign_pids=foreign_pids,
        foreign_samples=n_foreign,
        foreign_share=round(n_foreign / n, 4),
    )
    reasons = []
    if unverifiable:
        reasons.append(unverifiable)
    if gpu_run and not busy:
        out["warnings"].append("no busy GPU samples: the clock check had nothing to judge")
    if gpu_run and busy and statistics.median(busy) < clock_floor_mhz:
        reasons.append(f"median SM clock {statistics.median(busy):.0f} MHz < {clock_floor_mhz:.0f} MHz over "
                       f"{len(busy)} busy samples (stuck-low-clock fault?)")
    if gpu_run and n_foreign / n > foreign_max_share:
        reasons.append(f"foreign GPU processes {foreign_pids[:8]} in {n_foreign}/{n} samples "
                       f"(> {foreign_max_share:.0%})")
    out["reasons"] = reasons
    out["verdict"] = "void" if reasons else "ok"
    return out


def compute_apps(query: Callable[[list[str]], str] | None = None) -> list[tuple[int, float | None]]:
    """The GPU's compute processes now (for the run record's snapshot); [] if nvidia-smi fails."""
    try:
        return parse_apps((query or _run)(APPS_QUERY))
    except Exception:  # noqa: BLE001
        return []
