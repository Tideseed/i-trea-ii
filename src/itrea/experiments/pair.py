"""One run = one (instance, arm, seed, budget): launch the arm's processes, hold them at a shared start barrier, release
them together, end them, CHECK EVERY EXIT CODE, and write the run record (vault `10-design/Process interface.md`).

Why the runner is this strict (predecessor incidents, berkorbay/optopt):
- X1: a dead SCIP was logged "ok" because nobody read its exit code — here a run is `completed` only if every process
  exited 0, or was killed by the runner after a partner ended the race with a proven outcome (cuOpt has no interrupt
  API; `processes[*].stopped_by_runner`); anything else is `failed` (kept, truncated at the crash; D-002 §6) or `void`
  (an external cause: thermal, memory floor, GPU health, interruption, a launcher failure before ready — the whole
  block is rerun by the pool).
- a reused mailbox with a stale `start` file released a process at once on an old t0 — the runner alone creates the
  mailbox and refuses one that exists (file directory or Redis key space);
- start-up (imports, instance parse, CUDA context, kernel warm-up) happens before the barrier and is recorded; the
  clock starts for every process at the same monotonic `t0`;
- GPU memory is not charged to a systemd `MemoryMax` scope on GB10, so the EA's CuPy pool and cuOpt's RMM limit are
  passed explicitly (`CUPY_GPU_MEMORY_LIMIT`, `ITREA_GPU_MEM_GB`) and counted by the memory gate.

Footprint (D-002 §3): SCIP on one Cortex-X925 core (5–9, 15–19); the partner (EA on the GPU or the CPU, cuOpt) on the
GPU plus 3 Cortex-A725 cores (1–4, 10–14); core 0 belongs to the Redis container. Every process runs inside
`systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0` (`use_systemd=False` for tests) under `taskset`, with
OMP/OPENBLAS/MKL_NUM_THREADS = 1 — except a whole-lane process (A6m's cuOpt), from whose environment the runner
REMOVES the three (they are also in the runner's own environment, set below at import): GAMS ran cuOpt with them
unset, and libcuopt links libomp (`Proc.env_unset`, recorded in run.json). The three-way arm A9 (SCIP <-> EA <- cuOpt, D-015) runs two partners, which must not
share cores: SCIP on the lane's first X925 core, the EA on its second (`ProcSpec.x925`), cuOpt on the A725 cores
(num_cpu_threads = 3 = their number on the GPU lane); two CUDA processes, so the memory gate counts the EA pool and
cuOpt's RMM limit (both = gpu_mem_gb) plus two CUDA overheads. The runner gives the EA `exchange.from_cuopt: true`
(arm-implied, like A5's "no exchange") and cuOpt `--send` (its own channels `to_ea_cuopt` / `info_cuopt`); SCIP gets
A3's flags. GPU-health voiding treats every process the runner launched (and its descendants) as the run's own. A9's
matched control is A3x (SCIP on x925[0], the EA on x925[1], no cuOpt), not A3: A3's EA runs on the three A725 cores,
and its HiGHS LP worker (`seeding.lp_source: highs_async`, which inherits the EA's affinity) beside it, while A9's EA
shares one faster X925 core with that worker.

Arm A6m (D-020, the MIPfeas calibration) is cuOpt alone as GAMS ran it for Mittelmann's MIPfeas: num_cpu_threads 24
and mip_absolute_gap = mip_relative_gap = 0 (`--mip-gap-zero`), whatever the lane, pinned to every core of the
whole-box lane `full` (all 20 cores, `FULL_LANE`; `--layout full` in the pool) and refused on any other lane, as the
whole-box lane refuses every other arm. Its run hash folds in the 24 threads and the zero gaps, so it never shares a
run id with A6. Its RMM limit and MemoryMax come from the runner's options (D-020: `--gpu-mem-gb 36 --cuopt-mem-gb
24`); the RMM limit is part of the run hash, as for A6.

ITX-1 (D-021), on the GPU lane g0 whose cores are F (X925 5–6 + A725 1–3 + the GPU today, taken from the lane):
- **A6F** = cuOpt alone pinned to all of F, num_cpu_threads = |F| (`ProcSpec.f_share`); **A2F** = SCIP on F's first
  X925 core, cuOpt on the other |F| - 1 cores with |F| - 1 threads, no exchange (as A2). F's size is part of their
  run hash; with a harness F's cores themselves are (`harness_footprint`: a changed F is another configuration, R5);
- the job-level **harness** (`resolve_harness`; preset "ext" or a dict of overrides): every cuOpt process gets MIP gaps
  0 (R13), the thread variables removed (R14), its own RMM limit and host MemoryMax (R29: 24 / 24 GB, separate from the
  EA's 12-GB pool, which the EA keeps); every process gets the project's CUDA JIT cache (`CUDA_CACHE_PATH`,
  `CUDA_CACHE_MAXSIZE`, R15); `cuopt_memory_void` (default true; M3: false) says whether a confirmed cap hit of cuOpt
  voids the run. The resolved dict and its footprint (F's cores, SCIP's and the EA's host caps) enter the run hash only
  when the field is present, so every run id without it is unchanged; it is refused on A6m and (pool) on test /
  screening jobs;
- **named scopes** `itrea-run-<sha1(run directory name)[:12]>-<role>-<epoch_ms>` (R27; unique per launch; never part
  of a hash or of argv), each with `TimeoutStopSec=5s`, cuOpt's also `KillSignal=SIGKILL` (no interrupt API);
- **void causes** (`void_checks`, after the run, in this order): `thermal_watch` (a thermal-watch 98 °C stop line that
  names one of this run's scopes, R27), `gpu_cap` (a GPU run with a busy SM median below `cap_floor_mhz` = 2,300 MHz,
  overlapping a fan-governor clock-cap interval, or with a fan state stamped inside the timed window that reads cap=1,
  R25), `cuopt_memory` (harness: cuOpt's cap CONFIRMED — an RMM allocation error with the peak at >= 0.8 x the limit,
  or a SIGKILL with the scope's Result oom-kill or maxrss >= 0.95 x MemoryMax, R29), `external_kill` (any arm: a
  SIGKILL of any process the runner did not send that is neither a cap kill nor thermal-watch's), `compiled_on_clock`
  (harness: an entry of the CUDA cache — or the EA's CuPy cache — written after t0, R15). A failure of SCIP or the EA,
  their caps included, stays an arm outcome; what decides nothing is recorded in `void_info`. run.json also records
  the fan governor's state at the start and the end (`fan_state_start` / `fan_state_end`) and the start temperature
  (`acpi_start_c`, R28).

D-021 phase 2 (two optional harness switches, absent from the resolved harness — and so from the run hash — unless on;
preset `itx1` = ext + both, carried by the v1 tier jobs and A0's external jobs):
- **`ref_stop` (R12, stop at the proven optimum):** for an instance MIPLIB lists as `=opt=` (`miplib_stop_ref`; never
  supportcase22's `=best=`, never PGLib-UC or ML4CO) the runner reads z* and the objective (c, offset) before t0 —
  no process receives z* — and while the clock runs tails every process's incumbent records (`RefMonitor`: the
  `.part` files and `names.json`, cuOpt's in A6F / A2F / A9 included), recomputes obj = c·x + offset from each stored
  vector and ends the run (stop kind `ref_reached`, `stop` written to the mailbox) at the first record before T — the
  earliest by its time stamp over every process, not the first role read — whose GAMS gamma
  round((obj − z*)/max(|obj|, |z*|, 1), 6) is <= 0 AND whose Berthold gap is <= 1e-9 (`ref_reached`). After
  the processes ended, the triggering incumbent is checked by the independent exact checker (`check_ref_trigger`);
  unconfirmed, the run is void (`ref_rejected`) and the pool reruns its block once with the stop off (`ref_stop_off`
  in the job, a runtime switch outside the run id). Processes killed after the stop are `stopped_by_runner` (not
  crashes), and the last checked incumbent is held to T in every metric, so `pt_gams` and Berthold's P are those of a
  run to T (M0's replay checks it). run.json `ref_stop`: the reference, the records read per role, the trigger, every
  role's first qualifying record (`candidates`), the check;
- **`cuopt_kill_on_end`:** after a definitive outcome (`definitive_stop`: a partner's proof — exit 0 with an exit
  reason in DEFINITIVE — or ref_reached; not a partner's crash) every cuOpt process still running is SIGKILLed at once
  (`killed` "SIGKILL (definitive outcome)", stopped_by_runner) instead of after kill_after_s + 10 s; without the
  switch the runner behaves as before.

Process launch, barrier and stop adapted from berkorbay/optopt@4dd48395 (MIT), `src/optopt/experiments/xchg_pair.py`.

    python -m itrea.experiments.pair --instance /tmp/toy.mps --arm A0 --seed 0 --budget 10 --set smoke

A toy (outside data/, not a screening candidate) runs as is. Any other instance (e.g. a PGLib-UC `ferc` case) needs the
box-window lease, the thermal start gate and systemd scopes, and a screening candidate needs the `prereg-v1` tag
(D-002 §1); the runner lock is always taken, so this never runs beside a pool.
"""
from __future__ import annotations

import os as _os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # the runner itself, before NumPy loads
    _os.environ.setdefault(_k, "1")

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import numpy as np
import yaml

from itrea import paths
from itrea.config import EAConfig
from itrea.experiments import gates, gpu_health, record_env
from itrea.trace import read_events

ROLE_MODULES = {"scip": "itrea.solvers.scip_run", "highs": "itrea.solvers.highs_run",
                "cuopt": "itrea.solvers.cuopt_run", "ea": "itrea.ea.run"}
SOLVER_KINDS = ("scip", "highs")
GPU_KINDS = ("cuopt",)          # plus "ea" when its backend is cupy
A1_SEED_OFFSET = 10_000         # D-002 v3 §3: A1's second SCIP uses seed s + 10,000
PARTNER_THREADS = 3             # D-002 §3: the partner footprint is 3 A725 cores (cuOpt num_cpu_threads = 3)
DEFINITIVE = ("optimal", "infeasible", "unbounded", "infeasible_or_unbounded")   # exit reasons that end a race
ACPI_SAMPLE_S = 5.0            # the hottest ACPI zone, read this often while the clock runs (run.json `acpi_during`)
ABORT_KILL_AFTER_S = 5.0      # an abort (memory floor, thermal pause, interrupt): stop → this → SIGTERM → SIGKILL
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
RUNNER_CORES = (4, 10, 11, 12, 13, 14)   # A725 cores no lane uses (1–3: partner, 0: Redis): the runner's own work —
                                         # except the whole-box lane `full`, which holds every core: there the runner's
                                         # threads (the GPU-health sampler, the stop loop) share cuOpt's cores, a small,
                                         # recorded load (run.json `runner_cores`) accepted for A6m (D-020)
X925_CORES = (5, 6, 7, 8, 9, 15, 16, 17, 18, 19)
A725_CORES = (0, 1, 2, 3, 4, 10, 11, 12, 13, 14)
GAMS_CUOPT_THREADS = 24         # D-020: GAMS ran cuOpt with num_cpu_threads 24 on the 20-core GB10 (arm A6m)
MAILBOX_ROOT = Path("/dev/shm/itrea")
CUPY_CACHE_DIR = Path.home() / ".cache/itrea/cupy"
CUDA_CACHE_DIR = Path.home() / ".cache/itrea/cuda"   # D-021 R15: the project's own CUDA JIT cache (harness cuda_cache)
CUDA_CACHE_MAXSIZE = 4294967296                       # 4 GiB: large enough never to evict (the shared cache holds 1.1 GB)
REDIS_URL = "redis://127.0.0.1:6390/0"
# D-021 (ITX-1) harness settings: a job-level field `harness` = a preset name or {"preset": name, <key>: value, ...}.
# Absent = the runner as before (no key enters the hash); present = the resolved dict (every key below) enters the run
# hash. cuOpt keys act on every cuOpt process of the arm (A9, A6F, A2F, A6, A2, H2); cuda_cache on every process.
HARNESS_KEYS = ("cuopt_gap_zero", "cuopt_env_unset", "cuopt_rmm_gb", "cuopt_host_gb", "cuda_cache", "cuopt_memory_void")
# D-021 phase 2: optional switches, false by default and present in the resolved dict ONLY when true, so a harness that
# leaves them off (every phase-1 job, preset "ext", or a dict setting them false) resolves — and hashes — as before.
#   ref_stop           R12: the run ends when its best incumbent reaches the proven optimum (`RefMonitor`; MIPLIB =opt=
#                      references only), stop kind `ref_reached`
#   cuopt_kill_on_end  after a definitive outcome (a partner's proof, ref_reached) every cuOpt process still running is
#                      SIGKILLed at once (it has no interrupt API) instead of after kill_after_s + 10 s
HARNESS_OPTIONAL = ("ref_stop", "cuopt_kill_on_end")
HARNESS_PRESETS: dict[str, dict] = {
    # R13 gaps 0, R14 cuOpt's thread variables unset, R29 cuOpt RMM 24 GB separate from the EA's pool and host
    # MemoryMax 24 GB, R15 the project-own CUDA JIT cache for every process; a confirmed cap hit of cuOpt voids the run
    # (R29; M3's memory probe sets cuopt_memory_void false: the cap hit is its measurement, recorded, not voided)
    "ext": dict(cuopt_gap_zero=True, cuopt_env_unset=True, cuopt_rmm_gb=24.0, cuopt_host_gb=24.0, cuda_cache=True,
                cuopt_memory_void=True),
    # ITX-1's v1 external runs (D-021 phase 2): ext + R12's stop at the proven optimum + cuOpt killed at a definitive
    # outcome; the tier jobs (`make-jobs itx-v1`) and A0's external jobs (`itx-a0`) carry it
    "itx1": dict(cuopt_gap_zero=True, cuopt_env_unset=True, cuopt_rmm_gb=24.0, cuopt_host_gb=24.0, cuda_cache=True,
                 cuopt_memory_void=True, ref_stop=True, cuopt_kill_on_end=True),
}
_HARNESS_BOOL = ("cuopt_gap_zero", "cuopt_env_unset", "cuda_cache", "cuopt_memory_void") + HARNESS_OPTIONAL
SCOPE_STOP_TIMEOUT = "5s"   # every run scope: systemd's stop (thermal-watch's `systemctl stop`) kills within 5 s, not 90
RMM_CAP_SHARE = 0.8         # an RMM allocation error counts as cuOpt's cap only when its peak reached this share of the
                            # limit (R29's 0.8 x cap reading)
HOST_CAP_SHARE = 0.95       # a SIGKILL counts as a host-cap kill when maxrss reached this share of MemoryMax (or the
                            # scope's Result says oom-kill)
_HARNESS_GB = ("cuopt_rmm_gb", "cuopt_host_gb")


class RunError(RuntimeError):
    """The run could not be set up (bad job, stale state); nothing was launched."""


class StaleRunError(RunError):
    pass


class StaleMailboxError(RunError):
    pass


# --------------------------------------------------------------------------------------------------------- arms
@dataclass(frozen=True)
class ProcSpec:
    kind: str                  # "scip" | "highs" | "cuopt" | "ea"
    role: str                  # mailbox role: "solver" | "solver2" | "ea" | "cuopt"
    seed_offset: int = 0
    exchange: bool = False     # solver: --recv/--send derived from the EA configuration (A3, A4, H3); cuOpt: --send
                               # when the EA configuration has exchange.from_cuopt (A9)
    emphasis: bool = False     # solver: --emphasis from the job (A0*)
    backend: str | None = None  # ea: "cupy" | "numpy"
    x925: bool = False         # ea: one X925 core of its own, after the solvers' (A9: cuOpt holds the A725 cores;
                               # A3x: the same footprint without cuOpt)
    cpu_threads: int | None = None   # cuopt: num_cpu_threads fixed by the arm, whatever the lane (A6m: 24, D-020);
                                     # None = the partner footprint's (PairOptions.cuopt_threads or PARTNER_THREADS)
    gap_zero: bool = False     # cuopt: --mip-gap-zero, both MIP gaps 0 as GAMS ran it (A6m, D-020)
    whole_lane: bool = False   # pinned to every core of its lane (A6m on the whole-box lane)
    f_share: bool = False      # cuopt: every core of the lane F that no other process of the arm holds (F minus the
                               # arm's X925 cores: A6F all of F, A2F F minus SCIP's core), num_cpu_threads = their
                               # number (D-021 R6 / R8)

    @property
    def gpu(self) -> bool:
        return self.kind in GPU_KINDS or (self.kind == "ea" and self.backend == "cupy")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    procs: tuple[ProcSpec, ...]
    ea_overrides: tuple[tuple[str, object], ...] = ()   # arm-implied EA configuration changes
    lane: str = "solo"         # "partner": GPU lane / partner footprint (one at a time) | "solo": solver-only CPU lane
                               # | "full": the whole-box lane only (A6m, D-020)
    doc: str = ""

    @property
    def gpu(self) -> bool:
        return any(p.gpu for p in self.procs)

    @property
    def has_ea(self) -> bool:
        return any(p.kind == "ea" for p in self.procs)

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(p.role for p in self.procs)

    @property
    def n_x925(self) -> int:
        return sum(1 for p in self.procs if p.kind in SOLVER_KINDS or p.x925)

    @property
    def needs_a725(self) -> bool:
        return any((p.kind == "ea" and not p.x925) or p.kind == "cuopt" for p in self.procs)


_SCIP = ProcSpec("scip", "solver")
_SCIP_X = ProcSpec("scip", "solver", exchange=True)
_HIGHS = ProcSpec("highs", "solver")
_HIGHS_X = ProcSpec("highs", "solver", exchange=True)
_EA_GPU = ProcSpec("ea", "ea", backend="cupy")
_EA_CPU = ProcSpec("ea", "ea", backend="numpy")
_CUOPT = ProcSpec("cuopt", "cuopt")
_CUOPT_X = ProcSpec("cuopt", "cuopt", exchange=True)                  # A9: --send on cuOpt's own channels
_EA_GPU_X925 = ProcSpec("ea", "ea", backend="cupy", x925=True)        # A9: the A725 cores belong to cuOpt
_CUOPT_GAMS = ProcSpec("cuopt", "cuopt", cpu_threads=GAMS_CUOPT_THREADS, gap_zero=True, whole_lane=True)   # A6m
_CUOPT_F = ProcSpec("cuopt", "cuopt", f_share=True)                   # A6F / A2F: the lane's cores the arm leaves free
_NOX = (("exchange.enabled", False), ("lp_stream.enabled", False))   # the stream needs the exchange (review 770df17)

ARMS: dict[str, ArmSpec] = {a.name: a for a in [
    ArmSpec("A0", (_SCIP,), doc="SCIP alone, harness-matched (plugin installed and inert, null partner)"),
    ArmSpec("A0star", (ProcSpec("scip", "solver", emphasis=True),), doc="SCIP alone with the tuned static emphasis"),
    ArmSpec("SCREEN", (_SCIP,), doc="screening pass: SCIP default, plugin inert (D-002 §1)"),
    ArmSpec("A1", (_SCIP, ProcSpec("scip", "solver2", seed_offset=A1_SEED_OFFSET)),
            doc="SCIP || SCIP (seed s + 10,000), no exchange"),
    ArmSpec("A2", (_SCIP, _CUOPT), lane="partner", doc="SCIP || cuOpt race, no exchange (as-shipped cuOpt)"),
    ArmSpec("A3", (_SCIP_X, _EA_GPU), lane="partner", doc="SCIP <-> I-TREA II on the GPU, two-way exchange"),
    ArmSpec("A3r", (_SCIP, _EA_GPU), _NOX, lane="partner", doc="SCIP || I-TREA II on the GPU, NO exchange"),
    ArmSpec("A4", (_SCIP_X, _EA_CPU), lane="partner", doc="SCIP <-> I-TREA II on the CPU (NumPy), two-way"),
    ArmSpec("A5", (_EA_GPU,), _NOX, lane="partner", doc="I-TREA II alone (GPU)"),
    ArmSpec("A5x0", (_EA_GPU,), _NOX + (("ranks.x", 0.0),), lane="partner", doc="A5 with x = 0 (no Rank 0)"),
    ArmSpec("A5y0", (_EA_GPU,), _NOX + (("ranks.y", 0.0),), lane="partner", doc="A5 with y = 0 (no protected Rank 1)"),
    ArmSpec("A6", (_CUOPT,), lane="partner", doc="cuOpt alone"),
    ArmSpec("H0", (_HIGHS,), doc="HiGHS alone, harness-matched"),
    ArmSpec("H2", (_HIGHS, _CUOPT), lane="partner", doc="HiGHS || cuOpt race"),
    ArmSpec("H3", (_HIGHS_X, _EA_GPU), lane="partner", doc="HiGHS <-> I-TREA II on the GPU, two-way"),
    ArmSpec("A9", (_SCIP_X, _EA_GPU_X925, _CUOPT_X), (("exchange.from_cuopt", True),), lane="partner",
            doc="SCIP <-> I-TREA II on the GPU <- cuOpt (D-015): cuOpt (as shipped, never receives) streams its "
                "incumbents to the EA, which exchanges with SCIP as in A3; development arm"),
    ArmSpec("A3x", (_SCIP_X, _EA_GPU_X925), lane="partner",
            doc="A3 with A9's EA footprint (the EA on the lane's second X925 core, no cuOpt): A9's matched control "
                "(D-015 addendum); development arm"),
    ArmSpec("A6m", (_CUOPT_GAMS,), lane="full",
            doc="cuOpt alone as GAMS ran it for MIPfeas (D-020): num_cpu_threads 24, MIP gaps 0, all 20 cores "
                "(whole-box lane, --layout full); out of v1 (D-021 R9)"),
    ArmSpec("A6F", (_CUOPT_F,), lane="partner",
            doc="cuOpt alone on all of F, A9's footprint (D-021 R6): pinned to every core of the GPU lane, "
                "num_cpu_threads = |F|; ITX-1's 'cuOpt alone' and its calibration arm"),
    ArmSpec("A2F", (_SCIP, _CUOPT_F), lane="partner",
            doc="SCIP || cuOpt race on F (D-021 R8): SCIP on F's first X925 core, cuOpt on the other |F| - 1 cores "
                "with |F| - 1 threads, no exchange (as A2); the live race"),
]}
# canonical order (the Williams design permutes the arms of a block in this order)
ARM_ORDER = tuple(ARMS)


def arm_label(job: dict) -> str:
    """The treatment label: the arm, plus `~<ablation>` for a named ablation (configs/ablations/<name>.yaml) and
    `@<variant>` for a configuration variant (E4 tuning: one label per candidate configuration)."""
    return job["arm"] + (f"~{job['ablation']}" if job.get("ablation") else "") + \
        (f"@{job['variant']}" if job.get("variant") else "")


def arm_spec(job: dict) -> ArmSpec:
    try:
        return ARMS[job["arm"]]
    except KeyError:
        raise RunError(f"unknown arm {job.get('arm')!r} (known: {', '.join(ARMS)})") from None


# --------------------------------------------------------------------------------------------------------- lanes
@dataclass(frozen=True)
class Lane:
    """A fixed set of cores a run is pinned to. x925[0] = the solver, x925[1] = A1's second solver or the EA of A9 /
    A3x, a725 = the partner (EA or cuOpt; cuOpt's num_cpu_threads = PARTNER_THREADS = len(a725) on the GPU lane).
    whole_box: the lane holds every core (A6m only, and A6m only there; D-020)."""
    name: str
    x925: tuple[int, ...]
    a725: tuple[int, ...] = ()
    gpu: bool = False
    whole_box: bool = False

    def fits(self, arm: ArmSpec) -> bool:
        if (arm.lane == "full") != self.whole_box:
            return False
        return len(self.x925) >= arm.n_x925 and (not arm.needs_a725 or len(self.a725) >= 1) and (self.gpu or not arm.gpu)


GPU_LANE = Lane("g0", (5, 6), (1, 2, 3), gpu=True)
FULL_LANE = Lane("full", X925_CORES, A725_CORES, gpu=True, whole_box=True)   # the whole box, one run at a time (D-020)


# ------------------------------------------------------------------------------------------------------- options
@dataclass
class PairOptions:
    traces_dir: Path = paths.TRACES            # run directories: <traces_dir>/raw/<set>/<run_id>/
    mailbox: str = "file"                      # "file" (/dev/shm directory) | "redis" (itrea-redis, D-006)
    mailbox_root: Path = MAILBOX_ROOT
    redis_url: str = REDIS_URL
    use_systemd: bool = True                   # False = --no-systemd (tests): no scope, no MemoryMax
    pin: bool = True                           # taskset to the lane's cores
    lane: Lane = GPU_LANE
    solver_mem_gb: float = 9.0                 # MemoryMax per solver process (screening: 9 GiB, D-002 §1)
    ea_mem_gb: float = 6.0                     # MemoryMax of the EA process (host side only; GPU pool separate)
    cuopt_mem_gb: float = 12.0                 # MemoryMax of the cuOpt process (host side)
    gpu_mem_gb: float | None = None            # CuPy pool / RMM limit; None = the EA configuration's gpu_mem_gb
    cuopt_gpu_peak_gb: float | None = None     # cuOpt's measured RMM peak for the memory gate (None = its limit)
    cuopt_threads: int | None = None           # None = PARTNER_THREADS (the run id does not depend on the lane)
    grace_s: float = 30.0                      # after t0 + budget, before the runner writes `stop`
    ready_timeout_s: float = 900.0             # start-up (parse, CUDA context, warm-up) before the barrier
    kill_after_s: float = 60.0                 # after `stop`: then SIGTERM, 10 s later SIGKILL
    stop_partners_on_exit: bool = True         # a crash (exit != 0) or a proven outcome (exit_reason optimal /
                                               # infeasible in its summary) stops the others; any other early end
                                               # (a 0 exit at the process's own limit) leaves them running to the
                                               # budget, so a partner's early stop cannot truncate a control's run
    python: str = sys.executable
    cuopt_python: str | None = None            # another interpreter for cuOpt (D-001; same venv since 2026-10-05)
    cupy_cache_dir: Path = CUPY_CACHE_DIR
    cuda_cache_dir: Path = CUDA_CACHE_DIR      # harness cuda_cache: CUDA_CACHE_PATH of every process (D-021 R15)
    cap_floor_mhz: float | None = gpu_health.CAP_FLOOR_MHZ   # a GPU run with a busy SM median below this, or one
                                               # overlapping a governor "clock cap on" interval, is void (gpu_cap,
                                               # D-021 R25); None = off. The 2,000-MHz stuck-clock halt is separate
    thermal_watch_wait_s: float = 30.0         # after an external stop of a process: wait this long for thermal-watch's
                                               # 98 °C stop line (it is written after the units are stopped)
    gpu_health: bool = True
    gpu_period_s: float = 1.0
    tolerate_serving_gpu: bool = False         # the model servers' GPU processes are recorded, not foreign (pool sets it
                                               # from --allow-serving; never for test or screening jobs, D-010)
    shared_sampler: object | None = None       # a running gpu_health.Sampler owned by the pool (one for every lane);
                                               # None = this run starts and stops its own
    module_override: dict[str, list[str]] = field(default_factory=dict)  # kind -> argv prefix (tests: fake process)
    abort: threading.Event | None = None       # set by the pool's watchdog: stop now, the run is void
    abort_reason: Callable[[], str] | None = None
    extra_own_pids: Callable[[], set[int]] | None = None   # other processes of the same run set (not "foreign")
    window_id: str | None = None
    allow_test: bool = False                   # set by the pool after its freeze check (D-002 §1.8); a direct call
                                               # cannot run a test-split job
    redis_gb: float | None = None              # None = gates.REDIS_CAP_GB when mailbox == "redis"
    reserve_gb: float = gates.RESERVE_GB
    check_memory: bool = True                  # the memory gate before launch (the pool also checks)
    record_snapshots: bool = True              # gates.snapshot() before/after (systemctl, /proc scan)
    log: Callable[[str], None] = print
    # D-021 R12 (a harness with ref_stop): where the stop's reference comes from — a callable job -> (StopRef | None,
    # why); None = `miplib_stop_ref` (MIPLIB 2017 =opt= lines only; tests inject a toy's optimum) — how often the
    # runner reads the incumbent records, and whether the triggering incumbent is checked exactly after the run (off
    # only in tests that test something else)
    ref_lookup: Callable[[dict], tuple] | None = None
    ref_poll_s: float = 0.5
    ref_check: bool = True


# ----------------------------------------------------------------------------------------------- identity
def stem(path: str) -> str:
    n = os.path.basename(str(path))
    for ext in (".mps.gz", ".mps.bz2", ".mps", ".lp.gz", ".lp", ".gz"):
        if n.endswith(ext):
            return n[: -len(ext)]
    return n


def fmt_budget(b: float) -> str:
    return f"{float(b):g}"


def ablation_overrides(name: str | None, root: Path | None = None) -> dict:
    """Dotted overrides from configs/ablations/<name>.yaml (a partial configuration; `name` ignored)."""
    if not name:
        return {}
    p = (root or paths.CONFIGS) / "ablations" / f"{name}.yaml"
    if not p.exists():
        raise RunError(f"unknown ablation {name!r}: {p} does not exist")
    part = yaml.safe_load(p.read_text()) or {}
    sections = {f.name for f in dataclasses.fields(EAConfig) if dataclasses.is_dataclass(f.default_factory)}
    out = {}
    for k, v in part.items():
        if k == "name":
            continue
        if k in sections and isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
        else:
            out[k] = v
    return out


def effective_config(job: dict, opts: PairOptions | None = None) -> tuple[EAConfig, dict]:
    """The EA configuration a run uses: the job's config file (or the defaults), then the ablation file, the job's
    overrides, the arm's implied changes (x = 0, no exchange, backend) and the runner's GPU limit. Returns (config,
    all overrides applied)."""
    opts = opts or PairOptions()
    arm = arm_spec(job)
    base = EAConfig.from_yaml(paths.resolve(job["config"])) if job.get("config") else EAConfig().validate()
    ov: dict = {}
    ov.update(ablation_overrides(job.get("ablation")))
    ov.update(job.get("overrides") or {})
    ov.update(dict(arm.ea_overrides))
    for p in arm.procs:
        if p.kind == "ea" and p.backend:
            ov["backend"] = p.backend
    if opts.gpu_mem_gb is not None:
        ov["gpu_mem_gb"] = float(opts.gpu_mem_gb)
    cfg = base.with_overrides(ov) if ov else base
    return cfg, ov


def base_config_hash(job: dict) -> str:
    """Hash of the job's configuration file alone (what the freeze check compares with frozen-e2)."""
    base = EAConfig.from_yaml(paths.resolve(job["config"])) if job.get("config") else EAConfig().validate()
    return base.hash()


def resolve_harness(spec) -> dict | None:
    """The job-level harness (D-021) resolved to every key of HARNESS_KEYS — plus each switch of HARNESS_OPTIONAL that
    is on (an optional switch that is off is absent, so it leaves the dict and the run hash as they were) — or None
    when the job has none. `spec` is a preset name ("ext", "itx1") or a dict {"preset": name (default "ext"), <key>:
    value, ...} that overrides the preset's keys (M3: {"preset": "ext", "cuopt_rmm_gb": 40, "cuopt_host_gb": 40}; M1's
    env-only contrast: {"preset": "ext", "cuopt_env_unset": false}; {"preset": "ext", "ref_stop": true}). Unknown names
    or keys and wrongly typed values raise RunError. Sizes become floats, so 40 and 40.0 give one hash."""
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec not in HARNESS_PRESETS:
            raise RunError(f"unknown harness {spec!r} (presets: {', '.join(HARNESS_PRESETS)})")
        pre = HARNESS_PRESETS[spec]
        out = {k: pre[k] for k in HARNESS_KEYS}
        out.update({k: True for k in HARNESS_OPTIONAL if pre.get(k)})     # optional switches only when on
        return out
    if not isinstance(spec, dict):
        raise RunError(f"harness must be a preset name or a dict, not {spec!r}")
    d = dict(spec)
    out = resolve_harness(d.pop("preset", "ext"))
    unknown = sorted(set(d) - set(HARNESS_KEYS) - set(HARNESS_OPTIONAL))
    if unknown:
        raise RunError(f"unknown harness keys {unknown} (known: {', '.join(HARNESS_KEYS + HARNESS_OPTIONAL)})")
    for k, v in d.items():
        if k in _HARNESS_BOOL:
            if not isinstance(v, bool):
                raise RunError(f"harness {k} must be true or false, not {v!r}")
            if k in HARNESS_OPTIONAL and not v:
                out.pop(k, None)                  # off = absent: the same dict, and hash, as a harness without it
            else:
                out[k] = v
        else:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
                raise RunError(f"harness {k} must be a positive number of GB, not {v!r}")
            out[k] = float(v)
    return out


def job_harness(job: dict) -> dict | None:
    """The job's resolved harness (`resolve_harness`); refused for A6m, which carries D-020's own settings."""
    h = resolve_harness(job.get("harness"))
    if h is not None and arm_spec(job).lane == "full":
        raise RunError(f"arm {job['arm']} carries D-020's own settings; a harness is not allowed on it (D-021 R9)")
    return h


def cuopt_cores(spec: ProcSpec, arm: ArmSpec, lane: Lane) -> list[int]:
    """The cores of a cuOpt process: every core of the lane (A6m), the lane's cores that no other process of the arm
    holds (A6F: all of F; A2F: F minus SCIP's X925 core), or the lane's A725 cores (the partner footprint)."""
    if spec.whole_lane:
        return sorted(lane.x925 + lane.a725)
    if spec.f_share:
        return sorted(lane.x925[arm.n_x925:] + lane.a725)
    return list(lane.a725)


def cuopt_threads(spec: ProcSpec, opts: PairOptions, arm: ArmSpec | None = None) -> int:
    """num_cpu_threads of a cuOpt process: the arm's own (A6m: 24, whatever the lane), the number of its cores on the
    lane (A6F: |F|, A2F: |F| - 1; F = the cores of opts.lane, so F's size is part of these arms' identity) or the
    partner footprint's (`opts.cuopt_threads`, default PARTNER_THREADS). An option that contradicts a count the arm
    fixes is refused."""
    if spec.cpu_threads is not None:
        if opts.cuopt_threads not in (None, spec.cpu_threads):
            raise RunError(f"cuOpt's thread count is fixed by the arm ({spec.cpu_threads}); cuopt_threads "
                           f"{opts.cuopt_threads} contradicts it")
        return int(spec.cpu_threads)
    if spec.f_share:
        if arm is None:
            raise RunError("cuOpt on F: its thread count needs the arm")
        n = len(cuopt_cores(spec, arm, opts.lane))
        if opts.cuopt_threads not in (None, n):
            raise RunError(f"arm {arm.name}: cuOpt's thread count is its number of cores on lane {opts.lane.name} "
                           f"({n}); cuopt_threads {opts.cuopt_threads} contradicts it")
        return n
    return int(opts.cuopt_threads or PARTNER_THREADS)


def cuopt_rmm_gb(cfg: EAConfig, harness: dict | None) -> float:
    """cuOpt's RMM limit: the harness's own (separate from the EA's pool, D-021 R29) or the EA configuration's
    gpu_mem_gb (= the EA's CuPy pool, D-002 §3)."""
    return float(harness["cuopt_rmm_gb"]) if harness is not None else float(cfg.gpu_mem_gb)


def harness_footprint(arm: ArmSpec, opts: PairOptions) -> dict:
    """What a harness job's identity adds beyond the harness itself (D-021 R5 / R29): F's cores — the lane's sorted
    X925 and A725 cores — for an arm on the GPU lane (a re-laid F of the same size is another configuration), and the
    host caps of the arm's other processes (SCIP's `solver_mem_gb`, the EA's `ea_mem_gb`); cuOpt's caps are in the
    harness. Only jobs with a harness carry it, so no earlier run id moves."""
    d: dict = {"cores": (dict(x925=sorted(opts.lane.x925), a725=sorted(opts.lane.a725)) if arm.lane == "partner"
                         else None)}
    if any(p.kind in SOLVER_KINDS for p in arm.procs):
        d["solver_mem_gb"] = float(opts.solver_mem_gb)
    if arm.has_ea:
        d["ea_mem_gb"] = float(opts.ea_mem_gb)
    return d


def solver_settings(job: dict, opts: PairOptions, cfg: EAConfig) -> dict:
    """What identifies a run of an arm without an EA (its <cfg-hash>). Only A6m's dict carries `mip_gap_zero` and only
    a job with a harness carries `harness` and `footprint` (D-021), so every other arm's dict — and hash — is what it
    was before A6m existed (tests pin them). `gpu_mem_gb` is cuOpt's RMM limit (the harness's when there is one)."""
    arm = arm_spec(job)
    h = job_harness(job)
    d: dict = {"arm": arm.name, "emphasis": job.get("emphasis") if any(p.emphasis for p in arm.procs) else None}
    cu = [p for p in arm.procs if p.kind == "cuopt"]
    if cu:
        d.update(gpu_mem_gb=cuopt_rmm_gb(cfg, h) if h is not None else cfg.gpu_mem_gb,
                 cuopt_threads=cuopt_threads(cu[0], opts, arm))
        if cu[0].gap_zero:
            d["mip_gap_zero"] = True
    if h is not None:
        d["harness"] = h
        d["footprint"] = harness_footprint(arm, opts)
    return d


def run_hash(job: dict, opts: PairOptions | None = None) -> str:
    """<cfg-hash> of the run id: the effective EA configuration hash for arms with an EA (it equals the EA summary's
    config_hash); a hash of the solver-side settings otherwise. An arm with an EA AND cuOpt (A9) folds in cuOpt's
    thread count when it is not the default PARTNER_THREADS, so a thread-count variant gets its own run id (the default
    id stays the EA configuration hash; build_runs reads the EA hash from `run.json["config"]["ea_hash"]`). A job's
    harness (D-021) and its footprint (`harness_footprint`) are folded in the same way, only when the job has one."""
    opts = opts or PairOptions()
    cfg, _ = effective_config(job, opts)
    arm = arm_spec(job)
    if arm.has_ea:
        h = cfg.hash()
        extra: dict = {}
        n = opts.cuopt_threads or PARTNER_THREADS
        if any(p.kind == "cuopt" for p in arm.procs) and n != PARTNER_THREADS:
            extra["cuopt_threads"] = n
        hz = job_harness(job)
        if hz is not None:
            extra["harness"] = hz
            extra["footprint"] = harness_footprint(arm, opts)
        if extra:
            h = hashlib.sha256(json.dumps({"ea": h, **extra}, sort_keys=True).encode()).hexdigest()[:12]
        return h
    return hashlib.sha256(json.dumps(solver_settings(job, opts, cfg), sort_keys=True).encode()).hexdigest()[:12]


def run_id_for(job: dict, opts: PairOptions | None = None) -> str:
    """<instance-stem>__<arm>__s<seed>__<budget>s__<cfg-hash>."""
    name = job.get("name") or stem(job["instance"])
    return f"{name}__{arm_label(job)}__s{int(job['seed'])}__{fmt_budget(job['budget'])}s__{run_hash(job, opts)}"


def run_dir_for(job: dict, opts: PairOptions | None = None) -> Path:
    opts = opts or PairOptions()
    return Path(opts.traces_dir) / "raw" / job["set"] / run_id_for(job, opts)


def read_record(run_dir: str | os.PathLike) -> dict | None:
    try:
        return json.loads((Path(run_dir) / "run.json").read_text())
    except (OSError, ValueError):
        return None


def read_status(run_dir: str | os.PathLike) -> str | None:
    """`completed` | `failed` | `void` | `running` (runner died) | None (no record)."""
    rec = read_record(run_dir)
    return rec.get("status") if rec else None


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return repr(o)


# ---------------------------------------------------------------------------------------------- processes
@dataclass
class Proc:
    spec: ProcSpec
    argv: list[str]
    cmd: list[str]
    cores: list[int]
    mem_gb: float
    gpu_gb: float | None
    env_extra: dict
    log_path: Path
    env_unset: tuple[str, ...] = ()    # removed from the inherited environment before env_extra applies (A6m's cuOpt;
                                       # every cuOpt process under a harness with cuopt_env_unset, D-021 R14)
    scope_unit: str | None = None      # the systemd scope's unit name (itrea-run-<sha1(run)>-<role>-<epoch_ms>.scope)
    popen: subprocess.Popen | None = None
    launch_mono: float | None = None
    launch_epoch: float | None = None
    ready_mono: float | None = None
    exit_code: int | None = None
    exit_mono: float | None = None
    rusage: object | None = None
    killed: str | None = None
    stopped_by_runner: bool = False    # killed after a partner ended the race (proven outcome): not a crash

    @property
    def role(self) -> str:
        return self.spec.role

    @property
    def pid(self) -> int | None:
        return self.popen.pid if self.popen else None

    def poll(self) -> int | None:
        """Reap with wait4 so the record gets the child's CPU seconds and peak RSS."""
        if self.exit_code is not None or self.popen is None:
            return self.exit_code
        try:
            pid, status, ru = os.wait4(self.popen.pid, os.WNOHANG)
        except ChildProcessError:
            rc = self.popen.returncode
            self.exit_code = rc if rc is not None else -999
            self.exit_mono = time.monotonic()
            return self.exit_code
        if pid == 0:
            return None
        self.exit_code = os.waitstatus_to_exitcode(status)
        self.exit_mono = time.monotonic()
        self.rusage = ru
        self.popen.returncode = self.exit_code
        return self.exit_code

    def signal(self, sig: int) -> None:
        if self.popen is None or self.poll() is not None:
            return
        try:
            os.killpg(self.popen.pid, sig)        # the process group (start_new_session): helpers die too
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(self.popen.pid, sig)
            except ProcessLookupError:
                pass


def _fmt_set(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float, str)):
        return str(v)
    return json.dumps(v)


def cuopt_flags(spec: ProcSpec, cfg: EAConfig | None) -> list[str]:
    """`--send` for cuOpt only in an exchange arm whose EA reads cuOpt (A9: exchange.enabled and from_cuopt); the
    controls A2 / A6 never send (cuOpt never receives in any arm)."""
    if spec.exchange and cfg is not None and cfg.exchange.enabled and cfg.exchange.from_cuopt:
        return ["--send"]
    return []


def solver_flags(spec: ProcSpec, cfg: EAConfig | None) -> list[str]:
    """--recv/--send/--inert for a solver process. Exchange arms follow the EA configuration (an ablation that turns a
    direction off turns the matching solver flag off); every other solver arm carries the inert plugin (D-006:
    every SCIP arm carries the same plugin, harness-matched A0)."""
    if not spec.exchange or cfg is None or not cfg.exchange.enabled:
        return ["--inert"]
    e = cfg.exchange
    recv = bool(e.to_solver)
    ls = cfg.lp_stream
    send = bool(e.from_solver or e.use_dual_bound or e.use_root_lp or ls.enabled)
    out = []
    if recv:
        out.append("--recv")
    if send:
        out += ["--send", "--inject-band", _fmt_set(e.inject_band), "--inject-k", str(e.inject_k)]
    if ls.enabled:                                   # D-011: node LP points are read through SCIP's plugin only
        if spec.kind != "scip":
            raise RunError(f"lp_stream.enabled needs a SCIP partner, not {spec.kind}")
        out += ["--lp-stream", "--lp-stream-share", _fmt_set(ls.share), "--lp-stream-ring", str(ls.ring),
                "--lp-stream-min-gap", _fmt_set(ls.min_gap), "--lp-stream-q", _fmt_set(ls.quantile),
                "--lp-stream-patience", _fmt_set(ls.patience)]
    return out or ["--inert"]


def scope_unit(run_name: str, role: str, epoch_ms: int) -> str:
    """The run scope's unit name `itrea-run-<sha1(run directory name)[:12]>-<role>-<epoch_ms>` (D-021 R27: host tools
    such as thermal-watch find the project's processes by the prefix; never part of a hash or of argv). The run
    directory's name (= the run id: instance, arm, seed, budget, configuration) and the plan's millisecond make it
    unique per launch, so concurrent lanes never ask systemd for one name twice."""
    h = hashlib.sha1(str(run_name).encode()).hexdigest()[:12]
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"itrea-run-{h}-{role}-{int(epoch_ms)}")[:200]


def plan(job: dict, opts: PairOptions, run_dir: Path, mailbox_spec: str, cfg: EAConfig,
         epoch_ms: int | None = None) -> list[Proc]:
    """The processes of a run (nothing is launched). `epoch_ms` names the scopes (default: now, in milliseconds)."""
    arm = arm_spec(job)
    lane = opts.lane
    hz = job_harness(job)
    epoch_ms = int(time.time() * 1000) if epoch_ms is None else int(epoch_ms)
    if hz is not None and opts.cuopt_gpu_peak_gb is not None and any(p.kind == "cuopt" for p in arm.procs):
        raise RunError("a job with a harness is budgeted on its caps (D-021 R29): a measured cuOpt peak "
                       "(--cuopt-gpu-peak-gb) is refused")
    if arm.lane == "full" and not lane.whole_box:
        raise RunError(f"arm {arm.name} runs only on the whole-box lane (pool --layout full, D-020), not on lane "
                       f"{lane.name}")
    if lane.whole_box and (opts.mailbox == "redis" or opts.cuopt_gpu_peak_gb is not None):
        raise RunError("the whole-box lane refuses the Redis mailbox (core 0 is inside the lane) and a measured cuOpt "
                       "peak for the memory gate (D-020's 81 GB counts the RMM limit)")
    if lane.whole_box and arm.lane != "full":
        full = ", ".join(a for a, sp in ARMS.items() if sp.lane == "full")
        raise RunError(f"the whole-box lane {lane.name} runs only {full} (D-020; a full-box A9 needs its own arm and "
                       f"decision), not {arm.name}")
    if not lane.fits(arm):
        raise RunError(f"lane {lane.name} (x925={lane.x925}, a725={lane.a725}, gpu={lane.gpu}) cannot run {arm.name}")
    label = arm_label(job)
    seed0 = int(job["seed"])
    gpu_gb = float(cfg.gpu_mem_gb)
    procs = []
    n_x = 0                                          # X925 cores handed out: solvers first, then A9's EA
    for spec in arm.procs:
        if spec.kind in opts.module_override:
            argv = list(opts.module_override[spec.kind])
        else:
            py = opts.cuopt_python if (spec.kind == "cuopt" and opts.cuopt_python) else opts.python
            argv = [py, "-m", ROLE_MODULES[spec.kind]]
        argv += ["--instance", str(paths.resolve(job["instance"])), "--seed", str(seed0 + spec.seed_offset),
                 "--budget", fmt_budget(job["budget"]), "--run-dir", str(run_dir), "--mailbox", mailbox_spec,
                 "--arm", label]
        if spec.kind == "ea":
            argv += ["--sol-out", str(run_dir / f"{spec.role}.sol.json.gz")]
        else:      # solver processes write <role>.sol.json.gz themselves and check the mailbox's run id
            argv += ["--run-id", run_dir.name]
        if spec.kind in SOLVER_KINDS:
            cores = [lane.x925[n_x]]
            n_x += 1
            mem = opts.solver_mem_gb
            argv += ["--mem-mb", str(int(max(512, mem * 1024 - 1024)))]   # the solver's own limit below the cap
            argv += solver_flags(spec, cfg if arm.has_ea else None)
            if spec.role != "solver":
                argv += ["--role-name", spec.role]
            if spec.emphasis:
                if not job.get("emphasis"):
                    raise RunError(f"arm {arm.name} needs the job's 'emphasis'")
                argv += ["--emphasis", str(job["emphasis"])]
            g = None
        elif spec.kind == "ea":
            if spec.x925:                            # A9: a core of its own; cuOpt keeps the A725 cores
                if cfg.repair.continuous == "lp_decode":
                    # D-017: the LP decode worker would share the EA's one core (with the LP worker of
                    # seeding.lp_source highs_async) and a pending batch more takes the candidate set's concatenate
                    # to CuPy's 8-part limit; refused until a decision allows it
                    raise RunError(f"arm {arm.name} runs the EA on one X925 core: repair.continuous 'lp_decode' is "
                                   "refused there (D-017) until a decision allows it")
                cores = [lane.x925[n_x]]
                n_x += 1
            else:
                cores = list(lane.a725)
            mem = opts.ea_mem_gb
            argv += ["--config", str(run_dir / "config.yaml")]
            g = gpu_gb if spec.gpu else None
        elif spec.kind == "cuopt":
            cores = cuopt_cores(spec, arm, lane)
            if hz is not None:                       # D-021 R29: cuOpt's own caps, separate from the EA's pool
                mem, g = float(hz["cuopt_host_gb"]), cuopt_rmm_gb(cfg, hz)
            else:
                mem, g = opts.cuopt_mem_gb, gpu_gb       # RMM limit = the EA's pool limit (D-002 §3)
            argv += ["--cpu-threads", str(cuopt_threads(spec, opts, arm)), "--gpu-mem-gb", _fmt_set(g)]
            if spec.gap_zero or (hz is not None and hz["cuopt_gap_zero"]):
                argv.append("--mip-gap-zero")
            argv += cuopt_flags(spec, cfg if arm.has_ea else None)
        else:
            raise RunError(f"unknown process kind {spec.kind!r}")
        env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
               "ITREA_RUN_ID": run_dir.name, "ITREA_ROLE": spec.role, "CUPY_CACHE_DIR": str(opts.cupy_cache_dir)}
        unset: tuple[str, ...] = ()
        # A6m: the thread variables unset, as GAMS ran cuOpt (D-020); under a harness with cuopt_env_unset every cuOpt
        # process (D-021 R14), confined by taskset
        if spec.whole_lane or (spec.kind == "cuopt" and hz is not None and hz["cuopt_env_unset"]):
            unset = THREAD_VARS
            for k in unset:
                env.pop(k)
        if g is not None:
            env["CUPY_GPU_MEMORY_LIMIT"] = str(int(g * 1024**3))   # CuPy reads it at import (EA); cuOpt: RMM limit
            env["ITREA_GPU_MEM_GB"] = _fmt_set(g)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""      # a CPU process must not create a CUDA context (A4 = NumPy EA)
        if hz is not None and hz["cuda_cache"]:   # D-021 R15: the project's own CUDA JIT cache, never evicting
            env["CUDA_CACHE_PATH"] = str(opts.cuda_cache_dir)
            env["CUDA_CACHE_MAXSIZE"] = str(CUDA_CACHE_MAXSIZE)
        cmd = list(argv)
        if opts.pin and cores:
            cmd = ["taskset", "-c", ",".join(map(str, cores))] + cmd
        unit = None
        if opts.use_systemd:
            # D-021 R27: a named scope; a stop (thermal-watch's `systemctl --user stop`) kills within 5 s instead of
            # systemd's 90 s, and cuOpt — which has no interrupt API and ignores SIGTERM while it solves — is stopped
            # with SIGKILL, as the runner's own abort path does
            unit = scope_unit(run_dir.name, spec.role, epoch_ms)
            props = ["-p", f"TimeoutStopSec={SCOPE_STOP_TIMEOUT}"] + \
                (["-p", "KillSignal=SIGKILL"] if spec.kind == "cuopt" else [])
            cmd = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}",
                   f"--description=itrea {run_dir.name} {spec.role}",
                   "-p", f"MemoryMax={int(round(mem * 1024))}M", "-p", "MemorySwapMax=0"] + props + cmd
        procs.append(Proc(spec=spec, argv=argv, cmd=cmd, cores=cores, mem_gb=mem, gpu_gb=g, env_extra=env,
                          log_path=run_dir / f"{spec.role}.log", env_unset=unset, scope_unit=unit))
    return procs


def budget_for(procs: list[Proc], opts: PairOptions) -> gates.Budget:
    """D-003's memory formula for one run."""
    host = sum(p.mem_gb for p in procs)
    gpu = 0.0
    for p in procs:
        if p.gpu_gb is None:
            continue
        gpu += (opts.cuopt_gpu_peak_gb if (p.spec.kind == "cuopt" and opts.cuopt_gpu_peak_gb) else p.gpu_gb)
    redis = (opts.redis_gb if opts.redis_gb is not None else gates.REDIS_CAP_GB) if opts.mailbox == "redis" else 0.0
    return gates.Budget(host_caps_gb=host, gpu_caps_gb=gpu, reserve_gb=opts.reserve_gb,
                        cuda_procs=sum(1 for p in procs if p.gpu_gb is not None), redis_gb=redis)


def job_budget(job: dict, opts: PairOptions | None = None) -> gates.Budget:
    """The memory budget of a job without launching it (the pool's gate)."""
    opts = opts or PairOptions()
    cfg, _ = effective_config(job, opts)
    return budget_for(plan(job, opts, Path("/nonexistent") / run_id_for(job, opts), "-", cfg), opts)


# ----------------------------------------------------------------------------------------------- mailbox
def _mailbox_spec(run_id: str, opts: PairOptions) -> str:
    if opts.mailbox == "redis":
        return f"{opts.redis_url}?run={quote(run_id, safe='')}"
    if opts.mailbox == "file":
        return str(Path(opts.mailbox_root) / run_id)
    raise RunError(f"unknown mailbox kind {opts.mailbox!r}")


def _create_mailbox(run_id: str, opts: PairOptions):
    from itrea.exchange.redis_mailbox import open_mailbox
    spec = _mailbox_spec(run_id, opts)
    if opts.mailbox == "file":
        if Path(spec).exists():
            raise StaleMailboxError(f"mailbox {spec} exists (stale run or a run still in flight); refusing")
        Path(opts.mailbox_root).mkdir(parents=True, exist_ok=True)
        try:
            return spec, open_mailbox(spec, "runner", run_id=run_id, create=True)
        except FileExistsError:
            raise StaleMailboxError(f"mailbox {spec} exists; refusing") from None
    try:
        return spec, open_mailbox(spec, "runner", run_id=run_id, create=True)
    except RuntimeError as e:
        raise StaleMailboxError(str(e)) from None


def _remove_mailbox(mb, spec: str, opts: PairOptions) -> None:
    if opts.mailbox == "redis":
        try:
            mb.cleanup()
        except Exception as e:  # noqa: BLE001
            opts.log(f"redis cleanup failed for {spec}: {e!r}")
        return
    p = Path(spec)
    # only the run's own directory under the mailbox root, created by this runner
    if p.parent == Path(opts.mailbox_root) and p.is_dir():
        shutil.rmtree(p, ignore_errors=True)


def _exchange_log(mb, opts: PairOptions, run_dir: Path, cuopt: bool = False) -> dict:
    """Copy what the mailbox knows about the exchange into the run directory before the mailbox is removed. `cuopt`:
    cuOpt posted to the EA (A9), so its stream `to_ea_cuopt` is recorded too."""
    out: dict = {}
    try:
        verdicts = mb.verdicts()
    except Exception as e:  # noqa: BLE001
        verdicts = []
        out["verdicts_error"] = repr(e)
    if verdicts:
        with open(run_dir / "verdicts.jsonl", "w") as f:
            for v in verdicts:
                f.write(json.dumps(v) + "\n")
    out["verdicts"] = len(verdicts)
    out["accepted"] = sum(1 for v in verdicts if v.get("accepted"))
    if opts.mailbox == "redis":
        hist = {}
        for name in ("to_solver", "to_ea", "lp", "lps") + (("to_ea_cuopt",) if cuopt else ()):
            try:
                hist[name] = [[t, seq, [float(o) for o in objs]] for t, seq, objs in getattr(mb, name).history()]
            except Exception:  # noqa: BLE001
                hist[name] = None
        out["streams"] = {k: (len(v) if v is not None else None) for k, v in hist.items()}
        (run_dir / "exchange.json").write_text(json.dumps(hist))
    return out


def clear_stale(job: dict, opts: PairOptions, wait_s: float | None = None) -> dict:
    """Make a run launchable again after an interrupted attempt: stop processes of the old attempt that are still
    alive (only those whose command line carries this run id; their whole process group), then remove the stale
    mailbox. Returns what was done. Raises RunError if a process of the old attempt survives SIGKILL (the pool must
    not launch beside it). It reads the pids from <run_dir>/run.json, so call it BEFORE the run directory is moved
    aside (the pool moves it: it is evidence)."""
    run_id = run_id_for(job, opts)
    spec = _mailbox_spec(run_id, opts)
    done: dict = {"run_id": run_id, "killed": [], "mailbox_removed": False}
    rec = read_record(run_dir_for(job, opts)) or {}
    alive = []
    for pr in rec.get("processes", []):
        pid = pr.get("pid")
        if pid and _pid_runs(pid, run_id):
            alive.append(pid)
    if alive:
        try:
            from itrea.exchange.redis_mailbox import open_mailbox
            open_mailbox(spec, "runner", run_id=run_id).stop("runner: stale attempt")
        except Exception:  # noqa: BLE001
            pass
        end = time.monotonic() + (opts.kill_after_s if wait_s is None else wait_s)
        while time.monotonic() < end and any(_pid_runs(p, run_id) for p in alive):
            time.sleep(0.5)
        for p in alive:
            if not _pid_runs(p, run_id):
                continue
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if not _pid_runs(p, run_id):          # the cmdline check guards the group kill against pid reuse
                    break
                try:
                    os.killpg(p, sig)                 # pid == pgid (start_new_session): helpers die too
                except (ProcessLookupError, PermissionError):
                    try:
                        os.kill(p, sig)
                    except ProcessLookupError:
                        break
                t_end = time.monotonic() + 2.0
                while time.monotonic() < t_end and _pid_runs(p, run_id):
                    time.sleep(0.1)
            done["killed"].append(p)
        survivors = [p for p in alive if _pid_runs(p, run_id)]
        if survivors:
            raise RunError(f"processes {survivors} of the stale attempt {run_id} survive SIGKILL; refusing to launch "
                           f"beside them")
    if opts.mailbox == "file":
        p = Path(spec)
        if p.parent == Path(opts.mailbox_root) and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            done["mailbox_removed"] = True
    else:
        from itrea.experiments import exchange_ctl
        try:
            done["mailbox_removed"] = exchange_ctl.flush(run_id) > 0
        except Exception as e:  # noqa: BLE001
            done["mailbox_error"] = repr(e)
    return done


def _pid_runs(pid: int, run_id: str) -> bool:
    """The pid is alive AND its command line names this run (guards against pid reuse)."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return run_id in cmdline


# --------------------------------------------------------------------------------------------- trajectories
def merge_trajectories(run_dir: Path, roles: tuple[str, ...], sense: int, solver_role: str | None = "solver") -> dict:
    """Merged incumbent trajectories from `<role>.events.jsonl` (objectives as REPORTED by the processes; the analysis
    rebuilds them from checked incumbents). portfolio = best of all processes at every moment: [t, obj, role, source];
    solver = what the solver itself held (including solutions it accepted from the EA)."""
    per: dict[str, list] = {}
    for r in roles:
        p = run_dir / f"{r}.events.jsonl"
        evs = read_events(p) if p.exists() else []
        per[r] = [(float(e["t"]), float(e["obj"]), r, e.get("source", "self")) for e in evs
                  if e.get("kind") == "incumbent" and e.get("obj") is not None and e.get("t") is not None]

    def improving(items):
        out, best = [], None
        for t, obj, role, src in sorted(items, key=lambda x: (x[0], sense * x[1])):
            if best is None or sense * obj < sense * best:
                best = obj
                out.append([round(t, 6), obj, role, src])
        return out

    allv = [x for r in roles for x in per[r]]
    return dict(sense=sense,
                portfolio=improving(allv),
                solver=improving(per[solver_role]) if solver_role in per else None,
                first_incumbent={r: (min(x[0] for x in per[r]) if per[r] else None) for r in roles},
                incumbent_events={r: len(per[r]) for r in roles})


def _count_files(d: Path) -> int | None:
    if not d.is_dir():
        return 0
    n = 0
    for _, _, files in os.walk(d):
        n += len(files)
    return n


# ------------------------------------------------------------------------------- R12: stop at the proven optimum
REF_STOP_GAP = 1e-9                   # D-021 R12: Berthold's gap at the stop, beside GAMS's rounded gamma <= 0
_PART_HDR = struct.Struct("<ddBi")    # a .part record's header (solvers.common / ea.engine): t, obj, source, nnz


@dataclass(frozen=True)
class StopRef:
    """R12's reference: z* in the user's sense of an instance MIPLIB lists as proven optimal. Read by the runner
    before the clock starts; no process ever receives it."""
    value: float
    kind: str
    source: str


def miplib_stop_ref(job: dict, solu_path: str | os.PathLike | None = None) -> tuple[StopRef | None, str]:
    """R12's eligibility and reference: (StopRef, "") for a MIPLIB 2017 instance (a path under data/miplib) whose line
    in MIPLIB's solu file is `=opt=`; (None, why) otherwise — `=best=` (supportcase22: never), no line, or any other
    dataset (PGLib-UC, ML4CO: never)."""
    from itrea.benchmarks import references as R
    inst = str(job["instance"])
    if not R.is_miplib(None, inst):
        return None, f"{inst} is not a MIPLIB 2017 instance (R12 applies to MIPLIB =opt= references only)"
    name = stem(inst)
    try:
        solu = R.read_solu(solu_path)
    except (OSError, ValueError) as e:
        return None, f"no MIPLIB solu file ({e})"
    e = solu.get(name)
    if e is None:
        return None, f"{name} has no line in MIPLIB's solu file"
    if e.kind != "opt" or not e.has_value:
        return None, f"{name} is {e.kind} in MIPLIB's solu file (R12 needs a proven optimum, =opt=)"
    return StopRef(float(e.value), "opt", "MIPLIB 2017 solu (=opt=)"), ""


def ref_reached(obj: float, z: float, sense: int = 1, gap_tol: float = REF_STOP_GAP) -> tuple[bool, float, float]:
    """R12's test of one objective against z*: GAMS's gamma, round((obj − z*)/max(|obj|, |z*|, 1), 6) in the objective's
    sense (`metrics.gams_gap`), is <= 0 AND Berthold's gap (`metrics.primal_gap`) is <= gap_tol — the second guards
    z* near 0, where GAMS's denominator of 1 alone would stop at an absolute difference of 5e-7. An objective better
    than z* passes both (gamma 0, gap 0). Returns (reached, gamma, gap)."""
    from itrea.analysis import metrics as M
    g, _ = M.gams_gap(obj, z, sense)
    b, _ = M.primal_gap(obj, z, sense)
    return bool(g <= 0.0 and b <= gap_tol), float(g), float(b)


class RefMonitor:
    """D-021 R12 inside the runner. While the clock runs it tails every process's crash-safe incumbent records
    (`<role>.incumbents.part`, the format of `solvers.common.SolverProcess.record_incumbent` and `ea.engine`), maps
    each vector by `<role>.names.json` onto the runner's variable order, recomputes the objective c·x + offset from the
    stored vector (c and offset were read before t0, from the runner's own parse) and reports the first record stamped
    before `t_max` (the budget) that meets `ref_reached` — first by its time stamp across ALL roles, not by the order
    the roles are read in: each poll reads every role, keeps each role's first qualifying record (its candidate; within
    a role the file order is time order) and, while no trigger is set, takes the candidate with the smallest t (a tie
    goes to the earlier role of `roles`). In A9 cuOpt's record of z*, the EA's re-record of the same vector and SCIP's
    re-record of the EA's post usually land in one poll; the trigger is cuOpt's, the earliest (as M0's replay judges).
    Once set the trigger never changes; `summary()` lists every role's candidate (`candidates`, by t; `after_trigger`:
    read only after the trigger was chosen, e.g. by `drain()` at the end). Each .part file is opened once and the
    descriptor kept: the process unlinks the file after writing its .npz, and an open descriptor still reads every
    byte. A role whose names the runner cannot map, or whose records do not parse, is left out with the reason
    (`summary`) — never a crash of the run."""

    def __init__(self, run_dir: Path, roles, names, c, offset: float, sense: int, ref: StopRef, t_max: float,
                 gap_tol: float = REF_STOP_GAP):
        self.run_dir = Path(run_dir)
        self.index = {str(n): j for j, n in enumerate(names)}
        self.c = np.asarray(c, dtype=np.float64)
        self.offset, self.sense, self.ref, self.t_max, self.gap_tol = float(offset), int(sense), ref, float(t_max), \
            float(gap_tol)
        self.st = {r: dict(fh=None, buf=b"", k=0, cl=None, error=None, last_t=None, first=None) for r in roles}
        self.trigger: dict | None = None
        self.records = 0

    def _read_all(self) -> None:
        for role, st in self.st.items():
            if st["error"] is None:
                try:
                    self._read(role, st)
                except Exception as e:  # noqa: BLE001 — a fault of this reader leaves the role out, nothing else
                    st["error"] = f"{type(e).__name__}: {e}"[:300]

    def poll(self) -> dict | None:
        """Read what the processes appended since the last call; the trigger (once found, always the same: the
        earliest candidate by t over every role read so far, ties by role order) or None."""
        self._read_all()
        if self.trigger is None:
            cands = [(st["first"]["t"], i, st["first"]) for i, st in enumerate(self.st.values())
                     if st["first"] is not None]
            if cands:
                self.trigger = dict(min(cands, key=lambda c: (c[0], c[1]))[2])
        return self.trigger

    def drain(self) -> None:
        """Read what is left after the stop (records written but not yet read when the trigger was chosen) into the
        candidates; the trigger never changes and none is set here (a run that ended otherwise keeps none)."""
        self._read_all()

    def _read(self, role: str, st: dict) -> None:
        if st["cl"] is None:
            p = self.run_dir / f"{role}.names.json"
            if not p.exists():
                return                                    # not recording yet (the EA starts its log at t0)
            try:
                local = json.loads(p.read_text())
            except ValueError:
                # names.json is written (not atomically) before the .part file is opened (solvers.common, ea.engine):
                # a half-written file is read again later, an error only once the .part file exists
                if not (self.run_dir / f"{role}.incumbents.part").exists():
                    return
                local = json.loads(p.read_text())
            idx = [self.index.get(str(n)) for n in local]
            miss = sum(1 for i in idx if i is None)
            if miss:
                raise ValueError(f"{miss} of {len(local)} variable names of {role} are unknown to the runner")
            st["cl"] = self.c[np.asarray(idx, dtype=np.int64)] if idx else np.zeros(0)
        if st["fh"] is None:
            try:
                st["fh"] = open(self.run_dir / f"{role}.incumbents.part", "rb")
            except FileNotFoundError:
                return
        data = st["fh"].read()
        if not data:
            return
        buf, off, H, cl = st["buf"] + data, 0, _PART_HDR.size, st["cl"]
        while len(buf) - off >= H:
            t, obj, src, nnz = _PART_HDR.unpack_from(buf, off)
            if nnz < 0 or nnz > cl.size:
                raise ValueError(f"record {st['k']} of {role}: {nnz} non-zeros for {cl.size} variables")
            end = off + H + 12 * nnz
            if end > len(buf):
                break                                     # the rest of this record is still being written
            ix = np.frombuffer(buf, dtype="<i4", count=nnz, offset=off + H)
            val = np.frombuffer(buf, dtype="<f8", count=nnz, offset=off + H + 4 * nnz)
            if nnz and (int(ix.min()) < 0 or int(ix.max()) >= cl.size):
                raise ValueError(f"record {st['k']} of {role}: an index outside the variables")
            o = float(np.dot(cl[ix], val)) + self.offset
            k = st["k"]
            st["k"] += 1
            st["last_t"] = t
            self.records += 1
            if st["first"] is None and t < self.t_max and math.isfinite(o):
                ok, g, b = ref_reached(o, self.ref.value, self.sense, self.gap_tol)
                if ok:
                    st["first"] = dict(role=role, k=k, t=t, obj=o, obj_reported=obj, source=int(src), gamma=g, gap=b,
                                       after_trigger=self.trigger is not None)
            off = end
        st["buf"] = buf[off:]

    def close(self) -> None:
        for st in self.st.values():
            if st["fh"] is not None:
                try:
                    st["fh"].close()
                except OSError:
                    pass
                st["fh"] = None

    def summary(self) -> dict:
        cands = sorted((st["first"] for st in self.st.values() if st["first"] is not None),
                       key=lambda c: (c["t"], list(self.st).index(c["role"])))
        return dict(records=self.records, trigger=self.trigger, candidates=cands,
                    roles={r: dict(records=st["k"], last_t=st["last_t"], error=st["error"]) for r, st in self.st.items()})


def check_ref_trigger(run_dir: Path, trig: dict, instance: str, ref: StopRef, sense: int, arm: str,
                      gap_tol: float = REF_STOP_GAP) -> dict:
    """R12's confirmation, after the processes ended: the incumbent that triggered the stop is taken from its role's
    `<role>.incumbents.npz` (recovered from the .part file for a killed process) — record k, or the record with the
    same time and reported objective —, checked by the independent exact checker (`check_solutions`, the other
    library's parse) and its EXACT objective must meet `ref_reached` too. Returns {ok, reason, ...}; ok False makes the
    run void (`ref_rejected`: rerun once with the stop off). A model the checker cannot read is not a confirmation."""
    from itrea.analysis import check_solutions as CS
    from itrea.trace import load_incumbents
    role, k = trig["role"], int(trig["k"])
    out: dict = dict(role=role, k=k, ok=False)
    try:
        names, recs = load_incumbents(run_dir / f"{role}.incumbents.npz")
    except (OSError, ValueError, KeyError) as e:
        return dict(out, reason=f"no stored incumbents for {role}: {type(e).__name__}: {e}"[:300])
    same = lambda r: r[0] == trig["t"] and r[1] == trig["obj_reported"]      # noqa: E731
    if not (k < len(recs) and same(recs[k])):
        hits = [i for i, r in enumerate(recs) if same(r)]
        if not hits:
            return dict(out, reason=f"the triggering record ({role} #{k}, t {trig['t']:.6f}) is not among the "
                                    f"{len(recs)} stored incumbents")
        k = out["k_stored"] = hits[0]
    t, obj, src, x = recs[k]
    try:
        prod = CS.producer_of(role, _summary(run_dir, role), arm)
        model = CS.ModelCache(1).get(instance, CS.reader_for(prod))
        r = CS.check_named(model, names, x, obj)
    except Exception as e:  # noqa: BLE001 — unchecked is not confirmed
        return dict(out, reason=f"unchecked: {type(e).__name__}: {e}"[:300])
    oc = float(r["obj_recomputed"])
    reached, g, b = ref_reached(oc, ref.value, sense, gap_tol) if math.isfinite(oc) else (False, math.nan, math.nan)
    ok = bool(r["ok"] and reached)
    why = "" if ok else (f"the checker rejects it ({r['reason']})" if not r["ok"] else
                         f"its exact objective {oc!r} does not meet the stop (gamma {g:g}, gap {b:.3g})")
    return dict(out, ok=ok, reason=why, producer=prod, reader=model.reader, feasible=bool(r["feasible"]),
                obj_recomputed=oc, gamma_exact=g, gap_exact=b, checker_version=CS.CHECKER_VERSION)


# --------------------------------------------------------------------------------------------------- run
def run_pair(job: dict, opts: PairOptions | None = None) -> dict:
    """Run one job; returns the run record (also written to <run_dir>/run.json). Raises RunError (nothing launched)
    for a bad job or stale state; every other outcome is a record with status completed | failed | void."""
    opts = opts or PairOptions()
    for k in ("instance", "set", "arm", "seed", "budget"):
        if k not in job:
            raise RunError(f"job lacks {k!r}: {job}")
    arm = arm_spec(job)
    if job.get("split") == "test" and not opts.allow_test:
        raise RunError("test-split jobs run only through the pool (freeze check: frozen-e2 hash, prereg-v1)")
    cfg, overrides = effective_config(job, opts)
    harness = job_harness(job)
    run_id = run_id_for(job, opts)
    run_dir = Path(opts.traces_dir) / "raw" / job["set"] / run_id
    if run_dir.exists():
        raise StaleRunError(f"run directory {run_dir} exists (status {read_status(run_dir)!r}); move it aside first")
    spec = _mailbox_spec(run_id, opts)
    if opts.mailbox == "file" and Path(spec).exists():
        raise StaleMailboxError(f"mailbox {spec} exists (stale run or a run still in flight); refusing")

    from itrea.problem import load
    t_load = time.monotonic()
    prob = load(job["instance"])
    load_s = time.monotonic() - t_load
    # keep only what the record needs: the parsed arrays would otherwise sit in the (unbudgeted) runner for the
    # whole run, once per lane
    names, sense, prob_summary = list(prob.names), prob.sense, prob.summary()
    # D-021 R12: the stop's reference and the objective (c, offset) are read here, before the clock; no process
    # receives either. `ref_stop_off` (set by the pool on the rerun of a block voided by `ref_rejected`) is a runtime
    # switch outside the run id.
    ref_rec, stop_ref, obj_c, obj_off = None, None, None, 0.0
    if harness is not None and harness.get("ref_stop"):
        if job.get("ref_stop_off"):
            ref_rec = dict(enabled=False, reason=f"off for this attempt: {job['ref_stop_off']}")
        else:
            stop_ref, why = (opts.ref_lookup or miplib_stop_ref)(job)
            if stop_ref is None:
                ref_rec = dict(enabled=False, reason=f"not eligible: {why}")
            else:
                ref_rec = dict(enabled=True, reason="", z=stop_ref.value, kind=stop_ref.kind, source=stop_ref.source,
                               sense=sense, gap_tol=REF_STOP_GAP, poll_s=opts.ref_poll_s)
                obj_c, obj_off = np.array(prob.c, dtype=np.float64), float(prob.offset)
    del prob

    procs = plan(job, opts, run_dir, spec, cfg)
    budget = budget_for(procs, opts)
    if opts.check_memory:
        gates.check_memory(budget)

    spec, mb = _create_mailbox(run_id, opts)         # refuses a stale mailbox (file directory or Redis keys)
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        _remove_mailbox(mb, spec, opts)
        raise StaleRunError(f"run directory {run_dir} appeared concurrently") from None
    mb.write_order(names)
    (run_dir / "order.json").write_text(json.dumps(names))
    if arm.has_ea:
        d = cfg.to_dict()
        (run_dir / "config.yaml").write_text(yaml.safe_dump(d, sort_keys=False))

    rec: dict = dict(
        run_id=run_id, status="running", reasons=[], arm=arm_label(job), base_arm=arm.name,
        ablation=job.get("ablation"), arm_doc=arm.doc, gpu_arm=arm.gpu, lane_kind=arm.lane,
        job=job, set=job["set"], instance=job["instance"], name=job.get("name") or stem(job["instance"]),
        seed=int(job["seed"]), budget=float(job["budget"]), grace_s=opts.grace_s,
        block=job.get("block"), position=job.get("position"), window=opts.window_id or job.get("window"),
        stratum=job.get("stratum"), split=job.get("split"),
        config=dict(path=job.get("config"), overrides=overrides, base_hash=base_config_hash(job),
                    ea_hash=cfg.hash() if arm.has_ea else None, run_hash=run_hash(job, opts),
                    solver_settings=None if arm.has_ea else solver_settings(job, opts, cfg),
                    harness=harness),
        problem=dict(prob_summary, load_s=round(load_s, 3)),
        mailbox=dict(kind=opts.mailbox, spec=spec),
        lane=dataclasses.asdict(opts.lane), systemd=opts.use_systemd, pinned=opts.pin,
        runner_cores=sorted(os.sched_getaffinity(0)),   # the runner's own threads; on the whole-box lane they share
                                                        # the run's cores (RUNNER_CORES)
        memory_budget=dict(dataclasses.asdict(budget), need_gb=round(budget.need_gb, 2)),
        processes=[], created_epoch=time.time())
    rec["env"] = record_env.record()
    own: set[int] = set()
    rec["gates_before"] = gates.snapshot() if opts.record_snapshots else None
    if rec["gates_before"] is not None:
        rec["gates_before"]["gpu_apps"] = gpu_health.compute_apps()
    # D-021 R28: the start temperature and the fan governor's state at the start (and at the end, below)
    rec["acpi_start_c"] = (rec["gates_before"] or {}).get("acpi_max_c")
    rec["fan_state_start"] = gates.fan_state() if opts.record_snapshots else None
    rec["ref_stop"] = ref_rec                          # D-021 R12 (None: the job's harness has no ref_stop)
    if harness is not None and harness["cuda_cache"]:
        opts.cuda_cache_dir.mkdir(parents=True, exist_ok=True)

    sampler, own_sampler = None, False
    if opts.gpu_health:
        if opts.shared_sampler is not None:      # the pool's one sampler: its own_pids are every run in flight
            sampler = opts.shared_sampler
        else:
            sampler = gpu_health.Sampler(own_pids=lambda: own | (opts.extra_own_pids() if opts.extra_own_pids
                                                                 else set()), period_s=opts.gpu_period_s,
                                         tolerate=(gpu_health.is_serving_pid if opts.tolerate_serving_gpu else None))
            sampler.start()
            own_sampler = True

    reasons: list[str] = rec["reasons"]
    acpi = dict(max_c=None, samples=0, period_s=ACPI_SAMPLE_S) if opts.record_snapshots else None
    status = None
    t0 = None
    cache_t0 = None
    stop_info: dict | None = None
    monitor: RefMonitor | None = None
    opts.cupy_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        # ---------------------------------------------------------------- launch
        for p in procs:
            env = launch_env(p)
            logf = open(p.log_path, "ab")
            try:
                p.launch_mono, p.launch_epoch = time.monotonic(), time.time()
                p.popen = subprocess.Popen(p.cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           env=env, cwd=str(paths.ROOT), start_new_session=True)
            finally:
                logf.close()
            own.add(p.popen.pid)
        rec["processes"] = [_proc_record(p, None) for p in procs]
        _atomic_json(run_dir / "run.json", rec)            # "running": a crash of the runner leaves the pids on file

        # ----------------------------------------------------------- barrier
        end = time.monotonic() + opts.ready_timeout_s
        while True:
            for p in procs:
                if p.ready_mono is None and mb.all_ready((p.role,)):
                    p.ready_mono = time.monotonic()
            if all(p.ready_mono is not None for p in procs):
                break
            dead = [p for p in procs if p.ready_mono is None and p.poll() is not None]
            if dead:
                launcher = [p for p in dead if _launcher_failure(run_dir, p)]
                if launcher:     # exited by itself before its own code wrote a summary: the harness, not the arm
                    status = "void"
                    rec["void_cause"] = "runner_error"
                    reasons += [f"{p.role} exited rc={p.exit_code} before ready without a summary (launcher or "
                                f"environment failure, see {p.log_path.name})" for p in launcher]
                else:
                    status = "failed"
                reasons += [f"{p.role} exited rc={p.exit_code} before ready" for p in dead if p not in launcher]
                break
            if opts.abort is not None and opts.abort.is_set():
                status = "void"
                reasons.append("aborted before start: " + (opts.abort_reason() if opts.abort_reason else "abort"))
                break
            if time.monotonic() > end:
                status = "failed"
                reasons.append(f"not ready after {opts.ready_timeout_s:.0f} s: "
                               + ", ".join(p.role for p in procs if p.ready_mono is None))
                break
            time.sleep(0.05)
        if status is None:
            dead = [p for p in procs if p.poll() is not None]        # alive at the moment of release?
            if dead:
                status = "failed"
                reasons += [f"{p.role} exited rc={p.exit_code} at the barrier" for p in dead]
        cache_t0 = _count_files(opts.cupy_cache_dir)

        # ---------------------------------------------------------- the clock
        if status is None:
            t0 = mb.release()
            rec["t0"], rec["t0_epoch"] = t0, time.time()
            t_end = t0 + float(job["budget"]) + opts.grace_s
            next_acpi = next_ref = 0.0
            if stop_ref is not None:
                monitor = RefMonitor(run_dir, arm.roles, names, obj_c, obj_off, sense, stop_ref, float(job["budget"]))
            while True:
                alive = [p for p in procs if p.poll() is None]
                if not alive:
                    break
                now = time.monotonic()
                if acpi is not None and now >= next_acpi:      # the box's heat while this run holds it
                    next_acpi = now + ACPI_SAMPLE_S
                    c = gates.acpi_max_c()
                    if c is not None:
                        acpi["max_c"] = c if acpi["max_c"] is None else max(acpi["max_c"], c)
                        acpi["samples"] += 1
                if opts.abort is not None and opts.abort.is_set():
                    stop_info = dict(reason="abort: " + (opts.abort_reason() if opts.abort_reason else ""),
                                     t=now - t0, kind="abort")
                    status = "void"
                    reasons.append(stop_info["reason"])
                    break
                if now >= t_end:
                    stop_info = dict(reason="budget + grace", t=now - t0, kind="budget")
                    break
                if monitor is not None and now >= next_ref:      # D-021 R12: the proven optimum reached
                    next_ref = now + opts.ref_poll_s
                    trig = monitor.poll()
                    if trig is not None:
                        stop_info = dict(reason=f"ref_reached: {trig['role']} incumbent #{trig['k']} at "
                                                f"t {trig['t']:.3f} s, objective {trig['obj']!r} (z* {stop_ref.value!r},"
                                                f" gamma {trig['gamma']:g}, gap {trig['gap']:.3g})",
                                         t=time.monotonic() - t0, kind="ref_reached")
                        break
                ended = [p for p in procs if p.exit_code is not None]
                decisive = [p for p in ended if p.exit_code != 0 or _exit_reason(run_dir, p.role) in DEFINITIVE]
                if decisive and opts.stop_partners_on_exit:
                    stop_info = dict(reason=", ".join(
                        f"{p.role} ended rc={p.exit_code}" + (f" ({_exit_reason(run_dir, p.role)})" if p.exit_code == 0
                                                              else "") for p in decisive), t=now - t0,
                        kind="partner_end",
                        proven=any(p.exit_code == 0 and _exit_reason(run_dir, p.role) in DEFINITIVE for p in decisive))
                    break
                time.sleep(0.2)
        else:
            aborted = status == "void" and rec.get("void_cause") != "runner_error"
            stop_info = dict(reason="start failed", t=None, kind="abort" if aborted else "start_failed")
    except KeyboardInterrupt:
        status = "void"
        reasons.append("interrupted (KeyboardInterrupt)")
        stop_info = dict(reason="interrupted", t=(time.monotonic() - t0) if t0 else None, kind="interrupted")
        raise
    except Exception as e:
        # a fault of the harness (cannot launch, mailbox gone), not of the arm: void, so the pool halts and reruns
        status = "void"
        rec["void_cause"] = "runner_error"
        reasons.append(f"runner error: {e!r}")
        stop_info = dict(reason="runner error", t=(time.monotonic() - t0) if t0 else None, kind="runner_error")
        raise
    finally:
        # ------------------------------------------------------------ stop
        if any(p.popen is not None and p.poll() is None for p in procs):
            try:
                mb.stop((stop_info or {}).get("reason", "runner"))
            except Exception as e:  # noqa: BLE001
                reasons.append(f"could not write stop: {e!r}")
            if (stop_info or {}).get("kind") in ("abort", "interrupted"):
                # the run is void anyway: do not hold memory / the GPU for kill_after_s below the memory floor or
                # at the thermal limit. cuOpt has no interrupt API (SIGTERM only sets a flag): SIGKILL it now.
                for p in procs:
                    if p.spec.kind == "cuopt" and p.popen is not None and p.poll() is None:
                        p.killed = "SIGKILL (abort)"
                        p.signal(signal.SIGKILL)
                _wait_or_kill(procs, ABORT_KILL_AFTER_S)
            else:
                if harness is not None and harness.get("cuopt_kill_on_end") and definitive_stop(stop_info):
                    # D-021 phase 2: the outcome is decided (a partner's proof, or R12's ref_reached) and cuOpt cannot
                    # be interrupted — SIGKILL it at once instead of after kill_after_s + 10 s (stopped_by_runner)
                    for p in procs:
                        if p.spec.kind == "cuopt" and p.popen is not None and p.poll() is None:
                            p.killed = "SIGKILL (definitive outcome)"
                            p.signal(signal.SIGKILL)
                _wait_or_kill(procs, opts.kill_after_s)
        if sampler is not None and own_sampler:
            sampler.stop()
        try:
            rec["exchange"] = _exchange_log(mb, opts, run_dir,
                                            cuopt=any(p.spec.kind == "cuopt" and "--send" in p.argv for p in procs))
        finally:
            _remove_mailbox(mb, spec, opts)
        rec["stop"] = stop_info
        rec["cupy_cache"] = dict(dir=str(opts.cupy_cache_dir), files_at_t0=cache_t0,
                                 files_at_end=_count_files(opts.cupy_cache_dir))
        if rec["cupy_cache"]["files_at_t0"] is not None and rec["cupy_cache"]["files_at_end"] is not None:
            rec["cupy_cache"]["compiled_on_clock"] = rec["cupy_cache"]["files_at_end"] > rec["cupy_cache"]["files_at_t0"]
        # A process the runner killed because a partner ended the race (a proven outcome, or a crash that keeps its
        # own rc) is stopped_by_runner, not a crash of its own: cuOpt cannot be interrupted, so after a partner's
        # proof it is SIGKILLed after kill_after_s; build_runs must not count it in the per-arm crash numbers.
        # A run that R12 stopped at the proven optimum (`ref_reached`) is treated the same way: the processes the
        # runner killed then are stopped_by_runner, and the run's last checked incumbent is held to T in every metric.
        for p in procs:
            p.stopped_by_runner = bool(p.killed and (stop_info or {}).get("kind") in ("partner_end", "ref_reached"))
        rec["processes"] = [_proc_record(p, t0) for p in procs]
        rec["exit_codes"] = {p.role: p.exit_code for p in procs}
        for p in procs:   # a crashed process: its crash-safe .part file becomes <role>.incumbents.npz, so the
            if p.exit_code != 0 or not (run_dir / f"{p.role}.incumbents.npz").exists():   # checker can still read it
                try:
                    from itrea.solvers.common import recover_incumbents
                    recover_incumbents(run_dir, p.role)
                except Exception as e:  # noqa: BLE001 — recovery is best effort; the reason is recorded
                    reasons.append(f"recover {p.role} incumbents: {e!r}")
        if monitor is not None:                          # D-021 R12: what the runner read, and the confirmation
            monitor.drain()                              # records left unread at the stop: candidates only
            monitor.close()
            ref_rec.update(monitor.summary())
            trig = monitor.trigger
            if trig is not None and (stop_info or {}).get("kind") == "ref_reached":
                trig["detected_t"] = round(float(stop_info["t"]), 6)
                if opts.ref_check:
                    try:
                        ref_rec["check"] = check_ref_trigger(run_dir, trig, job["instance"], stop_ref, sense,
                                                             arm_label(job))
                    except Exception as e:  # noqa: BLE001 — an unconfirmed stop voids the run
                        ref_rec["check"] = dict(ok=False, reason=f"check failed: {type(e).__name__}: {e}"[:300])
        rec["wall_s"] = (max((p.exit_mono or time.monotonic()) for p in procs) - t0) if t0 else None
        # GPU health over the timed window
        if sampler is not None:
            t_to = None
            if t0 is not None:
                t_to = min(t0 + float(job["budget"]), max((p.exit_mono or time.monotonic()) for p in procs))
            rec["gpu_health"] = sampler.summary(t_from=t0, t_to=t_to, gpu_run=arm.gpu)
        else:
            rec["gpu_health"] = None
        rec["gates_after"] = gates.snapshot() if opts.record_snapshots else None
        rec["acpi_during"] = acpi
        rec["fan_state_end"] = gates.fan_state() if opts.record_snapshots else None
        rec["cuda_cache"] = (cuda_cache_on_clock(opts.cuda_cache_dir, rec.get("t0_epoch"))
                             if harness is not None and harness["cuda_cache"] else None)
        rec["compiled_on_clock_cuda"] = (rec["cuda_cache"] or {}).get("compiled_on_clock")
        # status: void (external) > failed (any exit code != 0 — a SIGKILL included —, a start failure) > completed.
        # A process that exits 0 after the runner's SIGTERM is recorded in processes[*].killed, not as a failure; a
        # stopped_by_runner process (above) is not a failure either.
        if status is None:
            bad = [p for p in procs if p.exit_code != 0 and not p.stopped_by_runner]
            if bad:
                status = "failed"
                reasons += [f"{p.role} rc={p.exit_code}" + (f" ({p.killed})" if p.killed else "") for p in bad]
            else:
                status = "completed"
        if status != "void" and arm.gpu and t0 is not None and rec["gpu_health"] and \
                rec["gpu_health"].get("verdict") == "void":
            reasons += ["gpu health: " + r for r in rec["gpu_health"]["reasons"]]
            status = "void"
            rec["void_cause"] = "gpu_health"
        # D-021's external causes (R15, R25, R27, R29 and unexplained kills), each recorded whether or not it decides
        # the status; the first in VOID_ORDER voids a run that is not void already (a failure of SCIP or the EA,
        # their caps included, stays an arm outcome); what decides nothing is in void_info
        checks, rec["void_info"] = void_checks(job, arm, harness, procs, run_dir, rec, opts, t0)
        rec["void_checks"] = {c: why for c, why in checks}
        if status != "void" and checks:
            cause, why = checks[0]
            status = "void"
            rec["void_cause"] = cause
            reasons.append(f"{cause}: {why}")
        if status == "void" and "void_cause" not in rec:
            rec["void_cause"] = "abort" if any(r.startswith(("abort", "aborted")) for r in reasons) else "interrupted"
        rec["status"] = status
        rec["trajectories"] = merge_trajectories(run_dir, arm.roles, sense,
                                                 "solver" if any(p.role == "solver" for p in arm.procs) else None)
        rec["ended_epoch"] = time.time()
        _atomic_json(run_dir / "run.json", rec)
        opts.log(f"{run_id}: {status}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    return rec


VOID_ORDER = ("thermal_watch", "gpu_cap", "cuopt_memory", "external_kill", "compiled_on_clock",
              "ref_rejected")                                                                       # by precedence


def definitive_stop(stop_info: dict | None) -> bool:
    """The run's outcome is decided: R12's ref_reached, or a partner that ended with a proven outcome (exit 0 and an
    exit reason in DEFINITIVE) — not a partner's crash."""
    s = stop_info or {}
    return s.get("kind") == "ref_reached" or (s.get("kind") == "partner_end" and bool(s.get("proven")))


def cuda_cache_on_clock(d: Path, t0_epoch: float | None) -> dict:
    """D-021 R15: entry files of the project's CUDA JIT cache written at or after t0 (mtime >= t0_epoch; the `index`
    file, rewritten on lookups, is ignored) = a compile on the clock. `compiled_on_clock` None when the run never
    started."""
    out: dict = dict(dir=str(d), t0_epoch=t0_epoch, n_on_clock=None, entries_on_clock=[], compiled_on_clock=None)
    if t0_epoch is None:
        return out
    names = []
    for root, _, files in os.walk(d):
        for f in files:
            if f == "index":
                continue
            p = Path(root) / f
            try:
                if p.stat().st_mtime >= t0_epoch:
                    names.append(os.path.relpath(p, d))
            except OSError:
                continue
    out.update(n_on_clock=len(names), entries_on_clock=sorted(names)[:50], compiled_on_clock=bool(names))
    return out


def _summary(run_dir: Path, role: str) -> dict:
    try:
        d = json.loads((run_dir / f"{role}.summary.json").read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _exit_epoch(p: Proc) -> float | None:
    if p.launch_epoch is None or p.launch_mono is None or p.exit_mono is None:
        return None
    return p.launch_epoch + (p.exit_mono - p.launch_mono)


def _maxrss_gb(p: Proc) -> float | None:
    return round(p.rusage.ru_maxrss / 1048576, 3) if p.rusage is not None else None


def _unsent_sigkill(p: Proc) -> bool:
    """A SIGKILL the runner did not send (its own kills set `killed`)."""
    return p.exit_code == -signal.SIGKILL and p.killed is None


def cap_kill(p: Proc, scope_result: Callable[[str], str | None] | None = None) -> str | None:
    """A SIGKILL the runner did not send that was the process's own host cap (D-021 R29): its scope's Result is
    `oom-kill`, or its peak RSS reached HOST_CAP_SHARE of its MemoryMax. None otherwise (also for any other exit)."""
    if not _unsent_sigkill(p):
        return None
    res = (scope_result or gates.scope_result)(p.scope_unit) if p.scope_unit else None
    rss = _maxrss_gb(p)
    if res == "oom-kill" or (rss is not None and rss >= HOST_CAP_SHARE * p.mem_gb):
        return f"{p.role}: killed at its {p.mem_gb:g} GB host cap (scope Result {res}, maxrss {rss} GB)"
    return None


def rmm_error(p: Proc, run_dir: Path) -> dict | None:
    """cuOpt's RMM allocation error from its summary (error status 6 "Memory allocation failed", or an RMM / cuOpt
    out-of-memory exception; the warm-up's included): {reason, peak, limit, at_cap}; at_cap = the peak reached
    RMM_CAP_SHARE of the limit (R29's 0.8 x cap reading). None when there is none."""
    if p.spec.kind != "cuopt":
        return None
    s = _summary(run_dir, p.role)
    err = " ".join(str(s.get(k) or "") for k in ("error", "error_message"))
    low = err.lower()
    if not ((s.get("error_status") == 6 and "memory allocation failed" in low) or
            any(k in low for k in ("out_of_memory", "outofmemory", "std::bad_alloc"))):
        return None
    peak, lim = s.get("rmm_peak_bytes"), s.get("rmm_limit_bytes")
    at_cap = bool(peak and lim and float(peak) >= RMM_CAP_SHARE * float(lim))
    return dict(reason=f"{p.role}: RMM allocation error, peak {peak} of limit {lim} bytes ({err.strip()[:160]})",
                peak=peak, limit=lim, at_cap=at_cap)


def cuopt_memory_fault(p: Proc, run_dir: Path, scope_result: Callable[[str], str | None] | None = None,
                       capped: str | None = None) -> str | None:
    """D-021 R29: cuOpt failed at its own cap, CONFIRMED — an RMM allocation error whose peak reached 0.8 x the limit,
    or a SIGKILL at its host cap (`cap_kill`, or its already computed result `capped`: the scope's Result can be read
    only once, it is reset after reading). None otherwise."""
    if p.spec.kind != "cuopt":
        return None
    r = rmm_error(p, run_dir)
    if r is not None and r["at_cap"]:
        return r["reason"]
    return capped if capped is not None else cap_kill(p, scope_result)


def _external_stop(p: Proc, run_dir: Path) -> bool:
    """A process stopped by someone else than the runner (SIGTERM / SIGKILL it did not send, or a summary that says
    it ended on a SIGTERM): thermal-watch's `systemctl stop` of its scope looks like this."""
    if p.killed is not None:
        return False
    return p.exit_code in (-signal.SIGTERM, -signal.SIGKILL) or _summary(run_dir, p.role).get("exit_reason") == "sigterm"


def thermal_watch_hits(lines: list[str], procs: list[Proc]) -> tuple[list[str], set[str]]:
    """thermal-watch's HOT lines that name one of the run's scope units in their stopped list, and the roles they
    stopped."""
    units = {p.scope_unit: p.role for p in procs if p.scope_unit}
    hits, roles = [], set()
    for line in lines:
        named = {u for u in gates.parse_hot_line(line)["units"] if u in units}
        if named:
            hits.append(line)
            roles |= {units[u] for u in named}
    return hits, roles


def void_checks(job: dict, arm: ArmSpec, harness: dict | None, procs: list[Proc], run_dir: Path, rec: dict,
                opts: PairOptions, t0: float | None) -> tuple[list[tuple[str, str]], dict]:
    """D-021's external void causes of a finished run, in VOID_ORDER: ([(cause, reason)], info). `info` records what
    was seen but decides nothing (HOT lines that name none of our scopes, a fan state with cap=1 stamped outside the
    timed window, an RMM error below the cap, a cap hit of cuOpt under `cuopt_memory_void: false`, the gpu_cap
    details the pool reads: `gpu_cap.unknown_lock`).
    - thermal_watch (R27): a thermal-watch HOT line (>= 98 °C) whose stopped list names one of this run's scope units,
      stamped between the first launch and the last exit + opts.thermal_watch_wait_s (after an external stop the
      runner waits that long for it: the line is written after the units are stopped);
    - gpu_cap (R25, GPU arms, opts.cap_floor_mhz set): busy SM median < cap_floor_mhz, or the timed window overlaps a
      governor clock-cap interval (its log), or a fan state read cap=1 with its own timestamp inside the timed window;
    - cuopt_memory (R29, with a harness whose cuopt_memory_void is on): `cuopt_memory_fault` (a confirmed cap hit);
    - external_kill (every arm): a SIGKILL of any process that the runner did not send and that is neither its own
      host cap (`cap_kill`: an arm outcome for SCIP / the EA, cuopt_memory or an arm outcome for cuOpt) nor
      thermal-watch's stop;
    - compiled_on_clock (R15, with a harness whose cuda_cache is on): an entry of the project's CUDA cache, or of the
      EA's CuPy cache, written after t0 (the same terms for both GPU codes);
    - ref_rejected (R12, a harness with ref_stop): the incumbent that triggered the stop at the proven optimum is not
      confirmed by `check_ref_trigger` (the exact checker rejects it, its exact objective misses the stop, or it could
      not be checked); the pool reruns the block once with the stop off."""
    out: list[tuple[str, str]] = []
    info: dict = {}
    capped: dict[str, str] = {}            # once per process: reading a scope's Result resets the scope
    for p in procs:
        ck = cap_kill(p)
        if ck:
            capped[p.role] = ck
    launched = [p.launch_epoch for p in procs if p.launch_epoch is not None]
    tw_roles: set[str] = set()
    if launched:
        t_from = min(launched)
        exits = [e for e in (_exit_epoch(p) for p in procs) if e is not None]
        t_to = (max(exits) if exits else time.time()) + opts.thermal_watch_wait_s
        lines = gates.thermal_watch_stops(t_from, min(time.time(), t_to))
        hits, tw_roles = thermal_watch_hits(lines, procs)
        if not hits and any(_external_stop(p, run_dir) and p.role not in capped for p in procs):
            while not hits and time.time() < t_to:
                time.sleep(min(2.0, max(0.05, t_to - time.time())))
                lines = gates.thermal_watch_stops(t_from, min(time.time(), t_to))
                hits, tw_roles = thermal_watch_hits(lines, procs)
        if hits:
            out.append(("thermal_watch", f"thermal-watch stopped {sorted(tw_roles)} at >= "
                                         f"{gates.THERMAL_WATCH_C:g} °C: {hits[0]}"))
        other = [x for x in lines if x not in hits]
        if other:
            info["thermal_watch_lines"] = other[:5]       # the box was >= 98 °C, our scopes not named
    t0e = rec.get("t0_epoch")
    if arm.gpu and t0 is not None and opts.cap_floor_mhz is not None:
        why = []
        med = (rec.get("gpu_health") or {}).get("median_sm_mhz")
        low = med is not None and med < opts.cap_floor_mhz
        if low:
            why.append(f"busy SM median {med:.0f} MHz < {opts.cap_floor_mhz:.0f} MHz")
        ov: list = []
        span = min(float(job["budget"]), float(rec.get("wall_s") or 0.0))
        if t0e is not None:
            ov = gates.cap_overlaps(t0e, t0e + span)
            if ov:
                why.append(f"overlaps the governor's clock cap ({len(ov)} interval(s) from {ov[0]['on']:.0f})")
        caps = {}
        for k in ("fan_state_start", "fan_state_end"):
            st = rec.get(k) or {}
            caps[k] = st.get("cap")
            if st.get("cap") == 1:
                inside = t0e is not None and st.get("epoch") is not None and t0e <= st["epoch"] <= t0e + span
                if inside:
                    why.append(f"{k}: cap=1 at {st.get('iso')}, inside the timed window")
                else:
                    info.setdefault("fan_cap_outside_window", []).append(f"{k}: cap=1 at {st.get('iso')}")
        info["gpu_cap"] = dict(median_sm_mhz=med, floor_mhz=opts.cap_floor_mhz, overlaps=ov,
                               fan_cap_start=caps["fan_state_start"], fan_cap_end=caps["fan_state_end"],
                               unknown_lock=bool(low and not ov and caps["fan_state_start"] == 0
                                                 and caps["fan_state_end"] == 0))
        if why:
            out.append(("gpu_cap", "; ".join(why)))
    for p in procs:
        r = rmm_error(p, run_dir)
        if r is not None and not r["at_cap"]:
            info.setdefault("rmm_error_below_cap", []).append(r["reason"])
    if capped:
        info["cap_kills"] = capped                       # SCIP / EA at their caps: arm outcomes (R29)
    if harness is not None:
        mem = [r for r in (cuopt_memory_fault(p, run_dir, capped=capped.get(p.role)) for p in procs) if r]
        if mem:
            if harness.get("cuopt_memory_void", True):
                out.append(("cuopt_memory", "; ".join(mem)))
            else:                                        # M3: the cap hit is the measurement
                info["cuopt_memory"] = "; ".join(mem)
    ext = [p for p in procs if _unsent_sigkill(p) and p.role not in capped and p.role not in tw_roles]
    if ext:
        out.append(("external_kill", "; ".join(f"{p.role}: SIGKILL not sent by the runner, not at its "
                                               f"{p.mem_gb:g} GB cap (maxrss {_maxrss_gb(p)} GB)" for p in ext)))
    if harness is not None and harness["cuda_cache"]:
        cc, cp = rec.get("cuda_cache") or {}, rec.get("cupy_cache") or {}
        why = []
        if cc.get("compiled_on_clock"):
            why.append(f"{cc['n_on_clock']} CUDA cache entries after t0 ({', '.join(cc['entries_on_clock'][:3])})")
        if cp.get("compiled_on_clock") and any(p.spec.kind == "ea" and p.spec.gpu for p in procs):
            why.append(f"CuPy cache grew after t0 ({cp.get('files_at_t0')} -> {cp.get('files_at_end')} files)")
        if why:
            out.append(("compiled_on_clock", "; ".join(why)))
    chk = (rec.get("ref_stop") or {}).get("check")
    if chk is not None and not chk.get("ok"):           # D-021 R12: the stop was not confirmed by the exact checker
        out.append(("ref_rejected", f"the incumbent that triggered the stop at the proven optimum is not confirmed: "
                                    f"{chk.get('reason')}"))
    return sorted(out, key=lambda x: VOID_ORDER.index(x[0])), info


def launch_env(p: Proc, base: dict | None = None) -> dict:
    """A process's environment: the runner's (`base`, default os.environ) without CuPy's pool limit and without
    `p.env_unset`, then `p.env_extra`."""
    env = dict(os.environ if base is None else base)
    env.pop("CUPY_GPU_MEMORY_LIMIT", None)
    for k in p.env_unset:
        env.pop(k, None)
    env.update(p.env_extra)
    return env


def _launcher_failure(run_dir: Path, p: Proc) -> bool:
    """A process that ended before ready by its own exit status (not a signal) without writing `<role>.summary.json`:
    its own code never took control (systemd-run or taskset failed, the interpreter or an import failed, bad
    arguments from the runner). Every role process writes its summary on any exception (common.py / ea/run.py exit
    contract), so this is a fault of the harness or the environment, not an arm crash. A signal (an OOM kill under
    MemoryMax, a segfault while parsing) stays an arm failure."""
    return p.exit_code is not None and p.exit_code >= 0 and not (run_dir / f"{p.role}.summary.json").exists()


def _exit_reason(run_dir: Path, role: str) -> str | None:
    """`exit_reason` of a process's summary (written just before it exits), or None."""
    try:
        return json.loads((run_dir / f"{role}.summary.json").read_text()).get("exit_reason")
    except (OSError, ValueError, AttributeError):
        return None


def _wait_or_kill(procs: list[Proc], kill_after_s: float) -> None:
    end = time.monotonic() + kill_after_s
    while time.monotonic() < end and any(p.popen is not None and p.poll() is None for p in procs):
        time.sleep(0.1)
    for sig, name, wait in ((signal.SIGTERM, "SIGTERM after stop", 10.0), (signal.SIGKILL, "SIGKILL", 5.0)):
        left = [p for p in procs if p.popen is not None and p.poll() is None]
        if not left:
            return
        for p in left:
            p.killed = name
            p.signal(sig)
        end = time.monotonic() + wait
        while time.monotonic() < end and any(p.poll() is None for p in left):
            time.sleep(0.1)


def _proc_record(p: Proc, t0: float | None) -> dict:
    d = dict(role=p.role, kind=p.spec.kind, backend=p.spec.backend, cmd=p.cmd, cores=p.cores, memory_max_gb=p.mem_gb,
             gpu_limit_gb=p.gpu_gb, env={k: v for k, v in p.env_extra.items()}, env_unset=list(p.env_unset),
             scope_unit=p.scope_unit, pid=p.pid,
             launch_epoch=p.launch_epoch, log=p.log_path.name,
             startup_s=round(p.ready_mono - p.launch_mono, 3) if (p.ready_mono and p.launch_mono) else None,
             exit_code=p.exit_code, killed=p.killed, stopped_by_runner=p.stopped_by_runner,
             exit_t=round(p.exit_mono - t0, 3) if (p.exit_mono and t0) else None)
    if p.rusage is not None:
        d.update(cpu_user_s=round(p.rusage.ru_utime, 3), cpu_sys_s=round(p.rusage.ru_stime, 3),
                 maxrss_mb=round(p.rusage.ru_maxrss / 1024, 1))
    return d


# --------------------------------------------------------------------------------------------- runner process
def _term(signum, frame):
    """SIGTERM / SIGHUP -> KeyboardInterrupt, once: the finally blocks then stop every child (they are session leaders
    in their own scopes, so neither a plain kill of the runner nor `systemctl stop` of its unit reaches them)."""
    for sig in (signal.SIGTERM, signal.SIGHUP):      # a repeat must not escape the except-KeyboardInterrupt cleanup
        try:
            signal.signal(sig, signal.SIG_IGN)
        except (ValueError, OSError):
            pass
    raise KeyboardInterrupt


def install_term_handlers() -> dict:
    """Install `_term` for SIGTERM and (unless ignored: nohup) SIGHUP in the main thread; returns the previous handlers
    for `restore_handlers`. The abort path (ABORT_KILL_AFTER_S + 10 s SIGTERM + 5 s SIGKILL) fits systemd's default
    90 s stop timeout."""
    prev: dict = {}
    if threading.current_thread() is not threading.main_thread():
        return prev
    prev[signal.SIGTERM] = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _term)
    h = signal.getsignal(signal.SIGHUP)
    if h is not signal.SIG_IGN:
        prev[signal.SIGHUP] = h
        signal.signal(signal.SIGHUP, _term)
    return prev


def restore_handlers(prev: dict) -> None:
    for sig, h in prev.items():
        try:
            signal.signal(sig, h if h is not None else signal.SIG_DFL)
        except (ValueError, TypeError, OSError):
            pass


def pin_runner(cores=RUNNER_CORES) -> set[int] | None:
    """Pin the runner (and every thread and helper it starts later: samplers, nvidia-smi, the instance parse) to A725
    cores no lane uses, so its work never lands on a core that holds a timed solver. Children re-pin with taskset."""
    try:
        want = set(cores) & os.sched_getaffinity(0)
        if not want:
            return None
        os.sched_setaffinity(0, want)
        return want
    except (OSError, AttributeError):
        return None


# ------------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="itrea.experiments.pair", description="Run ONE job (smoke tests; the pool runs "
                                 "job files). A toy runs as is; any other instance needs the lease, the thermal gate "
                                 "and systemd scopes, a screening candidate the prereg-v1 tag; the runner lock is "
                                 "always taken.")
    ap.add_argument("--job", help="the job as a JSON object (overrides the other job arguments)")
    ap.add_argument("--instance")
    ap.add_argument("--arm", default="A0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--set", default="smoke", dest="set_")
    ap.add_argument("--config")
    ap.add_argument("--emphasis")
    ap.add_argument("--mailbox", choices=["file", "redis"], default="file")
    ap.add_argument("--no-systemd", action="store_true", help="toy instances only")
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--traces-dir", type=Path, default=paths.TRACES)
    ap.add_argument("--gpu-mem-gb", type=float)
    a = ap.parse_args(argv)
    if a.job:
        job = json.loads(a.job)
    else:
        if not a.instance:
            ap.error("--instance or --job is required")
        job = dict(instance=a.instance, set=a.set_, arm=a.arm, seed=a.seed, budget=a.budget, config=a.config,
                   overrides={}, split="smoke", emphasis=a.emphasis)
    from itrea.experiments import make_jobs, pool     # lazy: both import this module
    toy = job.get("split") == "smoke" and make_jobs.is_toy(job)
    if not toy and a.no_systemd:
        print("error: --no-systemd is for toy instances only (D-003: every process in a MemoryMax scope)",
              file=sys.stderr)
        return 2
    opts = PairOptions(traces_dir=a.traces_dir, mailbox=a.mailbox, use_systemd=not a.no_systemd, pin=not a.no_pin,
                       gpu_mem_gb=a.gpu_mem_gb)
    prev = install_term_handlers()
    if opts.pin:
        pin_runner()
    try:
        with gates.RunLock():                 # never beside a pool (or another pair run) on the same cores
            if not toy:
                pool.check_frozen([job], allow_unfrozen=True)      # a candidate / screening job needs prereg-v1
                gates.check_lease(pool.WHO)
                if pool.is_quiet(job):
                    gates.check_serving_stopped()
                gates.wait_cool(log=lambda m: print(m, file=sys.stderr))
            rec = run_pair(job, opts)
    except (RunError, gates.GateError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted: the run's processes were stopped and its record marks it void", file=sys.stderr)
        return 130
    finally:
        restore_handlers(prev)
    print(json.dumps({k: rec[k] for k in ("run_id", "status", "reasons", "exit_codes")}, indent=1))
    return 0 if rec["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
