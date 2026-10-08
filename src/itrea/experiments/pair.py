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
OMP/OPENBLAS/MKL_NUM_THREADS = 1. The three-way arm A9 (SCIP <-> EA <- cuOpt, D-015) runs two partners, which must not
share cores: SCIP on the lane's first X925 core, the EA on its second (`ProcSpec.x925`), cuOpt on the A725 cores
(num_cpu_threads = 3 = their number on the GPU lane); two CUDA processes, so the memory gate counts the EA pool and
cuOpt's RMM limit (both = gpu_mem_gb) plus two CUDA overheads. The runner gives the EA `exchange.from_cuopt: true`
(arm-implied, like A5's "no exchange") and cuOpt `--send` (its own channels `to_ea_cuopt` / `info_cuopt`); SCIP gets
A3's flags. GPU-health voiding treats every process the runner launched (and its descendants) as the run's own. A9's
matched control is A3x (SCIP on x925[0], the EA on x925[1], no cuOpt), not A3: A3's EA runs on the three A725 cores,
and its HiGHS LP worker (`seeding.lp_source: highs_async`, which inherits the EA's affinity) beside it, while A9's EA
shares one faster X925 core with that worker.

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
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import quote

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
ABORT_KILL_AFTER_S = 5.0      # an abort (memory floor, thermal pause, interrupt): stop → this → SIGTERM → SIGKILL
RUNNER_CORES = (4, 10, 11, 12, 13, 14)   # A725 cores no lane uses (1–3: partner, 0: Redis): the runner's own work
MAILBOX_ROOT = Path("/dev/shm/itrea")
CUPY_CACHE_DIR = Path.home() / ".cache/itrea/cupy"
REDIS_URL = "redis://127.0.0.1:6390/0"


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

    @property
    def gpu(self) -> bool:
        return self.kind in GPU_KINDS or (self.kind == "ea" and self.backend == "cupy")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    procs: tuple[ProcSpec, ...]
    ea_overrides: tuple[tuple[str, object], ...] = ()   # arm-implied EA configuration changes
    lane: str = "solo"         # "partner": GPU lane / partner footprint (one at a time) | "solo": solver-only CPU lane
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
    A3x, a725 = the partner (EA or cuOpt; cuOpt's num_cpu_threads = PARTNER_THREADS = len(a725) on the GPU lane)."""
    name: str
    x925: tuple[int, ...]
    a725: tuple[int, ...] = ()
    gpu: bool = False

    def fits(self, arm: ArmSpec) -> bool:
        return len(self.x925) >= arm.n_x925 and (not arm.needs_a725 or len(self.a725) >= 1) and (self.gpu or not arm.gpu)


GPU_LANE = Lane("g0", (5, 6), (1, 2, 3), gpu=True)


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


def solver_settings(job: dict, opts: PairOptions, cfg: EAConfig) -> dict:
    """What identifies a run of an arm without an EA (its <cfg-hash>)."""
    arm = arm_spec(job)
    d: dict = {"arm": arm.name, "emphasis": job.get("emphasis") if any(p.emphasis for p in arm.procs) else None}
    if any(p.kind == "cuopt" for p in arm.procs):
        d.update(gpu_mem_gb=cfg.gpu_mem_gb, cuopt_threads=opts.cuopt_threads or PARTNER_THREADS)
    return d


def run_hash(job: dict, opts: PairOptions | None = None) -> str:
    """<cfg-hash> of the run id: the effective EA configuration hash for arms with an EA (it equals the EA summary's
    config_hash); a hash of the solver-side settings otherwise. An arm with an EA AND cuOpt (A9) folds in cuOpt's
    thread count when it is not the default PARTNER_THREADS, so a thread-count variant gets its own run id (the default
    id stays the EA configuration hash; build_runs reads the EA hash from `run.json["config"]["ea_hash"]`)."""
    opts = opts or PairOptions()
    cfg, _ = effective_config(job, opts)
    arm = arm_spec(job)
    if arm.has_ea:
        h = cfg.hash()
        n = opts.cuopt_threads or PARTNER_THREADS
        if any(p.kind == "cuopt" for p in arm.procs) and n != PARTNER_THREADS:
            h = hashlib.sha256(json.dumps({"ea": h, "cuopt_threads": n}, sort_keys=True).encode()).hexdigest()[:12]
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


def plan(job: dict, opts: PairOptions, run_dir: Path, mailbox_spec: str, cfg: EAConfig) -> list[Proc]:
    """The processes of a run (nothing is launched)."""
    arm = arm_spec(job)
    lane = opts.lane
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
            cores = list(lane.a725)
            mem = opts.cuopt_mem_gb
            argv += ["--cpu-threads", str(opts.cuopt_threads or PARTNER_THREADS), "--gpu-mem-gb", _fmt_set(gpu_gb)]
            argv += cuopt_flags(spec, cfg if arm.has_ea else None)
            g = gpu_gb                                   # RMM limit = the EA's pool limit (D-002 §3)
        else:
            raise RunError(f"unknown process kind {spec.kind!r}")
        env = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
               "ITREA_RUN_ID": run_dir.name, "ITREA_ROLE": spec.role, "CUPY_CACHE_DIR": str(opts.cupy_cache_dir)}
        if g is not None:
            env["CUPY_GPU_MEMORY_LIMIT"] = str(int(g * 1024**3))   # CuPy reads it at import (EA); cuOpt: RMM limit
            env["ITREA_GPU_MEM_GB"] = _fmt_set(g)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""      # a CPU process must not create a CUDA context (A4 = NumPy EA)
        cmd = list(argv)
        if opts.pin and cores:
            cmd = ["taskset", "-c", ",".join(map(str, cores))] + cmd
        if opts.use_systemd:
            cmd = ["systemd-run", "--user", "--scope", "--quiet", f"--description=itrea {run_dir.name} {spec.role}",
                   "-p", f"MemoryMax={int(round(mem * 1024))}M", "-p", "MemorySwapMax=0"] + cmd
        procs.append(Proc(spec=spec, argv=argv, cmd=cmd, cores=cores, mem_gb=mem, gpu_gb=g, env_extra=env,
                          log_path=run_dir / f"{spec.role}.log"))
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
                    solver_settings=None if arm.has_ea else solver_settings(job, opts, cfg)),
        problem=dict(prob_summary, load_s=round(load_s, 3)),
        mailbox=dict(kind=opts.mailbox, spec=spec),
        lane=dataclasses.asdict(opts.lane), systemd=opts.use_systemd, pinned=opts.pin,
        memory_budget=dict(dataclasses.asdict(budget), need_gb=round(budget.need_gb, 2)),
        processes=[], created_epoch=time.time())
    rec["env"] = record_env.record()
    own: set[int] = set()
    rec["gates_before"] = gates.snapshot() if opts.record_snapshots else None
    if rec["gates_before"] is not None:
        rec["gates_before"]["gpu_apps"] = gpu_health.compute_apps()

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
    status = None
    t0 = None
    cache_t0 = None
    stop_info: dict | None = None
    opts.cupy_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        # ---------------------------------------------------------------- launch
        for p in procs:
            env = dict(os.environ)
            env.pop("CUPY_GPU_MEMORY_LIMIT", None)
            env.update(p.env_extra)
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
            while True:
                alive = [p for p in procs if p.poll() is None]
                if not alive:
                    break
                now = time.monotonic()
                if opts.abort is not None and opts.abort.is_set():
                    stop_info = dict(reason="abort: " + (opts.abort_reason() if opts.abort_reason else ""),
                                     t=now - t0, kind="abort")
                    status = "void"
                    reasons.append(stop_info["reason"])
                    break
                if now >= t_end:
                    stop_info = dict(reason="budget + grace", t=now - t0, kind="budget")
                    break
                ended = [p for p in procs if p.exit_code is not None]
                decisive = [p for p in ended if p.exit_code != 0 or _exit_reason(run_dir, p.role) in DEFINITIVE]
                if decisive and opts.stop_partners_on_exit:
                    stop_info = dict(reason=", ".join(
                        f"{p.role} ended rc={p.exit_code}" + (f" ({_exit_reason(run_dir, p.role)})" if p.exit_code == 0
                                                              else "") for p in decisive), t=now - t0,
                        kind="partner_end")
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
        for p in procs:
            p.stopped_by_runner = bool(p.killed and (stop_info or {}).get("kind") == "partner_end")
        rec["processes"] = [_proc_record(p, t0) for p in procs]
        rec["exit_codes"] = {p.role: p.exit_code for p in procs}
        for p in procs:   # a crashed process: its crash-safe .part file becomes <role>.incumbents.npz, so the
            if p.exit_code != 0 or not (run_dir / f"{p.role}.incumbents.npz").exists():   # checker can still read it
                try:
                    from itrea.solvers.common import recover_incumbents
                    recover_incumbents(run_dir, p.role)
                except Exception as e:  # noqa: BLE001 — recovery is best effort; the reason is recorded
                    reasons.append(f"recover {p.role} incumbents: {e!r}")
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
        if status == "void" and "void_cause" not in rec:
            rec["void_cause"] = "abort" if any(r.startswith(("abort", "aborted")) for r in reasons) else "interrupted"
        rec["status"] = status
        rec["trajectories"] = merge_trajectories(run_dir, arm.roles, sense,
                                                 "solver" if any(p.role == "solver" for p in arm.procs) else None)
        rec["ended_epoch"] = time.time()
        _atomic_json(run_dir / "run.json", rec)
        opts.log(f"{run_id}: {status}" + (f" ({'; '.join(reasons)})" if reasons else ""))
    return rec


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
             gpu_limit_gb=p.gpu_gb, env={k: v for k, v in p.env_extra.items()}, pid=p.pid,
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
                if job.get("split") in pool.QUIET_SPLITS:
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
