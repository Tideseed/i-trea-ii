"""What the three solver processes (`scip_run`, `highs_run`, `cuopt_run`) share: the command line, the start-up and
barrier sequence, the incumbent recorder, exchange accounting, and the exit-code contract.

Why one shared module: the three processes must produce the SAME files with the same meaning, or the merged
trajectories and the post accounting (vault `10-design/Trace format and metrics.md`) compare different things. The
per-solver modules only add what is specific to the solver's API.

Sequence of every solver process (vault `10-design/Process interface.md`):
    parse arguments (exit 2 on a bad one) -> open the mailbox (`open_mailbox(spec, role, run_id=..., create=False)`)
    -> read the instance (timed: `read_s`) -> map variables onto the canonical order by name (only when exchanging)
    -> solver set-up -> `mailbox.ready()`; t0 = `mailbox.wait_start()` (or `time.monotonic()` with `--no-barrier`)
    -> time limit = budget - (time.monotonic() - t0), set just before optimise -> solve -> files -> exit code.

Files (in `--run-dir`, `<role>` = `--role-name`, default "solver" / "cuopt"):
    <role>.events.jsonl      `itrea.trace.Trace` events, seconds since t0 (monotonic clock, system-wide)
    <role>.summary.json      identity, start-up record, exit reason, counters, exchange seconds and bytes
    <role>.incumbents.npz    EVERY incumbent vector (D-002: P(T) is built only from checked incumbents)
    <role>.sol.json.gz       the final incumbent as {name: value} of non-zeros (the independent checker's input)
    <role>.incumbents.part   crash-safe append-only copy of the incumbent vectors while the solve runs; removed once
                             the .npz exists. A run that died keeps it (with <role>.names.json) and
                             `recover_incumbents` turns it into the .npz — a crashed run is kept and truncated at the
                             crash (D-002 §6), which needs its incumbents to be checkable.

Exit codes: 0 = normal end (budget, optimality, stop file, SIGTERM), 1 = exception (traceback in the summary),
2 = bad arguments.

Origin bits per row of a solver -> EA message (`to_ea`): ORIGIN_SOLVER (the solver found it) or ORIGIN_EA_ECHO (a
vector the solver received from the EA earlier, coming back as part of its pool). Echo marking keeps EA-born solutions
from being credited to the solver in the lineage analysis (critique, systems lens, "Provenance of echoed solutions").
EA -> solver rows (`to_solver`) carry the EA's uint16 origin bits; in arm A9 (D-015) a row that is an UNCHANGED cuOpt
vector also carries POST_CUOPT (bit 16, above every EA bit: the message's origin is int64), so SCIP records the
incumbent it makes as source "cuopt", not "ea".

Adapted in part from berkorbay/optopt@4dd48395, MIT (`src/optopt/solvers/common.py`: the gzip JSON solution format
and its "never fail the run" rule).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import resource
import signal
import struct
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from itrea.paths import relative, resolve
from itrea.trace import Trace, save_incumbents

ORIGIN_SOLVER = 1          # row of a to_ea message that the solver found itself
ORIGIN_EA_ECHO = 2         # row the solver had received from the EA (its own post coming back)
POST_CUOPT = 1 << 16       # EA -> solver row that is an unchanged cuOpt vector (A9): the solver credits it to cuOpt

EXIT_OK, EXIT_EXCEPTION, EXIT_BAD_ARGS = 0, 1, 2

STOP_CHECK_S = 1.0         # stop-file / SIGTERM checks: control plane, at most once per second in EVERY mode
                           # (also --inert), so the runner can end any process; one stat or one Redis EXISTS each
SOURCE_CODES = {"self": 0, "ea": 1, "solver": 2, "cuopt": 3}


class BadArguments(ValueError):
    """An argument combination the process refuses (exit code 2)."""


class StoppedBeforeStart(RuntimeError):
    """The runner wrote `stop` before releasing the barrier (a partner died during start-up): a normal end, exit 0."""


# ---------------------------------------------------------------------------------------------------- arguments
def base_parser(prog: str, description: str, default_role: str, exchange: bool = True) -> argparse.ArgumentParser:
    """The arguments every solver process takes (Process interface), plus --mailbox/--run-id/--role-name.
    exchange=True adds the SCIP/HiGHS exchange flags (--recv, --send, --inject-band, --inject-k, --poll-s, --inert,
    --no-plugin, --mem-mb)."""
    ap = argparse.ArgumentParser(prog=prog, description=description)
    ap.add_argument("--instance", required=True, help="MPS/LP file (repository-relative or absolute)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True, help="wall-clock seconds after the shared start t0")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mailbox", default=None, help="a directory (file mailbox) or redis://host:port/db?run=<id>")
    ap.add_argument("--run-id", default=None, help="must match the mailbox's run id (default: none checked)")
    ap.add_argument("--role-name", default=default_role, help=f"role in the mailbox and file prefix ({default_role})")
    ap.add_argument("--no-barrier", action="store_true", help="start the clock at once instead of waiting for t0")
    ap.add_argument("--barrier-timeout", type=float, default=600.0, help="seconds to wait for the runner's start")
    ap.add_argument("--sol-out", default=None, help="extra copy of <role>.sol.json.gz at this path")
    ap.add_argument("--arm", default="", help="arm label, recorded in the summary")
    if exchange:
        ap.add_argument("--mem-mb", type=float, default=4000.0, help="solver memory limit in MB")
        ap.add_argument("--recv", action="store_true", help="accept the EA's solutions")
        ap.add_argument("--send", action="store_true", help="publish incumbents, pool, bounds (and the root LP)")
        ap.add_argument("--inject-band", type=float, default=0.01,
                        help="'very good' = within this relative gap of the best (max(1, |best|))")
        ap.add_argument("--inject-k", type=int, default=8, help="at most this many solutions per post")
        ap.add_argument("--poll-s", type=float, default=0.25, help="exchange at most this often (seconds)")
        ap.add_argument("--inert", action="store_true",
                        help="exchange callback installed but never polls or posts (harness-matched A0)")
        ap.add_argument("--no-plugin", action="store_true",
                        help="calibration only: no exchange callback at all (E0 'no exchange' reference)")
    return ap


def validate_common(args, exchange: bool = True) -> None:
    """Raise BadArguments for combinations no arm uses (they would silently measure something else)."""
    if not math.isfinite(args.budget) or args.budget <= 0:
        raise BadArguments("--budget must be a positive number of seconds")
    if not resolve(args.instance).is_file():
        raise BadArguments(f"--instance {args.instance}: no such file")
    if args.mailbox is None and not args.no_barrier:
        raise BadArguments("the start barrier needs --mailbox (or pass --no-barrier)")
    if args.barrier_timeout <= 0:
        raise BadArguments("--barrier-timeout must be positive")
    if not args.role_name or any(c in args.role_name for c in "/ :"):
        raise BadArguments("--role-name must be a plain word")
    if exchange:
        if (args.recv or args.send) and args.mailbox is None:
            raise BadArguments("--recv/--send need --mailbox")
        if args.inert and (args.recv or args.send):
            raise BadArguments("--inert excludes --recv/--send")
        if args.no_plugin and (args.recv or args.send or args.inert):
            raise BadArguments("--no-plugin excludes --recv/--send/--inert")
        if args.inject_k < 1 or args.inject_band < 0 or args.poll_s <= 0 or args.mem_mb <= 0:
            raise BadArguments("--inject-k >= 1, --inject-band >= 0, --poll-s > 0, --mem-mb > 0")


# ------------------------------------------------------------------------------------------------- small helpers
def finite_or_none(v, big: float = 1e19):
    """A solver value as a float, or None for 'no value' (infinite, NaN, SCIP's 1e20 infinity)."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and abs(v) < big else None


def vector_hash(x) -> bytes:
    """Identity of a solution vector for de-duplication and echo marking: values rounded to 1e-6 (an integer solution
    read back from a solver can differ in the last bits of its continuous part), negative zero folded into zero."""
    a = np.round(np.asarray(x, dtype=np.float64).ravel(), 6) + 0.0
    return hashlib.blake2b(a.tobytes(), digest_size=16).digest()


def band_select(objs, sense: int, band: float, k: int) -> list[int]:
    """Indices of the 'very good' solutions: key = sense * obj within band * max(1, |best key|) of the best, best
    first (stable on ties), at most k."""
    objs = np.asarray(objs, dtype=np.float64)
    if objs.size == 0:
        return []
    keys = sense * objs
    best = float(np.min(keys))
    lim = best + band * max(1.0, abs(best)) + 1e-12 * max(1.0, abs(best))
    order = np.argsort(keys, kind="stable")
    return [int(i) for i in order if keys[i] <= lim][:k]


def improves(new_obj: float, old_obj: float | None, sense: int, rel: float = 1e-9) -> bool:
    """new beats old by more than rel * max(1, |old|) in the objective sense (None = no incumbent yet)."""
    if old_obj is None:
        return True
    return sense * (new_obj - old_obj) < -rel * max(1.0, abs(old_obj))


def versions(**extra) -> dict:
    out = {"python": platform.python_version(), "numpy": np.__version__}
    out.update({k: v for k, v in extra.items() if v is not None})
    return out


def process_age_s() -> float | None:
    """Seconds since this process started (from /proc/self/stat), so start-up includes interpreter start and imports
    that happen before main()."""
    try:
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        start = int(fields[19]) / os.sysconf("SC_CLK_TCK")
        return round(time.clock_gettime(time.CLOCK_BOOTTIME) - start, 3)
    except (OSError, ValueError, IndexError):
        return None


def mem_available_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 1048576, 3)
    except OSError:
        pass
    return None


class _SparseVec:
    """A solution kept as its non-zeros while the solve runs (a 364k-variable dense vector per incumbent would not
    fit in memory for long runs); densified only when the .npz is written."""
    __slots__ = ("n", "idx", "val")

    def __init__(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n = x.size
        self.idx = np.flatnonzero(x).astype(np.int32)
        self.val = x[self.idx].copy()

    def dense(self) -> np.ndarray:
        out = np.zeros(self.n)
        out[self.idx] = self.val
        return out

    def __array__(self, dtype=None, copy=None):
        d = self.dense()
        return d if dtype is None else d.astype(dtype)


class ExchangeStats:
    """Solver-side exchange cost (critique: 'record per run the solver-side exchange seconds and the bytes moved').
    Seconds = perf_counter time inside exchange code on the solver's core; bytes = logical payload (float64 vectors and
    objectives) — the Redis transport may move fewer (sparse encoding)."""

    def __init__(self):
        self.seconds = 0.0
        self.bytes_out = 0
        self.bytes_in = 0
        self.c: dict[str, int] = {}

    def add(self, key: str, n: int = 1) -> None:
        self.c[key] = self.c.get(key, 0) + n

    def as_dict(self) -> dict:
        return dict(seconds=round(self.seconds, 6), bytes_out=self.bytes_out, bytes_in=self.bytes_in, **self.c)


# -------------------------------------------------------------------------------------------- the process context
class SolverProcess:
    """Per-process state shared by the solver adapters: trace, mailbox, name map, incumbent records, exchange
    accounting, timing, the stop/SIGTERM flags and the final files."""

    def __init__(self, args, solver: str, module: str, t_main: float):
        self.args = args
        self.solver = solver
        self.role = args.role_name
        self.t_main = t_main
        self.run_dir = Path(args.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace = Trace(self.run_dir, self.role, run_id=args.run_id or self.run_dir.name, solver=solver,
                           module=module, instance=relative(resolve(args.instance)), arm=args.arm, seed=args.seed,
                           budget=args.budget, args={k: v for k, v in vars(args).items()}, pid=os.getpid(),
                           host=platform.node(), cpus=sorted(os.sched_getaffinity(0)),
                           threads_env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                                       "MKL_NUM_THREADS")})
        self.trace.set(mem_available_gb_start=mem_available_gb())
        self.mailbox = None
        self.mailbox_kind = None
        self.local_names: list[str] | None = None
        self.namemap = None
        self.sense = 1
        self.t0: float | None = None
        self.records: list[tuple[float, float, str, _SparseVec]] = []
        self.best_obj: float | None = None
        self.stats = ExchangeStats()
        self.sigterm = False
        self.stop_seen = False
        self._next_stop_check = 0.0
        self._part = None
        self.timing: dict[str, float] = {}

    # ---------------------------------------------------------------- set-up
    def open_mailbox(self) -> None:
        if self.args.mailbox is None:
            return
        from itrea.exchange.redis_mailbox import open_mailbox
        spec = self.args.mailbox
        self.mailbox = open_mailbox(spec, self.role, run_id=self.args.run_id, create=False)
        self.mailbox_kind = "redis" if spec.startswith("redis://") else "file"
        self.trace.set(mailbox=spec if self.mailbox_kind == "redis" else relative(spec), mailbox_kind=self.mailbox_kind)

    def set_names(self, names: list[str], sense: int) -> None:
        """The process's own variable order (incumbent vectors are stored in it) and the objective sense."""
        self.local_names = [str(n) for n in names]
        self.sense = int(sense)
        p = self.run_dir / f"{self.role}.names.json"           # for recover_incumbents after a crash
        p.write_text(json.dumps(self.local_names))
        self._part = open(self.run_dir / f"{self.role}.incumbents.part", "ab", buffering=0)

    def map_order(self) -> None:
        """Map the local order onto the runner's canonical order (by name). Only exchanging processes need it."""
        from itrea.exchange.mailbox import NameMap
        try:
            canonical = self.mailbox.read_order()
        except (OSError, TypeError, ValueError) as e:
            raise BadArguments(f"mailbox has no canonical variable order (the runner writes it): {e}") from e
        self.namemap = NameMap(canonical, self.local_names)

    def install_sigterm(self) -> None:
        def _on_term(signum, frame):            # noqa: ARG001 — the solver is told at its next callback
            self.sigterm = True
        signal.signal(signal.SIGTERM, _on_term)

    def barrier(self) -> float:
        """Report ready and wait for t0 (monotonic). Records start-up seconds before the clock."""
        t_ready = time.monotonic()
        self.timing["startup_s"] = t_ready - self.t_main
        self.timing["process_age_at_ready_s"] = process_age_s()
        if self.args.no_barrier:
            t0 = time.monotonic()
        else:
            self.mailbox.ready()
            try:
                t0 = self.mailbox.wait_start(timeout=self.args.barrier_timeout)
            except RuntimeError as e:
                if "stopped before start" in str(e):
                    raise StoppedBeforeStart(str(e)) from e
                raise
        self.timing["barrier_wait_s"] = max(0.0, time.monotonic() - t_ready)
        self.t0 = t0
        self.trace.start(t0)
        self.trace.set(t0_epoch=time.time() - (time.monotonic() - t0))
        return t0

    def remaining(self) -> float:
        return self.args.budget - (time.monotonic() - self.t0)

    # --------------------------------------------------------------- control
    def should_stop(self, now: float | None = None) -> bool:
        """SIGTERM or the runner's stop file. The file is checked at most once per STOP_CHECK_S."""
        if self.sigterm or self.stop_seen:
            return True
        now = time.monotonic() if now is None else now
        if self.mailbox is not None and now >= self._next_stop_check:
            self._next_stop_check = now + STOP_CHECK_S
            if self.mailbox.stopped():
                self.stop_seen = True
                self.trace.event("note", msg="stop file seen")
        return self.stop_seen

    # ------------------------------------------------------------ incumbents
    def record_incumbent(self, obj: float, x_local, source: str = "self", t: float | None = None, **extra) -> None:
        """Every incumbent: trace event, in-memory sparse record and the crash-safe .part record."""
        t = self.trace.now() if t is None else t
        sv = _SparseVec(x_local)
        self.records.append((t, float(obj), source, sv))
        if self.best_obj is None or improves(obj, self.best_obj, self.sense, rel=0.0):
            self.best_obj = float(obj)
        self.trace.incumbent(obj, source=source, t=t, **extra)
        if self._part is not None:
            hdr = struct.pack("<ddBi", t, float(obj), SOURCE_CODES.get(source, 255), sv.idx.size)
            self._part.write(hdr + sv.idx.astype("<i4").tobytes() + sv.val.astype("<f8").tobytes())

    def final_x(self) -> np.ndarray | None:
        """The best recorded incumbent (the solver's final one is recorded before finish())."""
        if not self.records:
            return None
        keys = [self.sense * r[1] for r in self.records]
        return self.records[int(np.argmin(keys))][3].dense()

    # ---------------------------------------------------------------- finish
    def finish(self, exit_reason: str, **summary) -> None:
        """Write the incumbent vectors, the final solution and the summary. Never raises for the side files: a failure
        to store them is recorded in the summary instead (the summary must exist for every run)."""
        names = self.local_names or []
        try:
            if self.local_names is not None:
                save_incumbents(self.run_dir / f"{self.role}.incumbents.npz", names,
                                [(t, o, s, v) for t, o, s, v in self.records])
                if self._part is not None:
                    self._part.close()
                    self._part = None
                for side in (f"{self.role}.incumbents.part", f"{self.role}.names.json"):
                    (self.run_dir / side).unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001 — keep the .part file; record why
            summary.setdefault("store_errors", []).append(f"incumbents: {type(e).__name__}: {e}"[:300])
        x = self.final_x()
        if x is not None and names:
            try:
                sol = {nm: float(v) for nm, v in zip(names, x) if v != 0.0}
                dests = [self.run_dir / f"{self.role}.sol.json.gz"]
                if self.args.sol_out:
                    dests.append(Path(self.args.sol_out))
                for d in dests:
                    d.parent.mkdir(parents=True, exist_ok=True)
                    tmp = d.with_name(d.name + ".tmp")
                    with gzip.open(tmp, "wt") as fh:
                        json.dump(sol, fh)
                    os.replace(tmp, d)
                summary["sol_file"] = relative(dests[-1])
            except Exception as e:  # noqa: BLE001
                summary.setdefault("store_errors", []).append(f"sol: {type(e).__name__}: {e}"[:300])
        ru = resource.getrusage(resource.RUSAGE_SELF)
        self.trace.close(exit_reason=exit_reason, n_incumbents=len(self.records),
                         final_obj=self.best_obj, exchange=self.stats.as_dict(), timing=self.timing,
                         maxrss_mb=round(ru.ru_maxrss / 1024, 1), cpu_user_s=round(ru.ru_utime, 3),
                         cpu_sys_s=round(ru.ru_stime, 3), mem_available_gb_end=mem_available_gb(), **summary)


def recover_incumbents(run_dir: str | os.PathLike, role: str) -> Path | None:
    """Turn a crashed process's <role>.incumbents.part (+ <role>.names.json) into <role>.incumbents.npz. A truncated
    last record is dropped. Returns the .npz path, or None if there is nothing to recover."""
    d = Path(run_dir)
    part, names_p = d / f"{role}.incumbents.part", d / f"{role}.names.json"
    if not part.exists() or not names_p.exists():
        return None
    names = json.loads(names_p.read_text())
    inv = {v: k for k, v in SOURCE_CODES.items()}
    buf = part.read_bytes()
    hsz = struct.calcsize("<ddBi")
    recs, off = [], 0
    while off + hsz <= len(buf):
        t, obj, src, nnz = struct.unpack_from("<ddBi", buf, off)
        end = off + hsz + 12 * nnz
        if nnz < 0 or end > len(buf):
            break
        idx = np.frombuffer(buf, dtype="<i4", count=nnz, offset=off + hsz)
        val = np.frombuffer(buf, dtype="<f8", count=nnz, offset=off + hsz + 4 * nnz)
        x = np.zeros(len(names))
        x[idx] = val
        recs.append((t, obj, inv.get(src, "unknown"), x))
        off = end
    return save_incumbents(d / f"{role}.incumbents.npz", names, recs)


# ------------------------------------------------------------------------------------------------- main wrapper
def run_main(argv, parser: argparse.ArgumentParser, validate, body, solver: str, module: str) -> int:
    """Parse, validate and run `body(proc) -> (exit_reason, summary dict)` with the exit-code contract:
    0 normal end, 1 exception (traceback in the summary), 2 bad arguments. The summary is written in every case where
    the run directory is known."""
    t_main = time.monotonic()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:                     # argparse: 2 on a bad argument, 0 on --help
        return int(e.code) if isinstance(e.code, int) else EXIT_BAD_ARGS
    proc = None
    try:
        proc = SolverProcess(args, solver, module, t_main)
        validate(args)
        proc.install_sigterm()
        proc.open_mailbox()
        exit_reason, summary = body(proc)
        proc.finish(exit_reason, **summary)
        return EXIT_OK
    except StoppedBeforeStart as e:
        proc.finish("stopped_before_start", note=str(e))
        return EXIT_OK
    except BadArguments as e:
        print(f"{module}: bad arguments: {e}", file=sys.stderr)
        if proc is not None:
            proc.finish("bad_arguments", error=str(e))
        return EXIT_BAD_ARGS
    except Exception as e:  # noqa: BLE001 — every exception ends as exit 1 with its traceback recorded
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        if proc is not None:
            reason = "barrier_timeout" if isinstance(e, TimeoutError) else "exception"
            try:
                proc.finish(reason, error=f"{type(e).__name__}: {e}"[:1000], traceback=tb[-8000:],
                            **getattr(e, "summary", {}))
            except Exception:  # noqa: BLE001
                print(traceback.format_exc(), file=sys.stderr)
        return EXIT_EXCEPTION
