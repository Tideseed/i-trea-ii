"""The EA process: `python -m itrea.ea.run --instance P --config Y [--set k=v ...] --seed S --budget T --run-dir D
[--mailbox M] [--run-id R] [--role-name ea] [--no-barrier] [--barrier-timeout S] [--sol-out F] [--arm A]
[--backend cupy|numpy] [--gpu-mem-gb G]` (vault `10-design/Process interface.md`).

Sequence: read the instance (timed `read_s`) -> the configuration (`--config`, then every `--set`, then `--backend` and
`--gpu-mem-gb` as overrides, so the recorded hash is the configuration that actually ran) -> open the mailbox
(`open_mailbox(spec, role, run_id, create=False)`; the runner created it) -> `Engine.prepare()` (backend, device copy,
operators, warm-up on a synthetic problem — everything before the clock) -> `mailbox.ready()`, t0 =
`mailbox.wait_start()` (or `time.monotonic()` with `--no-barrier`) -> `Engine.run(budget, t0)` -> files -> exit code.

Files in `--run-dir` (`<role>` = `--role-name`, default "ea"): `<role>.events.jsonl` and `<role>.summary.json`
(`itrea.trace`), `<role>.incumbents.npz` (every EA incumbent vector), `<role>.genealogy.npz` (edge list),
`<role>.frontier.npz` (the archive), `<role>.sol.json.gz` (the best incumbent as {name: value} of non-zeros — the
independent checker's input; also copied to `--sol-out`).

The summary records what the analysis needs to trust the run (critique S3/S4): the backend actually used, the device
name and the CuPy pool limit, the chosen population size P and offspring count, the gene dtype, the full configuration
and its hash, library versions, the start-up seconds per step before t0, and `compiled_on_clock` — true if the CuPy
kernel cache (`CUPY_CACHE_DIR`, else `~/.cupy/kernel_cache`) gained files after t0, i.e. the warm-up missed a kernel
and NVRTC ran on the clock (None on NumPy, where Numba's compilations are not counted).

Exit codes: 0 = normal end (budget, stop file, SIGTERM, a stop before the start), 1 = exception (traceback in the
summary; whatever the engine held — incumbents, genealogy, frontier — is still written), 2 = bad arguments. Every
process of a run sets OMP/OPENBLAS/MKL_NUM_THREADS = 1 (the runner does; this module sets the defaults too, before
NumPy is imported by the engine), and CUPY_ACCELERATORS="" (CUB off) unless the environment sets it — the effective
accelerators are recorded in the start-up record (`startup.cupy_accelerators`).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# CUB off (before CuPy is imported): CUB's reductions are compiled per array-size class, which the warm-up on a small
# synthetic problem cannot cover, so they would compile with NVRTC after t0 (review F2, 2026-10-06). CuPy's own
# reduction kernels depend only on dtypes and dimensions; its scan's size classes are warmed by Engine._warm_scans.
os.environ.setdefault("CUPY_ACCELERATORS", "")

import argparse  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

MODULE = "itrea.ea.run"
EXIT_OK, EXIT_EXCEPTION, EXIT_BAD_ARGS = 0, 1, 2


class BadArguments(ValueError):
    """An argument the process refuses (exit code 2)."""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"python -m {MODULE}", description="I-TREA II EA process")
    ap.add_argument("--instance", required=True, help="MPS/LP file (repository-relative or absolute)")
    ap.add_argument("--config", default=None, help="EAConfig YAML (default: the built-in defaults)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                    help="dotted config override, value parsed as YAML (e.g. ranks.x=0.0); repeatable")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=float, required=True, help="wall-clock seconds after the shared start t0")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mailbox", default=None, help="a directory (file mailbox) or redis://host:port/db?run=<id>")
    ap.add_argument("--run-id", default=None, help="must match the mailbox's run id (default: $ITREA_RUN_ID)")
    ap.add_argument("--role-name", default="ea", help="role in the mailbox and file prefix (ea)")
    ap.add_argument("--no-barrier", action="store_true", help="start the clock at once instead of waiting for t0")
    ap.add_argument("--barrier-timeout", type=float, default=600.0)
    ap.add_argument("--sol-out", default=None, help="extra copy of <role>.sol.json.gz at this path")
    ap.add_argument("--arm", default="", help="arm label, recorded in the summary")
    ap.add_argument("--backend", choices=["cupy", "numpy"], default=None, help="override cfg.backend")
    ap.add_argument("--gpu-mem-gb", type=float, default=None, help="override cfg.gpu_mem_gb (CuPy pool limit)")
    return ap


def parse_overrides(items: list[str]) -> dict:
    """["ranks.x=0.0", "exchange.enabled=false"] -> {"ranks.x": 0.0, "exchange.enabled": False} (values as YAML)."""
    import yaml

    out = {}
    for it in items:
        if "=" not in it:
            raise BadArguments(f"--set {it!r}: expected KEY=VALUE")
        k, v = it.split("=", 1)
        k = k.strip()
        if not k:
            raise BadArguments(f"--set {it!r}: empty key")
        try:
            out[k] = yaml.safe_load(v)
        except yaml.YAMLError as e:
            raise BadArguments(f"--set {it!r}: {e}") from e
    return out


def load_config(args):
    """The configuration that runs: file -> --set -> --backend/--gpu-mem-gb. Raises BadArguments."""
    from itrea.config import EAConfig
    from itrea.paths import resolve

    try:
        if args.config:
            p = resolve(args.config)
            if not p.is_file():
                raise BadArguments(f"--config {args.config}: no such file")
            cfg = EAConfig.from_yaml(p)
        else:
            cfg = EAConfig().validate()
        ov = parse_overrides(args.overrides)
        if args.backend is not None:
            ov["backend"] = args.backend
        if args.gpu_mem_gb is not None:
            ov["gpu_mem_gb"] = float(args.gpu_mem_gb)
        if ov:
            cfg = cfg.with_overrides(ov)
    except BadArguments:
        raise
    except (KeyError, ValueError, TypeError) as e:
        raise BadArguments(f"configuration: {e}") from e
    if cfg.backend == "auto":
        raise BadArguments("backend 'auto' is refused for an EA run (use --backend cupy|numpy)")
    return cfg


def validate(args) -> None:
    from itrea.paths import resolve

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
    if args.gpu_mem_gb is not None and not args.gpu_mem_gb > 0:
        raise BadArguments("--gpu-mem-gb must be positive")


def cupy_cache_dir() -> Path:
    return Path(os.environ.get("CUPY_CACHE_DIR") or (Path.home() / ".cupy" / "kernel_cache"))


def count_files(d: Path) -> int | None:
    try:
        return sum(len(files) for _, _, files in os.walk(d))
    except OSError:
        return None


def list_files(d: Path) -> set[str] | None:
    """Relative paths of the files under d (the kernel cache), to name what was compiled after t0."""
    try:
        return {os.path.relpath(os.path.join(r, f), d) for r, _, files in os.walk(d) for f in files}
    except OSError:
        return None


def versions(backend: str) -> dict:
    import numpy as np
    import scipy

    out = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    from importlib.metadata import PackageNotFoundError, version
    for mod in ("highspy", "numba", "redis", "pyscipopt"):
        try:
            out[mod] = version(mod)
        except PackageNotFoundError:
            out[mod] = None
    if backend == "cupy":
        import cupy as cp
        out["cupy"] = cp.__version__
        out["cuda_runtime"] = cp.cuda.runtime.runtimeGetVersion()
        out["cuda_driver"] = cp.cuda.runtime.driverGetVersion()
    return {k: (str(v) if v is not None and not isinstance(v, (int, float)) else v) for k, v in out.items()}


def process_age_s() -> float | None:
    """Seconds since this process started (interpreter start and imports before main() count as start-up)."""
    try:
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        start = int(fields[19]) / os.sysconf("SC_CLK_TCK")
        return round(time.clock_gettime(time.CLOCK_BOOTTIME) - start, 3)
    except (OSError, ValueError, IndexError):
        return None


def write_solution(x, names: list[str], dests: list[Path]) -> None:
    sol = {nm: float(v) for nm, v in zip(names, x) if v != 0.0}
    for d in dests:
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_name(d.name + ".tmp")
        with gzip.open(tmp, "wt") as fh:
            json.dump(sol, fh)
        os.replace(tmp, d)


def main(argv=None) -> int:
    t_main = time.monotonic()
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:                     # argparse: 2 on a bad argument, 0 on --help
        return int(e.code) if isinstance(e.code, int) else EXIT_BAD_ARGS
    from itrea.paths import relative, resolve
    from itrea.trace import Trace

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    role = args.role_name if args.role_name and not any(c in args.role_name for c in "/ :") else "ea"
    run_id = args.run_id or os.environ.get("ITREA_RUN_ID")
    trace = Trace(run_dir, role, run_id=run_id or run_dir.name, module=MODULE, instance=relative(resolve(args.instance)),
                  arm=args.arm, seed=args.seed, budget=args.budget, args=vars(args), pid=os.getpid(),
                  host=platform.node(), cpus=sorted(os.sched_getaffinity(0)),
                  threads_env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                              "MKL_NUM_THREADS")})
    engine = None
    prob = None
    timing: dict = {}
    try:
        validate(args)
        cfg = load_config(args)
        from itrea.problem import load
        t = time.monotonic()
        prob = load(args.instance)
        timing["read_s"] = time.monotonic() - t
        trace.set(config=cfg.to_dict(), config_hash=cfg.hash(), config_name=cfg.name, sense=prob.sense,
                  problem=prob.summary(), overrides=parse_overrides(args.overrides))
        mailbox = None
        if args.mailbox is not None:
            from itrea.exchange.redis_mailbox import open_mailbox
            mailbox = open_mailbox(args.mailbox, role, run_id=run_id, create=False)
            trace.set(mailbox=args.mailbox if args.mailbox.startswith("redis://") else relative(args.mailbox),
                      mailbox_kind="redis" if args.mailbox.startswith("redis://") else "file")
        from itrea.ea.engine import Engine
        engine = Engine(prob, cfg, args.seed, mailbox=mailbox, trace=trace)

        def _on_term(signum, frame):        # noqa: ARG001 — the local search polls it between launches / slices
            engine.request_stop("sigterm")
        signal.signal(signal.SIGTERM, _on_term)
        cache = cupy_cache_dir()
        files_before = count_files(cache) if cfg.backend == "cupy" else None
        startup = engine.prepare()
        trace.set(backend=startup["backend"], device=startup["device"], pool_limit_bytes=startup["pool_limit_bytes"],
                  P=startup["P"], offspring=startup["offspring"], gene_dtype=startup["gene_dtype"],
                  local_search=startup["local_search"], versions=versions(startup["backend"]))
        # ------------------------------------------------------------------------------------ barrier
        t_ready = time.monotonic()
        timing.update({k: v for k, v in startup.items() if k.endswith("_s")})
        timing["startup_s"] = t_ready - t_main
        timing["process_age_at_ready_s"] = process_age_s()
        if args.no_barrier:
            t0 = time.monotonic()
        else:
            mailbox.ready()
            try:
                t0 = mailbox.wait_start(timeout=args.barrier_timeout)
            except RuntimeError as e:
                if "stopped before start" in str(e):
                    trace.close(exit_reason="stopped_before_start", startup=startup, timing=timing,
                                startup_s=timing["startup_s"], note=str(e))
                    return EXIT_OK
                raise
        timing["barrier_wait_s"] = max(0.0, time.monotonic() - t_ready)
        trace.start(t0)
        trace.set(t0_epoch=time.time() - (time.monotonic() - t0))
        names_t0 = list_files(cache) if cfg.backend == "cupy" else None     # one walk of the cache on the clock
        files_t0 = len(names_t0) if names_t0 is not None else None
        # ---------------------------------------------------------------------------------------- run
        engine.incumbent_log(run_dir, role)              # crash-safe .part file, removed once the .npz exists
        summary = engine.run(args.budget, t0)
        names_end = list_files(cache) if cfg.backend == "cupy" else None
        files_end = len(names_end) if names_end is not None else None
        new_after_t0 = (sorted(names_end - names_t0)[:50] if names_t0 is not None and names_end is not None
                        else None)
        best_x = summary.pop("best_x", None)
        summary.pop("frontier", None)
        outputs = engine.write_outputs(run_dir, role)
        sol_file = None
        if best_x is not None:
            dests = [run_dir / f"{role}.sol.json.gz"] + ([Path(args.sol_out)] if args.sol_out else [])
            write_solution(best_x, prob.names, dests)
            sol_file = relative(dests[-1])
        ru = resource.getrusage(resource.RUSAGE_SELF)
        trace.close(startup=startup, timing=timing, startup_s=timing["startup_s"], read_s=timing.get("read_s"),
                    cupy_cache=dict(dir=str(cache), files_before_prepare=files_before, files_at_t0=files_t0,
                                    files_at_end=files_end, new_after_t0=new_after_t0),
                    compiled_on_clock=(None if files_t0 is None or files_end is None else files_end > files_t0),
                    outputs={k: relative(v) for k, v in outputs.items()}, sol_file=sol_file,
                    maxrss_mb=round(ru.ru_maxrss / 1024, 1), cpu_user_s=round(ru.ru_utime, 3),
                    cpu_sys_s=round(ru.ru_stime, 3), **summary)
        return EXIT_OK
    except BadArguments as e:
        print(f"{MODULE}: bad arguments: {e}", file=sys.stderr)
        trace.close(exit_reason="bad_arguments", error=str(e), timing=timing)
        return EXIT_BAD_ARGS
    except Exception as e:  # noqa: BLE001 — every exception ends as exit 1 with its traceback recorded
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        extra: dict = {}
        if engine is not None:
            try:
                extra["outputs"] = {k: relative(v) for k, v in engine.write_outputs(run_dir, role).items()}
                bx = getattr(engine, "best_x", None)
                if bx is not None and prob is not None:
                    write_solution(bx, prob.names, [run_dir / f"{role}.sol.json.gz"] +
                                   ([Path(args.sol_out)] if args.sol_out else []))
            except Exception as e2:  # noqa: BLE001
                extra["store_error"] = f"{type(e2).__name__}: {e2}"[:300]
            extra["generations"] = getattr(engine, "gen", None)
            extra["counters_engine"] = getattr(engine, "c", None)
        from itrea.ea.engine import DoesNotFit
        reason = ("barrier_timeout" if isinstance(e, TimeoutError) else
                  "does_not_fit" if isinstance(e, DoesNotFit) else "exception")
        try:
            trace.close(exit_reason=reason, error=f"{type(e).__name__}: {e}"[:1000], traceback=tb[-8000:],
                        timing=timing, **extra)
        except Exception:  # noqa: BLE001
            print(traceback.format_exc(), file=sys.stderr)
        return EXIT_EXCEPTION


if __name__ == "__main__":
    raise SystemExit(main())
