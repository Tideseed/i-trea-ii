#!/usr/bin/env python
"""R1's mechanism check before any timed run (GPU: only inside a window with the lease held): does the EA host stop
burning its core while it waits on the GPU under `host.cuda_sched`, and does a CPU-bound process on the same core get
that core back? (vault `30-results/2026-10-14 C1 diagnosis and improvement candidates.md`, R1, "Before the window" 2;
D-025 "Before the first timed run", check 2, whose thresholds the GPU test applies).

A synthetic launch loop that waits the way the EA does — one kernel of `--launch-ms` per launch (a single thread that
spins on `clock64()`), then the feasibility jump's sync `int(cp.count_nonzero(...))`, the engine's per-generation
timing events (`engine._Phases`, created as the engine creates them under the mode: blocking under blocking_sync) read
every `--per-gen` launches, and `Backend.synchronize` — in two phases on ONE core (`--core`, default the first of
this process's affinity):
  A. alone: the host's CPU seconds per wall second (RUSAGE_SELF and the main thread's RUSAGE_THREAD) — about 1 when
     the host spins (mode auto), well below 1 under blocking sync;
  B. beside a competitor: a spawned process pinned to the same core runs a pure-Python loop (the LP worker's
     stand-in, at the host's nice) for the same seconds and reports its own CPU per wall second — about 0.5 when the
     host spins (the core is shared), close to 1 under blocking sync. With `--renice N` (D-025's fallback (a), `bnice`)
     every thread of the host takes nice +N relative to the stand-in right after the stand-in's spawn, exactly as the
     engine does after the LP worker's spawn (`engine.renice_threads`), so phase B measures the stand-in's share
     beside a spinning host at the lower priority. The renice is one-way: run it as its own process.
The flag goes through `itrea.ea.cuda_sched` exactly as `Engine.prepare` sets it (before CuPy creates the context),
and is verified with cuDevicePrimaryCtxGetState once the context exists. One mode per process (the flag must be set
before the context exists): run the tool once per mode.

    python tools/bsync_check.py --mode auto --seconds 12 --out /tmp/bsync/auto.json
    python tools/bsync_check.py --mode blocking_sync --seconds 12 --out /tmp/bsync/bsync.json
    python tools/bsync_check.py --mode yield --seconds 12 --out /tmp/bsync/yield.json
    python tools/bsync_check.py --mode auto --renice 10 --seconds 12 --out /tmp/bsync/bnice.json

Prints one JSON object (and writes it to `--out`): the mode, the driver record (under auto the flags read back), the
kernel's measured milliseconds, launches and GPU seconds per phase, `host_self_per_wall` / `host_thread_per_wall` (phase
A), `competitor_share` and `host_self_per_wall_b` (phase B), `renice` (the renice record, empty without `--renice`).
`tests/test_cuda_sched.py::test_bsync_check_tool_*` (marker gpu) runs it for auto, blocking_sync, yield and auto with
`--renice 10`, at 12 s per phase (>= 10 s of GPU work, D-025), and applies D-025's reading: under blocking_sync phase
A's host self-CPU per wall second <= 0.25 and phase B's competitor share > 0.8; under yield and under the renice the
competitor share >= 0.8; under auto > 0.8 and < 0.7, printed beside (the spin is measured, so the test cannot pass on a
host that never waited).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from itrea.ea import cuda_sched  # noqa: E402

SPIN_SRC = r"""
extern "C" __global__ void itrea_spin(long long cycles, int* out) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
    out[0] = 1;
}
"""


def _cpu(who: int) -> float:
    ru = resource.getrusage(who)
    return float(ru.ru_utime + ru.ru_stime)


def _competitor(core: int, seconds: float, conn) -> None:
    """The LP worker's stand-in: pinned to `core`, a pure-Python loop for `seconds` of wall time; sends its CPU
    seconds and wall seconds."""
    os.sched_setaffinity(0, {core})
    t0, c0 = time.monotonic(), _cpu(resource.RUSAGE_SELF)
    x = 0
    while time.monotonic() - t0 < seconds:
        for _ in range(10_000):
            x += 1
    conn.send(dict(cpu_s=_cpu(resource.RUSAGE_SELF) - c0, wall_s=time.monotonic() - t0, pid=os.getpid()))
    conn.close()


def _loop(cp, kern, cycles, out, phases, bk, seconds: float, per_gen: int) -> dict:
    """Launch / sync like the EA until `seconds` of wall time passed; returns launches and the host's CPU and wall."""
    s0, t0 = _cpu(resource.RUSAGE_SELF), _cpu(resource.RUSAGE_THREAD)
    w0 = time.monotonic()
    launches = 0
    gpu_s = 0.0
    while time.monotonic() - w0 < seconds:
        with phases("spin"):
            out[0] = 0
            kern((1,), (1,), (cycles, out))
            int(cp.count_nonzero(out))                   # the feasibility jump's sync after each launch
        launches += 1
        if launches % per_gen == 0:
            gpu_s += sum(phases.take().values())         # the engine's event wait on the last event
            bk.synchronize()                             # backend.synchronize = Stream.null.synchronize
    gpu_s += sum(phases.take().values())
    wall = time.monotonic() - w0
    return dict(launches=launches, wall_s=round(wall, 4), host_self_s=round(_cpu(resource.RUSAGE_SELF) - s0, 4),
                host_thread_s=round(_cpu(resource.RUSAGE_THREAD) - t0, 4), gpu_s=round(gpu_s, 4))


def run(mode: str, seconds: float, launch_ms: float, core: int, per_gen: int, competitor: bool = True,
        renice: int = 0) -> dict:
    os.sched_setaffinity(0, {core})
    rec = cuda_sched.apply(mode)                         # before CuPy is asked for anything
    from itrea.backend import get_backend
    from itrea.ea.engine import _Phases
    bk = get_backend("cupy", gpu_mem_gb=1.0)
    cp = bk.xp
    cp.zeros(1)
    rec = cuda_sched.verify(rec)
    khz = int(cp.cuda.Device().attributes["ClockRate"])
    cycles = cp.int64(int(khz * launch_ms))
    kern = cp.RawKernel(SPIN_SRC, "itrea_spin")
    out = cp.zeros(1, dtype=cp.int32)
    kern((1,), (1,), (cp.int64(1000), out))              # compile before anything is measured
    cp.cuda.Stream.null.synchronize()
    phases = _Phases(bk, block=cuda_sched.events_block(mode))
    with phases("probe"):
        kern((1,), (1,), (cycles, out))
    kernel_ms = round(1000.0 * phases.take()["probe"], 3)
    a = _loop(cp, kern, cycles, out, phases, bk, seconds, per_gen)
    res = dict(mode=mode, core=core, seconds=seconds, launch_ms=launch_ms, per_gen=per_gen, clock_khz=khz, renice={},
               kernel_ms=kernel_ms, phase_events_block=phases.block, cuda_sched=cuda_sched.public(rec), alone=a,
               host_self_per_wall=round(a["host_self_s"] / a["wall_s"], 4),
               host_thread_per_wall=round(a["host_thread_s"] / a["wall_s"], 4))
    if competitor:
        ctx = mp.get_context("spawn")
        recv, send = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_competitor, args=(core, seconds, send), daemon=True)
        proc.start()
        send.close()
        if renice:                                       # D-025 (a): the host below the stand-in it just spawned
            from itrea.ea.engine import renice_threads
            res["renice"] = renice_threads(renice)
        time.sleep(0.5)                                  # the competitor's interpreter start-up, before the loop
        b = _loop(cp, kern, cycles, out, phases, bk, seconds, per_gen)
        comp = recv.recv()
        proc.join(timeout=10)
        res.update(beside=b, competitor=comp, competitor_share=round(comp["cpu_s"] / max(comp["wall_s"], 1e-9), 4),
                   host_self_per_wall_b=round(b["host_self_s"] / b["wall_s"], 4))
    res["cuda_sched"].update(cuda_sched.state_at_end(rec))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", required=True, choices=tuple(cuda_sched.MODES))
    ap.add_argument("--seconds", type=float, default=12.0, help="wall seconds per phase (D-025: >= 10 s of GPU work)")
    ap.add_argument("--launch-ms", type=float, default=50.0, help="kernel milliseconds per launch (the FJ's ~50 ms)")
    ap.add_argument("--per-gen", type=int, default=8, help="launches between two reads of the timing events")
    ap.add_argument("--core", type=int, default=None, help="default: the first core of this process's affinity")
    ap.add_argument("--no-competitor", action="store_true", help="phase A only")
    ap.add_argument("--renice", type=int, default=0,
                    help="D-025 (a): after the competitor's spawn every host thread takes nice +N relative to it")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    core = a.core if a.core is not None else min(os.sched_getaffinity(0))
    if a.renice and a.no_competitor:
        ap.error("--renice acts after the competitor's spawn: it needs phase B")
    res = run(a.mode, a.seconds, a.launch_ms, core, a.per_gen, competitor=not a.no_competitor, renice=a.renice)
    txt = json.dumps(res, indent=1, sort_keys=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
