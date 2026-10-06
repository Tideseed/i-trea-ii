#!/usr/bin/env python
"""Local-search throughput benchmark on SYNTHETIC problems (never on real instances).

Why: D-002 E1 gate (b) asks for the GPU local search's moves/s and time to first feasible against a single-core CPU
feasibility-jump reference, and the design critique asks which repair variant gives the most feasible solutions per
wall second. This tool measures every available implementation on the same seeded start set X0:

    fj_cpu      itrea.ea.ls_fj_cpu.FJCpu   (Numba, one core)
    fj_gpu      itrea.ea.ls_fj_gpu.FJGpu   (CuPy RawKernel; "unavailable" if it does not import)
    dense_cpu   itrea.ea.ls_dense.DenseLS on NumPy (baseline)
    dense_gpu   itrea.ea.ls_dense.DenseLS on CuPy  (baseline)

Problems are generated in memory (seeded): `setcover` (binary covering, n columns, each in `deg` rows), `mknap`
(general integers in [0, 5], half <= capacity rows and half >= demand rows), `uc` (a unit-commitment-shaped mixed
model: binary on/off, continuous output linked to it, a balance equality per period with a costly singleton shedding
slack), and the toy models of `itrea.benchmarks.toy`.

Per (problem, implementation, P) it reports: wall seconds of one call (GPU timed with cp.cuda.Event, after a warm-up
call that compiles the kernels), moves and moves/s, total violation V before/after (`evaluate`, the population's V)
and its reduction per second, the share of individuals feasible after the call, the time to first feasible
("ttff_ladder_s": the wall time of the first call in a geometric `max_steps` ladder that returns at least one feasible
individual — a single batch call has no implementation-neutral "first feasible" moment, so this is defined
operationally), and peak memory (GPU: a CuPy MemoryHook on the pool's used bytes above the level before the call;
CPU: tracemalloc's peak of NumPy allocations above the level before the call — Numba's own scratch, O(n + m), is not
seen by tracemalloc and is reported separately as FJCpu.state_bytes).

    python tools/bench_ls.py --size small --P 16,64 --out traces/bench/ls_small.json
Writes JSON (config, environment, rows) and prints a short table. OMP_NUM_THREADS is pinned to 1.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import tracemalloc  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sps  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from itrea.backend import get_backend  # noqa: E402
from itrea.benchmarks import toy  # noqa: E402
from itrea.config import RepairCfg  # noqa: E402
from itrea.ea.evaluate import evaluate, to_device  # noqa: E402
from itrea.problem import Problem, from_arrays  # noqa: E402

SIZES = {          # n of each synthetic family per size class
    "tiny": dict(setcover=2_000, mknap=2_000, uc=(20, 50)),
    "small": dict(setcover=20_000, mknap=20_000, uc=(100, 100)),
    "medium": dict(setcover=100_000, mknap=100_000, uc=(250, 200)),
}


# ------------------------------------------------------------------------------------------------- generators
def gen_setcover(n: int, seed: int, deg: int = 4, rows_per_col: float = 0.25) -> Problem:
    """min c·x, every row covered >= 1, x binary; each column covers `deg` random rows; m = rows_per_col·n."""
    rs = np.random.default_rng(seed)
    m = max(1, int(rows_per_col * n))
    rows = rs.integers(0, m, size=(n, deg)).ravel()
    cols = np.repeat(np.arange(n), deg)
    A = sps.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(m, n))
    A.data[:] = 1.0                                     # duplicates summed -> back to 1
    empty = np.flatnonzero(np.diff(A.indptr) == 0)      # every row coverable
    if empty.size:
        extra = sps.csr_matrix((np.ones(empty.size), (empty, rs.integers(0, n, empty.size))), shape=(m, n))
        A = (A + extra).tocsr()
        A.data[:] = 1.0
    c = rs.integers(1, 10, size=n).astype(float)
    return from_arrays(A, np.ones(m), np.full(m, np.inf), np.zeros(n), np.ones(n), c, np.ones(n, bool),
                       path=f"<synthetic:setcover n={n} s={seed}>")


def gen_mknap(n: int, seed: int, deg: int = 4, rows_per_col: float = 0.01) -> Problem:
    """max v·x, x integer in [0, 5]; half the rows are capacities (w·x <= 0.3·5·Σw), half demands (d·x >= 0.25·5·Σd)."""
    rs = np.random.default_rng(seed)
    m = max(2, int(rows_per_col * n))
    rows = rs.integers(0, m, size=(n, deg)).ravel()
    cols = np.repeat(np.arange(n), deg)
    vals = rs.integers(1, 20, size=rows.size).astype(float)
    A = sps.csr_matrix((vals, (rows, cols)), shape=(m, n))
    rowsum = np.asarray(A.sum(axis=1)).ravel()
    cap = np.arange(m) < m // 2
    rl = np.where(cap, -np.inf, 0.25 * 5 * rowsum)
    ru = np.where(cap, 0.3 * 5 * rowsum, np.inf)
    v = rs.integers(1, 30, size=n).astype(float)
    return from_arrays(A, rl, ru, np.zeros(n), np.full(n, 5.0), v, np.ones(n, bool), sense=-1,
                       path=f"<synthetic:mknap n={n} s={seed}>")


def gen_uc(G: int, T: int, seed: int) -> Problem:
    """Unit-commitment-shaped mixed model: u[g,t] binary, p[g,t] in [0, Pmax_g], shed[t] >= 0 (cost 1e3/MW).
    Rows: Pmin_g u - p <= 0, p - Pmax_g u <= 0, Σ_g p[g,t] + shed[t] = D_t (equality, singleton slack)."""
    rs = np.random.default_rng(seed)
    pmax = rs.uniform(50, 400, G)
    pmin = pmax * rs.uniform(0.2, 0.5, G)
    D = pmax.sum() * rs.uniform(0.35, 0.7, T)
    nu = G * T
    iu = lambda g, t: g * T + t                         # noqa: E731
    ip = lambda g, t: nu + g * T + t                    # noqa: E731
    ish = 2 * nu
    rows, cols, vals, rl, ru = [], [], [], [], []
    r = 0
    for g in range(G):
        for t in range(T):
            rows += [r, r, r + 1, r + 1]
            cols += [iu(g, t), ip(g, t), ip(g, t), iu(g, t)]
            vals += [pmin[g], -1.0, 1.0, -pmax[g]]
            rl += [-np.inf, -np.inf]
            ru += [0.0, 0.0]
            r += 2
    for t in range(T):
        for g in range(G):
            rows.append(r), cols.append(ip(g, t)), vals.append(1.0)
        rows.append(r), cols.append(ish + t), vals.append(1.0)
        rl.append(D[t]), ru.append(D[t])
        r += 1
    n = 2 * nu + T
    A = sps.csr_matrix((vals, (rows, cols)), shape=(r, n))
    lb = np.zeros(n)
    ub = np.concatenate([np.ones(nu), np.repeat(pmax, T), np.full(T, np.inf)])
    c = np.concatenate([np.repeat(rs.uniform(100, 1000, G), T), np.repeat(rs.uniform(10, 60, G), T),
                        np.full(T, 1e3)])
    isint = np.concatenate([np.ones(nu, bool), np.zeros(nu + T, bool)])
    return from_arrays(A, rl, ru, lb, ub, c, isint, path=f"<synthetic:uc G={G} T={T} s={seed}>")


def start_set(p: Problem, P: int, seed: int) -> np.ndarray:
    """Seeded, mostly infeasible starts: binaries Bernoulli(0.05), integers / continuous uniform in the bounds
    (±10 where a bound is infinite)."""
    rs = np.random.default_rng(seed)
    lo = np.where(np.isfinite(p.lb), p.lb, 0.0)
    hi = np.where(np.isfinite(p.ub), p.ub, 10.0)
    X = lo + (hi - lo) * rs.random((P, p.n))
    X[:, p.ints] = np.round(X[:, p.ints])
    X[:, p.bins] = (rs.random((P, p.bins.size)) < 0.05).astype(float)
    return X


def problems(size: str, seed: int, which: list[str]) -> list[Problem]:
    s = SIZES[size]
    out = []
    for w in which:
        if w == "setcover":
            out.append(gen_setcover(s["setcover"], seed))
        elif w == "mknap":
            out.append(gen_mknap(s["mknap"], seed))
        elif w == "uc":
            out.append(gen_uc(*s["uc"], seed))
        elif w == "toy":
            out += [toy.knapsack(30), toy.set_cover(40, 25), toy.mixed()]
        else:
            raise SystemExit(f"unknown problem family {w!r}")
    return out


# ----------------------------------------------------------------------------------------------- measurement
class _PeakHook:
    """CuPy MemoryHook: the pool's used bytes after every malloc (peak above a base)."""

    def __new__(cls):
        import cupy as cp

        class Hook(cp.cuda.MemoryHook):
            name = "itrea_peak"

            def __init__(self):
                super().__init__()
                self.pool = cp.get_default_memory_pool()
                self.base = self.pool.used_bytes()
                self.peak = self.base

            def malloc_postprocess(self, **kw):
                self.peak = max(self.peak, self.pool.used_bytes())

            def alloc_postprocess(self, **kw):
                self.peak = max(self.peak, self.pool.used_bytes())

        return Hook()


def make_impl(name: str, dp, cfg: RepairCfg, mem_budget: int, seed: int):
    if name == "fj_cpu":
        from itrea.ea.ls_fj_cpu import FJCpu
        return FJCpu(dp, cfg, 1e-9, mem_budget, seed)
    if name == "fj_gpu":
        from itrea.ea.ls_fj_gpu import FJGpu
        return FJGpu(dp, cfg, 1e-9, mem_budget, seed)
    if name in ("dense_cpu", "dense_gpu"):
        from itrea.ea.ls_dense import DenseLS
        return DenseLS(dp, cfg, 1e-9, mem_budget, seed)
    raise ValueError(name)


def timed_run(impl, X, gpu: bool):
    """(counters, wall seconds, peak bytes above the level before the call)."""
    if gpu:
        import cupy as cp

        hook = _PeakHook()
        cp.cuda.Stream.null.synchronize()
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        with hook:
            e0.record()
            c = impl.run(X)
            e1.record()
            e1.synchronize()
        return c, cp.cuda.get_elapsed_time(e0, e1) / 1000.0, int(hook.peak - hook.base)
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    t0 = time.perf_counter()
    c = impl.run(X)
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    return c, dt, int(peak)


def bench_one(p: Problem, impl_name: str, P: int, args, X0: np.ndarray) -> dict:
    gpu = impl_name.endswith("_gpu")
    bk = get_backend("cupy", gpu_mem_gb=args.gpu_mem_gb) if gpu else get_backend("numpy")
    xp = bk.xp
    dp = to_device(p, bk, viol_norm=args.viol_norm)
    cfg = RepairCfg(method="dense" if impl_name.startswith("dense") else "fj", max_steps=args.max_steps,
                    stall_steps=args.stall_steps, steps=args.dense_steps)
    mem = int(args.mem_budget_mb) << 20
    row = dict(problem=p.path, n=p.n, m=p.m, nnz=p.nnz, impl=impl_name, P=P)
    # warm-up (kernel compilation / Numba cache load) on two individuals, outside the timing
    w = make_impl(impl_name, dp, cfg, mem, args.seed)
    w.run(xp.asarray(X0[:2].copy()))
    bk.synchronize()
    ev0 = evaluate(dp, xp.asarray(X0))
    V0 = float(bk.asnumpy(ev0.viol).sum())
    impl = make_impl(impl_name, dp, cfg, mem, args.seed)
    X = xp.asarray(X0.copy())
    c, dt, peak = timed_run(impl, X, gpu)
    ev = evaluate(dp, X)
    V1 = float(bk.asnumpy(ev.viol).sum())
    feas = bk.asnumpy(ev.feasible)
    row.update(seconds=dt, moves=int(c["moves"]), moves_per_s=c["moves"] / dt if dt > 0 else None,
               steps=int(c["steps"]), local_minima=int(c["local_minima"]), V0=V0, V1=V1,
               viol_reduction_per_s=(V0 - V1) / dt if dt > 0 else None, feasible_share=float(feas.mean()),
               finished=dict(feasible=c["finished_feasible"], stall=c["finished_stall"], budget=c["finished_budget"]),
               peak_bytes=peak, chunk=c.get("chunk"), state_bytes=c.get("state_bytes"),
               impl_peak_bytes=c.get("peak_bytes"))
    # time to first feasible: geometric max_steps ladder, first rung with >= 1 feasible individual
    ttff, rung = None, None
    for L in args.ladder:
        if L > args.max_steps:
            break
        cfgL = dataclasses.replace(cfg, max_steps=L, steps=min(L, args.dense_steps) if cfg.method == "dense"
                                   else cfg.steps)
        implL = make_impl(impl_name, dp, cfgL, mem, args.seed)
        XL = xp.asarray(X0.copy())
        _, dtL, _ = timed_run(implL, XL, gpu)
        if bool(bk.asnumpy(evaluate(dp, XL).feasible).any()):
            ttff, rung = dtL, L
            break
        if cfg.method == "dense" and L >= args.dense_steps:
            break                                       # dense: larger rungs repeat the same `steps`-bounded call
    if ttff is None and row["feasible_share"] > 0:
        ttff, rung = dt, args.max_steps
    row.update(ttff_ladder_s=ttff, ttff_rung=rung)
    del X, dp
    if gpu:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    return row


def available(impl: str) -> tuple[bool, str]:
    if impl.endswith("_gpu"):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                return False, "no CUDA device"
        except Exception as e:  # noqa: BLE001
            return False, f"CuPy unavailable: {e}"
    if impl == "fj_gpu":
        try:
            import itrea.ea.ls_fj_gpu  # noqa: F401
        except ImportError as e:
            return False, f"ls_fj_gpu does not import: {e}"
    return True, ""


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", choices=sorted(SIZES), default="small")
    ap.add_argument("--problems", default="setcover,mknap,uc")
    ap.add_argument("--impls", default="fj_cpu,fj_gpu,dense_cpu,dense_gpu")
    ap.add_argument("--P", default="16,64,256")
    ap.add_argument("--gpu-mem-gb", type=float, default=1.0)
    ap.add_argument("--mem-budget-mb", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--stall-steps", type=int, default=100)
    ap.add_argument("--dense-steps", type=int, default=50)
    ap.add_argument("--ladder", default="8,32,128,512,2000")
    ap.add_argument("--max-cpu-seconds", type=float, default=120.0,
                    help="skip larger P for a CPU implementation once one call took longer than this")
    ap.add_argument("--viol-norm", choices=["l2", "rhs", "count"], default="l2",
                    help="row divisor of the violation (the population's ranks.viol_norm; FJ uses the same)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    args.ladder = [int(v) for v in args.ladder.split(",")]
    Ps = [int(v) for v in args.P.split(",")]
    impls = [s for s in args.impls.split(",") if s]
    status = {i: available(i) for i in impls}
    rows, skipped = [], []
    for p in problems(args.size, args.seed, args.problems.split(",")):
        too_slow: set[str] = set()
        for P in Ps:
            X0 = start_set(p, P, args.seed + P)
            for impl in impls:
                ok, why = status[impl]
                if not ok:
                    continue
                if impl in too_slow:
                    skipped.append(dict(problem=p.path, impl=impl, P=P, reason="previous P exceeded max-cpu-seconds"))
                    continue
                try:
                    r = bench_one(p, impl, P, args, X0)
                except Exception as e:  # noqa: BLE001 - e.g. a pool limit: record and go on
                    skipped.append(dict(problem=p.path, impl=impl, P=P, reason=f"{type(e).__name__}: {e}"))
                    continue
                rows.append(r)
                if not impl.endswith("_gpu") and r["seconds"] > args.max_cpu_seconds:
                    too_slow.add(impl)
                print(f"{p.path:34s} {impl:9s} P={P:4d} {r['seconds']:8.3f}s moves/s={fmt(r['moves_per_s'], '10.3g')} "
                      f"feas={r['feasible_share']:.2f}", file=sys.stderr, flush=True)
    env = dict(python=platform.python_version(), machine=platform.machine(), numpy=np.__version__,
               omp=os.environ.get("OMP_NUM_THREADS"), unavailable={k: v[1] for k, v in status.items() if not v[0]})
    try:
        import cupy as cp
        env["cupy"] = cp.__version__
        env["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:  # noqa: BLE001
        pass
    try:
        import numba
        env["numba"] = numba.__version__
    except Exception:  # noqa: BLE001
        pass
    result = dict(config=vars(args) | {"P": Ps, "impls": impls}, env=env, rows=rows, skipped=skipped,
                  created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1, default=str))
    hdr = (f"{'problem':34s} {'impl':9s} {'P':>4s} {'sec':>8s} {'moves/s':>10s} {'dV/s':>10s} {'feas':>5s} "
           f"{'ttff_s':>8s} {'peak_MB':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['problem']:34s} {r['impl']:9s} {r['P']:4d} {r['seconds']:8.3f} {fmt(r['moves_per_s'], '10.3g')} "
              f"{fmt(r['viol_reduction_per_s'], '10.3g')} {r['feasible_share']:5.2f} {fmt(r['ttff_ladder_s'], '8.3f')} "
              f"{r['peak_bytes'] / 2**20:8.1f}")
    for k, v in env["unavailable"].items():
        print(f"unavailable: {k} ({v})")
    for s in skipped:
        print(f"skipped: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
