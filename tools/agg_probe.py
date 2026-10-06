"""Payoff probe for the two-variable equality aggregation (src/itrea/ea/aggregate.py), CPU only.

Same LP relaxation point, K randomised roundings, the CPU feasibility jump (Numba, `repair` defaults + the c01 budget)
on (a) the original problem and (b) the aggregated one (roundings drawn in the reduced space, so every equality chain
starts consistent). Reports how many children end feasible, the violation left, and the wall time; reduced-feasible
children are lifted and re-checked on the ORIGINAL problem.

Usage (repo root, pinned): OMP_NUM_THREADS=1 taskset -c 10-14 .venv/bin/python tools/agg_probe.py <instance> [K] [seed]
"""
import sys
import time

import highspy
import numpy as np

from itrea.backend import get_backend
from itrea.config import RepairCfg
from itrea.ea.aggregate import aggregate
from itrea.ea.evaluate import evaluate, to_device
from itrea.ea.repair import LocalSearch
from itrea.paths import resolve
from itrea.problem import load


def lp_point(path) -> np.ndarray:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solve_relaxation", True)
    h.setOptionValue("threads", 1)
    h.readModel(str(resolve(path)))
    h.run()
    return np.asarray(h.getSolution().col_value, dtype=np.float64)


def roundings(p, x, k, rng):
    f = np.floor(x)
    X = np.where(p.isint[None, :], f[None, :] + (rng.random((k, p.n)) < (x - f)[None, :]), x[None, :])
    return np.clip(X, p.lb, p.ub)


def run(p, X, label):
    bk = get_backend("numpy")
    dp = to_device(p, bk)
    cfg = RepairCfg(max_steps=20000, stall_steps=1000)
    ls = LocalSearch(bk, dp, cfg, seed=1, mem_budget_bytes=1 << 30)
    X = X.copy()
    t = time.perf_counter()
    st = ls.run(X)
    dt = time.perf_counter() - t
    ev = evaluate(dp, X, 1e-6, 1e-6)
    key, viol, feas = ev.key, ev.viol, ev.feasible
    viol = np.asarray(viol)
    print(f"{label:10s} n={p.n:6d} m={p.m:6d}  feasible {int(np.sum(feas)):3d}/{X.shape[0]}  V median {np.median(viol):9.3f} "
          f"min {viol.min():8.3f}  moves {st.get('moves')}  {dt:6.1f} s")
    return X, np.asarray(feas, bool), np.asarray(key)


def main():
    path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    p = load(path)
    ag = aggregate(p)
    print(path, ag.summary() if ag else "no aggregation")
    x = lp_point(path)
    rng = np.random.default_rng(seed)
    run(p, roundings(p, x, k, rng), "original")
    if ag is not None:
        rng = np.random.default_rng(seed)
        Y, feas, key = run(ag.reduced, roundings(ag.reduced, ag.project(x), k, rng), "aggregated")
        if feas.any():
            ok = [p.check(p.round_integers(ag.lift(Y[i]))).feasible for i in np.flatnonzero(feas)]
            print(f"lifted feasible on the original: {sum(ok)}/{len(ok)}; best objective "
                  f"{min(p.objective(ag.lift(Y[i])) * p.sense for i in np.flatnonzero(feas)) * p.sense}")


if __name__ == "__main__":
    main()
