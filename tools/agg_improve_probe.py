"""Improvement probe for the two-variable equality aggregation (CPU only).

In a coupled run SCIP hands the EA feasible solutions early; the EA's job is to improve them. Here: one of SCIP's
incumbents (from an A0 run's solver.incumbents.npz), K mutated copies (a few integer genes stepped or flipped), the CPU
feasibility jump with the objective cutoff at that incumbent (a child must be feasible AND strictly better), on (a)
the original problem and (b) the aggregated one (the incumbent projected; mutations drawn in the reduced space, so a
chain moves as one gene). Reports how many children end feasible and better, and the best objective found.

Usage (repo root, pinned): OMP_NUM_THREADS=1 taskset -c 10-14 .venv/bin/python tools/agg_improve_probe.py \
    <instance> <A0 run dir> [incumbent index] [K] [genes mutated] [seed]
"""
import sys
import time

import numpy as np

from itrea.backend import get_backend
from itrea.config import RepairCfg
from itrea.ea.aggregate import aggregate
from itrea.ea.evaluate import evaluate, to_device
from itrea.ea.repair import LocalSearch
from itrea.problem import load
from itrea.trace import load_incumbents


def mutate(p, x, k, genes, rng):
    X = np.repeat(x[None, :], k, axis=0)
    ints = np.flatnonzero(p.isint & (p.ub > p.lb))
    for r in range(k):
        js = rng.choice(ints, size=min(genes, ints.size), replace=False)
        for j in js:
            if p.lb[j] == 0 and p.ub[j] == 1:
                X[r, j] = 1 - X[r, j]
            else:
                X[r, j] = np.clip(X[r, j] + rng.choice([-2, -1, 1, 2]), p.lb[j], p.ub[j])
    return X


def run(p, X, inc_key, label, steps, stall):
    bk = get_backend("numpy")
    dp = to_device(p, bk)
    ls = LocalSearch(bk, dp, RepairCfg(max_steps=steps, stall_steps=stall), seed=1, mem_budget_bytes=1 << 30)
    ls.set_cutoff(inc_key)
    X = X.copy()
    t = time.perf_counter()
    st = ls.run(X)
    dt = time.perf_counter() - t
    ev = evaluate(dp, X, 1e-6, 1e-6)
    feas = np.asarray(ev.feasible, bool)
    key = np.asarray(ev.key)
    better = feas & (key < inc_key - 1e-9 * max(1.0, abs(inc_key)))
    best = key[better].min() if better.any() else None
    print(f"{label:10s} n={p.n:6d} better {int(better.sum()):3d}/{X.shape[0]}  feasible {int(feas.sum()):3d}  "
          f"best key {best}  median V {np.median(np.asarray(ev.viol)):.3f}  moves {st.get('moves')}  {dt:5.1f} s")
    return X, better, key


def main():
    path, run_dir = sys.argv[1], sys.argv[2]
    which = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    genes = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    p = load(path)
    names, incs = load_incumbents(f"{run_dir}/solver.incumbents.npz")
    pos = {nm: i for i, nm in enumerate(names)}
    t, obj, src, xs = incs[which]
    x = np.zeros(p.n)
    for j, nm in enumerate(p.names):
        x[j] = xs[pos[nm]]
    x = p.round_integers(x)
    assert p.check(x).feasible, "the incumbent must be feasible"
    inc_key = p.key(x)
    print(f"{path}: SCIP incumbent #{which} at {t:.2f} s, objective {obj} (key {inc_key}); {len(incs)} incumbents")
    ag = aggregate(p)
    print(" aggregation:", ag.summary() if ag else None)
    for steps, stall in ((2000, 200), (20000, 1000)):
        print(f" budget max_steps={steps} stall={stall}")
        run(p, mutate(p, x, k, genes, np.random.default_rng(seed)), inc_key, "original", steps, stall)
        if ag is not None:
            y = ag.project(x)
            assert ag.reduced.check(y).feasible
            Y, better, key = run(ag.reduced, mutate(ag.reduced, y, k, genes, np.random.default_rng(seed)), inc_key,
                                 "aggregated", steps, stall)
            for i in np.flatnonzero(better)[:3]:
                xo = p.round_integers(ag.lift(Y[i]))
                assert p.check(xo).feasible and abs(p.key(xo) - key[i]) < 1e-6 * max(1.0, abs(key[i]))


if __name__ == "__main__":
    main()
