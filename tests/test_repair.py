"""The repair dispatcher (`itrea.ea.repair.LocalSearch`) and the dense baseline (`itrea.ea.ls_dense.DenseLS`).

Toy models only. GPU tests (marker `gpu`) need CuPy and a device; they use a 1 GB pool at most.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest
import scipy.sparse as sps

from itrea.backend import Backend, get_backend
from itrea.benchmarks import toy
from itrea.config import RepairCfg
from itrea.ea.evaluate import evaluate, to_device
from itrea.ea.ls_dense import DenseLS
from itrea.ea.ls_fj_cpu import ST_FEASIBLE, FJCpu, var_types
from itrea.ea.repair import LocalSearch
from itrea.problem import from_arrays

BK = get_backend("numpy")
KEYS = ("moves", "local_minima", "steps", "finished_feasible", "finished_stall", "finished_budget", "seconds",
        "method")


def gpu_backend():
    pytest.importorskip("cupy")
    try:
        return get_backend("cupy", gpu_mem_gb=1.0)
    except Exception as e:  # pragma: no cover - no device
        pytest.skip(f"no CuPy device: {e}")


def random_start(p, k: int, seed: int = 0, density: float = 0.5, wild: bool = False) -> np.ndarray:
    rs = np.random.default_rng(seed)
    lo = np.where(np.isfinite(p.lb), p.lb, -10.0)
    hi = np.where(np.isfinite(p.ub), p.ub, 10.0)
    X = lo + (hi - lo) * rs.random((k, p.n))
    X[:, p.ints] = np.round(X[:, p.ints])
    X[:, p.bins] = (rs.random((k, p.bins.size)) < density).astype(float)
    if wild:
        X += rs.normal(0, 3, X.shape) * (rs.random(X.shape) < 0.3)
        X[rs.random(X.shape) < 0.03] = np.nan
    return X


def assert_projected(p, X):
    _, lb, ub = var_types(p)
    Xf = np.asarray(X, dtype=np.float64)
    assert np.all(np.isfinite(Xf))
    assert np.all(Xf >= lb - 1e-12) and np.all(Xf <= ub + 1e-12)
    xi = Xf[:, p.ints]
    assert np.array_equal(xi, np.round(xi))


PROBLEMS = {"knapsack": lambda: toy.knapsack(30), "set_cover": lambda: toy.set_cover(40, 25), "mixed": toy.mixed}


# ------------------------------------------------------------------------------------------------ dispatcher
def test_dispatch_numpy():
    p = toy.mixed()
    dp = to_device(p, BK)
    assert isinstance(LocalSearch(BK, dp, RepairCfg(method="fj")).impl, FJCpu)
    assert isinstance(LocalSearch(BK, dp, RepairCfg(method="dense")).impl, DenseLS)
    ls = LocalSearch(BK, dp, RepairCfg(method="none"))
    assert ls.impl_name == "none"
    with pytest.raises(ValueError):
        LocalSearch(BK, dp, RepairCfg(method="greedy"))


@pytest.mark.parametrize("method", ["fj", "dense", "none"])
def test_dispatch_counters_and_invariants(method):
    p = toy.mixed()
    dp = to_device(p, BK)
    ls = LocalSearch(BK, dp, RepairCfg(method=method), imp_tol=1e-9, mem_budget_bytes=16 << 20, seed=1)
    ls.set_cutoff(None)
    X = random_start(p, 16, seed=2, wild=True)
    c = ls.run(X)
    assert set(KEYS) <= set(c)
    assert c["finished_feasible"] + c["finished_stall"] + c["finished_budget"] == 16
    assert_projected(p, X)


def test_none_projects_int8_and_float():
    p = toy.knapsack(20)
    dp = to_device(p, BK)
    ls = LocalSearch(BK, dp, RepairCfg(method="none"), mem_budget_bytes=200)    # forces several chunks
    X = np.array([[2, -1] + [1] * 18, [0] * 20], dtype=np.int8)
    ls.run(X)
    assert X.dtype == np.int8 and X.min() >= 0 and X.max() <= 1
    Y = np.full((3, 20), 0.6)
    Y[0, 0] = np.nan
    ls.run(Y)
    assert_projected(p, Y)
    assert Y[0, 0] == 0.0 and Y[1, 1] == 1.0


def test_gpu_fj_without_gpu_module_raises(monkeypatch):
    """No silent CPU fallback: a CuPy-labelled backend whose FJ module cannot be imported is an ImportError."""
    p = toy.knapsack(10)
    dp = to_device(p, BK)
    fake_gpu = Backend("cupy", np, sps)                 # is_gpu = True without touching CUDA
    monkeypatch.setitem(sys.modules, "itrea.ea.ls_fj_gpu", None)
    with pytest.raises(ImportError, match="refusing to fall back"):
        LocalSearch(fake_gpu, dp, RepairCfg(method="fj"))


def test_set_cutoff_forwarded():
    p = toy.knapsack(30)
    dp = to_device(p, BK)
    for method in ("fj", "dense"):
        ls = LocalSearch(BK, dp, RepairCfg(method=method))
        ls.set_cutoff(-100.0)
        assert ls.impl.cut_active and ls.impl.cut_rhs == -101.0
        ls.set_cutoff(None)
        assert not ls.impl.cut_active


# ---------------------------------------------------------------------------------------------- dense (CPU)
@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_dense_invariants_and_rates(name):
    p = PROBLEMS[name]()
    dp = to_device(p, BK)
    ls = DenseLS(dp, RepairCfg(method="dense", steps=300), 1e-9, 4 << 20, seed=0)
    if name == "knapsack":
        X = np.ones((32, p.n))
    elif name == "set_cover":
        X = np.zeros((32, p.n))
    else:
        X = random_start(p, 32, seed=1, wild=True)
    c = ls.run(X)
    assert_projected(p, X)
    ev = evaluate(dp, X)
    feas = np.asarray(ev.feasible)
    assert np.all(feas[ls.last_status == ST_FEASIBLE])
    assert c["finished_feasible"] == int(np.sum(ls.last_status == ST_FEASIBLE))
    assert feas.mean() >= 0.9


def test_dense_chunking_does_not_change_results():
    p = toy.set_cover(40, 25)
    dp = to_device(p, BK)
    X0 = random_start(p, 24, seed=3, density=0.1)
    a = DenseLS(dp, RepairCfg(method="dense", steps=100), 1e-9, 1 << 30, seed=0)
    b = DenseLS(dp, RepairCfg(method="dense", steps=100), 1e-9, 1, seed=0)        # chunk = 1
    Xa, Xb = X0.copy(), X0.copy()
    ca, cb = a.run(Xa), b.run(Xb)
    assert ca["chunk"] == 24 and cb["chunk"] == 1
    assert np.array_equal(Xa, Xb)


def test_dense_feasible_input_unchanged():
    p = toy.knapsack(30)
    dp = to_device(p, BK)
    ls = DenseLS(dp, RepairCfg(method="dense", steps=200), 1e-9, 4 << 20, seed=0)
    X = random_start(p, 32, seed=2, density=0.3)
    ls.run(X)
    feas = np.asarray(evaluate(dp, X).feasible)
    for dt in (np.float64, np.int8):
        Y = X[feas].astype(dt)
        Y0 = Y.copy()
        c = ls.run(Y)
        assert np.array_equal(Y, Y0) and c["moves"] == 0


def test_dense_cutoff_strictly_better():
    p = toy.knapsack(30)
    dp = to_device(p, BK)
    ls = DenseLS(dp, RepairCfg(method="dense", steps=300), 1e-9, 4 << 20, seed=0)
    inc = -0.8 * p.toy_optimum
    ls.set_cutoff(inc)
    X = random_start(p, 32, seed=4)
    c = ls.run(X)
    ev = evaluate(dp, X)
    st = ls.last_status == ST_FEASIBLE
    assert c["finished_feasible"] >= 16
    assert np.all(np.asarray(ev.feasible)[st]) and np.all(np.asarray(ev.key)[st] <= inc - 1.0)


def test_dense_moves_per_step_disjoint_rows():
    # three independent rows (disjoint variables): three moves per step repair all three at once
    A = sps.csr_matrix(sps.block_diag([sps.csr_matrix(np.ones((1, 4)))] * 3))
    p = from_arrays(A, [1, 1, 1], [np.inf] * 3, np.zeros(12), np.ones(12), np.ones(12), np.ones(12, bool))
    dp = to_device(p, BK)
    ls = DenseLS(dp, RepairCfg(method="dense", steps=1, moves_per_step=3), 1e-9, 4 << 20, seed=0)
    X = np.zeros((4, 12))
    c = ls.run(X)
    assert c["moves"] == 12 and c["steps"] == 4
    assert np.all(np.asarray(evaluate(dp, X).feasible))
    # one row: only one move per step (the others share it)
    q = from_arrays(sps.csr_matrix(np.ones((1, 4))), [2], [np.inf], np.zeros(4), np.ones(4), np.ones(4),
                    np.ones(4, bool))
    ls = DenseLS(to_device(q, BK), RepairCfg(method="dense", steps=1, moves_per_step=3), 1e-9, 4 << 20, seed=0)
    c = ls.run(np.zeros((2, 4)))
    assert c["moves"] == 2


# ------------------------------------------------------------------------------------------------------ GPU
@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_dense_gpu_matches_cpu(name):
    bk = gpu_backend()
    import cupy as cp

    p = PROBLEMS[name]()
    X0 = random_start(p, 32, seed=5, density=0.2)
    cfg = RepairCfg(method="dense", steps=150)
    a = DenseLS(to_device(p, BK), cfg, 1e-9, 4 << 20, seed=0)
    b = DenseLS(to_device(p, bk), cfg, 1e-9, 4 << 20, seed=0)
    Xa, Xb = X0.copy(), cp.asarray(X0)
    ca, cb = a.run(Xa), b.run(Xb)
    assert np.allclose(cp.asnumpy(Xb), Xa, rtol=0, atol=1e-9)
    assert ca["moves"] == cb["moves"] and ca["finished_feasible"] == cb["finished_feasible"]
    assert cb["peak_bytes"] <= 4 << 20


@pytest.mark.gpu
def test_dispatch_gpu_fj_and_rates_agree_with_cpu():
    bk = gpu_backend()
    import cupy as cp

    try:
        import itrea.ea.ls_fj_gpu  # noqa: F401
    except ImportError as e:
        pytest.skip(f"ls_fj_gpu not importable yet: {e}")
    for name, start in (("knapsack", "ones"), ("set_cover", "zeros"), ("mixed", "random")):
        p = PROBLEMS[name]()
        X0 = (np.ones((64, p.n)) if start == "ones" else np.zeros((64, p.n)) if start == "zeros"
              else random_start(p, 64, seed=7))
        cfg = RepairCfg(method="fj")
        g = LocalSearch(bk, to_device(p, bk), cfg, 1e-9, 64 << 20, seed=1)
        c = LocalSearch(BK, to_device(p, BK), cfg, 1e-9, 64 << 20, seed=1)
        assert g.impl_name == "fj_gpu" and c.impl_name == "fj_cpu"
        Xg, Xc = cp.asarray(X0), X0.copy()
        cg, cc = g.run(Xg), c.run(Xc)
        rg = float(cp.mean(evaluate(to_device(p, bk), Xg).feasible))
        rc = float(np.mean(evaluate(to_device(p, BK), Xc).feasible))
        assert_projected(p, cp.asnumpy(Xg))
        assert abs(rg - rc) <= 0.15, (name, rg, rc, cg, cc)


@pytest.mark.parametrize("mode", ["weighted", "none"])
def test_dense_other_objective_modes(mode):
    p = toy.knapsack(30)
    dp = to_device(p, BK)
    ls = DenseLS(dp, RepairCfg(method="dense", steps=200, objective_mode=mode), 1e-9, 4 << 20, seed=0)
    assert ls.m_aug == p.m and (ls.cobj is not None) == (mode == "weighted")
    ls.set_cutoff(-100.0)                               # ignored outside cutoff_row mode
    assert not ls.cut_active
    X = np.ones((16, p.n))
    c = ls.run(X)
    assert_projected(p, X)
    assert c["finished_feasible"] == 16
    assert bool(np.all(np.asarray(evaluate(dp, X).feasible)))
