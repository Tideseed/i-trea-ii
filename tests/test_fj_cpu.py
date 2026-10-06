"""Feasibility-jump CPU reference (`itrea.ea.ls_fj_cpu`): the spec's invariants and toy feasibility rates.

Spec: vault `10-design/Feasibility jump local search.md` (Invariants). Toy models only; the first run compiles the
Numba kernel once per gene dtype (cached afterwards under src/itrea/ea/__pycache__).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sps

from itrea.backend import get_backend
from itrea.benchmarks import toy
from itrea.config import RepairCfg
from itrea.ea.evaluate import evaluate, to_device
from itrea.ea.ls_fj_cpu import (ST_BUDGET, ST_FEASIBLE, ST_STALL, FJCpu, cutoff_side, objective_is_integral,
                                var_types)
from itrea.problem import from_arrays

BK = get_backend("numpy")
KEYS = ("moves", "local_minima", "steps", "finished_feasible", "finished_stall", "finished_budget", "seconds")


def make_fj(p, seed: int = 0, imp_tol: float = 1e-9, **cfg):
    dp = to_device(p, BK)
    return dp, FJCpu(dp, RepairCfg(**cfg), imp_tol, 64 << 20, seed)


def random_start(p, k: int, seed: int = 0, density: float = 0.5, wild: bool = False) -> np.ndarray:
    """k points: binaries Bernoulli(density), integers / continuous uniform in the bounds (±10 where infinite).
    wild=True adds out-of-bound, fractional and non-finite genes (the projection must handle them)."""
    rs = np.random.default_rng(seed)
    lo = np.where(np.isfinite(p.lb), p.lb, -10.0)
    hi = np.where(np.isfinite(p.ub), p.ub, 10.0)
    X = lo + (hi - lo) * rs.random((k, p.n))
    X[:, p.ints] = np.round(X[:, p.ints])
    X[:, p.bins] = (rs.random((k, p.bins.size)) < density).astype(float)
    if wild:
        X += rs.normal(0, 3, X.shape) * (rs.random(X.shape) < 0.3)
        X[rs.random(X.shape) < 0.03] = np.nan
        X[rs.random(X.shape) < 0.02] = np.inf
    return X


def mixed_offset() -> object:
    """toy.mixed with an objective offset (neither toy has one): exercises key_offset in the cutoff side."""
    p = toy.mixed()
    return from_arrays(p.A, p.rl, p.ru, p.lb, p.ub, p.c, p.isint, sense=1, offset=7.5, names=p.names)


def knapsack_int(n: int = 30, seed: int = 0):
    """Bounded general-integer knapsack-like model with two rows (≥ and ≤): x_j in {0..3}."""
    rs = np.random.default_rng(seed)
    w = rs.integers(3, 15, size=n).astype(float)
    v = (w + rs.integers(0, 6, size=n)).astype(float)
    A = sps.csr_matrix(np.vstack([w, np.ones(n)]))
    return from_arrays(A, [-np.inf, n // 3], [w.sum() * 0.8, np.inf], np.zeros(n), np.full(n, 3.0), v,
                       np.ones(n, bool), sense=-1, path="<test:intknap>")


def assert_projected(p, X):
    _, lb, ub = var_types(p)
    Xf = np.asarray(X, dtype=np.float64)
    assert np.all(np.isfinite(Xf))
    assert np.all(Xf >= lb - 1e-12) and np.all(Xf <= ub + 1e-12)
    xi = Xf[:, p.ints]
    assert np.array_equal(xi, np.round(xi))


PROBLEMS = {
    "knapsack": lambda: toy.knapsack(30),
    "set_cover": lambda: toy.set_cover(40, 25),
    "mixed": toy.mixed,
    "mixed_offset": mixed_offset,
    "int_knapsack": knapsack_int,
}


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_invariants_integrality_bounds_activities(name):
    p = PROBLEMS[name]()
    dp, fj = make_fj(p, seed=3)
    X = random_start(p, 24, seed=1, wild=True)
    c = fj.run(X, keep_state=True)
    assert set(KEYS) <= set(c)
    assert_projected(p, X)
    # incremental activities equal A_aug x (to 1e-9 relative)
    act = (fj.rows.A @ X.T).T
    err = np.abs(act - fj.state["act"]) / np.maximum(1.0, np.abs(act))
    assert err.max(initial=0.0) <= 1e-9
    assert c["act_drift"] <= 1e-9
    # every individual finished exactly once; feasible finish => evaluate-feasible
    assert c["finished_feasible"] + c["finished_stall"] + c["finished_budget"] == X.shape[0]
    ev = evaluate(dp, X)
    st = fj.last_status
    assert np.all(ev.feasible[st == ST_FEASIBLE])
    assert np.all(fj.state["nviol"][st == ST_FEASIBLE] == 0)
    assert c["moves"] == int(fj.last_moves.sum()) and c["moves"] <= c["steps"]


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.float32, np.float64])
def test_gene_dtypes(dtype):
    p = toy.set_cover(40, 25)
    dp, fj = make_fj(p, seed=0)
    X = np.zeros((16, p.n), dtype=dtype)
    c = fj.run(X)
    assert X.dtype == dtype
    assert_projected(p, X)
    assert c["finished_feasible"] == 16
    assert bool(np.all(evaluate(dp, X).feasible))


def test_integer_dtype_with_continuous_raises():
    p = toy.mixed()
    _, fj = make_fj(p)
    with pytest.raises(TypeError):
        fj.run(np.zeros((2, p.n), dtype=np.int32))
    with pytest.raises(ValueError):
        fj.run(np.zeros((2, p.n + 1)))


@pytest.mark.parametrize("name", ["knapsack", "set_cover", "mixed", "int_knapsack"])
def test_feasible_input_unchanged_when_cutoff_inactive(name):
    p = PROBLEMS[name]()
    dp, fj = make_fj(p, seed=5)
    X = random_start(p, 32, seed=2)
    fj.run(X)
    feas = np.asarray(evaluate(dp, X).feasible) & (fj.last_status == ST_FEASIBLE)
    assert feas.sum() >= 8
    F = X[feas].copy()
    dtypes = [np.float64] + ([np.int8] if p.conts.size == 0 else [])
    for dt in dtypes:
        Y = F.astype(dt)
        Y0 = Y.copy()
        fj2 = FJCpu(dp, RepairCfg(), 1e-9, 64 << 20, seed=11)
        fj2.set_cutoff(None)
        c = fj2.run(Y)
        assert np.array_equal(Y, Y0), f"feasible input changed ({dt})"
        assert c["moves"] == 0 and c["finished_feasible"] == len(Y)


def test_input_better_than_cutoff_unchanged():
    p = toy.knapsack(30)
    dp, fj = make_fj(p, seed=1)
    X = random_start(p, 32, seed=4, density=0.3)
    fj.run(X)
    ev = evaluate(dp, X)
    keys = np.asarray(ev.key)[np.asarray(ev.feasible)]
    inc = float(np.max(keys)) + 5.0                    # every feasible point beats it by >= 5
    Y = X[np.asarray(ev.feasible)].copy()
    Y0 = Y.copy()
    fj.set_cutoff(inc)
    c = fj.run(Y)
    assert np.array_equal(Y, Y0) and c["moves"] == 0


def _greedy_by_index(p) -> np.ndarray:
    """A deliberately weak feasible knapsack point: items in index order while they fit."""
    w = p.A.toarray()[0]
    x = np.zeros(p.n)
    cap = p.ru[0]
    for j in range(p.n):
        if w[j] <= cap:
            x[j], cap = 1.0, cap - w[j]
    return x


def test_cutoff_gives_strictly_better_feasible_knapsack():
    p = toy.knapsack(30)
    dp, fj = make_fj(p, seed=7)
    x0 = _greedy_by_index(p)
    assert p.check(x0).feasible
    inc = p.key(x0)
    assert -p.toy_optimum <= inc - 1.0, "a strictly better solution must exist for this test to mean anything"
    assert fj.obj_integral
    fj.set_cutoff(inc)
    assert fj.cut_active and fj.cut_rhs == math.floor(inc - 1.0 + 1e-6)
    X = random_start(p, 64, seed=3)
    c = fj.run(X)
    ev = evaluate(dp, X)
    st = fj.last_status == ST_FEASIBLE
    assert c["finished_feasible"] >= 32
    keys = np.asarray(ev.key)
    assert np.all(np.asarray(ev.feasible)[st])
    assert np.all(keys[st] <= inc - 1.0)                 # integral objective: better by at least one unit


@pytest.mark.parametrize("name", ["mixed", "mixed_offset"])
def test_cutoff_gives_strictly_better_feasible_continuous_objective(name):
    p = PROBLEMS[name]()
    imp_tol = 1e-9
    dp, fj = make_fj(p, seed=2, imp_tol=imp_tol)
    assert not fj.obj_integral                          # costs sit on continuous variables too
    X = random_start(p, 64, seed=5)
    fj.run(X)
    ev = evaluate(dp, X)
    keys = np.asarray(ev.key)[np.asarray(ev.feasible)]
    inc = float(np.median(keys))
    assert keys.min() < inc
    fj.set_cutoff(inc)
    Y = random_start(p, 64, seed=6)
    c = fj.run(Y)
    ev = evaluate(dp, Y)
    st = fj.last_status == ST_FEASIBLE
    assert c["finished_feasible"] >= 16
    assert np.all(np.asarray(ev.feasible)[st])
    # strictly better under the rank rule key < inc - imp_tol * max(1, |inc|)
    assert np.all(np.asarray(ev.key)[st] < inc - imp_tol * max(1.0, abs(inc)))


def test_cutoff_side_rule():
    # integral objective: rounded down with a 1e-6 allowance for solver noise; offset handled
    assert cutoff_side(105.0, 0.0, 1e-9, True) == 104.0
    assert cutoff_side(104.99999999, 0.0, 1e-9, True) == 104.0
    assert cutoff_side(-3.0, 7.5, 1e-9, False) < -10.5 - 1e-9
    assert cutoff_side(None, 0.0, 1e-9, True) == math.inf
    assert cutoff_side(math.inf, 0.0, 1e-9, False) == math.inf
    # huge keys: delta follows imp_tol * |inc| once it exceeds the unit step
    assert cutoff_side(1e12, 0.0, 1e-9, True) == 1e12 - 1e3


def test_cutoff_inactive_without_objective():
    p = toy.set_cover(20, 10)
    q = from_arrays(p.A, p.rl, p.ru, p.lb, p.ub, np.zeros(p.n), p.isint)
    assert not objective_is_integral(q)
    _, fj = make_fj(q)
    fj.set_cutoff(5.0)
    assert not fj.cut_active


@pytest.mark.parametrize("name,start,min_rate", [
    ("knapsack", "ones", 0.95), ("set_cover", "zeros", 0.95), ("set_cover", "sparse", 0.95),
    ("mixed", "random", 0.9), ("int_knapsack", "random", 0.9)])
def test_feasibility_rates_from_random_starts(name, start, min_rate):
    p = PROBLEMS[name]()
    dp, fj = make_fj(p, seed=9)
    k = 64
    if start == "ones":
        X = np.ones((k, p.n))
    elif start == "zeros":
        X = np.zeros((k, p.n))
    elif start == "sparse":
        X = random_start(p, k, seed=8, density=0.05)
    else:
        X = random_start(p, k, seed=8)
    pre = float(np.mean(np.asarray(evaluate(dp, X).feasible)))
    c = fj.run(X)
    rate = float(np.mean(np.asarray(evaluate(dp, X).feasible)))
    assert pre < 0.5, "the start set must be mostly infeasible for the rate to mean anything"
    assert rate >= min_rate, (rate, c)


def test_deterministic_for_a_seed():
    p = toy.mixed()
    X0 = random_start(p, 32, seed=1)
    outs = []
    for seed in (4, 4, 5):
        _, fj = make_fj(p, seed=seed)
        X = X0.copy()
        fj.run(X)
        outs.append(X)
    assert np.array_equal(outs[0], outs[1])
    # a repeated call on the same object draws new random numbers, but is still a function of (seed, call count)
    _, fa = make_fj(p, seed=4)
    _, fb = make_fj(p, seed=4)
    for _ in range(2):
        Xa, Xb = X0.copy(), X0.copy()
        fa.run(Xa)
        fb.run(Xb)
        assert np.array_equal(Xa, Xb)


def test_individual_result_independent_of_batch():
    p = toy.set_cover(40, 25)
    X0 = random_start(p, 16, seed=3, density=0.1)
    _, fa = make_fj(p, seed=1)
    _, fb = make_fj(p, seed=1)
    Xa = X0.copy()
    fa.run(Xa)
    Xb = X0[:8].copy()
    fb.run(Xb)
    assert np.array_equal(Xa[:8], Xb)


def test_max_steps_zero_projects_only():
    p = toy.mixed()
    dp, fj = make_fj(p, max_steps=0)
    X = random_start(p, 16, seed=2, wild=True)
    c = fj.run(X)
    assert c["moves"] == 0
    assert_projected(p, X)
    feas = np.asarray(evaluate(dp, X).feasible)
    assert c["finished_feasible"] == int(feas.sum()) and c["finished_budget"] == 16 - int(feas.sum())


def test_infeasible_model_stops_by_stall():
    # x0 + x1 >= 3 with binaries: no feasible point; every individual must stop by the stall rule
    p = from_arrays(sps.csr_matrix(np.array([[1.0, 1.0]])), [3.0], [np.inf], [0, 0], [1, 1], [1.0, 1.0],
                    [True, True])
    _, fj = make_fj(p, max_steps=10_000, stall_steps=50)
    X = np.zeros((8, 2))
    c = fj.run(X)
    assert c["finished_stall"] == 8 and np.all(fj.last_status == ST_STALL)
    assert np.all(fj.last_steps >= 50)


def test_budget_stop():
    p = toy.set_cover(40, 25)
    _, fj = make_fj(p, max_steps=1)
    X = np.zeros((8, p.n))
    c = fj.run(X)
    assert np.all(fj.last_moves <= 1)
    assert c["finished_budget"] == int(np.sum(fj.last_status == ST_BUDGET)) >= 1


@pytest.mark.parametrize("mode", ["weighted", "none"])
def test_other_objective_modes(mode):
    p = toy.knapsack(30)
    dp, fj = make_fj(p, seed=1, objective_mode=mode)
    fj.set_cutoff(-100.0)                               # ignored outside cutoff_row mode
    assert not fj.cut_active
    X = np.ones((32, p.n))
    c = fj.run(X)
    assert_projected(p, X)
    assert c["finished_feasible"] >= 30
    assert bool(np.all(np.asarray(evaluate(dp, X).feasible)[fj.last_status == ST_FEASIBLE]))


def test_viol_norm_count_and_rhs():
    p = toy.mixed()
    for vn in ("count", "rhs"):
        dp = to_device(p, BK, viol_norm=vn)
        fj = FJCpu(dp, RepairCfg(), 1e-9, 64 << 20, 0)
        assert fj.rows.viol_norm == vn
        X = random_start(p, 16, seed=1)
        c = fj.run(X)
        assert c["finished_feasible"] >= 12


def test_random_walk_knob_and_row_scan_cap():
    p = toy.set_cover(60, 30, density=0.4)
    dp = to_device(p, BK)
    fj = FJCpu(dp, RepairCfg(), 1e-9, 64 << 20, 0, p_random_walk=0.2, row_scan_cap=4)
    X = np.zeros((16, p.n))
    c = fj.run(X)
    assert_projected(p, X)
    assert c["finished_feasible"] == 16
