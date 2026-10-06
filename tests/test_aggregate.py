"""Two-variable equality aggregation (src/itrea/ea/aggregate.py): exact maps, integrality, cycles, infeasibility."""
from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sps

from itrea.ea.aggregate import aggregate
from itrea.problem import from_arrays


def chain_problem():
    """x0 binary; x1 = x0; x2 = 1 - x1 (binary); c3 continuous = x2; c4 continuous = 0.5 + c5 (all-continuous tie, a
    fractional offset); a cycle row x2 + x0 = 1 (redundant: becomes 0 = 0); one knapsack row over x0, x2, c3, c4;
    general integer g6 = 2 - g7 (signs and offsets)."""
    rows = [
        ({0: 1, 1: -1}, 0, 0),             # x1 = x0
        ({1: 1, 2: 1}, 1, 1),              # x2 = 1 - x1
        ({2: 1, 3: -1}, 0, 0),             # c3 = x2
        ({4: 1, 5: -1}, 0.5, 0.5),         # c4 = c5 + 0.5
        ({2: 1, 0: 1}, 1, 1),              # cycle: x2 + x0 = 1
        ({0: 3, 2: 2, 3: 1, 4: 1}, -np.inf, 4.5),
        ({6: 1, 7: 1}, 2, 2),              # g6 = 2 - g7
        ({6: 1, 7: 2, 0: 1}, -np.inf, 6),
    ]
    m, n = len(rows), 8
    A = sps.lil_matrix((m, n))
    rl, ru = np.empty(m), np.empty(m)
    for i, (coef, lo, hi) in enumerate(rows):
        for j, v in coef.items():
            A[i, j] = v
        rl[i], ru[i] = lo, hi
    lb = np.array([0, 0, 0, 0, 0, 0, -3, -3], float)
    ub = np.array([1, 1, 1, 1, 2, 1.2, 5, 5], float)
    isint = np.array([1, 1, 1, 0, 0, 0, 1, 1], bool)
    c = np.array([1.0, 2.0, -1.0, 0.5, 1.0, -2.0, 1.0, 3.0])
    return from_arrays(A.tocsr(), rl, ru, lb, ub, c, isint, offset=7.0, names=[f"v{j}" for j in range(n)])


def test_maps_are_exact_and_feasibility_is_preserved():
    p = chain_problem()
    ag = aggregate(p)
    assert ag is not None
    r = ag.reduced
    assert ag.ties == 5 and ag.cycles == 1 and ag.refused == 0
    assert r.n == 3                                       # {x0, x1, x2, c3}, {c4, c5}, {g6, g7}
    # the cycle row x2 + x0 = 1 becomes 0 = 0 and is dropped as empty
    assert ag.emptied == 1 and r.m == 2
    # every reduced point: lifted feasibility == reduced feasibility, objective identical, project(lift(y)) == y
    grid = [np.arange(r.lb[0], r.ub[0] + 1), np.linspace(r.lb[1], r.ub[1], 7), np.arange(r.lb[2], r.ub[2] + 1)]
    n_feas = 0
    for y in itertools.product(*grid):
        y = np.array(y, float)
        x = ag.lift(y)
        assert np.allclose(ag.project(x), y)
        assert p.check(x).feasible == r.check(y).feasible
        n_feas += r.check(y).feasible
        assert abs(p.objective(x) - r.objective(y)) < 1e-9
    assert n_feas > 0
    # and every original feasible point (enumerated over the integers, continuous set by its ties) projects feasibly
    for x0, g7 in itertools.product((0, 1), range(-3, 6)):
        x = np.array([x0, x0, 1 - x0, 1 - x0, 0.6, 0.1, 2 - g7, g7], float)
        if p.check(x).feasible:
            y = ag.project(x)
            assert r.check(y).feasible and np.allclose(ag.lift(y), x)


def test_integer_root_and_bounds():
    p = chain_problem()
    ag = aggregate(p)
    r = ag.reduced
    col = {int(rt): k for k, rt in enumerate(ag.root)}
    k_bin = col[int(ag.root[ag.rep[0]])]
    assert r.isint[k_bin] and r.lb[k_bin] == 0 and r.ub[k_bin] == 1
    k_c = ag.rep[4]
    assert not r.isint[k_c]
    # c4 = c5 + 0.5 with c4 in [0, 2], c5 in [0, 1.2]: the root's interval is the intersection
    lo, hi = r.lb[k_c], r.ub[k_c]
    x_lo, x_hi = ag.lift(np.where(np.arange(r.n) == k_c, lo, 0.0)), ag.lift(np.where(np.arange(r.n) == k_c, hi, 0.0))
    for x in (x_lo, x_hi):
        assert 0 - 1e-12 <= x[4] <= 2 + 1e-12 and 0 - 1e-12 <= x[5] <= 1.2 + 1e-12
    k_g = ag.rep[6]
    assert r.isint[k_g] and r.lb[k_g] >= -3 and r.ub[k_g] <= 5


def test_fractional_offset_between_integers_is_refused_and_nothing_to_do_is_none():
    A = sps.csr_matrix(np.array([[1.0, -1.0]]))
    p = from_arrays(A, [0.5], [0.5], [0, 0], [3, 3], [1, 1], [True, True])
    assert aggregate(p) is None                          # x0 - x1 = 0.5 cannot hold between integers: refused
    q = from_arrays(sps.csr_matrix(np.array([[1.0, 2.0]])), [1], [1], [0, 0], [3, 3], [1, 1], [True, True])
    assert aggregate(q) is None                          # unequal magnitudes: not a tie


def test_infeasible_chain_returns_none():
    A = sps.csr_matrix(np.array([[1.0, -1.0], [1.0, 0.0]]))
    # x1 = x0 + 0 with x0 in [0, 1], x1 in [2, 3]: empty interval
    p = from_arrays(A, [0, -np.inf], [0, np.inf], [0, 2], [1, 3], [0, 0], [True, True])
    assert aggregate(p) is None
