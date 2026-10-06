"""Parent selection: comparator order, tournament frequencies, rank pools, distinct parents."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from itrea.config import SelectionCfg
from itrea.ea.population import NO_RANK, R0, R1, R2
from itrea.ea.selection import order_position, select_parents


def _cfg(**kw) -> SelectionCfg:
    return SelectionCfg(**kw)


# ------------------------------------------------------------------------------------------------ comparator order
def test_itrea_order_rank_before_key():
    # Rank 1 < Rank 0 < Rank 2 < unranked, whatever the keys
    rank = np.array([R2, R0, R1, NO_RANK])
    key = np.array([-100.0, 5.0, 50.0, -1e9])
    viol = np.array([0.0, 3.0, 0.0, 0.0])
    feas = np.array([True, False, True, True])
    pos = order_position("itrea", rank, key, viol, feas)
    assert list(np.argsort(pos)) == [2, 1, 0, 3]


def test_itrea_ties_key_then_violation():
    rank = np.array([R2, R2, R2, R2])
    key = np.array([1.0, 1.0, 0.5, 1.0])
    viol = np.array([2.0, 1.0, 9.0, 1.0])
    pos = order_position("itrea", rank, key, viol, np.zeros(4, bool))
    assert pos[2] < pos[1] < pos[0]
    assert pos[1] == pos[3]                           # exact tie -> same position (broken by a coin)


def test_deb_order():
    key = np.array([10.0, 5.0, -50.0, -60.0, np.nan])
    viol = np.array([0.0, 0.0, 1.0, 2.0, 0.5])
    feas = np.array([True, True, False, False, False])
    pos = order_position("deb", np.full(5, R2), key, viol, feas)
    # feasible by key, then infeasible by V (NaN key counts as worst only within the V tie)
    assert list(np.argsort(pos, kind="stable")) == [1, 0, 4, 2, 3]


def test_nan_key_is_worst():
    rank = np.array([R2, R2])
    pos = order_position("itrea", rank, np.array([np.nan, 1e300]), np.zeros(2), np.zeros(2, bool))
    assert pos[1] < pos[0]


# ------------------------------------------------------------------------------------------- tournament frequencies
def _win_share(cfg, rank, key, viol, feas, who, n=40000, seed=0):
    ia, ib = select_parents(rank, key, viol, feas, n, cfg, np.random.default_rng(seed))
    return float(np.mean(ia == who))


def test_k2_two_individuals_better_wins_three_quarters():
    # with replacement: the worse one wins only when drawn twice (1/4)
    rank = np.array([R0, R1])
    share = _win_share(_cfg(tournament_k=2, comparator="itrea"), rank, np.array([-5.0, 3.0]),
                       np.array([1.0, 0.0]), np.array([False, True]), who=1)
    assert abs(share - 0.75) < 0.01


def test_k1_is_uniform():
    rank = np.array([R1, R0, R2, R2])
    ia, _ = select_parents(rank, np.arange(4.0), np.zeros(4), np.ones(4, bool), 40000,
                           _cfg(tournament_k=1), np.random.default_rng(1))
    freq = np.bincount(ia, minlength=4) / ia.size
    assert np.all(np.abs(freq - 0.25) < 0.01)


def test_exact_tie_broken_fairly():
    rank = np.array([R2, R2])
    share = _win_share(_cfg(tournament_k=2), rank, np.array([1.0, 1.0]), np.array([0.5, 0.5]),
                       np.zeros(2, bool), who=0)
    assert abs(share - 0.5) < 0.01


def test_stochastic_ranking_probabilities():
    # individual 0: feasible, worse key; individual 1: infeasible, better key
    rank = np.array([R1, R0])
    key, viol, feas = np.array([10.0, 1.0]), np.array([0.0, 2.0]), np.array([True, False])
    for p_f, expect in [(0.0, 0.25), (1.0, 0.75), (0.45, 0.25 + 0.5 * 0.45)]:
        share = _win_share(_cfg(tournament_k=2, comparator="stochastic", p_f=p_f), rank, key, viol, feas, who=1)
        assert abs(share - expect) < 0.012, (p_f, share)


def test_stochastic_both_feasible_is_by_key():
    rank = np.array([R2, R2])
    share = _win_share(_cfg(tournament_k=2, comparator="stochastic", p_f=0.0), rank, np.array([2.0, 1.0]),
                       np.zeros(2), np.ones(2, bool), who=1)
    assert abs(share - 0.75) < 0.01


def test_deb_infeasible_never_beats_feasible_in_pair():
    rank = np.array([R1, R0])
    share = _win_share(_cfg(tournament_k=2, comparator="deb"), rank, np.array([10.0, 1.0]), np.array([0.0, 2.0]),
                       np.array([True, False]), who=1)
    assert abs(share - 0.25) < 0.01


# ------------------------------------------------------------------------------------------------------- pools
def test_rank_pools_follow_weights():
    rank = np.array([R0] * 3 + [R1] * 3 + [R2] * 10)
    N = rank.size
    key = np.linspace(0, 1, N)
    cfg = _cfg(tournament_k=1, parent_rank_weights=[0.2, 0.5, 0.3])
    ia, ib = select_parents(rank, key, np.zeros(N), np.ones(N, bool), 50000, cfg, np.random.default_rng(3))
    share = np.array([np.mean(rank[ia] == r) for r in (R0, R1, R2)])
    assert np.all(np.abs(share - [0.2, 0.5, 0.3]) < 0.01), share


def test_empty_pool_weights_renormalised():
    rank = np.array([R1] * 3 + [R2] * 5)            # no Rank 0
    cfg = _cfg(tournament_k=1, parent_rank_weights=[0.5, 0.25, 0.25])
    ia, _ = select_parents(rank, np.arange(8.0), np.zeros(8), np.ones(8, bool), 40000, cfg,
                           np.random.default_rng(4))
    assert abs(np.mean(rank[ia] == R1) - 0.5) < 0.01


def test_all_weighted_pools_empty_uses_whole_population():
    rank = np.full(6, NO_RANK)
    cfg = _cfg(tournament_k=1, parent_rank_weights=[1.0, 0.0, 0.0])
    ia, ib = select_parents(rank, np.arange(6.0), np.zeros(6), np.ones(6, bool), 30000, cfg,
                            np.random.default_rng(5))
    assert set(np.unique(ia)) == set(range(6))


def test_single_member_pool_second_parent_differs():
    rank = np.array([R1, R2, R2, R2])
    cfg = _cfg(tournament_k=2, parent_rank_weights=[0.0, 1.0, 0.0])
    ia, ib = select_parents(rank, np.arange(4.0), np.zeros(4), np.ones(4, bool), 2000, cfg,
                            np.random.default_rng(6))
    assert np.all(ia == 0) and np.all(ib != ia)


# ---------------------------------------------------------------------------------------------- general properties
@settings(max_examples=60, deadline=None)
@given(N=st.integers(1, 40), n_pairs=st.integers(0, 50), k=st.integers(1, 5),
       comp=st.sampled_from(["itrea", "deb", "stochastic"]), pools=st.booleans(), seed=st.integers(0, 2**31))
def test_select_parents_properties(N, n_pairs, k, comp, pools, seed):
    rs = np.random.default_rng(seed)
    rank = rs.choice([NO_RANK, R0, R1, R2], size=N)
    key = rs.normal(size=N)
    key[rs.random(N) < 0.1] = np.nan
    viol = np.abs(rs.normal(size=N))
    feas = rs.random(N) < 0.5
    w = list(rs.random(3)) if pools else None
    if w is not None and sum(w) == 0:
        w = None
    cfg = _cfg(tournament_k=k, comparator=comp, parent_rank_weights=w)
    ia, ib = select_parents(rank, key, viol, feas, n_pairs, cfg, np.random.default_rng(seed))
    assert ia.dtype == np.int64 and ib.dtype == np.int64
    assert ia.shape == ib.shape == (n_pairs,)
    assert np.all((ia >= 0) & (ia < N)) and np.all((ib >= 0) & (ib < N))
    if N > 1:
        assert np.all(ia != ib)


def test_large_k_picks_best_itrea():
    rs = np.random.default_rng(7)
    N = 30
    rank = rs.choice([R0, R2], size=N)
    rank[17] = R1                                    # the only Rank 1 member wins any tournament it enters
    key = rs.normal(size=N)
    ia, _ = select_parents(rank, key, np.zeros(N), np.ones(N, bool), 3000, _cfg(tournament_k=200),
                           np.random.default_rng(8))
    assert np.mean(ia == 17) > 0.99


def test_empty_population_raises():
    with pytest.raises(ValueError):
        select_parents(np.empty(0), np.empty(0), np.empty(0), np.empty(0, bool), 3, _cfg(), np.random.default_rng())
