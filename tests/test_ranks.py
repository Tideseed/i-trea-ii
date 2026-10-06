"""Three-rank survival (`itrea.ea.ranks.three_rank_survival`) called directly, on hand-made candidate sets.

Review 2026-10-06: F1 — Rank 1 is protected every generation, also when the number of feasible candidates falls between
generations (the ratchet `prev_r1`); F4 — in the feasibility phase (no incumbent) the parents compete with the
children by (V, key), so the least-violated individual is never lost to generational replacement. No solver, no GPU.
"""
from __future__ import annotations

import numpy as np

from itrea.config import RankCfg
from itrea.ea.population import R1, R2
from itrea.ea.ranks import three_rank_survival


def _gen_g():
    """Generation g, P = 32, defaults (x = y = 0.1, generational): 40 feasible candidates with keys 0..39 (32 old
    members, 8 new children)."""
    cfg = RankCfg()
    P = 32
    key = np.arange(40, dtype=float)
    ids = np.arange(40)
    is_new = np.zeros(40, bool)
    is_new[32:] = True
    s = three_rank_survival(key, np.zeros(40), np.ones(40, bool), P, cfg, ids=ids, is_new=is_new)
    return cfg, P, key, ids, s


def _gen_g1(cfg, P, key, ids, s, use_prev: bool, child_keys=None):
    """Generation g+1: the survivors of g plus 32 new children (infeasible unless child_keys is given)."""
    nchild = 32
    sk, sid = key[s.idx], ids[s.idx]
    if child_keys is None:
        ck, cv, cf = 100 + np.arange(nchild, dtype=float), np.ones(nchild), np.zeros(nchild, bool)
    else:
        ck = np.asarray(child_keys, float)
        nchild = ck.size
        cv, cf = np.zeros(nchild), np.ones(nchild, bool)
    K = np.r_[sk, ck]
    V = np.r_[np.zeros(s.idx.size), cv]
    F = np.r_[np.ones(s.idx.size, bool), cf]
    I = np.r_[sid, 1000 + np.arange(nchild)]
    new = np.r_[np.zeros(s.idx.size, bool), np.ones(nchild, bool)]
    prev = np.isin(I, sid[s.rank == R1]) if use_prev else None
    return three_rank_survival(K, V, F, P, cfg, ids=I, is_new=new, prev_r1=prev), I


def test_rank1_survives_when_the_feasible_count_falls():
    """F1: in g, n1 = ceil(0.1 * 32) = 4 (keys 0..3) and 12 individuals survive; in g+1 only those 12 are feasible,
    so ceil(0.1 * 12) = 2 — without the ratchet ids 2 and 3 die although nothing better displaced them."""
    cfg, P, key, ids, s = _gen_g()
    r1_g = ids[s.idx[s.rank == R1]]
    assert sorted(r1_g.tolist()) == [0, 1, 2, 3]
    s_old, I = _gen_g1(cfg, P, key, ids, s, use_prev=False)
    assert s_old.n1 == 2                                             # the defect the ratchet removes
    s2, I = _gen_g1(cfg, P, key, ids, s, use_prev=True)
    r1_now = set(I[s2.idx[s2.rank == R1]].tolist())
    assert s2.n1 == 4 and r1_now == {0, 1, 2, 3}
    assert set(r1_g.tolist()) <= set(I[s2.idx].tolist())


def test_rank1_member_leaves_only_when_displaced_by_better_ones():
    """With the ratchet, a previous Rank 1 member leaves only when n1 strictly better feasible keys exist."""
    cfg, P, key, ids, s = _gen_g()
    # two better children (keys -2, -1): Rank 1 = {-2, -1, 0, 1}; the members with keys 2 and 3 were displaced
    s2, I = _gen_g1(cfg, P, key, ids, s, use_prev=True, child_keys=[-2.0, -1.0] + [200.0 + i for i in range(30)])
    r1 = s2.idx[s2.rank == R1]
    K = np.r_[key[s.idx], [-2.0, -1.0] + [200.0 + i for i in range(30)]]
    assert s2.n1 >= 4
    assert np.sort(K[r1])[:4].tolist() == [-2.0, -1.0, 0.0, 1.0]
    # an equal-key younger twin never pushes a previous member out (twins keep the oldest)
    s3, I3 = _gen_g1(cfg, P, key, ids, s, use_prev=True, child_keys=[0.0, 1.0, 2.0, 3.0] + [200.0 + i for i in range(28)])
    assert set(I3[s3.idx[s3.rank == R1]].tolist()) == {0, 1, 2, 3}


def test_rank1_ratchet_with_y_zero_is_empty():
    """y = 0 (ablation A5y0): no Rank 1 at all, whatever prev_r1 says."""
    cfg = RankCfg(y=0.0)
    key = np.arange(20, dtype=float)
    prev = np.zeros(20, bool)
    prev[:3] = True
    s = three_rank_survival(key, np.zeros(20), np.ones(20, bool), 16, cfg, ids=np.arange(20),
                            is_new=np.ones(20, bool), prev_r1=prev)
    assert s.n1 == 0 and not np.any(s.rank == R1)


def test_rank1_ratchet_repeated_generations():
    """Random generations: with the ratchet, every lost Rank 1 member has at least n1(g) better-or-equal keys in the
    new Rank 1, and |Rank 1| never shrinks (nothing is demoted here)."""
    rng = np.random.default_rng(5)
    cfg = RankCfg()
    P = 40
    K = rng.normal(size=P)
    V = np.where(rng.random(P) < 0.5, 0.0, rng.random(P))
    I = np.arange(P)
    nxt = P
    r1_prev: dict[int, float] = {}
    for g in range(60):
        nchild = P
        ck = rng.normal(loc=0.5, size=nchild)
        feas_share = 0.6 if g % 3 else 0.05                      # the feasible count falls every third generation
        cv = np.where(rng.random(nchild) < feas_share, 0.0, rng.random(nchild) + 0.01)
        Kc, Vc = np.r_[K, ck], np.r_[V, cv]
        Ic = np.r_[I, nxt + np.arange(nchild)]
        nxt += nchild
        Fc = Vc == 0.0
        new = np.r_[np.zeros(K.size, bool), np.ones(nchild, bool)]
        prev = np.isin(Ic, np.fromiter(r1_prev, np.int64, len(r1_prev)))
        s = three_rank_survival(Kc, Vc, Fc, P, cfg, ids=Ic, is_new=new, prev_r1=prev)
        r1_keys = Kc[s.idx[s.rank == R1]]
        survivors = set(Ic[s.idx].tolist())
        assert r1_keys.size >= len(r1_prev)
        for iid, k in r1_prev.items():
            if iid not in survivors:
                assert int(np.sum(r1_keys <= k)) >= len(r1_prev)
        r1_prev = {int(Ic[i]): float(Kc[i]) for i in s.idx[s.rank == R1]}
        K, V, I = Kc[s.idx], Vc[s.idx], Ic[s.idx]


def test_feasibility_phase_keeps_the_least_violated_parent():
    """F4: no incumbent, generational Rank 2. Old population V = [0.01, 1, ..., 1], new children V = 5..12: the
    feasibility phase ranks ALL candidates by (V, key), so the old members (the least violated first) survive."""
    cfg = RankCfg()
    P = 8
    key = np.zeros(16)
    viol = np.r_[[0.01, 1, 1, 1, 1, 1, 1, 1], np.arange(5, 13, dtype=float)]
    new = np.r_[np.zeros(8, bool), np.ones(8, bool)]
    s = three_rank_survival(key, viol, np.zeros(16, bool), P, cfg, ids=np.arange(16), is_new=new)
    assert s.feasibility_phase
    assert 0 in s.idx and s.idx[0] == 0                              # the least violated, first
    assert np.all(viol[s.idx] <= 1.0) and np.all(s.rank == R2)


def test_generational_rank2_unchanged_once_an_incumbent_exists():
    """F4 changes nothing after the first feasible solution: an old infeasible Rank 2 member (not super-optimal)
    dies under generational replacement even when it is less violated than every child."""
    cfg = RankCfg()
    P = 8
    # one feasible old member (key 0 -> Rank 1); old infeasible members with key 5 (worse than inc: Rank 2), V small
    key = np.r_[0.0, np.full(7, 5.0), np.full(8, 6.0)]
    viol = np.r_[0.0, np.full(7, 0.01), np.arange(1, 9, dtype=float)]
    feas = np.r_[True, np.zeros(15, bool)]
    new = np.r_[np.zeros(8, bool), np.ones(8, bool)]
    s = three_rank_survival(key, viol, feas, P, cfg, ids=np.arange(16), is_new=new)
    assert not s.feasibility_phase
    assert set(s.idx.tolist()) == {0, *range(8, 15)}                 # Rank 1 + the 7 best children; old Rank 2 died
    # the "plus" replacement lets the old ones compete
    sp = three_rank_survival(key, viol, feas, P, RankCfg(rank2_replacement="plus"), ids=np.arange(16), is_new=new)
    assert set(range(1, 8)) <= set(sp.idx.tolist())
