"""Regression tests for the engine review of 2026-10-06 (findings F3, F6, F7; F1 / F4 are in `test_ranks.py` and in
`test_engine.run_checked`, F2 in `test_ea_run.py`).

F3 — the EA's work inside a generation is bounded: every local search polls `should_stop` and returns early, and the
engine drops a generation whose repair the budget or a stop cut short.
F6 — a solver rejection of the EA's incumbent reverts the incumbent to the newest one not rejected, so later solutions
are recorded and posted again; a re-created copy of a rejected vector is refused.
F7 — a demoted Rank 1 member leaves Rank 1 at once; the survival's float64 re-check loop runs until Rank 1 holds only
re-checked members.

Toy models only (`itrea.benchmarks.toy`); GPU tests (marker `gpu`) use a 1 GB CuPy pool.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

from itrea.backend import get_backend
from itrea.benchmarks import toy
from itrea.config import EAConfig, RepairCfg
from itrea.ea.engine import Engine
from itrea.ea.evaluate import to_device
from itrea.ea.ls_dense import DenseLS
from itrea.ea.ls_fj_cpu import FJCpu
from itrea.ea.population import R1, R2
from itrea.ea.repair import LocalSearch
from itrea.solvers.common import vector_hash

SMALL = {"population.size": 48, "population.offspring_per_gen": 48, "gpu_mem_gb": 1.0}
BK = get_backend("numpy")


def cfg_for(backend: str = "numpy", **over) -> EAConfig:
    d = dict(SMALL, backend=backend)
    d.update(over)
    return EAConfig().with_overrides(d)


def gpu_backend():
    pytest.importorskip("cupy")
    try:
        return get_backend("cupy", gpu_mem_gb=1.0)
    except Exception as e:  # pragma: no cover - no device
        pytest.skip(f"no CuPy device: {e}")


def random_start(p, k: int, seed: int = 0) -> np.ndarray:
    rs = np.random.default_rng(seed)
    lo = np.where(np.isfinite(p.lb), p.lb, -10.0)
    hi = np.where(np.isfinite(p.ub), p.ub, 10.0)
    X = lo + (hi - lo) * rs.random((k, p.n))
    X[:, p.ints] = np.round(X[:, p.ints])
    return X


class Countdown:
    """should_stop that answers False `n` times, then True."""

    def __init__(self, n: int):
        self.n = n
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls > self.n


class MiniMailbox:
    """A solver partner in memory: scripted verdicts on every posted row, a settable stop flag."""

    def __init__(self, verdict=None):
        self.posts: list = []
        self._verdicts: list[dict] = []
        self.verdict = verdict
        self.stop = False
        outer = self

        class _Q:
            def poll(self):
                return None

        class _To:
            def post(_s, X, obj, origin=None):
                outer.posts.append((np.asarray(X).copy(), np.asarray(obj).copy()))
                seq = len(outer.posts)
                if outer.verdict is not None:
                    for r in range(len(obj)):
                        outer._verdicts.append(dict(seq=seq, t=0.0, obj=float(obj[r]),
                                                    accepted=bool(outer.verdict(seq, X[r], obj[r])),
                                                    note=f'{{"row": {r}}}'))
                return seq

        self.to_ea, self.info, self.lp, self.to_solver = _Q(), _Q(), _Q(), _To()

    def stopped(self):
        return self.stop

    def verdicts(self):
        return list(self._verdicts)


# ------------------------------------------------------------------------------------------ F3: local searches
@pytest.mark.parametrize("name", ["knapsack", "set_cover", "mixed"])
def test_fj_cpu_stop_polling_does_not_change_results(name):
    """Slicing the rows (should_stop given) gives the same rows as one call: every row has its own seed."""
    p = {"knapsack": lambda: toy.knapsack(30), "set_cover": lambda: toy.set_cover(40, 25), "mixed": toy.mixed}[name]()
    dp = to_device(p, BK)
    X0 = random_start(p, 24, seed=3)
    out = []
    for stop in (None, lambda: False):
        X = X0.copy()
        fj = FJCpu(dp, RepairCfg(), seed=7)
        c = fj.run(X, should_stop=stop)
        out.append((X, fj.last_status.copy(), c))
    np.testing.assert_array_equal(out[0][0], out[1][0])
    np.testing.assert_array_equal(out[0][1], out[1][1])
    assert out[0][2]["moves"] == out[1][2]["moves"] and out[1][2]["interrupted"] == 0


def test_fj_cpu_stop_leaves_the_remaining_rows_untouched():
    p = toy.set_cover(40, 25)
    dp = to_device(p, BK)
    X0 = random_start(p, 32, seed=4)
    X = X0.copy()
    fj = FJCpu(dp, RepairCfg(), seed=1)
    c = fj.run(X, should_stop=Countdown(2))                        # two slices, then stop
    st = fj.last_status
    done = st != 0
    assert 1 <= done.sum() < 32 and np.all(done[:done.sum()])       # a prefix of the rows was processed
    np.testing.assert_array_equal(X[~done], X0[~done])              # the rest is exactly as it was
    assert c["interrupted"] == int((~done).sum())
    assert c["finished_feasible"] + c["finished_stall"] + c["finished_budget"] == 32
    X = X0.copy()
    c = FJCpu(dp, RepairCfg(), seed=1).run(X, should_stop=lambda: True)
    np.testing.assert_array_equal(X, X0)
    assert c["interrupted"] == 32 and c["moves"] == 0


@pytest.mark.parametrize("method", ["fj", "dense", "none"])
def test_dispatcher_forwards_should_stop(method):
    p = toy.knapsack(30)
    dp = to_device(p, BK)
    X0 = random_start(p, 16, seed=5)
    ls = LocalSearch(BK, dp, RepairCfg(method=method), 1e-9, 64 << 20, 0)
    X = X0.copy()
    c = ls.run(X, should_stop=lambda: True)
    assert c["finished_feasible"] + c["finished_stall"] + c["finished_budget"] == 16
    if method != "none":                                            # the projection is one pass: nothing to cut
        np.testing.assert_array_equal(X, X0)
        assert c["interrupted"] == 16


def test_dense_stop_inside_a_chunk():
    p = toy.set_cover(40, 25)
    dp = to_device(p, BK)
    X0 = random_start(p, 16, seed=6)
    ref = X0.copy()
    DenseLS(dp, RepairCfg(method="dense"), 1e-9, 64 << 20, 0).run(ref, should_stop=lambda: False)
    X = X0.copy()
    DenseLS(dp, RepairCfg(method="dense"), 1e-9, 64 << 20, 0).run(X)
    np.testing.assert_array_equal(X, ref)                           # polling alone changes nothing
    X = X0.copy()
    c = DenseLS(dp, RepairCfg(method="dense"), 1e-9, 64 << 20, 0).run(X, should_stop=Countdown(2))
    assert c["steps"] <= 16 and c["interrupted"] >= 1               # one step of one chunk, then cut
    assert c["finished_feasible"] + c["finished_stall"] + c["finished_budget"] == 16


@pytest.mark.gpu
def test_fj_gpu_stop():
    bk = gpu_backend()
    import cupy as cp
    from itrea.ea.ls_fj_gpu import ST_ACTIVE, FJGpu
    p = toy.set_cover(200, 120, density=0.05, seed=1)
    dp = to_device(p, bk)
    X0 = random_start(p, 24, seed=7).astype(np.int8)                # int8 genes: the non-direct write-back path
    cfg = RepairCfg(max_steps=2000, stall_steps=100)
    res = []
    for stop in (None, lambda: False):
        X = cp.asarray(X0)
        fj = FJGpu(dp, cfg, 1e-9, 64 << 20, seed=2)
        fj.run(X, should_stop=stop)
        res.append(cp.asnumpy(X))
    np.testing.assert_array_equal(res[0], res[1])
    X = cp.asarray(X0)
    c = FJGpu(dp, cfg, 1e-9, 64 << 20, seed=2).run(X, should_stop=lambda: True)
    np.testing.assert_array_equal(cp.asnumpy(X), X0)
    assert c["interrupted"] == 24 and c["launches"] == 0
    X = cp.asarray(X0)
    fj = FJGpu(dp, cfg, 1e-9, 64 << 20, seed=2)
    c = fj.run(X, should_stop=Countdown(1))                         # stop after the first launch
    assert c["launches"] == 1
    assert c["interrupted"] == int(np.sum(fj.last_status == ST_ACTIVE))
    assert c["finished_feasible"] + c["finished_stall"] + c["finished_budget"] == 24
    Xh = cp.asnumpy(X).astype(float)                                # written back, still integral and in bounds
    assert np.all(Xh >= p.lb) and np.all(Xh <= p.ub)
    moved = fj.last_moves > 0
    if moved.any():                                                 # the chunk in progress was written back
        assert np.any(Xh[moved] != X0[moved].astype(float))


# --------------------------------------------------------------------------------------------- F3: the engine
def _engine(p, cfg, seed=0, mailbox=None) -> Engine:
    eng = Engine(p, cfg, seed, mailbox=mailbox)
    eng.prepare()
    return eng


@pytest.mark.parametrize("backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)])
def test_generation_cut_short_at_the_deadline(backend):
    if backend == "cupy":
        gpu_backend()
    p = toy.set_cover(40, 25)
    eng = _engine(p, cfg_for(backend), seed=1)
    eng.begin(time.monotonic(), 3600.0)
    ids0 = eng.pop.ids.copy()
    children0 = eng.c["children"]
    eng.deadline = time.monotonic() - 1e-3                          # the budget ends during the next repair
    eng.step()
    assert eng.stop_reason == "budget" and eng.c["aborted_generations"] == 1 and eng.gen == 1
    np.testing.assert_array_equal(eng.pop.ids, ids0)                # the cut generation changed nothing
    assert eng.c["children"] == children0 and eng.c.get("ls_interrupted", 0) >= 1
    out = eng.finish()
    assert out["exit_reason"] == "budget" and out["max_generation_s"] > 0 and out["aborted_generations"] == 1


def test_stop_flag_is_read_inside_the_search():
    p = toy.set_cover(40, 25)
    mb = MiniMailbox()
    eng = _engine(p, cfg_for(**{"exchange.poll_s": 3600.0}), seed=2, mailbox=mb)   # no exchange tick in step()
    eng.begin(time.monotonic(), 3600.0)
    eng.next_poll = math.inf
    eng._next_stop_poll = -math.inf                                 # as if STOP_POLL_S had passed since the last read
    mb.stop = True
    eng.step()
    assert eng.stop_reason == "stop" and eng.c["aborted_generations"] == 1
    assert eng.finish()["exit_reason"] == "stop"


# ------------------------------------------------------------------------------------------- F6: rejections
def test_rejected_incumbent_is_reverted_and_posting_resumes():
    p = toy.knapsack(30)
    mb = MiniMailbox(verdict=lambda seq, x, obj: seq != 1)          # the first post is rejected, later ones accepted
    eng = _engine(p, cfg_for(**{"exchange.poll_s": 1e-6}), seed=6, mailbox=mb)
    eng.begin(time.monotonic(), 3600.0)
    assert len(mb.posts) == 1 and eng.best_x is not None and int(eng.h_feas.sum()) >= 2
    x_rej = mb.posts[0][0][0]
    key_rej = eng.best_key
    iid = eng.posted[1][0]
    cut0 = eng.cutoff
    eng._exchange_in()                                              # reads the verdict on seq 1
    assert eng.cutoff is not None and eng.cutoff >= cut0            # the cutoff follows the demotion at once
    assert eng.c["solver_rejections"] == 1 and eng.c["incumbents_reverted"] == 1
    assert eng.best_x is None and eng.best_key == math.inf and eng.best_obj is None
    pos = np.flatnonzero(eng.pop.ids == iid)
    assert pos.size and not eng.h_feas[pos[0]]
    assert eng.pop.rank[pos[0]] == R2 and iid not in eng.r1_ids     # F7: out of Rank 1 at once
    eng._update_incumbent()                                         # the next best feasible becomes the incumbent
    assert len(mb.posts) == 2 and vector_hash(mb.posts[1][0][0]) != vector_hash(x_rej)
    assert eng.best_x is not None and not np.array_equal(eng.best_x, x_rej) and eng.best_key >= key_rej
    # a copy of the rejected vector (new id) is refused by the float64 check, so it is never re-posted
    bk = eng.bk
    X = bk.xp.asarray(x_rej[None, :]).astype(eng.gene_dtype)
    ev = eng._evaluate(X)
    copy = eng._make_pop(X, ev, np.full(1, -1, np.int8), np.array([10 ** 9]), np.full((1, 2), -1, np.int64),
                         np.zeros(1, np.uint16), eng.gen)
    ok, obj, _ = eng._check_candidate(copy, 0)
    assert not ok and eng.c["rejected_copies"] == 1 and obj == pytest.approx(p.objective(x_rej))
    # the run continues normally and every incumbent after the rejection is a different vector
    for _ in range(5):
        eng.step()
    out = eng.finish()
    assert out["best_x"] is not None and vector_hash(out["best_x"]) != vector_hash(x_rej)
    assert all(vector_hash(X[0]) != vector_hash(x_rej) for X, _ in mb.posts[1:])


def test_reject_older_post_reverts_only_when_on_top():
    """Stack rule: a rejected incumbent below a newer accepted one changes nothing until the newer one is rejected
    too; then both go and the incumbent falls back past them."""
    p = toy.knapsack(30)
    eng = _engine(p, cfg_for(), seed=8)
    eng.begin(time.monotonic(), 3600.0)
    x = np.zeros(p.n)
    eng.inc_stack = [(1, -10.0, 10.0, x), (2, -20.0, 20.0, x + 1), (3, -30.0, 30.0, x + 2)]
    eng.best_key, eng.best_obj, eng.best_x = -30.0, 30.0, x + 2
    eng.rejected_ids.add(2)
    eng._revert_incumbent(2)
    assert eng.best_key == -30.0 and eng.c["incumbents_reverted"] == 0
    eng.rejected_ids.add(3)
    eng._revert_incumbent(3)
    assert eng.best_key == -10.0 and eng.best_obj == 10.0 and len(eng.inc_stack) == 1
    assert eng.c["incumbents_reverted"] == 1


# --------------------------------------------------------------------------------------- F7: survival re-check
def test_survival_recheck_runs_until_rank1_is_clean():
    """Twelve consecutive false-feasible Rank 1 entrants (more than the old fixed 8 rounds): the survival keeps
    demoting and repeating until Rank 1 holds only members that passed the float64 re-check."""
    p = toy.knapsack(30)
    eng = _engine(p, cfg_for(**{"ranks.y": 0.01}), seed=9)          # n1 = 1: one new entrant per round
    orig = eng._check_candidate
    failed: set[int] = set()

    def patched(popn, i):
        ok, obj, x = orig(popn, i)
        iid = int(popn.ids[i])
        if ok and len(failed) < 12 and iid not in failed:
            failed.add(iid)
            eng.checked[iid] = (False, obj)
            return False, obj, x
        return ok, obj, x

    eng._check_candidate = patched
    eng.begin(time.monotonic(), 3600.0)
    assert len(failed) == 12 and eng.c["false_feasible"] == 12
    r1 = eng.pop.rank == R1
    assert r1.sum() == 1 and np.all(eng.h_feas[r1])
    assert all(eng.checked[int(i)][0] for i in eng.pop.ids[r1])
    assert not set(int(i) for i in eng.pop.ids[r1]) & failed
