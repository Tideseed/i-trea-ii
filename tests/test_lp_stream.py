"""D-011 — the solver -> EA stream of good infeasible node LP points (Berk 2026-10-06).

Config (hash rule, validation), runner flags, the mailbox channel, the EA's admission (Berk's order: violation of the
nearest rounding, then objective; ordinary survival only) and SCIP's reader on a model that needs branching.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps

from itrea.benchmarks import toy
from itrea.config import EAConfig
from itrea.ea.population import ORIGIN_IMMIGRANT, ORIGIN_INJECTED, ORIGIN_REPAIR, ORIGIN_SEED_LP
from itrea.exchange.mailbox import Mailbox, Message
from itrea.experiments.pair import ProcSpec, RunError, solver_flags
from itrea.problem import from_arrays, load
from itrea.trace import read_events
from test_engine import FakeMailbox, SpyMailbox, check_invariants, cfg_for, prepared, run_checked

ROOT = Path(__file__).resolve().parents[1]
STREAM_BITS = ORIGIN_SEED_LP | ORIGIN_INJECTED


# ------------------------------------------------------------------------------------------------------ config
def test_disabled_section_keeps_every_hash():
    assert EAConfig.from_yaml(ROOT / "configs/itrea2-default.yaml").hash() == "9ac4a93bf93b"
    assert EAConfig.from_yaml(ROOT / "configs/itrea2-adaptive.yaml").hash() == "584ddee83747"
    base = EAConfig()
    off = base.with_overrides({"lp_stream.roundings": 9})          # inert while disabled
    assert off.hash() == base.hash()
    on = base.with_overrides({"lp_stream.enabled": True})
    assert on.hash() != base.hash()
    assert on.with_overrides({"lp_stream.roundings": 9}).hash() != on.hash()


@pytest.mark.parametrize("over", [{"lp_stream.enabled": True, "exchange.from_solver": False},
                                  {"lp_stream.enabled": True, "exchange.enabled": False},
                                  {"lp_stream.roundings": 0}, {"lp_stream.pool": 0}, {"lp_stream.share": 0.0},
                                  {"lp_stream.quantile": 1.5}, {"lp_stream.per_tick": 1.5}])
def test_validation(over):
    with pytest.raises(ValueError):
        EAConfig().with_overrides(over).validate()


def test_runner_passes_the_stream_flags_to_scip_only():
    cfg = EAConfig().with_overrides({"lp_stream.enabled": True, "lp_stream.share": 0.03, "lp_stream.ring": 2})
    f = solver_flags(ProcSpec("scip", "solver", exchange=True), cfg)
    assert "--send" in f and "--lp-stream" in f
    assert f[f.index("--lp-stream-share") + 1] == "0.03" and f[f.index("--lp-stream-ring") + 1] == "2"
    assert "--lp-stream" not in solver_flags(ProcSpec("scip", "solver", exchange=True), EAConfig())
    assert solver_flags(ProcSpec("scip", "solver", exchange=False), cfg) == ["--inert"]
    with pytest.raises(RunError):
        solver_flags(ProcSpec("highs", "solver", exchange=True), cfg)


# ----------------------------------------------------------------------------------------------------- mailbox
def test_file_channel_carries_point_ids(tmp_path):
    mb = Mailbox(tmp_path / "mb", "runner", create=True, run_id="x")
    X = np.arange(6, dtype=float).reshape(2, 3) / 4
    seq = mb.lps.post(X, np.array([1.0, 2.0]), origin=np.array([7, 8]))
    r = Mailbox(tmp_path / "mb", "ea", run_id="x")
    m = r.lps.poll()
    assert m.seq == seq and np.array_equal(m.X, X) and m.origin.tolist() == [7, 8]
    assert r.lps.poll() is None


# --------------------------------------------------------------------------------------------------------- EA
class StreamMailbox(FakeMailbox):
    """FakeMailbox plus an `lps` queue (one message per exchange tick)."""

    def __init__(self, lps=(), **kw):
        super().__init__(**kw)
        q = list(lps)
        self.lps_polls = 0
        outer = self

        class _Q:
            def poll(_s):
                outer.lps_polls += 1
                return q.pop(0) if q else None
        self.lps = _Q()


def knapsack_lp_points(p, k: int) -> np.ndarray:
    """k distinct fractional LP points of toy.knapsack: the greedy LP optimum with item j (the j-th best ratio) fixed
    to 0 — like node LPs after a branching on it."""
    w = np.asarray(p.A.toarray()[0], float)
    v = np.asarray(p.c, float)
    W = float(p.ru[0])
    order = np.argsort(-v / w)
    pts = []
    for j in range(k):
        x, cap = np.zeros(p.n), W
        for i in order:
            if i == order[j]:
                continue
            take = min(1.0, cap / w[i])
            x[i] = take
            cap -= take * w[i]
            if cap <= 1e-12:
                break
        pts.append(x)
    X = np.array(pts)
    assert np.all(np.abs(X - np.round(X)).max(axis=1) > 1e-6)          # each one fractional
    return X


def _msg(p, X, ids, seq):
    return Message(X=X, obj=np.array([p.objective(x) for x in X]), seq=seq, t=0.0, origin=np.asarray(ids))


def test_stream_points_ranked_and_turned_into_children():
    p = toy.knapsack(30)
    X = knapsack_lp_points(p, 3)
    # a ring: the second post repeats points 1 and 2 (must be ignored) and adds 3
    mb = StreamMailbox(lps=[_msg(p, X[:2], [1, 2], 1), _msg(p, X, [1, 2, 3], 2)])
    cfg = cfg_for(**{"lp_stream.enabled": True, "lp_stream.roundings": 4, "lp_stream.per_tick": 1,
                     "exchange.poll_s": 1e-6})
    eng = prepared(p, cfg, seed=3, mailbox=mb)
    run_checked(eng, 8)
    out = eng.finish()
    c = eng.c
    assert c["lps_msgs"] == 2 and c["lps_points"] == 3 and c["lps_dropped"] == 0
    pool = eng.lps_pool
    assert [(q["V"], q["key"]) for q in pool] == sorted((q["V"], q["key"]) for q in pool)   # Berk's order
    assert sorted(q["id"] for q in pool) == [1, 2, 3] and all(q["used"] for q in pool)
    # a stream point never becomes the EA's LP point (its source stays seeding.lp_source); all three gave children
    assert eng.lp_x is None and eng.lp_info["source"] == "solver"
    assert c["lps_used"] == 3 and c["lps_rows"] == 3 * 4 and c["lps_missed"] == 0
    assert out["lp_stream"]["pool"] == 3 and out["lp_stream"]["unused"] == 0
    gens = [e for e in read_events(eng.trace.path) if e["kind"] == "gen"] if getattr(eng.trace, "path", None) else []
    for g in gens:
        assert "rank0_stream" in g and g["rank0_stream"] <= eng.cfg.n_rank0_cap()


def test_stream_children_carry_the_stream_origin_and_are_repaired():
    p = toy.knapsack(30)
    X = knapsack_lp_points(p, 2)
    mb = StreamMailbox(lps=[_msg(p, X, [1, 2], 1)])
    cfg = cfg_for(**{"lp_stream.enabled": True, "lp_stream.roundings": 3, "exchange.poll_s": 1e-6})
    eng = prepared(p, cfg, seed=5, mailbox=mb)
    batches = []
    use = eng._lps_use

    def spy():
        n = len(eng.pending)
        use()
        batches.extend((b, eng.bk.asnumpy(b.X).astype(float).copy()) for b in eng.pending[n:] if b.src == "lps")
    eng._lps_use = spy
    eng.begin(time.monotonic(), 3600.0)
    eng.step()
    assert len(batches) == 2 and all(b.repair for b, _ in batches)       # per_tick 2: both points in one tick
    for (b, X0), q in zip(batches, eng.lps_pool):                        # best (V, key) first
        assert b.X.shape == (3, p.n)
        assert np.all((b.origin & ~np.uint16(ORIGIN_REPAIR)) == (STREAM_BITS | ORIGIN_IMMIGRANT))  # + REPAIR if moved
        assert np.array_equal(X0[0], q["y"])                             # row 0 = the nearest rounding, before repair
        assert np.all((X0 == 0) | (X0 == 1))
    assert eng.lp_x is None
    check_invariants(eng)
    assert eng.c["lps_used"] == 2 and eng.c["lps_rows"] == 6


def test_stream_ignores_malformed_and_disabled_never_reads_it():
    p = toy.knapsack(30)
    bad = Message(X=np.zeros((1, p.n + 1)), obj=np.zeros(1), seq=1, t=0.0, origin=np.array([1]))
    mb = StreamMailbox(lps=[bad])
    eng = prepared(p, cfg_for(**{"lp_stream.enabled": True, "exchange.poll_s": 1e-6}), seed=1, mailbox=mb)
    run_checked(eng, 3)
    assert eng.c["lps_points"] == 0
    mb2 = StreamMailbox(lps=[_msg(p, knapsack_lp_points(p, 1), [1], 1)])
    eng2 = prepared(p, cfg_for(**{"exchange.poll_s": 1e-6}), seed=1, mailbox=mb2)
    run_checked(eng2, 3)
    out = eng2.finish()
    assert mb2.lps_polls == 0 and "lp_stream" not in out and "lps_points" not in out


def test_one_way_arm_never_reads_the_stream():
    assert "lps" in SpyMailbox.FORBIDDEN


def test_used_points_leave_the_pool_first_and_missed_ids_are_counted():
    """Review of 770df17: with a full pool, new points worse than the best ever seen must still give children."""
    p = toy.knapsack(30)
    X = knapsack_lp_points(p, 9)
    cfg = cfg_for(**{"lp_stream.enabled": True, "lp_stream.pool": 7, "lp_stream.per_tick": 8,
                     "lp_stream.roundings": 1})                         # the best four stay; three free slots
    eng = prepared(p, cfg, seed=2, mailbox=StreamMailbox())
    eng.begin(time.monotonic(), 3600.0)
    for ids in ([1, 2, 3], [4, 5, 6], [7, 8, 9]):
        eng._lps_receive(_msg(p, X[np.array(ids) - 1], ids, ids[0]))
        eng._lps_use()
        assert len(eng.lps_pool) <= 7
    assert eng.c["lps_used"] == 9 and eng.c["lps_dropped"] == 0 and eng.c["lps_missed"] == 0
    keys = [(q["V"], q["key"]) for q in eng.lps_pool]
    assert keys == sorted(keys)
    eng2 = prepared(p, cfg, seed=2, mailbox=StreamMailbox())
    eng2.begin(time.monotonic(), 3600.0)
    eng2._lps_receive(_msg(p, X[:3], [1, 2, 3], 1))
    eng2._lps_receive(_msg(p, X[6:9], [7, 8, 9], 2))                   # ids 4-6 were overwritten before a poll
    assert eng2.c["lps_missed"] == 3 and eng2.c["lps_points"] == 6


def test_no_exchange_arms_switch_the_stream_off():
    from itrea.experiments.pair import effective_config
    for arm in ("A5", "A3r", "A5x0", "A5y0"):
        cfg, ov = effective_config({"instance": "x.mps", "arm": arm, "seed": 0, "budget": 10.0,
                                    "config": "configs/itrea2-adaptive-lps.yaml"})
        assert not cfg.lp_stream.enabled and not cfg.exchange.enabled, arm
    cfg, _ = effective_config({"instance": "x.mps", "arm": "A3", "seed": 0, "budget": 10.0,
                               "config": "configs/itrea2-adaptive-lps.yaml"})
    assert cfg.lp_stream.enabled


# ------------------------------------------------------------------------------------------------------- SCIP
def _mkp(path):
    """A 60-item, 5-row multidimensional knapsack: ~10k SCIP nodes in ~4 s (measured 2026-10-06)."""
    rng = np.random.default_rng(1)
    n, m = 60, 5
    W = rng.integers(10, 100, size=(m, n)).astype(float)
    val = W.mean(0) + rng.integers(0, 20, size=n)
    p = from_arrays(sps.csr_matrix(W), np.full(m, -np.inf), W.sum(1) * 0.5, np.zeros(n), np.ones(n), -val,
                    np.ones(n, bool), names=[f"x{j}" for j in range(n)])
    return toy.write_mps(p, path)


@pytest.mark.solver
def test_scip_streams_fractional_node_lp_points(tmp_path):
    from test_scip_exchange import MODULE, _launch, _new_mailbox, _summary
    path = _mkp(tmp_path / "mkp.mps")
    p = load(path)
    mb = _new_mailbox(tmp_path, p.names, "lps")
    run = tmp_path / "run"
    rc, out = _launch(MODULE, ["--instance", str(path), "--seed", "0", "--budget", "3", "--run-dir", str(run),
                               "--mailbox", str(tmp_path / "mb"), "--send", "--poll-s", "0.05", "--lp-stream",
                               "--lp-stream-share", "0.05", "--lp-stream-min-gap", "0.05", "--lp-stream-ring", "3"],
                      mb)
    assert rc == 0, out
    s = _summary(run)
    ls = s["lp_stream"]
    assert ls["reads"] >= 3 and ls["reads"] == ls["early"] + ls["quality"] + ls["patience"]
    assert "cutoff" in ls and ls["cutoff"] < ls["lp_solves"]
    assert ls["points"] == ls["reads"] and ls["lp_solves"] >= ls["reads"]
    assert ls["spent_s"] <= 0.05 * s["wall_s"] + 0.05            # the budget (+ start allowance and one read)
    msg = mb.lps.poll()
    assert msg is not None and 1 <= msg.X.shape[0] <= 3
    ids = msg.origin.tolist()
    assert ids == sorted(ids) and ids[-1] == ls["points"]
    for x, obj in zip(msg.X, msg.obj):
        assert p.objective(x) == pytest.approx(float(obj), rel=1e-9, abs=1e-9)
        assert np.all(x >= -1e-9) and np.all(x <= 1 + 1e-9)
        assert np.abs(x - np.round(x)).max() > 1e-6                    # never an integral point
    ev = [e for e in read_events(run / "solver.events.jsonl") if e["kind"] == "lps"]
    assert len(ev) == ls["reads"] and all(e["nfrac"] >= 1 for e in ev)


@pytest.mark.solver
def test_scip_lp_stream_needs_send(tmp_path):
    from test_scip_exchange import MODULE, _launch
    path = _mkp(tmp_path / "mkp.mps")
    rc, out = _launch(MODULE, ["--instance", str(path), "--seed", "0", "--budget", "2", "--run-dir",
                               str(tmp_path / "run"), "--no-barrier", "--lp-stream"])
    assert rc == 2, out
