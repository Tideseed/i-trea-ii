"""The three-way arm A9 = SCIP <-> EA <- cuOpt (D-015), on the CPU: cuOpt's own mailbox channels (file and Redis), the
EA reading them (`exchange.from_cuopt`: injection, attribution, duplicates by who delivered first, the external
incumbent, posts to SCIP marked POST_CUOPT — frontier posts too —, and nothing at all while off: the A3 path under
solver traffic equals the engine before the feature), the runner's plan for A9 and its matched control A3x (cores,
flags, memory budget, run ids, GPU-health own processes) without launching anything, and build_runs' cuOpt columns,
`reads_cuopt` and the runner-stopped exception of `crash_flags` on synthetic records.

Toy models only, NumPy backend; no process is launched (cuOpt and the GPU paths are tested in a GPU window)."""
from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

import _engine_reference as REF
from itrea import paths
from itrea.benchmarks import toy
from itrea.config import EAConfig
from itrea.ea.population import ORIGIN_INJECTED
from itrea.exchange.mailbox import Mailbox, Message
from itrea.experiments import gates, gpu_health, pair
from itrea.solvers.common import ORIGIN_SOLVER, POST_CUOPT
from itrea.trace import Trace, read_events
from test_engine import FakeMailbox, cfg_for, knapsack_optimum_x, prepared, run_checked

ON = {"exchange.from_cuopt": True}


# ------------------------------------------------------------------------------------------------- the mailbox
def _roundtrip(mb_writer, mb_reader):
    """cuOpt posts on its own channels; the solver's channels stay untouched (and vice versa)."""
    x = np.array([1.0, 0.0, 2.5])
    seq = mb_writer.to_ea_cuopt.post(x[None, :], np.array([7.0]), origin=np.array([ORIGIN_SOLVER], np.int64))
    mb_writer.info_cuopt.post(incumbent=7.0, dual_bound=3.0, status="solving", solver="cuopt")
    assert mb_reader.to_ea.poll() is None and mb_reader.info.poll() is None
    msg = mb_reader.to_ea_cuopt.poll()
    assert msg is not None and msg.seq == seq and np.allclose(msg.X[0], x) and msg.obj[0] == 7.0
    assert msg.origin.tolist() == [ORIGIN_SOLVER]
    info = mb_reader.info_cuopt.poll()
    assert info["incumbent"] == 7.0 and info["dual_bound"] == 3.0 and info["solver"] == "cuopt"
    assert mb_reader.to_ea_cuopt.poll() is None and mb_reader.info_cuopt.poll() is None   # seen once
    mb_writer.to_ea.post(x[None, :], np.array([9.0]))                                    # the solver's post ...
    assert mb_reader.to_ea_cuopt.poll() is None                                           # ... is not cuOpt's
    assert mb_reader.to_ea.poll().obj[0] == 9.0


def test_file_mailbox_cuopt_channels(tmp_path):
    runner = Mailbox(tmp_path / "mb", "runner", create=True, run_id="mb")
    _roundtrip(Mailbox(tmp_path / "mb", "cuopt", run_id="mb"), Mailbox(tmp_path / "mb", "ea", run_id="mb"))
    names = {p.name for p in runner.dir.iterdir()}
    assert {"to_ea_cuopt.npz", "to_ea_cuopt_info.json", "to_ea.npz"} <= names


def test_redis_mailbox_cuopt_channels(tmp_path):
    from itrea.experiments import exchange_ctl
    if not exchange_ctl.ping():
        pytest.skip("itrea-redis is not answering on 127.0.0.1:6390")
    from itrea.exchange.redis_mailbox import RedisMailbox
    rid = f"pytest-3way-{uuid.uuid4().hex[:10]}"
    url = f"redis://127.0.0.1:6390/0?run={rid}"
    runner = RedisMailbox(url, run_id=rid, role="runner", create=True)
    try:
        _roundtrip(RedisMailbox(url, run_id=rid, role="cuopt"), RedisMailbox(url, run_id=rid, role="ea"))
        keys = {k.decode() for k in runner.r.scan_iter(match=f"itrea:{rid}:*")}
        assert {f"itrea:{rid}:ch:to_ea_cuopt", f"itrea:{rid}:info_cuopt", f"itrea:{rid}:ch:to_ea"} <= keys
        # the runner's exchange record keeps cuOpt's stream only when cuOpt sent (A9)
        o = pair.PairOptions(mailbox="redis")
        out = pair._exchange_log(runner, o, tmp_path, cuopt=True)
        hist = json.loads((tmp_path / "exchange.json").read_text())
        assert out["streams"]["to_ea_cuopt"] == 1 and len(hist["to_ea_cuopt"]) == 1
        pair._exchange_log(runner, o, tmp_path, cuopt=False)
        assert "to_ea_cuopt" not in json.loads((tmp_path / "exchange.json").read_text())
    finally:
        runner.cleanup()


# ------------------------------------------------------------------------------------------------- the engine
class ThreeWayMailbox(FakeMailbox):
    """FakeMailbox plus cuOpt's channels: queued `to_ea_cuopt` messages and `info_cuopt` dicts."""

    def __init__(self, to_ea_cuopt=(), info_cuopt=(), **kw):
        super().__init__(**kw)
        q, qi = list(to_ea_cuopt), list(info_cuopt)

        class _Q:
            def __init__(self, items):
                self.items = items

            def poll(self):
                return self.items.pop(0) if self.items else None

        self.to_ea_cuopt, self.info_cuopt = _Q(q), _Q(qi)


class CuoptSpy(FakeMailbox):
    """A solver partner whose cuOpt channels raise on any access (an EA without from_cuopt must never read them)."""

    @property
    def to_ea_cuopt(self):
        raise AssertionError("read to_ea_cuopt while exchange.from_cuopt is off")

    @property
    def info_cuopt(self):
        raise AssertionError("read info_cuopt while exchange.from_cuopt is off")


def _msg(x, obj, seq=1):
    return Message(X=np.atleast_2d(x), obj=np.atleast_1d(np.asarray(obj, float)), seq=seq, t=0.0,
                   origin=np.full(np.atleast_2d(x).shape[0], ORIGIN_SOLVER, np.int64))


def test_off_never_reads_cuopt_channels(tmp_path):
    """A3's EA (from_cuopt false) with a talking solver: injection, info, posts and verdicts work, cuOpt's channels are
    never touched, and nothing A9-specific is recorded."""
    p = toy.knapsack(30)
    x_opt = knapsack_optimum_x(p)
    mb = CuoptSpy(to_ea=[_msg(x_opt, p.objective(x_opt))], info=[dict(incumbent=p.toy_optimum, dual_bound=None)],
                  verdict=lambda seq, x, obj: True)
    tr = Trace(tmp_path, "ea")
    eng = prepared(p, cfg_for(**{"exchange.inject_band": 0.5, "log_every": 1}), seed=3, mailbox=mb, trace=tr)
    run_checked(eng, 5)
    out = eng.finish()
    tr.close()
    assert out["injected"] == 1 and not eng.from_cuopt
    assert not any(k.startswith("cuopt_") for k in out) and "exchange_from_cuopt" not in out
    assert "cuopt" not in eng.genealogy() and not eng.cuopt_ids and eng.inc_cuopt == math.inf
    ev = read_events(tmp_path / "ea.events.jsonl")
    assert not any(e["kind"] == "inject_cuopt" for e in ev)
    assert not any(k in e for e in ev if e["kind"] in ("gen", "best", "post") for k in
                   ("inc_cuopt", "inc_solver", "cuopt", "cuopt_rows", "cuopt_pop"))
    assert all(origin is None or not (int(origin[0]) & POST_CUOPT) for _, _, origin in mb.posts)


class _CuoptLPMailbox(REF.LPMailbox):
    """The reference capture's partner (one LP point) with EMPTY cuOpt channels."""

    def __init__(self, lp_x, obj):
        super().__init__(lp_x, obj)

        class _E:
            def poll(self):
                return None

        self.to_ea_cuopt, self.info_cuopt = _E(), _E()


def test_on_with_a_silent_cuopt_equals_the_reference(monkeypatch):
    """from_cuopt on, cuOpt silent: no random number and no state change, so genealogy, final population, counters and
    best key equal the recorded reference of the engine before the feature (commit dd31d67) on every toy."""
    monkeypatch.setattr(REF, "LPMailbox", _CuoptLPMailbox)
    ref = REF.load_reference()
    for name in REF.TOYS:
        got = REF.capture(name, "numpy", cfg=REF.config("numpy", **ON))
        for k, v in got.items():
            r = ref[f"numpy/{name}/{k}"]
            assert r.shape == v.shape and np.array_equal(r, v, equal_nan=True), (name, k)


def test_cuopt_rows_are_injected_attributed_and_posted(tmp_path):
    """cuOpt's optimum enters the population as an injected row (never repaired), becomes the EA's incumbent with
    source "cuopt", is posted to SCIP (whose incumbent is worse) with POST_CUOPT, and SCIP's verdict is counted for
    cuOpt; the solver's counters stay the solver's, and cuOpt's bound is recorded but not used."""
    p = toy.knapsack(30)
    x_opt = knapsack_optimum_x(p)
    worse = x_opt.copy()
    worse[np.flatnonzero(x_opt)[:2]] = 0.0
    assert p.check(worse).feasible and p.objective(worse) < p.toy_optimum
    mb = ThreeWayMailbox(to_ea_cuopt=[_msg(x_opt, p.toy_optimum)],
                         info_cuopt=[dict(incumbent=p.toy_optimum, dual_bound=p.toy_optimum + 3.0, status="solving",
                                          solver="cuopt")],
                         info=[dict(incumbent=p.objective(worse), dual_bound=None, status="solving")],
                         verdict=lambda seq, x, obj: True)
    tr = Trace(tmp_path, "ea")
    cfg = cfg_for(**ON, **{"exchange.poll_s": 1e-6, "log_every": 1})
    eng = prepared(p, cfg, seed=9, mailbox=mb, trace=tr)                  # warm-up reads cuOpt's channels too
    run_checked(eng, 4)
    eng._exchange_in()                                                    # the verdict on the last post
    out = eng.finish()
    tr.close()
    k_opt = p.key_of_objective(p.toy_optimum)
    assert eng.from_cuopt and out["exchange_from_cuopt"] is True
    assert (out["cuopt_inject_msgs"], out["cuopt_inject_rows"], out["cuopt_injected"]) == (1, 1, 1)
    assert out["cuopt_injected_infeasible"] == 0 and out["cuopt_info_msgs"] == 1
    assert (out["inject_msgs"], out["injected"]) == (0, 0)                # the solver's counters: no solver rows
    # the external incumbent is the better of the two; posts compare with the SOLVER's
    assert eng.inc_cuopt == pytest.approx(k_opt) and eng.inc_solver == pytest.approx(-p.objective(worse))
    assert eng.inc_external == pytest.approx(k_opt) == out["inc_external"]
    assert eng.dual_key == -math.inf and eng.dual_key_cuopt == pytest.approx(-(p.toy_optimum + 3.0))
    # attribution
    cu = [r for r in eng.records if r[2] == "cuopt"]
    assert out["cuopt_incumbents"] == len(cu) >= 1 and eng.best_obj == pytest.approx(p.toy_optimum)
    g = eng.genealogy()
    rows = np.flatnonzero(g["cuopt"])
    assert rows.size == 1 and g["origin"][rows[0]] & ORIGIN_INJECTED and g["is_new_best"][rows[0]]
    cid = int(g["id"][rows[0]])
    assert eng.cuopt_ids == {cid}
    # the post: the cuOpt vector, marked, its verdict counted for cuOpt
    marked = [(X, obj, org) for X, obj, org in mb.posts if int(org[0]) & POST_CUOPT]
    assert len(marked) == 1 and np.allclose(marked[0][0][0], x_opt)
    assert int(marked[0][2][0]) & ORIGIN_INJECTED and out["cuopt_posts"] == 1
    assert out["cuopt_rows_accepted"] == 1 and out["cuopt_rows_rejected"] == 0
    assert cid in eng.cuopt_posted and g["is_accepted_post"][rows[0]]
    ev = read_events(tmp_path / "ea.events.jsonl")
    assert [e["k"] for e in ev if e["kind"] == "inject_cuopt"] == [1]
    assert any(e["kind"] == "incumbent" and e["source"] == "cuopt" for e in ev)
    assert any(e["kind"] == "best" and e.get("cuopt") is True for e in ev)
    posts = [e for e in ev if e["kind"] == "post"]
    assert [e["cuopt_rows"] for e in posts if e["cuopt_rows"]] == [[0]]
    gens = [e for e in ev if e["kind"] == "gen"]
    assert gens[-1]["inc_cuopt"] == pytest.approx(k_opt) and gens[-1]["cuopt_pop"] >= 1
    assert out["final_cuopt_pop"] >= 1


def test_a_better_solver_incumbent_blocks_the_cuopt_post():
    """SCIP already holds the optimum: cuOpt's (worse) vector still enters, but no post goes to SCIP — the post rule
    compares with the solver's incumbent."""
    p = toy.knapsack(30)
    x_opt = knapsack_optimum_x(p)
    worse = x_opt.copy()
    worse[np.flatnonzero(x_opt)[:2]] = 0.0
    mb = ThreeWayMailbox(to_ea_cuopt=[_msg(worse, p.objective(worse))],
                         info_cuopt=[dict(incumbent=p.objective(worse), dual_bound=None)],
                         info=[dict(incumbent=p.toy_optimum, dual_bound=None)], verdict=lambda seq, x, obj: True)
    eng = prepared(p, cfg_for(**ON, **{"exchange.inject_band": 0.5}), seed=2, mailbox=mb)
    eng.begin(time.monotonic(), 3600.0)
    assert eng.inc_external == pytest.approx(p.key_of_objective(p.toy_optimum))   # SCIP's is the better one
    for _ in range(2):
        eng.step()
    out = eng.finish()
    assert out["cuopt_injected"] == 1
    assert all(p.key_of_objective(obj[0]) < eng.inc_solver for _, obj, _ in mb.posts)
    assert out["cuopt_posts"] == 0 and not any(int(o[0]) & POST_CUOPT for _, _, o in mb.posts)


def test_cuopt_incumbent_alone_arms_the_cutoff_and_the_rank0_threshold():
    """Only cuOpt's info (no rows, no solver): its incumbent is the external incumbent of the survival and the
    cutoff, as the solver's would be."""
    p = toy.set_cover(40, 25)
    inc = 9.0                                                    # a minimisation incumbent (set cover costs 1..)
    mb = ThreeWayMailbox(info_cuopt=[dict(incumbent=inc, dual_bound=1.0)])
    eng = prepared(p, cfg_for(**ON), seed=4, mailbox=mb)
    eng.begin(time.monotonic(), 3600.0)
    k = p.key_of_objective(inc)
    assert eng.inc_external == pytest.approx(k) and eng.inc_solver == math.inf
    assert eng.last_surv.inc <= k + 1e-12
    pop_best = float(eng.h_key[eng.h_feas].min()) if eng.h_feas.any() else math.inf
    assert eng.cutoff == pytest.approx(min(pop_best, k))
    eng.step()
    assert eng.last_surv.inc <= k + 1e-12


def test_a_vector_both_deliver_stays_the_solvers():
    """SCIP and cuOpt deliver the same vector in one tick: the solver's copy is kept (it is read first), so the
    incumbent is the solver's find and nothing is credited to cuOpt."""
    p = toy.knapsack(30)
    x_opt = knapsack_optimum_x(p)
    mb = ThreeWayMailbox(to_ea=[_msg(x_opt, p.toy_optimum)], to_ea_cuopt=[_msg(x_opt, p.toy_optimum)])
    eng = prepared(p, cfg_for(**ON), seed=5, mailbox=mb)
    run_checked(eng, 2)
    out = eng.finish()
    assert out["injected"] == 1 and out["cuopt_injected"] == 1
    assert any(r[2] == "solver" for r in eng.records) and not any(r[2] == "cuopt" for r in eng.records)
    assert out["cuopt_incumbents"] == 0


def test_rejected_cuopt_post_is_counted_and_demoted():
    p = toy.knapsack(30)
    x_opt = knapsack_optimum_x(p)
    mb = ThreeWayMailbox(to_ea_cuopt=[_msg(x_opt, p.toy_optimum)], verdict=lambda seq, x, obj: False)
    eng = prepared(p, cfg_for(**ON, **{"exchange.poll_s": 1e-6}), seed=6, mailbox=mb)
    run_checked(eng, 3, elitist=False)
    eng._exchange_in()
    out = eng.finish()
    assert out["cuopt_posts"] >= 1 and out["cuopt_rows_rejected"] >= 1 and out["cuopt_rows_accepted"] == 0
    assert eng.cuopt_posted <= eng.rejected_ids


class _Late:
    """A queue that answers only from its (delay + 1)-th poll on (begin polls once, every generation once more)."""

    def __init__(self, items, delay):
        self.items, self.delay, self.n = list(items), delay, 0

    def poll(self):
        self.n += 1
        return self.items.pop(0) if self.n > self.delay and self.items else None


@pytest.mark.parametrize("first", ["solver", "cuopt"])
def test_duplicate_counters_follow_who_delivered_first(first):
    """The same vector from both channels, the second ticks later: the earlier copy is a member and the later one its
    duplicate, counted for the LATER deliverer (SCIP's `injected_dup` stays SCIP's); the incumbent is credited to the
    earlier one."""
    p = toy.knapsack(30)
    x_opt = knapsack_optimum_x(p)
    mb = ThreeWayMailbox(to_ea=[_msg(x_opt, p.toy_optimum)], to_ea_cuopt=[_msg(x_opt, p.toy_optimum)])
    late = "to_ea_cuopt" if first == "solver" else "to_ea"
    setattr(mb, late, _Late([_msg(x_opt, p.toy_optimum)], 3))
    eng = prepared(p, cfg_for(**ON, **{"exchange.poll_s": 1e-6}), seed=5, mailbox=mb)
    run_checked(eng, 5)
    out = eng.finish()
    assert out["injected"] == 1 and out["cuopt_injected"] == 1
    if first == "solver":
        assert (out["injected_dup"], out["cuopt_injected_dup"]) == (0, 1)
        assert any(r[2] == "solver" for r in eng.records) and out["cuopt_incumbents"] == 0
    else:
        assert (out["injected_dup"], out["cuopt_injected_dup"]) == (1, 0)
        assert any(r[2] == "cuopt" for r in eng.records) and out["cuopt_incumbents"] == 1
        assert not any(r[2] == "solver" for r in eng.records)


def test_frontier_post_marks_a_cuopt_member(tmp_path):
    """A cuOpt row that is near-optimal but not the best (SCIP delivers the optimum) enters the frontier and is posted
    from there: its row, and only its row, carries POST_CUOPT, and the `post` event lists it in `cuopt_rows`."""
    p = toy.knapsack(30)
    x_opt = knapsack_optimum_x(p)
    on = np.flatnonzero(x_opt)
    near = x_opt.copy()
    near[on[np.argmin(np.asarray(p.c)[on])]] = 0.0              # the cheapest item fewer: feasible, near-optimal,
    assert p.check(near).feasible                                # not dominated in (key, distance) by the EA's members
    mb = ThreeWayMailbox(to_ea=[_msg(x_opt, p.toy_optimum)], to_ea_cuopt=[_msg(near, p.objective(near))],
                         verdict=lambda seq, x, obj: True)
    tr = Trace(tmp_path, "ea")
    cfg = cfg_for(**ON, **{"exchange.poll_s": 1e-6, "exchange.inject_band": 0.5, "exchange.post_frontier": True,
                           "frontier.enabled": True, "frontier.update_every": 1, "frontier.band": 0.2})
    eng = prepared(p, cfg, seed=8, mailbox=mb, trace=tr)
    run_checked(eng, 4)
    out = eng.finish()
    tr.close()
    assert len(eng.cuopt_ids) == 1
    cid = next(iter(eng.cuopt_ids))
    fr = [e for e in read_events(tmp_path / "ea.events.jsonl") if e["kind"] == "post" and e["frontier"]
          and cid in e["ids"]]
    assert fr, "the cuOpt member never reached a frontier post"
    for e in fr:
        assert e["cuopt_rows"] == [e["ids"].index(cid)]
        _, _, org = mb.posts[e["seq"] - 1]
        marked = [r for r in range(len(org)) if int(org[r]) & POST_CUOPT]
        assert marked == e["cuopt_rows"]
    assert cid in eng.cuopt_posted and out["cuopt_posts"] >= len(fr)
    assert out["cuopt_rows_accepted"] >= len(fr)                      # one verdict per marked row


class _TrafficCuoptMailbox(REF.TrafficMailbox):
    """The traffic reference's partner with EMPTY cuOpt channels."""

    def __init__(self, **kw):
        super().__init__(**kw)

        class _E:
            def poll(self):
                return None

        self.to_ea_cuopt, self.info_cuopt = _E(), _E()


@pytest.mark.parametrize("on", [False, True])
def test_solver_traffic_equals_the_reference_before_the_feature(on):
    """The A3 exchange path under solver traffic (rows with an echo, an infeasible row and a repeated vector; info with
    incumbent and bound; an LP point and an LP-stream message; accept / reject verdicts; frontier posting; the
    strategy's solver / EA incumbent split) gives the same genealogy, population, counters, incumbent records, posts
    (rows, objectives, origins), external incumbent, bound and cutoff as the engine of commit 4db3f7c, before
    `exchange.from_cuopt` existed — with the flag off, and on with a silent cuOpt."""
    ref = REF.load_traffic_reference()
    for name in REF.TRAFFIC_TOYS:
        if on:
            got = REF.capture_traffic(name, cfg=REF.traffic_config(name, **ON), mailbox_cls=_TrafficCuoptMailbox)
        else:
            got = REF.capture_traffic(name)
        keys = {k.split("/", 1)[1] for k in ref if k.startswith(name + "/")}
        assert keys == set(got), (name, keys ^ set(got))
        for k, v in got.items():
            r = ref[f"{name}/{k}"]
            assert r.shape == v.shape and np.array_equal(r, v, equal_nan=True), (name, k)
    c = dict(zip(REF.TRAFFIC_COUNTERS, ref["knapsack/counters"]))       # the scenario exercises what it claims
    assert c["echoes"] >= 1 and c["injected_infeasible"] >= 1 and c["injected_dup"] >= 1
    assert c["solver_rejections"] >= 1 and c["solver_accepted"] >= 1 and c["posts_frontier"] >= 1
    assert 2 in ref["knapsack/rec_src"] and ref["knapsack/lps_counters"][2] >= 1


# ------------------------------------------------------------------------------------------------- the runner
@pytest.fixture
def toy_mps(tmp_path) -> Path:
    return toy.write_mps(toy.set_cover(30, 20, seed=1), tmp_path / "toy_sc.mps")


def _opts(tmp_path, **kw) -> pair.PairOptions:
    d = dict(traces_dir=tmp_path / "traces", mailbox_root=tmp_path / "mb", use_systemd=True, pin=True,
             gpu_health=False, record_snapshots=False, solver_mem_gb=9.0, ea_mem_gb=6.0, cuopt_mem_gb=12.0,
             cupy_cache_dir=tmp_path / "cupy-cache", log=lambda m: None)
    d.update(kw)
    return pair.PairOptions(**d)


def _job(inst, arm, **kw) -> dict:
    return dict(instance=str(inst), set="pytest", arm=arm, seed=1000, budget=300, split="smoke", **kw)


def _plan(inst, tmp_path, arm, **kw):
    o = _opts(tmp_path)
    j = _job(inst, arm, **kw)
    cfg, ov = pair.effective_config(j, o)
    rd = tmp_path / "traces" / "raw" / "pytest" / pair.run_id_for(j, o)
    return {p.role: p for p in pair.plan(j, o, rd, "SPEC", cfg)}, cfg, ov


def test_a9_plan_cores_flags_and_config(toy_mps, tmp_path):
    procs, cfg, ov = _plan(toy_mps, tmp_path, "A9")
    s, e, c = procs["solver"], procs["ea"], procs["cuopt"]
    lane = pair.GPU_LANE
    # SCIP on x925[0], the EA on x925[1], cuOpt on the A725 cores: no core shared
    assert s.cores == [lane.x925[0]] and e.cores == [lane.x925[1]] and c.cores == list(lane.a725)
    assert not (set(s.cores) & set(e.cores) or set(s.cores) & set(c.cores) or set(e.cores) & set(c.cores))
    assert e.cmd[e.cmd.index("taskset") + 2] == str(lane.x925[1])
    # flags: SCIP as in A3, cuOpt --send with num_cpu_threads = its cores, the EA reads cuOpt
    assert "--recv" in s.argv and "--send" in s.argv and "--inert" not in s.argv
    a3, _, _ = _plan(toy_mps, tmp_path, "A3")
    assert s.argv[s.argv.index("--recv"):] == a3["solver"].argv[a3["solver"].argv.index("--recv"):]
    assert "--send" in c.argv and c.argv[c.argv.index("--cpu-threads") + 1] == str(len(lane.a725))
    assert ov["exchange.from_cuopt"] is True and cfg.exchange.from_cuopt and cfg.backend == "cupy"
    # two CUDA processes with the same GPU limit
    g = EAConfig().gpu_mem_gb
    assert e.gpu_gb == c.gpu_gb == g
    assert e.env_extra["CUPY_GPU_MEMORY_LIMIT"] == c.env_extra["CUPY_GPU_MEMORY_LIMIT"] == str(int(g * 1024**3))
    assert pair.ARMS["A9"].gpu and pair.ARMS["A9"].lane == "partner" and pair.ARMS["A9"].roles == \
        ("solver", "ea", "cuopt")


def test_controls_never_send_from_cuopt(toy_mps, tmp_path):
    """A2 and A6 (the cuOpt controls) never pass --send; A3's EA never reads cuOpt; an A9 job whose configuration turns
    the exchange off sends nothing either (cuOpt quiet, SCIP inert)."""
    for arm in ("A2", "A6"):
        procs, _, _ = _plan(toy_mps, tmp_path, arm)
        assert "--send" not in procs["cuopt"].argv
    _, cfg3, ov3 = _plan(toy_mps, tmp_path, "A3")
    assert not cfg3.exchange.from_cuopt and "exchange.from_cuopt" not in ov3
    procs, _, _ = _plan(toy_mps, tmp_path, "A9", overrides={"exchange.enabled": False})
    assert "--send" not in procs["cuopt"].argv and "--inert" in procs["solver"].argv


def test_a9_memory_budget_counts_two_cuda_processes(toy_mps, tmp_path):
    o = _opts(tmp_path)
    b9 = pair.job_budget(_job(toy_mps, "A9"), o)
    b3 = pair.job_budget(_job(toy_mps, "A3"), o)
    g = EAConfig().gpu_mem_gb
    assert (b9.host_caps_gb, b9.gpu_caps_gb, b9.cuda_procs) == (9.0 + 6.0 + 12.0, 2 * g, 2)
    assert b9.need_gb == pytest.approx(27.0 + 2 * g + 2 * gates.GPU_OVERHEAD_GB + gates.RESERVE_GB)
    assert b9.need_gb - b3.need_gb == pytest.approx(12.0 + g + gates.GPU_OVERHEAD_GB)   # + cuOpt
    o.cuopt_gpu_peak_gb = 10.5                                    # the measured RMM peak replaces cuOpt's limit only
    assert pair.job_budget(_job(toy_mps, "A9"), o).gpu_caps_gb == pytest.approx(g + 10.5)


def test_a9_needs_two_x925_cores_and_a725_cores(toy_mps, tmp_path):
    for lane in (pair.Lane("one", (5,), (1, 2, 3), gpu=True), pair.Lane("noa", (5, 6), (), gpu=True),
                 pair.Lane("cpu", (7, 8))):
        with pytest.raises(pair.RunError, match="cannot run A9"):
            pair.job_budget(_job(toy_mps, "A9"), _opts(tmp_path, lane=lane))
    assert pair.ARMS["A9"].n_x925 == 2 and pair.ARMS["A9"].needs_a725
    assert pair.ARMS["A1"].n_x925 == 2 and pair.ARMS["A3"].n_x925 == 1          # unchanged for the other arms


def test_a9_run_ids_with_the_development_configuration(toy_mps, tmp_path):
    """A3 and A9 jobs share configs/itrea2-dev-rc-3way.yaml; the arm's from_cuopt gives A9 its own run hash."""
    o = _opts(tmp_path)
    j3, j9 = (_job(toy_mps, a, config="configs/itrea2-dev-rc-3way.yaml") for a in ("A3", "A9"))
    assert pair.base_config_hash(j3) == pair.base_config_hash(j9) == "1d12ad490dd0"
    assert pair.run_hash(j3, o) == "1d12ad490dd0"
    h9 = pair.run_hash(j9, o)
    want = EAConfig.from_yaml(paths.ROOT / "configs/itrea2-dev-rc-3way.yaml").with_overrides(
        {"exchange.from_cuopt": True, "backend": "cupy"}).hash()
    assert h9 == want != "1d12ad490dd0"
    assert "__A9__" in pair.run_id_for(j9, o)


DEV = "configs/itrea2-dev-rc-3way.yaml"


def test_a3x_is_a9_without_cuopt(toy_mps, tmp_path):
    """A9's matched control A3x (review of 2026-10-07, D-015 addendum): SCIP and the EA on A9's cores (x925[0],
    x925[1]) with A3's flags and A3's configuration (from_cuopt off), no cuOpt, A3's memory budget, A3's run hash with
    its own arm label; A3 itself puts the EA on the A725 cores. The arm is appended, so the canonical order of the
    other arms is unchanged."""
    from itrea.analysis import build_runs as B
    lane = pair.GPU_LANE
    p9, _, _ = _plan(toy_mps, tmp_path, "A9", config=DEV)
    px, cfgx, ovx = _plan(toy_mps, tmp_path, "A3x", config=DEV)
    p3, cfg3, ov3 = _plan(toy_mps, tmp_path, "A3", config=DEV)
    assert set(px) == {"solver", "ea"}
    assert px["solver"].cores == p9["solver"].cores == [lane.x925[0]]
    assert px["ea"].cores == p9["ea"].cores == [lane.x925[1]] and p3["ea"].cores == list(lane.a725)
    assert px["ea"].cmd[px["ea"].cmd.index("taskset") + 2] == str(lane.x925[1])
    tail = px["solver"].argv[px["solver"].argv.index("--recv"):]
    assert tail == p3["solver"].argv[p3["solver"].argv.index("--recv"):] == \
        p9["solver"].argv[p9["solver"].argv.index("--recv"):]
    assert "--send" in tail and "--lp-stream" in tail and "--inert" not in tail
    env = ({k: v for k, v in q["ea"].env_extra.items() if k != "ITREA_RUN_ID"} for q in (px, p3))
    assert px["ea"].gpu_gb == p3["ea"].gpu_gb and next(env) == next(env)
    assert ovx == ov3 and "exchange.from_cuopt" not in ovx and not cfgx.exchange.from_cuopt
    assert cfgx.hash() == cfg3.hash() == "1d12ad490dd0"
    o = _opts(tmp_path)
    jx, j3 = _job(toy_mps, "A3x", config=DEV), _job(toy_mps, "A3", config=DEV)
    assert pair.run_hash(jx, o) == pair.run_hash(j3, o) == "1d12ad490dd0"
    assert "__A3x__" in pair.run_id_for(jx, o) and pair.run_id_for(jx, o) != pair.run_id_for(j3, o)
    bx, b3 = pair.job_budget(jx, o), pair.job_budget(j3, o)
    assert (bx.host_caps_gb, bx.gpu_caps_gb, bx.cuda_procs, bx.need_gb) == \
        (b3.host_caps_gb, b3.gpu_caps_gb, b3.cuda_procs, b3.need_gb)
    ax = pair.ARMS["A3x"]
    assert ax.n_x925 == 2 and not ax.needs_a725 and lane.fits(ax) and ax.gpu and ax.lane == "partner"
    before_a6m = pair.ARM_ORDER[:pair.ARM_ORDER.index("A6m")]       # A6m (D-020) was appended after A3x
    assert before_a6m[-2:] == ("A9", "A3x") and pair.ARM_ORDER.index("A3") < pair.ARM_ORDER.index("A9")
    assert "A3x" in B.GPU_ARMS and "A3x" in B.EA_GPU_ARMS and "A3x" not in B.CUOPT_X_ARMS


def test_a9_run_id_folds_a_non_default_cuopt_thread_count(toy_mps, tmp_path):
    """cuOpt's --cpu-threads is not in the EA configuration, so the run hash of an arm with an EA and cuOpt folds it in
    when it differs from PARTNER_THREADS: a thread-count variant gets its own run id, the default id stays the EA
    configuration hash, and arms without cuOpt ignore the option."""
    o = _opts(tmp_path)
    j9 = _job(toy_mps, "A9", config=DEV)
    h = pair.run_hash(j9, o)
    assert h == pair.effective_config(j9, o)[0].hash()
    assert pair.run_hash(j9, _opts(tmp_path, cuopt_threads=pair.PARTNER_THREADS)) == h
    o2 = _opts(tmp_path, cuopt_threads=2)
    assert pair.run_hash(j9, o2) != h and pair.run_id_for(j9, o2) != pair.run_id_for(j9, o)
    rd = tmp_path / "traces" / "raw" / "pytest" / pair.run_id_for(j9, o2)
    cu = {p.role: p for p in pair.plan(j9, o2, rd, "SPEC", pair.effective_config(j9, o2)[0])}["cuopt"]
    assert cu.argv[cu.argv.index("--cpu-threads") + 1] == "2"
    for arm in ("A3", "A3x"):
        j = _job(toy_mps, arm, config=DEV)
        assert pair.run_hash(j, o2) == pair.run_hash(j, o)


def test_gpu_health_counts_both_partners_as_own():
    """The sampler treats every launched process (and its descendants) as the run's own: the EA and cuOpt on the GPU
    together are not foreign activity; a third process is."""
    ea_pid, cuopt_pid, other = 4242, 4343, 4444
    gpu = "2450, 70, 95\n"

    def query(apps):
        return lambda cmd: gpu if cmd == gpu_health.GPU_QUERY else "".join(f"{p}, [N/A]\n" for p in apps)

    s = gpu_health.Sampler(own_pids=lambda: {ea_pid, cuopt_pid}, query=query([ea_pid, cuopt_pid]), proc="/nonexistent")
    for _ in range(5):
        s.sample()
    assert s.summary(gpu_run=True)["verdict"] == "ok"
    s2 = gpu_health.Sampler(own_pids=lambda: {ea_pid, cuopt_pid}, query=query([ea_pid, cuopt_pid, other]),
                            proc="/nonexistent")
    for _ in range(5):
        s2.sample()
    sm = s2.summary(gpu_run=True)
    assert sm["verdict"] == "void" and sm["foreign_pids"] == [other]


def test_trajectories_merge_three_roles(tmp_path):
    """The run record's portfolio is the best of all three processes; `solver` is what SCIP held."""
    d = tmp_path / "run"
    for role, incs in (("solver", [(1.0, 50.0), (6.0, 30.0)]), ("ea", [(2.0, 40.0)]), ("cuopt", [(0.5, 60.0),
                                                                                                   (3.0, 35.0)])):
        tr = Trace(d, role)
        tr.start(time.monotonic())
        for t, obj in incs:
            tr.incumbent(obj, source="self", t=t)
        tr.close()
    m = pair.merge_trajectories(d, pair.ARMS["A9"].roles, sense=1)
    assert [(p[0], p[1], p[2]) for p in m["portfolio"]] == [(0.5, 60.0, "cuopt"), (1.0, 50.0, "solver"),
                                                             (2.0, 40.0, "ea"), (3.0, 35.0, "cuopt"),
                                                             (6.0, 30.0, "solver")]
    assert [p[1] for p in m["solver"]] == [50.0, 30.0]
    assert m["incumbent_events"] == {"solver": 2, "ea": 1, "cuopt": 2}


# ------------------------------------------------------------------------------------------------- build_runs
def test_build_runs_cuopt_columns(tmp_path):
    """A synthetic A9 run: cuOpt finds 8.5, the EA injects it (incumbent source "cuopt"), posts it to SCIP marked, SCIP
    accepts it (incumbent source "cuopt"), SCIP later finds 7.0 itself; a frontier post of two cuOpt rows (two
    verdicts) and a re-post of the 8.5 vector (a `dup` verdict) follow. Portfolio = best of three; cuOpt's
    contribution counted PER POST under post_accounting's rules (every post here is a cuOpt post, so the cuOpt post
    columns equal `posts*`); an A3 run of the same build leaves the cuOpt columns missing (-1)."""
    from itrea.analysis import build_runs as B
    from test_build_runs import X_70, X_85, make_run, toy_problem
    p, mps = toy_problem(tmp_path)
    raw = tmp_path / "raw"
    d9 = make_run(raw, p, mps, "A9", 1000, 10,
                  {"cuopt": [(1.0, X_85, "self")], "ea": [(1.5, X_85, "cuopt")],
                   "solver": [(2.0, X_85, "cuopt"), (6.0, X_70, "self")]},
                  summary_extra={"ea": {"exchange_from_cuopt": True, "backend": "cupy"}},
                  run_json_extra={"base_arm": "A9", "gpu_health": {"verdict": "ok", "gpu_run": True}})
    with open(d9 / "ea.events.jsonl", "a") as f:
        for e in (dict(t=1.4, kind="inject_cuopt", seq=1, k=1, rows=1, echo=0, best_obj=8.5),
                  dict(t=1.6, kind="post", seq=1, obj=8.5, gain=1.0, k=1, frontier=False, ids=[7], cuopt_rows=[0]),
                  dict(t=3.0, kind="post", seq=3, obj=9.0, gain=None, k=2, frontier=True, ids=[8, 9],
                       cuopt_rows=[0, 1]),
                  dict(t=4.0, kind="post", seq=4, obj=8.5, gain=0.1, k=1, frontier=False, ids=[7], cuopt_rows=[0]),
                  dict(t=12.0, kind="inject_cuopt", seq=2, k=1, rows=1, echo=0, best_obj=7.0)):   # after T
            f.write(json.dumps(e) + "\n")
    with open(d9 / "solver.events.jsonl", "a") as f:
        for v in (dict(t=2.0, seq=1, obj=8.5, row=0, stored=True, dup=False, primal_after=8.5),
                  dict(t=3.1, seq=3, obj=9.0, row=0, stored=True, dup=False, primal_after=8.5),
                  dict(t=3.2, seq=3, obj=9.5, row=1, stored=True, dup=False, primal_after=8.5),
                  dict(t=4.1, seq=4, obj=8.5, row=0, stored=False, dup=True, primal_after=8.5)):
            f.write(json.dumps(dict(v, kind="verdict", accepted=True, cuopt=True)) + "\n")
    make_run(raw, p, mps, "A3", 1000, 10, {"solver": [(4.0, X_85, "self"), (5.0, X_70, "ea")],
                                           "ea": [(4.5, X_70, "self")]})
    # a second seed whose SCIP wrote no verdict events: the runner's verdicts.jsonl answers (monotonic t, t0 = 100)
    d9b = make_run(raw, p, mps, "A9", 1001, 10, {"cuopt": [(1.0, X_85, "self")], "ea": [(1.5, X_85, "cuopt")],
                                                 "solver": [(2.0, X_85, "cuopt")]},
                   summary_extra={"ea": {"exchange_from_cuopt": True, "backend": "cupy"}},
                   run_json_extra={"base_arm": "A9", "t0": 100.0})
    with open(d9b / "ea.events.jsonl", "a") as f:
        f.write(json.dumps(dict(t=1.6, kind="post", seq=1, obj=8.5, gain=1.0, k=1, frontier=False, ids=[7],
                                cuopt_rows=[0])) + "\n")
    (d9b / "verdicts.jsonl").write_text(
        json.dumps(dict(seq=1, t=102.0, obj=8.5, accepted=True,
                        note=json.dumps(dict(row=0, cuopt=True, primal_after=8.5)))) + "\n" +
        json.dumps(dict(seq=2, t=115.0, obj=8.5, accepted=True,
                        note=json.dumps(dict(row=0, cuopt=True, primal_after=8.5)))) + "\n")
    out = tmp_path / "datasets"
    out.mkdir()
    df, inc = B.build(raw, out, log=lambda *a: None)
    rb = df.set_index("run_id").loc[d9b.name]
    assert (rb["cuopt_posts"], rb["cuopt_posts_accepted"], rb["cuopt_posts_became_incumbent"],
            rb["solver_inc_from_cuopt"]) == (1, 1, 1, 1)                     # the second verdict is after T
    r = df[df["seed"] == 1000].set_index("arm")
    g85 = 1.5 / 8.5
    assert r.loc["A9", "pt"] == pytest.approx((1 + 5 * g85) / 10)            # cuOpt's 8.5 at 1 s, SCIP's 7 at 6 s
    assert r.loc["A9", "cuopt_pt"] == pytest.approx((1 + 9 * g85) / 10)
    assert r.loc["A9", "ea_pt"] == pytest.approx((1.5 + 8.5 * g85) / 10)
    assert r.loc["A9", "solver_pt"] == pytest.approx((2 + 4 * g85) / 10)
    assert (r.loc["A9", "cuopt_inject_msgs"], r.loc["A9", "cuopt_injected"], r.loc["A9", "ea_inc_from_cuopt"],
            r.loc["A9", "solver_inc_from_cuopt"]) == (1, 1, 1, 1)
    # per post: 3 posts (4 verdict rows), all accepted (the dup verdict too, as posts_accepted); the incumbent after
    # seq 1 and after the re-post seq 4 (as posts_became_incumbent), not after the frontier post (its rows are worse)
    a9 = r.loc["A9"]
    assert (a9["cuopt_posts"], a9["cuopt_posts_accepted"], a9["cuopt_posts_became_incumbent"]) == (3, 3, 2)
    assert (a9["cuopt_posts"], a9["cuopt_posts_accepted"], a9["cuopt_posts_became_incumbent"]) == \
        (a9["posts"], a9["posts_accepted"], a9["posts_became_incumbent"])
    assert r.loc["A9", "solver_inc_from_ea"] == 0                            # credited to cuOpt, not to the EA
    assert bool(r.loc["A9", "gpu_arm"]) and not bool(r.loc["A9", "backend_mismatch"])
    for col in ("cuopt_inject_msgs", "cuopt_injected", "ea_inc_from_cuopt", "cuopt_posts", "cuopt_posts_accepted",
                "cuopt_posts_became_incumbent", "solver_inc_from_cuopt"):
        assert r.loc["A3", col] == -1, col
    assert r.loc["A3", "solver_inc_from_ea"] == 1


def _run(tmp_path, arm="A9", run_json=None, summaries=None, events=None):
    from itrea.analysis import build_runs as B
    return B.Run(path=tmp_path, set="pytest", run_id=f"x__{arm}__s0__10s__h", instance=None, name="x", arm=arm,
                 seed=0, budget=10.0, config_hash="h", run_json=run_json or {}, summaries=summaries or {},
                 events=events or {})


def test_reads_cuopt_follows_what_the_ea_recorded(tmp_path):
    """reads_cuopt decides from the EA's own record first: an A9 job whose configuration turns the exchange off (cuOpt
    gets no --send, the EA reads nothing) is NOT a cuOpt-reading run, so its columns stay missing (-1) instead of 0."""
    import yaml
    from itrea.analysis import build_runs as B
    on = {"exchange": {"enabled": True, "from_cuopt": True}}
    off = {"exchange": {"enabled": False, "from_cuopt": True}}
    a9 = {"base_arm": "A9"}
    assert B.reads_cuopt(_run(tmp_path, summaries={"ea": {"exchange_from_cuopt": True}}))
    assert not B.reads_cuopt(_run(tmp_path, run_json=a9, summaries={"ea": {"config": off}}))
    assert B.cuopt_attribution(_run(tmp_path, run_json=a9, summaries={"ea": {"config": off}}), None, 10.0) == {}
    assert B.reads_cuopt(_run(tmp_path, run_json=a9, summaries={"ea": {"config": on}}))        # crashed before finish
    assert not B.reads_cuopt(_run(tmp_path, "A3", summaries={"ea": {"config": {"exchange": {"enabled": True}}}}))
    # no EA configuration recorded: the runner's config.yaml, then the base arm and the run record's overrides
    assert B.reads_cuopt(_run(tmp_path, run_json=a9))
    assert not B.reads_cuopt(_run(tmp_path, run_json=dict(a9, config={"overrides": {"exchange.enabled": False}})))
    assert B.reads_cuopt(_run(tmp_path, "A3", events={"ea": [{"t": 1.0, "kind": "inject_cuopt"}]}))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(off))
    assert not B.reads_cuopt(_run(tmp_path, run_json=a9))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(on))
    assert B.reads_cuopt(_run(tmp_path, "A3x"))


def test_crash_flags_skip_a_process_the_runner_stopped(tmp_path):
    """A2 / A9: SCIP proves optimality, the runner SIGKILLs cuOpt (no interrupt API) and marks it stopped_by_runner;
    the run is completed and build_runs must not count it as a crash (pair.run_pair's promise). The same exit code
    without the mark is a crash; the dict shape of `processes` is read too."""
    from itrea.analysis import build_runs as B
    procs = [dict(role="solver", exit_code=0, killed=None, stopped_by_runner=False),
             dict(role="ea", exit_code=0, killed=None, stopped_by_runner=False),
             dict(role="cuopt", exit_code=-9, killed="SIGKILL", stopped_by_runner=True)]
    rj = dict(processes=procs, exit_codes={"solver": 0, "ea": 0, "cuopt": -9}, stop=dict(kind="partner_end"),
              status="completed")
    summ = {"solver": {"exit_reason": "optimal"}, "ea": {"exit_reason": "stop"}, "cuopt": {}}
    ev = {"cuopt": [{"t": 3.0, "kind": "incumbent"}]}
    f = B.crash_flags(_run(tmp_path, run_json=rj, summaries=summ, events=ev))
    assert not f["crashed"] and f["crashed_roles"] == "" and math.isnan(f["crash_t"])
    assert json.loads(f["exit_codes"])["cuopt"] == -9                        # still recorded
    rj_dict = dict(rj, processes={p["role"]: {k: v for k, v in p.items() if k != "role"} for p in procs})
    assert not B.crash_flags(_run(tmp_path, run_json=rj_dict, summaries=summ, events=ev))["crashed"]
    procs2 = [dict(p, stopped_by_runner=False) for p in procs]
    f2 = B.crash_flags(_run(tmp_path, run_json=dict(rj, processes=procs2), summaries=summ, events=ev))
    assert f2["crashed"] and f2["crashed_roles"] == "cuopt" and f2["crash_t"] == 3.0
