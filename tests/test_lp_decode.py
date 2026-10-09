"""The LP decode (`repair.continuous: lp_decode`, vault D-017, issue #27; `itrea.ea.lp_decode` and the engine's pick
site): the configuration and its hashes, the default path left bit-identical, the pure selection / budget / screen
functions, the worker process (marker `solver`: real HiGHS in a spawned process on a toy MPS), the engine with the real
worker, the warm-up stub, the runner's refusal of the single-core arms and the derived-table columns.

Toy models only: `toy.mixed`, a hand-built "codes" model whose integer parts give each worker code, and a small
generated mixed model (`gen_problem`, every continuous column in two or more rows) written with `write_mps`. Worker
processes are stopped by the code under test and checked gone by PID (`os.kill(pid, 0)`), never by name.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps

import _engine_reference as REF
from itrea.benchmarks import toy
from itrea.config import EAConfig
from itrea.ea import engine as E
from itrea.ea import lp_decode as L
from itrea.ea.engine import Engine, _Batch, plan_memory
from itrea.ea.population import NO_RANK, ORIGIN_LP_DECODE, ORIGIN_X_UNIFORM, gene_dtype_for
from itrea.problem import from_arrays, load

ROOT = Path(__file__).resolve().parents[1]
SMALL = {"population.size": 48, "population.offspring_per_gen": 48, "gpu_mem_gb": 1.0}
LPD = {"repair.continuous": "lp_decode"}


def cfg_for(backend: str = "numpy", **over) -> EAConfig:
    d = dict(SMALL, backend=backend, **LPD)
    d.update(over)
    return EAConfig().with_overrides(d)


def gen_problem(seed: int = 0, n_int: int = 30, n_cont: int = 60, m: int = 40):
    """A small mixed MIP: n_int binaries, n_cont continuous in [0, 10] (every one in two or more rows, negative costs,
    so the LP pushes them against the rows), m packing rows that hold for every binary point with y = 0, plus two rows
    that make the fixed-integer LP infeasible when x0 = x1 = 0: 10 x0 + y0 + y1 >= 15 and y0 + y1 - 10 x1 <= 5 (each
    reachable alone, so the range screen passes)."""
    rs = np.random.default_rng(seed)
    n = n_int + n_cont
    A = np.zeros((m + 2, n))
    for i in range(m):
        A[i, rs.choice(n_int, size=6, replace=False)] = rs.integers(1, 6, size=6)
        A[i, n_int + rs.choice(n_cont, size=8, replace=False)] = rs.uniform(0.5, 3.0, size=8)
    for j in range(n_int, n):
        rows = np.flatnonzero(A[:m, j])
        if rows.size < 2:
            for i in rs.choice(m, size=2 - rows.size, replace=False):
                A[i, j] = rs.uniform(0.5, 3.0)
    ru = np.r_[A[:m, :n_int].sum(axis=1) + 20.0, np.inf, 5.0]
    rl = np.r_[np.full(m, -np.inf), 15.0, -np.inf]
    A[m, 0], A[m, n_int], A[m, n_int + 1] = 10.0, 1.0, 1.0
    A[m + 1, 1], A[m + 1, n_int], A[m + 1, n_int + 1] = -10.0, 1.0, 1.0
    lb = np.zeros(n)
    ub = np.r_[np.ones(n_int), np.full(n_cont, 10.0)]
    c = np.r_[rs.integers(-5, 6, n_int).astype(float), -rs.uniform(0.5, 2.0, n_cont)]
    c[n_int:n_int + 2] = [1.0, 2.0]
    isint = np.r_[np.ones(n_int, bool), np.zeros(n_cont, bool)]
    return from_arrays(sps.csr_matrix(A), rl, ru, lb, ub, c, isint, sense=1, names=[f"v{j}" for j in range(n)],
                       path=f"<gen{seed}>")


def codes_problem():
    """x0..x5 binary, y0..y3 continuous (y0, y1 in [0, 10]; y2, y3 in [0, 2]), every y in two rows:
        r0: 10 x0 + y0 + y1 >= 15          r1: y0 + y1 - 10 x1 <= 5     (x0 = x1 = 0: LP infeasible, screen passes)
        r2: 3 (x2 + x3 + x4) + y2 + y3 >= 12                            (x2 + x3 + x4 < 3: the range screen fails)
        r3: y2 - y3 + x5 <= 1
    min x0 + x1 + y0 + 2 y1 - y2 - y3."""
    A = np.array([[10, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                  [0, -10, 0, 0, 0, 0, 1, 1, 0, 0],
                  [0, 0, 3, 3, 3, 0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 0, 1, 0, 0, 1, -1]], float)
    return from_arrays(sps.csr_matrix(A), [15, -np.inf, 12, -np.inf], [np.inf, 5, np.inf, 1], np.zeros(10),
                       np.r_[np.ones(6), 10, 10, 2, 2], [1, 1, 0, 0, 0, 0, 1, 2, -1, -1],
                       np.r_[np.ones(6, bool), np.zeros(4, bool)], names=[f"c{j}" for j in range(10)])


def singleton_problem():
    """Every continuous column in exactly one row: the singleton decoder owns them all, so lp_decode is inert."""
    A = np.array([[1, 1, 1, 0], [1, -1, 0, 1]], float)
    return from_arrays(sps.csr_matrix(A), [2.5, -np.inf], [np.inf, 1.0], np.zeros(4), [3, 3, 10, 10],
                       [1, 2, 3, 1], [True, True, False, False], names=["a", "b", "s0", "s1"])


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("lpd")
    out = {}
    for name, p in (("gen", gen_problem()), ("codes", codes_problem()), ("mixed", toy.mixed()),
                    ("big", gen_problem(seed=3, n_int=40, n_cont=150, m=80))):
        path = toy.write_mps(p, d / f"{name}.mps")
        out[name] = (path, load(path))
    return out


def _decoder(path, prob, per_gen=1000, time_limit=0.05, cold_limit=1.0, deadline=math.inf, **kw):
    ints = np.asarray(prob.ints)                              # the engine's payload dtype (int8 on these binaries)
    dt = L.int_part_dtype(np.asarray(prob.lb)[ints], np.asarray(prob.ub)[ints])
    return L.LPDecoder(path, L.model_digest(prob.names, prob.isint), dt, per_gen, time_limit, cold_limit, 1e-6,
                       deadline, **kw).start()


def _wait(dec, states=("idle",), timeout=60.0):
    end = time.monotonic() + timeout
    msgs = []
    while dec.state not in states and time.monotonic() < end:
        msgs += dec.poll()
        time.sleep(0.005)
    assert dec.state in states, (dec.state, dec.error, msgs)
    return msgs


def _solve(dec, rows, kinds=None, per_gen=None, timeout=60.0):
    rows = np.asarray(rows)
    kinds = np.zeros(rows.shape[0], np.uint8) if kinds is None else np.asarray(kinds, np.uint8)
    req = getattr(dec, "_test_req", 0) + 1
    dec._test_req = req
    assert dec.offer(req, rows, kinds, per_gen=per_gen)
    msgs = _wait(dec, timeout=timeout)
    res = [m for m in msgs if m[0] == "result"]
    assert len(res) == 1 and res[0][1] == req, msgs
    return res[0][2]


def _children() -> set[int]:
    """PIDs of this process's live multiprocessing children (a module-scoped worker may be among them)."""
    return {c.pid for c in mp.active_children()}


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _fresh_lp_objective(path, prob, x_int) -> tuple[str, float]:
    """A fresh HiGHS solve of the LP with the integers fixed (the reference for the worker's optimum)."""
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    # one thread: HiGHS's task scheduler is global per process and keeps the thread count of its first run, so a
    # default-threads run here would make every later threads=1 solve in this pytest process fail ("Not Set")
    h.setOptionValue("threads", 1)
    h.setOptionValue("solve_relaxation", True)
    h.readModel(str(path))
    ints = np.flatnonzero(prob.isint).astype(np.int32)
    x = np.asarray(x_int, float)
    h.changeColsBounds(ints.size, ints, x, x)
    h.run()
    return h.modelStatusToString(h.getModelStatus()), float(h.getInfo().objective_function_value)


# ============================================================================================ configuration / hashes
def test_treatment_hash_is_pinned_and_recorded_hashes_stand():
    """D-017's treatment = the development control configs/itrea2-dev-rc-3way.yaml + repair.continuous: lp_decode.
    The two new fields are left out of the hash at their defaults, so every recorded hash stands."""
    ctl = EAConfig.from_yaml(ROOT / "configs/itrea2-dev-rc-3way.yaml")
    assert ctl.hash() == "1d12ad490dd0"
    assert ctl.with_overrides(LPD).hash() == "f030e23909ad"
    assert EAConfig().hash() == "9ac4a93bf93b"


def test_new_fields_change_the_hash_only_away_from_their_defaults():
    base = EAConfig().with_overrides(LPD)
    assert base.with_overrides({"repair.lp_decode_cold_limit": 10.0, "repair.lp_decode_infeasible": True}).hash() \
        == base.hash()
    a = base.with_overrides({"repair.lp_decode_cold_limit": 20.0})
    b = base.with_overrides({"repair.lp_decode_infeasible": False})
    assert len({base.hash(), a.hash(), b.hash()}) == 3
    assert EAConfig().with_overrides({"repair.lp_decode_infeasible": False}).hash() != EAConfig().hash()


@pytest.mark.parametrize("over, what", [
    ({"repair.lp_decode_cold_limit": 1.0}, "cold_limit"),            # below lp_time_limit (2.0)
    ({"repair.lp_decode_per_gen": 0}, "per_gen"),
    ({"repair.lp_decode_per_gen": True}, "per_gen"),
    ({"repair.lp_decode_per_gen": 2.5}, "per_gen"),
    ({"repair.lp_time_limit": 0.0}, "lp_time_limit"),
    ({"repair.lp_decode_infeasible": "yes"}, "lp_decode_infeasible"),
])
def test_validation_refuses_bad_lp_decode_settings(over, what):
    with pytest.raises(ValueError, match=what):
        EAConfig().with_overrides(over)


def test_default_yaml_lists_the_new_fields_at_their_defaults():
    import yaml
    raw = yaml.safe_load((ROOT / "configs/itrea2-default.yaml").read_text())
    assert raw["repair"]["lp_decode_cold_limit"] == 10.0 and raw["repair"]["lp_decode_infeasible"] is True
    assert raw["repair"]["continuous"] == "auto"


# =============================================================================================== pure functions
def test_active_columns_and_continuous_cost():
    p = toy.mixed()                                          # y0 in rows 0 and 2, y1 in rows 1 and 2
    assert list(L.active_columns(p)) == [2, 3] and L.continuous_cost(p)
    assert L.active_columns(singleton_problem()).size == 0
    assert L.active_columns(toy.knapsack(10)).size == 0 and not L.continuous_cost(toy.knapsack(10))
    q = toy.mixed()
    q.lb[2] = q.ub[2] = 1.0                                   # a fixed continuous column is not active
    assert list(L.active_columns(q)) == [3]
    q.c[2:] = 0.0
    assert not L.continuous_cost(q)


def test_model_digest_and_int_part_dtype():
    p = toy.mixed()
    d = L.model_digest(p.names, p.isint)
    assert d == L.model_digest(list(p.names), np.asarray(p.isint))
    assert d != L.model_digest(list(reversed(p.names)), p.isint)
    assert d != L.model_digest(p.names, ~np.asarray(p.isint))
    # the smallest integer type that holds every bound (ca: all binary -> int8, a quarter of int32's payload)
    assert L.int_part_dtype(np.zeros(4), np.ones(4)) == np.int8
    assert L.int_part_dtype(np.array([0.0, -3.0]), np.array([5.0, 7.0])) == np.int8
    assert L.int_part_dtype(np.array([-128.0]), np.array([127.0])) == np.int8
    assert L.int_part_dtype(np.array([0.0]), np.array([128.0])) == np.int16
    assert L.int_part_dtype(np.array([-32768.0]), np.array([32767.0])) == np.int16
    assert L.int_part_dtype(np.array([0.0]), np.array([40000.0])) == np.int32
    assert L.int_part_dtype(np.array([-2.0 ** 31]), np.array([2.0 ** 31 - 1])) == np.int32
    assert L.int_part_dtype(np.zeros(0), np.zeros(0)) == np.int8
    assert L.int_part_dtype(np.array([0.0]), np.array([np.inf])) == np.float64
    assert L.int_part_dtype(np.array([0.5]), np.array([3.0])) == np.float64
    assert L.int_part_dtype(np.array([0.0]), np.array([2.0 ** 40])) == np.float64
    for lo, hi in ((-128.0, 127.0), (-300.0, 300.0), (-40000.0, 40000.0)):   # a clipped value survives the cast
        dt = L.int_part_dtype(np.array([lo]), np.array([hi]))
        v = np.array([lo, hi, 0.0])
        assert np.array_equal(v.astype(dt).astype(np.float64), v)


def _sel(**kw):
    n = 10
    key = np.array([5.0, 1.0, 1.0, 3.0, 2.0, 0.5, 0.2, 0.4, 0.1, np.nan])
    viol = np.array([0, 0, 0, 0, 0, 2.0, 1.0, 1.0, 3.0, 0.0])
    feas = viol == 0
    feas[9] = False
    d = dict(ids=np.arange(n) + 100, dup=np.zeros(n, bool), solver_find=np.zeros(n, bool), decoded=np.zeros(n, bool),
             offered=np.zeros(n, bool), inc=1.0, dual_key=-math.inf, imp_tol=1e-9, per_gen=2, cont_obj=True,
             infeasible=True)
    d.update(kw)
    return L.select_offers(key, viol, feas, **d)


def test_select_offers_orders_and_caps():
    F, I = _sel()
    assert list(F) == [1, 2]                                  # feasible by (key, id), at most N = 2
    assert list(I) == [6, 7, 5, 8]                            # infeasible below inc by (V, key, id), at most 3N = 6
    F, I = _sel(per_gen=1)
    assert list(F) == [1] and list(I) == [6, 7, 5]


def test_select_offers_exclusions():
    for mask in ("dup", "solver_find", "decoded", "offered"):
        m = np.zeros(10, bool)
        m[[1, 6]] = True
        F, I = _sel(**{mask: m})
        assert 1 not in F and 6 not in I, mask
        assert list(F) == [2, 4] and list(I) == [7, 5, 8], mask
    F, I = _sel()
    assert 9 not in F and 9 not in I                          # non-finite key


def test_select_offers_threshold_phase_and_switches():
    F, I = _sel(inc=0.45)                                     # only keys below the incumbent by imp_tol
    assert list(I) == [6, 7, 8]
    F, I = _sel(inc=0.4)                                      # 0.4 itself is not better than 0.4
    assert list(I) == [6, 8]
    F, I = _sel(inc=math.inf)                                 # feasibility phase: every infeasible row
    assert list(I) == [6, 7, 5, 8]
    F, I = _sel(infeasible=False)
    assert I.size == 0 and list(F) == [1, 2]
    F, I = _sel(cont_obj=False)                               # no continuous cost: no polish rows
    assert F.size == 0 and list(I) == [6, 7, 5, 8]
    F, I = _sel(cont_obj=False, dual_key=0.3)                 # ... and nothing beyond the proven bound
    assert list(I) == [7, 5]
    F, I = _sel(cont_obj=True, dual_key=0.3)                  # with continuous cost the bound does not exclude
    assert list(I) == [6, 7, 5, 8]


def test_split_budget():
    assert L.split_budget(4, 10, 4) == (2, 2)
    assert L.split_budget(1, 10, 4) == (1, 3)                 # a short kind passes its share
    assert L.split_budget(4, 1, 4) == (3, 1)
    assert L.split_budget(0, 10, 4) == (0, 4)
    assert L.split_budget(9, 0, 4) == (4, 0)
    assert L.split_budget(0, 0, 4) == (0, 0)
    assert L.split_budget(5, 5, 1) == (1, 0)                  # N // 2 = 0 reserved for completion with N = 1
    assert L.split_budget(0, 5, 1) == (0, 1)


def test_range_screen_on_hand_built_rows():
    p = codes_problem()
    cmin, cmax = L.continuous_activity_range(p.A, p.isint, p.lb, p.ub)
    assert np.allclose(cmin, [0, 0, 0, -2]) and np.allclose(cmax, [20, 20, 4, 2])
    Ai = p.A[:, np.flatnonzero(p.isint)]

    def ok(x):
        return L.range_screen(Ai @ np.asarray(x, float), cmin, cmax, p.rl, p.ru, 1e-6)
    assert ok([1, 1, 1, 1, 1, 0])
    assert ok([0, 0, 1, 1, 1, 0])                             # r0 and r1 each reachable: the LP is what fails
    assert not ok([1, 1, 1, 1, 0, 0])                         # r2: 6 + [0, 4] < 12
    q = from_arrays(sps.csr_matrix(np.array([[1.0, 1.0]])), [-np.inf], [3.0], [0, 0], [10, np.inf], [0, 0],
                    [True, False])
    cmin, cmax = L.continuous_activity_range(q.A, q.isint, q.lb, q.ub)
    assert cmin[0] == 0 and cmax[0] == np.inf
    a = q.A[:, [0]]
    assert L.range_screen(a @ np.array([3.0]), cmin, cmax, q.rl, q.ru, 1e-6)
    assert L.range_screen(a @ np.array([3.000002]), cmin, cmax, q.rl, q.ru, 1e-6)          # within feas_tol
    assert not L.range_screen(a @ np.array([3.1]), cmin, cmax, q.rl, q.ru, 1e-6)


# ===================================================================================== default path / inert mode
def test_inert_toys_equal_the_reference_capture():
    """knapsack and set cover have no continuous column: under lp_decode the run is the default run, bit for bit
    (no file needed, no worker, no extra random number)."""
    ref = REF.load_reference()
    for name in ("knapsack", "set_cover"):
        cur = REF.capture(name, "numpy", cfg=REF.config("numpy", **LPD))
        for k, v in cur.items():
            assert np.array_equal(ref[f"numpy/{name}/{k}"], v, equal_nan=True), (name, k)


def test_inert_toys_equal_the_traffic_reference():
    ref = REF.load_traffic_reference()
    for name in ("knapsack", "set_cover"):
        got = REF.capture_traffic(name, cfg=REF.traffic_config(name, **LPD))
        for k, v in got.items():
            r = ref[f"{name}/{k}"]
            assert r.shape == v.shape and np.array_equal(r, v, equal_nan=True), (name, k)


@pytest.mark.parametrize("make", [singleton_problem, lambda: toy.knapsack(30)], ids=["singletons", "knapsack"])
def test_inert_instance_has_no_decoder(make):
    """An instance without an active column: warm-up and run with no worker, no lpd_* counter, no event; the start-up
    record says inert. Neither problem has a file, which an active instance would need."""
    p = make()
    before = _children()
    eng = Engine(p, cfg_for(), seed=3)                       # with the warm-up
    st = eng.prepare()
    assert st["lp_decode"]["state"] == "inert" and not eng.lpd_on
    assert all("lp_decode_warmed" not in st["warmup"][k] for k in ("many", "one"))
    eng.begin(time.monotonic(), 3600.0)
    for _ in range(3):
        eng.step()
    out = eng.finish()
    assert eng.lpd is None and _children() == before
    assert not any(k.startswith("lpd_") for k in out) and "lp_decode" not in out
    assert "lpd_src" not in eng.genealogy()


def test_with_pending_without_batch_fields_gives_no_parents_and_zero_ancestry():
    p = toy.knapsack(20)
    eng = Engine(p, cfg_for(), seed=1, warmup=False)
    eng.prepare()
    eng.begin(time.monotonic(), 3600.0)
    X = eng.bk.xp.asarray(np.zeros((2, p.n))).astype(eng.gene_dtype)
    eng.pending.append(_Batch(X, np.full(2, ORIGIN_X_UNIFORM, np.uint16), repair=False, src="test", t=0.0))
    meta = dict(rank_pa=np.zeros(0, np.int8), rank_pb=np.zeros(0, np.int8), t_first=np.zeros(0))
    cand, meta = eng._with_pending(eng.pop.take(eng.bk, np.zeros(0, np.int64)), meta)
    assert cand.size == 2 and np.all(cand.parents == -1)
    assert not cand.anc_rank.any() and not cand.anc_ops.any() and np.all(cand.rank == NO_RANK)


def test_plan_memory_counts_n_more_pending_rows_only_where_the_decode_acts():
    over = {"population.size_rule": "fixed", "population.size": 64, "population.offspring_per_gen": 64}
    for p, active in ((toy.mixed(), True), (singleton_problem(), False), (toy.knapsack(30), False)):
        gdt = gene_dtype_for(p.lb, p.ub, p.isint)
        a = plan_memory(p, EAConfig().with_overrides(over), gdt)
        b = plan_memory(p, EAConfig().with_overrides(dict(over, **LPD)), gdt)
        g = np.dtype(gdt).itemsize
        dd = 8 if np.dtype(gdt).kind in "iu" else 16
        N = EAConfig().repair.lp_decode_per_gen
        d_surv = b.terms["peak_survival"] - a.terms["peak_survival"]
        d_front = b.terms["peak_frontier"] - a.terms["peak_frontier"]
        if active:
            assert d_surv == N * (2 * p.n * g + p.n * dd) and d_front == N * p.n * g
        else:
            assert d_surv == 0 and d_front == 0 and a.terms == b.terms


def test_active_instance_without_its_file_is_refused():
    with pytest.raises(ValueError, match="instance file"):
        Engine(toy.mixed(), cfg_for(), seed=1, warmup=False).prepare()


# ======================================================================================================= worker
@pytest.fixture(scope="module")
def codes_worker(files):
    path, prob = files["codes"]
    dec = _decoder(path, prob)
    _wait(dec)
    yield dec, path, prob
    dec.stop()


@pytest.mark.solver
def test_worker_ready_codes_and_optimum(codes_worker):
    dec, path, prob = codes_worker
    info = dec.ready_info
    assert info["n"] == 10 and info["n_int"] == 6 and info["n_cont"] == 4 and info["nice"] >= L.NICE
    assert info["omp_num_threads"] == "1" and dec.int_dtype == np.int8
    # the worker's own ready time (its clock, sent with "ready") precedes the EA's read of it
    assert 0.0 < info["ready_after_s_worker"] <= info["ready_after_s"] and info["t_sent"] <= info["t_read"]
    good, good2 = [1, 1, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1]
    infeas, screen_fail = [0, 0, 1, 1, 1, 0], [1, 1, 1, 0, 1, 0]
    rows = np.array([good, infeas, good2, screen_fail, good, good2], np.int32)
    kinds = np.array([L.KIND_F, L.KIND_F, L.KIND_I, L.KIND_I, L.KIND_F, L.KIND_I], np.uint8)
    rep = _solve(dec, rows, kinds, per_gen=4)
    codes = [L.CODE_NAMES[c] for c in rep["codes"]]
    # rows 4 and 5 repeat rows 0 and 2 inside the batch; the I row failing the screen never reaches HiGHS
    assert codes == ["ok", "infeasible", "ok", "range_fail", "cached", "cached"], codes
    assert list(rep["ok_rows"]) == [0, 2] and rep["conts"].shape == (2, 4)
    for j, r in enumerate(rep["ok_rows"]):
        st, obj = _fresh_lp_objective(path, prob, rows[r])
        assert st == "Optimal" and rep["obj"][r] == pytest.approx(obj, rel=1e-9, abs=1e-9)
        y = np.zeros(prob.n)
        y[prob.ints], y[prob.conts] = rows[r], rep["conts"][j]
        assert prob.check(y).feasible and prob.objective(y) == pytest.approx(obj, rel=1e-9, abs=1e-9)
    assert rep["limit"][0] == 1.0 and rep["limit"][2] == pytest.approx(0.05 * L.I_LIMIT_FACTOR)   # priming, I warm
    rep2 = _solve(dec, np.array([good, infeas, [1, 1, 0, 0, 0, 0]], np.int32), np.array([0, 0, 1], np.uint8),
                  per_gen=4)
    assert [L.CODE_NAMES[c] for c in rep2["codes"]] == ["cached", "cached", "range_fail"]   # solved before: no LP
    st = rep2["stats"]
    assert st["cache_hits"] == 4 and st["screen_fail"] == 2 and st["solves"] == 3 and st["priming_solves"] == 1
    assert st["priming_first_status"] == st["priming_last_status"] == "Optimal" and st["warm_solves"] == 2
    assert st["priming_first_s"] == pytest.approx(rep["secs"][0]) and st["priming_time_limit_hits"] == 0
    assert np.all(rep["cpu"][[0, 1, 2]] >= 0) and rep["cpu"][3] == 0 and st["solve_cpu_s_total"] > 0
    # the pipe's timing on both sides: the worker's send time, the EA's read time and recv() seconds
    assert rep["t_sent"] <= rep["t_read"] and rep["read_s"] >= 0
    ipc = dec.ipc
    assert ipc["offers"] == ipc["reads"] == 2 and ipc["offer_bytes_max"] == rows.size    # int8: one byte per value
    assert ipc["read_bytes_max"] == rep["conts"].nbytes and ipc["offer_s_max"] >= 0 and ipc["read_s_max"] >= 0


@pytest.mark.solver
def test_worker_budget_split_and_not_solved(files):
    path, prob = files["gen"]
    rs = np.random.default_rng(5)
    rows = rs.integers(0, 2, size=(8, 30)).astype(np.int32)
    rows[:, 0] = 1                                            # every row feasible
    with _decoder(path, prob) as dec:
        _wait(dec)
        rep = _solve(dec, rows, np.array([0, 0, 0, 0, 1, 1, 1, 1], np.uint8), per_gen=4)
        codes = [L.CODE_NAMES[c] for c in rep["codes"]]
        assert codes == ["ok", "ok", "not_solved", "not_solved", "ok", "ok", "not_solved", "not_solved"], codes
        rep = _solve(dec, rs.integers(0, 2, size=(3, 30)).astype(np.int32) | np.eye(1, 30, 0, dtype=np.int32),
                     np.array([0, 1, 1], np.uint8), per_gen=4)
        assert [L.CODE_NAMES[c] for c in rep["codes"]] == ["ok", "ok", "ok"]   # one F row: the I rows take three
        pid = dec.pid
    assert _gone(pid)


def _distinct_rows(seed: int, k: int, n_int: int = 30) -> np.ndarray:
    """k distinct binary integer parts of `gen` with x0 = 1, so every fixed-integer LP is feasible and screens."""
    rs = np.random.default_rng(seed)
    rows = rs.integers(0, 2, size=(k, n_int)).astype(np.int8)
    rows[:, 0] = 1
    assert len({r.tobytes() for r in rows}) == k               # no CACHED repeat inside the batch
    return rows


@pytest.mark.solver
@pytest.mark.parametrize("n_f, n_i, solved_f, solved_i", [(0, 6, 0, 4), (1, 5, 1, 3), (2, 5, 2, 2), (3, 1, 3, 1)])
def test_worker_short_kind_passes_its_share(files, n_f, n_i, solved_f, solved_i):
    """The worker's own budget at N = 4: a kind that is short passes its share to the other — with no polish row all
    four LPs go to completion rows (the screen lets four through), with one polish row three; with three polish rows
    and one completion row the polish rows take three. Rows beyond the budget are NOT_SOLVED. (A generous time limit,
    so contention cannot turn a solved row into a time-limit code: only solved / not solved is asserted.)"""
    path, prob = files["gen"]
    rows = _distinct_rows(13 + n_f + 10 * n_i, n_f + n_i)
    kinds = np.r_[np.full(n_f, L.KIND_F), np.full(n_i, L.KIND_I)].astype(np.uint8)
    with _decoder(path, prob, time_limit=5.0, cold_limit=10.0) as dec:
        _wait(dec)
        rep = _solve(dec, rows, kinds, per_gen=4)
        pid = dec.pid
    codes = [L.CODE_NAMES[c] for c in rep["codes"]]
    solved = [c not in ("not_solved", "cached", "range_fail", "after_deadline") for c in codes]
    assert solved == [True] * solved_f + [False] * (n_f - solved_f) + [True] * solved_i + [False] * (n_i - solved_i), \
        codes
    assert all(c == "not_solved" for c, s in zip(codes, solved) if not s), codes
    assert rep["stats"]["screen_pass"] == min(n_i, max(4 - n_f, 2)) and rep["stats"]["screen_fail"] == 0
    assert _gone(pid)


@pytest.mark.solver
def test_worker_rebases_the_cumulative_time_limit(files):
    """HiGHS's time_limit is cumulative over runs (and wall-clock), and the worker sets time_limit = getRunTime() + L
    at every solve. 100 solves at L = 0.02 s whose total HiGHS run time exceeds L: once the cumulative run time is past
    L, an un-rebased limit would stop every later solve at once (secs ~ 0); rebased, a solve stops only after running
    for about L. So the test fails on a "Time limit reached" row that stopped in less than 0.9 L (the probe's
    "spurious" rule) — a preemption of the niced worker on contended cores can legitimately stop a row at >= L wall
    seconds, and that is allowed."""
    L_s = 0.02
    path, prob = files["big"]
    rs = np.random.default_rng(7)
    rows = rs.integers(0, 2, size=(100, 40)).astype(np.int8)
    rows[:, 0] = 1
    with _decoder(path, prob, time_limit=L_s, cold_limit=L_s) as dec:
        _wait(dec)
        reps = [_solve(dec, rows[s:s + 10], per_gen=10) for s in range(0, 100, 10)]
    st = reps[-1]["stats"]
    assert st["run_time"] > L_s and st["solves"] == 100, st   # otherwise the test proves nothing
    after = [b for b in range(1, len(reps)) if reps[b - 1]["stats"]["run_time"] > L_s]
    assert after, [r["stats"]["run_time"] for r in reps]      # some batches ran entirely after cumulative > L
    spurious = [(b, i, r["status"][i], float(r["secs"][i])) for b, r in enumerate(reps) for i in range(10)
                if r["status"][i] == "Time limit reached" and r["secs"][i] < 0.9 * r["limit"][i]]
    assert spurious == [], spurious
    late = [s for b in after for s in reps[b]["status"]]
    assert "Optimal" in late and set(late) <= {"Optimal", "Time limit reached"}, late
    assert st["time_limit_hits"] == sum(s == "Time limit reached" for r in reps for s in r["status"])
    assert st["time_limit_hits"] == st["warm_time_limit_hits"] + st["priming_time_limit_hits"]


@pytest.mark.solver
def test_worker_priming_limit_and_basis_restore(files):
    """A cold solve stopped by its time limit leaves no anchor, so the next solve is again cold (cold limit). After an
    infeasible solve the last optimal basis is restored: the next row's simplex iterations are those it takes from that
    basis, not from the infeasible solve's."""
    path, prob = files["gen"]
    rs = np.random.default_rng(11)
    A, B, B2 = (rs.integers(0, 2, size=30).astype(np.int8) for _ in range(3))
    A[0] = B[0] = B2[0] = 1
    C = A.copy()
    C[:2] = 0                                                # x0 = x1 = 0: LP infeasible

    def sequence(hooks):
        # generous warm limits: contention must not turn a warm solve into a non-optimal status
        with _decoder(path, prob, time_limit=5.0, cold_limit=10.0, cache_cap=0, hooks=hooks) as dec:
            _wait(dec)
            r1 = _solve(dec, [A])
            assert r1["status"][0] == "Time limit reached" and r1["limit"][0] == 0.0, r1
            r2 = _solve(dec, [A])
            assert r2["limit"][0] == 10.0 and r2["status"][0] == "Optimal"     # still priming: the cold limit
            r3 = _solve(dec, [B2])
            assert r3["limit"][0] == 5.0 and r3["status"][0] == "Optimal"
            assert [r["cold"].tolist() for r in (r1, r2, r3)] == [[True], [True], [False]]   # priming, per row
            ok = _solve(dec, [A, B])
            bad = _solve(dec, [A, C, B])
        assert [L.CODE_NAMES[c] for c in bad["codes"]] == ["ok", "infeasible", "ok"]
        return ok, bad
    ok, bad = sequence({"cold_limits": [0.0]})
    # single-thread HiGHS from the same (restored) basis takes exactly the iterations it took from it before
    assert bad["iters"][2] == ok["iters"][1], (ok["iters"], bad["iters"])
    st = bad["stats"]
    assert st["restores"] == 1 and st["restore_errors"] == 0
    assert st["priming_solves"] == 2 and st["time_limit_hits"] == st["priming_time_limit_hits"] == 1
    assert st["priming_first_status"] == "Time limit reached" and st["priming_last_status"] == "Optimal"
    assert st["warm_solves"] == 6 and st["warm_time_limit_hits"] == 0
    # negative control: without the restore the same row starts from the infeasible solve's basis (seed 11: 15 vs 13)
    ok_n, bad_n = sequence({"cold_limits": [0.0], "no_restore": True})
    assert ok_n["iters"][1] == ok["iters"][1] and bad_n["stats"]["restores"] == 0
    assert bad_n["iters"][2] != ok_n["iters"][1], (ok_n["iters"], bad_n["iters"])


@pytest.mark.solver
def test_worker_counts_warm_time_limit_hits(files):
    """D-017 M2's numerator: an optimal priming solve sets the anchor; two later rows run warm under a zero limit and
    stop at it. The warm, priming and pooled hit counters and the per-row `cold` flag say exactly that."""
    path, prob = files["gen"]
    rows = _distinct_rows(13, 3)
    with _decoder(path, prob, time_limit=0.0, cold_limit=10.0) as dec:
        _wait(dec)
        r1 = _solve(dec, rows[:1])
        r2 = _solve(dec, rows[1:])
    assert r1["status"][0] == "Optimal" and r1["limit"][0] == 10.0 and r1["cold"].tolist() == [True], r1
    assert list(r2["status"]) == ["Time limit reached"] * 2 and r2["limit"].tolist() == [0.0, 0.0], r2
    assert r2["cold"].tolist() == [False, False]
    st = r2["stats"]
    assert st["priming_solves"] == 1 and st["warm_solves"] == 2 and st["solves"] == 3
    assert st["warm_time_limit_hits"] == 2 and st["priming_time_limit_hits"] == 0
    assert st["time_limit_hits"] == st["warm_time_limit_hits"] + st["priming_time_limit_hits"] == 2


@pytest.mark.solver
def test_worker_digest_mismatch_fails(files):
    path, prob = files["gen"]
    dec = L.LPDecoder(path, "0" * 32, "int8", 4, 0.05, 1.0, 1e-6).start()
    msgs = _wait(dec, states=("failed",))
    assert dec.state == "failed" and "digest" in dec.error and any(m[0] == "failed" for m in msgs)
    assert not dec.offer(1, np.zeros((1, 30), np.int8), [0]) and dec.poll() == []
    # the failure only signalled the worker (the EA never waits on it); stop() joins it, after the clock
    assert dec.stop() is False and dec.state == "failed" and _gone(dec.pid)


@pytest.mark.solver
def test_worker_poll_never_blocks_and_eof_ends_it(files):
    path, prob = files["gen"]
    dec = _decoder(path, prob, hooks={"sleep_s": 0.6})
    _wait(dec)
    rows = np.ones((2, 30), np.int32)
    rows[1, 5] = 0
    assert dec.offer(1, rows, [0, 0]) and not dec.offer(2, rows, [0, 0])       # one request at a time
    lat = []
    t_end = time.monotonic() + 0.8
    while time.monotonic() < t_end:                          # the batch takes >= 1.2 s
        t = time.perf_counter()
        assert dec.poll() == []
        lat.append(time.perf_counter() - t)
        time.sleep(0.02)
    assert dec.state == "busy" and max(lat) < 0.005, max(lat)
    msgs = _wait(dec, timeout=30)
    assert [m[0] for m in msgs] == ["result"]
    proc = dec._proc
    dec._conn.close()                                        # EOF: the worker leaves its recv and exits by itself
    dec._conn = None
    proc.join(timeout=20)
    assert proc.exitcode == 0
    dec.stop()
    assert _gone(dec.pid)


class _FakeProc:
    """Stands in for the worker's Process in LPDecoder unit tests (no process is started): records the calls."""

    def __init__(self, alive=True, on_is_alive=None):
        self.alive, self.on_is_alive = alive, on_is_alive
        self.calls: list = []
        self.exitcode = None if alive else 0
        self.pid = None

    def is_alive(self):
        self.calls.append("is_alive")
        if self.on_is_alive is not None:
            f, self.on_is_alive = self.on_is_alive, None
            f(self)
        return self.alive

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")

    def join(self, timeout=None):
        self.calls.append(("join", timeout))


def _bare_decoder(proc, state="busy"):
    dec = L.LPDecoder("<none>", "0" * 32, "int8", 4, 2.0, 10.0, 1e-6)
    parent, child = mp.Pipe(duplex=True)
    dec._proc, dec._conn, dec._t_start, dec.state = proc, parent, time.monotonic(), state
    dec.outstanding = 1 if state == "busy" else None
    return dec, child


def test_poll_survives_a_failure_sent_just_before_the_worker_exits():
    """The race of the review (2026-10-08): the worker sends ("failed", ...) and exits after poll()'s first read found
    nothing but before is_alive() ran. The drain after is_alive() must handle it and return the failure — not call
    poll() on the pipe that the failure closed (AttributeError, which would have ended the EA)."""
    def send_and_exit(p):
        child.send(("failed", "boom", "tb"))
        child.close()
        p.alive, p.exitcode = False, 1
    proc = _FakeProc(alive=True, on_is_alive=send_and_exit)
    dec, child = _bare_decoder(proc)
    out = dec.poll()
    assert out == [("failed", "boom", "tb")] and dec.state == "failed" and dec.error == "boom"
    assert dec._conn is None and dec.poll() == []


def test_fail_never_waits_for_the_worker():
    """A failure on the EA thread signals the worker and closes the pipe, without a join; poll() reaps it later
    without blocking; only stop() (after the clock) joins."""
    proc = _FakeProc(alive=True)
    dec, child = _bare_decoder(proc)
    child.send(("failed", "boom", "tb"))
    t = time.perf_counter()
    out = dec.poll()
    assert time.perf_counter() - t < 0.5
    assert out == [("failed", "boom", "tb")] and dec.state == "failed" and dec._conn is None
    assert "terminate" in proc.calls and not any(isinstance(c, tuple) for c in proc.calls), proc.calls
    proc.alive, proc.exitcode = False, -15                    # the worker died of the signal: reaped without a join
    assert dec.poll() == [] and dec.exitcode == -15 and not any(isinstance(c, tuple) for c in proc.calls)
    assert dec.stop() is False and dec.state == "failed"
    assert any(isinstance(c, tuple) and c[0] == "join" for c in proc.calls)   # stop() joins
    child.close()


@pytest.mark.solver
def test_offer_to_a_dead_worker_fails_without_waiting(files):
    """The worker dies while idle (an OOM kill, a segfault) between a poll and the next offer: the send fails, the
    decoder is "failed" with the error, and nothing waits."""
    import signal
    path, prob = files["gen"]
    dec = _decoder(path, prob)
    try:
        _wait(dec)
        os.kill(dec.pid, signal.SIGKILL)
        dec._proc.join(10)                                    # dead and reaped: its end of the pipe is closed
        ok = dec.offer(1, np.ones((1, 30), np.int8), [0])
        assert not ok and dec.state == "failed" and "send failed" in dec.error, (dec.state, dec.error)
        assert dec.poll() == []
    finally:
        dec.stop()
    assert _gone(dec.pid)


class _SendFailStub(E._WarmupDecoder):
    """A decoder whose send fails at its first offer, as LPDecoder.offer reports it (state "failed", `error`)."""

    error = None

    def offer(self, req_id, rows, kinds, per_gen=None):
        if self.state != "idle":
            return False
        self.state, self.error = "failed", "send failed: BrokenPipeError(32, 'Broken pipe')"
        return False


def test_engine_records_a_failed_send(files):
    """A failed send reaches the engine like any worker failure: `lpd_failed`, the error, a "failed" event; no
    request is counted (`lpd_req` and `lpd_offers` stay 0)."""
    path, prob = files["gen"]
    eng = Engine(prob, cfg_for(), seed=2, warmup=False, decoder_factory=_SendFailStub)
    eng.prepare()
    events = []
    real_event = eng.trace.event
    eng.trace.event = lambda kind, t=None, **d: (events.append((kind, d)), real_event(kind, t, **d))
    eng.begin(time.monotonic(), 3600.0)
    _run_until(eng, lambda e: e.lpd_failed, timeout=30)
    for _ in range(2):
        eng.step()
    out = eng.finish()
    assert out["lpd_failed"] == 1 and out["lp_decode"]["state"] == "failed"
    assert "send failed" in out["lp_decode"]["error"] and eng.lpd_req == 0 and out["lpd_offers"] == 0
    assert [d["error"] for k, d in events if k == "lp_decode" and d.get("what") == "failed"] == [eng.lpd_error]


@pytest.mark.solver
def test_worker_no_solve_after_the_deadline_and_stop_is_idempotent(files):
    path, prob = files["gen"]
    dec = _decoder(path, prob, deadline=time.monotonic() - 1.0)
    _wait(dec)
    rep = _solve(dec, np.ones((3, 30), np.int32) - np.eye(3, 30, dtype=np.int32), per_gen=4)
    assert [L.CODE_NAMES[c] for c in rep["codes"]] == ["after_deadline"] * 3
    assert rep["stats"]["solves"] == 0 and rep["stats"]["after_deadline"] == 3
    assert dec.offer(9, np.ones((1, 30), np.int32), [0])
    assert dec.stop() is True                                # the outstanding request is lost
    assert dec.stop() is False and dec.state == "stopped" and not dec.alive
    assert _gone(dec.pid)


# ======================================================================================================== engine
def _run_until(eng, cond, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not cond(eng):
        eng.step()
        time.sleep(0.002)


class _PostMailbox:
    """A partner that sends nothing and takes the EA's posts: `to_solver` is on, so incumbents made from decoded
    copies are posted (with no solver incumbent every new incumbent beats it) and `lpd_posts` counts."""

    def __init__(self):
        self.to_ea, self.lp, self.lps, self.to_solver = E._Chan(), E._Chan(), E._Chan(), E._Chan()
        self.info = E._Info(None)

    def stopped(self) -> bool:
        return False

    def verdicts(self) -> list:
        return []


SOLVED_CODES = ("ok", "infeasible", "time_limit", "other")   # rows that reached HiGHS


@pytest.mark.solver
def test_engine_decoded_copies(files, tmp_path, monkeypatch):
    """The real worker on the generated toy: copies carry ORIGIN_LP_DECODE, their source's strategy, no parents and
    `lpd_src`; their integer parts are their sources' (rounded), their continuous values LP-optimal; polish copies beat
    their sources by imp_tol; every decoded incumbent passes the float64 check; the counters add up; the genealogy
    goes through the lineage analysis; at most one decoded batch joins a candidate set; the per-kind counters agree
    with independent counts (entered: genealogy + the result rows' kinds; posts: the posted ids), the worker's M2
    statistics are recounted from the result rows, M0's ready time is the worker's own stamp, M3's gains of the copies
    that entered are recomputed from the rows; the worker is gone after finish."""
    from scipy.optimize import linprog
    from itrea.analysis.lineage import run_tables
    from itrea.trace import Trace, read_events
    path, prob = files["gen"]
    cfg = cfg_for(**{"strategy.enabled": True, "log_every": 1})
    tr = Trace(tmp_path, "ea")
    eng = Engine(prob, cfg, seed=5, mailbox=_PostMailbox(), trace=tr, warmup=False)
    eng.prepare()
    assert eng.startup["lp_decode"]["state"] == "configured" and eng.startup["lp_decode"]["active_columns"] == 60
    src_x, batches = {}, []
    real_offer, real_result = eng._lpd_offer, eng._lpd_result

    def offer(cand):
        before = eng.lpd_req
        real_offer(cand)
        info = eng.lpd_req_info
        if info is not None and info["req"] > before:
            pos = {int(v): i for i, v in enumerate(cand.ids)}
            for sid in info["ids"]:
                i = pos[int(sid)]
                src_x[int(sid)] = (np.asarray(cand.X[i], float).copy(), int(cand.strat[i]))

    def result(req_id, rep, site="exchange"):
        n0 = len(eng.pending)
        real_result(req_id, rep, site=site)
        batches.extend(b for b in eng.pending[n0:] if b.src == "lp_decode")
    per_set = []                                             # (generation, decoded batches, decoded rows) per set
    real_with = eng._with_pending

    def with_pending(cand, meta):
        d = [b for b in eng.pending if b.src == "lp_decode"]
        per_set.append((eng.gen, len(d), sum(int(b.X.shape[0]) for b in d)))
        return real_with(cand, meta)
    # per survival: the increments of lpd_entered_F / _I and the ids that entered; per new incumbent: its id and the
    # increments of lpd_posts_F / _I; every post's ids — checked below against the kinds of the result rows
    ent_log, inc_log, posted = [], [], []
    real_survive, real_new_inc, real_post = eng._survive, eng._new_incumbent, eng._post

    def survive(cand, Np, meta):
        f0, i0, n0 = eng.c["lpd_entered_F"], eng.c["lpd_entered_I"], len(eng.gen_rows)
        real_survive(cand, Np, meta)
        ent_log.append((eng.c["lpd_entered_F"] - f0, eng.c["lpd_entered_I"] - i0,
                        [int(v) for r in eng.gen_rows[n0:] for v in r["id"]]))

    def new_inc(i, x, obj, key):
        f0, i0, iid = eng.c["lpd_posts_F"], eng.c["lpd_posts_I"], int(eng.pop.ids[i])
        real_new_inc(i, x, obj, key)
        inc_log.append((iid, eng.c["lpd_posts_F"] - f0, eng.c["lpd_posts_I"] - i0))

    def post(X, obj, origin, ids, gain=math.nan, frontier=False):
        posted.append(([int(v) for v in ids], bool(frontier)))
        return real_post(X, obj, origin, ids, gain=gain, frontier=frontier)
    monkeypatch.setattr(eng, "_lpd_offer", offer)
    monkeypatch.setattr(eng, "_lpd_result", result)
    monkeypatch.setattr(eng, "_with_pending", with_pending)
    monkeypatch.setattr(eng, "_survive", survive)
    monkeypatch.setattr(eng, "_new_incumbent", new_inc)
    monkeypatch.setattr(eng, "_post", post)
    before = _children()
    eng.begin(time.monotonic(), 3600.0)
    assert eng.to_solver
    pid = eng.lpd.pid
    # ... and until some survival admitted unequal numbers of F and I copies (a swapped split then shows)
    _run_until(eng, lambda e: e.c["lpd_entered"] >= 3 and e.c["lpd_incumbents"] >= 1 and e.c["lpd_kept_F"] >= 2
               and e.c["lpd_replies"] >= 5 and e.c["lpd_posts"] >= 1 and any(a != b for a, b, _ in ent_log))
    out = eng.finish()
    eng.write_outputs(tmp_path)
    tr.close(sense=prob.sense, **{k: v for k, v in out.items() if k not in ("best_x", "frontier")})
    assert _gone(pid) and eng.lpd.state == "stopped" and _children() == before
    assert out["lpd_entered"] >= 3 and out["lpd_incumbents"] >= 1 and out["lpd_kept_F"] >= 2, out
    # the copies
    ints, conts = prob.ints, prob.conts
    for b in batches:
        assert np.all(b.origin == ORIGIN_LP_DECODE) and b.repair is False
        X = np.asarray(b.X, float)
        for r, sid in enumerate(b.src_ids):
            xs, strat = src_x[int(sid)]
            assert int(b.strat[r]) == strat
            assert np.array_equal(X[r, ints], np.clip(np.round(xs[ints]), prob.lb[ints], prob.ub[ints]))
            bounds = [(lo, hi) for lo, hi in zip(prob.lb, prob.ub)]
            for j in ints:
                bounds[j] = (X[r, j], X[r, j])
            lp = linprog(prob.c, A_ub=np.vstack([prob.A.toarray()[np.isfinite(prob.ru)],
                                                 -prob.A.toarray()[np.isfinite(prob.rl)]]),
                         b_ub=np.r_[prob.ru[np.isfinite(prob.ru)], -prob.rl[np.isfinite(prob.rl)]],
                         bounds=bounds, method="highs")
            assert lp.status == 0 and prob.objective(X[r]) == pytest.approx(lp.fun, rel=1e-7, abs=1e-7)
            assert prob.check(X[r]).feasible
    # the events: polish copies beat their sources, codes add up
    ev = [e for e in read_events(tmp_path / "ea.events.jsonl") if e["kind"] == "lp_decode"]
    assert ev[0]["what"] == "ready" and sum(e["what"] == "result" for e in ev) == out["lpd_replies"]
    rows = [r for e in ev if e["what"] == "result" for r in e["rows"]]
    for r in rows:
        if r["kind"] == "F" and r["kept"]:
            assert r["key"] < r["key_src"] - 1e-9 * max(1.0, abs(r["key_src"]))
    # a source is screened or solved at most once (NOT_SOLVED / AFTER_DEADLINE rows may be offered again)
    done = [r["src"] for r in rows if r["code"] not in ("not_solved", "after_deadline")]
    assert len(done) == len(set(done))
    # at most one decoded batch (at most N rows) per candidate set; copies enter the candidate set of the generation
    # that read the reply (site "candidates" / "exchange"), or the next one when it was read at the pick site, which
    # then offers nothing (lpd_skip_pending); a batch never joins one only when the run ends first (lpd_dropped)
    N = cfg.repair.lp_decode_per_gen
    assert max(n for _, n, _ in per_set) == 1 and max(k for _, _, k in per_set) <= N, per_set
    res = [e for e in ev if e["what"] == "result"]
    entering = {g: k for g, n, k in per_set if n}
    for e in res:
        assert e["wait_s"] is not None and e["wait_s"] >= 0 and e["t_sent"] <= e["t"] + 1e-6 and e["read_s"] >= 0
        if e["gen_enters"] is None:
            assert e["kept"] == 0 or e["site"] == "finish"
            continue
        if e["gen_enters"] > eng.gen:                        # read at the last pick site: the run ended first
            continue
        lag = e["gen_enters"] - e["gen_offered"]
        assert lag == (e["gen"] - e["gen_offered"]) + (1 if e["site"] == "pick" else 0) and lag >= 1, e
        assert entering.get(e["gen_enters"]) == e["kept"], (e["gen_enters"], entering)
    # replies are read just before the candidate set (the exact lags are pinned by the stub tests below: the niced
    # worker on contended cores may answer several generations late)
    assert any(e["site"] == "candidates" for e in res), [(e["site"], e["gen_offered"], e["gen_enters"]) for e in res]
    assert out["lpd_skip_pending"] == sum(e["site"] == "pick" and e["kept"] > 0 for e in res)
    dropped = [e for e in ev if e["what"] == "dropped"]
    assert out["lpd_kept"] == sum(k for _, _, k in per_set) + out["lpd_dropped"]
    assert out["lpd_dropped"] == sum(e["rows"] for e in dropped)
    assert {e["req"] for e in dropped} == {e["req"] for e in res if e["kept"] and (e["gen_enters"] is None
                                                                                 or e["gen_enters"] > eng.gen)}
    # genealogy
    g = eng.genealogy()
    dec = g["lpd_src"] >= 0
    assert dec.sum() == out["lpd_entered"]
    assert np.all(g["pa"][dec] == -1) and np.all(g["pb"][dec] == -1) and np.all(g["rank_pa"][dec] == -1)
    assert np.all(g["origin"][dec] & ORIGIN_LP_DECODE)
    all_src = {int(s) for b in batches for s in b.src_ids}
    assert set(int(s) for s in g["lpd_src"][dec]) <= all_src
    # decoded incumbents pass the float64 check
    n_dec = 0
    for iid, key, obj, x in eng.inc_stack:
        if iid in eng.lpd_ids:
            n_dec += 1
            assert prob.check(x).feasible and prob.objective(x) == pytest.approx(obj)
    assert n_dec == out["lpd_incumbents"] >= 1
    # counters
    c = out
    by_code = sum(c[f"lpd_{nm}"] for nm in L.CODE_NAMES)
    assert c["lpd_replies"] + c["lpd_lost"] == c["lpd_offers"]
    assert by_code <= c["lpd_rows_offered"] <= by_code + 16 * c["lpd_lost"]
    assert c["lpd_rows_F"] + c["lpd_rows_I"] == c["lpd_rows_offered"]
    assert c["lpd_solved"] == c["lpd_ok"] + c["lpd_infeasible"] + c["lpd_time_limit"] + c["lpd_other"]
    assert c["lpd_ok"] == c["lpd_kept"] + c["lpd_no_gain"] and c["lpd_kept"] == c["lpd_kept_F"] + c["lpd_kept_I"]
    assert c["lpd_entered"] <= c["lpd_kept"] and c["lpd_failed"] == 0
    # per kind of the copy's source (D-017 M3 / M4)
    for k in ("solved", "ok", "kept", "entered", "incumbents", "posts"):
        assert c[f"lpd_{k}_F"] + c[f"lpd_{k}_I"] == c[f"lpd_{k}"], k
    assert c["lpd_kept_F"] + c["lpd_no_gain"] == c["lpd_ok_F"] and c["lpd_kept_I"] == c["lpd_ok_I"]
    assert c["lpd_ok_F"] == sum(r["kind"] == "F" and r["code"] == "ok" for r in rows)
    kind_of_src = {r["src"]: r["kind"] for r in rows}
    for copy_id, src in eng.lpd_src_of.items():
        assert ("F" if eng.lpd_kind_of[copy_id] == L.KIND_F else "I") == kind_of_src[src]
    n_dec_F = sum(1 for iid, *_ in eng.inc_stack if iid in eng.lpd_ids and eng.lpd_kind_of[iid] == L.KIND_F)
    assert n_dec_F == c["lpd_incumbents_F"]
    # entered and posts per kind against independent counts (the kind from the result rows, through lpd_src), per
    # survival and per new incumbent, so a swapped split shows wherever the two kinds differ
    ent_kinds = [kind_of_src[int(s)] for s in g["lpd_src"][dec]]
    assert (c["lpd_entered_F"], c["lpd_entered_I"]) == (ent_kinds.count("F"), ent_kinds.count("I"))
    for dF, dI, ids in ent_log:
        kinds = [kind_of_src[eng.lpd_src_of[i]] for i in ids if i in eng.lpd_src_of]
        assert (dF, dI) == (kinds.count("F"), kinds.count("I")), (dF, dI, kinds)
    assert any(dF != dI for dF, dI, _ in ent_log)
    posted_inc = {ids[0] for ids, frontier in posted if not frontier}
    n_posts = 0
    for iid, dF, dI in inc_log:
        if iid in eng.lpd_ids and iid in posted_inc:
            n_posts += 1
            assert (dF, dI) == ((1, 0) if kind_of_src[eng.lpd_src_of[iid]] == "F" else (0, 1)), (iid, dF, dI)
        else:
            assert (dF, dI) == (0, 0), (iid, dF, dI)
    assert n_posts == c["lpd_posts"] >= 1
    # the funnel per kind: cached rows, I rows the screen looked at (= the worker's passes + fails), distinct sources
    assert c["lpd_cached_F"] + c["lpd_cached_I"] == c["lpd_cached"]
    for kd in ("F", "I"):
        assert c[f"lpd_cached_{kd}"] == sum(r["kind"] == kd and r["code"] == "cached" for r in rows)
        assert c[f"lpd_srcs_{kd}"] == len({r["src"] for r in rows if r["kind"] == kd and r["code"] not in
                                           ("not_solved", "after_deadline")})
    assert c["lpd_screened_I"] == sum(r["kind"] == "I" and r["code"] not in ("cached", "not_solved") for r in rows)
    assert c["lpd_srcs_F"] + c["lpd_srcs_I"] == len(eng.lpd_offered)
    blk = out["lp_decode"]
    assert blk["state"] in ("idle", "busy") and blk["worker"]["solves"] >= c["lpd_ok"] and blk["host_s"] > 0
    assert blk["ready"]["nice"] >= L.NICE and blk["host_s_max"] <= blk["host_s"]
    # M0: the worker's own ready time (its send stamp), which precedes the EA's read of it; M1 / IPC fields
    assert blk["ready_t_worker"] == pytest.approx(eng.lpd_ready_info["t_sent"] - tr.t0, abs=1e-5)
    assert 0 < blk["ready_t_worker"] < blk["ready_t"] and "t_sent" not in blk["ready"]
    assert 0 <= blk["sync_s_max"] <= blk["sync_s"] + 1e-12 and 0 <= blk["wait_s_max"] <= blk["wait_s_total"] + 1e-12
    assert blk["ipc"]["offers"] == c["lpd_offers"] and blk["ipc"]["reads"] == c["lpd_replies"]
    assert 1 <= blk["batch_rows_max"] <= N
    w = blk["worker"]
    assert w["priming_first_status"] == "Optimal" and w["warm_solves"] + w["priming_solves"] == w["solves"]
    # M2 recounted from the result rows (the same replies as the statistics: those read): a solve = a row that reached
    # HiGHS, priming = `cold`, a hit = the status "Time limit reached" (also on an OK row)
    sv = [r for r in rows if r["code"] in SOLVED_CODES]
    assert all(isinstance(r["cold"], bool) for r in sv) and sum(r["cold"] for r in sv) == w["priming_solves"] >= 1
    assert sum(not r["cold"] for r in sv) == w["warm_solves"] >= 1 and len(sv) == w["solves"]
    hit = [r for r in sv if r["status"] == "Time limit reached"]
    assert sum(not r["cold"] for r in hit) == w["warm_time_limit_hits"]
    assert sum(r["cold"] for r in hit) == w["priming_time_limit_hits"]
    assert c["lpd_screened_I"] == w["screen_pass"] + w["screen_fail"]
    # M3: the gains of the copies that entered, recomputed from the result rows joined through the genealogy
    kept_row = {r["src"]: r for r in rows if r["kept"]}                      # a kept row's source is unique
    for kd in ("F", "I"):
        gains = [(r["key_src"] - r["key"]) / max(1.0, abs(r["key_src"]))
                 for r in (kept_row[int(s)] for s in g["lpd_src"][dec]) if r["kind"] == kd]
        assert blk[f"gain_{kd}_entered_n"] == len(gains) == c[f"lpd_entered_{kd}"]
        if gains:
            assert blk[f"gain_{kd}_entered_median"] == pytest.approx(float(np.median(gains)), rel=1e-12, abs=1e-15)
        else:
            assert blk[f"gain_{kd}_entered_median"] is None
    assert blk["gain_F_entered_median"] > 0                                  # polish copies beat their sources
    from itrea.analysis import build_runs as B                # every field the derived table reads exists
    assert all(key in blk for _, key in B.LPD_BLOCK_COLS) and all(key in blk["ipc"] for _, key in B.LPD_IPC_COLS)
    assert all(key in w for _, key in B.LPD_WORKER_COLS) and all(k in out for k in B.LPD_INT_COLS)
    gens = [e for e in read_events(tmp_path / "ea.events.jsonl") if e["kind"] == "gen"]
    assert all(all(k in e for k in E.LPD_GEN_FIELDS) and "lpd_host_s" in e and "lpd_sync_s" in e for e in gens)
    assert gens[-1]["lpd_entered_F"] + gens[-1]["lpd_entered_I"] == gens[-1]["lpd_entered"] == c["lpd_entered"]
    lin = run_tables(tmp_path)
    assert lin["rows"] == len(g["id"]) and lin["cases_without_parents"] >= 0


@pytest.mark.solver
def test_engine_stops_the_worker_after_an_exception(files, monkeypatch):
    path, prob = files["gen"]
    eng = Engine(prob, cfg_for(), seed=2, warmup=False)
    eng.prepare()
    real, n = eng._step, {"k": 0}

    def boom():
        n["k"] += 1
        if n["k"] == 5:
            raise RuntimeError("injected")
        real()
    monkeypatch.setattr(eng, "_step", boom)
    pids = []
    real_begin = eng._lpd_begin

    def begin():
        real_begin()
        pids.append(eng.lpd.pid)
    monkeypatch.setattr(eng, "_lpd_begin", begin)
    before = _children()
    with pytest.raises(RuntimeError, match="injected"):
        eng.run(3600.0, time.monotonic())
    assert pids and _gone(pids[0]) and _children() == before


@pytest.mark.solver
def test_engine_run_to_its_budget_stops_the_worker(files):
    """`run()` to its budget: the counters add up (at most the last request is lost) and the worker is gone."""
    path, prob = files["gen"]
    eng = Engine(prob, cfg_for(), seed=6, warmup=False)
    eng.prepare()
    pids = []
    real_begin = eng._lpd_begin

    def begin():
        real_begin()
        pids.append(eng.lpd.pid)
    eng._lpd_begin = begin
    out = eng.run(3.0, time.monotonic())
    blk = out["lp_decode"]
    assert out["exit_reason"] == "budget" and blk["state"] in ("idle", "busy") and out["lpd_replies"] >= 1
    assert out["lpd_replies"] + out["lpd_lost"] == out["lpd_offers"] and out["lpd_lost"] <= 1
    assert blk["worker"]["requests"] == out["lpd_replies"]       # the stats are those of the last reply read
    assert pids and _gone(pids[0]) and eng.lpd.state == "stopped"


@pytest.mark.solver
def test_finish_reads_the_reply_waiting_at_the_end(files):
    """finish() reads a reply that is waiting in the pipe before it stops the worker — also when a decoded batch read
    at the last pick site is still pending (finish reads without the one-batch hold) — so it is not counted lost and
    the summary's statistics are those of the last request. Deterministic: the step that offered is the last one, and
    the test waits until the reply is in the pipe without reading it."""
    path, prob = files["gen"]
    eng = Engine(prob, cfg_for(), seed=6, warmup=False)
    eng.prepare()
    eng.begin(time.monotonic(), 3600.0)
    pid = eng.lpd.pid
    end = time.monotonic() + 60.0
    while eng.lpd_req_info is None and time.monotonic() < end:
        eng.step()
    assert eng.lpd_req_info is not None and eng.lpd.state == "busy"
    assert eng.lpd._conn.poll(30)                             # the reply is there; nothing has read it
    # as if the last pick site had read a previous reply: a decoded batch is pending (it never enters now)
    X = eng.bk.xp.asarray(np.zeros((1, prob.n))).astype(eng.gene_dtype)
    eng.pending.append(_Batch(X, np.full(1, ORIGIN_LP_DECODE, np.uint16), repair=False, src="lp_decode", t=0.0,
                              src_ids=np.array([-5]), src_kinds=np.array([L.KIND_I], np.uint8)))
    out = eng.finish()
    assert out["lpd_lost"] == 0 and out["lpd_replies"] == out["lpd_offers"] >= 1
    assert out["lp_decode"]["worker"]["requests"] == out["lpd_offers"] and out["lp_decode"]["state"] == "idle"
    assert _gone(pid) and eng.lpd.state == "stopped"


@pytest.mark.solver
def test_engine_worker_failure_is_recorded_and_never_waited_on(files):
    path, prob = files["gen"]

    def bad(eng):
        return L.LPDecoder(path, "f" * 32, eng.lpd_int_dtype, 4, 2.0, 10.0, 1e-6, eng.deadline)
    eng = Engine(prob, cfg_for(), seed=2, warmup=False, decoder_factory=bad)
    eng.prepare()
    eng.begin(time.monotonic(), 3600.0)
    _run_until(eng, lambda e: e.lpd_failed, timeout=60)
    for _ in range(3):
        eng.step()
    out = eng.finish()
    assert out["lpd_failed"] == 1 and out["lp_decode"]["state"] == "failed" and "digest" in out["lp_decode"]["error"]
    assert out["lpd_offers"] == 0 and _gone(eng.lpd.pid)


def test_warmup_runs_the_decode_path_with_the_stub(files, monkeypatch):
    """The warm-up drives the whole decode path on the synthetic problem with the in-process stub: batches of three
    rows ("many") and of one row ("one") enter a candidate set; no process is started before t0."""
    path, prob = files["gen"]
    seen = []
    real = Engine._lpd_entering

    def spy(self, ids, src_ids, kinds=None, gains=None):
        seen.append((str(self.prob.path), self.lpd_warm_rows, len(ids)))
        return real(self, ids, src_ids, kinds, gains)
    monkeypatch.setattr(Engine, "_lpd_entering", spy)
    before = _children()
    eng = Engine(prob, cfg_for(), seed=4)
    st = eng.prepare()
    assert _children() == before
    w = st["warmup"]
    # the start-up record: the largest decoded batch (rows) per pass — several rows in "many", one in "one"
    assert w["many"]["lp_decode_warmed"] >= 2 and w["one"]["lp_decode_warmed"] == 1, w
    assert w["many"]["lp_decode_warmed_batches"] >= 1 and w["one"]["lp_decode_warmed_batches"] >= 1, w
    assert {p for p, _, _ in seen} == {"<synthetic warm-up>"}
    assert {k for _, r, k in seen if r == 3} == {3} and {k for _, r, k in seen if r == 1} == {1}
    assert w["many"]["lp_decode_warmed"] == 3
    assert not hasattr(eng, "c")                             # nothing of the instance before t0


def test_warmup_stub_answers_on_the_next_poll():
    eng = Engine(toy.mixed(), cfg_for(), seed=1, warmup=False, decoder_factory=E._WarmupDecoder)
    eng.prepare()
    stub = E._WarmupDecoder(eng).start()
    assert stub.state == "starting" and not stub.offer(1, np.zeros((1, 2)), [0])
    assert stub.poll()[0][0] == "ready" and stub.state == "idle"
    assert stub.offer(1, np.zeros((2, 2)), [0, 1]) and stub.state == "busy"
    (kind, req, rep), = stub.poll()
    assert kind == "result" and req == 1 and list(rep["codes"]) == [L.OK, L.OK] and rep["conts"].shape == (2, 2)
    assert stub.poll() == [] and stub.stop() is False


# ============================================================================== engine: replies, without a worker
def _stub_engine(prob, lpd_warm_rows=0, **over):
    eng = Engine(prob, cfg_for(**over), seed=1, warmup=False, decoder_factory=E._WarmupDecoder,
                 lpd_warm_rows=lpd_warm_rows)
    eng.prepare()
    eng.begin(time.monotonic(), 3600.0)
    return eng


def _req_info(eng, rows, kinds, keys):
    k = len(kinds)
    kinds = np.asarray(kinds, np.uint8)
    return dict(req=77, gen=eng.gen, t=0.0, ids=np.arange(k, dtype=np.int64) + 1000, kinds=kinds,
                rows=np.asarray(rows, float), key=np.asarray(keys, float), V=np.zeros(k), feasible=kinds == L.KIND_F,
                strat=np.zeros(k, np.uint8), anc_rank=np.zeros((k, eng.depth), np.uint8),
                anc_ops=np.zeros((k, eng.depth), np.uint16), rank=np.full(k, -1, np.int8))


def _reply(codes, ok_rows, conts):
    k = len(codes)
    return dict(codes=np.asarray(codes, np.uint8), ok_rows=np.asarray(ok_rows, np.int64),
                conts=np.asarray(conts, float).reshape(len(ok_rows), -1), obj=np.zeros(k),
                iters=np.zeros(k, np.int64), secs=np.zeros(k), limit=np.zeros(k), status=["Optimal"] * k, stats={})


@pytest.mark.parametrize("warm_rows", [0, 1])
def test_no_gain_filter_keeps_only_polish_copies_that_beat_their_source(files, warm_rows):
    """D-017's F acceptance: a polish copy is kept only if its key beats its source's by imp_tol·max(1, |key|) — not
    when equal, not when better by less than that; a completion copy always. The warm-up (lpd_warm_rows > 0) keeps
    every copy, so its batches have the forced size."""
    path, prob = files["gen"]
    eng = _stub_engine(prob, lpd_warm_rows=warm_rows)
    imp = eng.cfg.ranks.imp_tol
    rs = np.random.default_rng(3)
    rows = _distinct_rows(21, 4).astype(float)
    conts = rs.uniform(0.0, 3.0, size=(4, len(prob.conts)))
    keys = []
    for r in range(4):                                        # the copies' keys, as the engine computes them
        y = np.zeros(prob.n)
        y[prob.ints], y[prob.conts] = rows[r], conts[r]
        keys.append(float(prob.key(y)))
    k0, k1, k2, k3 = keys
    src = [k0,                                                # F, equal: no gain
           k1 + 0.5 * imp * max(1.0, abs(k1)),                # F, better by half the tolerance: no gain
           k2 + 1.0,                                          # F, better by 1: kept
           k3 - 1.0]                                          # I, worse than its source: kept anyway
    eng.lpd_req_info = _req_info(eng, rows, [L.KIND_F, L.KIND_F, L.KIND_F, L.KIND_I], src)
    eng.pending = []
    eng._lpd_result(77, _reply([L.OK] * 4, [0, 1, 2, 3], conts), site="candidates")
    (b,) = [b for b in eng.pending if b.src == "lp_decode"]
    if warm_rows == 0:
        assert eng.c["lpd_no_gain"] == 2 and list(b.src_ids) == [1002, 1003]
        assert list(b.src_kinds) == [L.KIND_F, L.KIND_I] and eng.c["lpd_kept_F"] == eng.c["lpd_kept_I"] == 1
        Xk = np.asarray(b.X, float)
        assert [float(prob.key(x)) for x in Xk] == [k2, k3]
    else:
        assert eng.c["lpd_no_gain"] == 0 and list(b.src_ids) == [1000, 1001, 1002, 1003]
    assert eng.c["lpd_ok_F"] == 3 and eng.c["lpd_ok_I"] == 1
    eng.finish()


def test_sources_count_as_offered_only_once_screened_or_solved(files):
    """A source is recorded as offered (never offered again) when the reply screened or solved it — ok, infeasible,
    time limit, other, range fail, cached — but not when the worker's budget left it NOT_SOLVED or the deadline
    AFTER_DEADLINE: those stay eligible for a later batch. The funnel counters per kind: cached rows, I rows the range
    screen looked at (not cached, not NOT_SOLVED) and distinct sources recorded (a source counts once, also when a
    later reply covers it again)."""
    path, prob = files["gen"]
    eng = _stub_engine(prob)
    codes = [L.OK, L.OK, L.NOT_SOLVED, L.NOT_SOLVED, L.INFEASIBLE, L.TIME_LIMIT, L.OTHER, L.RANGE_FAIL, L.CACHED,
             L.AFTER_DEADLINE]
    kinds = [L.KIND_F] * 3 + [L.KIND_I] * 7                  # rows 2 (F) and 3 (I) NOT_SOLVED, row 8 (I) cached
    rows = _distinct_rows(5, len(codes)).astype(float)
    eng.lpd_req_info = _req_info(eng, rows, kinds, np.full(len(codes), np.inf))
    eng.lpd_offered = set()
    eng._lpd_result(77, _reply(codes, [0, 1], np.zeros((2, len(prob.conts)))))
    assert eng.lpd_offered == {1000, 1001, 1004, 1005, 1006, 1007, 1008}
    assert eng.c["lpd_solved_F"] == 2 and eng.c["lpd_solved_I"] == 3 and eng.c["lpd_not_solved"] == 2
    assert (eng.c["lpd_cached_F"], eng.c["lpd_cached_I"]) == (0, 1)
    assert eng.c["lpd_screened_I"] == 5                      # infeasible, time limit, other, range fail, after deadline
    assert (eng.c["lpd_srcs_F"], eng.c["lpd_srcs_I"]) == (2, 5)
    # a second reply: 1000 again (cached), the two NOT_SOLVED sources now solved
    info = _req_info(eng, rows[[0, 2, 3]], [L.KIND_F, L.KIND_F, L.KIND_I], np.full(3, np.inf))
    info["ids"] = np.array([1000, 1002, 1003], np.int64)
    eng.lpd_req_info = info
    eng._lpd_result(77, _reply([L.CACHED, L.OK, L.INFEASIBLE], [1], np.zeros((1, len(prob.conts)))))
    assert (eng.c["lpd_srcs_F"], eng.c["lpd_srcs_I"]) == (3, 6) and len(eng.lpd_offered) == 9
    assert (eng.c["lpd_cached_F"], eng.c["lpd_cached_I"], eng.c["lpd_screened_I"]) == (1, 1, 6)
    eng.finish()


def test_gains_count_only_the_copies_that_entered(files):
    """D-017 M3's gain median is over the decoded copies that ENTERED the population: a polish copy identical to a
    population member joins the candidate set but is eliminated as a duplicate, so neither `lpd_entered_F` nor
    `gain_F_entered_n` counts it."""
    path, prob = files["gen"]
    eng = Engine(prob, cfg_for(), seed=1, warmup=False, decoder_factory=E._WarmupDecoder)
    eng.prepare()
    eng._lpd_tick = lambda cand: None                        # no automatic offers: the test feeds the reply
    eng.begin(time.monotonic(), 3600.0)
    for _ in range(3):
        eng.step()
    i = int(np.argmin(eng.h_key))
    x = np.asarray(eng.bk.asnumpy(eng.pop.X[i]), float)
    ints, cn = np.asarray(prob.ints), np.asarray(prob.conts)
    key = float(prob.key(x))
    eng.lpd_req_info = _req_info(eng, x[ints][None, :], [L.KIND_F], [key + 1.0])   # the copy = x, gain > 0
    eng._lpd_result(77, _reply([L.OK], [0], x[cn][None, :]), site="exchange")
    assert eng.c["lpd_kept_F"] == 1 and eng._lpd_pending_batches() == 1
    dup0 = eng.c["duplicates"]
    eng.step()
    assert eng.c["lpd_batches"] == 1 and eng._lpd_pending_batches() == 0 and eng.c["duplicates"] > dup0
    out = eng.finish()
    assert out["lpd_entered_F"] == 0 and out["lp_decode"]["gain_F_entered_n"] == 0
    assert out["lp_decode"]["gain_F_entered_median"] is None and out["lpd_dropped"] == 0


def test_not_solved_sources_are_offered_again(files):
    """Two batches selected from the same candidate set: the polish rows the worker's budget left NOT_SOLVED are
    offered again; the rows it solved are not (the review's case: 4 F rows, 2 solved)."""
    path, prob = files["gen"]
    eng = Engine(prob, cfg_for(), seed=1, warmup=False, decoder_factory=E._WarmupDecoder)
    eng.prepare()
    held = {}
    eng._lpd_tick = lambda cand: held.__setitem__("cand", cand)       # no automatic offers: the test offers
    eng.begin(time.monotonic(), 3600.0)
    for _ in range(40):
        eng.step()
        if eng.lpd.state == "idle" and eng._lpd_surv is not None and eng._lpd_surv[2].sum() >= 6:
            break
    cand = held["cand"]
    assert eng.lpd.state == "idle" and eng._lpd_surv[2].sum() >= 6     # six feasible candidates at least
    eng._lpd_offer(cand)
    first = eng.lpd_req_info
    f_ids = [int(i) for i, k in zip(first["ids"], first["kinds"]) if k == L.KIND_F]
    assert len(f_ids) == eng.cfg.repair.lp_decode_per_gen == 4
    codes = [L.OK, L.OK, L.NOT_SOLVED, L.NOT_SOLVED] + [L.NOT_SOLVED] * (len(first["ids"]) - 4)
    eng._lpd_result(first["req"], _reply(codes, [0, 1], np.zeros((2, len(prob.conts)))))
    eng.lpd.state, eng.lpd._req = "idle", None                       # the stub's request is answered by hand
    eng._lpd_offer(cand)
    second = eng.lpd_req_info
    f2 = [int(i) for i, k in zip(second["ids"], second["kinds"]) if k == L.KIND_F]
    assert second["req"] == first["req"] + 1 and f2[:2] == f_ids[2:4] and not set(f2) & set(f_ids[:2]), (f_ids, f2)
    i1 = [int(i) for i, k in zip(first["ids"], first["kinds"]) if k == L.KIND_I]
    i2 = [int(i) for i, k in zip(second["ids"], second["kinds"]) if k == L.KIND_I]
    assert i2 == i1                                                  # every I row was left NOT_SOLVED
    eng.lpd_req_info = None
    eng.finish()


class _LagStub(E._WarmupDecoder):
    """The warm-up stub, but a reply becomes readable only at the `delays[j]`-th poll() after request j (cycling). The
    engine does not call poll() while a decoded batch is pending, so a held reply stays "in the pipe". With
    `exchange.poll_s` tiny the engine polls three times per generation (exchange tick, pre-candidate, pick site), so
    after an offer at a pick site delay 1 is read at the next exchange tick, 2 at the next pre-candidate poll and 3 at
    the next pick site."""

    delays = (3, 2, 1)

    def __init__(self, eng):
        super().__init__(eng)
        self.n_req, self.wait = 0, 0

    def offer(self, req_id, rows, kinds, per_gen=None):
        ok = super().offer(req_id, rows, kinds, per_gen)
        if ok:
            self.wait = self.delays[self.n_req % len(self.delays)]
            self.n_req += 1
        return ok

    def poll(self):
        if self.state == "busy" and self._req is not None:
            self.wait -= 1
            if self.wait > 0:
                return []
        return super().poll()


class _LateFirstStub(_LagStub):
    """The review's case: the first reply arrives during the survival (read at the next pick site), every later one
    within the generation of its offer."""

    delays = (3,) + (1,) * 199


class _PickStub(_LagStub):
    """Every reply is read at the pick site after the one of its offer."""

    delays = (3,)


def _lag_run(prob, factory, monkeypatch, steps, old_pick_site=False, hold=True):
    """`steps` generations on `prob` with a stub decoder, 2 rows per batch with every copy kept (`lpd_warm_rows`, the
    no-gain filter off) and an exchange tick every generation; the `lp_decode` events are collected, and per candidate
    set the decoded batches and rows that joined it. old_pick_site: the pick site as before the skip — a reply read
    there that kept copies is followed by an offer in the same tick (the batch is hidden from the real `_lpd_tick`,
    which then reads nothing new and offers). hold=False: every poll reads, also while a decoded batch is pending."""
    eng = Engine(prob, cfg_for(**{"exchange.poll_s": 1e-6}), seed=1, warmup=False, decoder_factory=factory,
                 lpd_warm_rows=2)
    eng.prepare()
    events = []
    eng.trace.event = lambda kind, t=None, **d: events.append((kind, d)) if kind == "lp_decode" else None
    if not hold:
        real_poll = eng._lpd_poll
        monkeypatch.setattr(eng, "_lpd_poll", lambda timed=True, site="exchange", hold=True:
                            real_poll(timed=timed, site=site, hold=False))
    if old_pick_site:
        real_tick = eng._lpd_tick

        def tick(cand):
            eng._lpd_poll(timed=False, site="pick")
            held = [b for b in eng.pending if b.src == "lp_decode"]
            eng.pending = [b for b in eng.pending if b.src != "lp_decode"]
            try:
                real_tick(cand)
            finally:
                eng.pending = held + eng.pending
        monkeypatch.setattr(eng, "_lpd_tick", tick)
    per_set = []
    real_with = eng._with_pending

    def with_pending(cand, meta):
        d = [b for b in eng.pending if b.src == "lp_decode"]
        per_set.append((eng.gen, len(d), sum(int(b.X.shape[0]) for b in d)))
        return real_with(cand, meta)
    monkeypatch.setattr(eng, "_with_pending", with_pending)
    eng.begin(time.monotonic(), 3600.0)
    for _ in range(steps):
        eng.step()
    return eng, events, per_set


@pytest.mark.parametrize("guards", [True, False], ids=["one_batch", "negative_control"])
def test_at_most_one_decoded_batch_joins_a_candidate_set(files, monkeypatch, guards):
    """Replies read at all three sites (delays 3, 2, 1): one decoded batch (at most N rows: what plan_memory counts)
    joins each candidate set; copies enter the candidate set of the generation that reads the reply (one generation
    after the offer), or the next one when the pick site reads it (two), which then offers nothing. The negative
    control removes both guards — the pick site's skip and the polls' hold — and does see two batches in one set (the
    test is not vacuous; the skip alone already keeps it to one, the hold is the second guard)."""
    path, prob = files["gen"]
    eng, events, per_set = _lag_run(prob, _LagStub, monkeypatch, 30, old_pick_site=not guards, hold=guards)
    out = eng.finish()
    most = max(n for _, n, _ in per_set)
    res = [d for k, d in events if d.get("what") == "result"]
    assert len(res) >= 10
    if not guards:
        assert most == 2, per_set
        return
    assert most == 1 and max(r for _, _, r in per_set) <= eng.cfg.repair.lp_decode_per_gen, per_set
    entering = {g: r for g, n, r in per_set if n}
    entered = [d for d in res if d["gen_enters"] is not None and d["gen_enters"] <= eng.gen]
    for d in entered:          # this generation's candidate set if read before it, the next one's if at the pick site
        assert d["gen_enters"] == d["gen"] + (1 if d["site"] == "pick" else 0), d
        assert d["gen_enters"] - d["gen_offered"] == (2 if d["site"] == "pick" else 1), d
        assert entering[d["gen_enters"]] == d["kept"] == 2
    assert {d["site"] for d in entered} == {"pick", "candidates", "exchange"}
    assert out["lpd_skip_pending"] == sum(d["site"] == "pick" for d in res) >= 3 and out["lpd_busy"] == 0
    assert out["lpd_kept"] == sum(r for _, _, r in per_set) + out["lpd_dropped"]


@pytest.mark.parametrize("skip", [True, False], ids=["skip", "negative_control"])
def test_a_pick_site_read_skips_the_offer_so_copies_return_to_lag_one(files, monkeypatch, skip):
    """The review's trap (delays 3, 1, 1, ...): the first reply is read at a pick site. That site offers nothing
    (`lpd_skip_pending`), so every later reply is read at the next exchange tick and its copies join the candidate
    set one generation after their offer. Negative control — the pick site offering in the same tick, as before the
    fix: the polls hold every later reply while the previous batch is pending, so all of them are read at a pick
    site, two generations after their offer."""
    path, prob = files["gen"]
    eng, events, per_set = _lag_run(prob, _LateFirstStub, monkeypatch, 25, old_pick_site=not skip)
    out = eng.finish()
    res = [d for k, d in events if d.get("what") == "result"]
    first, later = res[0], [d for d in res[1:] if d["site"] != "finish"]
    assert len(later) >= 8 and first["req"] == 1 and first["site"] == "pick"
    assert first["gen_enters"] - first["gen_offered"] == 2
    assert max(n for _, n, _ in per_set) == 1 and out["lpd_busy"] == 0
    if not skip:
        assert all(d["site"] == "pick" and d["gen_enters"] - d["gen_offered"] == 2 for d in later), later
        assert out["lpd_skip_pending"] == 0
        return
    assert all(d["site"] in ("exchange", "candidates") and d["gen_enters"] - d["gen_offered"] == 1 for d in later), \
        [(d["req"], d["site"], d["gen_offered"], d["gen_enters"]) for d in later]
    assert out["lpd_skip_pending"] == 1
    assert {(d["gen"], d["gen_offered"]) for d in later} == {(d["gen_offered"] + 1, d["gen_offered"]) for d in later}


@pytest.mark.parametrize("stub, site", [(E._WarmupDecoder, "exchange"), (_PickStub, "pick")])
def test_a_batch_left_pending_by_an_aborted_generation_is_recorded_dropped(files, monkeypatch, stub, site):
    """A decoded batch read at the exchange tick of a generation that is then cut short during the repair (or read
    at the pick site before it) never joins a candidate set, although its result event's `gen_enters` names that
    generation: finish counts its rows in `lpd_dropped` and names it by an `lp_decode` event "dropped" with its req,
    so a reader can exclude it (lpd_kept = rows that joined a candidate set + lpd_dropped)."""
    path, prob = files["gen"]
    eng, events, per_set = _lag_run(prob, stub, monkeypatch, 0)
    real_repair, arm = eng._repair, {"on": False}

    def repair(*a, **kw):
        moved = real_repair(*a, **kw)
        if arm["on"]:
            eng.request_stop("stop")
        return moved
    monkeypatch.setattr(eng, "_repair", repair)
    for _ in range(6):
        eng.step()
    if site == "pick":                                       # step until a pick-site read leaves a batch pending
        for _ in range(6):
            if eng._lpd_pending_batches():
                break
            eng.step()
        assert eng._lpd_pending_batches() == 1
    arm["on"] = True
    eng.step()                                               # its repair ends in a stop: the generation is dropped
    out = eng.finish()
    res = [d for k, d in events if d.get("what") == "result"]
    dropped = [d for k, d in events if d.get("what") == "dropped"]
    last = res[-1]
    assert out["aborted_generations"] == 1 and out["exit_reason"] == "stop"
    assert last["site"] == site and last["gen_enters"] == out["generations"] and last["kept"] == 2, last
    assert [(d["req"], d["site"], d["gen_enters"], d["rows"]) for d in dropped] == \
        [(last["req"], site, out["generations"], 2)]
    assert out["lpd_dropped"] == 2 and out["generations"] not in {g for g, n, _ in per_set if n}
    assert out["lpd_kept"] == sum(r for _, _, r in per_set) + out["lpd_dropped"]


def test_device_sync_is_timed_apart_from_host_seconds(files, monkeypatch):
    """D-017 M1: the device wait before the row copies is `lpd_sync_s`, not `lpd_host_s`. With a 50 ms
    `bk.synchronize` (on NumPy it does nothing): it is called once per offer, `lpd_sync_s` is about 50 ms per offer,
    and no host call reaches 50 ms."""
    SLEEP = 0.05
    path, prob = files["gen"]
    eng, events, per_set = _lag_run(prob, E._WarmupDecoder, monkeypatch, 0)
    calls = []

    def slow_sync():
        calls.append(eng.gen)
        time.sleep(SLEEP)
    monkeypatch.setattr(eng.bk, "synchronize", slow_sync)
    for _ in range(10):
        eng.step()
    out = eng.finish()
    n = out["lpd_offers"]
    assert n >= 8 and len(calls) == n, (n, calls)
    assert SLEEP * n <= eng.lpd_sync_s <= SLEEP * n + 0.5 and eng.lpd_sync_s_max >= SLEEP
    assert eng.lpd_host_s_max < SLEEP and out["lp_decode"]["host_s_max"] < SLEEP


# ======================================================================================================= runner
def test_runner_refuses_lp_decode_for_the_single_core_arms(tmp_path):
    from itrea.experiments import pair
    inst = toy.write_mps(toy.mixed(), tmp_path / "mixed.mps")
    o = pair.PairOptions(traces_dir=tmp_path / "traces", mailbox_root=tmp_path / "mb", gpu_health=False,
                         record_snapshots=False, cupy_cache_dir=tmp_path / "cc", log=lambda m: None)

    def job(arm, **ov):
        return dict(instance=str(inst), set="pytest", arm=arm, seed=1000, budget=300, split="smoke",
                    config="configs/itrea2-dev-rc-3way.yaml", overrides=ov)
    for arm in ("A9", "A3x"):
        with pytest.raises(pair.RunError, match="lp_decode"):
            pair.job_budget(job(arm, **LPD), o)
        pair.job_budget(job(arm), o)                         # the control is accepted
    pair.job_budget(job("A3", **LPD), o)                      # A3: the EA has the three A725 cores
    assert pair.run_hash(job("A3", **LPD), o) == "f030e23909ad"


# =================================================================================================== build_runs
def test_build_runs_lp_decode_columns(tmp_path):
    from itrea.analysis import build_runs as B
    from test_build_runs import X_70, X_85, make_run, toy_problem
    p, mps = toy_problem(tmp_path)
    raw = tmp_path / "raw"
    blk = dict(state="idle", host_s=0.5, host_s_max=0.01, sync_s=0.02, sync_s_max=0.004, ready_t=1.2,
               ready_t_worker=0.9, wait_s_total=0.3, wait_s_max=0.1, batch_rows_max=4, gain_F_entered_median=0.02,
               gain_F_entered_n=3, gain_I_entered_median=-0.25, gain_I_entered_n=2,
               ipc=dict(offers=9, offer_s_total=0.05, offer_s_max=0.012, offer_bytes_max=2342400, reads=8,
                        read_s_total=0.03, read_s_max=0.006, read_bytes_max=5098000),
               worker=dict(priming_s=0.3, priming_first_s=0.25, priming_first_status="Optimal", priming_solves=1,
                           priming_time_limit_hits=0, solves=12, warm_solves=11, time_limit_hits=1,
                           warm_time_limit_hits=1, time_limit_hits_starved=1, screen_pass=7, screen_fail=2,
                           solve_s_total=1.5, solve_cpu_s_total=1.25, ru_maxrss_mb=80.0))
    cnt = {k: i + 1 for i, k in enumerate(B.LPD_INT_COLS)}
    make_run(raw, p, mps, "A3", 1000, 10, {"solver": [(2.0, X_85, "self")], "ea": [(3.0, X_70, "self")]},
             summary_extra={"ea": dict(lp_decode=blk, **cnt)}, cfg="lpd000000000")
    make_run(raw, p, mps, "A3", 1001, 10, {"solver": [(2.0, X_85, "self")], "ea": [(3.0, X_70, "self")]},
             summary_extra={"ea": {"startup": {"lp_decode": {"state": "inert", "reason": "x"}}}})
    make_run(raw, p, mps, "A3", 1002, 10, {"solver": [(2.0, X_85, "self")], "ea": [(3.0, X_70, "self")]})
    out = tmp_path / "datasets"
    out.mkdir()
    df, _ = B.build(raw, out, log=lambda *a: None)
    r = df.set_index("seed")
    assert r.loc[1000, "lpd_state"] == "idle" and r.loc[1000, "lpd_host_s"] == 0.5
    assert r.loc[1000, "lpd_priming_s"] == 0.3 and r.loc[1000, "lpd_worker_solves"] == 12
    for k, v in cnt.items():
        assert r.loc[1000, k] == v, k
    for col, key in B.LPD_BLOCK_COLS:
        assert r.loc[1000, col] == blk[key], col
    for col, key in B.LPD_IPC_COLS:
        assert r.loc[1000, col] == blk["ipc"][key], col
    for col, key in B.LPD_WORKER_COLS:
        assert r.loc[1000, col] == blk["worker"][key], col
    assert r.loc[1000, "lpd_priming_first_status"] == "Optimal" and r.loc[1000, "lpd_ready_t_worker"] == 0.9
    assert r.loc[1001, "lpd_state"] == "inert" and r.loc[1001, "lpd_offers"] == -1
    assert r.loc[1002, "lpd_state"] == "" and r.loc[1002, "lpd_offers"] == -1 and math.isnan(r.loc[1002, "lpd_host_s"])
    assert r.loc[1002, "lpd_priming_first_status"] == "" and r.loc[1002, "lpd_warm_solves"] == -1
    assert r.loc[1002, "lpd_gain_F_entered_n"] == -1 and math.isnan(r.loc[1002, "lpd_gain_F_entered_median"])
    assert str(df["lpd_gain_F_entered_n"].dtype).startswith("int") and df["lpd_skip_pending"].notna().all()
    assert set(B.RUN_COLUMNS) >= {c for c, _ in B.LPD_BLOCK_COLS + B.LPD_IPC_COLS + B.LPD_WORKER_COLS}


# ======================================================================================================= CuPy path
@pytest.mark.gpu
def test_decoded_rows_on_cupy_equal_numpy(files):
    """The copy-building step (`_lpd_result`) on both backends from the same reply: the same batch rows."""
    from test_engine import gpu_or_skip
    gpu_or_skip()
    path, prob = files["gen"]
    outs = {}
    for backend in ("numpy", "cupy"):
        eng = Engine(prob, cfg_for(backend), seed=1, warmup=False, decoder_factory=E._WarmupDecoder)
        eng.prepare()
        eng.begin(time.monotonic(), 3600.0)
        k = 3
        rs = np.random.default_rng(0)
        rows = rs.integers(0, 2, size=(k, len(prob.ints))).astype(float)
        eng.lpd_req_info = dict(req=77, gen=0, t=0.0, ids=np.arange(k) + 1000, kinds=np.array([1, 1, 1], np.uint8),
                                rows=rows, key=np.full(k, np.inf), V=np.ones(k), feasible=np.zeros(k, bool),
                                strat=np.zeros(k, np.uint8), anc_rank=np.zeros((k, eng.depth), np.uint8),
                                anc_ops=np.zeros((k, eng.depth), np.uint16), rank=np.full(k, -1, np.int8))
        conts = rs.uniform(-1, 11, size=(k, len(prob.conts)))
        rep = dict(codes=np.zeros(k, np.uint8), ok_rows=np.arange(k), conts=conts, obj=np.zeros(k),
                   iters=np.zeros(k, np.int64), secs=np.zeros(k), limit=np.zeros(k), status=["Optimal"] * k, stats={})
        eng.pending = []
        eng._lpd_result(77, rep)
        (b,) = [b for b in eng.pending if b.src == "lp_decode"]
        outs[backend] = eng.bk.asnumpy(b.X)
        eng.finish()
    assert outs["numpy"].dtype == outs["cupy"].dtype and np.array_equal(outs["numpy"], outs["cupy"])


@pytest.mark.gpu
def test_cupy_cold_cache_compiles_the_decode_path_before_the_clock(tmp_path):
    """D-017 (lesson of D-016's void block): with an EMPTY CuPy cache, an lp_decode run on CuPy compiles nothing after
    t0 — and the test is not vacuous: decoded copies entered the population after t0 (`lpd_entered` >= 1)."""
    import subprocess
    from test_ea_run import MODULE, PY, SMALL as RSMALL, _env, _gpu_ok, _summary
    from itrea.trace import read_events
    if not _gpu_ok():
        pytest.skip("no CuPy device")
    insts = {"gen": toy.write_mps(gen_problem(), tmp_path / "gen.mps"),
             "mixed": toy.write_mps(toy.mixed(), tmp_path / "mixed.mps")}
    for name, inst in insts.items():
        rd = tmp_path / f"run_{name}"
        env = {k: v for k, v in _env(CUPY_CACHE_DIR=str(tmp_path / f"cache_{name}")).items()
               if k != "CUPY_ACCELERATORS"}
        pr = subprocess.run([PY, "-m", MODULE, "--instance", str(inst), "--seed", "0", "--budget", "8", "--run-dir",
                             str(rd), "--no-barrier", "--backend", "cupy", "--set", "repair.continuous=lp_decode",
                             "--set", "log_every=1", *RSMALL], env=env, capture_output=True, timeout=600)
        assert pr.returncode == 0, (pr.stdout + pr.stderr).decode(errors="replace")
        s = _summary(rd)
        assert s["gene_dtype"] == "float64" and s["startup"]["lp_decode"]["state"] == "configured"
        wu = s["startup"]["warmup"]                          # the largest decoded batch per pass (rows)
        assert wu["many"]["lp_decode_warmed"] >= 2 and wu["one"]["lp_decode_warmed"] == 1, wu
        ev = [e for e in read_events(rd / "ea.events.jsonl") if e["kind"] == "lp_decode"]
        assert any(e["what"] == "result" and e["kept"] >= 1 for e in ev), (name, "never reached")
        assert s["lpd_entered"] >= 1 and s["lp_decode"]["state"] != "failed", (name, s["lp_decode"])
        assert s["compiled_on_clock"] is False and s["cupy_cache"]["new_after_t0"] == [], (name, s["cupy_cache"])
