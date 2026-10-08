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
    return L.LPDecoder(path, L.model_digest(prob.names, prob.isint), "int32", per_gen, time_limit, cold_limit, 1e-6,
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
    assert L.int_part_dtype(np.array([0.0, -3.0]), np.array([5.0, 7.0])) == np.int32
    assert L.int_part_dtype(np.array([0.0]), np.array([np.inf])) == np.float64
    assert L.int_part_dtype(np.array([0.5]), np.array([3.0])) == np.float64
    assert L.int_part_dtype(np.array([0.0]), np.array([2.0 ** 40])) == np.float64


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
    assert info["omp_num_threads"] == "1"
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


@pytest.mark.solver
def test_worker_rebases_the_cumulative_time_limit(files):
    """HiGHS's time_limit is cumulative over runs: 100 warm solves at L = 0.02 s each, whose total HiGHS run time
    exceeds L, all end Optimal because every solve sets time_limit = getRunTime() + L."""
    path, prob = files["big"]
    rs = np.random.default_rng(7)
    rows = rs.integers(0, 2, size=(100, 40)).astype(np.int32)
    rows[:, 0] = 1
    with _decoder(path, prob, time_limit=0.02, cold_limit=0.02) as dec:
        _wait(dec)
        reps = [_solve(dec, rows[s:s + 25], per_gen=25) for s in range(0, 100, 25)]
    status = [s for r in reps for s in r["status"]]
    st = reps[-1]["stats"]
    assert st["run_time"] > 0.02, st                         # otherwise the test proves nothing
    assert status == ["Optimal"] * 100 and st["time_limit_hits"] == 0 and st["solves"] == 100


@pytest.mark.solver
def test_worker_priming_limit_and_basis_restore(files):
    """A cold solve stopped by its time limit leaves no anchor, so the next solve is again cold (cold limit). After an
    infeasible solve the last optimal basis is restored: the next row's simplex iterations are those it takes from that
    basis, not from the infeasible solve's."""
    path, prob = files["gen"]
    rs = np.random.default_rng(11)
    A, B, B2 = (rs.integers(0, 2, size=30).astype(np.int32) for _ in range(3))
    A[0] = B[0] = B2[0] = 1
    C = A.copy()
    C[:2] = 0                                                # x0 = x1 = 0: LP infeasible
    with _decoder(path, prob, time_limit=0.05, cold_limit=1.0, cache_cap=0, hooks={"cold_limits": [0.0]}) as dec:
        _wait(dec)
        r1 = _solve(dec, [A])
        assert r1["status"][0] == "Time limit reached" and r1["limit"][0] == 0.0, r1
        r2 = _solve(dec, [A])
        assert r2["limit"][0] == 1.0 and r2["status"][0] == "Optimal"          # still priming: the cold limit
        r3 = _solve(dec, [B2])
        assert r3["limit"][0] == 0.05 and r3["status"][0] == "Optimal"
        ok = _solve(dec, [A, B])
        bad = _solve(dec, [A, C, B])
        assert [L.CODE_NAMES[c] for c in bad["codes"]] == ["ok", "infeasible", "ok"]
        assert bad["iters"][2] <= ok["iters"][1] + 2, (ok["iters"], bad["iters"])
        assert bad["stats"]["restores"] >= 1 and bad["stats"]["restore_errors"] == 0
        assert bad["stats"]["priming_solves"] == 2 and bad["stats"]["time_limit_hits"] >= 1


@pytest.mark.solver
def test_worker_digest_mismatch_fails(files):
    path, prob = files["gen"]
    dec = L.LPDecoder(path, "0" * 32, "int32", 4, 0.05, 1.0, 1e-6).start()
    msgs = _wait(dec, states=("failed",))
    assert dec.state == "failed" and "digest" in dec.error and any(m[0] == "failed" for m in msgs)
    assert not dec.offer(1, np.zeros((1, 30), np.int32), [0])
    assert _gone(dec.pid) and dec.stop() is False and dec.state == "failed"


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


@pytest.mark.solver
def test_engine_decoded_copies(files, tmp_path, monkeypatch):
    """The real worker on the generated toy: copies carry ORIGIN_LP_DECODE, their source's strategy, no parents and
    `lpd_src`; their integer parts are their sources' (rounded), their continuous values LP-optimal; polish copies beat
    their sources by imp_tol; every decoded incumbent passes the float64 check; the counters add up; the genealogy
    goes through the lineage analysis; the worker is gone after finish."""
    from scipy.optimize import linprog
    from itrea.analysis.lineage import run_tables
    from itrea.trace import Trace, read_events
    path, prob = files["gen"]
    cfg = cfg_for(**{"strategy.enabled": True, "log_every": 1})
    tr = Trace(tmp_path, "ea")
    eng = Engine(prob, cfg, seed=5, trace=tr, warmup=False)
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

    def result(req_id, rep):
        n0 = len(eng.pending)
        real_result(req_id, rep)
        batches.extend(b for b in eng.pending[n0:] if b.src == "lp_decode")
    monkeypatch.setattr(eng, "_lpd_offer", offer)
    monkeypatch.setattr(eng, "_lpd_result", result)
    before = _children()
    eng.begin(time.monotonic(), 3600.0)
    pid = eng.lpd.pid
    _run_until(eng, lambda e: e.c["lpd_entered"] >= 3 and e.c["lpd_incumbents"] >= 1 and e.c["lpd_kept_F"] >= 2)
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
    blk = out["lp_decode"]
    assert blk["state"] in ("idle", "busy") and blk["worker"]["solves"] >= c["lpd_ok"] and blk["host_s"] > 0
    assert blk["ready"]["nice"] >= L.NICE and blk["host_s_max"] <= blk["host_s"]
    gens = [e for e in read_events(tmp_path / "ea.events.jsonl") if e["kind"] == "gen"]
    assert all("lpd_offers" in e and "lpd_host_s" in e for e in gens)
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
def test_engine_run_reads_the_last_reply_and_stops_the_worker(files):
    """`run()` to its budget: finish() reads a reply waiting at the deadline before it stops the worker, so the
    summary's statistics are those of the last reply read; the worker is gone afterwards."""
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

    def spy(self, ids, src_ids):
        seen.append((str(self.prob.path), self.lpd_warm_rows, len(ids)))
        return real(self, ids, src_ids)
    monkeypatch.setattr(Engine, "_lpd_entering", spy)
    before = _children()
    eng = Engine(prob, cfg_for(), seed=4)
    st = eng.prepare()
    assert _children() == before
    w = st["warmup"]
    assert w["many"]["lp_decode_warmed"] >= 1 and w["one"]["lp_decode_warmed"] >= 1, w
    assert {p for p, _, _ in seen} == {"<synthetic warm-up>"}
    assert {k for _, r, k in seen if r == 3} == {3} and {k for _, r, k in seen if r == 1} == {1}
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
    blk = dict(state="idle", host_s=0.5, host_s_max=0.01, ready_t=1.2,
               worker=dict(priming_s=0.3, solves=12, time_limit_hits=1, ru_maxrss_mb=80.0))
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
    assert r.loc[1001, "lpd_state"] == "inert" and r.loc[1001, "lpd_offers"] == -1
    assert r.loc[1002, "lpd_state"] == "" and r.loc[1002, "lpd_offers"] == -1 and math.isnan(r.loc[1002, "lpd_host_s"])


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
        assert all(s["startup"]["warmup"][p]["lp_decode_warmed"] >= 1 for p in ("many", "one"))
        ev = [e for e in read_events(rd / "ea.events.jsonl") if e["kind"] == "lp_decode"]
        assert any(e["what"] == "result" and e["kept"] >= 1 for e in ev), (name, "never reached")
        assert s["lpd_entered"] >= 1 and s["lp_decode"]["state"] != "failed", (name, s["lp_decode"])
        assert s["compiled_on_clock"] is False and s["cupy_cache"]["new_after_t0"] == [], (name, s["cupy_cache"])
