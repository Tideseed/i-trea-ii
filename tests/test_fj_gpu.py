"""Tests of the GPU feasibility-jump local search (`itrea.ea.ls_fj_gpu.FJGpu`).

The spec (vault `10-design/Feasibility jump local search.md`) lists the invariants: integer genes stay integral and in
bounds; `act` equals `A x` after the call; a feasible input with the cutoff inactive is returned unchanged; most
children reach feasibility on toy knapsack / set cover / mixed models; the GPU and a CPU implementation of the same
move rule reach feasibility at similar rates. The CPU side here is `ref_fj`, a deliberately plain NumPy version of the
rule for one individual (exact ∝ w·v row choice over ALL violated rows, where the kernel samples ≤ 32) — so behaviour
statistics are compared, not trajectories. If `itrea.ea.ls_fj_cpu` exists when the tests run, it is cross-checked too.

The throughput test (synthetic set cover n=50k, m=20k, P=64, generated in memory) prints moves/s per block size; it
asserts only that the search ran. The block-size default in `ls_fj_gpu.DEFAULT_THREADS` was chosen from it.
"""
from __future__ import annotations

import inspect
import math
import os

import numpy as np
import pytest
import scipy.sparse as sps

cp = pytest.importorskip("cupy")

from itrea.benchmarks import toy  # noqa: E402
from itrea.config import RepairCfg  # noqa: E402
from itrea.ea.evaluate import to_device  # noqa: E402
from itrea.ea.ls_fj_gpu import (FJGpu, ST_BUDGET, ST_FEASIBLE, ST_STALL, VT_BIN, VT_CONT,  # noqa: E402
                                VT_FIXED, VT_INT)
from itrea.problem import from_arrays  # noqa: E402

pytestmark = pytest.mark.gpu
os.environ.setdefault("OMP_NUM_THREADS", "1")


@pytest.fixture(scope="module")
def bk():
    from itrea.backend import get_backend
    try:
        return get_backend("cupy", gpu_mem_gb=1.0)
    except Exception as e:  # no device / driver
        pytest.skip(f"CuPy backend unavailable: {e}")


# ----------------------------------------------------------------------------------------------- helpers
def make_fj(prob, bk, mode="cutoff_row", max_steps=2000, stall_steps=100, budget=64 << 20, seed=0, **kw):
    dp = to_device(prob, bk)
    cfg = RepairCfg(objective_mode=mode, max_steps=max_steps, stall_steps=stall_steps)
    return FJGpu(dp, cfg, imp_tol=1e-9, mem_budget_bytes=budget, seed=seed, **kw)


def random_starts(prob, k, seed, p_one=None):
    """Random points in the bounds (window ±10 on infinite sides); integers integral. p_one: binaries ~ Bernoulli."""
    rs = np.random.default_rng(seed)
    lo = np.where(np.isfinite(prob.lb), prob.lb, -10.0)
    hi = np.where(np.isfinite(prob.ub), prob.ub, lo + 20.0)
    lo = np.where(np.isfinite(prob.lb), lo, hi - 20.0)
    X = lo + rs.random((k, prob.n)) * (hi - lo)
    X[:, prob.ints] = np.round(X[:, prob.ints])
    if p_one is not None:
        X[:, prob.bins] = (rs.random((k, len(prob.bins))) < p_one).astype(float)
    return np.clip(X, prob.lb, prob.ub)


def aug_matrix(fj, prob):
    A = prob.A
    if fj.has_cut:
        A = sps.vstack([A, sps.csr_matrix(prob.cmin.reshape(1, -1))], format="csr")
    return A


def host_isviol(a, lo, hi, tol=1e-9, cap=1e-7):
    with np.errstate(invalid="ignore"):
        el = np.minimum(tol * np.maximum(1.0, np.abs(lo)), cap)
        eu = np.minimum(tol * np.maximum(1.0, np.abs(hi)), cap)
        return ((a < lo) & (lo - a > el)) | ((a > hi) & (a - hi > eu))


def check_invariants(fj, prob, X):
    """Integrality, bounds, act == A_aug x, violated list == membership by the rule, weights >= 1."""
    Xh = cp.asnumpy(X).astype(np.float64)
    xi = Xh[:, prob.ints]
    assert np.all(xi == np.round(xi)), "integer gene not integral"
    lbe = np.where(prob.isint, np.ceil(prob.lb - 1e-9), prob.lb)
    ube = np.where(prob.isint, np.floor(prob.ub + 1e-9), prob.ub)
    assert np.all(Xh >= lbe - 1e-12) and np.all(Xh <= ube + 1e-12), "gene outside its bounds"
    st = fj.state
    A = aug_matrix(fj, prob)
    act_true = (A @ Xh.T).T
    rel = np.abs(st["act"] - act_true) / np.maximum(1.0, np.abs(act_true))
    assert rel.max(initial=0.0) <= 1e-9, f"act drifted from A x: {rel.max()}"
    rl = np.append(prob.rl, -np.inf) if fj.has_cut else prob.rl
    ru = np.append(prob.ru, cp.asnumpy(fj.d_ru)[-1]) if fj.has_cut else prob.ru
    for b in range(Xh.shape[0]):
        nv = int(st["nviol"][b])
        member = host_isviol(st["act"][b], rl, ru)
        lst = st["vlist"][b][:nv]
        assert len(set(lst.tolist())) == nv
        assert set(lst.tolist()) == set(np.flatnonzero(member).tolist())
        pos = st["vpos"][b]
        assert np.all(pos[lst] == np.arange(nv))
        assert np.all(pos[~member] == -1)
        assert np.all(st["w"][b] >= 1.0)
    return Xh


# ----------------------------------------------------------------------- NumPy reference (one individual)
def ref_fj(prob, x0, max_steps=2000, stall_steps=100, bump=1.0, cut_rhs=None, max_cand=16, seed=0,
           tol=1e-9, cap=1e-7, obj_weight=None):
    """Plain NumPy feasibility jump with the kernel's move rule. Returns (x, status, moves).
    obj_weight: `weighted` mode (score += obj_weight * cmin_j * d / max|cmin|); None = no objective term."""
    rng = np.random.default_rng(seed)
    A = sps.csr_matrix(prob.A)
    rl, ru, sc = prob.rl.astype(float).copy(), prob.ru.astype(float).copy(), prob.row_norm.astype(float).copy()
    if cut_rhs is not None:
        A = sps.vstack([A, sps.csr_matrix(prob.cmin.reshape(1, -1))], format="csr")
        rl, ru = np.append(rl, -np.inf), np.append(ru, cut_rhs)
        sc = np.append(sc, np.linalg.norm(prob.cmin) or 1.0)
    C = A.tocsc()
    C.sort_indices()
    M = A.shape[0]
    isint = prob.isint
    lb = np.where(isint, np.ceil(prob.lb - 1e-9), prob.lb)
    ub = np.where(isint, np.floor(prob.ub + 1e-9), prob.ub)
    vt = np.where(isint, VT_INT, VT_CONT)
    vt[isint & (lb == 0) & (ub == 1)] = VT_BIN
    vt[lb >= ub] = VT_FIXED
    x = np.where(isint, np.rint(x0), x0).astype(float)
    x = np.clip(x, lb, ub)
    act = A @ x
    w = np.ones(M)
    rows_all = np.arange(M)
    cobj = np.zeros(prob.n)
    if obj_weight is not None and np.abs(prob.cmin).max() > 0:
        cobj = obj_weight * prob.cmin / np.abs(prob.cmin).max()

    def vv(a, idx):
        lo, hi = rl[idx], ru[idx]
        with np.errstate(invalid="ignore"):
            return (np.where(a < lo, lo - a, 0.0) + np.where(a > hi, a - hi, 0.0)) / sc[idx]

    F = float(vv(act, rows_all).sum())
    Fbest, since, moves = F, 0, 0
    while True:
        V = np.flatnonzero(host_isviol(act, rl, ru, tol, cap))
        if V.size == 0:
            return x, ST_FEASIBLE, moves
        if moves >= max_steps:
            return x, ST_BUDGET, moves
        if since >= stall_steps:
            return x, ST_STALL, moves
        wv = w[V] * vv(act[V], V)
        r = int(rng.choice(V, p=wv / wv.sum()))
        best = (math.inf, 0.0, -1, 0.0)
        for p in range(A.indptr[r], A.indptr[r + 1]):
            j = int(A.indices[p])
            if vt[j] == VT_FIXED:
                continue
            rows, vals = C.indices[C.indptr[j]:C.indptr[j + 1]], C.data[C.indptr[j]:C.indptr[j + 1]]
            a = act[rows]
            xj = x[j]
            if vt[j] == VT_BIN:
                cands = [1.0 - xj]
            else:
                K = max(1, (max_cand - 4) // 2 if vt[j] == VT_INT else max_cand - 2)
                vm = host_isviol(a, rl[rows], ru[rows], tol, cap)
                rr, av, vl = rows[vm], a[vm], vals[vm]
                side = np.where(av < rl[rr], rl[rr], ru[rr])
                tt = xj + (side - av) / vl
                key = np.where(rr == r, np.inf, w[rr] * vv(av, rr))
                tk = tt[np.argsort(-key, kind="stable")[:K]]
                bounds = [b for b in (lb[j], ub[j]) if np.isfinite(b)]
                if vt[j] == VT_INT:
                    cands = [xj - 1.0, xj + 1.0] + bounds
                    for t in tk:
                        cands += [math.floor(t + 1e-9), math.ceil(t - 1e-9)]
                else:
                    cands = bounds + list(tk)
            v0 = vv(a, rows)
            for t in np.clip(np.asarray(cands, float), lb[j], ub[j]):
                d = t - xj
                if d == 0.0:
                    continue
                s = float((w[rows] * (vv(a + vals * d, rows) - v0)).sum())
                if s + cobj[j] * d < best[0]:
                    best = (s + cobj[j] * d, s, j, t)
        if best[0] < -1e-12:
            _, dv, j, t = best
            d = t - x[j]
            rows, vals = C.indices[C.indptr[j]:C.indptr[j + 1]], C.data[C.indptr[j]:C.indptr[j + 1]]
            act[rows] += vals * d
            x[j] = t
            moves += 1
            F += dv
        else:
            F += bump * float(vv(act[V], V).sum())
            w[V] += bump
        if F < Fbest - 1e-12 * max(1.0, abs(Fbest)):
            Fbest, since = F, 0
        else:
            since += 1


def _toys():
    return {
        "knapsack": (toy.knapsack(30, seed=0), 0.6),
        "set_cover": (toy.set_cover(40, 25, seed=0), 0.05),
        "set_cover_big": (toy.set_cover(200, 120, density=0.05, seed=1), 0.02),
        "mixed": (toy.mixed(seed=0), None),
    }


# ----------------------------------------------------------------------------------------------------- tests
@pytest.mark.parametrize("name", ["knapsack", "set_cover", "set_cover_big", "mixed"])
def test_invariants(bk, name):
    prob, p_one = _toys()[name]
    fj = make_fj(prob, bk, seed=3)
    if name == "knapsack":
        fj.set_cutoff(-0.8 * prob.toy_optimum)   # exercise the cutoff row's incremental activity too
    X = cp.asarray(random_starts(prob, 24, seed=11, p_one=p_one))
    c = fj.run(X, keep_state=True)
    Xh = check_invariants(fj, prob, X)
    assert c["finished_feasible"] + c["finished_stall"] + c["finished_budget"] == 24
    assert c["steps"] >= c["moves"] >= 0 and c["chunk"] >= 1 and c["peak_bytes"] > 0
    for b in np.flatnonzero(fj.last_status == ST_FEASIBLE):
        assert prob.check(Xh[b], feas_tol=1e-6).feasible


@pytest.mark.parametrize("name", ["knapsack", "set_cover", "set_cover_big", "mixed"])
def test_feasibility_rate_vs_reference(bk, name):
    prob, p_one = _toys()[name]
    k = 32
    X0 = random_starts(prob, k, seed=5, p_one=p_one)
    fj = make_fj(prob, bk, mode="none", seed=7)
    X = cp.asarray(X0)
    fj.run(X)
    Xh = cp.asnumpy(X)
    gpu_rate = np.mean([prob.check(x).feasible for x in Xh])
    assert np.all((fj.last_status == ST_FEASIBLE) == np.array([prob.check(x).feasible for x in Xh]))
    ref = [ref_fj(prob, X0[b], seed=100 + b) for b in range(k)]
    ref_rate = np.mean([prob.check(x).feasible for x, _, _ in ref])
    assert all((s == ST_FEASIBLE) == prob.check(x).feasible for x, s, _ in ref)
    assert gpu_rate >= 0.8, f"{name}: GPU feasibility rate {gpu_rate}"
    assert abs(gpu_rate - ref_rate) <= 0.2, f"{name}: GPU {gpu_rate} vs reference {ref_rate}"
    gm, rm = float(np.mean(fj.last_moves)), float(np.mean([mv for _, _, mv in ref]))
    assert gm <= 3 * rm + 10 and rm <= 3 * gm + 10, f"{name}: mean moves GPU {gm} vs reference {rm}"


def test_cross_check_cpu_module(bk):
    """If ls_fj_cpu exists (written in parallel), both reach feasibility at similar rates on the same starts."""
    try:
        import itrea.ea.ls_fj_cpu as cpu_mod
    except ImportError:
        pytest.skip("itrea.ea.ls_fj_cpu not present")
    cls = getattr(cpu_mod, "FJCpu", None)
    if cls is None:
        pytest.skip("ls_fj_cpu has no FJCpu class")
    from itrea.backend import get_backend
    nbk = get_backend("numpy")
    for name in ("knapsack", "set_cover_big", "mixed"):
        prob, p_one = _toys()[name]
        X0 = random_starts(prob, 32, seed=9, p_one=p_one)
        cfg = RepairCfg(objective_mode="none", max_steps=2000, stall_steps=100)
        kw = dict(cfg=cfg, imp_tol=1e-9, mem_budget_bytes=64 << 20, seed=1)
        params = inspect.signature(cls).parameters
        try:
            cpu = cls(to_device(prob, nbk), **{k: v for k, v in kw.items() if k in params})
            Xc = X0.copy()
            c_cpu = cpu.run(Xc)
        except Exception as e:  # the API was guessed; do not fail this file on another module's signature
            pytest.skip(f"ls_fj_cpu API differs from the guess: {e!r}")
        fj = make_fj(prob, bk, mode="none", seed=1)
        Xg = cp.asarray(X0)
        cg = fj.run(Xg)
        cpu_rate = np.mean([prob.check(x).feasible for x in np.asarray(Xc)])
        gpu_rate = np.mean([prob.check(x).feasible for x in cp.asnumpy(Xg)])
        assert abs(cpu_rate - gpu_rate) <= 0.2, f"{name}: CPU {cpu_rate} vs GPU {gpu_rate}"
        assert c_cpu["moves"] <= 2 * cg["moves"] + 50 and cg["moves"] <= 2 * c_cpu["moves"] + 50, (name, c_cpu, cg)


@pytest.mark.parametrize("cap,filt", [(0, False), (8, False), (8, True)])
def test_cutoff_row_gives_strictly_better_feasible_knapsack(bk, cap, filt):
    """row_scan_cap=8 sends both knapsack rows (30 entries) through the windowed path (with/without the filter)."""
    prob = toy.knapsack(30, seed=1)
    opt = prob.toy_optimum
    v = prob.c
    # feasible but poor starts: random subsets with ~20 % of the items, trimmed to capacity
    rs = np.random.default_rng(4)
    W = prob.ru[0]
    wts = np.asarray(prob.A.todense()).ravel()
    X0 = np.zeros((32, prob.n))
    for b in range(32):
        for j in rs.permutation(prob.n):
            if rs.random() < 0.2 and X0[b] @ wts + wts[j] <= W:
                X0[b, j] = 1.0
    # 0.85 * opt: the GPU reaches it from ~94 % of these starts and ref_fj from ~81 % (measured 2026-10-06); at
    # 0.9 * opt both fall to 25-35 % — single-flip moves on a two-row (capacity + cutoff) problem trap the search
    inc_obj = math.floor(0.85 * opt)
    assert np.all(X0 @ v < inc_obj)
    inc_key = prob.key_of_objective(inc_obj)

    # without a cutoff the feasible starts are returned unchanged
    kw = dict(mode="cutoff_row", seed=2, max_steps=5000, stall_steps=500, row_scan_cap=cap, row_scan_filter=filt)
    fj0 = make_fj(prob, bk, **kw)
    X = cp.asarray(X0)
    c0 = fj0.run(X)
    assert c0["moves"] == 0 and np.array_equal(cp.asnumpy(X), X0)

    fj = make_fj(prob, bk, **kw)
    fj.set_cutoff(inc_key)
    assert fj.obj_integral and fj.cut_rhs == -inc_obj - 1           # cmin·x <= -(inc_obj + 1): delta = 1
    X = cp.asarray(X0)
    c = fj.run(X, keep_state=True)
    check_invariants(fj, prob, X)
    Xh = cp.asnumpy(X)
    feas = fj.last_status == ST_FEASIBLE
    assert feas.mean() >= 0.6, f"only {feas.mean()} reached the cutoff: {c}"
    for b in np.flatnonzero(feas):
        assert prob.check(Xh[b]).feasible
        assert prob.key(Xh[b]) <= inc_key - 1.0 + 1e-9          # strictly better than the incumbent
    assert np.all(Xh[feas] @ v >= inc_obj + 1)


def test_cutoff_noninteger_objective_is_strict(bk):
    """mixed: continuous variables carry costs, so the objective is not integral (g = 0). A finished-feasible child must
    still be strictly better than the incumbent by imp_tol, despite the violated-set tolerance on the cutoff row."""
    prob = toy.mixed()
    inc_key = prob.key(np.array([3.0, 2.0, 0.0, 0.0]))          # a feasible point, key 7
    fj = make_fj(prob, bk, mode="cutoff_row", seed=3, max_steps=3000, stall_steps=300)
    fj.set_cutoff(inc_key)
    assert not fj.obj_integral and fj.cut_rhs < inc_key - 1e-9 * max(1.0, abs(inc_key))
    X = cp.asarray(random_starts(prob, 32, seed=4))
    fj.run(X, keep_state=True)
    Xh = check_invariants(fj, prob, X)
    feas = fj.last_status == ST_FEASIBLE
    assert feas.any()
    for b in np.flatnonzero(feas):
        assert prob.check(Xh[b]).feasible
        assert prob.key(Xh[b]) < inc_key - 1e-9 * max(1.0, abs(inc_key))


def test_feasible_input_unchanged(bk):
    cases = []
    sc = toy.set_cover(40, 25, seed=0)
    cases.append((sc, np.ones((8, sc.n))))
    ks = toy.knapsack(30, seed=0)
    cases.append((ks, np.zeros((8, ks.n))))
    mx = toy.mixed()
    cases.append((mx, np.tile([3.0, 2.0, 0.0, 0.0], (8, 1))))
    for prob, X0 in cases:
        assert all(prob.check(x).feasible for x in X0)
        for mode in ("cutoff_row", "weighted", "none"):
            fj = make_fj(prob, bk, mode=mode, seed=1)
            fj.set_cutoff(None)
            X = cp.asarray(X0)
            c = fj.run(X)
            assert c["moves"] == 0 and c["local_minima"] == 0 and c["finished_feasible"] == len(X0)
            assert np.array_equal(cp.asnumpy(X), X0)


def test_no_rows_only_rounds_and_clips(bk):
    """m = 0 (and no cutoff row: zero objective): nothing to repair; genes are rounded and clipped, nothing else."""
    n = 6
    prob = from_arrays(sps.csr_matrix((0, n)), np.zeros(0), np.zeros(0), np.zeros(n), np.full(n, 3.0), np.zeros(n),
                       np.array([True, True, True, False, False, False]))
    fj = make_fj(prob, bk)
    X = cp.asarray([[0.4, 2.6, 9.0, -1.0, 1.5, 7.0]])
    c = fj.run(X)
    assert c["finished_feasible"] == 1 and c["moves"] == 0
    assert np.array_equal(cp.asnumpy(X), [[0.0, 3.0, 3.0, 0.0, 1.5, 3.0]])


def test_chunking_matches_single_chunk(bk):
    prob = toy.set_cover(200, 120, density=0.05, seed=1)
    X0 = random_starts(prob, 40, seed=2, p_one=0.02)
    big = make_fj(prob, bk, seed=5)
    Xa = cp.asarray(X0)
    ca = big.run(Xa)
    assert ca["chunk"] == 40
    per_ind = 24 * big.m_aug + 80            # X is float64 and C-contiguous: the kernel works on X, no x buffer
    small = make_fj(prob, bk, seed=5, budget=7 * per_ind)
    Xb = cp.asarray(X0)
    cb = small.run(Xb, keep_state=True)
    assert cb["chunk"] == 7
    check_invariants(small, prob, Xb)
    assert np.array_equal(cp.asnumpy(Xa), cp.asnumpy(Xb))
    for key in ("moves", "local_minima", "steps", "finished_feasible", "finished_stall", "finished_budget"):
        assert ca[key] == cb[key], key


@pytest.mark.parametrize("dtype", ["int8", "int32", "float64"])
def test_gene_dtypes(bk, dtype):
    prob = toy.set_cover(200, 120, density=0.05, seed=1)
    X0 = random_starts(prob, 16, seed=3, p_one=0.02)
    ref = make_fj(prob, bk, seed=8)
    Xr = cp.asarray(X0)
    ref.run(Xr)
    fj = make_fj(prob, bk, seed=8, budget=5 * (16 * prob.n + 24 * 121 + 80))   # several chunks + a work buffer
    X = cp.asarray(X0.astype(dtype))
    c = fj.run(X)
    assert X.dtype == np.dtype(dtype)
    assert c["chunk"] < 16
    assert np.array_equal(cp.asnumpy(X).astype(np.float64), cp.asnumpy(Xr))


def test_weighted_mode_and_general_integers(bk):
    """General integers in [0, 60] with ranged rows (jump values matter) under `weighted`: invariants hold and the
    feasibility rate matches the NumPy reference. (Measured 2026-10-06 on 64 starts × 4 seeds: GPU 0.80-0.84, reference
    0.89; with 30 rows of band ±5 both reach 0 — too hard a toy for single-variable moves.)"""
    rs = np.random.default_rng(0)
    m, n = 15, 25
    A = sps.random(m, n, density=0.25, random_state=1, data_rvs=lambda s: rs.integers(1, 9, s)).tocsr()
    A = A + sps.eye(m, n, format="csr")
    xt = rs.integers(0, 40, n).astype(float)
    act = A @ xt
    prob = from_arrays(A, act - 20.0, act + 20.0, np.zeros(n), np.full(n, 60.0), rs.integers(1, 5, n).astype(float),
                       np.ones(n, bool), sense=1)
    X0 = random_starts(prob, 32, seed=6)
    fj = make_fj(prob, bk, mode="weighted", seed=4, max_steps=5000, stall_steps=300)
    X = cp.asarray(X0)
    c = fj.run(X, keep_state=True)
    Xh = check_invariants(fj, prob, X)
    rate = np.mean([prob.check(x).feasible for x in Xh])
    ref = [ref_fj(prob, X0[b], max_steps=5000, stall_steps=300, seed=b, obj_weight=1e-3) for b in range(32)]
    ref_rate = np.mean([s == ST_FEASIBLE for _, s, _ in ref])
    assert rate >= 0.6, f"general-integer feasibility rate {rate}: {c}"
    assert abs(rate - ref_rate) <= 0.2, f"GPU {rate} vs reference {ref_rate}"


def test_setcover_throughput(bk, capsys):
    """Synthetic set cover n=50k, m=20k, P=64 (in memory, not a real instance): moves/s per block size."""
    n, m, P = 50_000, 20_000, 64
    rs = np.random.default_rng(0)
    deg = rs.integers(1, 7, n)
    cols = np.repeat(np.arange(n), deg)
    rows = rs.integers(0, m, cols.size)
    unc = np.setdiff1d(np.arange(m), rows)
    rows = np.concatenate([rows, unc])
    cols = np.concatenate([cols, rs.integers(0, n, unc.size)])
    A = sps.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(m, n))
    A.data[:] = 1.0
    prob = from_arrays(A, np.ones(m), np.full(m, np.inf), np.zeros(n), np.ones(n), rs.integers(1, 100, n).astype(float),
                       np.ones(n, bool), sense=1)
    lines = [f"set cover n={n} m={m} nnz={prob.nnz} P={P}, start x = 0 (all rows violated)"]
    for threads in (32, 64, 128):
        fj = make_fj(prob, bk, mode="cutoff_row", seed=1, max_steps=20_000, stall_steps=200, budget=512 << 20,
                     threads=threads)
        fj.run(cp.zeros((2, n)))                                       # warm-up (NVRTC compile, adaptive slice)
        X = cp.zeros((P, n), dtype=cp.int8)
        c = fj.run(X)
        assert c["moves"] > 0
        mps = c["moves"] / c["seconds"]
        lines.append(f"threads={threads}: {c['moves']} moves in {c['seconds']:.3f} s = {mps:,.0f} moves/s "
                     f"({mps / P:,.0f} per individual), feasible {c['finished_feasible']}/{P}, "
                     f"budget {c['finished_budget']}, stall {c['finished_stall']}, launches {c['launches']}, "
                     f"peak {c['peak_bytes'] / 2**20:.0f} MiB")
        Xfeas = X.copy()
    # cutoff regime (threads=128): from those feasible covers, ask for a 3 % better key. The cutoff row spans all 50k
    # columns, so a full row scan costs ~nnz per step; row_scan_cap bounds it (off by default = the spec's rule).
    keys = cp.asnumpy(Xfeas).astype(np.float64) @ prob.c
    for cap, filt in ((0, False), (4096, False), (4096, True), (1024, True)):
        fj = make_fj(prob, bk, mode="cutoff_row", seed=1, max_steps=20_000, stall_steps=200, budget=512 << 20,
                     threads=128, row_scan_cap=cap, row_scan_filter=filt)
        fj.set_cutoff(0.97 * float(keys.min()))
        X = Xfeas.copy()
        c = fj.run(X)
        keys2 = cp.asnumpy(X).astype(np.float64) @ prob.c
        lines.append(f"cutoff active, row_scan_cap={cap} filter={filt}: {c['steps'] / c['seconds']:,.0f} steps/s, "
                     f"{c['moves'] / c['seconds']:,.0f} moves/s, {c['seconds']:.3f} s, reached {c['finished_feasible']}"
                     f"/{P}, mean key {keys.mean():,.0f} -> {keys2.mean():,.0f}")
    with capsys.disabled():
        print("\n" + "\n".join(lines))
