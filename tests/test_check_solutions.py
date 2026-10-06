"""The independent exact checker on toy models: both readers, the 1e-6 tolerance, the exact re-check, objective and
name mismatches, and the run-directory level (itrea.analysis.check_solutions)."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sps

from itrea.analysis import check_solutions as C
from itrea.benchmarks import toy
from itrea.problem import from_arrays
from itrea.trace import save_incumbents

READERS = ("highs", "scip")


@pytest.fixture(scope="module")
def mixed_mps(tmp_path_factory):
    p = toy.mixed()
    p.offset = 2.5
    return p, toy.write_mps(p, tmp_path_factory.mktemp("chk") / "mixed.mps")


@pytest.mark.parametrize("reader", READERS)
def test_readers_agree_on_the_model(mixed_mps, reader):
    p, mps = mixed_mps
    m = C.read_model(mps, reader)
    assert m.names == p.names and m.sense == 1 and m.offset == 2.5
    np.testing.assert_array_equal(m.lb, p.lb)
    np.testing.assert_array_equal(m.ub, p.ub)                  # SCIP's 1e20 normalised to inf
    np.testing.assert_array_equal(m.rl, p.rl)
    np.testing.assert_array_equal(m.ru, p.ru)
    np.testing.assert_array_equal(m.isint, p.isint)
    np.testing.assert_array_equal(m.A.toarray(), p.A.toarray())


@pytest.mark.parametrize("reader", READERS)
def test_feasible_accepted_infeasible_rejected_at_1e6(mixed_mps, reader):
    p, mps = mixed_mps
    m = C.read_model(mps, reader)
    x = np.array([3.0, 2.0, 0.0, 0.0])
    r = C.check_named(m, p.names, x, p.objective(x))
    assert r["ok"] and r["feasible"] and r["obj_recomputed"] == 9.5 and r["reason"] == ""
    # equality row r1 (x0 - x1 + y1 = 1) off by 2e-6: rejected; by 5e-7: accepted
    for dy, ok in ((2e-6, False), (5e-7, True)):
        y = np.array([3.0, 2.0, 0.0, dy])
        r = C.check_named(m, p.names, y, p.objective(y))
        assert r["ok"] is ok and (ok or r["reason"] == "row")
    # integrality off by 2e-6, bound off by 2e-6
    r = C.check_named(m, p.names, np.array([3.0 + 2e-6, 2.0, 0.0, -2e-6]), None)
    assert not r["ok"] and "integrality" in r["reason"]
    r = C.check_named(m, p.names, np.array([3.0, 2.0, -2e-6, 6e-6]), None)
    assert "bound" in r["reason"] and r["worst_bound"] == pytest.approx(2e-6)


@pytest.mark.parametrize("reader", READERS)
def test_objective_and_name_mismatch_fail(mixed_mps, reader):
    p, mps = mixed_mps
    m = C.read_model(mps, reader)
    x = np.array([3.0, 2.0, 0.0, 0.0])
    r = C.check_named(m, p.names, x, p.objective(x) + 1e-3)
    assert r["feasible"] and not r["obj_ok"] and not r["ok"] and r["reason"] == "objective"
    r = C.check_named(m, p.names[:3] + ["zz"], x, p.objective(x))
    assert not r["ok"] and r["unknown_vars"] == 1 and r["missing_vars"] == 1


@pytest.mark.parametrize("reader", READERS)
def test_maximisation_with_offset(tmp_path, reader):
    k = toy.knapsack(12, seed=1)
    k.offset = -3.0
    mps = toy.write_mps(k, tmp_path / "knap.mps")
    m = C.read_model(mps, reader)
    assert m.sense == -1 and m.offset == -3.0
    x = np.zeros(12)
    x[:3] = 1
    assert C.check_named(m, k.names, x, k.objective(x))["ok"]
    assert not C.check_named(m, k.names, np.ones(12), k.objective(np.ones(12)))["ok"]   # over capacity


def test_exact_recheck_decides_near_the_tolerance(tmp_path):
    # one row: sum of ten 0.1's <= 1 - 1e-6. The float sum is 0.9999999999999999, so the float violation sits within
    # its error bound of the tolerance and the row must be re-checked in exact arithmetic.
    n = 10
    ru = 1.0 - 1e-6
    p = from_arrays(sps.csr_matrix(np.ones((1, n))), [-np.inf], [ru], np.zeros(n), np.ones(n), np.zeros(n),
                    np.zeros(n, bool), names=[f"c{j}" for j in range(n)])
    mps = toy.write_mps(p, tmp_path / "near.mps")
    m = C.read_model(mps, "highs")
    x = np.full(n, 0.1)
    r = C.check_vector(m, x, 0.0)
    assert r["exact_rows"] == 1
    exact = sum(Fraction(0.1) for _ in range(n)) - Fraction(ru)
    assert r["worst_row_rel"] == pytest.approx(float(exact), rel=0, abs=1e-18)
    assert r["feasible"] == (exact <= Fraction(C.TOL))


def test_exact_dot():
    a, x = np.array([0.1, 0.2, 0.3, 1e300, -1e300]), np.array([1.0, 1.0, -1.0, 1.0, 1.0])
    assert C.exact_dot(a, x) == Fraction(0.1) + Fraction(0.2) - Fraction(0.3)
    assert C.exact_dot(np.zeros(3), np.ones(3)) == 0


def test_non_finite_vector_rejected(mixed_mps):
    p, mps = mixed_mps
    m = C.read_model(mps, "highs")
    r = C.check_vector(m, np.array([3.0, np.nan, 0.0, 0.0]), 1.0)
    assert not r["ok"] and r["reason"] == "non_finite"


def test_producer_reader_mapping():
    assert C.producer_of("solver", {}, "A3") == "scip" and C.reader_for("scip") == "highs"
    assert C.producer_of("solver", {}, "H3") == "highs" and C.reader_for("highs") == "scip"
    assert C.producer_of("ea", {}, "A3") == "ea" and C.reader_for("ea") == "scip"
    assert C.producer_of("cuopt", {}, "A2") == "cuopt" and C.reader_for("cuopt") == "highs"
    assert C.producer_of("solver2", {"solver": "highs"}, "A1") == "highs"


def test_check_run_dir_every_incumbent(tmp_path, mixed_mps):
    p, mps = mixed_mps
    d = tmp_path / "run"
    d.mkdir()
    (d / "run.json").write_text(f'{{"run_id": "mixed__A3__s1__10s__x", "arm": "A3", "instance": "{mps}"}}')
    good, bad = np.array([3.0, 2.0, 0.0, 0.0]), np.array([3.0, 2.0, 0.0, 0.5])
    save_incumbents(d / "solver.incumbents.npz", p.names, [(1.0, p.objective(good), "self", good)])
    # the EA writes its own variable order: reversed here, mapped back by name
    rev = p.names[::-1]
    save_incumbents(d / "ea.incumbents.npz", rev, [(0.5, p.objective(bad), "self", bad[::-1]),
                                                   (2.0, p.objective(good), "self", good[::-1])])
    rows = C.check_run_dir(d)
    assert [(r["role"], r["k"], r["ok"]) for r in rows] == [("ea", 0, False), ("ea", 1, True), ("solver", 0, True)]
    assert {r["reader"] for r in rows if r["role"] == "ea"} == {"scip"}
    assert C.main([str(d)]) == 1                                # a failed incumbent is an error


def test_unreadable_model_leaves_incumbents_unchecked(tmp_path, mixed_mps):
    p, _ = mixed_mps
    x = np.array([3.0, 2.0, 0.0, 0.0])
    npz = save_incumbents(tmp_path / "solver.incumbents.npz", p.names, [(1.0, 9.5, "self", x)])
    rows = C.check_incumbents(npz, str(tmp_path / "missing.mps"), "scip")
    assert len(rows) == 1 and not rows[0]["checked"] and not rows[0]["ok"]
    assert rows[0]["reason"].startswith("unchecked")
