"""Trajectory metrics against hand-computed values (itrea.analysis.metrics)."""
from __future__ import annotations

import math

import pytest

from itrea.analysis import metrics as M


def test_berthold_gap_cases():
    assert M.gap(None, 5.0) == 1.0
    assert M.gap(5.0, 5.0) == 0.0
    assert M.gap(0.0, 0.0) == 0.0
    assert M.gap(-1.0, 2.0) == 1.0                         # signs differ
    assert M.gap(8.0, 10.0) == pytest.approx(0.2)
    assert M.gap(3.0, 0.0) == 1.0                          # reference 0: any other value is gap 1


def test_primal_gap_flags():
    g, f = M.primal_gap(3.0, 0.0, 1)
    assert g == 1.0 and M.REF_ZERO in f and M.SIGN_CROSSING not in f
    g, f = M.primal_gap(-2.0, 5.0, 1)                       # min, reference 5, incumbent -2 BEATS it (gap 0)
    assert g == 0.0 and M.BEATS_REF in f
    g, f = M.primal_gap(-2.0, 5.0, -1)                      # max: -2 is worse than 5, signs differ
    assert g == 1.0 and M.SIGN_CROSSING in f and M.BEATS_REF not in f
    g, f = M.primal_gap(0.0, 4.0, -1)                       # max, incumbent exactly 0, reference 4: Berthold's exact
    assert g == 1.0 and M.SIGN_CROSSING not in f            # gap 1, not a sign crossing (review 2026-10-06, F3)
    g, f = M.primal_gap(7.0 - 1e-12, 7.0, 1)                # below the numerics margin: equality, no flag
    assert g == 0.0 and not f
    g, f = M.primal_gap(5.0, None, 1)
    assert math.isnan(g) and M.NO_REF in f


def test_primal_integral_step_function():
    # T = 10, reference 10 (min): incumbent 20 at t = 2 (gap 0.5), 10 at t = 6 (gap 0)
    steps = M.best_steps([(2.0, 20.0), (6.0, 10.0)], 1)
    P = M.primal_integral(steps, 10.0, 1, 10.0)
    assert P.value == pytest.approx((2 * 1.0 + 4 * 0.5 + 4 * 0.0) / 10)
    assert not P.flags and P.beats_ref_t is None
    # an incumbent after T is ignored; no incumbent at all is P = 1
    assert M.primal_integral(M.best_steps([(12.0, 10.0)], 1), 10.0, 1, 10.0).value == 1.0
    assert M.primal_integral([], 10.0, 1, 10.0).value == 1.0


def test_primal_integral_maximisation_and_ties():
    # max, reference 100: 50 at 1 (gap 0.5), 50 again at 2 (tie, no step), 80 at 4 (gap 0.2), 70 at 5 (worse, ignored)
    steps = M.best_steps([(1, 50.0), (2, 50.0), (4, 80.0), (5, 70.0)], -1)
    assert steps == [(1, 50.0), (4, 80.0)]
    P = M.primal_integral(steps, 100.0, -1, 10.0)
    assert P.value == pytest.approx((1 * 1 + 3 * 0.5 + 6 * 0.2) / 10)


def test_ref_zero_and_sign_crossing_flags_on_the_integral():
    P = M.primal_integral(M.best_steps([(1.0, 3.0), (5.0, 0.0)], 1), 0.0, 1, 10.0)
    assert P.value == pytest.approx(5 / 10) and M.REF_ZERO in P.flags     # gap 1 until the exact optimum
    P = M.primal_integral(M.best_steps([(1.0, -5.0), (4.0, 2.0)], -1), 10.0, -1, 10.0)
    assert M.SIGN_CROSSING in P.flags                                      # -5 vs +10 under maximisation
    assert P.value == pytest.approx((1 + 3 * 1 + 6 * 0.8) / 10)


def test_beats_reference_is_gap_zero_and_logged():
    P = M.primal_integral(M.best_steps([(2.0, 12.0), (3.0, 9.0)], 1), 10.0, 1, 10.0)
    assert P.value == pytest.approx((2 * 1 + 1 * (2 / 12)) / 10)
    assert M.BEATS_REF in P.flags and P.beats_ref_t == 3.0


def test_times_to_target_censored():
    steps = M.best_steps([(2.0, 20.0), (6.0, 10.2), (8.0, 10.0)], 1)
    assert M.time_to_first(steps, 10.0) == M.Censored(2.0, True)
    assert M.time_to_gap(steps, 10.0, 1, 10.0, 0.05) == M.Censored(6.0, True)   # 0.2/10.2 = 1.96 %
    assert M.time_to_gap(steps, 10.0, 1, 10.0, 0.01) == M.Censored(8.0, True)
    assert M.time_to_gap(steps, 10.0, 1, 7.0, 0.01) == M.Censored(7.0, False)   # censored at T
    assert M.time_to_first([], 300.0) == M.Censored(300.0, False)
    assert M.time_to_gap(steps, None, 1, 10.0, 0.01) == M.Censored(10.0, False)


def test_portfolio_merge_and_negative_time():
    a = M.best_steps([(1.0, 9.0), (5.0, 7.0)], 1)
    b = M.best_steps([(3.0, 8.0), (4.0, 7.5)], 1)
    assert M.merge_steps(a, b, sense=1) == [(1.0, 9.0), (3.0, 8.0), (4.0, 7.5), (5.0, 7.0)]
    assert M.best_steps([(-0.5, 3.0)], 1) == [(0.0, 3.0)] and M.clamped([(-0.5, 3.0)])


def test_dual_steps_and_primal_dual_integral():
    dual = M.dual_steps([(0.0, "-inf"), (1.0, 4.0), (2.0, 3.0), (3.0, 5.0)], 1)      # trace strings, monotone
    assert dual == [(1.0, 4.0), (3.0, 5.0)]
    primal = M.best_steps([(2.0, 10.0)], 1)
    v = M.primal_dual_integral(primal, dual, 10.0)
    assert v == pytest.approx((1 + 1 + 1 * 0.6 + 7 * 0.5) / 10)


def test_summarize_keys_and_values():
    s = M.summarize(M.best_steps([(2.0, 20.0), (6.0, 10.0)], 1), 10.0, 1, 10.0, prefix="x_")
    assert s["x_pt"] == pytest.approx(0.4) and s["x_final_gap"] == 0.0 and s["x_best_obj"] == 10.0
    assert s["x_t_gap1"] == 6.0 and s["x_t_gap1_event"] and s["x_n_steps"] == 2
