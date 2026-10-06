"""Pre-registered statistics (itrea.analysis.stats) and the lineage odds ratio (itrea.analysis.lineage)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as st

from itrea.analysis import lineage as L
from itrea.analysis import stats as S


# --------------------------------------------------------------------------------------------------- effect d
def test_effect_d_bounded_and_floored():
    assert S.effect_d(0.5, 0.25) == pytest.approx(0.5)          # arm better
    assert S.effect_d(0.25, 0.5) == pytest.approx(-0.5)         # arm worse
    assert S.effect_d(1.0, 0.0) == 1.0 and S.effect_d(0.0, 1.0) == -1.0
    assert S.effect_d(0.0, 0.0) == 0.0
    assert S.effect_d(1e-4, 0.0) == pytest.approx(1e-4 / 1e-3)  # floor 1e-3 keeps tiny P from blowing up


def test_instance_effects_average_seeds_then_d_and_exclude_double_ones():
    df = pd.DataFrame([
        dict(instance="a", seed=1000, arm="A3", pt=0.2), dict(instance="a", seed=1001, arm="A3", pt=0.4),
        dict(instance="a", seed=1000, arm="A0", pt=0.6), dict(instance="a", seed=1001, arm="A0", pt=0.6),
        dict(instance="b", seed=1000, arm="A3", pt=1.0), dict(instance="b", seed=1001, arm="A3", pt=1.0),
        dict(instance="b", seed=1000, arm="A0", pt=1.0), dict(instance="b", seed=1001, arm="A0", pt=1.0),
        dict(instance="c", seed=1000, arm="A3", pt=0.5), dict(instance="c", seed=1000, arm="A0", pt=0.4),
        dict(instance="c", seed=1001, arm="A0", pt=0.9),            # seed 1001 of A3 missing: common seeds only
    ])
    e = S.instance_effects(df, "A3", "A0").set_index("instance")
    assert e.loc["a", "d"] == pytest.approx((0.6 - 0.3) / 0.6) and e.loc["a", "n_seeds"] == 2
    assert bool(e.loc["b", "excluded"]) and not bool(e.loc["a", "excluded"])
    assert e.loc["c", "n_seeds"] == 1 and e.loc["c", "d"] == pytest.approx((0.4 - 0.5) / 0.5)
    with pytest.raises(ValueError):
        S.instance_effects(pd.concat([df, df.iloc[:1]]), "A3", "A0")


# --------------------------------------------------------------------------------------------------- Wilcoxon
def test_wilcoxon_method_choice_matches_scipy():
    d = np.array([0.3, -0.1, 0.25, 0.4, 0.0, 0.15, -0.05, 0.2])
    w = S.wilcoxon_d(d)
    assert w["method"] == "exact" and w["n_nonzero"] == 7 and not w["ties"]
    assert w["p"] == pytest.approx(st.wilcoxon(d[d != 0], method="exact").pvalue)
    t = S.wilcoxon_d(np.array([0.1, 0.1, -0.2, 0.3, 0.4]))           # tie in |d|: normal approximation
    assert t["ties"] and t["method"] == "asymptotic"
    big = np.linspace(-0.3, 0.9, 60)
    assert S.wilcoxon_d(big)["method"] == "asymptotic"                # n > 50
    assert S.wilcoxon_d(np.zeros(5))["p"] == 1.0


def test_bootstrap_deterministic():
    d = np.random.default_rng(1).normal(0.1, 0.3, 32)
    a, b = S.bootstrap_mean(d), S.bootstrap_mean(d)
    assert a == b and a["n_boot"] == 10_000 and a["lo"] < a["mean"] < a["hi"]
    c = S.bootstrap_mean(d, seed=7)
    assert (c["lo"], c["hi"]) != (a["lo"], a["hi"])


def test_holm_hand_computed():
    adj, rej = S.holm([0.01, 0.04, 0.03, 0.20])
    # sorted 0.01, 0.03, 0.04, 0.20 -> 0.04, 0.09, max(0.09, 0.08) = 0.09, max(0.09, 0.20) = 0.20
    np.testing.assert_allclose(adj, [0.04, 0.09, 0.09, 0.20])
    assert rej.tolist() == [True, False, False, False]
    assert S.holm([0.9, 0.8])[0].tolist() == [1.0, 1.0]


def test_decision_rule_cases():
    assert S.decide(0.01, 0.05, 0.30) == S.SUPPORTED
    assert S.decide(0.01, -0.30, -0.05) == S.ARM_WORSE          # p small but the interval is below 0
    assert S.decide(0.40, -0.10, 0.04) == S.NEGATIVE            # whole interval below the +0.05 margin
    assert S.decide(0.30, -0.10, 0.20) == S.INCONCLUSIVE
    assert S.decide(0.08, 0.01, 0.20) == S.INCONCLUSIVE         # interval above 0 but p >= alpha
    assert S.decide(0.01, -0.02, 0.30) == S.INCONCLUSIVE        # p < alpha but the interval includes 0
    assert S.intersection_union([S.SUPPORTED, S.SUPPORTED]) == S.SUPPORTED
    assert S.intersection_union([S.SUPPORTED, S.INCONCLUSIVE]) != S.SUPPORTED


def test_compare_end_to_end():
    rs = np.random.default_rng(5)
    rows = []
    for i in range(20):
        base = rs.uniform(0.3, 0.9)
        for s in (1000, 1001):
            rows += [dict(instance=f"i{i}", seed=s, arm="A0", pt=base),
                     dict(instance=f"i{i}", seed=s, arm="A3", pt=base * 0.7 + rs.normal(0, 0.01))]
    c = S.compare(pd.DataFrame(rows), "A3", "A0")
    assert c["n_instances"] == 20 and c["decision"] == S.SUPPORTED and c["better"] == 20
    assert c["ratio_of_means_gain"] == pytest.approx(0.3, abs=0.02)


# ------------------------------------------------------------------------------------------ censored times
def test_censored_median_rule():
    assert S.censored_median([10.0, 20.0], [True, True]) == (15.0, True)
    assert S.censored_median([10.0, 300.0], [True, False]) == (155.0, False)    # a lower bound
    assert S.censored_median([300.0, 300.0], [False, False]) == (300.0, False)
    assert S.censored_median([5.0, 9.0, 300.0], [True, True, False]) == (9.0, True)


def test_prentice_scores_without_censoring_are_wilcoxon_like():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    sc = S.prentice_scores(t, np.ones(4, bool))
    # S~ = 4/5, 3/5·… : scores increase with time and sum to (about) zero
    assert np.all(np.diff(sc) > 0)
    assert sc[0] == pytest.approx(1 - 2 * 4 / 5)


def test_paired_prentice_wilcoxon_agrees_in_sign_with_signed_rank():
    rs = np.random.default_rng(2)
    ctrl = rs.uniform(10, 200, 25)
    arm = ctrl * 0.6 + rs.normal(0, 5, 25)
    r = S.paired_prentice_wilcoxon(arm, np.ones(25, bool), ctrl, np.ones(25, bool))
    assert r["z"] > 0 and r["p"] < 0.05                           # arm faster: positive z, like d
    w = st.wilcoxon(ctrl - arm)
    assert w.pvalue < 0.05
    # all censored at T on both sides: nothing to see
    z = S.paired_prentice_wilcoxon([300.0] * 4, [False] * 4, [300.0] * 4, [False] * 4)
    assert z["z"] == 0.0 and z["p"] == 1.0
    # an arm that finds a solution where the control never does is faster
    q = S.paired_prentice_wilcoxon([50.0] * 6, [True] * 6, [300.0] * 6, [False] * 6)
    assert q["z"] > 0


def test_censored_sign_test():
    r = S.censored_sign_test([10, 20, 300, 50, 300], [True, True, False, True, False],
                             [30, 10, 40, 300, 300], [True, True, True, False, False])
    # pairs: faster, slower, slower (control event before arm censoring), faster, dropped (both censored)
    assert (r["arm_faster"], r["arm_slower"], r["n"]) == (2, 2, 4)


def test_compare_censored_on_seed_medians():
    df = pd.DataFrame([dict(instance="a", seed=s, arm=arm, t_first=t, t_first_event=e)
                       for s in (1000, 1001)
                       for arm, t, e in (("A3", 20.0, True), ("A0", 300.0, False))])
    r = S.compare_censored(df, "A3", "A0", "t_first", "t_first_event")
    assert r["n_instances"] == 1 and r["sign"]["arm_faster"] == 1


# ------------------------------------------------------------------------------------------ Mantel–Haenszel
def test_mantel_haenszel_single_table_and_pooling():
    one = S.mantel_haenszel([(10, 5, 4, 20)])
    assert one["or_mh"] == pytest.approx(10 * 20 / (5 * 4))
    se = math.sqrt(1 / 10 + 1 / 5 + 1 / 4 + 1 / 20)                  # RBG reduces to Woolf for one table
    assert one["se_log"] == pytest.approx(se)
    # two strata with the same OR pool to that OR
    two = S.mantel_haenszel([(8, 4, 2, 4), (2, 1, 3, 6)])
    assert two["or_mh"] == pytest.approx(4.0)
    assert math.isnan(S.mantel_haenszel([(0, 5, 0, 5)])["or_mh"])


# ------------------------------------------------------------------------------------------------- lineage
def _genealogy(path, with_flags=True):
    """Generation 1: 10 bred children; the 2 cases both have a Rank 0 parent; 3 of the 8 others do."""
    ids = np.arange(14)
    pa = np.array([-1, -1, -1, -1] + [0] * 10)
    pb = np.array([-1, -1, -1, -1] + [1] * 10)
    rank_pa = np.array([-1] * 4 + [0, 0, 0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int8)
    rank_pb = np.array([-1] * 4 + [1] * 10, dtype=np.int8)
    birth = np.array([0] * 4 + [1] * 10, dtype=np.int32)
    feasible = np.array([True, False, True, False] + [True, True, False, False, False, True, False, False, False,
                                                      False])
    key = np.array([10.0, 5.0, 12.0, 4.0] + [8.0, 7.0, 3.0, 3.0, 3.0, 11.0, 2.0, 2.0, 2.0, 2.0])
    cols = dict(id=ids, pa=pa, pb=pb, rank_pa=rank_pa, rank_pb=rank_pb, origin=np.zeros(14, np.uint16),
                birth=birth, key=key, V=np.where(feasible, 0.0, 1.0), feasible=feasible, t_first=np.zeros(14))
    if with_flags:
        cols["is_new_best"] = np.array([True, False, False, False, True, True] + [False] * 8)
        cols["is_accepted_post"] = np.zeros(14, bool)
    np.savez(path, **cols)


def test_lineage_tables_and_odds_ratio(tmp_path):
    d = tmp_path / "x__A5__s1000__300s__c"
    d.mkdir()
    _genealogy(d / "ea.genealogy.npz")
    r = L.run_tables(d)
    assert r["cases"] == 3 and r["cases_without_parents"] == 1 and not r["derived_flags"]
    assert len(r["tables"]) == 1
    t = r["tables"][0]
    assert (t.a, t.b, t.c, t.d) == (2, 5, 0, 5)                        # reference = all bred children
    tn = L.run_tables(d, reference="noncases")["tables"][0]
    assert (tn.a, tn.b, tn.c, tn.d) == (2, 3, 0, 5)
    pooled = L.pooled([d])
    assert pooled["or_mh"] == math.inf and pooled["runs"] == 1          # no unexposed case: OR unbounded


def test_lineage_derives_flags_when_absent(tmp_path):
    d = tmp_path / "x__A5__s1000__300s__c"
    d.mkdir()
    _genealogy(d / "ea.genealogy.npz", with_flags=False)
    df, derived = L.mark_cases(L.load_genealogy(d / "ea.genealogy.npz"), d)
    assert derived
    # running best over feasible keys in (birth, id) order: 10 (id0), then 8 (id4), 7 (id5); 12 and 11 are not
    assert df.loc[df["case"], "id"].tolist() == [0, 4, 5]
    assert L.find_genealogies(tmp_path, ["A5"]) == [d] and L.find_genealogies(tmp_path, ["A3"]) == []
