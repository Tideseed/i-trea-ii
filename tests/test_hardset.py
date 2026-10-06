"""Reference parsing (itrea.benchmarks.references) and the hard-pool freeze (itrea.benchmarks.hardset) on a FAKE
screening table; one real-data test parses the candidate lists only (no solve, no MPS read)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from itrea.benchmarks import hardset as H
from itrea.benchmarks import references as R
from itrea.paths import DATA

SOLU = """=opt=  air05   26374
=best= rail01  -70.5700000013
=inf=  bnatt500
=unbd= weird1
=unkn= open1
=opt=  zeroref 0
"""


# ------------------------------------------------------------------------------------------------ references
def test_parse_solu_kinds():
    s = R.parse_solu(SOLU)
    assert s["air05"] == R.SoluEntry("air05", "opt", 26374.0)
    assert s["rail01"].kind == "best_known" and s["rail01"].value == pytest.approx(-70.57, abs=1e-6)
    assert s["bnatt500"].kind == "infeasible" and not s["bnatt500"].has_value
    assert s["weird1"].kind == "unbounded" and s["open1"].kind == "unknown"
    assert R.is_zero(s["zeroref"].value)
    with pytest.raises(ValueError):
        R.parse_solu("=opt= x")
    with pytest.raises(ValueError):
        R.parse_solu("=xyz= x 1")
    with pytest.raises(ValueError):
        R.parse_solu("=opt= x 1\n=opt= x 2")


def test_reference_csv_round_trip_and_resolution(tmp_path):
    df = pd.DataFrame([dict(instance="ca__a", set="pglib_uc", reference=1234.5, sense=1, kind="reference_run",
                            ref_zero=False, source="REF"),
                       dict(instance="k0", set="ml4co_item_placement", reference=0.0, sense=1,
                            kind="screening_best", ref_zero=False, source="SCREEN")])
    p = R.write_reference_csv(df, tmp_path / "reference.csv")
    back = R.read_reference_csv(p)
    assert back["instance"].tolist() == ["ca__a", "k0"] and back["ref_zero"].tolist() == [False, True]
    solu = R.parse_solu(SOLU)
    # frozen first, then .solu (MIPLIB only), then provisional screening best, then best of study
    assert R.resolve("ca__a", "pglib_uc", back, solu).kind == "reference_run"
    r = R.resolve("air05", "miplib", back, solu, {"air05": 1.0})
    assert (r.value, r.kind, r.frozen) == (26374.0, "opt", True)
    r = R.resolve("open1", "miplib", back, solu, {"open1": 5.0})
    assert (r.value, r.kind, r.frozen) == (5.0, "screening_best_provisional", False)
    assert R.resolve("zz", "x", back, solu, None, {"zz": 2.0}).kind == "best_of_study"
    assert R.resolve("zz", "x", back, solu).kind == "none"
    bad = df.assign(kind="best_of_study")
    R.write_reference_csv(bad, tmp_path / "bad.csv")
    with pytest.raises(ValueError):
        R.read_reference_csv(tmp_path / "bad.csv")                      # provisional kinds cannot be frozen


def test_freeze_references():
    inst = pd.DataFrame(dict(instance=["air05", "ca__a", "nothing"], set=["miplib", "pglib_uc", "pglib_uc"],
                             sense=[1, -1, 1]))
    runs = pd.DataFrame(dict(instance=["ca__a", "ca__a", "ca__a"], arm=["SCREEN", "REF", "A3"],
                             best_obj=[10.0, 12.0, 99.0]))
    f = R.freeze_references(inst, runs, R.parse_solu(SOLU)).set_index("instance")
    assert f.loc["air05", "reference"] == 26374.0 and f.loc["air05", "kind"] == "opt"
    assert f.loc["ca__a", "reference"] == 12.0 and f.loc["ca__a", "kind"] == "reference_run"   # max, A3 not used
    assert "nothing" not in f.index


def test_stem():
    assert R.stem("data/miplib/inst/air05.mps.gz") == "air05"
    assert R.stem("ca__2014-09-01_reserves_0.mps.gz") == "ca__2014-09-01_reserves_0"


# ------------------------------------------------------------------------------------------- groups, families
def test_family_and_group_rules():
    assert H.miplib_prefix("neos-1122047") == "neos" == H.miplib_prefix("neos859080")
    assert H.miplib_prefix("supportcase10") == "supportcase" and H.miplib_prefix("comp07-2idx") == "comp"
    assert H.miplib_prefix("30n20b8") == "30n20b8" and H.miplib_prefix("50v-10") == "50v"
    assert H.pglib_group("ca__2014-09-01_reserves_3") == "ca__2014-09-01"
    assert H.pglib_group("rts_gmlc__2020-01-27") == "rts_gmlc__2020-01-27"
    assert H.pglib_family("rts_gmlc__2020-01-27") == "rts_gmlc"
    c = H.assign_groups(pd.DataFrame(dict(instance=["sp97ar", "sp150x300d", "air04", "air05", "xyz"],
                                          set="miplib", path="", miplib_group=["sp9", "sp9", "–", "–", "sp9"])))
    g = dict(zip(c["instance"], c["group"]))
    assert g["sp97ar"] == g["sp150x300d"] == g["xyz"]       # one MIPLIB Group joins different prefixes
    assert g["air04"] == g["air05"] != g["sp97ar"]


# ------------------------------------------------------------------------------------------- the pool freeze
def fake_inputs(seed: int = 0):
    """60 MIPLIB (12 families), 32 PGLib (5 ca days × 4 reserves + 12 rts_gmlc), 60 ML4CO; screening on SCIP."""
    rs = np.random.default_rng(seed)
    names, sets, groups = [], [], []
    for f in range(12):
        for j in range(5):
            names.append(f"fam{chr(97 + f)}-{j}")
            sets.append("miplib")
            groups.append("–")
    for day in range(5):
        for res in (0, 1, 3, 5):
            names.append(f"ca__day{day}_reserves_{res}")
            sets.append("pglib_uc")
            groups.append("")
    for day in range(12):
        names.append(f"rts_gmlc__2020-{day + 1:02d}-01")
        sets.append("pglib_uc")
        groups.append("")
    for i in list(range(9900, 9925)) + list(range(35)):
        names.append(f"item_placement_{i}")
        sets.append("ml4co_item_placement")
        groups.append("")
    cands = H.assign_groups(pd.DataFrame(dict(instance=names, set=sets, path=[f"data/{n}.mps.gz" for n in names],
                                              miplib_group=groups)))
    rows = []
    for k, (n, s) in enumerate(zip(names, sets)):
        kind = rs.choice(["opt", "feas", "easy"], p=[0.7, 0.15, 0.15]) if s == "miplib" else "opt"
        for seed_ in (900, 901):
            if kind == "feas":
                pt, ev = 1.0, seed_ == 900 and k % 2 == 0          # some never feasible, some feasible once
                pt = 1.0 if not ev else 0.6
            elif kind == "easy":
                pt, ev = float(rs.uniform(0, 0.04)), True
            else:
                pt, ev = float(rs.uniform(0.06, 0.95)), True
            if seed_ == 901 and kind == "easy":
                continue                                            # stage 2 not run for easy instances
            rows.append(dict(instance=n, set=s, arm="SCREEN", seed=seed_, budget=300.0, pt=pt,
                             t_first=float(rs.uniform(1, 299)) if ev else 300.0, t_first_event=ev, crashed=False,
                             solved_optimal=kind == "easy", ref_zero_instance=False, sign_crossing=False,
                             no_exchange_window=False, synthetic=False))
    runs = pd.DataFrame(rows)
    shapes = pd.DataFrame(dict(instance=names, n=1000, m=500, nnz=4000, gene_bytes=1))
    return cands, runs, shapes


def test_stage2_rule():
    mip = dict(set="screen1")
    runs = pd.DataFrame([
        dict(instance="a", arm="SCREEN", seed=900, budget=300.0, pt=0.04, t_first=5.0, t_first_event=True,
             instance_path="data/miplib/inst/a.mps.gz", **mip),
        dict(instance="b", arm="SCREEN", seed=900, budget=300.0, pt=0.05, t_first=5.0, t_first_event=True,
             instance_path="data/miplib/inst/b.mps.gz", **mip),
        dict(instance="c", arm="SCREEN", seed=900, budget=300.0, pt=1.0, t_first=300.0, t_first_event=False,
             instance_path="data/miplib/inst/c.mps.gz", **mip),
        dict(instance="d", arm="SCREEN", seed=900, budget=300.0, pt=0.5, t_first=5.0, t_first_event=True,
             instance_path="data/miplib/inst/d.mps.gz", **mip),
        dict(instance="d", arm="SCREEN", seed=901, budget=300.0, pt=0.5, t_first=5.0, t_first_event=True,
             instance_path="data/miplib/inst/d.mps.gz", **mip),
        dict(instance="e", arm="A0", seed=900, budget=300.0, pt=0.9, t_first=5.0, t_first_event=True,
             instance_path="data/miplib/inst/e.mps.gz", **mip),
    ])
    assert H.stage2_needed(runs) == ["b", "c"]


def test_stage2_always_for_pglib_and_ml4co():
    """Their stage-1 reference is the run's own best, so the P rule cannot prune them (D-002 v3.1 §1.2)."""
    runs = pd.DataFrame([
        dict(instance="ca__x", arm="SCREEN", seed=900, budget=300.0, pt=0.0, t_first=5.0, t_first_event=True,
             set="screen1", instance_path="data/pglib_uc/mps/ca__x.mps.gz"),
        dict(instance="item_placement_3", arm="SCREEN", seed=900, budget=300.0, pt=0.0, t_first=5.0,
             t_first_event=True, set="screen1", instance_path="data/ml4co/instances/1_item_placement/train/item_placement_3.mps.gz"),
        dict(instance="air05", arm="SCREEN", seed=900, budget=300.0, pt=0.0, t_first=5.0, t_first_event=True,
             set="screen1", instance_path="data/miplib/inst/air05.mps.gz"),
    ])
    assert H.stage2_needed(runs) == ["ca__x", "item_placement_3"]


def _check_pool(pool: pd.DataFrame, cap: int = 6):
    for st in H.STRATA:
        g = pool[pool["stratum"] == st.name]
        assert (g["side"] == "test").sum() == st.test and (g["side"] == "tuning").sum() == st.tuning, st.name
    sides = pool.groupby("group")["side"].nunique()
    assert (sides == 1).all()                                       # no group on both sides
    assert pool[pool["side"] == "test"].groupby("family").size().max() <= cap
    assert not ((pool["side"] == "tuning") & pool["instance"].str.startswith("rts_gmlc")).any()
    assert pool["instance"].is_unique


def test_pool_freeze_deterministic(tmp_path):
    cands, runs, shapes = fake_inputs()
    pool, table, rep = H.freeze(runs, cands=cands, shapes=shapes)
    assert not rep["shortfall"]
    _check_pool(pool)
    a = H.write_pool(pool, table, tmp_path / "a.csv", tmp_path / "ca.csv")
    # same inputs in another row order -> the same bytes
    pool2, table2, _ = H.freeze(runs.sample(frac=1.0, random_state=3), cands=cands.sample(frac=1.0, random_state=4),
                                shapes=shapes)
    b = H.write_pool(pool2, table2, tmp_path / "b.csv", tmp_path / "cb.csv")
    assert a == b and (tmp_path / "ca.csv").read_bytes() == (tmp_path / "cb.csv").read_bytes()
    # every pick spreads over the difficulty range: the opt stratum's test picks are not just the hardest ones
    opt = table[table["stratum"] == "miplib_opt"].sort_values("p_mean", ascending=False)
    picked = set(pool.loc[pool["stratum"] == "miplib_opt", "instance"])
    pos = [k for k, n in enumerate(opt["instance"]) if n in picked]
    assert max(pos) > len(picked)
    # exclusions are explained
    ex = table[table["stratum"] == ""]
    assert (ex["excluded_reason"] != "").all()


def test_pool_memory_fit_replacement():
    cands, runs, shapes = fake_inputs()
    pool0, table0, _ = H.freeze(runs, cands=cands, shapes=shapes)
    first = pool0[(pool0["stratum"] == "ml4co") & (pool0["bucket"] == 0)].iloc[0]["instance"]
    shapes2 = shapes.copy()
    shapes2.loc[shapes2["instance"] == first, "n"] = 10 ** 9          # 32 x 1e9 x 1 B > 4 GiB
    pool, table, rep = H.freeze(runs, cands=cands, shapes=shapes2)
    assert first not in set(pool["instance"]) and first in rep["unfit"]
    assert any(r["skipped"] == first for r in rep["replacements"])
    repl = pool[pool["replaces"] == first]
    assert len(repl) == 1 and repl.iloc[0]["bucket"] == 0
    _check_pool(pool)


def test_pool_ignores_ineligible_changes():
    cands, runs, shapes = fake_inputs()
    pool, _, _ = H.freeze(runs, cands=cands, shapes=shapes)
    easy = runs.groupby("instance")["pt"].max()
    easy = easy[easy < 0.05].index[0]
    runs2 = runs.copy()
    runs2.loc[runs2["instance"] == easy, "pt"] = 0.01                  # stays below the stage-2 threshold
    pool2, _, _ = H.freeze(runs2, cands=cands, shapes=shapes)
    pd.testing.assert_frame_equal(pool, pool2)


@pytest.mark.skipif(not (DATA / "miplib" / "tag_benchmark.html").exists(), reason="instance data not on disk")
def test_real_candidate_lists():
    c = H.candidates()
    assert c["set"].value_counts().to_dict() == {"miplib": 232, "ml4co_item_placement": 60, "pglib_uc": 32}
    assert "supportcase22" not in set(c["instance"]) and "bnatt500" not in set(c["instance"])
    assert c.loc[c["instance"] == "neos859080"].empty                    # =inf=
    assert c["path"].str.startswith("data/").all()                       # repository-relative, symlinks kept
    solu = R.read_solu()
    assert sum(1 for e in solu.values() if e.kind == "infeasible") >= 7
