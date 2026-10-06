"""Regression tests for the 2026-10-06 review findings on the hard-pool freeze (itrea.benchmarks.hardset):
F6 (PGLib split unit = base case), F10 (memory fit = the EA's own memory plan), F12 (tuning-side family cap),
F13 (screening runs stopped at SCIP's memory / node limit are flagged). FAKE screening tables only — no solve, no
instance read."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_hardset import _check_pool, fake_inputs

from itrea.benchmarks import hardset as H


def _table(cands, runs, shapes):
    return H.classify(cands, H.screening_table(runs)).merge(shapes, on="instance", how="left")


# ------------------------------------------------------------------------------------------------------- F6
def test_pglib_split_unit_is_the_base_case():
    """Pinned until D-002 §2's "PGLib by base system" is amended to "by base case": a ca day and its reserve variants
    are one group (one side); different ca days may sit on different sides; rts_gmlc days only on test."""
    assert H.pglib_group("ca__2014-09-01_reserves_5") == H.pglib_group("ca__2014-09-01_reserves_0") == "ca__2014-09-01"
    assert H.pglib_family("ca__2014-09-01_reserves_5") == "ca"
    for seed in (0, 1):
        cands, runs, shapes = fake_inputs(seed)
        pool, _, rep = H.freeze(runs, cands=cands, shapes=shapes)
        pool2, _, rep2 = H.freeze(runs, cands=cands, shapes=shapes)
        pd.testing.assert_frame_equal(pool, pool2)
        assert rep["sides"] == rep2["sides"]
        ca = {g: s for g, s in rep["sides"].items() if g.startswith("pglib:ca__")}
        assert set(ca.values()) == {"test", "tuning"}                    # a within-family split of ca
        assert {s for g, s in rep["sides"].items() if g.startswith("pglib:rts_gmlc__")} == {"test"}
        pg = pool[pool["stratum"] == "pglib"]
        assert set(pg.loc[pg["side"] == "tuning", "family"]) == {"pglib:ca"}
        assert pg.assign(day=pg["instance"].map(H.pglib_group)).groupby("day")["side"].nunique().max() == 1
        assert {"pglib:ca", "pglib:rts_gmlc"} == set(pg.loc[pg["side"] == "test", "family"])


# ------------------------------------------------------------------------------------------------------ F10
def _p_below_32_shape(n: int = 50_000, m: int = 1_000, gene_bytes: int = 1) -> tuple[int, int, int, int]:
    """A shape whose own EA memory plan fits only with 2 <= P < 32 at the default budget: bisect nnz (the device
    problem and local-search matrices grow with it) for the first value where the planned P drops below 32."""
    cfg = H.fit_config()
    lo, hi = 0, 1 << 40
    while lo < hi:
        mid = (lo + hi) // 2
        if H.memory_fit(n, m, mid, gene_bytes, cfg, p_fit=1)[0] < H.P_FIT:
            hi = mid
        else:
            lo = mid + 1
    return n, m, lo, gene_bytes


def test_memory_fit_is_the_engine_plan_with_p_at_least_32():
    cfg = H.fit_config()
    budget = H.gpu_budget_bytes()
    n, m, nnz, g = _p_below_32_shape()
    P, b, ok = H.memory_fit(n, m, nnz, g, cfg)
    assert 2 <= P < H.P_FIT and not ok
    assert H.P_FIT * n * g < budget / 1000                               # the old P x n x bytes rule admitted it
    assert H.memory_fit(n, m, nnz - 1, g, cfg)[0] >= H.P_FIT and H.memory_fit(n, m, nnz - 1, g, cfg)[2]
    P2, _, ok2 = H.memory_fit(1000, 500, 10 ** 11, 1, cfg)              # does not fit even at P = 2
    assert P2 == 2 and not ok2
    assert H.memory_fit(1000, 500, 4000, 1, cfg) == (256, H.memory_fit(1000, 500, 4000, 1, cfg)[1], True)
    small = H.fit_config(2 * 1024 ** 3)                                 # the budget override reaches the plan
    assert small.gpu_mem_gb == pytest.approx(2.0)
    assert H.memory_fit(n, m, nnz // 4, g, small)[0] < H.memory_fit(n, m, nnz // 4, g, cfg)[0]
    with pytest.raises(ValueError):
        H.memory_fit(10, 10, 10, 2, cfg)                                 # not a gene dtype of gene_dtype_for


def test_pool_replaces_an_instance_the_plan_cannot_hold():
    cands, runs, shapes = fake_inputs()
    pool0, _, _ = H.freeze(runs, cands=cands, shapes=shapes)
    first = pool0[(pool0["stratum"] == "ml4co") & (pool0["bucket"] == 0)].iloc[0]["instance"]
    n, m, nnz, g = _p_below_32_shape()
    shapes2 = shapes.copy()
    shapes2.loc[shapes2["instance"] == first, ["n", "m", "nnz", "gene_bytes"]] = [n, m, nnz, g]
    pool, table, rep = H.freeze(runs, cands=cands, shapes=shapes2)
    assert first not in set(pool["instance"]) and first in rep["unfit"]
    assert any(r["skipped"] == first for r in rep["replacements"])
    assert len(pool[pool["replaces"] == first]) == 1
    assert (pool["plan_p"] >= H.P_FIT).all() and pool["fits"].all()
    assert "plan_memory" in rep["fit_rule"] and rep["fit_config_hash"] == H.fit_config().hash()
    assert rep["fit_config_hash"] in rep["fit_rule"]
    _check_pool(pool)
    with pytest.raises(ValueError, match="nnz"):                        # no silent fallback to P x n x bytes
        H.select_pool(_table(cands, runs, shapes).drop(columns=["m", "nnz"]))


# ------------------------------------------------------------------------------------------------------ F12
def _with_big_family(seed: int = 0, k: int = 30):
    """fake_inputs plus one MIPLIB family of k optimisation-hard instances (the shape of `neos`, 55 candidates)."""
    cands, runs, shapes = fake_inputs(seed)
    rs = np.random.default_rng(1000 + seed)
    names = [f"neos-{100000 + j}" for j in range(k)]
    big = H.assign_groups(pd.DataFrame(dict(instance=names, set="miplib", path=[f"data/{x}.mps.gz" for x in names],
                                            miplib_group="–")))
    rows = [dict(instance=x, set="miplib", arm="SCREEN", seed=sd, budget=300.0, pt=float(rs.uniform(0.06, 0.95)),
                 t_first=5.0, t_first_event=True, crashed=False, solved_optimal=False, ref_zero_instance=False,
                 sign_crossing=False, no_exchange_window=False, synthetic=False) for x in names for sd in (900, 901)]
    return (pd.concat([cands, big], ignore_index=True), pd.concat([runs, pd.DataFrame(rows)], ignore_index=True),
            pd.concat([shapes, pd.DataFrame(dict(instance=names, n=1000, m=500, nnz=4000, gene_bytes=1))],
                      ignore_index=True))


@pytest.mark.parametrize("seed", [0, 1])
def test_tuning_side_family_cap(seed):
    cands, runs, shapes = _with_big_family(seed)
    table = _table(cands, runs, shapes)
    loose, _ = H.select_pool(table, tuning_family_cap=100)               # the pre-fix rule: test-side cap only
    assert (loose[loose["side"] == "tuning"]["family"] == "miplib:neos").sum() > 3
    pool, rep = H.select_pool(table)
    assert rep["tuning_family_cap"] == 3 and rep["family_cap"] == 6
    assert not rep["shortfall"]
    _check_pool(pool)
    tun = pool[pool["side"] == "tuning"]
    assert tun.groupby("family").size().max() <= 3
    assert rep["family_counts"]["tuning"] == tun.groupby("family").size().to_dict()
    assert rep["family_counts"]["test"] == rep["family_test_counts"] == \
        pool[pool["side"] == "test"].groupby("family").size().to_dict()


def test_tuning_cap_does_not_narrow_the_ca_tuning_supply():
    """The tuning cap (3) equals the tuning PGLib quota, so it cannot bind for ca: phase 1 must not stop sending ca
    days to tuning after the first (which would leave tuning PGLib = three reserve variants of one day)."""
    cands, runs, shapes = fake_inputs(0)
    _, _, rep = H.freeze(runs, cands=cands, shapes=shapes)
    assert sum(1 for g, s in rep["sides"].items() if g.startswith("pglib:ca__") and s == "tuning") >= 2


# ------------------------------------------------------------------------------------------------------ F13
def test_solver_limit_stops_are_flagged_not_hidden(tmp_path):
    cands, runs, shapes = fake_inputs()
    runs = runs.assign(solver_exit_reason="budget")
    pool0, _, rep0 = H.freeze(runs, cands=cands, shapes=shapes)
    assert not pool0["solver_limit_stop"].any() and rep0["solver_limit_stops"] == []
    two = pool0[pool0["n_seeds"] == 2]["instance"].tolist()
    mem, node = two[0], two[-1]
    runs2 = runs.copy()
    runs2.loc[(runs2["instance"] == mem) & (runs2["seed"] == 901), "solver_exit_reason"] = "memory_limit"
    runs2.loc[(runs2["instance"] == node) & (runs2["seed"] == 900), "solver_exit_reason"] = "node_limit"
    pool, table, rep = H.freeze(runs2, cands=cands, shapes=shapes)
    assert pool["instance"].tolist() == pool0["instance"].tolist()        # flagged, not excluded
    p = pool.set_index("instance")
    assert p.loc[mem, "solver_limit_stop"] and p.loc[mem, "exit_s901"] == "memory_limit"
    assert p.loc[mem, "exit_s900"] == "budget"
    assert p.loc[node, "solver_limit_stop"] and p.loc[node, "exit_s900"] == "node_limit"
    assert int(p["solver_limit_stop"].sum()) == 2
    listed = {r["instance"]: r for r in rep["solver_limit_stops"]}
    assert set(listed) == {mem, node} and listed[mem]["exit_s901"] == "memory_limit"
    assert listed[mem]["side"] == p.loc[mem, "side"]
    H.write_pool(pool, table, tmp_path / "pool.csv", tmp_path / "cands.csv")
    back = pd.read_csv(tmp_path / "pool.csv").set_index("instance")
    assert bool(back.loc[mem, "solver_limit_stop"]) and back.loc[mem, "exit_s901"] == "memory_limit"
    assert "solver_limit_stop" in pd.read_csv(tmp_path / "cands.csv").columns
    # runs without the column (older runs.parquet): empty reasons, no flag
    t = H.screening_table(runs.drop(columns=["solver_exit_reason"]))
    assert (t["exit_s900"] == "").all() and not t["solver_limit_stop"].any()


def test_p_degenerate_kept_out_of_p_strata_and_pglib_always_stage2():
    """D-002 v3.1 §2: P-degenerate instances never enter a P-scored stratum (feasibility stratum still allowed);
    PGLib/ML4CO need seed 901 regardless of their stage-1 P."""
    import pandas as pd
    from itrea.benchmarks import hardset as H
    cands = pd.DataFrame(dict(instance=["m_deg", "m_ok", "m_feas_deg", "u_low"], set=["miplib", "miplib", "miplib",
                                                                                      "pglib_uc"],
                              path=["a", "b", "c", "d"]))
    screen = pd.DataFrame(dict(
        instance=["m_deg", "m_ok", "m_feas_deg", "u_low"], set="x", has_s900=True, has_s901=[True, True, True, False],
        crashed=False, p_s900=[0.5, 0.5, 1.0, 0.01], feasible_s900=[True, True, False, True],
        n_seeds=[2, 2, 2, 1], feasible_seeds=[2, 2, 1, 1], p_mean=[0.5, 0.5, 1.0, 0.01],
        ref_zero=[True, False, True, False], sign_crossing=False))
    df = H.classify(cands, screen).set_index("instance")
    assert df.loc["m_deg", "stratum"] == "" and "P-degenerate" in df.loc["m_deg", "excluded_reason"]
    assert df.loc["m_ok", "stratum"] == "miplib_opt"
    assert df.loc["m_feas_deg", "stratum"] == "miplib_feas"            # feasibility stratum may take it
    assert df.loc["u_low", "stratum"] == "" and "stage 2" in df.loc["u_low", "excluded_reason"]
