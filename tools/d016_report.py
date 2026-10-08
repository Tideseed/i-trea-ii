"""D-016 report: Rank 1 by cuOpt's clearing rule (A3@cl1) vs the development default (A3@rc1d) and the size-only
control (A3@tp1), with the decision rules and the mechanism check fixed in the decision record before the runs.

Usage (repo root, after `itrea build-runs`):  OMP_NUM_THREADS=1 .venv/bin/python tools/d016_report.py
Reads datasets/runs.parquet (P(120) = `pt`) and every run's ea.summary.json / ea.events.jsonl (mechanism fields).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.analysis.stats import bootstrap_mean, instance_effects, wilcoxon_d

ARMS = {"ctrl": "A3@rc1d", "cl1": "A3@cl1", "tp1": "A3@tp1"}
BANDS = {"ca__Scenario400_reserves_1": 0.10, "fhnw-binpack4-48": 0.02, "glass4": 0.37, "item_placement_9903": 0.68,
         "neos-3024952-loue": 0.13, "neos-960392": 0.04, "radiationm18-12-05": 0.46, "rail01": 0.02, "sing44": 0.11}


def last_event(run_dir: Path, kind: str) -> dict | None:
    last = None
    p = run_dir / "ea.events.jsonl"
    if not p.exists():
        return None
    with open(p) as f:
        for line in f:
            if f'"kind": "{kind}"' in line:
                last = line
    return json.loads(last) if last else None


def mech_row(run_dir: Path) -> dict:
    s = json.loads((run_dir / "ea.summary.json").read_text())
    fr = last_event(run_dir, "frontier") or {}
    return dict(feasible=bool(s.get("feasible_found")), r1_n=s.get("final_r1_n"),
                r1_dmax_band=s.get("final_r1_dmax_band") or 0.0, r1_dmax=s.get("final_r1_dmax") or 0.0,
                r1_gap=s.get("final_r1_gap"), cleared=s.get("rank1_cleared"), frontier_max=fr.get("max_dist", 0.0) or 0.0,
                generations=s.get("generations"), compiled_on_clock=s.get("compiled_on_clock"))


def compare(df: pd.DataFrame, arm: str, ctrl: str) -> tuple[pd.DataFrame, dict]:
    eff = instance_effects(df, arm, ctrl)
    d = eff["d"].to_numpy()
    return eff, dict(n=len(d), better=int((d > 0).sum()), worse=int((d < 0).sum()), **{
        f"boot_{k}": v for k, v in bootstrap_mean(d).items() if k in ("mean", "lo", "hi")}, wilcoxon_p=wilcoxon_d(d)["p"])


def main() -> int:
    df = pd.read_parquet("datasets/runs.parquet")
    df = df[(df["set"] == "dev") & (df["arm"].isin(ARMS.values())) & (df["status"] == "completed")].copy()
    if df.empty:
        print("no D-016 runs in datasets/runs.parquet")
        return 2
    print(f"runs: {len(df)} completed ({df.groupby('arm').size().to_dict()})")
    mech = pd.DataFrame([dict(instance=r.instance, seed=r.seed, arm=r.arm, **mech_row(Path(r.run_dir)))
                         for r in df.itertuples()])
    print(f"compiled_on_clock true: {int(mech['compiled_on_clock'].fillna(False).astype(bool).sum())}")

    print("\n== P(120) per instance (seed means) ==")
    piv = df.pivot_table(index="instance", columns="arm", values="pt", aggfunc="mean")[list(ARMS.values())]
    print(piv.round(3).to_string())

    out = {}
    for arm, ctrl in (("A3@cl1", "A3@rc1d"), ("A3@tp1", "A3@rc1d"), ("A3@cl1", "A3@tp1")):
        eff, summ = compare(df, arm, ctrl)
        out[(arm, ctrl)] = (eff, summ)
        print(f"\n== {arm} vs {ctrl}: better/worse {summ['better']}/{summ['worse']} of {summ['n']}, mean d "
              f"{summ['boot_mean']:+.3f} [{summ['boot_lo']:+.3f}, {summ['boot_hi']:+.3f}], Wilcoxon p {summ['wilcoxon_p']:.3f}")
        for r in eff.itertuples():
            band = BANDS.get(r.instance, 0.02)
            flag = "  WORSE beyond band" if r.d < -band else ("  better beyond band" if r.d > band else "")
            print(f"   {r.instance[:28]:28s} P {r.p_ctrl:.3f} -> {r.p_arm:.3f}  d {r.d:+.3f}  band {band:.2f}{flag}")

    # decision (D-016): cl1 vs rc1d
    eff, summ = out[("A3@cl1", "A3@rc1d")]
    seeds = df.pivot_table(index=["instance", "seed"], columns="arm", values="pt", aggfunc="first")
    worse_band = [r.instance for r in eff.itertuples() if r.d < -BANDS.get(r.instance, 0.02)]
    worse_both = []
    for inst in worse_band:
        s = seeds.loc[inst].dropna(subset=["A3@cl1", "A3@rc1d"])
        dd = [(c - a) / max(c, a, 1e-3) for a, c in zip(s["A3@cl1"], s["A3@rc1d"])]
        if all(x < -BANDS.get(inst, 0.02) for x in dd):
            worse_both.append(inst)
    m = summ["boot_mean"]
    if m >= 0.05 and not worse_band:
        verdict = "ADOPT as development default"
    elif m <= -0.05 or worse_both:
        verdict = "REJECT"
    else:
        verdict = "INCONCLUSIVE -> seeds 2-3 on the instances inside their band"
    print(f"\nD-016 outcome rule: mean d {m:+.3f}; worse beyond band: {worse_band or 'none'}; on both seeds: "
          f"{worse_both or 'none'} -> {verdict}")

    # mechanism (amended D-016)
    print("\n== mechanism (at T, per instance: seed means) ==")
    g = mech.groupby(["instance", "arm"]).agg(feasible=("feasible", "all"), r1_n=("r1_n", "mean"),
                                               r1_dmax_band=("r1_dmax_band", "mean"), r1_dmax=("r1_dmax", "mean"),
                                               frontier_max=("frontier_max", "mean"), cleared=("cleared", "mean"),
                                               gens=("generations", "mean")).reset_index()
    print(g.round(4).to_string(index=False))
    both = [i for i in g["instance"].unique()
            if g[(g.instance == i) & (g.arm.isin(["A3@cl1", "A3@rc1d"]))]["feasible"].all()
            and len(g[(g.instance == i) & (g.arm.isin(["A3@cl1", "A3@rc1d"]))]) == 2]
    sel = g[g.instance.isin(both)]
    med = lambda arm, col: float(np.median(sel[sel.arm == arm][col].to_numpy(dtype=float)))  # noqa: E731
    mc, mr = med("A3@cl1", "r1_dmax_band"), med("A3@rc1d", "r1_dmax_band")
    nc, nr = med("A3@cl1", "r1_n"), med("A3@rc1d", "r1_n")
    cleared_ok = bool((sel[sel.arm == "A3@cl1"]["cleared"].fillna(0) > 0).all())
    band_ok = (mc >= 5 * mr) if mr > 0 else (mc > 0)
    print(f"\ninstances feasible in both arms: {len(both)} {both}")
    print(f"median r1_n: cl1 {nc:.1f} vs rc1d {nr:.1f} ({'>= 3x' if nc >= 3 * nr else '< 3x'})")
    print(f"median r1_dmax_band: cl1 {mc:.4f} vs rc1d {mr:.4f} ({'pass' if band_ok else 'FAIL'})")
    print(f"rank1_cleared > 0 on every one: {cleared_ok}")
    print(f"MECHANISM: {'HOLDS' if (band_ok and cleared_ok and nc >= 3 * nr) else 'DOES NOT HOLD'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
