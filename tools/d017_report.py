"""D-017 report: LP decode (A3@ld1 = control + repair.continuous: lp_decode) vs the development default re-run in the
same window (A3@rc1e), with the mechanism M0–M4 and the outcome rules fixed in the decision record before the runs.

Usage (repo root, after `itrea build-runs`):  OMP_NUM_THREADS=1 .venv/bin/python tools/d017_report.py
Reads datasets/runs.parquet (P(120) = `pt`) and every run's ea.summary.json / ea.events.jsonl.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.analysis.stats import bootstrap_mean, instance_effects, wilcoxon_d

CTRL, ARM = "A3@rc1e", "A3@ld1"
BANDS = {"ca__Scenario400_reserves_1": 0.10, "fhnw-binpack4-48": 0.02, "glass4": 0.37, "item_placement_9903": 0.68,
         "radiationm18-12-05": 0.46, "sing44": 0.11}


def events(run_dir: Path, kind: str) -> list[dict]:
    out = []
    p = run_dir / "ea.events.jsonl"
    if p.exists():
        with open(p) as f:
            for line in f:
                if f'"kind": "{kind}"' in line:
                    out.append(json.loads(line))
    return out


def mech(run_dir: Path) -> dict:
    s = json.loads((run_dir / "ea.summary.json").read_text())
    ld = s.get("lp_decode") or {}
    w = ld.get("worker") or {}
    res = [e for e in events(run_dir, "lp_decode") if e.get("what") == "result"]
    rows = [r for e in res for r in (e.get("rows") or [])]
    solved = [r for r in rows if r.get("code") in ("ok", "infeasible", "time_limit", "other")]
    warm = [r for r in solved if not r.get("cold")]
    warm_hits = [r for r in warm if r.get("status") == "Time limit reached"]
    wall = float(s.get("wall_s") or 120.0)
    return dict(
        compiled=bool(s.get("compiled_on_clock")), state=ld.get("state"), failed=int(s.get("lpd_failed") or 0),
        ready_w=ld.get("ready_t_worker"), host_s=ld.get("host_s"), host_max=ld.get("host_s_max"),
        sync_s=ld.get("sync_s"), host_share=(ld.get("host_s") or 0.0) / max(wall, 1e-9),
        warm_solves_ev=len(warm), warm_hits_ev=len(warm_hits), warm_solves=w.get("warm_solves"),
        warm_hits=w.get("warm_time_limit_hits"), priming_first=w.get("priming_first_status"),
        offers=s.get("lpd_offers"), skip=s.get("lpd_skip_pending"), kept=s.get("lpd_kept"),
        entered_F=s.get("lpd_entered_F"), entered_I=s.get("lpd_entered_I"),
        inc_F=s.get("lpd_incumbents_F"), inc_I=s.get("lpd_incumbents_I"), posts=s.get("lpd_posts"),
        rows_I=s.get("lpd_rows_I"), screened_I=s.get("lpd_screened_I"), ok_I=s.get("lpd_ok_I"),
        gain_F=ld.get("gain_F_entered_median"), gain_F_n=ld.get("gain_F_entered_n"),
        gain_I=ld.get("gain_I_entered_median"), lost=s.get("lpd_lost"), dropped=s.get("lpd_dropped"),
        P=s.get("P"), gene_dtype=s.get("gene_dtype"), gens=s.get("generations"))


def main() -> int:
    df = pd.read_parquet("datasets/runs.parquet")
    df = df[(df["set"] == "dev") & df["arm"].isin([CTRL, ARM]) & (df["status"] == "completed")].copy()
    if df.empty:
        print("no D-017 runs")
        return 2
    print(f"runs: {df.groupby('arm').size().to_dict()}")
    m = pd.DataFrame([dict(instance=r.instance, seed=r.seed, arm=r.arm, **mech(Path(r.run_dir)))
                      for r in df.itertuples()])
    t = m[m.arm == ARM]
    c = m[m.arm == CTRL]

    # M0 validity
    m0 = []
    for (inst, seed), g in m.groupby(["instance", "seed"]):
        a, b = g[g.arm == ARM], g[g.arm == CTRL]
        if len(a) != 1 or len(b) != 1:
            m0.append((inst, seed, "missing arm"))
            continue
        a, b = a.iloc[0], b.iloc[0]
        if a.P != b.P or a.gene_dtype != b.gene_dtype:
            m0.append((inst, seed, f"P {a.P}/{b.P} dtype {a.gene_dtype}/{b.gene_dtype}"))
        if a.compiled or b.compiled:
            m0.append((inst, seed, "compiled_on_clock"))
        if a.failed or a.state == "failed":
            m0.append((inst, seed, "worker failed"))
        if a.ready_w is None or not (a.ready_w <= 10.0):
            m0.append((inst, seed, f"ready_t_worker {a.ready_w}"))
    print(f"\nM0 validity: {'PASS' if not m0 else 'FAIL ' + str(m0)}")
    # M1
    m1 = t[(t.host_share > 0.01) | (t.host_max.fillna(0) > 0.050)]
    print(f"M1 GPU never waits (host ≤ 1 % of wall, max ≤ 50 ms): {'PASS' if m1.empty else 'FAIL'}; "
          f"host share max {t.host_share.max():.4f}, host_max max {t.host_max.max():.4f} s, sync_s max {t.sync_s.max()}")
    # M2 per instance (seeds pooled), from the result events' rows
    m2 = []
    for inst, g in t.groupby("instance"):
        ws, wh = g.warm_solves_ev.sum(), g.warm_hits_ev.sum()
        share = wh / ws if ws else 0.0
        if share > 0.10:
            m2.append((inst, f"{wh}/{ws}"))
    ca = t[t.instance.str.startswith("ca__")]
    ca_ok = bool((ca.priming_first == "Optimal").all()) if len(ca) else None
    print(f"M2 worker sane (warm hits ≤ 10 %; ca priming Optimal): {'PASS' if (not m2 and ca_ok is not False) else 'FAIL'} "
          f"{m2} ca priming {list(ca.priming_first)}")
    # M3
    def inst_rows(name):
        return t[t.instance == name]
    s44, cag = inst_rows("sing44"), inst_rows("ca__Scenario400_reserves_1")
    m3_s = bool(len(s44)) and bool((s44.gain_F.fillna(0) >= 1e-2).all()) and bool(((s44.inc_F.fillna(0) + s44.inc_I.fillna(0)) >= 1).all())
    m3_c = bool(len(cag)) and bool((cag.gain_F.fillna(0) >= 1e-3).all()) and bool(((cag.inc_F.fillna(0) + cag.inc_I.fillna(0)) >= 1).any())
    print(f"M3 polishing acts: sing44 {'PASS' if m3_s else 'FAIL'} (gainF {list(s44.gain_F.round(4))}, decoded incumbents "
          f"{list((s44.inc_F.fillna(0)+s44.inc_I.fillna(0)).astype(int))}); ca {'PASS' if m3_c else 'FAIL'} (gainF "
          f"{list(cag.gain_F.round(5))}, decoded incumbents {list((cag.inc_F.fillna(0)+cag.inc_I.fillna(0)).astype(int))})")
    # M4 completion funnel (reported)
    print("M4 completion funnel per instance (I rows offered / screened / LP ok / entered / incumbents):")
    for inst, g in t.groupby("instance"):
        print(f"   {inst[:28]:28s} {int(g.rows_I.fillna(0).sum())} / {int(g.screened_I.fillna(0).sum())} / "
              f"{int(g.ok_I.fillna(0).sum())} / {int(g.entered_I.fillna(0).sum())} / {int(g.inc_I.fillna(0).sum())}")
    any_I_ok = int(t.ok_I.fillna(0).sum()) > 0
    print(f"   completion LP feasible anywhere: {any_I_ok}")
    mech_ok = not m0 and m1.empty and not m2 and ca_ok is not False and m3_s and m3_c
    print(f"MECHANISM: {'HOLDS' if mech_ok else 'DOES NOT HOLD'}")

    # outcome
    eff = instance_effects(df, ARM, CTRL)
    d = eff["d"].to_numpy()
    bm = bootstrap_mean(d)
    print(f"\n== {ARM} vs {CTRL}: better/worse {(d > 0).sum()}/{(d < 0).sum()} of {len(d)}, mean d {bm['mean']:+.3f} "
          f"[{bm['lo']:+.3f}, {bm['hi']:+.3f}], Wilcoxon p {wilcoxon_d(d)['p']:.3f}")
    seeds = df.pivot_table(index=["instance", "seed"], columns="arm", values="pt", aggfunc="first")
    worse_band, worse_both = [], []
    for r in eff.itertuples():
        band = BANDS.get(r.instance, 0.02)
        flag = "  WORSE beyond band" if r.d < -band else ("  better beyond band" if r.d > band else "")
        g = m[(m.instance == r.instance) & (m.arm == ARM)]
        print(f"   {r.instance[:28]:28s} P {r.p_ctrl:.3f} -> {r.p_arm:.3f}  d {r.d:+.3f}  band {band:.2f}{flag}   "
              f"gens ctrl {c[c.instance == r.instance].gens.mean():.0f} / ld1 {g.gens.mean():.0f}")
        if r.d < -band:
            worse_band.append(r.instance)
            s = seeds.loc[r.instance].dropna(subset=[ARM, CTRL])
            dd = [(cc - aa) / max(cc, aa, 1e-3) for aa, cc in zip(s[ARM], s[CTRL])]
            if all(x < -band for x in dd):
                worse_both.append(r.instance)
    mean = bm["mean"]
    if mean >= 0.05 and not worse_band:
        verdict = "ADOPT as development default"
    elif mean <= -0.05 or worse_both:
        verdict = "REJECT"
    else:
        verdict = "INCONCLUSIVE -> seeds 2-3 on the in-band instances where the mechanism acted"
    mean9 = float(np.sum(d)) / 9.0
    print(f"\nD-017 outcome rule: mean d {mean:+.3f}; worse beyond band {worse_band or 'none'}; on both seeds "
          f"{worse_both or 'none'} -> {verdict}{'' if mech_ok else '  (mechanism does not hold: not read as lp_decode)'}")
    print(f"9-instance mean (d = 0 on the 3 inert instances): {mean9:+.3f}")
    print("\nper run:")
    print(t[["instance", "seed", "gens", "offers", "skip", "kept", "entered_F", "entered_I", "inc_F", "inc_I", "gain_F",
             "host_max", "ready_w", "lost", "dropped"]].sort_values(["instance", "seed"]).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
