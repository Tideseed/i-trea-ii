"""D-029's operating characteristic: the pass rates its rules imply under a normal model fitted to the development
default's own same-window control pairs (run hash 28e5a5e9c725, 120 s, harness ext). The numbers the record's "Why these
numbers" section and its 2026-10-15 amendment quote come from this script. Descriptive only: a seeded simulation with no
solver and no GPU.

Usage (repository root):

    OMP_NUM_THREADS=1 .venv/bin/python tools/d029_oc.py [--draws 200000] [--check]

`--check` compares the data below with `datasets/runs.parquet` and the runs' EA summaries (it is skipped when the table
is missing).

Model (the builder's and the review's, made reproducible):
  - "within": the two runs of a block share their window's mean (D-029's design); each window draws its mean from
    N(mu, sb), sb = sqrt(max(st^2 - sw^2, 0)), and each run adds N(0, sw);
  - "total": every run is independent at the all-runs SD st (wider);
  - st, sw and mu come from the control pairs below; "3 pairs" is the record's noise and "5 pairs" adds backoff2's two
    same-window ext pairs on item_placement (A9@bo1s, the default with the back-off only recording, and A9@boY) to the
    within-window SD (its st is max(st, sw));
  - seeds 0–1 and seeds 2–5 are two windows. "fresh" draws both; "given" fixes seeds 0–1 at r1-ysync's observed values;
  - an effect multiplies T's generations (G) or adds to T's P; a "collapse" multiplies one T run's G by 0.46 and adds
    +0.15 to its P (item seed 1's size in r1-ysync; a construction, not a known harm);
  - the four guard legs (rules 1 and 2 on item_placement and comp21) are drawn independently: G and P independent and
    the two instances independent. Rule 3 (ca) is left out of the joint rows and reported on its own line.
D-025 (for the two-stage rows) is approximated as the review did: per instance, the seed-mean generation ratio >= 1 -
max(0.05, the control's relative seed spread) and the P difference <= max(0.05, the control's P spread), on seeds 0–1;
the hint and the hint-instance bands are taken as passing.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

# ----------------------------------------------------------------------------------------------- the data
# Same-window control pairs (seed 0, seed 1) of the development default's configuration, 120 s, ext:
# itx-r13 A9@r13x (10-13), r1-bsync A9@r1ctl (10-15), r1-ysync A9@r1ctl-ysync (10-15).
PAIRS3 = {
    "itemG": [(1619, 1687), (1646, 1782), (1367, 1480)],
    "compG": [(248, 295), (261, 272), (287, 318)],
    "itemP": [(0.3575, 0.3700), (0.2994, 0.4241), (0.4791, 0.2495)],
    "compP": [(0.5519, 0.5434), (0.5418, 0.5028), (0.5952, 0.5533)],
}
# backoff2 (10-14, ext), item_placement only: A9@bo1s and A9@boY.
EXTRA_ITEM = {"itemG": [(1562, 1946), (1950, 1754)], "itemP": [(0.2339, 0.2023), (0.3315, 0.2839)]}
# r1-ysync's observed seeds 0–1: (T seed 0, T seed 1), (C seed 0, C seed 1).
OBS = {
    "itemG": ((1483, 678), (1367, 1480)), "compG": ((299, 294), (287, 318)),
    "itemP": ((0.4082, 0.5175), (0.4791, 0.2495)), "compP": ((0.5840, 0.5625), (0.5952, 0.5533)),
}
# ca P(120): control (the default's configuration) pairs and ysync's two runs.
CA_C_PAIRS = [(0.3912, 0.3959), (0.3912, 0.3938), (0.3919, 0.3928)]
CA_T_OBS = (0.2492, 0.2564)
CA_C_OBS = (0.3919, 0.3928)
G_MIN, P_MAX, CA_MAX = 0.90, 0.05, 0.95
COLLAPSE_G, COLLAPSE_P = 0.46, 0.15


def noise(pairs: list[tuple]) -> tuple[float, float, float]:
    """(mu, sw, st): the mean of all runs, the pooled within-pair SD, the SD of all runs (ddof 1)."""
    v = np.array(pairs, float)
    sw = math.sqrt(float(np.mean((v[:, 0] - v[:, 1]) ** 2 / 2)))
    return float(v.mean()), sw, float(v.ravel().std(ddof=1))


def noise_sets() -> dict[str, dict[str, tuple[float, float, float]]]:
    n3 = {k: noise(v) for k, v in PAIRS3.items()}
    n5 = dict(n3)
    for k, extra in EXTRA_ITEM.items():
        mu, _, st = n3[k]
        _, sw5, _ = noise(PAIRS3[k] + extra)
        n5[k] = (mu, sw5, max(st, sw5))
    return {"3 pairs": n3, "5 pairs": n5}


# ----------------------------------------------------------------------------------------------- the simulation
class Sim:
    def __init__(self, par: dict, model: str, n: int, rng: np.random.Generator):
        self.par, self.model, self.n, self.rng = par, model, n, rng

    def window(self, key: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """T and C draws (n x k) of k seeds in one window, no effect."""
        mu, sw, st = self.par[key]
        if self.model == "within":
            sb = math.sqrt(max(st ** 2 - sw ** 2, 0.0))
            m, s = mu + sb * self.rng.standard_normal((self.n, 1)), sw
        else:
            m, s = np.full((self.n, 1), mu), st
        return m + s * self.rng.standard_normal((self.n, k)), m + s * self.rng.standard_normal((self.n, k))

    def data(self, given: bool, g: float = 0.0, p: float = 0.0, q: float = 0.0, one_collapse: str | None = None,
             g_inst: dict | None = None) -> dict:
        """{inst: (TG, CG, TP, CP)} over seeds 0–5 (columns), seeds 0–1 one window, 2–5 another."""
        out = {}
        for inst in ("item", "comp"):
            arr = {}
            for what in ("G", "P"):
                t01, c01 = self.window(inst + what, 2)
                t25, c25 = self.window(inst + what, 4)
                arr[what] = [np.hstack([t01, t25]), np.hstack([c01, c25])]
            TG, CG = arr["G"]
            TP, CP = arr["P"]
            ge = (g_inst or {}).get(inst, g)
            TG, TP = TG * (1.0 + ge), TP + p
            lo = 2 if given else 0
            if q and inst == "item":
                hit = self.rng.random((self.n, 6)) < q
                hit[:, :lo] = False
                TG, TP = np.where(hit, TG * COLLAPSE_G, TG), np.where(hit, TP + COLLAPSE_P, TP)
            if one_collapse == inst:
                TG[:, 2] *= COLLAPSE_G
            if given:
                for a, key, j in ((TG, "G", 0), (CG, "G", 1), (TP, "P", 0), (CP, "P", 1)):
                    a[:, :2] = OBS[inst + key][j]
            out[inst] = (TG, CG, TP, CP)
        return out


def rule1(d, inst, thr=G_MIN):
    TG, CG, _, _ = d[inst]
    return TG.mean(1) / CG.mean(1) >= thr


def rule2(d, inst, thr=P_MAX):
    _, _, TP, CP = d[inst]
    return TP.mean(1) - CP.mean(1) <= thr


def joint(d, thr_g=G_MIN, thr_p=P_MAX):
    ok = np.ones(next(iter(d.values()))[0].shape[0], bool)
    for inst in d:
        ok &= rule1(d, inst, thr_g) & rule2(d, inst, thr_p)
    return ok


def d025(d):
    ok = np.ones(next(iter(d.values()))[0].shape[0], bool)
    for TG, CG, TP, CP in d.values():
        mc = CG[:, :2].mean(1)
        thr = 1 - np.maximum(0.05, np.abs(CG[:, 0] - CG[:, 1]) / mc)
        ok &= (TG[:, :2].mean(1) / mc >= thr) & \
              ((TP[:, :2].mean(1) - CP[:, :2].mean(1)) <= np.maximum(0.05, np.abs(CP[:, 0] - CP[:, 1])))
    return ok


def rate(x) -> str:
    return f"{100 * float(np.mean(x)):.1f}"


# ----------------------------------------------------------------------------------------------- output
def run(draws: int) -> None:
    sets = noise_sets()
    for name, par in sets.items():
        print(f"noise {name}: " + "; ".join(f"{k} mu {v[0]:.4g} sw {v[1]:.3g} st {v[2]:.3g}" for k, v in par.items()))
    combos = [(ns, m) for ns in sets for m in ("within", "total")]
    head = " | ".join(f"{ns[0]}p {m[0]}" for ns, m in combos)

    def table(title, rows):
        print(f"\n{title}  (% pass; columns: fresh [{head}] || given seeds 0–1 [{head}])")
        for lab, kw, test in rows:
            cols = []
            for given in (False, True):
                vals = []
                for i, (ns, m) in enumerate(combos):
                    rng = np.random.default_rng(29 + i + 100 * given)
                    vals.append(rate(test(Sim(sets[ns], m, draws, rng).data(given, **kw))))
                cols.append(" | ".join(vals))
            print(f"  {lab:<44} {cols[0]}  ||  {cols[1]}")

    r1i, r1c = (lambda d: rule1(d, "item")), (lambda d: rule1(d, "comp"))
    r2i, r2c = (lambda d: rule2(d, "item")), (lambda d: rule2(d, "comp"))
    table("Rule 1 (item_placement)", [
        ("none", {}, r1i), ("G -10 %", dict(g=-0.10), r1i), ("G -15 %", dict(g=-0.15), r1i),
        ("G -18 % (bsync's)", dict(g=-0.18), r1i), ("one x0.46 collapse among seeds 2–5", dict(one_collapse="item"), r1i)])
    table("Rule 1 (comp21)", [
        ("none", {}, r1c), ("G -10 %", dict(g=-0.10), r1c), ("G -15 %", dict(g=-0.15), r1c),
        ("G -18 %", dict(g=-0.18), r1c), ("one x0.46 collapse among seeds 2–5", dict(one_collapse="comp"), r1c)])
    table("Rule 2 (item_placement)", [
        ("none", {}, r2i), ("P +0.05", dict(p=0.05), r2i), ("P +0.10", dict(p=0.10), r2i)])
    table("Rule 2 (comp21)", [
        ("none", {}, r2c), ("P +0.05", dict(p=0.05), r2c), ("P +0.10", dict(p=0.10), r2c)])
    table("Rules 1 + 2 on both (four legs, independent)", [
        ("none", {}, joint), ("G -10 % on both", dict(g=-0.10), joint), ("G -15 % on both", dict(g=-0.15), joint),
        ("P +0.05 on both", dict(p=0.05), joint), ("bsync's (G -18 %, P +0.06)", dict(g=-0.18, p=0.06), joint),
        ("item collapse at rate q = 1/6", dict(q=1 / 6), joint), ("item collapse at rate q = 1/4", dict(q=0.25), joint)])

    print("\nRule 1 on item_placement against its threshold, given seeds 0–1 (% pass; 3 pairs within | total)")
    for thr in (0.90, 0.921, 0.93, 0.95):
        out = []
        for g in (0.0, -0.10, -0.15):
            v = [rate(rule1(Sim(sets["3 pairs"], m, draws, np.random.default_rng(7 + i)).data(True, g=g), "item", thr))
                 for i, m in enumerate(("within", "total"))]
            out.append(f"G {int(round(100 * g)):+d} %: {' | '.join(v)}")
        fresh = [rate(rule1(Sim(sets["3 pairs"], m, draws, np.random.default_rng(17 + i)).data(False), inst, thr))
                 for inst in ("item", "comp") for i, m in enumerate(("within", "total"))]
        both = [rate(joint(Sim(sets["3 pairs"], m, draws, np.random.default_rng(27 + i)).data(True), thr))
                for i, m in enumerate(("within", "total"))]
        print(f"  {thr:.3f}: given {'; '.join(out)};  fresh none item {' | '.join(fresh[:2])}, comp21 "
              f"{' | '.join(fresh[2:])};  rules 1 + 2 on both, given, none {' | '.join(both)}")

    print("\nTwo stages: D-025 alone (seeds 0–1) and D-025 then D-029 (adopted at D-025, or failed there and passed "
          "here on the same seeds 0–1 plus 2–5) (% adopted; [3p within | 3p total | 5p within | 5p total])")
    for lab, kw in (("none", {}), ("G -10 % on both", dict(g=-0.10)), ("G -15 % on both", dict(g=-0.15)),
                    ("P +0.05 on both", dict(p=0.05)), ("item collapse q = 1/6", dict(q=1 / 6)),
                    ("item collapse q = 1/4", dict(q=0.25))):
        a_, b_, c_ = [], [], []
        for i, (ns, m) in enumerate(combos):
            d = Sim(sets[ns], m, draws, np.random.default_rng(53 + i)).data(False, **kw)
            a, b = d025(d), joint(d)
            a_.append(rate(a)), b_.append(rate(a | b)), c_.append(rate(b[~a]) if (~a).any() else "-")
        print(f"  {lab:<24} D-025 alone {' | '.join(a_)};  then D-029 {' | '.join(b_)};  D-029 after a D-025 "
              f"failure {' | '.join(c_)}")

    print("\nWhat seeds 2–5 must show, given seeds 0–1 (the control at its 3-pair mean):")
    for inst in ("item", "comp"):
        (t01, c01), mu = OBS[inst + "G"], noise(PAIRS3[inst + "G"])[0]
        need = (G_MIN * (sum(c01) + 4 * mu) - sum(t01)) / (4 * mu)
        dp = sum(a - b for a, b in zip(*OBS[inst + "P"]))
        used = (sum(c01) - sum(t01)) / ((1 - G_MIN) * (sum(c01) + 4 * mu))
        print(f"  {inst}: G ratio at seeds 2–5 >= {need:.3f}; seeds 0–1 use {100 * used:.1f} % of rule 1's margin "
              f"(seed 1 alone {100 * (c01[1] - t01[1]) / ((1 - G_MIN) * (sum(c01) + 4 * mu)):.1f} %); mean P "
              f"difference at seeds 2–5 <= {(6 * P_MAX - dp) / 4:+.4f} (seeds 0–1 sum {dp:+.4f})")
        for c in (1424.0, 1597.0, 1700.0) if inst == "item" else ():
            print(f"    with the new window's control at {c:.0f}: {(G_MIN * (sum(c01) + 4 * c) - sum(t01)) / (4 * c):.3f}")
    t01, c01 = OBS["compG"]
    mu = noise(PAIRS3["compG"])[0]
    one = (sum(t01) + 3 * mu + COLLAPSE_G * mu) / (sum(c01) + 4 * mu)
    two = (sum(t01) + 2 * mu + 0.97 * mu + COLLAPSE_G * mu) / (sum(c01) + 4 * mu)
    print(f"  comp21: one x0.46 collapse at seeds 2–5, the rest at parity -> pooled ratio {one:.4f}; with one more seed "
          f"at 0.97 -> {two:.4f}")
    for inst in ("item",):
        sw = noise(PAIRS3[inst + "G"])[1]
        st = noise(PAIRS3[inst + "G"])[2]
        (t01, c01) = OBS[inst + "G"]
        print(f"  item seed 1 ({t01[1]} vs {c01[1]}) under no effect: z {(t01[1] - c01[1]) / (math.sqrt(2) * sw):.1f} "
              f"(within) / {(t01[1] - c01[1]) / (math.sqrt(2) * st):.1f} (total)")

    print("\nRule 3 (ca, seeds 2–3 alone: sum P(T) / sum P(C) <= 0.95):")
    mu_c, sw_c, st_c = noise(CA_C_PAIRS)
    s = max(sw_c, st_c)
    rng = np.random.default_rng(31)
    cs = mu_c + s * rng.standard_normal((draws, 2))
    for lab, mt in (("the gain repeats (T at ysync's seeds 0–1 mean)", float(np.mean(CA_T_OBS))),
                    ("no gain (T = C)", mu_c), ("harm (T = C + 0.05)", mu_c + 0.05)):
        ts = mt + s * rng.standard_normal((draws, 2))
        print(f"  {lab:<48} passes {rate(ts.sum(1) / cs.sum(1) <= CA_MAX)} %  (SD {s:.4f})")
    for t in (0.25, mu_c, 0.45, 0.493):
        pooled = (sum(CA_T_OBS) + 2 * t) / (sum(CA_C_OBS) + 2 * mu_c)
        print(f"  beside: T {t:.3f} per run at seeds 2–3 -> pooled seeds 0–3 ratio {pooled:.3f}, seeds 2–3 alone "
              f"{t / mu_c:.3f}")
    lo, hi = math.sqrt(3 / 9.348), math.sqrt(3 / 0.2158)
    print(f"\nA SD from 3 degrees of freedom: 95 % interval {lo:.2f}x to {hi:.2f}x the estimate.")


def check() -> None:
    """The data above against the run table (skipped without it)."""
    import json
    from pathlib import Path

    import pandas as pd

    from itrea.paths import DATASETS, ROOT
    p = DATASETS / "runs.parquet"
    if not p.exists():
        print("check: no run table, skipped")
        return
    r = pd.read_parquet(p)
    names = {"item": "item_placement_9903", "comp": "comp21-2idx"}
    src = {"item": [("itx-r13", "A9@r13x"), ("r1-bsync", "A9@r1ctl"), ("r1-ysync", "A9@r1ctl-ysync")],
           "comp": [("itx-r13", "A9@r13x"), ("r1-bsync", "A9@r1ctl"), ("r1-ysync", "A9@r1ctl-ysync")]}
    bad = []

    def get(set_, inst, arm, seed):
        x = r[(r["set"] == set_) & (r["instance"] == inst) & (r["arm"] == arm) & (r["seed"] == seed)
              & np.isclose(r["budget"].astype(float), 120.0)]
        if len(x) != 1:
            return None, None
        row = x.iloc[0]
        s = Path(ROOT) / str(row["run_dir"]) / "ea.summary.json"
        g = json.loads(s.read_text()).get("generations") if s.exists() else None
        return g, float(row["pt"])

    for inst, pairs in src.items():
        for i, (set_, arm) in enumerate(pairs):
            for seed in (0, 1):
                g, pt = get(set_, names[inst], arm, seed)
                if g != PAIRS3[inst + "G"][i][seed] or pt is None or abs(pt - PAIRS3[inst + "P"][i][seed]) > 6e-5:
                    bad.append(f"{set_} {arm} {inst} s{seed}: G {g} P {pt}")
    for i, arm in enumerate(("A9@bo1s", "A9@boY")):
        for seed in (0, 1):
            g, pt = get("backoff2", names["item"], arm, seed)
            if g != EXTRA_ITEM["itemG"][i][seed] or pt is None or abs(pt - EXTRA_ITEM["itemP"][i][seed]) > 6e-5:
                bad.append(f"backoff2 {arm} item s{seed}: G {g} P {pt}")
    for inst in ("item", "comp"):
        for j, arm in enumerate(("A9@ysync", "A9@r1ctl-ysync")):
            for seed in (0, 1):
                g, pt = get("r1-ysync", names[inst], arm, seed)
                if g != OBS[inst + "G"][j][seed] or pt is None or abs(pt - OBS[inst + "P"][j][seed]) > 6e-5:
                    bad.append(f"r1-ysync {arm} {inst} s{seed}: G {g} P {pt}")
    print("check: the data match the run table" if not bad else "check: MISMATCH\n  " + "\n  ".join(bad))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="d029_oc", description=__doc__.split("\n\n")[0])
    ap.add_argument("--draws", type=int, default=200_000)
    ap.add_argument("--check", action="store_true", help="compare the data with the run table first")
    a = ap.parse_args(argv)
    if a.check:
        check()
    run(a.draws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
