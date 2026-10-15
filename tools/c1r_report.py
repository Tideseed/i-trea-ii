"""C1 re-read report (D-030): C1 read again on the frozen-v2 candidate — D-022's C1 runs (the 17 development +
class-probe instances x {A9, A6F} x seeds 0-1, 120 s, harness ext) re-run in one set `c1r-<a9>` with A9 on the
candidate's configuration (D-026 ADOPT: `ld0`; or `ysync-ld0` if D-029 adopts ysync and Berk approves the
combination) and A6F re-run in the same window (Berk 2026-10-14, D-025's open question 1: "Re-run C1 on frozen hash").
The rules are D-022's (C1's hint and guard, its second amendment of 2026-10-13) and D-025 amendment 3's (the harm leg),
fixed before any run; D-030 records them. Nothing is chosen by eye, and nothing here decides C1: the go / no-go is
Berk's (D-019 addendum).

Usage (repository root, after `itrea build-runs`; untimed analysis: OMP_NUM_THREADS=1, never beside timed runs with
default thread counts):

    OMP_NUM_THREADS=1 .venv/bin/python tools/c1r_report.py --a9 ld0|ysync-ld0 [--set c1r-<a9>]
                                                           [--failed both|exclude|keep] [--json out]
    itrea c1r-report …                        (the same, through the entry point)

`--a9` (and `--set`) are the arguments the job file was written with (`make-jobs c1r-rerun --a9 …`): the expected runs,
labels and run ids come from `make_jobs.c1_rerun`, whose run hashes are pinned (A9@c1r 341267810eed for ld0,
f345de0ab936 for ysync-ld0; A6F@c1r 82a371305f88 = A6F@r13x's), so the report and the job file cannot drift apart.

Reads `datasets/runs.parquet` (`pt` = P(120), `status`, `attempt`, `run_dir`, `harness`, `cuda_compiled_on_clock`,
`window_id`, `block_void`), each run's `run.json` (`config.harness`, `compiled_on_clock_cuda`, `cupy_cache`) and, for
A9, `ea.summary.json` (`compiled_on_clock`, `cuda_sched`, `phase_events_block`; beside: `lp_decode`, `generations`).
Also from the run table: D-022's runs of set `itx-r13` (A9@r13x, A6F@r13x: the 10-13 reading and the harm leg's
reference and band), A0's runs of sets `dev` / `classprobe` (seeds 0-1, 120 s; vs SCIP alone, reported) and the
same-configuration A9 replicates of 10-15 (`NULL_A9`: A9@r1ctl, A9@r1ctl-ysync, A9@r2ctl — A9@r13x's run hash
28e5a5e9c725 re-run under the sets' own labels; beside the harm leg); the set's `void.jsonl` (beside).

Validity (gating; INVALID names the leg and no rule is read; a missing run alone -> INCOMPLETE, the rules read what is
there, each INCOMPLETE where it needs a missing run):
  - complete: 68 of 68 (17 x {A9@c1r, A6F@c1r} x seeds 0-1), each with P(120);
  - harness: every run's resolved harness (run.json `config.harness`, else the table's) is `pair.resolve_harness("ext")`;
  - config: each run's id is its job's (the run hash carries the EA configuration hash, the harness and the
    footprint) — the configurations as the job file;
  - compile: no compile on the clock — the project's CUDA cache on every run (run.json `compiled_on_clock_cuda`, the
    table's `cuda_compiled_on_clock`) and, on A9 (the arm with an EA GPU process), the CuPy cache (run.json
    `cupy_cache.compiled_on_clock`, the EA summary's `compiled_on_clock`); A6F is never charged with a CuPy flag
    (the runner's own rule, as D-022's report reads it);
  - window: the two runs of each block (instance x seed) ran in one window (the same `window_id`); the run may split
    across windows by whole blocks (a deadline or a stop never splits a block, and a half-done block found on resume
    is voided and rerun whole by the pool, `Pool.prepare`; this leg guards against a split that bypasses the pool);
  - flag (gating since D-030's amendment of 2026-10-15, as D-025 and D-026 gate it): each A9 run's CUDA scheduling
    flag as the candidate's file writes it — ld0: CUDA's default read back (`cuda_sched.state` "read", active "auto");
    ysync-ld0: yield set, verified, active and held at the end — and timing events not blocking; a failed A9 run
    without an EA summary is noted, not charged (an outcome of its arm under `keep`).
  Reported beside, not gating: each A9 run's LP decode gate (open / why / when) and generations; the set's voids (the
  pool's `void.jsonl`: count and reasons); whether `configs/frozen-v2.yaml` exists and hashes to the candidate.

Rules (D-030 cites them; the hint and the guard are computed by D-022's own functions, `tools/d022_report.py`
`c1_hint` / `c1_guard` / `c1_interim`, on these runs relabelled A9@c1r -> A9@r13x, A6F@c1r -> A6F@r13x):
  - HINT present iff the ratio of means of P(120) A9 / A6F over the FIXED 11 (D-022's C1_INSTANCES: doon,
    cryptanalysiskb128n5obj16, rail01, crna, loue, sing44, ca, item_placement, neos-960392, toguru, comp21; seed means,
    both seeds of both arms; not re-selected) is <= 0.95; INCOMPLETE while a run of the 11 is missing;
  - GUARD holds iff the mean bounded d of A9 vs A6F on seed means over all 17 (both-P = 1 at d = 0) is >= -0.05;
    INCOMPLETE (no partial mean) while a run of the 17 is missing;
  - HARM (D-025 amendment 3, labelled cross-day; band as D-030's amendment of 2026-10-15 fixes it): per development
    instance of the 17, the new A9 (A9@c1r) against A9@r13x of 10-13 (set itx-r13): d_i = (P(A9@r13x) - P(new)) /
    max(both, 1e-3) on seed means (`stats.effect_d`), band_i = max(0.05, the larger seed-0 vs seed-1 bounded difference
    |a - b| / max(a, b, 1e-3) of the control or of A6F) — D-023's `band_i`, which the amendment cites: the control =
    A9@r13x, A6F = A6F@r13x (the cuOpt-alone arm that ran beside it on 10-13), never the treatment; so band_i and the
    threshold (the smallest P̄ of the new A9 at which harm fires) are numbers fixed by 10-13's runs before any c1r run.
    harm iff d_i < -band_i. Any harm goes to Berk with the numbers beside the go / no-go, and the frozen-v2 candidate
    is not presumed: he decides which configuration freezes.
    Beside (not criteria): the band of D-030's first reading (A9@r13x or the new A9) and with A6F@c1r in A6F@r13x's
    place; raw ΔP and whether |ΔP| <= 0.005 (D-022's floor); where harm CANNOT fire (threshold >= 1: A9@r13x at P = 1
    on both seeds, or a band so wide that P would have to exceed 1 — rail01, cryptanalysiskb128n5obj16, doon, crna on
    10-13's runs), with each seed's P and the EA's generations 10-13 -> new; and the null lines — the same d_i / band_i
    on cuOpt alone (A6F@c1r against A6F@r13x: the identical job on two days) and on the same-configuration A9 replicates
    of 10-15 against A9@r13x (`NULL_A9`), each instance's fires counted.
  Reported, not criteria (D-013 terms, as `tools/d022_report.py` reports them): A9@c1r vs A6F@c1r (same window, equal
  footprint) on the fixed 11, and vs A0 (SCIP alone; sets dev / classprobe, earlier days) on the fixed 11 and on
  SCIP's weak set among them (A0 P >= 0.5), and on SCIP's weak set among all 17 (D-019's C1 asks for the 17; the
  fixed-11 line is D-022's reading of 10-13): improved k of n, worse, mean P before -> after, mean d with its 95 %
  bootstrap interval, the Wilcoxon test, worst P after. Beside: the 10-13 reading — the same functions on set itx-r13's
  A9@r13x / A6F@r13x (D-022's Result recorded hint 0.827, guard +0.063) — per instance and in total.

Failed runs: outcomes of their arm (D-002 §6, `keep`, the reading the output leads with: truncated at the crash, as
build-runs scores it); the letter (`exclude`: a failed run is missing, so what needs it is INCOMPLETE) printed beside
when a run failed. Output: HINT present / absent, GUARD holds / fails, the HARM list, and "go / no-go is Berk's".
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.analysis import c1_devtests as C
from itrea.analysis.backoff import bounded
from itrea.analysis.stats import D_FLOOR, effect_d
from itrea.experiments import make_jobs as mj
from itrea.experiments import pair
from itrea.paths import DATASETS, ROOT

_spec = importlib.util.spec_from_file_location("d022_report_for_c1r",
                                               Path(__file__).resolve().parent / "d022_report.py")
D22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D22)

A9, A6F = f"A9@{mj.C1R_VARIANT}", f"A6F@{mj.C1R_VARIANT}"
LABELS = (A9, A6F)
SEEDS, BUDGET = tuple(mj.C1R_SEEDS), float(mj.C1R_BUDGET)
OLD_SET, OLD_A9, OLD_A6F = D22.SET, D22.TREAT, D22.A6F          # itx-r13, A9@r13x, A6F@r13x (10-13)
TO_D022 = {A9: OLD_A9, A6F: OLD_A6F}                            # D-022's functions read these labels
FROM_D022 = {v: k for k, v in TO_D022.items()}
C1_INSTANCES = tuple(D22.C1_INSTANCES)                          # the fixed 11 (D-022 second amendment)
HARM_BAND_MIN = 0.05                                            # D-025 amendment 3 (D-023's band_i)
FLOOR_DP = D22.FLOOR_DP                                         # D-022's floor (0.005 raw P): a marker beside harm
OLD_A9_HASH = "28e5a5e9c725"                                    # A9@r13x's run hash under ext (-3way.yaml)
NULL_A9 = (("r1-bsync", "A9@r1ctl"), ("r1-ysync", "A9@r1ctl-ysync"), ("r2-ld", "A9@r2ctl"))   # 10-15, same config
RECORDED_1013 = dict(hint=0.827, guard=0.063)                   # D-022's Result, 2026-10-13 22:2x (as recorded)
LEGS = ("complete", "harness", "config", "compile", "window", "flag")
FLAG = {"ld0": "auto", "ysync-ld0": "yield"}                    # each candidate's CUDA scheduling flag as written
BERK = ("go / no-go on C1 is Berk's (D-019 addendum): the hint and the guard are its inputs; any harm goes to Berk "
        "with the numbers beside the go / no-go, and the frozen-v2 candidate is not presumed — he decides which "
        "configuration freezes (D-025 amendment 3)")


def development_instances() -> list[str]:
    return list(mj.development_names())


def expected(a9: str, set_: str) -> dict:
    """{(instance, label, seed): run id} of the set's jobs (`make_jobs.c1_rerun`, pinned run hashes)."""
    return {(j["name"], pair.arm_label(j), int(j["seed"])): pair.run_id_for(j) for j in mj.c1_rerun(a9, set_=set_)}


# ----------------------------------------------------------------------------------------------- one run
def cupy_compiled(rj: dict) -> bool:
    cc = rj.get("cupy_cache")
    return bool(isinstance(cc, dict) and C.flag(cc.get("compiled_on_clock")))


def flag_state(a9: str, row: dict, summ: dict) -> dict:
    """The A9 run's CUDA scheduling flag against the candidate's file (ld0: CUDA's default read back; ysync-ld0: yield
    set, verified and held; timing events not blocking in both); `flag_problems` gates on it."""
    if not summ:
        return dict(ok=None, text="no ea.summary.json" + (" (failed run)" if row.get("status") == "failed" else ""))
    cs = summ.get("cuda_sched") or {}
    want = FLAG[a9]
    if want == "auto":
        ok = cs.get("state") == "read" and cs.get("sched_active") == "auto"
    else:
        ok = cs.get("state") == "verified" and cs.get("sched_active") == want and cs.get("sched_at_end") == want
    ok = bool(ok and not C.flag(summ.get("phase_events_block")))
    return dict(ok=ok, want=want, state=cs.get("state"), active=cs.get("sched_active"), at_end=cs.get("sched_at_end"),
                events_block=summ.get("phase_events_block"),
                text=f"{cs.get('state')}/{cs.get('sched_active')}" + ("" if ok else f" NOT AS WRITTEN (want {want})"))


def flag_problems(a9: str, row: dict, summ: dict) -> tuple[list[str], str | None]:
    """The flag leg (gating, D-030's amendment of 2026-10-15; d025_report / d026_report's shape): (problems, note). A
    failed A9 run without an EA summary is noted, not charged (an outcome of its arm under `keep`); a completed one
    without a summary fails the leg."""
    if not summ:
        if row.get("status") == "failed":
            return [], "failed run without ea.summary.json: flag unread"
        return ["no ea.summary.json: the CUDA flag is unread"], None
    st = flag_state(a9, row, summ)
    if st["ok"]:
        return [], None
    return [f"CUDA flag not as written (want {st['want']}: state {st['state']!r}, active {st['active']!r}, at the end "
            f"{st['at_end']!r}, timing events blocking {st['events_block']!r})"], None


def decoder_state(summ: dict) -> dict:
    ld = summ.get("lp_decode") if isinstance(summ.get("lp_decode"), dict) else {}
    return dict(gate=ld.get("gate"), gate_why=ld.get("gate_why"), gate_t=ld.get("gate_t"),
                generations=summ.get("generations"))


# ----------------------------------------------------------------------------------------------- the rules
def _p_by(rows) -> dict:
    """{(instance, label, seed): P(120)} of the rows with a finite pt."""
    return {(r["instance"], r["arm"], int(r["seed"])): float(r["pt"]) for r in rows if C.fin(r.get("pt"))}


def _seeds(p: dict, i: str) -> list:
    return [p.get((i, s)) for s in SEEDS]


def _spread(v: list) -> float | None:
    """The bounded seed-0 vs seed-1 difference |a - b| / max(a, b, 1e-3); None without both seeds."""
    if any(x is None for x in v):
        return None
    x = bounded(v[0], v[1])
    return float(x) if C.fin(x) else None


def _band(*spreads) -> float:
    return max([HARM_BAND_MIN] + [x for x in spreads if x is not None])


def p_threshold(p_old: float, band: float) -> float:
    """The smallest seed-mean P of the new A9 at which harm fires (effect_d(p_old, P) < -band), given P̄(A9@r13x) and
    band_i: P > p_old / (1 - band) (P > p_old + band x 1e-3 below the 1e-3 floor); inf when band >= 1."""
    if band >= 1.0:
        return math.inf
    t = p_old / (1.0 - band)
    return t if t >= D_FLOOR else p_old + band * D_FLOOR


def harm_bands(old: dict, a6_old: dict, instances) -> dict:
    """{instance: band_i, threshold, cannot_fire} fixed by 10-13's runs alone (D-030's amendment of 2026-10-15): band_i
    = max(0.05, the bounded seed spread of A9@r13x (the control), that of A6F@r13x (A6F)); the threshold = the
    smallest P̄ of the new A9 at which harm fires; cannot_fire iff the threshold >= 1 (P <= 1). None for an instance
    without both seeds of A9@r13x."""
    out = {}
    for i in instances:
        po = _seeds(old, i)
        if any(v is None for v in po):
            out[i] = None
            continue
        so, sa = _spread(po), _spread(_seeds(a6_old, i))
        band = _band(so, sa)
        thr = p_threshold(float(np.mean(po)), band)
        out[i] = dict(band=band, spread_old=so, spread_a6_old=sa, p_old=float(np.mean(po)), threshold=thr,
                      cannot_fire=bool(thr >= 1.0))
    return out


def harm_leg(new: dict, old: dict, instances, a6_old: dict | None = None, a6_new: dict | None = None) -> dict:
    """D-025 amendment 3 on each instance, band as D-030's amendment of 2026-10-15 reads D-023's `band_i`: `new`
    {(instance, seed): P of the new A9}, `old` the same of A9@r13x (10-13), `a6_old` of A6F@r13x (10-13), `a6_new` of
    A6F@c1r (beside only). d_i = effect_d(P̄ old, P̄ new) (positive = the new A9 better); band_i = max(0.05,
    bounded(A9@r13x s0, s1), bounded(A6F@r13x s0, s1)) — never the treatment's own spread; harm iff d_i < -band_i.
    An instance without both seeds of both A9 arms is unread (INCOMPLETE). Beside: `band_first` (D-030's first
    reading: A9@r13x or the new A9), `band_a6_new` (A6F@c1r in A6F@r13x's place), `dp` (raw P̄ new - P̄ old),
    `floor_dp` (|dp| <= 0.005), `threshold` / `cannot_fire` (`harm_bands`)."""
    a6_old, a6_new = a6_old or {}, a6_new or {}
    bands = harm_bands(old, a6_old, instances)
    rows, harm, unread = [], [], []
    for i in instances:
        pn, po = _seeds(new, i), _seeds(old, i)
        b = bands[i] or {}
        if any(v is None for v in pn + po):
            unread.append(i)
            rows.append(dict(instance=i, read=False, p_new_seeds=pn, p_old_seeds=po, band=b.get("band"),
                             threshold=b.get("threshold"), cannot_fire=b.get("cannot_fire")))
            continue
        mo, mn = float(np.mean(po)), float(np.mean(pn))
        band = b["band"]
        d = effect_d(mo, mn)
        h = bool(d < -band)
        sn, sa_new = _spread(pn), _spread(_seeds(a6_new, i))
        rows.append(dict(instance=i, read=True, p_old=mo, p_new=mn, p_old_seeds=po, p_new_seeds=pn, d=d, band=band,
                         spread_old=b["spread_old"], spread_a6_old=b["spread_a6_old"], spread_new=sn,
                         band_first=_band(b["spread_old"], sn),
                         band_a6_new=_band(b["spread_old"], sa_new) if sa_new is not None else None,
                         threshold=b["threshold"], cannot_fire=b["cannot_fire"], dp=mn - mo,
                         floor_dp=bool(abs(mn - mo) <= FLOOR_DP), harm=h))
        if h:
            harm.append(i)
    state = "HARM" if harm else ("INCOMPLETE" if unread else "none")
    return dict(rows=rows, harm=harm, unread=unread, state=state, band_min=HARM_BAND_MIN,
                cannot_fire=[i for i in instances if (bands[i] or {}).get("cannot_fire")], bands=bands)


def null_read(ref: dict, rep: dict, bands: dict, instances) -> dict:
    """The harm rule applied to a pair with no change between them (`ref` {(instance, seed): P} on 10-13, `rep` the
    same configuration on another day): per instance with both seeds of both, d_i = effect_d(P̄ ref, P̄ rep) against
    the harm leg's band_i; `fired` = the instances where d_i < -band_i (noise, by construction)."""
    rows = []
    for i in instances:
        pr, pp, b = _seeds(ref, i), _seeds(rep, i), bands.get(i)
        if b is None or any(v is None for v in pr + pp):
            continue
        d = effect_d(float(np.mean(pr)), float(np.mean(pp)))
        rows.append(dict(instance=i, d=d, band=b["band"], fires=bool(d < -b["band"]), p_ref_seeds=pr, p_rep_seeds=pp))
    return dict(rows=rows, n=len(rows), fired=[r["instance"] for r in rows if r["fires"]])


def per_instance(p: dict, a9: str, a6f: str, instances) -> dict:
    """{instance: P̄ A6F, P̄ A9 and the bounded d of A9 vs A6F on seed means} where both seeds of both arms are present
    (the table printed beside the rules; the guard's own rows are D-022's)."""
    out = {}
    for i in instances:
        pa = [p.get((i, a6f, s)) for s in SEEDS]
        p9 = [p.get((i, a9, s)) for s in SEEDS]
        if any(v is None for v in pa + p9):
            continue
        ma, m9 = float(np.mean(pa)), float(np.mean(p9))
        out[i] = dict(p_a6f=ma, p_a9=m9, d=effect_d(ma, m9))
    return out


def _relabel(rows) -> pd.DataFrame:
    df = pd.DataFrame([dict(instance=r["instance"], seed=int(r["seed"]), arm=TO_D022[r["arm"]], pt=r.get("pt"))
                       for r in rows], columns=["instance", "seed", "arm", "pt"])
    df["pt"] = pd.to_numeric(df["pt"], errors="coerce")
    return df


def _ours(x: dict) -> dict:
    """A D-022 hint / guard result with its missing runs named by this set's labels (A9@c1r / A6F@c1r)."""
    return dict(x, missing=[(i, FROM_D022.get(a, a), s) for i, a, s in x.get("missing", [])])


def vs_a0_17(long: pd.DataFrame, a0: pd.DataFrame, instances) -> dict:
    """Reported, not a criterion: A9 (D-022's label TREAT in `long`) vs A0 in D-013 terms on SCIP's weak set among ALL
    the 17 (A0 P̄ >= 0.5) — D-019's C1 ("an interim A9 result on the 17 … vs SCIP alone and cuOpt alone, reported in
    D-013 terms"); the fixed-11 line beside it is D-022's reading of 10-13."""
    cols = ["instance", "seed", "arm", "pt"]
    lg = pd.concat([long[cols], a0[cols]], ignore_index=True)
    lg = lg[lg["pt"].notna() & lg["instance"].isin(list(instances))]
    m = lg[lg["arm"] == D22.A0].groupby("instance")["pt"].mean()
    weak = sorted(i for i in instances if i in m.index and m[i] >= D22.WEAK_P)
    return dict(weak_set=weak, n_baseline=int(len(m)), block=D22._weak_block(lg, D22.TREAT, D22.A0, weak))


def _gens(rows, root: Path) -> dict:
    """{(instance, seed): the EA's generations} of A9 rows (their ea.summary.json; None when unread)."""
    out = {}
    for r in rows:
        g = C.run_files(r, root)[1].get("generations") if str(r.get("run_dir") or "") not in ("", "nan", "None") \
            else None
        out[(r["instance"], int(r["seed"]))] = g
    return out


def null_lines(runs: pd.DataFrame, p_old: dict, bands: dict, instances, a6_old: dict, a6_new: dict) -> dict:
    """Beside the harm leg, never a criterion: the harm rule (the same d_i and band_i) on pairs with no change between
    them — cuOpt alone, A6F@c1r against A6F@r13x (the identical job, 10-13 and this window), and each same-configuration
    A9 replicate of 10-15 (`NULL_A9`, A9@r13x's run hash) against A9@r13x. Rows whose run hash is not A9@r13x's are
    left out (stated)."""
    same, skipped = [], []
    for set_, lab in NULL_A9:
        df = C.select(runs, set_, (lab,), SEEDS, BUDGET, keep_failed=True)[0]
        ok = df["run_id"].astype(str).map(lambda x: C.bare_id(x).rsplit("__", 1)[-1] == OLD_A9_HASH) \
            if len(df) else pd.Series(dtype=bool)
        skipped += sorted(str(x) for x in df[~ok]["run_id"]) if len(df) else []
        df = df[ok] if len(df) else df
        p = {(i, s): v for (i, a, s), v in _p_by(df.to_dict("records")).items()}
        if p:
            same.append(dict(set=set_, label=lab, **null_read(p_old, p, bands, instances)))
    n = sum(x["n"] for x in same)
    k = sum(len(x["fired"]) for x in same)
    return dict(cuopt=null_read(a6_old, a6_new, bands, instances), same_config=same, n=n, k=k, skipped=skipped)


def rules(rows, runs: pd.DataFrame, instances, root: Path = ROOT, gens_new: dict | None = None) -> dict:
    """C1's hint and guard (D-022's functions on the relabelled rows), the D-013 blocks vs A6F and vs A0 (and vs A0 on
    SCIP's weak set among the 17, beside), the harm leg against A9@r13x of 10-13 with its band from A9@r13x and
    A6F@r13x, the null lines and the cannot-fire instances' seeds and generations beside, and the 10-13 reading."""
    a0, _ = D22.select(runs, D22.A0_SETS, (D22.A0,), instances=list(C1_INSTANCES))
    a0_17, _ = D22.select(runs, D22.A0_SETS, (D22.A0,), instances=list(instances))
    long = _relabel(rows)
    c1 = D22.c1_interim(long, a0, list(C1_INSTANCES), list(instances))
    old = C.select(runs, OLD_SET, (OLD_A9, OLD_A6F), SEEDS, BUDGET, keep_failed=True)[0]
    old = old[old["instance"].isin(list(instances))]
    old_long = old[["instance", "seed", "arm", "pt"]].copy()
    then = D22.c1_interim(old_long, a0, list(C1_INSTANCES), list(instances))
    pn_all, po_all = _p_by(rows), _p_by(old.to_dict("records"))
    p_new = {(i, s): v for (i, a, s), v in pn_all.items() if a == A9}
    a6_new = {(i, s): v for (i, a, s), v in pn_all.items() if a == A6F}
    p_old = {(i, s): v for (i, a, s), v in po_all.items() if a == OLD_A9}
    a6_old = {(i, s): v for (i, a, s), v in po_all.items() if a == OLD_A6F}
    harm = harm_leg(p_new, p_old, instances, a6_old, a6_new)
    gens_old = _gens([r for r in old.to_dict("records") if r["arm"] == OLD_A9], root)
    gens_new = gens_new or {}
    harm["cannot_fire_rows"] = [dict(instance=i, p_old_seeds=_seeds(p_old, i), p_new_seeds=_seeds(p_new, i),
                                     gens_old=_seeds(gens_old, i), gens_new=_seeds(gens_new, i),
                                     threshold=harm["bands"][i]["threshold"], band=harm["bands"][i]["band"])
                                for i in harm["cannot_fire"]]
    nulls = null_lines(runs, p_old, harm["bands"], instances, a6_old, a6_new)
    return dict(hint=_ours(c1["hint"]), guard=_ours(c1["guard"]), vs_a6f=c1[D22.A6F][D22.TREAT]["all"],
                vs_a0=c1[D22.A0], vs_a0_17=vs_a0_17(long, a0_17, instances), harm=harm, nulls=nulls,
                table=per_instance(pn_all, A9, A6F, instances),
                then=dict(hint=then["hint"], guard=then["guard"], n_runs=int(len(old)), recorded=RECORDED_1013,
                          vs_a0_17=vs_a0_17(old_long, a0_17, instances),
                          table=per_instance(po_all, OLD_A9, OLD_A6F, instances)))


# ----------------------------------------------------------------------------------------------- one reading
def reading(runs: pd.DataFrame, root: Path, a9: str, set_: str, keep_failed: bool, ext: dict | None = None) -> dict:
    exp = expected(a9, set_)
    ext = ext if ext is not None else C.ext_harness()
    instances = development_instances()
    sel, dropped = C.select(runs, set_, LABELS, SEEDS, BUDGET, keep_failed)
    by = {(r["instance"], r["arm"], int(r["seed"])): r for r in sel.to_dict("records")}
    outside = sorted(str(r["run_id"]) for k, r in by.items() if k not in exp)
    failed = C.failed_keys(runs, set_, LABELS, SEEDS, BUDGET)
    legs: dict[str, list[str]] = {k: [] for k in LEGS}
    legs["complete"], present = C.completeness(list(exp), by, failed)
    cells, kept = {}, []
    for k in present:
        r = by[k]
        rj, summ = C.run_files(r, root)
        if k[1] != A9:
            summ = {}                                    # A6F has no EA: never charged with an EA / CuPy record
        name = f"{k[0]} {k[1]} s{k[2]}"
        for leg, probs in C.base_problems(r, rj, summ, exp[k], ext).items():
            legs[leg] += [f"{name}: {x}" for x in probs]
        if k[1] == A9 and cupy_compiled(rj) and not any(x.startswith(name) for x in legs["compile"]):
            legs["compile"].append(f"{name}: compiled on the clock (CuPy cache)")
        if not C.fin(r.get("pt")):
            legs["complete"].append(f"no P(120): {name}")
        if r.get("status") == "failed":
            kept.append(name)
        note = None
        if k[1] == A9:
            fp, note = flag_problems(a9, r, summ)
            legs["flag"] += [f"{name}: {x}" for x in fp]
        cells[k] = dict(row=r, summ=summ, flag=flag_state(a9, r, summ) if k[1] == A9 else None,
                        flag_note=note, decoder=decoder_state(summ) if k[1] == A9 else None)
    for i in instances:
        for s in SEEDS:
            have = [cells[(i, lab, s)] for lab in LABELS if (i, lab, s) in cells]
            if len(have) < 2:
                continue
            ws = [h["row"].get("window_id") for h in have]
            if not all(isinstance(w, str) and w for w in ws):
                legs["window"].append(f"{i} s{s}: a window id is missing")
            elif len(set(ws)) > 1:
                legs["window"].append(f"{i} s{s}: the two runs ran in different windows {sorted(set(ws))}")
    bad = [k for k in LEGS if legs[k]]
    verdict = "INVALID" if any(k != "complete" for k in bad) else ("INCOMPLETE" if bad else "VALID")
    val = dict(verdict=verdict, failing=bad, legs=legs,
               deviation=[f"{x} failed and is read as D-002 §6 keeps a crash (an outcome of its arm)" for x in kept])
    out = dict(keep_failed=keep_failed, a9=a9, set=set_, n_expected=len(exp), n_present=len(cells), validity=val,
               duplicates=dropped, outside=outside, cells=cells, rules=None, summary=None)
    if verdict != "INVALID":
        gens = {(k[0], k[2]): c["summ"].get("generations") for k, c in cells.items() if k[1] == A9}
        out["rules"] = rules([c["row"] for c in cells.values()], runs, instances, root, gens)
        out["summary"] = summary(out["rules"], verdict)
    else:
        out["summary"] = dict(validity="INVALID", hint="not read", guard="not read", harm="not read", berk=BERK)
    return out


def harm_notes(ru: dict) -> dict:
    """{instance: [notes]} beside each harm the rule found (never a change to the list): |ΔP| within D-022's floor
    0.005, the null reads on the instance (same-configuration A9 replicates, cuOpt alone across the two days) and
    whether one of them reached a d_i as low."""
    out = {}
    rows = {r["instance"]: r for r in ru["harm"]["rows"]}
    nl = ru["nulls"]
    for i in ru["harm"]["harm"]:
        r, notes = rows[i], []
        if r["floor_dp"]:
            notes.append(f"|ΔP| {abs(r['dp']):.4f} <= {FLOOR_DP:g} (D-022's floor)")
        reads = [(x["label"], y["d"]) for x in nl["same_config"] for y in x["rows"] if y["instance"] == i]
        cu = [y["d"] for y in nl["cuopt"]["rows"] if y["instance"] == i]
        if reads:
            low = min(d for _, d in reads)
            notes.append("same-configuration replicates here: " + ", ".join(f"{lab} d {d:+.3f}" for lab, d in reads)
                         + (" (one as low: within the null range)" if low <= r["d"] else " (none as low)"))
        else:
            notes.append("no same-configuration replicate on this instance")
        if cu:
            notes.append(f"cuOpt alone 10-13 -> this window d {cu[0]:+.3f}")
        out[i] = notes
    return out


def summary(ru: dict, verdict: str) -> dict:
    h, g, hm = ru["hint"], ru["guard"], ru["harm"]
    return dict(validity=verdict, hint=h["state"], hint_ratio=h.get("ratio"), guard=g["state"],
                guard_mean_d=g.get("mean_d"), harm=hm["state"], harm_list=list(hm["harm"]),
                harm_notes=harm_notes(ru), harm_unread=list(hm["unread"]), harm_cannot_fire=list(hm["cannot_fire"]),
                null_rate=(ru["nulls"]["k"], ru["nulls"]["n"]), berk=BERK)


def frozen_v2_state(a9: str, root: Path = ROOT) -> dict:
    """Beside: does configs/frozen-v2.yaml exist, and does it hash (EA hash and A9 run hash under ext; `name` is not
    hashed) to the candidate this set ran? The re-read describes frozen-v2 only if it does (D-030's amendment);
    `tests/test_cuda_sched.py` pins the same pair."""
    p = Path(root) / "configs" / "frozen-v2.yaml"
    if not p.exists():
        return dict(state="absent", text="configs/frozen-v2.yaml not written yet")
    try:
        from itrea.config import EAConfig
        want = (EAConfig.from_yaml(Path(root) / mj.C1R_A9[a9][0]).hash(), mj.C1R_A9[a9][1])
        j = dict(instance="data/miplib/inst/glass4.mps.gz", set="x", arm="A9", seed=0, budget=BUDGET,
                 config=str(p), harness="ext")
        got = (EAConfig.from_yaml(p).hash(), pair.run_hash(j))
    except Exception as e:                                            # beside only: never stops the report
        return dict(state="unreadable", text=f"configs/frozen-v2.yaml unreadable: {e}")
    same = got == want
    return dict(state="same" if same else "DIFFERS", got=got, want=want,
                text=(f"configs/frozen-v2.yaml = this set's candidate (EA {got[0]}, A9 run {got[1]})" if same else
                      f"configs/frozen-v2.yaml DIFFERS from this set's candidate (EA {got[0]} / run {got[1]} vs "
                      f"{want[0]} / {want[1]}): the re-read does not describe frozen-v2 — tell Berk"))


def report(runs: pd.DataFrame, root: Path = ROOT, a9: str = "ld0", set_: str | None = None,
           failed: str = "both") -> dict:
    if a9 not in mj.C1R_A9:
        raise ValueError(f"--a9: one of {', '.join(mj.C1R_A9)} (as the job file was written)")
    set_ = set_ or mj.C1R_SET_PREFIX + a9
    cand = C.candidates(runs, set_, LABELS, SEEDS, BUDGET)
    fails = sorted(str(r) for r in cand[cand["status"] == "failed"]["run_id"])
    modes = {"exclude": [False], "keep": [True], "both": [True, False] if fails else [True]}[failed]
    ext = C.ext_harness()
    voids = D22.void_log(runs, [set_], Path(root))
    reasons: dict[str, int] = {}
    for v in voids:
        key = str(v.get("reason") or "unreadable").split(" (")[0]
        reasons[key] = reasons.get(key, 0) + 1
    return dict(set=set_, a9=a9, config=mj.C1R_A9[a9][0], run_hash=mj.C1R_A9[a9][1], failed_runs=fails,
                other_rows=C.others(runs, set_, LABELS, SEEDS, BUDGET),
                voids=dict(n=len(voids), reasons=reasons, runs=[str(v.get("run_id")) for v in voids]),
                frozen_v2=frozen_v2_state(a9, root),
                readings=[reading(runs, root, a9, set_, k, ext) for k in modes])


# ----------------------------------------------------------------------------------------------- output
def _hint_line(h: dict, label: str) -> str:
    if h["state"] == "INCOMPLETE":
        return f"HINT INCOMPLETE ({label}): {len(h['missing'])} runs of the fixed {h['n']} missing {h['missing'][:4]}"
    return (f"HINT {h['state']} ({label}): ratio of means of P(120) A9 / A6F over the fixed {h['n']} = "
            f"{C.f(h['ratio'])} ({C.f(h['mean_P_a9'])} / {C.f(h['mean_P_a6f'])}; present iff <= {h['threshold']:g})")


def _guard_line(g: dict, label: str) -> str:
    if g["state"] == "INCOMPLETE":
        return f"GUARD INCOMPLETE ({label}): {len(g['missing'])} runs of the {g['n']} missing {g['missing'][:4]}"
    return (f"GUARD {g['state']} ({label}): mean bounded d of A9 vs A6F over the {g['n']} = {g['mean_d']:+.3f} "
            f"(holds iff >= {g['threshold']:g}; seed means, both-P = 1 at d = 0)")


def _fp(x, nd=3) -> str:
    """A P / band / threshold for the table ("inf" for an infinite threshold, "–" for none)."""
    if x is None:
        return "–"
    if not math.isfinite(float(x)):
        return "inf"
    return f"{float(x):.{nd}f}"


def render(rep: dict) -> str:
    L = [f"C1 re-read (D-030), set {rep['set']}: {A9} on {rep['config']} (run hash {rep['run_hash']}) vs {A6F} (A6F@r13x's "
         f"job, same window), 17 instances x seeds {list(SEEDS)}, {BUDGET:g} s, ext; rules fixed before any run "
         f"(D-022's hint / guard, D-025 amendment 3's harm leg, band as D-030's amendment of 2026-10-15); development, "
         f"descriptive (D-010)"]
    if rep["failed_runs"]:
        L.append(f"failed runs: {', '.join(rep['failed_runs'])}")
    if rep["other_rows"]:
        L.append(f"rows of the set no rule reads: {', '.join(rep['other_rows'])}")
    vo = rep.get("voids") or dict(n=0, reasons={})
    L.append(f"beside (not gating): voids in {rep['set']}/void.jsonl: {vo['n']}"
             + (f" ({', '.join(f'{k} {v}' for k, v in sorted(vo['reasons'].items()))}; each voided block was rerun "
                f"whole, its rerun is what is read)" if vo["n"] else ""))
    if rep.get("frozen_v2"):
        L.append(f"beside (not gating): {rep['frozen_v2']['text']}")
    for rd in rep["readings"]:
        L.append("")
        L.append(f"== reading: failed runs {'kept (D-002 §6; the reading that leads)' if rd['keep_failed'] else 'excluded (the letter)'} ==")
        v = rd["validity"]
        L.append(f"VALIDITY {v['verdict']}: {rd['n_present']} of {rd['n_expected']} runs present"
                 + (f"; failing leg(s): {', '.join(v['failing'])}" if v["failing"] else ""))
        for leg in LEGS:
            for x in v["legs"][leg][:12]:
                L.append(f"  ({leg}) {x}")
            if len(v["legs"][leg]) > 12:
                L.append(f"  ({leg}) … {len(v['legs'][leg]) - 12} more")
        for x in v["deviation"]:
            L.append(f"  kept: {x}")
        for d in rd["duplicates"]:
            L.append(f"  duplicate left out: {d['run_id']}")
        for x in rd["outside"]:
            L.append(f"  not one of the 68 (ignored): {x}")
        a9_cells = {k: c for k, c in rd["cells"].items() if k[1] == A9}
        if a9_cells:
            notes = [f"{k[0]} s{k[2]}: {c['flag_note']}" for k, c in sorted(a9_cells.items()) if c.get("flag_note")]
            L.append(f"{A9}'s CUDA flag as written ({FLAG[rd['a9']]}, gating) on "
                     f"{sum(1 for c in a9_cells.values() if c['flag']['ok'])} of {len(a9_cells)} runs"
                     + (f"; noted, not charged: {notes}" if notes else "") + "; beside (not gating): LP decode gate "
                     f"open on {sum(1 for c in a9_cells.values() if c['decoder']['gate'] == 'open')} of {len(a9_cells)}")
        if rd["rules"] is None:
            L.append("RULES NOT READ: validity fails -> INVALID")
            L.append(f"SUMMARY: INVALID — {BERK}")
            continue
        ru = rd["rules"]
        g_rows, t_rows = ru["table"], ru["then"]["table"]
        L.append("per instance (seed means; d vs A6F: + = A9 better; harm: d_i = bounded d of the new A9 vs A9@r13x "
                 "of 10-13, cross-day; band_i from A9@r13x and A6F@r13x, fixed before the runs; harm iff d_i < -band_i, "
                 "i.e. iff P̄ new > 'fires >'; beside: band_i of D-030's first reading (A9@r13x / new A9) and with "
                 "A6F@c1r, raw ΔP):")
        L.append("  " + "instance".ljust(28) + "P A6F".ljust(8) + "P A9".ljust(8) + "d".ljust(8) + "| 10-13: P A6F".ljust(15)
                 + "P A9".ljust(8) + "d".ljust(8) + "| harm d_i".ljust(11) + "band_i".ljust(8) + "fires >".ljust(9)
                 + "harm".ljust(13) + "| first".ljust(8) + "A6Fc1r".ljust(8) + "ΔP")
        hm = {r["instance"]: r for r in ru["harm"]["rows"]}
        for i in development_instances():
            g, t, h = g_rows.get(i), t_rows.get(i), hm.get(i, {})
            fixed = "*" if i in C1_INSTANCES else " "
            cant = bool(h.get("cannot_fire"))
            thr = "> 1" if cant else _fp(h.get("threshold"), 4)
            if h.get("read"):
                tag = "HARM" + (" (floor)" if h["floor_dp"] else "") if h["harm"] else ("cannot fire" if cant else "-")
                hcell = (f"| {h['d']:+.3f}    {_fp(h['band']):<8}{thr:<9}{tag:<13}| {_fp(h['band_first']):<6}  "
                         f"{_fp(h['band_a6_new']):<6}  {h['dp']:+.4f}")
            else:
                hcell = f"| unread    {_fp(h.get('band')):<8}{thr:<9}" + ("cannot fire" if cant else "")
            L.append(f"  {fixed}{i:<27}" + (f"{C.f(g['p_a6f']):<8}{C.f(g['p_a9']):<8}{g['d']:+.3f}  " if g else "–".ljust(24))
                     + (f"| {C.f(t['p_a6f']):<13}{C.f(t['p_a9']):<8}{t['d']:+.3f}  " if t else "| –".ljust(31)) + hcell)
        L.append("  (* = the fixed 11 of the hint; 'fires >' = the smallest P̄ of the new A9 at which harm fires, from "
                 "10-13's runs alone; '(floor)' = |ΔP| <= 0.005, D-022's floor — a marker, not a rule)")
        L.append(_hint_line(ru["hint"], "this set"))
        L.append(f"  beside, 10-13 (set {OLD_SET}, {ru['then']['n_runs']} runs): " + _hint_line(ru["then"]["hint"], "10-13")
                 + f"; recorded in D-022's Result: {RECORDED_1013['hint']:.3f}")
        L.append(_guard_line(ru["guard"], "this set"))
        L.append(f"  beside, 10-13: " + _guard_line(ru["then"]["guard"], "10-13")
                 + f"; recorded in D-022's Result: {RECORDED_1013['guard']:+.3f}")
        h = ru["harm"]
        s = rd["summary"]
        if h["harm"]:
            L.append("HARM (D-025 amendment 3, cross-day vs A9@r13x of 10-13) on " + "; ".join(
                f"{r['instance']} (d_i {r['d']:+.3f} < -{r['band']:.3f}; P {C.f(r['p_old'])} -> {C.f(r['p_new'])}, ΔP "
                f"{r['dp']:+.4f}; seeds {' / '.join(C.f(x) for x in r['p_old_seeds'])} -> "
                f"{' / '.join(C.f(x) for x in r['p_new_seeds'])}; beside: {'; '.join(s['harm_notes'][r['instance']])})"
                for r in h["rows"] if r.get("read") and r["harm"]) + " -> to Berk with these numbers")
        else:
            L.append("HARM (D-025 amendment 3, cross-day vs A9@r13x of 10-13): none"
                     + (f" on the {17 - len(h['unread'])} read" if h["unread"] else " on the 17"))
        if h["unread"]:
            L.append(f"  harm leg INCOMPLETE: unread (a seed of either A9 arm missing) {h['unread']}")
        if h["cannot_fire"]:
            L.append(f"  harm CANNOT fire on {len(h['cannot_fire'])} of the 17 (threshold > 1 on 10-13's runs; P <= 1) — "
                     "beside, not criteria, each seed's P and the EA's generations 10-13 -> new:")
            for r in h["cannot_fire_rows"]:
                L.append(f"    {r['instance']}: band_i {_fp(r['band'])}, fires > {_fp(r['threshold'], 3)}; P "
                         f"{' / '.join(C.f(x) for x in r['p_old_seeds'])} -> {' / '.join(C.f(x) for x in r['p_new_seeds'])}"
                         f"; generations {' / '.join(str(x) for x in r['gens_old'])} -> "
                         f"{' / '.join(str(x) for x in r['gens_new'])}")
        nl = ru["nulls"]
        L.append(f"null lines beside the harm leg (the same d_i and band_i on pairs with no change; not criteria): "
                 f"same-configuration A9 replicates of 10-15 vs A9@r13x fire on {nl['k']} of {nl['n']} instance reads")
        for x in nl["same_config"]:
            L.append(f"  {x['label']} ({x['set']}): " + ", ".join(
                f"{y['instance']} {y['d']:+.3f}{' FIRES' if y['fires'] else ''}" for y in x["rows"]))
        cu = nl["cuopt"]
        L.append(f"  cuOpt alone, {A6F} vs {OLD_A6F} (the identical job, 10-13 -> this window): fires on "
                 f"{len(cu['fired'])} of {cu['n']}" + (f" {cu['fired']}" if cu["fired"] else ""))
        if nl["skipped"]:
            L.append(f"  left out (run hash not A9@r13x's {OLD_A9_HASH}): {nl['skipped']}")
        L.append("reported, not criteria (D-013 terms) — vs cuOpt alone (A6F@c1r, same window, equal footprint): "
                 + D22._d013_line(ru["vs_a6f"], "the fixed 11", "instances"))
        c = ru["vs_a0"]
        L.append(f"reported, not criteria (D-013 terms) — vs {c['label']}: "
                 + D22._d013_line(c[D22.TREAT]["all"], "the fixed 11", "instances"))
        L.append(f"  SCIP's weak set among them (A0 P >= {D22.WEAK_P}) {c['weak_set']} ({len(c['weak_set'])} of "
                 f"{c['n_baseline']} with A0 runs)" + (": " + D22._d013_line(c[D22.TREAT]["weak"], "weak set", "weak")
                                                       if c["weak_set"] else ""))
        for lab, w in (("this set", ru["vs_a0_17"]), ("10-13, beside", ru["then"]["vs_a0_17"])):
            L.append(f"  SCIP's weak set among all 17 (D-019's C1; A0 P >= {D22.WEAK_P}: {len(w['weak_set'])} of "
                     f"{w['n_baseline']} with A0 runs), {lab}: "
                     + D22._d013_line(w["block"], "weak set", "weak"))
        L.append(f"SUMMARY ({'keep' if rd['keep_failed'] else 'exclude'}): validity {s['validity']}; HINT {s['hint']}"
                 f" ({C.f(s['hint_ratio'])}); GUARD {s['guard']} ({C.f(s['guard_mean_d'])}); HARM "
                 + ("; ".join(f"{i} [{'; '.join(s['harm_notes'][i])}]" for i in s["harm_list"]) if s["harm_list"]
                    else s["harm"])
                 + (f" (unread {s['harm_unread']})" if s["harm_unread"] else "")
                 + f"; harm cannot fire on {s['harm_cannot_fire']}; null: same-configuration replicates fire on "
                   f"{s['null_rate'][0]} of {s['null_rate'][1]} instance reads")
        L.append(f"  {BERK}")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="c1r_report", description=__doc__.split("\n\n")[0])
    ap.add_argument("--a9", required=True, choices=tuple(mj.C1R_A9),
                    help="the candidate, as the job file was written (make-jobs c1r-rerun --a9 …)")
    ap.add_argument("--set", default=None, help="default c1r-<a9>")
    ap.add_argument("--runs", type=Path, default=DATASETS / "runs.parquet")
    ap.add_argument("--root", type=Path, default=ROOT, help="the run directories' root (run_dir is relative to it)")
    ap.add_argument("--failed", choices=("both", "exclude", "keep"), default="both")
    ap.add_argument("--json", type=Path, help="write the report as JSON here too")
    a = ap.parse_args(argv)
    try:
        runs = pd.read_parquet(a.runs)
        rep = report(runs, a.root, a.a9, a.set, a.failed)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(render(rep))
    if a.json:
        out = dict(rep, readings=[{k: v for k, v in rd.items() if k != "cells"} | dict(
            runs={f"{i} {lab} s{s}": dict(run_id=c["row"]["run_id"], pt=c["row"].get("pt"),
                                          status=c["row"].get("status"), window_id=c["row"].get("window_id"),
                                          flag=c["flag"], decoder=c["decoder"])
                  for (i, lab, s), c in rd["cells"].items()}) for rd in rep["readings"]])
        a.json.write_text(json.dumps(C.jsonable(out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
