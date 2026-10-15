"""D-031 report: the phase-1 interim result — C1 on the frozen-v2 candidate (ysync + ld0,
`configs/itrea2-dev-rc-3way-ysync-ld0.yaml`, A9 run hash f345de0ab936) read on FOUR seeds: D-030's set `c1r-ysync-ld0`
(seeds 0-1) and D-031's set `c1r-ysync-ld0-s23` (seeds 2-3, `jobs/d031.jsonl`) as one reading, the two halves beside
(vault `50-decisions/D-031 …`; D-019's addendum of 2026-10-15 13:4x: "the same C1 set at seeds 2–3 so the phase-1 result
rests on four seeds"; Berk 13:5x "go ahead with recommendations"). The rules are D-022's (C1's hint and guard) and D-025
amendment 3's (the harm leg, band as D-030's amendment of 2026-10-15 reads it), fixed before any run; nothing is chosen
by eye, nothing here adopts anything, and the go / no-go on C1 is Berk's (D-019 addendum).

Usage (repository root, after `itrea build-runs`; untimed analysis: OMP_NUM_THREADS=1, never beside timed runs with
default thread counts):

    OMP_NUM_THREADS=1 .venv/bin/python tools/d031_report.py [--failed both|exclude|keep] [--json out] [--d030-json F]
    OMP_NUM_THREADS=1 .venv/bin/python tools/d031_report.py --write-jobs jobs/d031.jsonl

There is no `itrea d031-report` entry (adding one is an edit of `src/itrea/cli.py`; the script is the entry).
`--write-jobs` writes D-031's job file with ONE call of D-030's writer, `make_jobs.c1_rerun("ysync-ld0", seeds=(2, 3),
set_="c1r-ysync-ld0-s23")` (`make-jobs c1r-rerun` has no `--seeds`): the 17 development + class-probe instances x
{A9@c1r, A6F@c1r} x seeds 2-3, 120 s, harness ext, split dev, one block per instance x seed = 68 runs in 34 blocks. Every
field equals D-030's job of the same instance and arm except seed / set / stratum / block / position; the labels are
D-030's, so the four seeds pool under one label per arm, and every run id is new by its seed alone (s2 / s3; the set
names only the run directory, `traces/raw/<set>/<id>`; D-029's precedent for its treatment, D-022's itx-r13f).
Positions alternate by seed as in D-030's file (A6F first at even seeds). The configuration path and the two run
hashes are pinned literals here (CONFIG, A9_HASH, A6F_HASH): the writer refuses when `make_jobs`' table moved, and
refuses to write any run id already in a job file under `jobs/` (but the target), in the run table or under
`traces/raw/`.

Reads `datasets/runs.parquet` (`pt` = P(120), `status`, `attempt`, `run_dir`, `harness`, `cuda_compiled_on_clock`,
`window_id`, `block_void`), each run's `run.json` / `ea.summary.json` (as `tools/c1r_report.py` reads them), D-022's
runs of set `itx-r13` (A9@r13x, A6F@r13x, seeds 0-1, 10-13: the harm leg's reference and band) and A0's runs of sets
`dev` / `classprobe` (seeds 0-1, 120 s, earlier days: vs SCIP alone, reported); each set's `void.jsonl` (beside).

Validity (gating; one verdict over the 136 runs; INVALID names the set and the leg and no 4-seed rule is read; a
missing run alone -> INCOMPLETE):
  - D-030's set: D-030's own validity, `c1r_report.reading` unchanged (six legs: complete 68 of 68 with P(120), harness
    ext, config = each run id its job's, no compile on the clock — CuPy only on A9 —, the two runs of a block in one
    window, each A9 run's CUDA flag as the candidate writes it: yield set, verified and held, timing events not
    blocking);
  - D-031's set: the same six legs on its 68 (`read_set`, the same functions of `c1r_report` / `c1_devtests`; its
    expected ids are this writer's);
  - VALID iff both are; INVALID if either is (the set and leg named); otherwise INCOMPLETE (n of 136).
  Each block runs in one window, so every A9 / A6F comparison is same-window per block. The halves are NOT
  day-separated: D-030's seeds 0-1 span 10-15 and 10-16, and D-031's seeds 2-3 run on 10-16, possibly in the same window
  as D-030's last blocks (the output names each half's windows and any shared one).

Rules (the definitions are D-022's and D-025 amendment 3's; this file only chooses which runs they read):
  - SEED MEANS ON FOUR SEEDS: P̄(i, arm) = (P_s0 + P_s1 + P_s2 + P_s3) / 4, each seed weighted equally — equal to the
    mean of the two halves' seed means; failed runs are outcomes of their arm under `keep` (D-002 §6).
  - HINT present iff the ratio of means of P(120) A9 / A6F over the FIXED 11 (D-022's C1_INSTANCES) on those seed means
    is <= 0.95; GUARD holds iff the mean bounded d of A9 vs A6F on those seed means over all 17 is >= -0.05 (both P = 1
    at d = 0). Computed by D-022's own `c1_hint` / `c1_guard` / `c1_interim` (`tools/d022_report.py`) on the rows
    relabelled to D-022's labels (`c1r_report._relabel`). D-022's functions check completeness on seeds 0-1 by name
    (`d022_report.SEEDS`) and average over the rows they are given (a groupby mean), so this report gates completeness
    on its own seeds first (the 11 x 2 arms x the reading's seeds for the hint, the 17 x … for the guard; INCOMPLETE,
    no partial mean, while one is missing) and hands D-022's functions the rows with the seeds renumbered from 0
    (2, 3 -> 0, 1 for the seeds 2-3 reading; seeds 0-3 unchanged), which changes nothing in their arithmetic.
  - HARM (D-025 amendment 3, cross-day, band as D-030's amendment fixes it): per instance of the 17, d_i =
    effect_d(P̄ A9@r13x over ITS seeds 0-1 (10-13), P̄ of the new A9 over the reading's seeds); band_i and the threshold
    from `c1r_report.harm_bands` — max(0.05, the bounded seed-0 vs seed-1 spread of A9@r13x, that of A6F@r13x), runs of
    10-13 only — the numbers of D-030's amendment table as of the 15:06 build, except that ca__Scenario400_reserves_1's
    reference is provisional (`best_of_study`) and every build-runs after a better ca solution re-scores every ca P,
    10-13's included (D-031: the governing numbers are those of the one build-runs after the last run); harm iff
    d_i < -band_i; cannot fire where the threshold >= 1 (rail01, cryptanalysiskb128n5obj16, doon, crna). Beside, not
    criteria: raw ΔP and D-022's floor marker (|ΔP| <= 0.005); per harm, the same-configuration A9 replicates' d of
    10-15 on the instance and cuOpt alone's (`c1r_report.harm_notes`); the cuOpt-alone null line (the same rule on
    A6F@c1r over the reading's seeds vs A6F@r13x's seeds 0-1); the same-configuration A9 replicates' null rate of
    D-030's reading (seeds 0-1 only).
  - Reported, not criteria (D-013 terms, `tools/d022_report.py`'s blocks): A9 vs A6F (same window per block, equal
    footprint) on the fixed 11; vs A0 (SCIP alone, seeds 0-1, earlier days) on the fixed 11, SCIP's weak set among them
    and SCIP's weak set among all 17 (`c1r_report.vs_a0_17`) — each over the instances with EVERY seed of both arms at
    the reading's seeds (a "partial: k of n" tag when some lack one; never a mix of 2- and 4-seed means).
  The same rules on seeds 0-1 alone (= D-030's reading; a line checks that the numbers equal `c1r_report`'s on the same
  run table, and `--d030-json F` compares them with the JSON saved when D-030 was read) and on seeds 2-3 alone are
  printed beside, with whether the halves agree (hint, guard, harm, the sign of d vs A6F per instance) — descriptive:
  no agreement threshold exists and none is applied.

THE PHASE-1 INTERIM RESULT is the four-seed HINT and GUARD when the 136 runs read VALID. Otherwise (D-031 INCOMPLETE or
INVALID) it is D-030's reading (seeds 0-1), and the four-seed and seeds 2-3 numbers that could be read are printed
beside, labelled "beside, not the result". The hint stays futility-only (D-024 rule 2); when the four seeds lead, the
line carries the seeds 2-3-alone ratio as D-024 rule 2's selection check (the four-seed ratio includes the half whose
hint let D-031 run). Failed runs: the reading under `keep` leads and prints THE line; the letter (`exclude`) is printed
beside when a run failed, its phase-1 line labelled as the letter. Output ends with "go / no-go is Berk's".
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.analysis import c1_devtests as C
from itrea.analysis.stats import effect_d
from itrea.experiments import make_jobs as mj
from itrea.experiments import pair
from itrea.paths import DATASETS, ROOT

_spec = importlib.util.spec_from_file_location("c1r_report_for_d031",
                                               Path(__file__).resolve().parent / "c1r_report.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
D22 = R.D22

A9C = "ysync-ld0"                                       # the frozen-v2 candidate D-030 runs (its 15:0x section)
CONFIG = "configs/itrea2-dev-rc-3way-ysync-ld0.yaml"    # pinned literals: the expected ids never move with make_jobs
A9_HASH, A6F_HASH = "f345de0ab936", "82a371305f88"      # A9@c1r on the candidate; A6F@c1r = A6F@r13x's job
SET_01, SET_23 = "c1r-ysync-ld0", "c1r-ysync-ld0-s23"    # D-030's set (seeds 0-1), D-031's (seeds 2-3)
SEEDS_01, SEEDS_23 = (0, 1), (2, 3)
SEEDS = SEEDS_01 + SEEDS_23
A9, A6F = R.A9, R.A6F                                   # A9@c1r, A6F@c1r: the same labels in both sets
LABELS = R.LABELS
BUDGET = R.BUDGET
C1_INSTANCES = R.C1_INSTANCES                           # the fixed 11 (D-022 second amendment)
JOBS_OUT = "jobs/d031.jsonl"
# jobs/d031.jsonl as written 2026-10-15, before any run (D-031)
JOBS_SHA256 = "f83e6888934ac3b2e243be880a8c823ae83b787462982c5d5f59b316dc9a3198"
N_RUNS = 136
BERK = ("this is the phase-1 interim result, not an adoption: the go / no-go on C1 is Berk's (D-019 addendum) — the "
        "hint and the guard are its inputs; any harm goes to him with the numbers, and the frozen-v2 candidate is not "
        "presumed: he decides which configuration freezes (D-025 amendment 3)")


def development_instances() -> list[str]:
    return R.development_instances()


# ----------------------------------------------------------------------------------------------- the job file
def pinned() -> None:
    """Refuse when make_jobs' candidate table no longer says what D-031 pinned (a later edit would move the ids)."""
    if tuple(mj.C1R_A9.get(A9C, ())) != (CONFIG, A9_HASH) or mj.C1R_A6F_HASH != A6F_HASH:
        raise ValueError(f"make_jobs' c1r table moved: {A9C} -> {mj.C1R_A9.get(A9C)}, A6F {mj.C1R_A6F_HASH}; D-031 "
                         f"pins ({CONFIG}, {A9_HASH}) and A6F {A6F_HASH}")


def jobs_23(set_: str = SET_23) -> list[dict]:
    """D-031's 68 jobs: one call of D-030's writer at seeds 2-3 (its own pins checked too), every run hash pinned."""
    pinned()
    jobs = mj.c1_rerun(A9C, seeds=SEEDS_23, set_=set_)
    for j in jobs:
        want = (CONFIG, A9_HASH) if j["arm"] == "A9" else (None, A6F_HASH)
        got = (j.get("config"), pair.run_hash(j))
        if got != want:
            raise ValueError(f"{j['name']} {j['arm']} s{j['seed']}: (config, run hash) {got}, not the pinned {want}")
    return jobs


def expected_23(set_: str = SET_23) -> dict:
    """{(instance, label, seed): run id} of D-031's jobs."""
    return {(j["name"], pair.arm_label(j), int(j["seed"])): pair.run_id_for(j) for j in jobs_23(set_)}


def taken_ids(root: Path, runs: pd.DataFrame | None, skip: Path | None = None) -> dict[str, set[str]]:
    """Run ids already used: every job file under jobs/ (but `skip`), the run table, the run directories."""
    jobs: set[str] = set()
    for f in sorted((Path(root) / "jobs").glob("*.jsonl")):
        if skip is not None and f.resolve() == Path(skip).resolve():
            continue
        for line in f.read_text().splitlines():
            try:
                jobs.add(pair.run_id_for(json.loads(line)))
            except Exception:                    # noqa: BLE001 — a job whose configuration is gone shares no id
                continue
    table = set() if runs is None else {C.bare_id(x) for x in runs["run_id"]}
    raw = {p.name for p in (Path(root) / "traces" / "raw").glob("*/*") if p.is_dir()}
    return dict(jobs=jobs, table=table, raw=raw)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path: Path, root: Path = ROOT, runs: pd.DataFrame | None = None) -> tuple[Path, str]:
    """Write D-031's job file; refuse when one of its run ids is already used anywhere."""
    jobs = jobs_23()
    ids = {pair.run_id_for(j) for j in jobs}
    if len(ids) != len(jobs) or len(jobs) != 68:
        raise ValueError(f"{len(jobs)} jobs, {len(ids)} distinct run ids (want 68 and 68)")
    hit = {k: sorted(ids & v) for k, v in taken_ids(root, runs, skip=path).items() if ids & v}
    if hit:
        raise ValueError(f"run ids already used: {hit}")
    p = mj.write_jobs(jobs, path)
    return p, sha256(p)


def job_file_state(root: Path = ROOT) -> dict:
    p = Path(root) / JOBS_OUT
    if not p.exists():
        return dict(state="absent", text=f"{JOBS_OUT} absent")
    h = sha256(p)
    return dict(state="same" if h == JOBS_SHA256 else "DIFFERS", sha256=h,
                text=f"{JOBS_OUT} sha256 {h[:12]}… " + ("= D-031's" if h == JOBS_SHA256 else
                                                         f"DIFFERS from D-031's {JOBS_SHA256[:12]}…"))


# ----------------------------------------------------------------------------------------------- validity
def read_set(runs: pd.DataFrame, root: Path, set_: str, seeds, exp: dict, keep_failed: bool, ext: dict) -> dict:
    """The six validity legs of `c1r_report.reading` on one set at `seeds` against the expected ids `exp` (D-031's set;
    for D-030's set the report calls `c1r_report.reading` itself — a test checks the two agree)."""
    sel, dropped = C.select(runs, set_, LABELS, seeds, BUDGET, keep_failed)
    by = {(r["instance"], r["arm"], int(r["seed"])): r for r in sel.to_dict("records")}
    outside = sorted(str(r["run_id"]) for k, r in by.items() if k not in exp)
    failed = C.failed_keys(runs, set_, LABELS, seeds, BUDGET)
    legs: dict[str, list[str]] = {k: [] for k in R.LEGS}
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
        if k[1] == A9 and R.cupy_compiled(rj) and not any(x.startswith(name) for x in legs["compile"]):
            legs["compile"].append(f"{name}: compiled on the clock (CuPy cache)")
        if not C.fin(r.get("pt")):
            legs["complete"].append(f"no P(120): {name}")
        if r.get("status") == "failed":
            kept.append(name)
        note = None
        if k[1] == A9:
            fp, note = R.flag_problems(A9C, r, summ)
            legs["flag"] += [f"{name}: {x}" for x in fp]
        cells[k] = dict(row=r, summ=summ, flag=R.flag_state(A9C, r, summ) if k[1] == A9 else None,
                        flag_note=note, decoder=R.decoder_state(summ) if k[1] == A9 else None)
    for i in development_instances():
        for s in seeds:
            have = [cells[(i, lab, s)] for lab in LABELS if (i, lab, s) in cells]
            if len(have) < 2:
                continue
            ws = [h["row"].get("window_id") for h in have]
            if not all(isinstance(w, str) and w for w in ws):
                legs["window"].append(f"{i} s{s}: a window id is missing")
            elif len(set(ws)) > 1:
                legs["window"].append(f"{i} s{s}: the two runs ran in different windows {sorted(set(ws))}")
    bad = [k for k in R.LEGS if legs[k]]
    verdict = "INVALID" if any(k != "complete" for k in bad) else ("INCOMPLETE" if bad else "VALID")
    val = dict(verdict=verdict, failing=bad, legs=legs,
               deviation=[f"{x} failed and is read as D-002 §6 keeps a crash (an outcome of its arm)" for x in kept])
    return dict(set=set_, seeds=list(seeds), keep_failed=keep_failed, n_expected=len(exp), n_present=len(cells),
                validity=val, duplicates=dropped, outside=outside, cells=cells)


def combined(parts: list[tuple[str, dict, int, int]]) -> dict:
    """One verdict over both sets: [(set, validity, n present, n expected)]."""
    invalid = [(s, v["failing"]) for s, v, _, _ in parts if v["verdict"] == "INVALID"]
    incomplete = [s for s, v, _, _ in parts if v["verdict"] == "INCOMPLETE"]
    verdict = "INVALID" if invalid else ("INCOMPLETE" if incomplete else "VALID")
    return dict(verdict=verdict, invalid=invalid, incomplete=incomplete, n_present=sum(p[2] for p in parts),
                n_expected=sum(p[3] for p in parts))


# ----------------------------------------------------------------------------------------------- the rules
def context(runs: pd.DataFrame) -> dict:
    """What every reading shares (the selections of `c1r_report.rules`): A0 on the fixed 11 and on the 17 (seeds 0-1,
    sets dev / classprobe), A9@r13x / A6F@r13x of 10-13 (seeds 0-1) and the harm bands fixed by them."""
    inst = development_instances()
    a0, _ = D22.select(runs, D22.A0_SETS, (D22.A0,), instances=list(C1_INSTANCES))
    a0_17, _ = D22.select(runs, D22.A0_SETS, (D22.A0,), instances=list(inst))
    old = C.select(runs, R.OLD_SET, (R.OLD_A9, R.OLD_A6F), R.SEEDS, BUDGET, keep_failed=True)[0]
    old = old[old["instance"].isin(list(inst))]
    po = R._p_by(old.to_dict("records"))
    p_old = {(i, s): v for (i, a, s), v in po.items() if a == R.OLD_A9}
    a6_old = {(i, s): v for (i, a, s), v in po.items() if a == R.OLD_A6F}
    bands = R.harm_bands(p_old, a6_old, inst)
    # beside each harm, never a criterion: the same-configuration A9 replicates of 10-15 against A9@r13x
    # (`c1r_report.null_lines`; an empty A6F@c1r side reads no cuOpt rows here — that line is `null_cuopt` per reading)
    nulls = R.null_lines(runs, p_old, bands, inst, a6_old, {})
    return dict(a0=a0, a0_17=a0_17, p_old=p_old, a6_old=a6_old, bands=bands, n_old=int(len(old)), nulls=nulls)


def renumbered(rows, seeds) -> pd.DataFrame:
    """D-022's long frame (`c1r_report._relabel`) of the rows at `seeds`, each seed renumbered by its place in `seeds`
    (seeds 0-3 unchanged; 2, 3 -> 0, 1): D-022's functions name seeds 0-1 in their own completeness check and average
    over whatever rows they get."""
    long = R._relabel([r for r in rows if int(r["seed"]) in seeds])
    idx = {s: k for k, s in enumerate(seeds)}
    long["seed"] = [idx[int(s)] for s in long["seed"]]
    return long


def missing(p: dict, instances, seeds) -> list[tuple]:
    """The (instance, label, seed) keys of A9@c1r / A6F@c1r at `seeds` without a P(120) in `p`."""
    return [(i, lab, s) for i in instances for lab in (A9, A6F) for s in seeds if (i, lab, s) not in p]


def gated(fn, long: pd.DataFrame, p: dict, instances, seeds, threshold: float) -> dict:
    """D-022's `c1_hint` / `c1_guard` on the reading's seeds: INCOMPLETE (no partial mean) while a run is missing,
    else D-022's own result with its labels mapped back (`c1r_report._ours`)."""
    miss = missing(p, instances, seeds)
    if miss or not instances:
        return dict(n=len(instances), threshold=threshold, missing=miss, state="INCOMPLETE", ratio=math.nan,
                    mean_d=math.nan, mean_P_a9=math.nan, mean_P_a6f=math.nan, seeds=list(seeds))
    return dict(R._ours(fn(long, list(instances))), seeds=list(seeds))


def cross_day(new: dict, old: dict, bands: dict, instances, seeds) -> dict:
    """D-025 amendment 3's rule with D-030's band, on the new arm's `seeds`: per instance, d_i = effect_d(P̄ old over
    its seeds 0-1 (10-13), P̄ new over `seeds`), harm iff d_i < -band_i; band_i / threshold / cannot-fire are
    `c1r_report.harm_bands`' (10-13's runs only). `new` / `old` {(instance, seed): P}. An instance without every seed
    of `new` or both of `old` is unread (INCOMPLETE)."""
    rows, harm, unread = [], [], []
    for i in instances:
        b = bands.get(i)
        pn, po = [new.get((i, s)) for s in seeds], R._seeds(old, i)
        if b is None or any(v is None for v in pn + po):
            unread.append(i)
            rows.append(dict(instance=i, read=False, p_new_seeds=pn, p_old_seeds=po, band=(b or {}).get("band"),
                             threshold=(b or {}).get("threshold"), cannot_fire=(b or {}).get("cannot_fire")))
            continue
        mo, mn = float(np.mean(po)), float(np.mean(pn))
        d = effect_d(mo, mn)
        h = bool(d < -b["band"])
        rows.append(dict(instance=i, read=True, p_old=mo, p_new=mn, p_old_seeds=po, p_new_seeds=pn, d=d,
                         band=b["band"], threshold=b["threshold"], cannot_fire=b["cannot_fire"], dp=mn - mo,
                         floor_dp=bool(abs(mn - mo) <= R.FLOOR_DP), harm=h))
        if h:
            harm.append(i)
    return dict(rows=rows, harm=harm, unread=unread, seeds=list(seeds),
                state="HARM" if harm else ("INCOMPLETE" if unread else "none"),
                cannot_fire=[i for i in instances if (bands.get(i) or {}).get("cannot_fire")])


def table(p: dict, instances, seeds) -> dict:
    """{instance: P̄ A6F, P̄ A9 and the bounded d of A9 vs A6F on the seed means} where every seed of both arms is in."""
    out = {}
    for i in instances:
        pa, p9 = [p.get((i, A6F, s)) for s in seeds], [p.get((i, A9, s)) for s in seeds]
        if any(v is None for v in pa + p9):
            continue
        ma, m9 = float(np.mean(pa)), float(np.mean(p9))
        out[i] = dict(p_a6f=ma, p_a9=m9, d=effect_d(ma, m9))
    return out


def rules_on(rows: list[dict], seeds, ctx: dict, gens: dict | None = None) -> dict:
    """C1's hint and guard (D-022's functions), the harm leg (cross-day vs A9@r13x's seeds 0-1, D-030's bands), the
    cuOpt-alone null line and the D-013 blocks on the rows at `seeds`."""
    inst = development_instances()
    rows = [r for r in rows if int(r["seed"]) in seeds]
    p = R._p_by(rows)
    long = renumbered(rows, seeds)
    # the D-013 blocks read only instances with EVERY seed of both arms at `seeds` (D-022's blocks average whatever
    # rows they get, so a half-run instance would enter on fewer seeds under the reading's label; review 2026-10-15)
    full = [i for i in inst if not missing(p, [i], seeds)]
    lf = long[long["instance"].isin(full)]
    c1 = D22.c1_interim(lf, ctx["a0"], list(C1_INSTANCES), list(inst))        # the D-013 blocks (hint / guard: below)
    new = {(i, s): v for (i, a, s), v in p.items() if a == A9}
    a6 = {(i, s): v for (i, a, s), v in p.items() if a == A6F}
    harm = cross_day(new, ctx["p_old"], ctx["bands"], inst, seeds)
    gens = gens or {}
    harm["cannot_fire_rows"] = [dict(instance=i, p_old_seeds=R._seeds(ctx["p_old"], i),
                                     p_new_seeds=[new.get((i, s)) for s in seeds],
                                     gens_new=[gens.get((i, s)) for s in seeds], band=ctx["bands"][i]["band"],
                                     threshold=ctx["bands"][i]["threshold"]) for i in harm["cannot_fire"]]
    null_cuopt = cross_day(a6, ctx["a6_old"], ctx["bands"], inst, seeds)
    # beside each harm (c1r_report.harm_notes): D-022's floor marker, the replicates' d of 10-15 on the instance (two
    # seeds against the same fixed A9@r13x reference) and cuOpt alone's d over these seeds — never a change to the list
    harm["notes"] = R.harm_notes(dict(harm=harm, nulls=dict(
        same_config=ctx["nulls"]["same_config"],
        cuopt=dict(rows=[dict(instance=r["instance"], d=r["d"]) for r in null_cuopt["rows"] if r.get("read")]))))
    return dict(seeds=list(seeds), n_runs=len(p),
                hint=gated(D22.c1_hint, long, p, list(C1_INSTANCES), seeds, D22.HINT_RATIO),
                guard=gated(D22.c1_guard, long, p, list(inst), seeds, D22.GUARD_MEAN),
                harm=harm, null_cuopt=null_cuopt,
                vs_a6f=c1[D22.A6F][D22.TREAT]["all"], vs_a0=c1[D22.A0],
                vs_a0_17=R.vs_a0_17(lf, ctx["a0_17"], inst), table=table(p, inst, seeds),
                d013_full=dict(n17=len(full), n11=sum(1 for i in full if i in C1_INSTANCES), instances=full))


def _close(a, b) -> bool:
    if not C.fin(a) or not C.fin(b):
        return (not C.fin(a)) and (not C.fin(b))
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def same_as_d030(s01: dict, ru: dict | None) -> dict:
    """Does this report's seeds 0-1 reading equal D-030's (`c1r_report.reading`'s rules) — hint, guard, harm?"""
    if ru is None:
        return dict(ok=None, text="D-030's reading not read (INVALID)")
    h, g = ru["hint"], ru["guard"]
    checks = dict(hint=(s01["hint"]["state"] == h["state"] and _close(s01["hint"].get("ratio"), h.get("ratio"))),
                  guard=(s01["guard"]["state"] == g["state"] and _close(s01["guard"].get("mean_d"), g.get("mean_d"))),
                  harm=(s01["harm"]["state"] == ru["harm"]["state"] and s01["harm"]["harm"] == ru["harm"]["harm"]))
    ok = all(checks.values())
    return dict(ok=ok, checks=checks, text="equal (hint, guard, harm list) — a code check on this one run table, not "
                "a check against the reading taken when D-030 was read (that is --d030-json)" if ok else
                f"DIFFERS on {[k for k, v in checks.items() if not v]} — a bug in this report; do not quote it")


def _num(x):
    """A number from c1r_report's JSON ("inf" / "-inf" strings, None for NaN)."""
    if isinstance(x, str) and x in ("inf", "-inf"):
        return math.inf if x == "inf" else -math.inf
    return math.nan if x is None else float(x)


def _g(x) -> str:
    """A saved / recomputed number for a difference line (6 significant digits: ca's moves are ~1e-4 in P)."""
    v = _num(x)
    return "–" if math.isnan(v) else f"{v:.6g}"


def _near(a, b, tol: float = 1e-9) -> bool:
    a, b = _num(a), _num(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return a == b or abs(a - b) <= tol


def vs_saved_d030(s01: dict | None, saved: dict, keep_failed: bool, path: str = "") -> dict:
    """Seeds 0-1 here against the JSON `tools/c1r_report.py --a9 ysync-ld0 --json F` saved when D-030 was read: the
    hint (state, ratio), the guard (state, mean d), the harm list, and per instance the threshold and A9@r13x's seeds.
    `same_as_d030` compares two codes on ONE run table; this compares two run tables. Every difference is printed with
    both numbers. ca__Scenario400_reserves_1's reference is provisional (`best_of_study`, not frozen): each build-runs
    after a run that finds a better ca solution re-scores every ca P, 10-13's included (+0.0004 to +0.0010 seen on
    2026-10-15; its threshold 0.4142 -> 0.4151), so small moves on ca — and through ca on the hint ratio and the guard's
    mean d — are expected; anything else is a data change or a bug (D-031, Rules: the governing numbers are those of
    the one build-runs after the last run)."""
    if s01 is None:
        return dict(ok=None, diffs=[], text="not compared: seeds 0-1 not read here (D-030's set INVALID)")
    if saved.get("set") != SET_01 or saved.get("a9") != A9C:
        return dict(ok=False, diffs=[], text=f"{path}: not D-030's reading (set {saved.get('set')}, a9 "
                                             f"{saved.get('a9')}; want {SET_01}, {A9C}) — not compared")
    rds = saved.get("readings") or []
    rd = next((x for x in rds if bool(x.get("keep_failed")) == bool(keep_failed)), None)
    if rd is None or not rd.get("rules"):
        return dict(ok=False, diffs=[], text=f"{path}: no {'keep' if keep_failed else 'exclude'} reading with rules "
                                             "saved — not compared")
    ru, diffs = rd["rules"], []
    for name, key in (("hint", "ratio"), ("guard", "mean_d")):
        a, b = ru[name], s01[name]
        if a.get("state") != b["state"] or not _near(a.get(key), b.get(key)):
            diffs.append(f"{name} {a.get('state')} {_g(a.get(key))} (saved) vs {b['state']} {_g(b.get(key))} (now)")
    if list(ru["harm"].get("harm") or []) != list(s01["harm"]["harm"]):
        diffs.append(f"harm {ru['harm'].get('harm')} (saved) vs {s01['harm']['harm']} (now)")
    now = {r["instance"]: r for r in s01["harm"]["rows"]}
    for r in ru["harm"].get("rows") or []:
        n = now.get(r.get("instance"))
        if n is None:
            continue
        if not _near(r.get("threshold"), n.get("threshold")):
            diffs.append(f"{r['instance']} threshold {_g(r.get('threshold'))} (saved) vs {_g(n.get('threshold'))} "
                         "(now)")
        po, pn = r.get("p_old_seeds") or [], n.get("p_old_seeds") or []
        if len(po) != len(pn) or not all(_near(x, y) for x, y in zip(po, pn)):
            diffs.append(f"{r['instance']} A9@r13x seeds {' / '.join(map(_g, po))} (saved) vs "
                         f"{' / '.join(map(_g, pn))} (now)")
    ok = not diffs
    return dict(ok=ok, diffs=diffs, text=(f"equal to the reading saved in {path} (hint, guard, harm list, thresholds, "
                                          "A9@r13x's seeds)") if ok else
                (f"DIFFERS from the reading saved in {path}: " + "; ".join(diffs) + " — ca's reference is "
                 "best_of_study (re-scored by every build-runs): small moves there are expected; any other is a data "
                 "change or a bug"))


def _sign(x: float) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def halves(s01: dict, s23: dict) -> dict:
    """Do the halves agree? Reported, no threshold: each half's hint / guard state and harm list, and per instance the
    sign of the bounded d of A9 vs A6F in each half (same sign / opposite / a zero)."""
    rows = []
    for i in development_instances():
        a, b = s01["table"].get(i), s23["table"].get(i)
        if a and b:
            sa, sb = _sign(a["d"]), _sign(b["d"])
            rows.append(dict(instance=i, d01=a["d"], d23=b["d"],
                             agree="same" if sa == sb else ("opposite" if sa * sb < 0 else "a zero")))
    count = {k: sum(1 for r in rows if r["agree"] == k) for k in ("same", "opposite", "a zero")}
    st = lambda x: x["state"]                                                         # noqa: E731
    return dict(rows=rows, n=len(rows), count=count,
                hint=(st(s01["hint"]), st(s23["hint"])), guard=(st(s01["guard"]), st(s23["guard"])),
                harm=(list(s01["harm"]["harm"]), list(s23["harm"]["harm"])),
                hint_agree=st(s01["hint"]) == st(s23["hint"]) and "INCOMPLETE" not in (st(s01["hint"]),
                                                                                      st(s23["hint"])),
                guard_agree=st(s01["guard"]) == st(s23["guard"]) and "INCOMPLETE" not in (st(s01["guard"]),
                                                                                         st(s23["guard"])),
                opposite=[r["instance"] for r in rows if r["agree"] == "opposite"])


def phase1(valid: dict, four: dict | None, s01: dict | None, d030_verdict: str) -> dict:
    """THE phase-1 interim result (D-031): the four-seed hint and guard when the 136 runs read VALID; otherwise D-030's
    reading (seeds 0-1), D-031 beside."""
    if valid["verdict"] == "VALID" and four is not None:
        return dict(source="four seeds 0-3 (D-030 + D-031, 136 runs)", seeds=list(SEEDS), hint=four["hint"],
                    guard=four["guard"], harm=four["harm"], fallback=False)
    if s01 is None or d030_verdict == "INVALID":
        return dict(source="none: D-030's set is INVALID", seeds=[], hint=None, guard=None, harm=None, fallback=True)
    why = "; ".join([f"{s} INVALID ({', '.join(legs)})" for s, legs in valid["invalid"]]
                    + [f"{s} INCOMPLETE" for s in valid["incomplete"]]) \
        + f"; {valid['n_present']} of {valid['n_expected']} runs"
    return dict(source=f"D-030's reading, seeds 0-1 ({why}; D-031's runs beside)", seeds=list(SEEDS_01),
                hint=s01["hint"], guard=s01["guard"], harm=s01["harm"], fallback=True)


# ----------------------------------------------------------------------------------------------- one reading
def reading(runs: pd.DataFrame, root: Path, keep_failed: bool, ext: dict, ctx: dict, exp23: dict,
            saved: tuple[str, dict] | None = None) -> dict:
    d030 = R.reading(runs, root, A9C, SET_01, keep_failed, ext)          # D-030's own validity and rules (seeds 0-1)
    v23 = read_set(runs, root, SET_23, SEEDS_23, exp23, keep_failed, ext)
    valid = combined([(SET_01, d030["validity"], d030["n_present"], d030["n_expected"]),
                      (SET_23, v23["validity"], v23["n_present"], v23["n_expected"])])
    cells = {**d030["cells"], **v23["cells"]}                           # keys carry the seed: no overlap
    rows = [c["row"] for c in cells.values()]
    gens = {(k[0], k[2]): c["summ"].get("generations") for k, c in cells.items() if k[1] == A9}
    ok01 = d030["validity"]["verdict"] != "INVALID"
    ok23 = v23["validity"]["verdict"] != "INVALID"
    four = rules_on(rows, SEEDS, ctx, gens) if valid["verdict"] != "INVALID" else None
    s01 = rules_on(rows, SEEDS_01, ctx, gens) if ok01 else None
    s23 = rules_on(rows, SEEDS_23, ctx, gens) if ok23 else None
    out = dict(keep_failed=keep_failed, validity=valid, d030=d030, d031=v23, four=four, s01=s01, s23=s23,
               same_as_d030=same_as_d030(s01, d030["rules"]) if s01 is not None else None,
               halves=halves(s01, s23) if s01 is not None and s23 is not None else None,
               phase1=phase1(valid, four, s01, d030["validity"]["verdict"]),
               vs_saved_d030=vs_saved_d030(s01, saved[1], keep_failed, saved[0]) if saved is not None else None)
    return out


def _void_summary(runs: pd.DataFrame, set_: str, root: Path) -> dict:
    voids = D22.void_log(runs, [set_], Path(root))
    reasons: dict[str, int] = {}
    for v in voids:
        key = str(v.get("reason") or "unreadable").split(" (")[0]
        reasons[key] = reasons.get(key, 0) + 1
    return dict(n=len(voids), reasons=reasons, runs=[str(v.get("run_id")) for v in voids])


def report(runs: pd.DataFrame, root: Path = ROOT, failed: str = "both", d030_json: Path | None = None) -> dict:
    """`d030_json`: the JSON `tools/c1r_report.py --a9 ysync-ld0 --json F` wrote when D-030 was read (Friday); seeds
    0-1 here are compared with it (`vs_saved_d030`), beside."""
    saved = (str(d030_json), json.loads(Path(d030_json).read_text())) if d030_json is not None else None
    exp23 = expected_23()
    fails = []
    for set_, seeds in ((SET_01, SEEDS_01), (SET_23, SEEDS_23)):
        cand = C.candidates(runs, set_, LABELS, seeds, BUDGET)
        fails += sorted(str(r) for r in cand[cand["status"] == "failed"]["run_id"])
    modes = {"exclude": [False], "keep": [True], "both": [True, False] if fails else [True]}[failed]
    ext = C.ext_harness()
    ctx = context(runs)
    windows = {}
    for set_, seeds in ((SET_01, SEEDS_01), (SET_23, SEEDS_23)):
        w = C.candidates(runs, set_, LABELS, seeds, BUDGET)
        windows[set_] = {str(k): int(v) for k, v in w.groupby(w["window_id"].astype(str)).size().items()} \
            if len(w) and "window_id" in w else {}
    return dict(a9=A9C, config=CONFIG, run_hash=A9_HASH, a6f_hash=A6F_HASH, sets=[SET_01, SET_23],
                failed_runs=fails, windows=windows, n_old=ctx["n_old"],
                other_rows={s: C.others(runs, s, LABELS, sd, BUDGET) for s, sd in ((SET_01, SEEDS_01),
                                                                                  (SET_23, SEEDS_23))},
                voids={s: _void_summary(runs, s, root) for s in (SET_01, SET_23)},
                frozen_v2=R.frozen_v2_state(A9C, root), job_file=job_file_state(root),
                readings=[reading(runs, root, k, ext, ctx, exp23, saved) for k in modes])


# ----------------------------------------------------------------------------------------------- output
def _harm_line(h: dict, label: str) -> str:
    if h["harm"]:
        notes = h.get("notes") or {}
        body = "; ".join(f"{r['instance']} (d_i {r['d']:+.3f} < -{r['band']:.3f}; P̄ {C.f(r['p_old'])} -> "
                         f"{C.f(r['p_new'])}, fires > {R._fp(r['threshold'], 4)}, ΔP {r['dp']:+.4f}"
                         + (f" [beside: {'; '.join(notes[r['instance']])}]" if notes.get(r["instance"]) else "") + ")"
                         for r in h["rows"] if r.get("read") and r["harm"])
        return f"HARM ({label}) on {body} -> to Berk with these numbers"
    return (f"HARM ({label}): none" + (f" on the {len(h['rows']) - len(h['unread'])} read; INCOMPLETE, unread "
                                       f"{h['unread']}" if h["unread"] else f" on the {len(h['rows'])}"))


def _state_line(ru: dict, label: str) -> str:
    h, g, hm = ru["hint"], ru["guard"], ru["harm"]
    return (f"{label}: HINT {h['state']} ({C.f(h.get('ratio'))}); GUARD {g['state']} ({C.f(g.get('mean_d'))}); "
            f"HARM {', '.join(hm['harm']) if hm['harm'] else hm['state']}")


def _legs(L: list, name: str, v: dict, n_present: int, n_expected: int) -> None:
    L.append(f"  {name}: {v['verdict']}, {n_present} of {n_expected} runs present"
             + (f"; failing leg(s): {', '.join(v['failing'])}" if v["failing"] else ""))
    for leg in R.LEGS:
        for x in v["legs"][leg][:8]:
            L.append(f"    ({leg}) {x}")
        if len(v["legs"][leg]) > 8:
            L.append(f"    ({leg}) … {len(v['legs'][leg]) - 8} more")
    for x in v["deviation"]:
        L.append(f"    kept: {x}")


def _rules_block(L: list, ru: dict, label: str) -> None:
    L.append(R._hint_line(ru["hint"], label))
    L.append(R._guard_line(ru["guard"], label))
    L.append(_harm_line(ru["harm"], f"{label}; cross-day vs A9@r13x's seeds 0-1 of 10-13"))
    cu = ru["null_cuopt"]
    L.append(f"  beside, null line: cuOpt alone ({A6F} over these seeds vs {R.OLD_A6F}'s seeds 0-1, the identical job "
             f"on other days) fires on {len(cu['harm'])} of {len(cu['rows']) - len(cu['unread'])}"
             + (f" {cu['harm']}" if cu["harm"] else ""))


def render(rep: dict) -> str:
    inst = development_instances()
    L = [f"D-031: the phase-1 interim result — C1 on the frozen-v2 candidate ({rep['config']}, A9 run hash "
         f"{rep['run_hash']}; A6F@c1r = A6F@r13x's job, {rep['a6f_hash']}), 17 instances x {{A9@c1r, A6F@c1r}} x seeds "
         f"0-3, 120 s, ext: sets {SET_01} (D-030, seeds 0-1) + {SET_23} (D-031, seeds 2-3); rules fixed before any run "
         f"(D-022's hint / guard on four-seed means, D-025 amendment 3's harm leg with D-030's bands); development, "
         f"descriptive (D-010)"]
    L.append("seed means: P̄(i, arm) = the mean of the P(120) of its seeds (four: (s0 + s1 + s2 + s3) / 4, each seed "
             "equal = the mean of the two halves' seed means); A9@r13x (the harm reference) and A0 on their seeds 0-1")
    for s in rep["sets"]:
        w = rep["windows"].get(s) or {}
        vo = rep["voids"][s]
        L.append(f"beside (not gating): {s}: windows {w if w else 'none yet'}; voids {vo['n']}"
                 + (f" ({', '.join(f'{k} {v}' for k, v in sorted(vo['reasons'].items()))}; each voided block rerun "
                    f"whole)" if vo["n"] else "")
                 + (f"; rows no rule reads: {rep['other_rows'][s]}" if rep["other_rows"][s] else ""))
    L.append(f"beside (not gating): {rep['frozen_v2']['text']}; {rep['job_file']['text']}")
    if rep["failed_runs"]:
        L.append(f"failed runs: {', '.join(rep['failed_runs'])}")
    for rd in rep["readings"]:
        L.append("")
        how = "kept (D-002 §6; the reading that leads)" if rd["keep_failed"] else "excluded (the letter)"
        L.append(f"== reading: failed runs {how} ==")
        v = rd["validity"]
        L.append(f"VALIDITY {v['verdict']} (both sets): {v['n_present']} of {v['n_expected']} runs present"
                 + (f"; INVALID: {v['invalid']}" if v["invalid"] else "")
                 + (f"; INCOMPLETE: {v['incomplete']}" if v["incomplete"] and not v["invalid"] else ""))
        _legs(L, f"{SET_01} (D-030's own legs, c1r_report)", rd["d030"]["validity"], rd["d030"]["n_present"],
              rd["d030"]["n_expected"])
        _legs(L, f"{SET_23} (D-031)", rd["d031"]["validity"], rd["d031"]["n_present"], rd["d031"]["n_expected"])
        for d in rd["d030"]["duplicates"] + rd["d031"]["duplicates"]:
            L.append(f"  duplicate left out: {d['run_id']}")
        for x in rd["d030"]["outside"] + rd["d031"]["outside"]:
            L.append(f"  not one of the 136 (ignored): {x}")
        a9c = {k: c for k, c in {**rd["d030"]["cells"], **rd["d031"]["cells"]}.items() if k[1] == A9}
        if a9c:
            L.append(f"{A9}'s CUDA flag as written (yield, gating) on {sum(1 for c in a9c.values() if c['flag']['ok'])}"
                     f" of {len(a9c)} runs; beside (not gating): LP decode gate open on "
                     f"{sum(1 for c in a9c.values() if c['decoder']['gate'] == 'open')} of {len(a9c)}")
        four, s01, s23 = rd["four"], rd["s01"], rd["s23"]
        if four is None:
            L.append("FOUR-SEED RULES NOT READ: validity fails -> INVALID")
        else:
            L.append("per instance (seed means; d vs A6F: + = A9 better; harm d_i: the new A9 vs A9@r13x's seeds 0-1 "
                     "of 10-13, cross-day, band_i from A9@r13x and A6F@r13x (10-13), harm iff d_i < -band_i, i.e. iff "
                     "P̄ new > 'fires >'):")
            L.append("  " + "instance".ljust(28) + "0-3: P A6F".ljust(12) + "P A9".ljust(8) + "d".ljust(8)
                     + "| d 0-1".ljust(9) + "d 2-3".ljust(8) + "| harm d_i".ljust(11) + "band_i".ljust(8)
                     + "fires >".ljust(9) + "harm".ljust(13) + "| d_i 0-1".ljust(10) + "d_i 2-3")
            hm = {r["instance"]: r for r in four["harm"]["rows"]}
            h01 = {r["instance"]: r for r in (s01["harm"]["rows"] if s01 else [])}
            h23 = {r["instance"]: r for r in (s23["harm"]["rows"] if s23 else [])}
            for i in inst:
                g = four["table"].get(i)
                a = (s01 or {}).get("table", {}).get(i)
                b = (s23 or {}).get("table", {}).get(i)
                h = hm.get(i, {})
                cant = bool(h.get("cannot_fire"))
                thr = "> 1" if cant else R._fp(h.get("threshold"), 4)
                tag = (("HARM" + (" (floor)" if h["floor_dp"] else "")) if h["harm"] else
                       ("cannot fire" if cant else "-")) if h.get("read") else "unread"
                hd = f"{h['d']:+.3f}" if h.get("read") else "–"
                x01 = h01.get(i, {})
                x23 = h23.get(i, {})
                L.append(f"  {'*' if i in C1_INSTANCES else ' '}{i:<27}"
                         + (f"{C.f(g['p_a6f']):<12}{C.f(g['p_a9']):<8}{g['d']:+.3f}  " if g else "–".ljust(28))
                         + (f"| {a['d']:+.3f}  " if a else "| –".ljust(9)) + (f"{b['d']:+.3f}  " if b else "–".ljust(8))
                         + f"| {hd:<9}{R._fp(h.get('band')):<8}{thr:<9}{tag:<13}"
                         + (f"| {x01['d']:+.3f}   " if x01.get("read") else "| –".ljust(10))
                         + (f"{x23['d']:+.3f}" if x23.get("read") else "–"))
            L.append("  (* = the fixed 11 of the hint; 'fires >' = the smallest P̄ of the new A9 at which harm fires, "
                     "from 10-13's runs alone, on this run table — D-030's amendment table as of the 15:06 build; "
                     "ca__Scenario400_reserves_1's reference is best_of_study, so every build-runs after a better ca "
                     "solution re-scores its P and moves its threshold, 10-13's runs included)")
            L.append("SEEDS 0-3 (the four-seed reading" + (f" — {_not_result(rd)}):" if rd["phase1"]["fallback"]
                                                          else " — the phase-1 interim result's source):"))
            _rules_block(L, four, "seeds 0-3")
            if four["harm"]["cannot_fire"]:
                L.append(f"  harm CANNOT fire on {len(four['harm']['cannot_fire'])} of the 17 (threshold > 1 on 10-13's "
                         "runs) — beside, each seed's P and the EA's generations:")
                for r in four["harm"]["cannot_fire_rows"]:
                    L.append(f"    {r['instance']}: band_i {R._fp(r['band'])}, fires > {R._fp(r['threshold'], 3)}; P "
                             f"{' / '.join(C.f(x) for x in r['p_old_seeds'])} (10-13) -> "
                             f"{' / '.join(C.f(x) for x in r['p_new_seeds'])}; generations "
                             f"{' / '.join('–' if x is None else str(x) for x in r['gens_new'])}")
        for lab, ru in (("seeds 0-1 alone (= D-030's reading)", s01), ("seeds 2-3 alone (D-031's runs)", s23)):
            if ru is None:
                L.append(f"{lab.upper()}: not read (its set is INVALID)")
                continue
            L.append(f"{lab.upper()}:")
            _rules_block(L, ru, lab.split(" (")[0])
            if ru is s01 and rd["same_as_d030"] is not None:
                L.append(f"  seeds 0-1 here vs c1r_report's D-030 reading on this run table: "
                         f"{rd['same_as_d030']['text']}")
                vs = rd.get("vs_saved_d030")
                L.append("  seeds 0-1 here vs the D-030 reading saved when it was read: "
                         + (vs["text"] if vs is not None else "not compared (no --d030-json given)"))
                ru30 = rd["d030"]["rules"]
                if ru30 is not None:
                    nl, th = ru30["nulls"], ru30["then"]
                    gd = th["guard"].get("mean_d")
                    L.append(f"  beside (D-030's reading): same-configuration A9 replicates of 10-15 vs A9@r13x fire "
                             f"on {nl['k']} of {nl['n']} instance reads; 10-13 (A9@r13x / A6F@r13x, the old default): "
                             f"HINT {th['hint']['state']} ({C.f(th['hint'].get('ratio'))}), GUARD {th['guard']['state']} "
                             f"({f'{gd:+.3f}' if C.fin(gd) else '–'}); recorded in D-022's Result 0.827 / +0.063")
        hv = rd["halves"]
        if hv is not None:
            L.append(f"DO THE HALVES AGREE? (reported, no threshold): hint {hv['hint'][0]} / {hv['hint'][1]}"
                     f" ({'same' if hv['hint_agree'] else 'not the same or not both read'}); guard {hv['guard'][0]} / "
                     f"{hv['guard'][1]} ({'same' if hv['guard_agree'] else 'not the same or not both read'}); harm "
                     f"{hv['harm'][0] or 'none'} / {hv['harm'][1] or 'none'}; sign of d vs A6F on the {hv['n']} "
                     f"instances read in both: same {hv['count']['same']}, opposite {hv['count']['opposite']}"
                     + (f" {hv['opposite']}" if hv["opposite"] else "") + f", a zero {hv['count']['a zero']}")
            L.append("  " + _halves_windows(rep))
        if four is not None:
            pre = f"{_not_result(rd)}: " if rd["phase1"]["fallback"] else ""
            L.append(f"{pre}reported, not criteria (D-013 terms, seeds 0-3) — vs cuOpt alone (A6F@c1r, same window per "
                     "block, equal footprint): " + _d013(four, four["vs_a6f"], "the fixed 11", "instances", 11))
            c = four["vs_a0"]
            L.append(f"{pre}reported, not criteria (D-013 terms, seeds 0-3 vs A0's seeds 0-1, earlier days) — vs "
                     f"{c['label']}: " + _d013(four, c[D22.TREAT]["all"], "the fixed 11", "instances", 11))
            L.append(f"  SCIP's weak set among them (A0 P >= {D22.WEAK_P}) {c['weak_set']}: "
                     + _d013(four, c[D22.TREAT]["weak"], "weak set", "weak", 11))
            w = four["vs_a0_17"]
            L.append(f"  SCIP's weak set among all 17 (D-019's C1; {len(w['weak_set'])} of {w['n_baseline']} with A0 "
                     f"runs): " + _d013(four, w["block"], "weak set", "weak", 17))
            for lab, ru in (("seeds 0-1", s01), ("seeds 2-3", s23)):
                if ru is not None:
                    L.append(f"  beside, {lab}: vs A6F " + _d013(ru, ru["vs_a6f"], "the fixed 11", "instances", 11)
                             + "; vs A0 on SCIP's weak set among the 17 "
                             + _d013(ru, ru["vs_a0_17"]["block"], "weak set", "weak", 17))
        L.append(_phase1_line(rep, rd))
        L.append(f"  {BERK}")
    return "\n".join(L)


def _not_result(rd: dict) -> str:
    v = rd["validity"]
    return f"beside, not the result ({v['verdict']}, {v['n_present']} of {v['n_expected']} runs)"


def _d013(ru: dict, w: dict, label: str, of: str, total: int) -> str:
    """D-022's D-013 line on the instances that have every seed of both arms at the reading's seeds (`rules_on`), the
    left-out count named as such."""
    full = ru.get("d013_full") or {}
    k = full.get("n11" if total == 11 else "n17", total)
    sd = ru["seeds"]
    seeds = f"{sd[0]}-{sd[-1]}" if len(sd) > 1 else str(sd[0])
    tag = f" [partial: {k} of {total} instances have every seed {seeds} of both arms; only they are read]" \
        if k < total else ""
    return D22._d013_line(w, label + tag, of).replace(
        f" {of} without both arms' runs)", f" {of} without every seed {seeds} of both arms, left out)")


def _halves_windows(rep: dict) -> str:
    """The halves are not day-separated: each set's windows (the pool's window ids) and any window both share."""
    w01, w23 = sorted(rep["windows"].get(SET_01) or {}), sorted(rep["windows"].get(SET_23) or {})
    shared = sorted(set(w01) & set(w23))
    return (f"the halves are not day-separated (descriptive): seeds 0-1 ran in window(s) {w01 or 'none yet'}, seeds "
            f"2-3 in {w23 or 'none yet'}" + (f"; shared {shared}" if shared else "")
            + " — a difference between the halves mixes seed, day and window")


def _phase1_line(rep: dict, rd: dict) -> str:
    """THE line — only for the reading that leads (`keep`, or the only reading when no run failed); the letter's
    (failed runs excluded) is labelled beside so that one line reads PHASE-1 INTERIM RESULT."""
    p1 = rd["phase1"]
    lead = rd["keep_failed"] or not rep["failed_runs"]
    head = "PHASE-1 INTERIM RESULT" if lead else ("the letter (failed runs excluded), beside — not the phase-1 "
                                                  "interim result, which is the kept reading's line")
    if p1["hint"] is None:
        return f"{head}: {p1['source']} — no result"
    line = (f"{head} ({p1['source']}): HINT {p1['hint']['state']} (ratio {C.f(p1['hint'].get('ratio'))}, present iff "
            f"<= 0.95; futility-only, D-024 rule 2: a hint never declares success), GUARD {p1['guard']['state']} (mean "
            f"d {C.f(p1['guard'].get('mean_d'))}, holds iff >= -0.05); beside: harm "
            f"{', '.join(p1['harm']['harm']) if p1['harm']['harm'] else p1['harm']['state']}")
    s23 = rd.get("s23")
    if not p1["fallback"] and s23 is not None:
        h = s23["hint"]
        line += (f"; selection check (D-024 rule 2, the stage-2-only figure): seeds 2-3 alone HINT {h['state']} "
                 f"(ratio {C.f(h.get('ratio'))}) — the four-seed ratio includes seeds 0-1, whose hint let D-031 run, "
                 f"so it is optimistic given that")
    return line


def _json(rep: dict) -> dict:
    def strip(rd: dict) -> dict:
        out = {k: v for k, v in rd.items() if k not in ("d030", "d031")}
        for name in ("d030", "d031"):
            x = rd[name]
            out[name] = {k: v for k, v in x.items() if k != "cells"} | dict(
                runs={f"{i} {lab} s{s}": dict(run_id=c["row"]["run_id"], pt=c["row"].get("pt"),
                                              status=c["row"].get("status"), window_id=c["row"].get("window_id"),
                                              flag=c["flag"], decoder=c["decoder"])
                      for (i, lab, s), c in x["cells"].items()})
        return out
    return C.jsonable(dict(rep, readings=[strip(rd) for rd in rep["readings"]]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="d031_report", description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=DATASETS / "runs.parquet")
    ap.add_argument("--root", type=Path, default=ROOT, help="the run directories' root (run_dir is relative to it)")
    ap.add_argument("--failed", choices=("both", "exclude", "keep"), default="both")
    ap.add_argument("--json", type=Path, help="write the report as JSON here too")
    ap.add_argument("--d030-json", type=Path, default=None, metavar="F",
                    help="the JSON `tools/c1r_report.py --a9 ysync-ld0 --json F` saved when D-030 was read: seeds 0-1 "
                         "here are compared with it (beside; ca's best_of_study reference is re-scored by build-runs)")
    ap.add_argument("--write-jobs", type=Path, default=None, metavar="PATH",
                    help=f"write D-031's job file ({JOBS_OUT}: set {SET_23}, 68 runs) and print its sha256")
    a = ap.parse_args(argv)
    try:
        runs = pd.read_parquet(a.runs) if (a.write_jobs is None or a.runs.exists()) else None
        if a.write_jobs is not None:
            p, h = write(a.write_jobs, a.root, runs)
            print(f"wrote {p}: 68 runs, set {SET_23}, sha256 {h}")
            return 0
        rep = report(runs, a.root, a.failed, a.d030_json)
    except (OSError, ValueError, mj.JobError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(render(rep))
    if a.json:
        a.json.write_text(json.dumps(_json(rep), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
