"""D-029 report: the last R1 test for v1 — fallback (b) `ysync` (CU_CTX_SCHED_YIELD) against the A9 development default,
its guards judged on every seed run so far (vault `50-decisions/D-029 …`; Berk 2026-10-15, question panel: "Follow-up,
guards on all seeds (Recommended)"). The rules are D-029's as amended 2026-10-15 12:2x, fixed before any run of set
`r1f-ysync`; nothing is chosen by eye, and this file implements exactly what the record says.

Usage (repository root, after `itrea build-runs`; untimed analysis, never beside timed runs with default thread counts):

    OMP_NUM_THREADS=1 .venv/bin/python tools/d029_report.py [--failed both|exclude|keep] [--json out]
    OMP_NUM_THREADS=1 .venv/bin/python tools/d029_report.py --write-jobs jobs/d029.jsonl
    OMP_NUM_THREADS=1 .venv/bin/python tools/d029_report.py --write-jobs jobs/d029-2.jsonl --set r1f-ysync-2 \\
        --blocks item_placement_9903:s3,comp21-2idx:s5                     (a rerun, the amendment's rule)

There is no `itrea d029-report` entry: adding one is an edit of `src/itrea/cli.py` (owed; the script is the entry until
then). `--write-jobs` writes D-029's job file: `make-jobs r1-bsync` fixes the control label of a fallback set at
`r1ctl-<variant>` and has no option for another, so the jobs are built here from `make_jobs`' own pieces (`job`,
`_alternated`, `instance_paths`, `_sorted`, `write_jobs`) — the same fields, configuration files, harness and order rule
as `make_jobs.r1_bsync`. The configuration paths and the run hashes are pinned literals (the amendment): the report's
expected run ids do not move when `make_jobs`' default constants or a configuration file change later, and
`--write-jobs` refuses to write a job whose run id differs from the pinned one.

The job file (amendment): one warm-up run first — the control's configuration on item_placement_9903 at seed 6, label
A9@r1fwarm, set `<set>-warmup`, `priority` 0 — then the read runs at `priority` 1, so whatever disturbs a pool's start
(D-025's ysync seed 1) lands on a run no rule reads. The warm-up is never read.

The runs the rules read (28):
  - set `r1-ysync` (D-025's fallback (b), 2026-10-15 10:00–11:09), seeds 0–1, item_placement_9903 and comp21-2idx x
    {A9@ysync, A9@r1ctl-ysync} = 8 runs;
  - set `r1f-ysync` (`jobs/d029.jsonl`): item_placement_9903 and comp21-2idx at seeds 2–5, ca at seeds 2–3 x
    {A9@ysync, A9@r1fctl} = 20 runs, 10 blocks; A9@r1fctl = the development default (configs/itrea2-dev-rc-3way.yaml,
    run hash 28e5a5e9c725 = A9@r13x's) under a label of its own, so every run id is new. A block present in a rerun set
    `r1f-ysync-<N>` (N >= 2; the latest N) is read from there instead, both arms.
  Read beside rule 3, not by it and not validity-gated: r1-ysync's ca runs at seeds 0–1 (4).
  T = A9@ysync in both sets; C = A9@r1ctl-ysync (seeds 0–1) and A9@r1fctl (seeds 2–5). Each block holds T and C at
  one seed in one window; the arms' order alternates over the seeds (the control first at even seeds).

Reads `datasets/runs.parquet` (`pt` = P(120) on the call clock, `status`, `attempt`, `run_dir`, `harness`,
`cuda_compiled_on_clock`, `window_id`, `block_void`) and each run's `run.json` (`config.harness`,
`compiled_on_clock_cuda`, `created_epoch`, `ended_epoch`, `gates_before.loadavg_1m`) and `ea.summary.json` (`P`,
`gene_dtype`, `compiled_on_clock`, `cuda_sched`, `phase_events_block`, `host_renice`, `generations`, `wall_s`,
`ls_launches`, `cpu_clock`, `lp`).

Validity (D-025's legs on the 28, plus the amendment's window leg): complete (28 of 28 present); harness ext; config
(each run id is its job's, pinned); no compile on the clock; equal P and gene dtype per block; the two runs of a block
in one window (`window_id`); the flag state (T: yield set, verified and held, timing events not blocking; C: CUDA's
default read back, no renice). A failed leg among the present runs -> INVALID; otherwise a missing run -> INCOMPLETE
(INVALID dominates). Failed runs (`--failed`): a failed TREATMENT run is an outcome of its arm (D-002 §6, `keep`, the
verdict's reading) and a figure it cannot give (no EA summary -> no G) fails the rule that needs it; a failed CONTROL
run never decides a rule (the amendment): a rule that fails with it in still fails (a truncated control only looks worse,
so the treatment could only gain from it), otherwise the rule is unread -> INCOMPLETE and the block is rerun, both arms,
as a rerun set. The letter (`exclude`: a failed run is not completed, so the set is INCOMPLETE) is printed beside.

Rules (gating; thresholds fixed in D-029 before any r1f-ysync run, rule 3 as amended):
  1. generation guard on item_placement and on comp21 over seeds 0–5 (n = 6 per arm): Ḡ(T) / Ḡ(C) >= 0.90 (ratio of
     means; the ratio of medians, each seed, and seeds 0–1 / 2–5 — the two windows — separately reported beside);
  2. P no-harm on item_placement and on comp21 over seeds 0–5: P̄(T) − P̄(C) <= 0.05 (raw P, one-sided); a paired
     bootstrap one-sided 95 % upper bound beside, not gating (resampling the 6 blocks = seeds with replacement, B =
     10,000, numpy default_rng(29), the 0.95 quantile of the resampled mean difference);
  3. gain replication on ca at seeds 2–3 alone (n = 2 per arm): ΣP(T) / ΣP(C) <= 0.95; each seed and the pooled seeds
     0–3 beside.
Verdict: NOT ADOPTED if a rule fails — final for v1 (R1 -> v1.1); otherwise INCOMPLETE while a rule is unread for a
failed control run (rerun the named blocks); otherwise ADOPT — ysync becomes A9's (and A3x's) development default and C1
is re-run on the frozen hash (68 runs, A6F in the same window). No further R1 follow-up for v1 whatever happens.
Reported, not gating: the per-run mechanism (D-025's) with generations per second and the load average at the run's
gates, and each pool's first run (the warm-up when the job file has one) with the gap since the previous pool in this
checkout's run table.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.analysis import c1_devtests as C
from itrea.experiments import make_jobs as mj
from itrea.experiments import pair
from itrea.paths import DATASETS, ROOT

_spec = importlib.util.spec_from_file_location("d025_report_for_d029", Path(__file__).resolve().parent / "d025_report.py")
D25 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D25)

ITEM, COMP, CA = "item_placement_9903", "comp21-2idx", "ca__Scenario400_reserves_1"
INSTANCES = (ITEM, COMP, CA)
VARIANT = "ysync"
SET_OLD, SET_NEW = "r1-ysync", "r1f-ysync"
WARMUP_SUFFIX = "-warmup"
RERUN_RE = re.compile(r"^r1f-ysync-(\d+)$")                # r1f-ysync-2, -3, …: reruns of whole r1f-ysync blocks
T_LABEL = "A9@ysync"
C_OLD, C_NEW_VARIANT = "A9@r1ctl-ysync", "r1fctl"
C_NEW = "A9@" + C_NEW_VARIANT
WARMUP_VARIANT = "r1fwarm"
WARMUP_LABEL = "A9@" + WARMUP_VARIANT
WARMUP_INSTANCE, WARMUP_SEED = ITEM, 6
PRIORITY_WARMUP, PRIORITY_READ = 0, 1
BUDGET = 120.0
OLD_SEEDS = (0, 1)
NEW_SEEDS = {ITEM: (2, 3, 4, 5), COMP: (2, 3, 4, 5), CA: (2, 3)}
GUARD_INSTANCES, GUARD_SEEDS = (ITEM, COMP), (0, 1, 2, 3, 4, 5)
CA_SEEDS, CA_BESIDE_SEEDS = (2, 3), (0, 1)
G_MIN, P_MAX, CA_MAX = 0.90, 0.05, 0.95
BOOT_B, BOOT_RNG, BOOT_Q = 10_000, 29, 0.95
JOBS_OUT = "jobs/d029.jsonl"
LEGS = C.BASE_LEGS + ("window", "flag")
# Pinned (the amendment): the development default and ysync's file as of 2026-10-15, and their runs' run hashes.
CONTROL_CONFIG = "configs/itrea2-dev-rc-3way.yaml"         # A9@r13x's configuration, the control
TREATMENT_CONFIG = "configs/itrea2-dev-rc-3way-ysync.yaml"  # D-025's fallback (b), the treatment
RUN_HASH = {T_LABEL: "a08221cca653", C_OLD: "28e5a5e9c725", C_NEW: "28e5a5e9c725", WARMUP_LABEL: "28e5a5e9c725"}


# ----------------------------------------------------------------------------------------------- jobs
def run_id(inst: str, label: str, seed: int) -> str:
    """The pinned run id of (instance, label, seed) (`pair.run_id_for`'s form with the pinned hash)."""
    return f"{inst}__{label}__s{int(seed)}__{pair.fmt_budget(BUDGET)}s__{RUN_HASH[label]}"


def _new_blocks(blocks=None) -> list[tuple[str, int]]:
    want = [(n, s) for n, seeds in NEW_SEEDS.items() for s in seeds]
    if blocks is None:
        return want
    bad = sorted(set(blocks) - set(want))
    if bad:
        raise ValueError(f"not a block of {SET_NEW}: {bad} (blocks: {want})")
    return [b for b in want if b in set(blocks)]


def check_set(set_: str) -> str:
    if set_ != SET_NEW and not (RERUN_RE.match(set_) and int(RERUN_RE.match(set_).group(1)) >= 2):
        raise ValueError(f"--set: {SET_NEW} or a rerun set {SET_NEW}-<N> with N >= 2, not {set_!r}")
    return set_


def d029_jobs(set_: str = SET_NEW, blocks=None, tables=mj.INSTANCE_TABLES) -> list[dict]:
    """D-029's read runs: item_placement_9903 and comp21-2idx at seeds 2–5, ca at seeds 2–3 x {A9@r1fctl (the
    development default, configs/itrea2-dev-rc-3way.yaml), A9@ysync (-3way-ysync.yaml)}, 120 s, the ext harness, split
    dev, set `set_` (r1f-ysync; a rerun set holds only `blocks`, {(instance, seed)}), one block per instance x seed, the
    control first at even seeds (`make_jobs._alternated`), priority 1 (after the warm-up)."""
    want = _new_blocks(blocks)
    where = mj.instance_paths(sorted({n for n, _ in want}), tables)
    out = []
    for n, s in want:
        common = dict(name=n, stratum=set_, split="dev", block=f"{n}__s{int(s)}__{set_}", harness=mj.ITX_EXT,
                      priority=PRIORITY_READ)
        out += mj._alternated(s, mj.job(where[n], set_, "A9", s, BUDGET, config=CONTROL_CONFIG,
                                        variant=C_NEW_VARIANT, **common),
                              mj.job(where[n], set_, "A9", s, BUDGET, config=TREATMENT_CONFIG, variant=VARIANT,
                                     **common))
    return mj._sorted(out)


def warmup_job(set_: str = SET_NEW, tables=mj.INSTANCE_TABLES) -> dict:
    """The pool's first run (the amendment): the control's configuration on item_placement_9903 at seed 6 under a label
    of its own (A9@r1fwarm), set `<set_>-warmup`, priority 0. Never read."""
    ws = set_ + WARMUP_SUFFIX
    where = mj.instance_paths([WARMUP_INSTANCE], tables)
    return mj.job(where[WARMUP_INSTANCE], ws, "A9", WARMUP_SEED, BUDGET, config=CONTROL_CONFIG,
                  variant=WARMUP_VARIANT, name=WARMUP_INSTANCE, stratum=ws, split="dev",
                  block=f"{WARMUP_INSTANCE}__s{WARMUP_SEED}__{ws}", harness=mj.ITX_EXT, priority=PRIORITY_WARMUP)


def job_file(set_: str = SET_NEW, blocks=None, tables=mj.INSTANCE_TABLES) -> list[dict]:
    """The job file's lines: the warm-up and the read runs."""
    check_set(set_)
    return mj._sorted([warmup_job(set_, tables)] + d029_jobs(set_, blocks, tables))


def pinned_problems(jobs: list[dict]) -> list[str]:
    """Jobs whose run id (from today's configuration files) is not the pinned one."""
    out = []
    for j in jobs:
        rid, want = pair.run_id_for(j), run_id(j["name"], pair.arm_label(j), j["seed"])
        if rid != want:
            out.append(f"{rid} is not the pinned {want} (a configuration file changed since D-029 was fixed)")
    return out


def parse_blocks(text: str) -> list[tuple[str, int]]:
    """'item_placement_9903:s3,comp21-2idx:s5' -> [(instance, seed), …]."""
    out = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        inst, _, s = part.rpartition(":")
        if not inst or not s.lstrip("s").isdigit():
            raise ValueError(f"--blocks: <instance>:s<seed>, not {part!r}")
        out.append((inst, int(s.lstrip("s"))))
    return out


# ----------------------------------------------------------------------------------------------- the runs read
def rerun_sets(runs: pd.DataFrame) -> list[str]:
    """The rerun sets in the table, `r1f-ysync-<N>` with N >= 2, by N (never `…-warmup`)."""
    if "set" not in runs:
        return []
    found = {str(s) for s in runs["set"].dropna().unique()}
    keep = [s for s in found if RERUN_RE.match(s) and int(RERUN_RE.match(s).group(1)) >= 2]
    return sorted(keep, key=lambda s: int(RERUN_RE.match(s).group(1)))


def rerun_sources(runs: pd.DataFrame | None) -> dict:
    """{(instance, seed): the latest rerun set holding a row of that r1f-ysync block (either arm; not synthetic, not in
    a voided block)}."""
    out = {}
    if runs is None:
        return out
    for set_ in rerun_sets(runs):
        for inst, seeds in NEW_SEEDS.items():
            cand = C.candidates(runs, set_, (T_LABEL, C_NEW), seeds, BUDGET)
            for r in cand[cand["instance"] == inst].to_dict("records"):
                out[(inst, int(r["seed"]))] = set_
    return out


def expected(runs: pd.DataFrame | None = None) -> dict:
    """{(set, instance, label, seed): run id} of the 28 runs the rules read: r1-ysync's item_placement and comp21 at
    seeds 0–1 (8) and r1f-ysync's 20, each r1f-ysync block read from its latest rerun set when `runs` holds one."""
    src = rerun_sources(runs)
    out = {}
    for inst in GUARD_INSTANCES:
        for lab in (T_LABEL, C_OLD):
            for s in OLD_SEEDS:
                out[(SET_OLD, inst, lab, s)] = run_id(inst, lab, s)
    for inst, s in _new_blocks():
        set_ = src.get((inst, s), SET_NEW)
        for lab in (C_NEW, T_LABEL):
            out[(set_, inst, lab, s)] = run_id(inst, lab, s)
    return out


def beside_keys() -> dict:
    """{key: run id} of r1-ysync's ca runs at seeds 0–1: printed beside rule 3, not read by it, not validity-gated."""
    return {(SET_OLD, CA, lab, s): run_id(CA, lab, s) for lab in (T_LABEL, C_OLD) for s in CA_BESIDE_SEEDS}


def role(label: str) -> str:
    return "T" if label == T_LABEL else "C"


# ----------------------------------------------------------------------------------------------- start of pool
def pool_starts(runs: pd.DataFrame, windows, root: Path, read_ids: set) -> dict:
    """{window_id: the pool's first run (earliest run.json `created_epoch` among the table's rows of that window, voided
    blocks left out), whether a rule reads it, the load average at its gates, and the gap since the latest `ended_epoch`
    of the previous window in the table}. Reported, not gating."""
    out = {}
    if "window_id" not in runs:
        return out
    allw = sorted({str(x) for x in runs["window_id"].dropna() if isinstance(x, str) and x})
    rows = runs
    if "block_void" in runs:
        rows = runs[~runs["block_void"].fillna(False).astype(bool)]
    for w in sorted({str(x) for x in windows if isinstance(x, str) and x}):
        best = None
        for r in rows[rows["window_id"].astype(str) == w].to_dict("records"):
            rj, _ = C.run_files(r, root)
            t = rj.get("created_epoch")
            if C.fin(t) and (best is None or float(t) < best[0]):
                best = (float(t), str(r["run_id"]), rj)
        if not best:
            continue
        prev = [x for x in allw if x < w]
        gap, prev_w = None, (prev[-1] if prev else None)
        if prev_w:
            ends = [C.run_files(r, root)[0].get("ended_epoch")
                    for r in runs[runs["window_id"].astype(str) == prev_w].to_dict("records")]
            ends = [float(e) for e in ends if C.fin(e)]
            gap = best[0] - max(ends) if ends else None
        out[w] = dict(run_id=best[1], read=best[1] in read_ids, created_epoch=best[0],
                      load=C.dig(best[2], "gates_before", "loadavg_1m"), prev_window=prev_w, gap_s=gap)
    return out


# ----------------------------------------------------------------------------------------------- the rules
def _vals(cells: dict, inst: str, seeds, what: str) -> tuple[list, list]:
    """(T values, C values) per seed; `what` = "pt" (the run table) or "generations" (the EA summary)."""
    def one(r, s):
        cl = cells.get((inst, r, s))
        if cl is None:
            return None
        return cl["row"].get("pt") if what == "pt" else C.dig(cl, "summ", "generations")
    return [one("T", s) for s in seeds], [one("C", s) for s in seeds]


def _c_failed(cells: dict, inst: str, seeds) -> list[int]:
    """The seeds whose control run failed (it never decides a rule, the amendment)."""
    return [s for s in seeds if (cells.get((inst, "C", s)) or {}).get("row", {}).get("status") == "failed"]


def _ratio(t: list, c: list):
    if not t or not all(C.fin(x) for x in t + c) or float(np.sum(c)) <= 0:
        return None
    return float(np.sum(t)) / float(np.sum(c))


def _judge(ok: bool, inst: str, seeds, cells: dict) -> tuple[bool | None, str | None]:
    """A rule's state: a fail stands (a failed control only looks worse); a pass with a failed control is unread."""
    if not ok:
        return False, None
    cf = _c_failed(cells, inst, seeds)
    if cf:
        return None, f"a control run failed (seed(s) {cf}): unread, the block(s) are rerun"
    return True, None


def gen_guard(cells: dict, inst: str) -> dict:
    """Rule 1: Ḡ(T) / Ḡ(C) >= 0.90 over seeds 0–5. A missing treatment G (a failed run without a summary) fails it; a
    missing or failed control leaves it unread (INCOMPLETE) unless it fails anyway."""
    gt, gc = _vals(cells, inst, GUARD_SEEDS, "generations")
    per = {s: dict(t=a, c=b, ratio=(float(a) / float(b) if C.fin(a) and C.fin(b) and float(b) > 0 else None))
           for s, a, b in zip(GUARD_SEEDS, gt, gc)}
    out = dict(per=per, threshold=G_MIN, ratio=None, ok=False)
    if not all(C.fin(x) for x in gt):
        return dict(out, why="a treatment generation count is missing (a failed run without an EA summary)")
    miss = [s for s, x in zip(GUARD_SEEDS, gc) if not C.fin(x)]
    if miss or float(np.sum([float(x) for x in gc])) <= 0:
        return dict(out, ok=None, rerun=[(inst, s) for s in (miss or GUARD_SEEDS)],
                    why=f"a control generation count is missing (seed(s) {miss}): unread, the block(s) are rerun")
    t, c = [float(x) for x in gt], [float(x) for x in gc]
    ratio = float(np.mean(t)) / float(np.mean(c))
    ok, why = _judge(ratio >= G_MIN - 1e-12, inst, GUARD_SEEDS, cells)
    return dict(out, mean_t=float(np.mean(t)), mean_c=float(np.mean(c)), ratio=ratio,
                median_ratio=float(np.median(t)) / float(np.median(c)), seeds01=_ratio(t[:2], c[:2]),
                seeds25=_ratio(t[2:], c[2:]), ok=ok, why=why,
                rerun=[(inst, s) for s in _c_failed(cells, inst, GUARD_SEEDS)] if ok is None else [])


def boot_upper(d: list[float], b: int = BOOT_B, seed: int = BOOT_RNG, q: float = BOOT_Q) -> float:
    """The paired bootstrap's one-sided upper bound of the mean of the per-block differences d (blocks resampled with
    replacement)."""
    d = np.asarray(d, float)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(b, len(d)))
    return float(np.quantile(d[idx].mean(axis=1), q))


def p_noharm(cells: dict, inst: str) -> dict:
    """Rule 2: P̄(T) − P̄(C) <= 0.05 over seeds 0–5 (raw P, one-sided); the bootstrap bound beside, not gating."""
    pt, pc = _vals(cells, inst, GUARD_SEEDS, "pt")
    out = dict(t=pt, c=pc, threshold=P_MAX, diff=None, ok=False)
    if not all(C.fin(x) for x in pt):
        return dict(out, why="a treatment P value is missing")
    miss = [s for s, x in zip(GUARD_SEEDS, pc) if not C.fin(x)]
    if miss:
        return dict(out, ok=None, rerun=[(inst, s) for s in miss],
                    why=f"a control P value is missing (seed(s) {miss}): unread, the block(s) are rerun")
    t, c = [float(x) for x in pt], [float(x) for x in pc]
    d = [a - b for a, b in zip(t, c)]
    diff = float(np.mean(t) - np.mean(c))
    ok, why = _judge(diff <= P_MAX + 1e-12, inst, GUARD_SEEDS, cells)
    return dict(out, mean_t=float(np.mean(t)), mean_c=float(np.mean(c)), diff=diff, per_seed=dict(zip(GUARD_SEEDS, d)),
                seeds01=float(np.mean(d[:2])), seeds25=float(np.mean(d[2:])), boot_upper=boot_upper(d),
                ok=ok, why=why, rerun=[(inst, s) for s in _c_failed(cells, inst, GUARD_SEEDS)] if ok is None else [])


def ca_gain(cells: dict) -> dict:
    """Rule 3 (as amended): ΣP(T) / ΣP(C) over ca x seeds 2–3 alone <= 0.95; each seed and the pooled seeds 0–3
    (r1-ysync's seeds 0–1 read beside) beside."""
    allseeds = CA_BESIDE_SEEDS + CA_SEEDS
    at, ac = _vals(cells, CA, allseeds, "pt")
    per = {s: dict(t=a, c=b, ratio=(float(a) / float(b) if C.fin(a) and C.fin(b) and float(b) > 0 else None))
           for s, a, b in zip(allseeds, at, ac)}
    pt, pc = _vals(cells, CA, CA_SEEDS, "pt")
    out = dict(per=per, threshold=CA_MAX, ratio=None, ok=False, pooled03=_ratio(at, ac))
    if not all(C.fin(x) for x in pt):
        return dict(out, why="a treatment P value is missing")
    miss = [s for s, x in zip(CA_SEEDS, pc) if not C.fin(x)]
    ratio = _ratio(pt, pc)
    if miss or ratio is None:
        return dict(out, ok=None, rerun=[(CA, s) for s in (miss or CA_SEEDS)],
                    why=f"a control P value is missing (seed(s) {miss}): unread, the block(s) are rerun")
    ok, why = _judge(ratio <= CA_MAX + 1e-12, CA, CA_SEEDS, cells)
    return dict(out, ratio=ratio, sum_t=float(np.sum(pt)), sum_c=float(np.sum(pc)), ok=ok, why=why,
                rerun=[(CA, s) for s in _c_failed(cells, CA, CA_SEEDS)] if ok is None else [])


def rules(cells: dict) -> dict:
    g = {i: gen_guard(cells, i) for i in GUARD_INSTANCES}
    p = {i: p_noharm(cells, i) for i in GUARD_INSTANCES}
    ca = ca_gain(cells)
    named = [(f"generation guard ({i})", g[i]) for i in GUARD_INSTANCES] + \
            [(f"P no-harm ({i})", p[i]) for i in GUARD_INSTANCES] + [("ca gain replication", ca)]
    fails = [n for n, x in named if x["ok"] is False]
    unread = [n for n, x in named if x["ok"] is None]
    rerun = sorted({b for _, x in named if x["ok"] is None for b in x.get("rerun", [])})
    if fails:
        verdict = "NOT ADOPTED"
        nxt = "final for v1: the R1 line closes and R1 moves to v1.1; no further R1 follow-up for v1"
    elif unread:
        verdict = "INCOMPLETE"
        nxt = (f"no verdict: a failed control run leaves {', '.join(unread)} unread; rerun the block(s) "
               f"{', '.join(f'{i}:s{s}' for i, s in rerun)}, both arms, as set {SET_NEW}-<N> "
               f"(`--write-jobs … --set {SET_NEW}-<N> --blocks …`) and read again")
    else:
        verdict = "ADOPT"
        nxt = ("ysync (configs/itrea2-dev-rc-3way-ysync.yaml) becomes A9's and A3x's development default (D-025 "
               "'What adoption would mean'); C1 is re-run on the frozen hash with A6F in the same window (68 runs) "
               "before Fri 10-23")
    return dict(gen=g, p=p, ca=ca, fails=fails, unread=unread, rerun=rerun, verdict=verdict, next=nxt)


# ----------------------------------------------------------------------------------------------- reading
def _by_set(keys) -> dict:
    out: dict = {}
    for k in keys:
        out.setdefault(k[0], []).append(k)
    return out


def reading(runs: pd.DataFrame, root: Path, keep_failed: bool, ext: dict | None = None) -> dict:
    exp = expected(runs)
    bes = beside_keys()
    ext = ext if ext is not None else C.ext_harness()
    by, failed, dropped = {}, set(), []
    for set_, keys in _by_set(list(exp) + list(bes)).items():
        labs, seeds = sorted({k[2] for k in keys}), sorted({k[3] for k in keys})
        sel, dup = C.select(runs, set_, labs, seeds, BUDGET, keep_failed)
        for r in sel.to_dict("records"):
            k = (set_, r["instance"], r["arm"], int(r["seed"]))
            if k in exp or k in bes:
                by[k] = r
        dropped += [dict(d, set=set_) for d in dup if (set_, d["instance"], d["arm"], d["seed"]) in exp]
        failed |= {(set_,) + k for k in C.failed_keys(runs, set_, labs, seeds, BUDGET)}
    legs: dict[str, list[str]] = {k: [] for k in LEGS}
    for k in exp:
        if k not in by:
            legs["complete"].append(f"{'failed (not completed)' if k in failed else 'missing'} {k[0]} {k[1]} {k[2]} "
                                    f"s{k[3]}")
    raw, beside, notes, kept = {}, {}, [], []
    for k in list(exp) + list(bes):
        if k not in by:
            continue
        r = by[k]
        rj, summ = C.run_files(r, root)
        cell = dict(row=r, summ=summ, rj=rj, role=role(k[2]), mech=D25.mechanism(summ),
                    load=C.dig(rj, "gates_before", "loadavg_1m"), beside=k in bes)
        if k in bes:
            beside[k] = cell
            continue
        name = f"{k[0]} {k[1]} {k[2]} s{k[3]}"
        for leg, probs in C.base_problems(r, rj, summ, exp[k], ext).items():
            legs[leg] += [f"{name}: {x}" for x in probs]
        fp, note = D25.flag_problems("treatment" if role(k[2]) == "T" else "control", VARIANT, r, summ)
        legs["flag"] += [f"{name}: {x}" for x in fp]
        if note:
            notes.append(f"{name}: {note}")
        if not C.fin(r.get("pt")):
            legs["complete"].append(f"no P(120): {name}")
        if r.get("status") == "failed":
            kept.append(name)
        raw[k] = cell
    blocks = [(f"{s_} {i}", s, [b for b in exp if b[0] == s_ and b[1] == i and b[3] == s])
              for s_, i, s in sorted({(k[0], k[1], k[3]) for k in exp})]
    legs["block"] = C.block_problems(raw, blocks)
    for bname, s, keys in blocks:
        have = [raw[k] for k in keys if k in raw]
        if len(have) < 2:
            continue
        ws = [h["row"].get("window_id") for h in have]
        if not all(isinstance(w, str) and w for w in ws):
            legs["window"].append(f"{bname} s{s}: a window id is missing")
        elif len(set(ws)) > 1:
            legs["window"].append(f"{bname} s{s}: the two runs ran in different windows {sorted(set(ws))}")
    bad = [k for k in LEGS if legs[k]]
    verdict = ("INVALID" if any(k != "complete" for k in bad) else "INCOMPLETE" if bad else "VALID")
    val = dict(verdict=verdict, failing=bad, legs=legs, notes=notes,
               deviation=[f"{x} failed: a treatment run is read as an outcome of its arm (D-002 §6), a control "
                          f"run never decides a rule (D-029, amended 12:2x)" for x in kept])
    cells = {(k[1], v["role"], k[3]): v for k, v in list(raw.items()) + list(beside.items())}
    read_ids = {str(v["row"]["run_id"]) for v in raw.values()}
    starts = pool_starts(runs, [v["row"].get("window_id") for v in raw.values()], root, read_ids)
    first_ids = {x["run_id"] for x in starts.values()}
    for v in list(raw.values()) + list(beside.values()):
        v["pool_first"] = str(v["row"]["run_id"]) in first_ids
    out = dict(keep_failed=keep_failed, n_expected=len(exp), n_present=len(raw), validity=val, duplicates=dropped,
               runs=raw, beside=beside, cells=cells, pool_starts=starts, reruns=sorted({k[0] for k in exp} -
                                                                                     {SET_OLD, SET_NEW}),
               rules=None, verdict=verdict)
    if verdict == "VALID":
        out["rules"] = rules(cells)
        out["verdict"] = out["rules"]["verdict"]
    return out


def report(runs: pd.DataFrame, root: Path = ROOT, failed: str = "both") -> dict:
    exp = expected(runs)
    fails = []
    for set_, keys in _by_set(exp).items():
        labs, seeds = sorted({k[2] for k in keys}), sorted({k[3] for k in keys})
        cand = C.candidates(runs, set_, labs, seeds, BUDGET)
        want = {(k[1], k[2], k[3]) for k in keys}
        fails += sorted(str(r["run_id"]) for r in cand[cand["status"] == "failed"].to_dict("records")
                        if (r["instance"], r["arm"], int(r["seed"])) in want)
    modes = {"exclude": [False], "keep": [True], "both": [True, False] if fails else [True]}[failed]
    read = {(k[0], v) for k, v in list(exp.items()) + list(beside_keys().items())}
    sets = [SET_OLD, SET_NEW] + rerun_sets(runs)
    other = sorted(str(r["run_id"]) for r in runs[runs["set"].isin(sets)].to_dict("records")
                   if (str(r["set"]), C.bare_id(r["run_id"])) not in read)
    warm = sorted(str(r) for r in runs[runs["set"].astype(str).str.startswith(SET_NEW)
                                       & runs["set"].astype(str).str.endswith(WARMUP_SUFFIX)]["run_id"])
    # failed runs in blocks a rerun set replaced (an earlier r1f-ysync / r1f-ysync-<N> row): reported, not read
    blocks = {(k[1], k[3]) for k in exp if k[0] not in (SET_OLD, SET_NEW)}
    replaced = sorted(str(r["run_id"]) for r in runs[runs["set"].isin(sets[1:])].to_dict("records")
                      if r.get("status") == "failed" and (r["instance"], int(r["seed"])) in blocks
                      and (str(r["set"]), C.bare_id(r["run_id"])) not in read)
    ext = C.ext_harness()
    return dict(sets=sets, failed_runs=fails, failed_replaced=replaced, other_rows=other, warmup_rows=warm,
                readings=[reading(runs, root, k, ext) for k in modes])


# ----------------------------------------------------------------------------------------------- output
def _seq(v) -> str:
    return " / ".join(C.f(x) if not isinstance(x, (int, np.integer)) else str(x) for x in v)


def _hms(t) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(float(t)).strftime("%H:%M:%S") if C.fin(t) else "–"


def _state(ok) -> str:
    return "ok" if ok is True else "FAIL" if ok is False else "UNREAD"


def render(rep: dict) -> str:
    L = [f"D-029 report (R1 ysync follow-up, Berk 2026-10-15): sets {' + '.join(rep['sets'])}; T = {T_LABEL}, "
         f"C = {C_OLD} (seeds 0–1) / {C_NEW} (seeds 2–5); rules fixed in D-029 (as amended) before any {SET_NEW} run; "
         f"the failing seeds stay in; NOT ADOPTED is final for v1"]
    if rep["failed_runs"]:
        L.append(f"failed runs: {', '.join(rep['failed_runs'])}")
    if rep.get("failed_replaced"):
        L.append(f"failed runs in blocks a rerun set replaced (reported, not read): {', '.join(rep['failed_replaced'])}")
    if rep["other_rows"]:
        L.append(f"rows of these sets no rule reads ({len(rep['other_rows'])}): {', '.join(rep['other_rows'])}")
    if rep.get("warmup_rows"):
        L.append(f"warm-up runs (never read): {', '.join(rep['warmup_rows'])}")
    for rd in rep["readings"]:
        L.append("")
        how = "kept (the verdict reading)" if rd["keep_failed"] else "excluded (the letter)"
        L.append(f"== reading: failed runs {how} ==")
        v = rd["validity"]
        L.append(f"VALIDITY {v['verdict']}: {rd['n_present']} of {rd['n_expected']} runs present"
                 + (f"; failing leg(s): {', '.join(v['failing'])}" if v["failing"] else "")
                 + (f"; blocks read from rerun set(s) {', '.join(rd['reruns'])}" if rd["reruns"] else ""))
        for leg in LEGS:
            for x in v["legs"][leg]:
                L.append(f"  ({leg}) {x}")
        for x in v["notes"]:
            L.append(f"  note: {x}")
        for x in v["deviation"]:
            L.append(f"  kept: {x}")
        for d in rd["duplicates"]:
            L.append(f"  duplicate left out: {d['run_id']}")
        for w, x in sorted(rd["pool_starts"].items()):
            L.append(f"  pool {w}: first run {x['run_id']} ({'read' if x['read'] else 'not read'}) at "
                     f"{_hms(x['created_epoch'])}, load average {C.f(x['load'], 2)} at its gates; previous window "
                     f"in the table {x['prev_window'] or '–'}, ended {C.f(x['gap_s'], 0)} s before")
        L.append("per run (P = pt; mechanism reported, not gating; 'pool first' = the first run of its pool, "
                 "reported, not gating; 'beside' = read beside rule 3 only):")
        allruns = list(rd["runs"].items()) + list(rd.get("beside", {}).items())
        for k, cl in sorted(allruns, key=lambda kv: (kv[0][1], kv[0][3], kv[1]["role"])):
            m, r = cl["mech"], cl["row"]
            L.append(f"  {k[1]:<28} s{k[3]} {cl['role']} {k[2]:<15} P {C.f(r.get('pt'))}  G {m['generations']}  "
                     f"gen/s {C.f(m['gens_per_s'], 2)}  launch/s {C.f(m['launches_per_s'], 1)}  EA self-CPU/wall "
                     f"{C.f(m['self_per_wall'])}  LP done {C.f(m['lp_done_t'], 1)} s  LP share {C.f(m['lp_share'])}  "
                     f"load {C.f(cl.get('load'), 2)}  {r.get('status')}"
                     + ("  [pool first]" if cl.get("pool_first") else "") + ("  [beside]" if cl.get("beside") else ""))
        if v["verdict"] != "VALID":
            L.append(f"RULES NOT READ: {v['verdict']}"
                     + (" — a validity leg fails: fixed and re-checked, with a dated note in D-029, before the INVALID "
                        "blocks are rerun as a rerun set" if v["verdict"] == "INVALID"
                        else " — not every expected run is in (no verdict yet)"))
            continue
        ru = rd["rules"]
        L.append(f"Rule 1, generation guard over seeds 0–5 (Ḡ(T) / Ḡ(C) >= {G_MIN}):")
        for i, x in ru["gen"].items():
            seeds = "; ".join(f"s{s} {y['t']} vs {y['c']} ({C.f(y['ratio'])})" for s, y in x["per"].items())
            if x["ratio"] is None:
                L.append(f"  {i}: {seeds}; {x['why']} -> {_state(x['ok'])}")
            else:
                L.append(f"  {i}: {seeds}; means {x['mean_t']:.1f} / {x['mean_c']:.1f} = {x['ratio']:.3f} -> "
                         f"{_state(x['ok'])}{' (' + x['why'] + ')' if x.get('why') else ''} (beside: medians "
                         f"{x['median_ratio']:.3f}, seeds 0–1 {C.f(x['seeds01'])}, seeds 2–5 {C.f(x['seeds25'])})")
        L.append(f"Rule 2, P no-harm over seeds 0–5 (P̄(T) − P̄(C) <= {P_MAX}; bootstrap bound beside, not gating):")
        for i, x in ru["p"].items():
            if x["diff"] is None:
                L.append(f"  {i}: {x['why']} -> {_state(x['ok'])}")
            else:
                L.append(f"  {i}: T {_seq(x['t'])} (mean {x['mean_t']:.3f}), C {_seq(x['c'])} (mean {x['mean_c']:.3f});"
                         f" diff {x['diff']:+.3f} -> {_state(x['ok'])}{' (' + x['why'] + ')' if x.get('why') else ''} "
                         f"(beside: one-sided 95 % bootstrap upper bound {x['boot_upper']:+.3f}; seeds 0–1 "
                         f"{x['seeds01']:+.3f}, seeds 2–5 {x['seeds25']:+.3f})")
        ca = ru["ca"]
        seeds = "; ".join(f"s{s} {C.f(y['t'])} vs {C.f(y['c'])}" for s, y in ca["per"].items())
        if ca["ratio"] is None:
            L.append(f"Rule 3, ca gain replication at seeds 2–3: {seeds}; {ca['why']} -> {_state(ca['ok'])}")
        else:
            L.append(f"Rule 3, ca gain replication at seeds 2–3 alone (ΣP(T) / ΣP(C) <= {CA_MAX}): {seeds}; "
                     f"{ca['sum_t']:.4f} / {ca['sum_c']:.4f} = {ca['ratio']:.3f} -> {_state(ca['ok'])}"
                     f"{' (' + ca['why'] + ')' if ca.get('why') else ''} (beside: seeds 0–3 pooled "
                     f"{C.f(ca['pooled03'])})")
        L.append(f"VERDICT: {ru['verdict']}" + (f" — failing: {', '.join(ru['fails'])}" if ru["fails"] else "")
                 + (f" — unread: {', '.join(ru['unread'])}" if ru["unread"] and not ru["fails"] else ""))
        L.append(f"  next: {ru['next']}")
    return "\n".join(L)


def _json(rep: dict) -> dict:
    def runs_of(d):
        return {f"{k[0]} {k[1]} {k[2]} s{k[3]}": dict(
                    run_id=cl["row"]["run_id"], role=cl["role"], pt=cl["row"].get("pt"), status=cl["row"].get("status"),
                    pool_first=cl.get("pool_first"), load=cl.get("load"), mechanism=cl["mech"])
                for k, cl in d.items()}
    return dict(rep, readings=[{k: v for k, v in rd.items() if k not in ("runs", "beside", "cells")}
                               | dict(runs=runs_of(rd["runs"]), beside=runs_of(rd.get("beside", {})))
                               for rd in rep["readings"]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="d029_report", description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, default=DATASETS / "runs.parquet")
    ap.add_argument("--root", type=Path, default=ROOT, help="the run directories' root (run_dir is relative to it)")
    ap.add_argument("--failed", choices=("both", "exclude", "keep"), default="both")
    ap.add_argument("--json", type=Path, help="write the report as JSON here too")
    ap.add_argument("--write-jobs", type=Path, default=None, metavar="PATH",
                    help=f"write D-029's job file ({JOBS_OUT}; the warm-up and the read runs) and exit")
    ap.add_argument("--set", default=SET_NEW,
                    help=f"with --write-jobs: {SET_NEW}, or a rerun set {SET_NEW}-<N> (N >= 2)")
    ap.add_argument("--blocks", default=None, help="with --write-jobs and a rerun set: <instance>:s<seed>,… to rerun")
    a = ap.parse_args(argv)
    if a.write_jobs is not None:
        try:
            check_set(a.set)
            blocks = parse_blocks(a.blocks) if a.blocks else None
            if a.set != SET_NEW and not blocks:
                raise ValueError("a rerun set needs --blocks (the INVALID or unread blocks, both arms)")
            if a.set == SET_NEW and blocks:
                raise ValueError(f"--blocks only with a rerun set ({SET_NEW}-<N>)")
            jobs = job_file(a.set, blocks)
            bad = pinned_problems(jobs)
            if bad:
                raise ValueError("; ".join(bad))
        except (mj.JobError, OSError, KeyError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        mj.write_jobs(jobs, a.write_jobs)
        print(f"{a.write_jobs}: {len(jobs)} jobs in {len({(j['set'], j['block']) for j in jobs})} blocks "
              f"(1 warm-up, priority {PRIORITY_WARMUP}; the rest priority {PRIORITY_READ})")
        return 0
    try:
        runs = pd.read_parquet(a.runs)
        rep = report(runs, a.root, a.failed)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(render(rep))
    if a.json:
        a.json.write_text(json.dumps(C.jsonable(_json(rep)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
