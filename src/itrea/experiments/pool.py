"""Run a job file (JSONL, one run per line — `itrea.experiments.make_jobs`) under every control of D-002 and D-003.

Order (D-002 §6, design critique "run order"): a **block** is one instance × seed (all its arms share the SCIP seed:
common random numbers). Each block's arm order is a row of a **Williams design** (balanced for first-order carryover —
GPU arms heat the box, so position matters) indexed by hash(instance, seed, 20261005); blocks run in a seeded random
order with the strata interleaved, so no stratum is concentrated at one time of night. **A block is never split across
a stop**: a stop (deadline, lease end, stop file) only prevents NEW blocks; a block found half-done when the pool
resumes was split by a crash or a kill, and all its runs are moved aside (void) and rerun together.

Lanes (D-002 §3, §7): one **GPU lane** `g0` — SCIP on X925 core 5 (6 for A1's second SCIP) and the partner on A725
cores 1–3 plus the GPU — runs every arm with a partner process (A2, A3, A3r, A4, A5*, A6, H2, H3) strictly one at a
time, so the partner footprint is identical for cuOpt and the EA. Solver-only arms (A0, A0*, A1, SCREEN, H0) run on
the same lane (serial, the default) or on `--cpu-lanes N` fixed CPU lanes when the E0 calibration found the
interference small (`configs/e0-calibration.yaml`: `cpu_arms_concurrent: true`). The screening layout has NINE
single-core X925 lanes and no GPU lane: 9 x 9 GiB + the 20 GB reserve = 101 GB fits the ~108 GB MemAvailable with both
model servers stopped; ten lanes (110 GB) never could. The **full** layout (D-020) is one lane `full` holding all 20
cores and the GPU, for arm A6m (cuOpt as GAMS ran it for MIPfeas) only: any other arm is refused under it and A6m under
any other layout; it implies the serving check (both local model servers stopped, `--allow-serving` refused) and D-020's
memory settings for A6m (`--cuopt-mem-gb 24 --gpu-mem-gb 36`: 24 + 36 + 1 CUDA + 20 reserve = 81 GB with the file
mailbox). The runner's own threads (RUNNER_CORES) share that lane's cores: a small load, recorded in run.json. The
full layout refuses `--mailbox redis` (the container's core 0 is inside the lane) and `--cuopt-gpu-peak-gb` (it would
let the memory gate admit A6m below D-020's 81 GB).

Gates before EVERY run, re-evaluated together in one loop until a single pass clears them all (a long thermal or memory
wait never releases a run on a stale lease or serving check): the box-window lease (`i-trea-ii (Claude Code)`), the
local model servers stopped for test and screening jobs and on the whole-box lane, fan-governor's rest flag (D-009),
the thermal start gate (< 80 °C GPU arms, < 86 °C CPU-only arms, D-008, D-021 R24; 88 until 2026-10-11), and D-003's
memory formula over the run plus
every run already ADMITTED (lanes still waiting at the gate never count; the anonymous memory the admitted runs already
use is credited back, since MemAvailable no longer contains it). Admission is atomic. A memory wait that times out
voids and requeues its block while other runs are in flight, and halts only when nothing is. A live watchdog stops the
run set below 18 GB free (halt) or at ≥ 92 °C (pause: the blocks in flight are voided and rerun once the box is below
80 °C). A GPU-health void (stuck-low clock, foreign GPU work, too few samples) voids the block and halts the run set
for a look. A block voided by the thermal pause is rerun first, at most `--max-thermal-requeues` times (default 1, so
two attempts): a block voided a further time is skipped (status file `skipped`, reason thermal, with its run ids) and
the pool moves on. Before 2026-10-11 there was no cap, and the loop happened: in the screening's stage 2 (2026-10-06
06:04–06:57, eight lanes) the same 8 blocks, rerun first after every pause, were voided 8 times each
(`traces/raw/screen2/void.jsonl`); under the cap they would have been skipped after their second void and other blocks
run. Every run is checked again just before it starts, after its gate waits (thermal, rest flag, memory): if its
estimate (`_est_run`) would end after the deadline or the lease end, it does not start; for the first run of a block
the block counts as never opened and the pool stops as at a deadline, while a block already partly run keeps running (a
stop never splits a block). Test-split jobs run only with the frozen configuration (its hash =
configs/frozen-e2.yaml's) and the registered harness settings, with `prereg-v1` an ancestor of HEAD and a clean tree;
screening jobs and any job on a screening candidate need `prereg-v1` too (screening also a clean tree), and other
non-toy jobs need it unless `--allow-unfrozen`. Only toy instances (outside data/, not named like a candidate) get the
smoke exemptions.

ITX-1 (D-021, 2026-10-11): a GPU run also starts only while fan-governor is alive (its state file younger than 30 s)
with its clock cap off (R24; `--no-governor` for a host without it; a governor gone for `governor_timeout_s` halts, also
while its rest flag holds the pool — only the governor removes it); jobs with a harness are quiet like test and
screening jobs (both model servers stopped, checked before every run; `--allow-serving` refused); R26's breaker: after
`--max-heat-voids` (2) thermal or cap voids — one thermal pause counts once — no new block opens and the pool ends
`halted`; a cap void with the governor's cap off before and after and no cap interval in its log (an unknown clock
lock) halts at once; `external_kill` (a SIGKILL nobody in the harness sent) is rerun once like the causes below;
`--pause-c` lowers the watchdog's pause / void line (M2's soak: 90; never above 92); the runner's new void causes
(pair.void_checks: `gpu_cap` below `--cap-floor-mhz` 2,300 or on a governor clock cap, `compiled_on_clock` and
`cuopt_memory` under a harness) void the block, which is rerun once, first and whole (a cap-voided block only after
the fan reads AUTO again), then skipped and listed (status `skipped`, reason = the cause); `thermal_watch` (98 °C) is
requeued as the thermal pause; every rerun is counted by cause (`counts.requeues`). A job's `harness` (D-021) is
validated before anything runs and refused on test and screening jobs (a prereg-v2 item); a block whose every job
carries `position` keeps that order (M2's AB / BA) instead of the Williams row; `--dry-run` prints D-003's arithmetic
per treatment (e.g. A9 under the ext harness: 9 + 6 + 24 host + 12 + 24 GPU + 2 + 20 = 97 GB) and the governor's state.

D-021 phase 2 (2026-10-12): a job may carry an integer `priority` — when every job of the file does, the block groups
run in increasing priority, each group in the seeded stratified order (`order_blocks`; A0's H_cu ∪ W_SCIP unseen first,
M6's loads before its measured runs); a block voided by `ref_rejected` (R12's stop at the proven optimum not confirmed
by the exact checker) is rerun once, whole and first, with the stop off (`ref_stop_off` in its jobs, outside the run
id; a resumed pool reads it from the set's void.jsonl); external jobs (split `ext`) of an arm with the EA need the tag
`prereg-v2` as an ancestor of HEAD and the configuration registered with it (`configs/frozen-v2.yaml`, unchanged since
the tag; every A9 job's config hash equal to it, no overrides), and every external job needs a clean tree under src/
and configs/ (R33, `check_frozen`).

Resumable: run directories whose run.json says completed (or failed: a crash is kept, D-002 §6) are skipped. One pool
at a time (RunLock). Log: logs/pool-<date>.log; status: logs/pool-status.json; voided runs: traces/raw/<set>/void/ and
void.jsonl (rerun counts per arm are reported from it).

Adapted in part from berkorbay/optopt@4dd48395 (MIT), `src/optopt/experiments/pool.py`.

    python -m itrea.experiments.pool jobs/screen1.jsonl --layout screen
    python -m itrea.experiments.pool jobs/e2.jsonl --mailbox redis --deadline 07:15
    python -m itrea.experiments.pool jobs/x.jsonl --dry-run
    python -m itrea.experiments.pool <make-jobs mipfeas-cal --out FILE> --layout full --cuopt-mem-gb 24 --gpu-mem-gb 36
    python -m itrea.experiments.pool jobs/itx-m2.jsonl --pause-c 90        (D-021 M2: the soak at F)
"""
from __future__ import annotations

import os as _os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # the runner itself, before NumPy loads
    _os.environ.setdefault(_k, "1")

import argparse
import dataclasses
import hashlib
import json
import os
import random
import shutil
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from itrea import paths
from itrea.config import EAConfig
from itrea.experiments import gates, gpu_health, pair, record_env

MASTER_SEED = 20261005
WHO = "i-trea-ii (Claude Code)"
FROZEN_CONFIG = "configs/frozen-e2.yaml"
E0_FLAG = "configs/e0-calibration.yaml"
CONFIRM_SPLITS = ("test",)
EXT_SPLIT = "ext"                         # D-021's external (ITX-1) v1 runs: the tiers and A0 (`make-jobs itx-v1`, `itx-a0`)
PREREG_V2_TAG = "prereg-v2"               # D-021 R33: no A9 outcome on an unseen instance before this tag
FROZEN_V2_CONFIG = "configs/frozen-v2.yaml"   # A9's v1 configuration (the C1 freeze), committed in the prereg-v2 commit
QUIET_SPLITS = ("test", "screen")         # both local model servers must be stopped (D-002 §1, D-003)


def is_quiet(job: dict) -> bool:
    """Both local model servers must be stopped for this job: test and screening jobs (D-002 §1, D-003) and every job
    with a D-021 harness (R26: ITX windows run with both servers off)."""
    return job.get("split") in QUIET_SPLITS or job.get("harness") is not None
JOB_FIELDS = ("instance", "set", "name", "arm", "seed", "budget", "config", "overrides", "stratum", "split", "block")

# GPU lane + CPU lanes for pair runs; X925 = 5–9, 15–19; A725 = 1–4, 10–14 (core 0: Redis)
PAIR_GPU_LANE = pair.GPU_LANE
PAIR_CPU_LANES = (pair.Lane("c1", (7, 8)), pair.Lane("c2", (9, 15)), pair.Lane("c3", (16, 17)),
                  pair.Lane("c4", (18, 19)))
# nine, not ten: D-003's gate (Σ 9 GiB caps + 20 GB reserve) must fit MemAvailable (~108 GB, both servers stopped)
SCREEN_LANES = tuple(pair.Lane(f"x{c}", (c,)) for c in (5, 6, 7, 8, 9, 15, 16, 17, 18))
FULL_ARMS = tuple(a for a, spec in pair.ARMS.items() if spec.lane == "full")     # ("A6m",): the whole-box lane's arms
# rerun once (thermal_watch: as a thermal void); ref_rejected's rerun runs with R12's stop off (D-021 phase 2)
REQUEUE_CAUSES = ("gpu_cap", "compiled_on_clock", "cuopt_memory", "external_kill", "ref_rejected")
A6M_MEM_GB = dict(cuopt_mem_gb=24.0, gpu_mem_gb=36.0)    # D-020: A6m's host MemoryMax and RMM limit (GAMS's max RSS
                                                         # <= 16 GB on the sample, 49 GB at most over the 233)
# harness settings a run may not change from PairOptions() defaults, per split (D-002 §1.2 / §1.8, D-003)
REGISTERED_HARNESS = {
    "test": ("gpu_mem_gb", "pin", "solver_mem_gb", "ea_mem_gb", "cuopt_mem_gb", "cuopt_threads", "use_systemd"),
    "screen": ("pin", "solver_mem_gb", "use_systemd"),
}



def _arm_uses_gpu(job: dict) -> bool:
    """True if the job's arm starts a process on the GPU (EA on CuPy, cuOpt) — those keep the 80 °C start gate."""
    try:
        return bool(pair.ARMS[str(job.get("arm"))].gpu)
    except KeyError:
        return True                                   # unknown arm: the stricter gate

class PoolError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------- Williams
def williams_rows(k: int) -> list[list[int]]:
    """Williams design for k treatments: k rows (k even) or 2k rows (k odd, the mirror rows added). Every treatment
    appears once per position per k rows, and every ordered pair (a, b) appears as neighbours equally often."""
    if k < 1:
        return []
    if k == 1:
        return [[0]]
    first, lo, hi = [0], 1, k - 1
    take_lo = True
    while len(first) < k:
        if take_lo:
            first.append(lo)
            lo += 1
        else:
            first.append(hi)
            hi -= 1
        take_lo = not take_lo
    rows = [[(x + i) % k for x in first] for i in range(k)]
    if k % 2 == 1:
        rows += [list(reversed(r)) for r in rows]
    return rows


def block_hash(instance: str, seed: int, master: int = MASTER_SEED) -> int:
    """hash(instance, seed, 20261005): a stable 64-bit integer (sha256; Python's hash() is salted per process)."""
    return int(hashlib.sha256(f"{instance}|{int(seed)}|{int(master)}".encode()).hexdigest()[:16], 16)


def canonical_arms(labels) -> list[str]:
    """Labels in the canonical arm order (ARM_ORDER; ablation labels after their base arm, by name)."""
    order = {a: i for i, a in enumerate(pair.ARM_ORDER)}

    def key(lbl: str):
        base = lbl.split("~")[0].split("@")[0]
        return (order.get(base, len(order)), base, lbl)
    return sorted(set(labels), key=key)


def williams_order(labels, instance: str, seed: int, master: int = MASTER_SEED) -> list[str]:
    arms = canonical_arms(labels)
    rows = williams_rows(len(arms))
    if not rows:
        return []
    row = rows[block_hash(instance, seed, master) % len(rows)]
    return [arms[j] for j in row]


# ------------------------------------------------------------------------------------------------- blocks
@dataclass
class Run:
    job: dict
    block: "Block"
    state: str = "pending"          # pending | running | done
    status: str | None = None       # completed | failed | void (after it ran)
    run_dir: Path | None = None     # set by the pool (cached: computing it reads the configuration)

    @property
    def label(self) -> str:
        return pair.arm_label(self.job)

    @property
    def kind(self) -> str:
        return pair.arm_spec(self.job).lane


@dataclass
class Inflight:
    """A lane's current run: registered when the lane takes it (for status), counted by other lanes' memory gate
    only once `admitted` (it passed the gate)."""
    run: Run
    budget: gates.Budget
    admitted: bool = False


@dataclass
class Block:
    set: str
    id: str
    name: str
    seed: int
    stratum: str
    runs: list[Run] = field(default_factory=list)
    state: str = "unstarted"        # unstarted | open | done | voiding
    void_reason: str | None = None
    requeue: bool = False
    requeue_cause: str | None = None    # "thermal" | "memory" | a D-021 cause (REQUEUE_CAUSES): why it is to be rerun
    thermal_voids: int = 0              # times the thermal pause voided it (capped by max_thermal_requeues)
    voids: dict = field(default_factory=dict)   # D-021 causes: times each voided it (capped by max_cause_requeues)
    wait_auto: bool = False             # requeued after a cap void: its next run waits for the fan on AUTO (R25)
    ref_stop_off: str | None = None     # D-021 R12: voided by `ref_rejected` — its rerun runs without the stop (the
                                        # reason travels in the job as `ref_stop_off`, outside the run id)
    priority: int | None = None         # D-021 phase 2: the jobs' `priority` (lower runs first; only when every job of
                                        # the file carries one, `order_blocks`)

    @property
    def key(self) -> tuple[str, str]:
        return (self.set, self.id)


def load_jobs(path: str | os.PathLike) -> list[dict]:
    jobs = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            jobs.append(json.loads(line))
        except ValueError as e:
            raise PoolError(f"{path}:{i}: not JSON ({e})") from None
    return jobs


def validate_job(job: dict) -> dict:
    for k in ("instance", "set", "arm", "seed", "budget"):
        if k not in job:
            raise PoolError(f"job lacks {k!r}: {job}")
    pair.arm_spec(job)
    if float(job["budget"]) <= 0:
        raise PoolError(f"job budget must be > 0: {job}")
    try:
        pair.job_harness(job)                     # D-021: a preset name or a dict of known keys
    except pair.RunError as e:
        raise PoolError(f"{pair.arm_label(job)} of {job.get('name') or pair.stem(job['instance'])}: {e}") from None
    pr = job.get("priority")
    if pr is not None and (isinstance(pr, bool) or not isinstance(pr, int)):
        raise PoolError(f"job priority must be an integer, not {pr!r}: {job}")
    return job


def _fixed_positions(runs: list[Run], bid: str) -> dict[str, int] | None:
    """A block whose every job carries `position` keeps that order (a writer's explicit AB/BA, D-021 M2): {label:
    position}, or None when no job carries one. Some but not all, or not a permutation of 0..k-1: PoolError."""
    pos = [r.job.get("position") for r in runs]
    if all(p is None for p in pos):
        return None
    if any(p is None for p in pos) or sorted(pos) != list(range(len(runs))) or \
            any(isinstance(p, bool) or not isinstance(p, int) for p in pos):
        raise PoolError(f"block {bid}: positions {pos} must be given for every run, as a permutation of "
                        f"0..{len(runs) - 1}")
    return {r.label: int(p) for r, p in zip(runs, pos)}


def make_blocks(jobs: list[dict], master: int = MASTER_SEED) -> list[Block]:
    """Group jobs into blocks (set, block id) and order each block's runs by the Williams design — or by the jobs' own
    `position` when every job of the block carries one (`_fixed_positions`; no committed job file before D-021 has
    the key, so every earlier block keeps its Williams order)."""
    blocks: dict[tuple[str, str], Block] = {}
    for j in jobs:
        validate_job(j)
        name = j.get("name") or pair.stem(j["instance"])
        bid = j.get("block") or f"{name}__s{int(j['seed'])}"
        b = blocks.get((j["set"], bid))
        if b is None:
            b = blocks[(j["set"], bid)] = Block(j["set"], bid, name, int(j["seed"]), j.get("stratum") or "-",
                                                priority=j.get("priority"))
        elif b.name != name:
            raise PoolError(f"block {bid} mixes instances {b.name} and {name}")
        elif b.priority != j.get("priority"):
            raise PoolError(f"block {bid}: its jobs carry different priorities ({b.priority}, {j.get('priority')})")
        b.runs.append(Run(j, b))
    for b in blocks.values():
        labels = [r.label for r in b.runs]
        if len(set(labels)) != len(labels):
            raise PoolError(f"block {b.id}: an arm appears twice ({sorted(labels)})")
        pos = _fixed_positions(b.runs, b.id)
        if pos is None:
            order = williams_order(labels, b.name, b.seed, master)
            pos = {lbl: i for i, lbl in enumerate(order)}
        b.runs.sort(key=lambda r: pos[r.label])
        for r in b.runs:
            r.job = dict(r.job, position=pos[r.label], block=b.id)
    return list(blocks.values())


def order_blocks(blocks: list[Block], seed: int = MASTER_SEED) -> list[Block]:
    """Seeded random order, strata interleaved: within a stratum the blocks are shuffled; the i-th of n_s blocks of
    stratum s gets the key (i + u)/n_s with u uniform in [0, 1), and the blocks run in key order.

    When every block carries a `priority` (D-021 phase 2: the jobs' field, e.g. `itx-a0`'s H_cu ∪ W_SCIP unseen first,
    `itx-m6`'s load before the measured runs) the groups run in increasing priority, each group in its own order by
    the rule above (same seed); some blocks with and some without is refused. No job file before D-021 phase 2 has the
    field, so every earlier order is unchanged."""
    pr = {b.priority for b in blocks}
    if None in pr and len(pr) > 1:
        raise PoolError("some blocks carry a priority and some do not: give every job one, or none")
    if pr and None not in pr:
        return [b for p in sorted(pr) for b in _order_group([x for x in blocks if x.priority == p], seed)]
    return _order_group(blocks, seed)


def _order_group(blocks: list[Block], seed: int) -> list[Block]:
    """`order_blocks`' rule on one group of blocks."""
    rng = random.Random(seed)
    by: dict[str, list[Block]] = {}
    for b in sorted(blocks, key=lambda b: b.key):
        by.setdefault(b.stratum, []).append(b)
    keyed = []
    for s in sorted(by):
        lst = by[s]
        rng.shuffle(lst)
        n = len(lst)
        for i, b in enumerate(lst):
            keyed.append(((i + rng.random()) / n, s, b.key, b))
    keyed.sort(key=lambda x: (x[0], x[1], x[2]))
    return [x[3] for x in keyed]


# ---------------------------------------------------------------------------------------------- freeze check
def check_frozen(jobs: list[dict], allow_unfrozen: bool = False, root: Path | None = None,
                 frozen: str = FROZEN_CONFIG, frozen_v2: str = FROZEN_V2_CONFIG) -> dict:
    """D-002 §1 / §1.8. `prereg-v1` (an ancestor of HEAD) precedes the first timed run on any candidate instance:
    test-split jobs, screening jobs and any job whose instance is a screening candidate need it, never bypassed;
    other non-toy jobs need it unless allow_unfrozen. Test and screening jobs also need a clean, committed tree under
    src/ and configs/; test jobs need the frozen configuration and no job-level overrides beyond a named ablation file
    (committed with the tag). Toy smoke jobs (outside data/, not named like a candidate) need nothing. D-021 R33:
    external jobs (split "ext") of an arm with the EA (A9) need the tag `prereg-v2` as an ancestor of HEAD — before it,
    no A9 outcome on an unseen instance is run (A0 and A6F's external jobs need only prereg-v1) — AND the configuration
    registered with it: `configs/frozen-v2.yaml` (the C1 freeze) must exist, be byte-identical to the file in the
    tagged commit, and every such job's configuration must hash to it, with no free overrides or ablation (as test jobs
    against frozen-e2). Every external job (A0, A6F, A2F and A9 alike) needs a clean, committed tree under src/ and
    configs/. Raises GateError; returns what was checked."""
    from itrea.experiments import make_jobs                 # lazy: make_jobs imports this module
    root = root or paths.ROOT
    names, roots = make_jobs.candidate_names(), make_jobs.data_roots()
    test = [j for j in jobs if j.get("split") in CONFIRM_SPLITS]
    screen = [j for j in jobs if j.get("split") == "screen"]
    cand = [j for j in jobs if make_jobs.is_candidate(j, names)]
    toy = [j for j in jobs if j.get("split") == "smoke" and make_jobs.is_toy(j, names, roots)]
    other = [j for j in jobs if j.get("split") not in CONFIRM_SPLITS and j.get("split") != "screen"
             and not (j.get("split") == "smoke" and make_jobs.is_toy(j, names, roots))]
    out: dict = dict(test_jobs=len(test), screen_jobs=len(screen), candidate_jobs=len(cand), other_jobs=len(other),
                     toy_jobs=len(toy))
    always = bool(test or screen or cand)
    need_prereg = always or (bool(other) and not allow_unfrozen)
    if need_prereg:
        if record_env.tag_commit(record_env.PREREG_TAG, root) is None:
            hint = ("" if always else "; --allow-unfrozen for non-candidate, non-test, non-screening jobs")
            raise gates.GateError(f"git tag {record_env.PREREG_TAG} does not exist: the pre-registration is not "
                                  f"frozen (D-002 §1: the tag precedes the first timed run on any candidate "
                                  f"instance, the screening pass included){hint}")
        if not record_env.is_ancestor(record_env.PREREG_TAG, "HEAD", root):
            raise gates.GateError(f"{record_env.PREREG_TAG} is not an ancestor of HEAD")
        out["prereg"] = record_env.tag_commit(record_env.PREREG_TAG, root)
    ext = [j for j in jobs if j.get("split") == EXT_SPLIT]
    ext_ea = [j for j in ext if pair.arm_spec(j).has_ea]
    out["ext_jobs"] = len(ext)
    if ext_ea:                    # D-021 R33 (sealing): A9's external runs only after prereg-v2 is tagged
        if record_env.tag_commit(PREREG_V2_TAG, root) is None or \
                not record_env.is_ancestor(PREREG_V2_TAG, "HEAD", root):
            raise gates.GateError(f"{len(ext_ea)} external (split {EXT_SPLIT!r}) job(s) run the EA "
                                  f"({sorted({pair.arm_label(j) for j in ext_ea})}): D-021 R33 seals A9's outcomes on "
                                  f"the unseen instances until git tag {PREREG_V2_TAG} exists and is an ancestor of HEAD")
        out["prereg_v2"] = record_env.tag_commit(PREREG_V2_TAG, root)
        # ... and only on the configuration registered with the tag (the C1 freeze), never on a development default
        fz = Path(root) / frozen_v2
        if not fz.exists():
            raise gates.GateError(f"{frozen_v2} does not exist: external EA jobs run only on the configuration "
                                  f"registered with {PREREG_V2_TAG} (the C1 freeze, committed in the tagged commit)")
        at_tag = record_env.file_at(PREREG_V2_TAG, frozen_v2, root)
        if at_tag is None:
            raise gates.GateError(f"{frozen_v2} is not in the commit tagged {PREREG_V2_TAG}: the configuration must be "
                                  f"registered with the tag")
        if at_tag != fz.read_bytes():
            raise gates.GateError(f"{frozen_v2} differs from the file registered at {PREREG_V2_TAG}")
        fh = EAConfig.from_yaml(fz).hash()
        for j in ext_ea:
            h = pair.base_config_hash(j)
            if h != fh:
                raise gates.GateError(f"external job {pair.stem(j['instance'])} {pair.arm_label(j)}: config hash {h} "
                                      f"!= {PREREG_V2_TAG}'s {fh} ({frozen_v2}; write the tiers with "
                                      f"`make-jobs itx-v1`, whose A9 default is that file)")
            if j.get("overrides") or j.get("ablation"):
                raise gates.GateError(f"external job {pair.stem(j['instance'])} {pair.arm_label(j)} carries overrides "
                                      f"or an ablation ({j.get('overrides') or j.get('ablation')}); v1 runs the "
                                      f"registered configuration only")
        out["frozen_v2_hash"] = fh
    if test or screen or ext:
        dirty = record_env.dirty_paths(root=root)
        if dirty is None or dirty:
            kind = "test" if test else "screening" if screen else "external (split ext)"
            raise gates.GateError(f"{kind} jobs need a clean, committed tree under src/ and configs/ (dirty: {dirty})")
    if test:
        fz = Path(root) / frozen
        if not fz.exists():
            raise gates.GateError(f"{frozen} does not exist: E4 has not frozen a configuration")
        fh = EAConfig.from_yaml(fz).hash()
        for j in test:
            h = pair.base_config_hash(j)
            if h != fh:
                raise gates.GateError(f"test job {pair.stem(j['instance'])} {j['arm']}: config hash {h} != frozen {fh}")
            if j.get("overrides"):
                raise gates.GateError(f"test job {pair.stem(j['instance'])} {j['arm']} carries free overrides "
                                      f"{j['overrides']}; ablations go by name (configs/ablations/<name>.yaml)")
        out["frozen_hash"] = fh
    return out


def check_layout(jobs: list[dict], layout: str) -> None:
    """The whole-box lane and its arms go together (D-020): `--layout full` runs FULL_ARMS only, and those arms run only
    under it. Raises PoolError."""
    full = sorted({pair.arm_label(j) for j in jobs if pair.arm_spec(j).lane == "full"})
    other = sorted({pair.arm_label(j) for j in jobs if pair.arm_spec(j).lane != "full"})
    if layout == "full" and other:
        raise PoolError(f"--layout full (all 20 cores, one run at a time) runs only {', '.join(FULL_ARMS)} (D-020); "
                        f"the job file also has {other}")
    if layout != "full" and full:
        raise PoolError(f"{full} run only under --layout full (D-020: cuOpt as GAMS ran it, on all 20 cores), not "
                        f"--layout {layout}")


def check_a6m_memory(jobs: list[dict], po: pair.PairOptions) -> None:
    """A6m runs with D-020's host MemoryMax and RMM limit (A6M_MEM_GB). The RMM limit is in the run hash; MemoryMax is
    not, so without this check a run under a smaller cap would carry the calibration's run id. Raises GateError."""
    a6m = [j for j in jobs if j.get("arm") == "A6m"]
    if not a6m:
        return
    rmm = {pair.effective_config(j, po)[0].gpu_mem_gb for j in a6m}
    if po.cuopt_mem_gb != A6M_MEM_GB["cuopt_mem_gb"] or rmm != {A6M_MEM_GB["gpu_mem_gb"]}:
        raise gates.GateError(f"A6m runs with D-020's memory settings: --cuopt-mem-gb {A6M_MEM_GB['cuopt_mem_gb']:g} "
                              f"--gpu-mem-gb {A6M_MEM_GB['gpu_mem_gb']:g} (got host MemoryMax {po.cuopt_mem_gb:g} GB, "
                              f"RMM limit {sorted(rmm)} GB)")


def check_full_options(po: pair.PairOptions) -> None:
    """Options the whole-box layout refuses (D-020, review 2026-10-11): the Redis mailbox (its container is pinned to
    core 0, inside the lane) and a measured cuOpt peak (it would let the memory gate admit A6m below 81 GB)."""
    if po.mailbox == "redis":
        raise gates.GateError("--layout full refuses --mailbox redis: the itrea-redis container's core 0 is inside the "
                              "whole-box lane (use the file mailbox)")
    if po.cuopt_gpu_peak_gb is not None:
        raise gates.GateError("--layout full refuses --cuopt-gpu-peak-gb: A6m's memory gate counts the RMM limit "
                              "(D-020: 24 + 36 + 1 + 20 = 81 GB)")


def check_registered_harness(jobs: list[dict], po: pair.PairOptions) -> None:
    """Test and screening jobs run with the registered harness settings only (PairOptions() defaults): the CLI's
    --gpu-mem-gb (the EA pool and, with population.size_rule: memory, P; cuOpt's RMM limit in A2), --no-pin,
    --solver-mem-gb (SCIP's own --mem-mb), … would otherwise change a confirmatory configuration the freeze check
    does not see (it hashes the configuration file alone). Raises GateError."""
    d = pair.PairOptions()
    for split, fields in REGISTERED_HARNESS.items():
        js = [j for j in jobs if j.get("split") == split]
        if not js:
            continue
        hz = sorted({pair.arm_label(j) for j in js if j.get("harness") is not None})
        if hz:      # D-021's job-level harness is a prereg-v2 item: not registered under prereg-v1
            raise gates.GateError(f"{split} jobs run without a job-level harness until prereg-v2 registers it "
                                  f"(D-021); {hz} carry one")
        bad = {k: getattr(po, k) for k in fields if getattr(po, k) != getattr(d, k)}
        if bad:
            raise gates.GateError(f"{split} jobs run with the registered harness settings only; changed: {bad}")
        if split in CONFIRM_SPLITS:
            for j in js:
                arm = pair.arm_spec(j)
                _, ov = pair.effective_config(j, po)
                allowed = set(dict(arm.ea_overrides)) | {"backend"} | set(pair.ablation_overrides(j.get("ablation")))
                extra = sorted(set(ov) - allowed)
                if extra:
                    raise gates.GateError(f"test job {pair.stem(j['instance'])} {pair.arm_label(j)}: effective "
                                          f"configuration differs from the frozen one in {extra}")


def e0_concurrent(root: Path | None = None) -> bool | None:
    """The E0 calibration flag (D-002 §7): may the CPU arms run concurrently with the GPU lane? None = not measured."""
    p = Path(root or paths.ROOT) / E0_FLAG
    if not p.exists():
        return None
    d = yaml.safe_load(p.read_text()) or {}
    v = d.get("cpu_arms_concurrent")
    return bool(v) if v is not None else None


# --------------------------------------------------------------------------------------------------- options
@dataclass
class PoolOptions:
    who: str = WHO
    require_lease: bool = True             # False only for toy smoke tests (every job split == "smoke")
    layout: str = "pair"                   # "pair" (GPU lane + CPU lanes) | "screen" (nine single X925 lanes) |
                                           # "full" (one lane, all 20 cores and the GPU: A6m only, D-020)
    cpu_lanes: int = 0                     # pair: CPU lanes beside the GPU lane (0 = serial); screen: lanes used
    deadline: datetime | None = None       # no NEW block whose estimate ends after this
    allow_unfrozen: bool = False
    allow_serving: bool = False            # model servers up: tuning / exploratory jobs only (never test or screen)
    rerun_failed: bool = False             # non-test only: failed blocks are moved aside and rerun
    order_seed: int = MASTER_SEED
    logs_dir: Path = paths.ROOT / "logs"
    status_file: Path | None = None        # default logs/pool-status.json
    stop_file: Path | None = None          # default logs/pool.stop: touch it to stop before the next block
    lock_path: Path = Path("/tmp/itrea-runner.lock")
    thermal_start_c: float = gates.THERMAL_WAIT_C          # GPU arms (any process on the GPU)
    thermal_start_c_cpu: float = gates.THERMAL_WAIT_C_CPU  # CPU-only arms (D-008)
    thermal_timeout_s: float = 3600.0
    thermal_poll_s: float = 30.0
    mem_wait_s: float = 1800.0
    mem_poll_s: float = 15.0
    stale_wait_s: float = 30.0             # resume: how long an earlier attempt's processes get to obey `stop`
    watchdog: bool = True
    watchdog_poll_s: float = 5.0
    est_overhead_s: float | None = None    # per-run overhead for the never-split estimate; None = from the harness:
                                           # grace + kill_after + 15 s (SIGTERM/SIGKILL waits) + start-up (GPU 120 s,
                                           # CPU 30 s) until E0's start-up records give a p95
    open_margin_s: float = 600.0           # a new block must end this long before the deadline / the lease end
    max_open_blocks: int | None = None     # default: number of lanes
    max_blocks: int | None = None          # stop after opening this many blocks (tests, trial windows)
    max_thermal_requeues: int = 1          # a block voided by the thermal pause is rerun at most this often (2026-10-11)
    max_cause_requeues: int = 1            # D-021: a block voided by gpu_cap / compiled_on_clock / cuopt_memory /
                                           # external_kill is rerun once (per cause), then skipped and listed
    max_heat_voids: int = 2                # D-021 R26: after this many thermal / cap voids no new block (0 = off)
    pause_c: float = gates.THERMAL_PAUSE_C  # the watchdog's pause / void line (D-021 M2: 90 for the soak); <= 92
    governor: bool = True                  # D-021 R24: GPU runs start only with the fan governor alive and its cap off
                                           # (False: tests, or a host without the governor)
    governor_timeout_s: float = 120.0      # halt when the governor's state stays stale this long (the governor is gone)
    dry_run: bool = False
    root: Path = paths.ROOT                # git checks, configs
    pair_opts: pair.PairOptions = field(default_factory=pair.PairOptions)


# ------------------------------------------------------------------------------------------------------ pool
class Pool:
    def __init__(self, jobs: list[dict], opts: PoolOptions | None = None):
        self.o = opts or PoolOptions()
        self.jobs = jobs
        self.window = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
        o = self.o
        self.o.logs_dir = Path(o.logs_dir)
        self.status_file = Path(o.status_file or o.logs_dir / "pool-status.json")
        self.stop_file = Path(o.stop_file or o.logs_dir / "pool.stop")
        self.log_path = o.logs_dir / f"pool-{datetime.now():%Y-%m-%d}.log"
        self.cv = threading.Condition()
        self.abort = threading.Event()
        self.abort_kind: str | None = None
        self.abort_text: str = ""
        self.halt_reason: str | None = None
        self.breaker: str | None = None    # D-021 R26: set after max_heat_voids thermal / cap voids (no new block)
        self.heat_voids: set = set()       # the thermal / cap voids counted for it
        self.thermal_episode = 0           # watchdog thermal pauses so far (one pause = one void for R26)
        self.skipped: list[dict] = []      # blocks given up (thermal requeue cap): set, block, reason, run ids
        self.stopping: str | None = None
        self.inflight: dict[str, Inflight] = {}
        self.sampler = None                # the pool's one GPU-health sampler (shared by every lane)
        self.counts = dict(completed=0, failed=0, void=0, skipped=0, blocks_done=0, blocks_void=0,
                           blocks_skipped_thermal=0, requeues={}, blocks_skipped_cause={})   # by cause (D-021 R15/R25)
        self.opened = 0
        self.watchdog: gates.Watchdog | None = None
        self.blocks: list[Block] = []
        self.lanes: list[pair.Lane] = []

    # ----------------------------------------------------------------- logging
    def log(self, msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        try:
            self.o.logs_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def write_status(self, state: str) -> None:
        d = dict(state=state, pid=os.getpid(), window=self.window, updated=datetime.now().astimezone().isoformat(),
                 jobs=len(self.jobs), blocks=len(self.blocks), opened=self.opened, counts=self.counts,
                 stopping=self.stopping, halt=self.halt_reason or self.breaker,
                 abort=self.abort_text if self.abort.is_set() else None,
                 skipped=self.skipped,
                 lanes=[dataclasses.asdict(l) for l in self.lanes],
                 running=[dict(lane=k, run=e.run.run_dir.name if e.run.run_dir else None, block=e.run.block.id,
                               admitted=e.admitted) for k, e in self.inflight.items()])
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.status_file.with_name(f".{self.status_file.name}.tmp")
            tmp.write_text(json.dumps(d, indent=1, default=str))
            os.replace(tmp, self.status_file)
        except OSError:
            pass

    # ------------------------------------------------------------------- setup
    def _lanes(self) -> list[pair.Lane]:
        o = self.o
        if o.layout == "pair":
            n = max(0, int(o.cpu_lanes))
            if n > len(PAIR_CPU_LANES):
                raise PoolError(f"--cpu-lanes {n} > {len(PAIR_CPU_LANES)} CPU lanes in the pair layout")
            return [PAIR_GPU_LANE, *PAIR_CPU_LANES[:n]]
        if o.layout == "screen":
            n = int(o.cpu_lanes) or len(SCREEN_LANES)
            if n > len(SCREEN_LANES):
                raise PoolError(f"--cpu-lanes {n} > {len(SCREEN_LANES)} screening lanes: {len(SCREEN_LANES)} x 9 GiB + "
                                f"the 20 GB reserve is what MemAvailable (~108 GB, both model servers stopped) holds "
                                f"(D-002 §1.2, D-003)")
            return list(SCREEN_LANES[:n])
        if o.layout == "full":
            if int(o.cpu_lanes):
                raise PoolError("--layout full is one lane holding every core: --cpu-lanes must be 0")
            return [pair.FULL_LANE]
        raise PoolError(f"unknown layout {o.layout!r}")

    def _accepts(self, lane: pair.Lane, run: Run) -> bool:
        arm = pair.arm_spec(run.job)
        if not lane.fits(arm):
            return False
        if lane.whole_box:      # the whole box: its own arms only (fits() already pairs them, D-020)
            return arm.lane == "full"
        if lane.gpu:            # the GPU lane takes partner arms, and solver-only arms when there is no CPU lane
            return arm.lane == "partner" or len(self.lanes) == 1
        return arm.lane == "solo"

    def _popts(self, lane: pair.Lane | None) -> pair.PairOptions:
        base = self.o.pair_opts
        return dataclasses.replace(base, lane=lane or base.lane, abort=self.abort,
                                   abort_reason=lambda: self.abort_text, extra_own_pids=self._inflight_pids,
                                   window_id=self.window, log=self.log, check_memory=False,
                                   shared_sampler=self.sampler if base.gpu_health else None,
                                   allow_test=getattr(self, "freeze", None) is not None)

    def _inflight_pids(self) -> set[int]:
        """Every process of the run set in flight (not 'foreign' to the GPU-health sampler)."""
        out: set[int] = set()
        for rd in self._inflight_dirs():
            rec = pair.read_record(rd) or {}
            out |= {p["pid"] for p in rec.get("processes", []) if p.get("pid")}
        return out

    def _inflight_dirs(self) -> list[Path]:
        with self.cv:
            return [e.run.run_dir for e in self.inflight.values() if e.run.run_dir is not None]

    def preflight(self) -> None:
        """Refusals before anything runs (GateError / PoolError)."""
        from itrea.experiments import make_jobs             # lazy: make_jobs imports this module
        o = self.o
        po = o.pair_opts
        names, roots = make_jobs.candidate_names(), make_jobs.data_roots()
        smoke_only = all(j.get("split") == "smoke" and make_jobs.is_toy(j, names, roots) for j in self.jobs)
        check_layout(self.jobs, o.layout)
        full = o.layout == "full"                  # D-020: the box to itself, no foreign GPU process
        quiet = any(is_quiet(j) for j in self.jobs) or full
        if not o.require_lease and not smoke_only:
            raise gates.GateError("--no-lease is for toy smoke tests only (every job split == 'smoke' on a toy "
                                  "instance outside data/)")
        if o.rerun_failed and any(j.get("split") in CONFIRM_SPLITS for j in self.jobs):
            raise gates.GateError("--rerun-failed is not allowed with test jobs (a crash is kept, D-002 §6)")
        if o.allow_serving and quiet:
            raise gates.GateError("--allow-serving is not allowed with test or screening jobs, jobs with a D-021 "
                                  "harness or --layout full "
                                  "(D-002 §1.2, D-003, D-020: both local model servers stopped)")
        self.freeze = check_frozen(self.jobs, o.allow_unfrozen, o.root)
        check_registered_harness(self.jobs, po)
        check_a6m_memory(self.jobs, po)
        if full:
            check_full_options(po)
        if not (gates.THERMAL_RESUME_C < o.pause_c <= gates.THERMAL_PAUSE_C):
            raise gates.GateError(f"--pause-c {o.pause_c:g} must lie above the resume line ({gates.THERMAL_RESUME_C:g} "
                                  f"°C) and never above {gates.THERMAL_PAUSE_C:g} °C (D-008; D-021 M2 lowers it to 90)")
        if o.layout == "pair" and o.cpu_lanes > 0 and not smoke_only and e0_concurrent(o.root) is not True:
            raise gates.GateError(f"--cpu-lanes {o.cpu_lanes} needs the E0 calibration flag "
                                  f"({E0_FLAG}: cpu_arms_concurrent: true); run serially (--cpu-lanes 0)")
        for b in self.blocks:
            for r in b.runs:
                if not any(self._accepts(l, r) for l in self.lanes):
                    raise PoolError(f"no lane of layout {o.layout} (cpu_lanes={o.cpu_lanes}) can run {r.label}")
        if o.require_lease:
            gates.check_lease(o.who)
        if quiet:
            gates.check_serving_stopped()
        if not po.use_systemd and not smoke_only:
            raise gates.GateError("--no-systemd is for toy smoke tests only (D-003: every process in a MemoryMax scope)")
        if po.gpu_health and any(pair.arm_spec(j).gpu for j in self.jobs):
            from itrea.experiments import gpu_health
            if gpu_health.Sampler().sample() is None:
                raise gates.GateError("GPU health: nvidia-smi gave no sample, so no GPU run could be verified (D-003)")

    def memory_plan(self) -> tuple[gates.Budget | None, dict[str, gates.Budget], list[str]]:
        """The worst case of every lane busy at once: per lane the largest run budget it may take, summed with the
        reserve counted once. Returns (total, per lane, notes)."""
        cache: dict = {}
        per: dict[str, gates.Budget] = {}
        notes: list[str] = []
        for lane in self.lanes:
            worst = None
            for b in self.blocks:
                for r in b.runs:
                    if not self._accepts(lane, r):
                        continue
                    key = (lane.name, r.label, r.job.get("config"), r.job.get("ablation"),
                           json.dumps(r.job.get("overrides") or {}, sort_keys=True, default=str),
                           json.dumps(r.job.get("harness"), sort_keys=True, default=str))
                    if key not in cache:
                        try:
                            cache[key] = pair.job_budget(r.job, self._popts(lane))
                        except pair.RunError as e:
                            cache[key] = None
                            notes.append(f"{lane.name} {r.label}: {e}")
                    bud = cache[key]
                    if bud is not None and (worst is None or bud.need_gb > worst.need_gb):
                        worst = bud
            if worst is not None:
                per[lane.name] = worst
        total = None
        for bud in per.values():
            total = bud if total is None else total + bud
        return total, per, notes

    def memory_lines(self) -> list[str]:
        total, per, notes = self.memory_plan()
        if total is None:
            return ["memory: no run to budget"]
        avail = gates.mem_available_gb()
        out = [f"memory: {len(per)} lane(s) at once need up to {total.need_gb:.1f} GB ({total.describe()}); "
               f"MemAvailable now {avail:.1f} GB"]
        if total.need_gb > avail:
            out.append(f"WARNING: the worst case exceeds MemAvailable by {total.need_gb - avail:.1f} GB: not every lane "
                       f"can be admitted at once (lanes wait at the memory gate; fewer lanes would not)")
        return out + [f"memory: cannot budget {n}" for n in notes]

    def memory_arm_lines(self) -> list[str]:
        """D-003's arithmetic per treatment (label and harness) on the lane that takes it: every process's host
        MemoryMax and GPU cap (CuPy pool / cuOpt RMM), the CUDA overheads, Redis and the reserve — e.g. A9 under the
        ext harness: 9 + 6 + 24 host + 12 + 24 GPU + 2 x 1 + 20 = 97 GB (D-021 R29)."""
        out, seen = [], set()
        for b in self.blocks:
            for r in b.runs:
                lane = next((l for l in self.lanes if self._accepts(l, r)), None)
                key = (r.label, json.dumps(r.job.get("harness"), sort_keys=True, default=str), r.job.get("config"),
                       lane.name if lane else None)
                if lane is None or key in seen:
                    continue
                seen.add(key)
                po = self._popts(lane)
                try:
                    cfg, _ = pair.effective_config(r.job, po)
                    procs = pair.plan(r.job, po, Path("/nonexistent") / pair.run_id_for(r.job, po), "-", cfg)
                except pair.RunError as e:
                    out.append(f"memory per run: {r.label}: cannot plan ({e})")
                    continue
                bud = pair.budget_for(procs, po)
                parts = " + ".join(f"{p.role} {p.mem_gb:g}" for p in procs)
                peak = po.cuopt_gpu_peak_gb
                gpu = " + ".join(f"{p.role} {(peak if p.spec.kind == 'cuopt' and peak else p.gpu_gb):g}"
                                 for p in procs if p.gpu_gb is not None) or "0"
                hz = pair.job_harness(r.job)
                out.append(f"memory per run: {r.label} on {lane.name}"
                           + (f" (harness {json.dumps(r.job.get('harness'), sort_keys=True)})" if hz else "")
                           + f": host [{parts}] + GPU [{gpu}] + {bud.cuda_procs} CUDA x {bud.gpu_overhead_gb:g} + "
                             f"Redis {bud.redis_gb:g} + reserve {bud.reserve_gb:g} = {bud.need_gb:.1f} GB")
        return out

    def prepare(self) -> None:
        """Resume: skip finished blocks; a block with some but not all runs finished (or one left `running`) was split
        by an interruption — move its runs aside and rerun it whole."""
        popts = self._popts(None)
        for b in self.blocks:
            st = [pair.read_status(pair.run_dir_for(r.job, popts)) for r in b.runs]
            finished = [s in ("completed", "failed") for s in st]
            if self.o.rerun_failed and "failed" in st:
                self._move_block_aside(b, "rerun-failed")
                continue
            if all(finished):
                b.state = "done"
                for r, s in zip(b.runs, st):
                    r.state, r.status = "done", s
                self.counts["skipped"] += len(b.runs)
                self.counts["blocks_done"] += 1
                continue
            if any(s is not None for s in st):
                self._move_block_aside(b, "split block (resume)" if any(finished) else "stale run (resume)")
            prior = self._ref_rejected_before(b, popts)
            if prior:                              # D-021 R12: a resumed block keeps its rerun's stop off (and cap)
                b.ref_stop_off = f"rerun after ref_rejected in {prior} (void.jsonl, resume)"
                b.voids["ref_rejected"] = max(1, b.voids.get("ref_rejected", 0))
            for r in b.runs:                       # orphaned mailboxes of an earlier attempt
                done = pair.clear_stale(r.job, popts, wait_s=self.o.stale_wait_s)
                if done["killed"] or done["mailbox_removed"]:
                    self.log(f"cleared stale state of {done['run_id']}: {done}")

    def _ref_rejected_before(self, b: Block, popts: pair.PairOptions) -> str | None:
        """The run id of an earlier attempt of this block voided by `ref_rejected` (the set's void.jsonl), or None: on
        resume its rerun still runs with R12's stop off."""
        ids = {pair.run_id_for(r.job, popts) for r in b.runs}
        p = Path(popts.traces_dir) / "raw" / b.set / "void.jsonl"
        try:
            lines = p.read_text().splitlines()
        except OSError:
            return None
        for line in lines:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("run_id") in ids and str(d.get("reason") or "").startswith("ref_rejected"):
                return str(d["run_id"])
        return None

    def _move_block_aside(self, b: Block, reason: str) -> None:
        """Move every run directory of the block to void/ (evidence) and log it in void.jsonl. Processes of an earlier
        attempt still alive are stopped FIRST, while run.json (their pids) is still in place; pair.RunError if one
        survives SIGKILL (nothing more is moved; the caller refuses or halts)."""
        popts = self._popts(None)
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        for r in b.runs:
            rd = pair.run_dir_for(r.job, popts)
            if not rd.exists():
                continue
            rec = pair.read_record(rd) or {}
            done = pair.clear_stale(r.job, popts, wait_s=self.o.stale_wait_s)   # reads rd/run.json: before the move
            if done["killed"]:
                self.log(f"stopped processes {done['killed']} of the earlier attempt {done['run_id']}")
            dest = rd.parent / "void" / f"{rd.name}__{stamp}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(rd), str(dest))
            with open(rd.parent / "void.jsonl", "a") as f:
                f.write(json.dumps(dict(run_id=rd.name, block=b.id, arm=r.label, reason=reason,
                                        status_before=rec.get("status"), window=self.window,
                                        moved_to=str(dest.relative_to(rd.parent)),
                                        at=datetime.now().astimezone().isoformat())) + "\n")
        self.log(f"block {b.set}/{b.id} moved aside: {reason}")

    # ---------------------------------------------------------------- estimates
    def _est_run(self, r: Run) -> float:
        """Budget + the harness's own worst case per run: grace + kill_after + the SIGTERM/SIGKILL waits (15 s) +
        start-up (GPU arms 120 s: CUDA context, NVRTC, cuOpt warm-up; CPU 30 s). est_overhead_s overrides it."""
        o = self.o
        if o.est_overhead_s is not None:
            return float(r.job["budget"]) + float(o.est_overhead_s)
        po = o.pair_opts
        return (float(r.job["budget"]) + po.grace_s + po.kill_after_s + 15.0
                + (120.0 if pair.arm_spec(r.job).gpu else 30.0))

    def _est_backlog(self, extra: Block | None) -> float:
        """Seconds to finish every open block (+ extra): partner runs serially on the GPU lane, solver-only runs
        shared by the CPU lanes (or the GPU lane when serial). Never less than the longest single run left: a run
        cannot be shared between lanes, so dividing the backlog by the lane count alone underestimates a block."""
        blocks = [b for b in self.blocks if b.state == "open"] + ([extra] if extra else [])
        partner = solo = longest = 0.0
        for b in blocks:
            for r in b.runs:
                if r.state == "done":
                    continue
                e = self._est_run(r)
                longest = max(longest, e)
                if r.kind == "partner":
                    partner += e
                else:
                    solo += e
        cpu = [l for l in self.lanes if not l.gpu]
        if self.o.layout == "screen":
            return max(longest, (partner + solo) / max(1, len(self.lanes)))
        if not cpu:
            return partner + solo
        return max(longest, partner, solo / len(cpu))

    def _now(self) -> datetime:
        return datetime.now().astimezone()

    def _too_late(self, r: Run) -> str | None:
        """Just before a run starts (after its gate waits): a reason it would end after the deadline or the lease end
        by its own estimate (`_est_run`, no margin), or None."""
        o = self.o
        need = self._est_run(r)
        if o.deadline is not None and self._now() + timedelta(seconds=need) > o.deadline:
            return f"deadline {o.deadline:%H:%M}: {r.label} of {r.block.id} needs ~{need / 60:.0f} min"
        if o.require_lease:
            left = gates.lease_minutes_left() * 60
            if left < need:
                return f"lease ends in {left / 60:.0f} min; {r.label} of {r.block.id} needs ~{need / 60:.0f} min"
        return None

    def _may_open(self, b: Block) -> str | None:
        """Reason not to open a new block (a stop), or None."""
        o = self.o
        if self.stop_file.exists():
            return f"stop file {self.stop_file}"
        if o.max_blocks is not None and self.opened >= o.max_blocks:
            return f"max_blocks {o.max_blocks} reached"
        need = self._est_backlog(b)
        margin = float(o.open_margin_s)
        if o.deadline is not None and self._now() + timedelta(seconds=need + margin) > o.deadline:
            return (f"deadline {o.deadline:%H:%M}: the next block needs ~{need / 60:.0f} min "
                    f"+ {margin / 60:.0f} min margin")
        if o.require_lease:
            left = gates.lease_minutes_left() * 60
            if left < need + margin:
                return (f"lease ends in {left / 60:.0f} min; the next block needs ~{need / 60:.0f} min "
                        f"+ {margin / 60:.0f} min margin")
        return None

    # ---------------------------------------------------------------- scheduling
    def _next(self, lane: pair.Lane):
        """Under self.cv. Returns a Run to execute, None (nothing now: wait), or "DONE" (nothing ever again)."""
        if self.halt_reason:
            return "DONE"
        if self.abort.is_set():
            paused = self.watchdog is not None and self.watchdog.paused.is_set()
            if self.abort_kind == "thermal" and not self.inflight and not paused:
                self.log(f"thermal pause over ({gates.acpi_max_c()} °C): resuming")
                self.abort.clear()
                self.abort_kind, self.abort_text = None, ""
            else:
                return None
        # 1. a pending run of an open block that this lane can run (blocks in order, runs in Williams order)
        for b in self.blocks:
            if b.state == "open":
                for r in b.runs:
                    if r.state == "pending" and self._accepts(lane, r):
                        return r
        # 2. open the next block, unless stopping or too many blocks are open
        n_open = sum(1 for b in self.blocks if b.state in ("open", "voiding"))
        nxt = next((b for b in self.blocks if b.state == "unstarted"), None)
        if self.stopping is None and nxt is not None and n_open < (self.o.max_open_blocks or len(self.lanes)):
            why = self._may_open(nxt)
            if why:
                self.stopping = why
                self.log(f"stopping before block {nxt.set}/{nxt.id}: {why}")
                self.cv.notify_all()
            else:
                nxt.state = "open"
                self.opened += 1
                self.log(f"block {nxt.set}/{nxt.id} [{nxt.stratum}] opened: " + " ".join(r.label for r in nxt.runs))
                self.cv.notify_all()
                for r in nxt.runs:
                    if r.state == "pending" and self._accepts(lane, r):
                        return r
                return None
        # 3. nothing now; DONE only when no more work can reach this lane
        more = any(b.state in ("open", "voiding") for b in self.blocks) or \
            (self.stopping is None and any(b.state == "unstarted" for b in self.blocks))
        return None if more else "DONE"

    def lane_loop(self, lane: pair.Lane) -> None:
        while True:
            with self.cv:
                r = self._next(lane)
                while r is None:
                    self.cv.wait(timeout=5.0)
                    r = self._next(lane)
                if r == "DONE":
                    return
                r.state = "running"
                try:
                    r.run_dir = pair.run_dir_for(r.job, self._popts(lane))
                    budget = pair.job_budget(r.job, self._popts(lane))
                except Exception as e:  # noqa: BLE001 — a bad job: halt rather than leave the block hanging
                    self.log(f"cannot plan {r.label} of {r.block.id}: {e!r}")
                    self._halt(f"cannot plan {r.label} of {r.block.id}: {e!r}")
                    r.block.state, r.block.void_reason = "voiding", f"plan error: {e!r}"
                    self._finish(r, "not-run")
                    self.cv.notify_all()
                    continue
                self.inflight[lane.name] = Inflight(r, budget)       # not admitted: waiting lanes never count
                self.write_status("running")
            try:
                status = self._execute(lane, r)
            except Exception as e:  # noqa: BLE001 — a fault of the harness halts the run set
                self.log(f"runner error on {r.label} of {r.block.id}: {e!r}\n{traceback.format_exc()}")
                status = "void"
                with self.cv:
                    self._halt(f"runner error: {e!r}")
                    if r.block.state != "voiding":
                        r.block.state, r.block.void_reason = "voiding", f"runner error: {e!r}"
            with self.cv:
                self.inflight.pop(lane.name, None)
                self._finish(r, status)
                self.write_status("running")
                self.cv.notify_all()

    def _halt(self, reason: str) -> None:
        if self.halt_reason is None:
            self.halt_reason = reason
            self.log(f"HALT: {reason}")
        self.abort_kind = self.abort_kind or "halt"
        self.abort_text = self.abort_text or reason
        self.abort.set()

    def _on_breach(self, kind: str, reason: str) -> None:
        with self.cv:
            self.log(f"watchdog: {reason}")
            if kind == "memory":
                self._halt(reason)
            elif kind == "thermal" and not self.abort.is_set():
                self.abort_kind, self.abort_text = "thermal", reason
                self.thermal_episode += 1
                self.abort.set()
            self.cv.notify_all()

    def _heat_void(self, key: tuple, cause: str) -> None:
        """Under self.cv. D-021 R26: a window closes after `max_heat_voids` thermal or cap voids. One thermal pause
        episode is one void however many blocks it voided; each gpu_cap or thermal_watch run void is one. At the limit
        no new block opens (blocks already open finish: a stop never splits a block) and the pool ends `halted`."""
        self.heat_voids.add(key)
        n = self.o.max_heat_voids
        if n and len(self.heat_voids) >= n and self.breaker is None:
            kinds = sorted({k[0] for k in self.heat_voids})
            self.breaker = (f"R26: {len(self.heat_voids)} thermal / cap voids in this run set ({', '.join(kinds)}; last "
                            f"{cause}): no new block; the window closes")
            self.stopping = self.stopping or self.breaker
            self.log(f"BREAKER: {self.breaker}")
            self.cv.notify_all()

    def _run_use_gb(self, run: Run) -> float:
        """Anonymous memory the processes of an admitted run use now (each capped at its MemoryMax): already missing
        from MemAvailable, so the gate credits it back instead of counting it twice. GPU allocations are not in it,
        so their caps stay double-counted (conservative)."""
        rec = pair.read_record(run.run_dir) if run.run_dir is not None else None
        if not rec:
            return 0.0
        total = 0.0
        for pr in rec.get("processes", []):
            pid = pr.get("pid")
            if not pid or pr.get("exit_code") is not None or not pair._pid_runs(pid, rec.get("run_id", "\0")):
                continue
            use = gates.rss_anon_gb(pid)
            cap = pr.get("memory_max_gb")
            total += min(use, float(cap)) if cap else use
        return total

    def _gate_wait(self, lane: pair.Lane, r: Run) -> tuple[str, str] | None:
        """Every per-run gate, re-evaluated together in ONE loop until a single pass clears them all, so a long wait
        never releases a run on a stale lease or serving check: the box-window lease, the model servers (test and
        screening jobs), the thermal start gate, the fan governor for GPU runs (D-021 R24: alive, cap off; AUTO for a
        block requeued after a cap void), then D-003's memory formula over this run plus the runs already
        ADMITTED (lanes still waiting never count), against MemAvailable plus what the admitted runs already use.
        Admission is set in the same critical section as the check. Each wait's timeout starts at its first failure.
        Returns None (admitted, or an abort is set: _execute handles it), ("halt", reason), or ("requeue", reason): a
        memory wait that times out while other runs are in flight voids and requeues the block (D-002 §6: the memory
        gate is an external cause); with nothing admitted it halts."""
        o = self.o
        t_end = m_end = g_end = g_kind = r_end = None
        last_log = [0.0]

        def wait_log(msg: str) -> None:            # at most one line a minute per lane
            if time.monotonic() - last_log[0] >= 60.0:
                last_log[0] = time.monotonic()
                self.log(f"[{lane.name}] {msg}")

        while True:
            if self.abort.is_set():
                return None
            try:
                if o.require_lease:
                    gates.check_lease(o.who)
                if is_quiet(r.job) or lane.whole_box:
                    gates.check_serving_stopped()
            except gates.GateError as e:
                return ("halt", str(e))
            now = time.monotonic()
            if gates.rest_due():                   # D-009: honour fan-governor's rest — no new run until it clears
                if o.governor:                     # only the governor removes the flag: is it still running?
                    st = gates.fan_state()
                    if st is None or (st.get("age_s") or 0.0) >= gates.GOVERNOR_MAX_AGE_S:
                        r_end = r_end or now + o.governor_timeout_s
                        if now > r_end:
                            return ("halt", f"rest flag present but the fan governor is not running (state "
                                            f"{'missing' if st is None else str(st.get('age_s')) + ' s old'}; "
                                            f"waited {o.governor_timeout_s:.0f} s)")
                    else:
                        r_end = None
                wait_log("fan-governor rest flag: starting nothing until the box has rested")
                self.abort.wait(min(o.thermal_poll_s, 10.0))
                continue
            r_end = None
            t = gates.acpi_max_c()
            start_c = o.thermal_start_c if _arm_uses_gpu(r.job) else o.thermal_start_c_cpu
            if t is not None and t >= start_c:
                t_end = t_end or now + o.thermal_timeout_s
                if now > t_end:
                    return ("halt", f"thermal gate: still {t:.1f} °C after {o.thermal_timeout_s:.0f} s")
                wait_log(f"thermal gate: {t:.1f} °C >= {start_c:.0f} °C, waiting")
                self.abort.wait(min(o.thermal_poll_s, 10.0))
                continue
            t_end = None
            if o.governor and _arm_uses_gpu(r.job):    # D-021 R24 / R25: governor alive, cap off (AUTO: cap void)
                st = gates.fan_state()
                why = gates.governor_block(st, need_auto=r.block.wait_auto)
                if why:
                    stale = st is None or (st.get("age_s") or 0.0) >= gates.GOVERNOR_MAX_AGE_S
                    lim = o.governor_timeout_s if stale else o.thermal_timeout_s
                    if g_end is None or g_kind != stale:      # the wait's clock restarts when its kind changes
                        g_end, g_kind = now + lim, stale
                    if now > g_end:
                        return ("halt", f"{why} (waited {lim:.0f} s)")
                    wait_log(f"{why}; waiting")
                    self.abort.wait(min(o.thermal_poll_s, 10.0))
                    continue
                g_end = None
                r.block.wait_auto = False
            with self.cv:
                ent = self.inflight.get(lane.name)
                need = ent.budget if ent is not None else pair.job_budget(r.job, self._popts(lane))
                credit, n_adm = 0.0, 0
                for name, e in self.inflight.items():
                    if name == lane.name or not e.admitted:
                        continue
                    need = need + e.budget
                    credit += self._run_use_gb(e.run)
                    n_adm += 1
                avail = gates.mem_available_gb()
                if avail + credit >= need.need_gb:
                    if ent is not None:
                        ent.admitted = True
                    return None
            msg = (f"memory gate: MemAvailable {avail:.1f} GB + {credit:.1f} GB in use by {n_adm} admitted run(s) "
                   f"< {need.describe()}")
            m_end = m_end or now + o.mem_wait_s
            if now > m_end:
                return ("requeue" if n_adm else "halt", f"{msg} (waited {o.mem_wait_s:.0f} s)")
            wait_log(f"{msg}; waiting")
            self.abort.wait(o.mem_poll_s)

    def _execute(self, lane: pair.Lane, r: Run) -> str:
        gate = self._gate_wait(lane, r)
        if gate:
            kind, why = gate
            with self.cv:
                if kind == "requeue":
                    self.log(f"[{lane.name}] {r.label} of {r.block.id} not started: {why}; the block is voided and "
                             f"requeued (other runs are in flight)")
                    if r.block.state != "voiding":
                        r.block.state, r.block.void_reason = "voiding", f"memory gate: {why}"
                        r.block.requeue, r.block.requeue_cause = True, "memory"
                else:
                    self._halt(f"gate before {r.label} of {r.block.id}: {why}")
                    r.block.state, r.block.void_reason = "voiding", f"gate: {why}"
            return "not-run"
        if self.abort.is_set():
            with self.cv:
                if r.block.state != "voiding":
                    r.block.state, r.block.void_reason = "voiding", f"abort before start: {self.abort_text}"
                    r.block.requeue = self.abort_kind == "thermal"
                    r.block.requeue_cause = "thermal" if r.block.requeue else None
                    if r.block.requeue:
                        self._heat_void(("pause", self.thermal_episode), "thermal")
            return "not-run"
        late = self._too_late(r)
        if late:
            with self.cv:
                b = r.block
                if all(x is r or x.state == "pending" for x in b.runs):    # its first run: the block never opened
                    b.state = "unstarted"
                    self.opened -= 1
                    if self.stopping is None:
                        self.stopping = late
                    self.log(f"[{lane.name}] {r.label} of {b.id} not started: {late}; the block counts as not opened, "
                             f"no new block")
                    self.cv.notify_all()
                    return "deferred"
                self.log(f"[{lane.name}] {late}; block {b.id} is partly run, so it continues (a stop never splits a "
                         f"block)")
        popts = self._popts(lane)
        job = dict(r.job, ref_stop_off=r.block.ref_stop_off) if r.block.ref_stop_off else r.job
        self.log(f"[{lane.name}] start {pair.run_id_for(job, popts)} (block {r.block.id}, position "
                 f"{r.job.get('position')})" + (" with R12's stop off" if r.block.ref_stop_off else ""))
        rec = pair.run_pair(job, popts)
        st = rec["status"]
        if st == "void":
            with self.cv:
                cause = rec.get("void_cause")
                if cause == "gpu_health":
                    self._halt(f"GPU health voided {rec['run_id']}: {'; '.join(rec['reasons'])}")
                    r.block.requeue = False
                elif cause == "runner_error":          # a launcher / environment failure: halt for a look
                    self._halt(f"runner error in {rec['run_id']}: {'; '.join(rec['reasons'])[:300]}")
                    r.block.requeue = False
                elif cause == "gpu_cap" and ((rec.get("void_info") or {}).get("gpu_cap") or {}).get("unknown_lock"):
                    # a low clock with the governor reading cap=0 at the start and the end and no cap interval in its
                    # log: a lock the governor does not know about (left by a restart) and will not release
                    self._halt(f"unknown GPU clock lock in {rec['run_id']}: {rec['void_checks'].get('gpu_cap')} with "
                               f"the governor's cap off before and after and no cap interval (`nvidia-smi -rgc` "
                               f"needs root; look before resuming)")
                    r.block.requeue = False
                elif self.abort_kind == "thermal" or cause == "thermal_watch":    # 98 °C stop: as the thermal pause
                    r.block.requeue, r.block.requeue_cause = True, "thermal"
                    self._heat_void(("pause", self.thermal_episode) if self.abort_kind == "thermal"
                                    else ("thermal_watch", rec["run_id"], time.time()), cause or "thermal")
                elif cause in REQUEUE_CAUSES:          # D-021: requeued once, then skipped and listed
                    r.block.requeue, r.block.requeue_cause = True, cause
                    if cause == "ref_rejected":        # R12: the rerun of the whole block runs without the stop
                        r.block.ref_stop_off = f"rerun after ref_rejected in {rec['run_id']}"
                    if cause == "gpu_cap":
                        r.block.wait_auto = True
                        self._heat_void(("gpu_cap", rec["run_id"], time.time()), cause)
                if r.block.state != "voiding":
                    r.block.state = "voiding"
                    r.block.void_reason = f"{cause}: {'; '.join(rec['reasons'])[:300]}"
        return st

    def _finish(self, r: Run, status: str) -> None:
        """Under self.cv: book the run's outcome; finalize its block when nothing of it is in flight. "deferred" (too
        late to start, its block never opened: _execute) puts the run back."""
        if status == "deferred":
            r.state, r.status = "pending", None
            return
        r.state, r.status = "done", status
        if status in ("completed", "failed", "void"):
            self.counts[status] += 1
        b = r.block
        if b.state == "voiding":
            if not any(x.state == "running" for x in b.runs):
                try:
                    self._move_block_aside(b, b.void_reason or "void")
                except pair.RunError as e:          # a process of the block survives SIGKILL: nothing more runs
                    self._halt(f"cannot clear block {b.set}/{b.id}: {e}")
                    b.state = "abandoned"
                    return
                self.counts["blocks_void"] += 1
                for x in b.runs:
                    x.state, x.status = "pending", None
                skip = False
                if b.requeue and b.requeue_cause == "thermal":
                    b.thermal_voids += 1
                    if b.thermal_voids > self.o.max_thermal_requeues:      # the cap: give it up, move on
                        b.requeue, skip = False, True
                        popts = self._popts(None)
                        self.skipped.append(dict(set=b.set, block=b.id, reason="thermal", thermal_voids=b.thermal_voids,
                                                 run_ids=[pair.run_id_for(x.job, popts) for x in b.runs],
                                                 void_reason=b.void_reason))
                        self.counts["blocks_skipped_thermal"] += 1
                        self.log(f"block {b.set}/{b.id} skipped: voided by the thermal pause {b.thermal_voids} times "
                                 f"(--max-thermal-requeues {self.o.max_thermal_requeues})")
                elif b.requeue and b.requeue_cause in REQUEUE_CAUSES:    # D-021: once per cause, then skipped
                    c = b.requeue_cause
                    b.voids[c] = b.voids.get(c, 0) + 1
                    if b.voids[c] > self.o.max_cause_requeues:
                        b.requeue, skip = False, True
                        popts = self._popts(None)
                        self.skipped.append(dict(set=b.set, block=b.id, reason=c, voids=b.voids[c],
                                                 run_ids=[pair.run_id_for(x.job, popts) for x in b.runs],
                                                 void_reason=b.void_reason))
                        sk = self.counts["blocks_skipped_cause"]
                        sk[c] = sk.get(c, 0) + 1
                        self.log(f"block {b.set}/{b.id} skipped: voided by {c} {b.voids[c]} times "
                                 f"(at most {self.o.max_cause_requeues} rerun)")
                if b.requeue:                        # every rerun is counted by its cause (D-021 R15: reported)
                    rq = self.counts["requeues"]
                    rq[b.requeue_cause or "?"] = rq.get(b.requeue_cause or "?", 0) + 1
                b.state = "unstarted" if b.requeue else ("skipped" if skip else "abandoned")
                b.requeue, b.requeue_cause = False, None
                if b.state == "unstarted":           # rerun first, before any new block
                    self.blocks.remove(b)
                    i = next((k for k, y in enumerate(self.blocks) if y.state == "unstarted"), len(self.blocks))
                    self.blocks.insert(i, b)
                    self.opened -= 1
            return
        if all(x.state == "done" for x in b.runs):
            b.state = "done"
            self.counts["blocks_done"] += 1
            self.log(f"block {b.set}/{b.id} done: " + " ".join(f"{x.label}={x.status}" for x in b.runs))

    # --------------------------------------------------------------------- run
    def plan_lines(self) -> list[str]:
        out = []
        popts = self._popts(None)
        for i, b in enumerate(self.blocks):
            st = [pair.read_status(pair.run_dir_for(r.job, popts)) for r in b.runs]
            out.append(f"{i:4d} {b.set}/{b.id} [{b.stratum}] " +
                       " ".join(f"{r.label}{'(' + s + ')' if s else ''}" for r, s in zip(b.runs, st)))
        return out

    def run(self) -> int:
        o = self.o
        self.lanes = self._lanes()
        self.blocks = order_blocks(make_blocks(self.jobs, o.order_seed), o.order_seed)
        if o.dry_run:
            for line in self.plan_lines():
                print(line)
            print(f"{len(self.blocks)} blocks, {len(self.jobs)} runs; lanes: "
                  + ", ".join(f"{l.name}(x925={l.x925}, a725={l.a725}{', gpu' if l.gpu else ''})" for l in self.lanes))
            for line in self.memory_lines() + self.memory_arm_lines():
                print(line)
            print(f"thermal: GPU arms start below {o.thermal_start_c:g} °C, CPU-only below "
                  f"{o.thermal_start_c_cpu:g} °C; pause / void at {o.pause_c:g} °C; cap void below "
                  f"{o.pair_opts.cap_floor_mhz} MHz busy SM median or on a governor clock cap")
            if o.governor:
                st = gates.fan_state()
                print(f"fan governor: {st['raw'] if st else 'no state file'}"
                      + (f" ({st['age_s']:g} s old)" if st else "")
                      + f" -> {gates.governor_block(st) or 'a GPU run may start'}")
            else:
                print("fan governor: not checked (--no-governor)")
            try:                                  # read-only: the same refusals a real start would meet
                self.preflight()
                print("preflight: passes (lease, serving, freeze, harness)")
            except (gates.GateError, PoolError, pair.RunError) as e:
                print(f"preflight: would refuse: {e}")
            return 0
        try:
            with gates.RunLock(o.lock_path):
                return self._run_locked()
        except gates.GateError as e:            # the lock is held by another runner
            self.log(f"refused: {e}")
            return 2

    def _run_locked(self) -> int:
        o = self.o
        try:
            self.preflight()
        except (gates.GateError, PoolError) as e:
            self.log(f"refused: {e}")
            self.write_status("refused")
            return 2
        if o.pair_opts.mailbox == "redis":
            from itrea.experiments import exchange_ctl
            exchange_ctl.up(log=self.log)
        try:
            self.prepare()
        except pair.RunError as e:              # an earlier attempt's process survives SIGKILL
            self.log(f"refused: {e}")
            self.write_status("refused")
            return 2
        for line in self.memory_lines() + self.memory_arm_lines():     # D-003's arithmetic, per treatment too
            self.log(line)
        self._resources_record()
        self.log(f"pool {self.window}: {len(self.blocks)} blocks, {len(self.jobs)} runs, lanes "
                 + ", ".join(l.name for l in self.lanes) + f", mailbox {o.pair_opts.mailbox}")
        self.write_status("running")
        if o.watchdog:
            self.watchdog = gates.Watchdog(self._on_breach, pause_c=o.pause_c, poll_s=o.watchdog_poll_s)
            self.watchdog.start()
        if o.pair_opts.gpu_health:                 # ONE sampler for every lane (2 nvidia-smi calls/s, not 2 per lane)
            from itrea.experiments import gpu_health
            self.sampler = gpu_health.Sampler(
                own_pids=self._inflight_pids, period_s=o.pair_opts.gpu_period_s,
                tolerate=(gpu_health.is_serving_pid if o.allow_serving else None))   # D-010: dev runs beside [local LLM agent]
            self.sampler.start()
        threads = [threading.Thread(target=self.lane_loop, args=(l,), name=f"lane-{l.name}", daemon=True)
                   for l in self.lanes]
        for t in threads:
            t.start()
        try:
            for t in threads:
                while t.is_alive():
                    t.join(timeout=1.0)
        except KeyboardInterrupt:
            with self.cv:
                self._halt("interrupted (KeyboardInterrupt)")
                self.cv.notify_all()
            for t in threads:
                t.join(timeout=600)
        finally:
            if self.watchdog:
                self.watchdog.stop()
            if self.sampler is not None:
                self.sampler.stop()
        halted = self.halt_reason or self.breaker
        state = "halted" if halted else ("stopped" if self.stopping else "done")
        self.write_status(state)
        self.log(f"pool {self.window} {state}: {json.dumps(self.counts)}"
                 + (f"; {halted}" if halted else "")
                 + (f"; {self.stopping}" if self.stopping and not halted else ""))
        return 3 if halted else 0

    def _resources_record(self) -> None:
        """Once per pool session: what the run set was allotted and what else ran (kept OUT of traces/raw)."""
        o = self.o
        po = o.pair_opts
        rec = dict(window=self.window, started=datetime.now().astimezone().isoformat(), layout=o.layout,
                   lanes=[dataclasses.asdict(l) for l in self.lanes], cpu_lanes=o.cpu_lanes,
                   e0_concurrent=e0_concurrent(o.root), mailbox=po.mailbox, use_systemd=po.use_systemd,
                   memory_max_gb=dict(solver=po.solver_mem_gb, ea=po.ea_mem_gb, cuopt=po.cuopt_mem_gb),
                   gpu_mem_gb=po.gpu_mem_gb, reserve_gb=po.reserve_gb, freeze=getattr(self, "freeze", None),
                   thermal_start_c=o.thermal_start_c, thermal_start_c_cpu=o.thermal_start_c_cpu, gates=gates.snapshot(), env=record_env.record(),
                   jobs=len(self.jobs), blocks=len(self.blocks), order_seed=o.order_seed,
                   pause_c=o.pause_c, cap_floor_mhz=po.cap_floor_mhz, governor=o.governor,
                   fan_state=gates.fan_state() if o.governor else None,
                   harnesses=sorted({json.dumps(j.get("harness"), sort_keys=True) for j in self.jobs
                                     if j.get("harness") is not None}))
        d = Path(po.traces_dir) / "resources"
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / f"pool-{self.window}.json").write_text(json.dumps(rec, indent=1, default=str))
        except OSError as e:
            self.log(f"resources record not written: {e!r}")


# ------------------------------------------------------------------------------------------------------- CLI
def _deadline(s: str | None) -> datetime | None:
    if not s:
        return None
    now = datetime.now().astimezone()
    if "T" in s or "-" in s:
        t = datetime.fromisoformat(s)
        return t if t.tzinfo else t.astimezone()
    h, m = (int(x) for x in s.split(":"))
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return t if t > now else t + timedelta(days=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="itrea.experiments.pool", description=__doc__.split("\n\n")[0])
    ap.add_argument("jobs", type=Path)
    ap.add_argument("--who", default=WHO, help=f"must be {WHO!r}: the run set runs only under this project's lease")
    ap.add_argument("--no-lease", action="store_true", help="toy smoke tests only (every job split == smoke)")
    ap.add_argument("--layout", choices=["pair", "screen", "full"], default="pair",
                    help="pair: GPU lane g0 (+ --cpu-lanes); screen: nine X925 lanes; full: one lane with all 20 cores "
                         "and the GPU, arm A6m only (D-020; needs --cuopt-mem-gb 24 --gpu-mem-gb 36)")
    ap.add_argument("--cpu-lanes", type=int, default=0)
    ap.add_argument("--mailbox", choices=["file", "redis"], default="file")
    ap.add_argument("--deadline", help="HH:MM or ISO time: no new block that would end after it")
    ap.add_argument("--allow-unfrozen", action="store_true", help="non-test jobs without the prereg-v1 tag")
    ap.add_argument("--allow-serving", action="store_true",
                    help="model servers up: tuning / exploratory jobs only (never test or screening)")
    ap.add_argument("--rerun-failed", action="store_true", help="non-test only: rerun blocks with a failed run")
    ap.add_argument("--no-systemd", action="store_true", help="no scopes / MemoryMax (toy instances only)")
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--solver-mem-gb", type=float, default=9.0)
    ap.add_argument("--ea-mem-gb", type=float, default=6.0)
    ap.add_argument("--cuopt-mem-gb", type=float, default=12.0)
    ap.add_argument("--gpu-mem-gb", type=float,
                    help="override the configuration's gpu_mem_gb (CuPy pool, RMM); refused for test jobs")
    ap.add_argument("--cuopt-gpu-peak-gb", type=float, help="cuOpt's measured RMM peak for the memory gate")
    ap.add_argument("--traces-dir", type=Path, default=paths.TRACES)
    ap.add_argument("--logs-dir", type=Path, default=paths.ROOT / "logs")
    ap.add_argument("--order-seed", type=int, default=MASTER_SEED)
    ap.add_argument("--max-blocks", type=int)
    ap.add_argument("--max-thermal-requeues", type=int, default=1,
                    help="rerun a block voided by the thermal pause at most this often; a further void skips it")
    ap.add_argument("--pause-c", type=float, default=gates.THERMAL_PAUSE_C,
                    help="the watchdog's thermal pause: at or above it the blocks in flight are voided and rerun once "
                         "the box is below 80 °C (default 92; D-021 M2's soak: 90; never above 92)")
    ap.add_argument("--cap-floor-mhz", type=float, default=gpu_health.CAP_FLOOR_MHZ,
                    help="a GPU run whose busy SM median is below this, or that overlaps a fan-governor clock cap, is "
                         "a cap void (gpu_cap: the block is rerun once after the fan is back on AUTO; D-021 R25); "
                         "the 2,000-MHz stuck-clock halt is separate")
    ap.add_argument("--max-heat-voids", type=int, default=2,
                    help="D-021 R26: after this many thermal or cap voids (one thermal pause = one) open no new block "
                         "and end the pool `halted` (0 = off)")
    ap.add_argument("--no-governor", action="store_true",
                    help="do not require the fan governor (alive, cap off) before GPU runs: hosts without it only")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.who != WHO:
        print(f"error: --who must be {WHO!r} (this project's box-window lease); got {a.who!r}", file=sys.stderr)
        return 2
    try:
        jobs = load_jobs(a.jobs)
    except (OSError, PoolError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    po = pair.PairOptions(traces_dir=a.traces_dir, mailbox=a.mailbox, use_systemd=not a.no_systemd, pin=not a.no_pin,
                          solver_mem_gb=a.solver_mem_gb, ea_mem_gb=a.ea_mem_gb, cuopt_mem_gb=a.cuopt_mem_gb,
                          gpu_mem_gb=a.gpu_mem_gb, cuopt_gpu_peak_gb=a.cuopt_gpu_peak_gb, cap_floor_mhz=a.cap_floor_mhz)
    from itrea.experiments import make_jobs
    names, roots = make_jobs.candidate_names(), make_jobs.data_roots()
    if a.no_systemd and not all(j.get("split") == "smoke" and make_jobs.is_toy(j, names, roots) for j in jobs):
        print("error: --no-systemd is for toy smoke tests only (every job split == smoke on a toy instance)",
              file=sys.stderr)
        return 2
    opts = PoolOptions(who=a.who, require_lease=not a.no_lease, layout=a.layout, cpu_lanes=a.cpu_lanes,
                       deadline=_deadline(a.deadline), allow_unfrozen=a.allow_unfrozen, allow_serving=a.allow_serving,
                       rerun_failed=a.rerun_failed, order_seed=a.order_seed, logs_dir=a.logs_dir,
                       max_blocks=a.max_blocks, max_thermal_requeues=a.max_thermal_requeues, dry_run=a.dry_run,
                       pause_c=a.pause_c, governor=not a.no_governor, max_heat_voids=a.max_heat_voids,
                       pair_opts=po)
    po.tolerate_serving_gpu = bool(a.allow_serving)
    prev = pair.install_term_handlers()        # SIGTERM / SIGHUP -> the KeyboardInterrupt cleanup (halt, stop runs)
    if not a.no_pin:
        pair.pin_runner()                      # the runner's own work off the timed cores (before any thread starts)
    try:
        return Pool(jobs, opts).run()
    except (PoolError, pair.RunError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted before the lanes started", file=sys.stderr)
        return 130
    finally:
        pair.restore_handlers(prev)


if __name__ == "__main__":
    raise SystemExit(main())
