"""Run directories -> `datasets/runs.parquet` (one row per run) and `datasets/incumbents_checked.parquet` (one row per
stored incumbent of every process), the two tables every result of I-TREA II is computed from (vault D-002 §4,
`10-design/Trace format and metrics.md`).

Pattern adapted from berkorbay/optopt@4dd48395, MIT (`src/optopt/analysis/build_runs.py`): one row per run, every
metric cut at the budget, no silent reference fallback. What is new, and why:

- **Only checked incumbents count.** Every `<role>.incumbents.npz` is checked by `itrea.analysis.check_solutions`
  (another library, exact arithmetic); trajectories use the RECOMPUTED objective of the incumbents that pass. Failed
  and unchecked incumbents are dropped from the trajectory and counted (`n_inc_failed`, `n_inc_unchecked`,
  `n_obj_mismatch`). Check results are cached in `incumbents_checked.parquet`, keyed by (run_id, role, sha256 of the
  npz, `check_solutions.CHECKER_VERSION`), so a rebuild without `--recheck` needs neither the instances nor the time,
  and a changed checker re-checks instead of reusing old verdicts; `--recheck` re-reads every model. The instance
  file's sha256 is recorded per row (`instance_sha256`, not in the key); `--verify-instances` re-checks cached rows
  whose instance file changed.
- **Three trajectories per run:** the portfolio (`pt`, best held by any process at every moment — primary), what
  SCIP itself held (`solver_*`, role `solver` only, including solutions it accepted from the EA), the EA's own
  (`ea_*`). Synthetic rows (`synthetic` True, arm `A0+A5`) hold the **uncoupled portfolio** of D-002 v3 H2: the
  pointwise best of the A0 and A5 trajectories of the same block.
- **References** resolved per instance by `itrea.benchmarks.references.resolve` (frozen reference.csv → MIPLIB solu →
  provisional best of the SCREEN/REF runs → best of study), with `ref_kind`. Whether an instance is MIPLIB (and so
  scored against .solu) is decided from its instance PATH, not the job's set name (`reference_datasets`; the sets are
  screen1/e2/…, review 2026-10-06 F1). `pt_mf` is the MIPfeas-style integral (D-002 §4 secondary). `pt_bos`
  repeats P against the best checked value of the whole study (D-002 sensitivity analysis). `pt_screen` uses the
  screening reference only (.solu, else the best SCREEN run) — the pool freeze ranks on it, so the pool does not move
  when reference.csv is frozen later (D-002 orders the pool before the reference). `sign_crossing*` is persistent:
  the held incumbent crossed the reference's sign for >= 5 % of [0, T] (`*_share`; `*_any` = any crossing, for
  information only).
- **Clock rule:** `n_inc_bad_t` / `pre_t0` count stored incumbents stamped before t0 (t < 0) or before the trace
  clock started (t NaN); prereg refuses such runs.
- **Post accounting** (D-002 integrity): every EA post up to T is tried-accepted, tried-rejected, superseded-untried (a
  newer post replaced it — the mailbox keeps only the latest) or pending-at-T, from the EA's `post` events and the
  solver's `verdict` events. Tried-accepted = `accepted` (the SCIP adapter's meaning: SCIP's own feasibility check
  passed), `posts_stored` = also kept by `trySol`; a post of several rows counts as accepted if any row is. Verdicts
  are read from the run directory's `verdicts.jsonl` (monotonic time, converted with t0) only when the solver wrote
  no verdict events.
- **Three-way arm A9** (SCIP <-> EA <- cuOpt, D-015): the portfolio is the best of all three processes and each has
  its own per-role P (`solver_*`, `ea_*`, `cuopt_*`), as for every run. cuOpt's contribution through the EA is counted
  up to T (`cuopt_attribution`): `cuopt_inject_msgs` / `cuopt_injected` (cuOpt messages the EA read / rows it
  injected, `inject_cuopt` events), `ea_inc_from_cuopt` (checked EA incumbents that are unchanged cuOpt vectors, source
  `cuopt`), `cuopt_posts` (EA posts carrying such a vector, `post` events' `cuopt_rows`), `cuopt_posts_accepted` /
  `cuopt_posts_became_incumbent` (those posts SCIP accepted / took as its incumbent, from verdicts with `cuopt: true`,
  one per post), `solver_inc_from_cuopt` (checked SCIP incumbent records made from one: SCIP records them with source
  `cuopt`, so `solver_inc_from_ea` counts only EA-made vectors). The post columns (`posts`, `posts_accepted`,
  `posts_became_incumbent` …) count EVERY EA post, cuOpt's unchanged vectors included: in A9 the EA-made share is
  `posts_*` minus `cuopt_posts*`. Units in `cuopt_attribution`. Runs whose EA does not read cuOpt's channels
  (`reads_cuopt`: decided from the EA's own record) leave these columns missing (-1), not 0. A9's matched control is
  A3x (A3 with A9's EA footprint, no cuOpt); A3 differs from A9 by the EA's cores as well.
- **GPU back-off** (A9's EA, D-018; `backoff_stats`, from the EA summary's `backoff` block — missing for runs
  without the governor): `bo_mode`; `bo_busy_share` (busy / wall seconds since t0) and `bo_paused_s`; phase Y's length
  and exit reason (`bo_phase_y_s`, `bo_phase_y_exit`: peek | tick | own | expiry | never); wall seconds per level
  (`bo_level0_s` … `bo_level3_s`; NaN in forced mode); material leads and assists (`bo_n_lead`, `bo_n_assist`); the
  EA's own-feasible and first-lead times (`bo_own_feasible_s`, `bo_first_lead_s` wall, `bo_first_lead_busy_s` busy);
  the first step down in busy seconds (`bo_first_escalation_s`: in shadow mode the would-be one) and their number;
  wakes; the EA's lane steps (`bo_ls_steps`). `itrea backoff-report` reads the run directories for the rest.
- **Crashes are kept and truncated** (no event after the crash exists to count); `crashed`, `crashed_roles`. A
  process the runner killed after a partner's proven outcome (`processes[*].stopped_by_runner`, e.g. cuOpt in A2 / A9
  after SCIP proved optimality) is not a crash.
- **GPU-health voiding:** a GPU run whose `run.json["gpu_health"]` verdict is "void" (median busy SM clock < 2,000 MHz
  or foreign GPU processes in > 2 % of samples; `itrea.experiments.gpu_health`) sets `gpu_void`, and every run of its
  block (instance × seed × budget) gets `block_void`.

Run directory contract (`traces/raw/<set>/<run_id>/`, `run_id = <stem>__<arm>__s<seed>__<budget>s__<cfg-hash>`;
the stem may itself contain "__", so the id is parsed from the right). Keys read, all optional (`.get`):
`run.json`: run_id, set, instance, arm, seed, budget, config_hash, t0, attempt, window_id, williams_pos,
git_rev | git_revision, prereg_tag, gpu_health {verdict, median_sm_mhz, foreign_share, gpu_run},
processes [{role, exit_code | returncode}] or {role: {...}} or exit_codes {role: code};
`<role>.summary.json`: instance, arm, seed, budget, config_hash, sense, t0, exit_reason, status, backend, solver,
startup_s (number or {step: seconds}) | startup {step: seconds}, exchange_s, exchange_bytes (top level or counters);
`<role>.events.jsonl` kinds: incumbent, bound, post, verdict, frontier, callback.

    python -m itrea.analysis.build_runs [--raw traces/raw] [--out datasets] [--recheck | --verify-instances]
                                        [--workers N] [--strict]
Untimed analysis: OMP_NUM_THREADS=1, never beside timed runs with default thread counts (D-003).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from itrea.analysis import metrics as M
from itrea.analysis.check_solutions import CHECKER_VERSION, ModelCache, check_incumbents, file_sha256, producer_of
from itrea.benchmarks import references as R
from itrea.paths import DATASETS, ROOT, TRACES, resolve
from itrea.trace import read_events

ROLES = ("solver", "solver2", "ea", "cuopt")
SOLVER_ROLE, EA_ROLE = "solver", "ea"
GPU_ARMS = ("A2", "A3", "A3r", "A5", "A5x0", "A5y0", "A6", "H2", "H3", "A9", "A3x")
EA_GPU_ARMS = ("A3", "A3r", "A5", "A5x0", "A5y0", "H3", "A9", "A3x")
CUOPT_X_ARMS = ("A9",)                    # the EA reads cuOpt's channels (exchange.from_cuopt, D-015)
EA_CPU_ARMS = ("A4",)
SCREEN_ARM = "SCREEN"
REF_ARMS = (SCREEN_ARM, "REF")
UNCOUPLED = ("A0", "A5", "A0+A5")         # D-002 v3 H2: solver alone, EA alone, the computed portfolio
CLOCK_FLOOR_MHZ, FOREIGN_MAX_SHARE = 2000.0, 0.02
ROLE_KEYS = ("pt", "t_first", "t_first_event", "t_gap5", "t_gap5_event", "t_gap1", "t_gap1_event", "final_gap",
             "best_obj", "n_steps", "sign_crossing", "beats_ref", "pt_mf")
SOLU_PATH = "data/miplib/solu.txt"

INC_COLUMNS = {
    "run_id": "str", "role": "str", "k": "int", "t": "float", "source": "str", "obj_reported": "float",
    "obj_recomputed": "float", "checked": "bool", "ok": "bool", "feasible": "bool", "obj_ok": "bool", "reason": "str",
    "worst_bound": "float", "worst_int": "float", "worst_row_rel": "float", "worst_row_scip": "float",
    "obj_rel_err": "float", "unknown_vars": "int", "missing_vars": "int", "exact_rows": "int", "n_viol_rows": "int",
    "producer": "str", "reader": "str", "sense": "int", "npz_sha256": "str", "beats_ref": "bool",
    "checker_version": "str", "instance_sha256": "str",
}


# --------------------------------------------------------------------------------------------- run directories
def parse_run_id(run_id: str) -> dict:
    """`<stem>__<arm>__s<seed>__<budget>s__<cfg>` parsed from the right ({} if it does not have that shape)."""
    parts = str(run_id).split("__")
    if len(parts) < 5 or not parts[-3].startswith("s") or not parts[-2].endswith("s"):
        return {}
    try:
        seed = int(parts[-3][1:])
        budget = float(parts[-2][:-1])
    except ValueError:
        return {}
    return dict(stem="__".join(parts[:-4]), arm=parts[-4], seed=seed, budget=budget, config_hash=parts[-1])


def _read_json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def is_run_dir(p: Path) -> bool:
    return p.is_dir() and ((p / "run.json").exists() or any(p.glob("*.events.jsonl")))


def is_void_dir(d: str | os.PathLike, raw_root: str | os.PathLike) -> bool:
    """True for a run directory the pool moved aside as void (`<set>/void/<run_id>__<stamp>/`)."""
    try:
        return "void" in Path(d).relative_to(Path(raw_root)).parts[:-1]
    except ValueError:
        return False


def discover(raw_root: str | os.PathLike, include_void: bool = False) -> list[Path]:
    """Run directories under the raw root. Runs the pool moved aside as void (`<set>/void/<run_id>__<stamp>/`, the
    whole block is rerun) are skipped unless `include_void`."""
    root = Path(raw_root)
    if not root.is_dir():
        return []
    found = {f.parent for pat in ("run.json", "*.events.jsonl") for f in root.rglob(pat)}
    if not include_void:
        found = {d for d in found if not is_void_dir(d, root)}
    return sorted(found)


@dataclass
class Run:
    path: Path
    set: str
    run_id: str
    instance: str | None          # repository-relative instance path
    name: str                     # instance stem
    arm: str
    seed: int
    budget: float
    config_hash: str
    run_json: dict
    summaries: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)

    def summary(self, role: str) -> dict:
        return self.summaries.get(role, {})


def load_run(path: str | os.PathLike, raw_root: str | os.PathLike | None = None) -> Run:
    d = Path(path)
    rj = _read_json(d / "run.json")
    roles = sorted({f.name.split(".")[0] for f in d.glob("*.events.jsonl")} |
                   {f.name.split(".")[0] for f in d.glob("*.summary.json")} |
                   {f.name[: -len(".incumbents.npz")] for f in d.glob("*.incumbents.npz")})
    summ = {r: _read_json(d / f"{r}.summary.json") for r in roles}
    ev = {r: (read_events(d / f"{r}.events.jsonl") if (d / f"{r}.events.jsonl").exists() else []) for r in roles}
    run_id = str(rj.get("run_id") or d.name)
    pr = parse_run_id(run_id) or parse_run_id(d.name)

    def ident(key, default=None):
        if rj.get(key) not in (None, ""):
            return rj[key]
        for r in roles:
            if summ[r].get(key) not in (None, ""):
                return summ[r][key]
        return pr.get(key, default)

    inst = ident("instance")
    set_name = rj.get("set")
    if not set_name and raw_root is not None:
        try:
            rel = d.resolve().relative_to(Path(raw_root).resolve())
            set_name = rel.parts[0] if len(rel.parts) > 1 else ""
        except ValueError:
            set_name = ""
    rc = rj.get("config") if isinstance(rj.get("config"), dict) else {}
    cfg = rj.get("config_hash") or rc.get("ea_hash") or rc.get("run_hash") or \
        summ.get(EA_ROLE, {}).get("config_hash") or pr.get("config_hash") or ""
    name = rj.get("name") or (R.stem(inst) if inst else pr.get("stem", d.name))
    return Run(path=d, set=str(set_name or ""), run_id=run_id, instance=str(inst) if inst else None,
               name=str(name), arm=str(ident("arm", "") or ""),
               seed=_int(ident("seed", -1), -1),
               budget=float(ident("budget", math.nan) or math.nan), config_hash=str(cfg), run_json=rj,
               summaries=summ, events=ev)


# ------------------------------------------------------------------------------------------- incumbent checks
_WORKER_CACHE: ModelCache | None = None
_INSTANCE_SHA: dict[tuple, str] = {}


def instance_sha256(instance: str | None) -> str | None:
    """sha256 of an instance file (None if it is missing), memoised per process by (path, size, mtime)."""
    if not instance:
        return None
    p = resolve(instance)
    try:
        st = p.stat()
    except OSError:
        return None
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _INSTANCE_SHA:
        _INSTANCE_SHA[key] = file_sha256(p)
    return _INSTANCE_SHA[key]


def _check_task(task: tuple) -> list[dict]:
    """(run_id, role, npz, instance, producer, sha[, stale]) -> per-incumbent rows. Runs in a worker or inline.
    `stale` names the checker version(s) of a cached verdict that could not be reused (empty if none)."""
    global _WORKER_CACHE
    run_id, role, npz, instance, producer, sha = task[:6]
    stale = task[6] if len(task) > 6 else ""
    if _WORKER_CACHE is None:
        _WORKER_CACHE = ModelCache()
    if instance is None or not resolve(instance).exists():
        from itrea.trace import load_incumbents
        _, recs = load_incumbents(npz)
        why = "unchecked: instance file missing" if instance else "unchecked: no instance path in the run"
        if stale:
            why += f"; the cached verdict (checker version {stale}) is stale, current version {CHECKER_VERSION}"
        return [dict(run_id=run_id, role=role, k=k, t=t, source=s, obj_reported=o, checked=False, ok=False,
                     feasible=False, obj_ok=False, reason=why, producer=producer, npz_sha256=sha,
                     checker_version=CHECKER_VERSION, instance_sha256="")
                for k, (t, o, s, _) in enumerate(recs)]
    rows = check_incumbents(npz, instance, producer, _WORKER_CACHE)
    ish = instance_sha256(instance) or ""
    return [dict(run_id=run_id, role=role, npz_sha256=sha, checker_version=CHECKER_VERSION, instance_sha256=ish, **r)
            for r in rows]


def check_runs(runs: list[Run], cached: pd.DataFrame | None = None, recheck: bool = False,
               workers: int = 1, log=print, verify_instances: bool = False) -> pd.DataFrame:
    """Checked-incumbent table for every role of every run.

    Cache hits by (run_id, role, npz sha256, CHECKER_VERSION): a verdict of another checker version is re-checked, or
    reported as stale (unchecked) when the instance file is gone — never silently reused. The instance file's sha256
    is recorded at check time but is NOT part of the key (a rebuild needs neither the instances nor the time);
    `verify_instances` re-checks every cache hit whose instance file changed since (a missing file is logged and the
    cached verdict kept). With `recheck` everything is re-checked and, if `cached` is given, the instance files and
    verdicts that changed since the cached check are logged."""
    rows, tasks = [], []
    cache_idx: dict[tuple, pd.DataFrame] = {}
    stale_idx: dict[tuple, str] = {}
    old = None
    if cached is not None and len(cached):
        cached = cached.copy()
        for col in ("checker_version", "instance_sha256"):
            if col not in cached.columns:
                cached[col] = ""
            cached[col] = cached[col].fillna("").astype(str)
        if recheck:
            old = cached
        else:
            for key, g in cached.groupby(["run_id", "role", "npz_sha256"], sort=False):
                if not g["checked"].astype(bool).all():     # an "unchecked" row (model missing) is never reused
                    continue
                vers = sorted(set(g["checker_version"]))
                if vers == [CHECKER_VERSION]:
                    cache_idx[key] = g
                else:
                    stale_idx[key] = ",".join(v or "(none)" for v in vers)
    n_stale = n_changed = n_unverifiable = 0
    for run in runs:
        _recover_parts(run, log)
        for npz in sorted(run.path.glob("*.incumbents.npz")):
            role = npz.name[: -len(".incumbents.npz")]
            sha = file_sha256(npz)
            key = (run.run_id, role, sha)
            hit = cache_idx.get(key)
            if hit is not None and verify_instances:
                cur = instance_sha256(run.instance)
                if cur is None:
                    n_unverifiable += 1
                elif set(hit["instance_sha256"]) != {cur}:
                    n_changed += 1
                    log(f"{run.run_id}/{role}: instance file changed since its check: re-checking")
                    hit = None
            if hit is not None:
                rows += hit.to_dict("records")
                continue
            n_stale += key in stale_idx
            prod = producer_of(role, run.summary(role), run.arm)
            tasks.append((run.run_id, role, str(npz), run.instance, prod, sha, stale_idx.get(key, "")))
    tasks.sort(key=lambda t: (str(t[3]), t[0], t[1]))     # same instance together: model cache hits
    if n_stale:
        log(f"{n_stale} cached incumbent file(s) were checked by another checker version: re-checking")
    if n_unverifiable:
        log(f"--verify-instances: {n_unverifiable} cached incumbent file(s) kept unverified (instance file missing)")
    if tasks:
        log(f"checking {len(tasks)} incumbent files ({len(rows)} rows from the cache)")
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(workers, initializer=_worker_env) as ex:
            for r in ex.map(_check_task, tasks, chunksize=1):
                rows += r
    else:
        for t in tasks:
            rows += _check_task(t)
    out = _normalise(pd.DataFrame(rows), INC_COLUMNS).sort_values(["run_id", "role", "k"], kind="mergesort") \
        .reset_index(drop=True)
    if old is not None and len(out):
        _log_recheck_changes(old, out, log)
    return out


def _log_recheck_changes(old: pd.DataFrame, new: pd.DataFrame, log=print) -> None:
    """After --recheck: the incumbent files whose instance file or verdict changed since the cached check."""
    keys = ["run_id", "role", "npz_sha256"]
    o = old[old["checked"].astype(bool)].groupby(keys)
    n = new[new["checked"].astype(bool)].groupby(keys)
    common = sorted(set(o.groups) & set(n.groups))
    inst, verdict = [], []
    for k in common:
        go, gn = o.get_group(k), n.get_group(k)
        a, b = set(go["instance_sha256"]) - {""}, set(gn["instance_sha256"]) - {""}
        if a and b and a != b:
            inst.append(k)
        if go.sort_values("k")["ok"].astype(bool).tolist() != gn.sort_values("k")["ok"].astype(bool).tolist():
            verdict.append(k)
    if inst:
        log(f"--recheck: instance file changed since the cached check for {len(inst)} incumbent file(s), e.g. "
            f"{inst[0][0]}/{inst[0][1]}")
    if verdict:
        log(f"--recheck: verdicts changed for {len(verdict)} incumbent file(s), e.g. {verdict[0][0]}/{verdict[0][1]}")


def _recover_parts(run: Run, log=print) -> None:
    """A process that died keeps `<role>.incumbents.part` (+ names.json) instead of the .npz; turn it into the .npz
    (`itrea.solvers.common.recover_incumbents`) so its incumbents are checked and the crashed run is kept, truncated
    at the crash (D-002 §6)."""
    for part in sorted(run.path.glob("*.incumbents.part")):
        role = part.name[: -len(".incumbents.part")]
        if (run.path / f"{role}.incumbents.npz").exists():
            continue
        try:
            from itrea.solvers.common import recover_incumbents
            if recover_incumbents(run.path, role) is not None:
                log(f"{run.run_id}: recovered {role}.incumbents.npz from the crash-safe .part file")
        except Exception as e:  # noqa: BLE001 — reported as incumbents_missing by the integrity columns
            log(f"{run.run_id}: could not recover {role} incumbents: {e!r}")


def _worker_env():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


# ------------------------------------------------------------------------------------------- run-level parts
def _fl(v) -> float:
    f = M._fl(v)
    return math.nan if f is None else f


def role_steps(inc: pd.DataFrame, run_id: str, sense: int) -> dict[str, list]:
    """Best-so-far CHECKED steps per role, with the recomputed objective."""
    out = {}
    g = inc[(inc["run_id"] == run_id) & inc["ok"]]
    for role, gr in g.groupby("role", sort=True):
        out[role] = M.best_steps(zip(gr["t"].to_numpy(), gr["obj_recomputed"].to_numpy()), sense)
    return out


def bound_steps(run: Run, sense: int) -> list:
    pts = []
    for role in ROLES:
        if role.startswith("solver") or role == "cuopt":
            pts += [(e.get("t"), e.get("bound")) for e in run.events.get(role, []) if e.get("kind") == "bound"]
    return M.dual_steps(pts, sense)


def post_accounting(run: Run, T: float) -> dict:
    """posted / tried-accepted / tried-rejected / superseded-untried / pending-at-T (see module docstring)."""
    posts = sorted(((int(e["seq"]), float(e["t"]), _fl(e.get("obj"))) for e in run.events.get(EA_ROLE, [])
                    if e.get("kind") == "post" and e.get("seq") is not None and e.get("t") is not None),
                   key=lambda p: p[0])
    verdicts: dict[int, dict] = {}

    def add(v: dict) -> None:
        """One verdict per seq; a post of several rows (frontier posting) is accepted / stored if any row is."""
        seq = int(v["seq"])
        acc, sto = bool(v.get("accepted", v.get("stored", False))), bool(v.get("stored", False))
        if seq in verdicts:
            w = verdicts[seq]
            w["accepted"], w["stored"] = w["accepted"] or acc, w["stored"] or sto
            w["t"], w["rows"] = min(_fl(w["t"]), _fl(v.get("t"))), w["rows"] + 1
        else:
            verdicts[seq] = dict(v, accepted=acc, stored=sto, rows=1)

    for role in ROLES:
        if not role.startswith("solver"):
            continue
        for e in run.events.get(role, []):
            if e.get("kind") == "verdict" and e.get("seq") is not None:
                add(e)
    src = "events"
    vf = run.path / "verdicts.jsonl"
    if not verdicts and vf.exists():
        t0 = _fl(run.run_json.get("t0", run.summary(SOLVER_ROLE).get("t0")))
        src = "verdicts.jsonl"
        for line in vf.read_text().splitlines():
            try:
                v = json.loads(line)
            except ValueError:
                continue
            if v.get("seq") is None:
                continue
            note = v.get("note")
            if isinstance(note, str) and note.startswith("{"):
                try:
                    v = dict(json.loads(note), **v)
                except ValueError:
                    pass
            add(dict(v, t=(_fl(v.get("t")) - t0) if not math.isnan(t0) else math.nan))
    out = dict(posts=0, posts_accepted=0, posts_rejected=0, posts_superseded=0, posts_pending=0, posts_stored=0,
               posts_became_incumbent=0, posts_after_T=0, verdicts_unmatched=0, verdict_source=src if posts else "")
    in_T = [p for p in posts if p[1] <= T]
    out["posts_after_T"] = len(posts) - len(in_T)
    seqs_in_T = [p[0] for p in in_T]
    for seq, t, obj in in_T:
        out["posts"] += 1
        v = verdicts.get(seq)
        if v is not None and _fl(v.get("t")) <= T:
            if v["accepted"]:
                out["posts_accepted"] += 1
                out["posts_stored"] += int(v["stored"])
                pa = _fl(v.get("primal_after"))
                if not math.isnan(pa) and not math.isnan(obj) and abs(pa - obj) <= 1e-9 * max(1.0, abs(obj)):
                    out["posts_became_incumbent"] += 1
            else:
                out["posts_rejected"] += 1
        elif any(s > seq for s in seqs_in_T):
            out["posts_superseded"] += 1
        else:
            out["posts_pending"] += 1
    all_seqs = {p[0] for p in posts}
    out["verdicts_unmatched"] = sum(1 for s in verdicts if s not in all_seqs)
    return out


def reads_cuopt(run: Run) -> bool:
    """The run's EA read cuOpt's channels (A9: `exchange.enabled` and `exchange.from_cuopt`), decided from what the EA
    recorded: its summary's `exchange_from_cuopt` (written only then), else the configuration in its summary. Only an
    EA without a recorded configuration (crashed before writing one) falls back to the runner's `config.yaml`, and last
    to the base arm or `inject_cuopt` events, unless the run record's overrides turn the exchange off. So an A9 job
    whose ablation sets `exchange.enabled: false` (cuOpt then gets no --send) is not counted as reading cuOpt."""
    s = run.summary(EA_ROLE)
    if s.get("exchange_from_cuopt"):
        return True
    cfg = s.get("config") if isinstance(s.get("config"), dict) else None
    if cfg is None:
        try:
            cfg = yaml.safe_load((Path(run.path) / "config.yaml").read_text())
        except (OSError, TypeError, yaml.YAMLError):
            cfg = None
    if isinstance(cfg, dict):
        ex = cfg.get("exchange") if isinstance(cfg.get("exchange"), dict) else {}
        return bool(ex.get("enabled", True) and ex.get("from_cuopt", False))
    rc = run.run_json.get("config") if isinstance(run.run_json.get("config"), dict) else {}
    if (rc.get("overrides") or {}).get("exchange.enabled") is False:
        return False
    base = str(run.run_json.get("base_arm") or run.arm).split("~")[0].split("@")[0]
    return base in CUOPT_X_ARMS or any(e.get("kind") == "inject_cuopt" for e in run.events.get(EA_ROLE, []))


def _verdict_rows(run: Run) -> list[dict]:
    """The solvers' verdicts, one dict per received row (seq, t, obj, accepted and the note's fields: row, dup,
    primal_after, cuopt …): the `verdict` events, or — only when the solvers wrote none — the runner's
    `verdicts.jsonl` (monotonic t converted with t0), as in post_accounting."""
    vs = [e for r in ROLES if r.startswith("solver") for e in run.events.get(r, []) if e.get("kind") == "verdict"]
    vf = run.path / "verdicts.jsonl"
    if vs or not vf.exists():
        return vs
    t0 = _fl(run.run_json.get("t0", run.summary(SOLVER_ROLE).get("t0")))
    for line in vf.read_text().splitlines():
        try:
            v = json.loads(line)
            note = json.loads(v["note"]) if str(v.get("note", "")).startswith("{") else {}
        except (ValueError, KeyError, TypeError):
            continue
        if not isinstance(v, dict) or v.get("seq") is None:
            continue
        vs.append(dict(note, **{k: x for k, x in v.items() if k not in ("note", "t")},
                       t=(_fl(v.get("t")) - t0) if not math.isnan(t0) else math.nan))
    return vs


def cuopt_attribution(run: Run, inc: pd.DataFrame, T: float) -> dict:
    """A9: what cuOpt contributed through the EA, up to T (module docstring). {} for a run that does not read cuOpt's
    channels (the columns stay missing).

    Units: `cuopt_posts`, `cuopt_posts_accepted` and `cuopt_posts_became_incumbent` count EA POSTS (seq) under
    post_accounting's rules (a post counts once, accepted if any of its cuOpt rows is — a `dup` verdict included, as
    for `posts_accepted` — and became the incumbent if such a row's `primal_after` equals that row's posted objective
    within 1e-9 relative), so they are the cuOpt share of `posts`, `posts_accepted`, `posts_became_incumbent` (the EA's
    own share = the difference). For a single-row post (every incumbent post) this is post_accounting's match; for a
    several-row frontier post it matches on the cuOpt row, not on the post's best objective. `ea_inc_from_cuopt` and
    `solver_inc_from_cuopt` count checked incumbent RECORDS with source `cuopt`, the unit of `solver_inc_from_ea`: one
    SCIP `trySol` can record two (BESTSOLFOUND fires twice, the second on SCIP's slightly adjusted copy)."""
    if not reads_cuopt(run):
        return {}
    ev = [e for e in run.events.get(EA_ROLE, []) if _fl(e.get("t")) <= T]
    inj = [e for e in ev if e.get("kind") == "inject_cuopt"]
    posts = {int(e["seq"]) for e in ev if e.get("kind") == "post" and e.get("cuopt_rows") and e.get("seq") is not None}
    acc, became = set(), set()
    for v in _verdict_rows(run):
        if not (v.get("cuopt") and v.get("accepted") and v.get("seq") is not None and _fl(v.get("t")) <= T):
            continue
        seq = int(v["seq"])
        if seq not in posts:
            continue
        acc.add(seq)
        pa, obj = _fl(v.get("primal_after")), _fl(v.get("obj"))
        if not math.isnan(pa) and not math.isnan(obj) and abs(pa - obj) <= 1e-9 * max(1.0, abs(obj)):
            became.add(seq)
    out = dict(cuopt_inject_msgs=len(inj), cuopt_injected=int(sum(int(e.get("k") or 0) for e in inj)),
               cuopt_posts=len(posts), cuopt_posts_accepted=len(acc), cuopt_posts_became_incumbent=len(became),
               ea_inc_from_cuopt=0, solver_inc_from_cuopt=0)
    if len(inc):
        g = inc[(inc["run_id"] == run.run_id) & inc["ok"] & (inc["source"] == "cuopt") & (inc["t"] <= T)]
        out["ea_inc_from_cuopt"] = int((g["role"] == EA_ROLE).sum())
        out["solver_inc_from_cuopt"] = int((g["role"] == SOLVER_ROLE).sum())
    return out


LPD_INT_COLS = ("lpd_offers", "lpd_rows_offered", "lpd_rows_F", "lpd_rows_I", "lpd_busy", "lpd_skip_pending",
                "lpd_replies", "lpd_solved", "lpd_solved_F", "lpd_solved_I", "lpd_ok", "lpd_ok_F", "lpd_ok_I",
                "lpd_infeasible", "lpd_time_limit", "lpd_other", "lpd_range_fail", "lpd_cached", "lpd_not_solved",
                "lpd_after_deadline", "lpd_cached_F", "lpd_cached_I", "lpd_screened_I", "lpd_srcs_F", "lpd_srcs_I",
                "lpd_no_gain", "lpd_kept", "lpd_kept_F", "lpd_kept_I", "lpd_batches", "lpd_dropped", "lpd_entered",
                "lpd_entered_F", "lpd_entered_I", "lpd_incumbents", "lpd_incumbents_F", "lpd_incumbents_I", "lpd_posts",
                "lpd_posts_F", "lpd_posts_I", "lpd_lost", "lpd_failed")
# (column, key in the summary's lp_decode block) and (column, key in its worker statistics): float columns, but
# LPD_BLOCK_INT_COLS and the worker counts are int
LPD_BLOCK_COLS = (("lpd_host_s", "host_s"), ("lpd_host_s_max", "host_s_max"), ("lpd_sync_s", "sync_s"),
                  ("lpd_sync_s_max", "sync_s_max"), ("lpd_ready_t", "ready_t"),
                  ("lpd_ready_t_worker", "ready_t_worker"), ("lpd_wait_s_total", "wait_s_total"),
                  ("lpd_wait_s_max", "wait_s_max"), ("lpd_batch_rows_max", "batch_rows_max"),
                  ("lpd_gain_F_entered_median", "gain_F_entered_median"), ("lpd_gain_F_entered_n", "gain_F_entered_n"),
                  ("lpd_gain_I_entered_median", "gain_I_entered_median"), ("lpd_gain_I_entered_n", "gain_I_entered_n"))
LPD_BLOCK_INT_COLS = ("lpd_batch_rows_max", "lpd_gain_F_entered_n", "lpd_gain_I_entered_n")
LPD_IPC_COLS = (("lpd_offer_s_max", "offer_s_max"), ("lpd_offer_s_total", "offer_s_total"),
                ("lpd_offer_bytes_max", "offer_bytes_max"), ("lpd_read_s_max", "read_s_max"),
                ("lpd_read_s_total", "read_s_total"), ("lpd_read_bytes_max", "read_bytes_max"))
LPD_WORKER_COLS = (("lpd_priming_s", "priming_s"), ("lpd_priming_first_s", "priming_first_s"),
                   ("lpd_priming_solves", "priming_solves"), ("lpd_priming_time_limit_hits", "priming_time_limit_hits"),
                   ("lpd_worker_solves", "solves"), ("lpd_warm_solves", "warm_solves"),
                   ("lpd_worker_time_limit_hits", "time_limit_hits"),
                   ("lpd_warm_time_limit_hits", "warm_time_limit_hits"),
                   ("lpd_time_limit_hits_starved", "time_limit_hits_starved"), ("lpd_screen_pass", "screen_pass"),
                   ("lpd_screen_fail", "screen_fail"), ("lpd_worker_solve_s", "solve_s_total"),
                   ("lpd_worker_solve_cpu_s", "solve_cpu_s_total"), ("lpd_worker_maxrss_mb", "ru_maxrss_mb"))


def lp_decode_cols(run: Run) -> dict:
    """The LP decode's record (`repair.continuous: lp_decode`, D-017) from the EA's summary: its state, the counters
    in LPD_INT_COLS (per kind of source where D-017's M3 / M4 need it), the block's EA-side times (LPD_BLOCK_COLS: host
    and device-sync seconds for M1, the worker's own ready time for M0, the replies' wait in the pipe) and the median
    relative gain of the decoded copies that entered the population, per kind (M3), the pipe's
    send / read times and bytes (LPD_IPC_COLS) and the worker's statistics (LPD_WORKER_COLS, with the priming / warm
    split for M2), plus `lpd_priming_first_status`. Runs without the decode leave every column missing (-1 / NaN /
    ""); an instance on which it was inert has `lpd_state` "inert" and nothing else."""
    s = run.summary(EA_ROLE)
    blk = s.get("lp_decode")
    if not isinstance(blk, dict):
        st = (s.get("startup") or {}).get("lp_decode") if isinstance(s.get("startup"), dict) else None
        return {"lpd_state": str(st.get("state") or "")} if isinstance(st, dict) else {}
    w = blk.get("worker") if isinstance(blk.get("worker"), dict) else {}
    ipc = blk.get("ipc") if isinstance(blk.get("ipc"), dict) else {}
    out: dict = {"lpd_state": str(blk.get("state") or "")}
    out.update({k: _counter(s, k) for k in LPD_INT_COLS})
    out.update({col: _fl(blk.get(key)) for col, key in LPD_BLOCK_COLS})
    out.update({col: _fl(ipc.get(key)) for col, key in LPD_IPC_COLS})
    out.update({col: _fl(w.get(key)) for col, key in LPD_WORKER_COLS})
    out["lpd_priming_first_status"] = str(w.get("priming_first_status") or "")
    return out


def backoff_stats(run: Run) -> dict:
    """The GPU back-off's columns (module docstring) from the EA summary's `backoff` block; {} without it."""
    b = run.summary(EA_ROLE).get("backoff")
    if not isinstance(b, dict):
        return {}

    def sub(k):
        v = b.get(k)
        return v if isinstance(v, dict) else {}
    py, spl, own, fl, fe = sub("phase_y"), sub("seconds_per_level"), sub("own_feasible"), sub("first_lead"), \
        sub("first_escalation")
    forced = b.get("mode") == "forced"
    out = dict(bo_mode=str(b.get("mode") or ""), bo_busy_share=_fl(b.get("busy_share")),
               bo_paused_s=_fl(b.get("paused_s")), bo_phase_y_s=_fl(py.get("length_s")),
               bo_phase_y_exit=str(py.get("reason") or ""), bo_n_lead=_int(b.get("n_lead"), -1),
               bo_n_assist=_int(b.get("n_assist"), -1), bo_own_feasible_s=_fl(own.get("t")),
               bo_first_lead_s=_fl(fl.get("t")), bo_first_lead_busy_s=_fl(fl.get("busy_s")),
               bo_first_escalation_s=_fl(fe.get("busy_s")), bo_escalations=_int(b.get("escalations"), -1),
               bo_wakes=_int(b.get("wakes"), -1), bo_ls_steps=_fl(b.get("ls_steps")))
    for k in range(4):
        out[f"bo_level{k}_s"] = math.nan if forced else _fl(spl.get(str(k), 0.0))
    return out


def _exit_codes(rj: dict) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    procs = rj.get("processes")
    if isinstance(procs, dict):
        procs = [dict(v, role=k) for k, v in procs.items() if isinstance(v, dict)]
    for p in procs or []:
        if isinstance(p, dict) and p.get("role"):
            c = p.get("exit_code", p.get("returncode"))
            out[str(p["role"])] = None if c is None else int(c)
    for k, c in (rj.get("exit_codes") or {}).items():
        out.setdefault(str(k), None if c is None else int(c))
    return out


def _stopped_by_runner(rj: dict) -> set[str]:
    """Roles the runner killed after a partner ended the race with a proven outcome (`processes[*].stopped_by_runner`;
    cuOpt has no interrupt API, so it is SIGKILLed after `kill_after_s`): not crashes of their own."""
    procs = rj.get("processes")
    if isinstance(procs, dict):
        procs = [dict(v, role=k) for k, v in procs.items() if isinstance(v, dict)]
    return {str(p["role"]) for p in procs or [] if isinstance(p, dict) and p.get("role") and p.get("stopped_by_runner")}


def crash_flags(run: Run) -> dict:
    codes = _exit_codes(run.run_json)
    stopped = _stopped_by_runner(run.run_json)
    crashed = []
    for role in sorted(set(run.summaries) | set(codes)):
        if role in stopped:                      # the runner's kill, not a crash (pair.run_pair)
            continue
        c = codes.get(role)
        s = run.summaries.get(role, {})
        if c is not None:
            bad = c != 0
        else:
            bad = (not s) or str(s.get("exit_reason", "")).startswith("exception")
        if bad:
            crashed.append(role)
    crash_t = max((float(e["t"]) for r in crashed for e in run.events.get(r, []) if e.get("t") is not None),
                  default=math.nan)
    return dict(crashed=bool(crashed), crashed_roles=",".join(crashed), crash_t=crash_t,
                exit_codes=json.dumps({k: codes[k] for k in sorted(codes)}, sort_keys=True))


def gpu_flags(run: Run) -> dict:
    gh = run.run_json.get("gpu_health") or {}
    gpu_arm = run.arm in GPU_ARMS or bool(gh.get("gpu_run"))
    verdict = str(gh.get("verdict") or "")
    med, foreign = _fl(gh.get("median_sm_mhz")), _fl(gh.get("foreign_share"))
    if verdict:
        void = verdict == "void" and gpu_arm
    else:
        void = gpu_arm and ((not math.isnan(med) and med < CLOCK_FLOOR_MHZ) or
                            (not math.isnan(foreign) and foreign > FOREIGN_MAX_SHARE))
        verdict = "computed" if gh else ("missing" if gpu_arm else "")
    return dict(gpu_arm=gpu_arm, gpu_health=verdict, gpu_median_sm_mhz=med, gpu_foreign_share=foreign,
                gpu_void=bool(void))


def _startup(s: dict) -> float:
    v = s.get("startup_s", s.get("startup"))
    if isinstance(v, dict):
        vals = [_fl(x) for x in v.values()]
        vals = [x for x in vals if not math.isnan(x)]
        return float(sum(vals)) if vals else math.nan
    return _fl(v)


def _counter(s: dict, key: str) -> float:
    v = s.get(key)
    if v is None:
        v = (s.get("counters") or {}).get(key)
    return _fl(v)


def frontier_stats(run: Run, T: float) -> dict:
    fr = [e for e in run.events.get(EA_ROLE, []) if e.get("kind") == "frontier" and _fl(e.get("t")) <= T]
    out = dict(frontier_size=math.nan, frontier_hv=math.nan, frontier_mean_pairwise=math.nan,
               frontier_max_dist=math.nan, frontier_mean_dist_to_inc=math.nan)
    if fr:
        e = fr[-1]
        out.update(frontier_size=_fl(e.get("size")), frontier_hv=_fl(e.get("hv")),
                   frontier_mean_pairwise=_fl(e.get("mean_pairwise")), frontier_max_dist=_fl(e.get("max_dist")),
                   frontier_mean_dist_to_inc=_fl(e.get("mean_dist_to_inc")))
    elif (run.path / "ea.frontier.npz").exists():
        try:
            with np.load(run.path / "ea.frontier.npz") as z:
                k = next((z[n] for n in ("keys", "key") if n in z.files), None)
                out["frontier_size"] = float(len(k)) if k is not None else math.nan
        except (OSError, ValueError):
            pass
    return out


def callback_stats(run: Run, T: float) -> dict:
    """First callback, first SOLVING-stage callback and the largest gap (SCIP `callback` events and the summary's
    `callbacks`). `no_exchange_window` (D-002 §5 subgroup): the first SOLVING-stage callback came after T/2, or the
    plugin reported its callbacks and never reached SOLVING."""
    cb = [e for e in run.events.get(SOLVER_ROLE, []) if e.get("kind") == "callback"]
    sc = run.summary(SOLVER_ROLE).get("callbacks")
    if isinstance(sc, dict):
        cb.append(sc)
    reported = any("t_first_solving_callback" in e for e in cb)
    first = solving = gap = math.nan
    for e in cb:
        what = e.get("what")
        if what in ("first", "first_solving") and e.get("t") is not None:
            e = {("first" if what == "first" else "first_solving"): e["t"]}
        for k in ("first", "t_first", "first_callback", "t_first_callback"):
            if e.get(k) is not None:
                first = _fl(e[k]) if math.isnan(first) else min(first, _fl(e[k]))
        for k in ("first_solving", "t_first_solving", "first_solving_callback", "t_first_solving_callback"):
            if e.get(k) is not None:
                solving = _fl(e[k]) if math.isnan(solving) else min(solving, _fl(e[k]))
        if str(e.get("stage", "")).upper() == "SOLVING" and e.get("t") is not None:
            solving = _fl(e["t"]) if math.isnan(solving) else min(solving, _fl(e["t"]))
        for k in ("max_gap", "max_callback_gap", "max_gap_s"):
            if e.get(k) is not None:
                gap = _fl(e[k]) if math.isnan(gap) else max(gap, _fl(e[k]))
    late = (not math.isnan(solving) and solving > T / 2) or (reported and math.isnan(solving))
    return dict(cb_first_s=first, cb_first_solving_s=solving, cb_max_gap_s=gap, no_exchange_window=bool(late))


# ------------------------------------------------------------------------------------------------ the build
def instance_senses(inc: pd.DataFrame, runs: list[Run]) -> tuple[dict[str, int], set[str]]:
    """Objective sense per run_id's instance: from the checker's model parse, else the summaries, else +1."""
    rid2name = {r.run_id: r.name for r in runs}
    sense: dict[str, int] = {}
    conflict: set[str] = set()
    if len(inc):
        chk = inc[inc["checked"] & (inc["sense"] != 0)]
        for name, g in chk.assign(name=chk["run_id"].map(rid2name)).groupby("name"):
            vals = sorted(set(g["sense"].astype(int)))
            if len(vals) > 1:
                conflict.add(name)
            sense[name] = int(g["sense"].astype(int).mode().iloc[0])
    for r in runs:
        prob = r.run_json.get("problem") if isinstance(r.run_json.get("problem"), dict) else {}
        if r.name not in sense and prob.get("sense") in (1, -1):
            sense[r.name] = int(prob["sense"])
        if r.name not in sense:
            for s in r.summaries.values():
                if s.get("sense") in (1, -1):
                    sense[r.name] = int(s["sense"])
                    break
    return sense, conflict


def _best(vals, s: int) -> float | None:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return min(vals) if s == 1 else max(vals)


def reference_datasets(runs: list[Run]) -> dict[str, str]:
    """The dataset name each instance's reference is resolved under: "miplib" when the instance is a MIPLIB 2017
    one, else "" (for every other dataset the name plays no part in `references.resolve`).

    Decided from the instance PATH (data/miplib/inst/<name>.mps.gz) or a set named miplib*, never from the job's set
    alone: the job sets are screen1/screen2/e1/e4/e2/…, so `resolve(name, r.set, …)` never consulted the .solu file and
    every MIPLIB run was scored against its own best SCREEN run (review 2026-10-06, F1). One verdict per instance name,
    independent of run order; a name that is MIPLIB in one run and not in another is a name collision and raises."""
    mip: dict[str, bool] = {}
    paths: dict[str, set] = {}
    for r in runs:
        m = R.is_miplib(r.set, r.instance)
        paths.setdefault(r.name, set()).add(r.instance or r.set)
        if mip.setdefault(r.name, m) != m:
            raise ValueError(f"instance name {r.name!r} is a MIPLIB instance in some runs and not in others: "
                             f"{sorted(map(str, paths[r.name]))[:4]}")
    return {n: ("miplib" if m else "") for n, m in mip.items()}


def build(raw_root: str | os.PathLike = TRACES / "raw", out_dir: str | os.PathLike = DATASETS, recheck: bool = False,
          workers: int = 1, write: bool = True, log=print, solu_path: str | os.PathLike = SOLU_PATH,
          verify_instances: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_root, out_dir = Path(raw_root), Path(out_dir)
    runs = sorted((load_run(p, raw_root) for p in discover(raw_root)), key=lambda r: r.run_id)
    ids = [r.run_id for r in runs]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate run ids (a rerun must get its own directory and run_id): {dup[:5]}")
    inc_path = out_dir / "incumbents_checked.parquet"
    cached = pd.read_parquet(inc_path) if inc_path.exists() else None    # under --recheck: only to log changes
    inc = check_runs(runs, cached, recheck, workers, log, verify_instances=verify_instances)
    sense, conflict = instance_senses(inc, runs)
    rid2run = {r.run_id: r for r in runs}

    # references
    solu_p = resolve(solu_path)
    solu = R.read_solu(solu_p) if solu_p.exists() else {}
    ref_p = out_dir / "reference.csv"
    frozen = R.read_reference_csv(ref_p) if ref_p.exists() else None
    okinc = inc[inc["ok"]].assign(name=lambda d: d["run_id"].map(lambda i: rid2run[i].name),
                                  arm=lambda d: d["run_id"].map(lambda i: rid2run[i].arm))
    screening_best, screen_only, study_best = {}, {}, {}
    for name, g in okinc.groupby("name"):
        s = sense.get(name, 1)
        sb = _best(g.loc[g["arm"].isin(REF_ARMS), "obj_recomputed"].tolist(), s)
        if sb is not None:
            screening_best[name] = sb
        so = _best(g.loc[g["arm"] == SCREEN_ARM, "obj_recomputed"].tolist(), s)
        if so is not None:
            screen_only[name] = so
        study_best[name] = _best(g["obj_recomputed"].tolist(), s)
    ds = reference_datasets(runs)
    refs = {n: R.resolve(n, d, frozen, solu, screening_best, study_best) for n, d in ds.items()}
    # the screening reference: what the pool freeze (D-002 step 3) ranks on — .solu, else the best of the SCREEN runs.
    # It never changes once the screening is done, so the frozen pool stays re-derivable after reference.csv exists.
    refs_screen = {n: R.resolve(n, d, None, solu, screen_only) for n, d in ds.items()}

    # per-incumbent beats-reference flag
    def _beats(row):
        rr = refs.get(rid2run[row["run_id"]].name)
        if not row["ok"] or rr is None or rr.value is None:
            return False
        s_ = sense.get(rid2run[row["run_id"]].name, 1)
        return M.BEATS_REF in M.primal_gap(row["obj_recomputed"], rr.value, s_)[1]
    inc["beats_ref"] = inc.apply(_beats, axis=1).astype(bool) if len(inc) else pd.Series(dtype=bool)

    rows, steps_of = [], {}
    for run in runs:
        s = sense.get(run.name, 1)
        ref = refs[run.name]
        T = run.budget
        steps = role_steps(inc, run.run_id, s)
        steps_of[run.run_id] = steps
        port = M.merge_steps(*steps.values(), sense=s)
        dual = bound_steps(run, s)
        row = _identity(run)
        row.update(_ref_cols(ref, s, study_best.get(run.name), run.name in conflict))
        row.update(_metric_cols(port, steps, ref, s, T, dual, study_best.get(run.name), set(run.summaries)))
        rs_ = refs_screen[run.name]
        ps = M.primal_integral(port, rs_.value, s, T)
        share = M.sign_crossing_share(port, rs_.value, s, T)
        row.update(ref_screen=math.nan if rs_.value is None else rs_.value, pt_screen=ps.value,
                   ref_zero_screen=rs_.ref_zero, sign_crossing_screen=bool(share >= M.SIGN_CROSSING_MIN_SHARE),
                   sign_crossing_screen_any=M.SIGN_CROSSING in ps.flags, sign_crossing_screen_share=share)
        row.update(_integrity_cols(run, inc))
        row.update(post_accounting(run, T))
        row["solver_inc_from_ea"] = int(((inc["run_id"] == run.run_id) & (inc["role"] == SOLVER_ROLE) & inc["ok"] &
                                         (inc["source"] == "ea") & (inc["t"] <= T)).sum()) if len(inc) else 0
        row.update(cuopt_attribution(run, inc, T))
        row.update(lp_decode_cols(run))
        row.update(backoff_stats(run))
        row.update(crash_flags(run))
        row.update(gpu_flags(run))
        row.update(_resource_cols(run))
        row.update(frontier_stats(run, T))
        row.update(callback_stats(run, T))
        rows.append(row)

    # the uncoupled portfolio (D-002 v3 H2): pointwise best of A0 and A5 of the same block. It inherits every
    # per-run validity attribute from its parents (review 2026-10-06, F2): the EA backend and its mismatch flag, the
    # code revision (both parents' kept in `parent_git_revs`), status, split and the clock-rule flag — otherwise
    # prereg would drop a CPU-fallback A5 run but keep the portfolio built from its trajectory.
    row_of = {r["run_id"]: r for r in rows}
    by_block: dict[tuple, dict[str, Run]] = {}
    for run in runs:
        if run.arm in UNCOUPLED[:2]:
            by_block.setdefault((run.name, run.seed, run.budget), {})[run.arm] = run
    for (name, seed, budget), arms in sorted(by_block.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        if set(arms) != set(UNCOUPLED[:2]):
            continue
        a0, a5 = arms[UNCOUPLED[0]], arms[UNCOUPLED[1]]
        s, ref = sense.get(name, 1), refs[name]
        st = {"solver": steps_of[a0.run_id].get(SOLVER_ROLE, []), "ea": steps_of[a5.run_id].get(EA_ROLE, [])}
        port = M.merge_steps(*st.values(), sense=s)
        row = dict(run_id=f"{name}__{UNCOUPLED[2]}__s{seed}__{budget:g}s__computed", set=a0.set, instance=name,
                   instance_path=a0.instance or "", arm=UNCOUPLED[2], seed=seed, budget=budget,
                   config_hash=a5.config_hash, synthetic=True, run_dir="",
                   attempt=max(_int(a0.run_json.get("attempt"), 0), _int(a5.run_json.get("attempt"), 0)))
        row.update(_ref_cols(ref, s, study_best.get(name), name in conflict))
        row.update(_metric_cols(port, st, ref, s, budget, bound_steps(a0, s), study_best.get(name),
                                {SOLVER_ROLE, EA_ROLE}))
        c0, c5, g5 = crash_flags(a0), crash_flags(a5), gpu_flags(a5)
        row.update(crashed=c0["crashed"] or c5["crashed"],
                   crashed_roles=",".join(x for x in (c0["crashed_roles"] and "A0:" + c0["crashed_roles"],
                                                      c5["crashed_roles"] and "A5:" + c5["crashed_roles"]) if x),
                   gpu_arm=True, gpu_health=g5["gpu_health"], gpu_void=g5["gpu_void"] or gpu_flags(a0)["gpu_void"])
        i0, i5, r5 = _identity(a0), _identity(a5), _resource_cols(a5)
        p0, p5 = row_of[a0.run_id], row_of[a5.run_id]
        row.update(backend=r5["backend"], backend_mismatch=r5["backend_mismatch"], git_rev=i5["git_rev"],
                   parent_git_revs=json.dumps({UNCOUPLED[0]: i0["git_rev"], UNCOUPLED[1]: i5["git_rev"]},
                                              sort_keys=True),
                   base_hash=i5["base_hash"], prereg_tag=i5["prereg_tag"],
                   status=_parent_status(i5["status"], i0["status"]),
                   split=i5["split"] if i5["split"] == i0["split"] else f"{i0['split']}|{i5['split']}",
                   n_inc_bad_t=p0["n_inc_bad_t"] + p5["n_inc_bad_t"], pre_t0=bool(p0["pre_t0"] or p5["pre_t0"]))
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        void = df["gpu_void"].fillna(False).astype(bool) | (df.get("status", pd.Series("", index=df.index))
                                                             .fillna("") == "void")
        df["block_void"] = void.groupby([df["instance"], df["seed"], df["budget"]]).transform("any").astype(bool)
    df = _normalise(df, RUN_COLUMNS).sort_values("run_id", kind="mergesort").reset_index(drop=True)
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_parquet(df, out_dir / "runs.parquet")
        _write_parquet(inc, inc_path)
    return df, inc


def _parent_status(*statuses: str) -> str:
    """Status of a computed row: the first parent status that is set and not "completed", else "completed" if a
    parent completed, else ""."""
    bad = [s for s in statuses if s and s != "completed"]
    if bad:
        return bad[0]
    return "completed" if "completed" in statuses else ""


def _int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _identity(run: Run) -> dict:
    """Identity columns. `emphasis` and `solver_settings` (pair.solver_settings, the source of a non-EA arm's
    config_hash) are exported so prereg can check the non-EA arms against their frozen settings (review F4)."""
    rj = run.run_json
    rc = rj.get("config") if isinstance(rj.get("config"), dict) else {}
    git = ((rj.get("env") or {}).get("git") or {}) if isinstance(rj.get("env"), dict) else {}
    ss = rc.get("solver_settings") if isinstance(rc.get("solver_settings"), dict) else None
    job = rj.get("job") if isinstance(rj.get("job"), dict) else {}
    emph = (ss or {}).get("emphasis") or job.get("emphasis") or ""
    return dict(run_id=run.run_id, set=run.set, instance=run.name, instance_path=run.instance or "", arm=run.arm,
                base_arm=str(rj.get("base_arm") or run.arm), seed=run.seed, budget=run.budget,
                config_hash=run.config_hash, base_hash=str(rc.get("base_hash") or ""),
                config_overrides=json.dumps(rc.get("overrides") or {}, sort_keys=True, default=str),
                emphasis=str(emph), solver_settings=json.dumps(ss, sort_keys=True, default=str) if ss else "",
                synthetic=False, status=str(rj.get("status") or ""), void_cause=str(rj.get("void_cause") or ""),
                split=str(rj.get("split") or ""), stratum=str(rj.get("stratum") or ""),
                attempt=_int(rj.get("attempt"), 0), block=str(rj.get("block") or ""),
                window_id=str(rj.get("window_id") or rj.get("window") or ""),
                williams_pos=_int(rj.get("williams_pos", rj.get("position")), -1),
                git_rev=str(rj.get("git_rev") or rj.get("git_revision") or git.get("revision") or ""),
                prereg_tag=str(rj.get("prereg_tag") or git.get("prereg_tag") or ""), run_dir=_rel(run.path))


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(p)


def _ref_cols(ref: R.Reference, s: int, bos: float | None, conflict: bool) -> dict:
    return dict(sense=s, ref=math.nan if ref.value is None else ref.value, ref_kind=ref.kind, ref_frozen=ref.frozen,
                ref_zero_instance=ref.ref_zero, ref_bos=math.nan if bos is None else bos, sense_conflict=conflict)


def _metric_cols(port, steps: dict, ref: R.Reference, s: int, T: float, dual, bos: float | None,
                 roles: set[str]) -> dict:
    """Portfolio metrics, P(300) of a longer run, the best-of-study sensitivity, and the per-role trajectories of
    the roles the run has (`roles`; a role the run lacks stays NaN, not "gap 1")."""
    out = M.summarize(port, ref.value, s, T, dual)
    out["pt300"] = M.primal_integral(port, ref.value, s, 300.0).value if T >= 300 else math.nan
    out["pt_bos"] = M.primal_integral(port, bos, s, T).value if bos is not None else math.nan
    for role in (SOLVER_ROLE, EA_ROLE, "cuopt"):
        if role in roles:
            sm = M.summarize(steps.get(role, []), ref.value, s, T)
            for k in ROLE_KEYS:
                out[f"{role}_{k}"] = sm[k]
    return out


def _integrity_cols(run: Run, inc: pd.DataFrame) -> dict:
    g = inc[inc["run_id"] == run.run_id] if len(inc) else inc
    n_ev = {r: sum(1 for e in run.events.get(r, []) if e.get("kind") == "incumbent") for r in run.events}
    have = set(g["role"]) if len(g) else set()
    missing = sorted(r for r, n in n_ev.items() if n > 0 and r not in have)
    mism = sorted(r for r in have if n_ev.get(r) is not None and n_ev[r] != int((g["role"] == r).sum()))
    # clock rule (D-002 §4 integrity, review F6): an incumbent stamped before t0 (t < 0) or before the barrier started
    # the trace clock (t NaN) was produced outside the timed window. best_steps clamps the first to t = 0 and drops
    # the second silently, so both are counted here, over every stored incumbent whether or not it passed the check.
    tt = g["t"].astype(float).to_numpy() if len(g) else np.zeros(0)
    bad_t = int(np.sum(~np.isfinite(tt) | (tt < 0)))
    return dict(n_inc_bad_t=bad_t, pre_t0=bad_t > 0,
                n_inc=int(len(g)), n_inc_ok=int(g["ok"].sum()) if len(g) else 0,
                n_inc_failed=int((g["checked"] & ~g["ok"]).sum()) if len(g) else 0,
                n_inc_unchecked=int((~g["checked"]).sum()) if len(g) else 0,
                n_obj_mismatch=int((g["checked"] & g["feasible"] & ~g["obj_ok"]).sum()) if len(g) else 0,
                incumbents_missing=",".join(missing), inc_count_mismatch=",".join(mism))


def _resource_cols(run: Run) -> dict:
    out = {}
    starts = []
    procs = run.run_json.get("processes")
    pstart = {str(p.get("role")): _fl(p.get("startup_s")) for p in procs if isinstance(p, dict)} \
        if isinstance(procs, list) else {}
    for role in ROLES:
        s = run.summary(role)
        v = _startup(s) if role in run.summaries else math.nan
        if math.isnan(v) and isinstance(s.get("timing"), dict):
            v = _fl(s["timing"].get("startup_s"))
        if math.isnan(v):
            v = pstart.get(role, math.nan)
        out[f"{role}_startup_s"] = v
        if not math.isnan(v):
            starts.append(v)
    out["startup_s"] = max(starts) if starts else math.nan
    ss = run.summary(SOLVER_ROLE)
    ex = ss.get("exchange") if isinstance(ss.get("exchange"), dict) else {}
    out["exchange_s"] = _fl(ex.get("seconds")) if ex else _counter(ss, "exchange_s")
    out["exchange_bytes"] = (_fl(ex.get("bytes_out", 0)) + _fl(ex.get("bytes_in", 0))) if ex else \
        _counter(ss, "exchange_bytes")
    out["solver_nodes_per_s"] = _fl(ss.get("nodes_per_s"))
    out["solver_lp_iter_per_s"] = _fl(ss.get("lp_iterations_per_s"))
    out["solver_status"] = str(ss.get("status") or "")
    out["solver_exit_reason"] = str(ss.get("exit_reason") or "")
    out["solved_optimal"] = "optimal" in (out["solver_status"].lower(), out["solver_exit_reason"].lower())
    be = str(run.summary(EA_ROLE).get("backend") or "")
    out["backend"] = be
    out["backend_mismatch"] = bool(be and ((run.arm in EA_GPU_ARMS and be != "cupy") or
                                           (run.arm in EA_CPU_ARMS and be != "numpy")))
    return out


# ---------------------------------------------------------------------------------------------- normalising
def _role_cols() -> dict:
    types = dict(pt="float", t_first="float", t_first_event="bool", t_gap5="float", t_gap5_event="bool",
                 t_gap1="float", t_gap1_event="bool", final_gap="float", best_obj="float", n_steps="int",
                 sign_crossing="bool", beats_ref="bool", pt_mf="float")
    return {f"{r}_{k}": types[k] for r in (SOLVER_ROLE, EA_ROLE, "cuopt") for k in ROLE_KEYS}


RUN_COLUMNS = {
    # identity
    "run_id": "str", "set": "str", "instance": "str", "instance_path": "str", "arm": "str", "base_arm": "str",
    "seed": "int", "budget": "float", "config_hash": "str", "base_hash": "str", "config_overrides": "str",
    "emphasis": "str", "solver_settings": "str", "parent_git_revs": "str", "synthetic": "bool", "status": "str",
    "void_cause": "str", "split": "str", "stratum": "str", "attempt": "int",
    "block": "str", "window_id": "str", "williams_pos": "int", "git_rev": "str", "prereg_tag": "str", "run_dir": "str",
    # reference
    "sense": "int", "ref": "float", "ref_kind": "str", "ref_frozen": "bool", "ref_zero_instance": "bool",
    "ref_bos": "float", "sense_conflict": "bool",
    # portfolio (primary) and sensitivity
    "pt": "float", "pt300": "float", "pt_bos": "float", "ref_screen": "float", "pt_screen": "float",
    "ref_zero_screen": "bool", "sign_crossing_screen": "bool", "sign_crossing_screen_any": "bool",
    "sign_crossing_screen_share": "float", "ref_zero": "bool", "sign_crossing": "bool", "sign_crossing_any": "bool",
    "sign_crossing_share": "float", "pt_mf": "float",
    "beats_ref": "bool", "beats_ref_t": "float", "t_first": "float", "t_first_event": "bool", "t_gap5": "float",
    "t_gap5_event": "bool", "t_gap1": "float", "t_gap1_event": "bool", "final_gap": "float", "best_obj": "float",
    "n_steps": "int", "pdi": "float",
    **_role_cols(),
    # integrity
    "n_inc": "int", "n_inc_ok": "int", "n_inc_failed": "int", "n_inc_unchecked": "int", "n_obj_mismatch": "int",
    "incumbents_missing": "str", "inc_count_mismatch": "str", "n_inc_bad_t": "int", "pre_t0": "bool",
    "posts": "int", "posts_accepted": "int", "posts_rejected": "int", "posts_superseded": "int",
    "posts_pending": "int", "posts_stored": "int", "posts_became_incumbent": "int", "posts_after_T": "int",
    "verdicts_unmatched": "int",
    "verdict_source": "str", "solver_inc_from_ea": "int",
    # A9 (D-015): cuOpt's contribution through the EA, up to T; -1 for runs that do not read cuOpt's channels
    "cuopt_inject_msgs": "int", "cuopt_injected": "int", "ea_inc_from_cuopt": "int", "cuopt_posts": "int",
    "cuopt_posts_accepted": "int", "cuopt_posts_became_incumbent": "int", "solver_inc_from_cuopt": "int",
    # D-017: the LP decode (repair.continuous: lp_decode); missing (-1 / NaN / "") for runs without it
    "lpd_state": "str", **{k: "int" for k in LPD_INT_COLS},
    **{c: ("int" if c in LPD_BLOCK_INT_COLS else "float") for c, _ in LPD_BLOCK_COLS},
    **{c: ("int" if c.endswith("_bytes_max") else "float") for c, _ in LPD_IPC_COLS},
    **{c: ("float" if c.endswith(("_s", "_mb")) else "int") for c, _ in LPD_WORKER_COLS},
    "lpd_priming_first_status": "str",
    # A9's GPU back-off (D-018): missing for runs without the governor
    "bo_mode": "str", "bo_busy_share": "float", "bo_paused_s": "float", "bo_phase_y_s": "float",
    "bo_phase_y_exit": "str", **{f"bo_level{k}_s": "float" for k in range(4)}, "bo_n_lead": "int",
    "bo_n_assist": "int", "bo_own_feasible_s": "float", "bo_first_lead_s": "float", "bo_first_lead_busy_s": "float",
    "bo_first_escalation_s": "float", "bo_escalations": "int", "bo_wakes": "int", "bo_ls_steps": "float",
    "crashed": "bool", "crashed_roles": "str", "crash_t": "float", "exit_codes": "str",
    "gpu_arm": "bool", "gpu_health": "str", "gpu_median_sm_mhz": "float", "gpu_foreign_share": "float",
    "gpu_void": "bool", "block_void": "bool",
    # resources and covariates
    **{f"{r}_startup_s": "float" for r in ROLES}, "startup_s": "float", "exchange_s": "float",
    "exchange_bytes": "float", "solver_nodes_per_s": "float", "solver_lp_iter_per_s": "float", "solver_status": "str",
    "solver_exit_reason": "str", "solved_optimal": "bool", "backend": "str",
    "backend_mismatch": "bool",
    # frontier (exploratory) and exchange window
    "frontier_size": "float", "frontier_hv": "float", "frontier_mean_pairwise": "float", "frontier_max_dist": "float",
    "frontier_mean_dist_to_inc": "float", "cb_first_s": "float", "cb_first_solving_s": "float",
    "cb_max_gap_s": "float", "no_exchange_window": "bool",
}


def _normalise(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Fixed columns and dtypes (missing values: NaN for floats, -1 for ints, False for bools, "" for strings), so
    the parquet content is identical from run to run (reproduce compares by content)."""
    out = {}
    n = len(df)
    for col, typ in schema.items():
        s = df[col] if col in df.columns else pd.Series([None] * n, dtype=object)
        if typ == "float":
            out[col] = pd.to_numeric(s, errors="coerce").astype("float64").to_numpy()
        elif typ == "int":
            out[col] = pd.to_numeric(s, errors="coerce").fillna(-1).astype("int64").to_numpy()
        elif typ == "bool":
            out[col] = s.astype(object).where(s.notna(), False).astype(bool).to_numpy()
        else:
            out[col] = np.array(["" if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v)
                                 for v in s.tolist()], dtype=object)
    extra = sorted(c for c in df.columns if c not in schema)
    res = pd.DataFrame(out, index=pd.RangeIndex(n))
    for c in extra:
        res[c] = df[c].to_numpy()
    return res


def frame_digest(df: pd.DataFrame) -> str:
    """Content hash of a table (column names, dtypes, values) — independent of the parquet writer's version, so a
    result can name its input table and `reproduce` can compare tables by content."""
    import hashlib

    h = hashlib.sha256()
    h.update(json.dumps([[str(c), str(t)] for c, t in df.dtypes.items()]).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="run directories -> datasets/runs.parquet + incumbents_checked.parquet")
    ap.add_argument("--raw", default=str(TRACES / "raw"))
    ap.add_argument("--out", default=str(DATASETS))
    ap.add_argument("--recheck", action="store_true", help="re-check every incumbent (needs the instance files)")
    ap.add_argument("--verify-instances", action="store_true",
                    help="re-check cached verdicts whose instance file changed since the check (hashes every instance)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--strict", action="store_true", help="exit 1 if any incumbent failed or is unchecked")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    if not discover(a.raw):
        print(f"no run directories under {a.raw}")
        return 2
    df, inc = build(a.raw, a.out, a.recheck, a.workers, verify_instances=a.verify_instances)
    bad = inc[~inc["ok"]] if len(inc) else inc
    print(f"{len(df)} runs ({int(df['synthetic'].sum())} synthetic), {len(inc)} incumbents, {len(bad)} failed or "
          f"unchecked -> {a.out}")
    if len(df):
        print(df[~df["synthetic"]].groupby(["set", "arm"]).agg(n=("pt", "size"), pt=("pt", "mean"),
                                                                crashed=("crashed", "sum")).to_string())
    return 1 if (a.strict and len(bad)) else 0


if __name__ == "__main__":
    sys.exit(main())
