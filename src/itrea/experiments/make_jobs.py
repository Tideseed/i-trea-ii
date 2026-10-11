"""Job files (JSONL, one run per line) for every phase of D-002: the two-stage screening pass, E1 and E4 on the tuning
set, and the confirmatory E2/E3 blocks on the test set. The pool (`itrea.experiments.pool`) runs them.

A job line has the fields
    instance, set, name, arm, seed, budget, config, overrides, stratum, split, block
and, where an arm needs them, `emphasis` (A0*), `ablation` (a named A3 ablation, configs/ablations/<name>.yaml) and
`variant` (E4: one label per candidate configuration). Instance paths are repository-relative. Output is
deterministic: the same inputs give byte-identical files (sorted inputs, fixed key order, no clock, no salted hash),
so a job file can be regenerated and diffed.

Seeds are disjoint by phase (D-002 §3): screening 900 / 901, tuning 0–99, confirmatory 1000 / 1001; the screening
seeds are never reused in a confirmatory arm (that would replay the SCIP path that selected the instance).

Stage 2 is derived from the run table itself (`screen2 --runs datasets/runs.parquet` → `hardset.stage2_needed`, the
one implementation of the rule); a hand-made stage-1 table (`--results`) must carry `name`, `p300` and `feasible`
for every row and a P for every feasible row, or it is refused (no fail-open selection).

`is_toy` / `is_candidate` classify a job's instance for the runners: a toy lies outside data/ (symbolic links into
optopt's checkout included) and is not named like a screening candidate; only toys get the smoke exemptions.

    python -m itrea.experiments.make_jobs screen1 --out jobs/screen1.jsonl
    python -m itrea.experiments.make_jobs screen2 --runs datasets/runs.parquet --out jobs/screen2.jsonl
    python -m itrea.experiments.make_jobs e1 --pool datasets/hard_pool.csv --out jobs/e1.jsonl
    python -m itrea.experiments.make_jobs e4 --pool datasets/hard_pool.csv --configs configs/e4/*.yaml --out jobs/e4.jsonl
    python -m itrea.experiments.make_jobs confirm --pool datasets/hard_pool.csv --config configs/frozen-e2.yaml \
        --emphasis <tuned> --out jobs/e2.jsonl
    python -m itrea.experiments.make_jobs smoke --instance /tmp/toy.mps --arms A0,A1 --out /tmp/smoke.jsonl
    python -m itrea.experiments.make_jobs backoff0 --out jobs/dev-backoff0.jsonl
    python -m itrea.experiments.make_jobs backoff1 --seeds 0 --out jobs/dev-backoff1-s0.jsonl   (and --seeds 1)
    python -m itrea.experiments.make_jobs mipfeas-cal [--out jobs/mipfeas-cal-26.08.jsonl]
    python -m itrea.experiments.make_jobs itx-m1 | itx-m2 | itx-m3 | itx-m4 | itx-m5 | itx-r13   [--out jobs/itx-<k>.jsonl]

`backoff0` writes D-018's stage-0 calibration of A9's GPU back-off (development, D-010): seed 2, 120 s,
chromaticindex1024-7, sing44 and comp21-2idx x {A6, A9@bo0, A9@bof50, A9@bof25, A9@bof12}, ca__Scenario400_reserves_1
and glass4 x A9@bo0, and sing44 x A2 (SCIP || cuOpt, zero EA duty: rule (b)'s same-day reference, D-018 amendment) =
18 runs, one block per instance (the pool interleaves a block's arms in Williams order). The A9 variants run the
configuration files configs/itrea2-dev-rc-3way-<variant>.yaml (the runner adds exchange.from_cuopt); A6 and A2 run no
EA (config null). Instance paths come from datasets/dev_set.csv and datasets/classprobe_set.csv.

`mipfeas-cal` writes D-020's calibration of the MIPfeas harness against GAMS's published cuOpt 26.08 GB10 row: the 30
sampled MIPLIB 2017 instances (`itrea.analysis.mipfeas.STRATA`, in D-020's order) x arm A6m (cuOpt as GAMS ran it),
seed 0, 600 s, set `mipfeas`, one block per instance, stratum = D-020's stratum index "0"–"5", instance paths
data/miplib/inst/<name>.mps.gz (each checked on disk). Split `dev`: development (D-010), never `test` — so no frozen
configuration is asked for and the pool's --gpu-mem-gb is allowed (D-020: --gpu-mem-gb 36 --cuopt-mem-gb 24, under
--layout full); 29 of the 30 are screening candidates, so the pool needs `prereg-v1` as an ancestor of HEAD. D-021
replaced this calibration by M4's (A6F at A9's footprint); the A6m file is no longer in jobs/, its writer stays.

`itx-*` write D-021's (ITX-1) phase-1 job files, all split `dev`, seed 0 unless stated, the A9 jobs on
`configs/itrea2-dev-rc-3way.yaml` (the A9 development default), every GPU arm on the GPU lane g0 (F), a job-level
`harness` on every job but R13's control: `itx-m1` (`jobs/itx-m1.jsonl`, set itx-m1: fhnw-binpack4-48, glass4,
radiationm18-12-05 x {A9@m1, A6F@m1, A2F@m1} under the ext harness + A6F@m1env with the thread variables kept, 120 s;
12 runs), `itx-m2` (itx-m2: comp21-2idx, neos-4763324-toguru x {A9@m2, A6F@m2} ext, 600 s, AB / BA by `position`; 4),
`itx-m3` (itx-m3: the 8 instances of R29 x A6F@m3 at cuOpt RMM 40 / host 40 GB, 600 s; 8), `itx-m4` (set mipfeas:
D-020's 30 x A6F ext — the plain label, for `itrea mipfeas-report --arm A6F` —, 600 s, strata "0"–"5"; 30), `itx-m5`
(itx-m5: the 15 development / class-probe instances among MIPfeas's 233 x A9@m5 ext, 600 s; 15) and `itx-r13` (itx-r13:
the 17 x seeds 0–1 x {A9@r13 without a harness, A9@r13x and A6F@r13x under ext}, 120 s, one block per instance x
seed; 102). Each set's own variant keeps every run id distinct from every other set's (run ids carry no set).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

from itrea import paths
from itrea.experiments.pair import ARMS, RunError, resolve_harness, stem
from itrea.experiments.pool import MASTER_SEED, canonical_arms

SCREEN_SEEDS = (900, 901)
TUNING_SEED = 0                         # tuning seeds are 0–99
CONFIRM_SEEDS = (1000, 1001)
SCREEN_BUDGET = 300.0
P_STAGE2 = 0.05                          # stage 2 only for stage-1 P(300) >= 0.05 or no solution (D-002 §1)
E4_MAX_CONFIGS, E4_MAX_INSTANCES, E4_MAX_BUDGET = 12, 16, 120.0
A3R_INSTANCES = 8                        # the live uncoupled-portfolio arm: 8 test instances x 2 seeds (D-002 v3)
# D-002 v3 / D-007: primary A3 vs A0 and A5; secondaries A2, A1, A0*; mechanism A5x0. A3r on a subset, A6 one seed.
CONFIRM_ARMS = ("A0", "A0star", "A1", "A2", "A3", "A5", "A5x0")
EXPLORATORY_ARMS = ("A4", "A5y0")
MIPLIB_EXCLUDE = ("supportcase22",)      # open; E5 only
PGLIB_FAMILIES = ("ca", "rts_gmlc")      # ferc is exploratory only
ML4CO_IP = tuple([f"valid/item_placement_{i}" for i in range(9900, 9925)] +
                 [f"train/item_placement_{i}" for i in range(35)])
FIELDS = ("instance", "set", "name", "arm", "seed", "budget", "config", "overrides", "stratum", "split", "block")
# D-018 stage 0 (A9 GPU back-off calibration)
BACKOFF0_SET, BACKOFF0_SEED, BACKOFF0_BUDGET = "backoff0", 2, 120.0
BACKOFF0_DUTY_INSTANCES = ("chromaticindex1024-7", "sing44", "comp21-2idx")
BACKOFF0_LEAD_INSTANCES = ("ca__Scenario400_reserves_1", "glass4")
BACKOFF0_VARIANTS = ("bo0", "bof50", "bof25", "bof12")
BACKOFF0_A2_INSTANCES = ("sing44",)      # rule (b)'s zero-duty reference (D-018 amendment, 2026-10-09)
BACKOFF_BASE = "configs/itrea2-dev-rc-3way"
# D-018 stage 1 (A9@bo0 shadow vs A9@bo1 active, same-day A6 in every block; two windows: seed 0, then seed 1)
BACKOFF1_SET, BACKOFF1_BUDGET = "backoff1", 120.0
BACKOFF1_VARIANTS = ("bo1s", "bo1")        # control bo1s = bo1's values in shadow mode (M6 reads its would-be timeline)
INSTANCE_TABLES = ("datasets/dev_set.csv", "datasets/classprobe_set.csv")
# D-020: the MIPfeas harness's calibration (arm A6m on D-020's 30-instance sample)
MIPFEAS_SET, MIPFEAS_ARM, MIPFEAS_SEED, MIPFEAS_BUDGET = "mipfeas", "A6m", 0, 600.0
MIPFEAS_CAL_OUT = "jobs/mipfeas-cal-26.08.jsonl"
# D-021 (ITX-1) phase 1: the measurements before C1 (M1 smoke, M2 soak, M3 memory probe, M4 calibration, M5 A9
# preview) and the R13 + R14 development test. Every set's labels carry a variant of its own (m1, m2, …), so no run id
# of one set equals one of another set or of an earlier set; M4 keeps the plain label A6F (`mipfeas-report --arm A6F`)
ITX_SEED = 0
ITX_A9_CONFIG = "configs/itrea2-dev-rc-3way.yaml"   # the A9 development default (D-015 / D-018: back-off not adopted)
ITX_EXT = "ext"                                      # pair.HARNESS_PRESETS: gaps 0, env unset, RMM / host 24, CUDA cache
ITX_ENV_KEPT = {"preset": "ext", "cuopt_env_unset": False}     # M1's env-only contrast (R14)
ITX_M3_HARNESS = {"preset": "ext", "cuopt_rmm_gb": 40.0, "cuopt_host_gb": 40.0}   # uncensored memory probe (R29)
# M1: fast development instances with continuous variables, one of them an easy one cuOpt proves quickly
ITX_M1_INSTANCES = ("fhnw-binpack4-48", "glass4", "radiationm18-12-05")
ITX_M2_INSTANCES = ("comp21-2idx", "neos-4763324-toguru")
# R29: the 8 instances whose RSS exceeded 14.4 GiB in GAMS's cuOpt run
ITX_M3_INSTANCES = ("eilA101-2", "irp", "nw04", "proteindesign121hz512p9", "proteindesign122trx11p8",
                    "snp-02-004-104", "square47", "supportcase6")
ITX_OUT = {k: f"jobs/itx-{k}.jsonl" for k in ("m1", "m2", "m3", "m4", "m5", "r13")}
OPTIONAL = ("emphasis", "ablation", "variant", "harness", "position")


class JobError(ValueError):
    pass


# ----------------------------------------------------------------------------------------------------- one job
def job(instance: str, set_: str, arm: str, seed: int, budget: float, *, name: str | None = None,
        config: str | None = None, overrides: dict | None = None, stratum: str | None = None, split: str = "smoke",
        block: str | None = None, **extra) -> dict:
    if arm not in ARMS:
        raise JobError(f"unknown arm {arm!r}")
    name = name or stem(instance)
    d = dict(instance=str(instance), set=set_, name=name, arm=arm, seed=int(seed), budget=float(budget),
             config=config, overrides=dict(overrides or {}), stratum=stratum, split=split,
             block=block or f"{name}__s{int(seed)}")
    for k in OPTIONAL:
        if extra.get(k) is not None:
            d[k] = extra[k]
    unknown = set(extra) - set(OPTIONAL)
    if unknown:
        raise JobError(f"unknown job fields {sorted(unknown)}")
    try:
        resolve_harness(d.get("harness"))
    except RunError as e:
        raise JobError(str(e)) from None
    return d


def write_jobs(jobs: list[dict], path: str | os.PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    os.replace(tmp, p)
    return p


def _sorted(jobs: list[dict]) -> list[dict]:
    """Stable file order: by set, block, then the canonical arm order (the pool reorders by its design anyway)."""
    order = {a: i for i, a in enumerate(canonical_arms({_label(j) for j in jobs}))}
    return sorted(jobs, key=lambda j: (j["set"], j["block"], order[_label(j)]))


def _label(j: dict) -> str:
    return j["arm"] + (f"~{j['ablation']}" if j.get("ablation") else "") + (f"@{j['variant']}" if j.get("variant")
                                                                          else "")


# -------------------------------------------------------------------------------------------------- candidates
def _rel(f: Path) -> str:
    """Repository-relative path WITHOUT resolving symbolic links (data/ holds links into optopt's checkout, and the
    job files must say data/miplib/inst/x.mps.gz, not the link target)."""
    try:
        return str(Path(f).relative_to(paths.ROOT))
    except ValueError:
        return paths.relative(f)


def miplib_infeasible(solu: Path) -> set[str]:
    out = set()
    if solu.exists():
        for line in solu.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "=inf=":
                out.add(parts[1])
    return out


def candidates(data: Path | None = None) -> list[dict]:
    """The screening candidates of D-002 §2 that are on disk: MIPLIB 2017 benchmark minus the `=inf=` instances and
    supportcase22; PGLib-UC `ca` and `rts_gmlc`; ML4CO item placement valid 9900–9924 and train 0–34.
    Rows: {instance (repository-relative), name, family, stratum}."""
    data = Path(data or paths.DATA)
    rows = []
    inst = data / "miplib" / "inst"
    infeasible = miplib_infeasible(data / "miplib" / "solu.txt")
    for f in sorted(inst.glob("*.mps.gz")) if inst.is_dir() else []:
        n = stem(f.name)
        if n in infeasible or n in MIPLIB_EXCLUDE:
            continue
        rows.append(dict(instance=_rel(f), name=n, family=miplib_family(n), stratum="miplib"))
    mps = data / "pglib_uc" / "mps"
    for f in sorted(mps.glob("*.mps.gz")) if mps.is_dir() else []:
        fam = f.name.split("__")[0]
        if fam in PGLIB_FAMILIES:
            rows.append(dict(instance=_rel(f), name=stem(f.name), family=fam, stratum="pglib_uc"))
    ip = data / "ml4co" / "instances" / "1_item_placement"
    for rel in ML4CO_IP:
        f = ip / f"{rel}.mps.gz"
        if f.exists():
            rows.append(dict(instance=_rel(f), name=stem(f.name), family="ml4co_ip", stratum="ml4co_ip"))
    return rows


def candidate_names(data: Path | None = None) -> set[str]:
    return {c["name"] for c in candidates(data)}


def data_roots(data: Path | None = None) -> set[Path]:
    """data/ and the real directories behind its entries (data/miplib, data/ml4co, data/pglib_uc are symbolic links
    into optopt's checkout, so a resolved instance path is never under data/ itself)."""
    d = Path(data or paths.DATA)
    roots = {d.absolute()}
    try:
        roots.add(d.resolve())
        if d.is_dir():
            for c in d.iterdir():
                try:
                    roots.add(c.resolve())
                except OSError:
                    continue
    except OSError:
        pass
    return roots


def is_candidate(job: dict, names: set[str] | None = None) -> bool:
    """The job's instance is a D-002 §2 screening candidate (by name: the file stem or the job's `name`)."""
    names = candidate_names() if names is None else names
    keys = {stem(str(job["instance"]))}
    if job.get("name"):
        keys.add(str(job["name"]))
    return bool(keys & names)


def is_toy(job: dict, names: set[str] | None = None, roots: set[Path] | None = None) -> bool:
    """A toy instance: outside data/ (absolute or resolved through symbolic links) and not named like a screening
    candidate. Only toys may use the smoke exemptions (no lease, no systemd scopes, no prereg-v1)."""
    p = Path(paths.resolve(job["instance"]))
    forms = {p.absolute()}
    try:
        forms.add(p.resolve())
    except OSError:
        pass
    roots = data_roots() if roots is None else roots
    if any(f == r or f.is_relative_to(r) for f in forms for r in roots):
        return False
    return not is_candidate(job, names)


def miplib_family(name: str) -> str:
    """Family prefix for the group split (D-002 §2): `neos-*`, `supportcase*`, `comp*`, … — the leading run of letters
    (with `neos` normalised so `neos-123-abc` and `neos859080` are one family)."""
    m = re.match(r"[A-Za-z]+", name)
    fam = (m.group(0) if m else name).lower()
    return "neos" if fam.startswith("neos") else fam


# ---------------------------------------------------------------------------------------------------- screening
def screening_stage1(cands: list[dict], budget: float = SCREEN_BUDGET, seed: int = SCREEN_SEEDS[0],
                     set_: str = "screen1") -> list[dict]:
    """Every candidate with seed 900, arm SCREEN (SCIP default, plugin inert)."""
    return _sorted([job(c["instance"], set_, "SCREEN", seed, budget, name=c["name"], stratum=c.get("stratum"),
                        split="screen") for c in sorted(cands, key=lambda c: c["name"])])


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def _float_or_none(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "na"):
        return None
    return float(s)


def read_rows(path: str | os.PathLike) -> list[dict]:
    """CSV (header row) or JSONL rows."""
    p = Path(path)
    if p.suffix in (".jsonl", ".json"):
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def stage2_selected(results: list[dict], p_min: float = P_STAGE2) -> list[str]:
    """Names that get the second screening seed: stage-1 P(300) >= p_min, or no feasible solution found.
    Every row needs `name` (or `instance`), `p300` and `feasible`; p300 may be blank only for an infeasible row.
    Anything else raises JobError (a malformed table must not select every instance)."""
    if not results:
        raise JobError("stage-1 table is empty")
    out = []
    for i, r in enumerate(results, 1):
        missing = [k for k in ("p300", "feasible") if k not in r]
        if missing:
            raise JobError(f"stage-1 table row {i} lacks {missing} (needs name, p300, feasible)")
        name = r.get("name") or (stem(r["instance"]) if r.get("instance") else None)
        if not name:
            raise JobError(f"stage-1 table row {i} has no name")
        feas = _truthy(r["feasible"])
        try:
            p = _float_or_none(r["p300"])
        except ValueError:
            raise JobError(f"stage-1 table row {i} ({name}): p300 {r['p300']!r} is not a number") from None
        if feas and p is None:
            raise JobError(f"stage-1 table row {i} ({name}): feasible but p300 is blank or NaN")
        if not feas or p >= p_min:
            out.append(name)
    return sorted(set(out))


def stage2_from_runs(runs_path: str | os.PathLike) -> list[str]:
    """The stage-2 list from the run table (`itrea build-runs` → datasets/runs.parquet): hardset.stage2_needed, the
    one implementation of the rule (seed-900 P(300) >= 0.05 or no solution, seed 901 not yet run)."""
    import pandas as pd

    from itrea.benchmarks import hardset
    return sorted(set(hardset.stage2_needed(pd.read_parquet(runs_path))))


def screening_stage2(results: list[dict], cands: list[dict], budget: float = SCREEN_BUDGET,
                     seed: int = SCREEN_SEEDS[1], set_: str = "screen2", p_min: float = P_STAGE2) -> list[dict]:
    return screening_stage2_names(stage2_selected(results, p_min), cands, budget, seed, set_)


def screening_stage2_names(names, cands: list[dict], budget: float = SCREEN_BUDGET, seed: int = SCREEN_SEEDS[1],
                           set_: str = "screen2") -> list[dict]:
    keep = set(names)
    by = {c["name"]: c for c in cands}
    missing = keep - set(by)
    if missing:
        raise JobError(f"stage-1 results name instances that are not candidates: {sorted(missing)[:5]}")
    return _sorted([job(by[n]["instance"], set_, "SCREEN", seed, budget, name=n, stratum=by[n].get("stratum"),
                        split="screen") for n in sorted(keep)])


# ---------------------------------------------------------------------------------------------------- the pool
def read_pool(path: str | os.PathLike, split: str | None = None) -> list[dict]:
    """datasets/hard_pool.csv rows: instance, name, stratum, split (or `side`: test | tuning)."""
    rows = []
    for r in read_rows(path):
        sp = r.get("split") or r.get("side")
        if split is not None and sp != split:
            continue
        rows.append(dict(instance=r["instance"], name=r.get("name") or stem(r["instance"]),
                         stratum=r.get("stratum") or "-", split=sp))
    return sorted(rows, key=lambda r: r["name"])


def a3r_subset(rows: list[dict], k: int = A3R_INSTANCES, seed: int = MASTER_SEED) -> list[str]:
    """k test instances for the live uncoupled arm A3r, spread over the strata: strata in name order, instances
    shuffled within each stratum with a fixed seed, then taken round robin."""
    rng = random.Random(seed)
    by: dict[str, list[str]] = {}
    for r in sorted(rows, key=lambda r: r["name"]):
        by.setdefault(r["stratum"], []).append(r["name"])
    for s in sorted(by):
        rng.shuffle(by[s])
    out: list[str] = []
    while len(out) < k and any(by.values()):
        for s in sorted(by):
            if by[s] and len(out) < k:
                out.append(by[s].pop(0))
    return sorted(out)


def confirmatory(rows: list[dict], config: str, arms=CONFIRM_ARMS, seeds=CONFIRM_SEEDS, budget: float = 300.0,
                 set_: str = "e2", emphasis: str | None = None, a3r: list[str] | None = None,
                 a6_seeds=(CONFIRM_SEEDS[0],), ablations=(), split: str = "test") -> list[dict]:
    """Blocks = test instance x seed; every arm of a block carries the frozen configuration (the runner checks its
    hash), A3r only on its subset, A6 on one seed, named A3 ablations as extra treatments of the block."""
    rows = [r for r in rows if r.get("split") in (split, None)]
    if not rows:
        raise JobError(f"no {split} instances in the pool")
    if "A0star" in arms and not emphasis:
        raise JobError("A0star needs --emphasis (the static SCIP emphasis chosen on the tuning set)")
    if not config:
        raise JobError("confirmatory jobs need the frozen configuration (--config configs/frozen-e2.yaml)")
    a3r_names = set(a3r if a3r is not None else a3r_subset(rows))
    out = []
    for r in rows:
        for s in seeds:
            for arm in arms:
                out.append(job(r["instance"], set_, arm, s, budget, name=r["name"], config=config,
                               stratum=r["stratum"], split=split, emphasis=emphasis if arm == "A0star" else None))
            if r["name"] in a3r_names:
                out.append(job(r["instance"], set_, "A3r", s, budget, name=r["name"], config=config,
                               stratum=r["stratum"], split=split))
            if s in a6_seeds:
                out.append(job(r["instance"], set_, "A6", s, budget, name=r["name"], config=config,
                               stratum=r["stratum"], split=split))
            for ab in ablations:
                out.append(job(r["instance"], set_, "A3", s, budget, name=r["name"], config=config,
                               stratum=r["stratum"], split=split, ablation=ab))
    return _sorted(out)


# ----------------------------------------------------------------------------------------------------- tuning
def e1_jobs(rows: list[dict], config: str | None = None, budget: float = 300.0, seed: int = TUNING_SEED,
            set_: str = "e1") -> list[dict]:
    """E1 (D-002 §7): the EA alone (A5) vs cuOpt alone (A6) on the tuning set."""
    rows = [r for r in rows if r.get("split") in ("tuning", None)]
    out = []
    for r in rows:
        for arm in ("A5", "A6"):
            out.append(job(r["instance"], set_, arm, seed, budget, name=r["name"], config=config,
                           stratum=r["stratum"], split="tuning"))
    return _sorted(out)


def e4_jobs(rows: list[dict], configs: list[str], default_config: str | None = "configs/itrea2-default.yaml",
            budget: float = E4_MAX_BUDGET, seed: int = TUNING_SEED, set_: str = "e4") -> list[dict]:
    """E4 (D-002 §7): at most 12 configurations x 16 tuning instances x 120 s x 1 seed, every configuration logged;
    arm A5, one treatment per configuration (`variant` = the file's stem), plus A5-default as the reference."""
    rows = [r for r in rows if r.get("split") in ("tuning", None)]
    cfgs = sorted(set(configs))
    if len(cfgs) > E4_MAX_CONFIGS:
        raise JobError(f"E4: {len(cfgs)} configurations > {E4_MAX_CONFIGS} (pre-registered cap)")
    if len(rows) > E4_MAX_INSTANCES:
        raise JobError(f"E4: {len(rows)} instances > {E4_MAX_INSTANCES} (pre-registered cap)")
    if float(budget) > E4_MAX_BUDGET:
        raise JobError(f"E4: budget {budget} s > {E4_MAX_BUDGET} s (pre-registered cap)")
    variants = [(Path(c).stem, c) for c in cfgs]
    if default_config and default_config not in cfgs:
        variants.append(("default", default_config))
    names = [v for v, _ in variants]
    if len(set(names)) != len(names):
        raise JobError(f"E4: configuration file stems must be unique: {names}")
    out = []
    for r in rows:
        for v, c in variants:
            out.append(job(r["instance"], set_, "A5", seed, budget, name=r["name"], config=c, stratum=r["stratum"],
                           split="tuning", variant=v))
    return _sorted(out)


def instance_paths(names, tables=INSTANCE_TABLES) -> dict[str, str]:
    """{name: repository-relative instance path} from the development tables (datasets/dev_set.csv,
    datasets/classprobe_set.csv); a name in none of them is an error."""
    found: dict[str, str] = {}
    for t in tables:
        p = paths.resolve(t)
        if Path(p).exists():
            for r in read_rows(p):
                found.setdefault(r.get("name") or stem(r["instance"]), r["instance"])
    missing = [n for n in names if n not in found]
    if missing:
        raise JobError(f"instances not in {', '.join(tables)}: {missing}")
    return {n: found[n] for n in names}


def backoff_stage0(budget: float = BACKOFF0_BUDGET, seed: int = BACKOFF0_SEED, set_: str = BACKOFF0_SET,
                   tables=INSTANCE_TABLES) -> list[dict]:
    """D-018 stage 0 (module docstring): 3 instances x {A6, A9@bo0, A9@bof50, A9@bof25, A9@bof12} + 2 x A9@bo0 +
    sing44 x A2 (18 runs)."""
    names = BACKOFF0_DUTY_INSTANCES + BACKOFF0_LEAD_INSTANCES
    where = instance_paths(names, tables)
    out = []
    for n in names:
        common = dict(name=n, stratum=set_, split="dev", block=f"{n}__s{int(seed)}__{set_}")
        variants = BACKOFF0_VARIANTS if n in BACKOFF0_DUTY_INSTANCES else BACKOFF0_VARIANTS[:1]
        for v in variants:
            out.append(job(where[n], set_, "A9", seed, budget, config=f"{BACKOFF_BASE}-{v}.yaml", variant=v, **common))
        if n in BACKOFF0_DUTY_INSTANCES:
            out.append(job(where[n], set_, "A6", seed, budget, config=None, **common))
        if n in BACKOFF0_A2_INSTANCES:
            out.append(job(where[n], set_, "A2", seed, budget, config=None, **common))
    return _sorted(out)


def backoff_stage1(seeds=(0, 1), budget: float = BACKOFF1_BUDGET, set_: str = BACKOFF1_SET,
                   tables=INSTANCE_TABLES) -> list[dict]:
    """D-018 stage 1: the 14 instances of `analysis.backoff.STAGE1` x seeds x {A9@bo1s (control: bo1's values in shadow mode), A9@bo1, A6} (84 runs for seeds
    0-1). `configs/itrea2-dev-rc-3way-bo1.yaml` is written from the stage-0 report before this file is used; the pool
    refuses a job whose configuration file is missing."""
    from itrea.analysis.backoff import STAGE1
    where = instance_paths(STAGE1, tables)
    out = []
    for n in STAGE1:
        for seed in seeds:
            common = dict(name=n, stratum=set_, split="dev", block=f"{n}__s{int(seed)}__{set_}")
            for v in BACKOFF1_VARIANTS:
                out.append(job(where[n], set_, "A9", seed, budget, config=f"{BACKOFF_BASE}-{v}.yaml", variant=v,
                               **common))
            out.append(job(where[n], set_, "A6", seed, budget, config=None, **common))
    return _sorted(out)


def mipfeas_calibration(budget: float = MIPFEAS_BUDGET, seed: int = MIPFEAS_SEED, set_: str = MIPFEAS_SET,
                        data: Path | None = None) -> list[dict]:
    """D-020's calibration jobs (module docstring), in D-020's order. The stratum is written as a string: the pool and
    build_runs read a falsy stratum (0) as none. A missing instance file is an error."""
    from itrea.analysis.mipfeas import STRATA
    inst = Path(data or paths.DATA) / "miplib" / "inst"
    missing = [n for s in STRATA for n in s if not (inst / f"{n}.mps.gz").is_file()]
    if missing:
        raise JobError(f"instance files missing under {inst}: {missing}")
    return [job(_rel(inst / f"{n}.mps.gz"), set_, MIPFEAS_ARM, seed, budget, name=n, config=None, stratum=str(k),
                split="dev", block=f"{n}__s{int(seed)}__{set_}")
            for k, s in enumerate(STRATA) for n in s]


# ------------------------------------------------------------------------------------------- D-021 (ITX-1) phase 1
def mipfeas_233(data: Path | None = None) -> list[str]:
    """MIPfeas's 233 instances: `data/miplib/benchmark-v2.test` minus its `=inf=` instances in solu.txt (D-020; not
    `candidates()`, which also drops the open supportcase22). Anything but 233 names is an error."""
    mip = Path(data or paths.DATA) / "miplib"
    test = mip / "benchmark-v2.test"
    if not test.is_file():
        raise JobError(f"{test} does not exist")
    names = {stem(x) for x in test.read_text().split()} - miplib_infeasible(mip / "solu.txt")
    if len(names) != 233:
        raise JobError(f"{test} minus the =inf= instances gives {len(names)} names, not MIPfeas's 233")
    return sorted(names)


def development_names(tables=INSTANCE_TABLES) -> list[str]:
    """The 17 development + class-probe instances (datasets/dev_set.csv, datasets/classprobe_set.csv), in table order."""
    out: list[str] = []
    for t in tables:
        p = paths.resolve(t)
        for r in read_rows(p):
            n = r.get("name") or stem(r["instance"])
            if n not in out:
                out.append(n)
    return out


def _miplib_paths(names, data: Path | None = None) -> dict[str, str]:
    inst = Path(data or paths.DATA) / "miplib" / "inst"
    missing = [n for n in names if not (inst / f"{n}.mps.gz").is_file()]
    if missing:
        raise JobError(f"instance files missing under {inst}: {missing}")
    return {n: _rel(inst / f"{n}.mps.gz") for n in names}


def _itx(where: str, name: str, set_: str, arm: str, seed: int, budget: float, variant: str | None, harness,
         **kw) -> dict:
    return job(where, set_, arm, seed, budget, name=name, config=ITX_A9_CONFIG if arm == "A9" else None,
               split="dev", block=kw.pop("block", f"{name}__s{int(seed)}__{set_}"),
               stratum=kw.pop("stratum", set_), variant=variant, harness=harness, **kw)


def itx_m1(budget: float = 120.0, seed: int = ITX_SEED, set_: str = "itx-m1", tables=INSTANCE_TABLES) -> list[dict]:
    """M1 smoke (D-021): 3 development instances x {A9, A6F, A2F} under the ext harness, plus the env-only contrast
    A6F with the thread variables kept (`A6F@m1env`, ext otherwise) beside A6F under ext; 120 s, seed 0, one block per
    instance (12 runs)."""
    where = instance_paths(ITX_M1_INSTANCES, tables)
    out = []
    for n in ITX_M1_INSTANCES:
        for arm in ("A9", "A6F", "A2F"):
            out.append(_itx(where[n], n, set_, arm, seed, budget, "m1", ITX_EXT))
        out.append(_itx(where[n], n, set_, "A6F", seed, budget, "m1env", ITX_ENV_KEPT))
    return _sorted(out)


def itx_m2(budget: float = 600.0, seed: int = ITX_SEED, set_: str = "itx-m2", tables=INSTANCE_TABLES) -> list[dict]:
    """M2 soak at F (D-021): comp21-2idx and neos-4763324-toguru x {A9, A6F} under the ext harness, 600 s, seed 0, in
    AB / BA order (`position`: A9 first on comp21-2idx, A6F first on toguru — the Williams row is the same for both
    blocks, so the order is written into the jobs); 4 runs. The first from fan AUTO; run with `--pause-c 90`."""
    where = instance_paths(ITX_M2_INSTANCES, tables)
    out = []
    for k, n in enumerate(ITX_M2_INSTANCES):
        order = ("A9", "A6F") if k % 2 == 0 else ("A6F", "A9")
        for arm in ("A9", "A6F"):
            out.append(_itx(where[n], n, set_, arm, seed, budget, "m2", ITX_EXT, position=order.index(arm)))
    return _sorted(out)


def itx_m3(budget: float = 600.0, seed: int = ITX_SEED, set_: str = "itx-m3", data: Path | None = None) -> list[dict]:
    """M3 memory probe (D-021 R29): the 8 instances whose RSS exceeded 14.4 GiB in GAMS's run x A6F alone at cuOpt RMM
    40 / host 40 GB (memory gate 101 GB), 600 s, seed 0, one block each (8 runs)."""
    where = _miplib_paths(ITX_M3_INSTANCES, data)
    return _sorted([_itx(where[n], n, set_, "A6F", seed, budget, "m3", ITX_M3_HARNESS) for n in ITX_M3_INSTANCES])


def itx_m4(budget: float = MIPFEAS_BUDGET, seed: int = MIPFEAS_SEED, set_: str = MIPFEAS_SET,
           data: Path | None = None) -> list[dict]:
    """M4 calibration (D-021, replaces D-020's A6m file): D-020's 30-instance sample x A6F under the ext harness
    (plain label `A6F`, for `itrea mipfeas-report --arm A6F`), 600 s, seed 0, set `mipfeas`, stratum "0"–"5", one block
    each, in D-020's order (30 runs)."""
    from itrea.analysis.mipfeas import STRATA
    names = [n for s in STRATA for n in s]
    where = _miplib_paths(names, data)
    return [_itx(where[n], n, set_, "A6F", seed, budget, None, ITX_EXT, stratum=str(k))
            for k, s in enumerate(STRATA) for n in s]


def itx_m5(budget: float = 600.0, seed: int = ITX_SEED, set_: str = "itx-m5", tables=INSTANCE_TABLES,
           data: Path | None = None) -> list[dict]:
    """M5 A9 preview (D-021): the development / class-probe instances that are among MIPfeas's 233 (15) x A9 under the
    ext harness, 600 s, seed 0, one block each (15 runs)."""
    the233 = set(mipfeas_233(data))
    names = [n for n in development_names(tables) if n in the233]
    where = instance_paths(names, tables)
    return _sorted([_itx(where[n], n, set_, "A9", seed, budget, "m5", ITX_EXT) for n in names])


def itx_r13(budget: float = 120.0, seeds=(0, 1), set_: str = "itx-r13", tables=INSTANCE_TABLES) -> list[dict]:
    """The A9 + R13 + R14 development test (D-021 R14): the 17 development + class-probe instances x seeds 0–1 x {A9
    (`A9@r13`, the development default, no harness), A9 under the ext harness (`A9@r13x`), A6F under it
    (`A6F@r13x`)}, 120 s, one block per instance x seed (102 runs). Its decision record fixes the rules first."""
    names = development_names(tables)
    where = instance_paths(names, tables)
    out = []
    for n in names:
        for seed in seeds:
            out.append(_itx(where[n], n, set_, "A9", seed, budget, "r13", None))
            out.append(_itx(where[n], n, set_, "A9", seed, budget, "r13x", ITX_EXT))
            out.append(_itx(where[n], n, set_, "A6F", seed, budget, "r13x", ITX_EXT))
    return _sorted(out)


def smoke(instances: list[str], arms, seeds=(0,), budget: float = 5.0, set_: str = "smoke",
          config: str | None = None) -> list[dict]:
    """Toy smoke-test jobs (split "smoke": the only split the pool runs without a lease)."""
    out = []
    for inst in sorted(instances):
        for s in seeds:
            for a in arms:
                out.append(job(inst, set_, a, s, budget, config=config, stratum="toy", split="smoke"))
    return _sorted(out)


# ------------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="itrea.experiments.make_jobs", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("screen1", help="stage 1: every candidate, seed 900")
    s2 = sub.add_parser("screen2", help="stage 2: seed 901 for stage-1 P(300) >= 0.05 or no solution")
    src = s2.add_mutually_exclusive_group(required=True)
    src.add_argument("--runs", help="datasets/runs.parquet (itrea build-runs): the rule is hardset.stage2_needed")
    src.add_argument("--results", help="a stage-1 table: name, p300 (blank only when infeasible), feasible")
    for p in (s1, s2):
        p.add_argument("--budget", type=float, default=SCREEN_BUDGET)
        p.add_argument("--data", type=Path, default=paths.DATA)
    e1 = sub.add_parser("e1")
    e4 = sub.add_parser("e4")
    e4.add_argument("--configs", nargs="+", required=True)
    e4.add_argument("--default-config", default="configs/itrea2-default.yaml")
    cf = sub.add_parser("confirm")
    cf.add_argument("--arms", default=",".join(CONFIRM_ARMS))
    cf.add_argument("--emphasis")
    cf.add_argument("--ablations", default="", help="comma list of configs/ablations/<name>.yaml names")
    cf.add_argument("--a3r", default="", help="comma list of instance names (default: 8 spread over the strata)")
    cf.add_argument("--seeds", default=",".join(map(str, CONFIRM_SEEDS)))
    for p in (e1, e4, cf):
        p.add_argument("--pool", required=True, help="datasets/hard_pool.csv")
    for p in (e1, cf):
        p.add_argument("--config")
        p.add_argument("--budget", type=float, default=300.0)
    e4.add_argument("--budget", type=float, default=E4_MAX_BUDGET)
    sm = sub.add_parser("smoke")
    sm.add_argument("--instance", nargs="+", required=True)
    sm.add_argument("--arms", default="A0")
    sm.add_argument("--budget", type=float, default=5.0)
    sm.add_argument("--config")
    b0 = sub.add_parser("backoff0", help="D-018 stage 0: A9 GPU back-off calibration (18 runs)")
    b0.add_argument("--budget", type=float, default=BACKOFF0_BUDGET)
    b0.add_argument("--seed", type=int, default=BACKOFF0_SEED)
    b1 = sub.add_parser("backoff1", help="D-018 stage 1: A9@bo1s (shadow control) vs A9@bo1 + A6 (42 runs per seed)")
    b1.add_argument("--budget", type=float, default=BACKOFF1_BUDGET)
    b1.add_argument("--seeds", default="0,1", help="comma-separated (one window per seed: 0, then 1)")
    for p in (s1, s2, e1, e4, cf, sm, b0, b1):
        p.add_argument("--out", required=True, type=Path)
    mf = sub.add_parser("mipfeas-cal", help="D-020: the MIPfeas calibration, A6m on 30 sampled MIPLIB instances (30 runs)")
    mf.add_argument("--budget", type=float, default=MIPFEAS_BUDGET)
    mf.add_argument("--seed", type=int, default=MIPFEAS_SEED)
    mf.add_argument("--data", type=Path, default=paths.DATA)
    mf.add_argument("--out", type=Path, default=paths.ROOT / MIPFEAS_CAL_OUT)
    itx_help = {"m1": "D-021 M1 smoke: 3 dev instances x {A9, A6F, A2F} ext + A6F env kept, 120 s (12 runs)",
                "m2": "D-021 M2 soak: comp21-2idx, neos-4763324-toguru x {A9, A6F} ext, 600 s, AB / BA (4 runs)",
                "m3": "D-021 M3 memory probe: 8 instances x A6F at cuOpt RMM 40 / host 40, 600 s (8 runs)",
                "m4": "D-021 M4 calibration: D-020's 30 x A6F ext, 600 s, set mipfeas (30 runs)",
                "m5": "D-021 M5 A9 preview: the 15 dev / class-probe instances in the 233 x A9 ext, 600 s (15 runs)",
                "r13": "D-021 R13 + R14 test: 17 x seeds 0-1 x {A9, A9 ext, A6F ext}, 120 s (102 runs)"}
    itx = {k: sub.add_parser(f"itx-{k}", help=h) for k, h in itx_help.items()}
    for k, p in itx.items():
        p.add_argument("--out", type=Path, default=paths.ROOT / ITX_OUT[k])
        if k in ("m3", "m4", "m5"):
            p.add_argument("--data", type=Path, default=paths.DATA)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "screen1":
            jobs = screening_stage1(candidates(a.data), a.budget)
        elif a.cmd == "screen2":
            if a.runs:
                jobs = screening_stage2_names(stage2_from_runs(a.runs), candidates(a.data), a.budget)
            else:
                jobs = screening_stage2(read_rows(a.results), candidates(a.data), a.budget)
        elif a.cmd == "e1":
            jobs = e1_jobs(read_pool(a.pool, "tuning"), a.config, a.budget)
        elif a.cmd == "e4":
            jobs = e4_jobs(read_pool(a.pool, "tuning"), a.configs, a.default_config, a.budget)
        elif a.cmd == "confirm":
            jobs = confirmatory(read_pool(a.pool, "test"), a.config, tuple(x for x in a.arms.split(",") if x),
                                tuple(int(x) for x in a.seeds.split(",")), a.budget, emphasis=a.emphasis,
                                a3r=[x for x in a.a3r.split(",") if x] or None,
                                ablations=tuple(x for x in a.ablations.split(",") if x))
        elif a.cmd == "backoff0":
            jobs = backoff_stage0(a.budget, a.seed)
        elif a.cmd == "backoff1":
            jobs = backoff_stage1(tuple(int(x) for x in a.seeds.split(",") if x), a.budget)
        elif a.cmd == "mipfeas-cal":
            jobs = mipfeas_calibration(a.budget, a.seed, data=a.data)
        elif a.cmd == "itx-m1":
            jobs = itx_m1()
        elif a.cmd == "itx-m2":
            jobs = itx_m2()
        elif a.cmd == "itx-m3":
            jobs = itx_m3(data=a.data)
        elif a.cmd == "itx-m4":
            jobs = itx_m4(data=a.data)
        elif a.cmd == "itx-m5":
            jobs = itx_m5(data=a.data)
        elif a.cmd == "itx-r13":
            jobs = itx_r13()
        else:
            jobs = smoke(a.instance, tuple(a.arms.split(",")), budget=a.budget, config=a.config)
    except (JobError, OSError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    write_jobs(jobs, a.out)
    blocks = len({(j["set"], j["block"]) for j in jobs})
    print(f"{a.out}: {len(jobs)} jobs in {blocks} blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
