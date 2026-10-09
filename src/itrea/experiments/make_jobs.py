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

`backoff0` writes D-018's stage-0 calibration of A9's GPU back-off (development, D-010): seed 2, 120 s,
chromaticindex1024-7, sing44 and comp21-2idx x {A6, A9@bo0, A9@bof50, A9@bof25, A9@bof12}, ca__Scenario400_reserves_1
and glass4 x A9@bo0, and sing44 x A2 (SCIP || cuOpt, zero EA duty: rule (b)'s same-day reference, D-018 amendment) =
18 runs, one block per instance (the pool interleaves a block's arms in Williams order). The A9 variants run the
configuration files configs/itrea2-dev-rc-3way-<variant>.yaml (the runner adds exchange.from_cuopt); A6 and A2 run no
EA (config null). Instance paths come from datasets/dev_set.csv and datasets/classprobe_set.csv.
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
from itrea.experiments.pair import ARMS, stem
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
BACKOFF1_VARIANTS = ("bo0", "bo1")
INSTANCE_TABLES = ("datasets/dev_set.csv", "datasets/classprobe_set.csv")
OPTIONAL = ("emphasis", "ablation", "variant")


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
    """D-018 stage 1: the 14 instances of `analysis.backoff.STAGE1` x seeds x {A9@bo0, A9@bo1, A6} (84 runs for seeds
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
    b1 = sub.add_parser("backoff1", help="D-018 stage 1: A9@bo0 vs A9@bo1 + A6 (42 runs per seed)")
    b1.add_argument("--budget", type=float, default=BACKOFF1_BUDGET)
    b1.add_argument("--seeds", default="0,1", help="comma-separated (one window per seed: 0, then 1)")
    for p in (s1, s2, e1, e4, cf, sm, b0, b1):
        p.add_argument("--out", required=True, type=Path)
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
