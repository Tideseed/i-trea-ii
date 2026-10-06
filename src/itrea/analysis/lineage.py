"""Lineage — the descriptive Rank 0 mechanism metric of D-002 §4: depth-1 matched odds ratio from the genealogy edge
lists (`ea.genealogy.npz`), pooled by Mantel–Haenszel over generations and runs.

Why depth 1 and why matched (critique 2026-10-05, algorithm lens): the OR-accumulated ancestry bit masks saturate —
under neutral selection nearly every child has a Rank 2 ancestor within five generations — so "share of improvements
with a Rank 0 ancestor" against "Rank 0's share of parents" reports an effect whatever Rank 0 does. Here only the two
direct parents count, and every success is compared with the other children born in the SAME generation of the SAME
run (one 2 × 2 table per stratum):

    exposure  = at least one parent was in Rank 0 (infeasible good) at mating time
    cases     = children that became a new population best, or that the solver accepted as a post
    reference = "all" (default, D-002's wording: all children of the generation, cases included)
                | "noncases" (sensitivity: the children that are not cases — the textbook case-control table)

Only bred children enter (pa ≥ 0): seeds, immigrants and injected solutions have no parents, cannot be exposed, and
are counted separately (`cases_without_parents`). The causal claim about Rank 0 rests on H6 (x = 10 % vs 0); this is
description.

Genealogy columns (vault `10-design/Trace format and metrics.md`): id, pa, pb (−1 = none; mutation-only children have
pa = pb), rank_pa, rank_pb (ranks at mating), origin, birth (generation), key, V (or viol), feasible, t_first.
Optional: is_new_best, is_accepted_post. When those two are absent they are DERIVED and the result says so:
is_new_best = a feasible row whose key is strictly below every earlier feasible key (rows ordered by birth, id);
is_accepted_post = the earliest feasible row whose objective matches an accepted post (sense from the summary).

Which runs are pooled (review 2026-10-06, F7): the run directories of `build_runs.discover` (runs the pool moved to
`<set>/void/` are skipped explicitly), filtered on run.json — `set` (default e2, the confirmatory runs; E1 tuning,
E4 variants and smoke runs use other configurations) and `split` (default test) — and on the arm; how many directories
each filter dropped is written beside the estimate. An empty filter value means "any".

    python -m itrea.analysis.lineage traces/raw [--arms A5,A3] [--sets e2] [--split test] [--out datasets/lineage.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from itrea.analysis.stats import Table2x2, mantel_haenszel
from itrea.trace import _clean, read_events

R0 = 0
REQUIRED = ("id", "pa", "pb", "rank_pa", "rank_pb", "birth", "key", "feasible")
KEY_TOL = 1e-9


def load_genealogy(path: str | os.PathLike) -> pd.DataFrame:
    """The edge list as a DataFrame (all columns of the npz; `V` is also exposed as `viol`)."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: np.asarray(z[k]) for k in z.files}
    df = pd.DataFrame(d)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: genealogy lacks columns {missing}")
    if "V" in df.columns and "viol" not in df.columns:
        df["viol"] = df["V"]
    df["feasible"] = df["feasible"].astype(bool)
    return df


def derive_new_best(df: pd.DataFrame) -> np.ndarray:
    """Feasible rows that improved on every earlier feasible key (order: birth, then id)."""
    o = np.lexsort((df["id"].to_numpy(), df["birth"].to_numpy()))
    key = df["key"].to_numpy(dtype=np.float64)[o]
    feas = df["feasible"].to_numpy(dtype=bool)[o]
    out_sorted = np.zeros(len(df), dtype=bool)
    best = math.inf
    for i in range(len(df)):
        if feas[i] and key[i] < best - KEY_TOL * max(1.0, abs(best) if math.isfinite(best) else 1.0):
            out_sorted[i] = True
            best = key[i]
    out = np.zeros(len(df), dtype=bool)
    out[o] = out_sorted
    return out


def derive_accepted_posts(df: pd.DataFrame, accepted_keys: list[float]) -> np.ndarray:
    """For each accepted post key, the earliest feasible row with that key (|Δkey| <= 1e-9 relative)."""
    out = np.zeros(len(df), dtype=bool)
    if not accepted_keys:
        return out
    o = np.lexsort((df["id"].to_numpy(), df["birth"].to_numpy()))
    key = df["key"].to_numpy(dtype=np.float64)
    feas = df["feasible"].to_numpy(dtype=bool)
    for k in accepted_keys:
        tol = KEY_TOL * max(1.0, abs(k))
        hit = [i for i in o if feas[i] and abs(key[i] - k) <= tol]
        if hit:
            out[hit[0]] = True
    return out


def _accepted_post_keys(run_dir: Path, sense: int) -> list[float]:
    """Keys (sense · objective) of EA posts the solver accepted (stored), from the events of the run."""
    posts, accepted = {}, set()
    ea = run_dir / "ea.events.jsonl"
    if ea.exists():
        for e in read_events(ea):
            if e.get("kind") == "post" and e.get("seq") is not None and e.get("obj") is not None:
                posts[int(e["seq"])] = float(e["obj"])
    for f in sorted(run_dir.glob("solver*.events.jsonl")):
        for e in read_events(f):
            if e.get("kind") == "verdict" and bool(e.get("accepted", e.get("stored", False))):
                accepted.add(int(e["seq"]))
    return [sense * posts[s] for s in sorted(accepted) if s in posts]


def mark_cases(df: pd.DataFrame, run_dir: Path | None = None, sense: int = 1) -> tuple[pd.DataFrame, bool]:
    """Adds `case` (new best or accepted post). Returns (df, derived) — derived = flags were not in the file."""
    df = df.copy()
    derived = False
    if "is_new_best" in df.columns:
        nb = df["is_new_best"].to_numpy(dtype=bool)
    else:
        nb, derived = derive_new_best(df), True
    if "is_accepted_post" in df.columns:
        ap = df["is_accepted_post"].to_numpy(dtype=bool)
    else:
        keys = _accepted_post_keys(run_dir, sense) if run_dir is not None else []
        ap, derived = derive_accepted_posts(df, keys), True
    df["case"] = nb | ap
    return df, derived


def generation_tables(df: pd.DataFrame, rank: int = R0, reference: str = "all") -> list[Table2x2]:
    """One 2 × 2 table per generation (birth) over bred children (pa >= 0). df needs `case`."""
    if reference not in ("all", "noncases"):
        raise ValueError("reference must be 'all' or 'noncases'")
    bred = df[df["pa"].to_numpy() >= 0]
    rank_pa = bred["rank_pa"].to_numpy()
    pb = bred["pb"].to_numpy()
    rank_pb = bred["rank_pb"].to_numpy()
    exposed = (rank_pa == rank) | ((pb >= 0) & (rank_pb == rank))
    case = bred["case"].to_numpy(dtype=bool)
    birth = bred["birth"].to_numpy()
    tabs = []
    for g in np.unique(birth):
        m = birth == g
        ex, ca = exposed[m], case[m]
        a, c = int(np.sum(ex & ca)), int(np.sum(~ex & ca))
        if reference == "all":
            b, d = int(np.sum(ex)), int(np.sum(~ex))
        else:
            b, d = int(np.sum(ex & ~ca)), int(np.sum(~ex & ~ca))
        tabs.append(Table2x2(a, b, c, d))
    return tabs


def run_tables(run_dir: str | os.PathLike, rank: int = R0, reference: str = "all") -> dict:
    """Tables and counts of one run directory with an `ea.genealogy.npz`."""
    d = Path(run_dir)
    p = d / "ea.genealogy.npz"
    summ = {}
    try:
        summ = json.loads((d / "ea.summary.json").read_text())
    except (OSError, ValueError):
        pass
    sense = int(summ.get("sense", 1) or 1)
    df = load_genealogy(p)
    df, derived = mark_cases(df, d, sense)
    tabs = generation_tables(df, rank, reference)
    return dict(run_dir=str(d), rows=int(len(df)), bred=int((df["pa"] >= 0).sum()), cases=int(df["case"].sum()),
                cases_without_parents=int((df["case"] & (df["pa"] < 0)).sum()), derived_flags=derived, tables=tabs)


def pooled(run_dirs: list[str | os.PathLike], rank: int = R0, reference: str = "all") -> dict:
    """Mantel–Haenszel over every generation of every run (strata = run × generation)."""
    tabs, runs, derived, no_parent, cases = [], 0, False, 0, 0
    for rd in run_dirs:
        r = run_tables(rd, rank, reference)
        tabs += r["tables"]
        runs += 1
        derived |= r["derived_flags"]
        no_parent += r["cases_without_parents"]
        cases += r["cases"]
    mh = mantel_haenszel(tabs)
    return dict(rank=rank, reference=reference, runs=runs, cases=cases, cases_without_parents=no_parent,
                derived_flags=derived, **mh)


def _run_json(d: Path) -> dict:
    try:
        v = json.loads((d / "run.json").read_text())
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


def find_genealogies(root: str | os.PathLike, arms: list[str] | None = None, sets: list[str] | None = None,
                     splits: list[str] | None = None, dropped: dict | None = None) -> list[Path]:
    """Run directories with an `ea.genealogy.npz`, without the runs the pool moved aside as void (the rule of
    `build_runs.discover`), filtered on run.json's set and split and on the arm (run.json, else the run id); a filter
    that is None or empty accepts everything (a directory without run.json then passes the set/split filters only
    when they are empty). `dropped`, if given, receives the number of directories each filter removed (keys void,
    set, split, arm)."""
    from itrea.analysis.build_runs import is_void_dir, parse_run_id

    root = Path(root)
    found = {p.parent for p in root.rglob("ea.genealogy.npz")} if root.is_dir() else set()
    live = {d for d in found if not is_void_dir(d, root)}
    cnt = dict(void=len(found - live), set=0, split=0, arm=0)
    out = []
    for d in sorted(live):
        rj = _run_json(d)
        if sets and str(rj.get("set") or "") not in sets:
            cnt["set"] += 1
            continue
        if splits and str(rj.get("split") or "") not in splits:
            cnt["split"] += 1
            continue
        arm = rj.get("arm") or parse_run_id(str(rj.get("run_id") or d.name)).get("arm")
        if arms and arm not in arms:
            cnt["arm"] += 1
            continue
        out.append(d)
    if dropped is not None:
        dropped.update(cnt)
    return out


def main(argv: list[str] | None = None) -> int:
    from itrea.paths import DATASETS, TRACES

    ap = argparse.ArgumentParser(description="depth-1 matched odds ratio of a Rank 0 parent (D-002 §4)")
    ap.add_argument("root", nargs="?", default=str(TRACES / "raw"))
    ap.add_argument("--arms", default="A5,A3", help="comma-separated arms (empty = all)")
    ap.add_argument("--sets", default="e2", help="comma-separated run.json sets (empty = all)")
    ap.add_argument("--split", default="test", help="comma-separated run.json splits (empty = all)")
    ap.add_argument("--out", default=str(DATASETS / "lineage.json"))
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    arms = [x for x in a.arms.split(",") if x]
    sets = [x for x in a.sets.split(",") if x]
    splits = [x for x in a.split.split(",") if x]
    res = {}
    for arm in (arms or [None]):
        dropped: dict = {}
        dirs = find_genealogies(a.root, [arm] if arm else None, sets, splits, dropped)
        if not dirs:
            continue
        res[arm or "all"] = {ref: pooled(dirs, R0, ref) for ref in ("all", "noncases")}
        res[arm or "all"]["selection"] = dict(sets=sets, splits=splits, dropped=dropped)
    if not res:
        print(f"no ea.genealogy.npz under {a.root} for arms {arms or 'any'}, sets {sets or 'any'}, splits "
              f"{splits or 'any'}: nothing written")
        return 2
    Path(a.out).write_text(json.dumps(_clean(res), indent=1, sort_keys=True) + "\n")
    print(f"lineage: {sum(v['all']['runs'] for v in res.values())} runs -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
