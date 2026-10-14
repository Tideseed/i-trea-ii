"""The shared reading of the C1 diagnosis's targeted development tests D-025 (R1, set `r1-*`) and D-026 (R2, set
`r2-ld`): two A9 arms — a treatment and the same-day development default re-run under the set's own control label —
in one block per instance x seed, 120 s, the ext harness. `tools/d025_report.py` and `tools/d026_report.py` apply
their records' rules on top of this module; nothing here decides a verdict.

What is shared:
  - the selection (`select`): the rows of one set x the two labels x the test's seeds x 120 s, not synthetic, not in a
    voided block, completed (and failed with `keep_failed`); one row per (instance, label, seed): the bare run id before
    a `<run_id>#<set>` copy (build_runs.qualify_reused_ids), then the highest attempt; the duplicates are listed;
  - each run's files (`run_files`): `run.json` and `ea.summary.json` under the row's `run_dir` (relative to `--root`);
  - the validity legs every such record states (`base_problems`, `block_problems`, `completeness`): complete (every
    expected key present; a failed run is "failed (not completed)" under the letter, present under `keep`); harness
    (run.json `config.harness` equals `pair.resolve_harness("ext")`); config (the row's bare run id equals the id of the
    job the writer made — the run hash carries the EA configuration hash, the harness and the footprint); compile (no
    CUDA compile on the clock: run.json `compiled_on_clock_cuda`, the table's `cuda_compiled_on_clock`, the EA
    summary's `compiled_on_clock`); block (equal P and gene dtype in the two runs of a block, from the EA summaries);
  - the one-sided seed-mean band (`band_leg`): P̄(treatment) − P̄(control) <= max(floor, the control's seed spread)
    (two seeds: |P_s0 − P_s1|; more seeds: max − min), in raw P (D-023's correction: a no-harm rule never fails a
    treatment for helping);
  - the ratio of means (`ratio_leg`): ΣP(treatment) / ΣP(control) over the hint cells, per instance beside;
  - JSON output (`jsonable`).

Failed runs: D-025 and D-026 read them as outcomes of their arm (D-002 §6, `--failed keep`, D-022's precedent) and
print the letter's reading (`exclude`: a failed run is not completed, so the complete leg fails) beside; a reading
under `keep` names each kept failure. A failed run's P is the run table's (truncated at the crash, as build-runs scores
it); a figure its EA summary cannot give (no summary) fails every gating leg that needs it, stated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REUSE_SEP = "#"
BASE_LEGS = ("complete", "harness", "config", "compile", "block")


def fin(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def flag(v) -> bool:
    """A true flag; None and NaN (a column the run lacks) are false."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    return bool(v)


def read_json(p: Path) -> dict:
    try:
        d = json.loads(Path(p).read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def ext_harness() -> dict:
    from itrea.experiments.make_jobs import ITX_EXT
    from itrea.experiments.pair import resolve_harness
    return resolve_harness(ITX_EXT)


def bare_id(run_id) -> str:
    return str(run_id).split(REUSE_SEP)[0]


# ----------------------------------------------------------------------------------------------- selection
def candidates(runs: pd.DataFrame, set_: str, labels, seeds, budget: float) -> pd.DataFrame:
    """Every row of `set_` x `labels` x `seeds` at `budget`, not synthetic, not in a voided block, whatever its
    status."""
    df = runs[(runs["set"] == set_) & runs["arm"].isin(list(labels)) & runs["seed"].isin(list(seeds))]
    if "budget" in df:
        df = df[np.isclose(df["budget"].astype(float), float(budget))]
    for col in ("synthetic", "block_void"):
        if col in df:
            df = df[~df[col].fillna(False).astype(bool)]
    return df


def select(runs: pd.DataFrame, set_: str, labels, seeds, budget: float,
           keep_failed: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """One row per (instance, label, seed) of `candidates`, completed (and failed with `keep_failed`): the bare run id
    before a `#<set>` copy, then the highest attempt. Returns (rows, the duplicates left out)."""
    df = candidates(runs, set_, labels, seeds, budget)
    df = df[df["status"].isin({"completed", "failed"} if keep_failed else {"completed"})].copy()
    df["_reused"] = df["run_id"].astype(str).str.contains(REUSE_SEP, regex=False)
    df["_attempt"] = df["attempt"].fillna(0).astype(int) if "attempt" in df else 0
    df = df.sort_values(["instance", "arm", "seed", "_reused", "_attempt"], ascending=[True, True, True, True, False],
                        kind="mergesort")
    dup = df.duplicated(["instance", "arm", "seed"], keep="first")
    dropped = [dict(instance=r["instance"], arm=r["arm"], seed=int(r["seed"]), run_id=r["run_id"])
               for r in df[dup].to_dict("records")]
    return df[~dup].drop(columns=["_reused", "_attempt"]), dropped


def failed_keys(runs: pd.DataFrame, set_: str, labels, seeds, budget: float) -> set[tuple[str, str, int]]:
    cand = candidates(runs, set_, labels, seeds, budget)
    return {(r["instance"], r["arm"], int(r["seed"])) for r in cand[cand["status"] == "failed"].to_dict("records")}


def others(runs: pd.DataFrame, set_: str, labels, seeds, budget: float) -> list[str]:
    """Rows of the set that no rule reads (other labels, seeds, budgets, or a voided block)."""
    keep = set(candidates(runs, set_, labels, seeds, budget)["run_id"])
    return sorted(str(r) for r in runs[(runs["set"] == set_) & ~runs["run_id"].isin(keep)]["run_id"])


# ----------------------------------------------------------------------------------------------- run files
def run_dir(row: dict, root: Path) -> Path:
    p = Path(str(row.get("run_dir") or ""))
    return p if p.is_absolute() else Path(root) / p


def run_files(row: dict, root: Path) -> tuple[dict, dict]:
    """(run.json, ea.summary.json) of one run; {} for a file that is missing or unreadable."""
    d = run_dir(row, root)
    return read_json(d / "run.json"), read_json(d / "ea.summary.json")


# ----------------------------------------------------------------------------------------------- validity
def base_problems(row: dict, rj: dict, summ: dict, expected_id: str | None, ext: dict) -> dict[str, list[str]]:
    """{leg: problems} of one run for the legs harness, config, compile (empty = valid)."""
    out: dict[str, list[str]] = {}
    h = (rj.get("config") or {}).get("harness") if isinstance(rj.get("config"), dict) else None
    if h is None and isinstance(row.get("harness"), str) and row["harness"]:
        try:
            h = json.loads(row["harness"])
        except ValueError:
            h = None
    if h != ext:
        out["harness"] = [f"harness {h!r} is not ext"]
    if expected_id is not None and bare_id(row["run_id"]) != expected_id:
        out["config"] = [f"run id {bare_id(row['run_id'])} is not the job's {expected_id} (configuration, harness or "
                         f"footprint differ from the job file)"]
    if flag(rj.get("compiled_on_clock_cuda")) or flag(row.get("cuda_compiled_on_clock")) \
            or flag(summ.get("compiled_on_clock")):
        out["compile"] = ["compiled on the clock"]
    return out


def block_problems(cells: dict, blocks) -> list[str]:
    """Equal P and gene dtype in the two runs of each block (instance, seed) — from the EA summaries."""
    out = []
    for inst, seed, keys in blocks:
        have = [cells[k] for k in keys if k in cells]
        if len(have) < 2:
            continue
        missing = [c for c in have if not c["summ"]]
        if missing:                                  # a failed run without a summary is an outcome, not a problem
            if any(c["row"].get("status") != "failed" for c in missing):
                out.append(f"{inst} s{seed}: an EA summary is missing (P / gene dtype unread)")
            continue
        ps = {c["summ"].get("P") for c in have}
        dts = {c["summ"].get("gene_dtype") for c in have}
        if None in ps or None in dts:
            out.append(f"{inst} s{seed}: P / gene dtype missing from an EA summary")
        elif len(ps) > 1 or len(dts) > 1:
            out.append(f"{inst} s{seed}: P {sorted(ps)} / gene dtype {sorted(map(str, dts))} differ between the arms")
    return out


def completeness(expected, by: dict, failed: set) -> tuple[list[str], list[tuple]]:
    """The complete leg: the problems (a missing key, a failed run under the letter) and the keys present."""
    probs, present = [], []
    for k in expected:
        if k in by:
            present.append(k)
        else:
            probs.append(f"{'failed (not completed)' if k in failed else 'missing'} {k[0]} {k[1]} s{k[2]}")
    return probs, present


# ----------------------------------------------------------------------------------------------- legs
def spread(v: list[float]) -> float:
    """A control's seed spread in raw P: |P_s0 − P_s1| for two seeds, max − min for more."""
    return float(max(v) - min(v)) if v else math.nan


def band_leg(t: list, c: list, floor: float) -> dict:
    """One-sided no-harm on seed means: ok iff P̄(t) − P̄(c) <= max(floor, the control's seed spread)."""
    if not t or not c or not all(fin(x) for x in t + c):
        return dict(t=t, c=c, ok=False, why="a P value is missing", diff=None, band=None)
    diff = float(np.mean(t) - np.mean(c))
    band = max(float(floor), spread([float(x) for x in c]))
    return dict(t=t, c=c, mean_t=float(np.mean(t)), mean_c=float(np.mean(c)), diff=diff, band=band,
                ok=bool(diff <= band + 1e-12))


def ratio_leg(t_by_inst: dict, c_by_inst: dict, threshold: float) -> dict:
    """The hint: ΣP(t) / ΣP(c) over every cell <= threshold (futility-only), the per-instance ratio beside."""
    t_all = [x for v in t_by_inst.values() for x in v]
    c_all = [x for v in c_by_inst.values() for x in v]
    if not t_all or not all(fin(x) for x in t_all + c_all) or len(t_all) != len(c_all):
        return dict(ratio=None, present=False, why="a P value is missing", per={})
    st, sc = float(np.sum(t_all)), float(np.sum(c_all))
    ratio = st / sc if sc > 0 else math.inf
    per = {i: dict(t=t_by_inst[i], c=c_by_inst[i],
                   ratio=(float(np.sum(t_by_inst[i])) / float(np.sum(c_by_inst[i]))
                          if float(np.sum(c_by_inst[i])) > 0 else math.inf)) for i in t_by_inst}
    return dict(sum_t=st, sum_c=sc, ratio=ratio, threshold=float(threshold), present=bool(ratio <= threshold + 1e-12),
                per=per)


def dig(d: dict, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def f(v, nd=3) -> str:
    return "inf" if v == math.inf else (f"{float(v):.{nd}f}" if fin(v) else "–")


def jsonable(o):
    if isinstance(o, dict):
        return {(" | ".join(map(str, k)) if isinstance(k, tuple) else str(k)): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None if math.isnan(o) else ("inf" if o > 0 else "-inf")
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return jsonable(o.item())
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)
