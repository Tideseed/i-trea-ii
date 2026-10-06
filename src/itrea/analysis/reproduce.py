"""Re-run the analysis chain and compare every derived file with its committed version.

    python -m itrea.analysis.reproduce [--recheck] [--restore]

Pattern adapted from berkorbay/optopt@4dd48395, MIT (`src/optopt/analysis/reproduce.py`). Every number in the paper
comes from this chain, so a reader (or the next session) can check that the committed tables follow from the traces:

    1. itrea.analysis.build_runs     -> datasets/runs.parquet, datasets/incumbents_checked.parquet
    2. itrea.benchmarks.hardset select -> datasets/hard_pool.csv, datasets/hard_pool_candidates.csv
    3. itrea.analysis.lineage        -> datasets/lineage.json
    4. itrea.analysis.prereg         -> datasets/prereg_results.json, paper/generated/prereg_results.md

A step whose outputs do not exist yet is skipped (there is nothing committed to reproduce). Each step runs as its own
process with OMP_NUM_THREADS=1 (D-003). The exact solution checks are taken from the cached
`incumbents_checked.parquet` (keyed by the sha256 of every incumbents file and the checker version), so the default
run needs neither the instances nor the time — unless the checker changed (`check_solutions.CHECKER_VERSION`): such
rows are re-checked, or turn "unchecked: … stale" without the instances, and the table then differs. The versions in
the cache are printed. `--recheck` re-reads every model and re-checks every incumbent. prereg is re-run in the mode
the committed JSON records, without the git ancestry check (an archive may have no git history).

Comparison: every file under `datasets/` and `paper/generated/` against its version at git HEAD when git tracks it,
else against the file as it was before the run — text byte for byte, parquet by content (columns, dtypes, values).
The regenerated files stay in the working tree (`git diff` shows the change); the previous versions of changed files
are copied to a backup directory that is printed (`--restore` puts them back). Exit 0 = identical, 1 = a step failed
or a file changed or appeared, 2 = no run directories under traces/raw (the raw traces are not committed: unpack the
release archive or re-run the jobs).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

from itrea.paths import ROOT

WATCH = ("datasets", "paper/generated")
STEPS = (  # module, argv, outputs (repository-relative) — skipped if none of the outputs exists
    ("itrea.analysis.build_runs", [], ["datasets/runs.parquet", "datasets/incumbents_checked.parquet"]),
    ("itrea.benchmarks.hardset", ["select"], ["datasets/hard_pool.csv"]),
    ("itrea.analysis.lineage", [], ["datasets/lineage.json"]),
    ("itrea.analysis.prereg", ["--no-git-check"], ["datasets/prereg_results.json"]),
)


def parquet_digest(data: bytes) -> str:
    from itrea.analysis.build_runs import frame_digest

    return frame_digest(pd.read_parquet(io.BytesIO(data)))


def digest(rel: str, data: bytes) -> str:
    return parquet_digest(data) if rel.endswith(".parquet") else hashlib.sha256(data).hexdigest()


def _git_head(rel: str, root: Path) -> bytes | None:
    try:
        r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=root, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def watched_files(root: Path) -> dict[str, bytes]:
    out = {}
    for d in WATCH:
        base = root / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if f.is_file() and not f.name.endswith(".tmp"):
                out[str(f.relative_to(root))] = f.read_bytes()
    return out


def baseline(root: Path, before: dict[str, bytes]) -> dict[str, tuple[str, bytes]]:
    """(source, content) per watched file: git HEAD when tracked, else the pre-run copy."""
    base = {}
    for rel, data in before.items():
        head = _git_head(rel, root)
        base[rel] = ("HEAD", head) if head is not None else ("pre-run", data)
    return base


def has_traces(root: Path) -> bool:
    raw = root / "traces" / "raw"
    return raw.is_dir() and any(raw.rglob("*.events.jsonl"))


def step_argv(mod: str, argv: list[str], root: Path, recheck: bool) -> list[str]:
    argv = list(argv)
    if mod.endswith("build_runs") and recheck:
        argv.append("--recheck")
    if mod.endswith("prereg"):
        try:
            if json.loads((root / "datasets/prereg_results.json").read_text()).get("mode") == "exploratory":
                argv.append("--exploratory")
        except (OSError, ValueError):
            pass
    return argv


def checker_versions(root: Path) -> dict[str, int]:
    """Checker version -> number of incumbent rows in the cached check table (empty if there is none)."""
    p = root / "datasets" / "incumbents_checked.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    if "checker_version" not in df.columns:
        return {"(none)": int(len(df))}
    return {str(k or "(none)"): int(v) for k, v in sorted(df["checker_version"].fillna("").value_counts().items())}


def _print_checker_versions(root: Path) -> None:
    from itrea.analysis.check_solutions import CHECKER_VERSION

    v = checker_versions(root)
    if v:
        print(f"incumbent checks by checker version (current {CHECKER_VERSION}): "
              + ", ".join(f"{k}: {n}" for k, n in v.items()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="re-derive datasets/ and paper/generated/ and compare")
    ap.add_argument("--recheck", action="store_true", help="re-check every incumbent (needs the instance files)")
    ap.add_argument("--restore", action="store_true", help="put the previous versions of changed files back")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    root = ROOT
    if not has_traces(root):
        print("No run directories in traces/raw/. Raw traces are not committed: unpack the release archive of the\n"
              "traces into the workspace, or re-run the jobs (see README, Reproducibility).")
        return 2
    before = watched_files(root)
    base = baseline(root, before)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    failed = []
    for mod, args, outputs in STEPS:
        if not any((root / o).exists() for o in outputs):
            print(f"{mod}: skipped (no committed output: {', '.join(outputs)})")
            continue
        cmd = [sys.executable, "-m", mod] + step_argv(mod, args, root, a.recheck)
        t = time.time()
        r = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
        status = "ok" if r.returncode == 0 else f"exit {r.returncode}"
        print(f"{' '.join(cmd[2:])}: {status} ({time.time() - t:.0f} s)", flush=True)
        if r.returncode != 0:
            failed.append(" ".join(cmd[2:]))
            print((r.stdout + r.stderr)[-2000:])
    _print_checker_versions(root)
    after = watched_files(root)
    changed, new = [], sorted(set(after) - set(before))
    for rel, (src, data) in sorted(base.items()):
        if rel not in after:
            changed.append((rel, src, "deleted"))
        elif digest(rel, data) != digest(rel, after[rel]):
            changed.append((rel, src, "changed"))
    print(f"\n{len(base)} derived files checked; changed {len(changed)}, new {len(new)}")
    for rel, src, what in changed:
        print(f"  {what} (vs {src}): {rel}")
    for rel in new:
        print(f"  new: {rel}")
    for f in failed:
        print(f"  FAILED step: {f}")
    if changed:
        bdir = Path(tempfile.mkdtemp(prefix="itrea-reproduce-"))
        for rel, _, _ in changed:
            dst = bdir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(before[rel])
            if a.restore:
                shutil.copyfile(dst, root / rel)
        print(f"previous versions of the changed files: {bdir}" + (" (restored)" if a.restore else ""))
    return 1 if (changed or new or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
