---
name: itrea-reproduce
description: >
  Rebuild I-TREA II's derived tables (datasets/) and paper tables (paper/generated/) from the raw traces and decide
  whether any difference is real. Use before a paper build or submission, after any change to the analysis code or
  the checker, after unpacking a traces archive, and when someone asks whether a published number still holds.
---

# Reproducing I-TREA II's numbers

Status: the chain is built — `itrea reproduce [--recheck] [--restore]` (`src/itrea/analysis/reproduce.py`) runs
`build_runs` → `hardset select` → `lineage` → `prereg`, each step only if its outputs exist, and compares every file
under `datasets/` and `paper/generated/` with git HEAD (or, for an untracked file, with the copy before the run); exit
0 = identical, 1 = a step failed or a file changed or appeared, 2 = no traces under `traces/raw/`. On 2026-10-10
`datasets/runs.parquet` and `incumbents_checked.parquet` are untracked, and `hard_pool.csv` / `reference.csv` do not
exist yet (#1, #2). Model: optopt's `optopt reproduce`.

**Old commit hashes.** The history was rewritten on 2026-10-17 (vault `00-index/History rewrite 2026-10-17.md`), so
a hash recorded before then (a run record's `env.git.revision` / `prereg_tag`, a note, an issue) names an old commit.
`build_runs` and `prereg` translate through `COMMIT_MAP.tsv` on their own; by hand,
`python -m itrea.analysis.commit_map <hash> [<hash> …]`. `python -m itrea.analysis.commit_map --check-traces` checks
that every recorded revision translates to a commit that descends from `prereg-v1` (exit 0 = all pass).

Development tests (D-010) are outside that compare set: re-derive them with `itrea build-runs`, then the test's
report (`tools/d016_report.py`, `tools/d017_report.py`, `itrea backoff-report --stage 0|1 [--json out.json]`),
single-threaded, and compare with the decision record's Result section and the Development log.

## Two different things
- **Re-analysis** — the same traces, re-derived. Expected: every committed table reproduced **exactly** (text byte
  for byte, parquet by content). Any difference is a defect in the analysis, the checker or the commit, and is
  investigated before anything else is done.
- **Replication** — the jobs re-run (skill `itrea-new-experiment`). Expected: the same conclusions, not the same
  numbers: P(T), time to target and throughput are wall-clock measurements on one DGX Spark, and GPU arithmetic is not
  bit-reproducible (atomic scatter order). Compare verdicts and intervals, not digits.

## Re-analysis
1. Inputs: raw traces in `traces/raw/<set>/` (from the runs or unpacked from the versioned archive, checksums verified),
   the instances in `data/` (for the checker), the frozen `datasets/hard_pool.csv` and `datasets/reference.csv`.
2. Run single-threaded (`OMP_NUM_THREADS=1`), never beside timed runs: the exact checker over **every** stored
   incumbent, then metrics, then the tests and tables, then `paper/generated/`.
3. Compare every output with its committed version and list each changed file. Exit non-zero on any change or any
   failed step.

## What counts as a difference
- **Real (must be explained):** a changed verdict, a changed n, a checker result that flips, a number outside the last
  printed digit, a missing or extra run, a run whose recorded backend or config hash no longer matches its arm.
- **Not a difference in a re-analysis:** nothing — re-analysis must be exact. Floating-point summation order in the
  analysis is fixed (sorted inputs, single thread) precisely so this holds.
- **Expected in a replication:** different per-run values; the comparison is whether each pre-registered verdict and
  its interval land the same way, reported per hypothesis.

## When something differs
Write what differs, and by how much, in the journal, and log it in the incident log of the private operations
repository (skill `itrea-journal`, "What goes where"); if a published or quoted number
changes, add a dated correction block to its result note (skill `itrea-result-note`) and to AGENTS.md / the paper,
keeping the old value visible.
