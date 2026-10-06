---
name: itrea-reproduce
description: >
  Rebuild I-TREA II's derived tables (datasets/) and paper tables (paper/generated/) from the raw traces and decide
  whether any difference is real. Use before a paper build or submission, after any change to the analysis code or
  the checker, after unpacking a traces archive, and when someone asks whether a published number still holds.
---

# Reproducing I-TREA II's numbers

Status: the analysis chain is being built (`src/itrea/analysis/`); this skill is the contract it must meet. Model:
optopt's `optopt reproduce`.

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
Write it in the journal and in `vault/60-incidents/Mistakes and incidents.md`; if a published or quoted number
changes, add a dated correction block to its result note (skill `itrea-result-note`) and to AGENTS.md / the paper,
keeping the old value visible.
