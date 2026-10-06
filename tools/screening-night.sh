#!/usr/bin/env bash
# The SCIP screening pass of D-002 v3.1 §1.2, chained: stage 1 (seed 900, all 324 candidates) → run table (every
# incumbent checked) → stage 2 (seed 901 per the rule) — all under one deadline. Write once, never edit while running.
#   tools/screening-night.sh HH:MM        (deadline for new blocks)
set -u
cd "$(dirname "$0")/.."
DEADLINE="${1:?deadline HH:MM}"
LOG=logs/screening-night.log
PY=.venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
log() { echo "$(date --iso-8601=seconds) $*" | tee -a "$LOG"; }
log "stage 1 start (deadline $DEADLINE), HEAD $(git rev-parse --short HEAD)"
$PY -m itrea.experiments.pool jobs/screen1.jsonl --layout screen --cpu-lanes 9 --deadline "$DEADLINE" >> "$LOG" 2>&1
rc=$?; log "stage 1 pool exit $rc"
[ "$rc" = 0 ] || { log "stage 1 did not finish cleanly — stopping the chain"; exit "$rc"; }
log "build_runs (stage 1)"
$PY -m itrea.analysis.build_runs --workers 8 >> "$LOG" 2>&1 || { log "build_runs failed — stopping"; exit 1; }
$PY -m itrea.experiments.make_jobs screen2 --runs datasets/runs.parquet --out jobs/screen2.jsonl >> "$LOG" 2>&1 \
  || { log "screen2 job builder failed — stopping"; exit 1; }
log "stage 2: $(wc -l < jobs/screen2.jsonl) jobs"
$PY -m itrea.experiments.pool jobs/screen2.jsonl --layout screen --cpu-lanes 9 --deadline "$DEADLINE" >> "$LOG" 2>&1
rc=$?; log "stage 2 pool exit $rc"
$PY -m itrea.analysis.build_runs --workers 8 >> "$LOG" 2>&1 && log "build_runs (stages 1+2) done" || log "build_runs failed"
log "chain end"
