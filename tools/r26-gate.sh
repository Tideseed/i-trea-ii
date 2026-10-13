#!/usr/bin/env bash
# D-021 R26 opening gate for an ITX window (added 2026-10-13 after W-1013b opened warm): wait IDLE_S (180) after the LLM
# servers stopped, then until the median hottest zone over the last 60 s of the 5-s fine log is <= LIMIT_C (50) and the fan
# governor is alive (fan-state written within 30 s, cap=0). Exits 0 when open, 1 at the deadline (HH:MM, today).
#   r26-gate.sh [--deadline HH:MM] [--idle-s N] [--limit-c C]
# Log lines go to stdout (the chain's .out). Env for tests: R26_THERMAL_DIR (default ~/.local/state/thermal).
set -u
deadline="" idle=180 limit=50
while [ $# -gt 0 ]; do
  case "$1" in
    --deadline) deadline="$2"; shift 2 ;;
    --idle-s) idle="$2"; shift 2 ;;
    --limit-c) limit="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
dir="${R26_THERMAL_DIR:-$HOME/.local/state/thermal}"
dl_epoch=""
[ -n "$deadline" ] && dl_epoch=$(date -d "today $deadline" +%s)
echo "r26: idle ${idle} s from $(date +%T), then median hottest zone (60 s) <= ${limit} C and governor alive"
sleep "$idle"
while :; do
  fine="$dir/fine-$(date +%F).log"
  med=$(tail -n 12 "$fine" 2>/dev/null | python3 -c '
import re, sys, statistics
v = [max(int(x) for x in re.findall(r"\bT[A-Z0-9]+=(\d+)", l)) for l in sys.stdin if re.search(r"\bT[A-Z0-9]+=\d+", l)]
print(statistics.median(v) if len(v) >= 10 else "")')
  st="$dir/fan-state"
  alive=0
  if [ -f "$st" ] && [ $(( $(date +%s) - $(stat -c %Y "$st") )) -le 30 ] && grep -q "cap=0" "$st"; then alive=1; fi
  if [ -n "$med" ] && python3 -c "import sys; sys.exit(0 if float('$med') <= $limit else 1)" && [ "$alive" = 1 ]; then
    echo "r26: OPEN $(date +%T) median ${med} C, governor alive"; exit 0
  fi
  if [ -n "$dl_epoch" ] && [ "$(date +%s)" -ge "$dl_epoch" ]; then
    echo "r26: NOT OPEN by $deadline (median ${med:-?} C, governor alive=$alive)"; exit 1
  fi
  sleep 15
done
