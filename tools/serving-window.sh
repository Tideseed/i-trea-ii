#!/usr/bin/env bash
# Stop / restore the two local model servers for an I-TREA II experiment window (Berk's yes: 2026-10-05 23:2x).
#   serving-window.sh schedule 'YYYY-MM-DD HH:MM:00'
#                               create the restore timer (transient itrea-restore-serving.timer + .service) with
#                               OnFailure=agent-failure-alert@itrea-restore-serving.service.service, so a failed
#                               restore is announced in [ops channel]. Schedule it BEFORE `stop`.
#   serving-window.sh stop      pause the canaries, add hold-units lines, stop ornith-vllm + llama-qwen38
#   serving-window.sh restore   renew the box-window lease for the model load (now + 45 min), wait up to 7 min for the
#                               runner lock (a run set still in flight), start both units, wait up to 15 min for
#                               health, resume the canaries, drop our hold-units lines, release the lease (if ours).
#                               Exits 1 when the servers are not healthy (the remaining steps still run first).
# Log: ~/.local/state/itrea/serving-window.log.
# Env: ITREA_HOLD_UNTIL (ISO time; default the next 07:45). For tests (tests/test_review_serving_window.py):
#      ITREA_RUNNER_LOCK, ITREA_LOCK_WAIT_S (420), ITREA_HEALTH_TRIES (90), ITREA_HEALTH_SLEEP (10 s).
set -u
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"   # a transient timer must find hermes and box-window
LOG=~/.local/state/itrea/serving-window.log
HOLD=~/.config/cronrestore/hold-units
WHO="i-trea-ii (Claude Code)"
UNITS="ornith-vllm llama-qwen38"
CANARIES="f663edfdcbff 36f278b0ee63"   # local-model-canary, canary-hello
RESTORE_UNIT=itrea-restore-serving     # transient timer + service made by `schedule`
RUNNER_LOCK="${ITREA_RUNNER_LOCK:-/tmp/itrea-runner.lock}"   # gates.RunLock: held for a whole run set
LOCK_WAIT_S="${ITREA_LOCK_WAIT_S:-420}"
HEALTH_TRIES="${ITREA_HEALTH_TRIES:-90}"
HEALTH_SLEEP="${ITREA_HEALTH_SLEEP:-10}"
# The next 07:45: today while it is still ahead, else tomorrow (`tomorrow 07:45` run after midnight held ~31 h).
UNTIL="${ITREA_HOLD_UNTIL:-$(t=$(date -d 'today 07:45' +%s); if [ "$t" -gt "$(date +%s)" ]; then date -d "@$t" --iso-8601=seconds; else date -d 'tomorrow 07:45' --iso-8601=seconds; fi)}"
mkdir -p "$(dirname "$LOG")"
log() { echo "$(date --iso-8601=seconds) $*" | tee -a "$LOG"; }
case "${1:-}" in
  schedule)
    [ -n "${2:-}" ] || { echo "usage: $0 schedule 'YYYY-MM-DD HH:MM:00'"; exit 2; }
    # Literal instance name (= what %n gives a unit file); %n is not relied on through systemd-run's transient path.
    if systemd-run --user --unit "$RESTORE_UNIT" --on-calendar "$2" \
         --description "I-TREA II: restore ornith-vllm + llama-qwen38 after the experiment window" \
         -p "OnFailure=agent-failure-alert@$RESTORE_UNIT.service.service" \
         "$(readlink -f "$0")" restore; then
      log "scheduled restore: $RESTORE_UNIT.timer at $2 (OnFailure alert)"
    else
      log "schedule FAILED for $2"; exit 1
    fi
    ;;
  stop)
    for j in $CANARIES; do hermes cron pause "$j" >/dev/null 2>&1 && log "paused cron $j"; done
    mkdir -p "$(dirname "$HOLD")"; touch "$HOLD"
    for u in $UNITS; do grep -q "^$u until" "$HOLD" || echo "$u until $UNTIL  # i-trea-ii window" >> "$HOLD"; done
    log "hold-units: $UNITS until $UNTIL"
    for u in $UNITS; do systemctl --user stop "$u" && log "stopped $u"; done
    ;;
  restore)
    # The window's lease may end the very minute this runs; hold the box through the lock wait and the model load
    # (worst case 7 + 30 min), so no other session claims it while MemAvailable still looks high.
    if box-window claim --who "$WHO" --why "restore ornith-vllm + llama-qwen38 after the i-trea-ii window" \
         --until "$(date -d '+45 minutes' --iso-8601=seconds)" >>"$LOG" 2>&1; then
      log "lease renewed for the restore"
    else
      log "lease claim refused (another session holds the box); restoring anyway"
    fi
    # Do not load the models beside a run set that is still going (vLLM alone takes ~51 GiB).
    if flock -w "$LOCK_WAIT_S" "$RUNNER_LOCK" true; then
      log "runner lock free"
    else
      log "runner lock $RUNNER_LOCK still held after ${LOCK_WAIT_S}s; starting the servers anyway"
    fi
    for u in $UNITS; do systemctl --user start "$u" && log "started $u"; done
    ok=0; a=0; b=0
    for i in $(seq 1 "$HEALTH_TRIES"); do   # default up to 15 min for both to load
      a=$(curl -sf --max-time 5 http://127.0.0.1:8094/v1/models >/dev/null && echo 1 || echo 0)
      b=$(curl -sf --max-time 5 http://127.0.0.1:8092/health >/dev/null && echo 1 || echo 0)
      if [ "$a$b" = "11" ]; then ok=1; break; fi
      sleep "$HEALTH_SLEEP"
    done
    log "health after start: ornith=$a qwen38=$b (ok=$ok)"
    [ "$ok" = 1 ] || log "RESTORE FAILED: ornith=$a qwen38=$b (not healthy after the wait); this run exits 1"
    for j in $CANARIES; do hermes cron resume "$j" >/dev/null 2>&1 && log "resumed cron $j"; done
    if [ -f "$HOLD" ]; then sed -i '/# i-trea-ii window$/d' "$HOLD"; log "hold-units lines removed"; fi
    box-window release --who "$WHO" >>"$LOG" 2>&1 || true
    for u in $UNITS; do log "$u is-active=$(systemctl --user is-active $u) is-enabled=$(systemctl --user is-enabled $u)"; done
    [ "$ok" = 1 ] || exit 1
    ;;
  *) echo "usage: $0 schedule 'YYYY-MM-DD HH:MM:00' | stop | restore"; exit 2 ;;
esac
