---
name: itrea-window
description: >
  Open, run and close an I-TREA II experiment window on the shared DGX Spark: claim the box-window lease, stop and
  restore the two local LLM servers with tools/serving-window.sh, pass the memory / thermal / GPU-health gates, and
  verify the box afterwards. Use before ANY timed run (screening, calibration, E1–E5), any GPU benchmark, or any batch
  needing more than ~10 GB, and when a window ends or is abandoned.
---

# I-TREA II experiment window

Rules: `vault/50-decisions/D-003 Resources, memory and thermal rules.md`. Code: `src/itrea/experiments/gates.py`,
`gpu_health.py`. The box is shared with other Claude Code sessions, the host's cron jobs and two local LLM servers;
Berk usually does not coordinate the sessions — the lease does.

## Before
1. `box-window status` — if another session holds an unexpired lease, **wait or ask it**; never `--force`.
2. Look at what is running: `systemctl --user is-active ornith-vllm llama-qwen38`, `nvidia-smi` (foreign GPU
   processes), MemAvailable, the ACPI maximum (`gates.acpi_max_c()`); a GPU arm must not start above 80 °C.
3. Claim, with the exact `who` string the runner checks:
   ```bash
   box-window claim --who "i-trea-ii (Claude Code)" --why "<experiment id and what runs>" --until HH:MM \
     [--stops ornith-vllm,llama-qwen38] [--mem <GB>]
   ```
   `--until` covers the whole run set plus the restore (restore time + 20 min); a block is never split across windows.
4. Confirmatory runs (and the screening pass) need **both LLM servers stopped** (vLLM serving Ornith-1.5-35B-A3B on
   :8094, llama.cpp serving Qwen3.8-27B on :8092; Berk's yes of 2026-10-05). Schedule the restore **first**, then stop:
   ```bash
   ~/git_repositories/i-trea-ii/tools/serving-window.sh schedule "YYYY-MM-DD HH:MM:00"
   # creates itrea-restore-serving.timer with OnFailure=agent-failure-alert@… so a failed restore alerts
   ITREA_HOLD_UNTIL=<ISO time ≥ the lease end> tools/serving-window.sh stop
   ```
   `stop` pauses the two canaries, adds `hold-units` lines (so the host's experiment-guard does not restart the units
   inside the window) and stops both units. Log: `~/.local/state/itrea/serving-window.log`. A transient timer does not
   survive a reboot; the host's experiment-guard (07:00, 12:00) is the backstop, not the plan.
5. Exchange platform (if the run set uses Redis): `python -m itrea.experiments.exchange_ctl up` (only the container
   `itrea-redis`; never the host's other Redis instances).
6. Journal line (skill `itrea-journal`): window id, lease, what was stopped, MemAvailable before and after.

## During
- The runner applies the gates on every run: memory budget (Σ MemoryMax + Σ CuPy pool limits + 1 GB per CUDA process +
  cuOpt RMM limit + Redis 2 GB + 20 GB reserve ≤ MemAvailable), thermal start < 80 °C, pause ≥ 92 °C / resume < 80 °C,
  watchdog floor 18 GB, one GPU arm at a time, one runner (`/tmp/itrea-runner.lock`).
- A GPU-health fault (median SM clock < 2,000 MHz, or foreign GPU activity > 2 % of a run) voids the whole block and
  halts the run set: look before resuming. The stuck-low-clock fault has no remote fix (full power disconnect).
- Do not poll processes by name from a Claude Code shell (`pgrep -f` matches its own wrapper); wait on output files or
  log lines. Never run analysis with default thread counts beside timed runs.
- Stop at the first anomaly (crash pattern, memory below the floor, unexpected temperatures) and write it down.

## After (also when abandoning a window early)
1. `python -m itrea.experiments.exchange_ctl flush <run_id>` for any run whose keys remain; `down` if nothing else needs it.
2. Restore — run it yourself when the work ends; do not wait for the timer:
   `tools/serving-window.sh restore` (starts both units, waits up to 15 min for health, resumes the canaries, removes
   the `# i-trea-ii window` hold lines, releases the lease). Then stop the now-unneeded timer:
   `systemctl --user stop itrea-restore-serving.timer`.
3. Verify, and write the result in the journal:
   `systemctl --user is-active ornith-vllm llama-qwen38` **and** `systemctl --user is-enabled ornith-vllm llama-qwen38`
   (active-but-disabled would not survive a reboot), `curl -sf 127.0.0.1:8094/v1/models`, `curl -sf
   127.0.0.1:8092/health`, `hermes cron list` (canaries resumed), `box-window status` (lease released),
   `grep 'i-trea-ii window' ~/.config/cronrestore/hold-units` (nothing).
4. Fill the window row in `vault/10-design/Compute resources.md` (runs, MemAvailable minimum, ACPI maximum, notes).
