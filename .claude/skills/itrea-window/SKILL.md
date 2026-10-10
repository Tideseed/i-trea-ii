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
   # Before claiming: daytime windows only — no I-TREA II window in the night rest (00:00–07:00): Berk, 2026-10-10,
   # "No nights" (D-019; until then D-009 asked for his yes for each night). Long run sets (MIPfeas ≈ 78 h) are split
   # into daytime windows, with the fan governor's rests. Check `tail ~/.local/state/thermal/fan-governor.log`
   # and `ls ~/.local/state/thermal/rest`: the runner waits while the rest flag exists; plan ~30 min work / 10–30 min
   # rest cycles.
   box-window claim --who "i-trea-ii (Claude Code)" --why "<experiment id and what runs>" --until HH:MM \
     [--stops ornith-vllm,llama-qwen38] [--mem <GB>]
   ```
   `--until` covers the whole run set plus the restore (restore time + 20 min); a block is never split across windows.
   `--mem` = the memory gate's requirement, which the pool's `--dry-run` prints (`memory: … need up to N GB`): 48 GB
   for an A3 pool (host caps 15 + GPU caps 12 + 1 CUDA × 1 + reserve 20), 73 GB for an A9 pool (27 + 24 + 2 CUDA × 1
   + 20), as claimed on 2026-10-08/09.
4. Confirmatory runs (and the screening pass) need **both LLM servers stopped** (vLLM serving Ornith-1.5-35B-A3B on
   :8094, llama.cpp serving Qwen3.8-27B on :8092; Berk's yes of 2026-10-05). Schedule the restore **first**, then stop:
   ```bash
   ~/git_repositories/i-trea-ii/tools/serving-window.sh schedule "YYYY-MM-DD HH:MM:00"
   # creates itrea-restore-serving.timer with OnFailure=agent-failure-alert@… so a failed restore alerts
   ITREA_HOLD_UNTIL=<ISO time ≥ the lease end> tools/serving-window.sh stop
   ```
   `stop` pauses the two canaries, adds `hold-units` lines (so the host's experiment-guard does not restart the units
   inside the window) and stops both units. Log: `~/.local/state/itrea/serving-window.log`. A transient timer does not
   survive a reboot; the host's experiment-guard (07:00, 12:00) is the backstop, not the plan. Always set
   `ITREA_HOLD_UNTIL`: the script's default is the next 07:45 (a night-window default).
   Development windows have also stopped both servers since 2026-10-07. The D-010 variant — `ITREA_UNITS=llama-qwen38`
   (the llama.cpp server only; §7) with the pool's `--allow-serving` (its 08:2x addendum) — is for development runs
   that fit beside the vLLM server `ornith-vllm`, never for confirmatory or screening runs.
5. GPU tests (they use the GPU, so only inside the window, before the timed pool): after any change to GPU code run
   `pytest -m gpu` on the touched modules, including the cold-cache tests (`test_cupy_cold_cache_*` in
   `tests/test_ea_run.py`, `test_elite.py`, `test_lp_decode.py`), then a short smoke run whose summary says
   `compiled_on_clock = false` (skill `itrea-new-experiment` §4).
6. Exchange platform (if the run set uses Redis): `python -m itrea.experiments.exchange_ctl up` (only the container
   `itrea-redis`; never the host's other Redis instances).
7. Journal line (skill `itrea-journal`): window id, lease, what was stopped, MemAvailable before and after.

## During
- The runner applies the gates on every run: memory budget (Σ MemoryMax + Σ CuPy pool limits + 1 GB per CUDA process +
  cuOpt RMM limit + Redis 2 GB + 20 GB reserve ≤ MemAvailable), thermal start < 80 °C for GPU arms and < 88 °C for
  CPU-only arms (D-008), pause ≥ 92 °C / resume < 80 °C, fan-governor's rest flag (D-009), watchdog floor 18 GB, one
  GPU arm at a time, one runner (`/tmp/itrea-runner.lock`).
- A GPU-health fault (median SM clock < 2,000 MHz, or foreign GPU activity > 2 % of a run) voids the whole block and
  halts the run set: look before resuming. The stuck-low-clock fault has no remote fix (full power disconnect).
- Do not poll processes by name from a Claude Code shell (`pgrep -f` matches its own wrapper); wait on output files or
  log lines. Never run analysis with default thread counts beside timed runs.
- Stop at the first anomaly (crash pattern, memory below the floor, unexpected temperatures) and write it down.
- **Chain scripts.** A pool is launched by a new `logs/chain-<id>.sh` (`setsid nohup bash logs/chain-<id>.sh >
  logs/chain-<id>.out 2>&1 < /dev/null & echo $! > logs/chain-<id>.pid`), which either ends with
  `tools/serving-window.sh restore && systemctl --user stop itrea-restore-serving.timer` or does not restore (another
  pool follows in the same window); its header comment and the journal say which. To stop a pool and restart it in the
  same window: kill the chain script first (`kill $(cat logs/chain-<id>.pid)`; otherwise it restores the servers when
  the pool ends), then `touch logs/pool.stop` — the pool opens no new block. Before relaunching, `rm -f logs/pool.stop`:
  the pool only tests that the file exists and nothing removes it, so a new pool would open no block and end `stopped`.
- **Extending the window** (as on 2026-10-09 14:11 and 16:12): re-claim with the same `--who` and the new `--until`,
  then move the restore timer and the hold lines with it:
  ```bash
  box-window claim --who "i-trea-ii (Claude Code)" --why "…" --until HH:MM --stops ornith-vllm,llama-qwen38 --mem <GB>
  systemctl --user stop itrea-restore-serving.timer
  systemctl --user reset-failed itrea-restore-serving.timer itrea-restore-serving.service
  tools/serving-window.sh schedule "YYYY-MM-DD HH:MM:00"
  sed -i 's/until <old ISO>  # i-trea-ii window/until <new ISO>  # i-trea-ii window/' ~/.config/cronrestore/hold-units
  grep -n 'i-trea-ii window' ~/.config/cronrestore/hold-units; systemctl --user list-timers itrea-restore-serving.timer
  ```
  Re-running `serving-window.sh stop` does not move existing hold lines. A running pool keeps the `--deadline` it was
  started with; only the lease end is re-read before each block.

## After (also when abandoning a window early)
1. `python -m itrea.experiments.exchange_ctl flush <run_id>` for any run whose keys remain; `down` if nothing else needs it.
2. Restore — run it yourself when the work ends (or let a restoring chain script do it); do not wait for the timer:
   `tools/serving-window.sh restore` (starts both units, waits up to 15 min for health, resumes the canaries, removes
   the `# i-trea-ii window` hold lines, releases the lease). Then stop the now-unneeded timer:
   `systemctl --user stop itrea-restore-serving.timer`.
3. Verify, and write the result in the journal:
   `systemctl --user is-active ornith-vllm llama-qwen38` **and** `systemctl --user is-enabled ornith-vllm llama-qwen38`
   (active-but-disabled would not survive a reboot), `curl -sf 127.0.0.1:8094/v1/models`, `curl -sf
   127.0.0.1:8092/health`, `hermes cron list` (canaries resumed), `box-window status` (lease released),
   `grep 'i-trea-ii window' ~/.config/cronrestore/hold-units` (nothing).
4. Fill the window row in `vault/10-design/Compute resources.md` (runs, MemAvailable minimum, ACPI maximum, notes).
