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
   **[other project]:** before stopping `ornith-vllm`, ask the [other project] session (cross-session message) — agreed on
   2026-10-12, when [other project]'s hosted model had run out of [hosted model provider] credit and `ornith-vllm` was its only way to answer
   on Slack: that day's M4 pool was stopped after 5 of 30 runs and both servers were restored 12:40–12:43 for it. Since
   2026-10-13 09:32 (Berk's decision) [other project] answers with Claude Haiku 5.5 on the Anthropic API, so both local servers
   may stop for our windows ([other project] session, 10-13); whether [other project] still uses `ornith-vllm` anywhere (a
   fallback, its nightly canary) is not recorded here — ask anyway, as agreed, and write its answer in the journal.
2. Look at what is running: `systemctl --user is-active ornith-vllm llama-qwen38`, `nvidia-smi` (foreign GPU
   processes), MemAvailable, the ACPI maximum (`gates.acpi_max_c()`); a GPU arm must not start above 80 °C. ITX-1
   windows (D-021 R26) open only if the hottest zone's median over the last 60 s is ≤ 50 °C after 3 min idle with
   both LLM servers stopped, and the fan governor is alive (`cat ~/.local/state/thermal/fan-state`: written within
   30 s, `cap=0`).
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
   + 20), as claimed on 2026-10-08/09. **ITX-1 windows (D-021 phase 1, `jobs/itx-*.jsonl`, `itrea make-jobs
   itx-…`)** — every job carries a `harness`, so cuOpt's caps come from it (`--cuopt-mem-gb` / `--gpu-mem-gb` do not
   change them; `--cuopt-gpu-peak-gb` is refused); both servers in `--stops` (the gates below exceed what is free with
   them up); `--dry-run` prints the arithmetic per treatment ("memory per run: …"):

   | step | file (set) | runs | memory gate (`--mem`) | run hours | pool flags |
   |---|---|---|---|---|---|
   | M1 smoke | `itx-m1.jsonl` (itx-m1): fhnw-binpack4-48, glass4, radiationm18-12-05 × {A9, A6F, A2F} ext + A6F with the thread variables kept, 120 s | 12 | 97 (A9 ext; A2F 78, A6F 69) | ≈ 0.3 (≤ 0.43) | — (first: `pytest -m gpu` with cold caches, step 5) |
   | M2 soak at F | `itx-m2.jsonl` (itx-m2): comp21-2idx, neos-4763324-toguru × {A9, A6F} ext, 600 s, AB / BA | 4 | 97 | 0.67 | `--pause-c 90`; start the first from fan AUTO |
   | M3 memory probe | `itx-m3.jsonl` (itx-m3): R29's 8 × A6F at cuOpt RMM 40 / host 40; a cap hit is recorded (`void_info.cuopt_memory`), not voided | 8 | **101** | 1.0–1.35 | — |
   | M4 calibration | `itx-m4.jsonl` (mipfeas): D-020's 30 × A6F ext, 600 s | 30 | 69 | 3.95–5.0 | then `itrea mipfeas-report --arm A6F`; **stopped after 19 of 30 (2026-10-13, performance runs first, D-024 rule 0)** — the pool resumes the rest (completed blocks are skipped); the strata-0–1 runs before v1's stage 1, the others only in otherwise idle daytime hours |
   | M5 A9 preview | `itx-m5.jsonl` (itx-m5): the 15 dev / class-probe instances in the 233 × A9 ext, 600 s | 15 | 97 | ≈ 2.0–2.5 | optional |
   | C1 diagnosis R1 (D-025, after D-023's verdict) | `r1-bsync.jsonl` (r1-bsync): ca, sing44, neos-960392, item_placement_9903, comp21-2idx × {A9@bsync or A9@boY-bsync, A9@r1ctl}, ext, seeds 0–1, 120 s | 20 | 97 | ≈ 1.15 | BEFORE the first timed run, untimed, lease held, no `nice`, `taskset -c 6`: `pytest -m gpu tests/test_cuda_sched.py` (flag set / read back incl. the default file, the spin check at 12 s: blocking sync <= 0.25, the stand-in >= 0.8 under yield and the +10 renice), cold cache; then `itrea d025-report` |
   | C1 diagnosis R2 (D-026, after D-025's verdict) | `r2-ld.jsonl` (r2-ld): item_placement_9903, sing44 × seeds 0–3 + ca, glass4, kumeu × seeds 0–1 × {A9@r2ld, A9@r2ctl}, ext, 120 s | 28 | 97 | ≈ 1.6 | BEFORE: `pytest -m gpu tests/test_lp_decode_gate.py` (the > 8-part concatenate, the gate on the stamps, the decoder's nice) and the repository's GPU tests; then `itrea d026-report --base … --nice …` |
   | R13 + R14 test + C1 hint (D-022, narrowed 2026-10-13) | `itx-r13.jsonl` (itx-r13): the 3 R13 instances × {A9@r13, A9@r13x, A6F@r13x} + C1's fixed 11 × {A9@r13x, A6F@r13x}, seeds 0–1, 120 s | 58 | 97 | ≈ 1.9 | its decision record first; then `itrea d022-report` |
   | C1 guard (D-022 second amendment) | `itx-r13b.jsonl` (itx-r13): chromaticindex1024-7, kumeu, glass4, radiationm18-12-05 × {A9@r13x, A6F@r13x} × seeds 0–1, 120 s | 16 | 97 | ≈ 0.5 | after the 58, same set |

   Run hours exclude the ~8 s per run of harness overhead and the governor's rests; window hours ≈ run hours / 0.55 +
   0.5 per window (D-021's budget). M4's stop rule: any thermal or cap void, or any run with a steady-state p95 above
   85 °C, in its first 2 h → back to M2's analysis. Read M2's numbers (R23: steady-state p95 ≤ 85 °C, no 5-s sample
   ≥ 90 °C, no cap, busy SM median ≥ 2,350 MHz) before M3 / M4. **A6m (D-020) is out of v1 (D-021 R9)**: its pool
   (`--layout full --cuopt-mem-gb 24 --gpu-mem-gb 36`, 81 GB, `make-jobs mipfeas-cal`) runs only through the heat
   ladder with Berk's yes for that window.
   **ITX-1 phase 2 (2026-10-12):** the v1 tiers (`itrea make-jobs itx-v1 --tier T1..T5 --out F`, written only once
   cuOpt 26.10 is in and `prereg-v2` is tagged with `configs/frozen-v2.yaml` — A9's configuration, the C1 freeze — in
   the tagged commit; the writer refuses an A9 tier without that file, and the pool refuses external A9 jobs unless the
   tag is an ancestor of HEAD, the file is unchanged since it and the jobs' config hashes to it, R33) carry preset
   `itx1` (ext + R12's stop at the proven optimum + cuOpt killed at a definitive outcome + `scoped_hash`: A0's run ids
   depend only on its SCIP-side harness): same gates as ext (A9 97, A2F 78, A6F 69 GB). Every external job (A0's too)
   needs a clean tree under `src/` and `configs/`. **Check the installed cuOpt against the jobs' variant tag
   (`v1c2610`) before the pool** — the pool does not. CPU-only windows (both servers off: every job has a harness; the
   memory column is the pool's gate, the 20-GB reserve counted once):
   | step | file (set) | runs | layout | memory | run hours |
   |---|---|---|---|---|---|
   | M6 alone | `itx-m6a.jsonl` (itx-m6a): crna, fhnw-binpack4-48, glass4 × A0 × 60 s | 3 | `--layout screen --cpu-lanes 1` | 9 + 20 = 29 GB | 0.05 |
   | M6 beside + heat | `itx-m6.jsonl` (itx-m6): 3 loads, then the 3 measured runs beside them, then 11 loads × 600 s (`priority`) | 17 | `--layout screen --cpu-lanes 4` | 4 × 9 + 20 = 56 GB | ≈ 0.6 (4 lanes, 30+ min) |
   | A0 | `itx-a0.jsonl` (itx-a0): A0@v1 × the 230, itx1, 600 s, seed 2000; H_cu ∪ W_SCIP unseen (81) first | 230 | `--layout screen --cpu-lanes 4` (3 if M6 says so) | 4 × 9 + 20 = 56 GB (3 lanes: 47) | 24–33 serial (D-021), ≈ 6–8.3 on 4 lanes |

   Read M6 with `itrea build-runs` then `itrea itx-report --m6` (4 lanes iff nodes/s beside is at most 3 % slower and
   the hottest zone stays < 85 °C). The screen layout's single-core X925 lanes need no E0 flag (the pair layout's
   `--cpu-lanes` does). A block voided by `ref_rejected` (R12's trigger not confirmed by the exact checker) is rerun once
   with the stop off — normal, counted in `counts.requeues`.
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
   **A chain scheduled for later (a transient `systemd-run --user --on-calendar` timer) must be proven at scheduling
   time, not at its start** (2026-10-14: the 07:05 chain's claim `--until 20:15` was refused — `box-window` refuses a
   lease longer than 12 h, "not a window; split it" — and the chain exited at 08:00 with nobody told; 7.5 h of box
   lost): (a) keep each lease ≤ 12 h (split a long day: the chain claims the second lease itself after the first
   pool); (b) give the chain's service `-p OnFailure=agent-failure-alert@%n.service` and make every early exit
   `exit 1`, so a refused claim alerts in [ops channel]; (c) at the chain's start time, check `logs/chain-*.out` within
   15 min — a waiting session does that, not the chain.
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
  cuOpt RMM limit + Redis 2 GB + 20 GB reserve ≤ MemAvailable), thermal start < 80 °C for GPU arms and < 86 °C for
  CPU-only arms (D-008, D-021 R24; 88 until 2026-10-11), pause ≥ 92 °C (`--pause-c`, never higher; M2: 90) / resume
  < 80 °C, fan-governor's rest flag (D-009; a governor dead for 120 s while it holds the pool halts it), and for GPU runs
  the governor itself (D-021 R24: `~/.local/state/thermal/fan-state` younger than 30 s and `cap=0`; it waits while the
  cap is on and halts after 120 s without a live governor; `--no-governor` only on a host without it), both model
  servers stopped for every job with a harness (checked before every run), watchdog floor 18 GB, one GPU arm at a
  time, one runner (`/tmp/itrea-runner.lock`). Every process runs in a named scope
  `itrea-run-<sha1(run)>-<role>-<epoch_ms>.scope` (`systemctl --user list-units 'itrea-run-*'`) that stops within 5 s
  (cuOpt's by SIGKILL).
- D-021's void causes (`run.json` `void_cause`, every one found in `void_checks`, the rest in `void_info`):
  `gpu_cap` (busy SM median below `--cap-floor-mhz` 2,300, the governor's 2,200-MHz clock cap during the run, or a fan
  state with cap=1 stamped inside the run) — the block is rerun once, after the fan reads AUTO; a low clock with the
  governor's cap off before and after and no cap interval (an unknown lock, e.g. left by a governor restart; it needs
  `sudo nvidia-smi -rgc`) halts the pool at once; `compiled_on_clock` (harness: a CUDA-cache entry or CuPy cache growth
  after t0), `cuopt_memory` (harness: cuOpt's cap confirmed), `external_kill` (any arm: a SIGKILL the runner did not
  send, at no cap) and `ref_rejected` (harness with `ref_stop`: the incumbent that stopped the run at the proven optimum
  is not confirmed; the rerun has the stop off) — rerun once; `thermal_watch` (thermal-watch stopped one of the run's scopes at ≥ 98 °C) — rerun as
  a thermal void. A further void of the same cause skips the block (`skipped`, reason = the cause);
  `counts.requeues` in `logs/pool-status.json` counts the reruns by cause. **R26:** after 2 thermal or cap voids
  (`--max-heat-voids`; one thermal pause counts once) the pool opens no new block and ends `halted` — close the
  window and look before another pool.
- A block voided by a thermal pause is rerun first, at most `--max-thermal-requeues` times (default 1); one voided again
  is skipped and the pool moves on — `logs/pool-status.json` lists it under `skipped` with its run ids. Those blocks
  are owed: rerun them in a later window (the pool treats their moved-aside runs as not run).
- Every run is re-checked just before it starts: if its estimate would end after `--deadline` or the lease end it does
  not start (after a long thermal, rest-flag or memory wait this can stop the pool early); a block already partly run
  is finished. Re-claim the lease before it runs out rather than let the pool stop.
- A GPU-health fault (median SM clock < 2,000 MHz, or foreign GPU activity > 2 % of a run) voids the whole block and
  halts the run set: look before resuming. The stuck-low-clock fault has no remote fix (full power disconnect). A
  2,000–2,300 MHz median is the governor's cap, not that fault: a `gpu_cap` void, no halt.
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
  Every chain writes its PID file at launch, a replacement chain included (`chain-itx-m4c`, `-r13` and `-r13b` of
  2026-10-13 have none, which leaves a chain waiting in its `until grep` loop findable only by a process scan).
- **The stop-file trap.** A chain whose later steps are guarded by `if [ ! -f logs/pool.stop ]` skips them when the
  file exists and falls through to `serving-window.sh restore`: stopping a pool with the stop file in order to run a
  different pool next (as on 2026-10-13 15:1x, when the M4 calibration made way for D-022) would restore the servers
  in the middle of the window. To switch pools, **replace the chain**, not just the pool: the new chain waits for the
  running pool's end line, removes the stop file itself, then runs the next pool unconditionally
  (`logs/chain-itx-r13.sh`):
  ```bash
  until grep -qE "pool <window-id> (done|stopped|halted)" logs/<old>.log; do sleep 15; done
  rm -f logs/pool.stop
  .venv/bin/python -m itrea.experiments.pool jobs/<next>.jsonl --layout pair --deadline HH:MM > logs/<next>.log 2>&1 < /dev/null
  ```
- **Replacing a running chain's tail** (2026-10-13 14:5x and 15:2x–15:3x; the running pool keeps running):
  1. Write the new chain (`logs/chain-<new>.sh`, header comment: which chain it replaces and why): it waits on the
     running pool's end line in that pool's log (`until grep -qE "pool <window-id> (done|stopped|halted)" …`), then
     runs the remaining pools and ends with the restore (or states that it does not restore).
  2. Find the old chain's bash: the running pool's PID is `pid` in `logs/pool-status.json` (also the window id's
     suffix, and in `/tmp/itrea-runner.lock`); the chain bash that launched it is its parent, `PP=$(ps -o ppid= -p
     <pool pid> | tr -d ' ')`. Confirm it before killing: `tr '\0' ' ' < /proc/$PP/cmdline` must name
     `logs/chain-<old>.sh`, and `ps -o user= -p $PP` must be yours. If the old chain is not the pool's parent (a
     replacement chain still in its wait loop) and has no PID file, scan `/proc/*/cmdline` with a pattern built by
     concatenation (`PAT="chain-itx""-r13b"`) so it cannot match your own shell, and exclude `$$` — never `pgrep -f` /
     `pkill -f`.
  3. `kill $PP` — the bash only. The pool (and its runs) continue: it is reparented to `systemd --user` (on 10-13
     pool 335372's parent became PID 2843, `systemd --user`). Never signal the pool or a run scope.
  4. To end the running pool early as well (switching pools), `touch logs/pool.stop` only now, after the old bash is
     gone; the new chain removes it when the pool's end line appears.
  5. Launch the new chain detached (`setsid nohup … & echo $! > logs/chain-<new>.pid`); check its `.out` and the
     journal entry name both chains. The lease, the restore timer and the hold lines must cover the new chain's end
     (extend them as below if not).
- **An idle box is used** (Berk 2026-10-13: "If the box is idle make the most of it"). Inside a daytime lease, chain the
  queued pools back to back — performance runs first (D-024), then calibration or CPU-only checks (M6) that fill the
  remaining hours — rather than ending a window with lease time left; never into the night rest (00:00–07:00): on
  10-13 the lease, the restore timer and the hold lines were moved to 23:45 for M6 and no further.
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
