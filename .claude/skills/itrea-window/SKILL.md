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

**Where to write.** A window's logistics (lease, stops and restores, chains, timers, messages with other sessions) go to
the private operations repository `Tideseed/i-trea-ii-ops`; what the window measured goes in the public journal and the
result notes (skill `itrea-journal`, "What goes where").

The two servers' systemd units are named in `SERVING_UNITS` (`src/itrea/experiments/gates.py`). Set them once per
shell, from the repository root, before the commands below — `$U1` is the vLLM server (:8094), `$U2` the llama.cpp
server (:8092):
```bash
read U1 U2 < <(.venv/bin/python -c 'from itrea.experiments.gates import SERVING_UNITS as u; print(*u)')
```

## Before
1. `box-window status` — if another session holds an unexpired lease, **wait or ask it**; never `--force`.
   **Another project's chat agent:** before stopping the vLLM server (`$U1`), ask that project's session
   (cross-session message; which session: the operational notes of 2026-10-12, in the private operations repository)
   — agreed on 2026-10-12, when the agent's hosted model
   had run out of credit and `$U1` was its only way to answer its users: that day's M4 pool was stopped after 5 of 30
   runs and both servers were restored 12:40–12:43 for it. Since 2026-10-13 09:32 (Berk's decision) the agent answers
   with a hosted model, so both local servers may stop for our windows (that session, 10-13); whether it still uses
   `$U1` anywhere (a fallback, its nightly canary) is not recorded here — ask anyway, as agreed, and write its answer
   in the operations repository (skill `itrea-journal`, "What goes where").
   **Other sessions' CPU jobs:** unpinned CPU-heavy work of other sessions can be scheduled onto our run cores (A725
   1–3, X925 5–6) and adds heat: another project's forecasting session ran CPU-heavy jobs unpinned and without a
   lease inside W-1014 and W-1015 (e.g. one unit 26 CPU-min 11:20–13:34; joblib workers at ~95 % CPU each;
   operational notes of 2026-10-15, private operations repository). Ask the session to pin such work to cores
   10–14 at nice 19 or hold it during our lease; that session has done so since 2026-10-15 19:0x.
2. Look at what is running: `systemctl --user is-active $U1 $U2`, `nvidia-smi` (foreign GPU
   processes), MemAvailable, the ACPI maximum (`gates.acpi_max_c()`); a GPU arm must not start above 80 °C. ITX-1
   windows (D-021 R26) open only if the hottest zone's median over the last 60 s is ≤ 50 °C after 3 min idle with
   both LLM servers stopped, and the fan governor is alive (`cat ~/.local/state/thermal/fan-state`: written within
   30 s, `cap=0`). The pool enforces R26's breaker but not this opening gate (#38): a chain runs
   `tools/r26-gate.sh --deadline HH:MM` right after `serving-window.sh stop` and restores if it exits 1 (written
   2026-10-13 after W-1013b opened at 52–54 °C; it opened W-1014 at 14:45, median 50.0 °C, and W-1015 / W-1016 at
   07:08, median 46.0 °C).
3. Claim, with the exact `who` string the runner checks:
   ```bash
   # Before claiming: daytime windows only — no I-TREA II window in the night rest (00:00–07:00): Berk, 2026-10-10,
   # "No nights" (D-019; until then D-009 asked for his yes for each night). Long run sets (MIPfeas ≈ 78 h) are split
   # into daytime windows, with the fan governor's rests. Check `tail ~/.local/state/thermal/fan-governor.log`
   # and `ls ~/.local/state/thermal/rest`: the runner waits while the rest flag exists. Operational notes of 2026-10-15
   # 14:4x (an estimate): four 30-min rests by 14:57, about every 1.5–2 h under A9 load, about 25 % of window time
   # (30 min every 1.5–2 h is 25–33 %) — budget window hours for it.
   box-window claim --who "i-trea-ii (Claude Code)" --why "<experiment id and what runs>" --until HH:MM \
     [--stops $U1,$U2] [--mem <GB>]
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
   | M4 calibration | `itx-m4.jsonl` (mipfeas): D-020's 30 × A6F ext, 600 s | 30 | 69 | 3.95–5.0 | then `itrea mipfeas-report --arm A6F`; stopped after 19 of 30 (2026-10-13, performance runs first, D-024 rule 0), resumed in idle hours (the pool skips completed blocks); **complete 30 of 30 on 2026-10-15 10:00, OUTSIDE (R 1.313, the footprint)** |
   | M5 A9 preview | `itx-m5.jsonl` (itx-m5): the 15 dev / class-probe instances in the 233 × A9 ext, 600 s | 15 | 97 | ≈ 2.0–2.5 | optional |
   | C1 diagnosis R1 (D-025, after D-023's verdict) | `r1-bsync.jsonl` (r1-bsync): ca, sing44, neos-960392, item_placement_9903, comp21-2idx × {A9@bsync or A9@boY-bsync, A9@r1ctl}, ext, seeds 0–1, 120 s; fallback (b) `r1-ysync.jsonl` (r1-ysync, `--variant ysync`), the same 20 (ran 2026-10-15 07:17–08:06 and 10:00–11:09) | 20 | 97 | ≈ 1.15 | BEFORE the first timed run, untimed, lease held, no `nice`, `taskset -c 6`: `pytest -m gpu tests/test_cuda_sched.py` (flag set / read back incl. the default file, the spin check at 12 s: blocking sync <= 0.25, the stand-in >= 0.8 under yield and the +10 renice), cold cache; then `itrea d025-report` |
   | C1 diagnosis R2 (D-026, after D-025's verdict) | `r2-ld.jsonl` (r2-ld): item_placement_9903, sing44 × seeds 0–3 + ca, glass4, kumeu × seeds 0–1 × {A9@r2ld, A9@r2ctl}, ext, 120 s | 28 | 97 | ≈ 1.6 | BEFORE: `pytest -m gpu tests/test_lp_decode_gate.py` (the > 8-part concatenate, the gate on the stamps, the decoder's nice) and the repository's GPU tests; then `itrea d026-report --base … --nice …` |
   | R13 + R14 test + C1 hint (D-022, narrowed 2026-10-13) | `itx-r13.jsonl` (itx-r13): the 3 R13 instances × {A9@r13, A9@r13x, A6F@r13x} + C1's fixed 11 × {A9@r13x, A6F@r13x}, seeds 0–1, 120 s | 58 | 97 | ≈ 1.9 | its decision record first; then `itrea d022-report` |
   | C1 guard (D-022 second amendment) | `itx-r13b.jsonl` (itx-r13): chromaticindex1024-7, kumeu, glass4, radiationm18-12-05 × {A9@r13x, A6F@r13x} × seeds 0–1, 120 s | 16 | 97 | ≈ 0.5 | after the 58, same set |
   | C1 diagnosis R3 (thread count, SCIP beside cuOpt) | `r3-diag.jsonl` (r3-diag): comp21-2idx, toguru, doon, chromaticindex1024-7 × {A6@r3, A2@r3} + toguru, doon, chromaticindex1024-7 × A6F@r3, ext, seeds 0–1, 120 s (D-027 as amended 17:5x and 18:3x: only what a rule reads; rule 4 = chromaticindex) | 22 | 78 (A2; A6 / A6F 69) | ≈ 1.3 | no EA code; any window after backoff2; its record D-027; then `itrea d027-report` |
   | C1 re-read on the frozen-v2 candidate (D-030; Berk's "Re-run C1 on frozen hash") | `c1r-ld0.jsonl` (c1r-ld0) — or `c1r-ysync-ld0.jsonl` (c1r-ysync-ld0) only if D-029 adopts ysync AND Berk approves the combination; never both; **`c1r-ysync-ld0` ran** (36 completed, 1 failed, 2 void on 2026-10-15 15:09–17:30, halted by R26's breaker; the 16 remaining blocks, 31 runs, on 10-16 07:08–08:17): the 17 × {A9@c1r, A6F@c1r} × seeds 0–1, ext, 120 s, arms alternating by seed | 68 | 97 (A6F 69) | 3.9–4.8 (68 × 2.0–2.1 min per run + 3–5 fan-governor rests of ≈ 30 min, measured on itx-r13 / r1-* / r2-ld; the 3.44 min/run of 10-13 already contains the rests) | `--until` ≥ 4.5 h of runs + the pool's 10-min open margin + the restore, else the last blocks fall to a second window; may split across windows by whole blocks (a deadline never splits a block; a block half-done after a crash is voided and rerun whole on resume; the same job file resumes); ysync-ld0 BEFORE its first timed run: `pytest -m gpu "tests/test_lp_decode_gate.py::test_cupy_cold_cache_r2_variant_in_the_a9_layout[ysync-ld0-0]"` (lease held); then `itrea c1r-report --a9 …` |
   | chromaticindex harness diagnostic (D-028) | `d028.jsonl` (d028): chromaticindex1024-7 × seeds 0–1 × {A9@boY under `r13-off`, A6F ext}, 120 s | 4 | ext's (A9 97, A6F 69) | ran 2026-10-15 11:09–11:17 | no code; `make-jobs backoff2 --harness r13-off --instances chromaticindex1024-7`, the bo1s rows dropped; read from the run table (`cuopt_t_first`) by D-028's rule |
   | R1 follow-up (D-029) | `d029.jsonl` (r1f-ysync, plus one unread warm-up run in `r1f-ysync-warmup` at `priority` 0): item_placement_9903, comp21-2idx × seeds 2–5 + ca × seeds 2–3 × {A9@ysync, A9@r1fctl}, ext, 120 s | 21 | 97 | ran 2026-10-15 13:16–14:59 (two governor rests) | written by `tools/d029_report.py --write-jobs`; read with `tools/d029_report.py` (no `itrea` entry) |
   | glass4 harm check (D-032) | `d032.jsonl` (d032): glass4 × seeds 0–3 × {A9@g4old `-3way.yaml`, A9@g4yl `-3way-ysync-ld0.yaml`, A9@g4ld `-3way-ld0.yaml`}, ext, 120 s, one block of 3 per seed | 12 | 97 | ran 2026-10-16 08:27–08:52 | rule fixed in D-032 before the runs; read from the run table (seed means, d against glass4's band 0.057) |
   | C1 on four seeds (D-031) | `d031.jsonl` (c1r-ysync-ld0-s23): D-030's jobs at seeds 2–3 | 68 | 97 (A6F 69; D-031's dry run) | 3.9–4.8 with rests (D-031) | starts only after D-030 reads 68 of 68, not INVALID, hint present (D-031's conditions 1–3). If D-030's harm leg fires, the default is the ld0 re-read and D-031 runs only on Berk's word (condition 4). Harm did fire, on glass4 and on krka at floor level; Berk chose the targeted glass4 check (D-032), whose DRIFT branch ("run D-031 today") queued it; krka's floor-level flag was stated, not re-tested; default `--order-seed`, no warm-up; written by `tools/d031_report.py --write-jobs`; read with `tools/d031_report.py --d030-json logs/d030-reading.json`; running since 2026-10-16 08:57 |

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
4. Confirmatory runs (and the screening pass) need **both LLM servers stopped** (vLLM serving a 35B-A3B
   mixture-of-experts model on :8094, llama.cpp serving Qwen3.8-27B on :8092; Berk's yes of 2026-10-05). Schedule
   the restore **first**, then stop:
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
   `exit 1`, so a refused claim alerts on the host's alert channel; (c) at the chain's start time, check
   `logs/chain-*.out` within 15 min — a waiting session does that, not the chain.
   The chains of 2026-10-15 and 10-16 (`logs/chain-1015.sh`,
   `chain-1016.sh`, transient timers at 07:05) followed (a)–(c) and opened on schedule: lease 07:05–18:55 (11 h 50 min,
   the claim retried until 08:00), restore timer 18:45, `serving-window.sh stop`, `tools/r26-gate.sh --deadline 08:30`
   (step 2), the GPU checks before any timed run (10-15), the pools, then a queue file (`logs/queue-1015.txt`, `queue-1016.txt`: job
   files appended by the session after each reading; the chain waits ≤ 45 min for the next, nothing after 18:00), then
   the restore. A session check is scheduled at the chain's start (07:20) and at the restore.
   Development windows have also stopped both servers since 2026-10-07. The D-010 variant — `ITREA_UNITS=$U2`
   (the llama.cpp server only; §7) with the pool's `--allow-serving` (its 08:2x addendum) — is for development runs
   that fit beside the vLLM server (`$U1`), never for confirmatory or screening runs.
5. GPU tests (they use the GPU, so only inside the window, before the timed pool): after any change to GPU code run
   `pytest -m gpu` on the touched modules, including the cold-cache tests (`test_cupy_cold_cache_*` in
   `tests/test_ea_run.py`, `test_elite.py`, `test_lp_decode.py`), then a short smoke run whose summary says
   `compiled_on_clock = false` (skill `itrea-new-experiment` §4).
6. Exchange platform (if the run set uses Redis): `python -m itrea.experiments.exchange_ctl up` (only the container
   `itrea-redis`; never the host's other Redis instances).
7. Operational note in the private operations repository (skill `itrea-journal`, "What goes where"): window id,
   lease, what was stopped, MemAvailable before and after. The public journal records only what the window measures.

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
  window and look before another pool. It fired on 2026-10-15 17:30 (pool `20261015-150955-187322`): after a 30-min
  governor rest at 45 °C the hottest zone reached 85–87 °C within ~4 min, the governor capped the clock at 2,200 MHz
  twice, and two `gpu_cap` voids followed; the next day's pool on the same job file resumed the 16 unfinished blocks,
  the voided one whole (journal 2026-10-15 17:30, 2026-10-16 07:05).
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
  pool follows in the same window); its header comment and the operational note say which. To stop a pool and
  restart it in the same window: kill the chain script first (`kill $(cat logs/chain-<id>.pid)`; otherwise it
  restores the servers when the pool ends), then `touch logs/pool.stop` — the pool opens no new block. Before relaunching, `rm -f logs/pool.stop`:
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
  5. Launch the new chain detached (`setsid nohup … & echo $! > logs/chain-<new>.pid`); check its `.out`, and name
     both chains in the operational note. The lease, the restore timer and the hold lines must cover the new chain's end
     (extend them as below if not).
- **An idle box is used** (Berk 2026-10-13: "If the box is idle make the most of it"). Inside a daytime lease, chain the
  queued pools back to back — performance runs first (D-024), then calibration or CPU-only checks (M6) that fill the
  remaining hours — rather than ending a window with lease time left; never into the night rest (00:00–07:00): on
  10-13 the lease, the restore timer and the hold lines were moved to 23:45 for M6 and no further.
- **Extending the window** (as on 2026-10-09 14:11 and 16:12): re-claim with the same `--who` and the new `--until`,
  then move the restore timer and the hold lines with it:
  ```bash
  box-window claim --who "i-trea-ii (Claude Code)" --why "…" --until HH:MM --stops $U1,$U2 --mem <GB>
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
3. Verify, and write the result in the operational note (private operations repository):
   `systemctl --user is-active $U1 $U2` **and** `systemctl --user is-enabled $U1 $U2`
   (active-but-disabled would not survive a reboot), `curl -sf 127.0.0.1:8094/v1/models`, `curl -sf
   127.0.0.1:8092/health`, `hermes cron list` (canaries resumed), `box-window status` (lease released),
   `grep 'i-trea-ii window' ~/.config/cronrestore/hold-units` (nothing).
4. Record the window: its logistics (lease, stop and restore times, chains, timers) in the operational note (private
   operations repository); the runs' conditions (runs, MemAvailable minimum, ACPI maximum, voids and pauses) in the
   public result note and the Experiment index row (skill `itrea-result-note`).
