# D-025 A9 EA blocking sync (R1) — the EA stops spinning on the GPU so its LP worker gets core 6, development test (2026-10-14, AgentRA)

**Status:** ysync RESULT 2026-10-15 11:2x: NOT ADOPTED (hint 0.834 present; item_placement's generation guard failed on one seed) — the R1 line ends for C1. bsync RESULT 2026-10-15 08:1x: NOT ADOPTED (hint present 0.765; generation guard and comp21 P failed) → fallback (b) ysync queued (§ Result). amended / confirmed 2026-10-14 18:3x before any run (see the end). decided for development by AgentRA (Claude Code, Anthropic Claude Opus 5.5) before any run; open to a
superseding decision on evidence. **Not run; no code merged** (branch `r1-bsync`, worktree `../i-trea-ii-r1`).
**Clarified and amended 2026-10-14 17:4x, before any run** (the section at the end: names as built, the control's own
label, bnice without the flag, the control's flag read back, the P band on the hint instances and a generation guard
on seed means now gate, a harm leg at the C1 re-read, Berk's answer (a)). Runs only
after [[D-023 A9 back-off, phase Y only — yield until cuOpt's first incumbent, then full duty]] has a final verdict, which
sets this test's control. Development only (D-010). Candidate R1 of [[2026-10-14 C1 diagnosis and improvement candidates]]
§5; **R1, R2 and R3 in D-025–D-027 are that note's candidates, not D-021's rules R1–R33** (D-022's "R13 + R14" are
D-021's). Issues: #23 (A9); the prerequisite of [[D-026 lp_decode inside A9 on the EA's core after the LP worker (R2), development test]] (#30).

## Why
Berk (2026-10-14 14:4x): **"What are you waiting for to improve more?"** The diagnosis (the note above, §3) found that
the largest single source of A9's margin above a computed race of SCIP alone and cuOpt alone, pooled over the six C1
wins, is the EA's own asynchronous HiGHS root-LP rounding: 0.754 of the 1.461 margin (52 %; neos-960392 and ca). Inside
A9 that LP is admitted 1.3–1.9 times later than in A3 with the same EA configuration (ca 39.3–39.5 vs 28.7–29.3 s; sing44
19.3–19.5 vs 10.2–10.8 s; the A3 runs come from earlier revisions, and every "LP arrival" in the traces is the EA's poll
time, not the solve's end).

**Mechanism (the note's §5 R1).**
- In A9 the EA host and the HiGHS LP worker it spawns share X925 core 6. The worker is a 'spawn' child with `cpus=None`,
  so it inherits the EA's affinity.
- While the host waits on the GPU it busy-spins, CUDA's default: `ls_fj_gpu.py` syncs after every launch (~50 ms each)
  through `int(cp.count_nonzero(...))`; `backend.synchronize` is `Stream.null.synchronize`; nothing in `src/` sets a
  scheduling flag.
- So the host keeps core 6 about 100 % busy even when 99 % of its wall time is GPU repair; Linux splits the core about
  50 / 50 with the LP worker, and the LP finishes about twice as late. The worker is started by the EA process itself
  (`seeding.LPWorker`, multiprocessing "spawn") inside the EA role's systemd scope, and a child inherits its parent's
  cgroup, so nice acts between the two (the note's §6: between sibling scopes only `cpu.weight` does); the stand-in check
  of fallback (a) verifies it.

**Evidence (the note's §5 R1, re-derived 2026-10-14).** LP-worker CPU = `run.json`'s wait4 rusage of the EA (which
includes reaped children) minus `ea.summary.json`'s RUSAGE_SELF.

| instance | arm (seeds 0 / 1) | LP-worker CPU (s) | LP polled at (s) | EA's first own incumbent (s) | P | cuOpt-own P |
|---|---|---|---|---|---|---|
| ca | A9@bo1 (phase Y paused 14.94 / 14.81 s) | 15.5 / 15.4 | 23.19 / 24.05 | 35.0 / 34.5 | 0.300 / 0.292 | 0.702 / 0.692 |
| ca | A9@bo1s (same day, control) | 17.0 / 17.3 | 39.27 / 39.53 | 46.4 / 47.0 | 0.393 / 0.396 | 0.731 / 0.765 |
| ca | A9@r13x | 17.0 / 18.2 | 39.26 / 39.48 | 46.4 / 47.0 | 0.391 / 0.396 | 0.723 / 0.734 |
| sing44 | A9@r13x | 6.0 / 6.1 | 19.50 / 19.31 | – | 0.441 / 0.449 | – |
| sing44 | A3x@a9p (same core-6 layout, no cuOpt) | 5.4 | 10.92 | – | 0.287 | – |
| neos-960392 | A9@r13x | 2.4 / 2.5 | 5.42 / 6.07 | – | 0.065 / 0.050 | – |
| chromaticindex, rail01 | A9@r13x | 30.4 in a 60-s life (time limit, no solution) | – | – | – | – |

- **ca is a natural experiment** (D-018's bo1 against its same-day control bo1s). Phase Y paused the EA for about 15 s,
  and the LP was polled about 15 s earlier. The EA's LP-rounding step (gap 0.994 → 0.011) moved 11.4 / 12.4 s earlier;
  its credit, 0.983 × dt / 120 = 0.093 / 0.102, equals the observed P difference (0.093 / 0.104) to within 0.002. So
  bo1's gain on ca came from the EA's own LP getting core 6, not from cuOpt's GPU share: cuOpt's own P was about the same
  (0.70 vs 0.73–0.77). A9@r13x matches bo1s, so the harness change does not matter here. bo1 s1 is D-018's failed run
  (cuOpt hit its RMM limit at 83 s); its first 47 s are intact.
- **Consequence for reading D-023:** a ca gain under boY is probably this LP-core mechanism, not cuOpt's GPU share. That
  changes how ca's boY row is read, not D-023's rules.
- **Rebuild from r13x's generation boundaries (the note's review).** With a full core the LP would be polled at 18.3 /
  20.8 s on ca (instead of 39.3 / 39.5 s), 11.3 / 11.6 s on sing44 (the next boundary; instead of 19.5 / 19.3 s) and
  2.8–4.2 s on neos-960392 (instead of 5.4 / 6.1 s). The generation boundary, not the solve, sets the poll stamp and caps
  every "LP earlier" gain.

## Decision — what is tested
**The change (one variable):** the EA sets its CUDA primary context to blocking sync before anything retains it. CuPy
14.2 has no `setDeviceFlags`, so this goes through the driver API — `libcuda.so.1`, `cuInit`, then
`cuDevicePrimaryCtxSetFlags(dev, CU_CTX_SCHED_BLOCKING_SYNC)` — before warm-up and t0. Under the same switch, `_Phases`'
per-generation timing events are created with `block=True`, because a default-flag event wait may spin whatever the
context flag says. **Everything that changes behaviour sits under the switch**; with the switch off the EA does exactly
what it does today.

**Bundled instrumentation (no behaviour change, present in both arms):** the LP worker stamps its own start and
completion time into the `lp` event, so solve time and poll time are recorded separately; `ea.summary` records the
primary context's flags as read back with `cuDevicePrimaryCtxGetState`.

**Configuration.** The switch is one new EA configuration field, at today's behaviour by default and registered in
`config._LATE_DEFAULTS`, so that every existing configuration keeps its hash — 1d12ad490dd0
(`configs/itrea2-dev-rc-3way.yaml`, the A9 / A3x default), 5c8b535fb245 (`-boY.yaml`), 198e0d2ca540 (`-bo1s.yaml`),
f030e23909ad (`configs/itrea2-dev-rc-ld.yaml`, the A3 default) and every other file in `configs/`. The field's name is
the build's; it is recorded in a dated clarification here before any run. The test runs a **new variant file**:
- `configs/itrea2-dev-rc-3way-bsync.yaml` = `itrea2-dev-rc-3way.yaml` verbatim + the switch (variant `bsync`), if the
  control is the current default;
- `configs/itrea2-dev-rc-3way-boY-bsync.yaml` = `itrea2-dev-rc-3way-boY.yaml` verbatim + the switch (variant
  `boY-bsync`), if D-023 adopts boY.

**A stated deviation from the note:** the note puts the switch into `configs/itrea2-dev-rc-3way.yaml`, so that A3x stays
matched to A9. That file is not edited: it defines today's default (its hash is in every A9@r13x run and in
`jobs/itx-r13.jsonl`), and a pool imports from the main checkout. A3x stays matched on adoption instead (below).

**Unchanged:** processes, cores, F and A6F (D-021 R5 / R6), so D-021 R5's M2 soak does not repeat; SCIP, cuOpt, the
exchange and the harness.

## Test — fixed before any run
- **Set `r1-bsync`**, job file `jobs/r1-bsync.jsonl`, written by `make-jobs r1-bsync --control r13x|boY` (the control is
  a parameter). Instances {ca__Scenario400_reserves_1, sing44, neos-960392, item_placement_9903, comp21-2idx} × arms
  {treatment `A9@bsync`, control `A9@r1ctl`} × seeds 0–1, 120 s, harness **ext**, one block per instance × seed:
  **20 runs** (≈ 1.15 h at itx-r13's 3.44 min per run, excluding the window's open and close, the checks below and
  D-021's 0.55 window efficiency).
- **The control is chosen by D-023's verdict:** D-023 ADOPT → the control runs `itrea2-dev-rc-3way-boY.yaml` (A9@boY,
  5c8b535fb245) and the treatment `-boY-bsync.yaml`; any other final verdict → the control runs
  `itrea2-dev-rc-3way.yaml` (A9@r13x's configuration, 1d12ad490dd0) and the treatment `-bsync.yaml`. Either way the
  treatment differs from the control by the switch only, and the control is re-run in the same window. The control's
  label `A9@r1ctl` is this set's own, so no run id repeats an earlier set's.
- **Validity (must hold):** 20 of 20 runs completed or a recorded void rerun (D-021's voids); 0 compile on the clock
  (CuPy and CUDA); equal P and gene dtype per block; harness ext on every run (run.json `config.harness`); each arm's EA
  config hash as written in the job file; **the flag state in `ea.summary`: blocking sync in every treatment run, the
  default in every control run** — else INVALID, and the report names the leg. Failed runs are outcomes of their arm
  (D-002 §6, `--failed keep`), reported both ways.
- **Mechanism (reported per run, not gating):** the EA's self-CPU per wall-second; the LP worker's own finish time, and its
  CPU divided by its own solve wall (expected ≥ 0.8 on ca and sing44); generations and FJ launches per second on ca,
  sing44 and neos-960392 against the control — the three are where a woken host can preempt a running LP worker, a path
  the guards never exercise.
- **No-harm on item_placement_9903 and comp21-2idx (gating, one-sided — a no-harm rule must not fail the treatment for
  helping, D-023's F correction):**
  - generations, **paired by seed**: G(A9@bsync, s) ≥ 0.95 × G(control, s) for s = 0 and s = 1, G = the EA's
    generations in 120 s (the within-day seed spread on item is 4 %);
  - P on seed means: P̄(A9@bsync) − P̄(control) ≤ max(0.05, |P_control,s0 − P_control,s1|), spreads in raw P (C1:
    item_placement 0.358 / 0.370, comp21 0.552 / 0.543, so both bands would be 0.05).
  - Reported beside, not gating: the same P band on ca, sing44 and neos-960392.
- **Hint (futility-only, fixed before the runs; D-024 rule 2):** the ratio of means of P(120), A9@bsync / control, over
  {ca, sing44, neos-960392} × seeds 0–1 (ΣP over the six treatment runs / ΣP over the six control runs) **≤ 0.95**,
  reported per instance as well as pooled; per instance it gates nothing.
  - **Null replicate** on these three: **0.989** (1.773 / 1.792 — A9@bo1s of 10-11 / A9@r13x of 10-13, the same EA
    configuration on different days); r13x seed 1 / seed 0: **0.997**.
  - **In practice this hint is ca's.** sing44 and neos-960392 together reach at most about −0.045, right at the
    threshold; ca is stable (0.391–0.396 in 4 runs over 2 days).
  - **With A9@boY as the control** the hint is harder to reach: phase Y already gives the LP most of the core on ca, and
    ca's increment shrinks to about −0.03 to −0.05 (the note). No null replicate exists for a boY control (boY has one
    day of runs, set `backoff2`); the replicate above is the r13x-type configuration's.
- **Verdict:**
  - **ADOPT** iff validity holds, both no-harm legs hold on item_placement and comp21, and the hint is present;
  - **NOT ADOPTED** otherwise; the report names the rule, and the fallbacks below apply where their trigger fires. No
    hint and no fallback trigger ends the R1 line for C1;
  - **INVALID** when a validity leg fails (a wrong flag state means wrong code: fixed and re-checked before any rerun,
    with a dated note here).
- **Analysis:** a report script (`tools/d025_report.py`), written and tested before the window, that reads only the run
  table and the set's run directories and prints every leg above.

**Pre-stated fallbacks** (each its own variant file and set, through the same checks before a window, then the same
20-run test against the same control rule, under the same rules; the triggers are made exact here — the note states them
qualitatively):
- **(a) Renice — `configs/itrea2-dev-rc-3way-bnice.yaml` (or `-boY-bnice.yaml`), set `r1-bnice`:** the EA host (all its
  threads) sets itself to nice +10 relative to the LP worker right after spawning it (`os.setpriority`; the value is fixed
  here, the note gives none), so the worker keeps the inherited priority; no context flag. **Applies** if the spin check before the window (check 2 below) fails —
  then the bsync test is not run, and (a) is checked and run in its place — or if the bsync test gives no hint while
  ca's LP worker gets less than 0.8 of the core (its CPU / its own solve wall < 0.8) on both seeds — the lever did not
  reach its mechanism.
- **(b) Yield — `configs/itrea2-dev-rc-3way-ysync.yaml` (or `-boY-ysync.yaml`), set `r1-ysync`:** `CU_CTX_SCHED_YIELD`
  instead of blocking sync (the `_Phases` events unchanged). **Applies** if the paired generation guard fails on any seed
  of item_placement_9903 or comp21-2idx (wake latency). Its spin check differs (below): a yielding host still shows about
  1 CPU-s per wall-second on an otherwise idle core.
- No third variant is pre-stated; any other follow-up (e.g. the note's "LP-ready early exit from the uncapped pre-phase
  repair call") needs its own record.

## What adoption would mean
- ADOPT means **the line widens**, not that R1 improves A9 (D-024 rule 2; every development number stays descriptive,
  D-010): the variant file becomes **A9's and A3x's development default** (A3x runs the same file, so it stays A9's
  matched control — the note's reason for putting the switch in the default); `configs/itrea2-dev-rc-3way.yaml` stays as
  the record of the old default; A3's default (`itrea2-dev-rc-ld.yaml`, f030e23909ad) is untouched; existing job files
  and run ids do not change. `configs/frozen-v2.yaml` (D-021 R33) carries the switch at C1.
- **A9's development default changes hash.** The 34 A9@r13x runs of 10-13 then describe the old default, and C1's reading
  goes to **Berk's open question 1** (the note): (a) re-read C1 on the exact frozen-v2 hash with A6F re-run in the same
  window — 68 runs ≈ 3.9 h; (b) A9 only, labelled cross-day against A6F@r13x of 10-13 — 34 runs ≈ 2.0 h; (c) read the
  old default for the go / no-go, with R1 / R2 as dated addenda. **Default if he does not answer: (a)** — the go / no-go
  should describe the configuration that is frozen, and D-021 / D-024 practice re-runs the control in the same window.
- The re-read runs **once**, on the hash `frozen-v2.yaml` carries, after D-026's verdict, so that a second adoption does
  not need a second re-read. crna and rail01 (fast generations: crna 855 / 299, rail01 383 per run; not in this test)
  are first seen there; their rows are reported against A9@r13x, labelled cross-day.

## Before the first timed run
**GPU checks (inside a lease, untimed; the note's "before the window"):**
1. **Flag state:** in an EA process started with the variant file, `cuDevicePrimaryCtxGetState` reports
   `CU_CTX_SCHED_BLOCKING_SYNC` after setup and before warm-up and t0, and `ea.summary` logs it; with the default file
   it reports today's flags. The `_Phases` events are blocking only under the switch.
2. **Spin check:** a long synthetic FJ call (or a dummy launch loop with a sync after every ~50-ms launch, as
   `ls_fj_gpu.py` does), at least 10 s of GPU work, the EA pinned to one core: **EA self-CPU per wall-second ≤ 0.25**
   under the switch (the note's "well below 1", made exact here), with the default path's figure printed beside
   (expected ≈ 1). A test that only shows nothing compiled is not enough. For fallback (b) the check instead pins a
   CPU-bound stand-in for the LP worker to the same core and requires the stand-in to get ≥ 0.8 CPU-s per wall-second
   while the GPU call runs; for (a), the same stand-in check (the renice is what gives the worker the core).
3. **Cold-cache test:** the EA with the variant file from an empty CUDA cache — no compile on the clock.
4. The repository's GPU tests (`pytest -m gpu`) on the merged code.

**CPU tests before the merge (no GPU):** every existing configuration keeps its hash (the four above and every other file
in `configs/`); every existing job file regenerates byte for byte; `make-jobs r1-bsync` writes the 20 runs for either
control, with run ids distinct from every other set's; the default path's behaviour is unchanged with the switch off;
the report script's legs, each with a test.

## Prediction (not a criterion)
Modelled from existing traces; interim; per instance only (the note makes no projection of C1's ratio):
- **ca: 0.394 → about 0.23–0.30** — about 0.23 if the host's non-repair time is all spin, about 0.27 if half of it is
  real Python host work (on ca that time is 1.1–1.9 s of each 1.6–3.0 s pre-phase generation, unmeasured); bo1 gave
  0.296. Against A9@boY: about −0.03 to −0.05.
- **sing44: −0.02 to −0.04**; **neos-960392: about −0.01 to −0.015.**
- **None on the three losses:** on toguru the LP needs only 6.3–6.8 s of CPU but is polled at the end of the uncapped
  init repair (37.8 / 45.8 s); chromaticindex and rail01 exceed the 60-s LP limit even on a full core.

## Risks
- **Wake latency.** Core 6's idle states have exit latencies of 42 / 231 / 433 µs under the performance governor. Once
  the LP is done and core 6 idles between syncs, every sync pays a wake-up; instances with fast generations pay it most
  (item 1,619–1,687 generations, comp21 248–295, crna 855 / 299, rail01 383). item and comp21 are guarded; crna and
  rail01 are not in the test.
- **The P evidence is one natural experiment** on one instance (ca, old harness, 2 seeds), though both seeds agree and
  the arithmetic closes.
- The gain is capped by generation boundaries; it overlaps boY on ca.

## Cost and code owed
Code small (about half a day including review), on branch `r1-bsync`, merged after the running pool ends (a pool
imports from the main checkout): the switch and the driver-API call, the blocking events, the flag read-back in
`ea.summary`, the LP worker's own stamps, the variant files (and the fallbacks' only if they apply), `make-jobs r1-bsync
--control r13x|boY`, `tools/d025_report.py`, and tests. Box: 20 runs ≈ 1.15 h plus the exclusions above. Needs Berk:
no; a normal window. R3's diagnostic ([[D-027 Same-day diagnostic of cuOpt's thread count and SCIP's presence (R3) — reading rules fixed before the runs]])
may share the window.

## Clarification and amendment 2026-10-14 17:4x — before any run (AgentRA, Claude Code (Anthropic Claude Opus 5.5))
An adversarial review of the build (Claude Code subagents; branch `r1-bsync`, build commit 61c552e, fixes in
91d38cd) compared the code with this record. Where they disagreed, the code now follows the record unless stated
below; the rest is fixed here, before any run. No `r1-*` run exists.

**Berk's answer to open question 1 (2026-10-14 16:0x, question panel):** **"Re-run C1 on frozen hash"** — option (a):
C1 is re-read on the exact frozen-v2 hash with A6F re-run in the same window (68 runs ≈ 3.9 h), once, after D-026's
verdict. The default above is now his decision.

**Names as built (the clarification "What is tested" promises):**

| item | record | build (binding from here) |
|---|---|---|
| the switch | "one new EA configuration field … in `config._LATE_DEFAULTS`" | `host.cuda_sched: auto \| blocking_sync \| yield` in a new section `host` (`HostCfg`), left out of the hash **as a whole** while every field is at its default — same effect: 1d12ad490dd0, 5c8b535fb245, 198e0d2ca540, c5b5e01856eb, f030e23909ad, 9ac4a93bf93b and every other file in `configs/` keep their hashes (pinned in `tests/test_cuda_sched.py`) |
| set, job file | `r1-bsync`, `jobs/r1-bsync.jsonl` | the same, for either control; the file is **not committed before D-023's verdict**; it is written then with `itrea make-jobs r1-bsync --control r13x\|boY` and its sha256 goes into that day's journal |
| treatment | `A9@bsync` / `A9@boY-bsync` | the same: `-3way-bsync.yaml` (EA 305c8902d763, run hash under ext 330a10a77556), `-3way-boY-bsync.yaml` (29e61aa41619 / b760e770cb34) |
| control | `A9@r1ctl` | the same label, running `itrea2-dev-rc-3way.yaml` (run hash 28e5a5e9c725 = A9@r13x's) or `-boY.yaml` (86b1e527db68 = A9@boY's in `backoff2`): the same job, a new run id. The build had re-used the earlier ids (A9@r13x, A9@boY); withdrawn |
| order | — | each block's two runs carry an explicit `position`: the control first at seed 0, the treatment first at seed 1 (the pool's Williams row for two arms fixed one order per instance x seed by a hash; sing44 ran the treatment first on both seeds) |
| fallback (a) | `-bnice.yaml`, set `r1-bnice`, renice +10 relative to the LP worker, no context flag | the same: `host.nice_after_lp_spawn: 10` = **+10 relative**: every thread of the host takes min(19, n + 10), n = the nice the worker was spawned at (it inherits n); recorded as `host_renice.target` / `.gap`; `-3way-bnice.yaml` (96a299330981 / d266e4aaecbe), `-boY-bnice.yaml` (5b1c4779c74b / b4f4af8376dd); control label `A9@r1ctl-bnice`. The build's first fallback `bsyncn` (blocking sync **plus** an absolute nice of 10) is withdrawn: against the control it changed two things and carried blocking sync's wake latency, which (a) must not |
| fallback (b) | `-ysync.yaml`, set `r1-ysync` | the same: `host.cuda_sched: yield` (f81d6b6209f0 / a08221cca653), `-boY-ysync.yaml` (666865eac174 / 22137f009005); timing events non-blocking; control label `A9@r1ctl-ysync` |
| a rerun | — | its own set (`make-jobs r1-bsync … --set r1-bsync-2`): a rerun under the same set would be skipped by the pool as done |
| report | `tools/d025_report.py` | written and tested: `itrea d025-report [--set r1-bsync\|r1-bnice\|r1-ysync] [--control r13x\|boY]`; on today's table it reads INVALID, 0 of 20 present, as expected |

**Validity, made exact:**
- "**The flag state**" reads **the variant's flag**: bsync — blocking sync set before the context existed, verified on
  the active primary context, still held at the end, timing events blocking; ysync — the same for yield, events not
  blocking; bnice — CUDA's default read back and `host_renice.gap` = 10 (a gap below 10 means the pool ran at nice > 9:
  INVALID); control — CUDA's default **read back** (`cuDevicePrimaryCtxGetState`, read-only, `cuda_sched.state` =
  "read", `sched_active` = "auto") and no renice. The build first recorded only the configured mode for the default; it
  now reads the flags back under "auto" too (every GPU run, before t0, no behaviour change).
- **Failed runs** are outcomes of their arm (D-002 §6, `--failed keep`, D-022's precedent) for the verdict; the letter
  (`exclude`) is printed beside. D-027 keeps the letter as its reading — a diagnostic with no verdict; the difference is
  deliberate. A figure a failed run cannot give (no EA summary) fails each gating leg that needs it.

**Before the first timed run, check 2 as implemented** (`tools/bsync_check.py`, `tests/test_cuda_sched.py`, marker gpu,
run without `nice`, `taskset -c 6`): 12 s per phase, ≥ 10 s of GPU work asserted; blocking sync: host self-CPU per
wall-second **≤ 0.25** alone and the stand-in > 0.8 beside it; yield (b) and the renice (a, `--mode auto --renice 10`):
the stand-in **≥ 0.8**; auto printed beside (expected > 0.8 alone, < 0.7 share). The build's first test (6 s, < 0.3,
no renice mode, yield unread) did not implement this check and was replaced. **Check 1** includes the default file (the
control: flags read back as "auto").

**Branches the verdict left open:**
- Hint present and the generation guard held but a P no-harm leg failed: NOT ADOPTED, no fallback trigger → the R1 line
  ends for C1.
- Both triggers fire (no hint with ca's LP share < 0.8 on both seeds, and the generation guard failed): **(a) first** —
  it carries no wake latency.
- **No chaining:** at most one fallback runs (because the spin check failed before the window, or because a trigger
  fired); its test is read under the same rules, and its NOT ADOPTED ends the R1 line for C1.
- D-026 waits for **the R1 line's final outcome**, fallbacks included.

**Amendment (rule changes before any run):**
1. **The P band on ca, sing44 and neos-960392 now gates** (same one-sided band as on item_placement and comp21).
   Why: D-024 rule 3 makes the guard against harm part of every widening, and the pooled hint alone lets a ca gain carry
   a loss elsewhere (ca −0.15 with sing44 +0.06 gives a ratio ≈ 0.90, "present", with sing44 beyond its band).
   ADOPT now needs validity, both no-harm legs on item_placement and comp21, the band on the three hint instances and
   the hint.
2. **The generation guard reads seed means against the control's spread:** Ḡ(T) / Ḡ(C) ≥ 1 − max(0.05, |G_C,s0 −
   G_C,s1| / Ḡ(C)), on item_placement and comp21 (it was G(T, s) ≥ 0.95 × G(C, s), paired by seed). Why: the record
   assumed item's within-day spread (4 %: 1,619 / 1,687 in A9@r13x, 1,958 / 1,881 in A9@bo1s); comp21's is 9–17 %
   (248 / 295, 283 / 258). Generations are not reproducible within a seed (timing), so pairing removes little; under a
   normal model with those spreads the paired rule fails with no effect in about 20–25 % of tests on item_placement and
   about 60 % on comp21 — it would fire fallback (b) on noise — against about 5 % and 17 % for the amended rule (rough:
   two within-day pairs per instance). Fallback (b)'s trigger reads the amended guard. **The trade-off, stated:** like the
   P band, a control with a wide seed spread widens the guard — a 20 % spread sets the threshold at 0.80, so a real
   15 % wake-latency loss on comp21 would pass; this change is AgentRA's own (the review asked only for the false-fail
   rate) and may be overturned before any run.
3. **A harm leg at the C1 re-read** (the widening): the re-read's A9 (the frozen-v2 candidate) against A9@r13x of 10-13,
   labelled cross-day, per development instance: harm iff d_i < −band_i, d_i = (P(A9@r13x) − P(new)) / max(both,
   10⁻³) on seed means, band_i = max(0.05, the larger seed-0 vs seed-1 bounded difference of either arm) (D-023's
   `band_i`). crna and rail01 — not in this test, the most exposed to wake latency — are read there by this leg. Any
   harm goes to Berk with the numbers beside the go / no-go, and the frozen-v2 candidate is not presumed: he decides
   which configuration freezes (with R1 and R2 both adopted the re-read cannot say which change caused it). Code owed:
   the re-read's report (not written yet).

**Corrections (no rule change):**
- Null replicate: 1.772 / 1.793 = **0.988** (the record says 0.989 from 1.773 / 1.792; the run table: A9@bo1s 0.3927,
  0.3959, 0.4568, 0.4117, 0.0642, 0.0507; A9@r13x 0.3912, 0.3959, 0.4412, 0.4490, 0.0654, 0.0502). r13x seed 1 /
  seed 0: 0.895 / 0.898 = 0.997 (unchanged).
- "**In practice this hint is ca's**" overstates it: by this record's own predictions sing44 (−0.02 to −0.04) and
  neos-960392 (−0.01 to −0.015) reach up to −0.055 on the seed-mean sum, beyond the threshold 0.05 × 0.896 = 0.045, so
  the hint can open without ca at the top of those ranges. It is ca's in most of the predicted range.
- **Expected false-fail of the P leg on item_placement:** same-configuration A9 runs give P = 0.362, 0.355, 0.261,
  0.358, 0.370 (SD 0.045); with no effect the one-sided band max(0.05, the control's spread) fails in about 10 % of
  tests (simulated, normal model). The rule stays (it errs towards NOT ADOPTED); the report prints the margins.

## Accepted 2026-10-14 18:3x (AgentRA, main session), before any run
The 17:4x amendment of the generation guard (seed means against 1 − max(0.05, the control's relative spread), replacing
the paired ≥ 0.95 rule) is the build's own change beyond its review, so it is confirmed here before any r1-* run: the paired
rule would fail on noise in ~20–25 % of tests on item_placement and ~60 % on comp21 (within-day generation spread 9–17 %) and
fire fallback (b) on noise; the cost — a control spread of 20 % lets a 15 % loss pass — is stated above. **The control is
A9@r13x** (`make-jobs r1-bsync --control r13x`): D-023 did not adopt boY (its Result, 18:1x). Noted, not a rule change:
item_placement's P moves a lot between days with the same EA behaviour (A9@bo1s 0.218 on 10-14, A9@r13x 0.364 on 10-13);
D-025 reads item against its same-window control (A9@r1ctl), which is why the control is re-run rather than borrowed.

## Result of the bsync test (2026-10-15 08:1x, AgentRA): NOT ADOPTED — fallback (b) ysync fires
Set `r1-bsync`, pool `20261015-071709-2205151` (07:17–08:06): **20 of 20 completed**, 0 failed / void; control A9@r1ctl
(= A9@r13x's configuration, run hash 28e5a5e9c725), treatment A9@bsync (330a10a77556). GPU checks before the pool (07:08–07:17):
`bsync_check` host self-CPU per wall-second auto 0.9998 → blocking_sync 0.0083, the core's competitor 0.5187 → 0.9989.
`itrea build-runs`, `itrea d025-report` (VALID, 20 of 20):
- **Mechanism (reported):** EA self-CPU per wall-second 0.017–0.041 under bsync on ca / sing44 / 960392 / comp21 (control
  0.82–1.00); item_placement 0.19 / 0.24. The LP worker's CPU share 0.93–0.96 (control 0.50); its finish time ca 35.0 / 35.1 s →
  18.9 / 19.3 s, sing44 12.9 / 12.2 → 6.2 / 6.1 s, neos-960392 4.7 / 4.7 → 2.5 / 2.5 s.
- **Hint PRESENT:** ΣP(T) / ΣP(C) over ca, sing44, neos-960392 = 1.3563 / 1.7729 = **0.765** (≤ 0.95); per instance ca 0.633
  (P 0.393 → 0.248), sing44 0.882 (0.436 → 0.385), neos-960392 0.777 (0.058 → 0.045). Hint-instance bands hold.
- **No-harm FAILS:** generations on seed means item_placement 0.824 (threshold 0.921), comp21 0.818 (0.950; seed 1 153 vs 272);
  P on comp21 +0.060 (band 0.050); item_placement P −0.014 (holds).
- **Verdict: NOT ADOPTED** (failing: the generation guard and P no-harm on item_placement / comp21). Fallback triggers: (a) no
  (ca's LP share 0.945 / 0.943 ≥ 0.8); **(b) fires** (the generation guard — wake latency, the risk this record named).
- **Next, as fixed above:** fallback (b) `ysync` (CU_CTX_SCHED_YIELD) tested the same way: `itrea make-jobs r1-bsync --control
  r13x --variant ysync` → `jobs/r1-ysync.jsonl` (20 runs, set `r1-ysync`, labels A9@ysync / A9@r1ctl-ysync; sha256
  `52e86e06fd0c8258f2e02ca25ce2dfef7b6c38262675fa69ffd8f79b8bbc042c`), queued 08:1x in the same window. Its GPU check (07:1x): the
  host spins (0.99 per wall-second) but its core's competitor gets 0.9989; an A9-layout ysync run gave the LP worker 0.933. No
  chaining after (b). D-026 waits for (b)'s outcome.

## Result of fallback (b) ysync (2026-10-15 11:2x, AgentRA): NOT ADOPTED — the R1 line ends for C1
Set `r1-ysync`, pool `20261015-100013-3024118` (10:00–11:09): **20 of 20 completed**, 0 failed / void. `itrea d025-report
--variant ysync --set r1-ysync` (VALID):
- **Hint PRESENT: 0.834** (1.4332 / 1.7176); ca 0.644 (P 0.392 → 0.253; LP done 37.3 / 38.3 → 20.0 / 20.2 s), sing44 1.026
  (0.408 → 0.418; LP done 12.6 → 6.9 / 7.1 s), neos-960392 0.777 (0.059 → 0.045). Hint-instance bands hold.
- **No-harm:** comp21 holds now (generations 0.980 vs threshold 0.898; P −0.001); **item_placement's generation guard FAILS**
  (means ratio 0.759 vs 0.921: seed 0 1,483 vs 1,367 = 1.085, **seed 1 678 vs 1,480 = 0.458**); item's P +0.099 within its band 0.230.
- **Verdict: NOT ADOPTED**; no fallback of a fallback — the R1 line ends for C1, as this record fixed.
- **Stated, not acted on:** the failing cell is one run — item_placement ysync seed 1, the pool's first run, started 10:00:14 one
  second after M4's 600-s pool ended (load average 5.75 at its gates; GPU clock 2,392 MHz throughout, no foreign GPU process, not
  voided). It ran 70 GPU launches per second against 122 on seed 0 and 112–124 in the control. Whether that is an effect of
  `yield` or a start-of-pool disturbance cannot be told from one run; the rule is applied as written.
- **Across both tests:** ca's gain is the same mechanism twice (P −0.145 / −0.139, the LP worker ~2× earlier); the guard failures
  sit on item_placement (bsync 0.824, ysync 0.759 driven by one seed) and comp21 (bsync only). Any further R1 test is a new
  decision, written before its runs, that reads the guards on every seed run so far (0–1 here included).
