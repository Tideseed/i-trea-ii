# Agent companion — detail and history

**What this is.** The long form behind `AGENTS.md` at the repository root. `AGENTS.md` keeps every rule, the state at
a glance and the code map in short form; this note keeps the detail it no longer carries: the dated status narrative,
the full notes on each module and tool, every configuration file and every job writer, and (§§ 7–10) the current-state,
readings, decisions and changelog tables.

**Provenance.** §§ 1–6 were moved here on 2026-10-16 from `AGENTS.md` as it stood at commit `3cbc7a1` (10:25 CEST;
`AGENTS.md` unchanged since d58cf7f, 10:03). The sentences are unchanged; they are only split under headings and into
bullets. §§ 7–10 were moved later the same day from the restructured `AGENTS.md` (working tree, not committed), after
its comp21 figure was corrected; only their headings are numbered and one clause is added to the changelog's last row.
This note is a snapshot: newer facts go to the decision records, the [[Development log]] and `AGENTS.md`, not here. The
four-seed result (D-031) stays in `AGENTS.md`. Words such as "below" and "section below" inside the moved text refer to
`AGENTS.md`'s layout at the time; quoted section names ("Terms", "Setting") are `AGENTS.md`'s, and "the detail note" is
this note.

**Other snapshots of the same day:** [[Home — long form]] (the old front page) and [[Project background]] (the old
README). They have drifted once already (the comp21 figure, § 1.4). `AGENTS.md` and [[Home]] hold the current state.

| § | what | short form in `AGENTS.md` |
|---|---|---|
| 1 | status narrative as of 2026-10-16 morning | "At a glance" (the other short forms moved on to §§ 7–10) |
| 2 | cuOpt in detail (versions, #2073, 26.10) | "Setting" |
| 3 | test-suite history and test coverage | "Map" |
| 4 | full notes on the long code-map rows | "Map" |
| 5 | configurations: the full record | "Map" → "Configurations" |
| 6 | job writers (`make_jobs.py`): the full record | "Map" → "Job writers" |
| 7 | current state and the open issues (moved from `AGENTS.md`) | "At a glance"; pointer table "State, readings and decisions" |
| 8 | interim readings and how to read them (moved) | "At a glance"; "Four-seed result (D-031)" |
| 9 | development decisions D-011 … D-032 (moved) | pointer table |
| 10 | changelog (moved) | "Changelog" (pointer) |

## 1. Status narrative as of 2026-10-16 morning

### 1.1 Phase, pre-registration, development set and defaults
- **Status (2026-10-16): development (phase 1), which ends at checkpoint C1 on Sat 2026-10-17 (Berk 2026-10-15: "Assume
  we are wrapping up phase 1 on Saturday"; D-019 addendum); no confirmatory run yet.**
- The pre-registration D-002 v3.1 is binding since the tag `prereg-v1` (2026-10-06).
- The SCIP screening's stage 1 is complete (324 candidates, seed 900); stage 2 is deferred (D-010 §5), so the hard pool
  is not frozen (#1).
- The algorithm is developed openly (D-010) at 120 s on 17 instances — the development set (9, D-010, since 2026-10-06,
  `datasets/dev_set.csv`) and the class probe (8, D-014, since 2026-10-07, `datasets/classprobe_set.csv`), both
  permanently excluded from the test set; every development result is descriptive and lives in
  `vault/30-results/Development log.md`.
- Development defaults: **A3** (SCIP ↔ EA) = `configs/itrea2-dev-rc-ld.yaml` (lp_decode adopted, D-017); **A9** (SCIP ↔
  EA ← cuOpt, D-015) and its matched control A3x = `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` since 2026-10-15 (EA hash
  `835c7c5a249c`, A9 run hash under ext `f345de0ab936`) = the old default `configs/itrea2-dev-rc-3way.yaml` + lp_decode
  gated after the EA's LP worker (D-026) + the EA's CUDA context at `yield` (D-029); by D-032's rule it is the
  configuration that freezes for phase 1 (`configs/frozen-v2.yaml` is written on Saturday).
- **`make_jobs.ITX_A9_CONFIG` still names `-3way.yaml`** (the old default) until a `src/` change after the running pool:
  a writer that uses it (A9 in `itx-m1` … `itx-m5`, `itx-r13`, `itx-r13f`; the `r13x` control of `r1-bsync` and `r2-ld`)
  writes the old default; `itx-v1` takes `configs/frozen-v2.yaml` and `c1r-rerun` its pinned files.
- Built and not adopted: Rank 1 by cuOpt's clearing rule (D-016), A9's GPU back-off (D-018 stage 1, 2026-10-11: every
  outcome rule held, the mechanism failed on M1 / M3 / M5; an analysis bug — `_band` read the seed spread of the old
  control `A9@bo0` — was fixed before the verdict) and its phase Y alone (D-023, 2026-10-14: failed M2, M5, T, G — the
  back-off line is closed for v1).
- A9 pilot (seed 0, 7 instances, descriptive): vs SCIP alone 7 better / 0 worse, mean d +0.68 [+0.42, +0.90]; vs cuOpt
  alone 3 / 4, −0.07 [−0.42, +0.25].

### 1.2 External harness
- **External harness.** D-020 (GAMS's MIPfeas archive, downloaded on Berk's yes; GAMS's primal integral ported exactly —
  `gams_integral` equals GAMS's own function on all 3,495 of its traces) is narrowed by **D-021, our own standard
  ITX-1**, whose [Berk] items Berk accepted on 2026-10-11 (section below).
- Its footprint F is accepted on M2's soak (steady p95 66–73 °C, maximum 76 °C over 4 runs of 600 s); M3 fixed the "does
  not fit F" list (eilA101-2, nw04, supportcase6; decision set 215); phase 1 and phase 2 are merged (phase 2: 6f0339d,
  2026-10-12; 1,009 CPU tests passed on main after the merge; on 10-13 the runner's GPU tests passed — 4 passed, 1
  skipped — and a cold-cache EA test).
- The M4 calibration (A6F on D-020's 30 instances, 600 s) is complete (30 of 30, 2026-10-15) and reads **OUTSIDE** the
  band [0.870, 1.150]: R = 1.313 [1.091, 1.618] on the call clock (launch clock 1.963 [1.606, 2.408]), feasible 27 vs
  GAMS 27, late-clock check passed (median −0.119 s over strata 0–1) — with the clock checked, the difference is the
  footprint F (5 threads + the GPU against GAMS's 24-thread whole box) and the machine, so v1's targets stay anchored on
  our own A6F (result note `vault/30-results/2026-10-11 D-021 measurements M1–M3 and the interim calibration.md`,
  addendum 2026-10-15; the interim 19-of-30 reading, R 1.390, is superseded).
- A0 (SCIP alone, `itx1`, 600 s) on the 81 instances v1's decisions read (D-024 (f)): 80 completed + 1 failed (s100:
  SCIP did not stop at 600 s, killed at 701 s; kept as an A0 failure; journal 2026-10-14 21:48).

### 1.3 Staged testing (D-024)
- **Staged testing (D-024, Berk 2026-10-13).** Performance runs first; targeted runs where a change can act, wider only
  after a hint whose threshold is fixed before the runs (futility-only: a hint gates widening, never declares success).
- Berk accepted v1's staging: stage 1 = the 20-instance panel plus the 5 H_cu-unseen instances where cuOpt finds no
  solution, × {A9, A6F}, hint = ratio of means ≤ 0.95; the guard-only stage runs whatever the hint says; A0 only on the
  81 instances a decision reads (only the 25 stage-1 instances if there is no hint, D-024 (f)).

### 1.4 Phase 1's last week (2026-10-13 → 10-16)
- **Phase 1's last week (2026-10-13 → 10-16; each verdict by its record's rules, fixed before any of its runs was read —
  D-022's second amendment came with 7 of the 58 runs of its first pool done, none read; development, n = 2–6 seeds,
  descriptive).**
- Adopted: **D-022** (74 runs) — A9 under ext (R13 + R14): valid, no harm on the 3 instances where cuOpt's own 1e-4 stop
  occurs; mechanism met only vacuously (SCIP proves first), so the adoption rests on D-021's fairness and GAMS fidelity;
  **D-026** (28 runs) — lp_decode inside A9 after the LP worker: hint 0.758 (≤ 0.90; item_placement P 0.330 → 0.198,
  sing44 0.473 → 0.411, lower on all four seeds of both), mechanisms and no-harm legs held; **D-029** (21 runs) —
  `yield`, guards over seeds 0–5 with D-025's failing seed kept: generations item_placement 1.020, comp21 1.152 (≥
  0.90), P +0.016 / +0.015 (≤ +0.05), ca at seeds 2–3 0.675 (≤ 0.95) — thresholds set with seeds 0–1 (8 of its 28 read
  runs) already known, item_placement's generation bar loosened from D-025's 0.921 to 0.90 (D-029 amendment B1–B2); at
  D-025's bars the verdict is the same (item 1.020 > 0.921; comp21 1.152 > 0.898 / 0.950), so this qualifies the claim,
  not the result.
- Not adopted: **D-023** (66 runs: chromaticindex d +0.030, needed +0.30; cuOpt's first incumbent inside A9 63.47 /
  63.41 s vs A6F 10.87 / 10.52 s); **D-025** (R1, 20 runs each) — bsync (hint 0.765; generation guard failed on
  item_placement and comp21, comp21 P +0.060 > band 0.050) and its fallback ysync at seeds 0–1 (hint 0.834;
  item_placement's generation guard failed on one seed, 678 vs 1,480), reopened by D-029.
- Diagnostics: **D-027** (22 runs) — rule 4 EA PRESENCE: cuOpt's first incumbent on chromaticindex1024-7 at 9.80 / 7.33
  s alone on F, 9.24 / 8.21 s on 3 threads, 6.53 / 9.65 s beside SCIP, vs 62–66 s inside A9; rule 1 closed the layout
  line, rule 3 ruled SCIP out; **D-028** (4 runs) — not R13 + R14 (A9@boY under r13-off 63.25 / 63.24 s vs L = 19.82 s).
- chromaticindex stays a known v1 loss; CUDA MPS is not for v1 (Berk 2026-10-15).
- **C1 readings** (A9 vs A6F in the same window, seeds 0–1; hint = ratio of mean P(120) over the fixed 11 ≤ 0.95,
  futility-only; guard = mean bounded d over the 17 ≥ −0.05): old default (D-022's Result, 10-13) 0.827 / +0.063;
  **frozen-v2 candidate (D-030, 68 runs, 10-15/16) HINT 0.809, GUARD +0.050**, read with cuOpt alone's SIGSEGV on
  comp21-2idx at both seeds kept as A6F outcomes (the reading gains from them, A6F P 1; left out, hint and guard are
  INCOMPLETE) [Corrected by D-030 § Correction 2026-10-16 10:0x: the saved report scores the two crashed runs at P 0.776
  / 0.782, truncated at the crash, not 1.]; D-030's harm leg (cross-day, against 10-13's A9@r13x) fired on glass4 (d_i
  −0.088 vs −0.057; krka at floor) → **D-032** (12 runs, glass4 × seeds 0–3 in one window: old default 0.2176, candidate
  0.2233, lp_decode alone 0.2177; d −0.025 ≥ −0.057) → **DRIFT**: the candidate freezes for phase 1 by D-032's rule
  (fixed before its runs); D-031's four-seed harm reading may still put which configuration freezes to Berk.
- The go / no-go on C1 is Berk's.
- Carried to phase 2: #26 (not built), #35 two-way (v1.1, Berk 10-14), CUDA MPS (v1.1, Berk 10-15), chromaticindex as a
  known loss; due before prereg-v2, not Saturday: #36 and #32's test-set size.
- Saturday: `configs/frozen-v2.yaml`, the phase-1 results note, checkpoint, docs, issues, the go / no-go (D-019
  addendum).

### 1.5 cuOpt 26.10, open questions, the v1 target
- **cuOpt 26.10:** NVIDIA/cuopt#2073 is fixed by PR #2098 (merged 2026-10-12, in 26.10; Berk confirmed 10-13) and
  verified on the nightly `26.10.0a294` in `~/venvs/cuopt-2610n` (presolve stays on with a SET callback; injected
  solutions accepted and exact; invalid offers rejected); limitation: cuOpt polls SET callbacks only when its GPU
  population records a new best. A9 two-way through it is #35 (not started).
- **Open:** our exact checker accepted incumbents below MIPLIB's proven optimum on glass4 (cuOpt rejects that point) and
  mas74, and rejected cuOpt incumbents on snowy (1) and neos-933966 (3) — a tolerance mismatch to investigate (#36).
- **v1 target (D-019, Berk 2026-10-10):** checkpoint C1 on Sat 2026-10-17 (was Fri 10-23; D-019 addendum 2026-10-15;
  development closes on cuOpt 26.08), v1 at the end of November 2026 (aim Fri 11-27, hard end Mon 11-30) on cuOpt 26.10,
  daytime windows only; targets in the next-but-one section.
- Newest checkpoint: `vault/90-checkpoints/2026-10-14-2350.md`.
- Every hypothesis and target below is pre-registered or planned, not a finding.
- Do not write, in any file, that I-TREA II "beats", "improves on" or "matches" anything until the corresponding
  experiment note in `vault/30-results/` says so with its numbers.
- (History: on 2026-10-06 01:4x the status read "built and tested on toy models; no commit, no tag, no experiment, no
  results" — 413 passed, 2 skipped.)

## 2. cuOpt in detail (from the setting)

- Software: Python 3.12.3; NumPy 2.4.6, SciPy 1.18.1; CuPy 14.2.0 (GPU) and NumPy (CPU) as one code path
  (`itrea.backend`); solvers SCIP 10 (PySCIPOpt 6.2.1, primary partner), HiGHS 1.15.1 (replication partner, after its
  user-solution round-trip test; also the EA's own LP and the lp_decode worker), NVIDIA cuOpt 26.08
  (`cuopt-cu13==26.8.*`).
- cuOpt is no longer only a control: alone it is arm A6, and since D-015 (2026-10-07) it is the third process of arm
  **A9** (SCIP ↔ EA ← cuOpt), where it streams its incumbents and bound to the EA on its own channels and the EA passes
  them on to SCIP.
- cuOpt never receives a solution mid-run: on the installed 26.08 its `SetSolutionCallback` disables its presolve
  (feature request NVIDIA/cuopt#2073, filed 2026-10-07).
- **Fixed in 26.10** by PR #2098 (merged 2026-10-12; Berk confirmed 2026-10-13) and verified on the nightly
  `26.10.0a294` in the separate venv `~/venvs/cuopt-2610n` (`tools/probe_cuopt_setsol.py`): presolve stays on, injected
  solutions in original order are accepted within 1–6 ms and echoed exactly, infeasible offers are rejected; cuOpt polls
  SET callbacks only when its GPU population records a new best (crna: 11 calls, all at 0.70–0.71 s; sing44: first call
  at 28.3 s).
- A9 stays one-way until a two-way arm has its own record and test (#35).
- A MIP start at launch is used with presolve on (verified on glass4 2026-10-07, ~10 s overhead per restart), but no arm
  uses it yet.
- **v1 runs on cuOpt 26.10** once it is released (separate venv, API check, A6 re-baselined on the 17 instances; #31);
  if 26.10 is not out by Fri 2026-11-06, the v1 runs proceed on 26.08 and 26.10 follows as an addendum (D-019; to be
  confirmed with Berk then).
- Numba 0.64.0 (the CPU feasibility-jump reference); redis-py 8.1.0 for the exchange platform; pandas 3.0.3 / PyArrow
  23.0.1 for the derived tables. All pinned in `pyproject.toml`. One entry point: `itrea <command>` (`src/itrea/cli.py`;
  `itrea --help`).

## 3. Test-suite history and test coverage (from the map's preamble)

- Status: **core** = written before the design critique of 2026-10-05 and revised after it; **built** = written by build
  agents from 2026-10-06 on (the first build 00:0x–01:2x; development modules through 2026-10-09) and covered by the
  test suite (CPU 816 passed and the GPU tests of back-off, lp_decode, elite and EA run 40 passed on 2026-10-09; CPU
  1,009 passed on 2026-10-12 after D-021 phase 2; non-GPU 1,256 passed, 1 xfailed on main after the R1–R3 merges of
  2026-10-14; 1,339 passed on branch `c1-rerun` before its merge on 2026-10-15; GPU suites 45 passed, 1 skipped on
  2026-10-15).
- Caveat: `ea/population.py`, `ea/evaluate.py`, `ea/frontier.py`, `exchange/redis_mailbox.py`, `trace.py` and
  `problem.py` have no test file of their own — they are exercised through the engine, EA-process and runner tests
  (default settings); `ea/ranks.py` has `tests/test_ranks.py` and `tests/test_elite.py`, the file mailbox
  `tests/test_review_mailbox.py`.
- Check a non-default branch before relying on it (`vault/80-reviews/2026-10-05 design critique — response.md`).

## 4. Full notes on the long code-map rows

Each entry is the row's whole "what" cell as it stood; the status column is given after the path.

### `src/itrea/ea/lp_decode.py` — built

the LP decode (`repair.continuous: lp_decode` = auto + a HiGHS worker process that fixes the integers of a few
candidates per generation and solves the LP over the continuous variables; D-017, issue #27): pure selection / budget /
screen functions and the `LPDecoder` worker; the runner refuses it for A9 / A3x (one EA core) unless the worker is gated
after the EA's own HiGHS LP worker (`repair.lp_decode_gate: lp_worker` with `seeding.lp_source: highs_async`, nice
`repair.lp_decode_nice` over the host — R2 of the 2026-10-14 C1 diagnosis, D-026, built 2026-10-14 in branch `r1-bsync`,
review fixes the same day; run in D-026's test 2026-10-15 and adopted inside A9, D-026) — built 2026-10-08, review fixes
2026-10-09 (`vault/80-reviews/2026-10-08 lp_decode review (workflow).md`), GPU tests passed; run in D-017's test (24
runs, 2026-10-09) and **adopted as the A3 development default**; inside A9 it runs on the EA's core, gated after the LP
worker (D-026 ADOPT, 2026-10-15; #30)

### `src/itrea/ea/backoff.py` — built

A9's GPU back-off (D-018): a contribution-gated duty cycle — phase Y (reduced duty until the first incumbent anywhere),
then a ladder of duties stepped down after a state-dependent patience in busy seconds and restored by material leads /
SCIP-confirmed assists; a host-side token bucket the engine sleeps on between generations and between FJ launches
(`backoff`, off by default, inert unless `exchange.from_cuopt`; modes active / shadow / forced) — built 2026-10-09;
stage 0 (18 runs: forced duties ½ / ¼ / ⅛ against shadow bo0, A6 and a same-day A2) complete 2026-10-09: PROCEED; stage
1 (bo1 vs bo1s, 84 runs) 2026-10-11: NOT ADOPTED (mechanism); D-023 tested phase Y only
(`configs/itrea2-dev-rc-3way-boY.yaml`: patience unbounded): NOT ADOPTED 2026-10-14 (failed M2, M5, T, G)

### `src/itrea/ea/engine.py` — built

the generation loop, exchange in/out, genealogy, frontier updates; since 2026-10-14 (branch `r1-bsync`) the EA host's
settings (`host`, R1 / D-025: the CUDA scheduling flag before the backend exists — under CUDA's default the active flags
are read back —, blocking timing events, the renice +N relative to the LP worker after its spawn), the LP decode's gate
(R2 / D-026), the warm-up's >8-part concatenate, and in every run the LP worker's own clock and CPU in the `lp` event,
the on-clock CPU (`cpu_clock`), `ls_launches` and `concat_parts_max`

### `src/itrea/ea/cuda_sched.py` — built

R1 of the 2026-10-14 C1 diagnosis: the EA's primary CUDA context's scheduling flag (`host.cuda_sched`: blocking_sync /
yield) through the driver API (`libcuda.so.1`: cuInit, cuDeviceGet, cuDevicePrimaryCtxSetFlags_v2), refused once the
context is active, verified with cuDevicePrimaryCtxGetState after CuPy created it; under "auto" the flags are only read
back (`read_state`, D-025's control); CPU tests use a fake driver — built 2026-10-14 (branch `r1-bsync`), GPU checks
passed 2026-10-15 07:08–07:17 (`test_cuda_sched.py` 5 passed); `yield` adopted by D-029 (2026-10-15), `blocking_sync`
not (D-025)

### `src/itrea/experiments/pair.py` — built

one run: launch the arm's processes, start barrier, stop, every exit code checked, `run.json`; A9's three processes and
its matched control A3x (D-015); arm A6m (cuOpt as GAMS ran it for MIPfeas: 24 threads, MIP gaps 0, all 20 cores on the
whole-box lane `full`; D-020); ITX-1 (D-021): arms A6F / A2F on F (the GPU lane's cores), the job-level `harness`
(preset `ext`: cuOpt gaps 0, thread variables unset, RMM 24 GB separate from the EA's pool, host 24 GB, the project's
CUDA JIT cache), named scopes `itrea-run-<run-hash>-<role>-<epoch>`, the void causes `thermal_watch`, `gpu_cap`,
`cuopt_memory`, `compiled_on_clock`; phase 2 (2026-10-12): R12's stop at the proven optimum (`RefMonitor`: tails every
process's `.part` records, the earliest qualifying record over every process, `ref_reached`, the exact check
`check_ref_trigger`, void `ref_rejected`), cuOpt's SIGKILL after a definitive outcome, the optional harness switches
`ref_stop` / `cuopt_kill_on_end` / `scoped_hash` (`hashed_harness`) and preset `itx1`; R2 of the 2026-10-14 C1 diagnosis
(D-026, branch `r1-bsync`): lp_decode in A9 / A3x is accepted only with `repair.lp_decode_gate: lp_worker` and
`seeding.lp_source: highs_async`

### `src/itrea/experiments/pool.py` — built

a job file under every D-002/D-003 control: Williams arm order per block, blocks never split, gates per run; layouts
`pair` (GPU lane g0 + `--cpu-lanes`), `screen` (nine X925 lanes), `full` (one lane, the whole box: A6m only, serving
check implied, D-020's `--cuopt-mem-gb 24 --gpu-mem-gb 36`); D-021: the fan-governor start gate (`--no-governor`),
`--pause-c`, `--cap-floor-mhz`, R26's breaker (`--max-heat-voids`), the new void causes rerun once then skipped,
explicit `position` order, per-treatment memory arithmetic in `--dry-run`; phase 2: the job `priority` (block groups),
R33's gate (external EA jobs need `prereg-v2` and `configs/frozen-v2.yaml` registered with it; external jobs a clean
tree), the `ref_rejected` rerun with the stop off (kept on resume)

### `src/itrea/analysis/build_runs.py` — built

run directories → `datasets/runs.parquet`, `datasets/incumbents_checked.parquet`; MIPfeas columns `pt_gams` (call
clock), `pt_gams_launch` (launch clock) and `startup_offset_s` (D-020); D-021's run-record columns appended (fan state
and cap, ACPI at start and during, `void_info`, CUDA compile flag, `stop_kind`, R12's `ref_stop_*`); a `ref_reached` run
reads like a run to T

### `src/itrea/analysis/itx_report.py` — built

D-021's external analysis (`itrea itx-report`): completeness, target (read only on the complete H_cu-unseen), minimum
(ε-floored guard + solutions found; basis full, or R31's stratified estimates while T3 is incomplete, else incomplete),
the computed race and the Holm secondaries, BCa beside (`--bca`), geo-means, D-019's literal readings, H_cu at P >= 0.5
and live A2F vs the race beside, published rows on our subset, both clocks with the start-up cost, every decision again
without R33's exposed instances; only runs with the registered settings (itx1; A9 on `configs/frozen-v2.yaml`,
`--frozen`); `--m6` M6's lanes

### `src/itrea/analysis/mipfeas.py` — built

D-020's MIPfeas calibration (`itrea mipfeas-report`): A6m against GAMS's published cuOpt 26.08 GB10 row — paired R with
a bootstrap interval on the call and launch clocks, feasible counts, strata, first-incumbent differences, GAMS's own
GB10 26.04 / B200 rows, verdict PASS / FAIL / INCOMPLETE; since D-021 also `--arm A6F` (M4) with the late-clock check
(one median of first incumbent ours − GAMS over strata 0–1 ≤ +0.3 s, per stratum reported), only runs under the ext
harness at F counted, and D-021's reading INSIDE / OUTSIDE / CLOCK-FAIL; the GAMS archive's readers (`read_gams_trc`,
z*, published P) and D-020's sample

### `src/itrea/analysis/backoff.py` — built

D-018's stage-0 calibration rules and stage-1 criteria (M1–M5, outcome, fallbacks) applied to the run table and run
directories, with a verdict (`itrea backoff-report --stage 0|1`); the sanity rule gates on the measured duty only and
reports the lane-steps ratio (D-018's second stated deviation; `--steps-gate` = the rule as first written); stage-1
control `A9@bo1s` — stage 0 read 2026-10-09; stage 1 read 2026-10-11 (`--failed keep`); `--stage 1y [--harness]` applies
D-023 as amended 2026-10-13 to set `backoff2`: the 11 instances / 66 runs (T, G, the ladder and the mean guard on the
nine, F one-sided on krka / fhnw, M1–M5 on all 11; G's noise = no sleep in phase Y, "still beats A6F" binds; other
instances and seeds named as outside, read by no rule; "n of 66")

### `tools/d022_report.py` — built

D-022's rules on the run table (`itrea d022-report [--followup]`): the second amendment's rule on the 3 instances where
R13 can act (validity incl. each arm's harness, the mechanism, a one-sided floor / band no-harm leg; no mean), and C1's
interim hint (ratio of means ≤ 0.95 on the fixed 11) and guard (mean d ≥ −0.05 over the 17) vs A6F, plus SCIP alone in
D-013 terms

### `tools/d027_report.py` — built

D-027's rules (the C1 diagnosis's R3, fixed before any run) on set `r3-diag`
(`itrea d027-report [--failed both|exclude|keep] [--json F]`): validity as one gate (22 of 22 with P(120), harness ext,
each run's layout — cores, `--cpu-threads`, X925 6 idle, no EA — read from its run.json, no compile on the clock;
INVALID names the leg); rule 1 the thread effect A6@r3 vs A6F@r3 on toguru and doon (band max(0.05, the larger raw seed
spread); present → the layout question to Berk, else closed for v1), rule 2 comp21's part 2 from cuOpt's final objective
in A2@r3 (precondition A6@r3 ≤ 133 on both seeds; ≤ 133 → the EA's presence remains, the MPS question to Berk as one
candidate), rule 3 SCIP beside cuOpt on toguru and doon, rule 4 (18:3x) cuOpt's first incumbent on chromaticindex1024-7
(`cuopt_t_first`, none = 120 s; L = 2 × max(A6F@r3): A6@r3 ≤ L on both seeds, then A2@r3 ≤ L → EA PRESENCE, the
GPU-sharing / CUDA MPS question to Berk, ≥ 50 s → SCIP PRESENCE; A6@r3 ≥ 50 s on both → 3-THREAD LANE UNDER EXT, the
layout question to Berk; else inconclusive at n = 2; prints the six t and L); A6F@r3 vs A6F@r13x beside on toguru and
doon — read 2026-10-15 09:2x: rule 4 EA PRESENCE, rule 1 closed, rule 3 SCIP ruled out, rule 2 not read (D-027's Result)

### `tools/c1r_report.py` — built

D-030's C1 re-read on set `c1r-<a9>` (`itrea c1r-report --a9 ld0|ysync-ld0 [--failed both|exclude|keep] [--json F]`):
validity as one gate (68 of 68 with P(120), harness ext, each run id = its job's, no compile on the clock — CuPy only on
A9 —, the two runs of a block in one window, each A9 run's CUDA flag as the candidate's file writes it (gating since
D-030's amendment of 2026-10-15); INVALID names the leg, a missing run alone INCOMPLETE); C1's hint (ratio of means A9 /
A6F over the fixed 11 ≤ 0.95) and guard (mean bounded d over the 17 ≥ −0.05) by D-022's own functions; D-025 amendment
3's harm leg (the new A9 vs A9@r13x of 10-13 per instance, cross-day; band_i = max(0.05, the seed spread of A9@r13x or
of A6F@r13x) — D-023's `band_i` as D-030's amendment reads it, never the treatment's — so each instance's threshold is
fixed by 10-13's runs; harm CANNOT fire on rail01, cryptanalysiskb128n5obj16, doon and crna); beside, not criteria: the
other readings' bands, D-022's floor marker (|ΔP| ≤ 0.005), the null lines (the same rule on A6F@c1r vs A6F@r13x and on
the same-configuration A9 replicates of 10-15: 2 of 15 instance reads fire on noise), the cannot-fire instances' seeds
and generations, vs SCIP alone on SCIP's weak set among all 17, voids, frozen-v2's hash; D-013 terms vs A6F and A0, the
10-13 reading beside (0.827 / +0.063 reproduced from the run table); the LP decode gate beside; failed runs kept, the
letter beside; 'go / no-go is Berk's' — built 2026-10-15 (branch `c1-rerun`, merged d3fe14f); read 2026-10-16 08:2x on
`c1r-ysync-ld0` (68 of 68, VALID; `--json logs/d030-reading.json`, D-030's Result)

### `tools/d031_report.py` — built

D-031: C1 on the frozen-v2 candidate (`-3way-ysync-ld0.yaml`) read on four seeds — D-030's set `c1r-ysync-ld0` (seeds
0–1) and D-031's `c1r-ysync-ld0-s23` (seeds 2–3) as one reading
(`.venv/bin/python tools/d031_report.py [--failed both|exclude|keep] [--json G] [--d030-json logs/d030-reading.json]`;
no `itrea` entry — the script is the entry): validity as one gate over the 136 runs (`c1r_report`'s six legs on each
set; INVALID names the set and the leg, else INCOMPLETE (n of 136)); D-022's hint and guard and D-025 amendment 3's harm
leg (D-030's bands) on four-seed means, with seeds 0–1 alone, seeds 2–3 alone (the selection check, D-024 rule 2) and
whether the halves agree beside; the comparison with the saved D-030 reading; one `PHASE-1 INTERIM RESULT` line (the
four seeds only when the 136 read VALID, else D-030's reading); `--write-jobs jobs/d031.jsonl` (one call of
`make_jobs.c1_rerun("ysync-ld0", seeds=(2, 3), set_="c1r-ysync-ld0-s23")`, pinned configuration and run hashes, refuses
a used run id) — built 2026-10-15 (`tests/test_d031_report.py` + `test_c1r_report.py` 59 passed); its set runs since
2026-10-16 08:57, not read yet

### `tools/r26-gate.sh` — built

D-021 R26's opening gate for a chain (`tools/r26-gate.sh [--deadline HH:MM] [--idle-s 180] [--limit-c 50]`): waits 3 min
after the servers stopped, then until the hottest zone's median over the last 60 s is ≤ 50 °C and the fan governor is
alive (fan-state written within 30 s, `cap=0`); exit 0 open, 1 at the deadline (the chain then restores) — a stopgap
written 2026-10-13 after W-1013b opened warm, until the pool enforces the gate itself (#38); it opened the 10-15 and
10-16 windows at 07:08 (median 46.0 °C)

### `tools/bsync_check.py` — built

R1's mechanism check before its window (GPU: only inside a lease): a launch loop of ~50-ms kernels synced as the EA
syncs, on one core, alone (host self-CPU per wall-second) and beside a busy process pinned to the same core (its CPU
share), per `--mode auto|blocking_sync|yield`, `--renice N` (D-025's fallback (a): the host +N relative to the stand-in
after its spawn), 12 s per phase by default; run by `tests/test_cuda_sched.py` (marker gpu) with D-025's thresholds
(blocking sync <= 0.25 alone; the stand-in >= 0.8 under yield and the renice) — built 2026-10-14 (branch `r1-bsync`);
run 2026-10-15 07:08–07:17 before R1's pool (figures in `traces/bench/bsync/`; host self-CPU per wall-second auto 0.9998
→ blocking_sync 0.0083)

### `tools/d025_report.py` — built

D-025's rules on the run table and the run directories
(`itrea d025-report [--set r1-bsync|r1-bnice|r1-ysync] [--control r13x|boY]`): validity incl. the flag state per arm
(read from `ea.summary.json`), the generation guard on seed means (>= 1 − max(0.05, the control's relative seed spread),
the amendment) and the one-sided P bands (item_placement, comp21 and — the amendment — the three hint instances), the
hint (ratio <= 0.95), the verdict and the fallback triggers; failed runs kept for the verdict, the letter beside — built
2026-10-14 (branch `r1-bsync`); read 2026-10-15 on `r1-bsync` and `r1-ysync` (both NOT ADOPTED, D-025's Results)

### `tools/d026_report.py` — built

D-026's rules (`itrea d026-report --base … [--nice 10|0]`): validity incl. the decoder in every treatment run and in
none of the controls, the three mechanism checks (ready <= 10 s after the gate, decoded incumbents, ca's LP solve wall
<= 1.10 x), the one-sided no-harm bands, the hint (<= 0.90), the gate's lag and A3@ld1 beside — built 2026-10-14 (branch
`r1-bsync`); read 2026-10-15 13:2x on `r2-ld` with `--base r13x` (ADOPT, D-026's Result)

### `tools/d029_report.py` — built

D-029's rules (as amended 2026-10-15 12:2x) over `r1-ysync` (seeds 0–1) + `r1f-ysync` (seeds 2–5)
(`.venv/bin/python tools/d029_report.py [--failed both|exclude|keep] [--json F]`; no `itrea` entry): validity; rule 1
generations on seeds 0–5 (≥ 0.90), rule 2 P on seeds 0–5 (≤ +0.05), rule 3 ca on seeds 2–3 alone (≤ 0.95); a failed
control run never decides a rule; rerun sets `r1f-ysync-<N>`; `--write-jobs` writes `jobs/d029.jsonl` (an unread warm-up
run at `priority` 0, then the 20 read runs; pinned configuration paths and run hashes) — read 2026-10-15 15:0x: ADOPT
(D-029's Result)

## 5. Configurations — the full record (`configs/`)

- `itrea2-default.yaml` (all defaults, generated from `EAConfig()`; hash `9ac4a93bf93b`; `tests/test_configs.py` keeps
  it equal to the code).
- **Current development defaults** (D-010; never a confirmatory configuration): **A3** — `itrea2-dev-rc-ld.yaml`
  (`itrea2-dev-rc-3way` + `repair.continuous: lp_decode`, hash `f030e23909ad`, D-017); **A9 / A3x** —
  `itrea2-dev-rc-3way-ysync-ld0.yaml` since D-029's ADOPT (2026-10-15 15:0x; EA hash `835c7c5a249c`, A9 run hash under
  ext `f345de0ab936`) = `itrea2-dev-rc-3way.yaml` + lp_decode gated after the LP worker on the EA's core, nice 0
  (D-026's ADOPT, 13:2x: `-3way-ld0.yaml`, `a05af2c732dc` / `341267810eed`) + `host.cuda_sched: yield` (D-029); the
  configuration that freezes for phase 1 by D-032's rule (DRIFT, 2026-10-16), copied to `configs/frozen-v2.yaml` on
  Saturday (not yet written); `make_jobs.ITX_A9_CONFIG` still names `-3way.yaml` until a `src/` change after the running
  pool; `itrea2-dev-rc-3way.yaml` (repair clock active, adaptive repair strategy, the EA's own HiGHS LP, LP stream, deep
  repair; hash `1d12ad490dd0`, A9 run hash `28e5a5e9c725` = A9@r13x's; the runner adds `exchange.from_cuopt` for A9 and
  refuses an ungated lp_decode there) stays as the record of the old default;
- **D-018 stage 1** — `itrea2-dev-rc-3way-bo1.yaml` (back-off active with the stage-0 values: D_Y ½, floor ⅛,
  yield_until 30 s, patiences 40 / 40; run hash `f67354be775d`) against `-bo1s.yaml` (the same values in shadow mode =
  the control; run hash `62bf82b236f6`);
- **D-023** — `itrea2-dev-rc-3way-boY.yaml` (phase Y only: bo1's values, active, patiences 100000 so the ladder never
  steps down; hash `5c8b535fb245`) against `-bo1s.yaml`, NOT ADOPTED (D-023's Result, 2026-10-14).
- **R1 / R2 of the 2026-10-14 C1 diagnosis** (each = its base + one change; EA hash / A9 run hash under ext; run
  2026-10-15: R1 bsync NOT ADOPTED, its fallback ysync NOT ADOPTED at seeds 0–1 (D-025's Results), then ysync ADOPTED on
  seeds 0–5 (D-029); R2 ld0 ADOPTED, D-026):
- R1 on the A9 default — `-3way-bsync.yaml` (`host.cuda_sched: blocking_sync`; `305c8902d763` / `330a10a77556`), D-025's
  fallbacks `-bnice.yaml` ((a): `host.nice_after_lp_spawn: 10`, +10 relative to the LP worker, no context flag;
  `96a299330981` / `d266e4aaecbe`) and `-ysync.yaml` ((b): `yield`; `f81d6b6209f0` / `a08221cca653`); the same on boY —
  `-boY-bsync` (`29e61aa41619` / `b760e770cb34`), `-boY-bnice` (`5b1c4779c74b` / `b4f4af8376dd`), `-boY-ysync`
  (`666865eac174` / `22137f009005`);
- R2 — `-bsync-ld.yaml` (bsync + lp_decode gated after the LP worker, nice 10; `119c6e17f830` / `dc45bdbb771f`),
  `-bsync-ld0` (nice 0; `b7af15f4b62a` / `93a8ed06b59b`), the fallback without R1 `-ld0` (`a05af2c732dc` /
  `341267810eed`; adopted, D-026: part of A9's and A3x's default above), and on boY `-boY-bsync-ld` (`c296898cddc4` /
  `3ef4bf1254fe`), `-boY-bsync-ld0` (`2b6baa898edf` / `37a47a123c67`), `-boY-ld0` (`aabac30b5056` / `58f72d467a23`).
- **D-030 / D-031 / D-032**: `-3way-ysync-ld0.yaml` = `-ld0` + `host.cuda_sched: yield` (`835c7c5a249c` /
  `f345de0ab936`; built for D-030, its A9-layout cold-cache test passed 2026-10-15 15:07; run as A9@c1r in D-030 (seeds
  0–1) and D-031 (seeds 2–3, running) and as A9@g4yl in D-032, beside `-3way.yaml` (A9@g4old) and `-3way-ld0.yaml`
  (A9@g4ld)).
- Every file's hash is pinned in `tests/test_cuda_sched.py` (`ALL_CONFIG_HASHES`).
- History: `-3way-bo0/-bof50/-bof25/-bof12.yaml` (D-018 stage-0 arms: shadow, forced duty ½ / ¼ / ⅛),
  `itrea2-adaptive.yaml`, `-adaptive-lps.yaml`, `-adaptive-lps-rc.yaml` (earlier development steps, D-010–D-012),
  `itrea1-faithful.yaml` (a record of part I's choices, not a planned comparison), `ablations/*.yaml` (D-002's E3)

## 6. Job writers — the full record (`src/itrea/experiments/make_jobs.py`)

- job files: `screen1`, `screen2`, `e1`, `e4`, `confirm`, `smoke`, `backoff0` / `backoff1` (D-018 stages 0 / 1),
  `mipfeas-cal` (D-020: A6m on the 30-instance sample; its file left `jobs/` when D-021's M4 replaced it), `itx-m1` …
  `itx-m5`, `itx-r13` (D-021 phase 1 → `jobs/itx-*.jsonl`), `itx-r13f` (D-022's follow-up at seeds 2–3),
  `backoff2 --harness ext|r13-off [--instances amended|stage1|a,b]` (D-023; default the 11 of its 2026-10-13 amendment =
  66 runs, `jobs/backoff2.jsonl`; `stage1` = D-018's 14),
- phase 2: `itx-v1 --tier T1..T5 [--cuopt 26.10] --out F` (the v1 tiers from `datasets/mipfeas_strata.csv`; A9 on
  `configs/frozen-v2.yaml`, refused while it does not exist; not written before 26.10),
  `itx-a0 [--select all|d024|stage1]`, `itx-m6`, `itx-m6a`,
- R1 / R2 of the 2026-10-14 C1 diagnosis (D-025 / D-026 targeted tests; the control = the development default's
  configuration re-run under the set's own label `A9@r1ctl` / `A9@r2ctl` — the same run hash, a new run id —; the two
  arms' order alternated over each instance's seeds, `position`):
  `r1-bsync --control r13x|boY [--variant bsync|bnice|ysync] [--set S]` (20 runs, set `r1-<variant>`,
  `jobs/r1-<variant>.jsonl`), `r2-ld --base bsync|boY-bsync --nice 10|0 | --ea-cpu X` or `--base r13x|boY` (28 runs, set
  `r2-ld`, `jobs/r2-ld.jsonl`, labels `A9@r2ld` / `A9@r2ctl`); each job file was committed only after the verdict it
  depended on (D-023 for R1, D-025 for R2): `jobs/r1-bsync.jsonl`, `r1-ysync.jsonl` (`--variant ysync`), `r2-ld.jsonl`
  (`--base r13x`) (`jobs/itx-a0.jsonl` = all 230, `itx-a0-81.jsonl` = D-024 (f)'s 81 with the 25 v1 stage-1 instances at
  priority 0 and the other 56 at 1, `stage1` = the 25; `itx-m6.jsonl`, `itx-m6a.jsonl`),
- `r3-diag` (the C1 diagnosis note's R3, D-027 as amended 2026-10-14 — only what a rule reads: comp21-2idx, toguru, doon
  × {A6@r3 = cuOpt alone 3 threads on A725 1–3, A2@r3} + toguru, doon × A6F@r3 = A6F@r13x's job under its own variant +
  chromaticindex1024-7 × {A6@r3, A6F@r3, A2@r3} (rule 4, the 18:3x amendment), ext, 120 s, seeds 0–1 = 22 runs in 8
  blocks, `jobs/r3-diag.jsonl`),
- `c1r-rerun --a9 ld0|ysync-ld0` (D-030: the C1 re-read on the frozen-v2 candidate — D-022's C1 runs again, the 17 ×
  {A9@c1r, A6F@c1r} × seeds 0–1, 120 s, ext = 68 runs in 34 blocks, set `c1r-<a9>`, `jobs/c1r-ld0.jsonl` /
  `jobs/c1r-ysync-ld0.jsonl`; A6F@c1r = A6F@r13x's job under a new label (run hash 82a371305f88, the same in both files:
  only one runs); the arms alternate by seed, A6F first at even seeds; run hashes pinned, the writer refuses a moved
  one; `c1r-ysync-ld0` ran 2026-10-15/16, D-030's Result), (`itrea make-jobs …`).
- Job files of this week written outside `make-jobs` (no `src/` edit while pools ran): `jobs/d028.jsonl` (D-028:
  `make-jobs backoff2 --harness r13-off --instances chromaticindex1024-7`, the bo1s rows dropped, set renamed `d028`),
  `jobs/d029.jsonl` (`tools/d029_report.py --write-jobs`: an unread warm-up run at `priority` 0, then 20 runs, set
  `r1f-ysync`), `jobs/d031.jsonl` (`tools/d031_report.py --write-jobs`: one call of
  `make_jobs.c1_rerun("ysync-ld0", seeds=(2, 3), set_="c1r-ysync-ld0-s23")`, 68 runs), `jobs/d032.jsonl` (D-032: from
  D-030's glass4 A9 job, config / variant / set / block / position / seed changed, 12 runs).
- **`ITX_A9_CONFIG` still names `configs/itrea2-dev-rc-3way.yaml`** (the old A9 default) until a `src/` change after the
  running pool; the new default is `-3way-ysync-ld0.yaml` (D-026 + D-029)

## 7. Current state (2026-10-16, morning) — moved from `AGENTS.md`

| item | state | source |
|---|---|---|
| Phase | development (phase 1) until C1, **Sat 2026-10-17** (moved from Fri 10-23; Berk 2026-10-15: "Assume we are wrapping up phase 1 on Saturday") | D-019 addendum |
| Confirmatory runs | none. Pre-registration D-002 v3.1 binding since the tag `prereg-v1` (2026-10-06); prereg-v2 (#32) pending | D-002 |
| Development instances | 17 at 120 s: the development set (9, D-010, since 2026-10-06, `datasets/dev_set.csv`) and the class probe (8, D-014, since 2026-10-07, `datasets/classprobe_set.csv`); both permanently excluded from the test set | D-010, D-014 |
| Hard pool | SCIP screening stage 1 complete (324 candidates, seed 900); stage 2 deferred (D-010 §5), so the pool is not frozen (#1) | `vault/10-design/Hard instance pool.md` |
| Development default, A3 | `configs/itrea2-dev-rc-ld.yaml`, hash `f030e23909ad` (lp_decode adopted) | D-017 |
| Development default, A9 and A3x | `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` since 2026-10-15: EA hash `835c7c5a249c`, A9 run hash under ext `f345de0ab936`. It is the old default `configs/itrea2-dev-rc-3way.yaml` + lp_decode gated after the EA's LP worker (D-026) + the EA's CUDA context at `yield` (D-029) | D-026, D-029 |
| Freeze candidate | that same file. It freezes for phase 1 by D-032's rule (DRIFT, fixed before its runs). `configs/frozen-v2.yaml` is written on Saturday (not yet). D-031's four-seed harm reading may still put which configuration freezes to Berk | D-032, D-031 |
| **Trap: stale job-writer default** | `make_jobs.ITX_A9_CONFIG` still names `configs/itrea2-dev-rc-3way.yaml` (the old default) until a `src/` change after the running pool. A writer that uses it writes the old default: A9 in `itx-m1` … `itx-m5`, `itx-r13`, `itx-r13f`; the `r13x` control of `r1-bsync` and `r2-ld`. `itx-v1` takes `configs/frozen-v2.yaml`; `c1r-rerun` takes its pinned files | `src/itrea/experiments/make_jobs.py` |
| Running now | D-031: set `c1r-ysync-ld0-s23` (`jobs/d031.jsonl`, 68 runs at seeds 2–3 on the candidate), started 2026-10-16 08:57; log `logs/d031-1016.log` | D-031 |
| External benchmark | ITX-1 built (phases 1 and 2 merged); F accepted on M2's soak; M3's "does not fit F" list fixed; M4 calibration complete: **OUTSIDE** (the footprint and the machine) | D-021; `vault/30-results/2026-10-11 D-021 measurements M1–M3 and the interim calibration.md` |
| A0 on the 81 | SCIP alone, `itx1`, 600 s, on the 81 instances v1's decisions read (D-024 (f)): 80 completed + 1 failed (s100: SCIP did not stop at 600 s, killed at 701 s; kept as an A0 failure) | journal 2026-10-14 21:48 |
| cuOpt | 26.08 installed. NVIDIA/cuopt#2073 is fixed in 26.10 (PR #2098); A9 stays one-way until #35 (see "Setting") | `vault/70-upstream/` |
| Due Saturday | `configs/frozen-v2.yaml`, the phase-1 results note, checkpoint, docs, issues, the go / no-go (Berk's) | D-019 addendum |
| Carried to phase 2 | #26 (not built), #35 two-way (v1.1, Berk 10-14), CUDA MPS (v1.1, Berk 10-15), chromaticindex as a known loss | D-019 addendum |
| v1 | end of November 2026 (aim Fri 11-27, hard end Mon 11-30) on cuOpt 26.10; daytime windows only | D-019 |
| Phase-1 results note | `vault/30-results/2026-10-17 Phase 1 — development until C1.md` (DRAFT until D-031 is read) | — |
| Newest checkpoint | `vault/90-checkpoints/2026-10-14-2350.md` | — |

### 7.1 Open issues (GitHub, `Tideseed/i-trea-ii`)

| issue | what | due |
|---|---|---|
| #1 | freeze the hard pool | v1 |
| #11 | decisions for Berk (`needs-berk`), among them the test-set size (D-010 §4 proposed 16) | prereg-v2 |
| #12 | the paper | v1 |
| #26 | recombine each new partner incumbent with the elite (not built) | phase 2 |
| #31 | cuOpt 26.10: separate venv, API check, A6 re-baselined on the 17 | v1 |
| #32 | prereg-v2: confirmatory and MIPfeas runs; its test-set size is due before prereg-v2, not Saturday | v1 |
| #33 | run-id reuse: the same id twice inside one set is still refused | before any pool reruns a job inside one set |
| #35 | A9 two-way through cuOpt 26.10's SET callback (not started) | v1.1 |
| #36 | checker tolerance: our exact checker accepted incumbents below MIPLIB's proven optimum on glass4 (cuOpt rejects that point) and mas74, and rejected cuOpt incumbents on snowy (1) and neos-933966 (3) | before prereg-v2 |
| #37 | code owed by D-024 (`make-jobs itx-v1`'s stage-1 selector, `itx-report`'s stage-1 reading, the pool's cuOpt-version check) | before v1 stage 1 |
| #38 | the pool enforces D-021 R26's opening gate (stopgap: `tools/r26-gate.sh`) | before v1's windows |
| #9, #16 | `needs-berk`: E5 long runs, islands | — |

Rows #33, #37 and #9 / #16 come from `vault/00-index/Roadmap.md` and `vault/00-index/Home.md`.

## 8. Interim readings — development, descriptive (moved from `AGENTS.md`)

All of these are development data on instances that were chosen and tuned on (D-010). None is a result of the
pre-registered test.

| reading | what ran | result | record |
|---|---|---|---|
| A9 pilot | seed 0, 7 instances | vs SCIP alone 7 better / 0 worse, mean d +0.68 [+0.42, +0.90]; vs cuOpt alone 3 / 4, −0.07 [−0.42, +0.25] | `vault/30-results/Development log.md` § "Three-way arm A9 — pilot" |
| C1, old default | D-022's Result, 10-13, seeds 0–1 | hint 0.827, guard +0.063 | D-022 |
| C1, frozen-v2 candidate | D-030, 68 runs, 10-15/16, seeds 0–1 | **HINT 0.809, GUARD +0.050** | D-030 § Result and § Correction 2026-10-16 10:0x |
| glass4 in one window | D-032, 12 runs, glass4 × seeds 0–3 | old default 0.2176, candidate 0.2233, lp_decode alone 0.2177; d −0.025 ≥ −0.057 → **DRIFT** | D-032 § Result |
| M4 calibration (the harness, not the method) | A6F on D-020's 30 instances, 600 s | **OUTSIDE** the band [0.870, 1.150]: R = 1.313 [1.091, 1.618] on the call clock | result note, addendum 2026-10-15 |

How to read them:
- **C1 readings** compare A9 with A6F in the same window, seeds 0–1 (hint and guard: see "Terms").
- **D-030's reading keeps two crashes.** cuOpt alone's SIGSEGV on comp21-2idx at both seeds is kept as A6F outcomes;
  the reading gains from them (the saved report scores the two crashed runs at A6F P 0.776 and 0.782, truncated at
  the crash, against A6F@r13x's 0.511 / 0.496 on 10-13; D-030 § Correction 2026-10-16 10:0x). Left out, hint and
  guard are INCOMPLETE.
- **D-030's harm leg** (cross-day, against 10-13's A9@r13x) fired on glass4 (d_i −0.088 vs −0.057; krka at floor).
  D-032 read glass4 in one window and found DRIFT: the candidate freezes for phase 1 by D-032's rule (fixed before its
  runs). D-031's four-seed harm reading may still put which configuration freezes to Berk.
- **M4** also: launch clock 1.963 [1.606, 2.408], feasible 27 vs GAMS 27, late-clock check passed (median −0.119 s over
  strata 0–1). With the clock checked, the difference is the footprint F (5 threads + the GPU against GAMS's 24-thread
  whole box) and the machine, so v1's targets stay anchored on our own A6F. The interim 19-of-30 reading, R 1.390, is
  superseded.
- **The go / no-go on C1 is Berk's.**

## 9. Development decisions — what is in the default and what is not (moved from `AGENTS.md`)

Each verdict follows its record's rules, fixed before any of its runs was read (D-022's second amendment came with 7 of
the 58 runs of its first pool done, none read). Development data, n = 2–6 seeds, descriptive. Full numbers:
`vault/30-results/Development log.md` and each record's § Result.

| record | change | verdict | key numbers |
|---|---|---|---|
| D-011 | SCIP streams good infeasible node-LP points to the EA (the LP stream) | in the default | — |
| D-012 | repair clock (`ea/repair_clock.py`) | active in the default since D-012 addendum 2 | — |
| D-016 | Rank 1 by cuOpt's clearing rule | NOT ADOPTED | seeds 0–3: mean d −0.033 [−0.122, +0.050]; size-only variant −0.000 [−0.044, +0.049] |
| D-017 | lp_decode for A3 (24 runs, 2026-10-09) | **ADOPTED** (A3 default) | mean d +0.219 [+0.041, +0.429], 6 instances × 2 seeds (Roadmap, #27) |
| D-018 | A9's GPU back-off, stage 1 (bo1 vs bo1s, 84 runs, 2026-10-11) | NOT ADOPTED | every outcome rule held; the mechanism failed on M1 / M3 / M5; an analysis bug (`_band` read the seed spread of the old control `A9@bo0`) was fixed before the verdict |
| D-022 | A9 under ext (R13 + R14), 74 runs | **ADOPTED** | valid; no harm on the 3 instances where cuOpt's own 1e-4 stop occurs; mechanism met only vacuously (SCIP proves first), so the adoption rests on D-021's fairness and GAMS fidelity |
| D-023 | phase Y of the back-off alone, 66 runs (2026-10-14) | NOT ADOPTED — the back-off line is closed for v1 | failed M2, M5, T, G: chromaticindex d +0.030, needed +0.30; cuOpt's first incumbent inside A9 63.47 / 63.41 s vs A6F 10.87 / 10.52 s |
| D-025 | R1: bsync (20 runs) | NOT ADOPTED | hint 0.765; generation guard failed on item_placement and comp21, comp21 P +0.060 > band 0.050 |
| D-025 | R1 fallback: ysync at seeds 0–1 (20 runs) | NOT ADOPTED, reopened by D-029 | hint 0.834; item_placement's generation guard failed on one seed, 678 vs 1,480 |
| D-026 | lp_decode inside A9 after the LP worker, 28 runs | **ADOPTED** | hint 0.758 (≤ 0.90; item_placement P 0.330 → 0.198, sing44 0.473 → 0.411, lower on all four seeds of both); mechanisms and no-harm legs held |
| D-027 | diagnostic, 22 runs | rule 4 EA PRESENCE; rule 1 closed the layout line; rule 3 ruled SCIP out | cuOpt's first incumbent on chromaticindex1024-7 at 9.80 / 7.33 s alone on F, 9.24 / 8.21 s on 3 threads, 6.53 / 9.65 s beside SCIP, vs 62–66 s inside A9 |
| D-028 | diagnostic, 4 runs | not R13 + R14 | A9@boY under r13-off 63.25 / 63.24 s vs L = 19.82 s |
| D-029 | `yield`, 21 runs, guards over seeds 0–5 with D-025's failing seed kept | **ADOPTED** | generations item_placement 1.020, comp21 1.152 (≥ 0.90); P +0.016 / +0.015 (≤ +0.05); ca at seeds 2–3 0.675 (≤ 0.95) |
| D-030 | C1 re-read on the candidate, 68 runs | see "Interim readings" (§ 8 here) | HINT 0.809, GUARD +0.050 |
| D-032 | glass4 in one window, 12 runs | DRIFT → the candidate freezes | see "Interim readings" (§ 8 here) |
| D-031 | C1 at seeds 2–3, 68 runs | running | placeholder |

Notes:
- **D-029's thresholds** were set with seeds 0–1 (8 of its 28 read runs) already known, and item_placement's generation
  bar was loosened from D-025's 0.921 to 0.90 (D-029 amendment B1–B2). At D-025's bars the verdict is the same (item
  1.020 > 0.921; comp21 1.152 > 0.898 / 0.950), so this qualifies the claim, not the result.
- **chromaticindex stays a known v1 loss.** CUDA MPS is not for v1 (Berk 2026-10-15).

## 10. Changelog (moved from `AGENTS.md`)

Dated milestones of the project as this file has tracked them. Day-by-day detail: `vault/30-results/Development log.md`
(§ Changes) and the journal. The status narrative this table condenses is in the detail note § 1.

| date | what changed | record |
|---|---|---|
| 2026-10-05 | Berk's brief; the design critique | `vault/80-reviews/2026-10-05 design critique.md` |
| 2026-10-06 | first build (00:0x–01:2x); at 01:4x the status read "built and tested on toy models; no commit, no tag, no experiment, no results" — 413 passed, 2 skipped. Main claim (D-007); pre-registration D-002 v3.1 tagged `prereg-v1`; SCIP screening stage 1 (324 candidates, seed 900); development set (D-010); LP stream (D-011) | D-002, D-007, D-010, D-011 |
| 2026-10-07 | aim refined (D-013); class probe (D-014); arm A9 and its pilot (D-015); NVIDIA/cuopt#2073 filed; cuOpt's MIP start verified on glass4 | D-013, D-014, D-015 |
| 2026-10-08 | Rank 1 by cuOpt's clearing rule built and run (D-016) | D-016 |
| 2026-10-09 | D-016 NOT ADOPTED (seeds 0–3); lp_decode ADOPTED as the A3 default (D-017); A9's GPU back-off built, stage 0 complete: PROCEED (D-018); CPU tests 816 passed and the GPU tests of back-off, lp_decode, elite and EA run 40 passed | D-016, D-017, D-018 |
| 2026-10-10 | v1 decided: C1 Fri 10-23, v1 end of November, "No nights" | D-019 |
| 2026-10-11 | D-018 stage 1 NOT ADOPTED; the MIPfeas harness merged (D-020); ITX-1 accepted by Berk, its phase 1 built (D-021); CPU-only thermal start line 88 → 86 °C | D-018, D-020, D-021 |
| 2026-10-12 | D-021 phase 2 merged (6f0339d; 1,009 CPU tests passed on main); cuOpt PR #2098 merged upstream | D-021 |
| 2026-10-13 | staged testing (D-024, Berk); D-022 ADOPTED and C1's first reading (hint 0.827, guard +0.063); the runner's GPU tests passed (4 passed, 1 skipped) and a cold-cache EA test; #2073's fix confirmed for 26.10 and verified on the nightly; `tools/r26-gate.sh` written | D-022, D-024 |
| 2026-10-14 | D-023 NOT ADOPTED — the back-off line closed for v1; the C1 diagnosis; R1–R3 built and merged (non-GPU 1,256 passed, 1 xfailed on main); A0 on the 81 (80 + 1 failed); a lease over 12 h refused, 7 h 36 min lost; #35 → v1.1 (Berk) | D-023, D-025–D-027 |
| 2026-10-15 | D-025 NOT ADOPTED (bsync; ysync at seeds 0–1); D-027 and D-028 diagnostics; D-026 ADOPT; D-029 ADOPT (yield) → A9's default becomes `-3way-ysync-ld0.yaml`; M4 complete, OUTSIDE; C1 moved to Sat 10-17 (D-019 addendum); CUDA MPS → v1.1 (Berk); 1,339 passed on `c1-rerun`, GPU suites 45 passed, 1 skipped; D-030's runs start; R26's breaker fired at 17:30 | D-019, D-025–D-030 |
| 2026-10-16 | D-030 read (HINT 0.809, GUARD +0.050, harm flag on glass4); D-032: DRIFT → the candidate freezes for phase 1; D-031 started 08:57; phase-1 docs drafted; this file restructured (detail moved to the detail note); later the same day the current state, readings, decisions and this changelog moved to the detail note §§ 7–10 | D-030, D-031, D-032 |

## Sources

- `AGENTS.md` at commit `3cbc7a1` (`git show 3cbc7a1:AGENTS.md`): the verbatim original of §§ 1–6.
- §§ 7–10: `AGENTS.md`'s restructured working tree of 2026-10-16 (not committed before the move); their content traces
  to the same commit and to the records each row cites.
- [[Home]], [[Roadmap]], [[Development log]], [[Experiment index]], the decision records D-001 … D-032.
