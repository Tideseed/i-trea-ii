# D-027 Same-day diagnostic of cuOpt's thread count and SCIP's presence (R3) — reading rules fixed before the runs (2026-10-14, AgentRA)

**Status:** amended / confirmed 2026-10-14 18:3x before any run (see the end). decided for development by AgentRA (Claude Code, Anthropic Claude Opus 5.5) before any run; open to a
superseding decision on evidence. **Not run; job writer not merged** (branch `r3-diag`, worktree `../i-trea-ii-r3`).
**Amended 2026-10-14 17:5x, before any run: 16 runs, only what a rule reads (D-024 rule 4); rule 2's sources and its
"≤ 133" wording clarified** (the section at the end). A
diagnostic, not a lever: no hint and no adoption — **reading rules fixed before the runs**. Configuration only, no EA
code. Runs in any window after `backoff2`, before or alongside [[D-025 A9 EA blocking sync (R1) — the EA stops spinning on the GPU so its LP worker gets core 6, development test]]
(it may share that window). Development only (D-010). Candidate R3 of [[2026-10-14 C1 diagnosis and improvement candidates]]
§5 (the note's candidate, not D-021's rule R3); it replaces the dropped candidate 3 (§6). Issues: #30 (the layout
question); C1's diagnosis.

## Why
Both reviewers of the diagnosis asked for it — reviewer 1 as the replacement for candidate 3 and as its first missing
item, reviewer 2 as its seventh missing item — and it is the only work before cuOpt 26.10 that addresses comp21-2idx, a
C1 loss.
- **Thread / core effect.** The note's §2 footprint column compares old-harness 3-thread A6 with ext 5-thread A6F, so it
  is indicative only: that comparison also changes the thread variables (OMP / OPENBLAS / MKL = 1 vs unset), the gaps
  (default vs 0), RMM (12 vs 24 GB), the CUDA cache and the day. D-021 R6 records counter-evidence (A6 vs GAMS's 24-thread
  run, R 1.00 [0.89, 1.12]). No 3-thread cuOpt-alone run exists under harness ext.
- **comp21's part 2.** After about 50 s, cuOpt inside A9 stops improving at 146–172 at every EA duty (⅛ duty: 151,
  n = 1); alone it reaches 114–133. Diffing comp21's cuOpt parameters in A9@r13x and A6F@r13x leaves only cores (3 vs 5)
  and `send=True` (0.11 s of exchange time over 166 calls); seed, gaps, RMM and environment are identical. The note's §2
  rules out the thread count, memory (RMM 5.26 GB everywhere) and the send callback; what remains is SCIP beside cuOpt,
  or the EA's second CUDA context. No 3-thread cuOpt-alone run under ext exists, so rule 2's precondition re-checks the
  thread count under ext.
- **SCIP's presence.** Where it was measured, SCIP beside cuOpt did not delay cuOpt (A2 on chromaticindex 6.50 s, fhnw
  0.62, kumeu 3.46, radiation 0.86). It is untested on comp21, toguru and doon.

## Runs — fixed before any run
**Set `r3-diag`**, job file `jobs/r3-diag.jsonl`, seeds 0–1, 120 s, harness **ext** on every run, one window, one block
per instance × seed:

| arm (job label) | what it is | instances | runs |
|---|---|---|---|
| A6@ext (`A6@r3`) | cuOpt alone, 3 threads on A725 1–3 | comp21-2idx, neos-4763324-toguru, neos-2746589-doon, sing44, item_placement_9903, ca__Scenario400_reserves_1 | 12 |
| A6F@ext (`A6F@r3`) | cuOpt alone on F, re-run in the same window | the same 6 | 12 |
| A2@ext (`A2@r3`) | SCIP on X925 5 + 3-thread cuOpt on A725 1–3, no EA, core 6 idle | comp21-2idx, neos-4763324-toguru, neos-2746589-doon | 6 |

**30 runs ≈ 1.7 h** (3.44 min per run, excluding the window's open and close and D-021's 0.55 efficiency).

**Validity (must hold before any rule is read):** 30 of 30 completed or a recorded void rerun; harness ext on every run;
cuOpt's thread count and cores as in the table, read from each run's `run.json` (3 on A725 1–3 in A6@ext and A2@ext, 5 on
F in A6F@ext); SCIP on X925 5 and no process on core 6 in A2@ext; no EA process in any run; 0 compile on the clock.
Otherwise INVALID, and the report names the leg.

**Quantities.** P = P(120) on the call clock. In A6@ext and A6F@ext the arm's P is cuOpt's; in A2@ext, "P of cuOpt" is
cuOpt's own P (`cuopt_pt`), not the portfolio's. "cuOpt's final objective" is the objective of cuOpt's own last
incumbent at 120 s. A seed spread is |P_s0 − P_s1| in raw P.

## Reading rules (fixed before the runs)
1. **Thread effect on instance i** is present iff P̄(A6@ext) − P̄(A6F@ext) > max(0.05, the larger seed-0 vs seed-1
   spread of either arm). These same-day numbers replace the note's §2 indicative footprint / slowdown split on all six.
   - Present on toguru or doon (where cuOpt holds or would hold A9's portfolio) → **the layout question** (a cuOpt thread
     on X925 6) **goes to Berk with these numbers.** The note's §6 lists what that design would need: between sibling
     scopes the CPU is shared by `cpu.weight`, not nice (e.g. `-p CPUWeight=` 10–20 against 100); layout code for cores
     {1, 2, 3, 6}; an exception to AGENTS.md's "Timed solvers pinned one per X925 core" and D-015 / D-021's layout; and
     an answer on D-021 R5 (does M2's soak repeat?).
   - Otherwise **the layout line is closed for v1.**
   - On comp21, sing44, item_placement and ca the reading is reported (it replaces §2's split there) and sends nothing to
     Berk by itself.
2. **comp21's part 2** is read from cuOpt's final objective in A2@ext:
   - **≥ 146 on both seeds** → SCIP beside cuOpt is the cause; no question for Berk, and the MPS question is not raised;
   - **≤ 133 on both seeds** → the EA's CUDA context remains, and **the MPS question** (a host change, Berk's) becomes
     concrete and **goes to Berk** with these numbers;
   - anything else (mixed, or between 133 and 146) → inconclusive at n = 2, stated as such.
   - **Precondition:** A6@ext's own cuOpt reaches **≤ 133 on both seeds** on comp21 (made exact here; the note says
     "must reach ≤ 133"). If it ends **≥ 146 on both seeds**, the stall is the 3-thread footprint under ext, and rule 2
     does not apply — reported with rule 1's comp21 row (it does not send the layout question by itself; rule 1 limits
     that to toguru and doon, but it is quoted beside it if rule 1 sends it). Anything else → the precondition is not
     met, rule 2 is not read, and comp21's part 2 stays inconclusive at n = 2, stated.
3. **SCIP's presence on toguru and doon:** P of cuOpt in A2@ext against P(A6@ext), with rule 1's band: present iff
   P̄_cuOpt(A2@ext) − P̄(A6@ext) > max(0.05, the larger seed spread of either arm). Present → SCIP's presence is recorded
   as a cause of cuOpt's slowdown on that instance; absent → ruled out there at n = 2. **Either way it is reported: it
   names no lever and raises no question for Berk by itself** (the note names no consequence; any SCIP-side change would
   need its own record).

**Reported beside, read by no rule:** A6F@ext against A6F@r13x of 10-13 on the six (a day-to-day replicate of cuOpt
alone on F).

**What goes to Berk, and when:** only rule 1 present on toguru or doon (the layout exception) and rule 2 "≤ 133 on both
seeds" with its precondition met (CUDA MPS) — the two items the note holds back ("Not asked now"). Every other outcome is
recorded in the Development log and the diagnosis note's §7, and asks him nothing.

## Before the window
- **Dependency:** `make_jobs` and the runner must accept A6 and A2 under harness ext. A2F ran under ext in D-021's M1, and
  the harness is allowed on partner-lane arms; if A6 or A2 is refused, the fix is small jobs / runner code on `r3-diag`,
  merged after the running pool ends.
- **CPU tests before the merge:** `make-jobs r3-diag` writes the 30 runs with run ids distinct from every other set's;
  the runner resolves each arm's cores, threads and harness as in the table; every existing job file regenerates byte
  for byte; the report script (`tools/d027_report.py`, reading the run table and cuOpt's own traces) and each rule's
  outcome, with tests.
- **No new GPU check:** no EA code and no new kernels; the window's usual gates (skill `itrea-window`).

## Risks
n = 2 per cell. doon is a ceiling case: A6F found it at 94.1 and 119.4 s, P 0.792 / 0.996. Rule 2's thresholds (146,
133) come from 10-13's runs on two seeds (A9 inside 146–172, alone 114–133). Needs Berk: no, until a rule sends him a
question.

## Addendum 2026-10-14 16:4x — implementation, before any run (AgentRA, Claude Code (Anthropic Claude Opus 5.5))
Built on branch `r3-diag` (worktree `../i-trea-ii-r3`; commits 29d54d1, 32472b5; not merged, no run).
- **No runner change.** `pair.plan` already lays A6 and A2 out under ext as the table says (A6: cuOpt on A725 1–3,
  `--cpu-threads 3`; A2: SCIP on X925 5 + that cuOpt, nothing on X925 6; A6F: cuOpt on 1, 2, 3, 5, 6, `--cpu-threads 5`;
  every cuOpt with gaps 0, thread variables unset, RMM 24 / host 24 GB); the pool refuses a harness only on A6m and on
  test / screening jobs. New tests pin it (`tests/test_make_jobs.py`).
- **Job file** `jobs/r3-diag.jsonl` (`itrea make-jobs r3-diag`), sha256
  `2c5d50a913548ea6cf4149bf0598a0c9167a4d9fc289c4f8e3a5e27074aa151e`: 30 runs in 12 blocks, set `r3-diag`, variant `r3`
  on every arm. Run hashes A6@r3 `e54170304dad`, A6F@r3 `82a371305f88` (= A6F@r13x's: the same job — config, overrides,
  harness, budget — under another label, checked field by field against the committed `jobs/itx-r13.jsonl`), A2@r3
  `9c3031af075f`. Run ids carry the label, so none equals an A6F@r13x id or any id of another committed job file; the
  directories are `traces/raw/r3-diag/`, so the pool's resume skips nothing of R3's for an earlier run.
- **Pool `--dry-run`** (worktree, `taskset -c 10-14`, nice 19, `nvidia-smi` shadowed by a failing stub so no GPU query
  ran): 12 blocks, 30 runs, lane g0; memory per run A6@r3 69, A6F@r3 69, A2@r3 78 GB (worst case 78 against
  MemAvailable 108.5 GB); preflight passed the lease, serving, freeze and harness checks and stopped only at the
  stubbed GPU-health sample.
- **CPU tests:** the writer's file regenerates byte for byte and so do every other writer-backed job file except
  `jobs/itx-r13.jsonl` (narrowed by hand, #37; already a strict xfail); the whole `-m "not gpu"` suite passed, pinned
  (1,076 passed, 1 skipped, 1 xfailed at 29d54d1; 1,078 passed, 1 skipped, 1 xfailed at 32472b5).
- **Report** `tools/d027_report.py` (`itrea d027-report [--failed both|exclude|keep] [--json F]`) applies this record:
  validity as one gate (leg names complete / harness / layout / compile), the three rules, "what goes to Berk", and
  A6F@r3 against A6F@r13x beside. It reads the run table (`pt`, `cuopt_pt`, `cuopt_best_obj` = cuOpt's own last checked
  incumbent at 120 s) and each run's `run.json` (process roles, cores, `--cpu-threads`, harness, compile flag).
- **Readings where this record is silent (fixed now, before any run; not rule changes):** rules 1 and 3 are one-sided
  as written — "present" only when the 3-thread / SCIP-beside arm has the higher P by more than the band; a run whose
  cuOpt holds no checked incumbent at 120 s has final objective +∞; comp21's sense is checked to be minimisation
  (another sense leaves rule 2 not read); the 146 / 133 thresholds are compared with a 1e-6 tolerance; failed runs:
  `exclude` is the letter (a failed run is not completed → INVALID), `keep` reads it as D-002 §6 keeps a crash and the
  report names that as a stated deviation; `both` (default) prints both readings when one exists.

## Amendment and clarification 2026-10-14 17:5x — before any run (AgentRA, Claude Code (Anthropic Claude Opus 5.5))
From an adversarial review of the R1–R3 builds (Claude Code subagents). No `r3-diag` run exists; the 16:4x addendum
above stands except where this section changes it.

**Amendment — only the runs a rule reads (D-024 rule 4: "Breadth no decision reads is not run").** Of the 30 runs, 14
fed no reading rule: A6@r3 and A6F@r3 on sing44, item_placement_9903 and ca (12; rule 1 there was "reported, sends
nothing"), and A6F@r3 on comp21 (2; rule 1 there was report-only and rule 2 reads A6@r3 and A2@r3). They are dropped
(≈ 0.8 h of run time, ≈ 1.5 window hours at D-021's 0.55, before the 10-23 freeze). **Set `r3-diag` = 16 runs in 6
blocks:**

| arm (job label) | instances | runs | read by |
|---|---|---|---|
| A6@ext (`A6@r3`) | comp21-2idx, neos-4763324-toguru, neos-2746589-doon | 6 | rule 1 (toguru, doon), rule 2's precondition (comp21), rule 3 |
| A6F@ext (`A6F@r3`) | toguru, doon | 4 | rule 1 |
| A2@ext (`A2@r3`) | comp21, toguru, doon | 6 | rules 2 and 3 |

- `jobs/r3-diag.jsonl` regenerated by `itrea make-jobs r3-diag` (branch `r3-diag`, commit fe3424d): sha256
  `849c28671335ab73c3bc84c3d7de6305f6b475b8fc94b7c0454b652ebe172933` (replaces 2c5d50a9… of 16:4x). Run hashes
  unchanged: A6@r3 e54170304dad, A6F@r3 82a371305f88 (= A6F@r13x's), A2@r3 9c3031af075f. Box: 16 runs ≈ 0.9 h.
- Rule 1 is read on toguru and doon only; on the other four instances the note's §2 footprint / slowdown split stays
  indicative, as the note labels it. Validity reads 16 of 16. The A6F@r3 vs A6F@r13x replicate is printed on toguru and
  doon only. Rule 2's precondition no longer quotes "rule 1's comp21 row" (A6F does not run on comp21).

**Clarification — rule 2's thresholds and what its outcomes mean (thresholds unchanged):**
- **The source sentence in Risks is corrected.** cuOpt's own final objective at 120 s on comp21-2idx, from the run
  table: inside A9 under ext, **A9@r13x 149 / 146** (the only ext references); inside A9 on the old harness at full duty
  A9@bo1s 175 / 160 and A9@bo0 172; under active back-off A9@bo1 153 / **138** — which contradicts "stops improving at
  146–172 at every duty" —; forced duties one run each (bof12 151, bof25 161, bof50 147, old harness). Alone: 3-thread
  A6 on the old harness 114, 116, 123, 124, 128; A6F@r13x (ext) 133 / 117.
- **146 is the lower of the two ext references**, so the rule is lopsided: "SCIP is the cause" needs A2's cuOpt at or
  above A9's lowest ext run on both seeds, while "≤ 133 on both" is reached if A2 behaves like 3-thread A6 (114–128).
  Stated, not changed.
- **The "≤ 133 on both seeds" branch reads: the EA's presence remains** — its CUDA context, residual GPU work, the send
  callback (A2's cuOpt never sends, `pair.py`), host CPU and memory traffic on core 6, and SCIP's inputs from the EA —
  **and the MPS question goes to Berk with that wording: MPS is one candidate, not an attribution.** The design cannot
  separate these. `tools/d027_report.py` prints that text (outcome `EA PRESENCE`).
- **Failed runs:** this record keeps the letter (`exclude`: a failed run is not completed → INVALID) as its reading; D-025
  and D-026 read failed runs as outcomes of their arm (`keep`, D-022's precedent). The difference is deliberate: a
  diagnostic has no verdict to protect from a crash, and a missing cell leaves its rule unread.

## Amendment 2026-10-14 18:3x — chromaticindex1024-7 and rule 4, before any run (AgentRA, Claude Code (Anthropic Claude Opus 5.5))
Why: D-023's Result (18:1x) — phase Y at duty ½ did not give cuOpt its first incumbent back on chromaticindex under ext
(63.47 / 63.41 s vs A6F's 10.87 / 10.52 s). The run table shows cuOpt's first incumbent inside A9 at 62–66 s on every A9 run
on record (A9@r13x 65.26 / 65.30, A9@bo1s 65.31 / 65.46, A9@boY 63.47 / 63.41 under ext; A9@bo0 65.30, A9@a9p 65.35, A9@bof12
62.29 on the old harness) and early without the EA (A2 6.50 on the old harness, A6 3-thread 8.1–10.2, A6F under ext 6.5–10.9),
with the exceptions bof25 9.81 / bof50 8.33 and bo1 9.38 / 12.35 (old harness). chromaticindex is A9's largest loss to cuOpt alone
(C1 d −0.889) and no candidate R1–R5 acts on it. Whether the cause is the EA's presence, SCIP's or the 3-thread lane under ext is
read by 6 more runs in the same window, D-024 rule 1 (where the claim lives) and rule 4 (each read by rule 4):

| arm | instance | runs |
|---|---|---|
| A6@r3, A6F@r3, A2@r3 | chromaticindex1024-7 | 3 × seeds 0–1 = 6 |

**Set `r3-diag` = 22 runs in 8 blocks** (job file regenerated by `itrea make-jobs r3-diag`; its sha256 in the journal).

**Rule 4 (fixed now).** t = cuOpt's first-incumbent time (`cuopt_t_first`; no incumbent = 120 s), seeds 0–1, same window;
L = 2 × max(t of A6F@r3 on seeds 0, 1).
- (a) A6@r3's t ≤ L on both seeds → the 3-thread lane under ext is not the cause; then A2@r3's t ≤ L on both seeds → **EA
  PRESENCE (chromaticindex)**: the EA's presence (its GPU work, CUDA context, LP worker's CPU, send callback, memory traffic)
  delays cuOpt; the GPU-sharing / CUDA MPS question goes to Berk with that wording (MPS one candidate, not an attribution);
  A2@r3's t ≥ 50 s on both seeds → **SCIP PRESENCE (chromaticindex)**; otherwise inconclusive at n = 2.
- (b) A6@r3's t ≥ 50 s on both seeds → **3-THREAD LANE UNDER EXT (chromaticindex)**; the layout question goes to Berk.
- (c) Otherwise inconclusive at n = 2.
Missing or failed runs: as this record's validity (exclude: a missing cell leaves rule 4 unread). `tools/d027_report.py` applies it.
