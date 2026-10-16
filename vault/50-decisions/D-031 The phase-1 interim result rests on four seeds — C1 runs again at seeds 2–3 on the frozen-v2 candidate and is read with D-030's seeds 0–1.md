# D-031 The phase-1 interim result rests on four seeds — C1 runs again at seeds 2–3 on the frozen-v2 candidate and is read with D-030's seeds 0–1 (2026-10-15, AgentRA on Berk's "go ahead with recommendations")

**Status:** RESULT 2026-10-16 13:1x: HINT present 0.798, GUARD holds +0.045 on four seeds (136 runs VALID); harm flags glass4 (drift per D-032) and comp21 (edge) beside (§ Result). decided by AgentRA (Claude Code, Anthropic Claude Opus 5.5) under Berk's "go ahead with recommendations"
(2026-10-15 13:5x) and the plan of [[D-019 v1 — deadline and performance target]]'s addendum of 2026-10-15 13:4x —
**before any run of set `c1r-ysync-ld0-s23`**; open to a superseding decision on evidence. **Not run.** A change to any
rule below before the first run of the set is a dated amendment here. Development only
([[D-010 Develop the algorithm first on a small development set]]): every number it produces is descriptive. **Nothing here
adopts anything: this is the phase-1 interim result, and the go / no-go on C1 is Berk's** (D-019 addendum). Issues: #23
(A9), #30; milestone C1 — Sat 2026-10-17. *(Revised the same afternoon after a review, before commit and before any run;
the list is at the end.)*

## Why
- **Berk, 2026-10-15 13:4x:** *"Assume we are wrapping up phase 1 on Saturday."* D-019's addendum (AgentRA's plan, defaults
  unless Berk says otherwise) lists the runs before Saturday: D-029, then the C1 re-read on the frozen-v2 candidate (D-030),
  *"then — box time allowing, Friday — the same C1 set at seeds 2–3 so the phase-1 result rests on four seeds (a widening
  after C1's hint, D-024 rule 3; its rules written before its runs)."* **Berk, 13:5x:** *"go ahead with recommendations."*
  This record is those rules.
- **Two seeds are thin for a go / no-go.** D-030's reading rests on seeds 0–1. Its own null line (the same-configuration A9
  replicates of 10-15 against A9@r13x) fires on 2 of 15 instance reads, and single instances swing between seeds
  (A9@r13x on neos-2657525-crna: 0.752 / 0.170). The mean of four seeds has half the variance of the mean of two (seeds
  independent); the hint and guard below are computed on those means.
- [[D-024 Targeted first, wider only on a hint — staged testing for development and v1]] rule 3 ("wider only after a hint")
  is why this widening waits for C1's hint on the candidate (start conditions below); rule 2: the hint stays futility-only,
  and because the widening is conditional on that hint, the four-seed estimate is slightly optimistic — so the
  stage-2-only figure (seeds 2–3 alone) is printed beside it as the selection check (Rules 1).

## When it runs — start conditions (fixed now)
1. **After D-030's 68 runs have completed and been read** (`itrea build-runs`, then `itrea c1r-report --a9 ysync-ld0
   --json F`, F kept and named in the journal — Rules, "Seeds 0–1 against D-030's reading"); D-030's remaining blocks
   run first in Friday's window. **If D-030 reads INCOMPLETE** (a block skipped after two thermal or cap voids, R26's breaker, box
   time), its missing blocks run first — the same job file and set, whole blocks — and D-031 starts only once D-030 reads
   68 of 68 and conditions 2–3 hold: the four-seed result needs all 136 runs, so no D-031 run can become the result while
   D-030 is incomplete.
2. **D-030's reading is not INVALID.** If it is, D-031 does not start; the failing leg goes to the journal and to Berk.
3. **D-030's HINT is present** (ratio ≤ 0.95 on the candidate; D-024 rule 3). If it is absent, D-031 does not run unless
   Berk asks for it, and the phase-1 interim result is D-030's reading. (This gate is why the four-seed ratio is
   optimistic given that it is read at all — Rules 1.)
4. **If D-030's harm leg fires, the default is the C1 re-read on ld0 alone** (`jobs/c1r-ld0.jsonl`): it is the
   recommendation Berk approved at 13:5x (D-030, "Which candidate runs": *"if the harm leg fires there, to re-read C1 on
   ld0 alone on Friday and freeze the variant without harm (what freezes stays his)"*). Only one of the two 68-run sets
   fits before Saturday, so **D-031 then does not run unless Berk asks for it instead**; he is told when D-030 is read,
   and the window does not wait for an answer. Seeds 2–3 on ld0 would be a new record. Such a file would share these 34
   A6F@c1r run ids (A6F@c1r is one job in both, and a run id carries no set), as D-030's two files share theirs at seeds
   0–1; the run table would mark the second copy `#<set>`.
5. Daytime window only, the lease held by this project, both LLM servers stopped (a harness), the GPU gates as every
   pool (AGENTS.md). No new GPU code: the configuration is D-030's, whose cold-cache check passed at 15:07.

## The runs
- **Set `c1r-ysync-ld0-s23`, job file `jobs/d031.jsonl`, sha256
  `f83e6888934ac3b2e243be880a8c823ae83b787462982c5d5f59b316dc9a3198`** — the 17 development + class-probe instances
  (glass4, sing44, radiationm18-12-05, neos-960392, neos-3024952-loue, fhnw-binpack4-48, rail01,
  ca__Scenario400_reserves_1, item_placement_9903, neos-3004026-krka, neos-3656078-kumeu, chromaticindex1024-7,
  cryptanalysiskb128n5obj16, comp21-2idx, neos-4763324-toguru, neos-2657525-crna, neos-2746589-doon) × {A9@c1r, A6F@c1r} ×
  **seeds 2–3**, 120 s, harness ext, split dev, one block per instance × seed = **68 runs in 34 blocks**.
- **The same jobs as D-030's file at other seeds.** A9@c1r = `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` (EA hash
  `835c7c5a249c`, A9 run hash under ext `f345de0ab936`); A6F@c1r = A6F@r13x's job (run hash `82a371305f88`). Every field of
  every job equals D-030's job of the same instance and arm except seed, set, stratum, block and position (checked by
  `tests/test_d031_report.py`).
- **Labels kept, run ids new.** The labels are D-030's (A9@c1r, A6F@c1r), so the four seeds pool under one label per arm;
  each run id is new **by its seed alone** (`…__s2__…`, `…__s3__…`; `pair.run_id_for` is
  `<instance>__<label>__s<seed>__<budget>s__<run hash>` — the set names only the run directory `traces/raw/<set>/<id>`)
  — D-029's precedent for its treatment label, and D-022's itx-r13f. Checked 2026-10-15 15:1x: none of the 68 ids is in
  `datasets/runs.parquet` (1,349 rows, built 15:06), in any of the 47 committed job files (1,789 ids) or under
  `traces/raw/`.
- **Order — alternated by seed, as D-030's file:** A6F first at even seeds (2), A9 first at odd ones (3); over seeds 0–3
  every instance runs each order twice.
- **No warm-up run, as D-030** (stated 2026-10-15 after the review, before any run). The pool takes one job file, so
  D-031 is a fresh pool start on Friday, possibly seconds after D-030's remaining blocks end — the condition that D-029's
  12:2x amendment met with an unread warm-up (the start-of-pool disturbance on r1-ysync seed 1). D-030 declined one
  (its "Not decided here"); D-031 does the same, so its jobs stay D-030's but for the seed and the set. **Launch with the
  pool's default `--order-seed`:** the first block is then fhnw-binpack4-48 seed 2 with A6F@c1r first (cuOpt alone, on
  an instance that counts only for the guard), then that block's A9@c1r run — the same opening as D-030's file
  (fhnw-binpack4-48 seed 0, A6F first). Adding a warm-up would give every job a `priority`, change the file's sha256 and
  move the pins in the tool and here: a dated amendment before the first run.
- **Writer:** `OMP_NUM_THREADS=1 .venv/bin/python tools/d031_report.py --write-jobs jobs/d031.jsonl` — one call of D-030's
  writer, `make_jobs.c1_rerun("ysync-ld0", seeds=(2, 3), set_="c1r-ysync-ld0-s23")` (`make-jobs c1r-rerun` has no
  `--seeds`; no `src/` edit). The configuration path and both run hashes are pinned literals in the tool, which refuses when
  `make_jobs`' table moved and refuses any run id already used (job files, run table, run directories).
- **Dry-run** (pinned to cores 10–14, `nice 19`, 15:22, while D-030's pool ran):
  `python -m itrea.experiments.pool jobs/d031.jsonl --layout pair --dry-run` → 34 blocks, 68 runs, lane g0 (X925 5–6,
  A725 1–3, GPU); memory A9@c1r 97.0 GB, A6F@c1r 69.0 GB per run (MemAvailable 105.5 GB at that moment); the fan governor
  allowed a GPU run; **preflight passes** (lease, serving, freeze, harness).

## Validity — one gate over the 136 runs, fixed before any run
`tools/d031_report.py`: **D-030's set** — D-030's own six legs, `tools/c1r_report.py`'s `reading` unchanged (complete 68 of
68 with P(120); harness ext; each run id its job's; no compile on the clock — the project's CUDA cache on every run, the
CuPy cache on A9 —; the two runs of a block in one window; each A9 run's CUDA flag as the candidate writes it: yield set,
verified and held to the end, timing events not blocking). **D-031's set** — the same six legs on its 68, with the expected
ids from its writer. **VALID iff both are (136 of 136); INVALID if either is, the set and the leg named, and no four-seed
rule is read; otherwise INCOMPLETE (n of 136).** Each block runs in one window, so every A9 / A6F comparison is
same-window per block. **The halves are not day-separated:** D-030's seeds 0–1 span 10-15 and 10-16 (its pool stops at
18:35 today with blocks left for Friday), and D-031's seeds 2–3 run on 10-16, likely in the same window as D-030's last
blocks; the output names each half's windows and any window both share. A failed run is an outcome of its arm (D-002 §6,
`keep`, the reading that leads); the letter (`exclude`) is printed beside. Beside, not gating: each set's voids, windows,
the LP decode gate, frozen-v2's hash (D-030 item 7), the job file's sha256.

## Rules — D-022's and D-025 amendment 3's, on four seeds
**Seed means on four seeds:** P̄(i, arm) = (P_s0 + P_s1 + P_s2 + P_s3) / 4 — every seed weighted equally, which equals the
mean of the two halves' seed means. A0 and A9@r13x enter on their own seeds 0–1.

**Which run table governs (stated after the review, before any run):** every number below — each P̄, d_i, band_i and
threshold — is computed on the one `itrea build-runs` after D-031's last run (one table for both halves and for 10-13's
runs). D-030's amendment table gives them as of the 15:06 build. Sixteen of the 17 references do not move: 15 are frozen
optimal values (`ref_kind` opt) and item_placement_9903's is the screening best (it moves only with SCREEN / REF runs).
**ca__Scenario400_reserves_1's reference is provisional (`best_of_study`, not frozen):** every build-runs after a run that
finds a better ca solution re-scores every ca P, 10-13's included. A scratch rebuild after 12 of D-030's runs moved it
33840.04 → 33795.54, ca's P by +0.0004 to +0.0010 (71 rows) and ca's threshold 0.4142 → 0.4151; effect ≤ 0.001 in P, on
ca only. Freezing references is D-002's step 4, not this record's.
1. **HINT** ([[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]],
   second amendment): present iff the ratio of means of P(120) A9@c1r / A6F@c1r over **the fixed 11** (neos-2746589-doon,
   cryptanalysiskb128n5obj16, rail01, neos-2657525-crna, neos-3024952-loue, sing44, ca__Scenario400_reserves_1,
   item_placement_9903, neos-960392, neos-4763324-toguru, comp21-2idx; not re-selected) on the four-seed means is
   **≤ 0.95**. INCOMPLETE while any of its 88 runs is missing (no partial mean). **Futility-only** (D-024 rule 2): a hint
   declares no success. **Selection check (D-024 rule 2):** D-031 runs only after D-030's hint (condition 3), so the
   four-seed ratio includes the half that was kept because it read ≤ 0.95 and is slightly optimistic given that it is
   read at all; the **seeds 2–3-alone ratio is the stage-2-only figure** and is printed on THE line beside the four-seed
   ratio.
2. **GUARD** (same amendment): the mean bounded d of A9@c1r vs A6F@c1r on the four-seed means over **all 17** (both P = 1 at
   d = 0) **≥ −0.05**. INCOMPLETE while any of the 136 is missing.
   Both are computed by D-022's own functions (`tools/d022_report.py` `c1_hint`, `c1_guard`, `c1_interim`). Those
   functions check completeness on seeds 0–1 by name, so the report gates completeness on its own seeds first and hands
   them the rows with the seeds renumbered from 0 for the seeds 2–3 reading (their arithmetic is a mean over the rows they
   receive; nothing else changes). Tested against the definitions by hand and against D-022's functions called directly.
3. **HARM leg** (D-025 amendment 3, band as D-030's amendment of 14:3x fixes it), **cross-day**: per instance of the 17,
   d_i = (P̄(A9@r13x) − P̄(new)) / max(both, 10⁻³), where **P̄(new) is the four-seed mean of A9@c1r (seeds 0–1 from D-030's
   window(s), seeds 2–3 from D-031's) and P̄(A9@r13x) the mean of A9@r13x's seeds 0–1 of 10-13**; band_i = max(0.05, the
   bounded seed-0 vs seed-1 spread of A9@r13x, that of A6F@r13x) — 10-13's runs only, so **every band_i and threshold is
   the number in D-030's amendment table** (as of the 15:06 build — ca's moves with its reference, above; glass4 0.2290,
   sing44 0.5105, …; harm cannot fire on rail01, cryptanalysiskb128n5obj16, neos-2746589-doon and neos-2657525-crna);
   harm iff d_i < −band_i. Beside, not criteria: raw ΔP with D-022's floor marker (|ΔP| ≤ 0.005); per harm, the
   same-configuration A9 replicates' d of 10-15 on that instance (two seeds, against the same fixed A9@r13x reference)
   and cuOpt alone's d; the cuOpt-alone null line (the same rule on A6F@c1r's four-seed mean against A6F@r13x's seeds
   0–1); D-030's same-configuration replicate null rate (seeds 0–1).
   **What four seeds change, stated (corrected after the review, before any run).** A9@r13x's two runs of 10-13 are the
   reference in every reading, so their sampling error is a fixed offset per instance that more seeds of the new arm do
   not shrink. On noise alone a four-seed mean of the new A9 makes the leg fire **less** often where the same
   configuration's true mean lies below the threshold, and **more reliably** where it lies past it (a low 10-13 draw or a
   day-to-day shift). glass4 is such a case: threshold 0.2290; A9@r13x's run hash on other days reads 0.2272 (itx-m1 s0)
   and 0.2283 / 0.2317 (r2-ld), mean 0.2291 — at the threshold. D-030's 2-of-15 replicate rate is a two-seed figure, not
   a bound for this leg. The band stays as D-030 fixed it (a change would be a dated amendment before the first run); the
   notes above are what Berk reads beside each harm.
- **Reported, not criteria** (D-013 terms, D-022's blocks): A9@c1r vs A6F@c1r on the fixed 11 (same window per block,
  equal footprint); vs A0 (SCIP alone, sets `dev` / `classprobe`, seeds 0–1, earlier days) on the fixed 11, SCIP's weak set
  among them, and SCIP's weak set among all 17 (D-019's C1): improved k of n, worse, mean P before → after, mean d with its
  95 % bootstrap interval, Wilcoxon, worst P after — each over the instances with **every seed of both arms** at the
  reading's seeds, tagged "partial: k of n" when some lack one (never a mix of two- and four-seed means), and labelled
  "beside, not the result" when the 136 do not read VALID.
- **The halves, beside:** the same three rules on **seeds 0–1 alone** (= D-030's reading) and on **seeds 2–3 alone**, and
  **do the halves agree** — each half's hint and guard state, its harm list, and per instance the sign of d vs A6F in each
  half (same / opposite / a zero). Descriptive: no agreement threshold exists and none is applied; a disagreement is
  information for Berk, not a rule. The halves differ in day and window as well as seed (Validity), so a difference
  between them is not a seed effect alone.
- **Seeds 0–1 against D-030's reading, two checks.** (a) On the same run table, a line checks that the numbers equal
  `c1r_report`'s — a code check, blind to re-scoring. (b) **D-030's report is saved as JSON when D-030 is read**
  (condition 1), and `tools/d031_report.py --d030-json F` compares seeds 0–1 here with it — hint, guard, harm list, each
  threshold and A9@r13x's seeds — printing every difference with both numbers. A move on ca (and through ca on the hint
  ratio and the guard's mean d) is the re-scoring stated above; any other difference is a data change or a bug and goes to
  the journal before the result is quoted.

## The reading — what the tool prints
`itrea build-runs`, then (pinned, `OMP_NUM_THREADS=1`) `.venv/bin/python tools/d031_report.py --d030-json F [--json G]`.
It prints: the validity line over both sets and each set's legs; the per-instance table (four-seed P̄ A6F / P̄ A9 / d, d in
each half, the four-seed harm d_i, band_i, "fires >", and d_i in each half); **SEEDS 0-3** — HINT, GUARD, HARM (each harm
with its threshold and the replicates' and cuOpt alone's d beside) and the cuOpt null line, the header labelled "beside,
not the result (…, n of 136 runs)" unless the 136 read VALID; the cannot-fire instances' seeds and generations; **SEEDS
0-1 ALONE** (with the same-table line "equal (hint, guard, harm list)" against `c1r_report`, the comparison with the
saved D-030 JSON, the replicate null rate and the 10-13 reading beside); **SEEDS 2-3 ALONE**; **DO THE HALVES AGREE?**
with each half's windows; the D-013 lines (four seeds, the halves beside; "partial: k of n" where an instance lacks a
seed); and

**`PHASE-1 INTERIM RESULT (…): HINT … (ratio …, present iff <= 0.95; futility-only, D-024 rule 2 …), GUARD … (mean d
…); beside: harm …; selection check (D-024 rule 2, the stage-2-only figure): seeds 2-3 alone HINT … (ratio …)` — THE
line.** It carries the **four-seed hint and guard when the 136 runs read VALID**, with the seeds 2–3-alone ratio beside as
the selection check. Otherwise it carries D-030's reading (seeds 0–1) and names why (D-031 INCOMPLETE or INVALID, n of
136), with the four-seed and seeds 2–3 numbers that could be read printed beside and labelled, never as the result. If
D-030's own set is INVALID, there is no phase-1 interim result. **It is printed once**, for the reading that leads
(failed runs kept); when a run failed, the letter's line (failed runs excluded) is printed beside under its own label,
not as `PHASE-1 INTERIM RESULT`.

## What goes to Berk
The PHASE-1 INTERIM RESULT line (hint present / absent with its ratio — futility-only —, guard holds / fails with its
mean d) and on which seeds it rests; when four seeds lead, the seeds 2–3-alone ratio beside it as D-024 rule 2's selection
check, with the plain statement that the four-seed ratio is slightly optimistic because D-031 ran only after D-030's hint;
the four-seed harm list with each instance's P̄ (10-13 → new), ΔP, d_i, band_i, threshold, the replicates' and cuOpt
alone's d and the null line beside, and the statement that more seeds do not shrink the 10-13 reference's own error (Rules
3); whether the halves agree (the hint and guard of each, the sign flips by instance), noting that they differ in day and
window too; the comparison with the saved D-030 reading (any move on ca named); the D-013 lines vs cuOpt alone and vs
SCIP alone (fixed 11, SCIP's weak set among them and among all 17); validity, voids and frozen-v2's hash; and "go / no-go is
Berk's". **Berk decides the go / no-go on C1 and, if harm shows, which configuration freezes** (D-025 amendment 3). Nothing
here writes `configs/frozen-v2.yaml`.

## What makes it INCOMPLETE or INVALID
- **INCOMPLETE:** a run of the 68 missing — box time runs out before Saturday, a block skipped after its rerun (two thermal
  or cap voids, `--max-thermal-requeues`), R26's breaker halting the pool, or a failed run under the letter (`exclude`).
  A missing run of the 11 leaves the four-seed hint INCOMPLETE; any missing run leaves the guard INCOMPLETE; no partial
  means. The phase-1 interim result is then D-030's reading, D-031 reported beside as "INCOMPLETE (n of 68)".
  A D-030 that reads INCOMPLETE is start condition 1: its missing blocks run first, and D-031 waits for 68 of 68.
- **Fine, not INCOMPLETE:** the set split across windows **by whole blocks** (a deadline or lease end never splits a block;
  a half-done block found on resume is voided and rerun whole by the pool, D-030 item 9); a voided block rerun whole.
- **INVALID:** a failing leg on any present run of either set (harness, configuration, compile on the clock, a block across
  two windows, the CUDA flag not as written). No four-seed rule is read.

## Cost and windows
- **68 runs ≈ 4.3 h** at the pooled pace of the 142 C1-type 120-s runs of 10-13 and 10-15 (itx-r13's two windows, r2-ld,
  r1-ysync, r1f-ysync: 543.7 min = 3.83 min per run, the fan governor's rests included). Range **3.9–4.8 h**: launch to end
  2.0–2.1 min per run (68 × 2.08 ≈ 141 min), gaps under a second, plus 3–5 rests of ≈ 30 min (one every 14–20 runs; a rest
  every 10 runs, as r1f-ysync had, would make ≈ 5.8 h). Size the lease at ≥ 5 h plus the window's open and restore.
- **Friday 2026-10-16:** D-030's remaining blocks first (its pool stops at 18:35 today; at the same pace ≈ 50–55 of its 68
  runs finish today), its report, then D-031 if the start conditions hold — about 5–6 h of one daytime window.

## No further C1 runs in phase 1
After D-031 no further C1 runs are planned for phase 1: the phase-1 interim result rests on D-030's seeds 0–1 and D-031's
seeds 2–3 of one configuration. The only other C1 run set phase 1 can see is the ld0 re-read that Berk's 13:5x
recommendation keeps for the case that D-030's harm leg fires — instead of D-031, not beside it (start condition 4). More
seeds, more instances or another configuration are phase 2 and a new record.

## Not decided here
- Saturday's `configs/frozen-v2.yaml`, the phase-1 results note and the go / no-go (D-019 addendum; Berk's).
- A `priority` that runs the fixed 11's blocks first (so the four-seed hint could be read if the window ends early) was
  considered and not taken: D-031's jobs stay D-030's but for the seed and the set. If Friday's box time looks short, it is a
  dated amendment here before the first run.
- An `itrea d031-report` entry (an edit of `src/itrea/cli.py`; the script is the entry).
- Links: [[D-030 C1 is re-read once on the frozen-v2 candidate (ld0) with A6F in the same window — 68 runs under D-022's hint and guard and D-025's harm leg]],
  [[D-029 ysync gets one follow-up at seeds 2–5, judged on all seeds 0–5 with the failing seed kept — NOT ADOPTED ends R1 for v1]],
  [[D-026 lp_decode inside A9 on the EA's core after the LP worker (R2), development test]],
  [[D-025 A9 EA blocking sync (R1) — the EA stops spinning on the GPU so its LP worker gets core 6, development test]],
  [[D-013 Aim — lift each solver's weakest cases]], [[D-002 Pre-registration — hypotheses, arms, analysis]].

## Code and tests (2026-10-15, not committed — the main session commits)
`tools/d031_report.py` (the report and `--write-jobs`; it reuses `tools/c1r_report.py` — `reading` for D-030's set, the
flag / compile / decoder legs, `harm_bands`, `vs_a0_17`, `frozen_v2_state` — and `tools/d022_report.py`'s `c1_hint`,
`c1_guard`, `c1_interim`, `select`, `void_log`; no formula of D-022's is rewritten), `jobs/d031.jsonl`,
`tests/test_d031_report.py` (synthetic run tables: the job file is the writer's output with the pinned sha256, D-030's jobs
at seeds 2–3, new run ids, the writer's refusals; four-seed hint and guard equal D-022's functions and the definitions by
hand; the halves; the harm leg with D-030's bands and its seeds 0–1 equal to `c1r_report.harm_leg`; the six legs equal to
`c1r_report.reading`'s on D-030's set; INVALID / INCOMPLETE / failed-run fallbacks; the CLI). Pinned (`taskset -c 10-14`,
`nice -n 19`, `OMP_NUM_THREADS=1`, no GPU, `-m "not gpu"`), 15:2x: `tests/test_d031_report.py` + `tests/test_c1r_report.py`
**55 passed**; `tests/test_make_jobs.py` (scans every job file for id collisions) **55 passed, 1 xfailed** (the known itx-r13
writer, #37). Read-only on main's run table (built 15:06): both sets INCOMPLETE, 0 of 68 each, as expected; the 17
thresholds equal D-030's table; "seeds 0-1 here vs c1r_report's D-030 reading: equal".

**After the review (16:0x, pinned as above):** `tests/test_d031_report.py` + `tests/test_c1r_report.py` **59 passed** (four
new tests: D-013 lines on a partial D-031, the halves' windows, harm notes beside, `--d030-json`; the failed-run test also
checks that one line reads PHASE-1 INTERIM RESULT); `tests/test_make_jobs.py` **55 passed, 1 xfailed** (#37). Read-only
on main's table (15:06): both sets 0 of 68, the 17 thresholds as D-030's table, same-table check "equal". On the
reviewer's scratch rebuild (after 12 of D-030's runs) with `--d030-json` = `c1r_report`'s JSON of the 15:06 table, the
comparison prints exactly ca's move (threshold 0.414249 → 0.415088; A9@r13x 0.391198 / 0.395875 → 0.391999 / 0.396669)
and nothing else. `jobs/d031.jsonl` unchanged (sha256 above).

## Revised after review, before commit and before any run (2026-10-15 15:5x–16:0x, AgentRA)
A review of this record and its tool found twelve points; all were confirmed and applied here and in the tool. No rule
of D-022 or D-025 amendment 3 changed, and the job file did not change.
1. **D-013 lines on whole instances only** (tool): the "seeds 0-3" D-013 blocks used to average whatever runs existed,
   mixing 2-, 3- and 4-seed means under a four-seed label while D-031 was incomplete. They now read only instances with
   every seed of both arms, tagged "partial: k of n". The four-seed block and lines carry "beside, not the result" unless
   the 136 read VALID.
2. **Selection check and futility-only** (Why, Rules 1, the line, What goes to Berk): the seeds 2–3-alone ratio is
   named as D-024 rule 2's stage-2-only figure and is printed on THE line. The four-seed ratio is stated to be optimistic
   given condition 3.
3. **The harm leg on four seeds** (Rules 3): the claim "no more often than D-030's" was wrong. The 10-13 reference's
   error is fixed, so more seeds fire less often below the threshold and more reliably past it (glass4's same-hash runs
   average at its threshold). The 2-of-15 rate is a two-seed figure. Each harm now carries the replicates' and cuOpt
   alone's d beside it.
4. **ca's reference is provisional** (Rules, "Which run table governs"; the halves' two checks): the numbers are those
   of the one build-runs after the last run. The same-table "equal" line is a code check only, and `--d030-json`
   compares seeds 0–1 with D-030's report saved when it is read (condition 1).
5. **If D-030's harm fires, the default is the approved ld0 re-read** (condition 4); D-031 runs then only on Berk's word.
6. **Run ids are new by the seed alone**: the set is not in a run id. An ld0 file at seeds 2–3 would share the 34 A6F ids
   (condition 4).
7. **The halves are not day-separated** (Validity, the halves): the output names each half's windows and any shared one.
8. **D-030 INCOMPLETE** (condition 1): its missing blocks run first, and D-031 waits for 68 of 68.
9. **One THE line** (tool): with a failed run, only the kept reading prints `PHASE-1 INTERIM RESULT`; the letter's line
   is labelled beside.
10. **No warm-up, stated** (The runs): the same as D-030; launch with the default `--order-seed` (first run A6F@c1r
    fhnw-binpack4-48 s2, guard only).
The build summary's statement that the run ids are new "because the seed and the set differ" is withdrawn with point 6.
`tools/c1r_report.py` (D-030's committed tool) has the same unguarded D-013 blocks on an incomplete D-030. It was not
changed here; that is an open question for the main session.

## Confirmed by the main session 2026-10-15 16:1x, before any run (AgentRA)
The 16:0x text stands, including condition 4's default: if D-030's harm leg fires, the ld0-only re-read runs instead (the recommendation
Berk approved at 13:5x, "go ahead with recommendations"), and D-031 runs then only on Berk's word. Procedure owed on Friday: when D-030
is read, save `itrea c1r-report --a9 ysync-ld0 --json logs/d030-reading.json` and name it in the journal; D-031 is read with
`python tools/d031_report.py --d030-json logs/d030-reading.json`; D-031's pool is launched with the default `--order-seed`.

## Result (2026-10-16 13:1x, AgentRA): THE PHASE-1 INTERIM RESULT — HINT present 0.798, GUARD holds +0.045 (four seeds, 136 runs)
Set `c1r-ysync-ld0-s23`, pool `20261016-085742-499378` (08:57–12:55): **67 completed, 1 failed** (comp21-2idx A6F@c1r seed 2, cuOpt rc −11,
the third cuOpt-alone SIGSEGV on comp21 in four seeds). `itrea build-runs`; `python tools/d031_report.py --d030-json logs/d030-reading.json`
(JSON `logs/d031-reading.json`): **VALID, 136 of 136** (verdict reading: failed runs kept, D-002 §6).
- **PHASE-1 INTERIM RESULT (seeds 0–3):** **HINT present, ratio 0.798** (P̄ 0.550 / 0.690 over the fixed 11; ≤ 0.95; futility-only, it
  declares nothing) and **GUARD holds, mean bounded d +0.045** over the 17 (≥ −0.05). Selection check (stage-2-only figure): seeds 2–3 alone
  0.787 / +0.034; seeds 0–1 alone (= D-030) 0.809 / +0.050 — the halves agree (they also differ in window and day).
- **Harm leg (cross-day vs A9@r13x's seeds 0–1 of 10-13):** glass4 d_i −0.114 (P̄ 0.216 → 0.244; band 0.057) — D-032 read glass4 in one
  window as drift; **comp21-2idx d_i −0.051** (P̄ 0.548 → 0.577; band 0.050) — at the edge, with the same-configuration replicates' null lines
  far wider on comp21 (−0.353 on 10-15). Beside, not criteria; any harm goes to Berk with the go / no-go.
- **Reported (D-013 terms):** vs cuOpt alone (same window per block) on the fixed 11: improved 7, worse 3, P 0.690 → 0.550 (−20 %), d +0.249
  [+0.078, +0.441], Wilcoxon p 0.027 (seeds 2–3 alone: 7 / 1, −21 %); vs SCIP alone (A0, earlier days, seeds 0–1): fixed 11 8 / 0, −38 %;
  SCIP's weak set among the 17 (15): 12 / 0, 0.889 → 0.448 (−50 %), d +0.499 [+0.323, +0.676], p 0.0005.
- **Exclude reading** (the three crashed A6F runs left out): INCOMPLETE (comp21's A6F missing at seeds 0, 1, 2) — hint and guard unread; the
  kept reading counts cuOpt alone's crashes on comp21 against cuOpt alone (its P̄ there 0.692 vs 0.504 on 10-13), which favours A9 on comp21.
- **For Berk:** the go / no-go on C1 and what freezes (the candidate `configs/itrea2-dev-rc-3way-ysync-ld0.yaml`) are his (D-019 addendum,
  D-025 amendment 3), with these numbers and the two harm flags beside.
