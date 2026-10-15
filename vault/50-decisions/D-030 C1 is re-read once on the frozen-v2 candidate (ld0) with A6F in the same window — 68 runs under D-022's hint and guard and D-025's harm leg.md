# D-030 C1 is re-read once on the frozen-v2 candidate (ld0) with A6F in the same window — 68 runs under D-022's hint and guard and D-025's harm leg (2026-10-15, AgentRA on Berk's choice)

**Status:** decided by AgentRA (Claude Code, Anthropic Claude Opus 5.5) on Berk's choice of 2026-10-14 (question panel,
below) — **before any run of set `c1r-ld0` or `c1r-ysync-ld0`**; open to a superseding decision on evidence. **Not run.**
Code built on branch `c1-rerun` (worktree `../i-trea-ii-c1r`, commit `COMMIT_SHA`), not merged: a pool is running from the
main checkout (no commits to `src/` or `configs/` while a pool can launch runs). Development only
([[D-010 Develop the algorithm first on a small development set]]): every number it produces is descriptive, and it decides
nothing about C1 — **the go / no-go is Berk's** ([[D-019 v1 — deadline and performance target]], addendum). Issues: #23 (A9),
#30; milestone C1 (Fri 2026-10-23). A change to any rule below before the first `c1r-*` run is a dated amendment here.

## Why
- **[[D-026 lp_decode inside A9 on the EA's core after the LP worker (R2), development test]] ADOPT (2026-10-15 13:2x):**
  `configs/itrea2-dev-rc-3way-ld0.yaml` (EA hash `a05af2c732dc`; A9 run hash under ext `341267810eed`) is A9's and A3x's
  development default and the frozen-v2 candidate. C1's interim reading (D-022's Result, 2026-10-13: hint 0.827, guard
  +0.063) describes the old default `configs/itrea2-dev-rc-3way.yaml` (A9@r13x, run hash `28e5a5e9c725`), not what would
  be frozen.
- **Berk, 2026-10-14 16:0x (question panel; D-025's open question 1), verbatim choice: "Re-run C1 on frozen hash
  (Recommended)"** — the option's text: *"Re-read C1 on the exact frozen configuration, with cuOpt alone re-run in the same
  window: 68 runs ≈ 3.9 h of box. The go / no-go then describes what is frozen."* Recorded in
  [[D-025 A9 EA blocking sync (R1) — the EA stops spinning on the GPU so its LP worker gets core 6, development test]]
  (clarification and amendment 2026-10-14 17:4x) and in D-026's Result: C1 is re-read **once**, on the frozen-v2 hash.
- D-025 amendment 3 (2026-10-14 17:4x, before any run) fixed a harm leg for this re-read; D-026's amendment (item 2) says
  the one re-read covers both changes.

## The two candidates — exactly one runs
| candidate | configuration | EA hash | A9 run hash (ext) | job file (sha256) | set | runs when |
|---|---|---|---|---|---|---|
| **ld0** | `configs/itrea2-dev-rc-3way-ld0.yaml` | `a05af2c732dc` | `341267810eed` | `jobs/c1r-ld0.jsonl` (`6a358b88a07a18100c84fdb4089f9f0d7a1e7b33a5f31d4491d90014593fa32d`) | `c1r-ld0` | the default: D-026 ADOPT |
| **ysync-ld0** | `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` (new: `-ld0.yaml` + `host.cuda_sched: yield`, the only difference) | `835c7c5a249c` | `f345de0ab936` | `jobs/c1r-ysync-ld0.jsonl` (`50108596c9f9297ca89347311d5168a5a866fb2a4d920cd6905bec91e8995588`) | `c1r-ysync-ld0` | **only if** [[D-029 ysync gets one follow-up at seeds 2–5, judged on all seeds 0–5 with the failing seed kept — NOT ADOPTED ends R1 for v1]] ADOPTs ysync **and** Berk approves the combination |

- ysync and ld0 have **never run together**: D-029 tests ysync against `-3way.yaml`, D-026 tested ld0 against `-3way.yaml`.
  If D-029 adopts, frozen-v2 would combine two changes each tested alone; whether that combination freezes is Berk's call
  before the re-run (D-026's Result, D-029 "What ADOPT triggers"). If D-029 does not adopt, or Berk does not approve the
  combination, **ld0** runs. Never both (below: the two files share their A6F run ids).
- Whichever runs, the harm leg is the only check of what changed since 10-13, and the re-read cannot say which change
  caused a difference (D-025 amendment 3).

## The runs
- **D-022's C1 runs again**, as the C1 rows of `jobs/itx-r13.jsonl` + `jobs/itx-r13b.jsonl` (A9@r13x / A6F@r13x): the 17
  development + class-probe instances (glass4, sing44, radiationm18-12-05, neos-960392, neos-3024952-loue,
  fhnw-binpack4-48, rail01, ca__Scenario400_reserves_1, item_placement_9903, neos-3004026-krka, neos-3656078-kumeu,
  chromaticindex1024-7, cryptanalysiskb128n5obj16, comp21-2idx, neos-4763324-toguru, neos-2657525-crna,
  neos-2746589-doon) × {A9, A6F} × seeds 0–1, 120 s, harness ext, split dev, one block per instance × seed =
  **68 runs in 34 blocks**, set `c1r-<a9>`, both LLM servers stopped (a harness).
- **Labels A9@c1r / A6F@c1r — new run ids.** Checked 2026-10-15 13:3x against `datasets/runs.parquet` (1,329 rows) and
  every committed job file (1,687 ids): no collision. A9@c1r runs the candidate's file. **A6F@c1r is A6F@r13x's job**
  under the new label — every field but set / stratum / block / variant / position equal, so the same run hash
  `82a371305f88` — re-run in the same window as A9 (Berk's "cuOpt alone re-run in the same window").
- The A6F jobs of the two files are the same jobs, so their 34 run ids coincide: only one file runs. (Were both ever to
  run, the run table would qualify the second copy as `<run_id>#<set>` and the report reads by set.)
- **Order — alternated by seed, not the Williams row:** each block's two runs carry explicit positions, **A6F first at
  even seeds, A9 first at odd ones** (as `r1-bsync`, `r2-ld` and D-029). Why: each instance then runs both orders, so the
  position effect (heat carried over from the previous run) cancels inside each seed mean that the hint, the guard and
  the harm leg read; the pool's two-arm Williams row, fixed by a hash of instance × seed, would give 5 of the 17
  instances the same first arm on both seeds (sing44, radiationm18-12-05, neos-3024952-loue, rail01, neos-3004026-krka).
  itx-r13 ran in Williams order: a stated difference from 10-13, not a rule.
- **Writer:** `itrea make-jobs c1r-rerun --a9 ld0|ysync-ld0` (default `jobs/c1r-<a9>.jsonl`); its run hashes are pinned
  (A9 `341267810eed` / `f345de0ab936`, A6F `82a371305f88`) and it refuses a job whose hash moved. **Dry-run** of both
  files (pinned, `--layout pair --dry-run`, 13:3x): 34 blocks, 68 runs, lane g0; memory A9@c1r 97 GB, A6F@c1r 69 GB;
  preflight passes.

## Validity — one gate, fixed before any run
`tools/c1r_report.py`: **complete** (68 of 68, each with P(120)); **harness** ext on every run (run.json
`config.harness`); **configurations as the job file** (each run id is its job's: the run hash carries the EA configuration
hash, the harness and the footprint); **no compile on the clock** (the project's CUDA cache on every run; the CuPy cache
on A9, never charged to A6F — the runner's own rule, as D-022's report reads it); **the two runs of each block in one
window** (`window_id`). A failed leg → INVALID, the leg named, no rule read; a missing run alone → INCOMPLETE, each rule
reads what is there and is INCOMPLETE where it needs a missing run. Reported beside, not gating: each A9 run's CUDA flag
as written (ld0: CUDA's default read back; ysync-ld0: yield set, verified and held) and its LP decode gate.

## Rules — exactly as cited, fixed before any run
1. **HINT** ([[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]],
   second amendment 2026-10-13 15:3x): *"**Hint present** iff the ratio of means of P(120) A9@r13x / A6F@r13x over those 11
   (seed means, both seeds, A9 failures counted as A9 outcomes, D-002 §6 reading) is **≤ 0.95**"* — here A9@c1r / A6F@c1r.
   The weak set is **the fixed 11** (neos-2746589-doon, cryptanalysiskb128n5obj16, rail01, neos-2657525-crna,
   neos-3024952-loue, sing44, ca__Scenario400_reserves_1, item_placement_9903, neos-960392, neos-4763324-toguru,
   comp21-2idx), not re-selected on these runs. INCOMPLETE while a run of the 11 is missing.
2. **GUARD** (same amendment): *"mean bounded d of A9@r13x vs A6F@r13x over all 17 ≥ −0.05"* — here A9@c1r vs A6F@c1r, on
   seed means, both P = 1 at d = 0, no partial mean (INCOMPLETE while a run of the 17 is missing).
   The hint and the guard are computed by D-022's own functions (`tools/d022_report.py` `c1_hint`, `c1_guard`,
   `c1_interim`); on today's table the report reproduces 10-13's 0.827 and +0.063 from the run table.
3. **HARM leg** (D-025 amendment 3, 2026-10-14 17:4x): *"the re-read's A9 (the frozen-v2 candidate) against A9@r13x of
   10-13, labelled cross-day, per development instance: harm iff d_i < −band_i, d_i = (P(A9@r13x) − P(new)) / max(both,
   10⁻³) on seed means, band_i = max(0.05, the larger seed-0 vs seed-1 bounded difference of either arm)"* — either arm =
   A9@r13x and the new A9 (A6F never widens the band); every one of the 17 (crna and rail01, the most exposed to wake
   latency, are first read here). *"Any harm goes to Berk with the numbers beside the go / no-go, and the frozen-v2
   candidate is not presumed: he decides which configuration freezes."*
- **Reported, not criteria** (D-013 terms, as D-022's Result reports them): A9@c1r vs A6F@c1r (same window, equal
  footprint) and vs A0 (SCIP alone; sets `dev` / `classprobe`, earlier days) on the fixed 11 and on SCIP's weak set among
  them — improved k of n, worse, mean P before → after, mean d with its 95 % bootstrap interval, Wilcoxon, worst P after;
  **the 10-13 reading beside** (hint 0.827 = 0.585 / 0.707, guard +0.063), per instance and in total.
- **Failed runs:** outcomes of their arm (D-002 §6, `keep`), the reading that leads; the letter (`exclude`: a failed run
  is missing) printed beside.
- The hint is futility-only ([[D-024 Targeted first, wider only on a hint — staged testing for development and v1]] rule 2):
  it gates whether the line widens and declares nothing.

## What goes to Berk
`itrea c1r-report --a9 <the candidate that ran>` (after `itrea build-runs`, pinned, `OMP_NUM_THREADS=1`): validity;
**HINT present / absent** (the ratio); **GUARD holds / fails** (the mean d); **the HARM list** with each instance's P̄
(10-13 → new), both seeds, d_i and band_i; the D-013 lines; the 10-13 reading beside; and the line "go / no-go is
Berk's". **The go / no-go on C1 is Berk's** (D-019 addendum), and so is which configuration freezes when harm shows (D-025
amendment 3) or when ysync-ld0 is the one that ran. Nothing here writes `configs/frozen-v2.yaml`.

## Cost and windows
- **68 runs ≈ 3.9 h** at 3.44 min per run (itx-r13's rate, the gaps between runs included), plus the fan governor's rests
  and the window's open and close; before Fri 2026-10-23.
- **The run may split across windows by whole blocks:** a block (instance × seed, both arms) never splits — the pool's
  rule, checked by the report's window leg; the next window resumes the same job file (completed runs are skipped).
- Needs Berk: only the ysync-ld0 combination (above); ld0 runs in a normal window.

## Before the first timed run
- **ld0:** nothing new — its cold-cache test (`tests/test_lp_decode_gate.py::test_cupy_cold_cache_r2_variant_in_the_a9_layout[ld0-0]`)
  passed in W-1015's GPU checks (2026-10-15 07:08–07:17) and it ran 14 timed runs in `r2-ld` (11:17–13:16, VALID).
- **ysync-ld0 (a new configuration combination, never run): the GPU check in A9's layout, inside the lease, before its
  first timed run** — `pytest -m gpu "tests/test_lp_decode_gate.py::test_cupy_cold_cache_r2_variant_in_the_a9_layout[ysync-ld0-0]"`
  (added on `c1-rerun`): an empty CuPy cache; yield set, verified before t0 and held to the end, timing events not
  blocking; the LP worker done, the gate open on it, the decoder ready after it and decoded copies entered; nothing
  compiled after t0 (AGENTS.md "Cold-cache GPU test before the timed run") — plus the repository's GPU tests on the merged
  code.
- Both: `c1-rerun` merged into main after the running pool ends; the job file run from the main checkout.

## Code (branch `c1-rerun`, commit `COMMIT_SHA`)
`configs/itrea2-dev-rc-3way-ysync-ld0.yaml`; `make_jobs.c1_rerun` (`itrea make-jobs c1r-rerun --a9 …`, pinned run hashes);
`jobs/c1r-ld0.jsonl`, `jobs/c1r-ysync-ld0.jsonl`; `tools/c1r_report.py` (`itrea c1r-report`); tests: every configuration
hash pinned and ysync-ld0 = ld0 + yield only (`tests/test_cuda_sched.py`), the writer against the committed itx-r13 /
itx-r13b files (`tests/test_make_jobs.py`), the report on synthetic tables (`tests/test_c1r_report.py`), the GPU cold-cache
case above (`tests/test_lp_decode_gate.py`, not run: GPU). TEST_COUNTS. AGENTS.md map, README and the skills
(`itrea-window`, `itrea-new-experiment`, `itrea-result-note`) updated on the branch.

## Not decided here
- D-029's unread warm-up run (its 12:2x amendment, after the start-of-pool disturbance on r1-ysync seed 1) is not part of
  these 68; adding one to the job file is a dated amendment here, before any run.
- Links: [[D-013 Aim — lift each solver's weakest cases]], [[D-002 Pre-registration — hypotheses, arms, analysis]],
  [[2026-10-14 C1 diagnosis and improvement candidates]].
