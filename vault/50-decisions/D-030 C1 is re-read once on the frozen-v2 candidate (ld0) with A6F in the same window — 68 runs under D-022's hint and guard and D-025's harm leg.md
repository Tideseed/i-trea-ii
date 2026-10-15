# D-030 C1 is re-read once on the frozen-v2 candidate (ld0) with A6F in the same window — 68 runs under D-022's hint and guard and D-025's harm leg (2026-10-15, AgentRA on Berk's choice)

**Status:** decided by AgentRA (Claude Code, Anthropic Claude Opus 5.5) on Berk's choice of 2026-10-14 (question panel,
below) — **before any run of set `c1r-ld0` or `c1r-ysync-ld0`**; open to a superseding decision on evidence. **Not run.**
**Amended 2026-10-15 14:3x, before any run** (the section at the end: the harm leg's band read as the D-023 `band_i` its
rule cites — A9@r13x or A6F@r13x, never the new A9 —, so every threshold is fixed now; harm cannot fire on 4 of the 17;
null lines and markers beside the HARM list; the CUDA flag gates; the vs-SCIP line on the 17; frozen-v2 bound to the hash
that ran; cost 3.9–4.8 h; the split-block bullet corrected).
Code built on branch `c1-rerun` (worktree `../i-trea-ii-c1r`, commits `9e9e7de` + `c52d980` + `17b21a3`), not merged: a pool is running from the
main checkout (no commits to `src/` or `configs/` while a pool can launch runs). Development only
([[D-010 Develop the algorithm first on a small development set]]): every number it produces is descriptive, and it decides
nothing about C1 — **the go / no-go is Berk's** ([[D-019 v1 — deadline and performance target]], addendum). Issues: #23 (A9),
#30; milestone C1 — **Sat 2026-10-17** since D-019's addendum of 2026-10-15 13:4x (Berk: "Assume we are wrapping up phase 1 on
Saturday."; was Fri 10-23). A change to any rule below before the first `c1r-*` run is a dated amendment here.

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
- **Writer:** `itrea make-jobs c1r-rerun --a9 ld0|ysync-ld0` (default `jobs/c1r-<a9>.jsonl`; with `--set S`, `jobs/S.jsonl`
  since 14:3x, item 10 below); its run hashes are pinned
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
as written (ld0: CUDA's default read back; ysync-ld0: yield set, verified and held) and its LP decode gate. *(Amended
14:3x below: the CUDA flag is a sixth, gating leg; voids named; the LP decode gate stays beside.)*

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
   candidate is not presumed: he decides which configuration freezes."* *(Amended 14:3x below: the quote stopped before
   the rule's "(D-023's `band_i`)"; band_i = max(0.05, the spread of A9@r13x or of A6F@r13x), never the new A9's; harm
   cannot fire on rail01, cryptanalysiskb128n5obj16, doon or crna, so this leg cannot show harm on crna or rail01.)*
- **Reported, not criteria** (D-013 terms, as D-022's Result reports them): A9@c1r vs A6F@c1r (same window, equal
  footprint) and vs A0 (SCIP alone; sets `dev` / `classprobe`, earlier days) on the fixed 11 and on SCIP's weak set among
  them — improved k of n, worse, mean P before → after, mean d with its 95 % bootstrap interval, Wilcoxon, worst P after;
  **the 10-13 reading beside** (hint 0.827 = 0.585 / 0.707, guard +0.063), per instance and in total. *(Added 14:3x
  below: vs SCIP alone on SCIP's weak set among all 17, as D-019's C1 asks.)*
- **Failed runs:** outcomes of their arm (D-002 §6, `keep`), the reading that leads; the letter (`exclude`: a failed run
  is missing) printed beside.
- The hint is futility-only ([[D-024 Targeted first, wider only on a hint — staged testing for development and v1]] rule 2):
  it gates whether the line widens and declares nothing.

## What goes to Berk
`itrea c1r-report --a9 <the candidate that ran>` (after `itrea build-runs`, pinned, `OMP_NUM_THREADS=1`): validity;
**HINT present / absent** (the ratio); **GUARD holds / fails** (the mean d); **the HARM list** with each instance's P̄
(10-13 → new), both seeds, d_i and band_i; the D-013 lines; the 10-13 reading beside; and the line "go / no-go is
Berk's". **The go / no-go on C1 is Berk's** (D-019 addendum), and so is which configuration freezes when harm shows (D-025
amendment 3) or when ysync-ld0 is the one that ran. Nothing here writes `configs/frozen-v2.yaml`. *(Amended 14:3x
below: the HARM list carries its notes — ΔP, D-022's floor marker, the null reads — beside it, with the instances where
harm cannot fire and the null lines; the vs-SCIP line on the 17; voids; frozen-v2 bound to the hash that ran, item 7.)*

## Cost and windows
- **68 runs ≈ 3.9 h** at 3.44 min per run (itx-r13's rate, the gaps between runs included), plus the fan governor's rests
  and the window's open and close *(corrected 14:3x below: 3.9–4.8 h — 3.44 min already holds the rests; lease ≥ 4.5 h of
  runs)*; before C1 on Sat 2026-10-17 (D-019 addendum 13:4x: after D-029, before Saturday).
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

## Code (branch `c1-rerun`, commits `9e9e7de` and `c52d980`; the 14:3x amendment's code `17b21a3`)
`configs/itrea2-dev-rc-3way-ysync-ld0.yaml`; `make_jobs.c1_rerun` (`itrea make-jobs c1r-rerun --a9 …`, pinned run hashes);
`jobs/c1r-ld0.jsonl`, `jobs/c1r-ysync-ld0.jsonl`; `tools/c1r_report.py` (`itrea c1r-report`); tests: every configuration
hash pinned and ysync-ld0 = ld0 + yield only (`tests/test_cuda_sched.py`), the writer against the committed itx-r13 /
itx-r13b files (`tests/test_make_jobs.py`), the report on synthetic tables (`tests/test_c1r_report.py`), the GPU cold-cache
case above (`tests/test_lp_decode_gate.py`, not run: GPU). Whole non-GPU suite, pinned (`taskset -c 10-14`,
`nice -n 19`, `OMP_NUM_THREADS=1`, no GPU), 2026-10-15 13:5x: **1,339 passed, 2 skipped, 1 xfailed** (the known itx-r13
writer, #37), 104 GPU tests deselected; after `c52d980` the affected files again: 156 passed, 1 skipped, 1 xfailed.
AGENTS.md map, README and the skills
(`itrea-window`, `itrea-new-experiment`, `itrea-result-note`) updated on the branch.
`c52d980`: the run-id test ignores the c1r sets' own rows in the run table (so it stays green after the re-read
runs). `ALL_CONFIG_HASHES` fails on any new file under `configs/` by design: `configs/frozen-v2.yaml` (Saturday) enters
that table in the commit that adds it.

## Not decided here
- The same C1 set at seeds 2–3 (D-019 addendum 13:4x, Friday, box time allowing: "a widening after C1's hint, D-024
  rule 3; its rules written before its runs") is not part of this record; it needs its own rules first (the writer
  takes `seeds`, the CLI does not expose them yet).
- D-029's unread warm-up run (its 12:2x amendment, after the start-of-pool disturbance on r1-ysync seed 1) is not part of
  these 68; adding one to the job file is a dated amendment here, before any run.
- A block whose two runs land in different windows (only if a pool halts inside a block; a deadline never splits one)
  fails the window leg (INVALID). Its remedy — rerun that block whole in a set of its own (`make-jobs c1r-rerun --set
  c1r-<a9>-2`, filtered to the block) with the report reading it in place — is not built; it would be a dated amendment.
  *(Corrected 14:3x below, item 9: the pool never leaves a split block — `Pool.prepare()` voids it and reruns it whole on
  resume; no remedy is owed and the recipe is withdrawn.)*
- Links: [[D-013 Aim — lift each solver's weakest cases]], [[D-002 Pre-registration — hypotheses, arms, analysis]],
  [[2026-10-14 C1 diagnosis and improvement candidates]].

## Clarification and amendment 2026-10-15 14:3x — before any run (AgentRA, Claude Code (Anthropic Claude Opus 5.5))
An adversarial review of the build (Claude Code subagents; commits `9e9e7de` + `c52d980`) compared it with this record and
with the rules it cites. Every finding was checked against the run table, the pool's code and the cited records; all held.
The rule changes and corrections below are fixed **before any `c1r-*` run** (none exists; the job files are unchanged:
sha256 as in the table above). Code: branch `c1-rerun`, commit `17b21a3`.

**1. The harm leg's band — the citation, read as written (rule change).** D-025 amendment 3 reads in full: *"harm iff
d_i < −band_i, d_i = (P(A9@r13x) − P(new)) / max(both, 10⁻³) on seed means, band_i = max(0.05, the larger seed-0 vs
seed-1 bounded difference of either arm) (D-023's `band_i`)"*. The quote under "Rules" item 3 stopped before the
parenthesis. The two halves conflict: "either arm" of this comparison is A9@r13x and the new A9, but D-023's `band_i` is
*"the larger seed-0 vs seed-1 bounded difference of the control or of A6F"* (`backoff._band`) — never the treatment. The
build followed "either arm", so the candidate's own seed spread widened its own harm band. On the same-configuration
replicate A9@r2ctl (10-15, A9@r13x's run hash) kumeu read d −0.368 with band 0.598 taken from the replicate's own seeds
(0.2315 / 0.093): a loss that size would have been hidden. The order of this set makes it systematic: the new A9 runs
second at even seeds and first at odd ones, so its seed-0 vs seed-1 spread carries the position effect by design.
**Fixed reading: band_i = max(0.05, the bounded seed spread of A9@r13x (D-023's control), that of A6F@r13x (D-023's
A6F: the cuOpt-alone arm that ran beside the control, 10-13)).** Why: it is the construction the amendment cites; every
band in this project keeps the treatment out (D-018, D-022, D-023, D-025's P band); and both arms are runs of 10-13, so
every band_i and every threshold are numbers fixed **now**, before any run. A6F@c1r is not in the band: it is the in-set
null line (item 3). The report prints, beside each instance and not as criteria, the band of the first reading (A9@r13x
or the new A9) and the band with A6F@c1r in A6F@r13x's place.

Harm fires iff P̄(new A9) > the threshold P̄(A9@r13x) / (1 − band_i) (from 10-13's runs; seed means):

| instance | A9@r13x s0 / s1 | A6F@r13x s0 / s1 | band_i | harm iff P̄ new > |
|---|---|---|---|---|
| glass4 | 0.2212 / 0.2108 | 0.3064 / 0.2890 | 0.057 | 0.2290 |
| sing44 | 0.4412 / 0.4490 | 0.5241 / 0.4570 | 0.128 | 0.5105 |
| radiationm18-12-05 | 0.0109 / 0.0111 | 0.0085 / 0.0085 | 0.050 | 0.0115 |
| neos-960392 | 0.0654 / 0.0502 | 0.5260 / 0.8932 | 0.411 | 0.0981 |
| neos-3024952-loue | 0.5629 / 0.5648 | 0.7610 / 0.7502 | 0.050 | 0.5936 |
| fhnw-binpack4-48 | 0.0079 / 0.0145 | 0.0041 / 0.0069 | 0.458 | 0.0207 |
| rail01 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.050 | cannot fire (1.053) |
| ca__Scenario400_reserves_1 | 0.3912 / 0.3959 | 0.5872 / 0.5856 | 0.050 | 0.4142 |
| item_placement_9903 | 0.3575 / 0.3700 | 0.5439 / 0.5265 | 0.050 | 0.3829 |
| neos-3004026-krka | 0.0026 / 0.0034 | 0.0056 / 0.0056 | 0.240 | 0.0039 |
| neos-3656078-kumeu | 0.0937 / 0.1115 | 0.1111 / 0.0784 | 0.294 | 0.1454 |
| chromaticindex1024-7 | 0.5438 / 0.5441 | 0.0544 / 0.0668 | 0.185 | 0.6678 |
| cryptanalysiskb128n5obj16 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.050 | cannot fire (1.053) |
| comp21-2idx | 0.5519 / 0.5434 | 0.5114 / 0.4965 | 0.050 | 0.5764 |
| neos-4763324-toguru | 0.6303 / 0.5759 | 0.5386 / 0.5593 | 0.086 | 0.6600 |
| neos-2657525-crna | 0.7518 / 0.1704 | 0.7272 / 0.7859 | 0.773 | cannot fire (2.035) |
| neos-2746589-doon | 1.0000 / 1.0000 | 0.7925 / 0.9963 | 0.205 | cannot fire (1.257) |

**2. Where the harm leg cannot fire (correction).** "crna and rail01, the most exposed to wake latency, are first read
here" (item 3 above, and D-025 amendment 3's "are read there by this leg") overstates it: **harm cannot fire on 4 of the
17** whatever the new A9 does, under either band reading. On rail01, cryptanalysiskb128n5obj16 and neos-2746589-doon
A9@r13x has P = 1 on both seeds, so P(new) ≤ 1 gives d ≥ 0; on neos-2657525-crna A9@r13x's seeds 0.7518 / 0.1704 give a
band of 0.773, so harm would need P̄ > 2.03. Wake-latency harm on crna and rail01 (relevant if ysync-ld0 runs) is
therefore not checked by this leg. Reported beside, not criteria: on these four, each seed's P and the EA's generations
10-13 → new (A9@r13x: crna 855 / 299, rail01 383 / 377, crypt 246 / 252, doon 31 / 30). They gate nothing; a rule on them
would need its own dated amendment before any run.

**3. Noise beside the harm list (reported, no rule change).** The cross-day leg fires on day-to-day noise. Under the fixed
reading, the same-configuration A9 replicates of 10-15 (A9@r1ctl, A9@r1ctl-ysync, A9@r2ctl: A9@r13x's run hash) against
A9@r13x fire on **2 of 15 instance reads** (A9@r2ctl: glass4 d −0.061 vs band 0.057; kumeu −0.368 vs 0.294) — about 13 %
per read; over the 13 instances where harm can fire, at least one false flag on noise alone is more likely than not
(1 − 0.87¹³ ≈ 0.84; independence assumed, rough). On floor-level instances small absolute changes fire: radiationm18-12-05
at ΔP +0.0005, neos-3004026-krka at +0.0009 (the table above). So the report prints, beside the HARM list and never
changing it: the null lines — the same d_i / band_i on cuOpt alone (A6F@c1r against A6F@r13x: the identical job on two
days) and on the three same-configuration A9 replicates against A9@r13x, with their counts —; for each harm, the raw ΔP
with D-022's floor marker when |ΔP| ≤ 0.005, the replicates' d on that instance (and whether one reached as low) and
cuOpt alone's d. D-022's one-sided floor limb (ΔP ≤ +0.005) was considered as a rule here and **not adopted**: the leg
decides nothing — any harm goes to Berk with its numbers — and a marker keeps D-025's rule as it was fixed. For scale
(a preview, not a reading): ld0's own runs in `r2-ld` (A9@r2ld, seeds 0–1, five instances) against A9@r13x fire on none.

**4. The CUDA flag gates (rule change).** A sixth validity leg, as D-025 and D-026 gated it: every A9@c1r run's flag as the
candidate's file writes it — ld0: CUDA's default read back (`cuda_sched.state` "read", active "auto"; r2-ld's A9@r2ld
runs read so), ysync-ld0: yield set, verified, active and still held at the end — and timing events not blocking;
otherwise INVALID, leg named. A failed A9 run without an EA summary is noted, not charged (an outcome of its arm under
`keep`); a completed one without a summary fails the leg. Why: "the configurations as the job file" checked only the run
hash, which carries the YAML's `cuda_sched`, not whether the context took it. The LP decode gate stays beside.

**5. Voids (clarification).** The pool moves a block aside (`void/`, logged in `void.jsonl`) and reruns it whole when one
of its runs is voided (a thermal or cap void, a block found split on resume); build-runs marks a GPU-health void on the
whole block (`block_void`). Voided rows are never read, and the complete leg needs every one of the 68 present with
P(120) — D-022's "every run completed or a recorded void rerun"; a block voided and not rerun reads as missing
(INCOMPLETE). The report prints the set's `void.jsonl` count and reasons beside the validity line (not gating).

**6. The vs-SCIP line on the 17 (reported, added).** D-019's C1 asks for "an interim A9 result on the 17 … vs SCIP alone and
cuOpt alone, reported in D-013 terms"; the line under "Reported, not criteria" reads A0 on the fixed 11 (D-022's reading
of 10-13). Added beside it: D-013 terms on SCIP's weak set among all 17 (A0 P̄ ≥ 0.5: 15 of 17, all but glass4 and
radiationm18-12-05), for this set and for 10-13 (10-13: improved 12 of 15, worse 0, mean P 0.889 → 0.473, mean d +0.465
[+0.293, +0.642]; the fixed-11 line read 8 of 11, 0.880 → 0.585, d +0.331).

**7. frozen-v2 is bound to what ran (rule).** The re-read describes frozen-v2 only if `configs/frozen-v2.yaml`'s EA hash
**and** A9 run hash under ext equal those of the candidate that ran (ld0 `a05af2c732dc` / `341267810eed`; ysync-ld0
`835c7c5a249c` / `f345de0ab936`; `Cfg.hash()` drops `name`, so a copy matches); otherwise Berk is told before the go /
no-go is quoted for frozen-v2. `tests/test_cuda_sched.py::test_frozen_v2_is_a_configuration_c1_was_re_read_on` (skips
while the file does not exist, fails otherwise) and the report's beside line check it.

**8. Cost and the lease (correction).** "68 runs ≈ 3.9 h at 3.44 min per run …, plus the fan governor's rests" counts the
rests twice: 3.44 min is itx-r13's span over its 74 runs, rests included (and six short A9@r13 runs pull it down). Measured
from run.json epochs: 2.0–2.1 min per run launch to end (itx-r13 window 2 2.06, r2-ld 2.09, r1-ysync 1.99), gaps 0.1–0.4 min
per window, and a fan-governor rest of ≈ 30 min every 14–20 runs (itx-r13: 3 in 58, then 1 in 16; r2-ld 2 in 28). **68 runs
≈ 141 min + 3–5 rests = 3.9–4.8 h.** Size the lease (`--until`) at ≥ 4.5 h of runs plus the pool's 10-min open margin and
the window's open and restore; a shorter one leaves the last blocks to a second window (allowed, whole blocks).

**9. A split block (correction to "Not decided here", third bullet).** The pool never leaves one: a stop (deadline, lease,
stop file) only prevents new blocks, and on resume `Pool.prepare()` moves a half-finished block to `void/` (logged in
`void.jsonl`) and reruns it whole, so both runs share one `window_id`. The window leg guards only against a split that
bypasses the pool. No remedy is owed; the suggested `--set c1r-<a9>-2` recipe is withdrawn.

**10. The writer's default path (code).** `make-jobs c1r-rerun --set S` without `--out` now writes `jobs/S.jsonl`; before,
it overwrote the committed `jobs/c1r-<a9>.jsonl` with another set's content.

**What goes to Berk, as amended:** validity (six legs); HINT present / absent; GUARD holds / fails; the HARM list with
each instance's P̄ (10-13 → new), both seeds, d_i, band_i, ΔP and its notes; the instances where harm cannot fire with their
seeds and generations; the null lines; the D-013 lines (vs A6F, vs A0 on the fixed 11 and on SCIP's weak set among the 17);
voids; frozen-v2's hash; the 10-13 reading beside; "go / no-go is Berk's".

**Tests (pinned: `taskset -c 10-14`, `nice -n 19`, `OMP_NUM_THREADS=1`, no GPU), 2026-10-15 14:4x:** whole non-GPU suite **1,346 passed, 3 skipped, 1 xfailed** (the known itx-r13 writer, #37), 104 GPU
tests deselected; the new skip is the frozen-v2 test (no file yet). New / rewritten tests: `tests/test_c1r_report.py`
(the band reading, thresholds and cannot-fire, null lines and harm notes, the vs-SCIP line on the 17, voids and
frozen-v2, the flag leg), `tests/test_make_jobs.py` (the `--set` default path), `tests/test_cuda_sched.py` (frozen-v2).
Read-only on main's run table the report reproduces 10-13's 0.827 / +0.063 and the numbers above; this set INCOMPLETE,
0 of 68, as expected. Job files unchanged (sha256 above).

## Which candidate runs (2026-10-15 15:0x, AgentRA, before any c1r run)
D-029 adopted ysync (15:0x), so the frozen-v2 candidate is **ysync + ld0** (`configs/itrea2-dev-rc-3way-ysync-ld0.yaml`, A9 run hash
f345de0ab936). The combination's approval: Berk (13:5x) "go ahead with recommendations" — the recommendation stated to him was to re-read
C1 on ysync + ld0 if D-029 adopts, and, if the harm leg fires there, to re-read C1 on ld0 alone on Friday and freeze the variant
without harm (what freezes stays his). GPU check before the first timed run (inside the lease, no pool running, 15:07): A9-layout
cold-cache test for ysync-ld0 **passed** (decoder gate lag 0.135 s, `concat_parts_max` 2, nothing compiled after t0). `c1-rerun`
merged d3fe14f; affected non-GPU tests 256 passed. **`jobs/c1r-ysync-ld0.jsonl` (sha256 50108596…5588) started 15:09:55** in
W-1015 (queue item 5, `--layout pair --deadline 18:35`); unfinished blocks continue in Friday's window (whole blocks).
