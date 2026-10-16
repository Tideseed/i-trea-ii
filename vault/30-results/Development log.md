# Development log (D-010)

One row per (configuration, instance, seed) on the development set (`datasets/dev_set.csv`), 120 s unless stated.
Descriptive only. `cfg` = EAConfig hash; `P_EA` = the EA's own primal integral, `P_port` = portfolio (SCIP ⇄ EA),
`P_SCIP` = SCIP alone (A0) of the same seed; `ttf` = time to first feasible of the EA; `posts/acc` = EA posts and
SCIP-accepted posts. Change notes say what was changed and why, with the commit.

## Changes
| date | commit | change | why |
|---|---|---|---|
| 2026-10-06 | 4127f10 / e05886c | baselines: default config `9ac4a93bf93b`, A0 / A5 / A3 × 9 × seeds 0–1 + A6, 120 s, beside an idle [local LLM agent] (`--allow-serving`) | D-010 yardstick |
| 2026-10-06 | 40fecba | self-adaptive repair strategy (`strategy` section, off by default — the default hash stays `9ac4a93bf93b` and a disabled run is identical to one before the change, regression-tested on NumPy and CuPy): every individual carries (repair row scale l2/rhs) × (LP guidance off/on), inherited from parent A, redrawn with p = 0.1, each strategy ≥ 5 % of the offspring; ONE local search with a per-individual row-scale table (one repair call and one mutation call per generation, whatever the number of strategies); V stays on `ranks.viol_norm`. `configs/itrea2-adaptive.yaml` = default + `strategy.enabled` + `seeding.lp_source: highs_async` (hash `584ddee83747`). Review fixes (same day, before any run): the first version repaired each row scale in its own GPU call, one after the other (+13–39 % per generation with two scales — it would have biased a fixed-budget comparison against the fixed-strategy arms; now within 2–5 % of the strategy-off generation time on `set_cover(2000, 300, 0.02)`, P = 256); injected solver rows drew sorted strategies, so the best solver row got strategy 0 ~90 % of the time (now independent draws); solver finds were credited to a strategy (now `incumbents_solver` / `best_events_solver`). Also for every configuration: the warm-up now forces `violation_guided`'s draw over violated rows (one seed in eight compiled 14 kernels on the clock at generation 1; results unchanged). Not yet run on the development set | feasibility probe 1: the LP point and the rhs row scale each unlock instances the other cannot, and each hurts where the other helps — let selection choose per instance ([[I-TREA II algorithm]], "Self-adaptive repair strategy") |

| 2026-10-06 | 770df17 / c192bb5 | LP stream (D-011): SCIP streams good infeasible node LP points (2 % time budget, few fractional variables preferred); the EA ranks them (V of the nearest rounding, key) and turns each into 4 repaired children, ordinary survival only. `configs/itrea2-adaptive-lps.yaml` (`a2ad94e9188d`); with the c01 budget `4a96b872ffdc`. Reviewed before the first GPU run (3 medium findings fixed). Measured as `lps1` against `ad1` | Berk 15:2x: "when solver comes up with suitable good infeasible solutions (minimize infeasibility then maximize objective function) it might be able to inject to EA's population … We are aiming for more diversity in EA" |

| 2026-10-06 | (this commit) | `ranks.feas_alt_scale` (off by default, every hash unchanged): two-scale Pareto survival in the feasibility phase over (V_l2, V_rhs). Not yet run | fhnw-binpack4-48: rhs-repaired children die under l2 survival (ad1/lps1 final population 8–9 of 256); consult with GPT-6.1 Sol |
| 2026-10-06 | 4d044ac | aggregation of two-variable equality rows (`src/itrea/ea/aggregate.py`), exact; NOT wired in (payoff probe negative on radiationm) | radiationm/rail01 equality chains |
| 2026-10-06 | (this commit) | repair clock, Stages 0–1 of [[2026-10-06 Repair-depth design (workflow)]] (`repair_clock` section, off by default — every recorded hash unchanged: 9ac4a93bf93b, 584ddee83747, a2ad94e9188d, 4a96b872ffdc): a lane-step horizon per repair call (status 4) in the CPU feasibility jump and, host-side between launches, in the GPU wrapper (no kernel edit); per-call launch profiles and per-lane steps; the controller `ea/repair_clock.py` (levels 2^7–2^16 + "none", "none" until the population's first feasible member, counterfactual gain per level, decayed rates, 5 generations / 2 material gains / 1.25× to switch, probes, LP batch pinned, 10 s cap); engine wiring with one pairing helper, `rclock` events, `rc_*` gen fields, a summary block. `configs/itrea2-adaptive-lps-rc.yaml` = lps1 + the clock in SHADOW mode (plays "none": the search is lps1's — identical genealogy to the section off for the same generation count, tested on NumPy; in a timed run its host work can shift generation counts, so its runs are not lps1 runs). After three adversarial reviews (same evening), deviations from the synthesis plan, for D-012: the gain unit dg = min((cutoff − key) / max(\|cutoff\|, \|key\|, \|dual\|), 1) (the plan's unit gave ~1e9 per improvement over an incumbent near 0 without a dual bound — neos-960392); the phase ends with a feasible member of the EA's own (an injected solver row ended it); the immigrant calls after the survival are left out of the gain; a cut children call enters the cap's median as a censored time (the cap could never rise); shadow mode accumulates only what its decision would reveal (it never probed); plus an uncut CPU call reveals every level, and each call logs cutoff, dual_key and the improving lanes that did not end FEASIBLE. CPU suite only; the GPU wrapper's horizon/deadline path and its gpu-marked tests have NOT run. Not yet run | the repair depth is instance-dependent (lps1 wins 6/9, the default depth only item_placement); Stage 1 = A3@rc0 shadow on item_placement s0–1, glass4, sing44, neos-960392, loue, ca s0 with the design note's pass criteria (a)–(c) |
| 2026-10-07 | (this commit) | three-way arm **A9 = SCIP ⇄ EA ← cuOpt** ([[D-015 Use cuOpt's feasible solutions — three-way arm A9]]): cuOpt (as shipped, never receives) posts its incumbents on its own channels `to_ea_cuopt` / `info_cuopt` (`cuopt_run --send`, A9 only; it used to write SCIP's `to_ea` / `info`, where the latest message wins); `exchange.from_cuopt` (off by default, out of the hash while off: every recorded hash unchanged — 9ac4a93bf93b, 584ddee83747, a2ad94e9188d, 4a96b872ffdc, 26481ba0c675, 1d12ad490dd0; the runner turns it on for A9): the EA injects cuOpt's rows like SCIP's, counts cuOpt's incumbent in its external incumbent (min with SCIP's: Rank 0 threshold, immigrants, cutoff), records but does not use cuOpt's bound, posts to SCIP whatever beats SCIP's incumbent (so an unchanged cuOpt vector reaches SCIP); attribution by id (`cuopt_ids`): incumbent source `cuopt`, posts marked `POST_CUOPT` (bit 16), which SCIP records as source `cuopt` (not `ea`); runner: SCIP on X925 5, the EA on X925 6, cuOpt on A725 1–3 (3 threads), two CUDA processes in the memory gate (73 GB with the reserve); `build_runs`: A9 in the GPU / EA-GPU arm lists, cuOpt attribution columns. `configs/itrea2-dev-rc-3way.yaml` = the repair clock's Stage 2 configuration (rc + active, `1d12ad490dd0`). After three reviews (same evening): A9's matched control is the new arm **A3x** (A3 with A9's EA footprint — SCIP on X925 5, the EA on X925 6, no cuOpt; same file, same run hash as A3), NOT A3, whose EA runs on the three A725 cores with the `highs_async` LP worker beside it while A9's shares one X925 core with that worker (D-015 addendum); per-post cuOpt columns (`cuopt_posts_accepted` one per post, new `cuopt_posts_became_incumbent`; `solver_inc_from_cuopt` counts SCIP incumbent records like `solver_inc_from_ea`), the EA's verdict counters renamed `cuopt_rows_accepted` / `_rejected`, `reads_cuopt` decided from the EA's own record, a non-default cuOpt thread count in A9's run hash, and `crash_flags` no longer counts a process the runner stopped after a partner's proof (`stopped_by_runner`; no existing run has one). CPU suite only: cuOpt's `--send` on the new channels, the EA warm-up with the flag on CuPy and any A9 launch have NOT run. Not yet run | Berk 2026-10-07: "We can use cuopt's feasible solutions" — the class probe (D-014): the EA's own feasibility search found nothing on 5 of 8 new instances, where cuOpt alone found feasible points in seconds |
| 2026-10-08 | (this commit) | **Rank 1 as a distinct elite** ([[D-016 Rank 1 as a distinct elite — cuOpt's clearing rule, development test]], #25): `ranks.rank1_mode: clearing` chooses Rank 1 best-first by cuOpt's clearing rule (a feasible candidate joins only if it differs from every member already chosen in ≥ k integer variables, k from n_int as in cuOpt's `get_max_var_threshold`, at most ⌈y·P⌉ = 26 members, no twin rule, no ratchet); `ea/elite.py` (blocked device distances, one copy per 32 rows), `ranks.rank1_cap`, `r1_n / r1_dmax / r1_dmax_band / r1_gap` in every gen event in both modes, `r1_cleared` / `rank1_cleared` in clearing mode; the new kernels are warmed in both modes. Off by default; `rank1_*` in `_LATE_DEFAULTS` — every recorded hash unchanged (9ac4a93bf93b, 1d12ad490dd0). Adversarial review (4 lenses, 2 refuters each): 12 confirmed, none a blocker, all fixed before any run (warm-up, per-member syncs, no-integer fallback, size record, GPU test, and three changes to D-016's mechanism test). Test: `jobs/dev-cl1.jsonl` = 18 blocks × (A3@rc1d control re-run, A3@cl1, A3@tp1 size-only control) | Berk's "keep a percentage of good feasible solutions as well as the best … more diversity": measured, Rank 1 keeps 1–9 near-copies; cuOpt's population keeps up to 32 distinct ones ([[2026-10-08 cuOpt population and diversity — what transfers]]) |
| 2026-10-09 | 87bc759 (lp_decode 6be1c22…71eefc0) | **LP decode** ([[D-017 LP decode — re-optimise the continuous variables of selected candidates]], #27): `repair.continuous: lp_decode` — a persistent HiGHS worker process re-optimises the continuous variables of up to 4 candidates per generation with their integers fixed (polish feasible rows, complete super-optimal infeasible ones), copies enter as ordinary rows; refused in A9 / A3x. Two adversarial reviews (27 + 9 findings), three fix rounds; the warm-up now runs every crossover / mutation operator on 1/2/3-child batches (a cold-cache GPU test caught a one-child block-crossover kernel compiling after t0). **Adopted 2026-10-09 as the A3 development default** (`configs/itrea2-dev-rc-ld.yaml`, f030e23909ad) | cuOpt runs this LP after every recombination; FJ moves continuous values by jump values only — probe on stored vectors: sing44 −30 % objective |
| 2026-10-09 | a1fd414 … 4493045 | **A9 GPU back-off** ([[D-018 A9 — the EA backs off the GPU beside cuOpt (contribution-gated duty cycle)]], #23): section `backoff` (off by default; inert unless `exchange.from_cuopt`) — phase Y at reduced duty until the first incumbent, a busy-second patience ladder in phase P, a token bucket at two empty-queue points, shadow / forced modes, busy-time accounting for the repair clock and FJ; `itrea backoff-report`; review 21 findings (15 confirmed) fixed; stage 0 complete → PROCEED (D_Y ½, floor ⅛, 30 s, 40 / 40); stage-1 configs bo1 / bo1s | the A9 pilot lost to cuOpt alone through GPU contention (chromaticindex 65 s vs 8 s) |
| 2026-10-11 | 78ae305 | `backoff-report`: `_band` reads the stage-1 control `A9@bo1s` (it read the old control `A9@bo0`, so every stage-1 band fell back to A6's spread or 0.05); `--failed keep` (D-002 §6). Fixed before D-018 stage 1's verdict; the verdict is the same either way | a bug found while reading stage 1 |
| 2026-10-11 | 2a18819 | MIPfeas harness ([[D-020 MIPfeas harness — GAMS's clock and integral, paired calibration of cuOpt 26.08]], #29): arm A6m and `--layout full`, GAMS's integral ported exactly (`pt_gams` on the call clock, `pt_gams_launch`), `itrea mipfeas-report`; build_runs qualifies a run id reused in another set as `<run_id>#<set>` (#33). No existing arm's plan changed | the external benchmark's metric and clock (D-019) |
| 2026-10-11 | 0708835, 7998ab4 | D-021 phase 1 (ITX-1, [[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]]): arms A6F / A2F on F, the job-level `harness` (preset `ext`), named scopes, the fan-governor start gate, cap / thermal-watch / cuOpt-memory / external-kill / CUDA-compile voids, R26's breaker, job files `itx-m1` … `itx-m5`, `itx-r13`, `mipfeas-report --arm A6F`. A harness enters the run hash only when present: every earlier run id unchanged (20,432 plan records byte-identical); A9's EA configuration unchanged | Berk 10-11: "a better (more fair) and safer (not burning up dgx spark) set of rules" |
| 2026-10-12 | 6f0339d | D-021 phase 2: R12's stop at the proven optimum (`ref_stop`), cuOpt SIGKILLed after a definitive end (`cuopt_kill_on_end`), `scoped_hash`, preset `itx1`, `datasets/mipfeas_strata.csv`, `make-jobs itx-v1 / itx-a0 / itx-m6`, `itrea gams-trc`, `itrea itx-report`, `tools/m0_replay.py`, the prereg-v2 registration gate; 1,009 CPU tests pass on main. Every pre-existing plan byte-identical (35,337 records) | v1's external run (D-021) |
| 2026-10-12 | 3fe5612 | D-022 / D-023 tooling: `tools/d022_report.py` (`itrea d022-report`), `configs/itrea2-dev-rc-3way-boY.yaml` (5c8b535fb245: bo1's values, ladder disabled by unbounded patience), `make-jobs backoff2 --harness ext\|r13-off`, `make-jobs itx-r13f`, `backoff-report --stage 1y` | the two decision records, written before any run |
| 2026-10-13 | e1403fa, ae46e7f; 75caa6d … fd75476 | `jobs/itx-r13.jsonl` narrowed from 102 to 58 runs and `jobs/itx-r13b.jsonl` (16) added, both before any run of them was read; `d022_report` implements D-022's second amendment and its clarification (no mean; floor limb one-sided; C1 hint and guard); 57 tests | Berk 10-13: targeted tests first ([[D-024 Targeted first, wider only on a hint — staged testing for development and v1]]) |
| 2026-10-13 | 038c908 | `tools/d022_report.py`: when the runner stopped cuOpt after a partner's proof, cuOpt's gaps are read from the runner's record of its command line (`--mip-gap-zero`) — a stopped cuOpt writes no summary; test added (58 pass). Fixed before any D-022 verdict was recorded | a reading bug at the first interim report (17:45: validity failed on 8 "gap settings wrong"; [[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]] § Result) |
| 2026-10-13 | 0b41e31, 74e2297, 3fd80f5 → merge b84d192 (records bcc9378) | D-023 amended before any backoff2 run: `make-jobs backoff2 --instances amended\|stage1\|a,b` (default the 11), `jobs/backoff2.jsonl` (66 runs, harness ext); `backoff-report --stage 1y` reads the amended set and rules (stage 1 byte-identical); `make-jobs itx-a0 --select all\|d024\|stage1` → `jobs/itx-a0-81.jsonl` (81 runs); `tools/r26-gate.sh` (stopgap for #38: D-021 R26's opening gate). Non-GPU suite 1,041 passed, 1 skipped, 1 xfailed | D-024 rule 5 (test where phase Y can act) and D-024 (f) (A0 on the 81); #37 |
| 2026-10-14 | 61c552e, 91d38cd → merge bda6b83 | R1 + R2 of the C1 diagnosis: `host.cuda_sched` (the EA's CUDA context at blocking sync or yield; `src/itrea/ea/cuda_sched.py`), lp_decode inside A9 gated after the EA's LP worker; configs `itrea2-dev-rc-3way-bsync.yaml`, `-3way-ysync.yaml` (the yield fallback, named `byield` in 61c552e), `-3way-ld0.yaml` and variants; `make-jobs r1-bsync / r2-ld`; `tools/bsync_check.py`, `tools/d025_report.py`, `tools/d026_report.py`. Built and reviewed before any run, merged 23:41 after W-1014's restore | [[2026-10-14 C1 diagnosis and improvement candidates]] R1 / R2 |
| 2026-10-14 | 29d54d1, 32472b5, fe3424d, 7fbb840 → merge da80772 | R3: `make-jobs r3-diag` (22 runs after D-027's amendments: A6@r3, A6F@r3, A2@r3), `tools/d027_report.py` (`itrea d027-report`, rules 1–4). Non-GPU suite on main after both merges 1,256 passed, 1 xfailed | [[D-027 Same-day diagnostic of cuOpt's thread count and SCIP's presence (R3) — reading rules fixed before the runs]] |
| 2026-10-14 | 28f1eae | `jobs/r1-bsync.jsonl` (20 runs, `--control r13x` after D-023's verdict); `jobs/r3-diag.jsonl` (22 runs) arrived with merge da80772 (last changed in 7fbb840) and is not in 28f1eae, although 28f1eae's message and the journal 10-14 23:4x credit both files to it | window W-1015 |
| 2026-10-15 | 52dd402 | `tools/d029_report.py` (rules over seeds 0–5, `--write-jobs` with pinned run ids), `tools/d029_oc.py`, tests; `jobs/d029.jsonl` (21 lines: 20 read + 1 unread warm-up) | [[D-029 ysync gets one follow-up at seeds 2–5, judged on all seeds 0–5 with the failing seed kept — NOT ADOPTED ends R1 for v1]] |
| 2026-10-15 | 9e9e7de, c52d980, 17b21a3 → merge d3fe14f | `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` (ld0 + yield only; EA 835c7c5a249c, A9 run hash f345de0ab936), `make-jobs c1r-rerun --a9 ld0\|ysync-ld0` → `jobs/c1r-ld0.jsonl` / `jobs/c1r-ysync-ld0.jsonl` (68 runs each), `tools/c1r_report.py` (`itrea c1r-report`); non-GPU suite on the branch 1,339 passed | [[D-030 C1 is re-read once on the frozen-v2 candidate (ld0) with A6F in the same window — 68 runs under D-022's hint and guard and D-025's harm leg]] (Berk 10-14: "Re-run C1 on frozen hash") |
| 2026-10-15 | 967fd6a | `tools/d031_report.py` (the four-seed reading and `--write-jobs`), 31 tests; `jobs/d031.jsonl` (68 runs, set `c1r-ysync-ld0-s23`) | [[D-031 The phase-1 interim result rests on four seeds — C1 runs again at seeds 2–3 on the frozen-v2 candidate and is read with D-030's seeds 0–1]] |
| 2026-10-16 | 1f02eb6 | `jobs/d032.jsonl` (12 runs, set `d032`) | [[D-032 glass4's C1 harm flag is read in one window — old default vs yield + lp_decode vs lp_decode, 4 seeds]] |

## Results
| date | cfg | instance | seed | P_SCIP | P_EA | P_port | ttf (s) | posts/acc | notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-06 | 9ac4a93bf93b | glass4 | 0 | 0.332 | 1.000 | 0.257 | ∞ | 228/227 | SCIP-held 0.257; cuOpt 0.308; ref opt |
| 2026-10-06 | 9ac4a93bf93b | glass4 | 1 | 0.356 | 1.000 | 0.238 | ∞ | 279/279 | SCIP-held 0.238; cuOpt –; ref opt |
| 2026-10-06 | 9ac4a93bf93b | sing44 | 0 | 0.518 | 1.000 | 0.656 | ∞ | 27/6 | SCIP-held 0.663; cuOpt 0.726; ref opt |
| 2026-10-06 | 9ac4a93bf93b | sing44 | 1 | 0.532 | 1.000 | 0.646 | ∞ | 36/9 | SCIP-held 0.654; cuOpt –; ref opt |
| 2026-10-06 | 9ac4a93bf93b | radiationm18-12-05 | 0 | 0.154 | 1.000 | 0.167 | ∞ | 0/0 | SCIP-held 0.167; cuOpt 0.009; ref opt |
| 2026-10-06 | 9ac4a93bf93b | radiationm18-12-05 | 1 | 0.304 | 1.000 | 0.307 | ∞ | 0/0 | SCIP-held 0.307; cuOpt –; ref opt |
| 2026-10-06 | 9ac4a93bf93b | neos-960392 | 0 | 1.000 | 0.067 | 0.037 | 0.30 | 43/7 | SCIP-held 0.061; cuOpt 0.571; ref opt |
| 2026-10-06 | 9ac4a93bf93b | neos-960392 | 1 | 1.000 | 0.075 | 0.048 | 0.30 | 43/7 | SCIP-held 0.070; cuOpt –; ref opt |
| 2026-10-06 | 9ac4a93bf93b | neos-3024952-loue | 0 | 1.000 | 0.905 | 0.375 | 54.90 | 114/110 | SCIP-held 0.377; cuOpt 0.760; ref opt |
| 2026-10-06 | 9ac4a93bf93b | neos-3024952-loue | 1 | 1.000 | 0.882 | 0.312 | 47.27 | 145/141 | SCIP-held 0.313; cuOpt –; ref opt |
| 2026-10-06 | 9ac4a93bf93b | fhnw-binpack4-48 | 0 | 1.000 | 1.000 | 1.000 | ∞ | 0/0 | SCIP-held 1.000; cuOpt 0.005; ref opt |
| 2026-10-06 | 9ac4a93bf93b | fhnw-binpack4-48 | 1 | 1.000 | 1.000 | 1.000 | ∞ | 0/0 | SCIP-held 1.000; cuOpt –; ref opt |
| 2026-10-06 | 9ac4a93bf93b | rail01 | 0 | 1.000 | 1.000 | 1.000 | ∞ | 0/0 | SCIP-held 1.000; cuOpt 0.927; ref opt |
| 2026-10-06 | 9ac4a93bf93b | rail01 | 1 | 1.000 | 1.000 | 1.000 | ∞ | 0/0 | SCIP-held 1.000; cuOpt –; ref opt |
| 2026-10-06 | 9ac4a93bf93b | ca__Scenario400_reserves_1 | 0 | 1.000 | 1.000 | 1.000 | ∞ | 0/0 | SCIP-held 1.000; cuOpt 0.643; ref best_of_study |
| 2026-10-06 | 9ac4a93bf93b | ca__Scenario400_reserves_1 | 1 | 1.000 | 1.000 | 1.000 | ∞ | 0/0 | SCIP-held 1.000; cuOpt –; ref best_of_study |
| 2026-10-06 | 9ac4a93bf93b | item_placement_9903 | 0 | 0.631 | 0.495 | 0.522 | 0.05 | 1845/440 | SCIP-held 0.523; cuOpt 0.666; ref screening_best_provisional |
| 2026-10-06 | 9ac4a93bf93b | item_placement_9903 | 1 | 0.689 | 0.396 | 0.277 | 0.05 | 1375/317 | SCIP-held 0.278; cuOpt –; ref screening_best_provisional |

## Baselines 2026-10-06 — per instance (seeds averaged)

| instance | SCIP alone | EA alone | uncoupled (best of both) | coupled | cuOpt alone | d vs SCIP | d vs EA | d vs uncoupled | d vs cuOpt |
|---|---|---|---|---|---|---|---|---|---|
| glass4 | 0.344 | 1.000 | 0.344 | 0.247 | 0.308 | 0.28 | 0.75 | 0.28 | 0.20 |
| sing44 | 0.525 | 1.000 | 0.525 | 0.651 | 0.726 | -0.19 | 0.35 | -0.19 | 0.10 |
| radiationm18-12-05 | 0.229 | 1.000 | 0.229 | 0.237 | 0.009 | -0.03 | 0.76 | -0.03 | -0.96 |
| neos-960392 | 1.000 | 0.071 | 0.071 | 0.042 | 0.571 | 0.96 | 0.40 | 0.40 | 0.93 |
| neos-3024952-loue | 1.000 | 0.893 | 0.893 | 0.344 | 0.760 | 0.66 | 0.62 | 0.62 | 0.55 |
| fhnw-binpack4-48 | 1.000 | 1.000 | 1.000 | 1.000 | 0.005 | 0.00 | 0.00 | 0.00 | -1.00 |
| rail01 | 1.000 | 1.000 | 1.000 | 1.000 | 0.927 | 0.00 | 0.00 | 0.00 | -0.07 |
| ca__Scenario400_reserves_1 | 1.000 | 1.000 | 1.000 | 1.000 | 0.643 | 0.00 | 0.00 | 0.00 | -0.36 |
| item_placement_9903 | 0.660 | 0.445 | 0.421 | 0.400 | 0.666 | 0.39 | 0.10 | 0.05 | 0.40 |

Mean bounded d (positive = coupled better): vs SCIP +0.23 (4 better, 2 worse, 3 equal), vs EA +0.33 (6 better, 0 worse), vs the uncoupled portfolio +0.12 (4 better, 2 worse), vs cuOpt alone -0.02 (5 better, 4 worse). Descriptive: 9 instances, 2 seeds, 120 s, one-seed screening references for ML4CO.

**Reading.**
- The coupling works where the EA contributes: **glass4** (the EA polishes SCIP's solutions — 213 of 228 posts became SCIP incumbents), **neos-960392** (SCIP alone holds only the trivial 0; coupled reaches the proven optimum −238 on seed 0), **neos-3024952-loue** (SCIP alone has no solution; the EA's first at ~5 s, 110–141 accepted posts), **item_placement_9903**. On these, coupled beats both SCIP alone and the uncoupled portfolio.
- **The EA alone finds no feasible solution on 6 of 9** (glass4, sing44, radiationm18-12-05, fhnw-binpack4-48, rail01, ca__Scenario400_reserves_1) in 120 s, while **cuOpt alone finds one on all 9** (fhnw P 0.005, radiationm 0.009). Our GPU feasibility jump is far behind cuOpt's heuristics on feasibility — the first target.
- **Interference:** on sing44 coupled SCIP found the *same* first solution at 7.25 s instead of 3.65 s and got through 1 node instead of 3 — the GPU EA (92 % utilisation, never feasible there) competes with SCIP's root LP for memory bandwidth. Coupled loses to SCIP alone there (0.65 vs 0.53). The EA should back off when it is not contributing.
- Run hygiene: 63/63 completed, every incumbent checked (0 unchecked), **compiled_on_clock False on all runs** (#4 holds on the dev set), GPU health ok, no thermal pause or rest.

**Next (D-010 order):** (b) the EA's feasibility search — probe 1 (EA alone, 6 instances): longer FJ budget (c01), + rhs row scale + an LP point after t0 (c02); then (c) an EA back-off when it is not contributing (interference); then operators and the exchange.

## Feasibility probe 1 (2026-10-06 11:54–13:02) — EA alone, seed 0, 120 s
Variants (overrides on `9ac4a93bf93b`): **c01** longer feasibility jump (max_steps 20,000, stall 1,000); **c02** c01 +
`ranks.viol_norm: rhs` + an LP point after t0 (`seeding.lp_source: highs_async`); **c03** c01 + LP point only;
**c04** c01 + rhs only. P(120) (1.0 = never feasible):

| instance | default | c01 | c02 | c03 | c04 | SCIP alone | cuOpt alone |
|---|---|---|---|---|---|---|---|
| ca__Scenario400_reserves_1 | 1.000 | 1.000 | 0.255 | **0.229** | 1.000 | 1.000 | 0.643 |
| sing44 | 1.000 | 0.975 | 1.000 | **0.415** | 1.000 | 0.518 | 0.726 |
| glass4 | 1.000 | 1.000 | 0.300 | 1.000 | **0.277** (opt.) | 0.332 | 0.308 |
| fhnw-binpack4-48 | 1.000 | 1.000 | **0.196** (opt.) | 1.000 | 0.253 (opt.) | 1.000 | 0.005 |
| radiationm18-12-05 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.154 | 0.009 |
| rail01 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.927 |

Coupled with c04 (A3@c04, seed 0) vs coupled default: glass4 0.218 vs 0.257; sing44 0.566 vs 0.656 (SCIP alone 0.518);
neos-3024952-loue 0.417 vs 0.375; fhnw-binpack4-48 0.635 (optimum found at 76 s, vs 30 s for the EA alone — in coupled
mode the EA also gets SCIP's root LP, which c03 shows hurts fhnw; plus interference).

**Reading.** No single repair setting wins: the **LP point** unlocks the UC case and sing44 (the EA alone then beats SCIP
alone and cuOpt alone on both); **rhs row scaling** unlocks glass4 and fhnw-binpack4-48 (proven optima in seconds);
each hurts where the other helps. radiationm18-12-05 and rail01 stay out of reach (cuOpt: 0.009 / 0.927).
**Next idea (to test):** self-adaptive repair strategy — every individual carries its row scale (l2 / rhs) and LP
guidance (on / off) as inherited, mutable genes; the local search runs per strategy group; the strategy that produces
survivors spreads, so the population acts as a portfolio on every instance. Then the EA back-off (interference) and the
FJ moves for general integers (radiationm).

## Coupled adaptive strategy (2026-10-06 15:07–15:55) — A3@ad1, 9 instances × seeds 0–1, 120 s
`ad1` = `configs/itrea2-adaptive.yaml` + c01 repair budget (`repair.max_steps 20000, repair.stall_steps 1000`), effective
hash `7ee4d999f11d`, commit 0e69732. Judged on the coupled arm (D-011 §1; the A5@ad1 runs were stopped after 2 blocks).
P(120), seeds averaged (P_port; lower is better):

| instance | SCIP alone (A0) | coupled default (A3) | **coupled adaptive (A3@ad1)** | cuOpt alone (A6) |
|---|---|---|---|---|
| ca__Scenario400_reserves_1 | 1.000 | 1.000 | **0.296** | 0.645 |
| sing44 | 0.525 | 0.651 | **0.361** | 0.726 |
| glass4 | 0.344 | 0.247 | 0.241 | 0.308 |
| neos-960392 | 1.000 | 0.042 | 0.032 | 0.571 |
| item_placement_9903 | 0.660 | 0.400 | 0.522 | 0.666 |
| neos-3024952-loue | 1.000 | 0.344 | 0.441 | 0.760 |
| radiationm18-12-05 | 0.229 | 0.237 | 0.235 | 0.009 |
| fhnw-binpack4-48 | 1.000 | 1.000 | 1.000 | 0.005 |
| rail01 | 1.000 | 1.000 | 1.000 | 0.927 |

Provenance caveat: the 18 runs span code states 0e69732 (9 clean, 7 with the uncommitted LP-stream code — off in
these runs) and 770df17 (4); a disabled stream is bit-identical (review of 770df17), so the numbers stand
([[Mistakes and incidents]], 2026-10-06 15:15–16:35).
Mean bounded d of A3@ad1 (positive = A3@ad1 better): vs SCIP alone **+0.34** (6 better, 1 worse — radiationm −0.02 —,
2 equal); vs coupled default +0.11 (4 better, 2 worse: item_placement −0.23, loue −0.22); vs cuOpt alone +0.09 (6
better, 3 worse: fhnw, radiationm, rail01). Hygiene: 18/18 completed, every incumbent checked (0 failed, 0 unchecked),
GPU health ok on all.

**Reading — three changes at once, so not yet attributable.** `ad1` differs from the coupled default in (1) the
self-adaptive strategy, (2) `seeding.lp_source: highs_async` (the EA solves its own LP with HiGHS after t0 instead of
waiting for SCIP's root LP) and (3) the longer repair budget.
- **ca and sing44 — the LP's timing.** In the coupled default the EA's LP point is SCIP's root LP: on sing44 it arrived
  at 73–76 s, on ca never (SCIP does not finish the root LP in 120 s). With HiGHS in the EA it arrives at 10.4 s (sing44)
  and 29 s (ca). Then the coupled system beats SCIP alone and cuOpt alone on both (ca 0.296 vs 1.0 / 0.645; sing44 0.361
  vs 0.525 / 0.726) and the sing44 interference loss is gone. The surviving strategies are mostly LP-guided on sing44
  seed 1 (`rhs/lp` 189 of 256) but `rhs/nolp` on seed 0 (105) — selection does not settle on one strategy.
- **loue and item_placement — slower generations.** The EA posts 3–4× fewer improvements (loue 49/38 vs 114/145;
  item 443/462 vs 1845/1375), consistent with the 20,000-step repair making each generation longer; SCIP's node rate on
  item_placement also falls (148/120 vs 224/203 nodes/s).
- radiationm18-12-05, fhnw-binpack4-48, rail01: no change. fhnw's optimum, found by A3@c04 (rhs only, seed 0) at 76 s,
  is not found here: the rhs strategies die out there (final population 9 and 8 of 256).
**Next:** separate the three: `lp1` = coupled default + `seeding.lp_source: highs_async` only (one variable), after the
LP-stream runs (`lps1` vs `ad1` isolates the stream).

## Aggregation probe (2026-10-06 16:1x) — CPU only, not wired into the EA
`src/itrea/ea/aggregate.py` substitutes two-variable equality rows (equal magnitudes; integrality kept) exactly:
x_v = ±y_root + o. On the dev set: radiationm18-12-05 40,623 → 14,919 variables (rows 40,935 → 15,231); rail01
117,527 → 96,736 (rows 46,843 → 26,052); sing44 59,708 → 57,448; ca__Scenario400 305,712 → 305,328 (32,160 two-variable
equalities with unequal coefficients refused); glass4, fhnw, loue: nothing. SCIP's and cuOpt's solutions of radiationm,
rail01, sing44 and ca project to feasible reduced points and lift back exactly (same objective).
Payoff test (`tools/agg_probe.py`, the CPU feasibility jump with the c01 budget from the same LP roundings,
radiationm18-12-05, 32 children): original 0/32 feasible, median V 127; aggregated 0/32, median V 88; aggregated with
10× the budget 0/16, median V 42. The rows still violated on the aggregated model are 154 equality rows over ≥ 9
general integers (exact sums), 76 rows over continuous variables and a few 3-variable integer equalities — not the
chains. **Verdict: not wired in.** The chains are not radiationm's blocker; exact integer sums are (an LP-guided
rounding with propagation is the candidate — what cuOpt's 0.8 s first solution suggests). The module stays, tested
(`tests/test_aggregate.py`), for rail01-type instances if a later probe shows the chains matter there.

## LP stream (2026-10-06 16:03–16:35) — A3@lps1 vs A3@ad1, 9 instances × seeds 0–1, 120 s
`lps1` = `ad1` + `lp_stream.enabled` (effective hash `4a96b872ffdc`, commit c192bb5) — one variable against `ad1`.
P(120), seeds averaged:

| instance | SCIP alone | coupled default | coupled adaptive (ad1) | **+ LP stream (lps1)** | cuOpt alone |
|---|---|---|---|---|---|
| ca__Scenario400_reserves_1 | 1.000 | 1.000 | 0.296 | 0.297 | 0.645 |
| fhnw-binpack4-48 | 1.000 | 1.000 | 1.000 | **0.651** (s0 0.302, s1 1.000) | 0.005 |
| glass4 | 0.344 | 0.247 | 0.241 | 0.228 | 0.308 |
| item_placement_9903 | 0.660 | 0.400 | 0.522 | 0.569 | 0.666 |
| neos-3024952-loue | 1.000 | 0.344 | 0.441 | 0.369 (s0 0.478 → 0.332) | 0.760 |
| neos-960392 | 1.000 | 0.042 | 0.032 | 0.032 | 0.571 |
| radiationm18-12-05 | 0.229 | 0.237 | 0.235 | 0.234 | 0.009 |
| rail01 | 1.000 | 1.000 | 1.000 | 1.000 | 0.927 |
| sing44 | 0.525 | 0.651 | 0.361 | 0.362 | 0.726 |

Mean bounded d of lps1: vs ad1 **+0.05** (3 better: fhnw, loue, glass4; 1 worse: item_placement −0.08); vs SCIP alone
**+0.38** (7 better, 0 worse, radiationm −0.02 ≈ equal, rail01 equal); vs cuOpt alone +0.10 (6 better, 3 worse: fhnw,
radiationm, rail01). Hygiene: 18/18 completed, every incumbent checked, GPU health ok.

**Stream counters.** SCIP spent ≤ 0.7 s of 120 s on the stream except radiationm (2.43 s — the bucket's ceiling, 174
reads). No node LP exists on ca, rail01 (root LP never finishes), and only 1–2 / 10–11 on neos-960392 / sing44 (SCIP
stays at the root), so the stream cannot act there. The EA took 16–384 points per run; on glass4/fhnw/loue it **missed
95–358 ids** between two polls (SCIP posts every 0.25 s, the EA reads once per exchange tick — the reads were paid for
nothing) and dropped 17–29 unused ones. Stream-born members rarely sit in Rank 0 at the end (0–11).
**Attribution — weak.** fhnw seed 0's optimum (objective 0, at 36.2 s; SCIP accepted it at 36.4 s) came from an EA
child (uniform crossover + toward_lp mutation + repair, strategy l2/lp) whose ancestry over three generations shows LP
immigrants but no stream-born member; seed 1 found nothing. The +0.05 is a two-seed, three-instance signal — the stream
is kept on (cheap, never harmful by more than item_placement's −0.08) but not yet credited.
**Next for the stream:** match SCIP's read rate to the EA's poll rate (missed ids are wasted SCIP time), then judge it
on more seeds once the feasibility work (below) is in.

## Finding (16:3x): the c01 repair budget starves the evolution
`ad1` and `lps1` carried the c01 budget (`repair.max_steps 20000, repair.stall_steps 1000`) for comparability with
feasibility probe 1. It makes every generation 6–9× slower, because the GPU repair waits for the slowest individual:
coupled glass4 296 generations in 120 s (default) → 34 (ad1) / 27 (lps1); neos-3024952-loue 303 → 48; item_placement
2,060 → 465 (repair moves per child 99–166 → 460–890). The EA alone on glass4: 956 → 110. This explains the ad1 losses
on loue and item_placement (3–4× fewer posts) and the stream's missed ids (the EA reads the mailbox between
generations, every 2.5–4.5 s). In probe 1, c01 alone never unlocked an instance — the LP point and the rhs scale did.
So today's ad1/lps1 gains came with an EA that hardly evolved. **Next run `lps0`** = `itrea2-adaptive-lps` at the
DEFAULT repair budget (hash `a2ad94e9188d`), after `lp1`.
Improvement probe (`tools/agg_improve_probe.py`, 16:5x): from SCIP's incumbent #1 of radiationm18-12-05 (objective
83,244 at 0.31 s; the optimum is near 17,500), 32 copies with 5 integer genes mutated, the CPU feasibility jump with the
objective cutoff: **0/32 better-and-feasible** on the original and on the aggregated model, at both budgets (children
end with V ≈ 0.003 — stuck just short). The objective is ONE integer variable (a min-max structure): lowering it by 1
tightens every row it bounds at once, which no single or chain move does. radiationm needs a non-local operator
(fix-and-propagate around the incumbent, or a residual sub-MIP), not better local moves; parked.

## Repair budget, one variable (2026-10-06 16:55–17:30) — A3@lps0 vs A3@lps1, 9 instances × seeds 0–1, 120 s
`lps0` = `itrea2-adaptive-lps` at the DEFAULT repair budget (hash `a2ad94e9188d`, code f8adb63); `lps1` = the same +
the c01 budget (`repair.max_steps 20000, repair.stall_steps 1000`). P(120), seeds averaged:

| instance | SCIP alone | coupled default | lps1 (deep repair) | **lps0 (default repair)** | generations lps1 → lps0 | cuOpt alone |
|---|---|---|---|---|---|---|
| ca__Scenario400_reserves_1 | 1.000 | 1.000 | 0.297 | 0.395 | — → 67–79 | 0.645 |
| fhnw-binpack4-48 | 1.000 | 1.000 | 0.651 | 1.000 | 37 → 183–308 | 0.005 |
| glass4 | 0.344 | 0.247 | 0.228 | 0.323 | 27 → 220–224 | 0.308 |
| item_placement_9903 | 0.660 | 0.400 | 0.569 | **0.358** | 371 → 1770–1840 | 0.666 |
| neos-3024952-loue | 1.000 | 0.344 | 0.369 | 0.440 | 48 → 429–436 | 0.760 |
| neos-960392 | 1.000 | 0.042 | 0.032 | 0.039 | — → 99–365 | 0.571 |
| radiationm18-12-05 | 0.229 | 0.237 | 0.234 | 0.235 | — → 508–512 | 0.009 |
| rail01 | 1.000 | 1.000 | 1.000 | 1.000 | — → 854–855 | 0.927 |
| sing44 | 0.525 | 0.651 | 0.362 | 0.551 | — → 448–451 | 0.726 |

lps0 vs lps1: mean bounded d **−0.13** (1 better: item_placement +0.37; 6 worse: fhnw −0.35, sing44 −0.34, glass4
−0.29, ca −0.25, neos-960392 −0.18, loue −0.16; 2 equal). lps0 vs SCIP alone +0.29 (5 better, 2 worse: sing44 −0.05,
radiationm −0.02). Hygiene: 18/18 completed, every incumbent checked, GPU health ok. Stream: with fast generations the
EA misses almost nothing (0–17 ids, was 95–358).

**Correction to the 16:3x finding.** I read "6–9× fewer generations" as "the EA hardly evolved, so the gains came despite
the budget" and expected the default budget to do as well or better. Measured, the deep repair wins on 6 of 9: on
these instances the repair, not the number of generations, does most of the work (sing44 and ca need it to turn LP
roundings feasible; glass4's 209 posts at the default budget improved SCIP less than lps1's 22). Only item_placement —
many small improvements, 1,700+ posts — gains from fast generations. **The repair depth is an instance-dependent
trade-off**, the same kind the self-adaptive strategy was built for; but the GPU repair call waits for its slowest
individual, so mixing depths inside one call would not make a generation faster. Candidates: a time budget per repair
call instead of a step budget; alternating deep and shallow generations; or depth as a strategy gene with the
generation time charged to the strategy. **Development baseline for now: lps1** (deep repair + adaptive + HiGHS LP +
stream), the best measured configuration.

## fhnw two-scale survival (2026-10-06 17:5x–18:08) — alt1 = lps0 + `ranks.feas_alt_scale: rhs`, fhnw-binpack4-48, seeds 0–1
| arm | seed 0 | seed 1 | rhs-repaired members at the end (of 256) |
|---|---|---|---|
| A3@alt1 (coupled) | 1.000 | 1.000 | 136 / 223 |
| A5@alt1 (EA alone, diagnostic) | 1.000 | 1.000 | 245 / 139 |
| A3@lps0 (same, one scale) | 1.000 | 1.000 | 0 / 0 |
| A3@lps1 (deep repair, one scale) | **0.302** | 1.000 | 36 / 13 |
| A5@c04 (deep repair, rhs on repair AND survival) | **0.253** | — | — |
**Reading.** The two-scale survival does what it was built for — the rhs-repaired individuals survive (136–245 of 256,
against 0 under one scale) — but at the default repair depth no feasible point appears in 120 s. Both fhnw finds so
far had deep repair (c04 at 30 s, lps1 at 36 s). Next: `lps1 + feas_alt_scale rhs` (deep repair + two scales).
`lp1` (LP source alone) got 1 of 18 runs before the window closed (neos-3024952-loue s1 0.355); the rest waits.

## Repair-depth design (2026-10-06 17:36–18:40)
A design workflow (code map, run-data analysis, literature, three designs, two adversarial judges) chose a **repair
clock**: a lane-step horizon per repair call enforced between GPU launches, chosen per generation from the measured
gain per second of each horizon, with deep repair kept until the EA's own first feasible point, a materiality floor,
and the LP batch always repaired deep. Staged: instrumentation + shadow mode first (no behaviour change; pass criteria
stated in advance), the active controller only if the shadow signal separates item_placement from glass4/loue.
Full record: [[2026-10-06 Repair-depth design (workflow)]].

## Repair clock, Stage 1 — shadow (2026-10-07 09:01–09:26) — A3@rc0 vs in-window A3@lps1w, 7 blocks
Both LLM servers stopped (Berk 09:0x), GPU otherwise idle. Stage 0b first: full suite 656 passed / 2 skipped
(gpu-marked horizon tests included); on CuPy, shadow and shadow-forced runs are bit-identical to the disabled run in
6 of 6 toy scenarios (26 arrays each). 14/14 runs completed. `logs/rc_stage1.py` (to be moved to `tools/`):

| run | generations rc0 / control | first own feasible | `would` in the post phase | criterion |
|---|---|---|---|---|
| item_placement s0 | 411 / 376 | 0.1 s | 2^7 59 %, 2^8 14 %, 2^12 10 % | (a) PASS — left none at 0.4 s for 2^7, rate ratio 3.05 |
| item_placement s1 | 394 / 395 | 0.1 s | 2^7 36 %, 2^9 17 %, 2^8 16 % | (a) PASS — left none at 0.4 s for 2^7, rate ratio 2.14 |
| glass4 s0 | 17 / 25 | 5.5 s | none 35 %, 2^12 29 %, 2^13 24 %, 2^11 12 % | (b) FAIL 35 %; (c) FAIL −32 % |
| neos-3024952-loue s0 | 47 / 52 | 2.1 s | 2^11 46 %, 2^7 23 %, none 21 % | (b) FAIL 21 % |
| neos-960392 s0 | 9 / 9 | 0.9 s | none 56 %, 2^7 22 %, 2^8 22 % | (b) FAIL 56 % |
| sing44 s0 | 70 / 69 | 12.7 s | none 91 % | (c) PASS |
| ca s0 | 18 / 19 | 34.8 s | none (1 post generation) | (c) PASS |

**Verdict by D-012's letter: FAIL** ((a) pass, (b) fail, (c) fail on glass4). Two findings qualify it:
- **(c) is not overhead.** Seconds per generation outside the repair calls are equal within a few % (glass4 1,008 vs
  1,073 ms; loue 122 vs 125; item 93 vs 96; sing44 117 vs 103); glass4's 17 vs 25 generations come from longer repair
  calls on its trajectory (4.8 vs 3.7 s per generation — coupled runs diverge after the first exchange). The proxy
  "generation count within 10 %" is too noisy for an instance with ~20 generations.
- **(b) tested a premise, not only the controller.** It assumed "glass4, loue, neos-960392 want deep repair" from
  lps0 < lps1 — but the default budget is max_steps 2,000 AND stall_steps **100**; the clock's shallow levels keep
  stall 1,000 and only cap the total at 2^11–2^13. glass4's two material gains came from lanes of 1,073 and 287 steps.
**Next — test the premise, one variable each** (`logs/dev-premise.jsonl`, queued): `cap4k` (lps1 with max_steps
4,096, stall 1,000 — what the clock chose) and `stall100` (lps1 with stall 100 — what made lps0 shallow) on glass4,
loue, neos-960392, item_placement × seeds 0–1, against lps1w. If cap4k ≈ lps1 there, the controller's shallow choice
is harmless and Stage 2 (active) is the real test; if cap4k loses, the counterfactual is myopic and D-012's fallback
(the per-generation bandit) follows.

## fhnw-binpack4-48: deep repair + two-scale survival (2026-10-07 09:27–09:51) — A3@alt2 vs in-window A3@lps1w, seeds 0–3
| arm | s0 | s1 | s2 | s3 | solved |
|---|---|---|---|---|---|
| alt2 = lps1 + `ranks.feas_alt_scale: rhs` | 1.000 | **0.214** (first feasible 25.7 s) | 1.000 | 1.000 | 1/4 |
| lps1w (deep repair, one scale) | 1.000 | 1.000 | 1.000 | 1.000 | 0/4 |
All fhnw runs so far — solved / runs: adaptive + deep repair (lps1, lps1w, alt2) **2 / 10**; adaptive + default depth (lps0,
alt1) 0 / 4; **pure rhs scale on repair and survival, no adaptive strategy (c04: A5 at 30 s, A3 at 76 s) 2 / 2**.
Reading: two-scale survival keeps the rhs-repaired members alive (236–244 of 256 at the end of the unsolved alt2 runs)
but does not change the solve rate measurably (1/4 vs 0/4; n is tiny). The configuration that solved fhnw every time
is the pure rhs scale — in the adaptive strategy rhs is one of four strategies, and selection does not concentrate on
it within 120 s. Hygiene: 8/8 completed, every incumbent checked, GPU health ok.

## The EA's own LP alone (2026-10-07 09:51–10:23) — A3@lp1 = coupled default + `seeding.lp_source: highs_async`, 9 × seeds 0–1
One variable against the coupled default (18/18 runs; 1 from 2026-10-06), every incumbent checked, GPU health ok.

| instance | SCIP alone | coupled default | **+ own LP (lp1)** | lps0 (+ strategy + stream, default depth) | lps1 (+ strategy + stream + deep) | cuOpt alone |
|---|---|---|---|---|---|---|
| ca__Scenario400_reserves_1 | 1.000 | 1.000 | 0.509 | 0.395 | **0.300** | 0.647 |
| fhnw-binpack4-48 | 1.000 | 1.000 | 1.000 | 1.000 | 0.651 | **0.005** |
| glass4 | 0.344 | 0.247 | 0.300 | 0.323 | **0.228** | 0.308 |
| item_placement_9903 | 0.660 | 0.400 | **0.218** | 0.358 | 0.569 | 0.666 |
| neos-3024952-loue | 1.000 | **0.344** | 0.369 | 0.440 | 0.369 | 0.760 |
| neos-960392 | 1.000 | 0.042 | **0.027** | 0.039 | 0.032 | 0.571 |
| radiationm18-12-05 | 0.229 | 0.237 | 0.224 | 0.235 | 0.234 | **0.009** |
| rail01 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.927** |
| sing44 | 0.525 | 0.651 | 0.492 | 0.551 | **0.362** | 0.726 |

| comparison | better / worse | mean d [95 % CI] | Wilcoxon p | D-002 rule |
|---|---|---|---|---|
| lp1 vs coupled default | 5 / 2 | +0.19 [+0.01, +0.37] | 0.156 | inconclusive |
| lp1 vs SCIP alone | **7 / 0** | **+0.43 [+0.19, +0.68]** | **0.016** | supported |
| lps1 vs lp1 | 4 / 4 | +0.06 [−0.17, +0.25] | 0.64 | inconclusive |
| **lps0 vs lp1** (strategy + stream at default depth) | 1 / 6 | **−0.12 [−0.25, +0.01]** | 0.156 | **negative** |
| lps1 vs lps0 (deep vs default depth) | 7 / 1 | +0.15 [−0.02, +0.28] | 0.195 | inconclusive |
| lp1 vs cuOpt alone | 6 / 3 | +0.05 [−0.38, +0.44] | 0.65 | inconclusive |

**Reading.** The EA solving its own LP right after t0 is the single change that carries the gain over SCIP alone (lp1
matches lps1's +0.43 against SCIP), and it gives the best item_placement result so far (0.218). At the default depth
the adaptive strategy and the LP stream add nothing measurable — slightly negative (1 / 6). Deep repair trades
instances (ca, sing44, glass4, fhnw gain; item_placement and neos-960392 lose). **Owed next:** `lpd` = lp1 + deep repair,
no strategy, no stream (queued) — it isolates strategy + stream at the deep depth (lps1 vs lpd).

## Premise test for D-012's criterion (b) (2026-10-07 10:23–11:17) — one variable each vs deep repair (lps1w)
`cap4k` = lps1 with repair.max_steps 4,096 (stall 1,000 kept — the regime of the clock's shallow levels);
`stall100` = lps1 with repair.stall_steps 100 (max_steps 20,000 kept — the stall rule of the default budget).
24/24 runs (incl. controls), every incumbent checked, GPU health ok. P(120):

| instance | seed | lps1w (deep) | cap4k | stall100 | generations lps1w / cap4k / stall100 |
|---|---|---|---|---|---|
| glass4 | 0 / 1 | 0.226 / 0.229 | **0.217 / 0.222** | 0.241 / 0.315 | 26 / 28 / 158 |
| item_placement_9903 | 0 / 1 | 0.573 / 0.408 | 0.517 / 0.543 | **0.371 / 0.317** | 389 / 395 / 1,792 |
| neos-3024952-loue | 0 / 1 | 0.433 / 0.371 | **0.336** / 0.419 | 0.463 / 0.457 | 49 / 44 / 444 |
| neos-960392 | 0 / 1 | 0.032 / 0.031 | **0.025 / 0.027** | 0.067 / 0.043 | 10 / 20 / 174 |
cap4k vs lps1w: 3 better / 1 worse, mean d **+0.05 [−0.04, +0.15]**; stall100 vs lps1w: 1 / 3, mean d **−0.11
[−0.36, +0.18]** (glass4 −0.18, loue −0.13, neos-960392 −0.43; item_placement +0.30).

**Reading.** The default budget lost on glass4, loue and neos-960392 because of its **stall rule** (100 steps), not its
step cap: capping the total at 4,096 lane steps with the 1,000-step stall costs nothing (slightly better). So
criterion (b)'s premise ("these instances want the horizon at none") is refuted for the clock's regime, and Stage 1's
(b) failure says nothing against the controller there. item_placement's gain needs many generations (stall100: 4–5×
more), which only a short repair gives — the clock chose 2^7 there (Stage 1, criterion (a) passed). **Decision:**
proceed to Stage 2 (active) under D-012's Stage-2 rejection rules, instead of the Stage-1 fallback — recorded as a
deviation in D-012. Caveat: the clock also chose 2^7 on loue 23 % of the time; whether that short regime hurts there
(as stall100 did) is exactly what Stage 2 measures.

## fhnw-binpack4-48 with the pure rhs scale (2026-10-07 11:17–11:20) — A3@c04 (default config + deep repair + `ranks.viol_norm: rhs`), seeds 0–3
P(120): s0 0.635 (first feasible 76 s; run of 2026-10-06), **s1 0.229 (27 s), s2 0.460 (55 s)**, s3 1.000 → **solved
3 of 4**. All fhnw arms so far, solved / runs: pure rhs (A3@c04, A5@c04) **4 / 5**; adaptive strategy + deep repair
(lps1, lps1w, alt2) 2 / 10; adaptive at the default depth 0 / 4; SCIP alone 0 / 2; cuOpt alone 1 / 1 (P 0.005).
Reading: on fhnw the single rhs scale works where the adaptive mix of four strategies rarely does — selection does
not concentrate on rhs within 120 s. This argues for per-scale demes (#16) over the strategy gene for the row scale.

## Own LP + deep repair, nothing else (2026-10-07 11:20–12:23) — A3@lpd, 9 × seeds 0–1
`lpd` = coupled default + `seeding.lp_source: highs_async` + deep repair (max_steps 20,000, stall 1,000); no adaptive
strategy, no LP stream (effective hash `05998e37504d`). 18/18 completed, every incumbent checked, GPU health ok.

| instance | SCIP | cuOpt | lp1 (own LP) | **lpd (own LP + deep)** | lps1 (+ strategy + stream + deep) |
|---|---|---|---|---|---|
| ca__Scenario400_reserves_1 | 1.000 | 0.647 | 0.509 | 0.305 | **0.300** |
| fhnw-binpack4-48 | 1.000 | **0.005** | 1.000 | 1.000 | 0.651 |
| glass4 | 0.344 | 0.308 | 0.300 | **0.193** | 0.228 |
| item_placement_9903 | 0.660 | 0.666 | **0.218** | 0.613 | 0.569 |
| neos-3024952-loue | 1.000 | 0.760 | **0.369** | 0.451 | **0.369** |
| neos-960392 | 1.000 | 0.571 | **0.027** | 0.038 | 0.032 |
| radiationm18-12-05 | 0.229 | **0.009** | 0.224 | 0.231 | 0.234 |
| rail01 | 1.000 | **0.927** | 1.000 | 1.000 | 1.000 |
| sing44 | 0.525 | 0.726 | 0.492 | **0.344** | 0.362 |

| comparison | better / worse | mean d [95 % CI] | p | rule |
|---|---|---|---|---|
| lpd vs SCIP alone | **6 / 1** | **+0.44 [+0.21, +0.67]** | 0.031 | supported |
| lpd vs cuOpt alone | 6 / 3 | +0.11 [−0.33, +0.50] | 0.65 | inconclusive |
| lps1 vs lpd (strategy + stream at deep depth) | 5 / 3 | +0.07 [−0.03, +0.17] | 0.25 | inconclusive |
| lpd vs lp1 (deep vs default depth, neither) | 3 / 4 | −0.01 [−0.28, +0.24] | 0.94 | inconclusive |
D-013 weak sets: SCIP's (7) mean P 0.88 → 0.54 (−39 %, 5 of 7 improved); cuOpt's (6) 0.72 → 0.46 (−36 %, 5 of 6).

**Reading.** The own LP carries the gain over SCIP in every configuration (+0.43 to +0.44). The adaptive strategy and
the LP stream add little: +0.07 at deep depth (fhnw +0.35 — one solved seed —, loue +0.18, neos-960392 +0.16; glass4
−0.15), −0.12 at the default depth. Repair depth is a clean instance trade (deep: ca +0.40, glass4 +0.36, sing44 +0.30;
default: item_placement +0.64, neos-960392 +0.27, loue +0.18) — the target of the repair clock (Stage 2 running).

## Class probe — D-014 result (2026-10-07 13:21–15:13) — 8 new public instances, A0 / A6 / A3@lp1 × seeds 0–1
48/48 runs completed, every incumbent checked, GPU health ok; predictions committed at 12:04 (01dacce), before the runs.
P(120), seed means:

| instance | SCIP alone | cuOpt alone | coupled (lp1) | prediction | outcome |
|---|---|---|---|---|---|
| neos-3004026-krka | 0.659 | **0.004** | 0.662 | lose to cuOpt | HIT (d −0.99) |
| neos-3656078-kumeu | 1.000 | **0.102** | 1.000 | lose to cuOpt | HIT (d −0.90) |
| chromaticindex1024-7 | 1.000 | **0.075** | 1.000 | lose to cuOpt; no gain vs SCIP (LP 124.8 s) | HIT (d −0.93); (b) holds (0.0 / 0.0) |
| comp21-2idx | 0.891 | **0.498** | 0.656 | beat cuOpt | MISS (d −0.24; vs SCIP +0.26) |
| neos-4763324-toguru | 0.772 | **0.583** | 0.772 | beat cuOpt | MISS (d −0.24) |
| neos-2657525-crna | 0.833 | 0.782 | **0.578** | — | vs SCIP +0.31, vs cuOpt +0.26 |
| cryptanalysiskb128n5obj16 | 1.000 | 1.000 | 1.000 | — | nobody |
| neos-2746589-doon | 1.000 | 1.000 | 1.000 | — | nobody |

**D-014 verdict: INCONCLUSIVE** — (a) 3 of 5 (needed ≥ 4 to support, ≤ 2 refutes), (b) holds. The "cuOpt closes early ⇒
we lose" half held on all three; the "cuOpt stalls ⇒ we win" half failed on both — on comp21-2idx and toguru cuOpt
stalled at 0.50–0.58 but the coupled system was worse still. **Out of sample the coupling is far weaker than on the
development set:** vs SCIP alone 3 better / 1 worse of 4 informative instances, mean d +0.14 [−0.00, +0.28]; SCIP's
weak set (8 of 8) improves only on 2 (comp21-2idx −26 %, crna −31 %), mean P 0.89 → 0.83; vs cuOpt alone 1 / 5,
mean d −0.51 [−0.84, −0.12]. The development set (chosen by looking, tuned on) overstates the effect — the expected
optimism of development data, now measured.

## Repair clock, Stage 2 — first 8 blocks (2026-10-07 12:23–13:21) — A3@rc1 (active) vs same-day A3@lps1w
16/16 completed, every incumbent checked. rc1 vs lps1w: **item_placement s0 0.279 vs 0.570 (d +0.51; 2,087 vs ~390
generations; 91.5 % of the post-feasible time below "none")**; glass4 s0 0.148 vs 0.220 (+0.33) and s1 0.235 vs 0.243
(+0.03) — the clock played "none" throughout on glass4, so this is trajectory noise, not the clock; ca s0/s2 −0.003 /
−0.004 (stayed "none", 13–14 cap hits); loue s1 −0.014 (88.6 % below none); neos-960392 s0 +0.014; fhnw s0 1.0 / 1.0.
No guard instance below −0.05 so far. The remaining 14 blocks run in the evening window (16:48–19:10).

## cuOpt MIP start — verification on the GPU (2026-10-07 16:46, glass4, `tools/cuopt_verify_mipstart.py`)
cuOpt 26.08 with `DataModel.set_initial_primal_solution`, 3-s solves: **the start is used** (final 1.6000131e9 =
the start's objective, echoed; control without a start 2.1e9; a second Solve on the same DataModel the same);
**presolve stays on** (PaPILO in the log, no "presolve disabled"); partial (NaN) starts are refused ("Error cannot add
the provided initial solution"); an out-of-bounds start is a ValidationError; SIGTERM is handled only after Solve
returns; deterministic mode ignores the start. **Wall time of a 3-s solve was 12–17 s** on this 322-variable model
(deterministic mode: 3.0 s) — each restart costs ~10 s beyond its time limit, which the restart design's slice
lengths must absorb. (The script's own summary line "start_used: False" uses a stricter echo test; the objectives above
are the evidence.)

## Repair clock, Stage 2 — complete (2026-10-07 12:23–13:21 and 16:48–18:03) — A3@rc1 (active) vs same-day A3@lps1w
22 blocks, 44 runs (both LLM servers stopped in both windows), every incumbent checked, GPU health ok on all.

| instance | seeds | P rc1 (seed values) | P lps1w | mean d | time below "none" | D-012 rule |
|---|---|---|---|---|---|---|
| item_placement_9903 | 0–3 | 0.279, 0.417, 0.260, 0.134 | 0.570, 0.555, 0.501, 0.553 | **+0.50** (4/4 better) | 94 % | target: ≥ 3/4 and ≥ +0.15 — **PASS** |
| sing44 | 0–1 | 0.315, 0.352 | 0.366, 0.452 | +0.18 | 87 % | ≥ −0.05 — PASS |
| glass4 | 0–1 | 0.148, 0.235 | 0.220, 0.243 | +0.18 | 0 % | ≥ −0.05 — PASS |
| neos-3024952-loue | 0–1 | 0.380, 0.332 | 0.356, 0.328 | −0.04 | 76–89 % | ≥ −0.05 — PASS |
| ca__Scenario400_reserves_1 | 0–3 | 0.296, 0.296, 0.280, 0.313 | 0.295, 0.322, 0.279, 0.310 | +0.02 | 0 % | ≥ −0.05 — PASS |
| radiationm18-12-05 | 0–1 | 0.165, 0.306 | 0.164, 0.294 | −0.02 | 0 % | within ±0.01 — **FAIL by the letter** |
| rail01 | 0–1 | 1.0, 1.0 | 1.0, 1.0 | 0.00 | 100 % | within ±0.01 — PASS |
| fhnw-binpack4-48 | 0–1 | 1.0, 1.0 | 1.0, 1.0 | 0.00 | 100 % | reported |
| neos-960392 | 0–1 | 0.032, 0.030 | 0.032, 0.030 | +0.005 | 52–76 % | reported |
Overall rc1 vs lps1w: 5 better / 2 worse of 7 informative, mean d **+0.12 [+0.00, +0.26]**. Generations: item_placement
1,813–2,087 vs ~390; loue 57/106 vs 47/42; neos-960392 21/52 vs 9/12.

**Two rules fail by the letter.** (1) radiationm −0.02 against a ±0.01 band — the clock played "none" for the whole
run (no horizon applied), so rc1 and its control ran the same search; the band was set tighter than seed-to-trajectory
noise. (2) Criterion (c), non-repair seconds per generation within 10 % of the control (ratio 0.35–2.25): not
overhead — on ca the 10-s per-call cap at "none" (active mode, after the first feasible point) fired 16 times, moving
time from repair (65–73 vs 95–98 s) to immigrants (31–40 vs 9–11 s) with no change in P (+0.02); on item_placement the
ratio falls because generations are far shorter. **Outcome-level: the target is met strongly and no guard is harmed.**
Decision in D-012 addendum 2.

## Three-way arm A9 — pilot (2026-10-07 19:58–20:41) — 7 instances, seed 0, 120 s, both LLM servers stopped
A9 = SCIP ↔ EA ← cuOpt (`configs/itrea2-dev-rc-3way.yaml`, repair clock active); A3x = its matched control (SCIP ↔ EA,
EA on the same X925 core, no cuOpt); A2 = SCIP ‖ cuOpt race (no exchange); A0 / A6 = seed-0 runs from the development
set and the class probe. 21/21 completed (+ a 30-s glass4 smoke run), every incumbent checked, GPU health ok.

| instance | SCIP alone | cuOpt alone | race A2 | A3x | **A9** | SCIP inside A9 | cuOpt inside A9 |
|---|---|---|---|---|---|---|---|
| neos-3004026-krka | 0.318 | 0.004 | 0.003 | 0.303 | **0.003** | 0.071 | 0.003 |
| fhnw-binpack4-48 | 1.000 | 0.005 | 0.005 | 0.301 | **0.009** | 0.114 | 0.009 |
| radiationm18-12-05 | 0.154 | 0.009 | 0.007 | 0.164 | **0.009** | 0.023 | 0.011 |
| neos-3656078-kumeu | 1.000 | 0.107 | 0.111 | 1.000 | 0.148 | 0.299 | 0.148 |
| chromaticindex1024-7 | 1.000 | 0.071 | 0.054 | 1.000 | 0.545 | 1.000 | 0.545 |
| item_placement_9903 | 0.631 | 0.666 | 0.617 | 0.379 | **0.362** | 0.362 | 0.620 |
| sing44 | 0.518 | 0.726 | 0.572 | **0.287** | 0.477 | 0.510 | 0.920 |

| A9 vs | better / worse (of 7) | mean d [95 % CI] |
|---|---|---|
| SCIP alone | **7 / 0** | **+0.68 [+0.42, +0.90]** |
| A3x (same coupling without cuOpt) | 6 / 1 | +0.55 [+0.16, +0.89] |
| cuOpt alone | 3 / 4 | −0.07 [−0.42, +0.25] |
| race A2 (SCIP ‖ cuOpt, no EA) | 2 / 5 | −0.18 [−0.48, +0.09] |

**Reading.** (1) cuOpt's solutions, passed through the EA, fix the coupled system's blind spot: where the EA alone found
nothing (krka, kumeu, chromaticindex1024-7) or little, A9 is 0.46–0.99 better than SCIP alone and than A3x — and SCIP
*itself* improves inside A9 (fhnw 1.0 → 0.114, kumeu 1.0 → 0.30, krka 0.32 → 0.07, radiationm 0.15 → 0.02; 1–3
cuOpt vectors per run reached SCIP as incumbents). (2) Against cuOpt alone and the race, A9 wins where the EA adds its
own improvements (item_placement +0.46 / +0.41, sing44 +0.34 / +0.17, krka +0.32 vs cuOpt) and loses where **cuOpt runs
slower beside the EA**: cuOpt inside A9 reached 0.545 on chromaticindex1024-7 against 0.071 alone, and 0.92 against
0.73 on sing44 — GPU contention, the effect optopt's A4 measured (continuous GPU work beside cuOpt doubled its P).
**Next:** the EA should yield the GPU to cuOpt when it is not contributing (a back-off / duty-cycle rule) — the D-010
"EA back-off" item, now with a measured cost; then seeds 1–2 and the D-013 weak-set report.

## Rank 1 as a distinct elite — D-016 result (2026-10-08 11:26–14:04) — A3@rc1d / A3@cl1 / A3@tp1 × 9 × seeds 0–1
[[D-016 Rank 1 as a distinct elite — cuOpt's clearing rule, development test]]. Pool 20261008-112631, commit 6ae6a6f,
54/54 completed, 0 void, every incumbent checked, `compiled_on_clock` false in all 54 (a first attempt's block at
11:11 was voided for kernels compiled on the clock — `60-incidents`). rc1d = the development default re-run in the
window (`1d12ad490dd0`); cl1 = + `ranks.rank1_mode: clearing` (`8d01ac13f25f`); tp1 = + `ranks.elite_basis:
population`, the size-only control (`e0483418b66d`). Both LLM servers stopped. `tools/d016_report.py`.

| instance | P rc1d | P cl1 | P tp1 | d cl1 vs rc1d (band) | d tp1 vs rc1d | gens rc1d / cl1 / tp1 |
|---|---|---|---|---|---|---|
| ca__Scenario400_reserves_1 | 0.297 | 0.298 | 0.298 | −0.002 (0.10) | −0.001 | 21 / 21 / 21.5 |
| fhnw-binpack4-48 | 1.000 | **0.653** | 1.000 | **+0.347** (0.02) | 0.000 | 34 / 22 / 33.5 |
| glass4 | 0.166 | 0.224 | 0.242 | −0.257 (0.37) | −0.312 | 36.5 / 27 / 26.5 |
| item_placement_9903 | 0.347 | 0.547 | 0.391 | −0.367 (0.68) | −0.113 | 1908 / 1316 / 1781 |
| neos-3024952-loue | 0.381 | 0.332 | 0.313 | +0.128 (0.13) | **+0.178** | 89.5 / 180.5 / 246.5 |
| neos-960392 | 0.031 | 0.033 | 0.031 | −0.032 (0.04) | +0.010 | 22 / 34.5 / 23 |
| radiationm18-12-05 | 0.235 | 0.235 | 0.230 | +0.002 (0.46) | +0.024 | 294.5 / 280.5 / 261.5 |
| rail01 | 1.000 | 1.000 | 1.000 | 0 (0.02) | 0 | 675.5 / 675 / 675.5 |
| sing44 | 0.360 | 0.339 | 0.275 | +0.060 (0.11) | **+0.238** | 80 / 120.5 / 93.5 |

| comparison | better / worse (of 9) | mean d [95 % CI] | Wilcoxon p |
|---|---|---|---|
| cl1 vs rc1d (D-016's test) | 4 / 4 | −0.013 [−0.141, +0.109] | 1.00 |
| tp1 vs rc1d (size only) | 4 / 3 | +0.003 [−0.096, +0.096] | 0.81 |
| cl1 vs tp1 (distinctness beyond size) | 2 / 6 | −0.019 [−0.120, +0.095] | 0.55 |

**Mechanism (amended D-016): HOLDS.** On the 7 instances feasible in both arms, median Rank 1 size 26 (cl1) vs 5
(rc1d); median banded spread `r1_dmax_band` 0.0296 vs 0.0007 (42×, gate 5×); the distance rule cleared candidates on
every one (`rank1_cleared` 35–3,570 per run). The clearing does what cuOpt's rule does: Rank 1 now holds up to 26
solutions that differ in ≥ k integers instead of 1–9 near-copies.

**Outcome (D-016's rule): INCONCLUSIVE** — mean d −0.013, no instance worse than its band, so seeds 2–3 follow on the
instances inside their band. Reading:
- **Diversity alone did not pay on average.** The elite is far more diverse, but the mean P did not move; the effect is
  instance-specific and large in both directions.
- **item_placement loses 0.18–0.22 in P on both seeds** (within its very wide 0.68 band) and runs **31 % fewer
  generations** (s0: 954 vs 1,823) — distant Rank 1 parents give children with larger disagreement sets, so more repair
  per generation; the size-only arm loses only 7 % of its generations. Checked on the phase timers: it is not the
  clearing's own cost — survival time FELL (3.6 / 5.7 s vs 11.4 / 10.4 s in rc1d) — but repair time rose (72.8 / 59.9 s vs
  50.9 / 48.5 s): children of distant parents need more repair, so diversity is paid for in generations.
- **fhnw (+0.347):** in cl1 seed 0 the EA found its own feasible point on fhnw (P 0.306) — the first time in any rc-family
  run; seed 1 did not. One seed: chance until replicated.
- **Size, not distinctness, helped sing44 and loue:** tp1 is better than rc1d beyond the band on both (+0.238, +0.178)
  and better than cl1 on sing44 (−0.189 for cl1 vs tp1, beyond band). Keeping more good feasible solutions (Berk's "a
  percentage of good feasible solutions") helps there even without the distance rule.
- glass4: all three arms are within its 0.37 band; the control's seed 0 (0.113) was its luckiest seed so far.
**Next:** seeds 2–3 on the in-band instances for cl1 AND tp1 (tp1 was not a pre-registered candidate for adoption; if it
holds up, it becomes its own decision); look at item_placement's repair time per generation under clearing.

## Rank 1 as a distinct elite — D-016 final (seeds 0–3; seeds 2–3 run 2026-10-09 12:40–14:38) — NOT ADOPTED
Pool 20261009-124048 (`jobs/dev-cl1b.jsonl`, commit 221ad82 code = 6ae6a6f): 48/48, 0 void, 0 `compiled_on_clock`;
together with 10-08: 102 runs (rail01 seeds 0–1 only, inert). `tools/d016_report.py` on seeds 0–3:

| instance | P rc1d | P cl1 | P tp1 | d cl1 vs rc1d (band) | d tp1 vs rc1d | gens rc1d / cl1 / tp1 |
|---|---|---|---|---|---|---|
| ca__Scenario400_reserves_1 | 0.299 | 0.293 | 0.299 | +0.019 (0.10) | −0.000 | 21 / 20.8 / 21.3 |
| fhnw-binpack4-48 | 1.000 | **0.826** | 1.000 | **+0.174** (0.02) | 0 | 35.5 / 29.5 / 35.5 |
| glass4 | 0.215 | 0.243 | 0.233 | −0.112 (0.37) | −0.075 | 31.8 / 28 / 28.5 |
| item_placement_9903 | 0.357 | 0.503 | 0.341 | −0.292 (0.68) | +0.044 | 1931 / 1565 / 1893 |
| neos-3024952-loue | 0.381 | 0.340 | 0.323 | +0.108 (0.13) | **+0.153** | 71 / 191 / 242 |
| neos-960392 | 0.032 | 0.034 | 0.032 | **−0.050** (0.04) | −0.015 | 21 / 27 / 21.5 |
| radiationm18-12-05 | 0.226 | 0.225 | 0.224 | +0.007 (0.46) | +0.012 | 286 / 280 / 264 |
| rail01 | 1.000 | 1.000 | 1.000 | 0 (0.02) | 0 | 675.5 / 675 / 675.5 |
| sing44 | 0.327 | 0.387 | 0.372 | **−0.155** (0.11) | **−0.120** | 71 / 77 / 65 |

| comparison | better / worse (of 9) | mean d [95 % CI] | Wilcoxon p |
|---|---|---|---|
| cl1 vs rc1d | 4 / 4 | −0.033 [−0.122, +0.050] | 0.64 |
| tp1 vs rc1d | 3 / 4 | −0.000 [−0.044, +0.049] | 0.94 |
| cl1 vs tp1 | 2 / 6 | −0.033 [−0.120, +0.039] | 0.25 |

Mechanism still HOLDS (Rank 1 26 vs 5, banded spread 0.0266 vs 0.0010, clearing active on all 7). **Reading:** a
diverse elite does not lower P here. Seeds 2–3 reversed the seed 0–1 signs of the size-only arm on sing44 (+0.238 →
seed-mean −0.120) — the two-seed effects were noise; sing44 and neos-960392 are now worse than their bands under
clearing (seed means; no instance worse on every seed, so the letter of the rule says "inconclusive", but the mean is
≤ 0 with an interval centred below zero after the pre-stated extra seeds). fhnw's gain is one seed of four (the EA found
its own feasible point once). item_placement pays for diversity in repair (−19 % generations, −0.29). **Decision: not
adopted; the development default stays A3@rc1.** Clearing and tp1 remain available as options (`ranks.rank1_mode`,
`ranks.elite_basis`). A natural follow-up is clearing with a much smaller k only where repair is cheap — not queued.

## LP decode — D-017 result (2026-10-09 14:49–16:07) — A3@rc1e vs A3@ld1 × 6 instances × seeds 0–1 — ADOPTED
[[D-017 LP decode — re-optimise the continuous variables of selected candidates]], issue #27. Pool 20261009-144950,
main 87bc759; 24/24, 0 void, 0 `compiled_on_clock`; both LLM servers stopped. `tools/d017_report.py`.

| instance | P rc1e | P ld1 | d (band) | gens rc1e / ld1 | decoded copies entered F / I | decoded incumbents F / I | median gain of entered F copies |
|---|---|---|---|---|---|---|---|
| ca__Scenario400_reserves_1 | 0.298 | 0.296 | +0.005 (0.10) | 21 / 21 | 19 / 3 | 3 / 0 | 0.9–1.0 % |
| fhnw-binpack4-48 | 1.000 | 0.652 | +0.348 (0.02) — not lp_decode's | 35 / 22 | 0 / 0 | 0 / 0 | — |
| glass4 | 0.219 | 0.230 | −0.047 (0.37) | 26 / 30 | 30 / 87 | 12 / 3 | 1.1–1.7 % |
| item_placement_9903 | 0.311 | **0.118** | **+0.619** (0.68) | 1,834 / 1,096 | 253 / 5,214 | 62 / 28 | 7.6–12.4 % |
| radiationm18-12-05 | 0.235 | 0.230 | +0.020 (0.46) | 292 / 288 | 0 / 0 | 0 / 0 | — |
| sing44 | 0.377 | **0.238** | **+0.369** (0.11) | 104 / 71 | 12 / 1 | 10 / 0 | 2.2–4.8 % |

Mean d **+0.219 [+0.041, +0.429]**, 5 better / 1 worse (Wilcoxon p 0.16); without fhnw +0.193; 9-instance mean (inert
three at 0) +0.146. Mechanism: M0, M2, M3 hold; M4 (completion) works on item_placement (5,215 LP-feasible completions
of 5,439 screened rows) and glass4, never on radiationm (0 / 6,879) or fhnw; **M1 fails by the letter on
item_placement** (decoder host time 1.39 / 1.63 % of wall vs ≤ 1 %; per call ≤ 20.5 ms everywhere). Reading: polishing
the continuous part of feasible solutions and completing super-optimal infeasible ones with an LP is the largest single
gain of the development so far — item_placement drops from 0.31 to 0.12 although the EA runs 40 % fewer generations, and
sing44's EA finds 10 incumbents through decoded copies. fhnw's gain is a trajectory change (the decoder kept nothing
there). **Adopted as the A3 development default** with the M1 deviation stated (a cost to the arm): `configs/itrea2-dev-rc-ld.yaml`.

## A9 GPU back-off — stage-0 calibration, partial (2026-10-09 16:12–17:10) — 13 of 18 runs
See [[D-018 A9 — the EA backs off the GPU beside cuOpt (contribution-gated duty cycle)]], "Stage 0, partial result":
fixed EA duty ½ brings cuOpt's first chromaticindex incumbent from 65.3 s (full duty) back to 8.3 s (alone 8.1 s) and
A9's P there from 0.544 to 0.069 (cuOpt alone 0.068); on sing44 duty ½ gives A9 P 0.361 vs 0.453 at full duty, 0.579 for
SCIP ‖ cuOpt, 0.734 for cuOpt alone, and the late stretch drops from 1.84 to 1.14. ⅛ duty: chromaticindex late again
(62.3 s) — one-seed anomaly. comp21 block not run (deadline); the pre-stated lane-steps sanity limb fails in both
directions (a refuted premise), deviation proposed in D-018. Verdict by the letter: INCOMPLETE.
Stage 0 complete (2026-10-09 19:16, comp21 block): comp21's cuOpt-inside P stays above cuOpt alone at every duty
(0.69 / 0.60 / 0.62 / 0.59 at 1 / ½ / ¼ / ⅛ vs 0.52) → the floor goes to ⅛ by the rule's shortfall; D_Y = ½. Verdict
PROCEED; bo1 / bo1s written (D-018 "Stage 0 complete").

## A9 GPU back-off — D-018 stage 1 (2026-10-11 09:40–13:40) — A9@bo1 vs A9@bo1s + same-day A6, 14 × seeds 0–1 — NOT ADOPTED
Pools 20261011-094039-3927304 (seed 0) and 20261011-112456-173723 (seed 1): 82 completed, 2 failed (ca seed 1, A6 and
A9@bo1: cuOpt's 12-GB RMM limit, "Memory allocation failed" at 78 s / 83 s; kept as D-002 §6 keeps crashes), 0 void,
0 compile on the clock. `itrea backoff-report --stage 1 --failed keep` (main after 2a18819; `_band` fixed to read the
control `A9@bo1s`, see the decision record).

| instance | P A6 | P bo1s (ctrl) | P bo1 | d bo1 vs ctrl | band |
|---|---|---|---|---|---|
| chromaticindex1024-7 | 0.083 | 0.545 | 0.091 | +0.834 | 0.05 |
| neos-3656078-kumeu | 0.095 | 0.116 | 0.090 | +0.227 | 0.40 |
| comp21-2idx | 0.507 | 0.575 | 0.557 | +0.032 | 0.08 |
| neos-4763324-toguru | 0.574 | 0.598 | 0.616 | −0.030 | 0.06 |
| item_placement_9903 | 0.643 | 0.308 | 0.319 | −0.037 | 0.26 |
| ca__Scenario400_reserves_1 | 0.671 | 0.394 | 0.296 | +0.249 | 0.05 |
| neos-3024952-loue | 0.749 | 0.539 | 0.492 | +0.087 | 0.10 |
| neos-2657525-crna | 0.771 | 0.394 | 0.439 | −0.103 | 0.87 |
| neos-960392 | 0.637 | 0.057 | 0.058 | −0.004 | 0.21 |
| glass4 | 0.339 | 0.237 | 0.224 | +0.054 | 0.22 |
| sing44 | 0.755 | 0.434 | 0.384 | +0.116 | 0.10 |
| neos-3004026-krka | 0.005 | 0.003 | 0.006 | −0.498 | 0.71 |
| fhnw-binpack4-48 | 0.006 | 0.012 | 0.009 | +0.214 | 0.26 |
| radiationm18-12-05 | 0.011 | 0.013 | 0.013 | +0.026 | 0.38 |

- Outcome rules all hold: T (chromaticindex d +0.834, P 0.091), G, ladder, floor, mean over 11 non-floor +0.130.
- Mechanism fails: M1 (toguru phase-Y exit 0.56 / 0.80 s after cuOpt's first incumbent, rule 0.5 s), M3 (EA leads not
  protected: item_placement 196 vs 289 and 143 vs 292, loue s1, crna s1), M5 (LS cap hits per call up on ca).
- bo1 vs bo1s: 9 better / 5 worse, mean d +0.083 [−0.054, +0.232].
- vs cuOpt alone (A6, 3 A725 cores): bo1s 8 / 6, +0.135 [−0.109, +0.362]; bo1 8 / 6, +0.195 [+0.016, +0.383]; cuOpt's
  weak set here (8): P 0.663 → 0.412 (bo1s, −38 %) / 0.395 (bo1, −40 %), d +0.360 [+0.150, +0.571] / +0.384 [+0.172,
  +0.593]. vs SCIP alone (A0 of dev / classprobe): 14 / 0 for both, d +0.600 [+0.452, +0.753] / +0.649 [+0.502,
  +0.798]. Weak sets selected on the compared control runs (descriptive).
- crna is bimodal: control 0.698 / 0.090, bo1 0.144 / 0.734 by seed — the EA either leads early or not at all.
- Next candidate: phase Y only (no ladder), its own record.

## D-021 measurements M1–M3 and the interim calibration (2026-10-11 15:12–18:13; 2026-10-12 12:06–12:40; 2026-10-13 12:32–15:16) — the external harness, not the method; M4 INTERIM, INCOMPLETE (19 of 30)
Full record: [[2026-10-11 D-021 measurements M1–M3 and the interim calibration]]. Harness and calibration runs of
[[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]] at A9's
footprint F (X925 5–6, A725 1–3, the GPU), both LLM servers stopped, 0 void and 0 compile on the clock in every pool.
- **M1 smoke** (pool 20261011-151204-1160038; fhnw-binpack4-48, glass4, radiationm18-12-05 × {A9, A6F, A6F with the
  thread variables kept, A2F} × 120 s, seed 0): 12 / 12; every harness setting as specified (cuOpt threads 5 / 4 / 3 by
  arm, gaps 0, RMM 24 GB / host 24 GB, named scopes, SM 2,392–2,405 MHz, ≤ 78.5 °C). R14 inert: cuOpt CPU-s per wall-s
  5.02 / 4.99 / 5.03 with the thread variables unset, 5.02 / 4.98 / 5.03 with them set.
- **M2 soak** (20261011-153534-1276706; comp21-2idx, neos-4763324-toguru × {A9, A6F} × 600 s): 4 / 4; steady-state
  p95 66–73 °C, maximum 76 °C, no cap, busy SM 2,392 MHz → **F accepted** for v1 (R23).
- **M3 memory probe** (20261011-163415-1522373; 8 × A6F at cuOpt RMM 40 / host 40 GB): eilA101-2 and nw04 killed at
  the 40-GB host cap, supportcase6 at 22.6 GiB host (> 19.2) → **does not fit F: eilA101-2, nw04, supportcase6**,
  excluded from both arms; decision set 218 → 215, H_cu-unseen 65 → 64.
- **M4 calibration — INTERIM, INCOMPLETE (19 of 30)** (A6F × D-020's 30-instance sample × 600 s, seed 0; pools 20261012-120607-2036840, 5 runs,
  paused for [other project], and 20261013-123201-3850111, 14 runs): stopped at 19 of 30 by D-024 rule 0
  (performance runs first). Late-clock check on the 6 strata-0–1 instances run: median first-incumbent offset ours −
  GAMS −0.119 s — passes (≤ +0.3 s); 4 strata-0–1 instances still owed before v1's first external run. Footprint ratio,
  call clock: R 1.390 [1.081, 1.837] over 19 (1.416 [1.087, 1.903] over the 18 read at 14:4x); launch clock 2.036
  [1.572, 2.654]; feasible 16 vs GAMS 16. The excess is in strata 3–4 (csched008 first incumbent 85.6 s vs 13.8 s,
  dano3_3 43.0 vs 14.2 s), consistent with our cuOpt on 5 threads on F vs GAMS's 24 on the whole box; no footprint
  reading before 30 of 30. *(Completed 2026-10-15: 30 of 30, OUTSIDE — § "External harness in phase 1's last week" below.)*
- **Checker tolerance, found, to investigate:** in M4 our checker passed a mas74 cuOpt incumbent below MIPLIB's proven
  optimum (11,801.18516 vs 11,801.18572) and rejected 4 cuOpt incumbents (neos-4338804-snowy 1: a bound off by
  8.3e-6; neos-933966 3: integrality off by 4.8–6.5e-6; our tolerance is 1e-6). In the 2026-10-11 run table, 60 checked
  incumbents in 8 of 57 glass4 runs beat glass4's proven optimum by GAMS's rounded test (`beats_ref`; 33 EA-role, 27
  SCIP records of EA posts; 62 lie strictly below z*, the first at 44.5 s) — all 8 are EA arms: A3@ad1, cl1, lpd, rc0,
  rc1, rc1d, stall100, A5@c04; first `beats_ref` at 47.8–119.9 s; worst row violation ≤ 4.5e-7 relative; cuOpt 26.10 rejected the lowest
  of them (1,200,006,662.64) when offered it. The effect on any glass4 development reading has not been computed (#36).
- Running after M4 in the same window (W-1013): D-022 (set `itx-r13`: `jobs/itx-r13.jsonl`, 58 runs, then `jobs/itx-r13b.jsonl`, 16, same set) from 15:16 — no
  result exists yet; its rules were fixed before any reading ([[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]],
  second amendment f52e3e8, clarification fd75476). Then M6 (A0's CPU lanes). *(Read 2026-10-13 22:2x: § "A9 under the
  external harness — D-022 result and C1's first interim reading" below.)*

## cuOpt 26.10 nightly — the cuopt#2073 probe (2026-10-13 10:1x–10:5x) — no A9 result
Berk (10:00): "If the nightly for the next version is up can you check if our feature request in Github issues is
implemented properly?" NVIDIA/cuopt#2073 (filed 2026-10-07: a `SetSolutionCallback` disables cuOpt's presolve) was
closed on 2026-10-12 by PR #2098, merged into `main` = 26.10.00. Probe: `tools/probe_cuopt_setsol.py` and
`tools/probe_cuopt_setsol_report.py` (e5217ae) on the nightly `cuopt-cu13 == libcuopt-cu13 ==
26.10.0a294.post261013051018` in a separate venv `~/venvs/cuopt-2610n` (installed on Berk's ask; the repository venv
stays on 26.08); 31 solves of 20–30 s (most on the nightly; the injection also on 26.08 in the repository venv), 3 threads, RMM 8 GB, short lease, both LLM servers left up (for [other project],
as agreed on 10-12). Five development instances (glass4, sing44, neos-3024952-loue, neos-2657525-crna,
item_placement_9903).
- **Implemented properly:** with a SET callback, presolve stays on in 26.10 (26.08: "Presolve is disabled because
  set_solution callbacks are provided"); solutions injected in the original variable order are accepted 1–6 ms after the
  offer, echoed back unchanged (max |Δ| ≤ 1e-12) and pass our exact checker on all five; infeasible offers (a fractional
  integer, a shifted vector) are rejected; the glass4 point below MIPLIB's proven optimum is rejected. Papilo's dual
  reductions are off while a SET callback is registered: the presolved size was unchanged on 4 of 5, crna 308 vs 305
  variables.
- **Limitation (not changed by #2098):** cuOpt polls SET callbacks only when its GPU population records a new best —
  crna: 11 calls, all at 0.70–0.71 s, none in the remaining 29 s; sing44: the first call at 28.3 s. An outside solution
  enters only while cuOpt's own heuristics improve, not when it has stalled.
- **Berk (11:xx): "I just got word it is integrated to 26.10."** The two upstream drafts (a thanks comment; a request to
  poll SET on a timer) stay unposted (`vault/70-upstream/2026-10-07 NVIDIA-cuopt#2073 — SetSolutionCallback with
  presolve.md`, `vault/70-upstream/drafts/`). Latest cuOpt release on 2026-10-13: still v26.08.00.
- **No A9 result:** no arm ran on 26.10, and A9 is unchanged (one-way cuOpt → EA). What it opens: A9 two-way — the EA's
  and SCIP's incumbents offered to cuOpt through the SET callback with presolve on (#35); a development item after the
  26.10 re-baseline (#31) or for v1.1, with its own decision record and test before any run.

## Phase 1's last week (2026-10-13 afternoon – 2026-10-16) — what ran, in run order
The development sets below ([[D-010 Develop the algorithm first on a small development set]]) ran at 120 s under harness
ext (except D-022's A9@r13 and D-028's A9@boY under r13-off: ext with R13 + R14 off) with both LLM servers stopped,
n = 2–6 seeds on a few development instances — **descriptive**; the external-harness section (M6, A0 on the 81, M4) is harness and v1 groundwork, not method development. No line here says that A9 or a change "improves" or "beats"
anything beyond the numbers given. Each section quotes its decision record's
Result (and the journal where named); run counts and run hashes are from `datasets/runs.parquet` (built 2026-10-16 08:5x).
Phase 1 (the Roadmap's stage 1, development until C1) wraps up **Sat 2026-10-17** (Berk 2026-10-15 13:4x: "Assume we are
wrapping up phase 1 on Saturday"; [[D-019 v1 — deadline and performance target]] § Addendum 2026-10-15).

## A9 under the external harness — D-022 result and C1's first interim reading (2026-10-13 15:16–19:30) — ADOPT R13 + R14
Set `itx-r13`: pool 20261013-151610-335372 (`jobs/itx-r13.jsonl`, 58 runs, 15:16–18:28) and the C1 guard pool
20261013-182821-1425786 (`jobs/itx-r13b.jsonl`, 16 runs, 18:28–19:30) — **74 of 74 completed**, 0 failed / void. Arms
A9@r13 (ext with R13 + R14 off; 6 runs), A9@r13x (ext, run hash 28e5a5e9c725; 34), A6F@r13x (cuOpt alone on F,
82a371305f88; 34); seeds 0–1. `itrea build-runs`, `itrea d022-report`. Record:
[[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]] § Result.
- **D-022's rule** (fhnw-binpack4-48, neos-3004026-krka, neos-960392; 18 runs): validity holds; the mechanism is met only
  vacuously — 0 of 6 A9 runs in either arm ended on cuOpt's "optimal" (SCIP proves first), while A6F@r13x ended that way on
  5 of 6; no harm: fhnw ΔP −0.0016, krka −0.0001 (limit ΔP ≤ +0.005), neos-960392 d +0.022 (band 0.41) → **ADOPT R13 + R14
  for A9**; the C1 freeze carries harness ext.
- **C1's first interim reading** (68 of 68; A9 on the then default `configs/itrea2-dev-rc-3way.yaml`): **HINT present 0.827**
  (ratio of mean P(120) A9@r13x / A6F@r13x over the fixed 11 = 0.585 / 0.707; present iff ≤ 0.95, futility-only), **GUARD
  holds +0.063** (mean bounded d over the 17; holds iff ≥ −0.05). Reported, not criteria: vs cuOpt alone on the 11 improved 6,
  worse 3, mean P −17 %, d +0.185 [+0.032, +0.368], Wilcoxon p 0.074; vs SCIP alone (A0, earlier days) improved 8, worse 0,
  0.880 → 0.585 (−34 %), d +0.331 [+0.172, +0.500], p 0.0078. Worse than cuOpt alone on comp21-2idx, neos-4763324-toguru,
  neos-2746589-doon among the 11, and among the other 6 on four: chromaticindex1024-7 (0.061 → 0.544, d −0.889),
  fhnw-binpack4-48 (−0.508), radiationm18-12-05 (−0.226) and neos-3656078-kumeu (−0.076); fhnw, radiationm18 (and krka)
  are at floor-level P (D-022 § Result, "The other 6").
- Caveats (the record's): the fixed 11 were chosen on earlier 3-core A6 runs; A0 is from earlier days; neos-960392 carries
  48 % of the 11's summed P reduction (without it the ratio would be 0.902).

## A9 back-off, phase Y only — D-023 result (2026-10-14 14:45–17:49) — A9@boY vs A9@bo1s + same-window A6F, 11 × seeds 0–1 — NOT ADOPTED
Set `backoff2`, pool 20261014-144534-2277541 (`jobs/backoff2.jsonl`, 22 blocks): **66 of 66 completed**, 0 failed / void.
A9@boY (`configs/itrea2-dev-rc-3way-boY.yaml`, EA 5c8b535fb245; run hash under ext 86b1e527db68) vs A9@bo1s (f22e8571f52e) +
A6F (82a371305f88). `itrea backoff-report --stage 1y --harness ext`. Record:
[[D-023 A9 back-off, phase Y only — yield until cuOpt's first incumbent, then full duty]] § Result.
- **T fails:** chromaticindex1024-7 d +0.030 (P 0.545 → 0.529; needs d ≥ +0.30 and P ≤ 0.30). **M2 fails:** cuOpt's first
  incumbent inside boY at 63.47 / 63.41 s vs A6F's 10.87 / 10.52 s (D-018 stage 1, old harness, same phase-Y values: 9.4 /
  12.4 s).
- **G fails** on sing44's "still beats A6F" leg (+0.111 < 0.15; the control +0.229). **M5 fails:** local-search cap hits per
  call on ca 0.94 / 1.00 vs the control's 0.61. M1, M3, M4, F, the ladder targets and the mean over the nine (+0.023) hold.
- Reported, not criteria: ca d +0.247 (0.395 → 0.298), the LP-core mechanism the diagnosis note predicted; kumeu +0.255. The
  control's item_placement P (0.218) lies well below A9@r13x's on 10-13 (0.364): item_placement's day-to-day spread is large.
- **Consequence:** the back-off line (D-018, D-023) is closed for v1 without a GPU-sharing change; the next A9 control is
  A9@r13x.

## C1 diagnosis (2026-10-14, analysis of existing runs, no new runs)
[[2026-10-14 C1 diagnosis and improvement candidates]] (54c2faa; an analysis, not a decision): A9's losses come from cuOpt
running slower inside A9 with no partner making up for it (inside-A9 slowdown +0.064 mean over the 17, partner gain +0.147;
journal 2026-10-14 15:xx). Ranked candidates: R1 (the EA at blocking sync so its LP worker gets core 6) → D-025; R2
(lp_decode inside A9 after the LP worker) → D-026; R3 (same-day thread / SCIP diagnostic) → D-027; R4 (two-way exchange,
cuOpt 26.10) → v1.1 (Berk 10-14 16:0x: "v1.1, no nightly runs"); R5 (aggregation probe, CPU). Berk (16:0x): if R1 / R2
change A9's default before the freeze, **"Re-run C1 on frozen hash"** (→ D-030).

## External harness in phase 1's last week — M6 → 3 lanes, A0 on the 81 (validity only), M4 complete (OUTSIDE)
- **M6** (A0's lanes, CPU only; pools 20261013-193050-1695474, set `itx-m6a`, 3 runs, and 20261013-193352-1723855, set
  `itx-m6`, 17 runs; 10-13 19:30–20:14): SCIP nodes/s slowdown beside 3 other lanes **13.2 %** (fhnw 15.0, glass4 12.1, crna
  12.4) > 3 %; hottest 78.6 °C → **3 lanes** (rule fixed before the runs: 4 lanes iff ≤ 3 % and < 85 °C, else 3). `itrea
  itx-report --m6`; journal 2026-10-13 22:2x.
- **A0 (SCIP alone) on the 81** of D-024 (f) (H_cu ∪ W_SCIP unseen; set `itx-a0`, `jobs/itx-a0-81.jsonl`, A0@v1 run hash
  ccaa8af273ca, 600 s, seed 2000, `--layout screen --cpu-lanes 3`, the 25 v1 stage-1 instances first): W-1013b 22:34–23:24
  (pool 20261013-223431-2448203, 16 completed) and W-1014 17:49–21:48 (pool 20261014-174913-3077187, 64 completed + 1
  failed) → **80 completed + 1 failed of 81**. The failure: s100 — SCIP did not stop at its 600-s limit and the runner killed
  it at 701 s (rc −9; scope peak 3.8 GB, under the 9-GB cap); kept as an A0 failure (D-002 §6). **Validity only:** A0 may run
  before prereg-v2's tag with only validity read (D-021 R33); no outcome of these runs is read or reported. Deviation on
  record: W-1013b started 8 s after the servers stopped, without D-021 R26's opening gate (hottest zone 52–54 °C); CPU-only,
  kept; the pool's missing gate is #38 (stopgap `tools/r26-gate.sh`, used at every window opening since). Journal
  2026-10-13 22:34 / 23:24, 2026-10-14 21:48.
- **M4 complete — OUTSIDE** (set `mipfeas`, A6F, 600 s, seed 0): the last 11 runs in pools 20261014-214807-4086700 (9; 10-14
  21:48–23:09) and 20261015-091023-2810143 (2; done 10:00) → **30 of 30**. `itrea mipfeas-report --arm A6F`: **R 1.313
  [1.091, 1.618]** on the call clock (launch clock 1.963 [1.606, 2.408]); feasible 27 vs GAMS 27; late-clock check passed
  (median −0.119 s over strata 0–1, n 10) → **OUTSIDE** the band [0.870, 1.150]: the difference is the footprint F (5 threads +
  the GPU against GAMS's 24-thread whole box) and the machine; v1's targets stay anchored on our own A6F (D-021). The interim
  R above (1.390 on 19 runs; 1.416 on 18) is superseded. Record: [[2026-10-11 D-021 measurements M1–M3 and the interim calibration]]
  § "Addendum 2026-10-15 — M4 complete (30 of 30): OUTSIDE".

## R1 — the EA's CUDA context at blocking sync: D-025 result (2026-10-15 07:17–08:06) — A9@bsync vs A9@r1ctl, 5 × seeds 0–1 — NOT ADOPTED
Set `r1-bsync`, pool 20261015-071709-2205151: **20 of 20 completed**, 0 failed / void; ca, sing44, neos-960392 (hint),
item_placement_9903, comp21-2idx (no-harm); A9@r1ctl (= A9@r13x's configuration, 28e5a5e9c725) vs A9@bsync (330a10a77556).
GPU checks before the pool (07:08–07:17): `tools/bsync_check.py` host self-CPU per wall-second auto 0.9998 → blocking_sync
0.0083; the core's competitor 0.5187 → 0.9989. `itrea d025-report` (VALID). Record:
[[D-025 A9 EA blocking sync (R1) — the EA stops spinning on the GPU so its LP worker gets core 6, development test]] § Result.
- Mechanism (reported): the LP worker's CPU share 0.93–0.96 (control 0.50); its finish ca 35.0 / 35.1 → 18.9 / 19.3 s.
- **Hint PRESENT 0.765** (ca 0.393 → 0.248, sing44 0.436 → 0.385, neos-960392 0.058 → 0.045).
- **No-harm FAILS:** generations on seed means item_placement 0.824 (threshold 0.921), comp21 0.818 (0.950); comp21's P +0.060
  (band 0.050) → **NOT ADOPTED**; fallback (b) `ysync` fires (wake latency).

## R3 — same-day diagnostic of cuOpt's threads and SCIP's presence: D-027 result (2026-10-15 08:06–09:10) — 22 runs
Set `r3-diag`, pool 20261015-080645-2420956: **22 of 22 completed**; A6@r3 (3-thread lane, e54170304dad; 8), A6F@r3
(82a371305f88; 6), A2@r3 (beside SCIP, 9c3031af075f; 8) on chromaticindex1024-7, comp21-2idx, neos-2746589-doon,
neos-4763324-toguru × seeds 0–1. `itrea d027-report` (VALID). Record:
[[D-027 Same-day diagnostic of cuOpt's thread count and SCIP's presence (R3) — reading rules fixed before the runs]] § Result.
- **Rule 4 → EA PRESENCE** (chromaticindex1024-7): cuOpt's first incumbent A6F@r3 9.80 / 7.33 s, A6@r3 9.24 / 8.21 s, A2@r3
  6.53 / 9.65 s, against 62–66 s inside A9 on every A9 run on record except three back-off arms on the old harness (bo1
  9.38 / 12.35 s, bof25 9.81 s, bof50 8.33 s; D-027 amendment 18:3x, run table `cuopt_t_first`). The EA's generation 0
  there is a 48.3-s GPU repair
  (~1,300 back-to-back launches of ~40 ms).
- **Rule 1 → closed** (no thread effect on toguru or doon): the layout line is closed for v1. **Rule 3 → SCIP's presence ruled
  out** on toguru and doon at n = 2. **Rule 2 → not read** (precondition not met).
- Beside: doon's cuOpt-alone P 1.000 / 1.000 here vs 0.792 / 0.996 on 10-13 — doon's C1 "loss" sits on cuOpt-alone's edge.
- Berk (09:3x, question panel, CUDA MPS): **"Not for v1"** → v1.1.

## R1 fallback (b) — yield instead of spin: D-025 result (2026-10-15 10:00–11:09) — A9@ysync vs A9@r1ctl-ysync — NOT ADOPTED
Set `r1-ysync`, pool 20261015-100013-3024118: **20 of 20 completed**; the same 5 instances; A9@ysync
(`configs/itrea2-dev-rc-3way-ysync.yaml`, a08221cca653) vs A9@r1ctl-ysync (28e5a5e9c725). `itrea d025-report --variant ysync
--set r1-ysync` (VALID). Record: D-025 § "Result of fallback (b) ysync".
- **Hint PRESENT 0.834** (ca 0.392 → 0.253, the LP worker done 37.3 / 38.3 → 20.0 / 20.2 s; sing44 0.408 → 0.418; neos-960392
  0.059 → 0.045).
- comp21's guard holds (generations 0.980; P −0.001); **item_placement's generation guard fails** — 0.759 vs 0.921: seed 0
  1.085, **seed 1 678 vs 1,480 generations = 0.458** — the pool's first run, started one second after M4's 600-s pool ended
  (load average 5.75 at its gates; not voided) → **NOT ADOPTED**; "the R1 line ends for C1" — reopened once by D-029 on Berk's
  choice (11:3x).

## d028 — does D-022's harness stop phase Y from restoring cuOpt on chromaticindex? (2026-10-15 11:09–11:17) — NOT R13 + R14
Set `d028`, pool 20261015-110921-3312868: **4 of 4 completed**; chromaticindex1024-7 × seeds 0–1 × {A9@boY under `r13-off`
(36a2c7e5c4e8), A6F under ext}. cuOpt's first incumbent: A6F 7.45 / 9.91 s → L = 19.82 s; **A9@boY under r13-off 63.25 / 63.24
s** (≥ 50 s on both seeds) → **not R13 + R14**. D-022's adoption is not implicated; chromaticindex stays a known v1 loss to
cuOpt alone (MPS is v1.1). Record:
[[D-028 Does D-022's harness (R13 + R14) stop phase Y from restoring cuOpt on chromaticindex — a 4-run diagnostic]] § Result.

## R2 — lp_decode inside A9 on the EA's core after the LP worker: D-026 result (2026-10-15 11:17–13:16) — ADOPT
Set `r2-ld`, pool 20261015-111740-3344629: **28 of 28 completed**; item_placement_9903 and sing44 × seeds 0–3 (hint), ca,
glass4, neos-3656078-kumeu × seeds 0–1 (no-harm); A9@r2ld (`configs/itrea2-dev-rc-3way-ld0.yaml`, EA a05af2c732dc, run hash
341267810eed) vs A9@r2ctl (`-3way.yaml`, 28e5a5e9c725). Run on the branch "R1 not adopted" (`make-jobs r2-ld --base r13x`).
`itrea d026-report --base r13x` (VALID). Record:
[[D-026 lp_decode inside A9 on the EA's core after the LP worker (R2), development test]] § Result.
- Mechanisms 1–3 hold (decoder ready 0.23–0.26 s after the gate, 5.83 / 5.91 s on ca; decoded incumbents on every seed of the
  hint instances; ca's LP worker solve wall ratio 1.039 / 0.940 ≤ 1.1).
- No harm on every leg, all in the treatment's favour: ca −0.006 (band 0.021), glass4 −0.030 (0.050), kumeu −0.054 (0.138),
  item_placement −0.133 (0.104), sing44 −0.062 (0.050).
- **Hint PRESENT 0.758** (ΣP 2.4362 / 3.2145; threshold 0.90): item_placement 0.330 → 0.198 (lower on all four seeds), sing44
  0.473 → 0.411 (lower on all four seeds) → **ADOPT**: `-3way-ld0.yaml` becomes A9's and A3x's development default and the
  frozen-v2 candidate. The record: "nothing claims that R2 improves A9 in general (development, n = 2–4 seeds on 5
  instances)".

## R1 follow-up — ysync judged on seeds 0–5: D-029 result (2026-10-15 13:16–14:59) — ADOPT
Set `r1f-ysync` (+ one unread warm-up run in `r1f-ysync-warmup`), pool 20261015-131611-3862708: **21 of 21 completed**;
item_placement_9903 and comp21-2idx × seeds 2–5, ca × seeds 2–3; A9@ysync (a08221cca653) vs A9@r1fctl (28e5a5e9c725).
`tools/d029_report.py` over r1-ysync (seeds 0–1) + r1f-ysync (seeds 2–5): VALID, 28 of 28. Record:
[[D-029 ysync gets one follow-up at seeds 2–5, judged on all seeds 0–5 with the failing seed kept — NOT ADOPTED ends R1 for v1]]
§ Result — its title states the pre-run framing (a NOT ADOPTED would have ended R1 for v1); **its Result is ADOPT**.
- Rule 1 (generations, seeds 0–5, ≥ 0.90): item_placement **1.020** (seed 1's 0.458 kept in), comp21 **1.152** — hold.
- Rule 2 (P, seeds 0–5, ≤ +0.05): item_placement **+0.016**, comp21 **+0.015** — hold.
- Rule 3 (ca, seeds 2–3 alone, ≤ 0.95): **0.675** — ca's gain replicates (seeds 0–3 pooled 0.659).
- **ADOPT:** `host.cuda_sched: yield` joins A9's development default with D-026's lp_decode → the frozen-v2 candidate
  `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` (EA 835c7c5a249c, A9 run hash f345de0ab936). Seed 1's collapse did not recur in
  four more seeds; its cause stays unknown.

## C1 re-read on the frozen-v2 candidate: D-030 result (2026-10-15 15:09–17:30, 2026-10-16 07:08–08:17) — HINT 0.809, GUARD +0.050, HARM flag on glass4
Set `c1r-ysync-ld0` (`jobs/c1r-ysync-ld0.jsonl`): the 17 development + class-probe instances × {A9@c1r = `-3way-ysync-ld0.yaml`
(f345de0ab936), A6F@c1r (82a371305f88)} × seeds 0–1, both arms of a block in one window. Pools 20261015-150955-187322
(**halted 17:30:56 by D-021 R26's breaker**, heat — the ✗ row DEV-X4 of the [[Experiment index]]; the journal's count at the
halt, 36 completed, 1 failed, 2 void, counted the rerun's A9, which had completed and was then voided with its block: the
run table holds 35 completed + 1 failed there, and `traces/raw/c1r-ysync-ld0/void.jsonl` three voided run directories — the
first A9 (gpu_cap, 17:19:18), then at 17:30:56 the rerun's completed A9 and its A6F; D-030's "2 voids" are the two the pool
voided at the gpu_cap check) and 20261016-070810-4125812 (the 16 remaining blocks, the voided item_placement s1 block rerun
whole; 31 completed + 1 failed). Run table: A9@c1r 34
completed, A6F@c1r 32 completed + 2 failed → **68 runs present, VALID** (failed runs kept). The failures: **comp21-2idx A6F@c1r
seeds 0 and 1, cuOpt rc −11 (SIGSEGV)** — cuOpt alone crashed, on both days; kept as A6F outcomes (D-002 §6). `itrea c1r-report
--a9 ysync-ld0 --json logs/d030-reading.json` (saved for D-031). Record:
[[D-030 C1 is re-read once on the frozen-v2 candidate (ld0) with A6F in the same window — 68 runs under D-022's hint and guard and D-025's harm leg]] § Result.
- **HINT present 0.809** (P̄ 0.548 / 0.677 over the fixed 11; 10-13 on the old default: 0.827). **GUARD holds +0.050** over the
  17 (10-13: +0.063). Futility-only: the hint gates the widening to four seeds (D-031) and declares nothing.
- **HARM** (D-025 amendment 3, cross-day against A9@r13x of 10-13) **on glass4:** d_i −0.088 < −0.057 (P 0.216 → 0.237); beside,
  the unchanged configuration's replicate (A9@r2ctl, 10-15) d −0.061 and cuOpt alone −0.035 — a possible day effect, checked by
  D-032. neos-3004026-krka ΔP +0.0010, below D-022's 0.005 floor (marked).
- **The letter (the two crashed A6F runs excluded): INCOMPLETE** — hint and guard unread (comp21's A6F missing on both seeds); vs
  cuOpt alone on the 10 with both arms: improved 6, worse 3, P 0.667 → 0.546 (−18 %). The kept reading gains from comp21's
  crashes (A6F P = 1 on both seeds); stated beside the go / no-go.
- Reported (D-013 terms, not criteria): vs cuOpt alone (same window) on the fixed 11: improved 7, worse 3, P 0.677 → 0.548
  (−19 %), d +0.235 [+0.052, +0.433], Wilcoxon p 0.049; vs SCIP alone (A0, earlier days): fixed 11 improved 8, worse 0, 0.880 →
  0.548 (−38 %); SCIP's weak set among the 17 (15 instances): improved 12, worse 0, 0.889 → 0.447 (−50 %), d +0.502 [+0.329,
  +0.675], p 0.0005 (10-13's default: −47 %).
- Berk (08:2x, question panel): **"Targeted glass4 check (Recommended)"** → D-032.

## glass4's harm flag read in one window: D-032 result (2026-10-16 08:27–08:52) — DRIFT
Set `d032`, pool 20261016-082749-317710: **12 of 12 completed**, 0 failed / void; glass4 × seeds 0–3 × {A9@g4old =
`-3way.yaml` (28e5a5e9c725, the 10-13 default), A9@g4yl = `-3way-ysync-ld0.yaml` (f345de0ab936, the candidate), A9@g4ld =
`-3way-ld0.yaml` (341267810eed)}, positions rotated by seed. Record:
[[D-032 glass4's C1 harm flag is read in one window — old default vs yield + lp_decode vs lp_decode, 4 seeds]] § Result.
- Seed means of P(120) over seeds 0–3: old default **0.2176**, yield + lp_decode **0.2233**, lp_decode **0.2177**; d(g4yl) =
  **−0.025 ≥ −0.057 → DRIFT**; d(g4ld) −0.0004. In one window the three configurations are level on glass4; D-030's flag
  (0.216 on 10-13 vs 0.237 on 10-15/16) was day-to-day drift.
- **Consequence, as fixed before the runs:** yield + lp_decode (`configs/itrea2-dev-rc-3way-ysync-ld0.yaml`) is the
  configuration that freezes for phase 1 per D-032's rule (`configs/frozen-v2.yaml` is written on Saturday); D-031's
  four-seed harm reading may still put which configuration freezes to Berk (D-031 § What goes to Berk). D-031 queued and
  started 08:57:43.

## Four-seed result (D-031) — filled in after the run
**PLACEHOLDER (D-031) — nothing below this line is a result; D-031 is running and no number from it exists in this note.**
- Set `c1r-ysync-ld0-s23`, `jobs/d031.jsonl` (sha256 `f83e6888934ac3b2e243be880a8c823ae83b787462982c5d5f59b316dc9a3198`):
  the 17 instances × {A9@c1r, A6F@c1r} × **seeds 2–3** = 68 runs in 34 blocks, the same jobs as D-030's file except seed,
  set, stratum, block and position (D-031 § The runs). Started **2026-10-16 08:57:43** in window W-1016 from the main checkout (journal 2026-10-16 08:52);
  expected to end about 13:30.
- To be read with `itrea build-runs`, then `OMP_NUM_THREADS=1 nice -n 19 taskset -c 10-14 .venv/bin/python
  tools/d031_report.py --d030-json logs/d030-reading.json` — rules fixed before any run in
  [[D-031 The phase-1 interim result rests on four seeds — C1 runs again at seeds 2–3 on the frozen-v2 candidate and is read with D-030's seeds 0–1]].
- To fill in: validity (n of 136); THE line `PHASE-1 INTERIM RESULT` (four-seed HINT ratio and GUARD mean d if the 136 runs
  read VALID, the seeds 2–3-alone ratio beside as the selection check — otherwise D-030's reading, with the reason); the
  four-seed harm list with the replicates' and cuOpt alone's d beside; whether the halves (seeds 0–1, seeds 2–3) agree; the
  D-013 lines vs cuOpt alone and vs SCIP alone; the comparison with the saved D-030 reading; the plain statement that the
  four-seed ratio is slightly optimistic because D-031 ran only after D-030's hint, and that more seeds do not shrink the
  10-13 reference's own error (D-031 § What goes to Berk). The go / no-go on C1 is Berk's.

## Operations in phase 1's last week — what cost box time (journal 2026-10-13 to 2026-10-16; [[Mistakes and incidents]])
- **12-h lease limit.** 10-14 07:05: the scheduled chain claimed `--until 20:15` (13 h 10 min); `box-window` refuses leases
  longer than 12 h, the claim retried until 08:00 and the chain exited with nothing stopped; the unit had no failure alert, so
  the box idled 07:05–14:41 (7 h 36 min). Fix: leases ≤ 12 h, `OnFailure=agent-failure-alert@` on every scheduled chain, the
  chain's output checked within 15 min (itrea-window skill). The 10-15 and 10-16 chains (07:05–18:55, 11 h 50 min) opened on
  schedule.
- **Idle after a restore.** 10-13: the window's chain ended at its restore (20:17) with nothing queued; the box idled 2 h 17 min
  until Berk's "You can continue performance runs" (22:2x). Since 10-15 each window has a queue file the chain drains.
- **R26's opening gate** is not enforced by the pool (W-1013b opened at 52–54 °C; #38); `tools/r26-gate.sh` opens every window
  since 10-14 (10-14 14:45:34 at a 50.0 °C median; 10-15 07:08:04 and 10-16 07:08:10 at 46.0 °C).
- **R26's breaker tripped 10-15 17:30:56** (D-030's pool): after the governor's rest (16:44–17:14) the hottest zone reached
  85–87 °C within ~4 min at maximum fan with the GPU at 15–20 W, the governor capped the clock at 2,200 MHz, 2 runs were voided
  and the pool halted. Another session's CPU-heavy work ran unpinned and without a lease inside W-1014 and W-1015 (load averages
  5.7–6.0 at pool starts) and is the likely contributor; since 10-15 19:0x its jobs run on cores 10–14 at nice 19 (our runs use
  cores 1–3 and 5–6). Both arms of a block share their window, so comparisons stay paired; noise rises.
- **The fan governor's rests** (30 min each, about every 1.5–2 h under load) take about 25 % of window time (journal 10-15
  13:4x–14:4x); every run-time estimate now includes them (D-031: 3.83 min per 120-s run).
- **Staging.** 10-15 13:43: `git add vault` swept D-030 into a commit while a build agent was still writing it (placeholders);
  no run read it and the final text (d50c03c) was committed before any C1 re-read run. Lesson: stage named files, not
  `vault/`, while an agent edits the vault.
- Day totals of W-1015 (07:05–18:04): 153 completed, 1 failed, 2 void over 8 pools (journal 2026-10-15 17:30).
