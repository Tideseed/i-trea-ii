# I-TREA II vault — home

**I-TREA II — Interactive Three Ranks Evolutionary Algorithm, part II.** A GPU-resident evolutionary algorithm with
three ranks (Rank 0 infeasible good — infeasible solutions that beat the incumbent, at most x % of the population ·
Rank 1 feasible good — the protected best y % of the feasible · Rank 2 rejects) running beside exact MIP solvers on hard
MIP instances: SCIP exchanges solutions with it both ways (arm A3), and in the three-way arm A9 (SCIP ↔ EA ← cuOpt,
[[D-015 Use cuOpt's feasible solutions — three-way arm A9]]) cuOpt streams its incumbents into the EA as well.
**Main claim under test** ([[D-007 Main claim — the coupling beats either method alone]]): the coupled system beats
either method alone — a hypothesis, not a finding. **Aim, refined by Berk on 2026-10-07**
([[D-013 Aim — lift each solver's weakest cases]]): lift each solver's weakest cases — measured on each solver's weak
set (P ≥ 0.5 alone, on seeds not used for evaluation), with the mean as a guard. **v1**
([[D-019 v1 — deadline and performance target]]): A9 frozen, checkpoint C1 on Fri 2026-10-23, v1 at the end of
November 2026 on cuOpt 26.10.
Successor of Berk Orbay's master's research I-TREA (Orbay 2011, METU M.S. thesis, open access, hdl:11511/20509; 2015
manuscript — the starting point and the idea of keeping good infeasible solutions) and of the optopt study
(`berkorbay/optopt`, whose CPU version of this idea found no gain at 60 s — see [[Research question and spec]]).

**State (2026-10-13):** development on the 9-instance development set
([[D-010 Develop the algorithm first on a small development set]]) and the 8 class-probe instances, judged on the
coupled arm at 120 s ([[D-011 Solver streams good infeasible solutions to the EA]]); **no confirmatory experiment has
run** and every development number is descriptive. Development defaults: A3 = `configs/itrea2-dev-rc-ld.yaml`
(`f030e23909ad`, with LP decode, D-017); A9 / A3x = `configs/itrea2-dev-rc-3way.yaml` (`1d12ad490dd0`) — A9's GPU
back-off (D-018) was not adopted after stage 1 (2026-10-11). The external benchmark is our own standard ITX-1 (D-021,
accepted by Berk 2026-10-11; built and merged; its calibration M4 INTERIM, INCOMPLETE (19 of 30)); v1 is staged (D-024,
Berk 2026-10-13). Running on 2026-10-13: D-022 (A9 with cuOpt's gaps 0, 58 + 16 runs), which also gives C1's interim A9
hint and guard — no result yet. Daytime windows only (D-019: "No nights"); performance runs first, targeted before wide
(Berk 2026-10-13, D-024).

| start here | |
|---|---|
| `90-checkpoints/` | session checkpoints — **read the newest first** ([[2026-10-13-1600]]) |
| [[Roadmap]] | the aim in Berk's words, v1 and C1, the stages and the GitHub issues of each — **what is next** |
| [[D-019 v1 — deadline and performance target]] | v1: dates, the confirmatory and MIPfeas targets, Berk's answers (addendum); the external standard is [[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]], the staging [[D-024 Targeted first, wider only on a hint — staged testing for development and v1]] |
| [[Development log]] | every development change and result since 2026-10-06, failures included |
| [[Research question and spec]] | Berk's brief and main claim, the predecessor's null result, the three ranks; dated note on what D-013 and D-019 changed |
| [[D-002 Pre-registration — hypotheses, arms, analysis]] | v3.1, binding since `prereg-v1`: primary H1a (A3 < A0, SCIP alone) **and** H1b (A3 < A5, EA alone); secondary H2–H5; mechanism H6. D-013's aim and D-019's targets go into prereg-v2 before any confirmatory run |
| [[I-TREA II algorithm]] | the algorithm, every parameter |
| [[Process interface]] | how a run is assembled: processes, mailbox, barrier, files |
| [[Trace format and metrics]] | every file, event and counter a run writes, the derived tables, every metric |
| [[Experiment index]] | every experiment with its status and outcome, failures included: the pre-registered plan (S1–S2, E0–E6), the development experiments since 2026-10-06 (details in the [[Development log]]) and the planned C1 / v1 work |

## Milestones (GitHub, `Tideseed/i-trea-ii`)
| milestone | due | issues |
|---|---|---|
| **C1 checkpoint (2026-10-23)** | Fri 2026-10-23 | #23 A9 (cuOpt coupled; back-off D-018 not adopted, D-023 phase Y next; D-022 running), #26 recombine each partner incumbent with the elite (not built), #27 lp_decode (adopted for A3; open for A9 and follow-ups), #29 external harness (D-020 → D-021 ITX-1, built; M4 calibration INTERIM, INCOMPLETE (19 of 30)), #30 A9 core layout (LP decode in A9), #33 control runs moved aside by run-id reuse (cross-set case fixed), #36 checker tolerance vs cuOpt (open), #37 code owed by D-024 (D-023's amendment, A0 job file, v1 stage-1 selector and reading); the C1 freeze `configs/frozen-v2.yaml` (D-021 R33). Status per item: [[Roadmap]] |
| **v1 (end of November 2026)** | aim Fri 11-27, hard end Mon 11-30 | #1 screening and pool freeze (`pool-v1`), #12 paper, #31 cuOpt 26.10 (venv, API check, re-baseline; cuopt#2073 is fixed in 26.10), #32 prereg-v2 (tagged before any external A9 run; date open, D-019's addendum puts it after the 26.10 re-baseline), confirmatory and external runs (ITX-1, staged by D-024); not in a milestone: #35 A9 two-way through cuOpt's SET callback (26.10) |

C1 closes development on cuOpt 26.08 with an interim A9 result on the 17 development and class-probe instances × 2 seeds
against SCIP alone and cuOpt alone, and a go / no-go for the final phase. If 26.10 is not released by Fri 11-06, the v1
runs proceed on 26.08 and 26.10 follows as an addendum (to be confirmed with Berk then; D-019). The external runs follow ITX-1 (D-021) in stages (D-024): stage
1 = the 20-instance panel + 5 instances where cuOpt finds no solution, the rest of the core only after a hint (ratio
≤ 0.95), a guard stage always, A0 on 81 instances (only the 25 stage-1 instances if there is no hint, D-024 (f)). The calendar is deferred by Berk ("We'll cross that bridge when we
come to it", 2026-10-11).

## Decisions
A superseding decision says what changed and why; statuses as of 2026-10-13.

| decision | one line | status |
|---|---|---|
| [[D-001 Language, solvers and process model]] | Python 3.12 + CuPy / NumPy as one code path; SCIP primary partner, HiGHS replication partner, cuOpt control (its set-solution callback disables presolve in 26.08; fixed in 26.10, cuopt#2073); solver and EA as separate pinned processes behind a shared start barrier, exchanging through a mailbox | decided |
| [[D-002 Pre-registration — hypotheses, arms, analysis]] | hypotheses H1a/H1b (primary), H2–H6, arms, pool rules, bounded d, Wilcoxon, bootstrap | v3.1, binding since `prereg-v1`; prereg-v2 owed (D-013, D-019) |
| [[D-003 Resources, memory and thermal rules]] | runner-enforced: lease, memory gate with a 20 GB reserve, 12 GB GPU limit for the EA pool and cuOpt (cuOpt's RMM 24 GB under D-021's harness), pinning, thermal and GPU-health gates | in force; amended by D-008, D-009 |
| [[D-004 Mapping I-TREA (2015) to I-TREA II]] | where the defaults came from | record; narrowed by D-007 |
| [[D-005 The frontier of diverse solutions]] | single objective first, plus a frontier archive of diverse near-optimal solutions | decided (Berk); exploratory metrics only |
| [[D-006 Exchange platform — Redis, file mailbox as fallback]] | Redis as the common platform, the file mailbox with the same API as fallback | decided; the confirmatory transport is set by the E0b overhead measurement (not run; runs so far use the file mailbox) |
| [[D-007 Main claim — the coupling beats either method alone]] | "a GPU-based evolutionary algorithm metaheuristic coupled with an exact solver can beat either single solution" | Berk; amended by D-013 |
| [[D-008 Thermal start gate by arm type]] | GPU arms start below 80 °C, CPU-only arms below 86 °C (88 until 2026-10-11; D-021 R24); pause at 92 °C | decided |
| [[D-009 Honour the fan governor's rest breaks and the night rest]] | the runner honours the fan governor's rest flag; no window through the night rest | decided; Berk answered the night question 2026-10-10: "No nights" (D-019) |
| [[D-010 Develop the algorithm first on a small development set]] | 9 development instances (`datasets/dev_set.csv`), permanently out of the test set; 120 s, ≥ 2 seeds | decided (Berk); its test-set size of 16 is a proposal (issue #11) — D-019 and #32 plan about 32 |
| [[D-011 Solver streams good infeasible solutions to the EA]] | SCIP streams good infeasible node-LP points (LP stream); development is judged on the coupled arm A3, the EA alone (A5) is a diagnostic | decided (Berk); built |
| [[D-012 Repair clock — adaptive repair depth, staged]] | a per-generation repair-depth controller (`repair_clock`) | development default since Stage 2 (2026-10-07), with two stated deviations |
| [[D-013 Aim — lift each solver's weakest cases]] | primary = improvement on each solver's weak set (P ≥ 0.5 alone, separate seeds); guard = mean d ≥ −0.05 | aim decided (Berk); metric used for development reports, into prereg-v2 |
| [[D-014 Class probe — predictions fixed before the runs]] | 8 new public instances, predictions fixed before the runs | INCONCLUSIVE (3 of 5); out of sample far weaker: vs cuOpt alone 1 better / 5 worse, mean d −0.51 [−0.84, −0.12] |
| [[D-015 Use cuOpt's feasible solutions — three-way arm A9]] | A9 = SCIP ↔ EA ← cuOpt; matched control A3x | built 2026-10-07; pilot run (7 instances, seed 0); one-way (cuOpt never receives); a two-way variant on cuOpt 26.10 is #35, not started |
| [[D-016 Rank 1 as a distinct elite — cuOpt's clearing rule, development test]] | Rank 1 chosen by cuOpt's clearing rule (`ranks.rank1_mode`, off by default) | **NOT ADOPTED** 2026-10-09: mean d −0.033 [−0.122, +0.050] (seeds 0–3); #25 closed |
| [[D-017 LP decode — re-optimise the continuous variables of selected candidates]] | a HiGHS worker fixes the integers of a few candidates per generation and solves the LP over the continuous variables (`repair.continuous: lp_decode`) | **ADOPTED** 2026-10-09 as the A3 development default (`itrea2-dev-rc-ld.yaml`, `f030e23909ad`): mean d +0.219 [+0.041, +0.429]; refused by the runner in A9 / A3x (#30) |
| [[D-018 A9 — the EA backs off the GPU beside cuOpt (contribution-gated duty cycle)]] | in A9 the EA lowers its GPU duty while it does not contribute (`backoff`, off by default) | **NOT ADOPTED** 2026-10-11 after stage 1 (84 runs): every outcome rule holds (chromaticindex1024-7 d +0.834, P 0.545 → 0.091; mean over the 11 non-floor instances +0.130), mechanism M1, M3, M5 fail (the patience ladder steps the EA down while it leads); `_band` analysis bug fixed before the verdict; follow-up D-023 |
| [[D-019 v1 — deadline and performance target]] | v1 = A9 frozen; C1 Fri 2026-10-23; v1 end of November 2026 on cuOpt 26.10; targets on both weak sets and under Mittelmann's MIPfeas protocol | decided by Berk 2026-10-10 (addendum supersedes the proposal's dates, cuOpt version, night windows and external anchor); its external rows read through D-021 (Berk's "Accept all", 10-11), staged by D-024 |
| [[D-020 MIPfeas harness — GAMS's clock and integral, paired calibration of cuOpt 26.08]] | GAMS's per-instance MIPfeas archive (downloaded with Berk's yes, 2026-10-11); GAMS's integral ported exactly (`pt_gams`: equal to GAMS's own function on all 3,495 traces of the archive); the call clock (D-019's "from the first launch" corrected); arm A6m (cuOpt as GAMS ran it, whole box) | narrowed by D-021 (A6m's whole-box calibration replaced by A6F at A9's footprint); the archive, the integral and the clocks stay |
| [[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]] | ITX-1: equal footprint F = A9's layout for every GPU arm (A6F, A2F), no whole-box run, call clock deciding with claims on both clocks, 600 s with a stop at the proven optimum, gaps 0 and thread variables unset (harness `ext` / `itx1`), independent hard / weak sets (`datasets/mipfeas_strata.csv`), decisions on the 215 unseen instances, ε-floored guard + solutions found as the minimum, governor-aware thermal gates and voids | decided; Berk accepted every [Berk] item 2026-10-11 ("Accept all"; rests kept; 600 s; calendar deferred). F accepted by the M2 soak (steady p95 66–73 °C, max 76 °C); M3: eilA101-2, nw04, supportcase6 do not fit F; phases 1 and 2 built and merged (1,009 CPU tests on main); M4 calibration interim at 19 of 30, INCOMPLETE; v1 staging amended by D-024 |
| [[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]] | A9 under the external harness (R13 + R14), judged on validity, mechanism and a one-sided no-harm limb on the 3 instances where cuOpt's own 1e-4 stop occurs; the same runs give C1's interim hint (ratio of means ≤ 0.95 on the fixed 11 weak instances) and guard (mean d ≥ −0.05 over the 17) | decided for development; narrowed and amended twice before any reading (2026-10-13); runs in progress (58 + 16), **no result** |
| [[D-023 A9 back-off, phase Y only — yield until cuOpt's first incumbent, then full duty]] | the back-off without its patience ladder (`configs/itrea2-dev-rc-3way-boY.yaml`) against the shadow control, D-018's outcome rules | decided for development, **not run**; amended before any run (2026-10-13 22:5x: 11 instances, 66 runs — D-024 drafted 54 on 9; clarified 23:3x: G's noise test, "still beats", the universe); its code on branch `d023-nine`; runs on ext (D-022 adopted) |
| [[D-024 Targeted first, wider only on a hint — staged testing for development and v1]] | performance runs first; targeted tests first, wider only on a hint fixed before the runs (futility-only); D-022 / D-023 narrowed; v1 external stage 1 = the panel + 5 no-solution instances, hint = ratio ≤ 0.95, guard stage always, A0 on 81 (only the 25 stage-1 instances if no hint) | development rules decided (Berk's instructions 2026-10-13); v1 staging accepted by Berk 2026-10-13 15:5x ("Yes, stage it"; "Guard-only stage anyway"; "Yes, 81"); its code is owed |
| [[D-025 A9 EA blocking sync (R1) — the EA stops spinning on the GPU so its LP worker gets core 6, development test]] | the EA's CUDA context at blocking sync (variant `configs/itrea2-dev-rc-3way-bsync.yaml`, or `-boY-bsync.yaml` if D-023 adopts boY) so its HiGHS LP worker gets core 6; set `r1-bsync`, 5 instances × 2 arms × seeds 0–1 = 20 runs against the same-day default; no-harm on item_placement and comp21; hint = ratio ≤ 0.95 over ca, sing44, neos-960392; fallbacks (a) bnice (renice, no flag), (b) ysync | decided for development 2026-10-14, clarified and amended 17:4x before any run (own control label `A9@r1ctl`, P band on the hint instances and generation guard on seed means gate, harm leg at the C1 re-read), **not run**; after D-023's verdict (it sets the control); GPU checks owed before the first timed run |
| [[D-026 lp_decode inside A9 on the EA's core after the LP worker (R2), development test]] | lp_decode in A9 on the EA's core, gated after the LP worker (#30); set `r2-ld`, 28 runs; hint = ratio ≤ 0.90 over item_placement, sing44 (seeds 0–3); no-harm on ca (0.02), glass4, kumeu (+ item_placement, sing44 by the amendment) | decided for development 2026-10-14, clarified and amended 17:4x before any run (labels `A9@r2ld` / `A9@r2ctl`), **not run**; after the R1 line's final outcome |
| [[D-027 Same-day diagnostic of cuOpt's thread count and SCIP's presence (R3) — reading rules fixed before the runs]] | A6@ext (3 threads) / A6F@ext / A2@ext on 3 / 2 / 3 instances, 16 runs (amended 17:5x before any run: only what a rule reads); three reading rules fixed before the runs (thread effect; comp21's stall: SCIP or the EA's CUDA context; SCIP's presence) | decided for development 2026-10-14 (diagnostic, no adoption), **not run**; any window after backoff2 |

## Results (`30-results/`, descriptive unless a note says otherwise)
| note | what it holds |
|---|---|
| [[Development log]] | every development configuration, run set and reading: baselines, feasibility probe, adaptive strategy, LP stream, repair budget and clock, class probe, A9 pilot, D-016, D-017, D-018 stages 0 and 1 |
| [[2026-10-11 D-021 measurements M1–M3 and the interim calibration]] | the external harness, not the method: M1 smoke passed; M2 soak passed, F accepted (steady p95 66–73 °C, max 76 °C); M3 does-not-fit list (eilA101-2, nw04, supportcase6; decision set 215); M4 calibration **INTERIM, INCOMPLETE (19 of 30)** — late-clock check passes on the 6 of 10 strata-0–1 instances run, R 1.390 [1.081, 1.837] on the call clock, no footprint reading before 30 of 30; the checker tolerance question (#36) |
| [[Quantification 2026-10-06 (dev set)]] | A3@lps1 vs SCIP alone: 7 of 8 lower P, mean d +0.43 [+0.23, +0.65], p = 0.016; vs cuOpt alone 6 / 3, +0.11 [−0.33, +0.50] (inconclusive) |
| [[Where cuOpt struggles (optopt background, 2026-10-07)]] | from optopt's runs: cuOpt is worse than SCIP on 16 of 22 continuous-heavy MIPLIB instances (mean d −0.35); 9 of 80 without a solution at 60 s. Background only, never a selection rule |
| [[Where the coupling gains — instance classes (2026-10-07)]] | no pre-run structural rule separates gains from losses on the 9 instances; run-time patterns (cuOpt's early close / slow start / stall; the EA's foothold before SCIP) |
| [[Experiment index]] | every experiment with its status and outcome, failures included: the pre-registered plan (S1–S2, E0–E6), the development experiments since 2026-10-06 and the planned C1 / v1 work |

## Sections
| section | contents |
|---|---|
| `00-index/` | this page, [[Research question and spec]], [[Roadmap]] |
| `10-design/` | [[I-TREA II algorithm]], [[Feasibility jump local search]], [[Process interface]], [[Trace format and metrics]], [[Hard instance pool]], [[Datasets]], [[Compute resources]] (with the window log), [[Islands (draft)]] |
| `20-journal/` | dated research journal, append-only, times in CEST (newest: [[2026-10-13]]) |
| `30-results/` | see Results above |
| `40-literature/` | [[Prior art]] (synthesis of sweeps A–E: 91 works, 78 legally readable) with the sweep notes in `prior-art/` ([[A Constraint handling with infeasible solutions]], [[B Metaheuristics cooperating with MIP solvers]], [[C GPU primal heuristics and GPU EC]], [[D Benchmarks and metrics]], [[E Novelty check 2026-10-05]]); [[Paywalled research]] (the only home for paywalled works); part I: the 2011 thesis (https://open.metu.edu.tr/handle/11511/20509) (unpublished; the open source is the 2011 thesis, `orbay2011itrea`), the part-I references and `I-TREA (2015) references.bib` (an audit record, not citable as a whole) |
| `50-decisions/` | D-001 … D-027 — see Decisions above |
| `60-incidents/` | [[Mistakes and incidents]] |
| `70-upstream/` | drafts and records of upstream reports — **filed only with Berk's per-post yes**; filed so far: NVIDIA/cuopt#2073 (`SetSolutionCallback` with presolve; Berk's yes, 2026-10-07; record `2026-10-07 NVIDIA-cuopt#2073 — SetSolutionCallback with presolve.md`) — fixed by PR #2098 in cuOpt 26.10 and verified on the 26.10 nightly 2026-10-13 (SET callbacks are polled only on GPU-population improvements); two follow-up drafts in `drafts/`, not posted |
| `80-reviews/` | design critique: [[2026-10-05 design critique]] and [[2026-10-05 design critique — response]]; [[2026-10-06 build review]]; consults with Hermes Agent and OpenAI GPT-6.1 Sol: [[2026-10-06 Feasibility targets consult (GPT-6.1 Sol)]], [[2026-10-06 LP stream design consult (GPT-6.1 Sol)]]; Claude Code workflows: [[2026-10-06 Repair-depth design (workflow)]], [[2026-10-07 cuOpt restart design (workflow)]], [[2026-10-08 lp_decode design (workflow)]], [[2026-10-08 lp_decode review (workflow)]], [[2026-10-08 A9 GPU back-off design (workflow)]], [[2026-10-09 A9 GPU back-off review (workflow)]]; [[2026-10-08 cuOpt population and diversity — what transfers]] |
| `90-checkpoints/` | session checkpoints (newest [[2026-10-13-1600]]) |

Tracking: GitHub issues on `Tideseed/i-trea-ii` (labels `phase:E0` … `phase:E6`, `dev`, `infra`, `paper`,
`after-tuning`, `needs-berk`; milestones above). Questions waiting for Berk: the issues labelled
`needs-berk` — #11 (decisions for Berk), #16 (islands), #9 (E5 long runs).

Rules carried from the host (DGX Spark) and from this project's incidents: public data only; no downloads without
Berk's ask; claim `box-window` before any timed run; memory gate with a 20 GB reserve and the CuPy pool limit as the
only GPU cap (12 GB for the EA and for cuOpt; A9 needs about 73 GB, 97 GB under D-021's harness with cuOpt's RMM at 24 GB, so both LLM servers are stopped); thermal gate and
the fan governor's rests; **no night windows** (D-009, D-019); production serving is restored by the session that
stopped it; **no commits to `src/` or `configs/` while a pool can still launch runs** — each run imports the current
source, so development goes in a git worktree ([[Mistakes and incidents]], 2026-10-06); **before a window that runs
new GPU code, the GPU-marked cold-cache tests pass and reach the new code after `t0`** (practice from the incidents of
2026-10-08 and journal 2026-10-09; the warm-up now runs every crossover and mutation operator on 1-, 2- and 3-child
batches); nothing is filed upstream or submitted (MIPLIB, arXiv) without Berk's explicit yes; no personal data (e-mail
addresses) to outside services; never `pkill -f` / `pgrep -f` from an agent shell; **a rule or hint is fixed, with
its number, before the runs it reads** and a narrowed test gets its own rule; the stop file `logs/pool.stop` stops a
pool, and a chain script that tests it after the pool goes on to restore the servers — to pause a pool inside a window,
kill the chain first by its PID ([[Mistakes and incidents]], 2026-10-13).
