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

**State (2026-10-10):** development on the 9-instance development set
([[D-010 Develop the algorithm first on a small development set]]) and the 8 class-probe instances, judged on the
coupled arm at 120 s ([[D-011 Solver streams good infeasible solutions to the EA]]); **no confirmatory experiment has
run** and every development number is descriptive. Development defaults: A3 = `configs/itrea2-dev-rc-ld.yaml`
(`f030e23909ad`, with LP decode, D-017); A9 / A3x = `configs/itrea2-dev-rc-3way.yaml` (`1d12ad490dd0`). Next: D-018
stage 1 (A9's GPU back-off, 84 runs in two daytime windows), then the C1 items below. Daytime windows only (D-019:
"No nights").

| start here | |
|---|---|
| `90-checkpoints/` | session checkpoints — **read the newest first** ([[2026-10-09-1925]]; the day after it: journal [[2026-10-10]]) |
| [[Roadmap]] | the aim in Berk's words, v1 and C1, the stages and the GitHub issues of each — **what is next** |
| [[D-019 v1 — deadline and performance target]] | v1: dates, the confirmatory and MIPfeas targets, Berk's answers (addendum) |
| [[Development log]] | every development change and result since 2026-10-06, failures included |
| [[Research question and spec]] | Berk's brief and main claim, the predecessor's null result, the three ranks; dated note on what D-013 and D-019 changed |
| [[D-002 Pre-registration — hypotheses, arms, analysis]] | v3.1, binding since `prereg-v1`: primary H1a (A3 < A0, SCIP alone) **and** H1b (A3 < A5, EA alone); secondary H2–H5; mechanism H6. D-013's aim and D-019's targets go into prereg-v2 before any confirmatory run |
| [[I-TREA II algorithm]] | the algorithm, every parameter |
| [[Process interface]] | how a run is assembled: processes, mailbox, barrier, files |
| [[Trace format and metrics]] | every file, event and counter a run writes, the derived tables, every metric |
| [[Experiment index]] | every experiment with its status and outcome, failures included: the pre-registered plan (S1–S2, E0–E6), the development experiments since 2026-10-06 (details in the [[Development log]]) and the planned C1 / v1 work; state 2026-10-10 |

## Milestones (GitHub, `Tideseed/i-trea-ii`)
| milestone | due | issues |
|---|---|---|
| **C1 checkpoint (2026-10-23)** | Fri 2026-10-23 | #23 A9 (cuOpt coupled; back-off stage 1), #26 recombine each partner incumbent with the elite, #27 lp_decode (adopted for A3; open for A9 and follow-ups), #29 MIPfeas harness + calibration on cuOpt 26.08, #30 A9 core layout (LP decode in A9) |
| **v1 (end of November 2026)** | aim Fri 11-27, hard end Mon 11-30 | #1 screening and pool freeze (`pool-v1`), #12 paper, #31 cuOpt 26.10 (venv, API check, re-baseline), #32 prereg-v2, confirmatory and MIPfeas runs |

C1 closes development on cuOpt 26.08 with an interim A9 result on the 17 development and class-probe instances × 2 seeds
against SCIP alone and cuOpt alone, and a go / no-go for the final phase. If 26.10 is not released by Fri 11-06, the v1
runs proceed on 26.08 and 26.10 follows as an addendum. The MIPfeas runs need about 78 h of run time (8–10 long daytime
windows with both LLM servers stopped).

## Decisions
A superseding decision says what changed and why; statuses as of 2026-10-10.

| decision | one line | status |
|---|---|---|
| [[D-001 Language, solvers and process model]] | Python 3.12 + CuPy / NumPy as one code path; SCIP primary partner, HiGHS replication partner, cuOpt control (its set-solution callback disables presolve); solver and EA as separate pinned processes behind a shared start barrier, exchanging through a mailbox | decided |
| [[D-002 Pre-registration — hypotheses, arms, analysis]] | hypotheses H1a/H1b (primary), H2–H6, arms, pool rules, bounded d, Wilcoxon, bootstrap | v3.1, binding since `prereg-v1`; prereg-v2 owed (D-013, D-019) |
| [[D-003 Resources, memory and thermal rules]] | runner-enforced: lease, memory gate with a 20 GB reserve, 12 GB GPU limit for the EA pool and cuOpt, pinning, thermal and GPU-health gates | in force; amended by D-008, D-009 |
| [[D-004 Mapping I-TREA (2015) to I-TREA II]] | where the defaults came from | record; narrowed by D-007 |
| [[D-005 The frontier of diverse solutions]] | single objective first, plus a frontier archive of diverse near-optimal solutions | decided (Berk); exploratory metrics only |
| [[D-006 Exchange platform — Redis, file mailbox as fallback]] | Redis as the common platform, the file mailbox with the same API as fallback | decided; the confirmatory transport is set by the E0b overhead measurement (not run; runs so far use the file mailbox) |
| [[D-007 Main claim — the coupling beats either method alone]] | "a GPU-based evolutionary algorithm metaheuristic coupled with an exact solver can beat either single solution" | Berk; amended by D-013 |
| [[D-008 Thermal start gate by arm type]] | GPU arms start below 80 °C, CPU-only arms below 88 °C; pause at 92 °C | decided |
| [[D-009 Honour the fan governor's rest breaks and the night rest]] | the runner honours the fan governor's rest flag; no window through the night rest | decided; Berk answered the night question 2026-10-10: "No nights" (D-019) |
| [[D-010 Develop the algorithm first on a small development set]] | 9 development instances (`datasets/dev_set.csv`), permanently out of the test set; 120 s, ≥ 2 seeds | decided (Berk); its test-set size of 16 is a proposal (issue #11) — D-019 and #32 plan about 32 |
| [[D-011 Solver streams good infeasible solutions to the EA]] | SCIP streams good infeasible node-LP points (LP stream); development is judged on the coupled arm A3, the EA alone (A5) is a diagnostic | decided (Berk); built |
| [[D-012 Repair clock — adaptive repair depth, staged]] | a per-generation repair-depth controller (`repair_clock`) | development default since Stage 2 (2026-10-07), with two stated deviations |
| [[D-013 Aim — lift each solver's weakest cases]] | primary = improvement on each solver's weak set (P ≥ 0.5 alone, separate seeds); guard = mean d ≥ −0.05 | aim decided (Berk); metric used for development reports, into prereg-v2 |
| [[D-014 Class probe — predictions fixed before the runs]] | 8 new public instances, predictions fixed before the runs | INCONCLUSIVE (3 of 5); out of sample far weaker: vs cuOpt alone 1 better / 5 worse, mean d −0.51 [−0.84, −0.12] |
| [[D-015 Use cuOpt's feasible solutions — three-way arm A9]] | A9 = SCIP ↔ EA ← cuOpt; matched control A3x | built 2026-10-07; pilot run (7 instances, seed 0) |
| [[D-016 Rank 1 as a distinct elite — cuOpt's clearing rule, development test]] | Rank 1 chosen by cuOpt's clearing rule (`ranks.rank1_mode`, off by default) | **NOT ADOPTED** 2026-10-09: mean d −0.033 [−0.122, +0.050] (seeds 0–3); #25 closed |
| [[D-017 LP decode — re-optimise the continuous variables of selected candidates]] | a HiGHS worker fixes the integers of a few candidates per generation and solves the LP over the continuous variables (`repair.continuous: lp_decode`) | **ADOPTED** 2026-10-09 as the A3 development default (`itrea2-dev-rc-ld.yaml`, `f030e23909ad`): mean d +0.219 [+0.041, +0.429]; refused by the runner in A9 / A3x (#30) |
| [[D-018 A9 — the EA backs off the GPU beside cuOpt (contribution-gated duty cycle)]] | in A9 the EA lowers its GPU duty while it does not contribute (`backoff`, off by default) | built; stage 0 complete 2026-10-09: PROCEED (duty ½ before the first incumbent, floor ⅛, yield 30 s, patiences 40 / 40; configs `bo1` / `bo1s`); **stage 1 owed** |
| [[D-019 v1 — deadline and performance target]] | v1 = A9 frozen; C1 Fri 2026-10-23; v1 end of November 2026 on cuOpt 26.10; targets on both weak sets and under Mittelmann's MIPfeas protocol | decided by Berk 2026-10-10 (addendum supersedes the proposal's dates, cuOpt version, night windows and external anchor) |

## Results (`30-results/`, descriptive unless a note says otherwise)
| note | what it holds |
|---|---|
| [[Development log]] | every development configuration, run set and reading: baselines, feasibility probe, adaptive strategy, LP stream, repair budget and clock, class probe, A9 pilot, D-016, D-017, D-018 stage 0 |
| [[Quantification 2026-10-06 (dev set)]] | A3@lps1 vs SCIP alone: 7 of 8 lower P, mean d +0.43 [+0.23, +0.65], p = 0.016; vs cuOpt alone 6 / 3, +0.11 [−0.33, +0.50] (inconclusive) |
| [[Where cuOpt struggles (optopt background, 2026-10-07)]] | from optopt's runs: cuOpt is worse than SCIP on 16 of 22 continuous-heavy MIPLIB instances (mean d −0.35); 9 of 80 without a solution at 60 s. Background only, never a selection rule |
| [[Where the coupling gains — instance classes (2026-10-07)]] | no pre-run structural rule separates gains from losses on the 9 instances; run-time patterns (cuOpt's early close / slow start / stall; the EA's foothold before SCIP) |
| [[Experiment index]] | every experiment with its status and outcome, failures included: the pre-registered plan (S1–S2, E0–E6), the development experiments since 2026-10-06 and the planned C1 / v1 work; state 2026-10-10 |

## Sections
| section | contents |
|---|---|
| `00-index/` | this page, [[Research question and spec]], [[Roadmap]] |
| `10-design/` | [[I-TREA II algorithm]], [[Feasibility jump local search]], [[Process interface]], [[Trace format and metrics]], [[Hard instance pool]], [[Datasets]], [[Compute resources]] (with the window log), [[Islands (draft)]] |
| `20-journal/` | dated research journal, append-only, times in CEST (newest: [[2026-10-10]]) |
| `30-results/` | see Results above |
| `40-literature/` | [[Prior art]] (synthesis of sweeps A–E: 91 works, 78 legally readable) with the sweep notes in `prior-art/` ([[A Constraint handling with infeasible solutions]], [[B Metaheuristics cooperating with MIP solvers]], [[C GPU primal heuristics and GPU EC]], [[D Benchmarks and metrics]], [[E Novelty check 2026-10-05]]); [[Paywalled research]] (the only home for paywalled works); part I: the 2011 thesis (https://open.metu.edu.tr/handle/11511/20509) (unpublished; the open source is the 2011 thesis, `orbay2011itrea`), the part-I references and `I-TREA (2015) references.bib` (an audit record, not citable as a whole) |
| `50-decisions/` | D-001 … D-019 — see Decisions above |
| `60-incidents/` | [[Mistakes and incidents]] |
| `70-upstream/` | drafts and records of upstream reports — **filed only with Berk's per-post yes**; filed so far: NVIDIA/cuopt#2073 (`SetSolutionCallback` with presolve; Berk's yes, 2026-10-07; record `2026-10-07 NVIDIA-cuopt#2073 — SetSolutionCallback with presolve.md`) |
| `80-reviews/` | design critique: [[2026-10-05 design critique]] and [[2026-10-05 design critique — response]]; [[2026-10-06 build review]]; consults with Hermes Agent and OpenAI GPT-6.1 Sol: [[2026-10-06 Feasibility targets consult (GPT-6.1 Sol)]], [[2026-10-06 LP stream design consult (GPT-6.1 Sol)]]; Claude Code workflows: [[2026-10-06 Repair-depth design (workflow)]], [[2026-10-07 cuOpt restart design (workflow)]], [[2026-10-08 lp_decode design (workflow)]], [[2026-10-08 lp_decode review (workflow)]], [[2026-10-08 A9 GPU back-off design (workflow)]], [[2026-10-09 A9 GPU back-off review (workflow)]]; [[2026-10-08 cuOpt population and diversity — what transfers]] |
| `90-checkpoints/` | session checkpoints (newest [[2026-10-09-1925]]) |

Tracking: GitHub issues on `Tideseed/i-trea-ii` (labels `phase:E0` … `phase:E6`, `dev`, `infra`, `paper`,
`after-tuning`, `needs-berk`; milestones above). Questions waiting for Berk: the issues labelled
`needs-berk` — #11 (decisions for Berk), #16 (islands), #9 (E5 long runs).

Rules carried from the host (DGX Spark) and from this project's incidents: public data only; no downloads without
Berk's ask; claim `box-window` before any timed run; memory gate with a 20 GB reserve and the CuPy pool limit as the
only GPU cap (12 GB for the EA and for cuOpt; A9 needs about 73 GB, so both LLM servers are stopped); thermal gate and
the fan governor's rests; **no night windows** (D-009, D-019); production serving is restored by the session that
stopped it; **no commits to `src/` or `configs/` while a pool can still launch runs** — each run imports the current
source, so development goes in a git worktree ([[Mistakes and incidents]], 2026-10-06); **before a window that runs
new GPU code, the GPU-marked cold-cache tests pass and reach the new code after `t0`** (practice from the incidents of
2026-10-08 and journal 2026-10-09; the warm-up now runs every crossover and mutation operator on 1-, 2- and 3-child
batches); nothing is filed upstream or submitted (MIPLIB, arXiv) without Berk's explicit yes; no personal data (e-mail
addresses) to outside services; never `pkill -f` / `pgrep -f` from an agent shell.
