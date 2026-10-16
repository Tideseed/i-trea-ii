# I-TREA II vault — home

**I-TREA II (Interactive Three Ranks Evolutionary Algorithm, part II) is a GPU evolutionary algorithm (EA) that runs beside
exact MIP solvers (SCIP, NVIDIA cuOpt) on hard MIP instances and exchanges solutions with them while they run.**
- **Claim under test, a hypothesis and not a finding:** the coupled system beats either method alone
  ([[D-007 Main claim — the coupling beats either method alone|D-007]]).
- **Aim:** lift each solver's weakest cases ([[D-013 Aim — lift each solver's weakest cases|D-013]]).
- **More:** [[Home — long form]] § Project description (the three ranks, the predecessors I-TREA 2011 and optopt) ·
  Berk's brief: [[Research question and spec]].

## Where it stands (2026-10-16)

Every number on this page is development data, interim and descriptive. No confirmatory experiment has run.

- **Phase 1 (development) ends Sat 2026-10-17 at checkpoint C1** (moved from Fri 10-23;
  [[D-019 v1 — deadline and performance target|D-019]] addendum 2026-10-15). The go / no-go is Berk's.
- **Configuration to freeze:** A9 (SCIP ↔ EA ← cuOpt) with two adopted changes: LP re-optimisation of the continuous
  variables (lp_decode, D-026) and a CUDA context that yields its core (yield, D-029). File
  `configs/itrea2-dev-rc-3way-ysync-ld0.yaml`, by D-032's rule; D-031's four-seed harm reading may still put the choice
  to Berk. `configs/frozen-v2.yaml` is not written yet.
- **C1 on that configuration (D-030; 17 instances × 2 seeds, 120 s):** hint 0.809 (ratio of mean P, A9 / cuOpt alone,
  on the fixed 11; needs ≤ 0.95), guard +0.050 (needs ≥ −0.05). Reported beside: vs cuOpt alone on the fixed 11,
  7 improved / 3 worse, mean P −19 %; vs SCIP alone (A0 from earlier days), 8 / 0 and −38 % on the fixed 11, 12 / 0 and
  −50 % on SCIP's 15 weak instances. glass4's harm flag read as drift (D-032).
- **Running:** D-031, the same 68 runs at seeds 2–3 (set `c1r-ysync-ld0-s23`), since 2026-10-16 08:57:43. Its
  four-seed reading becomes the phase-1 interim result. **No number exists yet.**
- **Known loss and next:** chromaticindex1024-7 (the EA's presence slows cuOpt; CUDA MPS is "Not for v1"). After C1:
  cuOpt 26.10, prereg-v2, the staged external runs (ITX-1); v1 at the end of November 2026.

Phase 1 in one note: [[2026-10-17 Phase 1 — development until C1]] (DRAFT until D-031 is read).

## Terms

| term | meaning | record |
|---|---|---|
| P(T) | primal integral, (1/T) ∫₀ᵀ γ(t) dt ∈ [0, 1], from the best checked incumbent held at t; **lower is better**. Development reads P(120) on the call clock | [[Trace format and metrics]] |
| d | bounded d per instance on seed means, (P_ctrl − P_arm) / max(P_ctrl, P_arm, 10⁻³) ∈ [−1, 1]; **+ means the arm is better** | D-002 |
| three ranks | Rank 0 infeasible good (infeasible solutions that beat the incumbent, at most x % of the population) · Rank 1 feasible good (the protected best y % of the feasible) · Rank 2 rejects | [[I-TREA II algorithm]] |
| A0 | SCIP alone | D-002 |
| A3 | SCIP ↔ EA, two-way exchange (the coupled arm of the pre-registration) | D-002 |
| A5 | the EA alone | D-002 |
| A6 / A6F | cuOpt alone / cuOpt alone on footprint F | D-002, D-021 |
| A9 | SCIP ↔ EA ← cuOpt: SCIP and the EA exchange both ways; cuOpt streams its incumbents into the EA, one way | D-015 |
| A3x | A3 on A9's EA footprint — A9's matched control | D-015 |
| the 17 | the 9 development instances (D-010) + the 8 class-probe instances (D-014) | D-010, D-014 |
| fixed 11 | the 11 cuOpt-weak development instances that C1's hint reads | D-022 |
| weak set W_B | instances where solver B alone has P ≥ 0.5, on seeds not used for the evaluation | D-013 |
| hint | ratio of mean P, A9 / A6F, over the fixed 11; present iff ≤ 0.95. **Futility-only:** it gates widening and declares nothing | D-022, D-024 rule 2 |
| guard | mean bounded d over the 17 (in C1: A9 vs A6F); holds iff ≥ −0.05 | D-013, D-022 |
| harm leg / DRIFT | per instance, the new A9 against 10-13's A9@r13x; harm iff d_i < −band_i, band_i = max(0.05, 10-13's seed spread). DRIFT = D-032's reading that glass4's flag is day-to-day variation | D-025 amendment 3, D-030, D-032 |
| C1 | the checkpoint that closes phase 1: an interim A9 result on the 17 × 2 seeds against SCIP alone and cuOpt alone, then Berk's go / no-go | D-019 |
| footprint F | the cores, GPU and per-role memory caps of A9's frozen layout (today X925 5–6, A725 1–3, the GPU); every GPU arm gets F | D-021 R5 |
| ITX-1 | our own external benchmark standard: equal footprint F, call clock deciding, 600 s, decisions on the 215 unseen instances | D-021 |
| harness ext | cuOpt's MIP gaps 0 and its thread variables unset (R13 + R14) | D-022 |
| ld0 / ysync | lp_decode on the EA's core, gated after its LP worker / `host.cuda_sched: yield` on the EA's CUDA context | D-026 / D-029 |

## Start here

1. **Where it stands** (above); agents also read `AGENTS.md` "At a glance" and "Read first" at the repository root.
2. **The newest checkpoint:** [[2026-10-14-2350]] (all checkpoints in `90-checkpoints/`). It predates the results of
   D-025–D-027 and everything from D-028 on, so read it after the state above.
3. [[2026-10-17 Phase 1 — development until C1]] — phase 1's record: the method to freeze, D-010 to D-032, the C1 readings.
4. [[Roadmap]] — the aim in Berk's words, v1 and C1, the stages and their GitHub issues: **what is next**.
5. [[Development log]] — every development change and result since 2026-10-06, failures included.
6. [[Experiment index]] — every experiment with its status and outcome, failures included.
7. Design: [[I-TREA II algorithm]] (every parameter), [[Process interface]] (how a run is assembled),
   [[Trace format and metrics]] (every file, event, counter and metric).
8. Rules for agents: `AGENTS.md` at the repository root; the box and research rules are summarised below.

## Status

| item | state (2026-10-16) | record |
|---|---|---|
| confirmatory experiments | none run; pre-registration v3.1 binding since `prereg-v1`; prereg-v2 owed before any external A9 run | [[D-002 Pre-registration — hypotheses, arms, analysis\|D-002]], #32 |
| A9 configuration to freeze | `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` (EA hash `835c7c5a249c`, A9 run hash under ext `f345de0ab936`) | D-026, D-029, D-032 |
| A3 development default | `configs/itrea2-dev-rc-ld.yaml` (`f030e23909ad`, with LP decode) | D-017 |
| C1, 2 seeds | old default (10-13): hint 0.827, guard +0.063 · candidate (10-16): above, harm flag glass4 → DRIFT. comp21's two crashed cuOpt-alone runs are kept as outcomes (P 0.776 / 0.782, D-030 § Correction); with them excluded the letter reads INCOMPLETE | D-022, D-030, D-032 |
| external standard ITX-1 | built and merged; M4 calibration 30 of 30, **OUTSIDE**: R 1.313 [1.091, 1.618] on the call clock — the footprint and the machine | D-021; [[2026-10-11 D-021 measurements M1–M3 and the interim calibration\|M1–M4 note]] |
| v1 external runs | ITX-1 in stages: stage 1 = the 20-instance panel + 5 instances where cuOpt finds no solution; the rest of the core only after a hint (ratio ≤ 0.95); a guard stage always | D-021, D-024 |
| cuOpt 26.10 | our cuopt#2073 is fixed in 26.10; on release: venv, API check, re-baseline (#31) | [[Roadmap]] |
| v1 | end of November 2026 (aim Fri 11-27, hard end Mon 11-30) on cuOpt 26.10; on 26.08 if 26.10 is not out by Fri 11-06 (to be confirmed with Berk then) | D-019 |
| windows | daytime only (Berk: "No nights") | D-009, D-019 |
| C1 milestone issues | #23, #26 (carried to phase 2), #27, #29, #30, #33, #36, #37; the C1 freeze `configs/frozen-v2.yaml` | [[Roadmap]] (status per issue) |
| v1 milestone issues | #1 pool freeze, #12 paper, #31 cuOpt 26.10, #32 prereg-v2; #35 (two-way through cuOpt's SET callback) is in no milestone | [[Roadmap]] |
| questions for Berk | issues labelled `needs-berk`: #11 (decisions for Berk), #16 (islands), #9 (E5 long runs) | GitHub `Tideseed/i-trea-ii` |

Full milestone lists and the staging paragraph: [[Home — long form]] § Milestones.

## Decisions D-001 to D-032

One line each; the status is as of 2026-10-16. The full lines are in [[Home — long form]] § Decisions; the record itself
is the source. A superseding decision says what changed and why. Titles of D-029 and D-030 state the pre-run framing:
D-029 was ADOPTED; D-030 ran ysync-ld0 (D-030 § Which candidate runs).

| record | decides | outcome |
|---|---|---|
| [[D-001 Language, solvers and process model\|D-001]] | Python 3.12 + CuPy / NumPy; SCIP primary partner, HiGHS replication, cuOpt control; separate pinned processes and a mailbox | decided |
| [[D-002 Pre-registration — hypotheses, arms, analysis\|D-002]] | hypotheses H1a / H1b (primary), H2–H6, arms, pool rules, bounded d, Wilcoxon, bootstrap | v3.1 binding since `prereg-v1`; prereg-v2 owed (D-013, D-019) |
| [[D-003 Resources, memory and thermal rules\|D-003]] | lease, memory gate (20 GB reserve), 12 GB GPU limit, pinning, thermal and GPU-health gates | in force; amended by D-008, D-009 |
| [[D-004 Mapping I-TREA (2015) to I-TREA II\|D-004]] | where the defaults came from | record; narrowed by D-007 |
| [[D-005 The frontier of diverse solutions\|D-005]] | single objective first, plus a frontier archive of diverse near-optimal solutions | decided (Berk); exploratory metrics only |
| [[D-006 Exchange platform — Redis, file mailbox as fallback\|D-006]] | Redis as the platform, the file mailbox as fallback | decided; transport set by E0b (not run); runs so far use the file mailbox |
| [[D-007 Main claim — the coupling beats either method alone\|D-007]] | main claim: the coupling beats either method alone | Berk; amended by D-013 |
| [[D-008 Thermal start gate by arm type\|D-008]] | start GPU arms below 80 °C, CPU-only arms below 86 °C; pause at 92 °C | decided |
| [[D-009 Honour the fan governor's rest breaks and the night rest\|D-009]] | honour the fan governor's rests; no window through the night rest | decided; Berk 10-10: "No nights" |
| [[D-010 Develop the algorithm first on a small development set\|D-010]] | develop on 9 instances, 120 s, ≥ 2 seeds; they stay out of the test set | decided (Berk) |
| [[D-011 Solver streams good infeasible solutions to the EA\|D-011]] | SCIP streams good infeasible node-LP points to the EA (LP stream); judge development on A3 | decided (Berk); built |
| [[D-012 Repair clock — adaptive repair depth, staged\|D-012]] | per-generation repair-depth controller (`repair_clock`) | development default since 2026-10-07 |
| [[D-013 Aim — lift each solver's weakest cases\|D-013]] | aim: each solver's weak set first, the mean as a guard | decided (Berk); into prereg-v2 |
| [[D-014 Class probe — predictions fixed before the runs\|D-014]] | 8 new instances, predictions fixed before the runs | INCONCLUSIVE (3 of 5); vs cuOpt alone d −0.51 [−0.84, −0.12] |
| [[D-015 Use cuOpt's feasible solutions — three-way arm A9\|D-015]] | three-way arm A9; matched control A3x | built 2026-10-07; one-way (cuOpt never receives) |
| [[D-016 Rank 1 as a distinct elite — cuOpt's clearing rule, development test\|D-016]] | Rank 1 by cuOpt's clearing rule | **NOT ADOPTED** 10-09: d −0.033 [−0.122, +0.050] |
| [[D-017 LP decode — re-optimise the continuous variables of selected candidates\|D-017]] | LP decode: re-optimise the continuous variables of a few candidates | **ADOPTED** 10-09 for A3: d +0.219 [+0.041, +0.429]; in A9 via D-026 |
| [[D-018 A9 — the EA backs off the GPU beside cuOpt (contribution-gated duty cycle)\|D-018]] | A9: the EA backs off the GPU while it does not contribute | **NOT ADOPTED** 10-11: mechanism M1, M3, M5 fail |
| [[D-019 v1 — deadline and performance target\|D-019]] | v1's deadline and targets; C1's content | decided (Berk 10-10); C1 moved to Sat 10-17 (addendum 10-15) |
| [[D-020 MIPfeas harness — GAMS's clock and integral, paired calibration of cuOpt 26.08\|D-020]] | GAMS's MIPfeas archive, integral and clock | narrowed by D-021 |
| [[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope\|D-021]] | ITX-1, our external standard at equal footprint F | decided (Berk 10-11: "Accept all"); M4 **OUTSIDE** 10-15: R 1.313 [1.091, 1.618] |
| [[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test\|D-022]] | A9 under harness ext (R13 + R14); C1's hint and guard | **ADOPTED** 10-13; C1 on the old default: hint 0.827, guard +0.063 |
| [[D-023 A9 back-off, phase Y only — yield until cuOpt's first incumbent, then full duty\|D-023]] | back-off phase Y only: yield until cuOpt's first incumbent | **NOT ADOPTED** 10-14 (failed M2, M5, T, G); back-off closed for v1 |
| [[D-024 Targeted first, wider only on a hint — staged testing for development and v1\|D-024]] | targeted tests first, wider only on a hint fixed before the runs | decided (Berk 10-13); code partly owed (#37) |
| [[D-025 A9 EA blocking sync (R1) — the EA stops spinning on the GPU so its LP worker gets core 6, development test\|D-025]] | R1: blocking sync, then fallback ysync, so the LP worker gets core 6 | both **NOT ADOPTED** 10-15 → D-029 |
| [[D-026 lp_decode inside A9 on the EA's core after the LP worker (R2), development test\|D-026]] | R2: lp_decode inside A9, gated after the LP worker (ld0) | **ADOPT** 10-15: hint 0.758 → ld0 joins A9's default |
| [[D-027 Same-day diagnostic of cuOpt's thread count and SCIP's presence (R3) — reading rules fixed before the runs\|D-027]] | R3 diagnostic: cuOpt's thread count, SCIP's presence | **RESULT** 10-15: EA PRESENCE on chromaticindex1024-7; Berk: MPS "Not for v1" |
| [[D-028 Does D-022's harness (R13 + R14) stop phase Y from restoring cuOpt on chromaticindex — a 4-run diagnostic\|D-028]] | do R13 + R14 stop phase Y from helping cuOpt on chromaticindex? | **RESULT** 10-15: not R13 + R14 |
| [[D-029 ysync gets one follow-up at seeds 2–5, judged on all seeds 0–5 with the failing seed kept — NOT ADOPTED ends R1 for v1\|D-029]] | ysync once more at seeds 2–5, judged on seeds 0–5 | **ADOPT** 10-15: yield joins A9's default |
| [[D-030 C1 is re-read once on the frozen-v2 candidate (ld0) with A6F in the same window — 68 runs under D-022's hint and guard and D-025's harm leg\|D-030]] | C1 re-read on the frozen-v2 candidate, A6F in the same window | **RESULT** 10-16: hint 0.809, guard +0.050, harm flag on glass4 → D-032 |
| [[D-031 The phase-1 interim result rests on four seeds — C1 runs again at seeds 2–3 on the frozen-v2 candidate and is read with D-030's seeds 0–1\|D-031]] | C1 at seeds 2–3, read with D-030's seeds 0–1 | **RUNNING** since 10-16 08:57:43; no result yet |
| [[D-032 glass4's C1 harm flag is read in one window — old default vs yield + lp_decode vs lp_decode, 4 seeds\|D-032]] | glass4's harm flag in one window, 4 seeds | **RESULT** 10-16: DRIFT → yield + lp_decode freezes |

## Results notes (`30-results/`; descriptive unless a note says otherwise)

| note | what it holds |
|---|---|
| [[2026-10-17 Phase 1 — development until C1]] | phase 1 in one place: the method to freeze, D-010 to D-032, the C1 readings, known losses, what moves to phase 2. DRAFT until D-031 is read |
| [[Development log]] | every development configuration, run set and reading since 2026-10-06, failures included |
| [[Experiment index]] | every experiment with its status and outcome: the pre-registered plan (S1–S2, E0–E6), development, C1 / v1 |
| [[2026-10-14 C1 diagnosis and improvement candidates]] | why A9 loses on C1 and the candidates for targeted tests (analysis, not a decision) |
| [[2026-10-11 D-021 measurements M1–M3 and the interim calibration]] | the external harness, not the method: M1–M3, and M4 complete (addendum 2026-10-15): **OUTSIDE**, R 1.313 [1.091, 1.618] |
| [[Quantification 2026-10-06 (dev set)]] | A3@lps1 vs SCIP alone: 7 of 8 lower P, mean d +0.43 [+0.23, +0.65], p = 0.016; vs cuOpt alone inconclusive |
| [[Where cuOpt struggles (optopt background, 2026-10-07)]] | background from optopt's runs; never a selection rule |
| [[Where the coupling gains — instance classes (2026-10-07)]] | no pre-run structural rule separates gains from losses on the 9 instances |

Full descriptions: [[Home — long form]] § Results notes.

## Vault map

| section | contents |
|---|---|
| `00-index/` | this page, [[Home — long form]], [[Research question and spec]], [[Roadmap]], [[Project background]] (the detail behind the README), [[Agent companion — detail and history]] (the detail behind `AGENTS.md`) |
| `10-design/` | [[I-TREA II algorithm]], [[Feasibility jump local search]], [[Process interface]], [[Trace format and metrics]], [[Hard instance pool]], [[Datasets]], [[Compute resources]] (with the window log), [[Islands (draft)]] |
| `20-journal/` | dated research journal, append-only, times in CEST (newest: [[2026-10-16]]) |
| `30-results/` | see Results notes above |
| `40-literature/` | [[Prior art]] (91 works, 78 legally readable; sweeps A–E in `prior-art/`); [[Paywalled research]] (the only home for paywalled works); part I: the 2011 thesis (https://open.metu.edu.tr/handle/11511/20509) |
| `50-decisions/` | D-001 … D-032 — see Decisions above |
| `60-incidents/` | [[Mistakes and incidents]] |
| `70-upstream/` | upstream drafts and records — **filed only with Berk's per-post yes**; filed so far: NVIDIA/cuopt#2073 (fixed in cuOpt 26.10) |
| `80-reviews/` | design critiques, consults and Claude Code workflow reviews |
| `90-checkpoints/` | session checkpoints (newest [[2026-10-14-2350]]) |

Every link of every section: [[Home — long form]] § Vault sections.

## Rules for every session

Carried from the host (DGX Spark) and from this project's incidents. The full agent rules are in `AGENTS.md`.

**Data and the outside world**
- Public data only.
- No downloads without Berk's ask.
- Nothing is filed upstream or submitted (MIPLIB, arXiv) without Berk's explicit yes.
- No personal data (e-mail addresses) to outside services.

**The box**
- Claim `box-window` before any timed run.
- Memory gate with a 20 GB reserve; the CuPy pool limit is the only GPU cap (12 GB for the EA and for cuOpt). A9 needs about
  73 GB, 97 GB under D-021's harness with cuOpt's RMM at 24 GB, so both LLM servers are stopped.
- Thermal gate and the fan governor's rests; **no night windows** (D-009, D-019).
- Production serving is restored by the session that stopped it.
- Never `pkill -f` / `pgrep -f` from an agent shell.
- The stop file `logs/pool.stop` stops a pool, and a chain script that tests it after the pool goes on to restore the
  servers. To pause a pool inside a window, kill the chain first by its PID ([[Mistakes and incidents]], 2026-10-13).

**Code and runs**
- **No commits to `src/` or `configs/` while a pool can still launch runs** — each run imports the current source, so
  development goes in a git worktree ([[Mistakes and incidents]], 2026-10-06).
- **Before a window that runs new GPU code, the GPU-marked cold-cache tests pass and reach the new code after `t0`**
  (practice from the incidents of 2026-10-08 and journal 2026-10-09; the warm-up now runs every crossover and mutation
  operator on 1-, 2- and 3-child batches).

**Analysis**
- **A rule or hint is fixed, with its number, before the runs it reads**, and a narrowed test gets its own rule.

## Tracking

GitHub issues on `Tideseed/i-trea-ii`: labels `phase:E0` … `phase:E6`, `dev`, `infra`, `paper`, `after-tuning`,
`needs-berk`; milestones C1 and v1 (Status above). Questions waiting for Berk carry `needs-berk`.
