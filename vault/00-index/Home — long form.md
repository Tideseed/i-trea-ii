# Home — long form (texts moved from the front page on 2026-10-16)

**The long texts of the old front page, moved here verbatim so nothing is lost.** On 2026-10-16 [[Home]] became a short
front page (Berk, 2026-10-16: "In all text prioritize readability for humans, and comprehension for agents").
- **This is a snapshot.** Its statuses are as of 2026-10-16 (~09:00) and are **not** kept current here.
- **Since this snapshot (2026-10-16 afternoon):** D-031 was read at 12:55 — four seeds, 136 runs VALID: hint 0.798,
  guard +0.045, the phase-1 interim result. Berk's C1 go at 13:1x closed phase 1 and wrote `configs/frozen-v2.yaml`
  ([[D-033 C1 go — phase 1 closes with configs-frozen-v2.yaml = yield + lp_decode]]). Lines below that say `frozen-v2.yaml` is written on Saturday, that D-031
  is running or that its harm reading may still put the choice to Berk are superseded; the current state is in [[Home]]
  and `AGENTS.md`.
- **The living sources** are the decision records (`50-decisions/`), the [[Roadmap]] and the [[Experiment index]].
- **Edited 2026-10-17:** session checkpoints, the incident log and procedural journal entries moved to the private
  operations repository; the rows and links below that pointed to them now say so.
- **Other snapshots of the same day:** [[Project background]] (the old README) and
  [[Agent companion — detail and history]] (the old `AGENTS.md` detail). `AGENTS.md` and [[Home]] hold the current state.

| section below | what it holds | where Home now has the short form |
|---|---|---|
| Project description | the three ranks, arms A3 / A9, the main claim, the aim, v1, the predecessors | Home, first lines and "Terms" |
| State as of 2026-10-13 | the front page's state paragraph before 2026-10-16 — **superseded** (e.g. A9's default has since changed, D-026 / D-029) | Home, "Where it stands (2026-10-16)" |
| Start here — full descriptions | the start-here table with its long descriptions | Home, "Start here" |
| Milestones — full issue lists | the C1 and v1 milestones with every issue, and the C1 / v1 staging paragraph | Home, "Status"; [[Roadmap]] |
| Decisions — full lines | D-001 … D-032 with the full one-line summary and status | Home, "Decisions" (one line each) |
| Results notes — full descriptions | what each `30-results/` note holds | Home, "Results notes" |
| Vault sections — full contents | every section with every note it links | Home, "Vault map" |

## Project description

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
Successor of Berk Orbay's master's research I-TREA (Orbay 2011, METU M.S. thesis, open access,
[hdl:11511/20509](https://open.metu.edu.tr/handle/11511/20509) — the starting point and the idea of keeping good
infeasible solutions) and of the optopt study
(`berkorbay/optopt`, whose CPU version of this idea found no gain at 60 s — see [[Research question and spec]]).

## State as of 2026-10-13 (superseded on 2026-10-16 — kept as the record of that day)

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

## Start here — full descriptions

| start here | |
|---|---|
| session checkpoints | kept in the private operations repository (not in this vault) |
| [[Roadmap]] | the aim in Berk's words, v1 and C1, the stages and the GitHub issues of each — **what is next** |
| [[D-019 v1 — deadline and performance target]] | v1: dates, the confirmatory and MIPfeas targets, Berk's answers (addendum); the external standard is [[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]], the staging [[D-024 Targeted first, wider only on a hint — staged testing for development and v1]] |
| [[Development log]] | every development change and result since 2026-10-06, failures included |
| [[Research question and spec]] | Berk's brief and main claim, the predecessor's null result, the three ranks; dated note on what D-013 and D-019 changed |
| [[D-002 Pre-registration — hypotheses, arms, analysis]] | v3.1, binding since `prereg-v1`: primary H1a (A3 < A0, SCIP alone) **and** H1b (A3 < A5, EA alone); secondary H2–H5; mechanism H6. D-013's aim and D-019's targets go into prereg-v2 before any confirmatory run |
| [[I-TREA II algorithm]] | the algorithm, every parameter |
| [[Process interface]] | how a run is assembled: processes, mailbox, barrier, files |
| [[Trace format and metrics]] | every file, event and counter a run writes, the derived tables, every metric |
| [[Experiment index]] | every experiment with its status and outcome, failures included: the pre-registered plan (S1–S2, E0–E6), the development experiments since 2026-10-06 (details in the [[Development log]]) and the planned C1 / v1 work |

## Milestones — full issue lists (GitHub, `Tideseed/i-trea-ii`)
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

## Decisions — full lines
A superseding decision says what changed and why; statuses as of 2026-10-16 (~09:00).

| decision | one line | status |
|---|---|---|
| [[D-001 Language, solvers and process model]] | Python 3.12 + CuPy / NumPy as one code path; SCIP primary partner, HiGHS replication partner, cuOpt control (its set-solution callback disables presolve in 26.08; fixed in 26.10, cuopt#2073); solver and EA as separate pinned processes behind a shared start barrier, exchanging through a mailbox | decided |
| [[D-002 Pre-registration — hypotheses, arms, analysis]] | hypotheses H1a/H1b (primary), H2–H6, arms, pool rules, bounded d, Wilcoxon, bootstrap | v3.1, binding since `prereg-v1`; prereg-v2 owed (D-013, D-019) |
| [[D-003 Resources, memory and thermal rules]] | runner-enforced: lease, memory gate with a 20 GB reserve, 12 GB GPU limit for the EA pool and cuOpt (cuOpt's RMM 24 GB under D-021's harness), pinning, thermal and GPU-health gates | in force; amended by D-008, D-009 |
| [[D-004 Mapping I-TREA (2011) to I-TREA II]] | where the defaults came from | record; narrowed by D-007 |
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
| [[D-017 LP decode — re-optimise the continuous variables of selected candidates]] | a HiGHS worker fixes the integers of a few candidates per generation and solves the LP over the continuous variables (`repair.continuous: lp_decode`) | **ADOPTED** 2026-10-09 as the A3 development default (`itrea2-dev-rc-ld.yaml`, `f030e23909ad`): mean d +0.219 [+0.041, +0.429]; inside A9 since D-026's ADOPT (2026-10-15: lp_decode gated after the EA's LP worker, ld0) |
| [[D-018 A9 — the EA backs off the GPU beside cuOpt (contribution-gated duty cycle)]] | in A9 the EA lowers its GPU duty while it does not contribute (`backoff`, off by default) | **NOT ADOPTED** 2026-10-11 after stage 1 (84 runs): every outcome rule holds (chromaticindex1024-7 d +0.834, P 0.545 → 0.091; mean over the 11 non-floor instances +0.130), mechanism M1, M3, M5 fail (the patience ladder steps the EA down while it leads); `_band` analysis bug fixed before the verdict; follow-up D-023 |
| [[D-019 v1 — deadline and performance target]] | v1 = A9 frozen; C1 Sat 2026-10-17 (moved from Fri 10-23 by the addendum of 2026-10-15); v1 end of November 2026 on cuOpt 26.10; targets on both weak sets and under Mittelmann's MIPfeas protocol | decided by Berk 2026-10-10 (addendum supersedes the proposal's dates, cuOpt version, night windows and external anchor); its external rows read through D-021 (Berk's "Accept all", 10-11), staged by D-024; addendum 2026-10-15 13:4x (Berk: "Assume we are wrapping up phase 1 on Saturday"): C1 moves to Sat 2026-10-17, #26, #35 and CUDA MPS are carried to phase 2 |
| [[D-020 MIPfeas harness — GAMS's clock and integral, paired calibration of cuOpt 26.08]] | GAMS's per-instance MIPfeas archive (downloaded with Berk's yes, 2026-10-11); GAMS's integral ported exactly (`pt_gams`: equal to GAMS's own function on all 3,495 traces of the archive); the call clock (D-019's "from the first launch" corrected); arm A6m (cuOpt as GAMS ran it, whole box) | narrowed by D-021 (A6m's whole-box calibration replaced by A6F at A9's footprint); the archive, the integral and the clocks stay |
| [[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]] | ITX-1: equal footprint F = A9's layout for every GPU arm (A6F, A2F), no whole-box run, call clock deciding with claims on both clocks, 600 s with a stop at the proven optimum, gaps 0 and thread variables unset (harness `ext` / `itx1`), independent hard / weak sets (`datasets/mipfeas_strata.csv`), decisions on the 215 unseen instances, ε-floored guard + solutions found as the minimum, governor-aware thermal gates and voids | decided; Berk accepted every [Berk] item 2026-10-11 ("Accept all"; rests kept; 600 s; calendar deferred). F accepted by the M2 soak (steady p95 66–73 °C, max 76 °C); M3: eilA101-2, nw04, supportcase6 do not fit F; phases 1 and 2 built and merged (1,009 CPU tests on main); M4 complete 2026-10-15, 30 of 30, **OUTSIDE**: R 1.313 [1.091, 1.618] on the call clock (the footprint and the machine; result note § Addendum 2026-10-15); v1 staging amended by D-024 |
| [[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]] | A9 under the external harness (R13 + R14), judged on validity, mechanism and a one-sided no-harm limb on the 3 instances where cuOpt's own 1e-4 stop occurs; the same runs give C1's interim hint (ratio of means ≤ 0.95 on the fixed 11 weak instances) and guard (mean d ≥ −0.05 over the 17) | decided for development; narrowed and amended twice before any reading (2026-10-13); **ADOPTED** 2026-10-13 22:2x (74 runs): validity holds, no harm on the 3, mechanism met only vacuously (SCIP proves first) → the C1 freeze carries harness ext; C1 on the old default: hint present 0.827, guard +0.063 |
| [[D-023 A9 back-off, phase Y only — yield until cuOpt's first incumbent, then full duty]] | the back-off without its patience ladder (`configs/itrea2-dev-rc-3way-boY.yaml`) against the shadow control, D-018's outcome rules | decided for development, amended before any run (2026-10-13 22:5x: 11 instances, 66 runs; clarified 23:3x); **NOT ADOPTED** 2026-10-14 18:1x (backoff2, ext): failed M2, M5, T, G — chromaticindex1024-7 d +0.030, cuOpt's first incumbent inside A9 63.47 / 63.41 s vs 10.87 / 10.52 s alone; the back-off line is closed for v1 |
| [[D-024 Targeted first, wider only on a hint — staged testing for development and v1]] | performance runs first; targeted tests first, wider only on a hint fixed before the runs (futility-only); D-022 / D-023 narrowed; v1 external stage 1 = the panel + 5 no-solution instances, hint = ratio ≤ 0.95, guard stage always, A0 on 81 (only the 25 stage-1 instances if no hint) | development rules decided (Berk's instructions 2026-10-13); v1 staging accepted by Berk 2026-10-13 15:5x ("Yes, stage it"; "Guard-only stage anyway"; "Yes, 81"); its code: the A0-81 writer and D-023's sets are done (b84d192); the v1 stage-1 selector, the stage-1 reading and the cuOpt-version check are still owed (#37) |
| [[D-025 A9 EA blocking sync (R1) — the EA stops spinning on the GPU so its LP worker gets core 6, development test]] | the EA's CUDA context at blocking sync (variant `configs/itrea2-dev-rc-3way-bsync.yaml`, or `-boY-bsync.yaml` if D-023 adopts boY) so its HiGHS LP worker gets core 6; set `r1-bsync`, 5 instances × 2 arms × seeds 0–1 = 20 runs against the same-day default; no-harm on item_placement and comp21; hint = ratio ≤ 0.95 over ca, sing44, neos-960392; fallbacks (a) bnice (renice, no flag), (b) ysync | decided for development 2026-10-14, amended 17:4x before any run; bsync **NOT ADOPTED** 2026-10-15 08:1x (hint present 0.765; generation guard on item_placement / comp21 and comp21's P failed); fallback (b) ysync **NOT ADOPTED** 11:2x (hint present 0.834; item_placement's generation guard failed on one seed) → follow-up D-029 |
| [[D-026 lp_decode inside A9 on the EA's core after the LP worker (R2), development test]] | lp_decode in A9 on the EA's core, gated after the LP worker (#30); set `r2-ld`, 28 runs; hint = ratio ≤ 0.90 over item_placement, sing44 (seeds 0–3); no-harm on ca (0.02), glass4, kumeu (+ item_placement, sing44 by the amendment) | decided for development 2026-10-14, amended 17:4x before any run; **ADOPT** 2026-10-15 13:2x (r2-ld, 28 runs): hint 0.758 — item_placement 0.330 → 0.198, sing44 0.473 → 0.411, lower on all four seeds of each; every no-harm leg holds → `configs/itrea2-dev-rc-3way-ld0.yaml` is A9's and A3x's development default |
| [[D-027 Same-day diagnostic of cuOpt's thread count and SCIP's presence (R3) — reading rules fixed before the runs]] | A6@ext (3 threads) / A6F@ext / A2@ext on 3 / 2 / 3 instances, 16 runs (amended 17:5x before any run: only what a rule reads); three reading rules fixed before the runs (thread effect; comp21's stall: SCIP or the EA's CUDA context; SCIP's presence) | decided for development 2026-10-14 (diagnostic, no adoption), amended before any run (22 runs with rule 4); **RESULT** 2026-10-15 09:2x: rule 4 EA PRESENCE on chromaticindex1024-7 (cuOpt's first incumbent 6.53–9.80 s without the EA vs 62–66 s inside A9); rule 1 closed (no thread effect: the layout line closed for v1); rule 3 SCIP ruled out; rule 2 not read. Berk 10-15: CUDA MPS "Not for v1" |
| [[D-028 Does D-022's harness (R13 + R14) stop phase Y from restoring cuOpt on chromaticindex — a 4-run diagnostic]] | A9@boY under `r13-off` vs a same-window A6F on chromaticindex1024-7 × seeds 0–1 (4 runs, set `d028`): do R13 + R14 stop phase Y from giving cuOpt its first incumbent back? | diagnostic, decided before any run 2026-10-15; **RESULT** 11:4x: **not R13 + R14** (cuOpt's first incumbent 63.25 / 63.24 s vs L 19.82 s) — D-022 not implicated; chromaticindex1024-7 stays a known v1 loss to cuOpt alone (MPS v1.1) |
| [[D-029 ysync gets one follow-up at seeds 2–5, judged on all seeds 0–5 with the failing seed kept — NOT ADOPTED ends R1 for v1]] | ysync once more (item_placement and comp21 at seeds 2–5, ca at 2–3) against a same-window control, the guards pooled over seeds 0–5 with the failing seed kept (Berk's choice, 10-15); the title states the pre-run framing | decided before any run (amended 12:2x); **ADOPT** 2026-10-15 15:0x (21 runs incl. one unread warm-up): generations 1.020 / 1.152 (≥ 0.90), P +0.016 / +0.015 (≤ +0.05), ca at seeds 2–3 0.675 (≤ 0.95) → yield joins A9's default; the frozen-v2 candidate is `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` |
| [[D-030 C1 is re-read once on the frozen-v2 candidate (ld0) with A6F in the same window — 68 runs under D-022's hint and guard and D-025's harm leg]] | C1 re-read once on the frozen-v2 candidate with A6F in the same window (Berk 10-14: "Re-run C1 on frozen hash"), 68 runs (set `c1r-ysync-ld0`), D-022's hint and guard and D-025's harm leg | decided before any run (amended 14:3x); **RESULT** 2026-10-16 08:2x: hint present 0.809, guard holds +0.050, **harm flag on glass4** (d_i −0.088 vs −0.057; krka at floor); comp21's two A6F runs crashed (cuOpt SIGSEGV), kept as outcomes → Berk → D-032 |
| [[D-031 The phase-1 interim result rests on four seeds — C1 runs again at seeds 2–3 on the frozen-v2 candidate and is read with D-030's seeds 0–1]] | the phase-1 interim result on four seeds: the C1 set at seeds 2–3 on the frozen-v2 candidate (68 runs, set `c1r-ysync-ld0-s23`), read pooled with D-030's seeds 0–1; adopts nothing | decided before any run (2026-10-15 16:1x); ran 2026-10-16 08:57:43–12:55; **RESULT** 12:55 (filled in after the snapshot): VALID 136 / 136, **hint 0.798, guard +0.045** — the phase-1 interim result; harm flags glass4 (drift) and comp21 (edge) beside; Berk's C1 go followed (D-033) |
| [[D-032 glass4's C1 harm flag is read in one window — old default vs yield + lp_decode vs lp_decode, 4 seeds]] | glass4 × seeds 0–3 in one window: old default vs yield + lp_decode vs lp_decode alone (12 runs, set `d032`; Berk: "Targeted glass4 check") | decided before any run (2026-10-16 08:2x); **RESULT** 08:5x: **DRIFT** — seed means 0.2176 / 0.2233 / 0.2177, d −0.025 ≥ −0.057 → yield + lp_decode freezes for phase 1 per D-032's rule (fixed before its runs); D-031's four-seed harm reading may still put which configuration freezes to Berk (D-031 § What goes to Berk); D-031 started |

## Results notes — full descriptions (`30-results/`, descriptive unless a note says otherwise)
| note | what it holds |
|---|---|
| [[Development log]] | every development configuration, run set and reading: baselines, feasibility probe, adaptive strategy, LP stream, repair budget and clock, class probe, A9 pilot, D-016, D-017, D-018 stages 0 and 1; phase 1's last week (2026-10-13 to 10-16): D-022 and C1's first reading, D-023, M6 / A0 on the 81 / M4 complete, D-025 bsync and ysync, D-027, D-028, D-026, D-029, the C1 re-read D-030, D-032, D-031's four-seed result and the C1 go (filled in after the snapshot) and the operations lessons |
| [[2026-10-11 D-021 measurements M1–M3 and the interim calibration]] | the external harness, not the method: M1 smoke passed; M2 soak passed, F accepted (steady p95 66–73 °C, max 76 °C); M3 does-not-fit list (eilA101-2, nw04, supportcase6; decision set 215); M4 calibration interim at 19 of 30 (R 1.390 [1.081, 1.837], superseded), then its **Addendum 2026-10-15 — M4 complete (30 of 30): OUTSIDE** — R 1.313 [1.091, 1.618] on the call clock, feasible 27 vs GAMS 27, late-clock check passed: the footprint and the machine; the checker tolerance question (#36) |
| [[Quantification 2026-10-06 (dev set)]] | A3@lps1 vs SCIP alone: 7 of 8 lower P, mean d +0.43 [+0.23, +0.65], p = 0.016; vs cuOpt alone 6 / 3, +0.11 [−0.33, +0.50] (inconclusive) |
| [[Where cuOpt struggles (optopt background, 2026-10-07)]] | from optopt's runs: cuOpt is worse than SCIP on 16 of 22 continuous-heavy MIPLIB instances (mean d −0.35); 9 of 80 without a solution at 60 s. Background only, never a selection rule |
| [[Where the coupling gains — instance classes (2026-10-07)]] | no pre-run structural rule separates gains from losses on the 9 instances; run-time patterns (cuOpt's early close / slow start / stall; the EA's foothold before SCIP) |
| [[Experiment index]] | every experiment with its status and outcome, failures included: the pre-registered plan (S1–S2, E0–E6), the development experiments since 2026-10-06 and the planned C1 / v1 work |

## Vault sections — full contents
| section | contents |
|---|---|
| `00-index/` | this page, [[Research question and spec]], [[Roadmap]] |
| `10-design/` | [[I-TREA II algorithm]], [[Feasibility jump local search]], [[Process interface]], [[Trace format and metrics]], [[Hard instance pool]], [[Datasets]], [[Compute resources]] (machine, caps and gates), [[Islands (draft)]] |
| `20-journal/` | dated research journal, append-only, times in CEST (newest: [[2026-10-16]]); research developments only since 2026-10-17 |
| `30-results/` | see Results above |
| `40-literature/` | [[Prior art]] (synthesis of sweeps A–E: 91 works, 78 legally readable) with the sweep notes in `prior-art/` ([[A Constraint handling with infeasible solutions]], [[B Metaheuristics cooperating with MIP solvers]], [[C GPU primal heuristics and GPU EC]], [[D Benchmarks and metrics]], [[E Novelty check 2026-10-05]]); [[Paywalled research]] (the only home for paywalled works); part I: the open 2011 M.S. thesis (`orbay2011itrea`, [hdl:11511/20509](https://open.metu.edu.tr/handle/11511/20509)), summarised in [[E Novelty check 2026-10-05]] |
| `50-decisions/` | D-001 … D-032 — see Decisions above |
| incident log | kept in the private operations repository (not in this vault) |
| `70-upstream/` | drafts and records of upstream reports — **filed only with Berk's per-post yes**; filed so far: NVIDIA/cuopt#2073 (`SetSolutionCallback` with presolve; Berk's yes, 2026-10-07; record `2026-10-07 NVIDIA-cuopt 2073 — SetSolutionCallback with presolve.md`) — fixed by PR #2098 in cuOpt 26.10 and verified on the 26.10 nightly 2026-10-13 (SET callbacks are polled only on GPU-population improvements); two follow-up drafts in `drafts/`, not posted |
| `80-reviews/` | design critique: [[2026-10-05 design critique]] and [[2026-10-05 design critique — response]]; [[2026-10-06 build review]]; consults with Hermes Agent and OpenAI GPT-6.1 Sol: [[2026-10-06 Feasibility targets consult (GPT-6.1 Sol)]], [[2026-10-06 LP stream design consult (GPT-6.1 Sol)]]; Claude Code workflows: [[2026-10-06 Repair-depth design (workflow)]], [[2026-10-07 cuOpt restart design (workflow)]], [[2026-10-08 lp_decode design (workflow)]], [[2026-10-08 lp_decode review (workflow)]], [[2026-10-08 A9 GPU back-off design (workflow)]], [[2026-10-09 A9 GPU back-off review (workflow)]]; [[2026-10-08 cuOpt population and diversity — what transfers]] |
| session checkpoints | kept in the private operations repository (not in this vault) |
