# Roadmap (living; updated 2026-10-13)

**v1 target ([[D-019 v1 — deadline and performance target]], Berk 2026-10-10):** checkpoint C1 **Fri 2026-10-23**
(development frozen on cuOpt 26.08, external harness calibrated, interim A9 result, go / no-go); **v1 end of November**
(aim Fri 11-27, hard end Mon 11-30) on cuOpt 26.10 (on 26.08 if 26.10 is not released by Fri 11-06, with 26.10 as an
addendum — to be confirmed with Berk then; D-019); no night windows. Targets: lift SCIP's weak set (min: significant;
target −40 % mean P) and cuOpt's weak set (min: significant; target −25 %) on unseen hard instances, never worse than either solver on average; externally, at
least our own cuOpt-26.10-alone result on the same GB10 (target −25 % on the hard part). The external benchmark is our
own standard ITX-1 ([[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]],
Berk accepted its readings 2026-10-11), run in stages ([[D-024 Targeted first, wider only on a hint — staged testing for development and v1]],
Berk 2026-10-13). Every target is a plan, not a finding.

**Aim refined (Berk, 2026-10-07, [[D-013 Aim — lift each solver's weakest cases]]):** "help both SCIP, cuOpt (and
Highs if possible) in some of the hardest problems where they are the weakest. Our performance metric is improvement
upon the worst performances, of course with a regard to mean to some degree." → primary = improvement on each solver's
weak set (P ≥ 0.5 alone, chosen on separate seeds), guard = the mean (non-inferiority).

**Aim (Berk, 2026-10-06):** make the **hardest** problems easier — a GPU EA coupled with an exact solver beats either
alone ([[D-007 Main claim — the coupling beats either method alone]]). Being **head to head with cuOpt** (NVIDIA's GPU
heuristics + B&B) is the bar; a split result against it is a strong position, not a setback. **Time to first solution
is not a primary objective** — "we don't expect to solve simple problems a few seconds faster at all"; the primal
integral on hard instances is. **Practice first, then scale:** develop on the small set until the algorithm settles,
then go to more seeds and the larger problem set.

**How box time is spent (Berk, 2026-10-13; D-024, AGENTS.md):** "Shall we do targeted and surgical tests? and only if we
see hints of improvement we can go wider to test the methodology's limits." — a hint is one threshold stated before the
runs, and it only gates whether a line widens. "We shall also always aim for performance runs unless something
structural is buggy in our experimentation methodology." — calibration and harness runs only for a structural bug or to
fill idle hours. "If the box is idle make the most of it."

Each stage lists its GitHub issues (`Tideseed/i-trea-ii`). Newest checkpoint: [[2026-10-13-1600]].

## C1 checkpoint — Fri 2026-10-23 (status 2026-10-13)
| issue | item | status | next |
|---|---|---|---|
| #23 | A9 (SCIP ↔ EA ← cuOpt, D-015) and the EA's GPU back-off | A9 built 10-07, pilot run. Back-off [[D-018 A9 — the EA backs off the GPU beside cuOpt (contribution-gated duty cycle)]] stage 1 **NOT ADOPTED** 10-11 (outcome rules hold — chromaticindex1024-7 P 0.545 → 0.091; mechanism M1 / M3 / M5 fail: the patience ladder steps the EA down while it leads); `_band` analysis bug fixed before the verdict. Phase Y only = [[D-023 A9 back-off, phase Y only — yield until cuOpt's first incumbent, then full duty]]: decided and tooled, **not run**. R13 + R14 (cuOpt gaps 0, thread variables unset) = [[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]]: narrowed to the 3 instances where cuOpt's own 1e-4 stop occurs, amended twice before any reading; **running 10-13** (`jobs/itx-r13.jsonl` 58 + `jobs/itx-r13b.jsonl` 16 runs, all in set `itx-r13`), no result yet | read D-022; D-023 amended before any run (2026-10-13 22:5x: 11 instances, 66 runs, D-024 drafted 54; clarified 23:3x), its code on branch `d023-nine` (merge before any backoff2 run); run it on ext (D-022 adopted) |
| #26 | recombine each new partner incumbent with the whole elite | not built; no decision record yet | its record (test fixed first) after D-023; D-019 lists it as decided by C1 |
| #27 | `lp_decode` (LP over the continuous variables with the integers fixed) | **ADOPTED for A3** ([[D-017 LP decode — re-optimise the continuous variables of selected candidates]], 10-09: mean d +0.219 [+0.041, +0.429], descriptive, 6 development instances × 2 seeds); refused inside A9 (one EA core) | inside A9 only through #30 |
| #30 | A9 core layout so the EA + decoder get ≥ 2 cores | open, no record; F (A9's layout) is now D-021's accepted footprint, so any change to F repeats the M2 soak | D-019 lists lp_decode in A9 for C1; a record first |
| #29 | external harness + calibration on cuOpt 26.08 | [[D-020 MIPfeas harness — GAMS's clock and integral, paired calibration of cuOpt 26.08]] built (GAMS's archive; `pt_gams` equals GAMS's integral on all 3,495 traces), narrowed by **D-021 (ITX-1)**: phases 1 and 2 merged (1,009 CPU tests on main), M0 replay done, M1 smoke 12/12, M2 soak → **F accepted** (steady p95 66–73 °C, max 76 °C), M3 → does-not-fit list eilA101-2, nw04, supportcase6 (decision set 215). **M4 calibration INTERIM, INCOMPLETE (19 of 30)**: late-clock check passed on the 6 of 10 strata-0–1 instances run (median −0.119 s); R = 1.390 [1.081, 1.837] on the call clock at the stop, 19 runs (2.036 on the launch clock; 1.416 [1.087, 1.903] on the 18 read at 14:4x); no footprint reading before 30 of 30; stopped 10-13 by the performance-first rule ([[2026-10-11 D-021 measurements M1–M3 and the interim calibration]]) | the 4 remaining strata-0–1 runs before v1 stage 1; the other 7 when the box is idle; M6 (A0's lane check, CPU) queued for 10-13 evening |
| (D-019) | interim A9 result on the 17 × 2 seeds vs SCIP alone and cuOpt alone; go / no-go | read from D-022's runs: **C1 hint** = ratio of means of P(120) A9 / A6F over the fixed 11 weak development instances ≤ 0.95; **C1 guard** = mean bounded d ≥ −0.05 over all 17; vs SCIP alone from the earlier A0 runs. Fixed before any reading | to Berk with the D-022 reading; the go / no-go is his |
| #36 | checker tolerance vs cuOpt's (found 10-13) | our checker accepted below-optimum incumbents (glass4, mas74); cuOpt rejects the glass4 point; some cuOpt incumbents fail our check (snowy, neos-933966); investigation open, no result revised | one rule for every arm, decided before prereg-v2 |
| #37 | code owed by D-024 | D-023's instance sets parameterised; `jobs/itx-a0.jsonl` as 81 runs; `make-jobs itx-v1`'s stage-1 selector; `itx-report`'s stage-1 reading; the pool's cuOpt-version check | before D-023's runs, the A0 runs and v1 stage 1 respectively |
| #33 | 13 completed A3@lps1w control runs moved aside by run-id reuse (infra) | the cross-set case is fixed (14c4832, merged 2a18819: a later run of the same id in another set becomes `<run_id>#<set>`); the same id twice inside one set is still refused and open | before any pool reruns a job inside one set |
| #32 (v1 milestone) | the C1 freeze and prereg-v2 | `configs/frozen-v2.yaml` = the C1 freeze (A9 as it stands after C1; D-021 R33); prereg-v2 is tagged before any external A9 run — the pool refuses external A9 jobs without both (R33) | `frozen-v2.yaml` at C1; prereg-v2's date is open: D-019's addendum places the freeze and prereg-v2 after the 26.10 re-baseline (its first proposal had them on 10-23) |

## v1 — end of November 2026 (plan as of 2026-10-13)
- **cuOpt 26.10 (#31).** Our feature request cuopt#2073 is fixed by PR #2098 in 26.10 (Berk 10-13: "I just got word it is
  integrated to 26.10"); verified on the 26.10 nightly (presolve stays on with a SET callback; injections accepted;
  invalid offers rejected; SET is polled only when cuOpt's GPU population improves). Not released at 10-13 17:1x (latest
  v26.08.00). On release: separate venv, API check, A6F re-baselined on the 17 instances. A9 two-way through the SET
  callback is a new mechanism (#35), not started; it needs its own record and test and is not part of the C1 freeze.
- **External run = ITX-1 (D-021).** Equal footprint F = A9's layout for every GPU arm (A6F = cuOpt alone on F), no
  whole-box run, call clock deciding with claims on both clocks, T = 600 s with a stop at the proven optimum (R12),
  independent hard / weak sets (`datasets/mipfeas_strata.csv`), decisions on the 215 unseen instances, ε-floored guard +
  solutions found as the minimum, the shift-0.001 geo-mean reported but not decisive. Berk's answers (10-11): "Accept
  all"; fan rests "Keep rests as now"; "Keep 600 s"; calendar: **"We'll cross that bridge when we come to it."** — no
  near-daily schedule or 26.08 insurance runs are planned; the question returns when 26.10's date (or Fri 11-06) is
  known.
- **Staged (D-024, Berk 2026-10-13: "Yes, stage it"; "Guard-only stage anyway"; "Yes, 81"):**
  1. **Stage 1:** the 20-instance panel + the 5 H_cu-unseen instances where cuOpt finds no solution × {A9, A6F} at seed
     2000 under `itx1`, plus A0 on them (≈ 6.1–6.4 h + ≈ 1.7 h of GPU runs).
  2. **Hint** (one registered futility-only look): ratio of means of call-clock `pt_gams` A9 / A6F over the 25 ≤ 0.95.
  3. **After a hint:** the rest of T1 and T2 (≈ 22–24 h of GPU runs; the whole core including stage 1 ≈ 30–32 h); target, guard and
     solutions found read as D-021 R19–R21 on every instance; T3–T5 only if box time remains; D-019's literal readings
     and the 233-wide geo-mean only in that limits stage.
  4. **Always:** the guard stage (T2's E-sample + W-only, ≈ 8.5–9.4 h), so v1's minimum is evaluated whatever the hint.
  5. **A0** (SCIP alone, CPU windows) on the 81 instances of H_cu ∪ W_SCIP unseen, stage-1 instances first — only the 25
     stage-1 instances if there is no hint (D-024 (f)).
  6. One frozen configuration, cuOpt version and harness hash across stages, else stage 1's pairs are not pooled.
- **Confirmatory test (part A, #32):** prereg-v2 written and tagged before any run; test-set size still open (16 vs 32,
  #11); a futility-only first stage on a fixed subsample in the same pattern, written into prereg-v2.
- **Code owed for v1 (D-024, #37):** `itx-report`'s stage-1 reading; `make-jobs itx-v1`'s stage-1 selector;
  `jobs/itx-a0.jsonl` as 81 runs; the checker rule settled (#36) before prereg-v2. Then #1 (screening stage 2, pool freeze `pool-v1`) and #12 (paper).

## 1. Development on the 17 instances (9 development + 8 class probe) — until C1 ([[D-010 Develop the algorithm first on a small development set]], [[D-011 Solver streams good infeasible solutions to the EA]])
Baseline: A3@lps1 (adaptive repair strategy + the EA's own HiGHS LP + LP stream + deep repair) — vs SCIP alone 7 of 8
better, mean d +0.43 [+0.23, +0.65]; vs cuOpt alone 6 of 9 better ([[Quantification 2026-10-06 (dev set)]]; descriptive).
- #14 repair clock — adaptive repair depth: active since 2026-10-07 (D-012 addendum 2), part of every development default
- #15 fhnw-binpack4-48 — deep repair + two-scale survival
- #17 radiationm18-12-05 — a non-local operator for the min-max objective
- #18 LP stream follow-ups; #19 owed one-variable runs; #20 pool guard (infra)
- #16 islands — scale demes, then region islands from SCIP's open nodes (Berk's "large chunks"; draft
  [[Islands (draft)]])

- D-013 follow-ups: HiGHS alone (H0) and HiGHS ↔ EA (H3) on the development set (#22).
- D-015: three-way arm A9 (SCIP ↔ EA ← cuOpt) — use cuOpt's feasible solutions (#23); back-off D-018 not adopted,
  D-023 (phase Y only) next; R13 + R14 under the external harness D-022 (running 10-13). With cuOpt 26.10: A9 two-way
  through cuOpt's SET callback (#35).
- #25 elite reserve — Rank 1 by cuOpt's clearing rule: built, NOT adopted after seeds 0–3 (D-016; issue closed). Then
  partner-incumbent recombination bursts in A9 (#26, after D-023) and `lp_decode` (D-017: ADOPTED for A3; in A9 pending
  the core layout, #30) ([[2026-10-08 cuOpt population and diversity — what transfers]]).
- D-012: the repair clock (active) is the development default for repair depth (Stage 2, 2026-10-07).

## 2. Practice → scale
- More seeds on the development set (4 per instance), then the larger problem set: finish the stage-2 screening and
  freeze the pool (#1), frozen references (#2), calibration (#3), runner gates (#4, #5, #13).

## 3. Freeze the configuration
- For v1: A9 frozen at C1 as `configs/frozen-v2.yaml`, registered with prereg-v2 (D-021 R33; D-019's addendum puts
  prereg-v2 after the 26.10 re-baseline).
- E4 tuning within its cap (#7) → the frozen configuration of the pre-registered plan (D-002).

## 4. Confirmatory
- E2/E3 on unseen instances (#8); test size as proposed in D-010 (16), open in #11; for v1 as prereg-v2 fixes it (#32).

## 5. After tuning
- #21 **Laya** decision model: offline test on recorded decision states (repair clock, strategy, islands) against the
  measured rules and LightGBM; live only at a low duty cycle if it wins offline; the decision maker in E6.
- E5 long runs and open instances (#9); E6 multi-objective — I-TREA's interactive decision maker (#10).

## 6. Paper (#12)
