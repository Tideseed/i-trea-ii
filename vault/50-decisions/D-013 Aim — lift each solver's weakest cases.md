# D-013 Aim — lift each solver's weakest cases (2026-10-07 11:1x, Berk)

**Berk:** "I don't aim our coupling to be generally better but to help both SCIP, cuOpt (and Highs if possible) in some
of the hardest problems where they are the weakest. Our performance metric is improvement upon the worst performances,
of course with a regard to mean to some degree."
Earlier the same week: "make hardest problems easier"; "being better in a subset of problems is still a strong win";
time to first solution is not a primary objective ([[Roadmap]]).

**Status:** the aim is Berk's (decided). The metric below is my proposal to make it measurable — adopted for development
reporting now; it amends [[D-007 Main claim — the coupling beats either method alone]] and must be carried into a
revised pre-registration (prereg-v2) **before** any confirmatory run (E2/E3) — none has run.

## The metric (proposal)
For each baseline solver B ∈ {SCIP, cuOpt, HiGHS}:
- **Weak set W_B** — the instances where B alone is weak: P_B(T) ≥ 0.5 (no solution, or a poor one for much of the
  budget) measured on **screening/reference seeds that are not used for the evaluation** (selecting on the evaluation
  seeds would inflate the improvement by regression to the mean: B's unlucky seeds would define its weak set).
  Sensitivity check: B's worst third by P_B(T).
- **Primary — improvement on W_B:** "B coupled with the EA" against B alone on W_B: share of W_B improved; mean P
  reduction; mean bounded d with the bootstrap interval and the Wilcoxon test (D-002 machinery); and the worst case
  after coupling (max P over W_B).
- **Guard — the mean:** over all instances, B + EA must not be worse than B alone by more than the non-inferiority
  margin (mean d ≥ −0.05, D-002 §6).
- **Couplings:** SCIP ↔ EA (arm A3, built); HiGHS ↔ EA (arm H3, built — no development runs yet); **cuOpt + EA: to
  design** — cuOpt cannot take solutions mid-run (its set-solution callback disables presolve, D-001), so candidates
  are (i) cuOpt ‖ EA with cuOpt's incumbents streamed one way into the EA and the portfolio scored, (ii) warm starts
  of cuOpt from the EA at (re)start if the API allows, (iii) SCIP ↔ EA ← cuOpt (three-way).

## Where we stand (development set, 9 instances, 2 seeds; descriptive; weak sets here from the same seeds — the
caveat above applies, though SCIP's weak-set P values are mostly 1.0 on both seeds)
| baseline B | W_B (P_B ≥ 0.5) | coupled (SCIP ↔ EA, lps1) improves | mean P on W_B | mean d on W_B | worst after | guard (all) |
|---|---|---|---|---|---|---|
| SCIP alone | 7 of 9 | **6 of 7** | 0.88 → **0.47** (−47 %) | +0.44 | rail01 1.0 | +0.43 [+0.23, +0.65] |
| cuOpt alone (vs our SCIP ↔ EA, not yet cuOpt + EA) | 6 of 9 | **5 of 6** | 0.72 → **0.44** (−39 %) | +0.43 | rail01 1.0 (cuOpt 0.93) | +0.11 [−0.33, +0.50] |
| HiGHS alone | — | — | — | — | — | no runs yet |
(lp1 — the EA's own LP only — gives SCIP 5 of 7, −42 %; cuOpt 5 of 6, −39 %.)

## Consequences
- Development reports lead with the weak-set numbers per solver, then the guard.
- Next measurements: HiGHS alone (H0) and HiGHS ↔ EA (H3) on the development set; a cuOpt + EA coupling design (issue).
- The test-set selection (D-010 §4) must contain each solver's weak instances in sufficient number — selected by the
  pre-registered rule on screening seeds, never on optopt's background table.
- Background: cuOpt's weak spots (continuous-heavy MIPs, small instances, 9/80 MIPLIB without a solution at 60 s) —
  [[Where cuOpt struggles (optopt background, 2026-10-07)]].
