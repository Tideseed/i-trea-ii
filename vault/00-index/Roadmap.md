# Roadmap (living; updated 2026-10-06 evening)

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

Each stage lists its GitHub issues (`Tideseed/i-trea-ii`). Status as of 2026-10-06 21:00.

## 1. Development on the 9-instance set — NOW ([[D-010 Develop the algorithm first on a small development set]], [[D-011 Solver streams good infeasible solutions to the EA]])
Baseline: A3@lps1 (adaptive repair strategy + the EA's own HiGHS LP + LP stream + deep repair) — vs SCIP alone 7 of 8
better, mean d +0.43 [+0.23, +0.65]; vs cuOpt alone 6 of 9 better ([[Quantification 2026-10-06 (dev set)]]).
- #14 repair clock — adaptive repair depth (Stage 0/1 being built; shadow measurement next window)
- #15 fhnw-binpack4-48 — deep repair + two-scale survival
- #17 radiationm18-12-05 — a non-local operator for the min-max objective
- #18 LP stream follow-ups; #19 owed one-variable runs; #20 pool guard (infra)
- #16 islands — scale demes, then region islands from SCIP's open nodes (Berk's "large chunks"; draft
  [[Islands (draft)]])

- D-013 follow-ups: HiGHS alone (H0) and HiGHS ↔ EA (H3) on the development set; a cuOpt + EA coupling design.

## 2. Practice → scale
- More seeds on the development set (4 per instance), then the larger problem set: finish the stage-2 screening and
  freeze the pool (#1), frozen references (#2), calibration (#3), runner gates (#4, #5, #13).

## 3. Freeze the configuration
- E4 tuning within its cap (#7) → the frozen configuration.

## 4. Confirmatory
- E2/E3 on unseen instances (#8); test size as proposed in D-010 (16), open in #11.

## 5. After tuning
- #21 **Laya** decision model: offline test on recorded decision states (repair clock, strategy, islands) against the
  measured rules and LightGBM; live only at a low duty cycle if it wins offline; the decision maker in E6.
- E5 long runs and open instances (#9); E6 multi-objective — I-TREA's interactive decision maker (#10).

## 6. Paper (#12)
