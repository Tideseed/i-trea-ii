# Quantification of the development baseline, 2026-10-06 (descriptive)

**Arm:** A3@lps1 — SCIP coupled with the GPU EA: adaptive repair strategy + the EA's own HiGHS LP + LP stream + deep
repair (`itrea2-adaptive-lps` + c01 budget, effective hash `4a96b872ffdc`). 9 development instances, 120 s, seeds 0–1
(cuOpt alone: seed 0 only). Metric: primal integral P(120), lower is better. Statistics: the pre-registered machinery
(`itrea.analysis.stats.compare`, D-002 §6 — bounded d, Wilcoxon signed-rank exact, 10,000-resample bootstrap over
instances, decision rule). **Descriptive only:** the development set was chosen by looking at screening results and the
configuration was chosen on it (D-010), so these are not confirmatory numbers — the confirmatory test runs on unseen
instances (E2/E3).

## Per instance (seed means)
| instance | SCIP alone | cuOpt alone | coupled default | **coupled best (lps1)** | vs SCIP | vs cuOpt | final gap SCIP → best |
|---|---|---|---|---|---|---|---|
| ca__Scenario400_reserves_1 | 1.000 | 0.645 | 1.000 | **0.297** | −70 % | −54 % | none → 0.2 % |
| fhnw-binpack4-48 | 1.000 | **0.005** | 1.000 | 0.651 | −35 % | worse | none → 50 % (one seed solved) |
| glass4 | 0.344 | 0.308 | 0.247 | **0.228** | −34 % | −26 % | 31 % → 15 % |
| item_placement_9903 | 0.660 | 0.666 | 0.400 | 0.569 | −14 % | −15 % | 64 % → 38 % |
| neos-3024952-loue | 1.000 | 0.760 | 0.344 | **0.369** | −63 % | −51 % | none → 26 % |
| neos-960392 | 1.000 | 0.571 | 0.042 | **0.032** | −97 % | −94 % | none → 0 (optimal) |
| radiationm18-12-05 | 0.229 | **0.009** | 0.237 | 0.234 | +2 % | worse | 15 % → 14 % |
| rail01 | 1.000 | **0.927** | 1.000 | 1.000 | 0 | worse | none → none |
| sing44 | 0.525 | 0.726 | 0.651 | **0.362** | −31 % | −50 % | 0.8 % → 1.3 % |
("none" = no feasible solution in 120 s; final gap = primal gap to the reference at 120 s.)

## Aggregates
| comparison | instances | better / worse | mean d [95 % CI] | Wilcoxon p | mean P | geo-mean P ratio | D-002 rule |
|---|---|---|---|---|---|---|---|
| best vs **SCIP alone** | 8 (rail01: both unsolved) | 7 / 1 | **+0.43 [+0.23, +0.65]** | **0.016** | 0.72 → 0.34 (−52 %) | **0.42** | supported |
| best vs **cuOpt alone** | 9 (cuOpt: 1 seed) | 6 / 3 | +0.11 [−0.33, +0.50] | 0.65 | 0.51 → 0.36 (−30 %) | 1.17 | inconclusive |
| best vs coupled default | 8 | 6 / 2 | +0.18 [−0.02, +0.39] | 0.15 | 0.49 → 0.34 (−30 %) | — | inconclusive |
| best vs coupled adaptive (ad1) | 8 | 4 / 4 | +0.06 [−0.02, +0.15] | 0.46 | 0.39 → 0.34 (−12 %) | — | inconclusive |
Time to first feasible and to a 1 % gap vs SCIP alone: no difference (paired Prentice–Wilcoxon p = 0.36 / 0.46;
first feasible 3 faster, 4 slower) — the gain is in the quality reached over time, not in the first solution.

## Reading
- Against SCIP alone the win is large and consistent on this set: the primal integral is less than half
  (geometric mean ratio 0.42), 7 of 8 instances better, the interval excludes 0.
- Against cuOpt alone it is a split: 15–94 % better on 6 instances, far worse on 3 (fhnw, radiationm, rail01), so the
  geometric mean ratio is 1.17 (worse) while the arithmetic mean P is 30 % lower. Two seeds (one for cuOpt) and 9
  instances give wide intervals; more seeds per instance are the cheapest way to narrow them.
