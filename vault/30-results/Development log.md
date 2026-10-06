# Development log (D-010)

One row per (configuration, instance, seed) on the development set (`datasets/dev_set.csv`), 120 s unless stated.
Descriptive only. `cfg` = EAConfig hash; `P_EA` = the EA's own primal integral, `P_port` = portfolio (SCIP ⇄ EA),
`P_SCIP` = SCIP alone (A0) of the same seed; `ttf` = time to first feasible of the EA; `posts/acc` = EA posts and
SCIP-accepted posts. Change notes say what was changed and why, with the commit.

## Changes
| date | commit | change | why |
|---|---|---|---|
| 2026-10-06 | 4127f10 / e05886c | baselines: default config `9ac4a93bf93b`, A0 / A5 / A3 × 9 × seeds 0–1 + A6, 120 s, beside an idle [local LLM agent] (`--allow-serving`) | D-010 yardstick |

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
