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
