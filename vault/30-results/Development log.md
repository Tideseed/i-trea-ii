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

| 2026-10-06 | (this commit) | `ranks.feas_alt_scale` (off by default, every hash unchanged): two-scale Pareto survival in the feasibility phase over (V_l2, V_rhs). Not yet run | fhnw-binpack4-48: rhs-repaired children die under l2 survival (ad1/lps1 final population 8–9 of 256); consult with GPT-6.1 Sol |
| 2026-10-06 | 4d044ac | aggregation of two-variable equality rows (`src/itrea/ea/aggregate.py`), exact; NOT wired in (payoff probe negative on radiationm) | radiationm/rail01 equality chains |
| 2026-10-06 | (this commit) | repair clock, Stages 0–1 of [[2026-10-06 Repair-depth design (workflow)]] (`repair_clock` section, off by default — every recorded hash unchanged: 9ac4a93bf93b, 584ddee83747, a2ad94e9188d, 4a96b872ffdc): a lane-step horizon per repair call (status 4) in the CPU feasibility jump and, host-side between launches, in the GPU wrapper (no kernel edit); per-call launch profiles and per-lane steps; the controller `ea/repair_clock.py` (levels 2^7–2^16 + "none", "none" until the population's first feasible member, counterfactual gain per level, decayed rates, 5 generations / 2 material gains / 1.25× to switch, probes, LP batch pinned, 10 s cap); engine wiring with one pairing helper, `rclock` events, `rc_*` gen fields, a summary block. `configs/itrea2-adaptive-lps-rc.yaml` = lps1 + the clock in SHADOW mode (plays "none": the search is lps1's — identical genealogy to the section off for the same generation count, tested on NumPy; in a timed run its host work can shift generation counts, so its runs are not lps1 runs). After three adversarial reviews (same evening), deviations from the synthesis plan, for D-012: the gain unit dg = min((cutoff − key) / max(\|cutoff\|, \|key\|, \|dual\|), 1) (the plan's unit gave ~1e9 per improvement over an incumbent near 0 without a dual bound — neos-960392); the phase ends with a feasible member of the EA's own (an injected solver row ended it); the immigrant calls after the survival are left out of the gain; a cut children call enters the cap's median as a censored time (the cap could never rise); shadow mode accumulates only what its decision would reveal (it never probed); plus an uncut CPU call reveals every level, and each call logs cutoff, dual_key and the improving lanes that did not end FEASIBLE. CPU suite only; the GPU wrapper's horizon/deadline path and its gpu-marked tests have NOT run. Not yet run | the repair depth is instance-dependent (lps1 wins 6/9, the default depth only item_placement); Stage 1 = A3@rc0 shadow on item_placement s0–1, glass4, sing44, neos-960392, loue, ca s0 with the design note's pass criteria (a)–(c) |

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

Provenance caveat: the 18 runs span code states 0e69732 (9 clean, 7 with the uncommitted LP-stream code — off in
these runs) and 770df17 (4); a disabled stream is bit-identical (review of 770df17), so the numbers stand
([[Mistakes and incidents]], 2026-10-06 15:15–16:35).
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

## Aggregation probe (2026-10-06 16:1x) — CPU only, not wired into the EA
`src/itrea/ea/aggregate.py` substitutes two-variable equality rows (equal magnitudes; integrality kept) exactly:
x_v = ±y_root + o. On the dev set: radiationm18-12-05 40,623 → 14,919 variables (rows 40,935 → 15,231); rail01
117,527 → 96,736 (rows 46,843 → 26,052); sing44 59,708 → 57,448; ca__Scenario400 305,712 → 305,328 (32,160 two-variable
equalities with unequal coefficients refused); glass4, fhnw, loue: nothing. SCIP's and cuOpt's solutions of radiationm,
rail01, sing44 and ca project to feasible reduced points and lift back exactly (same objective).
Payoff test (`tools/agg_probe.py`, the CPU feasibility jump with the c01 budget from the same LP roundings,
radiationm18-12-05, 32 children): original 0/32 feasible, median V 127; aggregated 0/32, median V 88; aggregated with
10× the budget 0/16, median V 42. The rows still violated on the aggregated model are 154 equality rows over ≥ 9
general integers (exact sums), 76 rows over continuous variables and a few 3-variable integer equalities — not the
chains. **Verdict: not wired in.** The chains are not radiationm's blocker; exact integer sums are (an LP-guided
rounding with propagation is the candidate — what cuOpt's 0.8 s first solution suggests). The module stays, tested
(`tests/test_aggregate.py`), for rail01-type instances if a later probe shows the chains matter there.

## LP stream (2026-10-06 16:03–16:35) — A3@lps1 vs A3@ad1, 9 instances × seeds 0–1, 120 s
`lps1` = `ad1` + `lp_stream.enabled` (effective hash `4a96b872ffdc`, commit c192bb5) — one variable against `ad1`.
P(120), seeds averaged:

| instance | SCIP alone | coupled default | coupled adaptive (ad1) | **+ LP stream (lps1)** | cuOpt alone |
|---|---|---|---|---|---|
| ca__Scenario400_reserves_1 | 1.000 | 1.000 | 0.296 | 0.297 | 0.645 |
| fhnw-binpack4-48 | 1.000 | 1.000 | 1.000 | **0.651** (s0 0.302, s1 1.000) | 0.005 |
| glass4 | 0.344 | 0.247 | 0.241 | 0.228 | 0.308 |
| item_placement_9903 | 0.660 | 0.400 | 0.522 | 0.569 | 0.666 |
| neos-3024952-loue | 1.000 | 0.344 | 0.441 | 0.369 (s0 0.478 → 0.332) | 0.760 |
| neos-960392 | 1.000 | 0.042 | 0.032 | 0.032 | 0.571 |
| radiationm18-12-05 | 0.229 | 0.237 | 0.235 | 0.234 | 0.009 |
| rail01 | 1.000 | 1.000 | 1.000 | 1.000 | 0.927 |
| sing44 | 0.525 | 0.651 | 0.361 | 0.362 | 0.726 |

Mean bounded d of lps1: vs ad1 **+0.05** (3 better: fhnw, loue, glass4; 1 worse: item_placement −0.08); vs SCIP alone
**+0.38** (7 better, 0 worse, radiationm −0.02 ≈ equal, rail01 equal); vs cuOpt alone +0.10 (6 better, 3 worse: fhnw,
radiationm, rail01). Hygiene: 18/18 completed, every incumbent checked, GPU health ok.

**Stream counters.** SCIP spent ≤ 0.7 s of 120 s on the stream except radiationm (2.43 s — the bucket's ceiling, 174
reads). No node LP exists on ca, rail01 (root LP never finishes), and only 1–2 / 10–11 on neos-960392 / sing44 (SCIP
stays at the root), so the stream cannot act there. The EA took 16–384 points per run; on glass4/fhnw/loue it **missed
95–358 ids** between two polls (SCIP posts every 0.25 s, the EA reads once per exchange tick — the reads were paid for
nothing) and dropped 17–29 unused ones. Stream-born members rarely sit in Rank 0 at the end (0–11).
**Attribution — weak.** fhnw seed 0's optimum (objective 0, at 36.2 s; SCIP accepted it at 36.4 s) came from an EA
child (uniform crossover + toward_lp mutation + repair, strategy l2/lp) whose ancestry over three generations shows LP
immigrants but no stream-born member; seed 1 found nothing. The +0.05 is a two-seed, three-instance signal — the stream
is kept on (cheap, never harmful by more than item_placement's −0.08) but not yet credited.
**Next for the stream:** match SCIP's read rate to the EA's poll rate (missed ids are wasted SCIP time), then judge it
on more seeds once the feasibility work (below) is in.

## Finding (16:3x): the c01 repair budget starves the evolution
`ad1` and `lps1` carried the c01 budget (`repair.max_steps 20000, repair.stall_steps 1000`) for comparability with
feasibility probe 1. It makes every generation 6–9× slower, because the GPU repair waits for the slowest individual:
coupled glass4 296 generations in 120 s (default) → 34 (ad1) / 27 (lps1); neos-3024952-loue 303 → 48; item_placement
2,060 → 465 (repair moves per child 99–166 → 460–890). The EA alone on glass4: 956 → 110. This explains the ad1 losses
on loue and item_placement (3–4× fewer posts) and the stream's missed ids (the EA reads the mailbox between
generations, every 2.5–4.5 s). In probe 1, c01 alone never unlocked an instance — the LP point and the rhs scale did.
So today's ad1/lps1 gains came with an EA that hardly evolved. **Next run `lps0`** = `itrea2-adaptive-lps` at the
DEFAULT repair budget (hash `a2ad94e9188d`), after `lp1`.
Improvement probe (`tools/agg_improve_probe.py`, 16:5x): from SCIP's incumbent #1 of radiationm18-12-05 (objective
83,244 at 0.31 s; the optimum is near 17,500), 32 copies with 5 integer genes mutated, the CPU feasibility jump with the
objective cutoff: **0/32 better-and-feasible** on the original and on the aggregated model, at both budgets (children
end with V ≈ 0.003 — stuck just short). The objective is ONE integer variable (a min-max structure): lowering it by 1
tightens every row it bounds at once, which no single or chain move does. radiationm needs a non-local operator
(fix-and-propagate around the incumbent, or a residual sub-MIP), not better local moves; parked.

## Repair budget, one variable (2026-10-06 16:55–17:30) — A3@lps0 vs A3@lps1, 9 instances × seeds 0–1, 120 s
`lps0` = `itrea2-adaptive-lps` at the DEFAULT repair budget (hash `a2ad94e9188d`, code f8adb63); `lps1` = the same +
the c01 budget (`repair.max_steps 20000, repair.stall_steps 1000`). P(120), seeds averaged:

| instance | SCIP alone | coupled default | lps1 (deep repair) | **lps0 (default repair)** | generations lps1 → lps0 | cuOpt alone |
|---|---|---|---|---|---|---|
| ca__Scenario400_reserves_1 | 1.000 | 1.000 | 0.297 | 0.395 | — → 67–79 | 0.645 |
| fhnw-binpack4-48 | 1.000 | 1.000 | 0.651 | 1.000 | 37 → 183–308 | 0.005 |
| glass4 | 0.344 | 0.247 | 0.228 | 0.323 | 27 → 220–224 | 0.308 |
| item_placement_9903 | 0.660 | 0.400 | 0.569 | **0.358** | 371 → 1770–1840 | 0.666 |
| neos-3024952-loue | 1.000 | 0.344 | 0.369 | 0.440 | 48 → 429–436 | 0.760 |
| neos-960392 | 1.000 | 0.042 | 0.032 | 0.039 | — → 99–365 | 0.571 |
| radiationm18-12-05 | 0.229 | 0.237 | 0.234 | 0.235 | — → 508–512 | 0.009 |
| rail01 | 1.000 | 1.000 | 1.000 | 1.000 | — → 854–855 | 0.927 |
| sing44 | 0.525 | 0.651 | 0.362 | 0.551 | — → 448–451 | 0.726 |

lps0 vs lps1: mean bounded d **−0.13** (1 better: item_placement +0.37; 6 worse: fhnw −0.35, sing44 −0.34, glass4
−0.29, ca −0.25, neos-960392 −0.18, loue −0.16; 2 equal). lps0 vs SCIP alone +0.29 (5 better, 2 worse: sing44 −0.05,
radiationm −0.02). Hygiene: 18/18 completed, every incumbent checked, GPU health ok. Stream: with fast generations the
EA misses almost nothing (0–17 ids, was 95–358).

**Correction to the 16:3x finding.** I read "6–9× fewer generations" as "the EA hardly evolved, so the gains came despite
the budget" and expected the default budget to do as well or better. Measured, the deep repair wins on 6 of 9: on
these instances the repair, not the number of generations, does most of the work (sing44 and ca need it to turn LP
roundings feasible; glass4's 209 posts at the default budget improved SCIP less than lps1's 22). Only item_placement —
many small improvements, 1,700+ posts — gains from fast generations. **The repair depth is an instance-dependent
trade-off**, the same kind the self-adaptive strategy was built for; but the GPU repair call waits for its slowest
individual, so mixing depths inside one call would not make a generation faster. Candidates: a time budget per repair
call instead of a step budget; alternating deep and shallow generations; or depth as a strategy gene with the
generation time charged to the strategy. **Development baseline for now: lps1** (deep repair + adaptive + HiGHS LP +
stream), the best measured configuration.

## fhnw two-scale survival (2026-10-06 17:5x–18:08) — alt1 = lps0 + `ranks.feas_alt_scale: rhs`, fhnw-binpack4-48, seeds 0–1
| arm | seed 0 | seed 1 | rhs-repaired members at the end (of 256) |
|---|---|---|---|
| A3@alt1 (coupled) | 1.000 | 1.000 | 136 / 223 |
| A5@alt1 (EA alone, diagnostic) | 1.000 | 1.000 | 245 / 139 |
| A3@lps0 (same, one scale) | 1.000 | 1.000 | 0 / 0 |
| A3@lps1 (deep repair, one scale) | **0.302** | 1.000 | 36 / 13 |
| A5@c04 (deep repair, rhs on repair AND survival) | **0.253** | — | — |
**Reading.** The two-scale survival does what it was built for — the rhs-repaired individuals survive (136–245 of 256,
against 0 under one scale) — but at the default repair depth no feasible point appears in 120 s. Both fhnw finds so
far had deep repair (c04 at 30 s, lps1 at 36 s). Next: `lps1 + feas_alt_scale rhs` (deep repair + two scales).
`lp1` (LP source alone) got 1 of 18 runs before the window closed (neos-3024952-loue s1 0.355); the rest waits.

## Repair-depth design (2026-10-06 17:36–18:40)
A design workflow (code map, run-data analysis, literature, three designs, two adversarial judges) chose a **repair
clock**: a lane-step horizon per repair call enforced between GPU launches, chosen per generation from the measured
gain per second of each horizon, with deep repair kept until the EA's own first feasible point, a materiality floor,
and the LP batch always repaired deep. Staged: instrumentation + shadow mode first (no behaviour change; pass criteria
stated in advance), the active controller only if the shadow signal separates item_placement from glass4/loue.
Full record: [[2026-10-06 Repair-depth design (workflow)]].

## Repair clock, Stage 1 — shadow (2026-10-07 09:01–09:26) — A3@rc0 vs in-window A3@lps1w, 7 blocks
Both LLM servers stopped (Berk 09:0x), GPU otherwise idle. Stage 0b first: full suite 656 passed / 2 skipped
(gpu-marked horizon tests included); on CuPy, shadow and shadow-forced runs are bit-identical to the disabled run in
6 of 6 toy scenarios (26 arrays each). 14/14 runs completed. `logs/rc_stage1.py` (to be moved to `tools/`):

| run | generations rc0 / control | first own feasible | `would` in the post phase | criterion |
|---|---|---|---|---|
| item_placement s0 | 411 / 376 | 0.1 s | 2^7 59 %, 2^8 14 %, 2^12 10 % | (a) PASS — left none at 0.4 s for 2^7, rate ratio 3.05 |
| item_placement s1 | 394 / 395 | 0.1 s | 2^7 36 %, 2^9 17 %, 2^8 16 % | (a) PASS — left none at 0.4 s for 2^7, rate ratio 2.14 |
| glass4 s0 | 17 / 25 | 5.5 s | none 35 %, 2^12 29 %, 2^13 24 %, 2^11 12 % | (b) FAIL 35 %; (c) FAIL −32 % |
| neos-3024952-loue s0 | 47 / 52 | 2.1 s | 2^11 46 %, 2^7 23 %, none 21 % | (b) FAIL 21 % |
| neos-960392 s0 | 9 / 9 | 0.9 s | none 56 %, 2^7 22 %, 2^8 22 % | (b) FAIL 56 % |
| sing44 s0 | 70 / 69 | 12.7 s | none 91 % | (c) PASS |
| ca s0 | 18 / 19 | 34.8 s | none (1 post generation) | (c) PASS |

**Verdict by D-012's letter: FAIL** ((a) pass, (b) fail, (c) fail on glass4). Two findings qualify it:
- **(c) is not overhead.** Seconds per generation outside the repair calls are equal within a few % (glass4 1,008 vs
  1,073 ms; loue 122 vs 125; item 93 vs 96; sing44 117 vs 103); glass4's 17 vs 25 generations come from longer repair
  calls on its trajectory (4.8 vs 3.7 s per generation — coupled runs diverge after the first exchange). The proxy
  "generation count within 10 %" is too noisy for an instance with ~20 generations.
- **(b) tested a premise, not only the controller.** It assumed "glass4, loue, neos-960392 want deep repair" from
  lps0 < lps1 — but the default budget is max_steps 2,000 AND stall_steps **100**; the clock's shallow levels keep
  stall 1,000 and only cap the total at 2^11–2^13. glass4's two material gains came from lanes of 1,073 and 287 steps.
**Next — test the premise, one variable each** (`logs/dev-premise.jsonl`, queued): `cap4k` (lps1 with max_steps
4,096, stall 1,000 — what the clock chose) and `stall100` (lps1 with stall 100 — what made lps0 shallow) on glass4,
loue, neos-960392, item_placement × seeds 0–1, against lps1w. If cap4k ≈ lps1 there, the controller's shallow choice
is harmless and Stage 2 (active) is the real test; if cap4k loses, the counterfactual is myopic and D-012's fallback
(the per-generation bandit) follows.

## fhnw-binpack4-48: deep repair + two-scale survival (2026-10-07 09:27–09:51) — A3@alt2 vs in-window A3@lps1w, seeds 0–3
| arm | s0 | s1 | s2 | s3 | solved |
|---|---|---|---|---|---|
| alt2 = lps1 + `ranks.feas_alt_scale: rhs` | 1.000 | **0.214** (first feasible 25.7 s) | 1.000 | 1.000 | 1/4 |
| lps1w (deep repair, one scale) | 1.000 | 1.000 | 1.000 | 1.000 | 0/4 |
All fhnw runs so far — solved / runs: adaptive + deep repair (lps1, lps1w, alt2) **2 / 10**; adaptive + default depth (lps0,
alt1) 0 / 4; **pure rhs scale on repair and survival, no adaptive strategy (c04: A5 at 30 s, A3 at 76 s) 2 / 2**.
Reading: two-scale survival keeps the rhs-repaired members alive (236–244 of 256 at the end of the unsolved alt2 runs)
but does not change the solve rate measurably (1/4 vs 0/4; n is tiny). The configuration that solved fhnw every time
is the pure rhs scale — in the adaptive strategy rhs is one of four strategies, and selection does not concentrate on
it within 120 s. Hygiene: 8/8 completed, every incumbent checked, GPU health ok.

## The EA's own LP alone (2026-10-07 09:51–10:23) — A3@lp1 = coupled default + `seeding.lp_source: highs_async`, 9 × seeds 0–1
One variable against the coupled default (18/18 runs; 1 from 2026-10-06), every incumbent checked, GPU health ok.

| instance | SCIP alone | coupled default | **+ own LP (lp1)** | lps0 (+ strategy + stream, default depth) | lps1 (+ strategy + stream + deep) | cuOpt alone |
|---|---|---|---|---|---|---|
| ca__Scenario400_reserves_1 | 1.000 | 1.000 | 0.509 | 0.395 | **0.300** | 0.647 |
| fhnw-binpack4-48 | 1.000 | 1.000 | 1.000 | 1.000 | 0.651 | **0.005** |
| glass4 | 0.344 | 0.247 | 0.300 | 0.323 | **0.228** | 0.308 |
| item_placement_9903 | 0.660 | 0.400 | **0.218** | 0.358 | 0.569 | 0.666 |
| neos-3024952-loue | 1.000 | **0.344** | 0.369 | 0.440 | 0.369 | 0.760 |
| neos-960392 | 1.000 | 0.042 | **0.027** | 0.039 | 0.032 | 0.571 |
| radiationm18-12-05 | 0.229 | 0.237 | 0.224 | 0.235 | 0.234 | **0.009** |
| rail01 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **0.927** |
| sing44 | 0.525 | 0.651 | 0.492 | 0.551 | **0.362** | 0.726 |

| comparison | better / worse | mean d [95 % CI] | Wilcoxon p | D-002 rule |
|---|---|---|---|---|
| lp1 vs coupled default | 5 / 2 | +0.19 [+0.01, +0.37] | 0.156 | inconclusive |
| lp1 vs SCIP alone | **7 / 0** | **+0.43 [+0.19, +0.68]** | **0.016** | supported |
| lps1 vs lp1 | 4 / 4 | +0.06 [−0.17, +0.25] | 0.64 | inconclusive |
| **lps0 vs lp1** (strategy + stream at default depth) | 1 / 6 | **−0.12 [−0.25, +0.01]** | 0.156 | **negative** |
| lps1 vs lps0 (deep vs default depth) | 7 / 1 | +0.15 [−0.02, +0.28] | 0.195 | inconclusive |
| lp1 vs cuOpt alone | 6 / 3 | +0.05 [−0.38, +0.44] | 0.65 | inconclusive |

**Reading.** The EA solving its own LP right after t0 is the single change that carries the gain over SCIP alone (lp1
matches lps1's +0.43 against SCIP), and it gives the best item_placement result so far (0.218). At the default depth
the adaptive strategy and the LP stream add nothing measurable — slightly negative (1 / 6). Deep repair trades
instances (ca, sing44, glass4, fhnw gain; item_placement and neos-960392 lose). **Owed next:** `lpd` = lp1 + deep repair,
no strategy, no stream (queued) — it isolates strategy + stream at the deep depth (lps1 vs lpd).

## Premise test for D-012's criterion (b) (2026-10-07 10:23–11:17) — one variable each vs deep repair (lps1w)
`cap4k` = lps1 with repair.max_steps 4,096 (stall 1,000 kept — the regime of the clock's shallow levels);
`stall100` = lps1 with repair.stall_steps 100 (max_steps 20,000 kept — the stall rule of the default budget).
24/24 runs (incl. controls), every incumbent checked, GPU health ok. P(120):

| instance | seed | lps1w (deep) | cap4k | stall100 | generations lps1w / cap4k / stall100 |
|---|---|---|---|---|---|
| glass4 | 0 / 1 | 0.226 / 0.229 | **0.217 / 0.222** | 0.241 / 0.315 | 26 / 28 / 158 |
| item_placement_9903 | 0 / 1 | 0.573 / 0.408 | 0.517 / 0.543 | **0.371 / 0.317** | 389 / 395 / 1,792 |
| neos-3024952-loue | 0 / 1 | 0.433 / 0.371 | **0.336** / 0.419 | 0.463 / 0.457 | 49 / 44 / 444 |
| neos-960392 | 0 / 1 | 0.032 / 0.031 | **0.025 / 0.027** | 0.067 / 0.043 | 10 / 20 / 174 |
cap4k vs lps1w: 3 better / 1 worse, mean d **+0.05 [−0.04, +0.15]**; stall100 vs lps1w: 1 / 3, mean d **−0.11
[−0.36, +0.18]** (glass4 −0.18, loue −0.13, neos-960392 −0.43; item_placement +0.30).

**Reading.** The default budget lost on glass4, loue and neos-960392 because of its **stall rule** (100 steps), not its
step cap: capping the total at 4,096 lane steps with the 1,000-step stall costs nothing (slightly better). So
criterion (b)'s premise ("these instances want the horizon at none") is refuted for the clock's regime, and Stage 1's
(b) failure says nothing against the controller there. item_placement's gain needs many generations (stall100: 4–5×
more), which only a short repair gives — the clock chose 2^7 there (Stage 1, criterion (a) passed). **Decision:**
proceed to Stage 2 (active) under D-012's Stage-2 rejection rules, instead of the Stage-1 fallback — recorded as a
deviation in D-012. Caveat: the clock also chose 2^7 on loue 23 % of the time; whether that short regime hurts there
(as stall100 did) is exactly what Stage 2 measures.
