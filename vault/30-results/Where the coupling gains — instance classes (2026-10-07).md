# Where the coupling gains — instance-class analysis (2026-10-07)

**Question (Berk, 2026-10-07):** "Is there a clear distinction between model sets that our algorithm's performance
gains can be understood better?" Answered by a multi-agent workflow (Claude Code subagents, CPU only, read-only on the
repository): structure features of the 9 development instances, performance and mechanism signals from the runs, an
open-access literature check, one characterisation, two adversarial attacks (statistics/overfitting; mechanism
plausibility), one final answer. Supporting tables were written to /tmp/itrea-classes/ (not kept in the repository).
Development data — descriptive; related: [[Where cuOpt struggles (optopt background, 2026-10-07)]],
[[D-013 Aim — lift each solver's weakest cases]].

## Answer (final, verbatim)

## What separates wins from losses

Nothing we can read from a model before a run separates them on these 9 instances. Both reviews rejected the proposed structural classes (objective density, share of two-variable equalities). The split shows up during the run instead, and part of it is the outcome measured a second way.

**Against cuOpt, cuOpt's own trajectory decides the result.** There are three patterns:
- **Early close** (fhnw optimal at 0.56 s; radiationm gap 0.0003 at 3.2 s): we lose by 140–214× and 25×. Both objectives have at most one non-zero. Losing here is nearly arithmetic: a gap of 0.05 or less at 10 s caps cuOpt's P(120) at 0.13.
- **Slow start** (presolve 23–26 s, 0 nodes, improvements only after 64 s): we win on ca, neos-960392 and sing44 by 21–95 %. rail01 is a near tie (+8 %); nobody does well there.
- **Stall** (fast first solution, many nodes, then little progress): we win on glass4, item_placement and loue by 3–67 %.

**Against SCIP, we gain wherever the EA finds solutions before SCIP does.** It gets there four different ways:
- The EA's LP finishes before SCIP's root LP: ca (−49 % / −70 %), sing44 (−6 % / −31 %).
- Repair from random starts, with no LP needed: neos-960392 (−97 %).
- LP point plus repair, where SCIP's own heuristics find nothing: loue (−63 %).
- Polishing SCIP's solution: item_placement (−67 % / −14 %), glass4 (−13 % / −34 %).

**The three instances with no gain fail for three different reasons**, which is why no single rule fits:
- rail01: the LP takes 93 s on one thread, longer than our 60 s cap, so the EA never gets an LP point.
- fhnw: how the big-M rows are scaled. The rhs scaling solves it in 4 of 5 runs, so this is a setting, not a property of the model.
- radiationm: our repair gives 0–2 improving children per run, while cuOpt keeps improving before its presolve has even finished. The gap is in our operator.

## How sure we are / how to test it

- **Nine instances cannot support a structural rule.** In 48 % of random 3-instance "loss sets", some feature separates them perfectly. A two-feature "A or B" rule fits 81 of 84.
- **Seed stability is weaker than the headline.** The gain keeps its sign on both seeds on 5 instances for lp1 (p = 0.0625) and on 6 for lps1 (p = 0.031). radiationm and sing44 (lp1) flip sign between seeds. The +0.43 mean leaves out instances where every run on both sides has P = 1. Over all 9 the means are +0.33 (lp1) and +0.38 (lps1).
- **The earlier study (optopt) gives leads only.** It compares cuOpt with SCIP, has no EA runs, uses one seed, and the patterns were found after the fact. On 27 instances where SCIP struggles, an almost empty objective goes with cuOpt being strong (4 of 9 against 1 of 18, p = 0.03).

**Cheapest test (about 2 h, one window):**
1. Before any run, write a dated note that fixes one configuration (e.g. lp1 with its row scaling) and two predictions:
   - (a) We lose to cuOpt on krka, kumeu and chromaticindex1024-7, where cuOpt finds a solution, and beat it on comp21-2idx and toguru, where it stalls.
   - (b) We beat SCIP (d > 0.05 on both seeds) only where the one-thread LP takes 30 s or less offline. That cut-off rests on just two comparisons of in-run against offline LP time.
2. Run SCIP alone, cuOpt alone and the coupled system, 2 seeds each at 120 s, on 8 public screening instances where SCIP struggles, from families not in the development set:
   - objective with at most 4 non-zeros: neos-3004026-krka, neos-3656078-kumeu, chromaticindex1024-7, cryptanalysiskb128n5obj16;
   - dense objective: comp21-2idx, neos-4763324-toguru;
   - many two-variable equality rows: neos-2657525-crna, neos-2746589-doon.

The cuOpt results on these are already in optopt; only our side is new. Eight instances can refute the distinction but only weakly confirm it. They also leave the final test pool, which drops from 69 to 61 SCIP-weak MIPLIB candidates outside the development set.

## Feature table

Negative % means our P is lower (better); ×N means our P is N times cuOpt's. Columns give lp1 / lps1. * = the sign depends on the seed. Two-variable equalities are rows of the form x = ±y + c.

| instance | class/structure | key features (pre-run) + cuOpt pattern | P vs SCIP | P vs cuOpt |
|---|---|---|---|---|
| ca__Scenario400_reserves_1 | unit commitment, 52 % continuous | objective non-zeros 38 % of variables; LP 14.4 s; cuOpt slow start | −49 % / −70 % | −21 % / −54 % |
| neos-960392 | pure binary, set packing | objective non-zeros 2.7 % of variables; LP 3.3 s; cuOpt slow start | −97 % / −97 % | −95 % / −94 % |
| sing44 | variable bound + cardinality | objective non-zeros 44 % of variables; LP 7.2 s; cuOpt slow start | −6 %* / −31 % | −32 % / −50 % |
| neos-3024952-loue | pure general integer, precedence | objective non-zeros 94 % of variables; LP 0.3 s; cuOpt stall | −63 % / −63 % | −51 % / −51 % |
| item_placement_9903 | set partitioning | objective non-zeros 3 % of variables; LP bound useless; cuOpt stall | −67 % / −14 % | −67 % / −15 % |
| glass4 | big-M mixed binary | objective non-zeros 5.9 % of variables; LP 0.0 s; cuOpt stall | −13 % / −34 % | −3 % / −26 % |
| fhnw-binpack4-48 | bin packing, pure feasibility | empty objective; LP rounding breaks 47 % of rows; cuOpt early close | 0 % / −35 %* | ×214 / ×140 |
| radiationm18-12-05 | 63 % two-variable equalities, general integers | objective has 1 non-zero; LP 0.2 s, gap 0.07 %; cuOpt early close | −2 %* / +2 %* | ×25 / ×26 |
| rail01 | 97 % equality rows, 117k binaries | 44 % two-variable equalities; LP 93 s (above the 60 s cap); cuOpt slow start | 0 / 0 (all P = 1) | +8 % / +8 % |

Supporting files are in /tmp/itrea-classes/: features.csv, perf.csv and perf_runs.csv, sk/lanes_all_arms.csv, and optopt_modes_joined.csv. The 8 test instances come from datasets/runs.parquet (screening set screen1) joined with /tmp/itrea-classes/cheap80.csv.

## Attacks (verbatim)

**Verdict: does not hold.**

**Problems.** The question asks whether something in the models themselves, readable before a run, separates our wins from cuOpt's. On these 9 instances nothing structural does better than chance. The two "perfect" run-time separators are the outcome measured a second way. I enumerated every possible labelling (CPU only, cores 10-14, single thread; scripts in /tmp/itrea-classes/sk/).

1. Structural separators are what chance gives. There are 84 numeric, non-constant structural features and 84 ways to pick 3 "losses" out of 9 instances.
   - 40 of the 84 random 3-instance loss sets (48 %) have at least one feature that separates them perfectly.
   - The real loss set {fhnw, radiationm, rail01} has 2 such features. 30 % of random loss sets have 2 or more.
   - Every random loss set has at least one feature that gets 8 of 9 right (mean 12 such features; the real set has 14).
   - A two-feature rule of the form "feature A beyond a cut OR feature B beyond a cut", with both cuts free, separates 81 of 84 random 3-instance loss sets perfectly, and 122 of 126 random 4-instance sets. The proposed F1/F2 classes are a rule of this kind, with cuts placed in the gaps of these same 9 points. A perfect post-hoc fit is close to guaranteed.
   - Class II {radiationm, rail01}: 7 features separate it, all of them the same equality-share idea (share_eq, two_var_eq_share, aggregation share, ...). But 25 of 36 random pairs have a separator, and 6 of 36 (17 %) have 7 or more.
   - Even so, the structural classes do not explain all three losses: fhnw sits in class III with three wins.

2. The rho of -0.86 between lp_frac_int_n and d versus SCIP is the best of 84 features. Over all 9! = 362,880 permutations of the outcome, the best of 84 features reaches |rho| of 0.862 or more 15.5 % of the time; the typical best is 0.78. That is not distinguishable from choosing the best feature after looking.

3. D2 (cuOpt's mode) restates the outcome. If cuOpt's gap is at most 0.05 at 10 s, its P(120) can be at most (10 + 110 × 0.05) / 120 = 0.129. Our best P is 0.22 or more on 8 of 9 instances; the exception is neos-960392 at 0.027. So "we lose wherever cuOpt closes early" holds by arithmetic. It also means class III (4 of 9 dev instances) predicts nothing against cuOpt without running cuOpt, which is the comparator itself.

4. D1 (at least 10 repair lanes ending feasible) is measured inside the same run, so it describes how the EA gains, not what predicts it. It is also not perfect once all arms are counted. Across all 140 completed coupled dev runs (every A3 variant), compared with the SCIP run of the same seed, gain > 0.05 agrees with D1 in 128 and disagrees in 12:
   - D1 fails but the run gains, breaking "no gain where D1 fails" (5 runs): fhnw under alt2 s1 (2 lanes, d +0.79), c04 s0 and s2 (2 and 1 lanes, d +0.37 and +0.54), lps1 s0 (1 lane, d +0.70); ca under lps0 s0 (7 lanes, d +0.51). On a pure-feasibility instance one feasible lane is the solution, so the cut of 10 is arbitrary.
   - D1 holds but the run loses to SCIP, breaking "gain ≈ headroom × D1" (5 runs, all sing44): A3 default s0 and s1 (66 and 86 lanes, d -0.21 and -0.18), lps0 s0 and s1 (54 and 48 lanes, d -0.08 and -0.02), lp1 s0 (85 lanes, d -0.22).
   - Borderline (2 runs): glass4 lps0 s0 (665 lanes, d exactly 0.050) and radiationm lp1 s1 (1 lane, d +0.075, within SCIP's seed spread).
   - The "perfect" split comes from looking only at lp1 and lps1, the two configurations picked after the dev results were in.
   - P3 ("where cuOpt stalls and D1 holds, we win") is broken, marginally, by glass4 under lps0: d against cuOpt -0.024 and -0.068 on both seeds, with 145-665 lanes.

5. The pre-run version of D1 misclassifies radiationm. Its LP point arrives at 1.0 s, rounding it violates 2.5 % of rows (under the 20 % cut), and SCIP supplies an incumbent, so both clauses say "foothold". The run shows 0-2 feasible lanes. The rule covers this with class II, a class of 2 instances. P3 also says "a quarter of the budget" where D1 says "a third".

6. Which instances count as losses depends on the configuration. With lp1, lps1 or ad1 the losses are {fhnw, radiationm, rail01}. The default A3 adds ca (d against cuOpt -0.35). lps0 adds glass4.

7. Seed stability.
   - Our side, against cuOpt: both seeds give the same sign on all 9 instances. But glass4 under lp1 wins by only +0.023 / +0.031.
   - cuOpt has one seed. rail01's "loss" (d -0.073) rests on one cuOpt incumbent at 107.5 s; cuOpt found nothing on rail01 within 60 s in the optopt run. One failed cuOpt seed would make rail01 a tie, leaving a single real member in class II (radiationm).
   - Against SCIP with lp1, comparing runs of the same seed: radiationm goes -0.07 / +0.08 and sing44 -0.22 / +0.40, so their signs flip between seeds. "Better on 7 of 7, p = 0.016" therefore rests on 5 instances whose sign holds on both seeds; the Wilcoxon test on those 5 gives p = 0.0625. For lps1, radiationm (-0.06 / +0.00) and fhnw (+0.70 / 0.00) are seed-dependent; the remaining 6 give p = 0.031.
   - SCIP alone on radiationm gives 0.154 / 0.304, a larger spread than any arm effect.

8. The broader-data test is not an out-of-sample test of this claim.
   - (a) It contains no EA runs. The outcome there is cuOpt's own P(60) from one seed, so D1, "class II → no gain for us" and "class → ours against cuOpt" are untested outside the 9 instances that set the cuts.
   - (b) The reported p = 0.06 belongs to a different outcome: cuOpt P(60) ≤ 0.15 (0 of 11 class I against 5 of 16 others). Under D2's own definition (gap ≤ 0.05 at 10 s) it is 0 of 11 against 3 of 16, p = 0.25.
   - (c) Across SCIP-hardness cuts of 0.3, 0.4, 0.5, 0.6 and 0.7, the p-value for that outcome is 0.25, 0.35, 0.06, 0.12 and 0.26. The reported value is the minimum, and D2's p is about 0.25 at every cut.
   - (d) On all 80 MIPLIB instances, class I is more often cuOpt-strong (24 of 42 against 17 of 38) and more often closes early (12 of 42 against 8 of 38, p = 0.61). The sign appears only after conditioning on SCIP-hard instances, and that condition was added after the unconditional test came out null. About 16 features were scanned.
   - (e) "cuOpt P(60) ≥ 0.2 on all 11 class I" rests on opm2-z10-s4 at 0.202, a margin of 0.002.
   - (f) Class II's "6 of 6 strong or nobody" includes rail01, which is a dev instance, and radiationm40-10-02, a sibling of the dev radiationm. Against a base rate of 43 of 74 for the other instances, Fisher p = 0.077; without rail01 p = 0.15.
   - (g) The family checks use the dev instances' own families (item placement 100 instances, PGLib ca 20), so each adds one effective data point, not a class-level test. rts_gmlc is class I, yet cuOpt is within 0.02 of optimal by 60 s on 12 of 12; the rule had to name it as an exception in advance.

9. A small point. "Mean effect +0.43" follows D-002's rule of excluding instances where both arms have P = 1 on every seed (7 instances for lp1, 8 for lps1). Over all 9 instances the means are +0.33 (lp1) and +0.38 (lps1); state which one is quoted.

**What can be claimed instead.** What can honestly be said.

1. No model property we have found separates where cuOpt beats us. On 9 instances, the structural separators (lp_frac_int_n, int_vars_in_rigid_eq3_share, objective density, two-variable chain share) are as frequent and as strong as chance and post-hoc selection produce: 48 % of random loss sets have a perfect separator, a two-feature OR rule fits 81 of 84, and a best |rho| of 0.86 or more arises with p = 0.155. The three classes are a hypothesis to test, not a finding.

2. Descriptively, inside the runs, our gains go with the EA getting a foothold of its own. In 128 of 140 coupled dev runs, at least 10 repair lanes ending feasible goes with a gain over SCIP of more than 0.05. The exceptions:
   - fhnw: one feasible lane can be the solution;
   - ca under lps0;
   - sing44, where the EA has a foothold yet loses to SCIP under A3, lps0 and lp1 seed 0.
   This describes how the gain happens; it does not predict it.

3. We lose to cuOpt exactly where cuOpt's heuristic phase nearly solves the instance early (fhnw, radiationm). That follows from the arithmetic: if cuOpt's gap is at most 0.05 by 10 s, its P(120) is at most 0.13. rail01 is a near-tie (d -0.07) resting on one cuOpt seed. With lp1, lps1 or ad1 the win/loss labels against cuOpt keep the same sign on both of our seeds. But the set of losses depends on the configuration (the default A3 also loses ca, lps0 also loses glass4), and glass4 under lp1 wins by only 2-3 %.

4. Against SCIP, the strong gains (d of 0.49-0.97, stable across seeds) are on ca, loue, neos-960392 and item_placement. On radiationm and sing44 (lp1) the sign depends on the seed. Counting only sign-stable instances, lp1 against SCIP gives p = 0.0625 (n = 5) and lps1 p = 0.031 (n = 6).

5. In the predecessor optopt data (cuOpt against SCIP, 60 s, one seed, no EA): on SCIP-hard MIPLIB instances, a dense objective goes with cuOpt not being strong (0 of 11 with cuOpt P(60) ≤ 0.15, against 5 of 16 others, p = 0.06). That holds only at the SCIP-hardness cut of 0.5, under that outcome definition, and after scanning about 16 features. With D2's own definition it is p = 0.25. On all 80 instances the direction reverses. Call it a lead, not support.

6. An out-of-sample test would need:
   - (a) the class rules and the D1 and D2 cuts frozen in a dated note before any run;
   - (b) A0, A6 with at least 2 cuOpt seeds, and one pre-chosen configuration of ours (not the better of lp1 and lps1 per instance), on instances from families not in the dev set (not item placement, not PGLib ca, not rail01 or radiationm);
   - (c) structural features scored only on those new runs.
   The proposed test (a) on item placement 9925-9999 and PGLib ferc only checks generalisation within families already seen.

Supporting files: /tmp/itrea-classes/sk/enum.py, rho.py, seeds.py, lanes.py and lanes_all_arms.csv (feasible lanes per run for all 140 coupled dev runs).

**Verdict: does not hold.**

**Problems.** The per-instance accounts of who found what and when mostly match the traces. The characterisation built on them does not hold. Its central run-time separator is a restatement of the outcome, its pre-run proxy and prediction P4 already fail on the nine instances it was fitted to, and three of its mechanism claims contradict the traces or the model. Most severe first:

1. **D1 is the outcome restated, not a diagnostic.**
   - The config sets `repair.objective_mode: cutoff_row`, and `ls_fj_gpu.set_cutoff` puts a cutoff row at min(population best, SCIP incumbent) − δ, with δ ≥ 1 unit when the objective is integral. Once any incumbent exists, a lane can only end "feasible" if it is strictly better than that incumbent.
   - So the lane count is the EA's incumbent count. Across 36 coupled runs (lp1/lps1), Spearman rho between feasible lanes and posts is 0.92, and posts = ea_own_n_inc in every run. Examples: glass4 lp1 779 lanes / 307 posts; item_placement 12,322 / 1,901; radiationm 0–2 / 0–2.
   - "D1 separates wins from losses perfectly" therefore says "we win where the EA finds its own incumbents."
   - It does not track gain size either:
     - glass4 lp1: 262–779 lanes, but the EA's share of the P reduction is 0.00002–0.0006.
     - sing44 lp1 seed 0: 85 lanes, yet P 0.668 is worse than SCIP alone (0.518).
     - fhnw lps1 seed 0: 1 lane, yet P 0.302 (d +0.70 for that run).

2. **The pre-run D1 proxy predicts a foothold on radiationm, which has none.** All three of its conditions hold there: the LP arrives at 1.0 s, rounding it violates 2.5 % of rows, and SCIP injects 5–18 incumbents. Actual result: 0–2 lanes per run, and the EA alone (A5, every config) is never feasible.
   - On neos-960392 the foothold does not come from the LP at all. The EA's own first solution arrives at 0.54 s, before its LP at 4.3 s, and the EA alone with no LP and no SCIP (A5) reaches P 0.071.

3. **P4 is falsified on its own fitting set.** It says d ≥ 0.3 only where SCIP's P ≥ about 0.6 and lanes ≥ 10. Counterexamples:
   - glass4 lps1: d 0.34 with SCIP P 0.344.
   - glass4 A3@rc0: P 0.145, so d 0.58.
   - sing44 lps1: d 0.31 with SCIP P 0.525.
   - fhnw lps1: d 0.35 with 0–1 lanes.

4. **The cuOpt mechanism is contradicted by cuOpt's own traces.** The claim is that cuOpt closes early only where finding a feasible point nearly solves the problem (citing radiationm's tight LP).
   - radiationm: cuOpt's first incumbent has gap 0.72 at 0.80 s. Then come 12 improving incumbents, 11 of them before its presolve ends at 3.19 s: gap 0.10 at 1.18 s, 0.053 at 1.87 s, 0.0003 at 3.21 s. This is an improvement phase that runs before presolve finishes, so it is not the PDLP pump and not its B&B (48 nodes). The trace does not name the component.
   - fhnw: optimal at 0.56 s, against a presolve time of 1.96 s.
   - So on the very instance where our repair produces 0 improving children, a cuOpt heuristic phase that runs before presolve does improve. That points at our operator or configuration, not at a structure that defeats local heuristics.

5. **"Dense objective gives a poor first cuOpt solution" is contradicted.** First-incumbent gaps: neos-960392 0.99 (objective density F1 0.027), item_placement 0.96 (F1 0.030), glass4 0.68 (F1 0.059), radiationm 0.72 (F1 about 0). The first solution is poor everywhere except fhnw, whose objective is empty.

6. **D2 has a hole, and "stall" is mislabelled.**
   - ca's first cuOpt incumbent at 24.06 s fits no bucket: "stall" requires a first solution within 5 s, "late" requires one after 30 s. The text calls it "late + stall".
   - ca, neos-960392, sing44 and rail01 are latency cases, not stalls. cuOpt's presolve takes 26.2 / 23.6 / 23.5 / 24.1 s, it explores 0 B&B nodes, and its next improvements come at 77.5 / 77.7 / 64.0 / 107.5 s. It reaches gap 0.01 on ca (at 77.6 s) and the optimum on neos-960392 (at 90.7 s).
   - Genuine stalls are only glass4, item_placement and loue: presolve 0.4–1.4 s, then 617k / 67k / 15k nodes, final gaps 0.25 / 0.65 / 0.75.
   - Holding each run's final gap flat to the pre-registered 300 s horizon, we still lead on ca (0.124 vs 0.265) and neos-960392 (0.011 vs 0.228). The sign of the comparison holds; the label does not.

7. **Class I is no better a marker than simpler alternatives, and the test was asymmetric.** On the 27 SCIP-hard optopt instances:
   - An almost empty objective (obj_nnz_share < 0.01) separates cuOpt-strong instances (P(60) < 0.2): 4 of 9 against 1 of 18, Fisher p = 0.03.
   - Size (log10 nnz ≥ 5) separates them at p = 0.047.
   - Class I gives 0 of 11 against 5 of 16, p = 0.06.
   - The sparse-objective marker was declared "not supported" using an unconditioned test, while class I was conditioned on SCIP-hardness.
   - On the dev set, cuOpt's two early closes are exactly the two instances with at most one objective non-zero (fhnw 0, radiationm 1).
   - Two of the three dev class I members (ca with 305k variables, sing44 with 59k) are cuOpt latency cases; only loue is a genuine stall.

8. **radiationm's "min-max objective" is contradicted by the model, and class II has no shared mechanism.**
   - The objective variable X_INTRODUCED_3901_ has column degree 1. Its only row is the equality obj = 325·Beamtime + K, and Beamtime and K each appear in one 13-variable equality. The top of the objective is an exact integer sum; there are no rows that "lowering it by 1" would tighten.
   - The figure "154 rows over ≥ 9 general integers" is the number of rows still violated in the aggregation probe, not a structural count. The structure has 326 such rows, and loue has 105 and is a win.
   - The dev log's aggregation probe already shows the two-variable chains are not radiationm's blocker. rail01 fails on LP time: HiGHS needs 93 s, the EA's LP cap is 60 s.
   - So class II is two instances with two different mechanisms.

9. **A simpler explanation covers two of the six wins (ca, sing44): HiGHS finishes the LP before SCIP's root LP does.**
   - ca: the EA's LP arrives at 29–34 s; SCIP's root LP never finishes in 120 s.
   - sing44: the EA's LP arrives at 10.3–13.0 s; SCIP's root LP finishes at 33 s alone and 35–51 s coupled. SCIP's big step is pinned to root-LP completion: step at 33.1 s with the LP at 33.1 s alone, step at 51.2 s with the LP at 51.1 s in lp1 seed 0.
   - Without the EA's own LP (A3 default), P is 1.000 on ca and 0.651 on sing44.
   - The EA alone with its own LP and deep repair (A5@c03; one seed, a different window) reaches 0.232 on ca, better than every coupled arm (best 0.296). Coupling adds nothing there.
   - This explanation does not cover neos-960392 or loue.

10. **The sing44 lp1 seed-0 loss is misattributed.**
    - SCIP's first-solution delay (3.65 s to 7.9 s, while the gap only moves 1.0 to 0.985) costs about 0.0005 in P.
    - The root-LP delay (33 s to 51 s) moves the 0.985 → 0.564 step 18 s later, which costs about 0.06.
    - Seed 1 had the same 7.8 s first solution and won. What separates the seeds is the EA's own first solution: 19.7 s against 59.3 s.

11. **fhnw's class membership is a configuration choice.**
    - With the pure rhs row scale it is solved 4 of 5 times (dev log; A3@c04 3/4, A5@c04 1/1). A5@c02, which uses the rhs scale with the LP and deep repair, also solves it, with the EA alone at 0.196.
    - D1 "fails" there only under the l2 scale.

12. **item_placement's "fast generations beat deep repair by 2.6×" isolates the wrong variable.**
    - Every arm with stall_steps 1000 lands at 0.49–0.58: lps1 0.569, cap4k 0.530, ad1 0.522, rc0 0.521, lps1w 0.490. The step cap (4,096 vs 20,000) changes nothing.
    - lp1 vs lps1 also changes the adaptive strategy and the stream.
    - lp1 vs A3 differ only in LP source (0.218 vs 0.400) on an instance whose LP bound is useless. That difference is within A3's seed spread (0.277 / 0.522).

13. **The gain formula omits a negative interference term.** SCIP sits on core 5 in every arm and the EA on cores 1–3, yet SCIP's presolve slows as the EA runs more generations:
    - ca: 31 s alone, 46.6 s at 18 generations (lps1), 59–61 s at 79–95 generations (lp1).
    - rail01: 8.3 s alone, 15.4 s at about 680 generations, 18.7 s at about 872.
    - The direction is consistent with memory-bandwidth contention, but the runs come from different windows, so this is uncontrolled.

14. **Minor numbers.**
    - radiationm injections are 5–18, not 9–17.
    - The EA's incumbent hold is 0.37 on sing44 lp1 seed 0 and 0.47 on ca lp1 seed 0, so not "49–99 % on every win".
    - The glass4 lps1 "chain" (EA step, then SCIP extends it) is plausible but has no lineage evidence: the SCIP-on-EA share is 0.04–0.07, and SCIP alone already varies by seed (gap 0.314 / 0.351 at 60 s).

**What can be claimed instead.** These are descriptive only: 9 instances, 2 seeds, cuOpt with 1 seed. No pre-run structural rule that separates the instances is established.

**Against SCIP alone: the EA gains wherever it produces incumbents before SCIP does. The mechanism differs by instance:**
- **ca and sing44:** the EA's HiGHS LP finishes before SCIP's root LP, and rounding plus repair turns it into an incumbent. Deep repair converts it sooner: on ca the first feasible point arrives at about 35 s against 58–64 s.
  - On ca the EA alone does as well (A5@c03 0.232, one seed), so this is not a coupling effect.
- **neos-960392:** feasibility jump from random seeds finds a solution at 0.54 s; no LP is needed (A5 0.071).
  - SCIP's root bound equals the optimum at 10.6 s, but it never leaves the root or finds a non-trivial solution.
- **loue:** an LP point (the EA's or SCIP's) plus repair. SCIP's own heuristics find nothing despite 1.6k nodes.
  - Coupled runs (0.33–0.41) beat the EA alone (0.52–0.89).
- **glass4 and item_placement:** the EA polishes SCIP's incumbent; under the default configuration the EA alone never becomes feasible on glass4.
  - item_placement's result depends on the stall rule (stall_steps 1000 hurts).
  - glass4 lps1's gain shows up mostly in SCIP's own steps (the EA's direct share is 0.05).
- **Interference:** running the EA beside SCIP appears to delay SCIP's presolve and root LP (direction consistent, uncontrolled). On sing44 seed 0 that delay outweighs the EA.

**The three losses:**
- **radiationm:** our repair produces no improving child (0–2 lanes per run) and the EA alone is never feasible. Yet cuOpt's phase that runs before its presolve ends reaches gap 0.053 by 1.87 s. This is an operator gap on our side, not a structural impossibility.
- **fhnw:** a row-scale configuration issue; the pure rhs scale solves it 4 of 5 times.
- **rail01:** LP time (HiGHS 93 s against the 60 s cap). Nobody does well there.

**Against cuOpt alone: cuOpt's own trace shows three regimes:**
- **Early close:** fhnw and radiationm. Both have at most one objective non-zero, and both close before cuOpt's presolve ends. We lose both.
- **Latency:** ca, neos-960392, sing44 and rail01. Presolve takes 23–26 s, cuOpt explores 0 nodes, and improvements come at 64–108 s. We win all except rail01.
- **Genuine stall:** glass4, item_placement and loue, with fast presolve and many nodes. We win all three.

**Candidates worth testing out of sample (all post hoc):**
- HiGHS LP time against SCIP's root-LP time, both measurable from SCIP-alone runs.
- cuOpt's presolve-end time and whether the objective is nearly empty.
- A short EA-alone feasibility probe on the chosen row scale.

On SCIP-hard optopt instances, a nearly empty objective (p = 0.03) and instance size (p = 0.047) predict cuOpt's strength at least as well as class I does (p = 0.06).

## The characterisation that was attacked (verbatim)

**characterisation.** Short answer: yes. The split is clear on the run-time side and only suggestive on the structural side.

The rule uses two structural features, read from the original MPS with no presolve, which is what the EA sees:
- **F1, objective density** = obj_nnz / n.
- **F2, chain share** = share of rows that are unit two-variable equalities (x = ±y + c).

They give three classes. The cuts were drawn from the gaps in the 9 instances: F1 sits at 0.134 or below versus 0.383 or above; F2 sits at 0.041 or below versus 0.444 or above.
- **Class II, chained** (F2 ≥ 0.20): radiationm (0.63), rail01 (0.44).
- **Class I, dense objective** (F1 ≥ 0.25 and F2 < 0.20): ca (0.38), sing44 (0.44), loue (0.94).
- **Class III, sparse objective** (F1 < 0.25 and F2 < 0.20): glass4 (0.059), item_placement (0.030), neos-960392 (0.027), fhnw (0).

Two cheap run-time diagnostics complete the rule:
- **D1, EA foothold.** In a run: at least 10 repair lanes end feasible per run. Before a run: the EA's LP point arrives inside about a third of the budget, and rounding it violates no more than 20 % of rows, or SCIP supplies an incumbent.
- **D2, cuOpt mode.** "Early close" means cuOpt-alone gap ≤ 0.05 at 10 s. "Stall" means a first solution within 5 s and gap ≥ 0.1 at 60 s. "Late/none" means a first solution after 30 s, or none.

**(1) Versus SCIP alone: gain ≈ SCIP primal headroom × D1.**
- Large gains (best-config d 0.49–0.97) occur only where SCIP alone has P(120) ≥ 0.66 and D1 holds: ca, loue, neos-960392, item_placement.
- Modest gains occur where SCIP P is 0.34–0.53 and D1 holds: glass4 +0.13/+0.34, sing44 +0.06/+0.31 (lp1/lps1).
- There is no gain where D1 fails: radiationm, rail01, and fhnw except with deep repair, where one of two lps1 seeds solved it.
- D1 separates the instances perfectly. Every win has 10 or more lanes ending feasible per run, and the EA's own feasible point appears in 24 of 24 runs. Every non-gain has at most 2 lanes per run, and an own point appears in only 3 of 12 runs, late (36 s, 68 s, 110 s).
- Class II predicts "D1 fails" (2 of 2). Within class III, D1 fails only on fhnw, whose LP rounding violation is 0.47 (others 0.11 or less).

**(2) Versus cuOpt alone: the outcome follows cuOpt's mode (D2), not our structure.**
- Where we have a foothold, our P is about 0.03–0.57. cuOpt is the bimodal one.
- We lose 25–200x where cuOpt closes early: fhnw (optimal at 0.56 s) and radiationm (gap 0.0003 at 3.2 s).
- We win by 15–95 % on all six instances where cuOpt stalls or comes late (gap at 60 s 0.32–0.99).
- rail01 is a near-tie where nobody does well.
- Structural marker for D2, tested on broader data: class I means cuOpt does not close early (dev 3 of 3; optopt SCIP-hard MIPLIB 11 of 11). Class II means cuOpt is strong or nobody finds anything (dev 2 of 2; optopt 6 of 6, never a stall). Class III is bimodal, so it needs the D2 probe.

**Status of each claim.**
- Supported on the dev set: the headroom × D1 split versus SCIP; D1 and D2 as perfect run-time separators.
- Suggestive: class I → no cuOpt early close (Fisher p = 0.06, n = 27, one seed, 60 s, found during a scan of about 16 features); class II → no gain for us (n = 2).
- Not supported: objective with at most 1 non-zero as a cuOpt-strong marker; continuous share; LP tightness.
- Class I does NOT mean cuOpt is worse there on average. Class III's median cuOpt P is higher (0.91 against 0.71); class I is the reliably non-early-closing group.

**mechanism.** **How the EA gains over SCIP.** It acts as an independent primal source wherever SCIP's primal side is gated on root processing or finds nothing.
- **ca:** SCIP stays at 1 node and has no root-LP bound within 120 s. The EA's own HiGHS LP arrives at 29–34 s, and its first feasible point (35 s in lps1, 58–64 s in lp1) is the first incumbent of the whole run. The EA accounts for 100 % of the P reduction.
- **neos-960392:** SCIP holds only the trivial 0 and stays at 1 node, even though its root bound equals the optimum at 10.7 s. The EA finds a real solution at 0.54 s, from random seeds plus repair, and reaches the optimum by 60 s.
- **loue:** SCIP alone never finds a solution. The EA's first point arrives at 2–10 s and its share of the P reduction is 0.84–0.92.
- **Where SCIP finds first, the EA polishes.** item_placement: 371–439 accepted posts, EA holds the incumbent 98.5 % of the time, P 0.218. glass4: 290–306 accepted posts in lp1, plus a few large lps1 steps that SCIP then extended. sing44: EA share 0.84–0.94 in lps1.
- SCIP rejected no posts anywhere.

**Why the EA fails, when it does.** The three non-gains fail for three different reasons. That is why a single structural rule fails, and why class II is a marker rather than the cause.
- **rail01:** HiGHS needs 93 s on one thread (124k iterations). In the run the 60 s LP cap is hit, so there is no LP point and 0 feasible lanes. SCIP's root LP does not finish when coupled.
- **fhnw:** The objective is empty, so the LP point gives no direction, and rounding it violates 47 % of rows. Under the l2 row scale at least 99.97 % of lanes end by stall. The pure rhs row scale solved 4 of 5 runs (dev log), so the obstacle is the row scaling of big-M rows.
- **radiationm:** The EA receives 9–17 injected SCIP incumbents per run, yet only about 1e-5 of lanes end feasible. The aggregation probe found 0 of 32 children feasible on both the original and the aggregated model. The blocker is 154 equality rows over 9 or more general integers, plus a min-max objective that is one integer variable — not the chains.
- The literature's class weaknesses of feasibility jump are not reproduced here: loue (83 % precedence rows) and sing44 (58 % variable-bound rows) are wins. The LP point and the SCIP partner supply the start that plain feasibility jump lacks.

**How cuOpt behaves.** Its GPU heuristics work as a feasibility engine: feasibility jump at launch, then a feasibility pump with PDLP and propagation rounding after presolve.
- With a dense objective, its first solution is poor: gap 0.93–0.99 at the first incumbent on loue, neos-960392, sing44, ca and item_placement. It runs 0 B&B nodes on ca, neos-960392 and sing44, and its gap at 60 s is still 0.32–0.99 on all six wins. That is the stall.
- It closes early only where finding a feasible point nearly solves the problem. fhnw has an empty objective, and the first incumbent at 0.56 s is optimal. radiationm has a tight LP (0.07 %), and cuOpt has gap 0.0003 at 3.2 s.
- Our advantage is therefore an improvement-phase advantage: the EA holds the incumbent 49–99 % of the time on the wins.
- Objective density is a marker on cuOpt's side, not ours. glass4, item_placement and neos-960392 have sparse objectives (on continuous variables, or 3 % of binaries) and the EA still wins through many small accepted posts.

**per_instance_fit.** All 9 fit, but the fit is post hoc: the cuts come from these same 9 instances. P(120) is given as lp1 / lps1; d is positive when ours is better.

**Class I**
- **ca:** fits. SCIP has no incumbent (P 1.0), so headroom is large. D1 holds (10–32 feasible lanes per run). cuOpt's first incumbent comes at 24 s with gap 0.994 at 60 s (late + stall). P 0.509 / 0.300; d vs SCIP +0.49 / +0.70; d vs cuOpt +0.21 / +0.54.
- **loue:** fits. SCIP has no solution (P 1.0). D1 holds (107–329 lanes). cuOpt stalls (gap 0.82 at 10 s, 0.75 at 60 s). d vs SCIP +0.63; d vs cuOpt +0.51. lps1 seeds 0.332 / 0.407.
- **sing44:** fits. Headroom is only moderate (SCIP P 0.525). D1 holds (28–153 lanes). cuOpt stalls (gap 0.978 at 60 s). d vs SCIP +0.06 / +0.31; d vs cuOpt +0.32 / +0.50. The lp1 result rests on one seed: 0.668 / 0.317, with seed 0 worse than SCIP alone (likely interference: SCIP's first solution came at 7.9 s instead of 3.65 s).

**Class II**
- **radiationm:** fits (no gain, cuOpt strong). d vs SCIP +0.02 / −0.02, below SCIP's seed variance (0.154 / 0.304). D1 fails (0–2 lanes). cuOpt closes early (gap 0.0003 at 3.2 s). d vs cuOpt −0.96.
- **rail01:** fits (no gain; cuOpt in its "nobody" branch). D1 fails because there is no LP point (0 lanes). cuOpt's first and only incumbent comes at 107.5 s with gap 0.30. d vs SCIP 0; d vs cuOpt −0.07.

**Class III**
- **glass4:** fits. SCIP P 0.344. D1 holds (51–779 lanes). cuOpt stalls (gap 0.32 at 60 s). d vs SCIP +0.13 / +0.34; d vs cuOpt +0.03 / +0.26.
- **item_placement:** fits. SCIP P 0.66. D1 holds (10.8k–12.3k lanes). cuOpt stalls (gap 0.79 at 10 s), and stalls family-wide in optopt. d vs SCIP +0.67 / +0.14; d vs cuOpt +0.67 / +0.15.
- **neos-960392:** fits. SCIP P 1.0 (trivial 0 only, stuck at the root). D1 holds (186–268 lanes). cuOpt reaches only gap 0.99 at 10 s and the optimum at 91 s. d vs SCIP +0.97; d vs cuOpt +0.95.
- **fhnw:** fits the class III branch where cuOpt closes early and the EA has no foothold. D1 fails (0–1 lanes; LP rounding violation 0.47 > 0.20). cuOpt is optimal at 0.56 s. d vs cuOpt −0.99. d vs SCIP is 0 for lp1 and +0.35 for lps1, which rests on one seed (0.302 / 1.000). Pure rhs repair solves 4 of 5 runs.

**Two instances sit near the cuts.**
- rail01's objective density (0.134) is the closest sparse value to the cut, but its chain share puts it in class II regardless.
- sing44's chain share (0.041) is the highest among the unchained instances.

**predictions.** Thresholds are stated verbatim as tested: obj_nnz/n ≥ 0.25; unit two-variable equality share ≥ 0.20; cuOpt gap ≤ 0.05 at 10 s; EA feasible lanes ≥ 10 per run; LP rounding violation > 0.20. Read the predictions at the same 120 s horizon.

**P1 — class I, SCIP-hard.**
- cuOpt-alone gap at 10 s stays above 0.05 on every such instance.
- The coupled system (lp1 or lps1) beats cuOpt where cuOpt-alone P(120) ≥ about 0.45 and EA lanes ≥ 10.
- Cases with cuOpt P 0.2–0.45 are toss-ups by our own numbers (our P was 0.30–0.37 on dev class I).
- rts_gmlc is the likely counter-case: in optopt, cuOpt is within 0.02 of optimal by 60 s on 12 of 12. Expect a tie or a loss against cuOpt there, despite class I.
- PGLib ca-family: beat both SCIP and cuOpt.
- Falsified by a cuOpt early close on any class I instance, or by losing to cuOpt on most class I instances with cuOpt P ≥ 0.45.

**P2 — class II.**
- |d vs SCIP| ≤ 0.05, d vs cuOpt ≤ 0 on at least 3 of 4 instances, and EA lanes ≤ 2 per run.
- cuOpt is either strong (P ≤ 0.15) or finds nothing — no stall mode.

**P3 — class III.**
- The sign of d vs cuOpt follows D2: if cuOpt's gap at 10 s is ≤ 0.05, we lose; otherwise we win when D1 holds.
- D1 fails where LP rounding violation > 0.20 or the LP point needs more than about a quarter of the budget.

**P4 — versus SCIP.**
- d ≥ 0.3 only where SCIP P ≥ about 0.6 and EA lanes ≥ 10.
- ML4CO item placement (class III, cuOpt stalls family-wide): positive against cuOpt, but a smaller gain against SCIP than 9903's +0.67, because only 1 of 60 candidates has SCIP stage-1 P(300) ≥ 0.5.

**Concrete tests.**
- **(a) Now, development-legal.** Use only non-candidates, because MIPLIB candidates feed the test set (D-010).
  - Instances: 4 ML4CO item placement instances from valid/ outside the first 25 (item_placement_9925–9999; e.g. 9950: class III, F1 0.030, F2 0), and 2–4 PGLib ferc instances (exploratory family; class I by structure).
  - Arms: A0, A6 (1 seed) and A3@lp1 × 2 seeds, 120 s.
  - Predictions: item placement → lp1 beats cuOpt on at least 3 of 4. ferc is 471k variables and 2M non-zeros, and in optopt neither cuOpt (23/24) nor SCIP (24/24) finds a solution in 60 s. It therefore tests the LP-arrival part of D1 (rail01-type), not class I: no gain unless the in-run LP arrives within about 40 s.
- **(b) Before E2.** When pool-v1 is frozen, compute F1 and F2 read-only for every test instance and record P1–P4 in a dated decision note. This is structure, not selection: the quota rule is unchanged. Evaluate the classes as exploratory strata after E2/E3, and add a 10 s cuOpt gap column from the A6 runs as D2.

**test_on_broader_data.** **What was run.** Everything was CPU-only, pinned to cores 10–14 with single threads, read-only on both repositories, with no solver runs.
- Data: the optopt runs.parquet tables — C0 = cuOpt 26.08 as shipped, ONE seed (seed 0; the vault note's "5 seeds" is wrong for cuOpt), 60 s for MIPLIB and PGLib, 30 s for ML4CO — and S0 = SCIP.
- Structural features (features.cheap) were computed for all 80 optopt MIPLIB instances (each at most 16.7 s), plus representative rts_gmlc, ferc, ca and item placement files.

**1. An objective with at most 1 non-zero does NOT predict cuOpt strength.** 1 of 12 such instances has cuOpt P(60) ≤ 0.05, against 15 of 68 others (Fisher p = 0.44). Most are mid or no-solution.

**2. Across all 80 MIPLIB instances, class I versus the rest shows no relation.** Early close: 12/42 vs 8/38, p = 0.61. SCIP-easy instances are easy for cuOpt too, so the marker is conditional on SCIP-hardness.

**3. On SCIP-hard MIPLIB instances (SCIP P(60) ≥ 0.5, n = 27), class I is reliably non-early-closing.**
- Class I (11): 0 close early. cuOpt P(60) ≥ 0.2 on all 11: 0.20 opm2-z10-s4, 0.22 s100, 0.47 h80x6320d, 0.49 neos-1456979, 0.68–1.0 for the rest; median 0.71.
- Other classes (16): 5 have cuOpt P ≤ 0.15 (krka 0.010, net12 0.063, crna 0.065, radiationm40-10-02 0.107 against SCIP 0.62, kumeu 0.146).
- Fisher p = 0.06.
- The pattern holds for every objective-density cut from 0.1 to 0.5: the dense group never has cuOpt P(60) < 0.2.
- Caveat: this came out of a scan of about 16 features. Some small-n comparisons gave p < 0.05 by chance (share of general integers, integer range), so treat this as suggestive.
- Class III on SCIP-hard instances is bimodal: 3 of 12 strong, 9 of 12 poor or none.

**4. Class II is "cuOpt strong or nobody" in 6 of 6 MIPLIB instances, with no stalls.** Strong: crna 0.065, murg 0.038, radiationm40 0.107, traininstance6 0.088. Nobody: rail01, neos-2746589-doon.

**5. Family level.**
- **Item placement (class III, F1 0.030):** cuOpt stalls on 80 of 100 and is within 1 % at 10 s on 0 of 100. Its P(30) median is 0.47 against SCIP's 0.36.
- **PGLib-UC (class I: F1 0.26–0.38, F2 ≤ 0.06):**
  - ca family: cuOpt's first solution at about 19 s, gap 0.99 at 60 s (20/20); SCIP finds nothing in 60 s (20/20).
  - ferc: cuOpt finds nothing on 23 of 24, SCIP on 24 of 24.
  - rts_gmlc: cuOpt's first solution at 0.5–1.3 s, gap 0.29–0.55 at 10 s, but ≤ 0.02 by 60 s on 12 of 12. cuOpt P(60) 0.13–0.41 against SCIP 0.16–0.39. So "no early close at 10 s" can still mean a cuOpt win at 120 s.

**6. Screening headroom (SCIP stage 1, P(300)).** P ≥ 0.5 on 73 of 232 MIPLIB candidates (34 with no solution), 20 of 32 PGLib (19 with no solution), and 1 of 60 ML4CO.

**Caveats.** cuOpt here is P(60) with one seed, against P(120) on the dev set. supportcase22 (not a candidate) is included as background only. No EA runs exist outside the dev set, so the EA side (D1, class II → no gain) is not tested beyond 9 instances.

**Files**, all under /tmp/itrea-classes/:
- classes_dev.csv
- optopt_miplib80_classes.csv
- optopt_miplib80_features_modes.csv
- objdensity_sweep.csv
- optopt_cuopt_modes.csv
- optopt_family_ml4co_item_placement.csv, optopt_family_pglib_uc.csv
- cheap80.csv, cheap_families.csv
- scripts: modes_check.py, objdensity_check.py, class_test.py, classes_dev.py, cheap80.py, cheap_fam.py, family_check.py
