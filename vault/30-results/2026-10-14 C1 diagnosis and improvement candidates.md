# C1 diagnosis and improvement candidates (2026-10-14)

**Status:** analysis, not a decision; candidates for D-024 targeted tests (2026-10-14). Every number is development data
(D-010), descriptive and **INTERIM, n = 2 seeds** unless a line says otherwise. Nothing here says that A9 or any
candidate "improves", "beats" or "matches" anything. Each candidate becomes a test only through its own decision record.

**Question (Berk, 2026-10-14 14:4x):** "What are you waiting for to improve more?" The answer came from the workflow
`itrea-improvement-candidates`: Claude Code subagents, read-only on the repository, CPU cores 10–14 at nice 19, no GPU,
run while W-1014's `backoff2` pool was running. It had five stages: three analyses (why A9 loses on C1, where its gains
come from, the lever inventory), one candidate stage, and two adversarial reviews. This note applies the reviewers'
fixes and drops the one candidate a reviewer rejected.

Inputs:
- `datasets/runs.parquet` (rebuilt 2026-10-13 22:2x; holds every itx-r13 run) and `datasets/incumbents_checked.parquet`.
- `traces/raw/itx-r13/<run>/` (`run.json`, `ea.summary.json`, `ea.events.jsonl`, `ea.genealogy.npz`, `*.incumbents.npz`,
  `solver.*`, `cuopt.*`), plus the earlier sets dev, classprobe, a9pilot, backoff0 and backoff1.
- `configs/`, `src/itrea/`, and decisions D-001 to D-024.

Related: [[D-013 Aim — lift each solver's weakest cases]],
[[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]] (C1's
interim hint 0.827 and guard +0.063), [[D-023 A9 back-off, phase Y only — yield until cuOpt's first incumbent, then full duty]]
(being run today, no run read),
[[D-024 Targeted first, wider only on a hint — staged testing for development and v1]],
[[Where the coupling gains — instance classes (2026-10-07)]].

**Arms used below.**
- **A9@r13x**: SCIP ↔ EA ← cuOpt on F, harness ext, run hash 28e5a5e9c725, EA config 1d12ad490dd0.
- **A6F@r13x**: cuOpt alone on F (5 cores), run hash 82a371305f88.
- **A0**: SCIP alone, run hash 27f3171fac78, sets dev / classprobe, 10-06 / 10-07.
- **A6**: cuOpt alone on 3 threads (cores 1–3) under the OLD harness: MIP gap 1e-4, OMP = 1, RMM 12 GB, 10-06 to 10-11.
- **A2**: SCIP + 3-thread cuOpt, no EA (a9pilot).
- **A3 / A3x**: SCIP ↔ EA without cuOpt.

P = P(120) on the call clock; lower is better. d = bounded d on seed means; + means A9 is better.

## 1. What C1 shows per instance

C1 is D-022's set itx-r13 plus itx-r13b: 120 s, seeds 0–1. A9@r13x and A6F@r13x ran on 2026-10-13 between 15:16 and
19:28 CEST. A0 comes from the earlier same-budget runs, and each row names its set.

ΔP = mean P(A9) − mean P(A6F), so **+ means A9 is worse**.

| # | instance | A9@r13x P s0 / s1 (mean) | A6F@r13x P s0 / s1 (mean) | A0 P s0 / s1 (mean) [set] | t_first s0 / s1: A9 · A6F · A0 (s) | final obj s0 / s1: A9 · A6F · A0 (ref) | ΔP A9−A6F | d A9 vs A6F |
|---|---|---|---|---|---|---|---|---|
| 1 | neos-2746589-doon | 1.000 / 1.000 (1.000) | 0.792 / 0.996 (0.894) | 1.000 / 1.000 (1.000) [classprobe] | none / none · 94.1 / 119.4 · none / none | – / – · 2054.3 / 2811.3 · – / – (2008.2) | +0.1056 | −0.106 |
| 2 | cryptanalysiskb128n5obj16 | 1.000 / 1.000 (1.000) | 1.000 / 1.000 (1.000) | 1.000 / 1.000 (1.000) [classprobe] | none · none · none | – · – · – (0) | 0 | 0.000 |
| 3 | rail01 | 1.000 / 1.000 (1.000) | 1.000 / 1.000 (1.000) | 1.000 / 1.000 (1.000) [dev] | none · none · none | – · – · – (−70.570) | 0 | 0.000 |
| 4 | neos-2657525-crna | 0.752 / 0.170 (0.461) | 0.727 / 0.786 (0.757) | 0.795 / 0.871 (0.833) [classprobe] | 0.64 / 0.64 · 0.43 / 0.38 · 1.58 / 1.28 | 7.231 / 1.811 · 1.811 / 8.281 · 5.680 / 13.733 (1.811) | −0.2955 | +0.391 |
| 5 | neos-3024952-loue | 0.563 / 0.565 (0.564) | 0.761 / 0.750 (0.756) | 1.000 / 1.000 (1.000) [dev] | 0.08 / 0.07 · 0.04 / 0.05 · none | 40368 / 37101 · 106481 / 101590 · – (26756) | −0.1917 | +0.254 |
| 6 | sing44 | 0.441 / 0.449 (0.445) | 0.524 / 0.457 (0.491) | 0.518 / 0.532 (0.525) [dev] | 0.28 / 0.27 · 0.18 / 0.19 · 3.65 / 3.66 | 8.181e6 / 8.182e6 · 8.295e6 / 8.171e6 · 8.210e6 / 8.181e6 (8.129e6) | −0.0454 | +0.093 |
| 7 | ca__Scenario400_reserves_1 | 0.391 / 0.396 (0.394) | 0.587 / 0.586 (0.586) | 1.000 / 1.000 (1.000) [dev] | 34.2 / 35.2 · 20.3 / 20.2 · none | 33916.8 / 34127.4 · 34361.2 / 34200.9 · – (33840) | −0.1928 | +0.329 |
| 8 | item_placement_9903 | 0.358 / 0.370 (0.364) | 0.544 / 0.527 (0.535) | 0.631 / 0.689 (0.660) [dev] | 0.02 / 0.02 · 0.01 / 0.02 · 0.02 / 0.02 | 8.219 / 11.078 · 14.129 / 13.548 · 20.663 / 26.601 (8.411) | −0.1715 | +0.320 |
| 9 | neos-960392 | 0.065 / 0.050 (0.058) | 0.526 / 0.893 (0.710) | 1.000 / 1.000 (1.000) [dev] (a) | 0.08 / 0.07 · 0.09 / 0.09 · 0.07 / 0.07 | −238 / −238 · −238 / −215 · 0 / 0 (−238) | −0.6518 | +0.919 |
| 10 | neos-4763324-toguru | 0.630 / 0.576 (0.603) | 0.539 / 0.559 (0.549) | 0.772 / 0.772 (0.772) [classprobe] | 0.94 / 0.82 · 0.61 / 0.54 · 5.24 / 5.24 | 3547.2 / 3308.0 · 2208.8 / 2486.3 · 6760.7 / 6760.7 (1613.0) | +0.0541 | −0.090 |
| 11 | comp21-2idx | 0.552 / 0.543 (0.548) | 0.511 / 0.496 (0.504) | 0.880 / 0.901 (0.891) [classprobe] | 0.10 / 0.10 · 0.06 / 0.09 · 1.50 / 1.79 | 120 / 122 · 133 / 117 · 440 / 739 (74) | +0.0436 | −0.080 |
| 12 | chromaticindex1024-7 (guard) | 0.544 / 0.544 (0.544) | 0.054 / 0.067 (0.061) | 1.000 / 1.000 (1.000) [classprobe] | 65.3 / 65.3 · 6.53 / 8.01 · none | 4 / 4 · 4 / 4 · – (4) | +0.4834 | −0.889 |
| 13 | neos-3656078-kumeu (guard) | 0.094 / 0.112 (0.103) | 0.111 / 0.078 (0.095) | 1.000 / 1.000 (1.000) [classprobe] | 4.06 / 4.97 · 2.08 / 1.65 · none | −12770.8 / −12349.6 · −11930.8 / −12473.8 · – (−13172.2) | +0.0078 | −0.076 |
| 14 | glass4 (guard) | 0.221 / 0.211 (0.216) | 0.306 / 0.289 (0.298) | 0.332 / 0.356 (0.344) [dev] | 0.03 / 0.03 · 0.01 / 0.01 · 0.02 / 0.03 | 1.200e9 / 1.475e9 · 1.600e9 / 1.493e9 · 1.700e9 / 1.800e9 (1.200e9) | −0.0817 | +0.274 |
| 15 | radiationm18-12-05 (guard) | 0.011 / 0.011 (0.011) | 0.009 / 0.008 (0.008) | 0.154 / 0.304 (0.229) [dev] | 0.29 / 0.29 · 0.75 / 0.74 · 0.25 / 0.26 | 17573 / 17576 · 17578 / 17575 · 17904 / 24403 (17566) | +0.0025 | −0.226 |
| 16 | fhnw-binpack4-48 (guard) | 0.008 / 0.015 (0.011) | 0.004 / 0.007 (0.006) | 1.000 / 1.000 (1.000) [dev] | 0.94 / 1.74 · 0.49 / 0.83 · none | 0 / 0 · 0 / 0 · – (0) | +0.0057 | −0.508 |
| 17 | neos-3004026-krka (guard) | 0.003 / 0.003 (0.003) | 0.006 / 0.006 (0.006) | 0.318 / 1.000 (0.659) [classprobe] (b) | 0.31 / 0.41 · 0.68 / 0.67 · 38.2 / none | 0 / 0 · 0 / 0 · 0 / – (0) | −0.0026 | +0.465 |

(a) A0 found objective 0 at 0.07 s. Against the reference −238 that is gap 1 by the metric, so P = 1.
(b) A0 seed 1 found nothing.

These rows reproduce D-022's report:
- Ratio of means over the fixed 11 = **0.827** (0.585 / 0.707). Improved 6, worse 3 (doon, toguru, comp21).
- Mean d over the 17 = **+0.063**.
- Over the 17, A9 is worse than A6F on the seed mean on 7 instances: those 3 plus chromaticindex, fhnw, radiationm18 and
  kumeu.
- Without chromaticindex the mean d would be +0.122.
- A9 is not worse than A0 on any of the 17. d vs A0 is 0 on doon, crypt and rail01 and positive on the rest.

**Noise caveat for every C1-level number (review).** crna is bimodal in A9-type runs. SCIP's polishing cascade happened
in 3 of 6 runs (P 0.144, 0.090, 0.170) and not in the other 3 (0.734, 0.698, 0.752). At n = 2 this one instance moves
the fixed-11 ratio by about ±0.04. That is more than any candidate below could move it. EA posts to SCIP do not
separate cascade runs from the others (4 / 12 / 10 vs 4 / 10 / 7), so no lever was found there. **This note therefore
makes no projection of the C1 ratio; projections are per instance only.**

## 2. Why A9 loses where it loses

**Summary.** The losses do not come from SCIP or from memory. They come from cuOpt running slower inside A9 than alone,
and on the losing instances the partners do not make that up. Three causes, kept apart:
- the EA's GPU work slowing cuOpt (contention);
- the 3-thread layout (footprint);
- nobody contributing.

**Decomposition (seed means).**
- **P cuOpt-own** = the P of cuOpt's own incumbents inside A9 (`cuopt_pt`).
- **footprint** = A6 − A6F. It uses the old-harness 3-thread A6 runs, so it is **indicative only**.
- **inside-A9 slowdown** = cuOpt-own − A6.
- **partner gain** = cuOpt-own − A9.
- **net ΔP** = A9 − A6F. This sum is same-day and exact.
- **band** is D-018's rule *by analogy* (the larger seed-0 vs seed-1 bounded difference of either arm, at least 0.05).

| instance | P A6F | P A6 3-thr (n) | P cuOpt-own in A9 | P A9 | footprint | inside-A9 slowdown | partner gain | net ΔP | band | d |
|---|---|---|---|---|---|---|---|---|---|---|
| neos-2746589-doon | 0.894 | 1.000 (2) | 1.000 | 1.000 | +0.106 | +0.000 | +0.000 | +0.106 | 0.205 | −0.106 |
| cryptanalysiskb128n5obj16 | 1.000 | 1.000 (2) | 1.000 | 1.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0.050 | +0.000 |
| rail01 | 1.000 | 0.927 (1) | 1.000 | 1.000 | −0.073 | +0.073 | +0.000 | +0.000 | 0.050 | +0.000 |
| neos-2657525-crna | 0.757 | 0.776 (4) | 0.771 | 0.461 | +0.019 | −0.005 | +0.310 | −0.295 | 0.773 | +0.391 |
| neos-3024952-loue | 0.756 | 0.752 (3) | 0.784 | 0.564 | −0.003 | +0.032 | +0.220 | −0.192 | 0.050 | +0.254 |
| sing44 | 0.491 | 0.742 (4) | 0.858 | 0.445 | +0.252 | +0.116 | +0.413 | −0.045 | 0.128 | +0.093 |
| ca__Scenario400_reserves_1 | 0.586 | 0.658 (2) | 0.729 | 0.394 | +0.072 | +0.071 | +0.335 | −0.193 | 0.050 | +0.329 |
| item_placement_9903 | 0.535 | 0.651 (3) | 0.653 | 0.364 | +0.115 | +0.003 | +0.290 | −0.171 | 0.050 | +0.320 |
| neos-960392 | 0.710 | 0.614 (3) | 0.701 | 0.058 | −0.095 | +0.086 | +0.643 | −0.652 | 0.411 | +0.919 |
| neos-4763324-toguru | 0.549 | 0.578 (4) | 0.627 | 0.603 | +0.029 | +0.048 | +0.024 | +0.054 | 0.086 | −0.090 |
| comp21-2idx | 0.504 | 0.506 (5) | 0.653 | 0.548 | +0.002 | +0.146 | +0.105 | +0.044 | 0.050 | −0.080 |
| chromaticindex1024-7 | 0.061 | 0.077 (5) | 0.544 | 0.544 | +0.016 | +0.467 | +0.000 | +0.483 | 0.185 | −0.889 |
| neos-3656078-kumeu | 0.095 | 0.098 (4) | 0.103 | 0.103 | +0.004 | +0.004 | +0.000 | +0.008 | 0.294 | −0.076 |
| glass4 | 0.298 | 0.328 (3) | 0.366 | 0.216 | +0.030 | +0.038 | +0.150 | −0.082 | 0.057 | +0.274 |
| radiationm18-12-05 | 0.008 | 0.010 (3) | 0.013 | 0.011 | +0.002 | +0.002 | +0.002 | +0.002 | 0.050 | −0.226 |
| fhnw-binpack4-48 | 0.006 | 0.005 (3) | 0.011 | 0.011 | −0.000 | +0.006 | +0.000 | +0.006 | 0.458 | −0.508 |
| neos-3004026-krka | 0.006 | 0.004 (4) | 0.003 | 0.003 | −0.001 | −0.001 | +0.000 | −0.003 | 0.240 | +0.465 |

Means over the 17: footprint +0.028, inside-A9 slowdown +0.064, partner gain +0.147, net ΔP −0.055.

**cuOpt's process, A9@r13x vs A6F@r13x.**
- CPU threads: 3 (cores 1–3) inside A9 vs 5 (1, 2, 3, 5, 6) alone.
- CPU-s per wall-s: 2.91–3.00 inside A9 vs 4.85–5.03 alone, so cuOpt is saturated in both.
- RMM peak is equal in both arms: chromaticindex 5.42 GB, comp21 5.26, toguru 6.12, doon 8.57 / 8.44.
- MemAvailable at start was about 110 GB.
- SM clock 2392–2405 MHz; foreign GPU share 0 in every r13 run.
- Run-level GPU utilisation: 95–96 % in A9 vs 78–90 % in A6F.

| instance | cuOpt presolve_s A9 vs A6F | LP iterations A9 vs A6F | nodes A9 vs A6F |
|---|---|---|---|
| chromaticindex | 24.78 vs 23.96 | 71,712 vs 72,679 / 76,555 | 0 vs 0 |
| comp21 | 13.06 vs 12.62 | 17,013 vs 139,961 / 71,080 | 2 vs 348 / 56 |
| toguru | 24.2 / 25.2 vs 19.7 / 20.2 | 9,071 vs 9,094 | 0 vs 0 |
| doon | 11.7 vs 23.6 | 15,227 vs 15,921 / 18,144 | 1 vs 4 |
| rail01 | 12.4 vs 11.7 | 73.4k / 74.9k vs 115.0k / 112.7k | 0 vs 0 |
| crypt | 11.3 vs 11.1 | 30.7k / 30.2k vs 44.8k / 37.1k | 0 vs 0 |

**Per instance.**
- **chromaticindex1024-7** (guard; d −0.889). Mechanism: **contention plus missing contribution**. It is systematic;
  confidence high (a dose–response, 6 replicate runs, A2 and A6 controls).
  - cuOpt's only incumbent (4, the optimum) comes at 65.26 / 65.30 s inside A9. Alone it comes at 6.53 / 8.01 s.
  - Every full-duty A9 run gives the same time: 6 runs at 65.26–65.46 s (a9p, bo0, bo1s ×2, r13x ×2).
  - The delay depends on the EA's duty (D-018 stage 0): ½ duty 8.33 s, ¼ duty 9.81 s. The ⅛-duty run at 62.29 s is
    D-018's recorded one-seed anomaly.
  - Controls: 3-thread cuOpt alone takes 8.12–10.15 s (5 runs); A2 takes 6.50 s.
  - cuOpt's CPU path is the same in both arms (presolve 24.78 vs 23.97 s, LP iterations 71,712 vs 72,679, 0 nodes).
  - The EA is in repair phase 'pre' with cap None, so single FJ calls run 48.3 s (init) and 42.4 s (gen 1). It never
    finds a feasible point itself: still infeasible at 48.8 s, best violation 8,715.
  - cuOpt's 4.0 reaches the EA 24–26 s late, the length of one generation. SCIP never receives it: it spends the whole
    run in the root LP.
- **comp21-2idx** (fixed 11; d −0.080). Mechanism: **contention in two parts**, partly offset by the EA's own finds.
  Thread split ruled out. Confidence high for part 1; part 2's cause is open.
  - Part 1 depends on the EA's duty. cuOpt reaches ≤ 300 at 22.9–23.8 s alone but at 41.5 / 46.8 s inside A9. At ⅛ / ¼ / ½
    duty it gets there at 24.4 / 26.2 / 31.3 s.
  - The review added a root-LP signature. cuOpt's root-LP iterations rise with duty: ⅛ 10,099; ¼ 13,528; ½ 15,893; full
    17,013 in all 6 full-duty runs. 3-thread A6 takes 8,984–13,404.
  - Part 2 does not depend on duty. After about 50 s, cuOpt inside A9 stops improving at 146–172 at every duty (⅛ duty:
    151). Alone it reaches 114–133. Thread count, memory (RMM 5.26 GB everywhere) and the send callback (0.11 s over 166
    calls) are ruled out. What remains is SCIP beside cuOpt, or the second CUDA context.
  - 91 of 145 and 90 of 140 portfolio improvements are the EA's own, and the final best is the EA's (120 / 122 vs cuOpt's
    149 / 146 inside A9). Partner gain +0.105.
- **neos-4763324-toguru** (fixed 11; d −0.090, at the band of 0.086). Mechanism: **footprint (+0.029) plus inside-A9
  slowdown (+0.048)**, small partner gain (+0.024). Contention is *inferred*, not shown: no forced-duty run exists on
  toguru. Confidence medium.
  - cuOpt's LP iterations are 9,071 in every 3-thread run. Its big improvement run starts at 63–70 s (5 threads alone),
    73–77 s (3 threads alone) and 77–86 s (inside A9).
  - The EA's init FJ call is uncapped at 37.4 s. It runs 5 generations in 120 s; repair is 98.6–99.4 % of the wall.
  - A9 loses mostly in 90–120 s.
- **neos-2746589-doon** (fixed 11; d −0.106, inside the band of 0.205). Mechanism: **footprint plus missing
  contribution**. Close to noise: seed 1 is a near-tie (0.996 vs 1.000). Confidence medium-high.
  - 3-thread cuOpt finds nothing in 4 of 4 runs (A6 ×2, inside A9 ×2). 5-thread A6F finds a point at 94.1 / 119.4 s.
  - The EA stays infeasible (best violation 1.57 / 1.23; 31 / 30 generations), and SCIP finds nothing.
- **rail01, cryptanalysiskb128n5obj16** (ties at P 1). Mechanism: **missing contribution**: no arm finds a point.
  - rail01: SCIP never finishes its root LP and the EA's LP needs about 93 s on one thread (limit 60 s). The only rail01
    incumbent on record is one 3-thread A6 run on 10-06 (−49.26 at 107.5 s).
  - crypt: SCIP's presolve takes 104 s; nothing at 600 s either (A6F, M4).
- **neos-3656078-kumeu** (guard; d −0.076). **Noise**: the band (0.294) is far above |d|. A first-incumbent delay
  (4.06 / 4.97 s vs 2.08 / 1.65 s) is offset by later finds.
- **fhnw-binpack4-48, radiationm18-12-05** (guard; d −0.508, −0.226). **Floor-level.** Sub-second delays in cuOpt's start
  come from the EA's uncapped init call. ΔP is 0.006 and 0.003; d is large only because P is about 0.01. Together they
  contribute −0.043 to the 17-mean d. That is an observation about the metric, not a proposal to change it.

**Cross-cutting.**
- **The layout costs cuOpt 2 cores.** By D-021's layout, SCIP (core 5) and the EA host (core 6) take 2 of F's 5 cores,
  so cuOpt inside A9 always runs 3 threads. That is a fixed cost of the layout.
- **SCIP's presence.** Where it was measured, SCIP beside cuOpt did not delay cuOpt: A2 on chromaticindex 6.50 s, fhnw
  0.62, kumeu 3.46, radiation 0.86. It is untested on comp21, toguru and doon.
- **The EA's repair clock in phase 'pre'.** Before the EA has its own incumbent the repair clock has cap None. Each FJ
  call is then one GPU workload of 7.5–48 s; after the first incumbent the cap is 10 s. Exchanges pass only at
  generation boundaries, so a relayed vector waits up to a generation's length.

## 3. Where A9's gains come from

On the 6 C1 wins (crna, loue, sing44, ca, item_placement, neos-960392) and on glass4 and krka, almost none of the gain
is SCIP's own. A0 has a higher P than A9 on all 8. A *computed* race of A0 and A6F (per seed, the pointwise best of the
two) would give only 0.087 of the pooled 1.549 gain (5.6 %), and 0.082 of that is sing44. The gain comes from the EA
and from SCIP ⇄ EA interplay. cuOpt's own trajectory inside A9 adds almost nothing above the race.

The credit columns add up exactly: gain = (cuOpt credit inside A9 − cuOpt-alone credit) + SCIP + EA.
Gain = P̄(A6F) − P̄(A9), so **+ means A9 is better**.

| instance | P̄ A6F → A9 (A9 s0 / s1) | gain | cuOpt credit A9 − alone | SCIP native | SCIP improving a partner row (lag; integers changed) | EA total | EA credit by lineage | race P̄ | A9 above the race: who held the incumbent |
|---|---|---|---|---|---|---|---|---|---|
| neos-2657525-crna | 0.757 → 0.461 (0.752 / **0.170**) | +0.296 (seed 1 only) | −0.014 | 0 | **0.282** (1.8–6.5 ms; 1–5 of 524 integers) | 0.028 | small, mixed | 0.755 | 0.294: SCIP improving EA rows within 6.5 ms (seed 1) |
| neos-3024952-loue | 0.756 → 0.564 | +0.192 | −0.049 | 0 | 0.044 (3–12 s; 46–310 integers) | 0.197 | **LP stream 0.113**, random 0.046, own root LP 0.036 | 0.756 | 0.192, of which 0.165 EA LP-stream lineage |
| sing44 | 0.491 → 0.445 | +0.045 | −0.491 | 0 | 0.054 (38–46 s) | 0.483 | own root LP 0.245, random 0.120, LP stream 0.089, cuOpt rows 0.029 | **0.408** | **−0.037**: the race is lower than A9 |
| ca__Scenario400_reserves_1 | 0.586 → 0.394 | +0.193 | −0.409 | 0 | 0 | **0.602** | **own root LP 0.602** (one immigrant at 46.4 / 47.0 s; gap 0.994 → 0.011 / 0.008) | 0.586 | 0.193: EA own root LP |
| item_placement_9903 | 0.535 → 0.364 | +0.172 | −0.396 | 0.080 | 0.105 (incl. one 45-ms step changing 198 of 1,050 integers) | 0.382 (3,179 steps) | random 0.177, LP immigrant 0.112, LP stream 0.079 | 0.532 | 0.168: EA random 0.094, LP immigrant 0.043, LP stream 0.027 |
| neos-960392 | 0.710 → 0.058 | +0.652 | −0.263 (substitution, not lost resources) | 0 | 0 | **0.915** | **own root LP 0.592**, cuOpt rows 0.167, random 0.156 | 0.710 | 0.652: EA own root LP 0.569 |
| glass4 (guard) | 0.298 → 0.216 | +0.082 | −0.253 | 0.180 | 0.033 (0.27–18 s) | 0.121 | mostly random | 0.292 | 0.076: SCIP on EA rows 0.040, EA 0.028 |
| neos-3004026-krka (guard) | 0.0056 → 0.0030 | +0.003 | +0.003 | 0 | 0 | 0 | none | 0.0056 | 0.003: cuOpt (timing; reference 0, P degenerate) |

**Pooled over the 6 C1 wins** (gain sum 1.549; dominated by single instances):
- A9's margin above the race is 1.461. It splits as:
  - EA own root LP 0.754 (52 %; 960392 + ca)
  - SCIP improving EA rows within 50 ms 0.309 (21 %; crna's seed 1)
  - EA LP-stream lineage 0.167 (11 %; loue)
  - EA random lineage 0.152
  - EA cuOpt-row lineage 0.021
  - cuOpt-held ≈ 0
- **Zero by construction in these runs:**
  - LP decode: `repair.continuous: auto`, and the runner refuses lp_decode in A9 (#30).
  - cuOpt receiving partner solutions: 26.08 is one-way (#35).
  - The back-off: disabled.
- In all 4,524 A9 portfolio steps, no copied solution ever set a step.

**What this says about the levers.**
- **The largest single source of the margin is the EA's own async HiGHS root-LP rounding** (ca, 960392, part of sing44).
  Inside A9 that LP is admitted 1.3–1.9 times later than in A3 with the same EA config (ca 39.3–39.5 vs 28.7–29.3 s;
  sing44 19.3–19.5 vs 10.2–10.8 s).
  - In A9 the EA host is pinned to X925 core 6 alone and saturated. The LP worker inherits that affinity. In A3 the EA
    had A725 1–3.
  - The A3 runs come from earlier revisions.
- **Next come the SCIP ⇄ EA channels:** the LP stream (loue) and SCIP improving EA rows (crna seed 1, glass4).
- **cuOpt's own trajectory adds almost nothing above the race**, and it is crowded out on sing44, ca and item.

**Methods and their limits.**
- The portfolio is rebuilt from the checked incumbents as `build_runs` does. It matches `pt` on 102 of 102 runs within
  1e-9.
- Step credit: (γ before − γ after) × (T − t_k)/T. Credit goes to the first finder along the realised path.
- SCIP's own finds are split into "native" and "improving a partner row" at a **50-ms lag threshold chosen after seeing
  the distribution**: 5 lags of 1.8–6.5 ms, 1 of 45 ms, 34 between 0.25 and 54 s. The trace does not name SCIP's
  heuristic.
- EA lineage is a **pedigree share (expected ancestry, not gene identity)** over the roots in `ea.genealogy.npz`.
- The race is **optimistic**: each solver had its own resources, and A0 ran on the 10-06/07 revisions.

**Interpretation flag: every "LP arrival" in today's traces is poll time.** `wall_seconds` is stamped when the EA
receives the result (`LPWorker._receive`). The EA polls only in `_exchange_in`, at generation boundaries and in back-off
pauses, never inside a repair call. So "1.3–1.9 times later" mixes poll latency with solve time; A3's figures are poll
times too.
- Illustration (re-derived today): on sing44, bo1s seed 0 and seed 1 are the same arm with the same LP-worker CPU (6.0 /
  5.9 s), yet the LP is stamped at 19.45 s and 13.84 s. (bo1 s0's 14.38 s is not used here: phase Y paused the EA there,
  which is R1's mechanism, not the boundary effect.)
- The generation boundary, not the solve, sets the stamp, and it caps every "LP earlier" gain.

## 4. Lever inventory (short)

The inventory covers 34 levers. A9 = `configs/itrea2-dev-rc-3way.yaml` (EA config 1d12ad490dd0) + `exchange.from_cuopt`,
harness ext.

| state | levers |
|---|---|
| **Adopted on development evidence** | the EA's own HiGHS LP (`lp_source highs_async`, 60 s, 1 thread; DEV-07 +0.43 vs A0); deep FJ repair (20,000 / 1,000; DEV-05, DEV-10); the active repair clock (D-012, uncapped before the EA's first feasible point, 10-s cap after); one-way cuOpt → EA (D-015); R13 + R14 (D-022). Kept but never isolated in A9: the LP stream (D-011) and the self-adaptive strategy gene |
| **Adopted in A3, blocked in A9** | lp_decode (D-017, A3: +0.219 [+0.041, +0.429]); the runner refuses it with one EA core (#30), and it is a C1 item (D-019 addendum) → **R2** |
| **Built, not adopted** | Rank 1 clearing (D-016, −0.033 [−0.122, +0.050]); D-018's patience ladder (M1 / M3 / M5 failed); two-scale survival (DEV-06/09); default repair depth 2,000 / 100 (DEV-05); two-variable equality aggregation (built, not wired; PR-1 radiationm 0/32 vs 0/32) → **R5** |
| **Under test today** | phase Y only (D-023, `boY`, set `backoff2`, 66 runs, none read) |
| **Never varied** | population 256 / 256; Rank 0 cap x 0.10 (it binds: n0 = 25 on 9 of 17); Rank 0 demotion and thinning; selection; crossover and mutation mixes; stagnation immigrants; exchange parameters (poll 0.25 s, band 0.01, k 8); `post_frontier`; eval dtype; the EA LP's method, limit and core; SCIP settings inside A9 (`--emphasis` is wired only for A0*); cuOpt's thread count and the core split inside F (#30); GPU sharing mode (MPS; host change, Berk) |
| **Not built** | islands (#16); non-local operator (#17, parked); #26 recombine each partner incumbent with the elite; cuOpt restart from the EA's best (A8 design); two-way exchange via the 26.10 SET callback (#35) → **R4** |
| **Arms never run** | A5x0, A0*, A3r, H3 (HiGHS ↔ EA), A2F beyond M1's 3 instances; the y0, no_bound_demote, post_frontier and one-way ablations |

## 5. Candidates, ranked

They are ranked by expected gain on D-013's weak cases per unit of cost.

**Box costs** use the measured itx-r13 rate of **3.44 min per run, including the gaps between runs**. The rate excludes:
- window open and close;
- the GPU and cold-cache tests that new EA code needs before its first timed run;
- D-021's 0.55 window efficiency.

**Hints** follow D-024: futility-only, one-sided, fixed before the runs. A hint gates whether a line widens; it never
declares success. Beside each hint is its **null replicate**, measured from two A9 arms with the same EA config on
different days: A9@bo1s (10-11) / A9@r13x (10-13).

**Seeds.** R2 and R4 use seeds 0–3 on their hint instances, at the cost of more runs:
- R2: item's within-day seed spread is large (0.355 / 0.261 in bo1s). That is the noise source, and more seeds address
  it while the modelled effect stays reachable.
- R4: its effect is not modelled at all.

### R1. The EA stops busy-waiting on the GPU (blocking sync), so its own HiGHS LP worker gets X925 core 6

**Mechanism.**
- In A9 the EA host and the HiGHS LP worker it spawns share core 6. The worker is a 'spawn' child with `cpus=None`, so it
  inherits the EA's affinity.
- While the host waits on the GPU it busy-spins, which is CUDA's default:
  - `ls_fj_gpu.py` syncs after every launch (~50 ms each) through `int(cp.count_nonzero(...))`;
  - `backend.synchronize` is `Stream.null.synchronize`;
  - nothing in `src/` sets a scheduling flag.
- So the host keeps core 6 about 100 % busy even when 99 % of its wall time is GPU repair. Linux splits the core about
  50/50 with the LP worker, and the LP finishes about twice as late.
- **The change:** set the EA's primary context to blocking sync before anything retains it. CuPy 14.2 has no
  `setDeviceFlags`, so this goes through the driver API: `libcuda.so.1`, `cuInit`, then
  `cuDevicePrimaryCtxSetFlags(dev, CU_CTX_SCHED_BLOCKING_SYNC)`, before warm-up and t0. Also create `_Phases`' per-generation
  timing events with `block=True`, because a default-flag event wait may spin whatever the context flag says.
- **Bundled instrumentation (no behaviour change):** the LP worker stamps its own completion time into the `lp` event,
  so solve time and poll time are recorded separately.
- **Unchanged:** processes, cores, F and A6F (D-021 R5 / R6).
- **Where the switch goes:** in `configs/itrea2-dev-rc-3way.yaml`, so that A3x stays matched to A9. This gives a new EA
  config hash.

**Evidence.** LP-worker CPU = `run.json`'s wait4 rusage of the EA (which includes reaped children) minus
`ea.summary.json`'s RUSAGE_SELF. Re-derived today:

| instance | arm (seeds 0 / 1) | LP-worker CPU (s) | LP polled at (s) | EA's first own incumbent (s) | P | cuOpt-own P |
|---|---|---|---|---|---|---|
| ca | A9@bo1 (phase Y paused 14.94 / 14.81 s) | 15.5 / 15.4 | 23.19 / 24.05 | 35.0 / 34.5 | 0.300 / 0.292 | 0.702 / 0.692 |
| ca | A9@bo1s (same day, control) | 17.0 / 17.3 | 39.27 / 39.53 | 46.4 / 47.0 | 0.393 / 0.396 | 0.731 / 0.765 |
| ca | A9@r13x | 17.0 / 18.2 | 39.26 / 39.48 | 46.4 / 47.0 | 0.391 / 0.396 | 0.723 / 0.734 |
| sing44 | A9@r13x | 6.0 / 6.1 | 19.50 / 19.31 | – | 0.441 / 0.449 | – |
| sing44 | A3x@a9p (same core-6 layout, no cuOpt) | 5.4 | 10.92 | – | 0.287 | – |
| neos-960392 | A9@r13x | 2.4 / 2.5 | 5.42 / 6.07 | – | 0.065 / 0.050 | – |
| chromaticindex, rail01 | A9@r13x | 30.4 in a 60-s life (time limit, no solution) | – | – | – | – |

- **ca is a natural experiment** (D-018's bo1 vs its same-day control bo1s).
  - Phase Y paused the EA for about 15 s, and the LP was polled about 15 s earlier.
  - The EA's LP-rounding step (gap 0.994 → 0.011) moved 11.4 / 12.4 s earlier. Its credit, 0.983 × dt / 120 = 0.093 /
    0.102, equals the observed P difference (0.093 / 0.104) to within 0.002.
  - So bo1's gain on ca came from the EA's own LP getting core 6, not from cuOpt's GPU share: cuOpt's own P was about the
    same (0.70 vs 0.73–0.77).
  - A9@r13x matches bo1s, so the harness change does not matter here.
  - bo1 s1 is D-018's failed run (cuOpt hit its RMM limit at 83 s); its first 47 s are intact.
  - **Consequence for reading D-023 today:** a ca gain under boY is probably this LP-core mechanism, not cuOpt's GPU
    share. This changes how ca's boY row is read, not D-023's rules; no backoff2 run has been read.
- **Review rebuild from r13x generation boundaries.** With a full core the LP would be polled at:
  - ca: 18.3 / 20.8 s instead of 39.3 / 39.5 s;
  - sing44: 11.3 / 11.6 s (the next boundary) instead of 19.5 / 19.3 s;
  - neos-960392: 2.8–4.2 s instead of 5.4 / 6.1 s.

**Instances.** ca, sing44, neos-960392. Guards: item_placement_9903 and comp21-2idx (wake latency).

**Expected effect** (modelled from existing traces; interim; per instance only):
- **ca: 0.394 → about 0.23–0.30.** About 0.23 if the host's non-repair time is all spin. About 0.27 if half of it is
  real Python host work: on ca that time is 1.1–1.9 s of each 1.6–3.0 s pre-phase generation, and it is unmeasured.
  bo1 gave 0.296. If D-023 adopts boY first, phase Y already gives the LP most of the core on ca, and ca's increment
  shrinks to about −0.03 to −0.05.
- **sing44: −0.02 to −0.04.**
- **neos-960392: about −0.01 to −0.015.**
- **None expected on the three losses.** On toguru the LP needs only 6.3–6.8 s of CPU but is polled at the end of the
  uncapped init repair (37.8 / 45.8 s). chromaticindex and rail01 exceed the 60-s LP limit even on a full core.

**Cost.**
- Code: small, about half a day including review. Build in a worktree; merge after the running pool ends.
- Box: 20 runs ≈ 1.15 h, plus the exclusions above.

**Before the window (no timing):**
1. Assert the flag with `cuDevicePrimaryCtxGetState` and log it in `ea.summary`.
2. Run a long synthetic FJ call (or a dummy launch loop) and show EA self-CPU per wall-second well below 1. A test that
   only shows nothing compiled is not enough.
3. Run the cold-cache test.

**Targeted test.**
- Instances {ca, sing44, neos-960392, item_placement_9903, comp21-2idx} × arms {A9@bsync, the same-day development
  default} × seeds 0–1, 120 s, harness ext: **20 runs**. The control is A9@r13x, or A9@boY if D-023 adopts it.
- **Reported per run (mechanism, not gating):**
  - EA self-CPU per wall-second;
  - the LP worker's own finish time, and its CPU divided by its own solve wall (expected ≥ 0.8 on ca and sing44);
  - generations and launches per second on ca, sing44 and 960392 against the control. Those three are where a woken host
    can preempt a running LP worker, a path the guards never exercise.
- **No-harm (development-record form):**
  - item_placement and comp21: generations ≥ 0.95 × the control's, **paired by seed** (the within-day seed spread on item
    is 4 %);
  - P within max(0.05, the control's seed-0 vs seed-1 spread).
- **Hint (futility-only, fixed before the runs):** ratio of means of P(120), A9@bsync / control, over {ca, sing44,
  neos-960392} **≤ 0.95**, reported per instance as well as pooled.
  - Null replicate on these three: **0.989** (1.773 / 1.792). r13x seed 1 / seed 0: 0.997.
  - **In practice this hint is ca's.** sing44 and 960392 together reach at most about −0.045, right at the threshold.
    ca is stable: 0.391–0.396 in 4 runs over 2 days.
- **Pre-stated fallbacks**, each its own variant, tested the same way before a window:
  - (a) If EA self-CPU does not fall (some wait path still spins): renice the EA host below the LP worker after the spawn
    (`os.setpriority`).
  - (b) If the generation guard on item or comp21 fails (wake latency): `CU_CTX_SCHED_YIELD` instead of blocking sync.

**Dependencies.** Read D-023's verdict first, because it sets the control. Nothing beyond cuOpt 26.08. **Needs Berk:**
no; a development decision record and a normal window.

**Risks.**
- **Wake latency.** Core 6's idle states have exit latencies of 42 / 231 / 433 µs under the performance governor. Once
  the LP is done and core 6 idles between syncs, every sync pays a wake-up.
- **Instances with fast generations pay it most:** item 1,619–1,687 generations, comp21 248–295, crna 855 / 299, rail01
  383. item and comp21 are guarded; crna and rail01 are not in the test.
- **The P evidence is one natural experiment** on one instance (ca, old harness, 2 seeds), though both seeds agree and
  the arithmetic closes.
- The gain is capped by generation boundaries (see §3).
- It overlaps boY on ca.

### R2. lp_decode inside A9 on the EA's own core 6 (resolves #30's lp_decode question without re-splitting F)

**Mechanism.** D-017's LP decode fixes the integers of a few candidates per generation and solves the LP over the
continuous variables. It polishes feasible candidates and completes infeasible ones, and it is A3's default.

`pair.py` refuses it in A9 for two reasons:
- the decoder would share the EA's single core;
- one more pending batch takes the candidate set's concatenate past CuPy 14.2's 8-part limit. Above 8 parts CuPy
  switches to a pointer-array kernel that nothing warms, so it would compile on the clock and void the run.

After R1, core 6 has room for the decoder.

**The change:**
- **Concatenate:** add deterministic warm-up entries for a 9+-part concatenate in each dtype used (gene dtype, float64
  key / violation, bool feasible). Do not restructure the pending batches.
- **Decoder placement:** put it on the EA's core.
- **Gate:** the decoder's priming starts only after the LP worker has reported done or failed, so it never takes CPU
  from the LP worker in any branch. This resolves the conflict with R1 on ca: in A3@ld1 the decoder used about 10 s of
  solve CPU on ca and was ready at 4.3 s, which would overlap ca's 17-s LP.
- **Priority:** after the gate, the decoder's priority against the EA host is set by R1's measured item EA
  self-CPU: nice 10, or nice 0 if it is ≥ 0.5 per wall-second. In r13x it is about 1.0 (120.55 CPU-s over 120 s), so
  whether that is spin or real host work decides this branch.
- **The refusal** is lifted by a development decision record.
- **Unchanged:** F, SCIP's core and cuOpt's cores, so D-021 R5's M2 repeat does not apply.
- **Memory:** fits. A3@ld1 ran the EA, the LP worker and the decoder under the same 6-GB MemoryMax, with EA maxrss
  0.65–0.86 GB.

**Evidence.**
- D-017 (A3, 6 instances × 2 seeds): mean d +0.219 [+0.041, +0.429]. item_placement 0.311 → 0.118, sing44 0.377 → 0.238;
  ca +0.005, glass4 about 0.
- **The mechanism is solid:**
  - item had 43 / 47 decoded incumbents per run, 11 / 17 of them from completion;
  - sing44 had 5 per seed, and its +0.369 lies beyond its band of 0.11.
- **The P anchor is weaker:**
  - item's D-017 gain lies inside its own D-016 band of 0.68;
  - A3 controls on item range 0.134–0.469 across seeds;
  - D-017 also records that lp_decode changes the random stream: fhnw moved +0.348 with no decoder row.
- Decoder CPU in A3@ld1: item about 2.4–2.9 s, sing44 about 2.3 s, ca about 14–15 s, glass4 about 0.3 s per 120-s run.
- In A9 the EA carries both target instances (EA credit 0.382 on item, 0.483 on sing44, §3).
- Non-singleton continuous columns: item 33, sing44 15,995, ca 100,704, glass4 19, kumeu 658.

**Instances.** item_placement_9903, sing44. No-harm: ca, glass4, neos-3656078-kumeu.

**Expected effect** (modelled; interim; the transfer from A3 to A9 is unmeasured). If ¼ to ¾ of D-017's effect
transfers: item −0.06 to −0.18, sing44 −0.04 to −0.10, which gives a ratio over the two of about 0.64–0.88. ca, glass4
and kumeu are expected near 0. No action on the three losses.

**Cost.**
- Code: moderate, about 1 day plus review. The engine's pending-batch path has not been read, so the scope is
  uncertain. A GPU cold-cache test must provably reach the decoder path in the A9 layout (EA on one X925 core, with
  `from_cuopt`). New hash.
- If D-023 adopts boY, lp_decode runs beside active back-off for the first time (`_bo_pace` / `_lpd_poll` / `lpd_sync`
  interact), so back-off × lp_decode CPU tests are added.
- Box: 28 runs ≈ 1.6 h.

**Targeted test.** Run it after R1's verdict, not in R1's window before that verdict (that would stack two changes).
- Arms {A9@bsync+ld, A9@bsync}, 120 s, ext.
- Hint instances {item_placement, sing44} × seeds 0–3 (16 runs); no-harm instances {ca, glass4, kumeu} × seeds 0–1
  (12 runs). **28 runs.**
- **Mechanism checks:**
  - decoder ready ≤ 10 s (D-017 M0);
  - **≥ 1 decoded incumbent per seed** on item and sing44;
  - `lpd_wait_s_max` and `lpd_host_s` compared with A3@ld1;
  - ca's LP-worker solve time within 10 % of R1's.
- **No-harm:**
  - **ca within max(0.02, the control's seed spread)**; ca's spread is 0.005, so 0.05 would let a lost R1 gain pass;
  - glass4 and kumeu within max(0.05, the control's seed spread).
- **Hint (futility-only, fixed before the runs):** ratio of means of P(120), A9@bsync+ld / A9@bsync, over {item, sing44},
  seeds 0–3, **≤ 0.90**.
  - Null replicate at 2 seeds: **0.918**. The null SD is about 0.06 at 2 seeds and about 0.04 at 4, so 0.90 is about 2.4
    null SDs.
  - The modelled range 0.64–0.88 stays reachable.

**Dependencies.**
- R1 adopted.
- **Pre-written fallback if R1 is not adopted:** the decoder still gated after the LP, at nice 0 beside the spinning
  host, tested against the default.
- It is a C1 item (D-019 addendum: "lp_decode in A9 (core layout)"; #30). **Needs Berk:** no; a development decision
  record.

**Risks.**
- The transfer from A3 (3 A725 cores, no cuOpt) to A9 (core 6, GPU contention) is unmeasured.
- Host overhead per offer: D-017's M1 failed by its letter on item (1.39–1.63 % of wall).
- A new pending-batch shape is a compile-on-clock risk until the cold-cache test reaches it.

### R3. Same-day diagnostic window: thread count and SCIP's presence (config only; replaces the dropped candidate 3)

This is a diagnostic, not a lever, so it has **reading rules fixed before the runs** instead of a hint. Both reviewers
asked for it: reviewer 1 as the replacement for candidate 3 and as its first missing item, reviewer 2 as its seventh
missing item. It is the only work before 26.10 that addresses comp21, a C1 loss.

**Mechanism under test.**
- **Thread / core effect.** The footprint column in §2 compares old-harness 3-thread A6 with ext 5-thread A6F. That
  comparison also changes:
  - the thread variables (OMP / OPENBLAS / MKL = 1 vs unset);
  - the gaps (default vs 0);
  - RMM (12 vs 24 GB);
  - the CUDA cache;
  - the day.
  D-021 R6 records counter-evidence (A6 vs 24-thread GAMS, R 1.00 [0.89, 1.12]).
- **comp21 part 2.** Diffing comp21's cuOpt parameters in A9@r13x and A6F@r13x leaves only two differences: cores (3 vs 5)
  and `send=True` (0.11 s of exchange time). Seed, gaps, RMM and environment are identical. Part 2 does not depend on
  the EA's duty (⅛ duty still stalls at 151, n = 1). What remains is SCIP beside cuOpt, or the EA's second CUDA context.

**Runs.** Seeds 0–1, 120 s, harness ext, one window:

| arm | what it is | instances | runs |
|---|---|---|---|
| A6@ext | cuOpt alone, 3 threads on A725 1–3 | comp21, toguru, doon, sing44, item_placement, ca | 12 |
| A6F@ext | cuOpt alone on F, re-run in the same window | the same 6 | 12 |
| A2@ext | SCIP on X925 5 + 3-thread cuOpt on A725 1–3, no EA, core 6 idle | comp21, toguru, doon | 6 |

**30 runs ≈ 1.7 h.**

**Reading rules (fixed before the runs).**
1. **Thread effect on instance i** is present iff mean P(A6@ext) − mean P(A6F@ext) > max(0.05, the larger seed-0 vs
   seed-1 spread of either arm). These same-day numbers replace §2's indicative footprint / slowdown split.
   - Present on toguru or doon (where cuOpt holds or would hold A9's portfolio) → the layout question (a cuOpt thread on
     X925 6) goes to Berk with these numbers. The rejected section below lists what that design would need.
   - Otherwise the layout line is closed for v1.
2. **comp21 part 2** is read from cuOpt's final objective in A2@ext:
   - **≥ 146 on both seeds** → SCIP beside cuOpt is the cause;
   - **≤ 133 on both seeds** → the EA's CUDA context remains, and the MPS question (a host change, Berk's) becomes
     concrete;
   - mixed → inconclusive at n = 2, stated as such.
   - **Precondition:** A6@ext's own cuOpt must reach ≤ 133 on comp21. If it ends ≥ 146 too, the stall is the 3-thread
     footprint under ext, and rule 2 does not apply.
3. **SCIP's presence on toguru and doon:** P of cuOpt in A2@ext vs A6@ext, with the same band as rule 1.

**Cost.** No EA code. Box ≈ 1.7 h.

**Dependencies.** First check that `make_jobs` and the runner accept A6 and A2 under harness ext. A2F ran under ext in
M1, and the harness is allowed on partner-lane arms. If they do not, it is small jobs / runner code. It can run in any
window after backoff2, before or alongside R1. **Needs Berk:** no.

**Risks.** n = 2 per cell. doon is a ceiling case: A6F found it at 94.1 and 119.4 s, P 0.792 / 0.996.

### R4. Two-way cuOpt exchange through the 26.10 SetSolutionCallback (#35): post-freeze, v1.1 unless Berk decides otherwise

**Mechanism.** EA and SCIP incumbents are offered to cuOpt mid-run. With PR #2098, presolve stays on, and offers are
checked on the original model. There are two limits:
- cuOpt polls SET only when its GPU population records a new best, so an offer enters only while cuOpt's own heuristics
  are still improving;
- registering a SET callback turns Papilo's dual reductions off, which costs something even when nothing is offered
  (crna: 308 vs 305 presolved columns).

**Evidence.**
- **Opportunity, not payoff.** Share of time after cuOpt's first incumbent in which A9's system best is strictly better
  than cuOpt's own best inside A9: comp21 0.95 / 0.93, item 0.99 / 0.99, loue 0.89 / 0.89, ca 0.64 / 0.86, toguru
  0.39 / 0.51.
- **The late window is thin.** Improving GET reports (an upper bound on SET polls) include comp21's 166 / 199. But the
  review recounted cuOpt's own incumbents on comp21 after 60 s: only 3 per seed (77.8 / 79.8 / 87.7 s; 105.6 / 110.0 /
  113.6 s). That is exactly the window of the 146–172 stall.
- **Plausible target.** Most poll opportunities fall before 50 s, so the plausible target on comp21 is part 1 (cuOpt
  reaching ≤ 300 at 41–47 s instead of about 23 s), not the late stall.
- **Probe on the 26.10 nightly** (20 s, standalone; raw outputs in `/tmp/setsol`, scratch):
  - SET calls: item 23–24, loue 13–14, glass4 13;
  - crna 11, all at 0.70–0.71 s;
  - sing44's first SET call at 28.3 s;
  - comp21 and toguru not probed.
- No data exists on what cuOpt does with a better point accepted mid-run.

**Instances.** comp21-2idx, neos-3024952-loue, item_placement_9903, neos-4763324-toguru. Blocked on crna by the polling
limit. Cannot act on doon, rail01 or cryptanalysis, where nobody has an incumbent to pass.

**Expected effect.** Not modelled.

**Cost.**
- Code: large, several days:
  - a SET callback in `cuopt_run` fed from the EA / SCIP mailbox in original variable order;
  - the EA → cuOpt channel and its records;
  - a no-op-SET control arm;
  - `cuopt_python` plumbed through the pool and `make_jobs` (`PairOptions.cuopt_python` exists, `pair.py:796`, but
    nothing passes it);
  - a per-cuOpt-version `CUDA_CACHE_PATH`, warmed outside the timed set (R15), so that 26.10's kernels neither JIT on
    the clock nor mix with the frozen 26.08 cache.
- Box:
  - the probe ≈ 0.15 h;
  - the test, 32 runs ≈ 1.8 h;
  - #31's A6F re-baseline on released 26.10 (34 runs ≈ 1.95 h), which v1 owes anyway.

**Targeted test.**
1. **Probe first,** inside a lease (it is GPU work): `tools/probe_cuopt_setsol.py --mode noop --time 120 --threads 3` on
   comp21 and toguru.
   - Proceed only if there is **≥ 1 SET call after 50 s per run** on at least one of them.
   - Otherwise restate the target as comp21's part 1 and drop toguru.
2. **Then** instances {comp21, loue, item_placement, toguru} × arms {A9-2way@26.10, A9-1way@26.10 with a never-offering
   SET callback} × **seeds 0–3**, 120 s, ext: 32 runs.
   - **Mechanism:** SET calls and accepted offers per run; cuOpt's own P inside A9.
   - **Hint (futility-only):** ratio of means of P(120), 2-way / 1-way-noop, over the four **≤ 0.95**. The control's
     seed-mean sum is about 2.08, so the hint needs about −0.10.
   - Null replicate over the four at 2 seeds: **0.972**, about one null SD, which is why 4 seeds are used.
   - **Any A9-2way vs A6F figure carries the caveat that cuOpt's dual reductions are off in A9-2way and on in A6F.**

**Dependencies.**
- The cuOpt 26.10 release (latest v26.08.00; D-019's fallback date is 11-06), #31 and #35.
- **It cannot enter v1 as things stand.** The D-019 addendum closes development on 26.08 at C1, and `configs/frozen-v2.yaml`
  plus R33 seal A9's v1 configuration. So it is v1.1 unless Berk supersedes.
- **Nightly use:** measurement only; never in a frozen arm or in the v1 baseline. #31's re-baseline uses the released
  26.10.

**Needs Berk:**
- yes, for any timed A9 run on the nightly before release;
- yes, for v1 inclusion;
- per post, for the unposted stall-polling feature request (`vault/70-upstream/drafts`).

**Risks.** SET may never be polled during comp21's stall. Dual reductions off changes cuOpt itself.

### R5. Two-variable equality aggregation for the no-incumbent trio: a CPU payoff probe first

**Mechanism.** `itrea.ea.aggregate` (built, not wired) substitutes chains x_k = ±x_j + b, so that a single FJ move keeps
a whole chain satisfied. The three instances where no A9 process finds an incumbent carry 3.0 of A9's 6.44 P-sum over
the fixed 11 (47 %).

**Evidence.**
- Counts verified today on CPU cores 10–14:
  - doon 50,936 → 42,648 columns and 31,530 → 23,242 rows (8,288 ties; 0 cycles, 0 refused);
  - cryptanalysis 48,950 → 44,282 columns and 98,021 → 93,353 rows (4,668 ties);
  - rail01 20,791 ties (Development log, 10-06);
  - comp21 and toguru: none.
- The EA ends infeasible on all three (best violation: doon 1.57 / 1.23, crypt 286 / 362, rail01 28.4 / 29.5).
- **The only payoff probe so far is negative:** radiationm, 0/32 vs 0/32 feasible, with the aggregated median violation
  *worse* (127 vs 88). Its blocker was exact integer sums, not chains.

**Instances.** neos-2746589-doon, cryptanalysiskb128n5obj16, rail01.

**Expected effect.** Low probability, large payoff if it works (any feasible point lowers P from 1.0). No estimate is
possible.

**Cost.**
- Stage 0 needs a small scratch variant of `tools/agg_probe.py` in `/tmp`. The tool hard-codes
  `RepairCfg(max_steps=20000, stall_steps=1000)` and checks lifted vectors with the EA's own float64 `p.check`. The
  variant takes the budget as an argument and sends every lifted vector through `itrea.analysis.check_solutions`, the
  independent exact checker (#36, the checker tolerance question, is still open).
- No GPU. rail01's LP takes about 93 s.
- **Run safety:** run under `systemd-run --user --scope -p MemoryMax=4G`, `taskset -c 10-14`, `OMP_NUM_THREADS=1`, and only
  when no M6 or A0 CPU-lane window is active. nice 19 does not protect the pool: the probe sits in another cgroup, and
  CPU is shared by cgroup weight; memory bandwidth would also contaminate M6's interference measurement.
- Stage 1: wiring aggregation into the GPU genome is large and comes after the freeze, so it is v1.1. At most 12 runs
  (≈ 0.7 h).

**Targeted test.**
- **Stage 0:** arms original and aggregated, each at 1× and 10× budget (matched), K = 32, seeds 0–1, on doon,
  cryptanalysis and rail01. Report violation min and median per arm.
- **Hint (comparative, futility-only):** at the same budget, the aggregated arm's count of children that lift to a point
  the exact checker accepts is **greater than the original arm's**, on any of the three instances.
- **Stated before the run:** a null says nothing about aggregation, because a 32-child CPU FJ is much weaker than the GPU
  EA that already fails on these instances. Only a positive result is informative.
- **Stage 1 (only after the hint, v1.1):** A9@agg vs A9 on the passing instance(s) × seeds 0–1, 120 s, ext.

**Dependencies.** None for stage 0. **Needs Berk:** no.

**Risks.** The chains may not be the blocker (the radiationm precedent). CPU FJ is not the GPU FJ. cryptanalysis has
other structure: SCIP's presolve alone takes 104 s.

## 6. Rejected or not proposed

**Candidate 3: cuOpt takes core 6's idle cycles (4 threads on A725 1–3 + X925 6, cuOpt at lower priority). Dropped:
reviewer 1 rejected it (keep = false).** Replaced by R3's diagnostic.
- **The hint does not discriminate.** A9@bo1s / A9@r13x on {doon, toguru, sing44, item_placement} = **0.970**, exactly the
  proposed 0.97 threshold. doon is 1.0 in both because 3-thread cuOpt never finds it. A near-null replicate pair already
  opens the hint.
- **The motivating evidence is fully confounded:** no 3-thread cuOpt-alone run exists under harness ext (see R3), and
  D-021 R6 records counter-evidence.
- **Per target, little can move:**
  - toguru's 3-thread A6 (0.556–0.591) overlaps A6F's 0.539 / 0.559;
  - on sing44 and item the EA carries the portfolio (the system best is below cuOpt's own 78–99 % of the time), so a
    faster cuOpt barely moves P;
  - doon is a one-seed ceiling case.
  - Without doon, the modelled gain is about one null SD.
- **Reviewer 2 kept it but showed the mechanism as written cannot work on this host.** Each role runs in its own systemd
  scope under `app.slice` with the cpu controller enabled. Between sibling scopes the CPU is shared by `cpu.weight`, not
  by nice, so "cuOpt's scope at nice +5" would still take about half of core 6 from the EA scope. Any future version
  would need `-p CPUWeight=` (e.g. 10–20 against 100).
- **Reviewer 2 also corrected the code note.** For A9, `cuopt_threads()` returns `opts.cuopt_threads or 3`, so
  oversubscription is already a job option. Cores {1, 2, 3, 6} would need layout code.
- **It would need Berk** (AGENTS.md: "Timed solvers pinned one per X925 core"; D-015 / D-021's layout) and an answer on
  D-021 R5 (does M2's soak repeat?).

**Not proposed by the candidate stage** (one reason each):
- **cuOpt oversubscription on cores 1–3:** adds no CPU, since cuOpt is saturated at 2.91–3.00 CPU-s per wall-second.
- **Phase-Y variants:** wait for D-023's verdict; backoff2 is unread.
- **#26** (recombine partner incumbents with the elite): no evidence beyond lineage shares; it is its own C1 item.
- **Population, operator mix, Rank 0 cap, selection:** never varied, and no instance-level evidence.
- **SCIP emphasis inside A9:** needs A0* as the fair control.
- **CUDA MPS:** a host change for Berk with only indirect evidence (comp21's duty-insensitive part). R3 makes it concrete
  or closes it.
- **D-016 clearing, D-018's ladder, two-scale survival:** rejected; the evidence is unchanged.

**Follow-ups named by the review (not ranked).**
- **LP-ready early exit from the uncapped pre-phase repair call**, as its own variable after R1. The FJ loop already
  polls `should_stop()` after every launch.
  - toguru's LP is ready by about 13 s on half a core (about 7 s with R1) but is polled at 37.8 / 45.8 s. Step credits
    give about −0.02 on toguru against a 0.054 loss that lies mostly in 90–120 s, plus a few thousandths on sing44.
  - Small and cheap; not a strong lever alone. D-012 kept deep pre-feasible repair on evidence from sing44, ca, loue and
    fhnw.
- **What no ranked candidate reaches:**
  - **chromaticindex1024-7**, the largest guard loss (d −0.889). Only D-023's phase Y (running today) and an unproposed
    pre-feasible repair cap act on it. Without either, the guard stays carried by this one instance.
  - **comp21's part 2** is reached before 26.10 only by R3's diagnosis.

## 7. Planning consequences

- **Order.** R1 and R2 are built in worktrees now and merged after the running pool ends. R1 is tested in one window
  right after D-023's verdict; R2 follows R1's verdict. R3 can share a window with either. R5's stage 0 runs on the CPU
  outside M6 and A0 CPU-lane windows. R4 waits for 26.10.
- **Time before the C1 freeze (Fri 10-23).** Also due are D-023's verdict, M4's 11 remaining runs, A0 on the 81 and
  #26's decision. R1 + R2 + R3 is about 4.45 h of run time, or about 8 h of window at D-021's 0.55 efficiency. They fit
  only if built now and run in the windows right after backoff2.
- **If R1 or R2 is adopted before the freeze, A9's development default changes hash.** The 34 A9@r13x runs then describe
  the old default. See the first question for Berk.

## Open questions for Berk (real decisions only)

1. **Which C1 reading does the go / no-go use if R1 or R2 changes A9's default before the freeze?**
   - (a) Re-read C1 on the exact frozen-v2 hash with A6F re-run in the same window: 68 runs ≈ 3.9 h.
   - (b) A9 only, labelled cross-day against A6F@r13x of 10-13: 34 runs ≈ 2.0 h.
   - (c) Read the old default for the go / no-go, with R1 / R2 as dated addenda.
   - **Default if no answer:** (a). The go / no-go should describe the configuration that is frozen, and D-021 / D-024
     practice re-runs the control in the same window.
2. **Two-way exchange (#35): v1 or v1.1?** Under the D-019 addendum and R33 it is v1.1 unless you supersede.
   - Separately: may timed A9 runs on the 26.10 nightly happen before the release, as measurement only? The SET-poll
     probe (R4 step 1) needs only a normal lease and is not a timed A9 run.
   - **Default if no answer:** v1.1, and no timed A9 runs on the nightly.

Not asked now (they become questions only if R3's reading rules point there):
- the layout exception for a cuOpt thread on X925 6;
- CUDA MPS.

Already pending elsewhere: the per-post yes for the upstream stall-polling feature request.

## Sources and scratch

- **Inputs** as listed at the top. All commands ran with `OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c
  10-14 .venv/bin/python`. One read-only JSON-parsing call by a reviewer ran without that prefix (no GPU, no writes).
- **Scratch (not in the repository, may vanish):**
  - `/tmp/c1attr` (attribution: `core.py`, `steps.py`, `lineage3.py`, `final3.py`, `race.py` and their CSVs);
  - `/tmp/c1cand` (`lp.py`, `lpcpu.py`, `ld.py`, `README.txt`);
  - `/tmp/setsol` (the 26.10 nightly SET probe);
  - the loss diagnosis's scratch (`/tmp/c1loss`) was not kept.
- **Nothing else** in the repository was written or committed.
