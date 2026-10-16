# Phase 1 — development until C1 (2026-10-06 to 2026-10-16)

## Summary

- **Result (the phase-1 interim result, four seeds, D-031):** on the 11 instances where cuOpt alone is weak (the fixed 11), A9 lowers mean P(120) from 0.690 to 0.550 (−20 %) against cuOpt alone on the same footprint, 7 better / 3 worse, d +0.249 [+0.078, +0.441], Wilcoxon p 0.027. C1's hint is present (0.798 ≤ 0.95) and its guard holds (+0.045 ≥ −0.05) over 136 runs, VALID. The halves agree: seeds 2–3 alone 0.787 / +0.034, seeds 0–1 (D-030) 0.809 / +0.050. On the old default (10-13, two seeds) they read 0.827 and +0.063 (§ 4, § 7).
- **Against SCIP alone (A0 from earlier days, not the same window or footprint):** on the fixed 11, 8 improved / 0 worse, mean P −38 %; on SCIP's weak set among the 17 (15), 12 / 0, 0.889 → 0.448 (−50 %), d +0.499 [+0.323, +0.676], p 0.0005, a set selected on the compared runs (§ 7).
- **Harm flags beside, not criteria:** glass4 (d_i −0.114; D-032 read it in one window as day-to-day drift) and comp21-2idx (d_i −0.051 against its band 0.050, at the edge). cuOpt alone crashed (SIGSEGV) on comp21 on 3 of 4 seeds; the reading keeps those runs as outcomes, and the reading that leaves them out is INCOMPLETE (§ 7).
- **Frozen for phase 1:** harness ext (D-022), lp_decode (ld0, D-026), yield (ysync, D-029) → `configs/frozen-v2.yaml`, a byte copy of `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` (EA hash `835c7c5a249c`; § 2). Not adopted: the back-off (D-018, D-023), blocking sync (D-025).
- **Tested:** developing the coupling (first A3, then A9 = SCIP ↔ EA ← cuOpt) on 9 development + 8 class-probe instances at 120 s, toward D-013's aim: lift each solver's weak set, the mean as a guard (§ 1, § 3).
- **Known loss:** chromaticindex1024-7. Inside A9 cuOpt's first incumbent comes at about 65 s; without the EA it comes in 6.5–11 s (§ 6).
- **Harness calibration (M4):** our cuOpt-alone runs (A6F) on D-020's 30 MIPfeas instances give a footprint ratio R = 1.313 [1.091, 1.618] against GAMS's published runs, OUTSIDE the band [0.870, 1.150]. With the clock checked, the gap is the footprint (5 threads + the GPU, against GAMS's 24-thread whole box) and the machine, so v1's external targets are anchored on our own A6F (§ 5).
- **Decided: C1 go** (Berk, 2026-10-16 13:1x: *"Go: freeze the candidate (Recommended)"*). Phase 1 closes on these numbers; the two harm flags, the cuOpt-alone crashes on comp21 and the chromaticindex loss are carried as stated caveats into phase 2 / v1 ([[D-033 C1 go — phase 1 closes with configs-frozen-v2.yaml = yield + lp_decode]]).
- **To phase 2:** #26; #35 and CUDA MPS (v1.1). Due before prereg-v2: #36 and #32's test-set size (§ 8, § 9).
- **Every A9 number is interim and descriptive:** n = 4 seeds for the C1 result (2 in § 4's tables) on instances that were chosen and tuned on.

## Status of this note

- **Final** (2026-10-16, after D-031's reading and Berk's C1 go). § 7 holds the four-seed result.
- **INTERIM and descriptive:** development data on instances chosen and tuned on (n = 4 seeds for the C1 result; 2 in § 4).
- **The go / no-go on C1 was Berk's: go** ([[D-033 C1 go — phase 1 closes with configs-frozen-v2.yaml = yield + lp_decode]]; [[D-019 v1 — deadline and performance target]], addendum 2026-10-15).

Author, the futility-only hint, the no-claim rule and the freeze file: "About this note" at the end.

## Terms

| term | meaning |
|---|---|
| P | P(120), the primal integral at 120 s on the call clock; lower is better |
| d | bounded d on seed means; + means A9 is better than its comparator |
| A9 | SCIP ↔ EA ← cuOpt |
| A6F | cuOpt alone on A9's footprint F |
| A0 | SCIP alone |
| fixed 11 | the 11 cuOpt-weak development instances of D-022's hint |
| the 17 / class probe | the 9 development instances (D-010) + the 8 class-probe instances, added 2026-10-07 with predictions fixed before the runs (D-014) |
| weak set W_B | instances where solver B alone has P ≥ 0.5, on seeds not used for the evaluation (D-013) |
| C1 | the checkpoint that closes phase 1: planned Sat 2026-10-17 (D-019); reached Fri 2026-10-16 13:1x on Berk's go (D-033) |
| hint | ratio of means, A9 / A6F, over the fixed 11; present iff ≤ 0.95. Futility-only: it gates widening and declares nothing (D-022, D-024 rule 2) |
| guard | mean bounded d over the 17, A9 vs A6F; holds iff ≥ −0.05 (D-022) |
| harm leg / DRIFT | per instance, the new A9 against 10-13's A9@r13x; harm iff d_i < −band_i, band_i = max(0.05, 10-13's seed spread) (D-025 amendment 3). DRIFT = D-032's reading that glass4's flag is day-to-day variation |
| F | footprint: A9's cores and GPU (X925 5–6, A725 1–3, the GPU); every compared GPU arm gets it (D-021) |
| harness ext | cuOpt's MIP gaps 0, its thread variables unset, its own RMM limit 24 GB (D-022) |
| ld0 / ysync | lp_decode on the EA's core, gated after its LP worker (D-026) / `host.cuda_sched: yield` on the EA's CUDA context (D-029) |
| M4 / R / OUTSIDE | the harness calibration (D-021): footprint ratio R = (SGM_ours + 0.001) / (SGM_GAMS + 0.001) on the call clock; OUTSIDE = R outside the band [0.870, 1.150] |

## 1. What phase 1 set out to do

**Takeaway: every C1 item is settled — done, or carried to phase 2 (#26). Phase 1 closed on Fri 2026-10-16 13:1x on
Berk's C1 go (D-033), one day before the planned date.** Phase 1 developed the coupling toward D-013's aim and had to
end in C1: an interim A9 result, a calibrated external harness and a configuration to freeze.

**The aim** ([[D-013 Aim — lift each solver's weakest cases]], Berk, 2026-10-07): *"I don't aim our coupling to be
generally better but to help both SCIP, cuOpt (and Highs if possible) in some of the hardest problems where they are the
weakest. Our performance metric is improvement upon the worst performances, of course with a regard to mean to some
degree."* The metric, for each solver B, has three parts:
- **Weak set.** W_B is the set of instances where B alone has P ≥ 0.5, measured on seeds that are not used for the
  evaluation.
- **Primary.** The coupling against B alone on W_B: share improved, mean P reduction, mean bounded d with its bootstrap
  interval, the Wilcoxon test, and the worst P after coupling.
- **Guard.** Mean d ≥ −0.05 over all instances.

**What C1 had to contain** (D-019 addendum 2026-10-10): development on cuOpt 26.08 closes with the items in the table.
The addendum of 2026-10-15 13:4x (Berk: *"Assume we are wrapping up phase 1 on Saturday."*) keeps that content, adds
`configs/frozen-v2.yaml` and moves C1 from Fri 10-23 to **Sat 10-17**.

| C1 item | state on 2026-10-16 (13:1x) | record |
|---|---|---|
| D-018 stage 1 (GPU back-off) | NOT ADOPTED 10-11; its follow-up D-023 (phase Y only) NOT ADOPTED 10-14 | D-018, D-023 |
| #26 decided (recombine a partner incumbent with the elite) | not built; carried to phase 2 | D-019 addendum 10-15 |
| lp_decode inside A9 (#30) | ADOPT (ld0) 10-15 | D-026 |
| external harness built and calibrated | M1–M3 done 10-11; M4 complete 10-15, **OUTSIDE** (the footprint) | D-020, D-021, § 5 |
| interim A9 result, 17 × 2 seeds, against SCIP alone and cuOpt alone in D-013 terms | 10-13 on the old default (D-022); 10-16 on the candidate (D-030, seeds 0–1); **on four seeds (D-031, read 10-16 12:55): hint 0.798, guard +0.045** | § 4, § 7 |
| configuration to freeze | yield + lp_decode (`configs/itrea2-dev-rc-3way-ysync-ld0.yaml`) | D-029, D-030, D-032 |
| `configs/frozen-v2.yaml`, go / no-go | **go** (Berk, 10-16 13:1x); `configs/frozen-v2.yaml` written the same afternoon, a byte copy of the candidate, its hash pinned in `tests/test_cuda_sched.py` | D-033 |
| moved to phase 2 | #26, #35, CUDA MPS | D-019 addendum 10-15 |

## 2. The method frozen for phase 1 (`configs/frozen-v2.yaml`, D-033)

**Takeaway: the frozen configuration is A9's old default plus two changes, ld0 and ysync, each tested alone before
they ran together.**

**A9** ([[D-015 Use cuOpt's feasible solutions — three-way arm A9]]) runs three processes on one GB10, on footprint F
(D-021):

| process | where it runs | role |
|---|---|---|
| SCIP 10 | X925 core 5 | exchanges solutions with the EA both ways |
| the GPU EA (three ranks; CuPy pool 12 GB) | X925 core 6 | receives from SCIP and cuOpt, posts to SCIP |
| cuOpt 26.08 | A725 cores 1–3 + the GPU | streams its incumbents and bound to the EA (one way) |

- **cuOpt → EA, one way.** cuOpt never receives a solution: on 26.08 its `SetSolutionCallback` disables presolve. The
  EA injects cuOpt's incumbents (survival decides) and counts cuOpt's incumbent in its own cutoff. It passes to SCIP
  whatever beats SCIP's incumbent.
- **SCIP ↔ EA, both ways.** The EA's new incumbents go to SCIP through a heuristic plugin. SCIP sends its incumbent,
  pool solutions, its dual bound, its root LP point, and the LP stream: good infeasible node-LP points
  ([[D-011 Solver streams good infeasible solutions to the EA]]).
- **Harness `ext`** ([[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]]):
  cuOpt's MIP gaps are 0 and its thread variables unset; its RMM limit is 24 GB, separate from the EA's pool; host
  limit 24 GB; the project's CUDA JIT cache.

**The EA's configuration** = `configs/itrea2-dev-rc-3way.yaml` (EA hash `1d12ad490dd0`, A9 run hash under ext
`28e5a5e9c725`, A9@r13x of 10-13) plus two changes. The base file sets:
- the three ranks with the default Rank 1 (`top`; D-016's clearing rule was not adopted);
- the self-adaptive repair strategy (`strategy.enabled`);
- the EA's own HiGHS LP (`seeding.lp_source: highs_async`);
- the LP stream (`lp_stream.enabled`, D-011);
- deep repair (`repair.max_steps 20000`, `repair.stall_steps 1000`);
- the repair clock, active ([[D-012 Repair clock — adaptive repair depth, staged]]);
- no back-off (D-018 and D-023 were not adopted).

**What ysync-ld0 adds.** `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` (EA hash `835c7c5a249c`, A9 run hash under ext
`f345de0ab936`) makes exactly two changes to the old default:
1. **ld0** ([[D-026 lp_decode inside A9 on the EA's core after the LP worker (R2), development test]]):
   `repair.continuous: lp_decode`, `lp_decode_gate: lp_worker`, `lp_decode_nice: 0`.
   - What it does: a HiGHS worker re-optimises the continuous variables of a few candidates per generation, with their
     integers fixed ([[D-017 LP decode — re-optimise the continuous variables of selected candidates]]). In A9 it runs
     on the EA's core 6 and starts only after the EA's own HiGHS LP worker has finished.
   - Mechanism in D-026's runs: the decoder was ready 0.23–0.26 s after the gate (5.83 / 5.91 s on ca). Decoded
     incumbents per seed: item_placement 54 / 22 / 45 / 38, sing44 6 / 2 / 5 / 3. ca's LP-worker solve took 1.039× and
     0.940× of the control's (rule ≤ 1.1).
2. **ysync** ([[D-029 ysync gets one follow-up at seeds 2–5, judged on all seeds 0–5 with the failing seed kept — NOT ADOPTED ends R1 for v1]]):
   `host.cuda_sched: yield`.
   - What it does: CU_CTX_SCHED_YIELD on the EA's CUDA context. While the EA waits on the GPU, its host thread still
     spins at about 1.0 core per wall-second, but it yields to a runnable process on the same core. The LP worker that
     shares core 6 therefore gets CPU time.
   - GPU check of 10-15 (journal 10-15 07:05–07:17; D-025's bsync Result):
     - a competing process on the same core got 0.5187 of a core under CUDA's default and 0.9989 under `yield`, with
       the host still at 1.00;
     - in an A9-layout run under ysync the LP worker's CPU share was 0.933.
   - In D-025's ysync runs the LP worker finished about twice as early on ca (37.3 / 38.3 → 20.0 / 20.2 s).

The two changes were tested apart (D-026, D-029), each against the old default. They first ran together in D-030 (§ 4).
**`configs/frozen-v2.yaml`** is a byte copy of `-3way-ysync-ld0.yaml` (EA hash `835c7c5a249c`, A9 run hash under ext
`f345de0ab936`), written after Berk's C1 go and pinned in `tests/test_cuda_sched.py` (D-033). It stays byte-identical
until the prereg-v2 tag (D-021 R33). `make_jobs.ITX_A9_CONFIG` keeps naming `-3way.yaml`, the historical baseline the
committed job files were written with; new A9 jobs use `FROZEN_V2_CONFIG`.
A6F@c1r = A6F@r13x's job (run hash `82a371305f88`). A0 = SCIP alone (run hash `27f3171fac78`, sets `dev` /
`classprobe`, 2026-10-06 / 10-07).

## 3. The development record, D-010 to D-033

**Takeaway: 24 records; three tests changed A9 itself (ext D-022, ld0 D-026, ysync D-029); the back-off, blocking
sync and Rank 1 by clearing (on A3) were not adopted.**

How to read the "kind" column:
- **F** = framework or aim (not a test).
- **T** = development test with rules fixed before any of its runs was read, with two qualifications:
  - D-022's second amendment came with 7 of 58 runs done, none read;
  - D-029's thresholds were fixed before its new runs but with seeds 0–1 (8 of its 28 read runs, D-025's) already known
    (its amendment B1–B2).
- **D** = diagnostic (no adoption).
- **R** = the C1 reading.

Each row's numbers come from that record's Result or Status line. The D-011, D-012 and D-014 rows also use the
[[Development log]].

| record | kind | question | test | verdict | key numbers |
|---|---|---|---|---|---|
| [[D-010 Develop the algorithm first on a small development set]] | F | where to develop | — | decided (Berk 10-06) | 9 development instances at 120 s, ≥ 2 seeds; permanently out of the test set. The 8 class-probe instances (D-014) make the 17 |
| [[D-011 Solver streams good infeasible solutions to the EA]] | T | does SCIP's stream of good infeasible node-LP points help the coupled system (A3)? | A3@lps1 vs A3@ad1, 9 × seeds 0–1 | kept on; "not yet credited" (Development log) | mean d +0.05 (3 better / 1 worse); in A9's configuration |
| [[D-012 Repair clock — adaptive repair depth, staged]] | T | a per-generation repair depth | Stage 2, A3@rc1 vs same-day A3@lps1w | development default; two stated deviations | item_placement 4 / 4 seeds better, mean d +0.50; in A9's configuration |
| [[D-013 Aim — lift each solver's weakest cases]] | F | what success means | — | aim decided (Berk 10-07) | weak sets first, mean as guard (d ≥ −0.05) |
| [[D-014 Class probe — predictions fixed before the runs]] | T | does the coupling (A3) hold on 8 new instances? | 8 × {A0, A6, A3@lp1} × seeds 0–1, 48 runs | INCONCLUSIVE (3 of 5 predictions) | vs cuOpt alone 1 better / 5 worse, mean d −0.51 [−0.84, −0.12]. This loss out of sample motivated A9 |
| [[D-015 Use cuOpt's feasible solutions — three-way arm A9]] | T (pilot) | use cuOpt's solutions in the EA (A9) | pilot, 7 instances, seed 0 | built; pilot descriptive | vs SCIP alone 7 / 0, mean d +0.68 [+0.42, +0.90]; vs cuOpt alone 3 / 4, −0.07 [−0.42, +0.25]; vs the race A2 2 / 5, −0.18. chromaticindex P 0.545 inside A9 vs 0.071 alone |
| [[D-016 Rank 1 as a distinct elite — cuOpt's clearing rule, development test]] | T | Rank 1 by cuOpt's clearing rule (on A3) | cl1 vs rc1d, seeds 0–3, 102 runs | NOT ADOPTED | mean d −0.033 [−0.122, +0.050]; A9 keeps Rank 1 `top` |
| [[D-017 LP decode — re-optimise the continuous variables of selected candidates]] | T | LP decode on A3 | ld1 vs rc1e, 6 × seeds 0–1, 24 runs | ADOPTED for A3; refused in A9 (one EA core) → #30 → D-026 | mean d +0.219 [+0.041, +0.429], 5 better / 1 worse |
| [[D-018 A9 — the EA backs off the GPU beside cuOpt (contribution-gated duty cycle)]] | T | the EA yields the GPU to cuOpt while it does not contribute | stage 1, bo1 vs bo1s + A6, 14 × seeds 0–1, 84 runs | **NOT ADOPTED** (mechanism M1, M3, M5 fail; every outcome rule holds) | chromaticindex d +0.834 (P 0.545 → 0.091); the patience ladder stepped the EA down while it led |
| [[D-019 v1 — deadline and performance target]] | F | v1's deadline and targets | — | decided (Berk 10-10); C1 moved to Sat 10-17 (10-15) | targets in § 4 |
| [[D-020 MIPfeas harness — GAMS's clock and integral, paired calibration of cuOpt 26.08]] | F | the external metric and clock | — | narrowed by D-021 | GAMS's integral ported exactly: equal to GAMS's own function on all 3,495 traces; the call clock |
| [[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]] | F + measurements | an equal-footprint external standard | M1–M4 (§ 5) | accepted by Berk 10-11; M4 OUTSIDE 10-15 | § 5 |
| [[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]] | T + R (rules fixed before any reading, 7 of 58 itx-r13 runs already done) | A9 under ext (cuOpt gaps 0, thread variables unset); C1's interim hint and guard | set itx-r13 (job files itx-r13 + itx-r13b), 74 runs | **ADOPT** R13 + R14. The mechanism is met only vacuously (0 of 6) | no harm: fhnw ΔP −0.0016, krka −0.0001, neos-960392 d +0.022 (band 0.41). **C1 on the old default: HINT 0.827, GUARD +0.063** |
| [[D-023 A9 back-off, phase Y only — yield until cuOpt's first incumbent, then full duty]] | T | phase Y only (yield until cuOpt's first incumbent) | backoff2, 11 instances, 66 runs | **NOT ADOPTED** (failed M2, M5, T, G) | chromaticindex d +0.030; cuOpt's first incumbent inside boY 63.47 / 63.41 s vs A6F 10.87 / 10.52 s |
| [[D-024 Targeted first, wider only on a hint — staged testing for development and v1]] | F | how to test | — | decided (Berk 10-13) | performance runs first; targeted tests; a futility-only hint fixed before the runs |
| [[D-025 A9 EA blocking sync (R1) — the EA stops spinning on the GPU so its LP worker gets core 6, development test]] | T | R1: the EA stops spinning on the GPU, so its LP worker gets core 6 | r1-bsync, then fallback r1-ysync, 20 runs each | **NOT ADOPTED** (both variants) | bsync: hint 0.765, generation guard failed (item_placement 0.824 vs 0.921, comp21 0.818 vs 0.950). ysync: hint 0.834, item_placement's generation guard failed on one seed (678 vs 1,480 generations) |
| [[D-026 lp_decode inside A9 on the EA's core after the LP worker (R2), development test]] | T | R2: lp_decode in A9, gated after the LP worker | r2-ld, 28 runs | **ADOPT** (ld0) | hint 0.758 (≤ 0.90): item_placement P 0.330 → 0.198, sing44 0.473 → 0.411, both lower on all four seeds; every no-harm leg holds |
| [[D-027 Same-day diagnostic of cuOpt's thread count and SCIP's presence (R3) — reading rules fixed before the runs]] | D | why A9 loses on chromaticindex, toguru, doon, comp21 | r3-diag, 22 runs | rule 4 **EA PRESENCE**; rule 1 closed; rule 3 SCIP ruled out; rule 2 not read | cuOpt's first incumbent on chromaticindex: 9.80 / 7.33 s alone on F, 9.24 / 8.21 on the 3-thread lane, 6.53 / 9.65 beside SCIP, vs 62–66 s inside A9 |
| [[D-028 Does D-022's harness (R13 + R14) stop phase Y from restoring cuOpt on chromaticindex — a 4-run diagnostic]] | D | does R13 + R14 stop phase Y from helping on chromaticindex? | d028, 4 runs | **NOT R13 + R14** | A9@boY under r13-off 63.25 / 63.24 s vs L 19.82 s |
| [[D-029 ysync gets one follow-up at seeds 2–5, judged on all seeds 0–5 with the failing seed kept — NOT ADOPTED ends R1 for v1]] | T | ysync again at seeds 2–5, judged on seeds 0–5 with the failing seed kept | r1f-ysync, 21 runs (warm-up unread); 28 read | **ADOPT** (ysync) | generations item_placement 1.020, comp21 1.152 (≥ 0.90); P +0.016 / +0.015 (≤ +0.05); ca at seeds 2–3 0.675 (≤ 0.95) |
| [[D-030 C1 is re-read once on the frozen-v2 candidate (ld0) with A6F in the same window — 68 runs under D-022's hint and guard and D-025's harm leg]] | R | C1 on the frozen-v2 candidate (ysync-ld0), cuOpt alone re-run in the same window | c1r-ysync-ld0, 68 runs | VALID; **HINT 0.809, GUARD +0.050, HARM glass4** (+ krka at floor) → Berk → D-032 | § 4 |
| [[D-031 The phase-1 interim result rests on four seeds — C1 runs again at seeds 2–3 on the frozen-v2 candidate and is read with D-030's seeds 0–1]] | R | the phase-1 interim result on four seeds | c1r-ysync-ld0-s23, 68 runs (read with D-030's 68) | VALID 136 / 136; **HINT 0.798, GUARD +0.045**; harm beside: glass4, comp21 (edge) | § 7 |
| [[D-032 glass4's C1 harm flag is read in one window — old default vs yield + lp_decode vs lp_decode, 4 seeds]] | T (targeted) | is glass4's harm flag real or day-to-day drift? | d032, glass4 × 3 configurations × seeds 0–3, 12 runs | **DRIFT** → yield + lp_decode freezes | means old 0.2176, yield + ld 0.2233, ld 0.2177; d(candidate) −0.025 ≥ −0.057 |
| [[D-033 C1 go — phase 1 closes with configs-frozen-v2.yaml = yield + lp_decode]] | F | the C1 go / no-go | — | **go** (Berk 10-16 13:1x): phase 1 closes | `configs/frozen-v2.yaml` = yield + lp_decode (EA `835c7c5a249c`); harm flags, comp21's crashes and chromaticindex carried as caveats (§ 7, § 8) |

- **Built and not adopted for A9:** the back-off (D-018, D-023), blocking sync (D-025 bsync), and Rank 1 by clearing
  (D-016, on A3).
- **The ysync line took a forking path.** It was NOT ADOPTED in D-025 (whose Result said "no fallback of a fallback — the
  R1 line ends for C1") and then ADOPTED by a new record, D-029, written on Berk's choice after that failure.
  - Its rules were fixed before its runs, but its item generation bar was loosened after the failure was seen
    (0.921 → 0.90).
  - D-029 states that the follow-up adds about 12–19 points to the overall chance of adopting a harmful ysync (D-029
    § Forking paths).

## 4. The C1 readings

**Takeaway: in both two-seed readings the hint is present and the guard holds; on the candidate, glass4's harm flag
read as day-to-day drift (D-032). D-030 keeps comp21's two crashed cuOpt-alone runs as outcomes; by the letter (failed
runs excluded) it reads INCOMPLETE. The four-seed result, which closed phase 1, is in § 7.**

Two readings exist, each 17 instances × {A9, A6F} × seeds 0–1, 120 s, under harness ext, with both arms of each block
in one window:

| reading | date | arm | configuration | runs | set |
|---|---|---|---|---|---|
| 1 | 10-13 | A9@r13x | the old default `-3way.yaml` | 68 of 68 | set itx-r13 (job files itx-r13 + itx-r13b; D-022) |
| 2 | 10-15 / 10-16 | A9@c1r | the candidate `-3way-ysync-ld0.yaml` | 68 of 68, VALID | c1r-ysync-ld0 (D-030) |

In reading 2 cuOpt alone was re-run in the same window as A9 (A6F@c1r). D-030's pools were `20261015-150955-187322`,
halted by R26's breaker, and `20261016-070810-4125812`.

| instance | fixed 11 | 10-13: A6F@r13x | A9@r13x | d | D-030: A6F@c1r | A9@c1r | d | A0 (SCIP alone) |
|---|---|---|---|---|---|---|---|---|
| neos-960392 | yes | 0.710 | 0.058 | +0.919 | 0.644 | 0.042 | +0.935 | 1.000 |
| ca__Scenario400_reserves_1 | yes | 0.587 | 0.394 | +0.328 | 0.559 | 0.252 | +0.550 | 1.000 |
| item_placement_9903 | yes | 0.535 | 0.364 | +0.320 | 0.469 | 0.249 | +0.469 | 0.660 |
| sing44 | yes | 0.491 | 0.445 | +0.093 | 0.471 | 0.332 | +0.295 | 0.525 |
| neos-3024952-loue | yes | 0.756 | 0.564 | +0.254 | 0.760 | 0.539 | +0.291 | 1.000 |
| comp21-2idx | yes | 0.504 | 0.548 | −0.080 | 0.779 (a) | 0.567 | +0.272 | 0.891 |
| neos-2657525-crna | yes | 0.757 | 0.461 | +0.391 | 0.531 | 0.436 | +0.180 | 0.833 |
| neos-4763324-toguru | yes | 0.549 | 0.603 | −0.090 | 0.558 | 0.613 | −0.090 | 0.772 |
| rail01 | yes | 1.000 | 1.000 | 0 | 0.958 (b) | 1.000 | −0.042 | 1.000 |
| neos-2746589-doon | yes | 0.894 | 1.000 | −0.106 | 0.722 | 1.000 | −0.278 | 1.000 |
| cryptanalysiskb128n5obj16 | yes | 1.000 | 1.000 | 0 | 1.000 | 1.000 | 0 | 1.000 |
| glass4 | guard | 0.298 | 0.216 | +0.274 | 0.309 | 0.237 | +0.232 | 0.344 |
| neos-3656078-kumeu | guard | 0.095 | 0.103 | −0.076 | 0.086 | 0.118 | −0.272 | 1.000 |
| chromaticindex1024-7 | guard | 0.061 | 0.544 | −0.889 | 0.076 | 0.544 | −0.860 | 1.000 |
| radiationm18-12-05 | guard (floor) | 0.008 | 0.011 | −0.226 | 0.008 | 0.010 | −0.231 | 0.229 |
| fhnw-binpack4-48 | guard (floor) | 0.006 | 0.011 | −0.508 | 0.005 | 0.010 | −0.528 | 1.000 |
| neos-3004026-krka | guard (floor) | 0.006 | 0.003 | +0.465 | 0.004 | 0.004 | −0.077 | 0.659 |

Seed means. "(floor)" marks instances where both arms are at P ≤ 0.011, so a tiny ΔP gives a large bounded d.
- (a) **comp21-2idx: cuOpt alone crashed (rc −11, SIGSEGV) on both seeds**, on 10-15 and on 10-16. Both runs are kept as
  A6F outcomes (D-002 §6), truncated at the crash. In the saved report they score P 0.776 (seed 0) and 0.782 (seed 1);
  on 10-13 cuOpt alone scored 0.511 / 0.496 there.
- (b) rail01: cuOpt alone found an incumbent once (seed 1, P 0.916). A9 found none on either seed, so rail01 is no
  longer a tie at 1.000 in reading 2.
- The 10-13 columns are the report's recomputation on the run table of D-030's reading (10-16 08:2x); the 08:57 table
  gives the same values. They equal D-022's Result except on
  ca__Scenario400_reserves_1: 0.587 / +0.328 here, 0.586 / +0.329 there. ca's reference is provisional
  (`best_of_study`) and re-scores every ca run when a better ca solution is found (D-031 § Rules).

Read from the rows (counts, no new statistics):
- Over the 17, A9@c1r's seed mean is lower than A6F@c1r's on 8 instances and higher on 8, with 1 tie.
- A9 finds no incumbent in 120 s on doon, rail01 and cryptanalysiskb128n5obj16 in either reading.

**C1's rules (D-022 second amendment, fixed before any reading, with 7 of 58 itx-r13 runs already done; D-030's fixed
before any c1r run):**

| | 10-13 (old default) | D-030 (candidate, failed runs kept) | rule |
|---|---|---|---|
| HINT: ratio of means A9 / A6F over the fixed 11 | **0.827** (0.585 / 0.707) | **0.809** (0.548 / 0.677) | present iff ≤ 0.95 (futility-only) |
| GUARD: mean bounded d over the 17 | **+0.063** | **+0.050** | holds iff ≥ −0.05 |
| HARM leg (D-025 amendment 3: new A9 vs A9@r13x of 10-13, cross-day) | — | fires on **glass4** (d_i −0.088 < −0.057). Fires on **krka** with ΔP +0.0010 (below D-022's 0.005 floor, marked). Cannot fire on rail01, cryptanalysis, crna, doon | harm iff d_i < −band_i |

- **The letter (failed runs excluded):** D-030 reads INCOMPLETE. The HINT and the GUARD are unread because comp21's A6F
  runs are missing on both seeds. On the 10 instances with both arms: improved 6, worse 3, mean P 0.667 → 0.546 (−18 %).
- **The kept reading gains from comp21's crashes:** comp21's d moves from −0.080 (10-13) to +0.272. D-030 states this
  beside the go / no-go.
- **glass4's flag → D-032.** In one window with seeds 0–3, the three configurations are level on glass4: old default
  0.2176, yield + lp_decode 0.2233, lp_decode alone 0.2177. d(candidate) = −0.025 ≥ −0.057, so the reading is
  **DRIFT** and yield + lp_decode freezes for phase 1, by the rule Berk chose (08:2x), fixed before D-032's runs.
  Berk's C1 go then froze it, with D-031's four-seed harm flags beside (§ 7; D-033).
  - Beside the harm flag: the unchanged configuration's 10-15 replicate (A9@r2ctl) read d −0.061 on glass4, and cuOpt
    alone moved by −0.035 from 10-13 to D-030's window (D-030 Result).
  - The same harm rule applied to cuOpt alone (A6F@c1r vs A6F@r13x) fires on chromaticindex (d −0.203) and on comp21
    (−0.353, the crashes). The same-configuration A9 replicates of 10-15 fire on 2 of 15 instance reads. Both are
    measures of day-to-day drift (saved report, null lines).

**In D-013's terms.**

*vs cuOpt alone, on W_cuOpt = the fixed 11.* The 11 were chosen on earlier 3-core A6 runs (cuOpt-alone P ≥ 0.5), which
are separate runs from the ones compared.

| | improved / worse (of 11) | mean P A6F → A9 | mean d [95 % CI] | Wilcoxon p | worst P after |
|---|---|---|---|---|---|
| 10-13, old default (D-022 Result) | 6 / 3 | 0.707 → 0.585 (−17 %) | +0.185 [+0.032, +0.368] | 0.074 | 1.000 |
| D-030, candidate (D-030 Result; saved report) | 7 / 3 | 0.677 → 0.548 (−19 %) | +0.235 [+0.052, +0.433] | 0.049 | 1.000 |
| D-030, failed runs excluded (10 instances; saved report, `readings[1]`) | 6 / 3 | 0.667 → 0.546 (−18 %) | +0.231 [+0.031, +0.448] | 0.074 | 1.000 |

Guard vs cuOpt alone over the 17: +0.063 (10-13), +0.050 (D-030).

*vs SCIP alone (A0, sets `dev` / `classprobe`, seeds 0–1, 2026-10-06 / 10-07).*

| | n | improved / worse | mean P A0 → A9 | mean d [95 % CI] | Wilcoxon p |
|---|---|---|---|---|---|
| fixed 11 (all in SCIP's weak set), 10-13 (D-022 Result) | 11 | 8 / 0 | 0.880 → 0.585 (−34 %) | +0.331 [+0.172, +0.500] | 0.0078 |
| fixed 11, D-030 (D-030 Result; d and p from the saved report) | 11 | 8 / 0 | 0.880 → 0.548 (−38 %) | +0.382 [+0.207, +0.563] | 0.0078 |
| SCIP's weak set among the 17 (A0 P ≥ 0.5), 10-13 (saved report, `then`) | 15 | 12 / 0 | 0.889 → 0.473 (−47 %) | +0.465 [+0.293, +0.641] | 0.0005 |
| SCIP's weak set among the 17, D-030 (D-030 Result) | 15 | 12 / 0 | 0.889 → 0.447 (−50 %) | +0.502 [+0.329, +0.675] | 0.0005 |

- **The worst P after coupling is 1.000 in every row:** doon, rail01 and cryptanalysis, where neither A9 nor A0 finds
  an incumbent. These ties at 1.000 are counted at d = 0 in the mean P and the mean d, and dropped from the Wilcoxon
  test, whose zeros are dropped (`tools/d022_report.py` `_weak_block`, which both reports use).
- **No guard against SCIP alone is computed.** No report computes D-013's guard against SCIP alone (mean d over all 17 vs
  A0), and this note does not compute one. Read from the rows, A9's seed mean is not above A0's on any of the 17 in
  either reading.

**D-019's confirmatory targets, for scale only — not a test.** For the unseen hard instances, D-019 sets:
- **vs SCIP alone on W_SCIP:** minimum mean d > 0 with its 95 % CI > 0; target mean P −40 % and ≥ 70 % of W improved.
- **vs cuOpt alone on W_cuOpt:** the same minimum; target −25 % and ≥ 60 % improved.

Why for scale only:
- The development numbers above come from instances that were chosen and tuned on.
- The class probe already measured how much weaker the coupling was out of sample (D-014: vs cuOpt alone 1 better /
  5 worse, mean d −0.51 [−0.84, −0.12], with the older A3).

**Caveats of the C1 readings.**
- n = 2 seeds in this section (4 in § 7); development instances chosen and tuned on (D-010).
- **The fixed 11 were chosen on earlier 3-core A6 runs** under the old harness, and not re-selected. On A6F@r13x sing44
  (0.491) would fall just outside; comp21 was borderline at 0.506 (D-022 Result).
- **A0 is from earlier days** (10-06 / 10-07) on SCIP's own harness, not from the same window.
- **SCIP's weak set is selected on the same A0 runs it is compared with** (`tools/c1r_report.py`, `tools/d022_report.py`).
  D-013 asks for separate seeds. 9 of the 15 sit at A0 P = 1.000 on both seeds (no incumbent in 120 s). For those, the
  selection cannot regress, but sing44 (0.525), item_placement (0.660) and neos-3004026-krka (0.659; seeds 0.318 /
  1.000) are near the line (run table). krka is in SCIP's weak set only through one seed.
- **comp21's cuOpt-alone crashes are kept as A6F outcomes** and help the kept reading (above); the letter is INCOMPLETE.
- **Concentration:** on 10-13, neos-960392 alone carried 48 % of the fixed 11's summed P reduction (0.652 of 1.345); the
  ratio without it would have been 0.902 (D-022 Result).
- crna is bimodal in A9-type runs: SCIP's polishing cascade happened in 3 of 6 runs. At n = 2 that moves the fixed-11
  ratio by about ±0.04 ([[2026-10-14 C1 diagnosis and improvement candidates]] § 1).
- **Day-to-day drift is of the same size as the harm band:** glass4 (D-032) and the null lines above. In D-023's
  control item_placement read 0.218 against 0.364 on 10-13 (D-023 Result).
- **The candidate combines two changes, each tested alone** against the old default (D-026, D-029). Over the 17, D-030's
  harm leg is the only check of the combination, and it cannot say which change caused a difference. D-032 then compared
  the combination with lp_decode alone and the old default on glass4 only, in one window (DRIFT; d(g4ld) −0.0004).
- **ysync, half of the candidate, was adopted on a forking path.** D-025 did not adopt it ("no fallback of a fallback —
  the R1 line ends for C1"); D-029, a new record written on Berk's choice after that failure, adopted it. Its item
  generation bar was loosened after the failure was seen (0.921 → 0.90), and D-029 states that the follow-up adds about
  12–19 points to the overall chance of adopting a harmful ysync (D-029 § Forking paths; amendment B1–B2).
- **#36 touches glass4.** In the run table 11 glass4 runs have a checked incumbent below MIPLIB's proven optimum
  (1,200,012,599.97): 7 A3 runs of the dev set, A5@c04, A9@r2ld s0, A9@r13x s0 and A9@g4old s2. Two of them are in the
  C1 sets and D-032: A9@r13x seed 0 (best 1,200,012,216.4, P 0.221; part of the harm leg's 10-13 reference) and
  A9@g4old seed 2 (1,200,012,201.9, P 0.182). A third, A9@r2ld seed 0 (best 1,200,010,797.8, P 0.191), is D-026's ld0
  treatment on glass4; a re-score that rejected it would raise the treatment's P, moving D-026's glass4 leg (−0.030,
  band 0.050) against ld0. How #36's resolution would re-score these runs is not known. (A9@r13x item_placement seed 0
  also beats its reference, but that reference is a provisional screening best, not an optimum.)
- **D-030 spans two windows.** Its pool was halted by R26 on 10-15 and resumed on 10-16. Each block ran in one window,
  so every A9 / A6F pair is same-window.

## 5. What the external harness calibration showed (M1–M4)

**Takeaway: M1 and M2 passed and M3 dropped three instances; M4 reads OUTSIDE its band (R 1.313 against [0.870,
1.150]), and with the clock checked the difference is the footprint and the machine.**

These measurements test the harness, not the method
([[2026-10-11 D-021 measurements M1–M3 and the interim calibration]]).
- **M1 smoke: gate passed** (12 / 12). Every harness setting acts as specified. R14 (cuOpt's thread variables unset)
  measured inert: cuOpt CPU-s per wall-s was 5.02 / 4.99 / 5.03 with them unset and 5.02 / 4.98 / 5.03 with them set.
- **M2 soak: gate passed, F accepted** (4 runs of 600 s). Steady-state p95 66–73 °C, maximum 76 °C; no 5-s sample
  ≥ 90 °C, no clock cap; busy SM clock 2,392 MHz.
- **M3: three instances do not fit F** (eilA101-2, nw04, supportcase6). The decision set shrinks from 218 to 215
  instances, and H_cu-unseen from 65 to 64.
- **M4 calibration, complete (30 of 30), OUTSIDE** (note addendum 2026-10-15; journal 10-15 11:1x). cuOpt 26.08 alone on
  F (A6F) against GAMS's published cuOpt-26.08 GB10 run on D-020's 30-instance sample at 600 s:
  - R = **1.313 [1.091, 1.618]** on the call clock (launch clock 1.963 [1.606, 2.408]), against the band
    [0.870, 1.150];
  - feasible 27 vs GAMS's 27;
  - the late-clock check passed: median first incumbent, ours − GAMS's, over strata 0–1 was −0.119 s (n 10).

  With the clock checked, the difference is **the footprint** (5 threads + the GPU, against GAMS's 24-thread whole box)
  **and the machine**, as D-021 expected. v1's external targets are therefore anchored on our own A6F, not on the
  published row. The interim figures quoted before 30 of 30 are superseded.
- Also measured for the external runs:
  - **M6:** A0 runs on 3 lanes. The nodes/s slowdown beside 3 other lanes was 13.2 % (> 3 %); hottest 78.6 °C
    (journal 10-13 22:2x).
  - **A0 on the 81 instances D-024 (f) reads:** 80 completed + 1 failed. In s100 (seed 2000) SCIP did not stop at its
    600-s limit and was killed at 701 s (rc −9) (journal 10-14 21:48).
  - D-020: GAMS's integral is ported exactly (equal to GAMS's own function on all 3,495 traces).

## 6. Known losses

**Takeaway: the largest loss, chromaticindex1024-7, comes from the EA's presence slowing cuOpt; doon sits at cuOpt
alone's edge; four more instances are worse than cuOpt alone in both readings.**

- **chromaticindex1024-7: the EA's presence slows cuOpt.** Inside A9, cuOpt's first incumbent comes at about 65 s;
  without the EA it comes in 6.5–11 s.
  - D-027 rule 4: alone on F 9.80 / 7.33 s, on the 3-thread lane 9.24 / 8.21 s, beside SCIP 6.53 / 9.65 s, against
    62–66 s inside A9 on every A9 run on record except three back-off arms on the old harness: bo1 9.38 / 12.35 s and
    the forced duties bof25 9.81 s and bof50 8.33 s (run table `cuopt_t_first`; D-027 amendment 18:3x). Phase Y alone
    (boY, under ext and under r13-off) and the forced duty bof12 read 62.3–63.5 s.
  - Run table, D-030: inside A9@c1r 65.24 / 65.22 s (P 0.544 / 0.544), against 7.94 / 10.31 s for A6F@c1r (P 0.066 /
    0.086).
  - The EA's generation 0 there is a 48-s GPU repair: about 1,300 back-to-back launches of about 40 ms (D-027).
  - What was ruled out on chromaticindex: SCIP's presence and the 3-thread lane (both by D-027 rule 4). Rules 1 and 3
    closed the thread and SCIP questions on toguru and doon. Phase Y at duty ½ does not restore cuOpt under ext (D-023:
    63.47 / 63.41 s), and that is not caused by R13 + R14 (D-028: 63.25 / 63.24 s under r13-off).
  - Under the old harness, D-018's bo1 did give cuOpt its first incumbent back at 9.38 / 12.35 s (run table
    `cuopt_t_first`; D-023's Result rounds it to 9.4 / 12.4 s). What in ext differs (the separate 24-GB RMM pool, the
    host cap, the cache, or the day) is not pursued for v1 (D-028).
  - CUDA MPS is the GPU-sharing candidate; Berk 10-15: **"Not for v1"**. chromaticindex is a known v1 loss to cuOpt
    alone, d −0.889 (10-13) and −0.860 (D-030), and the largest single negative term in the guard.
- **neos-2746589-doon sits at cuOpt alone's edge.** A9 finds no incumbent in 120 s on any seed (P 1.000). cuOpt alone
  finds one only sometimes:

  | run | seed 0 | seed 1 |
  |---|---|---|
  | A6F@r13x (10-13) | 0.792 | 0.996 |
  | A6F@r3 (D-027) | 1.000 | 1.000 |
  | A6F@c1r (D-030) | 0.476 | 0.967 |

  D-027 reads doon's C1 "loss" as sitting on cuOpt alone's edge. Its d is −0.106 (10-13) and −0.278 (D-030); the harm
  leg cannot fire there.
- **Also worse than cuOpt alone in both readings (from the table):**
  - neos-4763324-toguru: d −0.090 both times;
  - neos-3656078-kumeu: d −0.076 and −0.272 (P 0.086 → 0.118);
  - the floor instances radiationm18 and fhnw-binpack4-48, where both arms are at P ≤ 0.011.
- **The no-incumbent trio** (doon, rail01, cryptanalysis): A9 finds no incumbent in 120 s on any seed of either reading
  (run table: `best_obj` empty, `t_first` 120 s).

## 7. Four-seed result (D-031) — the phase-1 interim result

**Takeaway: on four seeds C1's hint is present (0.798) and its guard holds (+0.045); the two halves agree. Two harm
flags stand beside, not as criteria: glass4 (read as drift by D-032) and comp21-2idx (at the edge of its band).**

- **Set:** `c1r-ysync-ld0-s23`.
- **Job file:** `jobs/d031.jsonl`, sha256 `f83e6888934ac3b2e243be880a8c823ae83b787462982c5d5f59b316dc9a3198` (the report
  confirms it is D-031's).
- **Runs:** 68 = the 17 × {A9@c1r, A6F@c1r} × **seeds 2–3**, 120 s, harness ext.
- **Arms:** the same jobs as D-030's file except seed, set, stratum, block and position (A9 run hash `f345de0ab936`, A6F
  `82a371305f88`).
- **Ran:** pool `20261016-085742-499378` in window W-1016, 2026-10-16 08:57:43–12:55:12, 34 blocks: **67 completed,
  1 failed, 0 void**.
- **Read with:** `itrea build-runs`, then
  `OMP_NUM_THREADS=1 nice -n 19 taskset -c 10-14 .venv/bin/python tools/d031_report.py --d030-json logs/d030-reading.json`
  (saved as `logs/d031-reading.json`; [[D-031 The phase-1 interim result rests on four seeds — C1 runs again at seeds 2–3 on the frozen-v2 candidate and is read with D-030's seeds 0–1]] § Result).

| field (D-031 § "The reading") | value |
|---|---|
| validity over 136 runs (D-030's 68 + D-031's 68) | **VALID, 136 of 136** (failed runs kept as outcomes of their arm, D-002 §6) |
| runs completed / failed / void in `c1r-ysync-ld0-s23`, pool id(s), window(s) | 67 / 1 / 0; pool `20261016-085742-499378`, W-1016. The failure: comp21-2idx A6F@c1r seed 2, cuOpt rc −11 (SIGSEGV) |
| **PHASE-1 INTERIM RESULT** line, verbatim ¹ | `PHASE-1 INTERIM RESULT (four seeds 0-3 (D-030 + D-031, 136 runs)): HINT present (ratio 0.798, present iff <= 0.95; futility-only, D-024 rule 2: a hint never declares success), GUARD holds (mean d 0.045, holds iff >= -0.05); beside: harm glass4, comp21-2idx; selection check (D-024 rule 2, the stage-2-only figure): seeds 2-3 alone HINT present (ratio 0.787) — the four-seed ratio includes seeds 0-1, whose hint let D-031 run, so it is optimistic given that` |
| four-seed HINT: ratio of means A9 / A6F over the fixed 11 (present iff ≤ 0.95; futility-only) | **present, 0.798** (P̄ 0.550 / 0.690) |
| selection check (D-024 rule 2): seeds 2–3-alone HINT ratio | **0.787**, present (seeds 0–1 alone, = D-030: 0.809) |
| four-seed GUARD: mean bounded d over the 17 (holds iff ≥ −0.05) | **holds, +0.045** (seeds 2–3 alone +0.034; seeds 0–1 +0.050) |
| four-seed HARM list (per harm: P̄ 10-13 → new, ΔP, d_i, band_i, the replicates' and cuOpt alone's d) ² | **glass4:** P̄ 0.216 → 0.244, ΔP +0.028, d_i −0.114, band 0.057; same-configuration replicate A9@r2ctl d −0.061; cuOpt alone, 10-13 → this window, d −0.067. **comp21-2idx:** P̄ 0.548 → 0.577, ΔP +0.030, d_i −0.051, band 0.050; replicates A9@r1ctl +0.046 and A9@r1ctl-ysync −0.046; cuOpt alone d −0.272. Cannot fire: rail01, cryptanalysiskb128n5obj16, neos-2657525-crna, neos-2746589-doon |
| do the halves agree (each half's hint and guard; per-instance sign flips) | **yes.** Both halves: hint present, guard holds. Per instance the sign of d against A6F is the same in both halves on 15 of 17, opposite on none, and zero in one half on 2 (rail01, doon). Each half's own harm list (not the four-seed list): seeds 0–1 glass4, krka (floor); seeds 2–3 glass4, radiationm18-12-05, comp21-2idx |
| seeds 0–1 against the saved D-030 JSON (every difference; a move on ca is the re-scoring D-031 states) | **equal** (hint, guard, harm list, thresholds, A9@r13x's seeds); no difference, ca included |
| D-013 lines, four seeds: vs A6F on the fixed 11; vs A0 on the fixed 11 and on SCIP's weak set among the 17 | vs A6F (same window per block), fixed 11: improved 7, worse 3, P 0.690 → 0.550 (−20 %), d +0.249 [+0.078, +0.441], Wilcoxon p 0.027 (seeds 2–3 alone: 7 / 1, −21 %). vs A0 (earlier days, seeds 0–1): fixed 11 8 / 0, −38 %; SCIP's weak set among the 17 (15): 12 / 0, 0.889 → 0.448 (−50 %), d +0.499 [+0.323, +0.676], p 0.0005 |
| comp21's A6F runs at seeds 2–3 (crash or not) and the letter's reading | seed 2 crashed (SIGSEGV; the third cuOpt-alone crash on comp21 in four seeds, after D-030's seeds 0 and 1); seed 3 completed. **The letter** (the three crashed runs left out): **INCOMPLETE** — comp21's A6F is missing at seeds 0, 1 and 2, so hint and guard are unread. The kept reading counts the crashes against cuOpt alone (its P̄ on comp21 0.692, against 0.504 on 10-13), which favours A9 on comp21 |

1. Re-printed from the saved JSON by the report's own line function (`_phase1_line` in `tools/d031_report.py`); the line
   ends as the tool words it.
2. The replicates' and cuOpt alone's d come from the saved reading's harm notes (`logs/d031-reading.json`,
   `readings[0].four.harm.notes`); they are measures of day-to-day drift.

Stated before the reading (D-031 § Rules), and still true:
- The four-seed ratio is slightly optimistic, because D-031 ran only after D-030's hint was present. The seeds 2–3-alone
  ratio (0.787) is printed beside it as the selection check.
- More seeds of the new A9 do not shrink the error of the 10-13 reference (A9@r13x, seeds 0–1) that the harm leg reads.
- The halves are not day-separated. 16 of D-030's 34 blocks (32 runs) ran on 10-16 in W-1016, the same day and lease as
  D-031. The halves differ in pool and window id, and partly in day, as well as in seed (D-031 § Validity).
- D-031 read VALID, so the four-seed reading is the phase-1 interim result; D-030's reading (§ 4) is its seeds 0–1 half.

**What Berk decided** (2026-10-16 13:1x, question panel): *"Go: freeze the candidate (Recommended)"* —
`configs/frozen-v2.yaml` = yield + lp_decode; phase 1 closes on these numbers; the two harm flags, the cuOpt-alone crashes
on comp21 and the chromaticindex loss go into phase 2 / v1 as stated caveats ([[D-033 C1 go — phase 1 closes with configs-frozen-v2.yaml = yield + lp_decode]]).

## 8. What moves to phase 2, and why

**Takeaway: #26, #35, CUDA MPS and chromaticindex's known loss move to phase 2, with the caveats Berk's C1 go carries;
v1's own work follows C1.**

(D-019 addendum 2026-10-15 13:4x; D-033.)
- **Caveats carried into phase 2 / v1 (D-033, as Berk chose):** the harm flags on glass4 (cross-day; D-032 read it in
  one window as drift) and comp21-2idx (d_i −0.051 against its band 0.050); cuOpt alone's SIGSEGV on comp21-2idx on 3 of
  4 seeds (10-15 / 10-16, not 10-13), counted against cuOpt alone, with the reading that leaves them out INCOMPLETE;
  chromaticindex1024-7's loss; development results on 17 tuned-on instances, descriptive, SCIP alone from earlier days.
- **#26 — recombine each new partner incumbent with the whole elite.** "Not built" (D-019 addendum 2026-10-15); it has
  no decision record yet ([[Roadmap]]). D-019 had listed it as decided by C1; the 10-15 addendum carries it to phase 2
  with its record still to write.
- **#35 — two-way exchange through cuOpt 26.10's `SetSolutionCallback` (v1.1).** cuopt#2073 is fixed in 26.10 (PR #2098)
  and was verified on the nightly 26.10.0a294. But cuOpt polls SET callbacks only when its GPU population records a new
  best (crna: 11 calls, all at 0.70–0.71 s; sing44: first call at 28.3 s; journal 10-13 10:00), which limits the gain.
  Berk (10-14): **"v1.1, no nightly runs"**. v1 keeps the one-way exchange.
- **CUDA MPS (v1.1)** — the GPU-sharing candidate for chromaticindex's loss (§ 6). Berk (10-15): **"Not for v1"**.
- **chromaticindex1024-7 as a known loss.** It stays in every reading and is reported, not excluded.
- After C1, the v1 work is: cuOpt 26.10 (#31: separate venv, API check, A6 re-baselined on the 17), prereg-v2 (#32), the
  confirmatory and external runs (D-021 ITX-1, staged by D-024), the pool freeze (#1) and the paper (#12) (D-019
  addendum 10-10).

## 9. Due before prereg-v2 (not in phase 1)

**Takeaway: three items are due before prereg-v2, after phase 1; v1 runs on cuOpt 26.10 (D-033).**

- **#36 — one checker rule for every arm.**
  - Our exact checker accepted incumbents below MIPLIB's proven optimum on glass4 (cuOpt rejects that point) and on
    mas74.
  - It rejected cuOpt incumbents on snowy (1) and neos-933966 (3).
  - In the C1 sets it touches glass4 (§ 4, caveats).
- **#32 — the confirmatory test-set size, Berk's choice.** D-002 v3.1 says 32 instances × 2 seeds; D-010 §4 proposed 16
  (#11, minimum detectable effect ≈ 0.3 at 16 vs ≈ 0.20–0.25 at 32); D-019's time estimate assumed about 32.
- **The prereg-v2 tag itself.** It gates only external A9 runs, which wait for cuOpt 26.10 (D-019).

## 10. Lessons (box and operations)

**Takeaway: window time was lost to a refused lease, to heat and to the fan governor's rests; the fixes and planning
rules are below.**

- **A lease longer than 12 h is refused.**
  - What happened (10-14): the scheduled chain claimed `--until 20:15` (13 h 10 min) at 07:05. Every claim was refused
    ("a lease longer than 12 h is not a window"), the claim retried until 08:00, and the chain exited with nothing
    stopped.
  - The transient unit had no OnFailure alert, so 7 h 36 min of a daytime window were lost.
  - Fixed in the itrea-window skill: leases ≤ 12 h; `OnFailure=agent-failure-alert@` on the chain; the session checks
    the chain's output within 15 min (journal 10-14 07:05; [[Mistakes and incidents]]).
- **R26's breaker tripped on heat (10-15 17:30).**
  - After the fan governor's rest (16:44–17:14), the hottest zone reached 85–87 °C within about 4 min at maximum fan,
    with the GPU at 15–20 W.
  - The governor capped the clock at 2,200 MHz, which voided item_placement seed 1's A9 run and then its rerun's A6F run
    (busy SM median 2,177 MHz). The block was skipped and the pool halted at 36 completed, 1 failed, 2 void of 68 (the
    journal's count). The void log holds three entries: the rerun's A9 had completed and was voided with its block, so
    the run table keeps 35 completed + 1 failed for that pool.
  - Another session's CPU-heavy, unpinned work had run inside W-1014 and W-1015 without a lease. Load averages at pool
    starts were 5.7–6.0. Unpinned processes can land on our pinned cores and add heat.
  - Since 10-15 19:0x that session's units run on cores 10–14 at nice 19 (its answer, checked with `systemctl --user
    show`).
  - Both arms of every block share their window, so the comparisons stay paired, but noise rises (journal 10-15 17:30,
    19:0x).
- **The fan governor's rests take about 25 % of window time** (the journal's estimate).
  - By 14:57 on 10-15 there had been four rests of 30 min (the fourth 14:27–14:57), about every 1.5–2 h under load
    (journal 10-15 13:4x–14:4x). A fifth followed at 16:44–17:14 (journal 10-15 17:30).
  - Plan 120-s run sets at 3.44–3.83 min per run with the rests included (D-030, D-031 cost sections).
- **Stage named files, not `vault/`, while an agent edits the vault.**
  - A 13:43 commit (`git add vault`) on 10-15 swept in D-030 while its build agent was still writing it (placeholders
    COMMIT_SHA, TEST_COUNTS).
  - No run read that version, and the final text (d50c03c) was committed before any c1r run (journal 10-15
    13:4x–14:4x).
- Earlier the same week ([[Mistakes and incidents]], 10-13):
  - A chain whose tail is the restore left the box idle for 2 h 17 min of a daytime window. A chain's tail should be
    the next performance run set.
  - The pool enforces R26's breaker but not its opening gate (#38; stopgap `tools/r26-gate.sh`).

## About this note

The full status of this note, moved here from the top on 2026-10-16:

- **Final.** Written 2026-10-16 by AgentRA (Claude Code (Anthropic Claude Opus 5.5)) as a draft while D-031 ran; § 7
  and the summary were filled in after D-031's reading (12:55) and Berk's C1 go (13:1x), the same day.
- **D-031 is read.** Set `c1r-ysync-ld0-s23` (`jobs/d031.jsonl`, 68 runs) ran 08:57:43–12:55:12; its reading with
  D-030's seeds 0–1 is § 7.
- **What kind of data.** Every A9 result here is development data
  ([[D-010 Develop the algorithm first on a small development set]]). § 5's harness measurements are calibration runs on
  D-020's MIPfeas sample and validity counts on unseen instances, not results of the method.
- **INTERIM and descriptive.** The A9 results rest on 17 development and class-probe instances that were chosen and
  tuned on: n = 4 seeds for the C1 result (§ 7), n = 2 in § 4's tables.
- **"HINT present" is futility-only**
  ([[D-024 Targeted first, wider only on a hint — staged testing for development and v1]] rule 2). It gates widening and
  declares nothing.
- **No claim beyond the numbers.** Nothing in this note says that A9 "beats", "improves on" or "matches" anything beyond
  the numbers printed.
- **The go / no-go on C1 was Berk's: go** ([[D-033 C1 go — phase 1 closes with configs-frozen-v2.yaml = yield + lp_decode]];
  [[D-019 v1 — deadline and performance target]], addendum 2026-10-15).
- **The freeze file.** `configs/frozen-v2.yaml` was written 2026-10-16 after Berk's go: a byte copy of
  `-3way-ysync-ld0.yaml` (EA hash `835c7c5a249c`), its hash pinned in `tests/test_cuda_sched.py` (D-033). The D-030
  and D-031 reports, run before it existed, record it as absent. It holds yield + lp_decode, as
  [[D-032 glass4's C1 harm flag is read in one window — old default vs yield + lp_decode vs lp_decode, 4 seeds]]'s rule
  (fixed before its runs) said it would.

## Sources

Each number above comes from one of these:
- **Summary, § 1, § 2** — [[D-013 Aim — lift each solver's weakest cases]] (quote, metric); [[D-019 v1 — deadline and performance target]]
  (addenda 2026-10-10 and 2026-10-15); D-033 (Berk's go, `configs/frozen-v2.yaml`, the caveats); D-031 § Result.
- **§ 2** — the configuration files `configs/itrea2-dev-rc-3way.yaml`, `-3way-ld0.yaml` and `-3way-ysync-ld0.yaml` (read,
  not edited); hashes from D-030's candidate table and D-026; the decoder mechanism from D-026's Result; the yield GPU
  check from journal 2026-10-15 07:05–07:17; the LP worker finish times from D-025's ysync Result; A0's run hash from
  the run table.
- **§ 3** — each record's Result or Status line. D-011's +0.05, D-012's Stage 2 numbers and D-014's vs-cuOpt line
  (1 / 5, mean d −0.51 [−0.84, −0.12]) also come from [[Development log]] ("LP stream"; D-012 addendum 2; "Class probe —
  D-014 result"). The legend's D-022 and D-029 timing: D-022's second-amendment heading and its 17:5x addendum (commit
  times); D-029 amendment B1–B2 and its 12:3x confirmation. The ysync sentence: D-025's ysync Result; D-029 § Forking
  paths.
- **§ 4** — the per-instance table:
  - 10-13 columns: the saved D-030 report `logs/d030-reading.json` (`readings[0].rules.then`, the 10-13 reading the
    tool recomputes from the run table; equal to D-022's Result except ca's re-scored third decimal, noted under the
    table);
  - D-030 columns: the same file (`readings[0].rules.guard.rows`), cross-checked against `datasets/runs.parquet` (built
    2026-10-16 08:57);
  - A0 column: the run table (arm A0, sets `dev` / `classprobe`, seeds 0–1, 120 s);
  - comp21's per-seed P, rail01's seed 1 and doon's seeds: `readings[0].runs` and `rules.nulls` of the same file.

  The rules table and the D-013 lines come from D-022's Result and D-030's Result. The D-030 vs A0 d, CI and p on the
  fixed 11 and the 10-13 line on SCIP's weak set among the 17 come from the saved report (`vs_a0`, `then.vs_a0_17`);
  the failed-runs-excluded line's d, CI, p and worst P from `readings[1].rules.vs_a6f`. D-032's numbers come from its
  Result. The caveats cite D-022, D-023, D-026 (glass4 leg), D-029 (Forking paths) and D-030 Results, the C1 diagnosis
  note § 1, and the run table (`beats_ref`, `best_obj`, `ref`, `n_inc_ok` for glass4 and item_placement; A0's `pt` on
  krka).
- **§ 5** — [[2026-10-11 D-021 measurements M1–M3 and the interim calibration]] (verdicts; addendum 2026-10-15 for M4);
  journal 2026-10-15 11:1x (M4); journal 2026-10-13 22:2x (M6); journal 2026-10-14 21:48 (A0 on the 81); D-020's
  summary (3,495 traces).
- **§ 6** — D-027 Result (times, generation 0, rules 1, 3 and 4) and its amendment 18:3x (the back-off exceptions);
  D-028 Result; D-023 Result; the run table (`cuopt_t_first` for chromaticindex on every A9 run, incl. bo1, bof12,
  bof25, bof50 and boY; `pt` for chromaticindex in set c1r-ysync-ld0); `logs/d030-reading.json` (doon's seeds); journal
  2026-10-15 09:2x (Berk on MPS).
- **§ 7** — D-031 (set, job file and sha256, The runs, Validity, rules, what the tool prints) and its § Result;
  `logs/d031-reading.json` (validity, the four-seed and per-half readings, `halves`, `vs_saved_d030`, the harm rows and
  notes, comp21's runs); the PHASE-1 line re-printed from that file by `tools/d031_report.py`'s `_phase1_line`;
  `logs/d031-1016.log` (pool end 12:55:12: 67 completed, 1 failed, 0 void); journal 2026-10-16 08:52 (start), 12:55
  and 13:1x; D-033 (Berk's decision); the run table (`window_id` of set c1r-ysync-ld0: 36 runs in
  20261015-150955-187322, 32 in 20261016-070810-4125812).
- **§ 8–9** — D-033 (caveats; still due before prereg-v2); D-019 addendum 2026-10-15; journal 2026-10-13 10:00 and 11:xx (cuopt#2073, polling); journal 2026-10-14
  16:0x (Berk on #35); journal 2026-10-15 09:2x (Berk on MPS); AGENTS.md (#36's instances, #32's sizes); D-010 §4.
- **§ 10** — journal 2026-10-14 07:05–14:41; journal 2026-10-15 13:4x–14:4x, 17:30, 19:0x; [[Mistakes and incidents]]
  (rows 2026-10-13 20:17, 22:34; 2026-10-14 07:05); D-030 and D-031 cost sections.
