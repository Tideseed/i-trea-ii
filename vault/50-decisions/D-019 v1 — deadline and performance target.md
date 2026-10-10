# D-019 v1 — deadline and performance target (2026-10-10, Berk; proposal by AgentRA)

**Status:** decided by Berk 2026-10-10 12:2x (addendum below: his answers change the deadline, the cuOpt version and the night rule; the proposal follows as first written). Berk (2026-10-10): "So let's put a target (both as deadline and performance) to our little project
for v1." This record is the proposal; it becomes decided with his answer (dated addendum), and its performance targets
enter prereg-v2 verbatim before any confirmatory run.

## What v1 is
**I-TREA II v1 = arm A9 frozen: SCIP 10 ↔ GPU EA ← cuOpt 26.08 on one GB10 (DGX Spark)**, with the development
defaults that survive stage 1 of D-018 (back-off) and #26, plus two results: (i) a pre-registered confirmatory test on
unseen hard instances (the D-013 aim) and (ii) an external run under Mittelmann's MIPfeas protocol. cuOpt stays at
26.08 for v1 — the version of the published MIPfeas rows; 26.10 is v1.1.

## Deadline — Fri 2026-11-13 (5 weeks), milestones
| by | milestone |
|---|---|
| Fri 10-16 | D-018 stage 1 (two windows) decided; the full-box A9 layout (cores for SCIP / EA + lp_decode / cuOpt) decided; MIPfeas harness: clock from the first process's launch, Mittelmann's primal integral (0–2, 600 s), the 233 instances; cuOpt-alone calibration on ~20 of them |
| Fri 10-23 | #26 decided; **configuration frozen**; prereg-v2 written (test-set rule, the targets below, the analysis) and tagged |
| Fri 11-06 | runs done: confirmatory (~32 unseen hard instances × 2 seeds × {A0 SCIP, A6 cuOpt, A9}, T = 300 s ≈ 16 h) and MIPfeas (233 × 600 s × {cuOpt alone, A9} ≈ 78 h) — ≈ 100–110 h of box time with rests: needs night windows |
| Fri 11-13 | analysis, results notes, tag `v1.0`, short report (the paper's first results section, #12) |

## Performance target (minimum = v1 passes / target / stretch)
**A. Confirmatory, unseen hard instances (primary — D-013: lift each solver's weakest cases, mean as guard).**
Weak set W_B = instances where solver B alone has P ≥ 0.5 on screening seeds (never the evaluation seeds).
| | minimum | target | stretch | development set today (for scale; optimistic — tuned there) |
|---|---|---|---|---|
| vs SCIP alone, on W_SCIP | mean d > 0, 95 % CI > 0 | mean P −40 %, ≥ 70 % of W improved | −55 % | 0.884 → 0.388 (−56 %), 6 of 7, d +0.48 |
| vs cuOpt alone, on W_cuOpt | mean d > 0, 95 % CI > 0 | mean P −25 %, ≥ 60 % improved | −40 % | 0.716 → 0.344 (−52 %), 5 of 6 (A3; A9 not yet measured on the full set) |
| guard, all instances | mean d ≥ −0.05 vs EACH solver alone | | | vs cuOpt alone +0.19 [−0.29, +0.60] |
Out-of-sample warning recorded now: the class probe (2026-10-07, older A3) helped SCIP on only 2 of 8 weak instances;
A9 (cuOpt's solutions in the loop) is what fixed those in the pilot.

**B. External — Mittelmann's MIPfeas protocol** (233 feasible MIPLIB 2017 benchmark instances, 600 s, primal integral
0–2, shifted geometric mean with shift 0.001; published 16 Sep 2026: cuOpt-26.08 on GB10 **0.0393**, on B200 0.0286,
SCIPConc-11 0.0274, ReXi 0.0165, SCIP-10.0.1 0.0755, HiGHS-1.15.1 0.0453). The page publishes no per-instance results,
so we run cuOpt alone ourselves (calibration and the per-instance baseline).
| | |
|---|---|
| calibration (validity, not a target) | our cuOpt-alone geo-mean within ±15 % of the published GB10 0.0393 — otherwise the published row is not our reference and we say so |
| minimum | A9 ≤ our cuOpt-alone geo-mean (no worse than cuOpt on its own home metric, easy instances included), solutions found ≥ cuOpt's |
| target | A9 ≤ **0.0286** (a GB10 running the coupling matches cuOpt on a B200) and, on the hard part (our cuOpt-alone PI ≥ 0.1 or no solution), mean PI −25 % |
| stretch | A9 ≤ 0.0274 (SCIPConc-11 level: the best open-source entry but ReXi) |
Caveat stated in advance: the geometric mean with shift 0.001 rewards fast first solutions on EASY instances — not our
aim (Berk, 10-06: "we don't expect to solve simple problems a few seconds faster at all"); hence the external minimum
is non-inferiority and the hard-part reduction is the target. The 233 overlap our development / class-probe instances;
the external result is reported with and without them.

## What it needs
- Box time ≈ 100–110 h between 10-26 and 11-06 with both LLM servers stopped inside the windows: **night windows
  (D-009 needs Berk's yes)**.
- No downloads (the 233 instances are on disk: `data/miplib/benchmark-v2.test` minus its 7 infeasible ones).
- If a target is missed, v1 still ships on 11-13 with the honest numbers and a v1.1 scope.

## Addendum — Berk's decision (2026-10-10 12:2x)
Asked (deadline; night windows), Berk: **"we shall have a checkpoint in the next 2 weeks and end in november with the new
cuopt version"** and **"No nights"**. So:
- **Checkpoint C1 — Fri 2026-10-23.** Development on cuOpt 26.08 closes: D-018 stage 1 decided; #26 decided; lp_decode in
  A9 (core layout); MIPfeas harness built and **calibrated** — cuOpt-26.08 alone on a ~30-instance sample of the 233 vs the
  published GB10 row (0.0393; validates our clock and metric); an interim A9 result on the 17 development + class-probe
  instances × 2 seeds vs SCIP alone and cuOpt alone, reported in D-013 terms; go / no-go for the final phase.
- **v1 — end of November (aim Fri 2026-11-27, hard end Mon 11-30), on the new cuOpt (26.10).** When 26.10 is released:
  separate venv, API check, re-baseline A6 on the 17 instances, then freeze, prereg-v2 (targets above, the external ones
  re-anchored below) and the runs. **Contingency, stated now:** if 26.10 is not released by Fri 11-06, the v1 runs proceed
  on 26.08 so v1 still lands in November, and 26.10 follows as an addendum (to be confirmed with Berk at that point).
- **No night windows** (00:00–07:00): daytime windows only, with the fan governor's rests; the MIPfeas runs (≈ 78 h of
  run time for {cuOpt alone, A9} × 233 × 600 s) need ≈ 8–10 long daytime windows — the LLM servers are off inside them.
- **External targets re-anchored to our own same-version run** (the published MIPfeas rows are cuOpt 26.08): minimum —
  A9's geo-mean ≤ our cuOpt-26.10-alone geo-mean on the same GB10; target — mean PI −25 % on the hard part (our
  cuOpt-alone PI ≥ 0.1 or no solution) and geo-mean −10 %; stretch — geo-mean ≤ 0.0274 (SCIPConc-11, published). The
  confirmatory targets (part A) stand as proposed.
