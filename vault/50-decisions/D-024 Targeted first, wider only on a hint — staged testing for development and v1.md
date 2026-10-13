# D-024 Targeted first, wider only on a hint — staged testing for development and v1 (2026-10-13, Berk; drafted by AgentRA)

**Status:** the development rules (0–4, "Development") are decided by Berk's instructions of 2026-10-13 and AgentRA's
reading of them, open to a superseding decision on evidence. **The v1 changes ("v1, proposed") change designs Berk
accepted on 10-11 (D-021 R21, R32) and wait for his yes per item ("Open for Berk"); until then D-021 stands for v1.**
Revised 15:4x after an adversarial check (two Claude Code subagents: the staging is statistically sound; the draft had
no threshold for C1's hint, an unrepresentative v1 stage-1 sample, a coin-flip "improved > worse" limb, a D-022 rule
that read noise, and a D-023 selection that dropped where phase Y acts — all fixed below or in D-022's second amendment).

Berk (15:0x): **"Meanwhile why do we need that many experiments? Shall we do targeted and surgical tests? and only if we
see hints of improvement we can go wider to test the methodology's limits. It will be a more efficient way than blanket
testing."** (15:1x): **"We shall also always aim for performance runs unless something structural is buggy in our
experimentation methodology."** (14:5x): **"If the box is idle make the most of it."**

## The rules
0. **Performance runs first:** calibration, harness and methodology runs only for a structural bug or to fill idle
   daytime hours. Applied at once: D-021's M4 calibration stopped after 19 of 30 runs. Its late-clock check passed on
   6 of the 10 strata-0–1 instances (median −0.119 s); the remaining 4 strata-0–1 runs finish before v1's first external
   run (the clock every v1 decision reads), and the footprint ratio stays INCOMPLETE until 30 of 30.
1. **Targeted:** run first only where a change can act (named from our traces, with the evidence, before the run) or
   where the claim lives (D-013's weak cases).
2. **A hint, fixed before the runs** — one stated, one-sided threshold on the targeted runs' own data. A hint gates only
   whether a line widens; it never declares success (**futility-only**), so the later decisive test keeps its error
   rate (simulated: P(better shown | no effect) 0.028–0.032 with the gate vs 0.030–0.034 without). Conditional on a
   continuation, the estimates are slightly optimistic (the 64-instance ratio by 1.2–2.5 points in simulation;
   "better shown" ≈ 5 % instead of ≈ 3 % under a null); a stage-2-only figure is printed beside as a selection check.
3. **Wider only after a hint:** generality, the guard against harm, then the limits, if box time allows.
4. **Breadth no decision reads is not run.**
5. **A test of a mechanism on every instance where it can act is a complete test**, not a stage 1 (D-022, D-023 below).

## Development (decided, applied)
- **D-022** (second amendment, f52e3e8): R13 + R14 judged on validity, the mechanism and a floor / band no-harm check
  on the three instances where cuOpt's own 1e-4 stop occurs (assumption stated: gaps act only as a termination test).
  **C1's interim hint** (fixed before any reading): ratio of means of P(120) A9@r13x / A6F@r13x over the fixed 11 weak
  instances ≤ 0.95; **guard** over all 17 (mean bounded d ≥ −0.05; the 4 dropped instances queued, 16 runs). Inputs
  to C1's go / no-go, which stays Berk's.
- **D-023** (to be amended before any backoff2 run): the instances where phase Y acts (by phase-Y length in D-018 stage
  1's traces — chromaticindex1024-7, ca__Scenario400_reserves_1, neos-3656078-kumeu, neos-4763324-toguru) plus M3's four
  (item_placement_9903, neos-3024952-loue, neos-960392, neos-2657525-crna) and sing44 (above its band in stage 1),
  × {boY, bo1s, A6F} × seeds 0–1 = 54 runs (was 84). Dropped and stated: the floor instances (inert) and the 11-instance
  mean over instances not run; the outcome rules read over the 9. (amended 2026-10-13, see D-023: on the phase-Y
  evidence, 11 instances, 66 runs — also in place of the Box-time row's 54.)

## v1, proposed (waits for Berk; D-021 stands until then)
- **(a) Stage 1 of the external set:** the 20-instance panel **plus the 5 H_cu-unseen instances where cuOpt finds no
  solution** (bnatt400, neos-3216931-puriri, peg-solitaire-a3, supportcase19, supportcase22 — they carry 26 % of the
  target's mass; the panel drew none) × {A9, A6F} at seed 2000 under `itx1`, plus A0 on them. ≈ 6.1–6.4 h + ≈ 1.7 h of
  GPU runs (≈ 15 window-hours at 0.55).
- **(b) The hint:** the ratio of means of call-clock `pt_gams` A9 / A6F over the 25 ≤ 0.95 (A9 failures are A9
  outcomes, R29; a block skipped after its requeue makes the hint INCOMPLETE; reported with and without the exposed
  instances, decided with them). Simulated P(hint): 0.21–0.29 under no effect, 0.80–0.87 at −10 %, 0.99–1.00 at −25 %
  (effect concentrated on cuOpt's worst cases).
- **(c) Stage 2 after a hint:** the rest of T1 and T2 (≈ 22–24 h of GPU runs), the decisive target / guard / solutions
  found read exactly as D-021 R19–R21 on all instances, stage 1's included; T3–T5 only if box time remains.
- **(d) After "no hint":** v1's external minimum (guard + solutions found, D-019's pass line) is **not evaluated** —
  unless Berk prefers a guard-only stage 2 (T2's E-sample + W-only ≈ 8.5–9.4 h) that runs whatever the hint says.
- **(e) Conditions:** R21's "no interim looks" becomes one registered futility-only look at stage 1; stages 1 and 2 must
  share the frozen configuration hash, the cuOpt version and the `itx1` harness hash, else stage 1's pairs are not
  pooled; "no hint" ends the line for that configuration, and the 25 join the exposed list for any later configuration.
- **(f) A0** only on H_cu ∪ W_SCIP unseen (81; stage-1 instances first): ≈ 2.9–3.4 h on 4 lanes (was 230). With no
  hint, only the stage-1 instances.
- **(g) Part A** (confirmatory): a futility-only first stage on a fixed subsample, written into prereg-v2.
- **(h)** D-019's literal readings and the 233-wide geo-mean only if the limits stage runs.

## Code owed (before the runs that need it)
`itx-report`: a stage-1 reading (the 25's ratio, with / without exposed, "hint present / no hint / incomplete") and the
stage-2-only figure; `make-jobs itx-v1`: a stage-1 selector (panel + the 5 at seed 2000); `jobs/itx-a0.jsonl`
regenerated as 81 runs (stage-1 instances first); `backoff.py` / `make-jobs backoff2`: D-023's instance sets
parameterised; `tools/d022_report.py`: D-022's second amendment (in progress).

## Box time (from the strata file and GAMS's traces, D-021's 0.55 efficiency)
| | D-021 | D-024 if a hint | D-024 if no hint |
|---|---|---|---|
| D-022 + C1 | 102 runs | 74 runs | — |
| D-023 | 84 runs | 54 runs | — |
| v1 external GPU | 51.3–56.1 h full / 28–30 h core | ≈ 30–32 h (core incl. the 5) | ≈ 7.8–8.1 h |
| A0 | 230 × 600 s | 81 | 25 |

## Open for Berk (v1)
Yes / no per item: (a)–(h) above. The most consequential: (d) — with no hint, is v1's external minimum left unevaluated,
or does a guard-only stage 2 (≈ 9 h) run anyway?

## Addendum — Berk's answers (2026-10-13 15:5x, question panel)
- **Staging: "Yes, stage it"** — items (a), (b), (c), (e) and (h) are decided for prereg-v2: stage 1 = the panel + the 5
  no-solution H_cu-unseen instances, the hint = ratio of means ≤ 0.95 (the one registered futility-only look, R21
  amended), the rest of T1 + T2 only after a hint, one frozen configuration / cuOpt version / harness across stages,
  D-019's literal readings and the 233-wide geo-mean only in the limits stage.
- **No-hint case: "Guard-only stage anyway"** — item (d): the guard stage (T2's E-sample + W-only ≈ 8.5–9.4 h of GPU
  runs) runs whatever the hint says, so v1's minimum (guard + solutions found, D-019) is always evaluated.
- **SCIP-alone runs: "Yes, 81"** — item (f).
- Item (g) (part A's futility stage) was not asked separately; it follows the same pattern and is written into prereg-v2
  with the test-set size question (#32).
