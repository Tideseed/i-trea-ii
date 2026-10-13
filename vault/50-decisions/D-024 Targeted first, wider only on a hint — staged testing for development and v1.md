# D-024 Targeted first, wider only on a hint — staged testing for development and v1 (2026-10-13, Berk; drafted by AgentRA)

**Status:** decided by Berk's instruction (2026-10-13 15:0x); the stage rules below are AgentRA's reading, open to a
superseding decision on evidence. Narrows [[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]]
(run order, tiers, A0's scope), [[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]]
and [[D-023 A9 back-off, phase Y only — yield until cuOpt's first incumbent, then full duty]], before any run of theirs.
Berk: **"Meanwhile why do we need that many experiments? Shall we do targeted and surgical tests? and only if we see
hints of improvement we can go wider to test the methodology's limits. It will be a more efficient way than blanket
testing."**

## The rule
0. **Performance runs first** (Berk, 15:1x: "We shall also always aim for performance runs unless something
   structural is buggy in our experimentation methodology."): calibration, harness and methodology runs only for a
   structural bug or to fill idle daytime hours. Applied at once: D-021's M4 calibration stopped after 19 of 30 runs
   (its clock check had passed; the remaining 11 fill idle time later) so the performance runs start at once.
1. **Stage 1, surgical.** Run first only where the change can act or where the claim lives: for a mechanism, the
   instances where our traces show it can act (named, with the evidence, before the run); for an outcome claim, the
   weak cases D-013 names. Two seeds in development; a fixed, already-drawn sample in v1.
2. **A hint, fixed before stage 1:** a stated, one-sided threshold on stage 1's own data. Without it the change stops
   there (reported as "no hint at stage 1", with its numbers) — **futility-only**: stage 1 can stop a line of tests,
   never declare success, so the stage-2 test keeps its error rate (it is the pre-stated test on all its instances,
   stage 1's included; the only cost is a small loss of power, stated).
3. **Stage 2, wider:** only after a hint — the rest of the decisive set (generality, the guard against harm), then the
   limits (other instance classes, longer T) if box time allows.
4. Breadth that no decision reads is not run (e.g. A0 on instances no comparison uses; the 233-wide geo-mean).

## Applied
- **D-022** (amended before any run, 15a5c81): R13 on the 3 instances where cuOpt's 1e-4 "optimal" stop occurs
  (18 runs, was 102); C1's interim hint A9@r13x vs A6F@r13x on cuOpt's 11 weak development instances (44 runs).
  `jobs/itx-r13.jsonl` = 58 runs.
- **D-023** (to be amended before any run, same way): boY vs bo1s on the instances where phase Y or the ladder showed
  an effect in D-018 stage 1 — chromaticindex1024-7 (T, M2: + A6F), item_placement_9903, neos-3024952-loue,
  neos-2657525-crna (M3, G), sing44, ca__Scenario400_reserves_1, glass4 (G) — × seeds 0–1 ≈ 30 runs (was 84). The
  floor instances (krka, fhnw, radiationm: inert) and the ladder targets (kumeu, comp21, toguru: they tested the
  ladder's step-down, which boY removes) are not run; the outcome rules read over the instances run.
- **v1 external (D-021):**
  - **Stage 1 = the repeatability panel** (20 of H_cu-unseen, already drawn: 10 with GAMS cuOpt P ≥ 0.5 or none, 10
    with 0.1 ≤ P < 0.5) × {A9, A6F} at seed 2000 under `itx1`, plus A0 on the panel (the computed race). ≈ 5.6–5.9 h
    of GPU runs ≈ 1 window; A0 ≈ 0.9 h on 4 CPU lanes.
  - **Hint (pre-stated):** the mean `pt_gams` ratio A9 / A6F on the panel ≤ 0.95, **or** more panel instances improved
    than worse (d_ε ±0.10). Neither → v1's external line stops: "no hint at stage 1" with the panel's numbers.
  - **Stage 2:** the other 44 of H_cu-unseen and T2 (W-only 17 + E-sample 40) × {A9, A6F}; the decisive target, the
    guard and solutions found are then read exactly as D-021 R19–R21 on all 64 / the decision set (stratified
    estimates, R31), stage 1's pairs included. ≈ 21–23 h of GPU runs ≈ 4–5 windows. T3 (the rest of E), T4 (seed 2001)
    and T5 (live A2F) only if box time remains — the limits stage.
  - **A0** only on H_cu ∪ W_SCIP unseen (81 instances; the race and the A9-vs-A0 secondary read nothing else), panel
    first: ≈ 2.1–2.9 h on 4 lanes (was 230 instances, 6–8.3 h).
- **Part A (confirmatory, D-019):** staged the same way in prereg-v2 (a futility-only first stage on a fixed subsample
  of the test set).

## What it saves (box time, planning numbers of D-021)
| | before | after, if the hint holds | after, if no hint |
|---|---|---|---|
| D-022 | 102 runs (3.4 h) | 58 runs (≈ 2 h) | — |
| D-023 | 84 runs (≈ 3 h) | ≈ 30 runs (≈ 1.1 h) | — |
| v1 external GPU | 51–57 h of runs (full) / 27–29 (core) | ≈ 27–29 h (core only; T3–T5 only if time) | ≈ 6 h |
| A0 | 230 × 600 s | 81 × 600 s | 20 × 600 s |

## Risks
- A real but small effect missed at stage 1 (the hint threshold is loose on purpose: 5 % or a majority of instances).
- Selection from traces (where a mechanism "can act") is itself a choice; each record names its evidence before the
  run.
- The 233-wide published-comparable geo-mean is no longer produced unless the limits stage runs (reported as such).
