# D-017 LP decode — re-optimise the continuous variables of selected candidates (2026-10-08, AgentRA on Berk's word)

**Status:** decided for development by AgentRA (Claude Code, Anthropic Claude Opus 5.5) under Berk's 2026-10-08 "let's
get to work" on [[2026-10-08 cuOpt population and diversity — what transfers]] (item 6); open to a superseding decision on
evidence. Issue #27. Development only (D-010). Design: [[2026-10-08 lp_decode design (workflow)]]. Built on branch
`dev-next` (f99a7de), merged into main only after its adversarial review and after the D-016 pool ends.

## Why
cuOpt's MIP heuristics run, after every recombination and local search, an LP with all integers fixed and keep both
versions (`run_lp_with_vars_fixed`, 1 s / 2 s). Our EA decodes singleton continuous columns in closed form and moves every
other continuous variable by feasibility-jump jump values; `repair.continuous: lp_decode` was configured but raised
NotImplementedError. Non-singleton continuous columns exist on 7 of our 17 instances (ca 100,704; sing44 15,995;
radiationm 14,688; kumeu 658; fhnw 105; item_placement 33; glass4 19). Probe on stored A3@rc1 seed-0 vectors (CPU, 1
thread, 2026-10-08 12:2x): re-optimising the continuous part lowers the objective by a median **30 %** on sing44 (max
46 %), 0.9 % on ca, 6.0 % on item_placement, ≈ 0 on glass4 (tolerance edge) and 0 on radiationm (no continuous cost).
These are motivating numbers on stored vectors, not results.

## Decision — what is built
`repair.continuous: "lp_decode"` = "auto" (closed-form singleton decoding) plus an LP decode: a persistent spawned HiGHS
worker (1 thread, nice 10) started at t0 beside the root-LP worker; every generation, after survival, at most one
request of up to N = `lp_decode_per_gen` (4) rows to solve — feasible candidates to polish (only if a continuous column
carries cost) and super-optimal infeasible candidates to complete (`lp_decode_infeasible`, D-011's (V, key) order),
split 2 + 2 when both exist; the LP fixes the integer part and re-optimises the active continuous columns; the time limit
is rebased on HiGHS's cumulative clock (2 s feasible, 1 s infeasible, 10 s for the first, priming solve); a basis is
restored after a non-optimal solve. Decoded copies enter the next candidate set as ordinary rows (origin
`ORIGIN_LP_DECODE`, the source's strategy and ancestry, `lpd_src` in the genealogy) and pass the same float64 re-check
and survival. Inert (no worker, no counters) when no active column exists. Refused by the runner in the single-core EA
arms (A9, A3x). New fields `lp_decode_cold_limit`, `lp_decode_infeasible` in `_LATE_DEFAULTS`; the default hash
9ac4a93bf93b and the development default 1d12ad490dd0 are unchanged; treatment = control + `repair.continuous:
lp_decode` (on the rc1 control: f030e23909ad).

## Test — fixed before any run
- **Control** = the development default at the window, re-run in the same window: A3@rc1 (`itrea2-dev-rc-3way`), or the
  D-016 arm if D-016 is adopted. **Treatment** "ld1" = control + `repair.continuous: lp_decode` (one field).
- Instances where it can act: ca, sing44, radiationm, fhnw, item_placement, glass4; seeds 0–1, 120 s, A3 cores, both LLM
  servers stopped, interleaved per block; 24 runs. loue, neos-960392, rail01 are covered by the CPU test that the mode is
  inert there.
- **Mechanism (must hold, else the outcome is not read as lp_decode's):**
  - M0 validity: per block equal P and gene dtype between the arms; `compiled_on_clock` false; the worker never
    "failed"; ready ≤ 10 s after t0.
  - M1 the GPU never waits: `lpd_host_s` ≤ 1 % of the EA's wall time per run, `lpd_host_s_max` ≤ 50 ms.
  - M2 the worker is sane: after priming, time-limit hits ≤ 10 % of solved LPs per instance; ca's priming solve optimal.
  - M3 polishing acts: sing44 median relative gain of admitted feasible-source copies ≥ 1e-2 and ≥ 1 decoded EA
    incumbent in each seed; ca median gain ≥ 1e-3 and ≥ 1 decoded incumbent in at least one seed.
  - M4 completion is reported (offered, passed the screen, LP feasible, entered, incumbents); if no infeasible-source LP
    is feasible in any of the 12 treatment runs, completion is recorded as dead weight here and
    `lp_decode_infeasible: false` becomes its own variant.
- **Outcome** (seed means, bounded d of ld1 vs control, positive = ld1 better), D-016's rules: adopt if the mean d over
  the 6 is ≥ +0.05 and no instance is worse than −band_i; reject if the mean d ≤ −0.05 or any instance is worse than
  −band_i on both seeds; otherwise inconclusive → seeds 2–3 on the in-band instances where the mechanism acted. Bands:
  ca 0.10, fhnw 0.02, glass4 0.37, item_placement 0.68, radiationm 0.46, sing44 0.11 (D-016); if the control is the
  D-016 arm, band_i = max(D-016's band, the new control's own two-seed spread, 0.02).
- Also reported: the 9-instance mean with d = 0 for the three inert instances, and the D-013 weak-set view.
- **Predictions (stated, not criteria):** sing44 positive; ca small positive; glass4 and item_placement ≈ 0; radiationm
  rests on completion; fhnw inert unless completion fires.
- **Known risks recorded before the run:** the cutoff row (decoded keys lower the population best, so FJ then aims at
  LP-level keys and feasible children may become rarer — sing44 above all); warm starts are only fast for nearby integer
  parts; the decoder shares the EA's 3 A725 cores in A3; D-016 and D-017 interact in Rank 1 (under clearing a source and
  its copy share a niche).
