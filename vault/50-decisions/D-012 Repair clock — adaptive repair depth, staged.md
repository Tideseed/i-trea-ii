# D-012 Repair clock — adaptive repair depth, staged (2026-10-06 evening)

**Status:** decided and built for Stages 0–1 (CPU-verified; the GPU path is untested until a GPU window). Issue #14.

## Why
One-variable test on the development set (Development log, "Repair budget, one variable"): deep repair
(`repair.max_steps 20000, stall_steps 1000`, arm lps1) beats the default depth (lps0) on 6 of 9 instances — mean
bounded d −0.13 for the default — while the default wins item_placement_9903 by a lot (P 0.358 vs 0.569, 4–5× more
generations). The right depth is instance-dependent. A GPU repair call lasts as long as its slowest lane, so mixing
depths inside one call cannot make a generation faster; the depth has to be chosen per call.

## Decision
A **repair clock** chosen by a multi-agent design workflow (three designs, two adversarial judges:
[[2026-10-06 Repair-depth design (workflow)]]): per generation, one lane-step **horizon** h ∈ {2⁷ … 2¹⁶, none} for the
generation's repair calls, enforced between GPU launches (no kernel edit; lanes still active at h end with status
ST_HORIZON = 4, their projected x kept), plus an optional wall-time cap. A horizon-h call is a prefix of the unbounded
call, so the gain each level WOULD have produced is measured exactly from the unbounded call's per-lane steps and the
per-launch (steps, time) profile. The controller compares decayed gain-per-second rates and switches only with
evidence (≥ 5 observed generations, ≥ 2 material gains, a rate > 1.25× the current level's).
- **No change before the EA's own first feasible point** (an injected SCIP solution does not count): deep, uncapped —
  this keeps deep repair's early wins (sing44, ca, loue, fhnw).
- **Gain unit** dg = min((cutoff − key) / max(|cutoff|, |key|, |dual|, 10⁻⁹), 1) — Berthold's primal-gap convention,
  bounded in [0, 1]; gains below 10⁻³ count as zero (glass4's many tiny polish steps must not drive it shallow).
- The one-shot LP batch is always repaired deep (ca's and sing44's early wins come from it).
- Configuration section `repair_clock` (off by default, out of the hash while off): `configs/itrea2-adaptive-lps-rc.yaml`
  = lps1 + the clock in **shadow** mode (hash `26481ba0c675`).

## Stages and pre-stated criteria
- **Stage 0 (CPU)** — built 2026-10-06: horizon in the CPU and GPU feasibility jump, controller module, engine wiring,
  tests. Review: 12 findings (1 blocker — the unbounded gain unit —, 3 major — injected rows ended the pre-feasible
  phase; the cap could never rise; the same gain unit —, 8 minor); 11 fixed, 1 rejected (log volume, measured 0.25 ms
  per generation). Verified: CPU suite 576 passed; every recorded hash unchanged; disabled AND shadow mode
  bit-identical to the previous code (26 arrays, 6 toy × scenario runs).
- **Stage 0b (GPU window):** the gpu-marked tests (`tests/test_fj_gpu.py`: all active lanes share S_STEPS after every
  launch; horizon = CPU reference; multi-chunk projection) and CuPy shadow equivalence.
- **Stage 1 (shadow, ~17 min):** A3@rc0 on item_placement s0/s1 and glass4, sing44, neos-960392, loue, ca s0. Pass
  needs all three: (a) item_placement, both seeds: within 10 s of the first feasible point the controller would leave
  "none" for a level whose rate is > 1.25× that of "none"; (b) glass4, loue, neos-960392: "none" or one level below in
  ≥ 80 % of the post-incumbent generations; (c) generation counts within 10 % of lps1 (no hidden overhead). If (a) or
  (b) fails: fall back to the per-generation bandit (design 1) with the same phase rule and LP pin.
- **Stage 2 (active, only after Stage 1 passes):** A3@rc1 interleaved with lps1, 9 × seeds 0–1, then seeds 2–3 on
  item_placement and ca. Reject unless item_placement beats lps1 on ≥ 3 of 4 seeds with mean d ≥ +0.15; sing44,
  glass4, loue and ca each d ≥ −0.05; radiationm and rail01 within ±0.01; the mechanism agrees (≥ 60 % of
  item_placement's time below "none"; ≥ 60 % of glass4's and loue's at "none" or one below); every incumbent checked,
  GPU health ok, nothing compiled on the clock. fhnw and neos-960392 reported, not judged.
- Still owed before Stage 1: an analysis script for criteria (a)–(c) over the `rclock` events.
