# D-016 Rank 1 as a distinct elite — cuOpt's clearing rule, development test (2026-10-08, AgentRA on Berk's word)

**Status:** decided for development by AgentRA (Claude Code, Anthropic Claude Opus 5.5) under Berk's 2026-10-08 "let's
get to work" on [[2026-10-08 cuOpt population and diversity — what transfers]]; open to a superseding decision on
evidence. Issue #25. Development only (D-010) — never a confirmatory configuration without prereg-v2.

## Why
Berk's mechanism (2026-10-06): "We also keep a percentage of good feasible solutions as well as the best solution. We
are aiming for more diversity in EA." Measured (review note, A3@rc1, 120 s): Rank 1 ends with 1–9 members on the six
feasible development instances other than item_placement (26), because n1 = ⌈y·#feasible⌉ with 0–13 % feasible; the
frontier archive's farthest member differs from the incumbent in 0–5 % of the integer variables on 6 of 7. We keep about
one basin. Berk (2026-10-08): "There is no bad news in this case, we have found more ways to improve. Also if we can
improve upon those heuristics it will benefit to everyone."

## Decision — what is built
`ranks.rank1_mode: "top"` (default, unchanged) | `"clearing"`. Under "clearing", Rank 1 is chosen from the feasible
candidates in (key, age) order by cuOpt's clearing rule (`population_t::add_solution`, NVIDIA/cuopt `dfbbcc5a`, applied
best-first): a candidate joins Rank 1 when it differs from every member already chosen in at least k integer variables;
a candidate within k of a better member is cleared (it competes as Rank 2 if new, dies if old). The best feasible
candidate always joins first. At most `rank1_size` members.
- `ranks.rank1_size: 0` → ⌈y·P⌉ (26 at P = 256 — Berk's y read as "a percentage of the population kept as good
  feasible"; cuOpt keeps 32); a positive integer overrides.
- `ranks.rank1_min_diff: 0` → cuOpt's rule (`get_max_var_threshold`): k = 1 / 2 / 4 / 8 / 10 for n_int < 50 / 80 /
  200 / 1000 / beyond (k = 1: distinct integer parts); a positive integer overrides.
- Under "clearing" the twin rule (`rank1_twins`) and the ratchet (`prev_r1`) do not apply: equal-key members that differ
  in ≥ k integers are distinct basins and both stay; a member leaves only when a better candidate lies within k of it or
  `rank1_size` better distinct candidates exist (a fixed cap, so #feasible falling cannot shrink it).
- The float64 re-check of new Rank 1 entrants is unchanged (the clearing is recomputed after every demotion round).
- Without integer variables "clearing" falls back to "top" (the engine runs "top" and records
  `rank1.mode = "top (clearing requested, no integer variables)"`).
- New fields enter `_LATE_DEFAULTS` at their defaults: every recorded configuration hash stands; "top" is bit-identical
  to the code before this decision.
- Logged in both modes (every `log_every` generations, `gen` events; `final_*` in the summary): `r1_n`, `r1_dmax` /
  `r1_dmed` (share of integer variables in which the Rank 1 members differ from the best member), `r1_gap` (worst
  member's relative objective gap) and `r1_dmax_band` (r1_dmax over the members within the frontier's 5 % band). In
  clearing mode also `r1_cleared` (candidates the distance rule skipped in the last survival) and the summary counter
  `rank1_cleared`.

Not in this step (one variable at a time): reserve-sourced shortfall immigrants; the routing engine's measured, relaxed
radius (variant B, #25); these follow only if this step's mechanism holds.

## Test — fixed before any run (amended 2026-10-08 11:4x, still before any run — see the amendment below)
- Arms: A3@rc1d (control: `configs/itrea2-dev-rc-3way.yaml` with A3's cores, the development default, re-run in this window) and A3@cl1 (the
  same + `ranks.rank1_mode: clearing`), interleaved per block, 9 development instances × seeds 0–1, 120 s, both LLM
  servers stopped. 36 runs.
- **Mechanism (must hold, else the outcome is not read as this mechanism's):** on the instances where both arms hold a
  feasible member at T, the median of r1_n is ≥ 3× the control's, and the median of r1_dmax is ≥ 5× the control's
  frontier farthest member (`frontier` event `max_dist`).
- **Outcome** (seed means, bounded d of cl1 vs rc1, positive = cl1 better):
  - adopt as the development default if mean d over the 9 is ≥ +0.05 and no instance is worse than −band_i;
  - reject if mean d ≤ −0.05 or any instance is worse than −band_i on both seeds;
  - otherwise inconclusive → seeds 2–3 on the instances inside their band, same rules.
- Per-instance bands = the control's largest seed-to-seed bounded difference on 2026-10-07 (A3@rc1; floor 0.02):
  ca 0.10, fhnw 0.02, glass4 0.37, item_placement 0.68, loue 0.13, neos-960392 0.04, radiationm 0.46, rail01 0.02,
  sing44 0.11. They are wide: per-instance verdicts at two seeds can catch only large harm; the mean is the test.
- Also reported in D-013 terms (each solver's weak set, mean as guard) and the instances where the EA never becomes
  feasible (fhnw, rail01) as a check that nothing changed there (the rule is inert without feasible candidates).
- Expected small or none on the few-generation instances (fhnw 31–38, glass4 19–28, ca 20–22 generations).

## Amendment before any run (2026-10-08 11:4x) — after the adversarial code review (4 lenses, 2 refuters per finding)
The review confirmed 12 findings, none a blocker; all fixed in code before any run (warm-up of the new kernels in
both modes, blocked distance fetches instead of one device sync per accepted member, the no-integer fallback, the size
record, the GPU test). Three concern this test and change it before any run:
1. **Like for like.** The original mechanism compared cl1's `r1_dmax` (no quality band) with the control's frontier
   `max_dist` (a 5 % band): one poor distant member would pass it. Replaced by: on the instances where both arms are
   feasible at T, the median of cl1's `final_r1_dmax_band` ≥ 5× the median of the control's `final_r1_dmax_band`
   (both banded, the same function; None counts as 0; if the control's median is 0, cl1's must be > 0). The frontier
   `max_dist` of both arms is reported beside it.
2. **The distance rule must act.** `r1_n` ≥ 3× can be met by the larger cap alone (⌈y·P⌉ instead of ⌈y·#feasible⌉).
   Added: `rank1_cleared` > 0 on every instance read for the mechanism.
3. **Size-only control.** A third arm **A3@tp1** = the control + `ranks.elite_basis: population` (top mode with the same
   cap ⌈y·P⌉ = 26 — the best feasible by key, twins dropped, no distance rule) separates "a larger Rank 1" from "a
   distinct Rank 1". Interleaved in the same blocks (54 runs). The D-016 decision reads cl1 vs rc1d (rules above);
   cl1 vs tp1 is reported with the same statistics and decides whether the distance rule, rather than the size, is
   the active ingredient (if tp1 ≥ cl1 within the bands, the record says "size, not distinctness").
