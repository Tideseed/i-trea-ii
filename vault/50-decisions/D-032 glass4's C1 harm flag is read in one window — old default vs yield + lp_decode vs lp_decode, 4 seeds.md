# D-032 glass4's C1 harm flag is read in one window — old default vs yield + lp_decode vs lp_decode, 4 seeds (2026-10-16, AgentRA on Berk's word)

**Status:** decided before any run (08:2x) by AgentRA (Claude Code, Anthropic Claude Opus 5.5) on Berk's choice; open to a superseding
decision on evidence. A targeted check (D-024 rule 1), not a stage of D-030 or D-031.

## Why
D-030's C1 re-read (set c1r-ysync-ld0, 68 / 68 VALID, read 08:2x; `logs/d030-reading.json`): HINT present 0.809, GUARD holds +0.050, but
D-025 amendment 3's harm leg fired on **glass4** (d_i −0.088 < −0.057; P 0.216 → 0.237, ΔP +0.021; A9@r13x of 10-13 vs the new A9, cross-day)
and on **neos-3004026-krka** (ΔP +0.0010, below D-022's 0.005 floor — a floor marker). Beside it, the unchanged configuration's replicate
of 10-15 (A9@r2ctl) also read d −0.061 on glass4 and cuOpt alone drifted −0.035 there: the flag may be day-to-day drift. D-031 condition 4
sent the choice to Berk. Berk (08:2x, question panel): **"Targeted glass4 check (Recommended)"** — "12 runs ≈ 45 min now, same window: old
default vs yield + lp_decode vs lp_decode alone on glass4 × seeds 0–3, with the rule written first. Drift → freeze yield + lp_decode and
run D-031 today (phase-1 result on 4 seeds). Real → pick the component without it, re-read C1 on that config (~4 h), Saturday morning if
needed." krka's flag is at floor level (ΔP +0.0010 on P ≈ 0.004) and is not re-tested; stated.

## Runs (fixed before any run)
`jobs/d032.jsonl` (set `d032`, sha256 `88289b5400db2ead6554bab428f035b045938894a6eba1ca3dc30edb73b0c777`): glass4 × seeds 0–3 ×
{A9@g4old = `configs/itrea2-dev-rc-3way.yaml` (run hash 28e5a5e9c725, the 10-13 default), A9@g4yl = `-3way-ysync-ld0.yaml` (f345de0ab936,
the candidate), A9@g4ld = `-3way-ld0.yaml` (341267810eed)}, 120 s, harness ext, one block of 3 per seed, positions rotated by seed
(s0 old / yl / ld, s1 yl / ld / old, s2 ld / old / yl, s3 old / yl / ld) = **12 runs**, written from D-030's glass4 A9 job with only
config / variant / set / block / position / seed changed. Run ids new.

## Reading rule (fixed now)
P = P(120) (`pt`), seed means over seeds 0–3 within this set; d(X) = (P̄_old − P̄_X) / max(P̄_old, P̄_X, 10⁻³) (bounded d, + = X better);
band = **0.057** (glass4's band_i fixed in D-030 from the 10-13 arms).
- **d(g4yl) ≥ −0.057 → DRIFT:** the candidate is not worse than the old default on glass4 in one window; **yield + lp_decode freezes** and
  D-031 (`jobs/d031.jsonl`) is queued today.
- **d(g4yl) < −0.057 and d(g4ld) ≥ −0.057 → YIELD on glass4:** lp_decode alone freezes; C1 is re-read on `-3way-ld0.yaml` (68 runs; the
  reader is fixed first for the shared A6F run ids) — today if box time allows, else Saturday morning; D-031 is not run for phase 1.
- **both < −0.057 → BOTH:** the question goes back to Berk with the numbers.
Failed runs: kept as outcomes (P of the run as recorded); a missing cell → INCOMPLETE, and Berk decides. Each seed's P and the mean
generations are printed beside.

## Cost
12 runs ≈ 45 min including gaps, queued in W-1016 now.
