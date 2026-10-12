# D-023 A9 back-off, phase Y only — yield until cuOpt's first incumbent, then full duty (2026-10-12, AgentRA on Berk's word)

**Status:** decided for development by AgentRA (Claude Code, Anthropic Claude Opus 5.5) before any run; open to a
superseding decision on evidence. The follow-up named in [[D-018 A9 — the EA backs off the GPU beside cuOpt (contribution-gated duty cycle)]]
§ "Stage 1 result". Development only (D-010). Runs after [[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]]
is decided, on the A9 configuration and harness that D-022 leaves.

## Why
D-018 stage 1 (2026-10-11): every outcome rule held — chromaticindex d +0.834 (the EA's phase-Y yield gives cuOpt its
first incumbent back: P 0.545 → 0.091, cuOpt alone 0.083) — but the mechanism failed: the patience ladder afterwards
stepped the EA down while it still led (M3: item_placement 196 vs 289 and 143 vs 292 leads; loue s1; crna s1), with M1
(phase-Y exit latency on toguru 0.56 / 0.80 s) and M5 (local-search cap hits on ca) also failing. The part that helps
is phase Y; the part that costs is the ladder.

## Decision — what is tested
**boY** = bo1's values (`yield_level 1` → duty ½ in phase Y, `yield_until_s 30`) in ACTIVE mode with the ladder
disabled by configuration: `patience_search_s = patience_own_s = patience_min_s = patience_max_s = 100000` (longer than
any budget, so phase P never steps down; no code change — the validator needs only positive values with min ≤ max).
File `configs/itrea2-dev-rc-3way-boY.yaml` (written after the M4 pool, since no configs/ commit is made while a pool can
launch runs). Control = **bo1s** (bo1's values in shadow mode = today's behaviour with bookkeeping), as in D-018 stage 1.

## Test — fixed before any run
- 14 instances of D-018 stage 1 × seeds 0–1 × {A9@boY, A9@bo1s, same-day A6F}, 120 s, under D-022's harness outcome;
  set `backoff2`; 84 runs (one window, ~3 h + rests).
- **Mechanism:** M1 — phase Y ends within **1.0 s** of the first partner incumbent on every run (D-018 used 0.5 s; the
  failures were 0.56 / 0.80 s, inside the governor's 0.25-s slice + peek period, and cost nothing measurable in a 120-s
  run — the limit is relaxed here, before any run, and the latency is reported); M2 — chromaticindex's cuOpt first
  incumbent inside boY ≤ 2 × A6F's (both seeds); M3 — the EA's material leads under boY ≥ 0.8 × the control's on
  item_placement, loue, neos-960392 and crna (summed over seeds; with no ladder they should match up to noise); M4 — boY's
  busy share in phase Y ≤ 0.6 (pooled over the arm; clarification below); M5 — the busy / paused accounting within 2 % and local-search cap hits per call within
  +0.10 of the control's.
- **Outcome:** D-018 stage 1's rules unchanged — T: chromaticindex d ≥ +0.30 and P ≤ 0.30; G: item_placement, ca, loue,
  crna, glass4, sing44 each d ≥ −band_i, neos-960392 ΔP ≥ −max(0.01, spread), guard mean d ≥ −0.05, every guard where
  the control beats A6F by ≥ 0.15 still does; ladder targets (kumeu, comp21, toguru) within band; floor |ΔP| ≤ 0.005;
  mean over the 11 non-floor instances ≥ 0. band_i = max(0.05, the larger seed-0 vs seed-1 bounded difference of the
  control or of A6F) (the fixed `_band`).
- **Adopt boY as A9's development default** if the mechanism and every outcome rule hold; otherwise not adopted, and the
  report names the rule. Failed runs are read as D-002 §6 keeps crashes (`--failed keep`), reported both ways.
- **Prediction (not a criterion):** T and the ladder targets as in stage 1; M3 restored (leads ≈ the control's);
  bo-vs-control mean d slightly above stage 1's +0.083 because the ladder's losses are gone.
- **Analysis:** `itrea backoff-report` gains a stage `1y` (written and tested before the window): D-018 stage 1's
  outcome rules unchanged, with the treatment `A9@boY`, set `backoff2`, and the mechanism rules M1–M5 exactly as stated
  here (D-018's own M1 / M3 / M5 are printed beside, for comparison).

## Clarifications before any run (2026-10-12 13:1x, AgentRA, after the review of the build; no backoff2 run exists)
- **M4 is read pooled over the arm:** Σ `phase_y.exit_busy_s` / Σ `phase_y.exit_t` over every boY run (a phase Y that
  never ended: `length_s` and `busy_per_level["Y"]`) ≤ 0.6. Per-run shares are reported, not gating: a phase Y shorter
  than ~2 × `burst_s` ends before the first sleep falls due and has share 1.0 by construction. This reading was in the
  code (`backoff-report --stage 1y`, 4369b6f) before any backoff2 run; on D-018 stage 1's bo1 runs it gives 0.553
  (all 28, the failed one kept; 0.573 over the 27 completed), while 20 of the 28 per-run shares exceed 0.6 by that
  construction — so the reading decides, and is recorded here rather than only in the code.
- **D-022's outcome sets the harness of the two A9 arms only:** A6F runs under ext in either outcome (D-021 R13 — gaps 0
  for every cuOpt process; R6 — A6F is our cuOpt alone; D-022's reject clause covers only A9's rows), so A6F's runs are
  the same reference under both outcomes (`itrea make-jobs backoff2 --harness ext|r13-off`).
- **The harness is checked:** `backoff-report --stage 1y [--harness ext|r13-off]` requires one resolved harness on every
  A9 run of the set (boY and the control, every attempt), ext on every A6F run, and — given `--harness` — D-022's
  outcome on the A9 runs; otherwise the verdict is INVALID. The report names the harness it found.
