# D-023 A9 back-off, phase Y only — yield until cuOpt's first incumbent, then full duty (2026-10-12, AgentRA on Berk's word)

**Status:** decided for development by AgentRA (Claude Code, Anthropic Claude Opus 5.5) before any run; open to a
superseding decision on evidence. **Amended 2026-10-13 22:5x, before any run (D-024 rule 5): 11 instances, 66 runs,
harness ext — see the amendment at the end; clarified 23:3x, before any run: G's noise test is "slept 0 s in phase Y", "still beats A6F" binds every guard, the universe is D-018's 14, a T failure inside its band is not adopted, other seeds are outside.** The follow-up named in [[D-018 A9 — the EA backs off the GPU beside cuOpt (contribution-gated duty cycle)]]
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
- **(13:5x, AgentRA, after merging the tooling; still no backoff2 run)** Two further readings, recorded before any run:
  the ladder targets (kumeu, comp21, toguru within band) are an **adoption gate** under D-023 — "every outcome rule" —
  whereas D-018 stage 1 reported them without gating; and D-018's G noise exemption ("never left level 0 and phase Y
  < 1 s on both seeds") is nearly always met under boY, which has no ladder, so a guard whose partner posts within a
  second is exempt from its own band and binds only through the guard-set mean and "still beats A6F". An ADOPT verdict
  for boY is therefore weaker on the guards than D-018's would have been; the report prints every guard's d and band
  either way.

## Amendment before any run (2026-10-13 22:5x) — the instances where phase Y can act (D-024 rule 5); no backoff2 run exists
Berk (15:0x, quoted in [[D-024 Targeted first, wider only on a hint — staged testing for development and v1]]): **"Shall we
do targeted and surgical tests? and only if we see hints of improvement we can go wider to test the methodology's
limits."** D-024 rule 5: a test of a mechanism on every instance where it can act is a complete test, not a stage 1.
D-024's Development bullet drafted the set as nine instances (54 runs): the four with the longest phase Y in D-018 stage
1's traces (ca 30 s, chromaticindex 9.5–12.5 s, kumeu 3.4–3.9 s, toguru 2.0 s — a cutoff between 1.6 and 2.0 s), M3's
four and sing44. Checked here against the per-run phase-Y records before any backoff2 run, the length cut misleads in
both directions, and the set below follows the evidence instead: **two floor instances added, toguru kept for another
reason — 11 instances, 66 runs.** A stated deviation from D-024's draft list (D-024 points here).

**Evidence (D-018 stage 1, set `backoff1`, seeds 0–1; read-only, `backoff.RunView` on the run table rebuilt 22:32).**
boY's phase Y is bo1's rule with bo1's values (yield_level 1 = duty ½, yield_until_s 30, burst_s 0.25, slice_s 0.25;
`configs/itrea2-dev-rc-3way-boY.yaml` differs from `-bo1.yaml` only in its name, comments and the patience fields), so
A9@bo1's phase-Y records are what boY's phase Y does, up to seed noise and the harness change. **Phase Y can act only
where the governor sleeps in it:** at duty ½ the credit falls by ½ per busy second, a sleep falls due once it is below
−burst_s (after 0.5 busy seconds), and it is taken only at a pace point (between generations, between FJ launches) — so
a phase Y shorter than ~0.5 s, or one whose first pace point is its exit, cannot pause. The measure used: seconds paused
in phase Y = `phase_y.exit_t` − `phase_y.exit_busy_s` of the EA summary's `backoff` block (the governor's own record of
the exit). The shadow control's own phase-Y length is not used: its governor updates once per generation, so its exit is
detection-limited (fhnw 7.6 / 7.4 s, toguru 30 s expiry, while the first partner post came at 0.8–1.6 s); the control's
first partner post is shown instead.

| instance | D-018 role | bo1 phase Y s0 / s1 (s), exit | paused in phase Y s0 / s1 (s) | first partner post, bo1 / bo1s (s) | acts? | amended set |
|---|---|---|---|---|---|---|
| chromaticindex1024-7 | target | 9.54 / 12.47, peek / peek | **4.64 / 6.23** | 9.38, 12.35 / 65.29, 65.46 | yes | in |
| ca__Scenario400_reserves_1 | guard | 30.00 / 29.91, expiry / peek | **14.94 / 14.81** | 30.52, 29.75 / 35.33, 36.24 | yes | in |
| neos-3656078-kumeu | ladder | 3.90 / 3.40, tick / peek | **1.85 / 1.57** | 3.80, 3.34 / 4.42, 5.11 | yes | in |
| neos-2657525-crna | guard (M3) | 0.77 / 0.76, peek / peek | **0.25 / 0.25** (one slice each) | 0.61, 0.60 / 0.65, 0.63 | yes | in |
| fhnw-binpack4-48 | floor | 1.03 / 1.62, peek / peek | **0.50 / 0.54** | 0.87, 1.34 / 1.19, 1.61 | yes | **added** |
| neos-3004026-krka | floor | 0.49 / 1.31, peek / peek | 0.00 / **0.53** | 0.32, 1.15 / 0.33, 0.41 | yes (seed 1) | **added** |
| neos-4763324-toguru | ladder | 2.01 / 2.06, peek / peek | 0.00 / 0.00 | 1.45, 1.26 / 1.20, 0.83 | **no** — the first pace point is the exit (first generation ends at 55 / 106 s) | in (M1, ladder) |
| sing44 | guard | 0.53 / 0.53, peek / peek | 0.00 / 0.00 | 0.28, 0.28 / 0.28, 0.30 | no (at the 0.5-s edge) | in (above its band) |
| item_placement_9903 | guard (M3) | 0.09 / 0.07, tick / tick | 0.00 / 0.00 | 0.03, 0.03 / 0.02, 0.03 | no | in (M3) |
| neos-3024952-loue | guard (M3) | 0.30 / 0.29, peek / peek | 0.00 / 0.00 | 0.07, 0.07 / 0.07, 0.08 | no | in (M3) |
| neos-960392 | guard (M3) | 0.23 / 0.24, peek / peek | 0.00 / 0.00 | 0.12, 0.13 / 0.12, 0.13 | no | in (M3) |
| radiationm18-12-05 | floor | 0.495 / 0.511, peek / peek | 0.00 / 0.00 | 0.45, 0.46 / 0.45, 0.46 | no (at the edge) | not run |
| glass4 | guard | 0.29 / 0.02, peek / peek | 0.00 / 0.00 | 0.03, 0.01 / 0.03, 0.04 | no | not run |
| comp21-2idx | ladder | 0.19 / 0.19, peek / peek | 0.00 / 0.00 | 0.10, 0.11 / 0.10, 0.10 | no | not run |

ca seed 1 under bo1 is D-018's failed run (cuOpt's 12-GB RMM limit, stopped at 83 s); its phase-Y exit at 29.9 s
precedes the crash and is valid evidence. Under bo1 the yield moved the first partner post earlier wherever it slept
long enough (chromaticindex 9.4 / 12.3 s vs 65.3 / 65.5 s; ca 30.5 / 29.8 vs 35.3 / 36.2; kumeu 3.8 / 3.3 vs 4.4 / 5.1;
fhnw 0.87 / 1.34 vs 1.19 / 1.61), not on krka seed 1 (1.15 vs 0.41 s).

**The amended set — 11 instances × {A9@boY, A9@bo1s, A6F} × seeds 0–1 = 66 runs** (was 84; D-024 drafted 54), 120 s, set
`backoff2`, harness **ext** on all three arms (D-022 ADOPT, 2026-10-13 22:2x: `itrea make-jobs backoff2 --harness ext`,
`itrea backoff-report --stage 1y --harness ext`): - **the nine (outcome set):** chromaticindex1024-7,
ca__Scenario400_reserves_1, neos-3656078-kumeu, neos-4763324-toguru, item_placement_9903, neos-3024952-loue,
neos-960392, neos-2657525-crna, sing44 — phase Y slept on chromaticindex, ca, kumeu and crna; item_placement, loue,
neos-960392 and crna are M3's four (where bo1's ladder cost the EA's leads, which boY must not); toguru is **not** where
phase Y pauses (0 s on both seeds) — it is in as M1's binding case (exit 0.56 / 0.80 s after cuOpt's post, the reason M1
is 1.0 s here) and a ladder target; sing44 was above its band in stage 1 (d +0.116, band 0.10) and its phase Y (0.53 s)
sits at the edge where a first sleep can fall due; - **two floor instances, read by F and the mechanism rules only (not
d, band or a mean):** fhnw-binpack4-48 (slept 0.50 / 0.54 s) and neos-3004026-krka (0.53 s on seed 1). D-024's draft
dropped "the floor instances (inert)"; that is true of radiationm18-12-05, not of these two. - **Not run, by the
evidence above:** comp21-2idx and glass4 (phase Y ≤ 0.29 s, 0 s paused on both seeds), radiationm18-12-05 (0.495 / 0.511
s, 0 s paused; stage-1 ΔP 0.0003).

**The rules as they read over the set (fixed now, before any backoff2 run; everything not named here is unchanged):** -
**T:** chromaticindex d ≥ +0.30 and P ≤ 0.30. - **G:** item_placement, ca, loue, crna and sing44 each d ≥ −band_i;
neos-960392 ΔP ≥ −max(0.01, spread); the guard-set mean d ≥ −0.05 — the code today averages the bounded d of D-018's 7
guards (item_placement, ca, loue, crna, neos-960392, glass4, sing44); over the set it is the same 6 without glass4
(stage 1, failed kept: +0.052 over 7, +0.051 over 6); every guard in the set where the control beats A6F by ≥ 0.15 still
does. D-018's noise exemption exactly as the 13:5x clarification above describes, over these 6. - **Ladder targets (an
adoption gate, 13:5x):** kumeu and toguru within band (comp21 not run). - **F, on fhnw-binpack4-48 and neos-3004026-krka
— one-sided:** fails only when boY loses more than 0.005, P(boY) − P(bo1s) > +0.005 on seed means. **Changed before any
run** from D-018's two-sided |ΔP| ≤ 0.005, which would fail boY for helping — the error D-022's 16:0x clarification
found for R13 — and phase Y did help cuOpt's first incumbent on fhnw in stage 1 (above). radiationm18 is not run. -
**Mean guard:** the mean d over the nine ≥ 0 (was over the 11 non-floor instances; stage 1, failed kept: +0.130 over the
11, +0.149 over the nine). - **band_i** unchanged: max(0.05, the larger seed-0 vs seed-1 bounded difference of the
control or of A6F). - **Mechanism:** M1 — phase Y ends within 1.0 s of the first posted partner incumbent on every boY
run of the 11 where a partner posted; M2 — chromaticindex, as stated; M3 — the four, as stated (all in the set); M4 —
pooled over every boY run of the 11 ≤ 0.6 (stage 1's bo1 runs, failed kept: 0.546 over the 11, 0.541 over the nine,
0.553 over the 14); M5 — over the 11, as stated. D-018's own M1 / M3 / M5 printed beside, as now. - **Completeness:** 66
of 66 (`--failed both|keep|exclude` as now); a run of any other instance found in set `backoff2` is named as outside the
amended set and read by no rule. - **Adopt boY** iff M1–M5, T, G, the ladder targets, F and the mean guard all hold;
otherwise not adopted, and the report names the rule.

**What this does to the reading's strength, both ways.** Stronger as a test of phase Y: every instance where the
governor slept in phase Y in stage 1 is in the set, including the two floor instances the draft would have dropped, and
the M4 pool barely moves (the dropped phase Ys were ≤ 0.51 s at share 1.0). Weaker as a guard: (1) the mean over the
nine concentrates where phase Y acts — chromaticindex's d (+0.834 in stage 1) weighs 1/9 instead of 1/11, so "mean ≥ 0"
is easier to pass (+0.149 vs +0.130 on stage 1's own runs); (2) glass4 (a binding guard; stage 1 d +0.054, band 0.22)
and comp21 (a ladder target, and the instance where cuOpt's P inside A9 exceeded A6's + 0.02 at every forced duty in
stage 0 — rule (b)'s shortfall) are no longer checked; on both, boY differs from the control only by active mode's
overhead (phase Y ended at ≤ 0.29 s without a sleep, and no ladder follows), which item_placement, loue, neos-960392,
sing44 and toguru still check; (3) "where it acts" rests on two seeds of bo1 under the old harness (RMM 12 GB, cuOpt's
default gaps) — phase Ys at the 0.5-s edge (sing44 0.53 s, radiationm18 0.495 / 0.511 s, krka seed 0 0.493 s) can fall
either side under ext; sing44 and krka are in, radiationm18 is not; (4) F reads two instances at floor-level P
(0.003–0.012) whose stage-1 margins were thin (krka: bo1 worse by 0.0030; fhnw: better by 0.0025), and is now one-sided.
T, M2, M3 and the per-instance guards are unchanged.

**Prediction (not a criterion), updated:** T and the ladder targets as in stage 1; M3 restored (leads ≈ the control's);
the mean d over the nine slightly above stage 1's +0.149 on the same nine (the ladder's losses gone); F holds.

**Code owed before the window (#37):** `backoff-report --stage 1y` over this set and these rules, and `make-jobs
backoff2` writing the 66 runs, with tests — on branch `d023-nine` (worktree `../i-trea-ii-d023`), merged before any
backoff2 run. The evidence above was computed read-only from the run table and the backoff1 run directories (cores
10–14). No `backoff2` job file, run directory or run-table row exists.

## Clarification before any run (2026-10-13 23:3x, AgentRA, after a review of the amendment and its code by Claude Code (Anthropic Claude Opus 5.5) subagents); no backoff2 run exists
Six points the amendment left open or stated wrongly, fixed now — the job file is unchanged (66 runs). Items 1 and 2
replace the amendment's "D-018's noise exemption exactly as the 13:5x clarification above describes"; the rest adds to
it. Everything not named here stands as amended at 22:5x.

1. **"Still beats A6F" binds every guard in the set, noise or not.** The 13:5x clarification described D-018's noise
   exemption as leaving a noise guard bound "through the guard-set mean and 'still beats A6F'", and the amendment
   adopted that description. D-018's design text ("a guard that fails by the letter while the events show the governor
   never left ℓ0 and phase Y lasted < 1 s there is noise: reported, not counted"; design note § 5) does not say whether
   the cross-arm "still beats" leg is a guard failing, and the stage-1 code read it as one: since 2f078bd (2026-10-09;
   unchanged in 4369b6f) it exempted a noise guard's "still beats" failure too. Record and code disagreed and nobody
   had noticed. Fixed as this record states it: **in 1y the noise rule exempts only a guard's own band (and
   neos-960392's ΔP limit); the guard-set mean and "still beats A6F" bind every guard in the set.** D-018 stage 1 is
   unaffected — its noise list was empty and every guard held under both readings — and its code path is unchanged
   (`backoff-report --stage 1 --set backoff1`, `--failed keep` and `exclude`, prints and writes byte for byte the same
   as main's code). For orientation, not a rule: under ext the A9 default beat A6F by ≥ 0.15 on item_placement, ca,
   loue, crna and neos-960392 in C1 (+0.320, +0.329, +0.254, +0.391, +0.919; sing44 +0.093, below 0.15), so "still beats"
   is expected to bind on those five.
2. **The noise test is "slept 0 s in phase Y", not "phase Y < 1 s".** D-018's test reads phase-Y length; this
   amendment chose its instances by seconds slept because length misleads, and the length test would exempt crna,
   which is in the set because phase Y slept there (0.25 s inside a 0.77-s phase Y on both seeds). **In 1y a guard is
   noise only if, on every boY run, the governor never left level 0, phase Y ended, and it slept 0 s in it**
   (`phase_y.exit_t − exit_busy_s` ≤ 1 ms; a sleep is at least one 0.25-s slice, and stage 1's bo1 records are exactly 0
   or ≥ 0.25 s). This is D-023's own test, no longer D-018's. Expected from stage 1's bo1 records, to be read on the
   backoff2 runs themselves: item_placement, loue and neos-960392 slept 0 s (noise-eligible), sing44 0 s at the
   0.5-s edge, ca (14.9 s) and crna (0.25 s) not noise-eligible.
   **Strength item (2) above, corrected.** It said the overhead risk of dropping glass4 and comp21 is still checked
   by "item_placement, loue, neos-960392, sing44 and toguru". By their own band, only where they are not noise: if
   item_placement, loue, neos-960392 and sing44 sleep 0 s in phase Y, as expected, they check it only through the
   6-guard mean and, after item 1, "still beats A6F" (item_placement, loue, neos-960392; sing44 is unlikely to be in the
   "still beats" set). toguru checks it by its own ladder band (no noise rule applies to the ladder). The per-instance
   bands bind on ca and crna and on any guard where boY sleeps. Under the amendment's text the per-instance bands would
   have bound on ca alone (the length test exempting crna as well), and under the code as it stood "still beats" too —
   a weaker G than the amendment's strength paragraph claimed.
3. **The universe is D-018's 14.** "Every instance where the governor slept in phase Y in stage 1 is in the set" is true
   of stage 1's 14, and D-024 rule 5's "complete test" means complete over those 14, not over the 17 development and
   class-probe instances. On the other three, A9 found no incumbent in 120 s under ext (itx-r13: A9@r13x P 1.000 on
   both seeds of neos-2746589-doon, rail01 and cryptanalysiskb128n5obj16), so boY's phase Y would run to its 30-s
   expiry and the governor would sleep (≈ 15 s at duty ½): phase Y can act there too. Not run, stated: rail01 and
   cryptanalysiskb128n5obj16 — no incumbent by any arm or seed under ext (A9@r13x and A6F@r13x P 1.000), no decision
   reads them (D-024 rule 4); neos-2746589-doon — cuOpt alone finds an incumbent late (A6F@r13x 94.1 / 119.4 s, P 0.792
   / 0.996) and A9 does not, so the control's P is at the 1.0 ceiling and boY can only tie or help on d. It would add a
   30-s phase Y to M4 and M5, not evidence to any outcome rule as written, and D-018 named no rule for it. Adding it
   before the window (6 runs, read by the mechanism rules only, like fhnw and krka) would be a further dated amendment.
4. **A T failure inside its band is NOT ADOPTED.** D-018's design adds "inside the band: seeds 3 and 4 on those
   instances, same rules"; D-023 says "otherwise not adopted, and the report names the rule". In 1y the verdict stands:
   T failing with chromaticindex's |d| ≤ its band means NOT ADOPTED. T is the only gate that can fail inside its band,
   since a failing ladder row is below −band. The report prints D-018's "inside the band" class with that note, as
   information. Seeds 3–4 are not pre-registered here and would need their own record.
5. **Runs at other seeds are outside.** A run of the 11 at a seed other than 0–1 in set `backoff2` is named as outside
   the amended set and read by no rule, as a run of another instance is: not by M4, M5, D-018's M5, the relay and LP
   reports, the harness leg or the kept failed runs. The job file has seeds 0–1 only.
6. **F's one-sided limit is compared with a tolerance of 1e-9**, so that boY worse by exactly 0.005 holds whatever the
   float rounding (0.012 vs 0.017 computes to −0.005000000000000001).

**Code:** branch `d023-nine`, 74e2297 (`backoff.py`: `_outcome(noise_test="slept", noise_exempts_beat=False)` from stage 1y only,
`_y_slept`, `Y_NOISE_SLEPT_S` 1 ms; seeds filtered with the instances; F's tolerance; the T-inside-band note), with a
test for each item; non-GPU suite 1041 passed, 1 skipped, 1 xfailed (r13's stale job file, #37). The stage-1 report is unchanged; `jobs/backoff2.jsonl` is unchanged (66 lines, sha256
`17dd3c73…4c94`). A rehearsal of 1y on D-018 stage 1's runs relabelled (bo1 as boY, A6 as A6F; not a backoff2 reading)
reproduces the amendment's figures: M4 0.546 over 22, guard-set mean +0.051 over 6, mean over the nine +0.149, F holds.
Its verdict is INVALID only because those runs predate the ext harness.
