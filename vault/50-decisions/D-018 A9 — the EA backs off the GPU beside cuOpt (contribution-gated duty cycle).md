# D-018 A9 — the EA backs off the GPU beside cuOpt by a contribution-gated duty cycle (2026-10-09, AgentRA on Berk's word)

**Status:** decided for development by AgentRA (Claude Code, Anthropic Claude Opus 5.5) under Berk's "let's get to work"
(2026-10-08) and "Go ahead for another round" (2026-10-09); open to a superseding decision on evidence. Issues #23 → #26.
Development only (D-010). Full design, evidence and alternatives: [[2026-10-08 A9 GPU back-off design (workflow)]] (three
designs, two adversarial judges); this record fixes what is built and the tests before anything runs.

## Why
The A9 pilot (2026-10-07, [[D-015 Use cuOpt's feasible solutions — three-way arm A9]]) beat SCIP alone 7/0 but tied cuOpt
alone (3/4, −0.07) because cuOpt runs slower beside a busy EA: its single incumbent on chromaticindex1024-7 (the optimum)
came at 65.35 s inside A9 against 8.56 s alone; sing44 1.98× stretched; the EA's generation-0 repair is 1.5–1.9× slower
beside cuOpt; GPU mean utilisation 91–96 % vs 71–89 % for cuOpt alone. optopt's A4: continuous GPU work beside cuOpt
doubled cuOpt's P; low duty cycles did not. Duty is the lever; the EA must keep full duty where it carries the system
(item_placement 0.362 vs the no-EA portfolio 0.629).

## Decision — what is built (section `backoff`, off by default, out of the hash while off; inert unless `exchange.from_cuopt`)
- **Phase Y** (no incumbent anywhere): the EA runs at duty D_Y; it ends at the first SCIP/cuOpt incumbent (non-consuming
  peek of the info channels inside repair calls), the EA's own first incumbent, or `yield_until_s`.
- **Phase P:** full duty; while no material lead (the repair clock's gap unit, ≥ 1e-3) it steps down 1 → ½ → ¼ (→ ⅛)
  after a patience in busy seconds that depends on its state (no own feasible point / own feasible, no lead yet / led:
  κ / decayed lead rate); any material lead or SCIP-confirmed assist restores full duty.
- **Enforcement:** a host-side token bucket sleeping only where the EA's GPU queue is empty — between FJ launches and
  between generations (after a sync) — in slices of ≤ 0.25 s that keep the stop file and exchange ticks alive; a partner
  improvement wakes it (credit forgiven at most once per 5 s). The repair clock and the FJ cap run on busy time, so the
  pause is not a second variable.
- Modes: `active`; `shadow` (bookkeeping only — the stage-1 control); `forced` (fixed duty — stage-0 calibration only).

## Tests — fixed before any run (details and rationale in the design note, §5)
- **Stage 0, calibration (mechanism, not adoption):** seed 2; chromaticindex1024-7, sing44, comp21-2idx × {A6, bo0
  shadow, forced ½, ¼, ⅛} + ca and glass4 × bo0 = 17 runs. Rules (applied by `itrea backoff-report`, nothing by eye):
  sanity (measured duty within ±0.03, lane-steps per wall second within [0.75, 1.5]·d of bo0's); D_Y = the largest d with
  chromaticindex S_first ≤ 2.0 — **if none, stop: no stage 1** (the mechanism cannot protect cuOpt's first incumbent;
  MPS would be next and needs Berk); floor = the largest d with sing44 stretch ≤ 1.15 and comp21 cuOpt-inside P within
  A6 + max(0.02, A6's spread); yield_until_s, patience_search_s (from ca) and patience_own_s (from sing44 / glass4) by the
  design's formulas.
- **Stage 1, one variable:** A9@bo0 (shadow) vs A9@bo1 (active, stage-0 values) with same-day A6 in every block, 14
  instances × seeds 0–1 (84 runs, two windows). Mechanism M1–M5 (phase-Y exit within 0.5 s of the first partner
  incumbent; chromaticindex cuOpt first incumbent ≤ 2× A6's; protection of the EA's leads on item_placement, loue,
  neos-960392, crna; back-off measured independently on kumeu, comp21, toguru, radiationm; busy/paused accounting).
  Outcome: target chromaticindex d ≥ +0.30 and P ≤ 0.30; binding guards (item_placement, ca, loue, crna, glass4, sing44)
  each d ≥ −band_i, neos-960392 ΔP ≥ −max(0.01, spread), guard mean d ≥ −0.05, every guard where bo0 beats A6 by ≥ 0.15
  still does; ladder targets within band; floor |ΔP| ≤ 0.005; mean over the 11 non-floor instances ≥ 0. Diagnostic
  fallbacks for a ca or glass4/sing44 guard failure as in the design. Predictions stated there (not criteria).
- Before stage 0: CPU suite, the GPU-marked pace tests, and the analysis command committed.

## Order
Built on its own branch while lp_decode (D-017) is fixed and tested; merged after D-017's merge (both change the
generation loop). Stage 0 needs A9's memory (≈ 73 GB → both LLM servers stopped).

## Build note (2026-10-09 13:3x, before any run)
Built on branch `dev-backoff` (ec5477c, 3a9cb74, 22dbe96) by a Claude Code agent; CPU 700 passed; every recorded hash
unchanged; review workflow running. Builder's note recorded here before stage 0: on the A9 pilot the sing44 shared-prefix
stretch as defined in rule (b) is already **1.136 over 55 shared values at full duty** (53 of the 55 in the first 7 s), i.e.
under the 1.15 limit — so rule (b)'s sing44 limb may not bind and the floor will rest mostly on comp21's limb. The rule
stays as written (no change after seeing data); the report will say which limb decided the floor.

## Amendment before stage 0 (2026-10-09 13:5x) — after the adversarial review of the build (5 lenses, 2 refuters)
**Correction to the build note above:** the "1.136" was the same-day A9/A2 ratio, not rule (b)'s A9/A6 statistic,
which is **1.334** on the pilot (n = 55). Both readings show the real problem the review found: 53 of the 55 shared
values lie in cuOpt's first 5 s (t_A6 2.3–5.1 s), where the full-duty EA adds only 1.12–1.16 over A2, while the A6
reference has no SCIP beside cuOpt (A2/A6 = 1.186, day drift included). The median therefore responds to the duty by
only ~5 % across ½ / ¼ / ⅛, against a threshold (1.15) calibrated on the fair-share model 1.33 / 1.14 / 1.07 that holds
for the late window (A9/A6 = 1.98 at ~49 s vs A2/A6 = 1.04). As written the limb passes or fails at every duty
depending on the baseline, not on the duty.
**Rule (b), sing44 limb, redefined (no data from stage 0 seen):** add one same-day **sing44 × A2** run (SCIP ‖ cuOpt,
no EA = zero EA duty, seed 2) to stage 0 (18 runs). Let v* = the best cuOpt objective A2 reaches by 60 s. For each EA arm,
S_late = t_arm(v*) / t_A2(v*), where t is the first time cuOpt (inside the arm) reaches an objective ≤ v*; if the arm
never reaches v* within 120 s, S_late = 120 / t_A2(v*) and the limb fails. The limb passes at d if S_late ≤ 1.15 (the
fair-share model predicts 1.33 / 1.14 / 1.07). The comp21 limb is unchanged. If A2 has no cuOpt incumbent by 60 s, the
sing44 limb is "n/a" and the floor rests on comp21 (stated in the report).
**Other review-driven clarifications (code fixes, not rule changes):** forced mode does not forgive the duty debt on a
wake (the stage-0 dose stays fixed); a stage-0 verdict distinguishes INVALID (sanity failed) from STOP (no duty
protects cuOpt); cuOpt "first incumbent" times censored at the budget count as missing/∞, never as 120 s; shadow mode
logs ladder steps that fall due before a lead resets them, dated at their due time.

## Stage 0, partial result (2026-10-09 16:12–17:10) — 13 of 18 runs; verdict INCOMPLETE (comp21 block missing)
Pool 20261009-161249 (main a1fd414), seed 2, both LLM servers stopped; 13/13 completed, 0 void, 0 compile on the clock;
the comp21-2idx block (5 runs) did not fit before the deadline. `itrea backoff-report --stage 0`:

| arm | duty (measured) | chromaticindex: cuOpt first incumbent (S_first vs A6 8.12 s) | A9 P | sing44 S_late vs A2 (model) | sing44 A9 P |
|---|---|---|---|---|---|
| A6 / A2 | — | 8.12 s (A6) | 0.068 (A6) | 1 (A2, v* reached at 48.4 s) | 0.734 (A6), 0.579 (A2) |
| bo0 (full) | 1.00 | 65.30 s (8.04) | 0.544 | 1.84 (1.00) | 0.453 |
| bof50 | 0.501 | **8.33 s (1.03)** | **0.069** | **1.14** (1.33) | **0.361** |
| bof25 | 0.252 | 9.81 s (1.21) | 0.082 | 1.06 (1.14) | 0.548 |
| bof12 | 0.126 | 62.29 s (7.67) — anomaly, one seed | 0.519 | 1.00 (1.07) | 0.598 |

ca bo0: first partner incumbent 32.6 s, the EA's first material lead 44.3 s → (d) patience_search_s = 40 (clamp);
(e) lags sing44 24.6 s, glass4 79.4 s → patience_own_s = 40 (clamp).
**Sanity rule:** the duty limb passes everywhere (0.501 / 0.252 / 0.126 within ±0.03); the lane-steps limb ([0.75, 1.5]·d
× bo0's steps per wall second) fails on 5 of the 6 forced runs **in both directions** — chromaticindex 0.46d / 0.72d /
0.72d (fewer steps per busy second when paced: launches restart short), sing44 0.92d / 1.72d / 1.84d (more steps per busy
second at low duty: less contention from cuOpt). The limb encoded a premise (EA work ∝ duty) that contention refutes —
the same situation as D-012's criterion (b). By the letter the counted set is empty, so (a) and (b) are "n/a".
**Proposed deviation (stated now, applied when the comp21 block has run, unless Berk objects):** the sanity gate keeps
the duty limb (the dose is what stage 0 sets) and reports the steps ratio; with it, (a) D_Y = ½ (S_first 1.03 ≤ 2.0; ¼
also qualifies at 1.21, ⅛ does not at 7.67), (c) yield_until_s = max(30, ⌈2 × 8.33⌉) = 30, and (b) waits for comp21
(sing44's limb passes at ½, ¼ and ⅛). The ⅛ anomaly (cuOpt late again) is to be replicated before ⅛ is ever used.
**Reading (one seed, not an adoption test):** at duty ½ the three-way system matches cuOpt alone on chromaticindex and
beats every arm on sing44 — the contention cost of A9 is recoverable by duty alone.
