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
