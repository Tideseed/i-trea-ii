# Roadmap (living; updated 2026-10-16)

**v1 target ([[D-019 v1 — deadline and performance target]], Berk 2026-10-10):** checkpoint C1 **Sat 2026-10-17** (moved from Fri 10-23: Berk 2026-10-15, "Assume we are wrapping up phase 1 on Saturday"; D-019 addendum)
(development frozen on cuOpt 26.08, external harness calibrated, interim A9 result, go / no-go); **v1 end of November**
(aim Fri 11-27, hard end Mon 11-30) on cuOpt 26.10 (on 26.08 if 26.10 is not released by Fri 11-06, with 26.10 as an
addendum — to be confirmed with Berk then; D-019); no night windows. Targets: lift SCIP's weak set (min: significant;
target −40 % mean P) and cuOpt's weak set (min: significant; target −25 %) on unseen hard instances, never worse than either solver on average; externally, at
least our own cuOpt-26.10-alone result on the same GB10 (target −25 % on the hard part). The external benchmark is our
own standard ITX-1 ([[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]],
Berk accepted its readings 2026-10-11), run in stages ([[D-024 Targeted first, wider only on a hint — staged testing for development and v1]],
Berk 2026-10-13). Every target is a plan, not a finding.

**Aim refined (Berk, 2026-10-07, [[D-013 Aim — lift each solver's weakest cases]]):** "help both SCIP, cuOpt (and
Highs if possible) in some of the hardest problems where they are the weakest. Our performance metric is improvement
upon the worst performances, of course with a regard to mean to some degree." → primary = improvement on each solver's
weak set (P ≥ 0.5 alone, chosen on separate seeds), guard = the mean (non-inferiority).

**Aim (Berk, 2026-10-06):** make the **hardest** problems easier — a GPU EA coupled with an exact solver beats either
alone ([[D-007 Main claim — the coupling beats either method alone]]). Being **head to head with cuOpt** (NVIDIA's GPU
heuristics + B&B) is the bar; a split result against it is a strong position, not a setback. **Time to first solution
is not a primary objective** — "we don't expect to solve simple problems a few seconds faster at all"; the primal
integral on hard instances is. **Practice first, then scale:** develop on the small set until the algorithm settles,
then go to more seeds and the larger problem set.

**How box time is spent (Berk, 2026-10-13; D-024, AGENTS.md):** "Shall we do targeted and surgical tests? and only if we
see hints of improvement we can go wider to test the methodology's limits." — a hint is one threshold stated before the
runs, and it only gates whether a line widens. "We shall also always aim for performance runs unless something
structural is buggy in our experimentation methodology." — calibration and harness runs only for a structural bug or to
fill idle hours. "If the box is idle make the most of it."

Each stage lists its GitHub issues (`Tideseed/i-trea-ii`). Newest checkpoint: [[2026-10-14-2350]].

## C1 checkpoint — Sat 2026-10-17, end of phase 1: stage 1 (development) wraps up Sat 10-17 (moved from Fri 10-23; table status 2026-10-16 ~09:00)
Berk 2026-10-15 13:4x: **"Assume we are wrapping up phase 1 on Saturday."** ([[D-019 v1 — deadline and performance target]]
§ Addendum 2026-10-15.) Saturday: `configs/frozen-v2.yaml`, the phase-1 results note, checkpoint, docs and issues; the go /
no-go is Berk's. The development numbers below (n = 2–6 seeds on development instances, 120 s) are descriptive and quoted
from the decision records' § Result; the #29 / M4 numbers (600 s, D-020's 30 MIPfeas instances, seed 0) come from the
result note's addendum, M6 from the journal (2026-10-13 22:2x), and the A0 row (600 s, 81 unseen instances, seed 2000) is
validity only. The runs are in the [[Experiment index]] (DEV-21 to DEV-31) and the [[Development log]] § "Phase 1's last
week".

| issue | item | status (2026-10-16) | next |
|---|---|---|---|
| #23 | A9 (SCIP ↔ EA ← cuOpt, D-015) and the EA's GPU back-off | A9 built 10-07. Back-off: [[D-018 A9 — the EA backs off the GPU beside cuOpt (contribution-gated duty cycle)]] stage 1 **NOT ADOPTED** 10-11 (mechanism M1 / M3 / M5 fail); phase Y only, [[D-023 A9 back-off, phase Y only — yield until cuOpt's first incumbent, then full duty]] **NOT ADOPTED** 10-14 (66 runs; failed M2, M5, T, G: chromaticindex1024-7 d +0.030, cuOpt's first incumbent inside A9 63.47 / 63.41 s vs 10.87 / 10.52 s alone) → the back-off line is closed for v1. Harness: [[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]] **ADOPTED** R13 + R14 10-13 (74 runs; no harm on the 3; mechanism met only vacuously) → the C1 freeze carries ext. Why chromaticindex loses: [[D-027 Same-day diagnostic of cuOpt's thread count and SCIP's presence (R3) — reading rules fixed before the runs]] rule 4 **EA PRESENCE** (cuOpt's first incumbent 6.53–9.80 s without the EA vs 62–66 s inside A9), [[D-028 Does D-022's harness (R13 + R14) stop phase Y from restoring cuOpt on chromaticindex — a 4-run diagnostic]] **not R13 + R14** (63.25 / 63.24 s); Berk 10-15: CUDA MPS **"Not for v1"** | phase 2: MPS (v1.1); chromaticindex1024-7 reported as a known loss to cuOpt alone |
| #26 | recombine each new partner incumbent with the whole elite | not built; no decision record | **phase 2** (D-019 addendum 2026-10-15) |
| #27 | `lp_decode` (LP over the continuous variables with the integers fixed) | ADOPTED for A3 ([[D-017 LP decode — re-optimise the continuous variables of selected candidates]], 10-09: mean d +0.219 [+0.041, +0.429], 6 instances × 2 seeds); inside A9 through D-026 (next row) | — |
| #30 (D-026) | R2: lp_decode inside A9 on the EA's core, gated after the LP worker | **ADOPT** 10-15 13:2x ([[D-026 lp_decode inside A9 on the EA's core after the LP worker (R2), development test]] § Result; set `r2-ld`, 28 runs): hint 0.758 (threshold 0.90) — item_placement 0.330 → 0.198 and sing44 0.473 → 0.411, lower on all four seeds of each; every no-harm leg and mechanism holds → `configs/itrea2-dev-rc-3way-ld0.yaml` is A9's and A3x's development default. No new core layout: D-026 runs the decoder on the EA's own core, gated after its LP worker. Separately, D-027 rule 1 closed the cuOpt-thread layout question (a cuOpt thread on X925 6) for v1 | in the frozen-v2 candidate (with yield) |
| (D-025 → D-029) | R1: the EA's CUDA context stops spinning so its LP worker gets core 6 | [[D-025 A9 EA blocking sync (R1) — the EA stops spinning on the GPU so its LP worker gets core 6, development test]]: bsync **NOT ADOPTED** 10-15 08:1x (hint 0.765; generation guard item_placement 0.824, comp21 0.818; comp21 P +0.060), fallback ysync **NOT ADOPTED** 11:2x (hint 0.834; item_placement's generation guard 0.759, one seed at 0.458). Berk 11:3x: "Follow-up, guards on all seeds" → [[D-029 ysync gets one follow-up at seeds 2–5, judged on all seeds 0–5 with the failing seed kept — NOT ADOPTED ends R1 for v1]] **ADOPT** 15:0x (its title states the pre-run framing): generations over seeds 0–5 item_placement 1.020, comp21 1.152; P +0.016 / +0.015; ca at seeds 2–3 0.675 → `host.cuda_sched: yield` joins A9's default | R1 line closed (yield in the frozen-v2 candidate) |
| (D-027, D-028) | diagnostics: cuOpt's threads, SCIP's presence, R13 + R14 on chromaticindex | **done** 10-15 (22 + 4 runs): EA presence; layout closed; SCIP ruled out; rule 2 not read; not R13 + R14 | MPS → v1.1 |
| #29 | external harness + calibration on cuOpt 26.08 | D-020 → D-021 (ITX-1) built and merged; M1 smoke 12 / 12, M2 soak → F accepted, M3 → does-not-fit list (decision set 215), M6 → 3 A0 lanes (13.2 % nodes/s slowdown at 4); **M4 complete 10-15, 30 of 30: OUTSIDE** — R 1.313 [1.091, 1.618] on the call clock, feasible 27 vs GAMS 27, late-clock check passed: the footprint and the machine; v1's targets stay anchored on our own A6F ([[2026-10-11 D-021 measurements M1–M3 and the interim calibration]] § Addendum 2026-10-15) | done for C1 |
| (D-024 (f)) | A0 (SCIP alone) on the 81 instances of H_cu ∪ W_SCIP unseen | **done** 10-14: 80 completed + 1 failed (s100, SCIP killed at 701 s); validity only before prereg-v2's tag (D-021 R33) | stands if the SCIP harness hash is unchanged at the tag |
| (D-019) | interim A9 result on the 17 × 2 seeds vs SCIP alone and cuOpt alone; go / no-go | C1's rules (D-022's second amendment, fixed before any reading, with 7 of 58 itx-r13 runs already done; D-030's fixed before any c1r run): **hint** = ratio of mean P(120) A9 / A6F over the fixed 11 ≤ 0.95 (futility-only), **guard** = mean bounded d over the 17 ≥ −0.05. Readings: (1) old default, 10-13 (D-022 § Result): **hint 0.827, guard +0.063**; (2) the frozen-v2 candidate, 10-15/16 ([[D-030 C1 is re-read once on the frozen-v2 candidate (ld0) with A6F in the same window — 68 runs under D-022's hint and guard and D-025's harm leg]] § Result): **hint 0.809, guard +0.050, harm flag on glass4** (d_i −0.088 vs −0.057; krka at floor); comp21's two A6F runs crashed (cuOpt SIGSEGV) and are kept as outcomes — the letter (excluded) reads INCOMPLETE; (3) glass4 in one window ([[D-032 glass4's C1 harm flag is read in one window — old default vs yield + lp_decode vs lp_decode, 4 seeds]] § Result): **DRIFT** (seed means 0.2176 old / 0.2233 candidate / 0.2177 lp_decode) → yield + lp_decode freezes for phase 1 per D-032's rule (fixed before its runs); D-031's four-seed harm reading may still put which configuration freezes to Berk (D-031 § What goes to Berk). **Four-seed result (D-031) — filled in after the run:** *PLACEHOLDER — set `c1r-ysync-ld0-s23` (68 runs) running since 08:57:43; no number exists* ([[D-031 The phase-1 interim result rests on four seeds — C1 runs again at seeds 2–3 on the frozen-v2 candidate and is read with D-030's seeds 0–1]]; [[Development log]] § "Four-seed result (D-031) — filled in after the run") | Saturday: the phase-1 results note and the numbers to Berk; the go / no-go is his |
| #32 (v1 milestone) | the C1 freeze and prereg-v2 | the candidate `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` (lp_decode + yield; EA 835c7c5a249c, A9 run hash under ext f345de0ab936) freezes for phase 1 after D-032's DRIFT, by D-032's rule (fixed before its runs); D-031's four-seed harm reading may still put which configuration freezes to Berk (D-031 § What goes to Berk); `configs/frozen-v2.yaml` is not written yet; prereg-v2 gates only external A9 runs (D-019 addendum 2026-10-15; D-021 R33) | Saturday: `frozen-v2.yaml` bound to the hash that ran (D-030); prereg-v2 after the 26.10 re-baseline (D-019 addendum); its test-set size is Berk's |
| #36 | checker tolerance vs cuOpt's (found 10-13) | our checker accepted below-optimum incumbents (glass4, mas74); cuOpt rejects the glass4 point; some cuOpt incumbents fail our check (snowy, neos-933966); open, no result revised | one rule for every arm, **before prereg-v2** (not Saturday) |
| #37 | code owed by D-024 | done 10-13 (b84d192): D-023's instance sets, `backoff-report --stage 1y`, `make-jobs itx-a0 --select d024` (the 81); owed: `make-jobs itx-v1`'s stage-1 selector, `itx-report`'s stage-1 reading, the pool's cuOpt-version check; `make-jobs itx-r13` still writes 102 runs against the hand-narrowed 58 (a strict xfail) | before v1 stage 1 |
| #38 | the pool enforces D-021 R26's opening gate (found 10-13 22:34) | open; the stopgap `tools/r26-gate.sh` has opened every window since 10-14 | before v1's windows |
| #33 | 13 completed A3@lps1w control runs moved aside by run-id reuse (infra) | the cross-set case is fixed (14c4832, merged 2a18819); the same id twice inside one set is still refused and open | before any pool reruns a job inside one set |

## Phase 2 — after C1 (carried with their records; D-019 addendum 2026-10-15; D-031 § "No further C1 runs in phase 1")
- #26 recombine each new partner incumbent with the whole elite (not built).
- #35 A9 two-way through cuOpt's SET callback — **v1.1** (Berk 10-14: "v1.1, no nightly runs"); needs cuOpt 26.10 (#31).
- CUDA MPS for sharing the GPU between the EA and cuOpt — **v1.1** (Berk 10-15: "Not for v1").
- chromaticindex1024-7 stays a **known loss** to cuOpt alone (D-027 rule 4, D-028), reported as such.
- More seeds, more instances or another configuration on the C1 set: phase 2, each with a new record (D-031).
- **Due before prereg-v2, not before Saturday:** #36 (one checker rule for every arm), #32's test-set size (Berk's), the
  prereg-v2 tag itself.
- Box lessons carried into the next windows ([[Development log]] § "Operations in phase 1's last week"): leases ≤ 12 h with a
  failure alert on every scheduled chain (the 10-14 07:05 chain was refused and 7 h 36 min were lost); R26's opening gate (#38)
  and its breaker (tripped 10-15 17:30, heat; another session's unpinned CPU jobs ran in our windows and are now pinned to cores
  10–14); the fan governor's rests take about 25 % of window time; stage named files, not `vault/`, while an agent edits it.

## v1 — end of November 2026 (plan as of 2026-10-13)
- **cuOpt 26.10 (#31).** Our feature request cuopt#2073 is fixed by PR #2098 in 26.10 (Berk 10-13: "I just got word it is
  integrated to 26.10"); verified on the 26.10 nightly (presolve stays on with a SET callback; injections accepted;
  invalid offers rejected; SET is polled only when cuOpt's GPU population improves). Not released at 10-13 17:1x (latest
  v26.08.00). On release: separate venv, API check, A6F re-baselined on the 17 instances. A9 two-way through the SET
  callback is a new mechanism (#35), not started; it needs its own record and test and is not part of the C1 freeze.
- **External run = ITX-1 (D-021).** Equal footprint F = A9's layout for every GPU arm (A6F = cuOpt alone on F), no
  whole-box run, call clock deciding with claims on both clocks, T = 600 s with a stop at the proven optimum (R12),
  independent hard / weak sets (`datasets/mipfeas_strata.csv`), decisions on the 215 unseen instances, ε-floored guard +
  solutions found as the minimum, the shift-0.001 geo-mean reported but not decisive. Berk's answers (10-11): "Accept
  all"; fan rests "Keep rests as now"; "Keep 600 s"; calendar: **"We'll cross that bridge when we come to it."** — no
  near-daily schedule or 26.08 insurance runs are planned; the question returns when 26.10's date (or Fri 11-06) is
  known.
- **Staged (D-024, Berk 2026-10-13: "Yes, stage it"; "Guard-only stage anyway"; "Yes, 81"):**
  1. **Stage 1:** the 20-instance panel + the 5 H_cu-unseen instances where cuOpt finds no solution × {A9, A6F} at seed
     2000 under `itx1`, plus A0 on them (≈ 6.1–6.4 h + ≈ 1.7 h of GPU runs).
  2. **Hint** (one registered futility-only look): ratio of means of call-clock `pt_gams` A9 / A6F over the 25 ≤ 0.95.
  3. **After a hint:** the rest of T1 and T2 (≈ 22–24 h of GPU runs; the whole core including stage 1 ≈ 30–32 h); target, guard and
     solutions found read as D-021 R19–R21 on every instance; T3–T5 only if box time remains; D-019's literal readings
     and the 233-wide geo-mean only in that limits stage.
  4. **Always:** the guard stage (T2's E-sample + W-only, ≈ 8.5–9.4 h), so v1's minimum is evaluated whatever the hint.
  5. **A0** (SCIP alone, CPU windows) on the 81 instances of H_cu ∪ W_SCIP unseen, stage-1 instances first — only the 25
     stage-1 instances if there is no hint (D-024 (f)). *(Ran ahead 10-13 / 10-14: 80 completed + 1 failed, validity only
     (D-021 R33).)*
  6. One frozen configuration, cuOpt version and harness hash across stages, else stage 1's pairs are not pooled.
- **Confirmatory test (part A, #32):** prereg-v2 written and tagged before any run; test-set size still open (16 vs 32,
  #11); a futility-only first stage on a fixed subsample in the same pattern, written into prereg-v2.
- **Code owed for v1 (D-024, #37):** `itx-report`'s stage-1 reading; `make-jobs itx-v1`'s stage-1 selector;
  `jobs/itx-a0.jsonl` as 81 runs (done 10-13: `make-jobs itx-a0 --select d024`); the pool's cuOpt-version check; #38 (R26's
  opening gate in the pool); the checker rule settled (#36) before prereg-v2. Then #1 (screening stage 2, pool freeze `pool-v1`) and #12 (paper).

## 1. Development on the 17 instances (9 development + 8 class probe) — until C1; stage 1 = phase 1, **wraps up Sat 10-17** ([[D-010 Develop the algorithm first on a small development set]], [[D-011 Solver streams good infeasible solutions to the EA]])
Baseline: A3@lps1 (adaptive repair strategy + the EA's own HiGHS LP + LP stream + deep repair) — vs SCIP alone 7 of 8
better, mean d +0.43 [+0.23, +0.65]; vs cuOpt alone 6 of 9 better ([[Quantification 2026-10-06 (dev set)]]; descriptive).
- #14 repair clock — adaptive repair depth: active since 2026-10-07 (D-012 addendum 2), part of every development default
- #15 fhnw-binpack4-48 — deep repair + two-scale survival
- #17 radiationm18-12-05 — a non-local operator for the min-max objective
- #18 LP stream follow-ups; #19 owed one-variable runs; #20 pool guard (infra)
- #16 islands — scale demes, then region islands from SCIP's open nodes (Berk's "large chunks"; draft
  [[Islands (draft)]])

- D-013 follow-ups: HiGHS alone (H0) and HiGHS ↔ EA (H3) on the development set (#22).
- D-015: three-way arm A9 (SCIP ↔ EA ← cuOpt) — use cuOpt's feasible solutions (#23); back-off D-018 not adopted,
  D-023 (phase Y only) next; R13 + R14 under the external harness D-022 (running 10-13). With cuOpt 26.10: A9 two-way
  through cuOpt's SET callback (#35). *(As of 10-16: D-022 adopted, D-023 not adopted, D-026 (lp_decode) and D-029 (yield)
  adopted; #35 is v1.1 — the C1 table above.)*
- #25 elite reserve — Rank 1 by cuOpt's clearing rule: built, NOT adopted after seeds 0–3 (D-016; issue closed). Then
  partner-incumbent recombination bursts in A9 (#26, after D-023) and `lp_decode` (D-017: ADOPTED for A3; in A9 pending
  the core layout, #30) ([[2026-10-08 cuOpt population and diversity — what transfers]]). *(As of 10-16: #26 goes to phase 2;
  lp_decode is in A9 since D-026's ADOPT, on the EA's own core — no new layout.)*
- D-012: the repair clock (active) is the development default for repair depth (Stage 2, 2026-10-07).

## 2. Practice → scale
- More seeds on the development set (4 per instance), then the larger problem set: finish the stage-2 screening and
  freeze the pool (#1), frozen references (#2), calibration (#3), runner gates (#4, #5, #13).

## 3. Freeze the configuration
- For v1: A9 frozen at C1 as `configs/frozen-v2.yaml`, registered with prereg-v2 (D-021 R33; D-019's addendum puts
  prereg-v2 after the 26.10 re-baseline). *(As of 10-16: the configuration that freezes for phase 1 is
  `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` — D-026 + D-029 — per D-032's rule, fixed before its runs; D-032 found no
  glass4 harm in one window (DRIFT), it did not confirm the configuration, and D-031's four-seed harm reading may still put
  which configuration freezes to Berk (D-031 § What goes to Berk); the file is written Sat 10-17.)*
- E4 tuning within its cap (#7) → the frozen configuration of the pre-registered plan (D-002).

## 4. Confirmatory
- E2/E3 on unseen instances (#8); test size as proposed in D-010 (16), open in #11; for v1 as prereg-v2 fixes it (#32).

## 5. After tuning
- #21 **Laya** decision model: offline test on recorded decision states (repair clock, strategy, islands) against the
  measured rules and LightGBM; live only at a low duty cycle if it wins offline; the decision maker in E6.
- E5 long runs and open instances (#9); E6 multi-objective — I-TREA's interactive decision maker (#10).

## 6. Paper (#12)
