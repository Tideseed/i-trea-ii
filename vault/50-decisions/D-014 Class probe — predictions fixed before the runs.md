# D-014 Class probe — predictions fixed before the runs (2026-10-07 12:1x, before any run)

**Why.** Berk asked whether a clear distinction explains where the coupling gains ([[Where the coupling gains — instance
classes (2026-10-07)]]); on the 9 development instances no pre-run structural rule survived the adversarial checks, and
the explanation that did — **cuOpt's own trajectory** (early close / slow start / stall) and **whether the EA gets a
foothold before SCIP** — was found after the fact. Berk: "go ahead with recommendations" → test it out of sample,
cheaply, with the predictions written here BEFORE any run.

## Fixed in advance
- **Configuration:** A3@lp1 — the coupled default + the EA's own HiGHS LP after t0 (`configs/itrea2-default.yaml` +
  `seeding.lp_source: highs_async`, effective hash `88ae43c371ee`); no other change. Controls: A0 (SCIP alone,
  harness-matched) and A6 (cuOpt alone, as shipped). Seeds 0 and 1, T = 120 s, both LLM servers stopped.
- **Instances (8, public MIPLIB, from families not in the development set; SCIP-weak in the stage-1 screening):**
  objective with ≤ 4 non-zeros — neos-3004026-krka, neos-3656078-kumeu, chromaticindex1024-7,
  cryptanalysiskb128n5obj16; dense objective — comp21-2idx, neos-4763324-toguru; many two-variable equality rows —
  neos-2657525-crna, neos-2746589-doon.
- **Offline one-thread HiGHS LP time** (measured 2026-10-07 12:1x, `threads=1`, before the runs): krka 0.2 s, kumeu
  0.3 s, chromaticindex1024-7 **124.8 s**, cryptanalysiskb128n5obj16 13.5 s, comp21-2idx 0.4 s, toguru 6.9 s, crna 0.0 s,
  doon 3.4 s.

## Predictions (d = bounded effect of A3@lp1 vs the control, seed means; positive = coupled better)
- **(a) vs cuOpt alone:** coupled LOSES (d < −0.05) on **krka, kumeu, chromaticindex1024-7** (cuOpt closes early
  there in optopt's runs); coupled WINS (d > +0.05) on **comp21-2idx, toguru** (cuOpt stalls). No prediction for
  cryptanalysiskb128n5obj16, crna, doon.
- **(b) vs SCIP alone:** NO gain (d ≤ +0.05 on at least one seed) on **chromaticindex1024-7**, whose LP takes longer
  than 30 s (the EA's own LP cannot arrive early; 124.8 s > the 60 s `seeding.lp_time_limit`).
- **Supported if:** (a) has the predicted sign on ≥ 4 of its 5 instances AND (b) holds. **Refuted if:** (a) has the
  predicted sign on ≤ 2 of 5, or (b) fails. Otherwise inconclusive. Eight instances can refute the distinction but only
  weakly confirm it; the result is descriptive.
- Also reported (no prediction): each solver's weak set among the 8 (D-013: P ≥ 0.5 alone) and the coupled
  improvement on it, with the mean as the guard.

## Consequences
- The 8 instances join the development set in being **excluded from the confirmatory test selection**
  (`datasets/classprobe_set.csv`); the pool freeze (#1) must read both exclusion lists — it does not yet (to fix).
- Runs: after the repair clock's Stage 2, in the same extended window (`jobs/classprobe1.jsonl`).

## Result (2026-10-07 15:13) — INCONCLUSIVE
(a) 3 of 5 (losses on krka, kumeu, chromaticindex1024-7: hits; wins on comp21-2idx and toguru: misses, d −0.24 each);
(b) holds. Out of sample the coupling helped SCIP on 2 of its 8 weak instances and lost to cuOpt on 5 of 6 — see
Development log, "Class probe — D-014 result".
