# D-033 C1 go — phase 1 closes with `configs/frozen-v2.yaml` = yield + lp_decode (2026-10-16, Berk)

**Status:** decided by Berk (question panel, 2026-10-16 13:1x); open to a superseding decision on evidence.

## Decision
Berk's answer to the C1 go / no-go, verbatim choice: **"Go: freeze the candidate (Recommended)"** — "Write configs/frozen-v2.yaml = yield +
lp_decode, close phase 1 on these numbers, and carry the two harm flags, the cuOpt-alone crashes on comp21 and the chromaticindex loss as
stated caveats into phase 2 / v1."

- **`configs/frozen-v2.yaml`** = a byte copy of `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` (EA hash **835c7c5a249c**; A9 run hash under
  ext f345de0ab936): the 10-13 development default + LP decode after the EA's LP worker ([[D-026 lp_decode inside A9 on the EA's core after the LP worker (R2), development test]])
  + the CUDA context at `yield` ([[D-029 ysync gets one follow-up at seeds 2–5, judged on all seeds 0–5 with the failing seed kept — NOT ADOPTED ends R1 for v1]]).
  `tests/test_cuda_sched.py` pins its hash, and `test_frozen_v2_is_a_configuration_c1_was_re_read_on` passes. It stays byte-identical
  until the prereg-v2 tag (D-021 R33: the pool refuses external A9 jobs unless frozen-v2.yaml is unchanged since prereg-v2).
- **`make_jobs.ITX_A9_CONFIG`** keeps naming `-3way.yaml` (the historical baseline the committed job files were written with); new A9 jobs
  use `FROZEN_V2_CONFIG`.
- **Phase 1 closes** on the four-seed interim result ([[D-031 The phase-1 interim result rests on four seeds — C1 runs again at seeds 2–3 on the frozen-v2 candidate and is read with D-030's seeds 0–1]]):
  HINT 0.798, GUARD +0.045 (136 runs VALID, kept reading).

## Caveats carried into phase 2 / v1 (stated, as Berk chose)
- Harm flags beside the result: glass4 (cross-day; [[D-032 glass4's C1 harm flag is read in one window — old default vs yield + lp_decode vs lp_decode, 4 seeds]]
  read it in one window as drift) and comp21-2idx (d_i −0.051 vs band 0.050, inside its replicates' swing).
- cuOpt alone crashed (SIGSEGV, rc −11) on comp21-2idx on 3 of 4 seeds under the ext harness on 10-15 / 10-16 (not on 10-13); the kept
  reading counts it against cuOpt alone; the exclude reading is INCOMPLETE.
- chromaticindex1024-7 stays a known loss to cuOpt alone (the EA's presence delays cuOpt's first incumbent; [[D-027 Same-day diagnostic of cuOpt's thread count and SCIP's presence (R3) — reading rules fixed before the runs]],
  [[D-028 Does D-022's harness (R13 + R14) stop phase Y from restoring cuOpt on chromaticindex — a 4-run diagnostic]]); CUDA MPS is v1.1.
- Development results on 17 tuned-on instances; descriptive; SCIP alone from earlier days.

## Still due before prereg-v2 (not phase 1)
#36 (one checker rule for every arm — it reaches glass4), #32's test-set size (Berk's), the prereg-v2 tag itself; v1 on cuOpt 26.10.
