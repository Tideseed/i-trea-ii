# Hard instance pool — protocol (2026-10-05; the list is filled after screening)

**Status (2026-10-10): screening stage 1 complete, stage 2 partial; the pool is not frozen and this note holds no
pool.** Stage 1 ran 2026-10-06 02:52–06:03 (all 324 candidates, seed 900); stage 2 ran 35 of its 234 jobs before the
window closed and is deferred (D-010 §5) — it is needed only for the freeze, now in the v1 milestone (#1; results below,
"Screening so far"). The rules are those of [[D-002 Pre-registration — hypotheses, arms, analysis]] §1–2 (v3.1); if
this note and D-002 disagree, D-002 wins and this note is corrected. The confirmatory test is to be re-specified in
prereg-v2 before any run (D-013, D-019, #32); the selection rules below stand until then. The frozen pool will live in
`datasets/hard_pool.csv` (tag `pool-v1`), the split in the same file, the references in `datasets/reference.csv`.

## Why a fresh screening pass
The brief asks for the hardest problems. "Hard" is measured on **SCIP alone** — the partner shared by A0–A3 — in a
fresh pass, never on cuOpt and never on optopt's grid. optopt's `datasets/hardness_from_optopt.csv` (386 rows) mixes
cuOpt into its "best strategy" columns, so selecting on it would remove exactly the instances where cuOpt is strong
and decide the SCIP ‖ cuOpt comparison through the selection; it is also one seed, unpinned lanes and 30–60 s budgets
outside MIPLIB (design critique, methodology lens). It is kept as background only. Reusing a selection seed in a
confirmatory arm would replay the selected (bad) SCIP path, so screening, tuning and confirmatory seeds are disjoint.

## Candidates (all on disk, no download — [[Datasets]])
| family | candidates | count |
|---|---|---|
| MIPLIB 2017 benchmark | the 240 instances in `data/miplib/inst/`, minus the 7 marked `=inf=` in `solu.txt` (`bnatt500`, `cryptanalysiskb128n5obj14`, `fhnw-binpack4-4`, `neos-2075418-temuka`, `neos-3402454-bohle`, `neos-3988577-wolgan`, `neos859080`) and minus `supportcase22` (open; E5 only). These 232 plus `supportcase22` are the 233 MIPfeas instances of D-019 ([[Datasets]]) | 232 |
| PGLib-UC | `ca` (20) and `rts_gmlc` (12) cases in `data/pglib_uc/mps/`; `ferc` (24) is exploratory only | 32 |
| ML4CO item placement | 60 instances: the 25 validation instances optopt used (`item_placement_9900`–`9924`, the first 25 of the 100 in `valid/`) and train `item_placement_0`–`34` | 60 |

The ML4CO row is my reading of D-002's "the 25 official-validation instances": `valid/` holds 100 instances; optopt
used the first 25. `pool-select` must state the exact list in the pool file.

## Screening pass (CPU only)
- SCIP 10 default, the exchange heuristic plugin installed and inert (as in every SCIP arm), 1 thread, 300 s, one run
  per Cortex-X925 core (5–9, 15–19), `MemoryMax` 9 GiB, both local LLM servers stopped, inside a box-window lease.
- **Stage 1:** every candidate with seed **900**.
- **Stage 2:** seed **901**, only for instances with stage-1 P(300) ≥ 0.05 or no solution. An instance that stage 1
  solves to optimality with P(300) < 0.05 cannot reach any stratum.
- Every incumbent is stored and checked (as in confirmatory runs). The reference for screening P is the `.solu` value
  for MIPLIB; for PGLib-UC and ML4CO a provisional best-of-screening value, replaced by the frozen reference before E2.
- Budget: 324 stage-1 runs × 300 s over 10 lanes ≈ 2.7 h (plus start-up), then stage 2 on the survivors.
- **As run (2026-10-06):** 9 lanes at 02:52, then **8 lanes** with a CPU-only start gate of **88 °C** from 02:59
  ([[D-008 Thermal start gate by arm type]]; the X925 cores held 3.9 GHz at 88–91 °C, so the measurement is unaffected);
  pause at ≥ 92 °C voids the in-flight blocks, which are requeued. Since 10-06 the runner also honours the fan
  governor's rest breaks ([[D-009 Honour the fan governor's rest breaks and the night rest]]). Stage 2's remainder is
  to run on **6 lanes with an 86 °C** CPU start gate (D-008 addendum 2) — in daytime windows only (D-019: no nights).

## Screening so far (2026-10-10)
| stage | ran | outcome |
|---|---|---|
| 1 (seed 900, `jobs/screen1.jsonl`, config `87f5f240d524`) | 2026-10-06 02:52–06:03, 324 / 324 candidates | 323 completed; `satellites2-40` killed at budget + grace (SIGKILL at 400 s; first incumbent 38.8 s, incumbents kept — #13). MIPLIB 232 → 34 without a feasible solution, 86 feasible with P(300) ≥ 0.10 (120 with P ≥ 0.10 counting the 34), 142 qualify for stage 2 (P ≥ 0.05 or no solution); PGLib-UC 32 → 19 without a solution; ML4CO 60 → all feasible (P against their own provisional best). Seven thermal pauses at ≥ 92 °C voided 57 runs: 9 at 02:58 in the 9-lane first start, 48 in six pauses of the 8-lane pool (03:32, 04:52, 05:08, 05:14, 05:27, 05:53); the interrupt for the restart on 8 lanes voided 9 more. All requeued and rerun in the pass (`logs/pool-2026-10-06.log`; journal 2026-10-05 and D-008 addendum 2 still say 2 pauses / 16 runs, the count at about 05:00) |
| 2 (seed 901, `jobs/screen2.jsonl`: 142 MIPLIB + all 92 PGLib-UC / ML4CO = 234 jobs) | 2026-10-06 06:04–06:57, stopped at the deadline | 35 ran: 34 completed, `satellites2-60-fs` killed at budget + grace (#13); 8 thermal pauses voided 64 runs (requeued). **~199 owed** (#1) |

The per-instance values are in `datasets/runs.parquet` (sets `screen1`, `screen2`); nothing below is final until
stage 2 is complete and `itrea pool-freeze` has run.

## Exclusions from the confirmatory selection
Two instance lists were chosen using the stage-1 screening (the development set by looking at its table, the class
probe by structural class among SCIP-weak instances) and are **permanently excluded** from the test set:
- the development set, 9 instances (`datasets/dev_set.csv`, [[D-010 Develop the algorithm first on a small development set]]);
- the class probe, 8 instances (`datasets/classprobe_set.csv`, [[D-014 Class probe — predictions fixed before the runs]]).

**`src/itrea/benchmarks/hardset.py` reads neither list yet** (checked 2026-10-10); the freeze (#1) must read both before
`pool-v1` is written. 15 of the 17 are MIPLIB candidates (7 development + 8 class probe), one is a PGLib-UC `ca` case
and one an ML4CO validation case (`item_placement_9903`).

## Strata and quotas (test / tuning)
**Open points, stated 2026-10-10 (not decided here):**
- **Test-set size.** D-002 fixes the quotas below (32 test / 16 tuning), and `hardset.py` still codes them; D-010 §4
  proposed 16 test instances (#11, open); D-019 and #32 plan "about 32 unseen hard instances". prereg-v2 states the
  number.
- **Tuning side.** D-010 §2 replaced the tuning side by the development set (E1 and E4 run on it); `hardset.py` still
  draws both sides.
- **Weak sets.** D-013 / D-019 judge the confirmatory test on each solver's weak set (P ≥ 0.5 alone) measured on
  screening seeds, never the evaluation seeds. The screening ran SCIP alone, so W_SCIP comes from seeds 900 / 901; how
  W_cuOpt is measured on non-evaluation seeds is for prereg-v2 (#32) to state.

| stratum | rule (screening, both seeds) | test | tuning |
|---|---|---|---|
| MIPLIB optimisation-hard | SCIP feasible on both seeds, mean P(300) ≥ 0.10 | 16 | 8 |
| MIPLIB feasibility-hard | SCIP finds no feasible solution on ≥ 1 seed | 4 | 2 |
| PGLib-UC | P(300) ≥ 0.10 or no solution; `rts_gmlc` held out of tuning (unseen family) | 6 | 3 (`ca` only) |
| ML4CO item placement | top half by P(300) | 6 | 3 |

**Selection within a stratum:** sort by mean screening P(300) (the feasibility stratum by time to first feasible) and
take every k-th until the quota is filled — deterministic and spread over difficulty.

**Split by group:** MIPLIB by family prefix (`neos-*`, `supportcase*`, `comp*`, …; one family never on both sides),
PGLib by base system, ML4CO by index. No generator or family supplies more than 20 % of the test set.

**Fit rule:** if an instance's population cannot fit (P × n × bytes above the GPU budget at P = 32), it is replaced by
the next instance of its stratum and the replacement is listed.

**Flags in the pool file:** P-degenerate (reference 0 or a sign crossing seen in screening — analysed on time to target
only), feasibility stratum (time to first feasible endpoint), the "no exchange window" candidates (first SOLVING-stage
SCIP callback after T/2 in screening).

## Freezing
0. (added 2026-10-10) Complete stage 2; the selection reads both exclusion lists above (#1).
1. `itrea pool-select` writes `datasets/hard_pool.csv` (instance, family, stratum, side, screening P per seed, time to
   first feasible, flags) — commit, tag `pool-v1`. (The command as built is `itrea pool-freeze`.)
2. Reference runs (SCIP 1 h × 2 seeds on CPU lanes; cuOpt 1 h) for families without published values →
   `datasets/reference.csv`, committed before E2.
3. The runner refuses a test-set instance unless the configuration hash equals `frozen-e2`'s and `prereg-v1` is an
   ancestor of HEAD.

## The pool (filled after screening; not frozen on 2026-10-10)
| instance | family | stratum | side | P(300) s900 | P(300) s901 | flags |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |
