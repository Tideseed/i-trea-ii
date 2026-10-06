# Hard instance pool — protocol (2026-10-05; the list is filled after screening)

**Status: protocol only. No instance has been screened yet; this note holds no pool.** The rules are those of
[[D-002 Pre-registration — hypotheses, arms, analysis]] §1–2 (v3); if this note and D-002 disagree, D-002 wins and this
note is corrected. The frozen pool will live in `datasets/hard_pool.csv` (tag `pool-v1`), the split in the same file,
the references in `datasets/reference.csv`.

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
| MIPLIB 2017 benchmark | the 240 instances in `data/miplib/inst/`, minus the 7 marked `=inf=` in `solu.txt` (`bnatt500`, `cryptanalysiskb128n5obj14`, `fhnw-binpack4-4`, `neos-2075418-temuka`, `neos-3402454-bohle`, `neos-3988577-wolgan`, `neos859080`) and minus `supportcase22` (open; E5 only) | 232 |
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

## Strata and quotas (test / tuning)
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
1. `itrea pool-select` writes `datasets/hard_pool.csv` (instance, family, stratum, side, screening P per seed, time to
   first feasible, flags) — commit, tag `pool-v1`.
2. Reference runs (SCIP 1 h × 2 seeds on CPU lanes; cuOpt 1 h) for families without published values →
   `datasets/reference.csv`, committed before E2.
3. The runner refuses a test-set instance unless the configuration hash equals `frozen-e2`'s and `prereg-v1` is an
   ancestor of HEAD.

## The pool (filled after screening)
| instance | family | stratum | side | P(300) s900 | P(300) s901 | flags |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |
