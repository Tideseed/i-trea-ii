# Datasets

Public instance sets only, all already on disk. **No downloads** — instances, wheels, models or anything else — without
Berk's explicit ask (host rule; the Wi-Fi link makes each download a cost to him). Customer data (customer) never
enters this project.

## Where the instances are
`data/` in the repository is **not in git** (`.gitignore`). On the author's machine it holds three symbolic links into
the predecessor's checkout, so the two studies read byte-identical files:

| link | target | contents |
|---|---|---|
| `data/miplib` | `~/git_repositories/optopt/data/miplib` | `inst/` (240 `.mps.gz`), `solu.txt` (`miplib2017-v33.solu`), `benchmark.zip`, `tag_benchmark.html`, `collection.csv`, `selected.json` (optopt's 80) |
| `data/pglib_uc` | `~/git_repositories/optopt/data/pglib_uc` | `mps/` (56 `.mps.gz`), `repo/` (the PGLib-UC JSON cases, `uc_model.py`, MODEL.pdf), `build.log` |
| `data/ml4co` | `~/git_repositories/optopt/data/ml4co` | `instances/1_item_placement/{train,valid}`, `instances/2_load_balancing/{train,valid}` (`.mps.gz` + `.json`) |

`$ITREA_DATA` overrides the location (`itrea.paths.DATA`). Job files and traces store instance paths relative to the
repository root (`data/miplib/inst/air05.mps.gz`), so a clone with its own `data/` runs unchanged. To recreate the
links: `ln -s ~/git_repositories/optopt/data/{miplib,ml4co,pglib_uc} data/`. On a machine without optopt, the sources
below are the public origins (optopt's `optopt fetch` reproduces them).

## Families
| family | public source | on disk | used here |
|---|---|---|---|
| **MIPLIB 2017 benchmark** (Gleixner et al. 2021) | miplib.zib.de, `benchmark.zip`, `miplib2017-v33.solu`, `tag_benchmark.html` | 240 instances, 609 MB | all except the 7 infeasible ones and `supportcase22` are screening candidates ([[Hard instance pool]]); `supportcase22` (open, 2.2 M non-zeros) only in E5; the 233 feasible ones are the MIPfeas set (D-019) |
| **PGLib-UC** (power-grid-lib/pglib-uc) | GitHub, 56 JSON cases, converted to MPS with the benchmark's own `uc_model.py` formulation (Knueven, Ostrowski and Watson 2020) | 56 MPS: `ca` 20, `ferc` 24, `rts_gmlc` 12; 482 MB | `ca` and `rts_gmlc` screened; `rts_gmlc` held out of tuning; `ferc` exploratory only |
| **ML4CO item placement** (Gasse et al. 2022) | ds4dm/ml4co-competition `instances.tar.gz` | train 9,900 + valid 100 instances | 60 screening candidates: `item_placement_9900`–`9924` (the 25 validation instances optopt used) and train `item_placement_0`–`34` |
| ML4CO load balancing | same archive | train + valid | not used: easy for every solver in optopt; quota 0 |

## Instance sets in use (2026-10-10)
| set | file | n | what | role |
|---|---|---|---|---|
| **Development set** ([[D-010 Develop the algorithm first on a small development set]]) | `datasets/dev_set.csv` (committed) | 9 | MIPLIB optimisation-hard: `glass4`, `sing44`, `radiationm18-12-05`, `neos-960392`; MIPLIB feasibility-hard (no SCIP solution in 300 s at screening stage 1): `neos-3024952-loue`, `fhnw-binpack4-48`, `rail01`; PGLib-UC `ca__Scenario400_reserves_1` (no solution at stage 1); ML4CO `item_placement_9903` | chosen 2026-10-06 by 08:13 (commit 4127f10) from the stage-1 screening (seed 900) after looking at it, for coverage and fast runs (nnz ≤ 400k); all development work (120 s); **permanently excluded from the confirmatory selection** |
| **Class probe** ([[D-014 Class probe — predictions fixed before the runs]]) | `datasets/classprobe_set.csv` (committed) | 8 | MIPLIB, families outside the development set, SCIP-weak at stage 1: objective with ≤ 4 non-zeros `neos-3004026-krka`, `neos-3656078-kumeu`, `chromaticindex1024-7`, `cryptanalysiskb128n5obj16`; dense objective `comp21-2idx`, `neos-4763324-toguru`; many two-variable equality rows `neos-2657525-crna`, `neos-2746589-doon` (the file also records each one's one-thread HiGHS LP time) | fixed 2026-10-07 12:04 before its runs; used in development since; **permanently excluded from the confirmatory selection** |
| the 17 | both files | 17 | development set + class probe | D-019's interim A9 result at checkpoint C1 and the cuOpt 26.10 re-baseline (#31) run on these; D-018 stage 1 uses 14 of them (not `rail01`, `cryptanalysiskb128n5obj16`, `neos-2746589-doon`) |
| **Hard pool** (confirmatory) | `datasets/hard_pool.csv` | — | selected from the screening candidates by D-002 §2 ([[Hard instance pool]]) | not frozen yet (#1): screening stage 2 is partial, and the freeze must read both exclusion lists above |
| **MIPfeas set** (D-019, #29) | none yet (the harness writes its list) | 233 | the 240 instances of `data/miplib/benchmark-v2.test` minus the 7 marked `=inf=` in `solu.txt` (`bnatt500`, `cryptanalysiskb128n5obj14`, `fhnw-binpack4-4`, `neos-2075418-temuka`, `neos-3402454-bohle`, `neos-3988577-wolgan`, `neos859080`). Checked 2026-10-10: 233 remain, 232 `=opt=` and 1 `=best=` (`supportcase22`); all on disk | v1's external run under Mittelmann's MIPfeas protocol (600 s, integral 0–2); calibration with cuOpt 26.08 alone on ~30 of them at C1 (#29) |

**MIPfeas set vs our other sets (computed 2026-10-10).** The 233 are exactly the 232 MIPLIB screening candidates plus
`supportcase22` (which the screening leaves to E5). Overlap with the 17 development and class-probe instances:

| set | in the 233 | not in the 233 |
|---|---|---|
| development (9) | 7: `glass4`, `sing44`, `radiationm18-12-05`, `neos-960392`, `neos-3024952-loue`, `fhnw-binpack4-48`, `rail01` | 2: `ca__Scenario400_reserves_1` (PGLib-UC), `item_placement_9903` (ML4CO) |
| class probe (8) | all 8 | — |

So 15 of the 17 are in the 233, and the MIPfeas result reported "without them" (D-019) covers 218 instances. The
confirmatory pool's MIPLIB instances come from the same 232 candidates, so every MIPLIB test instance will also be a
MIPfeas instance; its PGLib-UC and ML4CO instances will not.

## Facts that matter for the study
- **Infeasible MIPLIB instances on disk** (`=inf=` in `solu.txt`): `bnatt500`, `cryptanalysiskb128n5obj14`,
  `fhnw-binpack4-4`, `neos-2075418-temuka`, `neos-3402454-bohle`, `neos-3988577-wolgan`, `neos859080`. Excluded.
  (`solu.txt` is `miplib2017-v33.solu` for the whole collection: 44 `=inf=` entries, of which these 7 are benchmark
  instances.)
- **References.** MIPLIB: the `.solu` value (`=opt=` or `=best=`). PGLib-UC and ML4CO have no published optima here:
  the reference is the best of the screening runs and dedicated reference runs (SCIP 1 h × 2 seeds), frozen in
  `datasets/reference.csv` before E2 (D-002 §4). optopt's references for these sets were best-of-6 runs at 30–60 s and
  are not reused. What the runs so far used (`ref_kind` in `datasets/runs.parquet`, 2026-10-09): MIPLIB `opt` (the
  `.solu` value) everywhere; `ca__Scenario400_reserves_1` `best_of_study`; `item_placement_9903` and the other ML4CO
  cases `screening_best_provisional`; 18 of the 32 PGLib-UC screening cases have `ref_kind` `none` — no checked
  incumbent in any run so far (19 had none at stage 1; development runs gave `ca__Scenario400_reserves_1` one). All
  provisional until `reference.csv` (#2).
- **P-degenerate MIPLIB instances** (reference 0, so Berthold's gap is 1 for any other value): e.g.
  `academictimetablesmall`, `cbs-cta`, `cryptanalysiskb128n5obj16`, `neos-3004026-krka`, `ns1116954`, `ns1952667`
  (critique, methodology lens); the screening flags every case, including sign crossings.
- **Sizes that stress the GPU budget** (optopt features): `s100` (14,733 rows, 364,418 binaries, 1.78 M non-zeros),
  PGLib-UC `ferc` (n up to 485,233, 52–59 % continuous), `supportcase22` (2.2 M non-zeros). Genes are stored as int8 for
  pure-binary problems (s100 at P = 256: 93 MB instead of 746 MB as float64).
- **Continuous variables.** PGLib-UC is 52–59 % continuous with equality-coupled power balance; markshare-type
  instances have singleton slack columns. `repair.continuous: auto` decodes singleton continuous columns in closed form
  ([[Feasibility jump local search]]). Non-singleton continuous columns exist on 7 of the 17 development and
  class-probe instances (`ca__Scenario400_reserves_1` 100,704; `sing44` 15,995; `radiationm18-12-05` 14,688;
  `neos-3656078-kumeu` 658; `fhnw-binpack4-48` 105; `item_placement_9903` 33; `glass4` 19; counted 2026-10-08, D-017).
  `repair.continuous: lp_decode` re-optimises them with the integers fixed — the A3 development default
  since 2026-10-09 ([[D-017 LP decode — re-optimise the continuous variables of selected candidates]]).
- **`supportcase22`** is `=best=` (open) in `solu.txt`; it is not a screening candidate (E5 only) but is one of the 233
  MIPfeas instances.
- **Generators.** ML4CO item placement comes from one generator and PGLib-UC from three base systems, so a within-family
  split tests only seen-family generalisation; `rts_gmlc` is held out of tuning for an unseen-family check.

## Derived tables (`datasets/`, committed)
| file | what | status |
|---|---|---|
| `hardness_from_optopt.csv` | 386 instances: P(60)/P(300) from optopt's grid (strategies H0 H1 S0 S1 C0 C1, one seed), provenance `optopt@4dd48395` | background only — **never** a selection criterion (it mixes cuOpt into "best") |
| `dev_set.csv` | the 9 development instances with stratum, reason and stage-1 P(300) / first-feasible time (D-010) | committed 2026-10-06 |
| `classprobe_set.csv` | the 8 class-probe instances with group and one-thread LP time (D-014) | committed 2026-10-07 |
| `runs.parquet`, `incumbents_checked.parquet` | one row per run (837 runs: screening 359, development 476, smoke 2) plus 18 synthetic uncoupled-portfolio rows (`A0+A5`, `synthetic` true) — 855 rows on 2026-10-09 19:21 — and every checked incumbent, built from `traces/raw/` by `itrea build-runs` | regenerated after each pool; not committed yet (untracked on 2026-10-10). Run directories moved to `traces/raw/*/void/` are not read ([[Experiment index]], "Controls moved aside") |
| `hard_pool.csv` | the frozen pool, strata, split, flags | after screening stage 2 (tag `pool-v1`, #1) |
| `reference.csv` | frozen reference values with their source | before the confirmatory runs (#2) |
| per-experiment results | one table per experiment, written by `itrea.analysis` | after each experiment |

Raw traces (`traces/raw/`) and stored solutions are regenerated by the jobs and are not committed; resource records
(`traces/resources/`) are. A versioned archive of the raw traces will be published with the paper, as in optopt.
