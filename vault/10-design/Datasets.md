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

## GAMS's MIPfeas archive (D-020; on disk since 2026-10-11)
Downloaded on Berk's yes (2026-10-11 10:0x, asked whether to download the 38.5 MB archive: **"Yes, download it"**):
`data/mipfeas/2026-09-16_mipfeas_results_and_scripts.zip` (cloud.gams.com share
`dcyWLXABG53NNMW`, Last-Modified 2026-09-16 13:40:56 GMT, sha256
`b8d25892c3657e7205e5d7fab2aaf64612953fde7eb4bdd19f9c0506dbb3e327`), unpacked in `data/mipfeas/gams-2026-09-16/`;
like all of `data/`, not in git.

| file | contents |
|---|---|
| `create_primalintegral.py`, `create_plots.py` | GAMS's scoring script (`calculate_primal_integral`, 2026-09-15) and plotting script |
| `optimal_objective.csv` | MIPLIB's instance table: status (easy / hard / open), sizes, objective = the reference z* |
| `optimalindicator.csv` | a 0 / 1 value per instance and solver column (not used here) |
| `primalintegral.csv` | the published per-instance integral (0–2, 600 s) of every solver × machine |
| `mipfeas.xlsx` | GAMS's results workbook (not read here) |
| `trc_files/` (3,495 traces), `log_files/` (2,796 logs) | one GAMS trace (and log) per instance × solver × machine: cuOpt 26.08 on a GB10 (also 26.02, 26.04 on GB10; 26.04, 26.08 on a B200); CPU solvers on an Intel Core Ultra 9 285K and an AMD EPYC 9B45 |

Checked on 2026-10-11 (D-020): their script on their GB10 cuOpt-26.08 traces gives the published **0.0393** exactly;
every published geo-mean comes out of `primalintegral.csv`; their 233 instances = `data/miplib/benchmark-v2.test` minus
our 7 infeasible ones; their references equal our `solu.txt` on all 233 (|Δ| ≤ 1e-6 relative); all 233 on disk. What
GAMS ran (logs): cuOpt 26.08 through GAMS 54.4.0's link, `num_cpu_threads` 24, `time_limit` 600, both MIP gaps 0, the
20-core GB10 to itself; its trace clock starts at the solver link (D-020 § "The clock"). Our port of their integral
(`itrea.analysis.metrics.gams_integral`, column `pt_gams`) equals their function on all 3,495 traces, and
`itrea gams-trc` exports our runs so that their script reproduces `pt_gams` to 1e-12. Readers:
`itrea.analysis.mipfeas` (`read_gams_trc`, z*, the published P). Used for: H_cu (D-021 R3, from the cuOpt-GB10 26.08
column), the W_SCIP class of satellites2-40 and supportcase22 (GAMS's SCIP 10 trace), D-020's 30-instance sample and the
M4 calibration, M0's replay of R12, and R36's published rows re-scored on our subset.

## Families
| family | public source | on disk | used here |
|---|---|---|---|
| **MIPLIB 2017 benchmark** (Gleixner et al. 2021) | miplib.zib.de, `benchmark.zip`, `miplib2017-v33.solu`, `tag_benchmark.html` | 240 instances, 609 MB | all except the 7 infeasible ones and `supportcase22` are screening candidates ([[Hard instance pool]]); `supportcase22` (open, 2.2 M non-zeros) only in E5; the 233 feasible ones are the MIPfeas set (D-019) |
| **PGLib-UC** (power-grid-lib/pglib-uc) | GitHub, 56 JSON cases, converted to MPS with the benchmark's own `uc_model.py` formulation (Knueven, Ostrowski and Watson 2020) | 56 MPS: `ca` 20, `ferc` 24, `rts_gmlc` 12; 482 MB | `ca` and `rts_gmlc` screened; `rts_gmlc` held out of tuning; `ferc` exploratory only |
| **ML4CO item placement** (Gasse et al. 2022) | ds4dm/ml4co-competition `instances.tar.gz` | train 9,900 + valid 100 instances | 60 screening candidates: `item_placement_9900`–`9924` (the 25 validation instances optopt used) and train `item_placement_0`–`34` |
| ML4CO load balancing | same archive | train + valid | not used: easy for every solver in optopt; quota 0 |

## Instance sets in use (2026-10-10; updated 2026-10-13)
| set | file | n | what | role |
|---|---|---|---|---|
| **Development set** ([[D-010 Develop the algorithm first on a small development set]]) | `datasets/dev_set.csv` (committed) | 9 | MIPLIB optimisation-hard: `glass4`, `sing44`, `radiationm18-12-05`, `neos-960392`; MIPLIB feasibility-hard (no SCIP solution in 300 s at screening stage 1): `neos-3024952-loue`, `fhnw-binpack4-48`, `rail01`; PGLib-UC `ca__Scenario400_reserves_1` (no solution at stage 1); ML4CO `item_placement_9903` | chosen 2026-10-06 by 08:13 (commit 4127f10) from the stage-1 screening (seed 900) after looking at it, for coverage and fast runs (nnz ≤ 400k); all development work (120 s); **permanently excluded from the confirmatory selection** |
| **Class probe** ([[D-014 Class probe — predictions fixed before the runs]]) | `datasets/classprobe_set.csv` (committed) | 8 | MIPLIB, families outside the development set, SCIP-weak at stage 1: objective with ≤ 4 non-zeros `neos-3004026-krka`, `neos-3656078-kumeu`, `chromaticindex1024-7`, `cryptanalysiskb128n5obj16`; dense objective `comp21-2idx`, `neos-4763324-toguru`; many two-variable equality rows `neos-2657525-crna`, `neos-2746589-doon` (the file also records each one's one-thread HiGHS LP time) | fixed 2026-10-07 12:04 before its runs; used in development since; **permanently excluded from the confirmatory selection** |
| the 17 | both files | 17 | development set + class probe | D-019's interim A9 result at checkpoint C1 and the cuOpt 26.10 re-baseline (#31) run on these; D-018 stage 1 uses 14 of them (not `rail01`, `cryptanalysiskb128n5obj16`, `neos-2746589-doon`); D-022 (2026-10-13) runs 13 of them in set `itx-r13` and the other 4 (`chromaticindex1024-7`, `neos-3656078-kumeu`, `glass4`, `radiationm18-12-05`) in `itx-r13b` |
| **cuOpt's fixed 11** ([[D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test]], amendments of 2026-10-13) | named in D-022 | 11 | the development / class-probe instances where cuOpt alone had P(120) ≥ 0.5 in our earlier A6 runs (3 A725 cores): `neos-2746589-doon`, `cryptanalysiskb128n5obj16`, `rail01`, `neos-2657525-crna`, `neos-3024952-loue`, `sing44`, `ca__Scenario400_reserves_1`, `item_placement_9903`, `neos-960392`, `neos-4763324-toguru`, `comp21-2idx` (borderline, 0.506); fixed before any D-022 run, not re-selected on A6F | C1's interim hint: ratio of mean P A9@r13x / A6F@r13x ≤ 0.95 over these 11 (D-022's second amendment); the guard reads all 17 |
| **R13's three** (D-022) | named in D-022 | 3 | `fhnw-binpack4-48`, `neos-3004026-krka`, `neos-960392` — the only instances of the 17 where cuOpt stopped on its own 1e-4 "optimal" in our 120-s runs (9/9, 10/10, 3/7) | D-022's R13 + R14 test (2026-10-13) |
| **Hard pool** (confirmatory) | `datasets/hard_pool.csv` | — | selected from the screening candidates by D-002 §2 ([[Hard instance pool]]) | not frozen yet (#1): screening stage 2 is partial, and the freeze must read both exclusion lists above |
| **MIPfeas set** (D-019, #29) | `datasets/mipfeas_strata.csv` (since 2026-10-12; until then no file) | 233 | the 240 instances of `data/miplib/benchmark-v2.test` minus the 7 marked `=inf=` in `solu.txt` (`bnatt500`, `cryptanalysiskb128n5obj14`, `fhnw-binpack4-4`, `neos-2075418-temuka`, `neos-3402454-bohle`, `neos-3988577-wolgan`, `neos859080`). Checked 2026-10-10: 233 remain, 232 `=opt=` and 1 `=best=` (`supportcase22`); all on disk | v1's external run, now under our own standard D-021 (ITX-1: 600 s, GAMS's integral 0–2 on the call clock, equal footprint F); D-020's 30-instance calibration sample (A6F, M4: 19 of 30 run by 2026-10-13); the sets below |

**MIPfeas set vs our other sets (computed 2026-10-10).** The 233 are exactly the 232 MIPLIB screening candidates plus
`supportcase22` (which the screening leaves to E5). Overlap with the 17 development and class-probe instances:

| set | in the 233 | not in the 233 |
|---|---|---|
| development (9) | 7: `glass4`, `sing44`, `radiationm18-12-05`, `neos-960392`, `neos-3024952-loue`, `fhnw-binpack4-48`, `rail01` | 2: `ca__Scenario400_reserves_1` (PGLib-UC), `item_placement_9903` (ML4CO) |
| class probe (8) | all 8 | — |

So 15 of the 17 are in the 233, and the MIPfeas result reported "without them" (D-019) covers 218 instances. The
confirmatory pool's MIPLIB instances come from the same 232 candidates, so every MIPLIB test instance will also be a
MIPfeas instance; its PGLib-UC and ML4CO instances will not.

## D-021's external sets — `datasets/mipfeas_strata.csv` (fixed 2026-10-12, before any v1 run)
One row per instance of the 233, written by `tools/derive_mipfeas_strata.py` (`--check` re-derives it; reader
`itrea.analysis.itx_strata.read_strata`), deterministic, **sha256 `c0221d429f3368c4181177cbd33b3b3717ddad8946bf712ca5440f6fc4ab35e5`**
(commit c4f0154, after the phase-2 review fixes: the `exposed` column added and GAMS's published values read round
trip; every set and count unchanged from the first version `d2ad90ee040e…`, bb7f2e1). prereg-v2 will name the hash.
Rules: [[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]] R1–R4,
R29, R31, R33.

| set (column) | rule | n | unseen |
|---|---|---|---|
| the 233 | MIPLIB 2017 `benchmark-v2.test` minus its 7 `=inf=`; references = GAMS's `optimal_objective.csv` (= `solu.txt`) | 233 | 218 (the 15 development / class-probe instances, `dev`, are run and reported, never decided on) |
| **H_cu** (`H_cu`) | GAMS's cuOpt-GB10 26.08 `pt_gams` ≥ 0.1 or no solution (`p_cuopt`, from `primalintegral.csv`) — independent of every run of ours | 75 | 65; **64** after the does-not-fit list |
| H_cu at 0.5 (`H_cu_p05`) | the same at ≥ 0.5 or none; reported beside | 34 | 30 |
| **W_SCIP** (`W_SCIP`) | our screening stage 1 (seed 900, 300 s, one X925 core): Berthold P(300) ≥ 0.5 (D-013); satellites2-40 (screening run failed) and supportcase22 (never a candidate) classed by GAMS's SCIP 10 trace (`w_source` `gams-scip10`) | 74 | 62 |
| W-only (`W_only`) | W_SCIP and not H_cu | — | 17 |
| E (`E`) | neither | — | 136; 134 after the list |
| **does not fit F** (`does_not_fit`) | M3 (2026-10-11, A6F at RMM 40 / host 40 GB): `eilA101-2` (killed at the 40-GB host cap; in H_cu, GAMS P 0.230), `nw04` (killed at the cap; E), `supportcase6` (host peak 22.6 GiB > 0.8 × 24 GB; E) — excluded from both arms | 3 | 3 |
| **decision set** (`decide`) | unseen and not on the list | — | **215** |
| E-sample (`e_sample`, `e_quartile`) | 40 of E-unseen minus the list (134): 10 per quartile of `p_cuopt`, `numpy.default_rng(20261012)` | — | 40 |
| repeatability panel (`panel`, `panel_group`) | 20 of H_cu-unseen minus the list (64): 10 with GAMS P ≥ 0.5 or none (`p05`), 10 with 0.1 ≤ P < 0.5 (`p01`), the same rng stream after the E-sample | — | 20 |
| tiers (`tier`, `t4`, `t5`) | T1 = H_cu-unseen (64); T2 = W-only (17) + the E-sample (40) = 57; T3 = the rest of E (94) + the 15 development instances = 109; T4 = the panel at seed 2001; T5 = live A2F on the panel; "" = on the list | — | — |
| **exposed** (`exposed`, R33) | A6F's outcome on cuOpt 26.08 seen before prereg-v2: D-020's 30-instance calibration sample (M4) ∪ M3's 8 probe instances | 37 | 34; 31 in the decision set (T1 8, T2 11, T3 12) — decisions are also reported without them (184; 56 of H_cu) |
| `r12` | R12's stop at the proven optimum applies (`=opt=`; never supportcase22) | 232 | — |

**Not unseen:** the 15 development / class-probe instances among the 233 (`chromaticindex1024-7`, `comp21-2idx`,
`cryptanalysiskb128n5obj16`, `fhnw-binpack4-48`, `glass4`, `neos-2657525-crna`, `neos-2746589-doon`,
`neos-3004026-krka`, `neos-3024952-loue`, `neos-3656078-kumeu`, `neos-4763324-toguru`, `neos-960392`,
`radiationm18-12-05`, `rail01`, `sing44`).

**v1's stage-1 set** ([[D-024 Targeted first, wider only on a hint — staged testing for development and v1]], Berk
2026-10-13 "Yes, stage it"): the repeatability panel (20) plus the 5 H_cu-unseen instances where GAMS's cuOpt found no
solution — `bnatt400`, `neos-3216931-puriri`, `peg-solitaire-a3`, `supportcase19`, `supportcase22` (26 % of the
target's mass; the panel drew none) — 25 instances, all in T1, × {A9, A6F} at seed 2000 under `itx1`, plus A0. The
panel: `30n20b8`, `bab6`, `brazil3`, `markshare2`, `neos-1354092`, `neos-4532248-waihi`, `neos-5104907-jarama`,
`neos-5114902-kasavu`, `roi5alpha10n8`, `savsched1` (`p05`); `buildingenergy`, `cmflsp50-24-8-8`, `k1mushroom`,
`neos-4300652-rahue`, `neos-5049753-cuanza`, `net12`, `nexp-150-20-8-5`, `snp-02-004-104`, `square41`, `uccase12`
(`p01`). **4 of the 25 are on the exposed list** — `neos-4300652-rahue`, `neos-3216931-puriri`, `supportcase22` (M4) and
`snp-02-004-104` (M3); D-024 reports the stage-1 hint with and without the exposed instances and decides with them. A0 runs on H_cu ∪ W_SCIP unseen = 64 + 17 = **81** instances (Berk: "Yes, 81"), the stage-1 instances first. No
selector or job file for stage 1 exists yet (code owed, D-024, #37).

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
- **Incumbents better than a proven optimum (found 2026-10-13, to investigate).** Our checker's tolerance (1e-6:
  bounds and integrality absolute, rows relative to max(1, |side|)) accepts points whose objective lies below MIPLIB's
  `=opt=` value: in the run table of 2026-10-11, 60 checked incumbents in 8 of the 57 glass4 runs (all EA arms; worst
  row violation ≤ 4.5e-7 relative), and in the M4 calibration one cuOpt incumbent on mas74 (11,801.18516 vs
  11,801.18572). cuOpt 26.10 rejected the lowest glass4 one (1,200,006,662.64 vs 1,200,012,599.97) when offered it;
  GAMS's scoring script raises on any incumbent better than z* (ours sets γ = 0 and flags `beats_ref`). The other way,
  4 cuOpt incumbents in M4 failed our check (bound or integrality off by 4.8e-6 – 8.3e-6). Details:
  [[2026-10-11 D-021 measurements M1–M3 and the interim calibration]] § "Checker tolerance"; issue #36.
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
| `mipfeas_strata.csv` | D-021's external sets, one row per MIPfeas instance (section above); sha256 `c0221d429f33…` | committed 2026-10-12 (bb7f2e1, c4f0154); hashed in prereg-v2 |
| `runs.parquet`, `incumbents_checked.parquet` | one row per run (837 runs: screening 359, development 476, smoke 2) plus 18 synthetic uncoupled-portfolio rows (`A0+A5`, `synthetic` true) — 855 rows on 2026-10-09 19:21 — and every checked incumbent, built from `traces/raw/` by `itrea build-runs` | regenerated after each pool; not committed yet (untracked on 2026-10-10 and still on 2026-10-13; the copy on main was built 2026-10-11 14:09 — 939 rows, 20 reused run ids qualified `#<set>` (#33) — and does not hold the D-021 measurement runs). Run directories moved to `traces/raw/*/void/` are not read ([[Experiment index]], "Controls moved aside") |
| `hard_pool.csv` | the frozen pool, strata, split, flags | after screening stage 2 (tag `pool-v1`, #1) |
| `reference.csv` | frozen reference values with their source | before the confirmatory runs (#2) |
| per-experiment results | one table per experiment, written by `itrea.analysis` | after each experiment |

Raw traces (`traces/raw/`) and stored solutions are regenerated by the jobs and are not committed; resource records
(`traces/resources/`) are. A versioned archive of the raw traces will be published with the paper, as in optopt.
