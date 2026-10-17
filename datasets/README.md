# datasets/ — instance lists and derived tables

This folder holds the instance lists the experiments use and tables derived from our own runs. It holds **no instance
files**: the instances live in `data/`, which is not in git, and are fetched from their public sources (below). Detail
on where each family sits on disk and how it is used: `vault/10-design/Datasets.md`.

| file | what | made by |
|---|---|---|
| `dev_set.csv` | the 9 development instances, with stratum, reason and stage-1 SCIP measurements (D-010) | chosen by hand, 2026-10-06 |
| `classprobe_set.csv` | the 8 class-probe instances, with group and one-thread LP time (D-014) | chosen by hand, 2026-10-07 |
| `mipfeas_strata.csv` | D-021's fixed external sets over the 233 MIPfeas instances (hashed in `prereg-v2`) | `tools/derive_mipfeas_strata.py` |
| `hardness_from_optopt.csv` | 386 instances: P(60) / P(300) from the predecessor study's grid; background only, never a selection criterion | from the predecessor study `berkorbay/optopt` (MIT); provenance per row in its `source` column |
| `runs.parquet`, `incumbents_checked.parquet` | the run table and the checked incumbents | `itrea build-runs`; not committed (`.gitignore`) |

## Data sources and attribution

**No instance file is redistributed in this repository.** The CSV files above contain instance names, paths into
`data/`, our own measurements and, where stated below, values published by others. Anyone who reuses these tables or
the instances should cite the sources as follows. Bibliography keys refer to `paper/refs.bib`.

### MIPLIB 2017
- **Used:** the benchmark set (240 instances, `benchmark.zip`) and the solution file `miplib2017-v33.solu`, from
  <https://miplib.zib.de/>. The reference objective values in `hardness_from_optopt.csv` (`ref_source` = published) and
  the `ref_kind` column of `mipfeas_strata.csv` come from that solution file.
- **Cite:** Gleixner, A., Hendel, G., Gamrath, G., et al. (2021). MIPLIB 2017: data-driven compilation of the 6th
  mixed-integer programming library. *Mathematical Programming Computation* 13(3), 443–490,
  doi:[10.1007/s12532-020-00194-3](https://doi.org/10.1007/s12532-020-00194-3) (`gleixner2021miplib`; the article is
  open access under CC BY 4.0), and the website (`miplib2017web`).
- **Terms:** the website asks users to cite that article, reads "© by Zuse Institute Berlin (ZIB)", and carries a
  disclaimer that the data files may contain errors. We found no licence statement for the instance files on its home
  or download page (checked 2026-10-17), which is one reason they are not redistributed here: download them from
  miplib.zib.de.

### PGLib-UC
- **Used:** the unit-commitment cases of [power-grid-lib/pglib-uc](https://github.com/power-grid-lib/pglib-uc)
  (release v19.08; the case files on `master` are identical): `ca` (20), `ferc` (24) and `rts_gmlc` (12).
- **Changes made:** the JSON cases were converted to MPS with the repository's own formulation (`uc_model.py`). The
  MPS files are not redistributed.
- **Licence:** the case files are under the Creative Commons Attribution 4.0 International License
  (<https://creativecommons.org/licenses/by/4.0/>).
- **Cite** the repository with its version (`pglibuc2019`), as its README asks, and the original sources of the cases,
  as listed in that README:
  - `ca` and `ferc`: Knueven, B., Ostrowski, J., & Watson, J.-P. "On mixed integer programming formulations for the unit
    commitment problem" (`knueven2020uc`; published in *INFORMS Journal on Computing* 32, 857–876, 2020; the README
    links the 2018 [preprint](http://www.optimization-online.org/DB_FILE/2018/11/6930.pdf));
  - `ferc`, in addition: Krall, E., Higgins, M., & O'Neill, R. P. "RTO unit commitment test system." Federal Energy
    Regulatory Commission (2012),
    [ferc.gov](https://www.ferc.gov/industries-data/electric/power-sales-and-markets/increasing-efficiency-through-improved-software-1);
  - `rts_gmlc`: Barrows, C., Bloom, A., Ehlen, A., et al. "The IEEE Reliability Test System: A Proposed 2019 Update."
    *IEEE Transactions on Power Systems* (2019), [IEEE Xplore 8753693](https://ieeexplore.ieee.org/document/8753693).

### ML4CO
- **Used:** the item-placement instances (`1_item_placement`, train and valid) of the NeurIPS 2021 ML4CO competition,
  from the instance archive linked in the `DATA.md` of
  [ds4dm/ml4co-competition](https://github.com/ds4dm/ml4co-competition). The load-balancing instances are on disk but
  not used.
- **Licence:** the competition repository is under the BSD 3-Clause License (© 2021 CERC Data Science For Decision
  Making); its `DATA.md`, which links the instance archive, states no separate licence for the instances.
- **Cite**, as the repository asks: Gasse, M., Bowly, S., Cappart, Q., et al. (2022). The Machine Learning for
  Combinatorial Optimization Competition (ML4CO): Results and Insights. *Proceedings of the NeurIPS 2021 Competitions and
  Demonstrations Track*, PMLR 176, 220–231 (`gasse2022ml4co`).

### GAMS MIPfeas benchmark (published values)
- **Used:** GAMS's published results archive of 16 September 2026 (results, traces and the scoring script
  `create_primalintegral.py` of 2026-09-15; source and sha256 in `vault/10-design/Datasets.md` and D-020). The MIPfeas
  instance set is MIPLIB 2017's benchmark set minus its 7 infeasible instances (233).
- **Published values in `mipfeas_strata.csv`:** `p_cuopt` (the published per-instance primal integral of cuOpt 26.08 on
  the GB10, column `cuopt-gb10` of their `primalintegral.csv`), `p_scip10` (SCIP 10, column `scip`), `z_star` (their
  reference objective, `optimal_objective.csv`), and `p300_scip` on the rows with `w_source` = `gams-scip10`, which is
  computed from their published SCIP 10 traces. Every other column is ours.
- **Scoring:** `itrea.analysis.metrics.gams_integral` (column `pt_gams` in our tables) is our reimplementation of their
  `calculate_primal_integral`; on all 3,495 of their traces it reproduces their function. Neither their script nor their
  archive is redistributed here (the archive contains no licence file).
- **Cite:** Bussieck, M., & Dirkse, S. (2026). Expanding the Focus: Introducing the MIPfeas Benchmark. GAMS blog,
  <https://www.gams.com/blog/2026/03/expanding-the-focus-introducing-the-mipfeas-benchmark/> (`bussieck2026mipfeas`);
  for the results page, Bussieck, M. (2026). Open Source MIPFEAS Benchmark on the MIPLIB2017 Instances, in H. D.
  Mittelmann, *Benchmarks for Optimization Software*, <https://plato.asu.edu/ftp/mipfeas.html> (`mittelmann2026mipfeas`).

### This repository's own tables
Our own measurements and tables in this folder fall under the repository's default licence, MIT (`LICENSE`); see the
README's Licence section for the split with CC BY 4.0. The third-party values named above keep their owners' terms and
need the citations above.
