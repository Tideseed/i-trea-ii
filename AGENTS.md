# AGENTS.md — companion for AI agents (and humans)

Companion to the research project **I-TREA II** (Interactive Three Ranks Evolutionary Algorithm, part II) by Berk
Orbay (berk.orbay@tideseed.com). Repository: `Tideseed/i-trea-ii` (private while the study runs). Paths are relative
to the repository root; the code is the Python package `itrea` under `src/`. `CLAUDE.md` points here: this file holds
every rule for agents working in the repository.

**Status (2026-10-10): development phase; no confirmatory run yet.** The pre-registration D-002 v3.1 is binding since
the tag `prereg-v1` (2026-10-06). The SCIP screening's stage 1 is complete (324 candidates, seed 900); stage 2 is
deferred (D-010 §5), so the hard pool is not frozen (#1). The algorithm is developed openly (D-010) at 120 s on
17 instances — the development set (9, D-010, since 2026-10-06, `datasets/dev_set.csv`) and the class probe (8, D-014,
since 2026-10-07, `datasets/classprobe_set.csv`), both permanently excluded from the test set; every development
result is descriptive and lives in `vault/30-results/Development log.md`. Development defaults: **A3** (SCIP ↔ EA) =
`configs/itrea2-dev-rc-ld.yaml` (lp_decode adopted, D-017); **A9** (SCIP ↔ EA ← cuOpt, D-015) and its matched control A3x =
`configs/itrea2-dev-rc-3way.yaml`, with the GPU back-off (D-018) past stage 0 (PROCEED) and stage 1 owed. Built and
not adopted: Rank 1 by cuOpt's clearing rule (D-016). A9 pilot (seed 0, 7 instances, descriptive): vs SCIP alone 7
better / 0 worse, mean d +0.68 [+0.42, +0.90]; vs cuOpt alone 3 / 4, −0.07 [−0.42, +0.25]. Tests on 2026-10-09: CPU
suite 816 passed, the GPU tests of the new code 40 passed. **v1 target (D-019, Berk 2026-10-10):** checkpoint C1 on
Fri 2026-10-23 (development closes on cuOpt 26.08), v1 at the end of November 2026 (aim Fri 11-27, hard end Mon 11-30)
on cuOpt 26.10, daytime windows only; targets in the next-but-one section. Newest checkpoint:
`vault/90-checkpoints/2026-10-09-1925.md` (D-019 came after it: journal `vault/20-journal/2026-10-10.md`). Every
hypothesis and target below is pre-registered or planned, not a finding. Do not write, in any file, that I-TREA II
"beats", "improves on" or "matches" anything until the corresponding experiment note in `vault/30-results/` says so
with its numbers. (History: on 2026-10-06 01:4x the status read "built and tested on toy models; no commit, no tag, no
experiment, no results" — 413 passed, 2 skipped.)

## Setting
- Machine: NVIDIA DGX Spark — GB10 (10 × Cortex-X925 @ 3.9 GHz = cores 5–9, 15–19; 10 × Cortex-A725 @ 2.8 GHz = cores
  0–4, 10–14), Blackwell GPU, 121 GiB unified memory shared by CPU and GPU. Ubuntu 24.04.4, kernel 6.17.0-1031-nvidia,
  driver 580.173.02, CUDA 13.0. The machine is shared with other sessions and with two local LLM servers.
- Software: Python 3.12.3; NumPy 2.4.6, SciPy 1.18.1; CuPy 14.2.0 (GPU) and NumPy (CPU) as one code path
  (`itrea.backend`); solvers SCIP 10 (PySCIPOpt 6.2.1, primary partner), HiGHS 1.15.1 (replication partner, after its
  user-solution round-trip test; also the EA's own LP and the lp_decode worker), NVIDIA cuOpt 26.08
  (`cuopt-cu13==26.8.*`). cuOpt is no longer only a control: alone it is arm A6, and since D-015 (2026-10-07) it is
  the third process of arm **A9** (SCIP ↔ EA ← cuOpt), where it streams its incumbents and bound to the EA on its own
  channels and the EA passes them on to SCIP. cuOpt never receives a solution mid-run — its `SetSolutionCallback`
  disables its presolve (feature request NVIDIA/cuopt#2073, filed 2026-10-07); a MIP start at launch is used with
  presolve on (verified on glass4 2026-10-07, ~10 s overhead per restart), but no arm uses it yet. **v1 runs on cuOpt
  26.10** once it is released (separate venv, API check, A6 re-baselined on the 17 instances; #31); if 26.10 is not out
  by Fri 2026-11-06, the v1 runs proceed on 26.08 and 26.10 follows as an addendum (D-019). Numba 0.64.0 (the CPU
  feasibility-jump reference); redis-py 8.1.0 for the exchange platform; pandas 3.0.3 / PyArrow 23.0.1 for the derived
  tables. All pinned in `pyproject.toml`. One entry point: `itrea <command>` (`src/itrea/cli.py`; `itrea --help`).
- Instances (public, on disk): MIPLIB 2017 benchmark, PGLib-UC unit commitment, ML4CO item placement
  (`vault/10-design/Datasets.md`). The confirmatory study will use a frozen pool of **hard** instances chosen on SCIP
  alone (`vault/10-design/Hard instance pool.md`; not frozen yet, #1). Development: the 17 instances above. v1's
  external run uses the 233 feasible MIPLIB 2017 benchmark instances of Mittelmann's MIPfeas
  (`data/miplib/benchmark-v2.test` minus its 7 infeasible ones; no download needed).
- Metric: the primal integral **P(T)** ∈ [0, 1] (Berthold 2013), built only from incumbents that pass the independent
  exact checker; T = 120 s in development, 300 s in the confirmatory test; definitions in
  `vault/10-design/Trace format and metrics.md`. Results are read in D-013's terms (weak sets first, the mean as a
  guard). v1's external metric is Mittelmann's MIPfeas primal integral (0–2, 600 s, shifted geometric mean, shift
  0.001; D-019).
- Predecessor: `berkorbay/optopt` (public, MIT, doi:10.5281/zenodo.22947442). Its CPU evolutionary partner beside SCIP
  (experiment X2) found no gain at 60 s and lost to SCIP ‖ SCIP; its strongest result was the SCIP/HiGHS ‖ cuOpt race.
  Code reused from it carries the line "adapted from berkorbay/optopt@4dd48395, MIT".

## The question, the aim, the v1 targets and the pre-registered hypotheses
Berk's brief (2026-10-05) and main claim (2026-10-06, `vault/50-decisions/D-007`): *"a GPU-based evolutionary
algorithm metaheuristic coupled with an exact solver can beat either single solution."* I-TREA — Berk's master's
research (Orbay 2011, METU M.S. thesis, open access, `orbay2011itrea`; thesis
manuscript, unpublished) — is the starting point; the principle carried over is **keeping good infeasible solutions**
(Rank 0). Part I reserved a share of the population for them as a floor; part II caps it, as Berk's brief says.

**Aim (2026-10-07, `vault/50-decisions/D-013`; it amends D-007).** Berk: *"I don't aim our coupling to be generally
better but to help both SCIP, cuOpt (and Highs if possible) in some of the hardest problems where they are the
weakest. Our performance metric is improvement upon the worst performances, of course with a regard to mean to some
degree."* Metric: for each solver B, the weak set W_B = the instances where B alone has P ≥ 0.5 on screening /
reference seeds never used for the evaluation; primary = the coupling against B alone on W_B (share improved, mean P
reduction, mean bounded d with its bootstrap interval and the Wilcoxon test, worst case after coupling); guard = the
mean over all instances, d ≥ −0.05 against each solver alone.

**v1 (`vault/50-decisions/D-019`, decided by Berk 2026-10-10).** v1 = arm A9 frozen (SCIP 10 ↔ GPU EA ← cuOpt on one
GB10) with the development defaults that survive D-018 stage 1 and #26, plus a pre-registered confirmatory test on
unseen hard instances and an external run under Mittelmann's MIPfeas protocol. The targets enter prereg-v2 verbatim:

| | minimum (v1 passes) | target | stretch |
|---|---|---|---|
| unseen hard instances: A9 vs SCIP alone on W_SCIP | mean d > 0, 95 % CI > 0 | mean P −40 %, ≥ 70 % of W_SCIP improved | −55 % |
| unseen hard instances: A9 vs cuOpt alone on W_cuOpt | mean d > 0, 95 % CI > 0 | mean P −25 %, ≥ 60 % of W_cuOpt improved | −40 % |
| guard, all instances | mean d ≥ −0.05 against each solver alone | | |
| MIPfeas protocol (233 instances, 600 s) | A9's geo-mean ≤ our own cuOpt-26.10-alone geo-mean on the same GB10 | mean PI −25 % on the hard part (our cuOpt-alone PI ≥ 0.1 or no solution) and geo-mean −10 % | geo-mean ≤ 0.0274 (SCIPConc-11, published) |

The external targets are anchored to our own same-version cuOpt run because Mittelmann publishes only cuOpt 26.08 and
only aggregates (cuOpt-26.08 on GB10: 0.0393). Milestones (GitHub): **C1 checkpoint (2026-10-23)** — #23 (D-018 stage
1 decided), #26 (recombine each new partner incumbent with the elite, decided), #27 (lp_decode), #30 (lp_decode inside
A9: a core layout for the EA), #29 (MIPfeas harness, calibrated with cuOpt 26.08 alone on a ~30-instance sample against
the published GB10 row), an interim A9 result on the 17 instances × 2 seeds against SCIP alone and cuOpt alone, and a
go / no-go for the final phase; **v1 (end of November 2026)** — #31 (cuOpt 26.10), #32 (prereg-v2, confirmatory and
MIPfeas runs), #1 (pool freeze), #12 (paper). The MIPfeas runs need ≈ 78 h of run time — about 8–10 long daytime
windows with the LLM servers off. The confirmatory test-set size is open: D-002 v3.1 says 32 × 2 seeds, D-010 §4
proposed 16 (#11), D-019's time estimate assumed ~32; prereg-v2 fixes it.

**Pre-registration v3.1 — history; binding since the tag `prereg-v1` (2026-10-06); none of it tested.**
`vault/50-decisions/D-002 Pre-registration — hypotheses, arms, analysis.md` was written for arm A3 (SCIP ⇄ GPU EA)
before D-013 and D-019 changed the aim, the primary arm (A9) and the targets. Any change is a new, dated decision;
those changes go into **prereg-v2** (#32), which is pending and must be written and tagged before any confirmatory run.
The v3.1 summary, unchanged:

| | hypothesis (lower P(300) is better) | control |
|---|---|---|
| **H1a** (primary) | A3 SCIP ⇄ GPU EA, two-way exchange < A0 SCIP alone (harness-matched) | — |
| **H1b** (primary) | A3 < A5 the EA alone (GPU) | — |
| H2 (secondary, Holm) | A3 < the uncoupled portfolio: pointwise best of A0 and A5 of the same block (live check: A3r, no exchange) | "the exchange adds value beyond running both" |
| H3 | A3 < A2 SCIP ‖ cuOpt race on the same GPU and cores | is our EA a better partner than cuOpt? |
| H4 | A3 < A1 SCIP ‖ SCIP | equal CPU |
| H5 | A3 < A0* tuned SCIP alone | tuning budget equal to the EA's |
| H6 (mechanism) | A5 (x = 10 %) < A5x0 (x = 0), on the EA's own P(300) | do good infeasible solutions help? |

The main claim holds only if H1a **and** H1b are supported (intersection–union, each at α = 0.05). Per comparison:
d_i = (P_ctrl − P_arm)/max(P_ctrl, P_arm, 10⁻³) per instance (seeds averaged), Wilcoxon signed-rank on d_i, mean d_i
with a 95 % bootstrap interval; "negative" only if the whole interval lies below the +0.05 margin, otherwise
"inconclusive" unless supported. 32 test instances × 2 seeds (1000, 1001); minimum detectable effect about 0.20–0.25
in d. A MIPfeas-style integral is reported beside P(T) as a secondary metric (a published GB10 cuOpt baseline exists
under it). Exploratory: A5y0 (protected elite), A4 (CPU EA: the GPU's share), A6 (cuOpt alone), exchange-direction and
other ablations, HiGHS as partner, 1,800 s, strata.

## The three ranks (part I's names; `src/itrea/ea/ranks.py`)
| rank (I-TREA name) | who | size | rule |
|---|---|---|---|
| **Rank 0 — infeasible good** | infeasible individuals whose key beats the incumbent (the population's best feasible or the solver's incumbent, whichever is better) | at most ⌊x · P⌋, x = 10 % — a **total cap** | over the quota: Pareto fronts in (key, violation), the overflowing front thinned by nearest distance; candidates beyond the solver's **proven** bound are eliminated (the solver answers for part I's decision maker); over-quota candidates are eliminated, never kept as Rank 2 |
| **Rank 1 — feasible good** | the best feasible individuals | ⌈y · min(#feasible, P)⌉, y = 10 % | protected every generation; equal-key twins leave Rank 1. Option, **not adopted** (D-016, `ea/elite.py`): `ranks.rank1_mode: clearing` — up to ⌈y · P⌉ members chosen best-first by cuOpt's clearing rule (any two differ in ≥ k integer variables, k = 1–10 by the number of integers); seeds 0–3 against the default: mean d −0.033 [−0.122, +0.050]; the size-only variant `ranks.elite_basis: population` was not adopted either (−0.000 [−0.044, +0.049]). The default stays `top` |
| **Rank 2 — rejects** | everything else | the rest of P | generational by default: only this generation's offspring, injected and immigrant individuals compete (the old Rank 2 dies); non-dominated sorting in (key, violation) with crowding distance |

Tournament order Rank 1 < Rank 0 < Rank 2. Before any incumbent exists the population is in a feasibility phase
(ranked by violation). The solver is the "interactive" counterpart: it supplies its incumbent, very good pool
solutions, its proven bound and its root LP, and receives the EA's better incumbents as new starting points. Since
D-011 SCIP also streams good infeasible node-LP points to the EA (the LP stream), and in A9 cuOpt streams its incumbents
to the EA one way (D-015). Beside the ranks the EA keeps a **frontier**: an archive of diverse near-optimal feasible
solutions, non-dominated in (objective, distance from the incumbent) (`vault/50-decisions/D-005`; exploratory metrics
only).

## Map
Status: **core** = written before the design critique of 2026-10-05 and revised after it; **built** = written by build
agents from 2026-10-06 on (the first build 00:0x–01:2x; development modules through 2026-10-09) and covered by the
test suite (CPU 816 passed; the GPU tests of back-off, lp_decode, elite and EA run 40 passed; 2026-10-09). Caveat:
`ea/population.py`, `ea/evaluate.py`, `ea/frontier.py`, `exchange/redis_mailbox.py`, `trace.py` and `problem.py` have
no test file of their own — they are exercised through the engine, EA-process and runner tests (default settings);
`ea/ranks.py` has `tests/test_ranks.py` and `tests/test_elite.py`, the file mailbox `tests/test_review_mailbox.py`.
Check a non-default branch before relying on it (`vault/80-reviews/2026-10-05 design critique — response.md`).

| path | what | status |
|---|---|---|
| `src/itrea/config.py` | `EAConfig`: every parameter, validated, hashed; YAML in `configs/` | core |
| `src/itrea/problem.py` | MIP read with HiGHS into arrays; per-row float64 feasibility check under the partner's tolerance rule (`tol_mode`) | core |
| `src/itrea/backend.py` | CuPy / NumPy backend; the CuPy pool limit is the only GPU memory cap on GB10 | core |
| `src/itrea/trace.py` | per-process event stream, summary, every incumbent vector (`save_incumbents`); monotonic shared clock | core |
| `src/itrea/paths.py` | repository paths; instance paths stored relative to the root | core |
| `src/itrea/benchmarks/toy.py` | toy models (knapsack with DP optimum, set cover, mixed) and `write_mps` — the only models tests may solve | core |
| `src/itrea/benchmarks/references.py` | reference values: MIPLIB `.solu`, the frozen `datasets/reference.csv`, provisional references before the freeze | built |
| `src/itrea/ea/evaluate.py` | device copy of the problem; batched, chunked evaluation (key, violation V, feasibility) | core |
| `src/itrea/ea/population.py` | population arrays, ids, provenance bits, compact gene dtypes, exact duplicate detection | core |
| `src/itrea/ea/ranks.py` | three-rank survival | core |
| `src/itrea/ea/elite.py` | Rank 1 as a distinct elite (D-016, #25): cuOpt's clearing rule on the device (`clearing_pick`, blocked distances; `min_diff_auto` = cuOpt's k by the number of integers) and the Rank 1 spread metrics (`elite_spread`, `r1_*`, logged in both modes); used by `ranks.py` under `ranks.rank1_mode: clearing` — 2026-10-08, run in D-016's test (seeds 0–3), not adopted | built |
| `src/itrea/ea/frontier.py` | the frontier archive (k-objective non-dominated sort, 2-D hypervolume) | core |
| `src/itrea/ea/selection.py` | parent selection (tournament; comparators `itrea`, `deb`, `stochastic`) | built |
| `src/itrea/ea/crossover.py` | `agreement`, `uniform`, `block`, `k_point` | built |
| `src/itrea/ea/mutation.py` | `flip`, `step`, `reset`, `toward_lp`, `violation_guided`, `objective_guided` | built |
| `src/itrea/ea/seeding.py` | initial population, immigrants, LP point after `t0` | built |
| `src/itrea/ea/repair.py` | `LocalSearch` dispatcher over the three implementations below | built |
| `src/itrea/ea/ls_fj_cpu.py` | feasibility jump on one CPU core (reference and arm A4's repair) | built |
| `src/itrea/ea/ls_fj_gpu.py` | feasibility jump as a CuPy `RawKernel`, one thread block per individual | built |
| `src/itrea/ea/ls_dense.py` | vectorised full rescore (baseline only) | built |
| `src/itrea/ea/continuous.py` | closed-form decoding of singleton continuous columns (`repair.continuous: auto`) | built |
| `src/itrea/ea/lp_decode.py` | the LP decode (`repair.continuous: lp_decode` = auto + a HiGHS worker process that fixes the integers of a few candidates per generation and solves the LP over the continuous variables; D-017, issue #27): pure selection / budget / screen functions and the `LPDecoder` worker; the runner refuses it for A9 / A3x (one EA core) — built 2026-10-08, review fixes 2026-10-09 (`vault/80-reviews/2026-10-08 lp_decode review (workflow).md`), GPU tests passed; run in D-017's test (24 runs, 2026-10-09) and **adopted as the A3 development default**; inside A9 it needs a core layout first (#30) | built |
| `src/itrea/ea/repair_clock.py` | the repair clock's controller: one lane-step horizon per generation from each horizon's measured gain per second (`repair_clock`, off by default; shadow mode logs only) — built 2026-10-06; Stage 1 (shadow) and Stage 2 (active) run 2026-10-07; **active in the development default** since D-012 addendum 2 | built |
| `src/itrea/ea/backoff.py` | A9's GPU back-off (D-018): a contribution-gated duty cycle — phase Y (reduced duty until the first incumbent anywhere), then a ladder of duties stepped down after a state-dependent patience in busy seconds and restored by material leads / SCIP-confirmed assists; a host-side token bucket the engine sleeps on between generations and between FJ launches (`backoff`, off by default, inert unless `exchange.from_cuopt`; modes active / shadow / forced) — built 2026-10-09; stage 0 (18 runs: forced duties ½ / ¼ / ⅛ against shadow bo0, A6 and a same-day A2) complete 2026-10-09: PROCEED; stage 1 (bo1 vs bo1s, 84 runs) owed | built |
| `src/itrea/ea/aggregate.py` | exact aggregation of two-variable equality rows (a presolve-lite for the genome) — 2026-10-06, **not wired into the EA** (the payoff probes on radiationm were negative) | built |
| `src/itrea/ea/engine.py` | the generation loop, exchange in/out, genealogy, frontier updates | built |
| `src/itrea/ea/run.py` | EA process: `python -m itrea.ea.run …` | built |
| `src/itrea/exchange/mailbox.py` | file mailbox on `/dev/shm` (fallback transport) | core |
| `src/itrea/exchange/redis_mailbox.py` | Redis transport and shared pool; `open_mailbox(spec, role)` | core |
| `src/itrea/solvers/common.py` | what the solver processes share: arguments, barrier, recorder, exchange accounting, exit codes | built |
| `src/itrea/solvers/scip_run.py` | SCIP process; injection through a heuristic plugin (`trySol`) | built |
| `src/itrea/solvers/highs_run.py` | HiGHS process; user-solution callback | built |
| `src/itrea/solvers/cuopt_run.py` | cuOpt process; never receives solutions; RMM limit; in A9 (`--send`) it posts its incumbents and bound on its own channels (`to_ea_cuopt` / `info_cuopt`) | built |
| `src/itrea/experiments/gates.py` | lease, memory gate, thermal gate, run lock, watchdog | core |
| `src/itrea/experiments/gpu_health.py` | per-second SM clock, temperature and foreign-GPU sampling; block voiding | built |
| `src/itrea/experiments/exchange_ctl.py` | the `itrea-redis` container: up, status, flush one run's keys, down | built |
| `src/itrea/experiments/record_env.py` | versions, hardware, git state for every run record | built |
| `src/itrea/experiments/pair.py` | one run: launch the arm's processes, start barrier, stop, every exit code checked, `run.json`; A9's three processes and its matched control A3x (D-015) | built |
| `src/itrea/experiments/pool.py` | a job file under every D-002/D-003 control: Williams arm order per block, blocks never split, gates per run | built |
| `src/itrea/experiments/make_jobs.py` | job files: `screen1`, `screen2`, `e1`, `e4`, `confirm`, `smoke`, `backoff0` / `backoff1` (D-018 stages 0 / 1) (`itrea make-jobs …`) | built |
| `src/itrea/benchmarks/hardset.py` | the hard pool: screening candidates, the stage-2 rule, strata, family split, freeze (`itrea screen-candidates`, `screen-stage2`, `pool-freeze`) | built |
| `src/itrea/analysis/metrics.py` | primal gap, P(T), primal-dual integral, times to target (adapted from optopt) | built |
| `src/itrea/analysis/check_solutions.py` | independent exact check of every stored incumbent (adapted from optopt) | built |
| `src/itrea/analysis/build_runs.py` | run directories → `datasets/runs.parquet`, `datasets/incumbents_checked.parquet` | built |
| `src/itrea/analysis/stats.py` | d_i, Wilcoxon, bootstrap, decision rule, censored endpoints | built |
| `src/itrea/analysis/lineage.py` | depth-1 matched Rank 0 odds ratio, Mantel–Haenszel (descriptive) | built |
| `src/itrea/analysis/prereg.py` | the pre-registered analysis of D-002 v3 from `runs.parquet` (`itrea prereg`) | built |
| `src/itrea/analysis/reproduce.py` | re-derive `datasets/` and `paper/generated/` and compare (`itrea reproduce`) | built |
| `src/itrea/analysis/backoff.py` | D-018's stage-0 calibration rules and stage-1 criteria (M1–M5, outcome, fallbacks) applied to the run table and run directories, with a verdict (`itrea backoff-report --stage 0\|1`); the sanity rule gates on the measured duty only and reports the lane-steps ratio (D-018's second stated deviation; `--steps-gate` = the rule as first written); stage-1 control `A9@bo1s` — stage 0 read 2026-10-09 | built |
| `src/itrea/cli.py` | the `itrea` entry point: one table maps every command to its module | built |
| `configs/` | `itrea2-default.yaml` (all defaults, generated from `EAConfig()`; hash `9ac4a93bf93b`; `tests/test_configs.py` keeps it equal to the code). **Current development defaults** (D-010; never a confirmatory configuration): **A3** — `itrea2-dev-rc-ld.yaml` (`itrea2-dev-rc-3way` + `repair.continuous: lp_decode`, hash `f030e23909ad`, D-017); **A9 / A3x** — `itrea2-dev-rc-3way.yaml` (repair clock active, adaptive repair strategy, the EA's own HiGHS LP, LP stream, deep repair; hash `1d12ad490dd0`; the runner adds `exchange.from_cuopt` for A9 and refuses lp_decode there); **D-018 stage 1** — `itrea2-dev-rc-3way-bo1.yaml` (back-off active with the stage-0 values: D_Y ½, floor ⅛, yield_until 30 s, patiences 40 / 40; run hash `f67354be775d`) against `-bo1s.yaml` (the same values in shadow mode = the control; run hash `62bf82b236f6`). History: `-3way-bo0/-bof50/-bof25/-bof12.yaml` (D-018 stage-0 arms: shadow, forced duty ½ / ¼ / ⅛), `itrea2-adaptive.yaml`, `-adaptive-lps.yaml`, `-adaptive-lps-rc.yaml` (earlier development steps, D-010–D-012), `itrea1-faithful.yaml` (a record of part I's choices, not a planned comparison), `ablations/*.yaml` (D-002's E3) | core |
| `tools/serving-window.sh` | stop / restore the two local LLM servers for a window | core |
| `tools/bench_ls.py` | local-search throughput, GPU vs CPU reference (`itrea bench-ls`) | built |
| `tools/d016_report.py` | D-016's analysis: A3@cl1 vs A3@rc1d and the size-only A3@tp1 (mechanism and outcome rules of the record) from `runs.parquet` and the run directories | built |
| `tools/d017_report.py` | D-017's analysis: A3@ld1 vs A3@rc1e (M0–M4 and the outcome rules) | built |
| `tools/probe_lp_decode.py` | probes the LP decode worker on stored vectors (CPU, read-only; D-017's gates before a window) | built |
| `tools/cuopt_verify_mipstart.py` | cuOpt 26.08 MIP-start and stopping check on glass4 (GPU: only inside a window; run 2026-10-07) | built |
| `tools/agg_probe.py`, `tools/agg_improve_probe.py` | CPU probes of `ea/aggregate.py` (feasibility and improvement payoff; 2026-10-06, negative on radiationm) | built |
| `tools/lps_smoke.py` | hand-launched smoke run of the LP stream (SCIP + the NumPy EA through a file mailbox) | built |
| `tools/screening-night.sh` | the screening chain: stage 1 → `build_runs` → stage 2 under one deadline (D-002 v3.1 §1.2) | built |
| `datasets/` | derived tables; `dev_set.csv` and `classprobe_set.csv` are the two lists excluded from the test set; `runs.parquet` / `incumbents_checked.parquet` are rebuilt by `itrea build-runs` and not committed yet; `hardness_from_optopt.csv` is background only | — |
| `jobs/`, `traces/` | job files; raw traces (not committed) and resource records (committed) | — |
| `paper/` | LaTeX skeleton (`main.tex`, `sections/`), `refs.bib` (legally readable works only) | skeleton |
| `vault/` | the research vault (Obsidian): journal, design, decisions D-NNN, results, literature, incidents, reviews, checkpoints | — |
| `.claude/skills/` | repository skills: `itrea-window`, `itrea-journal`, `itrea-result-note`, `itrea-new-experiment`, `itrea-literature`, `itrea-reproduce` | — |

## Rules for new experiments
The runner enforces most of these (`itrea.experiments.gates`, `vault/50-decisions/D-003`); documentation alone is not
a control. Skills: `itrea-window`, `itrea-new-experiment`.
- **Lease first.** No timed run without the box-window lease held by this project:
  `box-window claim --who "i-trea-ii (Claude Code)" --why "…" --until HH:MM [--stops ornith-vllm,llama-qwen38] [--mem GB]`;
  `box-window release` when done. Other sessions share the box; a held lease by someone else means wait.
- **Daytime windows only.** No window in the night rest (00:00–07:00): Berk, 2026-10-10, "No nights" (D-019; before
  that D-009 asked for his yes for each night). Long run sets (v1's MIPfeas runs, ≈ 78 h) are split into daytime
  windows. The runner honours the host fan governor's rest flag (D-009): no new run starts while it exists.
- **Serving window.** Confirmatory runs need both local LLM servers stopped (vLLM `ornith-vllm` :8094, llama.cpp
  `llama-qwen38` :8092; Berk's yes, 2026-10-05); development windows stop both too whenever the memory gate needs it —
  A9 (two CUDA processes) needs ≈ 73 GB. Stop and restore only with `tools/serving-window.sh stop|restore`, schedule
  the restore on a timer before starting, and verify after restore: `is-active` **and** `is-enabled`, health
  endpoints answer, canaries resumed, lease released.
- **Memory gate.** Σ host `MemoryMax` + Σ CuPy pool limits + 1 GB non-pool overhead per CUDA process + cuOpt's RMM
  limit + the Redis cap (2 GB) + 20 GB reserve ≤ MemAvailable at launch; the watchdog stops the run set below 18 GB.
  `MemoryMax` does **not** charge GPU memory on GB10.
- **One GPU arm at a time.** Timed solvers pinned one per X925 core; the partner footprint (GPU + the same 3 A725
  cores + the same GPU memory limit, **12 GB**: the EA's CuPy pool = cuOpt's RMM limit, `EAConfig.gpu_mem_gb`, set by
  cuOpt's measured need of ~10.5 GB, D-003) is identical for cuOpt and the EA. In A9, SCIP runs on X925 5, the EA on
  X925 6 and cuOpt on A725 1–3 (two CUDA processes, 12 GB each); its matched control A3x has the same EA footprint
  without cuOpt (D-015). `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` in every process.
  Never run analysis with default thread counts beside timed runs.
- **Thermal.** GPU arms start only below 80 °C (ACPI maximum), CPU-only arms below 88 °C (D-008); pause the run set at
  ≥ 92 °C, resume below 80 °C.
- **GPU health.** A GPU run with a median SM clock below 2,000 MHz or foreign GPU activity above 2 % of its budget
  voids its whole block (all arms of that instance × seed) and halts the run set.
- **Frozen configuration.** Test-set instances run only with the configuration whose hash equals `frozen-e2`'s, with
  `prereg-v1` an ancestor of HEAD. Tuning happens on the tuning set only, within the logged E4 budget.
- **Every incumbent checked.** Every incumbent vector of every process is stored and checked by the independent exact
  checker; only checked incumbents enter a metric. Exit codes of every process are checked; a crash is kept and
  truncated, never silently re-run.
- **Clock rule.** Before the barrier only parse, imports, CUDA context and kernel warm-up on a synthetic problem; any
  LP, presolve or evaluation of the instance's population after `t0`, in every arm. A run that compiles a kernel after
  `t0` (`compiled_on_clock`) voids its block (D-002 v3.1).
- **Deterministic warm-up.** Before `t0` the warm-up runs every crossover and mutation operator on 1-, 2- and
  3-child batches with the run's gene dtype (87bc759, 2026-10-09) and the elite kernels on three copies of one warm-up
  vector, so they do not depend on how many distinct members the warm-up population holds (6ae6a6f, 2026-10-08) —
  operators drawn at random cover only by chance. A new kernel, operator or batch shape gets its own warm-up entry.
- **Cold-cache GPU test before the timed run.** Before the first timed run of new or changed GPU code, run the
  gpu-marked tests inside the window (lease held), including the cold-cache tests (`test_cupy_cold_cache_*` in
  `tests/test_ea_run.py`, `test_elite.py`, `test_lp_decode.py`: an empty `CUPY_CACHE_DIR`, nothing may compile after
  `t0`). The test must prove the run reached the new code, not only that nothing compiled (incident 2026-10-08: a
  vacuous pass let a block compile on the clock, voided; on 2026-10-09 the test caught the block crossover's one-child
  chunk compiling after `t0` before any timed run).
- **No commits to `src/` or `configs/` while a pool can launch runs** (including a chained pool): every run's processes
  import the current source and the pool reads the config file when it plans a run (incidents 2026-10-06 15:15–16:35:
  one pool's runs spanned several code states; 16:54: an edited config halted a pool). Develop in a git worktree on its
  own branch (`git worktree add -b <branch> ../i-trea-ii-<name>`; `dev-next` and `dev-backoff` were built that way),
  test there on the CPU only on cores outside the run
  lane (`CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 ../i-trea-ii/.venv/bin/python -m pytest -q -m
  "not gpu"`), and merge after the pool has ended.
- **Development changes: one variable, the test fixed first.** Each change gets a decision record whose mechanism
  check and outcome rules (per-instance bands from the control's seed-to-seed spread) are written before any run; the
  control is re-run in the same window; a rule that fails by the letter is recorded as a stated deviation, never
  silently re-read (D-012, D-016–D-018).
- **Redis.** Only the project's own container `itrea-redis` on 127.0.0.1:6390, keys under `itrea:<run_id>:`, deleted
  at the end of each run. Never the host's other Redis instances.
- **No downloads** (instances, wheels, models) without Berk's explicit ask. No customer data (customer) anywhere.
- **No personal data to outside services.** Never send Berk's (or anyone's) e-mail address or other identifying data
  to an outside API; where an API asks for a contact address (Unpaywall, Crossref, OpenAlex), send none or a project
  placeholder (incident 2026-10-05, `vault/60-incidents/`).
- **Never `pkill -f` / `pgrep -f` from an agent shell**: the pattern is in the shell's own command line, so it matches
  (and kills) the shell itself (exit 144; incident 2026-10-06 01:4x). Stop a helper by the PID it wrote or by the file
  it holds or polls; wait on an output file, not on a process name.
- **Toy models only in tests**: `itrea.benchmarks.toy` or tiny generated MPS; never a solver or the EA on a real
  MIPLIB / PGLib / ML4CO instance inside a test (parsing one file for a shape check is fine).
- **Record everything.** Journal entries (`vault/20-journal/`, append-only, host local time — CEST, CET from 2026-10-25), decision records D-NNN, a result
  note per experiment with failures included, incidents in `vault/60-incidents/`. Skills: `itrea-journal`,
  `itrea-result-note`.
- **Upstream reports and submissions** (issues, PRs, MIPLIB solutions, arXiv): drafts in `vault/70-upstream/`, filed
  only with Berk's yes for that one post.

## Writing rules
Plain, precise English; no hype. Never claim "first" or "novel" — `vault/40-literature/Prior art.md` (sweeps A–E,
91 works) records close prior art: cuOpt's own GPU population with recombination runs beside cuOpt's internal CPU
branch-and-bound with solutions flowing both ways, keeps infeasible members and protects the best feasible one; ReXi
runs local-search and LNS workers beside SCIP workers over a shared pool; Puchinger, Raidl and Gruber (2005) and
Nwana et al. (2005) exchange incumbents both ways between a heuristic and a B&B solver; CHAP pools GPU and CPU
heuristics' feasible and infeasible candidates; IDEA reserves a share of the population for infeasible solutions. None
found has the whole combination; the differences are stated as differences. Name harnesses and models plainly (e.g.
"Claude Code (Anthropic Claude Opus 5.5)"); no internal agent pseudonyms. Author line: Berk Orbay. Cite only works with
a legally readable full text; paywalled works live only in the vault's paywalled list and the paper's paywalled
appendix (skill `itrea-literature`).

AgentRA (Agent Research Assistant) — the AI agent that designs with the author and executes the experiments — is
mainly Claude Opus 5.5, working through Claude Code.

**Updates to Berk** (his rules, 2026-10-06 / 10-08 / 10-11): frequent, short, quantified (effects with intervals, counts,
p — not adjectives). Each update opens with a short **Performance** block — Good news / Bad news about how the approach
itself does (effects vs SCIP alone, cuOpt alone and the controls, weak-set numbers, targets met or missed; "nothing new
measured" if so) — then a separate Good news / Bad news block for implementation, harness, box and bugs. A gap that
comes with a concrete remedy is not bad news; bad news is a measured loss, failure or regression.

## Reproduce
```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev,plots,gpu]"   # see README for cuOpt
pytest -q                     # toy-scale correctness tests (GPU tests skip without CuPy; -m "not integration" is faster)
```
Once runs exist: `itrea build-runs`, `itrea prereg`, then `itrea reproduce` re-derives every table in `datasets/` and
`paper/generated/` from the traces and lists any difference (skill `itrea-reproduce`). Development reports (after
`itrea build-runs`, with `OMP_NUM_THREADS=1`): `itrea backoff-report --stage 0|1`, `tools/d016_report.py`,
`tools/d017_report.py`. Re-running experiments needs the public instances in `data/` and the job files in `jobs/`;
wall-clock numbers depend on the machine (`vault/10-design/Compute resources.md`).
