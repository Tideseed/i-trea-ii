# Trace format and metrics (contract, 2026-10-05; updated 2026-10-06 to the built code)

What every run writes, and how every reported number is computed from it. Writers: `itrea.trace` (events, summary,
incumbent vectors), the EA engine (genealogy, frontier), the solver adapters (`itrea.solvers.common` and the three
`*_run` modules), the runner (`run.json`). Readers: `itrea.analysis` (`build_runs`, `metrics`, `stats`, `lineage`,
`prereg`). Pre-registered definitions are in [[D-002 Pre-registration — hypotheses, arms, analysis]] §4 and §6; this
note is the field-level contract behind them. A change to a field is a code change **and** an edit here.

The 2026-10-06 update records what the build added: the event kinds `send`, `lp` and `callback`; the verdict fields of
both solver adapters; the EA counters; `.incumbents.part` + `.names.json` for every process; the genealogy columns
`V`, `t_first`, `is_new_best`, `is_accepted_post`; the derived tables, including `runs.parquet`'s `pt_screen`.

## Clock
All times are seconds since the shared start `t0`, read from `time.monotonic()` (CLOCK_MONOTONIC is system-wide on
Linux, so every process of a run shares it and NTP steps cannot shift events). The runner releases the barrier with
`t0`; each process records how long it spent before `t0` (start-up record). **Clock rule (D-002 v3):** before the
barrier only process start, imports, instance parse, CUDA context creation and kernel warm-up on a synthetic problem;
any LP, presolve, or evaluation of the instance's own population happens after `t0`, in every arm. Epoch time is
stored next to `t0` (`t0_epoch`) for humans only. Every trajectory is truncated at the budget T in the analysis, so an
overrun earns nothing.

## Files of one run
A run = one (instance, arm, seed, budget); its directory is `traces/raw/<set>/<run_id>/` with
`run_id = <instance-stem>__<arm>__s<seed>__<budget>s__<cfg-hash>` ([[Process interface]]; a named ablation appears in
the arm as `A3~<ablation>`; the stem may contain `__`, so the id is parsed from the right). `<role>` is `solver`,
`solver2`, `ea` or `cuopt`.

| file | writer | contents |
|---|---|---|
| `<role>.events.jsonl` | every process | one JSON object per line, `{t, kind, …}`; append-only, line-buffered, crash-safe (a truncated last line is ignored by `trace.read_events`); non-finite floats are written as `null` / `"inf"` / `"-inf"` |
| `<role>.summary.json` | every process, at exit (also after an exception) | identity, start-up record, counters, final incumbent, exit reason, the incumbent list `[t, obj, source]` |
| `<role>.incumbents.npz` | every process | **every** incumbent vector (D-002: P(T) is built only from checked incumbents) — `trace.save_incumbents` |
| `<role>.incumbents.part`, `<role>.names.json` | **every process** (solvers: `SolverProcess.record_incumbent`; EA: `Engine.incumbent_log`) | crash-safe, unbuffered append-only copy of each incumbent as it is found, and the variable names it needs; removed once the `.npz` exists. Record: header `<ddBi` (t, obj, source code 0 self / 1 ea / 2 solver / 3 cuopt, nnz), then nnz `int32` indices and nnz `float64` values. A process that died keeps both; the runner turns them into the `.npz` (`solvers.common.recover_incumbents`, truncated last record dropped), so the incumbents up to the crash can be checked (D-002 §6) |
| `<role>.sol.json.gz` | every process | the final incumbent as `{name: value}` of non-zeros (the checker's input format, as in optopt) |
| `ea.genealogy.npz` | EA engine | one row per individual that entered the population (edge list, below) |
| `ea.frontier.npz` | EA engine | the frontier archive at the end: X (canonical order, float64), keys, distances |
| `<role>.log` | runner redirect | stdout / stderr |
| `order.json` | runner | the canonical variable order (HiGHS column order), also written into the mailbox |
| `config.yaml` | runner, arms with an EA | the effective EA configuration (base file, ablation, overrides, arm changes, GPU limit applied) |
| `verdicts.jsonl` | runner, from the mailbox before it is removed | the solver's verdicts as the mailbox held them (monotonic `t`); read by the analysis only when the solver wrote no `verdict` events |
| `exchange.json` | runner, Redis transport only | the stream history `{to_solver, to_ea, lp: [[t, seq, [obj…]], …]}` |
| `run.json` | runner | the run record (below) |

### Event kinds (`<role>.events.jsonl`)
Every event has `t` (seconds since `t0`) and `kind`. Fields beyond those listed are allowed; the analysis reads only the
listed ones.

| kind | who | fields |
|---|---|---|
| `incumbent` | all | `obj` (user sense), `source`: `self` (found by this process), `ea` (a solver incumbent that came from an EA post: SCIP labels the BESTSOLFOUND fired inside its own `trySol`, HiGHS a solution matching an injected row), `solver` (EA: an injected solver vector that carries no EA operator bit). Extras: SCIP `stage`; EA `key`, `id`, `gen`; cuOpt `bound`; `late: true` for a final best recorded after the solve returned |
| `bound` | solvers (SCIP, HiGHS, cuOpt) | `bound` (dual bound, user sense); written only when it changes (SCIP: at the poll timer, SOLVING stage) |
| `post` | EA | `seq`, `obj` (best row of the message), `gain` (relative to the solver incumbent the EA knew; NaN for frontier posts), `k` (rows), `frontier` (bool), `ids` (individual ids of the rows, at most 64) |
| `inject` | EA | `seq`, `k` (rows taken into the population), `rows` (rows in the message), `echo` (rows that were the EA's own posts coming back, skipped), `best_obj` |
| `send` | solvers (SCIP, HiGHS, cuOpt with `--send`) | `seq`, `k` (rows posted to the EA), `best_obj`, `echo` (rows marked `ORIGIN_EA_ECHO`: vectors the solver had received from the EA). cuOpt: always `k = 1`, `echo = 0` |
| `verdict` | solvers receiving posts (SCIP, HiGHS) | `seq`, `obj` (as posted), `accepted`, `row`, `stored`, `primal_before`, `primal_after`, `dup`, plus per solver — **SCIP:** `accepted` = SCIP's own `checkSol` found the row feasible; `stored` = `trySol` kept it (it may still not be the incumbent); `feasible`, `nonfinite`, `obj_solver` (SCIP's objective of the row), `improved` (the primal value moved). **HiGHS:** `outcome` ∈ `primal_bound` (the next callback's primal bound shows it was taken), `solution_callback` (it came back through `cbMipSolution`), `dropped` (outside HiGHS' absolute tolerance, or not taken), `not_improving` (not tried: no better than HiGHS' incumbent), `duplicate_in_flight` (not tried); `tried`; `stored` = accepted **and** tried. So a HiGHS verdict can be `accepted: true, tried: false` |
| `lp` | SCIP (`--send`) | the root LP solution was posted to the EA: `obj` (once per run; a non-optimal root LP is a `note` instead) |
| `lp` | EA | an LP point arrived: `source` (`solver` or `highs_async`), `state` (`done`, `bad_point`, `no_solution`, `failed`, `stopped`), `obj`, `immigrants` (LP-rounded individuals queued), `wall_seconds` (`highs_async`), `error` |
| `gen` | EA, every `log_every` generations and at the end | `gen`, `best_key`, `median_key`, `feas_share`, `n0`, `n1`, `n2`, `pop`, `inc`, `inc_external` (the solver's incumbent key known to the EA), `dual_key`, `rank0_candidates`, `rank0_demoted`, `rank0_over_quota`, `superopt_survivors` (manipulation check: infeasible survivors with key < inc; must be ≤ ⌊x·P⌋), `shortfall`, `children_per_s`, `gpu_mem_mb`, `gpu_pool_mb`, `feasibility_phase`, `seconds` (per phase, CUDA events on the GPU), `ls_moves`, `oom_retries`, `best_viol` (feasibility phase); with `strategy.enabled` (self-adaptive repair strategy) also `strat_pop` (survivors per strategy), `strat_best` (credited `best` events per strategy since the previous `gen` event — solver finds excluded, see `strat_credited`), `strat_children` (children per strategy since the previous `gen` event) — lists indexed like the summary's `strategy.names` |
| `best` | EA | a new population best (an incumbent; during the feasibility phase a lower total violation): `key`, `obj`, `feasible`, `rank`, `origin` (bits), `ops` (their names), `anc_rank`, `anc_ops`, `parents`, `id`, `birth`, `V`, `gen`; with `strategy.enabled` also `strat` (its repair strategy) and `strat_credited` (false for a solver solution — injected, no EA operator bit — which is counted under `best_events_solver`, not credited to `strat`) |
| `frontier` | EA, every `frontier.update_every` generations | `gen`, `size`, `best_key`, `worst_key`, `max_dist`, `mean_dist_to_inc`, `mean_pairwise`, `hv` |
| `callback` | SCIP, HiGHS (not with `--no-plugin`) | `what`: `first` (SCIP adds `stage`, `timing`), `first_solving` (**SCIP only**: the first callback in the SOLVING stage, `timing`), `summary` at the end: `calls`, `calls_by_timing`, `t_first_callback`, `t_first_solving_callback`, `max_gap_s` (longest stretch without a callback, tail included), `callbacks_per_s`, `record_s`. The "no exchange window" subgroup (first SOLVING-stage callback after T/2) is therefore defined for SCIP only |
| `note` | all | `msg` (anything else: float64 re-check failures and demotions, injected rows infeasible under the EA's test, OOM retries with the halved budget, stagnation immigrants, stop seen, plugin/callback errors, a non-optimal root LP) |

`itrea.trace`'s module docstring lists only the core kinds and fields; this table is the contract.

### Summary (`<role>.summary.json`)
Identity: `run_id`, `role`, `instance` (repository-relative), `arm`, `seed`, `budget`, `t0`, `t0_epoch`; solver
processes add `solver`, `module`, `args`, `pid`, `host`, `cpus` (affinity), `threads_env`, `mailbox`, `mailbox_kind`,
`versions`, `emphasis`, `plugin` (`active` / `inert` / `none`), `n`, `m`, `sense`; the EA adds `config` (full dict) and
`config_hash`, `backend` actually used (the analysis drops a run whose backend does not match its arm), `device`,
`pool_limit_bytes`, `P`, `offspring`, `gene_dtype`, `local_search`, `versions`, `cupy_cache` (files before prepare, at
`t0`, at the end) and `compiled_on_clock` (true if the kernel cache grew after `t0`).
Timing: `read_s`, `startup_s` and the start-up steps before `t0` (`timing`, `startup`), `barrier_wait_s`, `wall_s`
after `t0`, `exit_reason` — solvers: `budget`, `optimal`, `stop`, `sigterm`, `infeasible`, …; EA: `budget`, `stop`,
`sigterm`, `max_gens`, `stopped_before_start`, `does_not_fit`, `barrier_timeout`, `bad_arguments`, `exception` (+ `error`,
`traceback`). Resources: `maxrss_mb`, `cpu_user_s`, `cpu_sys_s`; solvers also `mem_available_gb_start/_end`.
Common: `n_incumbents` (solvers; the EA counts them in `n_incumbents` below), `final_obj` / `best_obj`, `sol_file`,
`store_errors` (a side file that could not be written; the summary itself always exists).

**Counters (EA, top level of the summary):** `generations`, `children`, `evaluations`, `evaluations_per_s`,
`ls_moves`, `ls_moves_per_s`, `ls_calls` (local-search calls: one per repair — children, initial population, immigrant
batch, LP batch — whatever the number of row scales), and the search's own counters `ls_local_minima`, `ls_finished_feasible`,
`ls_finished_stall`, `ls_finished_budget`, `n_incumbents` (new EA incumbents),
`posts`, `posts_frontier`, `false_feasible` (float64 re-check failures), `inject_msgs`, `inject_rows`, `echoes`,
`injected`, `injected_infeasible` (injected rows infeasible under the EA's own test), `injected_dup` (an injected vector
already in the population), `duplicates`, `verdicts_seen`, `solver_accepted`, `solver_rejections` (each demotes the
posted individual), `demoted`, `oom_retries` (an out-of-memory error halved a phase's memory budget and retried),
`stagnation_events`, `immigrants_admitted`, `immigrants_dropped`, `lp_immigrants`; plus `feasible_found`, `best_key`,
`final_n0/_n1/_n2`, `final_pop`, `inc_external`, `dual_key`, `lp` (state of the LP point), `peak_gpu_used_mb`,
`peak_gpu_pool_mb`, `exchange_from_solver`, `exchange_to_solver`, `genealogy_rows`, `outputs` (paths of the npz files).
With `strategy.enabled` the EA summary adds `strategy`: `names` (`<row scale>/lp|nolp`, index = the strategy number),
`row_scale`, `lp_guidance`, `final_pop`, and per strategy (lists) `children`, `best_events` and `incumbents` (new
population bests / incumbents CREDITED to the strategy), `best_events_solver` and `incumbents_solver` (the same for
solver solutions — injected rows carrying no EA operator bit, the incumbent's source "solver" — listed under the
strategy their row was given but NOT credited to it: a solver find says nothing about the strategy; so
`incumbents + incumbents_solver` summed over strategies = `n_incumbents`), `repaired` (rows given to the local search),
`repaired_feasible` (rows the search finished feasible; always 0 under `repair.method: none`, which reports no per-row
status), `immigrants` (incl. the LP-rounded batch); plus `strategy_redrawn` (children whose strategy was redrawn) and
`strategy_reassigned` (children moved by the `min_share` floor). The start-up record (`startup`) then lists
`strategies`, `local_search_row_scales` (the search's row-scale table, in strategy order; `[null]` under `none`) and
`local_search_static_bytes`.
**Counters (solver, `exchange`):** `seconds` (perf-counter time in exchange code on the solver's core), `bytes_in`,
`bytes_out` (logical float64 payload; the Redis transport may move fewer bytes), `polls`, `msgs_in`, `seq_superseded`
(posts replaced before this process polled), `rows_in`, `rows_tried`, `rows_dup`, `rows_nonfinite`, SCIP
`rows_stored` / `rows_feasible_not_stored` / `rows_rejected`, HiGHS `rows_accepted` / `rows_dropped` /
`rows_not_improving`, `posts_out`, `rows_out`, `echo_rows_out`, `info_posts`, `lp_posts`, `plugin_errors`. Solver
statistics: `status`, `wall_s`, `overrun_s`, `final_dual`, `nodes`, `lp_iterations`, `nodes_per_s`,
`lp_iterations_per_s`, `presolve_s`, `n_sols_found`, `n_best_sols_found`, `callbacks` (the `callback` summary),
`received_vectors`, `posted_keys`, `root_lp_posted`. cuOpt adds `rmm_limit_bytes`, `rmm_peak_bytes`,
`rmm_total_bytes`, `rmm_current_bytes`, `rmm_peak_count`.

### Incumbent vectors (`<role>.incumbents.npz`)
`t`, `obj` (user sense), `source` (unicode), `names` (the process's own variable order, unicode), and the vectors in CSR
form (`ptr`, `idx`, `val` of the non-zeros). No pickle. Written atomically. The checker maps names onto the instance it
re-reads with a different library.

### Genealogy edge list (`ea.genealogy.npz`)
One row per individual that entered the population (survivors of each generation that were not already members, plus
seeds, immigrants and injected vectors that entered):
`id` int64, `pa`, `pb` int64 (−1 = none; mutation-only children have `pa = pb`), `rank_pa`, `rank_pb` int8 (the
parents' ranks **at mating**; −1 for seeds, immigrants and injected vectors), `origin` uint16 (`ORIGIN_*` bits, OR-merged
over exact duplicates, also after entry), `birth` int32 (generation), `key` float64, `V` float64 (total violation at
entry), `feasible` bool (at entry), `t_first` float64 (seconds since `t0` when the vector was made or received),
`is_new_best` bool (it became a new population best), `is_accepted_post` bool (the solver accepted it when posted),
`strat` uint8 (its repair strategy, an index into the summary's `strategy.names`; 0 while `strategy.enabled` is false —
added 2026-10-06, older genealogy files lack the column). Because `origin` is OR-merged over exact duplicates, a row
whose strategy has no LP guidance can carry `ORIGIN_M_TOWARD_LP` (a duplicate made by an LP-guided child; no no-LP child
ever makes that bit itself): an analysis of LP guidance must key on `strat` (or the `best` event's `strat`), never on
the origin bits.
The two flags are written explicitly so the lineage analysis does not derive them. About 40 bytes per row, well under
50 MB for a 300 s run. The OR-accumulated ancestry bit masks in `ea.population` are kept only as an online summary: they
saturate within a few generations (critique, algorithm lens).

### Run record (`run.json`)
Identity: `run_id`, `arm` (with `~<ablation>`), `base_arm`, `ablation`, `arm_doc`, `gpu_arm`, `lane_kind`, `job`,
`set`, `instance`, `name`, `seed`, `budget`, `grace_s`, `block`, `position` (Williams position of the arm in its block),
`window`, `stratum`, `split`; `config` {`path`, `overrides`, `base_hash` (the file alone — the freeze check compares
it with `frozen-e2`), `ea_hash` (the effective configuration), `run_hash`, `solver_settings` (arms without an EA)};
`problem` (shape, `load_s`); `mailbox` {kind, spec}; `lane`, `systemd`, `pinned`; `memory_budget` (the D-003 gate
terms and `need_gb`); `env` (versions, hardware, git state: `itrea.experiments.record_env`).
Processes: one record per process — `role`, `kind`, `backend`, `cmd`, `cores`, `memory_max_gb`, `gpu_limit_gb` (CuPy
pool or RMM limit), `env`, `pid`, `startup_s`, `exit_code`, `killed`, `exit_t`, `cpu_user_s`, `cpu_sys_s`,
`maxrss_mb`; `exit_codes` {role: code} — **every one is checked** (optopt X1 logged "ok" for dead SCIP processes).
Gates and health: `gates_before` / `gates_after` (MemAvailable, the hottest ACPI zone, the box-window lease, the local
model servers' state, other large processes; `gates_before` also the GPU compute apps), `gpu_health` (per-second samples summarised: median busy SM clock, foreign GPU share, `verdict`,
`reasons`), `cupy_cache` (`compiled_on_clock`).
Outcome: `status` (`completed` = every process exited 0; `failed` = any exit code ≠ 0, a kill or a start failure —
kept, truncated at the crash; `void` = an external cause: GPU health, thermal pause, memory floor, abort, runner error —
the whole block is rerun), `void_cause`, `reasons`, `stop`, `wall_s`, `exchange` (verdict counts; Redis stream
lengths), and the merged trajectories as reported by the processes: `portfolio` (best of all processes at every
moment, `[t, obj, role, source]`) and `solver` (what the solver itself held). The analysis rebuilds both from checked
incumbents.

## Derived tables (`itrea.analysis.build_runs` → `datasets/`)
`python -m itrea.analysis.build_runs [--raw traces/raw] [--out datasets] [--recheck]` (or `itrea build-runs`).

**`datasets/incumbents_checked.parquet`** — one row per stored incumbent of every process: `run_id`, `role`, `k`, `t`,
`source`, `obj_reported`, `obj_recomputed`, `checked`, `ok`, `feasible`, `obj_ok`, `reason`, the worst bound /
integrality / row violations (`worst_bound`, `worst_int`, `worst_row_rel`, `worst_row_scip`), `obj_rel_err`,
`unknown_vars`, `missing_vars`, `exact_rows`, `n_viol_rows`, `producer`, `reader`, `sense`, `npz_sha256`, `beats_ref`.
It caches the checker's results by (run_id, role, sha256 of the npz), so a rebuild needs neither the instances nor the
time unless `--recheck`.

**`datasets/runs.parquet`** — one row per run (fixed columns and dtypes, so a rebuild is identical by content):
- identity: `run_id`, `set`, `instance`, `instance_path`, `arm`, `base_arm`, `seed`, `budget`, `config_hash`,
  `base_hash`, `config_overrides`, `synthetic`, `status`, `void_cause`, `split`, `stratum`, `attempt`, `block`,
  `window_id`, `williams_pos`, `git_rev`, `prereg_tag`, `run_dir`;
- reference: `sense`, `ref`, `ref_kind`, `ref_frozen`, `ref_zero_instance`, `ref_bos`, `sense_conflict`;
- portfolio (primary) and sensitivity: `pt` (P(T) of the portfolio, checked incumbents only), `pt300`, `pt_bos`
  (against the best checked value of the whole study — D-002's sensitivity analysis), **`ref_screen`, `pt_screen`** (P
  against the **screening reference only**: the `.solu` value, else the best SCREEN run — the pool freeze ranks
  instances on `pt_screen`, so the pool does not move when `reference.csv` is frozen later; D-002 orders the pool
  before the reference), `ref_zero_screen`, `sign_crossing_screen`, `ref_zero`, `sign_crossing`, `beats_ref`,
  `beats_ref_t`, `t_first`, `t_gap5`, `t_gap1` (each with an `_event` flag: false = censored at T), `final_gap`,
  `best_obj`, `n_steps`, `pdi`;
- per role (`solver_*`, `ea_*`, `cuopt_*`): the same trajectory metrics for what that process held (`solver_pt` =
  SCIP-held P, `ea_pt` = the EA's own P for H1b and H6);
- integrity: `n_inc`, `n_inc_ok`, `n_inc_failed`, `n_inc_unchecked`, `n_obj_mismatch`, `incumbents_missing`,
  `inc_count_mismatch`; post accounting `posts`, `posts_accepted`, `posts_rejected`, `posts_superseded`,
  `posts_pending`, `posts_stored`, `posts_became_incumbent`, `posts_after_T`, `verdicts_unmatched`, `verdict_source`,
  `solver_inc_from_ea`; `crashed`, `crashed_roles`, `crash_t`, `exit_codes`; `gpu_arm`, `gpu_health`,
  `gpu_median_sm_mhz`, `gpu_foreign_share`, `gpu_void`, `block_void` (every run of a voided block);
- resources and covariates: `<role>_startup_s`, `startup_s`, `exchange_s`, `exchange_bytes`, `solver_nodes_per_s`,
  `solver_lp_iter_per_s`, `solver_status`, `solver_exit_reason`, `solved_optimal`, `backend`, `backend_mismatch`;
- frontier and exchange window: `frontier_size`, `frontier_hv`, `frontier_mean_pairwise`, `frontier_max_dist`,
  `frontier_mean_dist_to_inc`, `cb_first_s`, `cb_first_solving_s`, `cb_max_gap_s`, `no_exchange_window`.
- **synthetic rows** (`synthetic = True`, arm `A0+A5`): the uncoupled portfolio of D-002 v3 H2 — the pointwise best of
  the checked A0 and A5 trajectories of the same block.

## Metrics
All in the user's objective sense; computed by `itrea.analysis` (adapted from berkorbay/optopt@4dd48395, MIT,
`src/optopt/analysis/metrics.py`).

**Checked trajectory.** Only incumbents that pass the independent exact checker (`itrea.analysis.check_solutions`:
another library re-reads the model, bounds/integrality/objective in exact rational arithmetic, every row in floating
point with an exact re-check near the tolerance, 1e-6) enter any metric, with the **recomputed** objective. A failed
incumbent is dropped from the trajectory and counted.

**Primal gap** (Berthold 2013) of an incumbent value v against the reference r:
γ = 1 if there is no incumbent or v and r differ in sign; γ = 0 if v = r (including both 0);
γ = |v − r| / max(|v|, |r|) otherwise. A checked incumbent better than r gets γ = 0 and a `beats reference` record.

**Primal integral** P(T) = (1/T) ∫₀ᵀ γ(t) dt ∈ [0, 1], γ(t) from the best checked incumbent held at t; T = 300 s
(1,800 s on the long-budget subset). Integrated on exact event times (optopt's 2026-09-24 PDI correction: never round
times before the lookup).
- **Portfolio P** (primary): γ(t) of the best incumbent held by any process of the run at t.
- **SCIP-held P** (secondary): γ(t) of SCIP's own incumbent, including solutions it accepted from the EA.
- **EA P** (H1b, H6): the EA process's own trajectory.
- **Uncoupled portfolio** (H2): pointwise best of the A0 and A5 trajectories of the same block, computed from those
  runs (optimistic for the uncoupled side, because the two did not share the machine); the live arm A3r checks it on a
  subset.

**Reference r** (frozen in `datasets/reference.csv` before E2; resolved by `itrea.benchmarks.references`): frozen
reference → MIPLIB `.solu` → provisional best of the SCREEN / REF runs → best of study, with `ref_kind` recorded. The
dedicated reference runs are SCIP 1 h × 2 seeds. **P-degenerate** instances (r = 0, or a sign crossing during the run)
are flagged in the pool file and analysed on time to target only.

**MIPfeas-style integral** (secondary, D-002 §4; Bussieck and Dirkse 2026): the same relative gap but 2 with no
incumbent and the denominator max(|z(t)|, |z*|, 1), aggregated by the shifted geometric mean (s = 0.001). Reported
beside Berthold's P(T) because a published GB10 cuOpt baseline exists under it (0.0393 at 600 s on the 233 MIPLIB
benchmark instances).

**Time to target**: first time γ ≤ 5 %, γ ≤ 1 %, and time to first feasible; censored at T (Type I). Analysed per
instance on seed medians.

**Per-instance effect** (seeds averaged): **d_i = (P_ctrl − P_arm) / max(P_ctrl, P_arm, 10⁻³)** ∈ [−1, 1]; positive
means the arm is better. Wilcoxon signed-rank on d_i (`zero_method="wilcox"`), mean d_i with a 95 % percentile
bootstrap interval (10,000 resamples over instances); ratio of means as a secondary. Decision rule and margin: D-002 §6.

**Post accounting** (per run): every EA post up to T is tried-accepted / tried-rejected / superseded-untried (a newer
post replaced it before the solver polled: a gap in the solver's `seq`) / pending-at-T, from the EA's `post` events and
the solver's `verdict` events (or `verdicts.jsonl` when the solver wrote none). Tried-accepted reads `accepted`;
`posts_stored` reads `stored`; a post of several rows counts as accepted if any row is. For HiGHS, a verdict with
`tried: false` (`not_improving`, `duplicate_in_flight`) is not a tried post, although its `accepted` may be true. Solver
incumbents that originated in the EA = solver `incumbent` events with `source = ea` (`solver_inc_from_ea`).

**Mechanism (descriptive):** depth-1 matched comparison on the genealogy — odds of a Rank 0 parent among {new population
bests (`is_new_best`), accepted posts (`is_accepted_post`)} vs among all children born in the same generation,
Mantel–Haenszel over generations and instances (`itrea.analysis.lineage`). Descriptive only; the causal claim about
Rank 0 rests on H6 (x = 10 % vs 0).

**Frontier** ([[D-005 The frontier of diverse solutions]], exploratory): size; spread = mean pairwise distance
(normalised Hamming over integer variables, or range-normalised L1); hypervolume in (relative gap to the incumbent,
−distance) with reference point (band, 0).

**Resources** (covariates and integrity): SCIP nodes/s and LP iterations/s, cgroup `cpu.stat`, NVML GPU-seconds,
peak GPU pool, solver-side exchange seconds and bytes, start-up seconds per arm (and a sensitivity analysis with
start-up charged to the clock).
