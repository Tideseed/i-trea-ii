# Trace format and metrics (contract, 2026-10-05; updated 2026-10-06 to the built code)

What every run writes, and how every reported number is computed from it. Writers: `itrea.trace` (events, summary,
incumbent vectors), the EA engine (genealogy, frontier), the solver adapters (`itrea.solvers.common` and the three
`*_run` modules), the runner (`run.json`). Readers: `itrea.analysis` (`build_runs`, `metrics`, `stats`, `lineage`,
`prereg`). Pre-registered definitions are in [[D-002 Pre-registration — hypotheses, arms, analysis]] §4 and §6; this
note is the field-level contract behind them. A change to a field is a code change **and** an edit here.

The 2026-10-06 update records what the build added: the event kinds `send`, `lp` and `callback`; the verdict fields of
both solver adapters; the EA counters; `.incumbents.part` + `.names.json` for every process; the genealogy columns
`V`, `t_first`, `is_new_best`, `is_accepted_post`; the derived tables, including `runs.parquet`'s `pt_screen`.
2026-10-07: the three-way arm A9 (SCIP ⇄ EA ← cuOpt, [[D-015 Use cuOpt's feasible solutions — three-way arm A9]],
`exchange.from_cuopt`): the incumbent source `cuopt`, the EA event kind `inject_cuopt`, the `cuopt*` fields of `post`,
`best` and `gen` events, the EA's `cuopt_*` counters, SCIP's verdict note `cuopt` and counter `rows_in_cuopt`, the
genealogy column `cuopt`, and `runs.parquet`'s cuOpt attribution columns — all only in runs whose EA reads cuOpt.

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
| `exchange.json` | runner, Redis transport only | the stream history `{to_solver, to_ea, lp, lps: [[t, seq, [obj…]], …]}`, plus `to_ea_cuopt` when cuOpt posted to the EA (A9) |
| `run.json` | runner | the run record (below) |

### Event kinds (`<role>.events.jsonl`)
Every event has `t` (seconds since `t0`) and `kind`. Fields beyond those listed are allowed; the analysis reads only the
listed ones.

| kind | who | fields |
|---|---|---|
| `incumbent` | all | `obj` (user sense), `source`: `self` (found by this process), `ea` (a solver incumbent that came from an EA post: SCIP labels the BESTSOLFOUND fired inside its own `trySol`, HiGHS a solution matching an injected row), `solver` (EA: an injected solver vector that carries no EA operator bit), `cuopt` (A9 — EA: an injected cuOpt vector that carries no EA operator bit; SCIP: an incumbent made from an EA post row marked `POST_CUOPT`, i.e. an unchanged cuOpt vector the EA passed on). Extras: SCIP `stage`; EA `key`, `id`, `gen`; cuOpt `bound`; `late: true` for a final best recorded after the solve returned |
| `bound` | solvers (SCIP, HiGHS, cuOpt) | `bound` (dual bound, user sense); written only when it changes (SCIP: at the poll timer, SOLVING stage) |
| `post` | EA | `seq`, `obj` (best row of the message), `gain` (relative to the solver incumbent the EA knew — the solver's own, never cuOpt's; NaN for frontier posts), `k` (rows), `frontier` (bool), `ids` (individual ids of the rows, at most 64); A9 also `cuopt_rows` (rows that are unchanged cuOpt vectors, sent with `POST_CUOPT`; frontier rows are marked by entry) |
| `inject` | EA | `seq`, `k` (rows taken into the population), `rows` (rows in the message), `echo` (rows that were the EA's own posts coming back, skipped), `best_obj` |
| `inject_cuopt` | EA (A9) | the same fields for a message read from cuOpt's channel `to_ea_cuopt` (`echo` always 0: cuOpt never receives a vector) |
| `send` | solvers (SCIP, HiGHS, cuOpt with `--send`) | `seq`, `k` (rows posted to the EA), `best_obj`, `echo` (rows marked `ORIGIN_EA_ECHO`: vectors the solver had received from the EA). cuOpt: always `k = 1`, `echo = 0`, on its own channel `to_ea_cuopt` (A9) |
| `verdict` | solvers receiving posts (SCIP, HiGHS) | `seq`, `obj` (as posted), `accepted`, `row`, `stored`, `primal_before`, `primal_after`, `dup`, plus per solver — **SCIP:** `accepted` = SCIP's own `checkSol` found the row feasible; `stored` = `trySol` kept it (it may still not be the incumbent); `feasible`, `nonfinite`, `obj_solver` (SCIP's objective of the row), `improved` (the primal value moved), `cuopt` (true for a row marked `POST_CUOPT`, A9). **HiGHS:** `outcome` ∈ `primal_bound` (the next callback's primal bound shows it was taken), `solution_callback` (it came back through `cbMipSolution`), `dropped` (outside HiGHS' absolute tolerance, or not taken), `not_improving` (not tried: no better than HiGHS' incumbent), `duplicate_in_flight` (not tried); `tried`; `stored` = accepted **and** tried. So a HiGHS verdict can be `accepted: true, tried: false` |
| `lp` | SCIP (`--send`) | the root LP solution was posted to the EA: `obj` (once per run; a non-optimal root LP is a `note` instead) |
| `lps` | SCIP (`--lp-stream`, D-011) | a node LP point was read and posted: `seq`, `id` (point id, increasing), `depth`, `nodes`, `nfrac` (fractional integer variables), `obj` (LP objective, original space), `why` (`early` — one of the first 8 LP solves; `quality` — `nfrac` within the configured quantile; `patience`), `read_s`, `s` (read + post) |
| `lps` | EA (`lp_stream.enabled`) | stream points taken: `seq`, `ids` (new point ids), `V`, `key` (of each point's nearest rounding), `feasible` (how many of those roundings were feasible), `pool` (pool size after) |
| `lp` | EA | an LP point arrived: `source` (`solver` or `highs_async`; an LP-stream point never becomes the LP point), `state` (`done`, `bad_point`, `no_solution`, `failed`, `stopped`), `obj`, `immigrants` (LP-rounded individuals queued), `wall_seconds` (`highs_async`), `error` |
| `gen` | EA, every `log_every` generations and at the end | `gen`, `best_key`, `median_key`, `feas_share`, `n0`, `n1`, `n2`, `pop`, `inc`, `inc_external` (the external incumbent key known to the EA: the solver's; in A9 the better of the solver's and cuOpt's), `dual_key` (the solver's proven bound), `rank0_candidates`, `rank0_demoted`, `rank0_over_quota`, `superopt_survivors` (manipulation check: infeasible survivors with key < inc; must be ≤ ⌊x·P⌋), `shortfall`, `children_per_s`, `gpu_mem_mb`, `gpu_pool_mb`, `feasibility_phase`, `seconds` (per phase, CUDA events on the GPU), `ls_moves`, `oom_retries`, `best_viol` (feasibility phase); with `strategy.enabled` (self-adaptive repair strategy) also `strat_pop` (survivors per strategy), `strat_best` (credited `best` events per strategy since the previous `gen` event — solver finds excluded, see `strat_credited`), `strat_children` (children per strategy since the previous `gen` event) — lists indexed like the summary's `strategy.names`; with `lp_stream.enabled` also `rank0_stream` (Rank 0 members born from the stream: origin SEED_LP and INJECTED), `lps_pool`, `lps_unused`; with `repair_clock.enabled` also `rc_mode`, `rc_phase`, `rc_level` (the level played in the last generation: lane steps or `"none"`), `rc_would` (the controller's choice for it; = `rc_level` in active mode), `rc_next` (its choice for the next generation), `rc_probe` (probe generations since the previous `gen` event), `rc_H_eff` ([min, max] of the repair calls' H_eff since then), `rc_cap_hits` (calls the played cap cut since then), `rc_switches` (cumulative); with `exchange.from_cuopt` (A9) also `inc_solver`, `inc_cuopt` (the two incumbent keys behind `inc_external`), `dual_key_cuopt` (cuOpt's bound — recorded, not used), `cuopt_pop` (members that entered as cuOpt rows) |
| `best` | EA | a new population best (an incumbent; during the feasibility phase a lower total violation): `key`, `obj`, `feasible`, `rank`, `origin` (bits), `ops` (their names), `anc_rank`, `anc_ops`, `parents`, `id`, `birth`, `V`, `gen`; with `strategy.enabled` also `strat` (its repair strategy) and `strat_credited` (false for a solver solution — injected, no EA operator bit — which is counted under `best_events_solver`, not credited to `strat`; the same for a cuOpt solution); with `exchange.from_cuopt` (A9) also `cuopt` (the individual entered as a cuOpt row) |
| `frontier` | EA, every `frontier.update_every` generations | `gen`, `size`, `best_key`, `worst_key`, `max_dist`, `mean_dist_to_inc`, `mean_pairwise`, `hv` |
| `rclock` | EA (`repair_clock.enabled`), one per generation (generation 0 = the initial population in `begin`) | the repair clock ([[I-TREA II algorithm]], "Repair clock"): `gen`, `t_dec` (the decision time, seconds since t0), `phase` (`pre`: the population has held no feasible member of the EA's own yet — an unchanged solver solution does not count; every call is played at "none" without a cap and nothing is accumulated; `post`), `mode`, `level` (played: lane steps or `"none"`), `cap` (the played per-call deadline in seconds; null = none), `would`, `would_cap` (the controller's choice; equal to the played one in active mode), `probe`, `would_cap_hit` (a call's t(would level) exceeded `would_cap`), `next` (the level chosen for the next generation), `switched`, `gen_s` (measured generation seconds), `o_s` (seconds outside the counterfactual repair calls); per level, in the order of the start-up record's `repair_clock.levels` (2^7 … 2^16, `"none"`): `gain` (Δ_g(h) = the largest dg over the lanes of the counterfactual calls that ended FEASIBLE with steps ≤ h and are feasible under the evaluation, dg = min((cutoff − key) / max(\|cutoff\|, \|key\|, \|dual_key\| if finite, 1e-9), 1) ∈ [0, 1]; below `materiality` → 0; null = not observed — a counterfactual call of the generation was cut below h), `T` (T_g(h) = `o_s` + Σ t_call(h) over the counterfactual calls), `used` (per level: accumulated by the controller — the observed levels in active mode; in shadow mode only those its decision would have revealed: the levels ≤ `would`, or all if no lane of a call ran beyond it), `gain_approx` (the largest dg of the non-counterfactual calls, not in `gain`; null = none), `t_children` (the children call's t(h)), `rate` (decayed G / T after this generation), `n`, `events` (accumulated post-phase generations, and those with a material gain); `best` ({`site`, `steps`, `dg`} of the largest-gain lane counted in `gain`); `calls`, one per repair call: `site` (`init`, `children`, `lp`, `lps`, `admit`), `k` (lanes), `pinned` (the one-shot LP batch, `pin_lp`: its gain and time count for every level), `cf` (counterfactual: false for the immigrant calls after the survival — shortfall fill and stagnation — which depend on the generation's earlier calls; left out of `gain` and of what is observed, their seconds count in `o_s`), `H`, `cap` (played), `cap_hits`, `H_eff` (the largest lane step count — the cut point when `cut`), `cut` (some lane has status 4), `s` (seconds), `t` (t_call(h) per level; null = not observed), `cutoff`, `dual_key` (the keys dg was computed from; null cutoff = none armed), `steps` (per lane), `status` (per lane, one digit each: 0 interrupted, 1 feasible, 2 stall, 3 budget, 4 horizon), `lane_gains` ([steps, dg, status] of every lane that beats the cutoff and is feasible under the evaluation, whatever its status but 0, largest dg first, at most 64), `beat_stopped`, `beat_cut` ([lanes, best dg] of those lanes that did NOT end FEASIBLE: stopped by stall / budget, or cut — not counted in `gain`), `prof` (GPU: `chunks` with `cum` / `t` — cumulative steps and seconds since the call started, after every launch, from (0, chunk start) — `end`, `H_eff`, `done`, `cut`, `cap_hit`; CPU: `seconds`, `H`, `cap_s`, `cut_rows`) |
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
With `repair_clock.enabled` the EA summary adds the counters `ls_finished_horizon` (lanes cut by the horizon or the
call deadline — status 4; they are counted in `ls_finished_budget` too, so the move-cap stops are `ls_finished_budget −
ls_interrupted − ls_finished_horizon`) and `ls_cap_hits` (GPU chunks / CPU calls the deadline cut), and
`repair_clock`: the settings (`mode`, `levels`, `min_gens`, `min_events`, `materiality`, `half_life_s`, `hysteresis`,
`probe_share`, `cap_s`, `cap_factor`, `pin_lp`), `generations_pre`, `generations_post`, `t_post` (when the post phase
began, seconds since t0), `post_s`, `time_share_played` and `time_share_would` (per level, share of the post-phase
wall time), `probe_s`, `probe_time_share`, `switches`, `cap_hits` (calls the played cap cut), `cap_hits_would`,
`H_eff_calls` (repair calls counted by the smallest level ≥ their H_eff; the last entry = above the ladder), `final_level`
and the final decayed `rate`, `G`, `T`, `n`, `events` per level. The start-up record (`startup`) then holds
`repair_clock` (the settings and `levels`).
With `exchange.from_cuopt` (A9) the EA summary adds the counters `cuopt_inject_msgs`, `cuopt_inject_rows`,
`cuopt_injected` (rows queued from cuOpt's messages), `cuopt_injected_infeasible` (infeasible under the EA's own test),
`cuopt_injected_dup` (a cuOpt vector already in the population; SCIP's `injected_dup` then counts SCIP's rows only),
`cuopt_info_msgs`, `cuopt_incumbents` (EA incumbents that are unchanged cuOpt vectors, source `cuopt`; they are in
`n_incumbents` too and, with the strategy on, in `incumbents_solver`), `cuopt_posts` (posts carrying such a vector),
`cuopt_rows_accepted` / `cuopt_rows_rejected` (the solver's verdicts on those ROWS — the unit of `solver_accepted`: a
frontier post of several cuOpt rows or a re-posted vector gives several; `runs.parquet` counts per post), and
`exchange_from_cuopt`, `inc_solver`, `inc_cuopt`, `dual_key_cuopt`, `final_cuopt_pop`. SCIP's `exchange` adds
`rows_in_cuopt` (received rows marked `POST_CUOPT`).
With `lp_stream.enabled` the EA summary adds the counters `lps_msgs`, `lps_points`, `lps_used`, `lps_rows`,
`lps_dropped` (unused points pushed out of the pool), `lps_missed` (point ids skipped between two polls) and
`lp_stream`: `pool`, `unused`, `best` ({V, key} of the pool's
best point), `final_rank0_stream`, `final_stream` (members of the final population born from the stream).
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
`received_vectors`, `posted_keys`, `root_lp_posted`, `lp_stream` (with `--lp-stream`, D-011: `lp_solves` seen,
`cutoff` skipped (LP value ≥ the cutoff bound), `integral` skipped, `reads` = `early` + `quality` + `patience`, `bad`,
`read_s`, `post_s`, `spent_s` (the stream's whole cost: every decision as well as the reads and posts), `points`; else
null). cuOpt adds `rmm_limit_bytes`, `rmm_peak_bytes`,
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
the origin bits. With `exchange.from_cuopt` (A9) also `cuopt` bool (the row entered as an injected cuOpt vector: its
id was given to a cuOpt row; such a row's origin is ORIGIN_INJECTED like a solver row's, so only this column tells them
apart — added 2026-10-07, only in A9 genealogies).
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
  `solver_inc_from_ea`; A9 (−1 for runs whose EA does not read cuOpt — decided from the EA's own record: its
  `exchange_from_cuopt`, else its configuration's `exchange.enabled` and `from_cuopt`): `cuopt_inject_msgs`,
  `cuopt_injected`, `ea_inc_from_cuopt` (checked EA incumbent records with source `cuopt`), `cuopt_posts`,
  `cuopt_posts_accepted`, `cuopt_posts_became_incumbent` (per post, below), `solver_inc_from_cuopt` (checked SCIP
  incumbent records with source `cuopt`), all up to T;
  `crashed`, `crashed_roles`, `crash_t`, `exit_codes`; `gpu_arm`, `gpu_health`,
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
incumbents that originated in the EA = solver `incumbent` events with `source = ea` (`solver_inc_from_ea`); in A9 an
unchanged cuOpt vector the EA passed on is `source = cuopt` instead (`solver_inc_from_cuopt`), so it is credited to
cuOpt, not to the EA. Both count incumbent RECORDS, not posts: one SCIP `trySol` can fire BESTSOLFOUND twice (the
second on SCIP's slightly changed copy, objective within ~1e-12 relative), and both records carry the injected label
(sing44 dev runs: every accepted post gave two; over the A3 runs `solver_inc_from_ea` sums to 13,466 against 13,376
`posts_became_incumbent`). For "how many posts SCIP took" use the per-post columns. In A9 the post columns count every
EA post, cuOpt's unchanged vectors included; `cuopt_posts`, `cuopt_posts_accepted` and `cuopt_posts_became_incumbent`
are their cuOpt share under the same rules (one per post; accepted if any cuOpt row's verdict is, a `dup` verdict
included; became the incumbent if that row's `primal_after` equals its posted objective within 1e-9 relative — on the
row, not on the post's best objective, which differs only for a several-row frontier post), so the EA-made share is
`posts_*` minus `cuopt_posts*`. A process the runner killed after a partner's proven outcome
(`processes[*].stopped_by_runner`, cuOpt in A2 / A9) is not counted in `crashed` / `crashed_roles`.

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
