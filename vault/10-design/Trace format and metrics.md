# Trace format and metrics (contract, 2026-10-05; updated 2026-10-06 and 2026-10-10 to the built code)

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
2026-10-08: Rank 1 by cuOpt's clearing rule ([[D-016 Rank 1 as a distinct elite — cuOpt's clearing rule, development
test]], `ranks.rank1_mode: clearing`; not adopted 2026-10-09): the `r1_*` fields of `gen` events and the summary's
`final_r1_*` — in every run of the code of 2026-10-08 or later, in both Rank 1 modes —, `r1_cleared` and the counter
`rank1_cleared` in clearing runs only, and the EA start-up record's `rank1` and `warmup.*.elite_warmed`.
2026-10-09: the LP decode ([[D-017 LP decode — re-optimise the continuous variables of selected candidates]],
`repair.continuous: lp_decode`, with the fixes of [[2026-10-08 lp_decode review (workflow)]]): the EA event kind
`lp_decode`, the `lpd_*` fields of `gen` events, the `lpd_*` counters and the summary block `lp_decode`, the start-up
record's `lp_decode` and `warmup.*.lp_decode_warmed`, the genealogy column `lpd_src`, and `runs.parquet`'s `lpd_*`
columns — all only in runs whose configuration says `lp_decode` (and, but for the start-up record and `lpd_state`
"inert", only where the decode can act: an instance with an active column). Second round of fixes, 2026-10-09 (before
any run): `lpd_skip_pending` (a pick site that read a reply with copies offers nothing), `lpd_dropped` and the
`lp_decode` event `dropped`, the per-row `cold` of result events, the funnel counters `lpd_cached_F` / `_I`,
`lpd_screened_I`, `lpd_srcs_F` / `_I`, and the gains of the copies that entered (`gain_F_entered_median` / `_n`,
`gain_I_entered_median` / `_n`).
2026-10-09: A9's GPU back-off ([[D-018 A9 — the EA backs off the GPU beside cuOpt (contribution-gated duty cycle)]],
`backoff.enabled` and `exchange.from_cuopt`; `itrea.ea.backoff`): the EA event kinds `backoff` and `contrib`, the
`bo_*` and `ls_steps` fields of `gen` events, `bo_wall_s` / `bo_paused_s` in `rclock` events, the phase `paused` in
`gen` events' `seconds`, the summary's `backoff` block and start-up `backoff`, and `runs.parquet`'s `bo_*` columns — all
only in runs whose EA runs the governor. With it the repair clock's `t_dec`, `gen_s`, `o_s` and its calls' seconds are
BUSY seconds (wall minus the governor's sleeps).
2026-10-10 (documentation pass, no code change): the back-off's `t_due`, `dup_lead` and `n_assist_dup`, the summary's
`cupy_cache.new_after_t0`, and the EA start-up record's `warmup` block, which this note had not listed.
2026-10-11: the MIPfeas harness ([[D-020 MIPfeas harness — GAMS's clock and integral, paired calibration of cuOpt
26.08]]): the call and launch clocks, GAMS's integral exactly (`metrics.gams_integral`) and
`runs.parquet`'s `pt_gams`, `pt_gams_launch` and `startup_offset_s`; the cuOpt summary's `settings` and `mip_gap_zero`;
`run.json`'s `runner_cores`.

## Clock
All times are seconds since the shared start `t0`, read from `time.monotonic()` (CLOCK_MONOTONIC is system-wide on
Linux, so every process of a run shares it and NTP steps cannot shift events). The runner releases the barrier with
`t0`; each process records how long it spent before `t0` (start-up record). **Clock rule (D-002 v3):** before the
barrier only process start, imports, instance parse, CUDA context creation and kernel warm-up on a synthetic problem;
any LP, presolve, or evaluation of the instance's own population happens after `t0`, in every arm. Epoch time is
stored next to `t0` (`t0_epoch`), for humans and for the launch clock below. Every trajectory is truncated at the
budget T in the analysis, so an overrun earns nothing.

**Two clocks for MIPfeas (D-020).** The **call clock** is the one above: seconds since `t0`, after the barrier (parse,
CUDA context, RMM set-up and warm-up before it). It is the clock comparable with GAMS's MIPfeas traces, whose clock
starts at the GAMS↔solver link after model generation (GB10 cuOpt 26.08: glass4's first incumbent at 0.119 s), and
D-020's calibration passes or fails on it. The **launch clock** starts at the first process's launch: every time plus
`startup_offset_s` = run.json `t0_epoch` − min(`processes[*].launch_epoch`), so the parse and our warm-up count; it is
the strict clock for comparing our own arms (A9's start-up is longer than cuOpt's). Both are in `runs.parquet`
(`pt_gams`, `pt_gams_launch`); the stated handicaps either way are in D-020.

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
| `gen` | EA, every `log_every` generations and at the end | `gen`, `best_key`, `median_key`, `feas_share`, `n0`, `n1`, `n2`, `pop`, `inc`, `inc_external` (the external incumbent key known to the EA: the solver's; in A9 the better of the solver's and cuOpt's), `dual_key` (the solver's proven bound), `rank0_candidates`, `rank0_demoted`, `rank0_over_quota`, `superopt_survivors` (manipulation check: infeasible survivors with key < inc; must be ≤ ⌊x·P⌋), `shortfall`, `children_per_s`, `gpu_mem_mb`, `gpu_pool_mb`, `feasibility_phase`, `seconds` (per phase, CUDA events on the GPU), `ls_moves`, `oom_retries`, `best_viol` (feasibility phase), and Rank 1's size and spread in both Rank 1 modes (D-016; `elite.elite_spread`): `r1_n` (members), `r1_dmax` / `r1_dmed` (the max / median over the other members of the share of integer variables in which a member differs from Rank 1's best), `r1_gap` ((worst − best key) / max(1, \|best\|)), `r1_dmax_band` (`r1_dmax` over the members within `frontier.band` of the best; 0.0 if no other member is in the band) — all but `r1_n` null with fewer than two members, the distances also without integer variables; with `ranks.rank1_mode: clearing` also `r1_cleared` (candidates the distance rule skipped in the last survival's final round, before the walk reached the cap); with `strategy.enabled` (self-adaptive repair strategy) also `strat_pop` (survivors per strategy), `strat_best` (credited `best` events per strategy since the previous `gen` event — solver finds excluded, see `strat_credited`), `strat_children` (children per strategy since the previous `gen` event) — lists indexed like the summary's `strategy.names`; with `lp_stream.enabled` also `rank0_stream` (Rank 0 members born from the stream: origin SEED_LP and INJECTED), `lps_pool`, `lps_unused`; with `repair_clock.enabled` also `rc_mode`, `rc_phase`, `rc_level` (the level played in the last generation: lane steps or `"none"`), `rc_would` (the controller's choice for it; = `rc_level` in active mode), `rc_next` (its choice for the next generation), `rc_probe` (probe generations since the previous `gen` event), `rc_H_eff` ([min, max] of the repair calls' H_eff since then), `rc_cap_hits` (calls the played cap cut since then), `rc_switches` (cumulative); with `exchange.from_cuopt` (A9) also `inc_solver`, `inc_cuopt` (the two incumbent keys behind `inc_external`), `dual_key_cuopt` (cuOpt's bound — recorded, not used), `cuopt_pop` (members that entered as cuOpt rows); with `repair.continuous: lp_decode` (active) also `lpd_state` (the worker's state: `starting`, `idle`, `busy`, `failed`, `stopped`) and, cumulative, `lpd_offers`, `lpd_skip_pending`, `lpd_solved`, `lpd_kept`, `lpd_entered`, `lpd_entered_F`, `lpd_entered_I`, `lpd_incumbents`, `lpd_incumbents_F`, `lpd_incumbents_I`, `lpd_posts`, `lpd_host_s`, `lpd_host_s_max`, `lpd_sync_s` (summary, below); with the GPU back-off (A9, `backoff.enabled`) also `bo_mode`, `bo_phase`, `bo_state`, `bo_level`, `bo_duty`, `bo_busy_s`, `bo_paused_gen_s`, `bo_paused_call_s` (cumulative seconds slept between generations / inside repair calls — the phase seconds of a generation are CUDA-event intervals that include its in-call sleeps: subtract the difference of `bo_paused_call_s`), `bo_wakes`, `bo_contrib` (material contributions, cumulative), `bo_lam`, `bo_patience_s`, `ls_steps` (EA lane steps since the previous `gen` event); `seconds` then has a phase `paused` (between-generation sleeps) |
| `best` | EA | a new population best (an incumbent; during the feasibility phase a lower total violation): `key`, `obj`, `feasible`, `rank`, `origin` (bits), `ops` (their names), `anc_rank`, `anc_ops`, `parents`, `id`, `birth`, `V`, `gen`; with `strategy.enabled` also `strat` (its repair strategy) and `strat_credited` (false for a solver solution — injected, no EA operator bit — which is counted under `best_events_solver`, not credited to `strat`; the same for a cuOpt solution); with `exchange.from_cuopt` (A9) also `cuopt` (the individual entered as a cuOpt row) |
| `frontier` | EA, every `frontier.update_every` generations | `gen`, `size`, `best_key`, `worst_key`, `max_dist`, `mean_dist_to_inc`, `mean_pairwise`, `hv` |
| `rclock` | EA (`repair_clock.enabled`), one per generation (generation 0 = the initial population in `begin`) | the repair clock ([[I-TREA II algorithm]], "Repair clock"): `gen`, `t_dec` (the decision time, seconds since t0), `phase` (`pre`: the population has held no feasible member of the EA's own yet — an unchanged solver solution does not count; every call is played at "none" without a cap and nothing is accumulated; `post`), `mode`, `level` (played: lane steps or `"none"`), `cap` (the played per-call deadline in seconds; null = none), `would`, `would_cap` (the controller's choice; equal to the played one in active mode), `probe`, `would_cap_hit` (a call's t(would level) exceeded `would_cap`), `next` (the level chosen for the next generation), `switched`, `gen_s` (measured generation seconds), `o_s` (seconds outside the counterfactual repair calls); per level, in the order of the start-up record's `repair_clock.levels` (2^7 … 2^16, `"none"`): `gain` (Δ_g(h) = the largest dg over the lanes of the counterfactual calls that ended FEASIBLE with steps ≤ h and are feasible under the evaluation, dg = min((cutoff − key) / max(\|cutoff\|, \|key\|, \|dual_key\| if finite, 1e-9), 1) ∈ [0, 1]; below `materiality` → 0; null = not observed — a counterfactual call of the generation was cut below h), `T` (T_g(h) = `o_s` + Σ t_call(h) over the counterfactual calls), `used` (per level: accumulated by the controller — the observed levels in active mode; in shadow mode only those its decision would have revealed: the levels ≤ `would`, or all if no lane of a call ran beyond it), `gain_approx` (the largest dg of the non-counterfactual calls, not in `gain`; null = none), `t_children` (the children call's t(h)), `rate` (decayed G / T after this generation), `n`, `events` (accumulated post-phase generations, and those with a material gain); `best` ({`site`, `steps`, `dg`} of the largest-gain lane counted in `gain`); `calls`, one per repair call: `site` (`init`, `children`, `lp`, `lps`, `admit`), `k` (lanes), `pinned` (the one-shot LP batch, `pin_lp`: its gain and time count for every level), `cf` (counterfactual: false for the immigrant calls after the survival — shortfall fill and stagnation — which depend on the generation's earlier calls; left out of `gain` and of what is observed, their seconds count in `o_s`), `H`, `cap` (played), `cap_hits`, `H_eff` (the largest lane step count — the cut point when `cut`), `cut` (some lane has status 4), `s` (seconds), `t` (t_call(h) per level; null = not observed), `cutoff`, `dual_key` (the keys dg was computed from; null cutoff = none armed), `steps` (per lane), `status` (per lane, one digit each: 0 interrupted, 1 feasible, 2 stall, 3 budget, 4 horizon), `lane_gains` ([steps, dg, status] of every lane that beats the cutoff and is feasible under the evaluation, whatever its status but 0, largest dg first, at most 64), `beat_stopped`, `beat_cut` ([lanes, best dg] of those lanes that did NOT end FEASIBLE: stopped by stall / budget, or cut — not counted in `gain`), `prof` (GPU: `chunks` with `cum` / `t` — cumulative steps and seconds since the call started, after every launch, from (0, chunk start) — `end`, `H_eff`, `done`, `cut`, `cap_hit`; CPU: `seconds`, `H`, `cap_s`, `cut_rows`); with the GPU back-off also `bo_wall_s` (the generation's wall seconds) and `bo_paused_s` (seconds slept inside it), so `gen_s` + `bo_paused_s` = `bo_wall_s` (all its seconds — `t_dec`, `gen_s`, `o_s`, the calls' `s` and `t` — are then busy seconds) |
| `lp_decode` | EA (`repair.continuous: lp_decode`, active) | the LP decode worker ([[I-TREA II algorithm]] step 6; `ea.lp_decode`, `ea.engine` "LP decode"), `what` ∈ **`ready`** (once: `ready_t` = when the EA read it, `ready_t_worker` = when the worker sent it, by its own clock — D-017 M0 reads this one; the worker's `pid`, `read_s` (its read of the instance), `n`, `m`, `n_int`, `n_cont`, `nice`, `omp_num_threads`, `cpus`, `ready_after_s` / `ready_after_s_worker` (seconds after the worker's start)), **`result`** (one per reply: `req`, `gen_offered`, `t_offered` (the request), `gen` (when read), `site` (where it was read: `exchange` tick, `candidates` = just before the candidate set takes the pending batches, `pick` = the pick site after the survival, `finish`), `gen_enters` (the candidate set its copies are to join: `gen` when read before `_with_pending`, `gen + 1` at the pick site — which then offers nothing in that generation, `lpd_skip_pending`, so the next reply is read before a candidate set again; null when nothing was kept or read at `finish`; a batch that never joins one because the run ended first — read at the last pick site, or at the exchange tick of a generation cut short during the repair, whose `gen_enters` then equals `generations` — is named by a `dropped` event), `kept` (copies kept), `t_sent` (the worker's send time, seconds since t0), `wait_s` (the finished reply's wait in the pipe: EA read − worker send), `read_s` (seconds in recv()), `rows` — one per offered row: `src` (the source's id), `kind` (`F` polish / `I` completion), `code` (`ok`, `infeasible`, `time_limit`, `other`, `range_fail`, `cached`, `not_solved`, `after_deadline`), `key_src`, `V_src`, `rank_src` (the source's rank in that survival, −1 = eliminated), `key` (the copy's), `obj` (the LP's), `iters`, `s` (LP wall seconds), `cpu_s` (LP CPU seconds), `limit` (the time limit it ran under), `status` (HiGHS's model status), `cold` (a priming solve: no optimal basis existed yet; null from a stub), `kept` — and `stats` (the worker's cumulative statistics, as in the summary)), **`dropped`** (at the end of the run, one per decoded batch still pending, whose copies never joined a candidate set: `req` (its result event), `site` and `gen_enters` as recorded at the read, `rows`, `gen`; their rows are `lpd_dropped`), **`failed`** (`error`, `traceback`: digest mismatch, a worker exception, a dead worker, a failed send) |
| `backoff` | EA (A9 with `backoff.enabled`), at the start and on every change | the GPU back-off's state after the change: `gen`, `mode` (`active`, `shadow`, `forced`), `phase` (`Y`: no incumbent anywhere yet; `P`; `forced`), `state` (`search`: no feasible member of the EA's own; `own`; `led`: at least one material lead), `level` (the duty exponent: duty = 2^−level; in phase Y the yield level; null in forced mode), `duty`, `reason` (`start`; `y_peek` / `y_tick` / `y_own` / `y_expiry`: phase Y ended by a non-consuming peek of a partner's info channel at a pace point (between generations or inside a repair call), by an exchange tick, by the EA's own first incumbent, at `yield_until_s`; `own_feasible`; `escalate` (a step down; `busy_s` = the busy time it fell due — in shadow mode the would-be step); `lead` / `assist` (a material contribution that changed the level or state); `wake` (a partner improvement ended a between-generation sleep and the credit was forgiven)), `busy_s` (busy seconds since t0 — for a step down or a phase-Y exit, when it fell due), `t_due` (its wall time since t0: the due time, which can be earlier than the update that logs it — shadow mode updates once per generation; a phase-Y exit at or after `yield_until_s` is dated at `yield_until_s`), `paused_s` (seconds slept so far), `lam` (the decayed material-event rate λ̂, per busy second), `patience_s` (τ of the current state) |
| `contrib` | EA (A9 with `backoff.enabled`) | one per lead or assist: `what` (`lead`: an incumbent with source `self`; `assist`: an accepted SCIP verdict on an EA post whose note says `improved`, not an unchanged cuOpt row), `dg` (the repair clock's gap unit from `ref` to `key`: min((ref − key) / max(\|ref\|, \|key\|, \|dual_key\| if finite, 1e-9), 1); 1 if `ref` is +∞), `ref` (lead: min(the EA's best key before it, the external incumbent); assist: SCIP's `primal_before` as a key), `key`, `id`, `busy_s`, `material` (dg ≥ `backoff.materiality`; only material ones move the governor), `dup_lead` (present, = true, only on an assist whose row was already credited as a material lead — absent on every other `contrib` event; one improvement enters λ̂ once, so `material` is then false; counted in the summary's `n_assist_dup`) |
| `callback` | SCIP, HiGHS (not with `--no-plugin`) | `what`: `first` (SCIP adds `stage`, `timing`), `first_solving` (**SCIP only**: the first callback in the SOLVING stage, `timing`), `summary` at the end: `calls`, `calls_by_timing`, `t_first_callback`, `t_first_solving_callback`, `max_gap_s` (longest stretch without a callback, tail included), `callbacks_per_s`, `record_s`. The "no exchange window" subgroup (first SOLVING-stage callback after T/2) is therefore defined for SCIP only |
| `note` | all | `msg` (anything else: float64 re-check failures and demotions, injected rows infeasible under the EA's test, OOM retries with the halved budget, stagnation immigrants, stop seen, plugin/callback errors, a non-optimal root LP) |

`itrea.trace`'s module docstring lists only the core kinds and fields; this table is the contract.

### Summary (`<role>.summary.json`)
Identity: `run_id`, `role`, `instance` (repository-relative), `arm`, `seed`, `budget`, `t0`, `t0_epoch`; solver
processes add `solver`, `module`, `args`, `pid`, `host`, `cpus` (affinity), `threads_env`, `mailbox`, `mailbox_kind`,
`versions`, `emphasis`, `plugin` (`active` / `inert` / `none`), `n`, `m`, `sense`; the EA adds `config` (full dict) and
`config_hash`, `backend` actually used (the analysis drops a run whose backend does not match its arm), `device`,
`pool_limit_bytes`, `P`, `offspring`, `gene_dtype`, `local_search`, `versions`, `cupy_cache` (files before prepare, at
`t0`, at the end, and `new_after_t0`: the names of the files that appeared after `t0`, at most 50) and
`compiled_on_clock` (true if the kernel cache grew after `t0`).
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
Rank 1 (D-016; code of 2026-10-08 or later): `final_r1_n`, `final_r1_dmax`, `final_r1_dmed`, `final_r1_gap`,
`final_r1_dmax_band` (the `gen` event's `r1_*` for the final population, both Rank 1 modes); with
`ranks.rank1_mode: clearing` also the counter `rank1_cleared` (candidates the distance rule skipped, summed over the
final round of every survival; absent in `top` mode).
The EA's start-up record (`startup`) holds, besides the step timings (`*_s`) and the backend facts, `rank1` (`mode`:
`top`, `clearing`, or `top (clearing requested, no integer variables)`; `min_diff`: k, null in `top` mode; `n_int`;
`size`: the clearing cap, null in `top` mode) and `warmup` ([[I-TREA II algorithm]], "Warm-up before the clock"):
`many` and `one`, one per warm-up pass, each with `generations`, `feasible_found`, `posts`, `immigrants`,
`violation_guided_forced` (children mutated by the forced `violation_guided` call; 0 when it cannot apply),
`elite_warmed` (times `_warm_elite` ran: 1 when the instance has integer variables, else 0) and, with `lp_decode`,
`lp_decode_warmed` / `lp_decode_warmed_batches` (below); on the GPU `scan_ladder_max` (the largest cumsum size warmed);
`n`, `m` (the synthetic problem's shape). The deterministic operator sweep `_warm_operators` (2026-10-09) counts its
sweeps only in the discarded warm-up engine: no trace field records it, so `compiled_on_clock` /
`cupy_cache.new_after_t0` are the check.
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
With the GPU back-off (A9 with `backoff.enabled`) the EA summary adds `backoff`: the settings (`mode`, `forced_duty`,
`yield_level`, `levels`, `yield_until_s`, `patience_search_s`, `patience_own_s`, `materiality`, `patience_kappa`,
`patience_min_s`, `patience_max_s`, `half_life_s`, `step_s`, `burst_s`, `max_sleep_s`, `slice_s`, `wake_min_s`,
`assist`), `final_phase`, `final_state`, `final_level`, `final_duty`, `wall_s` / `busy_s` (since t0) and `busy_share`,
`paused_s`, `paused_gen_s`, `paused_call_s`, `sleeps` (sleep slices), `phase_y` ({`length_s`, `exit_t`, `exit_busy_s`,
`reason`: peek | tick | own | expiry | never}), `seconds_per_level` / `busy_per_level` (wall / busy seconds per level,
keys `Y`, `0` … `3`, `forced`), `own_feasible` and `first_lead` ({`t`, `busy_s`}; the lead with `dg`), `n_lead`,
`n_assist` (material), `n_lead_all`, `n_assist_all`, `escalations`, `first_escalation` ({`t`, `busy_s`, `level`,
`state`}; shadow mode: the would-be one), `wakes`, `wakes_forgiven`, `lam`, `timeline` ([t, busy_s, phase, level,
reason] per change, at most 4,000), `timeline_truncated`, `ls_steps` (EA lane steps), `n_assist_dup` (assists not
credited because their row was already a material lead), `max_launch_s` (the longest FJ launch). The start-up record
then holds `backoff` (the settings and `live`: whether the run reads cuOpt).
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
With `repair.continuous: lp_decode` the start-up record holds `lp_decode` (`state` "inert" with a `reason` where no
active column exists — then nothing else below appears — else "configured", `active_columns`, `cont_obj`, `per_gen`,
`time_limit`, `cold_limit`, `infeasible`, `int_dtype` (the integer parts' payload dtype: the smallest of int8 / int16 /
int32 holding every integer bound, else float64), `constants`) and `warmup.many` / `warmup.one` add
`lp_decode_warmed` (the largest decoded batch, in rows, that entered a candidate set in that pass: 3 / 1) and
`lp_decode_warmed_batches`. The EA summary then adds the counters `lpd_offers` (requests sent), `lpd_rows_offered`,
`lpd_rows_F` / `lpd_rows_I` (offered rows by the source's kind: F = feasible, to polish; I = infeasible, to complete —
row offers: CACHED rows and NOT_SOLVED rows offered again count each time), `lpd_busy` (pick sites that found the worker
busy or starting), `lpd_skip_pending` (pick sites that read a reply with copies and therefore offered nothing),
`lpd_replies`, `lpd_solved` (rows that reached HiGHS) and `lpd_solved_F` / `_I`, one counter per row code (`lpd_ok`,
`lpd_infeasible`, `lpd_time_limit`, `lpd_other`, `lpd_range_fail`, `lpd_cached`, `lpd_not_solved`,
`lpd_after_deadline`), `lpd_cached_F` / `_I` (rows the worker's cache answered), `lpd_screened_I` (I rows the range
screen looked at: neither cached nor NOT_SOLVED; = the worker's `screen_pass` + `screen_fail`; F rows are never
screened), `lpd_srcs_F` / `_I` (distinct sources the worker screened or solved — each is recorded as offered once),
`lpd_ok_F` / `_I` (LP feasible), `lpd_no_gain` (F copies dropped: no gain over the source by `imp_tol`), `lpd_kept`,
`lpd_kept_F` / `_I`, `lpd_batches` (decoded batches that joined a candidate set — at most one per set), `lpd_dropped`
(kept copies whose batch never joined one: pending when the run ended; `lpd_kept` = rows that joined + `lpd_dropped`),
`lpd_entered`, `lpd_entered_F` / `_I` (copies that entered the population), `lpd_incumbents`, `lpd_incumbents_F` / `_I` (EA incumbents that are copies; by copy id — origin bits
are OR-merged over duplicates, so this is attribution by entry, not by bits), `lpd_posts`, `lpd_posts_F` / `_I` (posts
made from a copy), `lpd_lost` (a request outstanding when the worker stopped or failed), `lpd_failed` (0/1); and
`lp_decode`: `state` (at stop; "failed" if it ever failed), `error`, `ready_t`, `ready_t_worker`, `ready` (the "ready"
message), `pid`, `host_s` / `host_s_max` (EA-thread seconds of the decode, perf_counter, total / per call; D-017 M1),
`sync_s` / `sync_s_max` (the device wait before the row copies, NOT in `host_s`: the survival's last gather may still
be running), `wait_s_total` / `wait_s_max` (replies' wait in the pipe), `ipc` (`offers`, `offer_s_total`,
`offer_s_max`, `offer_bytes_max`, `reads`, `read_s_total`, `read_s_max`, `read_bytes_max`: the EA thread's seconds in
send() and recv() and the payloads, included in `host_s`), `batch_rows_max`, `gain_F_entered_median` /
`gain_F_entered_n` and `gain_I_entered_median` / `gain_I_entered_n` (the median relative gain
(key_src − key) / max(1, |key_src|) of the decoded copies that ENTERED the population, by the kind of their source, on
the float64 keys of the result rows, and how many; positive = better than the source; null median when none), `worker`
(the statistics of the last reply read: `requests`, `rows`, `solves`, `optimal`, `ok`, `iterations`, `solve_s_total`, `solve_s_max`, `solve_cpu_s_total`,
`cache_hits`, `screen_pass`, `screen_fail`, `restores`, `restore_errors`, `after_deadline`, `run_time` (HiGHS's
cumulative), `ru_maxrss_mb`, `cpu_s`, and for M2 `priming_solves`, `priming_s`, `priming_first_status`,
`priming_first_s` (the first cold solve, set once), `priming_last_status`, `priming_time_limit_hits`, `warm_solves`,
`warm_time_limit_hits` (warm = an optimal basis existed), `time_limit_hits` (all), `time_limit_hits_starved` (a hit
whose solve got less than half a core: a starved worker, not a slow LP — HiGHS's limit is wall time)), and the
configuration echo (`active_columns`, `cont_obj`, `per_gen`, `time_limit`, `cold_limit`, `infeasible`, `int_dtype`,
`constants`).
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
`rmm_total_bytes`, `rmm_current_bytes`, `rmm_peak_count`, `cpu_threads`, `gpu_mem_gb`, `mip_gap_zero` and `settings`
(the parameters of the timed solve: `time_limit`, `log_to_console`, `num_cpu_threads`, `random_seed`, and with
`--mip-gap-zero` — arm A6m, D-020 — `mip_absolute_gap` = `mip_relative_gap` = 0; every other parameter is cuOpt's
default).

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
apart — added 2026-10-07, only in A9 genealogies). With `repair.continuous: lp_decode` (active) also `lpd_src` int64
(the id of the source an LP-decoded copy was made from, −1 for every other row; a copy has `pa = pb = −1`, origin
`ORIGIN_LP_DECODE` and its source's strategy — added 2026-10-08, only in lp_decode genealogies).
The two flags are written explicitly so the lineage analysis does not derive them. About 40 bytes per row, well under
50 MB for a 300 s run. The OR-accumulated ancestry bit masks in `ea.population` are kept only as an online summary: they
saturate within a few generations (critique, algorithm lens).

### Run record (`run.json`)
Identity: `run_id`, `arm` (with `~<ablation>`), `base_arm`, `ablation`, `arm_doc`, `gpu_arm`, `lane_kind`, `job`,
`set`, `instance`, `name`, `seed`, `budget`, `grace_s`, `block`, `position` (Williams position of the arm in its block),
`window`, `stratum`, `split`; `config` {`path`, `overrides`, `base_hash` (the file alone — the freeze check compares
it with `frozen-e2`), `ea_hash` (the effective configuration), `run_hash`, `solver_settings` (arms without an EA)};
`problem` (shape, `load_s`); `mailbox` {kind, spec}; `lane` (with `whole_box`), `systemd`, `pinned`; `memory_budget`
(the D-003 gate terms and `need_gb`); `env` (versions, hardware, git state: `itrea.experiments.record_env`); `t0`,
`t0_epoch` (at the barrier's release).
Processes: one record per process — `role`, `kind`, `backend`, `cmd`, `cores`, `memory_max_gb`, `gpu_limit_gb` (CuPy
pool or RMM limit), `env`, `pid`, `launch_epoch` (the launch clock's origin, D-020), `startup_s`, `exit_code`,
`killed`, `exit_t`, `cpu_user_s`, `cpu_sys_s`, `maxrss_mb`; `exit_codes` {role: code} — **every one is checked**
(optopt X1 logged "ok" for dead SCIP processes). `runner_cores`: the runner's own affinity (the GPU-health sampler and
the stop loop run there; on the whole-box lane `full` of arm A6m they share the run's cores, D-020).
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
  `best_obj`, `n_steps`, `pdi`; `pt_mf` (the MIPfeas-style integral, D-002 §4); **`pt_gams`** (GAMS's MIPfeas
  integral exactly, call clock, T = the budget; D-020), **`pt_gams_launch`** (the same on the launch clock) and
  **`startup_offset_s`** (NaN, and so `pt_gams_launch`, when the run record lacks `t0_epoch` or the launch epochs,
  and for synthetic rows); `t_first` is the portfolio's first checked incumbent on the call clock;
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
  D-017 LP decode (−1 / NaN / "" for runs without it; `lpd_state` "inert" and nothing else where it could not act):
  `lpd_state`, the summary counters above (`lpd_offers` … `lpd_failed`, with the `_F` / `_I` splits),
  `lpd_host_s`, `lpd_host_s_max`, `lpd_sync_s`, `lpd_sync_s_max`, `lpd_ready_t`, `lpd_ready_t_worker`,
  `lpd_wait_s_total`, `lpd_wait_s_max`, `lpd_batch_rows_max`, `lpd_gain_F_entered_median`, `lpd_gain_F_entered_n`,
  `lpd_gain_I_entered_median`, `lpd_gain_I_entered_n`, `lpd_offer_s_max`, `lpd_offer_s_total`,
  `lpd_offer_bytes_max`, `lpd_read_s_max`, `lpd_read_s_total`, `lpd_read_bytes_max`, and from the worker's statistics
  `lpd_priming_s`, `lpd_priming_first_s`, `lpd_priming_first_status`, `lpd_priming_solves`,
  `lpd_priming_time_limit_hits`, `lpd_worker_solves`, `lpd_warm_solves`, `lpd_worker_time_limit_hits`,
  `lpd_warm_time_limit_hits`, `lpd_time_limit_hits_starved`, `lpd_screen_pass`, `lpd_screen_fail`,
  `lpd_worker_solve_s`, `lpd_worker_solve_cpu_s`, `lpd_worker_maxrss_mb` (`build_runs.LPD_*`). D-017's mechanism
  reads: M0 `lpd_ready_t_worker` ≤ 10 s and `lpd_failed` = 0; M1 `lpd_host_s` ≤ 1 % of `ea` wall and
  `lpd_host_s_max` ≤ 50 ms (`lpd_sync_s` reported beside them); M2 `lpd_warm_time_limit_hits` / `lpd_warm_solves`
  and, on ca, `lpd_priming_first_status` (the worker's statistics are those of the last reply read; the result
  events' rows recount them exactly — a solve is a row with code ok / infeasible / time_limit / other, `cold` marks
  priming, a hit is `status` "Time limit reached", also on an ok row — and both miss the request still in flight at
  the deadline, which `lpd_lost` counts: report it beside M2); M3 / M4 the funnel per kind of source — offered
  `lpd_rows_F` / `_I` (row offers, cached and re-offered rows included) and `lpd_srcs_F` / `_I` (distinct sources),
  answered by the cache `lpd_cached_F` / `_I`, screened `lpd_screened_I` (I rows only), solved `lpd_solved_F` / `_I`,
  LP feasible `lpd_ok_F` / `_I`, kept `lpd_kept_F` / `_I`, entered `lpd_entered_F` / `_I`, incumbents and posts —
  and M3's median relative gain of the F copies that entered the population, `lpd_gain_F_entered_median` (over
  `lpd_gain_F_entered_n` copies; D-017 says "admitted" — entered is the reading these columns give; the result rows
  with `kept` give the kept reading, minus the batches named by `dropped` events);
  A9's GPU back-off (missing for runs without it; `build_runs.backoff_stats`): `bo_mode`, `bo_busy_share`,
  `bo_paused_s`, `bo_phase_y_s`, `bo_phase_y_exit`, `bo_level0_s` … `bo_level3_s` (wall seconds per level; NaN in
  forced mode), `bo_n_lead`, `bo_n_assist`, `bo_own_feasible_s`, `bo_first_lead_s`, `bo_first_lead_busy_s`,
  `bo_first_escalation_s` (busy; shadow: the would-be one), `bo_escalations`, `bo_wakes`, `bo_ls_steps`;
  Rank 1's spread (D-016) has no column: `tools/d016_report.py` reads `final_r1_*`, `rank1_cleared` and
  `compiled_on_clock` from the EA summaries and the last `frontier` event's `max_dist` from the events;
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
benchmark instances). Column `pt_mf`, unchanged by D-020.

**GAMS's MIPfeas integral, exactly** (D-020; `itrea.analysis.metrics.gams_integral`, column `pt_gams`): a port of
`calculate_primal_integral` from GAMS's scoring script (`create_primalintegral.py`, 2026-09-15, in
`data/mipfeas/gams-2026-09-16/`), on the checked best-so-far steps and our reference: γ = 2 before the first incumbent;
then, in their order, γ = 0 if round((v − z*)/max(|v|, |z*|, 1), 6) < 0 (v beats z*: their script raises, ours flags
`beats_ref`), γ = 1 if v·z* < 0, else that rounded ratio; every time capped at T (a step at or after T adds nothing),
the last γ held to T; P = Σ γ·Δt / T ∈ [0, 2]; T = the run's budget (600 s in MIPfeas). It differs from `pt_mf` in the
sign rule (1 instead of up to 2), the rounding and the cap. Their script assumes minimisation; the port takes the
difference in the objective's sense, so a maximisation instance scores as its minimisation mirror (none of the 233
maximises: no MPS file has an OBJSENSE section). Acceptance (2026-10-11): on GAMS's own traces read by
`itrea.analysis.mipfeas.read_gams_trc` (their exact `read_csv`), the port equals their function to ≤ 2.3e-16 on all
3,495 traces of the archive, every solver and machine (`tests/test_mipfeas.py` checks the 233 GB10 cuOpt 26.08 ones,
and that they equal `primalintegral.csv`). Aggregation: shifted geometric mean, s = 0.001; "feasible found" = P < 2.
`pt_gams_launch` is the same integral on the launch clock (every step time + `startup_offset_s`, T unchanged).

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
