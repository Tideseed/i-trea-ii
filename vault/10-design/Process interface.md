# Process interface — how a pair run is put together (contract, 2026-10-05; updated 2026-10-10 to the built code; 2026-10-11: arm A6m and the whole-box lane, D-020)

A **run** = one (instance, arm, seed, budget). The runner (`itrea.experiments.pair`) creates a run directory
`traces/raw/<set>/<run_id>/` and a mailbox directory `/dev/shm/itrea/<run_id>/`, writes `order.json` (canonical names,
HiGHS column order; into the mailbox and the run directory), launches the arm's processes, waits until every process
wrote `ready.<role>`, releases the barrier (`start` with `t0`), waits `budget + grace`, writes `stop`, waits for exits,
**checks every exit code** (optopt X1: a dead SCIP logged "ok"), records resources and merges the trajectories.

`run_id = <instance-stem>__<arm>__s<seed>__<budget>s__<cfg-hash>`.

## Process kinds (all: `python -m <module> …`; exit 0 = normal end, 1 = exception, 2 = bad arguments)
| module | role in mailbox | arguments |
|---|---|---|
| `itrea.ea.run` | `ea` | `--instance P --config Y [--set k=v …] --seed S --budget T --run-dir D [--mailbox M] [--run-id R (default $ITREA_RUN_ID)] [--role-name N] [--no-barrier] [--barrier-timeout S] [--sol-out F] [--arm A] [--backend cupy\|numpy] [--gpu-mem-gb G]` |
| `itrea.solvers.scip_run` | `solver` | `--instance P --seed S --budget T --run-dir D [--mailbox M] [--no-barrier] [--sol-out F] [--mem-mb MB] [--recv] [--send] [--inject-band B] [--inject-k K] [--poll-s S] [--inert] [--no-plugin] [--emphasis E] [--run-id R] [--role-name N] [--barrier-timeout S] [--arm A] [--lp-stream [--lp-stream-share F] [--lp-stream-ring R] [--lp-stream-min-gap S] [--lp-stream-q Q] [--lp-stream-patience S]]` |
| `itrea.solvers.highs_run` | `solver` | same as SCIP (`--recv` requires the user-solution callback test to pass) |
| `itrea.solvers.cuopt_run` | `cuopt` | `--instance P --seed S --budget T --run-dir D [--mailbox M] [--run-id R] [--role-name N] [--no-barrier] [--sol-out F] [--cpu-threads N (≥ 2, no upper limit)] [--gpu-mem-gb G] [--mip-gap-zero] [--send] [--no-warmup] [--arm A]` — run under `--cuopt-python` (an interpreter with cuOpt 26.08); never receives solutions (a SetSolutionCallback disables cuOpt's presolve); `--send` posts on cuOpt's own channels `to_ea_cuopt` / `info_cuopt` (A9 only); `--mip-gap-zero` sets `mip_absolute_gap` = `mip_relative_gap` = 0 (A6m only); the solve's parameters are recorded in the summary as `settings` |

The runner passes `--run-id` and, in every non-exchange SCIP arm, `--inert` (plugin installed, never polls — the
harness-matched control); cuOpt gets `--gpu-mem-gb <same limit as the EA pool>` and `--cpu-threads 3` (A6m: 24 and
`--mip-gap-zero`, below); A0* gets
`--emphasis`; the EA gets `--config <run_dir>/config.yaml` (the effective configuration, no `--set`).
`--recv`: accept the partner's solutions (SCIP: heuristic plugin + `trySol`; HiGHS: user-solution callback).
`--send`: publish own incumbents (+ pool within `--inject-band`, at most `--inject-k`), dual bound and the root LP
solution to the EA. `--lp-stream` (SCIP only, needs `--send`; D-011): also stream good infeasible node LP points to the
EA's `lps` channel under a time budget — the runner adds it with the stream's parameters when the EA configuration has
`lp_stream.enabled`. Arms without exchange omit both. `--role-name` lets two solver processes share a run dir (A1:
`solver` and `solver2`).

## Files each process writes in its run directory
- `<role>.events.jsonl` — event stream (`itrea.trace`), times since `t0`;
- `<role>.summary.json` — identity, start-up record (seconds before `t0`), final incumbent, counters, exit reason;
- `<role>.sol.json.gz` — the final incumbent as `{name: value}` (non-zeros) for the independent checker;
- `<role>.log` — stdout/stderr (the runner redirects).

The runner adds `run.json`: arm, processes with their cores, memory caps (host `MemoryMax`, GPU pool limit), exit
codes, MemAvailable and ACPI temperature before/after, co-running units (`box-window status`, LLM servers' state),
git revision, and the merged trajectories: `portfolio` (best of all processes at every moment) and `solver` (what the
solver itself held).

## Arms → processes
| arm | processes |
|---|---|
| A0 `scip` | scip_run |
| A1 `scip2` | scip_run (seed s) + scip_run (seed s + 10,000, role `solver2`) |
| A2 `scip_cuopt` | scip_run + cuopt_run |
| A3 `scip_ea_gpu` | scip_run `--recv --send` + ea.run (backend cupy) |
| A4 `scip_ea_cpu` | scip_run `--recv --send` + ea.run (backend numpy, pinned core) |
| A5 `ea_gpu` | ea.run alone (`--no-barrier` or a runner barrier with one role) |
| A6 `cuopt` | cuopt_run alone |
| A6m `cuopt_gams` | cuopt_run alone as GAMS ran it for MIPfeas: `--cpu-threads 24 --mip-gap-zero`, all 20 cores (whole-box lane, `--layout full`) — below |
| H0/H2/H3 | as A0/A2/A3 with highs_run |
| A9 `scip_ea_cuopt` | scip_run `--recv --send` + ea.run (backend cupy, `exchange.from_cuopt: true`) + cuopt_run `--send` — below |
| A3x `scip_ea_x925` | as A3, but the EA on the lane's second X925 core (A9's footprint, no cuOpt) — A9's matched control, below |
| ablations | A3 with `--set` overrides (`ranks.x=0`, `ranks.y=0`, `exchange.to_solver=false`, …) and the matching solver flags |

## Helper processes of the EA (after t0)
The EA starts up to two HiGHS processes of its own, both with multiprocessing "spawn" at t0, so they inherit its CPU
affinity and, as its descendants, count as the run's own for GPU health:
- the **LP worker** of `seeding.lp_source: highs_async` (the EA's own root LP, at most `seeding.lp_time_limit`, then it
  exits);
- the **LP decode worker** of `repair.continuous: lp_decode` ([[D-017 LP decode — re-optimise the continuous variables
  of selected candidates]], `ea/lp_decode.py`) on an instance with an active column: one persistent `highspy.Highs`, 1
  thread, niced 10, `OMP/OPENBLAS/MKL_NUM_THREADS=1` and `CUDA_VISIBLE_DEVICES=""`, a pipe to the EA, stopped by the EA
  after the clock. Its "ready" event records its `cpus`. In A3 (the development default `configs/itrea2-dev-rc-ld.yaml`
  since 2026-10-09) both workers share the EA's three A725 cores.

**The runner refuses `lp_decode` in A9 and A3x** (`pair.plan`: `RunError` for an EA on its own X925 core,
`ProcSpec.x925`, when the effective configuration says `repair.continuous: lp_decode`). `plan` runs when the pool
budgets a job (`job_budget`) and again before launch, so such a job fails before any process starts; A3 with the same
file is accepted (`tests/test_lp_decode.py::test_runner_refuses_lp_decode_for_the_single_core_arms`). Reason (D-017):
the decode worker would share the EA's one core with the EA's host loop and the LP worker, and one more pending batch
would take the candidate set's concatenate to CuPy's 8-part limit. It stays refused until a decision allows it (issue
#30, the A9 core layout, milestone C1).

## Three-way arm A9 (2026-10-07, [[D-015 Use cuOpt's feasible solutions — three-way arm A9]])
**SCIP ⇄ EA ← cuOpt.** cuOpt runs as shipped (no SetSolutionCallback, presolve on) beside SCIP and the GPU EA and
streams every incumbent to the EA; the EA injects them and keeps exchanging with SCIP exactly as in A3, so cuOpt's
vectors — unchanged or polished by the EA — reach SCIP through the EA's posts. Development arm (not pre-registered).
- **Channels.** cuOpt writes its OWN channels: `to_ea_cuopt` (file `to_ea_cuopt.npz`, Redis stream `ch:to_ea_cuopt`;
  one row, ORIGIN_SOLVER) and `info_cuopt` (`to_ea_cuopt_info.json` / key `info_cuopt`: incumbent, bound, status). A
  channel keeps only its latest message, so sharing SCIP's `to_ea` / `info` would let either process overwrite the
  other's. `cuopt_run --send` is passed only in A9; A2 and A6 never send. The EA reads cuOpt's channels only with
  `exchange.from_cuopt`, at every exchange tick after SCIP's.
- **Info-channel peek (D-018).** `InfoChannel.peek` (file mailbox) and `_Info.peek` (Redis) return the latest info
  message WITHOUT consuming it: the reader's `_seen` / `last_seq` are left alone, so the next `poll` still returns it.
  Only the EA's GPU back-off uses it: in phase Y, at most every `exchange.poll_s`, it peeks SCIP's `info` (when the EA
  reads the solver) and `info_cuopt` at both pace points — between generations and inside a repair call, where no
  exchange tick runs — and a finite incumbent ends phase Y (reason "peek"). Shadow mode never peeks.
- **Cores.** SCIP on the lane's first X925 core, the EA on its second (`ProcSpec.x925`), cuOpt on the three A725
  cores (`--cpu-threads 3`): the two partners never share a core. On the GPU lane: SCIP 5, EA 6, cuOpt 1–3.
- **Memory.** Two CUDA processes: the gate counts the EA's CuPy pool and cuOpt's RMM limit (both `gpu_mem_gb`, 12 GB)
  plus two CUDA overheads (1 GB each), and three `MemoryMax` caps (9 + 6 + 12 GB): 73 GB with the 20 GB reserve,
  75 GB with Redis (A3: 48 / 50 GB).
- **Configuration.** The runner adds `exchange.from_cuopt: true` (arm-implied, like A5's "no exchange"); SCIP gets A3's
  flags. Development jobs use `configs/itrea2-dev-rc-3way.yaml` (the repair clock's Stage 2 configuration, hash
  `1d12ad490dd0`) — not the A3 default `itrea2-dev-rc-ld.yaml`, because the runner refuses `lp_decode` here (above).
  The GPU back-off (D-018) is configured in files built on it: stage 0 `-bo0` (shadow) and `-bof50` / `-bof25` /
  `-bof12` (forced duty ½ / ¼ / ⅛); stage 1 `-bo1` (active, run hash f67354be775d) and `-bo1s` (the same values in
  shadow mode, the control, 62bf82b236f6). The governor exists only where `exchange.from_cuopt` is true, so an A3x job
  with a back-off file runs without one.
- **Matched control: A3x, not A3** (review of 2026-10-07). A3x = A3 with A9's EA footprint: SCIP on X925 5 with A3's
  flags, the EA on X925 6, the A725 cores idle, no cuOpt; same configuration file, same run hash as A3 (the arm label
  tells the run ids apart), A3's memory budget (48 GB with the reserve). A3 differs from A9 by the EA's cores as well:
  A3's EA has the three A725 cores (2.8 GHz), A9's one X925 core (3.9 GHz), and with `seeding.lp_source: highs_async`
  the HiGHS LP worker inherits the EA's affinity — beside the EA's host loop on A3's spare cores, on the same core in
  A9 and A3x for up to `seeding.lp_time_limit` (60 s; in the dev and class-probe runs so far: ca 23–34 s,
  cryptanalysiskb128n5obj16 22 s, neos-4763324-toguru 13–20 s, sing44 9–13 s, rail01 and chromaticindex1024-7 the
  full 60 s without a solution, the other instances ≤ 7 s). The LP worker cannot get a core of its
  own on the GPU lane (A725 1–3 are cuOpt's, 0 Redis's, 4 and 10–14 the runner's), so the control shares A9's layout.
- **What the EA does with cuOpt** (`itrea.ea.engine`, "Three-way arm A9"): injection as for SCIP's rows (never
  repaired, survival decides); cuOpt's incumbent joins the external incumbent (min over the two) for the survival's
  Rank 0 threshold, the immigrants' admission and the repair cutoff; cuOpt's bound is recorded, **not used** (Rank 0's
  bound demotion keeps SCIP's proven bound, so A9 differs from A3x only by cuOpt's process and its solutions). A post
  must beat **SCIP's** incumbent, not the min, or an unchanged cuOpt vector could never be handed to SCIP.
- **Attribution.** An individual that entered as a cuOpt row keeps that mark by id (the genealogy's `cuopt` column);
  an EA incumbent that is an unchanged cuOpt vector has source `cuopt`; its post to SCIP carries `POST_CUOPT` (bit
  16 of the message's origin), and SCIP records the incumbent it makes from it as source `cuopt`, not `ea`. A vector
  SCIP's channel delivered first stays SCIP's. `build_runs`: the portfolio is the best of the three processes, each
  has its per-role P (`solver_*`, `ea_*`, `cuopt_*`), and `cuopt_inject_msgs`, `cuopt_injected`, `ea_inc_from_cuopt`,
  `cuopt_posts`, `cuopt_posts_accepted`, `cuopt_posts_became_incumbent` (per post), `solver_inc_from_cuopt` (SCIP
  incumbent records) count cuOpt's contribution up to T; `posts_*` count every EA post, so the EA-made share is
  `posts_*` minus `cuopt_posts*` (units: [[Trace format and metrics]]).
- **GPU health.** Every process the runner launched (and its descendants) is the run's own, so the EA and cuOpt on the
  GPU together are not foreign activity.
- **Comparisons** (D-015 and its addendum): A9 vs A0, A6, A2 (SCIP ‖ cuOpt race), A3x (does cuOpt help the coupled
  system); A9 vs A3 is descriptive only (footprint confounded, above).

## Arm A6m and the whole-box lane (2026-10-11, [[D-020 MIPfeas harness — GAMS's clock and integral, paired calibration of cuOpt 26.08]])
**cuOpt alone as GAMS ran it for Mittelmann's MIPfeas** — the calibration arm of the MIPfeas harness (D-020, issue
#29). GAMS's logs show cuOpt 26.08 through GAMS 54.4.0's link with `num_cpu_threads 24`, `time_limit 600`,
`mip_absolute_gap 0`, `mip_relative_gap 0` and every other parameter at cuOpt's default, on a GB10 to itself.
- **Process.** `cuopt_run --cpu-threads 24 --mip-gap-zero --gpu-mem-gb <RMM limit>`, never `--send`; random_seed = the
  job's seed (0; GAMS left −1, the one deliberate difference). The same barrier, warm-up before `t0`, checker and
  run record as every arm. 24 threads whatever the lane (`ProcSpec.cpu_threads`; a `cuopt_threads` option that
  contradicts it is refused).
- **Lane.** `pair.FULL_LANE` = lane `full`: X925 5–9, 15–19 and A725 0–4, 10–14 plus the GPU (`whole_box`); cuOpt is
  pinned to all 20 cores. A6m runs only there and that lane runs only A6m (`Lane.fits`, `pair.plan` and the pool's
  `check_layout` refuse every other pairing with a clear error), so `itrea pair --arm A6m` on the GPU lane `g0` is
  refused. The pool's `--layout full` is this one lane; it implies the serving check (both local model servers
  stopped, `--allow-serving` refused) — D-020 wants no foreign GPU process. The runner's own threads (`RUNNER_CORES`,
  the GPU-health sampler, the stop loop) share the lane's cores: accepted, recorded as `runner_cores` in `run.json`.
  With the Redis mailbox the container's core 0 is inside the lane too; the calibration uses the file mailbox.
- **Memory.** D-020: RMM limit 36 GB (`--gpu-mem-gb 36`), host `MemoryMax` 24 GB (`--cuopt-mem-gb 24`); GAMS's max RSS
  was ≤ 16 GB on the sample, 49 GB at most (nw04) over the 233. The pool refuses A6m with other values
  (`pool.check_a6m_memory`: the RMM limit is in the run hash, `MemoryMax` is not). The gate: 24 + 36 + 1 (one CUDA
  process) + 20 reserve = **81 GB** with the file mailbox (`--dry-run` prints it).
- **Identity.** `solver_settings` = {arm A6m, emphasis null, `gpu_mem_gb` 36, `cuopt_threads` 24, `mip_gap_zero`
  true}: its own run hash, never A6's. Only A6m's dict has `mip_gap_zero`, so every other arm's hash is unchanged
  (`tests/test_pair.py` pins A6 `6e35f122f659`, A2 `125d69c76695`, A0 `27f3171fac78`, A9@bo1 `f67354be775d`, A9@bo1s
  `62bf82b236f6`, A3x `1d12ad490dd0`).
- **Jobs and analysis.** `itrea make-jobs mipfeas-cal` → `jobs/mipfeas-cal-26.08.jsonl` (D-020's 30 instances, seed 0,
  600 s, set `mipfeas`, split `dev`); `itrea build-runs` adds `pt_gams` / `pt_gams_launch` / `startup_offset_s`
  ([[Trace format and metrics]]); `itrea mipfeas-report` applies D-020's pass rule. A full-box A9 would need its own
  decision and its own layout entry.
