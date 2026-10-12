# Process interface — how a pair run is put together (contract, 2026-10-05; updated 2026-10-10 to the built code; 2026-10-11: arm A6m and the whole-box lane, D-020; then ITX-1's arms A6F / A2F, the harness, named scopes and new void causes, D-021); 2026-10-12: D-021 phase 2 — R12's stop at the proven optimum, cuOpt killed at a definitive outcome, preset itx1, job priority, R33's gate; review fixes the same day — the earliest trigger across roles, `scoped_hash`, R33 on the registered configuration

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
| `itrea.solvers.cuopt_run` | `cuopt` | `--instance P --seed S --budget T --run-dir D [--mailbox M] [--run-id R] [--role-name N] [--no-barrier] [--sol-out F] [--cpu-threads N (≥ 2, no upper limit)] [--gpu-mem-gb G] [--mip-gap-zero] [--send] [--no-warmup] [--arm A]` — run under `--cuopt-python` (an interpreter with cuOpt 26.08); never receives solutions (a SetSolutionCallback disables cuOpt's presolve); `--send` posts on cuOpt's own channels `to_ea_cuopt` / `info_cuopt` (A9 only); `--mip-gap-zero` sets `mip_absolute_gap` = `mip_relative_gap` = 0 (A6m, and every cuOpt process under a harness with `cuopt_gap_zero`, D-021 R13); the solve's parameters are recorded in the summary as `settings` |

The runner passes `--run-id` and, in every non-exchange SCIP arm, `--inert` (plugin installed, never polls — the
harness-matched control); cuOpt gets `--gpu-mem-gb <same limit as the EA pool>` and `--cpu-threads 3` (A6m: 24 and
`--mip-gap-zero`; A6F |F| and A2F |F| − 1 threads; under a harness its own RMM limit and `--mip-gap-zero` — below);
A0* gets
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

The runner adds `run.json`: arm, processes with their cores, memory caps (host `MemoryMax`, GPU pool limit), scope
unit names, exit codes, MemAvailable and ACPI temperature before/after, co-running units (`box-window status`, LLM
servers' state), git revision, and the merged trajectories: `portfolio` (best of all processes at every moment) and
`solver` (what the solver itself held). Since D-021 also the job's resolved harness (`config.harness`), the fan
governor's state at the start and the end, the start temperature, the CUDA-cache check and every D-021 void check
([[Trace format and metrics]]).

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
| A6m `cuopt_gams` | cuopt_run alone as GAMS ran it for MIPfeas: `--cpu-threads 24 --mip-gap-zero`, all 20 cores (whole-box lane, `--layout full`) — below; out of v1 (D-021 R9) |
| A6F `cuopt_F` | cuopt_run alone on all of F (the GPU lane's cores: X925 5–6 + A725 1–3), `--cpu-threads` \|F\| = 5 — ITX-1's "cuOpt alone" (D-021 R6), below |
| A2F `scip_cuopt_F` | scip_run (`--inert`) on F's first X925 core + cuopt_run on the other \|F\| − 1 cores, `--cpu-threads 4`, no exchange (as A2) — ITX-1's live race (R8), below |
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
- **Environment** (review 2026-10-11). Every other process gets `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS =
  MKL_NUM_THREADS = 1`; A6m's cuOpt gets the three REMOVED — they are also in the runner's own environment (set at
  `pair` import), so the runner drops them from the inherited environment (`Proc.env_unset`, `pair.launch_env`): GAMS
  ran cuOpt with them unset, and libcuopt links libomp. `run.json` records `env` and `env_unset` per process. Every
  other arm's child environment is pinned byte-identical to the one before A6m (`tests/test_pair.py`).
- **Lane.** `pair.FULL_LANE` = lane `full`: X925 5–9, 15–19 and A725 0–4, 10–14 plus the GPU (`whole_box`); cuOpt is
  pinned to all 20 cores. A6m runs only there and that lane runs only A6m (`Lane.fits`, `pair.plan` and the pool's
  `check_layout` refuse every other pairing with a clear error), so `itrea pair --arm A6m` on the GPU lane `g0` is
  refused. The pool's `--layout full` is this one lane; it implies the serving check (both local model servers
  stopped, `--allow-serving` refused) — D-020 wants no foreign GPU process. The runner's own threads (`RUNNER_CORES`,
  the GPU-health sampler, the stop loop) share the lane's cores: accepted, recorded as `runner_cores` in `run.json`.
  The layout refuses `--mailbox redis` (the container's core 0 is inside the lane) and `--cuopt-gpu-peak-gb` (it would
  let the memory gate admit A6m below 81 GB); `pair.plan` refuses both on the whole-box lane as well. Each run records
  the box's 1-min load average before and after (`gates_before` / `gates_after`) and the hottest ACPI zone while it
  ran (`acpi_during`): A6m loads all 20 cores and the GPU.
- **Memory.** D-020: RMM limit 36 GB (`--gpu-mem-gb 36`), host `MemoryMax` 24 GB (`--cuopt-mem-gb 24`); GAMS's max RSS
  was ≤ 16 GB on the sample, 49 GB at most (nw04) over the 233. The pool refuses A6m with other values
  (`pool.check_a6m_memory`: the RMM limit is in the run hash, `MemoryMax` is not). The gate: 24 + 36 + 1 (one CUDA
  process) + 20 reserve = **81 GB** with the file mailbox (`--dry-run` prints it).
- **Identity.** `solver_settings` = {arm A6m, emphasis null, `gpu_mem_gb` 36, `cuopt_threads` 24, `mip_gap_zero`
  true}: its own run hash, never A6's. Only A6m's dict has `mip_gap_zero`, so every other arm's hash is unchanged
  (`tests/test_pair.py` pins A6 `6e35f122f659`, A2 `125d69c76695`, A0 `27f3171fac78`, A9@bo1 `f67354be775d`, A9@bo1s
  `62bf82b236f6`, A3x `1d12ad490dd0`).
- **Jobs and analysis.** `itrea make-jobs mipfeas-cal` writes D-020's 30 instances, seed 0, 600 s, set `mipfeas`,
  split `dev` (`jobs/mipfeas-cal-26.08.jsonl` was removed from `jobs/` when D-021's M4 replaced this calibration; the
  writer stays); `itrea build-runs` adds `pt_gams` / `pt_gams_launch` / `startup_offset_s` ([[Trace format and
  metrics]]); `itrea mipfeas-report` applies D-020's pass rule. A full-box A9 would need its own decision and its own
  layout entry (D-021 R9: only through the heat ladder, with Berk's yes).

## ITX-1: arms on F, the harness, named scopes and void causes (2026-10-11, [[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]])
**F** = the cores and GPU of A9's frozen layout, i.e. the GPU lane `g0`: X925 5–6, A725 1–3 and the GPU (Redis on core
0 and the runner on 4 / 10–14 are shared infrastructure outside F). The runner takes F from the lane (`opts.lane`),
never from literals.
- **A6F** = cuOpt alone pinned to all of F, `num_cpu_threads` = |F| (5 today); **A2F** = SCIP (inert plugin) on F's
  first X925 core and cuOpt on the other |F| − 1 cores (1–3, 6) with |F| − 1 threads, no exchange. Both lane
  `partner`, appended after A6m in the canonical arm order (the order of every earlier arm is unchanged). The cuOpt
  process takes every core of the lane its arm's other processes do not hold (`ProcSpec.f_share`,
  `pair.cuopt_cores`); a `cuopt_threads` option that contradicts |F| (or |F| − 1) is refused. Their `solver_settings`
  carry that thread count, so F's size is part of their run hash; under a harness F's cores themselves are too (the
  footprint, below): a changed F is another configuration (R5; #30's layout change repeats M2).
- **Harness** (the job-level field `harness`; `pair.resolve_harness`). Absent = the runner as before: no key, every
  existing run id / hash / argv / env / cores unchanged (an old-vs-new comparison of `plan()` over every committed job
  file × every earlier arm × eight option sets, 20,432 records, is byte-identical apart from the scope's `--unit` and
  its two properties, below).
  Present = a preset name (`"ext"`) or a dict `{"preset": "ext", <key>: value, …}` overriding the preset's keys:
  | key | ext | effect |
  |---|---|---|
  | `cuopt_gap_zero` | true | every cuOpt process gets `--mip-gap-zero` (R13) |
  | `cuopt_env_unset` | true | every cuOpt process gets OMP / OPENBLAS / MKL_NUM_THREADS removed (`env_unset`, R14), confined by taskset |
  | `cuopt_rmm_gb` | 24 | cuOpt's RMM limit, separate from the EA's CuPy pool (the EA keeps `gpu_mem_gb`, 12 GB; R29) |
  | `cuopt_host_gb` | 24 | cuOpt's host `MemoryMax` (instead of `--cuopt-mem-gb`) |
  | `cuda_cache` | true | every process gets `CUDA_CACHE_PATH=~/.cache/itrea/cuda` and `CUDA_CACHE_MAXSIZE=4294967296` (R15) |
  | `cuopt_memory_void` | true | a confirmed cap hit of cuOpt voids the run (`cuopt_memory`); false records it and keeps the run failed (M3) |

  The resolved dict and its **footprint** (`pair.harness_footprint`: F's sorted X925 / A725 cores for an arm on the
  GPU lane, SCIP's `solver_mem_gb` and the EA's `ea_mem_gb` where the arm has them) enter the run hash only when the
  field is present (non-EA arms: `solver_settings.harness` / `.footprint`, whose `gpu_mem_gb` is then cuOpt's RMM
  limit; EA arms: folded into the EA hash as `cuopt_threads` is). Refused on A6m (D-020's own settings), on test and
  screening jobs (a prereg-v2 item; `pool.check_registered_harness`) and with `--cuopt-gpu-peak-gb` (the gate counts
  the harness's caps). A job with a harness is quiet (both model servers stopped, checked before every run;
  `--allow-serving` refused). M3 uses `{"preset": "ext", "cuopt_rmm_gb": 40, "cuopt_host_gb": 40,
  "cuopt_memory_void": false}`, M1's env-only contrast `{"preset": "ext", "cuopt_env_unset": false}`, R13's control
  `{"preset": "ext", "cuopt_gap_zero": false, "cuopt_env_unset": false}` (only R13 + R14 differ from its treatment).
  `runs.parquet` carries `run_hash` and `harness`: an EA arm's `config_hash` is the EA hash with or without a harness.
- **Memory gate** (D-003's formula, file mailbox; Redis + 2 GB): A9 under ext 9 + 6 + 24 host + 12 + 24 GPU + 2 × 1 +
  20 = **97 GB** (99), A2F 9 + 24 + 24 + 1 + 20 = **78** (80), A6F 24 + 24 + 1 + 20 = **69** (71), M3's A6F at 40 / 40
  = **101**; A9 without a harness stays 73. `itrea pool --dry-run` prints the arithmetic per treatment.
- **Named scopes** (R27). Every process runs in `systemd-run --user --scope
  --unit=itrea-run-<sha1(run directory name)[:12]>-<role>-<epoch_ms>` (`pair.scope_unit`; the run directory's name is
  the run id, `<epoch_ms>` the plan's millisecond: unique per launch, so two lanes starting in the same second never
  ask systemd for one name — the first version used the configuration hash and the second, and collided) instead of an
  anonymous scope, so host tools can find the project's processes — thermal-watch's 98 °C stop pattern gains
  `itrea-run` ([other repository], a separate edit). Every scope gets `-p TimeoutStopSec=5s` (a `systemctl stop`
  kills within 5 s, not systemd's 90), cuOpt's also `-p KillSignal=SIGKILL` (it has no interrupt API and keeps the
  GPU busy after SIGTERM). Never part of a hash or of argv; `run.json` records `scope_unit` per process.
- **Void causes** (`pair.void_checks`, after every run; each found cause is recorded in `void_checks`, the first in
  this order voids a run that is not void already; what decides nothing is recorded in `void_info`; the pool requeues
  the block):
  | cause | when | pool |
  |---|---|---|
  | `thermal_watch` | thermal-watch's `HOT <n>C stopped:` line (≥ 98 °C) whose stopped list names one of this run's scope units, stamped between the first launch and the last exit + 30 s (after an external stop of a process the runner waits for the line until then) — never an arm crash (R27). Other HOT lines (`stopped: none`, other units) are recorded in `void_info.thermal_watch_lines` | as the thermal pause (`--max-thermal-requeues`) |
  | `gpu_cap` | a GPU run whose busy SM median is below `--cap-floor-mhz` (2,300), whose timed window overlaps a fan-governor clock-cap interval (its log: `clock cap … on` → `clock cap off` or a governor restart), or whose fan state reads `cap=1` with its own timestamp inside the timed window (else recorded only) (R25). The 2,000-MHz stuck-clock fault stays a GPU-health void that halts | rerun once, only after the fan reads AUTO; an **unknown lock** (low clock, `cap=0` before and after, no cap interval: e.g. left by a governor restart) halts at once |
  | `cuopt_memory` | harness with `cuopt_memory_void`: cuOpt's cap CONFIRMED — an RMM allocation error (error status 6 "Memory allocation failed", or an RMM out-of-memory exception, warm-up included) whose peak reached 0.8 × the limit, or a SIGKILL the runner did not send with the scope's `Result=oom-kill` or maxrss ≥ 0.95 × `MemoryMax` (R29). An RMM error below that is recorded (`void_info.rmm_error_below_cap`) and the run fails; with `cuopt_memory_void: false` (M3) the cap hit is recorded (`void_info.cuopt_memory`) and the run stays failed. A failure of SCIP or the EA, their caps included (`void_info.cap_kills`), stays an arm outcome | rerun once |
  | `external_kill` | any arm, harness or not: a SIGKILL of any process that the runner did not send, at no cap (as above) and not explained by a thermal-watch line (memwatch, the global OOM killer, someone's `kill -9`) | rerun once |
  | `compiled_on_clock` | harness with `cuda_cache`: an entry of the CUDA cache with mtime ≥ `t0_epoch` (the `index` file ignored), or the EA's CuPy cache grown after t0 — the same terms for both GPU codes (R15). Without a harness CuPy's `compiled_on_clock` is recorded only, as before | rerun once |

  A rerun block runs first and whole; a further void of the same cause skips it (pool status `skipped`, reason = the
  cause, its run ids) and the pool moves on; reruns are counted by cause (`counts.requeues`). **R26's breaker:** after
  `--max-heat-voids` (2) thermal or cap voids in a run set — one thermal pause counts once however many blocks it
  voided — no new block opens (open blocks finish) and the pool ends `halted` with the reason.
- **Gates in the pool** (R24, M2): a GPU run starts only while fan-governor is alive (`~/.local/state/thermal/fan-state`
  younger than 30 s) with `cap=0` (`--no-governor` for a host without it; a governor gone for 120 s halts — also while
  its rest flag holds the pool, since only the governor removes the flag); `--pause-c` lowers the watchdog's pause /
  void line (M2: 90; never above 92). The rest flag (D-009) and the 80 °C GPU start gate are unchanged; the CPU-only
  start gate is 86 °C (was 88; D-008's 10-06 addendum, D-021 R24).
- **Explicit order.** A block whose every job carries `position` (a permutation of 0 … k − 1) runs in that order
  instead of its Williams row — M2's AB / BA, since both of its blocks hash to the same row.
- **Jobs.** `itrea make-jobs itx-m1 … itx-m5, itx-r13` → `jobs/itx-*.jsonl` (sets `itx-m1`, `itx-m2`, `itx-m3`,
  `mipfeas` for M4, `itx-m5`, `itx-r13`; labels carry a per-set variant, so no run id repeats another set's; M4 keeps
  the plain `A6F` for `itrea mipfeas-report --arm A6F`). The skill `itrea-window` lists them with their memory needs.

## ITX-1 phase 2: the stop at the proven optimum, cuOpt's kill, the v1 jobs (2026-10-12, [[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]])
- **Three optional harness switches** (`pair.HARNESS_OPTIONAL`), false by default and present in the resolved
  harness only when true — so `ext`, a dict that sets them false and every phase-1 job resolve and hash as before (old
  vs new `plan()` over every committed job file of the base × every arm × nine option sets, 35,337 records, and the
  pool's 961 blocks: byte-identical, scope units and properties included; re-checked after the review fixes):
  | key | itx1 | effect |
  |---|---|---|
  | `ref_stop` | true | R12: the run ends at the proven optimum (below) |
  | `cuopt_kill_on_end` | true | after a definitive outcome every cuOpt process still running is SIGKILLed at once |
  | `scoped_hash` | true | the run hash takes only the harness keys that act on the arm's processes (`pair.hashed_harness`): an arm without a cuOpt process (A0, A3x) drops every `cuopt_*` key (`pair.CUOPT_ONLY_KEYS`); arms with cuOpt hash the whole harness. R33: "A0 stands if the SCIP harness hash is unchanged at the tag" — a change to cuOpt's settings before prereg-v2 leaves A0's run ids, and so its runs, standing. run.json `config.harness` keeps the whole resolved harness; `solver_settings.harness` is the hashed part |

  Preset **`itx1`** = ext + all three: the v1 tier jobs (`make-jobs itx-v1`) and A0's external jobs (`itx-a0`).
- **R12, `ref_stop`** (`pair.RefMonitor`, `pair.ref_reached`, `pair.check_ref_trigger`). Eligible: an instance under
  `data/miplib` whose MIPLIB solu line is `=opt=` (`pair.miplib_stop_ref`; supportcase22's `=best=`, PGLib-UC and ML4CO
  never; tests inject a toy's optimum through `PairOptions.ref_lookup`). Before t0 the runner reads z* and the
  objective (c, offset) of its own parse; no process receives z*. From t0, every `ref_poll_s` (0.5 s), it reads what
  every process appended to `<role>.incumbents.part` (mapped by `<role>.names.json` onto its variable order — a role
  whose names it cannot map is left out with the reason), recomputes obj = c·x + offset from each stored vector and
  stops the run at the first record stamped before T with GAMS's gamma round((obj − z*)/max(|obj|, |z*|, 1), 6) <= 0
  **and** Berthold's gap <= 1e-9: `stop` kind `ref_reached`, `stop` written to the mailbox, cuOpt SIGKILLed under
  `cuopt_kill_on_end`. "First" is by the time stamp over every process, not by the order the roles are read (review
  fix 2026-10-12): each poll keeps every role's first qualifying record (its candidate) and, while no trigger is set,
  takes the earliest (ties: the earlier role of the arm) — in A9 cuOpt's record of z*, the EA's re-record of the same
  vector and SCIP's re-record of the EA's post usually land in one poll, and the trigger is cuOpt's, as M0's replay
  judges. Once set it never changes; after the stop `drain()` reads what was left into the candidates. Each `.part` file is opened once and the descriptor kept (the process unlinks the file after
  writing its `.npz`; the descriptor still reads every byte). After the processes ended (a killed one's `.part`
  recovered into its `.npz`), the triggering record is checked by the independent exact checker with the other
  library's parse, and its exact objective must meet the stop too; otherwise the run is **void `ref_rejected`** (after
  `compiled_on_clock` in the void order) and the pool reruns its block once — whole, first — with the stop off: the
  job then carries `ref_stop_off` (a reason string the pool injects; never in a job file; outside the run id; kept on
  resume by reading the set's `void.jsonl`). Processes killed after the stop are `stopped_by_runner`, so the run is
  completed and every metric holds its last checked incumbent to T (M0's replay: `pt_gams` unchanged, P and `pt_mf`
  within 1e-10 on every recorded run and GAMS trace). run.json `ref_stop`: `enabled` / `reason` (off: the attempt's
  `ref_stop_off`, or not eligible), `z`, `kind`, `source`, `sense`, `gap_tol`, `poll_s`, the records read per role
  (`roles.<role>.records / last_t / error`), `trigger` (role, record k, t, obj recomputed, obj reported, gamma, gap,
  `detected_t`), `candidates` (every role's first qualifying record by t, `after_trigger` for one read only after the
  trigger was chosen) and `check` (ok, reason, producer, reader, feasible, obj_recomputed, gamma_exact, gap_exact,
  checker_version); None when the harness has no `ref_stop`.
- **`cuopt_kill_on_end`.** A definitive outcome (`pair.definitive_stop`) is `ref_reached` or a `partner_end` whose
  `stop.proven` is true (a process ended with exit 0 and an exit reason in DEFINITIVE — not a crash): every cuOpt
  process still running gets SIGKILL at once (`killed` "SIGKILL (definitive outcome)", `stopped_by_runner`), instead of
  `kill_after_s` (60 s) + 10 s after its stop (M1 measured ~70 s of GPU load after SCIP's proof). Without the switch the
  runner behaves as before. `stop.proven` is recorded on every `partner_end`.
- **Job `priority`** (`pool.order_blocks`): when every job of a file carries an integer `priority`, the blocks run in
  increasing priority, each group in the usual seeded order with strata interleaved; some with and some without, or
  two values in one block, is refused. `itx-a0`: H_cu ∪ W_SCIP unseen first; `itx-m6`: three loads, the three measured
  runs, then the other loads.
- **R33's gate** (`pool.check_frozen`): an external job (split `ext`) of an arm with the EA (A9) needs the tag
  `prereg-v2` as an ancestor of HEAD — no A9 outcome on an unseen instance before it — and the configuration
  registered with it (review fix 2026-10-12): `configs/frozen-v2.yaml` (`pool.FROZEN_V2_CONFIG`, the C1 freeze) must
  exist, be byte-identical to the file in the tagged commit (`record_env.file_at`), and every such job's base
  configuration hash must equal its hash, with no overrides or ablation. Every external job — A0, A6F, A2F and A9 —
  needs a clean, committed tree under `src/` and `configs/`. A0 and A6F's external jobs need prereg-v1 only. The
  analysis takes only runs that carry the registered settings (`itx_report.registered_fault`).
- **Jobs.** `itrea make-jobs itx-v1 --tier T1|T2|T3|T4|T5 [--cuopt 26.10] --out FILE` (sets `itx-t1` … `itx-t5`,
  `-c<tag>` for another cuOpt version; labels `A9@v1c2610`, `A6F@v1c2610`, `A2F@v1c2610`; seed 2000, T4 2001; split
  `ext`; written only once 26.10 is in; A9 on `configs/frozen-v2.yaml` — a tier with A9 is refused while that file does
  not exist; `--a9-config` names another file for a dry run, which the pool refuses), `itx-a0` (`jobs/itx-a0.jsonl`,
  A0@v1 × 230,
  `--layout screen --cpu-lanes 4`), `itx-m6` / `itx-m6a` (`jobs/itx-m6.jsonl`, `jobs/itx-m6a.jsonl`, `--layout screen
  --cpu-lanes 4` / `1`; the screen layout's single-core X925 lanes need no E0 flag, the pair layout's `--cpu-lanes`
  does). The instance sets come from `datasets/mipfeas_strata.csv` (`itrea.analysis.itx_strata`). Memory under itx1
  = under ext: A9 97, A2F 78, A6F 69 GB; A0 9 GB per lane plus the 20-GB reserve once (4 lanes: 56 GB).
