# Process interface — how a pair run is put together (contract, 2026-10-05)

A **run** = one (instance, arm, seed, budget). The runner (`itrea.experiments.pair`) creates a run directory
`traces/raw/<set>/<run_id>/` and a mailbox directory `/dev/shm/itrea/<run_id>/`, writes `order.npy` (canonical names,
HiGHS column order), launches the arm's processes, waits until every process wrote `ready.<role>`, releases the
barrier (`start` with `t0`), waits `budget + grace`, writes `stop`, waits for exits, **checks every exit code** (optopt
X1: a dead SCIP logged "ok"), records resources and merges the trajectories.

`run_id = <instance-stem>__<arm>__s<seed>__<budget>s__<cfg-hash>`.

## Process kinds (all: `python -m <module> …`; exit 0 = normal end, 1 = exception, 2 = bad arguments)
| module | role in mailbox | arguments |
|---|---|---|
| `itrea.ea.run` | `ea` | `--instance P --config Y [--set k=v …] --seed S --budget T --run-dir D [--mailbox M] [--run-id R (default $ITREA_RUN_ID)] [--role-name N] [--no-barrier] [--barrier-timeout S] [--sol-out F] [--arm A] [--backend cupy\|numpy] [--gpu-mem-gb G]` |
| `itrea.solvers.scip_run` | `solver` | `--instance P --seed S --budget T --run-dir D [--mailbox M] [--no-barrier] [--sol-out F] [--mem-mb MB] [--recv] [--send] [--inject-band B] [--inject-k K] [--poll-s S] [--inert] [--no-plugin] [--emphasis E] [--run-id R] [--role-name N] [--barrier-timeout S] [--arm A]` |
| `itrea.solvers.highs_run` | `solver` | same as SCIP (`--recv` requires the user-solution callback test to pass) |
| `itrea.solvers.cuopt_run` | `cuopt` | `--instance P --seed S --budget T --run-dir D [--mailbox M] [--run-id R] [--role-name N] [--no-barrier] [--sol-out F] [--cpu-threads N (≥ 2)] [--gpu-mem-gb G] [--send] [--no-warmup] [--arm A]` — run under `--cuopt-python` (an interpreter with cuOpt 26.08); never receives solutions (a SetSolutionCallback disables cuOpt's presolve) |

The runner passes `--run-id` and, in every non-exchange SCIP arm, `--inert` (plugin installed, never polls — the
harness-matched control); cuOpt gets `--gpu-mem-gb <same limit as the EA pool>` and `--cpu-threads 3`; A0* gets
`--emphasis`; the EA gets `--config <run_dir>/config.yaml` (the effective configuration, no `--set`).
`--recv`: accept the partner's solutions (SCIP: heuristic plugin + `trySol`; HiGHS: user-solution callback).
`--send`: publish own incumbents (+ pool within `--inject-band`, at most `--inject-k`), dual bound and the root LP
solution to the EA. Arms without exchange omit both. `--role-name` lets two solver processes share a run dir (A1:
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
| H0/H2/H3 | as A0/A2/A3 with highs_run |
| ablations | A3 with `--set` overrides (`ranks.x=0`, `ranks.y=0`, `exchange.to_solver=false`, …) and the matching solver flags |
