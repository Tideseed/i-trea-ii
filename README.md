# I-TREA II

**A GPU evolutionary algorithm coupled with an exact MIP solver.** I-TREA II (Interactive Three Ranks Evolutionary
Algorithm, part II) runs a GPU-resident evolutionary algorithm beside a mixed-integer programming solver — SCIP as the
primary partner, HiGHS as a replication partner, NVIDIA cuOpt as a control — and lets the two exchange solutions in
both directions while they run: the solver's incumbent, very good pool solutions, proven bound and root LP flow into
the population, and the population's better incumbents flow into the solver as new starting points. The population
is organised in three ranks: Rank 0 holds infeasible solutions whose objective beats the best feasible one (at most
x % of the population), Rank 1 protects the best y % of the feasible solutions, and Rank 2 holds the rest. Beside the
ranks the EA keeps a frontier of diverse near-optimal feasible solutions. The project succeeds two pieces of the
author's work: his master's research I-TREA (Orbay 2011, M.S. thesis, Middle East Technical University, open access
at hdl:11511/20509; thesis) — an interactive three-rank evolutionary
algorithm for a bi-objective facility relocation problem — whose idea of keeping good infeasible solutions it carries
over (part I reserved a share of the population for them as a floor; part II caps it), and the optopt study ([`berkorbay/optopt`](https://github.com/berkorbay/optopt)), whose CPU
evolutionary partner beside SCIP found no gain at 60 s and whose verdict named the changes tested here: a GPU
population, longer budgets and instances where the solver's own heuristics struggle.

**The claim under test** (the author, 2026-10-06, vault `D-007`): *a GPU-based evolutionary algorithm metaheuristic
coupled with an exact solver can beat either single solution.* Pre-registered as (vault `D-002` v3): on hard MIP
instances, does the coupled system find good feasible solutions sooner — a lower primal integral over 300 s — than
**the solver alone and the EA alone** (both must hold), and does the exchange add anything beyond running the two side
by side? Secondary controls: SCIP ‖ cuOpt on the same GPU, SCIP ‖ SCIP, tuned SCIP; mechanism: the EA alone with and
without Rank 0.

None of the parts is new, and no "first" is claimed (`vault/40-literature/Prior art.md`): heuristics and
branch-and-bound exchanged incumbents both ways in Puchinger, Raidl & Gruber (2005) and Nwana et al. (2005); ReXi runs
local-search workers beside SCIP over a shared pool; CHAP pools GPU and CPU heuristics' feasible and infeasible
candidates; cuOpt itself runs a GPU population beside its own branch-and-bound; IDEA reserved a share of the population
for infeasible solutions. What is studied here is the combination: a GPU population coupled with an external, general
MIP solver, with a capped, incumbent-relative share of infeasible solutions.

## Status
**Built and tested on toy models (2026-10-06). No experiment has run; there are no results.** The whole pipeline is
written: configuration, problem reader, evaluation, three-rank survival, frontier, operators, feasibility-jump local
search (GPU kernel, single-core CPU reference, dense baseline), engine and EA process, the SCIP / HiGHS / cuOpt
processes, both exchange transports, the runner, job builder and pool with their safety gates, and the analysis
(exact checker, metrics, pre-registered statistics, lineage). The design was revised after a three-lens critique
(`vault/80-reviews/`). The test suite (toy models and synthetic problems only) passed 413 tests, 2 skipped, on
2026-10-06 01:2x. The repository has no commit yet; the hypotheses, arms, metrics and analysis become binding at the
git tag `prereg-v1`, before the first timed run on a candidate instance (the screening pass). Nothing here should be
read as a performance claim.

For AI agents and for anyone extending the work: start at [`AGENTS.md`](AGENTS.md); the research vault starts at
[`vault/00-index/Home.md`](vault/00-index/Home.md).

## Install
```bash
git clone https://github.com/Tideseed/i-trea-ii && cd i-trea-ii     # private while the study runs
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,plots]"          # CPU backend (NumPy/SciPy/Numba), HiGHS, SCIP, Redis client, tests, plots
uv pip install -e ".[gpu]"                # + CuPy for CUDA 13 (the GPU backend)
uv pip install -e ".[cuopt]" --extra-index-url https://pypi.nvidia.com   # + NVIDIA cuOpt 26.08 (control arms)
pytest -q                                 # toy-scale tests (GPU tests skip without CuPy; -m "not integration" is faster)
```
Versions are pinned in `pyproject.toml` to the ones used for the study: Python 3.12, NumPy 2.4.6, SciPy 1.18.1,
Numba 0.64.0, HiGHS 1.15.1, PySCIPOpt 6.2.1 / SCIP 10, pandas 3.0.3, PyArrow 23.0.1, redis-py 8.1.0, PyYAML 6.0.3,
CuPy 14.2.0 (`cupy-cuda13x`), cuOpt 26.08, pytest 9.1.1, Hypothesis 6.168.1. One entry point, `itrea <command>`, reaches
every tool (`itrea --help`; `src/itrea/cli.py`).

**On the author's machine the environment was built offline** (no network): every wheel came from uv's local cache;
cuOpt's dependency closure (28 distributions) was hard-linked from the optopt environment following each
distribution's `RECORD` file, because uv's cache had lost the NVIDIA index metadata; redis-py 8.1.0 came from the cache
without `hiredis` (the pure-Python parser is used). On a networked machine the commands above install the same
versions.

The exchange platform is a Redis 7 server of the project's own on 127.0.0.1:6390 (container `itrea-redis` from the
`redis:7-alpine` image, no persistence, 2 GB cap): `python -m itrea.experiments.exchange_ctl up|status|down`. Without
Redis, the file mailbox on `/dev/shm` offers the same API.

## Layout
| path | what |
|---|---|
| `src/itrea/` | the package: `cli` (the `itrea` command), `config`, `problem`, `backend`, `trace`, `paths`; `ea/` (evaluation, population, ranks, operators, feasibility-jump local search on GPU and CPU, continuous decoding, frontier, engine, EA process); `exchange/` (file mailbox, Redis); `solvers/` (SCIP, HiGHS, cuOpt processes); `experiments/` (gates, GPU health, Redis container control, environment record, runner, job builder, pool); `analysis/` (exact checker, metrics, run tables, pre-registered statistics, lineage, re-derivation); `benchmarks/` (toy models, hard-pool screening and freeze, reference values) |
| `configs/` | `itrea2-default.yaml` (every parameter at its default, generated; hash `9ac4a93bf93b`), `itrea1-faithful.yaml` (closest to part I; a record, not a planned comparison), `ablations/` (x0 — the H6 arm, y0, no_bound_demote, one-way exchange both ways, post_frontier); `tests/test_configs.py` checks them |
| `tests/` | correctness tests on toy models and synthetic problems (`-m gpu` needs CuPy, `-m solver` runs a real solver for seconds, `-m integration` launches real processes through the runner) |
| `tools/` | `serving-window.sh` (stop / restore the host's local LLM servers for an experiment window), `bench_ls.py` (local-search throughput, `itrea bench-ls`) |
| `jobs/`, `traces/`, `datasets/` | job files; run outputs (raw traces are not committed, resource records are); derived tables |
| `data/` | instances (not in git; symbolic links to the public sets, see `vault/10-design/Datasets.md`) |
| `paper/` | LaTeX source of the paper (skeleton; no results) |
| `vault/` | research vault (Obsidian): journal, design notes, decisions, results, literature, incidents, reviews, checkpoints |
| `tutorials/` | tutorials (planned) |
| `.claude/skills/` | repository skills for AI agents (window, journal, result note, new experiment, literature, reproduce) |

## How a run is assembled
A run is one (instance, arm, seed, budget). The runner (`itrea.experiments.pair`; job files run through
`itrea pool`, which applies every gate) creates a run directory
`traces/raw/<set>/<run_id>/` and a fresh exchange space — a directory `/dev/shm/itrea/<run_id>/` or the Redis key space
`itrea:<run_id>:*` (`redis://127.0.0.1:6390/0?run=<run_id>`) — writes the canonical variable order, and launches the
arm's processes, each one pinned, memory-capped and single-threaded:
```bash
python -m itrea.solvers.scip_run --instance P --seed S --budget T --run-dir D --mailbox M --recv --send
python -m itrea.ea.run --instance P --config configs/itrea2-default.yaml --seed S --budget T --run-dir D --mailbox M
```
Each process reads the instance and reports ready; the runner releases a start barrier with a shared `t0` on the
system-wide monotonic clock, waits for the budget, writes `stop`, checks every exit code, and writes `run.json` (cores,
caps, temperatures, GPU health, co-running services, git revision) with the merged trajectories. Every process stores
every incumbent vector, so the independent exact checker can verify each one before it enters a metric. Contract:
`vault/10-design/Process interface.md`; file formats and metrics: `vault/10-design/Trace format and metrics.md`.

`itrea.exchange.redis_mailbox.open_mailbox(spec, role)` returns either transport behind one API (`to_solver`,
`to_ea`, `info`, `lp` channels, barrier, stop flag, verdicts); Redis adds a shared solution pool and a history of every
exchange. Which transport the confirmatory runs use is decided by a pre-registered overhead measurement (vault `D-006`).

## Reproducibility plan
- **Pre-registration.** Hypotheses, arms, instance-pool rules, metrics and analysis are written before any run
  (vault `D-002`), committed and tagged `prereg-v1`; the pool is frozen as `pool-v1`, the configuration as `frozen-e2`.
  The pool refuses test-set jobs unless the configuration hash equals `frozen-e2`'s and `prereg-v1` is an ancestor of
  HEAD, and other non-smoke jobs without the tag. Later changes are dated decisions; results under the old plan are
  still reported.
- **Exact definitions.** Job files in `jobs/` define every run (instance, arm, seed, budget, configuration hash);
  seeds are fixed (screening 900/901, tuning 0–99, confirmatory 1000/1001).
- **Checked numbers.** Every reported objective comes from an incumbent re-checked with another library in exact
  arithmetic; every crash and void is reported.
- **Re-derivation.** `itrea build-runs` turns the run directories into `datasets/runs.parquet` and
  `datasets/incumbents_checked.parquet`; `itrea prereg` computes every pre-registered comparison from them; `itrea
  reproduce` re-derives every table in `datasets/` and `paper/generated/` from the traces and lists any difference. The
  raw traces will be published as a versioned archive with checksums, as optopt did.
- **What a re-run can reproduce.** Results are wall-clock measurements on one NVIDIA DGX Spark; GPU arithmetic is not
  bit-reproducible (atomic scatter order). Expect the conclusions, not the exact numbers, to transfer to other hardware.

## Safety rules on a shared machine
The study runs on a shared DGX Spark (unified memory, two local LLM servers, other sessions). The runner enforces the
rules in vault `D-003`: a box-window lease before any timed run; a run-set memory gate that counts GPU memory
explicitly (a systemd `MemoryMax` does not charge GPU allocations on this machine), with a 20 GB reserve and a live
watchdog; one GPU arm at a time, with the same GPU memory limit for the EA's pool and cuOpt (12 GB, set by cuOpt's
measured need); pinned single-threaded processes; a thermal gate (start below 80 °C, pause at 92 °C);
GPU-health voiding (stuck-low-clock fault, foreign GPU activity); the LLM servers stopped and restored only through
`tools/serving-window.sh`; no downloads; toy models only in tests. See `AGENTS.md` → "Rules for new experiments".

## Credits
Author: Berk Orbay (berk.orbay@tideseed.com). The design and the code are developed with **AgentRA** (Agent Research
Assistant) — AgentRA is mainly Claude Opus 5.5, working through Claude Code — under the author's direction. Code
adapted from the predecessor carries the line "adapted from berkorbay/optopt@4dd48395, MIT".

## License
[MIT](LICENSE). Third-party files keep their own terms.
