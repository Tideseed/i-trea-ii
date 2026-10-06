# AGENTS.md — companion for AI agents (and humans)

Companion to the research project **I-TREA II** (Interactive Three Ranks Evolutionary Algorithm, part II) by Berk
Orbay (berk.orbay@tideseed.com). Repository: `Tideseed/i-trea-ii` (private while the study runs). Paths are relative
to the repository root; the code is the Python package `itrea` under `src/`. `CLAUDE.md` points here: this file holds
every rule for agents working in the repository.

**Status (2026-10-06 01:4x): built and tested on toy models; no commit, no tag, no experiment, no results.** The
toy-scale suite passed (413 passed, 2 skipped at 01:2x); the next steps are the first commit, the tag `prereg-v1` and
the SCIP-only screening pass (newest checkpoint in `vault/90-checkpoints/`). Every hypothesis below is
pre-registered, not a finding. Do not write, in any file, that I-TREA II "beats", "improves on" or "matches" anything
until the corresponding experiment note in `vault/30-results/` says so with its numbers.

## Setting
- Machine: NVIDIA DGX Spark — GB10 (10 × Cortex-X925 @ 3.9 GHz = cores 5–9, 15–19; 10 × Cortex-A725 @ 2.8 GHz = cores
  0–4, 10–14), Blackwell GPU, 121 GiB unified memory shared by CPU and GPU. Ubuntu 24.04.4, kernel 6.17.0-1031-nvidia,
  driver 580.173.02, CUDA 13.0. The machine is shared with other sessions and with two local LLM servers.
- Software: Python 3.12.3; NumPy 2.4.6, SciPy 1.18.1; CuPy 14.2.0 (GPU) and NumPy (CPU) as one code path
  (`itrea.backend`); solvers SCIP 10 (PySCIPOpt 6.2.1, primary partner), HiGHS 1.15.1 (replication partner, after its
  user-solution round-trip test), NVIDIA cuOpt 26.08 (control arm and launch-time warm start only — its
  `SetSolutionCallback` disables its presolve); Numba 0.64.0 (the CPU feasibility-jump reference); redis-py 8.1.0 for
  the exchange platform; pandas 3.0.3 / PyArrow 23.0.1 for the derived tables. All pinned in `pyproject.toml`. One entry
  point: `itrea <command>` (`src/itrea/cli.py`; `itrea --help`).
- Instances (public, on disk): MIPLIB 2017 benchmark, PGLib-UC unit commitment, ML4CO item placement
  (`vault/10-design/Datasets.md`). The study uses a frozen pool of **hard** instances chosen on SCIP alone
  (`vault/10-design/Hard instance pool.md`).
- Metric: the primal integral **P(T)** ∈ [0, 1] (Berthold 2013), T = 300 s, built only from incumbents that pass the
  independent exact checker; definitions in `vault/10-design/Trace format and metrics.md`.
- Predecessor: `berkorbay/optopt` (public, MIT, doi:10.5281/zenodo.22947442). Its CPU evolutionary partner beside SCIP
  (experiment X2) found no gain at 60 s and lost to SCIP ‖ SCIP; its strongest result was the SCIP/HiGHS ‖ cuOpt race.
  Code reused from it carries the line "adapted from berkorbay/optopt@4dd48395, MIT".

## The question and the pre-registered hypotheses
Berk's brief (2026-10-05) and main claim (2026-10-06, `vault/50-decisions/D-007`): *"a GPU-based evolutionary
algorithm metaheuristic coupled with an exact solver can beat either single solution."* I-TREA — Berk's master's
research (Orbay 2011, METU M.S. thesis, open access, `orbay2011itrea`; thesis
manuscript, unpublished) — is the starting point; the principle carried over is **keeping good infeasible solutions**
(Rank 0). Part I reserved a share of the population for them as a floor; part II caps it, as Berk's brief says.

Pre-registration: `vault/50-decisions/D-002 Pre-registration — hypotheses, arms, analysis.md`, **v3**. It becomes
binding at the git tag `prereg-v1` (before the first timed run on a candidate instance); any later change is a new,
dated decision. Summary — **all pre-registered, none tested**:

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
| **Rank 1 — feasible good** | the best feasible individuals | ⌈y · min(#feasible, P)⌉, y = 10 % | protected every generation; equal-key twins leave Rank 1 |
| **Rank 2 — rejects** | everything else | the rest of P | generational by default: only this generation's offspring, injected and immigrant individuals compete (the old Rank 2 dies); non-dominated sorting in (key, violation) with crowding distance |

Tournament order Rank 1 < Rank 0 < Rank 2. Before any incumbent exists the population is in a feasibility phase
(ranked by violation). The solver is the "interactive" counterpart: it supplies its incumbent, very good pool
solutions, its proven bound and its root LP, and receives the EA's better incumbents as new starting points. Beside the
ranks the EA keeps a **frontier**: an archive of diverse near-optimal feasible solutions, non-dominated in (objective,
distance from the incumbent) (`vault/50-decisions/D-005`; exploratory metrics only).

## Map
Status: **core** = written before the design critique of 2026-10-05 and revised after it; **built** = written by the
build agents on 2026-10-06 (00:0x–01:2x) and covered by the toy-scale suite. Caveat: `ea/ranks.py`,
`ea/population.py`, `ea/evaluate.py`, `ea/frontier.py`, `exchange/*`, `trace.py` and `problem.py` have no test file of
their own — they are exercised through the engine, EA-process and runner tests (default settings); check a
non-default branch before relying on it (`vault/80-reviews/2026-10-05 design critique — response.md`).

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
| `src/itrea/ea/repair_clock.py` | the repair clock's controller: one lane-step horizon per generation from each horizon's measured gain per second (`repair_clock`, off by default; shadow mode logs only) — 2026-10-06, CPU-tested, not yet run on an instance | built |
| `src/itrea/ea/engine.py` | the generation loop, exchange in/out, genealogy, frontier updates | built |
| `src/itrea/ea/run.py` | EA process: `python -m itrea.ea.run …` | built |
| `src/itrea/exchange/mailbox.py` | file mailbox on `/dev/shm` (fallback transport) | core |
| `src/itrea/exchange/redis_mailbox.py` | Redis transport and shared pool; `open_mailbox(spec, role)` | core |
| `src/itrea/solvers/common.py` | what the solver processes share: arguments, barrier, recorder, exchange accounting, exit codes | built |
| `src/itrea/solvers/scip_run.py` | SCIP process; injection through a heuristic plugin (`trySol`) | built |
| `src/itrea/solvers/highs_run.py` | HiGHS process; user-solution callback | built |
| `src/itrea/solvers/cuopt_run.py` | cuOpt process; never receives solutions; RMM limit | built |
| `src/itrea/experiments/gates.py` | lease, memory gate, thermal gate, run lock, watchdog | core |
| `src/itrea/experiments/gpu_health.py` | per-second SM clock, temperature and foreign-GPU sampling; block voiding | built |
| `src/itrea/experiments/exchange_ctl.py` | the `itrea-redis` container: up, status, flush one run's keys, down | built |
| `src/itrea/experiments/record_env.py` | versions, hardware, git state for every run record | built |
| `src/itrea/experiments/pair.py` | one run: launch the arm's processes, start barrier, stop, every exit code checked, `run.json` | built |
| `src/itrea/experiments/pool.py` | a job file under every D-002/D-003 control: Williams arm order per block, blocks never split, gates per run | built |
| `src/itrea/experiments/make_jobs.py` | job files: `screen1`, `screen2`, `e1`, `e4`, `confirm`, `smoke` (`itrea make-jobs …`) | built |
| `src/itrea/benchmarks/hardset.py` | the hard pool: screening candidates, the stage-2 rule, strata, family split, freeze (`itrea screen-candidates`, `screen-stage2`, `pool-freeze`) | built |
| `src/itrea/analysis/metrics.py` | primal gap, P(T), primal-dual integral, times to target (adapted from optopt) | built |
| `src/itrea/analysis/check_solutions.py` | independent exact check of every stored incumbent (adapted from optopt) | built |
| `src/itrea/analysis/build_runs.py` | run directories → `datasets/runs.parquet`, `datasets/incumbents_checked.parquet` | built |
| `src/itrea/analysis/stats.py` | d_i, Wilcoxon, bootstrap, decision rule, censored endpoints | built |
| `src/itrea/analysis/lineage.py` | depth-1 matched Rank 0 odds ratio, Mantel–Haenszel (descriptive) | built |
| `src/itrea/analysis/prereg.py` | the pre-registered analysis of D-002 v3 from `runs.parquet` (`itrea prereg`) | built |
| `src/itrea/analysis/reproduce.py` | re-derive `datasets/` and `paper/generated/` and compare (`itrea reproduce`) | built |
| `src/itrea/cli.py` | the `itrea` entry point: one table maps every command to its module | built |
| `configs/` | `itrea2-default.yaml` (all defaults, generated from `EAConfig()`; hash `9ac4a93bf93b` on 2026-10-06), `itrea1-faithful.yaml` (a record of part I's choices, not a planned comparison), `itrea2-adaptive.yaml` (development: the self-adaptive repair strategy + the EA's own LP point), `ablations/*.yaml`; `tests/test_configs.py` keeps the default file equal to the code | core |
| `tools/serving-window.sh` | stop / restore the two local LLM servers for a window | core |
| `tools/bench_ls.py` | local-search throughput, GPU vs CPU reference (`itrea bench-ls`) | built |
| `datasets/` | derived tables (committed); `hardness_from_optopt.csv` is background only | — |
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
- **Serving window.** Confirmatory runs need both local LLM servers stopped (vLLM `ornith-vllm` :8094, llama.cpp
  `llama-qwen38` :8092; Berk's yes, 2026-10-05). Stop and restore only with `tools/serving-window.sh stop|restore`,
  schedule the restore on a timer before starting, and verify after restore: `is-active` **and** `is-enabled`, health
  endpoints answer, canaries resumed, lease released.
- **Memory gate.** Σ host `MemoryMax` + Σ CuPy pool limits + 1 GB non-pool overhead per CUDA process + cuOpt's RMM
  limit + the Redis cap (2 GB) + 20 GB reserve ≤ MemAvailable at launch; the watchdog stops the run set below 18 GB.
  `MemoryMax` does **not** charge GPU memory on GB10.
- **One GPU arm at a time.** Timed solvers pinned one per X925 core; the partner footprint (GPU + the same 3 A725
  cores + the same GPU memory limit, **12 GB**: the EA's CuPy pool = cuOpt's RMM limit, `EAConfig.gpu_mem_gb`, set by
  cuOpt's measured need of ~10.5 GB, D-003) is identical for cuOpt and the EA. `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` in every process.
  Never run analysis with default thread counts beside timed runs.
- **Thermal.** Start a run only below 80 °C (ACPI maximum); pause the run set at ≥ 92 °C, resume below 80 °C.
- **GPU health.** A GPU run with a median SM clock below 2,000 MHz or foreign GPU activity above 2 % of its budget
  voids its whole block (all arms of that instance × seed) and halts the run set.
- **Frozen configuration.** Test-set instances run only with the configuration whose hash equals `frozen-e2`'s, with
  `prereg-v1` an ancestor of HEAD. Tuning happens on the tuning set only, within the logged E4 budget.
- **Every incumbent checked.** Every incumbent vector of every process is stored and checked by the independent exact
  checker; only checked incumbents enter a metric. Exit codes of every process are checked; a crash is kept and
  truncated, never silently re-run.
- **Clock rule.** Before the barrier only parse, imports, CUDA context and kernel warm-up on a synthetic problem; any
  LP, presolve or evaluation of the instance's population after `t0`, in every arm.
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
- **Record everything.** Journal entries (`vault/20-journal/`, append-only, CEST), decision records D-NNN, a result
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

## Reproduce
```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev,plots,gpu]"   # see README for cuOpt
pytest -q                     # toy-scale correctness tests (GPU tests skip without CuPy; -m "not integration" is faster)
```
Once runs exist: `itrea build-runs`, `itrea prereg`, then `itrea reproduce` re-derives every table in `datasets/` and
`paper/generated/` from the traces and lists any difference (skill `itrea-reproduce`). Re-running experiments needs the public instances in `data/` and the
job files in `jobs/`; wall-clock numbers depend on the machine (`vault/10-design/Compute resources.md`).
