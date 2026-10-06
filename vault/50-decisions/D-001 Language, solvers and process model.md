# D-001 Language, solvers and process model (2026-10-05)

**Status:** decided by AgentRA (Claude Code), open to a superseding decision on evidence.

## Language: Python 3.12 + CuPy (GPU) / NumPy (CPU), one code path
- The EA is written against an array namespace `xp` that is either CuPy or NumPy (`itrea.backend`). The **same
  algorithm runs on the CPU** — that is both the test backend and an experimental control (arm A4: does the GPU, rather
  than the algorithm, carry any effect?).
- Sparse algebra: `cupyx.scipy.sparse` / `scipy.sparse` (cuSPARSE SpMM on the GPU). No custom kernels in the first
  version; a CuPy `RawKernel` for a feasibility-jump style local search (one block per individual, incremental updates)
  is a planned optimisation, measured against the vectorised version before adoption (issue tracker).
- Why not C++/CUDA: every compile on this box runs under the capped-build rule and the solver bindings
  (PySCIPOpt, highspy, cuOpt) and the predecessor's harness, metrics and checker are Python. Why not PyTorch: no
  NumPy twin for the CPU control; CuPy mirrors NumPy/SciPy one-to-one.
- CuPy compiles elementwise kernels with NVRTC on first use (small, cached in `~/.cupy/kernel_cache`). The runner warms
  every kernel **before** the shared clock starts.

## Solvers
| role | solver | why |
|---|---|---|
| **primary partner** | SCIP 10 (PySCIPOpt 6.2.1), 1 thread | mid-solve injection through a heuristic plugin (`trySol`) is proven (optopt `_XchgHeur`, no crashes over 160 runs); exposes its solution pool and dual bound |
| replication partner | HiGHS 1.15.1, 1 thread | `kCallbackMipUserSolution` allows mid-solve injection — **unproven from Python**; a round-trip test must pass before any HiGHS arm is run |
| control / warm start only | NVIDIA cuOpt 26.08 (GPU) | `SetSolutionCallback` disables cuOpt's presolve (optopt X1), so cuOpt is the equal-GPU race control and a launch-time warm start, never a two-way partner |

cuOpt 26.08.0 is the same installation as optopt's: uv's offline cache had lost the NVIDIA index metadata, so the
28 distributions of cuOpt's dependency closure were **hard-linked from optopt's venv** following each distribution's
RECORD file (2026-10-05 23:2x; no network). On a networked machine `uv pip install -e ".[cuopt]"
--extra-index-url https://pypi.nvidia.com` installs the same versions. The cuOpt arm still runs as its own process
(`--cuopt-python` stays available for a separate interpreter).

## Process model
Two processes per pair run, started by the runner, each reading the instance **before** a shared start barrier:
- **solver process** — pinned to one Cortex-X925 core, systemd scope with `MemoryMax`, `MemorySwapMax=0`;
- **EA process** — the GPU (CuPy memory-pool limit) plus one pinned core for its host loop, same systemd caps.

They exchange solutions through a **mailbox directory on `/dev/shm`** (atomic `os.replace` of `.npz` messages, one stat
per poll), carried over from optopt's `xchg.py` and extended to batches, dual bounds and provenance. Vectors travel
in one canonical variable order (the MPS column order as read by HiGHS); each side maps by variable name once, before
the clock. Each process writes its own trace with times relative to the shared start; the pair's "portfolio"
trajectory is the best of both at every moment (same definition as optopt's race).

## Consequences
- GPU memory is system RAM on GB10, and **a systemd `MemoryMax` scope does not charge it** (measured 2026-10-05: a
  600 MB CuPy allocation succeeded under `MemoryMax=300M`). The CuPy pool limit is the only real cap on the EA's GPU
  memory; the runner's memory gate counts it separately. See [[D-003 Resources, memory and thermal rules]].
- One GPU, one GPU arm at a time (concurrent GPU runs would confound timing).

## Verified API facts (build, 2026-10-06 00:xx–01:xx)
- **HiGHS 1.15.1 user solutions:** `h.cbMipUserSolution.subscribe(fn)`; inside `fn`, `e.data_in.setSolution(x)` with a
  DENSE float64 vector of length num_col in HiGHS column order (the canonical order). Classes are
  `highspy.cb.HighsCallbackInput` / `HighsCallbackOutput`. A user solution outside HiGHS' **absolute**
  `mip_feasibility_tolerance` (1e-6) is dropped silently, and a feasible but non-improving one is ignored without a
  signal; the verdict is read from the next callback's `mip_primal_bound`. Use `h.run()` (not `solve()`, which adds a
  thread); a second `Highs` object that ran with default threads in the same process makes a later `threads=1` run
  return kError.
- **cuOpt 26.08:** allocates two fixed GPU buffers of 2,399,998,904 bytes each regardless of instance size — it fails
  ("Memory allocation failed", status 6) under an RMM limit of 0.5 GB and of 4 GB. The equal-GPU-memory limit of
  A2/A3 is therefore set from cuOpt's measured need (see the journal), not from the EA's default. `num_cpu_threads`
  must be ≥ 2.
- **SCIP (PySCIPOpt 6.2.1):** the exchange heuristic fires in PRESOLVING with BEFOREPRESOL|DURINGPRESOLLOOP and
  accepts `createOrigSol(heur=self)` + `trySol` there; an inert plugin (one clock read per call) cost +0.5 % on a
  3.4 s toy (225 calls, same nodes, same optimum).
- **GPU feasibility jump (synthetic set cover n = 20k–50k, P = 256):** 7.1–8.2 M moves/s on the GB10 vs 0.54–0.57 M
  on one Cortex core with the Numba reference (≈ 12–15×); one GPU block is ~13× slower per individual than a CPU core —
  the gain comes only from running many individuals at once.
