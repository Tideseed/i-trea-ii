# D-003 Resources, memory and thermal rules (2026-10-05)

Carried from the host's rules and the predecessor's incidents (optopt D-003; host memwatch sheds on 2026-09-23,
09-24, 09-25, 09-27, 09-28; the thermal shutdown on 2026-10-01). **The runner enforces these; documentation alone
is not a control.**

| rule | enforced by |
|---|---|
| Claim the box before any timed run: `box-window claim --who i-trea-ii --why … --until … [--stops unit] [--mem GB]`; release after | runner refuses to start without a held lease (override flag for toy smoke tests only) |
| **Run-set memory gate:** Σ host `MemoryMax` + Σ CuPy pool limits + Σ non-pool GPU overhead per CUDA process (CUDA context, cuSPARSE/cuBLAS handles, NVRTC: budget 1 GB each until measured; the start-up record measures it) + cuOpt's RMM limit + the Redis container cap (2 GB) + 20 GB reserve ≤ MemAvailable at launch; re-checked before every run; live watchdog stops the run set below 18 GB free | runner (`experiments/gates.py`) |
| Every process: systemd scope `MemoryMax`, `MemorySwapMax=0`; EA process also a **CuPy pool limit** (MemoryMax does **not** charge GPU allocations on GB10 — measured 2026-10-05) | runner |
| One GPU arm at a time; cuOpt (RMM) memory is budgeted at its measured peak, not assumed | runner (GPU lane = 1) |
| Pin one timed solver per Cortex-X925 core (5–9, 15–19); never run analysis with default thread counts beside timed runs | runner; analysis sets `OMP_NUM_THREADS=1` |
| **Thermal gate:** read ACPI zones before each run; **start only below 80 °C**; pause the run set at ≥ 92 °C, resume < 80 °C (host `thermal-watch` stops experiment units at 98 °C) | runner (`experiments/gates.py`) |
| **GPU health per run:** sample SM clock, GPU temperature and per-process GPU utilisation every second; a GPU run whose median SM clock is below 2,000 MHz (the stuck-low-clock fault) or with foreign GPU activity above 2 % of its budget **voids its whole block** (all arms of that instance × seed) and halts the run set for a look | runner |
| cuOpt: an RMM `LimitingResourceAdaptor` with the same limit as the EA's pool; its measured peak recorded | cuOpt process |
| `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` in every EA and solver process | runner |
| **fan-governor rest flag** (`~/.local/state/thermal/rest`, written after 30 min of fan MAX): start nothing new while it exists; in-flight runs finish ([[D-009 Honour the fan governor's rest breaks and the night rest]]) | runner (`gates.rest_due`) |
| **Night rest** (host night-rest timer 00:00–07:00): no window through the night without Berk's yes for that night (D-009) | operator |
| Confirmatory windows run with **both local model servers stopped** (Berk's yes, 2026-10-05 23:2x: "You can deactivate agent fast and agent think llms"); hold-units + `box-window --stops`; a restore timer and the session itself restore them; `local-model-canary` / `canary-hello` paused inside the window | operator; the runner records the units' state per run and refuses a confirmatory run while either is active |
| No downloads (instances, wheels) without Berk's explicit ask | operator |
| Single-instance lock (`flock`) per run set | runner |

## Measured facts
- 2026-10-05 23:2x: `systemd-run --user --scope -p MemoryMax=300M` + a 600 MB CuPy allocation → allocation succeeded,
  process survived. GPU memory is not charged to the cgroup on this driver (580.173.02, kernel 6.17).
- Free memory with both model servers up: ~20–21 GB (2026-10-05 23:00) — below the 30 GB start gate. With both stopped
  (23:47): 109–114 GB.
- **cuOpt's GPU memory (2026-10-06 01:4x):** two fixed 2.4 GB buffers at start-up (it fails under 0.5 and 4 GB RMM
  limits); RMM peak 5.6 GB after the toy warm-up and **10.47 GB** on `ferc__2015-01-01_hw` (the largest PGLib-UC
  case, ~250k × 250k), 60 s, 3 CPU threads; MemAvailable fell 12.5 GB in total (CUDA context and host side included);
  no solution in 60 s; time limit overrun 8.6 s. → The equal-GPU-memory limit for the partner (EA pool = cuOpt RMM) is
  **12 GB** (`EAConfig.gpu_mem_gb` default), budgeted in the gate for both A2 and A3.
