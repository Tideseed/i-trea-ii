# Compute resources (for the paper and for every window)

## Machine
NVIDIA DGX Spark — GB10: 10 × Cortex-X925 @ 3.9 GHz (**cores 5–9, 15–19**) and 10 × Cortex-A725 @ 2.8 GHz
(**cores 0–4, 10–14**), Blackwell GPU, 121 GiB unified memory shared by CPU and GPU (MemTotal 127.6 GB), 3.7 TB disk.
Ubuntu 24.04.4, kernel 6.17.0-1031-nvidia (the 7.0 kernel is held by the host), driver 580.173.02, CUDA 13.0.
Measured in optopt: SCIP explores about 2× more nodes/s on an X925 than on an A725 (tr12-30, 20 s: 4,686 vs 2,413), so
timed solver runs are pinned to X925 cores only.

Software (this repository's `.venv`, built offline from uv's cache): Python 3.12.3, NumPy 2.4.6, SciPy 1.18.1,
CuPy 14.2.0 (`cupy-cuda13x`), HiGHS 1.15.1 (`highspy`), SCIP 10 via PySCIPOpt 6.2.1, NVIDIA cuOpt 26.08.00
(dependency closure hard-linked from optopt's venv, D-001), redis-py 8.1.0 (pure-Python parser), PyYAML 6.0.3,
pytest 9.1.1.

## The machine is shared
Other Claude Code sessions, the host's cron jobs and two local LLM servers use the same box:

| service | unit | port | memory | in a confirmatory window |
|---|---|---|---|---|
| vLLM serving Ornith-1.5-35B-A3B (NVFP4) | `ornith-vllm` | 8094 | tens of GB, GPU | **stopped** (Berk's yes, 2026-10-05 23:2x) |
| llama.cpp serving Qwen3.8-27B (UD-IQ3_XXS) | `llama-qwen38` | 8092 | tens of GB, GPU | **stopped** |
| Redis for the exchange (`itrea-redis`, image `redis:7-alpine`) | Docker container | 127.0.0.1:6390 | 2 GB cap, `maxmemory 1536mb` | running, pinned to A725 core 0 |

With both LLM servers up, MemAvailable was about 20–22 GB (2026-10-05 23:00) — below the 30 GB start gate; with both
stopped about 109 GB. Both servers share the GPU and the 273 GB/s memory bus with the EA and cuOpt: optopt measured that
continuous GPU co-inference doubled cuOpt's P(30) at an unchanged SM clock (bandwidth contention). That is why
confirmatory windows stop them and why foreign GPU activity voids a block (D-003). `tools/serving-window.sh stop|restore`
does the stopping and the restore (canaries paused, `hold-units` lines, health check on restart, lease release).

## Footprint per arm (D-002 v3 §3)
| arm | solver side | partner side | GPU memory |
|---|---|---|---|
| A0, A0* SCIP alone (harness-matched) | 1 X925 core | — | — |
| A1 SCIP ‖ SCIP | 2 X925 cores | — | — |
| A2 SCIP ‖ cuOpt (race) | 1 X925 core | GPU + 3 A725 cores (`num_cpu_threads = 3`) | RMM `LimitingResourceAdaptor` = the EA's pool limit |
| A3 SCIP ⇄ I-TREA II (GPU), A3r without exchange | 1 X925 core | GPU + the same 3 A725 cores | CuPy pool limit `gpu_mem_gb` |
| A4 SCIP ⇄ I-TREA II (CPU, NumPy) | 1 X925 core | the same 3 A725 cores, OMP 1 | — |
| A5, A5x0, A5y0 I-TREA II alone | — | partner footprint | as A3 |
| A6 cuOpt alone | — | partner footprint | as A2 |

The partner footprint is identical in A2 and A3 so that "same GPU" means the same compute; the exact A725 cores are
fixed in E0 and written into every `run.json` (core 0 hosts Redis). The EA's host loop uses one of those cores;
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` in every EA and solver process. Reported per arm: the
compute actually used (cgroup `cpu.stat`, NVML GPU-seconds), not only the compute allotted.

## Caps and gates (D-003; enforced by the runner, `itrea.experiments.gates`)
- **Per process:** a systemd scope with `MemoryMax` and `MemorySwapMax=0`; SCIP screening runs 9 GiB.
- **GPU memory is not charged to the cgroup** on this driver (measured 2026-10-05: a 600 MB CuPy allocation succeeded
  under `MemoryMax=300M`). The CuPy pool limit (`gpu_mem_gb`, default 4 GB) is the only real cap on the EA's device
  memory; cuOpt gets an RMM limit of the same size.
- **Run-set memory gate:** Σ host `MemoryMax` + Σ CuPy pool limits + Σ non-pool GPU overhead per CUDA process (context,
  cuSPARSE/cuBLAS handles, NVRTC: 1 GB each until measured — import + context alone took 0.36 GB) + cuOpt's RMM limit +
  the Redis cap (2 GB) + 20 GB reserve ≤ MemAvailable at launch; a live watchdog stops the run set below 18 GB.
- **One GPU arm at a time** (concurrent GPU runs would confound timing).
- **Thermal:** start a run only below 80 °C (ACPI maximum); pause the run set at ≥ 92 °C, resume below 80 °C. The host
  shut down at about 95 °C under sustained GPU load on 2026-10-01 (critical trip 104 °C); its `thermal-watch` stops
  experiment units at 98 °C.
- **GPU health:** SM clock, GPU temperature and per-process GPU utilisation sampled every second; a GPU run with median
  SM clock below 2,000 MHz (the stuck-low-clock fault: about −40 % while looking healthy; healthy under load
  2,400–2,500 MHz) or foreign GPU activity above 2 % of its budget voids its whole block.
- **Lease:** no timed run without the box-window lease held by this project (`box-window claim --who "i-trea-ii
  (Claude Code)" …`).

## Budget arithmetic (planning, D-002 v3 / D-007)
About 100 GPU runs per 10-hour night at 300 s plus ~60 s overhead. Per block the GPU arms are A3, A5, A2, A5x0
(4 × 300 s): 32 test instances × 2 seeds = 64 blocks = 256 GPU runs ≈ 2.6 nights. CPU arms (A0, A0*, A1) run
concurrently if E0 shows ≤ 3 % interference, else serially (+~2 nights). A3r on 8 instances × 2 seeds, A6 one seed,
tuning (E1/E4) about one night, a 10 % rerun reserve. Total about 5–6 nights.

## Record per window (filled as windows run)
| window | when (CEST) | lease | LLM servers | runs | MemAvailable min | ACPI max | notes |
|---|---|---|---|---|---|---|---|
| W0 | 2026-10-05 23:46 → 2026-10-06 07:30 | `i-trea-ii (Claude Code)`, stops `ornith-vllm,llama-qwen38` | stopped (restore timer `itrea-restore-serving.timer` 07:30) | build tests and the SCIP screening pass only | — | — | open |
