# D-021 measurements M1–M3 and the calibration M4 — INTERIM, INCOMPLETE (19 of 30) (2026-10-11 to 2026-10-13, windows W-1011b, W-1012, W-1013)

**Question.** Before any v1 run, [[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]]
§ "Measurements first" asks four things about the external harness at A9's footprint F (X925 5–6, A725 1–3, the GPU):
does every harness setting act as specified (M1, smoke); does F stay inside the thermal envelope for 600-s runs (M2,
soak, rule R23); which instances cuOpt cannot run at F's memory caps (M3, the "does not fit F" list, R29); and does
cuOpt 26.08 alone on F (arm A6F) under our call clock reproduce GAMS's published cuOpt-26.08-on-GB10 run on D-020's
30-instance sample (M4, calibration: the late-clock check, then the footprint ratio R). These are harness and calibration
measurements. **None of them tests the method** (A9 against the solvers); the A9 runs in M1 and M2 are there for the
harness and the heat, not for a comparison.

**Verdict.**
- **M1 — gate passed** (12 / 12 runs, 0 void): every harness setting as specified; R14 (cuOpt's thread variables
  unset) measured inert — cuOpt CPU-s per wall-s 5.02 / 4.99 / 5.03 unset vs 5.02 / 4.98 / 5.03 set.
- **M2 — gate passed, F accepted** (4 / 4, 0 void): R23 holds on every run — steady-state p95 66–73 °C, maximum
  76 °C, no 5-s sample ≥ 90 °C, no clock cap, busy SM 2,392 MHz.
- **M3 — measured; the list is fixed:** eilA101-2, nw04, supportcase6 do not fit F (2 cap hits at the 40-GB host
  cap, 1 host peak above 0.8 × 24 GB). The decision set goes from 218 to 215 instances, H_cu-unseen from 65 to 64.
- **M4 — INTERIM, INCOMPLETE (19 of 30)**, stopped by D-024's rule 0 (performance runs first). The late-clock check, computed
  on the 6 strata-0–1 instances run so far, passes: median first-incumbent offset ours − GAMS −0.119 s (rule
  ≤ +0.3 s); 4 strata-0–1 instances remain (neos-1171448, p200x1188c, seymour, swath3). The footprint ratio R on the
  call clock is 1.390 [1.081, 1.837] over the 19 (1.416 [1.087, 1.903] over the 18 read at 14:4x and quoted in the
  journal 2026-10-13 14:4x, AGENTS.md and the session checkpoint of 2026-10-13 16:00, kept in the private operations
  repository); no INSIDE / OUTSIDE reading is made before 30 of 30.

## What ran
Every pool: `.venv/bin/python -m itrea.experiments.pool <job file> --layout pair --deadline HH:MM` from a chain script
in `logs/` (not in git), inside a box-window lease held by `i-trea-ii (Claude Code)` with both local LLM servers
(vLLM and llama.cpp; units in `gates.SERVING_UNITS`) stopped. Run hashes as recorded in the run table (`run_hash`; for
the cuOpt-only arms it equals `config_hash`).

| | when (CEST) · pool | command (chain) | job file · sha256 | instances × arms @ run hash × seed × T | runs | code (`src/`) |
|---|---|---|---|---|---|---|
| **M1** smoke | 10-11 15:12–15:33 · 20261011-151204-1160038 | `pool jobs/itx-m1.jsonl --layout pair --deadline 21:00` (`chain-itx-m1.sh`) | `jobs/itx-m1.jsonl` (1c6a684) · `c5575608a773…` | fhnw-binpack4-48, glass4, radiationm18-12-05 × {A9@m1 `28e5a5e9c725` (EA config `0f011a2094a2`), A6F@m1 `82a371305f88`, A6F@m1env `ffae965f6746` (thread variables kept), A2F@m1 `d4d89da16786`} × seed 0 × 120 s, harness ext | 12 / 12 | merge 0708835 (recorded HEADs ba356f6 / 0708835 differ only outside `src/`, `configs/`, `jobs/`, `tools/`) |
| **M2** soak | 10-11 15:35–16:34 · 20261011-153534-1276706 | `pool jobs/itx-m2.jsonl --layout pair --pause-c 90 --deadline 21:00` (`chain-itx-m23.sh`) | `jobs/itx-m2.jsonl` (1c6a684) · `870ef274a4c8…` | comp21-2idx, neos-4763324-toguru × {A9@m2 `28e5a5e9c725`, A6F@m2 `82a371305f88`} × seed 0 × 600 s, AB / BA written as positions, harness ext | 4 / 4 | 7998ab4 (recorded 266b9cf / 7998ab4) |
| **M3** memory probe | 10-11 16:34–18:13 · 20261011-163415-1522373 | `pool jobs/itx-m3.jsonl --layout pair --deadline 21:00` (same chain) | `jobs/itx-m3.jsonl` (2fd1809) · `b2e4cf10d96b…` | R29's 8 instances × A6F@m3 `d3aba2773933` (cuOpt RMM 40 GB / host 40 GB, `cuopt_memory_void: false`: a cap hit is recorded, not voided) × seed 0 × 600 s | 8 (6 completed + 2 cap hits) | 7998ab4 (recorded 0613d58) |
| **M4** calibration, part 1 | 10-12 12:06–12:40 · 20261012-120607-2036840 | `pool jobs/itx-m4.jsonl --layout pair --deadline 22:30` (`chain-itx-m4.sh`) | `jobs/itx-m4.jsonl` (1c6a684) · `a8f02cb7e0e2…` | D-020's 30-instance sample × A6F `82a371305f88` (harness ext: cuOpt gaps 0, thread variables unset, RMM 24 GB / host 24 GB, the project CUDA cache) × seed 0 × 600 s, set `mipfeas` | 5, then stopped (stop file, for another project's session) | 4c3e9fd's runner (recorded 136dba6 / 90b9b72) |
| **M4** calibration, part 2 | 10-13 12:32–15:16 · 20261013-123201-3850111 | the same command, resumed (`chain-itx-m4b.sh`; the pool skips completed blocks) | same | same | 14 (+ 5 skipped as done), then stopped (stop file, D-024 rule 0) | 999995f's runner = merge 6f0339d's `src/` (recorded 756944f / 999995f / f6dfcf5) |

Job files' sha256 in full (`sha256sum jobs/itx-m*.jsonl`, 2026-10-13):
`c5575608a773fdee952205f39fae0770122731b30ae27a2428cde0b41c17e91c` (M1),
`870ef274a4c8556b5eeb926a20f73f8c4eaebae67ca4cd441aca8f064468272e` (M2),
`b2e4cf10d96b555f35955d41615fa03329ec73eeb57ce7510317621cdd3edd5c` (M3),
`a8f02cb7e0e200aa4d201b7faeaa8c372a809303e6251d502dcb9f28affa93b5` (M4). No job file changed after its runs.

M4's two parts ran on two runner versions: D-021 phase 2 (R12's stop at the proven optimum, cuOpt's SIGKILL after a
definitive end) was merged between them. A6F's plan and run hash are the same on both (phase 2's byte-identical plan
check), and R12 and the kill are off under the `ext` harness, so the 19 runs are one set. `src/` and `configs/` are
unchanged from 999995f to fd75476.

**Windows** (W-1011b is also in [[Compute resources]]):
- **W-1011b** (10-11 15:10 → 18:17): lease `--mem 101` (M3's A6F at RMM 40 / host 40), both servers stopped
  15:10:41–42, MemAvailable ≈ 110 GB; GPU tests before M1: `test_cuopt_run.py` (with `ITREA_TEST_CUOPT_GPU_GB=8`),
  `test_pair.py`, `test_three_way.py` — 3 + 3 passed. Restored by the chain 18:13:38–18:16:59 and verified (both units
  active and enabled, :8094 and :8092 healthy, lease free, no hold lines). Fan-governor rests 15:43 (20 min), 16:34
  (20 min), 17:25 (30 min; the M3 pool waited 17:28:34–17:55:17).
- **W-1012** (10-12 12:05 → 12:43): lease `--mem 69`, servers stopped 12:05:51–52, MemAvailable 110 GB. Paused after
  5 runs at the request of another project's session: its chat agent's hosted model had run out of credit and its
  fallback was the vLLM server, so the agent could not answer anyone. Stop file 12:32, pool stopped 12:40:21, servers
  restored 12:40:31–12:43:52 (healthy, enabled, canaries resumed, lease released).
- **W-1013** (10-13 from 12:31; lease until 23:00, moved to 23:45): `--mem 97` (D-022's A9 follows), servers stopped
  12:31:04–05, MemAvailable 110 GB. Possible because that project's agent has answered with a hosted model
  since 09:32 (Berk's decision). GPU tests before the pool: `test_cuopt_run`, `test_pair`, `test_three_way`,
  `test_ref_stop` (4 passed, 1 skipped) and one cold-cache EA test (passed). The pool waited on the fan governor's rest
  flag 13:03:59–13:12:30, 13:53:09–14:05:30 and 14:45:46–15:05:49. Stop file 15:11 (D-024 rule 0); the M4 pool ended
  15:15:56 after its current run, and the window went on with D-022 (set `itx-r13`, from 15:16) — not part of this note.

## Numbers

### M1 — smoke (fhnw-binpack4-48, glass4, radiationm18-12-05; 120 s; seed 0)
Checked per run (run.json and cuOpt's summary): A6F's cuOpt on cores 1–3, 5, 6 with 5 threads; A2F's cuOpt on 1–3, 6
with 4 threads (SCIP on 5); A9's cuOpt on 1–3 with 3 threads; both MIP gaps 0 in every cuOpt solve; OMP / OPENBLAS /
MKL unset except in A6F@m1env; RMM 24 GB, host 24 GB; named scopes `itrea-run-…`; busy SM medians 2,392–2,405 MHz
(GPU health ok); in-run ACPI maximum ≤ 78.5 °C; no compile on the clock (CuPy cache and project CUDA cache); no void
check fired. Two items of D-021's M1 list are not in the M1 record: a `ref_reached` stop (R12 was built only in
phase 2; its GPU test `test_ref_stop` passed on 10-13) and an RMM allocation error classified as a harness void (D-021 lists it as a
fake-process test; the journal's M1 entry does not record it).

Primal integral P(120) as recorded (Berthold, call clock). **A smoke record, not a comparison** — one seed, three
instances, arms with different core counts:

| instance | A9@m1 | A6F@m1 | A6F@m1env | A2F@m1 |
|---|---|---|---|---|
| fhnw-binpack4-48 | 0.0082 | 0.0045 | 0.0042 | 0.0054 |
| glass4 | 0.2272 | 0.3303 | 0.3102 | 0.2981 |
| radiationm18-12-05 | 0.0102 | 0.0071 | 0.0070 | 0.0065 |

### M2 — soak at F (600 s; `--pause-c 90`)
Per run, from the host's 5-s log; steady state = from t0 + 120 s or fan MAX + 60 s (D-021 § M1 and M2 results):

| run | start °C | steady p95 | steady max | first 60 s max | ≥ 90 °C | caps | busy SM (MHz) | GPU W (median) | cuOpt CPU-s / wall-s |
|---|---|---|---|---|---|---|---|---|---|
| comp21-2idx A6F | 43.7 (fan auto) | 70 | 70 | 68 | 0 | 0 | 2,392 | 18.9 | 4.97 |
| comp21-2idx A9 | 38.5 | 68 | 68 | 64 | 0 | 0 | 2,392 | 21.8 | 2.99 |
| toguru A6F | 52.7 | 73 | 76 | 70 | 0 | 0 | 2,392 | 24.8 | 4.88 |
| toguru A9 | 53.8 | 66 | 67 | 67 | 0 | 0 | 2,392 | 15.3 | 2.97 |

R23 (p95 ≤ 85 °C, no sample ≥ 90 °C, no cap, SM ≥ 2,350 MHz) holds on every run → **F is accepted for v1**. The box
never exceeded 76 °C in these 600-s runs, 16 °C below the void line. One 20-min governor rest fell between the comp21
arms.

Primal integrals as recorded — **a soak record, not a comparison** (one seed, two instances; A9's cuOpt has 3 of F's
5 cores, A6F's all 5): comp21-2idx A9 P(600) 0.468 (`pt_gams` 0.468) vs A6F 0.370 (0.370); toguru A9 0.447 (0.448) vs
A6F 0.333 (0.334). A9 has the higher (worse) value on both. The same direction appears at 120 s in D-018 stage 1 (A6
alone 0.507 / 0.574 vs the control A9@bo1s 0.575 / 0.598).

### M3 — memory probe (A6F at cuOpt RMM 40 GB / host 40 GB; 600 s)

| instance | status | cuOpt host maxrss | RMM peak | GAMS RSS | on the list? |
|---|---|---|---|---|---|
| eilA101-2 | killed at the 40-GB host cap, 124 s | 40.2 GiB | — | 42.8 GiB | yes (cap hit) |
| nw04 | killed at the 40-GB host cap, 141 s | 40.3 GiB | — | 46.9 GiB | yes (cap hit) |
| supportcase6 | completed (optimal, 336 s) | 22.6 GiB | 17.4 GB | 27.9 GiB | yes (> 0.8 × 24 = 19.2 GiB) |
| square47 | completed (600 s) | 17.9 GiB | 14.9 GB | 20.8 GiB | no |
| irp | completed (optimal, 70 s) | 17.6 GiB | 8.3 GB | 17.6 GiB | no |
| snp-02-004-104 | completed | 8.5 GiB | 6.6 GB | 22.6 GiB | no |
| proteindesign121hz512p9 | completed | 10.0 GiB | 7.0 GB | 15.3 GiB | no |
| proteindesign122trx11p8 | completed | 10.0 GiB | 6.7 GB | 15.8 GiB | no |

Both kills are confirmed cap hits (the scope's Result is oom-kill), recorded rather than voided by design. The list is
fixed before any v1 run and excludes the three from both arms: eilA101-2 (in H_cu: GAMS cuOpt P 0.230), nw04 and
supportcase6 (both in E). Decision sets: H_cu-unseen 65 → 64, unseen 218 → 215 (`datasets/mipfeas_strata.csv`,
column `does_not_fit`; [[Datasets]]). The 24-GB caps are GiB (MemoryMax 24576M, RMM 24 × 2³⁰ bytes).

### M4 — calibration, interim at the stop (19 of 30)
Report: `itrea mipfeas-report --arm A6F` on a run table built from the 19 run directories alone (a copy of
`traces/raw/mipfeas/` in `/tmp/m4raw-docs/mipfeas/`, so no other set was read):
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" taskset -c 15 \
  .venv/bin/itrea build-runs --raw /tmp/m4raw-docs --out /tmp/m4ds-docs --workers 1
OMP_NUM_THREADS=1 … taskset -c 15 .venv/bin/itrea mipfeas-report --arm A6F --runs /tmp/m4ds-docs/runs.parquet
```
(run 2026-10-13 17:xx, HEAD fd75476). The 18-run reading at 14:4x used the table `/tmp/ds-1013/` built at 14:48;
re-running the report on it gives the same figures as the journal.

Per instance (P = GAMS's integral `pt_gams`, 0–2, T = 600 s; "first inc." in seconds on the call clock; — = not run):

| stratum | instance | P ours (call) | P ours (launch) | P GAMS | first inc. ours | first inc. GAMS |
|---|---|---|---|---|---|---|
| 0 | assign1-5-8 | 0.0002 | 0.0078 | 0.0004 | 0.014 | 0.128 |
| 0 | ic97_potential | 0.0059 | 0.0131 | 0.0024 | 0.021 | 0.131 |
| 0 | neos-1171448 | — | — | 0.0019 | — | 0.157 |
| 0 | nu25-pr12 | 0.0019 | 0.0095 | 0.0027 | 0.031 | 0.172 |
| 0 | p200x1188c | — | — | 0.0014 | — | 0.140 |
| 1 | mas74 | 0.0028 | 0.0100 | 0.0060 | 0.013 | 0.137 |
| 1 | neos-4338804-snowy | 0.0136 | 0.0206 | 0.0134 | 0.019 | 0.142 |
| 1 | proteindesign122trx11p8 | 0.0084 | 0.0166 | 0.0065 | 0.262 | 0.249 |
| 1 | seymour | — | — | 0.0049 | — | 0.147 |
| 1 | swath3 | — | — | 0.0070 | — | 0.241 |
| 2 | CMS750_4 | 0.0464 | 0.0538 | 0.0246 | 1.152 | 1.172 |
| 2 | chromaticindex512-7 | 0.0120 | 0.0198 | 0.0143 | 3.614 | 4.281 |
| 2 | milo-v12-6-r2-40-1 | — | — | 0.0295 | — | 5.286 |
| 2 | neos-1122047 | 0.0268 | 0.0342 | 0.0163 | 8.022 | 4.893 |
| 2 | neos-5093327-huahum | — | — | 0.0323 | — | 6.229 |
| 3 | blp-ar98 | 0.0455 | 0.0530 | 0.0446 | 0.510 | 1.421 |
| 3 | csched008 | 0.3013 | 0.3090 | 0.0474 | 85.623 | 13.774 |
| 3 | dano3_3 | 0.1434 | 0.1505 | 0.0606 | 43.030 | 14.248 |
| 3 | map10 | — | — | 0.0570 | — | 0.368 |
| 3 | neos-4387871-tavua | — | — | 0.0434 | — | 1.847 |
| 4 | irish-electricity | 0.4686 | 0.4752 | 0.1323 | 13.810 | 14.210 |
| 4 | neos-4300652-rahue | 0.5103 | 0.5172 | 0.1477 | 0.209 | 0.289 |
| 4 | neos-5107597-kakapo | — | — | 0.2264 | — | 0.304 |
| 4 | neos-933966 | 0.0877 | 0.0971 | 0.0868 | 0.071 | 0.175 |
| 4 | sing44 | — | — | 0.1389 | — | 0.306 |
| 5 | cryptanalysiskb128n5obj16 | 2.0000 | 2.0000 | 2.0000 | — | — |
| 5 | neos-2746589-doon | 0.6828 | 0.6906 | 0.5711 | 201.358 | 169.808 |
| 5 | neos-3216931-puriri | 2.0000 | 2.0000 | 2.0000 | — | — |
| 5 | satellites2-60-fs | — | — | 0.5090 | — | 0.201 |
| 5 | supportcase22 | 2.0000 | 2.0000 | 2.0000 | — | — |

| reading | 19 instances (at the stop) | 18 instances (read 14:4x; journal 2026-10-13, AGENTS.md) |
|---|---|---|
| R = (SGM_ours + 0.001) / (SGM_GAMS + 0.001), call clock [95 % bootstrap] | **1.390 [1.081, 1.837]** | 1.416 [1.087, 1.903] |
| R, launch clock | 2.036 [1.572, 2.654] | 2.118 [1.629, 2.799] |
| shifted geo-mean (s = 0.001), ours call / launch vs GAMS on the same instances | 0.0662 / 0.0974 vs 0.0473 | 0.0546 / 0.0822 vs 0.0383 |
| feasible (P < 2), ours vs GAMS on the same instances | 16 vs 16 | 16 vs 16 |
| R per stratum 0 / 1 / 2 / 3 / 4 / 5 (call) | 1.112 / 0.880 / 1.364 / 2.463 / 2.304 / 1.046 (n 3, 3, 3, 3, 3, 4) | the same, stratum 5 1.061 (n 3) |
| first incumbent ours − GAMS, call clock, median [95 %] (16 instances with both) | −0.107 [−0.128, +0.792] s | the same |
| the same on the launch clock; median start-up offset | +2.173 [+2.050, +3.411] s; 2.267 s | the same |
| **late-clock check** (median ours − GAMS over strata 0–1, ≤ +0.3 s) | **−0.119 s (n 6): passes** — stratum 0 −0.114 (n 3), stratum 1 −0.123 (n 3) | the same |
| other strata, reported only | 2: −0.020 (n 3); 3: +28.782 (n 3); 4: −0.104 (n 3); 5: +31.550 (n 1) | the same |
| verdict | **INTERIM, INCOMPLETE (19 of 30)** (11 without a run) | INTERIM, INCOMPLETE (18 of 30) |

Configuration hash of every chosen run: `82a371305f88`. Reference check: GAMS's z* equals our references (max
relative difference 0). Scale from GAMS's own archive on the same 30 (D-020): GB10 cuOpt 26.04 vs 26.08 R 1.136 [0.909,
1.466]; B200 26.08 vs GB10 26.08 R 0.920 [0.845, 1.007].

**Reading (interim, not a verdict).** Our cuOpt 26.08 alone on F (5 cores, 5 threads) has a shifted geometric mean
about 1.4 times GAMS's (20 cores, 24 threads, the whole GB10) on the same 19 instances. The excess sits in strata 3–4:
csched008 (first incumbent 85.6 s vs 13.8 s; P 0.301 vs 0.047), dano3_3 (43.0 vs 14.2 s; 0.143 vs 0.061),
irish-electricity (0.469 vs 0.132), neos-4300652-rahue (0.510 vs 0.148). That is consistent with fewer threads, but the
reading is not made: D-021's footprint reading (band 0.870–1.150 on the call clock, feasible ±1) needs all 30, and no
interim R is read against the band. The late-clock check — the part that
gates the clock every v1 decision reads — passes on what has run (6 of the 10 strata-0–1 instances; INTERIM). D-021's earlier thin evidence that threads matter
little (A6 at 3 threads vs GAMS, R 1.00 [0.89, 1.12] on the 15 development instances at 120 s) came from different
instances and a shorter budget. v1's targets are anchored on our own A6F either way (D-021).

## Integrity
- **Checker coverage** (every incumbent vector stored and checked by the independent exact checker; only passed
  incumbents enter a metric):

  | | stored | passed | failed | unchecked |
  |---|---|---|---|---|
  | M1 | 371 | 371 | 0 | 0 |
  | M2 | 945 | 945 | 0 | 0 |
  | M3 | 75 | 75 | 0 | 0 |
  | M4 (19 runs) | 1,145 | 1,141 | 4 | 0 |

  M4's 4 failed incumbents are all cuOpt's: neos-4338804-snowy k 16 at 9.14 s (bound violated by 8.3e-6) and
  neos-933966 k 37, 41, 60 at 31.4, 39.7, 204.9 s (integrality violated by 6.5e-6, 4.8e-6, 6.3e-6). The checker's
  tolerance is 1e-6 (bounds and integrality absolute, rows relative to max(1, |side|)).
- **Crashes and voids:** M1, M2, M4 0 failed, 0 void, 0 requeue. M3's 2 failed runs are its intended cap hits
  (recorded, not voided). No run compiled on the clock (CuPy cache and the project CUDA cache `~/.cache/itrea/cuda`).
  M4: GPU health ok on all 19, busy SM median 2,392 MHz on every run.
- **M4 heat.** In-run maxima from run.json 66.5–86.0 °C (neos-2746589-doon 86.0 and csched008 85.1 on 10-12; every
  10-13 run ≤ 78.6). D-021's M4 stop rule (any thermal or cap void, or any run with steady-state p95 > 85 °C in the first
  2 h) was not evaluated by a tool during the window; computed for this note from the host's 5-s log
  (`~/.local/state/thermal/fine-<date>.log`, hottest zone per sample, t0 + 120 s to the run's end; the fan-MAX + 60 s
  start was not used): steady-state p95 65–77 °C on the 16 runs longer than 120 s, so the rule did not fire. No
  thermal void, no cap void, no thermal pause.

## Checker tolerance — found during M4 and the cuOpt probe, to investigate (issue #36)
Both directions of a tolerance mismatch between our checker and the solvers turned up:
- **Better than the proven optimum, accepted by our checker.** mas74 (M4): cuOpt's incumbent k 21 at 95.6 s has
  objective 11,801.18516 against MIPLIB's `=opt=` 11,801.18572 (integrality violation 1.03e-7, rows 2.2e-12); our
  checker passes it, P counts γ = 0 from there, and GAMS's scoring script would raise on it unless the trace is on its
  whitelist, where it sets the objective to z* (`mipfeas-report`'s "beats the reference" line). glass4: the cuOpt 26.10 probe (10-13) offered cuOpt a glass4 point from a development
  run (A3@lpd seed 0, the EA's incumbent k 30 at 92.9 s, objective 1,200,006,662.64) below MIPLIB's proven optimum
  1,200,012,599.97; cuOpt rejected it, our checker had accepted it.
- **Scope on glass4, counted for this note** in `datasets/runs.parquet` as built 2026-10-11 14:09 (939 rows: 921 runs + 18
  synthetic; glass4 is the only `=opt=` instance with such incumbents; item_placement_9903's references are provisional
  by design): **60 checked incumbents in 8 of the 57 glass4 runs (2 more glass4 rows are synthetic) beat z* by GAMS's
  rounded test (`beats_ref`)** — 33 with role `ea` and 27 SCIP records of EA posts (role `solver`, source `ea`); by a
  strict `obj < z*` test there are 62, the 2 more in A3@rc1d s0 at 44.5 s (9.8e-10 relative below z*). Every one passed,
  worst row violation ≤ 4.5e-7 relative (inside 1e-6). All 8 runs are EA arms — A3@ad1 s0, A3@cl1 s1, A3@lpd s0,
  A3@rc0 s0, A3@rc1 s0, A3@rc1d s0, A3@stall100 s0 (roles `ea` and `solver`: SCIP took the EA's posts as its incumbents)
  and A5@c04 s0 (`ea` only); the first `beats_ref` incumbent in each run comes at 47.8–119.9 s (the first strictly below
  z* at 44.5–119.9 s). P counts γ = 0 from that time in those runs; the effect on any development reading has not been
  computed.
- **Feasible for cuOpt, rejected by ours:** neos-4338804-snowy 1 and neos-933966 3 cuOpt incumbents (above). cuOpt's
  own tolerance values are not checked here.
- Open: compare our 1e-6 rule with SCIP's and cuOpt's feasibility tolerances on glass4 (coefficients up to 8.4e6) and
  decide how the checker treats an incumbent better than a proven optimum (flag, reject, or re-check at a tighter
  tolerance) before any v1 run reads `pt_gams`.

## Caveats
- One seed, one machine, one cuOpt version (26.08). M4 has 19 of 30 instances, 3 or 4 per stratum, so R and its
  interval are interim; even complete, the 30-instance sample resolves about ×/÷ 1.28 (D-021).
- GAMS's run used cuOpt 26.08 through GAMS 54.4.0, 24 threads on the whole box, its trace clock starting at the solver
  link; ours uses 5 threads on F and the call clock (t0 after our start barrier) — D-020 § "The clock". The launch-clock
  figures carry our ~2.3-s start-up.
- **Exposure (D-021 R33):** M4's 30 instances and M3's 8 are on the strata file's `exposed` list (37; 34 unseen; 31 in
  the decision set). Development never consults their A6F outcomes, and v1's decisions are reported with and without
  them.
- M1's and M2's primal integrals are recorded for completeness and are not comparisons.
- The M4 steady-state p95 above was computed for this note by hand from the host log, not by the pool.

## Failures
- M4 did not finish: paused 10-12 after 5 runs for another project's session (a user-facing service that needed the
  vLLM server); stopped 10-13 after 19 runs under D-024 rule 0 —
  Berk (15:1x): **"We shall also always aim for performance runs
  unless something structural is buggy in our experimentation methodology."** Its late-clock check had passed. The
  remaining 11 runs are owed: the 4 strata-0–1 instances before v1's first external run (the clock every v1 decision
  reads), the rest to fill otherwise idle daytime hours; the footprint ratio stays INCOMPLETE until 30 of 30.
- M3's two runs failed at the 40-GB host cap by design (the probe's measurement).
- M1, fhnw-binpack4-48 A9: SCIP proved optimality at 13.8 s; cuOpt, which has no interrupt API, was SIGKILLed by the
  runner at 84 s (`stopped_by_runner`, not a crash) — about 70 s of GPU load after the proof. Fixed in phase 2 by the
  harness switch `cuopt_kill_on_end` (in preset `itx1`, not in `ext`).
- The 4 cuOpt incumbents our checker rejected (Integrity) and the incumbents below the proven optimum it accepted
  (previous section).
- Not run: M5 (the optional A9 preview at 600 s). M6 (A0's CPU lane check) is queued after D-022's runs on 10-13.

## Files
- Raw run directories (not committed): `traces/raw/itx-m1/`, `traces/raw/itx-m2/`, `traces/raw/itx-m3/`,
  `traces/raw/mipfeas/`; pool logs `logs/itx-m1.log`, `logs/itx-m2.log`, `logs/itx-m3.log`, `logs/itx-m4.log`,
  `logs/itx-m4b.log`; pool resource records `traces/resources/pool-20261011-151204-1160038.json`,
  `pool-20261011-153534-1276706.json`, `pool-20261011-163415-1522373.json`, `pool-20261012-120607-2036840.json`,
  `pool-20261013-123201-3850111.json`.
- Analysis: `src/itrea/analysis/mipfeas.py` (`itrea mipfeas-report --arm A6F`: the late-clock check, R, D-021's
  reading); `src/itrea/analysis/check_solutions.py` (the checker, `TOL = 1e-6`); the sets in
  `datasets/mipfeas_strata.csv` (`tools/derive_mipfeas_strata.py`). The run table of these runs is not committed
  (`datasets/runs.parquet` in the main checkout, untracked, predates them).
- Records: D-021 § "M1 and M2 results", § "M3 result", § "Phase 2 built, M0 run, the sets fixed"; [[D-024 Targeted first, wider only on a hint — staged testing for development and v1]]
  rule 0; [[D-020 MIPfeas harness — GAMS's clock and integral, paired calibration of cuOpt 26.08]] (the sample, the
  integral, the clock); journal 2026-10-11 ("15:12–15:35", "15:35–18:17"), 2026-10-12 ("12:05", "12:3x–12:44"),
  2026-10-13 ("12:2x–12:32", "14:4x–14:5x", "15:1x"); [[Experiment index]] rows ITX-M1 … ITX-M4; [[Development log]]
  § "D-021 measurements M1–M3 and the interim calibration".

## Addendum 2026-10-15 — M4 complete (30 of 30): OUTSIDE
The last 11 runs ran 2026-10-14 21:48–23:09 and 2026-10-15 09:3x–10:00 (sets `mipfeas`, A6F, ext, 600 s). `itrea mipfeas-report
--arm A6F`: **R 1.313 [1.091, 1.618]** on the call clock (launch clock 1.963 [1.606, 2.408]); feasible 27 vs GAMS 27; late-clock
check passed (median first incumbent ours − GAMS over strata 0–1 −0.119 s, n 10); configuration hash 82a371305f88 on all 30.
**Verdict OUTSIDE** the band [0.870, 1.150]: with the clock checked, the difference is the footprint F (5 threads + the GPU against
GAMS's 24-thread whole box) and the machine, as D-021 expected; v1's targets are anchored on our own A6F. The interim R 1.416 on 19
runs above is superseded. The heat ladder (R9) could now be asked for; not asked (D-021 R9: Berk's yes, not needed for v1).

## Correction 2026-10-16 (AgentRA) to the 2026-10-15 addendum
"The interim R 1.416 on 19 runs above is superseded": 1.416 was the 18-run figure; the 19-run interim figure was 1.390. Both are
superseded by the final R 1.313 on 30 runs.
