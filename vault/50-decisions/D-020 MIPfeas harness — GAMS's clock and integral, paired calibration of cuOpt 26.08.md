# D-020 MIPfeas harness — GAMS's clock and integral, paired calibration of cuOpt 26.08 (2026-10-11, AgentRA on Berk's word)

**Status:** decided by AgentRA (Claude Code, Anthropic Claude Opus 5.5) for the C1 calibration (D-019, issue #29), before any
calibration run; open to a superseding decision on evidence. The clock and the metric for the v1 runs are fixed in
prereg-v2, not here. Berk (2026-10-11 10:0x), asked whether to download GAMS's published MIPfeas results (38.5 MB): **"Yes,
download it"**.

## What changed since D-019 / #29 — the GAMS archive
`data/mipfeas/2026-09-16_mipfeas_results_and_scripts.zip` (cloud.gams.com share `dcyWLXABG53NNMW`, Last-Modified 2026-09-16
13:40:56 GMT, sha256 `b8d25892c3657e7205e5d7fab2aaf64612953fde7eb4bdd19f9c0506dbb3e327`; unpacked in
`data/mipfeas/gams-2026-09-16/`, gitignored like all of `data/`) holds GAMS's scoring script `create_primalintegral.py`,
the reference values `optimal_objective.csv`, `primalintegral.csv` and a trace (`.trc`) and log for every instance ×
solver × machine — among them **cuOpt 26.08 on a GB10** (also 26.02, 26.04 on GB10; 26.04, 26.08 on B200; CPU solvers on
an Intel Core Ultra 9 285K and an AMD EPYC 9B45). Checked today:
- Their script on their GB10 26.08 traces gives **0.0393** exactly (max per-instance difference to `primalintegral.csv`
  2e-16); every published geo-mean of the results page is reproduced from `primalintegral.csv`.
- Their 233 instances = `data/miplib/benchmark-v2.test` minus our 7 infeasible ones (none missing, none extra); their
  references equal our `data/miplib/solu.txt` on all 233 (|Δ| ≤ 1e-6 relative); 215 easy, 17 hard, 1 open; all on disk.
- **Per-instance comparison is possible**, so the calibration is paired (below) instead of a sample against one number.

## The clock — D-019's wording was wrong; corrected here
D-019 and #29 say "the clock starts at the first process's launch (startup counted)". The published protocol does not:
the GAMS blog says the trace records "wall-clock time from solver startup", excluding model generation and reading, and
the traces show the clock starts at the GAMS↔cuOpt link — glass4's first incumbent is at **0.119 s** in the trace where
cuOpt's own log says 0.03 s (cuOpt's clock starts ≈ 0.09 s after the link's). On the GB10 26.08 run the first incumbent
lies within 0.4 s on 16 of the 30 sample instances below (earliest 0.128 s) — no process start-up of seconds is in it. Our clock `t0` (after the start barrier: parse, CUDA context,
RMM setup and a 1-s synthetic warm-up solve are before it) is therefore the comparable one ("call clock"). The harness
records both:
- **call clock** — seconds since t0 (what every run already records); used for the calibration's pass rule;
- **launch clock** — seconds since the first process's launch (`t0_epoch − min(launch_epoch)` added to every time;
  parse and our warm-up included). Reported beside it; it is the strict clock for comparing our own arms, where A9's
  start-up (the EA's warm-up, SCIP's start) is larger than cuOpt's.
Known handicap / advantage, stated: under the call clock our cuOpt has a warm CUDA context and warm kernels (GAMS's link
pays its context inside or before its clock — unknown); under the launch clock we pay our 1-s warm-up and the MPS parse,
which GAMS does not.

## The metric — GAMS's integral, exactly
A port of `calculate_primal_integral` (their file, 2026-09-15) on our **checked** incumbents (D-002 §4: the checker's
recomputed objective): γ = 2 before the first incumbent; 1 if obj·z* < 0; else round((obj − z*)/max(|obj|, |z*|, 1), 6);
P = Σ γ·Δt / 600 with every time capped at 600 and the last value held to 600. Their script raises on an incumbent better
than z* (with a list of tolerated files); ours sets γ = 0 and flags `beats_ref` (as D-002). Aggregation: shifted geometric
mean, shift 0.001; "feasible found" = P < 2. Unit-tested against their function on their own traces (every GB10 26.08
trace must give the same value to 1e-12). The D-002 §4 `pt_mf` (pre-registered secondary, denominator with the 1, no
sign rule, no rounding) stays as it is; the new column is `pt_gams`.

## cuOpt as GAMS ran it — arm A6m
Their logs: cuOpt 26.08 through GAMS 54.4.0's link, parameters set **num_cpu_threads 24, time_limit 600,
mip_absolute_gap 0, mip_relative_gap 0** (all others default; cuOpt's defaults: relative gap 1e-4, random_seed −1), the
20-core GB10 machine to itself (CPU 1,992 %), max RSS ≤ 16 GB on the sample (49 GB at most, nw04, over all 233).
- **A6m** = cuOpt alone with those four parameters, pinned to all 20 cores (`--layout full`: one lane, the whole box; the
  runner's own threads share it), RMM limit 36 GB, host MemoryMax 24 GB, random_seed = the job's seed (0) — the only
  deliberate difference (GAMS left −1); the same process, barrier and checker as every arm.
- Both LLM servers stopped (no foreign GPU process), one run at a time.

## Calibration — 30 instances, stratified, fixed now
Sample (numpy `default_rng(20261011)`; the 233 sorted by GAMS's GB10 26.08 P, six equal strata, five per stratum):
assign1-5-8, ic97_potential, neos-1171448, nu25-pr12, p200x1188c | mas74, neos-4338804-snowy, proteindesign122trx11p8,
seymour, swath3 | CMS750_4, chromaticindex512-7, milo-v12-6-r2-40-1, neos-1122047, neos-5093327-huahum | blp-ar98,
csched008, dano3_3, map10, neos-4387871-tavua | irish-electricity, neos-4300652-rahue, neos-5107597-kakapo, neos-933966,
sing44 | cryptanalysiskb128n5obj16, neos-2746589-doon, neos-3216931-puriri, satellites2-60-fs, supportcase22.
GAMS's geo-mean on this sample: 0.0393 (= the 233's); 27 of 30 feasible. Expected run time ≈ 4.1 h of solving (23 runs
reach 600 s) + ≈ 1 min per run + rests: one daytime window. Seed 0, one run per instance (their protocol).
- **Statistic:** R = exp(mean over the 30 of log((P_ours + 0.001)/(P_GAMS + 0.001))) on the call clock = the ratio of
  the two shifted geo-means on the sample; 95 % bootstrap interval (10,000 resamples of instances).
- **Pass:** R within ±15 % (0.870 ≤ R ≤ 1.150, D-019's tolerance, now paired) and our feasible count within ±1 of 27.
  For scale, fixed before the run: GAMS's own GB10 26.04 vs 26.08 over the 233 gives a paired log-ratio mean +0.079
  (R = 1.08) with SD 0.51 per instance, so with n = 30 the interval is roughly ×/÷ 1.2 — a pass says "no error larger
  than about 20 %", not more.
- **Reported, not criteria:** R on the launch clock; per stratum; first-incumbent time differences (median, ours −
  GAMS); the GB10 26.04 and B200 26.08 rows on the same sample.
- **If it fails:** the record says the published row is not our reference (D-019) and names the cause it can see
  (clock offsets → first-incumbent differences; resources → late trajectory); v1's external targets are anchored on our
  own cuOpt-alone run in any case (D-019 addendum).

## For v1 (recorded, not decided here)
- Under the launch clock A9 would hold cuOpt at the barrier while the EA warms up — on easy instances seconds at γ = 2
  cost more than every later gain. A9 on MIPfeas should release cuOpt as soon as it is ready (#30, prereg-v2).
- GAMS's data also gives per-instance P for SCIP 10.0.1, HiGHS 1.15.1, ReXi and SCIPConc on the Intel machine — usable
  to describe where cuOpt is weak on the 233 (D-013's weak set for the external run) without running them.
