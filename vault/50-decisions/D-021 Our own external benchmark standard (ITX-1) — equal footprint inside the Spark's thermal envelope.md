# D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope (2026-10-11, AgentRA on Berk's word)

**Status:** decided by AgentRA (Claude Code, Anthropic Claude Opus 5.5) for the harness, safety, calibration and
reporting rules; the items marked **[Berk]** change targets he decided in [[D-019 v1 — deadline and performance target]]
and enter prereg-v2 only with his yes (Open for Berk, 1) — until then D-019's literal readings decide and ITX-1's are
reported beside them. Open to a superseding decision on evidence. Nothing here is a result of ours; every A9 figure is a
plan or a proxy from GAMS's traces. Berk (2026-10-11 11:1x): **"You don't need to completely adhere to GAMS standards.
You can come up with a better (more fair) and safer (not burning up dgx spark) set of rules. Of course we may still get
the good ideas from GAMS and Mittelmann and other benchmarks but it is ok to come up with our own standards."**

How it was made: a design workflow (Claude Code subagents, read-only) wrote four proposals — fairness (SFM-1), safety
(envelope-first), statistics (ITB-1), schedule (ITREA-EXT-1); two judges checked their numbers against the GAMS
archive, `datasets/runs.parquet` and the run records; a synthesis drafted this record; two adversarial verifiers
recomputed it and refused it as drafted (2 blockers, 7 majors, 20 minors); a second round of two verifiers on the
revision found 1 blocker and 5 majors more (a block rule that replaced the governor's rests, the default regime's
budget, part A's budget, an asymmetric exclusion list, memory-probe coverage). This version addresses both rounds (list
at the end). Supersedes parts of [[D-020 MIPfeas harness — GAMS's clock and integral, paired calibration of cuOpt 26.08]].

## Why not copy GAMS
- **Heat.** GAMS ran cuOpt on all 20 cores (median 1,948 % CPU) for 600 s. Never measured here; nine busy X925 cores
  alone (screening, 300 s, no GPU) ended at a median 87 °C and peaked at 92.7 °C. The host fan governor locks the GPU
  at 2,200 MHz when a zone stays ≥ 85 °C for 20 s at fan MAX — CPU-only screening load triggered it on 10-06
  02:52–07:00 — and a capped run passes our 2,000-MHz health floor. The box shut itself off on 10-01 under ~1.5 h of
  full GPU load.
- **Footprint.** Development compares A9 (5 cores + GPU) with A6 (cuOpt on 3 A725 cores + GPU): "your method had 5
  cores, the baseline 3".
- **Selection.** D-019's hard part ("our cuOpt-alone PI ≥ 0.1") is selected on the run A9 is compared with, which gives
  A9 a free gain by regression to the mean: up to ~3 points of mean P on GAMS's paired runs (3.4 when 26.08 selects,
  0.2 when 26.04 selects; one pair of runs).
- **Start-up dominates the published aggregate.** A uniform delay moves cuOpt-GB10's shift-0.001 geo-mean by +3 / +9 /
  +26 / +65 % at 0.1 / 0.3 / 1 / 3 s, the hard-part mean by +0.5 % at 1 s and +1.4 % at 3 s. Berk (10-06): "we don't
  expect to solve simple problems a few seconds faster at all".
- **Corrections to 10-11 numbers.** The "perfect race" 0.0304 / hard mean 0.526 is the per-instance minimum of P (a
  virtual best); the pointwise race of the same GAMS traces (cuOpt-GB10 26.08, SCIP 10) is **0.0262** (−33 %) / **0.508**,
  and 0.587 → 0.490 on the unseen hard part. **15** of our 17 development / class-probe instances are among the 233, not
  13 (10 in cuOpt's hard part). D-020's "49 GB" for nw04 is 46.9 GiB; its sample has 22 (not 23) runs to 600 s, 3.95 h.

## The standard in one paragraph
ITX-1 keeps MIPfeas's 233 instances, T = 600 s, GAMS's integral exactly (`pt_gams`, on checked incumbents) and its call
clock. Every compared GPU arm gets the same machine share **F** — the cores, GPU and per-role memory caps of A9's frozen
layout — and the whole box is not used. cuOpt alone (A6F) uses all of F; SCIP alone (A0) keeps one core and is labelled
so; A9 is also measured against a race of A0 and A6F computed pointwise, which pays nothing for sharing the box.
Decisions use hard and weak sets fixed now from data independent of every v1 run, on the 218 instances never used in
development; the shift-0.001 geo-mean is reported, not decisive. A run ends when its incumbent reaches the proven
optimum (`pt_gams` unchanged). Runs go in tiers whose every prefix still decides the targets. Thermal gates use the
governor's own 85 °C line, a capped GPU clock voids a run, and a 600-s soak at F comes before any calibration run.

## Rules

### Instances and sets — fixed before any v1 run in `datasets/mipfeas_strata.csv` (hashed in prereg-v2)
- **R1 — the 233**, references = GAMS's `optimal_objective.csv` (= our `solu.txt` on all 233).
- **R2 — decisions on the 218 unseen** [Berk] (minus R29's fixed cuOpt list wherever "the 218" is a denominator). The
  15 development / class-probe instances are run and reported; no decision uses them.
- **R3 — independent hard and weak sets** [Berk].
  - **H_cu** = GAMS's cuOpt-GB10 26.08 `pt_gams` ≥ 0.1 or no solution: 75, **65 unseen**. D-019's own-run definition decides
    until Berk's yes and is reported beside afterwards; so is H_cu at P ≥ 0.5 (34, 30 unseen).
  - **W_SCIP** = our screening (seed 900, 300 s, 1 X925 core), Berthold P(300) ≥ 0.5 (D-013): 72 of the 231 screened,
    plus satellites2-40 and supportcase22 classed by GAMS's SCIP-10 row → 74, **62 unseen**, 17 of them outside H_cu
    (**W-only**). **E** = the other 136 unseen.
- **R4 — fixed samples** (`numpy.default_rng(20261012)`): an E-sample of 40 (10 per quartile of GAMS's cuOpt P) and a
  repeatability panel of 20 H_cu-unseen (10 with GAMS P ≥ 0.5 or none, 10 with 0.1 ≤ P < 0.5).

### Arms and footprint
- **R5 — equal allotment F.** F = the cores, GPU and per-role memory caps of A9's frozen layout (today X925 5–6, A725
  1–3, GPU; Redis on core 0 and the runner on 4 / 10–14 are shared infrastructure outside F). The soak (M2) qualifies
  the F it was run at; any change to F's cores (#30) repeats M2 before any calibration; at ≥ 10 cores R9's ladder and
  Berk's yes apply (a safety limit on D-019's "full-box A9 layout" milestone — Open for Berk, 1).
- **R6 — A6F = cuOpt alone on all of F**: `num_cpu_threads` = |F|, pinned to F; it is "our cuOpt alone" of D-019's
  external rows, whose footprint D-019 left open — choosing it is the harness rule Berk asked for on 10-11 (for part A,
  where D-002 pre-registered A6, it is a prereg-v2 item). Replaces A6 at no extra runs. Thin evidence that threads matter little: A6 (3 threads, 120 s) vs GAMS's 24-thread
  run, R 1.00 [0.89, 1.12] on the 15 development instances (10 in H_cu). A6F is probably at least as strong as A6, so
  the targets get harder.
- **R7 — A0 = SCIP alone, 1 X925 core per run** (4 concurrent CPU lanes, interference checked in M6 at D-002 §7's 3 %),
  harness-matched, CPU-only windows. Every A9-vs-A0 number carries both footprints; never called compute-equal.
- **R8 — race bar.** The **computed race** = at every moment the better of A0's and A6F's checked γ(t), same instance and
  seed. It has 6 cores and no sharing cost, so it flatters the race; "A9 below the race" is conservative. A **live A2F**
  (SCIP on F's first X925, cuOpt on the other |F|−1 cores) runs on the panel's 20 instances in its own blocks (T5, start
  temperature and fan state as covariates) to measure how much the computed race flatters a real one. The per-instance minimum of P is never called "the race". On GAMS's traces racing
  alone gives −17 % mean P on H_cu-unseen (0.834 [0.746, 0.908]).
- **R9 — no whole-box run in the standard.** A6m (D-020) stays built and out of v1. It runs only with Berk's yes for
  that window and then only through a ladder (10 threads on 10 cores, then A6m), each 600 s from fan AUTO, aborted at
  90 °C, failed by any governor cap.

### Clock, time limit, stop rule, solver settings
- **R10 — the call clock decides; claims must also hold on the launch clock** [Berk]. Call clock = t0 after the start barrier
  (D-020), comparable with GAMS's trace; launch clock = from the first process's exec. A target is written "met" only
  if it is met on both, otherwise "met on the call clock only" with the start-up cost. Measured (run.json, 69 completed A9 runs):
  SCIP ready 0.16 s, EA 1.51 s, cuOpt 3.07 s; A6 launch-to-t0 2.82 s. So in A9 cuOpt is the last process ready — the
  barrier holds SCIP and the EA for cuOpt, not the reverse (D-020's "release cuOpt when ready" is withdrawn) — and A9
  starts 0.2–0.25 s behind A6F (guard effect ≈ −0.005 to −0.007), about 2.9 s behind A0.
- **R11 — T = 600 s** (D-019, MIPfeas), hard stop at T + 30 s; P(300) from the same runs is a secondary (published rows
  are re-scorable at 300 s exactly).
- **R12 — stop at the proven optimum (external runs only).** A run ends when its best incumbent reaches GAMS γ =
  round((obj − z*)/max(|obj|, |z*|, 1), 6) ≤ 0 **and** Berthold's gap ≤ 1e-9 (z* ≈ 0 and |z*| < 1 occur on 16 of the
  233), obj recomputed from the stored vector (c·x + offset), or when a process proves optimality; only for references MIPLIB lists as proven optimal (232 of 233; never supportcase22). No process
  ever receives z*; stop kind `ref_reached`; the last checked incumbent is held to T; a checker rejection of the
  triggering incumbent voids the run, rerun once without the stop. `pt_gams` and Berthold P are unchanged by construction (the
  replay checks it); `pt_mf` may keep a residual gap ≤ 5e-7 to T. On GAMS's traces cuOpt's
  233-pass falls from 29.4 h as run to 22.8 h (116 instances reach γ = 0). Not applied to the confirmatory test.
- **R13 — gaps 0 for every cuOpt process**, A9's included (`mip_absolute_gap = mip_relative_gap = 0`; SCIP's are 0
  already). Today cuOpt's "optimal" at its default 1e-4 gap is DEFINITIVE in pair.py and ends A9's whole run; that could
  freeze γ ≤ 1e-4 (not yet observed: fhnw-binpack4-48 had reached γ = 0 before cuOpt's stop at 14.4 s).
- **R14 — cuOpt's thread environment** unset (OMP / OPENBLAS / MKL) for every cuOpt process in v1, confined by taskset;
  SCIP and the EA keep 1. R13 and R14 change A9's configuration: **a development test of A9 + R13 + R14 vs A9 (same-day
  A6F, rules written first, its own decision record) precedes the C1 freeze** — two changes bundled, a stated deviation
  from one variable: R14's own effect is expected to be small (cuOpt already uses 2.7–4.0 CPU-s per wall-s on 3 pinned
  cores under OMP = 1, but no run has had the variables unset), and M1 adds an env-only contrast (A6F with and without
  R14, 3 instances × 120 s).
- **R15 — persistent caches on the same terms for both GPU codes.** The EA's CuPy cache (its own directory, as now)
  and a **project-own CUDA JIT cache** for every itrea process (`CUDA_CACHE_PATH=~/.cache/itrea/cuda`,
  `CUDA_CACHE_MAXSIZE` large enough never to evict; the shared `~/.nv/ComputeCache` holds 1.1 GB / 30.7k files, is
  written by every CUDA program of the user and evicts when full) are warmed from the frozen code; an entry file in the
  project cache with mtime ≥ t0_epoch (index ignored) is a compile on the clock and voids the block (requeued once;
  such requeues are counted and reported).

### Metric and aggregation
- **R16 — metric.** Per instance: `pt_gams` (D-020's exact port, checked incumbents, call clock, T = 600; better than
  z* → γ = 0 and `beats_ref`). Also `pt_gams_launch`, Berthold P(600), `pt_mf`, P(300), time to first incumbent. Each
  run is exported as a GAMS `.trc` per clock; a test requires GAMS's own function on our `.trc` to give `pt_gams` to
  1e-12, so anyone can score us with GAMS's script.
- **R17 — aggregates.** Decisive: mean `pt_gams` on H_cu-unseen; the ε-floored guard on the 218; solutions found on the
  218. Reported, not decisive [Berk]: the shift-0.001 geo-mean on both clocks (233 and 218) and at shift 1/30 (the P of
  10 s without an incumbent; +7 / +21 % for 1 / 3 s of start-up instead of +26 / +65 %).

### Decisions and statistics (proposed for prereg-v2)
- **R18 — guard with an ε floor** [Berk]: d_i = (P_ctrl − P_arm)/max(P_ctrl, P_arm, ε), ε = 60 γ-seconds / T = 0.1 at
  T = 600. A 0.2 / 1 / 2.8-s offset on identical trajectories moves mean d by −0.005 / −0.026 / −0.074 over the 218
  (a D-002-style 10⁻³ floor: −0.075 / −0.202 / −0.326 — such a guard would fail on start-up
  jitter); GAMS's 26.04 vs
  26.08: −0.039 [−0.071, −0.009], SD 0.235.
- **R19 — D-019's external rows under ITX-1** [Berk]:
  - *Minimum:* (a) guard mean d_ε (A9 vs A6F) over the 218 ≥ −0.05 (point estimate, as D-019; harm "shown" only if the
    whole interval lies below −0.05); (b) solutions found (`pt_gams` < 2) by A9 ≥ A6F's **on the 218** (the 233 count
    reported beside). Replaces "A9's geo-mean ≤ our cuOpt-alone geo-mean".
  - *Target:* mean `pt_gams` ratio A9/A6F on H_cu-unseen ≤ 0.75 (D-019's −25 %); beside it "better shown" = the 95 %
    upper bound < 1, the improved / worse counts and the worst case.
  - *Geo-mean −10 % and stretch ≤ 0.0274:* reported with intervals, not decisive. Every literal D-019 reading is
    computed and reported beside.
- **R20 — per-instance labels.** "Improved" = d_ε ≥ +0.10, "worse" ≤ −0.10, reported with a sign test of improved vs
  worse; the sign test, not the share, says whether more instances improved than worsened. Under a version change alone
  about a third of H_cu-unseen read improved and a third worse (22 / 23 of 65), so a single label means little; the
  panel's 90th percentile of |d_ε| between its two seeds is reported as the per-instance noise scale.
- **R21 — tests.** Paired by instance; mean-P ratio and mean d_ε with 10,000-resample percentile bootstrap intervals
  (BCa as a sensitivity); Wilcoxon on d_ε. Minimum = the conjunction of R19 (a) and (b). Secondary, Holm-adjusted: A9 vs
  the computed race on H_cu-unseen; A9 vs A0 on W_SCIP-unseen (62). Published rows: descriptive paired ratios only. One
  analysis at a cut-off fixed in prereg-v2; no interim looks.
- **R22 — power, stated before any run** (proxies; A9's own spread may widen them). H_cu-unseen, n = 65: at a true
  −25 % the design shows "better" (upper bound < 1) with high probability, but the point estimate is ≤ 0.75 only about
  half the time — meeting the target reliably needs a true effect near −32 % (GAMS's version-pair spread) and
  larger, about −35 % or more, with development's A9-vs-A6 spread (SD of d 0.40–0.48 vs 0.29–0.32; an estimate). Guard SE ≈ 0.016 with GAMS's version-pair
  SD (0.235), ≈ 0.027–0.032 with development's A9-vs-A6 SD (0.40–0.48). W_SCIP-unseen, n = 62: 80 %-power detectable
  mean d ≈ 0.14 (σ ≈ 0.40). Published-comparable geo-mean ratio ×/÷ 1.07.

### Thermal and host safety (development and external runs now; part A only through prereg-v2)
- **R23 — the envelope of F (soak first).** F is accepted only if, in a 600-s soak (M2), per run from the 5-s fine log
  in steady state (from t0 + 120 s or fan MAX + 60 s, whichever comes first, to the end): hottest-zone p95 ≤ **85 °C** (the governor's line; 7 °C below our
  92 °C void, ~20 °C below the 104.8 °C trip), no 5-s sample ≥ 90 °C, no governor cap, busy SM median ≥ 2,350 MHz. The
  fan-AUTO start transient is reported separately (today's first A9 run: 80 / 87 / 82 °C at 09:41:39 / :44 / :50 in the
  5-s log, 77 °C in the 30-s log). Evidence today: A9 in-run maxima (120 s) median 66, p90 76, max 87 °C.
- **R24 — start gate.** A GPU run starts only if the hottest zone is < 80 °C (D-008), the rest flag is absent (D-009),
  the governor is alive (`fan-state` < 30 s old) and its cap is off. The governor's rests fall where they fall (D-009,
  unchanged; a rest between a block's arms delays the second arm, as now); arm order inside blocks is balanced AB/BA,
  and each arm's start temperature and fan state are recorded and reported (R28). CPU-only A0 lanes: < 86 °C (D-008's
  10-06 addendum; the code's gate is 88 °C today).
- **R25 — voids.** ≥ 92 °C: pause and void the in-flight block (as now). A GPU run with a busy SM median < **2,300 MHz**
  or overlapping a "clock cap on" line is a cap void (block void). A voided block is requeued once, only after the fan
  is back on AUTO, then skipped and listed (D-020's cap). The 2,000-MHz halt for the stuck-low-clock fault stays. No cap
  has engaged in any GPU-arm run so far (healthy runs 2,392–2,405 MHz).
- **R26 — window gates.** A window opens only if the median hottest zone over the last 60 s is ≤ 50 °C after 3 min idle
  with both LLM servers stopped (measured LLM-off idle on 10-11: 36–45 °C) and the governor is alive; it closes after 2
  thermal or cap voids; if more than 3 % of blocks are skipped for heat over the campaign, runs stop and F is
  re-characterised. Daytime only (07:00–24:00), lease held, rests honoured.
- **R27 — host safety layer.** Run scopes get unit names `itrea-run-<hash>-<role>-<epoch>`; [other repository]
  `thermal-watch` adds `itrea-run` to its 98 °C stop pattern (today it does not match our anonymous scopes). A run
  stopped that way is a **void** (cause `thermal_watch`), never an arm crash.
- **R28 — thermal record.** Per window, the slices of the 5-s fine log and `fan-governor.log`, the governor's source
  hash and the EC version go into the committed window record; per run, fan and cap state at start and end, the in-run
  ACPI maximum (`acpi_during`, D-020 fixes) and start temperature go into run.json; the report gives each arm's median
  start temperature and in-run p95.

### Memory
- **R29 — equal caps per role; symmetric only where the cap is shared.** cuOpt: RMM limit **24 GB** separate from the
  EA's CuPy pool (12 GB), host MemoryMax **24 GB**; SCIP 9 GB, EA host 6 GB, swap 0. Gates (D-003 arithmetic, file
  mailbox): A9 ≈ 97 GB (99 with Redis), A2F ≈ 78–80, A6F ≈ 69–71 GB against ≈ 110 GB with both LLM servers off.
  - The memory probe (M3) classifies the 8 instances whose RSS exceeded 14.4 GiB in GAMS's run (nw04 46.9, eilA101-2
    42.8, supportcase6 27.9, snp-02-004-104 22.6, square47 20.8, irp 17.6, proteindesign122trx11p8 15.8,
    proteindesign121hz512p9 15.3 GiB) by running A6F alone at RMM 40 / host 40 GB (gate 101 GB): a peak or cap hit above
    0.8 × 24 = 19.2 GB puts the instance on the **cuOpt "does not fit F" list, fixed before v1 and excluded from both
    arms** (it is cuOpt's cap in both). Our cuOpt's host maxrss has been 1.1–4.4× (median 2.1) below GAMS's whole-job RSS on the
    same instances; host RSS says nothing about RMM (GPU) need, so the v1 harness-void path below catches what M3 misses.
  - In v1, a kill or an RMM allocation error at **cuOpt's** cap is a harness void (rerun once, then listed separately,
    not added to the fixed list). A failure of A9's own processes (the EA's memory plan refusing, SCIP or the EA killed)
    is an **A9 outcome**: it counts against A9 in every decision, and A9's coverage of the 233 is reported. A0's SCIP
    kills count against A0.

### Seeds, run order, sealing
- **R30 — seeds.** Seed 2000 for every arm and instance (disjoint from screening 900/901, development 0–3,
  confirmatory 1000/1001); common random numbers per block. The panel adds seed 2001 for A9 and A6F.
- **R31 — blocks and tiers.** An external block = one instance × seed × {A9, A6F}, arms back to back in AB/BA order
  by hash, never split across windows; A2F runs in its own blocks (T5). A0 runs in its own CPU blocks (separate windows,
  covariates reported). Tiers, each in a seeded random order: **T1** H_cu-unseen (65); **T2** W-only (17) + E-sample
  (40); **T3** the rest of E (96) + the 15 development instances; **T4** the panel's seed 2001 (20); **T5** live A2F
  on the panel's 20 instances (seed 2000). If T3 is incomplete at the cut-off, the guard and
  solutions found on the 218 are stratified estimates (census of H_cu-unseen and W-only, 40 of 136 for E) — valid
  under ITX-1's readings. Under D-019's literal readings (the default until Berk answers Open 1) the minimum is a
  geo-mean over the 233, so T3 joins the never-cut core (budget below).
- **R32 — cut order.** (1) T5, (2) T4 20 → 12 → 0, (3) T3 — only under ITX-1's readings. Never cut: T1, T2 (and T3
  under the literal readings), A0 on H_cu ∪ W_SCIP unseen, the soak, the calibration, the gates, the confirmatory test.
- **R33 — sealing.** Before prereg-v2's tag no A9 outcome on an unseen instance is run or read. The calibration and the
  memory probe run A6F on 26.08 and expose A6F's outcomes on the 27 unseen calibration instances (6 of them in
  H_cu-unseen; 3 of the 30 are development instances) and the probe's instances: development (#26, #30) never consults
  them, and the decisions are reported with and without those instances. A0 may
  run before the tag with only validity read; it stands if the SCIP harness hash is unchanged at the tag. Any 26.08
  "insurance" external runs (Open for Berk, 3) are sealed (validity only) until the analysis cut-off or the
  contingency.

### Reporting
- **R34 — footprint table** per arm: cores by type, threads, CUDA processes, caps, and compute actually used
  (cgroup CPU-seconds, GPU-busy seconds, RMM / pool peaks).
- **R35 — per instance**, every arm, both clocks: `pt_gams`, P(600), P(300), time to first incumbent, start-up per
  process, stop reason, caps hit, voids and requeues by cause, start temperature, in-run p95 and maximum, busy SM
  median. Every aggregate names clock, shift, set, tier completion and seed count.
- **R36 — published rows** are re-scored from GAMS's traces on exactly our completed subset and printed with footprint,
  clock, version and machine named (e.g. "A9 on F (5 cores + GPU), call clock: x; GAMS's cuOpt-GB10 26.08 on 20 cores /
  24 threads: 0.0393"). No "beats" against a published row; CPU rows (Intel 285K) are context only.

### The confirmatory test (D-019 part A) — not decided here
Part A keeps its own design until prereg-v2 (#32). Proposed there, each [Berk]: A6F as its "cuOpt alone" (D-002
pre-registered A6); the same harness settings (R13–R15, R23–R29 — gaps 0, cuOpt RMM 24 GB instead of D-002 §3's 12,
the new void causes in D-002 §6); W_cuOpt labels from an independent A6F run on a non-evaluation seed (902), only for
the selected test instances (32 × 300 s ≈ 2.7 h), since no cuOpt screening exists and selecting on the evaluation runs
would bias it as R3 explains; live A2F as the "just run both" control (proposed on #32); its own ε-floored guard on
Berthold P at T = 300 (ε = 60 γ-seconds / 300 = 0.2) in place of the 10⁻³ floor; R12 not applied (its secondaries need
whole trajectories). Its budget is itemised below.

## Calibration (replaces D-020's A6m calibration)
A6F at F, cuOpt 26.08, seed 0, R13 + R14, RMM 24 GB, call clock, on D-020's 30-instance sample (unchanged), natural
gap-0 stops (no R12, so these runs also validate R12 offline).
- **Late-clock check:** median first-incumbent offset (ours − GAMS) on strata 0–1 within +0.3 s, reported per stratum
  (stratum 2 is thread-sensitive and only reported). A fail stops the calibration's reading: the barrier and the clock
  are diagnosed before any v1 run. (Measured: A6 at 3 threads − GAMS = −0.06 s median over the 12 seed-0 development / class-probe A6 runs, +0.05 s
  over all 42; chromaticindex1024-7 and kumeu +1.6 to +3.6 s.)
- **Footprint reading:** R = (SGM_ours + 0.001)/(SGM_GAMS + 0.001) with D-020's band 0.870–1.150 and feasible ±1. Inside
  the band: no footprint or harness difference larger than about 30 % is likely (the sample's interval is ×/÷ 1.28;
  8–14 % differences are not resolved), and the published row may be cited as a bridge. Outside it with the clock check
  passed: the difference is the footprint (and machine), stated as such; only then could the ladder (R9, Berk) be asked
  for. v1's targets are anchored on our own A6F in either case.
- **Cost:** 3.95 h at GAMS's natural stops (22 of 30 to 600 s), up to 5.0 h if A6F proves fewer; ≈ 8–10 h of window.

## Measurements first
- **M0 — offline, no box:** replay R12 on every recorded run and GAMS's 3,495 traces (`pt_gams` unchanged to 1e-12;
  other metrics within 5e-7); a shape-only memory plan of A9 on the 233 (an instance A9 cannot start on is predicted
  and reported as an A9 outcome, R29 — only cuOpt's cap fills the both-arms list); write and hash `datasets/mipfeas_strata.csv`.
- **M1 — smoke (~1 h):** GPU tests with cold caches, then 3 development instances × {A9, A6F, A2F} × 120 s: gaps 0, env
  unset (cuOpt CPU-s per wall-s), RMM 24, a `ref_reached` stop, named scopes, the start gate reading cap = 0, an RMM
  allocation error classified as a harness void (fake-process test).
- **M2 — soak at F (0.7 h of runs, ~1.5 h wall):** comp21-2idx and neos-4763324-toguru × {A9, A6F} × 600 s, the first
  from fan AUTO; the pool's pause threshold set to 90 °C for it (new `--pause-c`). Read R23's numbers, cuOpt CPU-s per
  wall-s (≈ its core count expected), the time from run end to 60 °C and to fan AUTO.
- **M3 — memory probe (1.0–1.35 h of runs):** A6F × 600 s at RMM 40 / host 40 GB on the 8 instances listed in R29.
- **M4 — calibration** (above). Stop rule: any thermal or cap void, or any run with steady-state p95 > 85 °C, in its
  first 2 h → back to M2's analysis.
- **M5 — optional A9 preview** at 600 s on the 15 development instances (1.7–1.84 h of runs): A9's 600-s heat and time
  to reference, replacing the A9 run-time proxy below.
- **M6 — A0 window opening (CPU):** 4 lanes for 30 min (keep 4 if the hottest zone stays < 85 °C, else 3); SCIP nodes/s
  alone vs beside 3 lanes on 3 development instances × 60 s — more than 3 % slower → fewer lanes.

## Box-time budget (planning; C1's windows replace the 0.55 efficiency with a measured one)
Per-run time: A6F = GAMS's cuOpt-GB10 time to γ = 0 (else its end, ≤ 600 s); A9 between GAMS's race time (low) and A6F
× 1.15 capped at T (high); 8 s overhead per run; window hours = run hours / 0.55 + 0.5 h per window (measured
efficiencies so far 0.59–0.81). Hours below are GPU run hours → window-hours.
- **C1 (Fri 10-23):** M1 0.6 + M2 0.7 + M3 1.0–1.35 + M4 4.0–5.0 + M5 1.7–1.8 = 8.0–9.5 h → ≈ 16–18 window-hours = **2
  windows**; plus the R13 development test (17 × 2 seeds × {A9, A9 with R13, A6F} × 120 s ≈ 3.4 h → ≈ 7 window-hours)
  and the other C1 development work.
- **v1 external (cuOpt 26.10):** T1 18.0–19.1, T2 8.5–9.4 (from the E-sample's expectation; recomputed once the strata
  file exists), T3 15.6–17.7, T4 5.6–5.9, T5 2.6–2.9, overhead 1.2 h.

| v1, GPU | run h | window-h |
|---|---|---|
| core under ITX-1's readings (T1 + T2, with their 8-s overheads) | 26.9–28.9 | ≈ 51–55 |
| core under D-019's literal readings (T1 + T2 + T3) | 42.9–47.0 | ≈ 82–90 |
| full external (T1–T5 + overhead) | 51.3–56.1 | ≈ 98–107 |
| part A as proposed: A9 + A6F (32 × 2 × 2 × 300 s) + W_cuOpt labels | 13.4 | ≈ 26 |
| + part A's live A2F (if adopted) | + 5.3 | + ≈ 10 |
| 26.10 re-baseline (A6F, 17 × 2 × 120 s) | 1.1 | ≈ 2.5 |
| **total, ITX-1 core / literal core / full** (low without part A's A2F, high with it) | | **≈ 79–94 / 110–129 / 126–146** |

≈ 8–10 / 11–13 / 13–15 ten-hour GPU windows. **CPU-only windows on top** (both LLM servers off): A0 on the 233 (24–33 h
serial → 6–8.3 h on 4 lanes) and part A's A0 (32 × 2 × 300 s ≈ 5.3 h serial) ≈ 2 windows; screening stage 2's
remainder (D-008 addendum) ≈ 1 window (to be confirmed). T = 300 s instead of 600 would cut the external run time by
about 40–44 %.

## Risks
- F may fail its envelope at 600 s (120-s A9 runs already touched 87 °C at a fan-AUTO start). Then v1 waits for Berk;
  the rules are not loosened to make runs count.
- The calendar: 26.10's date is unknown; the full set needs near-daily daytime windows with both LLM servers off.
- The fair controls are stronger (A6F on 5 cores; the pointwise race at −17 % on H_cu-unseen): D-019's targets may be
  missed because the comparison got fairer.
- One seed: A9-vs-A6F spread may exceed GAMS's version-pair noise (development SD 0.40–0.48); per-instance claims rest
  on the panel.
- The stop rule is new code (live γ from stored vectors): fake-process tests, M0's replay and the calibration's
  no-stop runs guard it.
- Version drift: v1 runs 26.10; H_cu, the calibration and the published rows are 26.08 (GAMS's 26.02 → 26.04 moved the
  geo-mean by R = 2.03, geo-means 0.0876 vs 0.0426).
- Power path: the 10-01 shutdown's cause is not settled; temperature rules do not bound it.

## What this supersedes
- **D-019 [Berk]:** the hard part → H_cu from GAMS's row (R3); decisions on the 218 (R2); the geo-mean minimum →
  ε-floored guard + solutions found (R18, R19); the geo-mean target and stretch → reported; targets judged on both
  clocks (R10). Not needing Berk (harness rules under his 10-11 instruction): "our cuOpt alone" = A6F at F (R6); the
  external run gains A0, the computed race and live A2F; the calibration applies to A6F at F; the external clock (call
  clock deciding, D-020 correction).
- **D-020:** the A6m calibration → A6F at F; the single whole-box block → soak and memory probe; env unset for A6m only →
  every cuOpt process (R14); "release cuOpt when ready" → withdrawn (R10); RMM 36 / host 24 → 24 / 24 (R29). **Stays:**
  the archive and its checks, the `pt_gams` port, both clocks, the 30-instance sample and R statistic with its band and
  ×/÷ 1.28 reading, the report-validity rules, the thermal requeue cap, the late-start check, seed as `random_seed`.
- **D-003:** cuOpt caps 12 GB RMM (= the EA's pool) / 12 GB host → 24 / 24, separate from the EA's pool (R29); "OMP =
  OPENBLAS = MKL = 1 in every process" → not for cuOpt processes (R14).
- **D-008 / D-009:** the GPU start gate gains governor-alive and cap-off (R24); cap voids (R25); the CPU start gate for
  A0 lanes 88 → 86 °C (as D-008's 10-06 addendum proposed); rests unchanged.
- **D-002 (prereg-v2 items, [Berk]):** §3's partner GPU memory 12 GB (RMM = EA pool) → 24 GB separate; §6's external
  void causes gain the cap void, `thermal_watch` and harness memory voids; §4/§6's 10⁻³-floored guard → ε-floored.

## Code changes (on `dev-mipfeas`, merged only when no pool can launch runs; every existing run hash stays identical)
A6F and A2F arms on F; a frozen harness-settings set (R13, R14, R29 caps); the R12 stop in pair.py's poll loop;
gates: governor alive, cap off, cap voids, `--pause-c`, `thermal_watch` void cause, RMM allocation errors as
harness voids; per-arm start temperature and fan state in run.json; named scopes;
a project-own CUDA JIT cache (`CUDA_CACHE_PATH`, `CUDA_CACHE_MAXSIZE`) and its entry check; `tools/derive_mipfeas_strata.py`; job writers for M1–M6 and the tiers; analysis (`pt_gams`
at 600 / 300 on both clocks, shift 1/30, computed race, d_ε, sign test, stratified estimates, `.trc` export); the M0
replay tool. Outside the repo: the one-line `thermal-watch` pattern edit in [other repository].

## Review trail
Blockers: M3 measured under the caps it sizes (now uncensored at 40 / 40 and a symmetric "does not fit F" list);
R19(b) on the 233 contradicted R2 (now on the 218; cut order conditional on Berk's readings). Majors: Berk-items list
extended (part A moved out); R22's "decided" restated; the ±0.10 band no longer claims to separate noise; |F| ceiling
and #30 link; cache rule referenced to t0; thermal-watch stop = void; part A given its own [Berk] list. Minors:
the 13 → 15 overlap, the race bar, the selection bias size, start-up lags, GiB units, the 30-s vs 5-s reading, idle
temperatures, budget highs capped at T, M2 / M3 hours, A9's 97 GB, R12's exactness limited to `pt_gams`, A0's lanes,
D-002 §7's 3 %, the window-gate median, the calibration's clock check, sealing. Second round (fixed here): the block
rule aligned to the governor's MAX cycle would have replaced its rests by ~3-min cool-downs (a fan policy left to Berk)
and could never open a 3-arm block — removed, A2F in its own blocks; R6 made consistent (external footprint = harness;
part A's = prereg-v2); budgets for both regimes and part A itemised; the exclusion list made symmetric only for cuOpt's
cap, A9-only failures count against A9; the probe covers all 8 instances above 14.4 GiB in GAMS's run; R12's stop also
requires Berthold's gap ≤ 1e-9 (z* near 0); R13 tested alone, R14 recorded inert; numbers (guard −0.005 at 0.2 s,
2.9 s vs A0, T2 / T4 highs, C1 = 2 windows, calibration 8–10 window-h, 27 unseen exposed, T = 300 saves ~40 %).
Third round (one verifier, fixed here): M0 no longer moves A9 non-starts to the both-arms list; a project-own CUDA JIT
cache (the shared one is written by other programs and evicts); budget rows recomputed (overheads in the core rows,
full 98–107, totals 79–94 / 110–129 / 126–146); 6 (not 7) H_cu-unseen in the calibration sample; maxrss ratio 1.1–4.4×;
the −0.06 s selection named; R13 + R14 bundled as a stated deviation with an env-only contrast in M1; R10 entirely
[Berk]; the ladder and stratified estimates as yes / no items; D-008's CPU gate listed.

## Open for Berk
1. **prereg-v2 readings** (yes / no per item): the hard part from GAMS's row; decisions on the 218 unseen; the minimum
   as the ε-floored guard + solutions found (instead of the geo-mean); the shift-0.001 geo-mean reported, not decisive;
   targets judged on both clocks (R10). For part A: A6F as "cuOpt alone", the harness settings (gaps 0, RMM 24, the new
   void causes), W_cuOpt from an A6F seed, live A2F, the ε-floored guard on Berthold P. Also yes / no: (a) a full-box
   A9 (≥ 10 cores, D-019's "full-box A9 layout" milestone) only through the heat ladder (R5, R9); (b) if T3 is cut,
   the guard and solutions found on the 218 become stratified estimates (R31).
2. **Fan policy (host, root):** a pre-emptive "experiment starting → fan MAX" hint in the fan governor (removes the
   fan-AUTO start transient), and with it short cool-downs instead of the escalating rests for long run sets (less
   window time, more fan-at-MAX time — to be measured). Until then the rests stay as they are.
3. **Calendar:** from 26.10's release (or Fri 11-06) to Tue 11-24: ≈ 8–10 ten-hour GPU windows for the ITX-1 core,
   11–13 under D-019's literal readings, 13–15 for the full set, plus ≈ 3 CPU-only windows — near-daily windows with
   both LLM servers off. May weeks 44–45 run the external core on 26.08 as sealed insurance (prereg-v2 tagged on the C1
   configuration first)? If windows are scarce, T = 300 s instead of 600 cuts the external run time by about 40 %
   (P300 and P600 rank instances alike, Spearman 0.996 for one solver) — your call.

## Addendum — Berk's answers (2026-10-11 13:2x, question panel)
1. **prereg-v2 readings: "Accept all"** — every item of Open 1 (the hard part from GAMS's row; decisions on the 218
   unseen; the ε-floored guard + solutions found as the minimum; the shift-0.001 geo-mean reported, not decisive; claims
   on both clocks; part A's A6F, harness settings, W_cuOpt from an A6F seed, live A2F, ε-floored guard on Berthold P;
   a full-box A9 only through the heat ladder; stratified estimates if T3 is cut) enters prereg-v2. D-019's literal
   readings are still computed and reported beside. The never-cut core is T1 + T2 (R31, R32).
2. **Fan policy: "Keep rests as now"** — no governor change; the fan-AUTO start transient stays, caught by R25's cap void.
3. **Calendar: "We'll cross that bridge when we come to it."** — no near-daily schedule or 26.08 insurance runs are
   planned now; the question returns when 26.10's date (or Fri 11-06) is known.
4. **Time limit: "Keep 600 s"** (R11 unchanged).

## M1 and M2 results (2026-10-11) — F accepted
- **M1 smoke** (15:12–15:33, 12/12, 0 void): every harness setting as specified (A6F cuOpt on 5 cores / 5 threads, A2F
  4 / 4, A9's cuOpt 3; gaps 0; env unset; RMM 24 / host 24; named scopes; no compile on the clock; SM 2,392–2,405 MHz;
  in-run maximum ≤ 78.5 °C). **R14 measured inert:** cuOpt CPU-s per wall-s 5.02 / 4.99 / 5.03 unset vs 5.02 / 4.98 /
  5.03 set.
- **M2 soak** (15:35–16:34, pool 20261011-153534-1276706, `--pause-c 90`; one 20-min governor rest between the comp21
  arms): 4/4 completed, 0 void. Per run, from the 5-s fine log, steady state = from t0 + 120 s or fan MAX + 60 s:

  | run | start °C | steady p95 | steady max | first 60 s max | ≥ 90 °C | caps | busy SM | GPU W (median) | cuOpt CPU-s / wall-s |
  |---|---|---|---|---|---|---|---|---|---|
  | comp21-2idx A6F | 43.7 (fan auto) | 70 | 70 | 68 | 0 | 0 | 2,392 | 18.9 | 4.97 |
  | comp21-2idx A9 | 38.5 | 68 | 68 | 64 | 0 | 0 | 2,392 | 21.8 | 2.99 |
  | toguru A6F | 52.7 | 73 | 76 | 70 | 0 | 0 | 2,392 | 24.8 | 4.88 |
  | toguru A9 | 53.8 | 66 | 67 | 67 | 0 | 0 | 2,392 | 15.3 | 2.97 |

  **R23 holds on every run → F (X925 5–6, A725 1–3, GPU) is accepted for v1** (p95 ≤ 85 °C, no sample ≥ 90 °C, no cap,
  SM ≥ 2,350 MHz). The whole box never exceeded 76 °C in 600-s runs at F, 16 °C below the void line.

## M3 result (2026-10-11 16:34–18:13) — the fixed "does not fit F" list: eilA101-2, nw04, supportcase6
Pool 20261011-163415-1522373 (A6F, cuOpt RMM 40 / host 40 GB, `cuopt_memory_void: false`): 6 completed, 2 failed — both
recorded as cap hits, not voided (scope Result oom-kill, confirmed):

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

(The 24-GB caps are GiB: MemoryMax 24576M, RMM 24 × 2³⁰ bytes.) The list is fixed now, before any v1 run: **eilA101-2
(in H_cu: GAMS cuOpt P 0.230), nw04 and supportcase6 (both in E)** are excluded from both arms of the external set.
Decision sets: H_cu-unseen 65 → **64**, the unseen set 218 → **215**. The M0 shape-only plan of A9 on the rest is still
owed.

## Phase 2 built, M0 run, the sets fixed (2026-10-12 12:5x, AgentRA, branch dev-itx2; not merged)
Code (CPU tests only; every existing arm's plan byte-identical old vs new — 35,337 plan records and 961 pool blocks of
the base's job files): R12's stop at the proven optimum in the runner (harness switch `ref_stop`; the triggering
incumbent checked exactly after the run, unconfirmed = void `ref_rejected`, rerun once with the stop off); cuOpt
SIGKILLed at once after a definitive outcome (switch `cuopt_kill_on_end`); preset `itx1` = ext + both (the v1 tier and
A0 jobs); `tools/derive_mipfeas_strata.py` → `datasets/mipfeas_strata.csv`; `make-jobs itx-v1 --tier T1..T5`
(no v1 file written: v1 runs on 26.10), `itx-a0`, `itx-m6` / `itx-m6a`; `itrea gams-trc` (GAMS's own script reproduces
`pt_gams` to 1e-12 on our exports and on its 233 GB10 26.08 traces re-exported); `itrea itx-report` (R17–R22, D-019's
literal readings beside, R36's published rows on our subset, M6's lane reading); build_runs' run-record columns;
`tools/m0_replay.py`.

**The sets (R3, R4, R31; sha256 `d2ad90ee040e4b6a046cda9abe67c6add86b48f64547288c2edc292a0c203842`).** Every count
above matches: H_cu 75 / 65 unseen (64 after M3's list), H_cu at 0.5: 34 / 30, W_SCIP 74 / 62, W-only 17, E 136, the
decision set 215. satellites2-40 and supportcase22 are in W_SCIP by Berthold's P(300) of GAMS's SCIP 10 trace (1.000
each). Choice (AgentRA, open to Berk): R4 was written before M3, so **both samples are drawn after M3's list** — the
E-sample from E minus nw04 and supportcase6 (134; 4 quartiles of GAMS's cuOpt P by D-020's stable-sort + array_split
pattern, 10 each), the panel from H_cu-unseen minus eilA101-2 (64: 10 with P >= 0.5 or none, 10 with 0.1 <= P < 0.5),
one `default_rng(20261012)` stream in that order. Tiers: T1 64, T2 57 (17 + 40), T3 109 (94 + 15 development), T4 /
T5 the panel (20).

**M0 replay of R12.** (a) Our runs — `datasets/runs.parquet` and `incumbents_checked.parquet` of the main checkout as
built 2026-10-11 14:09 (itx-m2 / m3 and the 10-12 M4 runs are not in it): 921 runs, 707 on =opt= references, 192 would
stop, 0 would be void (no triggering incumbent failed the check); `pt_gams` and its launch-clock value unchanged
(max difference 0), Berthold's P and `pt_mf` within 1.45e-11; 41.50 h as run → 39.60 h (1.90 h saved; 1.09 h in the
screening). (b) GAMS's 3,495 traces (3,480 on =opt=; supportcase22's 15 are never stopped): 1,588 reach the proven
optimum before 600 s; `pt_gams` unchanged (0), P and `pt_mf` within 7.04e-10. **Correction to § R12 above:** cuOpt-GB10
26.08's 233-pass falls from 29.39 h to **23.54 h with 105 instances stopped**, not 22.8 h / 116 — that estimate used
GAMS's gamma <= 0 alone; R12 also asks Berthold's gap <= 1e-9, which holds back 11 instances whose incumbent rounds to
gamma 0 but differs from z* by more (binkar10_1, graphdraw-domain, istanbul-no-cutoff, mas76, momentum1,
mushroom-best, neos-2978193-inde, neos-4722843-widden, roi2alpha3n4, supportcase26, supportcase7). Hours saved per
GAMS run 3.0–6.2 h of 23.7–31.6 h.

**M0 memory plan (A9 on the 233, shape only, `configs/itrea2-dev-rc-3way.yaml`):** no instance refused (gene dtype,
population fit in 12 GB, Rank 0 slot); the population is below 256 on supportcase12 (P = 174) and
neos-5114902-kasavu (192); the parse's peak RSS is at most 2.26 GB (square47), under the EA's 6-GB host cap. So no A9
non-start is predicted (R29's A9-outcome path stays for what the plan cannot see).

**Added beyond the code list above (each a safety or ordering control; open to a superseding decision):**
- R33 enforced: the pool refuses external jobs (split `ext`) of an arm with the EA unless `prereg-v2` is an ancestor
  of HEAD (A0 and A6F external jobs need prereg-v1 only).
- A job-level `priority` (blocks in increasing priority, each group in the usual seeded stratified order; no earlier
  job file has it): A0's H_cu ∪ W_SCIP unseen first (81 of 230); M6's three measured 60-s runs start while three load
  runs hold the other lanes.
- A0 and M6 run on single-core X925 lanes with `--layout screen --cpu-lanes 4` (or 1 for M6's alone runs), which
  needs no E0 flag — the pair layout's `--cpu-lanes` does.
- Not enforced: the pool does not compare the installed cuOpt version with a job's variant tag (`v1c2610`); the v1
  window's checklist must.

## Phase 2 review fixes (2026-10-12 14:0x, AgentRA, branch dev-itx2; not merged)
An adversarial review of phase 2 confirmed 12 findings (1 blocker, 3 major, 8 minor); each is fixed with a test. No
rule above changes: the code now does what R10, R12, R21, R31 and R33 say. Every pre-existing arm's plan is still
byte-identical old vs new (the base's 35,337 plan records and 961 pool blocks, re-run after the fixes).
- **R31 / R19 (blocker):** with T3 incomplete, the minimum and the R10 reading came from the plain mean over the pairs
  present. `itx-report` now reads the minimum on a stated basis: "full" (every decision pair present); "stratified"
  (T3 incomplete, census + E-sample complete: (a) the stratified guard >= −0.05, harm shown iff its interval's upper end
  < −0.05; (b) the stratified count of solutions found A9 − A6F >= 0); "incomplete" otherwise. In the reviewer's case
  (A9 +0.5 on H_cu-unseen, 0 on W-only, −0.7 on the E-sample, T3 not run) the plain guard was +0.033 and the stratified
  −0.287: the minimum now reads "not met", harm shown.
- **R10 / R32 (major):** the target is read only on the complete H_cu-unseen (T1 is never cut); with a pair missing,
  the target and the minimum read "incomplete (n of N)", never "met". Completeness is reported per basis (decision set,
  census + E-sample, T3); a gap in T1 is no longer called a T3 gap.
- **R33, sealing (major):** the v1 writer's A9 default was the development configuration and the pool checked only
  the tag. Now `make-jobs itx-v1` writes A9 on `configs/frozen-v2.yaml` (the C1 freeze, committed in the commit tagged
  `prereg-v2`) and refuses an A9 tier while that file does not exist; the pool refuses an external A9 job unless the
  file equals the tagged one and the job's configuration hashes to it (no overrides); every external job needs a clean
  tree under `src/` and `configs/`; `itx-report` takes only runs with the registered itx1 harness and, for A9, that
  configuration (`--frozen` for a development reading). Not done: re-deriving each run's full run hash (it also folds
  in the lane's footprint) in the analysis.
- **R33, reading (major + minor):** the strata file gains `exposed` = D-020's calibration sample ∪ M3's probe (37; 34
  unseen; 31 in the decision set: T1 8, T2 11, T3 12) and `itx-report` gives every decisive aggregate and reading again
  without them (184 decision instances, 56 of H_cu-unseen). Choice (AgentRA, open to Berk): all 8 probe instances are
  flagged (the 3 on the does-not-fit list are out of every decision anyway); a further pre-tag exposure joins the list
  before prereg-v2.
- **The strata file's floats (minor):** GAMS's `primalintegral.csv` is now read round trip; `p_cuopt` / `p_scip10` are
  the published decimals (173 / 135 values moved by ≤ 2.2e-16). Every set, sample and tier is unchanged. **New sha256
  `c0221d429f3368c4181177cbd33b3b3717ddad8946bf712ca5440f6fc4ab35e5`** (replaces `d2ad90ee…` above; not yet registered).
- **R33, A0's hash (minor):** new harness switch `scoped_hash` (in `itx1`): an arm without cuOpt hashes only the keys
  that act on it, so a change to cuOpt's settings before the tag (C1, R13's test) leaves A0's run ids and runs
  standing — "A0 stands if the SCIP harness hash is unchanged" is now a property of the run id.
- **R12's trigger (minor):** the runner's trigger is the earliest qualifying record over every process (as M0's replay
  judges), not the first role read in a poll; run.json lists every role's candidate.
- **R21 (minor):** `--bca` adds BCa intervals to every mean d_ε (target, guard, secondaries), not only the ratio.
- **R10 (minor):** "met on the call clock only" carries each arm's median / p90 start-up and the launch clock's figure.
- **R3 / R8 (minor):** the target's figures on H_cu at P ≥ 0.5 (30) and live A2F vs the computed race on the panel are
  reported beside.
- **R16 (minor):** `gams_trc`'s docstring no longer claims bit-exact read-back by GAMS's `read_csv` (pandas' default
  parser: within ~1e-12 relative; exact with `float_precision="round_trip"`).
- **Window skill (minor):** the CPU windows' memory is 4 × 9 + 20 = 56 GB (one lane 29), not "4 × 29".
