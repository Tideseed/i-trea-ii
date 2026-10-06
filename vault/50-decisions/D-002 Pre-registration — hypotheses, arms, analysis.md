# D-002 Pre-registration — hypotheses, arms, analysis (v3.1, 2026-10-06, before any run)

**Status:** v3.1 — binding from the `prereg-v1` tag. It becomes binding when tagged `prereg-v1` in git, which happens after the analysis code exists
and before the first timed run on any candidate instance (the screening pass). Any later change is a new, dated
decision saying what changed and why; results under the old plan are still reported.

**Changelog.** v3 → v3.1 (2026-10-06 03:0x, before any run; from the adversarial review of the build,
`80-reviews/2026-10-06 build review.md`): screening on 9 lanes (10 × 9 GiB + 20 GB reserve exceeds the free memory);
stage 2 always for PGLib-UC and ML4CO (their stage-1 reference is the run's own best); PGLib split by base case; ML4CO
list pinned; memory-fit rule = the EA's memory plan; tuning-side family cap; P-degenerate definition and their exclusion
from the P strata; solver-limit stops flagged; frozen solver settings file; kernel compilation on the clock voids a
block; partner GPU memory 12 GB.
v2 → v3 (2026-10-06 00:0x, before any run): Berk set the main claim — the coupled system beats
**either** method alone ([[D-007 Main claim — the coupling beats either method alone]]). H1 is now the pair H1a
(A3 < A0) and H1b (A3 < A5), both required; the uncoupled portfolio and SCIP‖cuOpt become secondaries; I-TREA (2015)
is the starting point only (no part-I-faithful arm).
v1 (23:0x) → v2 (23:4x), before any run. Berk supplied the part-I manuscript, so the rank names follow
I-TREA: Rank 0 infeasible good, Rank 1 feasible good, Rank 2 rejects ([[D-004 Mapping I-TREA (2015) to I-TREA II]]).
A three-lens design critique followed ([[2026-10-05 design critique — response]]). Changes made in response:
- the pool is chosen on SCIP alone, from a fresh screening pass;
- 2 seeds × 32 test instances instead of 3 × 20;
- H1 is the only primary hypothesis, with a non-inferiority margin;
- the per-instance effect is bounded;
- one clock rule applies to every arm;
- the SCIP controls are harness-matched and tuned;
- cuOpt and the EA get the same partner footprint;
- time-to-target endpoints are added;
- every incumbent is checked;
- a GPU-health fault voids the whole block.

## 1. Sequence (each step committed before the next)
1. Build and pass the E0 correctness tests (toy models only), then commit and tag **`prereg-v1`**.
2. **Screening pass (SCIP only, CPU)** with SCIP default and the exchange plugin inert, 300 s, one run per X925
   core on **9 lanes** (cores 5–9 and 15–18; core 19 idle — 10 × 9 GiB plus the 20 GB reserve exceeds the ~109 GB
   free with both model servers stopped), 9 GiB cap (SCIP's own limit 8 GiB), both local model servers stopped. Two
   stages:
   - stage 1: every candidate below with seed **900**;
   - stage 2: seed **901** — for MIPLIB only when stage-1 P(300) ≥ 0.05 against the published `.solu` value or there
     is no solution; for PGLib-UC and ML4CO **always** (their stage-1 reference is the run's own best value, so a P
     rule on one seed would measure time to its own result).
   A MIPLIB instance that stage 1 solves with P(300) < 0.05 cannot reach any stratum, so it needs no second seed.
3. **Pool freeze** (rules in §2): write `datasets/hard_pool.csv` and the split, commit, tag **`pool-v1`**.
4. **Reference values frozen** (§4): write `datasets/reference.csv`, commit.
5. **E0 calibration** (§7): CPU/GPU interference, exchange transport ([[D-006 Exchange platform — Redis, file mailbox as fallback]]), callback windows.
6. **E1** (tuning set): EA alone vs cuOpt alone, local-search throughput, gates in §7.
7. **E4** (tuning set, capped budget): choose the configuration and A0*'s emphasis; write `configs/frozen-e2.yaml`
   and `configs/frozen-e2-solver.json` (emphasis, cuOpt threads, partner GPU memory 12 GB); tag **`frozen-e2`**.
8. **E2/E3** confirmatory runs on the test set (§3–§6). The runner refuses test-set instances unless the config hash
   equals that of `frozen-e2` and `prereg-v1` is an ancestor of HEAD.

## 2. Instance pool, strata, split
**Candidates (all on disk, no download):**
- the 240 MIPLIB 2017 benchmark instances, minus the 7 marked `=inf=` (infeasible) in `solu.txt`;
- PGLib-UC `ca` and `rts_gmlc` cases (`ferc` is exploratory only);
- ML4CO item placement: 60 instances — `valid/item_placement_9900` … `9924` (the first 25 of the 100 on disk, the
  ones optopt used) and `train/item_placement_0` … `34`.

`supportcase22` (open) is used only in E5. Hardness is measured on **SCIP alone** in the screening pass. It is never
taken from cuOpt or from the optopt grid, which mixes cuOpt into "best strategy" and would bias H1.

**Strata and quotas (test / tuning):**

| stratum | rule (screening, both seeds) | test | tuning |
|---|---|---|---|
| MIPLIB optimisation-hard | SCIP feasible on both seeds, mean P(300) ≥ 0.10 | 16 | 8 |
| MIPLIB feasibility-hard | SCIP finds no feasible solution on ≥ 1 seed | 4 | 2 |
| PGLib-UC | P(300) ≥ 0.10 or no solution; `rts_gmlc` held out of tuning (unseen family) | 6 | 3 (`ca` only) |
| ML4CO item placement | top half by P(300) | 6 | 3 |

**Selection within a stratum:** instances are sorted by mean screening P(300) (the feasibility stratum by time to first
feasible), and every k-th is taken until the quota is filled. This is deterministic and spreads difficulty.

**Split by group:**
- MIPLIB by family prefix (`neos-*`, `supportcase*`, `comp*`, …); one family is never on both sides;
- PGLib by **base case**: one `ca` day together with its reserve variants, each `rts_gmlc` day alone; `ca` days may
  be on both sides, `rts_gmlc` on the test side only (the unseen family);
- ML4CO by index.

No generator or family supplies more than 20 % of the test set, nor more than 3 of the 16 tuning instances. If an
instance does not fit — the EA's memory plan (`engine.plan_memory`, `EAConfig()` defaults, 12 GiB) gives P < 32 — it
is replaced by the next instance in its stratum, and the replacement is listed.

**P-degenerate instances** — reference value 0, or the held incumbent on the other side of 0 from the reference for
at least 5 % of [0, T] in screening — are not selected into the P-scored strata (they would leave the primary test);
they may enter the feasibility stratum if they qualify. **Solver-limit stops** (SCIP ended at its own memory or node
limit in a screening run) are flagged in the pool file and reported, not excluded.

## 3. Arms
Common random numbers: a block is one instance × seed, and the SCIP seed s is the same in every arm of a block.

| id | arm | footprint |
|---|---|---|
| A0 | SCIP alone, **harness-matched** (exchange plugin installed, null partner) | 1 X925 core |
| A0* | SCIP alone with the best static emphasis chosen on the tuning set (same tuning budget as the EA) | 1 X925 core |
| A1 | SCIP ‖ SCIP (second SCIP seed s + 10,000), no exchange | 2 X925 cores |
| **A2** | **SCIP ‖ cuOpt race, no exchange** ("as-shipped cuOpt") | SCIP: 1 X925; partner: GPU + 3 A725 cores (`num_cpu_threads = 3`), RMM limit 12 GB = EA pool (cuOpt needs ~10.5 GB, D-003) |
| **A3** | **SCIP ‖ I-TREA II on the GPU, two-way exchange** | SCIP: 1 X925; partner: GPU + the same 3 A725 cores, same GPU memory limit |
| A4 | SCIP ‖ I-TREA II on the CPU (NumPy), two-way | SCIP: 1 X925; partner: the same 3 A725 cores (OMP 1) |
| A5 | I-TREA II alone (GPU) | partner footprint |
| A3r | SCIP ‖ I-TREA II, **no exchange** (live uncoupled portfolio; 8 test instances × 2 seeds) | as A3 |
| A5x0 / A5y0 | A5 with x = 0 / with y = 0 (A5y0 exploratory) | partner footprint |
| A6 | cuOpt alone | partner footprint |
| H0/H2/H3 | A0/A2/A3 with HiGHS instead of SCIP | exploratory, only after the HiGHS user-solution test passes |

Confirmatory seeds are **1000 and 1001**, disjoint from screening (900/901) and tuning (0–99). A6 runs one seed.

## 4. Metrics
- **Primary: portfolio primal integral P(T)**, T = 300 s (Berthold 2013). It is the mean over [0, T] of the relative
  gap to the reference, with gap 1 before the first incumbent. It is built **only from incumbents that pass the
  independent exact checker**, using the recomputed objective, with trajectories truncated at T.
- **Reference (frozen before E2):**
  - MIPLIB: the `.solu` value.
  - Elsewhere: the best of the screening runs and of dedicated reference runs (SCIP 1 h × 2 seeds on CPU lanes).
  - A later run that beats the reference gets gap 0 and a logged "beats reference" event, checked exactly. A
    sensitivity analysis repeats the comparison with the best-of-study reference.
  - **P-degenerate** instances (reference 0, or a sign crossing) are flagged in the pool file and analysed on
    time-to-target only.
- **Secondary:**
  - SCIP-held P(T), i.e. what SCIP itself holds;
  - time to first feasible, to gap ≤ 5 % and to gap ≤ 1 %, censored at T;
  - final gap;
  - P(1,800 s) on the long-budget subset;
  - the number of solver incumbents that originated in the EA;
  - the frontier's size, spread and hypervolume ([[D-005 The frontier of diverse solutions]], exploratory);
  - the **MIPfeas-style integral** (Bussieck & Dirkse 2026; results on Mittelmann's site): the same relative gap but 2
    with no incumbent and denominator max(|z(t)|, |z*|, 1), with the shifted geometric mean (s = 0.001) — reported
    beside Berthold's P(T) because a published GB10 cuOpt baseline exists under it (0.0393 at 600 s on the 233 MIPLIB
    benchmark instances; [[D Benchmarks and metrics]]).
- **Integrity:**
  - every incumbent vector of every process is stored and checked;
  - posts are classified per run as posted / superseded-untried / tried-accepted / tried-rejected / pending-at-T;
  - solver-side exchange seconds and bytes are recorded;
  - crash counts are reported per arm.
- **Mechanism (descriptive):**
  - Each run writes a genealogy edge list: id, parents, parents' ranks at mating, origin bits, birth, key, V, feasible.
  - Depth-1 matched comparison: the odds of having a Rank 0 parent among {new population bests, accepted posts},
    against all children of the same generation, pooled by Mantel–Haenszel over generations and instances.
  - The causal claim about Rank 0 rests on the ablation (H4), not on lineage.

## 5. Hypotheses
- **Primary — H1 (the coupling beats either method alone), both parts required (intersection–union test, each at
  α = 0.05, no adjustment):**
  - H1a: A3 (SCIP ⇄ EA) has lower P(300) than A0 (SCIP alone);
  - H1b: A3 has lower P(300) than A5 (EA alone).
- **Secondary family (Holm):**
  - H2: A3 < the uncoupled portfolio — pointwise best of the A0 and A5 trajectories of the same block (computed;
    optimistic for the uncoupled side), checked against the live A3r subset — "the exchange adds value beyond running
    both";
  - H3: A3 < A2 (SCIP ‖ cuOpt on the same GPU);
  - H4: A3 < A1 (SCIP ‖ SCIP);
  - H5: A3 < A0* (tuned SCIP alone).
- **Mechanism (good infeasible solutions):** H6: A5 (x = 10 %) < A5x0, on the EA's own P(300).
- **Exploratory:**
  - A5y0 (protected elite), A4 vs A3 (the GPU's share), A6 (cuOpt alone);
  - the A3 ablations: exchange direction, injection band, bound demotion, frontier posting;
  - HiGHS as the partner;
  - 1,800 s;
  - strata;
  - the "no exchange window" subgroup: instances whose first SOLVING-stage callback comes after T/2.

## 6. Analysis
- **Effect per instance i** (seeds averaged): **d_i = (P_ctrl − P_arm) / max(P_ctrl, P_arm, 10⁻³)**, which lies in
  [−1, 1]; positive means the arm is better.
- **Test:** two-sided Wilcoxon signed-rank on d_i (`zero_method="wilcox"`; exact when n ≤ 50 without ties, else the
  normal approximation), α = 0.05.
- **Estimate:** mean d_i with a 95 % percentile bootstrap interval (10,000 resamples over instances). The ratio of means
  is reported as a secondary, for continuity with optopt.
- **Exclusion:** an instance leaves a comparison only if both arms have P = 1 on all seeds; it is then reported under
  time to first feasible.
- **Feasibility stratum:** analysed on time to first feasible (Prentice–Wilcoxon on seed medians, sign test as a
  robustness check), not inside the P test.
- **Decision rule, per comparison (H1a, H1b, H2, …):**
  - **supported** if p < 0.05 and the interval excludes 0;
  - **negative** ("no better than the control") only if the whole interval lies below **+0.05**, the
    non-inferiority margin;
  - **arm worse** if the whole interval lies below 0;
  - **inconclusive** otherwise.
  The main claim holds only if H1a **and** H1b are supported. Reading with H2: H1 supported and H2 supported = the
  coupling creates value beyond running both; H1 supported, H2 not = running both helps, the exchange is not shown to.
- **Minimum detectable effect** at n = 32: about 0.20–0.25 in d (from the critique's simulation on optopt X1 noise). A
  smaller true effect will most likely read "inconclusive".
- **Crashes:** a crashed run is kept and truncated at the crash. There are no reruns except for pre-registered external
  causes — GPU-health voiding, thermal pause, memory gate, a runner error (a process that dies before ready without a
  summary), and **kernel compilation after t0** (`compiled_on_clock`). Those void and rerun the whole block, and rerun
  counts are reported per arm.
- **Run order:**
  - each block's arm order comes from a Williams design indexed by hash(instance, seed, 20261005);
  - blocks run in a random order, with strata interleaved;
  - a block is never split across windows;
  - the window id is kept as a covariate.

## 7. Calibration and gates
- **E0 interference:** SCIP nodes/s and LP iterations/s on 6 instances under {idle, + GPU EA, + cuOpt, + 1 SCIP lane},
  2 repetitions each. If the slowdown is ≤ 3 %, the CPU arms run concurrently in one fixed lane layout for every block;
  otherwise they run serially.
- **E0 transport:** the measurement in D-006 (file mailbox vs Redis) decides the exchange platform.
- **E1 gates (tuning set), before E2:**
  - (a) A5 reaches a first feasible solution within 2× of A6's time on at least half of the tuning instances where A6
    finds one;
  - (b) the GPU local search's moves/s and time to first feasible are reported against a single-core CPU
    feasibility-jump reference.
  - If (a) fails: stop, report the engineering gap, and run no E2.
- **E4 tuning budget:** at most 12 configurations × 16 tuning instances × 120 s × 1 seed, every configuration logged.
  The winner is chosen by mean d against A5-default.
