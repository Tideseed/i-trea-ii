# AGENTS.md — companion for AI agents (and humans)

Companion to the research project **I-TREA II** (Interactive Three Ranks Evolutionary Algorithm, part II) by Berk
Orbay (berk.orbay@tideseed.com). Repository: `Tideseed/i-trea-ii` (private while the study runs). Paths are relative
to the repository root; the code is the Python package `itrea` under `src/`. `CLAUDE.md` points here: this file holds
every rule for agents working in the repository.

**Status (2026-10-16): development (phase 1), which ends at checkpoint C1 on Sat 2026-10-17 (Berk 2026-10-15: "Assume
we are wrapping up phase 1 on Saturday"; D-019 addendum); no confirmatory run yet.** The pre-registration D-002 v3.1 is binding since
the tag `prereg-v1` (2026-10-06). The SCIP screening's stage 1 is complete (324 candidates, seed 900); stage 2 is
deferred (D-010 §5), so the hard pool is not frozen (#1). The algorithm is developed openly (D-010) at 120 s on
17 instances — the development set (9, D-010, since 2026-10-06, `datasets/dev_set.csv`) and the class probe (8, D-014,
since 2026-10-07, `datasets/classprobe_set.csv`), both permanently excluded from the test set; every development
result is descriptive and lives in `vault/30-results/Development log.md`. Development defaults: **A3** (SCIP ↔ EA) =
`configs/itrea2-dev-rc-ld.yaml` (lp_decode adopted, D-017); **A9** (SCIP ↔ EA ← cuOpt, D-015) and its matched control A3x =
`configs/itrea2-dev-rc-3way-ysync-ld0.yaml` since 2026-10-15 (EA hash `835c7c5a249c`, A9 run hash under ext `f345de0ab936`)
= the old default `configs/itrea2-dev-rc-3way.yaml` + lp_decode gated after the EA's LP worker (D-026) + the EA's CUDA
context at `yield` (D-029); by D-032's rule it is the configuration that freezes for phase 1 (`configs/frozen-v2.yaml`
is written on Saturday). **`make_jobs.ITX_A9_CONFIG` still names `-3way.yaml`** (the old default) until a `src/` change
after the running pool: a writer that uses it (A9 in `itx-m1` … `itx-m5`, `itx-r13`, `itx-r13f`; the `r13x` control of
`r1-bsync` and `r2-ld`) writes the old default; `itx-v1` takes `configs/frozen-v2.yaml` and `c1r-rerun` its pinned files.
Built and not adopted: Rank 1 by cuOpt's clearing rule (D-016), A9's GPU back-off (D-018 stage 1, 2026-10-11: every
outcome rule held, the mechanism failed on M1 / M3 / M5; an analysis bug — `_band` read the seed spread of the old
control `A9@bo0` — was fixed before the verdict) and its phase Y alone (D-023, 2026-10-14: failed M2, M5, T, G — the
back-off line is closed for v1). A9 pilot (seed 0, 7
instances, descriptive): vs SCIP alone 7 better / 0 worse, mean d +0.68 [+0.42, +0.90]; vs cuOpt alone 3 / 4, −0.07
[−0.42, +0.25].
**External harness.** D-020 (GAMS's MIPfeas archive, downloaded on Berk's yes; GAMS's primal integral ported exactly —
`gams_integral` equals GAMS's own function on all 3,495 of its traces) is narrowed by **D-021, our own standard ITX-1**,
whose [Berk] items Berk accepted on 2026-10-11 (section below). Its footprint F is accepted on M2's soak (steady p95
66–73 °C, maximum 76 °C over 4 runs of 600 s); M3 fixed the "does not fit F" list (eilA101-2, nw04, supportcase6;
decision set 215); phase 1 and phase 2 are merged (phase 2: 6f0339d, 2026-10-12; 1,009 CPU tests passed on main after
the merge; on 10-13 the runner's GPU tests passed — 4 passed, 1 skipped — and a cold-cache EA test). The M4
calibration (A6F on D-020's 30 instances, 600 s) is complete (30 of 30, 2026-10-15) and reads **OUTSIDE** the band
[0.870, 1.150]: R = 1.313 [1.091, 1.618] on the call clock (launch clock 1.963 [1.606, 2.408]), feasible 27 vs GAMS 27,
late-clock check passed (median −0.119 s over strata 0–1) — with the clock checked, the difference is the footprint F
(5 threads + the GPU against GAMS's 24-thread whole box) and the machine, so v1's targets stay anchored on our own A6F
(result note `vault/30-results/2026-10-11 D-021 measurements M1–M3 and the interim calibration.md`, addendum
2026-10-15; the interim 19-of-30 reading, R 1.390, is superseded). A0 (SCIP alone, `itx1`, 600 s) on the 81 instances
v1's decisions read (D-024 (f)): 80 completed + 1 failed (s100: SCIP did not stop at 600 s, killed at 701 s; kept as an
A0 failure; journal 2026-10-14 21:48).
**Staged testing (D-024, Berk 2026-10-13).** Performance runs first; targeted runs where a change can act, wider only
after a hint whose threshold is fixed before the runs (futility-only: a hint gates widening, never declares success).
Berk accepted v1's staging: stage 1 = the 20-instance panel plus the 5 H_cu-unseen instances where cuOpt finds no
solution, × {A9, A6F}, hint = ratio of means ≤ 0.95; the guard-only stage runs whatever the hint says; A0 only on the
81 instances a decision reads (only the 25 stage-1 instances if there is no hint, D-024 (f)).
**Phase 1's last week (2026-10-13 → 10-16; each verdict by its record's rules, fixed before any of its runs was read —
D-022's second amendment came with 7 of the 58 runs of its first pool done, none read; development, n = 2–6 seeds,
descriptive).** Adopted: **D-022** (74 runs) — A9 under ext (R13 + R14): valid, no harm on the 3 instances where
cuOpt's own 1e-4 stop occurs; mechanism met only vacuously (SCIP proves first), so the adoption rests on D-021's fairness
and GAMS fidelity; **D-026** (28 runs) — lp_decode inside A9 after the LP worker: hint 0.758 (≤ 0.90; item_placement P
0.330 → 0.198, sing44 0.473 → 0.411, lower on all four seeds of both), mechanisms and no-harm legs held; **D-029** (21
runs) — `yield`, guards over seeds 0–5 with D-025's failing seed kept: generations item_placement 1.020, comp21 1.152
(≥ 0.90), P +0.016 / +0.015 (≤ +0.05), ca at seeds 2–3 0.675 (≤ 0.95) — thresholds set with seeds 0–1 (8 of its 28 read
runs) already known, item_placement's generation bar loosened from D-025's 0.921 to 0.90 (D-029 amendment B1–B2); at
D-025's bars the verdict is the same (item 1.020 > 0.921; comp21 1.152 > 0.898 / 0.950), so this qualifies the claim, not
the result. Not adopted: **D-023** (66 runs: chromaticindex
d +0.030, needed +0.30; cuOpt's first incumbent inside A9 63.47 / 63.41 s vs A6F 10.87 / 10.52 s); **D-025** (R1, 20 runs
each) — bsync (hint 0.765; generation guard failed on item_placement and comp21, comp21 P +0.060 > band 0.050) and its
fallback ysync at seeds 0–1 (hint 0.834; item_placement's generation guard failed on one seed, 678 vs 1,480), reopened by
D-029. Diagnostics: **D-027** (22 runs) — rule 4 EA PRESENCE: cuOpt's first incumbent on chromaticindex1024-7 at 9.80 /
7.33 s alone on F, 9.24 / 8.21 s on 3 threads, 6.53 / 9.65 s beside SCIP, vs 62–66 s inside A9; rule 1 closed the
layout line, rule 3 ruled SCIP out; **D-028** (4 runs) — not R13 + R14 (A9@boY under r13-off 63.25 / 63.24 s vs L =
19.82 s). chromaticindex stays a known v1 loss; CUDA MPS is not for v1 (Berk 2026-10-15). **C1 readings** (A9 vs A6F in
the same window, seeds 0–1; hint = ratio of mean P(120) over the fixed 11 ≤ 0.95, futility-only; guard = mean bounded d
over the 17 ≥ −0.05): old default (D-022's Result, 10-13) 0.827 / +0.063; **frozen-v2 candidate (D-030, 68 runs,
10-15/16) HINT 0.809, GUARD +0.050**, read with cuOpt alone's SIGSEGV on comp21-2idx at both seeds kept as A6F outcomes
(the reading gains from them, A6F P 1; left out, hint and guard are INCOMPLETE); D-030's harm leg (cross-day, against
10-13's A9@r13x) fired on glass4 (d_i −0.088 vs −0.057; krka at floor) → **D-032** (12 runs, glass4 × seeds 0–3 in one
window: old default 0.2176, candidate 0.2233, lp_decode alone 0.2177; d −0.025 ≥ −0.057) → **DRIFT**: the candidate
freezes for phase 1 by D-032's rule (fixed before its runs); D-031's four-seed harm reading may still put which
configuration freezes to Berk. The go / no-go on C1 is Berk's. Carried to phase 2: #26 (not built), #35 two-way (v1.1, Berk
10-14), CUDA MPS (v1.1, Berk 10-15), chromaticindex as a known loss; due before prereg-v2, not Saturday: #36 and #32's
test-set size. Saturday: `configs/frozen-v2.yaml`, the phase-1 results note, checkpoint, docs, issues, the go / no-go
(D-019 addendum).

**Four-seed result (D-031) — filled in after the run.** *Placeholder — no D-031 number exists yet.* Set
`c1r-ysync-ld0-s23` (`jobs/d031.jsonl`, 68 runs at seeds 2–3 on the candidate, started 2026-10-16 08:57) is read with
D-030's seeds 0–1 by `tools/d031_report.py --d030-json logs/d030-reading.json`; its PHASE-1 INTERIM RESULT line goes
here once the 136 runs are read. Until then the phase-1 interim reading is D-030's two-seed one.

**cuOpt 26.10:** NVIDIA/cuopt#2073 is fixed by PR #2098 (merged 2026-10-12, in 26.10; Berk confirmed 10-13) and verified on the nightly `26.10.0a294` in
`~/venvs/cuopt-2610n` (presolve stays on with a SET callback; injected solutions accepted and exact; invalid offers
rejected); limitation: cuOpt polls SET callbacks only when its GPU population records a new best. A9 two-way through
it is #35 (not started). **Open:**
our exact checker accepted incumbents below MIPLIB's proven optimum on glass4 (cuOpt rejects that point) and mas74,
and rejected cuOpt incumbents on snowy (1) and neos-933966 (3) — a tolerance mismatch to investigate (#36). **v1 target
(D-019, Berk 2026-10-10):** checkpoint C1 on Sat 2026-10-17 (was Fri 10-23; D-019 addendum 2026-10-15; development
closes on cuOpt 26.08), v1 at the end of November 2026 (aim Fri 11-27, hard end Mon 11-30) on cuOpt 26.10, daytime
windows only; targets in the next-but-one section. Newest checkpoint: `vault/90-checkpoints/2026-10-14-2350.md`. Every hypothesis and target below is
pre-registered or planned, not a finding. Do not write, in any file, that I-TREA II "beats", "improves on" or
"matches" anything until the corresponding experiment note in `vault/30-results/` says so with its numbers. (History:
on 2026-10-06 01:4x the status read "built and tested on toy models; no commit, no tag, no experiment, no results" —
413 passed, 2 skipped.)

## Setting
- Machine: NVIDIA DGX Spark — GB10 (10 × Cortex-X925 @ 3.9 GHz = cores 5–9, 15–19; 10 × Cortex-A725 @ 2.8 GHz = cores
  0–4, 10–14), Blackwell GPU, 121 GiB unified memory shared by CPU and GPU. Ubuntu 24.04.4, kernel 6.17.0-1031-nvidia,
  driver 580.173.02, CUDA 13.0. The machine is shared with other sessions and with two local LLM servers.
- Software: Python 3.12.3; NumPy 2.4.6, SciPy 1.18.1; CuPy 14.2.0 (GPU) and NumPy (CPU) as one code path
  (`itrea.backend`); solvers SCIP 10 (PySCIPOpt 6.2.1, primary partner), HiGHS 1.15.1 (replication partner, after its
  user-solution round-trip test; also the EA's own LP and the lp_decode worker), NVIDIA cuOpt 26.08
  (`cuopt-cu13==26.8.*`). cuOpt is no longer only a control: alone it is arm A6, and since D-015 (2026-10-07) it is
  the third process of arm **A9** (SCIP ↔ EA ← cuOpt), where it streams its incumbents and bound to the EA on its own
  channels and the EA passes them on to SCIP. cuOpt never receives a solution mid-run: on the installed 26.08 its
  `SetSolutionCallback` disables its presolve (feature request NVIDIA/cuopt#2073, filed 2026-10-07). **Fixed in 26.10**
  by PR #2098 (merged 2026-10-12; Berk confirmed 2026-10-13) and verified on the nightly `26.10.0a294` in the separate
  venv `~/venvs/cuopt-2610n` (`tools/probe_cuopt_setsol.py`): presolve stays on, injected solutions in original order
  are accepted within 1–6 ms and echoed exactly, infeasible offers are rejected; cuOpt polls SET callbacks only when
  its GPU population records a new best (crna: 11 calls, all at 0.70–0.71 s; sing44: first call at 28.3 s). A9 stays
  one-way until a two-way arm has its own record and test (#35). A MIP start at launch is used with presolve on
  (verified on glass4 2026-10-07, ~10 s overhead per restart), but no arm uses it yet. **v1 runs on cuOpt
  26.10** once it is released (separate venv, API check, A6 re-baselined on the 17 instances; #31); if 26.10 is not out
  by Fri 2026-11-06, the v1 runs proceed on 26.08 and 26.10 follows as an addendum (D-019; to be confirmed with Berk
  then). Numba 0.64.0 (the CPU
  feasibility-jump reference); redis-py 8.1.0 for the exchange platform; pandas 3.0.3 / PyArrow 23.0.1 for the derived
  tables. All pinned in `pyproject.toml`. One entry point: `itrea <command>` (`src/itrea/cli.py`; `itrea --help`).
- Instances (public, on disk): MIPLIB 2017 benchmark, PGLib-UC unit commitment, ML4CO item placement
  (`vault/10-design/Datasets.md`). The confirmatory study will use a frozen pool of **hard** instances chosen on SCIP
  alone (`vault/10-design/Hard instance pool.md`; not frozen yet, #1). Development: the 17 instances above. v1's
  external run uses the 233 feasible MIPLIB 2017 benchmark instances of Mittelmann's MIPfeas
  (`data/miplib/benchmark-v2.test` minus its 7 infeasible ones; no download needed).
- Metric: the primal integral **P(T)** ∈ [0, 1] (Berthold 2013), built only from incumbents that pass the independent
  exact checker; T = 120 s in development, 300 s in the confirmatory test; definitions in
  `vault/10-design/Trace format and metrics.md`. Results are read in D-013's terms (weak sets first, the mean as a
  guard). v1's external metric is Mittelmann's MIPfeas primal integral (0–2, 600 s, shifted geometric mean, shift
  0.001; D-019).
- Predecessor: `berkorbay/optopt` (public, MIT, doi:10.5281/zenodo.22947442). Its CPU evolutionary partner beside SCIP
  (experiment X2) found no gain at 60 s and lost to SCIP ‖ SCIP; its strongest result was the SCIP/HiGHS ‖ cuOpt race.
  Code reused from it carries the line "adapted from berkorbay/optopt@4dd48395, MIT".

## The question, the aim, the v1 targets and the pre-registered hypotheses
Berk's brief (2026-10-05) and main claim (2026-10-06, `vault/50-decisions/D-007`): *"a GPU-based evolutionary
algorithm metaheuristic coupled with an exact solver can beat either single solution."* I-TREA — Berk's master's
research (Orbay 2011, METU M.S. thesis, open access, `orbay2011itrea`; thesis
manuscript, unpublished) — is the starting point; the principle carried over is **keeping good infeasible solutions**
(Rank 0). Part I reserved a share of the population for them as a floor; part II caps it, as Berk's brief says.

**Aim (2026-10-07, `vault/50-decisions/D-013`; it amends D-007).** Berk: *"I don't aim our coupling to be generally
better but to help both SCIP, cuOpt (and Highs if possible) in some of the hardest problems where they are the
weakest. Our performance metric is improvement upon the worst performances, of course with a regard to mean to some
degree."* Metric: for each solver B, the weak set W_B = the instances where B alone has P ≥ 0.5 on screening /
reference seeds never used for the evaluation; primary = the coupling against B alone on W_B (share improved, mean P
reduction, mean bounded d with its bootstrap interval and the Wilcoxon test, worst case after coupling); guard = the
mean over all instances, d ≥ −0.05 against each solver alone.

**v1 (`vault/50-decisions/D-019`, decided by Berk 2026-10-10).** v1 = arm A9 frozen (SCIP 10 ↔ GPU EA ← cuOpt on one
GB10) with the development defaults that survive D-018 stage 1 and #26, plus a pre-registered confirmatory test on
unseen hard instances and an external run under Mittelmann's MIPfeas protocol. The targets enter prereg-v2 verbatim:

| | minimum (v1 passes) | target | stretch |
|---|---|---|---|
| unseen hard instances: A9 vs SCIP alone on W_SCIP | mean d > 0, 95 % CI > 0 | mean P −40 %, ≥ 70 % of W_SCIP improved | −55 % |
| unseen hard instances: A9 vs cuOpt alone on W_cuOpt | mean d > 0, 95 % CI > 0 | mean P −25 %, ≥ 60 % of W_cuOpt improved | −40 % |
| guard, all instances | mean d ≥ −0.05 against each solver alone | | |
| MIPfeas protocol (233 instances, 600 s) | A9's geo-mean ≤ our own cuOpt-26.10-alone geo-mean on the same GB10 | mean PI −25 % on the hard part (our cuOpt-alone PI ≥ 0.1 or no solution) and geo-mean −10 % | geo-mean ≤ 0.0274 (SCIPConc-11, published) |

**External benchmark = our own standard D-021 (ITX-1), accepted by Berk 2026-10-11:** equal footprint F = A9's layout for every GPU arm (A6F = cuOpt alone on F), no whole-box run, call clock deciding with claims on both clocks, 600 s, independent hard / weak sets (GAMS's cuOpt-GB10 row, our screening), decisions on the unseen instances (215 after M3's "does not fit F" list), an ε-floored guard + solutions found as the minimum, the shift-0.001 geo-mean reported, not decisive. D-019's literal readings are reported beside. **Phase 2 built 2026-10-12** (branch `dev-itx2`; D-021's addendum): R12's stop at the proven optimum, the sets fixed in `datasets/mipfeas_strata.csv` (sha256 `c0221d429f33…` after the same day's review fixes — R33's `exposed` column, GAMS's values read round trip; every set and D-021 count unchanged), the v1 job writers, `itrea itx-report`, `itrea gams-trc` and M0 (replay: `pt_gams` unchanged; A9's memory plan: no instance refused). Review fixes (12 findings, D-021's addendum): the minimum read on R31's stratified estimates while T3 is incomplete, "incomplete (n of N)" readings, every decision also without R33's exposed instances, external A9 jobs only on `configs/frozen-v2.yaml` registered with prereg-v2, A0's run ids on its SCIP-side harness only, R12's trigger = the earliest record across processes.

The external targets are anchored to our own same-version cuOpt run because Mittelmann publishes only cuOpt 26.08 and
only aggregates (cuOpt-26.08 on GB10: 0.0393). Milestones (GitHub): **C1 checkpoint (2026-10-23)** — the date moved to
Sat 2026-10-17 (D-019 addendum of 2026-10-15; the milestone keeps its old name) — #23 (D-018 stage 1 decided; D-023
NOT ADOPTED), #26 (recombine each new partner incumbent with the elite: carried to phase 2, not built), #27 (lp_decode),
#30 (lp_decode inside A9: no new core, the decoder gated after the EA's LP worker — D-026 ADOPT), #29 (MIPfeas harness,
calibrated with cuOpt 26.08 alone on a ~30-instance sample against the published GB10 row: M4 OUTSIDE, the footprint),
an interim A9 result on the 17 instances × 2 seeds against SCIP alone and cuOpt alone (D-030; four seeds with D-031),
and a go / no-go for the final phase; **v1 (end of November 2026)** — #31 (cuOpt 26.10), #32 (prereg-v2, confirmatory and
MIPfeas runs), #1 (pool freeze), #12 (paper). The MIPfeas runs need ≈ 78 h of run time — about 8–10 long daytime
windows with the LLM servers off. The confirmatory test-set size is open: D-002 v3.1 says 32 × 2 seeds, D-010 §4
proposed 16 (#11), D-019's time estimate assumed ~32; prereg-v2 fixes it.

**Pre-registration v3.1 — history; binding since the tag `prereg-v1` (2026-10-06); none of it tested.**
`vault/50-decisions/D-002 Pre-registration — hypotheses, arms, analysis.md` was written for arm A3 (SCIP ⇄ GPU EA)
before D-013 and D-019 changed the aim, the primary arm (A9) and the targets. Any change is a new, dated decision;
those changes go into **prereg-v2** (#32), which is pending and must be written and tagged before any confirmatory run.
The v3.1 summary, unchanged:

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
| **Rank 1 — feasible good** | the best feasible individuals | ⌈y · min(#feasible, P)⌉, y = 10 % | protected every generation; equal-key twins leave Rank 1. Option, **not adopted** (D-016, `ea/elite.py`): `ranks.rank1_mode: clearing` — up to ⌈y · P⌉ members chosen best-first by cuOpt's clearing rule (any two differ in ≥ k integer variables, k = 1–10 by the number of integers); seeds 0–3 against the default: mean d −0.033 [−0.122, +0.050]; the size-only variant `ranks.elite_basis: population` was not adopted either (−0.000 [−0.044, +0.049]). The default stays `top` |
| **Rank 2 — rejects** | everything else | the rest of P | generational by default: only this generation's offspring, injected and immigrant individuals compete (the old Rank 2 dies); non-dominated sorting in (key, violation) with crowding distance |

Tournament order Rank 1 < Rank 0 < Rank 2. Before any incumbent exists the population is in a feasibility phase
(ranked by violation). The solver is the "interactive" counterpart: it supplies its incumbent, very good pool
solutions, its proven bound and its root LP, and receives the EA's better incumbents as new starting points. Since
D-011 SCIP also streams good infeasible node-LP points to the EA (the LP stream), and in A9 cuOpt streams its incumbents
to the EA one way (D-015). Beside the ranks the EA keeps a **frontier**: an archive of diverse near-optimal feasible
solutions, non-dominated in (objective, distance from the incumbent) (`vault/50-decisions/D-005`; exploratory metrics
only).

## Map
Status: **core** = written before the design critique of 2026-10-05 and revised after it; **built** = written by build
agents from 2026-10-06 on (the first build 00:0x–01:2x; development modules through 2026-10-09) and covered by the
test suite (CPU 816 passed and the GPU tests of back-off, lp_decode, elite and EA run 40 passed on 2026-10-09;
CPU 1,009 passed on 2026-10-12 after D-021 phase 2; non-GPU 1,256 passed, 1 xfailed on main after the R1–R3 merges of
2026-10-14; 1,339 passed on branch `c1-rerun` before its merge on 2026-10-15; GPU suites 45 passed, 1 skipped on
2026-10-15). Caveat:
`ea/population.py`, `ea/evaluate.py`, `ea/frontier.py`, `exchange/redis_mailbox.py`, `trace.py` and `problem.py` have
no test file of their own — they are exercised through the engine, EA-process and runner tests (default settings);
`ea/ranks.py` has `tests/test_ranks.py` and `tests/test_elite.py`, the file mailbox `tests/test_review_mailbox.py`.
Check a non-default branch before relying on it (`vault/80-reviews/2026-10-05 design critique — response.md`).

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
| `src/itrea/ea/elite.py` | Rank 1 as a distinct elite (D-016, #25): cuOpt's clearing rule on the device (`clearing_pick`, blocked distances; `min_diff_auto` = cuOpt's k by the number of integers) and the Rank 1 spread metrics (`elite_spread`, `r1_*`, logged in both modes); used by `ranks.py` under `ranks.rank1_mode: clearing` — 2026-10-08, run in D-016's test (seeds 0–3), not adopted | built |
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
| `src/itrea/ea/lp_decode.py` | the LP decode (`repair.continuous: lp_decode` = auto + a HiGHS worker process that fixes the integers of a few candidates per generation and solves the LP over the continuous variables; D-017, issue #27): pure selection / budget / screen functions and the `LPDecoder` worker; the runner refuses it for A9 / A3x (one EA core) unless the worker is gated after the EA's own HiGHS LP worker (`repair.lp_decode_gate: lp_worker` with `seeding.lp_source: highs_async`, nice `repair.lp_decode_nice` over the host — R2 of the 2026-10-14 C1 diagnosis, D-026, built 2026-10-14 in branch `r1-bsync`, review fixes the same day; run in D-026's test 2026-10-15 and adopted inside A9, D-026) — built 2026-10-08, review fixes 2026-10-09 (`vault/80-reviews/2026-10-08 lp_decode review (workflow).md`), GPU tests passed; run in D-017's test (24 runs, 2026-10-09) and **adopted as the A3 development default**; inside A9 it runs on the EA's core, gated after the LP worker (D-026 ADOPT, 2026-10-15; #30) | built |
| `src/itrea/ea/repair_clock.py` | the repair clock's controller: one lane-step horizon per generation from each horizon's measured gain per second (`repair_clock`, off by default; shadow mode logs only) — built 2026-10-06; Stage 1 (shadow) and Stage 2 (active) run 2026-10-07; **active in the development default** since D-012 addendum 2 | built |
| `src/itrea/ea/backoff.py` | A9's GPU back-off (D-018): a contribution-gated duty cycle — phase Y (reduced duty until the first incumbent anywhere), then a ladder of duties stepped down after a state-dependent patience in busy seconds and restored by material leads / SCIP-confirmed assists; a host-side token bucket the engine sleeps on between generations and between FJ launches (`backoff`, off by default, inert unless `exchange.from_cuopt`; modes active / shadow / forced) — built 2026-10-09; stage 0 (18 runs: forced duties ½ / ¼ / ⅛ against shadow bo0, A6 and a same-day A2) complete 2026-10-09: PROCEED; stage 1 (bo1 vs bo1s, 84 runs) 2026-10-11: NOT ADOPTED (mechanism); D-023 tested phase Y only (`configs/itrea2-dev-rc-3way-boY.yaml`: patience unbounded): NOT ADOPTED 2026-10-14 (failed M2, M5, T, G) | built |
| `src/itrea/ea/aggregate.py` | exact aggregation of two-variable equality rows (a presolve-lite for the genome) — 2026-10-06, **not wired into the EA** (the payoff probes on radiationm were negative) | built |
| `src/itrea/ea/engine.py` | the generation loop, exchange in/out, genealogy, frontier updates; since 2026-10-14 (branch `r1-bsync`) the EA host's settings (`host`, R1 / D-025: the CUDA scheduling flag before the backend exists — under CUDA's default the active flags are read back —, blocking timing events, the renice +N relative to the LP worker after its spawn), the LP decode's gate (R2 / D-026), the warm-up's >8-part concatenate, and in every run the LP worker's own clock and CPU in the `lp` event, the on-clock CPU (`cpu_clock`), `ls_launches` and `concat_parts_max` | built |
| `src/itrea/ea/cuda_sched.py` | R1 of the 2026-10-14 C1 diagnosis: the EA's primary CUDA context's scheduling flag (`host.cuda_sched`: blocking_sync / yield) through the driver API (`libcuda.so.1`: cuInit, cuDeviceGet, cuDevicePrimaryCtxSetFlags_v2), refused once the context is active, verified with cuDevicePrimaryCtxGetState after CuPy created it; under "auto" the flags are only read back (`read_state`, D-025's control); CPU tests use a fake driver — built 2026-10-14 (branch `r1-bsync`), GPU checks passed 2026-10-15 07:08–07:17 (`test_cuda_sched.py` 5 passed); `yield` adopted by D-029 (2026-10-15), `blocking_sync` not (D-025) | built |
| `src/itrea/ea/run.py` | EA process: `python -m itrea.ea.run …` | built |
| `src/itrea/exchange/mailbox.py` | file mailbox on `/dev/shm` (fallback transport) | core |
| `src/itrea/exchange/redis_mailbox.py` | Redis transport and shared pool; `open_mailbox(spec, role)` | core |
| `src/itrea/solvers/common.py` | what the solver processes share: arguments, barrier, recorder, exchange accounting, exit codes | built |
| `src/itrea/solvers/scip_run.py` | SCIP process; injection through a heuristic plugin (`trySol`) | built |
| `src/itrea/solvers/highs_run.py` | HiGHS process; user-solution callback | built |
| `src/itrea/solvers/cuopt_run.py` | cuOpt process; never receives solutions; RMM limit; in A9 (`--send`) it posts its incumbents and bound on its own channels (`to_ea_cuopt` / `info_cuopt`); `--mip-gap-zero` (both MIP gaps 0, GAMS's MIPfeas settings, arm A6m, D-020); the solve's parameters recorded as the summary's `settings` | built |
| `src/itrea/experiments/gates.py` | lease, memory gate, thermal gate, run lock, watchdog; readers of the host's fan governor (state file: alive, cap; log: clock-cap intervals) and of thermal-watch's 98 °C stop lines (D-021) | core |
| `src/itrea/experiments/gpu_health.py` | per-second SM clock, temperature and foreign-GPU sampling; block voiding (stuck clock < 2,000 MHz halts; D-021's cap floor 2,300 MHz in `CAP_FLOOR_MHZ`) | built |
| `src/itrea/experiments/exchange_ctl.py` | the `itrea-redis` container: up, status, flush one run's keys, down | built |
| `src/itrea/experiments/record_env.py` | versions, hardware, git state for every run record | built |
| `src/itrea/experiments/pair.py` | one run: launch the arm's processes, start barrier, stop, every exit code checked, `run.json`; A9's three processes and its matched control A3x (D-015); arm A6m (cuOpt as GAMS ran it for MIPfeas: 24 threads, MIP gaps 0, all 20 cores on the whole-box lane `full`; D-020); ITX-1 (D-021): arms A6F / A2F on F (the GPU lane's cores), the job-level `harness` (preset `ext`: cuOpt gaps 0, thread variables unset, RMM 24 GB separate from the EA's pool, host 24 GB, the project's CUDA JIT cache), named scopes `itrea-run-<run-hash>-<role>-<epoch>`, the void causes `thermal_watch`, `gpu_cap`, `cuopt_memory`, `compiled_on_clock`; phase 2 (2026-10-12): R12's stop at the proven optimum (`RefMonitor`: tails every process's `.part` records, the earliest qualifying record over every process, `ref_reached`, the exact check `check_ref_trigger`, void `ref_rejected`), cuOpt's SIGKILL after a definitive outcome, the optional harness switches `ref_stop` / `cuopt_kill_on_end` / `scoped_hash` (`hashed_harness`) and preset `itx1`; R2 of the 2026-10-14 C1 diagnosis (D-026, branch `r1-bsync`): lp_decode in A9 / A3x is accepted only with `repair.lp_decode_gate: lp_worker` and `seeding.lp_source: highs_async` | built |
| `src/itrea/experiments/pool.py` | a job file under every D-002/D-003 control: Williams arm order per block, blocks never split, gates per run; layouts `pair` (GPU lane g0 + `--cpu-lanes`), `screen` (nine X925 lanes), `full` (one lane, the whole box: A6m only, serving check implied, D-020's `--cuopt-mem-gb 24 --gpu-mem-gb 36`); D-021: the fan-governor start gate (`--no-governor`), `--pause-c`, `--cap-floor-mhz`, R26's breaker (`--max-heat-voids`), the new void causes rerun once then skipped, explicit `position` order, per-treatment memory arithmetic in `--dry-run`; phase 2: the job `priority` (block groups), R33's gate (external EA jobs need `prereg-v2` and `configs/frozen-v2.yaml` registered with it; external jobs a clean tree), the `ref_rejected` rerun with the stop off (kept on resume) | built |
| `src/itrea/experiments/make_jobs.py` | job files: `screen1`, `screen2`, `e1`, `e4`, `confirm`, `smoke`, `backoff0` / `backoff1` (D-018 stages 0 / 1), `mipfeas-cal` (D-020: A6m on the 30-instance sample; its file left `jobs/` when D-021's M4 replaced it), `itx-m1` … `itx-m5`, `itx-r13` (D-021 phase 1 → `jobs/itx-*.jsonl`), `itx-r13f` (D-022's follow-up at seeds 2–3), `backoff2 --harness ext\|r13-off [--instances amended\|stage1\|a,b]` (D-023; default the 11 of its 2026-10-13 amendment = 66 runs, `jobs/backoff2.jsonl`; `stage1` = D-018's 14), phase 2: `itx-v1 --tier T1..T5 [--cuopt 26.10] --out F` (the v1 tiers from `datasets/mipfeas_strata.csv`; A9 on `configs/frozen-v2.yaml`, refused while it does not exist; not written before 26.10), `itx-a0 [--select all\|d024\|stage1]`, `itx-m6`, `itx-m6a`, R1 / R2 of the 2026-10-14 C1 diagnosis (D-025 / D-026 targeted tests; the control = the development default's configuration re-run under the set's own label `A9@r1ctl` / `A9@r2ctl` — the same run hash, a new run id —; the two arms' order alternated over each instance's seeds, `position`): `r1-bsync --control r13x\|boY [--variant bsync\|bnice\|ysync] [--set S]` (20 runs, set `r1-<variant>`, `jobs/r1-<variant>.jsonl`), `r2-ld --base bsync\|boY-bsync --nice 10\|0 \| --ea-cpu X` or `--base r13x\|boY` (28 runs, set `r2-ld`, `jobs/r2-ld.jsonl`, labels `A9@r2ld` / `A9@r2ctl`); each job file was committed only after the verdict it depended on (D-023 for R1, D-025 for R2): `jobs/r1-bsync.jsonl`, `r1-ysync.jsonl` (`--variant ysync`), `r2-ld.jsonl` (`--base r13x`) (`jobs/itx-a0.jsonl` = all 230, `itx-a0-81.jsonl` = D-024 (f)'s 81 with the 25 v1 stage-1 instances at priority 0 and the other 56 at 1, `stage1` = the 25; `itx-m6.jsonl`, `itx-m6a.jsonl`), `r3-diag` (the C1 diagnosis note's R3, D-027 as amended 2026-10-14 — only what a rule reads: comp21-2idx, toguru, doon × {A6@r3 = cuOpt alone 3 threads on A725 1–3, A2@r3} + toguru, doon × A6F@r3 = A6F@r13x's job under its own variant + chromaticindex1024-7 × {A6@r3, A6F@r3, A2@r3} (rule 4, the 18:3x amendment), ext, 120 s, seeds 0–1 = 22 runs in 8 blocks, `jobs/r3-diag.jsonl`), `c1r-rerun --a9 ld0\|ysync-ld0` (D-030: the C1 re-read on the frozen-v2 candidate — D-022's C1 runs again, the 17 × {A9@c1r, A6F@c1r} × seeds 0–1, 120 s, ext = 68 runs in 34 blocks, set `c1r-<a9>`, `jobs/c1r-ld0.jsonl` / `jobs/c1r-ysync-ld0.jsonl`; A6F@c1r = A6F@r13x's job under a new label (run hash 82a371305f88, the same in both files: only one runs); the arms alternate by seed, A6F first at even seeds; run hashes pinned, the writer refuses a moved one; `c1r-ysync-ld0` ran 2026-10-15/16, D-030's Result), (`itrea make-jobs …`). Job files of this week written outside `make-jobs` (no `src/` edit while pools ran): `jobs/d028.jsonl` (D-028: `make-jobs backoff2 --harness r13-off --instances chromaticindex1024-7`, the bo1s rows dropped, set renamed `d028`), `jobs/d029.jsonl` (`tools/d029_report.py --write-jobs`: an unread warm-up run at `priority` 0, then 20 runs, set `r1f-ysync`), `jobs/d031.jsonl` (`tools/d031_report.py --write-jobs`: one call of `make_jobs.c1_rerun("ysync-ld0", seeds=(2, 3), set_="c1r-ysync-ld0-s23")`, 68 runs), `jobs/d032.jsonl` (D-032: from D-030's glass4 A9 job, config / variant / set / block / position / seed changed, 12 runs). **`ITX_A9_CONFIG` still names `configs/itrea2-dev-rc-3way.yaml`** (the old A9 default) until a `src/` change after the running pool; the new default is `-3way-ysync-ld0.yaml` (D-026 + D-029) | built |
| `src/itrea/benchmarks/hardset.py` | the hard pool: screening candidates, the stage-2 rule, strata, family split, freeze (`itrea screen-candidates`, `screen-stage2`, `pool-freeze`) | built |
| `src/itrea/analysis/metrics.py` | primal gap, P(T), primal-dual integral, times to target (adapted from optopt); GAMS's MIPfeas integral exactly (`gams_integral`, D-020: equals GAMS's own function on all 3,495 traces of their archive) | built |
| `src/itrea/analysis/check_solutions.py` | independent exact check of every stored incumbent (adapted from optopt) | built |
| `src/itrea/analysis/build_runs.py` | run directories → `datasets/runs.parquet`, `datasets/incumbents_checked.parquet`; MIPfeas columns `pt_gams` (call clock), `pt_gams_launch` (launch clock) and `startup_offset_s` (D-020); D-021's run-record columns appended (fan state and cap, ACPI at start and during, `void_info`, CUDA compile flag, `stop_kind`, R12's `ref_stop_*`); a `ref_reached` run reads like a run to T | built |
| `src/itrea/analysis/itx_strata.py` | D-021's instance sets: the rules, R33's `exposed` list (calibration sample + M3's probe), `datasets/mipfeas_strata.csv`'s reader (`read_strata`) and counts | built |
| `src/itrea/analysis/itx_report.py` | D-021's external analysis (`itrea itx-report`): completeness, target (read only on the complete H_cu-unseen), minimum (ε-floored guard + solutions found; basis full, or R31's stratified estimates while T3 is incomplete, else incomplete), the computed race and the Holm secondaries, BCa beside (`--bca`), geo-means, D-019's literal readings, H_cu at P >= 0.5 and live A2F vs the race beside, published rows on our subset, both clocks with the start-up cost, every decision again without R33's exposed instances; only runs with the registered settings (itx1; A9 on `configs/frozen-v2.yaml`, `--frozen`); `--m6` M6's lanes | built |
| `src/itrea/analysis/gams_trc.py` | GAMS `.trc` export per run and clock (`itrea gams-trc`, R16): GAMS's own script reproduces `pt_gams` to 1e-12 (its default `read_csv` returns our repr floats within ~1e-12 relative, not bit for bit) | built |
| `src/itrea/analysis/stats.py` | d_i, Wilcoxon, bootstrap, decision rule, censored endpoints | built |
| `src/itrea/analysis/lineage.py` | depth-1 matched Rank 0 odds ratio, Mantel–Haenszel (descriptive) | built |
| `src/itrea/analysis/prereg.py` | the pre-registered analysis of D-002 v3 from `runs.parquet` (`itrea prereg`) | built |
| `src/itrea/analysis/reproduce.py` | re-derive `datasets/` and `paper/generated/` and compare (`itrea reproduce`) | built |
| `src/itrea/analysis/mipfeas.py` | D-020's MIPfeas calibration (`itrea mipfeas-report`): A6m against GAMS's published cuOpt 26.08 GB10 row — paired R with a bootstrap interval on the call and launch clocks, feasible counts, strata, first-incumbent differences, GAMS's own GB10 26.04 / B200 rows, verdict PASS / FAIL / INCOMPLETE; since D-021 also `--arm A6F` (M4) with the late-clock check (one median of first incumbent ours − GAMS over strata 0–1 ≤ +0.3 s, per stratum reported), only runs under the ext harness at F counted, and D-021's reading INSIDE / OUTSIDE / CLOCK-FAIL; the GAMS archive's readers (`read_gams_trc`, z*, published P) and D-020's sample | built |
| `src/itrea/analysis/backoff.py` | D-018's stage-0 calibration rules and stage-1 criteria (M1–M5, outcome, fallbacks) applied to the run table and run directories, with a verdict (`itrea backoff-report --stage 0\|1`); the sanity rule gates on the measured duty only and reports the lane-steps ratio (D-018's second stated deviation; `--steps-gate` = the rule as first written); stage-1 control `A9@bo1s` — stage 0 read 2026-10-09; stage 1 read 2026-10-11 (`--failed keep`); `--stage 1y [--harness]` applies D-023 as amended 2026-10-13 to set `backoff2`: the 11 instances / 66 runs (T, G, the ladder and the mean guard on the nine, F one-sided on krka / fhnw, M1–M5 on all 11; G's noise = no sleep in phase Y, "still beats A6F" binds; other instances and seeds named as outside, read by no rule; "n of 66") | built |
| `tools/d022_report.py` | D-022's rules on the run table (`itrea d022-report [--followup]`): the second amendment's rule on the 3 instances where R13 can act (validity incl. each arm's harness, the mechanism, a one-sided floor / band no-harm leg; no mean), and C1's interim hint (ratio of means ≤ 0.95 on the fixed 11) and guard (mean d ≥ −0.05 over the 17) vs A6F, plus SCIP alone in D-013 terms | built |
| `tools/d027_report.py` | D-027's rules (the C1 diagnosis's R3, fixed before any run) on set `r3-diag` (`itrea d027-report [--failed both|exclude|keep] [--json F]`): validity as one gate (22 of 22 with P(120), harness ext, each run's layout — cores, `--cpu-threads`, X925 6 idle, no EA — read from its run.json, no compile on the clock; INVALID names the leg); rule 1 the thread effect A6@r3 vs A6F@r3 on toguru and doon (band max(0.05, the larger raw seed spread); present → the layout question to Berk, else closed for v1), rule 2 comp21's part 2 from cuOpt's final objective in A2@r3 (precondition A6@r3 ≤ 133 on both seeds; ≤ 133 → the EA's presence remains, the MPS question to Berk as one candidate), rule 3 SCIP beside cuOpt on toguru and doon, rule 4 (18:3x) cuOpt's first incumbent on chromaticindex1024-7 (`cuopt_t_first`, none = 120 s; L = 2 × max(A6F@r3): A6@r3 ≤ L on both seeds, then A2@r3 ≤ L → EA PRESENCE, the GPU-sharing / CUDA MPS question to Berk, ≥ 50 s → SCIP PRESENCE; A6@r3 ≥ 50 s on both → 3-THREAD LANE UNDER EXT, the layout question to Berk; else inconclusive at n = 2; prints the six t and L); A6F@r3 vs A6F@r13x beside on toguru and doon — read 2026-10-15 09:2x: rule 4 EA PRESENCE, rule 1 closed, rule 3 SCIP ruled out, rule 2 not read (D-027's Result) | built |
| `tools/c1r_report.py` | D-030's C1 re-read on set `c1r-<a9>` (`itrea c1r-report --a9 ld0\|ysync-ld0 [--failed both\|exclude\|keep] [--json F]`): validity as one gate (68 of 68 with P(120), harness ext, each run id = its job's, no compile on the clock — CuPy only on A9 —, the two runs of a block in one window, each A9 run's CUDA flag as the candidate's file writes it (gating since D-030's amendment of 2026-10-15); INVALID names the leg, a missing run alone INCOMPLETE); C1's hint (ratio of means A9 / A6F over the fixed 11 ≤ 0.95) and guard (mean bounded d over the 17 ≥ −0.05) by D-022's own functions; D-025 amendment 3's harm leg (the new A9 vs A9@r13x of 10-13 per instance, cross-day; band_i = max(0.05, the seed spread of A9@r13x or of A6F@r13x) — D-023's `band_i` as D-030's amendment reads it, never the treatment's — so each instance's threshold is fixed by 10-13's runs; harm CANNOT fire on rail01, cryptanalysiskb128n5obj16, doon and crna); beside, not criteria: the other readings' bands, D-022's floor marker (|ΔP| ≤ 0.005), the null lines (the same rule on A6F@c1r vs A6F@r13x and on the same-configuration A9 replicates of 10-15: 2 of 15 instance reads fire on noise), the cannot-fire instances' seeds and generations, vs SCIP alone on SCIP's weak set among all 17, voids, frozen-v2's hash; D-013 terms vs A6F and A0, the 10-13 reading beside (0.827 / +0.063 reproduced from the run table); the LP decode gate beside; failed runs kept, the letter beside; 'go / no-go is Berk's' — built 2026-10-15 (branch `c1-rerun`, merged d3fe14f); read 2026-10-16 08:2x on `c1r-ysync-ld0` (68 of 68, VALID; `--json logs/d030-reading.json`, D-030's Result) | built |
| `tools/d031_report.py` | D-031: C1 on the frozen-v2 candidate (`-3way-ysync-ld0.yaml`) read on four seeds — D-030's set `c1r-ysync-ld0` (seeds 0–1) and D-031's `c1r-ysync-ld0-s23` (seeds 2–3) as one reading (`.venv/bin/python tools/d031_report.py [--failed both\|exclude\|keep] [--json G] [--d030-json logs/d030-reading.json]`; no `itrea` entry — the script is the entry): validity as one gate over the 136 runs (`c1r_report`'s six legs on each set; INVALID names the set and the leg, else INCOMPLETE (n of 136)); D-022's hint and guard and D-025 amendment 3's harm leg (D-030's bands) on four-seed means, with seeds 0–1 alone, seeds 2–3 alone (the selection check, D-024 rule 2) and whether the halves agree beside; the comparison with the saved D-030 reading; one `PHASE-1 INTERIM RESULT` line (the four seeds only when the 136 read VALID, else D-030's reading); `--write-jobs jobs/d031.jsonl` (one call of `make_jobs.c1_rerun("ysync-ld0", seeds=(2, 3), set_="c1r-ysync-ld0-s23")`, pinned configuration and run hashes, refuses a used run id) — built 2026-10-15 (`tests/test_d031_report.py` + `test_c1r_report.py` 59 passed); its set runs since 2026-10-16 08:57, not read yet | built |
| `src/itrea/cli.py` | the `itrea` entry point: one table maps every command to its module | built |
| `configs/` | `itrea2-default.yaml` (all defaults, generated from `EAConfig()`; hash `9ac4a93bf93b`; `tests/test_configs.py` keeps it equal to the code). **Current development defaults** (D-010; never a confirmatory configuration): **A3** — `itrea2-dev-rc-ld.yaml` (`itrea2-dev-rc-3way` + `repair.continuous: lp_decode`, hash `f030e23909ad`, D-017); **A9 / A3x** — `itrea2-dev-rc-3way-ysync-ld0.yaml` since D-029's ADOPT (2026-10-15 15:0x; EA hash `835c7c5a249c`, A9 run hash under ext `f345de0ab936`) = `itrea2-dev-rc-3way.yaml` + lp_decode gated after the LP worker on the EA's core, nice 0 (D-026's ADOPT, 13:2x: `-3way-ld0.yaml`, `a05af2c732dc` / `341267810eed`) + `host.cuda_sched: yield` (D-029); the configuration that freezes for phase 1 by D-032's rule (DRIFT, 2026-10-16), copied to `configs/frozen-v2.yaml` on Saturday (not yet written); `make_jobs.ITX_A9_CONFIG` still names `-3way.yaml` until a `src/` change after the running pool; `itrea2-dev-rc-3way.yaml` (repair clock active, adaptive repair strategy, the EA's own HiGHS LP, LP stream, deep repair; hash `1d12ad490dd0`, A9 run hash `28e5a5e9c725` = A9@r13x's; the runner adds `exchange.from_cuopt` for A9 and refuses an ungated lp_decode there) stays as the record of the old default; **D-018 stage 1** — `itrea2-dev-rc-3way-bo1.yaml` (back-off active with the stage-0 values: D_Y ½, floor ⅛, yield_until 30 s, patiences 40 / 40; run hash `f67354be775d`) against `-bo1s.yaml` (the same values in shadow mode = the control; run hash `62bf82b236f6`); **D-023** — `itrea2-dev-rc-3way-boY.yaml` (phase Y only: bo1's values, active, patiences 100000 so the ladder never steps down; hash `5c8b535fb245`) against `-bo1s.yaml`, NOT ADOPTED (D-023's Result, 2026-10-14). **R1 / R2 of the 2026-10-14 C1 diagnosis** (each = its base + one change; EA hash / A9 run hash under ext; run 2026-10-15: R1 bsync NOT ADOPTED, its fallback ysync NOT ADOPTED at seeds 0–1 (D-025's Results), then ysync ADOPTED on seeds 0–5 (D-029); R2 ld0 ADOPTED, D-026): R1 on the A9 default — `-3way-bsync.yaml` (`host.cuda_sched: blocking_sync`; `305c8902d763` / `330a10a77556`), D-025's fallbacks `-bnice.yaml` ((a): `host.nice_after_lp_spawn: 10`, +10 relative to the LP worker, no context flag; `96a299330981` / `d266e4aaecbe`) and `-ysync.yaml` ((b): `yield`; `f81d6b6209f0` / `a08221cca653`); the same on boY — `-boY-bsync` (`29e61aa41619` / `b760e770cb34`), `-boY-bnice` (`5b1c4779c74b` / `b4f4af8376dd`), `-boY-ysync` (`666865eac174` / `22137f009005`); R2 — `-bsync-ld.yaml` (bsync + lp_decode gated after the LP worker, nice 10; `119c6e17f830` / `dc45bdbb771f`), `-bsync-ld0` (nice 0; `b7af15f4b62a` / `93a8ed06b59b`), the fallback without R1 `-ld0` (`a05af2c732dc` / `341267810eed`; adopted, D-026: part of A9's and A3x's default above), and on boY `-boY-bsync-ld` (`c296898cddc4` / `3ef4bf1254fe`), `-boY-bsync-ld0` (`2b6baa898edf` / `37a47a123c67`), `-boY-ld0` (`aabac30b5056` / `58f72d467a23`). **D-030 / D-031 / D-032**: `-3way-ysync-ld0.yaml` = `-ld0` + `host.cuda_sched: yield` (`835c7c5a249c` / `f345de0ab936`; built for D-030, its A9-layout cold-cache test passed 2026-10-15 15:07; run as A9@c1r in D-030 (seeds 0–1) and D-031 (seeds 2–3, running) and as A9@g4yl in D-032, beside `-3way.yaml` (A9@g4old) and `-3way-ld0.yaml` (A9@g4ld)). Every file's hash is pinned in `tests/test_cuda_sched.py` (`ALL_CONFIG_HASHES`). History: `-3way-bo0/-bof50/-bof25/-bof12.yaml` (D-018 stage-0 arms: shadow, forced duty ½ / ¼ / ⅛), `itrea2-adaptive.yaml`, `-adaptive-lps.yaml`, `-adaptive-lps-rc.yaml` (earlier development steps, D-010–D-012), `itrea1-faithful.yaml` (a record of part I's choices, not a planned comparison), `ablations/*.yaml` (D-002's E3) | core |
| `tools/serving-window.sh` | stop / restore the two local LLM servers for a window | core |
| `tools/r26-gate.sh` | D-021 R26's opening gate for a chain (`tools/r26-gate.sh [--deadline HH:MM] [--idle-s 180] [--limit-c 50]`): waits 3 min after the servers stopped, then until the hottest zone's median over the last 60 s is ≤ 50 °C and the fan governor is alive (fan-state written within 30 s, `cap=0`); exit 0 open, 1 at the deadline (the chain then restores) — a stopgap written 2026-10-13 after W-1013b opened warm, until the pool enforces the gate itself (#38); it opened the 10-15 and 10-16 windows at 07:08 (median 46.0 °C) | built |
| `tools/derive_mipfeas_strata.py` | D-021's sets → `datasets/mipfeas_strata.csv` (counts beside D-021's, sha256; `--check` re-derives) | built |
| `tools/m0_replay.py` | D-021 M0: R12 replayed offline on recorded runs and GAMS's 3,495 traces, hours saved, A9's shape-only memory plan over the 233 | built |
| `tools/bench_ls.py` | local-search throughput, GPU vs CPU reference (`itrea bench-ls`) | built |
| `tools/bsync_check.py` | R1's mechanism check before its window (GPU: only inside a lease): a launch loop of ~50-ms kernels synced as the EA syncs, on one core, alone (host self-CPU per wall-second) and beside a busy process pinned to the same core (its CPU share), per `--mode auto\|blocking_sync\|yield`, `--renice N` (D-025's fallback (a): the host +N relative to the stand-in after its spawn), 12 s per phase by default; run by `tests/test_cuda_sched.py` (marker gpu) with D-025's thresholds (blocking sync <= 0.25 alone; the stand-in >= 0.8 under yield and the renice) — built 2026-10-14 (branch `r1-bsync`); run 2026-10-15 07:08–07:17 before R1's pool (figures in `traces/bench/bsync/`; host self-CPU per wall-second auto 0.9998 → blocking_sync 0.0083) | built |
| `tools/d025_report.py` | D-025's rules on the run table and the run directories (`itrea d025-report [--set r1-bsync\|r1-bnice\|r1-ysync] [--control r13x\|boY]`): validity incl. the flag state per arm (read from `ea.summary.json`), the generation guard on seed means (>= 1 − max(0.05, the control's relative seed spread), the amendment) and the one-sided P bands (item_placement, comp21 and — the amendment — the three hint instances), the hint (ratio <= 0.95), the verdict and the fallback triggers; failed runs kept for the verdict, the letter beside — built 2026-10-14 (branch `r1-bsync`); read 2026-10-15 on `r1-bsync` and `r1-ysync` (both NOT ADOPTED, D-025's Results) | built |
| `tools/d026_report.py` | D-026's rules (`itrea d026-report --base … [--nice 10\|0]`): validity incl. the decoder in every treatment run and in none of the controls, the three mechanism checks (ready <= 10 s after the gate, decoded incumbents, ca's LP solve wall <= 1.10 x), the one-sided no-harm bands, the hint (<= 0.90), the gate's lag and A3@ld1 beside — built 2026-10-14 (branch `r1-bsync`); read 2026-10-15 13:2x on `r2-ld` with `--base r13x` (ADOPT, D-026's Result) | built |
| `tools/d029_report.py` | D-029's rules (as amended 2026-10-15 12:2x) over `r1-ysync` (seeds 0–1) + `r1f-ysync` (seeds 2–5) (`.venv/bin/python tools/d029_report.py [--failed both\|exclude\|keep] [--json F]`; no `itrea` entry): validity; rule 1 generations on seeds 0–5 (≥ 0.90), rule 2 P on seeds 0–5 (≤ +0.05), rule 3 ca on seeds 2–3 alone (≤ 0.95); a failed control run never decides a rule; rerun sets `r1f-ysync-<N>`; `--write-jobs` writes `jobs/d029.jsonl` (an unread warm-up run at `priority` 0, then the 20 read runs; pinned configuration paths and run hashes) — read 2026-10-15 15:0x: ADOPT (D-029's Result) | built |
| `tools/d029_oc.py` | D-029's operating characteristic: the pass rates its rules imply under a normal model fitted to the development default's same-window control pairs (a seeded simulation, no solver, no GPU; `--check` compares its data with the run table) — the source of the numbers in D-029's "Why these numbers" and its amendment | built |
| `src/itrea/analysis/c1_devtests.py` | what the D-025 / D-026 reports share: the selection by set (reused `#<set>` ids, attempts), each run's `run.json` / `ea.summary.json`, the common validity legs (complete, harness, config = the job's run id, compile, block), the one-sided seed-mean band and the ratio of means | built |
| `tools/d016_report.py` | D-016's analysis: A3@cl1 vs A3@rc1d and the size-only A3@tp1 (mechanism and outcome rules of the record) from `runs.parquet` and the run directories | built |
| `tools/d017_report.py` | D-017's analysis: A3@ld1 vs A3@rc1e (M0–M4 and the outcome rules) | built |
| `tools/probe_lp_decode.py` | probes the LP decode worker on stored vectors (CPU, read-only; D-017's gates before a window) | built |
| `tools/cuopt_verify_mipstart.py` | cuOpt 26.08 MIP-start and stopping check on glass4 (GPU: only inside a window; run 2026-10-07) | built |
| `tools/probe_cuopt_setsol.py`, `tools/probe_cuopt_setsol_report.py` | cuopt#2073's check: `SetSolutionCallback` with presolve (modes none / noop / inject / bad / raw) under any interpreter with cuOpt (the 26.08 venv or `~/venvs/cuopt-2610n`), and its report (GPU: only inside a lease; run 2026-10-13: 31 solves, most on the 26.10 nightly, the injection also on 26.08) | built |
| `tools/agg_probe.py`, `tools/agg_improve_probe.py` | CPU probes of `ea/aggregate.py` (feasibility and improvement payoff; 2026-10-06, negative on radiationm) | built |
| `tools/lps_smoke.py` | hand-launched smoke run of the LP stream (SCIP + the NumPy EA through a file mailbox) | built |
| `tools/screening-night.sh` | the screening chain: stage 1 → `build_runs` → stage 2 under one deadline (D-002 v3.1 §1.2) | built |
| `datasets/` | derived tables; `dev_set.csv` and `classprobe_set.csv` are the two lists excluded from the test set; `mipfeas_strata.csv` = D-021's fixed external sets (hashed in prereg-v2); `runs.parquet` / `incumbents_checked.parquet` are rebuilt by `itrea build-runs` and not committed yet; `hardness_from_optopt.csv` is background only | — |
| `jobs/`, `traces/` | job files; raw traces (not committed) and resource records (committed) | — |
| `paper/` | LaTeX skeleton (`main.tex`, `sections/`), `refs.bib` (legally readable works only) | skeleton |
| `vault/` | the research vault (Obsidian): journal, design, decisions D-NNN, results, literature, incidents, reviews, checkpoints | — |
| `.claude/skills/` | repository skills: `itrea-window`, `itrea-journal`, `itrea-result-note`, `itrea-new-experiment`, `itrea-literature`, `itrea-reproduce` | — |

## Rules for new experiments
The runner enforces most of these (`itrea.experiments.gates`, `vault/50-decisions/D-003`); documentation alone is not
a control. Skills: `itrea-window`, `itrea-new-experiment`.
- **Lease first.** No timed run without the box-window lease held by this project:
  `box-window claim --who "i-trea-ii (Claude Code)" --why "…" --until HH:MM [--stops ornith-vllm,llama-qwen38] [--mem GB]`;
  `box-window release` when done. Other sessions share the box; a held lease by someone else means wait. `box-window`
  refuses a lease longer than 12 h ("not a window; split it"): on 2026-10-14 a scheduled chain's `--until 20:15` from
  07:05 was refused and 7 h 36 min of a window were lost, so a scheduled chain claims ≤ 12 h, runs under
  `OnFailure=agent-failure-alert@`, exits 1 on any early stop, and is checked within 15 min of its start (skill
  `itrea-window`; the 10-15 and 10-16 chains, 11 h 50 min each, opened on schedule).
- **Daytime windows only.** No window in the night rest (00:00–07:00): Berk, 2026-10-10, "No nights" (D-019; before
  that D-009 asked for his yes for each night). Long run sets (v1's MIPfeas runs, ≈ 78 h) are split into daytime
  windows. The runner honours the host fan governor's rest flag (D-009): no new run starts while it exists.
- **Serving window.** Confirmatory runs need both local LLM servers stopped (vLLM `ornith-vllm` :8094, llama.cpp
  `llama-qwen38` :8092; Berk's yes, 2026-10-05); development windows stop both too whenever the memory gate needs it —
  A9 (two CUDA processes) needs ≈ 73 GB. Stop and restore only with `tools/serving-window.sh stop|restore`, schedule
  the restore on a timer before starting, and verify after restore: `is-active` **and** `is-enabled`, health
  endpoints answer, canaries resumed, lease released.
- **Memory gate.** Σ host `MemoryMax` + Σ CuPy pool limits + 1 GB non-pool overhead per CUDA process + cuOpt's RMM
  limit + the Redis cap (2 GB) + 20 GB reserve ≤ MemAvailable at launch; the watchdog stops the run set below 18 GB.
  `MemoryMax` does **not** charge GPU memory on GB10.
- **One GPU arm at a time.** Timed solvers pinned one per X925 core; the partner footprint (GPU + the same 3 A725
  cores + the same GPU memory limit, **12 GB**: the EA's CuPy pool = cuOpt's RMM limit, `EAConfig.gpu_mem_gb`, set by
  cuOpt's measured need of ~10.5 GB, D-003) is identical for cuOpt and the EA. In A9, SCIP runs on X925 5, the EA on
  X925 6 and cuOpt on A725 1–3 (two CUDA processes, 12 GB each); its matched control A3x has the same EA footprint
  without cuOpt (D-015). The exception is A6m (D-020, the MIPfeas calibration: cuOpt as GAMS ran it, 24 threads, MIP
  gaps 0): it has the whole box to itself — the pool's `--layout full`, one lane with all 20 cores, the serving check
  implied, `--cuopt-mem-gb 24 --gpu-mem-gb 36` (81 GB), the file mailbox (Redis and `--cuopt-gpu-peak-gb` refused) —
  and runs nowhere else; no other arm runs on that lane (D-021 R9 keeps it out of v1).
  **ITX-1 (D-021):** every compared GPU arm gets the same share **F** = A9's layout (the GPU lane: X925 5–6, A725 1–3,
  the GPU): **A6F** = cuOpt alone on all of F with |F| = 5 threads, **A2F** = SCIP on X925 5 + cuOpt on the other four
  cores (4 threads), no exchange. A job's **`harness`** (`"ext"`, or a dict overriding its keys) gives every cuOpt
  process MIP gaps 0, no thread variables, its own RMM limit (24 GB, separate from the EA's 12-GB pool) and host
  `MemoryMax` 24 GB, and every process the project's CUDA JIT cache `~/.cache/itrea/cuda`; it enters the run hash only
  when present (every earlier run id is unchanged), and is refused on A6m and on test / screening jobs (a prereg-v2
  item). Gates under ext: A9 97 GB, A2F 78, A6F 69 (+ 2 with Redis); M3's A6F at RMM 40 / host 40: 101.
  **Phase 2 (2026-10-12):** three optional harness switches, absent from the resolved harness — and the hash — unless
  on: `ref_stop` (R12: the runner ends the run when a process's stored incumbent reaches MIPLIB's `=opt=` z* — GAMS's
  gamma <= 0 and Berthold's gap <= 1e-9, obj recomputed from the vector, the earliest such record over every process;
  z* never reaches a process; never supportcase22, PGLib, ML4CO; stop kind `ref_reached`, the trigger checked exactly
  after the run, unconfirmed = void `ref_rejected`, rerun once with the stop off), `cuopt_kill_on_end` (cuOpt
  SIGKILLed at once after a partner's proof or ref_reached) and `scoped_hash` (an arm without cuOpt hashes only the
  harness keys that act on it, so A0's run ids survive a change to cuOpt's settings — R33's "A0 stands if the SCIP
  harness hash is unchanged"). Preset **`itx1`** = ext + all three: the v1 tiers and A0's external jobs. External jobs
  (split `ext`) of an arm with the EA need the tag `prereg-v2` and the configuration registered with it —
  `configs/frozen-v2.yaml` (the C1 freeze) in the tagged commit, unchanged since, every A9 job hashing to it (R33's
  sealing, `pool.check_frozen`; `make-jobs itx-v1` refuses an A9 tier without that file); every external job needs a
  clean tree under `src/` and `configs/`. A job-level `priority` orders block groups (A0's H_cu ∪ W_SCIP unseen
  first). A0 and M6 run on single-core X925 lanes: `--layout screen --cpu-lanes 4` (M6's alone runs: 1; 4 × 9 GB +
  the 20-GB reserve = 56 GB).
  `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` in every process but A6m's cuOpt and cuOpt under a harness
  with `cuopt_env_unset`, whose environment has the three removed (GAMS ran it so; libcuopt links libomp). Never run
  analysis with default thread counts beside timed runs.
- **Thermal.** GPU arms start only below 80 °C (ACPI maximum), CPU-only arms below 86 °C (D-008's 10-06 addendum,
  D-021 R24; 88 until 2026-10-11); pause the run set at
  ≥ 92 °C, resume below 80 °C. A block voided by the pause is rerun first, at most `--max-thermal-requeues` times
  (default 1): a further void skips it (pool status `skipped`) — in screening stage 2 (2026-10-06) the same 8 blocks
  were voided 8 times each before the cap existed. Each run records the hottest ACPI zone seen while it ran
  (`run.json` `acpi_during`) and the 1-min load average before and after. **D-021:** a GPU run also starts only while
  the host fan governor is alive (`~/.local/state/thermal/fan-state` < 30 s old) with its clock cap off
  (`--no-governor` only on a host without it; a governor gone 120 s halts the pool, also during its rest flag);
  `--pause-c` lowers the pause / void line (M2's soak: 90, never above 92); a run whose scope thermal-watch stopped at
  ≥ 98 °C is void (`thermal_watch`, never an arm crash); run scopes are named `itrea-run-<sha1(run)>-<role>-<epoch_ms>`
  and stop within 5 s (cuOpt's by SIGKILL); every run records the fan state at its start and end and its start
  temperature. **R26's breaker:** after 2 thermal or cap voids in a run set (`--max-heat-voids`; one thermal pause
  counts once) no new block opens and the pool ends `halted` (it fired on 2026-10-15 17:30: two `gpu_cap` voids after a
  governor rest, the hottest zone at 85–87 °C within ~4 min; the next window resumed the unfinished blocks, the voided
  one whole). **R26's opening gate** (a 60-s median of the hottest zone ≤ 50 °C after 3 min idle with both servers off)
  is not enforced by the pool (#38): a chain runs `tools/r26-gate.sh --deadline HH:MM` after `serving-window.sh stop`
  and restores if it does not open. Jobs with a harness need both model servers stopped
  (`--allow-serving` refused). The governor's rests stay as they are (Berk, 2026-10-11); journal 2026-10-15 14:4x: four
  30-min rests by 14:57, about every 1.5–2 h under load, about 25 % of window time (an estimate; 30 min every 1.5–2 h
  is 25–33 %). Other sessions' CPU-heavy jobs belong on cores 10–14 at nice
  19 during our windows (agreed with the [other project] session 2026-10-15 19:0x); unpinned ones can land on the run cores.
- **Deadline and lease end at the start of every run.** After its gate waits, a run whose estimate would end after
  `--deadline` or the lease end does not start: a block not yet started counts as never opened and the pool stops; a
  block partly run keeps running (a stop never splits a block).
- **GPU health.** A GPU run with a median SM clock below 2,000 MHz or foreign GPU activity above 2 % of its budget
  voids its whole block (all arms of that instance × seed) and halts the run set. **Cap void (D-021 R25):** a GPU run
  whose busy SM median is below 2,300 MHz (`--cap-floor-mhz`) or that overlaps a fan-governor clock cap (2,200 MHz,
  which passes the 2,000 floor) is void (`gpu_cap`): its block is rerun once, after the fan reads AUTO, then skipped
  and listed; a low clock with the governor's cap off before and after and no cap interval in its log (an unknown
  lock) halts the pool at once. Under a harness cuOpt's CONFIRMED cap hit — an RMM error with the peak ≥ 0.8 × the
  limit, or a kill with the scope's Result oom-kill or maxrss ≥ 0.95 × MemoryMax — is void too (`cuopt_memory`, rerun
  once; M3's harness records it instead); a failure of SCIP or the EA, their caps included, stays an arm outcome; a
  SIGKILL nobody in the harness sent, at no cap, voids any arm's run (`external_kill`, rerun once).
- **Frozen configuration.** Test-set instances run only with the configuration whose hash equals `frozen-e2`'s, with
  `prereg-v1` an ancestor of HEAD. Tuning happens on the tuning set only, within the logged E4 budget.
- **Every incumbent checked.** Every incumbent vector of every process is stored and checked by the independent exact
  checker; only checked incumbents enter a metric. Exit codes of every process are checked; a crash is kept and
  truncated, never silently re-run.
- **Clock rule.** Before the barrier only parse, imports, CUDA context and kernel warm-up on a synthetic problem; any
  LP, presolve or evaluation of the instance's population after `t0`, in every arm. A run that compiles a kernel after
  `t0` (`compiled_on_clock`) voids its block (D-002 v3.1). Under a harness the runner does it itself (D-021 R15): an
  entry of the project's CUDA JIT cache written after `t0` (the `index` file aside) or CuPy cache growth after `t0`
  voids the run (`compiled_on_clock`) and the pool reruns the block once.
- **Deterministic warm-up.** Before `t0` the warm-up runs every crossover and mutation operator on 1-, 2- and
  3-child batches with the run's gene dtype (87bc759, 2026-10-09) and the elite kernels on three copies of one warm-up
  vector, so they do not depend on how many distinct members the warm-up population holds (6ae6a6f, 2026-10-08) —
  operators drawn at random cover only by chance; and (branch `r1-bsync`, 2026-10-14) a concatenate of 9 parts, every
  device array of a candidate set, one-row and mixed (CuPy 14.2 changes kernels above 8 parts). A new kernel, operator
  or batch shape gets its own warm-up entry.
- **Cold-cache GPU test before the timed run.** Before the first timed run of new or changed GPU code, run the
  gpu-marked tests inside the window (lease held), including the cold-cache tests (`test_cupy_cold_cache_*` in
  `tests/test_ea_run.py`, `test_elite.py`, `test_lp_decode.py`, and for R1 / R2 `test_cuda_sched.py`,
  `test_lp_decode_gate.py`: an empty `CUPY_CACHE_DIR`, nothing may compile after
  `t0`). The test must prove the run reached the new code, not only that nothing compiled (incident 2026-10-08: a
  vacuous pass let a block compile on the clock, voided; on 2026-10-09 the test caught the block crossover's one-child
  chunk compiling after `t0` before any timed run).
- **No commits to `src/` or `configs/` while a pool can launch runs** (including a chained pool): every run's processes
  import the current source and the pool reads the config file when it plans a run (incidents 2026-10-06 15:15–16:35:
  one pool's runs spanned several code states; 16:54: an edited config halted a pool). Develop in a git worktree on its
  own branch (`git worktree add -b <branch> ../i-trea-ii-<name>`; `dev-next` and `dev-backoff` were built that way),
  test there on the CPU only on cores outside the run
  lane (`CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 ../i-trea-ii/.venv/bin/python -m pytest -q -m
  "not gpu"`), and merge after the pool has ended.
- **Performance runs first (Berk 2026-10-13).** Box time goes to runs that measure the method (A9 against the solvers,
  development mechanisms where they can act), targeted first and wider only on a pre-stated hint (D-024). Calibration,
  harness and methodology runs only when something structural is broken, or to fill otherwise idle daytime hours.
- **Development changes: one variable, the test fixed first.** Each change gets a decision record whose mechanism
  check and outcome rules (per-instance bands from the control's seed-to-seed spread) are written before any run; the
  control is re-run in the same window; a rule that fails by the letter is recorded as a stated deviation, never
  silently re-read (D-012, D-016–D-018).
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
- **Record everything.** Journal entries (`vault/20-journal/`, append-only, host local time — CEST, CET from 2026-10-25), decision records D-NNN, a result
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

**Updates to Berk** (his rules, 2026-10-06 / 10-08 / 10-11): frequent, short, quantified (effects with intervals, counts,
p — not adjectives). Each update opens with a short **Performance** block — Good news / Bad news about how the approach
itself does (effects vs SCIP alone, cuOpt alone and the controls, weak-set numbers, targets met or missed; "nothing new
measured" if so) — then a separate Good news / Bad news block for implementation, harness, box and bugs. A gap that
comes with a concrete remedy is not bad news; bad news is a measured loss, failure or regression.

## Reproduce
```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev,plots,gpu]"   # see README for cuOpt
pytest -q                     # toy-scale correctness tests (GPU tests skip without CuPy; -m "not integration" is faster)
```
Once runs exist: `itrea build-runs`, `itrea prereg`, then `itrea reproduce` re-derives every table in `datasets/` and
`paper/generated/` from the traces and lists any difference (skill `itrea-reproduce`). Development reports (after
`itrea build-runs`, with `OMP_NUM_THREADS=1`): `itrea backoff-report --stage 0|1`, `itrea mipfeas-report --arm A6F`
(D-021's M4 calibration; `--arm A6m` = D-020's; GAMS's archive in `data/mipfeas/gams-2026-09-16/`),
`tools/d016_report.py`, `tools/d017_report.py`, `itrea d022-report`, `itrea d025-report`, `itrea d026-report`, `itrea
d027-report`, `itrea c1r-report --a9 …` (D-030), `tools/d029_report.py` and `tools/d031_report.py --d030-json F` (no
`itrea` entry; D-029, D-031). D-021 phase 2 (pinned, `OMP_NUM_THREADS=1`): `python
tools/derive_mipfeas_strata.py --runs datasets/runs.parquet [--check]` (the sets; the screening's run table),
`python tools/m0_replay.py --runs … --inc … --root . [--plan]` (M0, under a `MemoryMax` scope), `itrea itx-report
[--cuopt 26.10] [--frozen configs/frozen-v2.yaml] [--bca]` (the v1 analysis; `--m6` for M6), `itrea gams-trc --out DIR`
(GAMS traces of our runs).
Re-running experiments needs the public instances in `data/` and the job files in `jobs/`; wall-clock numbers depend on
the machine (`vault/10-design/Compute resources.md`).
