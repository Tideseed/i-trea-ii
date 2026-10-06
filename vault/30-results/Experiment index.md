# Experiment index (every experiment, including failures)

One row per experiment, in the order of [[D-002 Pre-registration — hypotheses, arms, analysis]] §1 (v3). Each
experiment gets its own note in `30-results/` when it runs (skill `itrea-result-note`): question, command, config
hash, numbers with intervals, checker coverage, caveats, failures. Aborted or broken runs get a ✗ row here and an
entry in [[Mistakes and incidents]]; nothing is deleted.

Outcome key: **+** positive · **−** negative / no effect · **~** inconclusive · **✗** aborted or broken · *planned*.

**State on 2026-10-06 01:4x: no experiment has run.** E0a (build and toy-scale correctness tests) is done; the
screening pass S1 is planned for tonight's window, after the first commit and the tag `prereg-v1` (D-002 §1 step 1:
the pool runner refuses non-smoke jobs on candidates without the tag). Nothing has been timed on a candidate instance.

| id | question | design | gate / hypothesis | tag before | status | outcome | note |
|---|---|---|---|---|---|---|---|
| E0a | Is the code correct? | unit and property tests on toy models (`itrea.benchmarks.toy`) and synthetic problems: operators, feasibility jump (CPU reference, GPU kernel, dense baseline), engine invariants incl. the Rank 0 cap and the one-way arm, solver round trips (SCIP `trySol`, HiGHS user solution, cuOpt RMM limit), file and Redis mailboxes through the runner, gates, analysis; configs (`tests/test_configs.py`) | all pass | — | **done** 2026-10-06 01:2x (build) | **+** 413 passed, 2 skipped (full suite, 01:2x); 2026-10-06 01:4x, without the integration test and the two cuOpt runs above 4 GB: 432 passed, **2 failed** — `tests/test_pair.py` still expects the old 4 GB GPU limit (default now 12 GB, D-003); to fix before `prereg-v1` | journal 2026-10-05 (00:0x–01:2x); gaps: no dedicated test files for `ea/ranks.py` (the y counterfactual), `ea/population.py`, `ea/evaluate.py`, `ea/frontier.py`, `exchange/*`, `trace.py`, `problem.py` — covered only through engine, EA-process and runner tests |
| E0b | Calibration | SCIP nodes/s and LP iterations/s on 6 tuning instances under {idle, + GPU EA, + cuOpt, + 1 SCIP lane} × 2 reps; exchange transport file vs Redis {none, file idle, Redis idle, Redis at 1 Hz}; SCIP callback windows | ≤ 3 % slowdown → CPU arms concurrent; > 2 % Redis cost → file mailbox (D-006) | `prereg-v1` | planned | — | already measured, outside E0b: cuOpt's GPU memory (RMM peak 10.47 GB on a PGLib-UC ferc case → partner GPU limit 12 GB, D-003); the inert SCIP plugin's cost on a toy (+0.5 %, D-001) |
| S1 | Which candidates are hard for SCIP? | screening stage 1: SCIP alone, 300 s, seed 900, 324 candidates, X925-pinned, 9 GiB (`itrea make-jobs screen1`, then `itrea pool`) | — | `prereg-v1` | planned **tonight** (window W0, lease until 07:30), after the commit and the tag | — | [[Hard instance pool]] |
| S2 | Same, second seed | stage 2: seed 901 on stage-1 P(300) ≥ 0.05 or no solution | strata and quotas → `pool-v1`; references → `datasets/reference.csv` | `prereg-v1` | planned | — | [[Hard instance pool]] |
| E1 | Can the GPU EA find feasible solutions at all, against cuOpt? | tuning set: A5 (EA alone) vs A6 (cuOpt alone), 300 s; GPU vs single-core CPU feasibility-jump throughput (moves/s, time to first feasible) | (a) A5 first feasible within 2× of A6 on ≥ half of the tuning instances where A6 finds one — else stop, report the engineering gap, no E2; (b) LS throughput reported | `pool-v1` | planned | — | — |
| E4 | Which configuration? | tuning set, ≤ 12 configurations × 16 instances × 120 s × 1 seed, every configuration logged; winner by mean d vs A5-default | → `configs/frozen-e2.yaml` | `pool-v1` | planned | — | — |
| E2 | **Main claim:** does SCIP ⇄ GPU EA beat either alone? | test set, 32 instances × seeds 1000/1001, 300 s, Williams arm order per block: A0, A0*, A1, A2, A3, A5, A5x0; A3r on 8 instances × 2 seeds; A6 one seed | H1a A3 < A0 **and** H1b A3 < A5 (intersection–union, α = 0.05 each); Holm family H2 (uncoupled portfolio), H3 (A2), H4 (A1), H5 (A0*); H6 A5 < A5x0 | `frozen-e2` | planned | — | — |
| E3 | Which parts of the coupling matter? (exploratory) | test set, A3 ablations (`configs/ablations/*.yaml`, or `--set` on `frozen-e2`): exchange direction both ways, bound demotion, frontier posting, injection band (`exchange.inject_band`); A4 (CPU EA, the GPU's share); A5y0; HiGHS as partner (H0/H2/H3, only after the HiGHS user-solution test passes) | exploratory | `frozen-e2` | planned | — | — |
| E5 | Longer budgets and the open instance (exploratory) | 1,800 s on a test subset; `supportcase22`; per-stratum reports (the strata are the subgroups, named now) | exploratory | `frozen-e2` | planned | — | — |
| E6 | Several objectives (not scheduled) | part I's Rank 0/1/2 definitions with objective vectors on a public multi-objective MIP library, to be chosen and verified first ([[D-005 The frontier of diverse solutions]]) | — | — | not scheduled | — | — |
