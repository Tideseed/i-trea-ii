# Research question and spec (Berk's brief, 2026-10-05 23:00 CEST, condensed)

> "Let's start a new act on optimization research. This time I want to run evolutionary algorithms on gpu in parallel
> with a solver. Evolutionary algorithms shall have a maximum x% (say 10) of the solutions reserved for better than the
> best feasible but infeasible solutions. Best y% of feasible solutions shall be protected in all iterations.
> Evolutionary algorithm and solver shall communicate and best and very good feasible solutions shall be injected to
> population and if evolutionary algorithm catches a better incumbent feasible solution it shall add it as a starting
> point. You may use highs, scip or cuopt for the solver counterpart. Evolutionary algorithm shall have the mutation,
> mixing (crossover), population and everything as parametric. Let's experiment on the hardest problems. […] private
> repo Tideseed/i-trea-ii (Interactive Three Ranks Evolutionary Algorithm Part II) as a successor of my masters
> research. I will add details later."

## Main claim (Berk, 2026-10-06 00:0x — [[D-007 Main claim — the coupling beats either method alone]])
> "main idea in i-trea ii is that a gpu based evolutionary algorithm metaheuristic coupled with an exact solver can beat
> either single solution." — and "take the i-trea as only the starting point and the idea of using good infeasible
> solutions."

**Primary question.** On hard MIP instances, does the coupled system — a GPU evolutionary algorithm and an exact solver
exchanging solutions both ways — find good feasible solutions sooner (lower primal integral) than **the solver alone and
the EA alone**? *(Refined on 2026-10-07 and given a v1 target on 2026-10-10 — see the dated note at the end.)*

Three secondary questions make the answer informative: does the coupling beat the same two methods running side by
side **without** exchange (is the interaction itself worth something, beyond running two methods)? Is our GPU EA a
better partner than NVIDIA cuOpt on the same GPU (optopt's strongest result was that race: HiGHS‖cuOpt +30 %,
SCIP‖cuOpt +21–25 % on P(60))? And do good infeasible solutions (Rank 0) help the EA?

## Where this starts: the predecessor's null result (optopt X2, 2026-09-24 night)
The idea was already measured once on the CPU (`berkorbay/optopt`, experiment X2, `src/optopt/solvers/ea.py`):
a Python EA on one CPU core beside SCIP, uniform crossover + ±1 mutation + greedy repair + HiGHS LP decoding of the
continuous part, ≤ 20 % super-optimal infeasible individuals ranked by infeasibility, 80 MIPLIB instances, 60 s.

> **Verdict (optopt journal, 2026-09-24 03:24):** EA feasible-only −0.1 % vs SCIP alone (n.s.); with ≤ 20 % super-optimal
> infeasible 0.0 % (n.s.); both EA arms lose ~1 % to the equal-compute control SCIP‖SCIP (p < 0.03). The EA posted 129
> improvements of its own population; only 11 beat SCIP's incumbent at the time. "A Python EA on one slow core does not
> keep up with SCIP's own heuristics at 60 s; Berk's infeasible share makes no measurable difference here. **What could
> change it: a GPU population (cuOpt already is one — and the cuOpt→SCIP arm is the one that helps), longer budgets,
> instance families where SCIP's heuristics fail.**"

I-TREA II is the test of exactly those three changes, one at a time where possible:
1. **GPU population** — the EA's evaluation, local search and survival run on the GPU (CuPy); the same code runs on
   the CPU (NumPy) as a control, so the GPU's share of any effect is measurable.
2. **Longer budgets** — 300 s and 1,800 s (60 s only as a sanity anchor).
3. **Hard instances** — instances where the solvers struggle to find or improve incumbents
   ([[Hard instance pool]]), not a representative MIPLIB sample.

Lessons X2 paid for and that are encoded in the design (see [[I-TREA II algorithm]]):
- Inject into SCIP only through a **heuristic plugin** (the event-handler route crashed SCIP on 9/80 instances).
- cuOpt's `SetSolutionCallback` **disables cuOpt's presolve** — cuOpt is a control arm and a launch-time warm start,
  never a mid-solve two-way partner.
- Check every candidate **per row** in float64 before posting (X2's first run summed violations and posted 26/176
  "better than optimal" solutions). Rejected posts are a first-class metric.
- Start-up (instance parse, CUDA context, kernel compile) happens before the shared clock and is recorded.

## The three ranks — from I-TREA (2015) to I-TREA II
Berk supplied the part-I manuscript on 2026-10-05 (the 2011 thesis (https://open.metu.edu.tr/handle/11511/20509)).
**Part I is the starting point, not a template** (Berk, 2026-10-06): the principle that carries over is keeping good
infeasible solutions; the rank names stay; every other choice is an ordinary parameter tuned like any other. Where
the defaults came from: [[D-004 Mapping I-TREA (2015) to I-TREA II]].

| rank (I-TREA name) | who | size | rule |
|---|---|---|---|
| **Rank 0 — infeasible good** | infeasible individuals whose objective beats the best feasible one known (the population's or the solver's incumbent) | at most ⌊x · P⌋ (x = 10 %), reserved up to that | over the quota: Pareto fronts in (objective, violation), then nearest-distance thinning (part I's two Rank 0 eliminations); eliminated if it beats the solver's **proven** bound |
| **Rank 1 — feasible good** | the best feasible individuals | ⌈y · #feasible⌉ (y = 10 %) | protected every generation; equal-objective twins leave Rank 1 |
| **Rank 2 — rejects** | everything else | the rest of P | non-dominated sorting in (objective, violation) with crowding distance (part I's Rank 2 eliminations) |

Tournament order as in part I: Rank 1 < Rank 0 < Rank 2. "Interactive": part I asked a decision maker pairwise
questions; in part II the **solver** is the counterpart — it supplies incumbents, very good solutions and its proven
bound, and receives the EA's better incumbents as new starting points.

## Out of scope (for now)
Learned control (Laya, LLM agents) of the EA — optopt measured those; revisit only once the EA itself adds value.
Proving optimality — the EA works on the primal side only; dual-side effects come from the solver.

## Berk's aims, restated (2026-10-06 evening)
"We are head to head with cuOpt which is developed by some of the best minds out there. Our aim is to make hardest
problems easier so time to first solution is not primary objective; we don't expect to solve simple problems a few
seconds faster at all. Once we practice enough we can go for the larger set of seeds and problems." Earlier the same
day: "being better in a subset of problems is still a strong win." → the primal integral on hard instances is the
measure; a split against cuOpt is a strong position; scale after practice. See [[Roadmap]].

## What D-013 and D-019 changed (note added 2026-10-10; Berk's words above are unchanged)
The text above is the brief of 2026-10-05/06. Where later decisions changed it:
- **Aim refined** (Berk, 2026-10-07, [[D-013 Aim — lift each solver's weakest cases]]): "I don't aim our coupling to
  be generally better but to help both SCIP, cuOpt (and Highs if possible) in some of the hardest problems where they
  are the weakest." The primary question is now the improvement on each solver's **weak set** W_B — the instances where
  solver B alone has P ≥ 0.5, chosen on seeds never used for the evaluation — for B = SCIP, cuOpt and, if possible,
  HiGHS; the mean over all instances is a guard (mean d ≥ −0.05 against B alone). It amends D-007 and enters prereg-v2
  before any confirmatory run.
- **The EA alone** is a diagnostic in development, not a target
  ([[D-011 Solver streams good infeasible solutions to the EA]], Berk 2026-10-06: "You are benchmarking EA but our EA
  shall be coupled with a solver similar to what cuopt does"). D-002 v3.1's H1b (coupled < EA alone) stays
  pre-registered until prereg-v2 is written; v1's confirmatory runs (D-019) are {SCIP alone, cuOpt alone, A9} and
  contain no EA-alone arm.
- **cuOpt is also a source, not only a rival** ([[D-015 Use cuOpt's feasible solutions — three-way arm A9]], Berk
  2026-10-07: "We can use cuopt's feasible solutions"). In arm A9 (SCIP ↔ EA ← cuOpt) cuOpt streams its incumbents into
  the EA, which passes to SCIP whatever beats SCIP's incumbent — cuOpt's vectors unchanged as well as the EA's own
  improvements. The X2 lesson above still holds: cuOpt never receives solutions
  mid-solve. Why: out of sample the coupling without cuOpt was far weaker than on the development set (class probe,
  [[D-014 Class probe — predictions fixed before the runs]]: against cuOpt alone 1 better / 5 worse, mean d −0.51
  [−0.84, −0.12]).
- **Budgets:** 120 s in development (D-010), 300 s in the confirmatory test, 600 s under the MIPfeas protocol (v1's
  external run); 1,800 s stays exploratory (E5).
- **v1** (Berk, 2026-10-10, [[D-019 v1 — deadline and performance target]]): "So let's put a target (both as deadline
  and performance) to our little project for v1"; then "we shall have a checkpoint in the next 2 weeks and end in
  november with the new cuopt version" and "No nights". v1 = A9 frozen on one GB10. Checkpoint C1 on Fri 2026-10-23
  (development closes on cuOpt 26.08); v1 at the end of November 2026 (aim Fri 11-27, hard end Mon 11-30) on cuOpt
  26.10, or on 26.08 if 26.10 is not released by Fri 11-06. Targets: on about 32 unseen hard instances × 2 seeds at
  300 s — SCIP's weak set at minimum a significant gain, target mean P −40 % with ≥ 70 % of the weak set improved
  (stretch −55 %); cuOpt's weak set at minimum a significant gain, target mean P −25 % with ≥ 60 % improved (stretch
  −40 %); mean d ≥ −0.05 against each solver alone. Under Mittelmann's MIPfeas protocol (233 feasible MIPLIB
  2017 benchmark instances, 600 s) — at minimum A9's geometric mean no worse than our own cuOpt-26.10-alone run on the
  same GB10; target −25 % mean integral on the instances hard for cuOpt and −10 % geometric mean; stretch 0.0274 (SCIP
  concurrent 11, published). The 233 overlap the development instances; that result is reported with and without them.
