# I-TREA II — the algorithm (design spec, 2026-10-05)

Implementation: `src/itrea/`. Every number below is a parameter in `itrea.config.EAConfig` (YAML in `configs/`);
the defaults are starting points to be tuned on the tuning set only ([[D-002 Pre-registration — hypotheses, arms, analysis]]).

## Problem form
`min  c·x   s.t.  rl ≤ A x ≤ ru,  lb ≤ x ≤ ub,  x_j ∈ ℤ for j ∈ I` — maximisation is converted by `sense`; the
population compares **keys** (`key = sense · objective`, lower is better). Read from MPS with HiGHS
(`itrea.problem.Problem`), canonical variable order = HiGHS's column order. Infinite bounds are `±inf`.

Row violation of individual x, row i: `v_i = max(rl_i − a_i, 0) + max(a_i − ru_i, 0)` with `a = A x`, normalised by
the row's 2-norm. **Total violation** `V(x) = Σ_i v_i/‖A_i‖ + Σ_j bound violations + Σ_{j∈I} |x_j − round(x_j)|`.
**Feasible** = every row and bound within `feas_tol` relative to `max(1, |rhs|)` and every integer within `int_tol`
(SCIP-style per-row test — never a sum against a total tolerance, the X2 bug).

## Population and the three ranks (I-TREA's names; mapping in [[D-004 Mapping I-TREA (2015) to I-TREA II]])
Population X (P × n) on the device, plus per-individual `key`, `viol`, `feasible`, `rank ∈ {0,1,2}` (−1 = not yet
ranked), `id` (age: lower = older), `parent ids`, `origin` (operator bits), `birth`, ancestry bit masks. Survival each
generation, from the candidate set C = population ∪ offspring ∪ injected (`itrea.ea.ranks.three_rank_survival`), in
part I's elimination order:

1. **Duplicates** (identical vectors) out.
2. **Incumbent key** `inc = min(best feasible key in C, solver's incumbent key known to the EA)`.
3. **Rank 1 — feasible good (protected):** the best `ceil(y · basis)` feasible individuals by key (`elite_basis`:
   `feasible` = #feasible, default; or `population` = P); with `rank1_twins: drop`, equal-key twins are not Rank 1
   (they compete in Rank 2). At least one when any feasible exists and y > 0.
4. **Rank 0 — infeasible good:** infeasible with `key < inc − imp_tol·max(1,|inc|)`; with `rank0_bound_demote`,
   those with key beyond the solver's **proven** bound are eliminated (the solver's answer replaces part I's
   decision maker). At most `floor(x · P)`; over the quota, `rank0_select`: `itrea` (non-dominated fronts in (key, V);
   the overflowing front thinned by repeatedly removing the point nearest to another in normalised (key, V), tie →
   worse key out), `violation`, `stochastic_rank` (the optopt X2 rule).
   **No incumbent yet** (`inc = +inf`): `no_incumbent_policy: violation` (default — feasibility phase, everyone ranked
   by (V, key) as Rank 2; a deviation from part I) or `itrea` (everyone is infeasible good; Rank 0 selection, no quota).
5. **Rank 2 — rejects:** the rest fill to P, `rank2_select`: `itrea` (non-dominated sorting in (key, V), crowding
   distance on the overflowing front, ties → older out) or `key_then_violation`.

Invariants (property-tested): |survivors| ≤ P; Rank 1 ⊆ feasible and holds the best feasible keys (up to twins);
|Rank 0| ≤ floor(x·P) when an incumbent exists; every Rank 0 member is infeasible with key < inc; a Rank 1 member of
generation g is in generation g+1 unless displaced by a better feasible individual (or a twin rule).

**Solver rejections:** an EA solution that the solver refuses (`trySol` false although our float64 check passed) is
marked infeasible in the population (it moves to Rank 2) and counted — part I's "deemed inferior by the decision maker".

## Generation
1. **Exchange in** (every `exchange_poll_s`): the solver's incumbent, its pool ("very good" = within `inject_band`
   relative gap of its best, at most `inject_k`), its dual bound and (once) its root LP solution. Injected vectors are
   evaluated and enter C (as part I's population takes in new members, they are ranked by the same rule).
2. **Parent selection:** tournament of size `tournament_k` with part I's comparator (`itrea`: Rank 1 < Rank 0 <
   Rank 2, ties by key then V), or `deb` / `stochastic` (stochastic ranking, `p_f`). Optional per-rank parent pools
   (`parent_rank_weights` = [Rank 0, Rank 1, Rank 2]; default one pool, as in part I where every member can enter the
   mating pool). Matings across ranks (Rank 0 × Rank 1) are recorded in the ancestry bits.
3. **Crossover** with probability `pc`, operator chosen by `crossover_weights`: `uniform(p_swap)`, `k_point(k)` (on the
   variable order), `block` (variables of `block_rows` random constraints taken from the second parent — constraint-
   aware), `none`.
4. **Mutation**, operator by `mutation_weights`: `flip` (binaries), `step` (±1…±`step_max` for general integers),
   `reset` (uniform in bounded domains), `toward_lp` (rounded LP-relaxation value, if known), `violation_guided`
   (variables in violated rows — part I's Second Chance analogue), `objective_guided` (moves that improve the
   objective, to be repaired — part I's Forced Closure analogue). Expected `mutation_rate · |I|` genes (≥ 1) per child.
5. **Repair / local search** (`repair: none | greedy | fj`), `ls_steps` steps, objective weight `ls_obj_weight`:
   batched over the population on the device. One step: compute activities `A Xᵀ` (SpMM), for every non-zero (i, j) the
   change of row i's weighted violation if x_j moves by its candidate step (binary flip; ±1 for general integers),
   reduce by column (sparse indicator SpMM) → score per (individual, variable); apply each individual's best improving
   move (`ls_moves_per_step` ≥ 1). `fj` adds feasibility-jump style row weights (Luteberget & Sartor 2023) that grow on
   rows violated at a local minimum. Work per step O(nnz · P), chunked over the population to stay inside the GPU
   memory budget (`gpu_mem_gb`).
6. **Continuous variables** (`continuous: gene | lp_decode`): `gene` treats them as real genes (moves to make a
   violated row tight); `lp_decode` fixes the integers and solves the LP on the CPU (HiGHS) for at most
   `lp_decode_per_gen` candidates per generation (those that would enter Rank 1 or be posted).
7. **Evaluate** offspring on the device (`eval_dtype`: float64 default; float32 measured in E1 for speed and its
   false-feasible rate), **survive** (three ranks), record lineage.
8. **Exchange out:** when the best feasible key beats the solver's incumbent by `post_min_gain`, re-check it on the CPU
   in float64 (per row) and post it. The solver adds it as a new incumbent / starting point (SCIP: heuristic plugin,
   `trySol`; HiGHS: user-solution callback). Every post and the solver's verdict are logged.
9. **Stagnation (trigger):** after `stagnation.gens` without a new best, replace `immigrant_frac` of Rank 2 with
   immigrants (random or LP-rounded); Rank 0 and Rank 1 are untouched (part I called itself "overly elitist").

## Seeding
`seed_mix`: random in bounds (integer-rounded), LP-relaxation rounding (HiGHS root LP on the CPU, before the clock,
counted in the start-up record), and the solver's incumbents as they arrive.

## What is logged
Per generation (every `log_every`): best/median key, feasible share, |Rank 0|, |Rank 1|, |Rank 2|, inc,
offspring/s, GPU memory peak, Rank 0 candidates and demotions. Per event: every new population best with its lineage; every post and verdict; every injection received.
Trace = JSON lines + a summary JSON (schema in [[Trace format and metrics]]).
