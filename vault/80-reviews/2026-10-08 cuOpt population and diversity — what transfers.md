# cuOpt's population and diversity management — what transfers to I-TREA II (2026-10-08)

**Why.** Berk (2026-10-08): "I heard there are good ideas at cuopt routing engine about diversity and population
management." Read from the source, NVIDIA/cuopt `main` at `dfbbcc5a` (2026-10-08, Apache-2.0): the routing engine
(`cpp/src/routing/diversity/{population.hpp, diverse_solver.hpp, diversity_config.hpp}`) and the MIP heuristics'
port of it (`cpp/src/mip_heuristics/diversity/{population.cu, diversity_manager.cu, multi_armed_bandit.cu,
recombiners/}`). The MIP population class is the routing one nearly line for line (same team), with MIP-specific
distance, weights and recombiners — so the MIP version is the one that applies to us; the routing version shows the
fuller design (two populations, a diversity schedule). Nothing here is measured on our side yet; this is a reading
plus our own run data.

## Where we stand — our run data (development + A9 pilot, 120 s, seeds 0–1; read from `ea.summary.json` / `gen` / `frontier` events)
- **We keep about one good feasible basin.** In the repair-clock runs (A3@rc1) Rank 1 ends with **1–9 members** on
  the 6 development instances where the EA is feasible other than item_placement (26); fhnw and rail01 have none.
  n1 = ⌈y·#feasible⌉ with a feasible share of 0–13 % at the end of the run (comp21 in the class probe: 64 %).
  Generational Rank 2 means a feasible solution outside Rank 1 dies after one generation unless it is re-created.
- **The frontier archive (D-005) confirms it:** 1–9 members at T (A9's item_placement 12), and the farthest member from
  the incumbent differs in **0–5 % of the integer variables** (sing44 0, radiationm 0.12 %, ca 0.19 %, item_placement
  0.38 %, neos-960392 0.85 %, glass4 5.3 %), with two exceptions (loue 14 %, kumeu in A9 13 %). In A9, radiationm's
  farthest member is at **14 %** against 0.08 % in its matched control A3x: cuOpt's solutions are the only distinct
  basin we hold there.
- **Immigrants already supply volume, not quality:** the survival leaves a shortfall of up to 116 of 256 slots per
  generation (rank-0 over-quota eliminations + generational Rank 2), filled with random/LP immigrants repaired from
  scratch (item_placement: ~100,000 immigrants admitted per run). Stagnation events are 0 on most runs, so
  `frontier.reinject` (a share of *stagnation* immigrants only, `engine._stagnation`) never fires: the archive is an
  output, not a source.
- **On the hardest instances there are few generations to manage:** fhnw 31–38, glass4 19–28, ca 20–22, sing44 83–94,
  chromaticindex 27–29 in 120 s (A9: 1–3 on fhnw, krka, chromaticindex). Repair dominates the generation (ca: about
  5.7 s per generation; the phase timer at 120 s shows 7.2 s in repair against 0.2 s in crossover and mutation).
  Survival rules have little room to act there.
- Exact duplicates are rare (0–3 % of children; comp21 10 %); near-duplicates are not measured.

## What cuOpt does (with the source line that does it)
1. **Forced distancing ("clearing radius") in solution space** — `population::add_solution`: every two members must be
   less similar than a threshold. A new solution finds the best member similar to it; if it is better, it evicts
   **every** similar worse member, otherwise it is rejected. MIP similarity = number of integer variables with equal
   values (`solution_t::calculate_similarity_radius`, continuous values compared within min((ub−lb)/10, 0.25)).
   Population size 32 (`heuristics_hyper_params.hpp`), sorted by quality; **slot 0 always holds the best feasible**,
   outside the distancing. Ours measures diversity only in (key, V) space (crowding, nearest thinning): two different
   vectors with equal (key, V) count as crowded, two near-copies with slightly different keys count as diverse.
2. **The radius: a fixed minimum in the MIP solver, a schedule in routing.** The MIP population runs with a fixed rule
   — two members must differ in at least ~10 integer variables (`get_max_var_threshold`: n_int − 10 equal integers
   for n_int ≥ 1000, fewer for small problems). Its population.cu also defines a measured-then-relaxed radius
   (`find_diversity`, `adjust_threshold`) and `halve_the_population`, but **nothing calls them at `dfbbcc5a`** (code
   search: no callers outside population.cu/.cuh). The routing engine does run a schedule: the reserve's threshold
   starts from the islands' measured spread and moves quadratically towards "only near-identical are similar" (0.99) by
   the time limit (`adjust_reserve_threshold`), so the search is diverse early and elitist late.
3. **Two populations (routing).** A long-lived *reserve* (fixed final weights, radius as in 2) and a short-lived
   *working* population rebuilt each cycle from 2–5 reserve members (random picks early, best only after 70 % of the
   time), improved until stagnation, then its members go back to the reserve and it is cleared. Within a cycle the
   radius steps down level by level (`improve_population`): stagnation at a level = k recombinations without an
   insertion in the top 3–4 positions → next level.
4. **Partner solutions are recombined with everyone at once** — `recombine_and_ls_with_all`: each solution from B&B
   or a user callback is added, local-searched, then crossed (every enabled recombiner) with **every feasible member**
   of the population; the offspring and their LP-polished versions are added. In the MIP solver this runs inside the
   `diversity_step`, which the feasibility-pump loop calls on stagnation (`local_search.cu`,
   `reset_alpha_and_run_recombiners`): it first drains the queue of external solutions this way, then recombines
   tournament pairs until 8 tries (`max_iterations_without_improvement`) give no insertion in the top 3.
5. **Caution from cuOpt's own code** — `population.cu`, `get_external_solutions`: CPU-FJ solutions are dropped unless
   better than the best feasible: *"It seems they worsen results on some instances despite the potential for improved
   diversity."* Diversity from a cheap source can cost more than it brings.
6. **Recombination = keep the agreement, re-solve the rest** (bound-propagation, feasibility-pump, line-segment and
   sub-MIP recombiners): common integers fixed, the disagreement set searched; the number of variables taken from the
   other parent is adaptive (200 initially, 20–10,000, ×1.1 / ×0.99); if fewer than 40 % of the free variables carry
   objective weight, objective variables are added to the subproblem. Then **an LP with the integers fixed** re-optimises
   the continuous variables (1 s if infeasible, 2 s if feasible) and both versions enter the population. Ours: the
   `agreement` crossover + FJ repair on the touched set is the same principle; our `repair.continuous: lp_decode` is
   configured but **not implemented** (`engine.py` raises NotImplementedError).
7. **Adaptive per-constraint penalty weights** (quality = obj + Σ wᵢ·excessᵢ): ×2 on rows the best member violates
   while it is infeasible (×5 overall in the first two rounds), ×0.9 when it is feasible, normalised, and floored at
   max_j c_j / a_ij so a unit of violation never buys more objective than it costs; the best infeasible is kept 10 %
   worse than the best feasible (`adjust_weights_according_to_best_feasible`). When the feasibility pump first finds a
   feasible point, the population's row weights are **copied from the feasibility jump's learned weights**
   (`local_search.cu`, `reset_alpha_and_save_solution`). Ours: Rank 0 by Pareto in (key, V), V a fixed normalised norm;
   our FJ kernel learns row weights too, but they stay inside the repair.
8. **Bandit over recombiners and local searches** (UCB, recency-weighted Q with α = 0.05 / 0.03; reward 8 for a new
   best feasible, 1 for beating the better parent while within 20 % of the best, scaled by the recombiner's run time).
   Ours: the repair clock (depth) and the strategy mix are the same kind of controller; the difference worth copying
   is the time-normalised reward.
9. Routing-only: EAX / SREX / OX / inversion recombiners, route minimisation (guided ejection search), islands only in
   the initial phase (≤ 5 islands of 3, improved one after another, best of each to the reserve). Not transferable as
   such; the island schedule is a data point for #16.

## Ranked for our measured gaps — each a one-variable development test (D-010; reported in D-013 terms)
1. **A genotype-diverse elite reserve (1 + 3, issue #25).** Rank 1 becomes a reserve of r feasible solutions
   (r = 32, cuOpt's MIP size), any two differing in at least k integer variables, clearing as in `add_solution` (a
   better newcomer evicts every similar worse member, a worse one is refused), the best feasible always kept; reserve
   members are protected and enter tournaments at Rank 1's order; shortfall immigrants draw a share from the reserve
   (perturbed by mutation). First variant = the MIP solver's live rule (fixed k from n_int, `get_max_var_threshold`);
   the routing schedule (k measured from the first feasible solutions, relaxed by T) is a second, separate variant.
   This is Berk's "keep a percentage of good feasible solutions as well as the best … aim for more diversity" made
   literal: today we keep ~1. **Test:** A3@rc1 + reserve vs A3@rc1, 9 development instances × seeds 0–1, interleaved.
   **Mechanism (pre-stated):** median pairwise distance in the reserve ≥ 5× the control's frontier at T on the
   instances that reach feasibility. **Outcome:** D-013 weak-set improvement over the control; guard mean d ≥ −0.05;
   per-instance bands from the control's measured seed spread (D-012 addendum 2), set before the run. Expected not to
   matter on the few-generation instances (fhnw, glass4, ca).
2. **Recombine each new partner incumbent with the whole elite (4).** When SCIP's or cuOpt's incumbent arrives, the
   next generation spends its first pairs on (incumbent × each Rank 1 / reserve / Rank 0 member), the rest as usual.
   Serves A9 directly (cuOpt's solutions are our only distinct basin on radiationm). **Test:** A9 + burst vs A9 on the
   7 A9-pilot instances, seeds 0–1. **Mechanism:** share of EA incumbents whose genealogy includes a partner row within
   10 generations of its arrival. Depends on the GPU back-off for A9 being settled first (contention is A9's open
   problem).
3. **LP with integers fixed (6)** — implement `lp_decode` on the CPU LP worker for the few best children per generation
   (the knob exists: `lp_decode_per_gen: 4`). For instances with many non-singleton continuous variables, where FJ moves
   continuous values by jump values. Size the candidate set before building.
4. Lower: the FJ's learned row weights in Rank 0's V, as cuOpt copies them into its population weights (7) — this
   changes which infeasible points count as "good", so it touches Berk's infeasible-injection rule (D-011) and needs its
   own decision; time-normalised reward in the strategy/clock controllers (8);
   population size 32–64 vs 256 as a one-variable test on the few-generation instances (cuOpt runs 32; whether a
   smaller GPU batch buys generations is unmeasured — a repair call waits for its slowest lane).

**Not proposed:** blind diversity injection (5 says it can hurt), the routing recombiners (9).

## Sources
NVIDIA/cuopt `dfbbcc5a7693968144531f5d6a3b7fe38102ebd5`, files named inline (read 2026-10-08 via the GitHub API into
`/tmp/cuopt-src`, not kept). The routing design follows the hybrid genetic search tradition (forced distancing,
penalty adaptation, reserve/working populations); the MIP diversity manager's comments credit no paper.
