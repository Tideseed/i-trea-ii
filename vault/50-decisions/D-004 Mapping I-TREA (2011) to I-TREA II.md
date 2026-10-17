# D-004 Mapping I-TREA (2011) to I-TREA II (2026-10-05)

**Source for part I:** Berk Orbay's open M.S. thesis (METU, 2011; `orbay2011itrea`, hdl:11511/20509,
https://open.metu.edu.tr/handle/11511/20509): ranks §5.3, crossover §5.6.2, mutation with Forced Closure and Second
Chance §5.6.3, the elimination order §5.6.5, the interactive process §5.6.6, termination §5.6.7, "overly elitist" §7.1.

**Status:** record of where the defaults came from. **Narrowed by [[D-007 Main claim — the coupling beats either method
alone]] (Berk, 2026-10-06):** part I is the starting point only; the principle that carries over is keeping good
infeasible solutions (Rank 0). Faithfulness to the rest is not a goal — every row below is an ordinary parameter, tuned
on the tuning set; the "part-I-faithful configuration" is no longer a planned comparison.

| I-TREA (2011 thesis) | I-TREA II (2026) | parameter | faithful? |
|---|---|---|---|
| two objectives (coverage, #facilities) | one objective, the MIP's; "dominated by a feasible solution" ⇔ key ≥ best feasible key | — | adapted |
| Rank 0 infeasible good: not dominated by any feasible solution | infeasible, key < incumbent key (population's or the solver's) | — | yes |
| Rank 0 quota "reserved", e.g. 10 % (exempt from elimination below it) | reserved **and capped**: at most ⌊x·P⌋ (Berk's 2026 brief: "maximum x %") | `ranks.x` = 0.10 | brief overrides |
| Rank 1 feasible good: non-dominated feasible | the best ⌈y·#feasible⌉ feasible, protected every generation (Berk's 2026 brief) | `ranks.y` = 0.10, `elite_basis` | brief overrides |
| Rank 2 rejects | everything else that is not "infeasible good" (feasible non-elite, infeasible with key ≥ incumbent); since the critique, over-quota and bound-demoted infeasible-good candidates are eliminated, and Rank 2 is generational by default | `rank2_replacement` | adapted |
| tournament: Rank 1 < Rank 0 < Rank 2, ties by estimated utility | same order, ties by key then violation | `selection.comparator: itrea` | yes |
| duplicates eliminated first | same (identical vectors) | — | yes |
| Rank 0: drop the dominated in (objectives, violation), then thin by minimal radial distance (tie → lower utility out) | Pareto fronts in (key, V), the overflowing front thinned by nearest distance in normalised (key, V), tie → worse key out | `rank0_select: itrea` | yes |
| Rank 1 twins (equal objective values) eliminated | equal-key twins leave Rank 1 (stay in the pool as Rank 2) | `rank1_twins: drop` | adapted (moved, not deleted) |
| Rank 2: dominated in (objectives, violation) out; ties → higher violation, then older out; then NSGA-II crowding | non-dominated sorting in (key, V), crowding on the overflowing front, ties → older out | `rank2_select: itrea` | yes |
| decision maker prefers incumbent over a Rank 0 partition ideal → it and all it dominates go to Rank 2 | the **solver's proven bound** answers instead: a Rank 0 point with key beyond the bound can never be feasible → eliminated. A solution the solver **rejects** (`trySol` false) is marked infeasible | `rank0_bound_demote: true` | analogue |
| incumbent = DM's most preferred feasible | incumbent = best feasible key (population ∪ solver) | — | adapted |
| single-point crossover, population doubles | crossover mix (uniform / block / k-point); `k_point` with k = 1 is the faithful operator; offspring = P | `crossover.*` | parametric (faithful config available) |
| problem-specific mutation: revoke + reassign by potentials | generic: flip, step, reset, toward-LP, violation-guided, **objective-guided**, then batched local-search repair | `mutation.*`, `repair.*` | adapted |
| Forced Closure (triggered; improves the facility objective) | `objective_guided` mutation (moves that improve the objective, repaired afterwards) | `mutation.weights.objective_guided` | analogue |
| Second Chance (triggered; revoke assignments below their potential) | `violation_guided` mutation + local search (genes in violated rows) | `mutation.weights.violation_guided` | analogue |
| triggers (interaction, Forced Closure, Second Chance, termination) | triggers: exchange (on new solver message), stagnation immigrants, posting on a new best | `exchange.*`, `stagnation.*` | analogue |
| no feasible-search phase (mutation is capacity-aware) | **deviation:** before any feasible solution, rank by violation (feasibility phase); the faithful alternative treats all as Rank 0 | `no_incumbent_policy: violation` | deviation |
| "overly elitist" — diversification left for future work | stagnation immigrants replace part of Rank 2 | `stagnation.*` | future work of part I |
| termination by population progress | wall-clock budget shared with the solver | — | adapted |

## Configurations
- `configs/itrea2-default.yaml` — the pre-registered main arm (the defaults above).
- `configs/itrea1-faithful.yaml` — single-point crossover only, `no_incumbent_policy: itrea`, plus-selection, no
  immigrants: kept as a record; not a planned comparison since D-007.
- Ablations: `ranks.x = 0` (no Rank 0 — mechanism hypothesis H6) and `ranks.y = 0` (no protected Rank 1 — exploratory).
