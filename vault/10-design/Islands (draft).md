# Islands — searching large chunks of the space on the GPU (DRAFT for Berk, 2026-10-06)

**Berk (15:3x):** "Can EA be targeted to search large chunks of solutions space via GPU?" This note is the concrete
answer, for discussion — nothing here is built. Related: [[D-011 Solver streams good infeasible solutions to the EA]]
("Next"), [[2026-10-06 Feasibility targets consult (GPT-6.1 Sol)]] (per-scale demes).

## The idea in one paragraph
SCIP partitions the space as it branches: every open node is a region (the root's bounds plus the branching decisions
on its path). SCIP explores one node at a time on one core; the GPU can hold several sub-populations at once, each
confined to a region SCIP thinks is promising, and search inside them in parallel. A region island's good solutions go
to SCIP as usual (an incumbent found inside a node is valid everywhere); SCIP's pruning of a node tells the EA to close
the island. This is the population analogue of RINS/RENS-style neighbourhoods, run on the GPU and fed by SCIP's tree.

## One mechanism, two uses
An **island** = a slice of the population with its own survival quota, parents drawn within the island (migration with
a small probability), and optional island-local rules:
1. **Scale demes** (fhnw-binpack4-48): two islands that differ only in the row scale used for repair AND survival
   (l2 / rhs); migration re-evaluates a vector under the destination scale. Fixes the selection/repair misalignment the
   adaptive strategy showed (rhs-repaired children dying under l2 survival).
2. **Region islands** (Berk's chunks): an island whose genes on the node's branching path are frozen at the node's
   bounds (for binaries — almost all branchings here — that is a fixing), seeded by roundings of that node's LP point
   (the LP stream already sends it).

## Ranks across islands
- **Rank 1 stays global** — the best feasible solutions are protected whatever island found them (feasibility and key
  do not depend on the island).
- **Rank 0 is split:** the x·P cap is shared out across islands (super-optimal infeasible points are judged against
  the global incumbent; a region island's Rank 0 shows where inside that node the LP optimum lies).
- **Rank 2 per island**, generational as now.

## What it needs (estimate)
| piece | where | size |
|---|---|---|
| island id per individual (like the strategy gene), per-island quotas in survival, parent selection within an island | engine, ranks | medium; the strategy machinery is the template |
| per-island frozen-gene mask: mutation and crossover skip frozen genes; the feasibility jump never moves them (candidate filter in the kernel) | operators, `ls_fj_gpu` kernel, `ls_fj_cpu` | medium; one mask row per island, the kernel reads its island's row |
| SCIP side: the node's branching path with each LP-stream point (`getParentBranchings` walked to the root — depth ≤ a few dozen, cheap) and a "node pruned" signal | `scip_run` | small |
| island lifecycle: open on a good region (Berk's order: least infeasible, then best LP value), close when SCIP prunes the node or the island stagnates; at most K islands (e.g. 4) | engine | small–medium |

## Where it can and cannot help (from today's measurements)
- Helps only where SCIP has a tree: glass4, fhnw, loue, item_placement, radiationm (thousands of nodes in 120 s).
- Cannot help where SCIP stays at the root in 120 s: ca__Scenario400, sing44, neos-960392, rail01 (no nodes, so no
  regions) — there the EA's own HiGHS LP and its global search remain the tools.
- A region does not remove equality chains or min-max objectives (radiationm); it shrinks the space the repair works
  in, which helps only if the repair can make progress inside it.

## Order I would propose
1. Settle today's baseline (adaptive + HiGHS LP + stream at the default repair budget: run `lps0`).
2. Build islands with **scale demes first** (fhnw is a clear, measurable target; no SCIP-side change).
3. Then **region islands** on the same machinery, measured on the five instances with a tree.
