# D-005 The frontier of diverse solutions (2026-10-05, Berk)

**Berk, 23:4x:** "itrea is multiobjective but we can start single objective but we shall also define a 'frontier' of
diverse solutions."

**Decision.** Start with one objective (the MIP's) and keep the code multi-objective-ready; add a **frontier**: an
archive of diverse near-optimal **feasible** solutions, maintained beside the three ranks (`itrea.ea.frontier`,
`EAConfig.frontier`).

- **Definition (default `pareto_incumbent`):** feasible solutions inside a quality band (key ≤ inc + band·max(1,|inc|),
  band = 5 %) that are non-dominated in **(objective ↓, distance from the incumbent ↑)** — for every level of quality, the
  solution farthest from the incumbent. Thinned to `max_size` (32) by crowding distance; the incumbent and the farthest
  member are always kept. Alternative `maxmin`: greedy max-min diverse set inside the band.
- **Distance:** share of integer variables that differ from the incumbent (normalised Hamming); `l1` (range-normalised)
  for problems without integers.
- **Why this shape:** part I's Rank 1 was a Pareto front the decision maker chose from. With one objective the front
  of objectives is a single point; making diversity the second axis keeps the "front of good alternatives" idea, and
  with several objectives the same archive becomes the Pareto front (`nondominated_fronts_k` handles k objectives).
- **Uses:** (1) an output — diverse alternative solutions, reported with size, spread (mean pairwise distance), and the
  2-D hypervolume in (relative gap, distance); (2) a source of immigrants after stagnation (`frontier.reinject`);
  (3) optionally posted to the solver (`exchange.post_frontier`; SCIP keeps non-improving solutions in its solution
  storage, where its crossover-type heuristics can use them) — an E3 ablation, off in the main arm (the brief asks the EA
  to hand over *better incumbents*).
- **Pre-registration:** frontier metrics are **exploratory** (no confirmatory hypothesis); D-002's H1–H5 are unchanged.
  Added exploratory comparison: frontier hypervolume and spread of A3 vs the solution pools SCIP and cuOpt end with.

**Multi-objective later (phase E6, not scheduled):** objective vectors `F(x) = (c₁·x, …, c_k·x)`, Rank 1 = feasible
non-dominated, Rank 0 = infeasible not dominated by any feasible, Rank 2 = the rest — part I's definitions verbatim;
instances from a public multi-objective MIP library (to be chosen and verified before any claim).
