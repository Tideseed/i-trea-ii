# 2026-10-06 Feasibility targets consult — Hermes Agent with OpenAI GPT-6.1 Sol

**What:** the two instances where cuOpt alone finds solutions and the coupled system does not, put to Hermes Agent
with OpenAI GPT-6.1 Sol (reasoning high), one-shot, 2026-10-06 16:13 (Berk: "You can use [the hosted GPT agent] to help
with the development").
Context: [[Development log]] (coupled adaptive results; aggregation probe). Brief: design and our own
measurements only.

## Take-aways (ours)
- **fhnw-binpack4-48 — selection/repair misalignment.** Recommended: two *protected demes*, one per row scale, each
  ranking its infeasible members by its own fixed V; feasibility always by the one original-model checker; migration
  re-evaluates a vector under the destination scale. Second choice: Pareto survival over (V_l2, V_rhs) with each
  metric's best protected. Rejected: own-scale V in one population (a label change improves the score), rank-sum,
  violated-row count, population-median normalisation. This is an island model — the same machinery Berk's "large
  chunks" question needs (islands per region of SCIP's tree), so one island framework could serve both.
- **radiationm18-12-05 — a neighbourhood barrier, not a scale problem.** Recommended first: *residual-correcting pair
  / block moves* in the repair (for a violated integer equality, change two or a few integers so the row's residual
  closes, gcd and bounds checked; score over all incident rows; one move per individual per step), then a short
  residual sub-MIP on a few elites as a diagnostic, then fix-and-propagate; an LP over the continuous variables with
  integers fixed only as a completion step. Its note on cuOpt (open access, arXiv:2510.20499): propagation/probing
  rounding, local search, PDLP feasibility-pump projections and continuous LP polishing; it is unsure which produced
  the 0.8 s solution.
- Both first experiments are EA-alone repair diagnostics (D-011 §1 allows A5 as a diagnostic), 60 s, matched starts.

## Brief (verbatim)
You are helping a research prototype (private code, public MIPLIB instances). Critical MIP-heuristics expert; concrete; max ~900 words; no preamble.

SYSTEM. A GPU evolutionary algorithm (CuPy, population 256) runs beside SCIP and exchanges solutions both ways (EA incumbents -> SCIP via trySol; SCIP solutions, dual bound, root LP and now a stream of node LP points -> EA). Children are repaired by a Feasibility-Jump local search on the GPU (one thread block per individual; weighted violation sum_i w_i * viol_i / s_i with per-row scale s_i; moves = single-variable jumps: binary flip, integer +-1, bounds, and "tightening values" that make one violated row tight; weight bump at local minima; objective cutoff row once an incumbent exists). Survival ranks by (objective key, total violation V) with V on ONE row scale for every individual (default s_i = row 2-norm, "l2"). A self-adaptive strategy gives each individual its own REPAIR row scale (l2 or rhs: s_i = max(1,|rhs_i|)) and LP guidance on/off, inherited and mutated; survival V stays l2.

PROBLEM 1 (fhnw-binpack4-48, bin packing, 3,710 vars, 4,480 rows, no SCIP solution in 300 s; cuOpt feasible at ~0.6 s): with repair AND survival both on the rhs scale, the EA alone reaches the proven optimum in 30 s; with l2 on both it never finds a feasible point; with the self-adaptive strategy (repair rhs or l2 per individual, survival l2) the rhs strategies die out (final population 8-9 of 256) and nothing feasible is found. Question: what is a principled survival violation measure for the feasibility phase when individuals were repaired under different row scales? Options I see: (a) survival V on each individual's own scale (incomparable?), (b) rank-sum or Pareto over V_l2 and V_rhs, (c) number of violated rows, (d) per-row normalisation by a population statistic (e.g. median violation of that row), (e) let the strategy also choose the survival scale and keep separate sub-populations (islands) per scale with migration. Which would you pick, why, and what pitfalls?

PROBLEM 2 (radiationm18-12-05, 40,623 vars: 14,688 binary, 11,247 general integers in [0,324] or [-324,324], 14,688 continuous; 40,935 rows; cuOpt finds a feasible point at 0.8 s, SCIP at 0.7 s, our EA never). After exactly substituting all two-variable equality chains (40.6k -> 14.9k variables), the feasibility jump from LP roundings still stalls at total violation 40-90; the rows left violated are ~154 equality rows over >= 9 general integers with exact integer right-hand sides (sums), ~76 inequality rows over 3 continuous variables, a few 3-variable integer equalities (i1 = i2 - i3). Question: what repair move or construction would get such points feasible quickly on a GPU, in a population setting? Candidates: fix-and-propagate from the LP point (bound propagation, fix in LP-fractionality order, backtrack-free with repair at the end), "pair moves" that shift +d on one integer and -d on another inside the same equality row, line search towards the LP point, an LP over the continuous variables with integers fixed (CPU), a short sub-MIP. Rank them by expected value for this structure and say what cuOpt most likely does here (say "unsure" when unsure; cite only open-access work).

Answer each problem separately, with a recommended first experiment for each (cheap, on a 120 s budget).

## Answer (verbatim)
PROBLEM 1 — Protect competing feasibility landscapes

Pick (e): two protected demes, with a common survival scale within each. This is a portfolio over repair/survival landscapes, not a claim that either violation measure universally predicts distance to feasibility.

Your observation suggests selection–repair misalignment, but does not isolate it: changing both scales changes both search and selection. Extinction under l2 selection is not evidence that rhs repair is ineffective.

Use 128 survivor slots per scale. Within each deme, rank infeasible points by its fixed V; use objective only as a secondary criterion before feasibility. Determine feasibility everywhere using the same original-model checker and tolerances, never a scale-dependent V threshold. Keep the two repair-scale niches protected initially; adapt LP guidance within them. Later, migrate vectors by re-evaluating them under the destination scale, without importing scores or penalty weights.

Why not the alternatives?

  (a) Own-scale V is unsuitable for global comparison: changing a strategy label can improve its score without changing its vector. Within a fixed-scale deme it is fine.

  (b) Pareto survival on (V_l2, V_rhs) is my second choice. It retains both extremes without comparing magnitudes. Explicitly protect each metric’s best points; crowding can otherwise discard useful specialists. Rank-sum is weaker: an extreme specialist can lose to middling compromises, and rankings depend on population composition.

  (c) Violated-row count is useful telemetry or a tie-breaker, not the main score. It creates plateaus, ignores violation magnitude, and depends on redundant rows and tolerance.

  (d) Population-median normalisation introduces a moving target. Often the median is zero; floors then determine selection. A persistently difficult row can become downweighted precisely because everyone violates it. Freeze any such scales across a selection epoch and apply them to every candidate identically; I would not start here.

Pitfalls: protection spends resources on a bad strategy and halves each niche’s search population. Do not use raw V values to reallocate niche budgets. Eventually allocate by checked feasibility/improvement per repair time, with a nonzero exploration floor. Also, max(1, |rhs|) is not generally invariant to algebraic row rescaling; verify that the rhs advantage is not specific to arbitrary formulation units.

First experiment: two sequential 60 s EA-only runs, same initial vectors and repair budgets: current adaptive/l2 control versus protected, locally scored scale niches. Keep other settings unchanged and document the changed scale-inheritance rule. Log births and survivors by strategy, best V under both scales, violated-row count, and checked time-to-first-feasible. This is a diagnostic, not a statistical comparison; retain the existing pure-rhs result as context, not a matched control.

PROBLEM 2 — Repair integer equations, not just weighted violations

The remaining integer equalities suggest a neighbourhood barrier; a small aggregate V does not establish proximity to an integer-feasible point.

My ranking by expected incremental value:

1. Equality-aware residual correction and block moves. For a violated integer row, generate bounded integer changes satisfying a_p*delta_p + a_q*delta_q = b − a*x. A plain (+d, −d) move preserves an equal-coefficient sum; it cannot repair that row’s nonzero residual. Use residual-correcting pairs first, sum-preserving exchanges afterwards. With general integral coefficients, check gcd divisibility and bounds. For i1 = i2 − i3, include the corresponding coupled moves. If overlapping equations defeat pairs, expand to small connected blocks or cycle moves; preserving one row is not preserving the equality subsystem.

   GPU implementation: sample a capped number of pairs per violated row, score changes over the union of incident rows, then commit one move per individual. Avoid enumerating every pair or applying conflicting moves concurrently. Consider LP-guided, sum-preserving rounding as an initializer, but only call it equality-preserving when all overlapping equations are handled.

2. Short residual sub-MIP. Free variables around violated integer rows, include all their incident constraints, and expand the neighbourhood if necessary. Prefer an explicit move-distance objective over arbitrary tight bounds. This is a strong diagnostic of missing move support, but not a GPU-population primitive; use a few elites, not 256 CPU solves.

3. Fix-and-propagate. Promising construction, but backtrack-free rounding is risky with coupled exact sums. Propagate both sides of equalities with integer-safe bounds; prioritise tight domains/slack consumption, not fractionality alone. Allow bounded undo or restart. Consistent propagated bounds do not prove a feasible completion.

4. Continuous LP with integers fixed, alone. It cannot fix the integer-only equalities. As a completion step after integer repair, however, it is essential whenever continuous feasibility remains unresolved. Reuse a model/basis and process only promising elites.

5. Line search towards the LP point. Fractional movement followed by rounding generally recreates broken sums. Full LP projection alternating with rounding is a feasibility-pump mechanism, not merely line search.

cuOpt: its published GPU stack combines propagation/probing-based rounding, local search, PDLP feasibility-pump projections, and continuous LP polishing.[1] That combination plausibly explains the gap. Unsure which component produced this instance’s 0.8 s solution; timing alone cannot identify it.

First experiment: two 60 s repair replays from matched stalled populations: existing FJ versus residual-correcting/block proposals added to FJ. Give both arms identical conditional continuous-LP completion. Disable exchange to isolate repair; check reconstructed vectors against every original row, bound and integrality condition. Record exact integer-row residuals separately from continuous infeasibility. If pairs stall, test a short residual sub-MIP before investing in broader GPU neighbourhoods.

Sources:
[1] https://arxiv.org/html/2510.20499v1 — Çördük et al., GPU-Accelerated Primal Heuristics for Mixed Integer Programming (open access)
