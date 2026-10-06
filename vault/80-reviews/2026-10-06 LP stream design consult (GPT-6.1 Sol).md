# 2026-10-06 LP stream design consult — Hermes Agent with OpenAI GPT-6.1 Sol

**What:** before building [[D-011 Solver streams good infeasible solutions to the EA]], the design question was put to
Hermes Agent with OpenAI GPT-6.1 Sol (reasoning high), one-shot (`hermes -p gpt chat --oneshot`), 2026-10-06 15:17,
on Berk's suggestion ("You can use [local LLM agent] to help with the development btw"). The brief contained only the design,
public instance names and our own measurements — no keys, no private data. Answer received ~15:24.

**Correction to the brief:** it quoted ~0.1 s per LP read on 117k variables; measured minutes later: ~0.4 µs per
variable (16.5 ms on radiationm18-12-05, ~50 ms on rail01). The throttling advice is relative, so it stands.

## What was adopted
- Send raw fractional points; round and score in the EA on the GPU (not in SCIP).
- Infeasibility measured on the rounded point; rank (V, key) — Berk's order — inside the EA's pool.
- Nearest rounding + 3 randomised roundings per point, repaired, admitted through ordinary survival only (no Rank 0
  bypass).
- A time budget of ~2 % of SCIP's wall clock with a start allowance; measured cost, not a node count.
- Rank 0 flooding watched (`rank0_stream` in the generation record) rather than capped up front.
- Its expectation that better seeds will not fix equality chains (rail01, radiationm18-12-05) — the same diagnosis as
  ours; coordinated moves are the separate next item.

## What was not (yet) adopted, and why
- A provenance ceiling in Rank 0 (≤ 8 stream-born members) and ≤ 4 admitted per generation: offered as "experimental
  safeguards"; we measure `rank0_stream` first (`per_tick` = 2 points × 4 children already bounds the inflow).
- Novelty selection within quality bands (farthest integer genotype among comparably good points): Berk's order is the
  rule under test; novelty is a later variant if the pool turns out to be near-copies.
- A capture-and-discard arm to isolate the stream's cost: SCIP's own `lp_stream.spent_s` measures the read cost
  directly; the arm is run only if the effect is ambiguous.
- Its scaling of row residuals by max(||a_i||, |lhs|, |rhs|): our V uses `ranks.viol_norm` (l2 by default), the same
  scale the survival uses.

## Brief (verbatim)
You are helping design one mechanism in a research prototype (private research code; public benchmark instances only). Please answer as a critical MIP/metaheuristics expert. Be concrete and brief (max ~900 words), no preamble.

SYSTEM. I-TREA II: a GPU evolutionary algorithm (population P=256, CuPy) runs concurrently with SCIP 10 (PySCIPOpt 6.2, single thread) on one MIP instance, for a fixed wall-clock budget (120 s in development). Two processes, a file/Redis mailbox between them.
- EA population has three ranks: Rank 0 = infeasible individuals whose objective is better than the incumbent (capped at x = 10 % of P; selected by non-dominated sorting on (objective, violation)); Rank 1 = protected best y = 10 % of the feasible individuals plus the best; Rank 2 = the rest. Children are repaired by a GPU Feasibility-Jump local search (weighted violation, single-variable jumps, objective cutoff row at the incumbent).
- EA -> SCIP today: every new EA incumbent (and a few near-best feasible ones) is handed to SCIP through a heuristic plugin (createOrigSol + trySol), so SCIP uses it as an incumbent/starting point. Works well on some instances (glass4: 213 of 228 EA posts became SCIP incumbents).
- SCIP -> EA today: SCIP's feasible solutions (incumbent + solution pool within a band), its dual bound, and ONCE the root LP solution, which the EA rounds/perturbs to seed individuals. That root LP point alone makes the EA find feasibility on two instances where it otherwise never does (a unit-commitment model and sing44).

THE MECHANISM TO ADD (the PI's idea): "when the solver comes up with suitable good infeasible solutions (minimize infeasibility, then optimize the objective) it injects them into the EA population". Goal: more diversity in the EA, and super-optimal infeasible material for Rank 0 that the EA's repair can turn feasible. The external solver stays SCIP (a general solver, not embedded heuristics).

OBSERVATIONS on the 9 development instances (120 s): coupled beats SCIP alone where the EA contributes (glass4, neos-960392, neos-3024952-loue, item_placement). On sing44 and radiationm18-12-05 coupled is slightly worse than SCIP alone (GPU competes for memory bandwidth; EA never feasible, 0 accepted posts). rail01 (117k binaries, 45.5k equality rows with rhs 0 and +-1 coefficients, flow-conservation-like): nobody finds a solution in 120 s except cuOpt late. radiationm18-12-05 (general integers + continuous, equality chains cont=int, cont=bin, b1+b2=1, i1=i2-i3): cuOpt is feasible at 0.8 s, our FJ repair stalls at a weighted violation of 10-50.
API constraint: PySCIPOpt has no vectorised LP-solution accessor; reading the node LP solution means getSolVal(None, var) per variable (~0.1 s for 117k variables in Python), so the SCIP side must be throttled.

QUESTIONS
1. Which SCIP-side sources of "good infeasible solutions" are worth streaming, ranked by expected value per unit of SCIP time: node LP solutions at AFTERLPNODE (raw fractional or rounded?), LP solutions during the root cut loop, pseudo-solutions, solutions rejected by trySol, end points of SCIP's diving heuristics (reachable from Python?), anything else? For each: is its objective guaranteed better than the incumbent, and how diverse are successive samples?
2. The PI's ranking "minimise infeasibility, then optimise objective": on which representation should infeasibility be measured before sending (fractional LP point vs a rounded point; which violation norm), and how should SCIP choose k points to send per post so the stream is diverse rather than k near-copies from one dive?
3. Throttling: a defensible rule that keeps SCIP's overhead under ~2 % of wall clock while still sending enough points early (the first seconds matter for the primal integral).
4. EA admission: how should the EA turn a received point into individuals (round once, several randomised roundings, repair with FJ first?), and where should they enter (Rank 0 directly if objective < incumbent, or as immigrants that must survive selection)? How to stop the stream from flooding Rank 0 and collapsing diversity?
5. Known prior art or pitfalls for streaming B&B node LP solutions to a metaheuristic population (cite only works with open full text; say "unsure" when unsure).
6. Anything in this mechanism you expect NOT to work, and the cheapest experiment that would tell.

## Answer (verbatim)
1. Sources, ranked by expected value per SCIP second

Start with passive LP snapshots, not a new SCIP infeasibility-minimisation task. SCIP does not ordinarily optimise your violation metric.

  1) First root LP, then one post-cut root LP. Highest early value; you already have evidence for the first. Export raw fractional coordinates. Successive cut-round points may improve bounds without producing much genotype diversity.
  2) Selected ordinary node LP optima, preferably different branches rather than consecutive nodes in one plunge. Greater potential diversity; fewer fractional variables does not necessarily mean easier repair.
  3) Near-feasible integral candidates already available in a plugin you control. Potentially excellent, but wrapping your own trySol does not intercept SCIP’s internal C calls. Its false return means “not stored”, not necessarily “infeasible”. EA-origin rejects provide no new source diversity.[1]
  4) Diving intermediate points. Promising, but not a cheap Python-first implementation: SCIP’s generic diving uses probing, a working solution, and restores probing state. I found no Python callback exposing built-in dive endpoints; unsure about complete coverage without C instrumentation. An infeasible terminal LP is not a usable primal point.[4]
  5) Pseudo-solutions: lowest priority. Objective-favouring bound assignments can be extremely row-infeasible; successive assignments may barely differ.

No source unconditionally guarantees improvement over the incumbent. An optimal ordinary LP below the live cutoff normally does; explicitly compare its original-space objective. Rounding destroys that property. Pseudo-points, heuristic candidates, and objective-modified diving points need independent comparisons. Without an incumbent, “super-optimal” is undefined.

API correction: AFTERLPNODE is a heuristic timing, not an event. DURINGLPLOOP supports cut-loop sampling; LPSOLVED means the node LP completed cut-and-price, not every intermediate solve. Reject non-optimal LP statuses.[2][3]

2. Representation and selection

Send raw original-variable coordinates; round and score on the GPU. Otherwise SCIP pays for an O(nnz) evaluation merely to choose exports. Preserve variable mapping through presolve; distinguish transformed and original objectives.[1]

Measure infeasibility on the integer-rounded phenotype, after continuous decoding. Raw LP row residuals mostly measure numerical noise; raw integrality distance is only a diagnostic.

Use tolerance-aware original-row residuals, scaled by max(||a_i||₂, |lhs_i|, |rhs_i|, epsilon), ignoring infinite sides. Rank by summed scaled residual, objective second; retain maximum residual and violated-row count as diagnostics. Exclude the incumbent cutoff from “model infeasibility”.

Do not impose this lexicographic order globally: it would erase valuable super-optimal trade-offs. Use it within quality bands, then select for novelty.

Default k=1 per capture. If batching, k=2 from distinct archived captures: best quality, then farthest integer genotype among comparably good points. Do not manufacture k near-copies in SCIP. Track normalised integer-coordinate distance and node provenance; perform expensive novelty checks in the EA.

3. Throttling

Count extraction, serialisation, mailbox work, and skipped-callback costs using monotonic timing. Let C be accumulated overhead and q a conservative estimated next-export cost. Permit capture only when:

  C + q <= min(0.02T, 0.20 seconds + 0.015t).

For T=120, this allocates approximately 2 seconds, leaving headroom below 2.4 seconds. Reserve startup credit for the first root snapshot; skip unchanged captures, require receiver capacity, and never wait for it. Schedule attempts by elapsed time, not every N nodes. Update q from measurements; disable exports if actual accounting crosses the budget.

This is a measured target, not a worst-case guarantee. Strict 2% at every time prefix conflicts with immediate 0.1-second extraction. No policy guarantees useful early points while SCIP is still presolving.

4. EA admission

Generate one nearest rounding plus three independent stochastic floor/ceiling roundings; retain continuous coordinates initially, then decode and apply bounded FJ. Deduplicate before repair. Test relaxing the incumbent cutoff during initial feasibility search on instances that never become feasible.

Evaluate before and after repair. Admit as immigrants into ordinary survival: infeasible and genuinely better than the current incumbent can compete for Rank 0, never bypass its selection rules. Checked feasible candidates follow normal incumbent/Rank 1 handling.

Keep the total Rank 0 cap of 25. Initially test a provenance ceiling of eight external-derived members and four admitted candidates per generation. These are experimental safeguards, not established optimal values. Coalesce duplicates, expire stale queued points, preserve elites.

5. Prior art

RENS formalises feasible LP roundings. MRENS uses several fractional references generated by SCIP’s Lagromory separator; it closely supports the multiple-reference rationale, but is not external EA streaming.[5] Those separator points are another possible source if already generated—not a reason to enable expensive separation solely for exports.

Unsure of open-full-text prior art matching this exact node-LP-to-population mechanism; do not claim novelty.

6. Likely failures and cheapest experiment

Independent rounding plus single-variable FJ may still fail on rail01’s conservation equations and radiation’s linked equality chains. Better seeds do not supply coordinated moves. Diagnose equality propagation/block moves before building dive instrumentation.

First replay a small captured LP archive through identical rounding/FJ budgets versus root-only seeds. Measure feasibility, repair work, novelty and incumbent quality; this tests seed value, not coupled wall-clock benefit.

Then compare matched runs: root-only; capture-and-discard; capture-and-admit. The middle arm isolates extraction cost, not all memory-bandwidth interference. Record checked P(120), time to feasibility, SCIP throughput, overhead and injection lineage. Reject the mechanism if useful descendants cannot repay its wall-clock cost.

Sources:
[1] https://raw.githubusercontent.com/scipopt/PySCIPOpt/v6.2.1/src/pyscipopt/scip.pxi
[2] https://raw.githubusercontent.com/scipopt/scip/v10.0.0/src/scip/type_timing.h
[3] https://raw.githubusercontent.com/scipopt/scip/v10.0.0/src/scip/type_event.h
[4] https://raw.githubusercontent.com/scipopt/scip/v10.0.0/src/scip/heuristics.c
[5] https://arxiv.org/html/2408.00718
