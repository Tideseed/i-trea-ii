# D-007 Main claim — the coupling beats either method alone (2026-10-06 ~00:0x, Berk)

**Berk:** "take the i-trea as only the starting point and the idea of using good infeasible solutions. other than that
it is 15 year old research. main idea in i-trea ii is that a gpu based evolutionary algorithm metaheuristic coupled
with an exact solver can beat either single solution."

## Decisions
1. **Main claim (primary hypothesis):** the coupled system — GPU EA ⇄ exact solver with two-way exchange (arm A3) —
   has a lower primal integral than **each** of its parts alone: the solver alone (A0) **and** the EA alone (A5).
   Both comparisons must succeed (an intersection–union test: each at α = 0.05, no multiplicity adjustment, because
   the claim needs both).
2. **What makes the claim informative:** two methods side by side beat either alone partly by being two (at every
   moment the better trajectory counts). The key secondary therefore compares the coupling with the **uncoupled
   portfolio** of the same two parts: the pointwise best of the A0 and A5 trajectories of the same block (computed
   from those runs — optimistic for the uncoupled side, because the two did not share the machine), checked live on a
   subset (arm A3r: SCIP ‖ EA, no exchange, 8 test instances × 2 seeds). If A3 beats A0 and A5 but not the portfolio,
   the honest reading is "running both helps; the exchange adds nothing measurable".
3. **Other controls stay, as secondaries:** SCIP ‖ cuOpt (A2: is our GPU EA a better partner than NVIDIA's GPU
   heuristics on the same GPU?), SCIP ‖ SCIP (A1: equal CPU), tuned SCIP alone (A0*).
4. **I-TREA (part I; Orbay 2011 M.S. thesis, `orbay2011itrea`) is the starting point, not a template.** What carries
   over as a principle: **keep good infeasible solutions** (Rank 0 — infeasible solutions better than the best feasible
   one, a reserved share x of the population). The rank names stay. Faithfulness to part I's elimination order,
   tournament or operators is no longer a goal: every such choice is an ordinary parameter, chosen on the tuning set
   like any other. [[D-004 Mapping I-TREA (2011) to I-TREA II]] stays as a record of where the defaults came from; its
   "part-I-faithful configuration" is dropped from the planned comparisons.
5. **Mechanism hypothesis kept:** Rank 0 (good infeasible solutions) helps — the EA alone with x = 10 % vs x = 0 (A5 vs
   A5x0). The protected elite (y) moves to exploratory.

## Effect on D-002
D-002 becomes v3 (hypotheses and arms only; pool, metrics, analysis and gates unchanged). Per block the GPU arms are
A3, A5, A2, A5x0 (4 × 300 s) — about 2.6 nights for 32 instances × 2 seeds; A3r on a subset; A6 one seed.
