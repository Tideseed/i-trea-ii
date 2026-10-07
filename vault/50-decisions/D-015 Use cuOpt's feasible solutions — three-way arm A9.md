# D-015 Use cuOpt's feasible solutions — three-way arm A9 (2026-10-07 18:2x, Berk)

**Berk:** "We can use cuopt's feasible solutions."

**Why.** The class probe (D-014) showed the coupling's limit out of sample: the EA's own feasibility search found a
feasible point on only 3 of 8 new instances (0 feasible repair lanes on krka, kumeu, chromaticindex1024-7,
cryptanalysiskb128n5obj16, doon), so on the rest the coupled system was just SCIP — while cuOpt alone solved krka
(P 0.004), kumeu (0.10) and chromaticindex1024-7 (0.075) in seconds. Rather than rebuilding cuOpt's feasibility
machinery (feasibility pump, propagation rounding) inside the EA first, take its solutions.

**Decision.** Arm **A9 = SCIP ↔ EA ← cuOpt**: cuOpt as shipped (no set-solution callback, so presolve stays on) runs
beside SCIP and the GPU EA and streams its incumbents and bound to the EA on its own channels; the EA injects them
(survival decides), counts cuOpt's incumbent in its cutoff, and keeps exchanging with SCIP as in A3 — so cuOpt's
solutions, polished by the EA, reach SCIP. Cores: SCIP and the EA on the lane's two X925 cores, cuOpt on the A725s;
two CUDA processes on the GPU (EA pool + cuOpt RMM, 12 GB each). Attribution keeps cuOpt-origin vectors apart from
EA-made ones (no EA credit for an unchanged cuOpt vector).
**Comparisons (D-013 metric, weak sets first, mean as guard):** A9 vs A0 (helps SCIP), A9 vs A6 (helps cuOpt), A9 vs
A2 (SCIP ‖ cuOpt race: does the EA add anything beyond racing the two solvers), A9 vs A3 (does cuOpt help the
coupled system). Restarting cuOpt from the EA's best ([[2026-10-07 cuOpt restart design (workflow)]]) can be added on
top later (it needs ~10 s per restart, measured on glass4).
**Status:** being built (workflow: implement → three reviews → fix → CPU verification); GPU test in the next window.
Issue #23.
