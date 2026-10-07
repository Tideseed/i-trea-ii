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
**Status:** built 2026-10-07 (workflow: implement → three adversarial reviews → fix → independent CPU verification).
Review: 12 findings — 3 major, all the same point (A3 is not A9's matched control: A9's EA sits on one X925 core, A3's
on three A725 cores, and its HiGHS LP worker shares them) → new development arm **A3x** (SCIP ↔ EA, EA on X925 core 6,
no cuOpt) as A9's matched control; 9 minor (attribution counting units, a latent verdicts.jsonl bug, crash flags for
runner-stopped processes, cuOpt threads in the run id, missing flag-off traffic tests); 3 parts rejected with evidence.
Verified: CPU suite 604 passed; every recorded hash unchanged; the flag-off engine bit-identical to HEAD on four
scenarios incl. scripted SCIP traffic (new fixture `tests/fixtures/engine_reference_traffic.npz`). **Not yet run on the
GPU** (cuOpt writing the new channels, the CuPy path, a real three-process launch). A9's memory gate needs 73 GB →
both LLM servers stopped. Config for development: `configs/itrea2-dev-rc-3way.yaml` (= the development default, repair
clock active; hash `1d12ad490dd0`). Issue #23.

## Addendum 2026-10-07 (evening) — the matched control is A3x, not A3
The build's reviews found that "A9 vs A3" mixes cuOpt's effect with a change of the EA's cores. In A9 cuOpt holds the
three A725 cores, so the runner puts the EA on the lane's second X925 core (6). A3's EA runs on the A725 cores (1–3).
The development configuration (`configs/itrea2-dev-rc-3way.yaml`) sets `seeding.lp_source: highs_async`, and the HiGHS
LP worker inherits the EA's affinity: in A3 it runs beside the EA's host loop on the spare A725 cores, in A9 it shares
core 6 with it for up to `lp_time_limit` (60 s; 9–34 s on sing44, ca, neos-4763324-toguru,
cryptanalysiskb128n5obj16 in past runs, the full 60 s on rail01 and chromaticindex1024-7). Outside that window A9's
EA runs on a faster core (3.9 vs 2.8 GHz). The LP worker cannot get a core of its own on the GPU lane (A725 1–3 are
cuOpt's, 0 Redis's, 4 and 10–14 the runner's).
**Decision:** a new development arm **A3x** = A3 with A9's EA footprint (SCIP on X925 5 with A3's flags, the EA on
X925 6, no cuOpt; same configuration file and run hash as A3, its own arm label). The fourth comparison becomes
**A9 vs A3x** (does cuOpt help the coupled system); A9 vs A3 is reported as descriptive only. A9's job files
include A3x jobs in the same blocks.
