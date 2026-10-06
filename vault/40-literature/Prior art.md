---
tags: [literature, prior-art, synthesis]
updated: 2026-10-05
---

# Prior art — synthesis (sweeps A–E, 2026-10-05)

**Rank names** follow the vault ([[D-004 Mapping I-TREA (2015) to I-TREA II]]): **Rank 0 = infeasible good**
(infeasible, key better than the incumbent, at most ⌊x·P⌋; the brief's "R2"), **Rank 1 = feasible good** (the best
⌈y·#feasible⌉ feasible, protected every generation; the brief's "R1"), **Rank 2 = rejects** (the rest; the brief's "R3").

## 1. The sweeps

| sweep | note | rows | legally readable | paywalled-only |
|---|---|---|---|---|
| A | [[A Constraint handling with infeasible solutions]] | 20 | 14 | 6 |
| B | [[B Metaheuristics cooperating with MIP solvers]] | 22 | 17 | 5 |
| C | [[C GPU primal heuristics and GPU EC]] | 23 | 22 | 1 |
| D | [[D Benchmarks and metrics]] | 20 | 19 | 1 |
| E | [[E Novelty check 2026-10-05]] | 16 | 15 | 1 |
| **unique works** | (8 works appear in more than one sweep) | **91** | **78** | **13** |

No work in any sweep was found not to exist. Rules for every citation: a legally readable copy (open venue, arXiv,
open-access journal or proceedings, author's or institution's page, PMLR, OpenReview, Optimization Online, ZIB reports)
or a verified preprint; paywalled-only works live in [[Paywalled research]] and are not cited in the main text; nothing
is called "first" or "novel" — a search that found nothing is recorded as a negative search result. The paper's
bibliography is `paper/refs.bib` (legally readable works only).

## 2. Lineage of the idea

**Constraint-handling EAs that keep infeasible solutions (sweep A).** Deb's feasibility rules (2000) made "feasible
beats infeasible" the parameter-free default; stochastic ranking (Runarsson & Yao 2000) and the ε-constrained method
(Takahama & Sakai) relaxed it. Several lines deliberately kept infeasible solutions: ASCHEA (Ben Hamida & Schoenauer
2000) steers the feasible proportion and protects a share of the best feasible individuals; FI-2Pop (Kimbrough et al.
2003/2008) keeps a separate infeasible population bred to reduce violation; Farmani & Wright (2003) single out the
infeasible solutions whose objective beats the best feasible one, to set penalties; SMES and Mezura-Montes & Coello
(2005) copy the best infeasible solution into the next population at a small rate; IDEA (Singh, Isaacs, Ray, Smith
2008; Ray et al. 2009; open account in Singh's 2011 thesis) reserves a fixed proportion α of the population for
infeasible solutions. **I-TREA (Orbay 2011, METU M.S. thesis; unpublished manuscript)** cites IDEA and adds three ranks for a
bi-objective problem with an interactive decision maker: Rank 0 infeasible good (not dominated by any feasible
solution) with a preserved share (a floor, e.g. 10 %), Rank 1 the feasible front, Rank 2 the rest.

**Solver and EA cooperation (sweep B, E).** Sequential hybrids came first: French et al. (2001) seed a GA from B&B node
information and graft its best back; Nieminen et al. (2003) use a GA's first integer solution as B&B's starting
incumbent. Parallel two-way exchange followed: Nwana et al. (2005) run SA and B&B concurrently and exchange incumbents
both ways on MIPLIB 0-1 programs; Puchinger, Raidl & Gruber (2005) run a memetic algorithm beside CPLEX branch-and-cut,
exchanging new best solutions both ways (a received solution replaces the population's worst) plus LP duals;
Gallardo et al. (2007) interleave an MA with beam-search B&B. Inside the solver, Rothberg's (2007) evolutionary
polishing and SCIP's crossover and mutation heuristics (Berthold 2006; adaptive since Hendel 2022) recombine solutions
from the solver's pool. Among exact solvers, racing with incumbent broadcast was measured (Carvajal et al. 2014:
"positive, but usually mild"; FiberSCIP 2018: 1.05× / 1.17× geometric-mean time at 4 / 8 threads), and ReXi (Mexi &
Rehfeldt 2026) combines racing SCIP workers with local-search and LNS workers over a shared pool — the leading
non-commercial solver on MIPfeas. Gurobi's concurrent MIP and ParBalans race without exchanging solutions.

**GPU heuristics (sweep C).** GPU island-model GAs date back at least to Pospichal et al. (2010), with large speed-ups on
tiny continuous problems. For MIP, the enabling pieces came later: first-order LP on the GPU (PDLP 2021, cuPDLP,
cuPDLPx), GPU bound propagation (Sofranac et al. 2022), and LP-free local search suited to batching (Feasibility Jump
2023, Local-MIP 2024). NVIDIA cuOpt combines them into a GPU primal-heuristic stack (Çördük et al. 2025) and — in its
open source — a GPU population with recombination beside its own CPU B&B. Kempke & Koch (2026) apply PDLP-guided
fix-and-propagate to unit-commitment-scale models; CHAP (2026) pools GPU and CPU heuristics' feasible and infeasible
candidates; PaNGEA (2026) explores many subproblems in parallel on the GPU; Sim et al. (2026) run penalty-tempered
chains on the GPU that may cross into infeasible regions.

I-TREA II sits where these three lines meet: a GPU population (line 3) with part I's ranks (line 1), beside an external
general MIP solver with two-way exchange (line 2).

## 3. The closest prior works

Judged against the full combination: GPU population + external general MIP solver + two-way exchange + capped,
incumbent-relative infeasible rank (Rank 0) + protected top-y % feasible rank (Rank 1).

| # | work | what it shares with I-TREA II | where it differs (evidence) |
|---|---|---|---|
| 1 | **cuOpt's population** (`nvidia2026cuoptsrc`, `corduk2025evoframework`) | GPU population with recombination; infeasible members kept; best feasible protected in slot 0; concurrent CPU B&B; B&B → population and population → B&B incumbent flow (the latter in opportunistic mode) | its weight rule keeps the best infeasible member **behind** the best feasible (the opposite of Rank 0); only the single best feasible is protected; the B&B is cuOpt's own, not an external general solver |
| 2 | **ReXi** (`mexi2026rexi`) | heuristic workers beside B&B workers, two-way exchange, crossover and mutation over a shared pool, local search restarted from the pool's best | CPU only; at most two pooled parents, no persistent population; no infeasible members; the pool admits only improving solutions |
| 3 | **Puchinger, Raidl & Gruber 2005** (`puchinger2005cooperating`) | an EA population and an exact B&C solver in parallel; new best solutions both ways; received solutions replace the population's worst | MKP-specific, CPU, no infeasible rank, no protected share; also passes LP duals (I-TREA II passes the root LP solution once and the proven bound) |
| 4 | **CHAP** (`tjusila2026chap`) | GPU + CPU heuristics sharing feasible incumbents and infeasible candidates under an objective cutoff | no B&B solver in the loop; tabu search and repair, not an EA; the infeasible pool guides repair and has no stated cap or "better than incumbent" rule |
| 5 | **IDEA** (`singh2011thesis`; paywalled originals `ray2009idea`, `singh2008idea`) | a fixed share of the population reserved for infeasible solutions, selected by non-dominated sorting in (objective, violation) | no test against the incumbent; the share is filled (and overfilled when feasible solutions are scarce) rather than capped; infeasible members rank above all feasible ones; no solver |

Close on single components: Farmani & Wright 2003 (Rank 0's exact membership set, used for penalties); ASCHEA 2000
(a protected share of the best feasible — the Rank 1 analogue); Nwana et al. 2005 and Gallardo et al. 2007 (two-way
heuristic ↔ B&B exchange); Vidal's HGS 2022 (managed infeasible share with preserved elites, CVRP).

## 4. How I-TREA II differs — only where the evidence supports it

1. **Rank 0's rule.** Membership requires the key to beat the incumbent, which may be the external solver's; at most
   ⌊x·P⌋ survive, over-quota and bound-demoted candidates are eliminated, and infeasible solutions that do not beat the
   incumbent may still live in Rank 2. The searches found no EA with this incumbent-relative cap for general MIP
   (sweeps A, C, E — a negative search result).
2. **The solver's proven bound demotes Rank 0** members that can never be feasible at their objective — the solver takes
   the decision maker's role of part I. None of the cooperative designs read in sweeps B and E does this (they pass
   incumbents, bounds or duals, and none keeps infeasible members on purpose); no search targeted this rule
   specifically, so it is an observation about the works read, not a search result.
3. **Partner and placement.** The population runs on the GPU and the partner is an external general MIP solver (SCIP,
   HiGHS); the cooperative EA designs found are CPU-only and problem-specific, and the GPU population found (cuOpt)
   works with its own B&B.
4. **Protection of a top-y % feasible share** rather than the single best feasible (cuOpt) — but protecting a feasible
   share as such is established (ASCHEA), so this is a parameter choice, not a contribution.
5. **Method.** Parametric operators, a pre-registered protocol with a cuOpt race on the same GPU as the primary
   control, lineage tracing of improvements to Rank 0 ancestors ([[D-002 Pre-registration — hypotheses, arms, analysis]]).

Nothing in the sweeps supports a stronger claim, and none should be made before the experiments.

## 5. Established — must not be claimed as ours

- Keeping a fixed share of infeasible solutions in an EA population (IDEA; I-TREA part I itself), keeping a separate
  infeasible population (FI-2Pop), protecting a share of the best feasible individuals (ASCHEA), and singling out
  infeasible solutions that beat the best feasible one (Farmani & Wright 2003).
- Two-way solution exchange between a population heuristic and an exact B&B solver running in parallel
  (Puchinger–Raidl–Gruber 2005; Nwana et al. 2005; Gallardo et al. 2007; ReXi 2026).
- Incumbent broadcast among racing solvers, with measured gains (FiberSCIP; Carvajal et al. 2014; Ralphs et al. 2018).
- A GPU MIP population with recombination, a protected best feasible, infeasible members and a concurrent CPU B&B
  (cuOpt). A GPU+CPU pool of feasible and infeasible candidates (CHAP).
- Crossover and mutation for MIP as sub-MIP neighbourhoods over a solution pool (Rothberg 2007; Berthold 2006;
  Hendel 2022; ReXi).
- Feasibility-Jump-style and Local-MIP-style local search, batched on the GPU (cuOpt, PaNGEA); GPU bound propagation;
  GPU first-order LP.
- Deb's rules, stochastic ranking and the ε-constrained method as comparators.
- The primal integral (Berthold 2013) and the MIPfeas protocol (Bussieck & Dirkse 2026).

## 6. Open questions for the experiments

1. **Which primal integral is the headline?** D-002 pre-registers Berthold's P(T)/T (1 before the first incumbent). The
   only published GB10 cuOpt baseline (0.0393) uses MIPfeas (2 before the first incumbent, P/T ∈ [0, 2], shifted
   geometric mean with s = 0.001), and the GAMS blog and the plato page disagree on the denominator's "+1"
   ([[D Benchmarks and metrics]] §2). D-002 may be amended before the first confirmatory run; computing both from the
   same traces costs nothing.
2. **Is SCIP alone the right CPU bar?** ReXi / SCIP's concurrent mode is the strongest non-commercial MIPfeas solver; a
   SCIP‖cuOpt or SCIP‖I-TREA II arm that does not beat concurrent SCIP on the same cores would be a weak result. D-002
   has no such arm.
3. **What does the EA add over cuOpt's own population?** A6 (cuOpt alone) and A2 (SCIP‖cuOpt) are in D-002; the
   mechanism comparison (Rank 0 rule versus cuOpt's "best infeasible behind best feasible" weighting) is not.
4. **Which HiGHS?** Vanilla HiGHS 1.15.1 already calls Feasibility Jump once; a patched build with FJ, FPR and LocalMIP
   on parallel workers exists (`spoorendonk2026highsheur`). The H* replication should state which.
5. **Does injection reach the solver's heuristics?** SCIP's crossover / mutation / ALNS draw on its solution storage;
   that injected EA solutions are used is our inference. Log accepted posts and later solver incumbents whose
   neighbourhood used them, where SCIP reports it.
6. **Expected effect size.** Exchange among exact solvers gave 5–17 % (FiberSCIP) or "usually mild" gains (Carvajal);
   optopt's CPU EA gave none at 60 s. Plan the instance count for small effects.
7. **Rank 0 floor versus cap.** Part I preserved a floor of 10 %; the brief caps at x %. D-002's x ablation covers the
   cap; a floor variant (part I's semantics) is not in the plan and would be exploratory.
8. **Feasibility test.** Align `feas_tol` / `int_tol` with the MIPLIB checker (1e-5 relative with the max{…, 1} scale,
   1e-4 absolute integrality) so that a Rank 1 member is feasible by the benchmark's own definition.
9. **Hard families.** PGLib-UC v19.08 with Knueven et al.'s formulation, larger SCUC cases from UnitCommitment.jl (with
   its independent checker), and the UC-scale models where Kempke & Koch report that commercial solvers find nothing
   in two days; MIPLIB status files pinned at `hard-v31` / `open-v30` / solufile 37.
10. **Variability.** FiberSCIP reports up to ~93 % time variation between repeats; MIPfeas runs once per pair. D-002's
    3 seeds are the minimum; report seed spread.

## 7. Citation hygiene

- **Alias keys** (the same work under two sweep keys): `corduk2025gpuheur` → `corduk2025gpuprimal`;
  `nvidia2025cuopt` → `nvidia2026cuoptsrc`; `luteberget2023fj` → `luteberget2023feasibility`. Use the canonical keys.
- **Keys keep the preprint year** where the journal version came later: `kempke2025fixprop` (MPC 2026),
  `lu2023cupdlp` (OR 2025), `huang2024evox` (TEVC 2025); `year` fields use the publication year.
- **Provenance caveats needing a strictness call** before the paper cites them: `cantupaz1998survey` (third-party
  mirror), `derrac2011` (publisher PDF on the authors' group site), `pospichal2010cudaga` (publisher-typeset copy on
  the faculty repository), `blum2008hybridizations` (time-limited COVID-19 grant on PMC).
- **Traps:** Runarsson & Yao (2000) is mislinked to arXiv:1510.04409 by OpenAlex and Semantic Scholar; Coello's author
  files are `.pdf.gz`; `corduk2025gpuprimal` is a workshop poster; `spoorendonk2026highsheur`'s heuristics are not in
  upstream HiGHS; `gallardo2007mabnb`'s open copy is a draft (do not quote its pagination).
- **The I-TREA thesis** (`orbay2011itrea`) is public and citable; the unpublished manuscript is not, without Berk.
