---
tags: [literature, prior-art, sweep-E, novelty]
updated: 2026-10-05
---

# Prior art E — novelty check, 2026-10-05: a GPU EA beside a solver with two-way exchange

Sweep date 2026-10-05. Question: has anyone run a population heuristic — on the GPU or not — beside a general MIP solver,
exchanging solutions in both directions, with a capped share of infeasible solutions that beat the incumbent? And what
exactly is the I-TREA lineage? This note is the **novelty record**: it lists what exists and what the searches did not
find. It makes no "first" claim; a search that finds nothing is reported as a negative search result, not as proof of
absence.

**Rank names** follow the vault ([[D-004 Mapping I-TREA (2011) to I-TREA II]]): **Rank 0 = infeasible good** (the
brief's "R2"), **Rank 1 = feasible good**, protected (the brief's "R1"), **Rank 2 = rejects** (the brief's "R3").

## 1. Headline: the I-TREA master's thesis is online

**Berk Orbay, *An Interactive Evolutionary Algorithm for the Multiobjective Relocation Problem with Partial Coverage*,
M.S. thesis in Operational Research, Middle East Technical University (METU/ODTÜ), April 2011 (approved
22.04.2011).** Persistent id hdl:11511/20509; landing page
https://open.metu.edu.tr/handle/11511/20509; open PDF https://open.metu.edu.tr/bitstream/handle/11511/20509/index.pdf
(143 pp.). §5.1 names the method "Interactive Three Ranks Evolutionary Algorithm (I-TREA)". Key `orbay2011itrea`.

The thesis is public and open access, and it is the only part-I source this repository cites.

What the thesis says, and how it bears on the design:

- **Ranks.** Rank 0 "infeasible good" = infeasible solutions not dominated by any feasible or inferior (DM-rejected)
  solution; Rank 1 "feasible good" = the whole feasible non-dominated front; Rank 2 "rejects" = everything else,
  feasible or not. Tournament preference Rank 1 > Rank 0 > Rank 2. Bi-objective, derived from NSGA-II.
- **The 10 % is a floor, not a cap.** §5.6.5: "Rank 0 solutions are exempt from further elimination if their number of
  members falls below a designated proportion (e.g. %10) of the population"; the pseudocode stops eliminating Rank 0
  when "the Rank 0 preservation limit" is reached (Appendix B: 10 % at population 100). The thesis states no maximum.
  Berk's 2026 brief ("maximum x %") is therefore a **change** from part I, not an inheritance.
- **"Interactive" means a human decision maker**: pairwise preference questions (a Rank 0 or Rank 1 solution against the
  incumbent); in the experiments the DM is simulated by linear and Chebyshev utility functions. I-TREA II's reading —
  the solver as the counterpart — is a reinterpretation.
- **Lineage.** The thesis cites Ray et al. (2009), IDEA, as the source of its infeasibility-driven idea (§2.4 and
  references). IDEA is paywalled-only; cite its mechanism through `singh2011thesis` ([[A Constraint handling with infeasible solutions]]).

**Consistency with the vault.** [[D-004 Mapping I-TREA (2011) to I-TREA II]] already records the Rank 0 share as
"reserved **and capped** … brief overrides", Rank 1 as "brief overrides", and the solver's role as an "analogue" of the
decision maker. The thesis confirms those entries from a public source; it does not contradict them. One vault
statement was out of date: the journal entry of 2026-10-05 23:00–23:20 ("I-TREA (part I) is not findable online"). The
journal is append-only, so the entry stands; the same journal file corrects it at 00:1x on 10-06 ("Correction to
23:00–23:20").

## 2. Novelty record — what exists (no "first" claim)

1. **A GPU population beside a CPU branch-and-bound with two-way exchange already exists inside NVIDIA cuOpt**
   (`nvidia2026cuoptsrc`, read at commit c2f3df5): a GPU population with recombiners, its own CPU B&B as a concurrent
   OpenMP task, B&B solutions entering the population, and new best population solutions pushed to B&B (in
   opportunistic mode only). CHAP (`tjusila2026chap`) is a GPU+CPU heuristic portfolio with a shared pool, but
   standalone, not beside a B&B. The searches did **not** surface a GPU-resident EA running beside an **external**
   general MIP solver (SCIP or HiGHS) with two-way exchange. The HiGHS+cuOpt pairing (2025) has no published exchange
   mechanism; the optopt race was a race, not an exchange.
2. **Two-way exchange between a population or primal heuristics and B&B has clear prior art**: Puchinger, Raidl &
   Gruber 2005 and Nwana et al. 2005 ([[B Metaheuristics cooperating with MIP solvers]]); Gallardo, Cotta & Fernández
   2007 (interleaved MA and beam-search B&B, problem-specific); ReXi / concurrent SCIP 2026 (CPU portfolio with a shared
   pool and crossover LNS; currently the leading non-commercial MIPfeas solver, so a natural extra baseline); and cuOpt
   internally. One-way GA → B&B warm starts go back at least to Nieminen et al. 2003.
3. **Reserved infeasible shares exist in EAs**: IDEA (a fixed share of infeasible solutions, chosen by non-dominated
   sorting in objectives and violation), FI-2Pop (a separate infeasible population selected by violation), Vidal's HGS
   (penalty-steered feasible ratio, implicitly preserved elites). cuOpt keeps infeasible solutions but reweights them so
   the best infeasible ranks behind the best feasible (`adjust_weights_according_to_best_feasible`) — the **opposite**
   of a super-optimal reserve. CHAP pools infeasible candidates with no stated rank or cap rule. **No work was found that
   caps a share of the population for infeasible individuals whose objective beats the incumbent, for general MIP.**
   This is a negative search result, not proof of absence.
4. **LLM-driven evolution** (EoH, ReEvo, DHEvo, EvoCut, MILP-Evo) evolves code, cuts or solver components offline. None
   of them runs a solution population concurrently with the solver.

The combination I-TREA II tests — a GPU population, an external general MIP solver, two-way exchange, an
incumbent-relative capped infeasible rank and a protected top-y % feasible rank — was not found as a whole. Each part
has prior art (above and in sweeps A–C).

## 3. Negative searches (verbatim, the novelty record)

1. "Interactive Three Ranks Evolutionary Algorithm" (exact phrase, web): nothing. The thesis is not indexed under that phrase; it surfaced only through the METU IE M.S.-theses list and a search on its full title.
2. Berk Orbay master's thesis evolutionary algorithm: nothing relevant
3. "Orbay" "evolutionary algorithm" tez yüksek lisans "three ranks" OR "üç sıra" OR "üç rank" OR "I-TREA": nothing
4. "Berk Orbay" thesis Boğaziçi OR Sabancı OR "Istanbul Technical" OR Bilkent industrial engineering: nothing. These were the wrong universities; the M.S. is from METU/ODTÜ (his PhD is from Boğaziçi).
5. YÖK Ulusal Tez Merkezi: site-restricted search (tez.yok.gov.tr) for Orbay / the thesis title returned no tez.yok.gov.tr hit. A direct non-interactive POST author search on 'Berk Orbay' returned the error page 'Hata Oluştu'. The search form needs an interactive session, so YÖK could not be checked from here.
6. "super-optimal" infeasible solutions population evolutionary algorithm mixed integer: nothing using the concept (only generic MINLP constraint-handling DE papers)
7. evolutionary algorithm integer programming keep infeasible solutions with objective better than best feasible "super-optimal": nothing. No work found that caps a population share for infeasible individuals by the rule 'objective better than the best feasible / the incumbent'.
8. genetic algorithm "integer programming" population infeasible individuals objective better than incumbent reserved fraction "branch-and-bound" hybrid: nothing matching. Only one-way GA-to-B&B warm starts (Nieminen et al. 2003; French et al. 2001, paywalled).
9. GPU evolutionary algorithm running concurrently with MIP solver solution exchange branch and bound: nothing beyond GPU B&B papers and cuOpt itself
10. GPU parallel genetic algorithm integer linear programming population tensorized MILP arXiv: nothing for MILP. Only GPU genetic-programming and constrained multi-objective tensorized EA frameworks.
11. GPU genetic algorithm MIPLIB benchmark evaluation mixed integer CUDA population heuristic: nothing beyond cuOpt (arXiv:2510.20499) and CHAP
12. "cuOpt" evolutionary OR "genetic algorithm" SCIP OR HiGHS hybrid concurrent incumbent exchange: nothing. Only the HiGHS+cuOpt pairing announcement (Edinburgh blog, 2025-03-18, no exchange mechanism described) and cuGenOpt (standalone GPU metaheuristic, arXiv:2603.19163).
13. GPU evolutionary algorithm SCIP heuristic plugin concurrent "solution exchange" mixed-integer arXiv 2025 2026: nothing beyond CHAP, ReXi and cuOpt
14. genetic algorithm injects solutions into SCIP heuristic plugin incumbent exchange hybrid metaheuristic MIP solver parallel: nothing specific (surveys and CHAP only)
15. Optimization Online (domain-restricted): genetic algorithm branch-and-bound concurrent solution exchange mixed integer GPU population heuristic. No optimization-online.org hit on the topic.
16. OpenReview (domain-restricted): evolutionary algorithm MILP GPU population solver exchange incumbent. Nothing new beyond DHEvo, MILP-Evo and CHAP (both arXiv).

**Caveats on these searches.** The `allowed_domains` filter of the web search did not appear to be honoured, so the
Optimization Online and OpenReview entries mean "no relevant hit surfaced", not an exhaustive site search. The YÖK
national thesis database could not be queried non-interactively.

## 4. Table (16 works: 15 legally readable, 1 paywalled-only)

| key | authors | year | venue | link | access | what it does | relation to I-TREA II | verification |
|---|---|---|---|---|---|---|---|---|
| `orbay2011itrea` | Berk Orbay | 2011 | M.S. thesis, Operational Research, Graduate School of Natural and Applied Sciences, METU, Ankara; hdl:11511/20509 | [OpenMETU PDF](https://open.metu.edu.tr/bitstream/handle/11511/20509/index.pdf); [landing](https://open.metu.edu.tr/handle/11511/20509) | open | I-TREA (§5.1): an interactive multi-objective GA derived from NSGA-II for a bi-objective capacitated facility relocation problem with partial coverage (maximise weighted coverage, minimise the number of facilities). Three ranks: Rank 0 infeasible good, Rank 1 feasible good (the feasible non-dominated front), Rank 2 rejects; tournament Rank 1 > Rank 0 > Rank 2. Rank 0 is exempt from elimination once it falls to a preservation proportion (e.g. 10 %; App. B: 10 % at population 100) — a floor, not a stated maximum. Pairwise preference questions to a decision maker (simulated by linear and Chebyshev utilities); benchmarked against exact solutions and a modified NSGA-II on random instances. | Lineage: part I. Its Rank 0 is the multi-objective analogue of I-TREA II's Rank 0 (with one objective, "not dominated by any feasible solution" means "objective at least as good as the best feasible"); its 10 % was a floor, so the "at most x %" cap is a change; its Rank 1 was the whole front, not a top-y % elite; its "interactive" was a human DM, not a solver. | Full text read (the handle URL returned HTTP 500 twice on 2026-10-05; the landing page works) |
| `ray2009idea` | Tapabrata Ray, Hemant Kumar Singh, Amitay Isaacs, Warren Smith | 2009 | SCI 198, Springer, pp. 145–165; doi:10.1007/978-3-642-00619-7_7 | — | **paywalled** | IDEA keeps and evolves a small proportion of marginally infeasible solutions so the constraint boundary is approached from both sides (per abstract snippets and the I-TREA thesis summary; method details not read in the chapter). | Lineage for item 3: the source the I-TREA thesis cites for its reserved infeasible share. See [[A Constraint handling with infeasible solutions]]. | Metadata only |
| `nvidia2026cuoptsrc` | NVIDIA cuOpt developers | 2026 | GitHub NVIDIA/cuopt, Apache-2.0, commit c2f3df54af31d51c106791812ad1829ab0ca5caa (2026-10-05); repo created 2025-04-08 | [population.cu at commit](https://github.com/NVIDIA/cuopt/blob/c2f3df54af31d51c106791812ad1829ab0ca5caa/cpp/src/mip_heuristics/diversity/population.cu) | open (Apache-2.0) | GPU-resident population (`population.cu`, `diversity_manager.cu`); bandit-chosen recombiners (bound-prop, FP, line-segment, sub-MIP) and LS configurations. CPU B&B runs as a concurrent OpenMP task (`solver.cu`); B&B solutions enter via `add_external_solution(…, BRANCH_AND_BOUND)` (`population.cu` 146–177); a new best feasible is pushed to B&B via `branch_and_bound_callback(…, HEURISTICS)` (264–284), wired only in `CUOPT_MODE_OPPORTUNISTIC` (a TODO in deterministic mode, `solver.cu` ~449–463). User solutions enter the same queue (`EXTERNAL`); CPU-FJ solutions are dropped unless better than the best feasible. Slot 0 is the best feasible (line 84). Infeasible members are scored by objective + weighted violation; when the best-by-quality member is infeasible, `adjust_weights_according_to_best_feasible()` (363–395) raises infeasibility importance by gap·1.1 / weighted violation, leaving the best infeasible 10 % of the former quality gap behind the best feasible. | Competitor / closest prior for items 1–3: a GPU population beside a CPU B&B with two-way exchange exists; its infeasible rule is the opposite of a super-optimal Rank 0. (Sweep E key `nvidia2025cuopt` = this entry.) | Full text (source) read |
| `corduk2025gpuprimal` | Akif Çördük, Piotr Sielski, Alice Boucher, Kumar Aatish | 2025 | arXiv:2510.20499 [math.OC, cs.DC]; poster, NeurIPS 2025 ScaleOpt workshop | [arXiv:2510.20499](https://arxiv.org/abs/2510.20499) | open (preprint) | GPU PDLP as approximate LP solver, a probing cache for fast rounding and early infeasibility detection, accelerated FP, FJ and fix-and-propagate; 221 feasible solutions and a 22 % gap on presolved MIPLIB 2017. | Competitor: cuOpt's GPU heuristic stack and the optopt race partner. The population and recombiners are attributed to the repository, not to this paper. (Sweep E key `corduk2025gpuheur` = this entry.) | Abstract read here; full text read in sweep C |
| `tjusila2026chap` | Tjusila, Hoen, Kempke, Mexi, Berthold, Gleixner, Koch, Pokutta | 2026 | arXiv:2605.05086 [math.OC] | [arXiv:2605.05086](https://arxiv.org/abs/2605.05086) | open (preprint) | GPU+CPU heuristic portfolio with a three-part shared pool ("LP solutions generated by cuPDLPx, and FeasPump, feasible incumbent solutions, and infeasible solutions used to guide FPR and tabu search"); no crossover; Land-Doig 2026 set (50 instances, 5 min): CHAP 47, Gurobi 44, cuOpt heuristics-only 43 "solving" (the paper's verb). Xpress 9.8.0 as an LP solver after FPR fixings. | Competitor for items 1–2: shared incumbents and infeasible candidates, but standalone, no recombination, no cap or "better than incumbent" rule for the infeasible part. | Full text read |
| `mexi2026rexi` | Gioni Mexi, Daniel Rehfeldt | 2026 | arXiv:2609.05954 [math.OC, cs.DC] (10 pages) | [arXiv:2609.05954](https://arxiv.org/abs/2609.05954) | open (preprint) | Ten workers (eight diversified SCIP B&B, ReXiLS, ReXiLNS) sharing primal solutions and bounds, bound reductions and presolve information through a pointer-based pool in original space and 32 mutex-sharded bound arrays, no coordinator; LNS mutation and crossover over pooled solutions; fastest non-commercial solver on MIPFEAS (233 instances, 600 s); pre-release SCIP 11 / SoPlex 9, ≤ 10 threads, no GPU. | Competitor / strongest CPU analogue for item 2; candidate baseline arm beside SCIP and SCIP‖cuOpt. | Full text read |
| `gallardo2007mabnb` | José E. Gallardo, Carlos Cotta, Antonio J. Fernández | 2007 | IEEE Trans. Systems, Man, and Cybernetics, Part B 37(1):77–83; doi:10.1109/TSMCB.2006.883266 | [co-author's site (pre-publication draft)](https://carloscotta.com/papers/gallardo07hybridization.pdf) | open (author draft) | Interleaves a memetic algorithm with truncated B&B (beam search): after l₀ levels the MA population is initialised from the best queue nodes (B&B → EA); the MA's best replaces the B&B incumbent if better, letting B&B purge nodes (EA → B&B). Multidimensional 0-1 knapsack and shortest common supersequence. | Background for item 2: an early, explicitly two-way incumbent and population exchange — interleaved on one CPU with a problem-specific B&B, not concurrent with a general MIP solver. | Full text read (15-page draft, TeX 2006-03-14; the published version is 7 pages — do not quote draft pagination) |
| `nieminen2003gamilp` | Kimmo Nieminen, Sampo Ruuth, István Maros | 2003 | Departmental Technical Report 2003/4 (ISSN 1469-4174), Dept. of Computing, Imperial College London, April 2003 | [Imperial DoC PDF](https://www.doc.ic.ac.uk/research/technicalreports/2003/DTR03-4.pdf) | open | A GA for general MILP finds a good first integer solution used as B&B's initial incumbent. Genes are fixed incrementally in random order with dual-simplex re-solves; on LP infeasibility the gene takes the rounded relaxed value and the rest are rounded. Fitness: objective if feasible, else sum of infeasibilities, with a feasibility flag. Preliminary results only. | Background for item 2: the one-way, sequential GA → B&B warm start that I-TREA II turns into concurrent two-way exchange. | Full text read |
| `luteberget2023feasibility` | Bjørnar Luteberget, Giorgio Sartor | 2023 | Mathematical Programming Computation 15(2):365–388 (CC BY 4.0) | [Springer PDF](https://link.springer.com/content/pdf/10.1007/s12532-023-00234-8.pdf) | open (CC BY 4.0) | Feasibility Jump (see [[C GPU primal heuristics and GPU EC]]); per secondary sources CP-SAT's ViolationLS builds on it and cuOpt runs FJ on the GPU. | Technique we use: the batched FJ-style repair on the GPU population. (Sweep E key `luteberget2023fj` = this entry; the NTNU handle hdl:11250/3133165 lists the same VoR but its file returned 403.) | Full text read |
| `kimbrough2008fi2pop` | Steven Orla Kimbrough, Gary J. Koehler, Ming Lu, David Harlan Wood | 2008 | EJOR 190(2):310–327 | [Penn repository](https://repository.upenn.edu/server/api/core/bitstreams/a83719e3-9b32-4200-876f-8eb9da988225/content) | open (author manuscript) | FI-2Pop (see [[A Constraint handling with infeasible solutions]]): feasible and infeasible populations; infeasible ones selected on violation only and not evaluated on the objective. | Background for item 3: a permanent infeasible population, not selected by "objective better than the incumbent" and not capped as a share of one population. | Full text read (32-page manuscript) |
| `vidal2022hgs` | Thibaut Vidal | 2022 | Computers & Operations Research 140:105643; doi:10.1016/j.cor.2021.105643 | [arXiv:2012.10384](https://arxiv.org/abs/2012.10384) | open (preprint) | Open-source HGS for the CVRP with the SWAP* neighbourhood. Two subpopulations (feasible, infeasible), each between μ and μ+λ; survivors by biased fitness (quality rank + diversity contribution), weighted so the top n_Elite (= 4) are preserved; infeasibility penalties adapted towards a target feasible fraction ξ_ref; infeasible solutions repaired with probability 50 % at 10× penalties. | Background for item 3 and Rank 1: a managed infeasible share and implicitly preserved elites, for the CVRP; the share is steered by penalties, not by "better than the incumbent", and elitism follows from the fitness formula, not a reserved rank. | Full text read |
| `liu2024eoh` | Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, Qingfu Zhang | 2024 | ICML 2024, PMLR 235:32201–32223 | [PMLR PDF](https://proceedings.mlr.press/v235/liu24bs/liu24bs.pdf) | open | Evolution of Heuristics: an LLM inside an evolutionary loop evolves heuristics held as a natural-language "thought" plus code; combinatorial heuristics such as online bin packing. | Background for item 4: evolves code, not solutions; out of scope. | Abstract / landing read |
| `ye2024reevo` | Haoran Ye, Jiarui Wang, Zhiguang Cao, Federico Berto, Chuanbo Hua, Haeyeon Kim, Jinkyoo Park, Guojie Song | 2024 | NeurIPS 2024 (Advances in NeurIPS 37), doi:10.52202/079017-1381 | [arXiv:2402.01145](https://arxiv.org/abs/2402.01145); [proceedings page](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4ced59d480e07d290b6f29fc8798f195-Abstract-Conference.html) | open | ReEvo: LLM hyper-heuristics with reflective evolution (reflections as verbal gradients); five algorithm types, six combinatorial problems. | Background for item 4. | Abstract / landing read |
| `zhang2025dhevo` | Zhihao Zhang, Siyuan Li, Chenxi Li, Feifan Liu, Mengjing Chen, Kai Li, Tao Zhong, Bo An, Peng Liu | 2025 | arXiv:2507.15615 [cs.NE]; ICLR 2026 submission (decision not verified) | [arXiv:2507.15615](https://arxiv.org/abs/2507.15615) | open (preprint) | LLM-based multi-agent data–algorithm co-evolution: selects representative instances of a MILP class and evolves primal heuristics; the generated heuristics replace SCIP's default diving heuristics. | Background for item 4: offline evolution of MIP heuristics; no solution population runs beside the solver. | Full text (HTML) read |
| `yazdani2025evocut` | Milad Yazdani, Mahdi Mostajabdaveh, Samin Aref, Zirui Zhou | 2025 | arXiv:2508.11850 (v2 12 Feb 2026) | [arXiv:2508.11850](https://arxiv.org/abs/2508.11850) | open (preprint) | LLM agents create a population of candidate "acceleration cuts" at the model level, screen them on a small verification set (must keep feasible solutions, cut LP points) and refine them by evolutionary crossover and mutation; up to 76 % smaller gaps and up to 7.2× faster to target gaps. | Background for item 4: a population of cuts (dual side), not of solutions. | Abstract read |
| `nie2026milpevo` | Jinbiao Nie, Kewei Feng, Xiaoyuan Zhang, Shan Yin, Zizhuo Wang, Bin Dong | 2026 | arXiv:2607.18252 [cs.AI, cs.NE] (v1 submitted 12 May 2026) | [arXiv:2607.18252](https://arxiv.org/abs/2607.18252) | open (preprint) | LLM-guided closed-loop program evolution of white-box MILP solver components scored by end-to-end behaviour, through PySCIPOpt; applied to joint design of a cut selector and a branching rule. | Background for item 4. | Abstract read |

## 5. Seen but not given a row

- MIPfeas benchmark (Bussieck & Dirkse) — in [[D Benchmarks and metrics]].
- CP-SAT-LP invited talk (Perron, Didier, Gay, CP 2023) — in [[B Metaheuristics cooperating with MIP solvers]].
- Raidl & Puchinger hybrid-metaheuristic survey (TU Wien, `raidl-08.pdf` seen in results, not opened).
- cuGenOpt (arXiv:2603.19163; standalone GPU metaheuristic, no solver pairing).
- Spoorendonk 2026, *Presolve heuristics in HiGHS* — in [[C GPU primal heuristics and GPU EC]].
- A massively parallel GPU EA for partial Latin square extension (arXiv:2103.10453).
- Paywalled-only, not cited: French, Robinson & Wilson 2001; Rothberg 2007; IDEA — see [[Paywalled research]].
