---
tags: [literature, paywalled]
updated: 2026-10-05
---

# Paywalled research

> **Disclaimer.** The works below are behind publisher paywalls and **we have not read them in full** (except where a
> cell says how a method was checked). No legal free copy was found: no preprint, accepted manuscript, repository or
> author copy, thesis or earlier open version. They are listed because they matter for this research. What we say about
> each is limited to the abstract, metadata and the open works that describe it; the evidence column says how far
> that goes. Nothing here was obtained by bypassing a paywall, and no link to an unauthorised copy is given.

This is the **only** place for paywalled literature. The other notes, the paper body and the tutorials point here and
cite the open route in the "cite instead" column. If a legal free copy turns up later, the work moves into the topic
note and `paper/refs.bib` with the link. Sweeps and rules: [[Prior art]].

**Rank names:** Rank 0 = infeasible good, Rank 1 = feasible good, Rank 2 = rejects ([[D-004 Mapping I-TREA (2015) to I-TREA II]]).

## 1. Paywalled-only works from sweeps A–E (`paywalled_only = true`)

| key | work | year | DOI | why it matters | cite instead (open) | evidence | status / action |
|---|---|---|---|---|---|---|---|
| `runarsson2000stochastic` | Runarsson & Yao, *Stochastic ranking for constrained evolutionary optimization*, IEEE TEVC 4(3):284–294 | 2000 | 10.1109/4235.873238 | **technique the code uses**: `selection.comparator: stochastic` with `p_f` (an optional comparator, not the default) | `mezuramontes2011survey` (describes stochastic ranking) | Crossref; mechanism read in an archived copy of the dead author page (Wayback snapshot 20170810164525 of `cs.bham.ac.uk/~xin/papers/published_tec_sep00_constraint.pdf`, 11 pp., the IEEE version) | **needs Berk** — project rule "drop unless crucial, then discuss". Options: (a) cite only the open survey for the comparator; (b) cite the original with the archived author copy as the route, if Berk accepts an archive of a former author page. **Trap:** OpenAlex / Semantic Scholar list arXiv:1510.04409 and hal-01215727 as its "green OA" copy — both are Chotard, Auger & Hansen, a different paper. |
| `rothberg2007polishing` | Rothberg, *An Evolutionary Algorithm for Polishing Mixed Integer Programming Solutions*, INFORMS J. Computing 19(4):534–541 | 2007 | 10.1287/ijoc.1060.0189 | lineage of crossover / mutation as sub-MIPs over a solution pool | `berthold2006primal` (describes Rothberg's operators, §4.3–4.4); `mexi2026rexi` (calls its crossover "closely related to" Rothberg's EA, "known as solution polishing in CPLEX") | Metadata + abstract (OpenAlex); INFORMS 403 | cite via the open works. Two claims are **unverified** — do not state them: "this is CPLEX's solution polishing" (attribute to ReXi if used) and "IJOC Test of Time award". The 2005 ILOG technical report has no online copy. |
| `ray2009idea` | Ray, Singh, Isaacs, Smith, *Infeasibility Driven Evolutionary Algorithm for Constrained Optimization*, in Mezura-Montes (ed.), SCI 198, pp. 145–165 | 2009 | 10.1007/978-3-642-00619-7_7 | IDEA, the fixed infeasible share; the source the I-TREA thesis cites | `singh2011thesis` (Ch. 3, Alg. 3.1) | Metadata + Springer abstract | cite the thesis. Pages 145–165 (Crossref), not 147–167. |
| `singh2008idea` | Singh, Isaacs, Ray, Smith, *Infeasibility Driven Evolutionary Algorithm (IDEA) for Engineering Design Optimization*, AI 2008, LNAI 5360, pp. 104–115 | 2008 | 10.1007/978-3-540-89378-3_11 | the IDEA conference paper with the author order the brief cites | `singh2011thesis` | Abstract / landing read | cite the thesis |
| `isaacs2008blessings` | Isaacs, Ray, Smith, *Blessings of maintaining infeasible solutions for constrained multi-objective optimization problems*, IEEE CEC 2008, pp. 2780–2787 | 2008 | 10.1109/CEC.2008.4631171 | CHEA, IDEA's precursor (extra objective = number of violated constraints) | `singh2011thesis` (§2, §3.2.4) | Metadata + abstract | cite the thesis |
| `kimbrough2003twopop` | Kimbrough, Lu, Wood, Wu, *Exploring a Two-Population Genetic Algorithm*, GECCO 2003, LNCS 2723, pp. 1148–1159 | 2003 | 10.1007/3-540-45105-6_123 | early FI-2Pop | `kimbrough2008fi2pop` (open author manuscript) | Crossref; abstract of an archived 6-page copy of a dead author page (the proceedings paper has 12 pages — not used) | cite the 2008 paper |
| `takahama2006epsilon` | Takahama & Sakai, *Constrained Optimization by the ε Constrained Differential Evolution with Gradient-Based Mutation and Feasible Elites*, IEEE CEC 2006, pp. 1–8 | 2006 | 10.1109/CEC.2006.1688283 | the ε-constrained method; a possible comparator variant | `mezuramontes2011survey` §3.3; `fan2019iepsilon` | Crossref. Method details were checked against a copy on the CEC 2006 organiser's GitHub that carries an IEEE institutional-licence stamp — **not a legal open copy; never link it** | cite the open routes. Its "feasible elites" are lowest-violation points, not a Rank 1 analogue. |
| `danna2005rins` | Danna, Rothberg, Le Pape, *Exploring relaxation induced neighborhoods to improve MIP solutions*, Math. Programming 102(1):71–90 | 2005 | 10.1007/s10107-004-0518-7 | RINS and guided dives: solver-side neighbourhoods that use injected incumbents | `berthold2006primal` (implements and describes RINS in SCIP); `hendel2022alns` | Abstract / landing read | cite via the open works |
| `french2001hybrid` | French, Robinson, Wilson, *Using a Hybrid Genetic-Algorithm/Branch and Bound Approach to Solve Feasibility and Optimization Integer Programming Problems*, J. Heuristics 7(6):551–564 | 2001 | 10.1023/A:1011921025322 | GA ↔ B&B hybrid with solver-to-population seeding (sequential) | `puchinger2005combining` (describes it) | Abstract / landing read; mechanism from the survey | cite the survey. Academia.edu / ResearchGate copies exist but are not legal-venue copies. |
| `fischetti2018matheuristics` | Fischetti & Fischetti, *Matheuristics*, in *Handbook of Heuristics*, Springer, pp. 121–153 | 2018 | 10.1007/978-3-319-07124-4_14 | matheuristics survey | `boschetti2022matheuristics` (CC BY 4.0) | Metadata + abstract (DTU Orbit) | cite the open survey; editor names unverified |
| `boschetti2009matheuristics` | Boschetti, Maniezzo, Roffilli, Bolufé Röhler, *Matheuristics: Optimization, Simulation and Control*, HM 2009, LNCS 5818, pp. 171–177 | 2009 | 10.1007/978-3-642-04918-7_13 | definitional matheuristics reference | `boschetti2022matheuristics` | Metadata only | cite the open successor |
| `davies2024violationls` | Davies, Didier, Perron, *ViolationLS: Constraint-Based Local Search in CP-SAT*, CPAIOR 2024, LNCS 14742, pp. 243–258 | 2024 | 10.1007/978-3-031-60597-0_16 | Feasibility Jump generalised and run as one portfolio worker | `perron2023cpsat` (CP-SAT portfolio with FJ-based local search) | Metadata only (abstract from search snippets) | do not cite in the main text unless an open copy appears |
| `lodi2013variability` | Lodi & Tramontani, *Performance Variability in Mixed-Integer Programming*, INFORMS TutORials in OR, pp. 1–12 | 2013 | 10.1287/educ.2013.0112 | the standard account of MIP performance variability | `fischetti2014erraticism`; `gleixner2021miplib` | Metadata only | **open-access status unverified, not confirmed paywalled**: Unpaywall reports a free publisher-hosted copy (bronze); every INFORMS URL returned 403 from this host. Open https://pubsonline.informs.org/doi/pdf/10.1287/educ.2013.0112 in a normal browser; if it loads without a login, move it out of this note. |

## 2. Closed versions mentioned inside other entries (metadata only)

These are separate publications named in an entry whose citable version is open (or in an entry above). Recorded so
that no one cites them by accident.

| work | DOI | mentioned in | note |
|---|---|---|---|
| Ben Hamida & Schoenauer, *ASCHEA: new results using adaptive segregational constraint handling*, IEEE CEC 2002, pp. 884–889 | 10.1109/CEC.2002.1007042 | `hamida2000aschea` | Crossref confirmed, closed; cite the PPSN 2000 paper (HAL) |
| Takahama & Sakai, ε-level control for PSO (ε-PSO), Advances in Soft Computing, pp. 1019–1029 | 10.1007/3-540-32391-0_105 | `takahama2006epsilon` | Crossref confirmed; closed per OpenAlex |
| Singh, Isaacs, Nguyen, Ray, Yao, dynamic IDEA, IEEE CEC 2009 | 10.1109/CEC.2009.4983339 | sweep A negative searches | Birmingham portal has metadata only |
| Lin, Cai, Zou, Lin, *Local-MIP: Efficient local search for mixed integer programming*, Artificial Intelligence 348 (2025) 104405 | 10.1016/j.artint.2025.104405 | `lin2024localmip` | journal version with a different author list; closed (Unpaywall); cite the CP 2024 LIPIcs version |
| Rothberg, *An Evolutionary Algorithm for Polishing Mixed Integer Programming Solutions*, ILOG technical report, 2005 | — | `rothberg2007polishing` | cited by Berthold 2006 (ref. [52]); no online copy found |

## 3. Seen in the sweeps, closed, not entered (content not read — do not cite without checking)

| work | DOI / id | sweep | why it was not entered |
|---|---|---|---|
| Liang et al., *Constrained multiobjective differential evolution algorithm with infeasible-proportion control mechanism*, Knowledge-Based Systems 250:109105 (2022) | 10.1016/j.knosys.2022.109105 | A | relevant by title (controls the infeasible proportion) but ScienceDirect 403, no abstract; content unverified |
| Chu & Beasley, GA for the generalised assignment problem (fitness / unfitness), Computers & OR 24:17–23 (1997) | 10.1016/S0305-0548(96)00032-9 | A | closed per OpenAlex; not read |
| Geng et al., *Infeasible Elitists and Stochastic Ranking Selection in Constrained Evolutionary Multi-objective Optimization*, SEAL 2006, LNCS | 10.1007/11903697_43 | A | Crossref only; not opened |
| Cotta & Troya 2003 (B&B embedded in an EA) | — | B | closed; not read |
| Munguía et al. 2018, Alternating Criteria Search (parallel LNS) | — | B | closed; not read |
| Fischetti et al. 2016, random-sampling racing (Math. Prog. Computation) | — | B | closed; not read |
| Denzinger & Offermann 1999, TECHS (GA + B&B agents exchanging solutions) | — | B | known only through `puchinger2005combining`; not verified |
| Janssen et al., GPU island-model GA, Concurrency and Computation: Practice and Experience (2022) | 10.1002/cpe.6286 | C | Wiley paywalled; no open copy verified |
| Koch et al., *MIPLIB 2010*, Mathematical Programming Computation 3:103–163 (2011); ZIB-Report 10-31 | — | D | Semantic Scholar: closed; the ZIB-Report link is behind a bot check from this host; open copy not verified (MIPLIB 2017 is used instead) |

## 4. Not found — do not cite

- **No work in sweeps A–E has `exists = false`.** Every listed work was confirmed to exist (Crossref, DataCite, arXiv,
  publisher or repository pages).
- **Mislinks, not works:** arXiv:1510.04409 and hal-01215727 are *not* Runarsson & Yao (2000) — see section 1.
- **Not found online:** the 2005 ILOG technical report version of Rothberg's polishing EA; an open copy of any IDEA
  paper other than Singh's thesis; an open copy of the Takahama–Sakai ε papers; an arXiv preprint of CCMO (Tian et al.,
  IEEE TEVC 25(1), 2021); a separate GPU Feasibility Jump paper beyond the cuOpt paper and PaNGEA; a cuOpt MIP
  technical report beyond arXiv:2510.20499 (the MIP 2025 talk has an abstract only); the primary source that first
  defined the "primal-dual integral" (only Berthold 2013's §4 variant and the ML4CO 2022 definition were found).
- **Not available from this host:** the IBM CPLEX solution-polishing documentation page (empty JavaScript shell) and the
  SCIP `heur_crossover` doxygen page (HTTP 429) — not cited; `berthold2006primal` is used instead.

## 5. Part I's own references

The unpublished manuscript's reference list is audited separately in the part-I references (12 reachable,
28 not; BibTeX in `paper/refs-itrea1.bib`, which also lists closed works). None of those closed works is listed here
yet; any of them that the part-II paper cites is added to section 1 first, with its open route or a "needs Berk" flag.
Two links between the audits:
- **IDEA (Ray et al. 2009; `ray2009infeasibility` in `refs-itrea1.bib` = `ray2009idea` here)** — that note flags it as
  "crucial and closed … unless an open IDEA paper turns up (not searched yet)". Sweep A searched: no open copy of an IDEA paper
  was found, but Singh's 2011 PhD thesis (`singh2011thesis`, UNSWorks, open) gives IDEA in full (Ch. 3, Alg. 3.1), so the
  mechanism can be cited without the closed chapter. Whether the chapter itself must also be cited is still Berk's call.
- **The I-TREA master's thesis** (`orbay2011itrea`, METU 2011) is open — see [[E Novelty check 2026-10-05]].
