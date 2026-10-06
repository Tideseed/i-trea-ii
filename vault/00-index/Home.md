# I-TREA II vault — home

**I-TREA II — Interactive Three Ranks Evolutionary Algorithm, part II.** A GPU-resident evolutionary algorithm with
three ranks (Rank 0 infeasible good — infeasible solutions that beat the incumbent, at most x % of the population ·
Rank 1 feasible good — the protected best y % of the feasible · Rank 2 rejects) running beside an exact MIP solver
(SCIP), exchanging solutions both ways, on hard MIP instances. **Main claim under test** ([[D-007 Main claim — the
coupling beats either method alone]]): the coupled system beats **both** the solver alone **and** the EA alone — each
comparison must succeed.
Successor of Berk Orbay's master's research I-TREA (Orbay 2011, METU M.S. thesis, open access, hdl:11511/20509; 2015
manuscript — the starting point and the idea of keeping good infeasible solutions) and of the optopt study
(`berkorbay/optopt`, whose CPU version of this idea found no gain at 60 s — see [[Research question and spec]]).

**State (2026-10-06 17:00):** development on the 9-instance dev set (D-010), judged on the coupled arm (D-011). Read the newest checkpoint.

| start here | |
|---|---|
| `90-checkpoints/` | session checkpoints — **read the newest first** ([[2026-10-06-1700]]) |
| [[Roadmap]] | the aim in Berk's words, the stages, and the GitHub issues of each — **what is next** |
| [[Research question and spec]] | Berk's brief and main claim, the predecessor's null result, the three ranks from I-TREA (2015) to I-TREA II |
| [[D-002 Pre-registration — hypotheses, arms, analysis]] | v3: primary H1a (A3 < A0, SCIP alone) **and** H1b (A3 < A5, EA alone); secondary H2 (uncoupled portfolio, live A3r), H3 (SCIP ‖ cuOpt), H4 (SCIP ‖ SCIP), H5 (tuned SCIP); mechanism H6 (A5 vs A5x0); exploratory A4, A5y0, A6; metrics, tests, phases — written before any run |
| [[I-TREA II algorithm]] | the algorithm, every parameter |
| [[Process interface]] | how a pair run is assembled: processes, mailbox, barrier, files |
| [[Trace format and metrics]] | every file, event and counter a run writes, the derived tables, every metric |
| [[Experiment index]] | every experiment with its status and outcome, failures included |

| section | contents |
|---|---|
| `00-index/` | this page, [[Research question and spec]] |
| `10-design/` | [[I-TREA II algorithm]], [[Feasibility jump local search]], [[Process interface]], [[Trace format and metrics]], [[Hard instance pool]], [[Datasets]], [[Compute resources]] |
| `20-journal/` | dated research journal, append-only, times in CEST |
| `30-results/` | [[Experiment index]] — every experiment with its outcome, failures included; one note per experiment |
| `40-literature/` | [[Prior art]] (synthesis of sweeps A–E: 91 works, 78 legally readable) with the sweep notes in `prior-art/` ([[A Constraint handling with infeasible solutions]], [[B Metaheuristics cooperating with MIP solvers]], [[C GPU primal heuristics and GPU EC]], [[D Benchmarks and metrics]], [[E Novelty check 2026-10-05]]); [[Paywalled research]] (the only home for paywalled works); part I: the 2011 thesis (https://open.metu.edu.tr/handle/11511/20509) (unpublished; the open source is the 2011 thesis, `orbay2011itrea`), the part-I references and `I-TREA (2015) references.bib` (an audit record, not citable as a whole) |
| `50-decisions/` | [[D-001 Language, solvers and process model]], [[D-002 Pre-registration — hypotheses, arms, analysis]], [[D-003 Resources, memory and thermal rules]], [[D-004 Mapping I-TREA (2015) to I-TREA II]] (a record of where the defaults came from), [[D-005 The frontier of diverse solutions]], [[D-006 Exchange platform — Redis, file mailbox as fallback]], [[D-007 Main claim — the coupling beats either method alone]], D-008 … D-011, [[D-012 Repair clock — adaptive repair depth, staged]]; a superseding decision says what changed and why |
| `60-incidents/` | [[Mistakes and incidents]] |
| `70-upstream/` | drafts of upstream issues — **filed only with Berk's per-post yes** |
| `80-reviews/` | [[2026-10-05 design critique]] (three lenses, 53 findings) and [[2026-10-05 design critique — response]] (status of each) |
| `90-checkpoints/` | session checkpoints |

Tracking: GitHub issues on `Tideseed/i-trea-ii` (labels `phase:E0` … `phase:E5`, `infra`, `paper`, `needs-berk`).

Rules carried from the host (DGX Spark): public data only; no downloads without Berk's ask; claim `box-window` before
any timed run; memory gate with a 20 GB reserve and the CuPy pool limit as the only GPU cap (12 GB for the EA and for
cuOpt); thermal gate; production serving is restored by the session that stopped it; nothing is filed upstream or
submitted (MIPLIB, arXiv) without Berk's explicit yes; no personal data (e-mail addresses) to outside services; never
`pkill -f` from an agent shell.
