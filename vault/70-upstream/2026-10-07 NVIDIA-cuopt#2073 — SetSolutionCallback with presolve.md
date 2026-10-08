# DRAFT — NVIDIA/cuopt feature request: SetSolutionCallback with presolve enabled

**Status:** FILED 2026-10-07 as **NVIDIA/cuopt#2073** (https://github.com/NVIDIA/cuopt/issues/2073) from Berk's
`gh` account (`berkorbay`), on his yes ("yeah go ahead with your recommendation"), stating that Claude Code wrote it.
Filed text = the draft below reflowed for GitHub, with two phrases softened to what is known (not measured by us):
"presolve is a large part of cuOpt's strength … cannot inject" → "Turning presolve off changes the solver we are
trying to help, so in practice we do not inject"; "which since #1104 survives presolve" → "which #1104 made
compatible with Papilo presolve". No comment was added to #186.

**Prior art checked (2026-10-07, `gh issue list -R NVIDIA/cuopt --state all` + `gh search issues`):**
- #482 "[FEA] Support get/set solution callbacks with presolve" — closed 2026-02-10: "get side is DONE / set side is
  tracked separately"; no public tracker for the set side was found.
- #513 "Restore initial user-provided solution support w/ papilo presolve" and #1060 "Convert solution in original
  space to solution in presolve space with Papilo" — closed by PR #1104 "Papilo primal/dual crush" (merged
  2026-07-02, so in v26.08.00 of 2026-08-06): MIP incumbents can now be crushed into Papilo's space for initial
  solutions.
- #186 "[FEA] cuOpt should exit the solve gracefully on SIGINT" — open.
- Installed cuOpt 26.08 (`solver_settings.pyx:272`): "Registering a SetSolutionCallback disables presolve."

---

**Title:** [FEA] Keep presolve enabled when a SetSolutionCallback is registered (crush injected solutions as #1104 does for initial solutions)

**Is your feature request related to a problem? Please describe.**
We run cuOpt 26.08 (Python API, MILP) concurrently with another primal heuristic — a GPU evolutionary algorithm — and
want cuOpt to use better incumbents that the other method finds *during* the solve. The only mid-solve path,
`SetSolutionCallback`, disables presolve ("Registering a SetSolutionCallback disables presolve", docstring of
`SolverSettings.set_mip_callback`). On the hard instances where outside solutions matter most, presolve is a large
part of cuOpt's strength, so in practice we cannot inject. The workaround is to stop cuOpt and restart it with the
improved solution as an initial solution (`set_initial_primal_solution`, which since #1104 survives presolve — thank
you), but a restart discards the search state and repeats presolve and the root LP; and since a solve cannot be
stopped early (#186), a restart means slicing the time limit or killing the process.

**Describe the solution you'd like**
Keep presolve on when a `SetSolutionCallback` is registered, and crush each injected solution into the presolved
space the same way #1104 now crushes initial solutions. If some Papilo reductions make that impossible for arbitrary
injected points, disabling only those reductions when a set callback is registered would still be much better than
disabling presolve entirely.

**Describe alternatives you've considered**
- Restarting with `set_initial_primal_solution` (works since #1104), at the cost of the search state.
- A graceful early stop (#186), which would at least make restarts cleaner.

**Additional context**
Prior art we found: #482 (closed — "set side is tracked separately"; we could not find that tracker, apologies if this
duplicates it), #513 / #1060 / #1104 (initial-solution crush), #186. This issue was drafted by Claude Code (Anthropic's
AI coding assistant) for Berk Orbay, as part of a research project on coupling GPU heuristics with MIP solvers.

## Edit 2026-10-07 (Berk: "you may also edit the issue to add how scip solves this problem as an example")
Section added before "Additional context" — "How SCIP handles this (for reference)": SCIP accepts original-space
solutions mid-solve with presolve on — `SCIPcreateOrigSol` (scip_sol.h L237, v10.0.0) + `SCIPtrySol` from a heuristic;
`SCIPtrySol` (scip_sol.c L4047) does not crush: "SCIPprimalTrySol() can only be called on transformed solutions;
therefore check solutions in original problem" → `SCIPsolCheckOrig` then `SCIPprimalAddSol`; the objective becomes
the primal bound / cutoff even where the point cannot be mapped into the presolved space. Our evidence: glass4, 227 of
228 posted EA solutions accepted, 213 became SCIP's incumbent (Development log, baselines). Suggested cuOpt
equivalent: accept original-space solutions mid-solve, check on the original model, use the objective as
incumbent/cutoff, crush into the presolved space when possible (#1104).

## Response (seen 2026-10-08)
NVIDIA labelled the issue **improvement, P1**; Akif Çördük (cuOpt, author of the GPU primal-heuristics paper we cite,
arXiv:2510.20499) on 2026-10-07 14:29 UTC: "Thank you @berkorbay for reporting that issue. I think it is doable, we are
close to the release so it might not make it to 26.10 but we can prioritize this for 26.12." Releases at that time:
v26.08.00 is the latest (installed here); the tag v26.10.00a marks the 26.10 development branch (RAPIDS convention), not
a release. Our arm A9 (one-way cuOpt → EA) does not depend on the feature; when it ships, mid-run injection into cuOpt
(EA → cuOpt) becomes possible without restarts. Installing a new cuOpt is a download → Berk's word first.
