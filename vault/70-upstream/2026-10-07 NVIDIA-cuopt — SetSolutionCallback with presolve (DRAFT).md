# DRAFT — NVIDIA/cuopt feature request: SetSolutionCallback with presolve enabled

**Status:** draft, NOT filed. Berk suggested it (2026-10-07: "we might add an issue to cuopt repo to want such
feature"); filing needs his explicit yes for this post (host rule: per-post permission). Would be filed from his `gh`
account (`berkorbay`), stating that Claude Code wrote it.

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
