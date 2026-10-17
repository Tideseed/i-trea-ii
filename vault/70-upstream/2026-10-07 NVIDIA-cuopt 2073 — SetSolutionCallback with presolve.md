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

## 2026-10-13 — fixed by PR #2098 (merged 2026-10-12, in 26.10); verified on the 26.10 nightly
Berk (2026-10-13): "If the nightly for the next version is up can you check if our feature request in Github issues is
implemented properly?" Issue closed 2026-10-12 06:09 UTC by **NVIDIA/cuopt#2098** "Keep presolve enabled for MIP
set-solution callbacks" (akifcorduk; approved by aliceb-nv and ramakrishnap-nv), merged into `main` = 26.10.00 (the
maintainer had expected 26.12). What it does (diff read): presolve stays as configured when a SET callback is
registered; Papilo's **dual reductions are turned off** while one is registered (so feasible user points stay
representable); an injected vector in ORIGINAL variable order is checked against the original model (bounds,
integrality, every row, with cuOpt's tolerances), crushed into the presolved space, re-checked, and queued only if
feasible, with its objective recomputed; GET callbacks keep returning original-space vectors; a SET callback that
offers nothing now skips (`continue`) instead of ending the callback loop. C++ tests cover presolve on/off ×
min/max × semi-continuous and three invalid offers; the Python callback tests are skipped upstream (cuopt#1909).

**Our probe** (`tools/probe_cuopt_setsol.py`, `tools/probe_cuopt_setsol_report.py`; nightly
`cuopt-cu13 == libcuopt-cu13 == 26.10.0a294.post261013051018` in a separate venv `~/venvs/cuopt-2610n`, installed on
Berk's ask; 26.08 = the repository venv; 20-s solves, 3 threads, RMM 8 GB, a short lease, LLM servers left up): five
development instances × {no SET callback, a SET callback that never offers, inject our best checked solution, inject
infeasible variants} on 26.10, the injection on 26.08, and a glass4 point that beats the proven optimum.
- **Presolve with a SET callback: on in 26.10** on all five (glass4 322 → 317 variables, sing44 59,708 → 51,461,
  loue 3,255 → 3,250, crna → 308; item_placement unchanged); **26.08: "Presolve is disabled because set_solution
  callbacks are provided."** The cost of dual reductions off: the presolved size is identical with and without a
  (never-offering) SET callback on 4 of 5; crna keeps 3 more variables (308 vs 305).
- **Injected solutions accepted, echoed exactly, valid:** glass4 (the optimum 1,200,012,599.999), loue (29,699 vs
  cuOpt's own 58,784), item_placement (4.7388 vs 33.15; cuOpt then improved it to 4.73877), crna (the optimum
  1.810748), sing44 (8,171,973.25) — each reported back by the GET callback 1–6 ms after the offer, the echoed vector
  equal to the offered one (max |Δ| ≤ 1e-12; sing44 9.1e-13), feasible by our exact checker (crna and sing44 from
  the rerun with offers from the first call, `/tmp/setsol/out2`).
- **Infeasible offers rejected:** a fractional integer and an all-ones shift (glass4, item_placement, loue): 0 reported.
  The glass4 point below MIPLIB's proven optimum (1,200,006,662.6, which our own checker's tolerance had accepted in a
  development run) is rejected by cuOpt — correct.
- **Limitation (pre-existing, not changed by #2098): cuOpt polls SET callbacks only when its GPU heuristics' population
  records a new best feasible solution** (`population_t::run_solution_callbacks`, called from `add_solution` on an
  improvement, for initial solutions and in `generate_quick_feasible_solution`; source at 1e0c261). Incumbents found by
  other components (CPU FJ, branch-and-bound) are reported to GET callbacks without polling SET. Measured on 26.10:
  crna — 11 SET calls, all at 0.70–0.71 s, none in the remaining 29 s; sing44 — the first SET call at 28.3 s, after 639
  GET reports. So an outside solution enters only while cuOpt's own heuristics are improving — not when cuOpt has
  stalled, which is when a partner's solution would help most (A9, D-015).
- **Verdict:** the feature request is implemented properly — presolve on, original-space injection crushed correctly,
  invalid offers rejected. A follow-up request (poll SET callbacks on a timer or at B&B events, so a stalled solver can
  still take a partner's solution) is drafted for Berk's yes; nothing posted.
- **2026-10-13 11:xx, Berk:** "I just got word it is integrated to 26.10" — the fix ships in cuOpt 26.10 (our check:
  `main` = 26.10.00, every nightly since the merge is a 26.10 build, no 26.12 nightlies yet). Drafts of the follow-up
  stay unposted (no yes).
