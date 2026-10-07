# 2026-10-07 cuOpt restart design (workflow)

**What:** Berk (2026-10-07): "for cuopt we may restart the solver with the new best solution". A design workflow
(Claude Code subagents, read-only on the repository, no GPU; cuOpt's source read on GitHub NVIDIA/cuopt, Apache-2.0):
facts → two designs (time-sliced restarts; kill-on-improvement) → an adversarial judge (checked the source) → a plan.
**Outcome:** in-process time-sliced restarts with a rule (restart at a quarter boundary only if the EA's best beats
cuOpt's held best by 0.1 %), arms A6 / A7 / A7s / A8 with a matched restart-cost control; a ~17.5-min first GPU test
with a go/no-go rule decides whether to build. Becomes **D-015** when built (D-014 is the class probe). Note: the plan's
clause "commits only on Berk's OK" predates Berk's standing permission for this repository (2026-10-06) and does not
apply. Upstream request filed meanwhile: NVIDIA/cuopt#2073.

## Plan (verbatim)

PLAN: cuOpt restarted from the EA's best (D-015)

DECISION
- Build A8's in-process, no-kill slicing with the judge's combined rule; not K2.
- Record as D-015 (exploratory), amending D-001's "launch-time warm start only" for the new arms only; A2, A6, H2 unchanged.
- Nothing enters the repository before the first GPU test passes; commits only on Berk's OK.

MECHANISM
- One cuopt_run process; parse and warm-up before the barrier, then repeated Solves on one DataModel, on the clock.
- Boundaries at T/4, T/2, 3T/4 (30/60/90 s at 120 s). Slices are boundary-anchored, so overruns never shift the schedule; a boundary leaving < 5 s is skipped.
- Slice 0 has no start and always ends at T/4 (one wasted restart on rail01 and the guard instances).
- Rule at each boundary, identical in A7s and A8: if the EA's newest to_solver row (cached in-process) beats cuOpt's held best by 0.1 % relative (--restart-min-gain 1e-3), the next slice ends at the next boundary; otherwise it runs to T, no further restarts. A7s reads the posts for this decision only, so A8 and A7s differ in one variable: the start vector (EA's row vs own best).
- Start: map to local order, round integers, clip to [lb, ub], refuse non-finite; log max row violation. Slice seed = seed + 7919k. Never set mip_determinism_mode.
- Echo: first improving callback with obj ≤ start + 1e-6·max(1,|start|) and integer part differing in ≤ h = 3 positions (strict: h = 0). Recorded as source "ea", posted ORIGIN_EA_ECHO; verdict accepted=True on echo, never accepted=False. A "cuOpt find" = checked incumbent differing in > h integer positions.
- Best dual bound carried across slices. Stop file and SIGTERM honoured at boundaries.
- Per slice: start, echo latency, status, nodes, overrun, MemAvailable, RSS, GPU memory, rmm_current.

ARMS AND CONTROLS (EA on X925 cores 5-6, cuOpt on A725 1-3; EA: lp1, use_root_lp and lp_stream off, tol_mode highs, to_solver on everywhere)
- A6 cuOpt alone (baseline); A7 one-way stream, unsliced; A7s rule-sliced, own start; A8 rule-sliced, EA start.
- Contrasts: A8 vs A7s: EA vector's value; A8 vs A7: adoption; A7s vs A7: matched restart cost; A7 vs A6: stream plus contention. (A6s, fixed quarters, own start: upper bound on restart cost; first test only.)
- D-013: W_cuOpt from A6 seed 0 (sing44, loue, rail01, ca, item_placement, neos-960392); evaluate on seeds 1–2 with A6 re-run. Primary: share improved, mean P, mean d with bootstrap interval, Wilcoxon, worst P. Guard over all 9: mean d ≥ −0.05. Secondary: cuOpt-held P ± echoes; finds after EA starts.
- Pilot: 48 runs (~2 h), Williams blocks; guard instances next; descriptive.

CHANGES
- Regression: --slices none keeps today's single-Solve path and settings.
- New pure src/itrea/solvers/cuopt_slices.py: schedule, rule, start preparation, echo matcher, bound carrier, seeds.
- cuopt_run.py: --slices {none,quarters,at:...}, --start-from {own,best}, --recv, --restart-min-gain, --echo-max-int-diff; per-slice summary; SIGTERM docstring corrected.
- pair.py: ProcSpec send/recv/slices; arms A7, A7s, A8; EA on lane.x925; hash includes schedule and policy; budget_for counts 2 CUDA processes, cuOpt at its 12 GB limit (gate unchanged).
- build_runs.py: cuOpt-held P ± echoes, finds, restart columns; verdicts from role cuopt. make_jobs.py: dev-cuopt-restart.
- Docs: D-015, Process interface, Development log, AGENTS.md, issue #23.
- CPU-testable via a fake cuopt module on toy MPS: single-Solve regression, run-to-end branch, echo strict/loose, no false verdicts, monotone bound, stop at boundary, cached post. GPU-only: toy knapsack end to end.

FIRST GPU TEST (~17.5 min; i-trea-ii lease, after the queued pool, < 80 °C, one GPU process, taskset 1-3)
- Step 0 (~1 min): verify_mipstart.py V0–V7, glass4.
- Step 1 (~1 min): SIGKILL probe; admits a later kill variant only if reap ≤ 5 s, no D-state, MemAvailable back ≤ 5 s.
- Step 2 (~15 min): new scratch chain_probe.py, seed 1. Per instance: shared slice 0 (30 s), own chain (= A6s) and EA chain, 3 × 30 s each; same-seed A6 (120 s) on loue and sing44 only. EA rows at each boundary = EA best at ≤ b in A3@lp1 (seed 0 loue/item_placement, seed 1 sing44).
- Step 3 (CPU, taskset 10-14): check_solutions on saved vectors; only checked incumbents count.
- DEAD: V1 shows the start unused, "Presolve is disabled", or an in-bounds start errors.
- GO (build, pilot): on ≥ 2 of 3, the EA start is echoed (loose) in ≥ 2 of 3 EA slices and ≥ 1 checked find beats the EA's recorded best at that moment.
- A7 ONLY: otherwise.
- Pre-declared switch: A6s worse than A6 by > 0.10 P on both loue and sing44, or median echo latency > 15 s → pilot uses one boundary at T/2.
- Cannot show contention, closed loop, statistics, or the run-to-end branch (EA leads on all three; CPU tests cover it); EA rows are SCIP-coupled (optimistic).

RISKS
- Dead time = PaPILO ≤ 0.1L + presolve ≤ 0.1L + uncapped B&B set-up (ca, 305k columns) + fixed-integer LP ≤ min(60, remaining/2): up to 15 s of a 30-s slice.
- Sliced presolve-heavy instances get no bound; neos-960392's 99.9-s proof lost (EA holds the optimum). rail01 likely 1.0 vs 0.927.
- cuOpt may only echo or polish; then polish belongs in the EA.
- Contention (optopt A4 doubled P(30)) may cancel gains.
- Dual reductions can cut off starts; one start per Solve drops cuOpt's own best.
- Pilot needs the serving window; thermal gate 80/92 °C; small n, prereg-v2 before confirmatory use.

## Judge (verbatim)

RECOMMENDATION: combine the two designs. Build A8's in-process, no-kill mechanism and its control arms, with the changes below. Do not build K2.

Checked in the cuOpt source and our code, and correct in both designs:
- Python passes one start per Solve (solver_wrapper.pyx:185, 218).
- Only a SetSolutionCallback turns presolve off (solve.cu:455-466).
- A start containing NaN is dropped (presolve_data.cu:32).
- Deterministic mode ignores starts (diversity_manager.cu:461-481).
- plan() pins both the EA and cuOpt to cores 1-3 (pair.py:493-497).
- A verdict with accepted=False demotes the individual and reverts the EA's incumbent (engine.py:1899-1924).

WRONG IN BOTH DESIGNS
1. The start does not wait for the root LP. add_user_given_solutions runs at diversity_manager.cu:520, before the heuristic root LP at :558. Dead time per restart is:
   - PaPILO presolve, at most 0.1·L;
   - cuOpt's own presolve, at most 0.1·L;
   - B&B setup (probing implied bounds, solver.cu:~340), which has no time cap and matters on ca's 305k columns;
   - the LP with integers fixed, at most min(60, remaining/2) (:244).
2. The echo that comes back through our callback has been mapped back through PaPILO's postsolve (population.cu:267-270). Columns that presolve removed or clamped come back as postsolve sets them. So an exact match on the integer part fails whenever presolve touched the start, and the EA's vector gets credited to cuOpt. Better rule: the first improving callback of the slice whose objective is at most the start's plus a tolerance, and whose integer part differs from the start's in at most a few positions. Report both strict and loose counts.
3. D-014 is already taken (Class probe). Use the next free number.

K2 PROBLEMS
- On loue, item_placement and sing44 the EA always leads, so condition (a) is always true and (c) opens as soon as the start is echoed. K2 restarts the moment each start lands, and cuOpt barely searches. It would need a minimum search time after the echo, at least as long as the previous dead time.
- The EA, its HiGHS LP subprocess (seeding.py:343 inherits the EA's cores), cuOpt's 3 threads and the standby's start-up all share A725 cores 1-3. The design doesn't count this CPU contention, so it confounds C1 against A6.
- The pilot would SIGKILL running CUDA processes dozens of times, which has never been done here. NVIDIA warns that killed workers can hang in uninterruptible D state. A worker left behind would make later runs void as foreign GPU use.
- The standby is a third CUDA process, plus a 1.4-s GPU warm-up beside the active solve after every kill. About 74 GB is needed, so both LLM servers must be stopped.
- The yoked arms cannot be randomised. The box heats up under sustained load, so the arm that runs later runs hotter.
- What K2 does better: a kill skips the end-of-Solve teardown, and cuOpt restarts only when the EA leads.

A8 PROBLEMS
- Every boundary pays the end-of-Solve teardown (ca overran its limit by 4.69 s) plus a fresh presolve. On ca a 30-s slice may be half dead time.
- Restarts happen even when there is nothing to pass in:
  - rail01: A8 itself predicts 1.0 against cuOpt alone's 0.927;
  - guard instances: glass4's 617k-node search tree is thrown away every 30 s; radiationm also restarts for nothing;
  - fhnw doesn't matter (cuOpt proves it optimal at 3.4 s).
- 30-s slices cap presolve at 3+3 s on sing44, neos-960392, rail01 and ca. At 120 s these spent 23-26 s in presolve and got their first bound at 72-78 s. Sliced, they never get a bound, and cuOpt becomes a restarted heuristic.
- Its first test (replay_restart.py, restart at 30 s, 90 s remaining) runs one 90-s Solve. That is K2's pattern, not the 30-s chain A8 would build. Its "own start vs cuOpt alone's continuation" also compares seed 1 with seed 0, one run each.

THE COMBINED DESIGN
1. One boundary rule for both A8 and its own-start control A7s. Slice 0 is 30 s. At each boundary, if the EA's best doesn't beat cuOpt's own by the gain threshold, the next slice runs to the end of the budget. That means one wasted restart on rail01 and the guard instances instead of three. The rule is the same in both arms (not the actual restart times), so the Williams randomisation stays.
2. Keep exchange.to_solver on in every EA arm, and have cuOpt ignore the posts in A7s, so posting cost is the same in all arms.
3. Use the echo match rule above.
4. Per slice, log MemAvailable, process RSS and GPU memory, not only rmm_current, which sees RMM allocations only.
5. Keep the EA on X925 cores 5-6, as A8 proposes.

CHANGES TO THE FIRST TEST
- Run the real chain: three 30-s Solves from 30 s for each start variant, with the echo deadline measured within the 30-s slice.
- Add cuOpt alone (A6) and its sliced version (A6s) on one shared seed, to measure restart cost (about 4 min per instance).
- Save the callback vectors and run our independent solution checker on the CPU; both scripts currently score objectives nobody has checked.
- Add K2's one-minute SIGKILL probe in a scratch process, to admit or rule out a later kill-based variant.

WHAT THE FIRST TEST CANNOT SHOW
- GPU contention: no EA runs beside cuOpt. In optopt A4, GPU load beside cuOpt doubled its P(30); this is the most likely thing to cancel any gain.
- How the closed loop behaves, or anything statistical: 1 seed, 3 instances, and EA vectors taken from a SCIP-coupled run as a stand-in.
- verify_mipstart runs on glass4 (322 variables, 0.4-s presolve). It can show the start path doesn't work, but it can't show presolve cutting off starts at the scale of ca or sing44.

This review is saved at /tmp/itrea-cuopt/judgement.txt.

## Facts (verbatim)

**mip_start.** Sources. The installed package is exactly the v26.08.00 release commit 400863c1a92914bd47f5d68245f9a00e2d0d2df4 (V/cuopt/GIT_COMMIT; GitHub commit message "REL v26.08.00 release", 2026-08-06). In the citations below, V = /home/coulson/git_repositories/i-trea-ii/.venv/lib/python3.12/site-packages and R = https://github.com/NVIDIA/cuopt/blob/400863c1a92914bd47f5d68245f9a00e2d0d2df4. The fetched sources are saved in /tmp/itrea-cuopt/src_*.

**API**
- Python: `DataModel.set_initial_primal_solution(x)` (V/cuopt/linear_programming/data_model/data_model.py:528).
  - For a MIP, `set_solver_setting` passes that one vector to `add_initial_mip_solution` (V/.../solver/solver_wrapper.pyx:218-222).
  - The C++ settings object is rebuilt on every Solve (pyx:185). So Python gives exactly **one** MIP start per Solve.
  - The start stays on the DataModel across later Solve calls until it is replaced. An empty array clears it (data_model_wrapper.pyx:449).
- Modelling API: `Variable.MIPStart` defaults to NaN (problem.py:109-130). `Problem` passes the whole NaN-filled vector if any entry is set (problem.py:1583, 1627-1628).
- C API: `cuOptAddMIPStart` can be called repeatedly, so it accepts several starts (cuopt_c.h:931-947; R/cpp/src/pdlp/cuopt_c.cpp#L1108-L1124). The symbol is exported (`nm -D libcuopt.so`), so ctypes could reach it, but only with C-API problem and settings handles, not the Python objects.

**Checked?** Yes, strictly.
- `check_initial_solution_representation` runs at solve.cu:378 (R/cpp/src/pdlp/utilities/problem_checking.cu#L55-L89).
- The length must equal n, and every value must lie within [lb-1e-8, ub+1e-8]. Otherwise it raises ValidationError "Initial solution violates variable bounds." and **the whole solve fails** (the start is not just skipped). Whether Python sees this as `error_status` or as a raised exception is unverified.
- NaN passes this check, because the comparisons are false.

**Presolve: not disabled, and the start is crushed through it.**
- Only a SetSolutionCallback turns presolve off: "Presolve is disabled because set_solution callbacks are provided." (R/cpp/src/mip_heuristics/solve.cu#L455-L469).
- Starts stay in the original space. After PaPILO presolve, cuOpt's own presolve and probing, and B&B set-up, `diversity_manager_t::add_user_given_solutions` handles them (R/cpp/src/mip_heuristics/diversity/diversity_manager.cu#L182-L276, called at L520). For each start it:
  1. crushes it with PaPILO (`crush_primal_solution`, third_party_presolve.cpp#L1249-L1285): replays parallel-column merges, projects onto the surviving columns, then std::clamps to the reduced bounds;
  2. maps it through cuOpt's trivial presolve (`pre_process_assignment`);
  3. solves an LP with the integer variables fixed: time limit min(60, remaining/2), `return_first_feasible` (L244-L249). It keeps the LP point if feasible, otherwise the raw crushed vector;
  4. calls `population.run_solution_callbacks` (R/.../diversity/population.cu#L297-L319). If the start is feasible and better than the best feasible so far, it goes to B&B as incumbent and upper bound (`set_solution_from_heuristics`, R/cpp/src/branch_and_bound/branch_and_bound.cpp#L514-L570; an infeasible one goes to B&B's repair queue) **and** to our GetSolutionCallback (uncrushed, original space);
  5. adds it to the GPU population even when it is infeasible.
- In short, the start is checked, crushed, "repaired" only by the clamp and the fixed-integer LP over continuous variables, and used as incumbent if feasible.
- Upstream: PR #1104 "Papilo primal/dual crush" (merged 2026-07-02; closes #513 and #1060) and PR #1568 "Clamp crushed solutions to dual-reduced bounds" (2026-07-14). Release notes 26.08 (docs/cuopt/source/release-notes.md lines 29 and 48): "Papilo-based primal/dual crush in MIP presolve" and "Fix a bug in MIP in variable fixing for initial solutions".
- Stale documentation:
  - cuopt_c.h:944 says "@attention Currently unsupported with presolve on". The code has no such check (cuopt_c.cpp#L1108-L1124).
  - FAQ "Does cuOpt take an initial solution? Currently, cuOpt doesn't" (faq.rst:290-292) sits in the Routing FAQ, not the MIP one.

**Caveats**
- PaPILO's dual reductions can cut off a start that is feasible in the original space. The code comment says so for early-FJ incumbents (solve.cu ~L487-L491), and those get an end-of-solve original-space fallback (solve.cu ~L700). User starts get no such fallback (from reading; unverified). So cuOpt's returned solution may be worse than the start. Our EA records its own incumbent anyway.
- **Partial starts are not supported.** `pre_process_assignment` discards any vector containing NaN (R/cpp/src/mip_heuristics/problem/presolve_data.cu#L32-L35); the log then says "Error cannot add the provided initial solution!" and the start is skipped. A NaN only on columns PaPILO removed might survive; unverified. A start of the wrong length is either a ValidationError or a logged skip ("Initial solution %zu has %zu vars, expected %d; skipping").
- The start takes effect only after presolve. PaPILO's limit is min(0.1·T, 60 s) (solve.cu#L565-L566); cuOpt's presolve uses the same ratio (solver.cu#L212-L222). Early FJ during presolve ignores the start. If early FJ already found something better, there is no echo.
- Deterministic mode (mip_determinism_mode=1): the deterministic branch of `run_solver` never calls `add_user_given_solutions` (diversity_manager.cu#L461-L481), so the start appears to be ignored. Unverified at runtime.
- `mip_heuristics_only`: the start is used (the L520 path).

**Unverified, and how to check in under 2 minutes on the GPU**
- Script: /tmp/itrea-cuopt/verify_mipstart.py. It is written and syntax-checked, **not run**.
- Run it inside an i-trea-ii box-window lease, below 80 °C:
  `cd /home/coulson/git_repositories/i-trea-ii && OMP_NUM_THREADS=1 taskset -c 1-3 .venv/bin/python /tmp/itrea-cuopt/verify_mipstart.py`
- Cost: one process, 9 Solves × 3 s, about 6 GB RMM peak.
- It uses glass4. cuOpt alone reaches 2.1000173e9 by 3 s and 1.6000131e9 at 120 s (traces/raw/dev/glass4__A6__s0__120s__6e35f122f659). The script passes the 120-s vector as the start with a 3-s limit, so a result ≤ 1.6000131e9 plus an echo with the start's objective proves the start was used.
- Variants:

| variant | what it checks |
|---|---|
| V0 | control, no start |
| V1 | start used; presolve kept (`log_file` shows "Papilo presolve time" and no "Presolve is disabled") |
| V2 | in-process restart overhead |
| V3 | partial (NaN) start |
| V4 | out-of-bounds start (error status or exception) |
| V5 | SIGTERM deferral |
| V6 | start with heuristics only |
| V7 | start in deterministic mode |

- The script prints a VERDICT line at the end.

**stopping.** The only way to end a running cuOpt 26.08 MIP solve early is to kill the process. The paths, cheapest first:

**1. `time_limit`** — the only graceful stop.
- It is set before Solve and cannot change during the solve.
- The timer starts before PaPILO, so presolve counts against it (solve.cu#L375; issue #1135).
- It is checked only periodically (mip-settings.rst:19-22, "may run slightly over"). Measured overrun on the dev set (traces/raw/dev/*__A6__*/cuopt.summary.json `overrun_s`): 0 to 0.41 s, and 4.69 s on ca__Scenario400.
- Nothing is lost: the solution, status, bound and stats all come back.
- Restarts built on it happen only at restart points fixed in advance (time slices).

**2. `node_limit`** — counts B&B nodes only (solver.cu#L349; milp-features.rst:92).
- On 5 of 9 dev instances cuOpt had nodes = 0 at 120 s (sing44, rail01, neos-960392, ca, fhnw), so it never fires there.
- It cannot be triggered from outside.

**3. `work_limit`** — passed to B&B only in deterministic mode; opportunistic mode sets it to infinity (solver.cu#L361-L365).

**4. Callbacks** — cannot stop a solve.
- Both callback types return void (internals.hpp:46-68). The C++ trampoline ignores the Python return value and any exception: a NULL result is just not DECREF'd (callbacks_implems.hpp:30-52).
- SetSolutionCallback also disables presolve.
- `concurrent_halt` is a C++-only `std::atomic<int>*` on the PDLP settings (pdlp/solver_settings.hpp:340), internal to the LP solve.
- There is no interrupt or cancel function in the Python, C (cuopt_c.h) or C++ API.

**5. Signals** — do not shorten a solve.
- Solve runs `with nogil` (solver_wrapper.pyx:467-471). Python-level handlers therefore run only after Solve returns: our SIGTERM flag (common.py:295-298) and KeyboardInterrupt on SIGINT.
- Issue #186 (open, 2025-07-03): "cuOpt ignores SIGINT ... SIGTERM results in an immediate termination" — that describes the default disposition.
- The docstring at cuopt_run.py:29 ("both are checked only when an incumbent arrives") overstates this. `proc.sigterm` stays False until Solve returns; inside the callback only the stop file is seen, and it only writes a note (cuopt_run.py:203-205).
- So the runner's SIGTERM does not shorten the solve; pair.py escalates to SIGKILL 10 s later (pair.py:984-1000).
- A Python watcher thread can run during Solve, because the GIL is released, but it can end the solve only by killing the whole process.

**6. Process kill (SIGKILL)** — the only immediate stop, and NVIDIA's own mechanism.
- For CancelJob on a running job, the gRPC server SIGKILLs the worker and spawns a replacement:
  - R/cpp/docs/grpc-server-architecture.md lines 296-317: "mid-solve workers do not poll the shutdown flag"; "the worker process is killed with SIGKILL ... a replacement worker is spawned";
  - docs/cuopt/source/cuopt-grpc/grpc-server-architecture.md:30;
  - the binary string "[Server] Cancelling running job %s (killing worker %d)";
  - release note "gRPC: cancel active jobs on delete".
- Contradiction: faq.rst:191-193 says a running solve "runs to completion; there is no mid-solve cancellation path". The code and the architecture documentation side with the kill, so the FAQ looks stale.
- What a kill loses: everything except what the callback already wrote — the final solution, the final bound and the statistics. Our crash-safe `.part` file plus `recover_incumbents` (common.py:396) keeps the incumbent vectors.
- Memory: the CUDA context is destroyed and its memory is reclaimed at process exit (GPU memory is system RAM on GB10). The RMM LimitingResourceAdaptor is per process, so nothing leaks across processes.
- NVIDIA warns that "Mid-CUDA workers can sit in uninterruptible D-state after SIGKILL, so waitpid is bounded" (cpp/docs/grpc-server-architecture.md:310). Such a straggler holds its memory until it exits, so the memory gate must count it.
- cuopt_run starts no child processes. OpenMP/B&B threads die with it, and pair.py's killpg covers the group.
- Untested here: all 9 i-trea-ii A6 runs exited 0, and no run.json shows a cuOpt kill.
- A replacement process pays about 2.5 s of start-up (imports 1.05 s, parse 0.01–1.7 s, warm-up 1.4 s), on the clock if it starts after t0.

**7. gRPC server route** (libcuopt/bin/cuopt_grpc_server with `cuopt.grpc.linear_programming.Client`).
- It offers submit, cancel, delete and an incumbent stream; returning False from the client's incumbent callback cancels the job (grpc_client.pyx ~L610).
- Underneath it is the same SIGKILL plus a replacement worker, and `result()` raises for a cancelled job.
- It adds a server, fork and pipes. Whether a replacement worker pre-creates its CUDA context is unverified.

**How to verify:** V5 in /tmp/itrea-cuopt/verify_mipstart.py sends SIGTERM to itself 1 s into a 3-s solve; the expected Solve return and handler time are about 3 s. A SIGKILL test is not included; the only exercise of that path would be a kill in a scratch process, watching `ps -o stat` for D-state and `MemAvailable` recovery.

**restart_cost.** **What a restart repeats**

In-process (same DataModel), each Solve:
- creates a new CUDA stream and raft handle, and copies the problem host-to-device again (R/cpp/src/pdlp/utilities/cython_solve.cu#L95-L116);
- re-runs the validation checks and MIP scaling;
- restarts early CPU and GPU feasibility jump from scratch;
- re-runs PaPILO presolve, limited to min(0.1·R, 60 s) where R is the remaining time (solve.cu#L565-L566);
- re-runs cuOpt's presolve (bounds propagation and probing cache), with the same ratio and cap (solver.cu#L212-L222);
- repeats trivial presolve and B&B set-up (problem conversion, probing implied bounds, clique table);
- re-solves the root LP (concurrent PDLP / barrier / dual simplex, limit min(15 s, 0.1·R); diversity_manager.cu#L500), then the root cut loop;
- rebuilds the population from scratch. Only then does the start enter (L520).

The parse and the CUDA context are **not** repeated in-process, and no warm-up is needed. `rmm_current_bytes` is 0 after warm-up (glass4 A6 summary), so nothing accumulates between Solves.

A restart in a new process adds about 2.5 s, on the clock:

| | dev-set A6 summaries |
|---|---|
| import + RMM | 1.04–1.07 s |
| parse (`read_s`) | 0.01–1.7 s |
| warm-up | 1.39–1.44 s |

The time limit scales with R, so a late restart pays less presolve but may get a time-truncated presolve. Restarting with the same `random_seed` repeats the same early trajectory (solve.cu#L372), so vary the seed per restart.

**Measured on the dev set** (A6, 120 s; traces/raw/dev/*__A6__s0__120s__6e35f122f659)

| instance | presolve_s | first B&B bound (s) | first incumbent (s) | nodes at 120 s |
|---|---|---|---|---|
| glass4 | 0.4 | 15.6 | <1 | 617,104 |
| item_placement | 0.6 | 6.1 | <1 | 67,238 |
| loue | 1.4 | 12.0 | <1 | 15,119 |
| fhnw | 2.0 | none | <1 | 0 |
| radiationm | 3.2 | none | <1 | 48 |
| sing44 | 23.5 | 72.3 | <1 | 0 |
| neos-960392 | 23.6 | 77.7 | <1 | 0 |
| rail01 | 24.1 | none | 107.5 | 0 |
| ca | 26.2 | 77.5 | 24.1 | 0 |

- 23–26 s is about 2 × 12 s, which suggests both presolve phases hit their 0.1 × 120 s caps. Unverified; confirm with "Papilo presolve time" in a `log_file`.
- So on presolve-heavy instances a restart re-spends up to about 0.2·R before the start can matter, plus up to min(15, 0.1·R) on the root LP. A restart after about 40 s on sing44, neos-960392 or ca would likely produce **no** new cuOpt dual bound before 120 s.

**Search state lost**
- the B&B tree, node queue, pseudocosts and reliability-branching statistics;
- the cuts and cut pool;
- the root LP solution and basis (no LP warm start inside a MIP);
- the probing cache, implications, clique table and symmetry data;
- the 32-member GPU population, the bandit recombiner statistics, FJ weights and RINS state;
- all incumbents except the one vector passed back (Python allows one);
- **the dual bound** — a restarted cuOpt reports none until its new root finishes. A bound proven earlier is still valid, so the runner should carry the best bound forward; the EA uses it (`use_dual_bound`, Rank 0 elimination).

**Estimate method**
1. Without the GPU, from existing traces: restart dead time ≈ presolve_s(R) + root LP + time to the start's echo, and lost-bound time ≈ first-bound time. Replay each A6 trace with a hypothetical restart at τ, using the EA's best at τ as the incumbent from τ + dead time, to estimate the P(T) effect.
2. On the GPU: V2 of /tmp/itrea-cuopt/verify_mipstart.py measures the in-process restart overhead on glass4 (Solve call to first callback). For a real instance, run one Solve with `log_file` and the instance's 120-s best as start at R ∈ {60, 120}, and record the echo time (dead time), "Papilo presolve time" and the root LP time. This needs its own lease and is about 1–2 min per instance.

The presolve and root-LP limits are exposed as parameters `mip_hyper_heuristic_presolve_time_ratio`, `_presolve_max_time`, `_root_lp_time_ratio` and `_root_lp_max_time` (names confirmed in the libcuopt strings; heuristics_hyper_params.hpp:17-27). Changing them would cap restart cost, but only in a coupled arm, since it changes the as-shipped cuOpt.

**our_code.** **Today**

cuopt_run.py:
- runs a single Solve (L214) with time_limit = remaining (L209);
- its GetSolutionCallback `Recorder` (L162-205) records every improving incumbent as source "self";
- with `--send` it posts each one to `to_ea` with ORIGIN_SOLVER, plus `info` (incumbent, dual_bound, status) (L191-201, final post L232);
- it never reads `to_solver`; `base_parser(..., exchange=False)` (L65, L74) means there is no `--recv`.

pair.py:
- `_CUOPT = ProcSpec("cuopt", "cuopt")` (L136) has exchange=False. It is used by A2 (L145), A6 (L152) and H2 (L154).
- `plan()` adds only `--cpu-threads` and `--gpu-mem-gb` for cuOpt (L493-496). `solver_flags` (L427) applies only to SCIP and HiGHS (L475).
- So **no arm passes `--send` today**. cuOpt incumbents reach the EA only if an arm adds `--send`; the EA then reads `to_ea` and `info` as it does from SCIP.

EA to solver:
- The engine posts an improving incumbent to `to_solver` when `exchange.enabled and to_solver` (engine.py:863), if its gain over the partner's incumbent (`inc_external`) exceeds `post_min_gain` (L1502-1506, `_post` L1526-1535). Frontier posts are optional (L1688).
- Message format: X (k, n) in canonical order, obj, origin bits.
- SCIP and HiGHS poll `mb.to_solver.poll()` (scip_run.py:281, highs_run.py:176) and write `mailbox.verdict`.

**Changes for "restart cuOpt with the EA's best"**

(1) cuopt_run:
- Add `--recv`, a restart policy (`--slice-s`, `--restart-min-gain`, `--max-restarts`) and `--mip-start FILE`. Call `proc.map_order()` when `--recv` is set.
- Parse once and reuse the DataModel. Loop: `Solve(min(slice, remaining))`, then poll `to_solver` (latest message wins) and take the best row.
- Prepare the row: `namemap.to_local`, **`np.clip` to [lb, ub]** (any value outside ±1e-8 fails the whole Solve with a ValidationError, and the process exits 1), round the integer columns, drop NaN or non-finite rows.
- If the row beats `proc.best_obj` by more than the gain threshold, `dm.set_initial_primal_solution(x)`. Otherwise pass cuOpt's own best, or every slice restarts from nothing.
- Use seed + k for slice k. Carry the best dual bound across slices and post it in `info`. Write one `mailbox.verdict` per EA row tried. Record per-slice timings and statistics in the summary.

(2) Echo provenance:
- After a restart, cuOpt reports the start back through the callback (population.cu#L297-L319). Its objective can differ, because the fixed-integer LP re-solves the continuous variables, so `vector_hash` (common.py:141, which rounds to 1e-6) may miss it.
- Match on the integer part (or the objective) and record it with source "ea" (`SOURCE_CODES` "ea": 1, common.py:63). With `--send`, post it with ORIGIN_EA_ECHO (common.py:57). Otherwise cuOpt gets credit for the EA's solution.

(3) Event-driven restarts (kill mid-slice) need a supervisor:
- (a) pair.py relaunches the cuopt role after t0 with `--mip-start` and the remaining budget, reusing t0 without a barrier. A killed slice must count as `stopped_by_runner`, as partner_end does (L917-921); `recover_incumbents` applies to the killed slice, and the summaries must be merged.
- (b) cuopt_run becomes a parent that forks a Solve worker per slice and owns the trace; incumbents come back over a pipe.
- (c) An optional pre-warmed standby worker hides the ~2.5 s start-up. It holds a CUDA context only, but adds one CUDA process to `budget_for` (L518-529).
- Our SIGKILL path for cuOpt (L897-903) has never fired in i-trea-ii.

(4) pair.py:
- New `ProcSpec("cuopt", "cuopt", exchange=True)` and new arms, leaving A2, A6 and H2 untouched: for example cuOpt ⇄ EA with restarts, plus a control that restarts cuOpt from its **own** best, to separate restart cost from EA value.
- Extend `solver_flags`/`plan` so they pass `--recv`/`--send` for cuOpt.
- Both processes would run on GPU_LANE (L189). They would share the A725 cores (1, 2, 3), where cuOpt uses 3 threads beside the EA, so the CPU footprint differs from A6.
- Memory: the EA pool (12 GB) + cuOpt RMM (12 GB limit; measured peak 5.6–7.7 GB) + 1 GB per CUDA process — `budget_for` already sums these. Bandwidth contention (optopt A4) still applies.

(5) Three-way SCIP ⇄ EA ← cuOpt: `to_ea` and `info` each take one writer — the file mailbox is latest-wins (mailbox.py docstring), and the EA reads only the newest Redis stream entry. A second writer would overwrite SCIP's posts, so cuOpt needs its own channel, such as `to_ea2` and `info2`.

(6) Configuration and documentation:
- The EA needs `exchange.to_solver` on.
- Correct the SIGTERM sentence in the cuopt_run.py docstring (L29).
- The new arm needs a D-0xx decision and a prereg amendment; D-001's "launch-time warm start only" stays true for A2/A6.
- The 70-upstream draft's claim that a start survives presolve "since #1104" is right. The draft cites #186 for "a solve cannot be stopped early"; #186 is about SIGINT handling (the solve has no mid-solve interrupt at all — see stopping).

## Designs (verbatim)

### A8: cuOpt ⇄ EA by boundary-anchored time slices. Each cuOpt slice starts from the better of its own best and the EA's best. No kills; a restart-cost-matched control family (A6s, A7, A7s).

**mechanism.** PROCESS STRUCTURE. The pair is the same as today under pair.py: one `cuopt_run` process (role `cuopt`) and one EA process (role `ea`) sharing the file or Redis mailbox.
- cuOpt keeps A6's exact footprint: GPU, A725 cores 1-3, num_cpu_threads 3, RMM limit 12 GB.
- The EA (CuPy pool 12 GB) moves to the lane's idle X925 pair (5, 6). Today `plan()` pins both `ea` and `cuopt` kinds to `lane.a725`, so in a two-partner arm they would share cores 1-3 and change cuOpt's footprint against A6.
- cuOpt parses once and warms up before the barrier, as today. After t0 it calls Solve repeatedly on the same DataModel, in the same process: no new CUDA context, no re-parse, no warm-up.

WHEN cuOpt RESTARTS. Boundaries are fixed in advance and identical in every arm.
- General form: a boundary list `at:b1,...` in seconds after t0. Default `fixed:4` gives 30/60/90 s at B = 120 and 75/150/225 s at B = 300.
- The list form lets any later adaptive policy be replayed as a yoked control with one flag.
- Boundary-anchored: slice k gets time_limit = min(t0 + b_{k+1} − now, remaining). An overrun (measured up to 0.41 s, 4.69 s on ca) shortens the next slice and never shifts the schedule.
- A boundary that would leave less than 5 s is skipped. The last slice ends at the budget.
- A slice returning Optimal, Infeasible or Unbounded ends cuOpt with that exit reason, and the runner stops the EA as today. TimeLimit or FeasibleFound starts the next slice.
- The stop file and SIGTERM are now honoured at every boundary. Today they are seen only after Solve returns.
- Random seed of slice k = seed + 7919·k in every arm.

WHICH SOLUTION IT STARTS FROM.
- Slice 0 has no start (`set_initial_primal_solution(empty array)` clears it). It is A6 except for its shorter time_limit.
- At boundary k the candidates are:
  - cuOpt's own held best (`proc.final_x()`);
  - the EA's best: the newest `to_solver` row, best by objective, mapped `to_local`. It is **cached in-process**, because `Channel.poll()` returns None once a message has been consumed, so a post read at an earlier boundary would otherwise be lost.
- Policy `own` (controls) passes the own best. Policy `best` (A8) passes the EA vector if `improves(ea_obj, own_obj, rel=1e-6)`, otherwise the own best.
- If neither exists, the start is cleared (rail01: cuOpt's first incumbent comes at 107.5 s).
- Preparation: round the integer columns, `np.clip` to [lb, ub] (outside ±1e-8 fails the whole Solve with a ValidationError), refuse non-finite values. The start's max absolute row violation, computed by cuopt_run from the DataModel's CSR, is logged per slice.
- Only a GetSolutionCallback is ever registered, never a SetSolutionCallback. So per 26.08 (PRs #1104 and #1568) PaPILO presolve stays on and the start is crushed through it. V1 of the first test confirms this.

WHAT FLOWS.
- cuOpt → EA (`--send`, in A7, A7s and A8):
  - every improving cuOpt incumbent goes to `to_ea` with ORIGIN_SOLVER;
  - `info` carries {incumbent = what cuOpt holds, dual_bound = the best bound carried across slices}.
  - Each slice restarts its bound from nothing, so a `bound` event is emitted only when the carried bound improves. bound_steps and the EA's dual_key never see a regression.
- EA → cuOpt (A8 only): the EA's improving incumbents (`post_min_gain` over cuOpt's known incumbent), read only at boundaries.
- Echo: cuOpt reports the start back after presolve and its fixed-integer LP.
  - It is matched on the rounded **integer part** only. The LP re-solves the continuous part and the objective, so `vector_hash` would miss it.
  - It is recorded with source `ea` and posted with ORIGIN_EA_ECHO, which the EA drops. obj_echo − obj_start is logged; it also checks that the HiGHS and cuOpt parses agree on the objective.
- Verdicts from cuopt_run are **accepted=True on an echo, or nothing — never accepted=False**. `Engine._verdicts` demotes the individual and reverts the EA's incumbent on a rejection. A missing echo can mean the start was cut off by dual reductions, beaten by early FJ, or failed cuOpt's tolerance; none of these means "infeasible".

TIME BUDGET. Everything after t0 is on the clock. Nothing is subtracted; it is measured, and the control arms assign it.
- What a restart costs: the poll, preparation and settings gap; the Solve's presolve; the fixed-integer LP for the start; the root LP; the echo latency.
  - Each of PaPILO and cuOpt's presolve is capped at min(0.1·L, 60 s), where L is the slice's time_limit.
  - The fixed-integer LP is capped at min(60, remaining/2).
  - The root LP is capped at min(15, 0.1·L).
  - v1 keeps the as-shipped hyperparameters.
- cuopt.summary.json `slices` holds one record per slice: k, t_call, limit, seed, start_src/obj/gain/seq/viol, gap_s, t_first_cb, t_echo, echo_obj_delta, n_cb, n_improving, status, presolve_s, nodes, final_dual, wall, overrun, rmm_peak, rmm_current.
- Totals: restart_gap_s, dead_s, starts {none, own, ea}, echoes.

**arms.** CONTRASTS FIRST (bounded d, D-002 machinery; positive = the second arm named is worse):
1. **A8 − A7s: Berk's proposal at a fixed restart cost.** The single variable is the vector passed at boundaries where the EA's best beats cuOpt's own. The number of such boundaries is logged; a run with none is A7s by construction. Reported on portfolio P and on cuOpt-role P.
2. **A8 − A7: the adoption question.** Restart coupling against simply streaming cuOpt into the EA with cuOpt unsliced (issue #23 option (i)). Restarts are kept only if A8 wins here.
3. **A6s − A6: pure restart cost**, on cuOpt only (cuOpt P). It is neither credited to nor blamed on the EA.
4. **Contention under slicing:** cuOpt-role P in A7s against A6s. In A7s cuOpt receives nothing; only the GPU and bandwidth sharing with the EA differ.
5. **D-013 headline:** A8 (portfolio P) against A6 on W_cuOpt, with the guard over all nine. The same table for A7, so the two coupling options can be compared on D-013 terms.

ARMS (all on the partner lane; one Williams block per instance × seed holds all five, so drift and contention are shared):
- **A6** (exists): cuOpt alone, one Solve. The D-013 baseline B.
- **A6s**: cuOpt alone, `fixed:4` slices, policy `own`.
- **A7**: cuOpt (one Solve, `--send`) ‖ EA, one-way cuOpt → EA, portfolio scored.
- **A7s**: A7 with A6s's slices and own-best starts. cuOpt never reads `to_solver`; the EA has `exchange.to_solver` false.
- **A8**: A7s, with each slice starting from the better of own best and the EA's best (`--recv --start-from best`, EA `to_solver` on).
- EA configuration for A7, A7s and A8: lp1, i.e. default + `seeding.lp_source: highs_async`. Arm-implied overrides:
  - `exchange.use_root_lp: false` (cuOpt never posts a root LP);
  - `lp_stream.enabled: false` (needs SCIP);
  - `ranks.tol_mode: highs` (absolute; cuOpt's MIP absolute tolerance is 1e-6, relative 1e-12);
  - one-way arms add `exchange.to_solver: false` and `post_frontier: false`, as `configs/ablations/one_way_solver_to_ea.yaml` already does.

D-013 REPORTING:
- **W_cuOpt comes from A6 seed 0.** That is the only A6 seed that exists, so it is the selection seed (P ≥ 0.5): sing44 0.726, neos-3024952-loue 0.760, rail01 0.927, ca__Scenario400_reserves_1 0.643, item_placement_9903 0.666, neos-960392 0.571 (6 of 9).
- **Every arm, A6 included, runs on seeds 1 and 2.** A6 must be re-run on those seeds; seed 0 is never used for evaluation.
- Primary, on W:
  - share improved (d > 0.05);
  - mean P, A6 → arm;
  - mean d with a 95 % bootstrap interval;
  - Wilcoxon p;
  - worst case after coupling (max P).
- Guard, on all 9 (glass4, radiationm18-12-05 and fhnw-binpack4-48 added): mean d ≥ −0.05 against A6.
- Sensitivity: cuOpt's worst third on seed 0.
- Mechanism and secondary:
  - cuOpt-role P, with echoes included and excluded;
  - boundaries with an EA start;
  - echo rate and latency;
  - dead time per slice;
  - the **"cuOpt helped" count**: cuOpt's own improvements, in slices that started from an EA vector, that set a new portfolio best. This separates "cuOpt was lifted" from "the EA carried the portfolio".
- All descriptive (D-010; 6 instances × 2 seeds).
- Run plan: Phase 1 = W × seeds 1–2 × 5 arms = 60 runs, about 2.3 h. Phase 2 = guard instances, 30 runs.

**implementation.** NEW `src/itrea/solvers/cuopt_slices.py`: pure Python, no cuOpt import, CPU-tested.
- `parse_slices(spec, budget, min_slice_s)`: accepts `none`, `fixed:K` or `at:...`.
- `next_limit(...)`: boundary-anchored limit, skips passed or tiny boundaries.
- `choose_start(policy, own, ea, sense, min_gain)`.
- `prepare_start(x, lb, ub, int_idx)`: round, clip, finite check, row violation.
- `EchoMatcher(int_idx)`: integer-part key.
- `BoundCarrier(sense)`: monotone best bound.
- `slice_seed(seed, k, stride=7919)`.

`src/itrea/solvers/cuopt_run.py`:
- Flags:
  - `--slices` (default `none`);
  - `--start-from {own,best}`;
  - `--recv`;
  - `--restart-min-gain` 1e-6;
  - `--min-slice-s` 5;
  - `--slice-seed-stride` 7919.
- `validate`:
  - `--recv` needs `--mailbox` and `--slices` ≠ none;
  - `best` needs `--recv`.
- Before the barrier, from the parsed DataModel: lb, ub and `get_variable_types` → int_idx, plus the constraint CSR for the violation log. Call `map_order` when `--send` or `--recv` is set.
- **`--slices none` keeps today's single-Solve path argument for argument** (A6 stays as-shipped; regression test).
- The slice loop as described under mechanism. Recorder gains `begin_slice()`, echo matching, and `slice`/`slice_start` tags on incumbent events.
- The EA's message is cached in-process. Verdicts are accepted=True on an echo, never accepted=False. Bound events are emitted only when the carried bound improves.
- Errors:
  - a Solve exception (`@catch_cuopt_exception`) or an error status raises CuoptError → exit 1, run failed;
  - a start-induced ValidationError is a bug in prepare_start, so it also fails the run (no silent retry without a start; D-002 "never silently re-run").
- Docstring:
  - "never receives solutions" becomes "never mid-solve; with --slices each slice starts from a MIP start, presolve kept";
  - correct the SIGTERM sentence at L29: the flag is seen only after Solve returns, and now also at boundaries.

`src/itrea/experiments/pair.py`:
- ProcSpec gets `send: bool` and `restart: "own" | "best" | None`.
- ARMS gains A6s, A7, A7s and A8, with the arm-implied EA overrides listed under arms.
- `plan()`:
  - cuOpt argv gains `--send`, `--recv --start-from best`, and `--slices job.get("cuopt_slices", "fixed:4")`;
  - an arm with both cuopt and ea pins cuOpt to `lane.a725` (1,2,3) and the EA to `lane.x925` (5,6). Record that this EA is not footprint-matched with A3's; the families are never compared.
- `solver_settings()` includes the slices and the restart policy, so A6s and A6 get different hashes.
- `run_pair` refuses a job-level `cuopt_slices` override without a `variant` label (EA arms are hashed by `cfg.hash()`).
- `merge_trajectories(solver_role="cuopt")` for arms that have cuopt and no solver.
- Memory options for this family, A6 included: `cuopt_mem_gb` 6 (measured RSS ≤ 3.1 GB) and `cuopt_gpu_peak_gb` 8 (measured ≤ 7.2 GiB).

`src/itrea/analysis/build_runs.py`:
- `cuopt_*` trajectory columns, with and without source `ea`;
- `cuopt_inc_from_ea` and `cuopt_new_best_after_ea_start`;
- restart columns from the summary;
- `post_accounting` also reads verdicts from role `cuopt` (today only roles starting with `solver`).

Elsewhere:
- `make_jobs.py`: job set `dev-cuopt-restart` (W × seeds 1–2 × 5 arms, one block per instance × seed, plus the guard set).
- D-013 tables and the five contrasts, through `analysis/stats.py`.
- Docs:
  - D-014 "cuOpt coupling by time-sliced restarts": exploratory, amends D-001's "launch-time warm start only" for these arms, carried into prereg-v2 per D-013;
  - Process interface (`cuopt --recv/--slices`);
  - Development log;
  - the AGENTS.md map;
  - issue #23.

TESTS, CPU (default suite):
- `tests/test_cuopt_slices.py`:
  - spec parsing;
  - anchoring under a simulated overrun;
  - tiny-tail skip;
  - `choose_start` with the min gain and a missing own or EA solution;
  - `prepare_start` with ±1e-7 out-of-bounds values, NaN, and integers at x.5;
  - echo matching when the continuous part changes;
  - monotone bound carrier.
- `tests/test_cuopt_run_slices.py`: a fake cuopt module (monkeypatched ParseMps, Solve and SolverSettings emitting scripted callbacks, statuses and overruns) on a toy MPS with a file mailbox. It checks:
  - `none` calls Solve once with A6's settings;
  - the start is cleared at slice 0 and when there is no start;
  - Optimal ends the loop;
  - an error gives exit 1 with a summary;
  - an echo is recorded as `ea` and posted with ORIGIN_EA_ECHO;
  - only accepted verdicts are written;
  - the info bound is monotone;
  - the stop file is honoured at a boundary;
  - a cached EA message is used at a later boundary.
- pair tests: argv, cores, distinct run ids, `budget_for` counting 2 CUDA processes, arm overrides, refusal without a variant.
- build_runs: a synthetic run directory.

GPU ONLY (`gpu` marker, inside a lease):
- a toy knapsack with `--slices fixed:3` end to end: echo seen, `rmm_current` back to baseline after each slice;
- `make-jobs smoke` for A6s and A8 on a toy.
- Real instances only in the scratch scripts `/tmp/itrea-cuopt/verify_mipstart.py` and `/tmp/itrea-cuopt/replay_restart.py`, never inside tests.

**risks.** 1. **Restart cost lands on cuOpt's weak instances.**
   - At 120 s, 4 of the 6 (sing44, neos-960392, rail01, ca) spend 23–26 s in presolve, and their first bounds come at 72–78 s.
   - With 30-s slices, PaPILO and cuOpt's presolve get 3 s each and the root LP 3 s. Presolve is weaker, B&B makes no progress, there is no bound, and the neos-960392 proof at 99 s is lost.
   - rail01: cuOpt's only incumbent comes at 107.5 s and the EA never finds a feasible one. A6s and A8 will likely score 1.0 against 0.927, a weak-set loss.
   - Variants if A6s − A6 is large: `fixed:2`; `at:60,90`; or the `mip_hyper_heuristic_presolve_time_ratio/_max_time` and `_root_lp_*` parameters to give each slice an absolute budget. That costs more per slice and moves further from as-shipped.
2. **The EA may carry the portfolio alone.** The EA's vectors at 30 s are far ahead of cuOpt's own:
   - loue: 49,699 vs 108,133;
   - item_placement: 11.08 vs 25.39;
   - sing44 (seed 1): 1.21e7 vs 3.76e8;
   - neos-960392: −238 vs −34.
   These come from A3@lp1, a SCIP-coupled EA. If cuOpt only echoes such a vector, then A8 ≈ A7s ≤ A7 and restarts cost without adding anything. The first test checks this.
3. **No echo** is ambiguous:
   - PaPILO's dual reductions can cut off a feasible start, and user starts get no original-space fallback (from reading the source);
   - the EA's tolerance can disagree with cuOpt's (hence `ranks.tol_mode: highs` and the logged start violation, which tell this apart from a cut-off);
   - early FJ during presolve ignores the start.
   Logged as the echo rate; never turned into a rejection.
4. The fixed-integer LP for the start is capped at min(60, remaining/2). On the continuous-heavy ca it can use up half a slice before the start is in.
5. Python allows **one start per Solve**. When the EA's vector is passed, cuOpt's own best leaves its population. The follow-up is the C API's repeatable `cuOptAddMIPStart` via ctypes, which needs C-API problem and settings handles.
6. **GPU bandwidth contention.** In optopt A4, GPU load beside cuOpt doubled its P(30). Contrast 4 measures this here, and it may cancel the gain.
7. **Memory.** Two CUDA processes need about 64 GB MemAvailable at launch, or 54 GB with `cuopt_mem_gb` 6 and peak 8.
   - That fits now (both LLM servers stopped, about 108 GB free) and in any window like it.
   - It does **not** fit beside a running [local LLM agent] (55–56 GB in the A6 runs); the pool's gate will refuse.
   - RMM growth across in-process Solves is unverified; `rmm_current` is recorded per slice and after each clear.
8. **Heat:** two GPU processes. D-008 start gate below 80 °C, pause at ≥ 92 °C.
9. **Unverified API behaviour:**
   - whether a ValidationError comes back as an error status or an exception (both are handled as a failed run);
   - deterministic mode appears to ignore starts, so never set `mip_determinism_mode`;
   - start used with presolve kept (V1 decides).
10. **Attribution.** Integer-part echo matching credits the EA conservatively. A cuOpt move away from an EA start counts as cuOpt's own, reported separately through the `slice_start` tags.
11. **Small evidence base.**
   - n is small: 6 instances × 2 seeds, with the weak set from one A6 seed.
   - The EA here sits on X925 cores 5–6, so A7 and A8 are not comparable with A3.
   - The arms are exploratory: D-014 and prereg-v2 are needed before any confirmatory use.

**first_test.** A replay of one restart at 30 s from stored vectors. **No repository code is needed.**

SCRIPT: `/tmp/itrea-cuopt/replay_restart.py`. Written and syntax-checked; its trace-reading helpers were checked on the CPU against the real npz files; **not run**.
- It checks the name mapping before the CUDA warm-up.
- It records `rmm_current` after every Solve and after every clear.

GATE:
- Inside the i-trea-ii lease, **after the queued pool (lpd, then Stage 2) has finished**: one GPU arm at a time.
- Below 80 °C.
- `OMP_NUM_THREADS=1 taskset -c 1-3`, RMM limit 12 GB, about 36 GB MemAvailable.

STEP 0 (about 1 min): `/tmp/itrea-cuopt/verify_mipstart.py`, V0–V4 on glass4. It shows whether the start is used, whether presolve is kept, whether there is an echo, the in-process restart overhead, and what an out-of-bounds start does.

STEP 1 (about 9.5 min): one process, 6 Solves of 90 s each, seed 1, ABBA order. Each restarts at τ = 30 s, the first boundary of `fixed:4` at 120 s, on three weak instances:
- **neos-3024952-loue** (presolve-light, A6 P 0.760): start = the EA's best at ≤ 30 s from A3@lp1 seed 0 (49,699), or cuOpt's own best at ≤ 30 s from A6 seed 0 (108,133).
- **item_placement_9903** (0.666): EA 11.08 (seed 0) or own 25.39.
- **sing44** (presolve-heavy, 0.726): EA 1.21e7 (A3@lp1 seed 1, because seed 0 had no EA vector by 30 s) or own 3.76e8 (A6 seed 0). This case mixes seeds.

The yardsticks come from traces: the EA's recorded trajectory after 30 s, and A6's own continuation.
- Both are **SCIP-coupled**. The EA vectors only approximate those of an EA seeded by cuOpt, and since the yardstick includes SCIP's help, it is conservative.

Per Solve the script prints:
- the echo latency (sing44's is the presolve-heavy dead-time number);
- presolve_s and nodes;
- incumbents better than the start;
- how many cuOpt incumbents **beat what the EA held at the same moment**.

It also prints the P integral over [30, 120] for:
- cuOpt from the EA's start;
- cuOpt from its own start;
- A6's continuation;
- the EA alone;
- the EA + A6 portfolio (≈ A7);
- the EA + cuOpt-from-EA portfolio (≈ A8).

The own-start Solves against A6's continuation give the first reading of restart cost.

DECISION (three outcomes):
1. V1 shows the start is not used, or presolve is disabled, or an in-bounds start errors → **the restart design is dead.** Build nothing for it.
2. The start is echoed, but on none of the three instances does cuOpt ever beat the EA's recorded value at the same moment → **build only A7** (one-way stream, #23 option (i)). Restarts would add cost and nothing else.
3. On at least 2 of 3 instances the EA start is echoed within 27 s (0.3 × 90 s) **and** at least one cuOpt incumbent beats the EA's value at that moment → **build A6s, A7s and A8** (the script prints this VERDICT line).

Total: about 11 minutes of GPU, one process.

If the test passes, the next cheapest step is A6, A6s, A7s and A8 on loue, item_placement and sing44 × seed 1: 12 runs, about 30 min. The full five-arm Phase 1 follows.

### K2: restart cuOpt from the EA's best when the EA pulls ahead (kill-on-improvement), checked against a yoked own-best restart control (K2y)

**mechanism.** PROCESS STRUCTURE
- The `cuopt` role becomes a supervisor process, a new module `itrea.solvers.cuopt_restart`. It is a SolverProcess: it owns the trace, mailbox, map_order, barrier, record_incumbent, finish, one ready file, one summary and one exit code. pair.py's contract stays the same.
- The supervisor never imports cuopt or rmm, so the parent holds no CUDA context.
- cuOpt runs in worker children, a new module `itrea.solvers.cuopt_worker`, started with subprocess.Popen. Workers stay in the supervisor's process group and systemd scope, with no setsid. That way pair.py's killpg, the scope's MemoryMax and gpu_health all cover them (gpu_health.descends_from walks the parent chain, so workers count as own processes, not foreign ones).
- Worker protocol, as JSON lines on stdin/stdout:
  1. Set-up: RMM LimitingResourceAdaptor(--gpu-mem-gb), import cuopt, ParseMps. The first worker only also dumps names, lb, ub and variable types to cuopt.vars.npz. Then the synthetic warm-up.
  2. Print READY, then block until a GO line arrives: {slice k, t0, time_limit, seed, start_path|null}.
  3. On GO: load the start, re-check length, finiteness and bounds (clip), call dm.set_initial_primal_solution, then Solve.
  4. A GetSolutionCallback appends every improving incumbent to cuopt.w<k>.part (the existing .part record format) and then writes a short notice {k, t on the shared t0 clock, obj, bound, file offset}.
  5. When Solve returns: a final notice (status, dual bound, nodes, presolve_s), w<k>.summary.json, exit 0.
- Worker 0 and one standby (worker 1) do their set-up before the barrier. The clock rule allows parse, CUDA context and synthetic warm-up before t0.
- At t0 the supervisor sends GO to worker 0 with time_limit = T, no start and seed s. Until the first kill, K2 is identical to C1.

WHEN CUOPT RESTARTS (checked every 0.25 s; all five conditions must hold)
- (a) Margin: the newest to_solver message's best row (from the EA) beats cuOpt's held best by at least γ·max(1,|held|), with γ = 0.005. The held best is the best over all slices, echoes included.
- (b) Minimum interval: at least τ_min = 20 s since the current worker's GO.
- (c) Progress gate, added to the rule as asked: the current worker has echoed its own start, or 2·τ_min has passed without an echo. Worker 0 has no start and is protected for τ_min only.
  - Reason: a start enters cuOpt only after PaPILO, cuOpt's presolve and the root LP (diversity_manager L500, then L520).
  - On sing44, neos-960392 and ca, cuOpt alone spends 23–26 s in presolve, and its first bound and large improvements come at 72–90 s (A6 traces).
- (d) At least R_min = 0.3·T remains (36 s at T = 120 s; 90 s at T = 300 s). A new worker spends up to 0.1R (PaPILO) + 0.1R (cuOpt presolve) + min(15, 0.1R) (root LP) before its start counts.
- (e) Fewer than max_restarts so far: 4 at 120 s, 6 at 300 s.

HOW THE TRIGGER BEHAVES ON cuOpt's WEAK SET (offline proxy: EA incumbents in A3@lp1/lps1 against cuOpt alone, A6 s0; /tmp/itrea-cuopt/proxy_triggers.py)
- **neos-3024952-loue, item_placement_9903, sing44:** the EA beats cuOpt at every 15-s check. Condition (a) is therefore always true, and the rule collapses to "restart every τ_min, subject to (c)–(e)". At 45 s the EA has 41,700 vs 108,133 (loue), 19.85 vs 24.29 (item_placement) and 1.29e7 vs 3.76e8 (sing44).
- **ca:** the trigger first fires at about 35–64 s (EA 34,212 vs cuOpt 5.77e6 at 45 s).
- **rail01:** the EA never has a feasible solution, so cuOpt never restarts and K2 equals C1.
- **neos-960392:** the EA holds the optimum −238 by about 1 s. A restart can only shorten cuOpt's proof; cuOpt alone proved it at 99.9 s.
- This is why the yoked control, not the margin, separates the EA's contribution from the cost of restarting.

KILL AND RELAUNCH
1. SIGKILL the active worker's pid with os.kill. Do not use killpg, because the supervisor shares the group.
2. Reap with a bounded wait of at most 5 s, and record the time. A worker not reaped by then is a straggler: the memory gate counts it, and the supervisor reaps every worker before it exits. A worker that outlives the supervisor would be reparented, and gpu_health would void the block as foreign GPU activity.
3. Rescan the dead worker's .part file from its last notified offset, to recover records written but not yet notified.
4. Prepare the start from the EA row:
   - map it to cuOpt's order (namemap.to_local);
   - round the integer columns;
   - clip to the worker's own lb/ub (any value outside them by more than 1e-8 fails the whole Solve with a ValidationError);
   - reject rows with NaN or non-finite values (cuOpt drops partial starts);
   - write it atomically as start_k.npy.
5. Send GO to the pre-warmed standby, with time_limit = T − (now − t0) and seed s + 7919·k.
6. Spawn the next standby. Its set-up takes about 2.5 s after t0, on the clock, on the same three A725 cores and the GPU, beside the active solve.
- `--no-standby` gives a cold relaunch. That adds about 2.5 s of dead time: imports 1.05 s, parse 0.01–1.7 s, warm-up 1.4 s. The standby is the default; both costs are measured.

WHAT FLOWS
- **cuOpt → EA**
  - Every improving held incumbent from any worker goes on to_ea.
  - A row that echoes the current start is posted with ORIGIN_EA_ECHO and recorded with source "ea", so the engine skips it and cuOpt gets no credit for it. A row is an echo when its rounded integer part equals the start's. Matching on the integer part is needed because the fixed-integer LP changes the continuous values and the objective, so vector_hash misses the echo.
  - A strictly better row that is not an echo is posted with ORIGIN_SOLVER and recorded with source "self".
  - `info` carries the held incumbent, the best dual bound over all slices, and status "solving" or "restarting". A proven bound stays valid across restarts, and a restarted worker reports none until its root finishes.
- **EA → cuOpt**
  - The engine's existing post rule applies: post when the EA beats inc_external by post_min_gain. The supervisor keeps only the newest message's best row.
  - When the echo of an EA start arrives, the supervisor writes `mb.verdict(seq, obj, True, note='{"row": r}')`. Engine._verdicts matches on seq and reads the row from the note.
  - It never writes `accepted=False`. On any false verdict the engine demotes the individual and reverts its incumbent, yet a missing echo is not infeasibility: PaPILO dual reductions or early feasibility jump can hide one. Missing echoes are logged as `start_no_echo` in the supervisor's trace instead.
- **End of the run**
  - A worker returning Optimal or Infeasible makes the supervisor exit with that exit_reason, so pair.py's DEFINITIVE rule ends the race.
  - A runner `stop` or a SIGTERM makes the supervisor SIGKILL and reap its workers, write cuopt.incumbents.npz and its summary, and exit 0.

TIME BUDGET
- Both processes share one t0 and one T. Each worker's time_limit is the remaining time at its GO. P(T) counts only incumbents with t ≤ T.
- Everything a restart costs is on the clock and recorded per slice:
  - kill to reap, and reap to GO;
  - GO to first callback, and GO to echo (the dead time);
  - the standby's set-up after t0;
  - presolve_s and root-LP time from the worker's log_file.
- The last worker may overrun like A6 (0–4.7 s measured) and gets the same grace.

**arms.** SHARED EA CONFIGURATION, identical in every arm with an EA
- configs/itrea2-default.yaml + seeding.lp_source=highs_async (the "lp1" variant; cuOpt exports no root LP), exchange.use_root_lp=false, lp_stream off.
- exchange.to_solver=true everywhere, so the EA's posting cost is the same in every arm. Only the supervisor's policy decides whether cuOpt reads the posts.
- tol_mode: check cuOpt's absolute feasibility tolerances first; use "highs" (absolute 1e-6) if that is at least as strict.

ARMS
- **A6** (exists): cuOpt alone, as shipped. This is the D-013 baseline B and stays plain cuopt_run.
- **C1:** cuOpt → EA one way, no restarts. It runs through the supervisor with policy `none`, the same worker and notice path, so supervisor overhead does not confound K2 vs C1. This is D-013 candidate (i).
- **K2:** kill-on-improvement; the start is the EA's best (policy `ea`).
- **K2y**, the yoked own-best control and the primary attribution control:
  - It replays K2's recorded kill and GO times for the same instance, seed, budget and configuration.
  - The start is cuOpt's held best at that moment: its own, including echoes of its own earlier starts.
  - Per-slice seeds and standby behaviour are the same as K2. The EA runs and posts, but cuOpt never reads the posts.
  - The only difference from K2 is the start vector. If no incumbent exists at a yoked time, the restart happens without a start, and that is recorded.
- **A6y** (secondary): cuOpt alone with K2's yoked pattern and its own best. It measures restart cost without the EA's GPU contention, on cuOpt-held P.
- **Synthetic A6+A5** (computed, no runs): the uncoupled portfolio. It has no contention, so it flatters the uncoupled side.

CONTRASTS (D-002 machinery: per-instance bounded d with seeds averaged, Wilcoxon, 95 % bootstrap interval)
1. **D-013 primary:** K2 vs A6 on the weak set W_cuOpt, i.e. P_A6 ≥ 0.5 on a reference seed that is not used for evaluation.
   - The reference is the existing A6 seed 0. Weak: sing44 0.726, neos-960392 0.571, loue 0.760, rail01 0.927, ca 0.643, item_placement 0.666 (6 of 9). Not weak: glass4 0.308, radiationm 0.009, fhnw 0.005.
   - Evaluation uses seeds 1 and 2, with A6 rerun on the same day.
   - Report the share improved, the mean P reduction, mean d with its interval, the Wilcoxon test and the worst P after coupling.
   - Guard: mean d over all 9 instances ≥ −0.05. On glass4, fhnw and radiationm cuOpt is strong and the EA rarely beats it, so few restarts are expected; the guard checks that.
2. **Attribution, the only contrast that credits the EA:** K2 vs K2y, on portfolio P and on cuOpt-held P.
3. **Restart cost or benefit, charged to the mechanism and not to the EA:** K2y vs C1 (with the EA running), and A6y vs A6 (cuOpt-held, no EA).
4. **The EA beside cuOpt:** C1 vs A6, which includes bandwidth contention, and C1 vs the synthetic A6+A5.

WHAT IS REPORTED FOR EVERY ARM
- Portfolio P: best of the cuopt and ea roles, which is D-013's "B coupled with the EA".
- cuOpt-held P: everything cuOpt held, echoed EA starts included, like the existing SCIP-held column. This answers D-013's caveat that the current cuOpt row is "our SCIP+EA against cuOpt, not cuOpt helped".
- cuOpt-own finds: incumbents from a restarted worker that are strictly better than its start, have a different integer part, and also beat the EA's best at that moment. This is the direct count of "cuOpt improved the EA's solution". LP polish of the start (same integer part, better objective) is counted separately.
- Mechanism counters: restarts, echo rate, dead time (median and maximum), start_no_echo, reap times, standby set-up seconds.

SCOPE AND RUN ORDER
- Only 4 of the 6 weak instances can show a restart effect: sing44, loue, ca, item_placement. rail01 and neos-960392 stay in W_cuOpt and count in the primary as measured.
- Development pilot on the weak set: 6 instances × 2 seeds × 4 arms (A6, C1, K2, K2y) = 48 runs at 120 s, about 2 h with start-up. The full development set adds A6y and the 3 strong instances, 90 runs, about 4 h.
- K2y and A6y read K2's run directory, so the pool permutes {A6, C1, K2} by the Williams design and appends the yoked arms last in each block. This is a stated limitation (time-of-day drift), mitigated by same-session blocks, the thermal gate and GPU health.
- These numbers are descriptive only. Nothing is confirmatory before a D-014 decision and prereg-v2.

**implementation.** NEW CODE (repository /home/coulson/git_repositories/i-trea-ii; nothing written there by this design)
- **src/itrea/solvers/cuopt_worker.py** (GPU; tested only on the GPU)
  - Reuses cuopt_run._rmm_setup, _settings, _warmup and STATUS.
  - READY/GO protocol over stdin/stdout JSON lines.
  - cuopt.vars.npz from the first worker: names, lb, ub, types. get_variable_types returns S1/U1 'I'/'C' codes; decode bytes or str.
  - Per-slice .part file plus notices, w<k>.summary.json, exit 0 or 1.
  - Never set mip_determinism_mode: deterministic mode ignores starts.
- **src/itrea/solvers/cuopt_restart.py** (the supervisor; CPU only)
  - Arguments: base_parser(exchange=False) plus --recv, --send, --policy {none,ea,yoked}, --yoke-from DIR, --restart-min-gain 0.005, --min-interval-s 20, --echo-wait-factor 2, --min-remaining-frac 0.3, --max-restarts 4, --standby/--no-standby, --reap-timeout-s 5, --seed-stride 7919, --cpu-threads, --gpu-mem-gb, and --worker-cmd (tests point it at a fake worker).
  - Pure, unit-testable functions:
    - prepare_start(x_can, namemap, lb, ub, is_int) → x or a reason;
    - should_restart(state, cand, now, params) → (bool, reason);
    - is_echo(x_local, start_local, is_int), matching on the integer part;
    - read_part_record(path, offset) and rescan_part(path, from_offset);
    - yoked_schedule(run_dir) → [(t_kill, k, seed)];
    - carry_bound(prev, new, sense).
  - Summary: a per-slice `restarts` table {k, t_kill, reap_s, t_go, t_echo, start_src (ea|own|none), start_obj, seq, gain, status, presolve_s, nodes, final_dual} plus totals.
  - It writes one cuopt.incumbents.npz with sources self and ea; the workers' .part files are kept as evidence.

CHANGES TO EXISTING FILES
- **src/itrea/experiments/pair.py**
  - ProcSpec gains `restart: str | None`. ROLE_MODULES gains the supervisor.
  - New arms C1, K2, K2y, A6y; A2, A6 and H2 are untouched.
  - plan():
    - passes --send (and --recv for policy `ea`) and the policy parameters from a job's `restart` dict;
    - for yoked arms, passes --yoke-from <traces>/raw/<set>/<run_id of the same job with arm K2>, and raises RunError if that run is missing or not completed;
    - sets cuopt_mem_gb to 16 for restart arms. ca's worker RSS is 3.1 GB, and active + standby + straggler + supervisor sits near the current 12 GB scope.
  - budget_for(): for restart arms, add one CUDA process (the standby, +1 GB) and one extra cuOpt RMM peak or limit (a possible D-state straggler). At T = 120 s that is about 12 (EA) + 2×7.7 (cuOpt peak) + 3×1 GPU, plus host 22, Redis 2 and the 20 reserve: about 74 GB. MemAvailable is 114 GB now.
  - solver_settings() and run_hash(): include the policy and its parameters. K2, K2y and C1 share the EA configuration hash, so without this their records would differ only by the arm label.
  - merge_trajectories(..., solver_role="cuopt") for cuOpt arms. The abort path's SIGKILL for kind cuopt already uses killpg, which reaches the workers.
- **src/itrea/solvers/cuopt_run.py:** behaviour unchanged (A6, A2 and H2 stay as shipped). Correct the docstring's SIGTERM sentence (about L29): Python sees SIGTERM only after Solve returns.
- **src/itrea/analysis/build_runs.py:** cuopt_* columns (held P, own-after-start counts, polish counts), restart counters, producer_of for the supervisor.
- **make_jobs.py:** a `cuopt-restart-dev` job file.
- **pool.py:** a "yoked arms last" rule within blocks.
- **New ablation file configs/ablations/cuopt_partner.yaml:** lp_source highs_async, use_root_lp false, lp_stream off.

VAULT RECORDS
- A D-014 decision on the cuOpt restart coupling. D-001's "launch-time warm start only" stays true for A2 and A6, and the "cuOpt NEVER receives solutions" docstring stays true for cuopt_run.
- A prereg-v2 amendment entry, and a result note per experiment.

TESTS ON THE CPU (toy models, fake worker, no cuOpt import)
- **tests/test_cuopt_restart.py**
  - prepare_start: clipping, integer rounding, NaN rejection, length mismatch, name mapping.
  - should_restart truth table: margin; interval; echo gate and its cap; R_min; max restarts; no candidate; no held incumbent.
  - is_echo when the continuous part changed.
  - .part reading, a truncated last record, and the rescan after a kill.
  - yoked_schedule round trip.
  - Supervisor end-to-end through --worker-cmd and --no-barrier, with a fake worker: a small script that speaks READY/GO/notice, emits scripted incumbents, echoes its start on request, sleeps, and can be SIGKILLed. Checks:
    - restarts fire at the expected times, and yoked mode replays them;
    - verdicts are written only on an echo, with note {"row": r};
    - records merge and the process exits 0;
    - a stop or SIGTERM mid-slice leaves every worker reaped.
- **tests/test_pair.py additions:** plan() argv for C1, K2, K2y and A6y; budget_for counts; distinct run_hash; RunError when the yoke source is missing. One run_pair end-to-end on a toy, with module_override pointing the cuopt role at the supervisor and a fake worker, plus the NumPy EA, and gpu_health off.

GPU ONLY (inside an i-trea-ii lease)
- /tmp/itrea-cuopt/verify_mipstart.py, then /tmp/itrea-cuopt/splice_probe.py (see first_test).
- After the build: a 10-s toy smoke of K2 with the real worker (the knapsack toy). It checks one restart, the echo, the files and every exit code.

**risks.** 1. **Dead time can erase cuOpt's late phase.** On sing44, neos-960392 and ca, cuOpt alone's real progress starts after its root, at 72–90 s. Each restart re-pays presolve (up to 0.2R) and the root LP (min(15, 0.1R)) before the start counts. Gates (c) and (d) limit this but do not remove it, so K2's cuOpt-held P can end up worse than A6's. The yoked control and A6y measure this cost; it must not be booked against the EA.
2. **The rule becomes fixed-interval restarts.** The EA beats cuOpt continuously on loue, item_placement and sing44, so K2 there is "restart every 20–40 s". Its value shows only against K2y.
3. **cuOpt may add nothing.** Given a strong start, it may only echo it back, or polish its continuous part — which the EA's own highs_async LP could do. Portfolio P would then barely move. The first test checks exactly this, and counts same-integer polish separately.
4. **Silent no-ops.**
   - PaPILO dual reductions can cut a feasible start off, and user starts get no original-space fallback.
   - cuOpt discards partial (NaN) starts.
   - Deterministic mode ignores starts.
   - All of these are counted as start_no_echo, and mip_determinism_mode stays 0.
5. **A start outside its bounds by more than 1e-8 fails the whole Solve** with a ValidationError. Clip in both the supervisor and the worker.
6. **SIGKILL of a CUDA process has never been exercised here.**
   - NVIDIA warns that a killed worker can sit in uninterruptible D-state and hold its memory until it exits.
   - Mitigations: a bounded reap, a memory gate that budgets one straggler, and the supervisor reaping every worker before it exits. An orphaned worker would be reparented, read as foreign GPU activity, and void the block.
   - The first test measures reap time and memory recovery.
7. **Bandwidth contention.** cuOpt, the EA and the standby's warm-up share the GPU; in optopt A4, continuous GPU inference beside cuOpt doubled cuOpt's P(30). K2 vs A6 includes this, which is honest for D-013. K2 vs C1 and K2 vs K2y are matched for contention, and A6y isolates restarts without the EA.
8. **Provenance.**
   - If cuOpt's feasibility jump changes an integer value of the start, the integer-part echo match fails, and the credit wrongly goes to cuOpt. Record match statistics, and cross-check with the objective.
   - Writing any accepted=False verdict would make Engine._verdicts demote the individual and revert the EA's incumbent.
   - A verdict without note {"row": r} goes unmatched.
9. **Design limits.**
   - The yoked arms cannot be randomised: they must follow their K2 run.
   - Only 4 effective weak instances × 2 seeds on the development set, so results are descriptive. The minimum detectable effect of 0.20–0.25 in d needs the test set.
   - W_cuOpt comes from a single cuOpt seed. D-013's sensitivity check (each solver's worst third) applies.
   - Changing seeds per slice is part of the restart effect, and is matched in K2y. cuOpt's opportunistic-mode nondeterminism makes seed matching approximate anyway.
10. **Harness complexity.** The new parent–worker protocol is where harness bugs would live: a lost notice, a GO that never arrives, a standby that is not ready at kill time (the supervisor waits, and the wait is on the clock and recorded). Fake-worker CPU tests come first. A worker not READY within the ready timeout fails the run; it is not void.
11. **Governance.** The new arms break the "cuOpt never receives solutions" statement for themselves only. That needs D-014 plus a prereg-v2 amendment, and issue #23 is labelled needs-berk. Nothing may be written as a finding before its result note. The first test's P uses reported, unchecked objectives: a go/no-go only, never a result.

**first_test.** ONE LEASE, cuOpt only: no EA process and no supervisor. About 13 min on the GPU. ACPI below 80 °C, nothing else on the GPU lane.

Step 0: MIP-start check (about 1–2 min, glass4)
- Run: `cd /home/coulson/git_repositories/i-trea-ii && OMP_NUM_THREADS=1 taskset -c 1-3 .venv/bin/python /tmp/itrea-cuopt/verify_mipstart.py`
- Continue only if the VERDICT line shows `start_used: True` and `presolve_kept: True` (V1), and V2 shows that an in-process re-Solve with a start works.
- Otherwise the design is dead.

Step 1: SIGKILL probe (about 1 min)
- Run: `/tmp/itrea-cuopt/splice_probe.py kill`
- A child process runs a 60-s Solve and is SIGKILLed mid-CUDA: glass4 at 5 s (presolve takes 0.4 s, so GPU heuristics and branch-and-bound are running) and sing44 at 30 s (past its presolve).
- It records the reap time, the process state letters while dying, MemAvailable 0.5, 2 and 5 s after the kill, and the last cuOpt log lines (which phase was killed).
- Needed: reap within 5 s, no lingering D state, memory back within 5 s.
- If not, add a wait-for-reap rule and straggler budgeting before any arm runs.

Step 2: open-loop splice (about 10–11 min)
- Run: `/tmp/itrea-cuopt/splice_probe.py splice`
- Instances: sing44, neos-3024952-loue, ca__Scenario400_reserves_1, item_placement_9903.
- One process with one CUDA context, a synthetic warm-up first. Then 8 Solves: τ = 45 s, R = 75 s, seed 1, same seed for both variants.
- **"ea" start:** the EA's best incumbent held at 45 s in A3@lps1 seed 1, a fixed rule with no per-instance choice: 1.285e7 / 41,700 / 34,212 / 19.85. These include solutions the EA adopted from SCIP, so they are an optimistic stand-in for a cuOpt-fed EA.
- **"own" start:** cuOpt alone's best at 45 s (A6 s0): 3.76e8 / 108,133 / 5.77e6 / 24.29.
- It records every callback, including whether its integer part equals the start's, the echo time (the dead time), presolve time, and the log flags (Papilo presolve, "Presolve is disabled", "cannot add").

Step 3: analysis (CPU, seconds)
- Run: `OMP_NUM_THREADS=1 taskset -c 10-14 .venv/bin/python /tmp/itrea-cuopt/splice_probe.py analyse`
- It computes spliced P(120) against the dev references in datasets/runs.parquet:
  - C1 = EA source ∪ A6;
  - K = EA source ∪ A6 up to 45 s ∪ restart from the EA's start;
  - Y = the same with the own start;
  - cuOpt-held versions of each.
- Each is computed twice: with the EA frozen at 45 s (pessimistic for the EA) and with it continuing (optimistic for the EA, since it is SCIP-assisted).

DECISION
- **GO** (build the supervisor and run the 48-run weak-set pilot) if, on at least 2 of the 4 instances, all three hold:
  - (i) cuOpt started from the EA's solution finds an incumbent better than its start with a different integer part before 120 s;
  - (ii) P(K) < P(C1) − 0.02 with the EA frozen;
  - (iii) cuOpt-held P(K) < P(Y).
- **STRONG GO** if (ii) also holds with the EA continuing.
- **STOP** if on at least 3 of the 4 cuOpt only echoes or LP-polishes the start. If cuOpt only polishes, the lesson is to do LP polish inside the EA, not to restart cuOpt.
- The measured dead times set τ_min and R_min.

STATE OF THE SCRIPT
- Written by me, not run on the GPU.
- CPU checks done:
  - it compiles; the helpers resolve all 8 start vectors without importing cuopt or rmm;
  - the analyse step dry-runs with placeholder probe results and reproduces cuOpt alone's P of 0.726, 0.760, 0.647 and 0.666;
  - the kill child's code parses;
  - the cuOpt API names were checked in data_model.py.
- Supporting read-only script: /tmp/itrea-cuopt/proxy_triggers.py.
- Leftover placeholders: /tmp/itrea-cuopt/dryrun_analyse/ (safe to delete).
