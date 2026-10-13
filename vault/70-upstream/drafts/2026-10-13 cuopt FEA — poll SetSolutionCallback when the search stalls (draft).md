# Draft — NVIDIA/cuopt follow-ups to #2073 (not posted; waits for Berk's yes, per post)

## (a) Comment on #2073 (closed)
> Thanks @akifcorduk — verified on the nightly `26.10.0a294.post261013051018` (aarch64, GB10). With a
> SetSolutionCallback registered, Papilo presolve now stays on, and solutions we inject in the original variable order are
> crushed, accepted and reported back unchanged through the GET callback within a few ms on glass4, sing44,
> neos-3024952-loue, neos-2657525-crna and item_placement_9903. Infeasible offers (a fractional integer, a shifted
> vector, and a glass4 point slightly below the proven optimum) are rejected. With dual reductions off, the presolved size
> was unchanged on four of the five and three columns larger on crna. One follow-up on *when* the callback is polled is in
> #NNNN. (Posted by Claude Code, an AI assistant working with @berkorbay.)

## (b) New issue
**Title:** [FEA] Poll SetSolutionCallback periodically, not only when the GPU population improves

**Body:**
> **Context.** Since #2098 (thanks!), a SetSolutionCallback keeps presolve on and injected solutions are crushed
> correctly. We run cuOpt beside other solvers and inject their incumbents through this callback.
>
> **What we see.** `set_solution` is only called from `population_t::run_solution_callbacks`, which runs when a solution
> enters the GPU population as a new best (`add_solution`), for initial solutions, and in
> `generate_quick_feasible_solution`. Incumbents found elsewhere (CPU FJ, branch-and-bound) reach the GET callback
> without polling SET callbacks. So once the GPU heuristics stop improving, cuOpt never asks for an outside solution.
> That is exactly when a partner's solution would help most.
>
> Measured on `26.10.0a294.post261013051018` (GB10, 3 threads, `time_limit` 30 s, default settings, a SET callback that
> offers a known feasible solution at its first call):
> - neos-2657525-crna: 11 SET calls, all between 0.70 and 0.71 s, then none in the remaining 29 s (GET reports continued
>   until 9.2 s);
> - sing44: the first SET call came at 28.3 s, after 639 GET reports.
>
> In both cases the offer was accepted within a few ms once a call happened.
>
> **Request.** Poll SET callbacks also on a timer (e.g. every 1 s, or a `mip_set_solution_poll_interval` parameter),
> and/or when branch-and-bound or CPU FJ find an incumbent, so external solutions can enter while the GPU population is
> stalled. Polling only when the population improves could stay as the default.
>
> Repro: a short script with ParseMps + Get/SetSolutionCallback that logs the call times (happy to attach).
> (Filed by Claude Code, an AI assistant working with @berkorbay.)
