# Feasibility-jump local search — the repair operator (spec, 2026-10-05)

One algorithm, three implementations that must agree in semantics (not bit-for-bit):
`ls_fj_cpu.py` (Numba, one core — the CPU reference and arm A4's repair), `ls_fj_gpu.py` (CuPy `RawKernel`, one thread
block per individual — the GPU EA's repair), `ls_dense.py` (vectorised full rescore — a baseline only). Dispatcher:
`repair.LocalSearch`. Basis: Feasibility Jump (Luteberget & Sartor 2023), weighted violation with jump values,
adapted to a population and to an objective cutoff (critique 2026-10-05).

## State per individual
`x` (n), activities `act = A x` (m), row weights `w` (m, init 1.0; per individual), the set of violated rows (kept
incrementally: an index list + position array), counters. Problem arrays shared: CSR (row_ptr, col_idx, val), CSC
(col_ptr, row_idx, val), rl, ru, lb, ub, isint, cmin, row scale `s_i` (the `viol_norm` divisor; 1 for "count").
With the self-adaptive repair strategy ([[I-TREA II algorithm]]) the one search keeps `s_i` for every row scale of
`strategy.row_scales` — an (R, m_aug) table, the cutoff row's entry included — and each individual of a call reads its
own row (a per-individual scale index; the GPU kernel offsets its `s` pointer per block). Everything else is shared, so
individuals of every scale are repaired in ONE call: on the GPU their blocks run side by side and the slowest
individual's tail is paid once. (Review 2026-10-06: one call per scale, one after another, cost +13–39 % per
generation with two scales; with one call the strategy-on generation time is within a few % of the strategy-off one.)

## Violation and objective cutoff
Row violation `v_i(a) = (max(rl_i − a, 0) + max(a − ru_i, 0)) / s_i` (infinite sides contribute 0). With
`objective_mode = cutoff_row` and a finite incumbent key `inc`, an extra row `r*`: `cmin·x ≤ inc − δ` with
`δ = max(imp_tol·max(1,|inc|), g)` (g = 1 when every cost sits on an integer variable with integer coefficient — the
objective is integral — else 0), its own weight `w*` (init 1, bumped like any row), its activity `cmin·x` kept
incrementally. `weighted`: no extra row; every move's score adds `obj_weight · cmin_j · Δx_j / max|cmin|`. `none`:
nothing. The cutoff (`inc`) is refreshed by the engine whenever the incumbent improves.

## Moves
Candidate values `t` for variable j (current value `x_j`):
- binary: `1 − x_j`;
- general integer: `x_j ± 1`, the bounds (if finite), and for each row i of column j the values that make row i tight
  at its violated side (`x_j + (side − act_i)/A_ij`), rounded down and up; clipped to [lb_j, ub_j]; at most
  `max_candidates` (default 16) — the tightening values of the rows with the largest weighted violation first;
- continuous (`continuous: auto` — not a singleton column decoded in closed form): the same tightening values without
  rounding, plus the bounds.
Score `Δ_j(t) = Σ_{i ∈ col j} w_i [v_i(act_i + A_ij (t − x_j)) − v_i(act_i)]` (+ the cutoff row's term, or the weighted
objective term). Lower is better; a move is **improving** if `Δ < −1e-12`.

## One step for one individual
1. If no row is violated (cutoff row included when active): the child is done (feasible; with an active cutoff it is
   strictly better than the incumbent).
2. Pick a violated row `i` at random with probability ∝ `w_i · v_i` (or uniformly among a bounded sample of ≤ 32
   violated rows — implementation's choice, documented).
3. For every variable j in row i: its best candidate value and score.
4. Best improving move → apply: `x_j ← t`, update `act` for rows in column j (and the cutoff activity), update the
   violated-row set for those rows. Count a move.
5. No improving move (local minimum): `w_k += weight_bump` for every violated row k (FJ's weighting; a sample is
   allowed for speed), cutoff row included; count a local minimum. Apply the best (least bad) move of row i with
   probability `p_random_walk` (default 0.0 — off; documented knob).

## Stop per individual
After `max_steps` moves, or `stall_steps` consecutive steps without a decrease of the best total weighted violation
seen in this call, or when done (1.). Lanes that finish early are idle (GPU) or skipped (CPU). The call returns when all
are done.

**Horizon (the repair clock, 2026-10-06; `run(horizon=h, cap_s=c)`, both optional — without them the call is exactly
the one above).** A lane still searching after h steps stops with **status 4 (`ST_HORIZON`)** and keeps its current,
projected x (as a stalled lane does). The rule is checked after done / budget / stall, so a lane that would stop at
step s ≤ h stops exactly as without the horizon: a horizon-h call is a prefix of the unbounded call (same x, status,
steps and moves for every lane with s ≤ h; tested on the CPU with the cutoff armed — the GPU twin of the test is
written, marker `gpu`, and has NOT run yet). h = 0 = projection and the stop rules only.
- CPU (`ls_fj_cpu`): one scalar `max_lane_steps` in the Numba kernel (2^62 without a horizon).
- GPU (`ls_fj_gpu`): host side only, no kernel change. An active block runs exactly `iters` steps per launch, so after
  every launch all active lanes have the same step count; the host truncates the next launch to min(iters, h − steps
  so far), stops relaunching at h and gives the still-active lanes status 4 on the host (STI is not rewritten).
- Call deadline `cap_s` (a backstop). GPU: split over the chunks (chunk c of C: t0 + cap·(c+1)/C); when it passes the
  host stops relaunching and the active lanes get status 4 at the equal step count reached (H_eff); a chunk reached
  after its deadline is still projected (prep, init and a launch of 0 steps — the stop rules only). CPU: a Numba call
  cannot be interrupted, so the rows are processed in slices and the rows NOT STARTED at the deadline are only
  projected (horizon 0). The two differ in what is cut (the chunk in progress vs the rows not started); documented,
  and the CPU is not a timed arm of the repair clock.
- A lane cut by the horizon or the deadline counts in `finished_budget` (so the three finish counters still add up to
  k) and in the new counter `finished_horizon`; `cap_hits` counts the chunks (GPU) or the call (CPU) the deadline
  cut. Both counters appear only when `horizon` or `cap_s` is given.
- Every call keeps `last_steps` (steps per lane) and `last_profile`: GPU `{kind: "launches", seconds, H, cap_s,
  chunks: [{k, cum, t, end, H_eff, done, cut, cap_hit}]}` — the cumulative steps after every launch with the seconds
  since the call started, recorded whatever the arguments (one clock read per launch); CPU `{kind: "sequential",
  seconds, H, cap_s, cap_hit, cut_rows}`.

## Invariants (tested)
Integer genes stay integral and inside bounds; `act` equals `A x` after the call (to 1e-9 relative); a feasible input
with the cutoff inactive is returned unchanged; on toy knapsack / set cover / mixed models from random starts most
children reach feasibility within the step budget; GPU and CPU reach feasibility on the same toy instances (rates
within noise).

## Counters returned
`moves, local_minima, steps, finished_feasible, finished_stall, finished_budget, seconds` (+ `chunk` and peak memory
on the GPU; + `finished_horizon` and `cap_hits` when a horizon or a deadline is given). Status codes per lane: 1
feasible, 2 stall, 3 budget, 4 horizon (cut by the horizon or the deadline), 0 interrupted by a run-level stop. The
engine adds them to the generation record.
