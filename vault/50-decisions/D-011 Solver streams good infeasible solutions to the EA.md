# D-011 Solver streams good infeasible solutions to the EA; develop on the coupled system (2026-10-06 15:2x, Berk)

**Berk (15:1x–15:2x):** "You are benchmarking EA but our EA shall be coupled with a solver similar to what cuopt does."
Then: "The basic mechanism I have on my mind is when EA finds a new feasible best solution it shall feed to solver and
when solver comes up with suitable good infeasible solutions (minimize infeasibility then maximize objective function)
it might be able to inject to EA's population. We also keep a percentage of good feasible solutions as well as the best
solution. We are aiming for more diversity in EA."

**Status:** decided (development, D-010). Built and tested the same day (16 tests; a 30 s CPU smoke run on
radiationm18-12-05 with `tools/lps_smoke.py`: SCIP read 19 points in 0.56 s of 30 s, the EA ranked and used 18 of them,
72 children; none feasible after repair, none in Rank 0 — that instance's equality chains, §4). First GPU measurement
pending ([[Development log]]).

## 1. Yardstick: the coupled system (amends D-010 §3)
Development changes are judged on the **coupled arm A3** (portfolio P(120) against SCIP alone A0 of the same seed and
cuOpt alone A6), with the EA's contribution counters (posts / accepted, SCIP-held vs EA-held incumbents). The EA alone
(A5) is a **diagnostic only**, run when a coupled result needs explaining — not a development target. The adaptive
strategy runs of 15:07 were cut to A3 only at 15:15 (`jobs/dev-adaptive1-a3.jsonl`).
Note (from the novelty record, [[E Novelty check 2026-10-05]]): cuOpt's own coupling is the same two-way exchange — its
GPU population takes B&B solutions and hands its bests to B&B. What differs here is the external general solver and the
rank rules; the coupling by exchange stays the mechanism.

## 2. What already existed
- EA → SCIP: every new EA incumbent (and near-best feasible ones) is tried in SCIP through the heuristic plugin and
  becomes SCIP's incumbent / starting point when SCIP accepts it (glass4: 213 of 228 posts accepted).
- Rank 1 keeps the best y % of the feasible plus the best (I-TREA's protected rank).
- SCIP → EA: feasible solutions (incumbent + pool band), the dual bound, and **once** the root LP point.

## 3. What is new: the LP stream (SCIP → EA, good infeasible points)
**Source.** Node LP optima of SCIP's own tree (AFTERLPNODE; DURINGLPLOOP for the cut rounds; never in probing). Each is
super-optimal (an open node's LP value beats the incumbent, else the node is pruned) and comes from a different part
of the tree — diversity from the solver's branching, which the EA cannot make itself. The root LP point alone already
made the EA feasible on two instances (feasibility probe 1, c03: ca__Scenario400_reserves_1 and sing44).
Rejected sources (reviewed with Hermes Agent with OpenAI GPT-6.1 Sol, [[2026-10-06 LP stream design consult (GPT-6.1 Sol)]]): diving end points (no
Python access; probing state), pseudo-solutions (very row-infeasible, little variation), trySol rejects (EA-origin, no
new diversity; "not stored" ≠ infeasible).

**Berk's order — minimise infeasibility, then optimise the objective — applied twice:**
- in SCIP, cheaply: an LP whose value reaches the cutoff bound is skipped (not super-optimal); a node is "good" when its count of fractional integer variables (`getNLPBranchCands`, free) is at
  most the 25 % quantile of the last 32 LP solves; integral LP points are skipped (SCIP checks them itself);
- in the EA, exactly: each point is rounded to the nearest integer point and evaluated on the GPU; the pool (16 points)
  is ranked lexicographically by (V of that rounding, key) and used best first.

**Cost bound.** Reading a point costs ~0.4 µs per variable in PySCIPOpt (no vectorised accessor; measured 2026-10-06:
16.5 ms on radiationm18-12-05, ~50 ms on rail01). A token bucket keeps the stream's read + post time under 2 % of the
wall clock since t0 (plus a start allowance of two reads), with ≥ 0.25 s between reads and a 1 s patience so a poor
dive cannot starve the stream. SCIP's summary records `lp_stream.spent_s`; the coupled arm pays it — it is part of the
coupling's cost, against a harness-matched A0.

**Admission.** Each used point gives 4 children: the nearest rounding + 3 randomised roundings, repaired by the
feasibility jump, then **ordinary survival only** — a super-optimal infeasible child competes for Rank 0 under its x·P
cap and NSGA-II selection, never bypassing them; a feasible one follows the incumbent / Rank 1 rules. At most 2 points
per exchange tick. Stagnation immigrants take their LP point from the pool's best four, round robin, instead of always
the first LP point. A stream point never becomes the EA's LP point (`lp_x` keeps `seeding.lp_source`). The pool keeps
16 points: used points leave first (except the best four), so new points keep giving children even when they are
worse than the best seen.
Marking: origin SEED_LP | INJECTED | IMMIGRANT (a combination no other path produces — the uint16 origin has no free
bit); the generation record counts `rank0_stream` (Rank 0 members born from the stream) to see whether it floods Rank 0.

**Transport.** A new channel `lps` (file `to_ea_lps.npz` / Redis `ch:lps`, history 8): every post is a snapshot of
SCIP's last 3 points with their ids, so an EA that misses a post loses nothing.

**Configuration.** Section `lp_stream` (off by default; left out of the hash while off — every existing hash unchanged).
`configs/itrea2-adaptive-lps.yaml` = the adaptive configuration + the stream (hash `a2ad94e9188d`). SCIP only (the
runner refuses a HiGHS partner).

**Review of 770df17 (same day, before any GPU run; fixed in the next commit).** (1) Adopting the first stream point
as the EA's LP point raced the HiGHS worker (`lp_source: highs_async`) — the LP source depended on timing and differed
between arms; removed. (2) The validation rule (stream needs the exchange) blocked every no-exchange arm (A3r, A5, A5x0,
A5y0); `_NOX` now switches the stream off too. (3) Used points never left the pool, so a full pool could silence the
stream; fixed (above). (4) `nodeinfeasible` and the cutoff were ignored; now a cut-off node or an LP at the cutoff is
skipped. (6) The budget charged only reads and posts; now every decision is charged (try/finally), and the quantile is
a 1 µs nearest-rank instead of a 33 µs `np.quantile`. (7) Ids skipped between two polls are counted (`lps_missed`).
Not changed: duplicate merging can OR the stream bits into an EA-bred member (`rank0_stream`/`final_stream` may
over-count slightly; P(T) and the attribution in `build_runs` do not read these bits); rounding before clipping, as in
`_receive`. Checked and fine: getSolVal(None, ·) over original variables at both timings, SCIP's search unchanged with
the stream on (identical nodes, LP iterations and callbacks on the toy), disabled configurations bit-identical.

## 4. What it will not fix
Better seeds give no coordinated moves: instances with long equality chains (rail01's 45.5k conservation rows;
radiationm18-12-05's cont = int, cont = bin, b1 + b2 = 1, i1 = i2 − i3) stall the single-variable feasibility jump
whatever the start. On rail01 SCIP does not finish the root LP in 60 s, so no node LP exists there within the budget.
Coordinated moves (equality-aware jumps, propagation) are the separate next item.

## 5. Measurement (development)
A3 with `itrea2-adaptive-lps` + the c01 repair budget (`repair.max_steps 20000, repair.stall_steps 1000`) against A3
with `itrea2-adaptive` + the same budget (the 15:15 runs), 9 instances × seeds 0–1, 120 s. Report P_port, SCIP-held vs
EA-held, `lp_stream` counters on both sides, `rank0_stream`, SCIP throughput (nodes/s) and `spent_s`. A
capture-and-discard arm (cost alone) only if the stream's effect is ambiguous.

## Next (Berk's question 15:3x, "Can EA be targeted to search large chunks of solutions space via GPU?")
Region islands: SCIP sends with each point the node's local bounds, and a sub-population searches inside that region
— many regions at once on the GPU, while SCIP works one node at a time. Worth building only after the stream shows
that solver-chosen points help, and together with coordinated moves (a region does not remove equality chains).
