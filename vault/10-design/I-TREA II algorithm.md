# I-TREA II — the algorithm (design spec, 2026-10-05)

Implementation: `src/itrea/`. Every number below is a parameter in `itrea.config.EAConfig` (YAML in `configs/`);
the defaults are starting points to be tuned on the tuning set only ([[D-002 Pre-registration — hypotheses, arms, analysis]]).

## Problem form
`min  c·x   s.t.  rl ≤ A x ≤ ru,  lb ≤ x ≤ ub,  x_j ∈ ℤ for j ∈ I` — maximisation is converted by `sense`; the
population compares **keys** (`key = sense · objective`, lower is better). Read from MPS with HiGHS
(`itrea.problem.Problem`), canonical variable order = HiGHS's column order. Infinite bounds are `±inf`.

Row violation of individual x, row i: `v_i = max(rl_i − a_i, 0) + max(a_i − ru_i, 0)` with `a = A x`, normalised by
the row's 2-norm. **Total violation** `V(x) = Σ_i v_i/‖A_i‖ + Σ_j bound violations + Σ_{j∈I} |x_j − round(x_j)|`.
**Feasible** = every row and bound within `feas_tol` relative to `max(1, |rhs|)` and every integer within `int_tol`
(SCIP-style per-row test — never a sum against a total tolerance, the X2 bug).

## Population and the three ranks (I-TREA's names; mapping in [[D-004 Mapping I-TREA (2015) to I-TREA II]])
Population X (P × n) on the device, plus per-individual `key`, `viol`, `feasible`, `rank ∈ {0,1,2}` (−1 = not yet
ranked), `id` (age: lower = older), `parent ids`, `origin` (operator bits), `birth`, ancestry bit masks. Survival each
generation, from the candidate set C = population ∪ offspring ∪ injected (`itrea.ea.ranks.three_rank_survival`), in
part I's elimination order:

1. **Duplicates** (identical vectors) out.
2. **Incumbent key** `inc = min(best feasible key in C, solver's incumbent key known to the EA)`.
3. **Rank 1 — feasible good (protected):** the best `ceil(y · basis)` feasible individuals by key (`elite_basis`:
   `feasible` = #feasible, default; or `population` = P); with `rank1_twins: drop`, equal-key twins are not Rank 1
   (they compete in Rank 2). At least one when any feasible exists and y > 0.
4. **Rank 0 — infeasible good:** infeasible with `key < inc − imp_tol·max(1,|inc|)`; with `rank0_bound_demote`,
   those with key beyond the solver's **proven** bound are eliminated (the solver's answer replaces part I's
   decision maker). At most `floor(x · P)`; over the quota, `rank0_select`: `itrea` (non-dominated fronts in (key, V);
   the overflowing front thinned by repeatedly removing the point nearest to another in normalised (key, V), tie →
   worse key out), `violation`, `stochastic_rank` (the optopt X2 rule).
   **No incumbent yet** (`inc = +inf`): `no_incumbent_policy: violation` (default — feasibility phase, everyone ranked
   by (V, key) as Rank 2; a deviation from part I) or `itrea` (everyone is infeasible good; Rank 0 selection, no quota).
5. **Rank 2 — rejects:** the rest fill to P, `rank2_select`: `itrea` (non-dominated sorting in (key, V), crowding
   distance on the overflowing front, ties → older out) or `key_then_violation`.

Invariants (property-tested): |survivors| ≤ P; Rank 1 ⊆ feasible and holds the best feasible keys (up to twins);
|Rank 0| ≤ floor(x·P) when an incumbent exists; every Rank 0 member is infeasible with key < inc; a Rank 1 member of
generation g is in generation g+1 unless displaced by a better feasible individual (or a twin rule).

**Solver rejections:** an EA solution that the solver refuses (`trySol` false although our float64 check passed) is
marked infeasible in the population (it moves to Rank 2) and counted — part I's "deemed inferior by the decision maker".

## Generation
1. **Exchange in** (every `exchange_poll_s`): the solver's incumbent, its pool ("very good" = within `inject_band`
   relative gap of its best, at most `inject_k`), its dual bound and (once) its root LP solution. Injected vectors are
   evaluated and enter C (as part I's population takes in new members, they are ranked by the same rule).
   With `lp_stream.enabled` ([[D-011 Solver streams good infeasible solutions to the EA]]) also the solver's stream of
   good infeasible node LP points: each new point is rounded to the nearest integer point and evaluated; the best
   `lp_stream.pool` points are ranked by (V of that rounding, key) — Berk's order, least infeasible first — and each
   unused point, best first (at most `per_tick` per tick), gives `roundings` children (the nearest rounding +
   randomised ones) that are repaired and join C like any child: Rank 0 takes the super-optimal infeasible ones only
   under its cap and selection rule. Stagnation LP-rounding immigrants then draw their LP point from the pool's best
   four, round robin.
2. **Parent selection:** tournament of size `tournament_k` with part I's comparator (`itrea`: Rank 1 < Rank 0 <
   Rank 2, ties by key then V), or `deb` / `stochastic` (stochastic ranking, `p_f`). Optional per-rank parent pools
   (`parent_rank_weights` = [Rank 0, Rank 1, Rank 2]; default one pool, as in part I where every member can enter the
   mating pool). Matings across ranks (Rank 0 × Rank 1) are recorded in the ancestry bits.
3. **Crossover** with probability `pc`, operator chosen by `crossover_weights`: `uniform(p_swap)`, `k_point(k)` (on the
   variable order), `block` (variables of `block_rows` random constraints taken from the second parent — constraint-
   aware), `none`.
4. **Mutation**, operator by `mutation_weights`: `flip` (binaries), `step` (±1…±`step_max` for general integers),
   `reset` (uniform in bounded domains), `toward_lp` (rounded LP-relaxation value, if known), `violation_guided`
   (variables in violated rows — part I's Second Chance analogue), `objective_guided` (moves that improve the
   objective, to be repaired — part I's Forced Closure analogue). Expected `mutation_rate · |I|` genes (≥ 1) per child.
5. **Repair / local search** (`repair: none | greedy | fj`), `ls_steps` steps, objective weight `ls_obj_weight`:
   batched over the population on the device. One step: compute activities `A Xᵀ` (SpMM), for every non-zero (i, j) the
   change of row i's weighted violation if x_j moves by its candidate step (binary flip; ±1 for general integers),
   reduce by column (sparse indicator SpMM) → score per (individual, variable); apply each individual's best improving
   move (`ls_moves_per_step` ≥ 1). `fj` adds feasibility-jump style row weights (Luteberget & Sartor 2023) that grow on
   rows violated at a local minimum. Work per step O(nnz · P), chunked over the population to stay inside the GPU
   memory budget (`gpu_mem_gb`).
6. **Continuous variables** (`repair.continuous: auto | gene | lp_decode`): `auto` (default) sets singleton
   continuous columns in closed form on the device and moves every other continuous variable by feasibility-jump
   moves; `gene` treats them as real genes (moves to make a violated row tight); `lp_decode` ([[D-017 LP decode —
   re-optimise the continuous variables of selected candidates]], issue #27; built 2026-10-08, two rounds of
   review fixes 2026-10-09, not yet run on an instance) is `auto` plus an LP decode. A persistent HiGHS worker
   process (1 thread, niced, started at t0) fixes the integer part of a few candidates and solves the LP over the
   continuous variables.
   Every generation, after the survival, over the whole candidate set, it is offered at most one batch (one request
   outstanding at a time): feasible candidates by key to **polish** (only when some continuous column carries cost)
   and infeasible candidates better than the incumbent, in D-011's (V, key) order, to **complete**
   (`lp_decode_infeasible`); the worker solves at most N = `lp_decode_per_gen` (4) of them, 2 + 2 when both kinds
   exist, after a cache and a range screen. A polish copy is kept only if its key beats its source's by `imp_tol`; a
   completion copy always. The copies (source's integer part + the LP's continuous values) enter a later candidate
   set as ordinary rows — at most one decoded batch per candidate set — with origin `ORIGIN_LP_DECODE`, the source's
   strategy and ancestry, and pass the same float64 re-check and survival. The worker is polled at the exchange tick,
   just before the candidate set takes its pending rows, and at the pick site. A reply that arrives before the next
   generation's candidate set (the worker answered within about a generation) is read there and its copies join that
   set: one generation after the offer. A reply that arrives later, while that set is assembled or survives, is read
   at the pick site; its copies join the following generation's set (two generations after the offer), and that pick
   site sends no new request, so the worker idles one generation and its next reply is again read before a candidate
   set (sending one in the same tick would keep every later reply two generations late). Inert
   (no worker) on an instance without an active column (continuous, lb < ub, in two or more rows). Trace fields:
   [[Trace format and metrics]] ("LP decode").
7. **Evaluate** offspring on the device (`eval_dtype`: float64 default; float32 measured in E1 for speed and its
   false-feasible rate), **survive** (three ranks), record lineage.
8. **Exchange out:** when the best feasible key beats the solver's incumbent by `post_min_gain`, re-check it on the CPU
   in float64 (per row) and post it. The solver adds it as a new incumbent / starting point (SCIP: heuristic plugin,
   `trySol`; HiGHS: user-solution callback). Every post and the solver's verdict are logged.
9. **Stagnation (trigger):** after `stagnation.gens` without a new best, replace `immigrant_frac` of Rank 2 with
   immigrants (random or LP-rounded); Rank 0 and Rank 1 are untouched (part I called itself "overly elitist").

## Self-adaptive repair strategy (2026-10-06, development; `strategy`, off by default)
**Why.** Feasibility probe 1 ([[Development log]], 2026-10-06): an LP point after t0 made the EA alone feasible on the
UC case and sing44, the rhs row scale did so on glass4 and fhnw-binpack4-48, and each setting hurt where the other
helped. Instead of choosing one per instance, the population carries both and selection decides — self-adaptation of a
strategy parameter as in evolution strategies, applied to the repair. Untested on the development set so far.

**What an individual carries.** A strategy index into `row_scales × lp_guidance` (default `[l2, rhs] × [off, on]`, 4
strategies, scale-major order). The row scale is the divisor s_i of the feasibility-jump violation in the repair
([[Feasibility jump local search]]); LP guidance decides whether its mutation may use the LP point (`toward_lp` with its
configured weight) once one exists. Nothing else differs: evaluation, the three ranks, selection, crossover, the
exchange and the frontier are the same for every strategy, and **V stays on the one scale `ranks.viol_norm`** for
everyone, so violations remain comparable across strategies and the survival is not biased towards a scale.

**Inheritance.** The initial population gets the strategies round robin. A child takes its first parent's strategy;
with probability `mutation_rate` (0.1) it is redrawn uniformly. Every strategy keeps at least
min(⌈`min_share`·O⌉, ⌊O/S⌋) of the O children (`min_share` 0.05): a strategy below that floor is topped up with
children drawn at random from the strategies above it, so a strategy that lost the population can return when the
instance enters another phase (e.g. after the first feasible solution, when the objective cutoff row is armed).
Immigrants and injected solver solutions draw uniform random strategies, each row independently (injected rows arrive
best first; a sorted draw — the first implementation — gave the best solver row the smallest of k draws, strategy 0
about 90 % of the time, review 2026-10-06); LP-rounded individuals (stagnation immigrants, the batch made when the LP
point arrives) are given only to strategies with LP guidance — none if no strategy has it.

**Per generation.** After parent selection the children's strategies are drawn; nothing is sorted or grouped. ONE
mutation call serves every child: a child whose strategy has no LP guidance draws its operator from the weights without
`toward_lp` (also when `violation_guided` re-draws it). ONE local-search call repairs every child, each with its own
row scale: the search keeps a row of divisors per scale (R × (m + 1) floats — the only per-scale data; the memory plan
counts the extra rows, `local_search_strategies`) and reads the row of each individual's scale. On the GPU this keeps
the individuals of all scales side by side, so a generation pays one slowest-individual tail; a call per scale, one
after the other, cost +13–39 % per generation with two scales (review 2026-10-06), which would have biased a
fixed-budget comparison of the adaptive arm against fixed-strategy arms. The objective cutoff is shared by all scales.

**Records.** Genealogy column `strat`; `best` events carry `strat` and `strat_credited`; `gen` events add survivors,
credited new bests and children per strategy; the summary holds the cumulative counts per strategy (children, best
events, incumbents, rows repaired and rows the repair finished feasible) — [[Trace format and metrics]]. A solver
solution that becomes the population best (injected, no EA operator bit — the incumbent's source "solver") is not
credited to the strategy its row was given: it is counted separately (`incumbents_solver`, `best_events_solver`), so
the per-strategy counts show what the EA's own strategies found. **Analyses of LP guidance must key on `strat`, not on
origin bits:** a genealogy row's `origin` is OR-merged over exact duplicates, so a row whose strategy has no LP
guidance can carry the `toward_lp` bit of a duplicate made by an LP-guided child (no such bit is ever made in a no-LP
child itself). With `strategy.enabled: false` the run is identical to one without the feature (regression test against
a run recorded before it) and the configuration hash is unchanged.
Configuration: `configs/itrea2-adaptive.yaml` = the defaults + `strategy.enabled` + `seeding.lp_source: highs_async`
(the EA alone gets an LP point too).

## Two-scale survival in the feasibility phase (2026-10-06, development; `ranks.feas_alt_scale`, off by default)
Before any feasible solution is known, survival ranks every candidate by V on the one scale `ranks.viol_norm`. With
the self-adaptive strategy, children repaired on the rhs scale are then judged on l2 — on fhnw-binpack4-48 they died
out (8–9 of 256) although rhs repair + rhs survival reaches the optimum there in 30 s (feasibility probe 1, c04).
With `ranks.feas_alt_scale` set (e.g. `rhs`), the feasibility phase keeps the non-dominated fronts over
(V on `viol_norm`, V on the second scale), crowding on the overflowing front — each scale's least-violated candidate
survives and neither scale's numbers are compared with the other's (the consult's second choice; protected per-scale
demes were its first, kept for later together with region islands). Costs one extra evaluation of the candidate set
per generation, in the feasibility phase only; outside it the option does nothing. Left out of the configuration hash
while `none` (`_LATE_DEFAULTS`).

## Repair clock (2026-10-06, development; `repair_clock`, off by default; built to Stage 1 = shadow mode)
**Why.** The repair depth is an instance-dependent trade-off: deep repair (`lps1`) won on 6 of 9 development instances,
the default depth only on item_placement, where many fast generations pay ([[Development log]]); and a GPU repair call
waits for its slowest lane, so mixing depths inside one call would not make a generation faster. Design, judges and
plan: [[2026-10-06 Repair-depth design (workflow)]].

**What.** One level per generation for every repair call of it: a horizon h = 2^7 … 2^16 lane steps, or "none" (the
configured repair). A lane still searching at h stops with status 4 and keeps its x ([[Feasibility jump local
search]], "Horizon"); on the GPU the host truncates the launches, so the cut is the same step count for every lane and
every call can be replayed from its logged H_eff. Until the population holds a feasible member of the EA's own (a
feasible member that is not an unchanged solver solution) every call is played at "none" without a cap (deep repair
keeps its earlier first feasible solutions); neither a finite solver incumbent nor an injected feasible solver row
ends this phase. After it, the controller (`ea/repair_clock.py`, no random numbers) reads each level's value from the
calls it just ran: a horizon-h call is a prefix of the unbounded call, so one call reveals the gain of every shallower
level (the largest improvement over the cutoff among the lanes that finished feasible within h steps, dg = min((cutoff
− key) / max(|cutoff|, |key|, |dual| if finite), 1) — in [0, 1] like P(T)'s gap; below 1e-3 it counts as
zero) and its cost from the measured per-launch (steps, seconds) profile. This is exact for the children, the pending
batches and an admit before the children; the immigrant calls after the survival (shortfall fill, stagnation) exist
only because of the generation's earlier calls, so their gain is logged apart and their time counts as overhead.
Gains and times decay with a 20 s half-life; the controller switches only after 5 observed generations and 2 material
gains, to a rate above 1.25 × the current level's, plays "none" with no evidence, re-measures "none" in probes (at most
10 % of the time), always repairs the one-shot LP batch at "none", and caps each call at min(10 s, 3 × the level's
median children-call time) (10 s at "none"); a children call the cap cut enters that median as a censored time, so the
cap can rise when the calls become slower.

**Deviations from the synthesis plan (review of 2026-10-06 evening; D-012 must carry them).** (1) The gain unit: the
plan's max(|cutoff|, |dual|, 1e-9) gave one improvement over an incumbent near 0 without a dual bound a weight of
~1e9 (neos-960392 has key 0 and no dual at generation 0); the point itself is now in the denominator and dg ≤ 1. (2)
The phase test is a feasible member of the EA's own, not `h_feas.any()`, which an injected solver row satisfies. (3)
The immigrant calls after the survival are left out of the gain (the plan claimed exactness only for the children and
the pending batches). (4) The cap's median takes censored times from cut calls. (5) Shadow mode accumulates only what
its decision would have revealed (below).

**Stages.** Stage 0 (built 2026-10-06, tested on the CPU only; not yet run on an instance): the horizon in both
feasibility-jump implementations (the GPU wrapper's tests are written, marker `gpu`, and wait for a GPU window), the
controller, the engine wiring and their tests. Stage 1: **shadow mode** (`mode: shadow`,
`configs/itrea2-adaptive-lps-rc.yaml` = lps1 + the clock in shadow) plays "none" with no cap — the search is lps1's:
no argument is added to a local-search call and no random number is drawn, so the genealogy is identical for the same
generation count (tested on NumPy, also with the controller held at a shallow level). In a timed run the extra host
work (the key copy-back, one `rclock` event per generation) can shift generation counts and exchange timing, which
Stage 1 criterion (c) checks, and its configuration hash differs from lps1's: its runs are not lps1 runs. It logs per
generation what it would play, every call's profile and each lane's steps and status (`rclock` events, [[Trace format
and metrics]]). The played calls show every level, but shadow mode accumulates only what its decision would have
revealed (the levels ≤ its level, or every level if no lane ran beyond it), so its probes and its evidence are the
active controller's; the event keeps the full information and marks the levels used. The active controller (Stage 2,
`mode: active`) only if the shadow signal passes the criteria stated in the design note. Before Stage 2: the
`rclock` event carries every lane's steps and status every generation (about 5.5 KB per generation at P = O = 256,
~24 MB per 300 s item_placement run) — thin the per-lane payload in active mode if long runs need it. Off by default
and left out of the configuration hash while disabled.

## Seeding
`seed_mix`: random in bounds (integer-rounded), LP-relaxation rounding (HiGHS root LP on the CPU, before the clock,
counted in the start-up record), and the solver's incumbents as they arrive.

## What is logged
Per generation (every `log_every`): best/median key, feasible share, |Rank 0|, |Rank 1|, |Rank 2|, inc,
offspring/s, GPU memory peak, Rank 0 candidates and demotions (with the self-adaptive repair strategy also the counts
per strategy). Per event: every new population best with its lineage; every post and verdict; every injection received.
Trace = JSON lines + a summary JSON (schema in [[Trace format and metrics]]).
