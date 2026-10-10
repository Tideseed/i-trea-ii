# I-TREA II — the algorithm (design spec, 2026-10-05; updated 2026-10-10 to the built code)

Implementation: `src/itrea/`. Every number below is a parameter in `itrea.config.EAConfig` (YAML in `configs/`);
the defaults are starting points to be tuned on the tuning set only ([[D-002 Pre-registration — hypotheses, arms, analysis]]).

## Problem form
`min  c·x   s.t.  rl ≤ A x ≤ ru,  lb ≤ x ≤ ub,  x_j ∈ ℤ for j ∈ I` — maximisation is converted by `sense`; the
population compares **keys** (`key = sense · objective`, lower is better). Read from MPS with HiGHS
(`itrea.problem.Problem`), canonical variable order = HiGHS's column order. Infinite bounds are `±inf`.

Row violation of individual x, row i: `v_i = max(rl_i − a_i, 0) + max(a_i − ru_i, 0)` with `a = A x`, normalised by
the row's scale `s_i` (`ranks.viol_norm`: `l2` = the row's 2-norm, default; `rhs` = max(1, |rhs|)).
**Total violation** `V(x) = Σ_i v_i/s_i + Σ_j bound violations + Σ_{j∈I} |x_j − round(x_j)|`. With
`viol_norm: count` the row term is instead the number of violated rows (one per violated row, whatever its size) and
the repair uses `s_i` = 1.
**Feasible** = every row within `feas_tol` (1e-6) under the partner solver's rule (`ranks.tol_mode`: `scip`, default,
relative to max(|a_i|, |rhs|, 1); `highs`, absolute; `rhs`, relative to max(1, |rhs|)), every bound within `feas_tol`
relative to max(1, |bound|) (absolute under `highs`) and every integer within `int_tol` (a per-row test — never a sum
against a total tolerance, the X2 bug).

## Population and the three ranks (I-TREA's names; mapping in [[D-004 Mapping I-TREA (2015) to I-TREA II]])
Population X (P × n) on the device, plus per-individual `key`, `viol`, `feasible`, `rank ∈ {0,1,2}` (−1 = not yet
ranked), `id` (age: lower = older), `parent ids`, `origin` (operator bits), `birth`, ancestry bit masks. Survival each
generation, from the candidate set C = population ∪ offspring ∪ injected (`itrea.ea.ranks.three_rank_survival`), in
part I's elimination order:

1. **Duplicates** (identical vectors) out.
2. **Incumbent key** `inc = min(best feasible key in C, solver's incumbent key known to the EA)`.
3. **Rank 1 — feasible good (protected):** `ranks.rank1_mode: top` (default): the best n1 feasible individuals by
   (key, age), n1 = max(`ceil(y · basis)`, the previous Rank 1 members still among the feasible candidates after the
   twin rule) (`elite_basis`: `feasible` = min(#feasible, P), default; or `population` = P). The second term is a
   ratchet: a previous member leaves Rank 1 only when n1 better feasible candidates precede it, never because the
   number of feasible candidates fell. With `rank1_twins: drop`, equal-key twins are not Rank 1 (they compete in
   Rank 2). At least one when any feasible exists and y > 0.
   `rank1_mode: clearing` ([[D-016 Rank 1 as a distinct elite — cuOpt's clearing rule, development test]], issue #25;
   built 2026-10-08; **not adopted** 2026-10-09: A3@cl1 vs A3@rc1d over seeds 0–3 (68 runs; 102 with the size-only
   control A3@tp1), mean d −0.033 [−0.122, +0.050] — the development default stays `top`): cuOpt's population rule
   (`population_t::add_solution`, NVIDIA/cuopt dfbbcc5a) applied best-first as clearing (`ea/elite.py`,
   `clearing_pick`). The feasible candidates are walked in (key, age) order; one joins Rank 1 when it differs from
   every member already chosen in at least k integer variables (values rounded; continuous variables do not count),
   up to `rank1_size` members (0 = `ceil(y · P)`). The best feasible candidate always joins; a cleared candidate
   competes in Rank 2 if it is new and dies if it is old (generational Rank 2). k = `rank1_min_diff`, or with 0
   cuOpt's threshold from the number of integer variables n_int: 1 / 2 / 4 / 8 / 10 for n_int < 50 / 80 / 200 /
   1000 / beyond. No twin rule and no ratchet (the cap is fixed). Without integer variables the engine runs `top`
   and records that; k > n_int is refused at start-up. Distances are computed on the device in chunks of a 64 MB
   comparison budget and copied to the host in blocks of 32 rows. The three fields are in `_LATE_DEFAULTS`, so every
   recorded hash stands at their defaults. Rank 1's size and spread are logged in both modes ([[Trace format and
   metrics]], `r1_*`).
4. **Rank 0 — infeasible good:** infeasible with `key < inc − imp_tol·max(1,|inc|)`; with `rank0_bound_demote`,
   those with key beyond the solver's **proven** bound are eliminated (the solver's answer replaces part I's
   decision maker). At most `floor(x · P)`; over the quota, `rank0_select`: `itrea` (non-dominated fronts in (key, V);
   the overflowing front thinned by repeatedly removing the point nearest to another in normalised (key, V), tie →
   worse key out), `violation` (ascending V), `rank_prob` (rank-weighted sampling without replacement, the optopt X2
   idea).
   **No incumbent yet** (`inc = +inf`): `no_incumbent_policy: violation` (default — feasibility phase, everyone ranked
   by (V, key) as Rank 2; a deviation from part I) or `itrea` (everyone is infeasible good; Rank 0 selection, no quota).
5. **Rank 2 — rejects:** the rest fill to P, `rank2_select`: `itrea` (non-dominated sorting in (key, V), crowding
   distance on the overflowing front, ties → older out) or `key_then_violation`.

Invariants (property-tested): |survivors| ≤ P; Rank 1 ⊆ feasible and, in `top` mode, holds the best feasible keys
(up to twins); |Rank 0| ≤ floor(x·P) when an incumbent exists; every Rank 0 member is infeasible with key < inc; in
`top` mode a Rank 1 member of generation g is in generation g+1 unless displaced by a better feasible individual (or
a twin rule). Every new Rank 1 entrant is re-checked on the host in float64; a failure is demoted and the survival
repeated until Rank 1 holds only re-checked members.

**Solver rejections:** an EA solution that the solver refuses (`trySol` false although our float64 check passed) is
marked infeasible in the population (it moves to Rank 2) and counted — part I's "deemed inferior by the decision maker".

## Generation
1. **Exchange in** (every `exchange.poll_s`, 0.25 s): the solver's incumbent, its pool ("very good" = within
   `exchange.inject_band` (1 %) relative gap of its best, at most `exchange.inject_k` (8)), its dual bound and (once)
   its root LP solution. Injected vectors are evaluated and enter C, never repaired (as part I's population takes in
   new members, they are ranked by the same rule). In arm A9 (`exchange.from_cuopt`, [[D-015 Use cuOpt's feasible
   solutions — three-way arm A9]]) also cuOpt's incumbents, from its own channels and injected the same way; cuOpt's
   incumbent joins the external incumbent (the min of the two) for Rank 0's threshold and the repair cutoff, its
   bound is recorded and not used ([[Process interface]], "Three-way arm A9").
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
3. **Crossover** with probability `crossover.pc` (0.9), operator chosen by `crossover.weights` (default `agreement`
   0.6, `uniform` 0.2, `block` 0.2): `agreement` (the child keeps the genes on which both parents agree; the others
   come from either parent and are marked), `uniform(p_swap)`, `k_point(k)` (on the variable order), `block`
   (variables of `block_rows` random constraints taken from the second parent — constraint-aware), `none`. The marked
   genes (`touched`) are not passed to the repair: restricting it to their rows is not wired.
4. **Mutation**, operator by `mutation.weights`: `flip` (binaries), `step` (±1…±`step_max` for general integers),
   `reset` (uniform in bounded domains), `toward_lp` (rounded LP-relaxation value, if known), `violation_guided`
   (variables in violated rows — part I's Second Chance analogue), `objective_guided` (moves that improve the
   objective, to be repaired — part I's Forced Closure analogue). Poisson(`mutation.genes` = 4) genes per child, at
   least `min_genes` (1); `mutation.rate`, if set, replaces it by a share of the integer genes.
5. **Repair / local search** (`repair.method`: `fj` default | `dense` | `none`; spec [[Feasibility jump local
   search]]): `fj` = feasibility jump (Luteberget & Sartor 2023) — weighted row violations, one variable moved per step
   to a jump value, row weights that grow at local minima — on the GPU a CuPy `RawKernel` with one thread block per
   individual, on the CPU a Numba reference; at most `max_steps` moves per child (2,000; the development
   configurations 20,000), stopped after `stall_steps` steps without a decrease of the weighted violation (100;
   development 1,000). With an incumbent the objective cutoff row `cmin·x ≤ inc − δ` joins the weighted rows
   (`objective_mode: cutoff_row`). `dense` (a baseline only) rescores every non-zero per step (`steps`,
   `moves_per_step`). Children are repaired in chunks inside the local search's share of the GPU memory budget
   (`gpu_mem_gb`).
6. **Continuous variables** (`repair.continuous: auto | gene | lp_decode`): `auto` (default) sets singleton
   continuous columns in closed form on the device and moves every other continuous variable by feasibility-jump
   moves; `gene` treats them as real genes (moves to make a violated row tight); `lp_decode` ([[D-017 LP decode —
   re-optimise the continuous variables of selected candidates]], issue #27; built 2026-10-08, two rounds of
   review fixes 2026-10-09) is `auto` plus an LP decode. **Adopted 2026-10-09 as the A3 development default**
   (`configs/itrea2-dev-rc-ld.yaml`, effective hash f030e23909ad): A3@ld1 vs A3@rc1e on the 6 development instances
   with an active column × seeds 0–1, mean d +0.219 [+0.041, +0.429], 5 better / 1 worse ([[Development log]], "LP
   decode — D-017 result"); the code default stays `auto`, and the runner refuses `lp_decode` in A9 and A3x, whose
   EA has one core ([[Process interface]]). A persistent HiGHS worker process (1 thread, niced, started at t0) fixes
   the integer part of a few candidates and solves the LP over the continuous variables.
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
   `trySol`; HiGHS: user-solution callback). Every post and the solver's verdict are logged. In A9 the post must beat
   SCIP's incumbent, not the min with cuOpt's, so an unchanged cuOpt vector can be handed to SCIP.
9. **Stagnation (trigger):** after `stagnation.gens` without a new best, replace the worst `immigrant_frac` of Rank 2
   with immigrants (frontier rows, `frontier.reinject` of them; LP roundings; random points); Rank 0 and Rank 1 are
   untouched (part I called itself "overly elitist").

## Self-adaptive repair strategy (2026-10-06, development; `strategy`, off by default)
**Why.** Feasibility probe 1 ([[Development log]], 2026-10-06): an LP point after t0 made the EA alone feasible on the
UC case and sing44, the rhs row scale did so on glass4 and fhnw-binpack4-48, and each setting hurt where the other
helped. Instead of choosing one per instance, the population carries both and selection decides — self-adaptation of a
strategy parameter as in evolution strategies, applied to the repair. First run on the development set on 2026-10-06
([[Development log]], "Coupled adaptive strategy"); `strategy.enabled` is part of the development configurations
`itrea2-dev-rc-3way` (A9, A3x) and `itrea2-dev-rc-ld` (A3).

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

## Repair clock (2026-10-06, development; `repair_clock`, off by default; active mode adopted 2026-10-07)
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

**Stages.** Stage 0 (built 2026-10-06; the gpu-marked horizon tests passed on 2026-10-07 before Stage 1): the horizon in
both feasibility-jump implementations, the controller, the engine wiring and their tests. Stage 1: **shadow mode**
(`mode: shadow`, `configs/itrea2-adaptive-lps-rc.yaml` = lps1 + the clock in shadow) plays "none" with no cap — the
search is lps1's: no argument is added to a local-search call and no random number is drawn, so the genealogy is
identical for the same generation count (tested on NumPy, also with the controller held at a shallow level). In a timed
run the extra host work (the key copy-back, one `rclock` event per generation) can shift generation counts and exchange
timing, which Stage 1 criterion (c) checks, and its configuration hash differs from lps1's: its runs are not lps1 runs.
It logs per generation what it would play, every call's profile and each lane's steps and status (`rclock` events,
[[Trace format and metrics]]). The played calls show every level, but shadow mode accumulates only what its decision
would have revealed (the levels ≤ its level, or every level if no lane ran beyond it), so its probes and its evidence
are the active controller's; the event keeps the full information and marks the levels used. Stage 2, the active
controller (`mode: active`), ran on 2026-10-07 (A3@rc1 vs same-day A3@lps1w): item_placement 4 of 4 better, mean d
+0.50; it failed D-012 by the letter on radiationm's band (−0.02 vs ±0.01; the clock never acted there) and on criterion
(c), and was adopted as the development default for repair depth as a stated deviation ([[D-012 Repair clock — adaptive
repair depth, staged]], addendum 2; `configs/itrea2-dev-rc-3way.yaml`, hash 1d12ad490dd0, and every development
configuration built on it). The `rclock` event still carries every lane's steps and status every generation (about 5.5
KB per generation at P = O = 256, ~24 MB per 300 s item_placement run) — thin the per-lane payload if long runs need it.
Off by default and left out of the configuration hash while disabled. With the GPU back-off (A9, below) the clock runs
on busy seconds.

## GPU back-off in A9 (2026-10-09, development; `backoff`, off by default; inert unless `exchange.from_cuopt`)
**Why.** In the A9 pilot (2026-10-07, [[D-015 Use cuOpt's feasible solutions — three-way arm A9]]) cuOpt ran slower
beside the EA: on chromaticindex1024-7 cuOpt's P was 0.545 inside A9 against 0.071 alone ([[Development log]],
"Three-way arm A9 — pilot"), its one incumbent coming at 65.35 s against 8.56 s (D-018). The back-off
changes only the EA's GPU duty — the share of wall time in which it keeps work on the GPU; every generation and every
exchange tick still happens. Decision and tests: [[D-018 A9 — the EA backs off the GPU beside cuOpt (contribution-gated
duty cycle)]]; design: [[2026-10-08 A9 GPU back-off design (workflow)]].

**Rule** (`ea/backoff.py`: host bookkeeping, no device array, no random number). Level ℓ gives duty 2^−ℓ.
- **Phase Y** (from t0, while no incumbent exists anywhere): ℓ = `yield_level`. It ends at the first SCIP or cuOpt
  incumbent — seen by a non-consuming peek of their info channels ([[Process interface]]) or at an exchange tick —, at
  the EA's own first incumbent, or at `yield_until_s` wall seconds; phase P then starts at full duty and phase Y never
  returns.
- **Phase P:** after a patience τ in busy seconds without a material contribution the duty steps down one level, then
  one more every `step_s`, down to 2^−`levels`. τ by the EA's state: `patience_search_s` with no feasible member of
  its own, `patience_own_s` with one but no material lead yet, clamp(`patience_kappa` / λ̂, `patience_min_s`,
  `patience_max_s`) after a lead (λ̂ = material events per busy second, decayed with half-life `half_life_s`). A
  material contribution (gain ≥ `materiality` in the repair clock's gap unit) restores full duty at once: a **lead** =
  an EA incumbent of its own (source "self"), its gain measured from min(the EA's best key, the external incumbent);
  an **assist** = SCIP's accepted verdict, noted `improved`, on an EA post that is not an unchanged cuOpt row
  (`backoff.assist`), its gain from SCIP's primal value before to after — not credited again when the row was already
  a material lead.
- **Enforcement:** a host-side token bucket (busy time lowers the credit by Δb·(1 − D), sleep raises it by s·D; below
  −`burst_s` a sleep of min(−credit / D, `max_sleep_s`) is due), slept only where the EA's GPU queue is empty, after a
  `synchronize`: (1) between generations, in slices of `slice_s` that keep the stop poll and the exchange ticks alive —
  a partner improvement at a tick forgives the credit (at most once per `wake_min_s`, never in forced mode) and only
  then ends the sleep; (2) inside each feasibility-jump call, before every chunk and between launches (`pace`,
  [[Feasibility jump local search]]), with the stop poll and the phase-Y peek but no exchange tick, so the cutoff stays
  constant within a generation. At full duty no sleep is due and the credit is reset.
- **Busy clock:** wall time minus the governor's sleeps. The patiences, the repair clock's decisions and generation
  seconds, and each feasibility-jump call's deadline and profile run on it, so the repair clock sees the same seconds
  at every duty.
- **Modes:** `active`; `shadow` — the same bookkeeping at generation boundaries, no sync, no sleep, no pace argument,
  no peek (its genealogy is that of the run without the governor; the stage-1 control); `forced` — duty `forced_duty`
  from t0, no phases, no ladder, no forgiven credit (stage-0 calibration only).
Enabling it needs `exchange.enabled` and `repair.method: fj`; the governor exists only when `exchange.from_cuopt` is
true (the runner sets it for A9 alone), so in every other arm nothing changes. Left out of the configuration hash while
disabled.

**Status.** Built 2026-10-09 (CPU suite and the gpu-marked pace tests passed the same day). Stage 0, the calibration (18
runs, 2026-10-09), gave PROCEED: phase-Y duty ½, floor ⅛ (by the rule's shortfall — on comp21-2idx cuOpt's P inside A9
stayed above A6 + 0.02 at every duty), `yield_until_s` 30, `patience_search_s` and `patience_own_s` 40. Stage 1 —
A9@bo1 (active, `configs/itrea2-dev-rc-3way-bo1.yaml`, run hash f67354be775d) vs A9@bo1s (the same values in shadow
mode, `-bo1s.yaml`, 62bf82b236f6) with same-day A6, 14 instances × seeds 0–1, 84 runs (`jobs/dev-backoff1-s0.jsonl`,
`-s1.jsonl`) — is prepared and not yet run; nothing is adopted. Records: [[Trace format and metrics]] (`backoff`,
`contrib`, the `bo_*` fields, the summary's `backoff` block).

## Warm-up before the clock (2026-10-06; revised 2026-10-08 and 2026-10-09)
The clock rule (D-002 v3) allows kernel warm-up on a synthetic problem only. `Engine.prepare` runs the whole engine on
`synthetic_problem` (the instance's dtypes and variable classes, not its data) with the run's backend, gene dtype and
operators, then discards it. Two passes: "many" (offspring = P_w = max(16, ⌈1/x⌉ + 1), injected messages of 3 rows)
and "one" (one child, one-row messages), because CuPy compiles a different kernel for a batch of one row. The warm-up
configuration switches every path on — exchange both ways through a stub mailbox, the frontier and its reinjection,
stagnation every generation, an LP point, every crossover and mutation operator, the LP decode through an in-process
stub with batches of 3 and 1 rows — and the GPU back-off off (it must never sleep before the clock). Each pass runs 3
generations, then forces the stagnation, shortfall and immigrant paths, `violation_guided` on a point that violates a
row, and two deterministic sweeps, then one more generation:
- `_warm_elite` (2026-10-08, D-016): the Rank 1 distance kernels of `elite_spread` and `clearing_pick` in both Rank 1
  modes (the spread is logged in both), on three copies of one vector, over two and three rows, with one-row, two-row
  and whole chunks;
- `_warm_operators` (2026-10-09): every crossover operator that applies to the instance and every mutation operator
  (`violation_guided` is forced separately; `toward_lp` needs an LP point) on batches of 1, 2 and 3 children with the
  run's gene dtype, parent B differing from A in every gene. The passes draw operators at random, so before this sweep
  an operator's one-child form was compiled only by chance.
On the GPU `_warm_scans` also runs the operators' two cumsum kernels for every power of two up to the operators'
memory budget. The check is the summary's `compiled_on_clock` (the CuPy kernel cache grew after t0;
`cupy_cache.new_after_t0` names the files) and three cold-cache GPU tests (`tests/test_ea_run.py`, `test_elite.py`,
`test_lp_decode.py`) that start from an empty kernel cache and assert that nothing compiled after t0; the elite and
lp_decode tests also assert that the timed part reached the new code. Two gaps were found this way: on 2026-10-08 the
elite kernels were never warmed (the synthetic population held 2 distinct members) and compiled on the clock in the
first block of the D-016 pool, which was voided ([[Mistakes and incidents]]); on 2026-10-09 the block crossover's
one-child chunk compiled after t0 in the lp_decode cold-cache test, before any timed run.

## Seeding
The initial population is drawn after t0 without any LP (`seeding.seed_population`: `seeding.random` = 0.5 of it uniform
in the bounds — integers over their integer points, an infinite side replaced by the finite bound ∓ 10; the rest
"zero-ish": every variable at the point of its bounds nearest 0 — the first such row exactly, the others with
Poisson(4), at least 1, random genes reset) and is
repaired and evaluated before its first survival. An LP point arrives only on the clock (`seeding.lp_source`: `solver` =
the solver's root LP through the mailbox; `highs_async` = HiGHS in a worker process started at t0, at most
`lp_time_limit` = 60 s; `none`); once it exists, LP-rounded individuals (share `seeding.lp_rounding` of the immigrant
count) join the next candidate set, `toward_lp` mutation uses it, and the same share of later immigrants are its
randomised roundings. The solver's incumbents and pool arrive through the exchange (step 1).

## What is logged
Per generation (every `log_every`): best/median key, feasible share, |Rank 0|, |Rank 1|, |Rank 2|, inc, offspring/s, GPU
memory peak, Rank 0 candidates and demotions, Rank 1's size and spread (both Rank 1 modes; with clearing also the
candidates the distance rule skipped), and with each optional feature its own fields (strategy, LP stream, repair clock,
A9, LP decode, back-off). Per event: every new population best with its lineage; every post and verdict; every injection
received. Trace = JSON lines + a summary JSON (schema in [[Trace format and metrics]]).
