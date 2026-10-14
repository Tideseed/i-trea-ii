"""Every EA parameter, in one validated, hashable configuration (`EAConfig`).

YAML files in `configs/` hold named configurations; a run records the configuration's full dict and its hash, so a
trace can always be tied to the exact parameters. Unknown keys are an error (a typo must not silently fall back to a
default). Defaults are starting points to be tuned on the tuning set only (vault D-002).

The rank parameters follow Berk's brief (2026-10-05) and the original I-TREA (Orbay 2011 thesis): at
most `x` of the population reserved for Rank 0 (infeasible individuals better than the best feasible one), and the best
`y` of the feasible individuals (Rank 1) protected in every generation. See vault `10-design/I-TREA II algorithm.md`
and `50-decisions/D-004 Mapping I-TREA (2015) to I-TREA II.md`.

`strategy` (`StrategyCfg`, off by default) makes the repair self-adaptive: every individual carries a (row scale, LP
guidance) strategy that children inherit and occasionally redraw (vault `10-design/I-TREA II algorithm.md`,
"Self-adaptive repair strategy"). While it is off the section is left out of the hash, so existing configuration hashes
are unchanged.

`lp_stream` (`LpStreamCfg`, off by default, D-011) streams the solver's good infeasible node LP points into the
population (Berk 2026-10-06). Same hash rule: left out while disabled.

`repair_clock` (`RepairClockCfg`, off by default) gives every repair call of a generation a lane-step horizon chosen
from the measured gain per second of each horizon (vault `80-reviews/2026-10-06 Repair-depth design (workflow).md`);
mode `shadow` only measures and logs. Same hash rule: left out while disabled.

`exchange.from_cuopt` (off by default, D-015) lets the EA also read cuOpt's incumbents in the three-way arm A9; it is
left out of the hash while false (`_LATE_DEFAULTS`), so every recorded hash stands. The runner switches it on for A9
(an arm-implied change, `itrea.experiments.pair.ARMS`), so A9 and its matched control A3x (A9's EA footprint, no
cuOpt; A3 differs from A9 by the EA's cores as well) share one configuration file.

`backoff` (`BackoffCfg`, off by default, D-018) lets the EA of arm A9 back off the GPU beside cuOpt by a
contribution-gated duty cycle (`ea.backoff`); inert unless `exchange.from_cuopt`. Same hash rule: left out while
disabled.

`ranks.rank1_mode` (default "top", D-016) chooses Rank 1 either as before or by cuOpt's clearing rule ("clearing": a
distinct elite of up to `rank1_size` feasible members, any two differing in at least `rank1_min_diff` integer
variables); the three fields are in `_LATE_DEFAULTS`, so every recorded hash stands while they are at their defaults.

`repair.continuous: "lp_decode"` (D-017, issue #27) is "auto" plus the LP decode: a HiGHS worker process started at
t0 fixes the integer part of a few candidates per generation and solves the LP over the continuous variables
(`ea.lp_decode`). Its later fields `repair.lp_decode_cold_limit`, `repair.lp_decode_infeasible`, `repair.lp_decode_gate`
and `repair.lp_decode_nice` are in `_LATE_DEFAULTS`, so every recorded hash stands while they are at their defaults.
`lp_decode_gate: lp_worker` (R2 of the 2026-10-14 C1 diagnosis) starts the decode worker only once the EA's own HiGHS
LP worker (`seeding.lp_source: highs_async`) has reached a final state, so the decoder never takes CPU from it on the
EA's one core in arm A9; the runner refuses lp_decode in A9 / A3x unless this gate is on (`experiments.pair`).
`lp_decode_nice` is the decode worker's nice increment over the EA host (it was the module constant `NICE`, 10).

`host` (`HostCfg`, R1 of the 2026-10-14 C1 diagnosis) is the EA host process itself: how its CUDA context waits on the
GPU (`cuda_sched`: "auto" = CUDA's default, which spins while there are fewer contexts than cores; "blocking_sync" or
"yield", set on the primary context through the driver API before anything retains it, `ea.cuda_sched`) and an
optional nice increment the host takes after it spawned the LP worker (`nice_after_lp_spawn`, relative to the nice the
worker inherited). The whole section is left out of the hash while every field is at its default, so every recorded
hash stands.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PopulationCfg:
    size: int = 256                    # P (upper bound when size_rule = "memory")
    offspring_per_gen: int = 256       # children created per generation
    eval_dtype: str = "float64"        # activities / violations: "float64" | "float32" (float32 only for pure-integer
                                       # problems with |x| < 2^24 and no continuous variable in an equality row);
                                       # objective keys are ALWAYS float64; posts are re-checked in float64
    gene_dtype: str = "auto"           # storage of X: "auto" (int8 if every variable is integer with bounds in
                                       # [-127, 127], int32 if integer within int32, else float64) | "float64" | "int32" | "int8"
    size_rule: str = "memory"          # "memory": P = min(size, largest P whose buffers fit gpu_mem_gb) — pre-registered
                                       # | "fixed": exactly `size` (fails if it does not fit)


@dataclass
class RankCfg:
    """The three ranks of I-TREA (Orbay 2011 thesis), single-objective form:
    Rank 0 = infeasible good, Rank 1 = feasible good (protected elite), Rank 2 = rejects."""
    x: float = 0.10                    # Rank 0 quota: at most floor(x * P) infeasible-good individuals (reserved up to it)
    y: float = 0.10                    # Rank 1: the best ceil(y * basis) feasible individuals are protected
    elite_basis: str = "feasible"      # "feasible": ceil(y * #feasible) | "population": ceil(y * P)
    rank1_twins: str = "drop"          # "drop" (I-TREA: equal-objective twins leave Rank 1 for Rank 2) | "keep"
    rank1_mode: str = "top"            # "top": the best n1 feasible by (key, age) with the twin rule and the ratchet
                                       # (above) | "clearing" (D-016, cuOpt's population rule): best-first, a feasible
                                       # candidate joins Rank 1 only if it differs from every member already chosen in
                                       # at least rank1_min_diff integer variables; twins rule and ratchet do not apply
    rank1_size: int = 0                # clearing: at most this many members; 0 = ceil(y * P)
    rank1_min_diff: int = 0            # clearing: k, the minimum number of differing integer variables; 0 = cuOpt's
                                       # rule from n_int (1 / 2 / 4 / 8 / 10 for n_int < 50 / 80 / 200 / 1000 / beyond)
    rank0_select: str = "itrea"        # over the quota: "itrea" (Pareto in (key, V), then nearest-distance thinning)
                                       # | "violation" (ascending V) | "rank_prob" (rank-weighted sampling without
                                       # replacement, the optopt X2 idea; fills the quota)
    rank0_bound_demote: bool = True    # infeasible-good candidates whose key beats the solver's PROVEN bound are
                                       # eliminated (the solver answers for part I's decision maker: such a point can
                                       # never be feasible at that value)
    rank2_select: str = "itrea"        # "itrea" (non-dominated sorting in (key, V) + crowding, NSGA-II style)
                                       # | "key_then_violation" (feasible by key, then least-violated infeasible)
    rank2_replacement: str = "generational"  # "generational": Rank 2 slots are filled from this generation's offspring
                                       # and injected solutions only — the old Rank 2 dies, so "protected" (Rank 1) and
                                       # "reserved" (Rank 0) mean something | "plus": parents compete too (part I)
    tol_mode: str = "scip"             # feasibility test: "scip" (|act-rhs| / max(|act|, |rhs|, 1)) | "highs" (absolute)
                                       # | "rhs" (relative to max(1, |rhs|)); must match the partner solver
    viol_norm: str = "l2"              # row violation scale in V: "l2" (row 2-norm) | "rhs" (max(1,|rhs|)) | "count"
    no_incumbent_policy: str = "violation"  # no feasible solution known yet: "violation" (rank all by V then key, all
                                            # Rank 2 — feasibility phase) | "itrea" (all are infeasible good: Rank 0
                                            # selection over everything, no quota)
    feas_alt_scale: str = "none"       # feasibility phase ("violation" policy) only: "none" — by V on viol_norm, as
                                       # before | "l2" | "rhs" | "count" — non-dominated fronts over (V on viol_norm,
                                       # V on this scale) with crowding, each scale's best kept (2026-10-06: under the
                                       # self-adaptive strategy, children repaired on the rhs scale died out under l2
                                       # survival on fhnw-binpack4-48; vault 80-reviews, feasibility targets consult)
    # Any shortfall below P (too few eligible candidates) is filled by the engine with immigrants, never with
    # over-quota infeasible-good individuals: the x quota is a TOTAL cap on super-optimal infeasible survivors.
    imp_tol: float = 1e-9              # "better than the incumbent" margin, relative to max(1, |inc|)
    feas_tol: float = 1e-6             # per-row / per-bound feasibility tolerance relative to max(1, |rhs|)
    int_tol: float = 1e-6              # integrality tolerance


@dataclass
class SelectionCfg:
    tournament_k: int = 2
    comparator: str = "itrea"          # "itrea": Rank 1 < Rank 0 < Rank 2, ties by key then V (I-TREA's tournament)
                                       # | "deb" (feasible beats infeasible; by key / by V) | "stochastic" (Runarsson-Yao)
    p_f: float = 0.45                  # stochastic ranking: probability of comparing by key when one side is infeasible
    parent_rank_weights: list[float] | None = None  # [w_rank0, w_rank1, w_rank2]; None = one pool (whole population)


@dataclass
class CrossoverCfg:
    pc: float = 0.9
    weights: dict[str, float] = field(default_factory=lambda: {"agreement": 0.6, "uniform": 0.2, "block": 0.2})
    p_swap: float = 0.1                # uniform: probability a gene comes from the second parent
    k: int = 1                         # k_point: cut points (k = 1 is part I's single-point crossover)
    block_rows: int = 4                # block: constraints whose variables come from the second parent
    block_closure: bool = True         # block: repair is restricted to the rows touched by the copied variables
    # agreement: the child keeps the genes where both parents agree; disagreeing genes are drawn from either parent
    # and marked, and the repair works only on rows that contain a disagreeing gene (Rothberg 2007-style recombination)


@dataclass
class MutationCfg:
    genes: float = 4.0                 # mean number of genes mutated per child (Poisson, at least `min_genes`) — absolute,
                                       # so the same setting means the same disturbance on markshare and on rail01
    rate: float | None = None          # if set, overrides `genes`: share of integer genes per child
    min_genes: int = 1
    weights: dict[str, float] = field(default_factory=lambda: {
        "flip": 0.3, "step": 0.15, "violation_guided": 0.25, "objective_guided": 0.2, "toward_lp": 0.1})
    step_max: int = 2                  # step: ±1 … ±step_max on general integers
    reset_window: float = 10.0         # reset on unbounded domains: uniform within ±window of the current value


@dataclass
class RepairCfg:
    method: str = "fj"                 # "fj": feasibility jump (Luteberget & Sartor 2023) — GPU RawKernel, one block per
                                       # individual, incremental updates, jump values; NumPy reference on the CPU
                                       # | "dense": vectorised full rescore per step (baseline) | "none"
    max_steps: int = 2000              # fj: moves per child per generation at most
    stall_steps: int = 100             # fj: stop a child after this many moves without weighted-violation decrease
    moves_per_step: int = 1            # dense: moves applied per individual per step (distinct rows)
    steps: int = 50                    # dense: steps per generation
    objective_mode: str = "cutoff_row" # "cutoff_row": once an incumbent exists, the row cmin.x <= inc - delta joins the
                                       # weighted rows (keeps a Rank 0 child's better objective while repairing)
                                       # | "weighted": + obj_weight * cmin.x / max|cmin| | "none"
    obj_weight: float = 1e-3           # weighted mode only (objective scaled by 1 / max|cmin|)
    weight_bump: float = 1.0           # fj: row-weight increase at a local minimum
    max_candidates: int = 16           # fj: candidate values per variable and step (x±1, bounds, tightening values)
    p_random_walk: float = 0.0         # fj / dense: at a local minimum, apply the least-bad move with this probability
    row_scan_cap: int = 0              # fj: 0 = score every variable of the chosen row (spec); > 0 = rows longer than
                                       # this (in practice the cutoff row) are scored on a random cyclic window of this size
    row_scan_filter: bool = False      # fj, GPU only: inside that window skip variables that cannot move in the
                                       # direction that reduces the row's violation (breaks CPU/GPU equivalence)
    continuous: str = "auto"           # "auto": singleton continuous columns decoded in closed form on the device, other
                                       # continuous variables moved by FJ jump values | "gene" | "lp_decode": "auto"
                                       # plus the LP decode (D-017, `ea.lp_decode`): a HiGHS worker process fixes the
                                       # integers of a few candidates per generation and solves the LP over the
                                       # continuous variables; the copies join the next candidate set
    lp_decode_per_gen: int = 4         # lp_decode: at most this many LPs per batch (N), at most one batch outstanding
    lp_time_limit: float = 2.0         # lp_decode: seconds per warm LP of a feasible source; half of it for an
                                       # infeasible source (the completion LPs)
    lp_decode_cold_limit: float = 10.0 # lp_decode: seconds for the worker's first ("priming") LP, and for every LP
                                       # while no optimal basis exists to warm-start from
    lp_decode_infeasible: bool = True  # lp_decode: also offer infeasible candidates better than the incumbent
                                       # (completion); false = feasible sources only (polish)
    lp_decode_gate: str = "none"       # lp_decode: "none" = the worker starts at t0 | "lp_worker" = it starts only once
                                       # the EA's HiGHS LP worker (seeding.lp_source highs_async) has reported a final
                                       # state (done, no_solution, failed or stopped; at t0 when there is none), so its
                                       # read and priming never take CPU from that LP on a shared core (R2, A9)
    lp_decode_nice: int = 10           # lp_decode: the worker's nice increment over the EA host (os.nice in the
                                       # worker; it inherits the host's nice at the spawn); 10 = the former constant


@dataclass
class ExchangeCfg:
    enabled: bool = True
    to_solver: bool = True             # post EA improvements to the solver
    from_solver: bool = True           # inject the solver's solutions into the population
    poll_s: float = 0.25               # how often the EA polls the mailbox
    inject_band: float = 0.01          # "very good": within this relative gap of the solver's best
    inject_k: int = 8                  # at most this many solver solutions per message
    post_min_gain: float = 1e-9        # post only if better than the solver's incumbent by this relative margin
    post_frontier: bool = False        # also post the frontier's diverse near-optimal solutions (SCIP keeps them in its
                                       # solution storage, where its crossover-type heuristics can use them)
    use_dual_bound: bool = True        # receive the solver's proven bound (needed by ranks.rank0_bound_demote)
    use_root_lp: bool = True           # receive the solver's root LP solution (seeding, toward_lp)
    from_cuopt: bool = False           # arm A9 (D-015): also read cuOpt's own channels (`to_ea_cuopt`, `info_cuopt`):
                                       # its incumbents are injected like the solver's, its incumbent counts in the
                                       # external incumbent; its bound is recorded, not used. Off: never read


@dataclass
class SeedingCfg:
    random: float = 0.5                # share of the initial population sampled in the bounds (the rest: bound-feasible
                                       # "zero-ish" points — each variable at the bound nearest 0 — then mutated)
    lp_rounding: float = 0.5           # share of immigrants from randomised rounding of an LP solution, once one exists
    lp_source: str = "solver"          # where the LP point comes from — ALWAYS on the clock: "solver" (the solver's root
                                       # LP via the mailbox) | "highs_async" (HiGHS in a worker on the EA's core, after
                                       # t0) | "none". Nothing instance-specific is solved before the shared start.
    lp_time_limit: float = 60.0        # highs_async: LP time limit


@dataclass
class StagnationCfg:
    gens: int = 50                     # generations without a new best before immigration
    immigrant_frac: float = 0.25       # share of Rank 2 replaced by immigrants (Rank 0 and Rank 1 untouched)


@dataclass
class FrontierCfg:
    """The frontier: an archive of diverse near-optimal FEASIBLE solutions (Berk, 2026-10-05: "I-TREA is
    multiobjective; we can start single objective but shall also define a frontier of diverse solutions").
    With one objective the archive is the non-dominated set in (key, distance from the incumbent) — quality vs
    diversity; with several objectives it becomes the Pareto front of those objectives (vault D-005)."""
    enabled: bool = True
    mode: str = "pareto_incumbent"     # "pareto_incumbent": non-dominated in (key ↓, distance to incumbent ↑)
                                       # | "maxmin": greedy max-min diverse set inside the quality band
    band: float = 0.05                 # quality band: key <= inc + band * max(1, |inc|)
    max_size: int = 32                 # archive size; the overflowing front is thinned by crowding distance
    distance: str = "hamming"          # "hamming" over integer genes (normalised by #ints) | "l1" (normalised by range)
    update_every: int = 5              # generations between archive updates
    reinject: float = 0.0              # share of stagnation immigrants drawn from the archive (0 = none)


@dataclass
class StrategyCfg:
    """Self-adaptive repair strategy (development finding 2026-10-06, vault `30-results/Development log.md`: the LP
    point unlocks some instances, rhs row scaling others, and each hurts where the other helps). Every individual
    carries a strategy — an index into the Cartesian product row_scales x lp_guidance, in that order (scale-major) —
    inherited from its first parent, redrawn with probability `mutation_rate`; selection decides per instance which
    strategy spreads. A strategy changes only the REPAIR (the row scale s_i of the feasibility-jump violation) and the
    MUTATION (whether toward_lp may use the LP point); the survival's V keeps the one scale `ranks.viol_norm` for every
    individual, so ranks stay comparable. Off by default: with `enabled: false` the section is inert and is left out of
    the configuration hash (`EAConfig.hash`)."""
    enabled: bool = False
    row_scales: list[str] = field(default_factory=lambda: ["l2", "rhs"])   # repair row scale: l2 | rhs | count
    lp_guidance: list[bool] = field(default_factory=lambda: [False, True])  # mutation may use the LP point (toward_lp
                                       # with its configured weight, and LP-rounding immigrants) once one exists
    mutation_rate: float = 0.1         # probability that a child's strategy is redrawn uniformly (else parent A's)
    min_share: float = 0.05            # each strategy gets at least min(ceil(min_share * O), O // S) of the O children
                                       # (re-assigned from strategies above that floor), so one can come back later

    def strategies(self) -> list[tuple[str, bool]]:
        """The strategy space, index = position: [(row_scale, lp_guidance), ...] (scale-major)."""
        return [(s, bool(g)) for s in self.row_scales for g in self.lp_guidance]


@dataclass
class LpStreamCfg:
    """The solver -> EA stream of good infeasible solutions (D-011; Berk 2026-10-06: "when solver comes up with suitable
    good infeasible solutions (minimize infeasibility then maximize objective function) it might be able to inject to
    EA's population"). The solver side (SCIP only: `scip_run --lp-stream`) reads node LP points under a time budget
    and prefers nodes with few fractional integer variables; the EA rounds each received point to the nearest integer
    point, evaluates it, keeps the best `pool` points ranked LEXICOGRAPHICALLY by (violation V of that rounding,
    objective key) — Berk's order — and turns each unused point, best first, into `roundings` children (the nearest
    rounding + randomised ones), repaired and then admitted through ordinary survival only: a super-optimal infeasible
    child competes for Rank 0 under its cap and selection rules, never bypassing them. Off by default: with
    `enabled: false` the section is inert and is left out of the configuration hash (`EAConfig.hash`)."""
    enabled: bool = False
    share: float = 0.02                # solver: the stream's read + post time stays under this share of the wall clock
    ring: int = 3                      # solver: points per post (the newest; a reader that misses a post loses none)
    min_gap: float = 0.25              # solver: seconds between two reads, at least
    quantile: float = 0.25             # solver: a node is good if its fractional count is at most this quantile of the
                                       # last 32 LP solves
    patience: float = 1.0              # solver: seconds a due read waits for a good node before it takes the next one
    roundings: int = 4                 # EA: children per point — the nearest rounding + (roundings - 1) randomised
    per_tick: int = 2                  # EA: at most this many new points turned into children per exchange tick
    pool: int = 16                     # EA: points kept, ranked (V of the nearest rounding, key)
    immigrants: bool = True            # EA: stagnation LP-rounding immigrants take their LP point from the pool's best
                                       # four, round robin (else always the first LP point)


@dataclass
class RepairClockCfg:
    """The repair clock (2026-10-06 repair-depth design, staged: instrumentation and shadow mode first). One level per
    generation applies to every repair call of it: a horizon h = 2^j lane steps (j = level_min_log2 .. level_max_log2)
    or "none" (the repair settings as configured). Lanes still active at h stop with status 4 (`ls_fj_*.ST_HORIZON`)
    and keep their projected x; on the GPU the host truncates the launches, so the cut is the same for every lane.
    Until the population holds a feasible member of the EA's own (a feasible member that is not an unchanged solver
    solution; neither a finite solver incumbent nor an injected feasible solver row counts) every call is played at
    "none" without a cap. After it, the controller (`ea.repair_clock`) reads each level's gain from the call it just
    ran — a horizon-h call is a prefix of the unbounded call — and its cost from the measured per-launch (steps, time)
    profile, and switches only on evidence. `shadow` plays "none" with no cap (no local-search call changes) and logs
    what it would choose; `active` plays it. Off by default:
    with `enabled: false` the section is inert and is left out of the configuration hash (`EAConfig.hash`)."""
    enabled: bool = False
    mode: str = "shadow"               # "shadow": play "none", log the choice | "active": play the choice
    level_min_log2: int = 7            # the ladder: h = 2^7 .. 2^16 lane steps, plus "none"
    level_max_log2: int = 16
    min_gens: int = 5                  # a level is eligible after this many observed post-feasible generations ...
    min_events: int = 2                # ... with at least this many material gains
    materiality: float = 1e-3          # a generation's gain below this (gap units) counts as zero
    half_life_s: float = 20.0          # gains and times decay with this half-life (seconds of wall time)
    hysteresis: float = 0.25           # switch only to a level whose rate is above (1 + hysteresis) x the current one's
    probe_share: float = 0.10          # away from "none": at most this share of post-feasible time re-measures "none"
    cap_s: float = 10.0                # per-call wall-clock cap (active, after the first feasible): min(cap_s,
    cap_factor: float = 3.0            # cap_factor x the level's median children-call time); "none": cap_s alone
    pin_lp: bool = True                # the one-shot LP batch is always repaired at "none", uncapped


@dataclass
class BackoffCfg:
    """The EA backs off the GPU beside cuOpt in arm A9 by a contribution-gated duty cycle (D-018; design: vault
    `80-reviews/2026-10-08 A9 GPU back-off design (workflow).md`, section 1). Only the EA's GPU DUTY changes — the share
    of wall time in which it may keep work on the GPU: a host-side token bucket sleeps between FJ launches and between
    generations (`ea.backoff`). Phase Y (no incumbent anywhere): duty 2^-yield_level until the first SCIP / cuOpt
    incumbent, the EA's own first incumbent, or `yield_until_s` (wall seconds since t0). Phase P: full duty, stepping
    down 1 -> 1/2 -> 1/4 (-> 1/8) to 2^-levels after a patience in BUSY seconds that depends on the EA's state (no own
    feasible member: patience_search_s; own feasible, no material lead yet: patience_own_s; led: clamp(kappa / the
    decayed material-event rate, patience_min_s, patience_max_s)), then step_s per further level; any material lead
    (>= materiality in the repair clock's gap unit) or SCIP-confirmed assist restores full duty. `mode`: "active" plays
    the rule; "shadow" only logs what it would do (the control arm: no sync, no sleep, no in-call pace); "forced" holds
    `forced_duty` from t0 to the end (stage-0 calibration only, never a default). Inert unless
    `exchange.from_cuopt` is true (the runner sets it for A9 alone). Off by default: with `enabled: false` the section
    is inert and is left out of the configuration hash (`EAConfig.hash`)."""
    enabled: bool = False
    mode: str = "active"               # "active" | "shadow" (bookkeeping only) | "forced" (fixed duty, stage 0 only)
    forced_duty: float = 1.0           # forced mode: the duty held from t0 to the end
    yield_level: int = 2               # phase Y duty 2^-yield_level (stage-0 rule (a))
    levels: int = 2                    # L: the ladder's floor duty 2^-L (stage-0 rule (b))
    yield_until_s: float = 30.0        # phase Y ends at this wall time since t0 at the latest (stage-0 rule (c))
    patience_search_s: float = 75.0    # busy seconds before the first step down with no own feasible member (rule (d))
    patience_own_s: float = 30.0       # ... with an own feasible member but no material lead yet (rule (e))
    materiality: float = 1e-3          # a lead / assist counts if its gain (gap units) is at least this
    patience_kappa: float = 3.0        # led: patience = clamp(kappa / rate, patience_min_s, patience_max_s)
    patience_min_s: float = 2.0
    patience_max_s: float = 40.0
    half_life_s: float = 20.0          # the material-event rate decays with this half-life (busy seconds)
    step_s: float = 4.0                # busy seconds without a contribution between two further steps down
    burst_s: float = 0.25              # the token bucket sleeps once the credit falls below -burst_s
    max_sleep_s: float = 4.0           # longest sleep at one pace point
    slice_s: float = 0.25              # a sleep is cut in slices of at most this; each keeps the stop poll alive
    wake_min_s: float = 5.0            # a partner improvement forgives the credit at most once per this many seconds
    assist: bool = True                # count SCIP-confirmed improvements of EA posts as contributions


@dataclass
class HostCfg:
    """The EA host process (R1 of vault `30-results/2026-10-14 C1 diagnosis and improvement candidates.md`). In arm A9
    the EA host and the HiGHS LP worker it spawns share one X925 core, and the host busy-waits on the GPU — CUDA's
    default scheduling spins while there are fewer contexts than cores — so it keeps that core about 100 % busy while
    99 % of its wall time is GPU repair, and the LP worker gets about half the core.
    `cuda_sched` sets how the EA's primary CUDA context waits: "auto" (CUDA's default; nothing is set, the active
    flags are read back after the context exists) | "blocking_sync" (CU_CTX_SCHED_BLOCKING_SYNC: the host thread sleeps on a semaphore until the GPU is done; the
    engine's per-generation timing events are then created with the blocking flag too, since a default-flag event wait
    may spin whatever the context says) | "yield" (CU_CTX_SCHED_YIELD: the host spins but yields its time slice). It is
    set through the driver API (`ea.cuda_sched`: cuInit, cuDeviceGet, cuDevicePrimaryCtxSetFlags) before anything
    retains the context, before the warm-up and t0, and checked with cuDevicePrimaryCtxGetState once the context
    exists; inert on the NumPy backend. `nice_after_lp_spawn` (0 = unchanged): right after the EA spawned the LP worker
    at t0 (`seeding.lp_source: highs_async`), every thread of the EA host takes nice `n + nice_after_lp_spawn`, n = the
    nice the worker was spawned at (capped at 19; os.setpriority; never lowered; `ea.engine.renice_threads`), so on a
    shared core the LP worker wins the CPU while it runs (+10: CFS weights 110 : 1024); the host keeps it to the end
    (an unprivileged process cannot raise its priority again). It is D-025's fallback (a) `bnice`, WITHOUT a context
    flag. The section is left out of the configuration hash while every field is at its default (`EAConfig.hash`)."""
    cuda_sched: str = "auto"           # "auto" | "blocking_sync" | "yield"
    nice_after_lp_spawn: int = 0       # 0 = unchanged | 1..19: the EA host's nice INCREMENT over the LP worker's,
                                       # taken right after the worker's spawn (capped at 19)


# Fields added after configuration hashes were recorded (runs, D-002): left out of the hash while at these defaults.
_LATE_DEFAULTS = {"ranks.feas_alt_scale": "none", "exchange.from_cuopt": False,
                  "ranks.rank1_mode": "top", "ranks.rank1_size": 0, "ranks.rank1_min_diff": 0,
                  "repair.lp_decode_cold_limit": 10.0, "repair.lp_decode_infeasible": True,
                  "repair.lp_decode_gate": "none", "repair.lp_decode_nice": 10}


@dataclass
class EAConfig:
    name: str = "default"
    population: PopulationCfg = field(default_factory=PopulationCfg)
    ranks: RankCfg = field(default_factory=RankCfg)
    selection: SelectionCfg = field(default_factory=SelectionCfg)
    crossover: CrossoverCfg = field(default_factory=CrossoverCfg)
    mutation: MutationCfg = field(default_factory=MutationCfg)
    repair: RepairCfg = field(default_factory=RepairCfg)
    exchange: ExchangeCfg = field(default_factory=ExchangeCfg)
    seeding: SeedingCfg = field(default_factory=SeedingCfg)
    stagnation: StagnationCfg = field(default_factory=StagnationCfg)
    frontier: FrontierCfg = field(default_factory=FrontierCfg)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    lp_stream: LpStreamCfg = field(default_factory=LpStreamCfg)
    repair_clock: RepairClockCfg = field(default_factory=RepairClockCfg)
    backoff: BackoffCfg = field(default_factory=BackoffCfg)
    host: HostCfg = field(default_factory=HostCfg)
    backend: str = "cupy"           # "cupy" | "numpy" — never a silent fallback ("auto" only with allow_cpu_fallback)
    gpu_mem_gb: float = 12.0           # CuPy pool limit (the only GPU memory cap on GB10) = cuOpt's RMM limit in A2:
                                       # equal GPU memory for both partners; cuOpt needs ~10.5 GB (measured 2026-10-06)
    log_every: int = 10                # generations between generation-log records
    lineage_depth: int = 5             # generations of ancestry recorded for every new best

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        """12-hex digest of the canonical JSON of every parameter except `name` — and except the `strategy`,
        `lp_stream`, `repair_clock` and `backoff` sections while disabled, the `host` section while every field of it is
        at its default, and the fields in `_LATE_DEFAULTS` while at their default: they are inert then, so every
        configuration written before a section or field existed keeps its hash (the default's `9ac4a93bf93b` among
        them)."""
        d = self.to_dict()
        d.pop("name", None)
        if not self.strategy.enabled:
            d.pop("strategy", None)
        if not self.lp_stream.enabled:
            d.pop("lp_stream", None)
        if not self.repair_clock.enabled:
            d.pop("repair_clock", None)
        if not self.backoff.enabled:
            d.pop("backoff", None)
        if self.host == HostCfg():                         # the whole section, not its keys one by one: an empty
            d.pop("host", None)                            # {"host": {}} would change every hash
        for dotted, default in _LATE_DEFAULTS.items():     # fields added after a hash was recorded: out while default
            sec, _, k = dotted.partition(".")
            if d.get(sec, {}).get(k, object()) == default:
                d[sec].pop(k)
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: dict) -> "EAConfig":
        return _build(cls, d or {}, "EAConfig")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EAConfig":
        cfg = cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})
        cfg.validate()
        return cfg

    def with_overrides(self, overrides: dict) -> "EAConfig":
        """New config with dotted-key overrides, e.g. {"ranks.x": 0.0, "population.size": 128}."""
        d = self.to_dict()
        for k, v in overrides.items():
            cur = d
            parts = k.split(".")
            for p in parts[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    raise KeyError(f"unknown config section {k!r}")
                cur = cur[p]
            if parts[-1] not in cur:
                raise KeyError(f"unknown config key {k!r}")
            cur[parts[-1]] = v
        cfg = EAConfig.from_dict(d)
        cfg.validate()
        return cfg

    # ------------------------------------------------------------ validate
    def validate(self) -> "EAConfig":
        p, r, s, c, m, rp, e, sd, st = (self.population, self.ranks, self.selection, self.crossover, self.mutation,
                                        self.repair, self.exchange, self.seeding, self.stagnation)
        _check(p.size >= 2, "population.size >= 2")
        _check(p.offspring_per_gen >= 1, "population.offspring_per_gen >= 1")
        _check(p.eval_dtype in ("float64", "float32"), "population.eval_dtype in float64|float32")
        _check(p.gene_dtype in ("auto", "float64", "int32", "int8"), "population.gene_dtype")
        _check(p.size_rule in ("memory", "fixed"), "population.size_rule in memory|fixed")
        _check(0.0 <= r.x < 1.0, "ranks.x in [0, 1)")
        _check(0.0 <= r.y <= 1.0, "ranks.y in [0, 1]")
        _check(r.elite_basis in ("feasible", "population"), "ranks.elite_basis in feasible|population")
        _check(r.rank1_twins in ("drop", "keep"), "ranks.rank1_twins in drop|keep")
        _check(r.rank1_mode in ("top", "clearing"), "ranks.rank1_mode in top|clearing")
        _check(r.rank1_size >= 0 and r.rank1_min_diff >= 0, "ranks.rank1_size >= 0 and ranks.rank1_min_diff >= 0")
        _check(r.rank0_select in ("itrea", "violation", "rank_prob"), "ranks.rank0_select in itrea|violation|rank_prob")
        _check(r.rank2_select in ("itrea", "key_then_violation"), "ranks.rank2_select")
        _check(r.rank2_replacement in ("generational", "plus"), "ranks.rank2_replacement in generational|plus")
        _check(r.tol_mode in ("scip", "highs", "rhs"), "ranks.tol_mode in scip|highs|rhs")
        _check(r.viol_norm in ("l2", "rhs", "count"), "ranks.viol_norm in l2|rhs|count")
        _check(r.feas_alt_scale in ("none", "l2", "rhs", "count") and r.feas_alt_scale != r.viol_norm,
               "ranks.feas_alt_scale in none|l2|rhs|count and different from ranks.viol_norm")
        _check(r.x == 0 or math.floor(r.x * p.size + 1e-12) >= 1, "ranks.x * population.size must give >= 1 Rank 0 slot")
        _check(r.y == 0 or r.elite_basis != "population" or math.ceil(r.y * p.size - 1e-12) >= 1, "ranks.y slots")
        _check(r.no_incumbent_policy in ("violation", "itrea"), "ranks.no_incumbent_policy in violation|itrea")
        _check(r.feas_tol > 0 and r.int_tol > 0 and r.imp_tol >= 0, "ranks tolerances")
        _check(s.tournament_k >= 1, "selection.tournament_k >= 1")
        _check(s.comparator in ("itrea", "deb", "stochastic"), "selection.comparator in itrea|deb|stochastic")
        _check(0.0 <= s.p_f <= 1.0, "selection.p_f in [0, 1]")
        if s.parent_rank_weights is not None:
            _check(len(s.parent_rank_weights) == 3 and all(w >= 0 for w in s.parent_rank_weights)
                   and sum(s.parent_rank_weights) > 0, "selection.parent_rank_weights: three non-negative weights")
        _check(0.0 <= c.pc <= 1.0, "crossover.pc in [0, 1]")
        _check(set(c.weights) <= {"agreement", "uniform", "block", "k_point", "none"} and sum(c.weights.values()) > 0
               and all(v >= 0 for v in c.weights.values()),
               "crossover.weights over agreement|uniform|block|k_point|none")
        _check(0.0 <= c.p_swap <= 1.0 and c.k >= 1 and c.block_rows >= 1, "crossover parameters")
        _check(m.genes >= 0 and (m.rate is None or 0.0 <= m.rate <= 1.0) and m.min_genes >= 0,
               "mutation.genes >= 0, rate in [0, 1] or null, min_genes >= 0")
        _check(set(m.weights) <= {"flip", "step", "reset", "toward_lp", "violation_guided", "objective_guided"}
               and sum(m.weights.values()) > 0 and all(v >= 0 for v in m.weights.values()),
               "mutation.weights over flip|step|reset|toward_lp|violation_guided|objective_guided")
        _check(m.step_max >= 1 and m.reset_window > 0, "mutation.step_max >= 1, reset_window > 0")
        _check(rp.method in ("none", "dense", "fj"), "repair.method in none|dense|fj")
        _check(rp.steps >= 0 and rp.moves_per_step >= 1 and rp.max_steps >= 0 and rp.stall_steps >= 1,
               "repair step counts")
        _check(rp.objective_mode in ("cutoff_row", "weighted", "none"), "repair.objective_mode")
        _check(4 <= rp.max_candidates <= 32 and 0.0 <= rp.p_random_walk <= 1.0 and rp.row_scan_cap >= 0,
               "repair fj knobs: max_candidates in [4, 32], p_random_walk in [0, 1], row_scan_cap >= 0")
        _check(rp.continuous in ("auto", "gene", "lp_decode"), "repair.continuous in auto|gene|lp_decode")
        _check(isinstance(rp.lp_decode_per_gen, int) and not isinstance(rp.lp_decode_per_gen, bool)
               and rp.lp_decode_per_gen >= 1, "repair.lp_decode_per_gen is an integer >= 1")
        _check(isinstance(rp.lp_time_limit, (int, float)) and not isinstance(rp.lp_time_limit, bool)
               and rp.lp_time_limit > 0, "repair.lp_time_limit > 0")
        _check(isinstance(rp.lp_decode_cold_limit, (int, float)) and not isinstance(rp.lp_decode_cold_limit, bool)
               and rp.lp_decode_cold_limit >= rp.lp_time_limit, "repair.lp_decode_cold_limit >= repair.lp_time_limit")
        _check(isinstance(rp.lp_decode_infeasible, bool), "repair.lp_decode_infeasible is a bool")
        _check(e.poll_s > 0 and e.inject_k >= 1 and e.inject_band >= 0, "exchange parameters")
        _check(isinstance(e.from_cuopt, bool), "exchange.from_cuopt is a bool")
        _check(0.0 <= sd.random <= 1.0 and 0.0 <= sd.lp_rounding <= 1.0, "seeding shares in [0, 1]")
        _check(sd.lp_source in ("solver", "highs_async", "none"), "seeding.lp_source in solver|highs_async|none")
        _check(st.gens >= 1 and 0.0 <= st.immigrant_frac <= 1.0, "stagnation parameters")
        fr = self.frontier
        _check(fr.mode in ("pareto_incumbent", "maxmin"), "frontier.mode in pareto_incumbent|maxmin")
        _check(fr.distance in ("hamming", "l1"), "frontier.distance in hamming|l1")
        _check(fr.band >= 0 and fr.max_size >= 1 and fr.update_every >= 1 and 0.0 <= fr.reinject <= 1.0,
               "frontier parameters")
        sg = self.strategy
        _check(isinstance(sg.enabled, bool), "strategy.enabled is a bool")
        _check(isinstance(sg.row_scales, list) and len(sg.row_scales) >= 1
               and all(isinstance(v, str) and v in ("l2", "rhs", "count") for v in sg.row_scales)
               and len(set(sg.row_scales)) == len(sg.row_scales),
               "strategy.row_scales: a non-empty list of distinct values in l2|rhs|count")
        _check(isinstance(sg.lp_guidance, list) and len(sg.lp_guidance) >= 1
               and all(isinstance(v, bool) for v in sg.lp_guidance)
               and len(set(sg.lp_guidance)) == len(sg.lp_guidance),
               "strategy.lp_guidance: a non-empty list of distinct booleans")
        _check(isinstance(sg.mutation_rate, (int, float)) and not isinstance(sg.mutation_rate, bool)
               and 0.0 <= sg.mutation_rate <= 1.0, "strategy.mutation_rate in [0, 1]")
        n_strat = len(sg.row_scales) * len(sg.lp_guidance)
        _check(isinstance(sg.min_share, (int, float)) and not isinstance(sg.min_share, bool)
               and 0.0 <= sg.min_share and sg.min_share * n_strat <= 1.0 + 1e-12,
               "strategy.min_share >= 0 and min_share * (number of strategies) <= 1")
        ls = self.lp_stream
        _check(isinstance(ls.enabled, bool) and isinstance(ls.immigrants, bool), "lp_stream.enabled/immigrants are bools")
        _check(0.0 < ls.share <= 1.0 and ls.min_gap >= 0.0 and 0.0 <= ls.quantile <= 1.0 and ls.patience >= 0.0,
               "lp_stream: share in (0, 1], min_gap >= 0, quantile in [0, 1], patience >= 0")
        _check(all(isinstance(v, int) and not isinstance(v, bool) for v in (ls.ring, ls.roundings, ls.per_tick, ls.pool))
               and ls.ring >= 1 and ls.roundings >= 1 and ls.per_tick >= 1 and ls.pool >= 1,
               "lp_stream: ring, roundings, per_tick, pool are integers >= 1")
        _check(not ls.enabled or (self.exchange.enabled and self.exchange.from_solver),
               "lp_stream.enabled needs exchange.enabled and exchange.from_solver")
        rc = self.repair_clock
        _check(isinstance(rc.enabled, bool) and isinstance(rc.pin_lp, bool), "repair_clock.enabled/pin_lp are bools")
        _check(rc.mode in ("shadow", "active"), "repair_clock.mode in shadow|active")
        _check(all(isinstance(v, int) and not isinstance(v, bool)
                   for v in (rc.level_min_log2, rc.level_max_log2, rc.min_gens, rc.min_events))
               and 0 <= rc.level_min_log2 <= rc.level_max_log2 <= 20 and rc.min_gens >= 1 and rc.min_events >= 1,
               "repair_clock: integers 0 <= level_min_log2 <= level_max_log2 <= 20 (the launch cap), min_gens >= 1, "
               "min_events >= 1")
        _check(rc.materiality >= 0 and rc.half_life_s > 0 and rc.hysteresis >= 0 and 0.0 < rc.probe_share < 1.0
               and rc.cap_s > 0 and rc.cap_factor > 0,
               "repair_clock: materiality >= 0, half_life_s > 0, hysteresis >= 0, probe_share in (0, 1), cap_s > 0, "
               "cap_factor > 0")
        # the gain signal relies on FEASIBLE => strictly below the cutoff (the cutoff row of the feasibility jump)
        _check(not rc.enabled or (rp.method == "fj" and rp.objective_mode == "cutoff_row"),
               "repair_clock.enabled needs repair.method fj and repair.objective_mode cutoff_row")
        bo = self.backoff
        _check(isinstance(bo.enabled, bool) and isinstance(bo.assist, bool), "backoff.enabled/assist are bools")
        _check(bo.mode in ("active", "shadow", "forced"), "backoff.mode in active|shadow|forced")
        # yield_level is bounded by the ladder's largest level (3), not by `levels`: stage 0 sets the phase-Y duty and
        # the floor by separate rules (D-018), and a phase-Y duty below the floor (e.g. 1/8 with a floor of 1/4) must
        # stay expressible
        _check(all(isinstance(v, int) and not isinstance(v, bool) for v in (bo.levels, bo.yield_level))
               and 1 <= bo.levels <= 3 and 0 <= bo.yield_level <= 3,
               "backoff: integers 1 <= levels <= 3 and 0 <= yield_level <= 3")
        _check(isinstance(bo.forced_duty, (int, float)) and not isinstance(bo.forced_duty, bool)
               and 0.0 < bo.forced_duty <= 1.0, "backoff.forced_duty in (0, 1]")
        secs = (bo.yield_until_s, bo.patience_search_s, bo.patience_own_s, bo.patience_min_s, bo.patience_max_s,
                bo.half_life_s, bo.step_s, bo.burst_s, bo.max_sleep_s, bo.slice_s, bo.wake_min_s)
        _check(all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in secs),
               "backoff: every *_s > 0")
        _check(bo.patience_min_s <= bo.patience_max_s, "backoff.patience_min_s <= backoff.patience_max_s")
        _check(bo.patience_kappa > 0 and bo.materiality >= 0, "backoff.patience_kappa > 0, materiality >= 0")
        _check(not bo.enabled or (self.exchange.enabled and rp.method == "fj"),
               "backoff.enabled needs exchange.enabled and repair.method fj")
        _check(rp.lp_decode_gate in ("none", "lp_worker"), "repair.lp_decode_gate in none|lp_worker")
        _check(rp.lp_decode_gate == "none" or rp.continuous == "lp_decode",
               "repair.lp_decode_gate lp_worker needs repair.continuous lp_decode")
        _check(isinstance(rp.lp_decode_nice, int) and not isinstance(rp.lp_decode_nice, bool)
               and 0 <= rp.lp_decode_nice <= 19, "repair.lp_decode_nice is an integer in [0, 19]")
        ho = self.host
        _check(ho.cuda_sched in ("auto", "blocking_sync", "yield"), "host.cuda_sched in auto|blocking_sync|yield")
        _check(isinstance(ho.nice_after_lp_spawn, int) and not isinstance(ho.nice_after_lp_spawn, bool)
               and 0 <= ho.nice_after_lp_spawn <= 19, "host.nice_after_lp_spawn is an integer in [0, 19]")
        # without the EA's own LP worker the renice never happens: the variant would be mislabelled
        _check(ho.nice_after_lp_spawn == 0 or sd.lp_source == "highs_async",
               "host.nice_after_lp_spawn needs seeding.lp_source highs_async (it acts after the LP worker's spawn)")
        _check(self.backend in ("cupy", "numpy", "auto"), "backend in cupy|numpy|auto")
        _check(self.gpu_mem_gb > 0, "gpu_mem_gb > 0")
        _check(self.log_every >= 1 and self.lineage_depth >= 0, "logging parameters")
        return self

    # --------------------------------------------------------- derived
    def n_rank0_cap(self) -> int:
        return math.floor(self.ranks.x * self.population.size)


def _check(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"invalid EA configuration: {what}")


def _build(cls, d: dict, where: str):
    if not isinstance(d, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(d).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(d) - set(fields)
    if unknown:
        raise KeyError(f"{where}: unknown keys {sorted(unknown)}")
    kw = {}
    for k, v in d.items():
        f = fields[k]
        sub = f.default_factory if f.default_factory is not dataclasses.MISSING else None
        if sub is not None and dataclasses.is_dataclass(sub):
            kw[k] = _build(sub, v, f"{where}.{k}")
        else:
            kw[k] = v
    return cls(**kw)
