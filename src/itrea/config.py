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
                                       # continuous variables moved by FJ jump values | "gene" | "lp_decode"
    lp_decode_per_gen: int = 4         # lp_decode: CPU LP decodes per generation (candidates for Rank 1 / posting)
    lp_time_limit: float = 2.0         # seconds per LP decode


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
    backend: str = "cupy"             # "cupy" | "numpy" — never a silent fallback ("auto" only with allow_cpu_fallback)
    gpu_mem_gb: float = 12.0           # CuPy pool limit (the only GPU memory cap on GB10) = cuOpt's RMM limit in A2:
                                       # equal GPU memory for both partners; cuOpt needs ~10.5 GB (measured 2026-10-06)
    log_every: int = 10                # generations between generation-log records
    lineage_depth: int = 5             # generations of ancestry recorded for every new best

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        """12-hex digest of the canonical JSON of every parameter except `name` — and except the `strategy` section
        while `strategy.enabled` is false: it is inert then, so every configuration written before the section existed
        keeps its hash (the default's `9ac4a93bf93b` among them)."""
        d = self.to_dict()
        d.pop("name", None)
        if not self.strategy.enabled:
            d.pop("strategy", None)
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
        _check(r.rank0_select in ("itrea", "violation", "rank_prob"), "ranks.rank0_select in itrea|violation|rank_prob")
        _check(r.rank2_select in ("itrea", "key_then_violation"), "ranks.rank2_select")
        _check(r.rank2_replacement in ("generational", "plus"), "ranks.rank2_replacement in generational|plus")
        _check(r.tol_mode in ("scip", "highs", "rhs"), "ranks.tol_mode in scip|highs|rhs")
        _check(r.viol_norm in ("l2", "rhs", "count"), "ranks.viol_norm in l2|rhs|count")
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
        _check(e.poll_s > 0 and e.inject_k >= 1 and e.inject_band >= 0, "exchange parameters")
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
