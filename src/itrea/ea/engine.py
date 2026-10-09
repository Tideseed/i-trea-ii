"""The I-TREA II main loop: one population on one device, the solver as the "interactive" partner.

    eng = Engine(prob, cfg, seed, mailbox=None, trace=None)
    startup = eng.prepare()          # BEFORE the shared clock: backend, device copy, sizes, operators, warm-up
    summary = eng.run(budget, t0)    # AFTER t0: seed, evaluate, survive, then generations until the budget / stop
    eng.write_outputs(run_dir)       # ea.incumbents.npz, ea.genealogy.npz, ea.frontier.npz
Tests and tools may drive the loop by hand: `begin(t0)`, then `step()` per generation (the population is `eng.pop`
with host mirrors `eng.h_key`, `eng.h_viol`, `eng.h_feas`), then `finish()`.

The clock rule (vault D-002 v3, design critique 2026-10-05, "clock / start-up fairness"). Before t0 only what every arm
may do: imports, the instance parse, the CUDA context, the device copy of the problem, building the operators, and a
kernel WARM-UP ON A SYNTHETIC PROBLEM with the instance's dtypes and variable classes (`synthetic_problem`), so NVRTC
and Numba compile before the clock without touching the instance's search space. Nothing about the instance is solved
or evaluated before t0: the initial population is drawn, repaired and evaluated in `run()`, and an LP point arrives
only on the clock — the solver's root LP through the mailbox (`seeding.lp_source: solver`) or HiGHS in a worker
process started at t0 (`highs_async`). The deadline is t0 + budget on `time.monotonic()`, the clock of the mailbox.

Backend. `cfg.backend` is "cupy" or "numpy"; "auto" is refused unless `allow_cpu_fallback=True` (tests only): a run
labelled as a GPU arm must never run on the CPU silently (critique S4).

Population size (`population.size_rule`). "memory": P = the largest P <= `population.size` whose buffers fit
`cfg.gpu_mem_gb` (`plan_memory`: population, offspring with their parents and crossover masks, the candidate set with
its duplicate-hash temporaries and the survivors' copy, the frontier update, immigrants, plus one transient budget for
evaluation / local search / operators, plus the device problem and the local search's matrices; 10 % headroom). The
offspring count scales with P (`offspring_per_gen · P / size`). P is a function of the instance's shape and of the
sizes in the configuration only: it is computed the same way on NumPy, and the switches that differ between the arms
of a block (backend, `exchange.enabled`, `frontier.enabled`, `ranks.x`) do not enter it — the injection rows and the
frontier archive are budgeted whether or not the arm uses them — so A3, A4, A5 and A5x0 of one block run the same P.
A P whose Rank 0 quota floor(x·P) is 0 while x > 0 is refused afterwards (it would silently be the x = 0 ablation):
the instance does not fit.
"fixed": exactly `size`; refused if it does not fit.
With `strategy.enabled` the plan adds `local_search_strategies` (`strategy_ls_bytes`): the one local search keeps a
row of divisors per row scale, so every scale beyond the first costs 8 bytes per augmented row (m + 1 with the cutoff
row); the per-call (k,) scale index and the dense baseline's gathered divisors live inside the local search's transient
budget. No peak term changes; P may still differ from the disabled arm's by that term.

One generation (`step`, vault `10-design/I-TREA II algorithm.md`):
  1. exchange in, every `exchange.poll_s`. Only when `exchange.enabled and from_solver` (and a mailbox exists) are the
     solver channels read at all — `to_ea`, `info`, `lp` AND `verdicts`; the one-way "EA -> solver" arm never sees a
     solver incumbent, bound, LP point or verdict (critique A12): inc_external = +inf, dual bound = -inf. Solutions:
     echo rows (ORIGIN_EA_ECHO, the EA's own posts coming back) are skipped and counted; the rest within
     `inject_band` of their best, at most `inject_k` (`solvers.common.band_select`), integers rounded, cast to the gene
     dtype, join the next candidate set with ORIGIN_INJECTED (never repaired). Info: incumbent and proven bound as
     keys (sense · objective). LP: the point (once) — LP-rounded individuals (`lp_round`, share
     `seeding.lp_rounding` of the immigrant count) join the next candidate set, and `toward_lp` mutation uses it.
     The stop file is checked at every tick whatever the direction (control plane). cuOpt's channels (arm A9) are read
     only with `exchange.from_cuopt` ("Three-way arm A9" below).
  2. parents: `selection.select_parents` on the host; 3. `crossover` -> (children, origin bits, touched); children
     that were not crossed are mutation-only (pb = pa); 4. `mutate` in place;
  5. repair: `SingletonDecoder` (when `repair.continuous` is "auto" or "lp_decode" and the problem has singleton
     continuous columns)
     -> `LocalSearch.run` -> decoder again. The search's objective cutoff is the incumbent key min(best feasible in
     the population, solver's incumbent), refreshed whenever it changes. `touched` is NOT passed: the repair has no
     restriction to the disagreement set yet (known gap). ORIGIN_REPAIR marks children the search moved.
     The search polls `_should_stop` (the stop request, the deadline t0 + budget, the mailbox's stop flag at most every
     STOP_POLL_S) between kernel launches / row slices / steps and returns early; if the budget ended or a stop
     arrived during the repair, the generation is cut short here (its children are dropped, `aborted_generations`).
  6. evaluate (`eval_dtype`, `ranks.tol_mode`); ancestry rows (`population.child_ancestry`, parents' ranks at mating,
     SRC bits of injected / seeded parents);
  7. candidates = population ∪ children ∪ pending (injected, LP-rounded) — population first, so one `DupHasher`
     (built once per run) keeps the existing member of an exact duplicate group; the dropped copies' origin bits are
     OR-ed into the kept member (critique A13; a solver vector already in the population is counted).
     `ranks.three_rank_survival(..., inc_external, dual_bound, rng, dup, ids, is_new, prev_r1)` (prev_r1 = the
     previous Rank 1, for the Rank 1 ratchet); every NEW Rank 1 entrant is re-checked on the host in float64
     (`Problem.check`, `tol_mode`, integers rounded) and a failure is demoted (feasible = False) and the survival
     repeated (counted `false_feasible`) until Rank 1 holds only re-checked members (at most #feasible + 1 rounds).
     The population = candidates.take(idx, rank=survival.rank). A survival shortfall is filled with immigrants (below).
  8. a new EA incumbent (best feasible key improved after rounding + float64 check): trace `incumbent` (source "self",
     or "solver" for an injected vector carrying no EA operator bit — "cuopt" when that vector came from cuOpt, A9) and
     `best` events, every incumbent vector kept for `trace.save_incumbents`; posted to the solver when
     `exchange.enabled and to_solver` and it beats the solver's incumbent known to the EA by `post_min_gain`
     (relative). During the feasibility phase a `best` event marks a lower
     total violation. Solver verdicts (read only when from_solver): a rejected post demotes that individual
     (`solver_rejections`; a Rank 1 member moves to Rank 2 at once), its vector hash is remembered so a re-created
     copy fails the float64 check (`rejected_copies`), and if it was the EA's current incumbent the incumbent falls
     back to the newest one not rejected (or to none: `incumbents_reverted`) — later solutions better than that are
     recorded and posted again, and the final best_x is never a rejected vector. An accepted post is marked in the
     genealogy.
  9. stagnation: after `stagnation.gens` generations without a new best (a new incumbent; during the feasibility phase
     a lower best violation) the worst `immigrant_frac` of Rank 2 is replaced by `seeding.immigrants` — frontier rows
     (`frontier.reinject` of them), LP roundings (`seeding.lp_rounding` of the rest, once an LP point exists), random
     points. Rank 0 and Rank 1 are untouched.
  9b. frontier (D-005), every `frontier.update_every` generations: `Frontier.update` with the feasible candidates in
     the band (at most 2 · max_size besides the incumbent — memory), trace `frontier`; with `exchange.post_frontier`
     the archive rows that pass the float64 check are posted as one message.
  10. a `gen` event every `log_every` generations and at the end (field list: vault `10-design/Trace format and
     metrics.md`, incl. `superopt_survivors`, the manipulation check of the Rank 0 cap, and per-phase seconds timed with
     CUDA events on the GPU).

Self-adaptive repair strategy (`strategy.enabled`, `StrategyCfg`; off by default, and then nothing below happens and
no extra random number is drawn — the run is the one without the feature). A strategy is an index into
`StrategyCfg.strategies()` = row_scales x lp_guidance (scale-major), carried per individual (`Population.strat`).
  - prepare: ONE LocalSearch with `row_scales=strategy.row_scales` (one row of divisors per scale; `repair: none` has
    no row scale and ignores it); the strategy -> row-scale index and strategy -> LP-guidance tables.
  - begin: the initial population gets the strategies round robin (row i -> i mod S).
  - step: after parent selection each child takes parent A's strategy, redrawn uniformly with probability
    `mutation_rate`; then every strategy below min(ceil(min_share·O), O // S) children is topped up with children drawn
    at random from the strategies above that floor (`strategy_reassigned`). Nothing is sorted or grouped: ONE mutate
    call with `lp_rows` = the children whose strategy has LP guidance (the others never draw toward_lp), and ONE
    LocalSearch call with `scale_idx` = each child's row-scale index, so all scales' individuals are repaired side by
    side (one GPU tail per generation; `ls_fj_gpu`). Evaluation, survival (V on `ranks.viol_norm` for everyone),
    exchange and the frontier are unchanged.
  - immigrants (shortfall, stagnation, empty population) and injected solver rows draw uniform random strategies (each
    row independently, in the order the rows come); LP roundings go only to immigrants whose strategy has LP guidance
    (lp_share of those), and the LP-rounded batch made when the LP point arrives gets random LP-guided strategies
    (none at all if no strategy has LP guidance). Every repair (children, initial population, immigrants, the LP
    batch) is one call with a scale index.
  - records: genealogy column `strat` (always written; 0 when off); `best` events carry `strat` and `strat_credited`;
    `gen` events add `strat_pop` (survivors per strategy), `strat_best` (credited best events per strategy since the
    previous gen record) and `strat_children`; the start-up record lists the strategies and the local search's row
    scales; the summary's `strategy` holds the cumulative counts per strategy. A new population best that is a solver
    solution — injected and carrying no EA operator bit, the rule of the incumbent's source "solver" — is NOT credited
    to the strategy the injected row was given: it is counted in `incumbents_solver` / `best_events_solver` instead of
    `incumbents` / `best_events` / `strat_best`. `repaired` counts rows given to the local search per strategy,
    `repaired_feasible` the rows it finished feasible (0 under `repair: none`, which reports no per-row status).
    `ls_calls` counts local-search calls (one per repair, whatever the number of scales).

Repair clock (`repair_clock.enabled`, `RepairClockCfg`, `ea.repair_clock`; off by default, and then nothing below
happens: no argument is added to a local-search call, no counter, no event field, no random number). One level per
generation — a lane-step horizon 2^j or "none" — for every repair call of it.
  - begin: generation 0 is decided "none" (no feasible member yet) and recorded like every later one.
  - step: right after the exchange tick `RepairClock.decide(now, has_feasible)`, has_feasible = a feasible member of
    the EA's own (`_rc_own_feasible`: an unchanged solver solution — `_is_solver_find` — does not count). `_ls_run`
    passes `horizon` / `cap_s` only in active mode, after the population first held such a member, and never for the
    one-shot LP batch (`pin_lp`); shadow mode never passes them, so no local-search call changes.
  - every repair call (initial population "init", children, each repaired pending batch "lp" / "lps", `_admit`'s
    seeded rows "admit") is kept with its lanes' steps, status and launch profile, and `_rc_pair` — the one pairing
    helper — hands it to the clock with the evaluated keys and feasibility of the same rows (lane i = row i, or row
    rep[i] in `_admit`, which repairs only the seeded rows but evaluates all). A call left unpaired by a generation cut
    short is dropped at the next decision. The calls made after the survival (`_after_survival`: shortfall fill,
    stagnation immigrants) are marked `counterfactual=False`: whether they happen at all depends on the generation's
    earlier calls, so the clock leaves them out of the gain.
  - end of the generation: `end_generation` (the measured generation seconds) and one `rclock` event per generation;
    `gen` events add the `rc_*` fields, the summary a `repair_clock` block, the counters `ls_finished_horizon` and
    `ls_cap_hits` (vault `10-design/Trace format and metrics.md`).

Three-way arm A9 (`exchange.from_cuopt`, D-015; off by default, and then cuOpt's channels are never read and nothing
below happens). cuOpt runs beside the solver and posts its incumbents on its OWN channels (`to_ea_cuopt`,
`info_cuopt`). At every exchange tick, after the solver's channels:
  - rows: as the solver's (step 1: band, integers rounded, cast, ORIGIN_INJECTED, never repaired, survival decides),
    trace kind `inject_cuopt`, counters `cuopt_inject_msgs`, `cuopt_inject_rows`, `cuopt_injected`,
    `cuopt_injected_infeasible`, `cuopt_injected_dup`. The ids they receive form `cuopt_ids`: an individual's vector
    never changes after entry, so the id marks "this is cuOpt's vector". A vector the solver's channel delivered
    first stays the solver's (the duplicate merges into the earlier member).
  - info: cuOpt's incumbent (key) is `inc_cuopt`; `inc_external` = min(the solver's `inc_solver`, `inc_cuopt`) is the
    external incumbent of the survival (Rank 0's "better than the incumbent"), `_admit`, the cutoff and the
    feasibility-phase test. cuOpt's bound is recorded (`dual_key_cuopt`) but NOT used: Rank 0's bound demotion keeps
    the solver's proven bound alone, so in the engine A9 differs from its matched control A3x (the runner's A3 with
    A9's EA footprint, no cuOpt) by cuOpt's solutions and incumbent only. Not from A3: the runner gives A9's EA one
    X925 core, which with `seeding.lp_source: highs_async` it shares with the HiGHS LP worker (it inherits the EA's
    affinity), where A3's EA has three A725 cores (`itrea.experiments.pair`).
  - attribution: an incumbent that is an unchanged cuOpt vector (`_is_solver_find` and an id in `cuopt_ids`) has
    source "cuopt" (counted in `cuopt_incumbents`; like a solver find, not credited to a repair strategy). A post is
    worth sending when it beats the SOLVER's incumbent (`inc_solver`), not the min — an unchanged cuOpt vector that
    beats SCIP's incumbent is handed to SCIP, which is the point of the arm — and its row carries POST_CUOPT (bit 16 of
    the message's int64 origin) so SCIP records the incumbent it makes as source "cuopt", not "ea". `post` events get
    `cuopt_rows`; `cuopt_posts` counts those posts, `cuopt_rows_accepted` / `cuopt_rows_rejected` the solver's
    verdicts on their cuOpt ROWS (the unit of `solver_accepted`: a frontier post of several cuOpt rows, or a re-posted
    vector, gives several; per-post counts are build_runs'); `best` events get `cuopt`, `gen` events `inc_solver`,
    `inc_cuopt`, `dual_key_cuopt`, `cuopt_pop`; the genealogy a `cuopt` column.

LP decode (`repair.continuous: lp_decode`, D-017, `ea.lp_decode`; "auto" plus the decode — with any other mode, or on
an instance without an active column (continuous, lb < ub, in two or more rows), nothing below happens: no worker, no
counter, no event, no extra random number; the start-up record says `lp_decode.state = "inert"`).
  - prepare: the active columns, `cont_obj`, the integer payload's dtype; the instance file is required (the worker
    reads it); `plan_memory` counts `lp_decode_per_gen` more pending rows (at most one decoded batch of at most N =
    `lp_decode_per_gen` rows joins a candidate set, below). The warm-up runs the whole path with an in-process stub
    (`_WarmupDecoder`), forced to batches of three rows and of one row: the start-up record's `lp_decode_warmed` is the
    largest decoded batch (rows) that entered a candidate set in that pass (3 in "many", 1 in "one"),
    `lp_decode_warmed_batches` their number.
  - begin (t0): the `LPDecoder` worker process starts beside the LP worker; `lpd_*` counters.
  - the pick site, every generation between the survival and `_after_survival`, over the whole candidate set: poll
    the worker; if it is not idle count `lpd_busy`, else select one batch from the survival's host arrays (after the
    demotions) with `select_offers`, wait for the device (`bk.synchronize()`, timed apart as `lpd_sync_s`: the
    survival's last gather is not the decode's cost), copy the rows to the host one by one and offer the rounded,
    clipped integer parts. At most one request is outstanding. A source counts as offered (never offered again) once a
    reply screened or solved it — rows left NOT_SOLVED or AFTER_DEADLINE by the worker's budget stay eligible.
  - reading replies: the worker is polled at every exchange tick, just before the candidate set takes the pending
    batches (`_with_pending`, after the children are evaluated) and at the pick site; `finish` reads the last one. At
    most ONE decoded batch joins a candidate set: while a decoded batch is pending, the exchange and pre-candidate polls
    leave the worker's messages in the pipe. A batch's copies join the candidate set of the generation that reads the
    reply before its `_with_pending` — the generation after the offer when the worker answers within a generation, the
    usual case — or, when a pick site reads it (it arrived during that generation's survival), the next generation's;
    the pick site reads before it offers, so the worker never idles a generation. The result event records the read
    `site` and `gen_enters`.
  - a reply: per row a code (ok / infeasible / time_limit / other / range_fail / cached / not_solved /
    after_deadline); a copy = the source's integer part + the LP's continuous values clipped to their bounds; a polish
    (F) copy is kept only if its key beats its source's by imp_tol (`lpd_no_gain` otherwise), a completion (I) copy
    always; the kept copies join a candidate set as one `_Batch` (src "lp_decode", never repaired) with origin
    ORIGIN_LP_DECODE (an EA bit: their incumbents have source "self"), their source's strategy (no draw), ancestry =
    `child_ancestry` with the source as both parents (its rank in that survival, NO_RANK if eliminated), pa = pb = -1;
    genealogy column `lpd_src` = the source's id. One `lp_decode` trace event per reply (and at "ready" / "failed").
  - incumbents and posts made from a copy are counted by id, pooled and per kind of its source (`lpd_incumbents`,
    `lpd_posts`, `_F` / `_I`); so are the copies that entered the population (`lpd_entered`, `_F` / `_I`). `gen`
    events add `lpd_state` and the counters in `LPD_GEN_FIELDS`, `lpd_host_s`, `lpd_host_s_max`, `lpd_sync_s`; the
    summary an `lp_decode` block: state, the EA's read time of "ready" (`ready_t`) and the worker's own ready time
    (`ready_t_worker`, M0), EA-thread seconds `host_s` / `host_s_max` (perf_counter; the device wait before the row
    copies excluded — it is `sync_s` / `sync_s_max`), the reply wait (`wait_s_total` / `wait_s_max`: worker send to
    EA read), the IPC times (`ipc`, `LPDecoder.ipc`), the largest decoded batch, the worker's statistics.
  - a failed worker (digest mismatch, exception, death, a failed send) stays failed for the run and is recorded
    (`lpd_failed`, `lp_decode` event "failed", the error); the EA never waits on it (the worker is signalled, not
    joined). `finish` reads a reply waiting at the end and then stops it (`run`'s finally stops it after an
    exception); a request still outstanding then counts as `lpd_lost` (usually one at the deadline).

Immigrants (shortfall fill and stagnation) are repaired like children (frontier rows are not: they are feasible
archive members), evaluated, de-duplicated against the population and admitted with the survival's rules: an
infeasible immigrant better than the incumbent takes a Rank 0 slot only while the quota floor(x·P) has room (and not
beyond the solver's proven bound) and is dropped otherwise — the quota stays a total cap; the rest are Rank 2.

The initial population (`seeding.seed_population`, no LP) is repaired and evaluated after t0 like any children before
its first survival — feasibility jump from random points is how the population first becomes feasible. (A decision of
this implementation; the design note's step list starts at the first generation.)

Genealogy (`ea.genealogy.npz`, D-002 §4, read by `itrea.analysis.lineage`): one row per individual that entered the
population — id, pa, pb, rank_pa, rank_pb (at mating; -1 for seeds, immigrants and injected), origin (OR-merged over
exact duplicates, also after entry), birth, key, V, feasible (at entry), t_first (seconds since t0 when it was made),
is_new_best, is_accepted_post (written explicitly, so the analysis does not have to derive them), strat (the repair
strategy index; 0 while the strategy is off). Because origin is OR-merged over duplicates, a row whose strategy has no
LP guidance can carry ORIGIN_M_TOWARD_LP (a duplicate made by an LP-guided child): analyses of LP guidance must key on
`strat`, not on origin bits.

Every incumbent vector of the EA goes to `ea.incumbents.npz` (`trace.save_incumbents`; vectors in the canonical order
`prob.names`) — P(T) is built only from checked incumbents (D-002). The frontier archive goes to `ea.frontier.npz`.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sps

from itrea.backend import Backend, get_backend
from itrea.config import EAConfig
from itrea.ea.continuous import SingletonDecoder
from itrea.ea.crossover import crossover
from itrea.ea.evaluate import DeviceProblem, evaluate, to_device
from itrea.ea.frontier import Frontier
from itrea.ea.mutation import VIOL_TOL as VG_VIOL_TOL
from itrea.ea.mutation import mutate
from itrea.ea.population import (NO_RANK, ORIGIN_IMMIGRANT, ORIGIN_INJECTED, ORIGIN_LP_DECODE, ORIGIN_REPAIR,
                                  ORIGIN_SEED_LP, ORIGIN_SEED_RANDOM, R0, R1, R2, DupHasher, IdCounter, Population,
                                  child_ancestry, gene_dtype_for, origin_names, src_bits_of_origin)
from itrea.ea.elite import clearing_pick, elite_spread, min_diff_auto
from itrea.ea import lp_decode as lpdec
from itrea.ea.ranks import Survival, rank1_cap, three_rank_survival
from itrea.ea.repair import LocalSearch
from itrea.ea.repair_clock import RepairClock
from itrea.ea.seeding import LPWorker, immigrants, lp_round, seed_population
from itrea.ea.selection import select_parents
from itrea.problem import Problem, from_arrays
from itrea.solvers.common import ORIGIN_EA_ECHO, POST_CUOPT, SOURCE_CODES, band_select, vector_hash
from itrea.trace import save_incumbents

HEADROOM = 0.9                 # share of gpu_mem_gb the memory plan may fill (pool fragmentation, library workspaces)
FRONTIER_OFFER = 2             # frontier.update gets at most FRONTIER_OFFER * max_size candidates besides the incumbent
MAX_DEMOTIONS = 8              # float64 re-check failures handled per incumbent search before giving up
STOP_POLL_S = 0.25             # inside a generation, the mailbox's stop flag is read at most this often (a Redis call)
OOM_RETRIES = 3
_GENE_BITS_EA = ~np.uint16(ORIGIN_INJECTED)   # every origin bit except "came from the solver"


class DoesNotFit(MemoryError):
    """The instance's population cannot fit the GPU budget (D-002 §2: such an instance is replaced in the pool)."""


# ------------------------------------------------------------------------------------------------- memory plan
@dataclass
class MemoryPlan:
    P: int
    offspring: int
    limit_bytes: int
    total_bytes: int
    terms: dict
    budgets: dict
    fits: bool

    def as_dict(self) -> dict:
        mb = 1024 ** 2
        return dict(P=self.P, offspring=self.offspring, limit_mb=round(self.limit_bytes / mb, 1),
                    total_mb=round(self.total_bytes / mb, 1), fits=self.fits,
                    terms_mb={k: round(v / mb, 2) for k, v in self.terms.items()},
                    budgets_mb={k: round(v / mb, 2) for k, v in self.budgets.items()})


def transient_budgets(cfg: EAConfig) -> dict:
    """Byte budgets of the chunked phases (evaluation, local search, operators); they never run at the same time."""
    lim = int(cfg.gpu_mem_gb * 1024 ** 3)
    b = int(min(1 << 30, max(32 << 20, lim // 8)))
    return dict(evaluate=b, local_search=b, operators=b)


def _offspring(cfg: EAConfig, P: int) -> int:
    pc = cfg.population
    if pc.size_rule == "fixed":
        return int(pc.offspring_per_gen)
    return max(1, int(round(pc.offspring_per_gen * P / pc.size)))


def plan_memory(prob: Problem, cfg: EAConfig, gene_dtype: np.dtype) -> MemoryPlan:
    """The population size under `population.size_rule` and the bytes it needs (see the module docstring)."""
    n, m, nnz = prob.n, prob.m, prob.nnz
    e = np.dtype(cfg.population.eval_dtype).itemsize
    g = np.dtype(gene_dtype).itemsize
    dd = 8 if np.dtype(gene_dtype).kind in "iu" else 16          # DupHasher: float64 cast (+ rint copy for floats)
    F = cfg.frontier.max_size                                     # counted whether or not the arm keeps the archive
    budgets = transient_budgets(cfg)
    op = max(budgets.values())
    limit = int(cfg.gpu_mem_gb * 1024 ** 3)
    fixed = dict(device_problem=nnz * (2 * e + 12) + m * (6 * e + 4) + n * (4 * e + 17),
                 local_search=24 * (nnz + n) + 4 * (m + n) + 32 * n,
                 crossover_patterns=2 * nnz * (e + 4),
                 dedup_projection=16 * n,
                 frontier_archive=F * n * e)
    if cfg.ranks.feas_alt_scale != "none":                        # a second device problem (ranks.feas_alt_scale)
        fixed["device_problem_alt"] = fixed["device_problem"]
    # the LP decode's copies: at most one decoded batch of at most lp_decode_per_gen rows joins a candidate set (the
    # module docstring "LP decode"), only where it can act
    K_lpd = (int(cfg.repair.lp_decode_per_gen)
             if cfg.repair.continuous == "lp_decode" and lpdec.active_columns(prob).size else 0)

    def terms(P: int) -> dict:
        O = _offspring(cfg, P)
        K = cfg.exchange.inject_k + math.ceil(cfg.seeding.lp_rounding * cfg.stagnation.immigrant_frac * P) + K_lpd
        C = P + O + K
        ph = dict(breed=(P + 3 * O) * n * g + O * n + op,
                  repair_evaluate=(P + O) * n * g + op,
                  survival=2 * C * n * g + C * n * dd + P * n * g,
                  frontier=(P + C) * n * g + (1 + (1 + FRONTIER_OFFER) * F) * n * (e + 16),
                  immigrants=5 * P * n * g + 2 * P * n * dd + op)
        return ph

    if cfg.strategy.enabled:
        fixed["local_search_strategies"] = strategy_ls_bytes(prob, cfg)

    def total(P: int) -> tuple[int, dict]:
        ph = terms(P)
        t = sum(fixed.values()) + max(ph.values())
        return t, dict(fixed, **{f"peak_{k}": v for k, v in ph.items()})

    usable = HEADROOM * limit
    size = int(cfg.population.size)
    if cfg.population.size_rule == "fixed":
        P = size
    else:
        lo, hi = 2, size
        if total(lo)[0] > usable:
            P = lo
        else:
            while lo < hi:                                      # largest P in [lo, hi] that fits (monotone in P)
                mid = (lo + hi + 1) // 2
                if total(mid)[0] <= usable:
                    lo = mid
                else:
                    hi = mid - 1
            P = lo
    t, tm = total(P)
    return MemoryPlan(P=P, offspring=_offspring(cfg, P), limit_bytes=limit, total_bytes=int(t), terms=tm,
                      budgets=budgets, fits=bool(t <= usable))


def strategy_ls_bytes(prob: Problem, cfg: EAConfig) -> int:
    """Device bytes the self-adaptive repair strategy adds to the local search (0 when it is off or the repair is
    "none", which has no row scale): the one LocalSearch keeps a float64 row of divisors per row scale (the stacked
    (R, m_aug) table of `ls_fj_gpu` / `ls_dense`, host arrays in `ls_fj_cpu`), so every scale beyond the first costs
    8 bytes per augmented row (m + 1 with the cutoff row). Nothing else is per scale."""
    sg = cfg.strategy
    if not sg.enabled or cfg.repair.method == "none":
        return 0
    return int((len(sg.row_scales) - 1) * 8 * (prob.m + 1))


# --------------------------------------------------------------------------------------------- synthetic problem
def synthetic_problem(prob: Problem, seed: int = 0) -> Problem:
    """A small problem with the instance's VARIABLE AND ROW CLASSES and objective sense, and a known feasible point
    (`toy_point`) — the warm-up target. Nothing in it depends on the instance's values.

    Each class gets min(the instance's count, a small cap) members, so a class with ONE member on the instance has one
    here too: CuPy specialises kernels on the number of dimensions left after size-1 axes collapse (a one-row problem
    makes (m, k) activity arrays effectively 1-D), so the warm-up must see the same degenerate shapes. Classes:
    binaries, general integers (bounded / unbounded), continuous (bounded / with an infinite side / singleton columns
    in one row), rows (<=, >=, =, ranged). Integer bounds stay inside [-3, 5] so every gene dtype holds them."""
    rs = np.random.default_rng(seed)
    isint = np.asarray(prob.isint, bool)
    lb, ub = np.asarray(prob.lb, float), np.asarray(prob.ub, float)
    rl, ru = np.asarray(prob.rl, float), np.asarray(prob.ru, float)
    fin_lb, fin_ub = np.isfinite(lb), np.isfinite(ub)
    deg = np.diff(sps.csc_matrix(prob.A).indptr) if prob.n else np.zeros(0, int)
    conts = np.asarray(prob.conts)
    single = conts[(deg[conts] == 1) & (lb[conts] < ub[conts])] if conts.size else conts
    other = np.setdiff1d(conts, single)
    gints = np.asarray(prob.gints)
    n_bin = min(int(prob.bins.size), 24)
    n_gint_b = min(int(np.sum(fin_lb[gints] & fin_ub[gints])), 6)
    n_gint_u = min(int(np.sum(~(fin_lb[gints] & fin_ub[gints]))), 2)
    n_cont_b = min(int(np.sum(fin_lb[other] & fin_ub[other])), 5)
    n_cont_u = min(int(np.sum(~(fin_lb[other] & fin_ub[other]))), 2)
    n_single = min(int(single.size), 2) if prob.m else 0
    eq = rl == ru
    fl, fu = np.isfinite(rl), np.isfinite(ru)
    n_le = min(int(np.sum(~fl & fu)), 3)
    n_ge = min(int(np.sum(fl & ~fu)), 3)
    n_eq = min(int(np.sum(eq)), 3)
    n_rg = min(int(np.sum(fl & fu & ~eq)), 3)
    if n_bin + n_gint_b + n_gint_u + n_cont_b + n_cont_u == 0:
        n_bin = 1
    cols: list[tuple[float, float, bool]] = []                   # (lb, ub, integer)
    cols += [(0.0, 1.0, True)] * n_bin
    cols += [(-3.0, 5.0, True)] * n_gint_b
    cols += [(0.0, np.inf, True)] * n_gint_u
    cols += [(0.0, 10.0, False)] * n_cont_b
    cols += [(-np.inf, np.inf, False)] * n_cont_u
    n_main = len(cols)
    cols += [(0.0, 50.0, False)] * n_single
    n = len(cols)
    lbs = np.array([c[0] for c in cols])
    ubs = np.array([c[1] for c in cols])
    ints = np.array([c[2] for c in cols], dtype=bool)
    x0 = np.zeros(n)
    for j, (lo, hi, it) in enumerate(cols):
        bot = lo if np.isfinite(lo) else (hi - 6.0 if np.isfinite(hi) else -3.0)
        top = hi if np.isfinite(hi) else bot + 6.0
        x0[j] = rs.integers(int(bot), int(top) + 1) if it else rs.uniform(bot, min(top, bot + 5.0))
    kinds = ["le"] * n_le + ["ge"] * n_ge + ["eq"] * n_eq + ["rg"] * n_rg
    m = len(kinds)
    rows, cidx, vals = [], [], []
    for i in range(m):
        for j in rs.choice(n_main, size=min(int(rs.integers(3, 6)), n_main), replace=False):
            rows.append(i)
            cidx.append(int(j))
            vals.append(float(rs.integers(1, 5) * rs.choice([-1, 1])))
    if m:
        seen = set(cidx)
        for j in range(n_main):                                  # every main column in at least one row
            if j not in seen:
                rows.append(int(rs.integers(m)))
                cidx.append(j)
                vals.append(1.0)
        for s_, j in enumerate(range(n_main, n)):                # singletons: exactly one row each
            rows.append(s_ % m)
            cidx.append(j)
            vals.append(1.0 if s_ % 2 == 0 else -1.0)
    A = sps.csr_matrix((vals, (rows, cidx)), shape=(m, n))
    A.sum_duplicates()
    act = A @ x0
    srl, sru = np.full(m, -np.inf), np.full(m, np.inf)
    for i, kd in enumerate(kinds):
        if kd == "le":
            sru[i] = act[i] + 1.0
        elif kd == "ge":
            srl[i] = act[i] - 1.0
        elif kd == "eq":
            srl[i] = sru[i] = act[i]
        else:
            srl[i], sru[i] = act[i] - 1.0, act[i] + 2.0
    c = np.where(ints, rs.integers(1, 10, n) * rs.choice([-1, 1], n), rs.uniform(-3, 3, n)).astype(float)
    p = from_arrays(A, srl, sru, lbs, ubs, c, ints, sense=prob.sense, names=[f"w{j}" for j in range(n)],
                    path="<synthetic warm-up>")
    p.toy_point = x0  # type: ignore[attr-defined]
    return p


# ------------------------------------------------------------------------------------------------- helpers
class _NullTrace:
    """Trace stand-in (warm-up, tests): keeps the clock, drops events."""

    def __init__(self):
        self.t0: float | None = None

    def start(self, t0: float) -> None:
        self.t0 = t0

    def now(self) -> float:
        return (time.monotonic() - self.t0) if self.t0 is not None else float("nan")

    def event(self, kind: str, t: float | None = None, **data) -> None:
        pass

    def incumbent(self, obj: float, source: str = "self", t: float | None = None, **extra) -> None:
        pass

    def set(self, **kv) -> None:
        pass


class _Phases:
    """Per-phase seconds: CUDA events on the GPU (read once per generation), perf_counter on the CPU."""

    def __init__(self, bk: Backend):
        self.gpu = bk.is_gpu
        self.acc: dict[str, float] = {}
        self._pending: list = []

    @contextlib.contextmanager
    def __call__(self, name: str):
        if self.gpu:
            import cupy as cp
            s = cp.cuda.Event()
            s.record()
            try:
                yield
            finally:
                e = cp.cuda.Event()
                e.record()
                self._pending.append((name, s, e))
        else:
            t = time.perf_counter()
            try:
                yield
            finally:
                self.acc[name] = self.acc.get(name, 0.0) + time.perf_counter() - t

    def collect(self) -> None:
        if self._pending:
            import cupy as cp
            self._pending[-1][2].synchronize()
            for name, s, e in self._pending:
                self.acc[name] = self.acc.get(name, 0.0) + cp.cuda.get_elapsed_time(s, e) / 1000.0
            self._pending = []

    def take(self) -> dict:
        self.collect()
        out = {k: round(v, 6) for k, v in self.acc.items()}
        self.acc = {}
        return out


class _SparseRow:
    """An incumbent vector kept as its non-zeros until the .npz is written."""
    __slots__ = ("n", "idx", "val")

    def __init__(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).ravel()
        self.n = x.size
        self.idx = np.flatnonzero(x)
        self.val = x[self.idx].copy()

    def dense(self) -> np.ndarray:
        out = np.zeros(self.n)
        out[self.idx] = self.val
        return out


@dataclass
class _Batch:
    """Individuals waiting for the next candidate set (injected solver solutions, LP roundings)."""
    X: object
    origin: np.ndarray
    repair: bool
    src: str
    t: float
    strat: np.ndarray | None = None      # (k,) uint8 repair strategies, row order (None = all 0)
    anc_rank: np.ndarray | None = None   # (k, depth) ancestry rows (None = zeros: no parent); LP decode copies only
    anc_ops: np.ndarray | None = None
    src_ids: np.ndarray | None = None    # (k,) LP decode copies: the id of each copy's source
    src_kinds: np.ndarray | None = None  # (k,) LP decode copies: the kind of each copy's source (KIND_F / KIND_I)


LPD_COUNTERS = ("lpd_offers", "lpd_rows_offered", "lpd_rows_F", "lpd_rows_I", "lpd_busy", "lpd_replies", "lpd_solved",
                "lpd_solved_F", "lpd_solved_I", *(f"lpd_{nm}" for nm in lpdec.CODE_NAMES), "lpd_ok_F", "lpd_ok_I",
                "lpd_no_gain", "lpd_kept", "lpd_kept_F", "lpd_kept_I", "lpd_batches", "lpd_entered",
                "lpd_entered_F", "lpd_entered_I", "lpd_incumbents", "lpd_incumbents_F", "lpd_incumbents_I",
                "lpd_posts", "lpd_posts_F", "lpd_posts_I", "lpd_lost", "lpd_failed")
# the counters every `gen` event carries in lp_decode mode (cumulative)
LPD_GEN_FIELDS = ("lpd_offers", "lpd_solved", "lpd_kept", "lpd_entered", "lpd_entered_F", "lpd_entered_I",
                  "lpd_incumbents", "lpd_incumbents_F", "lpd_incumbents_I", "lpd_posts")


def _lp_decoder_for(eng: "Engine") -> lpdec.LPDecoder:
    """The LP decode worker of a run (`repair.continuous: lp_decode`): the instance file, the model digest, the
    configured limits, the deadline t0 + budget."""
    from itrea.paths import resolve
    rp, prob = eng.cfg.repair, eng.prob
    return lpdec.LPDecoder(resolve(str(prob.path)), lpdec.model_digest(prob.names, prob.isint), eng.lpd_int_dtype,
                           rp.lp_decode_per_gen, rp.lp_time_limit, rp.lp_decode_cold_limit, eng.cfg.ranks.feas_tol,
                           eng.deadline)


def _is_solver_find(bits: int) -> bool:
    """An individual that came from the solver and was not changed by the EA: ORIGIN_INJECTED and no other origin bit
    (the rule behind an incumbent's source "solver")."""
    return bool((bits & ORIGIN_INJECTED) and not (bits & int(_GENE_BITS_EA)))


def _mem_available_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 1048576, 3)
    except OSError:
        pass
    return None


def _is_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or isinstance(e, MemoryError) and "Out of memory" in str(e)


# ------------------------------------------------------------------------------------------------------ engine
class Engine:
    def __init__(self, prob: Problem, cfg: EAConfig, seed: int, mailbox=None, trace=None, *,
                 backend: Backend | None = None, allow_cpu_fallback: bool = False, warmup: bool = True,
                 decoder_factory=None, lpd_warm_rows: int = 0):
        """mailbox: a file `Mailbox` or `RedisMailbox` (or None: no exchange, no stop file). backend: an existing
        backend to use (the warm-up passes its own); otherwise `prepare()` creates it from `cfg.backend`.
        decoder_factory: `factory(engine) -> decoder` for `repair.continuous: lp_decode` (tests and the warm-up's
        in-process stub; None = the `LPDecoder` worker process). lpd_warm_rows: warm-up only — run the LP decode path
        whatever the problem's columns, offering exactly this many rows per batch, with the no-gain filter off."""
        self.prob, self.cfg, self.seed = prob, cfg.validate(), int(seed)
        self.mailbox, self.trace = mailbox, (trace if trace is not None else _NullTrace())
        self.bk = backend
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self.warmup_enabled = bool(warmup)
        self.prepared = False
        self.startup: dict = {}
        self.stop_reason: str | None = None
        self.part = None                         # crash-safe incumbent log (incumbent_log)
        self.part_paths: tuple[Path, Path] | None = None
        self.rclock: RepairClock | None = None   # the repair clock (repair_clock.enabled; made in begin)
        self.rc_dec = None                       # its decision for the current generation
        self.rc_site = ""                        # the repair call site, set before every `_repair`
        self._rc_last: dict | None = None        # the last local-search call, waiting for its evaluated rows
        self._rc_cf = True                       # False for the repair calls made after the survival
        self.decoder_factory = decoder_factory
        self.lpd_warm_rows = int(lpd_warm_rows)
        self.lpd = None                          # the LP decode worker (repair.continuous: lp_decode; made in begin)
        self.lpd_on = False

    def incumbent_log(self, run_dir: str | os.PathLike, role: str = "ea") -> None:
        """Append every incumbent to `<role>.incumbents.part` as it is found (with `<role>.names.json`), in the solver
        processes' format (`solvers.common.SolverProcess.record_incumbent`), so a killed EA's incumbents can still be
        recovered and checked (`solvers.common.recover_incumbents`; D-002 §6: a crashed run is kept, truncated at the
        crash). `write_outputs` removes both once the .npz exists."""
        d = Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)
        names_p, part_p = d / f"{role}.names.json", d / f"{role}.incumbents.part"
        names_p.write_text(json.dumps([str(x) for x in self.prob.names]))
        self.part = open(part_p, "ab", buffering=0)
        self.part_paths = (part_p, names_p)

    # ===================================================================================================== prepare
    def prepare(self) -> dict:
        """Everything before the shared clock (module docstring). Returns the start-up record (seconds per step)."""
        cfg, prob = self.cfg, self.prob
        rec: dict = {}
        tt = time.monotonic()
        # -------------------------------------------------------------------------------- backend
        t = time.monotonic()
        mem0 = _mem_available_gb()
        if self.bk is None:
            if cfg.backend == "auto" and not self.allow_cpu_fallback:
                raise ValueError("backend 'auto' is refused for an EA run (a silent CPU fallback would mislabel a GPU "
                                 "arm); choose 'cupy' or 'numpy'")
            self.bk = get_backend(cfg.backend, gpu_mem_gb=cfg.gpu_mem_gb)
        bk = self.bk
        rec["backend"] = bk.name
        if bk.is_gpu:
            import cupy as cp
            pool = cp.get_default_memory_pool()
            want = int(cfg.gpu_mem_gb * 1024 ** 3)
            cp.zeros(1)                                          # create the context now (start-up, not the clock)
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            rec["device"] = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
            rec["pool_limit_bytes"] = int(pool.get_limit())
            env = os.environ.get("CUPY_GPU_MEMORY_LIMIT")
            rec["cupy_gpu_memory_limit_env"] = env
            if env is not None and env.isdigit() and int(env) != want:
                rec["pool_limit_env_mismatch"] = f"CUPY_GPU_MEMORY_LIMIT={env} but gpu_mem_gb gives {want}"
            if int(pool.get_limit()) != want:
                raise RuntimeError(f"CuPy pool limit {pool.get_limit()} != configured {want} bytes")
            # CUB-backed reductions / scans are specialised on the array size, so a warm-up on a small synthetic
            # problem cannot compile them for the instance: the EA process disables them (`ea.run` sets
            # CUPY_ACCELERATORS="" before CuPy is imported). Recorded so a run that inherited CUB is visible.
            from cupy import _core as _cpcore
            rec["cupy_accelerators"] = dict(env=os.environ.get("CUPY_ACCELERATORS"),
                                            routine=[int(a) for a in _cpcore.get_routine_accelerators()],
                                            reduction=[int(a) for a in _cpcore.get_reduction_accelerators()])
        else:
            rec["device"] = "cpu"
            rec["pool_limit_bytes"] = None
        mem1 = _mem_available_gb()
        rec["mem_available_gb_before_backend"] = mem0
        rec["mem_available_gb_after_backend"] = mem1
        rec["backend_s"] = time.monotonic() - t
        # -------------------------------------------------------------------------------- genes, sizes
        t = time.monotonic()
        gdt = gene_dtype_for(np.asarray(prob.lb), np.asarray(prob.ub), np.asarray(prob.isint, bool),
                             cfg.population.gene_dtype)
        if gdt.kind in "iu":
            if prob.conts.size:
                raise ValueError(f"population.gene_dtype={gdt.name} cannot hold the {prob.conts.size} continuous "
                                 "variables of this instance")
            info = np.iinfo(gdt)
            if prob.n and not (np.all(np.isfinite(prob.lb)) and np.all(np.isfinite(prob.ub))
                               and prob.lb.min() >= info.min and prob.ub.max() <= info.max):
                raise ValueError(f"population.gene_dtype={gdt.name} cannot hold this instance's integer bounds")
        self.gene_dtype = np.dtype(gdt)
        plan = plan_memory(prob, cfg, self.gene_dtype)
        if not plan.fits:
            raise DoesNotFit(f"population does not fit gpu_mem_gb={cfg.gpu_mem_gb}: {plan.as_dict()}")
        if cfg.ranks.x > 0 and math.floor(cfg.ranks.x * plan.P + 1e-12) < 1:
            raise DoesNotFit(f"P={plan.P} leaves no Rank 0 slot for x={cfg.ranks.x}")
        self.plan, self.P, self.O = plan, plan.P, plan.offspring
        self.budgets = plan.budgets
        rec.update(P=self.P, offspring=self.O, gene_dtype=self.gene_dtype.name, memory_plan=plan.as_dict())
        rec["plan_s"] = time.monotonic() - t
        # -------------------------------------------------------------------------------- device copy
        t = time.monotonic()
        self.dp: DeviceProblem = to_device(prob, bk, cfg.population.eval_dtype, cfg.ranks.tol_mode,
                                           cfg.ranks.viol_norm)
        # ranks.feas_alt_scale: the same problem with the second row scale, for the feasibility phase's 2-D survival
        self.dp_alt: DeviceProblem | None = None
        if cfg.ranks.feas_alt_scale != "none":
            self.dp_alt = to_device(prob, bk, cfg.population.eval_dtype, cfg.ranks.tol_mode, cfg.ranks.feas_alt_scale)
        rec["device_copy_s"] = time.monotonic() - t
        # -------------------------------------------------------------------------------- operators
        t = time.monotonic()
        rp = cfg.repair
        self._lpd_prepare(rec)
        if cfg.seeding.lp_source == "highs_async":
            from itrea.paths import resolve
            if not resolve(str(prob.path)).is_file():
                raise ValueError("seeding.lp_source='highs_async' needs the instance file (the LP worker reads it)")
        self._build_local_searches()
        self.decoder = None
        if rp.continuous in ("auto", "lp_decode") and prob.conts.size:   # lp_decode = auto + the LP decode
            dec = SingletonDecoder(self.dp, self.budgets["operators"])
            self.decoder = dec if len(dec) else None
        self.dup = DupHasher(bk, prob.n, salt=12345)
        self.frontier = Frontier(bk, self.dp, cfg.frontier) if cfg.frontier.enabled else None
        n_int = int(self.dp.ints.shape[0])
        self.r1_k = None                                         # D-016: Rank 1 by clearing (None: "top")
        self.rk_eff = cfg.ranks                                  # the rank settings survival uses
        mode = cfg.ranks.rank1_mode
        if mode == "clearing" and n_int > 0:
            self.r1_k = cfg.ranks.rank1_min_diff or min_diff_auto(n_int)
            if self.r1_k > n_int:
                raise ValueError(f"ranks.rank1_min_diff = {self.r1_k} exceeds the {n_int} integer variables: Rank 1 "
                                 "could never hold more than one member")
        elif mode == "clearing":                                 # no integer variable: no distance exists -> "top"
            self.rk_eff = dataclasses.replace(cfg.ranks, rank1_mode="top")
            mode = "top (clearing requested, no integer variables)"
        self.r1_cleared_last = 0
        rec["rank1"] = dict(mode=mode, min_diff=self.r1_k, n_int=n_int,
                            size=rank1_cap(cfg.ranks, self.P) if self.r1_k is not None else None)
        rec["operators_s"] = time.monotonic() - t
        rec["local_search"] = self.ls.impl_name
        rec["decoded_columns"] = len(self.decoder) if self.decoder is not None else 0
        if self.strat_on:
            rec["strategies"] = [dict(index=i, name=nm, row_scale=sc, lp_guidance=lp)
                                 for i, ((sc, lp), nm) in enumerate(zip(self.strat_list, self.strat_names))]
            rec["local_search_row_scales"] = list(self.ls.row_scales)
            rec["local_search_static_bytes"] = getattr(self.ls.impl, "static_bytes", None)
        if cfg.repair_clock.enabled:
            rec["repair_clock"] = RepairClock(cfg.repair_clock).describe()
        # -------------------------------------------------------------------------------- warm-up
        t = time.monotonic()
        if self.warmup_enabled:
            rec["warmup"] = self._warmup()
        rec["warmup_s"] = time.monotonic() - t
        rec["prepare_s"] = time.monotonic() - tt
        self.startup = rec
        self.prepared = True
        return rec

    def _build_local_searches(self) -> None:
        """`self.ls` — with the strategy, one LocalSearch over every row scale of `strategy.row_scales` — and the
        strategy tables (module docstring)."""
        cfg, bk = self.cfg, self.bk
        rp, budget = cfg.repair, self.budgets["local_search"]
        sg = cfg.strategy
        self.strat_on = bool(sg.enabled)
        if not self.strat_on:
            self.ls = LocalSearch(bk, self.dp, rp, cfg.ranks.imp_tol, budget, self.seed)
            self.strat_list = [(cfg.ranks.viol_norm, True)]
            self.strat_names = ["default"]
            self.S = 1
            return
        self.strat_list = sg.strategies()
        self.strat_names = [f"{sc}/{'lp' if lp else 'nolp'}" for sc, lp in self.strat_list]
        self.S = len(self.strat_list)
        scales = None if rp.method == "none" else list(sg.row_scales)
        self.ls = LocalSearch(bk, self.dp, rp, cfg.ranks.imp_tol, budget, self.seed, row_scales=scales)
        if scales is not None and self.ls.row_scales != scales:
            raise RuntimeError(f"local search row scales {self.ls.row_scales} != strategy.row_scales {scales}")
        # strategy -> index into strategy.row_scales (= the local search's table); "none": every index 0
        self.strat_scale = np.array([sg.row_scales.index(sc) if scales is not None else 0 for sc, _ in self.strat_list],
                                    np.int64)
        self.strat_lp = np.array([lp for _, lp in self.strat_list], bool)

    def _warmup(self) -> dict:
        """Run every code path on `synthetic_problem` with the same backend, dtypes and operators, then discard it.

        CuPy specialises an elementwise kernel on dtypes AND on the number of dimensions left after broadcasting, so a
        batch of ONE row (one child, one injected row, one immigrant, a duplicate pair) compiles kernels a batch of
        several does not. Two passes therefore run: batches of many rows, and batches of one."""
        syn = synthetic_problem(self.prob, seed=self.seed + 7919)
        x = self.cfg.ranks.x
        Pw = max(16, math.ceil(1.0 / x) + 1 if x > 0 else 16)
        over = {"population.size": Pw, "population.size_rule": "fixed",
                "population.gene_dtype": self.gene_dtype.name, "stagnation.gens": 1, "frontier.update_every": 1,
                "frontier.reinject": 0.5, "frontier.enabled": True, "log_every": 1, "exchange.enabled": True,
                "exchange.from_solver": True, "exchange.to_solver": True, "exchange.use_dual_bound": True,
                "exchange.use_root_lp": True, "exchange.poll_s": 1e-6, "exchange.post_frontier": True,
                "seeding.lp_source": "solver", "seeding.random": 0.5, "seeding.lp_rounding": 0.5,
                "crossover.pc": 1.0,
                "crossover.weights": {"agreement": 1.0, "uniform": 1.0, "block": 1.0, "k_point": 1.0},
                "mutation.weights": {k: 1.0 for k in ("flip", "step", "reset", "toward_lp", "violation_guided",
                                                      "objective_guided")}}
        if self.cfg.repair.continuous == "lp_decode" and not self.lpd_on:
            # inert on this instance (no active column), so the run is exactly "auto": warm "auto" (the synthetic
            # problem may hold active columns the instance does not have, and has no file for a worker)
            over["repair.continuous"] = "auto"
        out = {}
        for name, offspring, rows in (("many", Pw, 3), ("one", 1, 1)):
            wcfg = self.cfg.with_overrides(dict(over, **{"population.offspring_per_gen": offspring}))
            # LP decode: the in-process stub, forced to offer batches of three rows ("many") and of one row ("one")
            lkw = dict(decoder_factory=_WarmupDecoder, lpd_warm_rows=3 if name == "many" else 1) if self.lpd_on else {}
            w = Engine(syn, wcfg, seed=self.seed + 104729 + offspring, mailbox=_WarmupMailbox(syn, rows),
                       backend=self.bk, warmup=False, **lkw)
            w.prepare()
            w.begin(time.monotonic(), 3600.0)
            for _ in range(3):
                w.step()
            w._warm_paths(one=(name == "one"))
            w.step()
            res = w.finish()
            out[name] = dict(generations=res["generations"], feasible_found=res["feasible_found"],
                             posts=res["posts"], immigrants=res["immigrants_admitted"],
                             violation_guided_forced=int(getattr(w, "vg_warm", 0)),
                             elite_warmed=int(w.c.get("warm_elite", 0)))
            if self.lpd_on:                # the largest decoded batch (rows) that entered a candidate set, and how many
                out[name]["lp_decode_warmed"] = int(getattr(w, "lpd_batch_rows_max", 0))
                out[name]["lp_decode_warmed_batches"] = int(w.c.get("lpd_batches", 0))
        if self.bk.is_gpu:
            out["scan_ladder_max"] = self._warm_scans()
        self.bk.synchronize()
        out.update(n=syn.n, m=syn.m)
        return out

    def _warm_scans(self) -> int:
        """Warm-up only (GPU): CuPy's own scan (CUB off) is still specialised by size class — it compiled new kernels
        at 2^10 and 2^19 elements (measured 2026-10-06) — and the synthetic problem is far smaller than the instance.
        So the two cumsum calls of the operators run here, with their exact dtypes, on synthetic arrays of every power of
        two from 32 up to the largest flat size the operators' budget allows (input + output inside the budget):
        crossover k_point `cumsum(int32 (rows, s), axis=1, dtype=int32)` with one and two rows, mutation
        violation_guided `cumsum(bool (s,), dtype=int64)`. Nothing depends on the instance. Returns the largest s."""
        xp = self.bk.xp
        top = max(32, int(self.budgets["operators"]) // 16)
        s, last = 32, 32
        while s <= top:
            for rows in (1, 2):
                xp.cumsum(xp.ones((rows, s), dtype=xp.int32), axis=1, dtype=xp.int32)
            xp.cumsum(xp.ones(s, dtype=bool), dtype=xp.int64)
            last = s
            s *= 2
        self.bk.synchronize()
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        return last

    def _warm_paths(self, one: bool) -> None:
        """Warm-up only: force the stagnation, shortfall and immigrant paths (with batches of one row if `one`), and
        violation_guided's draw over violated rows (`_warm_violation_guided`)."""
        bk = self.bk
        self.vg_warm = getattr(self, "vg_warm", 0) + self._warm_violation_guided()
        self.since_best = self.cfg.stagnation.gens
        self._stagnation()
        keep = max(1, self.pop.size - (1 if one else 2))
        if keep < self.pop.size:
            self.pop = self.pop.take(bk, np.arange(keep))
            self.h_key, self.h_viol, self.h_feas = self.h_key[:keep], self.h_viol[:keep], self.h_feas[:keep]
        self._after_survival()
        self._warm_elite()
        if not one:
            return
        fr = self.frontier
        for k in (1, 2):                                        # the archive's distance on one and two rows
            fr.distance_to(self.pop.X[:k], self.pop.X[0])
        fe = np.flatnonzero(self.h_feas)
        if fe.size:                                             # an update offered nothing but the incumbent
            i = int(fe[0])
            fr.update(self.pop.X[:0], np.empty(0), np.empty(0, bool), np.empty(0, np.int64), self.pop.X[i],
                      float(self.h_key[i]))
        kw = dict(gene_dtype=self.gene_dtype.name, mem_budget_bytes=self.budgets["operators"])
        for lp_share, fr in ((1.0, None), (0.0, None), (0.0, self.frontier.sample(1, self.rng_host))):
            Xn, on = immigrants(bk, self.dp, 1, self.rng_dev, self.rng_host, lp_x=self.lp_x, frontier_rows=fr,
                                lp_share=lp_share, **kw)
            r2 = np.flatnonzero(self.pop.rank == R2)
            self._admit(Xn, on, replace=r2[-1:] if r2.size else None)
        for copies in (1, 2):                                   # duplicate groups of two and of three
            Xn = bk.xp.concatenate([self.pop.X[:1]] * copies)
            self._admit(Xn, np.full(copies, ORIGIN_IMMIGRANT, np.uint16), replace=None)

    def _warm_elite(self) -> None:
        """Warm-up only: compile the D-016 distance kernels on this run's gene dtype whatever `ranks.rank1_mode` is
        (`elite_spread` runs at every gen record in both modes): the spread's (rows, n_int) comparison on two and three
        rows, and the clearing's (chunk, m, n_int) comparison with one-row and two-row chunks over two and three
        candidates. The rows are copies of the population's first vector — the warm-up population of a small
        synthetic problem can hold as few as two distinct members (2026-10-08: with "pop.size >= 3" as the condition
        the kernels were never warmed and compiled on the clock) — and need not be feasible or distinct."""
        bk = self.bk
        if self.pop.size < 1 or int(self.dp.ints.shape[0]) == 0:
            return
        X = bk.xp.concatenate([self.pop.X[:1]] * 3)
        ints, key = self.dp.ints, np.array([0.0, 1.0, 2.0])
        for r in (2, 3):
            elite_spread(bk, X, np.arange(r), key[:r], ints, band=self.cfg.frontier.band)
            for block, budget in ((1, 1), (2, 1), (r, 1 << 30)):
                clearing_pick(bk, X, np.arange(r), ints, 1 << 30, r, block=block, budget=budget)
        self.c["warm_elite"] = self.c.get("warm_elite", 0) + 1

    def _warm_violation_guided(self) -> int:
        """Warm-up only. violation_guided draws "the r-th violated row of a child" with cumsum / searchsorted / take
        kernels that run only when some child HAS a violated row — which the random warm-up children need not have:
        on 2026-10-06 one seed in eight left 14 kernels to NVRTC at the first generation, on the clock. So mutate,
        with weights {violation_guided: 1}, copies of a point built to violate a row: the synthetic problem's feasible
        `toy_point` with ONE variable moved (integers first; to a bound or ±1000 within the bounds) until a row is
        violated beyond the operator's tolerance — one child, then three. Returns the number of children mutated (0
        when the operator cannot apply: no row, no integer variable, or no single move violates a row)."""
        prob = self.prob
        x0 = getattr(prob, "toy_point", None)
        if x0 is None or prob.m == 0 or prob.nnz == 0:
            return 0
        isint = np.asarray(prob.isint, bool)
        lb, ub = np.asarray(prob.lb, float), np.asarray(prob.ub, float)
        lo = np.where(isint, np.ceil(lb - 1e-9), lb)
        hi = np.where(isint, np.floor(ub + 1e-9), ub)
        if not np.any(isint & (hi > lo)):
            return 0
        A = sps.csc_matrix(prob.A)
        rl, ru = np.asarray(prob.rl, float), np.asarray(prob.ru, float)
        tl = VG_VIOL_TOL * np.where(np.isfinite(rl), np.maximum(1.0, np.abs(rl)), 1.0)
        tu = VG_VIOL_TOL * np.where(np.isfinite(ru), np.maximum(1.0, np.abs(ru)), 1.0)
        x0 = np.asarray(x0, float)
        act0 = A @ x0
        xv = None
        for j in np.r_[np.flatnonzero(isint), np.flatnonzero(~isint)]:
            col = A[:, [j]].toarray().ravel()
            for v in (lo[j], hi[j], x0[j] - 1000.0, x0[j] + 1000.0):
                v = float(np.clip(v, lo[j], hi[j]))
                if not np.isfinite(v) or v == x0[j]:
                    continue
                a = act0 + col * (v - x0[j])
                if np.any((rl - a > tl) | (a - ru > tu)):
                    xv = x0.copy()
                    xv[j] = v
                    break
            if xv is not None:
                break
        if xv is None:
            return 0
        mcfg = dataclasses.replace(self.cfg.mutation, weights={"violation_guided": 1.0})
        for k in (1, 3):
            C = self.bk.xp.asarray(np.repeat(xv[None, :], k, axis=0).astype(self.gene_dtype))
            mutate(self.bk, self.dp, C, mcfg, self.rng_dev, self.rng_host, lp_x=self.lp_x,
                   mem_budget_bytes=self.budgets["operators"])
        self.bk.synchronize()
        return 4

    # ========================================================================================================= run
    def run(self, budget: float, t0: float, max_gens: int | None = None) -> dict:
        """Seed, evaluate, survive, then generations until t0 + budget (monotonic), the stop file, a stop request or
        `max_gens`. Returns the summary (module docstring); `best_x` is the best EA incumbent (host float64)."""
        if not self.prepared:
            raise RuntimeError("prepare() first")
        self.begin(t0, budget)
        done = False
        try:
            while self._continue(max_gens):
                self.step()
            done = True
        finally:
            if self.lp_worker is not None:
                self.lp_worker.stop()
            if self.lpd is not None and not done:        # an exception: stop the LP decode worker here; otherwise
                self._lpd_stop()                         # finish() reads its last waiting reply first, then stops it
        return self.finish()

    def request_stop(self, reason: str = "sigterm") -> None:
        if self.stop_reason is None:
            self.stop_reason = reason

    def _continue(self, max_gens: int | None) -> bool:
        if self.stop_reason is not None:
            return False
        if time.monotonic() >= self.deadline:
            self.stop_reason = "budget"
            return False
        if max_gens is not None and self.gen >= max_gens:
            self.stop_reason = "max_gens"
            return False
        return True

    # ------------------------------------------------------------------------------------------------- begin
    def begin(self, t0: float, budget: float = math.inf) -> None:
        """Everything that happens at t0: state, the LP worker, the initial population and its first survival."""
        cfg, bk, prob = self.cfg, self.bk, self.prob
        self.t0 = float(t0)
        self.deadline = self.t0 + float(budget)
        if getattr(self.trace, "t0", None) is None:
            self.trace.start(self.t0)
        self.rng_host = np.random.default_rng([self.seed, 0x17EA])
        self.rng_dev = bk.rng(self.seed)
        self.idc = IdCounter()
        self.phase = _Phases(bk)
        self.gen = 0
        self.depth = int(cfg.lineage_depth)
        ex = cfg.exchange
        mb = self.mailbox
        self.from_solver = bool(mb is not None and ex.enabled and ex.from_solver)
        self.to_solver = bool(mb is not None and ex.enabled and ex.to_solver)
        self.from_cuopt = bool(mb is not None and ex.enabled and ex.from_cuopt)     # arm A9 (module docstring)
        self.inc_solver = math.inf               # solver incumbent (key) known to the EA: posts must beat it
        self.inc_cuopt = math.inf                # cuOpt's incumbent (key); stays +inf unless from_cuopt
        self.inc_external = math.inf             # min(inc_solver, inc_cuopt): the external incumbent
        self.dual_key = -math.inf                # solver's proven bound (key)
        self.lp_x: np.ndarray | None = None
        self.lp_info: dict = dict(source=cfg.seeding.lp_source, state="none")
        self.lp_worker = None
        self.pending: list[_Batch] = []
        self.best_key = math.inf                 # best EA incumbent (float64-checked, rounded)
        self.best_obj: float | None = None
        self.best_x: np.ndarray | None = None
        self.best_viol = math.inf                # feasibility phase
        # every EA incumbent as (id, key, objective, x), newest last: a solver rejection pops the rejected ones, and the
        # best_* fields fall back to the newest one not rejected (module docstring, step 8)
        self.inc_stack: list[tuple[int, float, float, np.ndarray]] = []
        self.rejected_ids: set[int] = set()
        self.rejected_hashes: set[bytes] = set()  # solvers.common.vector_hash of every rejected posted row
        self.posted_hashes: dict[int, list[bytes]] = {}   # to_solver seq -> vector hash of each posted row
        self._next_stop_poll = -math.inf
        self.max_gen_s = 0.0
        self.since_best = 0
        self.cutoff: float | None = None
        self.r1_ids: set[int] = set()
        self.checked: dict[int, tuple[bool, float]] = {}
        self.records: list[tuple[float, float, str, _SparseRow]] = []
        self.posted: dict[int, list[int]] = {}   # to_solver seq -> ids of its rows
        self.n_verdicts = 0
        self.next_poll = self.t0
        self.c: dict[str, int] = {k: 0 for k in (
            "children", "evaluations", "posts", "posts_frontier", "false_feasible", "inject_msgs", "inject_rows",
            "echoes", "injected", "injected_infeasible", "injected_dup", "solver_rejections", "solver_accepted",
            "verdicts_seen", "oom_retries", "stagnation_events", "immigrants_admitted", "immigrants_dropped",
            "lp_immigrants", "ls_moves", "ls_calls", "duplicates", "n_incumbents", "demoted", "incumbents_reverted",
            "rejected_copies", "aborted_generations")}
        # D-011 LP stream: the solver's node LP points, ranked (V of the nearest rounding, key) — Berk's order
        self.lps = cfg.lp_stream if (self.from_solver and cfg.lp_stream.enabled) else None
        self.lps_pool: list[dict] = []
        self.lps_seen = 0                        # highest point id taken from the stream
        self.lps_rr = 0                          # round robin over the pool's best for stagnation immigrants
        if self.lps is not None:
            self.c.update({k: 0 for k in ("lps_msgs", "lps_points", "lps_used", "lps_rows", "lps_dropped",
                                          "lps_missed")})
        # arm A9: ids of the injected cuOpt rows, and of those posted to the solver unchanged (both empty while off)
        self.cuopt_ids: set[int] = set()
        self.cuopt_posted: set[int] = set()
        self.dual_key_cuopt = -math.inf          # cuOpt's bound (key): recorded, never used
        if self.from_cuopt:
            self.c.update({k: 0 for k in ("cuopt_inject_msgs", "cuopt_inject_rows", "cuopt_injected",
                                          "cuopt_injected_infeasible", "cuopt_injected_dup", "cuopt_info_msgs",
                                          "cuopt_incumbents", "cuopt_posts", "cuopt_rows_accepted",
                                          "cuopt_rows_rejected")})
        S = self.S
        self.sc: dict[str, np.ndarray] = {k: np.zeros(S, np.int64) for k in (
            "children", "best_events", "incumbents", "best_events_solver", "incumbents_solver", "repaired",
            "repaired_feasible", "immigrants")}
        self.sc_since: dict[str, np.ndarray] = {k: np.zeros(S, np.int64) for k in ("children", "best_events")}
        self.c_strat = {"strategy_redrawn": 0, "strategy_reassigned": 0}
        self.gen_rows: list[dict] = []
        self.g_or: dict[int, int] = {}
        self.g_new_best: set[int] = set()
        self.g_accepted: set[int] = set()
        self.peak_used = 0
        self.peak_pool = 0
        self.last_log = (time.monotonic(), 0)
        self.last_surv: Survival | None = None
        self.logged_gen = -1
        self.frontier_posted: frozenset = frozenset()
        self.wall_start = time.monotonic()
        if cfg.seeding.lp_source == "highs_async":
            self.lp_worker = LPWorker(prob.path, cfg.seeding.lp_time_limit, cpus=None, names=prob.names).start()
            self.lp_info["state"] = "running"
        self._lpd_begin()                        # repair.continuous: lp_decode, an instance it can act on
        # repair clock (module docstring): generation 0 is before any feasible member, so it plays "none"
        self.rclock = RepairClock(cfg.repair_clock) if cfg.repair_clock.enabled else None
        self.rc_dec, self._rc_last = None, None
        if self.rclock is not None:
            self.c.update(ls_finished_horizon=0, ls_cap_hits=0)
            self._rc_t_gen = time.monotonic()
            self._rc_decide(False)
        # ------------------------------------------------------------------------- initial population
        with self.phase("seed"):
            X, origin = seed_population(bk, self.dp, self.P, cfg.seeding, self.rng_dev, self.rng_host,
                                        gene_dtype=self.gene_dtype.name, mem_budget_bytes=self.budgets["operators"])
        k = X.shape[0]
        strat0 = (np.arange(k) % self.S).astype(np.uint8)       # round robin (all 0 while the strategy is off)
        self.rc_site = "init"
        with self.phase("repair"):
            moved = self._repair(X, strat0)
        if moved is not None:
            origin = origin | np.where(moved, ORIGIN_REPAIR, 0).astype(np.uint16)
        with self.phase("evaluate"):
            ev = self._evaluate(X)
        self._rc_pair(ev[0], ev[2])
        now = self.trace.now()
        pop0 = self._make_pop(X, ev, np.full(k, NO_RANK, np.int8), self.idc.take(k),
                              np.full((k, 2), -1, np.int64), origin, 0, strat=strat0)
        meta = dict(rank_pa=np.full(k, -1, np.int8), rank_pb=np.full(k, -1, np.int8), t_first=np.full(k, now))
        self._exchange_in(force=True)
        cand, meta = self._with_pending(pop0, meta)
        with self.phase("survival"):
            self._survive(cand, 0, meta)
        self._after_survival()
        self._rc_end()
        self._gen_record()

    # -------------------------------------------------------------------------------------------------- step
    def step(self) -> None:
        """One generation (module docstring, steps 1–10). Cut short after the repair when the budget ends or a stop
        arrives during it (the local search polls `_should_stop`): the generation's children are dropped, the
        population stays as it was, `gen` keeps counting the started generation and `aborted_generations` is 1."""
        tg = time.monotonic()
        try:
            self._step()
        finally:
            self.max_gen_s = max(self.max_gen_s, time.monotonic() - tg)

    def _step(self) -> None:
        cfg, bk, dp, pop = self.cfg, self.bk, self.dp, self.pop
        self.gen += 1
        if self.rclock is not None:
            self._rc_t_gen = time.monotonic()
        if time.monotonic() >= self.next_poll:
            with self.phase("exchange"):
                self._exchange_in()
        if self.rclock is not None:                          # the level of this generation, after the exchange tick
            self._rc_decide(self._rc_own_feasible())
        if self.pop.size == 0:                               # every candidate died (duplicates / non-finite)
            Xn, on, sn = self._immigrants(self.P, lp_share=cfg.seeding.lp_rounding)
            if self._admit(Xn, on, replace=None, strat=sn) == 0:
                raise RuntimeError("the population is empty and no immigrant could be admitted")
        pop = self.pop
        Np = pop.size
        # --------------------------------------------------------------------- breed
        with self.phase("select"):
            ia, ib = select_parents(pop.rank, self.h_key, self.h_viol, self.h_feas, self.O, cfg.selection,
                                    self.rng_host)
            cs = None
            if self.strat_on:                                # children's strategies (module docstring)
                cs = self._child_strategies(pop.strat[ia])
        with self.phase("crossover"):
            dia, dib = bk.xp.asarray(ia), bk.xp.asarray(ib)
            C, xo, touched = crossover(bk, dp, pop.X[dia], pop.X[dib], cfg.crossover, self.rng_dev, self.rng_host,
                                       self.budgets["operators"])
            del touched, dia, dib
        pb = np.where(xo != 0, ib, ia)                       # mutation-only children: pb = pa
        with self.phase("mutation"):
            # with the strategy: one call; children whose strategy has no LP guidance never draw toward_lp
            lp_rows = self.strat_lp[cs] if (cs is not None and self.lp_x is not None) else None
            mo = mutate(bk, dp, C, cfg.mutation, self.rng_dev, self.rng_host, lp_x=self.lp_x,
                        mem_budget_bytes=self.budgets["operators"], lp_rows=lp_rows)
        origin = (xo | mo).astype(np.uint16)
        self.rc_site = "children"
        with self.phase("repair"):
            moved = self._repair(C, cs)
        if not self._continue(None):                         # budget / stop during the repair: drop this generation
            self.c["aborted_generations"] += 1
            self.trace.event("note", msg=f"generation {self.gen} cut short during the repair ({self.stop_reason})",
                             gen=self.gen)
            return
        if moved is not None:
            origin |= np.where(moved, ORIGIN_REPAIR, 0).astype(np.uint16)
        with self.phase("evaluate"):
            ev = self._evaluate(C)
        self._rc_pair(ev[0], ev[2])
        k = C.shape[0]
        self.c["children"] += k
        ar, ao = child_ancestry(self.depth, pop.rank[ia], pop.rank[pb], pop.origin[ia], pop.origin[pb],
                                pop.anc_rank[ia], pop.anc_rank[pb], pop.anc_ops[ia], pop.anc_ops[pb],
                                pa_src=src_bits_of_origin(pop.origin[ia]), pb_src=src_bits_of_origin(pop.origin[pb]))
        children = self._make_pop(C, ev, np.full(k, NO_RANK, np.int8), self.idc.take(k),
                                  np.stack([pop.ids[ia], pop.ids[pb]], axis=1), origin, self.gen, ar, ao, strat=cs)
        if cs is not None:
            cnt = np.bincount(cs, minlength=self.S)
            self.sc["children"] += cnt
            self.sc_since["children"] += cnt
        now = self.trace.now()
        meta = dict(rank_pa=pop.rank[ia].astype(np.int8), rank_pb=pop.rank[pb].astype(np.int8),
                    t_first=np.full(k, now))
        cand = Population.concat(bk, [pop, children])
        del children, C
        if self.lpd is not None:                             # LP decode: a reply that came during this generation
            self._lpd_poll(site="candidates")                # joins this candidate set (module docstring)
        cand, meta = self._with_pending(cand, meta)
        with self.phase("survival"):
            self._survive(cand, Np, meta)
        if self.lpd is not None:                             # LP decode: poll, then one batch (module docstring)
            self._lpd_tick(cand)
        self._after_survival()
        self._rc_end()
        if self.gen % cfg.log_every == 0:
            self._gen_record()

    # ------------------------------------------------------------------------------------- pieces of a step
    def _make_pop(self, X, ev, rank, ids, parents, origin, birth, anc_rank=None, anc_ops=None,
                  strat=None) -> Population:
        k = X.shape[0]
        if anc_rank is None:
            anc_rank = np.zeros((k, self.depth), np.uint8)
            anc_ops = np.zeros((k, self.depth), np.uint16)
        return Population(X=X, key=ev[0], viol=ev[1], feasible=ev[2], rank=np.asarray(rank, np.int8),
                          ids=np.asarray(ids, np.int64), parents=np.asarray(parents, np.int64),
                          origin=np.asarray(origin, np.uint16), birth=np.full(k, birth, np.int32),
                          anc_rank=anc_rank, anc_ops=anc_ops,
                          strat=np.zeros(k, np.uint8) if strat is None else np.asarray(strat, np.uint8))

    # ------------------------------------------------------------------------------------------- strategies
    def _child_strategies(self, parent: np.ndarray) -> np.ndarray:
        """Strategies of the O children from their parent A's (module docstring): inherited, redrawn uniformly with
        probability `strategy.mutation_rate`, then every strategy topped up to min(ceil(min_share·O), O // S)."""
        sg, S, rng = self.cfg.strategy, self.S, self.rng_host
        cs = np.asarray(parent, np.uint8).copy()
        O = cs.size
        redraw = rng.random(O) < sg.mutation_rate
        nr = int(redraw.sum())
        if nr:
            cs[redraw] = rng.integers(S, size=nr).astype(np.uint8)
        self.c_strat["strategy_redrawn"] += nr
        q = min(math.ceil(sg.min_share * O - 1e-9), O // S)
        if q > 0:
            cnt = np.bincount(cs, minlength=S)
            need = np.maximum(q - cnt, 0)
            if need.any():
                # donors: a random cnt - q of the children of every strategy above the floor, so none drops below it
                pool = np.concatenate([rng.permutation(np.flatnonzero(cs == t))[:cnt[t] - q]
                                       for t in np.flatnonzero(cnt > q)])
                take = rng.choice(pool, size=int(need.sum()), replace=False)
                cs[take] = np.repeat(np.arange(S), need).astype(np.uint8)
                self.c_strat["strategy_reassigned"] += int(need.sum())
        return cs

    def _random_strategies(self, k: int, lp_only: bool = False) -> np.ndarray:
        """k independent uniform strategy draws, in row order (never sorted: injected rows arrive best first, and a
        sort would hand the best row the smallest of k draws); `lp_only`: among the strategies with LP guidance. Zeros
        (no draw) while the strategy is off."""
        if not self.strat_on or k <= 0:
            return np.zeros(max(k, 0), np.uint8)
        pool = np.flatnonzero(self.strat_lp) if lp_only else np.arange(self.S)
        return pool[self.rng_host.integers(pool.size, size=k)].astype(np.uint8)

    def _immigrants(self, k: int, frontier_rows=None, lp_share: float | None = None):
        """`seeding.immigrants` with strategies: (X, origin, strat). Off: exactly the plain call, strat = 0. On: the k
        strategies are drawn uniformly first; LP roundings (lp_share of them) go only to rows whose strategy has LP
        guidance; frontier rows keep their draws."""
        cfg = self.cfg
        lp_share = cfg.seeding.lp_rounding if lp_share is None else lp_share
        kw = dict(gene_dtype=self.gene_dtype.name, mem_budget_bytes=self.budgets["operators"])
        lpx = self._lp_for_immigrants()
        if not self.strat_on:
            Xn, on = immigrants(self.bk, self.dp, k, self.rng_dev, self.rng_host, lp_x=lpx,
                                frontier_rows=frontier_rows, lp_share=lp_share, **kw)
            return Xn, on, np.zeros(Xn.shape[0], np.uint8)
        k = int(k)
        f = 0 if frontier_rows is None else min(k, int(frontier_rows.shape[0]))
        draws = self.rng_host.integers(self.S, size=k).astype(np.uint8)
        rest = draws[f:]
        on_lp = np.flatnonzero(self.strat_lp[rest]) if lpx is not None else np.zeros(0, np.int64)
        k_lp = int(round(lp_share * on_lp.size))
        Xn, on = immigrants(self.bk, self.dp, k, self.rng_dev, self.rng_host,
                            lp_x=lpx if k_lp else None, frontier_rows=frontier_rows, lp_share=0.0,
                            lp_count=k_lp, **kw)
        lp_rows = on_lp[:k_lp]                               # rest-positions that become LP roundings
        other = np.setdiff1d(np.arange(rest.size), lp_rows)
        strat = np.concatenate([draws[:f], rest[lp_rows], rest[other]]).astype(np.uint8)
        if strat.size != Xn.shape[0]:
            raise RuntimeError(f"immigrants returned {Xn.shape[0]} rows for {strat.size} strategies")
        self.sc["immigrants"] += np.bincount(strat, minlength=self.S)
        return Xn, on, strat

    def _evaluate(self, X):
        """(key, viol, feasible) device arrays; on a CuPy out-of-memory the budget is halved and the call retried."""
        ev = self._evaluate_on(self.dp, X)
        self.c["evaluations"] += int(X.shape[0])
        return ev.key, ev.viol, ev.feasible

    def _evaluate_on(self, dp, X):
        """`evaluate` on `dp` (the main device problem, or `dp_alt` for ranks.feas_alt_scale) with the OOM retry."""
        rk = self.cfg.ranks
        for attempt in range(OOM_RETRIES + 1):
            try:
                return evaluate(dp, X, rk.feas_tol, rk.int_tol, mem_budget_bytes=self.budgets["evaluate"])
            except Exception as e:  # noqa: BLE001
                if not _is_oom(e) or attempt == OOM_RETRIES:
                    raise
                self._oom("evaluate", e)

    def _oom(self, what: str, e: BaseException) -> None:
        self.c["oom_retries"] += 1
        if self.bk.is_gpu:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        key = {"evaluate": "evaluate", "local_search": "local_search"}[what]
        self.budgets[key] = max(1 << 20, self.budgets[key] // 2)
        if what == "local_search":
            self.ls.impl.mem_budget_bytes = self.budgets[key]
        self.trace.event("note", msg=f"out of memory in {what}; budget halved to {self.budgets[key]} bytes",
                         error=str(e)[:300])

    def _repair(self, X, strat: np.ndarray | None = None) -> np.ndarray | None:
        """Decoder -> local search -> decoder on X (in place). Returns the per-row 'moved by the search' flags.
        With the strategy on, `strat` gives each row's strategy and the ONE local-search call gets each row's row-scale
        index (module docstring); the per-strategy `repaired` / `repaired_feasible` counts come from that call."""
        k = X.shape[0]
        if k == 0:
            return None
        if self.decoder is not None:
            self.decoder.apply(X)
        if not self.strat_on or strat is None:
            moved = self._ls_run(X)
        else:
            strat = np.asarray(strat, np.int64)
            moved = self._ls_run(X, self.strat_scale[strat])
            self.sc["repaired"] += np.bincount(strat, minlength=self.S)
            stv = getattr(self.ls.impl, "last_status", None)
            if stv is not None and len(stv) == k:
                self.sc["repaired_feasible"] += np.bincount(strat, weights=(np.asarray(stv) == 1),
                                                            minlength=self.S).astype(np.int64)
        if self.decoder is not None:
            self.decoder.apply(X)
        return moved

    def _ls_run(self, X, scale_idx: np.ndarray | None = None) -> np.ndarray | None:
        """One local-search call with the out-of-memory retries; counters into `self.c`; the moved flags. With the
        repair clock the call also gets the generation's horizon / cap (active mode, after the first feasible member
        only — otherwise no argument is added, so the call is the one without the clock) and is kept, with its per-lane
        steps, status and launch profile, until `_rc_pair` pairs it with the evaluated rows."""
        ls = self.ls
        args = {} if scale_idx is None else {"scale_idx": scale_idx}
        if self.rclock is not None:
            args.update(self._rc_call_args())
        for attempt in range(OOM_RETRIES + 1):
            try:
                st = ls.run(X, should_stop=self._should_stop, **args)
                break
            except Exception as e:  # noqa: BLE001
                if not _is_oom(e) or attempt == OOM_RETRIES:
                    raise
                self._oom("local_search", e)
        self.c["ls_moves"] += int(st.get("moves", 0))
        self.c["ls_calls"] += 1
        for k in ("local_minima", "finished_feasible", "finished_stall", "finished_budget", "interrupted"):
            self.c[f"ls_{k}"] = self.c.get(f"ls_{k}", 0) + int(st.get(k, 0))
        if self.rclock is not None:
            self.c["ls_finished_horizon"] += int(st.get("finished_horizon", 0))
            self.c["ls_cap_hits"] += int(st.get("cap_hits", 0))
            impl = ls.impl
            self._rc_last = dict(steps=np.asarray(impl.last_steps).copy(), status=np.asarray(impl.last_status).copy(),
                                 profile=impl.last_profile, site=self.rc_site, pinned=self._rc_pinned(),
                                 horizon=args.get("horizon"), cap_s=args.get("cap_s"),
                                 cap_hits=int(st.get("cap_hits", 0)), counterfactual=self._rc_cf)
        lm = getattr(ls.impl, "last_moves", None)
        if lm is not None and len(lm) == X.shape[0]:
            return np.asarray(lm) > 0
        return None

    # ---------------------------------------------------------------------------------------------- repair clock
    def _rc_decide(self, has_feasible: bool) -> None:
        """The repair clock's level for the generation that starts now (after the exchange tick); an unpaired call
        left by a generation cut short is dropped."""
        self._rc_last, self._rc_cf = None, True
        self.rc_dec = self.rclock.decide(time.monotonic() - self.t0, has_feasible, gen=self.gen)

    def _rc_own_feasible(self) -> bool:
        """The repair clock's phase test: a feasible member that is not an unchanged solver solution (origin INJECTED
        and no other bit, `_is_solver_find`), so an injected solver row alone does not end the pre-feasible phase."""
        o = self.pop.origin
        solver = ((o & np.uint16(ORIGIN_INJECTED)) != 0) & ((o & _GENE_BITS_EA) == 0)
        return bool((self.h_feas & ~solver).any())

    def _rc_pinned(self) -> bool:
        """The one-shot LP batch is repaired at "none", uncapped (`repair_clock.pin_lp`); LP-stream batches are not."""
        return self.rc_site == "lp" and self.cfg.repair_clock.pin_lp

    def _rc_call_args(self) -> dict:
        """horizon / cap_s for the next local-search call: only in active mode, after the first feasible member, and
        never for the pinned LP batch. Empty otherwise (shadow mode plays "none" with no cap)."""
        d = self.rc_dec
        if d is None or not self.rclock.active or d.phase != "post" or self._rc_pinned():
            return {}
        out = {}
        if d.horizon is not None:
            out["horizon"] = int(d.horizon)
        if d.cap_s is not None:
            out["cap_s"] = float(d.cap_s)
        return out

    def _rc_pair(self, key, feasible, rows: np.ndarray | None = None) -> None:
        """THE pairing of a repair call's lanes with their evaluated rows, at every call site: lane i of the last
        local-search call is row i of the evaluation (`rows` = None: children, the initial population, a pending
        batch) or row rows[i] (`_admit`, which repairs only its seeded rows but evaluates all). `key` / `feasible` are
        the evaluation's arrays (device or host). No-op without the clock or without an unpaired call."""
        rec, self._rc_last = self._rc_last, None
        if rec is None or self.rclock is None:
            return
        key = np.asarray(self.bk.asnumpy(key), np.float64).reshape(-1)
        feasible = np.asarray(self.bk.asnumpy(feasible)).reshape(-1).astype(bool)
        if rows is not None:
            rows = np.asarray(rows, np.int64)
            key, feasible = key[rows], feasible[rows]
        if key.shape[0] != rec["steps"].shape[0]:
            raise RuntimeError(f"repair clock: {rec['steps'].shape[0]} lanes in the {rec['site']} call but "
                               f"{key.shape[0]} evaluated rows paired with them")
        self.rclock.add_call(keys=key, feasible=feasible, cutoff=self.cutoff, dual_key=self.dual_key, **rec)

    def _rc_end(self) -> None:
        """Close the generation in the repair clock and write its `rclock` event."""
        if self.rclock is None:
            return
        now = time.monotonic()
        ev = self.rclock.end_generation(now - self.t0, now - self._rc_t_gen)
        if ev is not None:
            self.trace.event("rclock", **ev)

    def _should_stop(self) -> bool:
        """Polled by the local search between kernel launches / row slices / steps, so the EA's work inside one
        generation is bounded by the budget and by the stop file (the runner's grace and kill windows are shorter than
        one generation on large instances). Sets `stop_reason` like `_continue` / `_exchange_in` would. The mailbox is
        read at most every STOP_POLL_S (a Redis round trip)."""
        if self.stop_reason is not None:
            return True
        now = time.monotonic()
        if now >= self.deadline:
            self.stop_reason = "budget"
            return True
        mb = self.mailbox
        if mb is not None and now >= self._next_stop_poll:
            self._next_stop_poll = now + STOP_POLL_S
            if mb.stopped():
                self.stop_reason = "stop"
                return True
        return False

    def _with_pending(self, cand: Population, meta: dict):
        """Append the pending batches (injected solutions, LP roundings) to the candidate set."""
        if not self.pending:
            return cand, meta
        parts, metas = [cand], [meta]
        now = self.trace.now()
        for b in self.pending:
            X = b.X
            if b.repair:
                self.rc_site = b.src
                with self.phase("repair"):
                    moved = self._repair(X, b.strat)
                if moved is not None:
                    b.origin = b.origin | np.where(moved, ORIGIN_REPAIR, 0).astype(np.uint16)
            with self.phase("evaluate"):
                ev = self._evaluate(X)
            if b.repair:
                self._rc_pair(ev[0], ev[2])
            k = X.shape[0]
            if b.src in ("solver", "cuopt"):
                feas = self.bk.asnumpy(ev[2])
                bad = int((~feas).sum())
                self.c["injected_infeasible" if b.src == "solver" else "cuopt_injected_infeasible"] += bad
                if bad:
                    who = "solver" if b.src == "solver" else "cuOpt"
                    self.trace.event("note", msg=f"{bad} of {k} injected {who} solutions are infeasible under the "
                                                 f"EA's test (tol_mode={self.cfg.ranks.tol_mode})")
            ids = self.idc.take(k)
            if b.src == "cuopt":                             # arm A9: the ids that carry cuOpt's vectors
                self.cuopt_ids.update(int(i) for i in ids)
            if b.src_ids is not None:                        # LP decode copies: id -> its source's id and kind
                self._lpd_entering(ids, b.src_ids, b.src_kinds)
            parts.append(self._make_pop(X, ev, np.full(k, NO_RANK, np.int8), ids,
                                        np.full((k, 2), -1, np.int64), b.origin, self.gen, b.anc_rank, b.anc_ops,
                                        strat=b.strat))
            metas.append(dict(rank_pa=np.full(k, -1, np.int8), rank_pb=np.full(k, -1, np.int8),
                              t_first=np.full(k, b.t)))
        self.pending = []
        cand = Population.concat(self.bk, parts)
        meta = {key: np.concatenate([m[key] for m in metas]) for key in meta}
        return cand, meta

    def _survive(self, cand: Population, Np: int, meta: dict) -> None:
        """Duplicates (provenance merged), three-rank survival with the float64 re-check of new Rank 1 entrants, the
        new population, genealogy rows for the entrants, the frontier update (while the candidates exist)."""
        cfg, bk = self.cfg, self.bk
        hk, hv, hf = cand.host_scores(bk)
        hf = hf.copy()
        hv = hv.copy()
        N = cand.size
        is_new = np.zeros(N, bool)
        is_new[Np:] = True
        dup, keep_of = self.dup(cand.X)
        if dup.any():
            self.c["duplicates"] += int(dup.sum())
            for b in np.flatnonzero(dup):
                kk = int(keep_of[b])
                bits = int(cand.origin[b])
                if bits & ~int(cand.origin[kk]):
                    cand.origin[kk] |= np.uint16(bits)
                    if kk < Np:                                  # an existing member: its genealogy row exists
                        self.g_or[int(cand.ids[kk])] = self.g_or.get(int(cand.ids[kk]), 0) | bits
                if (bits & ORIGIN_INJECTED) and kk < Np:
                    if self.from_cuopt and int(cand.ids[b]) in self.cuopt_ids:
                        self.c["cuopt_injected_dup"] += 1    # A9: SCIP's counter stays SCIP's
                    else:
                        self.c["injected_dup"] += 1
        # the previous Rank 1 (the ratchet of ranks.three_rank_survival): demotions below only hit NEW entrants (not in
        # r1_ids), and a member demoted by a solver verdict has left r1_ids already, so this mask stays valid
        prev_r1 = np.isin(cand.ids, np.fromiter(self.r1_ids, np.int64, len(self.r1_ids)))
        surv = None
        viol_alt = None
        if self.dp_alt is not None and not hf.any() and not math.isfinite(self.inc_external):
            with self.phase("evaluate"):                     # feasibility phase only: V on the second row scale
                viol_alt = bk.asnumpy(self._evaluate_on(self.dp_alt, cand.X).viol)
        pick = None
        pick_stats: dict = {}
        if self.r1_k is not None:                            # D-016: clearing over this candidate set's vectors
            def pick(F, cap, _X=cand.X):
                return clearing_pick(bk, _X, F, self.dp.ints, self.r1_k, cap, stats=pick_stats)
        # every failing round demotes at least one feasible candidate and the float64 checks are cached by id, so at
        # most #feasible + 1 rounds are needed; the last round always ends with a Rank 1 that passed the re-check
        for _ in range(int(hf.sum()) + 1):
            pick_stats.clear()                               # a round with no feasible candidate leaves it empty
            surv = three_rank_survival(hk, hv, hf, self.P, self.rk_eff, inc_external=self.inc_external,
                                       dual_bound=self.dual_key, rng=self.rng_host, dup=dup, ids=cand.ids,
                                       is_new=is_new, prev_r1=prev_r1, viol_alt=viol_alt, rank1_pick=pick)
            r1 = surv.idx[surv.rank == R1]
            failed = [int(i) for i in r1 if int(cand.ids[i]) not in self.r1_ids
                      and not self._check_candidate(cand, int(i))[0]]
            if not failed:
                break
            for i in failed:
                self._demote_host_dev(cand, hf, hv, i)
            self.c["false_feasible"] += len(failed)
            self.trace.event("note", msg=f"{len(failed)} new Rank 1 entrant(s) failed the float64 re-check; demoted",
                             ids=[int(cand.ids[i]) for i in failed])
        if np.any(~hf[surv.idx[surv.rank == R1]]):
            raise RuntimeError("survival left a demoted (infeasible) candidate in Rank 1")
        if self.r1_k is not None:                            # the distance rule's work in the final round
            self.r1_cleared_last = int(pick_stats.get("cleared", 0))
            self.c["rank1_cleared"] = self.c.get("rank1_cleared", 0) + self.r1_cleared_last
        idx = surv.idx
        self.pop = cand.take(bk, idx, rank=surv.rank)
        self.h_key, self.h_viol, self.h_feas = hk[idx], hv[idx], hf[idx]
        self.r1_ids = {int(i) for i in cand.ids[idx[surv.rank == R1]]}
        self.last_surv = surv
        ent = idx[is_new[idx]]
        if self.lpd is not None:                             # LP decode: the pick site reads this survival
            self._lpd_surv = (hk, hv, hf, dup, surv)
            if self._lpd_new:                                # decoded copies of this candidate set that survived
                for i in cand.ids[ent]:
                    kind = self._lpd_new.get(int(i))
                    if kind is not None:
                        self.c["lpd_entered"] += 1
                        self.c["lpd_entered_F" if kind == lpdec.KIND_F else "lpd_entered_I"] += 1
                self._lpd_new = {}
        if ent.size:
            j = ent - Np
            self.gen_rows.append(dict(
                id=cand.ids[ent].astype(np.int64), pa=cand.parents[ent, 0].astype(np.int64),
                pb=cand.parents[ent, 1].astype(np.int64), rank_pa=meta["rank_pa"][j].astype(np.int8),
                rank_pb=meta["rank_pb"][j].astype(np.int8), origin=cand.origin[ent].astype(np.uint16),
                birth=cand.birth[ent].astype(np.int32), key=hk[ent].astype(np.float64),
                V=hv[ent].astype(np.float64), feasible=hf[ent].astype(bool),
                t_first=meta["t_first"][j].astype(np.float64), strat=cand.strat[ent].astype(np.uint8)))
        if self.frontier is not None and self.gen % cfg.frontier.update_every == 0 and self.gen > 0:
            with self.phase("frontier"):
                self._frontier_update(cand, hk, hf)

    def _after_survival(self) -> None:
        """Shortfall fill, incumbent / posts, stagnation, cutoff, memory peaks."""
        if self.rclock is not None:
            self._rc_cf = False                              # these repair calls depend on the earlier ones
        surv = self.last_surv
        if surv is not None and self.pop.size < self.P:
            k = self.P - self.pop.size
            with self.phase("immigrants"):
                Xn, on, sn = self._immigrants(k, lp_share=self.cfg.seeding.lp_rounding)
                self._admit(Xn, on, replace=None, strat=sn)
        with self.phase("incumbent"):
            self._update_incumbent()
        self._stagnation()
        self._refresh_cutoff()
        if self.bk.is_gpu:
            self.peak_used = max(self.peak_used, self.bk.mem_used_bytes())
            self.peak_pool = max(self.peak_pool, self.bk.mem_total_bytes())

    def _check_candidate(self, popn: Population, i: int) -> tuple[bool, float, np.ndarray | None]:
        """float64 host check of member i (integers rounded first): (feasible, objective, rounded x). Cached by id."""
        iid = int(popn.ids[i])
        x = self.prob.round_integers(self.bk.asnumpy(popn.X[i]).astype(np.float64))
        if iid in self.checked:
            ok, obj = self.checked[iid]
            return ok, obj, x
        if self.rejected_hashes and vector_hash(x) in self.rejected_hashes:
            # a copy of a vector the solver rejected (re-created by the operators): not feasible for the EA either,
            # so it is neither re-posted nor taken as an incumbent nor protected in Rank 1
            obj = float(self.prob.objective(x))
            self.checked[iid] = (False, obj)
            self.c["rejected_copies"] += 1
            self.trace.event("note", msg="copy of a solver-rejected vector refused", id=iid)
            return False, obj, x
        rk = self.cfg.ranks
        res = self.prob.check(x, rk.feas_tol, rk.int_tol, rk.tol_mode)
        self.checked[iid] = (bool(res.feasible), float(res.objective))
        if not res.feasible:
            self.trace.event("note", msg="float64 re-check failed", id=iid, max_row_viol=res.max_row_viol,
                             max_bound_viol=res.max_bound_viol, max_int_viol=res.max_int_viol)
        return bool(res.feasible), float(res.objective), x

    def _demote_host_dev(self, popn: Population, hf: np.ndarray, hv: np.ndarray, i: int) -> None:
        """Mark member i infeasible (host mirrors and device arrays). A Rank 1 member moves to Rank 2 at once, so it
        neither wins tournaments as Rank 1 nor is recorded with rank_pa/rank_pb = 1 in the genealogy (the design note:
        a rejected solution "moves to Rank 2"); in `_survive` the label is overwritten by the survival anyway."""
        tol = self.cfg.ranks.feas_tol
        hf[i] = False
        hv[i] = max(float(hv[i]), tol)
        popn.feasible[int(i)] = False
        popn.viol[int(i)] = hv[i]
        if popn.rank[int(i)] == R1:
            popn.rank[int(i)] = R2
        self.c["demoted"] += 1

    def _demote_id(self, iid: int, why: str) -> bool:
        pos = np.flatnonzero(self.pop.ids == iid)
        self.checked[iid] = (False, self.checked.get(iid, (False, math.nan))[1])
        self.r1_ids.discard(int(iid))
        if pos.size == 0:
            return False
        i = int(pos[0])
        self._demote_host_dev(self.pop, self.h_feas, self.h_viol, i)
        self.trace.event("note", msg=f"individual {iid} demoted: {why}")
        return True

    # ------------------------------------------------------------------------------------- incumbent and posts
    def _update_incumbent(self) -> None:
        rk = self.cfg.ranks
        feas = np.flatnonzero(self.h_feas)
        if feas.size == 0:
            fin = np.flatnonzero(np.isfinite(self.h_viol))
            if fin.size:
                i = int(fin[np.lexsort((self.h_key[fin], self.h_viol[fin]))[0]])
                v = float(self.h_viol[i])
                if v < self.best_viol * (1 - 1e-9) - 1e-15:
                    self.best_viol = v
                    self.since_best = 0
                    self._best_event(i, feasible=False, obj=self.prob.objective_of_key(float(self.h_key[i])))
                    return
            self.since_best += 1
            return
        order = feas[np.argsort(self.h_key[feas], kind="stable")]
        improved = False
        for i in order[:MAX_DEMOTIONS]:
            i = int(i)
            tol = rk.imp_tol * max(1.0, abs(self.best_key)) if math.isfinite(self.best_key) else 0.0
            if not (self.h_key[i] < self.best_key - tol):
                break
            ok, obj, x = self._check_candidate(self.pop, i)
            if not ok:
                self._demote_id(int(self.pop.ids[i]), "float64 re-check failed")
                self.c["false_feasible"] += 1
                continue
            key = self.prob.key_of_objective(obj)
            if not (key < self.best_key - tol):
                break
            self._new_incumbent(i, x, obj, key)
            improved = True
            break
        self.since_best = 0 if improved else self.since_best + 1

    def _new_incumbent(self, i: int, x: np.ndarray, obj: float, key: float) -> None:
        pop = self.pop
        iid = int(pop.ids[i])
        bits = int(pop.origin[i])
        src = "solver" if _is_solver_find(bits) else "self"
        if src == "solver" and iid in self.cuopt_ids:       # A9: an unchanged cuOpt vector
            src = "cuopt"
            self.c["cuopt_incumbents"] += 1
        t = self.trace.now()
        self.best_key, self.best_obj, self.best_x = key, obj, x
        self.inc_stack.append((iid, key, obj, x))
        row = _SparseRow(x)
        self.records.append((t, obj, src, row))
        if self.part is not None:
            hdr = struct.pack("<ddBi", t, float(obj), SOURCE_CODES.get(src, 255), row.idx.size)
            self.part.write(hdr + row.idx.astype("<i4").tobytes() + row.val.astype("<f8").tobytes())
        self.c["n_incumbents"] += 1
        lpd_copy = self.lpd_on and iid in self.lpd_ids        # LP decode: by membership (origin bits are OR-merged)
        if lpd_copy:
            lpd_k = "F" if self.lpd_kind_of.get(iid) == lpdec.KIND_F else "I"
            self.c["lpd_incumbents"] += 1
            self.c[f"lpd_incumbents_{lpd_k}"] += 1
        if self.strat_on:                                    # a solver find is not the strategy's (module docstring)
            ext = src in ("solver", "cuopt")
            self.sc["incumbents_solver" if ext else "incumbents"][int(pop.strat[i])] += 1
        self.g_new_best.add(iid)
        self.trace.incumbent(obj, source=src, t=t, key=key, id=iid, gen=self.gen)
        self._best_event(i, feasible=True, obj=obj)
        if self.to_solver:
            ie = self.inc_solver                             # the SOLVER's incumbent, not the min with cuOpt's (A9)
            gain = (ie - key) / max(1.0, abs(ie)) if math.isfinite(ie) else math.inf
            if gain > self.cfg.exchange.post_min_gain:
                org = bits | (POST_CUOPT if src == "cuopt" else 0)
                self._post(x[None, :], np.array([obj]), np.array([org], np.int64), [iid], gain=gain)
                if lpd_copy:
                    self.c["lpd_posts"] += 1
                    self.c[f"lpd_posts_{lpd_k}"] += 1

    def _best_event(self, i: int, feasible: bool, obj: float) -> None:
        pop = self.pop
        bits = int(pop.origin[i])
        extra = {}
        if self.strat_on:
            st_i = int(pop.strat[i])
            credited = not _is_solver_find(bits)             # a solver solution is not credited to its strategy
            extra.update(strat=st_i, strat_credited=credited)
            if credited:
                self.sc["best_events"][st_i] += 1
                self.sc_since["best_events"][st_i] += 1
            else:
                self.sc["best_events_solver"][st_i] += 1
        if self.from_cuopt:
            extra["cuopt"] = int(pop.ids[i]) in self.cuopt_ids
        self.trace.event("best", key=float(self.h_key[i]), obj=obj, feasible=feasible, rank=int(pop.rank[i]),
                         origin=bits, ops=origin_names(bits), anc_rank=pop.anc_rank[i].tolist(),
                         anc_ops=pop.anc_ops[i].tolist(), parents=pop.parents[i].tolist(), id=int(pop.ids[i]),
                         birth=int(pop.birth[i]), V=float(self.h_viol[i]), gen=self.gen, **extra)

    def _post(self, X: np.ndarray, obj: np.ndarray, origin: np.ndarray, ids: list[int], gain: float = math.nan,
              frontier: bool = False) -> int:
        seq = self.mailbox.to_solver.post(X, obj, origin=origin)
        self.posted[int(seq)] = [int(v) for v in ids]
        self.posted_hashes[int(seq)] = [vector_hash(X[r]) for r in range(X.shape[0])]
        self.c["posts_frontier" if frontier else "posts"] += 1
        best = float(obj[np.argmin(self.prob.sense * obj)])
        extra = {}
        if self.from_cuopt:                                  # A9: rows that are unchanged cuOpt vectors
            rows = [r for r in range(len(ids)) if int(origin[r]) & POST_CUOPT]
            extra["cuopt_rows"] = rows[:64]
            if rows:
                self.c["cuopt_posts"] += 1
                self.cuopt_posted.update(int(ids[r]) for r in rows)
        self.trace.event("post", seq=int(seq), obj=best, gain=gain, k=int(X.shape[0]), frontier=frontier,
                         ids=[int(v) for v in ids][:64], **extra)
        return int(seq)

    def _refresh_cutoff(self) -> None:
        feas = self.h_feas
        pop_best = float(self.h_key[feas].min()) if feas.any() else math.inf
        inc = min(pop_best, self.inc_external)
        new = inc if math.isfinite(inc) else None
        if new != self.cutoff:
            self.cutoff = new
            self.ls.set_cutoff(new)

    # ---------------------------------------------------------------------------------------------- stagnation
    def _stagnation(self) -> None:
        st = self.cfg.stagnation
        if self.since_best < st.gens:
            return
        self.since_best = 0
        self.c["stagnation_events"] += 1
        r2 = np.flatnonzero(self.pop.rank == R2)
        k = int(round(st.immigrant_frac * r2.size))
        if k <= 0:
            return
        replace = r2[-k:]
        fr_rows = None
        if self.frontier is not None and self.frontier.key.size and self.cfg.frontier.reinject > 0:
            k_fr = int(round(self.cfg.frontier.reinject * k))
            if k_fr:
                fr_rows = self.frontier.sample(k_fr, self.rng_host)
        with self.phase("immigrants"):
            Xn, on, sn = self._immigrants(k, frontier_rows=fr_rows, lp_share=self.cfg.seeding.lp_rounding)
            self._admit(Xn, on, replace=replace, strat=sn)
        self.trace.event("note", msg="stagnation immigrants", gen=self.gen, replaced=int(k),
                         from_frontier=int(0 if fr_rows is None else fr_rows.shape[0]))

    def _admit(self, Xn, on: np.ndarray, replace: np.ndarray | None, strat: np.ndarray | None = None) -> int:
        """Immigrants into the population under the survival's rules (module docstring). Returns the number admitted.
        `strat`: their repair strategies (None: uniform draws while the strategy is on, else 0)."""
        bk, rk = self.bk, self.cfg.ranks
        k = int(Xn.shape[0])
        if k == 0:
            return 0
        if strat is None:
            strat = (self.rng_host.integers(self.S, size=k).astype(np.uint8) if self.strat_on
                     else np.zeros(k, np.uint8))
        strat = np.asarray(strat, np.uint8)
        on = np.asarray(on, np.uint16).copy()
        rep = np.flatnonzero((on & np.uint16(ORIGIN_SEED_RANDOM | ORIGIN_SEED_LP)) != 0)
        if rep.size:
            drep = bk.xp.asarray(rep)
            Xr = Xn[drep]
            self.rc_site = "admit"
            moved = self._repair(Xr, strat[rep])
            Xn[drep] = Xr
            if moved is not None:
                on[rep] |= np.where(moved, ORIGIN_REPAIR, 0).astype(np.uint16)
        ev = self._evaluate(Xn)
        nk, nv, nf = (bk.asnumpy(bk.xp.stack([ev[0], ev[1], ev[2].astype(bk.xp.float64)])))
        nf = nf > 0.5
        if rep.size:
            self._rc_pair(nk, nf, rows=rep)                  # lane i of the call = row rep[i] of the evaluation
        pop = self.pop
        keep = np.ones(pop.size, bool)
        if replace is not None and len(replace):
            keep[np.asarray(replace)] = False
        kp = np.flatnonzero(keep)
        Np = kp.size
        stacked = bk.xp.concatenate([pop.X[bk.xp.asarray(kp)], Xn])
        dup, keep_of = self.dup(stacked)
        del stacked
        hkp, hfp = self.h_key[kp], self.h_feas[kp]
        pop_best = float(hkp[hfp].min()) if hfp.any() else math.inf
        surv_inc = self.last_surv.inc if self.last_surv is not None else math.inf
        inc = min(pop_best, self.inc_external, surv_inc)     # the incumbent this generation's survival used
        feas_phase = not math.isfinite(inc)
        thr = inc - rk.imp_tol * max(1.0, abs(inc)) if not feas_phase else -math.inf
        cap0 = math.floor(rk.x * self.P + 1e-12)
        n0 = int(np.sum(pop.rank[kp] == R0))
        slots = self.P - Np
        chosen, ranks = [], []
        for j in range(k):
            if len(chosen) >= slots:
                break
            i = Np + j
            if dup[i]:
                kk = int(keep_of[i])
                if kk < Np:
                    iid = int(pop.ids[kp[kk]])
                    if int(on[j]) & ~int(pop.origin[kp[kk]]):
                        pop.origin[kp[kk]] |= on[j]
                        self.g_or[iid] = self.g_or.get(iid, 0) | int(on[j])
                continue
            if not (np.isfinite(nk[j]) and np.isfinite(nv[j])):
                continue
            if nf[j]:
                r = R2
            elif feas_phase:
                r = R0 if rk.no_incumbent_policy == "itrea" else R2
            elif nk[j] < thr:
                if rk.rank0_bound_demote and math.isfinite(self.dual_key) and \
                        nk[j] < self.dual_key - rk.imp_tol * max(1.0, abs(self.dual_key)):
                    continue
                if n0 >= cap0:
                    continue
                r = R0
                n0 += 1
            else:
                r = R2
            chosen.append(j)
            ranks.append(r)
        self.c["immigrants_admitted"] += len(chosen)
        self.c["immigrants_dropped"] += k - len(chosen)
        base = pop.take(bk, kp)
        hk, hv, hf = self.h_key[kp], self.h_viol[kp], self.h_feas[kp]
        if chosen:
            ch = np.asarray(chosen)
            dch = bk.xp.asarray(ch)
            ids = self.idc.take(ch.size)
            newp = self._make_pop(Xn[dch], (ev[0][dch], ev[1][dch], ev[2][dch]), np.asarray(ranks, np.int8), ids,
                                  np.full((ch.size, 2), -1, np.int64), on[ch], self.gen, strat=strat[ch])
            self.pop = Population.concat(bk, [base, newp])
            self.h_key = np.concatenate([hk, nk[ch]])
            self.h_viol = np.concatenate([hv, nv[ch]])
            self.h_feas = np.concatenate([hf, nf[ch]])
            now = self.trace.now()
            self.gen_rows.append(dict(
                id=ids, pa=np.full(ch.size, -1, np.int64), pb=np.full(ch.size, -1, np.int64),
                rank_pa=np.full(ch.size, -1, np.int8), rank_pb=np.full(ch.size, -1, np.int8), origin=on[ch],
                birth=np.full(ch.size, self.gen, np.int32), key=nk[ch].astype(np.float64),
                V=nv[ch].astype(np.float64), feasible=nf[ch].astype(bool), t_first=np.full(ch.size, now),
                strat=strat[ch].astype(np.uint8)))
        else:
            self.pop = base
            self.h_key, self.h_viol, self.h_feas = hk, hv, hf
        return len(chosen)

    # ------------------------------------------------------------------------------------------------- frontier
    def _frontier_update(self, cand: Population, hk: np.ndarray, hf: np.ndarray) -> None:
        cfg = self.cfg
        feas = np.flatnonzero(hf & np.isfinite(hk))
        if feas.size == 0:
            return
        bi = int(feas[np.argmin(hk[feas])])
        inc_key = float(hk[bi])
        lim = inc_key + cfg.frontier.band * max(1.0, abs(inc_key))
        sel = feas[(hk[feas] <= lim) & (feas != bi)]
        cap = FRONTIER_OFFER * cfg.frontier.max_size
        if sel.size > cap:
            sel = np.sort(self.rng_host.choice(sel, size=cap, replace=False))
        xp = self.bk.xp
        dsel = xp.asarray(sel)
        st = self.frontier.update(cand.X[dsel], hk[sel], hf[sel], cand.ids[sel], cand.X[bi], inc_key)
        self.trace.event("frontier", gen=self.gen, **st)
        ids_now = frozenset(int(v) for v in self.frontier.ids)
        if cfg.exchange.post_frontier and self.to_solver and self.frontier.key.size and ids_now != self.frontier_posted:
            self.frontier_posted = ids_now
            Xf, keys, _ = self.frontier.host_solutions()
            rows, objs, ids = [], [], []
            rk = cfg.ranks
            for r in range(Xf.shape[0]):
                x = self.prob.round_integers(Xf[r])
                res = self.prob.check(x, rk.feas_tol, rk.int_tol, rk.tol_mode)
                if res.feasible:
                    rows.append(x)
                    objs.append(res.objective)
                    ids.append(int(self.frontier.ids[r]))
            if rows:
                org = np.zeros(len(rows), np.int64)
                if self.from_cuopt:                          # A9: frontier rows are marked by entry (a cuOpt id)
                    org = np.array([POST_CUOPT if i in self.cuopt_ids else 0 for i in ids], np.int64)
                self._post(np.vstack(rows), np.array(objs), org, ids, frontier=True)

    # ------------------------------------------------------------------------------------------------ exchange
    def _exchange_in(self, force: bool = False) -> None:
        """One exchange tick (module docstring, step 1)."""
        cfg = self.cfg
        mb = self.mailbox
        self.next_poll = time.monotonic() + cfg.exchange.poll_s
        if mb is not None and mb.stopped() and self.stop_reason is None:
            self.stop_reason = "stop"
        if self.lp_worker is not None and self.lp_x is None:
            self._poll_lp_worker()
        if self.lpd is not None:                 # LP decode: ready / results / failure (module docstring)
            self._lpd_poll(site="exchange")
        if self.from_solver:
            self._solver_in()
        if self.from_cuopt:                      # after the solver's: a vector both delivered stays the solver's
            self._cuopt_in()

    def _solver_in(self) -> None:
        """The solver's channels (`exchange.from_solver`): solutions, info, the LP point, the LP stream, verdicts."""
        cfg, mb = self.cfg, self.mailbox
        ex = cfg.exchange
        msg = mb.to_ea.poll()
        if msg is not None:
            self._receive(msg)
        info = mb.info.poll()
        if info is not None:
            inc = info.get("incumbent")
            if inc is not None and math.isfinite(float(inc)):
                k = self.prob.key_of_objective(float(inc))
                if k < self.inc_solver:
                    self.inc_solver = k
                self.inc_external = min(self.inc_solver, self.inc_cuopt)
            db = info.get("dual_bound")
            if ex.use_dual_bound and db is not None and math.isfinite(float(db)):
                self.dual_key = max(self.dual_key, self.prob.key_of_objective(float(db)))
            if hasattr(self, "h_feas"):
                self._refresh_cutoff()
        if ex.use_root_lp and cfg.seeding.lp_source == "solver" and self.lp_x is None:
            lm = mb.lp.poll()
            if lm is not None and lm.X.size:
                self._lp_arrived(np.asarray(lm.X[0], dtype=np.float64), source="solver",
                                 obj=float(lm.obj[0]) if lm.obj.size else None)
        if self.lps is not None:
            sm = mb.lps.poll()
            if sm is not None:
                self._lps_receive(sm)
            if self.lps_pool:
                self._lps_use()
        if self.posted:
            n_rej = self.c["solver_rejections"]
            self._verdicts()
            if self.c["solver_rejections"] != n_rej and hasattr(self, "h_feas"):
                self._refresh_cutoff()           # a demoted member may have been the cutoff's population best

    def _cuopt_in(self) -> None:
        """Arm A9 (`exchange.from_cuopt`): cuOpt's channels — its incumbents injected like the solver's (src "cuopt"),
        its incumbent in `inc_external`, its bound recorded only (module docstring, "Three-way arm A9")."""
        mb = self.mailbox
        msg = mb.to_ea_cuopt.poll()
        if msg is not None:
            self._receive(msg, src="cuopt")
        info = mb.info_cuopt.poll()
        if info is None:
            return
        self.c["cuopt_info_msgs"] += 1
        inc = info.get("incumbent")
        if inc is not None and math.isfinite(float(inc)):
            self.inc_cuopt = min(self.inc_cuopt, self.prob.key_of_objective(float(inc)))
            self.inc_external = min(self.inc_solver, self.inc_cuopt)
        db = info.get("dual_bound")
        if db is not None and math.isfinite(float(db)):
            self.dual_key_cuopt = max(self.dual_key_cuopt, self.prob.key_of_objective(float(db)))
        if hasattr(self, "h_feas"):
            self._refresh_cutoff()

    def _receive(self, msg, src: str = "solver") -> None:
        """A solutions message (step 1): the solver's (`to_ea`), or cuOpt's (`to_ea_cuopt`, src "cuopt": its own
        counters and trace kind, so the solver's stay comparable with A3)."""
        cfg, prob = self.cfg, self.prob
        ex = cfg.exchange
        pre = "cuopt_" if src == "cuopt" else ""
        X = np.asarray(msg.X, dtype=np.float64)
        obj = np.asarray(msg.obj, dtype=np.float64)
        n_rows = X.shape[0]
        echo = (np.asarray(msg.origin) == ORIGIN_EA_ECHO) if msg.origin is not None else np.zeros(n_rows, bool)
        self.c[pre + "inject_msgs"] += 1
        self.c[pre + "inject_rows"] += n_rows
        if src == "solver":                      # cuOpt never receives a vector, so it has no echoes
            self.c["echoes"] += int(echo.sum())
        ok = np.flatnonzero(~echo & np.isfinite(obj) & np.all(np.isfinite(X), axis=1)) if n_rows else np.zeros(0, int)
        take = ok[band_select(obj[ok], prob.sense, ex.inject_band, ex.inject_k)] if ok.size else ok
        best = float(obj[take[np.argmin(prob.sense * obj[take])]]) if take.size else None
        self.trace.event("inject" if src == "solver" else "inject_cuopt", seq=int(msg.seq), k=int(take.size),
                         rows=int(n_rows), echo=int(echo.sum()), best_obj=best)
        if take.size == 0:
            return
        Y = X[take].copy()
        if prob.ints.size:
            Y[:, prob.ints] = np.round(Y[:, prob.ints])
        Y = np.clip(Y, prob.lb, prob.ub)
        self.c[pre + "injected"] += int(take.size)
        Xd = self.bk.xp.asarray(Y).astype(self.gene_dtype)
        self.pending.append(_Batch(Xd, np.full(take.size, ORIGIN_INJECTED, np.uint16), repair=False, src=src,
                                   t=self.trace.now(), strat=self._random_strategies(int(take.size))))

    def _lps_receive(self, msg) -> None:
        """New points of the LP stream (ids above the last seen): round each to the nearest integer point, evaluate
        that rounding, and keep them in the pool ranked (V, key). A stream point never becomes the EA's LP point
        (`lp_x` keeps its configured source, `seeding.lp_source` — review of 770df17: adopting the first stream point
        raced the HiGHS worker and made the LP source depend on timing). Ids skipped between two polls (more than
        `ring` posts) are counted in `lps_missed`."""
        prob, bk = self.prob, self.bk
        self.c["lps_msgs"] += 1
        X = np.asarray(msg.X, dtype=np.float64)
        ids = np.asarray(msg.origin, dtype=np.int64) if msg.origin is not None else np.zeros(0, np.int64)
        if ids.shape[0] != X.shape[0] or X.shape[1:] != (prob.n,):
            self.trace.event("note", msg=f"LP stream message {msg.seq} malformed: X {X.shape}, ids {ids.shape}")
            return
        fresh = ids > self.lps_seen
        if fresh.any():
            self.c["lps_missed"] += max(0, int(ids[fresh].min()) - self.lps_seen - 1)
        new = np.flatnonzero(fresh & np.all(np.isfinite(X), axis=1))
        if ids.size:
            self.lps_seen = max(self.lps_seen, int(ids.max()))
        if new.size == 0:
            return
        Y = X[new].copy()
        if prob.ints.size:
            Y[:, prob.ints] = np.round(Y[:, prob.ints])
        Y = np.clip(Y, prob.lb, prob.ub)
        with self.phase("evaluate"):
            ev = self._evaluate(bk.xp.asarray(Y).astype(self.gene_dtype))
        hk, hv, hf = (bk.asnumpy(a) for a in ev)
        now = self.trace.now()
        for j, i in enumerate(new):
            self.lps_pool.append(dict(id=int(ids[i]), x=X[i].copy(), y=Y[j].copy(), obj=float(msg.obj[i]),
                                      V=float(hv[j]), key=float(hk[j]), feas=bool(hf[j]), t=now, used=False))
        self.c["lps_points"] += int(new.size)
        self.lps_pool.sort(key=lambda q: (q["V"], q["key"]))
        self._lps_trim()
        self.trace.event("lps", seq=int(msg.seq), ids=[int(ids[i]) for i in new], V=[float(v) for v in hv],
                         key=[float(k) for k in hk], feasible=int(hf.sum()), pool=len(self.lps_pool))

    def _lps_trim(self) -> None:
        """Keep the pool at `lp_stream.pool` points: USED points leave first (worst first), except the best four, which
        the stagnation immigrants draw from; only then the worst unused ones (counted in `lps_dropped`). So a stream
        whose new points are worse than the best ever seen keeps producing children (review of 770df17)."""
        cap = self.lps.pool
        pool = self.lps_pool
        if len(pool) <= cap:
            return
        excess = len(pool) - cap
        for i in range(len(pool) - 1, 3, -1):                # used ones beyond the best four, worst first
            if excess == 0:
                break
            if pool[i]["used"]:
                del pool[i]
                excess -= 1
        if excess:
            self.c["lps_dropped"] += sum(1 for q in pool[-excess:] if not q["used"])
            del pool[-excess:]

    def _lps_use(self) -> None:
        """Turn up to `lp_stream.per_tick` unused pool points, best (V, key) first, into `lp_stream.roundings` children
        each — the nearest rounding plus randomised ones — repaired in the next candidate set and admitted by ordinary
        survival only. Origin SEED_LP | INJECTED | IMMIGRANT marks them (a combination no other path produces)."""
        ls, bk = self.lps, self.bk
        todo = [q for q in self.lps_pool if not q["used"]][:ls.per_tick]
        lp_only = bool(self.strat_on and self.strat_lp.any())
        for q in todo:
            q["used"] = True
            r = ls.roundings
            Xn = bk.xp.asarray(q["y"][None, :]).astype(self.gene_dtype)
            if r > 1:
                Xr, _ = lp_round(bk, self.dp, q["x"], r - 1, self.rng_dev, self.rng_host,
                                 gene_dtype=self.gene_dtype.name, mem_budget_bytes=self.budgets["operators"])
                Xn = bk.xp.concatenate([Xn, Xr.astype(self.gene_dtype)])
            sk = self._random_strategies(r, lp_only=lp_only)
            self.pending.append(_Batch(Xn, np.full(r, ORIGIN_SEED_LP | ORIGIN_INJECTED | ORIGIN_IMMIGRANT, np.uint16),
                                       repair=True, src="lps", t=self.trace.now(), strat=sk))
            self.c["lps_used"] += 1
            self.c["lps_rows"] += r
            if self.strat_on:
                self.sc["immigrants"] += np.bincount(sk, minlength=self.S)

    def _lp_for_immigrants(self):
        """The LP point for stagnation immigrants: with `lp_stream.immigrants`, the pool's best four, round robin."""
        if self.lps is not None and self.lps.immigrants and self.lps_pool:
            top = self.lps_pool[:4]
            q = top[self.lps_rr % len(top)]
            self.lps_rr += 1
            return q["x"]
        return self.lp_x

    def _lp_arrived(self, x: np.ndarray, source: str, obj: float | None = None, **info) -> None:
        if x.shape[0] != self.prob.n or not np.all(np.isfinite(x)):
            self.lp_info.update(state="bad_point", source=source)
            self.trace.event("lp", source=source, state="bad_point")
            return
        self.lp_x = x
        self.lp_info.update(state="done", source=source, t=self.trace.now(), obj=obj)
        cfg = self.cfg
        k = int(round(cfg.seeding.lp_rounding * cfg.stagnation.immigrant_frac * self.P))
        if cfg.seeding.lp_rounding > 0:
            k = max(1, k)
        if self.strat_on and not self.strat_lp.any():
            k = 0                                            # LP roundings belong to LP-guided strategies only
        self.trace.event("lp", source=source, state="done", obj=obj, immigrants=k, **info)
        if k:
            sk = self._random_strategies(k, lp_only=True)
            X, o = lp_round(self.bk, self.dp, x, k, self.rng_dev, self.rng_host, gene_dtype=self.gene_dtype.name,
                            mem_budget_bytes=self.budgets["operators"])
            self.pending.append(_Batch(X, (o | ORIGIN_IMMIGRANT).astype(np.uint16), repair=True, src="lp",
                                       t=self.trace.now(), strat=sk))
            self.c["lp_immigrants"] += k
            if self.strat_on:
                self.sc["immigrants"] += np.bincount(sk, minlength=self.S)

    def _poll_lp_worker(self) -> None:
        x, info = self.lp_worker.poll()
        state = info.get("state")
        if state == "done" and x is not None:
            self._lp_arrived(np.asarray(x, np.float64), source="highs_async", obj=info.get("objective"),
                             wall_seconds=info.get("wall_seconds"))
        elif state in ("no_solution", "failed", "stopped"):
            self.lp_info.update(state=state, error=info.get("error"))
            self.trace.event("lp", source="highs_async", state=state, error=info.get("error"))
            self.lp_worker.stop()
            self.lp_worker = None

    def _verdicts(self) -> None:
        vs = self.mailbox.verdicts()
        new = vs[self.n_verdicts:]
        self.n_verdicts = len(vs)
        for v in new:
            self.c["verdicts_seen"] += 1
            seq = int(v.get("seq", -1))
            note = v.get("note")
            row = 0
            if isinstance(note, str) and note.startswith("{"):
                try:
                    row = int(json.loads(note).get("row", 0))
                except (ValueError, TypeError):
                    row = 0
            ids = self.posted.get(seq)
            if not ids or row >= len(ids):
                continue
            iid = ids[row]
            if iid in self.cuopt_posted:                     # A9: a verdict on an unchanged cuOpt vector
                self.c["cuopt_rows_accepted" if v.get("accepted") else "cuopt_rows_rejected"] += 1
            if bool(v.get("accepted")):
                self.c["solver_accepted"] += 1
                self.g_accepted.add(iid)
            else:
                self.c["solver_rejections"] += 1
                hs = self.posted_hashes.get(seq)
                if hs is not None and row < len(hs):
                    self.rejected_hashes.add(hs[row])
                self.rejected_ids.add(iid)
                self._demote_id(iid, f"solver rejected post seq {seq}")
                self._revert_incumbent(iid)

    def _revert_incumbent(self, iid: int) -> None:
        """After a rejection: drop rejected incumbents from the top of the stack and fall back to the newest one the
        solver has not rejected (or to none), so that a later EA solution better than THAT is recorded and posted again
        and the rejected vector is not the final best_x. The incumbent records / .part file stay append-only (the
        independent checker decides on the rejected record)."""
        if not (self.inc_stack and self.inc_stack[-1][0] in self.rejected_ids):
            return
        while self.inc_stack and self.inc_stack[-1][0] in self.rejected_ids:
            self.inc_stack.pop()
        if self.inc_stack:
            _, self.best_key, self.best_obj, self.best_x = self.inc_stack[-1]
        else:
            self.best_key, self.best_obj, self.best_x = math.inf, None, None
        self.c["incumbents_reverted"] += 1
        self.trace.event("note", msg="incumbent rejected by solver; reverted", id=int(iid),
                         best_key=self.best_key if math.isfinite(self.best_key) else None, best_obj=self.best_obj)

    # ------------------------------------------------------------------------------------------------- LP decode
    def _lpd_prepare(self, rec: dict) -> None:
        """repair.continuous: lp_decode (module docstring "LP decode"): the active columns, `cont_obj`, the integer
        payload's dtype and the start-up record; the instance file is required only where the decode can act."""
        cfg, prob = self.cfg, self.prob
        rp = cfg.repair
        self.lpd_on = False
        if rp.continuous != "lp_decode":
            return
        cols = lpdec.active_columns(prob)
        self.lpd_cols = cols
        self.lpd_cont_obj = lpdec.continuous_cost(prob)
        ints = np.asarray(prob.ints)
        self.lpd_int_dtype = lpdec.int_part_dtype(np.asarray(prob.lb)[ints], np.asarray(prob.ub)[ints])
        if cols.size == 0 and self.lpd_warm_rows <= 0:
            rec["lp_decode"] = dict(state="inert", reason="no continuous column with lb < ub in two or more rows")
            return
        if self.decoder_factory is None:
            from itrea.paths import resolve
            if not resolve(str(prob.path)).is_file():
                raise ValueError("repair.continuous='lp_decode' needs the instance file (the LP decode worker reads it)")
        self.lpd_on = True
        rec["lp_decode"] = dict(state="configured", active_columns=int(cols.size), cont_obj=self.lpd_cont_obj,
                                per_gen=int(rp.lp_decode_per_gen), time_limit=float(rp.lp_time_limit),
                                cold_limit=float(rp.lp_decode_cold_limit), infeasible=bool(rp.lp_decode_infeasible),
                                int_dtype=self.lpd_int_dtype.name, constants=lpdec.constants())

    def _lpd_begin(self) -> None:
        """At t0: the LP decode's state and counters, and the worker (only where the decode can act)."""
        self.lpd = None
        if not self.lpd_on:
            return
        self.c.update({k: 0 for k in LPD_COUNTERS})
        self.lpd_req = 0
        self.lpd_req_info: dict | None = None        # the outstanding request: its sources, snapshotted at the offer
        self.lpd_offered: set[int] = set()           # ids a reply screened or solved (never offered again)
        self.lpd_ids: set[int] = set()               # ids of the decoded copies that joined a candidate set
        self.lpd_src_of: dict[int, int] = {}         # copy id -> source id (genealogy column lpd_src)
        self.lpd_kind_of: dict[int, int] = {}        # copy id -> its source's kind (KIND_F / KIND_I)
        self._lpd_new: dict[int, int] = {}           # copies in the current candidate set -> kind (for lpd_entered)
        self._lpd_surv = None
        self.lpd_host_s = 0.0                        # EA-thread seconds spent on the decode (perf_counter)
        self.lpd_host_s_max = 0.0
        self.lpd_sync_s = 0.0                        # of which: waiting for the device before the row copies
        self.lpd_sync_s_max = 0.0
        self._lpd_sync_dt = 0.0
        self.lpd_wait_s = 0.0                        # replies: worker send -> EA read (seconds, summed / max)
        self.lpd_wait_s_max = 0.0
        self.lpd_batch_rows_max = 0                  # the largest decoded batch that joined a candidate set
        self.lpd_ready_t: float | None = None
        self.lpd_ready_t_worker: float | None = None
        self.lpd_ready_info: dict | None = None
        self.lpd_stats: dict = {}
        self.lpd_error: str | None = None
        self.lpd_failed = False
        self.lpd_stopped = False
        self.lpd_state_at_stop: str | None = None
        factory = self.decoder_factory if self.decoder_factory is not None else _lp_decoder_for
        self.lpd = factory(self).start()

    def _lpd_time(self, dt: float) -> None:
        self.lpd_host_s += dt
        self.lpd_host_s_max = max(self.lpd_host_s_max, dt)

    def _lpd_tick(self, cand: Population) -> None:
        """The pick site (after the survival): read what the worker sent; if it is idle, send one batch. The device
        wait before the row copies (`_lpd_offer`) is timed apart (`lpd_sync_s`) and left out of `lpd_host_s`."""
        t = time.perf_counter()
        self._lpd_sync_dt = 0.0
        try:
            self._lpd_poll(timed=False, site="pick")         # no decoded batch is pending here (_with_pending ran)
            st = self.lpd.state
            if st != "idle":
                if st in ("starting", "busy"):
                    self.c["lpd_busy"] += 1
                return
            self._lpd_offer(cand)
        finally:
            sync = self._lpd_sync_dt
            self._lpd_time(max(0.0, time.perf_counter() - t - sync))
            if sync > 0.0:
                self.lpd_sync_s += sync
                self.lpd_sync_s_max = max(self.lpd_sync_s_max, sync)

    def _lpd_offer(self, cand: Population) -> None:
        """Select the batch (`lp_decode.select_offers`), wait for the device (timed into `_lpd_sync_dt`), copy its rows
        to the host one by one (a view and a copy, no gather kernel), round and clip the integer parts as `_receive`
        does, snapshot the sources' provenance and offer the integer parts to the worker. A failed send is recorded as
        the worker's failure (`_lpd_fail`). The offered ids are recorded when the reply arrives (`_lpd_result`)."""
        if self._lpd_surv is None:
            return
        cfg, prob, rp = self.cfg, self.prob, self.cfg.repair
        hk, hv, hf, dup, surv = self._lpd_surv
        origin = cand.origin
        solver_find = ((origin & np.uint16(ORIGIN_INJECTED)) != 0) & ((origin & _GENE_BITS_EA) == 0)
        decoded = (origin & np.uint16(ORIGIN_LP_DECODE)) != 0
        off = self.lpd_offered                               # a set lookup per candidate (the set only grows)
        offered = np.fromiter((int(i) in off for i in cand.ids), bool, cand.size)
        F, I = lpdec.select_offers(hk, hv, hf, ids=cand.ids, dup=dup, solver_find=solver_find, decoded=decoded,
                                   offered=offered, inc=float(surv.inc), dual_key=float(self.dual_key),
                                   imp_tol=cfg.ranks.imp_tol, per_gen=rp.lp_decode_per_gen,
                                   cont_obj=self.lpd_cont_obj, infeasible=rp.lp_decode_infeasible)
        if self.lpd_warm_rows > 0:                           # warm-up: exactly lpd_warm_rows rows
            F, I = self._lpd_warm_pick(F, I, hk, hv, dup, offered)
        idx = np.concatenate([F, I]).astype(np.int64)
        if idx.size == 0:
            return
        kinds = np.concatenate([np.full(F.size, lpdec.KIND_F), np.full(I.size, lpdec.KIND_I)]).astype(np.uint8)
        ints = np.asarray(prob.ints)
        xi = np.empty((idx.size, ints.size), dtype=np.float64)
        t_sync = time.perf_counter()                         # the survival's last gather may still run: not ours
        self.bk.synchronize()
        self._lpd_sync_dt = time.perf_counter() - t_sync
        for r, i in enumerate(idx):                          # one row at a time: a view and a copy, no gather kernel
            xi[r] = np.asarray(self.bk.asnumpy(cand.X[int(i)]), dtype=np.float64)[ints]
        xi = np.clip(np.round(xi), np.asarray(prob.lb)[ints], np.asarray(prob.ub)[ints])
        rank_of = np.full(cand.size, NO_RANK, np.int8)
        rank_of[surv.idx] = surv.rank                        # eliminated sources: NO_RANK
        r_src = rank_of[idx]
        o_src = cand.origin[idx]
        sb = src_bits_of_origin(o_src)
        ar, ao = child_ancestry(self.depth, r_src, r_src, o_src, o_src, cand.anc_rank[idx], cand.anc_rank[idx],
                                cand.anc_ops[idx], cand.anc_ops[idx], pa_src=sb, pb_src=sb)
        req = self.lpd_req + 1
        if not self.lpd.offer(req, xi.astype(self.lpd_int_dtype), kinds):
            if self.lpd.state == "failed":                   # the send failed: the worker is gone for the run
                self._lpd_fail(getattr(self.lpd, "error", None) or "send failed")
            return
        self.lpd_req = req
        ids = cand.ids[idx].astype(np.int64)
        self.lpd_req_info = dict(req=self.lpd_req, gen=self.gen, t=self.trace.now(), ids=ids, kinds=kinds, rows=xi,
                                 key=hk[idx].astype(np.float64), V=hv[idx].astype(np.float64),
                                 feasible=hf[idx].copy(), strat=cand.strat[idx].copy(), anc_rank=ar, anc_ops=ao,
                                 rank=r_src)
        self.c["lpd_offers"] += 1
        self.c["lpd_rows_offered"] += int(idx.size)
        self.c["lpd_rows_F"] += int(F.size)
        self.c["lpd_rows_I"] += int(I.size)

    def _lpd_warm_pick(self, F, I, hk, hv, dup, offered) -> tuple[np.ndarray, np.ndarray]:
        """Warm-up only: the selection trimmed or padded (with other finite, distinct, not yet offered candidates as I
        rows) to exactly `lpd_warm_rows` rows, so batches of several rows and of one row both occur."""
        want = int(self.lpd_warm_rows)
        sel = [int(i) for i in np.concatenate([F, I])]
        if len(sel) < want:
            have = set(sel)
            more = [int(i) for i in np.flatnonzero(~np.asarray(dup, bool) & np.isfinite(hk) & np.isfinite(hv)
                                                     & ~offered) if int(i) not in have]
            sel += more[:want - len(sel)]
        sel = sel[:want]
        fs = set(int(i) for i in F)
        return (np.array([i for i in sel if i in fs], np.int64), np.array([i for i in sel if i not in fs], np.int64))

    def _lpd_rel(self, t_mono) -> float | None:
        """A worker's / the pipe's time.monotonic() stamp as seconds since t0 (the trace's clock)."""
        if not isinstance(t_mono, (int, float)) or not math.isfinite(float(t_mono)):
            return None
        t0 = getattr(self.trace, "t0", None)
        return round(float(t_mono) - (t0 if t0 is not None else self.t0), 6)

    def _lpd_poll(self, timed: bool = True, site: str = "exchange", hold: bool = True) -> None:
        """Handle what the worker sent: "ready", a result (the copies join a candidate set), a failure. `hold`: while a
        decoded batch is pending, leave the worker's messages in the pipe, so at most one decoded batch joins a
        candidate set (module docstring); `finish` reads with hold=False. `site` goes into the result event."""
        if self.lpd is None:
            return
        if hold and any(b.src == "lp_decode" for b in self.pending):
            return
        t = time.perf_counter()
        try:
            for msg in self.lpd.poll():
                kind = msg[0]
                if kind == "ready":
                    self.lpd_ready_t = self.trace.now()
                    self.lpd_ready_info = dict(msg[1])
                    self.lpd_ready_t_worker = self._lpd_rel(self.lpd_ready_info.get("t_sent"))
                    self.trace.event("lp_decode", what="ready", ready_t=self.lpd_ready_t,
                                     ready_t_worker=self.lpd_ready_t_worker,
                                     **{k: v for k, v in self.lpd_ready_info.items() if k not in ("t_sent", "t_read")})
                elif kind == "result":
                    self._lpd_result(int(msg[1]), msg[2], site=site)
                elif kind == "failed":
                    self._lpd_fail(str(msg[1]), msg[2] if len(msg) > 2 else "")
        finally:
            if timed:
                self._lpd_time(time.perf_counter() - t)

    def _lpd_result(self, req_id: int, rep: dict, site: str = "exchange") -> None:
        """One reply: counters per code (and per kind), the sources the worker screened or solved recorded as offered,
        the copies (the source's integer part + the LP's continuous values clipped to their bounds; a polish copy only
        if its key beats its source's by imp_tol), one `_Batch` for a candidate set (origin ORIGIN_LP_DECODE, the
        source's strategy, kind and ancestry), the reply's wait in the pipe, one `lp_decode` trace event."""
        info = self.lpd_req_info
        if info is None or int(info["req"]) != int(req_id):
            self.trace.event("note", msg=f"LP decode reply {req_id} matches no outstanding request")
            return
        self.lpd_req_info = None
        prob, imp = self.prob, self.cfg.ranks.imp_tol
        codes = np.asarray(rep["codes"], np.int64)
        k = int(codes.size)
        is_f = np.asarray(info["kinds"])[:k] == lpdec.KIND_F
        self.lpd_stats = dict(rep.get("stats") or {})
        self.c["lpd_replies"] += 1
        cnt = np.bincount(codes, minlength=len(lpdec.CODE_NAMES))
        for code, nm in enumerate(lpdec.CODE_NAMES):
            self.c[f"lpd_{nm}"] += int(cnt[code])
        solved = np.isin(codes, (lpdec.OK, lpdec.INFEASIBLE, lpdec.TIME_LIMIT, lpdec.OTHER))
        self.c["lpd_solved"] += int(solved.sum())
        self.c["lpd_solved_F"] += int((solved & is_f).sum())
        self.c["lpd_solved_I"] += int((solved & ~is_f).sum())
        okm = codes == lpdec.OK
        self.c["lpd_ok_F"] += int((okm & is_f).sum())
        self.c["lpd_ok_I"] += int((okm & ~is_f).sum())
        # a source counts as offered once the worker screened or solved it; NOT_SOLVED / AFTER_DEADLINE stay eligible
        # (only one request is ever outstanding, and the pick site reads before it offers: no source is offered twice)
        done = ~np.isin(codes, (lpdec.NOT_SOLVED, lpdec.AFTER_DEADLINE))
        self.lpd_offered.update(int(v) for v in np.asarray(info["ids"])[:k][done])
        wait = None
        t_sent, t_read = rep.get("t_sent"), rep.get("t_read")
        if isinstance(t_sent, (int, float)) and isinstance(t_read, (int, float)):
            wait = max(0.0, float(t_read) - float(t_sent))   # the finished reply waited this long in the pipe
            self.lpd_wait_s += wait
            self.lpd_wait_s_max = max(self.lpd_wait_s_max, wait)
        ok_rows = np.asarray(rep["ok_rows"], np.int64)
        conts = np.asarray(rep["conts"], np.float64).reshape(ok_rows.size, int(np.asarray(prob.conts).size))
        keys = np.full(k, np.nan)
        kept: list[int] = []
        Y = np.zeros((ok_rows.size, prob.n))
        if ok_rows.size:
            ints, cn = np.asarray(prob.ints), np.asarray(prob.conts)
            Y[:, ints] = info["rows"][ok_rows]
            Y[:, cn] = np.clip(conts, np.asarray(prob.lb)[cn], np.asarray(prob.ub)[cn])
            for j, r in enumerate(ok_rows):
                key = float(prob.key(Y[j]))
                keys[r] = key
                if info["kinds"][r] == lpdec.KIND_F and self.lpd_warm_rows <= 0:
                    sk = float(info["key"][r])
                    if not key < sk - imp * max(1.0, abs(sk)):
                        self.c["lpd_no_gain"] += 1
                        continue
                kept.append(j)
        kept_rows = ok_rows[np.asarray(kept, np.int64)] if kept else np.zeros(0, np.int64)
        if kept:
            kj = np.asarray(kept, np.int64)
            X = self.bk.xp.asarray(Y[kj]).astype(self.gene_dtype)
            self.pending.append(_Batch(X, np.full(kj.size, ORIGIN_LP_DECODE, np.uint16), repair=False, src="lp_decode",
                                       t=self.trace.now(), strat=info["strat"][kept_rows],
                                       anc_rank=info["anc_rank"][kept_rows], anc_ops=info["anc_ops"][kept_rows],
                                       src_ids=info["ids"][kept_rows],
                                       src_kinds=np.asarray(info["kinds"], np.uint8)[kept_rows]))
            nf = int(np.sum(info["kinds"][kept_rows] == lpdec.KIND_F))
            self.c["lpd_kept"] += int(kj.size)
            self.c["lpd_kept_F"] += nf
            self.c["lpd_kept_I"] += int(kj.size) - nf
        ks = set(int(r) for r in kept_rows)
        # the candidate set the copies join: this generation's when read before `_with_pending`, the next one's when
        # read at the pick site; none when read by finish (or when nothing was kept)
        gen_enters = None if (not kept or site == "finish") else self.gen + (1 if site == "pick" else 0)

        def num(v):
            v = float(v)
            return v if math.isfinite(v) else None
        cpu = rep.get("cpu")
        rows = [dict(src=int(info["ids"][r]), kind="F" if info["kinds"][r] == lpdec.KIND_F else "I",
                     code=lpdec.CODE_NAMES[int(codes[r])], key_src=num(info["key"][r]), V_src=num(info["V"][r]),
                     rank_src=int(info["rank"][r]), key=num(keys[r]), obj=num(rep["obj"][r]),
                     iters=int(rep["iters"][r]), s=round(float(rep["secs"][r]), 6),
                     cpu_s=round(float(cpu[r]), 6) if cpu is not None else None, limit=num(rep["limit"][r]),
                     status=str(rep["status"][r]), kept=int(r) in ks) for r in range(k)]
        rs = rep.get("read_s")
        self.trace.event("lp_decode", what="result", req=int(req_id), gen_offered=int(info["gen"]),
                         t_offered=info["t"], gen=self.gen, site=site, gen_enters=gen_enters, kept=len(kept),
                         t_sent=self._lpd_rel(t_sent), wait_s=None if wait is None else round(wait, 6),
                         read_s=round(float(rs), 6) if isinstance(rs, (int, float)) else None, rows=rows,
                         stats=self.lpd_stats)

    def _lpd_entering(self, ids: np.ndarray, src_ids: np.ndarray, kinds: np.ndarray | None = None) -> None:
        """A decoded batch joins the candidate set (`_with_pending`): copy id -> source id and kind."""
        kinds = np.full(len(ids), lpdec.KIND_I, np.uint8) if kinds is None else np.asarray(kinds, np.uint8)
        for i_new, i_src, kd in zip(ids, src_ids, kinds):
            self.lpd_src_of[int(i_new)] = int(i_src)
            self.lpd_kind_of[int(i_new)] = int(kd)
            self._lpd_new[int(i_new)] = int(kd)
        self.lpd_ids.update(int(i) for i in ids)
        self.lpd_batch_rows_max = max(self.lpd_batch_rows_max, int(len(ids)))
        self.c["lpd_batches"] += 1

    def _lpd_fail(self, err: str, tb: str = "") -> None:
        """The worker failed (digest mismatch, an exception, death): recorded, never retried; the EA never waits."""
        self.lpd_failed = True
        self.lpd_error = err
        self.c["lpd_failed"] = 1
        if self.lpd_req_info is not None:
            self.c["lpd_lost"] += 1
            self.lpd_req_info = None
        self.trace.event("lp_decode", what="failed", error=str(err)[:1000], traceback=(tb or "")[-4000:] or None)

    def _lpd_stop(self) -> None:
        """Stop the worker (idempotent); an outstanding request is counted lost."""
        if self.lpd is None or self.lpd_stopped:
            return
        self.lpd_state_at_stop = self.lpd.state
        self.lpd.stop()
        self.lpd_stopped = True
        if self.lpd_req_info is not None:
            self.c["lpd_lost"] += 1
            self.lpd_req_info = None

    def _lpd_summary(self) -> dict:
        rp = self.cfg.repair
        dec = self.lpd
        ready = None
        if self.lpd_ready_info is not None:
            ready = {k: v for k, v in self.lpd_ready_info.items() if k not in ("t_sent", "t_read")}
        ipc = getattr(dec, "ipc", None)
        return dict(state="failed" if self.lpd_failed else (self.lpd_state_at_stop or (dec.state if dec else None)),
                    error=self.lpd_error, ready_t=self.lpd_ready_t, ready_t_worker=self.lpd_ready_t_worker,
                    ready=ready, pid=getattr(dec, "pid", None), host_s=round(self.lpd_host_s, 6),
                    host_s_max=round(self.lpd_host_s_max, 6), sync_s=round(self.lpd_sync_s, 6),
                    sync_s_max=round(self.lpd_sync_s_max, 6), wait_s_total=round(self.lpd_wait_s, 6),
                    wait_s_max=round(self.lpd_wait_s_max, 6),
                    ipc={k: (round(v, 6) if isinstance(v, float) else v) for k, v in ipc.items()} if ipc else None,
                    batch_rows_max=int(self.lpd_batch_rows_max), worker=dict(self.lpd_stats),
                    active_columns=int(self.lpd_cols.size), cont_obj=self.lpd_cont_obj,
                    per_gen=int(rp.lp_decode_per_gen), time_limit=float(rp.lp_time_limit),
                    cold_limit=float(rp.lp_decode_cold_limit), infeasible=bool(rp.lp_decode_infeasible),
                    int_dtype=self.lpd_int_dtype.name, constants=lpdec.constants())

    # ------------------------------------------------------------------------------------------------ records
    def _gen_record(self) -> None:
        surv = self.last_surv
        hk, hf, pop = self.h_key, self.h_feas, self.pop
        fin = hk[np.isfinite(hk)]
        inc = surv.inc if surv is not None else math.inf
        rk = self.cfg.ranks
        superopt = 0
        if math.isfinite(inc):
            thr = inc - rk.imp_tol * max(1.0, abs(inc))
            superopt = int(np.sum(~hf & (hk < thr)))
        now = time.monotonic()
        t_prev, ch_prev = self.last_log
        dt = max(now - t_prev, 1e-9)
        cps = (self.c["children"] - ch_prev) / dt
        self.last_log = (now, self.c["children"])
        phases = self.phase.take()
        self.logged_gen = self.gen
        extra = {}
        if self.strat_on:
            extra = dict(strat_pop=np.bincount(pop.strat, minlength=self.S).tolist(),
                         strat_best=self.sc_since["best_events"].tolist(),
                         strat_children=self.sc_since["children"].tolist())
            for v in self.sc_since.values():
                v[:] = 0
        if self.lps is not None:
            both = ORIGIN_SEED_LP | ORIGIN_INJECTED
            extra.update(rank0_stream=int(np.sum((pop.rank == R0) & ((pop.origin & both) == both))),
                         lps_pool=len(self.lps_pool), lps_unused=sum(1 for q in self.lps_pool if not q["used"]))
        if self.rclock is not None:
            extra.update(self.rclock.gen_fields())
        if self.from_cuopt:
            extra.update(inc_solver=self.inc_solver, inc_cuopt=self.inc_cuopt, dual_key_cuopt=self.dual_key_cuopt,
                         cuopt_pop=self._cuopt_members())
        extra.update(self._r1_spread())                     # both Rank 1 modes (D-016's mechanism check)
        if self.r1_k is not None:
            extra["r1_cleared"] = self.r1_cleared_last          # candidates the distance rule skipped, last survival
        if self.lpd_on:
            extra.update(lpd_state=self.lpd.state if self.lpd is not None else None,
                         **{k: self.c[k] for k in LPD_GEN_FIELDS}, lpd_host_s=round(self.lpd_host_s, 6),
                         lpd_host_s_max=round(self.lpd_host_s_max, 6), lpd_sync_s=round(self.lpd_sync_s, 6))
        self.trace.event(
            "gen", gen=self.gen, best_key=float(hk[hf].min()) if hf.any() else None,
            median_key=float(np.median(fin)) if fin.size else None, feas_share=float(hf.mean()) if hf.size else 0.0,
            n0=int(np.sum(pop.rank == R0)), n1=int(np.sum(pop.rank == R1)), n2=int(np.sum(pop.rank == R2)),
            pop=int(pop.size), inc=inc, inc_external=self.inc_external, dual_key=self.dual_key,
            rank0_candidates=surv.rank0_candidates if surv else 0, rank0_demoted=surv.rank0_demoted if surv else 0,
            rank0_over_quota=surv.rank0_over_quota if surv else 0, superopt_survivors=superopt,
            shortfall=surv.shortfall if surv else 0, children_per_s=round(cps, 3),
            gpu_mem_mb=round(self.bk.mem_used_bytes() / 2 ** 20, 2), gpu_pool_mb=round(self.bk.mem_total_bytes() / 2 ** 20, 2),
            feasibility_phase=bool(surv.feasibility_phase) if surv else True, seconds=phases,
            ls_moves=self.c["ls_moves"], oom_retries=self.c["oom_retries"],
            best_viol=self.best_viol if not hf.any() else 0.0, **extra)

    def _r1_spread(self) -> dict:
        """Rank 1's size and spread (`elite.elite_spread`): logged with every `gen` event and in the summary."""
        rows = np.flatnonzero(self.pop.rank == R1)
        return elite_spread(self.bk, self.pop.X, rows, self.h_key[rows], self.dp.ints, band=self.cfg.frontier.band)

    def finish(self) -> dict:
        """Final records and the summary (call once, after the last step)."""
        if self.lp_worker is not None:
            self.lp_worker.stop()
        if self.lpd is not None:                             # the last reply's statistics (after the clock; a pending
            self._lpd_poll(timed=False, site="finish", hold=False)   # decoded batch never enters now), then stop
            self._lpd_stop()
        if self.logged_gen != self.gen:
            self._gen_record()
        wall = time.monotonic() - self.wall_start
        pop = self.pop
        c = dict(self.c)
        out = dict(
            exit_reason=self.stop_reason or "budget", generations=self.gen, wall_s=round(wall, 6),
            max_generation_s=round(self.max_gen_s, 6),
            evaluations_per_s=round(c["evaluations"] / max(wall, 1e-9), 3),
            ls_moves_per_s=round(c["ls_moves"] / max(wall, 1e-9), 3),
            feasible_found=self.best_x is not None, best_key=self.best_key if self.best_x is not None else None,
            best_obj=self.best_obj, final_n0=int(np.sum(pop.rank == R0)), final_n1=int(np.sum(pop.rank == R1)),
            final_n2=int(np.sum(pop.rank == R2)), final_pop=int(pop.size), P=self.P, offspring=self.O,
            inc_external=self.inc_external, dual_key=self.dual_key, lp=dict(self.lp_info),
            peak_gpu_used_mb=round(self.peak_used / 2 ** 20, 2), peak_gpu_pool_mb=round(self.peak_pool / 2 ** 20, 2),
            exchange_from_solver=self.from_solver, exchange_to_solver=self.to_solver,
            genealogy_rows=int(sum(len(r["id"]) for r in self.gen_rows)), **c)
        out.update({f"final_{k}": v for k, v in self._r1_spread().items()})
        if self.strat_on:
            out["strategy"] = dict(names=list(self.strat_names), row_scale=[sc for sc, _ in self.strat_list],
                                   lp_guidance=[bool(lp) for _, lp in self.strat_list],
                                   final_pop=np.bincount(pop.strat, minlength=self.S).tolist(),
                                   **{k: v.tolist() for k, v in self.sc.items()}, **self.c_strat)
        if self.frontier is not None:
            Xf, keys, dist = self.frontier.host_solutions()
            out["frontier"] = dict(X=Xf, keys=keys, dist=dist, ids=self.frontier.ids.copy(),
                                   stats=self.frontier.stats(self.best_key if math.isfinite(self.best_key)
                                                             else (float(keys.min()) if keys.size else math.inf)))
        if self.lps is not None:
            both = ORIGIN_SEED_LP | ORIGIN_INJECTED
            out["lp_stream"] = dict(
                pool=len(self.lps_pool), unused=sum(1 for q in self.lps_pool if not q["used"]),
                best=dict(V=self.lps_pool[0]["V"], key=self.lps_pool[0]["key"]) if self.lps_pool else None,
                final_rank0_stream=int(np.sum((pop.rank == R0) & ((pop.origin & both) == both))),
                final_stream=int(np.sum((pop.origin & both) == both)))
        if self.rclock is not None:
            out["repair_clock"] = self.rclock.record()
        if self.from_cuopt:
            out.update(exchange_from_cuopt=True, inc_solver=self.inc_solver, inc_cuopt=self.inc_cuopt,
                       dual_key_cuopt=self.dual_key_cuopt, final_cuopt_pop=self._cuopt_members())
        if self.lpd_on:
            out["lp_decode"] = self._lpd_summary()
        out["best_x"] = self.best_x
        return out

    def _cuopt_members(self) -> int:
        """A9: members of the population that entered as injected cuOpt rows."""
        if not self.cuopt_ids:
            return 0
        return int(np.isin(self.pop.ids, np.fromiter(self.cuopt_ids, np.int64, len(self.cuopt_ids))).sum())

    # ------------------------------------------------------------------------------------------------- outputs
    def genealogy(self) -> dict:
        """The edge list as column arrays (flags and OR-merged origins applied)."""
        cols = ("id", "pa", "pb", "rank_pa", "rank_pb", "origin", "birth", "key", "V", "feasible", "t_first", "strat")
        dt = dict(id=np.int64, pa=np.int64, pb=np.int64, rank_pa=np.int8, rank_pb=np.int8, origin=np.uint16,
                  birth=np.int32, key=np.float64, V=np.float64, feasible=bool, t_first=np.float64, strat=np.uint8)
        if self.gen_rows:
            g = {c: np.concatenate([r[c] for r in self.gen_rows]).astype(dt[c]) for c in cols}
        else:
            g = {c: np.zeros(0, dt[c]) for c in cols}
        if self.g_or:
            pos = {int(v): i for i, v in enumerate(g["id"])}
            for iid, bits in self.g_or.items():
                if iid in pos:
                    g["origin"][pos[iid]] |= np.uint16(bits)
        g["is_new_best"] = np.isin(g["id"], np.fromiter(self.g_new_best, np.int64, len(self.g_new_best)))
        g["is_accepted_post"] = np.isin(g["id"], np.fromiter(self.g_accepted, np.int64, len(self.g_accepted)))
        if getattr(self, "from_cuopt", False):              # A9: the row entered as an injected cuOpt vector
            cu = getattr(self, "cuopt_ids", set())
            g["cuopt"] = np.isin(g["id"], np.fromiter(cu, np.int64, len(cu)))
        if getattr(self, "lpd_on", False):                  # LP decode: the source of a decoded copy, else -1
            src = getattr(self, "lpd_src_of", {})
            g["lpd_src"] = np.array([src.get(int(i), -1) for i in g["id"]], dtype=np.int64)
        return g

    def write_outputs(self, run_dir: str | os.PathLike, role: str = "ea") -> dict:
        """<role>.incumbents.npz (every EA incumbent vector), <role>.genealogy.npz and <role>.frontier.npz.
        Safe to call after an exception (writes what exists). Returns {name: path}."""
        d = Path(run_dir)
        d.mkdir(parents=True, exist_ok=True)
        out = {}
        recs = getattr(self, "records", None)
        if recs is not None:
            out["incumbents"] = str(save_incumbents(d / f"{role}.incumbents.npz", self.prob.names,
                                                    [(t, o, s, r.dense()) for t, o, s, r in recs]))
            if self.part is not None:
                self.part.close()
                self.part = None
                for f in self.part_paths:
                    f.unlink(missing_ok=True)
        if getattr(self, "gen_rows", None) is not None:
            p = d / f"{role}.genealogy.npz"
            tmp = p.with_name(p.name + ".tmp.npz")
            np.savez_compressed(tmp, **self.genealogy())
            os.replace(tmp, p)
            out["genealogy"] = str(p)
        if getattr(self, "frontier", None) is not None and getattr(self, "records", None) is not None:
            Xf, keys, dist = self.frontier.host_solutions()
            p = d / f"{role}.frontier.npz"
            tmp = p.with_name(p.name + ".tmp.npz")
            objs = np.array([self.prob.objective_of_key(k) for k in keys], dtype=np.float64)
            np.savez_compressed(tmp, X=Xf, keys=keys, obj=objs, dist=dist, ids=self.frontier.ids.astype(np.int64),
                                names=np.array(list(self.prob.names), dtype=str))
            os.replace(tmp, p)
            out["frontier"] = str(p)
        return out


# ------------------------------------------------------------------------------------------- warm-up mailbox
class _Chan:
    def __init__(self, msgs=None):
        self.msgs = list(msgs or [])
        self.seq = 0
        self.posted: list = []

    def post(self, X, obj, origin=None) -> int:
        self.seq += 1
        self.posted.append((self.seq, np.asarray(X), np.asarray(obj)))
        return self.seq

    def poll(self):
        return self.msgs.pop(0) if self.msgs else None


class _Info:
    def __init__(self, d):
        self.d = d

    def poll(self):
        d, self.d = self.d, None
        return d


class _WarmupMailbox:
    """In-memory partner for the warm-up: one solver message (a feasible point and an infeasible variant), one info
    message (incumbent and bound, so the cutoff row is armed), one LP point, one LP-stream ring, and a rejection of the
    first post; and cuOpt's channels (read only with `exchange.from_cuopt`, arm A9): one row and one info message."""

    def __init__(self, syn: Problem, rows: int = 3):
        from itrea.exchange.mailbox import Message
        from itrea.solvers.common import ORIGIN_SOLVER
        x0 = np.asarray(syn.toy_point, dtype=np.float64)       # type: ignore[attr-defined]
        bad = x0.copy()
        bad[0] = syn.ub[0] + 3.0 if np.isfinite(syn.ub[0]) else bad[0] + 3.0
        obj0 = syn.objective(x0)
        msg = Message(X=np.vstack([x0, bad, x0])[:rows], obj=np.array([obj0, obj0, obj0])[:rows], seq=1, t=0.0,
                      origin=np.array([ORIGIN_SOLVER, ORIGIN_SOLVER, ORIGIN_EA_ECHO], np.int64)[:rows])
        self.to_ea = _Chan([msg])
        self.to_solver = _Chan()
        self.info = _Info(dict(incumbent=obj0, dual_bound=obj0 - syn.sense * 50.0, status="solving"))
        lp = x0.copy()
        lp[syn.isint] += 0.3
        self.lp = _Chan([Message(X=lp[None, :], obj=np.array([obj0]), seq=1, t=0.0)])
        # the LP stream (D-011, read only with lp_stream.enabled): `rows` points in one ring, so the "one" pass
        # evaluates a single nearest rounding as well
        pts = np.vstack([x0 + 0.3 * syn.isint, x0 + 0.6 * syn.isint, x0 + 0.2 * syn.isint])[:rows]
        pts = np.clip(pts, syn.lb, syn.ub)
        self.lps = _Chan([Message(X=pts, obj=np.array([syn.objective(r) for r in pts]), seq=1, t=0.0,
                                  origin=np.arange(1, pts.shape[0] + 1, dtype=np.int64))])
        # cuOpt (A9): its one row is the feasible point as well, so it meets the solver's copy as a duplicate
        self.to_ea_cuopt = _Chan([Message(X=x0[None, :], obj=np.array([obj0]), seq=1, t=0.0,
                                          origin=np.array([ORIGIN_SOLVER], np.int64))])
        self.info_cuopt = _Info(dict(incumbent=obj0, dual_bound=obj0 - syn.sense * 50.0, status="solving",
                                     solver="cuopt"))

    def stopped(self) -> bool:
        return False

    def verdicts(self) -> list[dict]:
        if not self.to_solver.posted:
            return []
        return [dict(seq=self.to_solver.posted[0][0], t=0.0, obj=0.0, accepted=False, note='{"row": 0}')]


class _WarmupDecoder:
    """In-process stand-in for the LP decode worker during the warm-up (like `_WarmupMailbox`): "ready" at the first
    poll, then every offered batch answered at the next poll with every row OK and the continuous values at the point
    of their bounds nearest 0 — so the copy, batch, evaluation and survival kernels of the decode path compile before
    t0 on batches of several rows and of one row."""

    def __init__(self, eng: "Engine"):
        self.eng = eng
        self.state = "not_started"
        self.pid = None
        self._req = None

    def start(self) -> "_WarmupDecoder":
        self.state = "starting"
        return self

    def offer(self, req_id: int, rows: np.ndarray, kinds: np.ndarray, per_gen: int | None = None) -> bool:
        if self.state != "idle":
            return False
        self._req = (int(req_id), np.asarray(rows))
        self.state = "busy"
        return True

    def poll(self) -> list[tuple]:
        if self.state == "starting":
            self.state = "idle"
            return [("ready", dict(stub=True))]
        if self.state != "busy" or self._req is None:
            return []
        req_id, rows = self._req
        self._req = None
        self.state = "idle"
        prob = self.eng.prob
        k = int(rows.shape[0])
        cn = np.asarray(prob.conts)
        v = np.clip(np.zeros(cn.size), np.asarray(prob.lb)[cn], np.asarray(prob.ub)[cn])
        rep = dict(codes=np.full(k, lpdec.OK, np.uint8), ok_rows=np.arange(k, dtype=np.int64),
                   conts=np.repeat(v[None, :], k, axis=0), obj=np.zeros(k), iters=np.zeros(k, np.int64),
                   secs=np.zeros(k), limit=np.full(k, np.nan), status=["stub"] * k, stats={})
        return [("result", req_id, rep)]

    def stop(self) -> bool:
        lost = self.state == "busy"
        self.state = "stopped"
        self._req = None
        return lost
