"""SCIP solver process: `python -m itrea.solvers.scip_run --instance P --seed S --budget T --run-dir D [--mailbox M]
[--run-id R] [--role-name N] [--no-barrier] [--sol-out F] [--mem-mb MB] [--recv] [--send] [--inject-band B]
[--inject-k K] [--poll-s S] [--inert] [--no-plugin] [--emphasis E] [--arm A] [--lp-stream [--lp-stream-share F]
[--lp-stream-ring R] [--lp-stream-min-gap S] [--lp-stream-q Q] [--lp-stream-patience S]]`.

PySCIPOpt 6.2.1 / SCIP 10, one thread, `limits/memory` = --mem-mb, `randomization/randomseedshift` = seed (the only
seed parameter varied, as in optopt; common random numbers across arms of a block come from passing the same seed).

Why the exchange is a heuristic plugin. Solutions can enter SCIP safely only where SCIP calls a primal heuristic:
optopt's event-handler injection crashed SCIP on 9 of 80 instances per arm (2026-09-23), the heuristic route
(`_XchgHeur`, adapted here from berkorbay/optopt@2a3348c0a102, MIT, `src/optopt/solvers/scip.py`) never did. The plugin's
timing mask BEFOREPRESOL | DURINGPRESOLLOOP | BEFORENODE | DURINGLPLOOP | AFTERLPNODE also covers presolve and the
root cut loop (critique, systems lens: "SCIP injection window"). There is no callback inside a single simplex solve,
so an instance that spends long in one root LP has no exchange window; the callback statistics in the summary
(`t_first_callback`, `t_first_solving_callback`, `max_gap_s`) identify that "no exchange window" subgroup (D-002 §5).

What the plugin does at each call: read the monotonic clock (that is all it costs between polls). At most every
--poll-s it (1) samples the dual bound into the trace (every mode with the plugin, so --inert and active runs carry
the same tracing work), (2) with --recv polls `to_solver` and tries every new row with `createOrigSol(heur=self)` +
`setSolVal` (non-zeros only; the solution starts at zero) + `trySol` over the CACHED original variables (critique:
`getVars(transformed=False)` costs 6 ms per 100k variables per call), (3) with --send posts what the recorder marked
new. At most once per second, in every mode, it checks the runner's stop file and SIGTERM (control plane).

Verdict semantics (the EA engine marks an individual infeasible when `accepted` is false, so it must mean "SCIP's own
feasibility check passed", not "stored"): `stored` = `trySol`'s result (feasible AND kept — a duplicate of a stored
solution, or a solution SCIP does not keep, returns False); when not stored, a fresh copy goes through
`checkSol(original=True)` to separate "infeasible" from "feasible, not kept". Every row seen gets one verdict line:
{seq, obj, accepted} in the mailbox (`note` = JSON with stored, feasible, primal_before, primal_after, row, dup,
obj_solver) and a `verdict` trace event. Rows are tried even when worse than SCIP's incumbent: the frontier-posting
ablation (D-005) relies on SCIP storing non-improving solutions for its crossover heuristics. Rows are de-duplicated by
`vector_hash` (a re-posted vector reuses its first verdict). An incumbent SCIP makes from a received row is recorded
with source "ea" — or "cuopt" when the row's origin carries POST_CUOPT (arm A9, D-015: an unchanged cuOpt vector the
EA passed on), so the analysis credits it to cuOpt; the verdict note then has `cuopt: true`.

Sending (--send): the BESTSOLFOUND event handler records every incumbent (vector over the cached original variables;
the objective is read from the solution itself — at BESTSOLFOUND `getPrimalbound()` still holds the previous value,
optopt's recorder bug of 2026-09-23) and marks the pool dirty; it NEVER injects or posts. The next plugin tick posts to
`to_ea` the incumbent plus the stored solutions within --inject-band of it (at most --inject-k, best first, only those
not posted before — identified by (solution time, objective) so already-posted vectors are not re-read), coalesced to
one post per --poll-s. Each row carries ORIGIN_SOLVER or ORIGIN_EA_ECHO (its hash is one SCIP received from the EA).
The info channel gets {incumbent, dual_bound, status, stage, n_sols, nodes} when any of them changes; the root LP
solution (original space, `getSolVal(None, v)`) is posted once to `lp` at the first depth-0 AFTERLPNODE call with an
optimal LP.

LP stream (--lp-stream, D-011; Berk 2026-10-06: "when solver comes up with suitable good infeasible solutions
(minimize infeasibility then maximize objective function) it might be able to inject to EA's population"): at every
plugin call after an optimal node LP (AFTERLPNODE, and DURINGLPLOOP for the cut rounds; never in probing) the plugin
reads two free numbers — the count of fractional integer variables (`getNLPBranchCands`, the infeasibility proxy:
nearest rounding must repair at least that many coordinates) and the LP objective. An integral LP point is skipped
(SCIP checks it itself). A point is READ (`getSolVal(None, v)` over the cached original variables, ~0.4 µs per
variable, measured 2026-10-06) only when (a) a token bucket allows it — the stream's own time (read + post) stays
under --lp-stream-share of the wall clock since t0, plus a start allowance of two reads so the first point leaves at
once — (b) at least --lp-stream-min-gap seconds passed since the last read, and (c) the node is good: during the
first 8 LP solves any node; then its fractional count is at most the --lp-stream-q quantile of the last 32 LP solves,
or --lp-stream-patience seconds passed since (a) and (b) first allowed a read (so a long dive of poor nodes cannot
starve the stream). Every node LP has an objective no worse than the root bound and better than SCIP's incumbent
(else the node is pruned): the points are super-optimal and, rounded, are Rank 0 material for the EA. The last
--lp-stream-ring points are posted together to `lps` (newest last; `origin` carries the point ids), so an EA that
misses a post loses nothing.

--inert: the plugin is installed and does (1) and the stop checks, never polls or posts (harness-matched A0).
Without --recv/--send the plugin is inert as well. --no-plugin (calibration only, E0's "no exchange" reference): no
plugin; the stop file is then honoured only at incumbents and SIGTERM only through SCIP's own limits.
"""
from __future__ import annotations

import collections
import json
import time

import numpy as np

from itrea.solvers.common import (ORIGIN_EA_ECHO, ORIGIN_SOLVER, POST_CUOPT, BadArguments, band_select, base_parser,
                                  finite_or_none, improves, run_main, validate_common, vector_hash, versions)

MODULE = "itrea.solvers.scip_run"


_AREAS = ("heuristics", "separating", "presolving")
_SETTINGS = ("aggressive", "fast", "off")
_EMPHASES = ("feasibility", "optimality", "easycip", "hardlp", "cpsolver", "counter", "numerics", "benchmark",
             "phasefeas", "phaseimprove", "phaseproof")


def build_parser():
    ap = base_parser(f"python -m {MODULE}", "SCIP solver process with the I-TREA II exchange plugin", "solver")
    ap.add_argument("--emphasis", default=None,
                    help="static SCIP emphasis for A0* (D-002): 'default', one of "
                         f"{'|'.join(_EMPHASES)} (setEmphasis), or <area>_<setting> with area in {'|'.join(_AREAS)} "
                         f"and setting in {'|'.join(_SETTINGS)}")
    g = ap.add_argument_group("LP stream (D-011)")
    g.add_argument("--lp-stream", action="store_true",
                   help="stream good infeasible node LP points to the EA (needs --send)")
    g.add_argument("--lp-stream-share", type=float, default=0.02,
                   help="the stream's read+post time stays under this share of the wall clock")
    g.add_argument("--lp-stream-ring", type=int, default=3, help="points per post (the newest)")
    g.add_argument("--lp-stream-min-gap", type=float, default=0.25, help="seconds between two reads, at least")
    g.add_argument("--lp-stream-q", type=float, default=0.25,
                   help="a node is good if its fractional count is at most this quantile of the last 32 LP solves")
    g.add_argument("--lp-stream-patience", type=float, default=1.0,
                   help="seconds a due read waits for a good node before it takes the next one")
    return ap


def _parse_emphasis(name: str | None):
    """-> None | ("emph", NAME) | (area, SETTING). Raises BadArguments for an unknown name."""
    if name is None or name == "default":
        return None
    if name in _EMPHASES:
        return ("emph", name.upper())
    area, _, setting = name.partition("_")
    if area in _AREAS and setting in _SETTINGS:
        return (area, setting.upper())
    raise BadArguments(f"--emphasis {name!r} is not a known SCIP emphasis")


def apply_emphasis(m, name: str | None) -> None:
    """Applied BEFORE every other parameter: setEmphasis(DEFAULT-like) calls may reset limits and seeds.
    Adapted from berkorbay/optopt@2a3348c0a102, MIT (`EMPHASIS` in src/optopt/solvers/scip.py)."""
    from pyscipopt import SCIP_PARAMEMPHASIS, SCIP_PARAMSETTING
    e = _parse_emphasis(name)
    if e is None:
        return
    kind, val = e
    if kind == "emph":
        m.setEmphasis(getattr(SCIP_PARAMEMPHASIS, val))
    else:
        {"heuristics": m.setHeuristics, "separating": m.setSeparating,
         "presolving": m.setPresolve}[kind](getattr(SCIP_PARAMSETTING, val))


def validate(args) -> None:
    validate_common(args)
    if args.seed < 0:
        raise BadArguments("--seed must be >= 0 (randomization/randomseedshift)")
    _parse_emphasis(args.emphasis)
    if args.lp_stream and not args.send:
        raise BadArguments("--lp-stream needs --send")
    if not (0.0 < args.lp_stream_share <= 1.0 and args.lp_stream_ring >= 1 and args.lp_stream_min_gap >= 0.0
            and 0.0 <= args.lp_stream_q <= 1.0 and args.lp_stream_patience >= 0.0):
        raise BadArguments("--lp-stream-share in (0, 1], --lp-stream-ring >= 1, --lp-stream-min-gap >= 0, "
                           "--lp-stream-q in [0, 1], --lp-stream-patience >= 0")


def scip_version() -> str:
    from pyscipopt import Model
    return str(Model().version())


class _Exchange:
    """State shared by the recorder (event handler) and the plugin (heuristic) of one SCIP run."""

    def __init__(self, proc, model, orig_vars, mode: str):
        args = proc.args
        self.proc, self.m, self.ov, self.mode = proc, model, orig_vars, mode   # mode: none | inert | active
        self.n = len(orig_vars)
        self.recv, self.send = bool(args.recv), bool(args.send)
        self.poll_s, self.band, self.k = args.poll_s, args.inject_band, args.inject_k
        self.sense = proc.sense
        self.c_local = np.array([v.getObj() for v in orig_vars], dtype=np.float64)
        self.offset = float(model.getObjoffset(original=True))
        self.injecting = False
        self.inject_source = "ea"                  # "cuopt" while trying a POST_CUOPT row (A9)
        self.dirty = False
        self.next_poll = 0.0
        self.last_inc: tuple[tuple[float, float], np.ndarray] | None = None   # ((sol time, obj), x local)
        self.posted_keys: set[tuple[float, float]] = set()
        self.received: set[bytes] = set()          # hashes of every EA vector seen (echo marking)
        self.tried: dict[bytes, bool] = {}         # hash -> SCIP-feasible verdict (de-duplication)
        self.last_seq = 0
        self.lp_posted = False
        self.last_info = None
        self.last_bound = None
        self.record_s = 0.0
        # callback statistics
        self.calls = 0
        self.calls_by_timing: dict[int, int] = {}
        self.t_first = None
        self.t_first_solving = None
        self.t_last = None
        self.max_gap = 0.0
        from pyscipopt import SCIP_HEURTIMING, SCIP_LPSOLSTAT, SCIP_RESULT, SCIP_STAGE
        self.SOLVING, self.AFTERLPNODE = SCIP_STAGE.SOLVING, int(SCIP_HEURTIMING.AFTERLPNODE)
        self.SOLVED = SCIP_STAGE.SOLVED
        self.LPS_TIMING = int(SCIP_HEURTIMING.AFTERLPNODE) | int(SCIP_HEURTIMING.DURINGLPLOOP)
        # LP stream (D-011)
        self.lps_on = bool(getattr(args, "lp_stream", False)) and mode == "active" and self.send
        if self.lps_on:
            self.lps_share, self.lps_min_gap = float(args.lp_stream_share), float(args.lp_stream_min_gap)
            self.lps_q, self.lps_patience = float(args.lp_stream_q), float(args.lp_stream_patience)
            self.lps_ring: collections.deque = collections.deque(maxlen=int(args.lp_stream_ring))
        self.lps_win: collections.deque = collections.deque(maxlen=32)
        self.lps_est = 4e-7 * self.n + 2e-3          # first estimate of one read + post (s); then measured
        self.lps_init = 2 * self.lps_est              # start allowance: the first point leaves at once
        self.lps_spent = 0.0
        self.lps_last = -float("inf")
        self.lps_wait: float | None = None
        self.lps_id = 0
        self.lps_stats = dict(lp_solves=0, cutoff=0, integral=0, reads=0, early=0, quality=0, patience=0, read_s=0.0,
                              post_s=0.0, bad=0)
        self.LP_OPTIMAL = SCIP_LPSOLSTAT.OPTIMAL
        self.DIDNOTRUN, self.DIDNOTFIND, self.FOUNDSOL = (SCIP_RESULT.DIDNOTRUN, SCIP_RESULT.DIDNOTFIND,
                                                          SCIP_RESULT.FOUNDSOL)

    # --------------------------------------------------------------------------------------- recorder
    def read_x(self, sol) -> np.ndarray:
        m, ov = self.m, self.ov
        return np.fromiter((m.getSolVal(sol, v) for v in ov), dtype=np.float64, count=self.n)

    def on_bestsol(self) -> None:
        """BESTSOLFOUND: record the incumbent; never inject or post here."""
        tp = time.perf_counter()
        m = self.m
        sol = m.getBestSol()
        obj = float(m.getSolObjVal(sol))                         # NOT getPrimalbound(): it lags at this event
        x = self.read_x(sol)
        source = self.inject_source if self.injecting else "self"
        self.proc.record_incumbent(obj, x, source, stage=m.getStageName())
        self.last_inc = ((float(m.getSolTime(sol)), obj), x)
        self.dirty = True
        self.record_s += time.perf_counter() - tp
        if self.mode == "none" and self.proc.should_stop():   # --no-plugin: the only place a stop can land
            m.interruptSolve()

    # ----------------------------------------------------------------------------------------- plugin
    def on_call(self, heur, timing: int, nodeinfeasible: bool = False):
        """One plugin call: a clock read and a few counters unless a poll, a stop check or the root LP is due."""
        now = time.monotonic()
        t = now - self.proc.t0
        self.calls += 1
        self.calls_by_timing[timing] = self.calls_by_timing.get(timing, 0) + 1
        gap = t - (self.t_last if self.t_last is not None else 0.0)
        if gap > self.max_gap:
            self.max_gap = gap
        self.t_last = t
        m = self.m
        stage = m.getStage()
        if self.t_first_solving is None:
            if self.t_first is None:
                self.t_first = t
                self.proc.trace.event("callback", t=t, what="first", stage=m.getStageName(), timing=timing)
            if stage == self.SOLVING:
                self.t_first_solving = t
                self.proc.trace.event("callback", t=t, what="first_solving", timing=timing)
        if self.proc.should_stop(now):
            m.interruptSolve()
            return {"result": self.DIDNOTRUN}
        if (self.mode == "active" and self.send and not self.lp_posted and timing & self.AFTERLPNODE
                and stage == self.SOLVING and m.getDepth() == 0):
            self._post_root_lp()
        if self.lps_on and timing & self.LPS_TIMING and stage == self.SOLVING and not nodeinfeasible:
            self._lp_stream_tick(now, t)
        if now < self.next_poll:
            return {"result": self.DIDNOTRUN}
        self.next_poll = now + self.poll_s
        self._sample_bound(stage)
        if self.mode != "active":
            return {"result": self.DIDNOTRUN}
        result = self.DIDNOTRUN
        if self.recv and (stage != self.SOLVING or not m.inProbing()):
            result = self._receive(heur)
        if self.send:
            self._send(stage)
        return {"result": result}

    def _sample_bound(self, stage) -> None:
        if stage != self.SOLVING:
            return
        db = finite_or_none(self.m.getDualbound())
        if db is not None and (self.last_bound is None or abs(db - self.last_bound) > 1e-12 * max(1.0, abs(db))):
            self.last_bound = db
            self.proc.trace.event("bound", bound=db)

    # ---------------------------------------------------------------------------------------- receive
    def _best_obj(self):
        m = self.m
        return float(m.getSolObjVal(m.getBestSol())) if m.getNSols() > 0 else None

    def _receive(self, heur):
        tp = time.perf_counter()
        st = self.proc.stats
        mb = self.proc.mailbox
        st.add("polls")
        msg = mb.to_solver.poll()
        if msg is None:
            st.seconds += time.perf_counter() - tp
            return self.DIDNOTRUN
        st.add("msgs_in")
        if msg.seq > self.last_seq + 1:
            st.add("seq_superseded", msg.seq - self.last_seq - 1)      # replaced before this process polled
        self.last_seq = max(self.last_seq, msg.seq)
        st.bytes_in += int(msg.X.nbytes + msg.obj.nbytes)
        found = False
        org = msg.origin if msg.origin is not None and len(msg.origin) == msg.X.shape[0] else None
        for i in range(msg.X.shape[0]):
            cuopt = bool(org is not None and int(org[i]) & POST_CUOPT)
            found |= self._try_row(heur, msg.seq, i, msg.X[i], float(msg.obj[i]), cuopt=cuopt)
        st.seconds += time.perf_counter() - tp
        return self.FOUNDSOL if found else self.DIDNOTFIND

    def _make_sol(self, heur, x_local, nz):
        m, ov = self.m, self.ov
        s = m.createOrigSol(heur=heur)
        for j in nz:
            m.setSolVal(s, ov[j], float(x_local[j]))
        return s

    def _try_row(self, heur, seq: int, row: int, x_can: np.ndarray, obj: float, cuopt: bool = False) -> bool:
        st, m, mb = self.proc.stats, self.m, self.proc.mailbox
        st.add("rows_in")
        before = self._best_obj()
        note = dict(row=row, primal_before=before)
        if cuopt:                          # A9: an unchanged cuOpt vector the EA passed on
            note["cuopt"] = True
            st.add("rows_in_cuopt")
        if not (np.all(np.isfinite(x_can)) and np.isfinite(obj)):
            note.update(stored=False, feasible=False, dup=False, nonfinite=True, primal_after=before)
            self._verdict(seq, obj, False, note)
            st.add("rows_nonfinite")
            return False
        h = vector_hash(x_can)
        self.received.add(h)
        if h in self.tried:
            st.add("rows_dup")
            note.update(stored=False, feasible=self.tried[h], dup=True, primal_after=before)
            self._verdict(seq, obj, self.tried[h], note)
            return False
        x_local = self.proc.namemap.to_local(x_can)
        nz = np.flatnonzero(x_local)
        s = self._make_sol(heur, x_local, nz)
        obj_solver = float(m.getSolObjVal(s))
        self.injecting = True              # trySol fires BESTSOLFOUND re-entrantly: the recorder labels it "ea"
        self.inject_source = "cuopt" if cuopt else "ea"                 # (or "cuopt": A9)
        try:
            stored = bool(m.trySol(s, printreason=False, completely=False, checkbounds=True, checkintegrality=True,
                                   checklprows=True, free=True))
        finally:
            self.injecting = False
            self.inject_source = "ea"
        if stored:
            feasible = True
        else:
            s2 = self._make_sol(heur, x_local, nz)
            feasible = bool(m.checkSol(s2, printreason=False, completely=False, checkbounds=True,
                                       checkintegrality=True, checklprows=True, original=True))
            m.freeSol(s2)
        self.tried[h] = feasible
        st.add("rows_tried")
        st.add("rows_stored" if stored else ("rows_feasible_not_stored" if feasible else "rows_rejected"))
        after = self._best_obj()
        note.update(stored=stored, feasible=feasible, dup=False, primal_after=after, obj_solver=obj_solver,
                    improved=bool(after is not None and improves(after, before, self.sense)))
        self._verdict(seq, obj, feasible, note)
        return stored

    def _verdict(self, seq: int, obj: float, accepted: bool, note: dict) -> None:
        self.proc.mailbox.verdict(seq, obj, bool(accepted), note=json.dumps(note))
        self.proc.trace.event("verdict", seq=seq, obj=obj, accepted=bool(accepted), **note)

    # ------------------------------------------------------------------------------------------- send
    def final_flush(self, status: str) -> None:
        """After optimize(): post what the last tick did not (an incumbent found after it) and the final status and
        bound, so an EA that runs on after SCIP stops (optimality before the budget) starts from SCIP's last word."""
        if self.mode == "active" and self.send:
            self._send(self.m.getStage(), status=status)

    def _send(self, stage, status: str | None = None) -> None:
        tp = time.perf_counter()
        st, m, mb = self.proc.stats, self.m, self.proc.mailbox
        if self.dirty and m.getNSols() > 0:
            self.dirty = False
            sols = m.getSols()
            objs = [float(m.getSolObjVal(s)) for s in sols]
            rows, obj_rows, origin = [], [], []
            for i in band_select(objs, self.sense, self.band, self.k):
                key = (float(m.getSolTime(sols[i])), objs[i])
                if key in self.posted_keys:
                    continue
                if self.last_inc is not None and self.last_inc[0] == key:
                    x_local = self.last_inc[1]
                else:
                    x_local = self.read_x(sols[i])
                x_can = self.proc.namemap.to_canonical(x_local)
                echo = vector_hash(x_can) in self.received
                rows.append(x_can)
                obj_rows.append(objs[i])
                origin.append(ORIGIN_EA_ECHO if echo else ORIGIN_SOLVER)
                self.posted_keys.add(key)
            if rows:
                X = np.vstack(rows)
                seq = mb.to_ea.post(X, np.array(obj_rows), origin=np.array(origin, dtype=np.int64))
                st.add("posts_out")
                st.add("rows_out", len(rows))
                st.add("echo_rows_out", int(sum(o == ORIGIN_EA_ECHO for o in origin)))
                st.bytes_out += int(X.nbytes + 8 * len(obj_rows))
                self.proc.trace.event("send", seq=seq, k=len(rows),
                                      best_obj=min(obj_rows, key=lambda o: self.sense * o),
                                      echo=int(sum(o == ORIGIN_EA_ECHO for o in origin)))
        self._post_info(stage, status)
        st.seconds += time.perf_counter() - tp

    def _post_info(self, stage, status: str | None = None) -> None:
        m = self.m
        inc = self._best_obj()
        db = finite_or_none(m.getDualbound()) if stage in (self.SOLVING, self.SOLVED) else None
        status = status or m.getStageName()
        key = (inc, db, status)
        if key == self.last_info:
            return
        self.last_info = key
        nodes = int(m.getNNodes()) if stage == self.SOLVING else 0
        self.proc.mailbox.info.post(incumbent=inc, dual_bound=db, status=status, n_sols=int(m.getNSols()),
                                    nodes=nodes, solver="scip")
        self.proc.stats.add("info_posts")

    def _post_root_lp(self) -> None:
        m = self.m
        self.lp_posted = True                 # once, whatever happens below
        tp = time.perf_counter()
        try:
            if m.getLPSolstat() != self.LP_OPTIMAL:
                self.proc.trace.event("note", msg=f"root LP not optimal (solstat {m.getLPSolstat()}), not posted")
                return
            x = np.fromiter((m.getSolVal(None, v) for v in self.ov), dtype=np.float64, count=self.n)
            obj = float(self.c_local @ x + self.offset)
            x_can = self.proc.namemap.to_canonical(x)
            self.proc.mailbox.lp.post(x_can, obj)
            self.proc.stats.add("lp_posts")
            self.proc.stats.bytes_out += int(x_can.nbytes + 8)
            self.proc.trace.event("lp", obj=obj)
        finally:
            self.proc.stats.seconds += time.perf_counter() - tp

    def _lp_stream_tick(self, now: float, t: float) -> None:
        """One optimal LP solve seen by the plugin: decide whether its point is read (module docstring). Its whole
        cost — this decision included, not only the read and the post — is charged to the stream's budget."""
        tp = time.perf_counter()
        try:
            self._lp_stream_decide(now, t)
        finally:
            self.lps_spent += time.perf_counter() - tp

    def _lp_stream_decide(self, now: float, t: float) -> None:
        m = self.m
        if m.inProbing() or m.getLPSolstat() != self.LP_OPTIMAL:
            return
        st = self.lps_stats
        st["lp_solves"] += 1
        if m.getLPObjVal() >= m.getCutoffbound():        # not super-optimal: a node about to be cut off (review)
            st["cutoff"] += 1
            return
        nfrac = int(m.getNLPBranchCands())
        if nfrac == 0:
            st["integral"] += 1
            return
        self.lps_win.append(nfrac)
        if self.lps_spent + self.lps_est > self.lps_share * t + self.lps_init:
            return
        if now - self.lps_last < self.lps_min_gap:
            return
        if self.lps_wait is None:
            self.lps_wait = now
        if len(self.lps_win) < 8:
            why = "early"
        else:
            w = sorted(self.lps_win)                     # nearest-rank quantile: ~1 µs (np.quantile ~33 µs)
            if nfrac <= w[int(self.lps_q * (len(w) - 1))]:
                why = "quality"
            elif now - self.lps_wait >= self.lps_patience:
                why = "patience"
            else:
                return
        self._lp_stream_read(now, nfrac, why)

    def _lp_stream_read(self, now: float, nfrac: int, why: str) -> None:
        """Read, post, account. The time is charged by the caller (`_lp_stream_tick`, try/finally), so a post that
        raises is charged too and cannot make reads free."""
        m, st = self.m, self.lps_stats
        tp = time.perf_counter()
        self.lps_last, self.lps_wait = now, None
        x = np.fromiter((m.getSolVal(None, v) for v in self.ov), dtype=np.float64, count=self.n)
        t_read = time.perf_counter() - tp
        if not np.all(np.isfinite(x)):
            st["bad"] += 1
            return
        obj = float(self.c_local @ x + self.offset)
        self.lps_id += 1
        self.lps_ring.append((self.lps_id, self.proc.namemap.to_canonical(x), obj))
        X = np.vstack([r[1] for r in self.lps_ring])
        try:
            seq = self.proc.mailbox.lps.post(X, np.array([r[2] for r in self.lps_ring]),
                                             origin=np.array([r[0] for r in self.lps_ring], dtype=np.int64))
        finally:
            self.lps_est = 0.5 * self.lps_est + 0.5 * (time.perf_counter() - tp)
        dt = time.perf_counter() - tp
        st["reads"] += 1
        st[why] += 1
        st["read_s"] += t_read
        st["post_s"] += dt - t_read
        self.proc.stats.add("lps_posts")
        self.proc.stats.bytes_out += int(X.nbytes)
        self.proc.stats.seconds += dt
        self.proc.trace.event("lps", seq=seq, id=self.lps_id, depth=int(m.getDepth()), nodes=int(m.getNNodes()),
                              nfrac=nfrac, obj=obj, why=why, read_s=round(t_read, 6), s=round(dt, 6))

    def lp_stream_summary(self) -> dict | None:
        if not self.lps_on:
            return None
        st = dict(self.lps_stats)
        st.update(read_s=round(st["read_s"], 6), post_s=round(st["post_s"], 6), spent_s=round(self.lps_spent, 6),
                  points=self.lps_id)
        return st

    def callback_summary(self, t_end: float) -> dict:
        tail = t_end - (self.t_last if self.t_last is not None else 0.0)
        return dict(calls=self.calls, calls_by_timing={str(k): v for k, v in sorted(self.calls_by_timing.items())},
                    t_first_callback=self.t_first, t_first_solving_callback=self.t_first_solving,
                    max_gap_s=round(max(self.max_gap, tail), 6),
                    callbacks_per_s=round(self.calls / max(t_end, 1e-9), 3),
                    record_s=round(self.record_s, 6))


def _classes():
    """The PySCIPOpt plugin classes (imported inside the process so the import counts as start-up time)."""
    from pyscipopt import SCIP_EVENTTYPE, SCIP_RESULT, Eventhdlr, Heur

    class Recorder(Eventhdlr):
        def __init__(self, ex):
            super().__init__()
            self.ex = ex

        def eventinit(self):
            self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        def eventexit(self):
            self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

        def eventexec(self, event):
            self.ex.on_bestsol()

    class ExchangeHeur(Heur):
        def __init__(self, ex):
            super().__init__()
            self.ex = ex

        def heurexec(self, heurtiming, nodeinfeasible):
            try:
                return self.ex.on_call(self, int(heurtiming), bool(nodeinfeasible))
            except Exception as e:  # noqa: BLE001 — an exception inside a SCIP callback must not go unseen
                self.ex.proc.trace.event("note", msg=f"plugin error: {type(e).__name__}: {e}"[:500])
                self.ex.proc.stats.add("plugin_errors")
                return {"result": SCIP_RESULT.DIDNOTRUN}

    return Recorder, ExchangeHeur


_EXIT = {"optimal": "optimal", "timelimit": "budget", "infeasible": "infeasible", "unbounded": "unbounded",
         "inforunbd": "infeasible_or_unbounded", "memlimit": "memory_limit", "nodelimit": "node_limit"}


def body(proc):
    from pyscipopt import SCIP_HEURTIMING, Model
    import pyscipopt

    args = proc.args
    Recorder, ExchangeHeur = _classes()
    m = Model()
    m.hideOutput()
    apply_emphasis(m, args.emphasis)                        # first: an emphasis may touch any parameter
    m.setParam("parallel/maxnthreads", 1)
    m.setParam("lp/threads", 1)
    m.setParam("timing/clocktype", 2)                       # wall clock
    m.setParam("limits/memory", float(args.mem_mb))
    m.setParam("randomization/randomseedshift", int(args.seed))
    from itrea.paths import resolve
    tr0 = time.monotonic()
    m.readProblem(str(resolve(args.instance)))
    proc.timing["read_s"] = time.monotonic() - tr0
    ov = m.getVars(transformed=False)                       # cached once, before the clock
    sense = -1 if m.getObjectiveSense() == "maximize" else 1
    proc.set_names([v.name for v in ov], sense)
    mode = "none" if args.no_plugin else ("active" if (args.recv or args.send) else "inert")
    if mode == "active":
        proc.map_order()
    ex = _Exchange(proc, m, ov, mode)
    m.includeEventhdlr(Recorder(ex), "itrea_recorder", "records every incumbent")
    if mode != "none":
        mask = (SCIP_HEURTIMING.BEFOREPRESOL | SCIP_HEURTIMING.DURINGPRESOLLOOP | SCIP_HEURTIMING.BEFORENODE
                | SCIP_HEURTIMING.DURINGLPLOOP | SCIP_HEURTIMING.AFTERLPNODE)
        m.includeHeur(ExchangeHeur(ex), "itrea_xchg", "solutions from / to the I-TREA II partner", "Y",
                      priority=1_000_000, freq=1, freqofs=0, maxdepth=-1, timingmask=mask, usessubscip=False)
    proc.trace.set(versions=versions(pyscipopt=pyscipopt.__version__, scip=str(m.version())),
                   emphasis=args.emphasis or "default", plugin=mode, n=len(ov), m=int(m.getNConss()), sense=sense)
    t0 = proc.barrier()
    if proc.should_stop():
        return "stop", dict(status="not started")
    limit = max(0.0, proc.remaining())
    m.setParam("limits/time", limit)
    proc.timing.update(time_limit_s=limit, optimize_at_s=time.monotonic() - t0)
    m.optimize()
    t_end = time.monotonic() - t0
    status = m.getStatus()
    ex.final_flush(status)
    if m.getNSols() > 0:                                    # a final best the recorder did not see (retransformed)
        sol = m.getBestSol()
        fo = float(m.getSolObjVal(sol))
        if improves(fo, proc.best_obj, sense):
            proc.record_incumbent(fo, ex.read_x(sol), "self", t=t_end, late=True)
    if status == "userinterrupt":
        reason = "sigterm" if proc.sigterm else ("stop" if proc.stop_seen else "interrupted")
    else:
        reason = _EXIT.get(status, status)
    nodes, lpi = int(m.getNNodes()), int(m.getNLPIterations())
    cb = ex.callback_summary(t_end)
    if mode != "none":
        proc.trace.event("callback", t=t_end, what="summary", **cb)
    return reason, dict(status=status, wall_s=round(t_end, 6), overrun_s=round(max(0.0, t_end - args.budget), 6),
                        final_dual=finite_or_none(m.getDualbound()), nodes=nodes, lp_iterations=lpi,
                        nodes_per_s=round(nodes / max(t_end, 1e-9), 3),
                        lp_iterations_per_s=round(lpi / max(t_end, 1e-9), 3),
                        presolve_s=round(m.getPresolvingTime(), 6), n_sols_found=int(m.getNSolsFound()),
                        n_best_sols_found=int(m.getNBestSolsFound()), callbacks=cb if mode != "none" else None,
                        received_vectors=len(ex.received), posted_keys=len(ex.posted_keys), root_lp_posted=ex.lp_posted,
                        lp_stream=ex.lp_stream_summary())


def main(argv=None) -> int:
    return run_main(argv, build_parser(), validate, body, "scip", MODULE)


if __name__ == "__main__":
    raise SystemExit(main())
