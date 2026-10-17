"""HiGHS solver process: `python -m itrea.solvers.highs_run` with the same arguments as `scip_run` (vault
`10-design/Process interface.md`). highspy 1.15.1, `threads = 1`, `random_seed = seed`, `h.run()` (not `solve()`,
which adds a thread). HiGHS's own defaults otherwise (as-shipped; `mip_rel_gap` 1e-4 is recorded in the summary).

The HiGHS callback API (verified in this venv, 2026-10-06; critique, systems lens, "HiGHS injection API"):
- classes are `highspy.cb.HighsCallbackInput` / `HighsCallbackOutput`; a subscriber gets an event `e` with
  `e.data_out` (mip_primal_bound, mip_dual_bound, mip_solution, objective_function_value, running_time, …, all in the
  user's objective sense) and `e.data_in` (setSolution, user_interrupt, …);
- `cbMipSolution` fires for EVERY feasible solution HiGHS finds or accepts (also non-improving ones, often twice), and
  `mip_solution` is the full vector in the original column order; `cbMipImprovingSolution` misses accepted user
  solutions, so incumbents are recorded from `cbMipSolution` filtered by improvement;
- injection: `h.cbMipUserSolution.subscribe(fn)`, in fn `e.data_in.setSolution(x)` with a DENSE float64 vector of
  length num_col in HiGHS column order (the sparse overload means a partial solution). One solution per call; a
  message with several rows is queued and injected one per call, best first.
- `setSolution` returns kOk whatever happens next. A feasible improving solution comes straight back through
  `cbMipSolution` with its objective; a solution outside HiGHS's ABSOLUTE `mip_feasibility_tolerance` (1e-6) is
  silently dropped (a 5e-4 violation on a row with rhs 1e4 is SCIP-feasible and HiGHS-infeasible), and so is a
  feasible solution that does not improve HiGHS's incumbent (measured: no cbMipSolution, no change).
  Verdict, therefore: a row that does not improve HiGHS's incumbent at injection time is NOT injected (HiGHS would
  give no signal): verdict accepted=True with outcome "not_improving" and tried=false — never "rejected", because the
  EA engine marks an individual infeasible on accepted=false. An injected row is "accepted" when a cbMipSolution
  reports its objective (within 1e-9·max(1,|obj|)) or the next user-solution call's primal bound is at least as good
  (outcome "primal_bound"), else "dropped" (accepted=false). A row still pending when `run()` returns is resolved
  against the final primal bound. Do not call `repairSolution` in timed arms (it spends solver time for the EA).
- a second `Highs` object that `run()`s with default threads in the same process initialises HiGHS's GLOBAL scheduler
  with all cores, after which `threads = 1` fails (`run()` returns kError, model status kNotset — measured). Every
  `Highs` that runs in this process sets threads = 1.
- HiGHS has no memory limit option: --mem-mb is recorded, the runner's MemoryMax is the cap. HiGHS exposes no
  solution pool and no root LP channel: "very good" solutions come from a local pool of the cbMipSolution vectors, and
  the EA uses its own on-clock LP.

Modes: as in scip_run — the plugin (the user-solution callback) is installed and inert without --recv/--send or
with --inert, and absent with --no-plugin (calibration only). The incumbent recorder (`cbMipSolution`) and the
control-plane interrupt callback are installed in every mode: the stop file / SIGTERM are checked at most once per
second and the interrupt lands at the next `cbMipInterrupt` (`e.data_in.user_interrupt = True`).

Adapted in part from berkorbay/optopt@2a3348c0a102, MIT (`src/optopt/solvers/highs.py`).
"""
from __future__ import annotations

import json
import time

import numpy as np

from itrea.solvers.common import (ORIGIN_EA_ECHO, ORIGIN_SOLVER, BadArguments, band_select, base_parser,
                                  finite_or_none, improves, run_main, validate_common, vector_hash, versions)

MODULE = "itrea.solvers.highs_run"
POOL_FACTOR = 4                      # local pool keeps POOL_FACTOR * inject_k (>= 32) best distinct solutions


def build_parser():
    return base_parser(f"python -m {MODULE}", "HiGHS solver process with the I-TREA II exchange callbacks", "solver")


def validate(args) -> None:
    validate_common(args)
    if args.seed < 0:
        raise BadArguments("--seed must be >= 0 (HiGHS random_seed)")


def _tol(obj: float) -> float:
    return 1e-9 * max(1.0, abs(obj))


class _HighsExchange:
    def __init__(self, proc, n: int, mode: str):
        args = proc.args
        self.proc, self.n, self.mode = proc, n, mode
        self.sense = proc.sense
        self.recv, self.send = bool(args.recv), bool(args.send)
        self.poll_s, self.band, self.k = args.poll_s, args.inject_band, args.inject_k
        self.pool_max = max(32, POOL_FACTOR * self.k)
        self.pool: list[dict] = []                 # {obj, x (local), hash, posted}
        self.pool_hashes: set[bytes] = set()
        self.dirty = False
        self.next_poll = 0.0
        self.queue: list[dict] = []                # EA rows waiting for a user-solution call
        self.pending: list[dict] = []              # injected, verdict not yet known
        self.received: set[bytes] = set()
        self.tried: dict[bytes, bool] = {}
        self.last_seq = 0
        self.last_info = None
        self.last_bound = None
        self.last_pb = None
        self.user_calls = 0
        self.interrupt_calls = 0
        self.solution_calls = 0
        self.t_first = None
        self.t_last = None
        self.max_gap = 0.0
        self.record_s = 0.0
        self.namemap = None                        # set in active mode (canonical <-> HiGHS column order)

    # ------------------------------------------------------------------------------ cbMipSolution
    def on_solution(self, e) -> None:
        tp = time.perf_counter()
        self.solution_calls += 1
        d = e.data_out
        obj = float(d.objective_function_value)
        x = np.array(d.mip_solution, dtype=np.float64, copy=True)
        if x.size != self.n or not np.isfinite(obj):
            return
        h_can = vector_hash(self.namemap.to_canonical(x)) if self.namemap is not None else None
        from_ea = h_can is not None and h_can in self.received
        for p in list(self.pending):              # an injected row comes straight back here when HiGHS accepts it
            if p["hash"] == h_can or abs(obj - p["obj"]) <= _tol(p["obj"]):
                self._resolve(p, True, "solution_callback", after=obj)
                from_ea = True
        if improves(obj, self.proc.best_obj, self.sense):
            self.proc.record_incumbent(obj, x, "ea" if from_ea else "self")
            self.dirty = True
        if self.mode == "active" and self.send and h_can is not None and h_can not in self.pool_hashes:
            self._pool_add(obj, x, h_can)
        self.record_s += time.perf_counter() - tp

    def _pool_add(self, obj, x, h) -> None:
        self.pool.append(dict(obj=obj, x=x, hash=h, posted=False))
        self.pool_hashes.add(h)
        if len(self.pool) > self.pool_max:
            self.pool.sort(key=lambda r: self.sense * r["obj"])
            for r in self.pool[self.pool_max:]:
                self.pool_hashes.discard(r["hash"])
            del self.pool[self.pool_max:]

    # ------------------------------------------------------------------------- cbMipInterrupt
    def on_interrupt(self, e) -> None:
        self.interrupt_calls += 1
        if self.proc.should_stop():
            e.data_in.user_interrupt = True

    # ---------------------------------------------------------------------- cbMipUserSolution
    def on_user(self, e) -> None:
        now = time.monotonic()
        t = now - self.proc.t0
        self.user_calls += 1
        gap = t - (self.t_last if self.t_last is not None else 0.0)
        self.max_gap = max(self.max_gap, gap)
        self.t_last = t
        if self.t_first is None:
            self.t_first = t
            self.proc.trace.event("callback", t=t, what="first")
        d = e.data_out
        pb = finite_or_none(d.mip_primal_bound)
        self.last_pb = pb
        if self.pending:                           # verdicts of rows injected at an earlier call
            tp = time.perf_counter()
            for p in list(self.pending):
                ok = pb is not None and self.sense * (pb - p["obj"]) <= _tol(p["obj"])
                self._resolve(p, ok, "primal_bound" if ok else "dropped", after=pb)
            self.proc.stats.seconds += time.perf_counter() - tp
        if self.proc.should_stop(now):
            return                                 # the interrupt callback ends the solve
        if now >= self.next_poll:
            self.next_poll = now + self.poll_s
            db = finite_or_none(d.mip_dual_bound)
            if db is not None and (self.last_bound is None or abs(db - self.last_bound) > 1e-12 * max(1.0, abs(db))):
                self.last_bound = db
                self.proc.trace.event("bound", t=t, bound=db)
            if self.mode == "active":
                tp = time.perf_counter()
                if self.recv:
                    self._receive()
                if self.send:
                    self._send(pb, db)
                self.proc.stats.seconds += time.perf_counter() - tp
        if self.mode == "active" and self.recv and self.queue:
            tp = time.perf_counter()
            self._inject_one(e, pb)
            self.proc.stats.seconds += time.perf_counter() - tp

    def _receive(self) -> None:
        st = self.proc.stats
        st.add("polls")
        msg = self.proc.mailbox.to_solver.poll()
        if msg is None:
            return
        st.add("msgs_in")
        if msg.seq > self.last_seq + 1:
            st.add("seq_superseded", msg.seq - self.last_seq - 1)
        self.last_seq = max(self.last_seq, msg.seq)
        st.bytes_in += int(msg.X.nbytes + msg.obj.nbytes)
        for i in range(msg.X.shape[0]):
            st.add("rows_in")
            x_can, obj = msg.X[i], float(msg.obj[i])
            base = dict(seq=msg.seq, row=i, obj=obj)
            if not (np.all(np.isfinite(x_can)) and np.isfinite(obj)):
                st.add("rows_nonfinite")
                self._verdict(base, False, dict(outcome="nonfinite", tried=False, dup=False))
                continue
            h = vector_hash(x_can)
            self.received.add(h)
            if h in self.tried:
                st.add("rows_dup")
                self._verdict(base, self.tried[h], dict(outcome="duplicate", tried=False, dup=True))
                continue
            if any(q["hash"] == h for q in self.queue) or any(p["hash"] == h for p in self.pending):
                st.add("rows_dup")
                self._verdict(base, True, dict(outcome="duplicate_in_flight", tried=False, dup=True))
                continue
            self.queue.append(dict(base, hash=h, x=self.namemap.to_local(x_can)))
        self.queue.sort(key=lambda q: self.sense * q["obj"])

    def _inject_one(self, e, pb) -> None:
        st = self.proc.stats
        while self.queue:
            q = self.queue.pop(0)
            if pb is not None and not improves(q["obj"], pb, self.sense):
                st.add("rows_not_improving")
                self.tried[q["hash"]] = True
                self._verdict(q, True, dict(outcome="not_improving", tried=False, dup=False, primal_before=pb))
                continue
            e.data_in.setSolution(np.ascontiguousarray(q["x"], dtype=np.float64))
            st.add("rows_tried")
            q["primal_before"] = pb
            self.pending.append(q)
            return

    def _resolve(self, p, accepted: bool, outcome: str, after) -> None:
        self.pending.remove(p)
        self.tried[p["hash"]] = bool(accepted)
        self.proc.stats.add("rows_accepted" if accepted else "rows_dropped")
        self._verdict(p, accepted, dict(outcome=outcome, tried=True, dup=False, primal_before=p.get("primal_before"),
                                        primal_after=after))

    def _verdict(self, p, accepted: bool, note: dict) -> None:
        # `stored` as in the SCIP verdicts (trace schema): for HiGHS, "stored" = HiGHS took it as a new incumbent
        note = dict(note, row=p["row"], stored=bool(accepted and note.get("tried")))
        self.proc.mailbox.verdict(p["seq"], p["obj"], bool(accepted), note=json.dumps(note))
        self.proc.trace.event("verdict", seq=p["seq"], obj=p["obj"], accepted=bool(accepted), **note)

    def final_flush(self, pb, db, status: str) -> None:
        """After run(): post what the last tick did not and the final status/bound (the EA may run on alone)."""
        if self.mode == "active" and self.send:
            tp = time.perf_counter()
            self._send(pb, db, status=status)
            self.proc.stats.seconds += time.perf_counter() - tp

    def _send(self, pb, db, status: str = "solving") -> None:
        st, mb = self.proc.stats, self.proc.mailbox
        if self.dirty and self.pool:
            self.dirty = False
            objs = [r["obj"] for r in self.pool]
            rows = [self.pool[i] for i in band_select(objs, self.sense, self.band, self.k)]
            rows = [r for r in rows if not r["posted"]]
            if rows:
                X = np.vstack([self.namemap.to_canonical(r["x"]) for r in rows])
                origin = np.array([ORIGIN_EA_ECHO if r["hash"] in self.received else ORIGIN_SOLVER for r in rows],
                                  dtype=np.int64)
                obj = np.array([r["obj"] for r in rows])
                seq = mb.to_ea.post(X, obj, origin=origin)
                for r in rows:
                    r["posted"] = True
                st.add("posts_out")
                st.add("rows_out", len(rows))
                st.add("echo_rows_out", int((origin == ORIGIN_EA_ECHO).sum()))
                st.bytes_out += int(X.nbytes + obj.nbytes)
                self.proc.trace.event("send", seq=seq, k=len(rows), best_obj=float(obj[0]),
                                      echo=int((origin == ORIGIN_EA_ECHO).sum()))
        key = (self.proc.best_obj, db, status)
        if key != self.last_info:
            self.last_info = key
            mb.info.post(incumbent=self.proc.best_obj, dual_bound=db, status=status, primal_bound=pb,
                         solver="highs")
            st.add("info_posts")

    def finish_pending(self, final_pb) -> None:
        for p in list(self.pending):
            ok = final_pb is not None and self.sense * (final_pb - p["obj"]) <= _tol(p["obj"])
            self._resolve(p, ok, "final_primal_bound" if ok else "dropped_final", after=final_pb)
        for q in self.queue:                       # received, never injected before the end
            self.proc.stats.add("rows_pending_at_end")
            self._verdict(q, True, dict(outcome="pending_at_end", tried=False, dup=False))
        self.queue.clear()

    def callback_summary(self, t_end: float) -> dict:
        tail = t_end - (self.t_last if self.t_last is not None else 0.0)
        return dict(user_calls=self.user_calls, interrupt_calls=self.interrupt_calls,
                    solution_calls=self.solution_calls, t_first_callback=self.t_first,
                    max_gap_s=round(max(self.max_gap, tail), 6),
                    callbacks_per_s=round(self.user_calls / max(t_end, 1e-9), 3), record_s=round(self.record_s, 6))


_EXIT = {"Optimal": "optimal", "Time limit reached": "budget", "Infeasible": "infeasible", "Unbounded": "unbounded",
         "Primal infeasible or unbounded": "infeasible_or_unbounded", "Memory limit reached": "memory_limit"}


def body(proc):
    import highspy

    args = proc.args
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("random_seed", int(args.seed))
    from itrea.paths import resolve
    tr0 = time.monotonic()
    st = h.readModel(str(resolve(args.instance)))
    proc.timing["read_s"] = time.monotonic() - tr0
    if st == highspy.HighsStatus.kError:
        raise IOError(f"HiGHS could not read {args.instance}")
    lp = h.getLp()
    n = int(lp.num_col_)
    names = list(lp.col_names_) if len(lp.col_names_) == n else [f"x{j}" for j in range(n)]
    sense = -1 if lp.sense_ == highspy.ObjSense.kMaximize else 1
    proc.set_names(names, sense)
    mode = "none" if args.no_plugin else ("active" if (args.recv or args.send) else "inert")
    ex = _HighsExchange(proc, n, mode)
    if mode == "active":
        proc.map_order()
        ex.namemap = proc.namemap

    def guard(fn):
        def wrapped(e):
            try:
                fn(e)
            except Exception as err:  # noqa: BLE001 — never let an exception escape into HiGHS unseen
                proc.trace.event("note", msg=f"callback error: {type(err).__name__}: {err}"[:500])
                proc.stats.add("callback_errors")
        return wrapped

    h.cbMipSolution.subscribe(guard(ex.on_solution))       # the incumbent recorder, every mode
    h.cbMipInterrupt.subscribe(guard(ex.on_interrupt))     # stop file / SIGTERM (control plane), every mode
    if mode != "none":
        h.cbMipUserSolution.subscribe(guard(ex.on_user))
    proc.trace.set(versions=versions(highspy=h.version()), plugin=mode, n=n, m=int(lp.num_row_), sense=sense,
                   highs_options={k: h.getOptionValue(k)[1] for k in ("mip_rel_gap", "mip_feasibility_tolerance",
                                                                      "mip_heuristic_effort", "presolve")},
                   mem_mb_note="HiGHS has no memory limit option; the runner's MemoryMax is the cap")
    t0 = proc.barrier()
    if proc.should_stop():
        return "stop", dict(status="not started")
    limit = max(0.0, proc.remaining())
    h.setOptionValue("time_limit", float(limit))
    proc.timing.update(time_limit_s=limit, optimize_at_s=time.monotonic() - t0)
    rs = h.run()
    t_end = time.monotonic() - t0
    if rs == highspy.HighsStatus.kError:
        raise RuntimeError("HiGHS run() returned kError (global scheduler already initialised with other threads?)")
    ms = h.getModelStatus()
    status = h.modelStatusToString(ms)
    info = h.getInfo()
    final_pb = finite_or_none(info.objective_function_value) if info.primal_solution_status == 2 else None
    if final_pb is not None and improves(final_pb, proc.best_obj, sense):
        proc.record_incumbent(final_pb, np.array(h.getSolution().col_value, dtype=np.float64), "self", t=t_end,
                              late=True)
    if mode == "active":
        ex.finish_pending(final_pb)
        ex.final_flush(final_pb, finite_or_none(info.mip_dual_bound), status)
    if ms == highspy.HighsModelStatus.kInterrupt:
        reason = "sigterm" if proc.sigterm else ("stop" if proc.stop_seen else "interrupted")
    else:
        reason = _EXIT.get(status, status)
    nodes = int(info.mip_node_count)
    lpi = int(info.simplex_iteration_count)
    cb = ex.callback_summary(t_end)
    if mode != "none":
        proc.trace.event("callback", t=t_end, what="summary", **cb)
    return reason, dict(status=status, wall_s=round(t_end, 6), overrun_s=round(max(0.0, t_end - args.budget), 6),
                        final_dual=finite_or_none(info.mip_dual_bound), nodes=nodes, lp_iterations=lpi,
                        nodes_per_s=round(nodes / max(t_end, 1e-9), 3),
                        lp_iterations_per_s=round(lpi / max(t_end, 1e-9), 3),
                        mip_gap=finite_or_none(info.mip_gap), callbacks=cb if mode != "none" else None,
                        received_vectors=len(ex.received), pool_size=len(ex.pool))


def main(argv=None) -> int:
    return run_main(argv, build_parser(), validate, body, "highs", MODULE)


if __name__ == "__main__":
    raise SystemExit(main())
