"""A short deterministic engine run on the toy models, captured column by column — the regression reference for
"`strategy.enabled: false` behaves exactly as before the self-adaptive repair strategy existed".

`tests/fixtures/engine_reference_disabled.npz` was written by `write_reference()` with the engine of commit dd31d67
(2026-10-06, before `StrategyCfg` existed), so the test compares the current code against the old one, not against
itself. Every toy runs 12 generations with stagnation immigrants, frontier re-injection and an LP point delivered at t0
(toward_lp mutation and LP-rounding immigrants are exercised); `t_first` (wall clock) is not compared.

Regenerate ONLY when a deliberate behaviour change of the default path is made, and record it in the Development log:
    OMP_NUM_THREADS=1 .venv/bin/python -c "import sys; sys.path.insert(0, 'tests'); import _engine_reference as r; r.write_reference()"
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

REF_PATH = Path(__file__).resolve().parent / "fixtures" / "engine_reference_disabled.npz"
TOYS = ("knapsack", "set_cover", "mixed")
GEN_COLS = ("id", "pa", "pb", "rank_pa", "rank_pb", "origin", "birth", "key", "V", "feasible", "is_new_best",
            "is_accepted_post")
SUMMARY_KEYS = ("generations", "children", "evaluations", "ls_moves", "ls_calls", "n_incumbents", "posts",
                "stagnation_events", "immigrants_admitted", "immigrants_dropped", "lp_immigrants", "duplicates")
GENS = 12


def _problem(name: str):
    from itrea.benchmarks import toy
    return {"knapsack": lambda: toy.knapsack(30), "set_cover": lambda: toy.set_cover(40, 25),
            "mixed": toy.mixed}[name]()


def _lp_point(p) -> np.ndarray:
    """A fractional, finite point inside the bounds (an LP-like point; its value does not matter here)."""
    lb = np.where(np.isfinite(p.lb), p.lb, -3.0)
    ub = np.where(np.isfinite(p.ub), p.ub, lb + 6.0)
    return lb + 0.37 * (ub - lb)


class LPMailbox:
    """A partner that delivers one LP point (read at t0 by `begin`) and nothing else; posts are recorded."""

    def __init__(self, lp_x: np.ndarray, obj: float):
        from itrea.exchange.mailbox import Message
        lp = [Message(X=np.asarray(lp_x, float)[None, :], obj=np.array([obj]), seq=1, t=0.0)]
        outer = self
        self.posted: list = []

        class _Q:
            def __init__(self, q):
                self.q = list(q)

            def poll(self):
                return self.q.pop(0) if self.q else None

        class _To:
            def post(_s, X, obj, origin=None):
                outer.posted.append(np.asarray(X))
                return len(outer.posted)

        self.to_ea, self.info, self.lp, self.to_solver = _Q([]), _Q([]), _Q(lp), _To()

    def stopped(self) -> bool:
        return False

    def verdicts(self) -> list:
        return []


def config(backend: str = "numpy", **over):
    from itrea.config import EAConfig
    d = {"population.size": 48, "population.offspring_per_gen": 48, "gpu_mem_gb": 1.0, "backend": backend,
         "stagnation.gens": 3, "frontier.update_every": 2, "frontier.reinject": 0.5, "seeding.lp_source": "solver"}
    d.update(over)
    return EAConfig().with_overrides(d)


def capture(name: str, backend: str = "numpy", cfg=None, gens: int = GENS) -> dict:
    """{column: array} of one short run (genealogy columns, final population, summary counters)."""
    from itrea.ea.engine import Engine
    p = _problem(name)
    lp = _lp_point(p)
    cfg = cfg if cfg is not None else config(backend)
    eng = Engine(p, cfg, seed=17, mailbox=LPMailbox(lp, float(p.objective(lp))), warmup=False)
    eng.prepare()
    eng.begin(time.monotonic(), 3600.0)
    for _ in range(gens):
        eng.step()
    out = eng.finish()
    g = eng.genealogy()
    res = {f"gen_{c}": np.asarray(g[c]) for c in GEN_COLS}
    res["pop_ids"] = np.asarray(eng.pop.ids)
    res["pop_key"] = np.asarray(eng.h_key, np.float64)
    res["pop_rank"] = np.asarray(eng.pop.rank)
    res["summary"] = np.array([float(out[k]) for k in SUMMARY_KEYS])
    res["best_key"] = np.array([out["best_key"] if out["best_key"] is not None else np.inf], np.float64)
    return res


def write_reference(path: Path = REF_PATH, backends: tuple[str, ...] = ("numpy", "cupy")) -> Path:
    """Keys `<backend>/<toy>/<column>`; both runs are deterministic (checked twice in separate processes before the
    file was written: identical digests on NumPy and on CuPy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {}
    for backend in backends:
        for name in TOYS:
            for k, v in capture(name, backend).items():
                d[f"{backend}/{name}/{k}"] = v
    np.savez_compressed(path, **d)
    return path


def load_reference(path: Path = REF_PATH) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


# ------------------------------------------------------------------------------------- solver traffic (A3 path)
# `tests/fixtures/engine_reference_traffic.npz` was written by `write_traffic_reference()` with the engine of commit
# 4db3f7c (2026-10-07, before `exchange.from_cuopt` existed), so the A3 exchange path — injected rows with an echo, an
# infeasible row and a repeated vector, info with incumbent and bound, an LP point, an LP-stream message, accept /
# reject verdicts, frontier posting, the strategy's solver/EA incumbent split — is compared against the old code under
# solver traffic (review of the three-way arm A9). This capture uses only what 4db3f7c's engine has. Regenerate ONLY
# with the engine of a commit whose exchange path is deliberately changed, and record it in the Development log:
#   git archive <commit> src | tar -x -C /tmp/ref && PYTHONPATH=/tmp/ref/src OMP_NUM_THREADS=1 .venv/bin/python -c \
#     "import sys; sys.path.insert(0, 'tests'); import _engine_reference as r; r.write_traffic_reference()"
TRAFFIC_PATH = Path(__file__).resolve().parent / "fixtures" / "engine_reference_traffic.npz"
TRAFFIC_TOYS = ("knapsack", "set_cover", "mixed")
TRAFFIC_GENS = 10
TRAFFIC_COUNTERS = ("generations", "children", "evaluations", "posts", "posts_frontier", "inject_msgs", "inject_rows",
                    "echoes", "injected", "injected_infeasible", "injected_dup", "solver_rejections", "solver_accepted",
                    "verdicts_seen", "stagnation_events", "immigrants_admitted", "immigrants_dropped", "lp_immigrants",
                    "ls_moves", "ls_calls", "duplicates", "n_incumbents", "demoted", "incumbents_reverted",
                    "rejected_copies")
_SOURCES = {"self": 0, "ea": 1, "solver": 2}       # the engine's incumbent sources on this path


class TrafficMailbox:
    """A talking solver partner in memory: queued `to_ea` / `info` / `lp` / `lps` messages (one per poll), recorded
    posts and a verdict per posted row (every third rejected), visible at the next poll."""

    def __init__(self, to_ea=(), info=(), lp=(), lps=()):
        outer = self
        self.posts: list = []
        self._verdicts: list[dict] = []
        self._n = 0

        class _Q:
            def __init__(self, q):
                self.q = list(q)

            def poll(self):
                return self.q.pop(0) if self.q else None

        class _To:
            def post(_s, X, obj, origin=None):
                X = np.asarray(X, np.float64)
                obj = np.asarray(obj, np.float64)
                org = np.zeros(len(obj), np.int64) if origin is None else np.asarray(origin, np.int64)
                outer.posts.append((X.copy(), obj.copy(), org.copy()))
                seq = len(outer.posts)
                for r in range(len(obj)):
                    outer._n += 1
                    outer._verdicts.append(dict(seq=seq, t=0.0, obj=float(obj[r]), accepted=outer._n % 3 != 0,
                                                note=f'{{"row": {r}}}'))
                return seq

        self.to_ea, self.info, self.lp, self.lps = _Q(to_ea), _Q(info), _Q(lp), _Q(lps)
        self.to_solver = _To()

    def stopped(self) -> bool:
        return False

    def verdicts(self) -> list:
        return list(self._verdicts)


def traffic_config(name: str, **over):
    """The scenario's configuration: knapsack with the strategy, the LP stream and frontier posting; mixed with frontier
    posting; set cover plain."""
    d = {"exchange.poll_s": 1e-6, "exchange.inject_band": 1.0, "log_every": 1}
    if name == "knapsack":
        d.update({"strategy.enabled": True, "lp_stream.enabled": True, "exchange.post_frontier": True,
                  "frontier.enabled": True, "frontier.update_every": 1})
    elif name == "mixed":
        d.update({"exchange.post_frontier": True, "frontier.enabled": True, "frontier.update_every": 1})
    d.update(over)
    return config("numpy", **d)


def traffic_mailbox(name: str, p, mailbox_cls=TrafficMailbox):
    """The solver's messages for one toy: rows that are worse, better, an echo, infeasible under the EA's test and
    repeated; info whose incumbent and bound improve; one LP point; for the knapsack one LP-stream message."""
    from itrea.exchange.mailbox import Message
    from itrea.solvers.common import ORIGIN_EA_ECHO, ORIGIN_SOLVER

    def msg(rows, seq, origin=None):
        X = np.vstack([np.asarray(r, np.float64) for r in rows])
        return Message(X=X, obj=np.array([float(p.objective(r)) for r in X]), seq=seq, t=0.0,
                       origin=np.full(X.shape[0], ORIGIN_SOLVER, np.int64) if origin is None else
                       np.asarray(origin, np.int64))

    lp = _lp_point(p)
    lps = ()
    if name == "knapsack":
        x_opt = _knapsack_optimum_x(p)
        on = np.flatnonzero(x_opt)
        worse, worse2 = x_opt.copy(), x_opt.copy()
        worse[on[:2]] = 0.0
        worse2[on[:3]] = 0.0
        over = np.ones(p.n)                                   # infeasible under the EA's test (overweight)
        to_ea = [msg([worse2, worse, x_opt, over], 1, [ORIGIN_SOLVER, ORIGIN_SOLVER, ORIGIN_EA_ECHO, ORIGIN_SOLVER]),
                 msg([x_opt], 2), msg([x_opt], 3)]               # the optimum arrives again: a duplicate
        info = [dict(incumbent=float(p.objective(worse2)), dual_bound=float(p.objective(x_opt)) + 5.0),
                dict(incumbent=float(p.objective(worse)), dual_bound=float(p.objective(x_opt)) + 2.0),
                dict(incumbent=float(p.objective(x_opt)), dual_bound=float(p.objective(x_opt)))]
        pts = np.vstack([lp, np.clip(lp + 0.21, p.lb, p.ub), np.clip(lp - 0.13, p.lb, p.ub)])
        lps = [Message(X=pts, obj=np.array([float(p.objective(r)) for r in pts]), seq=1, t=0.0,
                       origin=np.arange(1, 4, dtype=np.int64))]
    elif name == "set_cover":
        full = np.ones(p.n)                                   # every column: feasible, poor
        half = full.copy()
        half[::2] = 0.0                                       # possibly infeasible
        to_ea = [msg([full, half], 1), msg([full], 2)]
        info = [dict(incumbent=float(p.objective(full)), dual_bound=1.0),
                dict(incumbent=float(p.objective(full)) - 1.0, dual_bound=2.0)]
    else:
        to_ea = []
        info = [dict(incumbent=20.0, dual_bound=1.0), dict(incumbent=15.0, dual_bound=2.0)]
    return mailbox_cls(to_ea=to_ea, info=info, lp=[Message(X=lp[None, :], obj=np.array([float(p.objective(lp))]),
                                                            seq=1, t=0.0)], lps=lps)


def _knapsack_optimum_x(p) -> np.ndarray:
    """An optimal vector of toy.knapsack by DP with backtracking (as test_engine.knapsack_optimum_x)."""
    w = np.asarray(p.A.toarray()[0], int)
    v = np.asarray(p.c, float)
    W = int(p.ru[0])
    n = w.size
    best = np.zeros((n + 1, W + 1))
    for j in range(n):
        best[j + 1] = best[j]
        best[j + 1, w[j]:] = np.maximum(best[j, w[j]:], best[j, :W + 1 - w[j]] + v[j])
    x, cap = np.zeros(n), W
    for j in range(n - 1, -1, -1):
        if best[j + 1, cap] != best[j, cap]:
            x[j] = 1
            cap -= w[j]
    return x


def capture_traffic(name: str, cfg=None, mailbox_cls=TrafficMailbox, gens: int = TRAFFIC_GENS) -> dict:
    """{column: array} of one short run under solver traffic: genealogy columns, final population, the exchange
    counters, incumbent records (source codes and objectives), every post (rows, objectives, origins, rows per post),
    the external incumbent, bound, cutoff and best key; with the strategy, its per-strategy incumbent counts."""
    from itrea.ea.engine import Engine
    p = _problem(name)
    cfg = cfg if cfg is not None else traffic_config(name)
    mb = traffic_mailbox(name, p, mailbox_cls)
    eng = Engine(p, cfg, seed=23, mailbox=mb, warmup=False)
    eng.prepare()
    eng.begin(time.monotonic(), 3600.0)
    for _ in range(gens):
        eng.step()                                           # one exchange tick per generation (poll_s = 1 us)
    eng._exchange_in(force=True)                             # the verdicts on the last posts
    out = eng.finish()
    g = eng.genealogy()
    res = {f"gen_{c}": np.asarray(g[c]) for c in GEN_COLS}
    res["pop_ids"] = np.asarray(eng.pop.ids)
    res["pop_key"] = np.asarray(eng.h_key, np.float64)
    res["pop_rank"] = np.asarray(eng.pop.rank)
    res["counters"] = np.array([float(out[k]) for k in TRAFFIC_COUNTERS])
    res["rec_src"] = np.array([_SOURCES[r[2]] for r in eng.records], np.int64)
    res["rec_obj"] = np.array([r[1] for r in eng.records], np.float64)
    if mb.posts:
        res["post_X"] = np.vstack([X for X, _, _ in mb.posts])
        res["post_obj"] = np.concatenate([o for _, o, _ in mb.posts])
        res["post_origin"] = np.concatenate([org for _, _, org in mb.posts])
    else:
        res["post_X"], res["post_obj"], res["post_origin"] = np.zeros((0, p.n)), np.zeros(0), np.zeros(0, np.int64)
    res["post_k"] = np.array([len(o) for _, o, _ in mb.posts], np.int64)
    res["keys"] = np.array([eng.inc_external, eng.dual_key, np.nan if eng.cutoff is None else eng.cutoff,
                            out["best_key"] if out["best_key"] is not None else np.inf], np.float64)
    if cfg.strategy.enabled:
        for k in ("incumbents", "incumbents_solver", "best_events", "best_events_solver"):
            res[f"sc_{k}"] = np.asarray(eng.sc[k])
    if cfg.lp_stream.enabled:
        res["lps_counters"] = np.array([float(out[k]) for k in ("lps_msgs", "lps_points", "lps_used", "lps_rows",
                                                                 "lps_dropped", "lps_missed")])
    return res


def write_traffic_reference(path: Path = TRAFFIC_PATH) -> Path:
    """Keys `<toy>/<column>`; NumPy only (the exchange path is host-side)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {f"{name}/{k}": v for name in TRAFFIC_TOYS for k, v in capture_traffic(name).items()}
    np.savez_compressed(path, **d)
    return path


def load_traffic_reference(path: Path = TRAFFIC_PATH) -> dict:
    return load_reference(path)
