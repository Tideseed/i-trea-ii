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
