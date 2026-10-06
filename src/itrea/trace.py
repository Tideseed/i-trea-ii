"""Run traces: one JSON-lines event stream per process (crash-safe) and one summary JSON at the end.

Every process of a pair run (solver, EA, cuOpt) writes its own trace in the run directory; times are seconds since the
shared start `t0` on the system-wide monotonic clock (`time.monotonic()`). Event kinds (the `kind` field):

    incumbent   a new incumbent of THIS process: {"obj", "source"} — source = "self" | "ea" | "solver" (injected)
    bound       solver dual bound update: {"bound"}
    post        EA -> solver post: {"seq", "obj", "gain"}
    inject      solver -> EA solutions received: {"seq", "k", "best_obj"}
    verdict     the solver's verdict on an EA post: {"seq", "obj", "accepted"}
    gen         EA generation record: {"gen", "best_key", "median_key", "feas_share", "n0", "n1", "n2", "inc",
                "children_per_s", "gpu_mem_mb", "feasibility_phase"}
    best        a new population best with ancestry: {"key", "obj", "feasible", "rank", "origin", "anc_rank",
                "anc_ops", "parents"}
    frontier    the diverse-solution archive (D-005): {"size", "best_key", "worst_key", "max_dist",
                "mean_dist_to_inc", "mean_pairwise", "hv"}
    note        anything else: {"msg"}

The summary holds identity (run_id, role, instance, arm, seed, budget, config + hash, versions), timing (read/start-up
seconds before the clock, wall seconds after), the final incumbent and counters. Schema: vault
`10-design/Trace format and metrics.md`.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):
        try:
            return _clean(v.item())
        except (ValueError, AttributeError):
            pass
    if hasattr(v, "tolist"):
        return _clean(v.tolist())
    return v


class Trace:
    def __init__(self, run_dir: str | os.PathLike, role: str, **identity):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.role = role
        self.t0: float | None = None
        self.summary: dict = dict(role=role, **identity)
        self.counters: dict[str, int] = {}
        self.incumbents: list[tuple[float, float, str]] = []
        self._f = open(self.dir / f"{role}.events.jsonl", "a", buffering=1)

    def start(self, t0: float) -> None:
        self.t0 = t0
        self.summary["t0"] = t0

    def now(self) -> float:
        """Seconds since the shared start t0 (time.monotonic(), system-wide)."""
        return (time.monotonic() - self.t0) if self.t0 is not None else float("nan")

    def event(self, kind: str, t: float | None = None, **data) -> None:
        t = self.now() if t is None else t
        self.counters[kind] = self.counters.get(kind, 0) + 1
        self._f.write(json.dumps(_clean(dict(t=round(t, 6), kind=kind, **data))) + "\n")

    def incumbent(self, obj: float, source: str = "self", t: float | None = None, **extra) -> None:
        t = self.now() if t is None else t
        self.incumbents.append((t, float(obj), source))
        self.event("incumbent", t=t, obj=float(obj), source=source, **extra)

    def set(self, **kv) -> None:
        self.summary.update(kv)

    def close(self, **final) -> Path:
        self.summary.update(final)
        self.summary["counters"] = self.counters
        self.summary["incumbents"] = [[round(t, 6), o, s] for t, o, s in self.incumbents]
        p = self.dir / f"{self.role}.summary.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(_clean(self.summary), indent=1))
        os.replace(tmp, p)
        self._f.close()
        return p


def read_events(path: str | os.PathLike) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    break  # a truncated last line from a killed process
    return out


# ------------------------------------------------------------------------------------------ incumbent vectors
def save_incumbents(path: str | os.PathLike, names: list[str], records: list[tuple[float, float, str, "object"]]) -> Path:
    """Store EVERY incumbent of a process (D-002: P(T) is built only from incumbents the exact checker accepts).

    records: (t since t0, user-sense objective, source, x in the process's own variable order `names`). Stored as one
    .npz without pickle: t, obj, source (unicode), names (unicode), and the vectors in CSR form (ptr, idx, val) of
    their non-zeros. Written atomically."""
    import numpy as np

    p = Path(path)
    ptr, idx, val = [0], [], []
    for _, _, _, x in records:
        x = np.asarray(x, dtype=np.float64).ravel()
        nz = np.flatnonzero(x)
        idx.append(nz.astype(np.int32))
        val.append(x[nz])
        ptr.append(ptr[-1] + nz.size)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez_compressed(
        tmp, t=np.array([r[0] for r in records], dtype=np.float64),
        obj=np.array([r[1] for r in records], dtype=np.float64),
        source=np.array([r[2] for r in records], dtype=str), names=np.array(list(names), dtype=str),
        ptr=np.array(ptr, dtype=np.int64), idx=np.concatenate(idx) if idx else np.empty(0, np.int32),
        val=np.concatenate(val) if val else np.empty(0))
    os.replace(tmp, p)
    return p


def load_incumbents(path: str | os.PathLike):
    """-> (names, list of (t, obj, source, x dense float64 in `names` order))."""
    import numpy as np

    with np.load(path) as z:
        names = [str(s) for s in z["names"]]
        n = len(names)
        out = []
        for k in range(len(z["t"])):
            x = np.zeros(n)
            a, b = z["ptr"][k], z["ptr"][k + 1]
            x[z["idx"][a:b]] = z["val"][a:b]
            out.append((float(z["t"][k]), float(z["obj"][k]), str(z["source"][k]), x))
    return names, out
