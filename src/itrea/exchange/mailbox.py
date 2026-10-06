"""Solution exchange between the solver process and the EA process: a mailbox directory (on /dev/shm).

Carried over from optopt's `xchg.py` (2026-09-24), extended to batches, scalar solver info, a start barrier and a
canonical variable order. Each message is one file replaced atomically (`os.replace`), so a reader never sees a
half-written one; a poll is one `stat`. "Latest message wins": every message is a full snapshot (the solver's current
best and very good solutions; the EA's current best), so an overwritten message loses nothing that matters.

Files in a run's mailbox directory:
    run_id             written by the runner when it creates the directory (processes check it)
    order.json         canonical variable names (written by the runner before either process starts)
    to_solver.npz      EA -> solver: X (k, n) float64 canonical order, obj (k,) user sense, seq, t, origin bits
    to_ea.npz          solver -> EA: X (k, n), obj (k,), seq, t
    to_ea_info.json    solver -> EA: {"seq", "t", "incumbent": obj|null, "dual_bound": obj|null, "status"}
    to_ea_lp.npz       solver -> EA, once: x (n,) root LP solution, obj
    ready.<role>       a process has read the instance and is waiting at the barrier
    start              the runner releases both: {"t0": monotonic seconds, "epoch"}
    stop               written when the pair must end (budget, failure); both sides stop promptly
    verdicts.jsonl     solver appends one line per EA post it tried: {"seq", "t", "obj", "accepted"}; readers take
                       complete (newline-terminated) lines only, since the append is not atomic for a reader

Times: `t0` and every event use `time.monotonic()` (CLOCK_MONOTONIC is system-wide on Linux, so all processes of a
run share it and NTP steps cannot shift events); messages also carry the epoch time for humans.

Lifecycle: the runner alone creates the directory (it must not exist: a reused directory with a stale `start` file
would release a process at once on an old t0) and writes `run_id`; the processes open it with `create=False`, check the
run id, and refuse to report ready if `start` already exists.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


@dataclass
class Message:
    X: np.ndarray        # (k, n) canonical order
    obj: np.ndarray      # (k,) user-sense objectives
    seq: int
    t: float             # monotonic seconds when posted
    origin: np.ndarray | None = None


class Channel:
    """One direction of the exchange: a single .npz file that the writer replaces and the reader polls."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.seq = 0
        self._seen: tuple[int, int] | None = None   # (inode, mtime_ns) of the last message read
        self.last_seq = 0

    def post(self, X: np.ndarray, obj, origin=None) -> int:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        obj = np.atleast_1d(np.asarray(obj, dtype=np.float64))
        if X.shape[0] != obj.shape[0]:
            raise ValueError("post: X rows and obj length differ")
        self.seq += 1
        arrays = dict(X=X, obj=obj, seq=np.int64(self.seq), t=np.float64(time.monotonic()),
                      epoch=np.float64(time.time()))
        if origin is not None:
            arrays["origin"] = np.atleast_1d(np.asarray(origin, dtype=np.int64))
        _atomic_savez(self.path, **arrays)
        return self.seq

    def poll(self) -> Message | None:
        """The newest message if it is new (by file identity AND sequence number), else None. The signature is taken
        from the OPEN file descriptor, so a replace between stat and read cannot re-deliver a message."""
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return None
        if (st.st_ino, st.st_mtime_ns) == self._seen:
            return None
        try:
            with open(self.path, "rb") as fh:
                fst = os.fstat(fh.fileno())
                with np.load(fh) as z:
                    msg = Message(X=z["X"].copy(), obj=z["obj"].copy(), seq=int(z["seq"]), t=float(z["t"]),
                                  origin=z["origin"].copy() if "origin" in z.files else None)
        except (OSError, ValueError, KeyError, EOFError):
            return None          # replaced while reading — take it at the next poll
        self._seen = (fst.st_ino, fst.st_mtime_ns)
        if msg.seq <= self.last_seq:
            return None
        self.last_seq = msg.seq
        return msg


class InfoChannel:
    """Scalar solver info as JSON (incumbent, dual bound, status)."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.seq = 0
        self._seen: tuple[int, int] | None = None

    def post(self, **info) -> None:
        self.seq += 1
        info = dict(info, seq=self.seq, t=time.monotonic(), epoch=time.time())
        _atomic_write_bytes(self.path, json.dumps(info).encode())

    def poll(self) -> dict | None:
        try:
            with open(self.path, "rb") as fh:
                fst = os.fstat(fh.fileno())
                sig = (fst.st_ino, fst.st_mtime_ns)
                if sig == self._seen:
                    return None
                d = json.loads(fh.read())
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return None
        self._seen = sig
        if d.get("seq", 0) <= getattr(self, "last_seq", 0):
            return None
        self.last_seq = d.get("seq", 0)
        return d


class Mailbox:
    """All channels of one pair run, seen from one role ("solver" or "ea")."""

    def __init__(self, directory: str | os.PathLike, role: str, create: bool = False, run_id: str | None = None):
        """create=True (runner only): make a NEW directory (error if it exists) and write `run_id`.
        create=False (processes): the directory must exist; if run_id is given it must match."""
        self.dir = Path(directory)
        self.role = role
        if create:
            self.dir.mkdir(parents=True, exist_ok=False)
            _atomic_write_bytes(self.dir / "run_id", (run_id or self.dir.name).encode())
        else:
            if not self.dir.is_dir():
                raise FileNotFoundError(f"mailbox {self.dir} does not exist (the runner creates it)")
            if run_id is not None and (self.dir / "run_id").read_text() != run_id:
                raise RuntimeError(f"mailbox {self.dir} belongs to another run")
        self.to_solver = Channel(self.dir / "to_solver.npz")
        self.to_ea = Channel(self.dir / "to_ea.npz")
        self.info = InfoChannel(self.dir / "to_ea_info.json")
        self.lp = Channel(self.dir / "to_ea_lp.npz")

    # ------------------------------------------------------------ order
    def write_order(self, names) -> None:
        names = [str(x) for x in names]
        _atomic_write_bytes(self.dir / "order.json", json.dumps(names).encode())

    def read_order(self) -> list[str]:
        return json.loads((self.dir / "order.json").read_text())

    # ---------------------------------------------------------- barrier
    def ready(self) -> None:
        if (self.dir / "start").exists():
            raise RuntimeError("stale start barrier: refusing to report ready")
        _atomic_write_bytes(self.dir / f"ready.{self.role}", str(time.monotonic()).encode())

    def all_ready(self, roles=("solver", "ea")) -> bool:
        return all((self.dir / f"ready.{r}").exists() for r in roles)

    def release(self, t0: float | None = None) -> float:
        """t0 in time.monotonic() seconds (system-wide clock)."""
        t0 = time.monotonic() if t0 is None else t0
        _atomic_write_bytes(self.dir / "start", json.dumps({"t0": t0, "epoch": time.time()}).encode())
        return t0

    def wait_start(self, timeout: float = 600.0, poll: float = 0.005) -> float:
        """Block until the runner releases the barrier; returns t0 (monotonic). Raises TimeoutError."""
        p = self.dir / "start"
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if p.exists():
                try:
                    return float(json.loads(p.read_text())["t0"])
                except (OSError, ValueError, KeyError):
                    pass
            if self.stopped():
                raise RuntimeError("stopped before start")
            time.sleep(poll)
        raise TimeoutError("start barrier not released")

    # ------------------------------------------------------------- stop
    def stop(self, reason: str = "") -> None:
        if not (self.dir / "stop").exists():
            _atomic_write_bytes(self.dir / "stop", json.dumps({"t": time.monotonic(), "by": self.role,
                                                               "reason": reason}).encode())

    def stopped(self) -> bool:
        return (self.dir / "stop").exists()

    # --------------------------------------------------------- verdicts
    def verdict(self, seq: int, obj: float, accepted: bool, note: str = "") -> None:
        with open(self.dir / "verdicts.jsonl", "a") as f:
            f.write(json.dumps({"seq": seq, "t": time.monotonic(), "obj": obj, "accepted": accepted,
                                "note": note}) + "\n")

    def verdicts(self) -> list[dict]:
        """Every verdict line the solver has COMPLETED so far, in file order.

        The solver appends while the EA polls, so the last line can be visible half-written. Only newline-terminated
        lines count, and parsing stops at the first line that does not parse (as `trace.read_events` does). Each result
        is therefore a prefix of every later one, so a caller that slices by the previous length (`Engine._verdicts`)
        picks the rest up at its next poll. Bytes, not text: a partial tail cannot fail the decode of the whole file."""
        p = self.dir / "verdicts.jsonl"
        try:
            data = p.read_bytes()
        except FileNotFoundError:
            return []
        out: list[dict] = []
        for line in data.split(b"\n")[:-1]:   # the last element is b"" or an unterminated (partial) line
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                break
        return out


class NameMap:
    """Permutation between a side's own variable order and the canonical order (by variable name)."""

    def __init__(self, canonical: list[str], local: list[str]):
        pos = {nm: j for j, nm in enumerate(canonical)}
        missing = [nm for nm in local if nm not in pos]
        if missing or len(local) != len(canonical):
            raise ValueError(f"variable sets differ (missing {missing[:5]}…, {len(local)} vs {len(canonical)})")
        self.local_to_can = np.array([pos[nm] for nm in local], dtype=np.int64)  # local j -> canonical index

    def to_canonical(self, x_local: np.ndarray) -> np.ndarray:
        x_local = np.asarray(x_local, dtype=np.float64)
        out = np.empty_like(x_local)
        out[..., self.local_to_can] = x_local
        return out

    def to_local(self, x_can: np.ndarray) -> np.ndarray:
        return np.asarray(x_can, dtype=np.float64)[..., self.local_to_can]
