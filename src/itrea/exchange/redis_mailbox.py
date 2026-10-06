"""Redis transport for the exchange — the "common platform" (Berk, 2026-10-05: "for solution sharing one needs a common
platform. perhaps redis is good for that?"). Same API as the file `Mailbox`, plus a shared solution pool.

Server: a dedicated container `itrea-redis` (image redis:7-alpine already on the host), loopback port 6390, no
persistence, `maxmemory 1536mb noeviction`, 2 GB container cap, pinned to an A725 core (never a timed X925 core).
Never the other Redis instances on the host (they belong to other services), never [other service]'s (read-only, remote).

Keys of one run (all under `itrea:<run_id>:`, deleted by the runner at the end; 1-day expiry as a backstop):
    order                       canonical variable names (JSON list)
    ch:<name>                   stream per channel (to_solver, to_ea, lp): fields x (encoded vector), obj, seq, t,
                                origin; MAXLEN ~64 — history of every exchange, the latest entry is the "message"
    info                        JSON string (solver incumbent, dual bound, status, seq, t)
    pool                        sorted set: member = solution id, score = key (lower is better) — the shared pool
    pool:x                      hash: solution id -> encoded vector;  pool:meta  hash: id -> JSON (obj, source, t)
    ready:<role>, start, stop   barrier and stop flags
    verdicts                    stream of the solver's verdicts on EA posts

Vectors are encoded sparsely when that is smaller (int32 indices + float64 values of the non-zeros), else dense.
Polling costs one round trip (~50 µs on loopback, measured 2026-10-05); callers must throttle polls by time, not
call them at every solver node.
"""
from __future__ import annotations

import json
import time
from urllib.parse import parse_qs, urlparse

import numpy as np

from itrea.exchange.mailbox import Message

DEFAULT_URL = "redis://127.0.0.1:6390/0"
MAXLEN = 64
EXPIRE_S = 86400


def encode(x: np.ndarray) -> bytes:
    """b'D' + float64 dense, or b'S' + n (int64) + k (int64) + idx int32[k] + val float64[k] — whichever is smaller."""
    x = np.ascontiguousarray(x, dtype=np.float64).ravel()
    nz = np.flatnonzero(x)
    if nz.size * 12 + 16 < x.size * 8:
        return b"S" + np.array([x.size, nz.size], dtype=np.int64).tobytes() + nz.astype(np.int32).tobytes() + \
            x[nz].tobytes()
    return b"D" + x.tobytes()


def decode(b: bytes) -> np.ndarray:
    if b[:1] == b"D":
        return np.frombuffer(b, dtype=np.float64, offset=1).copy()
    n, k = np.frombuffer(b, dtype=np.int64, count=2, offset=1)
    idx = np.frombuffer(b, dtype=np.int32, count=int(k), offset=17)
    val = np.frombuffer(b, dtype=np.float64, count=int(k), offset=17 + 4 * int(k))
    x = np.zeros(int(n))
    x[idx] = val
    return x


class _Stream:
    """One channel: XADD to post, newest entry to poll (latest message wins, as in the file mailbox)."""

    def __init__(self, r, key: str):
        self.r, self.key = r, key
        self.seq = 0
        self._last_id: bytes | None = None

    def post(self, X: np.ndarray, obj, origin=None) -> int:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        obj = np.atleast_1d(np.asarray(obj, dtype=np.float64))
        if X.shape[0] != obj.shape[0]:
            raise ValueError("post: X rows and obj length differ")
        self.seq += 1
        fields = {"k": X.shape[0], "obj": obj.tobytes(), "seq": self.seq, "t": repr(time.monotonic()),
                  "epoch": repr(time.time())}
        for i in range(X.shape[0]):
            fields[f"x{i}"] = encode(X[i])
        if origin is not None:
            fields["origin"] = np.atleast_1d(np.asarray(origin, dtype=np.int64)).tobytes()
        pipe = self.r.pipeline()
        pipe.xadd(self.key, fields, maxlen=MAXLEN, approximate=True)
        pipe.expire(self.key, EXPIRE_S)
        pipe.execute()
        return self.seq

    def poll(self) -> Message | None:
        got = self.r.xrevrange(self.key, count=1)
        if not got:
            return None
        sid, f = got[0]
        if sid == self._last_id:
            return None
        self._last_id = sid
        if int(f[b"seq"]) <= getattr(self, "_last_seq", 0):
            return None
        self._last_seq = int(f[b"seq"])
        k = int(f[b"k"])
        X = np.stack([decode(f[f"x{i}".encode()]) for i in range(k)])
        origin = np.frombuffer(f[b"origin"], dtype=np.int64).copy() if b"origin" in f else None
        return Message(X=X, obj=np.frombuffer(f[b"obj"], dtype=np.float64).copy(), seq=int(f[b"seq"]),
                       t=float(f[b"t"]), origin=origin)

    def history(self) -> list[tuple[float, int, np.ndarray]]:
        """(t, seq, objectives) of every retained message — for the run record."""
        out = []
        for _, f in self.r.xrange(self.key):
            out.append((float(f[b"t"]), int(f[b"seq"]), np.frombuffer(f[b"obj"], dtype=np.float64).copy()))
        return out


class _Info:
    def __init__(self, r, key: str):
        self.r, self.key = r, key
        self.seq = 0
        self._seen = None

    def post(self, **info) -> None:
        self.seq += 1
        self.r.set(self.key, json.dumps(dict(info, seq=self.seq, t=time.monotonic(), epoch=time.time())), ex=EXPIRE_S)

    def poll(self) -> dict | None:
        v = self.r.get(self.key)
        if v is None or v == self._seen:
            return None
        self._seen = v
        return json.loads(v)


class RedisMailbox:
    """All channels of one run on Redis, seen from one role ("solver", "ea", "cuopt", "runner", …)."""

    def __init__(self, url: str = DEFAULT_URL, run_id: str = "default", role: str = "runner", create: bool = False):
        """create=True (runner only): the run's key space must be empty (a stale run would leak a start barrier)."""
        import redis

        u = urlparse(url)
        q = parse_qs(u.query)
        run_id = q.get("run", [run_id])[0]
        self.r = redis.Redis(host=u.hostname or "127.0.0.1", port=u.port or 6390,
                             db=int((u.path or "/0").strip("/") or 0), socket_timeout=5)
        self.url, self.run_id, self.role = url, run_id, role
        self.p = f"itrea:{run_id}:"
        self.to_solver = _Stream(self.r, self.p + "ch:to_solver")
        self.to_ea = _Stream(self.r, self.p + "ch:to_ea")
        self.lp = _Stream(self.r, self.p + "ch:lp")
        self.info = _Info(self.r, self.p + "info")
        if create and next(iter(self.r.scan_iter(match=self.p + "*", count=100)), None) is not None:
            raise RuntimeError(f"redis key space {self.p}* is not empty (stale run?)")
        self.last_seq = 0

    # ------------------------------------------------------------ order
    def write_order(self, names) -> None:
        self.r.set(self.p + "order", json.dumps(list(map(str, names))), ex=EXPIRE_S)

    def read_order(self) -> list[str]:
        return json.loads(self.r.get(self.p + "order"))

    # ---------------------------------------------------------- barrier
    def ready(self) -> None:
        if self.r.exists(self.p + "start"):
            raise RuntimeError("stale start barrier: refusing to report ready")
        self.r.set(self.p + f"ready:{self.role}", repr(time.monotonic()), ex=EXPIRE_S)

    def all_ready(self, roles=("solver", "ea")) -> bool:
        return all(self.r.exists(self.p + f"ready:{r}") for r in roles)

    def release(self, t0: float | None = None) -> float:
        t0 = time.monotonic() if t0 is None else t0
        self.r.set(self.p + "start", json.dumps({"t0": t0, "epoch": time.time()}), ex=EXPIRE_S)
        return t0

    def wait_start(self, timeout: float = 600.0, poll: float = 0.005) -> float:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            v = self.r.get(self.p + "start")
            if v is not None:
                return float(json.loads(v)["t0"])
            if self.stopped():
                raise RuntimeError("stopped before start")
            time.sleep(poll)
        raise TimeoutError("start barrier not released")

    # ------------------------------------------------------------- stop
    def stop(self, reason: str = "") -> None:
        self.r.set(self.p + "stop", json.dumps({"t": time.monotonic(), "by": self.role, "reason": reason}),
                   nx=True, ex=EXPIRE_S)

    def stopped(self) -> bool:
        return bool(self.r.exists(self.p + "stop"))

    # --------------------------------------------------------- verdicts
    def verdict(self, seq: int, obj: float, accepted: bool, note: str = "") -> None:
        self.r.xadd(self.p + "verdicts", {"seq": seq, "t": repr(time.monotonic()), "obj": repr(float(obj)),
                                          "accepted": int(bool(accepted)), "note": note})
        self.r.expire(self.p + "verdicts", EXPIRE_S)

    def verdicts(self) -> list[dict]:
        out = []
        for _, f in self.r.xrange(self.p + "verdicts"):
            out.append({"seq": int(f[b"seq"]), "t": float(f[b"t"]), "obj": float(f[b"obj"]),
                        "accepted": bool(int(f[b"accepted"])), "note": f[b"note"].decode()})
        return out

    # ------------------------------------------------------ shared pool
    def pool_add(self, x: np.ndarray, obj: float, key: float, source: str, max_size: int = 256) -> str:
        """Add a feasible solution to the shared pool (any participant). Keeps the best `max_size` by key."""
        sid = f"{source}:{time.time_ns()}"
        pipe = self.r.pipeline()
        pipe.zadd(self.p + "pool", {sid: float(key)})
        pipe.hset(self.p + "pool:x", sid, encode(x))
        pipe.hset(self.p + "pool:meta", sid, json.dumps({"obj": float(obj), "source": source, "t": time.time()}))
        pipe.execute()
        extra = self.r.zrange(self.p + "pool", max_size, -1)      # worst beyond max_size
        if extra:
            pipe = self.r.pipeline()
            pipe.zrem(self.p + "pool", *extra)
            pipe.hdel(self.p + "pool:x", *extra)
            pipe.hdel(self.p + "pool:meta", *extra)
            pipe.execute()
        for k in ("pool", "pool:x", "pool:meta"):
            self.r.expire(self.p + k, EXPIRE_S)
        return sid

    def pool_top(self, k: int) -> list[tuple[np.ndarray, float, str]]:
        """The best k solutions of the shared pool: (x, objective, source)."""
        ids = self.r.zrange(self.p + "pool", 0, k - 1)
        if not ids:
            return []
        xs = self.r.hmget(self.p + "pool:x", ids)
        ms = self.r.hmget(self.p + "pool:meta", ids)
        out = []
        for x, m in zip(xs, ms):
            if x is None or m is None:
                continue
            meta = json.loads(m)
            out.append((decode(x), meta["obj"], meta["source"]))
        return out

    # ---------------------------------------------------------- cleanup
    def cleanup(self) -> int:
        keys = list(self.r.scan_iter(match=self.p + "*", count=1000))
        return self.r.delete(*keys) if keys else 0


def open_mailbox(spec: str, role: str, run_id: str | None = None, create: bool = False):
    """`spec` = a directory path / "file:<dir>" (file mailbox) or "redis://host:port/db?run=<id>" (Redis).
    Only the runner passes create=True."""
    if spec.startswith("redis://"):
        return RedisMailbox(spec, run_id=run_id or "default", role=role, create=create)
    from itrea.exchange.mailbox import Mailbox
    return Mailbox(spec[5:] if spec.startswith("file:") else spec, role, create=create, run_id=run_id)
