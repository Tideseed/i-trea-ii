"""The Redis exchange platform's container: start it, report its state, delete one run's keys (vault D-006).

The container `itrea-redis` is this project's own (image `redis:7-alpine`, already on the host — never pulled):
loopback port 6390, no persistence, `maxmemory 1536mb noeviction`, a 2 GB cap, pinned to A725 core 0 (never a timed
X925 core), `--restart no`. The host runs other Redis instances ([secrets manager], fhub, buzz) that belong to other services;
this module only ever touches the container named `itrea-redis` and only keys under `itrea:<run_id>:`.

    python -m itrea.experiments.exchange_ctl up        start (or create) itrea-redis and wait for PING
    python -m itrea.experiments.exchange_ctl status    container state, memory, number of itrea runs with keys
    python -m itrea.experiments.exchange_ctl flush RUN_ID   delete the keys of ONE run (itrea:RUN_ID:*)
    python -m itrea.experiments.exchange_ctl down      stop itrea-redis (nothing else)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time

CONTAINER = "itrea-redis"
IMAGE = "redis:7-alpine"
HOST, PORT = "127.0.0.1", 6390
URL = f"redis://{HOST}:{PORT}/0"
# exactly the D-006 setup; `--pull never`: the image must already be on the host (no downloads)
RUN_ARGS = ["docker", "run", "-d", "--pull", "never", "--name", CONTAINER, "--cpuset-cpus", "0",
            "--memory", "2g", "--memory-swap", "2g", "-p", f"{HOST}:{PORT}:6379", "--restart", "no", IMAGE,
            "redis-server", "--save", "", "--appendonly", "no", "--maxmemory", "1536mb",
            "--maxmemory-policy", "noeviction"]
_RUN_ID_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+~=@-]*$")   # no glob characters (* ? [ ]), no ':'


class ExchangeError(RuntimeError):
    pass


def _docker(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def container_state() -> str | None:
    """'running' | 'exited' | 'created' | … for itrea-redis; None if it does not exist."""
    r = _docker("inspect", "--format", "{{.State.Status}}", CONTAINER)
    return r.stdout.strip() if r.returncode == 0 else None


def client(host: str = HOST, port: int = PORT, timeout: float = 5.0):
    import redis
    return redis.Redis(host=host, port=port, db=0, socket_timeout=timeout)


def ping(timeout: float = 2.0) -> bool:
    try:
        return bool(client(timeout=timeout).ping())
    except Exception:  # noqa: BLE001
        return False


def up(wait_s: float = 30.0, log=print) -> dict:
    """Ensure itrea-redis runs and answers PING. Creates it exactly as in D-006 if it does not exist (local image
    only). Never touches any other container."""
    st = container_state()
    if st is None:
        r = _docker("image", "inspect", "--format", "{{.Id}}", IMAGE)
        if r.returncode != 0:
            raise ExchangeError(f"image {IMAGE} is not on the host; no pull (D-006, no downloads)")
        log(f"creating {CONTAINER} from the local {IMAGE}")
        r = subprocess.run(RUN_ARGS, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise ExchangeError(f"docker run {CONTAINER} failed: {r.stderr.strip()[:300]}")
    elif st != "running":
        log(f"starting {CONTAINER} (was {st})")
        r = _docker("start", CONTAINER)
        if r.returncode != 0:
            raise ExchangeError(f"docker start {CONTAINER} failed: {r.stderr.strip()[:300]}")
    end = time.monotonic() + wait_s
    while not ping():
        if time.monotonic() > end:
            raise ExchangeError(f"{CONTAINER} does not answer PING on {HOST}:{PORT} after {wait_s:.0f} s")
        time.sleep(0.5)
    return status()


def down(log=print) -> str | None:
    """Stop itrea-redis (by its exact name; no other container). Its keys vanish with it (no persistence)."""
    st = container_state()
    if st == "running":
        r = _docker("stop", CONTAINER)
        if r.returncode != 0:
            raise ExchangeError(f"docker stop {CONTAINER} failed: {r.stderr.strip()[:300]}")
        log(f"stopped {CONTAINER}")
    return container_state()


def run_ids(c=None) -> dict[str, int]:
    """Runs that have keys now: {run_id: number of keys}."""
    c = c or client()
    out: dict[str, int] = {}
    for k in c.scan_iter(match="itrea:*", count=1000):
        k = k.decode() if isinstance(k, bytes) else k
        parts = k.split(":")
        if len(parts) >= 3:
            out[parts[1]] = out.get(parts[1], 0) + 1
    return out


def status() -> dict:
    d: dict = dict(container=CONTAINER, state=container_state(), url=URL, ping=False)
    if d["state"] == "running" and ping():
        c = client()
        mem = c.info("memory")
        d.update(ping=True, used_memory_mb=round(mem.get("used_memory", 0) / 2**20, 1),
                 maxmemory_mb=round(mem.get("maxmemory", 0) / 2**20, 1), dbsize=c.dbsize(), runs=run_ids(c))
    return d


def check_run_id(run_id: str) -> str:
    """A run id safe to use in a key pattern: no glob characters, no ':' and not empty."""
    if not run_id or not _RUN_ID_OK.match(run_id):
        raise ExchangeError(f"refusing run id {run_id!r}: it must match {_RUN_ID_OK.pattern} (no glob characters)")
    return run_id


def flush(run_id: str, c=None) -> int:
    """Delete ONLY the keys under itrea:<run_id>:*. Returns the number deleted."""
    check_run_id(run_id)
    c = c or client()
    keys = list(c.scan_iter(match=f"itrea:{run_id}:*", count=1000))
    # belt and braces: the pattern cannot match another run, but check every key's prefix anyway
    pref = f"itrea:{run_id}:".encode()
    keys = [k for k in keys if (k if isinstance(k, bytes) else k.encode()).startswith(pref)]
    return int(c.delete(*keys)) if keys else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="itrea.experiments.exchange_ctl", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("up")
    sub.add_parser("status")
    sub.add_parser("down")
    f = sub.add_parser("flush")
    f.add_argument("run_id")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "up":
            print(json.dumps(up(), indent=1))
        elif a.cmd == "status":
            print(json.dumps(status(), indent=1))
        elif a.cmd == "down":
            print(down())
        elif a.cmd == "flush":
            print(flush(a.run_id))
    except ExchangeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
