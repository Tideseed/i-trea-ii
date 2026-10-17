"""The reproducibility record stored in every run record: package versions, CPU model per core, GPU, driver, kernel,
CUDA, and the git state (revision, dirty tree, the pre-registration tag).

Adapted from berkorbay/optopt@2a3348c0a102 (MIT), `src/optopt/experiments/record_env.py`. Versions are read from package
metadata, not by importing CuPy or cuOpt: importing cuOpt in the runner costs seconds and hundreds of MB, and a broken
GPU stack must not stop the record from being written (a missing package is recorded as null).

    python -m itrea.experiments.record_env            print the record
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from itrea.paths import ROOT

# distribution names (pip) of the packages the study depends on
DISTS = {"numpy": "numpy", "scipy": "scipy", "cupy": "cupy-cuda13x", "highspy": "highspy", "pyscipopt": "PySCIPOpt",
         "cuopt": "cuopt-cu13", "redis": "redis", "itrea": "i-trea-ii"}
PREREG_TAG = "prereg-v1"
FROZEN_TAG = "frozen-e2"


def sh(cmd: list[str], timeout: float = 20.0) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for k, dist in DISTS.items():
        try:
            out[k] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[k] = None
    if out.get("cupy") is None:            # another CUDA build of CuPy
        for d in metadata.distributions():
            name = (d.metadata["Name"] or "").lower()
            if name.startswith("cupy"):
                out["cupy"] = d.version
                break
    try:                                   # the SCIP library version (cheap; no model is built)
        import pyscipopt
        out["scip"] = str(pyscipopt.Model().version())
    except Exception:  # noqa: BLE001
        out["scip"] = None
    return out


def cpu_models() -> dict[str, str]:
    """{core: model name}, e.g. {"0": "Cortex-A725", "5": "Cortex-X925"} (from lscpu)."""
    txt = sh(["lscpu", "-e=CPU,MODELNAME"]) or ""
    out = {}
    for line in txt.splitlines()[1:]:
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            out[parts[0]] = parts[1].strip()
    return out


def gpu() -> dict[str, str | None]:
    q = sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    name = driver = None
    if q:
        parts = [p.strip() for p in q.splitlines()[0].split(",")]
        name, driver = (parts + [None, None])[:2]
    head = sh(["nvidia-smi"]) or ""
    m = re.search(r"CUDA Version:\s*([0-9.]+)", head)
    return dict(name=name, driver=driver, cuda=m.group(1) if m else None)


# --------------------------------------------------------------------------------------------------------- git
def _git(*args: str, root: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root or ROOT), *args], capture_output=True, text=True, timeout=30)


def git_revision(root: Path | None = None) -> str | None:
    r = _git("rev-parse", "HEAD", root=root)
    return r.stdout.strip() if r.returncode == 0 else None


def tag_commit(tag: str, root: Path | None = None) -> str | None:
    """The commit a tag points to, or None if the tag does not exist."""
    r = _git("rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}", root=root)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def is_ancestor(tag: str, rev: str = "HEAD", root: Path | None = None) -> bool:
    """True if `tag` exists and is an ancestor of `rev` (git merge-base --is-ancestor)."""
    if tag_commit(tag, root) is None:
        return False
    return _git("merge-base", "--is-ancestor", tag, rev, root=root).returncode == 0


def file_at(rev: str, path: str, root: Path | None = None) -> bytes | None:
    """The bytes of `path` (relative to the repository root) in commit `rev` (`git show rev:path`); None if the
    revision or the file does not exist there."""
    r = subprocess.run(["git", "-C", str(root or ROOT), "show", f"{rev}:{path}"], capture_output=True, timeout=30)
    return r.stdout if r.returncode == 0 else None


def dirty_paths(paths: tuple[str, ...] = ("src", "configs"), root: Path | None = None) -> list[str] | None:
    """Uncommitted changes (tracked or untracked) under `paths`; None if not a git checkout."""
    r = _git("status", "--porcelain", "--", *paths, root=root)
    if r.returncode != 0:
        return None
    return [line[3:] for line in r.stdout.splitlines() if line.strip()]


def git_state(root: Path | None = None) -> dict:
    dirty = dirty_paths(root=root)
    return dict(revision=git_revision(root), dirty=bool(dirty) if dirty is not None else None,
                dirty_paths=(dirty or [])[:20],
                prereg_tag=tag_commit(PREREG_TAG, root), prereg_ancestor=is_ancestor(PREREG_TAG, root=root),
                frozen_tag=tag_commit(FROZEN_TAG, root))


@lru_cache(maxsize=1)
def static_env() -> dict:
    """What does not change during a session (cached): versions, hardware, kernel."""
    return dict(versions=versions(), cpu_models=cpu_models(), gpu=gpu(), kernel=platform.release(),
                machine=platform.machine(), hostname=platform.node(),
                hardware="NVIDIA DGX Spark (GB10: 10x Cortex-X925 + 10x Cortex-A725, 128 GB unified memory)")


def record(root: Path | None = None) -> dict:
    """The full record: static environment + the git state now."""
    return dict(static_env(), git=git_state(root))


def main(argv=None) -> int:
    print(json.dumps(record(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
