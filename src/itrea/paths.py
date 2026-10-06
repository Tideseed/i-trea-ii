"""Repository paths. Instance paths in job files and traces are stored relative to the repository root."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("ITREA_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("ITREA_DATA", ROOT / "data"))
TRACES = ROOT / "traces"
DATASETS = ROOT / "datasets"
CONFIGS = ROOT / "configs"
JOBS = ROOT / "jobs"


def resolve(path: str | os.PathLike) -> Path:
    """Absolute path for a repository-relative instance path (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def relative(path: str | os.PathLike) -> str:
    """Repository-relative form of a path when it lies inside the repository (for traces and job files)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)
