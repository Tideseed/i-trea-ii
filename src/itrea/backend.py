"""Array backend: one EA code path for the GPU (CuPy) and the CPU (NumPy / SciPy).

Every EA module takes a `Backend` and uses `bk.xp` (the array namespace) and `bk.sparse` (the sparse-matrix module)
instead of importing NumPy or CuPy directly. The CPU backend is both the test backend and an experimental control
(arm A4 in D-002).

GPU memory on the DGX Spark (GB10) is system RAM, and a systemd `MemoryMax` scope does NOT charge it (measured
2026-10-05). `get_backend("cupy", gpu_mem_gb=...)` therefore always sets a CuPy memory-pool limit — the only real cap
on the EA's device memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np
import scipy.sparse as _sps


@dataclass
class Backend:
    name: str  # "numpy" | "cupy"
    xp: ModuleType
    sparse: ModuleType
    gpu_mem_limit: int | None = None  # bytes, CuPy pool limit (None on the CPU)
    _rng: Any = field(default=None, repr=False)

    @property
    def is_gpu(self) -> bool:
        return self.name == "cupy"

    def asnumpy(self, a) -> np.ndarray:
        if self.is_gpu:
            import cupy as cp
            return cp.asnumpy(a)
        return np.asarray(a)

    def asarray(self, a, dtype=None):
        return self.xp.asarray(a, dtype=dtype)

    def to_device_sparse(self, m: _sps.spmatrix, dtype=None):
        """A SciPy sparse matrix on this backend, in CSR form."""
        m = _sps.csr_matrix(m)
        if dtype is not None:
            m = m.astype(dtype)
        if self.is_gpu:
            import cupyx.scipy.sparse as cps
            return cps.csr_matrix(m)
        return m

    def synchronize(self) -> None:
        if self.is_gpu:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()

    def rng(self, seed: int):
        """Device random generator (CuPy's Generator has `random` and `integers`; use only those)."""
        return self.xp.random.default_rng(seed)

    def scatter_add(self, out, idx, vals) -> None:
        """out[idx] += vals with repeated indices accumulated."""
        if self.is_gpu:
            import cupy as cp
            cp.add.at(out, idx, vals)          # cupyx.scatter_add is deprecated in CuPy 14
        else:
            np.add.at(out, idx, vals)

    def mem_used_bytes(self) -> int:
        if self.is_gpu:
            import cupy as cp
            return int(cp.get_default_memory_pool().used_bytes())
        return 0

    def mem_total_bytes(self) -> int:
        if self.is_gpu:
            import cupy as cp
            return int(cp.get_default_memory_pool().total_bytes())
        return 0


def get_backend(name: str = "auto", gpu_mem_gb: float | None = None, allow_cpu_fallback: bool = False) -> Backend:
    """name: "numpy", "cupy" or "auto" (CuPy if it imports and a device is visible; NumPy only with
    allow_cpu_fallback=True — a run labelled GPU must never silently run on the CPU).

    For CuPy, `gpu_mem_gb` is required (no unbounded device allocations on a shared unified-memory box)."""
    if name in ("cupy", "auto"):
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cps
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
        except Exception:
            if name == "cupy" or not allow_cpu_fallback:
                raise
        else:
            if gpu_mem_gb is None:
                raise ValueError("gpu_mem_gb is required for the CuPy backend (CuPy pool limit = the only GPU cap)")
            limit = int(gpu_mem_gb * 1024**3)
            cp.get_default_memory_pool().set_limit(size=limit)
            return Backend("cupy", cp, cps, gpu_mem_limit=limit)
    if name not in ("numpy", "auto"):
        raise ValueError(f"unknown backend {name!r}")
    return Backend("numpy", np, _sps)
