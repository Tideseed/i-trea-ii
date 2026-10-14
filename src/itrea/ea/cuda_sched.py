"""How the EA host's CUDA context waits on the GPU (`host.cuda_sched`, R1 of vault `30-results/2026-10-14 C1 diagnosis
and improvement candidates.md`).

Why. CUDA's default scheduling (CU_CTX_SCHED_AUTO) spins while there are fewer active contexts than logical cores, so
every synchronisation of the EA — `ls_fj_gpu`'s `int(cp.count_nonzero(...))` after each ~50-ms launch, `backend.
synchronize` (`Stream.null.synchronize`), the per-generation timing events — keeps the host's core busy for the whole
GPU wait. In arm A9 the EA host shares one X925 core with the HiGHS LP worker it spawns at t0, which then gets about
half the core. Blocking sync (CU_CTX_SCHED_BLOCKING_SYNC) makes the host thread sleep on an OS primitive until the GPU
is done; yield (CU_CTX_SCHED_YIELD) keeps polling but yields the time slice at each poll.

How. CuPy 14.2 has no `setDeviceFlags`, so the flag goes through the driver API (`libcuda.so.1`, loaded only when a
non-default mode is asked for, or after CuPy's context exists for the read-only "auto" record — never at import, so
the CPU tests never touch a driver): cuInit(0), cuDeviceGet,
cuDevicePrimaryCtxGetState (refused if the primary context is already active: then something retained it first and the
flag would not be the one the run waits with), cuDevicePrimaryCtxSetFlags (the `_v2` entry point when exported, as in
driver 580), cuDevicePrimaryCtxGetState again (the flags must read back). `Engine.prepare` does this before the
backend exists (before CuPy is asked for anything, the warm-up and t0); once `cp.zeros(1)` has created the context it
checks again that the primary context is active with the requested scheduling flag (`verify`) and raises otherwise —
a run that would be labelled blocking sync but waited by spinning must not happen. Under "auto" the same place only
READS the active context's flags (`read_state`: the driver is loaded then, after CuPy loaded it; read-only, never
raises), so a control run records the flag the driver reports. The record goes into the start-up record and the summary
(`cuda_sched`); `finish` reads the state once more (`flags_at_end`).

    rec = apply("blocking_sync")              # before CuPy creates the context ("auto": calls nothing)
    ...                                       # cp.zeros(1)
    rec = verify(rec)                         # active, flags & CU_CTX_SCHED_MASK == the request ("auto": read only)
    api = DriverAPI.load()                    # the real driver (or a fake with the same five methods, in tests)

`DriverAPI` is the only place that calls the driver; tests pass a fake object with the same methods (`init`,
`device_get`, `primary_ctx_get_state`, `primary_ctx_set_flags`, `name`).

This module imports no other `itrea` module and never CuPy.
"""
from __future__ import annotations

import ctypes
import time

CU_CTX_SCHED_AUTO = 0x00
CU_CTX_SCHED_SPIN = 0x01
CU_CTX_SCHED_YIELD = 0x02
CU_CTX_SCHED_BLOCKING_SYNC = 0x04
CU_CTX_SCHED_MASK = 0x07
MODES = {"auto": CU_CTX_SCHED_AUTO, "yield": CU_CTX_SCHED_YIELD, "blocking_sync": CU_CTX_SCHED_BLOCKING_SYNC}
SCHED_NAMES = {CU_CTX_SCHED_AUTO: "auto", CU_CTX_SCHED_SPIN: "spin", CU_CTX_SCHED_YIELD: "yield",
               CU_CTX_SCHED_BLOCKING_SYNC: "blocking_sync"}
LIBCUDA = "libcuda.so.1"
CUDA_SUCCESS = 0


class CudaSchedError(RuntimeError):
    """The scheduling flag could not be set or does not hold (the run must not start mislabelled)."""


def sched_name(flags: int | None) -> str | None:
    if flags is None:
        return None
    return SCHED_NAMES.get(int(flags) & CU_CTX_SCHED_MASK, f"0x{int(flags) & CU_CTX_SCHED_MASK:x}")


def events_block(mode: str) -> bool:
    """Whether the engine's timing events need cudaEventBlockingSync: under blocking sync only (a default-flag event
    wait may spin whatever the context flag says; under "yield" the event wait follows the context's spin-and-yield)."""
    return mode == "blocking_sync"


class DriverAPI:
    """The five driver calls, through ctypes. `load()` opens libcuda.so.1; a missing symbol or library raises."""

    def __init__(self, lib, set_flags_symbol: str):
        self._lib = lib
        self.set_flags_symbol = set_flags_symbol
        self._set_flags = getattr(lib, set_flags_symbol)
        for fn in (lib.cuInit, lib.cuDeviceGet, lib.cuDevicePrimaryCtxGetState, self._set_flags):
            fn.restype = ctypes.c_int
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuDevicePrimaryCtxGetState.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint),
                                                   ctypes.POINTER(ctypes.c_int)]
        self._set_flags.argtypes = [ctypes.c_int, ctypes.c_uint]

    @classmethod
    def load(cls, path: str = LIBCUDA) -> "DriverAPI":
        lib = ctypes.CDLL(path)
        sym = "cuDevicePrimaryCtxSetFlags_v2" if hasattr(lib, "cuDevicePrimaryCtxSetFlags_v2") else \
            "cuDevicePrimaryCtxSetFlags"
        return cls(lib, sym)

    @property
    def name(self) -> str:
        return f"{LIBCUDA}:{self.set_flags_symbol}"

    @staticmethod
    def _ok(rc: int, what: str) -> None:
        if int(rc) != CUDA_SUCCESS:
            raise CudaSchedError(f"{what} returned CUresult {int(rc)}")

    def init(self) -> None:
        self._ok(self._lib.cuInit(0), "cuInit(0)")

    def device_get(self, ordinal: int) -> int:
        dev = ctypes.c_int(-1)
        self._ok(self._lib.cuDeviceGet(ctypes.byref(dev), int(ordinal)), f"cuDeviceGet({ordinal})")
        return int(dev.value)

    def primary_ctx_get_state(self, dev: int) -> tuple[int, bool]:
        flags, active = ctypes.c_uint(0), ctypes.c_int(0)
        self._ok(self._lib.cuDevicePrimaryCtxGetState(int(dev), ctypes.byref(flags), ctypes.byref(active)),
                 "cuDevicePrimaryCtxGetState")
        return int(flags.value), bool(active.value)

    def primary_ctx_set_flags(self, dev: int, flags: int) -> None:
        self._ok(self._set_flags(int(dev), int(flags)), f"{self.set_flags_symbol}({flags:#x})")


def apply(mode: str, ordinal: int = 0, api=None) -> dict:
    """Set the primary context's scheduling flag of device `ordinal` (module docstring) and return the record. "auto"
    calls nothing (the default stays CUDA's own). Raises CudaSchedError when the context is already active or the flag
    does not read back."""
    if mode not in MODES:
        raise ValueError(f"host.cuda_sched {mode!r} (known: {', '.join(MODES)})")
    want = MODES[mode]
    rec = dict(mode=mode, requested=want, ordinal=int(ordinal))
    if want == CU_CTX_SCHED_AUTO:
        rec.update(state="default", api=None)
        return rec
    api = api if api is not None else DriverAPI.load()
    t = time.perf_counter()
    api.init()
    dev = api.device_get(ordinal)
    flags0, active0 = api.primary_ctx_get_state(dev)
    rec.update(api=getattr(api, "name", type(api).__name__), device=dev, flags_before=flags0, active_before=active0)
    if active0:
        raise CudaSchedError(f"the primary context of device {ordinal} is already active (flags {flags0:#x}): "
                             f"host.cuda_sched={mode} must be set before anything retains it")
    # keep the other flag bits (e.g. CU_CTX_LMEM_RESIZE_TO_MAX), replace the scheduling bits
    api.primary_ctx_set_flags(dev, (flags0 & ~CU_CTX_SCHED_MASK) | want)
    flags1, active1 = api.primary_ctx_get_state(dev)
    rec.update(flags_set=flags1, active_after_set=active1, set_s=round(time.perf_counter() - t, 6))
    if flags1 & CU_CTX_SCHED_MASK != want:
        raise CudaSchedError(f"cuDevicePrimaryCtxGetState reads scheduling {sched_name(flags1)} after setting {mode}")
    rec["state"] = "set"
    rec["_api"] = api
    return rec


def read_state(rec: dict, api=None) -> dict:
    """Mode "auto", after the context exists: READ the primary context's flags (cuInit, cuDeviceGet,
    cuDevicePrimaryCtxGetState — read-only, nothing is set), so a run at CUDA's default records the flag the driver
    reports (D-025: the control's flag state is read back, not inferred from the configuration). libcuda.so.1 is
    loaded here, after CuPy has loaded it, never at import. Never raises: a failure is recorded as `read_error` (the
    report then reads the leg as unverified). Returns the record with `flags_active` / `active` / `sched_active` and
    state "read"."""
    try:
        api = api if api is not None else (rec.get("_api") or DriverAPI.load())
        api.init()
        dev = api.device_get(rec.get("ordinal", 0))
        flags, active = api.primary_ctx_get_state(dev)
    except Exception as e:  # noqa: BLE001 — a record, not a check
        rec.update(state="read_error", read_error=f"{type(e).__name__}: {e}"[:300])
        return rec
    rec.update(api=getattr(api, "name", type(api).__name__), device=dev, flags_active=flags, active=active,
               sched_active=sched_name(flags), state="read")
    rec["_api"] = api
    return rec


def verify(rec: dict, api=None) -> dict:
    """After the context exists (`cp.zeros(1)`): the primary context must be active with the requested scheduling
    flag. Returns the record with `flags_active` / `active` / `sched_active`; raises CudaSchedError otherwise. A record
    of mode "auto" is only read (`read_state`; never raises): CUDA's default is not a request this module made."""
    api = api if api is not None else rec.get("_api")
    if rec.get("requested", CU_CTX_SCHED_AUTO) == CU_CTX_SCHED_AUTO:
        return read_state(rec, api)
    flags, active = api.primary_ctx_get_state(rec["device"])
    rec.update(flags_active=flags, active=active, sched_active=sched_name(flags))
    if not active:
        raise CudaSchedError("the primary context is not active after the backend created it: CuPy did not use the "
                             "primary context the flag was set on")
    if flags & CU_CTX_SCHED_MASK != rec["requested"]:
        raise CudaSchedError(f"the active primary context waits by {sched_name(flags)}, not {rec['mode']}")
    rec["state"] = "verified"
    return rec


def state_at_end(rec: dict) -> dict:
    """The flag read once more at the end of the run (`flags_at_end`, `active_at_end`); never raises."""
    api = rec.get("_api")
    if api is None or "device" not in rec:
        return {}
    try:
        flags, active = api.primary_ctx_get_state(rec["device"])
    except Exception as e:  # noqa: BLE001 — a record, not a check
        return dict(end_error=f"{type(e).__name__}: {e}"[:300])
    return dict(flags_at_end=flags, active_at_end=active, sched_at_end=sched_name(flags))


def public(rec: dict | None) -> dict | None:
    """The record without the API handle (for the start-up record and the summary)."""
    if rec is None:
        return None
    return {k: v for k, v in rec.items() if not k.startswith("_")}
