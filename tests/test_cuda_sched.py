"""R1 of the C1 diagnosis (vault `30-results/2026-10-14 C1 diagnosis and improvement candidates.md`, D-025): the EA host
waits on the GPU by blocking sync (`host.cuda_sched`, `itrea.ea.cuda_sched`), its fallbacks (a) `bnice` — renice +10
relative to the LP worker right after its spawn (`host.nice_after_lp_spawn`), no context flag — and (b) `ysync` —
yield, the read-back of the flags under CUDA's default ("auto"), and the instrumentation recorded in every run (the LP worker's own
clock and CPU in the `lp` event, the EA's on-clock CPU, the FJ launch count).

CPU tests only touch a FAKE driver (no libcuda call, no CuPy device): the configuration, its hash rule and the variant
files, the driver-call sequence and its refusals, the engine's call order (the flag before the backend exists), the
blocking timing events, the renice (in a subprocess, never this process), the LP worker's stamps (a real HiGHS worker
on a toy, marker `solver`). The GPU checks (marker `gpu`, inside a window only) assert the flag on the real driver, the
host's CPU per wall second with and without blocking sync (`tools/bsync_check.py`), and the cold-cache test of every R1
variant in A9's layout (the EA alone on one core, `exchange.from_cuopt` with a file mailbox).
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from itrea.backend import Backend
from itrea.benchmarks import toy
from itrea.config import EAConfig, HostCfg
from itrea.ea import cuda_sched as CS
from itrea.ea import engine as E
from itrea.ea.engine import Engine, renice_threads

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SMALL = {"population.size": 48, "population.offspring_per_gen": 48, "gpu_mem_gb": 1.0}
PY = sys.executable
BSYNC = {"host.cuda_sched": "blocking_sync"}

# the variant files of R1 (and R2, test_lp_decode_gate.py) and their pinned hashes: EA configuration hash, and the
# run hash of an A9 job under the ext harness (exchange.from_cuopt added by the runner; glass4, 120 s)
R1_FILES = {"bsync": ("305c8902d763", "330a10a77556"), "bnice": ("96a299330981", "d266e4aaecbe"),
            "ysync": ("f81d6b6209f0", "a08221cca653"), "boY-bsync": ("29e61aa41619", "b760e770cb34"),
            "boY-bnice": ("5b1c4779c74b", "b4f4af8376dd"), "boY-ysync": ("666865eac174", "22137f009005")}


def _cfg(name: str) -> EAConfig:
    return EAConfig.from_yaml(CONFIGS / f"itrea2-dev-rc-3way-{name}.yaml" if name else CONFIGS /
                              "itrea2-dev-rc-3way.yaml")


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k != "weights":
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _diff(a: EAConfig, b: EAConfig) -> dict:
    fa, fb = _flat(a.to_dict()), _flat(b.to_dict())
    return {k: fb[k] for k in fb if k != "name" and fa[k] != fb[k]}


# ============================================================================================ the configuration
def test_default_host_section_is_outside_the_hash():
    """The whole `host` section is left out of the hash while every field is at its default — an empty {"host": {}}
    would move every recorded hash — and any non-default field brings it in."""
    base = EAConfig()
    assert base.host == HostCfg() and base.hash() == "9ac4a93bf93b"
    assert base.with_overrides({"host.cuda_sched": "auto", "host.nice_after_lp_spawn": 0}).hash() == base.hash()
    a = base.with_overrides(BSYNC)
    y = base.with_overrides({"host.cuda_sched": "yield"})
    assert len({base.hash(), a.hash(), y.hash()}) == 3
    lp = base.with_overrides({"seeding.lp_source": "highs_async"})
    assert lp.with_overrides({"host.nice_after_lp_spawn": 10}).hash() != lp.hash()


def test_recorded_hashes_stand():
    """No existing configuration moves: the default, the A9 / A3x development default, A3's, D-018's and D-023's."""
    want = {"": "1d12ad490dd0", "boY": "5c8b535fb245", "bo1": "c5b5e01856eb", "bo1s": "198e0d2ca540",
            "bo0": "b7ab026b8a5d", "bof50": "7427bf6ca4d7", "bof25": "9a8d7e711bf9", "bof12": "edca6f71f266"}
    assert {k: _cfg(k).hash() for k in want} == want
    assert EAConfig.from_yaml(CONFIGS / "itrea2-dev-rc-ld.yaml").hash() == "f030e23909ad"
    assert EAConfig.from_yaml(CONFIGS / "itrea2-default.yaml").hash() == "9ac4a93bf93b"


# Every configuration file's EA hash as of 2026-10-15 (D-030's build, commit after 75bbe12): a change to any file, or a
# new file not entered here, fails — "existing hashes unchanged" tested literally.
ALL_CONFIG_HASHES = {
    "itrea1-faithful.yaml": "b517bfc675d4", "itrea2-adaptive-lps-rc.yaml": "26481ba0c675",
    "itrea2-adaptive-lps.yaml": "a2ad94e9188d", "itrea2-adaptive.yaml": "584ddee83747",
    "itrea2-default.yaml": "9ac4a93bf93b", "itrea2-dev-rc-3way-bnice.yaml": "96a299330981",
    "itrea2-dev-rc-3way-bo0.yaml": "b7ab026b8a5d", "itrea2-dev-rc-3way-bo1.yaml": "c5b5e01856eb",
    "itrea2-dev-rc-3way-bo1s.yaml": "198e0d2ca540", "itrea2-dev-rc-3way-boY-bnice.yaml": "5b1c4779c74b",
    "itrea2-dev-rc-3way-boY-bsync-ld.yaml": "c296898cddc4", "itrea2-dev-rc-3way-boY-bsync-ld0.yaml": "2b6baa898edf",
    "itrea2-dev-rc-3way-boY-bsync.yaml": "29e61aa41619", "itrea2-dev-rc-3way-boY-ld0.yaml": "aabac30b5056",
    "itrea2-dev-rc-3way-boY-ysync.yaml": "666865eac174", "itrea2-dev-rc-3way-boY.yaml": "5c8b535fb245",
    "itrea2-dev-rc-3way-bof12.yaml": "edca6f71f266", "itrea2-dev-rc-3way-bof25.yaml": "9a8d7e711bf9",
    "itrea2-dev-rc-3way-bof50.yaml": "7427bf6ca4d7", "itrea2-dev-rc-3way-bsync-ld.yaml": "119c6e17f830",
    "itrea2-dev-rc-3way-bsync-ld0.yaml": "b7af15f4b62a", "itrea2-dev-rc-3way-bsync.yaml": "305c8902d763",
    "itrea2-dev-rc-3way-ld0.yaml": "a05af2c732dc", "itrea2-dev-rc-3way-ysync.yaml": "f81d6b6209f0",
    "itrea2-dev-rc-3way.yaml": "1d12ad490dd0", "itrea2-dev-rc-ld.yaml": "f030e23909ad",
    "itrea2-dev-rc-3way-ysync-ld0.yaml": "835c7c5a249c",          # D-030's second candidate (new)
}


def test_every_configuration_hash_is_pinned():
    assert {p.name: EAConfig.from_yaml(p).hash() for p in sorted(CONFIGS.glob("*.yaml"))} == ALL_CONFIG_HASHES


def test_ysync_ld0_is_ld0_plus_yield_only():
    """D-030's second frozen-v2 candidate: configs/itrea2-dev-rc-3way-ysync-ld0.yaml = D-026's adopted ld0 + D-029's
    `host.cuda_sched: yield` and nothing else — in the YAML files (only `name` and `host` differ, `host` holding only
    `cuda_sched: yield`) and in the effective configuration (the flattened diff); ysync-ld0 is also ysync + ld0's three
    decode fields; hashes pinned (EA 835c7c5a249c, A9 run hash under ext f345de0ab936), ld0's and ysync's unchanged."""
    import yaml
    from itrea.experiments import pair
    ld0, ysync, both = (yaml.safe_load((CONFIGS / f"itrea2-dev-rc-3way-{n}.yaml").read_text())
                        for n in ("ld0", "ysync", "ysync-ld0"))
    assert {k for k in set(ld0) | set(both) if ld0.get(k) != both.get(k)} == {"name", "host"}
    assert "host" not in ld0 and both["host"] == {"cuda_sched": "yield"}
    assert both["name"] == "itrea2-dev-rc-3way-ysync-ld0"
    assert _diff(_cfg("ld0"), _cfg("ysync-ld0")) == {"host.cuda_sched": "yield"}
    assert _diff(_cfg("ysync"), _cfg("ysync-ld0")) == {"repair.continuous": "lp_decode",
                                                       "repair.lp_decode_gate": "lp_worker",
                                                       "repair.lp_decode_nice": 0}
    hashes = {}
    for n in ("ld0", "ysync", "ysync-ld0"):
        j = dict(instance="data/miplib/inst/glass4.mps.gz", set="x", arm="A9", seed=0, budget=120,
                 config=f"configs/itrea2-dev-rc-3way-{n}.yaml", harness="ext")
        hashes[n] = (_cfg(n).hash(), pair.run_hash(j))
        pair.job_budget(j)                                       # the runner plans it (gated lp_decode in A9)
    assert hashes == {"ld0": ("a05af2c732dc", "341267810eed"), "ysync": ("f81d6b6209f0", "a08221cca653"),
                      "ysync-ld0": ("835c7c5a249c", "f345de0ab936")}


@pytest.mark.parametrize("over, what", [
    ({"host.cuda_sched": "spin"}, "cuda_sched"),
    ({"host.cuda_sched": "blocking"}, "cuda_sched"),
    ({"host.nice_after_lp_spawn": -1}, "nice_after_lp_spawn"),
    ({"host.nice_after_lp_spawn": 20}, "nice_after_lp_spawn"),
    ({"host.nice_after_lp_spawn": True}, "nice_after_lp_spawn"),
    ({"host.nice_after_lp_spawn": 2.5}, "nice_after_lp_spawn"),
    ({"host.nice_after_lp_spawn": 10}, "highs_async"),            # no LP worker: the renice would never happen
])
def test_validation_refuses_bad_host_settings(over, what):
    with pytest.raises(ValueError, match=what):
        EAConfig().with_overrides(over)


def test_unknown_host_key_is_refused():
    with pytest.raises(KeyError):
        EAConfig.from_dict({"host": {"cuda_schedule": "blocking_sync"}})


@pytest.mark.parametrize("name", sorted(R1_FILES))
def test_r1_variant_files_are_their_base_plus_one_change(name):
    """bsync / bnice / ysync on the A9 development default, and the same three on D-023's boY (the treatment when the
    control is A9@boY): each file is its base verbatim plus the one change — bnice is the renice WITHOUT the context
    flag (D-025's fallback (a)) — and its hashes are pinned."""
    from itrea.experiments import pair
    base = "boY" if name.startswith("boY-") else ""
    v = name.removeprefix("boY-")
    want = {"bsync": {"host.cuda_sched": "blocking_sync"},
            "bnice": {"host.nice_after_lp_spawn": 10},
            "ysync": {"host.cuda_sched": "yield"}}[v]
    cfg = _cfg(name)
    assert _diff(_cfg(base), cfg) == want
    assert cfg.name == f"itrea2-dev-rc-3way-{name}"
    j = dict(instance="data/miplib/inst/glass4.mps.gz", set="x", arm="A9", seed=0, budget=120,
             config=f"configs/itrea2-dev-rc-3way-{name}.yaml", harness="ext")
    assert (cfg.hash(), pair.run_hash(j)) == R1_FILES[name]


def test_the_a9_default_run_hashes_stand():
    from itrea.experiments import pair

    def h(config, harness=None):
        j = dict(instance="data/miplib/inst/glass4.mps.gz", set="x", arm="A9", seed=0, budget=120, config=config)
        if harness:
            j["harness"] = harness
        return pair.run_hash(j)
    assert h("configs/itrea2-dev-rc-3way.yaml", "ext") == "28e5a5e9c725"
    assert h("configs/itrea2-dev-rc-3way.yaml") == "0f011a2094a2"


# ============================================================================================ the driver layer
class FakeAPI:
    """The five driver calls in memory: the primary context's flags and active state, every call logged."""
    name = "fake"

    def __init__(self, flags=0, active=False, ignore_set=False, rc=None):
        self.flags, self.active, self.ignore_set = flags, active, ignore_set
        self.calls: list = []

    def init(self):
        self.calls.append(("init",))

    def device_get(self, ordinal):
        self.calls.append(("device_get", ordinal))
        return 7 + ordinal

    def primary_ctx_get_state(self, dev):
        self.calls.append(("get_state", dev))
        return self.flags, self.active

    def primary_ctx_set_flags(self, dev, flags):
        self.calls.append(("set_flags", dev, flags))
        if not self.ignore_set:
            self.flags = flags


def test_apply_sets_the_flag_before_the_context_exists():
    api = FakeAPI(flags=0x10)                                   # another flag bit (LMEM_RESIZE_TO_MAX) is kept
    rec = CS.apply("blocking_sync", api=api)
    assert api.calls == [("init",), ("device_get", 0), ("get_state", 7), ("set_flags", 7, 0x14), ("get_state", 7)]
    assert rec["state"] == "set" and rec["flags_before"] == 0x10 and rec["flags_set"] == 0x14
    assert rec["active_before"] is False and rec["requested"] == CS.CU_CTX_SCHED_BLOCKING_SYNC
    api.active = True                                           # the backend created the context
    assert CS.verify(rec)["state"] == "verified" and rec["sched_active"] == "blocking_sync"
    assert CS.state_at_end(rec) == dict(flags_at_end=0x14, active_at_end=True, sched_at_end="blocking_sync")
    pub = CS.public(rec)
    assert "_api" not in pub and json.dumps(pub)                # the record is plain JSON


def test_apply_yield_and_auto():
    api = FakeAPI()
    assert CS.apply("yield", api=api)["flags_set"] == CS.CU_CTX_SCHED_YIELD
    quiet = FakeAPI()
    rec = CS.apply("auto", api=quiet)                           # the default sets nothing before the context exists
    assert quiet.calls == [] and rec["state"] == "default"
    with pytest.raises(ValueError, match="cuda_sched"):
        CS.apply("spin", api=FakeAPI())


def test_auto_reads_the_active_flags_back_and_never_raises():
    """D-025's control: under "auto" `verify` (after the context exists) only READS the primary context's flags —
    no set call — records them (`sched_active`, state "read") and the state at the end; a flag that is not auto or an
    inactive context is recorded, never raised (the report reads it as a validity leg); a driver that cannot be read
    is recorded as `read_error`."""
    api = FakeAPI(flags=0x10, active=True)
    rec = CS.verify(CS.apply("auto", api=api), api=api)
    assert [c[0] for c in api.calls] == ["init", "device_get", "get_state"]
    assert rec["state"] == "read" and rec["sched_active"] == "auto" and rec["flags_active"] == 0x10
    assert rec["active"] is True and rec["device"] == 7
    assert CS.state_at_end(rec) == dict(flags_at_end=0x10, active_at_end=True, sched_at_end="auto")
    assert "_api" not in CS.public(rec) and json.dumps(CS.public(rec))
    odd = CS.verify(CS.apply("auto"), api=FakeAPI(flags=CS.CU_CTX_SCHED_SPIN, active=False))
    assert odd["state"] == "read" and odd["sched_active"] == "spin" and odd["active"] is False

    class Broken(FakeAPI):
        def init(self):
            raise CS.CudaSchedError("cuInit(0) returned CUresult 100")
    bad = CS.verify(CS.apply("auto"), api=Broken())
    assert bad["state"] == "read_error" and "100" in bad["read_error"] and CS.state_at_end(bad) == {}


def test_apply_refuses_an_active_context_and_a_flag_that_does_not_hold():
    with pytest.raises(CS.CudaSchedError, match="already active"):
        CS.apply("blocking_sync", api=FakeAPI(active=True))
    with pytest.raises(CS.CudaSchedError, match="reads scheduling"):
        CS.apply("blocking_sync", api=FakeAPI(ignore_set=True))


def test_verify_refuses_an_inactive_or_respun_context():
    api = FakeAPI()
    rec = CS.apply("blocking_sync", api=api)
    with pytest.raises(CS.CudaSchedError, match="not active"):
        CS.verify(rec)                                          # CuPy did not use the primary context
    api.active, api.flags = True, CS.CU_CTX_SCHED_SPIN
    with pytest.raises(CS.CudaSchedError, match="spin"):
        CS.verify(rec)


def test_events_block_only_under_blocking_sync():
    assert CS.events_block("blocking_sync") and not CS.events_block("yield") and not CS.events_block("auto")


class _FakeLib:
    """A stand-in for ctypes.CDLL("libcuda.so.1"): functions that write their outputs through byref (`._obj`)."""

    def __init__(self, v2=True, rc=0):
        self.rc, self.state = rc, [0, 0]
        lib = self

        class _F:
            def __init__(self, f):
                self.f, self.restype, self.argtypes = f, None, None

            def __call__(self, *a):
                return self.f(*a)
        self.cuInit = _F(lambda flags: lib.rc)
        self.cuDeviceGet = _F(lambda p, o: (setattr(p._obj, "value", 3 + o), lib.rc)[1])
        self.cuDevicePrimaryCtxGetState = _F(lambda d, f, a: (setattr(f._obj, "value", lib.state[0]),
                                                               setattr(a._obj, "value", lib.state[1]), lib.rc)[2])
        setf = _F(lambda d, fl: (lib.state.__setitem__(0, fl), lib.rc)[1])
        if v2:
            self.cuDevicePrimaryCtxSetFlags_v2 = setf
        else:
            self.cuDevicePrimaryCtxSetFlags = setf


@pytest.mark.parametrize("v2", [True, False])
def test_driver_api_symbols_and_byref_outputs(monkeypatch, v2):
    """DriverAPI through a fake CDLL: the `_v2` setter when exported (driver 580 exports both), the outputs read
    through byref, a non-zero CUresult raised as CudaSchedError. No real driver is loaded."""
    lib = _FakeLib(v2=v2)
    monkeypatch.setattr(ctypes, "CDLL", lambda path: lib)
    api = CS.DriverAPI.load()
    assert api.set_flags_symbol == ("cuDevicePrimaryCtxSetFlags_v2" if v2 else "cuDevicePrimaryCtxSetFlags")
    rec = CS.apply("blocking_sync", api=api)
    assert rec["device"] == 3 and rec["flags_set"] == 4 and lib.state[0] == 4
    assert api.name.endswith(api.set_flags_symbol)
    lib.rc = 100                                                # CUDA_ERROR_NO_DEVICE
    with pytest.raises(CS.CudaSchedError, match="CUresult 100"):
        api.init()


def test_the_module_never_loads_the_driver_at_import():
    """`ea.cuda_sched` (imported by the engine) opens libcuda only when a non-default mode is applied."""
    code = ("import ctypes, sys; calls = []; real = ctypes.CDLL; ctypes.CDLL = lambda *a, **k: (calls.append(a) if "
            "'libcuda' in str(a) else None) or real(*a, **k); import itrea.ea.engine, itrea.ea.cuda_sched as c; "
            "c.apply('auto'); print(len(calls))")
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=120,
                         env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES=""))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "0"


# ============================================================================================ the engine
def test_engine_sets_the_flag_before_the_backend_exists(monkeypatch):
    """Engine.prepare with backend cupy: cuda_sched.apply runs BEFORE get_backend (nothing may retain the context
    first); the record enters the start-up record. The backend here is a NumPy one (no device in a CPU test), so the
    post-context `verify` of the GPU branch is not reached; the GPU check runs it."""
    order = []

    def fake_apply(mode, ordinal=0, api=None):
        order.append(("apply", mode))
        return dict(mode=mode, requested=CS.MODES[mode], state="set", _api=object())

    def fake_backend(name, gpu_mem_gb=None, allow_cpu_fallback=False):
        order.append(("backend", name))
        return Backend("numpy", np, __import__("scipy.sparse", fromlist=["x"]))
    monkeypatch.setattr(CS, "apply", fake_apply)
    monkeypatch.setattr(E, "get_backend", fake_backend)
    cfg = EAConfig().with_overrides(dict(SMALL, backend="cupy", **BSYNC))
    eng = Engine(toy.knapsack(12), cfg, seed=0, warmup=False)
    st = eng.prepare()
    assert order == [("apply", "blocking_sync"), ("backend", "cupy")]
    assert st["cuda_sched"] == dict(mode="blocking_sync", requested=4, state="set")


def test_numpy_backend_and_handed_in_backends_never_call_the_driver(monkeypatch):
    monkeypatch.setattr(CS, "apply", lambda *a, **k: pytest.fail("the driver was called"))
    eng = Engine(toy.knapsack(12), EAConfig().with_overrides(dict(SMALL, backend="numpy", **BSYNC)), seed=0,
                 warmup=False)
    assert eng.prepare()["cuda_sched"]["state"] == "inert: numpy backend"
    eng.begin(time.monotonic(), 3600.0)
    eng.step()
    out = eng.finish()
    assert "cuda_sched" not in out and out["phase_events_block"] is True
    # the default: no record field beyond the mode, and the events as before
    eng2 = Engine(toy.knapsack(12), EAConfig().with_overrides(dict(SMALL, backend="numpy")), seed=0, warmup=False)
    assert "cuda_sched" not in eng2.prepare()
    eng2.begin(time.monotonic(), 3600.0)
    assert eng2.phase.block is False


@pytest.mark.parametrize("mode, block", [("blocking_sync", True), ("yield", False), ("auto", False)])
def test_phase_events_are_blocking_under_blocking_sync(monkeypatch, mode, block):
    """_Phases creates its CUDA events with cudaEventBlockingSync under blocking_sync, and exactly as before
    (`Event()`, no argument) otherwise. A fake `cupy` module records the calls; no device is touched."""
    made = []

    class Ev:
        def __init__(self, **kw):
            made.append(kw)

        def record(self):
            pass

        def synchronize(self):
            pass
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Ev, get_elapsed_time=lambda s, e: 2.0))
    monkeypatch.setitem(sys.modules, "cupy", fake)
    ph = E._Phases(Backend("cupy", np, None), block=CS.events_block(mode))
    with ph("repair"):
        pass
    assert made == ([{"block": True}] * 2 if block else [{}, {}])
    assert ph.take() == {"repair": 0.002}


def test_renice_threads_logic(monkeypatch, tmp_path):
    """renice_threads on fake thread ids (os.getpriority / os.setpriority patched; this process is never reniced): the
    target is the main thread's nice + the value (relative to the LP worker it just spawned), every thread below it is
    set, one at or above it is kept, a thread gone meanwhile is skipped; at a high starting nice the cap at 19 shows as
    a gap below the value."""
    for tid in ("11", "12", "13", "14"):
        (tmp_path / tid).mkdir()
    (tmp_path / "x").mkdir()                                    # not a thread id
    nice = {os.getpid(): 0, 11: 0, 12: 7, 13: 15}

    def getp(which, who):
        assert which == os.PRIO_PROCESS
        if who not in nice:
            raise ProcessLookupError(who)
        return nice[who]

    def setp(which, who, value):
        nice[who] = value
    monkeypatch.setattr(os, "getpriority", getp)
    monkeypatch.setattr(os, "setpriority", setp)
    rec = renice_threads(10, task_dir=tmp_path)
    assert nice == {os.getpid(): 0, 11: 10, 12: 10, 13: 15}     # 14 ended before it was read; 13 kept at 15
    assert rec == dict(value=10, threads=4, set=2, kept=1, before=0, target=10, gap=10, after=0)
    nice.update({os.getpid(): 12, 11: 12, 12: 12, 13: 12})
    rec = renice_threads(10, task_dir=tmp_path)                 # 12 + 10 > 19: the cap
    assert nice == {os.getpid(): 12, 11: 19, 12: 19, 13: 19}
    assert rec["target"] == 19 and rec["gap"] == 7 and rec["set"] == 3


def test_renice_threads_sets_every_thread_and_never_lowers():
    """In a subprocess (never this process): two extra threads, a renice by +3 sets every thread to base + 3; after
    os.nice(2) on the main thread (base + 5) a renice by +3 targets base + 8 and sets every other thread. Skipped when
    the test process runs above nice 11."""
    code = r"""
import json, os, threading, time
from itrea.ea.engine import renice_threads
ev = threading.Event()
ts = [threading.Thread(target=ev.wait) for _ in range(2)]
[t.start() for t in ts]
b = os.getpriority(os.PRIO_PROCESS, 0)
rec = renice_threads(3)
nices = sorted({os.getpriority(os.PRIO_PROCESS, int(t)) for t in os.listdir('/proc/self/task')})
os.nice(2)
rec2 = renice_threads(3)
nices2 = sorted({os.getpriority(os.PRIO_PROCESS, int(t)) for t in os.listdir('/proc/self/task')})
ev.set(); [t.join() for t in ts]
print(json.dumps(dict(rec=rec, nices=nices, rec2=rec2, nices2=nices2)))
"""
    if os.getpriority(os.PRIO_PROCESS, 0) > 11:
        pytest.skip("the test process runs above nice 11: no room to renice a child upward by 8")
    out = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=120,
                         env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES=""))
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout)
    b = r["rec"]["before"]
    assert r["rec"]["threads"] >= 3 and r["rec"]["set"] == r["rec"]["threads"] and r["rec"]["after"] == b + 3
    assert r["nices"] == [b + 3] and r["rec"]["gap"] == 3
    assert r["rec2"]["before"] == b + 5 and r["rec2"]["target"] == b + 8 and r["nices2"] == [b + 8]


def test_lp_worker_stamps_become_lp_event_fields():
    """The LP worker's own clock and CPU (sent with its result) as seconds since t0, its solve wall and CPU share."""
    eng = Engine(toy.knapsack(12), EAConfig().with_overrides(dict(SMALL, backend="numpy")), seed=0, warmup=False)
    eng.t0 = 100.0
    got = eng._lp_worker_stamps(dict(t_entry=101.0, t_done=104.0, cpu_entry_s=0.9, cpu_done_s=3.3, nice=0))
    assert got == dict(worker_entry_t=1.0, worker_done_t=4.0, worker_wall_s=3.0, worker_cpu_s=2.4,
                       worker_cpu_share=0.8, worker_cpu_total_s=3.3, worker_nice=0)
    assert eng._lp_worker_stamps(dict(state="running")) == {}


@pytest.mark.solver
def test_lp_worker_sends_its_own_clock_and_cpu(tmp_path):
    from itrea.ea.seeding import LPWorker
    path = toy.write_mps(toy.mixed(), tmp_path / "mixed.mps")
    t0 = time.monotonic()
    w = LPWorker(path, 10.0).start()
    try:
        end = time.monotonic() + 60
        x, info = w.poll()
        while info.get("state") == "running" and time.monotonic() < end:
            time.sleep(0.01)
            x, info = w.poll()
    finally:
        w.stop()
    assert info["state"] == "done" and x is not None
    assert t0 < info["t_entry"] <= info["t_done"] <= time.monotonic()
    assert 0.0 <= info["cpu_entry_s"] <= info["cpu_done_s"] and info["nice"] == os.getpriority(os.PRIO_PROCESS, 0)
    assert info["wall_seconds"] >= info["t_done"] - info["t_entry"]          # the EA read it after it finished


@pytest.mark.solver
def test_bnice_ea_run_renices_after_the_spawn_and_records_everything(tmp_path):
    """The EA process (NumPy backend, so no driver call) with the bnice variant (D-025's fallback (a): no context flag)
    on a toy file: the LP worker is spawned at the session's nice n, then every thread of the host takes n + 10 (capped
    at 19; `host` event, summary `host_renice` with `gap`); the `lp` event carries the worker's own clock; the summary
    has the on-clock CPU, the FJ launch count and no `cuda_sched` record (mode auto on NumPy)."""
    from itrea.trace import read_events
    path = toy.write_mps(toy.mixed(), tmp_path / "mixed.mps")
    rd = tmp_path / "run"
    pr = subprocess.run([PY, "-m", "itrea.ea.run", "--instance", str(path), "--seed", "0", "--budget", "4",
                         "--run-dir", str(rd), "--no-barrier", "--backend", "numpy",
                         "--config", str(CONFIGS / "itrea2-dev-rc-3way-bnice.yaml"),
                         "--set", "population.size=48", "--set", "population.offspring_per_gen=48"],
                        capture_output=True, timeout=300,
                        env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="",
                                 OMP_NUM_THREADS="1"))
    assert pr.returncode == 0, (pr.stdout + pr.stderr).decode(errors="replace")[-3000:]
    s = json.loads((rd / "ea.summary.json").read_text())
    assert "cuda_sched" not in s["startup"] and "cuda_sched" not in s
    assert s["startup"]["host_nice_after_lp_spawn"] == 10
    hr = s["host_renice"]
    assert hr["value"] == 10 and hr["t"] >= 0 and hr["threads"] >= 1
    assert hr["target"] == min(19, hr["before"] + 10) and hr["gap"] == hr["target"] - hr["before"]
    if hr["before"] < 19:                                       # (under `nice -n 19` nothing can be set: kept)
        assert hr["after"] == hr["target"] and hr["set"] >= 1
    else:
        assert hr["after"] == hr["before"] and hr["set"] == 0 and hr["gap"] == 0
    ev = list(read_events(rd / "ea.events.jsonl"))
    assert [e["what"] for e in ev if e["kind"] == "host"] == ["renice"]
    lp = [e for e in ev if e["kind"] == "lp" and e.get("state") == "done"]
    assert len(lp) == 1 and lp[0]["worker_done_t"] >= lp[0]["worker_entry_t"] >= 0
    assert 0 < lp[0]["worker_wall_s"] <= lp[0]["wall_seconds"] and lp[0]["worker_cpu_s"] >= 0
    assert s["lp"]["worker_done_t"] == lp[0]["worker_done_t"]
    assert s["lp"]["worker_nice"] == hr["before"]             # spawned before the renice
    c = s["cpu_clock"]
    assert c["wall_s"] > 0 and c["self_s"] > 0 and c["children_s"] > 0       # the LP worker was reaped
    assert set(c) == {"wall_s", "self_s", "main_thread_s", "children_s", "self_per_wall_s", "main_thread_per_wall_s"}
    assert s["ls_launches"] == 0                                # the CPU feasibility jump reports no launches
    assert s["concat_parts_max"] >= 0 and s["phase_events_block"] is False


def test_bsync_check_refuses_a_renice_without_its_competitor():
    """`--renice` acts after the stand-in's spawn (D-025's fallback (a)): without phase B the tool refuses before it
    touches a device."""
    pr = subprocess.run([PY, str(ROOT / "tools" / "bsync_check.py"), "--mode", "auto", "--renice", "10",
                         "--no-competitor"], capture_output=True, text=True, timeout=120,
                        env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES=""))
    assert pr.returncode == 2 and "competitor" in pr.stderr


def test_every_run_records_the_cpu_clock_and_launches():
    eng = Engine(toy.knapsack(12), EAConfig().with_overrides(dict(SMALL, backend="numpy")), seed=0, warmup=False)
    eng.prepare()
    eng.begin(time.monotonic(), 3600.0)
    eng.step()
    out = eng.finish()
    assert out["cpu_clock"]["self_s"] >= 0 and "ls_launches" in out and out["phase_events_block"] is False
    assert out["concat_parts_max"] >= 0
    assert "host_renice" not in out


# ============================================================================================ GPU checks (window)
def _gpu_ok() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


def _one_core() -> int:
    return min(os.sched_getaffinity(0))


def a9_layout_run(tmp_path, config: Path, budget: float = 15.0, extra=(), name="gen", problem=None,
                  cache: Path | None = None, backend: str = "cupy") -> tuple[dict, list[dict], Path]:
    """The EA as A9 runs it, without the other processes: pinned to ONE core with taskset, `exchange.from_cuopt` on,
    a file mailbox the test created (a cuOpt row and info waiting in it, so the A9 injection path runs), CuPy with an
    EMPTY kernel cache (`cache`), the given configuration file. Returns (summary, events, run dir). `backend` "numpy"
    runs the same harness on the CPU (its plumbing is tested there; the GPU checks use "cupy")."""
    from itrea.exchange.mailbox import Mailbox
    from itrea.problem import load
    from itrea.solvers.common import ORIGIN_SOLVER
    from itrea.trace import read_events
    from test_lp_decode import gen_problem
    p = problem if problem is not None else gen_problem()
    path = toy.write_mps(p, tmp_path / f"{name}.mps")
    prob = load(path)
    mbdir = tmp_path / f"mb_{name}"
    Mailbox(mbdir, "runner", create=True, run_id=f"r1{name}")
    cu = Mailbox(mbdir, "cuopt", run_id=f"r1{name}")
    x = np.clip(np.zeros(prob.n), prob.lb, prob.ub)
    cu.to_ea_cuopt.post(x[None, :], np.array([prob.objective(x)]), origin=np.array([ORIGIN_SOLVER], np.int64))
    cu.info_cuopt.post(incumbent=float(prob.objective(x)), dual_bound=None, status="solving", solver="cuopt")
    rd = tmp_path / f"run_{name}"
    env = {k: v for k, v in os.environ.items() if k != "CUPY_ACCELERATORS"}
    env.update(PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CUPY_CACHE_DIR=str(cache or tmp_path / f"cache_{name}"))
    cmd = ["taskset", "-c", str(_one_core()), PY, "-m", "itrea.ea.run", "--instance", str(path), "--seed", "0",
           "--budget", str(budget), "--run-dir", str(rd), "--mailbox", str(mbdir), "--run-id", f"r1{name}",
           "--no-barrier", "--backend", backend, "--gpu-mem-gb", "1", "--config", str(config),
           "--set", "exchange.from_cuopt=true", "--set", "population.size=48",
           "--set", "population.offspring_per_gen=48", "--set", "log_every=1", *extra]
    pr = subprocess.run(cmd, env=env, capture_output=True, timeout=900)
    assert pr.returncode == 0, (pr.stdout + pr.stderr).decode(errors="replace")[-4000:]
    s = json.loads((rd / "ea.summary.json").read_text())
    return s, list(read_events(rd / "ea.events.jsonl")), rd


@pytest.mark.solver
def test_a9_layout_harness_runs_on_numpy(tmp_path):
    """The GPU checks' harness (`a9_layout_run`) on the CPU, so its plumbing is known to work before a window: one
    core, the file mailbox with cuOpt's row and info read (from_cuopt), the LP worker done, the gated decoder started
    after it and offered batches; the flag inert on NumPy."""
    s, ev, _ = a9_layout_run(tmp_path, CONFIGS / "itrea2-dev-rc-3way-bsync-ld.yaml", budget=8.0, name="np",
                             backend="numpy")
    assert s["backend"] == "numpy" and s["exchange_from_cuopt"] is True and s["cuopt_inject_msgs"] >= 1
    assert s["cpus"] == [_one_core()] and s["startup"]["cuda_sched"]["state"] == "inert: numpy backend"
    assert s["lp"]["state"] == "done" and s["lp"]["worker_done_t"] >= 0
    ld = s["lp_decode"]
    assert ld["gate"] == "open" and ld["gate_why"] == "lp_done" and ld["ready"]["cpus"] == [_one_core()], ld
    assert s["lpd_offers"] >= 1 and ld["state"] != "failed"
    assert [e["what"] for e in ev if e["kind"] == "lp_decode" and e.get("what") == "gate_open"] == ["gate_open"]


@pytest.mark.gpu
@pytest.mark.parametrize("variant, sched", [("", "auto"), ("bsync", "blocking_sync"), ("bnice", "auto"),
                                            ("ysync", "yield")])
def test_cupy_cold_cache_r1_variant_in_the_a9_layout(tmp_path, variant, sched):
    """D-025's GPU checks 1 + 3 (before R1's first timed run): with an EMPTY CuPy cache, the variant's run in A9's
    layout compiles nothing after t0 — and the run reached the new code: the flag was set on the real driver and
    verified on the active primary context (cuDevicePrimaryCtxGetState) and still holds at the end — and with the
    DEFAULT file (the control) and bnice the flags read back are CUDA's default ("auto", state "read") — the timing
    events were blocking under blocking_sync only, the GPU feasibility jump ran launches, cuOpt's row was read
    (from_cuopt), and the LP worker reported with its own clock (bnice: spawned before the +10 renice). The mechanism
    figures (EA self-CPU per wall-second, the LP worker's CPU share) are printed, not asserted: the bsync_check test
    asserts the mechanism; one 15-s toy run cannot."""
    if not _gpu_ok():
        pytest.skip("no CuPy device")
    cfg = CONFIGS / (f"itrea2-dev-rc-3way-{variant}.yaml" if variant else "itrea2-dev-rc-3way.yaml")
    s, ev, _ = a9_layout_run(tmp_path, cfg, name=variant or "default")
    want = CS.MODES[sched]
    st = s["startup"]["cuda_sched"]
    if sched == "auto":
        assert st["state"] == "read" and st["active"] is True and st["sched_active"] == "auto", st
    else:
        assert st["state"] == "verified" and st["active_before"] is False and st["active"] is True, st
    assert st["flags_active"] & CS.CU_CTX_SCHED_MASK == want and st["sched_active"] == sched
    end = s["cuda_sched"]
    assert end["sched_at_end"] == sched and end["active_at_end"] is True
    print(f"\n[{variant or 'default'}] self_per_wall_s {s['cpu_clock']['self_per_wall_s']} lp worker_cpu_share "
          f"{s['lp'].get('worker_cpu_share')} worker_done_t {s['lp'].get('worker_done_t')} generations "
          f"{s['generations']} ls_launches {s['ls_launches']} concat_parts_max {s['concat_parts_max']}")
    assert s["phase_events_block"] is (sched == "blocking_sync")
    assert s["generations"] >= 2 and s["ls_launches"] >= 1 and s["exchange_from_cuopt"] is True
    assert s["cuopt_inject_msgs"] >= 1
    assert s["lp"]["state"] == "done" and s["lp"]["worker_done_t"] >= 0, s["lp"]
    if variant == "bnice":                                       # run the GPU checks without `nice`: gap 10
        hr = s["host_renice"]
        assert hr["value"] == 10 and hr["gap"] == 10 and hr["set"] >= 1 and hr["after"] == hr["before"] + 10, hr
        assert s["lp"]["worker_nice"] == hr["before"]               # the worker was spawned before the renice
    else:
        assert "host_renice" not in s
    assert s["compiled_on_clock"] is False and s["cupy_cache"]["new_after_t0"] == [], s["cupy_cache"]


def _bsync_check(tmp_path, mode: str, seconds: float = 12.0, renice: int = 0) -> dict:
    out = tmp_path / f"{mode}-n{renice}.json"
    pr = subprocess.run([PY, str(ROOT / "tools" / "bsync_check.py"), "--mode", mode, "--seconds", str(seconds),
                         "--renice", str(renice), "--core", str(_one_core()), "--out", str(out)],
                        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), capture_output=True, timeout=900)
    assert pr.returncode == 0, (pr.stdout + pr.stderr).decode(errors="replace")[-3000:]
    return json.loads(out.read_text())


@pytest.mark.gpu
def test_bsync_check_tool_host_cpu_falls_and_the_core_is_given_back(tmp_path):
    """D-025's GPU check 2 (R1's mechanism, the reading fixed before the runs): a launch loop of ~50-ms kernels synced
    as the EA syncs, on one core, >= 10 s of GPU work per phase. Mode auto (the default path, printed beside): the host
    spins (self-CPU per wall second > 0.8 alone) and a busy stand-in for the LP worker on the same core gets < 0.7 of
    it. Blocking sync: the host's self-CPU per wall second <= 0.25 alone (D-025) and the stand-in gets > 0.8 of the
    core. Fallback (b) yield and fallback (a) renice (mode auto, the host reniced +10 after the stand-in's spawn): the
    stand-in gets >= 0.8 of the core (D-025's stand-in check). Every run must have waited on the GPU (kernel >= 20 ms,
    >= 40 launches per phase alone, >= 20 beside)."""
    if not _gpu_ok():
        pytest.skip("no CuPy device")
    if os.getpriority(os.PRIO_PROCESS, 0) > 0:
        pytest.skip("run the GPU checks without `nice`: the renice check needs the host at nice 0 + 10")
    auto, bs = _bsync_check(tmp_path, "auto"), _bsync_check(tmp_path, "blocking_sync")
    yi, rn = _bsync_check(tmp_path, "yield"), _bsync_check(tmp_path, "auto", renice=10)
    for r in (auto, bs, yi, rn):
        assert r["kernel_ms"] >= 20.0 and r["alone"]["launches"] >= 40 and r["beside"]["launches"] >= 20, r
        assert r["alone"]["gpu_s"] >= 10.0 and r["seconds"] >= 10.0, r
        print(f"\n[{r['mode']} renice {r['renice'].get('value', 0) if r.get('renice') else 0}] host_self_per_wall "
              f"{r['host_self_per_wall']} competitor_share {r['competitor_share']} host_self_per_wall_b "
              f"{r['host_self_per_wall_b']}")
    assert bs["cuda_sched"]["sched_active"] == "blocking_sync" and bs["phase_events_block"] is True
    assert auto["cuda_sched"]["state"] == "read" and auto["cuda_sched"]["sched_active"] == "auto"
    assert yi["cuda_sched"]["sched_active"] == "yield" and yi["phase_events_block"] is False
    assert auto["host_self_per_wall"] > 0.8 and auto["competitor_share"] < 0.7, auto
    assert bs["host_self_per_wall"] <= 0.25 and bs["competitor_share"] > 0.8, bs
    assert yi["competitor_share"] >= 0.8, yi
    assert rn["renice"]["gap"] == 10 and rn["competitor_share"] >= 0.8, rn
