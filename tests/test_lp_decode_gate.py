"""R2 of the C1 diagnosis (vault `30-results/2026-10-14 C1 diagnosis and improvement candidates.md`): the LP decode
inside A9 — the decoder on the EA's core, started only once the EA's HiGHS LP worker has reported a final state
(`repair.lp_decode_gate: lp_worker`), at a configured nice against the host (`repair.lp_decode_nice`), the runner's
refusal lifted only for a gated configuration, and the warm-up's >8-part concatenate.

CPU tests: the two fields and their hash rule, the R2 variant files and their pinned hashes, the runner's refusal
logic, the gate in the engine (NumPy, the in-process stub decoder and a scripted LP worker: no process), the
configured nice reaching the worker (marker `solver`: a real decode worker on a toy). GPU checks (marker `gpu`,
inside a window): the >8-part concatenate compiles nothing after the warm-up, and the cold-cache test of R2's variants
in A9's layout reaching the decoded copies after the gate.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from itrea.benchmarks import toy
from itrea.config import EAConfig
from itrea.ea import engine as E
from itrea.ea import lp_decode as L
from itrea.ea.engine import Engine, _NullTrace
from itrea.problem import load

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
PY = sys.executable
SMALL = {"population.size": 48, "population.offspring_per_gen": 48, "gpu_mem_gb": 1.0}
GATE = {"repair.continuous": "lp_decode", "repair.lp_decode_gate": "lp_worker"}
# R2's variant files: (EA configuration hash, run hash of an A9 job under the ext harness; glass4, 120 s)
R2_FILES = {"bsync-ld": ("119c6e17f830", "dc45bdbb771f"), "bsync-ld0": ("b7af15f4b62a", "93a8ed06b59b"),
            "ld0": ("a05af2c732dc", "341267810eed"), "boY-bsync-ld": ("c296898cddc4", "3ef4bf1254fe"),
            "boY-bsync-ld0": ("2b6baa898edf", "37a47a123c67"), "boY-ld0": ("aabac30b5056", "58f72d467a23")}


def _cfg(name: str) -> EAConfig:
    return EAConfig.from_yaml(CONFIGS / (f"itrea2-dev-rc-3way-{name}.yaml" if name else "itrea2-dev-rc-3way.yaml"))


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


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    from test_lp_decode import gen_problem
    d = tmp_path_factory.mktemp("gate")
    path = toy.write_mps(gen_problem(), d / "gen.mps")
    return path, load(path)


# ============================================================================================ the configuration
def test_gate_and_nice_are_outside_the_hash_at_their_defaults():
    base = EAConfig()
    assert base.repair.lp_decode_gate == "none" and base.repair.lp_decode_nice == 10 == L.NICE
    assert base.with_overrides({"repair.lp_decode_gate": "none", "repair.lp_decode_nice": 10}).hash() == "9ac4a93bf93b"
    ld = EAConfig.from_yaml(CONFIGS / "itrea2-dev-rc-ld.yaml")
    assert ld.hash() == "f030e23909ad"                           # A3's default (D-017) stands
    assert len({ld.hash(), ld.with_overrides({"repair.lp_decode_gate": "lp_worker"}).hash(),
                ld.with_overrides({"repair.lp_decode_nice": 0}).hash()}) == 3


@pytest.mark.parametrize("over, what", [
    ({"repair.lp_decode_gate": "lp"}, "lp_decode_gate"),
    ({"repair.lp_decode_gate": "lp_worker"}, "needs repair.continuous lp_decode"),
    ({"repair.continuous": "lp_decode", "repair.lp_decode_nice": -1}, "lp_decode_nice"),
    ({"repair.continuous": "lp_decode", "repair.lp_decode_nice": 20}, "lp_decode_nice"),
    ({"repair.continuous": "lp_decode", "repair.lp_decode_nice": True}, "lp_decode_nice"),
    ({"repair.continuous": "lp_decode", "repair.lp_decode_nice": 1.5}, "lp_decode_nice"),
])
def test_validation_refuses_bad_gate_settings(over, what):
    with pytest.raises(ValueError, match=what):
        EAConfig().with_overrides(over)


@pytest.mark.parametrize("name", sorted(R2_FILES))
def test_r2_variant_files_are_their_base_plus_the_decode(name):
    """bsync-ld / bsync-ld0 = R1's bsync + the gated decode at nice 10 / 0; ld0 = the A9 default + the gated decode at
    nice 0 (the fallback if R1 is not adopted); the same three on D-023's boY. Pinned hashes."""
    from itrea.experiments import pair
    boy = name.startswith("boY-")
    v = name.removeprefix("boY-")
    base = ("boY-" if boy else "") + "bsync" if v.startswith("bsync") else ("boY" if boy else "")
    nice = 10 if v.endswith("-ld") else 0
    want = {"repair.continuous": "lp_decode", "repair.lp_decode_gate": "lp_worker"}
    if nice != 10:
        want["repair.lp_decode_nice"] = nice
    cfg = _cfg(name)
    assert _diff(_cfg(base), cfg) == want and cfg.repair.lp_decode_nice == nice
    j = dict(instance="data/miplib/inst/glass4.mps.gz", set="x", arm="A9", seed=0, budget=120,
             config=f"configs/itrea2-dev-rc-3way-{name}.yaml", harness="ext")
    assert (cfg.hash(), pair.run_hash(j)) == R2_FILES[name]
    pair.job_budget(j)                                           # the runner plans it (the refusal is lifted)


# ============================================================================================ the runner
def test_runner_refuses_lp_decode_in_the_single_core_arms_unless_gated(tmp_path):
    """The refusal of D-017 stands for A9 / A3x with an ungated decoder and is lifted only with the gate AND the EA's
    own LP worker (`seeding.lp_source: highs_async`; without it the gate opens at t0 and gates nothing) — R2, D-026;
    A3 is unchanged (f030e23909ad)."""
    from itrea.experiments import pair
    inst = toy.write_mps(toy.mixed(), tmp_path / "mixed.mps")
    o = pair.PairOptions(traces_dir=tmp_path / "traces", mailbox_root=tmp_path / "mb", gpu_health=False,
                         record_snapshots=False, cupy_cache_dir=tmp_path / "cc", log=lambda m: None)

    def job(arm, config="configs/itrea2-dev-rc-3way.yaml", **ov):
        return dict(instance=str(inst), set="pytest", arm=arm, seed=1000, budget=300, split="smoke", config=config,
                    overrides=ov)
    for arm in ("A9", "A3x"):
        with pytest.raises(pair.RunError, match="lp_decode_gate"):
            pair.job_budget(job(arm, **{"repair.continuous": "lp_decode"}), o)
        with pytest.raises(pair.RunError, match="lp_decode"):
            pair.job_budget(job(arm, config="configs/itrea2-dev-rc-ld.yaml"), o)
        with pytest.raises(pair.RunError, match="highs_async"):
            pair.job_budget(job(arm, **GATE, **{"seeding.lp_source": "solver"}), o)
        pair.job_budget(job(arm, **GATE), o)
        pair.job_budget(job(arm, config="configs/itrea2-dev-rc-3way-bsync-ld.yaml"), o)
    assert pair.run_hash(job("A3", **{"repair.continuous": "lp_decode"}), o) == "f030e23909ad"
    pair.job_budget(job("A3", **GATE), o)                         # A3 never refused it


# ============================================================================================ the gate in the engine
class FakeLP:
    """The EA's LP worker, scripted: "running" until `finish(state, x)`; records start / stop."""
    made: list = []

    def __init__(self, path, time_limit=60.0, cpus=None, names=None):
        self.state, self.x, self.started, self.stopped = "not_started", None, False, False
        FakeLP.made.append(self)

    def start(self):
        self.started, self.state = True, "running"
        return self

    def finish(self, state="done", x=None):
        self.state, self.x = state, x

    def poll(self):
        info = dict(state=self.state, wall_seconds=1.0)
        if self.state == "done":
            info.update(objective=0.0, t_entry=0.0, t_done=0.5, cpu_entry_s=0.1, cpu_done_s=0.5)
        return (self.x if self.state == "done" else None), info

    def stop(self):
        self.stopped = True


class StartStub(E._WarmupDecoder):
    """The in-process stub decoder, recording when it was started."""

    def start(self):
        self.started_gen = self.eng.gen
        return super().start()


class RecTrace(_NullTrace):
    def __init__(self):
        super().__init__()
        self.events: list = []

    def event(self, kind, t=None, **data):
        self.events.append((kind, data))


def _engine(files, monkeypatch, **over):
    FakeLP.made = []
    monkeypatch.setattr(E, "LPWorker", FakeLP)
    path, prob = files
    d = dict(SMALL, backend="numpy", **{"seeding.lp_source": "highs_async"}, **GATE)
    d.update(over)
    cfg = EAConfig().with_overrides(d)
    tr = RecTrace()
    eng = Engine(prob, cfg, seed=3, trace=tr, warmup=False, decoder_factory=StartStub)
    eng.prepare()
    eng.begin(time.monotonic(), 3600.0)
    return eng, tr


def _tick_step(eng, n=1):
    for _ in range(n):
        eng.next_poll = -math.inf                                # the exchange tick is due in every generation
        eng.step()


def _lp_point(prob) -> np.ndarray:
    lb = np.where(np.isfinite(prob.lb), prob.lb, 0.0)
    ub = np.where(np.isfinite(prob.ub), prob.ub, lb + 1.0)
    return lb + 0.4 * (ub - lb)


def test_the_decoder_waits_for_the_lp_worker_then_starts(files, monkeypatch):
    """While the LP worker runs: the decoder is not started, nothing is offered, the pick site counts `lpd_gated`.
    The tick that reads the LP worker's result opens the gate (event, time, reason), the decoder starts and decoded
    copies follow."""
    eng, tr = _engine(files, monkeypatch)
    lp = FakeLP.made[0]
    assert lp.started and eng.lpd.state == "not_started" and eng.lpd_gate == "waiting"
    _tick_step(eng, 4)
    assert eng.lpd.state == "not_started" and eng.c["lpd_gated"] == 4 and eng.c["lpd_offers"] == 0
    assert not [d for k, d in tr.events if k == "lp_decode"]
    lp.finish("done", _lp_point(eng.prob))
    _tick_step(eng)
    assert eng.lpd_gate == "open" and eng.lpd_gate_why == "lp_done" and eng.lpd.started_gen == eng.gen
    assert eng.lp_x is not None
    gate = [d for k, d in tr.events if k == "lp_decode" and d.get("what") == "gate_open"]
    assert len(gate) == 1 and gate[0]["why"] == "lp_done" and gate[0]["gate_t"] == eng.lpd_gate_t >= 0
    _tick_step(eng, 6)
    assert eng.c["lpd_offers"] >= 1 and eng.c["lpd_gated"] == 4
    out = eng.finish()
    s = out["lp_decode"]
    assert s["gate"] == "open" and s["gate_why"] == "lp_done" and s["gate_t"] == eng.lpd_gate_t
    assert s["nice"] == 10 and out["lpd_gated"] == 4


@pytest.mark.parametrize("state", ["no_solution", "failed", "stopped"])
def test_the_gate_opens_on_every_final_state(files, monkeypatch, state):
    eng, _ = _engine(files, monkeypatch)
    _tick_step(eng)
    FakeLP.made[0].finish(state)
    _tick_step(eng)
    assert eng.lpd_gate == "open" and eng.lpd_gate_why == f"lp_{state}" and eng.lp_worker is None
    assert eng.lpd.state in ("starting", "idle", "busy")


def test_without_an_lp_worker_the_gate_opens_at_t0(files, monkeypatch):
    """`seeding.lp_source` other than highs_async: no LP worker to wait for — the decoder starts at t0."""
    eng, tr = _engine(files, monkeypatch, **{"seeding.lp_source": "solver"})
    assert not FakeLP.made and eng.lpd_gate == "open" and eng.lpd_gate_why == "no_lp_worker"
    assert eng.lpd.started_gen == 0 and eng.lpd_gate_t is not None


def test_a_gate_that_never_opens_is_visible(files, monkeypatch):
    """The LP worker runs to the end (its 60-s limit beyond the budget): the decoder never starts, and the summary says
    so — state not_started, gate waiting, no offer, every pick site counted."""
    eng, _ = _engine(files, monkeypatch)
    _tick_step(eng, 3)
    out = eng.finish()
    s = out["lp_decode"]
    assert s["state"] == "not_started" and s["gate"] == "waiting" and s["gate_t"] is None
    assert out["lpd_gated"] == 3 and out["lpd_offers"] == 0 and out["lpd_lost"] == 0
    assert FakeLP.made[0].stopped


def test_no_gate_starts_at_t0_as_before(files, monkeypatch):
    eng, _ = _engine(files, monkeypatch, **{"repair.lp_decode_gate": "none"})
    assert eng.lpd_gate == "none" and eng.lpd.started_gen == 0 and "lpd_gated" not in eng.c
    out = eng.finish()
    assert "lpd_gated" not in out and out["lp_decode"]["gate"] == "none"


def test_warmup_overrides_keep_the_gated_and_reniced_configs_valid(files, tmp_path):
    """The warm-up runs with `seeding.lp_source: solver` and (inert lp_decode) `repair.continuous: auto`: its overrides
    switch the renice and the gate off with them, so a variant file never fails its own warm-up."""
    path, prob = files
    knap = load(toy.write_mps(toy.knapsack(12), tmp_path / "knap12.mps"))     # no continuous column: lp_decode inert
    for name in ("bnice", "bsync-ld", "ld0", "boY-bsync-ld0"):
        cfg = _cfg(name).with_overrides(dict(SMALL, backend="numpy"))
        assert Engine(prob, cfg, seed=0).prepare()["warmup"]["many"]["generations"] >= 1
        st = Engine(knap, cfg, seed=0).prepare()
        assert st.get("lp_decode", {}).get("state") in (None, "inert")


def test_the_configured_nice_reaches_the_decoder(files):
    path, prob = files
    eng = Engine(prob, EAConfig().with_overrides(dict(SMALL, backend="numpy", **GATE,
                                                      **{"repair.lp_decode_nice": 3})), seed=0, warmup=False)
    eng.prepare()
    eng.deadline = math.inf
    dec = E._lp_decoder_for(eng)
    assert dec.nice == 3 and dec.state == "not_started"
    assert L.LPDecoder(path, "d", "int8", 4, 2.0, 10.0, 1e-6).nice == L.NICE


@pytest.mark.solver
@pytest.mark.parametrize("nice", [0, 3])
def test_the_decode_worker_runs_at_the_hosts_nice_plus_its_increment(files, nice):
    """A real decode worker on a toy: its ready record's nice = this process's + the increment (capped at 19), on
    this process's cores."""
    path, prob = files
    ints = np.asarray(prob.ints)
    dt = L.int_part_dtype(np.asarray(prob.lb)[ints], np.asarray(prob.ub)[ints])
    dec = L.LPDecoder(path, L.model_digest(prob.names, prob.isint), dt, 4, 0.5, 1.0, 1e-6, nice=nice).start()
    try:
        end = time.monotonic() + 60
        while dec.state == "starting" and time.monotonic() < end:
            dec.poll()
            time.sleep(0.01)
        assert dec.state == "idle", dec.error
        r = dec.ready_info
        assert r["nice"] == min(19, os.getpriority(os.PRIO_PROCESS, 0) + nice)
        assert r["cpus"] == sorted(os.sched_getaffinity(0))
    finally:
        dec.stop()


class LateLP(FakeLP):
    """An LP worker that reports "done" at its third poll."""

    def poll(self):
        self.polls = getattr(self, "polls", 0) + 1
        if self.polls >= 3 and self.state == "running":
            lb = np.where(np.isfinite(self.prob.lb), self.prob.lb, 0.0)
            self.finish("done", lb.copy())
        return super().poll()


@pytest.mark.parametrize("mode", ["active", "forced"])
def test_backoff_and_the_gated_decode_together(files, monkeypatch, tmp_path, mode):
    """R2 on D-023's boY (`boY-bsync-ld`): lp_decode runs beside an ACTIVE back-off for the first time (`_bo_pace`,
    `_lpd_poll`, the decode's device sync). NumPy, SCIP and cuOpt in memory (cuOpt's incumbent arrives at 0.3 s, so
    phase Y yields first), the LP worker scripted to finish at its third poll, the stub decoder. Mode "active" is the
    file as written; "forced" holds a 1/2 duty for the whole run, so pauses and decode ticks interleave throughout.
    The run ends within its budget, the gate opens on the LP worker, batches are offered and answered while the
    back-off paces (the stub's copies carry no gain on these sources, so none is kept: the path, not the payoff)."""
    from itrea.trace import Trace, read_events
    from test_backoff import FAST, Partners, _Queue, _msg
    path, prob = files
    LateLP.prob = prob
    monkeypatch.setattr(E, "LPWorker", LateLP)
    over = dict(SMALL, backend="numpy", **FAST, **{"exchange.from_cuopt": True, "exchange.poll_s": 0.01,
                                                   "log_every": 1})
    if mode == "forced":
        over.update({"backoff.mode": "forced", "backoff.forced_duty": 0.5})
    cfg = _cfg("boY-bsync-ld").with_overrides(over)
    mb = Partners(prob, scip_inc=None)
    mb.lps = _Queue()
    x = np.where(np.isfinite(prob.lb), prob.lb, 0.0)
    mb.to_ea_cuopt.items.append(_msg(x, prob.objective(x)))
    tr = Trace(tmp_path, "ea")
    eng = Engine(prob, cfg, seed=5, mailbox=mb, trace=tr, decoder_factory=StartStub)
    eng.prepare()
    t0 = time.monotonic()
    import threading
    threading.Timer(0.3, lambda: mb.info_cuopt.post(incumbent=float(prob.objective(x)), dual_bound=None,
                                                    status="solving", solver="cuopt")).start()
    out = eng.run(2.0, t0)
    t_end = time.monotonic()
    tr.close()
    assert t_end - t0 <= 2.0 + 0.5
    bo, ld = out["backoff"], out["lp_decode"]
    assert bo["mode"] == mode
    if mode == "forced":                                       # paced from t0 to the end
        assert bo["paused_gen_s"] > 0 and bo["paused_call_s"] >= 0 and bo["busy_share"] < 0.8, bo
    assert ld["gate"] == "open" and ld["gate_why"] == "lp_done" and out["lpd_gated"] >= 1, ld
    assert out["lpd_offers"] >= 1 and out["lpd_replies"] >= 1 and out["lpd_failed"] == 0, out["lpd_offers"]
    assert out["lpd_solved"] == out["lpd_ok"] == out["lpd_no_gain"] + out["lpd_kept"] and ld["host_s"] > 0
    ev = read_events(tmp_path / "ea.events.jsonl")
    assert [e["what"] for e in ev if e["kind"] == "lp_decode" and e.get("what") == "gate_open"] == ["gate_open"]
    if mode == "active":                                       # phase Y ran and ended (own find, peek or tick)
        assert any(e["kind"] == "backoff" for e in ev)
        assert bo["phase_y"]["exit_t"] is not None and bo["phase_y"]["reason"] != "never", bo["phase_y"]


def test_warm_concat_runs_on_every_dtype_of_a_candidate_set(files, monkeypatch):
    """The warm-up's many-part concatenate (`_warm_concat`, GPU only in a run) on NumPy here: CONCAT_WARM_PARTS (9,
    above CuPy's 8) parts of X in the gene dtype (2-D), key / V (float64) and feasible (bool), all one-row and of
    mixed lengths."""
    path, prob = files
    seen = []
    real = np.concatenate

    def spy(parts, *a, **k):
        seen.append((len(parts), parts[0].dtype, parts[0].ndim, tuple(p.shape[0] for p in parts)))
        return real(parts, *a, **k)
    eng = Engine(prob, EAConfig().with_overrides(dict(SMALL, backend="numpy")), seed=0, warmup=False)
    eng.prepare()
    monkeypatch.setattr(np, "concatenate", spy)
    assert eng._warm_concat() == E.CONCAT_WARM_PARTS == 9
    assert len(seen) == 6 and all(n == 9 for n, *_ in seen)
    assert {(str(d), nd) for _, d, nd, _ in seen} == {(eng.gene_dtype.name, 2), ("float64", 1), ("bool", 1)}
    assert {rows for *_, rows in seen} == {(1,) * 9, (3,) + (1,) * 8}


# ============================================================================================ GPU checks (window)
def _gpu_ok() -> bool:
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


CONCAT_PROBE = r"""
import json, os, sys
import numpy as np
from itrea.config import EAConfig
from itrea.ea.engine import Engine
from itrea.ea.population import Population
from itrea.problem import load
prob = load(sys.argv[1])
backend = sys.argv[2] if len(sys.argv) > 2 else "cupy"
cache = os.environ["CUPY_CACHE_DIR"]
os.makedirs(cache, exist_ok=True)
count = lambda: sum(len(f) for _, _, f in os.walk(cache))
eng = Engine(prob, EAConfig().with_overrides({"population.size": 48, "population.offspring_per_gen": 48,
                                              "gpu_mem_gb": 1.0, "backend": backend}), seed=0)
st = eng.prepare()
before = count()
eng.begin(__import__("time").monotonic(), 3600.0)
eng.step()
mid = count()
pop = eng.pop
one = [pop.take(eng.bk, np.array([i % pop.size])) for i in range(12)]
for k in (9, 12):                                   # all one-row parts, then mixed lengths
    Population.concat(eng.bk, one[:k])
    Population.concat(eng.bk, [pop] + one[:k - 1])
eng.bk.synchronize()
print(json.dumps(dict(warm=st["warmup"].get("concat_many_parts"), before=before, mid=mid, after=count(),
                      gene_dtype=st["gene_dtype"])))
"""


def test_concat_probe_runs_on_numpy(tmp_path):
    """The GPU check's probe script (`CONCAT_PROBE`) on NumPy, so the script itself is known to run before a window:
    no warm-up entry on the CPU (the many-part concatenate is a GPU warm-up), the 9- and 12-part concatenates run."""
    from test_lp_decode import gen_problem
    path = toy.write_mps(gen_problem(), tmp_path / "gen.mps")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="",
               CUPY_CACHE_DIR=str(tmp_path / "cc"))
    pr = subprocess.run([PY, "-c", CONCAT_PROBE, str(path), "numpy"], env=env, capture_output=True, timeout=600)
    assert pr.returncode == 0, (pr.stdout + pr.stderr).decode(errors="replace")[-3000:]
    r = json.loads(pr.stdout.decode().strip().splitlines()[-1])
    assert r["warm"] is None and r["before"] == r["mid"] == r["after"] == 0 and r["gene_dtype"] == "float64"


@pytest.mark.gpu
def test_cupy_concat_of_more_than_eight_parts_is_warmed(tmp_path):
    """GPU check (R2): with an EMPTY CuPy cache, a candidate set of 9 and 12 parts — one-row parts and mixed lengths,
    every device array of a Population (X in the run's gene dtype, key, V, feasible) — compiles nothing after the
    warm-up (and after one generation). Reaches the pointer-array path by construction."""
    if not _gpu_ok():
        pytest.skip("no CuPy device")
    from test_lp_decode import gen_problem
    for name, p in (("gen", gen_problem()), ("knap", toy.knapsack(30))):     # float64 genes, int8 genes
        path = toy.write_mps(p, tmp_path / f"{name}.mps")
        env = {k: v for k, v in os.environ.items() if k != "CUPY_ACCELERATORS"}
        env.update(PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="1", CUPY_CACHE_DIR=str(tmp_path / f"cc_{name}"))
        pr = subprocess.run([PY, "-c", CONCAT_PROBE, str(path)], env=env, capture_output=True, timeout=900)
        assert pr.returncode == 0, (pr.stdout + pr.stderr).decode(errors="replace")[-3000:]
        r = json.loads(pr.stdout.decode().strip().splitlines()[-1])
        assert r["warm"] == 9 and r["before"] > 0, r
        assert r["after"] == r["mid"] == r["before"], (name, r)


@pytest.mark.gpu
@pytest.mark.parametrize("variant, nice", [("bsync-ld", 10), ("bsync-ld0", 0), ("ld0", 0)])
def test_cupy_cold_cache_r2_variant_in_the_a9_layout(tmp_path, variant, nice):
    """GPU check (R2, before its first timed run): with an EMPTY CuPy cache, the variant in A9's layout (the EA alone on
    one core with `exchange.from_cuopt` and a file mailbox) compiles nothing after t0 — and reached the decoded copies
    AFTER the gate: the LP worker finished, the gate opened on it, the decoder (on the EA's one core, at the host's
    nice + `nice`) was ready after the gate and its copies entered the population."""
    if not _gpu_ok():
        pytest.skip("no CuPy device")
    from test_cuda_sched import _one_core, a9_layout_run
    s, ev, _ = a9_layout_run(tmp_path, CONFIGS / f"itrea2-dev-rc-3way-{variant}.yaml", budget=20.0, name=variant)
    ld = s["lp_decode"]
    assert s["startup"]["lp_decode"]["gate"] == "lp_worker" and s["startup"]["lp_decode"]["nice"] == nice
    assert s["startup"]["warmup"]["concat_many_parts"] == 9
    assert s["lp"]["state"] == "done" and ld["gate"] == "open" and ld["gate_why"] == "lp_done", (s["lp"], ld)
    assert ld["ready_t"] is not None and ld["ready_t"] >= ld["gate_t"] >= 0
    assert ld["gate_t"] >= s["lp"]["worker_done_t"] >= 0, (ld["gate_t"], s["lp"])   # D-026 check 2, on the stamps
    assert ld["ready"]["cpus"] == [_one_core()]
    assert ld["ready"]["nice"] == min(19, os.getpriority(os.PRIO_PROCESS, 0) + nice)
    assert s["lpd_entered"] >= 1 and ld["state"] != "failed", ld
    gate = [e for e in ev if e["kind"] == "lp_decode" and e.get("what") == "gate_open"]
    first = [e for e in ev if e["kind"] == "lp_decode" and e.get("what") == "result"]
    assert len(gate) == 1 and first and first[0]["t"] >= gate[0]["t"]
    assert s["exchange_from_cuopt"] is True and s["cuopt_inject_msgs"] >= 1
    if variant.startswith("bsync"):
        assert s["cuda_sched"]["sched_at_end"] == "blocking_sync"
    # D-026 check 1 is met by this run (decoded copies entered, nothing compiled after t0) together with the concat
    # probe above (the > 8-part kernels are warm by construction): a toy run need not reach 9 parts, so it is printed
    print(f"\n[{variant}] concat_parts_max {s['concat_parts_max']} gate lag {ld['gate_t'] - s['lp']['worker_done_t']:.3f} s"
          f" ready_t {ld['ready_t']} lpd_entered {s['lpd_entered']}")
    assert s["compiled_on_clock"] is False and s["cupy_cache"]["new_after_t0"] == [], s["cupy_cache"]
