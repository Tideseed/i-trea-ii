"""Probe the LP decode worker on stored vectors before a D-017 window (vault D-017 §7 "Before the window"; issue #27).

For every instance: the run directory `<instance>__<variant>__s<seed>__*` under --traces (A3@rc1 seed 0 by default) is
read — READ-ONLY — for its stored vectors (ea.incumbents.npz, solver.incumbents.npz, ea.frontier.npz) and the instance
file named in its ea.summary.json. The real `LPDecoder` worker (`src/itrea/ea/lp_decode.py`: 1 HiGHS thread, niced,
OMP/OPENBLAS/MKL = 1, CUDA_VISIBLE_DEVICES="", inheriting this process's affinity) solves the distinct integer parts of
those vectors, in their stored order (incumbents by time, then the frontier), as polish (F) rows in batches of
--per-gen. Reported per instance: worker ready time, the priming LP's status and seconds, warm LP seconds (median,
p90, max), time-limit hits, worker maxrss and CPU seconds, codes, and the key gain of each copy over its stored vector
(relative to max(1, |key|)). Nothing is written unless --json is given.

Gates (D-017 §7), evaluated where they apply:
  - ca: priming LP "Optimal" within the cold limit (10 s);
  - every instance: median warm LP <= 1 s;
  - no spurious time-limit hit: no LP stopped by "Time limit reached" in less than 90 % of its own limit (HiGHS's
    limit is cumulative; the worker rebases it at every solve), counted separately after > 2 s of cumulative run time;
  - ca: worker maxrss <= 1.5 GB.

Run it pinned, on one core outside the GPU lane, inside the probe's time budget (the tool stops itself at --max-s):
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 taskset -c 15 .venv/bin/python tools/probe_lp_decode.py \\
        --traces traces/raw/dev --instances ca__Scenario400_reserves_1 sing44 radiationm18-12-05 glass4 \\
        item_placement_9903 --max-s 120
Exit code 1 if an evaluated gate fails, 2 on bad arguments / missing traces, else 0.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from itrea import paths  # noqa: E402
from itrea.ea import lp_decode as L  # noqa: E402
from itrea.problem import load  # noqa: E402
from itrea.trace import load_incumbents  # noqa: E402

DEFAULT_INSTANCES = ("ca__Scenario400_reserves_1", "sing44", "radiationm18-12-05", "glass4", "item_placement_9903")
GATE_COLD_S = 10.0
GATE_WARM_MEDIAN_S = 1.0
GATE_RSS_MB = 1.5 * 1024
CUMULATIVE_S = 2.0


def find_run(traces: Path, instance: str, variant: str, seed: int) -> Path | None:
    hits = sorted(traces.glob(f"{instance}__{variant}__s{seed}__*"))
    return hits[0] if hits else None


def instance_path(run_dir: Path, traces: Path, data_root: Path | None) -> Path:
    """The instance file of a run: absolute as recorded, else relative to --data-root, the repository that holds the
    traces (traces/raw/<set> -> its root), or this repository."""
    s = json.loads((run_dir / "ea.summary.json").read_text())
    rec = Path(s.get("instance") or json.loads((run_dir / "run.json").read_text())["instance"])
    if rec.is_absolute():
        return rec
    for root in [r for r in (data_root, traces.resolve().parents[2] if len(traces.resolve().parents) > 2 else None,
                             paths.ROOT) if r is not None]:
        if (root / rec).is_file():
            return root / rec
    return paths.resolve(rec)


def stored_vectors(run_dir: Path, names: list[str]) -> list[tuple[str, np.ndarray]]:
    """(label, x in `names` order): EA and solver incumbents by time, then the frontier rows."""
    pos = {nm: j for j, nm in enumerate(names)}
    out = []

    def reorder(x, own):
        if list(own) == list(names):
            return np.asarray(x, float)
        y = np.zeros(len(names))
        for j, nm in enumerate(own):
            y[pos[nm]] = x[j]
        return y
    recs = []
    for role in ("ea", "solver"):
        f = run_dir / f"{role}.incumbents.npz"
        if f.exists():
            own, rr = load_incumbents(f)
            recs += [(t, f"{role}@{t:.2f}s", reorder(x, own)) for t, _, _, x in rr]
    out += [(lab, x) for _, lab, x in sorted(recs, key=lambda r: r[0])]
    f = run_dir / "ea.frontier.npz"
    if f.exists():
        with np.load(f) as z:
            own = [str(v) for v in z["names"]] if "names" in z.files else names
            out += [(f"frontier[{k}]", reorder(z["X"][k], own)) for k in range(z["X"].shape[0])]
    return out


def probe_instance(name: str, run_dir: Path, inst: Path, args, budget_s: float) -> dict:
    t_start = time.monotonic()
    res: dict = dict(instance=name, run_dir=str(run_dir), file=str(inst))
    t = time.monotonic()
    prob = load(inst)
    res["parse_s"] = round(time.monotonic() - t, 3)
    res["n"], res["n_int"], res["n_cont"] = prob.n, int(prob.ints.size), int(prob.conts.size)
    res["active_columns"] = int(L.active_columns(prob).size)
    ints = prob.ints
    vecs = stored_vectors(run_dir, list(prob.names))
    seen, rows, labels, keys = set(), [], [], []
    for lab, x in vecs:
        xi = np.clip(np.round(x[ints]), prob.lb[ints], prob.ub[ints])
        h = xi.tobytes()
        if h in seen:
            continue
        seen.add(h)
        rows.append(xi)
        labels.append(lab)
        keys.append(prob.key(x))
        if len(rows) >= args.max_rows:
            break
    res["vectors"], res["distinct_int_parts"] = len(vecs), len(rows)
    if not rows:
        res["error"] = "no stored vector"
        return res
    dt = L.int_part_dtype(prob.lb[ints], prob.ub[ints])
    dec = L.LPDecoder(inst, L.model_digest(prob.names, prob.isint), dt, args.per_gen, args.time_limit,
                      args.cold_limit, 1e-6, math.inf).start()
    per_row = []
    stats: dict = {}
    try:
        end = t_start + budget_s
        while dec.state == "starting" and time.monotonic() < end:
            dec.poll()
            time.sleep(0.01)
        if dec.state != "idle":
            res["error"] = f"worker not ready: state {dec.state}, {dec.error}"
            return res
        res["ready_s"] = dec.ready_info.get("ready_after_s")
        res["worker_read_s"] = dec.ready_info.get("read_s")
        req = 0
        R = np.vstack(rows)
        for s in range(0, len(rows), args.per_gen):
            if time.monotonic() >= end:
                res["stopped_at_budget"] = True
                break
            req += 1
            batch = R[s:s + args.per_gen]
            rt_before = float(stats.get("run_time", 0.0))
            if not dec.offer(req, batch.astype(dt), np.zeros(batch.shape[0], np.uint8)):
                res["error"] = f"offer refused: state {dec.state}, {dec.error}"
                break
            rep = None
            while time.monotonic() < end + 60.0:          # an LP already started may overrun the budget by its limit
                for m in dec.poll():
                    if m[0] == "result":
                        rep = m[2]
                    elif m[0] == "failed":
                        res["error"] = f"worker failed: {m[1]}"
                if rep is not None or dec.state in ("failed", "stopped"):
                    break
                time.sleep(0.005)
            if rep is None:
                res.setdefault("error", "no reply")
                break
            stats = rep["stats"]
            ok_pos = {int(r): j for j, r in enumerate(rep["ok_rows"])}
            for r in range(batch.shape[0]):
                i = s + r
                code = L.CODE_NAMES[int(rep["codes"][r])]
                row = dict(label=labels[i], code=code, status=rep["status"][r], s=float(rep["secs"][r]),
                           limit=float(rep["limit"][r]), iters=int(rep["iters"][r]), run_time_before=rt_before)
                if r in ok_pos:
                    y = np.zeros(prob.n)
                    y[ints] = batch[r]
                    y[prob.conts] = np.clip(rep["conts"][ok_pos[r]], prob.lb[prob.conts], prob.ub[prob.conts])
                    k_new, k_old = float(prob.key(y)), float(keys[i])
                    row.update(key_stored=k_old, key_copy=k_new, gain=(k_old - k_new) / max(1.0, abs(k_old)),
                               check=bool(prob.check(y).feasible))
                per_row.append(row)
                rt_before = rt_before + row["s"]
    finally:
        dec.stop()
    res["worker_gone"] = _gone(dec.pid) if dec.pid else None
    solved = [r for r in per_row if r["status"]]
    prim = solved[0] if solved else None
    warm = [r["s"] for r in solved[1:]]
    tl = [r for r in solved if r["status"] == "Time limit reached"]
    spurious = [r for r in tl if r["s"] < 0.9 * r["limit"]]
    res.update(priming=dict(status=prim["status"], s=prim["s"], limit=prim["limit"]) if prim else None,
               solves=len(solved), warm_median_s=float(np.median(warm)) if warm else None,
               warm_p90_s=float(np.percentile(warm, 90)) if warm else None, warm_max_s=max(warm) if warm else None,
               time_limit_hits=len(tl), spurious_time_limit_hits=len(spurious),
               spurious_after_cumulative=sum(1 for r in spurious if r["run_time_before"] > CUMULATIVE_S),
               codes={c: sum(1 for r in per_row if r["code"] == c) for c in L.CODE_NAMES if
                      any(r["code"] == c for r in per_row)},
               gains=[r["gain"] for r in per_row if "gain" in r],
               copies_feasible=sum(1 for r in per_row if r.get("check")),
               worker=dict(maxrss_mb=stats.get("ru_maxrss_mb"), cpu_s=stats.get("cpu_s"),
                           run_time=stats.get("run_time"), iterations=stats.get("iterations"),
                           restores=stats.get("restores"), cache_hits=stats.get("cache_hits")),
               wall_s=round(time.monotonic() - t_start, 3))
    g = res["gains"]
    res["gain_median"] = float(np.median(g)) if g else None
    res["gain_max"] = float(np.max(g)) if g else None
    if args.rows:
        res["rows"] = per_row
    return res


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def gates(res: dict) -> dict:
    out = {}
    if res.get("error"):
        out["completed"] = False
    pr = res.get("priming")
    if res["instance"].startswith("ca__") and pr is not None:
        out["ca_priming_optimal_within_10s"] = pr["status"] == "Optimal" and pr["s"] <= GATE_COLD_S
        rss = (res.get("worker") or {}).get("maxrss_mb")
        out["ca_worker_rss_le_1.5GB"] = rss is not None and rss <= GATE_RSS_MB
    if res.get("warm_median_s") is not None:
        out["warm_median_le_1s"] = res["warm_median_s"] <= GATE_WARM_MEDIAN_S
    if pr is not None:
        out["no_spurious_time_limit_hit"] = res.get("spurious_time_limit_hits", 0) == 0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--traces", default=str(paths.TRACES / "raw" / "dev"), help="run directories (read-only)")
    ap.add_argument("--data-root", default=None, help="repository root for relative instance paths")
    ap.add_argument("--instances", nargs="+", default=list(DEFAULT_INSTANCES))
    ap.add_argument("--variant", default="A3@rc1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-gen", type=int, default=4)
    ap.add_argument("--time-limit", type=float, default=2.0)
    ap.add_argument("--cold-limit", type=float, default=10.0)
    ap.add_argument("--max-rows", type=int, default=64, help="distinct integer parts per instance, at most")
    ap.add_argument("--max-s", type=float, default=120.0, help="wall budget of the whole probe")
    ap.add_argument("--rows", action="store_true", help="keep every row in the JSON")
    ap.add_argument("--json", default=None, help="write the results here")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    if args.per_gen < 1 or args.time_limit <= 0 or args.cold_limit < args.time_limit or args.max_s <= 0:
        print("bad arguments: per-gen >= 1, time-limit > 0, cold-limit >= time-limit, max-s > 0", file=sys.stderr)
        return 2
    traces = Path(args.traces)
    data_root = Path(args.data_root) if args.data_root else None
    todo = []
    for name in args.instances:
        rd = find_run(traces, name, args.variant, args.seed)
        if rd is None:
            print(f"{name}: no run directory {name}__{args.variant}__s{args.seed}__* under {traces}", file=sys.stderr)
            continue
        todo.append((name, rd, instance_path(rd, traces, data_root)))
    if not todo:
        return 2
    t0 = time.monotonic()
    results, failed = [], False
    for k, (name, rd, inst) in enumerate(todo):
        left = args.max_s - (time.monotonic() - t0)
        if left <= 1.0:
            results.append(dict(instance=name, error="probe budget exhausted before this instance"))
            continue
        res = probe_instance(name, rd, inst, args, left / (len(todo) - k))
        res["gates"] = gates(res)
        failed |= not all(res["gates"].values())
        results.append(res)
        pr = res.get("priming") or {}
        print(f"{name}: n={res.get('n')} active={res.get('active_columns')} rows={res.get('distinct_int_parts')} "
              f"ready={res.get('ready_s')} priming={pr.get('status')}/{pr.get('s', float('nan')):.3f}s "
              f"warm median={res.get('warm_median_s')} max={res.get('warm_max_s')} "
              f"tl_hits={res.get('time_limit_hits')} spurious={res.get('spurious_time_limit_hits')} "
              f"rss={(res.get('worker') or {}).get('maxrss_mb')}MB gain median={res.get('gain_median')} "
              f"max={res.get('gain_max')} codes={res.get('codes')} gates={res['gates']}"
              + (f" ERROR {res['error']}" if res.get("error") else ""), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(args=vars(args), results=results), indent=1, default=float))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
