"""GAMS `.trc` export of our runs (D-021 R16): one trace per run and per clock, so anyone can score us with GAMS's own
script (`calculate_primal_integral` in data/mipfeas/gams-2026-09-16/create_primalintegral.py) — a test requires it to
reproduce our `pt_gams` to 1e-12.

A trace holds GAMS's six fields per record — lineNum, seriesID, node, seconds, bestFound, bestBound — after two `*`
header lines (their reader skips `*` as a comment): one `S` record at 0 s, one `I` record per improving CHECKED
incumbent of the run's portfolio (the trajectory every metric uses: `build_runs.role_steps` merged over the roles,
the exact checker's recomputed objective) stamped before T on that clock, and one `E` record. Seconds are on the call
clock (since t0, the run's own `pt_gams`) or the launch clock (+ `startup_offset_s`, `pt_gams_launch`). Floats are
written with `repr` (the shortest decimal that round-trips), so `pandas.read_csv(..., float_precision="round_trip")`
reads back exactly the doubles our metric used. GAMS's script calls `read_csv` with pandas' default parser, which is
not round-trip exact: values come back within about 1e-12 relative (measured: up to 9.8e-13 on objectives of
magnitude 1e-5 to 1e6, and 5.7e-14 s on the seconds, pandas 3.0.3), not bit for bit — so their score of our trace
agrees with our `pt_gams` to ~1e-12 (the test's tolerance), and could differ by up to 1e-6 · Δt / T only where an
objective lies within ~1e-12 relative of a boundary of GAMS's 6-decimal gamma rounding. Choices, each recorded in the
header:
- the `E` record is at T (our runs are held to T, R12) unless the caller gives an end time AND the final gamma is <= 0;
  GAMS's script raises on an `E` before T − 1 whose gamma is above 0, and an incumbent after T changes nothing in
  their integral, so neither is written;
- bestBound is `na` (only GAMS's optimal-solution indicator reads it, not the integral); node is 0;
- minimisation only (their script assumes it; every MIPfeas instance minimises): a maximisation run is refused;
- a checked incumbent better than z* (`beats_ref`; γ = 0 in our port) is written as it is — GAMS's script raises on
  it unless the trace is on its whitelist, where it sets the objective to z* — and flagged in the header.

File names follow GAMS's `<instance>_<platform>_<solver>.trc`, which their `process_directory` splits from the right:
`<instance>_gb10-f-<clock>_<arm label>-s<seed>.trc` (no "_" in the last two fields).

    itrea gams-trc [--runs datasets/runs.parquet] [--inc datasets/incumbents_checked.parquet] --out DIR
                   [--set S ...] [--clock call|launch|both]
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd

from itrea.analysis import metrics as M
from itrea.paths import DATASETS

FIELDS = "lineNum, seriesID, node, seconds, bestFound, bestBound"


def _f(x: float) -> str:
    return repr(float(x))


def trc_lines(steps, T: float = M.GAMS_T, z: float | None = None, end_t: float | None = None, sense: int = 1,
              header: str = "itrea") -> list[str]:
    """The trace's lines for best-so-far steps [(t, obj)] (module docstring)."""
    if sense != 1:
        raise ValueError("GAMS's .trc scoring assumes minimisation; a maximisation run is not exported")
    st = [(float(t), float(v)) for t, v in steps if float(t) < T]
    last = st[-1][1] if st else None
    end = float(T)
    beats = False
    if z is not None and st:
        g, _ = M.gams_gap(last, z, sense)
        beats = any(M.gams_gap(v, z, sense)[1] for _, v in st)
        if end_t is not None and g <= 0.0 and math.isfinite(end_t):
            end = min(float(T), max(float(end_t), st[-1][0]))
    out = [f"* mip_trace_ file: ID = {header} T = {T:g} E = {'end' if end < T else 'T'} bestBound = na"
           + (" beats_ref: GAMS's script raises on an incumbent better than z*" if beats else ""),
           f"* fields are {FIELDS}",
           "1, S, 0, 0, na, na"]
    k = 1
    for t, v in st:
        k += 1
        out.append(f"{k}, I, 0, {_f(t)}, {_f(v)}, na")
    k += 1
    out.append(f"{k}, E, 0, {_f(end)}, {_f(last) if last is not None else 'na'}, na")
    return out


def write_trc(path: str | Path, steps, T: float = M.GAMS_T, z: float | None = None, end_t: float | None = None,
              sense: int = 1, header: str = "itrea") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(trc_lines(steps, T, z, end_t, sense, header)) + "\n")
    return p


def _token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.@~+-]", "-", str(s))


def trc_name(instance: str, arm: str, seed: int, clock: str) -> str:
    return f"{instance}_gb10-f-{clock}_{_token(arm)}-s{int(seed)}.trc"


def run_steps(inc: pd.DataFrame, run_id: str, sense: int) -> list:
    """The portfolio's best-so-far steps of one run from the checked-incumbent table (as build_runs builds them)."""
    from itrea.analysis.build_runs import role_steps
    steps = role_steps(inc, run_id, sense)
    return M.merge_steps(*steps.values(), sense=sense)


def export_runs(runs: pd.DataFrame, inc: pd.DataFrame, out_dir: str | Path, clocks=("call", "launch"),
                sets=None) -> list[dict]:
    """One trace per run and clock (module docstring) for every non-synthetic run with a reference; returns what was
    written (and why a run was not)."""
    out_dir = Path(out_dir)
    done = []
    r = runs[~runs["synthetic"].astype(bool)] if "synthetic" in runs.columns else runs
    if sets:
        r = r[r["set"].astype(str).isin(list(sets))]
    for _, row in r.iterrows():
        rid, s, T, z = str(row["run_id"]), int(row["sense"]), float(row["budget"]), float(row["ref"])
        rec = dict(run_id=rid, files=[])
        if not math.isfinite(z) or s != 1:
            rec["skipped"] = "no reference" if not math.isfinite(z) else "maximisation"
            done.append(rec)
            continue
        steps = run_steps(inc, rid, s)
        for clock in clocks:
            off = 0.0 if clock == "call" else float(row.get("startup_offset_s", math.nan))
            if not math.isfinite(off):
                rec.setdefault("skipped_clocks", []).append(clock)
                continue
            p = out_dir / trc_name(str(row["instance"]), str(row["arm"]), int(row["seed"]), clock)
            write_trc(p, [(t + off, v) for t, v in steps], T, z, None, s, header=f"itrea {rid} clock={clock}")
            rec["files"].append(str(p))
        done.append(rec)
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="itrea gams-trc", description="GAMS .trc export of our runs, one per run and "
                                 "clock (D-021 R16): GAMS's own script scores them to our pt_gams")
    ap.add_argument("--runs", default=str(DATASETS / "runs.parquet"))
    ap.add_argument("--inc", default=str(DATASETS / "incumbents_checked.parquet"))
    ap.add_argument("--out", required=True, help="directory for the .trc files")
    ap.add_argument("--set", dest="sets", action="append", help="only these run sets (repeatable)")
    ap.add_argument("--clock", choices=["call", "launch", "both"], default="both")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    for p in (a.runs, a.inc):
        if not Path(p).exists():
            print(f"{p} does not exist (run `itrea build-runs` first)", file=sys.stderr)
            return 2
    clocks = ("call", "launch") if a.clock == "both" else (a.clock,)
    done = export_runs(pd.read_parquet(a.runs), pd.read_parquet(a.inc), a.out, clocks, a.sets)
    n = sum(len(d["files"]) for d in done)
    skipped = [d for d in done if d.get("skipped")]
    print(f"{n} traces for {len(done) - len(skipped)} runs -> {a.out}" +
          (f"; {len(skipped)} runs skipped ({', '.join(sorted({d['skipped'] for d in skipped}))})" if skipped else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
