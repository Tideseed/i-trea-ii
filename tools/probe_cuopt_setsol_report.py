"""Summarise tools/probe_cuopt_setsol.py runs (cuopt#2073 / PR #2098) and check the vectors cuOpt reported with our
exact checker (itrea.analysis.check_solutions, HiGHS reader).

    OMP_NUM_THREADS=1 .venv/bin/python tools/probe_cuopt_setsol_report.py /tmp/setsol/out [--sol-dir /tmp/setsol]

Per run: cuOpt version; presolve (Papilo's reduced size, or "disabled"); the offers the SET callback made; whether a
GET record equal to the injected objective followed (and how fast); cuOpt's best before the offer; the final
objective; for `bad` runs, whether any GET record matches an offered infeasible vector; the exact check of the GET
record equal to the injected solution and of the last GET record.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from itrea.analysis.check_solutions import check_named, read_model


def presolve_summary(lines: list[str]) -> str:
    txt = " | ".join(lines)
    if re.search(r"Presolve is disabled", txt):
        return "DISABLED (" + next((ln for ln in lines if "disabled" in ln), "") + ")"
    m = re.search(r"Presolved problem: (\d+) constraints, (\d+) variables", txt)
    rem = re.search(r"Presolve removed: ([^|]+)", txt)
    if m:
        return f"on: {m.group(2)} vars / {m.group(1)} rows after Papilo ({rem.group(1).strip() if rem else ''})"
    return "unknown: " + txt[:160]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--sol-dir", default="/tmp/setsol")
    a = ap.parse_args()
    models: dict[str, object] = {}
    rows = []
    for p in sorted(Path(a.outdir).glob("*.json")):
        r = json.loads(p.read_text())
        z = np.load(p.with_suffix(".npz"))
        names = [str(s) for s in z["names"]]
        gets = list(zip(z["t"].tolist(), z["obj"].tolist()))
        X = z["X"]
        inst = r["instance"]
        if inst not in models:
            models[inst] = read_model(inst, "highs")
        model = models[inst]
        offers = r["set_offers"]
        t_off = offers[0]["t"] if offers else None
        row = dict(run=p.stem, version=r["versions"].get("cuopt-cu13", "?"), presolve=presolve_summary(r["presolve_log"]),
                   final=r["final_obj"], n_get=len(gets), offers=len(offers))
        if t_off is not None:
            before = [o for t, o in gets if t < t_off]
            row["best_before_offer"] = min(before) if before else None
            row["offer_t"] = round(t_off, 2)
        if r["mode"] in ("inject", "raw"):
            inj = r["injected_obj"]
            hit = [i for i, (t, o) in enumerate(gets) if t >= (t_off or 0) - 1e-6 and abs(o - inj) <= 1e-6 * max(1.0, abs(inj))]
            row["injected_obj"] = inj
            row["reported_back"] = bool(hit)
            if hit:
                i = hit[0]
                row["delay_s"] = round(gets[i][0] - t_off, 3)
                chk = check_named(model, names, X[i], obj_reported=gets[i][1])
                row["check_injected_echo"] = f"{chk.get('ok')} feasible={chk.get('feasible')} worst_row_rel={chk.get('worst_row_rel')}"
                s = json.loads((Path(a.sol_dir) / (Path(inst).name.split('.')[0] + ('_beatsref' if r['mode'] == 'raw' else '') + '.json')).read_text())
                pos = {nm: k for k, nm in enumerate(s["names"])}
                xin = np.array([s["x"][pos[nm]] for nm in names])
                row["echo_equals_offer_maxabs"] = float(np.max(np.abs(X[i] - xin)))
        if r["mode"] == "bad":
            bad_objs = [o["obj"] for o in offers]
            row["bad_offers"] = len(offers)
            row["bad_reported"] = sum(1 for t, o in gets if any(abs(o - b) <= 1e-9 * max(1.0, abs(b)) for b in bad_objs))
        if len(gets):
            chk = check_named(model, names, X[-1], obj_reported=gets[-1][1])
            row["check_last_get"] = f"{chk.get('ok')}"
        rows.append(row)
    for row in rows:
        print(json.dumps(row, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
