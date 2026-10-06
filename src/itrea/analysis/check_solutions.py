"""Independent exact check of EVERY stored incumbent of every process (vault D-002 §4 "Integrity").

Adapted from berkorbay/optopt@4dd48395, MIT (`src/optopt/analysis/check_solutions.py`, issue #25 there). The rules
carried over, because each one was paid for:

- **Another library re-reads the model.** A solution is checked against the model as parsed by a different MPS reader
  than the one that produced it, so a reader bug cannot certify its own output: SCIP-produced vectors are checked
  against highspy's parse, HiGHS-produced (H* arms) and EA-produced vectors (the EA's `Problem` is read by HiGHS)
  against PySCIPOpt's, cuOpt's (own parser) against highspy's (`PRODUCER_READER`).
- **Exact arithmetic where it decides.** Bounds and integrality are decided exactly (a float comparison of two floats
  is exact, and |x − round(x)| is exact in binary floating point by Sterbenz' lemma); the objective is recomputed in
  exact rational arithmetic over the non-zeros; every row is computed in float64 first and **re-computed exactly** when
  its float verdict could be wrong. The float error bound is rigorous: |fl(a·x) − a·x| ≤ (k+1)·2⁻⁵²·Σ|a_j x_j| for a
  row with k non-zeros (optopt used 1e-12·Σ|a_j x_j|, which is not a bound for rows of ~10⁵ terms, e.g. rail01).
- **One tolerance for every arm:** 1e-6, rows relative to max(1, |side|) (MIPLIB-checker style), bounds and
  integrality absolute. SCIP's own rule, relative to max(|activity|, |side|, 1), is reported as a diagnostic column.
- **A recomputed objective that disagrees with the reported one (relative 1e-6) fails the check.** It means the two
  readers disagree about the model (e.g. the sign of an MPS objective constant) or the vector was mapped wrongly;
  silently using either objective would put a wrong number into P(T).
- **Copies, not views,** of highspy's arrays (optopt 2026-09-24: cached views of memory owned by a freed `Highs`
  object made the same solution pass and fail depending on what was checked before).
- Both readers' infinities (SCIP 1e20, HiGHS 1e30 / inf) are normalised to ±inf.

Input: `<role>.incumbents.npz` (`itrea.trace.save_incumbents` / `load_incumbents`): every incumbent with t, the
reported objective, source and the vector in the process's own variable order (`names`); the checker maps names onto
the model it re-read. Output: one row per incumbent (`check_run_dir`), the table `itrea.analysis.build_runs` turns
into `datasets/incumbents_checked.parquet`; P(T) is built only from rows with `ok`, using `obj_recomputed`.

    python -m itrea.analysis.check_solutions traces/raw/<set>[/<run_id>] … [--out table.csv|.parquet]
exits 1 if any incumbent fails (a failed check is an error, not a log line — optopt's rule), 2 on bad arguments.
Untimed work: run it with OMP_NUM_THREADS=1 on the efficiency cores, never beside timed runs with default threads.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.sparse as sps

from itrea.paths import resolve

# Version of the check logic. BUMP IT on any change that can alter a verdict or a recomputed objective (tolerances,
# the row rule, a reader fix, the name mapping): build_runs keys its cached verdicts on it, so rows checked by another
# version are re-checked (or reported as stale when the instance file is missing) instead of silently reused.
CHECKER_VERSION = "2026-10-06.1"
TOL = 1e-6                 # bounds, integrality (absolute) and rows (relative to max(1, |side|))
OBJ_TOL = 1e-6             # recomputed vs reported objective, relative to max(1, |reported|)
INF = 1e20                 # |value| >= INF is infinite (SCIP's infinity; HiGHS uses 1e30 or inf)
EPS = 2.0 ** -52
PRODUCER_READER = {"scip": "highs", "highs": "scip", "ea": "scip", "cuopt": "highs"}
INT_VTYPES = ("BINARY", "INTEGER", "IMPLINT")


class ModelUnsupported(RuntimeError):
    """The model holds a constraint the checker cannot evaluate (indicator, SOS, nonlinear): its incumbents are
    reported as unchecked, never as feasible."""


@dataclass
class Model:
    path: str
    reader: str                    # "highs" | "scip"
    names: list[str]
    lb: np.ndarray
    ub: np.ndarray
    isint: np.ndarray
    A: sps.csr_matrix              # m x n, model column order
    rl: np.ndarray
    ru: np.ndarray
    c: np.ndarray                  # user-sense objective coefficients
    offset: float
    sense: int                     # +1 minimise, -1 maximise
    absA: sps.csr_matrix = field(init=False, repr=False)
    row_nnz: np.ndarray = field(init=False, repr=False)
    index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        self.A = sps.csr_matrix(self.A, dtype=np.float64, copy=True)
        self.A.sum_duplicates()
        self.absA = abs(self.A)
        self.row_nnz = np.diff(self.A.indptr).astype(np.float64)
        self.index = {nm: j for j, nm in enumerate(self.names)}
        if len(self.index) != len(self.names):
            raise ValueError(f"{self.path}: duplicate variable names in the {self.reader} parse")

    @property
    def n(self) -> int:
        return len(self.names)

    @property
    def m(self) -> int:
        return self.A.shape[0]


def _inf(v) -> np.ndarray:
    v = np.array(v, dtype=np.float64, copy=True)
    v[v >= INF] = np.inf
    v[v <= -INF] = -np.inf
    return v


def read_highs(path: str | os.PathLike) -> Model:
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if h.readModel(str(resolve(path))) == highspy.HighsStatus.kError:
        raise IOError(f"HiGHS could not read {path}")
    lp = h.getLp()
    n, m = lp.num_col_, lp.num_row_
    a = lp.a_matrix_
    start, index, value = np.array(a.start_), np.array(a.index_), np.array(a.value_, dtype=np.float64)
    if a.format_ == highspy.MatrixFormat.kColwise:
        A = sps.csc_matrix((value, index, start), shape=(m, n)).tocsr()
    else:
        A = sps.csr_matrix((value, index, start), shape=(m, n))
    vt = highspy.HighsVarType
    integ = list(lp.integrality_) if len(lp.integrality_) else [vt.kContinuous] * n
    if any(t in (vt.kSemiContinuous, vt.kSemiInteger) for t in integ):
        raise ModelUnsupported(f"{path}: semi-continuous / semi-integer variables")
    isint = np.array([t in (vt.kInteger, vt.kImplicitInteger) for t in integ], dtype=bool)
    names = [str(s) for s in lp.col_names_] if len(lp.col_names_) == n else [f"x{j}" for j in range(n)]
    sense = -1 if lp.sense_ == highspy.ObjSense.kMaximize else 1
    return Model(path=str(path), reader="highs", names=names, lb=_inf(lp.col_lower_), ub=_inf(lp.col_upper_),
                 isint=isint, A=A, rl=_inf(lp.row_lower_), ru=_inf(lp.row_upper_),
                 c=np.array(lp.col_cost_, dtype=np.float64), offset=float(lp.offset_), sense=sense)


def read_scip(path: str | os.PathLike) -> Model:
    import pyscipopt

    s = pyscipopt.Model()
    s.hideOutput()
    s.readProblem(str(resolve(path)))
    try:
        vars_ = s.getVars(transformed=False)
        names = [v.name for v in vars_]
        pos = {nm: j for j, nm in enumerate(names)}
        lb = _inf([v.getLbOriginal() for v in vars_])
        ub = _inf([v.getUbOriginal() for v in vars_])
        isint = np.array([v.vtype() in INT_VTYPES for v in vars_], dtype=bool)
        c = np.array([v.getObj() for v in vars_], dtype=np.float64)
        rows, cols, vals, rl, ru = [], [], [], [], []
        for i, cons in enumerate(s.getConss()):
            hdlr = cons.getConshdlrName()
            if hdlr != "linear":
                raise ModelUnsupported(f"{path}: constraint {cons.name!r} of type {hdlr!r} (only linear rows are "
                                       f"checked)")
            coef = s.getValsLinear(cons)
            for nm, a in coef.items():
                rows.append(i)
                cols.append(pos[nm])
                vals.append(float(a))
            rl.append(s.getLhs(cons))
            ru.append(s.getRhs(cons))
        m = len(rl)
        A = sps.csr_matrix((np.array(vals, dtype=np.float64), (np.array(rows, dtype=np.int64),
                                                               np.array(cols, dtype=np.int64))), shape=(m, len(names)))
        sense = -1 if s.getObjectiveSense() == "maximize" else 1
        offset = float(s.getObjoffset(original=True))
    finally:
        s.freeProb()
    return Model(path=str(path), reader="scip", names=names, lb=lb, ub=ub, isint=isint, A=A, rl=_inf(rl),
                 ru=_inf(ru), c=c, offset=offset, sense=sense)


def read_model(path: str | os.PathLike, reader: str) -> Model:
    if reader == "highs":
        return read_highs(path)
    if reader == "scip":
        return read_scip(path)
    raise ValueError(f"unknown reader {reader!r}")


class ModelCache:
    """LRU cache of parsed models (default 2: a large instance in both parses fits the 4 GB per-process rule)."""

    def __init__(self, maxsize: int = 2):
        self.maxsize = maxsize
        self._d: OrderedDict[tuple[str, str], Model | Exception] = OrderedDict()

    def get(self, path: str | os.PathLike, reader: str) -> Model:
        k = (str(path), reader)
        if k in self._d:
            self._d.move_to_end(k)
        else:
            try:
                self._d[k] = read_model(path, reader)
            except (ModelUnsupported, IOError, OSError) as e:
                self._d[k] = e
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)
        v = self._d[k]
        if isinstance(v, Exception):
            raise v
        return v


# ---------------------------------------------------------------------------------------------- exact helpers
def exact_dot(a: np.ndarray, x: np.ndarray) -> Fraction:
    """Exact Σ a_j x_j for float64 inputs. Every float is n / 2^k exactly, so the products are integer numerators
    over powers of two and the sum needs one common shift — no gcd per term."""
    terms = []
    for ai, xi in zip(np.asarray(a, dtype=np.float64).tolist(), np.asarray(x, dtype=np.float64).tolist()):
        if ai == 0.0 or xi == 0.0:
            continue
        n1, d1 = ai.as_integer_ratio()
        n2, d2 = xi.as_integer_ratio()
        terms.append((n1 * n2, (d1 * d2).bit_length() - 1))
    if not terms:
        return Fraction(0)
    E = max(e for _, e in terms)
    return Fraction(sum(n << (E - e) for n, e in terms), 1 << E)


class NameMap:
    """Producer's variable order -> model order, built once per (model, names)."""

    def __init__(self, model: Model, names: list[str]):
        idx = model.index
        self.src = np.array([j for j, nm in enumerate(names) if nm in idx], dtype=np.int64)
        self.dst = np.array([idx[nm] for nm in names if nm in idx], dtype=np.int64)
        self.unknown = np.array([j for j, nm in enumerate(names) if nm not in idx], dtype=np.int64)
        self.missing = model.n - len(set(self.dst.tolist()))
        self.n_model = model.n

    def to_model(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_model, dtype=np.float64)
        out[self.dst] = np.asarray(x, dtype=np.float64)[self.src]
        return out


# ---------------------------------------------------------------------------------------------------- check
def check_vector(model: Model, x: np.ndarray, obj_reported: float | None = None, tol: float = TOL,
                 unknown_vars: int = 0, missing_vars: int = 0) -> dict:
    """Check one vector given in MODEL order. Returns the verdict and the worst violations (see module docstring)."""
    x = np.asarray(x, dtype=np.float64)
    reasons = []
    if not np.all(np.isfinite(x)):
        nan = math.nan
        return dict(feasible=False, obj_ok=False, ok=False, worst_bound=nan, worst_int=nan, worst_row_rel=nan,
                    worst_row_scip=nan, obj_recomputed=nan, obj_reported=_f(obj_reported), obj_rel_err=nan,
                    unknown_vars=unknown_vars, missing_vars=missing_vars, exact_rows=0, n_viol_rows=-1,
                    reason="non_finite")
    ftol = Fraction(tol)
    # bounds: decided exactly on the (few) entries outside their bounds
    worst_bound, bound_bad = 0.0, False
    for side, below in ((model.lb, True), (model.ub, False)):
        fin = np.isfinite(side)
        out = np.flatnonzero(fin & ((x < side) if below else (x > side)))
        for j in out.tolist():
            d = (Fraction(side[j]) - Fraction(x[j])) if below else (Fraction(x[j]) - Fraction(side[j]))
            worst_bound = max(worst_bound, float(d))
            bound_bad |= d > ftol
    # integrality: |x - round(x)| is exact in float64
    xi = x[model.isint]
    frac = np.abs(xi - np.round(xi))
    worst_int = float(frac.max(initial=0.0))
    int_bad = bool(np.any(frac > tol))
    # rows: float pass + rigorous error bound, exact re-check where the float verdict could be wrong
    act = model.A @ x
    err = (model.row_nnz + 1.0) * EPS * (model.absA @ np.abs(x)) + 1e-300
    with np.errstate(invalid="ignore"):
        sl, su = np.maximum(1.0, np.abs(model.rl)), np.maximum(1.0, np.abs(model.ru))
        lo_v = np.where(np.isfinite(model.rl), (model.rl - act) / sl, -np.inf)
        up_v = np.where(np.isfinite(model.ru), (act - model.ru) / su, -np.inf)
        v = np.maximum(lo_v, up_v)
        sc_l = np.maximum(np.maximum(np.abs(act), np.abs(model.rl)), 1.0)
        sc_u = np.maximum(np.maximum(np.abs(act), np.abs(model.ru)), 1.0)
        v_scip = np.maximum(np.where(np.isfinite(model.rl), (model.rl - act) / sc_l, -np.inf),
                            np.where(np.isfinite(model.ru), (act - model.ru) / sc_u, -np.inf))
    near = np.flatnonzero(np.abs(v - tol) <= err)
    row_bad_float = (v > tol)
    row_bad_float[near] = False
    exact_bad = 0
    if near.size:
        A = model.A
        for i in near.tolist():
            a, b = A.indptr[i], A.indptr[i + 1]
            ax = exact_dot(A.data[a:b], x[A.indices[a:b]])
            ve = None
            if np.isfinite(model.rl[i]):
                ve = (Fraction(model.rl[i]) - ax) / max(Fraction(1), abs(Fraction(model.rl[i])))
            if np.isfinite(model.ru[i]):
                vu = (ax - Fraction(model.ru[i])) / max(Fraction(1), abs(Fraction(model.ru[i])))
                ve = vu if ve is None else max(ve, vu)
            if ve is not None:
                v[i] = float(ve)
                exact_bad += int(ve > ftol)
    n_viol = int(row_bad_float.sum()) + exact_bad
    worst_row = float(max(0.0, v.max(initial=0.0))) if v.size else 0.0
    worst_scip = float(max(0.0, v_scip.max(initial=0.0))) if v_scip.size else 0.0
    # objective, exact over the non-zeros
    nz = np.flatnonzero(x != 0.0)
    obj = float(exact_dot(model.c[nz], x[nz]) + Fraction(model.offset))
    rep = _f(obj_reported)
    obj_rel = abs(obj - rep) / max(1.0, abs(rep)) if rep is not None else math.nan
    obj_ok = rep is None or obj_rel <= OBJ_TOL
    if bound_bad:
        reasons.append("bound")
    if int_bad:
        reasons.append("integrality")
    if n_viol:
        reasons.append("row")
    if unknown_vars:
        reasons.append("unknown_vars")
    if missing_vars:
        reasons.append("missing_vars")
    if not obj_ok:
        reasons.append("objective")
    feasible = not (bound_bad or int_bad or n_viol or unknown_vars or missing_vars)
    return dict(feasible=bool(feasible), obj_ok=bool(obj_ok), ok=bool(feasible and obj_ok), worst_bound=worst_bound,
                worst_int=worst_int, worst_row_rel=worst_row, worst_row_scip=worst_scip, obj_recomputed=obj,
                obj_reported=rep if rep is not None else math.nan, obj_rel_err=obj_rel, unknown_vars=int(unknown_vars),
                missing_vars=int(missing_vars), exact_rows=int(near.size), n_viol_rows=n_viol,
                reason=";".join(reasons))


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def check_named(model: Model, names: list[str], x: np.ndarray, obj_reported: float | None = None,
                tol: float = TOL, name_map: NameMap | None = None) -> dict:
    """Check a vector given in the producer's order `names`."""
    nm = name_map or NameMap(model, names)
    x = np.asarray(x, dtype=np.float64)
    return check_vector(model, nm.to_model(x), obj_reported, tol, unknown_vars=int(nm.unknown.size),
                        missing_vars=int(nm.missing))


def check_sol_json(model: Model, sol_path: str | os.PathLike, obj_reported: float | None = None,
                   tol: float = TOL) -> dict:
    """Check a `<role>.sol.json.gz` ({name: value} of non-zeros; optopt's format)."""
    import gzip

    vals = json.loads(gzip.open(sol_path, "rt").read())
    names = list(vals)
    x = np.array([float(vals[k]) for k in names], dtype=np.float64)
    nm = NameMap(model, names)
    unknown = int(nm.unknown.size)
    return check_vector(model, nm.to_model(x), obj_reported, tol, unknown_vars=unknown, missing_vars=0)


# ----------------------------------------------------------------------------------------------- run level
def producer_of(role: str, summary: dict | None = None, arm: str | None = None) -> str:
    """Which library produced a role's vectors: the summary's `solver` / `producer` field, else the role name and
    arm (H* arms run HiGHS as the solver)."""
    s = summary or {}
    for key in ("producer", "solver", "engine"):
        v = str(s.get(key, "") or "").lower()
        if v in PRODUCER_READER:
            return v
    r = role.lower()
    if r.startswith("ea"):
        return "ea"
    if r.startswith("cuopt"):
        return "cuopt"
    if r.startswith("solver"):
        a = str(arm or s.get("arm") or "").upper()
        return "highs" if a.startswith("H") else "scip"
    raise ValueError(f"cannot tell which library produced role {role!r}")


def reader_for(producer: str) -> str:
    return PRODUCER_READER[producer]


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def run_instance(run_dir: Path, run_json: dict | None = None, summaries: dict | None = None) -> str | None:
    """The run's instance path (run.json, else any role summary)."""
    rj = run_json if run_json is not None else _read_json(run_dir / "run.json")
    if rj.get("instance"):
        return str(rj["instance"])
    for s in (summaries or {}).values():
        if s.get("instance"):
            return str(s["instance"])
    for p in sorted(run_dir.glob("*.summary.json")):
        s = _read_json(p)
        if s.get("instance"):
            return str(s["instance"])
    return None


def file_sha256(p: str | os.PathLike) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_incumbents(npz_path: str | os.PathLike, instance: str, producer: str, cache: ModelCache | None = None,
                     tol: float = TOL) -> list[dict]:
    """One row per incumbent in an incumbents.npz. A model the checker cannot read leaves every row unchecked
    (`checked` False, `ok` False) with the reason."""
    from itrea.trace import load_incumbents

    cache = cache or ModelCache()
    reader = reader_for(producer)
    names, recs = load_incumbents(npz_path)
    base = dict(producer=producer, reader=reader)
    try:
        model = cache.get(instance, reader)
    except (ModelUnsupported, IOError, OSError, KeyError) as e:
        return [dict(base, k=k, t=t, source=src, obj_reported=obj, checked=False, ok=False, feasible=False,
                     obj_ok=False, reason=f"unchecked: {type(e).__name__}: {e}"[:300], sense=0)
                for k, (t, obj, src, _) in enumerate(recs)]
    nm = NameMap(model, names)
    rows = []
    for k, (t, obj, src, x) in enumerate(recs):
        r = check_named(model, names, x, obj, tol, nm)
        rows.append(dict(base, k=k, t=t, source=src, checked=True, sense=model.sense, **r))
    return rows


def check_run_dir(run_dir: str | os.PathLike, cache: ModelCache | None = None, tol: float = TOL,
                  instance: str | None = None) -> list[dict]:
    """Check every `<role>.incumbents.npz` in a run directory; rows carry run_id, role and the file's sha256."""
    d = Path(run_dir)
    rj = _read_json(d / "run.json")
    cache = cache or ModelCache()
    inst = instance or run_instance(d, rj)
    rows = []
    for npz in sorted(d.glob("*.incumbents.npz")):
        role = npz.name[: -len(".incumbents.npz")]
        summ = _read_json(d / f"{role}.summary.json")
        prod = producer_of(role, summ, rj.get("arm") or summ.get("arm"))
        sha = file_sha256(npz)
        if inst is None:
            raise ValueError(f"{d}: no instance path in run.json or any summary")
        for r in check_incumbents(npz, inst, prod, cache, tol):
            rows.append(dict(run_id=rj.get("run_id") or d.name, role=role, npz_sha256=sha,
                             checker_version=CHECKER_VERSION, **r))
    return rows


def find_run_dirs(paths: list[str | os.PathLike]) -> list[Path]:
    out = set()
    for p in paths:
        p = Path(p)
        if p.is_file() and p.name.endswith(".incumbents.npz"):
            out.add(p.parent)
        elif p.is_dir():
            for f in p.rglob("*.incumbents.npz"):
                out.add(f.parent)
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="run directories or roots searched for *.incumbents.npz")
    ap.add_argument("--out", help="write the per-incumbent table (.csv or .parquet)")
    ap.add_argument("--tol", type=float, default=TOL)
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    import pandas as pd

    cache = ModelCache()
    rows = []
    for d in find_run_dirs(a.paths):
        rows += check_run_dir(d, cache, a.tol)
    df = pd.DataFrame(rows)
    if a.out:
        (df.to_parquet if a.out.endswith(".parquet") else df.to_csv)(a.out, index=False)
    if df.empty:
        print("no incumbents found")
        return 0
    bad = df[~df["ok"].astype(bool)]
    print(f"{len(df)} incumbents checked in {df['run_id'].nunique()} runs, {len(bad)} failed or unchecked")
    if len(bad):
        cols = [c for c in ("run_id", "role", "k", "t", "reason", "worst_row_rel", "worst_bound", "worst_int",
                            "obj_rel_err") if c in bad.columns]
        print(bad[cols].head(50).to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
