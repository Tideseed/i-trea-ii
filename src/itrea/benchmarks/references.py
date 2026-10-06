"""Reference objective values: the MIPLIB `.solu` file, the frozen `datasets/reference.csv`, and the provisional
references used before the freeze (vault D-002 §4, `10-design/Trace format and metrics.md`).

Why a module of its own: the primal integral is only as good as its reference, and D-002 orders the steps so that the
reference is frozen AFTER the pool (step 4 follows step 3). The screening runs that choose the pool therefore need a
provisional reference, and every number must say which kind of reference it used. The resolution order is:

    1. the frozen `datasets/reference.csv` row (kind as frozen: opt | best_known | screening_best | reference_run);
    2. the MIPLIB `.solu` value (`=opt=` proven optimal, `=best=` best known);
    3. the best CHECKED objective over the screening (SCREEN) and reference (REF) runs  -> "screening_best";
    4. the best checked objective of any run of the study                               -> "best_of_study".

Kinds 3 and 4 are provisional: confirmatory analysis (itrea.analysis.prereg) refuses them.

`.solu` lines: `=opt=  name  value`, `=best=  name  value`, `=inf=  name`, `=unbd=  name`, `=unkn=  name`.

**Reference 0** makes Berthold's gap degenerate (γ = 1 for any other value, so P(T) becomes "time to the exact
optimum"); such instances are flagged `ref_zero` and, with sign crossings seen in screening, analysed on time to target
only (D-002 §4 "P-degenerate").
"""
from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from itrea.paths import DATA

SOLU_KINDS = {"=opt=": "opt", "=best=": "best_known", "=inf=": "infeasible", "=unbd=": "unbounded",
              "=unkn=": "unknown"}
FROZEN_KINDS = ("opt", "best_known", "screening_best", "reference_run")
PROVISIONAL_KINDS = ("screening_best_provisional", "best_of_study")
REFERENCE_COLUMNS = ["instance", "set", "reference", "sense", "kind", "ref_zero", "source"]
MPS_SUFFIXES = (".mps.gz", ".mps.bz2", ".mps", ".lp.gz", ".lp")


@dataclass(frozen=True)
class SoluEntry:
    name: str
    kind: str                 # opt | best_known | infeasible | unbounded | unknown
    value: float | None       # None for infeasible / unbounded / unknown

    @property
    def has_value(self) -> bool:
        return self.value is not None and math.isfinite(self.value)


def stem(path: str | os.PathLike) -> str:
    """Instance name without directory and MPS/LP suffix: `data/miplib/inst/air05.mps.gz` -> `air05`."""
    n = Path(str(path)).name
    for ext in MPS_SUFFIXES:
        if n.endswith(ext):
            return n[: -len(ext)]
    return n


def parse_solu(text: str) -> dict[str, SoluEntry]:
    """Parse the text of a MIPLIB `.solu` file. A name listed twice keeps its first line (the file has no duplicates;
    a duplicate would be an editing error, so it raises)."""
    out: dict[str, SoluEntry] = {}
    for ln, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        tag = parts[0]
        if tag not in SOLU_KINDS:
            raise ValueError(f"solu line {ln}: unknown tag {tag!r}")
        if len(parts) < 2:
            raise ValueError(f"solu line {ln}: no instance name")
        name, kind = parts[1], SOLU_KINDS[tag]
        value = None
        if kind in ("opt", "best_known"):
            if len(parts) < 3:
                raise ValueError(f"solu line {ln}: {tag} without a value")
            value = float(parts[2])
        if name in out:
            raise ValueError(f"solu line {ln}: {name} listed twice")
        out[name] = SoluEntry(name, kind, value)
    return out


def read_solu(path: str | os.PathLike | None = None) -> dict[str, SoluEntry]:
    """The MIPLIB 2017 solution file on disk (default `data/miplib/solu.txt`)."""
    p = Path(path) if path is not None else DATA / "miplib" / "solu.txt"
    return parse_solu(p.read_text())


def solu_sha256(path: str | os.PathLike | None = None) -> str:
    p = Path(path) if path is not None else DATA / "miplib" / "solu.txt"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def is_zero(v: float | None) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and float(v) == 0.0


def is_miplib(set_name: str | None, instance: str | None = None) -> bool:
    s = (set_name or "").lower()
    return s.startswith("miplib") or "miplib" in str(instance or "").lower()


# --------------------------------------------------------------------------------------------- reference.csv
def read_reference_csv(path: str | os.PathLike) -> pd.DataFrame:
    """Load and validate the frozen reference file. Columns: REFERENCE_COLUMNS. `sense` is +1 (minimise) or -1
    (maximise); `reference` is in the user's sense; `kind` is one of FROZEN_KINDS."""
    df = pd.read_csv(path, dtype={"instance": str, "set": str, "kind": str, "source": str})
    missing = [c for c in REFERENCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if df["instance"].duplicated().any():
        raise ValueError(f"{path}: duplicate instances {df.loc[df['instance'].duplicated(), 'instance'].tolist()[:5]}")
    bad = sorted(set(df["kind"]) - set(FROZEN_KINDS))
    if bad:
        raise ValueError(f"{path}: kinds {bad} are not frozen kinds {FROZEN_KINDS}")
    if not df["sense"].isin([1, -1]).all():
        raise ValueError(f"{path}: sense must be +1 or -1")
    if not pd.api.types.is_float_dtype(df["reference"]) and not pd.api.types.is_integer_dtype(df["reference"]):
        raise ValueError(f"{path}: reference must be numeric")
    if df["reference"].isna().any():
        raise ValueError(f"{path}: an instance without a reference value must not be in the frozen file")
    df["reference"] = df["reference"].astype(float)
    df["ref_zero"] = df["reference"] == 0.0
    return df[REFERENCE_COLUMNS].reset_index(drop=True)


def write_reference_csv(df: pd.DataFrame, path: str | os.PathLike) -> Path:
    """Write the frozen reference file deterministically (sorted by instance, fixed columns, repr floats)."""
    out = df[REFERENCE_COLUMNS].copy()
    out["reference"] = out["reference"].astype(float)
    out["sense"] = out["sense"].astype(int)
    out["ref_zero"] = out["reference"] == 0.0
    out = out.sort_values("instance", kind="mergesort").reset_index(drop=True)
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    out.to_csv(tmp, index=False, float_format="%.17g", lineterminator="\n")
    os.replace(tmp, p)
    return p


def freeze_references(instances: pd.DataFrame, runs: pd.DataFrame, solu: dict[str, SoluEntry],
                      ref_arms: tuple[str, ...] = ("SCREEN", "REF")) -> pd.DataFrame:
    """The frozen reference table (D-002 §4, step 4) for `instances` (columns instance, set, sense).

    MIPLIB: the `.solu` value. Elsewhere: the best checked objective of the screening (SCREEN) and dedicated
    reference (REF) runs — `runs` are rows of datasets/runs.parquet with columns instance, arm, best_obj, sense.
    An instance without any value is left out (and must not enter a confirmatory comparison)."""
    rows = []
    rr = runs[runs["arm"].isin(ref_arms) & runs["best_obj"].notna()] if len(runs) else runs
    for r in instances.itertuples(index=False):
        name, set_name, sense = r.instance, r.set, int(r.sense)
        e = solu.get(name)
        if is_miplib(set_name) and e is not None and e.has_value:
            rows.append(dict(instance=name, set=set_name, reference=float(e.value), sense=sense, kind=e.kind,
                             source="MIPLIB 2017 solu"))
            continue
        cand = rr[rr["instance"] == name] if len(rr) else rr
        if len(cand) == 0:
            continue
        vals = cand["best_obj"].astype(float).to_numpy()
        best = float(vals[(sense * vals).argmin()])
        arms = sorted(set(cand.loc[cand["best_obj"].astype(float) == best, "arm"]))
        kind = "reference_run" if "REF" in arms else "screening_best"
        rows.append(dict(instance=name, set=set_name, reference=best, sense=sense, kind=kind,
                         source="best checked incumbent of " + "+".join(arms) + " runs"))
    df = pd.DataFrame(rows, columns=[c for c in REFERENCE_COLUMNS if c != "ref_zero"])
    df["ref_zero"] = df["reference"].astype(float) == 0.0
    return df[REFERENCE_COLUMNS].sort_values("instance", kind="mergesort").reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ resolution
@dataclass(frozen=True)
class Reference:
    value: float | None
    kind: str                # a FROZEN_KINDS member, "screening_best_provisional", "best_of_study" or "none"
    source: str

    @property
    def frozen(self) -> bool:
        return self.kind in FROZEN_KINDS

    @property
    def ref_zero(self) -> bool:
        return is_zero(self.value)


def resolve(instance: str, set_name: str | None, frozen: pd.DataFrame | None, solu: dict[str, SoluEntry] | None,
            screening_best: dict[str, float] | None = None, study_best: dict[str, float] | None = None
            ) -> Reference:
    """The reference of one instance by the module's resolution order (frozen -> .solu -> screening -> study)."""
    if frozen is not None and len(frozen):
        hit = frozen[frozen["instance"] == instance]
        if len(hit):
            r = hit.iloc[0]
            return Reference(float(r["reference"]), str(r["kind"]), "datasets/reference.csv: " + str(r["source"]))
    if solu and is_miplib(set_name, instance):
        e = solu.get(instance)
        if e is not None and e.has_value:
            return Reference(float(e.value), e.kind, "MIPLIB 2017 solu")
    if screening_best and instance in screening_best:
        return Reference(float(screening_best[instance]), "screening_best_provisional",
                         "best checked incumbent of the SCREEN/REF runs (provisional, not frozen)")
    if study_best and instance in study_best:
        return Reference(float(study_best[instance]), "best_of_study",
                         "best checked incumbent of any run (provisional, not frozen)")
    return Reference(None, "none", "no reference")
