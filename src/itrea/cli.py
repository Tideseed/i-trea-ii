"""`itrea` — one entry point for the project's commands: `itrea <command> [arguments of that command]`.

Each command is a thin dispatch to the module that owns it; that module parses its own arguments (`itrea pair --help`
prints the pair runner's help). Modules are imported only when their command runs: the EA, the solvers and the analysis
pull in CuPy, PySCIPOpt or pandas, and `itrea --help` should not.

Why a single entry point: the runbooks and the vault name commands (`itrea exchange up`, `itrea build-runs`), and the
same command must reach the same code whether it is typed, run by the pool, or called from a skill. The table below is
the only place the mapping lives.

    itrea ea-run …             python -m itrea.ea.run              the EA process
    itrea scip-run …           python -m itrea.solvers.scip_run    SCIP process (heuristic-plugin exchange)
    itrea highs-run …          python -m itrea.solvers.highs_run   HiGHS process
    itrea cuopt-run …          python -m itrea.solvers.cuopt_run   cuOpt process (control arm)
    itrea pair …               one run (smoke tests; the pool runs job files)
    itrea pool JOBS …          a job file under every D-002 / D-003 control
    itrea make-jobs SUB …      job files: screen1 | screen2 | e1 | e4 | confirm | smoke | backoff0 | backoff1 |
                               mipfeas-cal
    itrea screen-candidates …  hard pool: the screening candidates          (itrea.benchmarks.hardset candidates)
    itrea screen-stage2 …      hard pool: the stage-2 list (seed 901)        (itrea.benchmarks.hardset stage2)
    itrea pool-freeze …        hard pool: strata, split, freeze              (itrea.benchmarks.hardset select)
    itrea check PATHS …        the independent exact check of stored incumbents
    itrea build-runs …         run directories -> datasets/runs.parquet, incumbents_checked.parquet
    itrea prereg …             the pre-registered analysis (D-002 v3.1)
    itrea lineage ROOT …       the descriptive Rank 0 lineage metric
    itrea reproduce …          re-derive datasets/ and paper/generated/ and compare
    itrea backoff-report …     A9 GPU back-off (D-018): the stage-0 / stage-1 rules applied, a verdict
    itrea mipfeas-report …     MIPfeas harness (D-020): A6m's paired calibration against GAMS's GB10 cuOpt row
    itrea exchange up|down|status|flush RUN_ID   the itrea-redis container (D-006)
    itrea bench-ls …           local-search benchmark (tools/bench_ls.py)
    itrea env                  the environment record (versions, hardware, git state)
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

# command -> (module or "path:<repo-relative file>", fixed leading arguments, one-line help)
COMMANDS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "ea-run": ("itrea.ea.run", (), "the EA process"),
    "scip-run": ("itrea.solvers.scip_run", (), "SCIP solver process"),
    "highs-run": ("itrea.solvers.highs_run", (), "HiGHS solver process"),
    "cuopt-run": ("itrea.solvers.cuopt_run", (), "cuOpt solver process (control arm)"),
    "pair": ("itrea.experiments.pair", (), "one run: launch, barrier, stop, exit codes, run.json"),
    "pool": ("itrea.experiments.pool", (), "run a job file under the D-002 / D-003 controls"),
    "make-jobs": ("itrea.experiments.make_jobs", (), "write job files (screen1, screen2, e1, e4, confirm, smoke, "
                  "backoff0, backoff1, mipfeas-cal)"),
    "screen-candidates": ("itrea.benchmarks.hardset", ("candidates",), "hard pool: screening candidates"),
    "screen-stage2": ("itrea.benchmarks.hardset", ("stage2",), "hard pool: stage-2 screening list"),
    "pool-freeze": ("itrea.benchmarks.hardset", ("select",), "hard pool: strata, split, freeze"),
    "check": ("itrea.analysis.check_solutions", (), "independent exact check of stored incumbents"),
    "build-runs": ("itrea.analysis.build_runs", (), "run directories -> datasets/*.parquet"),
    "prereg": ("itrea.analysis.prereg", (), "the pre-registered analysis (D-002 v3.1)"),
    "lineage": ("itrea.analysis.lineage", (), "descriptive Rank 0 lineage metric"),
    "reproduce": ("itrea.analysis.reproduce", (), "re-derive datasets/ and paper/generated/ and compare"),
    "backoff-report": ("itrea.analysis.backoff", (), "A9 GPU back-off (D-018): stage-0 / stage-1 rules, verdict"),
    "mipfeas-report": ("itrea.analysis.mipfeas", (), "MIPfeas harness (D-020): A6m vs GAMS's cuOpt GB10 row, verdict"),
    "exchange": ("itrea.experiments.exchange_ctl", (), "the itrea-redis container: up | down | status | flush RUN_ID"),
    "bench-ls": ("path:tools/bench_ls.py", (), "local-search benchmark"),
    "env": ("itrea.experiments.record_env", (), "environment record (versions, hardware, git state)"),
}


def usage() -> str:
    w = max(len(c) for c in COMMANDS)
    lines = ["usage: itrea <command> [arguments]   (itrea <command> --help for its arguments)", "", "commands:"]
    lines += [f"  {c:<{w}}  {h}" for c, (_, _, h) in COMMANDS.items()]
    return "\n".join(lines)


def resolve_main(command: str):
    """The `main(argv)` callable of a command (imported now). Raises KeyError for an unknown command."""
    target, _, _ = COMMANDS[command]
    if target.startswith("path:"):
        from itrea.paths import ROOT
        path = Path(ROOT) / target[5:]
        name = f"itrea_cli_{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(target)
    return mod.main


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print(f"itrea: unknown command {command!r}\n\n{usage()}", file=sys.stderr)
        return 2
    _, lead, _ = COMMANDS[command]
    fn = resolve_main(command)
    try:
        rc = fn(list(lead) + rest)
    except SystemExit as e:                      # modules whose argparse exits instead of returning
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return int(rc or 0)


if __name__ == "__main__":
    raise SystemExit(main())
