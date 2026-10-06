"""The `itrea` entry point (`itrea.cli`): usage, unknown commands, every command resolving to its module's `main`,
the fixed leading arguments, and an EA run through the CLI on a toy model."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from itrea import cli
from itrea.benchmarks import toy

ROOT = Path(__file__).resolve().parents[1]


def _env():
    return dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))


def _cli(*args, timeout=120):
    pr = subprocess.run([sys.executable, "-m", "itrea.cli", *args], env=_env(), capture_output=True, timeout=timeout)
    return pr.returncode, pr.stdout.decode(errors="replace"), pr.stderr.decode(errors="replace")


def test_usage_and_unknown_command(capsys):
    assert cli.main([]) == 2
    assert "usage: itrea" in capsys.readouterr().out
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for c in cli.COMMANDS:
        assert c in out
    assert cli.main(["no-such-command"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_requested_commands_exist():
    want = {"ea-run", "scip-run", "highs-run", "cuopt-run", "pair", "pool", "make-jobs", "screen-candidates",
            "pool-freeze", "check", "build-runs", "prereg", "reproduce", "exchange", "bench-ls", "env"}
    assert want <= set(cli.COMMANDS)


@pytest.mark.parametrize("command", sorted(cli.COMMANDS))
def test_every_command_resolves(command):
    fn = cli.resolve_main(command)
    assert callable(fn)


def test_leading_arguments_are_passed(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "resolve_main", lambda c: (lambda argv: seen.append((c, argv)) or 0))
    assert cli.main(["screen-candidates", "--out", "x.csv"]) == 0
    assert cli.main(["pool-freeze"]) == 0
    assert cli.main(["exchange", "status"]) == 0
    assert seen == [("screen-candidates", ["candidates", "--out", "x.csv"]), ("pool-freeze", ["select"]),
                    ("exchange", ["status"])]


def test_exit_code_of_a_command_that_raises_systemexit(monkeypatch):
    def fake(argv):
        raise SystemExit(3)
    monkeypatch.setattr(cli, "resolve_main", lambda c: fake)
    assert cli.main(["pair", "--bogus"]) == 3


@pytest.mark.parametrize("command", ["ea-run", "scip-run", "highs-run", "cuopt-run", "pair", "make-jobs", "check"])
def test_help_through_the_cli(command):
    rc, out, err = _cli(command, "--help")
    assert rc == 0, err
    assert "usage" in out.lower()


def test_ea_run_through_the_cli(tmp_path):
    path = toy.write_mps(toy.knapsack(20), tmp_path / "k20.mps")
    rd = tmp_path / "run"
    rc, out, err = _cli("ea-run", "--instance", str(path), "--seed", "0", "--budget", "1", "--run-dir", str(rd),
                        "--no-barrier", "--backend", "numpy", "--set", "population.size=32",
                        "--set", "population.offspring_per_gen=32", "--gpu-mem-gb", "1")
    assert rc == 0, err
    s = json.loads((rd / "ea.summary.json").read_text())
    assert s["exit_reason"] == "budget" and s["backend"] == "numpy" and s["P"] == 32


def test_env_prints_a_record():
    rc, out, err = _cli("env")
    assert rc == 0, err
    rec = json.loads(out)
    assert "versions" in rec
