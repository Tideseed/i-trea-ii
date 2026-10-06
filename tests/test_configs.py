"""The configuration files in `configs/` (itrea.config.EAConfig).

`configs/itrea2-default.yaml` is the pre-registered starting point (vault D-002 v3): it must hold every parameter at
its default in `src/itrea/config.py`, and the hash recorded in its header must be the hash of those defaults — a
default changed in the code without regenerating the file would otherwise go unnoticed until a run record disagreed.
Every other file is partial (unlisted keys keep their defaults) and must change at least one parameter; the
mechanism ablations must change exactly the one parameter their hypothesis is about.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from itrea.config import EAConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEFAULT = CONFIGS / "itrea2-default.yaml"
ALL = sorted(CONFIGS.rglob("*.yaml"))
PARTIAL = [p for p in ALL if p != DEFAULT]


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and k not in ("weights",):          # operator weight maps are one value each
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _changed(cfg: EAConfig) -> set[str]:
    a, b = _flat(cfg.to_dict()), _flat(EAConfig().to_dict())
    return {k for k in a if k != "name" and a[k] != b[k]}


def test_the_expected_files_exist():
    names = {p.relative_to(CONFIGS).as_posix() for p in ALL}
    assert {"itrea2-default.yaml", "itrea1-faithful.yaml", "ablations/x0.yaml", "ablations/y0.yaml",
            "ablations/no_bound_demote.yaml", "ablations/one_way_ea_to_solver.yaml",
            "ablations/one_way_solver_to_ea.yaml", "ablations/post_frontier.yaml"} <= names


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.relative_to(CONFIGS).as_posix())
def test_every_config_loads_and_validates(path):
    cfg = EAConfig.from_yaml(path)                                   # unknown keys and invalid values raise
    assert cfg.name and cfg.name != "default"                        # every file names itself
    assert re.fullmatch(r"[0-9a-f]{12}", cfg.hash())


def test_default_file_equals_the_code_defaults():
    cfg = EAConfig.from_yaml(DEFAULT)
    raw = yaml.safe_load(DEFAULT.read_text())
    assert cfg.name == "itrea2-default"
    a, b = cfg.to_dict(), EAConfig().to_dict()
    a.pop("name"), b.pop("name")
    assert a == b
    assert _flat(raw).keys() - {"name"} == _flat(b).keys()          # complete: no key left to an implicit default


def test_default_file_hash_is_the_code_hash_and_is_recorded():
    assert EAConfig.from_yaml(DEFAULT).hash() == EAConfig().hash()
    m = re.search(r"config hash ([0-9a-f]{12})", DEFAULT.read_text())
    assert m, "the header must record the configuration hash"
    assert m.group(1) == EAConfig().hash(), "regenerate configs/itrea2-default.yaml (command in its header)"


@pytest.mark.parametrize("path", PARTIAL, ids=lambda p: p.relative_to(CONFIGS).as_posix())
def test_partial_files_change_something(path):
    cfg = EAConfig.from_yaml(path)
    assert _changed(cfg), f"{path.name} changes no parameter (a no-op ablation would pass silently)"
    assert cfg.hash() != EAConfig().hash()


@pytest.mark.parametrize("name, key, value", [("x0", "ranks.x", 0.0), ("y0", "ranks.y", 0.0),
                                              ("no_bound_demote", "ranks.rank0_bound_demote", False)])
def test_single_variable_ablations(name, key, value):
    """A5x0 (H6) and the other single-switch ablations differ from the default in that switch only."""
    cfg = EAConfig.from_yaml(CONFIGS / "ablations" / f"{name}.yaml")
    assert _changed(cfg) == {key}
    assert _flat(cfg.to_dict())[key] == value


def test_one_way_ablations_switch_whole_directions():
    e2s = EAConfig.from_yaml(CONFIGS / "ablations" / "one_way_ea_to_solver.yaml").exchange
    assert e2s.to_solver and not (e2s.from_solver or e2s.use_dual_bound or e2s.use_root_lp)
    s2e = EAConfig.from_yaml(CONFIGS / "ablations" / "one_way_solver_to_ea.yaml").exchange
    assert not (s2e.to_solver or s2e.post_frontier) and s2e.from_solver
