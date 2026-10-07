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
    assert {"itrea2-default.yaml", "itrea1-faithful.yaml", "itrea2-adaptive.yaml", "itrea2-adaptive-lps.yaml",
            "itrea2-adaptive-lps-rc.yaml", "itrea2-dev-rc-3way.yaml", "ablations/x0.yaml",
            "ablations/y0.yaml", "ablations/no_bound_demote.yaml", "ablations/one_way_ea_to_solver.yaml",
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


def test_adaptive_config_switches_on_the_strategy_and_the_lp_worker():
    """itrea2-adaptive = the default + the self-adaptive repair strategy + an LP point of the EA's own (D-010)."""
    cfg = EAConfig.from_yaml(CONFIGS / "itrea2-adaptive.yaml")
    assert _changed(cfg) == {"strategy.enabled", "seeding.lp_source"}
    assert cfg.strategy.enabled and cfg.seeding.lp_source == "highs_async"
    assert cfg.strategy.strategies() == [("l2", False), ("l2", True), ("rhs", False), ("rhs", True)]
    assert cfg.ranks.viol_norm == "l2"                               # the survival keeps one scale for everyone


def test_disabled_strategy_section_is_outside_the_hash():
    """While strategy.enabled is false the section is inert: changing its other fields keeps the hash (so every
    configuration written before the section existed keeps its hash); enabling it changes the hash."""
    base = EAConfig()
    assert base.with_overrides({"strategy.row_scales": ["rhs"], "strategy.mutation_rate": 0.5}).hash() == base.hash()
    on = base.with_overrides({"strategy.enabled": True})
    assert on.hash() != base.hash()
    assert on.with_overrides({"strategy.mutation_rate": 0.5}).hash() != on.hash()


# the hashes recorded in run records and the Development log before the `repair_clock` section existed
LPS1 = {"repair.max_steps": 20000, "repair.stall_steps": 1000}


def test_recorded_hashes_are_unchanged():
    """Every section and field added later is left out of the hash while inert, so the recorded hashes stand: the
    default 9ac4a93bf93b, itrea2-adaptive 584ddee83747, itrea2-adaptive-lps a2ad94e9188d, lps1 (adaptive-lps + the
    c01 repair budget) 4a96b872ffdc, itrea2-adaptive-lps-rc (the repair clock in shadow mode) 26481ba0c675 and the
    clock's Stage 2 (rc + mode active, D-012) 1d12ad490dd0."""
    assert EAConfig().hash() == "9ac4a93bf93b"
    assert EAConfig.from_yaml(CONFIGS / "itrea2-adaptive.yaml").hash() == "584ddee83747"
    lps = EAConfig.from_yaml(CONFIGS / "itrea2-adaptive-lps.yaml")
    assert lps.hash() == "a2ad94e9188d"
    assert lps.with_overrides(LPS1).hash() == "4a96b872ffdc"
    rc = EAConfig.from_yaml(CONFIGS / "itrea2-adaptive-lps-rc.yaml")
    assert rc.hash() == "26481ba0c675"
    assert rc.with_overrides({"repair_clock.mode": "active"}).hash() == "1d12ad490dd0"


def test_from_cuopt_is_outside_the_hash_while_off():
    """exchange.from_cuopt (arm A9, D-015) was added after hashes were recorded: false is left out of the hash
    (_LATE_DEFAULTS), true changes it."""
    base = EAConfig()
    assert base.exchange.from_cuopt is False
    on = base.with_overrides({"exchange.from_cuopt": True})
    assert on.hash() != base.hash()
    with pytest.raises(ValueError, match="from_cuopt"):
        base.with_overrides({"exchange.from_cuopt": "yes"})


def test_dev_rc_3way_is_the_stage2_configuration():
    """itrea2-dev-rc-3way = itrea2-adaptive-lps-rc with the repair clock active — Stage 2's effective configuration
    (1d12ad490dd0); it leaves exchange.from_cuopt off (the runner switches it on for A9), so A9 and its matched control
    A3x run one configuration file."""
    cfg = EAConfig.from_yaml(CONFIGS / "itrea2-dev-rc-3way.yaml")
    rc = EAConfig.from_yaml(CONFIGS / "itrea2-adaptive-lps-rc.yaml")
    assert _changed(cfg) == _changed(rc) | {"repair_clock.mode"}
    assert cfg.repair_clock.enabled and cfg.repair_clock.mode == "active" and not cfg.exchange.from_cuopt
    assert cfg.hash() == "1d12ad490dd0" == rc.with_overrides({"repair_clock.mode": "active"}).hash()


def test_disabled_repair_clock_is_outside_the_hash():
    base = EAConfig()
    assert base.with_overrides({"repair_clock.mode": "active", "repair_clock.cap_s": 3.0,
                                "repair_clock.level_min_log2": 9}).hash() == base.hash()
    on = base.with_overrides({"repair_clock.enabled": True})
    assert on.hash() != base.hash()
    assert on.with_overrides({"repair_clock.mode": "active"}).hash() != on.hash()


def test_repair_clock_config_is_lps1_plus_the_clock_in_shadow_mode():
    """itrea2-adaptive-lps-rc = itrea2-adaptive-lps + the c01 repair budget (lps1) + repair_clock enabled, shadow."""
    cfg = EAConfig.from_yaml(CONFIGS / "itrea2-adaptive-lps-rc.yaml")
    assert _changed(cfg) == {"strategy.enabled", "seeding.lp_source", "lp_stream.enabled", "repair.max_steps",
                             "repair.stall_steps", "repair_clock.enabled"}
    assert cfg.repair_clock.enabled and cfg.repair_clock.mode == "shadow"
    lps1 = EAConfig.from_yaml(CONFIGS / "itrea2-adaptive-lps.yaml").with_overrides(LPS1)
    assert cfg.hash() == lps1.with_overrides({"repair_clock.enabled": True}).hash() != "4a96b872ffdc"
