"""Shared test setup: no test may depend on the state of the real machine.

The runner's gates read live values — MemAvailable (≈ 23 GB with the model servers up, ≈ 109 GB with them stopped) and
fan-governor's rest flag — so a test that launches through the pool or the pair runner could pass at night and stall in
the morning (it did, 2026-10-06 07:51). Every test therefore sees plenty of memory and no rest flag; tests that check a
gate set their own values on top of this, and `test_gates.py`, which tests the readers themselves, is left alone.
Since D-021 the runner also reads the fan governor (state file, clock-cap log) and thermal-watch's daily log: every
test sees a live governor on AUTO with its cap off, no cap interval and no 98 °C stop."""
import time

import pytest


@pytest.fixture(autouse=True)
def _box_independent(request, monkeypatch):
    if request.module.__name__.rsplit(".", 1)[-1] == "test_gates":
        return
    from itrea.experiments import gates
    monkeypatch.setattr(gates, "mem_available_gb", lambda *a, **k: 500.0)
    monkeypatch.setattr(gates, "rest_due", lambda *a, **k: False)
    monkeypatch.setattr(gates, "fan_state", lambda *a, **k: dict(
        gates.parse_fan_state(time.strftime("%Y-%m-%dT%H:%M:%S+00:00") + " state=auto for=60s zone=40 gpu=3W "
                              "rest=no cap=0"), raw="(test) state=auto cap=0", age_s=1.0))
    monkeypatch.setattr(gates, "cap_intervals", lambda *a, **k: [])
    monkeypatch.setattr(gates, "thermal_watch_stops", lambda *a, **k: [])
