"""Shared test setup: no test may depend on the state of the real machine.

The runner's gates read live values — MemAvailable (≈ 23 GB with the model servers up, ≈ 109 GB with them stopped) and
fan-governor's rest flag — so a test that launches through the pool or the pair runner could pass at night and stall in
the morning (it did, 2026-10-06 07:51). Every test therefore sees plenty of memory and no rest flag; tests that check a
gate set their own values on top of this, and `test_gates.py`, which tests the readers themselves, is left alone."""
import pytest


@pytest.fixture(autouse=True)
def _box_independent(request, monkeypatch):
    if request.module.__name__.rsplit(".", 1)[-1] == "test_gates":
        return
    from itrea.experiments import gates
    monkeypatch.setattr(gates, "mem_available_gb", lambda *a, **k: 500.0)
    monkeypatch.setattr(gates, "rest_due", lambda *a, **k: False)
