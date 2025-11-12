"""
Smoke tests for simulation modules.

Minimal entry point test, NO heavy computation.
Tests that module can be imported and has expected structure.
"""

import importlib
import pytest


@pytest.mark.fast
def test_agent_simulations_entry_point():
    """
    Test agent simulation entry point exists.

    Verifies that agent_simulations module can be imported
    and has callable simulation functions.
    """
    mod = importlib.import_module("larvaworld.lib.sim.agent_simulations")

    # Find public simulation entry points
    candidates = [
        n
        for n in dir(mod)
        if (n.startswith("run") or n.startswith("sim"))
        and callable(getattr(mod, n))
        and not n.startswith("_")
    ]

    assert len(candidates) > 0, "No public simulation entry found"
