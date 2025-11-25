import os
import sys

import pytest

from larvaworld.lib import reg, sim

RUN_WINDOWS_HEAVY = os.environ.get("LW_ENABLE_WINDOWS_HEAVY") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.heavy,
    pytest.mark.skipif(
        sys.platform.startswith("win") and not RUN_WINDOWS_HEAVY,
        reason="Slow on Windows CI (enabled weekly)",
    ),
]


def test_genetic_algorithm_no_video():
    """Run a genetic algorithm experiment without visualization on screen."""
    exp = "realism"
    ga1 = sim.GAlauncher(experiment=exp, duration=0.5)
    ga1.selector.Ngenerations = 3
    ga1.selector.Nagents = 20
    best1 = ga1.simulate()
    print(best1)
    assert best1 is not None


def test_genetic_algorithm_with_video():
    """Run a genetic algorithm experiment with visualization on screen."""
    exp = "realism"
    p = reg.conf.Ga.expand(exp)
    p.ga_select_kws.Ngenerations = 3
    p.ga_select_kws.Nagents = 20
    ga2 = sim.GAlauncher(
        parameters=p,
        duration=0.5,
        screen_kws={"show_display": True, "vis_mode": "video"},
    )
    best2 = ga2.simulate()
    print(best2)
    assert best2 is not None
