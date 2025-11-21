import pytest

from larvaworld.lib import sim
from larvaworld.lib.process import LarvaDataset

# NOTE :    This test requires the box2d-py package

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.optional_dep]

pytest.importorskip("Box2D")


def xtest_box2d_experiments():
    try:
        import Box2D

        ids = ["realistic_imitation"]
        for id in ids:
            r = sim.ExpRun.from_ID(id, store_data=False)
            for d in r.datasets:
                assert isinstance(d, LarvaDataset)
    except ImportError:
        print(
            "WARNING : Module box2d-py is not installed. Larvaworld Box2D extension tests aborted"
        )
