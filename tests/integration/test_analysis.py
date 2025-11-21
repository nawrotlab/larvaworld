import pytest
from larvaworld.lib.sim import ExpRun
from larvaworld.lib.process import LarvaDataset

pytestmark = [pytest.mark.integration, pytest.mark.slow]

expIDs = ["dispersion", "chemorbit"]


@pytest.mark.usefixtures("ensure_datasets_ready")
@pytest.mark.parametrize("id", expIDs)
def test_experiment_analysis(id):
    """Test experiment analysis with datasets ready (avoids HDF5 race)."""
    r = ExpRun.from_ID(id, duration=1, store_data=False)
    for d in r.datasets:
        assert isinstance(d, LarvaDataset)
    r.analyze()
    for d in r.datasets:
        assert isinstance(d, LarvaDataset)
