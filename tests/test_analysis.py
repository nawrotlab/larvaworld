import pytest
from larvaworld.lib.sim import ExpRun
from larvaworld.lib.process.dataset import LarvaDataset

expIDs = [
    "dispersion",
    "chemorbit"
]

@pytest.mark.parametrize("id", expIDs)
def test_experiment_analysis(id):
    r = ExpRun.from_ID(id, duration=1, store_data=False)
    for d in r.datasets:
        assert isinstance(d, LarvaDataset)
    r.analyze()
    for d in r.datasets:
        assert isinstance(d, LarvaDataset)

