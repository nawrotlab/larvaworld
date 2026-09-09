import pytest
from larvaworld.lib import reg
from larvaworld.lib.reg.graph import GraphRegistry
from larvaworld.lib.sim import ExpRun
from larvaworld.lib.process import LarvaDataset

pytestmark = [pytest.mark.integration, pytest.mark.heavy]

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


@pytest.mark.usefixtures("ensure_datasets_ready")
def test_plot_real_and_simulated_datasets_together():
    """Plot functions must accept a real (bundled) dataset and a freshly
    simulated one in the same call, not just same-origin datasets."""
    d_real = reg.loadRef(reg.default_refID)
    d_real.load()

    r = ExpRun.from_ID("dish", duration=0.5, store_data=False)
    d_sim = r.datasets[0]
    assert isinstance(d_sim, LarvaDataset)

    gr = GraphRegistry()
    for plot_id in ["trajectories", "epochs"]:
        result = gr.run(
            plot_id,
            datasets=[d_real, d_sim],
            labels=[d_real.id, d_sim.id],
            return_fig=True,
        )
        fig = result[0] if isinstance(result, tuple) else result
        assert hasattr(fig, "savefig")
