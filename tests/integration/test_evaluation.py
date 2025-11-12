import pytest
from larvaworld.lib import reg
from larvaworld.lib.process import Evaluation
from larvaworld.lib.sim import EvalRun

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.usefixtures("ensure_datasets_ready")
def test_evaluation():
    """Test evaluation with datasets ready (avoids HDF5 race)."""
    kws = {
        "refID": reg.default_refID,
        "cycle_curve_metrics": ["sv", "fov", "foa", "b"],
    }
    ev = Evaluation(**kws)
    assert ev.s_pars.exist_in(ev.target.step_data)
    assert ev.e_pars.exist_in(ev.target.endpoint_data)


@pytest.mark.usefixtures("ensure_datasets_ready")
def test_evaluation_simulation():
    """Run an evaluation simulation with datasets ready."""
    kws = {
        "refID": reg.default_refID,
        "modelIDs": ["RE_NEU_PHI_DEF", "RE_SIN_PHI_DEF"],
        "N": 5,
    }
    run = EvalRun(**kws)
    run.simulate()
    # run.plot_results()
    # run.plot_models()
