import pytest
from larvaworld.lib import reg
from larvaworld.lib.process import Evaluation
from larvaworld.lib.sim import EvalRun

pytestmark = [pytest.mark.integration, pytest.mark.heavy]


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


@pytest.mark.usefixtures("ensure_datasets_ready")
def test_evaluation_simulation_uses_target_dataset_arena():
    """
    EvalRun.simulate()'s live path (offline is a plain param.Boolean,
    never actually None, so this is the path always taken) used to build
    no explicit env_params, silently falling back to the "dispersion"
    experiment's own default rectangular arena instead of the reference
    dataset's actual arena (e.g. a circular dish) -- so a model evaluated
    against a circular-dish dataset was simulated in a rectangular arena.
    """
    run = EvalRun(
        refID=reg.default_refID,
        modelIDs=["RE_NEU_PHI_DEF"],
        N=2,
        duration=0.1,
        screen_kws={"show_display": False},
        store_data=False,
    )
    run.analyze = lambda *a, **kw: None  # error-plot analysis is out of scope here
    target_arena = run.target.config.env_params.arena

    run.simulate()

    sim_arena = run.datasets[0].config.env_params.arena
    assert sim_arena.geometry == target_arena.geometry
    assert sim_arena.dims == target_arena.dims
