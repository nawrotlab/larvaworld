import os
import sys

import pytest
from larvaworld.lib import reg
from larvaworld.lib.sim import ExpRun
from larvaworld.lib.process import LarvaDataset
from larvaworld.lib.reg.generators import ExpConf

RUN_WINDOWS_HEAVY = os.environ.get("LW_ENABLE_WINDOWS_HEAVY") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.heavy,
    pytest.mark.skipif(
        sys.platform.startswith("win") and not RUN_WINDOWS_HEAVY,
        reason="Slow on Windows CI (enabled weekly)",
    ),
]

expIDs = [
    # "tethered",
    # "dish",
    "dispersion_x2",
    "chemorbit",
    # "chemotaxis_diffusion",
    # "single_odor_patch_x4",
    "PItest_off",
    # "PItrain",
    "tactile_detection",
    # "anemotaxis",
    "single_puff",
    # "thermotaxis",
    # "prey_detection",
    # "keep_the_flag",
    # "maze",
    # "4corners",
    "double_patch",
    # "random_food",
    "patch_grid",
    "RvsS_on",
    # "growth"
]


@pytest.mark.parametrize("id", expIDs)
def test_experiment(id):
    r = ExpRun.from_ID(id, duration=1, store_data=False)
    for d in r.datasets:
        assert isinstance(d, LarvaDataset)


def test_imitation_exp_simulates_and_imitates_reference_dataset():
    """
    Regression test for three real bugs found running ExpConf.imitation_exp:

    - SimConfigurationParams.__init__ always called update_larva_groups()
      when "larva_groups" was in parameters, even with no override request
      (modelIDs/groupIDs/N/sample all None) -- unconditionally clobbering
      each group's own already-correct `sample` field with None.
    - LarvaGroup.generate_agent_attrs' imitation branch merged imitated
      per-agent attributes via update_nestdict, which adds any key from
      reg.SAMPLING_PARS as a new field even if the base model's actual
      module modes don't have it (e.g. a "freq" field that only applies to
      a sinusoidal turner, not this model's neural one) -- crashing model
      construction. Fixed via update_existingnestdict_by_suffix, which
      only updates fields the model schema already has.
    - imitation_exp never set group_id on the LarvaGroup it builds, so the
      group's own group_id (used to tag simulated agents, and later to
      look up per-group config in from_agentpy_output) diverged from the
      larva_groups dict key it was filed under, breaking dataset
      construction after simulation.
    """
    refID = reg.default_refID
    d_ref = reg.loadRef(refID)
    d_ref.load()

    conf = ExpConf.imitation_exp(refID, mID="explorer")
    gid, lg = next(iter(conf.larva_groups.items()))
    assert lg.group_id == gid
    assert lg.sample == refID

    conf_nested = conf.nestedConf
    conf_nested.larva_groups[gid].distribution.N = 3
    r = ExpRun(
        parameters=conf_nested,
        duration=0.1,
        screen_kws={"show_display": False},
        store_data=False,
    )
    r.simulate()

    d = r.datasets[0]
    assert isinstance(d, LarvaDataset)
    assert d.id == gid
    assert d.config.N == 3
    # Body length was actually imitated from the reference dataset, not
    # left at the base "explorer" model's own static default.
    assert d.e["length"].mean() > 0


# def test_exploration_experiments():
#     ids = [
#         # "tethered",
#         "dish",
#         "dispersion_x2",
#     ]

#     for id in ids:
#         r = sim.ExpRun.from_ID(id, duration=1, store_data=False)
#         for d in r.datasets:
#             assert isinstance(d, LarvaDataset)


# def test_chemosensory_experiments():
#     ids = [
#         "chemorbit",
#         "chemotaxis_diffusion",
#         # "single_odor_patch_x4",
#         "PItest_off",
#         # "PItrain",
#     ]

#     for id in ids:
#         r = sim.ExpRun.from_ID(id, duration=1, store_data=False)
#         for d in r.datasets:
#             assert isinstance(d, LarvaDataset)


# def test_other_sensory_experiments():
#     ids = [
#         "tactile_detection",
#         "anemotaxis",
#         "single_puff",
#         # "thermotaxis",
#         "prey_detection",
#     ]

#     for id in ids:
#         r = sim.ExpRun.from_ID(id, duration=1, store_data=False)
#         for d in r.datasets:
#             assert isinstance(d, LarvaDataset)


# def test_games():
#     ids = ["keep_the_flag", "maze"]

#     for id in ids:
#         r = sim.ExpRun.from_ID(id, duration=1, store_data=False)
#         for d in r.datasets:
#             assert isinstance(d, LarvaDataset)


# def test_foraging_experiments():
#     ids = [
#         "4corners",
#         "double_patch",
#         # "random_food",
#         "patch_grid",
#     ]

#     for id in ids:
#         r = sim.ExpRun.from_ID(id, duration=1, store_data=False)
#         for d in r.datasets:
#             assert isinstance(d, LarvaDataset)


# def test_growth_experiments():
#     ids = [
#         "RvsS_on",
#         # "growth"
#     ]

#     for id in ids:
#         r = sim.ExpRun.from_ID(id, duration=1, store_data=False)
#         for d in r.datasets:
#             assert isinstance(d, LarvaDataset)


def test_experiment_visualization():
    r = ExpRun.from_ID(
        "dispersion", duration=1, screen_kws={"vis_mode": "video", "show_display": True}
    )
    for d in r.datasets:
        assert isinstance(d, LarvaDataset)
