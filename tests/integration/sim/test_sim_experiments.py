import pytest
from larvaworld.lib.sim import ExpRun
from larvaworld.lib.process import LarvaDataset

pytestmark = [pytest.mark.integration, pytest.mark.slow]

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
