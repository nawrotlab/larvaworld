from larvaworld.lib import reg, sim
from larvaworld.lib.process.dataset import LarvaDataset


def test_exploration_experiments():
    ids = ["tethered", "dish", "dispersion_x2"]

    for id in ids:
        r = sim.ExpRun.from_ID(id, duration=1, store_data=False)
        for d in r.datasets:
            assert isinstance(d, LarvaDataset)


def test_chemosensory_experiments():
    ids = [
        "chemorbit",
        "chemotaxis_diffusion",
        "single_odor_patch_x4",
        "PItest_off",
        "PItrain",
    ]

    for id in ids:
        r = sim.ExpRun.from_ID(id, duration=1, store_data=False)
        for d in r.datasets:
            assert isinstance(d, LarvaDataset)


def test_other_sensory_experiments():
    ids = [
        "tactile_detection",
        "anemotaxis",
        "single_puff",
        "thermotaxis",
        "prey_detection",
    ]

    for id in ids:
        r = sim.ExpRun.from_ID(id, duration=1, store_data=False)
        for d in r.datasets:
            assert isinstance(d, LarvaDataset)


def test_games():
    ids = ["keep_the_flag", "maze"]

    for id in ids:
        r = sim.ExpRun.from_ID(id, duration=1, store_data=False)
        for d in r.datasets:
            assert isinstance(d, LarvaDataset)


def test_foraging_experiments():
    ids = ["4corners", "double_patch", "random_food", "patch_grid"]

    for id in ids:
        r = sim.ExpRun.from_ID(id, duration=1, store_data=False)
        for d in r.datasets:
            assert isinstance(d, LarvaDataset)


def test_growth_experiments():
    ids = ["RvsS_on", "growth"]

    for id in ids:
        r = sim.ExpRun.from_ID(id, duration=1, store_data=False)
        for d in r.datasets:
            assert isinstance(d, LarvaDataset)


def xtest_evaluation():
    # refID = 'exploration.merged_dishes'
    # mIDs = ['RE_NEU_PHI_DEF', 'RE_SIN_PHI_DEF']
    parameters = reg.par.get_null(
        "Eval",
        **{
            "refID": "exploration.merged_dishes",
            "modelIDs": ["RE_NEU_PHI_DEF", "RE_SIN_PHI_DEF"],
            # 'groupIDs': dIDs,
            "N": 3,
            # 'offline': False,
        },
    )
    evrun = sim.EvalRun(parameters=parameters, id=id, show=False)

    # evrun = sim.EvalRun(refID=refID, modelIDs=mIDs, N=3, show=False)
    evrun.simulate()
    evrun.plot_results()
    evrun.plot_models()


# def xtest_batch_run() :
#     for exp in ['PItest_off'] :
#         conf=reg.conf.Batch.expand(exp)
#         batch_run = sim.BatchRun(id=f'test_{exp}',batch_type=exp,**conf)
#         batch_run.simulate()
