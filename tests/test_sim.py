import larvaworld.lib.sim.genetic_algorithm
from larvaworld.lib import reg, sim


def test_replay() :
    refID =  'exploration.dish'
    replay_kws = {
        'normal': {
            'time_range': (60,80)
        },
        'dispersal': {
            'transposition': 'origin',
            'time_range': (10,30)
        },
        'fixed_point': {
            'agent_ids': [1],
            'close_view': True,
            'fix_point': 6,
            'time_range': (80, 100)
        },
        'fixed_segment': {
            'agent_ids': [1],
            'close_view': True,
            'fix_point': 6,
            'fix_segment': -1,
            'time_range': (100, 130)
        },
        'fixed_overlap': {
            'agent_ids': [1],
            'close_view': True,
            'fix_point': 6,
            'fix_segment': -1,
            'overlap_mode': True
        },
    }

    for mode, kws in replay_kws.items() :
        parameters = {
            'refID' : refID,
        **kws
                      }
        rep = sim.ReplayRun(parameters=parameters, id=f'{refID}_replay_{mode}', save_to= './media')
        rep.run()

def test_exp_run() :
    for exp in ['chemotaxis'] :
        conf=reg.stored.getExp(exp)
        conf.sim_params.duration=1
        exp_run = sim.ExpRun(parameters=conf)
        exp_run.simulate()
        for d in exp_run.datasets :
            assert isinstance(d, larvaworld.LarvaDataset)


def test_GA() :
    conf=reg.stored.expand('Ga', 'realism')
    conf.ga_select_kws.Ngenerations = 5

    ga_run = sim.GAlauncher(parameters=conf)
    best1=ga_run.run()
    print(best1)
    assert best1 is not None

    conf.offline=True
    conf.show_screen=False
    ga_run = sim.GAlauncher(parameters=conf)
    best2=ga_run.run()
    print(best2)
    assert best2 is not None

def test_evaluation() :
    # refID = 'exploration.merged_dishes'
    # mIDs = ['RE_NEU_PHI_DEF', 'RE_SIN_PHI_DEF']
    parameters = reg.get_null('Eval', **{
        'refID': 'exploration.merged_dishes',
        'modelIDs': ['RE_NEU_PHI_DEF', 'RE_SIN_PHI_DEF'],
        # 'dataset_ids': dIDs,
        'N': 3,
        # 'offline': False,
    })
    evrun = sim.EvalRun(parameters=parameters, id=id,show=False)

    # evrun = sim.EvalRun(refID=refID, modelIDs=mIDs, N=3, show=False)
    evrun.simulate()
    evrun.plot_results()
    evrun.plot_models()








def xtest_batch_run() :
    for exp in ['PItest_off'] :
        conf=reg.stored.expand('Batch', exp)
        # conf.sim_params.duration=1
        batch_run = sim.BatchRun(id=f'test_{exp}',batch_type=exp,**conf)
        batch_run.simulate()

