from lib import reg
from lib.process.larva_dataset import LarvaDataset


def test_replay() :
    from lib.sim.replay import ReplayRun
    refID =  'exploration.dish'
    replay_kws = {
        'normal': {
        },
        'dispersal': {
            'transposition': 'origin'
        },
        'fixed_point': {
            'agent_ids': [1],
            'close_view': True,
            'fix_point': 6,
        },
        'fixed_segment': {
            'agent_ids': [1],
            'close_view': True,
            'fix_point': 6,
            'fix_segment': -1,
        },
        'fixed_overlap': {
            'agent_ids': [1],
            'close_view': True,
            'fix_point': 6,
            'fix_segment': -1,
            'overlap_mode': True,
        },
    }

    for mode, kws in replay_kws.items() :
        rep = ReplayRun(refID=refID, id=f'{refID}_replay_{mode}', save_to= '. / media', **kws)
        rep.run()


def test_exp_run() :
    # from lib.sim.single_run import SingleRun
    from lib.sim.exp_run import ExpRun
    for exp in ['dish'] :
        conf=reg.expandConf('Exp', exp)
        conf.sim_params.duration=1

        # exp_run = SingleRun(id=f'test_{exp}',**conf)

        # exp_run = SingleRun(vis_kwargs = reg.get_null(name='visualization', mode='video', video_speed=60), **conf)
        # exp_run.run()

        exp_run = ExpRun(**conf)
        exp_run.simulate()
        for d in exp_run.datasets :
            assert isinstance(d, LarvaDataset)