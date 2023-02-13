from larvaworld.lib import reg

from larvaworld.lib.sim.dataset_replay import ReplayRun

mode = 'fixed_segment'
# refID = 'Rehydration/AttP2.Deprived'
refID = 'exploration.dish'
# refID = 'None.40controls'
# refID = 'naive_locomotion.20controls'
# refID = 'exploration.150controls'

replay_kws = {
    'normal': {
        'time_range': (60,80)
    },
    'dispersal': {
        'transposition': 'origin',
'time_range': (10,30)
    },
    'fixed_point': {
        # 'id':f'{refID}_replay_solo_fixed_point',
        'agent_ids': [1],
        'close_view': True,
        'fix_point': 6,
    },
    'fixed_segment': {
        # 'id':f'{refID}_replay_solo_fixed_point',
        'agent_ids': [1],
        'close_view': True,
        'fix_point': 6,
        'fix_segment': -1,
    },
    'fixed_overlap': {
        # 'id':f'{refID}_replay_solo_fixed_point',
        'agent_ids': [1],
        'close_view': True,
        'fix_point': 6,
        'fix_segment': -1,
        'overlap_mode': True,
    },
}

rep = ReplayRun(refID=refID, id=f'{refID}_replay_{mode}', save_to='./media', **replay_kws[mode])

rep.run()
