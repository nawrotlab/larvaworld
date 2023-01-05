from lib import reg

from lib.sim.replay import ReplayRun

mode = 'normal'
# refID = 'Rehydration/AttP2.Deprived'
refID = 'None.40controls'
# refID = 'naive_locomotion.20controls'
# refID = 'exploration.150controls'

replay_kws = {
    'normal': {
        # 'id': f'{refID}_replay'
    },
    'dispersal': {
        'transposition': 'origin'
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
