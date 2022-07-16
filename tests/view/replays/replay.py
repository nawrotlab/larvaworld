from lib.sim.replay.replay import ReplayRun

mode = 'fixed_overlap'
refID = 'exploration.dish'

replay_kws = {
    'normal': {
        # 'id': f'{refID}_replay'
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
