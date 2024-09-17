from larvaworld.lib import reg, aux

from larvaworld.lib.sim.dataset_replay import ReplayRun

# mode = '2segs'
# mode = 'fixed_overlap'
mode = 'all_segs'
# mode = 'normal'
# refID = 'Rehydration/AttP2.Deprived'
# refID = 'exploration.dish'
refID = 'exploration.40controls'
# refID = 'naive_locomotion.20controls'
# refID = 'exploration.150controls'
dataset=reg.loadRef(refID)

replay_kws = {
'2segs': {
            'draw_Nsegs': 2
        },
        'all_segs': {
            'draw_Nsegs': dataset.config.Npoints-1
        },
    'normal': {
        'time_range': (10,70)
    },
    'dispersal': {
        'transposition': 'origin',
'time_range': (30,130)
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
        'fix_segment': 'rear',
    },
    'fixed_overlap': {
        # 'id':f'{refID}_replay_solo_fixed_point',
        'agent_ids': [1],
        'close_view': True,
        'fix_point': 6,
        'fix_segment': 'rear',
        'overlap_mode': True,
    },
}
parameters = reg.gen.Replay(**aux.AttrDict({
            'refID': refID,
            # 'dataset' : dataset,
            **replay_kws[mode]
        })).nestedConf
rep = ReplayRun(parameters=parameters,dataset=dataset, id=f'{refID}_replay_{mode}', dir= f'./media/{mode}')
output=rep.run()
