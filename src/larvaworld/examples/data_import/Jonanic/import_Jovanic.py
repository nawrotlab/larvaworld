from larvaworld.lib import reg, aux, plot
from larvaworld.lib.process.building import import_datasets
from larvaworld.lib.process.dataset import LarvaDataset


kws0 = {
    'datagroup_id': 'Jovanic lab',
    # 'group_id': 'exploration',
    # 'add_reference': True,
    'enrich' : True,
    'merged' : False,
    'match_ids' : False,
}

# kws2 = {
#     'parent_dir': 'SS888',
#     'source_ids': ['AttP240', 'SS888Imp', 'SS888'],
#     # 'id': f'merged_dishes',
#     **kws0
# }

kws1 = {
    'parent_dir': 'Rehydration',
    'source_ids': ['Fe','Fe_w', 'Pd', 'Pd_w'],
    'ids': ['Fed','Fed -rehydrated', 'Sucrose', 'Sucrose -rehydrated'],
    'refIDs': ['Rehydration.Fed', 'Rehydration.Fed_w', 'Rehydration.Sucrose', 'Rehydration.Sucrose_w'],
    'colors':['black','grey', 'red', 'yellow'],
    'time_slice':(0,60),
    **kws0
}

kws2 = {
    'parent_dir': 'Rehydration_baseline',
    'source_ids': ['Fe','Fe_w', 'Pd', 'Pd_w'],
    'ids': ['Fed','Fed -rehydrated', 'Sucrose', 'Sucrose -rehydrated'],
    'refIDs': ['Rehydration_baseline.Fed', 'Rehydration_baseline.Fed_w', 'Rehydration_baseline.Sucrose', 'Rehydration_baseline.Sucrose_w'],
    'colors':['black','grey', 'red', 'yellow'],
    'time_slice':(0,60),
    **kws0
}

kws3 = {
    'parent_dir': 'Refeeding_yeastpaste',
    'source_ids': ['Fe','Fe_yeast_paste', 'Pd', 'Pd_yeast_paste'],
    'ids': ['Fed','Fed -refed with yeast', 'Sucrose', 'Sucrose -refed with yeast'],
    'refIDs': ['Refeeding.Fed', 'Refeeding.Fed_yeast', 'Refeeding.Sucrose', 'Refeeding.Sucrose_yeast'],
    'colors':['black','grey', 'red', 'orange'],
    'time_slice':(0,60),
    **kws0
}

# G=reg.graphs

# entry_list=analysis_dict.general
# graph_entries = G.eval_graphgroups(graphgroups=['general'], **kws)
ggs=['endpoint', 'dsp', 'general']

# for kws in [kws3] :
for kws in [kws1, kws2, kws3] :
#     try:
    ds = import_datasets(**kws)
    gd = reg.graphs.eval_graphgroups(graphgroups=ggs, datasets=ds, save_to=f'{reg.DATA_DIR}/JovanicGroup/plots_submission/3/{kws["parent_dir"]}', subfolder=None)
    # except :
    #     pass
    # for d in ds :
    #     assert isinstance(d, LarvaDataset)