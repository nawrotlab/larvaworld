from larvaworld.lib import reg, aux, plot
from larvaworld.lib.process.building import import_datasets


kws0 = {
    'datagroup_id': 'Jovanic lab',
    'enrich' : True,
    'merged' : False,
    'match_ids' : True,
}

kws1 = {
    'parent_dir': 'AttP240',
    'source_ids': ['Fed', 'Starved'],
    'ids': ['Fed','Starved'],
    'refIDs': ['AttP240.Fed', 'AttP240.Starved'],
    'colors':['black', 'red'],
    **kws0
}

ds = import_datasets(**kws1)
# ggs=['endpoint', 'dsp', 'general']
# gd = reg.graphs.eval_graphgroups(graphgroups=ggs, datasets=ds, save_to=f'{reg.DATA_DIR}/JovanicGroup/plots_submission/3/{kws1["parent_dir"]}', subfolder=None)
    # except :
    #     pass
    # for d in ds :
    #     assert isinstance(d, LarvaDataset)