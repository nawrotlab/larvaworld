from larvaworld.lib import reg


parent_dir='AttP240'
ds = [reg.loadRef(f'{parent_dir}.{k}', load=True,step=True,end=True) for k in ['Fed','Starved']]


kws = {
    'save_to': f'/home/panos/larvaworld_new/larvaworld/data/JovanicGroup/plots/{parent_dir}/',
    'show': False,
    'datasets': ds,
    'subfolder':None
}


ggs=['endpoint', 'dsp', 'general']
gd = reg.graphs.eval_graphgroups(graphgroups=ggs, **kws)

