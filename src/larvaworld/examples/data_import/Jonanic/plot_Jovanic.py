from larvaworld.lib import reg
import pandas as pd
# D=preg.dict
# p=D['cum_sd']

# par=p.d
# print(p.func)
# raise
# parent_dir='AttP2'
# parent_dir='5h'
# parent_dir='SS888'

parent_dir='SS888_0_60'
# parent_dir='18h'
# parent_dir='AttP240'


# ds = get_datasets(datagroup_id, names, last_common='processed', folders=None, suffixes=None,
#                  mode='load', load_data=True, ids=None, **kwargs)
# ds = import_Jovanic_datasets(parent_dir=parent_dir,source_ids=['AttP240', 'SS888Imp', 'SS888'], enrich=True)
# print(ds)


step = True
end = True
cols = ['green', 'blue', 'red']
ds = []

save_to = f'/home/panos/larvaworld_new/larvaworld/data/JovanicGroup/plots/{parent_dir}/'
# k0='AttP2'
for i, k in enumerate(['AttP240', 'SS888Imp', 'SS888']):
    # try:
    refID = f'{parent_dir}.{k}'
    d = reg.loadRef(refID)
    d.load(step=step,end=end, contour=False, midline=False)
    s,e,c=d.step_data, d.endpoint_data, d.config
    # fft_freqs(s, e, c)
    #e,c=d.endpoint_data, d.config
    # # dic = d.load_chunk_dicts()
    # # bbs=dic['Larva_101'].exec*c.dt
    # # print()
    # # b0s, b1s = bbs[:, 0], bbs[:, 1]
    # # j=0
    # # lines = [[(b0, j + 1), (b1, j + 1)] for b0, b1 in zip(b0s, b1s)]
    # # raise
    # # ddd=pd.read_hdf(d.dir_dict['aux_h5'],'trajectories')
    # # dddd=pd.read_hdf(d.dir_dict['aux_h5'],p.d)
    # # # df = d.read(key='pathlength', file='aux_h5')
    # # print(store.keys())
    # # store.close()
    # # raise
    # # print(vs)
    # #print(d.id, c.color)
    # #print(e[preg.getPar( 'tor20_std')])
    # #print(d.existing('end'))
    # # print(e[(getPar('sstr_d_mu'))])
    # # e['length_in_mm']=e['length']*1000
    # # e['velocity_in_mm_mean']=e['velocity_mean']*1000
    # # print(d.id,d.color )
    # c.color=cols[i]
    # c.color=cols[i]
    # #comp_dispersion(s=None,e=e,c=c,dsp_starts=[0], dsp_stops=[60], store=True)
    # #comp_straightness_index(s=None,e=e,c=c,tor_durs=[20], store=True)
    # d.save(step=step, contour=False, midline=False, add_reference=True)
    # d.save_config(add_reference=True)
    ds.append(d)
# raise
kws = {
    'save_to': save_to,
    'show': False,
    'datasets': ds,
    # 'subfolder':None
}


# from lib.conf.stored.analysis_conf import analysis_dict
G=reg.graphs

# entry_list=analysis_dict.general
graph_entries = G.eval_graphgroups(graphgroups=['general'], **kws)

