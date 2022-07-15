import pandas as pd

from lib.aux.combining import combine_pdfs
from lib.sim.eval.evaluation import EvalRun
from lib.sim.ga.functions import GA_optimization
from lib.sim.eval.model_fit import optimize_mID
from lib.registry.pars import preg
from lib.stor.managing import import_Jovanic_datasets

parent_dir = '18h'
idx = 4
save_to = f'/home/panos/larvaworld_new/larvaworld/data/JovanicGroup/plots/{parent_dir}/trial{idx}'
mID0 = 'RE_NEU_PHI_DEF_nav'
mIDs = ['Fed_18loco', 'Starved_18loco']
cols = ['green','red']
dIDs = ['Fed', 'Starved']
refIDs = [f'{parent_dir}.{dID}' for dID in dIDs]

def import_data() :
    ds = import_Jovanic_datasets(parent_dir=parent_dir, source_ids=dIDs, enrich=True)
    return ds

def replay(idx=None):
    for i, k in enumerate(dIDs):
        # for i, k in enumerate(['AttP240', 'SS888Imp', 'SS888']):
        # try:
        refID = f'{parent_dir}.{k}'
        d = preg.loadRef(refID)
        if idx is None:
            d.visualize(save_to=f'{save_to}/replay')
        elif type(idx)==int :
            d.visualize_single(id=idx, close_view=True, fix_point=6, fix_segment=-1, save_to=f'{save_to}/replay',
                               draw_Nsegs=None)

def load_real(step=True, end=True, h5_ks=['contour', 'midline', 'epochs', 'base_spatial', 'angular', 'dspNtor']):
    # parent_dir = 'SS888_0_60'
    # parent_dir='18h'
    # parent_dir='AttP240'

    # ds = get_datasets(datagroup_id, names, last_common='processed', folders=None, suffixes=None,
    #                  mode='load', load_data=True, ids=None, **kwargs)
    # ds = import_Jovanic_datasets(parent_dir=parent_dir,source_ids=['AttP240', 'SS888Imp', 'SS888'], enrich=True)
    # print(ds)

    # step = True
    # end = True
    # cols = ['green', 'red']
    ds = []
    # dfs={}
    # save_to = f'/home/panos/larvaworld_new/larvaworld/data/JovanicGroup/plots/{parent_dir}/'
    # k0='AttP2'
    for i, k in enumerate(dIDs):
        # for i, k in enumerate(['AttP240', 'SS888Imp', 'SS888']):
        # try:
        refID = f'{parent_dir}.{k}'
        d = preg.loadRef(refID)
        # print(d.load_h5_kdic)
        d.load(step=step, end=end,h5_ks=h5_ks)
        # xx = d.read(key=f'distro.velocity', file='aux_h5').dropna().values
        # xx = d.get_par('velocity', key='distro').dropna().values
        # df = pd.DataFrame(xx, columns=['velocity'])
        # df['DatasetID'] = d.id
        # ds.append(df)
        # print(d.load_h5_kdic)
        # store = pd.HDFStore(d.dir_dict.data_h5)
        # s=store['step']['x']
        # store.close()
        # raise

        # s, e, c = d.step_data, d.endpoint_data, d.config
        # fft_freqs(s, e, c)

        # e,c=d.endpoint_data, d.config
        # c.color=cols[i]
        # d.color=cols[i]
        # e['stride_scaled_dst_mean']=e['scaled_stride_dst_mean']
        # e['stride_scaled_dst_std'] =e['scaled_stride_dst_std']
        # # dic = d.load_chunk_dicts()
        # # bbs=dic['Larva_101'].run*c.dt
        # # print()
        # # b0s, b1s = bbs[:, 0], bbs[:, 1]
        # # j=0
        # # lines = [[(b0, j + 1), (b1, j + 1)] for b0, b1 in zip(b0s, b1s)]
        # # raise
        # # store = pd.HDFStore(c.aux_dir)
        # # ddd=pd.read_hdf(d.dir_dict['aux_h5'],'trajectories')
        # # dddd=pd.read_hdf(d.dir_dict['aux_h5'],p.d)
        # # # df = d.read(key='pathlength', file='aux_h5')
        # # print(store.keys())
        # # store.close()
        # # raise
        # # vs=pd.read_hdf(d.dir_dict['data_h5'], 'step')[par]
        # # print(vs)
        # #print(d.id, c.color)
        # #print(e[preg.getPar( 'tor20_std')])
        # print(d.id)
        # print(d.existing('step'))
        # # print(e[(getPar('sstr_d_mu'))])
        # # e['length_in_mm']=e['length']*1000
        # # e['velocity_in_mm_mean']=e['velocity_mean']*1000
        # # print(d.id,d.color )
        # c.color=cols[i]
        # #comp_dispersion(s=None,e=e,c=c,dsp_starts=[0], dsp_stops=[60], store=True)
        # #comp_straightness_index(s=None,e=e,c=c,tor_durs=[20], store=True)
        # d.annotate(store=True)
        # d.save(step=step,h5_ks=h5_ks, add_reference=True)
        # print(d.load_h5_kdic)
        # d.save_config(add_reference=True)
        # raise
        ds.append(d)
    return ds


def adapt_models(space_mkeys=['turner', 'interference'], init='model'):
    kws = {'fitness_target_kws': {'eval_metrics': {
        'angular kinematics': ['run_fov_mu', 'pau_fov_mu', 'b', 'fov', 'foa'],
        'spatial displacement': ['v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a',
                                 'dsp_0_40_max', 'dsp_0_60_max'],
        'temporal dynamics': ['fsv', 'ffov', 'run_tr', 'pau_tr'],
        # 'stride cycle': ['str_d_mu', 'str_d_std', 'str_sv_mu', 'str_fov_mu', 'str_fov_std', 'str_N'],
        # 'epochs': ['run_t', 'pau_t'],
        # 'tortuosity': ['tor5', 'tor20']
    },
        'cycle_curves': ['fov', 'foa']}}
    entries = {}
    M = preg.larva_conf_dict
    for mID, refID in zip(mIDs, refIDs):
        entry = M.adapt_mID(refID=refID, mID0=mID0, mID=mID, space_mkeys=space_mkeys,
                            init=init, show_screen=True,
                            save_to=f'{save_to}/GA/{mID}', Nagents=50, Nelits=5, Ngenerations=10, dur=0.4,
                            fit_dict=GA_optimization(fitness_target_refID=refID, **kws))
        entries.update(entry)


def plot_model_confs():
    table_dir = f'{save_to}/GA/tables'
    G = preg.graph_dict.dict
    M = preg.larva_conf_dict
    G['mpl'](data=M.diff_df(mIDs=mIDs), font_size=18, save_to=table_dir, name='18h')

    for mID in mIDs:
        try:
            _ = G['model table'](mID, save_to=table_dir)
        except:
            print('TABLE FAIL', mID)
    combine_pdfs(file_dir=table_dir, save_as="___ALL_MODEL_CONFIGURATIONS___.pdf", deep=False)


def plot_datasets(ds, graphgroup='general',subfolder='real'):
    G = preg.graph_dict
    kws = {
        'save_to': f'{save_to}/{subfolder}',
        'show': False,
        'datasets': ds,
        'subfolder':None
    }
    entry_list = G.groups[graphgroup]
    graph_entries = G.eval(entry_list, **kws)
    return graph_entries


def eval_models(rerun=False):
    dataset_ids = ['Fed', 'Starved']

    eval_dir = f'{save_to}'
    evruns = {}
    for refID in refIDs:
        id = f'{refID}_eval_models'
        evrun = EvalRun(refID=refID, id=id, modelIDs=mIDs, dataset_ids=dataset_ids, N=10, save_to=eval_dir,
                        bout_annotation=True, enrichment=True, show=False, offline=False)
        if len(evrun.datasets) == 0 or rerun:
            evrun.run(video=False)
            evrun.eval()
        evruns[refID] = evrun
    for refID, evrun in evruns.items():
        evrun.plot_models()
        evrun.plot_results()
    return evruns


if __name__ == '__main__':
    # datagroup_id = 'Jovanic lab'
    # g = preg.loadConf(id=datagroup_id, conftype='Group')
    # en=g.enrichment
    # print(en.metric_definition.keys())
    # for k, v in en.items():
    #
    #     print(k)
    #     print(v)
    # ds = import_data()
    # ds=load_real(step=True)
    # ds=load_real(step=True,h5_ks=[])
    ds=load_real(step=False,h5_ks=['angular'])
    # gd1=plot_datasets(ds,graphgroup='stride')
    # gd1=plot_datasets(ds,graphgroup='track')
    # gd2=plot_datasets(ds,graphgroup='endpoint')
    gd3=plot_datasets(ds,graphgroup='general')
    # df0 = pd.concat(ds)
    # print(df0['velocity'].values.shape)
    # replay(2)
    # adapt_models()
    # plot_model_confs()
    # evruns = eval_models()
