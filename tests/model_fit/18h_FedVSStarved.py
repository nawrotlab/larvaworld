import pandas as pd


from lib.aux.combining import combine_pdfs
from lib.sim.eval.evaluation import EvalRun
from lib.sim.ga.functions import GA_optimization
from lib.sim.eval.model_fit import optimize_mID
from lib.registry.pars import preg
from lib.stor.managing import import_datasets

datagroup_id = 'Jovanic lab'
parent_dir = '18h'
idx = 5
g = preg.loadConf(id=datagroup_id, conftype='Group')
group_dir = f'{preg.path_dict["DATA"]}/{g.path}'
group_plotdir = f'{group_dir}/plots'
save_to = f'{group_plotdir}/{parent_dir}/trial{idx}'

# mID0 = 'RE_NEU_PHI_DEF_nav'
mIDs = ['Fed_18loco', 'Starved_18loco']

dIDs = ['Fed', 'Starved']
cols = ['green', 'red']
refIDs = [f'{parent_dir}.{dID}' for dID in dIDs]


def replay(idx=None):
    for i, k in enumerate(dIDs):
        refID = f'{parent_dir}.{k}'
        d = preg.loadRef(refID)
        if idx is None:
            d.visualize(save_to=f'{save_to}/replay')
        elif type(idx)==int :
            d.visualize_single(id=idx, close_view=True, fix_point=6, fix_segment=-1, save_to=f'{save_to}/replay',
                               draw_Nsegs=None)

def load_real(refIDs=refIDs,step=True, end=True, h5_ks=['contour', 'midline', 'epochs', 'base_spatial', 'angular', 'dspNtor']):
    ds = []
    for i, refID in enumerate(refIDs):
        # for i, k in enumerate(['AttP240', 'SS888Imp', 'SS888']):
        # try:
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


def adapt_models(mIDs=mIDs,refIDs=refIDs,mID0 = 'RE_NEU_PHI_DEF_nav',space_mkeys=['turner', 'interference'], init='model',
                 save_to=save_to):
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
    return entries





def eval_models(mIDs=mIDs,dataset_ids=dIDs,refIDs=refIDs,save_to=save_to,rerun=False):

    evruns = {}
    for refID in refIDs:
        id = f'{refID}_eval_models'
        evrun = EvalRun(refID=refID, id=id, modelIDs=mIDs, dataset_ids=dataset_ids, N=10, save_to=save_to,
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
    # m0 = preg.larva_conf_dict.loadConf('RE_NEU_PHI_DEF_nav')
    # for mID, refID in zip(mIDs, refIDs):
    #     m = preg.larva_conf_dict.loadConf(mID)
    #     d = preg.loadRef(refID)
    #     d.load(step=False)
    #     e, c = d.endpoint_data, d.config
    #     m.brain.intermitter_params=preg.larva_conf_dict.adapt_intermitter(e=e, c=c, mode=m0.brain.intermitter_params.mode,
    #                                                                  conf=m0.brain.intermitter_params)
    #
    #     preg.larva_conf_dict.saveConf(conf=m, mID=mID, verbose=0)
    #     # .config.bout_distros
    # m = preg.larva_conf_dict.loadConf(mIDs[1]).brain.intermitter_params
    # print(m.run_dur)
    # print(m.run_dst)
    # print(m.run_count)
    # print(m.run_mode)
    # print(m.stridechain_dist)
    # print(m.run_dist)


    # ds = import_datasets(datagroup_id = 'Jovanic lab', source_ids=dIDs, parent_dir=parent_dir,merged=False,colors=cols)
    # ds=load_real(step=True)
    # ds=load_real(step=True,h5_ks=[])
    # ds=load_real(step=False,h5_ks=['angular'])
    # ds=[]
    # for refID in refIDs:
    #     d = preg.loadRef(refID)
    #     d.load(step=True, end=True,h5_ks=['contour', 'midline', 'epochs', 'base_spatial', 'angular', 'dspNtor'])
    #     ds.append(d)

    # gd =preg.graph_dict.eval_graphgroups(graphgroups=['stride', 'track', 'endpoint', 'general'],
    #                                      datasets=ds, save_to=f'{save_to}/real')


    # replay(2)
    # adapt_models()
    mt=preg.graph_dict.model_tables(mIDs=mIDs,dIDs=dIDs,save_to = f'{save_to}/GA/tables')
    # evruns = eval_models()
