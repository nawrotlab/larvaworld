from larvaworld.lib import reg, util

from larvaworld.lib.sim.model_evaluation import EvalRun

datagroup_id = 'Jovanic lab'
parent_dir = '18h'
idx = 5
g = reg.loadConf(id=datagroup_id, conftype='Group')
group_dir = f'{reg.DATA_DIR}/{g.path}'
group_plotdir = f'{group_dir}/plots'
save_to = f'{group_plotdir}/{parent_dir}/trial{idx}'

# mID0 = 'RE_NEU_PHI_DEF_nav'
mIDs = ['Fed_18loco', 'Starved_18loco']

dIDs = ['Fed', 'Starved']
cols = ['green', 'red']
refIDs = [f'{parent_dir}.{dID}' for dID in dIDs]






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
    for mID, refID in zip(mIDs, refIDs):
        entry = reg.Dic.MD.adapt_mID(refID=refID, mID0=mID0, mID=mID, space_mkeys=space_mkeys,
                            init=init, show_screen=True,
                            save_to=f'{save_to}/GA/{mID}', Nagents=50, Nelits=5, Ngenerations=10, dur=0.4,
                            fit_dict=util.GA_optimization(fitness_target_refID=refID, **kws))
        entries.update(entry)
    return entries





def eval_models(mIDs=mIDs,dataset_ids=dIDs,refIDs=refIDs,save_to=save_to,rerun=False):

    evruns = {}
    for refID in refIDs:
        id = f'{refID}_eval_models'
        evrun = EvalRun(refID=refID, id=id, modelIDs=mIDs, dataset_ids=dataset_ids, N=10, save_to=save_to,
                        enrichment=True, show=False, offline=False)
        if len(evrun.datasets) == 0 or rerun:
            evrun.simulate(video=False)
            # evrun.eval()
        evruns[refID] = evrun
    for refID, evrun in evruns.items():
        evrun.plot_models()
        evrun.plot_results()
    return evruns


if __name__ == '__main__':
    pass
    # ds = import_datasets(datagroup_id = 'Jovanic lab', source_ids=dIDs, parent_dir=parent_dir,merged=False,colors=cols)
    # ds=preg.conftype_dict.loadRefDs(refIDs,step=True, end=True, h5_ks=['contour', 'midline', 'epochs', 'base_spatial', 'angular', 'dspNtor'])


    # gd =preg.graph_dict.eval_graphgroups(graphgroups=['stride', 'track', 'endpoint', 'general'],
    #                                      datasets=ds, save_to=f'{save_to}/real')


    # replay(2)
    # adapt_models()
    # mt=preg.graph_dict.model_tables(mIDs=mIDs,dIDs=dIDs,save_to = f'{save_to}/GA/tables')
    # evruns = eval_models()
