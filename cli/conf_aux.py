

from lib import reg, aux




def update_exp_conf(exp, duration=None,store_data=True,Box2D=False, N=None, mIDs=None):
    conf = reg.expandConf(id=exp, conftype='Exp')
    conf.experiment = exp
    if duration is not None:
        conf.sim_params.duration =duration
    conf.sim_params.store_data = store_data
    conf.sim_params.Box2D = Box2D

    # conf.sim_params = aux.AttrDict(sim_params)
    #
    # if conf.sim_params.duration is None:
    #     conf.sim_params.duration =reg.loadConf(id=exp, conftype='Exp').sim_params.duration

    if mIDs is not None:
        Nm = len(mIDs)
        gConfs = list(conf.larva_groups.values())
        if len(conf.larva_groups) != Nm:
            gConfs = [gConfs[0]] * Nm
            for gConf, col in zip(gConfs, aux.N_colors(Nm)):
                gConf.default_color = col
        conf.larva_groups = aux.AttrDict({mID: {} for mID in mIDs})
        for mID, gConf in zip(mIDs, gConfs):
            conf.larva_groups[mID] = gConf
            conf.larva_groups[mID].model = reg.loadConf('Model', mID)

        # conf = update_exp_models(conf, models)
    if N is not None:
        for gID, gConf in conf.larva_groups.items():
            gConf.distribution.N = N

    return conf


# def update_exp_models(exp_conf, mIDs=None, N=None):
#     lgs = exp_conf.larva_groups
#     if mIDs is not None:
#         Nm = len(mIDs)
#
#         confs=list(lgs.values())
#         if len(lgs) != Nm:
#             confs=[confs[0]]*Nm
#             for conf,col in zip(confs,aux.N_colors(Nm)):
#                 conf.default_color = col
#         lgs = aux.AttrDict({mID: {} for mID in mIDs})
#         for mID, conf in zip(mIDs, confs):
#             lgs[mID] = conf
#             lgs[mID].model = reg.loadConf('Model', mID)
#     if N is not None:
#         for mID, conf in lgs.items():
#             conf.distribution.N = N
#     exp_conf.larva_groups = lgs
#     return exp_conf
