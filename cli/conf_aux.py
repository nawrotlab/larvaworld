

from lib import reg, aux




def update_exp_conf(exp, d=None, N=None, models=None, arena=None, conf_type='Exp', **kwargs):
    if conf_type == 'Batch':
        exp_conf = reg.loadConf(conftype=conf_type, id=exp)
        # batch_id = d['batch_setup']['batch_id']

        exp_conf.exp = update_exp_conf(exp_conf.exp, d, N, models, conf_type='Exp')
        # exp_conf.batch_id = batch_id
        exp_conf.batch_type = exp

        exp_conf.update(**kwargs)
        return exp_conf



    exp_conf = reg.expandConf(id=exp, conftype=conf_type)

    if arena is not None:
        exp_conf.env_params.arena = arena
    if d is None:
        d = {'sim_params': reg.get_null('sim_params')}

    exp_conf.sim_params = aux.AttrDict(d['sim_params'])
    if exp is not None and conf_type is not None:

        if exp_conf.sim_params.duration is None:
            if exp in reg.storedConf(conf_type):
                exp_conf.sim_params.duration =reg.loadConf(conf_type, exp).sim_params.duration
            else:
                exp_conf.sim_params.duration = 3.0

    if models is not None:
        if conf_type in ['Exp', 'Eval']:
            exp_conf = update_exp_models(exp_conf, models)
    if N is not None:
        if conf_type == 'Exp':
            for gID, gConf in exp_conf.larva_groups.items():
                gConf.distribution.N = N
    exp_conf.update(**kwargs)

    exp_conf.experiment=exp
    return exp_conf


def update_exp_models(exp_conf, mIDs=None, N=None):
    lgs = exp_conf.larva_groups
    if mIDs is not None:
        Nm = len(mIDs)

        confs=list(lgs.values())
        if len(lgs) != Nm:
            confs=[confs[0]]*Nm
            for conf,col in zip(confs,aux.N_colors(Nm)):
                conf.default_color = col
        lgs = aux.AttrDict({mID: {} for mID in mIDs})
        for mID, conf in zip(mIDs, confs):
            lgs[mID] = conf
            lgs[mID].model = reg.loadConf('Model', mID)
    if N is not None:
        for mID, conf in lgs.items():
            conf.distribution.N = N
    exp_conf.larva_groups = lgs
    return exp_conf
