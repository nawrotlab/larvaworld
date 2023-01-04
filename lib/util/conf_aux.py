import lib.util.data_aux
import lib.reg

from lib import reg, aux


def adjust_sim(exp, conf_type, sim):
    if exp is not None and conf_type is not None:

        if sim.duration is None:
            ct = reg.conf.dict[conf_type]
            if exp in ct.ConfIDs:
                sim.duration =ct.loadConf(exp).sim_params.duration
            else:
                sim.duration = 3.0

        if sim.sim_ID is None:
            sim.sim_ID = f'{exp}_{lib.reg.next_idx(id=exp, conftype=conf_type)}'
        if sim.path is None:
            if conf_type == 'Exp':
                sim.path = f'single_runs/{exp}'
            elif conf_type == 'Ga':
                sim.path = f'ga_runs/{exp}'
            elif conf_type == 'Batch':
                sim.path = f'batch_runs/{exp}'
            elif conf_type == 'Eval':
                sim.path = f'eval_runs/{exp}'
        return sim


def update_exp_conf(exp, d=None, N=None, models=None, arena=None, conf_type='Exp', **kwargs):
    if conf_type == 'Batch':
        exp_conf = reg.loadConf(conftype=conf_type, id=exp)
        batch_id = d['batch_setup']['batch_id']
        if batch_id is None:
            idx = lib.reg.next_idx(id=exp, conftype='Batch')
            batch_id = f'{exp}_{idx}'

        exp_conf.exp = update_exp_conf(exp_conf.exp, d, N, models)
        exp_conf.batch_id = batch_id
        exp_conf.batch_type = exp

        exp_conf.update(**kwargs)
        return exp_conf



    try:
        exp_conf = reg.expandConf(id=exp, conftype=conf_type)
    except:
        raise

    if arena is not None:
        exp_conf.env_params.arena = arena
    if d is None:
        d = {'sim_params': reg.get_null('sim_params')}

    exp_conf.sim_params = adjust_sim(exp=exp, conf_type=conf_type, sim=aux.NestDict(d['sim_params']))

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
        lgs = aux.NestDict({mID: {} for mID in mIDs})
        for mID, conf in zip(mIDs, confs):
            lgs[mID] = conf
            lgs[mID].model = reg.conftree.Model[mID]
    if N is not None:
        for mID, conf in lgs.items():
            conf.distribution.N = N
    exp_conf.larva_groups = lgs
    return exp_conf
