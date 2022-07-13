import lib.aux.dictsNlists as dNl
# from lib.ga.util.genome import Genome

from lib.registry.pars import preg


def GA_optimization(fitness_target_refID, fitness_target_kws):
    if fitness_target_refID is not None:
        ks=[]
        coef_dict={'KS':10, 'RSS':1}
        d = preg.loadRef(fitness_target_refID)
        func_dict = dNl.NestDict()
        for k, vs in fitness_target_kws.items():
            if k == 'eval_metrics':
                from lib.eval.eval_aux import RSS, arrange_evaluation
                evaluation, target_data = arrange_evaluation(d, evaluation_metrics=vs)
                s_shorts = dNl.flatten_list(evaluation['step']['shorts'].values.tolist())
                s_pars = dNl.flatten_list(evaluation['step']['pars'].values.tolist())
                s_symbols = dNl.NestDict(dict(zip(s_pars, s_shorts)))
                ks+=s_shorts
                def func(ss):
                    return dNl.NestDict({'KS': {sym: ks_2samp(target_data.step[p].values, ss[p].dropna().values)[0] for p, sym in
                                s_symbols.items()}})

                func_dict[k] = func

            if k == 'cycle_curves':
                cycle_mode_dict = dNl.NestDict({v: 'abs' if v == 'sv' else 'norm' for v in vs})
                all_ref_curves = d.config.pooled_cycle_curves
                ref_curves = dNl.NestDict(
                    ({sh: np.array(all_ref_curves[sh][mode]) for sh, mode in cycle_mode_dict.items()}))
                ks += vs
                def func(ss):
                    from lib.process.aux import cycle_curve_dict
                    c0 = cycle_curve_dict(s=ss, dt=d.config.dt, shs=vs)
                    eval_curves = dNl.NestDict(({sh: c0[sh][mode] for sh, mode in cycle_mode_dict.items()}))
                    return dNl.NestDict({'RSS' : {sh: RSS(ref_curve, eval_curves[sh]) for sh, ref_curve in ref_curves.items()}})

                func_dict[k] = func
                ks=dNl.unique_list(ks)
        # def get_fitness_dicts(ss) :
        #     return dNl.NestDict({kfunc(ss) for k, kfunc in func_dict.items()})
        def func(s,gd):
            # print(len(gd), len(s.index.unique('AgentID').values))
            for i, g in gd.items():
                ss = s.xs(i, level='AgentID')
                for k, kfunc in func_dict.items():
                    g.fitness_dict.update(kfunc(ss))
                fitness_means={k:-np.mean(list(dic.values())) for k, dic in g.fitness_dict.items()}
                if len(fitness_means)==1:
                    g.fitness=list(fitness_means.values())[0]
                else:
                    g.fitness=np.sum([coef_dict[k]*mean for k,mean in fitness_means.items()])

        return dNl.NestDict({'func': func, 'keys': ks})
    else:
        return None

#
#
# def arrange_fitness2(refID, evaluation_metrics):
#     from lib.eval.eval_aux import RSS, arrange_evaluation
#     # if refID is not None:
#     d = preg.loadRef(refID)
#     evaluation, target_data = arrange_evaluation(d, evaluation_metrics=evaluation_metrics)
#     s_shorts = dNl.flatten_list(evaluation['step']['shorts'].values.tolist())
#     symbols = dNl.flatten_list(evaluation['step']['symbols'].values.tolist())
#     s_pars = dNl.flatten_list(evaluation['step']['pars'].values.tolist())
#     s_symbols = dNl.NestDict(dict(zip(s_pars, s_shorts)))
#
#     def func(s, gd):
#         T = target_data.step
#         for i, g in gd.items():
#             ss = s.xs(i, level='AgentID')
#             g.fitness_dict = dNl.NestDict(
#                 {'KS': {sym: ks_2samp(T[p].values, ss[p].dropna().values) for p, sym in s_symbols.items()}})
#             g.fitness = -np.mean(list(g.fitness_dict.KS.values()))
#
#     def func2(s):
#         from lib.eval.eval_aux import RSS, eval_distro_fast
#
#         ks_dic = {id: {'KS':
#                            eval_distro_fast(s.xs(id, level='AgentID'), target_data.step, s_symbols, mode='pooled',
#                                             min_size=10)
#                        } for id in s.index.unique('AgentID').values}
#
#         ks_mu = {id: -np.mean(list(ks_dic[id]['KS'].values())) for id in ks_dic.keys()}
#
#         return ks_mu, ks_dic
#
#     return dNl.NestDict({'func': func, 'keys': s_shorts})


def arrange_fitness(fitness_func, fitness_target_refID, fitness_target_kws, dt, source_xy=None):
    cycle_ks, eval_kNps, cycle_mode_ks = None, None, None
    ks = []
    robot_dict = dNl.NestDict()
    if fitness_target_refID is not None:
        d = preg.loadRef(fitness_target_refID)
        if 'eval_shorts' in fitness_target_kws.keys():
            shs = fitness_target_kws['eval_shorts']

            eval_pars, eval_lims, eval_labels = preg.getPar(shs, to_return=['d', 'lim', 'lab'])
            fitness_target_kws['eval'] = {sh: d.get_par(p, key='distro').dropna().values for p, sh in
                                          zip(eval_pars, shs)}
            ks += shs
            eval_kNps = {sh: p for p, sh in zip(eval_pars, shs)}
            robot_dict.eval = {sh: [] for p, sh in zip(eval_pars, shs)}
            fitness_target_kws['eval_labels'] = eval_labels
        if 'pooled_cycle_curves' in fitness_target_kws.keys():
            curves = d.config.pooled_cycle_curves
            shorts = fitness_target_kws['pooled_cycle_curves']
            cycle_ks = shorts
            ks += shorts
            dic = {}
            for sh in shorts:
                dic[sh] = 'abs' if sh == 'sv' else 'norm'
            cycle_mode_ks = dic
            fitness_target_kws['cycle_curve_keys'] = dic
            fitness_target_kws['pooled_cycle_curves'] = {sh: curves[sh] for sh in shorts}
            robot_dict.cycle_curves = {sh: [] for sh in shorts}

        fitness_target = d
    else:
        fitness_target = None
    if 'source_xy' in fitness_target_kws.keys():
        fitness_target_kws['source_xy'] = source_xy
    robot_dict.step = None
    ks = dNl.unique_list(ks)

    def pre_func(gdict):
        G = {}
        if eval_kNps:
            KS = {}
            for sh, p in eval_kNps.items():
                KS[sh] = {i: g['step'][p].dropna().values for i, g in gdict.items()}
            G['KS'] = KS
        if cycle_mode_ks:
            from lib.process.aux import cycle_curve_dict
            cycle_ks = list(cycle_mode_ks.keys())
            rss = {i: cycle_curve_dict(s=g['step'], dt=dt, shs=cycle_ks) for i, g in gdict.items()}
            RSS = {}
            for sh, mode in cycle_mode_ks.items():
                RSS[sh] = {i: cc[sh][mode] for i, cc in rss.items()}
            G['RSS'] = RSS
            return dNl.NestDict(G)

    def robot_func(ss):
        gdict = dNl.NestDict()
        gdict.step = ss
        if cycle_ks:
            from lib.process.aux import cycle_curve_dict
            gdict.cycle_curves = cycle_curve_dict(s=ss, dt=dt, shs=cycle_ks)
        if eval_kNps:
            gdict.eval = {sh: ss[p].dropna().values for sh, p in eval_kNps.items()}
        return gdict

    # dic0 = self.fit_dict.robot_dict
    # cycle_ks, eval_ks = None, None
    # ks = []
    # if 'eval' in robot_dict.keys():
    #     eval_ks = fitness_target_kws['eval_shorts']
    #     ks += eval_ks
    # if 'cycle_curves' in robot_dict.keys():
    #     cycle_ks = list(fitness_target_kws['pooled_cycle_curves'].keys())
    #     ks += cycle_ks
    # ks = dNl.unique_list(ks)
    return dNl.NestDict({'func': fitness_func, 'target_refID': fitness_target_refID,
                         'keys': ks, 'robot_func': robot_func, 'pre_func': pre_func,
                         # 'keys' : {'eval' : eval_kNps, 'cycle':cycle_ks, 'all':ks},
                         'target_kws': fitness_target_kws, 'target': fitness_target, 'robot_dict': robot_dict})


import numpy as np
from scipy.stats import ks_2samp

from lib.registry.pars import preg
import lib.aux.dictsNlists as dNl
from lib.aux.xy_aux import eudi5x
# from lib.ga.robot.larva_robot import LarvaRobot, ObstacleLarvaRobot
from lib.eval.eval_aux import RSS


def interference_evaluation(gdict, pooled_cycle_curves, cycle_curve_keys, **kwargs):
    d1, d2 = gdict['cycle_curves'], pooled_cycle_curves
    RSS_dic = {sh: RSS(d1[sh][mode], np.array(d2[sh][mode])) for sh, mode in cycle_curve_keys.items()}
    return -np.mean(list(RSS_dic.values())), dNl.NestDict({'RSS': RSS_dic})



def distro_KS_evaluation(gdict, eval_shorts, eval_labels, eval, **kwargs):
    ks_dic = {s: ks_2samp(eval[s], gdict['eval'][s])[0] for s, l in zip(eval_shorts, eval_labels)}
    return -np.mean(list(ks_dic.values())), dNl.NestDict({'KS': ks_dic})


def distro_KS_interference_evaluation(gdict, eval_shorts, eval_labels, eval, pooled_cycle_curves, cycle_curve_keys):
    r1, ks_dic = distro_KS_evaluation(gdict, eval_shorts, eval_labels, eval)
    r2, RSS_dic = interference_evaluation(gdict, pooled_cycle_curves, cycle_curve_keys)
    dic = dNl.NestDict({**ks_dic, **RSS_dic})
    if np.isinf(r1) or np.isinf(r2):
        return -np.inf, dic
    else:
        return r1 * 10 + r2, dic


def dst2source_evaluation(gdict, source_xy):
    traj = gdict['step'][['x', 'y']].values
    dst = np.sqrt(np.diff(traj[:, 0]) ** 2 + np.diff(traj[:, 1]) ** 2)
    cum_dst = np.sum(dst)
    for label, pos in source_xy.items():
        dst2source = eudi5x(traj, np.array(pos))
        break
    return -np.mean(dst2source) / cum_dst, {}


def cum_dst(robot):
    return robot.cum_dst / robot.real_length


def bend_error_exclusion(robot):
    if robot.body_bend_errors >= 5:
        return True
    elif robot.negative_speed_errors >= 5:
        return True
    else:
        return False


fitness_funcs = dNl.NestDict({
    'interference': interference_evaluation,
    'distro_KS': distro_KS_evaluation,
    'distro_KS_interference': distro_KS_interference_evaluation,
    'dst2source': dst2source_evaluation,
    'cum_dst': cum_dst,
})


def distro_eval_step(s, eval_shorts, eval_labels, eval, **kwargs):
    from lib.eval.eval_aux import RSS, eval_distro_fast

    ks_dic = {id: eval_distro_fast(s.xs(id, level='AgentID'), eval, eval_labels, mode='pooled', min_size=10) for id
              in s.index}

    ks_mu = {id: -np.mean(list(vs.values())) for id, vs in ks_dic.items()}

    return ks_mu, dNl.NestDict({'KS': ks_dic})


def interference_eval_step(s, pooled_cycle_curves, cycle_curve_keys, dt, **kwargs):
    from lib.process.aux import cycle_curve_dict
    cycle_ks = list(cycle_curve_keys.keys())
    rss = {id: cycle_curve_dict(s=s.xs(id, level='AgentID'), dt=dt, shs=cycle_ks) for id, in s.index}
    RSS_dic = {
        i: {sh: RSS(cc[sh][mode], np.array(pooled_cycle_curves[sh][mode])) for sh, mode in cycle_curve_keys.items()} for
        i, cc in rss.items()}

    rss_mu = {id: -np.mean(list(vs.values())) for id, vs in RSS_dic.items()}

    return rss_mu, dNl.NestDict({'RSS': RSS_dic})


def distro_interference_eval_step(s, eval_shorts, eval_labels, eval, pooled_cycle_curves, cycle_curve_keys):
    ks_mu, ks_dic = distro_KS_evaluation(s, eval_shorts, eval_labels, eval)
    rss_mu, RSS_dic = interference_evaluation(s, pooled_cycle_curves, cycle_curve_keys)
    dics = {**ks_dic, **RSS_dic}

    fit_dic = dNl.NestDict({id: {k: v[id] for k, v in dics.items()} for id in s.index})

    fit_mu = {id: ks_mu[id] * 10 + rss_mu[id] for id, in s.index}

    return fit_mu, fit_dic


fitness_step_funcs = dNl.NestDict({
    'interference': interference_eval_step,
    'distro_KS': distro_eval_step,
    'distro_KS_interference': distro_interference_eval_step,
    'dst2source': dst2source_evaluation,
    'cum_dst': cum_dst,
})

exclusion_funcs = dNl.NestDict({
    'bend_errors': bend_error_exclusion
})


def approximate_fit_dict(refID, space_mkeys):
    target_kws = dNl.NestDict()
    eval_shorts = []
    if 'turner' in space_mkeys:
        eval_shorts += ['b', 'fov', 'foa']
    if 'crawler' in space_mkeys:
        eval_shorts += ['sv', 'sa']
    if len(eval_shorts) > 0:
        target_kws.eval_shorts = eval_shorts
    if 'interference' in space_mkeys:
        target_kws.pooled_cycle_curves = ['fov', 'foa', 'b']
    if 'eval_shorts' in target_kws.keys():
        if 'pooled_cycle_curves' in target_kws.keys():
            fitness_func = fitness_funcs['distro_KS_interference']
        else:
            fitness_func = fitness_funcs['distro_KS']
    elif 'pooled_cycle_curves' in target_kws.keys():
        fitness_func = fitness_funcs['interference']
    else:
        raise
    from lib.registry.pars import preg
    d = preg.loadRef(refID)
    d.load(step=False)
    e, c = d.endpoint_data, d.config

    fit_dict = arrange_fitness(fitness_func=fitness_func, fitness_target_refID=refID,
                               fitness_target_kws=target_kws,
                               dt=c.dt)
    return fit_dict
