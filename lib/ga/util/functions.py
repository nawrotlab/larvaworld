import lib.aux.dictsNlists as dNl
# from lib.ga.util.genome import Genome

from lib.registry.pars import preg

def arrange_fitness(fitness_func, fitness_target_refID, fitness_target_kws,dt, source_xy=None):
    cycle_ks, eval_kNps = None, None
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
            eval_kNps={sh: p for p, sh in zip(eval_pars, shs)}
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

            fitness_target_kws['cycle_curve_keys'] = dic
            fitness_target_kws['pooled_cycle_curves'] = {sh: curves[sh] for sh in shorts}
            robot_dict.cycle_curves = {sh: [] for sh in shorts}

        fitness_target = d
    else:
        fitness_target = None
    if 'source_xy' in fitness_target_kws.keys():
        fitness_target_kws['source_xy'] = source_xy
    robot_dict.step=None
    ks = dNl.unique_list(ks)

    def robot_func(ss) :
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
                         'keys' : ks, 'robot_func' : robot_func,
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


fitness_funcs = dNl.NestDict({
    'interference': interference_evaluation,
    'distro_KS': distro_KS_evaluation,
    'distro_KS_interference': distro_KS_interference_evaluation,
    'dst2source': dst2source_evaluation,
    'cum_dst': cum_dst,
})
