
import numpy as np
from scipy.stats import ks_2samp

from lib.registry.pars import preg
import lib.aux.dictsNlists as dNl
from lib.aux.xy_aux import eudi5x
from lib.sim.eval.eval_aux import RSS


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
    if robot.body_bend_errors >= 20:
        return True
    # elif robot.negative_speed_errors >= 5:
    #     return True
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
    from lib.sim.eval.eval_aux import RSS, eval_distro_fast

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