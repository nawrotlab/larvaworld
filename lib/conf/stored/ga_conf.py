import warnings

from lib.registry.pars import preg

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from scipy.stats import ks_2samp

import lib.aux.dictsNlists as dNl
from lib.aux.xy_aux import eudi5x
from lib.registry.ga_dict import ga_dict
from lib.ga.robot.larva_robot import LarvaRobot, ObstacleLarvaRobot
from lib.eval.eval_aux import RSS

ga_spaces = dNl.NestDict({
    'interference': ga_dict(name='interference', suf='brain.interference_params.',
                            excluded=['feeder_phi_range', 'crawler_phi_range', 'mode', 'suppression_mode']),
    'turner': ga_dict(name='turner', suf='brain.turner_params.', only=['base_activation']),
    'physics': ga_dict(name='physics', suf='physics.', only=['torque_coef']),
    'sensorimotor': ga_dict(name='obstacle_avoidance', suf='sensorimotor.', excluded=[]),
    'olfactor': {**ga_dict(name='olfactor', suf='brain.olfactor_params.', excluded=['input_noise']),
                 'brain.olfactor_params.odor_dict.Odor.mean': {'initial_value': 0.0, 'tooltip': 'Odor gain',
                                                               'dtype': float,
                                                               'name': 'Gain', 'min': -100.0, 'max': 1000.0}}
})


def interference_evaluation(gdict, pooled_cycle_curves, cycle_curve_keys, **kwargs):
    d1, d2 = gdict['cycle_curves'], pooled_cycle_curves
    RSS_dic = {sh: RSS(d1[sh][mode], np.array(d2[sh][mode])) for sh, mode in cycle_curve_keys.items()}
    return -np.mean(list(RSS_dic.values())), RSS_dic


def distro_KS_evaluation(gdict, eval_shorts, eval_labels, eval, **kwargs):
    ks_dic = {s: ks_2samp(eval[s], gdict['eval'][s])[0] for s, l in zip(eval_shorts, eval_labels)}
    return -np.mean(list(ks_dic.values())), ks_dic


def distro_KS_interference_evaluation(gdict, eval_shorts, eval_labels, eval, pooled_cycle_curves, cycle_curve_keys):
    r1, ks_dic = distro_KS_evaluation(gdict, eval_shorts, eval_labels, eval)
    r2, RSS_dic = interference_evaluation(gdict, pooled_cycle_curves, cycle_curve_keys)
    dic = dNl.NestDict({'KS': ks_dic, 'RSS': RSS_dic})
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


def bend_error_exclusion(robot):
    if robot.body_bend_errors >= 2:
        return True
    else:
        return False


def space_dic(mkeys,mID0):
    M = preg.larva_conf_dict2
    m = M.loadConf(mID0)
    mF = dNl.flatten_dict(m)
    dic = {}
    for mkey in mkeys:
        d0 = M.dict.model.init[mkey]
        d00 = M.dict.model.m[mkey]
        mod_v = mF[f'{d0.pref}mode']
        var_ks = d0.mode[mod_v].variable
        for k in var_ks:
            k0 = f'{d0.pref}{k}'
            dic[k0] = d00.mode[mod_v].args[k]
            dic[k0].v=mF[k0]

    return dNl.NestDict(dic)

def ga_conf(name, env_params,spaceIDs, scene='no_boxes', refID=None, fit_kws={}, dt=0.1, dur=3, N=30, Nel=3, m0='phasic_explorer',
            m1=None, sel={}, build={}, fitID=None, init='random', excl_func=None, robot_class=LarvaRobot, **kwargs):
    M=preg.larva_conf_dict2
    # m=M.loadConf(m0)
    # mF=dNl.flatten_dict(m)
    # dic={}
    # for mkey in spaceIDs:
    #     d0=M.dict.model.init[mkey]
    #     d00=M.dict.model.m[mkey]
    #     # pref=d0.pref
    #     # mod_k=f'{d0.pref}mode'
    #     mod_v=mF[f'{d0.pref}mode']
    #     var_ks=d0.mode[mod_v].variable
    #     for k in var_ks :
    #         k0=f'{d0.pref}{k}'
    #         p=d00.mode[mod_v].args[k]
    #         dic[k0]=p
    #     #     kws={
    #     #     'initial_value': p['initial_value'],
    #     #     'tooltip': p['tooltip'],
    #     #     'dtype': p['dtype'],
    #     #     'name': k,
    #     # }
    #
    #     # var_fuls=[f'{d0.pref}{k}' for k in var_ks]
    # dic=dNl.NestDict(dic)

    # space_dict = {}
    # for spaceID in spaceIDs:
    #     space_dict.update(ga_spaces[spaceID])

    build_kws = {
        'fitness_target_refID': refID,
        'fitness_target_kws': fit_kws,
        'base_model': m0,
        'bestConfID': m1,
        'exclude_func': excl_func,
        'init_mode': init,
        'robot_class': robot_class,
        'space_dict': spaceIDs,
        # 'space_dict': space_dict,
    }

    kws = {'sim_params': preg.get_null('sim_params', duration=dur, timestep=dt),
           'scene': scene,
           'experiment': name,
           'env_params': env_params,
           }

    if fitID is not None:
        build_kws['fitness_func'] = fitness_funcs[fitID]

    kws['ga_select_kws'] = preg.get_null('ga_select_kws', Nagents=N, Nelits=Nel, **sel)
    kws['ga_build_kws'] = preg.get_null('ga_build_kws', **build_kws, **build)
    kws.update(kwargs)

    conf = preg.get_null('GAconf', **kws)
    return {name: conf}


ga_dic = dNl.NestDict({
    **ga_conf('interference', dt=1 / 16, dur=3, refID='None.150controls', m0='phasic_explorer',
              m1='NEU_PHI',
              fit_kws={'pooled_cycle_curves': ['fov', 'rov', 'foa']}, init='model',
              spaceIDs=['interference', 'turner'], fitID='interference',
              Nel=2, N=6, env_params='arena_200mm'),
    **ga_conf('exploration', dur=0.5, dt=1 / 16, refID='None.150controls', m0='loco_default',
              m1='NEU_PHI', fit_kws={'eval_shorts': ['b', 'bv', 'ba', 'tur_t', 'tur_fou', 'tor2', 'tor10']},
              spaceIDs=['interference', 'turner'], fitID='distro_KS', init='random',
              excl_func=bend_error_exclusion,
              Nel=2, N=10, env_params='arena_200mm'),
    **ga_conf('realism', dur=1, dt=1 / 16, refID='None.150controls', m0='phasic_explorer', m1='NEU_PHI',
              fit_kws={'eval_shorts': ['b', 'fov', 'foa', 'v', 'a', 'tor5'],
                       # fit_kws={'eval_shorts': ['b', 'fov', 'foa', 'rov', 'tur_t', 'tur_fou', 'pau_t', 'run_t', 'tor2', 'tor10'],
                       'pooled_cycle_curves': ['fov', 'foa', 'b']},
              excl_func=bend_error_exclusion,
              spaceIDs=['interference', 'turner'], fitID='distro_KS_interference',
              init='random',
              Nel=5, N=20, env_params='arena_200mm'),
    **ga_conf('chemorbit', dur=5, m0='navigator', m1='best_navigator',
              spaceIDs=['olfactor'], fitID='dst2source', fit_kws={'source_xy': None},
              Nel=5, N=50, env_params='mid_odor_gaussian_square'),
    **ga_conf('obstacle_avoidance', dur=0.5, m0='obstacle_avoider', m1='obstacle_avoider2',
              spaceIDs=['sensorimotor'], fitID='cum_dst', robot_class=ObstacleLarvaRobot,
              Nel=2, N=15, env_params='dish_40mm', init='default',
              scene='obstacle_avoidance_700')
})

if __name__ == '__main__':
    mkeys=['interference', 'turner']
    mID0='PHIonNEU'
    spdic=space_dic(mkeys,mID0)
    for k,p in spdic.items():
        print(k,p.v)
    # print(ga_dict(name='physics', suf='physics.', excluded=None, only=['torque_coef','ang_damping','body_spring_k']))
