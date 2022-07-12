import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from scipy.stats import ks_2samp

from lib.registry.pars import preg
import lib.aux.dictsNlists as dNl
from lib.aux.xy_aux import eudi5x
from lib.ga.robot.larva_robot import LarvaRobot, ObstacleLarvaRobot
from lib.eval.eval_aux import RSS


def bend_error_exclusion(robot):
    if robot.body_bend_errors >= 2:
        return True
    else:
        return False



def ga_conf(name, env_params,space_mkeys, scene='no_boxes', refID=None, fit_kws={}, dt=0.1, dur=3, N=30, Nel=3, m0='phasic_explorer',
            m1=None, sel={}, build={}, fitID=None, init='random', excl_func=None, robot_class=LarvaRobot, **kwargs):
    from lib.ga.util.functions import fitness_funcs

    build_kws = {
        'fitness_target_refID': refID,
        'fitness_target_kws': fit_kws,
        'base_model': m0,
        'bestConfID': m1,
        'exclude_func': excl_func,
        'init_mode': init,
        'robot_class': robot_class,
        'space_mkeys': space_mkeys,
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
              space_mkeys=['interference', 'turner'], fitID='interference',
              Nel=2, N=6, env_params='arena_200mm'),
    **ga_conf('exploration', dur=0.5, dt=1 / 16, refID='None.150controls', m0='loco_default',
              m1='NEU_PHI', fit_kws={'eval_shorts': ['b', 'bv', 'ba', 'tur_t', 'tur_fou', 'tor2', 'tor10']},
              space_mkeys=['interference', 'turner'], fitID='distro_KS', init='random',
              excl_func=bend_error_exclusion,
              Nel=2, N=10, env_params='arena_200mm'),
    **ga_conf('realism', dur=1, dt=1 / 16, refID='None.150controls', m0='PHIonSIN', m1='PHIonSIN',
              fit_kws={'eval_shorts': ['b', 'fov', 'foa'],
                       # fit_kws={'eval_shorts': ['b', 'fov', 'foa', 'rov', 'tur_t', 'tur_fou', 'pau_t', 'run_t', 'tor2', 'tor10'],
                       'pooled_cycle_curves': ['fov', 'foa', 'b']},
              excl_func=bend_error_exclusion,
              space_mkeys=['interference', 'turner'], fitID='distro_KS_interference',
              init='model',
              Nel=3, N=10, env_params='arena_200mm'),
    **ga_conf('chemorbit', dur=5, m0='navigator', m1='best_navigator',
              space_mkeys=['olfactor'], fitID='dst2source', fit_kws={'source_xy': None},
              Nel=5, N=50, env_params='mid_odor_gaussian_square'),
    **ga_conf('obstacle_avoidance', dur=0.5, m0='obstacle_avoider', m1='obstacle_avoider2',
              space_mkeys=['sensorimotor'], fitID='cum_dst', robot_class=ObstacleLarvaRobot,
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
