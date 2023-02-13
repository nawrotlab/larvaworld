import warnings
import numpy as np


warnings.simplefilter(action='ignore', category=FutureWarning)
from larvaworld.lib import reg, aux


def dst2source_evaluation(robot, source_xy):
    traj = np.array(robot.trajectory)
    dst = np.sqrt(np.diff(traj[:, 0]) ** 2 + np.diff(traj[:, 1]) ** 2)
    cum_dst = np.sum(dst)
    l=[]
    for label, pos in source_xy.items():
        l.append(aux.eudi5x(traj, pos))
    fitness= - np.mean(np.min(np.vstack(l),axis=0))/ cum_dst
    return fitness

def cum_dst(robot, **kwargs):
    return robot.cum_dst / robot.real_length


def bend_error_exclusion(robot):
    if robot.body_bend_errors >= 20:
        return True
    # elif robot.negative_speed_errors >= 5:
    #     return True
    else:
        return False


fitness_funcs = aux.AttrDict({
    'dst2source': dst2source_evaluation,
    'cum_dst': cum_dst,
})



exclusion_funcs = aux.AttrDict({
    'bend_errors': bend_error_exclusion
})



def ga_conf(name, env_params,space_mkeys, scene='no_boxes', refID=None, fit_kws={}, dt=0.1, dur=3, N=30, Nel=3, m0='phasic_explorer',
            m1=None, sel={}, build={}, fitID=None, init='random', excludeID=None, robot_class='LarvaRobot', **kwargs):

    build_kws = {
        'fitness_target_refID': refID,
        'fitness_target_kws': fit_kws,
        'base_model': m0,
        'bestConfID': m1,
        'init_mode': init,
        'robot_class': robot_class,
        'space_mkeys': space_mkeys,
        # 'space_dict': space_dict,
    }
    # print(dur,name)
    kws = {'sim_params': reg.get_null('sim_params', duration=dur, timestep=dt),
           'scene': scene,
           'experiment': name,
           'env_params': env_params,
           }

    if fitID is not None:
        build_kws['fitness_func'] = fitness_funcs[fitID]
    if excludeID is not None:
        build_kws['exclude_func'] = exclusion_funcs[excludeID]

    kws['ga_select_kws'] = reg.get_null('ga_select_kws', Nagents=N, Nelits=Nel, **sel)
    kws['ga_build_kws'] = reg.get_null('ga_build_kws', **build_kws, **build)
    kws.update(kwargs)

    conf = reg.get_null('Ga', **kws)
    return {name: conf}


@reg.funcs.stored_conf("Ga")
def Ga_dict() :
    d = aux.AttrDict({
    **ga_conf('interference', dt=1 / 16, dur=3, refID='exploration.150controls', m0='loco_default',
              m1='NEU_PHI',
              fit_kws={'cycle_curves': ['fov', 'rov', 'foa']},
              # init='model',
              space_mkeys=['interference', 'turner'],
              Nel=2, N=6, env_params='arena_200mm'),
    **ga_conf('exploration', dur=0.5, dt=1 / 16, refID='exploration.150controls', m0='loco_default',
              m1='NEU_PHI',
              fit_kws={'eval_metrics':
                           {'angular kinematics': ['run_fov_mu', 'pau_fov_mu', 'b', 'fov', 'foa'],
                            'spatial displacement': ['v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a'],
                            'temporal dynamics': ['fsv', 'ffov', 'run_tr', 'pau_tr']}
                       },
              # fit_kws={'eval_shorts': ['b', 'bv', 'ba', 'tur_t', 'tur_fou', 'tor2', 'tor10']},
              space_mkeys=['interference', 'turner'],
              # init='random',
              excludeID='bend_errors',
              Nel=2, N=10, env_params='arena_200mm'),
    **ga_conf('realism', dur=1, dt=1 / 16, refID='exploration.150controls', m0='loco_default', m1='PHIonSIN',
              fit_kws={'eval_shorts': ['b', 'fov', 'foa'],
                       # fit_kws={'eval_shorts': ['b', 'fov', 'foa', 'rov', 'tur_t', 'tur_fou', 'pau_t', 'run_t', 'tor2', 'tor10'],
                       'pooled_cycle_curves': ['fov', 'foa', 'b']},
              excludeID='bend_errors',
              space_mkeys=['interference', 'turner'],
              # init='model',
              Nel=3, N=10, env_params='arena_200mm'),
    **ga_conf('chemorbit', dur=1, m0='RE_NEU_PHI_DEF_nav', m1='RE_NEU_PHI_DEF_nav2',
              # init='random',
              space_mkeys=['olfactor'], fitID='dst2source', fit_kws={'source_xy': None},
              Nel=5, N=50, env_params='mid_odor_gaussian_square'),
    **ga_conf('obstacle_avoidance', dur=0.5, m0='obstacle_avoider', m1='obstacle_avoider2',
              space_mkeys=['sensorimotor'], fitID='cum_dst', robot_class='ObstacleLarvaRobot',
              Nel=2, N=15, env_params='dish_40mm',
              # init='default',
              scene='obstacle_avoidance_700')
    })
    return d

