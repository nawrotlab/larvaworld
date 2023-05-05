import warnings
import numpy as np


warnings.simplefilter(action='ignore', category=FutureWarning)
from larvaworld.lib import reg, aux





def ga_conf(name, env_params,space_mkeys, scene='no_boxes', refID=None, fit_kws={}, dt=0.1, dur=3, N=30, Nel=3, m0='phasic_explorer',
            m1=None, fitID=None, init='random', excludeID=None):

    build_kws = {
        'ga_eval_kws': reg.get_null('ga_eval_kws', fitness_target_refID=refID,
                                            fitness_target_kws=fit_kws,
                                            fitness_func_name=fitID,
                                            exclude_func_name=excludeID,
                                            ),
        'ga_space_kws': reg.get_null('ga_space_kws', base_model=m0, bestConfID=m1,
                                             init_mode=init,
                                             space_mkeys=space_mkeys
                                             ),
        'ga_select_kws': reg.get_null('ga_select_kws', Nagents=N, Nelits=Nel),

    }
    kws = {'sim_params': reg.get_null('sim_params', duration=dur, dt=dt),
           'scene': scene,
           'experiment': name,
           'env_params': env_params,
           'ga_build_kws': reg.get_null('ga_build_kws', **build_kws),
           }
    conf = reg.get_null('Ga', **kws)
    return {name: conf}


@reg.funcs.stored_conf("Ga")
def Ga_dict() :
    d = aux.AttrDict({
    **ga_conf('interference', dt=1 / 16, dur=3, refID='exploration.150controls', m0='loco_default',
              m1='NEU_PHI',
              fit_kws={'cycle_curves': ['fov', 'rov', 'foa']},
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
              excludeID='bend_errors',
              Nel=2, N=10, env_params='arena_200mm'),
    **ga_conf('realism', dur=1, dt=1 / 16, refID='exploration.150controls', m0='loco_default', m1='PHIonSIN',
              fit_kws={'eval_metrics':
                           {'angular kinematics': ['run_fov_mu', 'pau_fov_mu','b', 'fov', 'foa'],
                            'spatial displacement': ['v', 'a'],
                            'temporal dynamics': ['fsv', 'ffov', 'run_tr', 'pau_tr']},
                       # fit_kws={'eval_shorts': ['b', 'fov', 'foa', 'rov', 'tur_t', 'tur_fou', 'pau_t', 'run_t', 'tor2', 'tor10'],
                       'cycle_curves': ['sv', 'fov', 'foa', 'b']},
              excludeID='bend_errors',
              space_mkeys=['interference', 'turner'],
              Nel=3, N=10, env_params='arena_200mm'),
    **ga_conf('chemorbit', dur=1, m0='RE_NEU_PHI_DEF_nav', m1='RE_NEU_PHI_DEF_nav2',
              space_mkeys=['olfactor'], fitID='dst2source', fit_kws={'source_xy': None},
              Nel=5, N=50, env_params='mid_odor_gaussian_square'),
    **ga_conf('obstacle_avoidance', dur=0.5, m0='obstacle_avoider', m1='obstacle_avoider2',
              space_mkeys=['sensorimotor'], fitID='cum_dst',
              Nel=2, N=15, env_params='dish_40mm',
              scene='obstacle_avoidance_700')
    })
    return d

