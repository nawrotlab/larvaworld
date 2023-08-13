import warnings
import numpy as np


warnings.simplefilter(action='ignore', category=FutureWarning)
from larvaworld.lib import reg, aux





def ga_conf(name, env,mkeys, scene='no_boxes', refID=None, fit_kws={},
            cycle_curve_metrics=[],eval_metrics=aux.AttrDict(), dt=0.1, dur=3, N=30, Nel=3, m0='phasic_explorer',
            m1=None, fitID=None, init='random', excludeID=None):
    from larvaworld.lib.reg import gen


    conf = gen.Ga(ga_select_kws = gen.GAselector(Nagents=N, Nelits=Nel, base_model=m0, bestConfID=m1,
                                             init_mode=init,
                                             space_mkeys=mkeys),
                  ga_eval_kws = gen.GAevaluation(
                      fit_kws=fit_kws,
                      cycle_curve_metrics=cycle_curve_metrics,
                      eval_metrics=eval_metrics,
                                            fitness_func_name=fitID,
                                            exclude_func_name=excludeID,
                                                 refID=refID,
                                            ),
                  env_params=env,
                  experiment=name,
                  scene=scene,

                  duration=dur, dt=dt).nestedConf
    return {name: conf}


@reg.funcs.stored_conf("Ga")
def Ga_dict() :
    d = aux.AttrDict({
    **ga_conf('interference', dt=1 / 16, dur=3, refID='exploration.40controls', m0='loco_default',
              m1='NEU_PHI',
              cycle_curve_metrics=['fov', 'foa', 'rov'],
              mkeys=['interference', 'turner'],
              Nel=2, N=6, env='arena_200mm'),
    **ga_conf('exploration', dur=0.5, dt=1 / 16, refID='exploration.40controls', m0='loco_default',
              m1='NEU_PHI',
              eval_metrics=aux.AttrDict({'angular kinematics': ['run_fov_mu', 'pau_fov_mu', 'b', 'fov', 'foa'],
                             'spatial displacement': ['v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a'],
                             'temporal dynamics': ['fsv', 'ffov', 'run_tr', 'pau_tr']}),
              mkeys=['interference', 'turner'],
              # excludeID='bend_errors',
              Nel=2, N=10, env='arena_200mm'),
    **ga_conf('realism', dur=1, dt=1 / 16, refID='exploration.40controls', m0='loco_default', m1='PHIonSIN',
              eval_metrics=aux.AttrDict({'angular kinematics': ['run_fov_mu', 'pau_fov_mu','b', 'fov', 'foa'],
                            'spatial displacement': ['v', 'a'],
                            'temporal dynamics': ['fsv', 'ffov', 'run_tr', 'pau_tr']}),
              cycle_curve_metrics=['sv', 'fov', 'foa', 'b'],
              # excludeID='bend_errors',
              mkeys=['interference', 'turner'],
              Nel=3, N=10, env='arena_200mm'),
    **ga_conf('chemorbit', dur=1, m0='RE_NEU_PHI_DEF_nav', m1='RE_NEU_PHI_DEF_nav2',
              mkeys=['olfactor'], fitID='dst2source', fit_kws={'source_xy': None},
              Nel=5, N=50, env='mid_odor_gaussian_square'),
    **ga_conf('obstacle_avoidance', dur=0.5, m0='obstacle_avoider', m1='obstacle_avoider2',
              mkeys=['sensorimotor'], fitID='cum_dst',
              Nel=2, N=15, env='dish_40mm',
              scene='obstacle_avoidance_700')
    })
    return d

