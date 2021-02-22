import sys

from lib.conf.larva_modes import *
from lib.conf.env_modes import *
from lib.conf.sim_modes import *

feed_scatter = {'fly_params': feeding_larva,
                'env_params': feed_scatter_exp_np,
                'collect_effectors': ['crawler', 'feeder', 'intermitter'],
                'draw_mode': draw_behavior
                }
feed_patchy = {'fly_params': feeding_odor_larva,
               'env_params': feed_patchy_exp_np,
               'collect_effectors': ['crawler', 'feeder', 'intermitter'],
               'draw_mode': draw_behavior
               }
feed_grid = {'fly_params': feeding_larva,
             'env_params': feed_grid_exp_np,
             'collect_effectors': ['crawler', 'feeder', 'intermitter'],
             'draw_mode': draw_behavior
             }
focus = {'fly_params': sample_exploring_larva,
         'env_params': focus_exp_np,
         'collect_effectors': ['turner','pose'],
         # 'modules': sole_turner,
         }
imitation = {'fly_params': imitation_larva,
             'env_params': imitation_exp_p,
             'traj_mode': no_traj,
             'draw_mode': draw_on_black
             }
reorientation = {'fly_params': sample_odor_larva,
                 'env_params': reorientation_exp_np,
                 'collect_effectors': ['turner', 'olfactor','pose'],
                 'modules': olfactor_turner,
                 'traj_mode': no_traj
                 }
growth = {'fly_params': growing_larva,
          'env_params': growth_exp_np,
          # 'collect_effectors': ['feeder'],
          'collect_effectors': ['feeder', 'deb'],
          'traj_mode': no_traj,
          'draw_mode': draw_behavior
          }
odor_pref = {'fly_params': sample_odor_larva_x2,
             'env_params': pref_exp_np,
             'collect_effectors': ['olfactor'],
             'end_pars': ['final_x'],
             'traj_mode': no_traj
             }
chemorbit = {'fly_params': sample_odor_larva,
             'env_params': chemorbit_exp_np,
             'collect_effectors': ['olfactor','pose'],
             'step_pars': ['dst_to_center', 'scaled_dst_to_center'],
             'end_pars': ['final_dst_to_center', 'final_scaled_dst_to_center',
                                               'max_dst_to_center', 'max_scaled_dst_to_center'],
             'draw_mode': draw_colors
             }
chemotax = {'fly_params': sample_odor_larva,
            'env_params': chemotax_exp_np,
            'collect_effectors': ['olfactor','pose'],
            'step_pars': ['dst_to_chemotax_odor', 'scaled_dst_to_chemotax_odor'],
            'end_pars': ['final_dst_to_chemotax_odor', 'final_scaled_dst_to_chemotax_odor'],
            'draw_mode': draw_colors
            }
dispersion = {'fly_params': sample_exploring_larva,
              'env_params': disp_exp_np,
'collect_effectors': ['pose'],
              'draw_mode': draw_colors
              }
dish = {'fly_params': sample_exploring_larva,
        'env_params': dish_exp_np,
'collect_effectors': ['pose'],
        'draw_mode': draw_colors,
        }
exp_types = {'feed_scatter': feed_scatter,
             'feed_patchy': feed_patchy,
             'feed_grid': feed_grid,
             'focus': focus,
             'imitation': imitation,
             'reorientation': reorientation,
             'growth': growth,
             'odor_pref': odor_pref,
             'chemorbit': chemorbit,
             'chemotax': chemotax,
             'dispersion': dispersion,
             'dish': dish,
             }
