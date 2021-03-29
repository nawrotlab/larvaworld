from lib.conf.larva_modes import *
from lib.conf.env_modes import *

dish = {'fly_params': exploring_larva,
        'env_params': dish_exp_np,
        'collections': ['pose'],
        }


feed_scatter = {'fly_params': feeding_larva,
                'env_params': feed_scatter_exp_np,
                'collections': ['crawler', 'feeder', 'intermitter'],
                }
feed_patchy = {'fly_params': feeding_odor_larva,
               'env_params': feed_patchy_exp_np,
               'collections': ['crawler', 'feeder', 'intermitter'],
               }
feed_grid = {'fly_params': feeding_larva,
             'env_params': feed_grid_exp_np,
             'collections': ['crawler', 'feeder', 'intermitter'],
             }
focus = {'fly_params': exploring_larva,
         'env_params': focus_exp_np,
         'collections': ['turner', 'pose'],
         }
imitation = {'fly_params': imitation_larva,
             'env_params': imitation_exp_p,
             'collections': ['midline', 'contour', 'pose'],
             }
reorientation = {'fly_params': odor_larva,
                 'env_params': reorientation_exp_np,
                 'collections': ['turner', 'olfactor', 'pose'],
                 }
growth = {'fly_params': growing_rover,
          'env_params': growth_exp_np,
          # 'collect_effectors': ['feeder'],
          'collections': ['feeder', 'deb'],
          # 'starvation_hours': [[0.2, 0.4]]
          }
growth_2x = {
    # 'fly_params': growing_sitter,
    'fly_params': [growing_rover, growing_sitter],
    'env_params': growth_exp_np,
    'collections': ['feeder', 'deb'],
    # 'age_in_hours' : 0
    # 'starvation_hours': [[24, 48]]
}
odor_pref = {'fly_params': odor_larva_x2,
             'env_params': pref_exp_np,
             'collections': ['olfactor'],
             }
chemorbit = {'fly_params': odor_larva,
             'env_params': chemorbit_exp_np,
             'collections': ['olfactor', 'pose', 'dst2center'],
             }
chemotax = {'fly_params': odor_larva,
            'env_params': chemotax_exp_np,
            'collections': ['olfactor', 'pose', 'chemotax_dst'],
            }
dispersion = {'fly_params': exploring_larva,
              'env_params': disp_exp_np,
              'collections': ['pose'],
              }

maze = {'fly_params': odor_larva,
        'env_params': maze_exp_np,
        'collections': ['olfactor', 'pose', 'dst2center'],
        }
exp_types = {
    'focus': focus,
    'dish': dish,
    'dispersion': dispersion,
    'chemorbit': chemorbit,
    'chemotax': chemotax,
    'odor_pref': odor_pref,
    'feed_grid': feed_grid,
    'feed_patchy': feed_patchy,
    'feed_scatter': feed_scatter,
    'growth': growth,
    'growth_2x': growth_2x,
    'maze': maze,
    'imitation': imitation,
    'reorientation': reorientation
}
