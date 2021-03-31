from lib.conf.larva_modes import *
from lib.conf.env_modes import *

dish = {'fly_params': exploring_larva,
        'env_params': dish_env,
        'collections': ['pose'],
        }

feed_scatter = {'fly_params': feeding_larva,
                'env_params': uniform_food_env,
                'collections': ['crawler', 'feeder', 'intermitter'],
                }
feed_patchy = {'fly_params': feeding_odor_larva,
               'env_params': patchy_food_env,
               'collections': ['crawler', 'feeder', 'intermitter'],
               }
feed_grid = {'fly_params': feeding_larva,
             'env_params': food_grid_env,
             'collections': ['crawler', 'feeder', 'intermitter'],
             }
focus = {'fly_params': exploring_larva,
         'env_params': focus_env,
         'collections': ['turner', 'pose'],
         }
imitation = {'fly_params': imitation_larva,
             'env_params': imitation_env_p,
             'collections': ['midline', 'contour', 'pose'],
             }
reorientation = {'fly_params': odor_larva,
                 'env_params': reorientation_env,
                 'collections': ['turner', 'olfactor', 'pose'],
                 }
growth = {'fly_params': growing_rover,
          'env_params': food_grid_env,
          # 'collect_effectors': ['feeder'],
          'collections': ['feeder', 'deb'],
          # 'starvation_hours': [[0.2, 0.4]]
          }
growth_2x = {
    # 'fly_params': growing_sitter,
    'fly_params': [growing_rover, growing_sitter],
    'env_params': food_grid_env,
    'collections': ['feeder', 'deb'],
    # 'age_in_hours' : 0
    # 'starvation_hours': [[24, 48]]
}
odor_pref = {'fly_params': odor_larva_x2,
             'env_params': pref_env,
             'collections': ['olfactor'],
             }
chemorbit = {'fly_params': odor_larva,
             'env_params': chemorbit_env,
             'collections': ['olfactor', 'pose', 'dst2center'],
             }
chemotax = {'fly_params': odor_larva,
            'env_params': chemotax_env,
            'collections': ['olfactor', 'pose', 'chemotax_dst'],
            }
dispersion = {'fly_params': exploring_larva,
              'env_params': dispersion_env,
              'collections': ['pose'],
              }

maze = {'fly_params': odor_larva,
        'env_params': maze_env,
        'collections': ['olfactor', 'pose', 'dst2center'],
        }
flag = {'fly_params': [flag_larva_L, flag_larva_R],
        'env_params': game_env,
        'collections': ['olfactor', 'pose', 'dst2center'],
        }
king = {'fly_params': [king_larva_L, king_larva_R],
        'env_params': game_env,
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
    'reorientation': reorientation,
    'flag': flag,
    'king': king,
}
