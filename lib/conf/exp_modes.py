from lib.conf.larva_modes import *
from lib.conf.env_modes import *

dish = {
        'env_params': 'dish',
        # 'env_params': dish_env,
        'collections': ['pose'],
        }

feed_scatter = {
                'env_params': 'uniform food',
                # 'env_params': uniform_food_env,
                'collections': ['crawler', 'feeder', 'intermitter'],
                }
feed_patchy = {
               'env_params': 'patchy food',
               # 'env_params': patchy_food_env,
               'collections': ['crawler', 'feeder', 'intermitter'],
               }
feed_grid = {
             'env_params': 'food grid',
             # 'env_params': food_grid_env,
             'collections': ['crawler', 'feeder', 'intermitter'],
             }
focus = {
         'env_params': 'focused view',
         # 'env_params': focus_env,
         'collections': ['turner', 'pose'],
         }
imitation = {
             'env_params': 'realistic imitation',
             # 'env_params': imitation_env_p,
             'collections': ['midline', 'contour', 'pose'],
             }
reorientation = {
                 'env_params': 'reorientation',
                 # 'env_params': reorientation_env,
                 'collections': ['turner', 'olfactor', 'pose'],
                 }
growth = {
          'env_params': 'growth',
          # 'env_params': growth_env,
          # 'collect_effectors': ['feeder'],
          'collections': ['feeder', 'deb'],
          }
growth_2x = {

    'env_params': 'rovers-sitters',
    # 'env_params': growth_2x_env,
    'collections': ['feeder', 'deb'],
    # 'age_in_hours' : 0
    # 'starvation_hours': [[24, 48]]
}
odor_pref = {
             'env_params': 'odor preference',
             # 'env_params': pref_env,
             'collections': ['olfactor'],
             }
chemorbit = {
             'env_params': 'chemotaxis local',
             # 'env_params': chemorbit_env,
             'collections': ['olfactor', 'pose', 'dst2center'],
             }
chemotax = {
            'env_params': 'chemotaxis approach',
            # 'env_params': chemotax_env,
            'collections': ['olfactor', 'pose', 'chemotax_dst'],
            }
dispersion = {
              'env_params': 'dispersion',
              # 'env_params': dispersion_env,
              'collections': ['pose'],
              }

maze = {
        'env_params': 'maze',
        # 'env_params': maze_env,
        'collections': ['olfactor', 'pose', 'dst2center'],
        }
flag = {
        'env_params': 'flag to base',
        # 'env_params': flag_env,
        'collections': ['olfactor', 'pose', 'dst2center'],
        }
king = {
        'env_params': 'keep the flag',
        # 'env_params': king_env,
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
