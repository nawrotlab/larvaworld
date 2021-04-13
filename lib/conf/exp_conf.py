from lib.conf.larva_conf import *
from lib.conf.env_conf import *

dish = {
        'env_params': 'dish',
        # 'env_params': dish_env,
        'collections': ['pose'],
        }

uniform_food = {
                'env_params': 'uniform_food',
                # 'env_params': uniform_food_env,
                'collections': ['crawler', 'feeder', 'intermitter'],
                }
patchy_food = {
               'env_params': 'patchy_food',
               # 'env_params': patchy_food_env,
               'collections': ['crawler', 'feeder', 'intermitter'],
               }
food_grid = {
             'env_params': 'food_grid',
             # 'env_params': food_grid_env,
             'collections': ['crawler', 'feeder', 'intermitter'],
             }
focus = {
         'env_params': 'focus',
         # 'env_params': focus_env,
         'collections': ['turner', 'pose'],
         }
imitation = {
             'env_params': 'realistic_imitation',
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
rovers_sitters = {

    'env_params': 'rovers_sitters',
    # 'env_params': growth_2x_env,
    'collections': ['feeder', 'deb'],
    # 'age_in_hours' : 0
    # 'starvation_hours': [[24, 48]]
}
odor_pref = {
             'env_params': 'odor_preference',
             # 'env_params': pref_env,
             'collections': ['olfactor'],
             }
chemorbit = {
             'env_params': 'chemotaxis_local',
             # 'env_params': chemorbit_env,
             # 'collections': ['dst2center'],
             'collections': ['olfactor', 'pose', 'dst2center'],
             }

chemorbit_diffusion = {
             'env_params': 'chemotaxis_diffusion',
             # 'env_params': chemorbit_env,
             'collections': ['dst2center'],
             }

chemotax = {
            'env_params': 'chemotaxis_approach',
            # 'env_params': chemotax_env,
            'collections': ['olfactor', 'pose', 'chemotax_dst'],
            }

chemotaxis_RL = {
             'env_params': 'chemotaxis_RL',
             # 'env_params': chemorbit_env,
             'collections': ['dst2center'],
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
capture_the_flag = {
        'env_params': 'capture_the_flag',
        # 'env_params': flag_env,
        'collections': ['olfactor', 'pose', 'dst2center'],
        }
keep_the_flag = {
        'env_params': 'keep_the_flag',
        # 'env_params': king_env,
        'collections': ['olfactor', 'pose', 'dst2center'],
        }
exp_types = {
    'focus': focus,
    'dish': dish,
    'dispersion': dispersion,
    'chemorbit': chemorbit,
    'chemorbit_diffusion': chemorbit_diffusion,
    'chemotax': chemotax,
    'chemotaxis_RL': chemotaxis_RL,
    'odor_pref': odor_pref,
    'food_grid': food_grid,
    'patchy_food': patchy_food,
    'uniform_food': uniform_food,
    'growth': growth,
    'rovers_sitters': rovers_sitters,
    'maze': maze,
    'imitation': imitation,
    'reorientation': reorientation,
    'flag': capture_the_flag,
    'king': keep_the_flag,
}
