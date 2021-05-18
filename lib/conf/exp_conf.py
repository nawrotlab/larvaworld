from lib.conf.larva_conf import *
from lib.conf.env_conf import *

dish = {
        'env_params': 'dish',
        'sim_params': dtypes.get_dict('sim_params'),
        'collections': ['pose'],
        }

uniform_food = {
                'env_params': 'uniform_food',
                'sim_params': dtypes.get_dict('sim_params'),
                'collections': ['crawler', 'feeder', 'intermitter'],
                }
patchy_food = {
               'env_params': 'patchy_food',
               'sim_params': dtypes.get_dict('sim_params'),
               'collections': ['crawler', 'feeder', 'intermitter'],
               }
food_grid = {
             'env_params': 'food_grid',
             'sim_params': dtypes.get_dict('sim_params'),
             'collections': ['crawler', 'feeder', 'intermitter'],
             }
focus = {
         'env_params': 'focus',
         'sim_params': dtypes.get_dict('sim_params'),
         'collections': ['turner', 'pose'],
         }
imitation = {
             'env_params': 'realistic_imitation',
             'sim_params': dtypes.get_dict('sim_params', Box2D=True),
             'collections': ['midline', 'contour', 'pose'],
             }
reorientation = {
                 'env_params': 'reorientation',
                 'sim_params': dtypes.get_dict('sim_params'),
                 'collections': ['turner', 'olfactor', 'pose'],
                 }
growth = {
          'env_params': 'growth',
          'sim_params': dtypes.get_dict('sim_params', sim_dur=600.0),
          # 'collect_effectors': ['feeder'],
          # 'collections': ['pose'],
          'collections': ['feeder', 'deb'],
          }
rovers_sitters = {

    'env_params': 'rovers_sitters',
    'sim_params': dtypes.get_dict('sim_params', sim_dur=600.0),
    'collections': ['feeder', 'deb'],
    # 'age_in_hours' : 0
    # 'starvation_hours': [[24, 48]]
}
odor_pref = {
             'env_params': 'odor_preference',
             'sim_params': dtypes.get_dict('sim_params'),
             'collections': ['olfactor'],
             }

odor_pref_test = {
             'env_params': 'odor_pref_test',
             'sim_params': dtypes.get_dict('sim_params'),
             'collections': ['olfactor'],
             }

odor_pref_train = {
             'env_params': 'odor_pref_train',
             'sim_params': dtypes.get_dict('sim_params', sim_dur=35.0),
             'collections': ['olfactor', 'memory'],
            'life_params' : dtypes.get_dict('life', starvation_hours=[(1/12, 2/12), (3/12, 4/12), (5/12, 6/12)])
             }

odor_pref_RL = {
             'env_params': 'odor_preference_RL',
             'sim_params': dtypes.get_dict('sim_params'),
             'collections': ['memory'],
             }

chemorbit = {
             'env_params': 'chemotaxis_local',
             'sim_params': dtypes.get_dict('sim_params', sim_dur=10.0),
             # 'collections': ['dst2center'],
             'collections': ['turner', 'olfactor', 'pose', 'dst2center'],
             }

chemorbit_diffusion = {
             'env_params': 'chemotaxis_diffusion',
             'sim_params': dtypes.get_dict('sim_params', sim_dur=10.0),
             'collections': ['turner', 'dst2center', 'pose', 'dst2center'],
             }

chemotax = {
            'env_params': 'chemotaxis_approach',
            'sim_params': dtypes.get_dict('sim_params'),
            'collections': ['turner', 'olfactor', 'pose', 'chemotax_dst'],
            }

chemotaxis_RL = {
             'env_params': 'chemotaxis_RL',
             'sim_params': dtypes.get_dict('sim_params'),
             'collections': ['dst2center', 'memory'],
             }

dispersion = {
              'env_params': 'dispersion',
              'sim_params': dtypes.get_dict('sim_params'),
              'collections': ['pose'],
              }

maze = {
        'env_params': 'maze',
        'sim_params': dtypes.get_dict('sim_params', sim_dur=10.0),
        'collections': ['olfactor', 'pose', 'dst2center'],
        }
capture_the_flag = {
        'env_params': 'capture_the_flag',
        'sim_params': dtypes.get_dict('sim_params', sim_dur=20.0),
        'collections': ['pose'],
        }
keep_the_flag = {
        'env_params': 'keep_the_flag',
        'sim_params': dtypes.get_dict('sim_params', sim_dur=20.0),
        'collections': ['pose'],
        }

catch_me = {
        'env_params': 'catch_me',
        'sim_params': dtypes.get_dict('sim_params', sim_dur=20.0),
        'collections': ['pose'],
        }

