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
    'collections': ['feeder'],
}
patchy_food = {
    'env_params': 'patchy_food',
    'sim_params': dtypes.get_dict('sim_params'),
    'collections': ['feeder'],
}
food_grid = {
    'env_params': 'food_grid',
    'sim_params': dtypes.get_dict('sim_params'),
    'collections': ['feeder', 'pose'],
}
focus = {
    'env_params': 'focus',
    'sim_params': dtypes.get_dict('sim_params'),
    'collections': ['pose'],
}
imitation = {
    'env_params': 'realistic_imitation',
    'sim_params': dtypes.get_dict('sim_params', Box2D=True),
    'collections': ['midline', 'contour', 'pose'],
}
reorientation = {
    'env_params': 'reorientation',
    'sim_params': dtypes.get_dict('sim_params'),
    'collections': ['olfactor', 'pose'],
}
growth = {
    'env_params': 'growth',
    'sim_params': dtypes.get_dict('sim_params', sim_dur=24*60.0, dt=0.2),
    # 'collect_effectors': ['feeder'],
    # 'collections': ['pose'],
    'collections': ['deb'],
}
rovers_sitters = {

    'env_params': 'rovers_sitters',
    'sim_params': dtypes.get_dict('sim_params', sim_dur=180.0, dt=0.1),
    'collections': ['deb', 'gut'],
    # 'collections': ['feeder', 'deb', 'gut'],
    # 'age_in_hours' : 0
    # 'epochs': [[24, 48]]
}
# odor_pref = {
#     'env_params': 'odor_preference',
#     'sim_params': dtypes.get_dict('sim_params', sim_dur=5.0),
#     'collections': ['olfactor'],
# }

odor_pref_test = {
    'env_params': 'odor_pref_test',
    'sim_params': dtypes.get_dict('sim_params', sim_dur=5.0),
    'collections': ['olfactor', 'pose'],
}

odor_pref_test_on_food = {
    'env_params': 'odor_pref_test_on_food',
    'sim_params': dtypes.get_dict('sim_params', sim_dur=5.0),
    'collections': ['olfactor', 'feeder'],
}

odor_pref_train = {
    'env_params': 'odor_pref_train',
    'sim_params': dtypes.get_dict('sim_params', sim_dur=41.0),
    'collections': ['olfactor', 'memory'],
    'life_params': dtypes.get_dict('life', epochs=[(1 / 12, 2 / 12), (3 / 12, 4 / 12), (5 / 12, 6 / 12), (7 / 12, 8 / 12)])
}

odor_pref_RL = {
    'env_params': 'odor_preference_RL',
    'sim_params': dtypes.get_dict('sim_params', sim_dur=105.0),
    'collections': ['memory'],
}

RL_4corners = {
    'env_params': '4corners',
    'sim_params': dtypes.get_dict('sim_params'),
    'collections': ['memory'],
}

chemorbit = {
    'env_params': 'chemotaxis_local',
    'sim_params': dtypes.get_dict('sim_params', sim_dur=1.0),
    # 'collections': ['dst2center'],
    'collections': ['olfactor', 'pose'],
}

chemorbit_diffusion = {
    'env_params': 'chemotaxis_diffusion',
    'sim_params': dtypes.get_dict('sim_params', sim_dur=10.0),
    'collections': ['pose'],
}

chemotax = {
    'env_params': 'chemotaxis_approach',
    'sim_params': dtypes.get_dict('sim_params'),
    'collections': ['olfactor', 'pose'],
}

chemotaxis_RL = {
    'env_params': 'chemotaxis_RL',
    'sim_params': dtypes.get_dict('sim_params'),
    'collections': ['memory'],
}

dispersion = {
    'env_params': 'dispersion',
    'sim_params': dtypes.get_dict('sim_params'),
    'collections': ['pose'],
}

maze = {
    'env_params': 'maze',
    'sim_params': dtypes.get_dict('sim_params', sim_dur=10.0),
    'collections': ['olfactor', 'pose'],
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

food_at_bottom = {
    'env_params': 'food_at_bottom',
    'sim_params': dtypes.get_dict('sim_params', sim_dur=1.5, dt=0.09, sample_dataset='Fed'),
    'collections': ['pose']
}

