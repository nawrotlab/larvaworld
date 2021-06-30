from lib.conf.larva_conf import *
from lib.conf.env_conf import *

# dish = {
#     'env_params': 'dish',
#     'sim_params': dtypes.get_dict('sim_params'),
#     'collections': ['pose'],
#     'enrichment': dtypes.default_enrichment()
# }
dish = dtypes.get_dict('exp_conf', env_params='dish', enrichment=dtypes.default_enrichment())
dispersion = dtypes.get_dict('exp_conf', env_params='dispersion', enrichment=dtypes.default_enrichment())
uniform_food = dtypes.get_dict('exp_conf', env_params='uniform_food', collections=['pose','feeder'])
patchy_food = dtypes.get_dict('exp_conf', env_params='patchy_food', collections=['pose','feeder'])
food_grid = dtypes.get_dict('exp_conf', env_params='food_grid', collections=['pose','feeder'])
focus = dtypes.get_dict('exp_conf', env_params='focus')
imitation = dtypes.get_dict('exp_conf', env_params='realistic_imitation',
                            sim_params=dtypes.get_dict('sim_params', Box2D=True),
                            collections=['midline', 'contour', 'pose'])
reorientation = dtypes.get_dict('exp_conf', env_params='reorientation', collections=['olfactor', 'pose'])
maze = dtypes.get_dict('exp_conf', env_params='maze', collections=['olfactor', 'pose'])
growth = dtypes.get_dict('exp_conf', env_params='growth',
                         sim_params=dtypes.get_dict('sim_params', sim_dur=24 * 60.0, dt=0.2), collections=['deb'])
rovers_sitters = dtypes.get_dict('exp_conf', env_params='rovers_sitters',
                                 sim_params=dtypes.get_dict('sim_params', sim_dur=180.0, dt=0.1),
                                 collections=['deb', 'gut'])
odor_pref_test = dtypes.get_dict('exp_conf', env_params='odor_pref_test',
                                 sim_params=dtypes.get_dict('sim_params', sim_dur=5.0),collections=['olfactor', 'pose'])
odor_pref_test_on_food = dtypes.get_dict('exp_conf', env_params='odor_pref_test_on_food',
                                 sim_params=dtypes.get_dict('sim_params', sim_dur=5.0),collections=['olfactor', 'pose', 'feeder'])
odor_pref_train = dtypes.get_dict('exp_conf', env_params='odor_pref_train',
                                 sim_params=dtypes.get_dict('sim_params', sim_dur=41.0),collections=['olfactor', 'memory'],
                                  life_params=dtypes.get_dict('life',
                                   epochs=[(1 / 12, 2 / 12), (3 / 12, 4 / 12), (5 / 12, 6 / 12), (7 / 12, 8 / 12)]))
odor_pref_RL = dtypes.get_dict('exp_conf', env_params='odor_preference_RL', sim_params=dtypes.get_dict('sim_params', sim_dur=105.0),collections=['memory'])
RL_4corners = dtypes.get_dict('exp_conf', env_params='4corners', collections=['memory'])
chemorbit = dtypes.get_dict('exp_conf', env_params='chemotaxis_local',
                                 sim_params=dtypes.get_dict('sim_params', sim_dur=1.0),collections=['olfactor', 'pose'],
                            enrichment=dtypes.default_enrichment(source=(0.0, 0.0), types=['spatial','angular','source']))
chemorbit_diffusion = dtypes.get_dict('exp_conf', env_params='chemotaxis_diffusion',
                                 sim_params=dtypes.get_dict('sim_params', sim_dur=10.0),collections=['olfactor', 'pose'],
                            enrichment=dtypes.default_enrichment(source=(0.0, 0.0), types=['spatial','angular','source']))
chemotax = dtypes.get_dict('exp_conf', env_params='chemotaxis_approach',collections=['olfactor', 'pose'],
                            enrichment=dtypes.default_enrichment(source=(0.04, 0.0), types=['spatial','angular','source']))
chemotaxis_RL = dtypes.get_dict('exp_conf', env_params='chemotaxis_RL',collections=['olfactor', 'pose', 'memory'],
                            enrichment=dtypes.default_enrichment(source=(0.04, 0.0), types=['spatial','angular','source']))
capture_the_flag = dtypes.get_dict('exp_conf', env_params='capture_the_flag', sim_params=dtypes.get_dict('sim_params', sim_dur=20.0))
keep_the_flag = dtypes.get_dict('exp_conf', env_params='keep_the_flag', sim_params=dtypes.get_dict('sim_params', sim_dur=20.0))
catch_me = dtypes.get_dict('exp_conf', env_params='catch_me', sim_params=dtypes.get_dict('sim_params', sim_dur=20.0))
food_at_bottom = dtypes.get_dict('exp_conf', env_params='food_at_bottom', sim_params=dtypes.get_dict('sim_params', sim_dur=1.5, dt=0.09, sample_dataset='Fed'),
                                 enrichment=dtypes.default_enrichment())