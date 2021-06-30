from lib.conf.larva_conf import *
from lib.conf.env_conf import *
from lib.conf import dtype_dicts as dtypes
# dish = {
#     'env_params': 'dish',
#     'sim_params': dtypes.get_dict('sim_params'),
#     'collections': ['pose'],
#     'enrichment': dtypes.base_enrich()
# }
def exp(env_params, en=False,sim={},**kwargs):
    sim_params = dtypes.get_dict('sim_params', **sim)
    if en :
        return dtypes.get_dict('exp_conf', env_params=env_params, enrichment=dtypes.base_enrich(), sim_params = sim_params, **kwargs)
    else :
        return dtypes.get_dict('exp_conf', env_params=env_params, sim_params = sim_params,**kwargs)

exp_dict = {
        'focus': exp('focus'),
        'dish': exp('dish', en=True),
        'dispersion': exp('dispersion', en=True),
        'chemotaxis_approach': exp('chemotaxis_approach', collections=['olfactor', 'pose'],
                                   enrichment=dtypes.base_enrich(source=(0.04, 0.0), types=['spatial', 'angular', 'source'])),
        'chemotaxis_local': exp('chemotaxis_local', sim={'duration':1.0}, collections=['olfactor', 'pose'],
                                enrichment=dtypes.base_enrich(source=(0.0, 0.0), types=['spatial', 'angular', 'source'])),
        'chemotaxis_diffusion': exp('chemotaxis_diffusion', sim={'duration':10.0}, collections=['olfactor', 'pose'],
                                    enrichment=dtypes.base_enrich(source=(0.0, 0.0), types=['spatial', 'angular', 'source'])),
        'odor_pref_test': exp('odor_pref_test',sim={'duration':5.0},collections=['olfactor', 'pose']),
        'odor_pref_test_on_food': exp('odor_pref_test_on_food',sim={'duration':5.0},collections=['olfactor', 'pose', 'feeder']),
        'odor_pref_train': exp('odor_pref_train',sim={'duration':41.0},collections=['olfactor', 'memory'],
                                  life_params=dtypes.get_dict('life',
                                   epochs=[(1 / 12, 2 / 12), (3 / 12, 4 / 12), (5 / 12, 6 / 12), (7 / 12, 8 / 12)])),
        'odor_pref_RL': exp('odor_preference_RL', sim={'duration':105.0},collections=['memory']),
        'patchy_food': exp('patchy_food', collections=['pose','feeder'],en=True),
        'uniform_food': exp('uniform_food', collections=['pose','feeder'],en=True),
        'food_grid': exp('food_grid', collections=['pose','feeder'], en=True),
        'growth': exp('growth',sim={'duration':24 * 60.0, 'timestep':0.2}, collections=['deb']),
        'rovers_sitters': exp('rovers_sitters',sim={'duration':180.0},collections=['deb', 'gut']),
        'reorientation': exp('reorientation', collections=['olfactor', 'pose']),
        'realistic_imitation': exp('realistic_imitation',sim={'Box2D':True},collections=['midline', 'contour', 'pose']),
        'maze': exp('maze', collections=['olfactor', 'pose']),
        'keep_the_flag': exp('keep_the_flag', sim={'duration':20.0}),
        'capture_the_flag': exp('capture_the_flag', sim={'duration':20.0}),
        'catch_me': exp('catch_me', sim={'duration':20.0}),
        'chemotaxis_RL': exp('chemotaxis_RL', collections=['olfactor', 'pose', 'memory'],
                             enrichment=dtypes.base_enrich(source=(0.04, 0.0), types=['spatial', 'angular', 'source'])),
        'food_at_bottom': exp('food_at_bottom', sim={'duration':20.0, 'timestep' : 0.09, 'sample' : 'Fed'},
                                 en=True),
        '4corners': exp('4corners', collections=['memory']),
    }
