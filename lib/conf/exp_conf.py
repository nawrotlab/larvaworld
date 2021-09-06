import lib.conf.dtype_dicts
import lib.conf.init_dtypes
from lib.conf.larva_conf import *
from lib.conf.env_conf import *
from lib.conf import dtype_dicts as dtypes


# dish = {
#     'env_params': 'dish',
#     'sim_params': dtypes.get_dict('sim_params'),
#     'collections': ['pose'],
#     'enrichment': dtypes.base_enrich()
# }
def exp(env_params, en=False, sim={}, c=[], **kwargs):
    kw = {
        'sim_params': dtypes.get_dict('sim_params', **sim),
        'env_params': env_params,
        'collections': ['pose'] + c,
        **kwargs
    }
    if en:
        conf = dtypes.get_dict('exp_conf', enrichment=lib.conf.dtype_dicts.base_enrich(), **kw)
    else:
        conf = dtypes.get_dict('exp_conf', **kw)
    return conf


exp_dict = {
    'focus': exp('focus'),
    'dish': exp('dish', en=True),
    'dispersion': exp('dispersion', en=True),
    'chemotaxis_approach': exp('chemotaxis_approach', c=['olfactor'],
                               enrichment=lib.conf.dtype_dicts.base_enrich(source=(0.04, 0.0),
                                                                           types=['spatial', 'angular', 'source'])),
    'chemotaxis_local': exp('chemotaxis_local', sim={'duration': 1.0}, c=['olfactor'],
                            enrichment=lib.conf.dtype_dicts.base_enrich(source=(0.0, 0.0), types=['spatial', 'angular', 'source'])),
    'chemotaxis_diffusion': exp('chemotaxis_diffusion', sim={'duration': 10.0}, c=['olfactor'],
                                enrichment=lib.conf.dtype_dicts.base_enrich(source=(0.0, 0.0),
                                                                            types=['spatial', 'angular', 'source'])),
    'odor_pref_test': exp('odor_pref_test', sim={'duration': 5.0}, c=['olfactor']),
    'odor_pref_test_on_food': exp('odor_pref_test_on_food', sim={'duration': 5.0}, c=['olfactor', 'feeder']),
    'odor_pref_train': exp('odor_pref_train', sim={'duration': 41.0}, c=['olfactor', 'memory'],life_params='odor_preference'),
    'odor_pref_RL': exp('odor_preference_RL', sim={'duration': 105.0}, c=['olfactor','memory']),
    'patchy_food': exp('patchy_food', c=['feeder'], en=True),
    'uniform_food': exp('uniform_food', c=['feeder'], en=True),
    'food_grid': exp('food_grid', c=['feeder'], en=True),
    'growth': exp('growth', sim={'duration': 24 * 60.0, 'timestep': 0.2}, c=['feeder', 'gut']),
    'rovers_sitters': exp('rovers_sitters', sim={'duration': 180.0}, c=['feeder', 'gut'],
                          enrichment=dtypes.get_dict('enrichment', types=['spatial'])),
    'reorientation': exp('reorientation', collections=['olfactor', 'pose']),
    'realistic_imitation': exp('realistic_imitation', sim={'Box2D': True}, c=['midline', 'contour']),
    'maze': exp('maze', c=['olfactor']),
    'keep_the_flag': exp('keep_the_flag', sim={'duration': 20.0}),
    'capture_the_flag': exp('capture_the_flag', sim={'duration': 20.0}),
    'catch_me': exp('catch_me', sim={'duration': 20.0}),
    'chemotaxis_RL': exp('chemotaxis_RL', c=['olfactor', 'memory'],
                         enrichment=lib.conf.dtype_dicts.base_enrich(source=(0.04, 0.0), types=['spatial', 'angular', 'source'])),
    'food_at_bottom': exp('food_at_bottom', sim={'duration': 20.0, 'timestep': 0.09, 'sample': 'Fed'}, en=True),
    '4corners': exp('4corners', c=['memory'])
}


