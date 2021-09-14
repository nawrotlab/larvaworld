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
        conf = dtypes.get_dict('exp_conf', enrichment=dtypes.base_enrich(), **kw)
    else:
        conf = dtypes.get_dict('exp_conf', **kw)
    return conf

PI=dtypes.base_enrich(types=['PI'], bouts=[])

def source_enrich(source=(0.0, 0.0)) :
    return dtypes.base_enrich(source=source,types=['spatial', 'angular', 'source'])

def chemotaxis_exp(name, source=(0.0, 0.0), c=['olfactor'], dur=1.0) :
    e = exp(name, sim={'duration': dur}, c=c,enrichment=source_enrich(source))
    return {name: e}

def food_exp(name, c=['feeder'], dur=10.0) :
    e = exp(name, sim={'duration': dur}, c=c, en=True)
    return {name : e}

def game_exp(name, c=[], dur=20.0) :
    e = exp(name, sim={'duration': dur}, c=c)
    return {name : e}

def simple_exp(name, dur=10.0, en=True) :
    e = exp(name, sim={'duration': dur}, en=en)
    return {name: e}

def pref_exp(name, dur=5.0, c=['olfactor'], enrichment=PI, **kwargs) :
    e = exp(name, sim={'duration': dur}, c=c, enrichment=enrichment, **kwargs)
    return {name: e}


exp_dict = {
    **simple_exp('focus'),
    **simple_exp('dish'),
    **simple_exp('dispersion'),
    **chemotaxis_exp('chemotaxis_approach', source=(0.04, 0.0), dur=5.0),
    **chemotaxis_exp('chemotaxis_local', dur=5.0),
    **chemotaxis_exp('chemotaxis_diffusion', dur=10.0),
    **chemotaxis_exp('chemotaxis_RL', source=(0.04, 0.0), dur=10.0, c=['olfactor', 'memory']),
    **pref_exp('odor_pref_test'),
    **pref_exp('odor_pref_test_on_food'),
    # **pref_exp('odor_pref_train', dur=41.0, c=['olfactor', 'memory'],life_params='odor_preference_1'),
    **pref_exp('odor_pref_train', dur=41.0, c=['olfactor', 'memory'],life_params='odor_preference'),
    **pref_exp('odor_pref_RL', dur=105.0, c=['olfactor', 'memory']),
    **food_exp('patchy_food'),
    **food_exp('uniform_food'),
    **food_exp('food_grid'),
    'growth': exp('growth', sim={'duration': 24 * 60.0}, c=['feeder', 'gut']),
    'rovers_sitters': exp('rovers_sitters', sim={'duration': 180.0}, c=['feeder', 'gut'],enrichment=dtypes.base_enrich(types=['spatial'], bouts=[])),
    'reorientation': exp('reorientation', collections=['olfactor', 'pose']),
    'realistic_imitation': exp('realistic_imitation', sim={'Box2D': True}, c=['midline', 'contour']),
    'maze': exp('maze', c=['olfactor']),
    **game_exp('keep_the_flag'),
    **game_exp('capture_the_flag'),
    **game_exp('catch_me'),
    'food_at_bottom': exp('food_at_bottom', sim={'duration': 20.0, 'timestep': 0.09, 'sample': 'Fed'}, en=True),
    '4corners': exp('4corners', c=['memory'])
}


