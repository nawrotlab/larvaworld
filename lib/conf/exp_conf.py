import copy

from lib.conf import dtype_dicts as dtypes
import lib.aux.functions as fun
import lib.aux.naming as nam
from lib.conf.conf import imitation_exp
from lib.stor import paths
import numpy as np



def exp(env_name, exp_name=None, en=False, sim={}, c=[], as_entry=True, **kwargs):
    kw = {
        'sim_params': dtypes.get_dict('sim_params', **sim),
        'env_params': env_name,
        'collections': ['pose'] + c,
        # **kwargs
    }
    kw.update(kwargs)
    if en:
        exp_conf = dtypes.get_dict('exp_conf', enrichment=dtypes.base_enrich(), **kw)
    else:
        exp_conf = dtypes.get_dict('exp_conf', **kw)
    if not as_entry:
        return exp_conf
    else:
        if exp_name is None:
            exp_name = env_name
        return {exp_name: exp_conf}


PI = dtypes.base_enrich(types=['PI'], bouts=[])


def source_enrich(source=(0.0, 0.0)):
    return dtypes.base_enrich(source=source, types=['spatial', 'angular', 'source'])


def chemotaxis_exp(name, source=(0.0, 0.0), c=['olfactor'], dur=5.0, **kwargs):
    return exp(name, sim={'duration': dur}, c=c, enrichment=source_enrich(source), **kwargs)


def food_exp(name, c=['feeder'], dur=10.0, **kwargs):
    return exp(name, sim={'duration': dur}, c=c, en=True, **kwargs)


def game_exp(name, c=[], dur=20.0, **kwargs):
    return exp(name, sim={'duration': dur}, c=c, **kwargs)


def deb_exp(name, c=['feeder', 'gut'], dur=60.0, **kwargs):
    return exp(name, sim={'duration': dur}, c=c, **kwargs)


def simple_exp(name, dur=10.0, en=True, **kwargs):
    return exp(name, sim={'duration': dur}, en=en, **kwargs)


def pref_exp(name, dur=5.0, c=['olfactor'], enrichment=dtypes.base_enrich(types=['PI'], bouts=[]), **kwargs):
    return exp(name, sim={'duration': dur}, c=c, enrichment=enrichment, **kwargs)



grouped_exp_dict = {
    'exploration': {
        **simple_exp('focus'),
        **simple_exp('dish'),
        **simple_exp('nengo_dish', dur=3.0),
        # **simple_exp('nengo_dish', dur=2.0, enrichment=dtypes.base_enrich(preprocessing={'rescale_by' : 1000}), en=False),
        **simple_exp('dispersion')},

    'chemotaxis': {**chemotaxis_exp('chemotaxis_approach', source=(0.04, 0.0)),
                   **chemotaxis_exp('chemotaxis_local'),
                   **chemotaxis_exp('chemotaxis_diffusion', dur=10.0),
                   **chemotaxis_exp('chemotaxis_RL', source=(0.04, 0.0), dur=10.0, c=['olfactor', 'memory']),
                   **chemotaxis_exp('reorientation'),
                   **exp('food_at_bottom', sim={'duration': 20.0, 'timestep': 0.09}, en=True)},

    'odor_preference': {**pref_exp('odor_pref_test'),
                        **pref_exp('odor_pref_test_on_food'),
                        **pref_exp('odor_pref_train', exp_name='odor_pref_train_short', dur=1.56,
                                   c=['olfactor', 'memory'],
                                   life_params='odor_preference_short'),
                        **pref_exp('odor_pref_train', dur=41.0, c=['olfactor', 'memory'],
                                   life_params='odor_preference'),
                        **pref_exp('odor_pref_RL', dur=105.0, c=['olfactor', 'memory'])},

    'foraging': {**food_exp('patchy_food'),
                 **food_exp('uniform_food'),
                 **food_exp('food_grid'),
                 **food_exp('single_patch'),
                 **exp('4corners', c=['memory'])},

    'growth': {**deb_exp('growth', dur=24 * 60.0),
               **deb_exp('rovers_sitters', dur=180.0, enrichment=dtypes.base_enrich(types=['spatial'], bouts=[]))},

    'games': {**game_exp('maze', c=['olfactor']),
              **game_exp('keep_the_flag'),
              **game_exp('capture_the_flag'),
              **game_exp('catch_me')},

    'other': {
        **exp('realistic_imitation', sim={'Box2D': True}, c=['midline', 'contour']),
        'imitation' : imitation_exp(paths.RefConf, model='explorer'),
        # **exp('imitation', exp_name='imitation', sample=paths.RefConf),
              }

}

# exp_dict = fun.merge_dicts(list(grouped_exp_dict.values()))
# grouped_exp_dict = {k:{'simulations': list(v.keys())} for k,v in grouped_exp_dict.items()}
#
# if __name__ == "__main__":
#     print(exp_groups)
