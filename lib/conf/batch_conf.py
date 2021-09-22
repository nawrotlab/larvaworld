import subprocess

from lib.conf import dtype_dicts as dtypes
from lib.conf.init_dtypes import processing_types


def optimization(fit_par, minimize=True, threshold=0.0001, max_Nsims=10, Nbest=4,
                 operations={'mean': True, 'std': False, 'abs': False}):
    return {
        'fit_par': fit_par,
        'operations': operations,
        'minimize': minimize,
        'threshold': threshold,
        'max_Nsims': max_Nsims,
        'Nbest': Nbest
    }


def batch_methods(run='default', post='default', final='null'):
    return {'run': run,
            'post': post,
            'final': final}


def batch(exp, en=None, o=None, o_kws={}, **kwargs):
    if en is None:
        # enrichment=dtypes.base_enrich()
        enrichment = dtypes.get_dict('enrichment')
    elif en == 'PI':
        enrichment = dtypes.base_enrich(types=['PI'], bouts=[])
    elif 'source' in en.keys():
        enrichment = dtypes.get_dict('enrichment', source=en['source'], types=processing_types(['angular','spatial','source']))
    else:
        raise NotImplementedError
    exp_kws = {'enrichment': enrichment}
    if o is not None:
        opt = optimization(o, **o_kws)
    else:
        opt = None
    return dtypes.get_dict('batch_conf', exp=exp, exp_kws=exp_kws, optimization=opt, **kwargs)


batch_dict = {
    'navigation': batch('chemotaxis_approach', space_search={
        'pars': ['Odor.mean', 'decay_coef'],
        # 'pars': ['Odor.mean', 'decay_coef'],
        'ranges': [(300.0, 1300.0), (0.1, 0.5)],
        # 'ranges': [(300.0, 1300.0), (0.1, 0.5)],
        'Ngrid': [3, 3]
    }, o='final_dst_to_source', en={'source': (0.04, 0.0)}),
    'local-search': batch('chemotaxis_local', space_search={
        'pars': ['Odor.mean', 'decay_coef'],
        'ranges': [(300.0, 1300.0), (0.1, 0.5)],
        'Ngrid': [3, 3],'max_Nsims':8
    }, o='final_dst_to_center', en={'source': (0.0, 0.0)}),
    'odor-preference_test': batch('odor_pref_test', space_search={
        'pars': ['odor_dict.CS.mean', 'odor_dict.UCS.mean'],
        'ranges': [(-100.0, 100.0), (-100.0, 100.0)],
        'Ngrid': [3, 3]
    }, batch_methods=batch_methods(run='odor_preference', post='null', final='odor_preference'), en='PI'),
    'odor-preference_complete_short': batch('odor_pref_train_short', space_search={
        'pars': ['olfactor_noise', 'decay_coef'],
        'ranges': [(0.0, 0.4), (0.1, 0.5)],
        'Ngrid': [2, 2]
    }, batch_methods=batch_methods(run='odor_preference', post='null', final='odor_preference'), en='PI'),
    'odor-preference_complete': batch('odor_pref_train', space_search={
        'pars': ['olfactor_noise', 'decay_coef'],
        'ranges': [(0.0, 0.4), (0.1, 0.5)],
        'Ngrid': [2, 2]
    }, batch_methods=batch_methods(run='odor_preference', post='null', final='odor_preference'), en='PI'),
    'food_patchy': batch('patchy_food', space_search={
        'pars': ['EEB', 'initial_freq'],
        'ranges': [(0.0, 1.0), (1.5, 2.5)],
        'Ngrid': [3, 3]
    }, o='ingested_food_volume'),
    'food_grid': batch('food_grid', space_search={
        'pars': ['EEB', 'EEB_decay'],
        'ranges': [(0.0, 1.0), (0.1, 2.0)],
        'Ngrid': [6, 6]
    }, o='ingested_food_volume'),
    'growth': batch('growth', space_search={
        'pars': ['EEB', 'hunger_gain'],
        'ranges': [(0.5, 0.8), (0.0, 0.0)],
        'Ngrid': [8, 1]
    }, o='deb_f_deviation', o_kws={'max_Nsims': 20, 'operations': {'mean': True, 'abs': True}}),
    'roversVSsitters': batch('rovers_sitters', space_search={
        'pars': ['substrate_quality', 'hours_as_larva'],
        'ranges': [(0.5, 0.8), (0, 100)],
        'Ngrid': [2, 2]
    }, batch_methods=batch_methods(run='deb', post='null', final='deb')),
}