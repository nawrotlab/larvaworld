from lib.conf import dtype_dicts as dtypes
from lib.conf.init_dtypes import processing_types

from lib.conf.conf import imitation_exp


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
        enrichment = dtypes.get_dict('enrichment', source=en['source'],
                                     types=processing_types(['angular', 'spatial', 'source']))
    else:
        raise NotImplementedError
    exp_kws = {'enrichment': enrichment}
    if o is not None:
        opt = optimization(o, **o_kws)
    else:
        opt = None
    conf = dtypes.get_dict('batch_conf', exp=exp, exp_kws=exp_kws, optimization=opt, **kwargs)
    return {exp: conf}


batch_dict = {
    **batch('chemotaxis_approach', space_search={
        'pars': ['Odor.mean', 'decay_coef'],
        # 'pars': ['Odor.mean', 'decay_coef'],
        'ranges': [(300.0, 1300.0), (0.1, 0.5)],
        # 'ranges': [(300.0, 1300.0), (0.1, 0.5)],
        'Ngrid': [3, 3]
    }, o='final_dst_to_source', en={'source': (0.04, 0.0)}),
    **batch('chemotaxis_local', space_search={
        'pars': ['Odor.mean', 'decay_coef'],
        'ranges': [(300.0, 1300.0), (0.1, 0.5)],
        'Ngrid': [3, 3]
    }, o='final_dst_to_center', en={'source': (0.0, 0.0)}),
    **batch('odor_pref_test', space_search={
        'pars': ['odor_dict.CS.mean', 'odor_dict.UCS.mean'],
        'ranges': [(-100.0, 100.0), (-100.0, 100.0)],
        'Ngrid': [3, 3]
    }, batch_methods=batch_methods(run='odor_preference', post='null', final='odor_preference'), en='PI'),
    **batch('odor_pref_train_short', space_search={
        'pars': ['olfactor_noise', 'decay_coef'],
        'ranges': [(0.0, 0.4), (0.1, 0.5)],
        'Ngrid': [2, 2]
    }, batch_methods=batch_methods(run='odor_preference', post='null', final='odor_preference'), en='PI'),
    **batch('odor_pref_train', space_search={
        'pars': ['olfactor_noise', 'decay_coef'],
        'ranges': [(0.0, 0.4), (0.1, 0.5)],
        'Ngrid': [2, 2]
    }, batch_methods=batch_methods(run='odor_preference', post='null', final='odor_preference'), en='PI'),
    **batch('patchy_food', space_search={
        'pars': ['EEB', 'initial_freq'],
        'ranges': [(0.0, 1.0), (1.5, 2.5)],
        'Ngrid': [3, 3]
    }, o='ingested_food_volume'),
    **batch('food_grid', space_search={
        'pars': ['EEB', 'EEB_decay'],
        'ranges': [(0.0, 1.0), (0.1, 2.0)],
        'Ngrid': [6, 6]
    }, o='ingested_food_volume'),
    **batch('growth', space_search={
        'pars': ['EEB', 'hunger_gain'],
        'ranges': [(0.5, 0.8), (0.0, 0.0)],
        'Ngrid': [8, 1]
    }, o='deb_f_deviation', o_kws={'max_Nsims': 20, 'operations': {'mean': True, 'abs': True}}),
    **batch('rovers_sitters', space_search={
        'pars': ['substrate_quality', 'hours_as_larva'],
        'ranges': [(0.5, 0.8), (0, 100)],
        'Ngrid': [2, 2]
    }, batch_methods=batch_methods(run='deb', post='null', final='deb')),

}


def fit_tortuosity_batch(d, model='imitation', exp='dish', idx=0):
    conf = dtypes.get_dict('batch_conf', exp=None,
                           exp_kws={'enrichment': dtypes.base_enrich(types=['tortuosity'], bouts=[])},
                           space_search={
                               'pars': ['activation_noise', 'base_activation'],
                               'ranges': [(0.0, 2.0), (15.0, 25.0)],
                               'Ngrid': [3, 3]
                           },
                           optimization=optimization('tortuosity_20_mean',
                                                     **{'max_Nsims': 120, 'operations': {'mean': True}}))
    conf['exp'] = imitation_exp(d.config, model=model, exp=exp, idx=idx)
    conf['batch_id'] = f'imitation_batchrun_{idx}'
    conf['batch_type'] = 'imitation'
    return conf
