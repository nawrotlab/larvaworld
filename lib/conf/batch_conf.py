from lib.conf.init_dtypes import null_dict, enrichment_dict

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


def batch(exp, en=None, ss=None, o=None, o_kws={}, **kwargs):
    if en is None:
        enrichment = null_dict('enrichment')
    elif en == 'PI':
        enrichment = enrichment_dict(types=['PI'], bouts=[])
    elif 'source' in en.keys():
        enrichment = enrichment_dict(source=en['source'],types=['angular', 'spatial', 'source'])
    else:
        raise NotImplementedError
    exp_kws = {'enrichment': enrichment}
    if o is not None:
        opt = optimization(o, **o_kws)
    else:
        opt = None
    if ss is not None:
        ss = {p: null_dict('space_search_par', range=r, Ngrid=N) for p, (r, N) in ss.items()}
    conf = null_dict('batch_conf', exp=exp, exp_kws=exp_kws, optimization=opt, space_search=ss, **kwargs)
    return {exp: conf}


batch_dict = {
    **batch('chemotaxis_approach',
            ss={'Odor.mean': [(300.0, 1300.0), 3],'decay_coef': [(0.1, 0.5), 3]},
            o='final_dst_to_source', en={'source': (0.04, 0.0)}),
    **batch('chemotaxis_local',
            ss={'Odor.mean': [(300.0, 1300.0), 3],'decay_coef': [(0.1, 0.5), 3]},
            o='final_dst_to_center', en={'source': (0.0, 0.0)}),
    **batch('odor_pref_test',
ss={'odor_dict.CS.mean': [(-100.0, 100.0), 3],'odor_dict.UCS.mean': [(-100.0, 100.0), 3]},
            batch_methods=batch_methods(run='odor_preference', post='null', final='odor_preference'), en='PI'),
    **batch('odor_pref_train_short',
            ss={'olfactor_noise': [(0.0, 0.4), 2],'decay_coef': [(0.1, 0.5), 2]},
             batch_methods=batch_methods(run='odor_preference', post='null', final='odor_preference'), en='PI'),
    **batch('odor_pref_train', ss={'olfactor_noise': [(0.0, 0.4), 2],'decay_coef': [(0.1, 0.5), 2]},
            batch_methods=batch_methods(run='odor_preference', post='null', final='odor_preference'), en='PI'),
    **batch('patchy_food',
            ss={'EEB': [(0.0, 1.0), 3],'initial_freq': [(1.5, 2.5), 3]},
            o='ingested_food_volume'),
    **batch('food_grid',
            ss={'EEB': [(0.0, 1.0), 6],'EEB_decay': [(0.1, 2.0), 6]},
            o='ingested_food_volume'),
    **batch('growth',
            ss={'EEB': [(0.5, 0.8), 8],'hunger_gain': [(0.0, 0.0), 1]},
            o='deb_f_deviation', o_kws={'max_Nsims': 20, 'operations': {'mean': True, 'abs': True}}),
    **batch('rovers_sitters',
            ss={'substrate_quality': [(0.5, 0.8), 2], 'hours_as_larva': [(0, 100), 2]},

             batch_methods=batch_methods(run='deb', post='null', final='deb')),
    **batch('imitation',
            ss={'activation_noise': [(0.0, 0.8), 3], 'base_activation': [(15.0, 25.0), 3]},
            o='sample_fit', o_kws={'threshold': 1.0, 'max_Nsims': 20, 'operations': {'mean': False, 'abs': False}},
            batch_methods=batch_methods(run='exp_fit', post='default', final='null'), )

}


def fit_tortuosity_batch(d, model='imitation', exp='dish', idx=0):
    conf = null_dict('batch_conf', exp=None,
                           exp_kws={'enrichment': enrichment_dict(types=['tortuosity'], bouts=[])},
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
