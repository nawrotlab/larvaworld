from lib import reg

def batch(exp, proc=[], ss=None, ssbool=None, o=None, o_kws={}, bm={}, as_entry=True, **kwargs):
    if bm is None:
        bm_kws = {}
    elif bm == 'PI':
        bm_kws = {'exec': 'odor_preference', 'post': 'null', 'final': 'odor_preference'}
    elif bm == 'DEB':
        bm_kws = {'exec': 'deb', 'post': 'null', 'final': 'deb'}
    else:
        bm_kws = bm
    if ss is not None:
        ss = {p: reg.get_null('space_search_par', range=r, Ngrid=N) for p, (r, N) in ss.items()}
    else:
        ss = {}
    if ssbool is not None:
        ssbool = {p: {'values': [True, False]} for p in ssbool}
    else:
        ssbool = {}
    ss0 = {**ss, **ssbool}
    if len(ss0) == 0:
        ss0 = None

    enr=reg.get_null('enrichment',processing=reg.get_null('processing', **{pr : True for pr in proc}))
    conf = reg.get_null('Batch',
                         exp=exp,
                         exp_kws={'enrichment': enr, 'experiment': exp},
                         optimization=reg.get_null("optimization", fit_par=o, **o_kws) if o is not None else None,
                         space_search=ss0,
                         batch_methods=reg.get_null('batch_methods', **bm_kws),
                         **kwargs)
    # print(conf)
    if as_entry:
        return {exp: conf}
    else:
        return conf

@reg.funcs.stored_conf("Batch")
def Batch_dict():
    d = {
        **batch('chemotaxis',
                ss={'Odor.mean': [[300.0, 1300.0], 3], 'decay_coef': [[0.1, 0.5], 3]},
                o='final_dst_to_Source',
                proc=['angular', 'spatial', 'source']),
        **batch('chemorbit',
                ss={'Odor.mean': [[300.0, 1300.0], 3], 'decay_coef': [[0.1, 0.5], 3]},
                o='final_dst_to_Source',
                proc=['angular', 'spatial', 'source']),
        **batch('PItest_off',
                ss={'odor_dict.CS.mean': [[-100.0, 100.0], 6], 'odor_dict.UCS.mean': [[-100.0, 100.0], 6]},
                bm='PI',
                proc=['PI']),
        **batch('PItrain_mini',
                ss={'input_noise': [[0.0, 0.4], 2], 'decay_coef': [[0.1, 0.5], 2]},
                bm='PI',
                proc=['PI']),
        **batch('PItrain',
                ss={'input_noise': [[0.0, 0.4], 2], 'decay_coef': [[0.1, 0.5], 2]},
                bm='PI',
                proc=['PI']),
        **batch('patchy_food',
                ss={'EEB': [[0.0, 1.0], 3], 'initial_freq': [[1.5, 2.5], 3]},
                o='ingested_food_volume'),
        **batch('food_grid',
                ss={'EEB': [[0.0, 1.0], 6], 'EEB_decay': [[0.1, 2.0], 6]},
                o='ingested_food_volume'),
        **batch('growth',
                ss={'EEB': [[0.5, 0.8], 8], 'hunger_gain': [[0.0, 0.0], 1]},
                o='deb_f_deviation', o_kws={'max_Nsims': 20, 'operations': {'mean': True, 'abs': True, 'std': False}}),
        **batch('imitation',
                ss={'activation_noise': [[0.0, 0.8], 3], 'base_activation': [[15.0, 25.0], 3]},
                o='sample_fit',
                o_kws={'threshold': 1.0, 'max_Nsims': 20, 'operations': {'mean': False, 'abs': False, 'std': False}},
                bm={'exec': 'exp_fit'}),
        **batch('tactile_detection',
                ss={'initial_gain': [[25.0, 75.0], 10], 'decay_coef': [[0.01, 0.5], 4]},
                o='cum_food_detected', o_kws={'threshold': 100000.0, 'max_Nsims': 600, 'minimize': False, 'Nbest': 8,
                                              'operations': {'mean': True, 'abs': False, 'std': False}}),
        **batch('anemotaxis',
                ss={f'windsensor_params.weights.{m1}_{m2}': [[-20.0, 20.0], 3] for m1, m2 in
                    zip(['bend', 'hunch'], ['ang', 'lin'])},
                o='anemotaxis', o_kws={'threshold': 1000.0, 'max_Nsims': 100, 'minimize': False, 'Nbest': 8,
                                       'operations': {'mean': True, 'abs': False}}, proc=['wind'])
    }
    return d


def fit_tortuosity_batch(sample, model='explorer', exp='dish', idx=0, **kwargs):
    from lib.sim.imitation import imitation_exp
    conf = batch(exp=None,
                 ss={'activation_noise': [[0.0, 2.0], 3], 'base_activation': [[15.0, 25.0], 3]},
                 o='tortuosity_20_mean', o_kws={'max_Nsims': 120, 'operations': {'mean': True}},
                 en=reg.get_null('enrichment',processing=reg.get_null('processing', tortuosity=True)),
                 # en=preg.enr_dict(proc=['tortuosity']),
                 as_entry=False
                 )
    conf['exp'] = imitation_exp(sample, model=model, exp=exp, idx=idx, **kwargs)
    conf['batch_id'] = f'imitation_batchrun_{idx}'
    conf['batch_type'] = 'imitation'
    return conf


def fit_global_batch(sample, model='explorer', exp='dish', idx=0, **kwargs):
    from lib.sim.imitation import imitation_exp
    conf = batch(exp=None,
                 ss={
                     'turner_params.initial_amp': [[25.0, 50.0], 4],
                     'turner_params.noise': [[0.0, 4.0], 4],
                     'turner_params.activation_noise': [[0.0, 4.0], 4],
                 },
                 o='sample_fit',
                 o_kws={'threshold': 0.1, 'Nbest': 8, 'max_Nsims': 140, 'operations': {'mean': False, 'abs': False}},
                 bm={'exec': 'exp_fit'},
                 en=reg.get_null('enrichment', processing=reg.get_null('processing', angular=True)),
                 as_entry=False
                 )
    conf['exp'] = imitation_exp(sample, model=model, exp=exp, idx=idx, **kwargs)
    conf['batch_id'] = f'imitation_batchrun_{idx}'
    conf['batch_type'] = 'imitation'
    return conf


def run_fit_global_batch(sample, **kwargs):
    from lib.sim.exec_run import Exec
    from lib.sim.exp_fitter import ExpFitter
    conf = fit_global_batch(sample=sample, **kwargs)
    conf['proc_kws']['exp_fitter'] = ExpFitter(sample, valid_fields=['angular motion'])
    # conf['proc_kws']['exp_fitter'] = ExpFitter(sample, valid_fields=['angular motion', 'reorientation', 'spatial motion'])
    k = Exec('batch', conf)
    return k.exec_run()