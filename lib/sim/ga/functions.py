import numpy as np
from scipy.stats import ks_2samp

from lib.aux import dictsNlists as dNl, naming as nam, colsNstr as cNs
from lib.registry import reg
from lib.sim.eval.eval_aux import RSS, arrange_evaluation

def GA_optimization(fitness_target_refID, fitness_target_kws):
    d = reg.loadRef(fitness_target_refID)
    fit_dic0 = build_fitness(fitness_target_kws, d)

    func_dict = fit_dic0['func_global_dict']

    def func(s):
        fit_dicts = {}
        for k, kfunc in func_dict.items():
            fit_dicts.update(kfunc(s))
        return fit_dicts

    return dNl.NestDict({'func': func, 'keys': fit_dic0['keys'], 'func_arg': 's'})


def build_fitness(dic, refDataset):
    d = refDataset
    c = d.config
    func_global_dict = dNl.NestDict()
    func_solo_dict = dNl.NestDict()
    keys = []
    for k, vs in dic.items():
        if k == 'cycle_curves':
            cycle_dict = {'sv': 'abs', 'fov': 'norm', 'rov': 'norm', 'foa': 'norm', 'b': 'norm'}
            cycle_ks = vs
            cycle_modes = {sh: cycle_dict[sh] for sh in cycle_ks}
            T = d.config.pooled_cycle_curves
            target = dNl.NestDict({sh: np.array(T[sh][mod]) for sh, mod in cycle_modes.items()})
            rss_sym = {sh: sh for sh in vs}
            keys += cycle_ks

            def func(ss):
                from lib.process.aux import cycle_curve_dict
                c0 = cycle_curve_dict(s=ss, dt=d.config.dt, shs=vs)
                eval_curves = dNl.NestDict(({sh: c0[sh][mode] for sh, mode in cycle_modes.items()}))
                return dNl.NestDict(
                    {'RSS': {sh: RSS(ref_curve, eval_curves[sh]) for sh, ref_curve in target.items()}})

            func_solo_dict[k] = func

            def gfunc(s):
                from lib.process.aux import cycle_curve_dict_multi
                from lib.sim.eval.eval_aux import eval_RSS

                rss0 = cycle_curve_dict_multi(s=s, dt=d.config.dt, shs=cycle_ks)
                rss = dNl.NestDict(
                    {id: {sh: dic[sh][mod] for sh, mod in cycle_modes.items()} for id, dic in rss0.items()})
                return dNl.NestDict({'RSS': eval_RSS(rss, target, rss_sym, mode='1:pooled')})

            func_global_dict[k] = gfunc

        if k == 'eval_metrics':

            evaluation, target_data = arrange_evaluation(d, evaluation_metrics=vs)
            s_shorts = dNl.flatten_list(evaluation['step']['shorts'].values.tolist())
            s_pars = dNl.flatten_list(evaluation['step']['pars'].values.tolist())
            s_symbols = dNl.NestDict(dict(zip(s_pars, s_shorts)))
            keys += s_shorts

            def func(ss):
                return dNl.NestDict(
                    {'KS': {sym: ks_2samp(target_data.step[p].values, ss[p].dropna().values)[0] for p, sym in
                            s_symbols.items()}})

            func_solo_dict[k] = func

            def gfunc(s):
                from lib.sim.eval.eval_aux import eval_distro_fast
                return dNl.NestDict(
                    {'KS': eval_distro_fast(s, target_data.step, s_symbols, mode='1:pooled', min_size=10)})

            func_global_dict[k] = gfunc

    keys = dNl.unique_list(keys)
    return dNl.NestDict({'func_global_dict': func_global_dict, 'func_solo_dict': func_solo_dict, 'keys': keys})




