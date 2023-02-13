import itertools
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from matplotlib import cm, colors

from larvaworld.lib import reg, aux

def eval_end_fast(ee, e_data, e_sym, mode='pooled'):
    Eend = {}
    for p, sym in e_sym.items():
        e_vs = e_data[p]
        # sym=e_sym[p]
        Eend[sym] = None
        if p in ee.columns:
            if mode == '1:1':
                Eend[sym] = ((e_vs - ee[p]) ** 2).mean() ** .5
            elif mode == 'pooled':
                Eend[sym] = ks_2samp(e_vs.values, ee[p].values)[0]
    return Eend

def eval_distro_fast(ss, s_data, s_sym, mode='pooled', min_size=20):
    if mode == '1:1':
        Edistro = {}
        for p, sym in s_sym.items():
            if p in ss.columns:
                # s_vs = s_data[p]
                pps = []
                for id in s_data.index:
                    sp, ssp = s_data[p].loc[id].values, ss[p].xs(id, level="AgentID").dropna().values
                    if sp.shape[0] > min_size and ssp.shape[0] > min_size:
                        pps.append(ks_2samp(sp, ssp)[0])

                Edistro[sym] = np.median(pps)
    elif mode == 'pooled':
        Edistro = {}
        for p, sym in s_sym.items():
            if p in ss.columns:
                spp, sspp = s_data[p].values, ss[p].dropna().values
                if spp.shape[0] > min_size and sspp.shape[0] > min_size:
                    Edistro[sym] = ks_2samp(spp, sspp)[0]
    elif mode == '1:pooled':
        ids=ss.index.unique('AgentID').values
        Edistro = {id:{} for id in ids}
        for id in ids:
            sss = ss.xs(id, level="AgentID").dropna()
            for p, sym in s_sym.items():
                if p in ss.columns:
                    sp, ssp = s_data[p].values, sss[p].values
                    if sp.shape[0] > min_size and ssp.shape[0] > min_size:
                        Edistro[id][sym] = ks_2samp(sp, ssp)[0]

    return Edistro

def eval_fast(datasets, data, symbols, mode='pooled', min_size=20):
    GEend = {d.id: eval_end_fast(d.endpoint_data, data.end, symbols.end, mode=mode) for d in datasets}
    GEdistro = {d.id: eval_distro_fast(d.step_data, data.step, symbols.step, mode=mode, min_size=min_size) for d
                in datasets}
    # if mode == '1:1':
    #     labels = ['RSS error', r'median 1:1 distribution KS$_{D}$']
    # elif mode == 'pooled':
    #     labels = ['pooled endpoint KS$_{D}$', 'pooled distribution KS$_{D}$']
    # elif mode == '1:pooled':
    #     labels = ['individual endpoint KS$_{D}$', 'individual distribution KS$_{D}$']
    error_dict = {'end': pd.DataFrame.from_records(GEend).T,
                  'step': pd.DataFrame.from_records(GEdistro).T}
    error_dict['end'].index.name = 'metric'
    error_dict['step'].index.name = 'metric'
    return error_dict


def eval_RSS(rss,rss_target,rss_sym, mode='1:pooled') :
    if mode == '1:pooled':
        RSS_dic={}
        for id, rrss in rss.items():
            RSS_dic[id] = {}
            for p, sym in rss_sym.items():
                if p in rrss.keys():
                    RSS_dic[id][sym] = RSS(rrss[p], rss_target[p])
    return RSS_dic

def col_df(shorts, groups):

    group_col_dic = {
        'angular kinematics': 'Blues',
        'spatial displacement': 'Greens',
        'temporal dynamics': 'Reds',
        'dispersal': 'Purples',
        'tortuosity': 'Purples',
        'epochs': 'Oranges',
        'stride cycle': 'Oranges',

    }
    group_label_dic = {
        'angular kinematics': r'$\bf{angular}$ $\bf{kinematics}$',
        'spatial displacement': r'$\bf{spatial}$ $\bf{displacement}$',
        'temporal dynamics': r'$\bf{temporal}$ $\bf{dynamics}$',
        'dispersal': r'$\bf{dispersal}$',
        'tortuosity': r'$\bf{tortuosity}$',
        'epochs': r'$\bf{epochs}$',
        'stride cycle': r'$\bf{stride}$ $\bf{cycle}$',

    }
    df = pd.DataFrame(
        {'group': groups,
         'group_label': [group_label_dic[g] for g in groups],
         'shorts': shorts,
         'pars': [reg.getPar(sh) for sh in shorts],
         'symbols': [reg.getPar(sh, to_return='l') for sh in shorts],
         'group_color': [group_col_dic[g] for g in groups]
         })

    # print(shorts)
    # print(groups)

    df['cols'] = df.apply(lambda row: [(row['group'], p) for p in row['symbols']], axis=1)
    df['par_colors'] = df.apply(
        lambda row: [cm.get_cmap(row['group_color'])(i) for i in np.linspace(0.4, 0.7, len(row['pars']))], axis=1)
    df.set_index('group', inplace=True)
    return df

def arrange_evaluation(d, evaluation_metrics=None):
    if evaluation_metrics is None:
        evaluation_metrics = {
            'angular kinematics': ['run_fov_mu', 'pau_fov_mu', 'b', 'fov', 'foa', 'rov', 'roa', 'tur_fou'],
            'spatial displacement': ['cum_d', 'run_d', 'str_c_l', 'v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a',
                                     'dsp_0_40_max', 'dsp_0_60_max', 'str_N', 'tor5', 'tor20'],
            'temporal dynamics': ['fsv', 'ffov', 'run_t', 'pau_t', 'run_tr', 'pau_tr'],
            'stride cycle': ['str_d_mu', 'str_d_std', 'str_sv_mu', 'str_fov_mu', 'str_fov_std', 'str_N'],
            'epochs': ['run_t', 'pau_t'],
            'tortuosity': ['tor5', 'tor20']
        }



    Edata, Ddata = {}, {}
    dic = aux.AttrDict({'end': {'shorts': [], 'groups': []}, 'step': {'shorts': [], 'groups': []}})
    for g, shs in evaluation_metrics.items():
        Eshorts, Dshorts = [], []
        ps = reg.getPar(shs)
        for sh, p in zip(shs, ps):
            try:
                data = d.read(key='end')[p]
                if data is not None:
                    Edata[p] = data
                    Eshorts.append(sh)
            except:
                data = d.read(p, 'distro')
                if data is not None:
                    Ddata[p] = data.dropna()
                    Dshorts.append(sh)

        # Eshorts = [sh for sh, p in zip(shs, ps) if p in e.columns]
        # Dshorts = [sh for sh, p in zip(shs, ps) if p in s.columns]
        # Dshorts = [sh for sh in Dshorts if sh not in Eshorts]
        if len(Eshorts) > 0:
            dic.end.shorts.append(Eshorts)
            dic.end.groups.append(g)
        if len(Dshorts) > 0:
            dic.step.shorts.append(Dshorts)
            dic.step.groups.append(g)
    target_data = aux.AttrDict({'step': Ddata, 'end': Edata})


    ev = {k: col_df(**v) for k, v in dic.items()}

    return ev, target_data


def torsNdsps(pars):
    dsp= reg.getPar('dsp')
    tor_durs = [int(ii[len('tortuosity') + 1:]) for ii in pars if ii.startswith('tortuosity')]
    tor_durs = np.unique(tor_durs)
    tor_shorts = [f'tor{ii}' for ii in tor_durs]

    dsp_temp = [ii[len(dsp) + 1:].split('_') for ii in pars if ii.startswith(f'{dsp}_')]
    dsp_starts = np.unique([int(ii[0]) for ii in dsp_temp]).tolist()
    dsp_stops = np.unique([int(ii[1]) for ii in dsp_temp]).tolist()
    dsp_shorts0 = [f'dsp_{s0}_{s1}' for s0, s1 in itertools.product(dsp_starts, dsp_stops)]
    dsp_shorts = aux.flatten_list([[f'{ii}_max', f'{ii}_mu', f'{ii}_fin'] for ii in dsp_shorts0])
    return tor_durs, dsp_starts, dsp_stops


def RSS(vs0, vs):
    er = (vs - vs0)

    r = np.abs(np.max(vs0) - np.min(vs0))

    ee = (er / r) ** 2

    MSE = np.mean(np.sum(ee))
    return np.round(np.sqrt(MSE), 2)


def RSS_dic(dd, d):
    f = d.config.pooled_cycle_curves
    ff = dd.config.pooled_cycle_curves

    def RSS0(ff, f, sh, mode):
        vs0 = np.array(f[sh][mode])
        vs = np.array(ff[sh][mode])
        return RSS(vs0, vs)

    def RSS1(ff, f, sh):
        dic = {}
        for mode in f[sh].keys():
            dic[mode] = RSS0(ff, f, sh, mode)
        return dic

    dic = {}
    for sh in f.keys():
        dic[sh] = RSS1(ff, f, sh)

    stat = np.round(np.mean([dic[sh]['norm'] for sh in f.keys() if sh != 'sv']), 2)
    dd.config.pooled_cycle_curves_errors = aux.AttrDict({'dict': dic, 'stat': stat})
    return stat


def GA_optimization(fitness_target_refID, fitness_target_kws):
    d = reg.loadRef(fitness_target_refID)
    fit_dic0 = build_fitness(fitness_target_kws, d)

    func_dict = fit_dic0['func_global_dict']

    def func(s):
        fit_dicts = {}
        for k, kfunc in func_dict.items():
            fit_dicts.update(kfunc(s))
        return fit_dicts

    return aux.AttrDict({'func': func, 'keys': fit_dic0['keys'], 'func_arg': 's'})


def build_fitness(dic, refDataset):
    d = refDataset
    c = d.config
    func_global_dict, func_solo_dict = {},{}
    keys = []
    for k, vs in dic.items():
        if k == 'cycle_curves':
            cycle_dict = {'sv': 'abs', 'fov': 'norm', 'rov': 'norm', 'foa': 'norm', 'b': 'norm'}
            cycle_ks = vs
            cycle_modes = {sh: cycle_dict[sh] for sh in cycle_ks}
            target = aux.AttrDict({sh: np.array(c.pooled_cycle_curves[sh][mod]) for sh, mod in cycle_modes.items()})
            rss_sym = {sh: sh for sh in vs}
            keys += cycle_ks

            def func(ss):
                from larvaworld.lib.process.annotation import cycle_curve_dict
                c0 = cycle_curve_dict(s=ss, dt=c.dt, shs=vs)
                eval_curves = aux.AttrDict(({sh: c0[sh][mode] for sh, mode in cycle_modes.items()}))
                return aux.AttrDict(
                    {'RSS': {sh: RSS(ref_curve, eval_curves[sh]) for sh, ref_curve in target.items()}})

            func_solo_dict[k] = func

            def gfunc(s):
                from larvaworld.lib.process.annotation import cycle_curve_dict_multi

                rss0 = cycle_curve_dict_multi(s=s, dt=c.dt, shs=cycle_ks)
                rss = aux.AttrDict(
                    {id: {sh: dic[sh][mod] for sh, mod in cycle_modes.items()} for id, dic in rss0.items()})
                return aux.AttrDict({'RSS': eval_RSS(rss, target, rss_sym, mode='1:pooled')})

            func_global_dict[k] = gfunc

        if k == 'eval_metrics':

            evaluation, target_data = arrange_evaluation(d, evaluation_metrics=vs)
            s_shorts = aux.flatten_list(evaluation['step']['shorts'].values.tolist())
            s_pars = aux.flatten_list(evaluation['step']['pars'].values.tolist())
            s_symbols = aux.AttrDict(dict(zip(s_pars, s_shorts)))
            keys += s_shorts

            def func(ss):
                return aux.AttrDict(
                    {'KS': {sym: ks_2samp(target_data.step[p].values, ss[p].dropna().values)[0] for p, sym in
                            s_symbols.items()}})

            func_solo_dict[k] = func

            def gfunc(s):
                return aux.AttrDict(
                    {'KS': eval_distro_fast(s, target_data.step, s_symbols, mode='1:pooled', min_size=10)})

            func_global_dict[k] = gfunc

    keys = aux.unique_list(keys)
    return aux.AttrDict({'func_global_dict': func_global_dict, 'func_solo_dict': func_solo_dict, 'keys': keys})
