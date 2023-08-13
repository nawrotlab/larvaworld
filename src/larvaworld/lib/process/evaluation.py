import numpy as np
import pandas as pd
import param
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from larvaworld.lib import reg, aux
from larvaworld.lib.param import NestedConf


def eval_end_fast(ee, e_data, e_sym, mode='pooled'):
    E = {}
    for p, sym in e_sym.items():
        e_vs = e_data[p]
        E[sym] = None
        if p in ee.columns:
            if mode == '1:1':
                E[sym] = ((e_vs - ee[p]) ** 2).mean() ** .5
            elif mode == 'pooled':
                E[sym] = ks_2samp(e_vs.values, ee[p].values)[0]
    return E

def eval_distro_fast(ss, s_data, s_sym, mode='pooled', min_size=20):
    if mode == '1:1':
        E = {}
        for p, sym in s_sym.items():
            if p in ss.columns:
                pps = []
                for id in s_data.index:
                    sp, ssp = s_data[p].loc[id].values, ss[p].xs(id, level="AgentID").dropna().values
                    if sp.shape[0] > min_size and ssp.shape[0] > min_size:
                        pps.append(ks_2samp(sp, ssp)[0])

                E[sym] = np.median(pps)
    elif mode == 'pooled':
        E = {}
        for p, sym in s_sym.items():
            if p in ss.columns:
                spp, sspp = s_data[p].values, ss[p].dropna().values
                if spp.shape[0] > min_size and sspp.shape[0] > min_size:
                    E[sym] = ks_2samp(spp, sspp)[0]
    elif mode == '1:pooled':
        ids=ss.index.unique('AgentID').values
        E = {id:{} for id in ids}
        for id in ids:
            sss = ss.xs(id, level="AgentID")
            for p, sym in s_sym.items():
                if p in ss.columns:
                    sp, ssp = s_data[p].dropna().values, sss[p].dropna().values
                    if sp.shape[0] > min_size and ssp.shape[0] > min_size:
                        E[id][sym] = ks_2samp(sp, ssp)[0]
    else:
        raise

    return E

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
    E = aux.AttrDict({'end': pd.DataFrame.from_records(GEend).T,'step': pd.DataFrame.from_records(GEdistro).T})
    E.end.index.name = 'metric'
    E.step.index.name = 'metric'
    return E


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

def eval_RSS(rss,rss_target,rss_sym, mode='1:pooled') :
    assert mode == '1:pooled'
    RSS_dic={}
    for id, rrss in rss.items():
        RSS_dic[id] = {}
        for p, sym in rss_sym.items():
            if p in rrss.keys():
                RSS_dic[id][sym] = RSS(rrss[p], rss_target[p])
    return RSS_dic



def col_df(shorts, groups):
    from matplotlib import cm
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

    df['cols'] = df.apply(lambda row: [(row['group'], p) for p in row['symbols']], axis=1)
    df['par_colors'] = df.apply(
        lambda row: [cm.get_cmap(row['group_color'])(i) for i in np.linspace(0.4, 0.7, len(row['pars']))],
        axis=1)
    df.set_index('group', inplace=True)
    return df

# def arrange_evaluation(d, evaluation_metrics=None):
#     if evaluation_metrics is None:
#         evaluation_metrics = {
#             'angular kinematics': ['run_fov_mu', 'pau_fov_mu', 'b', 'fov', 'foa', 'rov', 'roa', 'tur_fou'],
#             'spatial displacement': ['cum_d', 'run_d', 'str_c_l', 'v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a',
#                                      'dsp_0_40_max', 'str_N', 'tor5', 'tor20'],
#             'temporal dynamics': ['fsv', 'ffov', 'run_t', 'pau_t', 'run_tr', 'pau_tr'],
#             'stride cycle': ['str_d_mu', 'str_d_std', 'str_sv_mu', 'str_fov_mu', 'str_fov_std'],
#             #'epochs': ['run_t', 'pau_t'],
#             #'tortuosity': ['tor5', 'tor20']
#         }
#
#
#     if not hasattr(d, 'step_data'):
#         d.load(h5_ks=['epochs','base_spatial','angular','dspNtor'])
#     s,e=d.step_data,d.endpoint_data
#     all_ks=aux.SuperList(evaluation_metrics.values()).flatten.unique
#     all_ps = aux.SuperList(reg.getPar(all_ks))
#     Eps = all_ps.existing(e)
#     Dps = all_ps.existing(s)
#     Dps=Dps.nonexisting(Eps)
#     Eks = reg.getPar(p=Eps, to_return='k')
#     Dks = reg.getPar(p=Dps, to_return='k')
#     target_data = aux.AttrDict({'step': {p:s[p].dropna() for p in Dps}, 'end': {p:e[p] for p in Eps}})
#
#     dic = aux.AttrDict({'end': {'shorts': [], 'groups': []}, 'step': {'shorts': [], 'groups': []}})
#     for g, shs in evaluation_metrics.items():
#         Eshorts, Dshorts = aux.existing_cols(shs,Eks), aux.existing_cols(shs,Dks)
#
#         if len(Eshorts) > 0:
#             dic.end.shorts.append(Eshorts)
#             dic.end.groups.append(g)
#         if len(Dshorts) > 0:
#             dic.step.shorts.append(Dshorts)
#             dic.step.groups.append(g)
#     ev = aux.AttrDict({k: col_df(**v) for k, v in dic.items()})
#
#     return ev, target_data

class Evaluation(NestedConf) :
    refID = reg.conf.Ref.confID_selector()
    eval_metrics = param.Dict(default=aux.AttrDict({
        'angular kinematics': ['run_fov_mu', 'pau_fov_mu', 'b', 'fov', 'foa', 'rov', 'roa', 'tur_fou'],
        'spatial displacement': ['cum_d', 'run_d', 'str_c_l', 'v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a',
                                 'dsp_0_40_max', 'str_N', 'tor5', 'tor20'],
        'temporal dynamics': ['fsv', 'ffov', 'run_t', 'pau_t', 'run_tr', 'pau_tr'],
        'stride cycle': ['str_d_mu', 'str_d_std', 'str_sv_mu', 'str_fov_mu', 'str_fov_std', 'str_N'],
        'epochs': ['run_t', 'pau_t'],
        'tortuosity': ['tor5', 'tor20']}),
        doc='Evaluation metrics to use')
    cycle_curve_metrics = param.List()


    def __init__(self, dataset=None,**kwargs):
        super().__init__(**kwargs)
        self.target = reg.conf.Ref.retrieve_dataset(dataset=dataset, id=self.refID)
        print(self.eval_metrics)

        if not hasattr(self.target, 'step_data'):
            self.target.load(h5_ks=['epochs', 'base_spatial', 'angular', 'dspNtor'])

        self.build()

    def build(self):
        if len(self.eval_metrics) > 0:
            self.evaluation, self.target_data = self.arrange_evaluation()
        else:
            self.s_shorts = []
        if len(self.cycle_curve_metrics)>0:
            if not hasattr(self.target.config,'pooled_cycle_curves') :
                from larvaworld.lib.process.annotation import compute_interference
                s, e, c = self.target.data
                self.target.config.pooled_cycle_curves = compute_interference(s, e, c=c, d=self.target, chunk_dicts=self.target.read('chunk_dicts'))

            cycle_dict = {'sv': 'abs', 'fov': 'norm', 'rov': 'norm', 'foa': 'norm', 'b': 'norm'}
            self.cycle_modes = {sh: cycle_dict[sh] for sh in self.cycle_curve_metrics}
            self.cycle_curve_target = aux.AttrDict({sh: np.array(self.target.config.pooled_cycle_curves[sh][mod]) for sh, mod in self.cycle_modes.items()})
            self.rss_sym = {sh: sh for sh in self.cycle_curve_metrics}





    def arrange_evaluation(self):

        s, e = self.target.step_data, self.target.endpoint_data
        all_ks = aux.SuperList(self.eval_metrics.values()).flatten.unique
        all_ps = aux.SuperList(reg.getPar(all_ks[:]))
        Eps = all_ps.existing(e)
        Dps = all_ps.existing(s)
        Dps = Dps.nonexisting(Eps)
        Eks = reg.getPar(p=Eps[:], to_return='k')
        Dks = reg.getPar(p=Dps[:], to_return='k')
        target_data = aux.AttrDict({'step': {p: s[p].dropna() for p in Dps}, 'end': {p: e[p] for p in Eps}})

        dic = aux.AttrDict({'end': {'shorts': [], 'groups': []}, 'step': {'shorts': [], 'groups': []}})
        for g, shs in self.eval_metrics.items():
            Eshorts, Dshorts = aux.existing_cols(shs, Eks), aux.existing_cols(shs, Dks)

            if len(Eshorts) > 0:
                dic.end.shorts.append(Eshorts)
                dic.end.groups.append(g)
            if len(Dshorts) > 0:
                dic.step.shorts.append(Dshorts)
                dic.step.groups.append(g)
        ev = aux.AttrDict({k: col_df(**v) for k, v in dic.items()})
        self.s_pars = aux.flatten_list(ev['step']['pars'].values.tolist())
        self.s_shorts = aux.flatten_list(ev['step']['shorts'].values.tolist())
        self.s_symbols = aux.flatten_list(ev['step']['symbols'].values.tolist())
        self.e_pars = aux.flatten_list(ev['end']['pars'].values.tolist())
        self.e_symbols = aux.flatten_list(ev['end']['symbols'].values.tolist())
        self.eval_symbols = aux.AttrDict(
            {'step': dict(zip(self.s_pars, self.s_symbols)), 'end': dict(zip(self.e_pars, self.e_symbols))})
        return ev, target_data

    @property
    def func_eval_metric_solo(self):
        def func(ss):
            return aux.AttrDict({'KS': {sym: ks_2samp(self.target_data.step[p].values, ss[p].dropna().values)[0] for
                                        p, sym in self.eval_symbols.step.items()}})

        return func

    @property
    def func_eval_metric_multi(self):
        def gfunc(s):
            return aux.AttrDict(
                {'KS': eval_distro_fast(s, self.target_data.step, self.eval_symbols.step, mode='1:pooled', min_size=10)})

        return gfunc

    @property
    def func_cycle_curve_solo(self):
        def func(ss):
            from larvaworld.lib.process.annotation import cycle_curve_dict
            c0 = cycle_curve_dict(s=ss, dt=self.target.config.dt, shs=self.cycle_curve_metrics)
            eval_curves = aux.AttrDict(({sh: c0[sh][mode] for sh, mode in self.cycle_modes.items()}))
            return aux.AttrDict(
                {'RSS': {sh: RSS(ref_curve, eval_curves[sh]) for sh, ref_curve in self.cycle_curve_target.items()}})

        return func

    @property
    def func_cycle_curve_multi(self):
        def gfunc(s):
            from larvaworld.lib.process.annotation import cycle_curve_dict_multi

            rss0 = cycle_curve_dict_multi(s=s, dt=self.target.config.dt, shs=self.cycle_curve_metrics)
            rss = aux.AttrDict(
                {id: {sh: dic[sh][mod] for sh, mod in self.cycle_modes.items()} for id, dic in rss0.items()})
            return aux.AttrDict({'RSS': eval_RSS(rss, self.cycle_curve_target, self.rss_sym, mode='1:pooled')})
        return gfunc

    @property
    def fit_func_multi(self):
        def fit_func(s):
            fit_dicts = aux.AttrDict()
            if len(self.cycle_curve_metrics) > 0:
                fit_dicts.update(self.func_cycle_curve_multi(s))
            if len(self.eval_metrics) > 0:
                fit_dicts.update(self.func_eval_metric_multi(s))
            return fit_dicts
        return fit_func

    @property
    def fit_func_solo(self):
        def fit_func(ss):
            fit_dicts = aux.AttrDict()
            if len(self.cycle_curve_metrics) > 0:
                fit_dicts.update(self.func_cycle_curve_solo(ss))
            if len(self.eval_metrics) > 0:
                fit_dicts.update(self.func_eval_metric_solo(ss))
            return fit_dicts

        return fit_func


    def eval_datasets(self,ds,mode, min_size=20):
        return eval_fast(datasets=ds, data=self.target_data, symbols=self.eval_symbols, mode=mode, min_size=min_size)


class DataEvaluation(Evaluation) :


    norm_modes = param.ListSelector(default=['raw', 'minmax'], objects=['raw', 'minmax', 'std'],
                                    doc='Normalization modes to use')
    eval_modes = param.ListSelector(default=['pooled'], objects=['pooled', '1:1', '1:pooled'],
                                    doc='Evaluation modes to use')


    def __init__(self, **kwargs):
        super().__init__(**kwargs)




        self.error_dicts = aux.AttrDict()







    def norm_error_dict(self, error_dict, mode='raw'):
        if mode == 'raw':
            return error_dict
        elif mode == 'minmax':
            return aux.AttrDict({k : pd.DataFrame(MinMaxScaler().fit(df).transform(df), index=df.index, columns=df.columns) for k, df in error_dict.items()})
        elif mode == 'std':
            return aux.AttrDict({k : pd.DataFrame(StandardScaler().fit(df).transform(df), index=df.index, columns=df.columns) for k, df in error_dict.items()})





