import copy
import itertools
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from lib.aux import naming as nam, dictsNlists as dNl, colsNstr as cNs
from lib.registry import reg


dst, v, sv, acc, sa, fou, rou, fo, ro, b, fov, rov, bv, foa, roa, ba, x, y, l, dsp, dsp_0_40, dsp_0_40_mu, dsp_0_40_max, str_fov_mu, run_fov_mu, pau_fov_mu, run_foa_mu, pau_foa_mu, str_fov_std, pau_fov_std, str_sd_mu, str_sd_std, str_d_mu, str_d_std, str_sv_mu, pau_sv_mu, str_v_mu, run_v_mu, run_sv_mu, pau_v_mu, str_tr, run_tr, pau_tr, Ltur_tr, Rtur_tr, Ltur_fou, Rtur_fou, run_t_min, cum_t, run_t, run_dst, pau_t = reg.getPar(
    ['d', 'v', 'sv', 'a', 'sa', 'fou', 'rou', 'fo', 'ro', 'b', 'fov', 'rov', 'bv', 'foa', 'roa', 'ba', 'x', 'y', 'l',
     "dsp", "dsp_0_40", "dsp_0_40_mu", "dsp_0_40_max", 'str_fov_mu', 'run_fov_mu', 'pau_fov_mu', 'run_foa_mu',
     'pau_foa_mu', 'str_fov_std', 'pau_fov_std', 'str_sd_mu', 'str_sd_std', 'str_d_mu', 'str_d_std', 'str_sv_mu',
     'pau_sv_mu', 'str_v_mu', 'run_v_mu', 'run_sv_mu', 'pau_v_mu', 'str_tr', 'run_tr', 'pau_tr', 'Ltur_tr', 'Rtur_tr',
     'Ltur_fou', 'Rtur_fou', 'run_t_min', 'cum_t', 'run_t', 'run_d', 'pau_t'])


def eval_endpoint(ee, e, e_shorts=None, e_pars=None, e_labs=None, mode='pooled'):
    if e_pars is None:
        e_pars = reg.getPar(e_shorts)
    if e_labs is None:
        e_labs = reg.getPar(d=e_pars, to_return=['lab'])
    Eend = {}
    for p, pl in zip(e_pars, e_labs):
        Eend[pl] = None
        if p in e.columns and p in ee.columns:
            if mode == '1:1':
                Eend[pl] = ((e[p] - ee[p]) ** 2).mean() ** .5
            elif mode == 'pooled':
                Eend[pl] = ks_2samp(e[p].values, ee[p].values)[0]
    return Eend


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


def eval_distro(ss, s, s_shorts=None, s_pars=None, s_labs=None, mode='pooled', min_size=20):
    if s_pars is None:
        s_pars = reg.getPar(s_shorts)
    if s_labs is None:
        s_labs = reg.getPar(d=s_pars, to_return=['lab'])

    Edistro = {}
    for p, pl in zip(s_pars, s_labs):
        Edistro[pl] = None
        if p in s.columns and p in ss.columns:
            if mode == '1:1':
                pps = []
                for id in s.index.unique('AgentID').values:
                    sp, ssp = s[p].xs(id, level="AgentID").dropna().values, ss[p].xs(id,
                                                                                     level="AgentID").dropna().values
                    if sp.shape[0] > min_size and ssp.shape[0] > min_size:
                        pps.append(ks_2samp(sp, ssp)[0])

                Edistro[pl] = np.median(pps)
            elif mode == 'pooled':
                spp, sspp = s[p].dropna().values, ss[p].dropna().values
                if spp.shape[0] > min_size and sspp.shape[0] > min_size:
                    Edistro[pl] = ks_2samp(spp, sspp)[0]
    return Edistro


def eval_distro_fast(ss, s_data, s_sym, mode='pooled', min_size=20):
    if mode == '1:1':
        Edistro = {}
        for p, sym in s_sym.items():
            if p in ss.columns:
                s_vs = s_data[p]
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


def eval_RSS(rss,rss_target,rss_sym, mode='1:pooled') :
    if mode == '1:pooled':
        RSS_dic={}
        for id, rrss in rss.items():
            RSS_dic[id] = {}
            for p, sym in rss_sym.items():
                if p in rrss.keys():
                    RSS_dic[id][sym] = RSS(rrss[p], rss_target[p])
    return RSS_dic


def eval_multi(datasets, s=None, e=None, s_shorts=None, e_shorts=None, mode='pooled', min_size=20):
    GEend, GEdistro = {}, {}
    if e is not None and e_shorts is not None:
        e_pars, e_labs = reg.getPar(e_shorts, to_return=['d', 'lab'])
        GEend = {d.id: eval_endpoint(d.endpoint_data, e, e_pars=e_pars, e_labs=e_labs, mode=mode) for d in datasets}
    if s is not None and s_shorts is not None:
        s_pars, s_labs = reg.getPar(s_shorts, to_return=['d', 'lab'])
        GEdistro = {d.id: eval_distro(d.step_data, s, s_pars=s_pars, s_labs=s_labs, mode=mode, min_size=min_size) for d
                    in datasets}
    if mode == '1:1':
        labels = ['RSS error', r'median 1:1 distribution KS$_{D}$']
    elif mode == 'pooled':
        labels = ['pooled endpoint KS$_{D}$', 'pooled distribution KS$_{D}$']
    error_dict = {'endpoint': pd.DataFrame.from_records(GEend).T,
                  'distro': pd.DataFrame.from_records(GEdistro).T}

    return error_dict


def eval_fast(datasets, data, symbols, mode='pooled', min_size=20):
    GEend, GEdistro = {}, {}
    GEend = {d.id: eval_end_fast(d.endpoint_data, data.end, symbols.end, mode=mode) for d in datasets}
    GEdistro = {d.id: eval_distro_fast(d.step_data, data.step, symbols.step, mode=mode, min_size=min_size) for d
                in datasets}
    if mode == '1:1':
        labels = ['RSS error', r'median 1:1 distribution KS$_{D}$']
    elif mode == 'pooled':
        labels = ['pooled endpoint KS$_{D}$', 'pooled distribution KS$_{D}$']
    elif mode == '1:pooled':
        labels = ['individual endpoint KS$_{D}$', 'individual distribution KS$_{D}$']
    error_dict = {'end': pd.DataFrame.from_records(GEend).T,
                  'step': pd.DataFrame.from_records(GEdistro).T}
    error_dict['end'].index.name = 'metric'
    error_dict['step'].index.name = 'metric'
    return error_dict




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
    dic = dNl.NestDict({'end': {'shorts': [], 'groups': []}, 'step': {'shorts': [], 'groups': []}})
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
    target_data = dNl.NestDict({'step': Ddata, 'end': Edata})


    ev = {k: cNs.col_df(**v) for k, v in dic.items()}

    return ev, target_data



def prepare_sim_dataset(e, c, id, color):
    cc = dNl.NestDict({
        'env_params': c.env_params,
        'bout_distros': c.bout_distros,
        'id': id,
        'dt': c.dt,
        'fr': c.fr,
        'N': c.N,
        'agent_ids': c.agent_ids,
        'Nticks': c.Nticks,
        'Npoints': 3,
        'color': color,
        'point': '',
        # 'parent_plot_dir': c.parent_plot_dir,

    })

    e_ps = ['length', nam.initial(x), nam.initial(y), nam.initial(fo)]
    ee = pd.DataFrame(e[e_ps].loc[cc.agent_ids], index=cc.agent_ids, columns=e_ps)
    ee[cum_t] = cc.Nticks * cc.dt
    return ee, cc


def prepare_dataset(d, Nids, id='experiment', color='black'):
    s0, e0, c0 = d.step_data, d.endpoint_data, d.config
    if Nids is None:
        ids = c0.agent_ids
        Nids = len(ids)
    else:
        ids = e0.nlargest(Nids, 'cum_dur').index.values
    s = copy.deepcopy(s0.loc[(slice(None), ids), slice(None)])
    e = copy.deepcopy(e0.loc[ids])

    c = dNl.NestDict(c0)
    c.id = id
    c.agent_ids = ids
    c.N = Nids
    c.color = color
    if "length" not in e.columns:
        e["length"] = np.ones(c.N) * 0.004
    return s, e, c


def prepare_validation_dataset(d, Nids):
    s, e, c = d.step_data, d.endpoint_data, d.config
    ids = e.nlargest(Nids, 'cum_dur').index.values
    ids2 = e.nlargest(2 * Nids, 'cum_dur').index.values[Nids:]
    s_val = s.loc[(slice(None), ids2), slice(None)]
    e_val = e.loc[ids2]
    e_val.rename(index=dict(zip(ids2, ids)), inplace=True)
    s_val.reset_index(level='Step', drop=False, inplace=True)
    s_val.rename(index=dict(zip(ids2, ids)), inplace=True)
    s_val.reset_index(drop=False, inplace=True)
    s_val.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True)

    s_val = copy.deepcopy(s.loc[(slice(None), ids2), slice(None)])
    e_val = copy.deepcopy(e.loc[ids2])
    c_val = dNl.NestDict(c)
    c_val.id = 'cross-val'
    c_val.agent_ids = ids2
    c_val.N = Nids
    c_val.color = 'grey'

    if "length" not in e_val.columns:
        e_val["length"] = np.ones(c_val.N) * 0.004

    return s_val, e_val, c_val


def torsNdsps(pars):
    tor_durs = [int(ii[len('tortuosity') + 1:]) for ii in pars if ii.startswith('tortuosity')]
    tor_durs = np.unique(tor_durs)
    tor_shorts = [f'tor{ii}' for ii in tor_durs]

    dsp_temp = [ii[len(dsp) + 1:].split('_') for ii in pars if ii.startswith(f'{dsp}_')]
    dsp_starts = np.unique([int(ii[0]) for ii in dsp_temp]).tolist()
    dsp_stops = np.unique([int(ii[1]) for ii in dsp_temp]).tolist()
    dsp_shorts0 = [f'dsp_{s0}_{s1}' for s0, s1 in itertools.product(dsp_starts, dsp_stops)]
    dsp_shorts = dNl.flatten_list([[f'{ii}_max', f'{ii}_mu', f'{ii}_fin'] for ii in dsp_shorts0])
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
    dd.config.pooled_cycle_curves_errors = dNl.NestDict({'dict': dic, 'stat': stat})
    return stat


def std_norm(df):
    from sklearn.preprocessing import StandardScaler

    df_std = StandardScaler().fit(df).transform(df)
    return pd.DataFrame(df_std, index=df.index, columns=df.columns)


def minmax(df):
    from sklearn.preprocessing import MinMaxScaler

    df_minmax = MinMaxScaler().fit(df).transform(df)
    return pd.DataFrame(df_minmax, index=df.index, columns=df.columns)


if __name__ == '__main__':
    refID = 'None.150controls'
    mID = 'forager'

    d = reg.simRef(refID, mID=mID, dur=3, dt=1 / 16, Nids=5, color='blue', enrichment=True)

