import copy
import itertools
import math
import os

from scipy.stats import ks_2samp
from typing import Union

import numpy as np
from matplotlib import cm
from scipy.stats import ks_2samp
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

import lib.aux.dictsNlists as dNl
import lib.aux.naming as nam
from lib.anal.fitting import std_norm, minmax
from lib.anal.plot_aux import plot_single_bout, dataset_legend
from lib.aux.colsNstr import N_colors, col_df
from lib.aux.combining import render_mpl_table
from lib.aux.sim_aux import circle_to_polygon, inside_polygon

from lib.conf.base.par import getPar, ParDict
from lib.conf.stored.conf import loadRef
from lib.process.aux import annotation, fit_bouts
from lib.process.spatial import scale_to_length, comp_straightness_index, comp_dispersion
from lib.process.store import get_dsp

# plt_conf = {'axes.labelsize': 25,
#             'axes.titlesize': 30,
#             'figure.titlesize': 30,
#             'xtick.labelsize': 20,
#             'ytick.labelsize': 20,
#             'legend.fontsize': 20,
#             'legend.title_fontsize': 25}
# plt.rcParams.update(plt_conf)
# plt.rcParams['text.usetex'] = True
# plt.rcParams['axes.labelpad'] = '18'

dic = ParDict(mode='load').dict
dst, v, sv, acc, sa, fou, rou, fo, ro, b, fov, rov, bv, foa, roa, ba, x, y, l, dsp, dsp_0_40, dsp_0_40_mu, dsp_0_40_max, sdsp, sdsp_0_40, sdsp_0_40_mu, sdsp_0_40_max, str_fov_mu, run_fov_mu, pau_fov_mu, str_fov_std, pau_fov_std, sstr_d_mu, sstr_d_std, str_d_mu, str_d_std, str_sv_mu, pau_sv_mu, str_v_mu, run_v_mu, run_sv_mu, pau_v_mu, str_tr, run_tr, pau_tr, Ltur_tr, Rtur_tr, Ltur_fou, Rtur_fou, run_t_min, run_t_max, pau_t_min, pau_t_max, cum_t, run_t, run_dst, pau_t = [
    dic[k]['d'] for k in
    ['d', 'v', 'sv', 'a', 'sa', 'fou', 'rou', 'fo', 'ro', 'b', 'fov', 'rov', 'bv', 'foa', 'roa', 'ba', 'x', 'y', 'l',
     "dsp", "dsp_0_40", "dsp_0_40_mu", "dsp_0_40_max", "sdsp", "sdsp_0_40", "sdsp_0_40_mu", "sdsp_0_40_max",
     'str_fov_mu', 'run_fov_mu', 'pau_fov_mu', 'str_fov_std', 'pau_fov_std', 'sstr_d_mu', 'sstr_d_std', 'str_d_mu',
     'str_d_std', 'str_sv_mu', 'pau_sv_mu', 'str_v_mu', 'run_v_mu', 'run_sv_mu', 'pau_v_mu', 'str_tr', 'run_tr',
     'pau_tr', 'Ltur_tr', 'Rtur_tr', 'Ltur_fou', 'Rtur_fou', 'run_t_min', 'run_t_max', 'pau_t_min', 'pau_t_max',
     'cum_t', 'run_t', 'run_d', 'pau_t']]
l, v_mu, cum_d, sv_mu, fov_mu, b_mu = [dic[k]['d'] for k in ['l', 'v_mu', "cum_d", "sv_mu", 'fov_mu', 'b_mu']]
tors = tor, tor2, tor2_mu, tor2_std, tor5, tor5_mu, tor5_std, tor10, tor10_mu, tor10_std, tor20, tor20_mu, tor20_std = [
    dic[k]['d'] for k in
    ["tor", "tor2", "tor2_mu", "tor2_std", "tor5", "tor5_mu", "tor5_std", "tor10", "tor10_mu", "tor10_std", "tor20",
     "tor20_mu", "tor20_std"]]
fsv, ffov = [dic[k]['d'] for k in ['fsv', 'ffov']]

att = 'attenuation'
att_max, att_min, phi_att_max, phi_sv_max = nam.max(att), nam.min(att), nam.max(f'phi_{att}'), nam.max(f'phi_{sv}')


def adapt_conf(conf0, ee, c):
    conf = copy.deepcopy(conf0)

    dic = {
        run_sv_mu: ee[str_sv_mu] if str_sv_mu in ee.index else ee[run_v_mu] / ee[l],
        run_v_mu: ee[run_v_mu],
        run_t_min: ee[run_t_min],
        run_t_max: ee[run_t_max],
        pau_t_min: ee[pau_t_min],
        pau_t_max: ee[pau_t_max],
        'ang_vel_headcast': np.deg2rad(ee[pau_fov_mu]),
        'theta_min_headcast': np.deg2rad(ee['headcast_q25_amp']),
        'theta_max_headcast': np.deg2rad(ee['headcast_q75_amp']),
        'theta_max_weathervane': np.deg2rad(ee['weathervane_q75_amp']),
        'ang_vel_weathervane': np.deg2rad(ee[run_fov_mu]),
        'turner_input_constant': ee['turner_input_constant'],
        'run_dist': c.bout_distros.run_dur,
        'pause_dist': c.bout_distros.pause_dur,
        'stridechain_dist': c.bout_distros.run_count}

    if c.Npoints > 1:
        ddic = {'initial_freq': ee[fsv],
                'step_mu': ee[sstr_d_mu],
                'step_std': ee[sstr_d_std],
                'attenuation_min': ee[att_min],
                'attenuation_max': ee[att_max],
                'max_vel_phase': ee[phi_sv_max]}
        dic.update(**ddic)
    for kk, vv in dic.items():
        if kk in conf.keys():
            conf[kk] = vv

    return conf


def sim_locomotor(L, N, dt, df_cols, e_id, tank_polygon):
    aL = np.ones([N, len(df_cols)]) * np.nan
    aL[0, :] = 0
    try:
        xy0s = e_id[nam.initial([fo, x, y])]
        if not None in xy0s:
            aL[0, 3:6] = xy0s
    except:
        pass
    l0 = e_id[l] / 2
    cur_fo = np.deg2rad(aL[0, 3])
    cur_x = aL[0, 4]
    cur_y = aL[0, 5]
    for i in range(N - 1):
        lin, ang, feed = L.step(length=e_id[l])
        cur_d = lin * dt
        if not tank_polygon.contains(Point(cur_x, cur_y)):
            cur_fo -= np.pi / 2

        # ang, cur_fo, (cur_x, cur_y) = assess_tank_contact(ang, cur_fo, cur_d, (cur_x, cur_y), dt,l0, tank_polygon)
        # L.ang_vel=ang

        cur_fo += ang * dt
        cur_x += np.cos(cur_fo) * cur_d
        cur_y += np.sin(cur_fo) * cur_d
        aL[i + 1, :7] = [lin, np.rad2deg(ang), cur_d, np.rad2deg(cur_fo), cur_x, cur_y, np.rad2deg(L.bend)]

    return aL


def sim_locomotor_parallel(Ls, N, dt, df_cols, e, tank_polygon):
    aaL = np.zeros([N, len(Ls), len(df_cols)]) * np.nan
    # aL = np.ones([N, len(df_cols)]) * np.nan
    aaL[0, :, :] = 0
    try:
        aaL[0, :, 3:6] = e[nam.initial([fo, x, y])].values
        # aaL[0,:, 3:6] = xy0s
    except:
        pass
    ls = e[l].values
    for j, L in enumerate(Ls):
        # l0 = e_id[l] / 2
        cur_fo = np.deg2rad(aaL[0, j, 3])
        cur_x = aaL[0, j, 4]
        cur_y = aaL[0, j, 5]
        for i in range(N - 1):
            lin, ang, feed = L.step(length=l[j])
            cur_d = lin * dt
            if not tank_polygon.contains(Point(cur_x, cur_y)):
                cur_fo -= np.pi / 2

            # ang, cur_fo, (cur_x, cur_y) = assess_tank_contact(ang, cur_fo, cur_d, (cur_x, cur_y), dt,l0, tank_polygon)
            # L.ang_vel=ang

            cur_fo += ang * dt
            cur_x += np.cos(cur_fo) * cur_d
            cur_y += np.sin(cur_fo) * cur_d
            aaL[i + 1, j, :7] = [lin, np.rad2deg(ang), cur_d, np.rad2deg(cur_fo), cur_x, cur_y, np.rad2deg(L.bend)]

    return aaL


def sim_dataset(d, s, e, c, ids, loco_id, loco_func, loco_conf, adapted=False, parallel=False, **kwargs):
    tank_polygon = get_tank_polygon(c)
    N = c.Nticks
    strides_enabled = True if 'initial_freq' in loco_conf.keys() else False
    vel_thr = c.vel_thr if c.Npoints == 1 else None

    # Define columns for the simulated dataset
    N = c.Nticks
    tank_polygon = get_tank_polygon(c)
    df_cols, = getPar(['v', 'fov', 'd', 'fo', 'x', 'y', 'b'], to_return=['d'])
    Nids = len(ids)
    Ncols = len(df_cols)

    l = "length"
    if l not in e.columns:
        e[l] = np.ones(Nids) * 0.004
    # empty=np.ones([c.Nticks*len(ids), len(df_cols)])*np.nan
    cc = dNl.AttrDict.from_nested_dicts({
        'id': loco_id,
        'dt': c.dt,
        'Nticks': N,
        'Npoints': c.Npoints,
        'point': '',

    })

    e_ps = [l, nam.initial(x), nam.initial(y), nam.initial(fo)]
    ee = pd.DataFrame(e[e_ps].loc[ids], index=ids, columns=e_ps)
    ee[cum_t] = N * c.dt

    aaL = np.zeros([N, Nids, Ncols]) * np.nan

    Ls = {}
    for jj, id in enumerate(ids):
        ee_id = e.loc[id]

        # Build locomotor instance of given model configuration adapted to the specific experimental larva
        if adapted == 'I':
            loco_conf = adapt_conf(loco_conf, e.loc[id], c)
        Ls[id] = loco_func(dt=c.dt, **loco_conf)

    if parallel:
        aaL = sim_locomotor_parallel(Ls, N, c.dt, df_cols, e, tank_polygon)
    else:
        aaL = np.zeros([N, Nids, Ncols]) * np.nan
        for jj, id in enumerate(ids):
            aaL[:, jj, :] = sim_locomotor(Ls[id], N, c.dt, df_cols, e.loc[id], tank_polygon)

    df_index = pd.MultiIndex.from_product([np.arange(N), ids], names=['Step', 'AgentID'])
    ss = pd.DataFrame(aaL.reshape([N * Nids, Ncols]), index=df_index, columns=df_cols)

    enrich_dataset(ss, ee, cc, **kwargs)

    aux_dic = annotation(ss, ee, cc, strides_enabled=strides_enabled, vel_thr=vel_thr, save_to=d.dir_dict.chunk_dicts)
    bout_dic = fit_bouts(aux_dic, loco_id, cc, save_to=d.dir_dict.group_bout_dicts)

    return ss, ee, cc, bout_dic


def enrich_dataset(ss, ee, cc, tor_durs=[2, 5, 10, 20], dsp_starts=[0], dsp_stops=[40]):
    ss[bv] = ss[b].groupby('AgentID').diff() / cc.dt
    ss[ba] = ss[bv].groupby('AgentID').diff() / cc.dt
    ss[foa] = ss[fov].groupby('AgentID').diff() / cc.dt
    ss[acc] = ss[v].groupby('AgentID').diff() / cc.dt

    ee[v_mu] = ss[v].dropna().groupby('AgentID').mean()
    ee[cum_d] = ss[dst].dropna().groupby('AgentID').sum()

    scale_to_length(ss, ee, keys=['d', 'v', 'a', 'v_mu'])

    comp_dispersion(ss, ee, cc.dt, cc.point, dsp_starts=dsp_starts, dsp_stops=dsp_stops)
    comp_straightness_index(ss, cc.dt, e=ee, tor_durs=tor_durs)


def eval_dataset(s, ss, e, ee, end_ps, distro_ps):
    ids = e.index.values
    Eend, Edistro = {}, {}
    Eend_pool, Edistro_pool = {}, {}

    # Averages
    ps1, ps1l = getPar(end_ps, to_return=['d', 'l'])
    for p, pl in zip(ps1, ps1l):
        Eend[pl] = None
        Eend_pool[pl] = None
        if p in e.columns and p in ee.columns:
            # error_end[pl]=eval_func(np.abs(e[p] - ee[p]))
            Eend[pl] = ((e[p] - ee[p]) ** 2).mean() ** .5
            Eend_pool[pl] = ks_2samp(e[p].values, ee[p].values)[0]

    # Distributions
    N = 20
    ps2, ps2l = getPar(distro_ps, to_return=['d', 'l'])
    for p, pl in zip(ps2, ps2l):
        Edistro[pl] = None
        Edistro_pool[pl] = None
        if p in s.columns and p in ss.columns:
            pps = []
            for id in ids:
                sp, ssp = s[p].xs(id, level="AgentID").dropna().values, ss[p].xs(id, level="AgentID").dropna().values
                if sp.shape[0] > N and ssp.shape[0] > N:
                    pps.append(ks_2samp(sp, ssp)[0])

            Edistro[pl] = np.median(pps)
            spp, sspp = s[p].dropna().values, ss[p].dropna().values
            if spp.shape[0] > N and sspp.shape[0] > N:
                Edistro_pool[pl] = ks_2samp(spp, sspp)[0]

    return Eend, Eend_pool, Edistro, Edistro_pool


def eval_all_datasets(loco_dict, s, e, save_to, suf, evaluation, mode='1:1',
                      norm_modes=['raw', 'minmax', 'std']):
    # Evaluation metrics
    end_ps = dNl.flatten_list(evaluation['end']['shorts'].values.tolist())
    distro_ps = dNl.flatten_list(evaluation['step']['shorts'].values.tolist())
    GEend, GEdistro = {}, {}
    for loco_id in loco_dict.keys():
        Eend, Edistro = {}, {}
        ss = loco_dict[loco_id]['step']
        ee = loco_dict[loco_id]['end']
        ids = e.index.values
        ps1, ps1l = getPar(end_ps, to_return=['d', 'lab'])
        for p, pl in zip(ps1, ps1l):
            Eend[pl] = None
            # Eend_pool[pl] = None
            if p in e.columns and p in ee.columns:
                if mode == '1:1':
                    Eend[pl] = ((e[p] - ee[p]) ** 2).mean() ** .5
                elif mode == 'pooled':
                    Eend[pl] = ks_2samp(e[p].values, ee[p].values)[0]
                    # Eend_pool[pl] = ks_2samp(e[p].values, ee[p].values)[0]

        # Distributions
        N = 20
        ps2, ps2l = getPar(distro_ps, to_return=['d', 'lab'])
        for p, pl in zip(ps2, ps2l):
            Edistro[pl] = None
            # Edistro_pool[pl] = None
            if p in s.columns and p in ss.columns:
                if mode == '1:1':
                    pps = []
                    for id in ids:
                        sp, ssp = s[p].xs(id, level="AgentID").dropna().values, ss[p].xs(id,
                                                                                         level="AgentID").dropna().values
                        if sp.shape[0] > N and ssp.shape[0] > N:
                            pps.append(ks_2samp(sp, ssp)[0])

                    Edistro[pl] = np.median(pps)
                elif mode == 'pooled':
                    spp, sspp = s[p].dropna().values, ss[p].dropna().values
                    if spp.shape[0] > N and sspp.shape[0] > N:
                        Edistro[pl] = ks_2samp(spp, sspp)[0]
                        # Edistro_pool[pl] = ks_2samp(spp, sspp)[0]

        GEend[loco_id] = Eend
        GEdistro[loco_id] = Edistro
        # Eend[loco_id], Eend_pool[loco_id], Edistro[loco_id], Edistro_pool[
        #     loco_id] = eval_dataset(s_exp, ss, e_exp, ee, end_ps, distro_ps)
    if mode == '1:1':
        error_names = ['RSS error', r'median 1:1 distribution KS$_{D}$']
    elif mode == 'pooled':
        error_names = ['pooled endpoint KS$_{D}$', 'pooled distribution KS$_{D}$']
    error_dict = {error_names[0]: pd.DataFrame.from_records(GEend).T,
                  error_names[1]: pd.DataFrame.from_records(GEdistro).T}

    # legend_titles = ['endpoint metrics', 'timeseries metrics']
    error_tables(error_dict, save_to=save_to, suf=suf)
    for norm in norm_modes:
        error_barplots(error_dict, normalization=norm, save_to=save_to, suf=suf, evaluation= evaluation)
    return error_dict


def get_tank_polygon(c):
    X, Y = c.env_params.arena.arena_dims
    arena_shape = c.env_params.arena.arena_shape
    if arena_shape == 'circular':
        # This is a circle_to_polygon shape from the function
        tank_shape = circle_to_polygon(60, X / 2)
    elif arena_shape == 'rectangular':
        # This is a rectangular shape
        tank_shape = np.array([(-X / 2, -Y / 2),
                               (-X / 2, Y / 2),
                               (X / 2, Y / 2),
                               (X / 2, -Y / 2)])
    return Polygon(tank_shape * 0.97)


def assess_tank_contact(ang_vel, o0, d, p0, dt, l0, tank_polygon):
    def avoid_border(ang_vel, counter, dd=0.1):
        if math.isinf(ang_vel) or ang_vel == 0:
            ang_vel = 1.0
        counter += 1
        # print(counter)
        ang_vel *= -(1 + dd * counter)
        return ang_vel, counter

    def check_in_tank(ang_vel, o0, d, p0, l0):
        o1 = o0 + ang_vel * dt
        k = np.array([math.cos(o1), math.sin(o1)])
        dxy = k * d
        p1 = p0 + dxy
        f1 = p1 + k * l0
        in_tank = tank_polygon.contains(Point(f1[0], f1[1]))
        return in_tank, o1, p1

    in_tank, o1, p1 = check_in_tank(ang_vel, o0, d, p0, l0)
    counter = -1
    while not in_tank:
        ang_vel, counter = avoid_border(ang_vel, counter)
        try:
            in_tank, o1, p1 = check_in_tank(ang_vel, o0, d, p0, l0)
        except:
            pass
    return ang_vel, o1, p1

def arrange_evaluation(s,e,evaluation_metrics) :
    d= dNl.AttrDict.from_nested_dicts({'end': {'shorts' : [], 'groups':[]}, 'step': {'shorts' : [], 'groups':[]}})
    for g, shs in evaluation_metrics.items() :
        ps=getPar(shs, to_return='d')[0]
        Dshorts=[sh for sh,p in zip(shs,ps) if p in s.columns]
        Eshorts = [sh for sh, p in zip(shs, ps) if p in e.columns]
        Eshorts = [sh for sh in Eshorts if sh not in Dshorts]
        if len(Eshorts)>0 :
            d.end.shorts.append(Eshorts)
            d.end.groups.append(g)
        if len(Dshorts)>0 :
            d.step.shorts.append(Dshorts)
            d.step.groups.append(g)
    return d



def run_locomotor_evaluation(d, locomotor_models, Nids=None, save_to=None,
                             stridechain_duration=False, cross_validation=True, evaluation_modes=['1:1', 'pooled'],
                             evaluation_metrics=None,
                             norm_modes=['raw', 'minmax', 'std'],
                             plots=['trajectories', 'bouts', 'distros', 'endpoint', 'dispersion']):
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
    s, e, c = d.step_data, d.endpoint_data, d.config
    if Nids is None:
        ids = c.agent_ids
        Nids = len(ids)
    else:
        ids = e.nlargest(Nids, 'cum_dur').index.values

    tor_durs = [int(ii[len(tor) + 1:]) for ii in s.columns if ii.startswith(tor)]
    tor_shorts = [f'tor{ii}' for ii in tor_durs]

    dsp_temp = [ii[len(dsp) + 1:].split('_') for ii in s.columns if ii.startswith(f'{dsp}_')]
    dsp_starts = np.unique([int(ii[0]) for ii in dsp_temp]).tolist()
    dsp_stops = np.unique([int(ii[1]) for ii in dsp_temp]).tolist()
    dsp_shorts0 = [f'dsp_{s0}_{s1}' for s0, s1 in itertools.product(dsp_starts, dsp_stops)]
    dsp_shorts = dNl.flatten_list([[f'{ii}_max', f'{ii}_mu', f'{ii}_fin'] for ii in dsp_shorts0])

    if evaluation_metrics is None:
        evaluation_metrics = {
            'angular motion': ['ffov', 'run_fov_mu', 'run_fov_std', 'pau_fov_mu', 'pau_fov_std', 'b', 'bv', 'ba', 'fov', 'foa'],
            'spatial motion': ['fsv', 'cum_d', 'v_mu', 'run_v_mu','v', 'a'],
            'time allocation': ['run_tr', 'pau_tr'],
            'dispersal': ['dsp_0_40_mu', 'dsp_0_40_max', 'dsp_0_40_fin'],
            'epochs': ['run_t', 'pau_t', 'run_d', 'tur_fou', 'tur_fov_max'],
            'tortuosity': ['tor5', 'tor20']}

    temp=arrange_evaluation(s, e, evaluation_metrics)
    evaluation = {k: col_df(**pars) for k,pars in temp.items()}
    end_ps = dNl.flatten_list(evaluation['end']['shorts'].values.tolist())
    distro_ps = dNl.flatten_list(evaluation['step']['shorts'].values.tolist())

    # Individual-specific model fitting
    s_exp = s.loc[(slice(None), ids), slice(None)]
    e_exp = e.loc[ids]

    loco_dict = {}
    for ii, (loco_id, (func, conf, adapted, col)) in enumerate(locomotor_models.items()):
        print(f'Simulating model {loco_id} on {Nids} larvae from dataset {d.id}')
        ss, ee, cc, bouts = sim_dataset(d, s_exp, e_exp, c, ids, loco_id, func, conf, adapted,
                                        tor_durs=tor_durs, dsp_starts=dsp_starts, dsp_stops=dsp_stops)
        loco_dict[loco_id] = {'step': ss, 'end': ee, 'color': col, 'bouts': bouts}
    print('Evaluating all models')
    error_dicts = {}
    for mode in evaluation_modes:
        suf = 'fitted'
        error_dicts[f'{mode} {suf}'] = eval_all_datasets(loco_dict, s_exp, e_exp, save_to, suf=suf, mode=mode,
                                                         norm_modes=norm_modes, evaluation=evaluation)

    if cross_validation and Nids <= c.N / 2:
        ids2 = e.nlargest(2 * Nids, 'cum_dur').index.values[Nids:]
        s_val = s.loc[(slice(None), ids2), slice(None)]
        e_val = e.loc[ids2]
        e_val.rename(index=dict(zip(ids2, ids)), inplace=True)
        s_val.reset_index(level='Step', drop=False, inplace=True)
        s_val.rename(index=dict(zip(ids2, ids)), inplace=True)
        s_val.reset_index(drop=False, inplace=True)
        s_val.set_index(keys=['Step', 'AgentID'], inplace=True, drop=True)
        for mode in evaluation_modes:
            suf = 'cross-val'
            error_dicts[f'{mode} {suf}'] = eval_all_datasets(loco_dict, s_val, e_val, save_to, suf=suf, mode=mode,
                                                             norm_modes=norm_modes, evaluation=evaluation)
        loco_dict['cross-val'] = {'step': s_val, 'end': e_val, 'color': 'grey', 'bouts': None}
    try:
        exp_bouts = d.load_group_bout_dict()
    except:
        exp_bouts = None
    loco_dict['experiment'] = {'step': s_exp, 'end': e_exp, 'color': 'black', 'bouts': exp_bouts}

    print('Generating comparative graphs')
    if 'distros' in plots:
        plot_distros(loco_dict, distro_ps, save_to=save_to)
    if 'endpoint' in plots:
        plot_endpoint(loco_dict, end_ps, save_to=save_to)
    if 'trajectories' in plots:
        plot_trajectories(loco_dict, save_to=save_to)
    if 'bouts' in plots:
        plot_bouts(loco_dict, save_to=save_to, stridechain_duration=stridechain_duration)
    if 'dispersion' in plots:
        for r0, r1 in itertools.product(dsp_starts, dsp_stops):
            plot_comparative_dispersion(loco_dict, c=c, dsp_range=(r0, r1), save_to=save_to, show=True)
    return error_dicts, loco_dict


def error_tables(error_dict, save_to=None, suf='fitted'):
    dic = {}
    for k, v in error_dict.items():
        ax, fig = render_mpl_table(np.round(v, 6).T, highlighted_cells='row_min', title=f'{suf} {k}')
        dic[k] = fig
        if save_to is not None:
            plt.savefig(f'{save_to}/error_{suf}_{k}.pdf', dpi=300)
    return dic
    # render_mpl_table(np.round(v, 6).T, highlighted_cells='row_min')
    # render_mpl_table(np.round(v, 6).T, highlighted_cells='row_min')
    # render_mpl_table(np.round(v, 6).T, highlighted_cells='row_min')


def error_barplots(error_dict,  evaluation, normalization='minmax', save_to=None, suf='',
                   legend_titles=[r'$\bf{endpoint}$ $\bf{metrics}$', r'$\bf{timeseries}$ $\bf{metrics}$']):
    par_cols = [dNl.flatten_list(evaluation['end']['par_colors'].values.tolist()),
                dNl.flatten_list(evaluation['step']['par_colors'].values.tolist())]
    import matplotlib.patches as mpatches
    Nplots = len(error_dict)
    ddf = {}
    fig, axs = plt.subplots(Nplots, 1, figsize=(20, Nplots * 6), sharex=True)
    axs = axs.ravel()
    for ii, (lab, df) in enumerate(error_dict.items()):
        eval_df=list(evaluation.values())[ii]
        color=dNl.flatten_list(eval_df['par_colors'].values.tolist())
        if normalization == 'minmax':
            df = minmax(df)
        elif normalization == 'std':
            df = std_norm(df)
        try:
            df.plot(kind='bar', ax=axs[ii], ylabel=lab, rot=0, legend=False, color=color)
        except:
            df.plot(kind='bar', ax=axs[ii], ylabel=lab, rot=0, legend=False)
        h, l = axs[ii].get_legend_handles_labels()
        empty = mpatches.Patch(color='none')
        counter=0
        for g in eval_df.index:
            # idx = len(eval_df['shorts'].loc[g])
            h.insert(counter, empty)
            l.insert(counter, fr'$\bf{{{g}}}$')
            counter+=(len(eval_df['shorts'].loc[g])+1)
        h.insert(0, empty)
        l.insert(0, '')
        axs[ii].legend(h,l, loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=15, title=legend_titles[ii])
        ddf[lab] = df.mean(axis=1)
    fig.subplots_adjust(hspace=0.05, top=0.99, bottom=0.15)
    if save_to is not None:
        fig.savefig(f'{save_to}/error_barplots_{suf}_{normalization}.pdf', dpi=300)
    df0 = pd.DataFrame.from_dict(ddf)

    axx, figg = render_mpl_table(df0, highlighted_cells='col_min')
    if save_to is not None:
        figg.savefig(f'{save_to}/mean_error_table_{suf}.pdf', dpi=300)
    return fig


def plot_trajectories(loco_dict, save_to=None, show=False):
    # Plot the trajectories
    Nmods = len(loco_dict)

    fig, axs = plt.subplots(1, Nmods, figsize=(5 * Nmods, 5), sharex=True, sharey=True)
    axs = axs.ravel()
    for ii, (lab, sim) in enumerate(loco_dict.items()):
        for id in sim['step'].index.unique('AgentID').values:
            xy = sim['step'][['x', 'y']].xs(id, level="AgentID").values
            axs[ii].plot(xy[:, 0], xy[:, 1])
        axs[ii].set_title(lab)
    fig.subplots_adjust(wspace=0.05)
    if save_to is not None:
        fig.savefig(f'{save_to}/comparative_trajectories.pdf', dpi=300)
    if show:
        plt.show()
    return fig


def plot_bouts(loco_dict, save_to=None, show=False, plot_fits='none', stridechain_duration=False, legend_outside=False,
               axs=None, fig=None):
    loco_dict = dNl.AttrDict.from_nested_dicts(loco_dict)
    if axs is None and fig is None:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    for j, (lab, sim) in enumerate(loco_dict.items()):
        v = sim['bouts']
        kws = {
            'marker': 'o' if lab == 'experiment' else '.',
            'plot_fits': plot_fits,
            'label': lab,
            'color': sim['color'],
            'legend_outside': legend_outside,
            'axs': axs,
            'x0': None

        }
        if v.pause_dur is not None:
            bout = 'pauses'
            fit_dic = v.pause_dur
            # print(lab, bout, fit_dic.best.pause_dur.best)
            plot_single_bout(fit_dic=fit_dic, discr=False, bout=bout, i=1, **kws)
        if stridechain_duration and v.run_dur is not None:
            bout = 'runs'
            fit_dic = v.run_dur
            # print(lab, bout, fit_dic.best.run_dur.best)
            plot_single_bout(fit_dic=fit_dic, discr=False, bout=bout, i=0, **kws)
        elif not stridechain_duration and v.run_count is not None:
            bout = 'stridechains'
            fit_dic = v.run_count
            # print(lab, bout, fit_dic.best.run_count.best)
            plot_single_bout(fit_dic=fit_dic, discr=True, bout=bout, i=0, **kws)
        # print()
    axs[1].yaxis.set_visible(False)
    if len(loco_dict.keys()) > 1:
        cols = [loco_dict[id]['color'] for id in loco_dict.keys()]
        dataset_legend(loco_dict.keys(), cols, ax=axs[1], loc='center left', fontsize=25, anchor=(1.0, 0.5))
    if save_to is not None:
        # fig.savefig(f'{save_to}/comparative_bouts.eps', dpi=300)
        fig.savefig(f'{save_to}/comparative_bouts.pdf', dpi=300)
        # fig.savefig(f'{save_to}/comparative_bouts.png', dpi=300)
    if show:
        plt.show()


def plot_comparative_dispersion(loco_dict, c, dsp_range, **kwargs):
    from lib.anal.plot_aux import BasePlot, plot_mean_and_range
    from lib.aux.colsNstr import random_colors
    r0, r1 = dsp_range
    p = f'dispersion_{r0}_{r1}'
    P = BasePlot(name=f'comparative_dispersal_{r0}_{r1}', **kwargs)
    P.build(figsize=(20, 6))
    t0, t1 = int(r0 * c.fr), int(r1 * c.fr)
    x = np.linspace(r0, r1, t1 - t0)
    lws = [3] * (len(loco_dict) - 1) + [8]
    for lw, (lab, sim) in zip(lws, loco_dict.items()):
        col = sim['color']
        ddsp = get_dsp(sim['step'], p)
        mean = ddsp['median'].values[t0:t1]
        lb = ddsp['upper'].values[t0:t1]
        ub = ddsp['lower'].values[t0:t1]
        P.axs[0].fill_between(x, ub, lb, color=col, alpha=.2)
        P.axs[0].plot(x, mean, col, label=lab, linewidth=lw, alpha=1.0)
    P.conf_ax(xlab='time, $sec$', ylab=r'dispersal $(mm)$', xlim=[x[0], x[-1]], ylim=[0, None], xMaxN=4, yMaxN=4,
              leg_loc='upper left')
    P.adjust((0.2 / 1, 0.95), (0.15, 0.95), 0.05, 0.005)
    return P.get()


def plot_distros(loco_dict, distro_ps, save_to=None, show=False):
    ps2, ps2l = getPar(distro_ps, to_return=['d', 'lab'])
    Nps = len(distro_ps)
    Ncols = 4
    Nrows = int(np.ceil(Nps / Ncols))
    fig, axs = plt.subplots(Nrows, Ncols, figsize=(5 * Ncols, 5 * Nrows), sharex=False, sharey=False)
    axs = axs.ravel()
    for i, (p, l) in enumerate(zip(ps2, ps2l)):
        vs = []
        for ii, (lab, sim) in enumerate(loco_dict.items()):
            vs.append(sim['step'][p].dropna().abs().values)
        vvs = np.hstack(vs).flatten()
        bins = np.linspace(0, np.quantile(vvs, q=0.9), 40)
        for ii, (lab, sim) in enumerate(loco_dict.items()):
            col = sim['color']
            weights = np.ones_like(vs[ii]) / float(len(vs[ii]))
            axs[i].hist(vs[ii], bins=bins, weights=weights, label=lab, color=col, histtype='step', linewidth=3,
                        facecolor=col, edgecolor=col, fill=True, alpha=0.2)
        axs[i].set_title(l)
    axs[2].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    if save_to is not None:
        fig.savefig(f'{save_to}/comparative_distros.pdf', dpi=300)
    if show:
        plt.show()
    return fig


def plot_endpoint(loco_dict, end_ps, save_to=None, show=False):
    ps2, ps2l = getPar(end_ps, to_return=['d', 'lab'])
    Nps = len(end_ps)
    Ncols = 4
    Nrows = int(np.ceil(Nps / Ncols))
    fig, axs = plt.subplots(Nrows, Ncols, figsize=(5 * Ncols, 5 * Nrows), sharex=False, sharey=False)
    axs = axs.ravel()
    for i, (p, l) in enumerate(zip(ps2, ps2l)):
        vs = []
        for ii, (lab, sim) in enumerate(loco_dict.items()):
            vs.append(sim['end'][p].dropna().values)
        vvs = np.hstack(vs).flatten()
        bins = np.linspace(np.min(vvs), np.max(vvs), 20)
        for ii, (lab, sim) in enumerate(loco_dict.items()):
            col = sim['color']
            weights = np.ones_like(vs[ii]) / float(len(vs[ii]))
            axs[i].hist(vs[ii], bins=bins, weights=weights, label=lab, color=col, histtype='step', linewidth=3,
                        facecolor=col, edgecolor=col, fill=True, alpha=0.2)
        axs[i].set_title(l)
    axs[2].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    if save_to is not None:
        fig.savefig(f'{save_to}/comparative_endpoint.pdf', dpi=300)
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    refID = 'None.100controls'
    # refID='None.Sims2019_controls'
    d = loadRef(refID)
    d.load(contour=False)
    s, e, c = d.step_data, d.endpoint_data, d.config
    trange = np.arange(0, c.Nticks * c.dt, c.dt)

    physics0_args = {
        'torque_coef': 1,
        'ang_damp_coef': 1,
        'body_spring_k': 1,
    }
    physics1_args = {
        'torque_coef': 0.4,
        'ang_damp_coef': 2.5,
        'body_spring_k': 0.25,
    }
    physics2_args = {
        'torque_coef': 1.77,
        'ang_damp_coef': 5.0,
        'body_spring_k': 0.39,
    }

    lat_osc_args = {
        'w_ee': 3.0,
        'w_ce': 0.1,
        'w_ec': 4.0,
        'w_cc': 4.0,
        'm': 100.0,
        'n': 2.0,
    }

    Lev = {
        run_v_mu: e[run_v_mu].mean(),
        'ang_vel_headcast': np.deg2rad(e[pau_fov_mu].mean()),  # 60,
        'run_dist': {'range': [1, 172.125],
                     'name': 'powerlaw',
                     'alpha': 1.46},
        'pause_dist': {'range': [0.4, 2.0],
                       'name': 'uniform'},
    }

    Wys = {
        run_v_mu: 0.001,  # in m/s
        'turner_input_constant': 19.0,
        "bend_correction_coef": 0,
        **lat_osc_args,
        **physics0_args
    }

    Dav = {
        run_v_mu: 0.001,
        run_t_min: 1,
        'theta_min_headcast': 37,
        'theta_max_headcast': 120,
        'theta_max_weathervane': 20,
        'ang_vel_weathervane': 60.0,
        'ang_vel_headcast': 240.0,
        'r_run2headcast': 0.148,
        'r_headcast2run': 2.0,
        'r_weathervane_stop': 2.0,
        'r_weathervane_resume': 1.0,
    }

    Sak = {
        'step_mu': 0.24,
        'step_std': 0.066,
        'initial_freq': 1.36,
        'turner_input_constant': 19.0,
        'attenuation_min': 0.2,
        'attenuation_max': 0.31,
        'max_vel_phase': 3.6,
        'stridechain_dist': c.bout_distros.run_count,
        # 'run_dist': c.bout_distros.run_dur,
        'pause_dist': c.bout_distros.pause_dur,
        "bend_correction_coef": 1.4,
        **lat_osc_args,
        **physics1_args
    }

    from lib.model.modules.locomotor import Sakagiannis2022, Levy_locomotor, Wystrach2016, Davies2015

    locos = {
        "Levy": [Levy_locomotor, Lev, False],
        "Levy+": [Levy_locomotor, Lev, True],
        "Wystrach": [Wystrach2016, Wys, False],
        "Wystrach+": [Wystrach2016, Wys, True],
        "Davies": [Davies2015, Dav, False],
        "Davies+": [Davies2015, Dav, True],
        "Sakagiannis": [Sakagiannis2022, Sak, False],
        "Sakagiannis+": [Sakagiannis2022, Sak, True],
        # "Sakagiannis++": [Sakagiannis2022, Sak2, False],
    }

    error_dict, loco_dict, bout_dict = run_locomotor_evaluation(d, locos, Nids=100,
                                                                save_to='/home/panos/larvaworld_new/larvaworld/tests/metrics/model_comparison/100l')
    # error_tables(error_dict)
    # error_barplots(error_dict, normalization='minmax')
    # plot_trajectories(loco_dict)
    # plot_bouts(d)
    # plot_comparative_dispersion(loco_dict, return_fig=True)
