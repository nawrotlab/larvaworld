import heapq
import itertools
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import stats
from sklearn.linear_model import LinearRegression


from larvaworld.lib import reg, aux
from larvaworld.lib.process.spatial import comp_centroid
from larvaworld.lib.process.annotation import detect_strides, process_epochs



def comp_stride_variation(d):

    s, e, c = d.step_data, d.endpoint_data, d.config
    N = c.Npoints
    points = aux.nam.midline(N, type='point')
    vels = aux.nam.vel(points)
    cvel = aux.nam.vel('centroid')
    lvels = aux.nam.lin(aux.nam.vel(points[1:]))

    all_point_idx = np.arange(N).tolist() + [-1] + np.arange(N).tolist()[1:]
    all_points = points + ['centroid'] + points[1:]
    lin_flag = [False] * N + [False] + [True] * (N - 1)
    all_vels0 = vels + [cvel] + lvels
    all_vels = aux.nam.scal(all_vels0)

    vel_num_strings = ['{' + str(i + 1) + '}' for i in range(N)]
    lvel_num_strings = ['{' + str(i + 2) + '}' for i in range(N - 1)]
    symbols = [rf'$v_{i}$' for i in vel_num_strings] + [r'$v_{cen}$'] + [rf'$v^{"c"}_{i}$' for i in
                                                                         lvel_num_strings]

    markers = ['o' for i in range(len(vels))] + ['s'] + ['v' for i in range(len(lvels))]
    cnum = 1 + N
    cmap0 = plt.get_cmap('hsv')
    cmap0 = [cmap0(1. * i / cnum) for i in range(cnum)]
    cmap0 = cmap0[1:] + [cmap0[0]] + cmap0[2:]

    dic = {all_vels[ii]: {'symbol': symbols[ii], 'marker': markers[ii], 'color': cmap0[ii],
                          'idx': ii, 'par': all_vels0[ii], 'point': all_points[ii], 'point_idx': all_point_idx[ii],
                          'use_component_vel': lin_flag[ii]} for ii in
           range(len(all_vels))}



    if any([vv not in s.columns for vv in vels + [cvel]]):
        from larvaworld.lib.process.spatial import comp_spatial
        comp_centroid(s, c, recompute=False)
        comp_spatial(s, e, c, mode='full')

    if any([vv not in s.columns for vv in lvels]):
        from larvaworld.lib.process.spatial import comp_linear
        from larvaworld.lib.process.angular import comp_orientations
        comp_orientations(s, e, c, mode='full')
        comp_linear(s, e, c, mode='full')

    if any([vv not in s.columns for vv in all_vels]):
        from larvaworld.lib.process.spatial import scale_to_length
        scale_to_length(s, e, c, pars=all_vels0)

    svels = [p for p in all_vels if p in s.columns]

    shorts = ['fsv', 'str_N', 'str_tr', 'str_t_mu', 'str_t_std', 'str_sd_mu', 'str_sd_std', 'str_t_var', 'str_sd_var']

    my_index = pd.MultiIndex.from_product([svels, c.agent_ids], names=['VelPar', 'AgentID'])
    df = pd.DataFrame(index=my_index, columns=reg.getPar(shorts))

    for ii in range(c.N):
        # print(ii)
        id = c.agent_ids[ii]
        ss, ee = s.xs(id, level='AgentID'), e.loc[id]
        for i, vv in enumerate(svels):
            cum_dur = ss[vv].dropna().values.shape[0] * c.dt
            a = ss[vv].values
            fr = aux.fft_max(a, c.dt, fr_range=(1, 2.5))
            strides = detect_strides(a, fr=fr, dt=c.dt, return_extrema=False, return_runs=False)
            if len(strides) == 0:
                row = [fr, 0, np.nan, np.nan, np.nan,
                       np.nan, np.nan, np.nan, np.nan]
            else:
                strides[:, 1] = strides[:, 1] - 1
                durs, amps, maxs = process_epochs(a, strides, dt=c.dt, return_idx=False)
                Nstr = strides.shape[0]

                t_cv = stats.variation(durs)
                s_cv = stats.variation(amps)
                row = [fr, Nstr, np.sum(durs) / cum_dur, np.mean(durs), np.std(durs),
                       np.mean(amps), np.std(amps), t_cv, s_cv]

            df.loc[(vv, id)] = row
    str_var = df[reg.getPar(['str_sd_var', 'str_t_var', 'str_tr'])].astype(float).groupby('VelPar').mean()
    for ii in ['symbol', 'color', 'marker','par', 'idx', 'point', 'point_idx', 'use_component_vel'] :
        str_var[ii]= [dic[jj][ii] for jj in str_var.index.values]
    dic = {'stride_data': df, 'stride_variability': str_var}

    sNt_cv = str_var[reg.getPar(['str_sd_var', 'str_t_var'])].sum(axis=1)
    best_idx = sNt_cv.argmin()
    c.metric_definition.spatial.fitted = aux.AttrDict(
        {'point_idx': int(str_var['point_idx'].iloc[best_idx]),
         'use_component_vel': bool(str_var['use_component_vel'].iloc[best_idx])})
    print('Stride variability analysis complete!')
    return dic


def comp_segmentation(d):
    s, e, c = d.step_data, d.endpoint_data, d.config
    avels = aux.nam.vel(d.angles)
    hov = aux.nam.vel(aux.nam.orient('front'))

    if not set(avels).issubset(s.columns.values):
        from larvaworld.lib.process.angular import comp_angles, comp_angular
        comp_angles(s, c.Npoints)
        comp_angular(s, c.dt,c.Npoints, mode='full')
    if not set(avels).issubset(s.columns.values):
        raise ValueError('Spineangle angular velocities do not exist in step')

    N = d.Nangles
    ss = s.loc[s[hov].dropna().index.values]
    y = ss[hov].values

    print('Computing linear regression of angular velocity based on segmental bending velocities')
    df_reg = []
    for i in range(N):
        p0 = avels[i]
        X0 = ss[[p0]].values
        reg0 = LinearRegression().fit(X0, y)
        sc0 = reg0.score(X0, y)
        c0 = reg0.coef_[0]
        p1 = avels[:i + 1]
        X1 = ss[p1].values
        reg1 = LinearRegression().fit(X1, y)
        sc1 = reg1.score(X1, y)
        c1 = reg1.coef_[0]

        df_reg.append({'idx': i + 1,
                       'single_par': p0, 'single_score': sc0, 'single_coef': c0,
                       'cum_pars': p1, 'cum_score': sc1, 'cum_coef': c1,
                       })
    df_reg = pd.DataFrame(df_reg)
    df_reg.set_index('idx', inplace=True)

    print('Computing correlation of angular velocity with combinations of segmental bending velocities')
    df_corr = []
    for i in range(int(N * 4 / 7)):
        for idx in itertools.combinations(np.arange(N), i + 1):
            if i == 0:
                idx = idx[0]
                idx0 = idx + 1
                ps = avels[idx]
                X = ss[ps].values
            else:
                idx = list(idx)
                idx0 = [ii + 1 for ii in idx]
                ps = [avels[ii] for ii in idx]
                X = ss[ps].sum(axis=1).values
            r, p = stats.pearsonr(y, X)

            df_corr.append({'idx': idx0, 'pars': ps, 'corr': r, 'p-value': p})

    df_corr = pd.DataFrame(df_corr)
    df_corr.set_index('idx', inplace=True)
    df_corr.sort_values('corr', ascending=False, inplace=True)
    dic = {'bend2or_regression': df_reg, 'bend2or_correlation': df_corr}
    best_combo = df_corr.index.values[0]
    best_combo_max = np.max(best_combo)
    front_body_ratio = best_combo_max / N

    c.metric_definition.angular.fitted = aux.AttrDict(
        {'best_combo': str(best_combo), 'front_body_ratio': front_body_ratio,
         'bend': 'from_vectors'})
    print('Angular velocity definition analysis complete!')
    return dic
