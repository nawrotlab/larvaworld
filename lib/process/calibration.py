import heapq
import itertools
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import stats
from sklearn.linear_model import LinearRegression

import lib.aux.naming as nam
from lib.anal.plotting import plot_spatiotemporal_variation, plot_bend2orientation_analysis, \
    plot_sliding_window_analysis
from lib.aux.dictsNlists import AttrDict
from lib.conf.base.opt_par import getPar
from lib.process.angular import comp_orientations, comp_angles, comp_angular
from lib.process.aux import interpolate_nans
from lib.process.basic import comp_extrema
from lib.process.bouts import detect_contacting_chunks
from lib.aux.parsing import multiparse_dataset_by_sliding_window
from lib.process.spatial import comp_centroid


def choose_velocity_flag(s=None, e=None, c=None, dt=None, Npoints=None, from_file=True, save_to=None, dataset=None,
                         **kwargs):
    if all([k is None for k in [s, e, c, dt, Npoints]]):
        if dataset is not None:
            d = dataset
            s = d.step_data
            e = d.endpoint_data
            c = d.config
            dt = d.dt
            Npoints = d.Npoints
        else:
            raise ValueError('No dataset provided')
    ids = s.index.unique('AgentID').values
    ps = nam.midline(Npoints, type='point')
    # Define all candidate velocities, their respective points and their key labels
    points = ['centroid'] + ps[1:] + ps
    vels = [nam.vel('centroid')] + nam.lin(nam.vel(ps[1:])) + nam.vel(ps)
    svels = nam.scal(vels)
    vels_minima = nam.min(vels)
    vels_maxima = nam.max(vels)
    svels_minima = nam.min(svels)
    svels_maxima = nam.max(svels)

    # self.comp_spatial(mode='full', is_last=True)
    # self.comp_orientations(mode='full', is_last=True)
    # self.comp_linear(mode='full', is_last=True)
    int = 0.3
    svel_max_thr = 0.1

    comp_extrema(s=s, dt=dt, parameters=svels, interval_in_sec=int,
                 threshold_in_std=None, abs_threshold=[np.inf, svel_max_thr])
    if not from_file:
        m_t_cvs = []
        m_s_cvs = []
        mean_crawl_ratios = []
        for sv, p, sv_min, sv_max in zip(svels, points, svels_minima, svels_maxima):
            detect_contacting_chunks(s=s, e=e, c=c, dt=dt, track_point=p, mid_flag=sv_max, edge_flag=sv_min,
                                     vel_par=sv, **kwargs)
            t_cvs = []
            s_cvs = []
            for id in ids:
                ss = s.xs(id, level='AgentID', drop_level=True)
                durs = ss['stride_dur'].dropna().values
                dsts = ss['scaled_stride_dst'].dropna().values
                t_cv = stats.variation(durs)
                s_cv = stats.variation(dsts)
                t_cvs.append(t_cv)
                s_cvs.append(s_cv)
            m_s_cvs.append(np.mean(s_cvs))
            m_t_cvs.append(np.mean(t_cvs))
            mean_crawl_ratios.append(e[nam.dur_ratio('stride')].mean())
        df = pd.DataFrame(list(zip(m_s_cvs, m_t_cvs)), index=svels,
                          columns=['spatial_cvs', 'temporal_cvs'])
        file_path = os.path.join(save_to, 'spatiotemporal_stride_cvs.csv')
        df.to_csv(file_path, index=True)
        print(f'Spatiotemporal cvs saved as {file_path}')
        a, b = np.min(mean_crawl_ratios), np.max(mean_crawl_ratios)
        mean_crawl_ratios = [10 + 100 * (c - a) / (b - a) for c in mean_crawl_ratios]
        if dataset is not None:
            plot_spatiotemporal_variation(dataset=dataset, spatial_cvs=m_s_cvs, temporal_cvs=m_t_cvs,
                                          sizes=mean_crawl_ratios,
                                          save_to=save_to,
                                          save_as=f'stride_variability_svel_max_{svel_max_thr}_interval_{int}_sized.pdf')
            plot_spatiotemporal_variation(dataset=dataset, spatial_cvs=m_s_cvs, temporal_cvs=m_t_cvs,
                                          sizes=[300 for c in mean_crawl_ratios],
                                          save_to=save_to,
                                          save_as=f'stride_variability_svel_max_{svel_max_thr}_interval_{int}.pdf')

    else:
        for flags, filename in zip([vels_minima, vels_maxima], ['velocity_minima_flags', 'velocity_maxima_flags']):
            m_s_cvs, m_t_cvs = compute_spatiotemporal_cvs(s, e, dt, flags=flags, points=points)
            df = pd.DataFrame(list(zip(m_s_cvs, m_t_cvs)), index=flags,
                              columns=['spatial_cvs', 'temporal_cvs'])
            file_path = os.path.join(save_to, f'{filename}.csv')
            df.to_csv(file_path, index=True)
            print(f'Spatiotemporal cvs saved as {file_path}')
            if dataset is not None:
                plot_spatiotemporal_variation(dataset=dataset, spatial_cvs=m_s_cvs, temporal_cvs=m_t_cvs,
                                              save_to=save_to, save_as=f'{filename}.pdf')


def choose_orientation_flag(s, segs, save_to=None):
    chunk = 'stride'
    ors = nam.orient(segs)
    stride_or = nam.orient(chunk)
    s_stride_or = s[stride_or].dropna().values
    s_ors_start = s[ors + [nam.start(chunk)]].dropna().values
    s_ors_stop = s[ors + [nam.stop(chunk)]].dropna().values
    rNps = np.zeros([len(ors), 4]) * np.nan
    for i, o in enumerate(ors):
        r1, p1 = stats.pearsonr(s_stride_or, s_ors_start[:, i])
        rNps[i, 0] = r1
        rNps[i, 1] = p1
        r2, p2 = stats.pearsonr(s_stride_or, s_ors_stop[:, i])
        rNps[i, 2] = r2
        rNps[i, 3] = p2
    df = pd.DataFrame(np.round(rNps, 4), index=ors)
    df.columns = ['Pearson r (start)', 'p-value (start)', 'Pearson r (stop)', 'p-value (stop)']
    if save_to is not None:
        filename = f'{save_to}/choose_orientation.csv'
        df.to_csv(filename)
        print(f'Stride orientation prediction saved as {filename}!')


def compute_spatiotemporal_cvs(s, e, dt, flags, points):
    ids = s.index.unique('AgentID').values
    all_t_cvs = []
    all_s_cvs = []
    for id in ids:
        data = s.xs(id, level='AgentID', drop_level=True)
        l = e['length'].loc[id]
        t_cvs = []
        s_cvs = []
        for f, p in zip(flags, points):
            indexes = data[f].dropna().index.values
            t_cv = stats.variation(np.diff(indexes) * dt)
            t_cvs.append(t_cv)

            coords = np.array(data[nam.xy(p)].loc[indexes])
            dx = np.diff(coords[:, 0])
            dy = np.diff(coords[:, 1])
            d = np.sqrt(dx ** 2 + dy ** 2)
            scaled_d = d / l
            s_cv = stats.variation(scaled_d)
            s_cvs.append(s_cv)
        all_t_cvs.append(t_cvs)
        all_s_cvs.append(s_cvs)
    m_t_cvs = np.mean(np.array(all_t_cvs), axis=0)
    m_s_cvs = np.mean(np.array(all_s_cvs), axis=0)
    return m_s_cvs, m_t_cvs


def choose_rotation_point(s=None, e=None, dt=None, Npoints=None, c=None, dataset=None):
    if all([k is None for k in [s, e, dt, Npoints]]):
        if dataset is not None:
            d = dataset
            s = d.step_data
            e = d.endpoint_data
            dt = d.dt
            Npoints = d.Npoints
            c = d.config
        else:
            raise ValueError('No dataset provided')
    points = nam.midline(Npoints, type='point')
    Nangles = np.clip(Npoints - 2, a_min=0, a_max=None)
    comp_orientations(s, e, c)
    comp_angles(s, e, c, mode='full')
    comp_angular(s, e, c, mode='full')

    if dataset is not None:
        dataset.save()
        best_combo = plot_bend2orientation_analysis(dataset=dataset)
        front_body_ratio = len(best_combo) / Nangles
        dataset.two_segment_model(front_body_ratio=front_body_ratio)


def stride_max_flag_phase_analysis(dataset, agent_id=None, flag=None, par=None):
    d = dataset
    if d.step_data is None:
        d.load()
    s, e = d.step_data, d.endpoint_data
    if agent_id is None:
        agent_id = e.num_ticks.nlargest(1).index.values[0]
    data = s.xs(agent_id, level='AgentID', drop_level=False)
    if flag is None:
        flag = nam.scal(d.velocity)
    if par is None:
        par = nam.scal(d.distance)
    f = e[nam.freq(flag)].loc[agent_id]
    r = float((1 / f) / 2)
    multiparse_dataset_by_sliding_window(data=data, par=par, flag=nam.max(flag),
                                         radius_in_ticks=np.ceil(r / d.dt),
                                         description_to=f'{d.data_dir}/{par}_around_{flag}', condition='True',
                                         description_as=None, overwrite=True)
    optimal_flag_phase, mean_stride_dst = plot_sliding_window_analysis(dataset=d, parameter=par,
                                                                       flag=nam.max(flag),
                                                                       radius_in_sec=r)
    print(f'Optimal flag phase at {optimal_flag_phase} rad')
    print(f'Mean stride dst at optimum : {mean_stride_dst} (possibly scal)')
    d.stride_max_flag_phase = optimal_flag_phase
    d.config['stride_max_flag_phase'] = optimal_flag_phase
    d.save_config()


def comp_stride_variation(d, component_vels=True):
    from lib.process.aux import detect_strides, process_epochs, fft_max
    s, e, c = d.step_data, d.endpoint_data, d.config
    N = c.Npoints
    points = nam.midline(N, type='point')
    vels = nam.vel(points)
    cvel = nam.vel('centroid')
    lvels = nam.lin(nam.vel(points[1:]))

    all_point_idx = np.arange(N).tolist() + [-1] + np.arange(N).tolist()[1:]
    all_points = points + ['centroid'] + points[1:]
    lin_flag = [False] * N + [False] + [True] * (N - 1)
    all_vels0 = vels + [cvel] + lvels
    all_vels = nam.scal(all_vels0)

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

    shorts = ['fsv', 'str_N', 'str_tr', 'str_t_mu', 'str_t_std', 'sstr_d_mu', 'sstr_d_std', 'str_t_var','sstr_d_var']
    pars = getPar(shorts)
    sstr_d_var, str_t_var, str_tr = pars[-1], pars[-2], pars[2]

    if any([vv not in s.columns for vv in vels + [cvel]]):
        from lib.process.spatial import comp_spatial
        comp_centroid(s, c, recompute=False)
        comp_spatial(s, e, c, mode='full')

    if any([vv not in s.columns for vv in lvels]):
        from lib.process.spatial import comp_linear
        from lib.process.angular import comp_orientations
        comp_orientations(s, e, c, mode='full')
        comp_linear(s, e, c, mode='full')

    if any([vv not in s.columns for vv in all_vels]):
        from lib.process.spatial import scale_to_length
        scale_to_length(s, e, c, pars=all_vels0)

    svels = [p for p in all_vels if p in s.columns]
    my_index = pd.MultiIndex.from_product([svels, c.agent_ids], names=['VelPar', 'AgentID'])
    df = pd.DataFrame(index=my_index, columns=pars)

    for ii in range(c.N):
        print(ii)
        id = c.agent_ids[ii]
        ss, ee = d.get_larva(ii)
        for i, vv in enumerate(svels):
            cum_dur = ss[vv].dropna().values.shape[0] * c.dt
            a = ss[vv].values
            fr = fft_max(a, c.dt, fr_range=(1, 2.5))
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
    print('dddd')
    str_var = df[[sstr_d_var, str_t_var, str_tr]].groupby('VelPar').mean()
    for ii in ['symbol', 'color', 'marker','par', 'idx', 'point', 'point_idx', 'use_component_vel'] :
        str_var[ii]= [dic[jj][ii] for jj in str_var.index.values]
    dic = {'stride_data': df, 'stride_variability': str_var}

    sNt_cv = str_var[[sstr_d_var, str_t_var]].sum(axis=1)
    best_idx = sNt_cv.argmin()
    c.metric_definition.spatial.fitted = AttrDict.from_nested_dicts(
        {'point_idx': int(str_var['point_idx'].iloc[best_idx]),
         'use_component_vel': bool(str_var['use_component_vel'].iloc[best_idx])})
    print('Stride variability analysis complete!')
    return dic


def comp_segmentation(d):
    s, e, c = d.step_data, d.endpoint_data, d.config
    avels = nam.vel(d.angles)
    hov = nam.vel(nam.orient('front'))

    if not set(avels).issubset(s.columns.values):
        from lib.process.angular import comp_angles, comp_angular
        comp_angles(s, e, c, mode='full')
        comp_angular(s, e, c, mode='full')
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

    c.metric_definition.angular.fitted = AttrDict.from_nested_dicts(
        {'best_combo': str(best_combo), 'front_body_ratio': front_body_ratio,
         'bend': 'from_vectors'})
    print('Angular velocity definition analysis complete!')
    return dic

def fit_ang_pars(refID) :
    from lib.conf.stored.conf import loadRef
    from lib.conf.base.par import ParDict
    from lib.model.modules.turner import NeuralOscillator
    from scipy.optimize import minimize, rosen, rosen_der

    d = loadRef(refID)
    d.load(contour=False)
    s, e, c = d.step_data, d.endpoint_data, d.config

    dic = ParDict(mode='load').dict
    fsv, ffov = [dic[k]['d'] for k in ['fsv', 'ffov']]




    def eval(B, V, B0, V0):
        eB = np.sum((B - B0) ** 2)  # /np.nanmean(np.abs(B0))
        eV = np.sum((V - V0) ** 2)  # /np.nanmean(np.abs(V0))
        error = eB + eV
        if np.isnan(error):
            raise
        return error

    def prepare_chunks(chunks):
        data = []
        for cc in chunks:
            id = cc['id']
            fr = e[ffov].loc[id]
            coef, intercept = 0.024, 5
            A_in = fr / coef + intercept
            Niters = int(1 / fr / c.dt)

            chunk = cc['chunk']
            B0 = np.deg2rad(chunk['bend'].values)
            V0 = np.deg2rad(chunk['front_orientation_velocity'].values)

            B0 = interpolate_nans(B0)
            V0 = interpolate_nans(V0)

            data.append([A_in, Niters, B0, V0])
        return data

    def run(q):
        torque_coef, ang_damp_coef, body_spring_k = q

        def compute_ang_vel(torque, v, b):
            return v + (-ang_damp_coef * v - body_spring_k * b + torque) * c.dt

        cum_error = []
        for A_in, Niters, B0, V0 in data:
            N = B0.shape[0]

            B = np.zeros([Niters, N]) * np.nan
            V = np.zeros([Niters, N]) * np.nan
            E = np.zeros(Niters) * np.nan

            B[:, 0] = B0[0]
            V[:, 0] = V0[0]

            for j in range(Niters):
                L = NeuralOscillator(dt=c.dt)
                b = B0[0]
                v = V0[0]
                for ii in range(j):
                    L.step(A_in)
                for i in range(N - 1):
                    L.step(A_in)
                    v = compute_ang_vel(torque_coef * L.activity, v, b)
                    b += v * c.dt
                    B[j, i + 1] = b
                    V[j, i + 1] = v

                E[j] = eval(B[j, :], V[j, :], B0, V0)
            cum_error.append(np.nanmin(E))
        return np.sum(cum_error)

    chunks = d.get_chunks(chunk='pause', shorts=['b', 'fov'], min_dur=3)
    data = prepare_chunks(chunks)
    res = minimize(run, (0.2, 2.6, 5.9), method='SLSQP', bounds=((0, 10), (0, 5), (0, 100)))

    from lib.conf.base.dtypes import null_dict
    tc, z, k = np.round(res.x, 1)
    phys_dic=null_dict('physics', torque_coef=tc, ang_damping=z, body_spring_k=k)

    return phys_dic