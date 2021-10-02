import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, spectrogram

import lib.aux.functions as fun
import lib.aux.naming as nam
import lib.conf.dtype_dicts as dtypes
from lib.anal.process.angular import angular_processing
from lib.anal.process.spatial import spatial_processing, compute_bearingNdst2source, compute_dispersion, \
    compute_tortuosity, compute_preference_index, align_trajectories
from lib.conf.par import getPar


def compute_extrema(s, dt, parameters, interval_in_sec, threshold_in_std=None, abs_threshold=None):

    if abs_threshold is None:
        abs_threshold = [+np.inf, -np.inf]
    order = np.round(interval_in_sec / dt).astype(int)
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Npars = len(parameters)
    Nticks = len(s.index.unique('Step'))
    t0 = s.index.unique('Step').min()

    min_array = np.ones([Nticks, Npars, Nids]) * np.nan
    max_array = np.ones([Nticks, Npars, Nids]) * np.nan

    for i, p in enumerate(parameters):
        p_min, p_max = nam.min(p), nam.max(p)
        s[p_min] = np.nan
        s[p_max] = np.nan
        d = s[p]
        std = d.std()
        mu = d.mean()
        if threshold_in_std is not None:
            thr_min = mu - threshold_in_std * std
            thr_max = mu + threshold_in_std * std
        else:
            thr_min, thr_max = abs_threshold
        for j, id in enumerate(ids):
            df = d.xs(id, level='AgentID', drop_level=True)
            i_min = argrelextrema(df.values, np.less_equal, order=order)[0]
            i_max = argrelextrema(df.values, np.greater_equal, order=order)[0]

            i_min_dif = np.diff(i_min, append=order)
            i_max_dif = np.diff(i_max, append=order)
            i_min = i_min[i_min_dif >= order]
            i_max = i_max[i_max_dif >= order]

            i_min = i_min[df.loc[i_min + t0] < thr_min]
            i_max = i_max[df.loc[i_max + t0] > thr_max]

            min_array[i_min, i, j] = True
            max_array[i_max, i, j] = True

        s[p_min] = min_array[:, i, :].flatten()
        s[p_max] = max_array[:, i, :].flatten()

def compute_freq(s, e, dt, parameters, freq_range=None, compare_params=False):
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Npars = len(parameters)
    V = np.zeros(Npars)
    F = np.ones((Npars, Nids)) * np.nan
    for i, p in enumerate(parameters):
        for j, id in enumerate(ids):
            d = s[p].xs(id, level='AgentID', drop_level=True)
            try:
                f, t, Sxx = spectrogram(d, fs=1 / dt)
                # keep only frequencies of interest
                if freq_range:
                    f0, f1 = freq_range
                    valid = np.where((f >= f0) & (f <= f1))
                    f = f[valid]
                    Sxx = Sxx[valid, :][0]
                max_freq = f[np.argmax(np.nanmedian(Sxx, axis=1))]
            except:
                max_freq = np.nan
            F[i, j] = max_freq
    if compare_params:
        ind = np.argmax(V)
        best_p = parameters[ind]
        existing = fun.common_member(nam.freq(parameters), e.columns.values)
        e.drop(columns=existing, inplace=True)
        e[nam.freq(best_p)] = F[ind]
    else:
        for i, p in enumerate(parameters):
            e[nam.freq(p)] = F[i]
    # if is_last:
    #     self.save()

def filter(s, dt, Npoints, config, freq=2, N=1, inplace=True, recompute=False,**kwargs):
    if freq in ['', None, np.nan]:
        return
    if 'filtered_at' in config and not recompute :
        print(f'Dataset already filtered at {config["filtered_at"]}. If you want to apply additional filter set recompute to True')
        return
    config['filtered_at'] = freq

    points = nam.midline(Npoints, type='point') + ['centroid', '']
    pars = nam.xy(points, flat=True)
    pars = [p for p in pars if p in s.columns]
    data = np.dstack(list(s[pars].groupby('AgentID').apply(pd.DataFrame.to_numpy)))
    f_array = fun.apply_filter_to_array_with_nans_multidim(data, freq=freq, fr=1/dt, N=N)
    fpars = nam.filt(pars) if not inplace else pars
    for j, p in enumerate(fpars):
        s[p] = f_array[:, j, :].flatten()
    print(f'All spatial parameters filtered at {freq} Hz')

def interpolate_nan_values(s, config, pars=None,**kwargs):
    if pars is None :
        N= config['Npoints'],
        Nc= config['Ncontour'],
        points = nam.midline(N, type='point') + ['centroid', ''] + nam.contour(Nc)
        pars = nam.xy(points, flat=True)
    pars = [p for p in pars if p in s.columns]
    for p in pars:
        for id in s.index.unique('AgentID').values:
            s.loc[(slice(None), id), p] = fun.interpolate_nans(s[p].xs(id, level='AgentID', drop_level=True).values)
    print('All parameters interpolated')

def rescale(s,e, config,Npoints=None,Ncontour=None, recompute=False, scale=1.0,**kwargs):
    if Npoints is None :
        Npoints=config['Npoints']
    if Ncontour is None :
        Ncontour=config['Ncontour']
    if scale in ['', None, np.nan]:
        return
    if 'rescaled_by' in config and not recompute :
        print(f'Dataset already rescaled by {config["rescaled_by"]}. If you want to rescale again set recompute to True')
        return
    config['rescaled_by'] = scale
    points = nam.midline(Npoints, type='point') + ['centroid','']
    contour_pars=nam.xy(nam.contour(Ncontour), flat=True)
    pars=nam.xy(points, flat=True) + nam.dst(points) + nam.vel(points) + nam.acc(points) + ['spinelength']+contour_pars
    lin_pars = [p for p in pars if p in s.columns]
    for p in lin_pars:
        s[p] = s[p].apply(lambda x: x * scale)
    if 'length' in e.columns:
        e['length'] = e['length'].apply(lambda x: x * scale)
    print(f'Dataset rescaled by {scale}.')

def exclude_rows(s,e, dt,  flag, accepted=None, rejected=None,**kwargs):
        if accepted is not None:
            s.loc[s[flag] != accepted[0]] = np.nan
        if rejected is not None:
            s.loc[s[flag] == rejected[0]] = np.nan

        p=getPar('cum_t', to_return=['d'])[0]
        for id in s.index.unique('AgentID').values:
            e.loc[id, p] = len(s.xs(id, level='AgentID', drop_level=True).dropna()) * dt

        print(f'Rows excluded according to {flag}.')

def preprocess(s,e,config, rescale_by=None,drop_collisions=False,interpolate_nans=False,filter_f=None,transposition=None,
               recompute=False,show_output=True,**kwargs) :
    c = {
        's': s,
        'e': e,
        'dt': config['dt'],
        'Npoints': config['Npoints'],
        'Ncontour': config['Ncontour'],
        'point': config['point'],
        'recompute': recompute,
        'config': config,
    }
    with fun.suppress_stdout(show_output):
        if rescale_by is not None :
            rescale(scale=rescale_by, **c)
        if drop_collisions :
            exclude_rows(flag='collision_flag', accepted=[0], **c)
        if interpolate_nans :
            interpolate_nan_values(**c)
        if filter_f is not None :
            filter(req=filter_f, **c)
        if transposition is not None :
            align_trajectories(mode=transposition, **c)
        return s,e

def generate_traj_colors(s, sp_vel=None, ang_vel=None):
    N = len(s.index.unique('Step'))
    if sp_vel is None :
        sp_vel =getPar('sv', to_return=['d'])[0]
    if ang_vel is None :
        ang_vel =getPar('fov', to_return=['d'])[0]
    pars = [sp_vel, ang_vel]
    edge_colors = [[(255, 0, 0), (0, 255, 0)], [(255, 0, 0), (0, 255, 0)]]
    labels = ['lin_color', 'ang_color']
    lims = [0.8, 300]
    for p, c, l, lim in zip(pars, edge_colors, labels, lims):
        if p in s.columns:
            (r1, b1, g1), (r2, b2, g2) = c
            r, b, g = r2 - r1, b2 - b1, g2 - g1
            temp = np.clip(s[p].abs().values / lim, a_min=0, a_max=1)
            s[l] = [(r1 + r * t, b1 + b * t, g1 + g * t) for t in temp]
        else:
            s[l] = [(np.nan, np.nan, np.nan)] * N
    return s

def process(s,e,config,mode='minimal',traj_colors=True,show_output=True,
            types={'angular':True, 'spatial':True, 'source':True, 'dispersion':True, 'tortuosity':True, 'PI':True},
            source=None,dsp_starts=[0], dsp_stops=[40], tor_durs=[2, 5, 10, 20],  **kwargs):
    c = {
        's': s,
        'e': e,
        'dt': config['dt'],
        'Npoints': config['Npoints'],
        'Ncontour': config['Ncontour'],
        'point': config['point'],
        'config': config,
        'mode': mode,
        'aux_dir': f'{config["dir"]}/data/aux.h5',
    }

    with fun.suppress_stdout(show_output):
        if types['angular']:
            angular_processing(**c, **kwargs)
        if types['spatial']:
            spatial_processing(**c, **kwargs)
        if types['source']:
            if source is not None:
                compute_bearingNdst2source(s, e, source=source, **kwargs)
        if types['dispersion'] and type(dsp_starts)==list and type(dsp_stops)==list:
            compute_dispersion(**c, starts=dsp_starts, stops=dsp_stops, **kwargs)
        if types['tortuosity'] and type(tor_durs)==list:
            compute_tortuosity(**c, durs_in_sec=tor_durs, **kwargs)
        if types['PI']:
            if 'x' in e.keys() :
                px = 'x'
                xs = e[px].values
            elif nam.final('x') in e.keys() :
                px = nam.final('x')
                xs = e[px].values
            elif 'x' in s.keys() :
                px = 'x'
                xs = s[px].dropna().groupby('AgentID').last().values
            elif 'centroid_x' in s.keys() :
                px = 'centroid_x'
                xs = s[px].dropna().groupby('AgentID').last().values
            else :
                raise ValueError ('No x coordinate found')
            PI, N, N_l, N_r = compute_preference_index(xs=xs, arena_xdim=config['env_params']['arena']['arena_dims'][0], return_num=True, return_all=True)
            config['PI']={'PI':PI, 'N':N, 'N_l':N_l, 'N_r':N_r}
        if traj_colors :
            try :
                generate_traj_colors(s=s, sp_vel=None, ang_vel=None)
            except :
                pass
        return s,e


