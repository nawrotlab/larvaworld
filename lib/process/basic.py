import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, spectrogram

from lib.conf.pars.pars import getPar, ParDict
from lib.process.aux import interpolate_nans, suppress_stdout
from lib.aux.sim_aux import apply_filter_to_array_with_nans_multidim
# from lib.aux.dictsNlists import common_member
import lib.aux.naming as nam
from lib.process.angular import angular_processing
from lib.process.spatial import spatial_processing, comp_source_metrics, comp_dispersion, comp_PI, \
    align_trajectories, comp_wind_metrics, comp_final_anemotaxis, comp_PI2, comp_straightness_index


def comp_extrema(s, dt, parameters, interval_in_sec, threshold_in_std=None, abs_threshold=None):
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

def filter(s, dt, Npoints, c, freq=2, N=1, inplace=True, recompute=False, **kwargs):
    if freq in ['', None, np.nan]:
        return
    if 'filtered_at' in c and not recompute:
        print(
            f'Dataset already filtered at {c["filtered_at"]}. If you want to apply additional filter set recompute to True')
        return
    c['filtered_at'] = freq

    points = nam.midline(Npoints, type='point') + ['centroid', '']
    pars = nam.xy(points, flat=True)
    pars = [p for p in pars if p in s.columns]
    data = np.dstack(list(s[pars].groupby('AgentID').apply(pd.DataFrame.to_numpy)))
    f_array = apply_filter_to_array_with_nans_multidim(data, freq=freq, fr=1 / dt, N=N)
    fpars = nam.filt(pars) if not inplace else pars
    for j, p in enumerate(fpars):
        s[p] = f_array[:, j, :].flatten()
    print(f'All spatial parameters filtered at {freq} Hz')


def interpolate_nan_values(s, config, pars=None, **kwargs):
    if pars is None:
        N = config['Npoints'],
        Nc = config['Ncontour'],
        points = nam.midline(N[0], type='point') + ['centroid', ''] + nam.contour(Nc[0]) # changed from N and Nc to N[0] and Nc[0] as comma above was turning them into tuples, which the naming function does not accept.
        pars = nam.xy(points, flat=True)
    pars = [p for p in pars if p in s.columns]
    for p in pars:
        for id in s.index.unique('AgentID').values:
            s.loc[(slice(None), id), p] = interpolate_nans(s[p].xs(id, level='AgentID', drop_level=True).values)
    print('All parameters interpolated')


def rescale(s, e, c, Npoints=None, Ncontour=None, recompute=False, scale=1.0, **kwargs):
    if Npoints is None:
        Npoints = c['Npoints']
    if Ncontour is None:
        Ncontour = c['Ncontour']
    if scale in ['', None, np.nan]:
        return
    if 'rescaled_by' in c and not recompute:
        print(
            f'Dataset already rescaled by {c["rescaled_by"]}. If you want to rescale again set recompute to True')
        return
    c['rescaled_by'] = scale
    points = nam.midline(Npoints, type='point') + ['centroid', '']
    contour_pars = nam.xy(nam.contour(Ncontour), flat=True)
    pars = nam.xy(points, flat=True) + nam.dst(points) + nam.vel(points) + nam.acc(points) + [
        'spinelength'] + contour_pars
    lin_pars = [p for p in pars if p in s.columns]
    for p in lin_pars:
        s[p] = s[p].apply(lambda x: x * scale)
    if 'length' in e.columns:
        e['length'] = e['length'].apply(lambda x: x * scale)
    print(f'Dataset rescaled by {scale}.')


def exclude_rows(s, e, dt, flag, accepted=None, rejected=None, **kwargs):
    if accepted is not None:
        s.loc[s[flag] != accepted[0]] = np.nan
    if rejected is not None:
        s.loc[s[flag] == rejected[0]] = np.nan

    for id in s.index.unique('AgentID').values:
        e.loc[id, getPar('cum_t')] = len(s.xs(id, level='AgentID', drop_level=True).dropna()) * dt

    print(f'Rows excluded according to {flag}.')


def preprocess(s, e, c, rescale_by=None, drop_collisions=False, interpolate_nans=False, filter_f=None,
               transposition=None,recompute=False, show_output=True, **kwargs):
    cc = {
        's': s,
        'e': e,
        'dt': c.dt,
        'Npoints': c.Npoints,
        'Ncontour': c.Ncontour,
        'point': c.point,
        'recompute': recompute,
        'c': c,
    }
    with suppress_stdout(show_output):
        if rescale_by is not None:
            rescale(scale=rescale_by, **cc)
        if drop_collisions:
            exclude_rows(flag='collision_flag', accepted=[0], **cc)
        if interpolate_nans:
            interpolate_nan_values(**cc)
        if filter_f is not None:
            filter(freq=filter_f, **cc)
        if transposition is not None:
            align_trajectories(mode=transposition, **cc)
        return s, e


def generate_traj_colors(s, sp_vel=None, ang_vel=None):
    N = len(s.index.unique('Step'))
    if sp_vel is None:
        sp_vel = getPar('sv')
    if ang_vel is None:
        ang_vel = getPar('fov')
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


def process(processing, s, e, c, mode='minimal', traj_colors=True, show_output=True, **kwargs):
    # print(processing)
    cc = {
        's': s,
        'e': e,
        'c': c,
        'dt': c.dt,
        'Npoints': c.Npoints,
        'Ncontour': c.Ncontour,
        'point': c.point,
        'mode': mode,
        **kwargs
    }

    with suppress_stdout(show_output):
        if processing['angular']:
            angular_processing(**cc)
        if processing['spatial']:
            spatial_processing(**cc)
        if processing['source']:
            comp_source_metrics(**cc)
        if processing['wind']:
            if processing['spatial']:
                comp_wind_metrics(**cc)
            else :
                comp_final_anemotaxis(**cc)
        if processing['dispersion'] :
            # raise
            comp_dispersion(**cc)
        if processing['tortuosity'] :
            # comp_tortuosity(**cc)
            comp_straightness_index(**cc)
        if processing['PI']:
            if 'x' in e.keys():
                px = 'x'
                xs = e[px].values
            elif nam.final('x') in e.keys():
                px = nam.final('x')
                xs = e[px].values
            elif 'x' in s.keys():
                px = 'x'
                xs = s[px].dropna().groupby('AgentID').last().values
            elif 'centroid_x' in s.keys():
                px = 'centroid_x'
                xs = s[px].dropna().groupby('AgentID').last().values
            else:
                raise ValueError('No x coordinate found')
            PI, N, N_l, N_r = comp_PI(xs=xs, arena_xdim=c.env_params.arena.arena_dims[0], return_num=True,
                                      return_all=True)
            c.PI = {'PI': PI, 'N': N, 'N_l': N_l, 'N_r': N_r}
            try :
                c.PI2 = comp_PI2(xys=s[nam.xy('')], arena_xdim=c.env_params.arena.arena_dims[0])
            except :
                pass
        if traj_colors:
            try:
                generate_traj_colors(s=s, sp_vel=None, ang_vel=None)
            except:
                pass
        return s, e


