import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, spectrogram

import lib.aux.functions as fun
import lib.aux.naming as nam
import lib.conf.dtype_dicts as dtypes
from lib.anal.process.store import store_aux_dataset


def compute_spineangles(s, angles, points, config=None, chunk_only=None, mode='full'):
    r = config['front_body_ratio'] if config is not None else 0.5
    bend_angles = angles[:int(np.round(r * len(angles)))]
    if chunk_only is not None:
        print(f'Computation restricted to {chunk_only} chunks')
        s = s.loc[s[nam.id(chunk_only)].dropna().index.values].copy(deep=False)
    xy = [nam.xy(points[i]) for i in range(len(points))]
    if mode == 'full':
        angles = angles
    elif mode == 'minimal':
        angles = bend_angles
    N = len(angles)
    print(f'Computing {N} angles')
    xy_pars = fun.flatten_list([xy[i] for i in range(N + 2)])
    xy_ar = s[xy_pars].values
    Npoints = int(xy_ar.shape[1] / 2)
    Nticks = xy_ar.shape[0]
    xy_ar = np.reshape(xy_ar, (Nticks, Npoints, 2))
    c = np.zeros([N, Nticks]) * np.nan
    for i in range(Nticks):
        c[:, i] = np.array([fun.angle(xy_ar[i, j + 2, :], xy_ar[i, j + 1, :], xy_ar[i, j, :]) for j in range(N)])
    for z, a in enumerate(angles):
        s[a] = c[z].T
    print('All angles computed')
    return bend_angles


def compute_bend(s, points, angles, config=None, mode='minimal'):
    b_conf = config['bend'] if config is not None else 'from_angles'
    if b_conf is None:
        print('Bending angle not defined. Can not compute angles')
        return
    elif b_conf == 'from_vectors':
        print(f'Computing bending angle as the difference between front and rear orients')
        s['bend'] = s.apply(lambda r: fun.angle_dif(r[nam.orient('front')], r[nam.orient('rear')]), axis=1)
    elif b_conf == 'from_angles':
        bend_angles = compute_spineangles(s, angles, points, config, mode=mode)
        print(f'Computing bending angle as the sum of the first {len(bend_angles)} front angles')
        s['bend'] = s[bend_angles].sum(axis=1, min_count=1)

    print('All bends computed')


def compute_LR_bias(s, e):
    for id in s.index.unique('AgentID').values:
        b = s['bend'].xs(id, level='AgentID', drop_level=True).dropna()
        bv = s[nam.vel('bend')].xs(id, level='AgentID', drop_level=True).dropna()
        e.loc[id, 'bend_mean'] = b.mean()
        e.loc[id, 'bend_vel_mean'] = bv.mean()
        e.loc[id, 'bend_std'] = b.std()
        e.loc[id, 'bend_vel_std'] = bv.std()
    print('LR biases computed')


def compute_orientations(s,e, points, segs, config=None, mode='full'):
    if config is None:
        f1, f2 = 1, 2
        r1, r2 = -2, -1
    else:
        for key in ['front_vector', 'rear_vector']:
            if config[key] is None:
                print('Front and rear vectors are not defined. Can not compute orients')
                return
        else:
            f1, f2 = config['front_vector']
            r1, r2 = config['rear_vector']

    xy = [nam.xy(points[i]) for i in range(len(points))]
    print(f'Computing front and rear orients')
    xy_pars = fun.flatten_list([xy[i] for i in [f2 - 1, f1 - 1, r2 - 1, r1 - 1]])
    xy_ar = s[xy_pars].values
    Npoints = int(xy_ar.shape[1] / 2)
    Nticks = xy_ar.shape[0]
    xy_ar = np.reshape(xy_ar, (Nticks, Npoints, 2))

    c = np.zeros([2, Nticks]) * np.nan
    for i in range(Nticks):
        c[:, i] = np.array([fun.angle_to_x_axis(xy_ar[i, 2 * j, :], xy_ar[i, 2 * j + 1, :]) for j in range(2)])
    for z, a in enumerate([nam.orient('front'), nam.orient('rear')]):
        s[a] = c[z].T
        e[nam.initial(a)] = s[a].dropna().groupby('AgentID').first()
    if mode == 'full':
        N = len(segs)
        print(f'Computing additional orients for {N} spinesegments')
        ors = nam.orient(segs)
        xy_pars = fun.flatten_list([xy[i] for i in range(N + 1)])
        xy_ar = s[xy_pars].values
        Npoints = int(xy_ar.shape[1] / 2)
        Nticks = xy_ar.shape[0]
        xy_ar = np.reshape(xy_ar, (Nticks, Npoints, 2))
        c = np.zeros([N, Nticks]) * np.nan
        for i in range(Nticks):
            c[:, i] = np.array([fun.angle_to_x_axis(xy_ar[i, j + 1, :], xy_ar[i, j, :]) for j in range(N)])
        for z, a in enumerate(ors):
            s[a] = c[z].T
    print('All orientations computed')
    return


def unwrap_orientations(s, segs):
    pars = list(set([p for p in [nam.orient('front'), nam.orient('rear')] + nam.orient(segs) if p in s.columns.values]))
    for p in pars:
        for id in s.index.unique('AgentID').values:
            ts = s.loc[(slice(None), id), p].values
            s.loc[(slice(None), id), nam.unwrap(p)] = fun.unwrap_deg(ts)
    print('All orients unwrapped')


def compute_angular_metrics(s, dt, segs, angles, mode='minimal'):
    ang_pars = [nam.orient('front'), nam.orient('rear'), 'bend']
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Nticks = len(s.index.unique('Step'))
    unwrap_orientations(s, segs)

    if mode == 'full':
        pars = angles + nam.orient(segs) + ang_pars
    elif mode == 'minimal':
        pars = ang_pars

    pars = [a for a in pars if a in s.columns]
    Npars = len(pars)
    print(f'Computing angular velocities and accelerations for {Npars} angular parameters')

    V = np.zeros([Nticks, Npars, Nids]) * np.nan
    A = np.zeros([Nticks, Npars, Nids]) * np.nan

    all_d = [s.xs(id, level='AgentID', drop_level=True) for id in ids]

    vels = nam.vel(pars)
    accs = nam.acc(pars)

    for i, p in enumerate(pars):
        if nam.unwrap(p) in s.columns:
            p = nam.unwrap(p)
        for j, d in enumerate(all_d):
            angle = d[p].values
            avel = np.diff(angle) / dt
            aacc = np.diff(avel) / dt
            V[1:, i, j] = avel
            A[2:, i, j] = aacc
    for k, (v, a) in enumerate(zip(vels, accs)):
        s[v] = V[:, k, :].flatten()
        s[a] = A[:, k, :].flatten()
    print('All angular parameters computed')


def angular_processing(s, e, config, dt, Npoints, aux_dir, recompute=False, mode='minimal', **kwargs):
    N = Npoints
    points = nam.midline(N, type='point')
    Nangles = np.clip(N - 2, a_min=0, a_max=None)
    angles = [f'angle{i}' for i in range(Nangles)]
    Nsegs = np.clip(N - 1, a_min=0, a_max=None)
    segs = nam.midline(Nsegs, type='seg')

    ang_pars = [nam.orient('front'), nam.orient('rear'), 'bend']
    if set(ang_pars).issubset(s.columns.values) and not recompute:
        print('Orientation and bend are already computed. If you want to recompute them, set recompute to True')
    else:
        compute_orientations(s,e, points, segs, config, mode=mode)
        compute_bend(s, points, angles, config, mode=mode)
    compute_angular_metrics(s, dt, segs, angles, mode=mode)
    compute_LR_bias(s, e)
    # if distro_dir is not None:
    #     create_par_distro_dataset(s, ang_pars + nam.vel(ang_pars) + nam.acc(ang_pars), dir=distro_dir)
    store_aux_dataset(s, pars=ang_pars + nam.vel(ang_pars) + nam.acc(ang_pars), type='distro', file=aux_dir)
    print(f'Completed {mode} angular processing.')
    return s,e



