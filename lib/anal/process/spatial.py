import itertools
from sklearn.metrics.pairwise import nan_euclidean_distances
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema, spectrogram

import lib.aux.functions as fun
import lib.aux.naming as nam
import lib.conf.dtype_dicts as dtypes
from lib.anal.process.store import create_dispersion_dataset


def raw_or_filtered_xy(s, points):
    r = nam.xy(points, flat=True)
    f = nam.filt(r)
    if all(i in s.columns for i in f):
        print('Using filtered xy coordinates')
        return f
    elif all(i in s.columns for i in r):
        print('Using raw xy coordinates')
        return r
    else:
        print('No xy coordinates exist. Not computing spatial metrics')
        return

def compute_linear_metrics(s, e, dt, Npoints,point,mode='minimal'):
    points = nam.midline(Npoints, type='point')
    Nsegs = np.clip(Npoints - 1, a_min=0, a_max=None)
    segs = nam.midline(Nsegs, type='seg')

    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Nticks = len(s.index.unique('Step'))
    if 'length' in e.columns.values:
        lengths = e['length'].values
    else:
        lengths = None

    if mode == 'full':
        print(f'Computing linear distances, velocities and accelerations for {len(points) - 1} points')
        points = points[1:]
        orientations = nam.orient(segs)
    elif mode == 'minimal':
        if point == 'centroid' or point == points[0]:
            print('Defined point is either centroid or head. Orientation of front segment not defined.')
            return
        else:
            print(f'Computing linear distances, velocities and accelerations for a single spinepoint')
            points = [point]
            orientations = ['rear_orientation']

    if not set(orientations).issubset(s.columns):
        print('Required orients not found. Component linear metrics not computed.')
        return

    xy_params = raw_or_filtered_xy(s, points)
    xy_params = fun.group_list_by_n(xy_params, 2)

    all_d = [s.xs(id, level='AgentID', drop_level=True) for id in ids]
    dsts = nam.lin(nam.dst(points))
    cum_dsts = nam.cum(nam.lin(dsts))
    vels = nam.lin(nam.vel(points))
    accs = nam.lin(nam.acc(points))

    for p, xy, dst, cum_dst, vel, acc, orient in zip(points, xy_params, dsts, cum_dsts, vels, accs, orientations):
        D = np.zeros([Nticks, Nids]) * np.nan
        Dcum = np.zeros([Nticks, Nids]) * np.nan
        V = np.zeros([Nticks, Nids]) * np.nan
        A = np.zeros([Nticks, Nids]) * np.nan
        sD = np.zeros([Nticks, Nids]) * np.nan
        sDcum = np.zeros([Nticks, Nids]) * np.nan
        sV = np.zeros([Nticks, Nids]) * np.nan
        sA = np.zeros([Nticks, Nids]) * np.nan

        for i, data in enumerate(all_d):
            v, d = fun.compute_component_velocity(xy=data[xy].values, angles=data[orient].values, dt=dt,
                                                  return_dst=True)
            a = np.diff(v) / dt
            cum_d = np.nancumsum(d)
            D[1:, i] = d
            Dcum[1:, i] = cum_d
            V[1:, i] = v
            A[2:, i] = a
            if lengths is not None:
                l = lengths[i]
                sD[1:, i] = d / l
                sDcum[1:, i] = cum_d / l
                sV[1:, i] = v / l
                sA[2:, i] = a / l

        s[dst] = D.flatten()
        s[cum_dst] = Dcum.flatten()
        s[vel] = V.flatten()
        s[acc] = A.flatten()
        e[nam.cum(dst)] = Dcum[-1, :]

        if lengths is not None:
            s[nam.scal(dst)] = sD.flatten()
            s[nam.cum(nam.scal(dst))] = sDcum.flatten()
            s[nam.scal(vel)] = sV.flatten()
            s[nam.scal(acc)] = sA.flatten()
            e[nam.cum(nam.scal(dst))] = sDcum[-1, :]

    print('All linear parameters computed')

def compute_spatial_metrics(s, e, dt, Npoints,point, mode='minimal'):
    points = nam.midline(Npoints, type='point')
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Nticks = len(s.index.unique('Step'))
    if 'length' in e.columns.values:
        lengths = e['length'].values
    else:
        lengths = None

    if mode == 'full':
        print(f'Computing distances, velocities and accelerations for {len(points)} points')
        points += ['centroid']
    elif mode == 'minimal':
        print(f'Computing distances, velocities and accelerations for a single spinepoint')
        points = [point]



    points = np.unique(points).tolist()
    points = [p for p in points if set(nam.xy(p)).issubset(s.columns.values)]

    xy_params = raw_or_filtered_xy(s, points)
    xy_params = fun.group_list_by_n(xy_params, 2)

    all_d = [s.xs(id, level='AgentID', drop_level=True) for id in ids]
    dsts = nam.dst(points)
    cum_dsts = nam.cum(dsts)
    vels = nam.vel(points)
    accs = nam.acc(points)

    for p, xy, dst, cum_dst, vel, acc in zip(points, xy_params, dsts, cum_dsts, vels, accs):
        D = np.zeros([Nticks, Nids]) * np.nan
        Dcum = np.zeros([Nticks, Nids]) * np.nan
        V = np.zeros([Nticks, Nids]) * np.nan
        A = np.zeros([Nticks, Nids]) * np.nan
        sD = np.zeros([Nticks, Nids]) * np.nan
        sDcum = np.zeros([Nticks, Nids]) * np.nan
        sV = np.zeros([Nticks, Nids]) * np.nan
        sA = np.zeros([Nticks, Nids]) * np.nan

        for i, data in enumerate(all_d):
            v, d = fun.compute_velocity(xy=data[xy].values, dt=dt, return_dst=True)
            a = np.diff(v) / dt
            cum_d = np.nancumsum(d)

            D[1:, i] = d
            Dcum[1:, i] = cum_d
            V[1:, i] = v
            A[2:, i] = a
            if lengths is not None:
                l = lengths[i]
                sD[1:, i] = d / l
                sDcum[1:, i] = cum_d / l
                sV[1:, i] = v / l
                sA[2:, i] = a / l

        s[dst] = D.flatten()
        s[cum_dst] = Dcum.flatten()
        s[vel] = V.flatten()
        s[acc] = A.flatten()
        e[nam.cum(dst)] = Dcum[-1, :]

        if lengths is not None:
            s[nam.scal(dst)] = sD.flatten()
            s[nam.cum(nam.scal(dst))] = sDcum.flatten()
            s[nam.scal(vel)] = sV.flatten()
            s[nam.scal(acc)] = sA.flatten()
            e[nam.cum(nam.scal(dst))] = sDcum[-1, :]

    print('All spatial parameters computed')

def compute_length(s, e, Npoints, mode='minimal',recompute=False):
    if 'length' in e.columns.values and not recompute:
        print('Length is already computed. If you want to recompute it, set recompute_length to True')
        return
    points = nam.midline(Npoints, type='point')
    xy_pars=nam.xy(points, flat=True)
    if not set(xy_pars).issubset(s.columns):
        print(f'XY coordinates not found for the {Npoints} midline points. Body length can not be computed.')
        return
    Nsegs = np.clip(Npoints - 1, a_min=0, a_max=None)
    segs = nam.midline(Nsegs, type='seg')
    t = len(s)
    xy = s[xy_pars].values
    L = np.zeros([1, t]) * np.nan
    S = np.zeros([Nsegs, t]) * np.nan

    if mode == 'full':
        print(f'Computing lengths for {Nsegs} segments and total body length')
        for j in range(xy.shape[0]):
            for i, seg in enumerate(segs):
                S[i, j] = np.sqrt(np.nansum((xy[j, 2 * i:2 * i + 2] - xy[j, 2 * i + 2:2 * i + 4]) ** 2))
            L[:, j] = np.nansum(S[:, j])
        for i, seg in enumerate(segs):
            s[seg] = S[i, :].flatten()
    elif mode == 'minimal':
        print(f'Computing body length')
        for j in range(xy.shape[0]):
            k = np.sum(np.diff(np.array(fun.group_list_by_n(xy[j, :], 2)), axis=0) ** 2, axis=1).T
            L[:, j] = np.sum([np.sqrt(kk) for kk in k]) if not np.isnan(np.sum(k)) else np.nan
    s['length'] = L.flatten()
    e['length'] = s['length'].groupby('AgentID').quantile(q=0.5)
    print('All lengths computed.')

def compute_centroid_from_contour(s, Ncontour, recompute=False):
    if set(nam.xy('centroid')).issubset(s.columns.values) and not recompute:
        print('Centroid is already computed. If you want to recompute it, set recompute_centroid to True')
    contour = nam.contour(Ncontour)
    con_pars = nam.xy(contour, flat=True)
    if not set(con_pars).issubset(s.columns) or Ncontour == 0:
        print(f'No contour found. Not computing centroid')
    else:
        print(f'Computing centroid from {Ncontour} contourpoints')
        contour = s[con_pars].values
        N = contour.shape[0]
        contour = np.reshape(contour, (N, Ncontour, 2))
        c = np.zeros([N, 2]) * np.nan
        for i in range(N):
            c[i, :] = np.array(fun.compute_centroid(contour[i, :, :]))
        s[nam.xy('centroid')[0]] = c[:, 0]
        s[nam.xy('centroid')[1]] = c[:, 1]
    print('Centroid coordinates computed.')

def store_global_linear_metrics(s,e, point):
    ids = s.index.unique('AgentID').values
    dic = {
        'x': nam.xy(point)[0],
        'y': nam.xy(point)[1],
        'dst': nam.dst(point),
        'vel': nam.vel(point),
        'acc': nam.acc(point),
        nam.scal('dst'): nam.scal(nam.dst(point)),
        nam.scal('vel'): nam.scal(nam.vel(point)),
        nam.scal('acc'): nam.scal(nam.acc(point)),
        nam.cum('dst'): nam.cum(nam.dst(point)),
        nam.cum(nam.scal('dst')): nam.cum(nam.scal(nam.dst(point)))}
    for k, v in dic.items():
        try:
            s[k] = s[v]
        except:
            pass
    e[nam.cum('dst')] = e[nam.cum(nam.dst(point))]
    e[nam.final('x')] = [s['x'].xs(id, level='AgentID').dropna().values[-1] for id in ids]
    e[nam.final('y')] = [s['y'].xs(id, level='AgentID').dropna().values[-1] for id in ids]
    e[nam.initial('x')] = [s['x'].xs(id, level='AgentID').dropna().values[0] for id in ids]
    e[nam.initial('y')] = [s['y'].xs(id, level='AgentID').dropna().values[0] for id in ids]
    e[nam.mean('vel')] = e[nam.cum(nam.dst(point))] / e['cum_dur']
    try:
        e[nam.cum(nam.scal('dst'))] = e[nam.cum(nam.scal(nam.dst(point)))]
        e[nam.mean(nam.scal('vel'))] = e[nam.mean('vel')] / e['length']
    except:
        pass

def spatial_processing(s, e, dt, Npoints, point, Ncontour, mode='minimal', recompute=False,
                       dsp_starts=[0], dsp_stops=[40], dsp_dir=None, tor_durs=[2, 5, 10, 20],source=None, **kwargs):

    compute_length(s, e, Npoints, mode=mode, recompute=recompute)
    compute_centroid_from_contour(s, Ncontour, recompute=recompute)
    compute_spatial_metrics(s, e, dt, Npoints,point,mode=mode)
    compute_linear_metrics(s, e, dt, Npoints,point,mode=mode)
    store_global_linear_metrics(s,e, point)
    compute_dispersion(s, e, dt, point, recompute=recompute, starts=dsp_starts, stops=dsp_stops, dir=dsp_dir)
    compute_tortuosity(s, e, dt, durs_in_sec=tor_durs)
    if source is not None:
        compute_orientation_to_origin(s,e,point, origin=source)
        compute_dst_to_origin(s,e,dt,point, origin=source)
    print(f'Completed {mode} spatial processing.')
    return s,e


def compute_dispersion(s,e,dt, point, recompute=False, starts=[0], stops=[40], dir=None):
    ids = s.index.unique('AgentID').values
    for s0, s1 in itertools.product(starts, stops):
        if s0 == 0 and s1 == 40:
            p = f'dispersion'
        else:
            p = f'dispersion_{s0}_{s1}'

        t0 = int(s0 / dt)
        # p40 = f'40sec_{p}'
        fp = nam.final(p)
        # fp, fp40 = nam.final([p, p40])
        mp = nam.max(p)
        # mp, mp40 = nam.max([p, p40])
        mup = nam.mean(p)

        if set([mp]).issubset(e.columns.values) and not recompute:
            print(f'Dispersion in {s0}-{s1} sec is already detected. If you want to recompute it, set recompute_dispersion to True')
            continue
        print(f'Computing dispersion in {s0}-{s1} sec based on {point}')
        for id in ids:
            xy = s[['x', 'y']].xs(id, level='AgentID', drop_level=True)
            try:
                origin_xy = list(xy.dropna().values[t0])
            except:
                print(f'No values to set origin point for {id}')
                s.loc[(slice(None), id), p] = np.empty(len(xy)) * np.nan
                continue
            d = nan_euclidean_distances(list(xy.values), [origin_xy])[:, 0]
            d[:t0] = np.nan
            s.loc[(slice(None), id), p] = d
            e.loc[id, mp] = np.nanmax(d)
            # e.loc[id, mp40] = np.nanmax(d[:int(40 / self.dt)])
            # e.loc[id, fp40] = d[int(40 / self.dt)]
            e.loc[id, mup] = np.nanmean(d)
            e.loc[id, fp] = s[p].xs(id, level='AgentID').dropna().values[-1]

            try:
                l = e.loc[id, 'length']
                s.loc[(slice(None), id), nam.scal(p)] = d / l
                e.loc[id, nam.scal(mp)] = e.loc[id, mp] / l
                # e.loc[id, nam.scal(mp40)] = e.loc[id, mp40] / l
                # e.loc[id, nam.scal(fp40)] = e.loc[id, fp40] / l
                e.loc[id, nam.scal(mup)] = e.loc[id, mup] / l
                e.loc[id, nam.scal(fp)] = e.loc[id, fp] / l
            except:
                pass
        create_dispersion_dataset(s,par=p, scaled=True, dir=dir)
        create_dispersion_dataset(s,par=p, scaled=False, dir=dir)
    print('Dispersions computed')

def compute_tortuosity(s,e,dt, durs_in_sec=[2, 5, 10, 20]):
    e['tortuosity'] = 1 - e['final_dispersion'] / e['cum_dst']
    durs = [int(1/dt * d) for d in durs_in_sec]
    Ndurs = len(durs)
    if Ndurs > 0:
        ids = s.index.unique('AgentID').values
        Nids = len(ids)
        ds = [s[['x', 'y']].xs(id, level='AgentID') for id in ids]
        ds = [d.loc[d.first_valid_index(): d.last_valid_index()].values for d in ds]
        for j, r in enumerate(durs):
            par = f'tortuosity_{durs_in_sec[j]}'
            par_m, par_s = nam.mean(par), nam.std(par)
            T_m = np.ones(Nids) * np.nan
            T_s = np.ones(Nids) * np.nan
            for z, id in enumerate(ids):
                si = ds[z]
                u = len(si) % r
                if u > 1:
                    si0 = si[:-u + 1]
                else:
                    si0 = si[:-r + 1]
                k = int(len(si0) / r)
                T = []
                for i in range(k):
                    t = si0[i * r:i * r + r + 1, :]
                    if np.isnan(t).any():
                        continue
                    else:
                        t_D = np.sum(np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1)))
                        t_L = np.sqrt(np.sum(np.array(t[-1, :] - t[0, :]) ** 2))
                        t_T = 1 - t_L / t_D
                        T.append(t_T)
                T_m[z] = np.mean(T)
                T_s[z] = np.std(T)
            e[par_m] = T_m
            e[par_s] = T_s

    print('Tortuosities computed')

def compute_orientation_to_origin(s,e,point, origin=np.array([0, 0])):
    ids = s.index.unique('AgentID').values
    o = 'orientation_to_origin'
    fo = 'front_orientation'
    abs_o = nam.abs(o)
    final_o = nam.final(o)
    mean_abs_o = nam.mean(abs_o)
    print(f'Computing orientation to origin based on {point}')
    s[o] = s.apply(lambda r: fun.angle_sum(fun.angle_to_x_axis(r[nam.xy(point)].values, origin), r[fo]), axis=1)
    s[abs_o] = np.abs(s[o].values)
    for id in ids:
        e.loc[id, final_o] = s[o].xs(id, level='AgentID').dropna().values[-1]
        e.loc[id, mean_abs_o] = s[abs_o].xs(id, level='AgentID').dropna().mean()
    print('Orientation to origin computed')

def compute_dst_to_origin(s,e,dt, point, origin=np.array([0, 0]), start_time_in_sec=0.0):
    ids = s.index.unique('AgentID').values
    p = point
    print(f'Computing distance to origin based on {p}')
    p_d = 'dst_to_origin'
    p_fin = nam.final(p_d)
    p_max = nam.max(p_d)
    p_mu = nam.mean(p_d)
    t0 = int(start_time_in_sec / dt)
    for id in ids:
        xy_data = s[nam.xy(p)].xs(id, level='AgentID', drop_level=True)
        d = nan_euclidean_distances(list(xy_data.values), [origin])[:, 0]
        s.loc[(slice(None), id), p_d] = d
        e.loc[id, p_max] = np.nanmax(d)
        e.loc[id, p_mu] = np.nanmean(d[t0:])
        e.loc[id, p_fin] = d[~np.isnan(d)][-1]
        try:
            l = e.loc[id, 'length']
            s.loc[(slice(None), id), nam.scal(p_d)] = d / l
            e.loc[id, nam.scal(p_max)] = e.loc[id, p_max] / l
            e.loc[id, nam.scal(p_mu)] = e.loc[id, p_mu] / l
            e.loc[id, nam.scal(p_fin)] = e.loc[id, p_fin] / l
        except:
            pass
    print('Distance to origin computed')


def align_trajectories(s, Npoints, Ncontour, track_point, arena_dims=None, mode='origin'):
    ids = s.index.unique(level='AgentID').values
    xy_pars = nam.xy(track_point)
    if not set(xy_pars).issubset(s.columns):
        raise ValueError('Defined point xy coordinates do not exist. Can not align trajectories! ')

    points = nam.midline(Npoints, type='point') + ['centroid']
    points_xy = nam.xy(points)
    contour = nam.contour(Ncontour)
    contour_xy = nam.xy(contour)

    all_xy_pars = points_xy + contour_xy + xy_pars
    all_xy_pars = [xy_pair for xy_pair in all_xy_pars if set(xy_pair).issubset(s.columns)]
    all_xy_pars = fun.group_list_by_n(np.unique(fun.flatten_list(all_xy_pars)), 2)
    if mode == 'origin':
        print('Aligning trajectories to common origin')
        xy = [s[xy_pars].xs(id, level='AgentID').dropna().values[0] for id in ids]
    elif mode == 'arena':

        if arena_dims is not None:
            print('Centralizing trajectories in arena center')
            x0, y0 = arena_dims
        else:
            raise ValueError ('Arena dimensions must be provided ')
        xy = [[x0 / 2, y0 / 2] for agent_id in ids]
    elif mode == 'center':
        print('Centralizing trajectories in trajectory center using min-max positions')
        xy_max = [s[xy_pars].xs(id, level='AgentID').max().values for id in ids]
        xy_min = [s[xy_pars].xs(id, level='AgentID').min().values for id in ids]
        xy = [(max + min) / 2 for max, min in zip(xy_max, xy_min)]

    for id, p in zip(ids, xy):
        for x, y in all_xy_pars:
            s.loc[(slice(None), id), x] -= p[0]
            s.loc[(slice(None), id), y] -= p[1]
    return s

def fixate_larva(s, Npoints, Ncontour, point, secondary_point=None, arena_dims=None):
    if arena_dims is None:
        raise ValueError('Arena dimensions must be provided ')
    ids = s.index.unique(level='AgentID').values
    points = nam.midline(Npoints, type='point') + ['centroid']
    points_xy = nam.xy(points, flat=True)
    contour = nam.contour(Ncontour)
    contour_xy = nam.xy(contour, flat=True)

    all_xy_pars = points_xy + contour_xy
    if len(ids) != 1:
        raise ValueError('Fixation only implemented for a single agent.')

    if type(point) == int:
        if point == -1:
            point = 'centroid'
        else:
            if secondary_point is not None:
                if type(secondary_point) == int and np.abs(secondary_point) == 1:
                    secondary_point = points[point + secondary_point]
            point = points[point]

    pars = [p for p in all_xy_pars if p in s.columns.values]
    if set(nam.xy(point)).issubset(s.columns):
        print(f'Fixing {point} to arena center')
        xy = [s[nam.xy(point)].xs(id, level='AgentID').copy(deep=True).values for id in ids]
        xy_start = [s[nam.xy(point)].xs(id, level='AgentID').copy(deep=True).dropna().values[0] for id in ids]
        bg_x = np.array([(p[:, 0] - start[0]) / arena_dims[0] for p, start in zip(xy, xy_start)])
        bg_y = np.array([(p[:, 1] - start[1]) / arena_dims[1] for p, start in zip(xy, xy_start)])
    else:
        raise ValueError(f" The requested {point} is not part of the dataset")
    for id, p in zip(ids, xy):
        for x, y in fun.group_list_by_n(pars, 2):
            s.loc[(slice(None), id), [x, y]] -= p

    if secondary_point is not None:
        if set(nam.xy(secondary_point)).issubset(s.columns):
            print(f'Fixing {secondary_point} as secondary point on vertical axis')
            xy_sec = [s[nam.xy(secondary_point)].xs(id, level='AgentID').copy(deep=True).values for id in ids]
            bg_a = np.array([np.arctan2(xy_sec[i][:, 1], xy_sec[i][:, 0]) - np.pi / 2 for i in range(len(xy_sec))])
        else:
            raise ValueError(f" The requested secondary {secondary_point} is not part of the dataset")

        for id, angle in zip(ids, bg_a):
            d = s[pars].xs(id, level='AgentID', drop_level=True).copy(deep=True).values
            s.loc[(slice(None), id), pars] = [fun.flatten_list(
                fun.rotate_multiple_points(points=np.array(fun.group_list_by_n(d[i].tolist(), 2)),
                                           radians=a)) for i, a in enumerate(angle)]
    else:
        bg_a = np.array([np.zeros(len(bg_x[0])) for i in range(len(ids))])
    bg = [np.vstack((bg_x[i, :], bg_y[i, :], bg_a[i, :])) for i in range(len(ids))]

    # There is only a single larva so :
    bg = bg[0]
    print('Fixed-point dataset generated')
    return s, bg

def compute_preference_index(poses, arena_dims, return_num=False, return_all=False):
    X,Y=arena_dims
    N=len(poses)
    xs=np.array([p[0] for p in poses])
    r = 0.2 * X
    N_l = xs[xs <= -r / 2].shape[0]
    N_r = xs[xs >= +r / 2].shape[0]
    N_m = xs[(xs <= +r / 2) & (xs >= -r / 2)].shape[0]
    # print(N,N_l,N_r,N_m)
    pI = np.round((N_l - N_r) / N, 3)
    if return_num:
        if return_all:
            return pI, N, N_l, N_r
        else:
            return pI, N
    else:
        return pI


if __name__ == '__main__':
    from lib.stor.managing import get_datasets
    d = get_datasets(datagroup_id='SimGroup', last_common='single_runs', names=['dish/ppp'], mode='load')[0]
    s=d.step_data
    # e=d.end
    # dt=d.dt
    # Npoints=d.Npoints
    # points=d.points
    # angles=d.angles
    # segs=d.segs
    # point=d.point
    # config=d.config
    # par_distro_dir=d.par_distro_dir
    print(s.columns)
    d.spatial_processing(show_output=True)
    # angular_processing(s,e,dt,Npoints,config, mode='full', dir=par_distro_dir)
    # compute_spatial_metrics(s,e,dt, points=['centroid'])
    # compute_extrema(s,dt, parameters=[nam.scal(nam.vel('centroid'))], interval_in_sec=0.3)
    # compute_freq(s,e,dt, parameters=[nam.scal(nam.vel('centroid'))], freq_range=[0.7, 1.8])
    # # s,e = compute_spatial_metrics(s,e,dt, points=['centroid'])
    print(s.columns)
    # d.save()