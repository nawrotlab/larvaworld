import itertools
import time

import numpy as np

from lib.process.aux import compute_component_velocity, compute_velocity, compute_centroid
from lib.aux.ang_aux import rotate_multiple_points, angle_dif
from lib.aux.dictsNlists import group_list_by_n, flatten_list
import lib.aux.naming as nam
from lib.process.store import store_aux_dataset
from lib.conf.base.par import getPar
from lib.aux.xy_aux import eudi5x


def raw_or_filtered_xy(s, points):
    r = nam.xy(points, flat=True)
    f = nam.filt(r)
    if all(i in s.columns for i in f):
        # print('Using filtered xy coordinates')
        return f
    elif all(i in s.columns for i in r):
        # print('Using raw xy coordinates')
        return r
    else:
        print('No xy coordinates exist. Not computing spatial metrics')
        return


def comp_linear(s, e, c, mode='minimal'):
    points = nam.midline(c.Npoints, type='point')
    Nsegs = np.clip(c.Npoints - 1, a_min=0, a_max=None)
    segs = nam.midline(Nsegs, type='seg')
    if mode == 'full':
        print(f'Computing linear distances, velocities and accelerations for {len(points) - 1} points')
        points = points[1:]
        orientations = nam.orient(segs)
    elif mode == 'minimal':
        if c.point == 'centroid' or c.point == points[0]:
            print('Defined point is either centroid or head. Orientation of front segment not defined.')
            return
        else:
            print(f'Computing linear distances, velocities and accelerations for a single spinepoint')
            points = [c.point]
            orientations = ['rear_orientation']

    if not set(orientations).issubset(s.columns):
        print('Required orients not found. Component linear metrics not computed.')
        return

    xy_params = raw_or_filtered_xy(s, points)
    xy_params = group_list_by_n(xy_params, 2)

    all_d = [s.xs(id, level='AgentID', drop_level=True) for id in c.agent_ids]
    dsts = nam.lin(nam.dst(points))
    cum_dsts = nam.cum(nam.lin(dsts))
    vels = nam.lin(nam.vel(points))
    accs = nam.lin(nam.acc(points))

    for p, xy, dst, cum_dst, vel, acc, orient in zip(points, xy_params, dsts, cum_dsts, vels, accs, orientations):
        D = np.zeros([c.Nticks, c.N]) * np.nan
        Dcum = np.zeros([c.Nticks, c.N]) * np.nan
        V = np.zeros([c.Nticks, c.N]) * np.nan
        A = np.zeros([c.Nticks, c.N]) * np.nan

        for i, data in enumerate(all_d):
            v, d = compute_component_velocity(xy=data[xy].values, angles=data[orient].values, dt=c.dt, return_dst=True)
            a = np.diff(v) / c.dt
            cum_d = np.nancumsum(d)
            D[1:, i] = d
            Dcum[1:, i] = cum_d
            V[1:, i] = v
            A[2:, i] = a

        s[dst] = D.flatten()
        s[cum_dst] = Dcum.flatten()
        s[vel] = V.flatten()
        s[acc] = A.flatten()
        e[nam.cum(dst)] = Dcum[-1, :]
    pars = flatten_list(xy_params) + dsts + cum_dsts + vels + accs
    scale_to_length(s, e, pars=pars)
    print('All linear parameters computed')


def comp_spatial(s, e, c, mode='minimal'):
    points = nam.midline(c.Npoints, type='point')
    if mode == 'full':
        print(f'Computing distances, velocities and accelerations for {len(points)} points')
        points += ['centroid']
    elif mode == 'minimal':
        print(f'Computing distances, velocities and accelerations for a single spinepoint')
        points = [c.point]
    points += ['']
    points = np.unique(points).tolist()
    points = [p for p in points if set(nam.xy(p)).issubset(s.columns.values)]

    xy_params = raw_or_filtered_xy(s, points)
    xy_params = group_list_by_n(xy_params, 2)

    all_d = [s.xs(id, level='AgentID', drop_level=True) for id in c.agent_ids]
    dsts = nam.dst(points)
    cum_dsts = nam.cum(dsts)
    vels = nam.vel(points)
    accs = nam.acc(points)

    for p, xy, dst, cum_dst, vel, acc in zip(points, xy_params, dsts, cum_dsts, vels, accs):
        D = np.zeros([c.Nticks, c.N]) * np.nan
        Dcum = np.zeros([c.Nticks, c.N]) * np.nan
        V = np.zeros([c.Nticks, c.N]) * np.nan
        A = np.zeros([c.Nticks, c.N]) * np.nan

        for i, data in enumerate(all_d):
            v, d = compute_velocity(xy=data[xy].values, dt=c.dt, return_dst=True)
            a = np.diff(v) / c.dt
            cum_d = np.nancumsum(d)
            D[:, i] = d
            Dcum[:, i] = cum_d
            V[:, i] = v
            A[:, i] = a
        s[dst] = D.flatten()
        s[cum_dst] = Dcum.flatten()
        s[vel] = V.flatten()
        s[acc] = A.flatten()
        e[nam.cum(dst)] = Dcum[-1, :]

    pars = flatten_list(xy_params) + dsts + cum_dsts + vels + accs
    scale_to_length(s, e, pars=pars)
    print('All spatial parameters computed')


def comp_length(s, e, Npoints, mode='minimal', recompute=False):
    if 'length' in e.columns.values and not recompute:
        print('Length is already computed. If you want to recompute it, set recompute_length to True')
        return
    points = nam.midline(Npoints, type='point')
    Npoints = len(points)
    xy_pars = nam.xy(points, flat=True)
    if not set(xy_pars).issubset(s.columns):
        print(f'XY coordinates not found for the {Npoints} midline points. Body length can not be computed.')
        return
    Nsegs = np.clip(Npoints - 1, a_min=0, a_max=None)
    segs = nam.midline(Nsegs, type='seg')
    t = len(s)
    xy = s[xy_pars].values
    Nids = xy.shape[0]
    L = np.zeros([1, t]) * np.nan
    S = np.zeros([Nsegs, t]) * np.nan

    if mode == 'full':
        print(f'Computing lengths for {Nsegs} segments and total body length')
        for j in range(Nids):
            for i, seg in enumerate(segs):
                S[i, j] = np.sqrt(np.nansum((xy[j, 2 * i:2 * i + 2] - xy[j, 2 * i + 2:2 * i + 4]) ** 2))
            L[:, j] = np.nansum(S[:, j])
        for i, seg in enumerate(segs):
            s[seg] = S[i, :].flatten()
    elif mode == 'minimal':
        print(f'Computing body length')
        xy2 = xy.reshape(xy.shape[0], Npoints, 2)
        xy3 = np.sum(np.diff(xy2, axis=1) ** 2, axis=2)
        L = np.sum(np.sqrt(xy3), axis=1)
    s['length'] = L
    e['length'] = s['length'].groupby('AgentID').quantile(q=0.5)
    print('All lengths computed.')


def comp_centroid(s, Ncontour, recompute=False):
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
            c[i, :] = np.array(compute_centroid(contour[i, :, :]))
        s[nam.xy('centroid')[0]] = c[:, 0]
        s[nam.xy('centroid')[1]] = c[:, 1]
    print('Centroid coordinates computed.')


def store_spatial(s, e, point):
    dst = nam.dst('')
    sdst = nam.scal(dst)
    cdst = nam.cum(dst)
    csdst = nam.cum(sdst)
    dic = {
        'x': nam.xy(point)[0],
        'y': nam.xy(point)[1],
        dst: nam.dst(point),
        nam.vel(''): nam.vel(point),
        nam.acc(''): nam.acc(point),
        cdst: nam.cum(nam.dst(point)),
    }
    for k, v in dic.items():
        try:
            s[k] = s[v]
        except:
            pass
    e[cdst] = s[dst].dropna().groupby('AgentID').sum()
    for i in ['x', 'y']:
        e[nam.final(i)] = s[i].dropna().groupby('AgentID').last()
        e[nam.initial(i)] = s[i].dropna().groupby('AgentID').first()
    e[nam.mean(nam.vel(''))] = e[cdst] / e[nam.cum('dur')]

    scale_to_length(s, e, pars=[dst, nam.vel(''), nam.acc('')])

    if sdst in s.columns :
        e[csdst] = s[sdst].dropna().groupby('AgentID').sum()
        e[nam.mean(nam.scal(nam.vel('')))] = e[csdst] / e[nam.cum('dur')]


def spatial_processing(s, e, c, mode='minimal', recompute=False, **kwargs):
    comp_length(s, e, c.Npoints, mode=mode, recompute=recompute)
    comp_centroid(s, c.Ncontour, recompute=recompute)
    comp_spatial(s, e, c, mode=mode)
    comp_linear(s, e, c, mode=mode)
    store_spatial(s, e, c.point)
    print(f'Completed {mode} spatial processing.')


def comp_dispersion(s, e, dt, point, c=None, recompute=False, dsp_starts=[0], dsp_stops=[40], **kwargs):
    if dsp_starts is None or dsp_stops is None:
        return

    ids = s.index.unique('AgentID').values
    ps = []
    pps = []
    for s0, s1 in itertools.product(dsp_starts, dsp_stops):
        p = f'dispersion_{s0}_{s1}'
        ps.append(p)

        t0 = int(s0 / dt)
        fp = nam.final(p)
        mp = nam.max(p)
        mup = nam.mean(p)
        pps += [fp, mp, mup]

        if set([mp]).issubset(e.columns.values) and not recompute:
            print(
                f'Dispersion in {s0}-{s1} sec is already detected. If you want to recompute it, set recompute_dispersion to True')
            continue
        #print(f'Computing dispersion in {s0}-{s1} sec based on {point}')
        for id in ids:
            xy = s[['x', 'y']].xs(id, level='AgentID', drop_level=True)
            try:
                origin_xy = xy.dropna().values[t0]
            except:
                print(f'No values to set origin point for {id}')
                s.loc[(slice(None), id), p] = np.empty(len(xy)) * np.nan
                continue
            d = eudi5x(xy.values, origin_xy)
            d[:t0] = np.nan
            s.loc[(slice(None), id), p] = d
            e.loc[id, mp] = np.nanmax(d)
            e.loc[id, mup] = np.nanmean(d)
            e.loc[id, fp] = s[p].xs(id, level='AgentID').dropna().values[-1]
    scale_to_length(s, e, pars=ps + pps)
    if c is not None:
        store_aux_dataset(s, pars=ps + nam.scal(ps), type='dispersion', file=c.aux_dir)
    #print('Dispersions computed')


def comp_tortuosity(s, e, dt, tor_durs=[2, 5, 10, 20], **kwargs):
    '''
    Trajectory tortuosity metrics
    In the simplest case a single value is computed as T=1-D/L where D is the dispersal and L the actual pathlength.
    This metric has been used in :
    [1] J. Loveless and B. Webb, “A Neuromechanical Model of Larval Chemotaxis,” Integr. Comp. Biol., vol. 58, no. 5, pp. 906–914, 2018.
    Additionally tortuosity can be computed over a given time interval in which case the result is a vector called straightness index in [2].
    The mean and std are then computed.

    TODO Check also for binomial distribution over the straightness index vector. If there is a switch between global exploration and local search there should be evidence over a certain time interval.
    Data from here is relevant :
    [2] D. W. Sims, N. E. Humphries, N. Hu, V. Medan, and J. Berni, “Optimal searching behaviour generated intrinsically by the central pattern generator for locomotion,” Elife, vol. 8, pp. 1–31, 2019.
    '''
    if tor_durs is None:
        return
    try:
        dsp_par = nam.final('dispersion') if nam.final('dispersion') in e.columns else 'dispersion'
        e['tortuosity'] = 1 - e[dsp_par] / e[nam.cum(nam.dst(''))]
    except:
        pass
    durs = [int(1 / dt * d) for d in tor_durs]
    Ndurs = len(durs)
    if Ndurs > 0:
        ids = s.index.unique('AgentID').values
        Nids = len(ids)
        ds = [s[['x', 'y']].xs(id, level='AgentID') for id in ids]
        ds = [d.loc[d.first_valid_index(): d.last_valid_index()].values for d in ds]
        for j, r in enumerate(durs):
            par = f'tortuosity_{tor_durs[j]}'
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


def rolling_window(a, w):
    # Get windows of size w from array a
    return np.vstack([np.roll(a, -i) for i in range(w)]).T[:-w + 1]


def rolling_window_xy(xy, w):
    # Get windows of size w from 2D array xy
    xs = rolling_window(xy[:, 0], w)
    ys = rolling_window(xy[:, 1], w)
    xys = np.dstack([xs, ys])
    return xys


def tortuosity(xy):
    # Compute tortuosity over a 2D xy array
    xy = xy[~np.isnan(xy).any(axis=1)]
    if xy.shape[0] < 2:
        return np.nan
    D = np.nansum(np.sqrt(np.nansum(np.diff(xy, axis=0) ** 2, axis=1)))
    L = np.sqrt(np.nansum(np.array(xy[-1, :] - xy[0, :]) ** 2))
    return 1 - L / D


def straightness_index(xy, w):
    # Compute tortuosity over intervals of duration w
    xys = rolling_window_xy(xy, w)
    k = xy.shape[0] - xys.shape[0]
    k1 = int(k / 2)
    SI = [np.nan] * k1 + [tortuosity(xys[i, :]) for i in range(xys.shape[0])] + [np.nan] * (k - k1)
    return np.array(SI)


def comp_straightness_index(s, dt, e=None, c=None, tor_durs=[2, 5, 10, 20], **kwargs):
    Nticks = len(s.index.unique('Step'))
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    for dur in tor_durs:
        par = f'tortuosity_{dur}'
        par_m, par_s = nam.mean(par), nam.std(par)
        r = int(dur / dt / 2)
        T = np.zeros([Nticks, Nids]) * np.nan
        T_m = np.ones(Nids) * np.nan
        T_s = np.ones(Nids) * np.nan
        for j, id in enumerate(ids):
            xy = s[['x', 'y']].xs(id, level='AgentID').values
            T[:, j] = straightness_index(xy, r)
            T_m[j] = np.nanmean(T[:, j])
            T_s[j] = np.nanstd(T[:, j])
        s[par] = T.flatten()

        if e is not None:
            e[par_m] = T_m
            e[par_s] = T_s

        if c is not None:
            store_aux_dataset(s, pars=[par], type='exploration', file=c.aux_dir)


def comp_source_metrics(s, e, c, **kwargs):
    fo = getPar(['fo'], to_return=['d'])[0][0]
    xy = nam.xy('')
    for n, pos in c.source_xy.items():
        print(f'Computing bearing and distance to {n} based on xy position')
        o, d = nam.bearing2(n), nam.dst2(n)
        pmax, pmu, pfin = nam.max(d), nam.mean(d), nam.final(d)
        pabs = nam.abs(o)
        temp = np.array(pos) - s[xy].values
        s[o] = (s[fo] + 180 - np.rad2deg(np.arctan2(temp[:, 1], temp[:, 0]))) % 360 - 180
        s[pabs] = s[o].abs()
        s[d] = eudi5x(s[xy].values, np.array(pos))
        e[pmax] = s[d].groupby('AgentID').max()
        e[pmu] = s[d].groupby('AgentID').mean()
        e[pfin] = s[d].dropna().groupby('AgentID').last()
        e[nam.mean(pabs)] = s[pabs].groupby('AgentID').mean()
        if 'length' in e.columns:
            l = e['length']

            def rowIndex(row):
                return row.name[1]

            def rowLength(row):
                return l.loc[rowIndex(row)]

            def rowFunc(row):
                return row[d] / rowLength(row)

            s[nam.scal(d)] = s.apply(rowFunc, axis=1)
            for p in [pmax, pmu, pfin]:
                e[nam.scal(p)] = e[p] / l

        print('Bearing and distance to source computed')


def comp_wind_metrics(s, e, c, **kwargs):
    w = c.env_params.windscape
    if w is not None:
        wo, wv = w.wind_direction, w.wind_speed
        woo = np.deg2rad(wo)
        ids = s.index.unique('AgentID').values
        for id in ids:
            xy = s[['x', 'y']].xs(id, level='AgentID', drop_level=True).values
            origin = e[[nam.initial('x'), nam.initial('y')]].loc[id]
            d = eudi5x(xy, np.array(origin))
            print(d)
            dx = xy[:, 0] - origin[0]
            dy = xy[:, 1] - origin[1]
            angs = np.arctan2(dy, dx)
            a = np.array([angle_dif(ang, woo) for ang in angs])
            s.loc[(slice(None), id), 'anemotaxis'] = d * np.cos(a)
        s[nam.bearing2('wind')] = s.apply(lambda r: angle_dif(r[nam.orient('front')], wo), axis=1)
        e['anemotaxis'] = s['anemotaxis'].groupby('AgentID').last()


def comp_final_anemotaxis(s, e, c, **kwargs):
    w = c.env_params.windscape
    if w is not None:
        wo, wv = w.wind_direction, w.wind_speed
        woo = np.deg2rad(wo)
        xy0 = s[['x', 'y']].groupby('AgentID').first()
        xy1 = s[['x', 'y']].groupby('AgentID').last()
        dx = xy1.values[:, 0] - xy0.values[:, 0]
        dy = xy1.values[:, 1] - xy0.values[:, 1]
        d = np.sqrt(dx ** 2 + dy ** 2)
        angs = np.arctan2(dy, dx)
        a = np.array([angle_dif(ang, woo) for ang in angs])
        e['anemotaxis'] = d * np.cos(a)
        # print(e['anemotaxis'])


def align_trajectories(s, track_point=None, arena_dims=None, mode='origin', config=None, **kwargs):
    ids = s.index.unique(level='AgentID').values

    xy_pairs = nam.xy(nam.midline(config.Npoints, type='point') + ['centroid', ''] + nam.contour(config.Ncontour))
    xy_pairs = [xy for xy in xy_pairs if set(xy).issubset(s.columns)]
    xy_pairs = group_list_by_n(np.unique(flatten_list(xy_pairs)), 2)
    if mode == 'arena':
        print('Centralizing trajectories in arena center')
        if arena_dims is None:
            arena_dims = config.env_params.arena.arena_dims
        x0, y0 = arena_dims
        X, Y = x0 / 2, y0 / 2
        for x, y in xy_pairs:
            s[x] -= X
            s[y] -= Y
        return s
    else:
        if track_point is None:
            track_point = config.point

        XY = nam.xy(track_point) if set(nam.xy(track_point)).issubset(s.columns) else ['x', 'y']
        if not set(XY).issubset(s.columns):
            raise ValueError('Defined point xy coordinates do not exist. Can not align trajectories! ')

        if mode == 'origin':
            print('Aligning trajectories to common origin')
            xy = [s[XY].xs(id, level='AgentID').dropna().values[0] for id in ids]
        elif mode == 'center':
            print('Centralizing trajectories in trajectory center using min-max positions')
            xy_max = [s[XY].xs(id, level='AgentID').max().values for id in ids]
            xy_min = [s[XY].xs(id, level='AgentID').min().values for id in ids]
            xy = [(max + min) / 2 for max, min in zip(xy_max, xy_min)]

        for id, p in zip(ids, xy):
            for x, y in xy_pairs:
                s.loc[(slice(None), id), x] -= p[0]
                s.loc[(slice(None), id), y] -= p[1]
        return s


def fixate_larva(s, config, point, arena_dims, fix_segment=None):
    ids = s.index.unique(level='AgentID').values
    points = nam.midline(config.Npoints, type='point') + ['centroid']
    points_xy = nam.xy(points, flat=True)
    contour = nam.contour(config.Ncontour)
    contour_xy = nam.xy(contour, flat=True)

    all_xy_pars = points_xy + contour_xy
    if len(ids) != 1:
        raise ValueError('Fixation only implemented for a single agent.')
    # id=ids[0]
    if type(point) == int:
        if point == -1:
            point = 'centroid'
        else:
            if fix_segment is not None and type(fix_segment) == int and np.abs(fix_segment) == 1:
                fix_segment = points[point + fix_segment]
            point = points[point]

    pars = [p for p in all_xy_pars if p in s.columns.values]
    xy_ps = nam.xy(point)
    if set(xy_ps).issubset(s.columns):
        print(f'Fixing {point} to arena center')
        X, Y = arena_dims
        xy = [s[xy_ps].xs(id, level='AgentID').copy(deep=True).values for id in ids]
        xy_start = [s[xy_ps].xs(id, level='AgentID').copy(deep=True).dropna().values[0] for id in ids]
        bg_x = np.array([(p[:, 0] - start[0]) / X for p, start in zip(xy, xy_start)])
        bg_y = np.array([(p[:, 1] - start[1]) / Y for p, start in zip(xy, xy_start)])
    else:
        raise ValueError(f" The requested {point} is not part of the dataset")
    for id, p in zip(ids, xy):
        for x, y in group_list_by_n(pars, 2):
            s.loc[(slice(None), id), [x, y]] -= p

    if fix_segment is not None:
        if set(nam.xy(fix_segment)).issubset(s.columns):
            print(f'Fixing {fix_segment} as secondary point on vertical axis')
            xy_sec = [s[nam.xy(fix_segment)].xs(id, level='AgentID').copy(deep=True).values for id in ids]
            bg_a = np.array([np.arctan2(xy_sec[i][:, 1], xy_sec[i][:, 0]) - np.pi / 2 for i in range(len(xy_sec))])
        else:
            raise ValueError(f" The requested secondary {fix_segment} is not part of the dataset")

        for id, angle in zip(ids, bg_a):
            d = s[pars].xs(id, level='AgentID', drop_level=True).copy(deep=True).values
            s.loc[(slice(None), id), pars] = [
                flatten_list(rotate_multiple_points(points=np.array(group_list_by_n(d[i].tolist(), 2)),
                                                    radians=a)) for i, a in enumerate(angle)]
    else:
        bg_a = np.array([np.zeros(len(bg_x[0])) for i in range(len(ids))])
    bg = [np.vstack((bg_x[i, :], bg_y[i, :], bg_a[i, :])) for i in range(len(ids))]

    # There is only a single larva so :
    bg = bg[0]
    print('Fixed-point dataset generated')
    return s, bg


def comp_PI2(arena_xdim, xys, x=0.04):
    Nticks = xys.index.unique('Step').size
    ids = xys.index.unique('AgentID').values
    N = len(ids)
    dLR = np.zeros([N, Nticks]) * np.nan
    for i, id in enumerate(ids):
        xy = xys.xs(id, level='AgentID').values
        dL = eudi5x(xy, np.array((-x, 0)))
        dR = eudi5x(xy, np.array((x, 0)))
        dLR[i, :] = (dR - dL) / (2 * x)
    dLR_mu = np.mean(dLR, axis=1)
    mu_dLR_mu = np.mean(dLR_mu)
    return mu_dLR_mu


def comp_PI(arena_xdim, xs, return_num=False, return_all=False):
    N = len(xs)
    r = 0.2 * arena_xdim
    xs = np.array(xs)
    N_l = len(xs[xs <= -r / 2])
    N_r = len(xs[xs >= +r / 2])
    N_m = len(xs[(xs <= +r / 2) & (xs >= -r / 2)])
    pI = np.round((N_l - N_r) / N, 3)
    if return_num:
        if return_all:
            return pI, N, N_l, N_r
        else:
            return pI, N
    else:
        return pI


def scale_to_length(s, e, pars=None, keys=None):
    l_par = 'length'
    if l_par not in e.keys():
        return
    l = e[l_par]
    if pars is None:
        if keys is not None:
            pars = getPar(keys, to_return=['d'])[0]
        else:
            raise ValueError('No parameter names or keys provided.')
    s_pars = [p for p in pars if p in s.columns]
    ids = s.index.get_level_values('AgentID').values
    ls = l.loc[ids].values
    if len(s_pars) > 0:
        s[nam.scal(s_pars)] = (s[s_pars].values.T / ls).T
    e_pars = [p for p in pars if p in e.columns]
    if len(e_pars) > 0:
        e[nam.scal(e_pars)] = (e[e_pars].values.T / l.values).T
