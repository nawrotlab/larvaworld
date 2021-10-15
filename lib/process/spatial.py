import itertools
from sklearn.metrics.pairwise import nan_euclidean_distances
import numpy as np

from lib.process.aux import compute_component_velocity, compute_velocity, compute_centroid
from lib.aux.ang_aux import rotate_multiple_points
from lib.aux.dictsNlists import group_list_by_n, flatten_list
import lib.aux.naming as nam
from lib.process.store import store_aux_dataset
from lib.conf.base.par import getPar


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


def comp_linear(s, e, dt, Npoints, point, mode='minimal'):
    points = nam.midline(Npoints, type='point')
    Nsegs = np.clip(Npoints - 1, a_min=0, a_max=None)
    segs = nam.midline(Nsegs, type='seg')

    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Nticks = len(s.index.unique('Step'))

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
    xy_params = group_list_by_n(xy_params, 2)

    all_d = [s.xs(id, level='AgentID', drop_level=True) for id in ids]
    dsts = nam.lin(nam.dst(points))
    cum_dsts = nam.cum(nam.lin(dsts))
    vels = nam.lin(nam.vel(points))
    accs = nam.lin(nam.acc(points))

    for p, xy, dst, cum_dst, vel, acc, orient in zip(points, xy_params, dsts, cum_dsts, vels, accs, orientations):
        # dic={a : np.zeros([Nticks, Nids]) * np.nan for a in [dst, cum_dst, vel, acc]}
        D = np.zeros([Nticks, Nids]) * np.nan
        Dcum = np.zeros([Nticks, Nids]) * np.nan
        V = np.zeros([Nticks, Nids]) * np.nan
        A = np.zeros([Nticks, Nids]) * np.nan

        for i, data in enumerate(all_d):
            v, d = compute_component_velocity(xy=data[xy].values, angles=data[orient].values, dt=dt,return_dst=True)
            a = np.diff(v) / dt
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

        # if lengths is not None:
        #     s[nam.scal(dst)] = sD.flatten()
        #     s[nam.cum(nam.scal(dst))] = sDcum.flatten()
        #     s[nam.scal(vel)] = sV.flatten()
        #     s[nam.scal(acc)] = sA.flatten()
        #     e[nam.cum(nam.scal(dst))] = sDcum[-1, :]

    pars = flatten_list(xy_params) + dsts + cum_dsts + vels + accs

    scale_to_length(s, e, pars=pars)
    print('All linear parameters computed')


def comp_spatial(s, e, dt, Npoints, point, mode='minimal'):
    points = nam.midline(Npoints, type='point')
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Nticks = len(s.index.unique('Step'))
    # if 'length' in e.columns.values:
    #     lengths = e['length'].values
    # else:
    #     lengths = None

    if mode == 'full':
        print(f'Computing distances, velocities and accelerations for {len(points)} points')
        points += ['centroid']
    elif mode == 'minimal':
        print(f'Computing distances, velocities and accelerations for a single spinepoint')
        points = [point]
    points += ['']

    points = np.unique(points).tolist()
    points = [p for p in points if set(nam.xy(p)).issubset(s.columns.values)]

    xy_params = raw_or_filtered_xy(s, points)
    xy_params = group_list_by_n(xy_params, 2)

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

        for i, data in enumerate(all_d):
            v, d = compute_velocity(xy=data[xy].values, dt=dt, return_dst=True)
            a = np.diff(v) / dt
            cum_d = np.nancumsum(d)

            D[1:, i] = d
            Dcum[1:, i] = cum_d
            V[1:, i] = v
            A[2:, i] = a
            # if lengths is not None:
            #     l = lengths[i]
            #     sD[1:, i] = d / l
            #     sDcum[1:, i] = cum_d / l
            #     sV[1:, i] = v / l
            #     sA[2:, i] = a / l

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
    dst=nam.dst('')
    sdst=nam.scal(dst)
    cdst=nam.cum(dst)
    csdst=nam.cum(sdst)
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
    for i in ['x', 'y'] :
        e[nam.final(i)] = s[i].dropna().groupby('AgentID').last()
        e[nam.initial(i)] = s[i].dropna().groupby('AgentID').first()
    e[nam.mean(nam.vel(''))] = e[cdst] / e[nam.cum('dur')]

    scale_to_length(s, e, pars=[dst, nam.vel(''), nam.acc('')])

    e[csdst] = s[sdst].dropna().groupby('AgentID').sum()
    e[nam.mean(nam.scal(nam.vel('')))] = e[csdst] / e[nam.cum('dur')]


def spatial_processing(s, e, dt, Npoints, point, Ncontour, mode='minimal', recompute=False, **kwargs):
    comp_length(s, e, Npoints, mode=mode, recompute=recompute)
    comp_centroid(s, Ncontour, recompute=recompute)
    comp_spatial(s, e, dt, Npoints, point, mode=mode)
    comp_linear(s, e, dt, Npoints, point, mode=mode)
    store_spatial(s, e, point)
    print(f'Completed {mode} spatial processing.')
    return s, e


def comp_dispersion(s, e, config, dt, point, recompute=False, starts=[0], stops=[40], **kwargs):
    aux_dir = config['aux_dir']
    ids = s.index.unique('AgentID').values
    ps = []
    pps = []
    for s0, s1 in itertools.product(starts, stops):
        if s0 == 0 and s1 == 40:
            p = f'dispersion'
        else:
            p = f'dispersion_{s0}_{s1}'
        ps.append(p)

        t0 = int(s0 / dt)
        fp = nam.final(p)
        mp = nam.max(p)
        mup = nam.mean(p)
        pps += [fp, mp, mup]

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
            e.loc[id, mup] = np.nanmean(d)
            e.loc[id, fp] = s[p].xs(id, level='AgentID').dropna().values[-1]
    scale_to_length(s, e, pars=ps + pps)
    store_aux_dataset(s, pars=ps + nam.scal(ps), type='dispersion', file=aux_dir)
    print('Dispersions computed')


def comp_tortuosity(s, e, dt, durs_in_sec=[2, 5, 10, 20], **kwargs):
    try:
        dsp_par = nam.final('dispersion') if nam.final('dispersion') in e.columns else 'dispersion'
        e['tortuosity'] = 1 - e[dsp_par] / e[nam.cum(nam.dst(''))]
    except:
        pass
    durs = [int(1 / dt * d) for d in durs_in_sec]
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



def comp_source_metrics(s, e, config, **kwargs):
    fo = getPar(['fo'], to_return=['d'])[0][0]
    xy = nam.xy('')
    sources=config['sources']
    for n,pos in sources.items() :
        print(f'Computing bearing and distance to {n} based on xy position')
        o, d = nam.bearing2(n), nam.dst2(n)
        pmax, pmu, pfin= nam.max(d), nam.mean(d), nam.final(d)
        pabs=nam.abs(o)
        temp = np.array(pos) - s[xy].values
        s[o] = (s[fo] + 180 - np.rad2deg(np.arctan2(temp[:, 1], temp[:, 0]))) % 360 - 180
        s[pabs] = s[o].abs()
        s[d] = nan_euclidean_distances(s[xy].values.tolist(), [pos])[:, 0]
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
            for p in [pmax, pmu, pfin] :
                e[nam.scal(p)] = e[p] / l

        print('Bearing and distance to source computed')


def align_trajectories(s, track_point=None, arena_dims=None, mode='origin', config=None, **kwargs):
    ids = s.index.unique(level='AgentID').values

    xy_pairs = nam.xy(nam.midline(config['Npoints'], type='point') + ['centroid', ''] + nam.contour(config['Ncontour']))
    xy_pairs = [xy for xy in xy_pairs if set(xy).issubset(s.columns)]
    xy_pairs = group_list_by_n(np.unique(flatten_list(xy_pairs)), 2)
    if mode == 'arena':
        print('Centralizing trajectories in arena center')
        if arena_dims is None:
            arena_dims = config['env_params']['arena']['arena_dims']
        x0, y0 = arena_dims
        X, Y = x0 / 2, y0 / 2
        for x, y in xy_pairs:
            s[x] -= X
            s[y] -= Y
        return s
    else :
        if track_point is None:
            track_point = config['point']

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
            for x, y in xy_pairs :
                s.loc[(slice(None), id), x] -= p[0]
                s.loc[(slice(None), id), y] -= p[1]
        return s


def fixate_larva(s, config, point, arena_dims, fix_segment=None):
    ids = s.index.unique(level='AgentID').values
    points = nam.midline(config['Npoints'], type='point') + ['centroid']
    points_xy = nam.xy(points, flat=True)
    contour = nam.contour(config['Ncontour'])
    contour_xy = nam.xy(contour, flat=True)

    all_xy_pars = points_xy + contour_xy
    if len(ids) != 1:
        raise ValueError('Fixation only implemented for a single agent.')

    if type(point) == int:
        if point == -1:
            point = 'centroid'
        else:
            if fix_segment is not None:
                if type(fix_segment) == int and np.abs(fix_segment) == 1:
                    fix_segment = points[point + fix_segment]
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
            s.loc[(slice(None), id), pars] = [flatten_list(rotate_multiple_points(points=np.array(group_list_by_n(d[i].tolist(), 2)),
                                                       radians=a)) for i, a in enumerate(angle)]
    else:
        bg_a = np.array([np.zeros(len(bg_x[0])) for i in range(len(ids))])
    bg = [np.vstack((bg_x[i, :], bg_y[i, :], bg_a[i, :])) for i in range(len(ids))]

    # There is only a single larva so :
    bg = bg[0]
    print('Fixed-point dataset generated')
    return s, bg


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

