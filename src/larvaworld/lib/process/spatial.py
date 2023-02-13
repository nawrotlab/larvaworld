import copy
import itertools

import numpy as np
import pandas as pd

from larvaworld.lib import reg, aux, decorators
from larvaworld.lib.aux import naming as nam

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

    xy_params = aux.xy.raw_or_filtered_xy(s, points)
    xy_params = aux.group_list_by_n(xy_params, 2)

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
            v, d = aux.compute_component_velocity(xy=data[xy].values, angles=data[orient].values, dt=c.dt, return_dst=True)
            a = np.diff(v) / c.dt
            cum_d = np.nancumsum(d)
            D[:, i] = d
            Dcum[:, i] = cum_d
            V[:, i] = v
            A[1:, i] = a

        s[dst] = D.flatten()
        s[cum_dst] = Dcum.flatten()
        s[vel] = V.flatten()
        s[acc] = A.flatten()
        e[nam.cum(dst)] = Dcum[-1, :]
    pars = aux.flatten_list(xy_params) + dsts + cum_dsts + vels + accs
    scale_to_length(s, e, c, pars=pars)
    print('All linear parameters computed')

def comp_spatial2(s, e, c, mode='minimal'):
    points = nam.midline(c.Npoints, type='point')
    if mode == 'full':
        print(f'Computing distances, velocities and accelerations for {len(points)} points')
        points += ['centroid']
    elif mode == 'minimal':
        print(f'Computing distances, velocities and accelerations for a single spinepoint')
        points = [c.point]
    points += ['']
    points = np.unique(points).tolist()


    for p in points :
        xy=nam.xy(p)
        if set(xy).issubset(s.columns.values) :
            dst=nam.dst(p)
            vel=nam.vel(p)
            s[dst] = aux.apply_per_level(s[xy], aux.eudist).flatten()
            s[nam.cum(dst)] = aux.apply_per_level(s[dst], np.nancumsum).flatten()
            s[vel] = s[dst]/c.dt
            s[nam.acc(p)] = aux.apply_per_level(s[vel], aux.rate, dt=c.dt).flatten()
            e[nam.cum(dst)] = s[nam.cum(dst)].values[-1, :]



    dsts = nam.dst(points)
    cum_dsts = nam.cum(dsts)
    vels = nam.vel(points)
    accs = nam.acc(points)

    pars = aux.raw_or_filtered_xy(s, points) + dsts + cum_dsts + vels + accs
    scale_to_length(s, e, c, pars=pars)
    print('All spatial parameters computed')

def comp_spatial(s, e, c, mode='minimal'):
    points = nam.midline(c.Npoints, type='point')
    if mode == 'full':
        reg.vprint(f'Computing distances, velocities and accelerations for {len(points)} points',1)
        points += ['centroid']
    elif mode == 'minimal':
        reg.vprint(f'Computing distances, velocities and accelerations for a single spinepoint',1)
        points = [c.point]
    points += ['']
    points = np.unique(points).tolist()
    points = [p for p in points if set(nam.xy(p)).issubset(s.columns.values)]

    xy_params = aux.raw_or_filtered_xy(s, points)
    xy_params = aux.group_list_by_n(xy_params, 2)

    dsts = nam.dst(points)
    cum_dsts = nam.cum(dsts)
    vels = nam.vel(points)
    accs = nam.acc(points)

    for p, xy, dst, cum_dst, vel, acc in zip(points, xy_params, dsts, cum_dsts, vels, accs):
        D = np.zeros([c.Nticks, c.N]) * np.nan
        Dcum = np.zeros([c.Nticks, c.N]) * np.nan
        V = np.zeros([c.Nticks, c.N]) * np.nan
        A = np.zeros([c.Nticks, c.N]) * np.nan

        for i, id in enumerate(c.agent_ids):
            D[:, i] = aux.eudist(s[xy].xs(id, level='AgentID').values)
            Dcum[:, i] = np.nancumsum(D[:, i])
            V[:, i] = D[:, i]/c.dt
            A[1:, i] = np.diff(V[:, i]) / c.dt
        s[dst] = D.flatten()
        s[cum_dst] = Dcum.flatten()
        s[vel] = V.flatten()
        s[acc] = A.flatten()
        # e[nam.cum(dst)] = Dcum[-1, :]
        e[nam.cum(dst)] = s[cum_dst].dropna().groupby('AgentID').last()

    pars = aux.flatten_list(xy_params) + dsts + cum_dsts + vels + accs
    scale_to_length(s, e, c, pars=pars)
    reg.vprint('All spatial parameters computed',1)

@reg.funcs.proc("length")
def comp_length(s, e, c=None, N=None, mode='minimal', recompute=False):
    if 'length' in e.columns.values and not recompute:
        reg.vprint('Length is already computed. If you want to recompute it, set recompute_length to True',1)
        return
    if N is None :
        N = c.Npoints
    points = nam.midline(N, type='point')
    xy_pars = nam.xy(points, flat=True)
    if not set(xy_pars).issubset(s.columns):
        reg.vprint(f'XY coordinates not found for the {N} midline points. Body length can not be computed.',1)
        return
    xy = s[xy_pars].values

    if mode == 'full':
        Nsegs = np.clip(N - 1, a_min=0, a_max=None)
        segs = nam.midline(Nsegs, type='seg')
        t = len(s)
        S = np.zeros([Nsegs, t]) * np.nan
        L = np.zeros([1, t]) * np.nan
        reg.vprint(f'Computing lengths for {Nsegs} segments and total body length',1)
        for j in range(t):
            for i, seg in enumerate(segs):
                S[i, j] = np.sqrt(np.nansum((xy[j, 2 * i:2 * i + 2] - xy[j, 2 * i + 2:2 * i + 4]) ** 2))
            L[:, j] = np.nansum(S[:, j])
        for i, seg in enumerate(segs):
            s[seg] = S[i, :].flatten()
    elif mode == 'minimal':
        reg.vprint(f'Computing body length')
        xy2 = xy.reshape(xy.shape[0], N, 2)
        xy3 = np.sum(np.diff(xy2, axis=1) ** 2, axis=2)
        L = np.sum(np.sqrt(xy3), axis=1)
    s['length'] = L
    e['length'] = s['length'].groupby('AgentID').quantile(q=0.5)
    reg.vprint('All lengths computed.',1)

@reg.funcs.proc("centroid")
def comp_centroid(s, c, recompute=False):
    if set(nam.xy('centroid')).issubset(s.columns.values) and not recompute:
        print('Centroid is already computed. If you want to recompute it, set recompute_centroid to True')
    Nc=c.Ncontour
    con_pars = nam.xy(nam.contour(Nc), flat=True)
    if not set(con_pars).issubset(s.columns) or Nc == 0:
        print(f'No contour found. Not computing centroid')
    else:
        print(f'Computing centroid from {Nc} contourpoints')
        xy = s[con_pars].values
        t = xy.shape[0]
        xy = np.reshape(xy, (t, Nc, 2))
        cen = np.zeros([t, 2]) * np.nan
        for i in range(t):
            cen[i, :] = np.sum(xy[i, :, :], axis=0)/Nc
        s[nam.xy('centroid')] = cen
    print('Centroid coordinates computed.')


def store_spatial(s, e, c, store=False, also_in_mm=False):
    point = c.point
    dst = nam.dst('')
    sdst = nam.scal(dst)
    cdst = nam.cum(dst)
    csdst = nam.cum(sdst)
    v = nam.vel('')
    a = nam.acc('')

    dic = {
        'x': nam.xy(point)[0],
        'y': nam.xy(point)[1],
        dst: nam.dst(point),
        v: nam.vel(point),
        a: nam.acc(point),
        cdst: nam.cum(nam.dst(point)),
    }
    for k1, k2 in dic.items():
        try:
            s[k1] = s[k2]
        except:
            pass

    e[cdst] = s[dst].dropna().groupby('AgentID').sum()
    s[cdst] = s[dst].groupby('AgentID').cumsum()

    for i in ['x', 'y']:
        e[nam.final(i)] = s[i].dropna().groupby('AgentID').last()
        e[nam.initial(i)] = s[i].dropna().groupby('AgentID').first()
    e[nam.mean(v)] = e[cdst] / e[nam.cum('dur')]

    scale_to_length(s, e, c, pars=[dst, v, a])

    if sdst in s.columns:
        s[csdst] = s[sdst].groupby('AgentID').cumsum()
        e[csdst] = s[sdst].dropna().groupby('AgentID').sum()
        e[nam.mean(nam.scal(v))] = e[csdst] / e[nam.cum('dur')]

    shorts = ['v', 'a', 'sv', 'sa']

    if also_in_mm:
        d_in_mm, v_in_mm, a_in_mm = reg.getPar(['d_in_mm', 'v_in_mm', 'a_in_mm'])
        s[d_in_mm] = s[dst] * 1000
        s[v_in_mm] = s[v] * 1000
        s[a_in_mm] = s[a] * 1000
        e[nam.cum(d_in_mm)] = e[cdst] * 1000
        e[nam.mean(v_in_mm)] = e[nam.mean(v)] * 1000
        shorts += ['v_in_mm', 'a_in_mm']

    if store:
        aux.store_distros(s, pars=reg.getPar(shorts), parent_dir=c.dir)
        ps=[p for p in [dst,cdst, sdst, csdst] if p in s.columns]
        aux.storeH5(s[ps], key='pathlength', path=reg.datapath('aux', c.dir))

        aux.storeH5(df=s[['x', 'y']], key='default', path=reg.datapath('traj', c.dir))


@reg.funcs.proc("spatial")
def spatial_processing(s, e, c, mode='minimal', recompute=False, store=False, **kwargs):
    comp_length(s, e, c, mode=mode, recompute=recompute)
    comp_centroid(s, c, recompute=recompute)
    comp_spatial(s, e, c, mode=mode)
    # comp_linear(s, e, c, mode=mode)
    store_spatial(s, e, c, store=store)
    try:
        align_trajectories(s, c, store=store, replace=False, transposition='origin')
    except :
        pass

    print(f'Completed {mode} spatial processing.')


@reg.funcs.proc("dispersion")
def comp_dispersion(s, e, c, dsp_starts=[0], dsp_stops=[40], store=True, **kwargs):
    if dsp_starts is None or dsp_stops is None:
        return
    xy0 = aux.read(key='default', path=reg.datapath('traj', c.dir)) if s is None else s[['x', 'y']]
    dsps = {}
    for t0, t1 in itertools.product(dsp_starts, dsp_stops):

        s0 = int(t0 / c.dt)
        s1 = int(t1 / c.dt)
        xy = xy0.loc[(slice(s0, s1), slice(None)), ['x', 'y']]


        AA = aux.apply_per_level(xy, aux.compute_dispersal_solo)
        Nt=AA.shape[0]
        AA0 = np.zeros([c.Nticks, c.N]) * np.nan
        AA0[s0:s0 + Nt, :] = AA

        p = f'dispersion_{int(t0)}_{int(t1)}'
        s[p] = AA0.flatten()

        fp = nam.final(p)
        mp = nam.max(p)
        mup = nam.mean(p)
        e[mp] = s[p].groupby('AgentID').max()
        e[mup] = s[p].groupby('AgentID').mean()
        e[fp] = s[p].dropna().groupby('AgentID').last()
        scale_to_length(s, e, c, pars=[p, fp, mp, mup])

        for par in [p, nam.scal(p)]:
            dsps[par] = get_disp_df(s[par], s0, Nt)


    if store:
        aux.save_dict(dsps, reg.datapath('dsp', c.dir))


def get_disp_df(dsp, s0, Nt):
    trange = np.arange(s0, s0 + Nt, 1)
    dsp_ar = np.zeros([Nt, 3]) * np.nan
    dsp_ar[:, 0] = dsp.groupby(level='Step').quantile(q=0.5).values[s0:s0 + Nt]
    dsp_ar[:, 1] = dsp.groupby(level='Step').quantile(q=0.75).values[s0:s0 + Nt]
    dsp_ar[:, 2] = dsp.groupby(level='Step').quantile(q=0.25).values[s0:s0 + Nt]
    return pd.DataFrame(dsp_ar, index=trange, columns=['median', 'upper', 'lower'])

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
    if D == 0:
        return np.nan
    else:
        L = np.sqrt(np.nansum(np.array(xy[-1, :] - xy[0, :]) ** 2))
        return 1 - L / D


def straightness_index(xy, w, match_shape=True):
    try:
        xy=xy.values
    except :
        pass
    # Compute tortuosity over intervals of duration w
    xys = rolling_window_xy(xy, w)
    k0, k1 = xy.shape[0], xys.shape[0]
    if match_shape:
        dk = int((k0 - k1) / 2)
        SI = np.zeros(k0) * np.nan
        for i in range(k1):
            SI[dk + i] = tortuosity(xys[i, :])
    else:
        SI = np.zeros(k1) * np.nan
        for i in range(k1):
            SI[i] = tortuosity(xys[i, :])
    return SI

@reg.funcs.proc("tortuosity")
def comp_straightness_index(s=None, e=None, c=None, dt=None, tor_durs=[1, 2, 5, 10, 20], store=False, **kwargs):
    if dt is None:
        dt = c.dt

    if s is None:
        ss = aux.read(key='step', path=reg.datapath('step',c.dir))[['x', 'y']]
        s = ss
    else:
        ss = s[['x', 'y']]
    pars = [reg.getPar(f'tor{dur}') for dur in tor_durs]
    for dur, p in zip(tor_durs, pars):
        r = int(dur / dt / 2)
        s[p] = aux.apply_per_level(ss, straightness_index, w=r).flatten()
        if e is not None:
            e[nam.mean(p)] = s[p].groupby('AgentID').mean()
            e[nam.std(p)] = s[p].groupby('AgentID').std()


    if store:
        dic = aux.get_distros(s, pars=pars)
        aux.storeH5(dic, path=reg.datapath('distro', c.dir))

@reg.funcs.proc("source")
def comp_source_metrics(s, e, c, **kwargs):
    fo = reg.getPar('fo')
    xy = nam.xy('')
    for n, pos in c.source_xy.items():
        print(f'Computing bearing and distance to {n} based on xy position')
        o, d = nam.bearing2(n), nam.dst2(n)
        pmax, pmu, pfin = nam.max(d), nam.mean(d), nam.final(d)
        pabs = nam.abs(o)
        temp = np.array(pos) - s[xy].values
        s[o] = (s[fo] + 180 - np.rad2deg(np.arctan2(temp[:, 1], temp[:, 0]))) % 360 - 180
        s[pabs] = s[o].abs()
        s[d] = aux.eudi5x(s[xy].values, pos)
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

@reg.funcs.proc("wind")
def comp_wind(**kwargs) :
    try :
        comp_wind_metrics(**kwargs)
    except :
        comp_final_anemotaxis(**kwargs)

def comp_wind_metrics(s, e, c, **kwargs):
    w = c.env_params.windscape
    if w is not None:
        wo, wv = w.wind_direction, w.wind_speed
        woo = np.deg2rad(wo)
        ids = s.index.unique('AgentID').values

        for id in ids:
            xy = s[['x', 'y']].xs(id, level='AgentID', drop_level=True).values
            origin = e[[nam.initial('x'), nam.initial('y')]].loc[id]
            d = aux.eudi5x(xy, origin)
            dx = xy[:, 0] - origin[0]
            dy = xy[:, 1] - origin[1]
            angs = np.arctan2(dy, dx)
            a = np.array([aux.angle_dif(ang, woo) for ang in angs])
            s.loc[(slice(None), id), 'anemotaxis'] = d * np.cos(a)
        s[nam.bearing2('wind')] = s.apply(lambda r: aux.angle_dif(r[nam.orient('front')], wo), axis=1)
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
        a = np.array([aux.angle_dif(ang, woo) for ang in angs])
        e['anemotaxis'] = d * np.cos(a)
        # print(e['anemotaxis'])


@reg.funcs.preproc("transposition")
def align_trajectories(s, c, track_point=None, arena_dims=None, transposition='origin', store=False,replace=True, **kwargs):
    if transposition in ['', None, np.nan]:
        return
    mode=transposition

    xy_pairs = nam.xy(nam.midline(c.Npoints, type='point') + ['centroid', ''] + nam.contour(c.Ncontour))
    xy_pairs = [xy for xy in xy_pairs if set(xy).issubset(s.columns)]
    xy_flat=np.unique(aux.flatten_list(xy_pairs))
    xy_pairs = aux.group_list_by_n(xy_flat, 2)

    if replace :
        ss=s
    else :
        ss = copy.deepcopy(s[xy_flat])


    if mode == 'arena':
        print('Centralizing trajectories in arena center')
        if arena_dims is None:
            arena_dims = c.env_params.arena.dims
        x0, y0 = arena_dims
        X, Y = x0 / 2, y0 / 2

        for x, y in xy_pairs:
            ss[x] -= X
            ss[y] -= Y
        return ss
    else:
        if track_point is None:
            track_point = c.point
        XY = nam.xy(track_point) if set(nam.xy(track_point)).issubset(s.columns) else ['x', 'y']
        if not set(XY).issubset(s.columns):
            raise ValueError('Defined point xy coordinates do not exist. Can not align trajectories! ')
        ids = s.index.unique(level='AgentID').values
        Nticks = len(s.index.unique('Step'))
        if mode == 'origin':
            print('Aligning trajectories to common origin')
            xy = [s[XY].xs(id, level='AgentID').dropna().values[0] for id in ids]
        elif mode == 'center':
            print('Centralizing trajectories in trajectory center using min-max positions')
            xy_max = [s[XY].xs(id, level='AgentID').max().values for id in ids]
            xy_min = [s[XY].xs(id, level='AgentID').min().values for id in ids]
            xy = [(max + min) / 2 for max, min in zip(xy_max, xy_min)]
        else :
            raise ValueError('Supported modes are "arena", "origin" and "center"!')
        xs= np.array([x for x, y in xy]*Nticks)
        ys= np.array([y for x, y in xy]*Nticks)

        for jj,(x, y) in enumerate(xy_pairs):
            ss[x] = ss[x].values-xs
            ss[y] = ss[y].values-ys

        if store:
            aux.storeH5(ss, key=mode, path=reg.datapath('traj', c.dir))

            print(f'traj_aligned2{mode} stored')
        return ss


def fixate_larva_multi(s, c, point, arena_dims=None, fix_segment=None):
    ids = s.index.unique(level='AgentID').values
    Nids=len(ids)
    points = nam.midline(c.Npoints, type='point') + ['centroid']
    points_xy = nam.xy(points, flat=True)
    contour = nam.contour(c.Ncontour)
    contour_xy = nam.xy(contour, flat=True)

    all_xy_pars = points_xy + contour_xy

    if type(point) == int:
        if point == -1:
            point = 'centroid'
        else:
            if fix_segment is not None and type(fix_segment) == int and np.abs(fix_segment) == 1:
                fix_segment = points[point + fix_segment]
            point = points[point]

    pars = [p for p in all_xy_pars if p in s.columns.values]
    xy_ps = nam.xy(point)
    if not set(xy_ps).issubset(s.columns):
        raise ValueError(f" The requested {point} is not part of the dataset")
    print(f'Fixing {point} to arena center')
    if arena_dims is None :
        arena_dims=c.env_params.arena.dims
    X, Y = arena_dims
    xy = [s[xy_ps].xs(id, level='AgentID').copy(deep=True).values for id in ids]
    xy_start = [s[xy_ps].xs(id, level='AgentID').copy(deep=True).dropna().values[0] for id in ids]
    bg_x = np.array([(p[:, 0] - start[0]) / X for p, start in zip(xy, xy_start)])
    bg_y = np.array([(p[:, 1] - start[1]) / Y for p, start in zip(xy, xy_start)])

    for id, p in zip(ids, xy):
        for x, y in aux.group_list_by_n(pars, 2):
            s.loc[(slice(None), id), [x, y]] -= p

    if fix_segment is not None:
        xy_ps2 = nam.xy(fix_segment)
        if not set(xy_ps2).issubset(s.columns):
            raise ValueError(f" The requested secondary {fix_segment} is not part of the dataset")

        print(f'Fixing {fix_segment} as secondary point on vertical axis')
        xy_sec = [s[xy_ps2].xs(id, level='AgentID').copy(deep=True).values for id in ids]
        bg_a = np.array([np.arctan2(xy_sec[i][:, 1], xy_sec[i][:, 0]) - np.pi / 2 for i in range(Nids)])

        for id, angle in zip(ids, bg_a):
            d = s[pars].xs(id, level='AgentID', drop_level=True).copy(deep=True).values
            s.loc[(slice(None), id), pars] = [
                aux.flatten_list(aux.rotate_points_around_point(points=np.array(aux.group_list_by_n(d[i].tolist(), 2)),
                                                                        radians=a)) for i, a in enumerate(angle)]
    else:
        bg_a = np.array([np.zeros(len(bg_x[0])) for i in range(Nids)])
    bg = [np.vstack((bg_x[i, :], bg_y[i, :], bg_a[i, :])) for i in range(Nids)]

    print('Fixed-point dataset generated')
    return s, bg


def fixate_larva(s, c, point, arena_dims=None, fix_segment=None):
    ids = s.index.unique(level='AgentID').values
    Nids = len(ids)
    N=s.index.unique('Step').size
    points = nam.midline(c.Npoints, type='point') + ['centroid']
    points_xy = nam.xy(points, flat=True)
    contour = nam.contour(c.Ncontour)
    contour_xy = nam.xy(contour, flat=True)

    all_xy_pars = points_xy + contour_xy
    if Nids != 1:
        raise ValueError('Fixation only implemented for a single agent.')
    id=ids[0]
    if type(point) == int:
        if point == -1:
            point = 'centroid'
        else:
            if fix_segment is not None and type(fix_segment) == int and np.abs(fix_segment) == 1:
                fix_segment = points[point + fix_segment]
            point = points[point]

    pars = [p for p in all_xy_pars if p in s.columns.values]
    xy_ps = nam.xy(point)
    if not set(xy_ps).issubset(s.columns):
        raise ValueError(f" The requested {point} is not part of the dataset")
    print(f'Fixing {point} to arena center')
    if arena_dims is None:
        arena_dims = c.env_params.arena.dims
    X, Y = arena_dims
    xy = s[xy_ps].values
    xy_start = s[xy_ps].dropna().values[0]
    bg_x = (xy[:, 0] - xy_start[0]) / X
    bg_y = (xy[:, 1] - xy_start[1]) / Y

    for x, y in aux.group_list_by_n(pars, 2):
        s[[x, y]] -= xy

    if fix_segment is not None:
        xy_ps2 = nam.xy(fix_segment)
        if not set(xy_ps2).issubset(s.columns):
            raise ValueError(f" The requested secondary {fix_segment} is not part of the dataset")

        print(f'Fixing {fix_segment} as secondary point on vertical axis')
        xy_sec = s[xy_ps2].values
        bg_a = np.arctan2(xy_sec[:, 1], xy_sec[:, 0]) - np.pi / 2

        s[pars] = [
            aux.flatten_list(aux.rotate_points_around_point(points=np.reshape(s[pars].values[i,:], (-1, 2)),
            # aux.flatten_list(aux.rotate_points_around_point(points=np.array(aux.group_list_by_n(s[pars].values[i].tolist(), 2)),
                                                            radians=bg_a[i])) for i in range(N)]
    else:
        bg_a = np.zeros(N)


    bg = np.vstack((bg_x, bg_y, bg_a))
    print('Fixed-point dataset generated')

    return s, bg


def comp_PI2(arena_xdim, xys, x=0.04):
    Nticks = xys.index.unique('Step').size
    ids = xys.index.unique('AgentID').values
    N = len(ids)
    dLR = np.zeros([N, Nticks]) * np.nan
    for i, id in enumerate(ids):
        xy = xys.xs(id, level='AgentID').values
        dL = aux.eudi5x(xy, [-x, 0])
        dR = aux.eudi5x(xy, [x, 0])
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


def scale_to_length(s, e, c=None, pars=None, keys=None):
    l_par = 'length'
    if l_par not in e.keys():
        comp_length(s, e, c=c, mode='minimal', recompute=True)
    l = e[l_par]
    if pars is None:
        if keys is not None:
            pars = reg.getPar(keys)
        else:
            raise ValueError('No parameter names or keys provided.')
    s_pars = [p for p in pars if p in s.columns]

    if len(s_pars) > 0:
        ids = s.index.get_level_values('AgentID').values
        ls = l.loc[ids].values
        s[nam.scal(s_pars)] = (s[s_pars].values.T / ls).T
    e_pars = [p for p in pars if p in e.columns]
    if len(e_pars) > 0:
        e[nam.scal(e_pars)] = (e[e_pars].values.T / l.values).T
