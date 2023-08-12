import copy
import itertools

import numpy as np
import pandas as pd

from larvaworld.lib import reg, aux
from larvaworld.lib.aux import nam
from larvaworld.lib.param import XYops


def comp_linear(s, e, c, mode='minimal'):
    assert isinstance(c, reg.DatasetConfig)
    points = c.midline_points
    if mode == 'full':
        reg.vprint(f'Computing linear distances, velocities and accelerations for {c.Npoints - 1} points')
        points = points[1:]
        orientations = c.seg_orientations
    elif mode == 'minimal':
        if c.point == 'centroid' or c.point == points[0]:
            reg.vprint('Defined point is either centroid or head. Orientation of front segment not defined.')
            return
        else:
            reg.vprint(f'Computing linear distances, velocities and accelerations for a single spinepoint')
            points = [c.point]
            orientations = ['rear_orientation']

    if not aux.cols_exist(orientations,s):
        reg.vprint('Required orients not found. Component linear metrics not computed.')
        return


    all_d = [s.xs(id, level='AgentID', drop_level=True) for id in c.agent_ids]
    dsts = nam.lin(nam.dst(points))
    cum_dsts = nam.cum(nam.lin(dsts))
    vels = nam.lin(nam.vel(points))
    accs = nam.lin(nam.acc(points))

    for p, dst, cum_dst, vel, acc, orient in zip(points, dsts, cum_dsts, vels, accs, orientations):
        D = np.zeros([c.Nticks, c.N]) * np.nan
        Dcum = np.zeros([c.Nticks, c.N]) * np.nan
        V = np.zeros([c.Nticks, c.N]) * np.nan
        A = np.zeros([c.Nticks, c.N]) * np.nan

        for i, data in enumerate(all_d):
            v, d = aux.compute_component_velocity(xy=data[nam.xy(p)].values, angles=data[orient].values, dt=c.dt, return_dst=True)
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
    pars = nam.xy(points) + dsts + cum_dsts + vels + accs
    scale_to_length(s, e, c, pars=pars)
    reg.vprint('All linear parameters computed')


def comp_spatial(s, e, c, mode='minimal'):
    if mode == 'full':
        reg.vprint(f'Computing distances, velocities and accelerations for {c.Npoints} points',1)
        points = c.midline_points + ['centroid', '']
    elif mode == 'minimal':
        reg.vprint(f'Computing distances, velocities and accelerations for a single spinepoint',1)
        points = [c.point, '']
    else:
        raise ValueError(f'{mode} not in supported modes : [minimal, full]')
    points = [p for p in aux.unique_list(points) if nam.xy(p).exist_in(s)]


    dsts = nam.dst(points)
    cum_dsts = nam.cum(dsts)
    vels = nam.vel(points)
    accs = nam.acc(points)

    for p, dst, cum_dst, vel, acc in zip(points, dsts, cum_dsts, vels, accs):
        D = np.zeros([c.Nticks, c.N]) * np.nan
        Dcum = np.zeros([c.Nticks, c.N]) * np.nan
        V = np.zeros([c.Nticks, c.N]) * np.nan
        A = np.zeros([c.Nticks, c.N]) * np.nan

        for i, id in enumerate(c.agent_ids):
            D[:, i] = aux.eudist(s[nam.xy(p)].xs(id, level='AgentID').values)
            Dcum[:, i] = np.nancumsum(D[:, i])
            V[:, i] = D[:, i]/c.dt
            A[1:, i] = np.diff(V[:, i]) / c.dt
        s[dst] = D.flatten()
        s[cum_dst] = Dcum.flatten()
        s[vel] = V.flatten()
        s[acc] = A.flatten()
        e[nam.cum(dst)] = s[cum_dst].dropna().groupby('AgentID').last()

    pars = nam.xy(points) + dsts + cum_dsts + vels + accs
    scale_to_length(s, e, c, pars=pars)
    reg.vprint('All spatial parameters computed')

@reg.funcs.proc("length")
def comp_length(s, e, c, mode='minimal', recompute=False):
    if 'length' in e.columns.values and not recompute:
        reg.vprint('Length is already computed. If you want to recompute it, set recompute_length to True',1)
        return
    if not c.midline_xy.exist_in(s):
        reg.vprint(f'XY coordinates not found for the {c.Npoints} midline points. Body length can not be computed.',1)
        return
    xy = s[c.midline_xy].values

    if mode == 'full':
        segs = c.midline_segs
        t = len(s)
        S = np.zeros([c.Nsegs, t]) * np.nan
        L = np.zeros([1, t]) * np.nan
        reg.vprint(f'Computing lengths for {c.Nsegs} segments and total body length',1)
        for j in range(t):
            for i, seg in enumerate(segs):
                S[i, j] = np.sqrt(np.nansum((xy[j, 2 * i:2 * i + 2] - xy[j, 2 * i + 2:2 * i + 4]) ** 2))
            L[:, j] = np.nansum(S[:, j])
        for i, seg in enumerate(segs):
            s[seg] = S[i, :].flatten()
    elif mode == 'minimal':
        reg.vprint(f'Computing body length')
        xy2 = xy.reshape(xy.shape[0], c.Npoints, 2)
        xy3 = np.sum(np.diff(xy2, axis=1) ** 2, axis=2)
        L = np.sum(np.sqrt(xy3), axis=1)
    s['length'] = L
    e['length'] = s['length'].groupby('AgentID').quantile(q=0.5)
    reg.vprint('All lengths computed.',1)

@reg.funcs.proc("centroid")
def comp_centroid(s, c, recompute=False):
    if c.centroid_xy.exist_in(s) and not recompute:
        reg.vprint('Centroid is already computed. If you want to recompute it, set recompute_centroid to True')
    if not c.contour_xy.exist_in(s) or c.Ncontour == 0:
        reg.vprint(f'No contour found. Not computing centroid')
    else:
        reg.vprint(f'Computing centroid from {c.Ncontour} contourpoints')
        s[c.centroid_xy] = np.sum(s[c.contour_xy].values.reshape([-1, c.Ncontour, 2]), axis=1)/c.Ncontour
    reg.vprint('Centroid coordinates computed.')


def store_spatial(s, e, c, d=None):
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

    if d is not None :
        try :
            d.store(s[aux.existing_cols([dst, cdst, sdst, csdst],s)], 'pathlength')
            d.store(s[['x', 'y']], 'traj.default')
        except:
            pass



@reg.funcs.proc("spatial")
def spatial_processing(s, e, c, d=None,mode='minimal', recompute=False, traj2origin=True, **kwargs):
    assert isinstance(c, reg.DatasetConfig)

    comp_length(s, e, c, mode=mode, recompute=recompute)
    comp_centroid(s, c, recompute=recompute)
    comp_spatial(s, e, c, mode=mode)
    store_spatial(s, e, c,d=d)
    if traj2origin :
        try:
            align_trajectories(s, c,d=d,  replace=False, transposition='origin')
        except :
            pass

    reg.vprint(f'Completed {mode} spatial processing.',1)


@reg.funcs.proc("dispersion")
def comp_dispersion(s, e, c,d=None, dsp_starts=[0], dsp_stops=[40], **kwargs):
    if dsp_starts is None or dsp_stops is None:
        return

    xy0 = d.load_traj(mode='default') if s is None else s[['x', 'y']]
    dsps = {}
    for t0, t1 in itertools.product(dsp_starts, dsp_stops):
        p = reg.getPar(f'dsp_{int(t0)}_{int(t1)}')
        s[p],Nt=aux.compute_dispersal_multi(xy0, t0, t1, c.dt)

        s0 = int(t0 / c.dt)

        fp = nam.final(p)
        mp = nam.max(p)
        mup = nam.mean(p)
        temp=s[p].dropna().groupby('AgentID')
        e[mp] = temp.max()
        e[mup] = temp.mean()
        e[fp] = temp.last()
        scale_to_length(s, e, c, pars=[p, fp, mp, mup])
        for par in [p, nam.scal(p)]:
            dsps[par] = get_disp_df(s[par], s0, Nt)


    aux.save_dict(dsps, f'{c.dir}/data/dsp.txt')

    reg.vprint(f'Completed dispersal processing.',1)


def get_disp_df(dsp, s0, Nt):
    trange = np.arange(s0, s0 + Nt, 1)
    dsp_ar = np.zeros([Nt, 3]) * np.nan
    dsp_ar[:, 0] = dsp.groupby(level='Step').quantile(q=0.5).values[s0:s0 + Nt]
    dsp_ar[:, 1] = dsp.groupby(level='Step').quantile(q=0.75).values[s0:s0 + Nt]
    dsp_ar[:, 2] = dsp.groupby(level='Step').quantile(q=0.25).values[s0:s0 + Nt]
    return pd.DataFrame(dsp_ar, index=trange, columns=['median', 'upper', 'lower'])

# def comp_tortuosity(s, e, dt, tor_durs=[2, 5, 10, 20], **kwargs):
#     '''
#     Trajectory tortuosity metrics
#     In the simplest case a single value is computed as T=1-D/L where D is the dispersal and L the actual pathlength.
#     This metric has been used in :
#     [1] J. Loveless and B. Webb, “A Neuromechanical Model of Larval Chemotaxis,” Integr. Comp. Biol., vol. 58, no. 5, pp. 906–914, 2018.
#     Additionally tortuosity can be computed over a given time interval in which case the result is a vector called straightness index in [2].
#     The mean and std are then computed.
#
#     TODO Check also for binomial distribution over the straightness index vector. If there is a switch between global exploration and local search there should be evidence over a certain time interval.
#     Data from here is relevant :
#     [2] D. W. Sims, N. E. Humphries, N. Hu, V. Medan, and J. Berni, “Optimal searching behaviour generated intrinsically by the central pattern generator for locomotion,” Elife, vol. 8, pp. 1–31, 2019.
#     '''
#     if tor_durs is None:
#         return
#     try:
#         dsp_par = nam.final('dispersion') if nam.final('dispersion') in e.columns else 'dispersion'
#         e['tortuosity'] = 1 - e[dsp_par] / e[nam.cum(nam.dst(''))]
#     except:
#         pass
#     durs = [int(1 / dt * d) for d in tor_durs]
#     Ndurs = len(durs)
#     if Ndurs > 0:
#         ids = s.index.unique('AgentID').values
#         Nids = len(ids)
#         ds = [s[['x', 'y']].xs(id, level='AgentID') for id in ids]
#         ds = [d.loc[d.first_valid_index(): d.last_valid_index()].values for d in ds]
#         for j, r in enumerate(durs):
#             par = f'tortuosity_{tor_durs[j]}'
#             par_m, par_s = nam.mean(par), nam.std(par)
#             T_m = np.ones(Nids) * np.nan
#             T_s = np.ones(Nids) * np.nan
#             for z, id in enumerate(ids):
#                 si = ds[z]
#                 u = len(si) % r
#                 if u > 1:
#                     si0 = si[:-u + 1]
#                 else:
#                     si0 = si[:-r + 1]
#                 k = int(len(si0) / r)
#                 T = []
#                 for i in range(k):
#                     t = si0[i * r:i * r + r + 1, :]
#                     if np.isnan(t).any():
#                         continue
#                     else:
#                         t_D = np.sum(np.sqrt(np.sum(np.diff(t, axis=0) ** 2, axis=1)))
#                         t_L = np.sqrt(np.sum(np.array(t[-1, :] - t[0, :]) ** 2))
#                         t_T = 1 - t_L / t_D
#                         T.append(t_T)
#                 T_m[z] = np.mean(T)
#                 T_s[z] = np.std(T)
#             e[par_m] = T_m
#             e[par_s] = T_s
#
#     reg.vprint('Tortuosities computed')


def rolling_window(a, w):
    # Get windows of size w from array a
    if a.ndim != 1:
        raise ValueError("Input array must be 1-dimensional")
    return np.vstack([np.roll(a, -i) for i in range(w)]).T[:-w + 1]


def rolling_window_xy(xy, w):
    # Get windows of size w from 2D array xy
    xs = rolling_window(xy[:, 0], w)
    ys = rolling_window(xy[:, 1], w)
    xys = np.dstack([xs, ys])
    return xys

def straightness_index(ss, rolling_ticks):
    ps=['x', 'y', 'dst']
    assert aux.cols_exist(ps,ss)
    sss=ss[ps].values
    temp=sss[rolling_ticks]
    Ds = np.nansum(temp[:, :, 2], axis=1)
    xys = temp[:, :, :2]

    k0, k1 = len(ss), rolling_ticks.shape[0]
    dk = int((k0 - k1) / 2)
    SI0 = np.zeros(k0) * np.nan
    for i in range(k1) :
        D=Ds[i]
        if D!= 0:
            xy = xys[i][~np.isnan(xys[i]).any(axis=1)]
            if xy.shape[0] >= 2:
                L = np.sqrt(np.nansum(np.array(xy[-1, :] - xy[0, :]) ** 2))
                SI0[dk+i] = 1 - L / D
    return SI0


# def straightness_index2(ss, w, match_shape=True):
#     xy0 = ss[['x', 'y']].values
#     # Compute tortuosity over intervals of duration w
#     xys = rolling_window_xy(xy0, w)
#
#
#     k0, k1 = xy0.shape[0], xys.shape[0]
#     l_xys = np.array([xys[i, :][~np.isnan(xys[i, :]).any(axis=1)] for i in range(k1)])
#     valid=np.where([l_xys[i].shape[0]>= 2 for i in range(k1)])[0]
#
#     if 'dst' in ss.columns:
#         dst = ss['dst'].values
#         Ds = np.nansum(rolling_window(dst, w), axis=1)[valid]
#
#     else:
#         Ds = np.array([np.nansum(np.sqrt(np.nansum(np.diff(l_xys[i], axis=0) ** 2, axis=1))) for i in valid])
#
#     valid2=np.where(Ds!=0)[0]
#     Ds=Ds[valid2]
#
#     valid_fin=valid[valid2]
#
#     dxys=np.array([l_xys[i][-1, :] -l_xys[i][0, :] for i in valid_fin])
#     Ls = np.sqrt(np.nansum(dxys ** 2, axis=1))
#     A=1 - Ls / Ds
#
#     if match_shape:
#         dk = int((k0 - k1) / 2)
#         SI0 = np.zeros(k0) * np.nan
#         SI0[dk+valid_fin] = A
#         return SI0
#     else:
#         SI = np.zeros(k1) * np.nan
#         SI[valid_fin] = A
#
#         return SI
#     '''
#         if match_shape:
#         dk = int((k0 - k1) / 2)
#         SI = np.zeros(k0) * np.nan
#         for i in range(k1):
#             SI[dk + i] = tortuosity(xys[i, :])
#     else:
#         SI = np.zeros(k1) * np.nan
#         for i in range(k1):
#             SI[i] = tortuosity(xys[i, :])
#     return SI
#     '''


@reg.funcs.proc("tortuosity")
def comp_straightness_index(s=None, e=None, c=None,d=None, dt=None, tor_durs=[1, 2, 5, 10, 20], **kwargs):
    if dt is None:
        dt = c.dt

    if s is None:
        s = pd.read_hdf(f'{c.dir}/data/data.h5', key='step')

    ticks=np.arange(c.Nticks)
    ps = ['x', 'y', 'dst']
    assert aux.cols_exist(ps,s)

    ss = s[ps]
    pars = [reg.getPar(f'tor{dur}') for dur in tor_durs]
    for dur, p in zip(tor_durs, pars):
        w = int(dur / dt / 2)
        rolling_ticks=rolling_window(ticks, w)
        s[p] = aux.apply_per_level(ss, straightness_index, rolling_ticks=rolling_ticks).flatten()
        if e is not None:
            e[nam.mean(p)] = s[p].groupby('AgentID').mean()
            e[nam.std(p)] = s[p].groupby('AgentID').std()

    reg.vprint(f'Completed tortuosity processing.',1)

@reg.funcs.proc("source")
def comp_source_metrics(s, e, c, **kwargs):
    fo = reg.getPar('fo')
    xy = nam.xy('')
    for n, pos in c.source_xy.items():
        reg.vprint(f'Computing bearing and distance to {n} based on xy position')
        o, d = nam.bearing_to(n), nam.dst_to(n)
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

        reg.vprint('Bearing and distance to source computed')

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
        s[nam.bearing_to('wind')] = s.apply(lambda r: aux.angle_dif(r[nam.orient('front')], wo), axis=1)
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


@reg.funcs.preproc("transposition")
def align_trajectories(s, c, d=None, track_point=None, arena_dims=None, transposition='origin', replace=True, **kwargs):
    if not isinstance(c, reg.DatasetConfig):
        c=reg.DatasetConfig(**c)

    if transposition in ['', None, np.nan]:
        return
    mode=transposition

    xy_flat=c.all_xy.existing(s)
    xy_pairs = xy_flat.in_pairs
    # xy_flat=np.unique(aux.flatten_list(xy_pairs))
    # xy_pairs = aux.group_list_by_n(xy_flat, 2)

    if replace :
        ss=s
    else :
        ss = copy.deepcopy(s[xy_flat])


    if mode == 'arena':
        reg.vprint('Centralizing trajectories in arena center')
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
        XY = nam.xy(track_point) if aux.cols_exist(nam.xy(track_point),s) else ['x', 'y']
        if not aux.cols_exist(XY,s):
            raise ValueError('Defined point xy coordinates do not exist. Can not align trajectories! ')
        ids = s.index.unique(level='AgentID').values
        Nticks = len(s.index.unique('Step'))
        if mode == 'origin':
            reg.vprint('Aligning trajectories to common origin')
            xy = [s[XY].xs(id, level='AgentID').dropna().values[0] for id in ids]
        elif mode == 'center':
            reg.vprint('Centralizing trajectories in trajectory center using min-max positions')
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

        if d is not None :
            d.store(ss, f'traj.{mode}')
            reg.vprint(f'traj_aligned2{mode} stored')
        return ss


@reg.funcs.preproc("fixation")
def fixate_larva(s, c, P1, P2=None):
    if not isinstance(c, reg.DatasetConfig):
        c = reg.DatasetConfig(**c)



    pars = c.all_xy.existing(s)
    if not nam.xy(P1).exist_in(s):
        raise ValueError(f" The requested {P1} is not part of the dataset")
    reg.vprint(f'Fixing {P1} to arena center')
    X, Y = c.env_params.arena.dims
    xy = s[nam.xy(P1)].values
    xy_start = s[nam.xy(P1)].dropna().values[0]
    bg_x = (xy[:, 0] - xy_start[0]) / X
    bg_y = (xy[:, 1] - xy_start[1]) / Y

    for x, y in pars.in_pairs:
        s[[x, y]] -= xy

    N = s.index.unique('Step').size
    if P2 is not None:
        if not nam.xy(P2).exist_in(s):
            raise ValueError(f" The requested secondary {P2} is not part of the dataset")
        reg.vprint(f'Fixing {P2} as secondary point on vertical axis')
        xy_sec = s[nam.xy(P2)].values
        bg_a = np.arctan2(xy_sec[:, 1], xy_sec[:, 0])- np.pi / 2

        s[pars] = [
            aux.flatten_list(aux.rotate_points_around_point(points=np.reshape(s[pars].values[i,:], (-1, 2)),
                                                            radians=bg_a[i])) for i in range(N)]
    else:
        bg_a = np.zeros(N)


    bg = np.vstack((bg_x, bg_y, bg_a))
    reg.vprint('Fixed-point dataset generated')

    return s, bg


def comp_PI2(xys, x=0.04):
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


def comp_PI(arena_xdim, xs, return_num=False):
    N = len(xs)
    r = 0.2 * arena_xdim
    xs = np.array(xs)
    N_l = len(xs[xs <= -r / 2])
    N_r = len(xs[xs >= +r / 2])
    # N_m = len(xs[(xs <= +r / 2) & (xs >= -r / 2)])
    pI = np.round((N_l - N_r) / N, 3)
    if return_num:
        return pI, N
    else:
        return pI


@reg.funcs.proc("PI")
def comp_dataPI(s,e,c, **kwargs):
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
    PI, N = comp_PI(xs=xs, arena_xdim=c.env_params.arena.dims[0], return_num=True)
    c.PI = {'PI': PI, 'N': N}
    try:
        c.PI2 = comp_PI2(xys=s[nam.xy('')])
    except:
        pass




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
    s_pars = aux.existing_cols(pars,s)

    if len(s_pars) > 0:
        ids = s.index.get_level_values('AgentID').values
        ls = l.loc[ids].values
        s[nam.scal(s_pars)] = (s[s_pars].values.T / ls).T
    e_pars = aux.existing_cols(pars,e)
    if len(e_pars) > 0:
        e[nam.scal(e_pars)] = (e[e_pars].values.T / l.values).T
