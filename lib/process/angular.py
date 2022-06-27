import numpy as np
import pandas as pd

from lib.aux.ang_aux import angle_dif, angle, angle_to_x_axis, unwrap_deg
from lib.aux.dictsNlists import flatten_list
import lib.aux.naming as nam

from lib.process.store import store_aux_dataset


def comp_angles(s, e, c, mode='full'):
    N=c.Npoints
    points = nam.midline(N, type='point')
    Nangles = np.clip(N - 2, a_min=0, a_max=None)
    angles = [f'angle{i}' for i in range(Nangles)]
    ang_conf=c.metric_definition.angular
    if ang_conf.fitted is None :
        r = ang_conf.hardcoded.front_body_ratio
    else :
        r = ang_conf.fitted.front_body_ratio
    bend_angles = angles[:int(np.round(r * Nangles))]
    xy = [nam.xy(points[i]) for i in range(N)]
    if mode == 'full':
        angles = angles
    elif mode == 'minimal':
        angles = bend_angles
    print(f'Computing {Nangles} angles')
    xy_pars = flatten_list([xy[i] for i in range(Nangles + 2)])
    Axy = s[xy_pars].values
    Npoints = int(Axy.shape[1] / 2)
    Nticks = Axy.shape[0]
    Axy = np.reshape(Axy, (Nticks, Npoints, 2))
    A = np.zeros([Nangles, Nticks]) * np.nan
    for i in range(Nticks):
        A[:, i] = np.array([angle(Axy[i, j + 2, :], Axy[i, j + 1, :], Axy[i, j, :]) for j in range(Nangles)])
    for z, a in enumerate(angles):
        s[a] = A[z].T
    print('All angles computed')
    return bend_angles


def comp_bend(s, e, c, mode='minimal'):
    ang_conf = c.metric_definition.angular
    if ang_conf.fitted is None:
        b_conf = ang_conf.hardcoded.bend
    else:
        b_conf = ang_conf.fitted.bend

    if b_conf is None:
        print('Bending angle not defined. Can not compute angles')
        return
    elif b_conf == 'from_vectors':
        print(f'Computing bending angle as the difference between front and rear orients')
        s['bend'] = s.apply(lambda r: angle_dif(r[nam.orient('front')], r[nam.orient('rear')]), axis=1)
    elif b_conf == 'from_angles':
        bend_angles = comp_angles(s, e, c, mode=mode)
        print(f'Computing bending angle as the sum of the first {len(bend_angles)} front angles')
        s['bend'] = s[bend_angles].sum(axis=1, min_count=1)

    print('All bends computed')


def compute_LR_bias(s, e):
    for id in s.index.unique('AgentID').values:
        for p in ['bend', nam.vel('bend'), nam.vel(nam.orient('front'))]:
            if p in s.columns:
                b = s[p].xs(id, level='AgentID', drop_level=True).dropna()
                e.loc[id, nam.mean(p)] = b.mean()
                e.loc[id, nam.std(p)] = b.std()
    print('LR biases computed')



def comp_orientations(s, e, c, mode='minimal'):
    Np = c.Npoints
    if Np == 1:
        comp_orientation_1point(s, e, c)
        return

    points = nam.midline(Np, type='point')
    segs = nam.midline(Np - 1, type='seg')
    temp=c.metric_definition.angular.hardcoded
    for key in ['front_vector', 'rear_vector']:
        if temp[key] is None:
            print('Front and rear vectors are not defined. Can not compute orients')
            return
    else:
        f1, f2 = temp.front_vector
        r1, r2 = temp.rear_vector

    xy = [nam.xy(points[i]) for i in range(len(points))]
    print(f'Computing front/rear body-vector and head/tail orientation angles')
    pars=nam.orient(['front','rear', 'head', 'tail'])
    Npars=len(pars)

    xy_pars = flatten_list([xy[i] for i in [f2 - 1, f1 - 1, r2 - 1, r1 - 1, 1,0,-1,-2]])
    xy_ar = s[xy_pars].values
    Nticks = xy_ar.shape[0]
    xy_ar = np.reshape(xy_ar, (Nticks, Npars*2, 2))



    cc = np.zeros([Npars, Nticks]) * np.nan
    for i in range(Nticks):
        for j in range(Npars):
            cc[j, i] = angle_to_x_axis(xy_ar[i, 2 * j, :], xy_ar[i, 2 * j + 1, :])
    for z, a in enumerate(pars):
        s[a] = cc[z].T
        e[nam.initial(a)] = s[a].dropna().groupby('AgentID').first()


    if mode == 'full':
        N = len(segs)
        print(f'Computing additional orients for {N} spinesegments')
        ors = nam.orient(segs)
        xy_pars = flatten_list([xy[i] for i in range(N + 1)])
        xy_ar = s[xy_pars].values
        # Npoints = int(xy_ar.shape[1] / 2)
        Nticks = xy_ar.shape[0]
        xy_ar = np.reshape(xy_ar, (Nticks, N*2, 2))
        cc = np.zeros([N, Nticks]) * np.nan
        for i in range(Nticks):
            cc[:, i] = np.array([angle_to_x_axis(xy_ar[i, j + 1, :], xy_ar[i, j, :]) for j in range(N)])
        for z, a in enumerate(ors):
            s[a] = cc[z].T
    print('All orientations computed')
    return


def comp_orientation_1point(s, e, c):
    if c.Npoints != 1:
        return
    fov = nam.orient('front')
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Nticks = len(s.index.unique('Step'))
    c = np.zeros([Nticks, Nids]) * np.nan
    for j, id in enumerate(ids):
        xy = s[['x', 'y']].xs(id, level='AgentID').values
        for i in range(Nticks - 1):
            c[i + 1, j] = angle_to_x_axis(xy[i, :], xy[i + 1, :], in_deg=True)
    s[fov] = c.flatten()
    e[nam.initial(fov)] = s[fov].dropna().groupby('AgentID').first()
    print('All orientations computed')
    return


def unwrap_orientations(s, segs):
    pars = list(set([p for p in [nam.orient('front'), nam.orient('rear')] + nam.orient(segs) if p in s.columns.values]))
    for p in pars:
        for id in s.index.unique('AgentID').values:
            ts = s.loc[(slice(None), id), p].values
            s.loc[(slice(None), id), nam.unwrap(p)] = unwrap_deg(ts)
    print('All orients unwrapped')


def comp_angular(s, e, c, mode='minimal'):
    ors = nam.orient(['front','rear', 'head', 'tail'])
    ang_pars = ors + ['bend']

    dt = c.dt
    Nangles = np.clip(c.Npoints - 2, a_min=0, a_max=None)
    angles = [f'angle{i}' for i in range(Nangles)]
    segs = nam.midline(c.Npoints - 1, type='seg')

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

    for i, p in enumerate(pars):
        if nam.unwrap(p) in s.columns:
            p = nam.unwrap(p)
        for j, d in enumerate(all_d):
            avel = np.diff(d[p].values) / dt
            aacc = np.diff(avel) / dt
            V[1:, i, j] = avel
            A[2:, i, j] = aacc
    for k, p in enumerate(pars):
        s[nam.vel(p)] = V[:, k, :].flatten()
        s[nam.acc(p)] = A[:, k, :].flatten()
    print('All angular parameters computed')


def angular_processing(s, e, c, recompute=False, mode='minimal', store=False, **kwargs):
    from lib.process.basic import comp_extrema
    ang_pars = [nam.orient('front'), nam.orient('rear'), 'bend']
    if set(ang_pars).issubset(s.columns.values) and not recompute:
        print('Orientation and bend are already computed. If you want to recompute them, set recompute to True')
    else:
        try:
            comp_orientations(s, e, c, mode=mode)
            comp_bend(s, e, c, mode=mode)
        except:
            comp_ang_from_xy(s, e, dt=c.dt)
    comp_angular(s, e, c, mode=mode)
    comp_extrema(s, dt=c.dt, parameters=[nam.vel(nam.orient('front'))], interval_in_sec=0.3)
    compute_LR_bias(s, e)
    if store :
        store_aux_dataset(s, pars=ang_pars + nam.vel(ang_pars) + nam.acc(ang_pars), type='distro', file=c.aux_dir)
    print(f'Completed {mode} angular processing.')


def ang_from_xy(x, y):
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    velocity = np.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    tangent = np.array([1 / ds_dt] * 2).transpose() * velocity
    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]

    deriv_tangent_x = np.gradient(tangent_x)
    deriv_tangent_y = np.gradient(tangent_y)

    dT_dt = np.array([[deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])

    length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)

    normal = np.array([1 / length_dT_dt] * 2).transpose() * dT_dt
    d2s_dt2 = np.gradient(ds_dt)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    t_component = np.array([d2s_dt2] * 2).transpose()
    n_component = np.array([curvature * ds_dt * ds_dt] * 2).transpose()

    acceleration = t_component * tangent + n_component * normal

    ang_vel = np.arctan(normal[:, 0] / normal[:, 1])

    ang_acc = np.arctan(acceleration[:, 0] / acceleration[:, 1])
    return ang_vel, ang_acc


def comp_ang_from_xy(s, e, dt):
    p = nam.orient('front')
    p_vel, p_acc = nam.vel(p), nam.acc(p)
    s[p_vel] = np.nan
    s[p_acc] = np.nan
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Nticks = len(s.index.unique('Step'))

    V = np.zeros([Nticks, 1, Nids]) * np.nan
    A = np.zeros([Nticks, 1, Nids]) * np.nan
    for j, id in enumerate(ids):
        x = s["x"].xs(id, level='AgentID').values
        y = s["y"].xs(id, level='AgentID').values
        avel, aacc = ang_from_xy(x, y)
        V[:, 0, j] = avel / dt
        A[:, 0, j] = aacc / dt
    s[p_vel] = V[:, 0, :].flatten()
    s[p_acc] = A[:, 0, :].flatten()
    e[nam.mean(p_vel)] = s[p_vel].dropna().groupby('AgentID').mean()
    e[nam.mean(p_acc)] = s[p_acc].dropna().groupby('AgentID').mean()

