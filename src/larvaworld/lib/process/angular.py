import numpy as np
import pandas as pd
import scipy

from larvaworld.lib import reg, aux



def comp_angles(s, Npoints, mode='full', front_body_ratio=0.5):
    if Npoints<=2 :
        return []
    Nangles = Npoints - 2
    angles = [f'angle{i}' for i in range(Nangles)]
    bend_angles = angles[:int(np.round(front_body_ratio * Nangles))]

    if mode == 'minimal':
        angles = bend_angles

    Axy = s[aux.nam.midline_xy(Npoints, flat=True)].values

    N = Axy.shape[0]
    Axy = np.reshape(Axy, (N, Npoints, 2))
    A = np.zeros([Nangles, N]) * np.nan
    for i in range(N):
        A[:, i] = np.array([aux.angle_from_3points(Axy[i, j + 2, :], Axy[i, j + 1, :], Axy[i, j, :]) for j in range(Nangles)])
    for z, a in enumerate(angles):
        s[a] = A[z].T
    print('All angles computed')
    return bend_angles


def comp_bend(s, c, mode='minimal'):
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
        s['bend'] = s.apply(lambda r: aux.angle_dif(r[aux.nam.orient('front')], r[aux.nam.orient('rear')]), axis=1)
    elif b_conf == 'from_angles':
        bend_angles = comp_angles(s, c.Npoints, mode=mode)
        print(f'Computing bending angle as the sum of the first {len(bend_angles)} front angles')
        s['bend'] = s[bend_angles].sum(axis=1, min_count=1)

    print('All bends computed')


def compute_LR_bias(s, e):
    for id in s.index.unique('AgentID').values:
        for p in ['bend', aux.nam.vel('bend'), aux.nam.vel(aux.nam.orient('front'))]:
            if p in s.columns:
                b = s[p].xs(id, level='AgentID', drop_level=True).dropna()
                e.loc[id, aux.nam.mean(p)] = b.mean()
                e.loc[id, aux.nam.std(p)] = b.std()
    print('LR biases computed')

def comp_orientations(s, e, c, mode='minimal'):
    N = s.values.shape[0]
    Np = c.Npoints
    if Np == 1:
        comp_orientation_1point(s, e)
        return

    temp=c.metric_definition.angular.hardcoded
    for key in ['front_vector', 'rear_vector']:
        if temp[key] is None:
            print('Front and rear vectors are not defined. Can not compute orients')
            return
    else:
        f1, f2 = temp.front_vector
        r1, r2 = temp.rear_vector

    xy = aux.nam.midline_xy(Np)
    print(f'Computing front/rear body-vector and head/tail orientation angles')
    pars=aux.nam.orient(['front','rear', 'head', 'tail'])
    Npars=len(pars)

    xy_pars = aux.flatten_list([xy[i] for i in [f2 - 1, f1 - 1, r2 - 1, r1 - 1, 1,0,-1,-2]])
    xy_ar = np.reshape(s[xy_pars].values, (N, Npars*2, 2))



    A = np.zeros([Npars, N]) * np.nan
    for i in range(N):
        for j in range(Npars):
            A[j, i] = aux.angle_to_x_axis(xy_ar[i, 2 * j, :], xy_ar[i, 2 * j + 1, :])
    for z, a in enumerate(pars):
        s[a] = A[z].T
        e[aux.nam.initial(a)] = s[a].dropna().groupby('AgentID').first()


    if mode == 'full':
        print(f'Computing additional orients for {Np-1} spinesegments')
        xy_pars = aux.flatten_list([xy[i] for i in range(Np)])
        xy_ar = np.reshape(s[xy_pars].values, (N, Np, 2))
        A = np.zeros([Np-1, N]) * np.nan
        for i in range(N):
            A[:, i] = np.array([aux.angle_to_x_axis(xy_ar[i, j + 1, :], xy_ar[i, j, :]) for j in range(Np-1)])
        for z, a in enumerate(aux.nam.orient(aux.nam.midline(Np - 1, type='seg'))):
            s[a] = A[z].T
    print('All orientations computed')
    return



def comp_orientation_1point(s, e):
    fov = aux.nam.orient('front')
    N = len(s.index.unique('Step'))

    def func(ss) :
        AA = np.zeros(N) * np.nan
        for i in range(N - 1):
            AA[i + 1] = aux.angle_to_x_axis(ss[i, :].values, ss[i + 1, :].values, in_deg=True)
        return AA

    s[fov]= aux.apply_per_level(s[['x', 'y']], func).flatten()
    e[aux.nam.initial(fov)] = s[fov].dropna().groupby('AgentID').first()
    print('All orientations computed')
    return

@reg.funcs.proc("unwrap")
def unwrap_orientations(s, segs):
    def unwrap_deg(ss):
        if isinstance(ss, pd.Series) :
            ss=ss.values
        b = np.copy(ss)
        b[~np.isnan(b)] = np.unwrap(b[~np.isnan(b)] * np.pi / 180) * 180 / np.pi
        return b

    for p in aux.nam.orient(['front', 'rear']+segs):
        if p in s.columns.values:
            s[aux.nam.unwrap(p)] = aux.apply_per_level(s[p], unwrap_deg).flatten()
    print('All orients unwrapped')

# def comp_angular2(s, c, mode='minimal'):
#     ang_pars = aux.nam.orient(['front','rear', 'head', 'tail']) + ['bend']
#     segs = aux.nam.midline(c.Npoints - 1, type='seg')
#     unwrap_orientations(s, segs)
#     if mode == 'full':
#         Nangles = np.clip(c.Npoints - 2, a_min=0, a_max=None)
#         angles = [f'angle{i}' for i in range(Nangles)]
#         pars = angles + aux.nam.orient(segs) + ang_pars
#     elif mode == 'minimal':
#         pars = ang_pars
#     pars = [a for a in pars if a in s.columns]
#     Npars = len(pars)
#     print(f'Computing angular velocities and accelerations for {Npars} angular parameters')
#
#     ids = s.index.unique('AgentID').values
#     Nids = len(ids)
#     Nticks = len(s.index.unique('Step'))
#
#
#     V = np.zeros([Nticks, Npars, Nids]) * np.nan
#     A = np.zeros([Nticks, Npars, Nids]) * np.nan
#
#     all_d = [s.xs(id, level='AgentID', drop_level=True) for id in ids]
#
#     for i, p in enumerate(pars):
#         if aux.nam.unwrap(p) in s.columns:
#             p = aux.nam.unwrap(p)
#         for j, d in enumerate(all_d):
#             avel = np.diff(d[p].values) / c.dt
#             aacc = np.diff(avel) / c.dt
#             V[1:, i, j] = avel
#             A[2:, i, j] = aacc
#     for k, p in enumerate(pars):
#         s[aux.nam.vel(p)] = V[:, k, :].flatten()
#         s[aux.nam.acc(p)] = A[:, k, :].flatten()
#     print('All angular parameters computed')


@reg.funcs.proc("ang_moments")
def comp_angular(s, dt,Npoints, mode='minimal'):
    ang_pars = aux.nam.orient(['front','rear', 'head', 'tail']) + ['bend']
    segs = aux.nam.midline(Npoints - 1, type='seg')
    unwrap_orientations(s, segs)
    if mode == 'full':
        Nangles = np.clip(Npoints - 2, a_min=0, a_max=None)
        angles = [f'angle{i}' for i in range(Nangles)]
        pars = angles + aux.nam.orient(segs) + ang_pars
    elif mode == 'minimal':
        pars = ang_pars
    else :
        raise ValueError ('Not implemented')

    for p in pars:
        if p in s.columns :
            pvel=aux.nam.vel(p)
            avel=aux.nam.acc(p)
            if aux.nam.unwrap(p) in s.columns:
                p = aux.nam.unwrap(p)
            s[pvel] = aux.apply_per_level(s[p], aux.rate, dt=dt).flatten()
            s[avel] = aux.apply_per_level(s[pvel], aux.rate, dt=dt).flatten()
    print('All angular parameters computed')

@reg.funcs.proc("angular")
def angular_processing(s, e, c, recompute=False, mode='minimal', store=False, **kwargs):
    ang_pars = [aux.nam.orient('front'), aux.nam.orient('rear'), 'bend']
    if set(ang_pars).issubset(s.columns.values) and not recompute:
        print('Orientation and bend are already computed. If you want to recompute them, set recompute to True')
    else:
        comp_orientations(s, e, c, mode=mode)
        comp_bend(s, c, mode=mode)
        # try:
        #     comp_orientations(s, e, c, mode=mode)
        #     comp_bend(s, c, mode=mode)
        # except:
        #     comp_ang_from_xy(s, e, dt=c.dt)
    comp_angular(s, c.dt,c.Npoints, mode=mode)
    comp_extrema(s, dt=c.dt, parameters=[aux.nam.vel(aux.nam.orient('front'))], interval_in_sec=0.3)
    compute_LR_bias(s, e)
    if store :
        pars=ang_pars + aux.nam.vel(ang_pars) + aux.nam.acc(ang_pars)
        dic = aux.get_distros(s, pars=pars)
        aux.storeH5(dic, path=reg.datapath('distro', c.dir))

    print(f'Completed {mode} angular processing.')


def ang_from_xy(xy):
    dx_dt = np.gradient(xy[:,0])
    dy_dt = np.gradient(xy[:,1])
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
    N = s.index.unique('Step').size
    p = aux.nam.orient('front')
    p_vel, p_acc = aux.nam.vel(p), aux.nam.acc(p)
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    V = np.zeros([N, 1, Nids]) * np.nan
    A = np.zeros([N, 1, Nids]) * np.nan
    for j, id in enumerate(ids):
        xy = s[["x", "y"]].xs(id, level='AgentID').values
        avel, aacc = ang_from_xy(xy)
        V[:, 0, j] = avel / dt
        A[:, 0, j] = aacc / dt
    s[p_vel] = V[:, 0, :].flatten()
    s[p_acc] = A[:, 0, :].flatten()
    e[aux.nam.mean(p_vel)] = s[p_vel].dropna().groupby('AgentID').mean()
    e[aux.nam.mean(p_acc)] = s[p_acc].dropna().groupby('AgentID').mean()


def comp_extrema(s, dt, parameters, interval_in_sec, threshold_in_std=None, abs_threshold=None):
    N = s.index.unique('Step').size
    if abs_threshold is None:
        abs_threshold = [+np.inf, -np.inf]
    order = np.round(interval_in_sec / dt).astype(int)
    ids = s.index.unique('AgentID').values
    Nids = len(ids)
    Npars = len(parameters)
    # Nticks = len(s.index.unique('Step'))
    t0 = s.index.unique('Step').min()

    min_array = np.ones([N, Npars, Nids]) * np.nan
    max_array = np.ones([N, Npars, Nids]) * np.nan
    for i, p in enumerate(parameters):
        p_min, p_max = aux.nam.min(p), aux.nam.max(p)
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
            i_min = scipy.signal.argrelextrema(df.values, np.less_equal, order=order)[0]
            i_max = scipy.signal.argrelextrema(df.values, np.greater_equal, order=order)[0]

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
