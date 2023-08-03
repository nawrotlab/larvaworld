import numpy as np
import pandas as pd
import scipy

from larvaworld.lib import reg, aux


def comp_orientations(s, e, c, mode='minimal'):
    Np = c.Npoints
    if Np == 1:
        comp_orientation_1point(s, e)
        return

    f1, f2 = c.front_vector
    r1, r2 = c.rear_vector


    xy_pars = c.midline_xy
    Axy=s[xy_pars].values

    reg.vprint(f'Computing front/rear body-vector and head/tail orientation angles')
    vector_idx={
        'front' : (f2 - 1, f1 - 1),
        'rear' : (r2 - 1, r1 - 1),
        'head' : (1,0),
        'tail' : (-1,-2),
    }

    if mode == 'full':
        reg.vprint(f'Computing additional orients for {c.Nsegs} spinesegments')
        for i, vec in enumerate(c.midline_segs):
             vector_idx[vec] = (i+1,i)

    for vec, (idx1, idx2) in vector_idx.items() :
        par = aux.nam.orient(vec)
        x, y = Axy[:,2*idx2]-Axy[:,2*idx1], Axy[:,2*idx2+1]-Axy[:,2*idx1+1]
        aa = np.arctan2(y, x)
        aa[aa < 0] += 2 * np.pi
        s[par] = aa
        e[aux.nam.initial(par)] = s[par].dropna().groupby('AgentID').first()
    reg.vprint('All orientations computed')



def comp_orientation_1point(s, e):
    def func(ss) :
        x,y=ss[:, 0].values, ss[:, 1].values
        dx,dy=np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)
        aa = np.arctan2(dy, dx)
        aa[aa < 0] += 2 * np.pi
        return aa

    return aux.apply_per_level(s[['x', 'y']], func).flatten()




@reg.funcs.proc("ang_moments")
def comp_angular(s,e, c, pars=None, **kwargs):
    vecs = ['head', 'tail', 'front', 'rear']
    ho,to,fo,ro=aux.nam.orient(vecs)
    if pars is None :
        if c.Npoints > 3 :
            base_pars = ['bend', ho, to, fo, ro]
            segs = c.midline_segs
            ang_pars = [f'angle{i}' for i in range(c.Nangles)]
            pars = base_pars + ang_pars+aux.nam.orient(segs)
        else :
            pars = [ho]

    pars = aux.existing_cols(aux.unique_list(pars),s)

    for p in pars:
        pvel = aux.nam.vel(p)
        avel = aux.nam.acc(p)
        ss=s[p]
        if p.endswith('orientation'):

            p_unw=aux.nam.unwrap(p)
            s[p_unw] = aux.apply_per_level(s[p], aux.unwrap_deg).flatten()
            ss=s[p_unw]

        s[pvel] = aux.apply_per_level(ss, aux.rate, dt=c.dt).flatten()
        s[avel] = aux.apply_per_level(s[pvel], aux.rate, dt=c.dt).flatten()

        if p in ['bend', ho, to, fo, ro]:
            for pp in [p,pvel,avel] :
                temp=s[pp].dropna().groupby('AgentID')
                e[aux.nam.mean(pp)] = temp.mean()
                e[aux.nam.std(pp)] = temp.std()
                e[aux.nam.initial(pp)] = temp.first()
                # s[[aux.nam.min(pp), aux.nam.max(pp)]] = comp_extrema_solo(sss, dt=dt, **kwargs).reshape(-1, 2)

    reg.vprint('All angular parameters computed')





@reg.funcs.proc("angular")
def angular_processing(s, e, c, d=None, recompute=False, mode='minimal', **kwargs):
    assert isinstance(c, reg.DatasetConfig)

    fo, ro = aux.nam.orient(['front', 'rear'])

    if c.Nangles == 0:
        or_pars =[fo]
        bend_pars=[]

    else :
        or_pars = [fo, ro]
        bend_pars=['bend']

    if not aux.cols_exist(or_pars+bend_pars,s) or recompute:
    # if not set(or_pars+bend_pars).issubset(s.columns.values) or recompute:
        if c.Npoints == 1:
            def func(ss):
                x, y = ss[:, 0].values, ss[:, 1].values
                dx, dy = np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)
                aa = np.arctan2(dy, dx)
                aa[aa < 0] += 2 * np.pi
                return aa

            s[fo]=  aux.apply_per_level(s[['x', 'y']], func).flatten()
        else :
            Axy = s[c.midline_xy].values
            Ax, Ay = Axy[:, ::2], Axy[:, 1::2]
            Adx = np.diff(Ax)
            Ady = np.diff(Ay)
            Aa = np.arctan2(Ady, Adx) % (2 * np.pi)
            if c.Npoints == 2 :
                s[fo] = Aa[:, 0]
            else :
                f1,f2=c.front_vector
                r1,r2=c.rear_vector

                fx, fy = Ax[:, f1-1] - Ax[:, f2-1], Ay[:, f1-1] - Ay[:, f2-1]
                rx, ry = Ax[:, r1-1] - Ax[:, r2-1], Ay[:, r1-1] - Ay[:, r2-1]
                s[fo] =Afo = np.arctan2(fy, fx)% (2 * np.pi)
                s[ro] =Aro = np.arctan2(ry, rx)% (2 * np.pi)

                Ada = np.diff(Aa) % (2 * np.pi)
                Ada[Ada > np.pi] -= 2 * np.pi

                if c.bend == 'from_vectors':
                    reg.vprint(f'Computing bending angle as the difference between front and rear orients')
                    a = np.remainder(Afo-Aro, 2 * np.pi)
                    a[a > np.pi] -= 2 * np.pi
                elif c.bend == 'from_angles':
                    reg.vprint(f'Computing bending angle as the sum of the first {c.Nbend_angles} front angles')
                    a = np.sum(Ada[:, :c.Nbend_angles], axis=1)
                else :
                    raise

                s['bend'] = np.degrees(a)

                if mode=='full' :
                    s[c.angles] = Ada
                    bend_pars += c.angles

                    s[c.seg_orientations] = Aa
                    or_pars =aux.unique_list(or_pars + c.seg_orientations)

    else :
        reg.vprint(
            'Orientation and bend are already computed. If you want to recompute them, set recompute to True', 1)
    ps = or_pars + bend_pars
    comp_angular(s, e, c, pars=ps)


    reg.vprint(f'Completed {mode} angular processing.')
#
#
# def ang_from_xy(xy):
#     dx_dt = np.gradient(xy[:,0])
#     dy_dt = np.gradient(xy[:,1])
#     velocity = np.array([[dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
#     ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
#     tangent = np.array([1 / ds_dt] * 2).transpose() * velocity
#     tangent_x = tangent[:, 0]
#     tangent_y = tangent[:, 1]
#
#     deriv_tangent_x = np.gradient(tangent_x)
#     deriv_tangent_y = np.gradient(tangent_y)
#
#     dT_dt = np.array([[deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])
#
#     length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)
#
#     normal = np.array([1 / length_dT_dt] * 2).transpose() * dT_dt
#     d2s_dt2 = np.gradient(ds_dt)
#     d2x_dt2 = np.gradient(dx_dt)
#     d2y_dt2 = np.gradient(dy_dt)
#
#     curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
#     t_component = np.array([d2s_dt2] * 2).transpose()
#     n_component = np.array([curvature * ds_dt * ds_dt] * 2).transpose()
#
#     acceleration = t_component * tangent + n_component * normal
#
#     ang_vel = np.arctan(normal[:, 0] / normal[:, 1])
#
#     ang_acc = np.arctan(acceleration[:, 0] / acceleration[:, 1])
#     return ang_vel, ang_acc


# def comp_ang_from_xy(s, e, dt):
#     N = s.index.unique('Step').size
#     p = aux.nam.orient('front')
#     p_vel, p_acc = aux.nam.vel(p), aux.nam.acc(p)
#     ids = s.index.unique('AgentID').values
#     Nids = len(ids)
#     V = np.zeros([N, 1, Nids]) * np.nan
#     A = np.zeros([N, 1, Nids]) * np.nan
#     for j, id in enumerate(ids):
#         xy = s[["x", "y"]].xs(id, level='AgentID').values
#         avel, aacc = ang_from_xy(xy)
#         V[:, 0, j] = avel / dt
#         A[:, 0, j] = aacc / dt
#     s[p_vel] = V[:, 0, :].flatten()
#     s[p_acc] = A[:, 0, :].flatten()
#     e[aux.nam.mean(p_vel)] = s[p_vel].dropna().groupby('AgentID').mean()
#     e[aux.nam.mean(p_acc)] = s[p_acc].dropna().groupby('AgentID').mean()

@reg.funcs.proc("extrema")
def comp_extrema_multi(s, pars=None, **kwargs):

    if pars is None:
        ps1 = ['bend', aux.nam.orient('front')]
        pars = ps1 + aux.nam.vel(ps1)


    for p in aux.existing_cols(pars,s):
        s[[aux.nam.min(p), aux.nam.max(p)]]=comp_extrema_solo(s[p],**kwargs).reshape(-1,2)



@reg.funcs.proc("extrema_solo")
def comp_extrema_solo(ss,dt=0.1, interval_in_sec=0.3,  threshold_in_std=None, abs_threshold=None):
    kws = {
        'order': np.round(interval_in_sec / dt).astype(int),
        'threshold': abs_threshold,
    }
    if threshold_in_std is not None:
        std = threshold_in_std * ss.std()
        mu = ss.mean()
        kws['threshold'] = (mu - std, mu + std)
    return aux.apply_per_level(ss, aux.comp_extrema, **kws).reshape(-1, 2)
