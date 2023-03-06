import numpy as np
import pandas as pd
import scipy

from larvaworld.lib import reg, aux, decorators


def comp_angles(s, Npoints):
    if Npoints<=2 :
        return
    xy_pars = aux.nam.midline_xy(Npoints, flat=True)
    Axy=s[xy_pars].values
    Adx=np.diff(Axy[:,::2])
    Ady=np.diff(Axy[:,1::2])
    Aa=np.arctan2(Ady, Adx)
    Ada=np.diff(Aa)%(2*np.pi)
    Ada[Ada > np.pi] -= 2 * np.pi
    Ada = np.degrees(Ada)
    angles=[f'angle{i}' for i in range(Npoints - 2)]
    s[angles]=Ada
    reg.vprint('All angles computed')




def comp_orientations(s, e, c, mode='minimal'):
    N = s.values.shape[0]
    Np = c.Npoints
    if Np == 1:
        comp_orientation_1point(s, e)
        return

    temp=c.metric_definition.angular
    for key in ['front_vector', 'rear_vector']:
        if temp[key] is None:
            reg.vprint('Front and rear vectors are not defined. Can not compute orients')
            return
    else:
        f1, f2 = temp.front_vector
        r1, r2 = temp.rear_vector

    # xy = aux.nam.midline_xy(Np)
    xy_pars = aux.nam.midline_xy(Np, flat=True)
    Axy=s[xy_pars].values

    reg.vprint(f'Computing front/rear body-vector and head/tail orientation angles')
    vector_idx={
        'front' : (f2 - 1, f1 - 1),
        'rear' : (r2 - 1, r1 - 1),
        'head' : (1,0),
        'tail' : (-1,-2),
    }


    if mode == 'full':
        reg.vprint(f'Computing additional orients for {Np - 1} spinesegments')
        for i, vec in enumerate(aux.nam.midline(Np - 1, type='seg')):
             vector_idx[vec] = (i+1,i)

    for vec, (idx1, idx2) in vector_idx.items() :
        par = aux.nam.orient(vec)
        x, y = Axy[:,2*idx2]-Axy[:,2*idx1], Axy[:,2*idx2+1]-Axy[:,2*idx1+1]
        aa = np.arctan2(y, x)
        aa[aa < 0] += 2 * np.pi
        s[par] = aa
        e[aux.nam.initial(par)] = s[par].dropna().groupby('AgentID').first()


    reg.vprint('All orientations computed')
    return



def comp_orientation_1point(s, e):
    # fo = aux.nam.orient('front')
    # N = len(s.index.unique('Step'))

    def func(ss) :
        x,y=ss[:, 0].values, ss[:, 1].values
        dx,dy=np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)
        aa = np.arctan2(dy, dx)
        aa[aa < 0] += 2 * np.pi
        return aa

    return aux.apply_per_level(s[['x', 'y']], func).flatten()




@reg.funcs.proc("ang_moments")
def comp_angular(s,e, dt,Npoints, pars=None, **kwargs):
    vecs = ['head', 'tail', 'front', 'rear']
    ho,to,fo,ro=aux.nam.orient(vecs)
    if pars is None :
        if Npoints > 3 :
            base_pars = ['bend', ho, to, fo, ro]
            segs = aux.nam.midline(Npoints - 1, type='seg')
            ang_pars = [f'angle{i}' for i in range(Npoints - 2)]
            pars = base_pars + ang_pars+aux.nam.orient(segs)
        else :
            pars = [ho]

    pars = [p for p in aux.unique_list(pars) if p in s.columns]

    for p in pars:
        pvel = aux.nam.vel(p)
        avel = aux.nam.acc(p)
        ss=s[p]
        if p.endswith('orientation'):
            p_unw=aux.nam.unwrap(p)
            s[p_unw] = aux.apply_per_level(s[p], aux.unwrap_deg).flatten()
            ss=s[p_unw]

        s[pvel] = aux.apply_per_level(ss, aux.rate, dt=dt).flatten()
        s[avel] = aux.apply_per_level(s[pvel], aux.rate, dt=dt).flatten()

        if p in ['bend', ho, to, fo, ro]:
            for pp in [p,pvel,avel] :
                sss=s[pp]

                temp=sss.dropna().groupby('AgentID')
                e[aux.nam.mean(pp)] = temp.mean()
                e[aux.nam.std(pp)] = temp.std()
                e[aux.nam.initial(pp)] = temp.first()
                s[[aux.nam.min(pp), aux.nam.max(pp)]] = comp_extrema_solo(sss, dt=dt, **kwargs).reshape(-1, 2)

    reg.vprint('All angular parameters computed')





@reg.funcs.proc("angular")
def angular_processing(s, e, c, recompute=False, mode='minimal', store=False, **kwargs):
    Np=c.Npoints
    dt = c.dt

    def ang_conf(p=c.metric_definition.angular):
        return p.bend,p.front_body_ratio, np.array(p.front_vector) - 1, np.array(p.rear_vector) - 1

    ho, to, fo, ro = aux.nam.orient(['head', 'tail', 'front', 'rear'])

    if Np < 3:
        Nangles = 0
        or_pars =[fo]
        bend_pars=[]

    else :
        Nangles = Np - 2
        or_pars = [fo, ro]
        bend_pars=['bend']









    if not set(or_pars+bend_pars).issubset(s.columns.values) or recompute:




        if Np == 1:
            def func(ss):
                x, y = ss[:, 0].values, ss[:, 1].values
                dx, dy = np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan)
                aa = np.arctan2(dy, dx)
                aa[aa < 0] += 2 * np.pi
                return aa

            s[fo]=  aux.apply_per_level(s[['x', 'y']], func).flatten()
        else :
            xy_pars = aux.nam.midline_xy(Np, flat=True)
            Axy = s[xy_pars].values
            Ax, Ay = Axy[:, ::2], Axy[:, 1::2]
            Adx = np.diff(Ax)
            Ady = np.diff(Ay)
            Aa = np.arctan2(Ady, Adx) % (2 * np.pi)
            if Np == 2 :
                s[fo] = Aa[:, 0]
            else :

                # s[ho] = Aa[:, 0]
                # s[to] = Aa[:, -1]




                bend_mode,front_body_ratio, (f1,f2), (r1,r2)= ang_conf()
                fx, fy = Ax[:, f1] - Ax[:, f2], Ay[:, f1] - Ay[:, f2]
                rx, ry = Ax[:, r1] - Ax[:, r2], Ay[:, r1] - Ay[:, r2]
                s[fo] =Afo = np.arctan2(fy, fx)% (2 * np.pi)
                s[ro] =Aro = np.arctan2(ry, rx)% (2 * np.pi)

                Ada = np.diff(Aa) % (2 * np.pi)
                Ada[Ada > np.pi] -= 2 * np.pi

                if bend_mode == 'from_vectors':
                    reg.vprint(f'Computing bending angle as the difference between front and rear orients')
                    a = np.remainder(Afo-Aro, 2 * np.pi)
                    a[a > np.pi] -= 2 * np.pi
                elif bend_mode == 'from_angles':
                    Nbend_angles = int(np.round(front_body_ratio * Nangles))
                    reg.vprint(f'Computing bending angle as the sum of the first {Nbend_angles} front angles')
                    a = np.sum(Ada[:, :Nbend_angles], axis=1)
                else :
                    raise

                s['bend'] = np.degrees(a)

                if mode=='full' :
                    ang_pars = [f'angle{i}' for i in range(Nangles)]
                    s[ang_pars] = Ada
                    bend_pars += ang_pars

                    segs = aux.nam.midline(Np - 1, type='seg')
                    seg_pars = aux.nam.orient(segs)
                    s[seg_pars] = Aa
                    or_pars =aux.unique_list(or_pars + seg_pars)

    else :
        reg.vprint(
            'Orientation and bend are already computed. If you want to recompute them, set recompute to True', 1)
    ps = or_pars + bend_pars
    comp_angular(s, e, dt,Np, pars=ps)
    comp_extrema_multi(s, dt=dt)
    if store :
        pars = ps + aux.nam.vel(ps) + aux.nam.acc(ps)
        aux.store_distros(s, pars, parent_dir=c.dir)


    reg.vprint(f'Completed {mode} angular processing.')


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

@reg.funcs.proc("extrema")
def comp_extrema_multi(s, pars=None, **kwargs):

    if pars is None:
        fo = aux.nam.orient('front')
        ps1 = ['bend', fo]
        pars = ps1 + aux.nam.vel(ps1)


    for p in pars:
        if p in s.columns :
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
