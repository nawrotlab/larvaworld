import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pandas as pd
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy.signal import sosfiltfilt, find_peaks, argrelextrema
from scipy.spatial import ConvexHull
import statsmodels.api as sm

# from lib.aux.dictsNlists import AttrDict, save_dict

from lib.aux.sim_aux import fft_max, fft_freqs
from lib.conf.base.opt_par import getPar
from lib.aux import naming as nam, dictsNlists as dNl

from lib.process.store import store_aux_dataset


def compute_centroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))
    return centroid


def convex_hull(xs=None, ys=None, N=None, interp_nans=True):
    Nrows, Ncols = xs.shape
    xs = [xs[i][~np.isnan(xs[i])] for i in range(Nrows)]
    ys = [ys[i][~np.isnan(ys[i])] for i in range(Nrows)]
    ps = [np.vstack((xs[i], ys[i])).T for i in range(Nrows)]
    xxs = np.zeros((Nrows, N))
    xxs[:] = np.nan
    yys = np.zeros((Nrows, N))
    yys[:] = np.nan

    for i, p in enumerate(ps):
        if len(p) > 0:
            try:
                b = p[ConvexHull(p).vertices]
                s = np.min([b.shape[0], N])
                xxs[i, :s] = b[:s, 0]
                yys[i, :s] = b[:s, 1]
                if interp_nans:
                    xxs[i] = interpolate_nans(xxs[i])
                    yys[i] = interpolate_nans(yys[i])
            except:
                pass
    return xxs, yys


def sign_changes(df, col):
    a = df[col].values
    u = np.sign(df[col])
    m = np.flatnonzero(u.diff().abs().eq(2))

    g = np.stack([m - 1, m], axis=1)
    v = np.abs(a[g]).argmin(1)

    res = df.iloc[g[np.arange(g.shape[0]), v]]
    return res


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


@contextmanager
def suppress_stdout(show_output):
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if not show_output:
            sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        # else :
        #     pass


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nans(y):
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def downsample_2d_array(a, step):
    return a[::step, ::step]


def compute_velocity(xy, dt, return_dst=False):
    x = xy[:, 0]
    y = xy[:, 1]

    dx = np.diff(x)
    dy = np.diff(y)
    d = np.sqrt(dx ** 2 + dy ** 2)
    v = d / dt
    v = np.insert(v, 0, np.nan)
    d = np.insert(d, 0, np.nan)
    if return_dst:
        return v, d
    else:
        return v


def compute_component_velocity(xy, angles, dt, return_dst=False):
    x = xy[:, 0]
    y = xy[:, 1]
    dx = np.diff(x, prepend=[np.nan])
    dy = np.diff(y, prepend=[np.nan])
    d_temp = np.sqrt(dx ** 2 + dy ** 2)
    # This is the angle of the displacement vector relative to x-axis
    rads = np.arctan2(dy, dx)
    # And this is the angle of the displacement vector relative to the front-segment orientation vector
    angles2ref = rads - angles
    angles2ref %= 2 * np.pi
    d = d_temp * np.cos(angles2ref)
    v = d / dt
    if return_dst:
        return v, d
    else:
        return v


def comp_bearing(xs, ys, ors, loc=(0.0, 0.0), in_deg=True):
    x0, y0 = loc
    dxs = x0 - np.array(xs)
    dys = y0 - np.array(ys)
    rads = np.arctan2(dys, dxs)
    drads = (ors - np.rad2deg(rads)) % 360
    drads[drads > 180] -= 360
    return drads if in_deg else np.deg2rad(rads)


def compute_velocity_threshold(v, Nbins=500, max_v=None, kernel_width=0.02):
    if max_v is None:
        max_v = np.nanmax(v)
    bins = np.linspace(0, max_v, Nbins)
    hist, bin_edges = np.histogram(v, bins=bins, density=True)
    vals = bin_edges[0:-1] + 0.5 * np.diff(bin_edges)
    hist += 1 / len(v)
    hist /= np.sum(hist)
    plt.figure()
    plt.semilogy(vals, hist)
    ker = sp.signal.gaussian(len(vals), kernel_width * Nbins / max_v)
    ker /= np.sum(ker)

    density = np.exp(np.convolve(np.log(hist), ker, 'same'))
    plt.semilogy(vals, density)

    mi, ma = argrelextrema(density, np.less)[0], argrelextrema(density, np.greater)[0]
    try:
        minimum = vals[mi][0]
    except:
        minimum = np.nan
    return minimum


def moving_average(a, n=3):
    # ret = np.cumsum(a, dtype=float)
    # ret[n:] = ret[n:] - ret[:-n]
    return np.convolve(a, np.ones((n,)) / n, mode='same')
    # return ret[n - 1:] / n


def slow_freq(a, dt, tmax=60.0):
    """
    Dominant slow frequency of signal.

    Compute the dominant frequency of a timeseries after smoothing it by a moving average over several (long) intervals.

    Parameters
    ----------
    a : array
        1D np.array : velocity timeseries
    dt : float
        Timestep of the timeseries
    tmax : float
        Maximum time interval over which to apply moving average in secs. Default is 60.0

    Returns
    -------
    fr_median : float
       The median of the dominant frequencies over all time intervals

    """
    lags = int(2 * tmax / dt)
    frs = []
    ts = np.arange(1, 60, 0.5)
    for t in ts:
        aa = moving_average(a, n=int(t / dt))
        autocor = sm.tsa.acf(aa, nlags=lags, missing="conservative")
        try:
            fr = 1 / (find_peaks(autocor)[0][0] * dt)
        except:
            fr = np.nan
        frs.append(fr)
    fr_median = np.nanmedian(frs)
    return fr_median



def process_epochs(a, epochs, dt, return_idx=True):
    if epochs.shape[0] == 0:
        stops = []
        durs = np.array([])
        slices = []
        amps = np.array([])
        idx = [] #np.array([])
        maxs = np.array([])
        if return_idx:
            return stops, durs, slices, amps, idx, maxs
        else:
            return durs, amps, maxs

    else:
        if epochs.shape == (2,):
            epochs = np.array([epochs, ])
        durs = (np.diff(epochs).flatten() + 1) * dt
        slices = [np.arange(r0, r1 + 1, 1) for r0, r1 in epochs]
        #slices=[p[~np.isnan(a[p])] for p in slices]
        amps = np.array([np.trapz(a[p][~np.isnan(a[p])], dx=dt) for p in slices])


        maxs = np.array([np.max(a[p]) for p in slices])
        if return_idx :
            stops = epochs[:, 1]
            idx = np.concatenate(slices) if len(slices) > 1 else slices[0]
            return stops, durs, slices, amps, idx, maxs
        else :
            return durs, amps, maxs


def detect_pauses(a, dt, vel_thr=0.3, runs=None, min_dur=None):
    """
    Annotates crawl-pauses in timeseries.

    Extended description of function.

    Parameters
    ----------
    a : array
        1D np.array : forward velocity timeseries
    dt : float
        Timestep of the timeseries
    vel_thr : float
        Maximum velocity threshold
    runs : list
        A list of pairs of the start-end indices of the runs.
        If provided pauses that overlap with runs will be excluded.

    Returns
    -------
    pauses : list
        A list of pairs of the start-end indices of the pauses.

    """
    if min_dur is None:
        min_dur = 2 * dt
    pauses = []
    idx = np.where(a <= vel_thr)[0]
    if runs is not None:
        for r0, r1 in runs:
            idx = idx[(idx <= r0) | (idx >= r1)]
    p0s = idx[np.where(np.diff(idx, prepend=[-np.inf]) != 1)[0]]
    p1s = idx[np.where(np.diff(idx, append=[np.inf]) != 1)[0]]
    pauses = np.vstack([p0s, p1s]).T
    durs = (np.diff(pauses).flatten() + 1) * dt
    pauses = pauses[durs >= min_dur]
    return pauses


def detect_runs(a, dt, vel_thr=0.3, min_dur=0.5):
    """
    Annotates crawl-runs in timeseries.

    Extended description of function.

    Parameters
    ----------
    a : array
        1D np.array : forward velocity timeseries
    dt : float
        Timestep of the timeseries
    vel_thr : float
        Maximum velocity threshold

    Returns
    -------
    runs : list
        A list of pairs of the start-end indices of the runs.


    """
    if min_dur is None:
        min_dur = 2 * dt
    idx = np.where(a >= vel_thr)[0]
    r0s = idx[np.where(np.diff(idx, prepend=[-np.inf]) != 1)[0]]
    r1s = idx[np.where(np.diff(idx, append=[np.inf]) != 1)[0]]
    runs = np.vstack([r0s, r1s]).T
    durs = (np.diff(runs).flatten() + 1) * dt
    runs = runs[durs >= min_dur]
    return runs


def detect_strides(a, dt, vel_thr=0.3, stretch=(0.75, 2.0), fr=None, return_extrema=True, return_runs=True):
    """
    Annotates strides-runs and pauses in timeseries.

    Extended description of function.

    Parameters
    ----------
    a : array
        1D np.array : forward velocity timeseries
    dt : float
        Timestep of the timeseries
    vel_thr : float
        Maximum velocity threshold
    stretch : Tuple[float,float]
        The min-max stretch of a stride relative to the default derived from the dominnt frequency

    Returns
    -------
    fr : float
        The dominant crawling frequency.
    strides : list
        A list of pairs of the start-end indices of the strides.
    i_min : array
        Indices of the local minima.
    i_max : array
        Indices of the local maxima.
    strides : list
         A list of pairs of the start-end indices of the strides.
    runs : list
         A list of pairs of the start-end indices of the runs/stridechains.
    runs_counts : list
         Stride-counts of the runs/stridechains.

    """
    if fr is None:
        fr = fft_max(a, dt, fr_range=(1, 2.5))
    tmin = stretch[0] // (fr * dt)
    tmax = stretch[1] // (fr * dt)
    i_min = find_peaks(-a, height=-3 * vel_thr, distance=tmin)[0]
    i_max = find_peaks(a, height=vel_thr, distance=tmin)[0]
    strides = []
    for m in i_max:
        try:
            s0, s1 = [i_min[i_min < m][-1], i_min[i_min > m][0]]
            if ((s1 - s0) <= tmax) and ([s0, s1] not in strides):
                strides.append([s0, s1])
        except:
            pass
    strides = np.array(strides)
    if not return_runs:
        if return_extrema:
            return i_min, i_max, strides
        else:
            return strides

    runs, run_counts = [], []
    s00, s11 = None, None

    count = 0
    for ii, (s0, s1) in enumerate(strides.tolist()):
        if ii == 0:
            s00, s11 = s0, s1
            count = 1
            continue
        if s11 == s0:
            s11 = s1
            count += 1
        else:
            runs.append([s00, s11])
            run_counts.append(count)
            count = 1
            s00, s11 = s0, s1
        if ii == len(strides) - 1:
            runs.append([s00, s11])
            run_counts.append(count)
            break
    runs = np.array(runs)
    if return_extrema:
        return i_min, i_max, strides, runs, run_counts
    else:
        return strides, runs, run_counts


def detect_turns(a, dt, min_dur=None):
    """
    Annotates turns in timeseries.

    Extended description of function.

    Parameters
    ----------
    a : array
        1D np.array : angular velocity timeseries
    dt : float
        Timestep of the timeseries

    Returns
    -------
    Lturns : list
        A list of pairs of the start-end indices of the Left turns.
    Rturns : list
        A list of pairs of the start-end indices of the Right turns.


    """
    if type(a)!=pd.core.series.Series :
        a=pd.Series(a)
    if min_dur is None:
        min_dur = 2 * dt
    i_zeros = np.where(np.sign(a).diff().ne(0) == True)[0]
    Rturns, Lturns = [], []
    for s0, s1 in zip(i_zeros[:-1], i_zeros[1:]):
        if (s1 - s0) <= 2:
            continue
        elif np.isnan(np.sum(a[s0:s1])):
            continue
        else:
            if all(a[s0:s1] >= 0):
                Lturns.append([s0, s1])
            elif all(a[s0:s1] <= 0):
                Rturns.append([s0, s1])
    Lturns = np.array(Lturns)
    Rturns = np.array(Rturns)
    return Lturns, Rturns


def stride_interference(a_sv, a_fov, strides, Nbins=64, strict=True, absolute=True):
    x = np.linspace(0, 2 * np.pi, Nbins)

    if strict:
        strides = [(s0, s1) for s0, s1 in strides if all(np.sign(a_fov[s0:s1]) >= 0) or all(np.sign(a_fov[s0:s1]) <= 0)]

    ar_sv = np.zeros([len(strides), Nbins])
    ar_fov = np.zeros([len(strides), Nbins])
    for ii, (s0, s1) in enumerate(strides):
        ar_fov[ii, :] = np.interp(x, np.linspace(0, 2 * np.pi, s1 - s0), a_fov[s0:s1])
        ar_sv[ii, :] = np.interp(x, np.linspace(0, 2 * np.pi, s1 - s0), a_sv[s0:s1])
    if absolute:
        ar_fov = np.abs(ar_fov)
    ar_fov_mu = np.nanquantile(ar_fov, q=0.5, axis=0)
    ar_sv_mu = np.nanquantile(ar_sv, q=0.5, axis=0)

    return ar_sv_mu, ar_fov_mu, x


def stride_max_vel_phis(s, e, c, Nbins=64):
    import lib.aux.naming as nam
    points = nam.midline(c.Npoints, type='point')
    l, sv, pau_fov_mu = getPar(['l', 'sv', 'pau_fov_mu'])
    x = np.linspace(0, 2 * np.pi, Nbins)
    phis = np.zeros([c.Npoints, c.N]) * np.nan
    for j, id in enumerate(c.agent_ids):
        ss = s.xs(id, level='AgentID')
        strides = detect_strides(ss[sv], c.dt, return_runs=False, return_extrema=False)
        strides = strides.tolist()
        for i, p in enumerate(points):
            ar_v = np.zeros([strides, Nbins])
            v_p = nam.vel(p)
            a = ss[v_p] if v_p in ss.columns else compute_velocity(ss[nam.xy(p)].values, dt=c.dt)
            for ii, (s0, s1) in enumerate(strides):
                ar_v[ii, :] = np.interp(x, np.linspace(0, 2 * np.pi, s1 - s0), a[s0:s1])
            ar_v_mu = np.nanquantile(ar_v, q=0.5, axis=0)
            phis[i, j] = x[np.argmax(ar_v_mu)]
    for i, p in enumerate(points):
        e[nam.max(f'phi_{nam.vel(p)}')] = phis[i, :]


def weathervanesNheadcasts(run_idx, pause_idx, turn_slices, Tamps):
    wvane_idx = [ii for ii, t in enumerate(turn_slices) if all([tt in run_idx for tt in t])]
    cast_idx = [ii for ii, t in enumerate(turn_slices) if all([tt in pause_idx for tt in t])]
    wvane_amps = Tamps[wvane_idx]
    cast_amps = Tamps[cast_idx]
    wvane_min, wvane_max = np.nanquantile(wvane_amps, 0.25), np.nanquantile(wvane_amps, 0.75)
    cast_min, cast_max = np.nanquantile(cast_amps, 0.25), np.nanquantile(cast_amps, 0.75)
    return wvane_min, wvane_max, cast_min, cast_max


def comp_chunk_dicts(s,e,c,vel_thr=0.3,strides_enabled=True,store=False) :
    fft_freqs(s, e, c)
    turn_dict = turn_annotation(s, e, c, store=store)
    crawl_dict = crawl_annotation(s, e, c, strides_enabled=strides_enabled, vel_thr=vel_thr, store=store)
    chunk_dicts = dNl.AttrDict.from_nested_dicts({id: {**turn_dict[id], **crawl_dict[id]} for id in c.agent_ids})
    if store :
        path = c.dir_dict.chunk_dicts
        os.makedirs(path, exist_ok=True)
        dNl.save_dict(chunk_dicts, f'{path}/{c.id}.txt', use_pickle=True)
        print('Individual larva bouts saved')
    return chunk_dicts

def comp_pooled_epochs(d,chunk_dicts=None,store=False,**kwargs):
    s, e, c = d.step_data, d.endpoint_data, d.config
    from lib.anal.fitting import fit_bouts
    if chunk_dicts is None :
        chunk_dicts = comp_chunk_dicts(s,e,c, store=store, **kwargs)
    pooled_epochs = fit_bouts(c=c, chunk_dicts=chunk_dicts, s=s, e=e, id=c.id)
    return pooled_epochs


def annotation(s, e, cc, **kwargs):
    chunk_dicts = comp_chunk_dicts(s=s,e=e,c=cc, **kwargs)
    turn_mode_annotation(e, chunk_dicts)
    comp_patch(s, e, cc)
    return chunk_dicts

def comp_patch(s,e,c):
    from lib.process.bouts import comp_patch_metrics, comp_chunk_bearing
    try:
        comp_patch_metrics(s, e)
    except :
        pass
    for b in ['stride', 'pause', 'turn']:
        try:
            comp_chunk_bearing(s, c, chunk=b)
            if b == 'turn':
                comp_chunk_bearing(s,  c, chunk='Lturn')
                comp_chunk_bearing(s, c, chunk='Rturn')
        except:
            pass

def stride_interp(a, strides,Nbins=64) :
    x = np.linspace(0, 2 * np.pi, Nbins)
    aa = np.zeros([strides.shape[0], Nbins])
    for ii, (s0, s1) in enumerate(strides):
        aa[ii, :] = np.interp(x, np.linspace(0, 2 * np.pi, s1 - s0), a[s0:s1])
    return aa

def mean_stride_curve(a, strides,da,Nbins=64) :
    aa=stride_interp(a, strides,Nbins)
    aa_minus = aa[da < 0]
    aa_plus = aa[da > 0]
    aa_norm = np.vstack([aa_plus, -aa_minus])
    dic= dNl.AttrDict.from_nested_dicts({
        'abs': np.nanquantile(np.abs(aa), q=0.5, axis=0).tolist(),
        'plus': np.nanquantile(aa_plus, q=0.5, axis=0).tolist(),
        'minus': np.nanquantile(aa_minus, q=0.5, axis=0).tolist(),
        'norm': np.nanquantile(aa_norm, q=0.5, axis=0).tolist(),
    })

    return dic

def cycle_curve_dict(s,dt) :
    strides = detect_strides(s.sv, dt, return_extrema=False, return_runs=False)
    da = np.array([np.trapz(s.fov[s0:s1]) for ii, (s0, s1) in enumerate(strides)])
    dic = {sh: mean_stride_curve(s[sh], strides, da) for sh in ['sv', 'fov', 'rov', 'foa', 'b']}
    return dNl.AttrDict.from_nested_dicts(dic)



def compute_interference(s, e, c, Nbins=64, chunk_dicts=None, store=False):
    import lib.aux.naming as nam
    x = np.linspace(0, 2 * np.pi, Nbins)

    sss={id:s.xs(id, level="AgentID") for id in c.agent_ids}

    if chunk_dicts is None:
        stride_dic={}

        stride_dic_dfo={}
        for jj, id in enumerate(c.agent_ids):

            ss= sss[id]
            stride_dic[id] = detect_strides(ss[getPar('sv')].values, c.dt, return_runs=False, return_extrema=False)
            a_fov=ss[getPar('fov')].values
            stride_dic_dfo[id]=np.array([np.trapz(a_fov[s0:s1]) for ii, (s0, s1) in enumerate(stride_dic[id])])
    else :
        stride_dic ={id:chunk_dicts[id]['stride'] for id in c.agent_ids}
        stride_dic_dfo ={id:chunk_dicts[id]['stride_Dor'] for id in c.agent_ids}


    pooled_curves={}
    cycle_curves={}
    mean_curves_abs={}
    for sh in ['sv','fov','rov','foa', 'b'] :
        par=getPar(sh)
        curves_abs =np.zeros([c.N, Nbins]) * np.nan
        curves_plus =np.zeros([c.N, Nbins]) * np.nan
        curves_minus =np.zeros([c.N, Nbins]) * np.nan
        curves_norm =np.zeros([c.N, Nbins]) * np.nan
        for jj, id in enumerate(c.agent_ids):
            ss= sss[id]
            aa=stride_interp(ss[par].values, stride_dic[id], Nbins=64)
            aa_plus=aa[stride_dic_dfo[id]>0]
            aa_minus=aa[stride_dic_dfo[id]<0]
            aa_norm=np.vstack([aa_plus, -aa_minus])
            curves_abs[jj, :]=np.nanquantile(np.abs(aa), q=0.5, axis=0)
            curves_plus[jj, :]=np.nanquantile(aa_plus, q=0.5, axis=0)
            curves_minus[jj, :]=np.nanquantile(aa_minus, q=0.5, axis=0)
            curves_norm[jj, :]=np.nanquantile(aa_norm, q=0.5, axis=0)
        mean_curves_abs[sh]=curves_abs
        cycle_curves[sh]=dNl.AttrDict.from_nested_dicts({
            'abs': curves_abs,
            'plus': curves_plus,
            'minus': curves_minus,
            'norm': curves_norm,
        })
        pooled_curves[sh]=dNl.AttrDict.from_nested_dicts({
            'abs': np.nanquantile(curves_abs, q=0.5, axis=0).tolist(),
            'plus': np.nanquantile(curves_plus, q=0.5, axis=0).tolist(),
            'minus': np.nanquantile(curves_minus, q=0.5, axis=0).tolist(),
            'norm': np.nanquantile(curves_norm, q=0.5, axis=0).tolist(),
        })


    att0s, att1s = np.min(mean_curves_abs['fov'], axis=1), np.max(mean_curves_abs['fov'], axis=1)

    e[nam.max('phi_attenuation')] = x[np.argmax(mean_curves_abs['fov'], axis=1)]
    e[nam.max(f'phi_{getPar("sv")}')] = x[np.argmax(mean_curves_abs['sv'], axis=1)]
    e[getPar('str_sv_max')] = np.max(mean_curves_abs['sv'], axis=1)
    try :
        e[nam.min('attenuation')] = att0s / e[getPar('pau_fov_mu')]
        e[nam.max('attenuation')] = (att1s - att0s) / e[getPar('pau_fov_mu')]
    except :
        pass


    c.pooled_cycle_curves = pooled_curves
    if store :
        dNl.save_dict(cycle_curves, c.dir_dict.cycle_curves, use_pickle=True)
        print('Individual mean cycle curves saved')
    return cycle_curves


def turn_mode_annotation(e, chunk_dicts):
    wNh = {}
    wNh_ps = ['weathervane_q25_amp', 'weathervane_q75_amp', 'headcast_q25_amp', 'headcast_q75_amp']
    for jj, id in enumerate(e.index.values):
        dic = chunk_dicts[id]
        wNh[id] = dict(zip(wNh_ps, weathervanesNheadcasts(dic.run_idx, dic.pause_idx, dic.turn_slice, dic.turn_amp)))
    e[wNh_ps] = pd.DataFrame.from_dict(wNh).T


def turn_annotation(s, e, c, store=False):
    fov, foa = getPar(['fov', 'foa'])

    turn_ps = getPar(['tur_fou', 'tur_t', 'tur_fov_max'])
    turn_vs = np.zeros([c.Nticks, c.N, len(turn_ps)]) * np.nan
    turn_dict = {}

    for jj, id in enumerate(c.agent_ids):
        a_fov = s[fov].xs(id, level="AgentID")
        Lturns, Rturns = detect_turns(a_fov, c.dt)

        Lturns1, Ldurs, Lturn_slices, Lamps, Lturn_idx, Lmaxs = process_epochs(a_fov.values, Lturns, c.dt)
        Rturns1, Rdurs, Rturn_slices, Ramps, Rturn_idx, Rmaxs = process_epochs(a_fov.values, Rturns, c.dt)
        Tamps = np.concatenate([Lamps, Ramps])
        Tdurs = np.concatenate([Ldurs, Rdurs])
        Tmaxs = np.concatenate([Lmaxs, Rmaxs])
        Tslices = Lturn_slices + Rturn_slices
        if Lturns.shape[0] > 0:
            turn_vs[Lturns[:, 1], jj, 0] = Lamps
            turn_vs[Lturns[:, 1], jj, 1] = Ldurs
            turn_vs[Lturns[:, 1], jj, 2] = Lmaxs
        if Rturns.shape[0] > 0:
            turn_vs[Rturns[:, 1], jj, 0] = Ramps
            turn_vs[Rturns[:, 1], jj, 1] = Rdurs
            turn_vs[Rturns[:, 1], jj, 2] = Rmaxs
        turn_dict[id] = {'Lturn': Lturns, 'Rturn': Rturns, 'turn_slice': Tslices, 'turn_amp': Tamps,
                         'turn_dur': Tdurs, 'turn_vel_max': Tmaxs}
    s[turn_ps] = turn_vs.reshape([c.Nticks * c.N, len(turn_ps)])
    if store :
        turn_ps = getPar(['tur_fou', 'tur_t', 'tur_fov_max'])
        store_aux_dataset(s, pars=turn_ps, type='distro', file=c.aux_dir)
    return turn_dict


def crawl_annotation(s, e, c, strides_enabled=True, vel_thr=0.3, store=False):
    if vel_thr is None:
        vel_thr = c.vel_thr
    l, v, sv, dst, acc, fov, foa, b, bv, ba, fv = \
        getPar(['l', 'v', 'sv', 'd', 'a', 'fov', 'foa', 'b', 'bv', 'ba', 'fv'])

    str_ps = getPar(['str_d_mu', 'str_d_std', 'str_sv_mu', 'str_fov_mu', 'str_fov_std', 'str_N'])
    lin_ps = getPar(['run_v_mu', 'pau_v_mu', 'run_a_mu', 'pau_a_mu', 'run_fov_mu', 'run_fov_std', 'pau_fov_mu', 'pau_fov_std',
         'run_foa_mu', 'pau_foa_mu', 'pau_b_mu', 'pau_b_std', 'pau_bv_mu', 'pau_bv_std', 'pau_ba_mu', 'pau_ba_std',
         'cum_run_t', 'cum_pau_t', 'run_t_min', 'run_t_max', 'pau_t_min', 'pau_t_max'])

    dt = c.dt
    lin_vs = np.zeros([c.N, len(lin_ps)]) * np.nan
    str_vs = np.zeros([c.N, len(str_ps)]) * np.nan

    run_ps = getPar(['pau_t', 'run_t', 'run_d', 'str_c_l', 'str_t', 'str_d', 'str_sd'])
    run_vs = np.zeros([c.Nticks, c.N, len(run_ps)]) * np.nan

    crawl_dict = {}


    for jj, id in enumerate(c.agent_ids):
        strides, str_chain_ls,stride_Dor,stride_durs,stride_dsts,stride_sdsts,strides1 = [], [], [], [], [], [], []
        a_v = s[v].xs(id, level="AgentID").values
        a_fov = s[fov].xs(id, level="AgentID").values

        if c.Npoints > 1:
            a_sv = s[sv].xs(id, level="AgentID").values
            if strides_enabled:
                strides, runs, str_chain_ls = detect_strides(a_sv, dt, fr=e[fv].loc[id],vel_thr=vel_thr, return_extrema=False)
                strides1, stride_durs, stride_slices, stride_dsts, stride_idx, stride_maxs = process_epochs(a_v,strides, dt)
                stride_sdsts = stride_dsts / e[l].loc[id]
                stride_Dor = np.array([np.trapz(a_fov[s0:s1+1]) for s0, s1 in strides])
                str_fovs = np.abs(a_fov[stride_idx])
                str_vs[jj, :] = [np.nanmean(stride_dsts),
                                 np.nanstd(stride_dsts),
                                 np.nanmean(a_sv[stride_idx]),
                                 np.nanmean(str_fovs),
                                 np.nanstd(str_fovs),
                                 np.nansum(str_chain_ls),
                                 ]
            else:
                runs = detect_runs(a_sv, dt)
            pauses = detect_pauses(a_sv, dt,vel_thr=vel_thr, runs=runs)
        else:

            runs = detect_runs(a_v, dt, vel_thr=vel_thr)
            pauses = detect_pauses(a_v, dt, runs=runs, vel_thr=vel_thr)

        pauses1, pause_durs, pause_slices, pause_dsts, pause_idx, pause_maxs = process_epochs(a_v, pauses, dt)
        runs1, run_durs, run_slices, run_dsts, run_idx, run_maxs = process_epochs(a_v, runs, dt)

        run_vs[pauses1, jj, 0] = pause_durs
        run_vs[runs1, jj, 1] = run_durs
        run_vs[runs1, jj, 2] = run_dsts
        run_vs[runs1, jj, 3] = str_chain_ls
        run_vs[strides1, jj, 4] = stride_durs
        run_vs[strides1, jj, 5] = stride_dsts
        run_vs[strides1, jj, 6] = stride_sdsts

        if b in s.columns:
            pau_bs = s[b].xs(id, level="AgentID").abs().values[pause_idx]
            pau_bvs = s[bv].xs(id, level="AgentID").abs().values[pause_idx]
            pau_bas = s[ba].xs(id, level="AgentID").abs().values[pause_idx]
            pau_b_temp = [np.mean(pau_bs), np.std(pau_bs), np.mean(pau_bvs), np.std(pau_bvs), np.mean(pau_bas),
                          np.std(pau_bas)]
        else:
            pau_b_temp = [np.nan] * 6
        a_foa = s[foa].xs(id, level="AgentID").abs().values
        a_acc = s[acc].xs(id, level="AgentID").values
        pau_fovs = np.abs(a_fov[pause_idx])
        run_fovs = np.abs(a_fov[run_idx])
        pau_foas = a_foa[pause_idx]
        run_foas = a_foa[run_idx]
        lin_vs[jj, :] = [
            np.mean(a_v[run_idx]),
            np.mean(a_v[pause_idx]),
            np.mean(a_acc[run_idx]),
            np.mean(a_acc[pause_idx]),
            np.mean(run_fovs), np.std(run_fovs),
            np.mean(pau_fovs), np.std(pau_fovs),
            np.mean(run_foas),
            np.mean(pau_foas),
            *pau_b_temp,
            np.sum(run_durs),
            np.sum(pause_durs),
            np.nanmin(run_durs) if len(run_durs) > 0 else 1,
            np.nanmax(run_durs) if len(run_durs) > 0 else 100,
            np.nanmin(pause_durs) if len(pause_durs) > 0 else dt,
            np.nanmax(pause_durs) if len(pause_durs) > 0 else 100,
        ]
        crawl_dict[id] = {'stride': strides,'stride_Dor': stride_Dor, 'run': runs, 'pause': pauses,
                          'run_idx': run_idx, 'pause_idx': pause_idx,
                          'run_count': str_chain_ls, 'run_dur': run_durs, 'run_dst': run_dsts, 'pause_dur': pause_durs}
    s[run_ps] = run_vs.reshape([c.Nticks * c.N, len(run_ps)])
    e[lin_ps] = lin_vs

    str_d_mu, str_d_std, str_sd_mu, str_sd_std, run_tr, pau_tr, cum_run_t, cum_pau_t, cum_t = \
        getPar(
            ['str_d_mu', 'str_d_std', 'str_sd_mu', 'str_sd_std', 'run_tr', 'pau_tr', 'cum_run_t', 'cum_pau_t', 'cum_t'])

    e[run_tr] = e[cum_run_t] / e[cum_t]
    e[pau_tr] = e[cum_pau_t] / e[cum_t]


    if c.Npoints > 1 and strides_enabled:
        e[str_ps] = str_vs
        e[str_sd_mu] = e[str_d_mu] / e[l]
        e[str_sd_std] = e[str_d_std] / e[l]
    if store :
        run_ps = getPar(['pau_t', 'run_t', 'run_d', 'str_c_l','str_d','str_sd'])
        store_aux_dataset(s, pars=run_ps, type='distro', file=c.aux_dir)
    return crawl_dict


def comp_bend_correction(refID='None.150controls'):
    from lib.conf.stored.conf import loadRef
    from lib.conf.base.opt_par import ParDict
    import copy
    import numpy as np
    d = loadRef(refID)
    d.load(contour=False)
    s, e, c = d.step_data, d.endpoint_data, d.config

    dic = ParDict(mode='load').dict
    fov, bv, b, sv = [dic[k]['d'] for k in ['fov', 'bv', 'b', 'sv']]

    ss = copy.deepcopy(s[[fov, bv, b, sv]].dropna())
    ss = ss.loc[ss[sv] > 0]
    ss = ss.loc[ss[fov] * ss[bv] > 0]
    ss = ss.loc[ss[fov].abs() - ss[bv].abs() > 0]
    ss['db'] = ss[fov] * c.dt - ss[bv] * c.dt
    ss['b0'] = ss[b] + ss['db']
    ss = ss.loc[ss['b0'] * ss['db'] > 0]
    ss['db0'] = ss['db'] / ss['b0']
    ss = ss.loc[ss['db0'] < 1]

    ddb = ss['db0']
    ddd = (2 * ss[sv]) * c.dt
    m, k = np.polyfit(ddd, ddb, 1)
    m = np.round(m, 2)
    k = np.round(k, 2)
    return m


class FuncParHelper:

    def __init__(self) :

        self.func_df=self.inspect_funcs()

    def get_func(self, func):
        module=self.func_df['module'].loc[func]
        return getattr(module, func)

    def apply_func(self,func,s,**kwargs):
        f=self.get_func(func)
        kws={k:kwargs[k] for k in kwargs.keys() if k in self.func_df['args'].loc[func]}
        f(s=s,**kws)
        return s

    def assemble_func_df(self,arg='s'):
        from lib.process import angular, spatial, bouts, basic
        arg_dicts = {}
        for module in [angular, spatial, bouts, basic]:
            dic = self.get_arg_dict(module, arg)
            arg_dicts.update(dic)
        df = pd.DataFrame.from_dict(arg_dicts,orient='index')

        return df

    def get_arg_dict(self, module, arg):
        from inspect import getmembers, isfunction, signature


        # funcnames = []
        arg_dict={}
        funcs = getmembers(module, isfunction)
        for k, f in funcs:
            args = signature(f)
            args = list(args.parameters.keys())
            if arg in args:
                if k!='store_aux_dataset' :
                    # funcnames.append(k)
                    arg_dict[k]= {'args' : args, 'module':module}
        return arg_dict

    def inspect_funcs(self, arg='s'):
        df=self.assemble_func_df(arg)
        new_cols=['requires', 'depends', 'computes']
        for col in new_cols :
            df[col]=np.nan

        df[new_cols]=self.manual_fill(df[new_cols])
        return df

    def manual_fill(self,df):
        df.loc['comp_ang_from_xy'] = ['x', 'y'], ['ang_from_xy'], ['fov', 'foa']
        df.loc['angular_processing'] = [], ['comp_orientations', 'comp_bend', 'comp_ang_from_xy', 'comp_angular',
                                            'comp_extrema', 'compute_LR_bias', 'store_aux_dataset'], []
        df.loc['comp_angular'] = ['fo', 'ro', 'b'], ['unwrap_orientations'], ['fov', 'foa', 'rov', 'roa', 'bv', 'ba']
        df.loc['unwrap_orientations'] = ['fo', 'ro'], [], ['fou', 'rou']
        df.loc['comp_orientation_1point'] = ['x', 'y'], [], ['fov']
        df.loc['compute_LR_bias'] = ['b', 'bv', 'fov'], [], []
        df.loc['comp_orientations'] = ['xys'], ['comp_orientation_1point'], ['fo', 'ro']
        df.loc['comp_bend'] = ['fo', 'ro'], ['comp_angles'], ['b']
        df.loc['comp_angles'] = ['xys'], [], ['angles']
        return df

    def is_computed_by(self, short):
        return [k for k in self.func_df.index if short in self.func_df['computes'].loc[k]]

    def requires(self, func):
        return self.func_df['requires'].loc[func]

    def depends(self,func):
        return self.func_df['depends'].loc[func]

    def requires_all(self, func):
        import lib.aux.dictsNlists as dNl
        shorts=[]
        shorts.append(self.requires(func))
        for f in self.depends(func) :
            shorts.append(self.requires_all(func))
        shorts=dNl.unique_list(shorts)
        return shorts

    def get_options(self, short):
        options={}
        for func in self.is_computed_by(short):
            options[func]=self.requires(func)
        return options

    def how_to_compute(self, s, par=None, short=None, **kwargs):
        if par is None :
            par = getPar(short)
        elif short is None :
            short=getPar(d=par, to_return='k')
        if par in s.columns :
            return True
        else :
            options=self.get_options(short)
            available= []
            for i,(func, shorts) in enumerate(options.items()) :
                pars = getPar(shorts)
                if all([p in s.columns for p in pars]):

                    available.append(func)
            if len(available)==0 :
                return False
            else :
                return available

    def compute(self,s,**kwargs):
        res=self.how_to_compute(s=s,**kwargs)
        if res in [True, False]:
            return res
        else:
            self.apply_func(res[0],s=s, **kwargs)
            return self.compute(s=s,**kwargs)

def finalize_eval(s, l, traj, ks, dt):
    from lib.process.spatial import straightness_index
    s.v = np.array(s.v)
    s.sv = s.v / l
    s.b = np.rad2deg(s.b)
    s.fov = np.rad2deg(s.fov)
    s.rov = np.rad2deg(s.rov)
    s.bv = np.diff(s.b, prepend=[np.nan]) / dt
    # self.eval.rov = self.eval.fov - self.eval.bv
    if 'ba' in ks:
        s.ba = np.diff(s.bv, prepend=[np.nan]) / dt
    if 'a' in ks:
        s.a = np.diff(s.v, prepend=[np.nan]) / dt
    if 'foa' in ks:
        s.foa = np.diff(s.fov, prepend=[np.nan]) / dt
    if 'tor5' in ks:
        s.tor5 = straightness_index(traj, int(5 / dt / 2), match_shape=False)
    if 'tor2' in ks:
        s.tor2 = straightness_index(traj, int(2 / dt / 2), match_shape=False)
    if 'tor1' in ks:
        s.tor1 = straightness_index(traj, int(1 / dt / 2), match_shape=False)
    if 'tor10' in ks:
        s.tor10 = straightness_index(traj, int(10 / dt / 2), match_shape=False)
    if 'tor20' in ks:
        s.tor20 = straightness_index(traj, int(20 / dt / 2), match_shape=False)
    if 'tur_fou' in ks:
        a_fov = pd.Series(s.fov)
        Lturns, Rturns = detect_turns(a_fov, dt)

        Ldurs, Lamps, Lmaxs = process_epochs(a_fov.values, Lturns, dt, return_idx=False)
        Rdurs, Ramps, Rmaxs = process_epochs(a_fov.values, Rturns, dt, return_idx=False)
        s.tur_fou = np.concatenate([Lamps, Ramps])
        s.tur_t = np.concatenate([Ldurs, Rdurs])
        s.tur_fov_max = np.concatenate([Lmaxs, Rmaxs])
    if 'run_t' in ks:
        a_sv = pd.Series(s.sv)
        fv = fft_max(a_sv, dt, fr_range=(1.0, 2.5), return_amps=False)
        strides, runs, run_counts = detect_strides(a_sv, dt, fr=fv, return_extrema=False)
        pauses = detect_pauses(a_sv, dt, runs=runs)
        pause_durs, pause_dsts, pause_maxs = process_epochs(a_sv, pauses, dt, return_idx=False)
        run_durs, run_dsts, run_maxs = process_epochs(a_sv, runs, dt, return_idx=False)
        s.run_d = run_dsts
        s.run_t = run_durs
        s.pau_t = pause_durs
    for k, vs in s.items():
        s[k] = vs[~np.isnan(vs)]
    return s