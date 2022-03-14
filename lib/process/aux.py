import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy.signal import sosfiltfilt, butter, find_peaks, argrelextrema
from scipy.spatial import ConvexHull
from scipy.fft import fft, fftfreq
import statsmodels.api as sm

from lib.aux.dictsNlists import flatten_list


def parse_array_at_nans(a):
    a = np.insert(a, 0, np.nan)
    a = np.insert(a, -1, np.nan)
    dif = np.diff(np.isnan(a).astype(int))
    de = np.where(dif == 1)[0]
    ds = np.where(dif == -1)[0]
    return ds, de


def apply_sos_filter_to_array_with_nans(array, sos, padlen=6):
    try:
        array_filt = np.full_like(array, np.nan)
        ds, de = parse_array_at_nans(array)
        for s, e in zip(ds, de):
            k = array[s:e]
            if len(k) > padlen:
                k_filt = sosfiltfilt(sos, k, padlen=padlen)
                array_filt[s:e] = k_filt
        return array_filt
    except:
        array_filt = sosfiltfilt(sos, array, padlen=padlen)
        return array_filt


def apply_filter_to_array_with_nans_multidim(array, freq, fr, N=1):
    sos = butter(N=N, Wn=freq, btype='lowpass', analog=False, fs=fr, output='sos')
    # The array chunks must be longer than padlen=6
    padlen = 6
    # 2-dimensional array must have each timeseries in different column
    if array.ndim == 1:
        return apply_sos_filter_to_array_with_nans(array=array, sos=sos, padlen=padlen)
    elif array.ndim == 2:
        return np.array([apply_sos_filter_to_array_with_nans(array=array[:, i], sos=sos, padlen=padlen) for i in
                         range(array.shape[1])]).T
    elif array.ndim == 3:
        return np.transpose([apply_filter_to_array_with_nans_multidim(array[:, :, i], freq, fr, N=1) for i in
                             range(array.shape[2])], (1, 2, 0))
    else:
        raise ValueError('Method implement for up to 3-dimensional array')


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
    if return_dst:
        return v, d
    else:
        return v


def compute_component_velocity(xy, angles, dt, return_dst=False):
    x = xy[:, 0]
    y = xy[:, 1]
    dx = np.diff(x)
    dy = np.diff(y)
    d_temp = np.sqrt(dx ** 2 + dy ** 2)
    # This is the angle of the displacement vector relative to x-axis
    rads = np.arctan2(dy, dx)
    # And this is the angle of the displacement vector relative to the front-segment orientation vector
    angles2ref = rads - angles[:-1]
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


def fft_max(a, dt, fr_range=(0.0, +np.inf)):
    """
    Powerspectrum of signal.

    Compute the powerspectrum of a signal abd its dominant frequency within some range.

    Parameters
    ----------
    a : array
        1D np.array : velocity timeseries
    dt : float
        Timestep of the timeseries
    fr_range : Tuple[float,float]
        Frequency range allowed. Default is (0.0, +np.inf)

    Returns
    -------
    xf : array
        Array of computed frequencies.
    yf : array
        Array of computed frequency powers.
    xmax : float
        Dominant frequency within range.

    """

    a = np.nan_to_num(a)
    Nticks = len(a)
    xf = fftfreq(Nticks, dt)[:Nticks // 2]
    yf = fft(a, norm="backward")
    yf = 2.0 / Nticks * np.abs(yf[:Nticks // 2])
    yf = 1000 * yf / np.sum(yf)
    # yf = moving_average(yf, n=21)
    xf_trunc = xf[(xf >= fr_range[0]) & (xf <= fr_range[1])]
    yf_trunc = yf[(xf >= fr_range[0]) & (xf <= fr_range[1])]
    xmax = xf_trunc[np.argmax(yf_trunc)]
    return xf, yf, xmax


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


def slowNfast_freqs(s, e, c, point_idx=8, scaled=False):
    from lib.conf.base.par import ParDict
    dic = ParDict(mode='load').dict
    l,fov = [dic[k]['d'] for k in ['l','fov']]

    if point_idx is None:
        xy_pair = ["x", "y"]
    else:
        import lib.aux.naming as nam
        xy_pair = nam.xy(nam.midline(c.Npoints, type='point'))[point_idx]
    fr0l, fr1l = "lin_short_fr", "lin_long_fr"
    e[fr0l] = np.nan
    e[fr1l] = np.nan
    fr0a, fr1a = "ang_short_fr", "ang_long_fr"
    e[fr0a] = np.nan
    e[fr1a] = np.nan
    for id in c.agent_ids:
        # print(id)
        df = s.xs(id, level="AgentID")
        # l_mu=np.nanmedian(df[l])
        xy = df[xy_pair].values
        v0 = compute_velocity(xy, dt=c.dt)
        if scaled:
            v0 /= e[l].loc[id]
        e[fr1l].loc[id] = slow_freq(v0, c.dt)
        e[fr0l].loc[id] = fft_max(v0, c.dt, fr_range=(0.5, +np.inf))[2]

        fov0 = df[fov].values
        e[fr1a].loc[id] = slow_freq(fov0, c.dt)
        e[fr0a].loc[id] = fft_max(fov0, c.dt, fr_range=(0.15, +np.inf))[2]

def detect_pauses(a, dt, vel_thr=0.3, runs=None):
    from lib.aux.dictsNlists import flatten_list
    """
    Annotates crawl-pauses in timeseries.

    Extended description of function.

    Parameters
    ----------
    a : array
        1D np.array : velocity timeseries
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
    pauses_durs : list
        Durations of the pauses.

    """
    pauses = []
    idx = np.where(a <= vel_thr)[0]
    if runs is not None:
        for r0, r1 in runs:
            idx = idx[(idx <= r0) | (idx >= r1)]
    p0s = idx[np.where(np.diff(idx, prepend=[-np.inf]) != 1)[0]]
    p1s = idx[np.where(np.diff(idx, append=[np.inf]) != 1)[0]]
    pauses = [[p0, p1] for p0, p1 in zip(p0s, p1s) if p0 != p1]
    pauses_durs = [(p1 - p0) * dt for p0, p1 in pauses]
    pause_vel_mu = np.mean(flatten_list([a[p0:p1] for p0, p1 in pauses]))
    return pauses, pauses_durs, pause_vel_mu


def detect_strides(a, dt, vel_thr=0.3, stretch=(0.75, 2.0)):
    from lib.aux.dictsNlists import flatten_list
    """
    Annotates strides-runs and pauses in timeseries.

    Extended description of function.

    Parameters
    ----------
    a : array
        1D np.array : velocity timeseries
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
    runs : list
         A list of pairs of the start-end indices of the runs/stridechains.
    runs_durs : list
        Durations of the runs.
    runs_counts : list
         Stride-counts of the runs/stridechains.
    pauses : list
        A list of pairs of the start-end indices of the pauses.
    pauses_durs : list
        Durations of the pauses.

    """
    xf, yf, fr = fft_max(a, dt, fr_range=(0.5, 3.0))
    tmin = stretch[0] // (fr * dt)
    tmax = stretch[1] // (fr * dt)
    i_min = find_peaks(-a, height=-2 * vel_thr, distance=tmin)[0]
    i_max = find_peaks(a, height=vel_thr, distance=tmin)[0]
    strides = []
    for m in i_max:
        try:
            s0, s1 = [i_min[i_min < m][-1], i_min[i_min > m][0]]
            if (s1 - s0) <= tmax:
                strides.append([s0, s1])
        except:
            pass

    runs = []
    s00, s11 = None, None
    runs_durs, runs_counts = [], []

    def register(s00, s11, count):
        runs.append([s00, s11])
        runs_durs.append((s11 - s00) * dt)
        runs_counts.append(count)

    for s0, s1 in strides:
        if s00 is None:
            s00, s11 = s0, s1
            count = 1
        elif s11 == s0:
            s11 = s1
            count += 1
        else:
            register(s00, s11, count)
            count = 1
            s00, s11 = s0, s1
        register(s00, s11, count)

    stride_vel_mu = np.mean(flatten_list([a[s0:s1] for s0, s1 in strides]))

    pauses, pauses_durs, pause_vel_mu = detect_pauses(a, dt, vel_thr=vel_thr, runs=runs)

    return fr, strides, i_min, i_max, runs, runs_durs, runs_counts, pauses, pauses_durs, stride_vel_mu, pause_vel_mu


def lin_annotate(s, e, c, point=None, p='scaled_velocity', **kwargs):
    import lib.aux.naming as nam
    from lib.anal.fitting import fit_bout_distros
    e["stride_dst_mean"] = np.nan
    e["stride_dst_std"] = np.nan
    e["stride_scaled_vel_mu"] = np.nan
    e["pause_scaled_vel_mu"] = np.nan
    e["crawl_freq"] = np.nan

    all_runs_durs,all_runs_counts, all_pauses_durs =[], [], []
    for id in c.agent_ids:
        a = s[p].xs(id, level="AgentID")
        fr, strides, i_min, i_max, runs, runs_durs, runs_counts, pauses, pauses_durs, stride_vel_mu, pause_vel_mu = detect_strides(
            a, c.dt, **kwargs)
        if point is None:
            point = c.point
        xy = s[nam.xy(point)].xs(id, level="AgentID").values
        stride_dsts = [np.sqrt(np.sum((xy[s1] - xy[s0]) ** 2)) for s0, s1 in strides]

        e["stride_dst_mean"].loc[id] = np.mean(stride_dsts)
        e["stride_dst_std"].loc[id] = np.std(stride_dsts)
        e["crawl_freq"].loc[id] = fr
        e["stride_scaled_vel_mu"].loc[id] = stride_vel_mu
        e["pause_scaled_vel_mu"].loc[id] = pause_vel_mu

        all_runs_durs.append(runs_durs)
        all_runs_counts.append(runs_counts)
        all_pauses_durs.append(pauses_durs)

    e["scaled_stride_dst_mean"] = e["stride_dst_mean"] / e["length"]
    e["scaled_stride_dst_std"] = e["stride_dst_std"] / e["length"]

    all_runs_durs=np.array(flatten_list(all_runs_durs))
    all_runs_counts=np.array(flatten_list(all_runs_counts))
    all_pauses_durs=np.array(flatten_list(all_pauses_durs))

    c.bout_distros={}

    dic = fit_bout_distros(all_runs_durs, discrete=False, print_fits=False,
                           dataset_id=c.id, bout='run_dur', combine=False, store=False)
    c.bout_distros["run_durs"]=dic["best"]["run_dur"]["best"]
    dic = fit_bout_distros(all_runs_counts, discrete=True, print_fits=False,
                           dataset_id=c.id, bout='run_count', combine=False, store=False)
    c.bout_distros["run_counts"] = dic["best"]["run_count"]["best"]
    dic = fit_bout_distros(all_pauses_durs, discrete=False, print_fits=False,
                           dataset_id=c.id, bout='pause_dur', combine=False, store=False)
    c.bout_distros["pause_durs"] = dic["best"]["pause_dur"]["best"]

