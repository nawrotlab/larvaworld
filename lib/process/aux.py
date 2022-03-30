import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pandas as pd
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from scipy.signal import sosfiltfilt, butter, find_peaks, argrelextrema
from scipy.spatial import ConvexHull
from scipy.fft import fft, fftfreq
import statsmodels.api as sm


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
    v=np.insert(v, 0, np.nan)
    d=np.insert(d, 0, np.nan)
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
    fr = xf_trunc[np.argmax(yf_trunc)]
    return fr


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
    l, fov = [dic[k]['d'] for k in ['l', 'fov']]

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
        df = s.xs(id, level="AgentID")
        # l_mu=np.nanmedian(df[l])
        xy = df[xy_pair].values
        v0 = compute_velocity(xy, dt=c.dt)
        if scaled:
            v0 /= e[l].loc[id]
        e[fr1l].loc[id] = slow_freq(v0, c.dt)
        e[fr0l].loc[id] = fft_max(v0, c.dt, fr_range=(0.5, +np.inf))

        fov0 = df[fov].values
        e[fr1a].loc[id] = slow_freq(fov0, c.dt)

        e[fr0a].loc[id] = fft_max(fov0, c.dt, fr_range=(0.15, +np.inf))


def process_epochs(a, epochs, dt) :
    if epochs.shape[0] == 0:
        stops = []
        durs = np.array([])
        slices = []
        amps = np.array([])
        idx = np.array([])
        maxs = np.array([])
    else:
        stops = epochs[:, 1]
        durs = np.diff(epochs).flatten() * dt
        slices = [np.arange(r0, r1 + 1, 1) for r0, r1 in epochs]
        amps = np.array([np.trapz(a[p], dx=dt) for p in slices])
        idx = np.concatenate(slices)
        maxs = np.array([np.max(a[p]) for p in slices])
    return stops, durs, slices, amps, idx, maxs

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
    durs = np.diff(pauses).flatten() * dt
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
    durs = np.diff(runs).flatten() * dt
    runs = runs[durs >= min_dur]
    return runs


def detect_strides(a, dt, vel_thr=0.3, stretch=(0.75, 2.0), fr=None):
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
    if fr is None :
        fr = fft_max(a, dt, fr_range=(1, 2.5))
    tmin = stretch[0] // (fr * dt)
    tmax = stretch[1] // (fr * dt)
    i_min = find_peaks(-a, height=-2 * vel_thr, distance=tmin)[0]
    i_max = find_peaks(a, height=vel_thr, distance=tmin)[0]
    strides = []
    for m in i_max:
        try:
            s0, s1 = [i_min[i_min < m][-1], i_min[i_min > m][0]]
            if ((s1 - s0) <= tmax) and ([s0, s1] not in strides):
                strides.append([s0, s1])
        except:
            pass

    runs, run_counts = [], []
    s00, s11 = None, None

    count = 0
    for ii, (s0, s1) in enumerate(strides):
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
    return i_min, i_max, strides, runs, run_counts


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
    if min_dur is None:
        min_dur = 2 * dt
    i_zeros = np.where(np.sign(a).diff().ne(0) == True)[0]
    Rturns, Lturns = [], []
    for s0, s1 in zip(i_zeros[:-1], i_zeros[1:]):
        if (s1 - s0) < 2:
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


def stride_interference(a_sv, a_fov, pau_fov_mu, strides, Nbins=64):
    x = np.linspace(0, 2 * np.pi, Nbins)

    ar_sv = np.zeros([len(strides), Nbins])
    ar_fov = np.zeros([len(strides), Nbins])
    for ii, (s0, s1) in enumerate(strides):
        ar_fov[ii, :] = np.interp(x, np.linspace(0, 2 * np.pi, s1 - s0), a_fov[s0:s1])
        ar_sv[ii, :] = np.interp(x, np.linspace(0, 2 * np.pi, s1 - s0), a_sv[s0:s1])
    ar_fov_mu = np.nanquantile(np.abs(ar_fov), q=0.5, axis=0)
    ar_sv_mu = np.nanquantile(ar_sv, q=0.5, axis=0)
    at_min, at_max, phi_at_max = np.min(ar_fov_mu) / pau_fov_mu, (np.max(ar_fov_mu) - np.min(ar_fov_mu)) / pau_fov_mu, \
                                 x[np.argmax(ar_fov_mu)]
    phi_sv_max = x[np.argmax(ar_sv_mu)]
    return [at_min, at_max, phi_at_max, phi_sv_max]


def weathervanesNheadcasts(run_idx, pause_idx, Lturn_slices, Rturn_slices, Lamps, Ramps):
    amps = np.concatenate([Lamps, Ramps])
    turn_slices = Lturn_slices + Rturn_slices
    wvane_idx = [ii for ii, t in enumerate(turn_slices) if all([tt in run_idx for tt in t])]
    cast_idx = [ii for ii, t in enumerate(turn_slices) if all([tt in pause_idx for tt in t])]
    wvane_amps = amps[wvane_idx]
    cast_amps = amps[cast_idx]
    wvane_min, wvane_max = np.nanquantile(wvane_amps, 0.25), np.nanquantile(wvane_amps, 0.75)
    cast_min, cast_max = np.nanquantile(cast_amps, 0.25), np.nanquantile(cast_amps, 0.75)
    return wvane_min, wvane_max, cast_min, cast_max


def annotation(s, e, c, point=None, vel_thr=None, strides_enabled=True, save_to=None, **kwargs):
    from lib.conf.base.par import getPar
    import lib.aux.naming as nam
    from lib.aux.dictsNlists import flatten_list, AttrDict, save_dict
    l, v, sv, dst, acc, fov, foa, b, bv, ba, fv,ffov = getPar(['l', 'v', 'sv', 'd', 'a', 'fov', 'foa', 'b', 'bv', 'ba', 'fv','ffov'], to_return=['d'])[0]
    e[fv] = s[v].groupby("AgentID").apply(fft_max, dt=c.dt, fr_range=(1.0, 2.5))
    e[ffov] = s[fov].groupby("AgentID").apply(fft_max, dt=c.dt, fr_range=(0.1, 0.8))
    e['turner_input_constant'] = (e[ffov] / 0.024) + 5

    # Parameter easy naming
    cum_t, cum_d, v_mu, sv_mu = getPar(['cum_t', 'cum_d', 'v_mu', 'sv_mu'], to_return=['d'])[0]
    sstr_d_mu, sstr_d_std, run_tr, pau_tr, cum_run_t, cum_pau_t = getPar(['sstr_d_mu', 'sstr_d_std', 'run_tr', 'pau_tr', 'cum_run_t', 'cum_pau_t'], to_return=['d'])[0]

    str_ps, = getPar(['str_d_mu', 'str_d_std', 'str_sv_mu', 'str_fov_mu', 'str_fov_std'], to_return=['d'])
    lin_ps, = getPar(
        ['run_v_mu', 'pau_v_mu', 'run_a_mu', 'pau_a_mu', 'run_fov_mu', 'run_fov_std', 'pau_fov_mu', 'pau_fov_std',
         'run_foa_mu', 'pau_foa_mu', 'pau_b_mu', 'pau_b_std', 'pau_bv_mu', 'pau_bv_std', 'pau_ba_mu', 'pau_ba_std',
         'cum_run_t', 'cum_pau_t', 'Ltur_tr', 'Rtur_tr', 'run_t_min'], to_return=['d'])

    att = 'attenuation'
    att_ps = [nam.min(att), nam.max(att), nam.max(f'phi_{att}'), nam.max(f'phi_{sv}')]

    ids = e.index.values
    Nids = len(ids)

    step_ps, = getPar(['tur_fou', 'tur_t', 'tur_fov_max', 'pau_t', 'run_t', 'run_d'], to_return=['d'])
    step_vs = np.zeros([c.Nticks, Nids, len(step_ps)]) * np.nan

    all_runs_durs, all_runs_counts, all_runs_dsts, all_pauses_durs, all_turns_durs, all_turns_angles, all_turns_fov_max = [], [], [], [], [], [], []
    vs_ps = np.zeros([Nids, len(lin_ps)]) * np.nan
    vs_str_ps = np.zeros([Nids, len(str_ps) + len(att_ps)]) * np.nan
    chunk_dicts = {}
    wNh = {}
    wNh_ps = ['weathervane_q25_amp', 'weathervane_q75_amp', 'headcast_q25_amp', 'headcast_q75_amp']
    for jj, id in enumerate(ids):
        chunk_dict = {}
        # Angular
        a_fov = s[fov].xs(id, level="AgentID")
        Lturns, Rturns = detect_turns(a_fov, c.dt)

        Lturns1, Ldurs, Lturn_slices, Lamps, Lturn_idx, Lmaxs = process_epochs(a_fov, Lturns, c.dt)
        Rturns1, Rdurs, Rturn_slices, Ramps, Rturn_idx, Rmaxs = process_epochs(a_fov, Rturns, c.dt)

        if Lturns.shape[0] > 0:
            step_vs[Lturns[:, 1], jj, 0] = Lamps
            step_vs[Lturns[:, 1], jj, 1] = Ldurs
            step_vs[Lturns[:, 1], jj, 2] = Lmaxs
        if Rturns.shape[0] > 0:
            step_vs[Rturns[:, 1], jj, 0] = Ramps
            step_vs[Rturns[:, 1], jj, 1] = Rdurs
            step_vs[Rturns[:, 1], jj, 2] = Rmaxs

        chunk_dict['Lturn'] = Lturns
        chunk_dict['Rturn'] = Rturns
        a_v = s[v].xs(id, level="AgentID")

        if c.Npoints > 1:
            a_sv = s[sv].xs(id, level="AgentID")
            if strides_enabled:
                i_min, i_max, strides, runs, run_counts = detect_strides(a_sv, c.dt, fr=e[fv].loc[id])
                strides1, stride_durs, stride_slices, stride_dsts, stride_idx, stride_maxs = process_epochs(a_v, strides, c.dt)
                chunk_dict['stride'] = strides
                str_fovs = a_fov.abs()[stride_idx]
                vs_str_ps[jj, :len(str_ps)] = [np.mean(stride_dsts),
                                               np.std(stride_dsts),
                                               np.mean(a_sv[stride_idx]),
                                               np.mean(str_fovs),
                                               np.std(str_fovs)]
                all_runs_counts.append(run_counts)
                pauses = detect_pauses(a_sv, c.dt, runs=runs)
            else:
                runs = detect_runs(a_sv, c.dt)
                pauses = detect_pauses(a_sv, c.dt, runs=runs)
        else:
            runs = detect_runs(a_v, c.dt, vel_thr=vel_thr)
            pauses = detect_pauses(a_v, c.dt, runs=runs, vel_thr=vel_thr)

        pauses1, pause_durs, pause_slices, pause_dsts, pause_idx, pause_maxs = process_epochs(a_v, pauses, c.dt)
        runs1, run_durs, run_slices, run_dsts, run_idx, run_maxs = process_epochs(a_v, runs, c.dt)

        wNh[id] = dict(
            zip(wNh_ps, weathervanesNheadcasts(run_idx, pause_idx, Lturn_slices, Rturn_slices, Lamps, Ramps)))
        chunk_dict['run'] = runs
        chunk_dict['pause'] = pauses

        step_vs[pauses1, jj, 3] = pause_durs
        step_vs[runs1, jj, 4] = run_durs
        step_vs[runs1, jj, 5] = run_dsts

        pau_bs = s[b].xs(id, level="AgentID").abs()[pause_idx]
        pau_bvs = s[bv].xs(id, level="AgentID").abs()[pause_idx]
        pau_bas = s[ba].xs(id, level="AgentID").abs()[pause_idx]
        a_foa = s[foa].xs(id, level="AgentID").abs()
        a_acc = s[acc].xs(id, level="AgentID")
        pau_fovs = a_fov.abs()[pause_idx]
        run_fovs = a_fov.abs()[run_idx]
        pau_foas = a_foa[pause_idx]
        run_foas = a_foa[run_idx]
        vs_ps[jj, :] = [
            np.mean(a_v[run_idx]),
            np.mean(a_v[pause_idx]),
            np.mean(a_acc[run_idx]),
            np.mean(a_acc[pause_idx]),
            np.mean(run_fovs), np.std(run_fovs),
            np.mean(pau_fovs), np.std(pau_fovs),
            np.mean(run_foas),
            np.mean(pau_foas),
            np.mean(pau_bs), np.std(pau_bs),
            np.mean(pau_bvs), np.std(pau_bvs),
            np.mean(pau_bas), np.std(pau_bas),
            np.sum(run_durs),
            np.sum(pause_durs),
            np.sum(Ldurs) / e[cum_t].loc[id],
            np.sum(Rdurs) / e[cum_t].loc[id],
            np.min(run_durs) if len(run_durs) > 0 else 1]
        if c.Npoints > 1 and strides_enabled:
            vs_str_ps[jj, len(str_ps):] = stride_interference(a_sv, a_fov.abs(), np.mean(pau_fovs), strides)

        all_runs_durs.append(run_durs)
        all_runs_dsts.append(run_dsts)
        all_pauses_durs.append(pause_durs)
        all_turns_durs.append(np.concatenate([Ldurs, Rdurs]))
        all_turns_angles.append(np.concatenate([Lamps, Ramps]))
        all_turns_fov_max.append(np.concatenate([Lmaxs, Rmaxs]))
        chunk_dicts[id] = chunk_dict
    e[wNh_ps] = pd.DataFrame.from_dict(wNh).T
    chunk_dicts = AttrDict(chunk_dicts)
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        save_dict(chunk_dicts, f'{save_to}/{c.id}.txt', use_pickle=True)
    s[step_ps] = step_vs.reshape([c.Nticks * Nids, len(step_ps)])
    e[lin_ps] = vs_ps
    e[run_tr] = e[cum_run_t] / e[cum_t]
    e[pau_tr] = e[cum_pau_t] / e[cum_t]

    if c.Npoints > 1 and strides_enabled:
        e[str_ps + att_ps] = vs_str_ps
        e[sstr_d_mu] = e[getPar(['str_d_mu'], to_return=['d'])[0][0]] / e[l]
        e[sstr_d_std] = e[getPar(['str_d_std'], to_return=['d'])[0][0]] / e[l]

    aux_dic = {
        'run_dur': np.array(flatten_list(all_runs_durs)),
        'run_dst': np.array(flatten_list(all_runs_dsts)),
        'pause_dur': np.array(flatten_list(all_pauses_durs)),
        'run_count': np.array(flatten_list(all_runs_counts)),
        'turn_dur': np.array(flatten_list(all_turns_durs)),
        'turn_amp': np.array(flatten_list(all_turns_angles)),
        'turn_vel_max': np.array(flatten_list(all_turns_fov_max))
    }
    return aux_dic


def fit_bouts(aux_dic, dataset_id, c, save_to=None):
    from lib.anal.fitting import fit_bout_distros
    from lib.aux.dictsNlists import AttrDict, load_dict, save_dict
    dic, best = {}, {}
    for k, v in aux_dic.items():
        discr = True if k == 'run_count' else False
        try:
            dic[k] = fit_bout_distros(v, dataset_id=dataset_id, bout=k, combine=False, discrete=discr)
            best[k] = dic[k]['best'][k]['best']
        except:
            dic[k] = None
            best[k] = None

    c.bout_distros = AttrDict(best)

    dic = AttrDict(dic)
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        save_dict(dic, f'{save_to}/{dataset_id}.txt', use_pickle=True)
    return dic
