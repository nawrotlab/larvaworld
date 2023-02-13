import math
import numpy as np
from shapely import geometry, ops

from scipy.signal import sosfiltfilt, butter

import larvaworld.lib.aux.xy
from larvaworld.lib import aux
from larvaworld.lib.aux import naming as nam


def LvsRtoggle(side):
    if side == 'Left':
        return 'Right'
    elif side == 'Right':
        return 'Left'
    else:
        raise ValueError(f'Argument {side} is neither Left nor Right')


def inside_polygon(points, tank_polygon):
    return all([tank_polygon.contains(geometry.Point(x, y)) for x, y in points])


def segment_body(N, xy0, seg_ratio=None, centered=True, closed=False):

    # If segment ratio is not provided, generate equal-length segments
    if seg_ratio is None:
        seg_ratio = [1 / N] * N

    # Create a polygon from the given body contour
    p = geometry.Polygon(xy0)
    # Get maximum y value of contour
    y0 = np.max(p.exterior.coords.xy[1])

    # Segment body via vertical lines
    ps = [p]
    for cum_r in np.cumsum(seg_ratio):
        l = geometry.LineString([(1 - cum_r, y0), (1 - cum_r, -y0)])
        new_ps = []
        for p in ps:
            new_ps += list(ops.split(p, l).geoms)
        ps = new_ps

    # Sort segments so that front segments come first
    ps.sort(key=lambda x: x.exterior.xy[0], reverse=True)

    # Transform to 2D array of coords
    ps = [p.exterior.coords.xy for p in ps]
    ps = [np.array([[x, y] for x, y in zip(xs, ys)]) for xs, ys in ps]

    # Center segments around 0,0
    if centered:
        for i, (r, cum_r) in enumerate(zip(seg_ratio, np.cumsum(seg_ratio))):
            ps[i] -= [(1 - cum_r) + r / 2, 0]
            # pass

    # Put front point at the start of segment vertices. Drop duplicate rows
    for i in range(len(ps)):
        if i == 0:
            ind = np.argmax(ps[i][:, 0])
            ps[i] = np.flip(np.roll(ps[i], -ind - 1, axis=0), axis=0)
        else:
            ps[i] = np.flip(np.roll(ps[i], 1, axis=0), axis=0)
        _, idx = np.unique(ps[i], axis=0, return_index=True)
        ps[i] = ps[i][np.sort(idx)]
        if closed:
            ps[i] = np.concatenate([ps[i], [ps[i][0]]])
    return ps


def generate_seg_shapes(Nsegs, points, seg_ratio=None, centered=True, closed=False, **kwargs):
    if seg_ratio is None:
        seg_ratio = np.array([1 / Nsegs] * Nsegs)
    ps = segment_body(Nsegs, np.array(points), seg_ratio=seg_ratio, centered=centered, closed=closed)
    seg_vertices = [np.array([p]) for p in ps]
    return seg_vertices


def rearrange_contour(ps0):
    ps_plus = [p for p in ps0 if p[1] >= 0]
    ps_plus.sort(key=lambda x: x[0], reverse=True)
    ps_minus = [p for p in ps0 if p[1] < 0]
    ps_minus.sort(key=lambda x: x[0], reverse=False)
    return ps_plus + ps_minus

def get_tank_polygon(c, k=0.97, return_polygon=True):
    X, Y = c.env_params.arena.dims
    shape = c.env_params.arena.shape
    if shape == 'circular':
        # This is a circle_to_polygon shape from the function
        tank_shape = larvaworld.aux.xy.circle_to_polygon(60, X / 2)
    elif shape == 'rectangular':
        # This is a rectangular shape
        tank_shape = np.array([(-X / 2, -Y / 2),
                               (-X / 2, Y / 2),
                               (X / 2, Y / 2),
                               (X / 2, -Y / 2)])
    if return_polygon:
        return geometry.Polygon(tank_shape * k)
    else:
        # tank_shape=np.insert(tank_shape,-1,tank_shape[0,:])
        return tank_shape


def parse_array_at_nans(a):
    a = np.insert(a, 0, np.nan)
    a = np.insert(a, -1, np.nan)
    dif = np.diff(np.isnan(a).astype(int))
    de = np.where(dif == 1)[0]
    ds = np.where(dif == -1)[0]
    return ds, de


def apply_sos_filter_to_array_with_nans(sos, x, padlen=6):

    try:
        array_filt = np.full_like(x, np.nan)
        ds, de = parse_array_at_nans(x)
        for s, e in zip(ds, de):
            k = x[s:e]
            if len(k) > padlen:
                array_filt[s:e] = sosfiltfilt(sos, x[s:e], padlen=padlen)
        return array_filt
    except:
        return sosfiltfilt(sos, x, padlen=padlen)


def apply_filter_to_array_with_nans_multidim(a, freq, fr, N=1):
    """
    Power-spectrum of signal.

    Compute the power spectrum of a signal and its dominant frequency within some range.

    Parameters
    ----------
    a : array
        1D,2D or 3D Array : the array of timeseries to be filtered
    freq : float
        The cut-off frequency to set for the butter filter
    fr : float
        The framerate of the dataset
    N: int
        order of the butter filter

    Returns
    -------
    yf : array
        Filtered array of same shape as a

    """


    # 2-dimensional array must have each timeseries in different column
    if a.ndim == 1:
        sos = butter(N=N, Wn=freq, btype='lowpass', analog=False, fs=fr, output='sos')
        return apply_sos_filter_to_array_with_nans(sos=sos, x=a)
    elif a.ndim == 2:
        sos = butter(N=N, Wn=freq, btype='lowpass', analog=False, fs=fr, output='sos')
        return np.array([apply_sos_filter_to_array_with_nans(sos=sos, x=a[:, i]) for i in
                         range(a.shape[1])]).T
    elif a.ndim == 3:
        return np.transpose([apply_filter_to_array_with_nans_multidim(a[:, :, i], freq, fr, N=1) for i in
                             range(a.shape[2])], (1, 2, 0))
    else:
        raise ValueError('Method implement for up to 3-dimensional array')


def fft_max(a, dt, fr_range=(0.0, +np.inf), return_amps=False):
    """
    Power-spectrum of signal.

    Compute the power spectrum of a signal and its dominant frequency within some range.

    Parameters
    ----------
    a : array
        1D np.array : velocity timeseries
    dt : float
        Timestep of the timeseries
    fr_range : Tuple[float,float]
        Frequency range allowed. Default is (0.0, +np.inf)
    return_amps: bool
        whether to return the whole array of frequency powers

    Returns
    -------
    yf : array
        Array of computed frequency powers.
    fr : float
        Dominant frequency within range.

    """
    from numpy.fft import fftfreq
    from scipy.fft import fft
    a = np.nan_to_num(a)
    Nticks = len(a)
    xf = fftfreq(Nticks, dt)[:Nticks // 2]
    yf = fft(a, norm="ortho")
    yf = 2.0 / Nticks * np.abs(yf[:Nticks // 2])
    yf = 1000 * yf / np.sum(yf)
    # yf = moving_average(yf, n=21)
    xf_trunc = xf[(xf >= fr_range[0]) & (xf <= fr_range[1])]
    yf_trunc = yf[(xf >= fr_range[0]) & (xf <= fr_range[1])]
    fr = xf_trunc[np.argmax(yf_trunc)]
    if return_amps:
        return fr, yf
    else:
        return fr


def fft_freqs(s, e, c):
    v, fov = nam.vel(['', nam.orient('front')])
    fv, fsv, ffov = nam.freq([v, nam.scal(v), fov])

    try:
        e[fv] = s[v].groupby("AgentID").apply(fft_max, dt=c.dt, fr_range=(1.0, 2.5))
        e[fsv] = e[fv]
    except:
        pass
    try:
        e[ffov] = s[fov].groupby("AgentID").apply(fft_max, dt=c.dt, fr_range=(0.1, 0.8))
        e['turner_input_constant'] = (e[ffov] / 0.024) + 5
    except:
        pass

def get_freq(d, par, fr_range=(0.0, +np.inf)):
    s, e, c = d.step_data, d.endpoint_data, d.config
    e[nam.freq(par)] = s[par].groupby("AgentID").apply(fft_max, dt=c.dt, fr_range=fr_range)



def get_all_odors(larva_groups, food_params):
    lg = [conf.odor.odor_id for conf in larva_groups.values()]
    su = [conf.odor.odor_id for conf in food_params.source_units.values()]
    sg = [conf.odor.odor_id for conf in food_params.source_groups.values()]
    ids = aux.unique_list([id for id in lg + su + sg if id is not None])
    return ids

def get_source_xy(food_params):
    sources_u = {k: v['pos'] for k, v in food_params['source_units'].items()}
    sources_g = {k: v['distribution']['loc'] for k, v in food_params['source_groups'].items()}
    return {**sources_u, **sources_g}

def get_all_foodtypes(food_params):
    sg = {k: v.default_color for k, v in food_params.source_groups.items()}
    su = {conf.group: conf.default_color for conf in food_params.source_units.values()}
    gr = {
        food_params.food_grid.unique_id: food_params.food_grid.default_color} if food_params.food_grid is not None else {}
    ids = {**gr, **su, **sg}
    ks = aux.unique_list(list(ids.keys()))
    try:
        ids = {k: list(np.array(ids[k]) / 255) for k in ks}
    except:
        ids = {k: ids[k] for k in ks}
    return ids


def test_rotation(ho0, ang_vel, hr0, l0, dt, to_return='front'):
    ho1 = ho0 + ang_vel * dt
    kk = np.array([math.cos(ho1), math.sin(ho1)])
    if to_return == 'front':
        return hr0 + kk * l0
    elif to_return == 'mid':

        return hr0 + kk * l0 / 2

def position_head_in_tank(hr0, ho0, l0, fov0,fov1, ang_vel, lin_vel, dt, tank, sf=1, go_err =0, turn_err =0):
    hf0 = hr0 + np.array([math.cos(ho0), math.sin(ho0)]) * l0
    def get_hf0(ang_vel):
        d_or = ang_vel * dt
        return np.array(aux.rotate_point_around_point(origin=hr0, point=hf0, radians=-d_or))


    def fov(ang_vel, turn_err =0):
        dv=8*np.pi / 90
        idx=0
        while not inside_polygon([get_hf0(ang_vel)], tank):
            if idx == 0:
                dv *= np.sign(ang_vel)
            ang_vel -= dv
            if ang_vel < fov0:
                ang_vel = fov0
                dv = np.abs(dv)
            elif ang_vel > fov1:
                ang_vel = fov1
                dv -= np.abs(dv)
            idx += 1
            if np.isnan(ang_vel) or idx > 100:
                turn_err += 1
                ang_vel = 0
                break
        return ang_vel, turn_err

    ang_vel, turn_err=fov(ang_vel, turn_err=turn_err)

    ho1 = ho0 + ang_vel * dt
    k = np.array([math.cos(ho1), math.sin(ho1)])
    hf01 = get_hf0(ang_vel)
    coef = dt * sf * k

    def get_hf1(lin_vel):
        return hf01 + coef * lin_vel
    def lv(lin_vel,go_err=0):
        dv = 0.00011
        idx = 0
        while not inside_polygon([get_hf1(lin_vel)], tank):
            idx += 1
            lin_vel -= dv
            if np.isnan(lin_vel) or idx > 100:
                go_err += 1
                lin_vel =0
                break
            if lin_vel < 0:
                lin_vel = 0
                break
        return lin_vel, go_err

    lin_vel, go_err=lv(lin_vel, go_err=go_err)
    d = lin_vel * dt
    hp1 = hr0 + k * (d * sf + l0 / 2)
    return d, ang_vel, lin_vel, hp1, ho1, turn_err, go_err


class Collision(Exception):

    def __init__(self, object1, object2):
        self.object1 = object1
        self.object2 = object2


