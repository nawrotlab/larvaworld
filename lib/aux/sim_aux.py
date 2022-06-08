import math

import numpy as np

from shapely.geometry import Point, Polygon, LineString


from lib.aux import naming as nam


def LvsRtoggle(side):
    if side == 'Left':
        return 'Right'
    elif side == 'Right':
        return 'Left'
    else:
        raise ValueError(f'Argument {side} is neither Left nor Right')


def mutate_value(v, range, scale=0.01):
    r0, r1 = range
    return np.clip(np.random.normal(loc=v, scale=scale * np.abs(r1 - r0)), a_min=r0, a_max=r1).astype(float)


def circle_to_polygon(sides, radius, rotation=0, translation=None):
    one_segment = np.pi * 2 / sides

    points = [
        (math.sin(one_segment * i + rotation) * radius,
         math.cos(one_segment * i + rotation) * radius)
        for i in range(sides)]

    if translation:
        points = [[sum(pair) for pair in zip(point, translation)]
                  for point in points]

    return np.array(points)


def inside_polygon(points, tank_polygon):
    return all([tank_polygon.contains(Point(x, y)) for x, y in points])
    # # // p is your point, p.x is the x coord, p.y is the y coord
    # Xmin, Xmax, Ymin, Ymax = space_edges_for_screen
    # # x, y = point
    #
    # if all([(x < Xmin or x > Xmax or y < Ymin or y > Ymax) for x, y in points]):
    #     # // Definitely not within the polygon!
    #     return False
    # else:
    #     # point = Point(x, y)
    #     # print(polygon.contains(point))
    #     return all([Polygon(vertices).contains(Point(x, y)) for x, y in points])


def body(points, start=[1, 0], stop=[0, 0]):
    xy = np.zeros([len(points) * 2+2, 2]) * np.nan
    xy[0, :] = start
    xy[len(points) + 1, :] = stop
    for i in range(len(points)):
        x, y = points[i]
        xy[1 + i, :] = x, y
        xy[-1 - i, :] = x, -y
    return xy

# Create a bilaterally symmetrical 2D contour with the long axis along x axis

# Arguments :
#   points : the points above x axis through which the contour passes
#   start : the front end of the body
#   stop : the rear end of the body


# Segment body in N segments of given ratios via vertical lines
def segment_body(N, xy0, seg_ratio=None, centered=True, closed=False):
    from shapely.ops import split
    # If segment ratio is not provided, generate equal-length segments
    if seg_ratio is None:
        seg_ratio = [1 / N] * N

    # Create a polygon from the given body contour
    p = Polygon(xy0)
    # Get maximum y value of contour
    y0 = np.max(p.exterior.coords.xy[1])

    # Segment body via vertical lines
    ps = [p]
    for cum_r in np.cumsum(seg_ratio):
        l = LineString([(1 - cum_r, y0), (1 - cum_r, -y0)])
        new_ps = []
        for p in ps:
            new_p = [new_p for new_p in split(p, l)]
            new_ps += new_p
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
        if closed :
            ps[i]=np.concatenate([ps[i], [ps[i][0]]])
        # ps[i][:,0]+=0.5
    # ps[0]=np.vstack([ps[0], np.array(ps[0][0,:])])

    # print(ps)
    return ps

def generate_seg_shapes(Nsegs,  points,seg_ratio=None, centered=True, closed=False, **kwargs):
    if seg_ratio is None :
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

def compute_dst(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def freq(d, dt, range=[0.7, 1.8]) :
    from scipy.signal import spectrogram
    try:
        f, t, Sxx = spectrogram(d, fs=1 / dt)
        # keep only frequencies of interest
        f0, f1 = range
        valid = np.where((f >= f0) & (f <= f1))
        f = f[valid]
        Sxx = Sxx[valid, :][0]
        max_freq = f[np.argmax(np.nanmedian(Sxx, axis=1))]
    except:
        max_freq = np.nan
    return max_freq

def get_tank_polygon(c, k=0.97, return_polygon=True):
    X, Y = c.env_params.arena.arena_dims
    shape = c.env_params.arena.arena_shape
    if shape == 'circular':
        # This is a circle_to_polygon shape from the function
        tank_shape = circle_to_polygon(60, X / 2)
    elif shape == 'rectangular':
        # This is a rectangular shape
        tank_shape = np.array([(-X / 2, -Y / 2),
                               (-X / 2, Y / 2),
                               (X / 2, Y / 2),
                               (X / 2, -Y / 2)])
    if return_polygon :
        return Polygon(tank_shape * k)
    else :
        # tank_shape=np.insert(tank_shape,-1,tank_shape[0,:])
        return tank_shape


def parse_array_at_nans(a):
    a = np.insert(a, 0, np.nan)
    a = np.insert(a, -1, np.nan)
    dif = np.diff(np.isnan(a).astype(int))
    de = np.where(dif == 1)[0]
    ds = np.where(dif == -1)[0]
    return ds, de


def apply_sos_filter_to_array_with_nans(array, sos, padlen=6):
    from scipy.signal import sosfiltfilt
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
    from scipy.signal import butter

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


def fft_max(a, dt, fr_range=(0.0, +np.inf), return_amps=False):
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
    from numpy.fft import fftfreq
    from scipy.fft import fft
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
    if return_amps:
        return fr, yf
    else:
        return fr


def fft_freqs(s, e, c):
    v,fov=nam.vel(['', nam.orient('front')])
    fv,fsv, ffov = nam.freq([v,nam.scal(v), fov])

    try:
        e[fv] = s[v].groupby("AgentID").apply(fft_max, dt=c.dt, fr_range=(1.0, 2.5))
        e[fsv] = e[fv]
    except:
        pass
    e[ffov] = s[fov].groupby("AgentID").apply(fft_max, dt=c.dt, fr_range=(0.1, 0.8))
    e['turner_input_constant'] = (e[ffov] / 0.024) + 5


def get_freq(d, par, fr_range=(0.0, +np.inf)) :
    s, e, c = d.step_data, d.endpoint_data, d.config
    e[nam.freq(par)]=s[par].groupby("AgentID").apply(fft_max, dt=c.dt, fr_range=fr_range)
