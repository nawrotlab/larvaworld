import copy
import math
import random

import numpy as np

from shapely.geometry import Point, Polygon, LineString

from lib.aux import naming as nam, dictsNlists as dNl
from lib.registry.pars import preg


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


def body(points, start=None, stop=None):
    if start is None:
        start = [1, 0]
    if stop is None:
        stop = [0, 0]
    xy = np.zeros([len(points) * 2 + 2, 2]) * np.nan
    xy[0, :] = start
    xy[len(points) + 1, :] = stop
    for i in range(len(points)):
        x, y = points[i]
        xy[1 + i, :] = x, y
        xy[-1 - i, :] = x, -y
    return xy


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


# def freq(d, dt, range=[0.7, 1.8]) :
#     from scipy.signal import spectrogram
#     try:
#         f, t, Sxx = spectrogram(d, fs=1 / dt)
#         # keep only frequencies of interest
#         f0, f1 = range
#         valid = np.where((f >= f0) & (f <= f1))
#         f = f[valid]
#         Sxx = Sxx[valid, :][0]
#         max_freq = f[np.argmax(np.nanmedian(Sxx, axis=1))]
#     except:
#         max_freq = np.nan
#     return max_freq

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
    if return_polygon:
        return Polygon(tank_shape * k)
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
    from scipy.signal import sosfiltfilt
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
    from scipy.signal import butter

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
    v, fov = nam.vel(['', nam.orient('front')])
    fv, fsv, ffov = nam.freq([v, nam.scal(v), fov])

    try:
        e[fv] = s[v].groupby("AgentID").apply(fft_max, dt=c.dt, fr_range=(1.0, 2.5))
        e[fsv] = e[fv]
    except:
        pass
    e[ffov] = s[fov].groupby("AgentID").apply(fft_max, dt=c.dt, fr_range=(0.1, 0.8))
    e['turner_input_constant'] = (e[ffov] / 0.024) + 5


def get_freq(d, par, fr_range=(0.0, +np.inf)):
    s, e, c = d.step_data, d.endpoint_data, d.config
    e[nam.freq(par)] = s[par].groupby("AgentID").apply(fft_max, dt=c.dt, fr_range=fr_range)


def get_source_xy(food_params):
    sources_u = {k: v['pos'] for k, v in food_params['source_units'].items()}
    sources_g = {k: v['distribution']['loc'] for k, v in food_params['source_groups'].items()}
    return {**sources_u, **sources_g}


def generate_larvae(N, sample_dict, base_model):
    RefPars = dNl.load_dict(preg.path_dict["ParRef"], use_pickle=False)
    from unflatten import unflatten
    from lib.aux.dictsNlists import load_dict, flatten_dict
    if len(sample_dict) > 0:
        all_pars = []
        modF = flatten_dict(base_model)
        for i in range(N):
            lF = copy.deepcopy(modF)
            for p, vs in sample_dict.items():
                p = RefPars[p] if p in RefPars.keys() else p
                lF.update({p: vs[i]})
            dic = dNl.NestDict(unflatten(lF))
            all_pars.append(dic)
    else:
        all_pars = [base_model] * N
    return all_pars


def get_sample_bout_distros0(Im, bout_distros):
    dic = {
        'pause_dist': ['pause', 'pause_dur'],
        'stridechain_dist': ['stride', 'run_count'],
        'run_dist': ['run', 'run_dur'],
    }

    ds = [ii for ii in ['pause_dist', 'stridechain_dist', 'run_dist'] if
          (ii in Im.keys()) and (Im[ii] is not None) and ('fit' in Im[ii].keys()) and (Im[ii]['fit'])]
    for d in ds:
        for sample_d in dic[d]:
            if sample_d in bout_distros.keys() and bout_distros[sample_d] is not None:
                Im[d] = bout_distros[sample_d]
    return Im


def get_sample_bout_distros(model, sample):
    if sample in [None, {}]:
        return model
    m = dNl.copyDict(model)
    if m.brain.intermitter_params:
        m.brain.intermitter_params = get_sample_bout_distros0(Im=m.brain.intermitter_params,
                                                              bout_distros=sample.bout_distros)

    return m


def sample_group(dir=None, N=1, sample_ps=[], e=None):
    if e is None:
        from lib.stor.larva_dataset import LarvaDataset
        d = LarvaDataset(dir, load_data=False)
        e = d.read(key='end')
    ps = [p for p in sample_ps if p in e.columns]
    means = [e[p].mean() for p in ps]
    if len(ps) >= 2:
        base = e[ps].values.T
        cov = np.cov(base)
        vs = np.random.multivariate_normal(means, cov, N).T
    elif len(ps) == 1:
        std = np.std(e[ps].values)
        vs = np.atleast_2d(np.random.normal(means[0], std, N))
    else:
        return {}
    dic = {p: v for p, v in zip(ps, vs)}
    return dic


def get_sample_ks(m, sample_ks=None):
    if sample_ks is None:
        sample_ks = []
    modF = dNl.flatten_dict(m)
    sample_ks += [p for p in modF if modF[p] == 'sample']
    return sample_ks


def sampleRef(mID=None, m=None, refID=None, refDataset=None, sample_ks=None, Nids=1, parameter_dict={}):
    if m is None:
        m = preg.loadConf(id=mID, conftype="Model")
    ks = get_sample_ks(m, sample_ks=sample_ks)
    sample_dict = {}
    if len(ks) > 0:
        RefPars = dNl.load_dict(preg.path_dict["ParRef"], use_pickle=False)
        invRefPars = {v: k for k, v in RefPars.items()}
        sample_ps = [invRefPars[k] for k in ks if k in invRefPars.keys()]
        if len(sample_ps) > 0:
            if refDataset is None:
                if refID is not None:
                    refDataset = preg.loadRef(refID, load=True, step=False)
            if refDataset is not None:
                m = get_sample_bout_distros(m, refDataset.config)
                e = refDataset.endpoint_data if hasattr(refDataset, 'endpoint_data') else refDataset.read(key='end')
                sample_ps = [p for p in sample_ps if p in e.columns]
                if len(sample_ps) > 0:
                    sample_dict = sample_group(N=Nids, sample_ps=sample_ps, e=e)
                    refID = refDataset.refID
    sample_dict.update(parameter_dict)
    return generate_larvae(Nids, sample_dict, m), refID


def imitateRef(mID=None, m=None, refID=None, refDataset=None, Nids=1, parameter_dict={}):
    if refDataset is None:
        if refID is not None:
            refDataset = preg.loadRef(refID, load=True, step=False)
        else:
            raise
    else:
        refID = refDataset.refID
    if Nids is None:
        Nids = refDataset.config.N

    e = refDataset.endpoint_data if hasattr(refDataset, 'endpoint_data') else refDataset.read(key='end')
    ids = random.sample(e.index.values.tolist(), Nids)
    RefPars = dNl.load_dict(preg.path_dict["ParRef"], use_pickle=False)
    sample_ps = [p for p in list(RefPars.keys()) if p in e.columns]
    sample_dict = {p: [e[p].loc[id] for id in ids] for p in sample_ps}
    sample_dict.update(parameter_dict)

    if m is None:
        m = preg.loadConf(id=mID, conftype="Model")
    m = get_sample_bout_distros(m, refDataset.config)
    all_pars = generate_larvae(Nids, sample_dict, m)
    ps = [tuple(e[['initial_x', 'initial_y']].loc[id].values) for id in ids]
    try:
        ors = [e['initial_front_orientation'].loc[id] for id in ids]
    except:
        ors = np.random.uniform(low=0, high=2 * np.pi, size=len(ids)).tolist()
    return ids, ps, ors, all_pars




class Collision(Exception):

    def __init__(self, object1, object2):
        self.object1 = object1
        self.object2 = object2
