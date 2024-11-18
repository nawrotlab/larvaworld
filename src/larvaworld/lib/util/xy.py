"""
Methods for managing spatial metrics (2D x-y arrays)
"""

import copy
import random

import numpy as np
import pandas as pd
from shapely import geometry, ops

# Avoid SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead
pd.options.mode.chained_assignment = None  # default='warn'
from typing import Optional

import scipy as sp
from scipy.signal import find_peaks

from ... import vprint
from . import AttrDict, cols_exist, flatten_list, nam, rotate_points_around_point

__all__ = [
    "fft_max",
    "detect_strides",
    "stride_interp",
    "mean_stride_curve",
    "comp_PI",
    "rolling_window",
    "straightness_index",
    "sense_food",
    "generate_seg_shapes",
    "Collision",
    "rearrange_contour",
    "comp_bearing",
    "comp_bearing_solo",
    "compute_dispersal_solo",
    "compute_dispersal_multi",
    "compute_component_velocity",
    "compute_velocity_threshold",
    "get_display_dims",
    "get_window_dims",
    "get_arena_bounds",
    "circle_to_polygon",
    "apply_per_level",
    "moving_average",
    "boolean_indexing",
    "concat_datasets",
    "body_contour",
    "unwrap_deg",
    "unwrap_rad",
    "rate",
    "eudist",
    "eudi5x",
    "eudiNxN",
    "compute_dst",
    "comp_extrema",
    "align_trajectories",
    "fixate_larva",
    "epoch_overlap",
    "epoch_slices",
]


def fft_max(a, dt, fr_range=(0.0, +np.inf), return_amps=False):
    """
    Power-spectrum of signal.

    Compute the power spectrum of a signal and its dominant frequency within some range.

    Parameters
    ----------
    a : array
        1D np.array : signal timeseries
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
    from scipy.fft import fft

    a = np.nan_to_num(a)
    N = len(a)
    if N == 0:
        return np.nan
    xf = np.fft.fftfreq(N, dt)[: N // 2]
    yf = fft(a, norm="ortho")
    yf = 2.0 / N * np.abs(yf[: N // 2])
    yf = 1000 * yf / np.sum(yf)

    fr_min, fr_max = fr_range
    xf_trunc = xf[(xf >= fr_min) & (xf <= fr_max)]
    yf_trunc = yf[(xf >= fr_min) & (xf <= fr_max)]
    fr = xf_trunc[np.argmax(yf_trunc)]
    if return_amps:
        return fr, yf
    else:
        return fr


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
    fr : float, optional
        The dominant crawling frequency.

    Returns
    -------
    strides : list
        A list of pairs of the start-end indices of the strides.


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
    return np.array(strides)


def stride_interp(a, strides, Nbins=64):
    x = np.linspace(0, 2 * np.pi, Nbins)
    aa = np.zeros([strides.shape[0], Nbins])
    for ii, (s0, s1) in enumerate(strides):
        aa[ii, :] = np.interp(x, np.linspace(0, 2 * np.pi, s1 - s0), a[s0:s1])
    return aa


def mean_stride_curve(a, strides, da, Nbins=64):
    aa = stride_interp(a, strides, Nbins)
    aa_minus = aa[da < 0]
    aa_plus = aa[da > 0]
    aa_norm = np.vstack([aa_plus, -aa_minus])
    dic = AttrDict(
        {
            "abs": np.nanquantile(np.abs(aa), q=0.5, axis=0).tolist(),
            "plus": np.nanquantile(aa_plus, q=0.5, axis=0).tolist(),
            "minus": np.nanquantile(aa_minus, q=0.5, axis=0).tolist(),
            "norm": np.nanquantile(aa_norm, q=0.5, axis=0).tolist(),
        }
    )

    return dic


def comp_PI(arena_xdim, xs, return_num=False):
    N = len(xs)
    r = 0.2 * arena_xdim
    xs = np.array(xs)
    N_l = len(xs[xs <= -r / 2])
    N_r = len(xs[xs >= +r / 2])
    # N_m = len(xs[(xs <= +r / 2) & (xs >= -r / 2)])
    pI = np.round((N_l - N_r) / N, 3)
    if return_num:
        return pI, N
    else:
        return pI


def rolling_window(a, w):
    # Get windows of size w from array a
    if a.ndim != 1:
        raise ValueError("Input array must be 1-dimensional")
    return np.vstack([np.roll(a, -i) for i in range(w)]).T[: -w + 1]


def straightness_index(ss, rolling_ticks):
    ps = ["x", "y", "dst"]
    assert cols_exist(ps, ss)
    sss = ss[ps].values
    temp = sss[rolling_ticks]
    Ds = np.nansum(temp[:, :, 2], axis=1)
    xys = temp[:, :, :2]

    k0, k1 = len(ss), rolling_ticks.shape[0]
    dk = int((k0 - k1) / 2)
    SI0 = np.zeros(k0) * np.nan
    for i in range(k1):
        D = Ds[i]
        if D != 0:
            xy = xys[i][~np.isnan(xys[i]).any(axis=1)]
            if xy.shape[0] >= 2:
                L = np.sqrt(np.nansum(np.array(xy[-1, :] - xy[0, :]) ** 2))
                SI0[dk + i] = 1 - L / D
    return SI0


def sense_food(pos, sources=None, grid=None, radius=None):
    if grid:
        cell = grid.get_grid_cell(pos)
        if grid.grid[cell] > 0:
            return cell
    elif sources and radius is not None:
        valid = sources.select(eudi5x(np.array(sources.pos), pos) <= radius)
        valid.select(valid.amount > 0)

        if len(valid) > 0:
            return random.choice(valid)
    return None


def generate_seg_shapes(
    Nsegs: int,
    points: np.ndarray,
    seg_ratio: Optional[np.ndarray] = None,
    centered: bool = True,
    closed: bool = False,
) -> np.ndarray:
    """
    Segments a body into equal-length or given-length segments via vertical lines.

    Args:
    - Nsegs: Number of segments to divide the body into.
    - points: Array with shape (M,2) representing the contour of the body to be segmented.
    - seg_ratio: List of N floats specifying the ratio of the length of each segment to the length of the body.
                Defaults to None, in which case equal-length segments will be generated.
    - centered: If True, centers the segments around the origin. Defaults to True.
    - closed: If True, the last point of each segment is connected to the first point. Defaults to False.

    Returns:
    - ps: Numpy array with shape (Nsegs,L,2), where L is the number of vertices of each segment.
          The first segment in the list is the front-most segment.

    """
    # If segment ratio is not provided, generate equal-length segments
    if seg_ratio is None:
        seg_ratio = np.array([1 / Nsegs] * Nsegs)

    # Create a polygon from the given body contour
    p = geometry.Polygon(points)
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
    return np.array(ps)


class Collision(Exception):
    def __init__(self, object1, object2):
        self.object1 = object1
        self.object2 = object2


def rearrange_contour(ps0):
    ps_plus = [p for p in ps0 if p[1] >= 0]
    ps_plus.sort(key=lambda x: x[0], reverse=True)
    ps_minus = [p for p in ps0 if p[1] < 0]
    ps_minus.sort(key=lambda x: x[0], reverse=False)
    return ps_plus + ps_minus


def comp_bearing(xs, ys, ors, loc=(0.0, 0.0), in_deg=True):
    """
    Compute the bearing (azimuth) of a set of oriented 2D point-vectors relative to a location point.

    Parameters
    ----------
    xs : array-like
        x-coordinates of the points.
    ys : array-like
        y-coordinates of the points.
    ors : float or array-like
        The orientations (in degrees) of the point-vectors.
    loc : tuple, optional
        The reference location's coordinates as a (x, y) tuple. Default is (0.0, 0.0).
    in_deg : bool, optional
        If True, returns bearings in degrees (default). If False, returns bearings in radians.

    Returns
    -------
    array-like
        An array of bearing angles in degrees or radians, depending on the 'in_deg' parameter.
        Positive angles indicate clockwise rotation from the positive x-axis.

    Examples
    --------
    xs = [1.0, 2.0, 3.0]
    ys = [1.0, 2.0, 0.0]
    ors = 90.0
    comp_bearing(xs, ys, ors)

    array([-135., -135.,  -90.])

    """
    x0, y0 = loc
    dxs = x0 - np.array(xs)
    dys = y0 - np.array(ys)
    rads = np.arctan2(dys, dxs)
    drads = (ors - np.rad2deg(rads)) % 360
    drads[drads > 180] -= 360
    return drads if in_deg else np.deg2rad(drads)


def comp_bearing_solo(x, y, o, loc=(0.0, 0.0)):
    x0, y0 = loc
    b = (o - np.arctan2(y0 - y, x0 - x)) % (2 * np.pi)
    if b > np.pi:
        b -= 2 * np.pi
    return b


def compute_dispersal_solo(
    xy, min_valid_proportion=0.2, max_start_proportion=0.1, min_end_proportion=0.9
):
    """
    Compute dispersal values for a given trajectory.

    This function calculates dispersal values based on a trajectory represented as a 2D array or DataFrame.
    It checks for the validity of the input trajectory and returns dispersal values accordingly.

    Parameters
    ----------
    xy : array-like or DataFrame
        The trajectory data, where each row represents a point in 2D space.
    min_valid_proportion : float, optional
        The minimum proportion of valid data points required in the trajectory.
        Defaults to 0.2, meaning at least 20% of non-missing data points are required.
    max_start_proportion : float, optional
        The maximum proportion of missing data allowed before the first valid point.
        Defaults to 0.1, meaning up to 10% of missing data is allowed at the start.
    min_end_proportion : float, optional
        The minimum proportion of data allowed before the last valid point.
        Defaults to 0.9, meaning up to 10% of missing data is allowed at the end.

    Returns
    -------
    array-like
        An array of dispersal values or NaNs based on the input trajectory's validity.

    Notes
    -----
    - The input trajectory should be a 2D array or a DataFrame with columns representing x and y coordinates.
    - The function checks for the proportion of valid data points and the presence of missing data at the start and end.
    - If the trajectory is valid, dispersal values are computed using a custom function (eudi5x).


    """
    if isinstance(xy, pd.DataFrame):
        xy = xy.values
    N = xy.shape[0]
    idx = np.where(~np.isnan(xy))[0]
    if (
        idx.shape[0] < N * min_valid_proportion
        or idx[0] > N * max_start_proportion
        or idx[-1] < N * min_end_proportion
    ):
        return np.zeros(N) * np.nan
    else:
        return eudi5x(xy, xy[idx[0]])


# def get_timeseries_slice(df, dt=0.1, time_range=None):
#     if time_range is None :
#         return df
#     else :
#         t0,t1=time_range
#         s0 = int(t0 / dt)
#         s1 = int(t1 / dt)
#         df_slice = df.loc[(slice(s0, s1), slice(None)), :]
#         return df_slice


def compute_dispersal_multi(xy0, t0, t1, dt, **kwargs):
    """
    Compute dispersal values for multiple agents over a time range.

    Parameters
    ----------
    xy0 : pd.DataFrame
        A DataFrame containing agent positions and timestamps.
    t0 : float
        The start time for dispersal computation in sec.
    t1 : float
        The end time for dispersal computation in sec.
    dt : float
        Timestep of the timeseries.
    **kwargs : keyword arguments
        Additional arguments to pass to compute_dispersal_solo.

    Returns
    -------
    np.ndarray
        An array of dispersal values for all agents at each time step.
    int
        The number of time steps.

    Example:
    --------
    xy0 = pd.DataFrame({'AgentID': [1, 1, 2, 2],
                       'Step': [0, 1, 0, 1],
                       'x': [0.0, 1.0, 2.0, 3.0],
                       'y': [0.0, 1.0, 2.0, 3.0]})

    AA, Nt = compute_dispersal_multi(xy0, t0=0, t1=1, dt=1)

    # AA will contain dispersal values, and Nt will be the number of time steps.

    """
    # xy=get_timeseries_slice(xy0, dt=dt, time_range=(t0,t1))

    s0 = int(t0 / dt)
    s1 = int(t1 / dt)
    xy = xy0.loc[(slice(s0, s1), slice(None)), ["x", "y"]]

    AA = apply_per_level(xy, compute_dispersal_solo, **kwargs)
    Nt = AA.shape[0]
    N = xy0.index.unique("AgentID").size
    Nticks = xy0.index.unique("Step").size

    AA0 = np.zeros([Nticks, N]) * np.nan
    AA0[s0 : s0 + Nt, :] = AA

    return AA0.flatten(), Nt


def compute_component_velocity(xy, angles, dt, return_dst=False):
    """
    Compute the component velocity along a given orientation angle.

    This function calculates the component velocity of a set of 2D points (xy) along
    the specified orientation angles. It can optionally return the displacement along
    the orientation vector as well.

    Parameters
    ----------
    xy : ndarray
        An array of shape (n, 2) representing the x and y coordinates of the points.
    angles : ndarray
        An array of shape (n,) containing the orientation angles in radians.
    dt : float
        The time interval for velocity calculation.
    return_dst : bool, optional
        If True, the function returns both velocities and displacements.
        If False (default), it returns only velocities.

    Returns
    -------
    ndarray
        An array of component velocities calculated along the specified angles.

    ndarray (optional)
        An array of displacements along the specified orientation angles.
        Returned only if `return_dst` is True.

    """
    dx = np.diff(xy[:, 0], prepend=np.nan)
    dy = np.diff(xy[:, 1], prepend=np.nan)
    d_temp = np.sqrt(dx**2 + dy**2)

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


def compute_velocity_threshold(v, Nbins=500, max_v=None, kernel_width=0.02):
    """
    Compute a velocity threshold using a density-based approach.

    Parameters
    ----------
    v : array-like
        The input velocity data.
    Nbins : int, optional
        Number of bins for the velocity histogram. Default is 500.
    max_v : float or None, optional
        Maximum velocity value. If None, it is computed from the data. Default is None.
    kernel_width : float, optional
        Width of the Gaussian kernel for density estimation. Default is 0.02.

    Returns
    -------
    float
        The computed velocity threshold.

    Notes
    -----
    This function calculates a velocity threshold by estimating the density of the velocity data.
    It uses a histogram with `Nbins` bins, applies a Gaussian kernel of width `kernel_width`,
    and identifies the minimum between local maxima and minima in the density curve.

    """
    import matplotlib.pyplot as plt

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

    density = np.exp(np.convolve(np.log(hist), ker, "same"))
    plt.semilogy(vals, density)

    mi, ma = (
        sp.signal.argrelextrema(density, np.less)[0],
        sp.signal.argrelextrema(density, np.greater)[0],
    )
    try:
        minimum = vals[mi][0]
    except:
        minimum = np.nan
    return minimum


def get_display_dims():
    import pygame

    pygame.init()
    W, H = pygame.display.Info().current_w, pygame.display.Info().current_h
    return int(W * 2 / 3 / 16) * 16, int(H * 2 / 3 / 16) * 16


def get_window_dims(arena_dims):
    X, Y = np.array(arena_dims)
    W0, H0 = get_display_dims()
    R0, R = W0 / H0, X / Y
    if R0 < R:
        return W0, int(W0 / R / 16) * 16
    else:
        return int(H0 * R / 16) * 16, H0


def get_arena_bounds(arena_dims, s=1):
    X, Y = np.array(arena_dims) * s
    return np.array([-X / 2, X / 2, -Y / 2, Y / 2])


def circle_to_polygon(N, r):
    one_segment = np.pi * 2 / N

    points = [
        (np.sin(one_segment * i) * r, np.cos(one_segment * i) * r) for i in range(N)
    ]

    return points


def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def concat_datasets(ddic, key="end", unit="sec"):
    dfs = []
    for l, d in ddic.items():
        if key == "end":
            try:
                df = d.endpoint_data
            except:
                df = d.read("end")
        elif key == "step":
            try:
                df = d.step_data
            except:
                df = d.read("step")
        else:
            raise
        df["DatasetID"] = l
        df["GroupID"] = d.group_id
        dfs.append(df)
    df0 = pd.concat(dfs)
    if key == "step":
        df0.reset_index(level="Step", drop=False, inplace=True)
        dts = np.unique([d.config.dt for l, d in ddic.items()])
        if len(dts) == 1:
            dt = dts[0]
            dic = {"sec": 1, "min": 60, "hour": 60 * 60, "day": 24 * 60 * 60}
            df0["Step"] *= dt / dic[unit]
    return df0


def moving_average(a, n=3):
    return np.convolve(a, np.ones((n,)) / n, mode="same")


def body_contour(points=[(0.9, 0.1), (0.05, 0.1)], start=(1, 0), stop=(0, 0)):
    xy = np.zeros([len(points) * 2 + 2, 2]) * np.nan
    xy[0, :] = start
    xy[len(points) + 1, :] = stop
    for i in range(len(points)):
        x, y = points[i]
        xy[1 + i, :] = x, y
        xy[-1 - i, :] = x, -y
    return xy


def apply_per_level(s, func, level="AgentID", **kwargs):
    """
    Apply a function to each subdataframe of a MultiIndex DataFrame after grouping by a specified level.

    Parameters
    ----------
    s : pandas.DataFrame
        A MultiIndex DataFrame with levels ['Step', 'AgentID'].
    func : function
        The function to apply to each subdataframe.
    level : str, optional
        The level by which to group the DataFrame. Default is 'AgentID'.
    **kwargs : dict
        Additional keyword arguments to pass to the 'func' function.

    Returns
    -------
    numpy.ndarray
        An array of dimensions [N_ticks, N_ids], where N_ticks is the number of unique 'Step' values,
        and N_ids is the number of unique 'AgentID' values.

    Notes
    -----
    This function groups the DataFrame 's' by the specified 'level', applies 'func' to each subdataframe, and
    returns the results as a numpy array.

    """

    def init_A(Ndims):
        ids = s.index.unique("AgentID").values
        Nids = len(ids)
        N = s.index.unique("Step").size
        if Ndims == 1:
            return np.zeros([N, Nids]) * np.nan
        elif Ndims == 2:
            return np.zeros([N, Nids, Ai.shape[1]]) * np.nan
        else:
            raise ValueError("Not implemented")

    A = None

    for i, (v, ss) in enumerate(s.groupby(level=level)):
        ss = ss.droplevel(level)
        Ai = func(ss, **kwargs)
        if A is None:
            A = init_A(len(Ai.shape))
        if level == "AgentID":
            A[:, i] = Ai
        elif level == "Step":
            A[i, :] = Ai
    return A


def unwrap_deg(a):
    if isinstance(a, pd.Series):
        a = a.values
    b = np.copy(a)
    b[~np.isnan(b)] = np.unwrap(b[~np.isnan(b)] * np.pi / 180) * 180 / np.pi
    return b


def unwrap_rad(a):
    if isinstance(a, pd.Series):
        a = a.values
    b = np.copy(a)
    b[~np.isnan(b)] = np.unwrap(b[~np.isnan(b)])
    return b


def rate(a, dt):
    if isinstance(a, pd.Series):
        a = a.values
    v = np.diff(a) / dt
    return np.insert(v, 0, np.nan)


def eudist(xy):
    if isinstance(xy, pd.DataFrame):
        xy = xy.values
    A = np.sqrt(np.nansum(np.diff(xy, axis=0) ** 2, axis=1))
    A = np.insert(A, 0, 0)
    return A


def eudi5x(a, b):
    """
    Calculate the Euclidean distance between points in arrays 'a' and 'b'.

    Parameters
    ----------
    a : numpy.ndarray
        An array containing the coordinates of the first set of points.
    b : numpy.ndarray
        An array containing the coordinates of the second set of points.

    Returns
    -------
    numpy.ndarray
        An array of Euclidean distances between each pair of points from 'a' and 'b'.

    """
    return np.sqrt(np.sum((a - np.array(b)) ** 2, axis=1))


def eudiNxN(a, b):
    b = np.array(b)
    return np.sqrt(np.sum(np.array([a - b[i] for i in range(b.shape[0])]) ** 2, axis=2))


def compute_dst(s, point=""):
    s[nam.dst(point)] = apply_per_level(s[nam.xy(point)], eudist).flatten()


def comp_extrema(a, order=3, threshold=None, return_2D=True):
    """
    Compute local extrema in a one-dimensional array or time series.

    Parameters
    ----------
    a : pd.Series
        The input time series data as a pandas Series.
    order : int, optional
        The order of the extrema detection. Default is 3.
    threshold : tuple, optional
        A tuple (min_threshold, max_threshold) to filter extrema based on values.
        Default is None, which means no thresholding is applied.
    return_2D : bool, optional
        If True, returns a 2D array with flags for minima and maxima.
        If False, returns a 1D array with -1 for minima, 1 for maxima, and NaN for non-extrema.
        Default is True.

    Returns
    -------
    np.ndarray
        An array with extrema flags based on the specified criteria.

    Notes
    -----
    - This function uses `scipy.signal.argrelextrema` for extrema detection.

    """
    A = a.values
    N = A.shape[0]
    i_min = sp.signal.argrelextrema(A, np.less_equal, order=order)[0]
    i_max = sp.signal.argrelextrema(A, np.greater_equal, order=order)[0]

    # i_min_dif = np.diff(i_min, append=order)
    # i_max_dif = np.diff(i_max, append=order)
    # i_min = i_min[i_min_dif >= order]
    # i_max = i_max[i_max_dif >= order]

    if threshold is not None:
        t0 = a.index.min()
        thr_min, thr_max = threshold
        i_min = i_min[a.loc[i_min + t0] < thr_min]
        i_max = i_max[a.loc[i_max + t0] > thr_max]

    if return_2D:
        aa = np.zeros([N, 2]) * np.nan
        aa[i_min, 0] = True
        aa[i_max, 1] = True
    else:
        aa = np.zeros(N) * np.nan
        aa[i_min] = -1
        aa[i_max] = 1
    return aa


def align_trajectories(
    s,
    c,
    d=None,
    track_point=None,
    arena_dims=None,
    transposition="origin",
    replace=True,
    **kwargs,
):
    if transposition in ["", None, np.nan]:
        return
    mode = transposition

    xy_flat = c.all_xy.existing(s)
    xy_pairs = xy_flat.in_pairs

    if replace:
        ss = s
    else:
        ss = copy.deepcopy(s[xy_flat])

    if mode == "arena":
        # reg.vprint('Centralizing trajectories in arena center')
        if arena_dims is None:
            arena_dims = c.env_params.arena.dims
        x0, y0 = arena_dims
        X, Y = x0 / 2, y0 / 2

        for x, y in xy_pairs:
            ss[x] -= X
            ss[y] -= Y
        return ss
    else:
        if track_point is None:
            track_point = c.point
        XY = nam.xy(track_point) if cols_exist(nam.xy(track_point), s) else ["x", "y"]
        if not cols_exist(XY, s):
            raise ValueError(
                "Defined point xy coordinates do not exist. Can not align trajectories! "
            )
        ids = s.index.unique(level="AgentID").values
        Nticks = len(s.index.unique("Step"))
        if mode == "origin":
            vprint("Aligning trajectories to common origin")
            xy = [s[XY].xs(id, level="AgentID").dropna().values[0] for id in ids]
        elif mode == "center":
            vprint(
                "Centralizing trajectories in trajectory center using min-max positions"
            )
            xy_max = [s[XY].xs(id, level="AgentID").max().values for id in ids]
            xy_min = [s[XY].xs(id, level="AgentID").min().values for id in ids]
            xy = [(max + min) / 2 for max, min in zip(xy_max, xy_min)]
        else:
            raise ValueError('Supported modes are "arena", "origin" and "center"!')
        xs = np.array([x for x, y in xy] * Nticks)
        ys = np.array([y for x, y in xy] * Nticks)

        for x, y in xy_pairs:
            ss[x] = ss[x].values - xs
            ss[y] = ss[y].values - ys

        if d is not None:
            d.store(ss, f"traj.{mode}")
            vprint(f"traj_aligned2{mode} stored")
        return ss


def fixate_larva(s, c, P1, P2=None):
    """
    Adjusts the coordinates of a larva in the dataset to fixate a primary point (P1) to the arena center,
    and optionally a secondary point (P2) to the vertical axis.

    Parameters:
    s (pd.DataFrame): The dataset's step_data containing the larva coordinates.
    c (object): The dataset's configuration dict.
    P1 (str): The primary point to be fixed to the arena center.
    P2 (str, optional): The secondary point to be fixed to the vertical axis. Defaults to None.

    Returns:
    pd.DataFrame: The modified dataset's step_data with adjusted coordinates.
    np.ndarray: Background array containing the transformations applied (bg_x, bg_y, bg_a).

    Raises:
    ValueError: If the primary or secondary point is not part of the dataset.
    """
    pars = c.all_xy.existing(s)
    if not nam.xy(P1).exist_in(s):
        raise ValueError(f" The requested {P1} is not part of the dataset")
    vprint(f"Fixing {P1} to arena center")
    X, Y = c.env_params.arena.dims
    xy = s[nam.xy(P1)].values
    xy_start = s[nam.xy(P1)].dropna().values[0]
    bg_x = (xy[:, 0] - xy_start[0]) / X
    bg_y = (xy[:, 1] - xy_start[1]) / Y

    for x, y in pars.in_pairs:
        s[[x, y]] -= xy

    N = s.index.unique("Step").size
    if P2 is not None:
        if not nam.xy(P2).exist_in(s):
            raise ValueError(
                f" The requested secondary {P2} is not part of the dataset"
            )
        vprint(f"Fixing {P2} as secondary point on vertical axis")
        xy_sec = s[nam.xy(P2)].values
        bg_a = np.arctan2(xy_sec[:, 1], xy_sec[:, 0]) - np.pi / 2

        s[pars] = [
            flatten_list(
                rotate_points_around_point(
                    points=np.reshape(s[pars].values[i, :], (-1, 2)), radians=bg_a[i]
                )
            )
            for i in range(N)
        ]
    else:
        bg_a = np.zeros(N)

    bg = np.vstack((bg_x, bg_y, bg_a))
    vprint("Fixed-point dataset generated")
    return s, bg


def epoch_overlap(epochs1, epochs2):
    """
    Find overlapping epochs between two sets of epochs.

    Parameters:
    epochs1 (numpy.ndarray): A 2D array where each row represents an epoch with a start and end time.
    epochs2 (numpy.ndarray): A 2D array where each row represents an epoch with a start and end time.

    Returns:
    numpy.ndarray: A 2D array containing the epochs from epochs1 that overlap with any epoch in epochs2.
    """
    valid = []
    if epochs1.shape[0] != 0 and epochs2.shape[0] != 0:
        for v in epochs1:
            temp = epochs2[epochs2[:, 0] <= v[0] and epochs2[:, 1] >= v[1]]
            if temp.shape[0] != 0:
                valid.append(v)
    return np.array(valid)


def epoch_slices(epochs):
    """
    Generate slices of indices for given epochs.

    Parameters:
    epochs (numpy.ndarray): A 2D array where each row represents an epoch with
                            start and end indices.

    Returns:
    list: A list of numpy arrays, each containing indices from start to end
          for each epoch.
    """
    if epochs.shape[0] == 0:
        return []
    else:
        return [np.arange(r0, r1, 1) for r0, r1 in epochs]
