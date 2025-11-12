"""
Methods for managing spatial metrics (2D x-y arrays)
"""

from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
from shapely import geometry, ops

# Avoid SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead
pd.options.mode.chained_assignment = None  # default='warn'
from typing import Any, Optional

import scipy as sp
from scipy.signal import find_peaks

from ... import vprint
from . import AttrDict, cols_exist, flatten_list, nam, rotate_points_around_point

__all__: list[str] = [
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
    # "align_trajectories",
    "fixate_larva",
    "epoch_overlap",
    "epoch_slices",
]


def fft_max(
    a: np.ndarray,
    dt: float,
    fr_range: tuple[float, float] = (0.0, +np.inf),
    return_amps: bool = False,
) -> float | tuple[float, np.ndarray]:
    """
    Compute power spectrum and dominant frequency of a signal.

    Args:
        a: 1D signal timeseries array
        dt: Timestep of the timeseries
        fr_range: Frequency range allowed (min, max)
        return_amps: If True, return both frequency and power spectrum array

    Returns:
        Dominant frequency within range, or tuple of (frequency, power spectrum array) if return_amps=True

    Example:
        >>> signal = np.sin(2 * np.pi * 1.5 * np.arange(0, 10, 0.1))
        >>> freq = fft_max(signal, dt=0.1, fr_range=(1.0, 2.0))
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


def detect_strides(
    a: np.ndarray,
    dt: float,
    vel_thr: float = 0.3,
    stretch: tuple[float, float] = (0.75, 2.0),
    fr: Optional[float] = None,
) -> np.ndarray:
    """
    Detect stride events in velocity timeseries.

    Args:
        a: 1D forward velocity timeseries array
        dt: Timestep of the timeseries
        vel_thr: Maximum velocity threshold for pause detection
        stretch: Min-max stretch of stride duration relative to frequency-based default
        fr: Dominant crawling frequency (auto-detected if None)

    Returns:
        Array of stride intervals, shape (N, 2) with [start_idx, end_idx] pairs

    Example:
        >>> velocity = np.array([0.1, 0.5, 0.8, 0.5, 0.1, 0.5, 0.8])
        >>> strides = detect_strides(velocity, dt=0.1)
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


def stride_interp(a: np.ndarray, strides: np.ndarray, Nbins: int = 64) -> np.ndarray:
    """
    Interpolate stride segments to uniform length.

    Args:
        a: 1D signal array
        strides: Array of stride intervals, shape (N, 2) with [start, end] indices
        Nbins: Number of bins for interpolation

    Returns:
        Array of interpolated strides, shape (N_strides, Nbins)

    Example:
        >>> signal = np.array([0, 1, 2, 1, 0, 1, 2, 1, 0])
        >>> strides = np.array([[0, 4], [4, 8]])
        >>> interp = stride_interp(signal, strides, Nbins=32)
    """
    x = np.linspace(0, 2 * np.pi, Nbins)
    aa = np.zeros([strides.shape[0], Nbins])
    for ii, (s0, s1) in enumerate(strides):
        aa[ii, :] = np.interp(x, np.linspace(0, 2 * np.pi, s1 - s0), a[s0:s1])
    return aa


def mean_stride_curve(
    a: np.ndarray, strides: np.ndarray, da: np.ndarray, Nbins: int = 64
) -> AttrDict:
    """
    Compute median stride curves separated by direction.

    Args:
        a: 1D signal array
        strides: Array of stride intervals
        da: Direction array (positive/negative values)
        Nbins: Number of bins for interpolation

    Returns:
        AttrDict with keys 'abs', 'plus', 'minus', 'norm' containing median stride curves

    Example:
        >>> signal = np.array([0, 1, 2, 1, 0, 1, 2, 1, 0])
        >>> strides = np.array([[0, 4], [4, 8]])
        >>> da = np.array([1, -1])
        >>> curves = mean_stride_curve(signal, strides, da)
    """
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


def comp_PI(
    arena_xdim: float, xs: np.ndarray, return_num: bool = False
) -> float | tuple[float, int]:
    """
    Compute preference index for spatial distribution.

    Calculates left-right preference index based on x-coordinates distribution
    in arena. Values range from -1 (all right) to +1 (all left).

    Args:
        arena_xdim: Arena x-dimension
        xs: Array of x-coordinates
        return_num: If True, also return sample count

    Returns:
        Preference index, or tuple of (index, count) if return_num=True

    Example:
        >>> xs = np.array([-0.3, -0.2, 0.1, 0.3])
        >>> pi = comp_PI(arena_xdim=1.0, xs=xs)
    """
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


def rolling_window(a: np.ndarray, w: int) -> np.ndarray:
    """
    Create rolling windows of size w from 1D array.

    Args:
        a: 1D input array
        w: Window size

    Returns:
        2D array of rolling windows, shape (N-w+1, w)

    Raises:
        ValueError: If input array is not 1-dimensional

    Example:
        >>> a = np.array([1, 2, 3, 4, 5])
        >>> windows = rolling_window(a, w=3)
    """
    # Get windows of size w from array a
    if a.ndim != 1:
        raise ValueError("Input array must be 1-dimensional")
    return np.vstack([np.roll(a, -i) for i in range(w)]).T[: -w + 1]


def straightness_index(ss: pd.DataFrame, rolling_ticks: np.ndarray) -> np.ndarray:
    """
    Compute straightness index over rolling windows.

    Straightness index is defined as 1 - (straight_line_distance / path_distance),
    ranging from 0 (perfectly straight) to 1 (highly tortuous).

    Args:
        ss: DataFrame with columns 'x', 'y', 'dst'
        rolling_ticks: Rolling window indices array

    Returns:
        Array of straightness index values

    Example:
        >>> ss = pd.DataFrame({'x': [0, 1, 2], 'y': [0, 0, 0], 'dst': [0, 1, 1]})
        >>> rolling_ticks = np.array([[0, 1], [1, 2]])
        >>> si = straightness_index(ss, rolling_ticks)
    """
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


def sense_food(
    pos: tuple[float, float],
    sources: Optional[Any] = None,
    grid: Optional[Any] = None,
    radius: Optional[float] = None,
) -> Any:
    """
    Detect food sources near a position.

    Args:
        pos: (x, y) position coordinates
        sources: Optional agent list with food sources
        grid: Optional grid object with food distribution
        radius: Detection radius for source-based sensing

    Returns:
        Grid cell coordinates, food source object, or None if no food detected

    Example:
        >>> pos = (0.5, 0.5)
        >>> cell = sense_food(pos, grid=food_grid)
    """
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
    Segment body contour into equal or custom-length segments via vertical lines.

    Args:
        Nsegs: Number of segments to divide the body into
        points: Array of shape (M, 2) representing body contour
        seg_ratio: Optional array of segment length ratios (default: equal segments)
        centered: If True, center segments around origin
        closed: If True, connect last point to first point in each segment

    Returns:
        Array of shape (Nsegs, L, 2) where L is vertices per segment, front segment first

    Example:
        >>> contour = np.array([[1, 0.1], [0.5, 0.1], [0, 0]])
        >>> segments = generate_seg_shapes(Nsegs=2, points=contour)
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
    """
    Exception raised when two objects collide.

    Attributes:
        object1: First colliding object
        object2: Second colliding object

    Example:
        >>> raise Collision(agent1, agent2)
    """

    def __init__(self, object1: Any, object2: Any) -> None:
        self.object1 = object1
        self.object2 = object2


def rearrange_contour(ps0: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Rearrange contour points by separating positive and negative y-values.

    Args:
        ps0: List of (x, y) contour points

    Returns:
        Rearranged list with positive y points (descending x) followed by negative y points (ascending x)

    Example:
        >>> points = [(1.0, 0.5), (0.5, -0.3), (0.8, 0.2)]
        >>> rearranged = rearrange_contour(points)
    """
    ps_plus = [p for p in ps0 if p[1] >= 0]
    ps_plus.sort(key=lambda x: x[0], reverse=True)
    ps_minus = [p for p in ps0 if p[1] < 0]
    ps_minus.sort(key=lambda x: x[0], reverse=False)
    return ps_plus + ps_minus


def comp_bearing(
    xs: np.ndarray,
    ys: np.ndarray,
    ors: float | np.ndarray,
    loc: tuple[float, float] = (0.0, 0.0),
    in_deg: bool = True,
) -> np.ndarray:
    """
    Compute bearing (azimuth) of oriented points relative to reference location.

    Args:
        xs: Array of x-coordinates
        ys: Array of y-coordinates
        ors: Orientation angles (in degrees)
        loc: Reference location (x, y)
        in_deg: If True, return bearings in degrees; if False, in radians

    Returns:
        Array of bearing angles, range (-180, 180] degrees or (-π, π] radians

    Example:
        >>> xs = np.array([1.0, 2.0, 3.0])
        >>> ys = np.array([1.0, 2.0, 0.0])
        >>> bearings = comp_bearing(xs, ys, ors=90.0)
        >>> # Returns [-135., -135., -90.]
    """
    x0, y0 = loc
    dxs = x0 - np.array(xs)
    dys = y0 - np.array(ys)
    rads = np.arctan2(dys, dxs)
    drads = (ors - np.rad2deg(rads)) % 360
    drads[drads > 180] -= 360
    return drads if in_deg else np.deg2rad(drads)


def comp_bearing_solo(
    x: float, y: float, o: float, loc: tuple[float, float] = (0.0, 0.0)
) -> float:
    """
    Compute bearing angle for single oriented point relative to location.

    Args:
        x: Point x-coordinate
        y: Point y-coordinate
        o: Orientation angle (radians)
        loc: Reference location (x, y)

    Returns:
        Bearing angle in radians, range (-π, π]

    Example:
        >>> bearing = comp_bearing_solo(x=1.0, y=1.0, o=np.pi/4, loc=(0.0, 0.0))
    """
    x0, y0 = loc
    b = (o - np.arctan2(y0 - y, x0 - x)) % (2 * np.pi)
    if b > np.pi:
        b -= 2 * np.pi
    return b


def compute_dispersal_solo(
    xy: np.ndarray | pd.DataFrame,
    min_valid_proportion: float = 0.2,
    max_start_proportion: float = 0.1,
    min_end_proportion: float = 0.9,
) -> np.ndarray:
    """
    Compute dispersal (distance from start) for single trajectory.

    Validates trajectory completeness before computing distances from initial position.

    Args:
        xy: Trajectory data, shape (N, 2) with [x, y] coordinates
        min_valid_proportion: Minimum proportion of non-NaN data points required (default: 0.2)
        max_start_proportion: Maximum proportion of NaN data allowed at start (default: 0.1)
        min_end_proportion: Minimum data proportion before last valid point (default: 0.9)

    Returns:
        Array of dispersal values, or NaN array if trajectory invalid

    Example:
        >>> xy = np.array([[0, 0], [1, 0], [2, 1]])
        >>> dispersal = compute_dispersal_solo(xy)
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


def compute_dispersal_multi(
    xy0: pd.DataFrame, t0: float, t1: float, dt: float, **kwargs: Any
) -> tuple[np.ndarray, int]:
    """
    Compute dispersal values for multiple agents over time range.

    Args:
        xy0: MultiIndex DataFrame with agent positions (levels: Step, AgentID)
        t0: Start time in seconds
        t1: End time in seconds
        dt: Timestep of timeseries
        **kwargs: Additional arguments passed to compute_dispersal_solo

    Returns:
        Tuple of (dispersal_array, n_timesteps) where dispersal_array is flattened

    Example:
        >>> xy_data = pd.DataFrame({...})  # MultiIndex DataFrame
        >>> dispersal, n_steps = compute_dispersal_multi(xy_data, t0=0, t1=10, dt=0.1)
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


def compute_component_velocity(
    xy: np.ndarray, angles: np.ndarray, dt: float, return_dst: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity component along orientation angles.

    Args:
        xy: Array of shape (N, 2) with [x, y] coordinates
        angles: Array of shape (N,) with orientation angles in radians
        dt: Time interval for velocity calculation
        return_dst: If True, return both velocities and displacements

    Returns:
        Velocity array, or tuple of (velocity, displacement) if return_dst=True

    Example:
        >>> xy = np.array([[0, 0], [1, 0], [2, 1]])
        >>> angles = np.array([0, 0, np.pi/4])
        >>> v = compute_component_velocity(xy, angles, dt=0.1)
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


def compute_velocity_threshold(
    v: np.ndarray,
    Nbins: int = 500,
    max_v: Optional[float] = None,
    kernel_width: float = 0.02,
) -> float:
    """
    Compute velocity threshold using density-based approach.

    Identifies minimum between local maxima and minima in smoothed density curve.

    Args:
        v: Input velocity data array
        Nbins: Number of histogram bins
        max_v: Maximum velocity value (auto-detected if None)
        kernel_width: Gaussian kernel width for density smoothing

    Returns:
        Computed velocity threshold

    Example:
        >>> velocities = np.random.exponential(0.5, 1000)
        >>> threshold = compute_velocity_threshold(velocities)
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


def get_display_dims() -> tuple[int, int]:
    """
    Get display dimensions scaled to 2/3 of screen size.

    Returns:
        Tuple of (width, height) in pixels, rounded to multiples of 16

    Example:
        >>> width, height = get_display_dims()
    """
    import pygame

    pygame.init()
    W, H = pygame.display.Info().current_w, pygame.display.Info().current_h
    return int(W * 2 / 3 / 16) * 16, int(H * 2 / 3 / 16) * 16


def get_window_dims(arena_dims: tuple[float, float]) -> tuple[int, int]:
    """
    Compute optimal window dimensions for arena visualization.

    Maintains aspect ratio while fitting within display bounds.

    Args:
        arena_dims: Arena dimensions (width, height)

    Returns:
        Tuple of (window_width, window_height) in pixels

    Example:
        >>> dims = get_window_dims(arena_dims=(0.2, 0.2))
    """
    X, Y = np.array(arena_dims)
    W0, H0 = get_display_dims()
    R0, R = W0 / H0, X / Y
    if R0 < R:
        return W0, int(W0 / R / 16) * 16
    else:
        return int(H0 * R / 16) * 16, H0


def get_arena_bounds(arena_dims: tuple[float, float], s: float = 1) -> np.ndarray:
    """
    Compute arena bounds centered at origin.

    Args:
        arena_dims: Arena dimensions (width, height)
        s: Scaling factor

    Returns:
        Array [x_min, x_max, y_min, y_max]

    Example:
        >>> bounds = get_arena_bounds(arena_dims=(1.0, 0.8))
        >>> # Returns [-0.5, 0.5, -0.4, 0.4]
    """
    X, Y = np.array(arena_dims) * s
    return np.array([-X / 2, X / 2, -Y / 2, Y / 2])


def circle_to_polygon(N: int, r: float) -> list[tuple[float, float]]:
    """
    Generate polygon vertices approximating a circle.

    Args:
        N: Number of vertices
        r: Radius of circle

    Returns:
        List of (x, y) vertex coordinates

    Example:
        >>> vertices = circle_to_polygon(N=8, r=1.0)
    """
    one_segment = np.pi * 2 / N

    points = [
        (np.sin(one_segment * i) * r, np.cos(one_segment * i) * r) for i in range(N)
    ]

    return points


def boolean_indexing(v: list[np.ndarray], fillval: float = np.nan) -> np.ndarray:
    """
    Convert list of variable-length arrays to 2D array with padding.

    Args:
        v: List of 1D numpy arrays with different lengths
        fillval: Value to use for padding shorter arrays

    Returns:
        2D array with shape (N, max_length), padded with fillval

    Example:
        >>> arrays = [np.array([1, 2]), np.array([3, 4, 5])]
        >>> result = boolean_indexing(arrays, fillval=0)
    """
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def concat_datasets(
    ddic: dict[str, Any], key: str = "end", unit: str = "sec"
) -> pd.DataFrame:
    """
    Concatenate multiple datasets into single DataFrame.

    Args:
        ddic: Dictionary mapping dataset IDs to dataset objects
        key: Data type to extract ('end' for endpoint_data, 'step' for step_data)
        unit: Time unit for step data ('sec', 'min', 'hour', 'day')

    Returns:
        Concatenated DataFrame with added DatasetID and GroupID columns

    Example:
        >>> datasets = {'exp1': dataset1, 'exp2': dataset2}
        >>> df = concat_datasets(datasets, key='step', unit='min')
    """
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


def moving_average(a: np.ndarray, n: int = 3) -> np.ndarray:
    """
    Compute moving average with window size n.

    Args:
        a: 1D input array
        n: Window size

    Returns:
        Array of moving averages (same length as input)

    Example:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> smoothed = moving_average(data, n=3)
    """
    return np.convolve(a, np.ones((n,)) / n, mode="same")


def body_contour(
    points: list[tuple[float, float]] = [(0.9, 0.1), (0.05, 0.1)],
    start: tuple[float, float] = (1, 0),
    stop: tuple[float, float] = (0, 0),
) -> np.ndarray:
    """
    Generate symmetric body contour from half-side points.

    Args:
        points: List of (x, y) points for upper half of body
        start: Starting point coordinates
        stop: Ending point coordinates

    Returns:
        Array of shape (2*N+2, 2) with full symmetric contour

    Example:
        >>> contour = body_contour(points=[(0.9, 0.1), (0.5, 0.15)])
    """
    xy = np.zeros([len(points) * 2 + 2, 2]) * np.nan
    xy[0, :] = start
    xy[len(points) + 1, :] = stop
    for i in range(len(points)):
        x, y = points[i]
        xy[1 + i, :] = x, y
        xy[-1 - i, :] = x, -y
    return xy


def apply_per_level(
    s: pd.DataFrame, func: Any, level: str = "AgentID", **kwargs: Any
) -> np.ndarray:
    """
    Apply function to each group in MultiIndex DataFrame.

    Args:
        s: MultiIndex DataFrame with levels ['Step', 'AgentID']
        func: Function to apply to each group
        level: Grouping level ('AgentID' or 'Step')
        **kwargs: Additional arguments passed to func

    Returns:
        Array of shape (N_steps, N_agents) with function results

    Example:
        >>> data = pd.DataFrame(...).set_index(['Step', 'AgentID'])
        >>> result = apply_per_level(data, np.mean, level='AgentID')
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


def unwrap_deg(a: np.ndarray | pd.Series) -> np.ndarray:
    """
    Unwrap angles in degrees to remove discontinuities.

    Args:
        a: Array or Series of angles in degrees

    Returns:
        Unwrapped angles in degrees

    Example:
        >>> angles = np.array([170, 180, -170, -160])
        >>> unwrapped = unwrap_deg(angles)
    """
    if isinstance(a, pd.Series):
        a = a.values
    b = np.copy(a)
    b[~np.isnan(b)] = np.unwrap(b[~np.isnan(b)] * np.pi / 180) * 180 / np.pi
    return b


def unwrap_rad(a: np.ndarray | pd.Series) -> np.ndarray:
    """
    Unwrap angles in radians to remove discontinuities.

    Args:
        a: Array or Series of angles in radians

    Returns:
        Unwrapped angles in radians

    Example:
        >>> angles = np.array([3.0, 3.14, -3.1, -3.0])
        >>> unwrapped = unwrap_rad(angles)
    """
    if isinstance(a, pd.Series):
        a = a.values
    b = np.copy(a)
    b[~np.isnan(b)] = np.unwrap(b[~np.isnan(b)])
    return b


def rate(a: np.ndarray | pd.Series, dt: float) -> np.ndarray:
    """
    Compute rate of change (derivative) of signal.

    Args:
        a: Input signal array or Series
        dt: Time step

    Returns:
        Array of rates, first element is NaN

    Example:
        >>> signal = np.array([0, 1, 3, 6])
        >>> velocity = rate(signal, dt=0.1)
    """
    if isinstance(a, pd.Series):
        a = a.values
    v = np.diff(a) / dt
    return np.insert(v, 0, np.nan)


def eudist(xy: np.ndarray | pd.DataFrame) -> np.ndarray:
    """
    Compute Euclidean distances between consecutive points in trajectory.

    Args:
        xy: Trajectory array or DataFrame, shape (N, 2) with [x, y] coordinates

    Returns:
        Array of cumulative distances, first element is 0

    Example:
        >>> xy = np.array([[0, 0], [1, 0], [1, 1]])
        >>> distances = eudist(xy)
        >>> # Returns [0, 1.0, 1.0]
    """
    if isinstance(xy, pd.DataFrame):
        xy = xy.values
    A = np.sqrt(np.nansum(np.diff(xy, axis=0) ** 2, axis=1))
    A = np.insert(A, 0, 0)
    return A


def eudi5x(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distances between points in arrays a and b.

    Args:
        a: Array of shape (N, D) with N points in D dimensions
        b: Single point or array of shape (D,) to measure distance from

    Returns:
        Array of N Euclidean distances

    Example:
        >>> a = np.array([[0, 0], [1, 0], [0, 1]])
        >>> b = np.array([0.5, 0.5])
        >>> distances = eudi5x(a, b)
    """
    return np.sqrt(np.sum((a - np.array(b)) ** 2, axis=1))


def eudiNxN(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between two sets of points.

    Args:
        a: Array of shape (N, M, 2) representing N sets of M points
        b: Array of shape (K, 2) representing K reference points

    Returns:
        Array of shape (N, M, K) with pairwise distances

    Example:
        >>> a = np.random.rand(5, 10, 2)
        >>> b = np.random.rand(3, 2)
        >>> distances = eudiNxN(a, b)
    """
    b = np.array(b)
    return np.sqrt(np.sum(np.array([a - b[i] for i in range(b.shape[0])]) ** 2, axis=2))


def compute_dst(s: pd.DataFrame, point: str = "") -> None:
    """
    Compute and add distance column to DataFrame (in-place).

    Args:
        s: MultiIndex DataFrame with trajectory data
        point: Point identifier (empty for default midpoint)

    Example:
        >>> compute_dst(step_data, point="head")
    """
    s[nam.dst(point)] = apply_per_level(s[nam.xy(point)], eudist).flatten()


def comp_extrema(
    a: pd.Series,
    order: int = 3,
    threshold: Optional[tuple[float, float]] = None,
    return_2D: bool = True,
) -> np.ndarray:
    """
    Compute local extrema in time series using scipy.signal.argrelextrema.

    Args:
        a: Input time series as pandas Series
        order: Order of extrema detection (minimum separation)
        threshold: Optional (min_threshold, max_threshold) to filter extrema by value
        return_2D: If True, return 2D array [minima_flags, maxima_flags]; if False, return 1D (-1/1/NaN)

    Returns:
        Array with extrema flags (shape (N, 2) if return_2D=True, else (N,))

    Example:
        >>> data = pd.Series([1, 3, 2, 4, 1, 5])
        >>> extrema = comp_extrema(data, order=1, return_2D=False)
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

    """
    NOTE:Refactored as a method of LarvaDataset class
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
    """


def fixate_larva(
    s: pd.DataFrame,
    c: Any,
    arena_dims: tuple[float, float],
    P1: str,
    P2: Optional[str] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Transform coordinates to fixate primary point to arena center.

    Optionally aligns secondary point to vertical axis via rotation.

    Args:
        s: Step data DataFrame with trajectory coordinates
        c: Dataset configuration object
        arena_dims: Arena dimensions (width, height)
        P1: Primary point identifier to fix to center
        P2: Optional secondary point to align to vertical axis

    Returns:
        Tuple of (transformed_dataframe, background_transformations) where background is [bg_x, bg_y, bg_angle]

    Raises:
        ValueError: If requested point not found in dataset

    Example:
        >>> s_fixed, bg = fixate_larva(step_data, config, (0.2, 0.2), P1='centroid', P2='head')
    """
    pars = c.all_xy.existing(s)
    if not nam.xy(P1).exist_in(s):
        raise ValueError(f" The requested {P1} is not part of the dataset")
    vprint(f"Fixing {P1} to arena center")
    X, Y = arena_dims
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


def epoch_overlap(epochs1: np.ndarray, epochs2: np.ndarray) -> np.ndarray:
    """
    Find epochs from epochs1 that overlap with any epoch in epochs2.

    Args:
        epochs1: Array of shape (N, 2) with [start, end] time pairs
        epochs2: Array of shape (M, 2) with [start, end] time pairs

    Returns:
        Array of overlapping epochs from epochs1

    Example:
        >>> epochs1 = np.array([[0, 5], [10, 15]])
        >>> epochs2 = np.array([[3, 12]])
        >>> overlapping = epoch_overlap(epochs1, epochs2)
    """
    valid = []
    if epochs1.shape[0] != 0 and epochs2.shape[0] != 0:
        for v in epochs1:
            temp = epochs2[epochs2[:, 0] <= v[0] and epochs2[:, 1] >= v[1]]
            if temp.shape[0] != 0:
                valid.append(v)
    return np.array(valid)


def epoch_slices(epochs: np.ndarray) -> list[np.ndarray]:
    """
    Generate index arrays for each epoch interval.

    Args:
        epochs: Array of shape (N, 2) with [start_idx, end_idx] pairs

    Returns:
        List of N index arrays, each covering one epoch interval

    Example:
        >>> epochs = np.array([[0, 3], [5, 8]])
        >>> slices = epoch_slices(epochs)
        >>> # Returns [array([0, 1, 2]), array([5, 6, 7])]
    """
    if epochs.shape[0] == 0:
        return []
    else:
        return [np.arange(r0, r1, 1) for r0, r1 in epochs]
