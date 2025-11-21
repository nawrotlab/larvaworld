"""
Parameter-computing functions.
Contains functions that compute/derive higher-order parameters from the existing ones in the dataset.
"""

from __future__ import annotations
from typing import Any, Callable

import copy

import numpy as np

from .. import util, funcs
from ..util import nam

__all__: list[str] = [
    "track_par_func",
    "chunk_func",
    "dsp_func",
    "tor_func",
    "mean_func",
    "std_func",
    "var_func",
    "min_func",
    "max_func",
    "fin_func",
    "init_func",
    "cum_func",
    "freq_func",
    "tr_func",
    "unwrap_func",
    "dst_func",
    "func_v_spatial",
]


@funcs.param("track_par")
def track_par_func(chunk: str, par: str) -> Callable[[Any], None]:
    """
    Create a parameter tracking function for a specific chunk.

    Factory function that generates a callable to track parameters within
    behavioral chunks (e.g., strides, pauses, runs).

    Args:
        chunk: Name of the behavioral chunk (e.g., 'str', 'pau', 'run')
        par: Parameter name to track within the chunk

    Returns:
        Callable that accepts a dataset and tracks the parameter in the chunk

    Example:
        >>> tracker = track_par_func('str', 'velocity')
        >>> tracker(dataset)  # Tracks velocity in stride chunks
    """

    def func(d):
        d.track_par_in_chunk(chunk, par)

    return func


@funcs.param("chunk")
def chunk_func(kc: str) -> util.AttrDict:
    """
    Create chunk annotation function with required parameters.

    Factory function that generates annotation functions for behavioral chunks
    (crawl or turn events) along with their required parameter keys.

    Args:
        kc: Chunk key - one of:
            - Crawl chunks: 'str', 'pau', 'exec', 'str_c', 'run'
            - Turn chunks: 'tur', 'Ltur', 'Rtur'

    Returns:
        AttrDict with:
            - func: Annotation function (or None for unknown chunks)
            - required_ks: List of required parameter keys

    Example:
        >>> result = chunk_func('str')
        >>> result.func(dataset)  # Performs crawl annotation
        >>> print(result.required_ks)  # ['a', 'sa', 'ba', 'foa', 'fv']
    """
    if kc in ["str", "pau", "exec", "str_c", "run"]:

        def func(d):
            d.crawl_annotation()

        required_ks = ["a", "sa", "ba", "foa", "fv"]
    elif kc in ["tur", "Ltur", "Rtur"]:

        def func(d):
            d.turn_annotation()

        required_ks = ["fov"]
    else:
        func = None
        required_ks = []
    return util.AttrDict({"func": func, "required_ks": required_ks})


@funcs.param("dsp")
def dsp_func(range: tuple[int, int]) -> Callable[[Any], None]:
    """
    Create dispersal computation function for a time range.

    Factory function that generates a callable to compute agent dispersal
    (spatial spread) between two time points.

    Args:
        range: Time range as (start_index, end_index) tuple

    Returns:
        Callable that accepts a dataset and computes dispersal

    Example:
        >>> dispersal_fn = dsp_func((0, 100))
        >>> dispersal_fn(dataset)  # Computes dispersal from t=0 to t=100
    """
    r0, r1 = range

    def func(d):
        d.comp_dispersal(r0, r1)

    return func


@funcs.param("tor")
def tor_func(dur: int) -> Callable[[Any], None]:
    """
    Create tortuosity computation function for a time window.

    Factory function that generates a callable to compute path tortuosity
    (straightness) over a specified duration.

    Args:
        dur: Duration window for tortuosity calculation (in time steps)

    Returns:
        Callable that accepts a dataset and computes tortuosity

    Example:
        >>> tortuosity_fn = tor_func(10)
        >>> tortuosity_fn(dataset)  # Computes tortuosity over 10-step windows
    """

    def func(d):
        d.comp_tortuosity(dur)

    return func


@funcs.param("mean")
def mean_func(par: str) -> Callable[[Any], None]:
    """
    Create function to compute per-agent mean of a parameter.

    Factory function that generates a callable to calculate mean values
    per agent and store in the endpoint dataset.

    Args:
        par: Parameter name to compute mean for

    Returns:
        Callable that accepts a dataset and computes per-agent means

    Example:
        >>> mean_vel = mean_func('velocity')
        >>> mean_vel(dataset)  # Adds 'velocity_mean' to endpoint data
    """

    def func(d):
        d.e[nam.mean(par)] = d.s[par].dropna().groupby("AgentID").mean()

    return func


@funcs.param("std")
def std_func(par: str) -> Callable[[Any], None]:
    """
    Create function to compute per-agent standard deviation of a parameter.

    Factory function that generates a callable to calculate standard deviation
    per agent and store in the endpoint dataset.

    Args:
        par: Parameter name to compute standard deviation for

    Returns:
        Callable that accepts a dataset and computes per-agent standard deviations

    Example:
        >>> std_vel = std_func('velocity')
        >>> std_vel(dataset)  # Adds 'velocity_std' to endpoint data
    """

    def func(d):
        d.e[nam.std(par)] = d.s[par].dropna().groupby("AgentID").std()

    return func


@funcs.param("var")
def var_func(par: str) -> Callable[[Any], None]:
    """
    Create function to compute per-agent coefficient of variation.

    Factory function that generates a callable to calculate coefficient of
    variation (mean/std) per agent and store in the endpoint dataset.

    Args:
        par: Parameter name to compute coefficient of variation for

    Returns:
        Callable that accepts a dataset and computes per-agent coefficients

    Example:
        >>> var_vel = var_func('velocity')
        >>> var_vel(dataset)  # Adds 'velocity_var' to endpoint data
    """

    def func(d):
        d.e[nam.var(par)] = (
            d.s[par].dropna().groupby("AgentID").mean()
            / d.s[par].dropna().groupby("AgentID").std()
        )

    return func


@funcs.param("min")
def min_func(par: str) -> Callable[[Any], None]:
    """
    Create function to compute per-agent minimum of a parameter.

    Factory function that generates a callable to calculate minimum values
    per agent and store in the endpoint dataset.

    Args:
        par: Parameter name to compute minimum for

    Returns:
        Callable that accepts a dataset and computes per-agent minimums

    Example:
        >>> min_vel = min_func('velocity')
        >>> min_vel(dataset)  # Adds 'velocity_min' to endpoint data
    """

    def func(d):
        d.e[nam.min(par)] = d.s[par].dropna().groupby("AgentID").min()

    return func


@funcs.param("max")
def max_func(par: str) -> Callable[[Any], None]:
    """
    Create function to compute per-agent maximum of a parameter.

    Factory function that generates a callable to calculate maximum values
    per agent and store in the endpoint dataset.

    Args:
        par: Parameter name to compute maximum for

    Returns:
        Callable that accepts a dataset and computes per-agent maximums

    Example:
        >>> max_vel = max_func('velocity')
        >>> max_vel(dataset)  # Adds 'velocity_max' to endpoint data
    """

    def func(d):
        d.e[nam.max(par)] = d.s[par].dropna().groupby("AgentID").max()

    return func


@funcs.param("final")
def fin_func(par: str) -> Callable[[Any], None]:
    """
    Create function to extract final value of a parameter per agent.

    Factory function that generates a callable to get the last (final) value
    per agent and store in the endpoint dataset.

    Args:
        par: Parameter name to extract final value for

    Returns:
        Callable that accepts a dataset and extracts final values

    Example:
        >>> final_pos = fin_func('position')
        >>> final_pos(dataset)  # Adds 'position_final' to endpoint data
    """

    def func(d):
        d.e[nam.final(par)] = d.s[par].dropna().groupby("AgentID").last()

    return func


@funcs.param("initial")
def init_func(par: str) -> Callable[[Any], None]:
    """
    Create function to extract initial value of a parameter per agent.

    Factory function that generates a callable to get the first (initial) value
    per agent and store in the endpoint dataset.

    Args:
        par: Parameter name to extract initial value for

    Returns:
        Callable that accepts a dataset and extracts initial values

    Example:
        >>> initial_pos = init_func('position')
        >>> initial_pos(dataset)  # Adds 'position_initial' to endpoint data
    """

    def func(d):
        d.e[nam.initial(par)] = d.s[par].dropna().groupby("AgentID").first()

    return func


@funcs.param("cum")
def cum_func(par: str) -> Callable[[Any], None]:
    """
    Create function to compute per-agent cumulative sum of a parameter.

    Factory function that generates a callable to calculate cumulative (total)
    sum per agent and store in the endpoint dataset.

    Args:
        par: Parameter name to compute cumulative sum for

    Returns:
        Callable that accepts a dataset and computes per-agent cumulative sums

    Example:
        >>> cum_dist = cum_func('distance')
        >>> cum_dist(dataset)  # Adds 'distance_cum' to endpoint data
    """

    def func(d):
        d.e[nam.cum(par)] = d.s[par].dropna().groupby("AgentID").sum()

    return func


@funcs.param("freq")
def freq_func(par: str) -> Callable[[Any], None]:
    """
    Create function to compute frequency spectrum of a parameter.

    Factory function that generates a callable to calculate frequency domain
    representation (FFT) of a time series parameter.

    Args:
        par: Parameter name to compute frequency spectrum for

    Returns:
        Callable that accepts a dataset and computes frequency spectrum

    Example:
        >>> freq_vel = freq_func('velocity')
        >>> freq_vel(dataset)  # Computes FFT of velocity time series
    """

    def func(d):
        d.comp_freq(par=par, fr_range=(0.0, +np.inf))

    return func


@funcs.param("tr")
def tr_func(pc: str) -> Callable[[Any], None]:
    """
    Create function to compute time ratio for a behavioral chunk.

    Factory function that generates a callable to calculate the ratio of
    time spent in a specific chunk relative to total time.

    Args:
        pc: Chunk name (e.g., 'pau' for pause, 'run' for run)

    Returns:
        Callable that accepts a dataset and computes time ratio

    Example:
        >>> pause_ratio = tr_func('pau')
        >>> pause_ratio(dataset)  # Adds 'pau_dur_ratio' to endpoint data
    """

    def func(d):
        d.e[nam.dur_ratio(pc)] = d.e[nam.cum(nam.dur(pc))] / d.e[nam.cum(nam.dur(""))]

    return func


@funcs.param("unwrap")
def unwrap_func(par: str, in_deg: bool) -> Callable[[Any], None]:
    """
    Create function to unwrap angular parameter discontinuities.

    Factory function that generates a callable to remove 360° discontinuities
    from angular time series data.

    Args:
        par: Angular parameter name to unwrap
        in_deg: Whether the parameter is in degrees (vs radians)

    Returns:
        Callable that accepts a dataset and unwraps the angular parameter

    Example:
        >>> unwrap_orient = unwrap_func('orientation', in_deg=True)
        >>> unwrap_orient(dataset)  # Adds 'orientation_unwrap' column
    """

    def func(d):
        s = copy.deepcopy(d.s[par])
        d.s[nam.unwrap(par)] = util.apply_per_level(s, util.unwrap_deg).flatten()

    return func


@funcs.param("dst")
def dst_func(point: str = "") -> Callable[[Any], None]:
    """
    Create function to compute distance from a reference point.

    Factory function that generates a callable to calculate distances from
    agents to a specified reference point or body segment.

    Args:
        point: Reference point name (e.g., 'centroid', 'head').
               Empty string uses default reference point.

    Returns:
        Callable that accepts a dataset and computes distances

    Example:
        >>> dist_to_head = dst_func('head')
        >>> dist_to_head(dataset)  # Computes distances from head point
    """

    def func(d):
        util.compute_dst(d.s, point)

    return func


@funcs.param("vel")
def func_v_spatial(p_d: str, p_v: str) -> Callable[[Any], None]:
    """
    Create function to compute velocity from displacement.

    Factory function that generates a callable to calculate velocity by
    dividing displacement by time step.

    Args:
        p_d: Displacement parameter name (e.g., 'distance')
        p_v: Velocity parameter name to create (e.g., 'velocity')

    Returns:
        Callable that accepts a dataset and computes velocity

    Example:
        >>> vel_calc = func_v_spatial('linear_displacement', 'lin_velocity')
        >>> vel_calc(dataset)  # Adds 'lin_velocity' column
    """

    def func(d):
        d.s[p_v] = d.s[p_d] / d.c.dt

    return func
