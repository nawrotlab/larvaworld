"""
Methods for managing nans in timeseries data
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy as sp
from scipy.signal import butter, sosfiltfilt

__all__: list[str] = [
    "interpolate_nans",
    "parse_array_at_nans",
    "apply_sos_filter_to_array_with_nans",
    "apply_filter_to_array_with_nans_multidim",
    "convex_hull",
]


def nan_helper(
    y: np.ndarray,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Handle indices and logical indices of NaNs.

    Args:
        y: 1D array with possible NaNs.

    Returns:
        A tuple of the logical NaN mask and a function converting that mask
        into the equivalent positional indices.

    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x = nan_helper(y)
        >>> y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_nans(y: np.ndarray) -> np.ndarray:
    """
    Replace NaNs in a 1D array by linear interpolation, in place.

    Args:
        y: 1D array with possible NaNs.

    Returns:
        The same array with its NaNs filled.
    """
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


def parse_array_at_nans(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Locate the contiguous non-NaN stretches of an array.

    Args:
        a: 1D array with possible NaNs.

    Returns:
        A tuple of the start and end indices of each non-NaN stretch.
    """
    a = np.insert(a, 0, np.nan)
    a = np.insert(a, -1, np.nan)
    dif = np.diff(np.isnan(a).astype(int))
    de = np.where(dif == 1)[0]
    ds = np.where(dif == -1)[0]
    return ds, de


def apply_sos_filter_to_array_with_nans(
    sos: np.ndarray, x: np.ndarray, padlen: int = 6
) -> np.ndarray:
    """
    Apply a second-order-sections filter, skipping over NaN gaps.

    Each contiguous non-NaN stretch longer than ``padlen`` is filtered
    independently, so that gaps do not smear across the signal. If the
    stretch-wise pass fails, the filter is applied to the whole array instead.

    Args:
        sos: Second-order-sections filter coefficients.
        x: The signal to filter.
        padlen: Minimum stretch length, and the filter padding length.

    Returns:
        The filtered signal, NaN wherever the input was NaN or the stretch was
        too short to filter.
    """
    try:
        A = np.full_like(x, np.nan)
        ds, de = parse_array_at_nans(x)
        for s, e in zip(ds, de):
            k = x[s:e]
            if len(k) > padlen:
                A[s:e] = sosfiltfilt(sos, x[s:e], padlen=padlen)
        return A
    except:
        return sosfiltfilt(sos, x, padlen=padlen)


def apply_filter_to_array_with_nans_multidim(
    a: np.ndarray, freq: float, fr: float, N: int = 1
) -> np.ndarray:
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
        sos = butter(N=N, Wn=freq, btype="lowpass", analog=False, fs=fr, output="sos")
        return apply_sos_filter_to_array_with_nans(sos=sos, x=a)
    elif a.ndim == 2:
        sos = butter(N=N, Wn=freq, btype="lowpass", analog=False, fs=fr, output="sos")
        return np.array(
            [
                apply_sos_filter_to_array_with_nans(sos=sos, x=a[:, i])
                for i in range(a.shape[1])
            ]
        ).T
    elif a.ndim == 3:
        return np.transpose(
            [
                apply_filter_to_array_with_nans_multidim(a[:, :, i], freq, fr, N=1)
                for i in range(a.shape[2])
            ],
            (1, 2, 0),
        )
    else:
        raise ValueError("Method implement for up to 3-dimensional array")


def convex_hull(
    xs: np.ndarray | None = None,
    ys: np.ndarray | None = None,
    N: int | None = None,
    interp_nans: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the convex hull of each row of a set of 2D point clouds.

    NaNs are dropped before the hull is computed. Rows whose hull has fewer
    than ``N`` vertices are padded with NaN, and rows for which no hull can be
    computed are left entirely NaN.

    Args:
        xs: Array of shape ``(Nrows, Ncols)`` holding the x coordinates.
        ys: Array of shape ``(Nrows, Ncols)`` holding the y coordinates.
        N: Number of hull vertices to keep per row.
        interp_nans: When True, interpolate the NaN padding of each hull row.

    Returns:
        The x and y coordinates of the hull vertices, each of shape
        ``(Nrows, N)``.
    """
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
                b = p[sp.spatial.ConvexHull(p).vertices]
                s = np.min([b.shape[0], N])
                xxs[i, :s] = b[:s, 0]
                yys[i, :s] = b[:s, 1]
                if interp_nans:
                    xxs[i] = interpolate_nans(xxs[i])
                    yys[i] = interpolate_nans(yys[i])
            except:
                pass
    return xxs, yys
