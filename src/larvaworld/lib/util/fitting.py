from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from numpy.lib import scimath
from scipy.optimize import minimize

# from scipy.stats import levy, norm, rv_discrete, ks_2samp


__all__: list[str] = [
    "simplex",
    "beta0",
    "exp_bout",
    "critical_bout",
]


def simplex(
    func: Callable[..., float], x0: float | np.ndarray, args: tuple[Any, ...] = ()
) -> float:
    """
    Nelder-Mead simplex optimization.

    Minimizes function using Nelder-Mead algorithm with tight tolerance.

    Args:
        func: Function to minimize.
        x0: Initial parameter value(s).
        args: Additional arguments passed to func.

    Returns:
        Optimized parameter value.

    Example:
        >>> def cost(x): return (x - 3)**2
        >>> result = simplex(cost, x0=0.0)
    """
    res = minimize(
        func,
        x0,
        args=args,
        method="nelder-mead",
        options={"xatol": 1e-8, "disp": False},
    ).x[0]
    return res


def beta0(x0: float, x1: float) -> float:
    """
    Compute beta function used in DEB textbook (p.58).

    Implements the beta function for Dynamic Energy Budget (DEB) modeling
    as defined in the DEB textbook, page 58.

    Args:
        x0: First parameter value
        x1: Second parameter value

    Returns:
        Real part of the computed beta function difference

    Example:
        >>> result = beta0(0.5, 1.0)
        >>> isinstance(result, float)
        True
    """
    x03 = x0 ** (1 / 3)
    x13 = x1 ** (1 / 3)
    a3 = math.sqrt(3)

    f1 = (
        -3 * x13
        + a3 * np.arctan((1 + 2 * x13) / a3)
        - scimath.log(x13 - 1)
        + scimath.log(1 + x13 + x13**2) / 2
    )
    f0 = (
        -3 * x03
        + a3 * np.arctan((1 + 2 * x03) / a3)
        - scimath.log(x03 - 1)
        + scimath.log(1 + x03 + x03**2) / 2
    )
    return np.real(f1 - f0)


def critical_bout(
    c: float = 0.9, sigma: float = 1, N: int = 1000, tmax: int = 1100, tmin: int = 1
) -> int:
    """
    Stochastic bout duration with critical dynamics.

    Simulates behavioral bout durations using critical point dynamics
    with population size N and control parameters c, sigma.

    Args:
        c: Control parameter (default: 0.9).
        sigma: Noise parameter (default: 1).
        N: Population size (default: 1000).
        tmax: Maximum bout duration (default: 1100).
        tmin: Minimum bout duration (default: 1).

    Returns:
        Bout duration in timesteps.

    Example:
        >>> duration = critical_bout(c=0.9, sigma=1.0, N=1000)
    """
    t = 0
    S = 1
    S_prev = 0
    while S > 0:
        p = (sigma * S - c * (S - S_prev)) / N
        p = np.clip(p, 0, 1)
        S_prev = S
        S = np.random.binomial(N, p)
        t += 1
        if t > tmax:
            t = 0
            S = 1
            S_prev = 0
        if S <= 0 and t < tmin:
            t = 0
            S = 1
            S_prev = 0
    return t


def exp_bout(beta: float = 0.01, tmax: int = 1100, tmin: int = 1) -> int:
    """
    Stochastic bout duration with exponential dynamics.

    Simulates behavioral bout durations using exponential waiting time
    with rate parameter beta.

    Args:
        beta: Rate parameter for exponential process (default: 0.01).
        tmax: Maximum bout duration (default: 1100).
        tmin: Minimum bout duration (default: 1).

    Returns:
        Bout duration in timesteps.

    Example:
        >>> duration = exp_bout(beta=0.01, tmax=1000)
    """
    t = 0
    S = 0
    while S <= 0:
        S = int(np.random.rand() < beta)
        t += 1
        if t > tmax:
            t = 0
            S = 0
        if S > 0 and t < tmin:
            t = 0
            S = 0
    return t
