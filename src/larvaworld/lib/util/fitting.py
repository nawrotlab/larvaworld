# import warnings
import math

import numpy as np
from numpy.lib import scimath
from scipy.optimize import minimize

# from scipy.stats import levy, norm, rv_discrete, ks_2samp


__all__ = [
    "simplex",
    "beta0",
    "exp_bout",
    "critical_bout",
]


def simplex(func, x0, args=()):
    res = minimize(
        func,
        x0,
        args=args,
        method="nelder-mead",
        options={"xatol": 1e-8, "disp": False},
    ).x[0]
    return res


def beta0(x0, x1):
    """
    Beta function used in the DEB textbook (p.58)

    Args:
        x0:float
        x1:float

    Returns:float

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


def critical_bout(c=0.9, sigma=1, N=1000, tmax=1100, tmin=1):
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


def exp_bout(beta=0.01, tmax=1100, tmin=1):
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
