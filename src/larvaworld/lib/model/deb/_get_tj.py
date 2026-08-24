r"""
Port of DEBtool_M/animal/get_tj.m — scaled ages and lengths at metamorphosis.

This is a faithful Python transcription of the constant-food branch only,
sufficient for the abp-family holometabolous insects (Drosophila).

See get_tj.m source at:
  c:\Users\Panos\AmP Drosophila matlab workspace\DEBtool_M\animal\get_tj.m

Citation:
  Bas Kooijman et al. "DEBtool_M" (MATLAB DEB model library)
  https://www.bio.vu.nl/thb/deb/deblab/
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .deb_equations import get_lb, get_ue0, beta0


def get_tj(
    p: tuple[float, float, float, float, float, float],
    f: Optional[float] = None,
) -> tuple[float, float, float, float, float, float, float, float, float, int]:
    """
    Scaled ages at metamorphosis, puberty, and birth; scaled lengths.

    Ported from DEBtool_M/animal/get_tj.m, constant-food branch only (no
    varying-food ODE integration).

    Parameters
    ----------
    p : 6-tuple
        (g, k, l_T, v_Hb, v_Hj, v_Hp)
        - g: energy investment ratio
        - k: k_J / k_M, maintenance ratio
        - l_T: scaled heating length {p_T} / [p_M] L_m
        - v_Hb: scaled maturity at birth
        - v_Hj: scaled maturity at end of acceleration (metamorphosis to juvenile)
        - v_Hp: scaled maturity at puberty
    f : float, optional
        Scaled functional response (default: 1)

    Returns
    -------
    (tau_j, tau_p, tau_b, l_j, l_p, l_b, l_i, rho_j, rho_B, info)
        - tau_*: scaled ages at metamorphosis (j), puberty (p), birth (b)
        - l_*: scaled structural lengths
        - l_i: ultimate scaled length
        - rho_j: exponential growth rate (birth to metamorphosis)
        - rho_B: von Bertalanffy growth rate (metamorphosis to ultimate)
        - info: 1 on success, 0 on failure

    Notes
    -----
    Multiply returned ages by k_M to get unscaled ages in days.
    Multiply returned lengths by L_m to get unscaled structural lengths in cm.
    """
    if f is None:
        f = 1.0

    g, k, l_T, v_Hb, v_Hj, v_Hp = p

    # Birth (embryo solution, reusing existing get_lb if available, else solve inline)
    tau_b, l_b = get_tb([g, k, v_Hb], f)
    e_b = f
    vel_b = np.array([v_Hb, e_b, l_b])

    # Growth rates
    rho_j = (f / l_b - 1.0 - l_T / l_b) / (1.0 + f / g)  # exponential
    rho_B = 1.0 / 3.0 / (1.0 + f / g)  # von Bertalanffy

    # Juvenile and adult (constant food, using fzero-style root-finding)
    info = 1
    try:
        l_j = _get_lj_root(v_Hj, l_b, v_Hb, l_T, rho_j, rho_B, k, g, f)
        if not (l_b < l_j < 1.0):
            info = 0
            return (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                info,
            )
        l_p = _get_lp_root(
            v_Hp, l_j, v_Hj, l_b, v_Hb, tau_b, l_T, rho_j, rho_B, k, g, f
        )
        if not (l_j < l_p < 1.0):
            info = 0
            return (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                info,
            )
    except Exception:
        info = 0
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            info,
        )

    s_M = l_j / l_b
    l_i = s_M * (f - l_T)
    tau_j = tau_b + np.log(s_M) * 3.0 / rho_j
    tau_p = tau_j + np.log((l_i - l_j) / (l_i - l_p)) / rho_B

    # Final checks
    if not (np.isfinite(tau_j) and np.isfinite(tau_p) and tau_j > 0 and tau_p > tau_j):
        info = 0

    return tau_j, tau_p, tau_b, l_j, l_p, l_b, l_i, rho_j, rho_B, info


def get_tb(p: tuple[float, float, float], f: float = 1.0) -> tuple[float, float]:
    """
    Scaled age and length at birth.

    Ported from DEBtool_M/animal/get_tb.m via the existing `get_lb` in
    deb_equations.py.

    Parameters
    ----------
    p : 3-tuple
        (g, k, v_Hb)
    f : float
        Scaled functional response

    Returns
    -------
    (tau_b, l_b)
        - tau_b: scaled age at birth
        - l_b: scaled length at birth
    """
    g, k, v_Hb = p
    l_b, info = get_lb(g=g, k=k, v_Hb=v_Hb, eb=f)
    if info == 0:
        raise ValueError(f"get_lb failed for p={p}, f={f}")
    # Compute tau_b from the maturity integral
    tau_b = _get_tau_b(g, k, l_b, v_Hb, f)
    return tau_b, l_b


def _get_tau_b(g: float, k: float, l_b: float, v_Hb: float, f: float) -> float:
    """Scaled age at birth from maturity integral (internal helper)."""
    xb = g / (f + g)
    n = int(1000 + round(1000 * max(0.0, k - 1.0)))
    dx = xb / n
    x = np.linspace(1e-5, xb, n)
    x3 = x ** (1.0 / 3.0)
    xb3 = xb ** (1.0 / 3.0)
    l_b3 = l_b ** (1.0 / 3.0)
    b = beta0(x, xb) / (3.0 * g)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        l = x3 / (xb3 / l_b3 - b)
        s = (k - x) / (1.0 - x) * l / g / x
        vv = np.exp(-dx * np.cumsum(s))
        r = g + l
        tau_b = v_Hb * dx * np.sum(r / vv)

    return float(tau_b)


def _get_lj_residual(
    l_j: float,
    v_Hj: float,
    l_b: float,
    v_Hb: float,
    l_T: float,
    rho_j: float,
    rho_B: float,
    k: float,
    g: float,
    f: float,
) -> float:
    """Residual for maturity condition at metamorphosis (used by root-finding)."""
    s_M = l_j / l_b
    s_j = s_M ** (-3.0 * k / rho_j)
    fn = (
        v_Hj
        - f * l_b**3 * (1.0 / l_b - rho_j / g) / (k + rho_j) * (s_M**3 - s_j)
        - v_Hb * s_j
    )
    return fn


def _get_lj_root(
    v_Hj: float,
    l_b: float,
    v_Hb: float,
    l_T: float,
    rho_j: float,
    rho_B: float,
    k: float,
    g: float,
    f: float,
) -> float:
    """Root of metamorphosis maturity condition."""
    # Bracket between l_b and 1.0
    try:
        return brentq(
            _get_lj_residual,
            l_b + 1e-9,
            0.9999,
            args=(v_Hj, l_b, v_Hb, l_T, rho_j, rho_B, k, g, f),
        )
    except ValueError:
        raise ValueError("lj root-finding failed")


def _get_lp_residual(
    l_p: float,
    v_Hp: float,
    l_j: float,
    v_Hj: float,
    l_b: float,
    v_Hb: float,
    tau_b: float,
    l_T: float,
    rho_j: float,
    rho_B: float,
    k: float,
    g: float,
    f: float,
) -> float:
    """Residual for maturity condition at puberty."""
    s_M = l_j / l_b
    l_i = s_M * (f - l_T)
    l_d = l_i - l_j
    tau_j = tau_b + np.log(s_M) * 3.0 / rho_j
    tau_p = tau_j + np.log((l_i - l_j) / (l_i - l_p)) / rho_B

    b3 = f / (f + g)
    b2 = f * s_M - b3 * l_i
    a0 = -(b2 + b3 * l_i) * l_i**2 / k
    a1 = -(2.0 * b2 + 3.0 * b3 * l_i) * l_i * l_d / (rho_B - k)
    a2 = (b2 + 3.0 * b3 * l_i) * l_d**2 / (2.0 * rho_B - k)
    a3 = -b3 * l_d**3 / (3.0 * rho_B - k)
    sum_a = a0 + a1 + a2 + a3
    sum_ae = (
        a0
        + a1 * np.exp(-rho_B * tau_p)
        + a2 * np.exp(-2.0 * rho_B * tau_p)
        + a3 * np.exp(-3.0 * rho_B * tau_p)
    )

    fn = v_Hp - (v_Hj + sum_a) * np.exp(-k * tau_p) + sum_ae
    return fn


def _get_lp_root(
    v_Hp: float,
    l_j: float,
    v_Hj: float,
    l_b: float,
    v_Hb: float,
    tau_b: float,
    l_T: float,
    rho_j: float,
    rho_B: float,
    k: float,
    g: float,
    f: float,
) -> float:
    """Root of puberty maturity condition."""
    s_M = l_j / l_b
    l_i = s_M * (f - l_T)
    try:
        return brentq(
            _get_lp_residual,
            l_j + 1e-9,
            l_i - 1e-9,
            args=(v_Hp, l_j, v_Hj, l_b, v_Hb, tau_b, l_T, rho_j, rho_B, k, g, f),
        )
    except ValueError:
        raise ValueError("lp root-finding failed")
