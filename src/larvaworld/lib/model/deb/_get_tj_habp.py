r"""
Port of DEBtool_M/animal/get_tj_habp.m — scaled ages at emergence for holometabolous insects.

Chains the larval acceleration (get_tj) with pupal ODE integration to find emergence.

See get_tj_habp.m source at:
  G:\Το Drive μου\DEB projects\Drosophila DEB model\code\Drosophilla_DEB_Evridiki\get_tj_habp.m

Citation:
  Bas Kooijman et al. "DEBtool_M" (MATLAB DEB model library)
  https://www.bio.vu.nl/thb/deb/deblab/
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from ._get_tj import get_tj


def get_tj_habp(
    p: tuple[float, ...],
    f: float = 1.0,
) -> tuple[float, float, float, float, float, float, float, float, int]:
    """
    Scaled ages and lengths at pupation and emergence for holometabolous abp-model.

    Ported from DEBtool_M/animal/get_tj_habp.m, constant-food branch.

    Parameters
    ----------
    p : 9-tuple
        (g, k, l_T, v_Hb, v_Hp, v_Hp_dummy, v_He, kap, kap_V)
        - g: energy investment ratio
        - k: k_J / k_M
        - l_T: scaled heating length
        - v_Hb: scaled maturity at birth
        - v_Hp: scaled maturity at pupation (larva->pupa)
        - v_Hp_dummy: ignored (for API compatibility, typically v_Hp + 1e-8)
        - v_He: scaled maturity at emergence (pupa->imago)
        - kap: allocation to soma in pupa
        - kap_V: conversion efficiency E->V->E at pupation
    f : float
        Scaled functional response (default: 1)

    Returns
    -------
    (tau_p, tau_e, tau_b, l_p, l_e, l_b, rho_j, u_Ee, info)
        - tau_p: scaled age at pupation
        - tau_e: scaled age at emergence
        - tau_b: scaled age at birth
        - l_p: scaled length at pupation (end of larval acceleration)
        - l_e: scaled length at emergence
        - l_b: scaled length at birth
        - rho_j: scaled exponential growth rate (larval)
        - u_Ee: scaled reserve at emergence
        - info: 1 on success, 0 on failure

    Notes
    -----
    This is the abp-model (for the Drosophila family where v_Hj = v_Hp).
    Multiply ages by k_M and lengths by L_m for unscaled values.
    """
    g, k, l_T, v_Hb, v_Hp, v_Hp_dummy, v_He, kap, kap_V = p

    # Get larval growth trajectory (birth to pupation)
    pars_tj = (g, k, l_T, v_Hb, v_Hp, v_Hp)  # v_Hj = v_Hp for abp
    tau_j, tau_p, tau_b, l_j, l_p, l_b, l_i, rho_j, rho_B, info_lj = get_tj(pars_tj, f)

    if info_lj == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0

    # Pupal ODE integration (pupation to emergence)
    s_M = l_j / l_b  # acceleration factor, stays constant in pupa
    u_Ej = l_j**3 * (kap * kap_V + f / g)  # scaled reserve at pupation

    def dget_tj_hex(tau: float, luEvH: np.ndarray) -> list[float]:
        """Pupal state derivatives (l, u_E, v_H)."""
        l, u_E, v_H = luEvH
        l2, l3, l4 = l * l, l * l * l, l * l * l * l
        u_E = max(1e-6, u_E)  # clamp to avoid singularities

        dl = (g * s_M * u_E - l4) / (u_E + l3) / 3.0
        du_E = -u_E * l2 * (g * s_M + l) / (u_E + l3)
        dv_H = -du_E - k * v_H

        return [dl, du_E, dv_H]

    def emergence_event(tau: float, luEvH: np.ndarray) -> float:
        """Terminal event: v_H reaches v_He."""
        return v_He - luEvH[2]

    emergence_event.terminal = True
    emergence_event.direction = 1

    # Integrate from pupation to emergence
    try:
        sol = solve_ivp(
            dget_tj_hex,
            t_span=(0, 300),
            y0=[0, u_Ej, 0],
            events=emergence_event,
            method="RK45",
            rtol=1e-8,
            atol=1e-12,
        )

        if not sol.t_events[0].size:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0

        tau_e_rel = sol.t_events[0][0]  # relative to pupation
        l_e = sol.y_events[0][0, 0]
        u_Ee = sol.y_events[0][0, 1]

        tau_e = tau_p + tau_e_rel  # absolute scaled age at emergence

        if u_Ee < 0 or not np.isfinite(tau_e) or not np.isfinite(l_e):
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0

        return tau_p, tau_e, tau_b, l_j, l_e, l_b, rho_j, u_Ee, 1

    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0
