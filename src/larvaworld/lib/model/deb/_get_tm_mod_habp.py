r"""
Port of DEBtool_M/animal/get_tm_mod_habp.m — mean lifespan for holometabolous insects.

Integrates the aging/survival ODE from emergence to death, computing mean lifespan.

See get_tm_mod_habp.m source at:
  G:\Το Drive μου\DEB projects\Drosophila DEB model\code\Drosophilla_DEB_Evridiki\get_tm_mod_habp.m

Citation:
  Bas Kooijman et al. "DEBtool_M" (MATLAB DEB model library)
  https://www.bio.vu.nl/thb/deb/deblab/
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from ._get_tj_habp import get_tj_habp


def get_tm_mod_habp(
    p: tuple[float, ...], f: float = 1.0
) -> tuple[float, float, float, int]:
    """
    Mean lifespan (as imago) for holometabolous abp-model with aging.

    Ported from DEBtool_M/animal/get_tm_mod_habp.m, 'hax' branch only.

    Parameters
    ----------
    p : 10-tuple
        (g, k, v_Hb, v_Hp, v_Hp_dummy, v_He, kap, kap_V, h_a, s_G)
        - g, k, v_Hb, v_Hp, v_He, kap, kap_V: as in get_tj_habp
        - h_a: Weibull aging acceleration (1/d^2)
        - s_G: Gompertz stress coefficient (-)
    f : float
        Scaled functional response (default: 1)

    Returns
    -------
    (tau_m, tau_e, tau_p, info)
        - tau_m: scaled mean lifespan as imago (relative to emergence)
        - tau_e: scaled age at emergence (absolute)
        - tau_p: scaled age at pupation (absolute)
        - info: 1 on success, 0 on failure

    Notes
    -----
    Divide tau_m by k_M and apply temperature correction to get unscaled lifespan in days.
    This is the imago aging, separate from pre-emergence (which this model ignores).
    """
    g, k, v_Hb, v_Hp, v_Hp_dummy, v_He, kap, kap_V, h_a, s_G = p

    # Get emergence time from pupal ODE
    pars_tj = (g, k, 0, v_Hb, v_Hp, v_Hp_dummy, v_He, kap, kap_V)
    tau_p, tau_e, tau_b, l_j, l_e, l_b, rho_j, u_Ee, info_habp = get_tj_habp(pars_tj, f)

    if info_habp == 0:
        return np.nan, np.nan, np.nan, 0

    s_M = l_j / l_b  # acceleration factor (constant in imago)

    def dget_qhSt_hex_ji(tau: float, qhSt: np.ndarray) -> list[float]:
        """
        Aging + survival ODE from emergence (imago).

        State: [q, h_A, S, t]
          - q: scaled aging acceleration
          - h_A: scaled hazard rate due to aging
          - S: survival probability
          - t: cumulative survival-weighted time (for mean lifespan)
        """
        q, h_A, S, t = qhSt
        S = max(0, S)  # clamp to [0, 1]

        if tau < tau_e:
            # Still in pupa (shouldn't reach here, but guard it)
            dq = 0
            dh_A = 0
        else:
            # Imago: aging proceeds
            dq = f * (q * l_e**3 * s_G + h_a) * g * s_M / l_e
            dh_A = q

        h = h_A  # total hazard (aging only, no background hazard modeled here)
        dS = -h * S
        dt = S

        return [dq, dh_A, dS, dt]

    def dead_for_sure(tau: float, qhSt: np.ndarray) -> float:
        """Terminal event: S (survival) drops near zero."""
        return qhSt[2] - 1e-6

    dead_for_sure.terminal = True
    dead_for_sure.direction = -1

    # Integrate from emergence to death
    try:
        sol = solve_ivp(
            dget_qhSt_hex_ji,
            t_span=(tau_e, tau_e + 1000),  # span from emergence forward
            y0=[0, 0, 1, 0],  # [q, h_A, S=1 at emergence, t=0]
            events=dead_for_sure,
            method="RK45",
            rtol=1e-7,
            atol=1e-10,
            dense_output=False,
        )

        if not sol.t_events[0].size:
            # Didn't die (unrealistic), use last time
            tau_m_abs = sol.t[-1]
            S_final = sol.y[2, -1]
            if S_final < 1e-7:
                tau_m = tau_m_abs - tau_e  # relative to emergence
            else:
                return np.nan, tau_e, tau_p, 0
        else:
            # Died at event
            tau_death = sol.t_events[0][0]
            tau_m = tau_death - tau_e  # relative to emergence, i.e., imago lifespan

        if not np.isfinite(tau_m) or tau_m <= 0:
            return np.nan, tau_e, tau_p, 0

        return tau_m, tau_e, tau_p, 1

    except Exception:
        return np.nan, tau_e, tau_p, 0
