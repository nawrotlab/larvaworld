"""
Ground-truth DEB equations for the holometabolous insect (Drosophila) model.

This module is a verbatim transcription of the authoritative model specification:

* **Table S1** -- fluxes for the metabolic processes (per life stage)
* **Table S2** -- state variables and model dynamics

Both tables are reproduced in full in the section headers below, so the code can
be checked against the specification without leaving the file.

The module is deliberately **self-contained**: it imports only ``numpy`` at module
scope (``scipy`` is imported lazily inside the closed-form solver) and has no
dependency on the rest of ``larvaworld``. It can be imported, run and tested in
isolation, which is what makes it usable as the reference implementation.

Note on duplication: :func:`beta0` and :func:`get_lb` also exist in
``larvaworld.lib.util.fitting`` and ``larvaworld.lib.model.deb.deb`` respectively.
They are re-implemented here to keep this module standalone; de-duplication is a
decision for the integration step, not for the reference implementation.

Structure
---------
1. Special functions and embryo solvers  (``beta0``, ``get_lb``, ``get_ue0``)
2. Parameters                            (``DEBPars``)
3. Fluxes -- Table S1                    (``powers``)
4. Dynamics -- Table S2                  (``derivatives``)
5. Stage machine                         (``Stage``, ``DEBState``, ``transition``)
6. Integration engines                   (``step``, ``run``)

Units follow the AmP/DEBtool convention: energies in J, lengths in cm, volumes in
cm^3, time in days.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, NamedTuple, Optional, Sequence

import json
import math

import numpy as np
from numpy.lib import scimath

__all__: list[str] = [
    "STAGES",
    "Stage",
    "Powers",
    "DEBPars",
    "DEBState",
    "Trajectory",
    "beta0",
    "get_lb",
    "get_ue0",
    "acceleration",
    "temperature_correction",
    "powers",
    "derivatives",
    "transition",
    "initial_state",
    "step",
    "run",
]


# ---------------------------------------------------------------------------
# 1. Special functions and embryo solvers
# ---------------------------------------------------------------------------


def beta0(x0: Any, x1: Any) -> Any:
    """
    Incomplete beta function ``B_x1(4/3, 0) - B_x0(4/3, 0)``.

    This is the integral ``int t^(1/3) (1-t)^-1 dt`` evaluated between ``x0`` and
    ``x1``; see Kooijman (2010), DEB textbook p. 58. Ported from
    ``DEBtool_M/lib/misc/beta0.m``.

    ``scimath.log`` is used so that the ``x > 1`` branch stays finite; the real
    part is returned because the imaginary contributions cancel.
    """
    x03 = np.asarray(x0, dtype=float) ** (1.0 / 3.0)
    x13 = np.asarray(x1, dtype=float) ** (1.0 / 3.0)
    a3 = math.sqrt(3.0)

    def _f(y: Any) -> Any:
        return (
            -3.0 * y
            + a3 * np.arctan((1.0 + 2.0 * y) / a3)
            - scimath.log(y - 1.0)
            + scimath.log(1.0 + y + y**2) / 2.0
        )

    return np.real(_f(x13) - _f(x03))


def get_lb(
    g: float,
    k: float,
    v_Hb: float,
    eb: float = 1.0,
    n_iter: int = 100,
    tol: float = 1e-18,
) -> tuple[float, int]:
    """
    Scaled structural length at birth ``l_b``. Ported from
    ``DEBtool_M/animal/get_lb.m``.

    A Newton-Raphson scheme with Euler quadrature is tried first. Following
    ``get_lb.m:105``, the result is rejected unless it lands in ``(0, 1)``, and a
    bracketing bisection on the same residual is used as the fallback (DEBtool
    delegates to the ``get_lb2`` shooting method at this point; bisecting the
    already-vectorised residual solves the identical equation far more cheaply).

    The fallback matters for parameter sets with a large ``v_Hb``: the standard
    initial guess ``l_b = v_Hb^(1/3)`` is the exact solution only for ``k = 1``,
    and for ``v_Hb > 1`` it starts outside the physical domain, from where Newton
    converges to a spurious negative root.

    Parameters
    ----------
    g : energy investment ratio
    k : maintenance ratio ``k_J / k_M``
    v_Hb : scaled maturity at birth
    eb : scaled reserve density at birth (1 = fully provisioned egg)

    Returns
    -------
    (lb, info) : ``info`` is 1 on convergence to a value in ``(0, 1)``, else 0.
    """
    if k == 1.0:
        return float(v_Hb ** (1.0 / 3.0)), 1  # exact solution for k = 1

    xb = g / (eb + g)
    n = int(1000 + round(1000 * max(0.0, k - 1.0)))
    xb3 = xb ** (1.0 / 3.0)
    x = np.linspace(1e-5, xb, n)
    dx = xb / n
    x3 = x ** (1.0 / 3.0)

    b = beta0(x, xb) / (3.0 * g)
    t0 = xb * g * v_Hb

    # l(x) = x^(1/3) / (xb3/lb - b) stays positive only while xb3/lb > max(b),
    # which bounds the physical domain of lb from above.
    lb_max = min(1.0, xb3 / float(np.max(b))) if np.max(b) > 0.0 else 1.0

    def residual(lb: float) -> float:
        """Maturity-condition residual; its root in (0, lb_max) is l_b."""
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            l = x3 / (xb3 / lb - b)
            s = (k - x) / (1.0 - x) * l / g / x
            vv = np.exp(-dx * np.cumsum(s))
            r = g + l
            return float(t0 / lb**3 / vv[-1] - dx * np.sum(r / vv))

    # --- Newton-Raphson ---------------------------------------------------
    lb = v_Hb ** (1.0 / 3.0)
    if not 0.0 < lb < lb_max:
        lb = 0.5 * lb_max
    norm = 1.0
    i = 0
    while i < n_iter and norm > tol:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            l = x3 / (xb3 / lb - b)
            s = (k - x) / (1.0 - x) * l / g / x
            vv = np.exp(-dx * np.cumsum(s))
            vb = vv[-1]
            r = g + l
            rv = r / vv
            t = t0 / lb**3 / vb - dx * np.sum(rv)
            dl = xb3 / lb**2 * l**2.0 / x3
            dlnv = np.exp(-dx * np.cumsum(s * dl / l))
            dlnvb = dlnv[-1]
            dt = -t0 / lb**3 / vb * (3.0 / lb + dlnvb) - dx * np.sum(
                (dl / r - dlnv) * rv
            )
            lb -= t / dt
            norm = t**2
        i += 1
        if not np.isfinite(lb):
            break

    if np.isfinite(lb) and np.isfinite(norm) and norm <= tol and 0.0 < lb < 1.0:
        return float(lb), 1

    # --- fallback: bracket + bisect --------------------------------------
    hi = lb_max * (1.0 - 1e-9)
    grid = np.linspace(1e-10, hi, 400)
    vals = np.array([residual(float(v)) for v in grid])
    ok = np.isfinite(vals)
    sign_change = np.where(
        ok[:-1] & ok[1:] & (np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)
    )[0]
    if sign_change.size == 0:
        return float("nan"), 0

    a, c = float(grid[sign_change[0]]), float(grid[sign_change[0] + 1])
    fa = residual(a)
    for _ in range(200):
        mid = 0.5 * (a + c)
        fm = residual(mid)
        if not np.isfinite(fm):
            return float("nan"), 0
        if abs(c - a) < 1e-14 or fm == 0.0:
            break
        if fa * fm < 0.0:
            c = mid
        else:
            a, fa = mid, fm
    lb = 0.5 * (a + c)
    return float(lb), int(0.0 < lb < 1.0)


def get_ue0(g: float, lb: float, eb: float = 1.0) -> float:
    """
    Scaled initial reserve ``u_E0`` of the egg. Ported from
    ``DEBtool_M/animal/get_ue0.m``.
    """
    xb = g / (eb + g)
    return float((3.0 * g / (3.0 * g * xb ** (1.0 / 3.0) / lb - beta0(0.0, xb))) ** 3)


# ---------------------------------------------------------------------------
# 2. Parameters
# ---------------------------------------------------------------------------

#: Ordered life stages of the holometabolous model.
STAGES: tuple[str, ...] = ("egg", "larva", "pupa", "imago")


class Stage:
    """Life-stage name constants (kept as plain strings for serialisability)."""

    EGG = "egg"
    LARVA = "larva"
    PUPA = "pupa"
    IMAGO = "imago"
    DEAD = "dead"


@dataclass
class DEBPars:
    """
    Primary DEB parameters (AmP symbol names) plus the compound parameters
    derived from them.

    Primary values default to the AmP ``abp`` fit for *Drosophila melanogaster*;
    :meth:`from_amp_json` loads a parameter export directly. Compound parameters
    follow ``DEBtool_M/lib/pet/parscomp_st.m`` and ``addchem.m`` and are computed
    once in ``__post_init__``. Use :meth:`with_` (``dataclasses.replace``) to vary
    a primary parameter -- it re-runs the whole derivation.
    """

    # --- primary: temperature ------------------------------------------------
    T_ref: float = 293.1  # K, reference temperature
    T_A: float = 28890.0  # K, Arrhenius temperature

    # --- primary: core -------------------------------------------------------
    z: float = 0.05054  # -, zoom factor
    v: float = 0.03568  # cm/d, energy conductance
    kap: float = 0.8024  # -, allocation fraction to soma
    p_M: float = 8674.0  # J/d.cm^3, [p_M] volume-specific somatic maintenance
    p_T: float = 0.0  # J/d.cm^2, {p_T}; absent from Table S1, kept for provenance
    k_J: float = 0.002  # 1/d, maturity maintenance rate coefficient
    E_G: float = 4434.0  # J/cm^3, [E_G] specific cost for structure

    # --- primary: maturity thresholds ---------------------------------------
    E_Hb: float = 0.1853  # J, maturity at birth (egg -> larva)
    E_Hp: float = 285.5  # J, maturity at pupation (larva -> pupa)
    E_He: float = 0.6475  # J, maturity at emergence (pupa -> imago)

    # --- primary: efficiencies ----------------------------------------------
    kap_R: float = 0.95  # -, reproduction efficiency
    kap_X: float = 0.8  # -, digestion efficiency of food to reserve
    kap_P: float = 0.1  # -, faecation efficiency of food to faeces
    kap_V: float = 0.99148  # -, conversion efficiency E -> V -> E at pupation

    # --- primary: feeding and shape -----------------------------------------
    F_m: float = 6.5  # l/d.cm^2, {F_m} max specific searching rate
    f: float = 1.0  # -, scaled functional response
    del_M: float = 0.629  # -, shape coefficient, larval head capsule
    del_Mw: float = 0.3808  # -, shape coefficient, imago wing length

    # --- primary: aging and instars (carried, not used by Table S1) ---------
    h_a: float = 2.653e-08  # 1/d^2, Weibull aging acceleration
    s_G: float = 1e-04  # -, Gompertz stress coefficient
    s_1: float = 4.526  # -, stress at instar 1
    s_2: float = 3.303  # -, stress at instar 2

    # --- chemistry: potentials, densities, molecular weights ----------------
    mu_X: float = 525000.0  # J/mol, chemical potential of food
    mu_V: float = 500000.0  # J/mol, chemical potential of structure
    mu_E: float = 550000.0  # J/mol, chemical potential of reserve
    mu_P: float = 480000.0  # J/mol, chemical potential of faeces
    d_V: float = 0.17  # g/cm^3, specific density of structure (Insecta)

    #: Chemical indices for water-free organics (Kooijman 2010, Fig 4.15),
    #: columns X, V, E, P; rows C, H, O, N.
    n_O: tuple[tuple[float, ...], ...] = (
        (1.00, 1.00, 1.00, 1.00),
        (1.80, 1.80, 1.80, 1.80),
        (0.50, 0.50, 0.50, 0.50),
        (0.15, 0.15, 0.15, 0.15),
    )

    #: Structural volume seed used for the egg and for the pupa after the
    #: metamorphic reset. Small but non-zero so ``V**(1/3)`` stays finite.
    V_seed: float = 1e-30

    # --- derived (populated by __post_init__) --------------------------------
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()
        self._derive()

    # -- validation ----------------------------------------------------------

    def _validate(self) -> None:
        if not 0.0 < self.kap < 1.0:
            raise ValueError(f"kap must lie in (0, 1); got {self.kap!r}")
        if not 0.0 < self.kap_V <= 1.0:
            # The AmP abp export for D. melanogaster reports kap_V = -1.526e-54,
            # i.e. the fit collapsed this parameter onto its lower bound. MATLAB
            # predict_Drosophila_melanogaster.m rejects the parameter set outright
            # when kap_V < 0, so we do too rather than silently clamping.
            raise ValueError(
                f"kap_V must lie in (0, 1]; got {self.kap_V!r}. An AmP export with a "
                "degenerate (<= 0) kap_V has collapsed this parameter and must be "
                "overridden explicitly."
            )
        for name in ("kap_X", "kap_P", "kap_R"):
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]; got {val!r}")
        for name in ("z", "v", "p_M", "E_G", "k_J", "E_Hb", "E_Hp", "E_He", "T_ref"):
            val = getattr(self, name)
            if not val > 0.0:
                raise ValueError(f"{name} must be strictly positive; got {val!r}")
        if not self.E_Hb < self.E_Hp:
            raise ValueError(
                f"E_Hb ({self.E_Hb}) must be below the pupation threshold "
                f"E_Hp ({self.E_Hp})"
            )
        # E_He is NOT required to exceed E_Hp: maturity is reset to zero at
        # pupation, so the pupa matures from 0 up to E_He on a fresh counter.

    # -- compound parameters (parscomp_st.m + addchem.m) ----------------------

    def _derive(self) -> None:
        # Molecular weights from the chemical indices (g/mol).
        atomic = np.array([12.0, 1.0, 16.0, 14.0])
        n_O = np.asarray(self.n_O, dtype=float)
        self.w_X, self.w_V, self.w_E, self.w_P = (atomic @ n_O).tolist()

        # Core compound parameters. L_m_ref = 1 cm keeps {p_Am} dimensionally
        # consistent, exactly as in parscomp_st.m.
        self.p_Am = self.z * self.p_M / self.kap  # J/d.cm^2
        self.E_m = self.p_Am / self.v  # J/cm^3, [E_m] reserve capacity
        self.g = self.E_G / (self.kap * self.E_m)  # -, investment ratio
        self.k_M = self.p_M / self.E_G  # 1/d, somatic maint rate coeff
        self.k = self.k_J / self.k_M  # -, maintenance ratio
        self.L_m = self.v / (self.k_M * self.g)  # cm, maximum structural length
        self.L_T = self.p_T / self.p_M  # cm, heating length
        self.l_T = self.L_T / self.L_m  # -, scaled heating length

        # Scaling factor that converts scaled to unscaled reserve.
        self.U_coeff = self.g**2 * self.k_M**3 / self.v**2

        # Scaled maturity levels.
        for suffix, E_H in (("b", self.E_Hb), ("p", self.E_Hp), ("e", self.E_He)):
            U_H = E_H / self.p_Am
            V_H = U_H / (1.0 - self.kap)
            setattr(self, f"M_H{suffix}", E_H / self.mu_E)
            setattr(self, f"U_H{suffix}", U_H)
            setattr(self, f"V_H{suffix}", V_H)
            setattr(self, f"v_H{suffix}", V_H * self.U_coeff)
            setattr(self, f"u_H{suffix}", U_H * self.U_coeff)

        # Mass/energy couplers.
        self.M_V = self.d_V / self.w_V  # mol/cm^3
        self.y_V_E = self.mu_E * self.M_V / self.E_G  # mol/mol
        self.y_E_V = 1.0 / self.y_V_E
        self.m_Em = self.y_E_V * self.E_m / self.E_G
        self.kap_G = self.mu_V * self.M_V / self.E_G  # -, growth efficiency
        self.E_V = self.d_V * self.mu_V / self.w_V  # J/cm^3

        # Assimilation / feeding couplers.
        self.y_E_X = self.kap_X * self.mu_X / self.mu_E
        self.y_X_E = 1.0 / self.y_E_X
        self.y_P_X = self.kap_P * self.mu_X / self.mu_P
        self.y_X_P = 1.0 / self.y_P_X
        self.p_Xm = self.p_Am / self.kap_X
        self.J_E_Am = self.p_Am / self.mu_E
        self.J_X_Am = self.y_X_E * self.J_E_Am
        self.K = self.J_X_Am / self.F_m  # half-saturation coefficient

        # Embryo solution: scaled length at birth and initial reserve.
        self.l_b, self.lb_info = get_lb(g=self.g, k=self.k, v_Hb=self.v_Hb)
        self.L_b_pred = self.l_b * self.L_m  # cm, structural length at birth
        self.u_E0 = get_ue0(g=self.g, lb=self.l_b)
        self.E_0 = self.p_Am * self.u_E0 / self.U_coeff  # J, initial reserve

    # -- constructors --------------------------------------------------------

    def with_(self, **overrides: Any) -> "DEBPars":
        """Return a copy with primary parameters overridden and compounds rederived."""
        return replace(self, **overrides)

    @classmethod
    def from_amp_json(
        cls, path: str, overrides: Optional[dict[str, Any]] = None
    ) -> "DEBPars":
        """
        Build from an AmP parameter export (as produced by the AmP results page).

        Only the ``parameters`` block is read. The ``data_predictions`` block is
        deliberately **ignored** for parameterisation: its ``Lb``/``L1``/``L2``/``Lj``
        entries are *physical* lengths (``Lw = L / del_M``), and using them where a
        *structural* length is required -- as ``s_M = L / L_b`` does -- introduces a
        systematic ``1 / del_M`` error. Use :func:`amp_predictions` to read that
        block separately for validation.
        """
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        fields = {f for f in cls.__dataclass_fields__ if f != "metadata"}
        values: dict[str, Any] = {
            e["symbol"]: e["value"]
            for e in payload.get("parameters", [])
            if e.get("symbol") in fields
        }
        if overrides:
            values.update(overrides)
        values["metadata"] = dict(payload.get("metadata", {}))
        return cls(**values)


def amp_predictions(path: str) -> dict[str, float]:
    """Read the ``data_predictions`` block of an AmP export, for validation only."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return {
        e["symbol"]: e["prd"]
        for e in payload.get("data_predictions", [])
        if e.get("prd") is not None
    }


# ---------------------------------------------------------------------------
# 3. Fluxes -- Table S1
# ---------------------------------------------------------------------------
#
#   Table S1: Fluxes for the metabolic processes.
#
#   Stage  Metabolic Process       Flux
#   -----------------------------------------------------------------------
#          Assimilation            p_A = 0
#          Mobilization            p_C = E ([E_G] v V^(2/3) + [p_M] V)
#                                        / (kap E + [E_G] V)
#   egg    Somatic Maintenance     p_S = [p_M] V
#          Growth                  p_G = kap p_C - p_S
#          Maturity Maintenance    p_J = k_J E_H
#          Maturation              p_R = (1 - kap) p_C - p_J
#   -----------------------------------------------------------------------
#          Assimilation            p_A = {p_Am} s_M f V^(2/3)
#          Mobilization            p_C = E ([E_G] v s_M V^(2/3) + [p_M] V)
#                                        / (kap E + [E_G] V)
#   larva  Somatic Maintenance     p_S = [p_M] V
#          Growth                  p_G = kap p_C - p_S
#          Maturity Maintenance    p_J = k_J E_H
#          Maturation              p_R = (1 - kap) p_C - p_J
#   -----------------------------------------------------------------------
#          Assimilation            p_A = 0
#          Mobilization            p_C = E ([E_G] v s_M V^(2/3) + [p_M] V)
#                                        / (kap E + [E_G] V)
#   pupa   Somatic Maintenance     p_S = [p_M] V
#          Growth                  p_G = kap p_C - p_S
#          Maturity Maintenance    p_J = k_J E_H
#          Maturation              p_R = (1 - kap) p_C - p_J
#   -----------------------------------------------------------------------
#          Assimilation            p_A = {p_Am} s_M f V^(2/3)
#          Mobilization            p_C = E v s_M V^(-1/3)
#   imago  Somatic Maintenance     p_S = [p_M] V
#          Growth                  p_G = 0
#          Maturity Maintenance    p_J = k_J E_H
#          Reproduction            p_R = p_C - p_S - p_J
#   -----------------------------------------------------------------------
#
#   Note: The acceleration factor s_M = max(1, min(L, L_p) / L_b), where L_b and
#   L_p are the structural lengths at hatch and pupation.
#
# {p_T} does not appear in any row of Table S1 and is therefore not applied.
# ---------------------------------------------------------------------------


class Powers(NamedTuple):
    """The six metabolic fluxes of Table S1, all in J/d."""

    p_A: float  # assimilation
    p_C: float  # mobilization
    p_S: float  # somatic maintenance
    p_G: float  # growth
    p_J: float  # maturity maintenance
    p_R: float  # maturation (non-imago) / reproduction (imago)


def acceleration(L: float, L_b: float, L_p: float) -> float:
    """
    Metabolic acceleration factor ``s_M = max(1, min(L, L_p) / L_b)``.

    ``L_b`` and ``L_p`` are the structural lengths at hatch and at pupation. Before
    those events occur they are ``inf``, which makes the expression collapse to
    ``s_M = 1`` before birth and to ``s_M = L / L_b`` between birth and pupation --
    the V1-morphic acceleration of the abj/abp family.
    """
    if not np.isfinite(L_b):
        return 1.0
    return max(1.0, min(L, L_p) / L_b)


def temperature_correction(pars: DEBPars, T: Optional[float] = None) -> float:
    """One-parameter Arrhenius correction factor ``exp(T_A/T_ref - T_A/T)``."""
    if T is None:
        return 1.0
    return float(np.exp(pars.T_A / pars.T_ref - pars.T_A / T))


def powers(
    stage: str,
    E: float,
    V: float,
    E_H: float,
    pars: DEBPars,
    f: Optional[float] = None,
    s_M: float = 1.0,
    TC: float = 1.0,
    p_A: Optional[float] = None,
) -> Powers:
    """
    Evaluate Table S1 for one stage at the given state.

    Parameters
    ----------
    stage : one of ``STAGES``
    E, V, E_H : reserve (J), structural volume (cm^3), maturity (J)
    pars : parameter set
    f : scaled functional response; defaults to ``pars.f``
    s_M : acceleration factor, from :func:`acceleration`
    TC : Arrhenius temperature correction, from :func:`temperature_correction`
    p_A : optional externally supplied assimilation flux (J/d). When given it
        overrides the Table S1 expression -- this is the hook through which a
        gut model or a behavioural simulation drives assimilation.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    if f is None:
        f = pars.f

    # Temperature-corrected rates.
    p_Am_T = TC * pars.p_Am
    p_M_T = TC * pars.p_M
    v_T = TC * pars.v
    k_J_T = TC * pars.k_J

    E_G = pars.E_G
    kap = pars.kap

    V = max(V, 0.0)
    V23 = V ** (2.0 / 3.0)

    # -- Assimilation -----------------------------------------------------
    if p_A is None:
        # Egg and pupa do not feed.
        p_A = 0.0 if stage in (Stage.EGG, Stage.PUPA) else p_Am_T * s_M * f * V23

    # -- Mobilization -----------------------------------------------------
    if stage == Stage.IMAGO:
        # p_C = E v s_M V^(-1/3); the imago does not grow, so the general
        # mobilization expression collapses to pure reserve turnover.
        p_C = E * v_T * s_M / V ** (1.0 / 3.0) if V > 0.0 else 0.0
    else:
        # The egg row of Table S1 carries no s_M; acceleration() already returns
        # 1.0 before birth, so the same expression covers all three stages.
        s = 1.0 if stage == Stage.EGG else s_M
        denom = kap * E + E_G * V
        p_C = E * (E_G * v_T * s * V23 + p_M_T * V) / denom if denom > 0.0 else 0.0

    # -- Somatic maintenance ----------------------------------------------
    p_S = p_M_T * V

    # -- Growth ------------------------------------------------------------
    p_G = 0.0 if stage == Stage.IMAGO else kap * p_C - p_S

    # -- Maturity maintenance ---------------------------------------------
    p_J = k_J_T * E_H

    # -- Maturation / reproduction ----------------------------------------
    if stage == Stage.IMAGO:
        p_R = p_C - p_S - p_J
    else:
        p_R = (1.0 - kap) * p_C - p_J

    return Powers(p_A=p_A, p_C=p_C, p_S=p_S, p_G=p_G, p_J=p_J, p_R=p_R)


# ---------------------------------------------------------------------------
# 4. Dynamics -- Table S2
# ---------------------------------------------------------------------------
#
#   Table S2: State variables and model dynamics for the DEB model.
#
#   State variable                     Dynamics
#   -----------------------------------------------------------------------
#   Energy reserves, E                 dE/dt   = p_A - p_C
#   Structural volume, V               dV/dt   = p_G / [E_G]
#   Energy used for maturation, E_H    dE_H/dt = 0     if imago, else p_R
#   Energy used for reproduction, E_R  dE_R/dt = p_R   if imago, else 0
#   -----------------------------------------------------------------------
# ---------------------------------------------------------------------------


def derivatives(
    stage: str, p: Powers, pars: DEBPars
) -> tuple[float, float, float, float]:
    """
    Evaluate Table S2 -- returns ``(dE, dV, dE_H, dE_R)`` in J/d, cm^3/d, J/d, J/d.
    """
    dE = p.p_A - p.p_C
    dV = p.p_G / pars.E_G
    if stage == Stage.IMAGO:
        dE_H = 0.0
        dE_R = p.p_R
    else:
        dE_H = p.p_R
        dE_R = 0.0
    return dE, dV, dE_H, dE_R


# ---------------------------------------------------------------------------
# 5. Stage machine
# ---------------------------------------------------------------------------


@dataclass
class DEBState:
    """
    Mutable state of one individual.

    ``L_b`` and ``L_p`` are recorded at the birth and pupation events and feed the
    acceleration factor; they are ``inf`` until the corresponding event occurs.
    """

    E: float  # J, reserve
    V: float  # cm^3, structural volume
    E_H: float = 0.0  # J, maturity
    E_R: float = 0.0  # J, reproduction buffer
    age: float = 0.0  # d, since oviposition
    stage: str = Stage.EGG
    L_b: float = math.inf  # cm, structural length at hatch
    L_p: float = math.inf  # cm, structural length at pupation

    @property
    def L(self) -> float:
        """Structural length (cm)."""
        return self.V ** (1.0 / 3.0)

    @property
    def alive(self) -> bool:
        return self.E > 0.0

    def s_M(self) -> float:
        return acceleration(self.L, self.L_b, self.L_p)

    def copy(self) -> "DEBState":
        return replace(self)


def initial_state(pars: DEBPars) -> DEBState:
    """Freshly laid egg: all reserve, negligible structure, zero maturity."""
    return DEBState(
        E=pars.E_0, V=pars.V_seed, E_H=0.0, E_R=0.0, age=0.0, stage=Stage.EGG
    )


def transition(state: DEBState, pars: DEBPars) -> bool:
    """
    Apply any stage transition that the current maturity level implies.

    Thresholds, in order: ``E_Hb`` (egg -> larva), ``E_Hp`` (larva -> pupa),
    ``E_He`` (pupa -> imago).

    At **pupation** the maturity counter is reset to zero and the larval structure
    is resorbed into the pupal reserve. This reset is what makes the AmP parameter
    ordering ``E_He < E_Hp`` consistent: the pupa matures from 0 up to ``E_He`` on a
    fresh counter, exactly as in ``DEBtool``'s ``get_tj_habp.m``, whose pupal
    integration starts from ``[l = 0, u_E = u_Ej, v_H = 0]``.

    The resorption rule is the unscaled form of that same initial condition. With
    ``u_Ej = l_j^3 (kap kap_V + f/g)`` and ``g E_m = [E_G]/kap``::

        E_pupa = E_larva + kap_V [E_G] V_larva

    i.e. the energy ``[E_G] V`` invested in larval structure returns to reserve with
    efficiency ``kap_V``. Using the live ``E`` rather than assuming ``e = f`` makes
    the rule exact for any reserve density and reduces to the DEBtool expression
    when ``e = f``.

    Returns
    -------
    bool : True if a transition was applied.
    """
    if state.stage == Stage.EGG and state.E_H >= pars.E_Hb:
        state.stage = Stage.LARVA
        state.L_b = state.L
        return True

    if state.stage == Stage.LARVA and state.E_H >= pars.E_Hp:
        state.L_p = state.L
        state.E = state.E + pars.kap_V * pars.E_G * state.V
        state.V = pars.V_seed
        state.E_H = 0.0
        state.stage = Stage.PUPA
        return True

    if state.stage == Stage.PUPA and state.E_H >= pars.E_He:
        state.stage = Stage.IMAGO
        return True

    return False


# ---------------------------------------------------------------------------
# 6. Integration engines
# ---------------------------------------------------------------------------
#
# Two interchangeable engines share the equations above:
#
#   "stepped"  fixed-dt explicit Euler, advanced one step at a time. This is the
#              engine a behavioural simulation drives, because the assimilation
#              flux is only known tick by tick (food is encountered, not
#              prescribed).
#   "closed"   scipy.integrate.solve_ivp with terminal events on the maturity
#              thresholds, integrating each stage as one segment. Accurate, but
#              requires f (or p_A) to be known as a function of time.
#
# Both accept the same stop criteria and work for every stage.
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """Recorded output of :func:`run`."""

    t: np.ndarray  # d, age
    E: np.ndarray  # J
    V: np.ndarray  # cm^3
    E_H: np.ndarray  # J
    E_R: np.ndarray  # J
    stage: list[str]
    events: dict[str, float] = field(default_factory=dict)  # stage -> age at entry

    @property
    def L(self) -> np.ndarray:
        """Structural length (cm)."""
        return self.V ** (1.0 / 3.0)

    def Lw(self, pars: DEBPars) -> np.ndarray:
        """Physical length (cm), ``L / del_M``."""
        return self.L / pars.del_M

    def e(self, pars: DEBPars) -> np.ndarray:
        """Scaled reserve density ``E / (V [E_m])``."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.V > 0, self.E / (self.V * pars.E_m), np.nan)

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "E": self.E,
            "V": self.V,
            "E_H": self.E_H,
            "E_R": self.E_R,
            "stage": self.stage,
            "events": dict(self.events),
        }


def _resolve_f(f: Any, t: float) -> float:
    return float(f(t)) if callable(f) else float(f)


def step(
    state: DEBState,
    pars: DEBPars,
    dt: float,
    f: Optional[float] = None,
    T: Optional[float] = None,
    p_A: Optional[float] = None,
) -> Powers:
    """
    Advance ``state`` by one explicit-Euler step of ``dt`` days, in place.

    ``p_A`` overrides the Table S1 assimilation flux, which is how a gut model or a
    behavioural simulation injects the food actually ingested during the tick.

    Returns the fluxes that were applied, so the caller can record them.
    """
    if not state.alive:
        return Powers(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    TC = temperature_correction(pars, T)
    p = powers(
        stage=state.stage,
        E=state.E,
        V=state.V,
        E_H=state.E_H,
        pars=pars,
        f=f,
        s_M=state.s_M(),
        TC=TC,
        p_A=p_A,
    )
    dE, dV, dE_H, dE_R = derivatives(state.stage, p, pars)

    state.E += dE * dt
    state.V = max(state.V + dV * dt, 0.0)
    state.E_H += dE_H * dt
    state.E_R += dE_R * dt
    state.age += dt

    transition(state, pars)
    return p


def _stop_reached(
    state: DEBState,
    until_age: Optional[float],
    until_stage: Optional[str],
    until_maturity: Optional[float],
) -> bool:
    if until_age is not None and state.age >= until_age:
        return True
    if until_stage is not None and state.stage == until_stage:
        return True
    if until_maturity is not None and state.E_H >= until_maturity:
        return True
    return False


def run(
    pars: DEBPars,
    state: Optional[DEBState] = None,
    engine: str = "stepped",
    dt: float = 1.0 / (24.0 * 60.0),
    f: Any = None,
    T: Optional[float] = None,
    until_age: Optional[float] = None,
    until_stage: Optional[str] = None,
    until_maturity: Optional[float] = None,
    max_steps: int = 10_000_000,
    record_every: int = 1,
) -> tuple[DEBState, Trajectory]:
    """
    Integrate the model with the selected engine until a stop criterion is met.

    Parameters
    ----------
    pars : parameter set
    state : starting state; a fresh egg (:func:`initial_state`) when omitted
    engine : ``"stepped"`` (fixed-dt explicit Euler) or ``"closed"``
        (``scipy.integrate.solve_ivp`` per stage segment with terminal events)
    dt : step size in days for the stepped engine; also the output sampling
        interval for the closed engine
    f : scaled functional response -- a constant, or a callable ``f(t)``
    T : absolute temperature in K; ``None`` means no correction (``TC = 1``)
    until_age : stop when age (d) reaches this value
    until_stage : stop on entering this stage
    until_maturity : stop when ``E_H`` (J) reaches this value
    max_steps : safety bound for the stepped engine
    record_every : record every n-th step

    Returns
    -------
    (state, trajectory) : the final state and the recorded trajectory.

    Notes
    -----
    Death (``E <= 0``) always terminates the run. At least one stop criterion
    should be supplied; otherwise the run continues until death or ``max_steps``.
    """
    if engine not in ("stepped", "closed"):
        raise ValueError(f"unknown engine {engine!r}; expected 'stepped' or 'closed'")
    if until_stage is not None and until_stage not in STAGES:
        raise ValueError(
            f"unknown until_stage {until_stage!r}; expected one of {STAGES}"
        )

    state = initial_state(pars) if state is None else state
    if engine == "stepped":
        return _run_stepped(
            pars,
            state,
            dt,
            f,
            T,
            until_age,
            until_stage,
            until_maturity,
            max_steps,
            record_every,
        )
    return _run_closed(pars, state, dt, f, T, until_age, until_stage, until_maturity)


def _new_recorder(state: DEBState) -> dict[str, list]:
    return {
        "t": [state.age],
        "E": [state.E],
        "V": [state.V],
        "E_H": [state.E_H],
        "E_R": [state.E_R],
        "stage": [state.stage],
    }


def _finish(rec: dict[str, list], events: dict[str, float]) -> Trajectory:
    return Trajectory(
        t=np.asarray(rec["t"], dtype=float),
        E=np.asarray(rec["E"], dtype=float),
        V=np.asarray(rec["V"], dtype=float),
        E_H=np.asarray(rec["E_H"], dtype=float),
        E_R=np.asarray(rec["E_R"], dtype=float),
        stage=list(rec["stage"]),
        events=events,
    )


def _run_stepped(
    pars: DEBPars,
    state: DEBState,
    dt: float,
    f: Any,
    T: Optional[float],
    until_age: Optional[float],
    until_stage: Optional[str],
    until_maturity: Optional[float],
    max_steps: int,
    record_every: int,
) -> tuple[DEBState, Trajectory]:
    rec = _new_recorder(state)
    events: dict[str, float] = {state.stage: state.age}

    n = 0
    while n < max_steps and state.alive:
        if _stop_reached(state, until_age, until_stage, until_maturity):
            break
        previous = state.stage
        step(
            state,
            pars,
            dt=dt,
            f=_resolve_f(f, state.age) if f is not None else None,
            T=T,
        )
        if state.stage != previous:
            events.setdefault(state.stage, state.age)
        n += 1
        if n % record_every == 0:
            for key, val in (
                ("t", state.age),
                ("E", state.E),
                ("V", state.V),
                ("E_H", state.E_H),
                ("E_R", state.E_R),
                ("stage", state.stage),
            ):
                rec[key].append(val)

    return state, _finish(rec, events)


def _run_closed(
    pars: DEBPars,
    state: DEBState,
    dt: float,
    f: Any,
    T: Optional[float],
    until_age: Optional[float],
    until_stage: Optional[str],
    until_maturity: Optional[float],
) -> tuple[DEBState, Trajectory]:
    from scipy.integrate import solve_ivp  # lazy: keeps module import cheap

    TC = temperature_correction(pars, T)
    #: maturity threshold that terminates each stage
    threshold = {
        Stage.EGG: pars.E_Hb,
        Stage.LARVA: pars.E_Hp,
        Stage.PUPA: pars.E_He,
        Stage.IMAGO: None,
    }

    rec = _new_recorder(state)
    events: dict[str, float] = {state.stage: state.age}
    t_end = state.age + 1e4 if until_age is None else until_age

    while state.alive and state.age < t_end:
        if _stop_reached(state, until_age, until_stage, until_maturity):
            break

        stage = state.stage

        def rhs(t: float, y: Sequence[float]) -> list[float]:
            E, V, E_H, _E_R = y
            s_M = acceleration(max(V, 0.0) ** (1.0 / 3.0), state.L_b, state.L_p)
            p = powers(
                stage=stage,
                E=E,
                V=max(V, 0.0),
                E_H=E_H,
                pars=pars,
                f=_resolve_f(f, t) if f is not None else None,
                s_M=s_M,
                TC=TC,
            )
            return list(derivatives(stage, p, pars))

        def death(t: float, y: Sequence[float]) -> float:
            return y[0]

        death.terminal = True
        death.direction = -1
        ev: list[Callable] = [death]

        E_H_target = threshold[stage]
        if E_H_target is not None:

            def matured(
                t: float, y: Sequence[float], _target: float = E_H_target
            ) -> float:
                return y[2] - _target

            matured.terminal = True
            matured.direction = 1
            ev.append(matured)

        if until_maturity is not None:

            def reached(
                t: float, y: Sequence[float], _target: float = until_maturity
            ) -> float:
                return y[2] - _target

            reached.terminal = True
            reached.direction = 1
            ev.append(reached)

        n_out = max(2, int(round((t_end - state.age) / dt)) + 1)
        sol = solve_ivp(
            rhs,
            t_span=(state.age, t_end),
            y0=[state.E, state.V, state.E_H, state.E_R],
            events=ev,
            t_eval=np.linspace(state.age, t_end, n_out),
            rtol=1e-8,
            atol=1e-12,
            method="LSODA",
        )
        if not sol.success:
            raise RuntimeError(
                f"closed-engine integration failed in stage {stage!r}: {sol.message}"
            )

        for i in range(1, sol.t.size):
            rec["t"].append(float(sol.t[i]))
            rec["E"].append(float(sol.y[0, i]))
            rec["V"].append(float(sol.y[1, i]))
            rec["E_H"].append(float(sol.y[2, i]))
            rec["E_R"].append(float(sol.y[3, i]))
            rec["stage"].append(stage)

        state.age = float(sol.t[-1])
        state.E, state.V, state.E_H, state.E_R = (float(v) for v in sol.y[:, -1])

        if sol.status != 1:  # no terminal event fired -> t_end reached
            break

        # A terminal event fired. Adopt the state at the event itself rather than at
        # the last t_eval grid point before it, otherwise every stage transition --
        # and hence L_b, L_p and all recorded event ages -- is biased early by up to
        # one output interval.
        fired = next(i for i, te in enumerate(sol.t_events) if te.size)
        state.age = float(sol.t_events[fired][0])
        state.E, state.V, state.E_H, state.E_R = (
            float(v) for v in sol.y_events[fired][0]
        )
        for key, val in (
            ("t", state.age),
            ("E", state.E),
            ("V", state.V),
            ("E_H", state.E_H),
            ("E_R", state.E_R),
            ("stage", stage),
        ):
            rec[key].append(val)

        if fired == 0:  # death
            state.E = 0.0
            break
        if E_H_target is not None and fired == 1:  # stage transition
            state.E_H = max(state.E_H, E_H_target)
            transition(state, pars)
            events.setdefault(state.stage, state.age)
            rec["stage"][-1] = state.stage
            rec["E"][-1] = state.E
            rec["V"][-1] = state.V
            rec["E_H"][-1] = state.E_H
            continue
        break  # the until_maturity event fired

    return state, _finish(rec, events)
