"""
DEB pipeline from literature
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import os

import numpy as np
import param

from .... import ROOT_DIR
from ... import util
from ...util import beta0, nam, simplex
from ...param import (
    ClassAttr,
    Life,
    NestedConf,
    OptionalPositiveNumber,
    PositiveNumber,
    Substrate,
)
from . import Gut

__all__: list[str] = [
    "DEB_model",
    "DEB_basic",
    "DEB",
]


# p.257 in S. a. L. M. Kooijman, "Dynamic Energy Budget theory for metabolic organisation : Summary of concepts of the third edition," Water, vol. 365, p. 68, 2010.
class DEB_model(NestedConf):
    """
    Dynamic Energy Budget (DEB) model parameters.

    Implements standard DEB theory parameters for metabolic organization
    following Kooijman (2010). Provides core DEB equations for growth,
    maintenance, reproduction, and aging.

    Key DEB parameters include surface-area specific rates (F_m, p_Am),
    volume-specific costs (E_G, p_M), allocation fractions (kap, kap_X),
    maturity thresholds (E_Hb, E_He), and aging parameters (h_a, s_G).

    Reference:
        Kooijman (2010). "Dynamic Energy Budget theory for metabolic
        organisation." Water, vol. 365, p. 68.

    Example:
        >>> deb = DEB_model(species="rover")
        >>> deb.compute_compound_pars()
    """

    F_m = PositiveNumber(
        6.5, doc="maximum surface-area specific searching rate (l cm**-2 d**-1)"
    )
    kap_X = param.Magnitude(0.8, doc="assimilation efficiency")
    p_Am = PositiveNumber(
        229.0, doc="maximum surface-area specific assimilation rate (J cm**-2 d**-1)"
    )
    E_G = PositiveNumber(4400.0, doc="volume-specific cost of structure (J/cm**3)")
    v = PositiveNumber(0.12, doc="energy conductance (cm/d)")
    p_M = PositiveNumber(
        210.0, doc="volume-specific somatic maintenance (J cm**-3 d**-1)"
    )
    kap = param.Magnitude(0.99, doc="fraction of mobilized reserve allocated to soma")
    k_J = PositiveNumber(0.002, doc="maturity maintenance rate coefficient (d**-1)")
    E_Hb = PositiveNumber(0.0006, doc="maturity threshold from embryo to juvenile (J)")
    E_He = PositiveNumber(0.05, doc="maturity threshold from juvenile to adult (J)")
    z = PositiveNumber(0.5, doc="zoom factor")
    del_M = PositiveNumber(0.5, doc="shape correction coefficient")
    s_G = PositiveNumber(0.0001, doc="Gompertz stress coefficient")
    h_a = PositiveNumber(0.0001, doc="Weibull ageing acceleration (d**-2)")
    kap_R = param.Magnitude(
        0.95, doc="fraction of the reproduction buffer fixed into eggs"
    )

    E_M = OptionalPositiveNumber(doc="maximum reserve capacity")
    k_M = OptionalPositiveNumber(doc="somatic maintenance rate coefficient")
    k = OptionalPositiveNumber(doc="maintenance ratio")
    g = OptionalPositiveNumber(doc="energy investment ratio")
    Lm = OptionalPositiveNumber(doc="maximum length")
    K = OptionalPositiveNumber(doc="half-saturation coefficient")

    p_T = PositiveNumber(0.0, doc="??")
    kap_V = param.Magnitude(
        0.99,
        doc="fraction of energy in mobilised larval structure fixed in pupal reserve: ylEV /yEV",
    )
    kap_P = param.Magnitude(0.18, doc="fraction of food energy fixed in faeces")

    eb = PositiveNumber(1.0, doc="scaled reserve density at birth")
    s_j = PositiveNumber(0.999, doc="??")

    T = PositiveNumber(298.15, doc="Temperature")
    T_ref = PositiveNumber(293.15, doc="Reference temperature")
    T_A = PositiveNumber(8000, doc="Arrhenius temperature")

    mu_E = PositiveNumber(550000.0, doc="specific chemical potential of compound E")
    mu_V = PositiveNumber(500000.0, doc="specific chemical potential of compound V")
    mu_X = PositiveNumber(525000.0, doc="specific chemical potential of compound X")
    mu_P = PositiveNumber(500000.0, doc="specific chemical potential of compound P")
    mu_C = PositiveNumber(480000.0, doc="specific chemical potential of compound C")
    mu_H = PositiveNumber(0.0, doc="specific chemical potential of compound H")
    mu_O = PositiveNumber(0.0, doc="specific chemical potential of compound O")
    mu_N = PositiveNumber(0.0, doc="specific chemical potential of compound N")
    d_V = PositiveNumber(0.17, doc="density of compound V")
    d_X = PositiveNumber(0.17, doc="density of compound X")
    d_E = PositiveNumber(0.17, doc="density of compound E")
    d_P = PositiveNumber(0.17, doc="density of compound P")
    w_V = PositiveNumber(23.9, doc="molar weight of compound V")
    w_E = PositiveNumber(23.9, doc="molar weight of compound E")
    w_X = PositiveNumber(23.9, doc="molar weight of compound X")
    w_P = PositiveNumber(23.9, doc="molar weight of compound P")
    y_E_X = PositiveNumber(
        0.7, doc="yield coefficient that couples mass flux E to mass flux X"
    )
    y_P_X = PositiveNumber(
        0.2, doc="yield coefficient that couples mass flux P to mass flux X"
    )
    y_E_V = PositiveNumber(
        0.2, doc="yield coefficient that couples mass flux E to mass flux V"
    )

    def __init__(self, print_output: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.print_output = print_output
        self.stages = ["embryo", "larva", "pupa", "imago"]
        self.stage_events = [
            "oviposition",
            "eclosion",
            "pupation",
            "emergence",
            "death",
        ]

        self.L0 = 10**-10

        self.derive_pars()
        self.compute_initial_state()
        self.predict_life_history()

    def derive_pars(self) -> None:
        # self.p_Am = self.z*self.p_M/self.kap
        self.E_M = self.p_Am / self.v  # maximum reserve density
        self.k_M = self.p_M / self.E_G
        self.k = self.k_J / self.k_M
        self.g = self.E_G / (self.kap * self.E_M)
        self.Lm = self.v / (self.g * self.k_M)
        self.xb = self.g / (self.eb + self.g)
        self.Ucoeff = self.g**2 * self.k_M**3 / self.v**2

        self.vHb = self.E_Hb / self.p_Am * self.Ucoeff / (1 - self.kap)
        self.vHe = self.E_He / self.p_Am * self.Ucoeff / (1 - self.kap)

        self.J_E_Am = self.p_Am / self.mu_E
        self.J_X_Am = self.J_E_Am / self.y_E_X
        self.K = self.J_X_Am / self.F_m

    def get_lb(self) -> float:
        g = self.g
        xb = self.xb
        n = 1000 + round(1000 * max(0, self.k - 1))
        xb3 = xb ** (1 / 3)
        x = np.linspace(10**-5, xb, n)
        dx = xb / n
        x3 = x ** (1 / 3)

        b = beta0(x, xb) / (3 * g)

        t0 = xb * g * self.vHb
        i = 0
        norm = 1
        ni = 100

        lb = self.vHb ** (1 / 3)

        while i < ni and norm > 1e-18:
            l = x3 / (xb3 / lb - b)
            s = (self.k - x) / (1 - x) * l / g / x
            vv = np.exp(-dx * np.cumsum(s))
            vb = vv[-1]
            r = g + l
            rv = r / vv
            t = t0 / lb**3 / vb - dx * np.sum(rv)
            dl = xb3 / lb**2 * l**2.0 / x3
            dlnv = np.exp(-dx * np.cumsum(s * dl / l))
            dlnvb = dlnv[-1]
            dt = -t0 / lb**3 / vb * (3 / lb + dlnvb) - dx * np.sum((dl / r - dlnv) * rv)
            lb -= t / dt  # Newton Raphson step
            norm = t**2
            i += 1
        return lb

    def get_tau_b(self) -> float:
        from scipy.integrate import quad

        def get_tb(x, ab, xb):
            return x ** (-2 / 3) / (1 - x) / (ab - beta0(x, xb))

        ab = 3 * self.g * self.xb ** (1 / 3) / self.lb
        return 3 * quad(func=get_tb, a=1e-15, b=self.xb, args=(ab, self.xb))[0]

    def get_E0(self) -> float:
        """
        This function calculates the maximum reserve density (E0) that an organism can achieve given its energy budget parameters.

        Parameters
        ----------
            kap (float): Fraction of assimilated energy that is used for somatic maintenance.
            v (float): Energy conductance.
            p_M (float): Specific somatic maintenance costs.
            p_Am (float): Maximum surface-specific assimilation rate.
            E_G (float): Energy investment ratio.
            eb (float, optional): Allocation fraction to reserve production. Defaults to 1.0.
            lb (float, optional): Length at birth. If not provided, it is calculated from the other parameters. Defaults to None.

        Returns
        -------
            float: Maximum reserve density that an organism can achieve.

        """
        # Calculate uE0 using the equation in the Dynamic Energy Budget textbook
        uE0 = np.real(
            (
                3
                * self.g
                / (3 * self.g * self.xb ** (1 / 3) / self.lb - beta0(0, self.xb))
            )
            ** 3
        )

        # Calculate U0 and E0 using the equations in the Dynamic Energy Budget textbook
        return self.p_Am * uE0 / self.Ucoeff

    def compute_initial_state(self) -> None:
        self.lb = self.get_lb()
        self.E0 = self.get_E0()
        self.Lw0 = self.L0 / self.del_M
        self.Ww0 = self.compute_Ww(V=self.L0**3, E=self.E0)

    def predict_embryo_stage(self) -> None:
        self.k_E = self.g * self.k_M / self.lb
        self.Lb = self.lb * self.Lm
        self.Lwb = self.Lb / self.del_M

        # TODO Compute Eb and Ej
        self.Eb = self.E0
        self.Wwb = self.compute_Ww(V=self.Lb**3, E=self.Eb)  # g, wet weight at birth

        self.v_Rm = (1 + self.lb / self.g) / (
            1 - self.lb
        )  # scaled max reprod buffer density
        self.v_Rj = self.s_j * self.v_Rm  # scaled reprod buffer density at pupation

        # self.E_Rm = (self.kap - 1) * self.E_M / self.v_Rm
        # self.E_Rm = (1 - self.kap) * self.g * self.E_M * (self.k_E + self.k_M) / (self.k_E - self.g * self.k_M)

        self.t_b = self.get_tau_b() / self.k_M / self.T_factor

        # For the larva the volume specific max assimilation rate p_Amm is used instead of the surface-specific p_Am
        # self.p_Amm = self.p_Am / self.Lb
        # self.J_X_Amm = self.J_X_Am / self.Lb
        # self.J_E_Amm = self.J_E_Am / self.Lb
        # self.F_mm = self.F_m / self.Lb

        # DEB textbook p.91
        # self.y_VE = (self.d_V / self.w_V)*self.mu_E/E_G
        # self.J_E_Am = self.p_Am/self.mu_E

        # self.U0 = self.uE0 * v ** 2 / g ** 2 / k_M ** 3
        # self.E0 = self.U0 * p_Am

    def predict_larva_stage(self, f: float = 1.0) -> None:
        g = self.g
        lb = self.lb
        c1 = f / g * (g + lb) / (f - lb)
        c2 = self.k * self.vHb / lb**3
        self.rho_j = (f / lb - 1) / (f / g + 1)  # scaled specific growth rate of larva

        def get_tj(tau_j):
            ert = np.exp(-tau_j * self.rho_j)
            return np.abs(self.v_Rj - c1 * (1 - ert) + c2 * tau_j * ert)

        self.tau_j = simplex(get_tj, 1)
        self.lj = lb * np.exp(self.tau_j * self.rho_j / 3)
        self.t_j = self.tau_j / self.k_M / self.T_factor
        self.Lj = self.lj * self.Lm
        self.Lwj = self.Lj / self.del_M

        self.E_Rm = self.v_Rm * (1 - self.kap) * g * self.E_M * self.Lj**3
        self.E_Rj = self.E_Rm * self.s_j
        self.E_eggs = self.E_Rm * self.kap_R
        # TODO Compute Eb and Ej
        self.uEj = self.lj**3 * (self.kap * self.kap_V + f / self.g)
        self.Ej = self.uEj / self.Ucoeff * self.p_Am
        # self.Ej = self.Eb * np.exp(self.tau_j * self.rho_j)
        self.Wwj = self.compute_Ww(
            V=self.Lj**3, E=self.Ej + self.E_Rj
        )  # g, wet weight at pupation

    def predict_pupa_stage(self) -> None:
        from scipy.integrate import solve_ivp

        g = self.g
        k_M = self.k_M

        def emergence(t, luEvH, terminal=True, direction=0):
            return self.vHe - luEvH[2]

        def get_te(t, luEvH):
            l = luEvH[0]
            u_E = max(1e-6, luEvH[1])
            ii = u_E + l**3
            dl = (g * u_E - l**4) / ii / 3
            du_E = -u_E * l**2 * (g + l) / ii
            dv_H = -du_E - self.k * luEvH[2]
            return [dl, du_E, dv_H]  # pack output

        sol = solve_ivp(
            fun=get_te, t_span=(0, 1000), y0=[0, self.uEj, 0], events=emergence
        )
        self.tau_e = sol.t_events[0][0]
        self.le, self.uEe = sol.y_events[0][0][:2]
        self.t_e = self.tau_e / k_M / self.T_factor
        self.Le = self.le * self.Lm
        self.Lwe = self.Le / self.del_M
        self.Ee = self.uEe / self.Ucoeff * self.p_Am
        self.Wwe = self.compute_Ww(
            V=self.Le**3, E=self.Ee + self.E_Rj
        )  # g, wet weight at emergence

    def predict_imago_stage(self, f: float = 1.0) -> None:
        # if np.abs(self.sG) < 1e-10:
        #     self.sG = 1e-10
        # self.uh_a =self.h_a/ self.k_M ** 2 # scaled Weibull aging coefficient
        self.lT = self.p_T / (self.p_M * self.Lm)  # scaled heating length {p_T}/[p_M]Lm
        self.li = f - self.lT
        # self.hW3 = self.ha * f * self.g/ 6/ self.li
        # self.hW = self.hW3**(1/3) # scaled Weibull aging rate
        # self.hG = self.sG * f * self.g * self.li**2
        # self.hG3 = self.hG**3;     # scaled Gompertz aging rate
        # self.tG = self.hG/ self.hW
        # self.tG3 = self.hG3/ self.hW3 # scaled Gompertz aging rate
        # # self.tau_m = sol.t_events[0][0]
        # # self.lm, self.uEm=sol.y_events[0][0][:2]
        # TODO compute tau_i and uEi
        self.tau_i = self.tau_e
        self.uEi = self.uEe

        self.t_i = self.tau_i / self.k_M / self.T_factor
        self.Li = self.li * self.Lm
        self.Lwi = self.Li / self.del_M
        self.Ei = self.uEi / self.Ucoeff * self.p_Am
        self.Wwi = self.compute_Ww(
            V=self.Li**3, E=self.Ei + self.E_Rj
        )  # g, imago wet weight

    def predict_life_history(self, f: float = 1.0) -> None:
        self.predict_embryo_stage()
        self.predict_larva_stage(f=f)
        self.predict_pupa_stage()
        self.predict_imago_stage(f=f)

        Es = [self.E0, self.Eb, self.Ej, self.Ee, self.Ei]
        Wws = [self.Ww0, self.Wwb, self.Wwj, self.Wwe, self.Wwi]
        Lws = [self.Lw0, self.Lwb, self.Lwj, self.Lwe, self.Lwi]
        Durs = [self.t_b, self.t_j, self.t_e, self.t_i]

        if self.print_output:
            self.print_life_history(Es, Wws, Lws, Durs)

    def print_life_history(
        self, Es: List[float], Wws: List[float], Lws: List[float], Durs: List[float]
    ) -> None:
        ages = np.cumsum(Durs).tolist()
        ages.insert(0, 0)

        ls = "Life stage           :"
        le = "Life events          :"
        lt = "Duration        (d)  :"
        la = "Age             (d)  :"
        lE = "Reserve energy  (J)  :"
        lW = "Wet weight      (mg) :"
        lL = "Physical length (mm) :"

        for i in range(len(Es)):
            le += f"{self.stage_events[i]}"
            ls += "                 "
            lt += "                 "
            la += f"  {np.round(ages[i], 3)}    "
            lE += f"  {np.round(Es[i], 3)}    "
            lW += f"  {np.round(1000 * Wws[i], 3)}    "
            lL += f"  {np.round(10 * Lws[i], 3)}  "
            try:
                ls += f"-- {self.stages[i]} --"
                lt += f"    {np.round(Durs[i], 3)}  "
                le += "                 "
                la += "                 "
                lE += "                 "
                lW += "                 "
                lL += "                 "
            except:
                pass
        for l in [ls, le, lt, lE, lL, lW]:
            print(l)

    @property
    def M_V(self) -> float:
        """Number of C-atoms per unit of structural body volume V : dV /wV"""
        return self.d_V / self.w_V

    @property
    def T_factor(self) -> float:
        return np.exp(self.T_A / self.T_ref - self.T_A / self.T)  # Arrhenius factor

    def compute_Ww(self, V: float, E: float) -> float:
        return V * self.d_V + E * self.w_E / self.mu_E

    @classmethod
    def from_file(cls, species: str = "default", **kwargs: Any) -> "DEB_model":
        # Drosophila model by default
        with open(f"{ROOT_DIR}/lib/model/deb/models/deb_{species}.csv") as tfp:
            d = json.load(tfp)
        kwargs.update(**d)
        return cls(**kwargs)


class DEB_basic(DEB_model):
    """
    Basic DEB model with species-specific parameters and gut integration.

    Extends DEB_model with species phenotypes (rover/sitter), gut assimilation,
    starvation strategies, and aging dynamics. Includes fitted parameters for
    Drosophila larva phenotypes.

    Attributes:
        id: Model identifier (default: "DEB model")
        species: Phenotype selection ("default", "rover", "sitter")
        starvation_strategy: Enable starvation response (default: False)
        aging: Enable aging dynamics (default: False)
        dt: Simulation timestep in days (default: 1/(24*60))
        substrate: Feeding substrate configuration
        assimilation_mode: Assimilation calculation method ("gut", "sim", "deb")

    Example:
        >>> deb = DEB_basic(species="rover", aging=True, dt=1/1440)
        >>> deb.step_basic(f=0.8, V=0.001)
    """

    id = param.String("DEB model", doc="The unique ID of the DEB model")
    species = param.Selector(
        objects=["default", "rover", "sitter"],
        label="phenotype",
        doc="The phenotype/species-specific fitted DEB model to use.",
    )  # Drosophila model by default
    starvation_strategy = param.Boolean(
        False, doc="Whether starvation strategy is active"
    )
    aging = param.Boolean(False, doc="Whether aging is active")
    dt = PositiveNumber(
        1 / (24 * 60), doc="The timestep of the DEB energetics module in days."
    )
    substrate = ClassAttr(Substrate, doc="The substrate where the agent feeds")
    assimilation_mode = param.Selector(
        objects=["gut", "sim", "deb"],
        label="assimilation mode",
        doc="The method used to calculate the DEB assimilation energy flow.",
    )

    def __init__(
        self,
        species: str = "default",
        save_dict: bool = True,
        V_bite: float = 0.001,
        gut_params: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Drosophila model by default
        with open(f"{ROOT_DIR}/lib/model/deb/models/deb_{species}.csv") as tfp:
            species_dict = json.load(tfp)
        kwargs.update(**species_dict)
        super().__init__(species=species, **kwargs)

        self.E_H = 0
        self.E_R = 0
        self.V = self.L0**3  # larval structure
        self.V2 = 0  # adult structur
        self.E_egg = 0  # egg buffer
        self.E = self.E0

        # Stage duration parameters
        self.age = 0

        self.epochs = []
        self.epoch_qs = []

        self.deb_p_A = 0
        self.sim_p_A = 0

        self.base_f = self.substrate.get_f(K=self.K)
        self.f = self.base_f
        self.V_bite = V_bite
        self.X_V_buffer = 0
        self.time_buffer = 0

        if gut_params is None:
            gut_params = {}
        self.gut = Gut(deb=self, save_dict=save_dict, **gut_params)
        self.scale_time()

    @property
    def dt_in_sec(self) -> float:
        return self.dt * 24 * 60 * 60

    @property
    def alive(self) -> bool:
        return self.E > 0

    @property
    def stage(self) -> str:
        if not self.alive:
            return "dead"
        if self.E_H < self.E_Hb:
            return "embryo"
        elif self.E_R < self.E_Rj and self.E_H < self.E_He:
            return "larva"
        elif self.E_H < self.E_He:
            return "pupa"
        else:
            return "imago"

    @property
    def Lw(self) -> float:
        return self.L / self.del_M

    @property
    def L(self) -> float:
        return (self.V + self.V2) ** (1 / 3)

    @property
    def Ww(self) -> float:
        return self.compute_Ww(V=self.V, E=self.E + self.E_R)

    @property
    def e(self) -> float:
        return self.E / self.V / self.E_M

    @property
    def Vw(self) -> float:
        return self.V + self.w_E / self.d_E / self.mu_E * self.E

    @property
    def pupation_buffer(self) -> float:
        return self.E_R / self.E_Rj

    @property
    def time_to_death_by_starvation(self) -> float:
        return self.v**-1 * self.L * np.log(self.kap**-1)

    def scale_time(self) -> None:
        dt = self.dt * self.T_factor
        self.v_dt = self.v * dt
        self.p_M_dt = self.p_M * dt
        self.p_T_dt = self.p_T * dt if self.p_T != 0.0 else 0.0
        self.k_J_dt = self.k_J * dt
        self.p_Amm_dt = self.p_Am / self.Lb * dt
        self.J_X_Amm_dt = self.J_X_Am / self.Lb * dt
        self.J_E_Amm_dt = self.J_E_Am / self.Lb * dt
        self.k_E_dt = self.k_E * dt

        self.J_X_A_array = np.ones(int(self.gut.residence_time / dt)) * self.J_X_A

    def hex_model(self) -> None:
        # p.161    [1] S. a. L. M. Kooijman, “Comments on Dynamic Energy Budget theory,” Changes, 2010.
        # For the larva stage
        # self.r = self.g * self.k_M * (self.e/self.lb -1)/(self.e+self.g) # growth rate at  constant food where e=f
        # self.k_E = self.v/self.Lb # Reserve turnover
        pass

    def apply_fluxes(self, **kwargs: Any) -> None:
        """
        Energy fluxes at different life stages of holometabolous insects.
        Based on 'A dynamic energy budget for the whole life-cycle of holometabolous insects' Llandres(2015) Table 5
        """
        ST = self.stage

        V = self.V  # larval structure
        V2 = self.V2  # imago structure
        E = self.E

        k = self.kap
        kR = self.kap_R
        kV = self.kap_V
        EG = self.E_G

        pM = self.p_M_dt
        pT = self.p_T_dt
        v = self.v_dt
        vj = self.v_dt
        kJ = self.k_J_dt

        kE = self.k_E_dt

        if ST == "embryo":
            p_S = pM * V + pT * V ** (2 / 3)
            p_C = E * (EG * v / V ** (1 / 3) + p_S) / (k * E / V + EG)
            p_G = k * p_C - p_S
            p_J = kJ * self.E_H
            p_R = (1 - k) * p_C - p_J
            self.E -= p_C
            self.V += p_G / EG
            self.E_H += p_R
        elif ST == "larva":
            p_A = self.get_p_A(**kwargs)
            p_S = pM * V
            p_C = E * (EG * kE + p_S) / (k * E / V + EG)
            p_G = k * p_C - p_S
            p_J = kJ * self.E_Hb
            p_R = (1 - k) * p_C - p_J
            self.E += p_A - p_C
            self.V += p_G / EG
            self.E_R += p_R
        elif ST == "pupa":
            p_L = V * kV  # kEl #Transformation of larval structure
            p_S = pM * V2
            p_C = E * (EG * vj / V2 ** (1 / 3) + p_S) / (k * E / V + EG)
            p_G = k * p_C - p_S
            p_J = kJ * self.E_H
            p_R = (1 - k) * p_C - p_J
            p_C2 = self.E_R * kE  # Mobilization of ER
            p_RO = (1 - kR) * p_C2  # Reproduction overhead
            p_R2 = p_C2 - p_RO  # Egg flux
            self.E += p_L * self.y_E_V * self.mu_E * self.M_V - p_C
            self.V -= p_L
            self.V2 += p_G / EG
            self.E_H += p_R
            self.E_R -= p_C2
            self.E_egg += p_R2
        elif ST == "imago":
            p_C = E * kE
            p_S = pM * V2
            p_J = kJ * self.E_He
            p_R = p_C - p_S - p_J
            p_A = p_S + p_J
            p_C2 = self.E_R * kE  # Mobilization of ER
            p_RO = (1 - kR) * p_C2  # Reproduction overhead
            p_R2 = p_C2 - p_RO  # Egg flux
            self.E += p_A - p_C
            self.E_R += p_R - p_C2
            self.E_egg += p_R2
        else:
            raise

    def run_stage(
        self, stage: str, assimilation_mode: str = "deb", **kwargs: Any
    ) -> float:
        t = 0
        while self.stage == stage and self.alive:
            self.apply_fluxes(assimilation_mode=assimilation_mode, **kwargs)
            t += self.dt
            self.age += self.dt
            self.update()
        return t

    def run_life_history(self, **kwargs: Any) -> None:
        if not self.age == 0:
            return
        Es, Wws, Lws, Durs = [self.E], [self.Ww], [self.Lw], []
        while self.alive:
            t = self.run_stage(self.stage, **kwargs)
            Durs.append(t)
            Es.append(self.E)
            Wws.append(self.Ww)
            Lws.append(self.Lw)

        if self.print_output:
            self.print_life_history(Es, Wws, Lws, Durs)

    def run(self, **kwargs: Any) -> None:
        if self.alive:
            self.age += self.dt
            if self.stage == "larva":
                self.apply_fluxes(**kwargs)
            elif self.stage == "pupa":
                self.pupation_time_in_hours_sim = np.round(self.age * 24, 2)
        self.update()

    def run_check(self, dt: float, X_V: float = 0) -> None:
        self.X_V_buffer += X_V
        self.time_buffer += dt
        if self.time_buffer >= self.dt_in_sec:
            self.run(X_V=self.X_V_buffer)
            self.X_V_buffer = 0
            self.time_buffer = 0

    def update(self) -> None:
        pass

    @property
    def J_X_A(self) -> float:
        return self.J_X_Am / self.Lb * self.V * self.base_f

    @property
    def F(self) -> float:
        """Vol specific filtering rate (cm**3/(d*cm**3) -> vol of environment/vol of individual*day"""
        return (
            self.J_X_Am
            * self.F_m
            / (self.Lb * (self.J_X_Am + self.substrate.X * self.F_m))
        )

    @property
    def fr_feed(self) -> float:
        freq = self.F / self.V_bite * self.T_factor
        freq /= 24 * 60 * 60
        return freq

    def get_best_EEB(self, cRef: Dict[str, Any]) -> float:
        z = np.poly1d(cRef["EEB_poly1d"])
        return np.clip(z(self.fr_feed), a_min=0, a_max=1)

    def grow_larva(self, epochs: List[Any], **kwargs: Any) -> None:
        self.run_stage(stage="embryo")
        tb = self.age * 24
        for e in epochs:
            if self.stage == "larva":
                c = {"assimilation_mode": "sim", "f": e.substrate.get_f(K=self.K)}
                if e.end is None:
                    self.run_stage(stage="larva", **c)
                else:
                    for i in range(e.ticks(self.dt)):
                        if self.stage == "larva":
                            self.run(**c)

                self.epochs.append([e.start + tb, self.age * 24])
                self.epoch_qs.append(e.substrate.quality)
        # self.gut.update()

    def get_p_A(
        self,
        f: Optional[float] = None,
        assimilation_mode: Optional[str] = None,
        X_V: float = 0.0,
    ) -> float:
        if f is None:
            f = self.base_f
        self.f = f
        self.deb_p_A = self.p_Amm_dt * self.base_f * self.V
        self.sim_p_A = self.p_Amm_dt * f * self.V
        if assimilation_mode is None:
            assimilation_mode = self.assimilation_mode
        if assimilation_mode == "sim":
            return self.sim_p_A
        elif assimilation_mode == "gut":
            self.gut.update(X_V)
            return self.gut.p_A
        elif assimilation_mode == "deb":
            return self.deb_p_A

    @property
    def steps_per_day(self) -> int:
        return int(1 / self.dt)

    @property
    def ingested_body_mass_ratio(self) -> float:
        return self.gut.ingested_mass() / self.Ww * 100

    @property
    def ingested_body_volume_ratio(self) -> float:
        return self.gut.ingested_volume / self.V * 100

    @property
    def ingested_gut_volume_ratio(self) -> float:
        return self.gut.ingested_volume / (self.V * self.gut.V_gm) * 100

    @property
    def ingested_body_area_ratio(self) -> float:
        return (self.gut.ingested_volume / self.V) ** (1 / 2) * 100

    @property
    def amount_absorbed(self) -> float:
        return self.gut.absorbed_mass("mg")

    @property
    def volume_ingested(self) -> float:
        return self.gut.ingested_volume

    @property
    def deb_f_deviation(self) -> float:
        return self.f - 1


class DEB(DEB_basic):
    """
    Full DEB model with hunger-driven behavior and data recording.

    Complete DEB implementation integrating energetics with behavioral control
    through hunger-driven exploration-exploitation balance (EEB). Tracks and
    records all DEB state variables over time for analysis.

    Attributes:
        hunger_as_EEB: Use hunger to modulate EEB (default: True)
        hunger_gain: Hunger sensitivity to reserve depletion (default: 1.0)
        dict: AttrDict storing timeseries of DEB state variables
              (age, mass, length, reserve, hunger, etc.)
        save_to: File path for saving DEB trajectory data

    Example:
        >>> deb = DEB(species="rover", hunger_as_EEB=True, save_to="results.h5")
        >>> deb.step(f=0.9, V_food=0.002, dt=0.01)
        >>> hunger = deb.get_hunger()
    """

    hunger_as_EEB = param.Boolean(
        True,
        doc="Whether the DEB-generated hunger drive informs the exploration-exploitation balance.",
    )
    hunger_gain = param.Magnitude(
        1.0,
        label="hunger sensitivity to reserve reduction",
        doc="The sensitivy of the hunger drive in deviations of the DEB reserve density.",
    )

    def __init__(
        self,
        save_dict: bool = True,
        save_to: str | None = None,
        base_hunger: float = 0.5,
        intermitter: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.set_intermitter(base_hunger, intermitter)
        self.save_to = save_to
        self.dict = (
            util.AttrDict(
                {
                    k: []
                    for k in [
                        "age",
                        "mass",
                        "length",
                        "reserve",
                        "reserve_density",
                        "hunger",
                        "pupation_buffer",
                        "f",
                        "deb_p_A",
                        "sim_p_A",
                        "EEB",
                    ]
                }
            )
            if save_dict
            else None
        )

    def set_intermitter(
        self, base_hunger: float = 0.5, intermitter: Any | None = None
    ) -> None:
        self.intermitter = intermitter
        if self.intermitter is not None:
            if self.hunger_as_EEB:
                base_hunger = self.intermitter.base_EEB
        self.base_hunger = base_hunger
        self.update_hunger()

    def update(self) -> None:
        self.update_hunger()
        self.update_dict()

    @property
    def birth_time_in_hours(self) -> float:
        try:
            t = self.t_b_comp
        except:
            t = self.t_b
        return np.round(t * 24, 1)

    @property
    def pupation_time_in_hours(self) -> float:
        try:
            return self.pupation_time_in_hours_sim
        except:
            try:
                t = self.t_j_comp
            except:
                t = self.t_j
            return self.birth_time_in_hours + np.round(t * 24, 1)

    @property
    def death_time_in_hours(self) -> float:
        if not self.alive:
            return self.age * 24
        else:
            return np.nan

    def update_hunger(self) -> None:
        self.hunger = np.clip(
            self.base_hunger + self.hunger_gain * (1 - self.e), a_min=0, a_max=1
        )
        if self.intermitter is not None:
            if self.hunger_as_EEB:
                self.intermitter.EEB = self.hunger

    @property
    def EEB(self) -> Optional[float]:
        if self.intermitter is None:
            return None
        else:
            return self.intermitter.EEB

    def update_dict(self) -> None:
        if self.dict is not None:
            dict_values = [
                self.age * 24,
                self.Ww * 1000,
                self.Lw * 10,
                self.E,
                self.e,
                self.hunger,
                self.pupation_buffer,
                self.f,
                self.deb_p_A / self.V,
                self.sim_p_A / self.V,
                self.EEB,
            ]
            for k, v in zip(self.dict.keylist, dict_values):
                self.dict[k].append(v)
            self.gut.update_dict()

    def finalize_dict(self) -> Dict[str, Any]:
        if self.dict is not None:
            d = self.dict
            d["species"] = self.species
            d["birth"] = self.birth_time_in_hours
            d["pupation"] = self.pupation_time_in_hours
            d["death"] = self.death_time_in_hours
            d["id"] = self.id
            d["epochs"] = self.epochs
            d["epoch_qs"] = self.epoch_qs
            d["fr"] = 1 / self.dt_in_sec
            d["feed_freq_estimate"] = self.fr_feed
            d["f_mean"] = np.mean(d["f"])
            d["f_deviation_mean"] = np.mean(np.array(d["f"]) - 1)
            d["Nfeeds"] = self.gut.Nfeeds
            d["mean_feed_freq"] = (
                self.gut.Nfeeds / (self.age - self.birth_time_in_hours) / (60 * 60)
            )
            d["gut_residence_time"] = self.gut.residence_time
            d.update(self.gut.dict)

            try:
                I = self.intermitter
                d["feed_freq_simulated"] = I.mean_feed_freq
                d_inter = I.build_dict()
                d.update(
                    {
                        **{
                            f"{q} ratio": np.round(d_inter[nam.dur_ratio(p)], 2)
                            for p, q in zip(
                                ["stridechain", "pause", "feedchain"],
                                ["crawl", "pause", "feed"],
                            )
                        },
                    }
                )
            except:
                pass

        return d

    def save_dict(self, path: Optional[str] = None) -> None:
        if path is None:
            if self.save_to is not None:
                path = self.save_to
            else:
                return
        if self.dict is not None:
            os.makedirs(path, exist_ok=True)
            d = {**self.dict, **self.gut.dict}
            util.save_dict(d, f"{path}/{self.id}.txt")

    @classmethod
    def default_growth(
        cls, id: str = "DEB default", life_history: Any | None = None, **kwargs: Any
    ) -> Dict[str, Any]:
        if life_history is None:
            life_history = Life.from_epoch_ticks(reach_pupation=True)
        d = cls(id=id, **kwargs)
        d.grow_larva(epochs=life_history.epochs)
        return d.finalize_dict()

    def run_larva_stage_offline(self, intermitter: Any) -> None:
        I = intermitter
        assert I is not None
        cum_feeds = 0
        while self.stage == "larva":
            I.step()
            self.run_check(dt=I.dt, X_V=self.V_bite * self.V * (I.Nfeeds - cum_feeds))
            cum_feeds = I.Nfeeds

    @classmethod
    def sim_run(
        cls,
        refID: Optional[str] = None,
        id: str = "DEB sim",
        EEB: Optional[float] = None,
        substrate: Substrate = Substrate(type="standard"),
        **kwargs: Any,
    ) -> Dict[str, Any]:
        from ... import reg

        if refID is None:
            refID = reg.default_refID
        c = reg.conf.Ref.getRef(refID)
        kws2 = c.intermitter
        if EEB is None:
            EEB = DEB_basic(substrate=substrate, **kwargs).get_best_EEB(c)
        kws2["EEB"] = EEB
        from ..modules.intermitter import OfflineIntermitter

        d = cls(
            id=id,
            assimilation_mode="gut",
            substrate=substrate,
            intermitter=OfflineIntermitter(**kws2),
            **kwargs,
        )
        d.run_stage(stage="embryo")
        d.run_larva_stage_offline(intermitter=d.intermitter)
        return d.finalize_dict()
