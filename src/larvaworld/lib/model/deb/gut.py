"""
The gut model, tracking ingestion, digestion and absorption.

Food taken in by the feeder passes through the gut before it becomes reserve.
This module models that transit, the residence time and the absorption
efficiency that determine how much of the ingested food reaches the DEB
reserve.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import param

from ... import util
from ...param import NestedConf, PositiveNumber

__all__: list[str] = [
    "Gut",
]


class Gut(NestedConf):
    """
    Gut digestion and absorption model for DEB energetics.

    Models gut dynamics including ingestion, enzymatic digestion, carrier-mediated
    absorption, and faecal production. Tracks gut content (food mass M_X, digested
    product M_P), enzyme dynamics (M_g), and carrier availability (M_c).

    Integrates with DEB model to compute assimilation energy flow (p_A) based
    on gut absorption rather than simplified functional response.

    Attributes:
        M_gm: Gut capacity per unit volume (C-moles, default: 0.01)
        k_abs: Absorption rate constant (default: 1.0)
        k_dig: Digestion rate constant (default: 1.0)
        k_c: Carrier release rate (default: 1.0)
        k_g: Enzyme decay rate (default: 1.0)
        constant_M_c: Keep carrier density constant (default: True)
        M_X: Current food mass in gut (C-moles)
        M_P: Current digested product in gut (C-moles)
        p_A: Assimilation power from gut absorption (J/day)

    Example:
        >>> gut = Gut(deb=deb_model, save_dict=True)
        >>> gut.update(V_X=0.001)  # Ingest food volume
        >>> power = gut.p_A  # Get assimilation power
    """

    M_gm = PositiveNumber(10**-2, doc="gut capacity in C-moles for unit of gut volume")
    r_w2l = param.Magnitude(0.2, doc="body width to length ratio")
    r_gut_w = param.Magnitude(0.7, doc="gut width relative to body width")
    y_P_X = param.Magnitude(0.9, doc="yield of product by food")
    k_abs = PositiveNumber(1.0, doc="rate constant for absorption : k_P * y_Pc")
    f_abs = param.Magnitude(
        1.0, doc="scaled functional response for absorption : M_P/(M_P+M_K_P)"
    )
    k_dig = PositiveNumber(1.0, doc="rate constant for digestion : k_X * y_Xg")
    f_dig = param.Magnitude(
        1.0, doc="scaled functional response for digestion : M_X/(M_X+M_K_X)"
    )
    k_c = PositiveNumber(1.0, doc="release rate of carriers")
    k_g = PositiveNumber(1.0, doc="decay rate of enzyme")
    M_c_per_cm2 = PositiveNumber(
        5 * 10**-8,
        doc="area specific amount of carriers in the gut per unit of gut surface",
    )
    J_g_per_cm2 = PositiveNumber(
        10**-2 / (24 * 60 * 60),
        doc="secretion rate of enzyme per unit of gut surface per day",
    )
    constant_M_c = param.Boolean(
        True,
        label="constant carrier density",
        doc="Whether to assume a constant amount of carrier enzymes on the gut surface.",
    )

    def __init__(self, deb: Any, save_dict: bool = True, **kwargs: Any) -> None:
        """Attach a gut to a DEB model.

        Args:
            deb: The DEB model this gut belongs to. Its size sets the gut
                capacity and its substrate sets the food composition.
            save_dict: When True, record the gut state over time.
            **kwargs: Gut parameters, forwarded to the parent class.
        """
        super().__init__(**kwargs)
        self.deb = deb
        # Arbitrary parameters
        r_gut_w2L = (
            0.5 * self.r_w2l * self.r_gut_w
        )  # gut radius relative to body length
        self.r_gut_V = np.pi * r_gut_w2L  # gut volume per unit of body volume
        self.r_gut_A = (
            2 * np.pi * r_gut_w2L
        )  # gut surface area per unit of body surface area
        self.A_g = self.r_gut_A * self.deb.L**2  # Gut surface area
        self.V_gm = self.r_gut_V * self.deb.V

        self.M_c_max = (
            self.M_c_per_cm2 * self.A_g
        )  # amount of carriers in the gut surface
        self.J_g = (
            self.J_g_per_cm2 * self.A_g
        )  # total secretion rate of enzyme in the gut surface

        self.M_X = 0
        self.M_P = 0
        self.M_Pu = 0
        self.M_g = 0
        self.M_c = self.M_c_max

        self.mol_not_digested = 0
        self.mol_not_absorbed = 0
        self.mol_faeces = 0
        self.p_A = 0
        self.mol_ingested = 0
        self.V = 0
        self.gut_X = 0
        self.gut_f = 0
        self.Nfeeds = 0

        if save_dict:
            self.dict = util.AttrDict(
                {
                    k: []
                    for k in [
                        "R_absorbed",
                        "mol_ingested",
                        "gut_p_A",
                        "M_X",
                        "M_P",
                        "M_Pu",
                        "M_g",
                        "M_c",
                        "R_M_c",
                        "R_M_g",
                        "R_M_X",
                        "R_M_P",
                        "R_M_X_M_P",
                    ]
                }
            )
        else:
            self.dict = None

    def update(self, V_X: float = 0) -> None:
        """Advance the gut by one timestep, optionally ingesting food.

        Rescales the gut capacity to the animal's current size, adds any newly
        ingested food, then runs digestion and enforces the capacity limit.

        Args:
            V_X: Volume of food ingested this timestep. Zero means no feeding
                motion occurred.
        """
        self.A_g = self.r_gut_A * self.deb.L**2  # Gut surface area
        self.V_gm = self.r_gut_V * self.deb.V
        self.M_c_max = (
            self.M_c_per_cm2 * self.A_g
        )  # amount of carriers in the gut surface
        self.J_g = (
            self.J_g_per_cm2 * self.A_g
        )  # total secretion rate of enzyme in the gut surface

        if V_X > 0:
            self.Nfeeds += 1
            self.V += V_X
            M_X_in = self.deb.substrate.X * V_X
            self.mol_ingested += M_X_in
            self.M_X += M_X_in
        self.digest()
        self.resolve_occupancy()

    def resolve_occupancy(self) -> None:
        """Void the gut contents that exceed its capacity.

        The excess is removed from the undigested food and the digested product
        in proportion to their current amounts, and accounted as not digested
        and not absorbed respectively.
        """
        dM = self.M_X + self.M_P - self.Cmax
        if dM > 0:
            rP = self.M_P / (self.M_P + self.M_X)
            dP = rP * dM
            self.M_P -= dP
            self.mol_not_absorbed += dP
            dX = (1 - rP) * dM
            self.M_X -= dX
            self.mol_not_digested += dX

    def digest(self) -> None:
        """Run one timestep of enzyme secretion, digestion and absorption.

        Food is converted to product at a rate set by the accumulated enzyme,
        and product is absorbed through the gut carriers. Absorption consumes
        carriers, which are replenished towards their maximum unless the gut is
        configured to hold them constant. The absorbed product sets the
        assimilation flux the DEB model reads.

        Notes:
            The enzyme balance carries a FIXME in the source: the secretion
            term is expected to be scaled by the gut surface area a second
            time. Left as-is; changing it would alter model output.
        """
        dt = self.deb.dt * 24 * 60 * 60
        # print(dt)
        # FIXME there should be another term A_g after J_g
        self.M_g += self.J_g * dt - self.k_g * self.M_g
        if self.M_X > 0:
            temp = self.k_dig * self.f_dig * self.M_g
            dM_X = -np.min([self.M_X, temp])
        else:
            dM_X = 0
        self.M_X += dM_X
        dM_P_added = -self.y_P_X * dM_X
        if self.M_P > 0 and self.M_c > 0:
            temp = self.k_abs * self.f_abs * self.M_c * dt
            dM_Pu = np.min([self.M_P, temp])
        else:
            dM_Pu = 0
        self.M_P += dM_P_added - dM_Pu
        self.M_Pu += dM_Pu

        if self.constant_M_c:
            self.M_c = self.M_c_max
        else:
            dM_c_released = (self.M_c_max - self.M_c) * self.k_c * dt
            dM_c = dM_c_released - dM_Pu
            self.M_c += dM_c
        self.p_A = dM_Pu * self.deb.mu_E

    @property
    def residence_time(self) -> float:
        """The mean time food spends in the gut, in seconds.

        Zero when the functional response is zero, since nothing is ingested.
        """
        f = self.deb.base_f
        if f == 0.0:
            return 0.0
        else:
            return self.r_gut_V * self.M_gm / (self.deb.J_X_Am / self.deb.Lb) / f

    @property
    def M_ingested(self) -> float:
        """The total mass of food ingested so far, in milligrams."""
        return self.mol_ingested * self.deb.substrate.get_w_X() * 1000

    @property
    def M_faeces(self) -> float:
        """The total mass of faeces produced so far, in milligrams."""
        return self.mol_faeces * self.deb.w_P * 1000

    @property
    def M_not_digested(self) -> float:
        """The mass of food voided before being digested, in milligrams."""
        return self.mol_not_digested * self.deb.substrate.get_w_X() * 1000

    @property
    def M_not_absorbed(self) -> float:
        """The mass of product voided before being absorbed, in milligrams."""
        return self.mol_not_absorbed * self.deb.w_P * 1000

    @property
    def R_absorbed(self) -> float:
        """The fraction of ingested food that has been absorbed."""
        return self.M_Pu / self.mol_ingested if self.mol_ingested != 0 else 0

    @property
    def R_faeces(self) -> float:
        """The fraction of ingested food excreted as faeces."""
        return self.mol_faeces / self.mol_ingested if self.mol_ingested != 0 else 0

    @property
    def R_not_digested(self) -> float:
        """The fraction of ingested food voided without being digested."""
        return (
            self.mol_not_digested / self.mol_ingested if self.mol_ingested != 0 else 0
        )

    @property
    def R_M_c(self) -> float:
        """The fraction of the gut's absorption carriers currently available."""
        return self.M_c / self.M_c_max

    @property
    def R_M_g(self) -> float:
        """The accumulated digestive enzyme, relative to its secretion rate."""
        return self.M_g / self.J_g

    @property
    def R_M_X_M_P(self) -> float:
        """The ratio of undigested food to digested product in the gut."""
        return self.M_X / self.M_P if self.M_P != 0 else 0

    @property
    def R_M_X(self) -> float:
        """The undigested food in the gut, as a fraction of its capacity."""
        return self.M_X / self.Cmax

    @property
    def R_M_P(self) -> float:
        """The digested product in the gut, as a fraction of its capacity."""
        return self.M_P / self.Cmax

    @property
    def occupancy(self) -> float:
        """The gut's filled volume, as a fraction of its maximum."""
        return self.V / self.Vmax

    @property
    def M(self) -> float:
        """The mass of the gut contents, in milligrams."""
        return self.V * self.deb.d_V * 1000

    @property
    def Vmax(self) -> float:
        """The gut's volume capacity, scaled to the animal's current size."""
        return self.V_gm * self.deb.V

    @property
    def Cmax(self) -> float:  # in mol
        """The gut's content capacity in moles."""
        return self.M_gm * self.V_gm

    def update_dict(self) -> None:
        """Append the current gut state to the recorded trajectory."""
        gut_dict_values = [
            self.R_absorbed,
            self.mol_ingested * 1000,
            self.p_A / self.deb.V,
            self.M_X,
            self.M_P,
            self.M_Pu * 1000,
            self.M_g,
            self.M_c,
            self.R_M_c,
            self.R_M_g,
            self.R_M_X,
            self.R_M_P,
            self.R_M_X_M_P,
        ]
        for k, v in zip(self.dict.keylist, gut_dict_values):
            self.dict[k].append(v)

    def ingested_mass(self, unit: str = "g") -> float:
        """The total mass of food ingested so far.

        Args:
            unit: ``"g"`` for grams, or ``"mg"`` for milligrams.

        Returns:
            The ingested mass in the requested unit.
        """
        m = self.mol_ingested * self.deb.substrate.get_w_X()
        if unit == "g":
            return m
        elif unit == "mg":
            return m * 1000

    def absorbed_mass(self, unit: str = "mg") -> float:
        """The total mass absorbed into reserve so far.

        Args:
            unit: ``"g"`` for grams, or ``"mg"`` for milligrams.

        Returns:
            The absorbed mass in the requested unit.
        """
        m = self.M_Pu * self.deb.w_E
        if unit == "g":
            return m
        elif unit == "mg":
            return m * 1000

    @property
    def ingested_volume(self) -> float:
        """The total volume of food ingested so far."""
        return self.mol_ingested / self.deb.substrate.X
        # return self.mol_ingested * self.deb.w_X / self.deb.d_X
