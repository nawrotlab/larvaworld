"""
DEB energetics wired into a larvaworld agent.

The physics lives in :mod:`~larvaworld.lib.model.deb.deb_equations`, which is a
transcription of the authoritative model tables. This module is the larvaworld-side
wrapper around it: parameter selection, the gut coupling, the hunger drive that
feeds the behavioural intermitter, the buffered stepping protocol the simulation
loop uses, and the recorded output dict.

Three layers, kept distinct because the registry relies on it:

``DEB_model``
    Holds only what is *not* part of a stored model configuration. Every parameter
    declared here is excluded from persisted configs by
    ``module_modes.energetics_kws``, which calls
    ``class_defaults(DEB, excluded=[DEB_model, "substrate", "id"])``. The
    physiological parameterisation is therefore an internal concern, carried by a
    :class:`~larvaworld.lib.model.deb.deb_equations.DEBPars` instance rather than by
    ``param`` attributes.
``DEB_basic``
    State, stage machine, gut coupling and the stepping protocol.
``DEB``
    Hunger/EEB coupling and trajectory recording.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import os

import numpy as np
import param

from ... import util
from ...param import (
    ClassAttr,
    Life,
    NestedConf,
    PositiveNumber,
    Substrate,
)
from ...util import nam
from . import Gut
from . import deb_equations as de
from .rover_sitter_model import SPECIES, get_species_pars

__all__: list[str] = [
    "DEB_model",
    "DEB_basic",
    "DEB",
]


#: Parameters read straight off the underlying :class:`DEBPars`. They are exposed as
#: read-only properties so that ``DEBPars`` stays the single source of truth while
#: ``gut.py``, the collectors and the plots keep the attribute names they have
#: always used.
_DELEGATED: tuple[str, ...] = (
    # primary
    "T_ref",
    "T_A",
    "z",
    "v",
    "kap",
    "p_M",
    "p_T",
    "k_J",
    "E_G",
    "E_Hb",
    "E_Hp",
    "E_He",
    "kap_R",
    "kap_X",
    "kap_P",
    "kap_V",
    "F_m",
    "del_M",
    "del_Mw",
    "h_a",
    "s_G",
    # chemistry
    "mu_X",
    "mu_V",
    "mu_E",
    "mu_P",
    "d_X",
    "d_V",
    "d_E",
    "d_P",
    "w_X",
    "w_V",
    "w_E",
    "w_P",
    # compound
    "p_Am",
    "E_m",
    "g",
    "k_M",
    "k",
    "L_m",
    "L_T",
    "l_T",
    "M_V",
    "y_V_E",
    "y_E_V",
    "m_Em",
    "kap_G",
    "E_V",
    "y_E_X",
    "y_X_E",
    "y_P_X",
    "y_X_P",
    "p_Xm",
    "J_E_Am",
    "J_X_Am",
    "K",
    "l_b",
    "E_0",
    "U_coeff",
    "v_Hb",
    "v_Hp",
    "v_He",
)

#: Legacy spellings kept alive because ``gut.py``, the collectors and downstream
#: code use them. Maps the larvaworld name to the DEBtool name on ``DEBPars``.
_ALIASES: dict[str, str] = {
    "E_M": "E_m",  # [E_m], reserve capacity
    "Lm": "L_m",
    "lb": "l_b",
    "E0": "E_0",
    "Ucoeff": "U_coeff",
    "vHb": "v_Hb",
    "vHe": "v_He",
}


def _delegate(name: str) -> property:
    def getter(self: "DEB_model") -> Any:
        return getattr(self.pars, name)

    getter.__name__ = name
    return property(getter, doc=f"``DEBPars.{name}`` of the active parameter set.")


class DEB_model(NestedConf):
    """
    Base of the DEB hierarchy, holding only non-configuration state.

    Everything declared here is excluded from stored larva-model configurations by
    construction, so nothing physiological belongs in a ``param`` attribute. The
    parameter set itself is a :class:`DEBPars`, selected by ``species`` on
    :class:`DEB_basic` and exposed through read-only delegating properties.

    Reference:
        Kooijman (2010). "Dynamic Energy Budget theory for metabolic organisation."

    Example:
        >>> deb = DEB_basic(species="rover")
        >>> deb.p_Am, deb.E_M          # doctest: +SKIP
    """

    T = PositiveNumber(298.15, doc="The ambient temperature (K)")

    def __init__(self, print_output: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.print_output = print_output
        self.stages = list(de.STAGES)
        self.stage_events = [
            "oviposition",
            "eclosion",
            "pupation",
            "emergence",
            "death",
        ]
        self._life_history: Optional[de.LifeHistory] = None

    # -- parameter set -------------------------------------------------------

    @property
    def pars(self) -> de.DEBPars:
        """The active :class:`DEBPars`. Shared and cached -- do not mutate."""
        return get_species_pars(getattr(self, "species", "default"))

    @property
    def T_factor(self) -> float:
        """Arrhenius temperature correction from ``T_ref`` to ``T``."""
        return de.temperature_correction(self.pars, self.T)

    # -- predicted life history (lazy: constructing a DEB must stay cheap) ----

    @property
    def life_history(self) -> de.LifeHistory:
        """
        Whole-life-cycle prediction at constant food, computed on first use.

        Replaces the legacy ``predict_*_stage`` chain. It is deliberately lazy:
        ``class_defaults`` builds a ``DEB`` every time the model registry resolves
        defaults, and a closed-form solve in ``__init__`` would make that expensive.
        """
        if self._life_history is None:
            self._life_history = de.run_life_cycle(
                self.pars, engine="closed", dt=1.0 / 24.0, f=1.0, T=self.T
            )
            if self.print_output:
                print(de.format_life_history(self._life_history))
        return self._life_history

    @property
    def t_b(self) -> float:
        """Predicted age at hatching (d)."""
        return self.life_history.age_at_birth

    @property
    def t_j(self) -> float:
        """Predicted time from hatching to pupation (d)."""
        return self.life_history.time_to_pupation

    @property
    def Lb(self) -> float:
        """
        Structural length at hatching (cm).

        The value actually reached once the individual has hatched, otherwise the
        prediction. ``gut.py`` divides by it to get volume-specific feeding rates.
        """
        L_b = getattr(getattr(self, "_state", None), "L_b", None)
        if L_b is not None and np.isfinite(L_b):
            return L_b
        return self.pars.L_b_pred

    @property
    def Lj(self) -> float:
        """Structural length at pupation (cm) -- reached if known, else predicted."""
        L_p = getattr(getattr(self, "_state", None), "L_p", None)
        if L_p is not None and np.isfinite(L_p):
            return L_p
        return self.life_history.L_p

    def compute_Ww(self, V: float, E: float) -> float:
        """
        Dry weight (g) of an individual with structure ``V`` and reserve ``E``.

        Kept under its historical name because ``LarvaMotile.mass`` and the recorded
        ``mass`` series read it. Both terms are dry mass; see
        :func:`~larvaworld.lib.model.deb.deb_equations.wet_weight` for the
        Kooijman wet-weight convention, which is roughly ``1/d_V`` larger.
        """
        return de.dry_weight(self.pars, V, E)

    def print_life_history(self) -> None:
        """Print the predicted life history as a table."""
        print(de.format_life_history(self.life_history))


for _name in _DELEGATED:
    setattr(DEB_model, _name, _delegate(_name))
for _alias, _target in _ALIASES.items():
    setattr(DEB_model, _alias, _delegate(_target))
del _name, _alias, _target


class DEB_basic(DEB_model):
    """
    DEB state, stage machine, gut coupling and the stepping protocol.

    Attributes:
        id: Model identifier
        species: Which parameter set to use (see ``rover_sitter_model.SPECIES``)
        dt: DEB timestep in days (default: one minute)
        substrate: The substrate the agent feeds on
        assimilation_mode: How the assimilation flux is obtained -- from the gut
            model, from the simulation's functional response, or from the DEB
            functional response alone.

    Example:
        >>> deb = DEB_basic(species="rover", dt=1/1440)   # doctest: +SKIP
    """

    id = param.String("DEB model", doc="The unique ID of the DEB model")
    species = param.Selector(
        objects=list(SPECIES),
        label="phenotype",
        doc="The species-specific or phenotype-specific DEB parameter set to use.",
    )
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
        super().__init__(species=species, **kwargs)

        self._state = de.initial_state(self.pars)
        #: Age (d) at which each stage was entered, filled in as they happen.
        self._events: Dict[str, float] = {self._state.stage: self._state.age}
        self.epochs: List[Any] = []
        self.epoch_qs: List[Any] = []
        self.deb_p_A = 0.0
        self.sim_p_A = 0.0

        self.base_f = self.substrate.get_f(K=self.K)
        self.f = self.base_f
        self.V_bite = V_bite
        self.X_V_buffer = 0.0
        self.time_buffer = 0.0

        self.gut = Gut(deb=self, save_dict=save_dict, **(gut_params or {}))

    # -- state ---------------------------------------------------------------

    @property
    def E(self) -> float:
        """Reserve (J)."""
        return self._state.E

    @property
    def V(self) -> float:
        """Structural volume (cm^3)."""
        return self._state.V

    @property
    def E_H(self) -> float:
        """Maturity (J). Resets to zero at pupation."""
        return self._state.E_H

    @property
    def E_R(self) -> float:
        """Reproduction buffer (J). Accumulates only in the imago."""
        return self._state.E_R

    @property
    def age(self) -> float:
        """Age since oviposition (d)."""
        return self._state.age

    @property
    def L(self) -> float:
        """Structural length (cm)."""
        return self._state.L

    @property
    def Lw(self) -> float:
        """Physical length (cm)."""
        return self.L / self.del_M

    @property
    def Ww(self) -> float:
        """Dry weight (g) of structure plus reserve."""
        return self.compute_Ww(V=self.V, E=self.E)

    @property
    def e(self) -> float:
        """Scaled reserve density ``E / (V [E_m])``. Converges to ``f``."""
        return self.E / self.V / self.E_M if self.V > 0 else 0.0

    @property
    def alive(self) -> bool:
        return self._state.alive

    @property
    def stage(self) -> str:
        """
        Current life stage, or ``"dead"``.

        Explicit state rather than a function of maturity: maturity resets to zero
        at pupation, so ``E_H`` alone can no longer identify the stage.
        """
        return self._state.stage if self.alive else "dead"

    @property
    def pupation_buffer(self) -> float:
        """
        Progress towards pupation, in ``[0, 1]``.

        Maturity relative to the pupation threshold, pinned at 1 once pupation has
        happened. Under the ground-truth dynamics the reproduction buffer stays at
        zero until the imago, so it can no longer serve as this indicator.
        """
        if self._state.stage in (de.Stage.PUPA, de.Stage.IMAGO):
            return 1.0
        return float(np.clip(self.E_H / self.E_Hp, 0.0, 1.0))

    @property
    def dt_in_sec(self) -> float:
        return self.dt * 24 * 60 * 60

    @property
    def steps_per_day(self) -> int:
        return int(1 / self.dt)

    @property
    def Vw(self) -> float:
        """Wet volume (cm^3) of structure plus reserve."""
        return de.wet_weight(self.pars, self.V, self.E)

    @property
    def time_to_death_by_starvation(self) -> float:
        """Rough time (d) to exhaust reserve at zero food."""
        return self.v**-1 * self.L * np.log(self.kap**-1)

    # -- fluxes and stepping -------------------------------------------------

    def get_p_A(
        self,
        f: Optional[float] = None,
        assimilation_mode: Optional[str] = None,
        X_V: float = 0.0,
    ) -> float:
        """
        Assimilation flux in J/d, by the selected mode.

        ``"deb"`` uses the substrate's own functional response, ``"sim"`` the one the
        simulation supplies, and ``"gut"`` the flux the gut model actually absorbed.

        Note the unit: fluxes are per day throughout, matching ``deb_equations``.
        The gut accumulates joules over one DEB timestep, so its total is divided by
        ``dt`` to become a rate.
        """
        if f is None:
            f = self.base_f
        self.f = f

        s_M = self._state.s_M()
        TC = self.T_factor
        base = TC * self.p_Am * s_M * self.V ** (2.0 / 3.0)
        self.deb_p_A = base * self.base_f
        self.sim_p_A = base * f

        if assimilation_mode is None:
            assimilation_mode = self.assimilation_mode
        if assimilation_mode == "sim":
            return self.sim_p_A
        if assimilation_mode == "gut":
            self.gut.update(X_V)
            return self.gut.p_A / self.dt
        return self.deb_p_A

    def apply_fluxes(self, **kwargs: Any) -> None:
        """
        Advance the state by one DEB timestep, applying Table S1/S2.

        The egg and the pupa do not feed, so no assimilation flux is supplied for
        them: passing one would override the zero Table S1 prescribes.
        """
        if self._state.stage in (de.Stage.EGG, de.Stage.PUPA):
            de.step(self._state, self.pars, dt=self.dt, T=self.T)
        else:
            p_A = self.get_p_A(**kwargs)
            de.step(self._state, self.pars, dt=self.dt, T=self.T, p_A=p_A)

    def run(self, **kwargs: Any) -> None:
        """Advance one DEB timestep, whatever the stage."""
        if self.alive:
            self.apply_fluxes(**kwargs)
        self.update()

    def run_stage(
        self, stage: str, assimilation_mode: str = "deb", **kwargs: Any
    ) -> float:
        """Run until the given stage ends. Returns the elapsed time in days."""
        stage = de.resolve_stage(stage)
        t = 0.0
        while self.stage == stage and self.alive:
            self.apply_fluxes(assimilation_mode=assimilation_mode, **kwargs)
            t += self.dt
            self.update()
        return t

    def run_life_history(self, **kwargs: Any) -> None:
        """Step through every stage in turn until death or the imago."""
        for _ in range(len(de.STAGES) + 1):
            before = self.stage
            if not self.alive or before == de.Stage.IMAGO:
                break
            self.run_stage(before, **kwargs)
            if self.stage == before:
                break
        if self.print_output:
            self.print_life_history()

    def run_check(self, dt: float, X_V: float = 0) -> None:
        """
        Accumulate a simulation tick, stepping the DEB when its timestep elapses.

        The simulation runs at 0.1 s while the DEB steps at 60 s, so ingested volume
        and elapsed time are buffered in between.
        """
        self.X_V_buffer += X_V
        self.time_buffer += dt
        if self.time_buffer >= self.dt_in_sec:
            self.run(X_V=self.X_V_buffer)
            self.X_V_buffer = 0
            self.time_buffer = 0

    def update(self) -> None:
        """Hook run after every step. Records the age at each stage transition."""
        self._events.setdefault(self._state.stage, self._state.age)

    def age_at(self, stage: str) -> Optional[float]:
        """Age (d) at which ``stage`` was entered, or None if it has not been."""
        return self._events.get(de.resolve_stage(stage))

    def grow_larva(self, epochs: List[Any], **kwargs: Any) -> None:
        """Age the individual through the embryo stage and the supplied epochs."""
        self.run_stage(stage=de.Stage.EGG)
        tb = self.age * 24
        for e in epochs:
            if self.stage != de.Stage.LARVA:
                continue
            c = {"assimilation_mode": "sim", "f": e.substrate.get_f(K=self.K)}
            if e.end is None:
                self.run_stage(stage=de.Stage.LARVA, **c)
            else:
                for _ in range(e.ticks(self.dt)):
                    if self.stage == de.Stage.LARVA:
                        self.run(**c)
            self.epochs.append([e.start + tb, self.age * 24])
            self.epoch_qs.append(e.substrate.quality)

    # -- feeding-related observables ----------------------------------------

    @property
    def J_X_A(self) -> float:
        return self.J_X_Am / self.Lb * self.V * self.base_f

    @property
    def F(self) -> float:
        """Volume-specific filtering rate (cm^3 of environment per cm^3 per day)."""
        return (
            self.J_X_Am
            * self.F_m
            / (self.Lb * (self.J_X_Am + self.substrate.X * self.F_m))
        )

    @property
    def fr_feed(self) -> float:
        freq = self.F / self.V_bite * self.T_factor
        return freq / (24 * 60 * 60)

    def get_best_EEB(self, cRef: Dict[str, Any]) -> float:
        z = np.poly1d(cRef["EEB_poly1d"])
        return np.clip(z(self.fr_feed), a_min=0, a_max=1)

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
    Full DEB model with the hunger drive and trajectory recording.

    Couples energetics to behaviour through the exploration-exploitation balance:
    reserve depletion raises hunger, which the behavioural intermitter reads as its
    EEB. Records every state variable over time for later analysis.

    Attributes:
        hunger_as_EEB: Whether hunger modulates the intermitter's EEB
        hunger_gain: Sensitivity of hunger to reserve depletion
        dict: Recorded timeseries, or None when recording is off
        save_to: Where :meth:`save_dict` writes

    Example:
        >>> deb = DEB(species="rover", hunger_as_EEB=True)   # doctest: +SKIP
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
        super().__init__(save_dict=save_dict, **kwargs)
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
        if self.intermitter is not None and self.hunger_as_EEB:
            base_hunger = self.intermitter.base_EEB
        self.base_hunger = base_hunger
        self.update_hunger()

    def update(self) -> None:
        super().update()
        self.update_hunger()
        self.update_dict()

    def update_hunger(self) -> None:
        self.hunger = np.clip(
            self.base_hunger + self.hunger_gain * (1 - self.e), a_min=0, a_max=1
        )
        if self.intermitter is not None and self.hunger_as_EEB:
            self.intermitter.EEB = self.hunger

    @property
    def EEB(self) -> Optional[float]:
        return None if self.intermitter is None else self.intermitter.EEB

    # -- life-event times ----------------------------------------------------

    @property
    def birth_time_in_hours(self) -> float:
        """Age at hatching (h): what actually happened, else the prediction."""
        t = self.age_at(de.Stage.LARVA)
        return np.round((self.t_b if t is None else t) * 24, 1)

    @property
    def pupation_time_in_hours(self) -> float:
        """
        Age at pupation (h): what actually happened, else the prediction.

        A simulated larva rarely reaches pupation -- experiments run for minutes
        while pupation takes days -- so the predicted duration is the usual answer.
        The GUI life-history tab uses ``pupation - birth`` as a slider range.
        """
        t = self.age_at(de.Stage.PUPA)
        if t is not None:
            return np.round(t * 24, 1)
        return self.birth_time_in_hours + np.round(self.t_j * 24, 1)

    @property
    def death_time_in_hours(self) -> float:
        return self.age * 24 if not self.alive else np.nan

    # -- recording -----------------------------------------------------------

    def update_dict(self) -> None:
        if self.dict is None:
            return
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
        d = self.dict
        if d is None:
            return {}
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
                    f"{q} ratio": np.round(d_inter[nam.dur_ratio(p)], 2)
                    for p, q in zip(
                        ["stridechain", "pause", "feedchain"],
                        ["crawl", "pause", "feed"],
                    )
                }
            )
        except Exception:
            pass

        return d

    def save_dict(self, path: Optional[str] = None) -> None:
        if path is None:
            path = self.save_to
        if path is None or self.dict is None:
            return
        os.makedirs(path, exist_ok=True)
        util.save_dict({**self.dict, **self.gut.dict}, f"{path}/{self.id}.txt")

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
        while self.stage == de.Stage.LARVA:
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
        from ..modules.intermitter import OfflineIntermitter

        if refID is None:
            refID = reg.default_refID
        c = reg.conf.Ref.getRef(refID)
        kws2 = c.intermitter
        if EEB is None:
            EEB = DEB_basic(substrate=substrate, **kwargs).get_best_EEB(c)
        kws2["EEB"] = EEB

        d = cls(
            id=id,
            assimilation_mode="gut",
            substrate=substrate,
            intermitter=OfflineIntermitter(**kws2),
            **kwargs,
        )
        d.run_stage(stage=de.Stage.EGG)
        d.run_larva_stage_offline(intermitter=d.intermitter)
        return d.finalize_dict()
