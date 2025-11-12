from __future__ import annotations
import numpy as np
import param
from scipy.stats import multivariate_normal

from .. import util
from .custom import (
    ClassAttr,
    ItemListParam,
    OptionalPositiveInteger,
    OptionalPositiveNumber,
    OptionalPositiveRange,
    PositiveInteger,
    PositiveNumber,
    StringRobust,
)
from .nested_parameter_group import NestedConf, expand_kws_shortcuts

__all__: list[str] = [
    "Compound",
    "Substrate",
    "substrate_dict",
    "Odor",
    "Epoch",
    "Life",
    "AirPuff",
]

__displayname__ = "Nutrition & Olfaction"


class Compound(NestedConf):
    """
    Chemical compound parameter group for nutritional composition.

    Defines molecular properties and elemental composition for substrate
    compounds used in larva nutrition modeling.

    Attributes:
        d: Density in g/cm³
        w: Molecular weight in g/mol
        nC: Number of carbon atoms
        nH: Number of hydrogen atoms
        nO: Number of oxygen atoms
        nN: Number of nitrogen atoms
        ww: Computed weighted molecular mass

    Example:
        >>> glucose = Compound(w=180.18, nC=6, nH=12, nO=6)
        >>> glucose.ww  # Weighted mass computed automatically
    """

    d = PositiveNumber(doc="density in g/cm**3")
    w = PositiveNumber(doc="molecular weight (g/mol)")
    nC = PositiveInteger(doc="number of carbon atoms")
    nH = PositiveInteger(doc="number of hydrogen atoms")
    nO = PositiveInteger(doc="number of oxygen atoms")
    nN = PositiveInteger(doc="number of nitrogen atoms")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ww = self.nC + self.nH / 12 * 1 + self.nO / 12 * 16 + self.nN / 12 * 14


# Compounds
compound_dict = util.AttrDict(
    {
        "glucose": Compound(w=180.18, nC=6, nH=12, nO=6),
        "dextrose": Compound(w=198.17, nC=6, nH=12, nO=7),
        "saccharose": Compound(w=342.30, nC=12, nH=22, nO=11),
        "yeast": Compound(w=274.3, nC=19, nH=14, nO=2),
        "agar": Compound(w=336.33, nC=0, nH=38, nO=19),
        "cornmeal": Compound(w=359.33, nC=27, nH=48, nO=20),
        # 'apple_juice': Compound(w=180.18,nC=6, nH=12, nO=6),
        "water": Compound(w=18.01528, nC=0, nH=2, nO=1),
    }
)

all_compounds = [a for a in compound_dict if a not in ["water"]]
nutritious_compounds = [a for a in compound_dict if a not in ["water", "agar"]]


class Substrate(NestedConf):
    """
    Substrate nutritional composition parameter group.

    Models substrate nutrition with compound concentrations, quality degradation,
    and molar concentration calculations for feeding/growth simulations.

    Attributes:
        composition: Dict of compound densities (g/cm³) per compound type
        quality: Quality factor (0-1, default: 1.0) for nutrient degradation
        d: Total substrate density
        C: Total molar concentration
        X: Nutrient molar concentration
        X_ratio: Nutrient/total concentration ratio

    Example:
        >>> substrate = Substrate(type='standard', quality=0.8)
        >>> substrate.get_f(K=0.1)  # Feeding response function
    """

    composition = param.Dict(
        {k: 0.0 for k in all_compounds}, doc="The substrate composition"
    )
    quality = param.Magnitude(
        1.0,
        step=0.01,
        doc="The substrate quality as percentage of nutrients relative to the intact substrate type",
    )

    def __init__(self, quality=1.0, type=None, **kwargs):
        if type is not None and type in substrate_dict:
            composition = substrate_dict[type].composition
        else:
            composition = {k: kwargs[k] if k in kwargs else 0.0 for k in all_compounds}
        super().__init__(composition=composition, quality=quality)
        self.d_water = 1
        self.d_yeast_drop = 0.125  # g/cm**3 https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi3iaeqipLxAhVPyYUKHTmpCqMQFjAAegQIAxAD&url=https%3A%2F%2Fwww.mdpi.com%2F2077-0375%2F11%2F3%2F182%2Fpdf&usg=AOvVaw1qDlMHxBPu73W8B1vZWn76
        self.V_drop = 0.05  # cm**3
        self.d = self.d_water + sum(list(self.composition.values()))
        self.C = self.get_C()
        self.X = self.get_X()
        self.X_ratio = self.get_X_ratio()

    def get_d_X(self, compounds=None, quality=None) -> float:
        if quality is None:
            quality = self.quality
        if compounds is None:
            compounds = nutritious_compounds
        return sum([self.composition[c] for c in compounds]) * quality

    def get_w_X(self, compounds=None) -> float:
        if compounds is None:
            compounds = nutritious_compounds
        d_X = self.get_d_X(compounds, quality=1)
        if d_X > 0:
            return (
                sum([self.composition[c] * compound_dict[c].ww for c in compounds])
                / d_X
            )
        else:
            return 0.0

    def get_X(self, quality=None, compounds=None) -> float:
        if quality is None:
            quality = self.quality
        if compounds is None:
            compounds = nutritious_compounds
        d_X = self.get_d_X(compounds, quality)
        if d_X > 0:
            return d_X / self.get_w_X(compounds)
        else:
            return 0.0

    def get_mol(self, V, **kwargs) -> float:
        return self.get_X(**kwargs) * V

    def get_f(self, K, **kwargs) -> float:
        X = self.get_X(**kwargs)
        return X / (K + X)

    def get_C(self, quality=None) -> float:
        return self.d_water / compound_dict["water"].w + self.get_X(
            quality, compounds=all_compounds
        )

    def get_X_ratio(self, **kwargs) -> float:
        return self.get_X(**kwargs) / self.get_C(**kwargs)


# Standard culture medium
# 50g Baker’s yeast; 100g sucrose; 16g agar; 0.1gKPO4; 8gKNaC4H4O6·4H2O; 0.5gNaCl; 0.5gMgCl2; and 0.5gFe2(SO4)3 per liter of tap water.
# Larvae were reared from egg-hatch to mid- third-instar (96±2h post-hatch) in 25°C at densities of 100 larvae per 35ml of medium in 100mm⫻15mm Petri dishes
# [1] K. R. Kaun, M. Chakaborty-Chatterjee, and M. B. Sokolowski, “Natural variation in plasticity of glucose homeostasis and food intake,” J. Exp. Biol., vol. 211, no. 19, pp. 3160–3166, 2008.
# --> 0.35 ml medium per larva for the 4 days

# Compound densities (g/cm**3)
substrate_dict = util.AttrDict(
    {
        "agar": Substrate(agar=0.016),
        "standard": Substrate(glucose=0.1, yeast=0.05, agar=0.016),
        "sucrose": Substrate(glucose=0.0171, agar=0.004),
        "cornmeal": Substrate(
            glucose=517 / 17000,
            dextrose=1033 / 17000,
            cornmeal=1716 / 17000,
            agar=93 / 17000,
        ),
        "cornmeal2": Substrate(
            dextrose=450 / 6400, yeast=90 / 6400, cornmeal=420 / 6400, agar=42 / 6400
        ),
        # [1] M. E. Wosniack, N. Hu, J. Gjorgjieva, and J. Berni, “Adaptation of Drosophila larva foraging in response to changes in food distribution,” bioRxiv, p. 2021.06.21.449222, 2021.
        "PED_tracker": Substrate(saccharose=0.01, yeast=0.1875, agar=5),
        # 'apple_juice': Substrate(glucose=0.00171, agar=0.004, apple_juice=0.02625)
    }
)


class Odor(NestedConf):
    """
    Odor stimulus parameter group for olfactory experiments.

    Defines odorant identity, concentration gradient (Gaussian distribution),
    and provides concentration computation at spatial positions.

    Attributes:
        id: Unique odorant identifier
        intensity: Peak concentration in micromoles (optional)
        spread: Gradient spread (standard deviation, optional)
        dist: Multivariate normal distribution (auto-computed)
        peak_value: Peak concentration value (auto-computed)

    Example:
        >>> odor = Odor(id='odorA', intensity=2.0, spread=0.01)
        >>> odor.gaussian_value([0.005, 0.005])  # Concentration at position
    """

    id = StringRobust(None, doc="The unique ID of the odorant")
    intensity = OptionalPositiveNumber(
        softmax=10.0, doc="The peak concentration of the odorant in micromoles"
    )
    spread = OptionalPositiveNumber(
        softmax=10.0, doc="The spread of the concentration gradient around the peak"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._update_distro()

    @param.depends("intensity", "spread", watch=True)
    def _update_distro(self) -> None:
        if self.intensity is not None and self.spread is not None:
            self.dist = multivariate_normal(
                [0, 0], [[self.spread, 0], [0, self.spread]]
            )
            self.peak_value = self.intensity / self.dist.pdf([0, 0])
        else:
            self.dist = None
            self.peak_value = 0.0

    def gaussian_value(self, pos) -> float | None:
        if self.dist:
            return self.dist.pdf(pos) * self.peak_value
        else:
            return None

    def draw_dist(self) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import multivariate_normal

        I = 2
        s = 0.0002
        r = 0.05
        x, y = np.mgrid[-r:r:0.001, -r:r:0.001]
        rv = multivariate_normal([0, 0], [[s, 0], [0, s]])
        p0 = rv.pdf((0, 0))
        data = np.dstack((x, y))
        z = rv.pdf(data) * I / p0
        plt.contourf(x, y, z, cmap="coolwarm")
        plt.show()

    @classmethod
    def oG(cls, c=1, id="Odor"):
        return cls(id=id, intensity=2.0 * c, spread=0.01 * np.sqrt(c))

    @classmethod
    def oD(cls, c=1, id="Odor"):
        return cls(id=id, intensity=300.0 * c, spread=0.1 * np.sqrt(c))

    @classmethod
    def oO(cls, o, **kwargs):
        if o == "G":
            return cls.oG(**kwargs)
        elif o == "D":
            return cls.oD(**kwargs)
        else:
            return cls(**kwargs)


class Epoch(NestedConf):
    """
    Life stage epoch parameter group with substrate and timing.

    Defines developmental epoch with age range and associated substrate
    nutrition, used for life history modeling.

    Attributes:
        age_range: Epoch duration in hours post-hatch (start, end)
        substrate: Substrate nutrition for this epoch
        start: Epoch start time (property)
        end: Epoch end time (property, can be None for final epoch)

    Example:
        >>> epoch = Epoch(age_range=(0.0, 96.0), substrate={'type': 'standard'})
        >>> epoch.ticks(dt=0.1)  # Simulation ticks for epoch duration
    """

    age_range = OptionalPositiveRange(
        default=(0.0, None),
        softmax=100.0,
        hardmax=250.0,
        doc="The beginning and end of the epoch in hours post-hatch.",
    )
    substrate = ClassAttr(
        Substrate, default=Substrate(type="standard"), doc="The substrate of the epoch"
    )

    def __init__(self, **kwargs):
        kwargs = expand_kws_shortcuts(kwargs)
        super().__init__(**kwargs)

    @property
    def start(self) -> float:
        return self.age_range[0]

    @property
    def end(self) -> float | None:
        return self.age_range[1]

    def ticks(self, dt) -> int | float:
        if self.end is not None:
            return int((self.end - self.start) / 24 / dt)
        else:
            return np.inf


class Life(NestedConf):
    """
    Life history parameter group for larva development.

    Defines complete life history with age, feeding epochs, and pupation.
    Supports construction from epoch ticks or pre-starvation protocols.

    Attributes:
        age: Starting age in hours post-hatch (default: 0.0, None = pupation)
        epochs: List of Epoch instances defining feeding schedule
        reach_pupation: Whether to grow to pupation (default: False)

    Example:
        >>> life = Life(age=96.0, epochs=[epoch1, epoch2])
        >>> life_prestarved = Life.prestarved(age=72.0, h_starved=24.0)
    """

    age = OptionalPositiveNumber(
        default=0.0,
        softmax=100.0,
        hardmax=250.0,
        doc="The larva age in hours post-hatch at the start of the behavioral simulation. "
        "The larva will grow to that age based on the DEB model. If age is None the "
        "larva will grow to pupation.",
    )
    epochs = ItemListParam(
        item_type=Epoch, doc="The feeding epochs comprising life history."
    )
    reach_pupation = param.Boolean(
        False, doc="If True the larva will grow to pupation."
    )

    @classmethod
    def from_epoch_ticks(cls, ticks=[], subs=None, reach_pupation=False):
        assert all([tick > 0 for tick in ticks])
        ticks.sort()
        age_range = []
        age = 0
        for tick in ticks:
            age_range.append((age, tick))
            age = tick
        if reach_pupation:
            age_range.append((age, None))
            age = None
        N = len(age_range)
        if subs is None:
            sub = [[1.0, "standard"]] * N
        elif len(subs) == 2 and isinstance(subs[0], float) and isinstance(subs[1], str):
            sub = [subs] * N
        else:
            assert len(subs) == N
            sub = subs
        epochs = util.ItemList(cls=Epoch, objs=N, sub=sub, age_range=age_range)
        return cls(age=age, epochs=epochs, reach_pupation=reach_pupation)

    @classmethod
    def prestarved(
        cls,
        age=0.0,
        h_starved=0.0,
        rearing_quality=1.0,
        starvation_quality=0.0,
        final_quality=None,
        substrate_type="standard",
        reach_pupation=False,
    ):
        if final_quality is None:
            final_quality = rearing_quality
        sub_r = [rearing_quality, substrate_type]
        sub_s = [starvation_quality, substrate_type]
        sub_p = [final_quality, substrate_type]
        if age == 0.0:
            ticks, subs = [], []
        else:
            if h_starved == 0:
                ticks, subs = [age], [sub_r]
            elif h_starved >= age:
                ticks, subs = [age], [sub_s]
            else:
                ticks, subs = [age - h_starved, age], [sub_r, sub_s]
        if reach_pupation:
            subs.append(sub_p)
        return cls.from_epoch_ticks(
            ticks=ticks, subs=subs, reach_pupation=reach_pupation
        )


class AirPuff(NestedConf):
    """
    Air puff stimulus parameter group for mechanosensory experiments.

    Defines air puff timing and strength for delivering mechanical
    stimulation during simulations.

    Attributes:
        duration: Puff duration in seconds (default: 1.0)
        start_time: Puff onset time in seconds (default: None)
        strength: Puff strength coefficient (default: 1.0)

    Example:
        >>> puff = AirPuff(duration=2.0, start_time=30.0, strength=0.8)
    """

    duration = PositiveNumber(
        default=1.0,
        softmax=100.0,
        step=0.1,
        doc="The duration of the air-puff in seconds.",
    )
    speed = PositiveNumber(
        default=10.0, softmax=1000.0, step=0.1, doc="The wind speed of the air-puff."
    )
    direction = PositiveNumber(
        default=0.0,
        softmax=100.0,
        step=0.1,
        doc="The directions of the air puff in radians.",
    )
    start_time = PositiveNumber(
        default=0.0,
        softmax=10000.0,
        step=1.0,
        doc="The starting time of the air-puff in seconds.",
    )
    N = OptionalPositiveInteger(
        default=None,
        softmax=10000,
        doc="The number of repetitions of the puff. If N>1 an interval must be provided.",
    )
    interval = PositiveNumber(
        default=5.0,
        softmax=10000.0,
        step=0.1,
        doc="Whether the puff will reoccur at constant time intervals in seconds. Ignored if N=1.",
    )
