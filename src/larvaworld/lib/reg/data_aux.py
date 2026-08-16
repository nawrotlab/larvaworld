"""
Larvaworld parameter class and associated methods
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional

import random
import typing

import numpy as np
import param

from .. import reg, util
from ..param.custom import List, StringRobust, Unit, resolve_param_class
from ..param.nested_parameter_group import Conf
from ..util import nam

if TYPE_CHECKING:
    # Only for type hints: importing at runtime would be circular, since
    # process.dataset depends on reg (e.g. reg.par.df_to_pint).
    from ..process.dataset import LarvaDataset

__all__: list[str] = [
    "SAMPLING_PARS",
    "sample_ps",
    "get_LarvaworldParam",
    "prepare_LarvaworldParam",
    "build_LarvaworldParam",
]


def _is_parclass(parclass: type, base: type) -> bool:
    """Whether `parclass` is `base` or one of its subclasses -- shared by
    `LarvaworldParam.randomize`/`.mutate`'s per-type-family dispatch."""
    return parclass in [base] + base.__subclasses__()


class LarvaworldParamName(Conf):
    """
    Base class holding a `LarvaworldParam`'s naming/identity attributes
    (`p`, `d`, `disp`, `k`, `sym`, `codename`, `flatname`) and the
    properties derived purely from them. Overrides `Conf.resolve_kwargs`
    with naming-specific cross-field inference (see `resolve_kwargs`).
    Kept separate from `LarvaworldParam` so the naming/identity concern
    doesn't mix with the value (`v`)/unit (`u`) machinery `LarvaworldParam`
    adds on top.
    """

    p = StringRobust(
        default="", doc="Name of the parameter", label="Name", precedence=6
    )
    d = StringRobust(
        default=None,
        doc="Dataset name of the parameter",
        label="Name in dataset",
        precedence=5,
    )
    disp = StringRobust(
        default=None,
        doc="Displayed name of the parameter",
        label="Display name",
        precedence=10,
    )
    k = StringRobust(
        default=None, doc="Key of the parameter", label="Key", precedence=9
    )
    sym = StringRobust(
        default=None, doc="Symbol of the parameter", label="Symbol", precedence=7
    )
    codename = StringRobust(
        default=None,
        doc="Name of the parameter in code",
        label="Name in code",
        precedence=2,
    )
    flatname = StringRobust(
        default=None,
        doc="Name of the parameter in model configuration",
        label="Name in config file",
        precedence=1,
    )

    def _disp_property(self):
        return self.disp

    #: `s`/`parameter` are aliases -- both just the display name.
    s = parameter = property(_disp_property)

    @property
    def short(self):
        return self.k

    @property
    def symbol(self):
        return self.sym

    @classmethod
    def resolve_kwargs(cls, kwargs: dict[str, Any]) -> util.AttrDict:
        """
        Resolve a `LarvaworldParam`'s naming/identity attributes (`p`,
        `d`, `disp`, `k`, `sym`, `codename`, `flatname`). Any not passed
        in `kwargs` fall back to this class's own declared field default
        (`""` for `p`, `None` for the rest, via `Conf.complete_kwargs`);
        each value still `None` after that is then inferred from the
        others: `codename`/`d` default to `p`; `disp` defaults to `d`;
        `k` defaults to `d`; `sym` defaults to `k`; `flatname` defaults to
        `p`'s entry in `SAMPLING_PARS` if present, else `p` itself.
        Overrides `Conf.resolve_kwargs` (plain defaults-merging) with
        this naming-specific cross-field inference.

        Args:
            kwargs: Any of `p`/`d`/`disp`/`k`/`sym`/`codename`/
                `flatname`, each optional.

        Returns:
            util.AttrDict: {"p", "d", "disp", "k", "sym", "codename",
            "flatname"}, fully resolved.
        """
        resolved = cls.complete_kwargs(kwargs)
        p = resolved["p"]
        if resolved["codename"] is None:
            resolved["codename"] = p
        if resolved["d"] is None:
            resolved["d"] = p
        if resolved["disp"] is None:
            resolved["disp"] = resolved["d"]
        if resolved["k"] is None:
            resolved["k"] = resolved["d"]
        if resolved["sym"] is None:
            resolved["sym"] = resolved["k"]
        if resolved["flatname"] is None:
            resolved["flatname"] = SAMPLING_PARS[p] if p in SAMPLING_PARS else p

        return resolved


class LarvaworldParam(LarvaworldParamName):
    """
    LarvaworldParam is a class that extends LarvaworldParamName (naming/
    identity) with a value (`v`), unit (`u`), and the rest of the
    structured attributes/methods for managing parameters in the
    Larvaworld package.

    """

    dtype = param.Parameter(
        default=float,
        doc="Data type of the parameter value",
        label="Data type",
        precedence=4,
    )
    func = param.Callable(
        default=None,
        doc="Function to get the parameter from a dataset",
        label="Computing function",
        allow_None=True,
        precedence=1,
    )
    required_ks = List(
        default=[],
        doc="Keys of prerequired parameters for computation in a dataset",
        label="Required param keys",
        precedence=1,
    )

    @property
    def l(self):
        return self.disp + "  " + self.ulabel

    @property
    def symunit(self):
        return self.sym + "  " + self.ulabel

    @property
    def ulabel(self):
        return Unit.label(self.u)

    @property
    def unit(self):
        return Unit.symbol(self.u)

    def _v_default_property(self):
        return self.param.v.default

    #: `v0`/`initial_value` are aliases -- both just "v"'s declared default.
    v0 = initial_value = property(_v_default_property)

    @property
    def value(self):
        return self.v

    @property
    def label(self):
        return self.param.v.label

    def _v_doc_property(self):
        return self.param.v.doc

    #: `tooltip`/`description`/`help` are aliases -- all just "v"'s doc string.
    tooltip = description = help = property(_v_doc_property)

    @property
    def parclass(self):
        return type(self.param.v)

    @property
    def lim(self):
        return getattr(self.param.v, "bounds", None)

    @property
    def min(self):
        lim = self.lim
        return lim[0] if lim else None

    @property
    def max(self):
        lim = self.lim
        return lim[1] if lim else None

    @property
    def step(self):
        p = self.parclass
        if _is_parclass(p, param.Number) or _is_parclass(p, param.Range):
            if self.param.v.step is not None:
                return self.param.v.step
        if p == param.Magnitude:
            return 0.01
        if p in [param.NumericTuple]:
            return 0.01
        return None

    @property
    def Ndec(self):
        if self.step is not None:
            return str(self.step)[::-1].find(".")
        else:
            return None

    def exists(self, dataset: "LarvaDataset"):
        """
        Check if the parameter exists in the given LarvaDataset.

        Args:
            dataset (LarvaDataset): The dataset to check for the parameter.

        Returns:
            AttrDict: A dictionary-like object with two keys:
                - 'step': A boolean indicating if the parameter exists in the dataset's step_data.
                - 'end': A boolean indicating if the parameter exists in the dataset's endpoint_data.

        """
        return util.AttrDict(
            {"step": self.d in dataset.step_ps, "end": self.d in dataset.end_ps}
        )

    def get(self, dataset: "LarvaDataset", compute: bool = True):
        """
        Retrieve the parameter's value from the dataset if it exists, otherwise compute it.

        Args:
            dataset (LarvaDataset): The dataset object from which to retrieve the parameter.
            compute (bool): Flag indicating whether to compute the parameter if it does not exist. Default is True.

        Returns:
            The parameter value if it exists or is successfully computed, otherwise None.

        Raises:
            None

        Notes:
            - The method first checks if the parameter exists in the dataset.
            - If the parameter exists, it retrieves and returns it.
            - If the parameter does not exist and `compute` is True, it computes the parameter and retries retrieval.
            - If the parameter does not exist and `compute` is False, it prints a message indicating the parameter was not found.

        """
        res = self.exists(dataset)
        for key, exists in res.items():
            if exists:
                return dataset.get_par(key=key, par=self.d)

        if compute:
            self.compute(dataset)
            return self.get(dataset, compute=False)
        else:
            print(f"Parameter {self.disp} not found")

    def compute(self, dataset: "LarvaDataset"):
        """
        Compute the parameter using the provided dataset.

        This method applies the parameter's computing function to the dataset if the function is defined.
        If the function is not defined, it prints a message indicating that the
        function is not defined.

        Args:
            dataset (LarvaDataset) : The dataset to be processed by the function.

        """
        if self.func is not None:
            self.func(dataset)
        else:
            print(f"Function to compute parameter {self.disp} is not defined")

    def randomize(self):
        """
        Randomizes the value of the parameter based on its type.

        This method assigns a random value to `self.v` depending on the type of
        the parameter class (`self.parclass`). The behavior varies as follows:

        - If the parameter is a `Number` or its subclass, a random float within
          the parameter's bounds is assigned.
        - If the parameter is an `Integer` or its subclass, a random integer
          within the parameter's bounds is assigned.
        - If the parameter is a `Magnitude` or its subclass, a random float
          between 0.0 and 1.0 is assigned.
        - If the parameter is a `Selector` or its subclass, a random choice
          from the parameter's objects is assigned.
        - If the parameter is a `Boolean`, a random boolean value (True or False)
          is assigned.
        - If the parameter is a `Range` or its subclass, a tuple of two random
          floats within the parameter's bounds is assigned, where the second
          float is greater than or equal to the first.

        """
        p = self.parclass
        if _is_parclass(p, param.Number):
            vmin, vmax = self.param.v.bounds
            self.v = self.param.v.crop_to_bounds(
                np.round(random.uniform(vmin, vmax), self.Ndec)
            )
        elif _is_parclass(p, param.Integer):
            vmin, vmax = self.param.v.bounds
            self.v = random.randint(vmin, vmax)
        elif _is_parclass(p, param.Magnitude):
            self.v = np.round(random.uniform(0.0, 1.0), self.Ndec)
        elif _is_parclass(p, param.Selector):
            self.v = random.choice(self.param.v.objects)
        elif p == param.Boolean:
            self.v = random.choice([True, False])
        elif _is_parclass(p, param.Range):
            vmin, vmax = self.param.v.bounds
            vv0 = np.round(random.uniform(vmin, vmax), self.Ndec)
            vv1 = np.round(random.uniform(vv0, vmax), self.Ndec)
            self.v = (vv0, vv1)

    def mutate(self, Pmut, Cmut):
        """
        Mutates the value of the parameter based on its class type.

        Args:
            Pmut (float): Probability of mutation.
            Cmut (float): Coefficient of mutation.

        Notes:
            - For `param.Magnitude` and its subclasses, the value is mutated using a Gaussian distribution and cropped to bounds.
            - For `param.Integer` and its subclasses, the value is mutated using a Gaussian distribution, converted to an integer, and cropped to bounds.
            - For `param.Number` and its subclasses, the value is mutated using a Gaussian distribution and cropped to bounds.
            - For `param.Selector` and its subclasses, the value is randomly chosen from the available objects.
            - For `param.Boolean`, the value is randomly chosen between True and False.
            - For `param.Range` and its subclasses, the range values are mutated using a Gaussian distribution, clipped to bounds, and rounded.
        """
        if random.random() < Pmut:
            p = self.parclass
            if _is_parclass(p, param.Magnitude):
                v0 = self.v if self.v is not None else 0.5
                vv = random.gauss(v0, Cmut)
                self.v = self.param.v.crop_to_bounds(np.round(vv, self.Ndec))
            elif _is_parclass(p, param.Integer):
                vmin, vmax = self.param.v.bounds
                vr = np.abs(vmax - vmin)
                v0 = self.v if self.v is not None else int(vmin + vr / 2)
                vv = random.gauss(v0, Cmut * vr)
                self.v = self.param.v.crop_to_bounds(int(vv))
            elif _is_parclass(p, param.Number):
                vmin, vmax = self.param.v.bounds
                vr = np.abs(vmax - vmin)
                v0 = self.v if self.v is not None else vmin + vr / 2
                vv = random.gauss(v0, Cmut * vr)
                self.v = self.param.v.crop_to_bounds(np.round(vv, self.Ndec))
            elif _is_parclass(p, param.Selector):
                self.v = random.choice(self.param.v.objects)
            elif p == param.Boolean:
                self.v = random.choice([True, False])
            elif _is_parclass(p, param.Range):
                vmin, vmax = self.param.v.bounds
                vr = np.abs(vmax - vmin)
                v0, v1 = self.v if self.v is not None else (vmin, vmax)
                vv0 = random.gauss(v0, Cmut * vr)
                vv1 = random.gauss(v1, Cmut * vr)
                vv0 = np.round(np.clip(vv0, a_min=vmin, a_max=vmax), self.Ndec)
                vv1 = np.round(np.clip(vv1, a_min=vv0, a_max=vmax), self.Ndec)
                self.v = (vv0, vv1)

    def to_config(self) -> "util.AttrDict":
        """
        Extract this parameter's complete configuration as an AttrDict:
        every declared param's current value (`self.param.values()`, minus
        `name`) plus the reconstruction metadata needed to rebuild an
        equivalent `v` parameter (doc, bounds, step, choices) that isn't
        captured by the values alone.

        `func` (the callable used to compute this parameter from a
        dataset) is replaced by `func_ref`, a "module.qualname" string
        pointing at its definition: `func` is frequently a local closure
        (e.g. built by `add_operators`), which neither pickle nor JSON can
        serialize, and closures can't be re-imported by name in general —
        so `func_ref` is informational only (documents where the compute
        logic lives), and `from_config` does not attempt to rebind it; a
        reconstructed parameter always gets `func=None`.

        The result is round-trippable via `LarvaworldParam.from_config`.
        """
        values = self.param.values()
        v_desc = self.param.objects()["v"]
        func = values.get("func")
        config = {k: v for k, v in values.items() if k not in ("name", "func")}
        config["func_ref"] = (
            f"{func.__module__}.{func.__qualname__}" if func is not None else None
        )
        config["doc"] = v_desc.doc
        config["lim"] = getattr(v_desc, "bounds", None)
        config["dv"] = getattr(v_desc, "step", None)
        config["vs"] = getattr(v_desc, "objects", None)
        return util.AttrDict(config)

    def save_config(self, file: str) -> None:
        """Save this parameter's full configuration to file, via
        AttrDict.save (pickle, falling back to JSON)."""
        self.to_config().save(file)

    @classmethod
    def from_config(cls, config: "util.AttrDict") -> "LarvaworldParam":
        """
        Create a new LarvaworldParam instance from a configuration dict, as
        produced by `to_config()` (typically loaded back via
        `AttrDict.load(file)`).
        """
        kwargs = dict(config)
        kwargs.pop("func_ref", None)  # informational only; not reconstructible
        u = kwargs.get("u")
        if u is not None:
            # A pint Unit round-tripped through pickle/JSON may deserialize
            # against a different UnitRegistry instance than the live
            # reg.units, which then fails equality/arithmetic checks
            # ("different registries"). Re-resolve it against reg.units.
            kwargs["u"] = reg.units.Unit(str(u))
        return build_LarvaworldParam(**kwargs)


def get_LarvaworldParam(
    v_param: Any, v0: Any = None, dv: Any = None, **kws: Any
) -> LarvaworldParam:
    """
    Create a LarvaworldParam instance with a custom "v" (value) parameter.

    Dynamically creates a LarvaworldParam subclass with the given
    instantiated value parameter, then instantiates it.

    Args:
        v_param: The instantiated "v" parameter (e.g. a param.Number()
            instance) -- see `build_value_param`.
        v0: Default value for parameter
        dv: Delta/range value for parameter
        **kws: Additional keyword arguments for LarvaworldParam

    Returns:
        Configured LarvaworldParam instance

    Example:
        >>> par = get_LarvaworldParam(param.Number(default=0.5), doc="Speed parameter")
    """

    class _LarvaworldParam(LarvaworldParam):
        v = v_param
        u = Unit(
            default=reg.units.dimensionless,
            doc="Unit of the parameter values",
            label="Unit",
            precedence=8,
        )

    par = _LarvaworldParam(**kws)
    return par


#: Bidirectional mapping between display names and configuration paths.
#:
#: Maps parameter display names to their nested configuration paths in model definition.
#:
#: Example:
#:     >>> SAMPLING_PARS['length']
#:     'body.length'
#:     >>> SAMPLING_PARS.inverse['body.length']
#:     ['length']
SAMPLING_PARS: util.bidict = util.bidict(
    util.AttrDict(
        {
            "length": "body.length",
            nam.freq(nam.scal(nam.vel(""))): "brain.crawler.freq",
            # nam.freq(nam.scal(nam.vel(''))): 'brain.intermitter.crawl_freq',
            nam.mean(
                nam.chunk_track("stride", nam.scal(nam.dst("")))
            ): "brain.crawler.stride_dst_mean",
            nam.std(
                nam.chunk_track("stride", nam.scal(nam.dst("")))
            ): "brain.crawler.stride_dst_std",
            nam.freq("feed"): "brain.feeder.freq",
            nam.max(
                nam.chunk_track("stride", nam.scal(nam.vel("")))
            ): "brain.crawler.max_scaled_vel",
            nam.phi(nam.max(nam.scal(nam.vel("")))): "brain.crawler.max_vel_phase",
            "attenuation": "brain.interference.attenuation",
            nam.max("attenuation"): "brain.interference.attenuation_max",
            nam.freq(nam.vel(nam.orient("front"))): "brain.turner.freq",
            nam.phi(nam.max("attenuation")): "brain.interference.max_attenuation_phase",
        }
    )
)


def sample_ps(ps: list[str], e: Optional[Any] = None) -> list[str]:
    """
    Filter parameters to those that exist in SAMPLING_PARS mapping.

    Gets parameters from list that exist in the inverse SAMPLING_PARS dictionary
    (i.e., parameters that have configuration paths defined). Optionally filters
    to parameters that also exist in endpoint DataFrame.

    Args:
        ps: List of parameter names to filter
        e: Endpoint DataFrame to further filter parameters

    Returns:
        list: List of parameters filtered to exist in the default SAMPLING_PARS dictionary and potentially filtered to exist in e

    Example:
        >>> sample_ps(['length', 'velocity', 'unknown_param'])
        ['length', 'velocity']  # Only those in SAMPLING_PARS
    """
    Sinv = SAMPLING_PARS.inverse
    ps = util.SuperList([Sinv[k] for k in util.existing_cols(Sinv, ps)]).flatten
    if e is not None:
        ps = ps.existing(e)
    return ps


def build_value_param(
    dtype: Any,
    v0: Any,
    doc: str,
    lab: str,
    lim: Optional[tuple[Any, Any]],
    dv: Any,
    vs: Optional[list[Any]],
    param_class: Optional[type[param.Parameter]] = None,
) -> param.Parameter:
    """
    Instantiate the "v" (value) parameter for a `LarvaworldParam`: resolves
    the param.Parameter (sub)class and its constructor kwargs via
    `resolve_param_class` (auto-selecting from `dtype`/`lim`/`vs` unless
    `param_class` is given), then instantiates it.

    Args:
        dtype: Data type, used to auto-select the class if `param_class`
            isn't given.
        v0: The default value for the parameter.
        doc: Documentation string for the parameter.
        lab: Label for the parameter.
        lim: Bounds for the parameter if applicable.
        dv: Step value for the parameter if applicable.
        vs: List of choices for the parameter if applicable (e.g., for
            param.Selector).
        param_class: Explicit param.Parameter subclass to use, or None to
            auto-select it from `dtype`/`lim`/`vs`.

    Returns:
        param.Parameter: An instantiated parameter object with the
        specified attributes.
    """
    resolved_class, param_kwargs = resolve_param_class(
        dtype=dtype,
        param_class=param_class,
        lim=lim,
        dv=dv,
        vs=vs,
        v0=v0,
        doc=doc,
        lab=lab,
    )
    return resolved_class(**param_kwargs, instantiate=True)


def resolve_value_param(
    *,
    dtype: Any,
    disp: str,
    u: Any,
    v0: Any,
    lim: Optional[tuple[Any, Any]],
    dv: Any,
    vs: Optional[list[Any]],
    lab: Optional[str],
    doc: Optional[str],
    param_class: Optional[type[param.Parameter]],
) -> param.Parameter:
    """
    Resolve the instantiated "v" (value) parameter for a `LarvaworldParam`:
    fills in a default label (`lab`) from `disp`/`u` if missing, defaults
    `doc` to that label if missing, then resolves the param.Parameter
    class and builds the instantiated parameter via `build_value_param`
    (which delegates class/kwarg resolution to `resolve_param_class`).

    Args:
        dtype: Data type, used to auto-select the class if `param_class`
            isn't given.
        disp: Display name, used to build a default `lab` if not given.
        u: Physical unit from reg.units, used to build a default `lab`.
        v0: Default value for the parameter.
        lim: Bounds for the parameter if applicable.
        dv: Step value for the parameter if applicable.
        vs: Value set for Selector parameters if applicable.
        lab: Label for the parameter, or None to derive it.
        doc: Documentation string, or None to default it to `lab`.
        param_class: The param.Parameter subclass to instantiate, or None
            to auto-select it from `dtype`/`lim`/`vs`.

    Returns:
        param.Parameter: The instantiated "v" parameter.
    """
    if lab is None:
        ulabel = Unit.label(u)
        lab = f"{disp} {ulabel}" if ulabel else disp
    doc = lab if doc is None else doc
    return build_value_param(
        dtype=dtype,
        v0=v0,
        doc=doc,
        lab=lab,
        lim=lim,
        dv=dv,
        vs=vs,
        param_class=param_class,
    )


def prepare_LarvaworldParam(
    p: str,
    dtype: Any = float,
    lab: Optional[str] = None,
    doc: Optional[str] = None,
    required_ks: list[str] = [],
    u: Any = reg.units.dimensionless,
    v0: Any = None,
    v: Any = None,
    lim: Optional[tuple[Any, Any]] = None,
    dv: Any = None,
    vs: Optional[list[Any]] = None,
    param_class: Optional[type[param.Parameter]] = None,
    func: Any = None,
    **kwargs: Any,
) -> util.AttrDict:
    """
    Format parameter attributes dictionary for LarvaworldParam creation.

    Prepares a comprehensive dictionary of parameter attributes including
    display properties, units, bounds, functions, and documentation.
    Naming/identity (`d`/`disp`/`k`/`sym`/`codename`/`flatname`), accepted
    via `**kwargs`, are resolved by `LarvaworldParamName.resolve_kwargs`;
    `v0`/`dv` defaults are resolved inline; the instantiated "v" parameter
    is resolved by `resolve_value_param`.

    Args:
        p: Primary parameter name
        dtype: Data type (int, float, list, tuple, etc.). Defaults to float
        lab: Label for plots. Auto-generated if not provided
        doc: Documentation string for parameter
        required_ks: List of required parameter keys that must be present
        u: Physical unit from reg.units. Defaults to dimensionless
        v0: Initial/default value. Uses v if not provided
        v: Current parameter value. Used as v0 if v0 not provided
        lim: Parameter bounds as (min, max) tuple
        dv: Step/delta value for parameter. Auto-inferred from dtype if not provided
        vs: Value set for Selector parameters (list of valid options)
        param_class: The param.Parameter subclass to instantiate for "v", or None to resolve it
        func: Computing function for derived parameters
        **kwargs: Naming/identity overrides -- any of `d`/`disp`/`k`/`sym`/
            `codename`/`flatname` (see `LarvaworldParamName.resolve_kwargs`)
            -- plus any other additional parameter attributes.

    Returns:
        util.AttrDict: Dictionary of formatted attributes for creating a LarvaworldParam instance

    Example:
        >>> attrs = prepare_LarvaworldParam('speed', k='v', dtype=float, v0=1.0, lim=(0, 10))
        >>> attrs['k']
        'v'
    """
    naming = LarvaworldParamName.resolve_kwargs({**kwargs, "p": p})
    return _complete_LarvaworldParam(
        naming,
        dtype=dtype,
        lab=lab,
        doc=doc,
        required_ks=required_ks,
        u=u,
        v0=v0,
        v=v,
        lim=lim,
        dv=dv,
        vs=vs,
        param_class=param_class,
        func=func,
    )


def _complete_LarvaworldParam(
    naming: util.AttrDict,
    *,
    dtype: Any,
    lab: Optional[str],
    doc: Optional[str],
    required_ks: list[str],
    u: Any,
    v0: Any,
    v: Any,
    lim: Optional[tuple[Any, Any]],
    dv: Any,
    vs: Optional[list[Any]],
    param_class: Optional[type[param.Parameter]],
    func: Any,
) -> util.AttrDict:
    """
    Finish building a `LarvaworldParam` attrs dict, picking up where
    `prepare_LarvaworldParam` leaves off once naming/identity is resolved:
    resolves `v0`/`dv` defaults, builds the instantiated "v" parameter via
    `resolve_value_param`, and assembles the final config dict.

    Args:
        naming: The resolved {"p", "d", "disp", "k", "sym", "codename",
            "flatname"} dict, as returned by
            `LarvaworldParamName.resolve_kwargs`.
        dtype: Data type (int, float, list, tuple, etc.).
        lab: Label for plots. Auto-generated if not provided.
        doc: Documentation string for parameter.
        required_ks: List of required parameter keys that must be present.
        u: Physical unit from reg.units.
        v0: Initial/default value. Uses `v` if not provided.
        v: Current parameter value. Used as `v0` if `v0` not provided.
        lim: Parameter bounds as (min, max) tuple.
        dv: Step/delta value for parameter. Auto-inferred from `dtype` if
            not provided.
        vs: Value set for Selector parameters (list of valid options).
        param_class: The param.Parameter subclass to instantiate for "v",
            or None to resolve it.
        func: Computing function for derived parameters.

    Returns:
        util.AttrDict: Dictionary of formatted attributes for creating a
        LarvaworldParam instance.
    """
    p = naming["p"]
    codename = naming["codename"]
    d = naming["d"]
    disp = naming["disp"]
    k = naming["k"]
    sym = naming["sym"]
    flatname = naming["flatname"]

    v0 = v if v is not None else v0
    if dv is None:
        if dtype in [
            float,
            typing.List[float],
            typing.List[typing.Tuple[float]],
            typing.Tuple[float],
        ]:
            dv = 0.01
        elif dtype in [int]:
            dv = 1

    value_param = resolve_value_param(
        dtype=dtype,
        disp=disp,
        u=u,
        v0=v0,
        lim=lim,
        dv=dv,
        vs=vs,
        lab=lab,
        doc=doc,
        param_class=param_class,
    )

    # locals() inside the comprehension below would see only the
    # comprehension's own scope (a separate implicit function in Python 3),
    # not this function's -- capture it here first.
    frame_locals = locals()
    config = {
        attr: frame_locals[attr] for attr in LarvaworldParam.param if attr != "name"
    }
    config.update(name=p, u=u, v_param=value_param, dv=dv, v0=v0)
    return util.AttrDict(config)


def build_LarvaworldParam(p: str, **kwargs: Any) -> LarvaworldParam:
    """
    Constructs a Larvaworld parameter object.

    This function prepares the input attributes using the `prepare_LarvaworldParam` function
    and then generates the Larvaworld parameter object using the `get_LarvaworldParam` function.

    Args:
        p: The primary parameter required for building the Larvaworld parameter object.
        **kwargs: Additional keyword arguments that are passed to the `prepare_LarvaworldParam` function.

    Returns:
        The constructed Larvaworld parameter object.

    """
    pre_p = prepare_LarvaworldParam(p=p, **kwargs)
    return get_LarvaworldParam(**pre_p)
