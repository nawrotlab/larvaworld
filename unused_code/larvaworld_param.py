"""Archived helper module kept outside the active source tree.

This file contains older `LarvaworldParam`-related helper classes and generated
value-parameter definitions that were intentionally removed from the live package
path. It is not imported by the active application and is retained only as a
historical reference. If you choose to restore it, move this file back to
`src/larvaworld/lib/param/larvaworld_param.py` and restore the corresponding
package exports and imports in the active code path.
"""

from __future__ import annotations

from typing import Any, Optional

import random

import numpy as np
import param

from larvaworld import units

from . import custom as _custom
from .units import (
    AngleUnitParam,
    AngularAccelerationUnitParam,
    AngularVelocityUnitParam,
    TimeUnitParam,
    TranslationalAccelerationUnitParam,
    TranslationalVelocityUnitParam,
    TypeParam,
    UnitParam,
)

__all__ = [
    "LarvaworldParamName",
    "LarvaworldParamHelper",
    "LarvaworldParam",
    "UnitParamMixin",
    "PositiveContinuousTime",
    "PositiveDiscreteTime",
    "AngleParam",
    "PhaseParam",
    "SignedPhaseParam",
    "TranslationalVelocityParam",
    "TranslationalAccelerationParam",
    "AngularVelocityParam",
    "AngularAccelerationParam",
]


class LarvaworldParamName(StringRobust):
    """Naming metadata base for Larvaworld parameter values."""

    p = param.String(default="", doc="Name of the parameter", label="Name")
    d = param.String(
        default=None, doc="Dataset name of the parameter", label="Name in dataset"
    )
    disp = param.String(
        default=None, doc="Displayed name of the parameter", label="Display name"
    )
    k = param.String(default=None, doc="Key of the parameter", label="Key")
    sym = param.String(default=None, doc="Symbol of the parameter", label="Symbol")
    codename = param.String(
        default=None, doc="Name of the parameter in code", label="Name in code"
    )
    flatname = param.String(
        default=None,
        doc="Name of the parameter in model configuration",
        label="Name in config file",
    )

    def __init__(
        self, default="", doc="Description of the parameter", label="Name", **kwargs
    ):
        super().__init__(default=default, doc=doc, label=label, **kwargs)

    @classmethod
    def resolve_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(kwargs)
        p = resolved.get("p", "")
        if resolved.get("codename") is None:
            resolved["codename"] = p
        if resolved.get("d") is None:
            resolved["d"] = p
        if resolved.get("disp") is None:
            resolved["disp"] = resolved["d"]
        if resolved.get("k") is None:
            resolved["k"] = resolved["d"]
        if resolved.get("sym") is None:
            resolved["sym"] = resolved["k"]
        if resolved.get("flatname") is None:
            resolved["flatname"] = p
        return resolved

    def _disp_property(self):
        return self.disp

    s = parameter = property(_disp_property)

    @property
    def short(self):
        return self.k

    @property
    def symbol(self):
        return self.sym


class LarvaworldParamHelper(LarvaworldParamName):
    """Shared base class for unit-aware parameter values."""

    unit = UnitParam(default=units.dimensionless, constant=True, readonly=True)
    dtype = TypeParam(default=type(None), constant=True, readonly=True)

    @property
    def l(self):
        return self.disp + "  " + UnitParam.label(self.unit)

    @property
    def symunit(self):
        return self.sym + "  " + UnitParam.label(self.unit)


class LarvaworldParam(LarvaworldParamHelper):
    """Full parameter helper including dataset/registry-oriented API."""

    func = param.Callable(
        default=None,
        doc="Function to get the parameter from a dataset",
        label="Computing function",
        allow_None=True,
        precedence=1,
    )
    required_ks = param.List(
        default=[],
        doc="Keys of prerequired parameters for computation in a dataset",
        label="Required param keys",
        precedence=1,
    )

    def _v_default_property(self):
        return self.default

    v0 = initial_value = property(_v_default_property)

    @property
    def value(self):
        return self.default

    def _v_doc_property(self):
        return self.doc

    tooltip = description = help = property(_v_doc_property)

    @property
    def parclass(self):
        return type(self)

    @property
    def lim(self):
        return getattr(self, "bounds", None)

    @property
    def min(self):
        lim = self.lim
        return lim[0] if lim else None

    @property
    def max(self):
        lim = self.lim
        return lim[1] if lim else None

    @property
    def Ndec(self):
        step = getattr(self, "step", None)
        if step is not None:
            return str(step)[::-1].find(".")
        return None

    def exists(self, dataset):
        return _custom.util.AttrDict(
            {"step": self.d in dataset.step_ps, "end": self.d in dataset.end_ps}
        )

    def get(self, dataset, compute: bool = True):
        res = self.exists(dataset)
        for key, exists in res.items():
            if exists:
                return dataset.get_par(key=key, par=self.d)
        if compute:
            self.compute(dataset)
            return self.get(dataset, compute=False)
        print(f"Parameter {self.disp} not found")

    def compute(self, dataset):
        if self.func is not None:
            self.func(dataset)
        else:
            print(f"Function to compute parameter {self.disp} is not defined")

    def randomize(self):
        p = self.parclass
        if _custom._is_parclass(p, param.Number):
            vmin, vmax = getattr(self, "bounds", (0.0, 1.0))
            self.default = self.crop_to_bounds(
                np.round(random.uniform(vmin, vmax), self.Ndec)
            )
        elif _custom._is_parclass(p, param.Integer):
            vmin, vmax = getattr(self, "bounds", (0, 1))
            self.default = random.randint(vmin, vmax)
        elif _custom._is_parclass(p, param.Magnitude):
            self.default = np.round(random.uniform(0.0, 1.0), self.Ndec)
        elif _custom._is_parclass(p, param.Selector):
            self.default = random.choice(getattr(self, "objects", []))
        elif p == param.Boolean:
            self.default = random.choice([True, False])
        elif _custom._is_parclass(p, param.Range):
            vmin, vmax = getattr(self, "bounds", (0.0, 1.0))
            vv0 = np.round(random.uniform(vmin, vmax), self.Ndec)
            vv1 = np.round(random.uniform(vv0, vmax), self.Ndec)
            self.default = (vv0, vv1)

    def mutate(self, Pmut, Cmut):
        if random.random() < Pmut:
            p = self.parclass
            if _custom._is_parclass(p, param.Magnitude):
                v0 = self.default if self.default is not None else 0.5
                vv = random.gauss(v0, Cmut)
                self.default = self.crop_to_bounds(np.round(vv, self.Ndec))
            elif _custom._is_parclass(p, param.Integer):
                vmin, vmax = getattr(self, "bounds", (0, 1))
                vr = abs(vmax - vmin)
                v0 = self.default if self.default is not None else int(vmin + vr / 2)
                vv = random.gauss(v0, Cmut * vr)
                self.default = self.crop_to_bounds(int(vv))
            elif _custom._is_parclass(p, param.Number):
                vmin, vmax = getattr(self, "bounds", (0.0, 1.0))
                vr = abs(vmax - vmin)
                v0 = self.default if self.default is not None else vmin + vr / 2
                vv = random.gauss(v0, Cmut * vr)
                self.default = self.crop_to_bounds(np.round(vv, self.Ndec))
            elif _custom._is_parclass(p, param.Selector):
                self.default = random.choice(getattr(self, "objects", []))
            elif p == param.Boolean:
                self.default = random.choice([True, False])
            elif _custom._is_parclass(p, param.Range):
                vmin, vmax = getattr(self, "bounds", (0.0, 1.0))
                vr = abs(vmax - vmin)
                v0, v1 = self.default if self.default is not None else (vmin, vmax)
                vv0 = random.gauss(v0, Cmut * vr)
                vv1 = random.gauss(v1, Cmut * vr)
                vv0 = np.round(np.clip(vv0, a_min=vmin, a_max=vmax), self.Ndec)
                vv1 = np.round(np.clip(vv1, a_min=vv0, a_max=vmax), self.Ndec)
                self.default = (vv0, vv1)

    def to_config(self):
        values = self.__class__.__dict__.copy()
        config = {
            k: v
            for k, v in values.items()
            if not k.startswith("__") and k not in ("name", "func")
        }
        config["func_ref"] = None
        config["doc"] = self.doc
        config["lim"] = getattr(self, "bounds", None)
        config["dv"] = getattr(self, "step", None)
        config["vs"] = getattr(self, "objects", None)
        return _custom.util.AttrDict(config)

    def save_config(self, file: str) -> None:
        self.to_config().save(file)

    @classmethod
    def from_config(cls, config):
        kwargs = dict(config)
        kwargs.pop("func_ref", None)
        u = kwargs.get("u")
        if u is not None:
            kwargs["u"] = units.Unit(str(u))
        return cls(**kwargs)


class UnitParamMixin:
    @staticmethod
    def build(
        param_cls: type[param.Parameter],
        *,
        unit_cls: type[UnitParam] = UnitParam,
        unit_default: Any = None,
        default_label: str = "",
        default_doc: Optional[str] = None,
        dtype: Optional[type] = None,
        class_name: Optional[str] = None,
    ) -> type[param.Parameter]:
        if not issubclass(param_cls, param.Parameter):
            raise TypeError(f"{param_cls!r} is not a param.Parameter subclass")
        if not issubclass(unit_cls, UnitParam):
            raise TypeError(f"{unit_cls!r} must be a subclass of UnitParam")
        if dtype is None or dtype is type(None):
            dtype = _infer_dtype(param_cls)
        if default_doc is None:
            default_doc = default_label

        def __init__(
            self,
            *args: Any,
            label: Optional[str] = None,
            doc: Optional[str] = None,
            **kwargs: Any,
        ):
            name_keys = {"p", "d", "disp", "k", "sym", "codename", "flatname"}
            name_kwargs = {
                key: kwargs.pop(key) for key in list(kwargs) if key in name_keys
            }
            LarvaworldParamName.resolve_kwargs(name_kwargs)
            param.Parameter.__init__(self)
            param_cls.__init__(self, *args, **kwargs)
            if label is None:
                ulabel = UnitParam.label(self.unit)
                label = f"{default_label} {ulabel}" if ulabel else default_label
            self.label = label
            if doc is None:
                doc = default_doc or label
            self.doc = doc

        if class_name is None:
            class_name = f"{param_cls.__name__}WithUnit"
        attrs = {
            "unit": unit_cls(default=unit_default, constant=True, readonly=True),
            "dtype": TypeParam(default=dtype, constant=True, readonly=True),
            "__init__": __init__,
            "__module__": __name__,
        }
        generated = type(class_name, (LarvaworldParamHelper, param_cls), attrs)
        generated.__module__ = __name__
        return generated


def _infer_dtype(base_cls: type[param.Parameter]) -> type:
    if issubclass(base_cls, param.Integer):
        return int
    if issubclass(base_cls, param.Number):
        return float
    if issubclass(base_cls, param.String):
        return str
    return object


_UNIT_VALUE_CLASS_NAMES = [
    "PositiveContinuousTime",
    "PositiveDiscreteTime",
    "AngleParam",
    "PhaseParam",
    "SignedPhaseParam",
    "TranslationalVelocityParam",
    "TranslationalAccelerationParam",
    "AngularVelocityParam",
    "AngularAccelerationParam",
]


def _build_value_classes() -> None:
    if all(name in globals() for name in _UNIT_VALUE_CLASS_NAMES):
        return
    globals()["PositiveContinuousTime"] = UnitParamMixin.build(
        _custom.PositiveNumber,
        unit_cls=TimeUnitParam,
        unit_default=units.s,
        default_label="Time",
        default_doc="Positive continuous time",
        dtype=float,
        class_name="PositiveContinuousTime",
    )
    globals()["PositiveDiscreteTime"] = UnitParamMixin.build(
        _custom.PositiveInteger,
        unit_cls=UnitParam,
        unit_default=units.dimensionless,
        default_label="Discrete time",
        default_doc="Positive discrete-time count",
        dtype=int,
        class_name="PositiveDiscreteTime",
    )
    globals()["AngleParam"] = UnitParamMixin.build(
        param.Number,
        unit_cls=AngleUnitParam,
        unit_default=units.rad,
        default_label="Angular value",
        default_doc="Angular value",
        dtype=float,
        class_name="AngleParam",
    )
    globals()["PhaseParam"] = UnitParamMixin.build(
        _custom.Phase,
        unit_cls=AngleUnitParam,
        unit_default=units.rad,
        default_label="Phase",
        default_doc="Phase value",
        dtype=float,
        class_name="PhaseParam",
    )
    globals()["SignedPhaseParam"] = UnitParamMixin.build(
        _custom.SignedPhase,
        unit_cls=AngleUnitParam,
        unit_default=units.rad,
        default_label="Signed phase",
        default_doc="Signed phase value",
        dtype=float,
        class_name="SignedPhaseParam",
    )
    globals()["TranslationalVelocityParam"] = UnitParamMixin.build(
        param.Number,
        unit_cls=TranslationalVelocityUnitParam,
        unit_default=units.m / units.s,
        default_label="Translational velocity",
        default_doc="Translational velocity value",
        dtype=float,
        class_name="TranslationalVelocityParam",
    )
    globals()["TranslationalAccelerationParam"] = UnitParamMixin.build(
        param.Number,
        unit_cls=TranslationalAccelerationUnitParam,
        unit_default=units.m / units.s**2,
        default_label="Translational acceleration",
        default_doc="Translational acceleration value",
        dtype=float,
        class_name="TranslationalAccelerationParam",
    )
    globals()["AngularVelocityParam"] = UnitParamMixin.build(
        param.Number,
        unit_cls=AngularVelocityUnitParam,
        unit_default=units.rad / units.s,
        default_label="Angular velocity",
        default_doc="Angular velocity value",
        dtype=float,
        class_name="AngularVelocityParam",
    )
    globals()["AngularAccelerationParam"] = UnitParamMixin.build(
        param.Number,
        unit_cls=AngularAccelerationUnitParam,
        unit_default=units.rad / units.s**2,
        default_label="Angular acceleration",
        default_doc="Angular acceleration value",
        dtype=float,
        class_name="AngularAccelerationParam",
    )


def __getattr__(name: str):
    if name in _UNIT_VALUE_CLASS_NAMES:
        _build_value_classes()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Build eagerly so the named value classes are available on import.
_build_value_classes()
