"""Unit-aware parameter classes built on the project Pint registry.

This module defines `UnitParam` and specialized unit validators for time,
distance, angle, and motion units. It is used whenever a parameter must carry
physical dimensional metadata (for example seconds, meters, radians, or
velocity/acceleration units). The helper methods validate dimensionality and
normalize labels so downstream configuration and model code can reason about
units consistently.
"""

from __future__ import annotations

from typing import Any, Optional

import param

from larvaworld import units

__all__ = [
    "UnitParam",
    "TimeUnitParam",
    "DistanceUnitParam",
    "AngleUnitParam",
    "TranslationalVelocityUnitParam",
    "TranslationalAccelerationUnitParam",
    "AngularVelocityUnitParam",
    "AngularAccelerationUnitParam",
    "TypeParam",
]


class UnitParam(param.Parameter):
    """Parameter holding a Pint unit from the active registry."""

    @staticmethod
    def _normalize_unit(value: Any) -> Any:
        if value is None:
            return units.dimensionless
        if isinstance(value, str):
            if value in ("", "-", "dimensionless"):
                return units.dimensionless
            return units.Unit(value)
        try:
            if isinstance(value, (type(units.dimensionless), units.Unit)):
                return units.Unit(str(value))
        except Exception:
            pass
        return units.Unit(str(value))

    def __init__(self, default=None, doc: Optional[str] = None, **kwargs):
        default = self._normalize_unit(default)
        if doc is None:
            doc = "Pint unit parameter"
        super().__init__(default=default, doc=doc, **kwargs)

    @staticmethod
    def _dimensionality(u: Any):
        value = getattr(u, "default", u)
        return UnitParam._normalize_unit(value).dimensionality

    @staticmethod
    def is_dimensionless(u: Any) -> bool:
        value = getattr(u, "default", u)
        if isinstance(value, str):
            return value in {"", "-", "dimensionless"}
        return bool(getattr(value, "dimensionless", False))

    @staticmethod
    def is_time(u: Any) -> bool:
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == units.s.dimensionality

    @staticmethod
    def is_distance(u: Any) -> bool:
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == units.m.dimensionality

    @staticmethod
    def is_angle(u: Any) -> bool:
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == units.rad.dimensionality

    @staticmethod
    def is_velocity(u: Any) -> bool:
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == (units.m / units.s).dimensionality or (
            value.dimensionality.get("[length]", 0) == 1
            and value.dimensionality.get("[time]", 0) == -1
        )

    @staticmethod
    def is_translational_velocity(u: Any) -> bool:
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == (units.m / units.s).dimensionality

    @staticmethod
    def is_translational_acceleration(u: Any) -> bool:
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == (units.m / units.s**2).dimensionality

    @staticmethod
    def is_angular_velocity(u: Any) -> bool:
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == (units.rad / units.s).dimensionality

    @staticmethod
    def is_angular_acceleration(u: Any) -> bool:
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == (units.rad / units.s**2).dimensionality

    @staticmethod
    def label(unit: Any) -> str:
        value = UnitParam._normalize_unit(getattr(unit, "default", unit))
        return str(value)


class TimeUnitParam(UnitParam):
    def __init__(self, default=units.s, doc: Optional[str] = None, **kwargs):
        if doc is None:
            doc = "Time unit parameter"
        if not UnitParam.is_time(default):
            raise ValueError(f"Expected a time unit, got {default!r}")
        super().__init__(default=default, doc=doc, **kwargs)


class DistanceUnitParam(UnitParam):
    def __init__(self, default=units.m, doc: Optional[str] = None, **kwargs):
        if doc is None:
            doc = "Distance unit parameter"
        if not UnitParam.is_distance(default):
            raise ValueError(f"Expected a distance unit, got {default!r}")
        super().__init__(default=default, doc=doc, **kwargs)


class AngleUnitParam(UnitParam):
    def __init__(self, default=units.rad, doc: Optional[str] = None, **kwargs):
        if doc is None:
            doc = "Angle unit parameter"
        if not UnitParam.is_angle(default):
            raise ValueError(f"Expected an angle unit, got {default!r}")
        super().__init__(default=default, doc=doc, **kwargs)


class TranslationalVelocityUnitParam(UnitParam):
    def __init__(self, default=units.m / units.s, doc: Optional[str] = None, **kwargs):
        if doc is None:
            doc = "Translational velocity unit parameter"
        if not UnitParam.is_translational_velocity(default):
            raise ValueError(f"Expected a translational velocity unit, got {default!r}")
        super().__init__(default=default, doc=doc, **kwargs)


class TranslationalAccelerationUnitParam(UnitParam):
    def __init__(
        self, default=units.m / units.s**2, doc: Optional[str] = None, **kwargs
    ):
        if doc is None:
            doc = "Translational acceleration unit parameter"
        if not UnitParam.is_translational_acceleration(default):
            raise ValueError(
                f"Expected a translational acceleration unit, got {default!r}"
            )
        super().__init__(default=default, doc=doc, **kwargs)


class AngularVelocityUnitParam(UnitParam):
    def __init__(
        self, default=units.rad / units.s, doc: Optional[str] = None, **kwargs
    ):
        if doc is None:
            doc = "Angular velocity unit parameter"
        if not UnitParam.is_angular_velocity(default):
            raise ValueError(f"Expected an angular velocity unit, got {default!r}")
        super().__init__(default=default, doc=doc, **kwargs)


class AngularAccelerationUnitParam(UnitParam):
    def __init__(
        self, default=units.rad / units.s**2, doc: Optional[str] = None, **kwargs
    ):
        if doc is None:
            doc = "Angular acceleration unit parameter"
        if not UnitParam.is_angular_acceleration(default):
            raise ValueError(f"Expected an angular acceleration unit, got {default!r}")
        super().__init__(default=default, doc=doc, **kwargs)


class TypeParam(param.Parameter):
    """Parameter storing a Python type object."""

    def __init__(self, default=type(None), **kwargs):
        if default is None:
            default = type(None)
        if not isinstance(default, type):
            raise TypeError("dtype must be a type object")
        super().__init__(default=default, **kwargs)

    def _validate(self, val):
        if val is None:
            return type(None)
        if not isinstance(val, type):
            raise TypeError("dtype must be a type object")
        return val
