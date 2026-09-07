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
        """Coerce a value into a pint unit.

        Args:
            value: A unit, a unit string, or None.

        Returns:
            The pint unit; dimensionless when None was given.
        """
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
        """Build the parameter.

        Args:
            default: See the class attributes.
            doc: See the class attributes.
            **kwargs: Forwarded to the parent class.
        """
        default = self._normalize_unit(default)
        if doc is None:
            doc = "Pint unit parameter"
        super().__init__(default=default, doc=doc, **kwargs)

    @staticmethod
    def _dimensionality(u: Any):
        """Return a unit's dimensionality.

        Args:
            u: The unit, or a parameter carrying one.

        Returns:
            The pint dimensionality.
        """
        value = getattr(u, "default", u)
        return UnitParam._normalize_unit(value).dimensionality

    @staticmethod
    def is_dimensionless(u: Any) -> bool:
        """Report whether a unit measures dimensionless.

        Args:
            u: The unit, or a parameter carrying one.

        Returns:
            True when the unit's dimensionality is dimensionless.
        """
        value = getattr(u, "default", u)
        if isinstance(value, str):
            return value in {"", "-", "dimensionless"}
        return bool(getattr(value, "dimensionless", False))

    @staticmethod
    def is_time(u: Any) -> bool:
        """Report whether a unit measures a time.

        Args:
            u: The unit, or a parameter carrying one.

        Returns:
            True when the unit's dimensionality is a time.
        """
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == units.s.dimensionality

    @staticmethod
    def is_distance(u: Any) -> bool:
        """Report whether a unit measures a distance.

        Args:
            u: The unit, or a parameter carrying one.

        Returns:
            True when the unit's dimensionality is a distance.
        """
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == units.m.dimensionality

    @staticmethod
    def is_angle(u: Any) -> bool:
        """Report whether a unit measures an angle.

        Args:
            u: The unit, or a parameter carrying one.

        Returns:
            True when the unit's dimensionality is an angle.
        """
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == units.rad.dimensionality

    @staticmethod
    def is_velocity(u: Any) -> bool:
        """Report whether a unit measures a velocity, translational or angular.

        Args:
            u: The unit, or a parameter carrying one.

        Returns:
            True when the unit's dimensionality is a velocity, translational or angular.
        """
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == (units.m / units.s).dimensionality or (
            value.dimensionality.get("[length]", 0) == 1
            and value.dimensionality.get("[time]", 0) == -1
        )

    @staticmethod
    def is_translational_velocity(u: Any) -> bool:
        """Report whether a unit measures a translational velocity.

        Args:
            u: The unit, or a parameter carrying one.

        Returns:
            True when the unit's dimensionality is a translational velocity.
        """
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == (units.m / units.s).dimensionality

    @staticmethod
    def is_translational_acceleration(u: Any) -> bool:
        """Report whether a unit measures a translational acceleration.

        Args:
            u: The unit, or a parameter carrying one.

        Returns:
            True when the unit's dimensionality is a translational acceleration.
        """
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == (units.m / units.s**2).dimensionality

    @staticmethod
    def is_angular_velocity(u: Any) -> bool:
        """Report whether a unit measures an angular velocity.

        Args:
            u: The unit, or a parameter carrying one.

        Returns:
            True when the unit's dimensionality is an angular velocity.
        """
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == (units.rad / units.s).dimensionality

    @staticmethod
    def is_angular_acceleration(u: Any) -> bool:
        """Report whether a unit measures an angular acceleration.

        Args:
            u: The unit, or a parameter carrying one.

        Returns:
            True when the unit's dimensionality is an angular acceleration.
        """
        value = UnitParam._normalize_unit(getattr(u, "default", u))
        return value.dimensionality == (units.rad / units.s**2).dimensionality

    @staticmethod
    def label(unit: Any) -> str:
        """Return a unit's display label.

        Args:
            unit: The unit, or a parameter carrying one.

        Returns:
            The unit rendered as text.
        """
        value = UnitParam._normalize_unit(getattr(unit, "default", unit))
        return str(value)


class TimeUnitParam(UnitParam):
    """A unit parameter restricted to units of time."""

    def __init__(self, default=units.s, doc: Optional[str] = None, **kwargs):
        """Build the parameter.

        Args:
            default: See the class attributes.
            doc: See the class attributes.
            **kwargs: Forwarded to the parent class.
        """
        if doc is None:
            doc = "Time unit parameter"
        if not UnitParam.is_time(default):
            raise ValueError(f"Expected a time unit, got {default!r}")
        super().__init__(default=default, doc=doc, **kwargs)


class DistanceUnitParam(UnitParam):
    """A unit parameter restricted to units of distance."""

    def __init__(self, default=units.m, doc: Optional[str] = None, **kwargs):
        """Build the parameter.

        Args:
            default: See the class attributes.
            doc: See the class attributes.
            **kwargs: Forwarded to the parent class.
        """
        if doc is None:
            doc = "Distance unit parameter"
        if not UnitParam.is_distance(default):
            raise ValueError(f"Expected a distance unit, got {default!r}")
        super().__init__(default=default, doc=doc, **kwargs)


class AngleUnitParam(UnitParam):
    """A unit parameter restricted to units of angle."""

    def __init__(self, default=units.rad, doc: Optional[str] = None, **kwargs):
        """Build the parameter.

        Args:
            default: See the class attributes.
            doc: See the class attributes.
            **kwargs: Forwarded to the parent class.
        """
        if doc is None:
            doc = "Angle unit parameter"
        if not UnitParam.is_angle(default):
            raise ValueError(f"Expected an angle unit, got {default!r}")
        super().__init__(default=default, doc=doc, **kwargs)


class TranslationalVelocityUnitParam(UnitParam):
    """A unit parameter restricted to units of translational velocity."""

    def __init__(self, default=units.m / units.s, doc: Optional[str] = None, **kwargs):
        """Build the parameter.

        Args:
            default: See the class attributes.
            doc: See the class attributes.
            **kwargs: Forwarded to the parent class.
        """
        if doc is None:
            doc = "Translational velocity unit parameter"
        if not UnitParam.is_translational_velocity(default):
            raise ValueError(f"Expected a translational velocity unit, got {default!r}")
        super().__init__(default=default, doc=doc, **kwargs)


class TranslationalAccelerationUnitParam(UnitParam):
    """A unit parameter restricted to units of translational acceleration."""

    def __init__(
        self, default=units.m / units.s**2, doc: Optional[str] = None, **kwargs
    ):
        """Build the parameter.

        Args:
            default: See the class attributes.
            doc: See the class attributes.
            **kwargs: Forwarded to the parent class.
        """
        if doc is None:
            doc = "Translational acceleration unit parameter"
        if not UnitParam.is_translational_acceleration(default):
            raise ValueError(
                f"Expected a translational acceleration unit, got {default!r}"
            )
        super().__init__(default=default, doc=doc, **kwargs)


class AngularVelocityUnitParam(UnitParam):
    """A unit parameter restricted to units of angular velocity."""

    def __init__(
        self, default=units.rad / units.s, doc: Optional[str] = None, **kwargs
    ):
        """Build the parameter.

        Args:
            default: See the class attributes.
            doc: See the class attributes.
            **kwargs: Forwarded to the parent class.
        """
        if doc is None:
            doc = "Angular velocity unit parameter"
        if not UnitParam.is_angular_velocity(default):
            raise ValueError(f"Expected an angular velocity unit, got {default!r}")
        super().__init__(default=default, doc=doc, **kwargs)


class AngularAccelerationUnitParam(UnitParam):
    """A unit parameter restricted to units of angular acceleration."""

    def __init__(
        self, default=units.rad / units.s**2, doc: Optional[str] = None, **kwargs
    ):
        """Build the parameter.

        Args:
            default: See the class attributes.
            doc: See the class attributes.
            **kwargs: Forwarded to the parent class.
        """
        if doc is None:
            doc = "Angular acceleration unit parameter"
        if not UnitParam.is_angular_acceleration(default):
            raise ValueError(f"Expected an angular acceleration unit, got {default!r}")
        super().__init__(default=default, doc=doc, **kwargs)


class TypeParam(param.Parameter):
    """Parameter storing a Python type object."""

    def __init__(self, default=type(None), **kwargs):
        """Build the parameter.

        Args:
            default: See the class attributes.
            **kwargs: Forwarded to the parent class.
        """
        if default is None:
            default = type(None)
        if not isinstance(default, type):
            raise TypeError("dtype must be a type object")
        super().__init__(default=default, **kwargs)

    def _validate(self, val):
        """Validate a value against this parameter's constraints.

        Args:
            val: The value under validation.

        Raises:
            ValueError: If the constraint is violated.
        """
        if val is None:
            return type(None)
        if not isinstance(val, type):
            raise TypeError("dtype must be a type object")
        return val
