from __future__ import annotations
import typing
from typing import Any, Optional, Sequence, Tuple, TypedDict

import random
from types import FunctionType

import numpy as np
import param

from .. import util

__all__: list[str] = [
    "List",
    "resolve_param_class",
    "Unit",
    "StringRobust",
    "PositiveNumber",
    "PositiveInteger",
    "Phase",
    "SignedPhase",
    "RangeRobust",
    "RangeInf",
    "PositiveRange",
    "PhaseRange",
    "OptionalPositiveNumber",
    "OptionalPositiveInteger",
    "RandomizedPhase",
    "RandomizedSignedPhase",
    "RandomizedColor",
    "OptionalPositiveRange",
    "OptionalPhaseRange",
    "OptionalSelector",
    "IntegerTuple",
    "PositiveIntegerTuple",
    "IntegerRange",
    "IntegerRangeOrdered",
    "PositiveIntegerRange",
    "PositiveIntegerRangeOrdered",
    "NegativeIntegerRangeOrdered",
    "OptionalPositiveIntegerRangeOrdered",
    "OptionalNegativeIntegerRangeOrdered",
    "NumericTuple2DRobust",
    "IntegerTuple2DRobust",
    "ListXYcoordinates",
    "XYLine",
    "ItemListParam",
    "ClassDict",
    "ClassAttr",
    "ModeSelector",
    "DataFrameIndexed",
    "StepDataFrame",
    "EndpointDataFrame",
]

__displayname__ = "Custom parameters"


def _is_null_value(val: Any) -> bool:
    """
    Robust check for None/nan/empty-string without triggering array truthiness.
    """
    if val is None:
        return True
    if isinstance(val, str):
        return val == ""
    if np.isscalar(val):
        try:
            return bool(np.isnan(val))
        except Exception:
            return False
    return False


class Unit(param.Parameter):
    """
    Parameter holding a pint physical unit (e.g. `reg.units.m`,
    `reg.units.dimensionless`), with class methods to derive its display
    forms directly from a unit instance -- a LaTeX-wrapped symbol string,
    or the empty-string/"-" placeholders conventionally used when the
    unit is `reg.units.dimensionless` specifically.

    A non-None `default` is resolved against the live `reg.units`
    registry at construction time (via `reg.units.Unit(str(default))`),
    so a plain string (e.g. `"m"`) or a pint Unit from a *different*
    UnitRegistry instance (e.g. one deserialized via pickle/JSON, which
    would otherwise fail equality/arithmetic checks against `reg.units` --
    see `LarvaworldParam.from_config`) both end up as a proper unit of
    this registry. `reg` is imported lazily inside `__init__` rather than
    at module level, since a top-level import here would be circular
    (`reg` imports this module); by the time any `Unit(...)` is actually
    constructed (always inside a function body, e.g.
    `get_LarvaworldParam`), the package has finished importing, so the
    deferred import is safe.

    The `symbol`/`label`/`is_dimensionless` display helpers below are
    kept dependency-free of `reg` by checking `str(u) == ""` rather than
    comparing against `reg.units.dimensionless` directly -- pint's
    dimensionless unit is the only one that stringifies to the empty
    string, so this is equivalent, and (unlike pint's own `u.dimensionless`
    flag, which is also True for e.g. radians, since angles are
    dimensionally trivial in SI) it matches exactly the *specific*
    dimensionless sentinel this codebase compared against before this
    class existed -- not every dimensionally-trivial unit.

    Args:
        default: Default unit -- a pint Unit instance, a string
            recognized by `reg.units` (e.g. "m"), or None.
        **kwargs: Additional keyword arguments passed to param.Parameter

    Example:
        >>> u_param = Unit(default="m")
        >>> Unit.symbol(u_param.default)
        '$meter$'
    """

    def __init__(self, default=None, **kwargs):
        if default is not None:
            from .. import reg

            default = reg.units.Unit(str(default))
        super().__init__(default=default, **kwargs)

    @staticmethod
    def is_dimensionless(u: Any) -> bool:
        """Whether `u` is specifically `reg.units.dimensionless` (see class docstring)."""
        return str(u) == ""

    @staticmethod
    def symbol(u: Any) -> str:
        """LaTeX-wrapped unit symbol (e.g. '$meter$'), or '-' if `u` is dimensionless."""
        return "-" if Unit.is_dimensionless(u) else rf"${u}$"

    @staticmethod
    def label(u: Any) -> str:
        """Parenthesized unit label (e.g. '(meter)'), or '' if `u` is dimensionless."""
        return "" if Unit.is_dimensionless(u) else f"({Unit.symbol(u)})"


class StringRobust(param.String):
    """
    Robust string parameter that converts any input to string.

    Extends param.String to automatically convert non-string inputs
    to string representation during initialization.

    Args:
        default: Default value (converted to str if not None)
        **kwargs: Additional keyword arguments passed to param.String

    Example:
        >>> str_param = StringRobust(default=123)
        >>> str_param.default  # "123"
    """

    def __init__(self, default="", **kwargs):
        if default is not None and not isinstance(default, str):
            default = str(default)
        super().__init__(default=default, **kwargs)


class PositiveNumber(param.Number):
    """
    Numeric parameter constrained to positive values.

    Extends param.Number with automatic positive bounds enforcement.
    Useful for physical quantities that must be non-negative (distances,
    frequencies, counts).

    Args:
        default: Default value (must be >= 0.0)
        softmin: Soft lower bound for UI sliders (default: 0.0)
        softmax: Soft upper bound for UI sliders (default: None)
        hardmin: Hard lower bound (default: 0.0, enforced)
        hardmax: Hard upper bound (default: None)
        bounds: Explicit bounds tuple (overrides hardmin/hardmax if provided)
        step: Step size for UI increments (default: 0.1)
        **kwargs: Additional keyword arguments passed to param.Number

    Example:
        >>> velocity = PositiveNumber(default=1.5, softmax=5.0, step=0.1)
    """

    def __init__(
        self,
        default=0.0,
        softmin=0.0,
        softmax=None,
        hardmin=0.0,
        hardmax=None,
        bounds=None,
        step=0.1,
        **kwargs,
    ):
        if bounds is None:
            bounds = (hardmin, hardmax)
        super().__init__(
            default=default,
            softbounds=(softmin, softmax),
            bounds=bounds,
            step=step,
            **kwargs,
        )


class PositiveInteger(param.Integer):
    """
    Integer parameter constrained to positive values.

    Extends param.Integer with automatic positive bounds enforcement.
    Useful for counts, indices, and discrete quantities that must be
    non-negative.

    Args:
        default: Default integer value (must be >= 0)
        softmin: Soft lower bound for UI sliders (default: 0)
        softmax: Soft upper bound for UI sliders (default: None)
        hardmin: Hard lower bound (default: 0, enforced)
        hardmax: Hard upper bound (default: None)
        step: Step size for UI increments (default: 1)
        **kwargs: Additional keyword arguments passed to param.Integer

    Example:
        >>> num_agents = PositiveInteger(default=10, softmax=100, step=5)
    """

    def __init__(
        self,
        default=0,
        softmin=0,
        softmax=None,
        hardmin=0,
        hardmax=None,
        step=1,
        **kwargs,
    ):
        super().__init__(
            default=default,
            softbounds=(softmin, softmax),
            bounds=(hardmin, hardmax),
            step=step,
            **kwargs,
        )


class Phase(param.Number):
    """
    Phase angle parameter constrained to [0, 2π] range.

    Extends param.Number for representing phase angles in radians,
    automatically bounded to the valid phase range [0, 2π]. See
    `SignedPhase` for the [-π, π] (centered/signed) variant.

    Args:
        default: Default phase value in radians (0.0 to 2π)
        softmin: Soft lower bound (default: 0.0)
        softmax: Soft upper bound (default: 2π)
        hardmin: Hard lower bound (default: 0.0, enforced)
        hardmax: Hard upper bound (default: 2π, enforced)
        step: Step size for UI increments (default: 0.1 radians)
        **kwargs: Additional keyword arguments passed to param.Number

    Example:
        >>> initial_phase = Phase(default=np.pi/2, step=0.05)
    """

    def __init__(
        self,
        default=0.0,
        softmin=0.0,
        softmax=2 * np.pi,
        hardmin=0.0,
        hardmax=2 * np.pi,
        step=0.1,
        **kwargs,
    ):
        super().__init__(
            default=default,
            softbounds=(softmin, softmax),
            bounds=(hardmin, hardmax),
            step=step,
            **kwargs,
        )


class SignedPhase(Phase):
    """
    Phase angle parameter constrained to [-π, π] range.

    Extends `Phase` for representing signed/centered phase angles in
    radians (e.g. turning angles, heading offsets), bounded to [-π, π]
    instead of `Phase`'s [0, 2π].

    Args:
        default: Default phase value in radians (-π to π)
        softmin: Soft lower bound (default: -π)
        softmax: Soft upper bound (default: π)
        hardmin: Hard lower bound (default: -π, enforced)
        hardmax: Hard upper bound (default: π, enforced)
        step: Step size for UI increments (default: 0.1 radians)
        **kwargs: Additional keyword arguments passed to Phase

    Example:
        >>> turn_angle = SignedPhase(default=0.0)
    """

    def __init__(
        self,
        default=0.0,
        softmin=-np.pi,
        softmax=np.pi,
        hardmin=-np.pi,
        hardmax=np.pi,
        step=0.1,
        **kwargs,
    ):
        super().__init__(
            default=default,
            softmin=softmin,
            softmax=softmax,
            hardmin=hardmin,
            hardmax=hardmax,
            step=step,
            **kwargs,
        )


class RangeRobust(param.Range):
    """
    Robust range parameter accepting both tuples and lists.

    Extends param.Range to automatically convert list inputs to tuples,
    providing more flexible range specification in configurations.

    Args:
        default: Default range as tuple or list (default: (0.0, 0.0))
        step: Step size for UI increments (default: 0.1)
        **kwargs: Additional keyword arguments passed to param.Range

    Example:
        >>> velocity_range = RangeRobust(default=[0.5, 2.0], step=0.1)
        >>> velocity_range.default  # (0.5, 2.0) - converted to tuple
    """

    def __init__(self, default=(0.0, 0.0), step=0.1, **kwargs):
        if default is not None and not isinstance(default, tuple):
            default = tuple(default)
        super().__init__(default=default, step=step, **kwargs)

    def _validate_value(self, val, allow_None):
        if val is not None and not isinstance(val, tuple):
            val = tuple(val)
        super(RangeRobust, self)._validate_value(val, allow_None)


class RangeInf(RangeRobust):
    """
    Range parameter allowing None values within tuple bounds.

    Extends RangeRobust to support unbounded ranges by accepting None
    for either lower or upper bound, enabling half-open intervals.

    Example:
        >>> unbounded_upper = RangeInf(default=(0.0, None))  # [0, ∞)
        >>> unbounded_lower = RangeInf(default=(None, 10.0))  # (-∞, 10]
    """

    def _validate_value(self, val, allow_None):
        super(param.NumericTuple, self)._validate_value(val, allow_None)
        if allow_None and val is None:
            return
        for n in val:
            if param._is_number(n) or allow_None and n is None:
                continue
            raise ValueError(
                "NumericTuple parameter %r only takes numeric "
                "values, not type %r." % (self.name, type(n))
            )

    def _validate_bounds(self, val, bounds, inclusive_bounds, kind):
        if bounds is not None:
            for pos, v in zip(["lower", "upper"], bounds):
                if v is None:
                    continue
                self._validate_bound_type(v, pos, kind)
        if kind == "softbound":
            return

        if bounds is None or (val is None and self.allow_None):
            return
        vmin, vmax = bounds
        incmin, incmax = inclusive_bounds
        for bound, v in zip(["lower", "upper"], val):
            if v is None and self.allow_None:
                continue
            too_low = (vmin is not None) and (v < vmin if incmin else v <= vmin)
            too_high = (vmax is not None) and (v > vmax if incmax else v >= vmax)
            if too_low or too_high:
                raise ValueError(
                    f"{param._utils._validate_error_prefix(self)} {bound} bound must be in "
                    f"range {self.rangestr()}, not {v}."
                )

    # def _validate_bounds(self, val, bounds, inclusive_bounds):
    #     if bounds is None or (val is None and self.allow_None):
    #         return
    #     vmin, vmax = bounds
    #     incmin, incmax = inclusive_bounds
    #     for bound, v in zip(['lower', 'upper'], val):
    #         if v is None and self.allow_None:
    #             continue
    #         too_low = (vmin is not None) and (v < vmin if incmin else v <= vmin)
    #         too_high = (vmax is not None) and (v > vmax if incmax else v >= vmax)
    #         if too_low or too_high:
    #             raise ValueError("Range parameter %r's %s bound must be in range %s."
    #                              % (self.name, bound, self.rangestr()))


class PositiveRange(RangeRobust):
    """
    Range parameter constrained to positive number tuples.

    Extends RangeRobust with automatic positive bounds enforcement
    for both lower and upper range values.

    Args:
        default: Default range tuple (both values >= 0.0)
        softmin: Soft lower bound (default: 0.0)
        softmax: Soft upper bound (default: None)
        hardmin: Hard lower bound (default: 0.0, enforced)
        hardmax: Hard upper bound (default: None)
        **kwargs: Additional keyword arguments passed to RangeRobust

    Example:
        >>> speed_range = PositiveRange(default=(0.5, 2.0), softmax=5.0)
    """

    def __init__(
        self,
        default=(0.0, 0.0),
        softmin=0.0,
        softmax=None,
        hardmin=0.0,
        hardmax=None,
        **kwargs,
    ):
        super().__init__(
            default=default,
            softbounds=(softmin, softmax),
            bounds=(hardmin, hardmax),
            **kwargs,
        )


class PhaseRange(RangeRobust):
    """
    Phase angle range parameter constrained to [0, 2π].

    Extends RangeRobust for representing phase angle ranges in radians,
    both bounds automatically constrained to [0, 2π].

    Args:
        default: Default phase range tuple (both values 0.0 to 2π)
        softmin: Soft lower bound (default: 0.0)
        softmax: Soft upper bound (default: 2π)
        hardmin: Hard lower bound (default: 0.0, enforced)
        hardmax: Hard upper bound (default: 2π, enforced)
        **kwargs: Additional keyword arguments passed to RangeRobust

    Example:
        >>> phase_bounds = PhaseRange(default=(0.0, np.pi), step=0.1)
    """

    def __init__(
        self,
        default=(0.0, 0.0),
        softmin=0.0,
        softmax=2 * np.pi,
        hardmin=0.0,
        hardmax=2 * np.pi,
        **kwargs,
    ):
        super().__init__(
            default=default,
            softbounds=(softmin, softmax),
            bounds=(hardmin, hardmax),
            **kwargs,
        )


class OptionalPositiveNumber(param.Number):
    """
    Optional numeric parameter constrained to positive values or None.

    Extends param.Number with positive bounds and explicit None support,
    useful for optional physical quantities that when specified must be positive.

    Args:
        default: Default value (None or >= 0.0, default: None)
        softmin: Soft lower bound (default: 0.0)
        softmax: Soft upper bound (default: None)
        hardmin: Hard lower bound (default: 0.0, enforced when not None)
        hardmax: Hard upper bound (default: None)
        step: Step size for UI increments (default: 0.1)
        **kwargs: Additional keyword arguments passed to param.Number

    Example:
        >>> max_duration = OptionalPositiveNumber(default=None, softmax=1000.0)
    """

    def __init__(
        self,
        default=None,
        softmin=0.0,
        softmax=None,
        hardmin=0.0,
        hardmax=None,
        step=0.1,
        **kwargs,
    ):
        super().__init__(
            default=default,
            softbounds=(softmin, softmax),
            bounds=(hardmin, hardmax),
            step=step,
            allow_None=True,
            **kwargs,
        )


class OptionalPositiveInteger(param.Integer):
    """
    Optional integer parameter constrained to positive values or None.

    Extends param.Integer with positive bounds and explicit None support,
    useful for optional counts or indices that when specified must be positive.

    Args:
        default: Default value (None or >= 0, default: None)
        softmin: Soft lower bound (default: 0)
        softmax: Soft upper bound (default: None)
        hardmin: Hard lower bound (default: 0, enforced when not None)
        hardmax: Hard upper bound (default: None)
        step: Step size for UI increments (default: 1)
        **kwargs: Additional keyword arguments passed to param.Integer

    Example:
        >>> max_iterations = OptionalPositiveInteger(default=None, softmax=1000)
    """

    def __init__(
        self,
        default=None,
        softmin=0,
        softmax=None,
        hardmin=0,
        hardmax=None,
        step=1,
        **kwargs,
    ):
        super().__init__(
            default=default,
            softbounds=(softmin, softmax),
            bounds=(hardmin, hardmax),
            step=step,
            allow_None=True,
            **kwargs,
        )


class RandomizedPhase(Phase):
    """
    Phase parameter with automatic random initialization.

    Extends Phase to randomly initialize from uniform [0, 2π] distribution
    when default is None or np.nan, useful for randomized initial conditions.
    See `RandomizedSignedPhase` for the [-π, π] (centered/signed) variant.

    Args:
        default: Initial phase (if None/nan, randomly sampled from [0, 2π])
        **kwargs: Additional keyword arguments passed to Phase

    Example:
        >>> random_phase = RandomizedPhase(default=None)  # Random each time
    """

    def __init__(self, default=None, **kwargs):
        if _is_null_value(default):
            default = np.random.uniform(0, 2 * np.pi)
        super().__init__(default=default, allow_None=True, **kwargs)

    def _validate_value(self, val, allow_None):
        if _is_null_value(val):
            val = np.random.uniform(0, 2 * np.pi)
        super(RandomizedPhase, self)._validate_value(val, allow_None)


class RandomizedSignedPhase(SignedPhase):
    """
    Signed phase parameter with automatic random initialization.

    Extends SignedPhase to randomly initialize from a uniform [-π, π]
    distribution when default is None or np.nan, useful for randomized
    initial conditions expressed in signed/centered form (e.g. random
    initial turning angle or heading offset).

    Args:
        default: Initial phase (if None/nan, randomly sampled from [-π, π])
        **kwargs: Additional keyword arguments passed to SignedPhase

    Example:
        >>> random_turn = RandomizedSignedPhase(default=None)  # Random each time
    """

    def __init__(self, default=None, **kwargs):
        if _is_null_value(default):
            default = np.random.uniform(-np.pi, np.pi)
        super().__init__(default=default, allow_None=True, **kwargs)

    def _validate_value(self, val, allow_None):
        if _is_null_value(val):
            val = np.random.uniform(-np.pi, np.pi)
        super(RandomizedSignedPhase, self)._validate_value(val, allow_None)


class RandomizedColor(param.Color):
    """
    Color parameter with automatic random initialization.

    Extends param.Color to randomly select from named colors when
    default is None/nan/empty, useful for auto-coloring agents or objects.

    Args:
        default: Initial color (if None/nan/"", randomly selected from named colors)
        instantiate: Create unique instances per parameter (default: True)
        allow_None: Allow None values (default: True)
        per_instance: Different values per class instance (default: True)
        **kwargs: Additional keyword arguments passed to param.Color

    Example:
        >>> agent_color = RandomizedColor(default=None)  # Random named color
    """

    def __init__(
        self,
        default=None,
        instantiate=True,
        allow_None=True,
        per_instance=True,
        **kwargs,
    ):
        if _is_null_value(default):
            default = random.choice(super()._named_colors)
        super().__init__(
            default=default,
            instantiate=instantiate,
            allow_None=allow_None,
            per_instance=per_instance,
            **kwargs,
        )

    def _validate_value(self, val, allow_None):
        if _is_null_value(val):
            val = random.choice(super()._named_colors)
        super(RandomizedColor, self)._validate_value(val, allow_None)


class OptionalPositiveRange(RangeInf):
    """
    Optional range parameter constrained to positive tuples or None.

    Extends RangeInf with positive bounds and None support for entire range,
    useful for optional bounded intervals that must be positive when specified.

    Args:
        default: Default range (None or tuple with values >= 0.0)
        softmin: Soft lower bound (default: 0.0)
        softmax: Soft upper bound (default: None)
        hardmin: Hard lower bound (default: 0.0, enforced)
        hardmax: Hard upper bound (default: None)
        **kwargs: Additional keyword arguments passed to RangeInf

    Example:
        >>> optional_range = OptionalPositiveRange(default=None, softmax=10.0)
    """

    def __init__(
        self,
        default=None,
        softmin=0.0,
        softmax=None,
        hardmin=0.0,
        hardmax=None,
        **kwargs,
    ):
        super().__init__(
            default=default,
            softbounds=(softmin, softmax),
            bounds=(hardmin, hardmax),
            allow_None=True,
            **kwargs,
        )


class OptionalPhaseRange(RangeRobust):
    """
    Optional phase range parameter constrained to [0, 2π] or None.

    Extends RangeRobust for optional phase angle ranges, with both
    bounds constrained to [0, 2π] when range is specified.

    Args:
        default: Default phase range (None or tuple with values 0.0 to 2π)
        softmin: Soft lower bound (default: 0.0)
        softmax: Soft upper bound (default: 2π)
        hardmin: Hard lower bound (default: 0.0, enforced)
        hardmax: Hard upper bound (default: 2π, enforced)
        **kwargs: Additional keyword arguments passed to RangeRobust

    Example:
        >>> phase_range = OptionalPhaseRange(default=(0.0, np.pi))
    """

    def __init__(
        self,
        default=None,
        softmin=0.0,
        softmax=2 * np.pi,
        hardmin=0.0,
        hardmax=2 * np.pi,
        **kwargs,
    ):
        super().__init__(
            default=default,
            softbounds=(softmin, softmax),
            bounds=(hardmin, hardmax),
            **kwargs,
        )


class OptionalSelector(param.Selector):
    """
    Selector parameter with automatic None support.

    Extends param.Selector to allow None as default value even when
    None is not in the objects list, useful for optional selections.

    Args:
        objects: List of valid selectable objects
        default: Default selected object (None allowed even if not in objects)
        **kwargs: Additional keyword arguments passed to param.Selector

    Example:
        >>> mode_select = OptionalSelector(objects=['A', 'B', 'C'], default=None)
    """

    def __init__(self, objects, default=None, **kwargs):
        kws = {
            "default": default,
            "objects": objects,
            # 'doc': f'The {conftype0.default} configuration ID',
            **kwargs,
        }
        if default is None:
            kws["empty_default"] = True
            kws["allow_None"] = True
        super().__init__(**kws)


class IntegerTuple(param.NumericTuple):
    """
    Numeric tuple parameter constrained to integer values.

    Extends param.NumericTuple to enforce that all tuple elements
    are integers, rejecting float or other numeric types.

    Example:
        >>> int_coords = IntegerTuple(default=(10, 20, 30), length=3)
    """

    def _validate_value(self, val, allow_None):
        super(param.NumericTuple, self)._validate_value(val, allow_None)
        for n in val:
            if isinstance(n, int):
                continue
            raise ValueError(
                "IntegerTuple parameter %r only takes integer "
                "values, not type %r." % (self.name, type(n))
            )


class IntegerRange(RangeRobust):
    """
    Range parameter constrained to integer tuple values.

    Extends RangeRobust to enforce both range bounds are integers,
    useful for discrete intervals and index ranges.

    Args:
        default: Default integer range tuple (default: (0, 0))
        step: Step size for UI increments (default: 1)
        **kwargs: Additional keyword arguments passed to RangeRobust

    Example:
        >>> age_range = IntegerRange(default=(0, 100), step=5)
    """

    def __init__(self, default=(0, 0), step=1, **kwargs):
        super().__init__(default=default, step=step, **kwargs)

    def _validate_value(self, val, allow_None):
        super(RangeRobust, self)._validate_value(val, allow_None)
        for n in val:
            if isinstance(n, int):
                continue
            raise ValueError(
                "IntegerRange parameter %r only takes integer "
                "values, not type %r." % (self.name, type(n))
            )


class PositiveIntegerTuple(IntegerTuple):
    """
    Integer tuple parameter constrained to non-negative values.

    Extends IntegerTuple with automatic positive bounds enforcement
    for every tuple element.

    Args:
        default: Default integer tuple (all values >= 0)
        softmin: Soft lower bound (default: 0)
        softmax: Soft upper bound (default: None)
        hardmin: Hard lower bound (default: 0, enforced)
        hardmax: Hard upper bound (default: None)
        **kwargs: Additional keyword arguments passed to IntegerTuple

    Example:
        >>> grid_dims = PositiveIntegerTuple(default=(51, 51), length=2, softmax=500)
    """

    __slots__ = ["softmin", "softmax", "hardmin", "hardmax"]

    def __init__(
        self,
        default=(0, 0),
        softmin=0,
        softmax=None,
        hardmin=0,
        hardmax=None,
        **kwargs,
    ):
        self.softmin = softmin
        self.softmax = softmax
        self.hardmin = hardmin
        self.hardmax = hardmax
        super().__init__(default=default, **kwargs)

    def _validate_value(self, val, allow_None):
        super()._validate_value(val, allow_None)
        for n in val:
            if self.hardmin is not None and n < self.hardmin:
                raise ValueError(
                    "PositiveIntegerTuple parameter %r only takes values >= %r."
                    % (self.name, self.hardmin)
                )
            if self.hardmax is not None and n > self.hardmax:
                raise ValueError(
                    "PositiveIntegerTuple parameter %r only takes values <= %r."
                    % (self.name, self.hardmax)
                )


class IntegerRangeOrdered(IntegerRange):
    """
    Ordered integer range parameter enforcing lower <= upper.

    Extends IntegerRange with validation to ensure first value
    is less than or equal to second value in tuple.

    Example:
        >>> ordered_range = IntegerRangeOrdered(default=(5, 15))  # OK
        >>> ordered_range = IntegerRangeOrdered(default=(15, 5))  # Raises ValueError
    """

    def _validate_value(self, val, allow_None):
        super(IntegerRange, self)._validate_value(val, allow_None)
        v1, v2 = val
        assert v1 <= v2
        # raise ValueError("IntegerRange parameter %r only takes integer "
        #                      "values, not type %r." % (self.name, type(n)))


class PositiveIntegerRange(IntegerRange):
    """
    Integer range parameter constrained to positive values.

    Extends IntegerRange with automatic positive bounds enforcement
    for both range endpoints.

    Args:
        default: Default integer range (both values >= 0)
        softmin: Soft lower bound (default: 0)
        softmax: Soft upper bound (default: None)
        hardmin: Hard lower bound (default: 0, enforced)
        hardmax: Hard upper bound (default: None)
        **kwargs: Additional keyword arguments passed to IntegerRange

    Example:
        >>> count_range = PositiveIntegerRange(default=(10, 50), softmax=100)
    """

    def __init__(
        self, default=(0, 0), softmin=0, softmax=None, hardmin=0, hardmax=None, **kwargs
    ):
        super().__init__(
            default=default,
            softbounds=(softmin, softmax),
            bounds=(hardmin, hardmax),
            **kwargs,
        )


class PositiveIntegerRangeOrdered(IntegerRangeOrdered):
    """
    Ordered positive integer range parameter.

    Combines IntegerRangeOrdered with positive bounds, ensuring both
    ordering (lower <= upper) and positivity constraints.

    Args:
        default: Default ordered integer range (both >= 0, first <= second)
        softmin: Soft lower bound (default: 0)
        softmax: Soft upper bound (default: None)
        hardmin: Hard lower bound (default: 0, enforced)
        hardmax: Hard upper bound (default: None)
        **kwargs: Additional keyword arguments passed to IntegerRangeOrdered

    Example:
        >>> id_range = PositiveIntegerRangeOrdered(default=(5, 20), softmax=100)
    """

    def __init__(
        self, default=(0, 1), softmin=0, softmax=None, hardmin=0, hardmax=None, **kwargs
    ):
        super().__init__(
            default=default,
            softbounds=(softmin, softmax),
            bounds=(hardmin, hardmax),
            **kwargs,
        )


class NegativeIntegerRangeOrdered(IntegerRangeOrdered):
    """
    Ordered negative integer range parameter.

    Combines IntegerRangeOrdered with negative/zero upper bound, ensuring
    both ordering and non-positive constraints (useful for negative indices).

    Args:
        default: Default ordered integer range (both <= 0, first <= second)
        softmin: Soft lower bound (default: None)
        softmax: Soft upper bound (default: 0)
        hardmin: Hard lower bound (default: None)
        hardmax: Hard upper bound (default: 0, enforced)
        **kwargs: Additional keyword arguments passed to IntegerRangeOrdered

    Example:
        >>> negative_range = NegativeIntegerRangeOrdered(default=(-10, -1))
    """

    def __init__(
        self,
        default=(-1, 0),
        softmin=None,
        softmax=0,
        hardmin=None,
        hardmax=0,
        **kwargs,
    ):
        super().__init__(
            default=default,
            softbounds=(softmin, softmax),
            bounds=(hardmin, hardmax),
            **kwargs,
        )


class OptionalPositiveIntegerRangeOrdered(PositiveIntegerRangeOrdered):
    """
    Optional ordered positive integer range parameter.

    Accepts either an ordered positive integer tuple or ``None``.
    """

    def __init__(
        self,
        default=None,
        softmin=0,
        softmax=None,
        hardmin=0,
        hardmax=None,
        **kwargs,
    ):
        super().__init__(
            default=default,
            softmin=softmin,
            softmax=softmax,
            hardmin=hardmin,
            hardmax=hardmax,
            allow_None=True,
            **kwargs,
        )

    def _validate_value(self, val, allow_None):
        if val is None and allow_None:
            return
        super()._validate_value(val, allow_None)


class OptionalNegativeIntegerRangeOrdered(NegativeIntegerRangeOrdered):
    """
    Optional ordered negative integer range parameter.

    Accepts either an ordered negative integer tuple or ``None``.
    """

    def __init__(
        self,
        default=None,
        softmin=None,
        softmax=0,
        hardmin=None,
        hardmax=0,
        **kwargs,
    ):
        super().__init__(
            default=default,
            softmin=softmin,
            softmax=softmax,
            hardmin=hardmin,
            hardmax=hardmax,
            allow_None=True,
            **kwargs,
        )

    def _validate_value(self, val, allow_None):
        if val is None and allow_None:
            return
        super()._validate_value(val, allow_None)


class NumericTuple2DRobust(param.NumericTuple):
    """
    2D numeric tuple parameter accepting both tuples and lists.

    Extends param.NumericTuple with automatic list-to-tuple conversion
    and fixed length=2, useful for XY coordinates and 2D vectors.

    Args:
        default: Default 2D point as tuple or list (default: (0.0, 0.0))
        **kwargs: Additional keyword arguments passed to param.NumericTuple

    Example:
        >>> position = NumericTuple2DRobust(default=[10.5, 20.3])
        >>> position.default  # (10.5, 20.3) - converted to tuple
    """

    def __init__(self, default=(0.0, 0.0), **kwargs):
        if not isinstance(default, tuple):
            default = tuple(default)

        super().__init__(default=default, length=2, **kwargs)


class IntegerTuple2DRobust(IntegerTuple):
    """
    2D integer tuple parameter accepting both tuples and lists.

    Extends IntegerTuple with automatic list-to-tuple conversion
    and fixed length=2, useful for pixel coordinates and grid indices.

    Args:
        default: Default 2D integer point as tuple or list (default: (0, 0))
        **kwargs: Additional keyword arguments passed to IntegerTuple

    Example:
        >>> pixel_pos = IntegerTuple2DRobust(default=[100, 200])
        >>> pixel_pos.default  # (100, 200) - converted to tuple
    """

    def __init__(self, default=(0, 0), **kwargs):
        if not isinstance(default, tuple):
            default = tuple(default)
        super().__init__(default=default, length=2, **kwargs)


class List(param.List):
    """
    param.List variant with list-length constraints under an unambiguous
    `length` name, instead of param.List's own overloaded `bounds` (which
    means item count, not value bounds -- easy to confuse with
    param.Number's `bounds`, which is value bounds).

    Args:
        length: (min, max) item-count bounds, or None for unbounded.
        **kwargs: Additional keyword arguments passed to param.List.
    """

    def __init__(self, default=None, length=None, **kwargs):
        super().__init__(
            default=default if default is not None else [], bounds=length, **kwargs
        )


class ListXYcoordinates(List):
    """
    List parameter for XY coordinate tuples.

    Extends List with tuple item_type and length bounds,
    useful for polylines, paths, and multi-point geometries.

    Args:
        default: Default list of XY tuples (default: [])
        minlen: Minimum list length (default: 0)
        maxlen: Maximum list length (default: None)
        **kwargs: Additional keyword arguments passed to param.List

    Example:
        >>> path = ListXYcoordinates(default=[(0,0), (10,5), (20,10)], minlen=2)
    """

    def __init__(self, default=[], minlen=0, maxlen=None, **kwargs):
        super().__init__(
            default=default, item_type=tuple, length=(minlen, maxlen), **kwargs
        )


class XYLine(ListXYcoordinates):
    """
    XY coordinate list parameter for line geometries.

    Extends ListXYcoordinates as specialized alias for line/polyline
    definitions in spatial configurations.

    Args:
        minlen: Minimum number of points (default: 0)
        **kwargs: Additional keyword arguments passed to ListXYcoordinates

    Example:
        >>> boundary = XYLine(default=[(0,0), (100,0), (100,100), (0,100)])
    """

    def __init__(self, minlen=0, **kwargs):
        super().__init__(minlen=minlen, **kwargs)


class ItemListParam(List):
    """
    Parameter for managed lists with ItemList functionality.

    Extends List to enable list management functionality provided by the
    lib.util.ItemList class, which inherits from a custom SuperList class
    as well as from agentpy.AgentSequence for agent-based modeling.

    Attributes:
        size: Tuple (min, max) specifying valid list length bounds
        bounds: Inherited bounds attribute
        item_type: Type constraint for list items
        class_: Class constraint for list items

    Args:
        default: Default ItemList instance (default: empty ItemList())
        size: Length bounds tuple (min, max) where None = unbounded
        **params: Additional keyword arguments passed to param.List

    Example:
        >>> agents = ItemListParam(default=util.ItemList(), size=(1, 100))
    """

    __slots__ = ["bounds", "item_type", "class_", "size"]

    def __init__(self, default=util.ItemList(), size=(0, None), **params):
        self.size = size
        if isinstance(default, list):
            default = util.ItemList(default)
        List.__init__(self, default=default, **params)
        self._validate(default)


class ClassDict(param.ClassSelector):
    """
    Dictionary parameter with class-constrained values.

    Extends param.ClassSelector for AttrDict values where all dict items
    must be instances of a specified type, useful for typed configuration dicts.

    Attributes:
        item_type: Required type for all dictionary values (None = no constraint)
        class_: Fixed to util.AttrDict
        is_instance: Inherited from ClassSelector

    Args:
        default: Default AttrDict instance (default: empty AttrDict())
        item_type: Required type for dict values (default: None = unconstrained)
        **params: Additional keyword arguments passed to param.ClassSelector

    Example:
        >>> configs = ClassDict(default=util.AttrDict(), item_type=NestedConf)
    """

    __slots__ = ["class_", "is_instance", "item_type"]

    def __init__(self, default=util.AttrDict(), item_type=None, **params):
        self.item_type = item_type
        param.ClassSelector.__init__(self, util.AttrDict, default=default, **params)

    def _validate(self, val):
        super(param.ClassSelector, self)._validate(val)
        self._validate_item_type(val, self.item_type)

    def _validate_item_type(self, val, item_type):
        if item_type is None or (self.allow_None and val is None):
            return
        for k, v in val.items():
            if isinstance(v, item_type):
                continue
            raise TypeError(
                "ClassDict parameter %r items must be instances "
                "of type %r, not %r." % (self.name, item_type, val)
            )


class ClassAttr(param.ClassSelector):
    """
    Class instance parameter with automatic initialization.

    Extends param.ClassSelector to automatically instantiate class from
    dict configs, supporting both instance and config dict as defaults.

    Args:
        class_: Target class or tuple of classes for validation
        **kwargs: Default value (as instance or config dict) plus ClassSelector args

    Example:
        >>> brain_param = ClassAttr(class_=Brain, default={'olfactor': {...}})
        >>> brain_param.default  # Brain instance created from dict
    """

    def __init__(self, class_, **kwargs):
        if not isinstance(class_, tuple):
            cc = class_
        else:
            cc = class_[0]
        if "default" not in kwargs:
            kwargs["default"] = cc()
        elif kwargs["default"] is None:
            kwargs["default"] = None
        elif not isinstance(kwargs["default"], class_):
            kwargs["default"] = cc(**kwargs["default"])
        super().__init__(class_=class_, **kwargs)


class ModeSelector(ClassAttr):
    """
    Mode selector parameter for choosing among class variants.

    Extends ClassAttr to select class from a dict of options by ID key,
    useful for behavior mode selection (e.g., 'RL' vs 'MB' memory types).

    Attributes:
        classDict: Dictionary mapping mode IDs to class types
        classID: Currently selected mode ID

    Args:
        classDict: Dict mapping mode names to classes (default: empty AttrDict())
        classID: Initial mode selection (default: None)
        class_: Explicit class override (overrides classDict[classID] if provided)
        **kwargs: Additional keyword arguments passed to ClassAttr

    Example:
        >>> memory_mode = ModeSelector(
        ...     classDict={'RL': RLmemory, 'MB': MBmemory},
        ...     classID='RL'
        ... )
    """

    __slots__ = ["classDict", "classID"]

    def __init__(self, classDict=util.AttrDict(), classID=None, class_=None, **kwargs):
        self.classDict = classDict
        self.classID = classID
        # if classID is None and len(classDict.keylist)>0:
        #     classID=classDict.keylist[0]
        if classID is not None:
            class_ = classDict[classID]
        super().__init__(class_=class_, **kwargs)


class DataFrameIndexed(param.DataFrame):
    """
    DataFrame parameter with index level validation.

    Extends param.DataFrame to enforce specific index level names,
    useful for structured datasets with required multi-index levels.

    Attributes:
        levels: Required index level names (None = no validation)
        rows: Inherited row constraint
        columns: Inherited column constraint
        ordered: Inherited ordering constraint

    Args:
        levels: List of required index level names (default: None)
        **params: Additional keyword arguments passed to param.DataFrame

    Example:
        >>> trajectory_df = DataFrameIndexed(
        ...     levels=['AgentID', 'Step'],
        ...     columns=['x', 'y', 'orientation']
        ... )
    """

    __slots__ = ["rows", "columns", "ordered", "levels"]

    def __init__(self, levels=None, **params):
        self.levels = levels
        param.DataFrame.__init__(self, **params)

    def _validate(self, val):
        super(param.DataFrame, self)._validate(val)
        self._validate_levels(val, self.levels)

    def _validate_levels(self, val, levels):
        if levels is None or (self.allow_None and val is None):
            return
        val_levels = list(val.index.names)
        if val_levels != levels:
            raise TypeError(
                "DataFrameIndexed parameter %r levels must be "
                " %r, not %r." % (self.name, levels, val_levels)
            )


class StepDataFrame(DataFrameIndexed):
    """
    DataFrame parameter for step-by-step timeseries data.

    Extends DataFrameIndexed with fixed index levels ['Step', 'AgentID'],
    used for trajectory and timeseries datasets across simulation steps.
    Each row represents one agent's state at one timestep.

    Args:
        **params: Additional keyword arguments passed to DataFrameIndexed
                  (e.g., columns=['x', 'y', 'v', 'a'])

    Example:
        >>> step_data = StepDataFrame(columns=['x', 'y', 'v', 'a'])
        >>> # Expected index: MultiIndex(['Step', 'AgentID'])
    """

    def __init__(self, **params):
        DataFrameIndexed.__init__(self, levels=["Step", "AgentID"], **params)


class EndpointDataFrame(DataFrameIndexed):
    """
    DataFrame parameter for endpoint/summary data per agent.

    Extends DataFrameIndexed with fixed index level ['AgentID'],
    used for final metrics and endpoint statistics aggregated per agent.
    Each row represents one agent's complete trajectory summary.

    Args:
        **params: Additional keyword arguments passed to DataFrameIndexed
                  (e.g., columns=['cum_d', 'cum_dur', 'max_v'])

    Example:
        >>> endpoint_data = EndpointDataFrame(columns=['cum_d', 'cum_dur', 'max_v'])
        >>> # Expected index: Index(['AgentID'])
    """

    def __init__(self, **params):
        DataFrameIndexed.__init__(self, levels=["AgentID"], **params)


#: dtype -> builtin param.Parameter class, for dtypes with no more specific
#: custom-class match in `resolve_param_class` below. `str`, `list` and its
#: typed variants, and both `Tuple[float]`/`Tuple[int]` are intentionally
#: absent here: `resolve_param_class` always resolves those to a more
#: specific custom class before reaching this fallback, so a builtin
#: mapping for them here would be dead code.
_DTYPE_TO_PARAM_CLASS: dict[Any, type[param.Parameter]] = {
    float: param.Number,
    int: param.Integer,
    bool: param.Boolean,
    dict: param.Dict,
    type: param.ClassSelector,
    FunctionType: param.Callable,
    TypedDict: param.Dict,
}

#: A float lim of exactly (0.0, 2*pi) -- used to detect Phase/PhaseRange.
_PHASE_LIM = (0.0, 2 * np.pi)

#: A float lim of exactly (-pi, pi) -- used to detect SignedPhase.
_SIGNED_PHASE_LIM = (-np.pi, np.pi)


def _select_param_class(
    dtype: Any, lim: Optional[tuple[Any, Any]], vs: Optional[list[Any]]
) -> type[param.Parameter]:
    """
    Select the param.Parameter (sub)class best matching a given data type,
    value limit, and value options -- preferring the more semantically
    precise custom classes in this module (PositiveNumber, Phase,
    RangeRobust, ...) over the generic builtins where the dtype/lim shape
    matches one of them, and falling back to the builtin otherwise.

    Args:
        dtype: The data type of the parameter.
        lim: (min, max) bounds for the parameter, or None.
        vs: The value options of the parameter, or None.

    Returns:
        type[param.Parameter]: The selected Param class.
    """
    if dtype == float and lim == (0.0, 1.0):
        return param.Magnitude
    if type(vs) == list and dtype in [str, int]:
        return param.Selector
    if dtype == float and lim is not None:
        if lim == _PHASE_LIM:
            return Phase
        if lim == _SIGNED_PHASE_LIM:
            return SignedPhase
        if lim[0] == 0.0:
            return PositiveNumber
    if dtype == int and lim is not None and lim[0] == 0:
        return PositiveInteger
    if dtype == str:
        return StringRobust
    if dtype == typing.Tuple[float]:
        if lim is not None:
            if lim == _PHASE_LIM:
                return PhaseRange
            if lim[0] == 0.0:
                return PositiveRange
        return RangeRobust
    if dtype == typing.Tuple[int]:
        return IntegerTuple
    if dtype == typing.List[typing.Tuple[float]]:
        return ListXYcoordinates
    if dtype in (list, typing.List[int], typing.List[str], typing.List[float]):
        return List
    if dtype in _DTYPE_TO_PARAM_CLASS:
        return _DTYPE_TO_PARAM_CLASS[dtype]
    return param.Parameter


def _accepts_kwarg(param_class: type[param.Parameter], name: str, value: Any) -> bool:
    """
    Whether `param_class`'s constructor accepts a keyword argument named
    `name` -- verified by an actual construction attempt with `value`,
    rather than a hardcoded per-class table. Several classes in this
    module build a kwarg like `bounds` internally from their own named
    args (e.g. PositiveNumber's `hardmin`/`hardmax`) and raise "multiple
    values for keyword argument" if that same name is *also* forwarded
    generically via **kwargs, while others (RangeRobust, param.Range, ...)
    forward it through unchanged and accept it fine -- a distinction only
    visible by actually attempting the call, not by inspecting signatures.

    A TypeError means `name` isn't a usable keyword for this class (either
    unexpected, duplicated against one the class already supplies
    internally, or a required positional arg is missing so the call can't
    be evaluated at all -- treated as "not accepted" too, since we can't
    tell). Any other exception (e.g. a ValueError from validating `value`
    against the class's own rules) means the keyword itself was accepted;
    only the probe value was rejected, which doesn't matter here.
    """
    try:
        param_class(**{name: value})
    except TypeError:
        return False
    except Exception:
        return True
    return True


#: Candidate keyword-argument name(s) for expressing a generic (min, max)
#: `lim` against an arbitrary param.Parameter subclass, tried in this
#: order: split min/max pairs first for classes that build their own
#: bounds-like kwarg internally from separately named args (PositiveNumber
#: & co.'s `hardmin`/`hardmax`, ListXYcoordinates' `minlen`/`maxlen`) --
#: trying the single-name forms first would wrongly match these via their
#: inherited **kwargs forwarding, but without going through the class's
#: own hard-min/max-length plumbing. Single paired kwargs (`bounds`,
#: `length`) are tried after, for classes with no split-pair alternative
#: (plain param.Number/param.Range/RangeRobust, List).
_LIM_KWARG_NAMES: tuple[Any, ...] = (
    ("hardmin", "hardmax"),
    ("minlen", "maxlen"),
    "bounds",
    "length",
)

#: Candidate keyword-argument name(s) for mirroring that same `lim` to a
#: class's *soft* bounds (the slider-rendering range), tried independently
#: of -- and in addition to -- `_LIM_KWARG_NAMES`, so slider rendering
#: doesn't visibly change relative to a plain `bounds=lim` parameter.
#: Split `softmin`/`softmax` (PositiveNumber & co.) is tried before the
#: single paired `softbounds` (plain param.Number/param.Range/RangeRobust)
#: for the same reason the hard-bounds candidates are ordered that way.
_SOFT_LIM_KWARG_NAMES: tuple[Any, ...] = (
    ("softmin", "softmax"),
    "softbounds",
)


def _lim_kwargs(
    param_class: type[param.Parameter], lim: tuple[Any, Any]
) -> dict[str, Any]:
    """
    Translate a generic (min, max) `lim` into whichever bounds-shaped
    kwarg(s) `param_class` actually accepts -- both the hard bounds (see
    `_LIM_KWARG_NAMES`) and, independently, the soft bounds (see
    `_SOFT_LIM_KWARG_NAMES`) -- each probed via `_accepts_kwarg` against
    the class itself, so any class exposing either concept gets `lim`
    mirrored into it. Classes whose own hardcoded defaults already encode
    the exact `lim` they were selected for (Phase/PhaseRange, chosen only
    when `lim == (0, 2*pi)`, their own default bounds) end up with the
    same effective values via the `hardmin`/`softmin` path, so no
    special-cased no-op branch is needed.
    """
    kwargs: dict[str, Any] = {}
    for candidates in (_LIM_KWARG_NAMES, _SOFT_LIM_KWARG_NAMES):
        for names in candidates:
            if isinstance(names, tuple):
                min_name, max_name = names
                if _accepts_kwarg(param_class, min_name, lim[0]):
                    kwargs[min_name] = lim[0]
                    kwargs[max_name] = lim[1]
                    break
            elif _accepts_kwarg(param_class, names, lim):
                kwargs[names] = lim
                break
    return kwargs


def resolve_param_class(
    dtype: Any,
    param_class: Optional[type[param.Parameter]] = None,
    **kwargs: Any,
) -> tuple[type[param.Parameter], dict[str, Any]]:
    """
    Resolve the param.Parameter (sub)class best matching `dtype` and the
    given attributes, together with the constructor kwargs to instantiate
    it with.

    Class selection (when `param_class` isn't given) delegates to
    `_select_param_class`, preferring the more semantically precise custom
    classes in this module over the generic builtins where the dtype/lim/
    vs shape matches one of them.

    kwargs building translates generic attribute names (`lim`, `dv`, `vs`,
    `v0`) into whichever concrete constructor kwarg(s) the resolved class
    actually accepts (`bounds`/`length`/`hardmin`+`hardmax` (mirrored to
    matching `softmin`/`softmax`)/`minlen`+`maxlen`, `step`, `objects`,
    `default`) -- verified against the class itself via `_accepts_kwarg`
    rather than a hardcoded per-class table, so any new param.Parameter
    subclass added to this module is supported automatically. Raw
    constructor kwarg names (`bounds`, `objects`, `default`, `label`) are
    also accepted directly, as aliases for their generic counterpart.

    Args:
        dtype: The data type of the parameter.
        param_class: Explicit param.Parameter subclass to use, or None to
            auto-select it from `dtype`/`lim`/`vs`.
        **kwargs: Attribute values used to select/configure the class --
            lim / bounds: (min, max) bounds, or None.
            vs / objects: value options, for param.Selector.
            dv / step: step size, or None.
            v0 / v / default: the default value.
            doc: documentation string.
            lab / label: display label.

    Returns:
        tuple[type[param.Parameter], dict[str, Any]]: the resolved class
        and the kwargs to instantiate it with.
    """
    lim = kwargs.get("lim", kwargs.get("bounds"))
    vs = kwargs.get("vs", kwargs.get("objects"))
    dv = kwargs.get("dv", kwargs.get("step"))
    v0 = kwargs.get("v")
    if v0 is None:
        v0 = kwargs.get("v0", kwargs.get("default"))
    doc = kwargs.get("doc")
    lab = kwargs.get("lab", kwargs.get("label"))

    if param_class is None:
        param_class = _select_param_class(dtype=dtype, lim=lim, vs=vs)

    param_kwargs: dict[str, Any] = {
        "default": v0,
        "doc": doc,
        "label": lab,
        "allow_None": True,
    }
    if lim is not None:
        param_kwargs.update(_lim_kwargs(param_class, lim))
    if dv is not None and _accepts_kwarg(param_class, "step", dv):
        param_kwargs["step"] = dv
    if vs is not None and _accepts_kwarg(param_class, "objects", vs):
        param_kwargs["objects"] = vs
    return param_class, param_kwargs
