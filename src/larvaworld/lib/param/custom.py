from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple

import random

import numpy as np
import param

from .. import util

__all__: list[str] = [
    "StringRobust",
    "PositiveNumber",
    "PositiveInteger",
    "Phase",
    "RangeRobust",
    "RangeInf",
    "PositiveRange",
    "PhaseRange",
    "OptionalPositiveNumber",
    "OptionalPositiveInteger",
    "RandomizedPhase",
    "RandomizedColor",
    "OptionalPositiveRange",
    "OptionalPhaseRange",
    "OptionalSelector",
    "IntegerTuple",
    "IntegerRange",
    "IntegerRangeOrdered",
    "PositiveIntegerRange",
    "PositiveIntegerRangeOrdered",
    "NegativeIntegerRangeOrdered",
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
    automatically bounded to the valid phase range [0, 2π].

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


class ListXYcoordinates(param.List):
    """
    List parameter for XY coordinate tuples.

    Extends param.List with tuple item_type and length bounds,
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
            default=default, item_type=tuple, bounds=(minlen, maxlen), **kwargs
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


class ItemListParam(param.List):
    """
    Parameter for managed lists with ItemList functionality.

    Extends param.List to enable list management functionality provided by the
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
        param.List.__init__(self, default=default, **params)
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
