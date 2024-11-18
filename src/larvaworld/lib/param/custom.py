import random

import numpy as np
import param

from .. import util

__all__ = [
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


class StringRobust(param.String):
    """Any input turned to string"""

    def __init__(self, default="", **kwargs):
        if default is not None and not isinstance(default, str):
            default = str(default)
        super().__init__(default=default, **kwargs)


class PositiveNumber(param.Number):
    """Number that must be positive"""

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
    """Integer that must be positive"""

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
    """Phase number within (0,2pi)"""

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
    """Range can be passed as list"""

    def __init__(self, default=(0.0, 0.0), step=0.1, **kwargs):
        if default is not None and not isinstance(default, tuple):
            default = tuple(default)
        super().__init__(default=default, step=step, **kwargs)

    def _validate_value(self, val, allow_None):
        if val is not None and not isinstance(val, tuple):
            val = tuple(val)
        super(RangeRobust, self)._validate_value(val, allow_None)


class RangeInf(RangeRobust):
    """Allow None inside tuple"""

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
    """Tuple range of positive numbers"""

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
    """Phase range within (0,2pi)"""

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
    """Number that must be positive"""

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
    """Integer that must be positive"""

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
    """Phase number within (0,2pi)"""

    def __init__(self, default=None, **kwargs):
        if default in [None, np.nan]:
            default = np.random.uniform(0, 2 * np.pi)
        super().__init__(default=default, allow_None=True, **kwargs)

    def _validate_value(self, val, allow_None):
        if val in [None, np.nan]:
            val = np.random.uniform(0, 2 * np.pi)
        super(RandomizedPhase, self)._validate_value(val, allow_None)


class RandomizedColor(param.Color):
    """Phase number within (0,2pi)"""

    def __init__(
        self,
        default=None,
        instantiate=True,
        allow_None=True,
        per_instance=True,
        **kwargs,
    ):
        if default in [None, np.nan, ""]:
            default = random.choice(super()._named_colors)
        super().__init__(
            default=default,
            instantiate=instantiate,
            allow_None=allow_None,
            per_instance=per_instance,
            **kwargs,
        )

    def _validate_value(self, val, allow_None):
        if val in [None, np.nan, ""]:
            val = random.choice(super()._named_colors)
        super(RandomizedColor, self)._validate_value(val, allow_None)


class OptionalPositiveRange(RangeInf):
    """Tuple range of positive numbers"""

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
    """Phase range within (0,2pi)"""

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
    """Select among objects. Default is None even if None not in objects"""

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
    """Tuple of integers"""

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
    """Tuple range of integers"""

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
    """Ordered range of integers"""

    def _validate_value(self, val, allow_None):
        super(IntegerRange, self)._validate_value(val, allow_None)
        v1, v2 = val
        assert v1 <= v2
        # raise ValueError("IntegerRange parameter %r only takes integer "
        #                      "values, not type %r." % (self.name, type(n)))


class PositiveIntegerRange(IntegerRange):
    """Tuple range of positive integers"""

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
    """Tuple range of positive integers"""

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
    """Tuple range of positive integers"""

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
    """XY point coordinates can be passed as list"""

    def __init__(self, default=(0.0, 0.0), **kwargs):
        if not isinstance(default, tuple):
            default = tuple(default)

        super().__init__(default=default, length=2, **kwargs)


class IntegerTuple2DRobust(IntegerTuple):
    """XY point coordinates can be passed as list"""

    def __init__(self, default=(0, 0), **kwargs):
        if not isinstance(default, tuple):
            default = tuple(default)
        super().__init__(default=default, length=2, **kwargs)


class ListXYcoordinates(param.List):
    """List of XY point coordinates"""

    def __init__(self, default=[], minlen=0, maxlen=None, **kwargs):
        super().__init__(
            default=default, item_type=tuple, bounds=(minlen, maxlen), **kwargs
        )


class XYLine(ListXYcoordinates):
    """List of XY point coordinates"""

    def __init__(self, minlen=0, **kwargs):
        super().__init__(minlen=minlen, **kwargs)


class ItemListParam(param.List):
    """
    Parameter whose value is a list of objects, usually of a specified type.

    Extends param.List to enable list management functionality provided by the
     lib.util.ItemList class, which inherits from a custom-made SuperList class
     as well as from the agentpy.AgentSequence
    """

    __slots__ = ["bounds", "item_type", "class_", "size"]

    def __init__(self, default=util.ItemList(), size=(0, None), **params):
        self.size = size
        if isinstance(default, list):
            default = util.ItemList(default)
        param.List.__init__(self, default=default, **params)
        self._validate(default)


class ClassDict(param.ClassSelector):
    """Dict of objects of specified class"""

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
    """An attribute og a given class"""

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
    """Select among objects. Default is None even if None not in objects"""

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
    __slots__ = ["rows", "columns", "ordered", "levels"]

    """A dataframe of specified index levels"""

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
    """A dataframe of specified index levels"""

    def __init__(self, **params):
        DataFrameIndexed.__init__(self, levels=["Step", "AgentID"], **params)


class EndpointDataFrame(DataFrameIndexed):
    """A dataframe of specified index levels"""

    def __init__(self, **params):
        DataFrameIndexed.__init__(self, levels=["AgentID"], **params)
