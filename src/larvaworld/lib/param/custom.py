import random

import numpy as np
import param
from param import Parameterized, Number, NumericTuple, Integer, Selector, Range, Magnitude, Boolean, ClassSelector, \
    Parameter, List, Dict, String, DataFrame

from larvaworld.lib import aux

class StringRobust(String):
    """Any input turned to string"""

    def __init__(self, default='', **kwargs):
        if default is not None and not isinstance(default,str) :
            default=str(default)
        super().__init__(default=default, **kwargs)


class PositiveNumber(Number):
    """Number that must be positive"""
    def __init__(self,default=0.0, softmin=0.0, softmax=None, hardmin=0.0, hardmax=None,bounds=None, **kwargs):
        if bounds is None:
            bounds = (hardmin, hardmax)
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=bounds,**kwargs)

class PositiveInteger(Integer):
    """Integer that must be positive"""
    def __init__(self,default=0, softmin=0, softmax=None, hardmin=0, hardmax=None, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)

class Phase(Number):
    """Phase number within (0,2pi)"""
    def __init__(self,default=0.0, softmin=0.0, softmax=2 * np.pi, hardmin=0.0, hardmax=2 * np.pi, **kwargs):

        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)

class RangeRobust(Range):
    """Range can be passed as list"""

    def __init__(self, default=(0.0, 0.0), **kwargs):
        if default is not None and not isinstance(default,tuple) :
            default=tuple(default)
        super().__init__(default=default, **kwargs)

    def _validate_value(self, val, allow_None):
        if val is not None and not isinstance(val,tuple) :
            val=tuple(val)
        super(RangeRobust, self)._validate_value(val, allow_None)


class RangeInf(RangeRobust):
    """Allow None inside tuple"""

    def _validate_value(self, val, allow_None):
        super(NumericTuple, self)._validate_value(val, allow_None)
        if allow_None and val is None:
            return
        for n in val:
            if param._is_number(n) or allow_None and n is None:
                continue
            raise ValueError("NumericTuple parameter %r only takes numeric "
                             "values, not type %r." % (self.name, type(n)))

    def _validate_bounds(self, val, bounds, inclusive_bounds):
        if bounds is None or (val is None and self.allow_None):
            return
        vmin, vmax = bounds
        incmin, incmax = inclusive_bounds
        for bound, v in zip(['lower', 'upper'], val):
            if v is None and self.allow_None :
                continue
            too_low = (vmin is not None) and (v < vmin if incmin else v <= vmin)
            too_high = (vmax is not None) and (v > vmax if incmax else v >= vmax)
            if too_low or too_high:
                raise ValueError("Range parameter %r's %s bound must be in range %s."
                                 % (self.name, bound, self.rangestr()))


class PositiveRange(RangeRobust):
    """Tuple range of positive numbers"""
    def __init__(self,default=(0.0, 0.0), softmin=0.0, softmax=None, hardmin=0.0, hardmax=None, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)

class PhaseRange(RangeRobust):
    """Phase range within (0,2pi)"""
    def __init__(self,default=(0.0, 0.0), softmin=0.0, softmax=2 * np.pi, hardmin=0.0, hardmax=2 * np.pi, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)


class OptionalPositiveNumber(Number):
    """Number that must be positive"""
    def __init__(self,default=None, softmin=0.0, softmax=None, hardmin=0.0, hardmax=None, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),allow_None=True,**kwargs)

class OptionalPositiveInteger(Integer):
    """Integer that must be positive"""
    def __init__(self,default=None, softmin=0, softmax=None, hardmin=0, hardmax=None, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),allow_None=True,**kwargs)

class RandomizedPhase(Phase):
    """Phase number within (0,2pi)"""

    def __init__(self, default=None, **kwargs):
        if default in [None, np.nan]:
            default = np.random.uniform(0, 2 * np.pi)
        super().__init__(default=default, allow_None=True,**kwargs)


    def _validate_value(self, val, allow_None):
        if val in [None, np.nan]:
            val = np.random.uniform(0, 2 * np.pi)
        super(RandomizedPhase, self)._validate_value(val, allow_None)

class RandomizedColor(param.Color):
    """Phase number within (0,2pi)"""

    def __init__(self, default=None, **kwargs):
        if default in [None, np.nan, '']:
            default = random.choice(super()._named_colors)
        if isinstance(default,tuple):
            default=aux.colortuple2str(default)
        super().__init__(default=default, **kwargs)


    def _validate_value(self, val, allow_None):
        if val in [None, np.nan, '']:
            val = random.choice(super()._named_colors)
        if isinstance(val,tuple):
            val=aux.colortuple2str(val)
        super(RandomizedColor, self)._validate_value(val, allow_None)

class OptionalPositiveRange(RangeInf):
    """Tuple range of positive numbers"""
    def __init__(self,default=None, softmin=0.0, softmax=None, hardmin=0.0, hardmax=None, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),allow_None=True,**kwargs)

class OptionalPhaseRange(RangeRobust):
    """Phase range within (0,2pi)"""
    def __init__(self,default=None, softmin=0.0, softmax=2 * np.pi, hardmin=0.0, hardmax=2 * np.pi, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)

class OptionalSelector(Selector):
    """Select among objects. Default is None even if None not in objects"""
    def __init__(self, objects,default=None,  **kwargs):
        kws = {
            'default' : default,
            'objects': objects,
            # 'doc': f'The {conftype0.default} configuration ID',
            **kwargs
        }
        if default is None :
            kws['empty_default']=True
            kws['allow_None']=True
        super().__init__(**kws)

class IntegerTuple(NumericTuple):
    """Tuple of integers"""

    def _validate_value(self, val, allow_None):
        super(NumericTuple, self)._validate_value(val, allow_None)
        for n in val:
            if isinstance(n, int):
                continue
            raise ValueError("IntegerTuple parameter %r only takes integer "
                             "values, not type %r." % (self.name, type(n)))

class IntegerRange(RangeRobust):
    """Tuple range of integers"""
    def __init__(self, default=(0, 0), **kwargs):
        super().__init__(default=default, **kwargs)

    def _validate_value(self, val, allow_None):
        super(RangeRobust, self)._validate_value(val, allow_None)
        for n in val:
            if isinstance(n, int):
                continue
            raise ValueError("IntegerRange parameter %r only takes integer "
                             "values, not type %r." % (self.name, type(n)))

class IntegerRangeOrdered(IntegerRange):
    """Ordered range of integers"""

    def _validate_value(self, val, allow_None):
        super(IntegerRange, self)._validate_value(val, allow_None)
        v1,v2=val
        assert (v1<=v2)
        # raise ValueError("IntegerRange parameter %r only takes integer "
        #                      "values, not type %r." % (self.name, type(n)))



class PositiveIntegerRange(IntegerRange):
    """Tuple range of positive integers"""
    def __init__(self,default=(0, 0), softmin=0, softmax=None, hardmin=0, hardmax=None, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)



class PositiveIntegerRangeOrdered(IntegerRangeOrdered):
    """Tuple range of positive integers"""
    def __init__(self,default=(0, 1), softmin=0, softmax=None, hardmin=0, hardmax=None, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)

class NegativeIntegerRangeOrdered(IntegerRangeOrdered):
    """Tuple range of positive integers"""
    def __init__(self,default=(-1, 0), softmin=None, softmax=0, hardmin=None, hardmax=0, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)



class NumericTuple2DRobust(NumericTuple):
    """XY point coordinates can be passed as list"""

    def __init__(self, default=(0.0, 0.0), **kwargs):
        if not isinstance(default,tuple) :
            default=tuple(default)
        super().__init__(default=default, length=2, **kwargs)

class IntegerTuple2DRobust(IntegerTuple):
    """XY point coordinates can be passed as list"""

    def __init__(self, default=(0, 0), **kwargs):
        if not isinstance(default,tuple) :
            default=tuple(default)
        super().__init__(default=default, length=2, **kwargs)



class ListXYcoordinates(List):
    """List of XY point coordinates"""
    def __init__(self, default=[],minlen=0, maxlen=None, **kwargs):
        super().__init__(default=default, item_type=tuple,bounds=(minlen,maxlen), **kwargs)

class XYLine(ListXYcoordinates):
    """List of XY point coordinates"""
    def __init__(self, minlen=0, **kwargs):
        super().__init__(minlen=minlen,**kwargs)


class ClassDict(ClassSelector):
    """Dict of objects of specified class"""

    __slots__ = ['class_', 'is_instance', 'item_type']

    def __init__(self, default=aux.AttrDict(),item_type=None,  **params):
        self.item_type = item_type
        ClassSelector.__init__(self,aux.AttrDict, default=default, **params)


    def _validate(self, val):
        super(ClassSelector, self)._validate(val)
        self._validate_item_type(val, self.item_type)

    def _validate_item_type(self, val, item_type):
        if item_type is None or (self.allow_None and val is None):
            return
        for k, v in val.items():
            if isinstance(v, item_type):
                continue
            raise TypeError("ClassDict parameter %r items must be instances "
                            "of type %r, not %r." % (self.name, item_type, val))

class ClassAttr(ClassSelector):
    """An attribute og a given class"""
    def __init__(self, class_,**kwargs):
        if not isinstance(class_, tuple):
            cc=class_
        else:
            cc = class_[0]
        if 'default' not in kwargs.keys() :
            kwargs['default'] = cc()
        elif kwargs['default'] is None :
            kwargs['default'] = None
        elif not isinstance(kwargs['default'], class_):
            kwargs['default'] = cc(**kwargs['default'])
        super().__init__(class_=class_, **kwargs)

class DataFrameIndexed(DataFrame):
    __slots__ = ['rows', 'columns', 'ordered', 'levels']

    """A dataframe of specified index levels"""
    def __init__(self, levels=None,  **params):
        self.levels = levels
        DataFrame.__init__(self,**params)

    def _validate(self, val):
        super(DataFrame, self)._validate(val)
        self._validate_levels(val, self.levels)

    def _validate_levels(self, val, levels):
        if levels is None or (self.allow_None and val is None):
            return
        val_levels=list(val.index.names)
        if val_levels != levels:
            raise TypeError("DataFrameIndexed parameter %r levels must be "
                            " %r, not %r." % (self.name, levels, val_levels))

class StepDataFrame(DataFrameIndexed):
    """A dataframe of specified index levels"""

    def __init__(self, **params):
        DataFrameIndexed.__init__(self, levels = ['Step', 'AgentID'], **params)

class EndpointDataFrame(DataFrameIndexed):
    """A dataframe of specified index levels"""

    def __init__(self, **params):
        DataFrameIndexed.__init__(self, levels = ['AgentID'], **params)



class NestedConf(Parameterized):

    def __init__(self,**kwargs):
        # for k in kwargs.keys():
        #     p=self.class_type(k)
        # print(self.name)

        param_classes = self.param.objects()
        for k, p in param_classes.items():
            try:
                if  k in kwargs.keys():
                    if type(p) == ClassAttr and not isinstance(kwargs[k], p.class_):
                        kwargs[k] = p.class_(**kwargs[k])
                    elif type(p) == ClassDict and not all(isinstance(vv, p.item_type) for kk,vv in kwargs[k].items()):
                        kwargs[k] = p.class_({kk: p.item_type(**vv) for kk, vv in kwargs[k].items()})
            except :
                pass
        super().__init__(**kwargs)




    @ property
    def nestedConf(self):
        d = aux.AttrDict(self.param.values())
        d.pop('name')
        for k, p in self.param.objects().items():
            if k in d and d[k] is not None :

                if type(p) == ClassAttr:
                    d[k] = d[k].nestedConf
                elif type(p) == ClassDict:
                    d[k] = aux.AttrDict({kk: vv.nestedConf for kk, vv in d[k].items()})
        return d

    #@ property
    def entry(self, id=None):
        d=self.nestedConf
        if 'distribution' in d.keys():
            if 'group' in d.keys():
                if id is not None:
                    d.group=id
                elif d.group is not None:
                    id=d.group
            if 'model' in d.keys():
                if id is None:
                    id=d.model
        elif 'unique_id' in d.keys():
            if id is None and d.unique_id is not None:
                id=d.unique_id
                d.pop('unique_id')
        assert id is not None
        return {id:d}

    @property
    def param_keys(self):
        ks= list(self.param.objects().keys())
        return aux.SuperList([k for k in ks if k not in ['name']])


    def params_missing(self, d):
        ks=self.param_keys
        return aux.SuperList([k for k in ks if k not in d.keys()])