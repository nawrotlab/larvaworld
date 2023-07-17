import copy
import random
from typing import Tuple

import numpy as np
import param
from param import Parameterized, Number,NumericTuple,Integer,Selector,String, ListSelector, Range, Magnitude, Boolean,ClassSelector,Parameter, List, Dict
from scipy.stats import multivariate_normal

from larvaworld.lib import aux




class PositiveNumber(Number):
    """Number that must be positive"""
    def __init__(self,default=0.0, softmin=0.0, softmax=None, hardmin=0.0, hardmax=None, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)

class PositiveInteger(Integer):
    """Integer that must be positive"""
    def __init__(self,default=0, softmin=0, softmax=None, hardmin=0, hardmax=None, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)

class Phase(Number):
    """Phase number within (0,2pi)"""
    def __init__(self,default=0.0, softmin=0.0, softmax=2 * np.pi, hardmin=0.0, hardmax=2 * np.pi, **kwargs):

        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)

class RangeInf(Range):
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


class PositiveRange(Range):
    """Tuple range of positive numbers"""
    def __init__(self,default=(0.0, 0.0), softmin=0.0, softmax=None, hardmin=0.0, hardmax=None, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)

class PhaseRange(Range):
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

class OptionalPhase(Number):
    """Phase number within (0,2pi)"""
    def __init__(self,default=None, softmin=0.0, softmax=2 * np.pi, hardmin=0.0, hardmax=2 * np.pi, **kwargs):
        # print(default)
        # if default==np.nan :
        #     default = None
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),allow_None=True,**kwargs)

class OptionalPositiveRange(RangeInf):
    """Tuple range of positive numbers"""
    def __init__(self,default=None, softmin=0.0, softmax=None, hardmin=0.0, hardmax=None, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),allow_None=True,**kwargs)

class OptionalPhaseRange(Range):
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

class IntegerRange(Range):
    """Tuple range of integers"""

    def _validate_value(self, val, allow_None):
        super(Range, self)._validate_value(val, allow_None)
        for n in val:
            if isinstance(n, int):
                continue
            raise ValueError("IntegerRange parameter %r only takes integer "
                             "values, not type %r." % (self.name, type(n)))


class PositiveIntegerRange(IntegerRange):
    """Tuple range of positive integers"""
    def __init__(self,default=(0, 0), softmin=0, softmax=None, hardmin=0, hardmax=None, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)


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

class ClassAttr(param.ClassSelector):
    """An attribute og a given class"""
    def __init__(self, class_,**kwargs):
        if 'default' not in kwargs.keys() :
            kwargs['default'] = class_()
        super().__init__(class_=class_, **kwargs)







class NestedConf(param.Parameterized):

    def __init__(self,**kwargs):
        # for k in kwargs.keys():
        #     p=self.class_type(k)


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
            if type(p) == ClassAttr:
                d[k] = d[k].nestedConf
            elif type(p) == ClassDict:
                d[k] = aux.AttrDict({kk: vv.nestedConf for kk, vv in d[k].items()})
        return d

    #@ property
    def entry(self, id):
        d=self.nestedConf
        if 'distribution' in d.keys():
            assert 'group' in d.keys()
            d.group=id
        else:
            assert 'unique_id' in d.keys()
            d.unique_id = id
            d.pop('unique_id')
        return {id:d}

# class SimTimeConf(NestedConf):
#     dt = PositiveNumber(0.1, softmax=1.0, step=0.01, doc='The timestep of the simulation in seconds.')
#     duration = OptionalPositiveNumber(5.0, softmax=1000.0, step=0.1, doc='The duration of the simulation in minutes.')
#     Nsteps = OptionalPositiveInteger(label='# simulation timesteps',doc='The number of simulation timesteps.')
#
#     def __init__(self,**kwargs):
#         super().__init__(**kwargs)
#         # Define N timesteps
#         if self.Nsteps is None and self.duration is not None:
#             self.Nsteps = int(self.duration * 60 / self.dt)
#         if self.duration is None and self.Nsteps is not None:
#             self.duration = self.Nsteps * self.dt / 60
#
# class SimConf(SimTimeConf):
#     Box2D = Boolean(False,doc='Whether to use the Box2D physics engine or not.')
#     store_data = Boolean(True, doc='Whether to store the simulation data')
#     larva_collisions = Boolean(True, doc='Whether to allow overlap between larva bodies.')
#     offline = Boolean(False,doc='Whether to launch a full Larvaworld environment')
#     show_display = Boolean(True,doc='Whether to launch the pygame-visualization.')
#
#     def __init__(self,offline=False, show_display=True, **kwargs):
#         if offline:
#             show_display=False
#         super().__init__(show_display=show_display,offline=offline,**kwargs)
#         # Define constant parameters
#         self.scaling_factor = 1000.0 if self.Box2D else 1.0



class PreprocessConf(NestedConf):
    rescale_by = OptionalPositiveNumber(softmax=1000.0, step=0.001, doc='Whether to rescale spatial coordinates by a scalar in meters.')
    filter_f = OptionalPositiveNumber(softmax=5.0, step=0.01, doc='Whether to filter spatial coordinates by a grade-1 low-pass filter of the given cut-off frequency.')
    transposition = OptionalSelector(['origin', 'arena', 'center'],doc='Whether to transpose spatial coordinates.')
    interpolate_nans = Boolean(False, doc='Whether to interpolate missing values.')
    drop_collisions = Boolean(False,doc='Whether to drop timepoints where larva collisions are detected.')


class TrackerFormat(NestedConf):
    dt = PositiveNumber(0.1, softmax=1.0, step=0.01, label='tracker timestep',doc='The tracking timestep (inverse of tracking framerate) in seconds.')
    constant_framerate = Boolean(True, doc='Whether the tracking framerate is constant.')
    XY_unit = Selector(default='m', objects=['m', 'mm'], doc='The spatial unit of the XY coordinate data')
    Npoints = PositiveInteger(1, softmax=20, label='# midline 2D points',doc='The number of points tracked along the larva midline.')
    Ncontour = PositiveInteger(1, softmax=100, label='# contour 2D points',doc='The number of points tracked around the larva contour.')


class Metric_Definition(NestedConf):
    bend = Selector(objects=['from_vectors', 'from_angles'],doc='Whether bending angle is computed as a sum of sequential segmental angles or as the angle between front and rear body vectors.')
    front_vector = IntegerRange((1, 2), softbounds=(-12, 12),doc='The initial & final segment of the front body vector.')
    rear_vector = IntegerRange((-2, -1), softbounds=(-12, 12),doc='The initial & final segment of the rear body vector.')
    front_body_ratio = Magnitude(0.5, doc='The fraction of the body considered front, relevant for bend computation from angles.')
    point_idx = OptionalPositiveInteger(softmax=20, doc='Index of midline point to use as the larva spatial position. Default is None meaning use the centroid.')
    use_component_vel = Boolean(False, doc='Whether to use the component velocity ralative to the axis of forward motion.')


class EnrichConf(NestedConf):
    metric_definition = ClassAttr(Metric_Definition,doc='The metric_definition')
    pre_kws = ClassAttr(PreprocessConf,doc='The preprocessing pipelines')
    proc_keys = ListSelector(default=['angular', 'spatial', 'dispersion', 'tortuosity'],objects=['angular', 'spatial', 'source', 'dispersion', 'tortuosity', 'PI', 'wind'], doc='The processing pipelines')
    anot_keys = ListSelector(default=['bout_detection', 'bout_distribution', 'interference'], objects=['bout_detection', 'bout_distribution', 'interference', 'source_attraction', 'patch_residency'], doc='The annotation pipelines')
    recompute = Boolean(False,doc='Whether to recompute')
    mode = Selector(objects=['minimal', 'full'],doc='The processing mode')