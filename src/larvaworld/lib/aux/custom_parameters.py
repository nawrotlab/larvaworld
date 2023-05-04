import copy

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
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)

class OptionalPositiveInteger(Integer):
    """Integer that must be positive"""
    def __init__(self,default=None, softmin=0, softmax=None, hardmin=0, hardmax=None, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)

class OptionalPhase(Number):
    """Phase number within (0,2pi)"""
    def __init__(self,default=None, softmin=0.0, softmax=2 * np.pi, hardmin=0.0, hardmax=2 * np.pi, **kwargs):
        super().__init__(default=default,softbounds=(softmin, softmax),bounds=(hardmin, hardmax),**kwargs)

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
    def __init__(self, objects,  **kwargs):
        super().__init__(default=None,empty_default=True,allow_None=True,objects=objects, **kwargs)

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


class ClassDict(ClassSelector):
    """Dict of objects of specified class"""

    __slots__ = ['class_', 'is_instance', 'item_type']

    def __init__(self, default=aux.AttrDict(),item_type=None,  **params):
        self.item_type = item_type
        # print(default, self.item_type)
        # for k, p in default.items():
        #     if not isinstance(p, self.item_type):
        #         try :
        #             default[k] = item_type(**p)
        #         except :
        #             pass
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
            if  k in kwargs.keys():
                if type(p) == ClassAttr and not isinstance(kwargs[k], p.class_):
                    kwargs[k] = p.class_(**kwargs[k])
                elif type(p) == ClassDict and not all(isinstance(vv, p.item_type) for kk,vv in kwargs[k].items()):
                    kwargs[k] = p.class_({kk: p.item_type(**vv) for kk, vv in kwargs[k].items()})
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
    #
    # @classmethod
    # def attr_classes(cls):
    #     d=copy.deepcopy(cls.param.objects())
    #     d.pop('name')
    #     return aux.AttrDict(d)
    #
    # @classmethod
    # def attr_ks(cls):
    #     return list(cls.attr_classes().keys())
    #
    # @classmethod
    # def attr_class_dict(cls):
    #     d=cls.attr_classes().copy()
    #     for k,v in d.items() :
    #         if isinstance(v, ClassAttr):
    #             d[k]=v.class_.attr_classes()
    #     return aux.AttrDict(d)
    #
    #
    # def class_type(self,k):
    #     p = self.param.objects()[k]
    #     return type(p)




class Odor(NestedConf):
    id = param.String(None, doc='The unique ID of the odorant')
    intensity = OptionalPositiveNumber(softmax=10.0, doc='The peak concentration of the odorant in micromoles')
    spread = OptionalPositiveNumber(softmax=10.0, doc='The spread of the concentration gradient around the peak')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._update_distro()

    @param.depends('intensity','spread', watch=True)
    def _update_distro(self):
        if self.intensity is not None and self.spread is not None:
            self.dist = multivariate_normal([0, 0], [[self.spread, 0], [0, self.spread]])
            self.peak_value = self.intensity / self.dist.pdf([0, 0])
        else:
            self.dist = None
            self.peak_value = 0.0

    def gaussian_value(self, pos):
        if self.dist :
            return self.dist.pdf(pos) * self.peak_value
        else :
            return None

class SimTimeConf(NestedConf):
    dt = PositiveNumber(0.1, softmax=1.0, step=0.01, doc='The timestep of the simulation in seconds.')
    duration = OptionalPositiveNumber(5.0, softmax=1000.0, step=0.1, doc='The duration of the simulation in minutes.')
    Nsteps = OptionalPositiveInteger(label='# simulation timesteps',doc='The number of simulation timesteps.')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # Define N timesteps
        if self.Nsteps is None and self.duration is not None:
            self.Nsteps = int(self.duration * 60 / self.dt)
        if self.duration is None and self.Nsteps is not None:
            self.duration = self.Nsteps * self.dt / 60

class SimConf(SimTimeConf):
    Box2D = Boolean(False,doc='Whether to use the Box2D physics engine or not.')
    store_data = Boolean(True, doc='Whether to store the simulation data')
    larva_collisions = Boolean(True, doc='Whether to allow overlap between larva bodies.')
    offline = Boolean(False,doc='Whether to launch a full Larvaworld environment')
    show_display = Boolean(True,doc='Whether to launch the pygame-visualization.')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # Define constant parameters
        self.scaling_factor = 1000.0 if self.Box2D else 1.0
        self.show_display = self.show_display and not self.offline

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