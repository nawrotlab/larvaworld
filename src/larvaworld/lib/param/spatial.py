import numpy as np
import param
from shapely import geometry

from larvaworld.lib import aux

from larvaworld.lib.param import NestedConf, PositiveNumber, RandomizedPhase, XYLine, \
    PositiveIntegerRange, PositiveRange, NumericTuple2DRobust, IntegerTuple2DRobust


class Pos2D(NestedConf):
    pos = NumericTuple2DRobust(doc='The xy spatial position coordinates')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_pos = self.pos

    def get_position(self):
        return tuple(self.pos)

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

class Pos2DPixel(Pos2D):
    pos = IntegerTuple2DRobust(doc='The xy spatial position coordinates')




class RadiallyExtended(Pos2D):
    radius = PositiveNumber(0.003, softmax=0.1, step=0.001, doc='The spatial radius of the source in meters')

    def get_shape(self, scale=1):
        p = self.get_position()
        return geometry.Point(p).buffer(self.radius * scale) if not np.isnan(p).all() else None

    def contained(self, point):
        return geometry.Point(self.get_position()).distance(geometry.Point(point)) <= self.radius

class OrientedPoint(RadiallyExtended):
    orientation = RandomizedPhase(label='orientation',doc='The absolute orientation in space.')

    def __init__(self,**kwargs):

        super().__init__(**kwargs)
        self.initial_orientation = self.orientation


class LineExtended(NestedConf):
    width = PositiveNumber(0.001, softmax=10.0, doc='The width of the Obstacle')
    vertices = XYLine(doc='The list of 2d points')
    closed = param.Boolean(False, doc='Whether the line is closed')

    @property
    def _edges(self):
        vs=self.vertices
        N=len(vs)
        edges=[[vs[i], vs[i+1]] for i in range(N-1)]
        if self.closed :
            edges.append([vs[N], vs[0]])
        return edges


class LineClosed(LineExtended):
    def __init__(self, **kwargs):
        super().__init__(closed=True,**kwargs)

class Area2D(NestedConf):
    dims = PositiveRange(doc='The arena dimensions')
    centered = param.Boolean(False, doc='Whether entity is active')

    @property
    def w(self):
        return self.dims[0]

    @property
    def h(self):
        return self.dims[1]

    def set_shape(self, pos=None):
        import pygame
        if pos is None and hasattr(self, 'pos'):
            pos=self.pos
        if pos is not None and not any(np.isnan(pos)):
            if self.centered:
                return pygame.Rect(pos[0] - self.w / 2, pos[1] - self.h / 2, self.w, self.h)
            else:
                return pygame.Rect(pos[0], pos[1], self.w, self.h)
        else:
            return None


class Area2DPixel(Area2D):
    dims = PositiveIntegerRange((100, 100), softmax=10000, step=1, doc='The arena dimensions in pixels')


class Area(Area2D):
    dims = PositiveRange((0.1, 0.1), softmax=1.0, step=0.01, doc='The arena dimensions in meters')
    geometry = param.Selector(objects=['circular', 'rectangular'], doc='The arena shape')
    torus = param.Boolean(False, doc='Whether to allow a toroidal space')


class BoundedArea(Area, LineClosed):

    def __init__(self,vertices=None,**kwargs):

        Area.__init__(self, **kwargs)
        X, Y = self.dims
        if vertices is None:
            if self.geometry == 'circular':
                # This is a circle_to_polygon shape from the function
                vertices = aux.circle_to_polygon(60, X / 2)
            elif self.geometry == 'rectangular':
                # This is a rectangular shape
                vertices = [(-X / 2, -Y / 2),
                            (-X / 2, Y / 2),
                            (X / 2, Y / 2),
                            (X / 2, -Y / 2)]
        LineClosed.__init__(self,vertices=vertices,**kwargs)

class Grid(NestedConf):
    grid_dims = PositiveIntegerRange((51, 51), softmax=500, doc='The spatial resolution of the food grid.')


class PosPixelRel2Point(Pos2DPixel):
    reference_point = param.ClassSelector(Pos2DPixel, doc='The reference position instance', is_instance=False)
    pos_scale = PositiveRange((0.5, 0.5), softmax=1.0, step=0.01,
                              doc='The position relative to reference position')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_pos()

    @param.depends('pos_scale', 'reference_point', watch=True)
    def update_pos(self):
        w, h = self.pos_scale
        x_pos = int(self.reference_point.x * w)
        y_pos = int(self.reference_point.y * h)
        self.pos = (x_pos, y_pos)


class PosPixelRel2Area(Pos2DPixel):
    reference_area = param.ClassSelector(Area2DPixel, doc='The reference position instance', is_instance=False)
    pos_scale = PositiveRange((0.5, 0.5), softmax=1.0, step=0.01,
                              doc='The position relative to reference position')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.update_pos()

    @param.depends('pos_scale', 'reference_area', watch=True)
    def update_pos(self):
        w, h = self.pos_scale
        x_pos = int(self.reference_area.w * w)
        y_pos = int(self.reference_area.h * h)
        self.pos = (x_pos, y_pos)



class PositionedArea2DPixel(Pos2DPixel, Area2DPixel): pass


class PositionedArea2DPixelRel2Area(PosPixelRel2Area, Area2DPixel): pass

