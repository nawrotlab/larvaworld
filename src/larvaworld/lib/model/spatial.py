import numpy as np
import param
from shapely import geometry

from larvaworld.lib import aux

class PointPositioned(aux.NestedConf):
    pos = param.XYCoordinates(doc='The xy spatial position coordinates')

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



class RadiallyExtended(PointPositioned):
    radius = aux.PositiveNumber(0.003, softmax=0.1, step=0.001, doc='The spatial radius of the source in meters')

    def get_shape(self, scale=1):
        p = self.get_position()
        return geometry.Point(p).buffer(self.radius * scale) if not np.isnan(p).all() else None

    def contained(self, point):
        return geometry.Point(self.get_position()).distance(geometry.Point(point)) <= self.radius

class OrientedPoint(RadiallyExtended):
    orientation = aux.Phase(label='orientation',doc='The absolute orientation in space.')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_orientation = self.orientation


class LineExtended(aux.NestedConf):
    width = aux.PositiveNumber(0.001, softmax=10.0, doc='The width of the Obstacle')
    vertices = aux.XYLine(doc='The list of 2d points')
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

class Area(aux.NestedConf):
    dims = aux.PositiveRange((0.1, 0.1), softmax=1.0, step=0.01, doc='The arena dimensions in meters')
    geometry = param.Selector(objects=['circular', 'rectangular'], doc='The arena shape')
    torus = param.Boolean(False, doc='Whether to allow a toroidal space')


class BoundedArea(Area, LineClosed):

    def __init__(self,vertices=None, **kwargs):
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

class Grid(aux.NestedConf):
    grid_dims = aux.PositiveIntegerRange((51, 51), softmax=500, doc='The spatial resolution of the food grid.')