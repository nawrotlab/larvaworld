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

