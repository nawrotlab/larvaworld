import numpy as np
import param

from shapely import geometry

from larvaworld.lib import aux

from larvaworld.lib.param import NestedConf, PositiveNumber, RandomizedPhase, XYLine, \
    PositiveIntegerRange, PositiveRange, NumericTuple2DRobust, IntegerTuple2DRobust, RangeRobust


class Pos2D(NestedConf):
    pos = NumericTuple2DRobust(doc='The xy spatial position coordinates')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_pos = self.pos
        self.last_pos = self.get_position()

    def get_position(self):
        return tuple(self.pos)

    def set_position(self, pos):
        if not isinstance(pos, tuple):
            pos=tuple(pos)
        self.last_pos = self.get_position()
        self.pos = pos

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    @property
    def last_delta_pos(self):
        x0,y0=self.last_pos
        x1,y1=self.get_position()
        return ((x1-x0)**2+(y1-y0)**2)**(1/2)

class Pos2DPixel(Pos2D):
    pos = IntegerTuple2DRobust(doc='The xy spatial position coordinates')



class RadiallyExtended(Pos2D):
    radius = PositiveNumber(0.003, softmax=0.1, step=0.001, doc='The spatial radius of the source in meters')

    def get_shape(self, scale=1):
        p = self.get_position()
        return geometry.Point(p).buffer(self.radius * scale) if not np.isnan(p).all() else None

    def contained(self, point):
        return geometry.Point(self.get_position()).distance(geometry.Point(point)) <= self.radius

class OrientedPoint(Pos2D):
    orientation = RandomizedPhase(label='orientation',doc='The absolute orientation in space.')

    def __init__(self,**kwargs):

        super().__init__(**kwargs)
        self.initial_orientation = self.orientation
        self.last_orientation=self.get_orientation()

    @property
    def rotationMatrix(self):
        a=-self.orientation
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def translate(self, point):
        p=np.array(self.pos) + np.array(point) @ self.rotationMatrix
        if isinstance(point, tuple):
            return tuple(p)
        else:
            return aux.np2Dtotuples(p)


    def set_orientation(self, orientation):
        self.last_orientation = self.get_orientation()
        self.orientation = orientation


    def get_orientation(self):
        return self.orientation

    def get_pose(self):
        return np.array(self.pos), self.orientation

    def update_poseNvertices(self, pos, orientation):
        self.set_position(pos)
        self.set_orientation(orientation% (np.pi * 2))

    @property
    def last_delta_orientation(self):
        a0=self.last_orientation
        a1=self.get_orientation()
        return a1-a0




class MobilePoint(OrientedPoint):

    def __init__(self,**kwargs):

        super().__init__(**kwargs)
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.ang_acc = 0.0
        self.cum_dst = 0.0
        self.dst = 0.0

    def get_angularvelocity(self):
        return self.ang_vel

    def get_linearvelocity(self):
        return self.lin_vel

    def set_linearvelocity(self, lin_vel):
        self.lin_vel = lin_vel

    def set_angularvelocity(self, ang_vel):
        self.ang_vel = ang_vel

    def update_all(self, pos, orientation, lin_vel, ang_vel):
        self.set_position(pos)
        self.set_orientation(orientation % (np.pi * 2))
        self.set_linearvelocity(lin_vel)
        self.set_angularvelocity(ang_vel)





class MobileVector(MobilePoint):
    length = PositiveNumber(1, doc='The initial length of the body in meters')

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    @property
    def front_end(self):
        return self.translate((self.length/2,0))

    @property
    def rear_end(self):
        return self.translate((-self.length / 2, 0))




class LineExtended(NestedConf):
    width = PositiveNumber(0.001, softmax=10.0, doc='The width of the line vertices')
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
    centered = param.Boolean(True, doc='Whether area is centered to (0,0)')

    @property
    def w(self):
        return self.dims[0]

    @property
    def h(self):
        return self.dims[1]

    @property
    def range(self):
        X, Y = self.dims
        return np.array([-X / 2, X / 2, -Y / 2, Y / 2])


class Area2DPixel(Area2D):
    dims = PositiveIntegerRange((100, 100), softmax=10000, step=1, doc='The arena dimensions in pixels')

    def get_rect_at_pos(self, pos=(0,0),**kwargs):
        import pygame
        if pos is not None and not any(np.isnan(pos)):
            if self.centered:
                return pygame.Rect(pos[0] - self.w / 2, pos[1] - self.h / 2, self.w, self.h)
            else:
                return pygame.Rect(pos[0], pos[1], self.w, self.h)
        else:
            return None

class Area(Area2D):
    dims = PositiveRange((0.1, 0.1), softmax=1.0, step=0.01, doc='The arena dimensions in meters')
    geometry = param.Selector(objects=['circular', 'rectangular'], doc='The arena shape')
    torus = param.Boolean(False, doc='Whether to allow a toroidal space')

class ScreenWindowAreaBasic(Area2DPixel):

    scaling_factor=PositiveNumber(1., doc='Scaling factor')
    space=param.ClassSelector(Area,default=Area(), doc='Arena')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dims = aux.get_window_dims(self.space.dims)

    def space2screen_pos(self, pos):
        if pos is None:
            return None
        if any(np.isnan(pos)):
            return (np.nan,np.nan)
        try:
            return self._transform(pos)
        except:
            X, Y = np.array(self.space.dims) * self.scaling_factor

            p = pos[0] * 2 / X, pos[1] * 2 / Y
            pp = ((p[0] + 1) * self.w / 2, (-p[1] + 1) * self.h)
            return pp

    def get_rect_at_pos(self, pos=(0,0), convert2screen_pos=True):
        if convert2screen_pos:
            pos=self.space2screen_pos(pos)
        return super().get_rect_at_pos(pos)

    def get_relative_pos(self, pos_scale):
        w, h = pos_scale
        x_pos = int(self.w * w)
        y_pos = int(self.h * h)
        return x_pos, y_pos

    def get_relative_font_size(self, font_size_scale):
        return int(self.w * font_size_scale)



class ScreenWindowAreaZoomable(ScreenWindowAreaBasic):
    zoom = PositiveNumber(1., doc='Zoom factor')
    center=param.Parameter(np.array([0., 0.]), doc='Center xy')
    center_lim=param.Parameter(np.array([0., 0.]), doc='Center xy lim')
    _scale = param.Parameter(np.array([[1., .0], [.0, -1.]]), doc='Scale of xy')
    _translation = param.Parameter(np.zeros(2), doc='Translation of xy')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_bounds()

    @ property
    def display_size(self):
        return (np.array(self.dims) / self.zoom).astype(int)

    @param.depends('zoom', 'center', watch=True)
    def set_bounds(self):
        left, right, bottom, top=self.space.range * self.scaling_factor
        assert right > left and top > bottom
        x = self.display_size[0] / (right - left)
        y = self.display_size[1] / (top - bottom)
        self._scale = np.array([[x, .0], [.0, -y]])
        self._translation = np.array([(-left * self.zoom) * x, (-bottom * self.zoom) * y]) + self.center * [-x, y]
        self.center_lim = (1 - self.zoom) * np.array([left, bottom])

    def _transform(self, position):
        return np.round(self._scale.dot(position) + self._translation).astype(int)

    def move_center(self, dx=0, dy=0, pos=None):
        if pos is None:
            pos = self.center - self.center_lim * [dx, dy]
        self.center = np.clip(pos, self.center_lim, -self.center_lim)

    def zoom_screen(self, d_zoom, pos=None):
        if pos is None:
            pos = self.mouse_position
        if 0.001 <= self.zoom + d_zoom <= 1:
            self.zoom = np.round(self.zoom + d_zoom, 2)
            self.center = np.clip(self.center - np.array(pos) * d_zoom, self.center_lim, -self.center_lim)
        if self.zoom == 1.0:
            self.center = np.array([0.0, 0.0])

    # @param.depends('zoom', watch=True)
    # def update_scale(self):
    #     def closest(lst, k):
    #         return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - k))]
    #
    #     # Get 1/10 of max real dimension, transform it to mm and find the closest reasonable scale
    #     self.scale_in_mm = closest(
    #         lst=[0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100, 250, 500, 750, 1000], k=self.space.w * self.zoom* 100)
    #     # self.text_font.set_text(f'{self.scale_in_mm} mm')
    #     # self.lines = self.compute_lines(self.x, self.y, self.scale_in_mm / self.zoom /1000)



class ScreenWindowArea(ScreenWindowAreaZoomable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



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
    reference_area = param.ClassSelector(Area2DPixel, doc='The reference position instance', is_instance=True)
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

