import numpy as np
from scipy.stats import multivariate_normal
from shapely.geometry import Point

import lib.aux.functions as fun
import lib.aux.rendering as ren


class LarvaworldAgent:
    def __init__(self,unique_id: str,model, pos=None, default_color=None, radius=None,visible=True,
                 # odor_id=None, odor_intensity=0.0, odor_spread=0.1,
                 odor={'odor_id':None, 'odor_intensity':0.0, 'odor_spread':0.1},
                 group='', can_be_carried=False):
        self.visible = visible
        self.selected = False
        self.unique_id = unique_id
        self.model = model
        self.group = group
        self.base_odor_id = f'{group}_base_odor'
        self.gain_for_base_odor = 100

        self.initial_pos = pos
        self.pos = self.initial_pos
        if type(default_color) == str:
            default_color = fun.colorname2tuple(default_color)
        self.default_color = default_color
        self.color = self.default_color
        self.radius = radius
        self.id_box = self.init_id_box()
        self.odor_id = odor['odor_id']
        self.odor_intensity = odor['odor_intensity']
        self.odor_spread = odor['odor_spread']
        if self.odor_spread is None:
            self.odor_spread = 0.1
        self.set_odor_dist()

        self.carried_objects = []
        self.can_be_carried = can_be_carried
        self.is_carried_by = None

    def get_position(self):
        return tuple(self.pos)

    def get_radius(self):
        return self.radius

    def init_id_box(self):
        id_box = ren.InputBox(visible=False, text=self.unique_id,
                              color_inactive=self.default_color, color_active=self.default_color,
                              screen_pos=None, agent=self)
        return id_box

    def set_id(self, id):
        self.unique_id = id
        self.id_box.text = self.unique_id

    def get_shape(self, scale=1):
        p = self.get_position()
        return Point(p).buffer(self.radius*scale) if not np.isnan(p).all() else None

    def set_color(self, color):
        self.color = color

    def contained(self, point):
        # return Point(self.get_position()).distance(Point(point))<=self.radius
        # return Circle(self.get_position(), radius=self.radius).contains_point(point)
        shape = self.get_shape()
        return shape.covers(Point(point)) if shape else False

    # @abc.abstractmethod
    def step(self):
        pass

    def set_default_color(self, color):
        self.default_color = color
        self.id_box.color = self.default_color
        self.set_color(color)

    def set_odor_dist(self, intensity=None, spread=None):
        if intensity is not None :
            self.odor_intensity=intensity
        if spread is not None :
            self.odor_spread=spread
        self.odor_dist = multivariate_normal([0, 0], [[self.odor_spread, 0], [0, self.odor_spread]])
        self.odor_peak_value = self.odor_intensity / self.odor_dist.pdf([0, 0])

    def get_gaussian_odor_value(self, pos):
        return self.odor_dist.pdf(pos) * self.odor_peak_value

    def draw(self, viewer, filled=True):
        if self.get_shape() is None :
            return
        p, c, r = self.get_position(), self.color, self.radius
        viewer.draw_polygon(self.get_shape().boundary.coords, c, filled, r/5)
        # viewer.draw_circle(p, r, c, filled, r / 5)

        if self.odor_intensity > 0:
            viewer.draw_polygon(self.get_shape(1.5).boundary.coords, c, False, r / 10)
            viewer.draw_polygon(self.get_shape(2.0).boundary.coords, c, False, r / 15)
            viewer.draw_polygon(self.get_shape(3.0).boundary.coords, c, False, r / 20)
            # viewer.draw_circle(p, r * 1.5, c, False, r / 10)
            # viewer.draw_circle(p, r * 2.0, c, False, r / 15)
            # viewer.draw_circle(p, r * 3.0, c, False, r / 20)
        if self.selected:
            viewer.draw_polygon(self.get_shape(1.1).boundary.coords, self.model.selection_color, False, r / 5)
            # viewer.draw_circle(p, r * 1.2, self.model.selection_color, False, r / 5)


