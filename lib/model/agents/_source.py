import numpy as np
from Box2D import Box2D, b2ChainShape
from scipy.spatial.distance import euclidean
from shapely import affinity
from shapely.geometry import Point, Polygon

import lib.aux.sim_aux
from lib.aux.xy_aux import eudis5, xy_uniform_circle
from lib.model.DEB.substrate import Substrate
from lib.model.agents._agent import LarvaworldAgent


class Source(LarvaworldAgent):
    def __init__(self, shape_vertices=None, shape='circle', **kwargs):
        super().__init__(**kwargs)
        self.shape_vertices = shape_vertices
        shape = lib.aux.sim_aux.circle_to_polygon(60, self.radius)

        if self.model.Box2D:
            self._body: Box2D.b2Body = self.model.space.CreateStaticBody(position=self.pos)
            self.Box2D_shape = b2ChainShape(vertices=shape.tolist())
            self._body.CreateFixture(shape=self.Box2D_shape)
            self._body.fixtures[0].filterData.groupIndex = -1
        else:
            self.model.space.place_agent(self, self.pos)
        # # put all agents into same group (negative so that no collisions are detected)
        # self._fixtures[0].filterData.groupIndex = -1

    def get_vertices(self):
        v0 = self.shape_vertices
        x0, y0 = self.get_position()
        if v0 is not None and not np.isnan((x0, y0)).all():
            return [(x + x0, y + y0) for x, y in v0]
        else:
            return None

    def get_shape(self, scale=1):
        p = self.get_position()
        if np.isnan(p).all():
            return None
        elif self.get_vertices() is None:
            return Point(p).buffer(self.radius * scale)
        else:
            p0 = Polygon(self.get_vertices())
            p = affinity.scale(p0, xfact=scale, yfact=scale)
            return p

    def step(self):
        if self.can_be_displaced:
            w = self.model.windscape
            dt = self.model.dt
            r = self.radius * 10000
            if w is not None:
                ws, wo = w.wind_speed, w.wind_direction
                if ws != 0.0:
                    self.pos = (self.pos[0] + np.cos(wo) * ws * dt / r, self.pos[1] + np.sin(wo) * ws * dt / r)
                    from lib.aux.sim_aux import inside_polygon
                    in_tank = inside_polygon(points=[self.pos], tank_polygon=self.model.tank_polygon)
                    if not in_tank:
                        if self.regeneration:
                            self.pos = xy_uniform_circle(1, **self.regeneration_pos)[0]
                        else :
                            self.model.delete_agent(self)


class Food(Source):
    def __init__(self, amount=1.0, quality=1.0, default_color=None, type='standard', **kwargs):
        if default_color is None:
            default_color = 'green'
        super().__init__(default_color=default_color, **kwargs)
        self.initial_amount = amount
        self.amount = self.initial_amount
        self.substrate = Substrate(type=type, quality=quality)

    def get_amount(self):
        return self.amount

    def subtract_amount(self, amount):
        prev_amount = self.amount
        self.amount -= amount
        if self.amount <= 0.0:
            self.amount = 0.0
            self.model.delete_agent(self)
        else:
            r = self.amount / self.initial_amount
            self.color = (1 - r) * np.array((255, 255, 255)) + r * np.array(self.default_color)
        return np.min([amount, prev_amount])

    def draw(self, viewer, filled=True):
        # if not self.visible :
        #     return
        # if self.get_shape() is None :
        #     return
        p, c, r = self.get_position(), self.color, self.radius
        # viewer.draw_polygon(self.get_shape().boundary.coords, c, filled, r/5)
        viewer.draw_circle(p, r, c, filled, r / 5)
        # viewer.draw_circle((p[0]-r, p[1]), r/20, 'red', filled, r / 15)
        # viewer.draw_circle((p[0]+r, p[1]), r/20, 'red', filled, r / 15)
        # print(r)

        if self.odor_id is not None:
            if self.odor_intensity > 0:
                if self.model.odor_aura:
                    viewer.draw_circle(p, r * 1.5, c, False, r / 10)
                    viewer.draw_circle(p, r * 2.0, c, False, r / 15)
                    viewer.draw_circle(p, r * 3.0, c, False, r / 20)
        if self.selected:
            # viewer.draw_polygon(self.get_shape(1.1).boundary.coords, self.model.selection_color, False, r / 5)
            viewer.draw_circle(p, r * 1.1, self.model.selection_color, False, r / 5)

    def contained(self, point):
        return eudis5(self.get_position(), point) <= self.radius
        # return euclidean(self.get_position(), point)<=self.radius
        # return Point(self.get_position()).distance(Point(point))<=self.radius
        # return Circle(self.get_position(), radius=self.radius).contains_point(point)

#@todo could add thermosource here.