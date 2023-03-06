import numpy as np
from shapely import affinity, geometry, measurement

from larvaworld.lib import aux
from larvaworld.lib.model.agents import LarvaworldAgent
from larvaworld.lib.model.deb.substrate import Substrate


class Source(LarvaworldAgent):
    def __init__(self, shape_vertices=None, can_be_carried=False, can_be_displaced=False, shape='circle', **kwargs):
        super().__init__(**kwargs)
        self.shape_vertices = shape_vertices
        self.can_be_carried = can_be_carried
        self.can_be_displaced = can_be_displaced
        self.is_carried_by = None

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
            return geometry.Point(p).buffer(self.radius * scale)
        else:
            p0 = geometry.Polygon(self.get_vertices())
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
                    in_tank = aux.inside_polygon(points=[self.pos], tank_polygon=self.model.space.polygon)
                    if not in_tank:
                        if self.regeneration:
                            self.pos = aux.xy_uniform_circle(1, **self.regeneration_pos)[0]
                        else :
                            self.model.delete_agent(self)


class Food(Source):
    def __init__(self, amount=1.0, quality=1.0, default_color='green', type='standard', **kwargs):
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
