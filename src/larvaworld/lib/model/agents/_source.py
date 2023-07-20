import numpy as np
import param
from shapely import affinity, geometry, measurement

from larvaworld.lib import aux
from larvaworld.lib.model.agents import PointAgent
from larvaworld.lib.model.deb.substrate import Substrate


class Source(PointAgent):
    can_be_carried = param.Boolean(False,label='carriable', doc='Whether the source can be carried around.')
    can_be_displaced = param.Boolean(False,label='displaceable', doc='Whether the source can be displaced by wind/water.')
    regeneration = param.Boolean(False, doc='Whether the agent can be regenerated')
    regeneration_pos = param.Parameter(None, doc='Where the agent appears if regenerated')


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_carried_by = None

        # # put all agents into same group (negative so that no collisions are detected)
        # self._fixtures[0].filterData.groupIndex = -1


    def step(self):
        if self.can_be_displaced:
            w = self.model.windscape
            if w is not None:
                ws, wo = w.wind_speed, w.wind_direction
                if ws != 0.0:
                    coef=ws * self.model.dt / self.radius * 10000
                    self.pos = (self.x + np.cos(wo) * coef, self.y + np.sin(wo) * coef)
                    in_tank = aux.inside_polygon(points=[self.pos], tank_polygon=self.model.space.polygon)
                    if not in_tank:
                        if self.regeneration:
                            self.pos = aux.xy_uniform_circle(1, **self.regeneration_pos)[0]
                        else :
                            self.model.delete_agent(self)



class Food(Source):
    default_color = param.Color(default='green')
    amount = aux.PositiveNumber(softmax=10.0, step=0.01, doc='The food amount in the source')
    substrate = aux.ClassAttr(Substrate, doc='The substrate where the agent feeds')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_amount = self.amount

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

    def draw(self, viewer, filled=None):
        filled = True if self.amount > 0 else False
        super().draw(viewer=viewer, filled=filled)



