import numpy as np

from larvaworld.lib import aux
from larvaworld.lib.param import OrientedPoint, RadiallyExtended, ClassAttr, MobilePoint
from larvaworld.lib.param.composition import Odor
from larvaworld.lib.screen import LabelledGroupedObject


class NonSpatialAgent(LabelledGroupedObject):
    """
                LarvaworldAgent base class for all agent types

                Note that the setup() method is called right after initialization as in the agentpy.Agent class
                This is contrary to the parent class

                Args:
                - odor: optional dictionary containing odor information of the agent.


            """
    odor = ClassAttr(Odor, doc='The odor of the agent')


    @property
    def dt(self):
        return self.model.dt

    def step(self):
        pass



class PointAgent(RadiallyExtended,NonSpatialAgent):


    def draw(self, v, filled=True):
        p, c, r = self.get_position(), self.color, self.radius
        if np.isnan(p).all():
            return
        # if v.manager.draw_centroid:


        if self.odor.peak_value > 0:
            if v.manager.odor_aura:
                v.draw_circle(p, r * 1.5, c, False, r / 10)
                v.draw_circle(p, r * 2.0, c, False, r / 15)
                v.draw_circle(p, r * 3.0, c, False, r / 20)
        if self.selected:
            v.draw_circle(p, r * 1.1, v.manager.selection_color, False, r / 5)


class OrientedAgent(OrientedPoint,PointAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MobileAgent(MobilePoint,PointAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

