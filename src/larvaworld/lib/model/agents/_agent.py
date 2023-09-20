import numpy as np
from larvaworld.lib import aux
from larvaworld.lib.param import OrientedPoint, RadiallyExtended, ClassAttr, MobilePoint, MobileVector
from larvaworld.lib.param.composition import Odor
from larvaworld.lib.screen import LabelledGroupedObject

__all__ = [
    'NonSpatialAgent',
    'PointAgent',
    'OrientedAgent',
    'MobilePointAgent',
    'MobileAgent',
]

__displayname__ = 'Agent'

class NonSpatialAgent(LabelledGroupedObject):
    """An agent lacking spatial positioning in space.

    Args:
        odor (dict) : An optional dictionary containing odor information of the agent.

    """

    __displayname__ = 'Non-spatial agent'

    odor = ClassAttr(Odor, doc='The odor of the agent')

    @property
    def dt(self):
        return self.model.dt

    def step(self):
        pass

class PointAgent(RadiallyExtended, NonSpatialAgent):
    """ Agent with a point spatial representation.
    This agent class extends the NonSpatialAgent class and represents agents as points.
    """

    __displayname__ = 'Point agent'

    def draw(self, v, filled=True):
        if self.odor.peak_value > 0:
            if v.manager.odor_aura:
                kws = {
                    'color': self.color,
                    'filled': False,
                    'position': self.get_position(),
                }
                for i in [1.5, 2.0, 3.0]:
                    v.draw_circle(
                        radius=self.radius * i, width=0.001 / i, **kws)

    def draw_selected(self, v, **kwargs):
        r = self.radius
        v.draw_circle(position=self.get_position(), radius=self.radius * 0.5,
                      color=v.manager.selection_color, filled=False, width=0.0002)

class OrientedAgent(OrientedPoint, PointAgent):
    """ An agent represented as an oriented point in space.
    This agent class extends the PointAgent class and adds orientation to the agent.
    """

    __displayname__ = 'Oriented agent'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MobilePointAgent(MobilePoint, PointAgent):
    """A mobile point agent.
    This agent class extends the PointAgent class and adds mobility with point representation.
    """

    __displayname__ = 'Mobile point agent'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MobileAgent(MobileVector, PointAgent):
    """ An agent represented in space as a mobile oriented vector.
    This agent class extends the PointAgent class and uses vector representation for mobility.
    """

    __displayname__ = 'Mobile agent'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def last_orientation_vel(self):
        """The last angular velocity of the agent."""

        return self.last_delta_orientation / self.dt

    @property
    def last_pos_vel(self):
        """The last translational velocity of the agent."""

        return self.last_delta_pos / self.dt

    @property
    def last_scaled_pos_vel(self):
        """The last translational velocity of the agent, scaled to its vector length."""

        return self.last_delta_pos / self.length / self.dt
