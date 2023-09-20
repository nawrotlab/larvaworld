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
    """
    Base class for LarvaworldAgent.

    This is the base class for all agent types in Larvaworld.

    Parameters
    ----------
    odor : dict, optional
        An optional dictionary containing odor information of the agent.

    Attributes
    ----------
    odor : Odor
        The odor of the agent.
    dt : float
        The time step used for the model simulation.

    Methods
    -------
    step()
        Placeholder method for agent's step behavior.
    """

    odor = ClassAttr(Odor, doc='The odor of the agent')

    @property
    def dt(self):
        return self.model.dt

    def step(self):
        pass

class PointAgent(RadiallyExtended, NonSpatialAgent):
    """
    Agent with a point representation.

    This agent class extends the NonSpatialAgent class and represents agents as points.

    Methods
    -------
    draw(v, filled=True)
        Draw the agent.
    draw_selected(v, **kwargs)
        Draw the selected agent.
    """

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
    """
    Agent with orientation.

    This agent class extends the PointAgent class and adds orientation to the agent.

    Methods
    -------
    __init__(**kwargs)
        Initialize the OrientedAgent.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MobilePointAgent(MobilePoint, PointAgent):
    """
    Mobile point agent.

    This agent class extends the PointAgent class and adds mobility with point representation.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MobileAgent(MobileVector, PointAgent):
    """
    Mobile agent with vector representation.

    This agent class extends the PointAgent class and uses vector representation for mobility.

    Properties
    ----------
    last_orientation_vel : float
        The last orientation velocity of the agent.
    last_pos_vel : float
        The last position velocity of the agent.
    last_scaled_pos_vel : float
        The last scaled position velocity of the agent.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def last_orientation_vel(self):
        return self.last_delta_orientation / self.dt

    @property
    def last_pos_vel(self):
        return self.last_delta_pos / self.dt

    @property
    def last_scaled_pos_vel(self):
        return self.last_delta_pos / self.length / self.dt
