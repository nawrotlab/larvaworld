from ...param import (
    ClassAttr,
    MobilePoint,
    MobileVector,
    OrientedPoint,
    RadiallyExtended,
    Viewable,
)
from ...param.composition import Odor
from ...screen.rendering import IDBox
from ...screen.drawing import ScreenManager
from ..object import GroupedObject

__all__ = [
    "NonSpatialAgent",
    "PointAgent",
    "OrientedAgent",
    "MobilePointAgent",
    "MobileAgent",
]

__displayname__ = "Agent"


class NonSpatialAgent(GroupedObject):
    """
    An agent lacking spatial positioning in space.

    Args:
        odor (dict) : An optional dictionary containing odor information of the agent.

    """

    __displayname__ = "Non-spatial agent"

    odor = ClassAttr(Odor, doc="The odor of the agent")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def dt(self):
        return self.model.dt

    def step(self):
        pass


class PointAgent(RadiallyExtended, NonSpatialAgent, Viewable):
    """
    Agent with a point spatial representation.
    This agent class extends the NonSpatialAgent class and represents agents as points.
    """

    __displayname__ = "Point agent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_default_color(self.color)
        self.id_box = IDBox(agent=self)

    def draw(self, v: ScreenManager, filled: bool = True) -> None:
        """
        Draws the agent on the screen.

        Parameters:
        v (ScreenManager): The screen manager responsible for rendering.
        filled (bool): Whether the agent should be drawn filled or not. Default is True.

        Returns:
        None
        """
        if self.odor.peak_value > 0:
            if v.odor_aura:
                kws = {
                    "color": self.color,
                    "filled": False,
                    "position": self.get_position(),
                }
                for i in [1.5, 2.0, 3.0]:
                    v.draw_circle(radius=self.radius * i, width=0.001 / i, **kws)

    def draw_selected(self, v: ScreenManager, **kwargs) -> None:
        """
        Draws a visual representation of the selected agent on the screen.

        Parameters:
        v (ScreenManager): The screen manager responsible for rendering.
        **kwargs: Additional keyword arguments.

        Returns:
        None
        """
        v.draw_circle(
            position=self.get_position(),
            radius=self.radius * 0.5,
            color=v.selection_color,
            filled=False,
            width=0.0002,
        )


class OrientedAgent(OrientedPoint, PointAgent):
    """
    An agent represented as an oriented point in space.
    This agent class extends the PointAgent class and adds orientation to the agent.
    """

    __displayname__ = "Oriented agent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MobilePointAgent(MobilePoint, PointAgent):
    """
    A mobile point agent.
    This agent class extends the PointAgent class and adds mobility with point representation.
    """

    __displayname__ = "Mobile point agent"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MobileAgent(MobileVector, PointAgent):
    """
    An agent represented in space as a mobile oriented vector.
    This agent class extends the PointAgent class and uses vector representation for mobility.
    """

    __displayname__ = "Mobile agent"

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
