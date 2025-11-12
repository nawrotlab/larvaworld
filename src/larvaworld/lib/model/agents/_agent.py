from __future__ import annotations
from typing import Any
import os
import warnings

# Deprecation: discourage deep imports from internal module paths
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.agents import MobileAgent'"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.agents import MobileAgent'",
        DeprecationWarning,
        stacklevel=2,
    )

from ...param import (
    ClassAttr,
    MobilePoint,
    MobileVector,
    OrientedPoint,
    RadiallyExtended,
    Viewable,
)
from ...param import Odor

# ScreenManager and IDBox imports deferred due to circular dependency - will be imported when needed
from ..object import GroupedObject

__all__: list[str] = [
    "NonSpatialAgent",
    "PointAgent",
    "OrientedAgent",
    "MobilePointAgent",
    "MobileAgent",
]

__displayname__ = "Agent"


class NonSpatialAgent(GroupedObject):
    """
    Base agent class without spatial positioning.

    Provides minimal agent functionality including unique ID, group membership,
    and odor signature, without position or movement capabilities. Used as
    foundation for spatial agent classes.

    Attributes:
        odor: Odor signature of the agent (Odor instance)
        model: Reference to the ABM model containing this agent
        unique_id: Unique identifier for this agent
        dt: Timestep duration from model (property)

    Example:
        >>> agent = NonSpatialAgent(unique_id='agent_001', odor={'id': 'odorA'})
        >>> agent.step()  # Base implementation (no-op)
    """

    __displayname__ = "Non-spatial agent"

    odor = ClassAttr(Odor, doc="The odor of the agent")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def dt(self) -> float:
        return self.model.dt

    def step(self) -> None:
        pass


class PointAgent(RadiallyExtended, NonSpatialAgent, Viewable):
    """
    Spatial agent with point geometry and visualization.

    Extends NonSpatialAgent with 2D position, radius, and drawing capabilities.
    Combines RadiallyExtended (circular shape) with Viewable (color/visibility)
    to provide basic spatial agent with visual representation.

    Attributes:
        pos: Current (x, y) position in meters
        radius: Agent radius for drawing and collision detection
        color: Display color (string name or RGB tuple)
        visible: Whether agent is currently visible
        id_box: ID label box for visualization

    Example:
        >>> agent = PointAgent(pos=(0.5, 0.5), radius=0.001, color='blue')
        >>> agent.draw(screen_manager, filled=True)
    """

    __displayname__ = "Point agent"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.set_default_color(self.color)
        # Lazy import to avoid circular dependency
        from ...screen.rendering import IDBox

        self.id_box = IDBox(agent=self)

    def draw(self, v: Any, filled: bool = True) -> None:
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

    def draw_selected(self, v: Any, **kwargs: Any) -> None:
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
    Spatial agent with position and directional orientation.

    Extends PointAgent with orientation angle, enabling directional movement
    and heading-based behaviors. Combines OrientedPoint spatial properties
    with PointAgent visualization.

    Attributes:
        orientation: Current heading angle in radians
        front_orientation: Forward-facing orientation angle
        rear_orientation: Backward-facing orientation angle

    Example:
        >>> agent = OrientedAgent(pos=(0.5, 0.5), orientation=np.pi/4)
        >>> agent.set_orientation(np.pi/2)  # Face north
    """

    __displayname__ = "Oriented agent"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class MobilePointAgent(MobilePoint, PointAgent):
    """
    Mobile spatial agent with velocity tracking.

    Extends PointAgent with linear and angular velocity properties,
    enabling dynamic movement in 2D space. Combines MobilePoint kinematics
    with PointAgent visualization and odor signature.

    Attributes:
        lin_vel: Linear velocity magnitude in m/s
        ang_vel: Angular velocity in rad/s

    Example:
        >>> agent = MobilePointAgent(pos=(0.5, 0.5), lin_vel=0.001)
        >>> agent.set_lin_vel(0.002)  # Update velocity
    """

    __displayname__ = "Mobile point agent"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class MobileAgent(MobileVector, PointAgent):
    """
    Mobile spatial agent with vector representation and velocity tracking.

    Extends PointAgent with oriented vector geometry (front/rear ends),
    linear and angular velocities, and comprehensive motion properties.
    Foundation for larva agents with directional movement.

    Attributes:
        length: Vector length in meters (body length)
        orientation: Heading angle in radians
        lin_vel: Linear velocity in m/s
        ang_vel: Angular velocity in rad/s
        last_pos_vel: Previous timestep translational velocity (property)
        last_scaled_pos_vel: Previous velocity scaled by body length (property)
        last_orientation_vel: Previous timestep angular velocity (property)

    Example:
        >>> agent = MobileAgent(pos=(0.5, 0.5), length=0.003, orientation=0)
        >>> agent.update_all(lin_vel=0.001, ang_vel=0.1)
    """

    __displayname__ = "Mobile agent"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def last_orientation_vel(self) -> float:
        """The last angular velocity of the agent."""
        return self.last_delta_orientation / self.dt

    @property
    def last_pos_vel(self) -> np.ndarray:
        """The last translational velocity of the agent."""
        return self.last_delta_pos / self.dt

    @property
    def last_scaled_pos_vel(self) -> np.ndarray:
        """The last translational velocity of the agent, scaled to its vector length."""
        return self.last_delta_pos / self.length / self.dt
