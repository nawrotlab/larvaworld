from __future__ import annotations
from typing import Any
import os
import warnings

# Deprecation: discourage deep imports from internal module paths
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.agents import Source, Food'"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib.model.agents import Source, Food'",
        DeprecationWarning,
        stacklevel=2,
    )
import numpy as np
import param

from ...param import ClassAttr, PositiveNumber, Substrate, xy_uniform_circle

# ScreenManager import deferred due to circular dependency - will be imported when needed
from . import PointAgent

__all__: list[str] = [
    "Source",
    "Food",
]

__displayname__ = "Food source"


class Source(PointAgent):
    """
    Base class for environmental resource sources.

    Represents sources of food, odor, or other resources in the environment.
    Supports carrying by larvae, displacement by wind, and regeneration
    after depletion or removal from arena.

    Attributes:
        can_be_carried: Whether larvae can carry this source (default: False)
        can_be_displaced: Whether wind/water can move this source (default: False)
        regeneration: Whether source regenerates after removal (default: False)
        regeneration_pos: Position parameters for regeneration (optional)
        is_carried_by: Reference to larva carrying this source (or None)

    Example:
        >>> source = Source(
        ...     pos=(0.5, 0.5),
        ...     radius=0.002,
        ...     can_be_displaced=True,
        ...     regeneration=True
        ... )
        >>> source.step()  # Update position if displaced by wind
    """

    can_be_carried = param.Boolean(
        False, label="carriable", doc="Whether the source can be carried around."
    )
    can_be_displaced = param.Boolean(
        False,
        label="displaceable",
        doc="Whether the source can be displaced by wind/water.",
    )
    regeneration = param.Boolean(False, doc="Whether the agent can be regenerated")
    regeneration_pos = param.Parameter(
        None, doc="Where the agent appears if regenerated"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.is_carried_by = None

        # # put all agents into same group (negative so that no collisions are detected)
        # self._fixtures[0].filterData.groupIndex = -1

    def step(self) -> None:
        if self.can_be_displaced:
            w = self.model.windscape
            if w is not None:
                ws, wo = w.wind_speed, w.wind_direction
                if ws != 0.0:
                    coef = ws * self.model.dt / self.radius * 10000
                    self.pos = (self.x + np.cos(wo) * coef, self.y + np.sin(wo) * coef)
                    in_tank = self.model.space.in_area(self.pos)
                    if not in_tank:
                        if self.regeneration:
                            self.pos = xy_uniform_circle(1, **self.regeneration_pos)[0]
                        else:
                            self.model.delete_agent(self)


class Food(Source):
    """
    Food source agent with nutritional substrate and depletion dynamics.

    Extends Source to represent consumable food patches with substrate
    quality, amount tracking, and visual depletion indication (color fading).
    Automatically removed from simulation when fully consumed.

    Attributes:
        amount: Current food amount available (0 to initial_amount)
        substrate: Nutritional substrate composition (Substrate instance)
        initial_amount: Original food amount at creation
        color: Visual color (fades from default to white as depleted)

    Example:
        >>> food = Food(
        ...     pos=(0.5, 0.5),
        ...     amount=5.0,
        ...     substrate={'type': 'standard', 'quality': 0.8}
        ... )
        >>> consumed = food.subtract_amount(1.0)  # Larva feeds
    """

    color = param.Color(default="green")
    amount = PositiveNumber(
        softmax=10.0, step=0.01, doc="The food amount in the source"
    )
    substrate = ClassAttr(Substrate, doc="The substrate where the agent feeds")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.initial_amount = self.amount

    def subtract_amount(self, amount: float) -> float:
        """
        Subtract a given amount of food from the source.

        Parameters
        ----------
        amount : float
            Amount of food to subtract.

        Returns
        -------
        float
            The actual amount subtracted, which may be less than the requested amount.

        Notes
        -----
        If the source runs out of food, it is deleted from the simulation.

        """
        prev_amount = self.amount
        if amount >= self.amount:
            self.amount = 0.0
            self.model.delete_source(self)
        else:
            self.amount -= amount
            r = self.amount / self.initial_amount
            try:
                self.color = (1 - r) * np.array((255, 255, 255)) + r * np.array(
                    self.default_color
                )
            except:
                pass
        return np.min([amount, prev_amount])

    def draw(self, v: Any, filled: bool | None = None) -> None:
        """
        Draws the agent on the screen.

        Parameters:
        v (ScreenManager): The screen manager responsible for drawing.
        filled (bool, optional): Whether the circle should be filled. Defaults to True if the agent's amount is greater than 0, otherwise False.

        Returns:
        None
        """
        filled = True if self.amount > 0 else False
        p, c, r = self.get_position(), self.color, self.radius
        v.draw_circle(p, r, c, filled, r / 5)
        super().draw(v=v, filled=filled)
