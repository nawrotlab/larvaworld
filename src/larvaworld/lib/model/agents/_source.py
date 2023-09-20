import numpy as np
import param

from larvaworld.lib import aux
from larvaworld.lib.model.agents import PointAgent
from larvaworld.lib.param import xy_uniform_circle,Substrate, PositiveNumber, ClassAttr

__all__ = [
    'Source',
    'Food',
]

__displayname__ = 'Food source'

class Source(PointAgent):
    """
    Base class for representing a source of something in the environment.

    Parameters:
    ----------
    can_be_carried : bool, default False
        Whether the source can be carried around.
    can_be_displaced : bool, default False
        Whether the source can be displaced by wind or water.
    regeneration : bool, default False
        Whether the agent can be regenerated.
    regeneration_pos : tuple or None, default None
        Where the agent appears if regenerated.

    Attributes:
    ----------
    is_carried_by : object or None
        Reference to the agent that is currently carrying this source.

    Methods:
    --------
    step()
        Perform a step in the simulation, possibly updating the source's position.

    Notes:
    ------
    This class is a base class for representing various types of sources in the environment.
    """

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
                    in_tank = self.model.space.in_area(self.pos)
                    if not in_tank:
                        if self.regeneration:
                            self.pos = xy_uniform_circle(1, **self.regeneration_pos)[0]
                        else :
                            self.model.delete_agent(self)



class Food(Source):
    """
    Class for representing a source of food in the environment.

    Parameters:
    ----------
    amount : float, default 10.0
        The initial amount of food in the source.
    substrate : Substrate, default None
        The substrate where the agent feeds.

    Attributes:
    ----------
    default_color : str
        Default color of the food source.
    initial_amount : float
        Initial amount of food in the source.

    Methods:
    --------
    subtract_amount(amount)
        Subtract a given amount of food from the source.
    draw(v, filled=None)
        Draw the food source on the visualization canvas.

    Notes:
    ------
    This class extends the `Source` class to represent food sources specifically.
    """

    default_color = param.Color(default='green')
    amount = PositiveNumber(softmax=10.0, step=0.01, doc='The food amount in the source')
    substrate = ClassAttr(Substrate, doc='The substrate where the agent feeds')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initial_amount = self.amount

    def subtract_amount(self, amount):
        """
        Subtract a given amount of food from the source.

        Parameters:
        ----------
        amount : float
            Amount of food to subtract.

        Returns:
        -------
        float
            The actual amount subtracted, which may be less than the requested amount.

        Notes:
        ------
        If the source runs out of food, it is deleted from the simulation.
        """
        prev_amount = self.amount
        self.amount -= amount
        if self.amount <= 0.0:
            self.amount = 0.0
            self.model.delete_agent(self)
        else:
            r = self.amount / self.initial_amount
            try:
                self.color = (1 - r) * np.array((255, 255, 255)) + r * np.array(self.default_color)
            except:
                pass
        return np.min([amount, prev_amount])

    def draw(self, v, filled=None):
        filled = True if self.amount > 0 else False
        p, c, r = self.get_position(), self.color, self.radius
        v.draw_circle(p, r, c, filled, r / 5)
        super().draw(v=v, filled=filled)



