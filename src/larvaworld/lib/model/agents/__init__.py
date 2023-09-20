"""
Agent classes for the agent-based-modeling simulations.
"""

from ._agent import PointAgent, OrientedAgent, MobileAgent


from ._source import Source, Food
# from .controller import BaseController
from ._larva import Larva, LarvaMotile

from ._larva_replay import LarvaReplay,LarvaReplayContoured, LarvaReplaySegmented
from ._larva_sim import BaseController, LarvaSim
from .larva_robot import LarvaRobot,ObstacleLarvaRobot
from .larva_offline import LarvaOffline
# from .physics_controller import ManualController
# from .Box2D_larva import Box2DController

__displayname__ = 'Agents'