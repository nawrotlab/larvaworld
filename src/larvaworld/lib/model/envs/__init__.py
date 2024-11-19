"""
Environment classes for the agent-based-modeling simulations,
including the arena and any objects and impassable obstacles located within it,
as well as any existing sensory landscapes.
"""

from .arena import Arena
from .obstacle import *
from .valuegrid import *

__displayname__ = "Environment"
