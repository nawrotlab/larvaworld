"""
Modules comprising the layered behavioral architecture modeling the nervous system,body and metabolism
"""


# from . import crawler, turner,crawl_bend_interference,intermitter

from .oscillator import *
from .basic import *
from .feeder import *
from .crawler import *
from .sensor import *
from .memory import *
from .turner import *
from .crawl_bend_interference import *
from .intermitter import Intermitter, BranchIntermitter
from .locomotor import *
from .brain import *
from .module_modes import *

__displayname__ = "Modular behavioral architecture"
