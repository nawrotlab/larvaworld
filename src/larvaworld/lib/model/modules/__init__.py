"""
Modules comprising the layered behavioral architecture modeling the nervous system,body and metabolism
"""

from .oscillator import Timer,Oscillator
from .basic import Effector, StepEffector, StepOscillator, SinOscillator
from .feeder import Feeder
from .crawler import SquareOscillator, PhaseOscillator, GaussOscillator
from .sensor import Olfactor, Toucher, Thermosensor, WindSensor
from .memory import RLOlfMemory, RLTouchMemory, RemoteBrianModelMemory
from .turner import NeuralOscillator
from .crawl_bend_interference import SquareCoupling, DefaultCoupling, PhasicCoupling
from .intermitter import Intermitter, BranchIntermitter, NengoIntermitter
from .locomotor import Locomotor, DefaultLocomotor
from .brain import Brain, DefaultBrain

__displayname__ = 'Modular behavioral architecture'



