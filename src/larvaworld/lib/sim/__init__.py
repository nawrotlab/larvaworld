"""
Launchers of the diverse available simulation modes
"""

from .ABM_model import ABModel
from .base_run import BaseRun
from .dataset_replay import ReplayRun


from .single_run import ExpRun
from .subprocess_run import Exec
from .model_evaluation import EvalRun


from .batch_run import BatchRun


# from .model_calibration import Calibration
from .genetic_algorithm import GAlauncher

__displayname__ = 'Simulation'