"""
Larvaworld : A Drosophila larva behavioral analysis and simulation platform
"""

__author__ = "Panagiotis Sakagiannis"
__license__ = "GNU GENERAL PUBLIC LICENSE"
__copyright__ = "2024, Panagiotis Sakagiannis"

# TODO : the automatic version naming requires the package itself to be installed. Woraround by simply naming it 0.0.0
try:
    import importlib.metadata

    __version__ = importlib.metadata.version("larvaworld")
except:
    __version__ = "1.0.0"

__displayname__ = "larvaworld"
__name__ = "larvaworld"

__all__ = [
    "vprint",
    "ROOT_DIR",
    "DATA_DIR",
    "SIM_DIR",
    "BATCH_DIR",
    "CONF_DIR",
    "TEST_DIR",
    "SIMTYPES",
    "CONFTYPES",
]

VERBOSE = 2


def vprint(text="", verbose=0):
    """
    Print text if the verbosity level is greater than or equal to the global VERBOSE level.

    Parameters
    ----------
    text (str): The text to print.
    verbose (int): The verbosity level of the message.

    """
    if verbose >= VERBOSE:
        print(text)


import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = f"{ROOT_DIR}/data"
SIM_DIR = f"{DATA_DIR}/SimGroup"
BATCH_DIR = f"{SIM_DIR}/batch_runs"
CONF_DIR = f"{ROOT_DIR}/lib/reg/confDicts"
TEST_DIR = f"{ROOT_DIR}/../../tests"

os.makedirs(CONF_DIR, exist_ok=True)


SIMTYPES = ["Exp", "Batch", "Ga", "Eval", "Replay"]
CONFTYPES = ["Env", "LabFormat", "Ref", "Model", "Trial", "Exp", "Batch", "Ga"]
# GROUPTYPES = ['LarvaGroup', 'FoodGroup', 'epoch']

from . import lib, cli
