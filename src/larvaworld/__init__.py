"""
Larvaworld : A Drosophila larva behavioral analysis and simulation platform
"""

from __future__ import annotations

import os

__author__ = "Panagiotis Sakagiannis"
__license__ = "MIT License"
__copyright__ = "2024, Panagiotis Sakagiannis"

# TODO : the automatic version naming requires the package itself to be installed. Woraround by simply naming it 0.0.0
try:
    import importlib.metadata

    __version__ = importlib.metadata.version("larvaworld")
except:
    __version__ = "2.0.0"

__displayname__ = "larvaworld"
__name__ = "larvaworld"

__all__: list[str] = [
    # Lazy-loaded subpackages (via __getattr__)
    "lib",
    "cli",
    # Functions and constants
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

VERBOSE: int = 2  #: Global verbosity level for vprint function.


def vprint(text: str = "", verbose: int = 0) -> None:
    """
    Print text if the verbosity level is greater than or equal to the global VERBOSE level.

    Parameters
    ----------
    text (str): The text to print.
    verbose (int): The verbosity level of the message.

    """
    if verbose >= VERBOSE:
        print(text)


ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))  #: Package root directory.
DATA_DIR: str = f"{ROOT_DIR}/data"  #: Data directory (data/ relative to package root).
SIM_DIR: str = f"{DATA_DIR}/SimGroup"  #: Simulation data directory.
BATCH_DIR: str = f"{SIM_DIR}/batch_runs"  #: Batch runs directory.
CONF_DIR: str = (
    f"{ROOT_DIR}/lib/reg/confDicts"  #: Configuration dictionaries directory.
)
TEST_DIR: str = f"{ROOT_DIR}/../../tests"  #: Tests directory (relative).

os.makedirs(CONF_DIR, exist_ok=True)


SIMTYPES: list[str] = ["Exp", "Batch", "Ga", "Eval", "Replay"]
#: List of available simulation types for CLI and API usage.

CONFTYPES: list[str] = [
    "Env",
    "LabFormat",
    "Ref",
    "Model",
    "Trial",
    "Exp",
    "Batch",
    "Ga",
]
#: List of available configuration types in the registry.
# GROUPTYPES = ['LarvaGroup', 'FoodGroup', 'epoch']


def __getattr__(name):
    """
    Lazily import selected subpackages to keep root import lightweight.

    This preserves access patterns like `larvaworld.lib` and `larvaworld.cli`
    without importing them eagerly at package import time.
    """
    if name in {"lib", "cli"}:
        from importlib import import_module

        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
