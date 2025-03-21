"""
This module sets up the larvaworld registry where most functions, classes, and configurations are registered.
It is initialized automatically when importing the package and serves as an accessible database for all functionalities.
"""

__displayname__ = "Registry"

__all__ = [
    "default_refID",
    "default_ref",
    "default_modelID",
    "default_model",
    "units",
    "controls",
    "par",
    "graphs",
    "getPar",
    "loadRef",
]

import warnings

warnings.simplefilter(action="ignore")

from .. import util
from ... import vprint, TEST_DIR

vprint("Initializing larvaworld registry", 2)
# vprint(f"Initializing larvaworld v.{__version__} registry", 2)

from pint import UnitRegistry

units = UnitRegistry()
units.default_format = "~P"
units.setup_matplotlib(True)

from . import keymap

controls = keymap.ControlRegistry()

from .data_aux import *
from .distro import *

vprint("Function registry complete", 1)

from . import parFunc, parDB

par = parDB.ParamRegistry()

vprint("Parameter registry complete", 1)

from . import config
from .config import conf

from . import generators
from .generators import gen

from . import graph

graphs = graph.GraphRegistry()

from . import stored_confs

vprint("Configuration registry complete", 1)


def getPar(k=None, p=None, d=None, to_return="d"):
    """
    Shortcut function for easy use directly via the registry.
    See 'par.getPar' for more information.
    """
    return par.getPar(k=k, d=d, p=p, to_return=to_return)


def loadRef(id, **kwargs):
    """
    Shortcut function for easy use directly via the registry.
    See 'conf.Ref.loadRef' for more information.
    """
    return conf.Ref.loadRef(id=id, **kwargs)


def loadRefs(ids, **kwargs):
    """
    Shortcut function for easy use directly via the registry.
    See 'conf.Ref.loadRefs' for more information.
    """
    return conf.Ref.loadRefs(ids=ids, **kwargs)


def define_default_refID_by_running_test():
    """
    Defines the default reference dataset ID by running a test if no reference datasets are available.

    This function checks if there are any reference datasets available in `conf.Ref.confIDs`.
    If none are available, it automatically imports a reference dataset by running a specified test method
    from a test file. The test method is executed using the `runpy.run_path` function.

    Returns:
        str: The first reference dataset ID from `conf.Ref.confIDs`.

    Raises:
        AssertionError: If no reference dataset IDs are available after running the test method.

    """
    if len(conf.Ref.confIDs) == 0:
        filename = "test_import.py"
        filepath = f"{TEST_DIR}/{filename}"
        import_method = "test_import_Schleyer"
        vprint("No reference datasets are available.", 2)
        vprint(
            f"Automatically importing one by running the {import_method} method in {filename} file.",
            2,
        )
        import runpy

        runpy.run_path(filepath, run_name="__main__")[import_method]()
        assert len(conf.Ref.confIDs) > 0
    return conf.Ref.confIDs[0]


def define_default_refID(id="exploration.30controls"):
    """
    Defines the default reference dataset ID for the package.

    This function performs the following steps:
    1. Purges the existing reference dataset IDs, deleting the ones where the corresponding datasets are missing.
    2. Checks if there are no reference dataset IDs available.
    3. If no reference datasets are available, it imports one from the experimental data folder.
    4. If the respective configuration is not present in LabFormat, it resets the configurations.
    5. Imports the default dataset with specified parameters.
    6. Processes and annotates the dataset.
    7. Ensures that exactly one reference dataset ID is available after import.

    Returns:
        str: The first reference dataset ID from the list of reference dataset IDs.

    """
    R = conf.Ref
    R.cleanRefIDs()
    if id in R.confIDs:
        return id
    elif id == "exploration.30controls" and len(R.confIDs) == 0:
        vprint(
            "No reference datasets available.Automatically importing one from the experimental data folder.",
            2,
        )
        if "Schleyer" not in conf.LabFormat.confIDs:
            config.resetConfs(conftypes=["LabFormat"])
        g = conf.LabFormat.get("Schleyer")
        N = 30
        kws = {
            "parent_dir": "exploration",
            "merged": True,
            "color": "blue",
            "max_Nagents": N,
            "min_duration_in_sec": 60,
            "id": f"{N}controls",
            "refID": f"exploration.{N}controls",
        }
        d = g.import_dataset(**kws)
        d.process(is_last=False)
        d.annotate(is_last=True)
        assert len(R.confIDs) == 1
        return id
    else:
        raise (ValueError(f"Reference dataset with ID {id} not found"))


default_refID = define_default_refID()


def default_ref():
    return loadRef(default_refID, load=True)


default_modelID = "explorer"


def default_model():
    return conf.Model.getID(default_modelID)


config.resetConfs(init=True)

vprint("Registry configured!", 2)
