"""
This module sets up the larvaworld registry where most functions, classes, and configurations are registered.
It is initialized automatically when importing the package and serves as an accessible database for all functionalities.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

__displayname__ = "Registry"

__all__: list[str] = [
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
    # Eagerly imported modules
    "conf",
    "gen",
]

import os
import warnings

# Deprecation: importing this private subpackage directly is discouraged for users
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib import reg' or higher-level APIs"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Use public API: 'from larvaworld.lib import reg' or higher-level APIs",
        DeprecationWarning,
        stacklevel=2,
    )

warnings.simplefilter(action="ignore")

from .. import util
from ... import vprint, TEST_DIR

vprint("Initializing larvaworld registry", 2)
# vprint(f"Initializing larvaworld v.{__version__} registry", 2)

# Runtime import; type hinted via TYPE_CHECKING to avoid heavy import at type-check time
from pint import UnitRegistry

units: UnitRegistry = UnitRegistry()
units.default_format = "~P"
units.setup_matplotlib(True)

from . import keymap

controls: keymap.ControlRegistry = keymap.ControlRegistry()

# Removed star-imports of internal helpers to avoid eager heavy dependencies
# Symbols from `data_aux` and `distro` are now resolved lazily via __getattr__ below

vprint("Function registry complete", 1)

# Provide early aliases for helper functions used by parDB during initialization
# This avoids a circular import on package init now that star-imports were removed
from . import data_aux as _data_aux

prepare_LarvaworldParam = _data_aux.prepare_LarvaworldParam
build_LarvaworldParam = _data_aux.build_LarvaworldParam
get_LarvaworldParam = _data_aux.get_LarvaworldParam
sample_ps = _data_aux.sample_ps

from . import parFunc, parDB

par: parDB.ParamRegistry = parDB.ParamRegistry()

vprint("Parameter registry complete", 1)

from . import config
from .config import conf

from . import generators
from .generators import gen

from . import graph

# Lazy graphs registry to avoid heavy initialization at import time
_GRAPHS = None

from . import stored_confs

vprint("Configuration registry complete", 1)


def getPar(
    k: Any | None = None,
    p: Any | None = None,
    d: Any | None = None,
    to_return: str = "d",
) -> Any:
    """
    Shortcut function for easy use directly via the registry.
    See 'par.getPar' for more information.
    """
    return par.getPar(k=k, d=d, p=p, to_return=to_return)


def loadRef(id: str, **kwargs: Any) -> Any:
    """
    Shortcut function for easy use directly via the registry.
    See 'conf.Ref.loadRef' for more information.
    """
    return conf.Ref.loadRef(id=id, **kwargs)


def loadRefs(ids: list[str], **kwargs: Any) -> Any:
    """
    Shortcut function for easy use directly via the registry.
    See 'conf.Ref.loadRefs' for more information.
    """
    return conf.Ref.loadRefs(ids=ids, **kwargs)


class _ProcSpinner:
    """CLI spinner running in a dedicated subprocess to remain responsive.

    The spinner writes to stdout independently of the main process so that
    it keeps animating even when the main process is CPU-bound. It stops
    when the sentinel path is removed.
    """

    def __init__(self, message: str = ""):
        self._message = message
        self._sentinel_dir = None
        self._sentinel_path = None
        self._proc = None

    def start(self) -> None:
        import os
        import sys
        import tempfile
        import subprocess
        import shutil

        # Check if spinner is disabled via environment variable
        if os.getenv("LARVAWORLD_SHOW_PROGRESS", "1") == "0":
            return  # Silent mode - no spinner

        # Create a temp directory with a sentinel file that the child polls
        self._sentinel_dir = tempfile.mkdtemp(prefix="lw_spin_")
        self._sentinel_path = os.path.join(self._sentinel_dir, "run")
        with open(self._sentinel_path, "w", encoding="utf-8") as f:
            f.write("1")

        code = (
            "import sys, os, time\n"
            "p=sys.argv[1]\n"
            "msg=sys.argv[2] if len(sys.argv)>2 else ''\n"
            "syms=['|','/','-', chr(92)]\n"
            "CR=chr(13); ESC=chr(27)\n"
            "i=0\n"
            "try:\n"
            "    while os.path.exists(p):\n"
            "        sys.stdout.write(CR+msg+syms[i%len(syms)])\n"
            "        sys.stdout.flush()\n"
            "        i+=1\n"
            "        time.sleep(0.1)\n"
            "    sys.stdout.write(CR+ESC+'[K')\n"
            "    sys.stdout.flush()\n"
            "except KeyboardInterrupt:\n"
            "    pass\n"
        )

        # Launch spinner process
        self._proc = subprocess.Popen(
            [sys.executable, "-c", code, self._sentinel_path, self._message],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    def stop(self) -> None:
        import os
        import time
        import shutil

        if self._sentinel_path and os.path.exists(self._sentinel_path):
            try:
                os.remove(self._sentinel_path)
            except Exception:
                pass
        # Give the child some time to exit gracefully
        if self._proc is not None:
            try:
                for _ in range(30):
                    if self._proc.poll() is not None:
                        break
                    time.sleep(0.05)
            except Exception:
                pass
            try:
                if self._proc.poll() is None:
                    self._proc.terminate()
            except Exception:
                pass
        if self._sentinel_dir and os.path.isdir(self._sentinel_dir):
            try:
                shutil.rmtree(self._sentinel_dir, ignore_errors=True)
            except Exception:
                pass


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
        # Cold start: show a spinner while importing the reference dataset
        spinner = _ProcSpinner("Loading registry (this may take several minutes)... ")
        spinner.start()
        try:
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
        finally:
            spinner.stop()
        # Print success message in green after completion
        try:
            GREEN = "\033[92m"
            RESET = "\033[0m"
            print(
                f"{GREEN}No reference datasets available. Automatically imported one from the experimental data folder.{RESET}"
            )
        except Exception:
            # Best-effort fallback without color
            print(
                "No reference datasets available. Automatically imported one from the experimental data folder."
            )
        return id
    else:
        raise (ValueError(f"Reference dataset with ID {id} not found"))


# Lazily compute default_refID to avoid heavy work at import time
_CACHED_DEFAULT_REFID = None


def __getattr__(name):
    global _CACHED_DEFAULT_REFID
    global _GRAPHS
    if name == "default_refID":
        if _CACHED_DEFAULT_REFID is None:
            _CACHED_DEFAULT_REFID = define_default_refID()
        return _CACHED_DEFAULT_REFID
    if name == "graphs":
        if _GRAPHS is None:
            # Ensure plot modules are loaded before creating GraphRegistry
            # This loads all @funcs.graph decorators
            from .. import plot

            # Force loading of all plot submodules to register @funcs.graph decorators
            for submodule in [
                "bar",
                "bearing",
                "box",
                "deb",
                "epochs",
                "freq",
                "grid",
                "hist",
                "metric",
                "scape",
                "stridecycle",
                "table",
                "time",
                "traj",
            ]:
                getattr(plot, submodule)
            _GRAPHS = graph.GraphRegistry()
        return _GRAPHS
    if name == "funcs":
        # Lazy import of funcs from data_aux and distro modules
        from . import data_aux, distro

        # Import all symbols from data_aux and distro
        for module in [data_aux, distro]:
            for attr_name in dir(module):
                if not attr_name.startswith("_"):
                    globals()[attr_name] = getattr(module, attr_name)
        # funcs should now be available in globals
        return globals().get("funcs")
    # Lazy export of symbols from internal submodules to keep import light
    # This replaces the previous `from .data_aux import *` and `from .distro import *`
    try:
        from importlib import import_module

        for module_path in (
            "larvaworld.lib.reg.data_aux",
            "larvaworld.lib.reg.distro",
            "larvaworld.lib.reg.generators",
            "larvaworld.lib.reg.larvagroup",
            "larvaworld.lib.reg.stored_confs.essay_conf",
            "larvaworld.lib.reg.stored_confs.data_conf",
            "larvaworld.lib.reg.stored_confs.sim_conf",
        ):
            try:
                mod = import_module(module_path)
                if hasattr(mod, name):
                    obj = getattr(mod, name)
                    globals()[name] = obj
                    return obj
            except Exception:
                # Best-effort lazy resolution; ignore and try next module
                continue
    except Exception:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def default_ref() -> Any:
    return loadRef(__getattr__("default_refID"), load=True)


default_modelID: str = "explorer"


def default_model() -> Any:
    return conf.Model.getID(default_modelID)


# Ensure Model_dict decorator is executed before resetConfs
# This is needed because resetConfs accesses funcs.stored_confs["Model"]
try:
    from importlib import import_module

    import_module("larvaworld.lib.model.modules.module_modes")
except Exception:
    pass

config.resetConfs(init=True)

vprint("Registry configured!", 2)

# Ensure CLI-facing registry entries are available (lazy side-effect registration)
try:
    # Register CLI-facing generators via side-effects when available
    from ..sim import model_evaluation as _model_evaluation  # noqa: F401
    from ..sim import genetic_algorithm as _genetic_algorithm  # noqa: F401
except Exception:
    pass
