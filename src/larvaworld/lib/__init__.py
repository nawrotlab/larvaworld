"""
The core functionalities of the larvaworld platform
"""

__displayname__ = "Core library"

__all__ = [
    # Global registry object
    "funcs",
    # Subpackages
    "param",
    "reg",
    "plot",
    "model",
    "process",
    "screen",
    "sim",
    # Re-exported from process.dataset
    "ParamLarvaDataset",
    "BaseLarvaDataset",
    "LarvaDataset",
    "LarvaDatasetCollection",
]

from .util import AttrDict


class FunctionDict:
    """
    Class that manages different groups of functions.
    Each group is registered as a dictionary under a certain class attribute.
    The attribute can then be used as a decorator to register a function under the specified group.

    Attributes:
        graphs (AttrDict): A dictionary to store graph-related functions.
        graph_required_data (AttrDict): A dictionary to store required data for graphs.
        stored_confs (AttrDict): A dictionary to store the default configuration-generating functions.
        param_computing (AttrDict): A dictionary to store parameter computing functions.

    Methods:
        param(name):
            Registers a function under the 'param_computing' group.

    Args:
                name (str): The name of the function to register.

    Returns:
                function: A wrapper function that registers the given function.

        graph(name, required={}):
            Registers a function under the 'graphs' group and stores its required data.

    Args:
                name (str): The name of the function to register.
                required (dict, optional): A dictionary of required data for the graph. Defaults to an empty dictionary.

    Returns:
                function: A wrapper function that registers the given function.

        stored_conf(name):
            Registers a function under the 'stored_confs' group.

    Args:
                name (str): The name of the function to register.

    Returns:
                function: A wrapper function that registers the given function.

        register_func(name, group):
            Registers a function under the specified group.

    Args:
                name (str): The name of the function to register.
                group (str): The group under which to register the function.

    Returns:
                function: A wrapper function that registers the given function.

    Raises:
                AttributeError: If the specified group does not exist.

    """

    def __init__(self):
        self.graphs = AttrDict()
        self.graph_required_data = AttrDict()
        self.stored_confs = AttrDict()
        self.param_computing = AttrDict()

    def param(self, name):
        return self.register_func(name, "param_computing")

    def graph(self, name, required={}):
        self.graph_required_data[name] = AttrDict(required)
        return self.register_func(name, "graphs")

    def stored_conf(self, name):
        return self.register_func(name, "stored_confs")

    def register_func(self, name, group):
        if not hasattr(self, group):
            raise
        d = getattr(self, group)

        def wrapper(func):
            d[name] = func
            return func

        return wrapper


funcs = FunctionDict()


def __getattr__(name):
    """
    Lazily import subpackages and dataset classes to keep lib import lightweight.
    """
    if name in {"param", "reg", "plot", "model", "process", "screen", "sim"}:
        from importlib import import_module

        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    # Lazy import dataset classes through process facade
    if name in {
        "ParamLarvaDataset",
        "BaseLarvaDataset",
        "LarvaDataset",
        "LarvaDatasetCollection",
    }:
        from . import process

        obj = getattr(process, name)
        globals()[name] = obj
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Dataset classes are available via lazy loading through process facade
