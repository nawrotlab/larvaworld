"""
This module contains all methods and classes relevant in data management,analysis, storage
 as well as the methods supporting the import of experimental tracker datasets
"""

from __future__ import annotations

__displayname__ = "Data management"

# Public API: expose core dataset classes, evaluation helpers, and lab importers.
__all__: list[str] = [
    # Datasets
    "DatasetConfig",
    "ParamLarvaDataset",
    "BaseLarvaDataset",
    "LarvaDataset",
    "LarvaDatasetCollection",
    # Evaluation
    "Evaluation",
    "DataEvaluation",
    # Lab-specific importers
    "import_Schleyer",
    "import_Jovanic",
    "import_Berni",
    "import_Arguello",
    "import_DeepLabCut",
    "lab_specific_import_functions",
    # Calibration functions
    "vel_definition",
    "comp_stride_variation",
    "fit_metric_definition",
    "comp_segmentation",
    # Import aux functions
    "read_timeseries_from_raw_files_per_parameter",
    "convert_spine_files_to_per_parameter_txt",
    "estimate_timestep_from_timeseries",
    "count_midline_points_in_raw_data",
    "estimate_arena_dimensions",
    "discover_deeplabcut_source_directories",
]

_NAME_TO_MODULE = {
    # Datasets
    "DatasetConfig": "larvaworld.lib.process.dataset",
    "ParamLarvaDataset": "larvaworld.lib.process.dataset",
    "BaseLarvaDataset": "larvaworld.lib.process.dataset",
    "LarvaDataset": "larvaworld.lib.process.dataset",
    "LarvaDatasetCollection": "larvaworld.lib.process.dataset",
    # Evaluation
    "Evaluation": "larvaworld.lib.process.evaluation",
    "DataEvaluation": "larvaworld.lib.process.evaluation",
    # Lab importers
    "import_Schleyer": "larvaworld.lib.process.importing",
    "import_Jovanic": "larvaworld.lib.process.importing",
    "import_Berni": "larvaworld.lib.process.importing",
    "import_Arguello": "larvaworld.lib.process.importing",
    "import_DeepLabCut": "larvaworld.lib.process.importing",
    "lab_specific_import_functions": "larvaworld.lib.process.importing",
    # Calibration functions
    "vel_definition": "larvaworld.lib.process.calibration",
    "comp_stride_variation": "larvaworld.lib.process.calibration",
    "fit_metric_definition": "larvaworld.lib.process.calibration",
    "comp_segmentation": "larvaworld.lib.process.calibration",
    # Import aux functions
    "read_timeseries_from_raw_files_per_parameter": "larvaworld.lib.process.import_aux",
    "convert_spine_files_to_per_parameter_txt": "larvaworld.lib.process.import_aux",
    "estimate_timestep_from_timeseries": "larvaworld.lib.process.import_aux",
    "count_midline_points_in_raw_data": "larvaworld.lib.process.import_aux",
    "estimate_arena_dimensions": "larvaworld.lib.process.import_aux",
    "discover_deeplabcut_source_directories": "larvaworld.lib.process.import_aux",
}


def __getattr__(name: str):
    module_path = _NAME_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod = import_module(module_path)
    obj = getattr(mod, name)
    globals()[name] = obj
    return obj


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
