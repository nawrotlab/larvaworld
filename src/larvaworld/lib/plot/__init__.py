"""
Plotting facade providing lazy access to submodules to keep imports lightweight.

Public usage remains the same: symbols are available under
`larvaworld.lib.plot.<submodule>` and are loaded on first access.
"""

from __future__ import annotations

__displayname__ = "Plotting"

__all__: list[str] = [
    "util",
    "base",
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
    "time",
    "traj",
    "table",
    # Classes from base
    "AutoBasePlot",
    "AutoPlot",
    "GridPlot",
    # Functions from util
    "plot_quantiles",
    "plot_mean_and_range",
    "circular_hist",
    "circNarrow",
    "confidence_ellipse",
    "dataset_legend",
    "label_diff",
    "annotate_plot",
    "dual_half_circle",
    "save_plot",
    "process_plot",
    "prob_hist",
    "single_boxplot",
    "configure_subplot_grid",
    "define_end_ks",
    "get_vs",
    "color_epochs",
    # Functions from table
    "diff_df",
    "mpl_table",
    # Functions from deb
    "plot_debs",
    # Functions from scape
    "plot_2d",
    "plot_3pars",
    "plot_heatmap_PI",
]

_SUBMODULES = {
    name: f"{__name__}.{name}"
    for name in __all__
    if name not in ["AutoBasePlot", "AutoPlot", "GridPlot"]
    and not name.startswith(
        (
            "plot_",
            "circ",
            "confidence_",
            "dataset_",
            "label_",
            "annotate_",
            "dual_",
            "save_",
            "process_",
            "prob_",
            "single_",
            "configure_",
            "define_",
            "get_",
            "color_",
            "diff_",
            "mpl_",
        )
    )
    and name
    not in [
        "diff_df",
        "mpl_table",
        "plot_debs",
        "plot_2d",
        "plot_3pars",
        "plot_heatmap_PI",
    ]
}
_CLASS_TO_MODULE = {
    "AutoBasePlot": f"{__name__}.base",
    "AutoPlot": f"{__name__}.base",
    "GridPlot": f"{__name__}.base",
}
_FUNCTION_TO_MODULE = {
    "plot_quantiles": f"{__name__}.util",
    "plot_mean_and_range": f"{__name__}.util",
    "circular_hist": f"{__name__}.util",
    "circNarrow": f"{__name__}.util",
    "confidence_ellipse": f"{__name__}.util",
    "dataset_legend": f"{__name__}.util",
    "label_diff": f"{__name__}.util",
    "annotate_plot": f"{__name__}.util",
    "dual_half_circle": f"{__name__}.util",
    "save_plot": f"{__name__}.util",
    "process_plot": f"{__name__}.util",
    "prob_hist": f"{__name__}.util",
    "single_boxplot": f"{__name__}.util",
    "configure_subplot_grid": f"{__name__}.util",
    "define_end_ks": f"{__name__}.util",
    "get_vs": f"{__name__}.util",
    "color_epochs": f"{__name__}.util",
    # Functions from table
    "diff_df": f"{__name__}.table",
    "mpl_table": f"{__name__}.table",
    # Functions from deb
    "plot_debs": f"{__name__}.deb",
    # Functions from scape
    "plot_2d": f"{__name__}.scape",
    "plot_3pars": f"{__name__}.scape",
    "plot_heatmap_PI": f"{__name__}.scape",
}


def __getattr__(name: str):
    module_path = _SUBMODULES.get(name)
    if module_path is None:
        # Check if it's a class
        class_module_path = _CLASS_TO_MODULE.get(name)
        if class_module_path is not None:
            from importlib import import_module

            mod = import_module(class_module_path)
            obj = getattr(mod, name)
            globals()[name] = obj
            return obj
        # Check if it's a function
        function_module_path = _FUNCTION_TO_MODULE.get(name)
        if function_module_path is not None:
            from importlib import import_module

            mod = import_module(function_module_path)
            obj = getattr(mod, name)
            globals()[name] = obj
            return obj
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod = import_module(module_path)
    globals()[name] = mod
    return mod


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
