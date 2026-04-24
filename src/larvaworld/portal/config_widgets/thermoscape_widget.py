from __future__ import annotations

from typing import Any, Callable

import panel as pn
import param

from .widget_base import classattr_section, parameterized_editor, widget_block

__all__ = ["build_thermoscape_widget"]


_DEFAULT_THERMO_SOURCES = {
    "0": [0.5, 0.05],
    "1": [0.05, 0.5],
    "2": [0.5, 0.95],
    "3": [0.95, 0.5],
}
_DEFAULT_THERMO_DTEMPS = {
    "0": 8.0,
    "1": -8.0,
    "2": 8.0,
    "3": -8.0,
}


def _ordered_names(instance: param.Parameterized, preferred: list[str]) -> list[str]:
    return [name for name in preferred if name in instance.param and name != "name"]


def _attribute_control(
    owner: Any,
    *,
    attr_name: str,
    widget: pn.widgets.Widget,
    doc: str | None = None,
    after_update: Callable[[], None] | None = None,
) -> pn.Column:
    state = {"syncing": False}
    widget.value = getattr(owner, attr_name)

    def _push(event: param.parameterized.Event) -> None:
        if state["syncing"]:
            return
        setattr(owner, attr_name, event.new)
        if after_update is not None:
            after_update()

    widget.param.watch(_push, "value")

    container = widget_block(widget, doc=doc)
    container._widget = widget
    return container


def _build_thermoscape_editor(thermoscape: param.Parameterized) -> object:
    if not hasattr(thermoscape, "plate_temp"):
        thermoscape.plate_temp = 22.0
    if not hasattr(thermoscape, "thermo_spread"):
        thermoscape.thermo_spread = 0.1
    if not hasattr(thermoscape, "thermo_sources"):
        thermoscape.thermo_sources = dict(_DEFAULT_THERMO_SOURCES)
    if not hasattr(thermoscape, "thermo_source_dTemps"):
        thermoscape.thermo_source_dTemps = dict(_DEFAULT_THERMO_DTEMPS)

    def _regenerate() -> None:
        if hasattr(thermoscape, "generate_thermoscape"):
            thermoscape.generate_thermoscape()

    base_editor = parameterized_editor(
        thermoscape,
        parameter_order=_ordered_names(
            thermoscape,
            ["unique_id", "color", "grid_dims", "initial_value", "fixed_max"],
        ),
    )
    return pn.Column(
        base_editor,
        _attribute_control(
            thermoscape,
            attr_name="plate_temp",
            widget=pn.widgets.FloatInput(
                name="Plate temperature",
                step=0.5,
                sizing_mode="stretch_width",
            ),
            doc="Baseline plate temperature in degrees Celsius.",
            after_update=_regenerate,
        ),
        _attribute_control(
            thermoscape,
            attr_name="thermo_spread",
            widget=pn.widgets.FloatInput(
                name="Thermal spread",
                step=0.01,
                sizing_mode="stretch_width",
            ),
            doc="Gaussian spread parameter for thermal sources.",
            after_update=_regenerate,
        ),
        _attribute_control(
            thermoscape,
            attr_name="thermo_sources",
            widget=pn.widgets.LiteralInput(
                name="Thermal source positions",
                sizing_mode="stretch_width",
            ),
            doc="Dictionary mapping source IDs to [x, y] thermal source coordinates.",
            after_update=_regenerate,
        ),
        _attribute_control(
            thermoscape,
            attr_name="thermo_source_dTemps",
            widget=pn.widgets.LiteralInput(
                name="Thermal source dTemps",
                sizing_mode="stretch_width",
            ),
            doc="Dictionary mapping source IDs to temperature deltas relative to the plate.",
            after_update=_regenerate,
        ),
        sizing_mode="stretch_width",
        margin=0,
    )


def build_thermoscape_widget(env_conf: param.Parameterized) -> object:
    return classattr_section(
        env_conf,
        name="thermoscape",
        parameter=env_conf.param["thermoscape"],
        title="Thermoscape",
        build_editor=_build_thermoscape_editor,
        box_css_classes=["lw-import-datasets-config-subfamily"],
        title_css_classes=["lw-import-datasets-config-subfamily-title"],
        enable_control="switch",
    )
