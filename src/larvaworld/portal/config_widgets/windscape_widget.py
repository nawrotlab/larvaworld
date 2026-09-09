"""
Widget for editing an environment's wind field.
"""

from __future__ import annotations

import param

from .widget_base import classattr_section, parameterized_editor

__all__ = ["build_windscape_widget"]


def _ordered_names(instance: param.Parameterized, preferred: list[str]) -> list[str]:
    """Order a configuration's fields for display.

    Args:
        obj: The object being edited.

    Returns:
        The field names, in display order.
    """
    return [name for name in preferred if name in instance.param and name != "name"]


def _build_windscape_editor(windscape: param.Parameterized) -> object:
    """Build the editor for the wind field.

    Args:
        windscape: The field being edited.

    Returns:
        The editor component.
    """
    return parameterized_editor(
        windscape,
        parameter_order=_ordered_names(
            windscape,
            ["unique_id", "color", "wind_direction", "wind_speed", "puffs"],
        ),
    )


def build_windscape_widget(env_conf: param.Parameterized) -> object:
    """Build the widget editing the environment's wind field.

    Args:
        **kwargs: Widget settings.

    Returns:
        The widget component.
    """
    return classattr_section(
        env_conf,
        name="windscape",
        parameter=env_conf.param["windscape"],
        title="Windscape",
        build_editor=_build_windscape_editor,
        box_css_classes=["lw-import-datasets-config-subfamily"],
        title_css_classes=["lw-import-datasets-config-subfamily-title"],
        enable_control="switch",
    )
