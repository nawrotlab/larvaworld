"""
Widget for editing the borders placed in an environment.
"""

from __future__ import annotations

import param

from .widget_base import (
    classdict_editor,
    collapsible_family_box,
    family_box,
    parameterized_editor,
)

__all__ = ["build_border_widget"]


def _ordered_names(instance: param.Parameterized, preferred: list[str]) -> list[str]:
    """Order a configuration's fields for display.

    Args:
        obj: The object being edited.

    Returns:
        The field names, in display order.
    """
    return [name for name in preferred if name in instance.param and name != "name"]


def _build_border_item_widget(border: param.Parameterized, key: str) -> object:
    """Build the editor for one border.

    Args:
        border: The border being edited.

    Returns:
        The editor component.
    """
    return family_box(
        key,
        parameterized_editor(
            border,
            parameter_order=_ordered_names(
                border,
                ["unique_id", "color", "width", "vertices"],
            ),
        ),
    )


def build_border_widget(env_conf: param.Parameterized) -> object:
    """Build the widget editing the environment's borders.

    Args:
        **kwargs: Widget settings.

    Returns:
        The widget component.
    """
    return collapsible_family_box(
        "Border list",
        classdict_editor(
            env_conf,
            name="border_list",
            parameter=env_conf.param["border_list"],
            item_label="border",
            build_item_editor=_build_border_item_widget,
            wrap=False,
        ),
        css_classes=[
            "lw-import-datasets-config-subfamily-card",
            "lw-import-datasets-config-compact-card",
        ],
    )
