from __future__ import annotations

import panel as pn
import param

from .arena_widget import build_area_widget
from .border_widget import build_border_widget
from .food_widget import build_food_conf_widget
from .odorscape_widget import build_odorscape_widget
from .thermoscape_widget import build_thermoscape_widget
from .widget_base import collapsible_family_box, family_box
from .windscape_widget import build_windscape_widget

__all__ = ["build_env_params_widget"]


def build_env_params_widget(
    env_conf: param.Parameterized,
    *,
    wrap: bool = True,
) -> object:
    children = [
        build_area_widget(env_conf.arena),
        build_food_conf_widget(env_conf.food_params),
        build_border_widget(env_conf),
        collapsible_family_box(
            "Environment scapes",
            build_odorscape_widget(env_conf),
            build_windscape_widget(env_conf),
            build_thermoscape_widget(env_conf),
            css_classes=[
                "lw-import-datasets-config-subfamily-card",
                "lw-import-datasets-config-compact-card",
            ],
        ),
    ]
    if not wrap:
        return pn.Column(*children, sizing_mode="stretch_width", margin=0)
    return family_box(
        "Environment",
        *children,
    )
