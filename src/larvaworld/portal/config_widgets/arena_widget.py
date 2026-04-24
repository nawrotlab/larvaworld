from __future__ import annotations

import param

from .widget_base import (
    collapsible_family_box,
    numeric_tuple_param_control,
    param_control,
)

__all__ = ["build_area_widget"]


def build_area_widget(area: param.Parameterized) -> object:
    return collapsible_family_box(
        "Arena",
        numeric_tuple_param_control(
            area,
            parameter_name="dims",
            labels=("Arena width", "Arena height"),
            numeric_type=float,
            doc=getattr(area.param["dims"], "doc", None),
            step=0.001,
        ),
        param_control(area, parameter_name="geometry"),
        param_control(area, parameter_name="torus"),
        css_classes=[
            "lw-import-datasets-config-subfamily-card",
            "lw-import-datasets-config-compact-card",
        ],
    )
