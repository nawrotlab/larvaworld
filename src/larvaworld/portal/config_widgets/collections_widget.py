from __future__ import annotations

import panel as pn
import param

from .widget_base import collapsible_family_box, family_box, param_controls

__all__ = ["build_collections_widget"]

_COLLECTIONS_FIELDS = ("collections",)


def build_collections_widget(
    owner: param.Parameterized,
    *,
    wrap: bool = True,
) -> object:
    fields = [field for field in _COLLECTIONS_FIELDS if field in owner.param]
    controls = param_controls(owner, parameters=fields)
    section = family_box(
        "Output Collections",
        controls,
        css_classes=[
            "lw-import-datasets-config-subfamily-card",
            "lw-import-datasets-config-compact-card",
        ],
    )
    if not wrap:
        return pn.Column(section, sizing_mode="stretch_width", margin=0)
    return collapsible_family_box("Collections", section)
