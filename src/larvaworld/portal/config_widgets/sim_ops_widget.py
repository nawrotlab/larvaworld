from __future__ import annotations

import panel as pn
import param

from .widget_base import collapsible_family_box, family_box, param_controls

__all__ = ["build_sim_ops_widget"]

_TIMING_FIELDS = ("duration", "Nsteps")
_FRAMERATE_FIELDS = ("fr", "dt", "constant_framerate")
_PHYSICS_FIELDS = ("Box2D", "larva_collisions")


def _present_fields(
    owner: param.Parameterized, candidate_fields: tuple[str, ...]
) -> list[str]:
    return [field for field in candidate_fields if field in owner.param]


def build_sim_ops_widget(
    owner: param.Parameterized,
    *,
    wrap: bool = True,
) -> object:
    sections: list[pn.viewable.Viewable] = []

    timing_fields = _present_fields(owner, _TIMING_FIELDS)
    if timing_fields:
        sections.append(
            family_box(
                "Timing",
                param_controls(owner, parameters=timing_fields),
                css_classes=[
                    "lw-import-datasets-config-subfamily-card",
                    "lw-import-datasets-config-compact-card",
                ],
            )
        )

    framerate_fields = _present_fields(owner, _FRAMERATE_FIELDS)
    if framerate_fields:
        sections.append(
            family_box(
                "Framerate",
                param_controls(owner, parameters=framerate_fields),
                css_classes=[
                    "lw-import-datasets-config-subfamily-card",
                    "lw-import-datasets-config-compact-card",
                ],
            )
        )

    physics_fields = _present_fields(owner, _PHYSICS_FIELDS)
    if physics_fields:
        sections.append(
            family_box(
                "Physics",
                param_controls(owner, parameters=physics_fields),
                css_classes=[
                    "lw-import-datasets-config-subfamily-card",
                    "lw-import-datasets-config-compact-card",
                ],
            )
        )

    if not wrap:
        return pn.Column(*sections, sizing_mode="stretch_width", margin=0)
    return collapsible_family_box("Simulation Settings", *sections)
