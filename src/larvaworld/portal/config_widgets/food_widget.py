from __future__ import annotations

import param

from .distribution_widget import build_distribution_widget
from .widget_base import (
    classattr_section,
    classdict_editor,
    collapsible_family_box,
    family_box,
    parameterized_editor,
)

__all__ = ["build_food_conf_widget"]


def _ordered_names(instance: param.Parameterized, preferred: list[str]) -> list[str]:
    return [name for name in preferred if name in instance.param and name != "name"]


def _build_food_grid_editor(food_grid: param.Parameterized) -> object:
    return parameterized_editor(
        food_grid,
        parameter_order=_ordered_names(
            food_grid,
            [
                "unique_id",
                "color",
                "grid_dims",
                "initial_value",
                "fixed_max",
                "substrate",
            ],
        ),
    )


def _build_food_unit_item(unit: param.Parameterized, key: str) -> object:
    return family_box(
        key,
        parameterized_editor(
            unit,
            parameter_order=_ordered_names(
                unit,
                [
                    "unique_id",
                    "group",
                    "pos",
                    "radius",
                    "color",
                    "amount",
                    "odor",
                    "substrate",
                    "can_be_carried",
                    "can_be_displaced",
                    "regeneration",
                    "regeneration_pos",
                ],
            ),
        ),
    )


def _build_food_group_item(group: param.Parameterized, key: str) -> object:
    return family_box(
        key,
        parameterized_editor(
            group,
            parameter_order=_ordered_names(
                group,
                [
                    "distribution",
                    "group",
                    "radius",
                    "color",
                    "amount",
                    "odor",
                    "substrate",
                    "can_be_carried",
                    "can_be_displaced",
                    "regeneration",
                    "regeneration_pos",
                ],
            ),
            custom_builders={
                "distribution": lambda inst,
                name,
                _parameter: build_distribution_widget(getattr(inst, name))
            },
        ),
    )


def build_food_conf_widget(food_conf: param.Parameterized) -> object:
    return collapsible_family_box(
        "Food params",
        classattr_section(
            food_conf,
            name="food_grid",
            parameter=food_conf.param["food_grid"],
            title="Food grid",
            build_editor=_build_food_grid_editor,
            box_css_classes=["lw-import-datasets-config-subfamily"],
            title_css_classes=["lw-import-datasets-config-subfamily-title"],
        ),
        classdict_editor(
            food_conf,
            name="source_units",
            parameter=food_conf.param["source_units"],
            title="Source units",
            item_label="source unit",
            build_item_editor=_build_food_unit_item,
            box_css_classes=["lw-import-datasets-config-subfamily"],
            title_css_classes=["lw-import-datasets-config-subfamily-title"],
        ),
        classdict_editor(
            food_conf,
            name="source_groups",
            parameter=food_conf.param["source_groups"],
            title="Source groups",
            item_label="source group",
            build_item_editor=_build_food_group_item,
            box_css_classes=["lw-import-datasets-config-subfamily"],
            title_css_classes=["lw-import-datasets-config-subfamily-title"],
        ),
        css_classes=[
            "lw-import-datasets-config-subfamily-card",
            "lw-import-datasets-config-compact-card",
        ],
    )
