from __future__ import annotations

from typing import Any

import param

from larvaworld.lib import reg, util
from larvaworld.lib.param.custom import ClassDict

from .distribution_widget import build_distribution_widget
from .widget_base import classdict_editor, family_box, parameterized_editor

__all__ = ["build_larva_group_widget", "build_larva_groups_widget"]


def _ordered_names(instance: param.Parameterized, preferred: list[str]) -> list[str]:
    return [name for name in preferred if name in instance.param and name != "name"]


def _coerce_group_like(
    value: Any, *, group_cls: type[param.Parameterized]
) -> param.Parameterized:
    if isinstance(value, param.Parameterized):
        return value
    if isinstance(value, dict):
        return group_cls(**dict(value))
    return group_cls()


def _coerce_classdict_items(
    owner: param.Parameterized,
    *,
    parameter_name: str,
    group_cls: type[param.Parameterized],
) -> None:
    raw_items = getattr(owner, parameter_name)
    if isinstance(raw_items, util.AttrDict):
        source_items = raw_items
    else:
        source_items = util.AttrDict(raw_items or {})
    coerced = util.AttrDict(
        {
            str(key): _coerce_group_like(value, group_cls=group_cls)
            for key, value in source_items.items()
        }
    )
    setattr(owner, parameter_name, coerced)


def build_larva_group_widget(
    group_conf: param.Parameterized | dict[str, Any],
    *,
    wrap: bool = True,
) -> object:
    if isinstance(group_conf, param.Parameterized):
        group_obj = group_conf
    else:
        group_obj = reg.gen.LarvaGroup(**dict(group_conf))

    editor = parameterized_editor(
        group_obj,
        parameter_order=_ordered_names(
            group_obj,
            [
                "group_id",
                "model",
                "sample",
                "color",
                "imitation",
                "distribution",
                "odor",
                "life_history",
            ],
        ),
        custom_builders={
            "distribution": (
                lambda inst, name, _parameter: build_distribution_widget(
                    getattr(inst, name)
                )
            )
        },
    )
    if not wrap:
        return editor
    title = str(getattr(group_obj, "group_id", "Larva group")) or "Larva group"
    return family_box(title, editor)


def _build_larva_group_item(group: param.Parameterized, key: str) -> object:
    return family_box(key, build_larva_group_widget(group, wrap=False))


def build_larva_groups_widget(
    owner: param.Parameterized,
    *,
    parameter_name: str = "larva_groups",
    wrap: bool = True,
) -> object:
    parameter = owner.param.objects(instance=False).get(parameter_name)
    if not isinstance(parameter, ClassDict):
        raise TypeError(
            f"{parameter_name!r} must be a ClassDict parameter on {type(owner).__name__}"
        )
    group_cls = parameter.item_type
    if group_cls is None:
        raise ValueError(f"{parameter_name!r} ClassDict is missing item_type")
    _coerce_classdict_items(owner, parameter_name=parameter_name, group_cls=group_cls)
    return classdict_editor(
        owner,
        name=parameter_name,
        parameter=parameter,
        title="Larva groups",
        item_label="larva group",
        build_item_editor=_build_larva_group_item,
        wrap=wrap,
    )
