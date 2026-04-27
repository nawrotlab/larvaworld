from __future__ import annotations

import param

from .widget_base import (
    classattr_section,
    numeric_tuple_param_control,
    parameterized_editor,
)

__all__ = ["build_odorscape_widget"]


def _ordered_names(instance: param.Parameterized, preferred: list[str]) -> list[str]:
    return [name for name in preferred if name in instance.param and name != "name"]


def _build_odorscape_editor(layer: param.Parameterized) -> object:
    custom_builders = {}
    if "grid_dims" in layer.param:
        custom_builders["grid_dims"] = (
            lambda inst, name, _parameter: numeric_tuple_param_control(
                inst,
                parameter_name=name,
                labels=("Odorscape grid X", "Odorscape grid Y"),
                numeric_type=int,
                doc=getattr(inst.param[name], "doc", None),
                step=1,
            )
        )
    if "gaussian_sigma" in layer.param:
        custom_builders["gaussian_sigma"] = (
            lambda inst, name, _parameter: numeric_tuple_param_control(
                inst,
                parameter_name=name,
                labels=("Diffusion sigma X", "Diffusion sigma Y"),
                numeric_type=float,
                doc=getattr(inst.param[name], "doc", None),
                step=0.05,
            )
        )
    return parameterized_editor(
        layer,
        parameter_order=_ordered_names(
            layer,
            [
                "unique_id",
                "color",
                "grid_dims",
                "initial_value",
                "fixed_max",
                "evap_const",
                "gaussian_sigma",
            ],
        ),
        exclude={"odorscape"},
        custom_builders=custom_builders,
    )


def build_odorscape_widget(env_conf: param.Parameterized) -> object:
    return classattr_section(
        env_conf,
        name="odorscape",
        parameter=env_conf.param["odorscape"],
        title="Odorscape",
        build_editor=_build_odorscape_editor,
        controls_layout="column",
        box_css_classes=["lw-import-datasets-config-subfamily"],
        title_css_classes=["lw-import-datasets-config-subfamily-title"],
        enable_control="switch",
    )
