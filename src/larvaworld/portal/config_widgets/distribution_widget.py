from __future__ import annotations

import param

from .widget_base import family_box, numeric_tuple_param_control, param_control

__all__ = ["build_distribution_widget"]


def build_distribution_widget(
    distribution: param.Parameterized,
    *,
    title: str = "Distribution",
) -> object:
    children = [
        param_control(distribution, parameter_name="shape"),
        param_control(distribution, parameter_name="mode"),
        param_control(distribution, parameter_name="N"),
        numeric_tuple_param_control(
            distribution,
            parameter_name="loc",
            labels=("Distribution center X", "Distribution center Y"),
            numeric_type=float,
            doc=getattr(distribution.param["loc"], "doc", None),
            step=0.001,
        ),
        numeric_tuple_param_control(
            distribution,
            parameter_name="scale",
            labels=("Distribution scale X", "Distribution scale Y"),
            numeric_type=float,
            doc=getattr(distribution.param["scale"], "doc", None),
            step=0.001,
        ),
    ]
    if "orientation_range" in distribution.param:
        children.append(
            numeric_tuple_param_control(
                distribution,
                parameter_name="orientation_range",
                labels=("Orientation min", "Orientation max"),
                numeric_type=float,
                doc=getattr(distribution.param["orientation_range"], "doc", None),
                step=1.0,
            )
        )
    return family_box(title, *children)
