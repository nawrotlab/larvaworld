from __future__ import annotations

import panel as pn
import param

from .widget_base import classattr_section, collapsible_family_box, parameterized_editor

__all__ = ["build_enrichment_widget", "build_preprocess_conf_widget"]


def _ordered_names(instance: param.Parameterized, preferred: list[str]) -> list[str]:
    return [name for name in preferred if name in instance.param and name != "name"]


def build_preprocess_conf_widget(preprocess_conf: param.Parameterized) -> object:
    return parameterized_editor(
        preprocess_conf,
        parameter_order=_ordered_names(
            preprocess_conf,
            [
                "rescale_by",
                "filter_f",
                "transposition",
                "interpolate_nans",
                "drop_collisions",
            ],
        ),
    )


def build_enrichment_widget(
    enrichment_conf: param.Parameterized,
    *,
    wrap: bool = True,
) -> object:
    children = [
        collapsible_family_box(
            "Preprocessing",
            classattr_section(
                enrichment_conf,
                name="pre_kws",
                parameter=enrichment_conf.param["pre_kws"],
                title="Preprocessing",
                show_title=False,
                build_editor=build_preprocess_conf_widget,
                box_css_classes=["lw-import-datasets-config-subfamily"],
                title_css_classes=["lw-import-datasets-config-subfamily-title"],
            ),
            css_classes=[
                "lw-import-datasets-config-subfamily-card",
                "lw-import-datasets-config-compact-card",
            ],
        ),
        collapsible_family_box(
            "Processing",
            parameterized_editor(
                enrichment_conf,
                parameter_order=_ordered_names(
                    enrichment_conf,
                    [
                        "proc_keys",
                        "dsp_starts",
                        "dsp_stops",
                        "tor_durs",
                    ],
                ),
                exclude={"pre_kws", "anot_keys", "recompute", "mode"},
            ),
            css_classes=[
                "lw-import-datasets-config-subfamily-card",
                "lw-import-datasets-config-compact-card",
            ],
        ),
        collapsible_family_box(
            "Annotations",
            parameterized_editor(
                enrichment_conf,
                parameter_order=_ordered_names(
                    enrichment_conf,
                    [
                        "anot_keys",
                        "recompute",
                        "mode",
                    ],
                ),
                exclude={"pre_kws", "proc_keys", "dsp_starts", "dsp_stops", "tor_durs"},
            ),
            css_classes=[
                "lw-import-datasets-config-subfamily-card",
                "lw-import-datasets-config-compact-card",
            ],
        ),
    ]
    if not wrap:
        return pn.Column(*children, sizing_mode="stretch_width", margin=0)
    return collapsible_family_box("Enrichment", *children)
