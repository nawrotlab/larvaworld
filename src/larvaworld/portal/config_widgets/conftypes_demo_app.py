from __future__ import annotations

from typing import Any

import panel as pn
import param

from larvaworld import CONFTYPES
from larvaworld.lib import reg
from larvaworld.portal.config_widgets.conftype_widget import build_conftype_widget


def _unsupported_message(conftype: str, reason: str) -> pn.Column:
    return pn.Column(
        pn.pane.Markdown(f"### {conftype}", margin=(0, 0, 8, 0)),
        pn.pane.HTML(
            (
                '<div style="font-size:13px;line-height:1.5;color:#52606d;">'
                f"{reason}"
                "</div>"
            ),
            margin=0,
        ),
        sizing_mode="stretch_width",
    )


def conftypes_demo_app() -> pn.Column:
    tabs: list[tuple[str, Any]] = []

    for conftype in CONFTYPES:
        conf_type = reg.conf[conftype]
        config_cls = conf_type.conf_class
        if not isinstance(config_cls, type) or not issubclass(
            config_cls, param.Parameterized
        ):
            tabs.append(
                (
                    conftype,
                    _unsupported_message(
                        conftype,
                        (
                            "This conftype does not currently expose a "
                            "parameterized configuration class, so the generic "
                            "editor cannot render it without an adapter."
                        ),
                    ),
                )
            )
            continue
        try:
            widget = build_conftype_widget(
                config_cls,
                conftype=conftype,
                title=f"{conftype} Configuration Editor",
            )
        except Exception as exc:
            widget = _unsupported_message(
                conftype,
                f"Could not build the generic editor: {exc}",
            )
        tabs.append((conftype, widget))

    return pn.Column(
        pn.pane.Markdown(
            "## Conftype Widget Demo",
            margin=(0, 0, 8, 0),
        ),
        pn.pane.HTML(
            (
                '<div style="font-size:13px;line-height:1.5;color:#52606d;">'
                "This dev/test view renders one generic configuration editor per "
                "registered conftype, using the shared helper instead of the "
                "special-purpose Environment Builder preset UI."
                "</div>"
            ),
            margin=(0, 0, 12, 0),
        ),
        pn.Tabs(*tabs, dynamic=True, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )
