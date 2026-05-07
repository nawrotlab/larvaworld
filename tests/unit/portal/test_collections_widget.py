from __future__ import annotations

import panel as pn

from larvaworld.lib.reg.generators import ExpConf
from larvaworld.portal.config_widgets import build_collections_widget
from larvaworld.portal.simulation.parameter_resolution import (
    resolve_base_experiment_parameters,
)


def test_collections_widget_builds_for_exp_conf_owner() -> None:
    owner = ExpConf(**dict(resolve_base_experiment_parameters("dish")))
    widget = build_collections_widget(owner)
    assert pn.Column(widget).get_root() is not None


def test_collections_widget_uses_core_options_and_nested_conf_roundtrip() -> None:
    owner = ExpConf(**dict(resolve_base_experiment_parameters("dish")))
    widget = build_collections_widget(owner, wrap=False)

    multichoice = next(iter(widget.select(pn.widgets.MultiChoice)))
    core_options = list(owner.param["collections"].objects)

    assert list(multichoice.options) == core_options

    selected = core_options[:2] if len(core_options) > 1 else core_options[:1]
    multichoice.value = selected

    assert list(owner.collections) == selected
    assert list(owner.nestedConf["collections"]) == selected
