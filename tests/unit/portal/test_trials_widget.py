from __future__ import annotations

import panel as pn
import pytest

from larvaworld.lib import reg, util
from larvaworld.lib.reg.generators import ExpConf
from larvaworld.portal.config_widgets import build_trials_widget
from larvaworld.portal.simulation.parameter_resolution import (
    resolve_base_experiment_parameters,
)


def _find_widget(
    viewable: pn.viewable.Viewable,
    name: str,
    widget_type: type[pn.widgets.Widget]
    | tuple[type[pn.widgets.Widget], ...] = pn.widgets.Widget,
):
    for widget in viewable.select(widget_type):
        if getattr(widget, "name", None) == name:
            return widget
    raise AssertionError(f"Could not find widget {name!r}.")


def _coerce_epochs(raw_epochs):
    if isinstance(raw_epochs, dict):
        values = list(raw_epochs.values())
    else:
        values = list(raw_epochs or [])
    return [reg.gen.Epoch(**dict(payload)) for payload in values]


def test_trials_widget_builds_for_exp_conf_owner() -> None:
    owner = ExpConf(**dict(resolve_base_experiment_parameters("dish")))
    widget = build_trials_widget(owner)
    assert pn.Column(widget).get_root() is not None


def test_trials_widget_missing_epochs_does_not_create_default_epoch() -> None:
    owner = ExpConf(**dict(resolve_base_experiment_parameters("dish")))
    owner.trials = util.AttrDict({"custom_flag": True})

    widget = build_trials_widget(owner, wrap=False)
    selected = _find_widget(widget, "Selected epoch", pn.widgets.Select)

    assert selected.value == ""
    assert "epochs" not in owner.trials
    assert owner.trials["custom_flag"] is True


def test_trials_widget_add_edit_delete_and_preserve_unknown_keys() -> None:
    owner = ExpConf(**dict(resolve_base_experiment_parameters("dish")))
    owner.trials = util.AttrDict({"custom_key": "keep-me"})
    widget = build_trials_widget(owner, wrap=False)

    add_epoch = _find_widget(widget, "Add epoch", pn.widgets.Button)
    delete_epoch = _find_widget(widget, "Delete epoch", pn.widgets.Button)
    selected = _find_widget(widget, "Selected epoch", pn.widgets.Select)

    add_epoch.clicks = add_epoch.clicks + 1

    assert "epochs" in owner.trials
    assert len(owner.trials["epochs"]) == 1
    assert owner.trials["custom_key"] == "keep-me"
    assert selected.value == "0"

    age_range = _find_widget(widget, "Age range", pn.widgets.LiteralInput)
    age_range.value = (1.0, 4.0)

    serialized = owner.trials["epochs"][0]
    assert tuple(serialized["age_range"]) == pytest.approx((1.0, 4.0))
    epoch = reg.gen.Epoch(**dict(serialized))
    assert tuple(epoch.age_range) == pytest.approx((1.0, 4.0))

    delete_epoch.clicks = delete_epoch.clicks + 1

    assert len(owner.trials["epochs"]) == 0
    assert owner.trials["custom_key"] == "keep-me"


def test_trials_widget_roundtrips_registry_trial_payload() -> None:
    owner = ExpConf(**dict(resolve_base_experiment_parameters("dish")))
    owner.trials = reg.conf.Trial.getID("odor_preference").get_copy()

    widget = build_trials_widget(owner, wrap=False)
    assert pn.Column(widget).get_root() is not None

    epochs = _coerce_epochs(owner.trials.get("epochs"))
    assert len(epochs) > 0
    for epoch in epochs:
        payload = util.AttrDict(epoch.nestedConf)
        roundtrip = reg.gen.Epoch(**dict(payload))
        assert roundtrip.age_range[0] == pytest.approx(epoch.age_range[0])
