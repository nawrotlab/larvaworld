from __future__ import annotations

import panel as pn
import pytest

from larvaworld.lib.reg.generators import ExpConf
from larvaworld.portal.config_widgets import build_sim_ops_widget
from larvaworld.portal.simulation.parameter_resolution import (
    resolve_base_experiment_parameters,
)


def _find_widget_any(
    viewable: pn.viewable.Viewable,
    candidates: tuple[str, ...],
    widget_type: type[pn.widgets.Widget]
    | tuple[type[pn.widgets.Widget], ...] = pn.widgets.Widget,
):
    normalized_candidates = {candidate.strip().lower() for candidate in candidates}
    for widget in viewable.select(widget_type):
        name = str(getattr(widget, "name", "")).strip().lower()
        if name in normalized_candidates:
            return widget
    raise AssertionError(f"Could not find any widget in {candidates!r}.")


def test_sim_ops_widget_builds_for_exp_conf_owner() -> None:
    owner = ExpConf(**dict(resolve_base_experiment_parameters("dish")))
    widget = build_sim_ops_widget(owner)
    assert pn.Column(widget).get_root() is not None


def test_sim_ops_widget_preserves_core_sync_and_serialization() -> None:
    owner = ExpConf(**dict(resolve_base_experiment_parameters("dish")))
    widget = build_sim_ops_widget(owner, wrap=False)

    duration = _find_widget_any(widget, ("Duration",), pn.widgets.Widget)
    dt = _find_widget_any(widget, ("Timestep", "Dt"), pn.widgets.Widget)
    box2d = _find_widget_any(widget, ("Box2d", "Box2D"), pn.widgets.Checkbox)
    collisions = _find_widget_any(
        widget,
        ("Larva collisions", "Larva Collisions"),
        pn.widgets.Checkbox,
    )

    duration.value = 2.0
    dt.value = 0.2
    box2d.value = True
    collisions.value = False

    expected_nsteps = int(owner.duration * 60 / owner.dt)
    assert owner.Nsteps == expected_nsteps
    assert owner.fr == pytest.approx(1 / owner.dt)
    assert owner.Box2D is True
    assert owner.larva_collisions is False

    nested = owner.nestedConf
    assert nested["duration"] == pytest.approx(2.0)
    assert nested["Nsteps"] == expected_nsteps
    assert nested["dt"] == pytest.approx(0.2)
    assert nested["fr"] == pytest.approx(5.0)
    assert nested["Box2D"] is True
    assert nested["larva_collisions"] is False
