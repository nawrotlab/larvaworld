from __future__ import annotations

import random

import numpy as np
import panel as pn
import pytest

from larvaworld.portal.models_architecture.module_inspector_app import (
    _ModuleInspectorController,
    module_inspector_app,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(0)
    np.random.seed(0)


def test_controller_builds_and_recomputes() -> None:
    controller = _ModuleInspectorController()
    assert controller.module_select.value == "crawler"
    assert controller.plot_view.objects
    # effector default: A_in visible, stimulus hidden.
    assert controller.a_in_slider.visible is True
    assert controller.waveform_select.visible is False


def test_mode_change_rebuilds_editor() -> None:
    controller = _ModuleInspectorController()
    controller.module_select.value = "turner"
    controller.mode_select.value = "neural"
    assert controller._mode == "neural"


def test_feeder_hides_a_in_and_stimulus() -> None:
    controller = _ModuleInspectorController()
    controller.module_select.value = "feeder"
    assert controller._kind() == "feeder"
    assert controller.a_in_slider.visible is False
    assert controller.waveform_select.visible is False
    assert set(controller.signal_checkbox.options) == {"phi", "complete_iteration"}


def test_sensor_shows_stimulus_controls() -> None:
    controller = _ModuleInspectorController()
    controller.module_select.value = "olfactor"
    assert controller._kind() == "sensor"
    assert controller.a_in_slider.visible is False
    assert controller.waveform_select.visible is True
    assert set(controller.signal_checkbox.options) == {"stimulus", "output"}
    assert controller.plot_view.objects


def test_app_factory_returns_template() -> None:
    app = module_inspector_app()
    assert isinstance(app, pn.template.MaterialTemplate)
