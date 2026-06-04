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


def test_mode_change_rebuilds_editor() -> None:
    controller = _ModuleInspectorController()
    controller.module_select.value = "turner"
    controller.mode_select.value = "neural"
    assert controller._mode == "neural"


def test_app_factory_returns_template() -> None:
    app = module_inspector_app()
    assert isinstance(app, pn.template.MaterialTemplate)
