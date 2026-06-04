from __future__ import annotations

import random

import numpy as np
import pytest

from larvaworld.portal.models_architecture import module_inspector_data as data
from larvaworld.portal.models_architecture.module_inspector_models import (
    ModuleInspectorError,
    ModuleTraceResult,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(0)
    np.random.seed(0)


def test_inspectable_modules_are_crawler_and_turner() -> None:
    assert data.INSPECTABLE_MODULES == ("crawler", "turner")


def test_module_modes_exclude_nengo() -> None:
    assert "nengo" not in data.module_modes("crawler")
    assert "nengo" not in data.module_modes("turner")
    assert "realistic" in data.module_modes("crawler")
    assert "neural" in data.module_modes("turner")


def test_invalid_module_raises() -> None:
    with pytest.raises(ModuleInspectorError):
        data.module_modes("feeder")


@pytest.mark.parametrize("module_id", ["crawler", "turner"])
def test_build_each_mode(module_id: str) -> None:
    for mode in data.module_modes(module_id):
        module = data.build_standalone_module(module_id, mode, dt=0.1)
        assert hasattr(module, "step")
        signals = data.detect_signals(module)
        assert "output" in signals


def test_neural_turner_has_activation_no_phi() -> None:
    module = data.build_standalone_module("turner", "neural", dt=0.1)
    signals = data.detect_signals(module)
    assert "activation" in signals
    assert "phi" not in signals


def test_run_module_trace_shape() -> None:
    result = data.run_module_trace("crawler", "realistic", steps=50, dt=0.1, a_in=0.0)
    assert isinstance(result, ModuleTraceResult)
    assert len(result.dataframe) == 50
    assert list(result.dataframe.columns)[0] == "time"
    for sig in result.signals:
        assert sig in result.dataframe.columns
    assert result.dataframe["time"].iloc[1] == pytest.approx(0.1)


def test_run_module_trace_validates_args() -> None:
    with pytest.raises(ModuleInspectorError):
        data.run_module_trace("crawler", "realistic", steps=0)
    with pytest.raises(ModuleInspectorError):
        data.run_module_trace("crawler", "realistic", dt=0.0)
