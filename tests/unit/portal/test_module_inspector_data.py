from __future__ import annotations

import random

import numpy as np
import pytest

from larvaworld.portal.models_architecture import module_inspector_data as data
from larvaworld.portal.models_architecture.module_inspector_models import (
    ModuleInspectorError,
    ModuleTraceResult,
    StimulusSpec,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(0)
    np.random.seed(0)


def test_inspectable_modules_cover_three_kinds() -> None:
    assert data.INSPECTABLE_MODULES == (
        "crawler",
        "turner",
        "feeder",
        "olfactor",
        "toucher",
        "windsensor",
        "thermosensor",
    )
    assert data.module_kind("crawler") == "effector"
    assert data.module_kind("feeder") == "feeder"
    assert data.module_kind("olfactor") == "sensor"


def test_module_modes_exclude_nengo_and_osn() -> None:
    assert "nengo" not in data.module_modes("crawler")
    assert "nengo" not in data.module_modes("turner")
    assert "nengo" not in data.module_modes("feeder")
    assert "osn" not in data.module_modes("olfactor")
    assert "realistic" in data.module_modes("crawler")
    assert "neural" in data.module_modes("turner")
    assert "default" in data.module_modes("feeder")


def test_invalid_module_raises() -> None:
    with pytest.raises(ModuleInspectorError):
        data.module_modes("bogus_module")
    with pytest.raises(ModuleInspectorError):
        data.module_kind("bogus_module")


@pytest.mark.parametrize("module_id", ["crawler", "turner"])
def test_build_each_effector_mode(module_id: str) -> None:
    for mode in data.module_modes(module_id):
        module = data.build_standalone_module(module_id, mode, dt=0.1)
        assert hasattr(module, "step")
        signals = data.detect_signals(module, "effector")
        assert "output" in signals


def test_neural_turner_has_activation_no_phi() -> None:
    module = data.build_standalone_module("turner", "neural", dt=0.1)
    signals = data.detect_signals(module, "effector")
    assert "activation" in signals
    assert "phi" not in signals


def test_run_effector_trace_shape() -> None:
    result = data.run_module_trace("crawler", "realistic", steps=50, dt=0.1, a_in=0.0)
    assert isinstance(result, ModuleTraceResult)
    assert result.kind == "effector"
    assert len(result.dataframe) == 50
    assert list(result.dataframe.columns)[0] == "time"
    for sig in result.signals:
        assert sig in result.dataframe.columns
    assert result.dataframe["time"].iloc[1] == pytest.approx(0.1)


def test_run_trace_validates_args() -> None:
    with pytest.raises(ModuleInspectorError):
        data.run_module_trace("crawler", "realistic", steps=0)
    with pytest.raises(ModuleInspectorError):
        data.run_module_trace("crawler", "realistic", dt=0.0)


def test_feeder_builds_and_traces_phase() -> None:
    module = data.build_standalone_module("feeder", "default", dt=0.1)
    signals = data.detect_signals(module, "feeder")
    assert "phi" in signals
    assert "complete_iteration" in signals
    result = data.run_module_trace("feeder", "default", steps=60, dt=0.1)
    assert result.kind == "feeder"
    assert "phi" in result.dataframe.columns
    # Started effector + oscillation -> phase must change over time.
    assert result.dataframe["phi"].nunique() > 1


@pytest.mark.parametrize(
    "module_id", ["olfactor", "toucher", "windsensor", "thermosensor"]
)
def test_each_sensor_builds_standalone(module_id: str) -> None:
    module = data.build_standalone_module(module_id, "default", dt=0.1)
    assert hasattr(module, "step")


@pytest.mark.parametrize(
    "module_id", ["olfactor", "toucher", "windsensor", "thermosensor"]
)
def test_sensor_sinusoid_produces_response(module_id: str) -> None:
    stim = StimulusSpec(
        waveform="sinusoid", baseline=1.0, amplitude=0.5, frequency=1.0, onset=0.0
    )
    result = data.run_module_trace(
        module_id, "default", steps=80, dt=0.1, stimulus=stim
    )
    assert result.kind == "sensor"
    assert list(result.dataframe.columns) == ["time", "stimulus", "output"]
    # A time-varying stimulus must produce a varying output (not flat).
    assert result.dataframe["output"].nunique() > 1


def test_stimulus_series_shapes() -> None:
    sinus = StimulusSpec(
        waveform="sinusoid", baseline=1.0, amplitude=0.5, frequency=1.0, onset=0.0
    )
    series = data.stimulus_series(sinus, steps=10, dt=0.1)
    assert len(series) == 10
    assert series[0] == pytest.approx(1.0)

    step = StimulusSpec(
        waveform="step", baseline=1.0, amplitude=0.5, frequency=1.0, onset=0.5
    )
    step_series = data.stimulus_series(step, steps=10, dt=0.1)
    assert step_series[0] == pytest.approx(1.0)
    assert step_series[-1] == pytest.approx(1.5)


def test_stimulus_to_input_keys() -> None:
    assert data.stimulus_to_input("olfactor", 0.7, 1.0) == {"odor": 0.7}
    assert data.stimulus_to_input("toucher", 0.7, 1.0) == {"touch": 0.7}
    assert data.stimulus_to_input("windsensor", 0.7, 1.0) == {"windsensor": 0.7}
    assert data.stimulus_to_input("thermosensor", 0.7, 1.0) == {
        "warm": 0.7,
        "cool": 1.0,
    }
