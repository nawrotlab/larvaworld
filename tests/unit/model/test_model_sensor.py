"""
Unit tests for larvaworld.lib.model.modules.sensor

Tests the Sensor base class and its subclasses (Olfactor, Toucher, Windsensor, Thermosensor, OSNOlfactor).
Focuses on sensory processing logic: perception modes, decay, gain updates, brute_force modulation.

Following FUNDAMENTAL RULE: Read source code first, test actual behavior with real assertions.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from larvaworld.lib.model.modules.sensor import (
    Sensor,
    Olfactor,
    Toucher,
    Windsensor,
    Thermosensor,
    OSNOlfactor,
)
from larvaworld.lib import util


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def brain_stub():
    """Minimal brain stub with locomotor access."""
    brain = Mock()
    brain.locomotor = Mock()
    brain.locomotor.crawler = Mock()
    brain.locomotor.turner = Mock()
    return brain


@pytest.fixture
def sensor_linear(brain_stub):
    """Sensor with linear perception mode."""
    return Sensor(
        brain=brain_stub,
        dt=0.1,
        perception="linear",
        decay_coef=0.1,
        gain_dict=util.AttrDict({"odor1": 1.0, "odor2": 0.5}),
    )


@pytest.fixture
def sensor_log(brain_stub):
    """Sensor with log perception mode."""
    return Sensor(
        brain=brain_stub,
        dt=0.1,
        perception="log",
        decay_coef=0.2,
        gain_dict=util.AttrDict({"odor1": 1.0}),
    )


@pytest.fixture
def sensor_null(brain_stub):
    """Sensor with null perception mode."""
    return Sensor(
        brain=brain_stub,
        dt=0.1,
        perception="null",
        gain_dict=util.AttrDict({"odor1": 1.0}),
    )


@pytest.fixture
def olfactor(brain_stub):
    """Olfactor sensor."""
    return Olfactor(
        brain=brain_stub,
        dt=0.1,
        perception="linear",
        decay_coef=0.1,
        gain_dict=util.AttrDict({"odor1": 1.0, "odor2": 0.5}),
    )


@pytest.fixture
def toucher(brain_stub):
    """Toucher sensor."""
    return Toucher(
        brain=brain_stub,
        dt=0.1,
        perception="linear",
        decay_coef=0.1,
        gain_dict=util.AttrDict({"touch": 1.0}),
    )


# ============================================================================
# Sensor Base Class Tests
# ============================================================================


def test_sensor_initialization(sensor_linear):
    """Test Sensor initializes with correct attributes."""
    assert sensor_linear.dt == 0.1
    assert sensor_linear.perception == "linear"
    assert sensor_linear.decay_coef == 0.1
    assert "odor1" in sensor_linear.X
    assert "odor2" in sensor_linear.X
    assert sensor_linear.X["odor1"] == 0.0
    assert sensor_linear.X["odor2"] == 0.0


def test_sensor_exp_decay_coef_calculation(sensor_linear):
    """Test exponential decay coefficient is calculated from decay_coef and dt."""
    # exp_decay_coef = exp(-dt * decay_coef) = exp(-0.1 * 0.1) = exp(-0.01)
    expected = np.exp(-0.1 * 0.1)
    assert np.isclose(sensor_linear.exp_decay_coef, expected)


def test_sensor_gain_ids_property(sensor_linear):
    """Test gain_ids property returns list of stimulus IDs."""
    gain_ids = sensor_linear.gain_ids
    assert isinstance(gain_ids, list)
    assert "odor1" in gain_ids
    assert "odor2" in gain_ids


def test_sensor_get_gain(sensor_linear):
    """Test get_gain returns current gain dictionary."""
    gains = sensor_linear.get_gain()
    assert gains["odor1"] == 1.0
    assert gains["odor2"] == 0.5


def test_sensor_set_gain(sensor_linear):
    """Test set_gain updates gain for specific stimulus."""
    sensor_linear.set_gain(0.8, "odor1")
    assert sensor_linear.gain["odor1"] == 0.8
    assert sensor_linear.gain["odor2"] == 0.5  # Unchanged


def test_sensor_reset_gain(sensor_linear):
    """Test reset_gain sets gain to initial value from gain_dict."""
    # Modify gain
    sensor_linear.set_gain(0.8, "odor1")
    assert sensor_linear.gain["odor1"] == 0.8

    # Reset to gain_dict value
    sensor_linear.reset_gain("odor1")
    assert sensor_linear.gain["odor1"] == sensor_linear.gain_dict["odor1"]


def test_sensor_reset_all_gains(sensor_linear):
    """Test reset_all_gains resets all gains by reassigning gain_dict."""
    sensor_linear.set_gain(0.3, "odor1")
    sensor_linear.set_gain(0.7, "odor2")

    # reset_all_gains does: self.gain = self.gain_dict (reference assignment)
    sensor_linear.reset_all_gains()

    # After reset, gain should point to gain_dict
    assert sensor_linear.gain is sensor_linear.gain_dict


def test_sensor_add_novel_gain(sensor_linear):
    """Test add_novel_gain adds new stimulus with concentration and gain."""
    sensor_linear.add_novel_gain("odor3", con=0.4, gain=0.6)
    assert "odor3" in sensor_linear.X
    assert sensor_linear.X["odor3"] == 0.4
    assert sensor_linear.gain["odor3"] == 0.6


def test_sensor_compute_single_dx_linear(sensor_linear):
    """Test compute_single_dx in linear mode."""
    # In linear mode: dx = cur - prev if prev != 0 else 0
    dx = sensor_linear.compute_single_dx(cur=0.8, prev=0.5)
    assert np.isclose(dx, 0.3)

    # Edge case: prev == 0 returns 0
    dx_zero = sensor_linear.compute_single_dx(cur=0.8, prev=0.0)
    assert dx_zero == 0.0


def test_sensor_compute_single_dx_log(sensor_log):
    """Test compute_single_dx in log mode."""
    # In log mode: dx = cur / prev - 1 if prev != 0 else 0
    dx = sensor_log.compute_single_dx(cur=0.8, prev=0.5)
    expected = 0.8 / 0.5 - 1  # = 1.6 - 1 = 0.6
    assert np.isclose(dx, expected)

    # Edge case: prev == 0 returns 0
    dx_zero = sensor_log.compute_single_dx(cur=0.8, prev=0.0)
    assert dx_zero == 0.0


def test_sensor_compute_single_dx_null(sensor_null):
    """Test compute_single_dx in null mode returns cur."""
    # In null mode: dx = cur (ignores prev)
    dx = sensor_null.compute_single_dx(cur=0.8, prev=0.5)
    assert dx == 0.8


def test_sensor_compute_dX_updates_dX_dict(sensor_linear):
    """Test compute_dX updates dX dictionary with computed deltas."""
    # Set previous values
    sensor_linear.X["odor1"] = 0.5
    sensor_linear.X["odor2"] = 0.3

    # Compute with new input
    sensor_linear.compute_dX({"odor1": 0.8, "odor2": 0.6})

    # Check dX is updated (linear mode)
    assert np.isclose(sensor_linear.dX["odor1"], 0.3)  # 0.8 - 0.5
    assert np.isclose(sensor_linear.dX["odor2"], 0.3)  # 0.6 - 0.3


def test_sensor_get_dX(sensor_linear):
    """Test get_dX returns dX dictionary."""
    sensor_linear.dX["odor1"] = 0.2
    sensor_linear.dX["odor2"] = -0.1

    dX_dict = sensor_linear.get_dX()
    assert dX_dict["odor1"] == 0.2
    assert dX_dict["odor2"] == -0.1


def test_sensor_update_gain_via_memory(sensor_linear):
    """Test update_gain_via_memory calls memory.step with dX."""
    # Mock memory
    mem = Mock()
    mem.step = Mock(return_value=util.AttrDict({"odor1": 0.9, "odor2": 0.4}))

    sensor_linear.dX["odor1"] = 0.1
    sensor_linear.dX["odor2"] = -0.05

    sensor_linear.update_gain_via_memory(mem)

    # Memory step should be called with dX
    mem.step.assert_called_once()
    call_kwargs = mem.step.call_args[1]
    assert "dx" in call_kwargs

    # Gain should be updated
    assert sensor_linear.gain["odor1"] == 0.9
    assert sensor_linear.gain["odor2"] == 0.4


def test_sensor_update_gain_via_memory_none(sensor_linear):
    """Test update_gain_via_memory does nothing when mem is None."""
    original_gain = sensor_linear.gain.copy()
    sensor_linear.update_gain_via_memory(mem=None)
    assert sensor_linear.gain == original_gain


# ============================================================================
# Olfactor Tests
# ============================================================================


def test_olfactor_initialization(olfactor):
    """Test Olfactor initializes correctly."""
    assert isinstance(olfactor, Sensor)
    assert olfactor.brain is not None


def test_olfactor_first_odor_concentration_property(olfactor):
    """Test first_odor_concentration returns first odor value."""
    olfactor.X["odor1"] = 0.7
    olfactor.X["odor2"] = 0.3
    assert olfactor.first_odor_concentration == 0.7


def test_olfactor_second_odor_concentration_property(olfactor):
    """Test second_odor_concentration returns second odor value."""
    olfactor.X["odor1"] = 0.7
    olfactor.X["odor2"] = 0.3
    assert olfactor.second_odor_concentration == 0.3


def test_olfactor_first_odor_concentration_change_property(olfactor):
    """Test first_odor_concentration_change returns first dX."""
    olfactor.dX["odor1"] = 0.2
    olfactor.dX["odor2"] = -0.1
    assert olfactor.first_odor_concentration_change == 0.2


def test_olfactor_second_odor_concentration_change_property(olfactor):
    """Test second_odor_concentration_change returns second dX."""
    olfactor.dX["odor1"] = 0.2
    olfactor.dX["odor2"] = -0.1
    assert olfactor.second_odor_concentration_change == -0.1


def test_olfactor_affect_locomotion_negative_output_stride_completed(olfactor):
    """Test affect_locomotion interrupts when output < 0 and stride_completed."""
    olfactor.output = -0.6

    L = olfactor.brain.locomotor
    L.stride_completed = True
    L.intermitter = Mock()

    # Mock random to always trigger interrupt (output=-0.6, need random <= 0.6)
    with patch("numpy.random.uniform", return_value=0.5):
        olfactor.affect_locomotion(L)

    # Should call interrupt_locomotion
    L.intermitter.interrupt_locomotion.assert_called_once()


def test_olfactor_affect_locomotion_positive_output(olfactor):
    """Test affect_locomotion does nothing when output >= 0."""
    olfactor.output = 0.5

    L = olfactor.brain.locomotor
    L.stride_completed = True
    L.intermitter = Mock()

    olfactor.affect_locomotion(L)

    # Should NOT call interrupt
    L.intermitter.interrupt_locomotion.assert_not_called()


# ============================================================================
# Toucher Tests
# ============================================================================


def test_toucher_initialization(toucher):
    """Test Toucher initializes correctly."""
    assert isinstance(toucher, Sensor)
    assert toucher.brain is not None


def test_toucher_affect_locomotion_contact_positive(toucher):
    """Test Toucher triggers locomotion when dX == 1 (contact)."""
    toucher.dX["touch"] = 1

    L = toucher.brain.locomotor
    L.intermitter = Mock()

    toucher.affect_locomotion(L)

    # Should call trigger_locomotion
    L.intermitter.trigger_locomotion.assert_called_once()


def test_toucher_affect_locomotion_contact_negative(toucher):
    """Test Toucher interrupts locomotion when dX == -1 (contact loss)."""
    toucher.dX["touch"] = -1

    L = toucher.brain.locomotor
    L.intermitter = Mock()

    toucher.affect_locomotion(L)

    # Should call interrupt_locomotion
    L.intermitter.interrupt_locomotion.assert_called_once()


def test_toucher_affect_locomotion_no_contact(toucher):
    """Test Toucher does nothing when dX is not +1 or -1."""
    toucher.dX["touch"] = 0.5

    L = toucher.brain.locomotor
    L.intermitter = Mock()

    toucher.affect_locomotion(L)

    # Should NOT call trigger or interrupt
    L.intermitter.trigger_locomotion.assert_not_called()
    L.intermitter.interrupt_locomotion.assert_not_called()


# ============================================================================
# Windsensor Tests
# ============================================================================


def test_windsensor_initialization(brain_stub):
    """Test Windsensor initializes with weights."""
    weights = [0.3, 0.7]
    wind = Windsensor(brain=brain_stub, weights=weights, dt=0.1)
    assert wind.weights == weights


# ============================================================================
# Thermosensor Tests
# ============================================================================


def test_thermosensor_initialization(brain_stub):
    """Test Thermosensor initializes correctly."""
    thermo = Thermosensor(brain=brain_stub, dt=0.1, perception="linear")
    assert isinstance(thermo, Sensor)


def test_thermosensor_warm_sensor_input(brain_stub):
    """Test warm_sensor_input property."""
    thermo = Thermosensor(
        brain=brain_stub, dt=0.1, gain_dict=util.AttrDict({"warm": 1.0, "cool": 0.5})
    )
    thermo.X["warm"] = 0.6
    assert thermo.warm_sensor_input == 0.6


def test_thermosensor_cool_sensor_input(brain_stub):
    """Test cool_sensor_input property."""
    thermo = Thermosensor(
        brain=brain_stub, dt=0.1, gain_dict=util.AttrDict({"warm": 1.0, "cool": 0.5})
    )
    thermo.X["cool"] = 0.4
    assert thermo.cool_sensor_input == 0.4


def test_thermosensor_warm_sensor_perception(brain_stub):
    """Test warm_sensor_perception property."""
    thermo = Thermosensor(
        brain=brain_stub, dt=0.1, gain_dict=util.AttrDict({"warm": 1.0, "cool": 0.5})
    )
    thermo.dX["warm"] = 0.2
    assert thermo.warm_sensor_perception == 0.2


def test_thermosensor_cool_sensor_perception(brain_stub):
    """Test cool_sensor_perception property."""
    thermo = Thermosensor(
        brain=brain_stub, dt=0.1, gain_dict=util.AttrDict({"warm": 1.0, "cool": 0.5})
    )
    thermo.dX["cool"] = -0.1
    assert thermo.cool_sensor_perception == -0.1


def test_thermosensor_warm_gain_property(brain_stub):
    """Test warm_gain property."""
    thermo = Thermosensor(
        brain=brain_stub, dt=0.1, gain_dict=util.AttrDict({"warm": 0.8, "cool": 0.5})
    )
    assert thermo.warm_gain == 0.8


def test_thermosensor_cool_gain_property(brain_stub):
    """Test cool_gain property."""
    thermo = Thermosensor(
        brain=brain_stub, dt=0.1, gain_dict=util.AttrDict({"warm": 0.8, "cool": 0.3})
    )
    assert thermo.cool_gain == 0.3


# ============================================================================
# OSNOlfactor Tests
# ============================================================================


def test_osnolfactor_initialization(brain_stub, monkeypatch):
    """Test OSNOlfactor initializes with Brian2 interface."""
    # Mock RemoteBrianModelInterface to avoid network connection
    mock_interface = Mock()
    monkeypatch.setattr(
        "larvaworld.lib.model.modules.sensor.RemoteBrianModelInterface",
        Mock(return_value=mock_interface),
    )

    osn = OSNOlfactor(
        brain=brain_stub,
        dt=0.1,
        response_key="OSN_rate",
        server_host="localhost",
        server_port=5795,
    )

    assert osn.response_key == "OSN_rate"
    assert osn.remote_dt == 100  # default
    assert osn.brian_warmup == 500  # default


def test_osnolfactor_normalized_sigmoid(brain_stub, monkeypatch):
    """Test normalized_sigmoid function."""
    # Mock RemoteBrianModelInterface
    mock_interface = Mock()
    monkeypatch.setattr(
        "larvaworld.lib.model.modules.sensor.RemoteBrianModelInterface",
        Mock(return_value=mock_interface),
    )

    osn = OSNOlfactor(brain=brain_stub, dt=0.1)

    # Test sigmoid: s = 1 / (1 + exp(b * (x - a)))
    # With a=0, b=1, x=0 → s = 1 / (1 + exp(0)) = 1 / 2 = 0.5
    result = osn.normalized_sigmoid(a=0.0, b=1.0, x=0.0)
    assert np.isclose(result, 0.5)

    # With a=0, b=1, x=-10 → s = 1 / (1 + exp(-10)) ≈ 1 (large negative exponent)
    result = osn.normalized_sigmoid(a=0.0, b=1.0, x=-10.0)
    assert result > 0.99


# ============================================================================
# Edge Cases
# ============================================================================


def test_sensor_compute_dX_with_missing_input_key(sensor_linear):
    """Test compute_dX handles missing input keys gracefully."""
    sensor_linear.X["odor1"] = 0.5

    # Input missing odor2 - compute_dX only updates provided keys
    sensor_linear.compute_dX({"odor1": 0.8})

    # Should update odor1
    assert np.isclose(sensor_linear.dX["odor1"], 0.3)

    # odor2 not in input, so not updated
    # X is replaced with input dict, so odor2 no longer in X
    assert "odor2" not in sensor_linear.X


def test_sensor_zero_decay_coef(brain_stub):
    """Test sensor with zero decay coefficient."""
    sensor = Sensor(
        brain=brain_stub,
        dt=0.1,
        decay_coef=0.0,
        gain_dict=util.AttrDict({"odor1": 1.0}),
    )
    # exp(-0.1 * 0.0) = exp(0) = 1.0
    assert sensor.exp_decay_coef == 1.0


def test_sensor_high_decay_coef(brain_stub):
    """Test sensor with high decay coefficient."""
    sensor = Sensor(
        brain=brain_stub,
        dt=0.1,
        decay_coef=10.0,
        gain_dict=util.AttrDict({"odor1": 1.0}),
    )
    # exp(-0.1 * 10) = exp(-1.0) ≈ 0.368
    assert np.isclose(sensor.exp_decay_coef, np.exp(-1.0))


def test_sensor_negative_concentration_log_mode(sensor_log):
    """Test sensor handles edge case in log mode."""
    # In log mode: dx = cur / prev - 1 if prev != 0 else 0
    # With cur=0.0, prev=0.1 → dx = 0.0 / 0.1 - 1 = -1.0
    dx = sensor_log.compute_single_dx(cur=0.0, prev=0.1)
    expected = 0.0 / 0.1 - 1  # = -1.0
    assert np.isclose(dx, expected)
