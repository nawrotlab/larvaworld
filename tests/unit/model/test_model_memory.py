"""
Unit tests for lib/model/modules/memory.py

Tests Q-learning (RLmemory), olfaction/touch specializations,
and mushroom body (RemoteBrianModelMemory) without requiring
real Brian2 server connection.

Philosophy: Test the LOGIC, not the physics.
"""

import pytest
import numpy as np
from types import SimpleNamespace

from larvaworld.lib.model.modules.memory import (
    Memory,
    RLmemory,
    RLOlfMemory,
    RLTouchMemory,
    RemoteBrianModelMemory,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def brain_stub():
    """Minimal brain stub with agent reference."""
    agent = SimpleNamespace(
        unique_id="test_agent", model=SimpleNamespace(id="test_model", dt=0.1)
    )
    return SimpleNamespace(agent=agent)


@pytest.fixture
def rl_memory_single(brain_stub):
    """RLmemory instance with single stimulus ID."""
    return RLmemory(
        brain=brain_stub,
        gain={"odor1": 0.0},
        dt=0.1,
        alpha=0.1,
        gamma=0.6,
        epsilon=0.2,
        state_spacePerSide=2,
        gain_space=[-100.0, 0.0, 100.0],
        update_dt=1.0,
        train_dur=10.0,
    )


@pytest.fixture
def rl_memory_dual(brain_stub):
    """RLmemory instance with two stimulus IDs."""
    return RLmemory(
        brain=brain_stub,
        gain={"odor_A": 0.0, "odor_B": 0.0},
        dt=0.1,
        alpha=0.05,
        gamma=0.8,
        epsilon=0.15,
        state_spacePerSide=1,
        gain_space=[-50.0, 50.0],
        update_dt=0.5,
        train_dur=5.0,
    )


# ============================================================================
# Memory Base Class Tests
# ============================================================================


def test_memory_initialization(brain_stub):
    """Test Memory base class initialization."""
    mem = Memory(
        brain=brain_stub, gain={"stim1": 1.0}, mode="RL", modality="olfaction", dt=0.1
    )

    assert mem.brain is brain_stub
    assert mem.gain == {"stim1": 1.0}
    assert mem.rewardSum == 0
    assert mem.mode == "RL"
    assert mem.modality == "olfaction"


def test_memory_step_increments_reward(brain_stub):
    """Test Memory.step() accumulates reward correctly."""
    mem = Memory(brain=brain_stub, gain={"stim1": 0.0}, dt=0.1)
    mem.active = True  # Set after initialization

    # Step with reward
    mem.step(reward=True)
    assert mem.rewardSum == 1 - 0.01  # int(True) - 0.01

    # Step without reward
    mem.step(reward=False)
    assert mem.rewardSum == pytest.approx(1 - 0.01 + 0 - 0.01)


def test_memory_step_returns_gain(brain_stub):
    """Test Memory.step() returns gain dict."""
    mem = Memory(brain=brain_stub, gain={"stim1": 5.0}, dt=0.1)

    result = mem.step(reward=False)
    assert result == {"stim1": 5.0}


# ============================================================================
# RLmemory Q-Learning Tests
# ============================================================================


def test_rl_memory_initialization(rl_memory_single):
    """Test RLmemory initializes Q-table and state space."""
    mem = rl_memory_single

    # Check Q-table shape
    # state_spacePerSide=2 → 2*2+1=5 states per stimulus
    # 1 stimulus → 5 states total
    # gain_space has 3 values → 3 actions
    assert mem.q_table.shape == (5, 3)

    # Q-table should be initialized to zeros
    assert np.all(mem.q_table == 0.0)

    # Check state space
    assert mem.state_space.shape[0] == 5  # 5 states

    # Check actions
    assert len(mem.actions) == 3  # 3 gain values
    assert mem.actions == [(-100.0,), (0.0,), (100.0,)]


def test_rl_memory_dual_stimulus(rl_memory_dual):
    """Test RLmemory with two stimuli."""
    mem = rl_memory_dual

    # state_spacePerSide=1 → 2*1+1=3 states per stimulus
    # 2 stimuli → 3^2=9 state combinations
    # gain_space has 2 values → 2^2=4 action combinations
    assert mem.q_table.shape == (9, 4)
    assert mem.state_space.shape[0] == 9
    assert len(mem.actions) == 4
    assert mem.actions == [(-50.0, -50.0), (-50.0, 50.0), (50.0, -50.0), (50.0, 50.0)]


def test_rl_memory_state_collapse_single():
    """Test state discretization for single stimulus."""
    mem = RLmemory(
        brain=None,
        gain={"odor1": 0.0},
        dt=0.1,
        Delta=0.1,
        state_spacePerSide=2,
        gain_space=[0.0],
        update_dt=1.0,
    )

    # Test zero input → middle state (index 2 for k=2)
    state = mem.state_collapse({"odor1": 0.0})
    assert state == 2  # Middle state

    # Test positive input
    state = mem.state_collapse({"odor1": 0.15})  # > Delta
    assert state > 2  # Higher state

    # Test negative input
    state = mem.state_collapse({"odor1": -0.15})  # < -Delta
    assert state < 2  # Lower state


def test_rl_memory_update_q_table():
    """Test Q-table updates with Q-learning formula."""
    mem = RLmemory(
        brain=None,
        gain={"odor1": 0.0},
        dt=0.1,
        alpha=0.1,
        gamma=0.5,
        state_spacePerSide=1,
        gain_space=[-1.0, 0.0, 1.0],
        update_dt=1.0,
    )

    # Set initial Q-value
    mem.q_table[1, 0] = 10.0  # lastState=1, lastAction=0
    mem.lastState = 1
    mem.lastAction = 0

    # Set Q-values for new state
    mem.q_table[2, :] = [5.0, 15.0, 8.0]  # Max is 15.0

    # Update: Q[s,a] = (1-α)*Q[s,a] + α*(r + γ*max(Q[s']))
    reward = 2.0
    mem.update_q_table(state=2, reward=reward)

    expected = (1 - 0.1) * 10.0 + 0.1 * (2.0 + 0.5 * 15.0)
    assert mem.q_table[1, 0] == pytest.approx(expected)
    assert mem.lastState == 2  # Updated


def test_rl_memory_epsilon_greedy_exploration(rl_memory_single):
    """Test epsilon-greedy action selection (exploration vs exploitation)."""
    mem = rl_memory_single

    # Set Q-values so action 2 is clearly best
    mem.q_table[2, :] = [0.0, 0.0, 100.0]  # Action 2 has highest Q

    # With epsilon=0.2, we should get ~20% random actions
    # Run multiple times to check randomness
    np.random.seed(42)
    actions = []
    for _ in range(50):
        gain = mem.update_ext_gain({"odor1": 0.0}, dx={"odor1": 0.0}, randomize=True)
        # Find which action was selected
        action_idx = mem.lastAction
        actions.append(action_idx)

    # Should have some action 2 (exploitation) and some others (exploration)
    assert 2 in actions  # Exploitation happened
    assert len(set(actions)) > 1  # Exploration happened (multiple actions selected)


def test_rl_memory_best_gain_property(rl_memory_single):
    """Test best_gain property returns highest average Q-value action."""
    mem = rl_memory_single

    # Set Q-values: action 1 has highest average
    mem.q_table[:, 0] = [1, 2, 3, 4, 5]  # avg = 3.0
    mem.q_table[:, 1] = [10, 10, 10, 10, 10]  # avg = 10.0 (best!)
    mem.q_table[:, 2] = [2, 3, 4, 5, 6]  # avg = 4.0

    best_gain = mem.best_gain
    assert best_gain == {"odor1": 0.0}  # Action 1 corresponds to gain_space[1]=0.0


def test_rl_memory_learning_on_property(rl_memory_single):
    """Test learning_on property checks training duration."""
    mem = rl_memory_single

    # Initially, learning should be on (if active)
    mem.active = True
    mem.total_ticks = 0  # total_t = total_ticks * dt
    assert mem.learning_on is True

    # After train_dur, learning should be off
    # total_t = dt * total_ticks, need total_t > train_dur * 60
    mem.total_ticks = int((mem.train_dur * 60 + 1) / mem.dt)
    assert mem.learning_on is False

    # If not active, learning is off
    mem.total_ticks = 0
    mem.active = False
    assert mem.learning_on is False


def test_rl_memory_condition_iterator():
    """Test condition checks if iterator >= Niters."""
    mem = RLmemory(
        brain=None,
        gain={"odor1": 0.0},
        dt=0.1,
        update_dt=1.0,  # 1 second
        state_spacePerSide=1,
        gain_space=[0.0],
    )

    # Niters = update_dt * 60 / dt = 1.0 * 60 / 0.1 = 600
    assert mem.Niters == 600

    # Initially, iterator is set to Niters
    mem.iterator = mem.Niters
    assert mem.condition({}) is True

    # Below Niters, condition is False
    mem.iterator = mem.Niters - 1
    assert mem.condition({}) is False


def test_rl_memory_update_gain_learning_mode(rl_memory_single):
    """Test update_gain during learning phase."""
    mem = rl_memory_single
    mem.active = True
    mem.total_ticks = 0  # learning_on = True
    mem.iterator = mem.Niters  # Condition will be True

    initial_gain = mem.gain.copy()

    # Update gain (should trigger update_ext_gain)
    mem.update_gain(dx={"odor1": 0.1})

    # Gain should have changed
    assert mem.gain != initial_gain

    # Iterator should reset
    assert mem.iterator == 0

    # rewardSum should reset
    assert mem.rewardSum == 0


def test_rl_memory_update_gain_exploitation_mode(rl_memory_single):
    """Test update_gain after training (exploitation only)."""
    mem = rl_memory_single
    mem.active = False  # learning_on = False
    mem.state_specific_best = False

    # Set Q-values so we know best action
    mem.q_table[:, 1] = [100, 100, 100, 100, 100]  # Action 1 is best (gain=0.0)

    mem.update_gain(dx={"odor1": 0.0})

    # Should use best_gain (not state-specific)
    assert mem.gain == mem.best_gain


# ============================================================================
# RLOlfMemory Tests
# ============================================================================


def test_rl_olf_memory_modality():
    """Test RLOlfMemory sets modality to olfaction."""
    mem = RLOlfMemory(
        brain=None,
        gain={"odor_A": 0.0, "odor_B": 0.0},
        dt=0.1,
        state_spacePerSide=1,
        gain_space=[0.0],
    )

    assert mem.modality == "olfaction"


def test_rl_olf_memory_best_gain_properties():
    """Test first_odor_best_gain and second_odor_best_gain properties."""
    mem = RLOlfMemory(
        brain=None,
        gain={"odor_A": 0.0, "odor_B": 0.0},
        dt=0.1,
        state_spacePerSide=1,
        gain_space=[-10.0, 10.0],
    )

    # Q-table shape: 2 stimuli, state_spacePerSide=1 → 3^2=9 states, 2^2=4 actions
    assert mem.q_table.shape == (9, 4)

    # Set Q-values: make action 3 have highest average
    for state in range(9):
        mem.q_table[state, :] = [1.0, 2.0, 3.0, 10.0]  # Action 3 is best

    # Actions: [(-10, -10), (-10, 10), (10, -10), (10, 10)]
    # Best average: action 3 → (10, 10)

    # first_odor_best_gain should be 10.0
    # second_odor_best_gain should be 10.0
    first = mem.first_odor_best_gain
    second = mem.second_odor_best_gain

    assert first == 10.0
    assert second == 10.0


# ============================================================================
# RLTouchMemory Tests
# ============================================================================


def test_rl_touch_memory_modality():
    """Test RLTouchMemory sets modality to touch."""
    mem = RLTouchMemory(
        brain=None,
        gain={"sensor_0": 0.0},
        dt=0.1,
        state_spacePerSide=1,
        gain_space=[0.0],
    )

    assert mem.modality == "touch"


def test_rl_touch_memory_condition_contact_positive():
    """Test RLTouchMemory condition triggers on +1 contact."""
    mem = RLTouchMemory(
        brain=None,
        gain={"sensor_0": 0.0},
        dt=0.1,
        state_spacePerSide=1,
        gain_space=[0.0],
        update_dt=1.0,
    )

    mem.iterator = 10
    mem.rewardSum = 0

    # Contact detected (+1)
    result = mem.condition({"sensor_0": 1})

    assert result is True
    assert mem.rewardSum == 1 / 10  # Reward is 1/iterator


def test_rl_touch_memory_condition_contact_negative():
    """Test RLTouchMemory condition triggers on -1 contact."""
    mem = RLTouchMemory(
        brain=None,
        gain={"sensor_0": 0.0},
        dt=0.1,
        state_spacePerSide=1,
        gain_space=[0.0],
        update_dt=1.0,
    )

    mem.iterator = 5
    mem.rewardSum = 0

    # Contact detected (-1)
    result = mem.condition({"sensor_0": -1})

    assert result is True
    assert mem.rewardSum == 5  # Reward is iterator


def test_rl_touch_memory_condition_no_contact():
    """Test RLTouchMemory condition returns False without contact."""
    mem = RLTouchMemory(
        brain=None,
        gain={"sensor_0": 0.0},
        dt=0.1,
        state_spacePerSide=1,
        gain_space=[0.0],
    )

    # No contact (0)
    result = mem.condition({"sensor_0": 0})
    assert result is False


# ============================================================================
# RemoteBrianModelMemory Tests (without real server)
# ============================================================================


def test_remote_brian_memory_initialization(brain_stub):
    """Test RemoteBrianModelMemory initialization."""
    mem = RemoteBrianModelMemory(
        brain=brain_stub,
        gain={"Odor": 0.0},
        dt=0.1,
        G=0.002,
        server_host="testhost",
        server_port=9999,
    )

    assert mem.mode == "MB"
    assert mem.server_host == "testhost"
    assert mem.server_port == 9999
    assert mem.G == 0.002
    assert mem.t_sim == 100  # dt * 1000 = 0.1 * 1000
    assert mem.step_id == 0
    assert mem.sim_id == "test_model"


def test_remote_brian_memory_step_without_server(brain_stub, monkeypatch):
    """Test RemoteBrianModelMemory.step() handles connection failure gracefully."""
    mem = RemoteBrianModelMemory(
        brain=brain_stub,
        gain={"Odor": 0.0},
        dt=0.1,
        G=0.001,
        server_host="localhost",  # Use valid hostname
        server_port=99999,  # Invalid port (connection will be refused)
    )

    # Mock runRemoteModel to return 0 (simulating connection failure)
    def mock_run_remote(*args, **kwargs):
        return 0  # Connection failed, returns 0

    monkeypatch.setattr(mem, "runRemoteModel", mock_run_remote)

    # Step should not crash, should return gain with value 0
    result = mem.step(dx={"Odor": 0.5}, reward=True, t_warmup=0)

    assert result == {"Odor": 0.0}  # G * mbon_dif = 0.001 * 0
    assert mem.step_id == 1  # Step ID incremented


# ============================================================================
# Edge Cases
# ============================================================================


def test_rl_memory_empty_gain_space():
    """Test RLmemory handles edge case of no stimuli (empty gain dict)."""
    # This should work but have no actions
    mem = RLmemory(brain=None, gain={}, dt=0.1, state_spacePerSide=1, gain_space=[0.0])

    # n=0 → state_space should be 1D array with single state
    assert mem.state_space.shape[0] == 1

    # actions should be list with single empty tuple
    assert mem.actions == [()]


def test_rl_memory_zero_state_space_per_side():
    """Test RLmemory with state_spacePerSide=0 (only one state)."""
    mem = RLmemory(
        brain=None,
        gain={"odor1": 0.0},
        dt=0.1,
        state_spacePerSide=0,  # 2*0+1 = 1 state
        gain_space=[0.0, 100.0],
    )

    # Only 1 state, 2 actions
    assert mem.state_space.shape[0] == 1
    assert mem.q_table.shape == (1, 2)


def test_memory_reward_accumulation_precision():
    """Test reward accumulation handles floating point precision."""
    mem = Memory(brain=None, gain={"stim1": 0.0}, dt=0.1)

    # Accumulate many small rewards
    for _ in range(1000):
        mem.step(reward=False)

    # rewardSum should be approximately -10.0 (1000 * -0.01)
    assert mem.rewardSum == pytest.approx(-10.0, rel=1e-5)
