"""
Unit tests for lib/model/modules/intermitter.py

Tests behavioral state machine (run/pause/feed), EEB decay,
bout generation, and stridechain logic.

Philosophy: Test the STATE TRANSITIONS, not the physics.
"""

import pytest
import numpy as np

from larvaworld.lib import util
from larvaworld.lib.model.modules.intermitter import (
    Intermitter,
    OfflineIntermitter,
    BranchIntermitter,
    FittedIntermitter,
    default_bout_distros,
    get_EEB_poly1d,
    get_EEB_time_fractions,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def intermitter_stridechain():
    """Intermitter with stridechain mode."""
    return Intermitter(
        dt=0.1,
        EEB=0.3,
        crawl_freq=1.42,
        feed_freq=2.0,
        run_mode="stridechain",
        feed_bouts=False,
        stridechain_dist=util.AttrDict({"range": [5, 20], "name": "uniform"}),
        pause_dist=util.AttrDict({"range": [0.5, 2.0], "name": "uniform"}),
    )


@pytest.fixture
def intermitter_exec():
    """Intermitter with exec (temporal) mode."""
    return Intermitter(
        dt=0.1,
        EEB=0.5,
        crawl_freq=1.0,
        feed_freq=2.0,
        run_mode="exec",
        feed_bouts=False,
        run_dist=util.AttrDict({"range": [1.0, 5.0], "name": "uniform"}),
        pause_dist=util.AttrDict({"range": [0.5, 2.0], "name": "uniform"}),
    )


@pytest.fixture
def intermitter_with_feeding():
    """Intermitter with feeding enabled."""
    return Intermitter(
        dt=0.1,
        EEB=0.4,
        crawl_freq=1.42,
        feed_freq=2.0,
        run_mode="stridechain",
        feed_bouts=True,
        feeder_reoccurence_rate=0.6,
        stridechain_dist=util.AttrDict({"range": [3, 10], "name": "uniform"}),
        pause_dist=util.AttrDict({"range": [0.3, 1.5], "name": "uniform"}),
    )


# ============================================================================
# Intermitter Base Class Tests
# ============================================================================


def test_intermitter_initialization_stridechain(intermitter_stridechain):
    """Test Intermitter initialization in stridechain mode."""
    inter = intermitter_stridechain

    assert inter.run_mode == "stridechain"
    assert inter.stridechain_generator is not None
    assert inter.run_generator is None
    assert inter.pause_generator is not None
    assert inter.cur_state is None
    assert inter.EEB == 0.3


def test_intermitter_initialization_exec(intermitter_exec):
    """Test Intermitter initialization in exec (temporal) mode."""
    inter = intermitter_exec

    assert inter.run_mode == "exec"
    assert inter.run_generator is not None
    assert inter.stridechain_generator is None
    assert inter.pause_generator is not None


def test_intermitter_uses_default_distributions():
    """Test Intermitter uses default bout distributions when None provided."""
    inter = Intermitter(
        dt=0.1,
        run_mode="stridechain",
        stridechain_dist=None,  # Should use default
        pause_dist=None,  # Should use default
    )

    # Should use default_bout_distros
    assert inter.stridechain_dist == default_bout_distros.run_count
    assert inter.pause_dist == default_bout_distros.pause_dur


def test_intermitter_eeb_as_feeder_reoccurence():
    """Test EEB is used as feeder_reoccurence_rate when None."""
    inter = Intermitter(
        dt=0.1,
        EEB=0.7,
        run_mode="exec",
        feeder_reoccurence_rate=None,  # Should use EEB
    )

    assert inter.feeder_reoccurence_rate == 0.7
    assert inter.use_EEB is True


def test_intermitter_explicit_feeder_reoccurence():
    """Test explicit feeder_reoccurence_rate overrides EEB."""
    inter = Intermitter(dt=0.1, EEB=0.3, run_mode="exec", feeder_reoccurence_rate=0.9)

    assert inter.feeder_reoccurence_rate == 0.9
    assert inter.use_EEB is False


# ============================================================================
# State Machine Tests
# ============================================================================


def test_intermitter_trigger_locomotion_stridechain(intermitter_stridechain):
    """Test trigger_locomotion generates stridechain."""
    inter = intermitter_stridechain

    # Initially no state
    assert inter.cur_state is None

    # Trigger locomotion
    inter.trigger_locomotion()

    assert inter.cur_state == "exec"
    assert inter.exp_Nstrides is not None  # Stridechain generated
    assert inter.exp_Nstrides >= 5  # Within range [5, 20]
    assert inter.exp_Nstrides <= 20
    assert inter.cur_Nstrides == 0
    assert inter.ticks == 0  # Reset


def test_intermitter_trigger_locomotion_exec(intermitter_exec):
    """Test trigger_locomotion generates temporal run."""
    inter = intermitter_exec

    inter.trigger_locomotion()

    assert inter.cur_state == "exec"
    assert inter.exp_Trun is not None  # Temporal run generated
    assert inter.exp_Trun >= 1.0
    assert inter.exp_Trun <= 5.0


def test_intermitter_interrupt_locomotion(intermitter_stridechain):
    """Test interrupt_locomotion transitions to pause."""
    inter = intermitter_stridechain

    # Start in exec state
    inter.trigger_locomotion()
    assert inter.cur_state == "exec"

    # Interrupt
    inter.interrupt_locomotion()

    assert inter.cur_state == "pause"
    assert inter.exp_Tpause is not None
    assert inter.exp_Tpause >= 0.5
    assert inter.exp_Tpause <= 2.0


def test_intermitter_interrupt_does_nothing_if_not_exec(intermitter_stridechain):
    """Test interrupt_locomotion does nothing if not in exec state."""
    inter = intermitter_stridechain

    # Start in pause
    inter.cur_state = "pause"
    inter.exp_Tpause = 1.0

    # Interrupt should do nothing
    inter.interrupt_locomotion()

    assert inter.cur_state == "pause"  # Unchanged


def test_intermitter_stridechain_completion(intermitter_stridechain):
    """Test stridechain_completed property."""
    inter = intermitter_stridechain

    inter.exp_Nstrides = 10
    inter.cur_Nstrides = 5
    assert inter.stridechain_completed is False

    inter.cur_Nstrides = 11  # Exceeded
    assert inter.stridechain_completed is True


def test_intermitter_run_completion(intermitter_exec):
    """Test run_completed property for temporal runs."""
    inter = intermitter_exec

    # Manually set exp_Trun and test the property logic
    inter.exp_Trun = 2.0  # 2 seconds
    inter.ticks = 0

    # At t=0, run should not be completed
    assert inter.t == 0.0
    assert inter.run_completed is False

    # At t=1.5 (< 2.0), still not completed
    inter.ticks = 15  # 15 * 0.1 = 1.5 seconds
    assert inter.t == 1.5
    assert inter.run_completed is False

    # At t=2.1 (> 2.0), run is completed
    inter.ticks = 21  # 21 * 0.1 = 2.1 seconds
    assert inter.t == 2.1
    assert inter.run_completed is True


def test_intermitter_pause_completion(intermitter_stridechain):
    """Test pause_completed property."""
    inter = intermitter_stridechain

    inter.exp_Tpause = 1.0
    inter.ticks = 0
    assert inter.pause_completed is False

    inter.ticks = int(1.1 / inter.dt)  # Beyond expected pause
    assert inter.pause_completed is True


def test_intermitter_alternate_crawl_and_pause_stridechain(intermitter_stridechain):
    """Test alternate_crawlNpause handles stridechain completion."""
    inter = intermitter_stridechain

    # Start locomotion
    inter.trigger_locomotion()
    initial_state = inter.cur_state

    # Complete strides up to expected
    for _ in range(inter.exp_Nstrides + 1):
        inter.alternate_crawlNpause(stride_completed=True)

    # Should have interrupted to pause
    assert inter.cur_state == "pause"


def test_intermitter_step_increments_time(intermitter_stridechain):
    """Test step() increments time via count_time()."""
    inter = intermitter_stridechain

    # Trigger state first (step calls update_state which may call trigger_locomotion)
    inter.trigger_locomotion()

    initial_total_ticks = inter.total_ticks
    inter.step()

    # total_ticks should increment (ticks may reset during state changes)
    assert inter.total_ticks == initial_total_ticks + 1


# ============================================================================
# Feed Bout Tests
# ============================================================================


def test_intermitter_feed_repeated_property(intermitter_with_feeding):
    """Test feed_repeated uses EEB or feeder_reoccurence_rate."""
    inter = intermitter_with_feeding

    # feeder_reoccurence_rate=0.6, so ~60% chance of repeat
    # Run multiple times to check stochastic behavior
    np.random.seed(42)
    repeats = [inter.feed_repeated for _ in range(50)]  # Reduced from 100

    # Should have both True and False
    assert True in repeats
    assert False in repeats


def test_intermitter_alternate_explore_exploit_on_food(intermitter_with_feeding):
    """Test alternate_exploreNexploit triggers feed on food."""
    inter = intermitter_with_feeding
    inter.cur_state = "exec"
    inter.cur_Nfeeds = None
    inter.EEB = 1.0  # Always feed when on food

    # On food, not feeding yet
    inter.alternate_exploreNexploit(feed_motion=False, on_food=True)

    # Should initiate feed
    assert inter.cur_Nfeeds == 1
    assert inter.cur_state == "feed"


def test_intermitter_alternate_explore_exploit_feed_success(intermitter_with_feeding):
    """Test successful feeding increments counters."""
    inter = intermitter_with_feeding
    inter.cur_state = "feed"
    inter.cur_Nfeeds = 1
    inter.feeder_reoccurence_rate = 1.0  # Always repeat

    # Feed motion while on food
    inter.alternate_exploreNexploit(feed_motion=True, on_food=True)

    assert inter.Nfeeds == 1
    assert inter.Nfeeds_success == 1
    assert inter.cur_Nfeeds == 2  # Repeated


def test_intermitter_alternate_explore_exploit_feed_failure(intermitter_with_feeding):
    """Test failed feeding (not on food) triggers locomotion."""
    inter = intermitter_with_feeding
    inter.cur_state = "feed"
    inter.cur_Nfeeds = 1

    # Feed motion but NOT on food
    inter.alternate_exploreNexploit(feed_motion=True, on_food=False)

    assert inter.Nfeeds == 1
    assert inter.Nfeeds_fail == 1
    assert inter.cur_state == "exec"  # Switched to exec


# ============================================================================
# Registration Tests
# ============================================================================


def test_intermitter_register_stridechain(intermitter_stridechain):
    """Test register() records stridechain bout."""
    inter = intermitter_stridechain
    inter.cur_state = "exec"
    inter.cur_Nstrides = 8
    inter.ticks = 50

    inter.register(bout="stridechain")

    assert inter.Nstridechains == 1
    assert len(inter.stridechain_lengths) == 1
    assert inter.stridechain_lengths[0] == 8
    assert len(inter.stridechain_durs) == 1
    assert inter.exp_Nstrides is None
    assert inter.cur_Nstrides == 0


def test_intermitter_register_pause(intermitter_stridechain):
    """Test register() records pause bout."""
    inter = intermitter_stridechain
    inter.cur_state = "pause"
    inter.ticks = 30

    inter.register(bout="pause")

    assert inter.Npauses == 1
    assert len(inter.pause_durs) == 1
    assert inter.exp_Tpause is None


def test_intermitter_register_run(intermitter_exec):
    """Test register() records run bout."""
    inter = intermitter_exec
    inter.cur_state = "exec"
    inter.ticks = 25

    inter.register(bout="run")

    assert inter.Nruns == 1
    assert len(inter.run_durs) == 1
    assert inter.exp_Trun is None


def test_intermitter_register_feedchain(intermitter_with_feeding):
    """Test register() records feedchain bout."""
    inter = intermitter_with_feeding
    inter.cur_state = "feed"
    inter.cur_Nfeeds = 3
    inter.ticks = 15

    inter.register(bout="feedchain")

    assert inter.Nfeedchains == 1
    assert len(inter.feedchain_lengths) == 1
    assert inter.feedchain_lengths[0] == 3
    assert len(inter.feedchain_durs) == 1
    assert inter.cur_Nfeeds is None


# ============================================================================
# OfflineIntermitter Tests
# ============================================================================


def test_offline_intermitter_initialization():
    """Test OfflineIntermitter calculates tick intervals."""
    offline = OfflineIntermitter(
        dt=0.1,
        crawl_freq=1.5,  # 1.5 Hz
        feed_freq=2.0,  # 2.0 Hz
        run_mode="exec",
    )

    # crawl_ticks = 1 / (1.5 * 0.1) = 6.67 → 7 ticks
    assert offline.crawl_ticks == 7

    # feed_ticks = 1 / (2.0 * 0.1) = 5 ticks
    assert offline.feed_ticks == 5


def test_offline_intermitter_step_auto_stride_completion():
    """Test OfflineIntermitter auto-detects stride completion."""
    offline = OfflineIntermitter(
        dt=0.1,
        crawl_freq=1.0,  # 1 Hz → 10 ticks per stride
        run_mode="exec",
        feed_bouts=False,
    )

    # Start locomotion
    offline.trigger_locomotion()
    assert offline.cur_state == "exec"

    # Step 9 times (not complete)
    for _ in range(9):
        result = offline.step(on_food=False)

    # Step 10th time (stride complete)
    result = offline.step(on_food=False)

    # cur_Nstrides should have incremented
    assert offline.cur_Nstrides > 0


def test_offline_intermitter_step_auto_feed_motion():
    """Test OfflineIntermitter auto-detects feed motion."""
    offline = OfflineIntermitter(
        dt=0.1,
        feed_freq=2.0,  # 2 Hz → 5 ticks per feed
        run_mode="exec",
        feed_bouts=True,
        EEB=1.0,  # Always feed on food
    )

    # Trigger feeding state
    offline.cur_state = "feed"
    offline.cur_Nfeeds = 1

    # Step to feed_ticks (5 ticks)
    for _ in range(4):
        offline.step(on_food=True)

    initial_feeds = offline.Nfeeds
    offline.step(on_food=True)

    # Feed should have occurred
    assert offline.Nfeeds > initial_feeds


# ============================================================================
# BranchIntermitter Tests
# ============================================================================


def test_branch_intermitter_initialization():
    """Test BranchIntermitter has feed_bouts=False."""
    branch = BranchIntermitter(
        dt=0.1, run_mode="stridechain", beta=4.7, c=0.7, sigma=1.0
    )

    assert branch.feed_bouts is False
    assert branch.beta == 4.7
    assert branch.c == 0.7
    assert branch.sigma == 1.0


def test_branch_intermitter_generate_stridechain(monkeypatch):
    """Test BranchIntermitter.generate_stridechain uses exp_bout."""

    # Mock util.exp_bout to return fixed value for fast testing
    def mock_exp_bout(beta, tmax, tmin):
        return 8  # Return fixed stridechain count

    # Mock at the module where it's used (intermitter imports util)
    monkeypatch.setattr(util, "exp_bout", mock_exp_bout)

    branch = BranchIntermitter(
        dt=0.1,
        run_mode="stridechain",
        beta=0.5,
        stridechain_dist=util.AttrDict(
            {"range": [3, 15], "name": "exponential", "beta": 0.5}
        ),
    )

    # Test that generate_stridechain calls exp_bout
    stridechain = branch.generate_stridechain()

    # Should return mocked value
    assert stridechain == 8


def test_branch_intermitter_generate_pause(monkeypatch):
    """Test BranchIntermitter.generate_pause uses critical_bout."""

    # Mock util.critical_bout to return fixed value for fast testing
    def mock_critical_bout(c, sigma, N, tmax, tmin):
        return 15  # Return fixed pause duration in ticks

    # Mock at the module where it's used (intermitter imports util)
    monkeypatch.setattr(util, "critical_bout", mock_critical_bout)

    branch = BranchIntermitter(
        dt=0.1,
        run_mode="stridechain",
        c=0.3,
        sigma=0.5,
        pause_dist=util.AttrDict(
            {"range": [0.2, 3.0], "name": "exponential", "beta": 0.5}
        ),
    )

    # Test that generate_pause calls critical_bout
    pause = branch.generate_pause()

    # Should return mocked value * dt (15 ticks * 0.1 = 1.5 seconds)
    assert pause == 1.5


# ============================================================================
# FittedIntermitter and Helper Function Tests
# NOTE: These tests are DEFERRED to PR-4A (Integration Tests)
# ============================================================================
#
# The following tests require either:
# 1. Registry initialization with real reference datasets
# 2. Long-running simulations (60+ minutes)
#
# They are moved to: tests/integration/test_fitted_intermitter.py
# - test_fitted_intermitter_initialization (requires registry)
# - test_get_eeb_poly1d (60-minute simulation)
# - test_get_eeb_time_fractions (60-minute simulation)
# ============================================================================

# ============================================================================
# build_dict Tests
# ============================================================================


def test_intermitter_build_dict(intermitter_stridechain):
    """Test build_dict generates statistics dictionary."""
    inter = intermitter_stridechain

    # Run some steps
    inter.trigger_locomotion()
    for _ in range(100):
        inter.step(stride_completed=(inter.ticks % 7 == 0))

    dic = inter.build_dict()

    # Check expected keys (actual keys from source code)
    assert "cum_t" in dic  # Not "cumulative_t"
    assert "num_ticks" in dic  # Not "N_tick"
    assert "stridechain_length" in dic
    assert "pause_dur" in dic
    assert "num_stridechains" in dic


# ============================================================================
# Edge Cases
# ============================================================================


def test_intermitter_invalid_run_mode():
    """Test Intermitter raises error for invalid run_mode."""
    # run_mode is param.Selector, so invalid value raises param error, not ValueError
    with pytest.raises(Exception):  # Will be param validation error
        Intermitter(dt=0.1, run_mode="invalid", run_dist=None, stridechain_dist=None)


def test_intermitter_force_trigger_from_exec(intermitter_stridechain):
    """Test trigger_locomotion with force=True overrides exec state."""
    inter = intermitter_stridechain

    # Start in exec
    inter.trigger_locomotion()
    first_exp_Nstrides = inter.exp_Nstrides

    # Force new locomotion (should regenerate)
    inter.trigger_locomotion(force=True)

    # Should have new stridechain (likely different)
    assert inter.cur_state == "exec"
    # exp_Nstrides may be different (regenerated)


def test_intermitter_register_with_no_cur_state(intermitter_stridechain):
    """Test register() returns early if no cur_state."""
    inter = intermitter_stridechain
    inter.cur_state = None

    # Should do nothing
    inter.register()

    # No bouts registered
    assert inter.Nstridechains == 0
    assert inter.Npauses == 0


def test_intermitter_counters_initialized_to_zero(intermitter_stridechain):
    """Test all bout counters initialized to zero."""
    inter = intermitter_stridechain

    assert inter.Nstrides == 0
    assert inter.Nstridechains == 0
    assert inter.Nruns == 0
    assert inter.Npauses == 0
    assert inter.Nfeeds == 0
    assert inter.Nfeedchains == 0
    assert inter.Nfeeds_success == 0
    assert inter.Nfeeds_fail == 0


def test_intermitter_active_bouts_property(intermitter_stridechain):
    """Test active_bouts property returns current bout states."""
    inter = intermitter_stridechain

    inter.exp_Nstrides = 10
    inter.cur_Nfeeds = 2
    inter.exp_Tpause = 1.5
    inter.exp_Trun = 3.0

    bouts = inter.active_bouts

    assert bouts == (10, 2, 1.5, 3.0)


def test_intermitter_mean_feed_freq_property(intermitter_with_feeding):
    """Test mean_feed_freq calculates correctly."""
    inter = intermitter_with_feeding

    inter.Nfeeds = 15
    inter.total_ticks = 100  # total_t = 100 * 0.1 = 10 seconds

    # mean_feed_freq = 15 / 10 = 1.5 Hz
    assert inter.mean_feed_freq == 1.5
