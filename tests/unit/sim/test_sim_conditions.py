"""
Unit tests for sim/conditions.py - Experiment completion condition checkers.

Tests cover:
- get_exp_condition() function
- ExpCondition base class
- PrefTrainCondition (preference training)
- CatchMeCondition (chase game)
- KeepFlagCondition (flag possession)
- CaptureFlagCondition (flag capture)

Target: Increase coverage from 18.5% to 60%+
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from shapely import geometry

from larvaworld.lib.sim import conditions


class TestGetExpCondition:
    """Test get_exp_condition() function"""

    def test_get_exp_condition_pitrain_mini(self):
        """Test get_exp_condition returns PrefTrainCondition for PItrain_mini."""
        result = conditions.get_exp_condition("PItrain_mini")
        assert result == conditions.PrefTrainCondition

    def test_get_exp_condition_pitrain(self):
        """Test get_exp_condition returns PrefTrainCondition for PItrain."""
        result = conditions.get_exp_condition("PItrain")
        assert result == conditions.PrefTrainCondition

    def test_get_exp_condition_catch_me(self):
        """Test get_exp_condition returns CatchMeCondition for catch_me."""
        result = conditions.get_exp_condition("catch_me")
        assert result == conditions.CatchMeCondition

    def test_get_exp_condition_keep_the_flag(self):
        """Test get_exp_condition returns KeepFlagCondition for keep_the_flag."""
        result = conditions.get_exp_condition("keep_the_flag")
        assert result == conditions.KeepFlagCondition

    def test_get_exp_condition_capture_the_flag(self):
        """Test get_exp_condition returns CaptureFlagCondition for capture_the_flag."""
        result = conditions.get_exp_condition("capture_the_flag")
        assert result == conditions.CaptureFlagCondition

    def test_get_exp_condition_unknown(self):
        """Test get_exp_condition returns None for unknown experiment type."""
        result = conditions.get_exp_condition("unknown_experiment")
        assert result is None

    def test_get_exp_condition_empty_string(self):
        """Test get_exp_condition with empty string."""
        result = conditions.get_exp_condition("")
        assert result is None


class TestExpConditionBase:
    """Test ExpCondition base class"""

    def test_expcondition_initialization(self):
        """Test ExpCondition initialization."""
        mock_env = Mock()
        mock_env.agents = []
        mock_env.sources = []

        cond = conditions.ExpCondition(env=mock_env)

        assert cond.env == mock_env

    def test_expcondition_check_default(self):
        """Test ExpCondition.check() returns False by default."""
        mock_env = Mock()
        cond = conditions.ExpCondition(env=mock_env)

        result = cond.check()
        assert result == False

    def test_expcondition_agents_property(self):
        """Test agents property returns env.agents."""
        mock_env = Mock()
        mock_agents = [Mock(), Mock()]
        mock_env.agents = mock_agents

        cond = conditions.ExpCondition(env=mock_env)

        assert cond.agents == mock_agents

    def test_expcondition_sources_property(self):
        """Test sources property returns env.sources."""
        mock_env = Mock()
        mock_sources = [Mock(), Mock()]
        mock_env.sources = mock_sources

        cond = conditions.ExpCondition(env=mock_env)

        assert cond.sources == mock_sources

    def test_expcondition_set_state_with_screen(self):
        """Test set_state() with screen manager available."""
        mock_env = Mock()
        mock_screen_state = Mock()
        mock_env.screen_manager.screen_state = mock_screen_state

        cond = conditions.ExpCondition(env=mock_env)
        cond.set_state("test state")

        mock_screen_state.set_text.assert_called_once_with("test state")

    def test_expcondition_set_state_without_screen(self):
        """Test set_state() without screen manager (should not crash)."""
        mock_env = Mock()
        mock_env.screen_manager = None

        cond = conditions.ExpCondition(env=mock_env)
        # Should not raise exception
        cond.set_state("test state")

    def test_expcondition_flash_text_with_input_box(self):
        """Test flash_text() with input_box available."""
        mock_env = Mock()
        mock_input_box = Mock()
        mock_env.input_box = mock_input_box

        cond = conditions.ExpCondition(env=mock_env)
        cond.flash_text("test flash")

        mock_input_box.flash_text.assert_called_once_with("test flash")

    def test_expcondition_flash_text_without_input_box(self):
        """Test flash_text() without input_box (should not crash)."""
        mock_env = Mock()
        mock_env.input_box = None

        cond = conditions.ExpCondition(env=mock_env)
        # Should not raise exception
        cond.flash_text("test flash")


class TestPrefTrainCondition:
    """Test PrefTrainCondition class"""

    def test_preftrain_initialization(self):
        """Test PrefTrainCondition initialization."""
        # Create mock sources with odor IDs
        cs_source1 = Mock()
        cs_source1.odor.id = "CS"
        cs_source2 = Mock()
        cs_source2.odor.id = "CS"
        ucs_source1 = Mock()
        ucs_source1.odor.id = "UCS"
        ucs_source2 = Mock()
        ucs_source2.odor.id = "UCS"

        mock_env = Mock()
        mock_env.sources = [cs_source1, cs_source2, ucs_source1, ucs_source2]

        cond = conditions.PrefTrainCondition(env=mock_env)

        assert cond.peak_intensity == 2.0
        assert cond.CS_counter == 0
        assert cond.UCS_counter == 0
        assert len(cond.CS_sources) == 2
        assert len(cond.UCS_sources) == 2

    def test_preftrain_toggle_odors_cs_on(self):
        """Test toggle_odors() with CS on, UCS off."""
        cs_source = Mock()
        cs_source.odor.id = "CS"
        ucs_source = Mock()
        ucs_source.odor.id = "UCS"

        mock_env = Mock()
        mock_env.sources = [cs_source, ucs_source]

        cond = conditions.PrefTrainCondition(env=mock_env)
        cond.toggle_odors(CS_intensity=2.0, UCS_intensity=0.0)

        assert cs_source.odor.intensity == 2.0
        assert cs_source.visible == True
        assert ucs_source.odor.intensity == 0.0
        assert ucs_source.visible == False

    def test_preftrain_toggle_odors_ucs_on(self):
        """Test toggle_odors() with CS off, UCS on."""
        cs_source = Mock()
        cs_source.odor.id = "CS"
        ucs_source = Mock()
        ucs_source.odor.id = "UCS"

        mock_env = Mock()
        mock_env.sources = [cs_source, ucs_source]

        cond = conditions.PrefTrainCondition(env=mock_env)
        cond.toggle_odors(CS_intensity=0.0, UCS_intensity=2.0)

        assert cs_source.odor.intensity == 0.0
        assert cs_source.visible == False
        assert ucs_source.odor.intensity == 2.0
        assert ucs_source.visible == True

    def test_preftrain_move_larvae_to_center(self):
        """Test move_larvae_to_center() calls reset_larva_pose on all agents."""
        agent1 = Mock()
        agent2 = Mock()

        mock_env = Mock()
        mock_env.agents = [agent1, agent2]
        mock_env.sources = []

        cond = conditions.PrefTrainCondition(env=mock_env)
        cond.move_larvae_to_center()

        agent1.reset_larva_pose.assert_called_once()
        agent2.reset_larva_pose.assert_called_once()


class TestCatchMeCondition:
    """Test CatchMeCondition class"""

    def test_catchme_initialization(self):
        """Test CatchMeCondition initialization."""
        # Create mock agents with groups and brain
        left_agent = Mock()
        left_agent.group = "Left"
        left_agent.brain.olfactor.gain = {"odor1": 1.0, "odor2": 2.0}

        right_agent = Mock()
        right_agent.group = "Right"
        right_agent.brain.olfactor.gain = {"odor1": 1.0, "odor2": 2.0}

        mock_env = Mock()
        mock_env.agents = [left_agent, right_agent]
        mock_env.sources = []

        with patch("random.uniform", return_value=0.3):  # Will select "Left"
            cond = conditions.CatchMeCondition(env=mock_env)

        assert cond.target_group in ["Left", "Right"]
        assert cond.follower_group in ["Left", "Right"]
        assert cond.target_group != cond.follower_group
        assert isinstance(cond.score, dict)
        assert cond.score[cond.target_group] == 0.0
        assert cond.score[cond.follower_group] == 0.0

    def test_catchme_set_target_group_left(self):
        """Test set_target_group() with Left as target."""
        left_agent = Mock()
        left_agent.group = "Left"
        left_agent.brain.olfactor.gain = {"odor1": 1.0}

        right_agent = Mock()
        right_agent.group = "Right"
        right_agent.brain.olfactor.gain = {"odor1": 1.0}

        mock_env = Mock()
        mock_env.agents = [left_agent, right_agent]
        mock_env.sources = []

        with patch("random.uniform", return_value=0.3):
            cond = conditions.CatchMeCondition(env=mock_env)

        cond.set_target_group("Left")

        assert cond.target_group == "Left"
        assert cond.follower_group == "Right"
        assert left_agent in cond.targets
        assert right_agent in cond.followers

    def test_catchme_set_target_group_right(self):
        """Test set_target_group() with Right as target."""
        left_agent = Mock()
        left_agent.group = "Left"
        left_agent.brain.olfactor.gain = {"odor1": 1.0}

        right_agent = Mock()
        right_agent.group = "Right"
        right_agent.brain.olfactor.gain = {"odor1": 1.0}

        mock_env = Mock()
        mock_env.agents = [left_agent, right_agent]
        mock_env.sources = []

        with patch("random.uniform", return_value=0.3):
            cond = conditions.CatchMeCondition(env=mock_env)

        cond.set_target_group("Right")

        assert cond.target_group == "Right"
        assert cond.follower_group == "Left"


class TestKeepFlagCondition:
    """Test KeepFlagCondition class"""

    def test_keepflag_initialization(self):
        """Test KeepFlagCondition initialization."""
        flag = Mock()
        flag.unique_id = "Flag"

        mock_env = Mock()
        mock_env.sources = [flag]
        mock_env.agents = []

        cond = conditions.KeepFlagCondition(env=mock_env)

        assert cond.flag == flag
        assert cond.l_t == 0
        assert cond.r_t == 0

    def test_keepflag_check_no_carrier(self):
        """Test check() when flag has no carrier."""
        flag = Mock()
        flag.unique_id = "Flag"
        flag.is_carried_by = None

        mock_env = Mock()
        mock_env.sources = [flag]
        mock_env.agents = []
        mock_env.Nticks = 10
        mock_env.dt = 0.1

        cond = conditions.KeepFlagCondition(env=mock_env)

        result = cond.check()

        assert result == False
        assert cond.l_t == 0
        assert cond.r_t == 0

    def test_keepflag_check_left_carrier(self):
        """Test check() when left group carries flag."""
        carrier = Mock()
        carrier.group = "Left"

        flag = Mock()
        flag.unique_id = "Flag"
        flag.is_carried_by = carrier

        mock_env = Mock()
        mock_env.sources = [flag]
        mock_env.agents = []
        mock_env.Nticks = 10
        mock_env.dt = 1.0  # 1 second per tick
        mock_env.screen_manager = None

        cond = conditions.KeepFlagCondition(env=mock_env)

        # Run check multiple times
        for i in range(5):
            result = cond.check()
            if i < 4:
                assert result == False

        # After 5 ticks, l_t should be 5 seconds
        assert cond.l_t == 5.0
        assert cond.r_t == 0

    def test_keepflag_check_win_condition(self):
        """Test check() returns True when carrier holds flag for 180 seconds."""
        carrier = Mock()
        carrier.group = "Left"

        flag = Mock()
        flag.unique_id = "Flag"
        flag.is_carried_by = carrier

        mock_env = Mock()
        mock_env.sources = [flag]
        mock_env.agents = []
        mock_env.Nticks = 10
        mock_env.dt = 50.0  # Large timestep
        mock_env.screen_manager = None

        cond = conditions.KeepFlagCondition(env=mock_env)

        # First check: l_t = 50
        result = cond.check()
        assert result == False
        assert cond.l_t == 50.0

        # Second check: l_t = 100
        result = cond.check()
        assert result == False

        # Third check: l_t = 150
        result = cond.check()
        assert result == False

        # Fourth check: l_t = 200 > 180 -> WIN!
        result = cond.check()
        assert result == True


class TestCaptureFlagCondition:
    """Test CaptureFlagCondition class"""

    def test_captureflag_initialization(self):
        """Test CaptureFlagCondition initialization."""
        flag = Mock()
        flag.unique_id = "Flag"
        flag.get_position.return_value = (0.0, 0.0)
        flag.radius = 0.01

        l_base = Mock()
        l_base.unique_id = "Left_base"
        l_base.get_position.return_value = (-0.1, 0.0)
        l_base.radius = 0.02

        r_base = Mock()
        r_base.unique_id = "Right_base"
        r_base.get_position.return_value = (0.1, 0.0)
        r_base.radius = 0.02

        mock_env = Mock()
        mock_env.sources = [flag, l_base, r_base]
        mock_env.agents = []

        cond = conditions.CaptureFlagCondition(env=mock_env)

        assert cond.flag == flag
        assert cond.l_base == l_base
        assert cond.r_base == r_base
        assert cond.l_base_p == (-0.1, 0.0)
        assert cond.r_base_p == (0.1, 0.0)

    def test_captureflag_check_no_win(self):
        """Test check() when flag is not at any base."""
        flag = Mock()
        flag.unique_id = "Flag"
        flag.get_position.return_value = (0.0, 0.0)
        flag.radius = 0.01

        l_base = Mock()
        l_base.unique_id = "Left_base"
        l_base.get_position.return_value = (-0.1, 0.0)
        l_base.radius = 0.02

        r_base = Mock()
        r_base.unique_id = "Right_base"
        r_base.get_position.return_value = (0.1, 0.0)
        r_base.radius = 0.02

        mock_env = Mock()
        mock_env.sources = [flag, l_base, r_base]
        mock_env.agents = []
        mock_env.Nticks = 10
        mock_env.screen_manager = None

        cond = conditions.CaptureFlagCondition(env=mock_env)

        result = cond.check()
        assert result == False

    def test_captureflag_check_left_wins(self):
        """Test check() when flag is captured by left base."""
        flag = Mock()
        flag.unique_id = "Flag"
        # Flag very close to left base
        flag.get_position.return_value = (-0.1, 0.0)
        flag.radius = 0.01

        l_base = Mock()
        l_base.unique_id = "Left_base"
        l_base.get_position.return_value = (-0.1, 0.0)
        l_base.radius = 0.02

        r_base = Mock()
        r_base.unique_id = "Right_base"
        r_base.get_position.return_value = (0.1, 0.0)
        r_base.radius = 0.02

        mock_env = Mock()
        mock_env.sources = [flag, l_base, r_base]
        mock_env.agents = []
        mock_env.Nticks = 10
        mock_env.screen_manager = None

        cond = conditions.CaptureFlagCondition(env=mock_env)

        result = cond.check()
        assert result == True

    def test_captureflag_check_right_wins(self):
        """Test check() when flag is captured by right base."""
        flag = Mock()
        flag.unique_id = "Flag"
        # Flag very close to right base
        flag.get_position.return_value = (0.1, 0.0)
        flag.radius = 0.01

        l_base = Mock()
        l_base.unique_id = "Left_base"
        l_base.get_position.return_value = (-0.1, 0.0)
        l_base.radius = 0.02

        r_base = Mock()
        r_base.unique_id = "Right_base"
        r_base.get_position.return_value = (0.1, 0.0)
        r_base.radius = 0.02

        mock_env = Mock()
        mock_env.sources = [flag, l_base, r_base]
        mock_env.agents = []
        mock_env.Nticks = 10
        mock_env.screen_manager = None

        cond = conditions.CaptureFlagCondition(env=mock_env)

        result = cond.check()
        assert result == True


class TestConditionSubclasses:
    """Test that all condition subclasses inherit from ExpCondition"""

    def test_preftrain_inherits_expcondition(self):
        """Test PrefTrainCondition inherits from ExpCondition."""
        assert issubclass(conditions.PrefTrainCondition, conditions.ExpCondition)

    def test_catchme_inherits_expcondition(self):
        """Test CatchMeCondition inherits from ExpCondition."""
        assert issubclass(conditions.CatchMeCondition, conditions.ExpCondition)

    def test_keepflag_inherits_expcondition(self):
        """Test KeepFlagCondition inherits from ExpCondition."""
        assert issubclass(conditions.KeepFlagCondition, conditions.ExpCondition)

    def test_captureflag_inherits_expcondition(self):
        """Test CaptureFlagCondition inherits from ExpCondition."""
        assert issubclass(conditions.CaptureFlagCondition, conditions.ExpCondition)

    def test_all_conditions_have_check_method(self):
        """Test all condition classes have check() method."""
        condition_classes = [
            conditions.ExpCondition,
            conditions.PrefTrainCondition,
            conditions.CatchMeCondition,
            conditions.KeepFlagCondition,
            conditions.CaptureFlagCondition,
        ]

        for cls in condition_classes:
            assert hasattr(cls, "check")
            assert callable(cls.check)


class TestModuleExports:
    """Test module __all__ exports"""

    def test_all_exports_defined(self):
        """Test __all__ contains expected exports."""
        assert "get_exp_condition" in conditions.__all__

    def test_get_exp_condition_accessible(self):
        """Test get_exp_condition is accessible."""
        assert hasattr(conditions, "get_exp_condition")
        assert callable(conditions.get_exp_condition)
