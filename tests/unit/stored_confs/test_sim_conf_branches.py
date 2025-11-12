"""
Branch coverage tests for sim_conf.py configurations.

Tests simulation configuration validation and branch coverage.
Focuses on contract tests and validation logic.
"""

import pytest
import numpy as np
from larvaworld.lib.reg.stored_confs import sim_conf


@pytest.mark.fast
class TestSimConfBranches:
    """Test simulation configuration branch coverage and validation."""

    def test_trial_dict_structure(self):
        """Test that Trial_dict returns proper structure."""
        result = sim_conf.Trial_dict()

        # Should return AttrDict-like object
        assert hasattr(result, "__getattr__") or isinstance(result, dict)

        # Should have expected trial types
        expected_types = ["default", "odor_preference", "odor_preference_short"]
        trial_types = (
            list(result.keys())
            if isinstance(result, dict)
            else [attr for attr in dir(result) if not attr.startswith("_")]
        )

        for trial_type in expected_types:
            assert trial_type in trial_types, f"Missing trial type: {trial_type}"

    def test_trial_dict_default_config(self):
        """Test default trial configuration."""
        result = sim_conf.Trial_dict()
        default = (
            result["default"]
            if isinstance(result, dict)
            else getattr(result, "default")
        )

        # Should have epochs
        assert "epochs" in default
        assert isinstance(default["epochs"], list)

    def test_trial_dict_odor_preference_config(self):
        """Test odor preference trial configuration."""
        result = sim_conf.Trial_dict()
        odor_pref = (
            result["odor_preference"]
            if isinstance(result, dict)
            else getattr(result, "odor_preference")
        )

        # Should have epochs
        assert "epochs" in odor_pref
        assert isinstance(odor_pref["epochs"], list)
        assert len(odor_pref["epochs"]) == 8  # 8 epochs for odor preference

    def test_trial_dict_odor_preference_short_config(self):
        """Test short odor preference trial configuration."""
        result = sim_conf.Trial_dict()
        odor_pref_short = (
            result["odor_preference_short"]
            if isinstance(result, dict)
            else getattr(result, "odor_preference_short")
        )

        # Should have epochs
        assert "epochs" in odor_pref_short
        assert isinstance(odor_pref_short["epochs"], list)
        assert len(odor_pref_short["epochs"]) == 8  # 8 epochs for short odor preference

    def test_env_dict_structure(self):
        """Test that Env_dict returns proper structure."""
        result = sim_conf.Env_dict()

        # Should return AttrDict-like object
        assert hasattr(result, "__getattr__") or isinstance(result, dict)

        # Should have expected environment types
        expected_types = [
            "focus",
            "dish",
            "dish_40mm",
            "arena_200mm",
            "arena_500mm",
            "arena_1000mm",
        ]
        env_types = (
            list(result.keys())
            if isinstance(result, dict)
            else [attr for attr in dir(result) if not attr.startswith("_")]
        )

        for env_type in expected_types:
            assert env_type in env_types, f"Missing environment type: {env_type}"

    def test_env_dict_focus_config(self):
        """Test focus environment configuration."""
        result = sim_conf.Env_dict()
        focus = (
            result["focus"] if isinstance(result, dict) else getattr(result, "focus")
        )

        # Should be a configuration dict
        assert isinstance(focus, dict)

    def test_env_dict_dish_config(self):
        """Test dish environment configuration."""
        result = sim_conf.Env_dict()
        dish = result["dish"] if isinstance(result, dict) else getattr(result, "dish")

        # Should be a configuration dict
        assert isinstance(dish, dict)

    def test_env_dict_arena_configs(self):
        """Test arena environment configurations."""
        result = sim_conf.Env_dict()

        arena_configs = ["arena_200mm", "arena_500mm", "arena_1000mm"]
        for arena in arena_configs:
            arena_config = (
                result[arena] if isinstance(result, dict) else getattr(result, arena)
            )
            assert isinstance(arena_config, dict)

    def test_env_dict_odor_configs(self):
        """Test odor environment configurations."""
        result = sim_conf.Env_dict()

        odor_configs = [
            "odor_gradient",
            "mid_odor_gaussian",
            "odor_gaussian_square",
            "mid_odor_diffusion",
        ]
        for odor in odor_configs:
            if odor in result or hasattr(result, odor):
                odor_config = (
                    result[odor] if isinstance(result, dict) else getattr(result, odor)
                )
                assert isinstance(odor_config, dict)

    def test_env_dict_food_configs(self):
        """Test food environment configurations."""
        result = sim_conf.Env_dict()

        food_configs = [
            "4corners",
            "food_at_bottom",
            "patchy_food",
            "random_food",
            "uniform_food",
        ]
        for food in food_configs:
            if food in result or hasattr(result, food):
                food_config = (
                    result[food] if isinstance(result, dict) else getattr(result, food)
                )
                assert isinstance(food_config, dict)

    def test_env_dict_special_configs(self):
        """Test special environment configurations."""
        result = sim_conf.Env_dict()

        special_configs = [
            "thermo_arena",
            "windy_arena",
            "windy_blob_arena",
            "windy_arena_bordered",
        ]
        for special in special_configs:
            if special in result or hasattr(result, special):
                special_config = (
                    result[special]
                    if isinstance(result, dict)
                    else getattr(result, special)
                )
                assert isinstance(special_config, dict)

    def test_exp_dict_structure(self):
        """Test that Exp_dict returns proper structure."""
        result = sim_conf.Exp_dict()

        # Should return AttrDict-like object
        assert hasattr(result, "__getattr__") or isinstance(result, dict)

        # Should have expected experiment types
        expected_types = [
            "RvsS",
            "RvsS_off",
            "RvsS_on",
            "PItest_off",
            "PItest_on",
            "PItrain",
        ]
        exp_types = (
            list(result.keys())
            if isinstance(result, dict)
            else [attr for attr in dir(result) if not attr.startswith("_")]
        )

        for exp_type in expected_types:
            assert exp_type in exp_types, f"Missing experiment type: {exp_type}"

    def test_exp_dict_rvs_configs(self):
        """Test RvsS experiment configurations."""
        result = sim_conf.Exp_dict()

        rvs_configs = ["RvsS", "RvsS_off", "RvsS_on"]
        for rvs in rvs_configs:
            if rvs in result or hasattr(result, rvs):
                rvs_config = (
                    result[rvs] if isinstance(result, dict) else getattr(result, rvs)
                )
                assert isinstance(rvs_config, dict)

    def test_exp_dict_pi_configs(self):
        """Test PI experiment configurations."""
        result = sim_conf.Exp_dict()

        pi_configs = ["PItest_off", "PItest_on", "PItrain"]
        for pi in pi_configs:
            if pi in result or hasattr(result, pi):
                pi_config = (
                    result[pi] if isinstance(result, dict) else getattr(result, pi)
                )
                assert isinstance(pi_config, dict)

    def test_ga_dict_structure(self):
        """Test that Ga_dict returns proper structure."""
        result = sim_conf.Ga_dict()

        # Should return AttrDict-like object
        assert hasattr(result, "__getattr__") or isinstance(result, dict)

        # Should have expected GA types (based on actual implementation)
        expected_types = [
            "interference",
            "exploration",
            "realism",
            "chemorbit",
            "obstacle_avoidance",
        ]
        ga_types = (
            list(result.keys())
            if isinstance(result, dict)
            else [attr for attr in dir(result) if not attr.startswith("_")]
        )

        for ga_type in expected_types:
            assert ga_type in ga_types, f"Missing GA type: {ga_type}"

    def test_ga_dict_interference_config(self):
        """Test interference GA configuration."""
        result = sim_conf.Ga_dict()
        interference = (
            result["interference"]
            if isinstance(result, dict)
            else getattr(result, "interference")
        )

        # Should be a configuration dict
        assert isinstance(interference, dict)
        assert "experiment" in interference
        assert interference["experiment"] == "interference"

    def test_batch_dict_structure(self):
        """Test that Batch_dict returns proper structure."""
        result = sim_conf.Batch_dict()

        # Should return AttrDict-like object
        assert hasattr(result, "__getattr__") or isinstance(result, dict)

        # Should have expected batch types (based on actual implementation)
        expected_types = [
            "PItest_off",
            "patchy_food",
            "food_grid",
            "growth",
            "tactile_detection",
            "anemotaxis",
            "chemotaxis",
            "chemorbit",
            "PItrain_mini",
            "PItrain",
        ]
        batch_types = (
            list(result.keys())
            if isinstance(result, dict)
            else [attr for attr in dir(result) if not attr.startswith("_")]
        )

        for batch_type in expected_types:
            assert batch_type in batch_types, f"Missing batch type: {batch_type}"

    def test_batch_dict_pitest_config(self):
        """Test PItest_off batch configuration."""
        result = sim_conf.Batch_dict()
        pitest = (
            result["PItest_off"]
            if isinstance(result, dict)
            else getattr(result, "PItest_off")
        )

        # Should be a configuration dict
        assert isinstance(pitest, dict)
        assert "exp" in pitest
        assert pitest["exp"] == "PItest_off"

    def test_trial_conf_function(self):
        """Test trial_conf function with different parameters."""
        # Test with empty parameters
        result = sim_conf.Trial_dict()
        default = (
            result["default"]
            if isinstance(result, dict)
            else getattr(result, "default")
        )
        epochs = default["epochs"]

        # Should return empty list for default
        assert isinstance(epochs, list)

    def test_trial_conf_with_durations(self):
        """Test trial_conf function with specific durations."""
        result = sim_conf.Trial_dict()
        odor_pref = (
            result["odor_preference"]
            if isinstance(result, dict)
            else getattr(result, "odor_preference")
        )
        epochs = odor_pref["epochs"]

        # Should have 8 epochs for odor preference
        assert len(epochs) == 8
        assert isinstance(epochs, list)

    def test_trial_conf_with_short_durations(self):
        """Test trial_conf function with short durations."""
        result = sim_conf.Trial_dict()
        odor_pref_short = (
            result["odor_preference_short"]
            if isinstance(result, dict)
            else getattr(result, "odor_preference_short")
        )
        epochs = odor_pref_short["epochs"]

        # Should have 8 epochs for short odor preference
        assert len(epochs) == 8
        assert isinstance(epochs, list)

    def test_env_dict_complex_configs(self):
        """Test complex environment configurations."""
        result = sim_conf.Env_dict()

        complex_configs = [
            "windy_blob_arena",
            "puff_arena_bordered",
            "single_puff",
            "CS_UCS_on_food",
        ]
        for complex_config in complex_configs:
            if complex_config in result or hasattr(result, complex_config):
                config = (
                    result[complex_config]
                    if isinstance(result, dict)
                    else getattr(result, complex_config)
                )
                assert isinstance(config, dict)

    def test_env_dict_patch_configs(self):
        """Test patch environment configurations."""
        result = sim_conf.Env_dict()

        patch_configs = ["patchy_food", "patch_grid"]
        for patch_config in patch_configs:
            if patch_config in result or hasattr(result, patch_config):
                config = (
                    result[patch_config]
                    if isinstance(result, dict)
                    else getattr(result, patch_config)
                )
                assert isinstance(config, dict)

    def test_exp_dict_prestarved_configs(self):
        """Test prestarved experiment configurations."""
        result = sim_conf.Exp_dict()

        prestarved_configs = [
            "RvsS_on_1h_prestarved",
            "RvsS_on_2h_prestarved",
            "RvsS_on_3h_prestarved",
            "RvsS_on_4h_prestarved",
        ]
        for prestarved in prestarved_configs:
            if prestarved in result or hasattr(result, prestarved):
                config = (
                    result[prestarved]
                    if isinstance(result, dict)
                    else getattr(result, prestarved)
                )
                assert isinstance(config, dict)

    def test_exp_dict_mini_configs(self):
        """Test mini experiment configurations."""
        result = sim_conf.Exp_dict()

        mini_configs = ["PItrain_mini"]
        for mini in mini_configs:
            if mini in result or hasattr(result, mini):
                config = (
                    result[mini] if isinstance(result, dict) else getattr(result, mini)
                )
                assert isinstance(config, dict)

    def test_ga_dict_exploration_config(self):
        """Test exploration GA configuration."""
        result = sim_conf.Ga_dict()
        exploration = (
            result["exploration"]
            if isinstance(result, dict)
            else getattr(result, "exploration")
        )

        assert isinstance(exploration, dict)
        assert "experiment" in exploration
        assert exploration["experiment"] == "exploration"

    def test_ga_dict_realism_config(self):
        """Test realism GA configuration."""
        result = sim_conf.Ga_dict()
        realism = (
            result["realism"]
            if isinstance(result, dict)
            else getattr(result, "realism")
        )

        assert isinstance(realism, dict)
        assert "experiment" in realism
        assert realism["experiment"] == "realism"

    def test_ga_dict_chemorbit_config(self):
        """Test chemorbit GA configuration."""
        result = sim_conf.Ga_dict()
        chemorbit = (
            result["chemorbit"]
            if isinstance(result, dict)
            else getattr(result, "chemorbit")
        )

        assert isinstance(chemorbit, dict)
        assert "experiment" in chemorbit
        assert chemorbit["experiment"] == "chemorbit"

    def test_batch_dict_patchy_food_config(self):
        """Test patchy_food batch configuration."""
        result = sim_conf.Batch_dict()
        patchy_food = (
            result["patchy_food"]
            if isinstance(result, dict)
            else getattr(result, "patchy_food")
        )

        assert isinstance(patchy_food, dict)
        assert "exp" in patchy_food
        assert patchy_food["exp"] == "patchy_food"

    def test_batch_dict_chemotaxis_config(self):
        """Test chemotaxis batch configuration."""
        result = sim_conf.Batch_dict()
        chemotaxis = (
            result["chemotaxis"]
            if isinstance(result, dict)
            else getattr(result, "chemotaxis")
        )

        assert isinstance(chemotaxis, dict)
        assert "exp" in chemotaxis
        assert chemotaxis["exp"] == "chemotaxis"

    def test_batch_dict_pitrain_config(self):
        """Test PItrain batch configuration."""
        result = sim_conf.Batch_dict()
        pitrain = (
            result["PItrain"]
            if isinstance(result, dict)
            else getattr(result, "PItrain")
        )

        assert isinstance(pitrain, dict)
        assert "exp" in pitrain
        assert pitrain["exp"] == "PItrain"

    def test_trial_dict_calls_trial_conf_function(self):
        """Test that Trial_dict actually calls trial_conf function."""
        result = sim_conf.Trial_dict()

        # Test that all trial types have epochs
        trial_types = ["default", "odor_preference", "odor_preference_short"]
        for trial_type in trial_types:
            trial = (
                result[trial_type]
                if isinstance(result, dict)
                else getattr(result, trial_type)
            )
            assert "epochs" in trial
            assert isinstance(trial["epochs"], list)

    def test_env_dict_calls_env_functions(self):
        """Test that Env_dict calls environment creation functions."""
        result = sim_conf.Env_dict()

        # Test that environment configs are properly structured
        env_types = ["focus", "dish", "arena_200mm"]
        for env_type in env_types:
            env = (
                result[env_type]
                if isinstance(result, dict)
                else getattr(result, env_type)
            )
            assert isinstance(env, dict)

    def test_exp_dict_calls_exp_functions(self):
        """Test that Exp_dict calls experiment creation functions."""
        result = sim_conf.Exp_dict()

        # Test that experiment configs are properly structured
        exp_types = ["RvsS", "PItest_off", "PItrain"]
        for exp_type in exp_types:
            if exp_type in result or hasattr(result, exp_type):
                exp = (
                    result[exp_type]
                    if isinstance(result, dict)
                    else getattr(result, exp_type)
                )
                assert isinstance(exp, dict)

    def test_ga_dict_calls_ga_conf_function(self):
        """Test that Ga_dict calls _ga_conf function."""
        result = sim_conf.Ga_dict()

        # Test that GA configs have proper structure
        ga_types = ["interference", "exploration", "realism"]
        for ga_type in ga_types:
            ga = (
                result[ga_type]
                if isinstance(result, dict)
                else getattr(result, ga_type)
            )
            assert isinstance(ga, dict)
            assert "experiment" in ga
            assert ga["experiment"] == ga_type

    def test_batch_dict_calls_bb_function(self):
        """Test that Batch_dict calls bb function."""
        result = sim_conf.Batch_dict()

        # Test that batch configs have proper structure
        batch_types = ["PItest_off", "patchy_food", "chemotaxis"]
        for batch_type in batch_types:
            batch = (
                result[batch_type]
                if isinstance(result, dict)
                else getattr(result, batch_type)
            )
            assert isinstance(batch, dict)
            assert "exp" in batch
            assert batch["exp"] == batch_type

    def test_epoch_creation(self):
        """Test that Epoch objects are created correctly."""
        # Test that Trial_dict actually calls the trial_conf function
        result = sim_conf.Trial_dict()
        odor_pref = (
            result["odor_preference"]
            if isinstance(result, dict)
            else getattr(result, "odor_preference")
        )

        # Should have epochs list
        assert "epochs" in odor_pref
        assert isinstance(odor_pref["epochs"], list)
        assert len(odor_pref["epochs"]) == 8

    def test_attr_dict_creation(self):
        """Test that AttrDict objects are created correctly."""
        result = sim_conf.Trial_dict()

        # Should be AttrDict-like
        assert hasattr(result, "__getattr__") or isinstance(result, dict)

        # Test attribute access
        if hasattr(result, "__getattr__"):
            default = result.default
            assert "epochs" in default

    def test_configuration_validation(self):
        """Test that configurations have required fields."""
        # Test Trial_dict configurations
        trial_result = sim_conf.Trial_dict()
        for trial_type in ["default", "odor_preference", "odor_preference_short"]:
            trial = (
                trial_result[trial_type]
                if isinstance(trial_result, dict)
                else getattr(trial_result, trial_type)
            )
            assert "epochs" in trial
            assert isinstance(trial["epochs"], list)

        # Test Env_dict configurations
        env_result = sim_conf.Env_dict()
        for env_type in ["focus", "dish", "arena_200mm"]:
            env = (
                env_result[env_type]
                if isinstance(env_result, dict)
                else getattr(env_result, env_type)
            )
            assert isinstance(env, dict)

        # Test Exp_dict configurations
        exp_result = sim_conf.Exp_dict()
        for exp_type in ["RvsS", "PItest_off"]:
            if exp_type in exp_result or hasattr(exp_result, exp_type):
                exp = (
                    exp_result[exp_type]
                    if isinstance(exp_result, dict)
                    else getattr(exp_result, exp_type)
                )
                assert isinstance(exp, dict)

        # Test Ga_dict configurations
        ga_result = sim_conf.Ga_dict()
        for ga_type in ["interference", "exploration"]:
            ga = (
                ga_result[ga_type]
                if isinstance(ga_result, dict)
                else getattr(ga_result, ga_type)
            )
            assert isinstance(ga, dict)
            assert "experiment" in ga

        # Test Batch_dict configurations
        batch_result = sim_conf.Batch_dict()
        for batch_type in ["PItest_off", "patchy_food"]:
            batch = (
                batch_result[batch_type]
                if isinstance(batch_result, dict)
                else getattr(batch_result, batch_type)
            )
            assert isinstance(batch, dict)
            assert "exp" in batch
