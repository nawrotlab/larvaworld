"""
Unit tests for batch_run.py module.

This module tests the batch run functionality including:
- OptimizationOps class
- BatchRun class
- space_search_sample function

Tests use mocks to isolate units and avoid heavy dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import agentpy as ap
from larvaworld.lib.sim.batch_run import (
    OptimizationOps,
    BatchRun,
    space_search_sample,
)


class TestOptimizationOps:
    """Test OptimizationOps class."""

    def test_optimization_ops_initialization(self):
        """Test OptimizationOps initialization with defaults."""
        ops = OptimizationOps()

        assert ops.fit_par is None
        assert ops.minimize == True
        assert ops.absolute == False
        assert ops.max_Nsims == 5
        assert ops.threshold == 0.001
        assert ops.operator == "mean"

    def test_optimization_ops_initialization_custom(self):
        """Test OptimizationOps initialization with custom values."""
        ops = OptimizationOps(
            fit_par="velocity",
            minimize=False,
            absolute=True,
            max_Nsims=10,
            threshold=0.01,
            operator="std",
        )

        assert ops.fit_par == "velocity"
        assert ops.minimize == False
        assert ops.absolute == True
        assert ops.max_Nsims == 10
        assert ops.threshold == 0.01
        assert ops.operator == "std"

    def test_threshold_reached_minimize_true(self):
        """Test threshold_reached when minimize=True."""
        ops = OptimizationOps(minimize=True, threshold=0.5)

        # Test with values below threshold
        fits = np.array([0.3, 0.4, 0.6])
        assert ops.threshold_reached(fits) == True

        # Test with values above threshold
        fits = np.array([0.6, 0.7, 0.8])
        assert ops.threshold_reached(fits) == False

        # Test with NaN values
        fits = np.array([np.nan, 0.3, 0.6])
        assert ops.threshold_reached(fits) == True

    def test_threshold_reached_minimize_false(self):
        """Test threshold_reached when minimize=False."""
        ops = OptimizationOps(minimize=False, threshold=0.5)

        # Test with values above threshold
        fits = np.array([0.6, 0.7, 0.8])
        assert ops.threshold_reached(fits) == True

        # Test with values below threshold (max is 0.6 which is still >= 0.5)
        fits = np.array([0.3, 0.4, 0.45])
        assert ops.threshold_reached(fits) == False

        # Test with NaN values
        fits = np.array([np.nan, 0.6, 0.3])
        assert ops.threshold_reached(fits) == True

    def test_check_max_sims_reached(self):
        """Test check method when max simulations reached."""
        ops = OptimizationOps(max_Nsims=3)
        fits = np.array([0.1, 0.2, 0.3, 0.4])  # 4 values >= max_Nsims

        with patch("larvaworld.lib.sim.batch_run.vprint") as mock_vprint:
            ops.check(fits)
            mock_vprint.assert_called_once_with(
                "Maximum number of simulations reached. Halting search", 2
            )

    def test_check_threshold_reached(self):
        """Test check method when threshold reached."""
        ops = OptimizationOps(minimize=True, threshold=0.5)
        fits = np.array([0.3, 0.4])  # min value below threshold

        with patch("larvaworld.lib.sim.batch_run.vprint") as mock_vprint:
            ops.check(fits)
            mock_vprint.assert_called_once_with(
                "Best result reached threshold. Halting search", 2
            )

    def test_check_threshold_not_reached(self):
        """Test check method when threshold not reached."""
        ops = OptimizationOps(minimize=True, threshold=0.1, max_Nsims=10)
        fits = np.array([0.3, 0.4])  # min value above threshold, less than max_Nsims

        with patch("larvaworld.lib.sim.batch_run.vprint") as mock_vprint:
            ops.check(fits)
            mock_vprint.assert_called_once_with(
                "Not reached threshold. Expanding space search", 2
            )


class TestBatchRun:
    """Test BatchRun class."""

    @pytest.fixture
    def mock_batch_run(self):
        """Create a mock BatchRun instance."""
        with (
            patch("larvaworld.lib.sim.batch_run.ap.Experiment") as mock_ap_experiment,
            patch(
                "larvaworld.lib.sim.batch_run.reg.generators.SimConfiguration"
            ) as mock_sim_conf,
            patch(
                "larvaworld.lib.sim.batch_run.reg.conf.Exp.expand"
            ) as mock_exp_expand,
        ):
            # Setup mocks
            mock_ap_experiment.return_value = Mock()
            mock_sim_conf.return_value = Mock()

            # Mock exp_conf with update method
            mock_exp_conf = Mock()
            mock_exp_conf.update = Mock()
            mock_exp_expand.return_value = mock_exp_conf

            batch_run = BatchRun(
                experiment="test_experiment",
                space_search={"param1": 1.0, "param2": {"values": [1, 2, 3]}},
                id="test_batch",
                space_kws={"n": 10},
                exp="test_exp",
                exp_kws={"duration": 100},
                store_data=True,
            )

            # Set mock attributes (avoid setting properties without setters)
            # batch_run.dir and plot_dir are properties from parent classes
            batch_run.datasets = {}
            batch_run.results = None
            batch_run.figs = {}
            batch_run.par_df = pd.DataFrame(
                {"param1": [1, 2, 3], "fit": [0.1, 0.2, 0.3]}
            )
            batch_run.par_names = ["param1"]
            batch_run.optimization = OptimizationOps()

            return batch_run

    def test_batch_run_initialization(self, mock_batch_run):
        """Test BatchRun initialization."""
        # BatchRun.experiment is set via parent class SimConfiguration
        assert mock_batch_run.datasets == {}
        assert mock_batch_run.results is None
        assert mock_batch_run.figs == {}

    def test_single_sim(self, mock_batch_run):
        """Test _single_sim method."""
        mock_batch_run.model = Mock()
        mock_batch_run.sample = [{"param1": 1.0}]
        mock_batch_run.exp_conf = Mock()
        mock_batch_run.exp_conf.update_existingnestdict_by_suffix.return_value = {
            "param1": 1.0
        }
        mock_batch_run._model_kwargs = {}
        mock_batch_run._random = None
        mock_batch_run.record = False

        mock_model_instance = Mock()
        mock_model_instance.simulate.return_value = [Mock()]
        mock_model_instance.output = {"test": "output"}
        mock_batch_run.model.return_value = mock_model_instance

        result = mock_batch_run._single_sim((0, "run1"))

        # Check that model was created with correct parameters
        mock_batch_run.model.assert_called_once_with(
            parameters={"param1": 1.0}, _run_id=(0, "run1"), **{}
        )

        # Check that simulate was called
        mock_model_instance.simulate.assert_called_once_with(display=False, seed=None)

        # Check that datasets was updated
        assert 0 in mock_batch_run.datasets

        # Check return value
        assert result == {"test": "output"}

    def test_single_sim_with_random_seed(self, mock_batch_run):
        """Test _single_sim method with random seed."""
        mock_batch_run.model = Mock()
        mock_batch_run.sample = [{"param1": 1.0}]
        mock_batch_run.exp_conf = Mock()
        mock_batch_run.exp_conf.update_existingnestdict_by_suffix.return_value = {
            "param1": 1.0
        }
        mock_batch_run._model_kwargs = {}
        mock_batch_run._random = {(0, "run1"): 42}
        mock_batch_run.record = False

        mock_model_instance = Mock()
        mock_model_instance.simulate.return_value = [Mock()]
        mock_model_instance.output = {"test": "output"}
        mock_batch_run.model.return_value = mock_model_instance

        result = mock_batch_run._single_sim((0, "run1"))

        # Check that simulate was called with seed
        mock_model_instance.simulate.assert_called_once_with(display=False, seed=42)

    def test_single_sim_with_variables_removal(self, mock_batch_run):
        """Test _single_sim method with variables removal."""
        mock_batch_run.model = Mock()
        mock_batch_run.sample = [{"param1": 1.0}]
        mock_batch_run.exp_conf = Mock()
        mock_batch_run.exp_conf.update_existingnestdict_by_suffix.return_value = {
            "param1": 1.0
        }
        mock_batch_run._model_kwargs = {}
        mock_batch_run._random = None
        mock_batch_run.record = False

        mock_model_instance = Mock()
        mock_model_instance.simulate.return_value = [Mock()]
        mock_model_instance.output = {
            "test": "output",
            "variables": {"var1": [1, 2, 3]},
        }
        mock_batch_run.model.return_value = mock_model_instance

        result = mock_batch_run._single_sim((0, "run1"))

        # Check that variables were removed from output
        assert "variables" not in mock_model_instance.output

    def test_default_processing_end_ps(self, mock_batch_run):
        """Test default_processing with end_ps parameter."""
        mock_batch_run.optimization.fit_par = "velocity"
        mock_batch_run.optimization.absolute = False
        mock_batch_run.optimization.operator = "mean"

        mock_dataset = Mock()
        mock_dataset.end_ps = ["velocity"]
        mock_dataset.step_ps = []

        # Mock e as a dict-like object
        mock_velocity = Mock()
        mock_velocity.values = np.array([1.0, 2.0, 3.0])
        mock_dataset.e = {"velocity": mock_velocity}

        result = mock_batch_run.default_processing(mock_dataset)

        assert result == 2.0  # mean of [1.0, 2.0, 3.0]

    def test_default_processing_step_ps(self, mock_batch_run):
        """Test default_processing with step_ps parameter."""
        mock_batch_run.optimization.fit_par = "velocity"
        mock_batch_run.optimization.absolute = False
        mock_batch_run.optimization.operator = "mean"

        mock_dataset = Mock()
        mock_dataset.end_ps = []
        mock_dataset.step_ps = ["velocity"]

        # Mock s as a dict-like object
        mock_velocity = Mock()
        mock_velocity.groupby.return_value.mean.return_value = np.array([1.0, 2.0, 3.0])
        mock_dataset.s = {"velocity": mock_velocity}

        result = mock_batch_run.default_processing(mock_dataset)

        assert result == 2.0  # mean of [1.0, 2.0, 3.0]

    def test_default_processing_absolute(self, mock_batch_run):
        """Test default_processing with absolute values."""
        mock_batch_run.optimization.fit_par = "velocity"
        mock_batch_run.optimization.absolute = True
        mock_batch_run.optimization.operator = "mean"

        mock_dataset = Mock()
        mock_dataset.end_ps = ["velocity"]
        mock_dataset.step_ps = []

        # Mock e as a dict-like object
        mock_velocity = Mock()
        mock_velocity.values = np.array([-1.0, -2.0, 3.0])
        mock_dataset.e = {"velocity": mock_velocity}

        result = mock_batch_run.default_processing(mock_dataset)

        assert result == 2.0  # mean of abs([-1.0, -2.0, 3.0])

    def test_default_processing_std_operator(self, mock_batch_run):
        """Test default_processing with std operator."""
        mock_batch_run.optimization.fit_par = "velocity"
        mock_batch_run.optimization.absolute = False
        mock_batch_run.optimization.operator = "std"

        mock_dataset = Mock()
        mock_dataset.end_ps = ["velocity"]
        mock_dataset.step_ps = []

        # Mock e as a dict-like object
        mock_velocity = Mock()
        mock_velocity.values = np.array([1.0, 2.0, 3.0])
        mock_dataset.e = {"velocity": mock_velocity}

        result = mock_batch_run.default_processing(mock_dataset)

        assert result == np.std([1.0, 2.0, 3.0])

    def test_default_processing_parameter_not_found(self, mock_batch_run):
        """Test default_processing when parameter not found."""
        mock_batch_run.optimization.fit_par = "nonexistent"

        mock_dataset = Mock()
        mock_dataset.end_ps = ["velocity"]
        mock_dataset.step_ps = ["position"]

        with pytest.raises(
            ValueError, match="Could not retrieve fit parameter from dataset"
        ):
            mock_batch_run.default_processing(mock_dataset)

    # Note: test_default_processing_invalid_operator is not testable
    # because param.Selector enforces valid values and doesn't allow
    # setting or patching invalid values. The else branch in default_processing
    # (line 206) is unreachable in practice due to param validation.

    def test_end(self, mock_batch_run):
        """Test end method."""
        mock_batch_run.output = Mock()
        mock_batch_run.output._combine_pars.return_value = pd.DataFrame(
            {"param1": [1, 2, 3]}
        )
        mock_batch_run.datasets = {0: [Mock()], 1: [Mock()], 2: [Mock()]}
        mock_batch_run.default_processing = Mock(return_value=0.5)
        mock_batch_run.optimization.check = Mock()

        mock_batch_run.end()

        # Check that _combine_pars was called
        mock_batch_run.output._combine_pars.assert_called_once()

        # Check that par_df was set
        assert "fit" in mock_batch_run.par_df.columns
        assert len(mock_batch_run.par_df) == 3

        # Check that default_processing was called for each dataset
        assert mock_batch_run.default_processing.call_count == 3

        # Check that optimization.check was called
        mock_batch_run.optimization.check.assert_called_once()

    def test_end_exception_handling(self, mock_batch_run):
        """Test end method with exception handling."""
        mock_batch_run.output = Mock()
        mock_batch_run.output._combine_pars.return_value = pd.DataFrame(
            {"param1": [1, 2, 3]}
        )
        mock_batch_run.datasets = {0: [Mock()], 1: [Mock()], 2: [Mock()]}
        mock_batch_run.default_processing = Mock(side_effect=Exception("Test error"))
        mock_batch_run.optimization.check = Mock()

        # Should not raise exception
        mock_batch_run.end()

        # Check that _combine_pars was called
        mock_batch_run.output._combine_pars.assert_called_once()

    def test_simulate(self, mock_batch_run):
        """Test simulate method."""
        mock_batch_run.run = Mock()
        mock_batch_run.experiment = "test_experiment"
        mock_batch_run.PI_heatmap = Mock()
        mock_batch_run.plot_results = Mock()
        mock_batch_run.par_df = pd.DataFrame({"param1": [1, 2, 3]})
        mock_batch_run.figs = {"fig1": Mock()}

        with patch("larvaworld.lib.sim.batch_run.util.storeH5") as mock_store_h5:
            result = mock_batch_run.simulate(n_jobs=4)

            # Check that run was called
            mock_batch_run.run.assert_called_once_with(n_jobs=4)

            # Check that PI_heatmap was not called (experiment doesn't contain "PI")
            mock_batch_run.PI_heatmap.assert_not_called()

            # Check that plot_results was called
            mock_batch_run.plot_results.assert_called_once()

            # Check that storeH5 was called
            mock_store_h5.assert_called_once()

            # Check return value
            assert result == (mock_batch_run.par_df, mock_batch_run.figs)

    def test_simulate_with_PI(self, mock_batch_run):
        """Test simulate method with PI experiment."""
        mock_batch_run.run = Mock()
        mock_batch_run.experiment = "test_PI_experiment"
        mock_batch_run.PI_heatmap = Mock()
        mock_batch_run.plot_results = Mock()
        mock_batch_run.par_df = pd.DataFrame({"param1": [1, 2, 3]})
        mock_batch_run.figs = {"fig1": Mock()}

        with patch("larvaworld.lib.sim.batch_run.util.storeH5") as mock_store_h5:
            result = mock_batch_run.simulate(n_jobs=4)

            # Check that PI_heatmap was called
            mock_batch_run.PI_heatmap.assert_called_once()

    def test_plot_results_single_parameter(self, mock_batch_run):
        """Test plot_results with single parameter."""
        mock_batch_run.par_df = pd.DataFrame(
            {"param1": [1, 2, 3], "target1": [0.1, 0.2, 0.3]}
        )
        mock_batch_run.par_names = ["param1"]
        mock_batch_run.figs = {}

        with (
            patch("larvaworld.lib.sim.batch_run.plot_2d") as mock_plot_2d,
            patch.object(
                type(mock_batch_run),
                "plot_dir",
                new_callable=lambda: property(lambda self: "/test/plots"),
            ),
        ):
            mock_plot_2d.return_value = Mock()

            mock_batch_run.plot_results()

            # Check that plot_2d was called
            mock_plot_2d.assert_called_once_with(
                labels=["param1", "target1"],
                pref="target1",
                df=mock_batch_run.par_df,
                save_to="/test/plots",
                show=True,
            )

    def test_plot_results_two_parameters(self, mock_batch_run):
        """Test plot_results with two parameters."""
        mock_batch_run.par_df = pd.DataFrame(
            {"param1": [1, 2, 3], "param2": [4, 5, 6], "target1": [0.1, 0.2, 0.3]}
        )
        mock_batch_run.par_names = ["param1", "param2"]
        mock_batch_run.figs = {}

        with (
            patch("larvaworld.lib.sim.batch_run.plot_3pars") as mock_plot_3pars,
            patch.object(
                type(mock_batch_run),
                "plot_dir",
                new_callable=lambda: property(lambda self: "/test/plots"),
            ),
        ):
            mock_plot_3pars.return_value = {"fig1": Mock(), "fig2": Mock()}

            mock_batch_run.plot_results()

            # Check that plot_3pars was called
            mock_plot_3pars.assert_called_once_with(
                vars=["param1", "param2"],
                target="target1",
                pref="target1",
                df=mock_batch_run.par_df,
                save_to="/test/plots",
                show=True,
            )

    def test_plot_results_multiple_parameters(self, mock_batch_run):
        """Test plot_results with multiple parameters."""
        mock_batch_run.par_df = pd.DataFrame(
            {
                "param1": [1, 2, 3],
                "param2": [4, 5, 6],
                "param3": [7, 8, 9],
                "target1": [0.1, 0.2, 0.3],
            }
        )
        mock_batch_run.par_names = ["param1", "param2", "param3"]
        mock_batch_run.figs = {}

        with (
            patch("larvaworld.lib.sim.batch_run.plot_3pars") as mock_plot_3pars,
            patch.object(
                type(mock_batch_run),
                "plot_dir",
                new_callable=lambda: property(lambda self: "/test/plots"),
            ),
        ):
            mock_plot_3pars.return_value = {"fig1": Mock(), "fig2": Mock()}

            mock_batch_run.plot_results()

            # Check that plot_3pars was called for each pair
            assert mock_plot_3pars.call_count == 3  # 3 pairs from 3 parameters

    def test_PI_heatmap(self, mock_batch_run):
        """Test PI_heatmap method."""
        mock_batch_run.par_df = pd.DataFrame({"param1": [1, 2, 3], "param2": [4, 5, 6]})
        mock_batch_run.datasets = {0: [Mock()], 1: [Mock()], 2: [Mock()]}

        # Setup mock datasets with PI config
        for i in range(3):
            mock_batch_run.datasets[i][0].config = Mock()
            mock_batch_run.datasets[i][0].config.PI = {"PI": 0.1 + i * 0.1}

        with (
            patch("larvaworld.lib.sim.batch_run.plot_heatmap_PI") as mock_plot_heatmap,
            patch("larvaworld.lib.sim.batch_run.pd.DataFrame") as mock_dataframe,
            patch.object(
                type(mock_batch_run),
                "plot_dir",
                new_callable=lambda: property(lambda self: "/test/plots"),
            ),
        ):
            mock_dataframe.return_value.to_csv.return_value = None
            mock_plot_heatmap.return_value = Mock()

            mock_batch_run.PI_heatmap()

            # Check that DataFrame was created
            mock_dataframe.assert_called_once()

            # Check that to_csv was called
            mock_dataframe.return_value.to_csv.assert_called_once_with(
                "/test/plots/PIs.csv", index=True, header=True
            )

            # Check that plot_heatmap_PI was called
            mock_plot_heatmap.assert_called_once()


class TestSpaceSearchSample:
    """Test space_search_sample function."""

    def test_space_search_sample_direct_values(self):
        """Test space_search_sample with direct values."""
        space_dict = {"param1": 1.0, "param2": "test", "param3": True}

        with patch("larvaworld.lib.sim.batch_run.ap.Sample") as mock_sample:
            result = space_search_sample(space_dict)

            # Check that ap.Sample was called with correct parameters
            mock_sample.assert_called_once_with(
                {"param1": 1.0, "param2": "test", "param3": True}, n=1
            )

            assert result == mock_sample.return_value

    def test_space_search_sample_discrete_values(self):
        """Test space_search_sample with discrete values."""
        space_dict = {
            "param1": {"values": [1, 2, 3, 4, 5]},
            "param2": {"values": ["a", "b", "c"]},
        }

        with (
            patch("larvaworld.lib.sim.batch_run.ap.Sample") as mock_sample,
            patch("larvaworld.lib.sim.batch_run.ap.Values") as mock_values,
        ):
            mock_values.return_value = Mock()
            result = space_search_sample(space_dict)

            # Check that ap.Values was called for each parameter
            assert mock_values.call_count == 2

            # Check that ap.Sample was called
            mock_sample.assert_called_once()

    def test_space_search_sample_continuous_range(self):
        """Test space_search_sample with continuous range."""
        space_dict = {
            "param1": {"range": (0.0, 1.0)},
            "param2": {"range": (10.0, 20.0)},
        }

        with (
            patch("larvaworld.lib.sim.batch_run.ap.Sample") as mock_sample,
            patch("larvaworld.lib.sim.batch_run.ap.Range") as mock_range,
        ):
            mock_range.return_value = Mock()
            result = space_search_sample(space_dict)

            # Check that ap.Range was called for each parameter
            assert mock_range.call_count == 2

            # Check that ap.Sample was called
            mock_sample.assert_called_once()

    def test_space_search_sample_int_range(self):
        """Test space_search_sample with integer range."""
        space_dict = {"param1": {"range": (1, 10)}, "param2": {"range": (100, 200)}}

        with (
            patch("larvaworld.lib.sim.batch_run.ap.Sample") as mock_sample,
            patch("larvaworld.lib.sim.batch_run.ap.IntRange") as mock_int_range,
        ):
            mock_int_range.return_value = Mock()
            result = space_search_sample(space_dict)

            # Check that ap.IntRange was called for each parameter
            assert mock_int_range.call_count == 2

            # Check that ap.Sample was called
            mock_sample.assert_called_once()

    def test_space_search_sample_grid_range(self):
        """Test space_search_sample with grid range."""
        space_dict = {
            "param1": {"range": (0.0, 1.0), "Ngrid": 5},
            "param2": {"range": (1, 10), "Ngrid": 3},
        }

        with (
            patch("larvaworld.lib.sim.batch_run.ap.Sample") as mock_sample,
            patch("larvaworld.lib.sim.batch_run.ap.Values") as mock_values,
            patch("larvaworld.lib.sim.batch_run.np.linspace") as mock_linspace,
        ):
            mock_linspace.return_value = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
            mock_values.return_value = Mock()
            result = space_search_sample(space_dict)

            # Check that np.linspace was called for each parameter
            assert mock_linspace.call_count == 2

            # Check that ap.Values was called for each parameter
            assert mock_values.call_count == 2

            # Check that ap.Sample was called
            mock_sample.assert_called_once()

    def test_space_search_sample_mixed_types(self):
        """Test space_search_sample with mixed parameter types."""
        space_dict = {
            "param1": 1.0,  # Direct value
            "param2": {"values": [1, 2, 3]},  # Discrete values
            "param3": {"range": (0.0, 1.0)},  # Continuous range
            "param4": {"range": (1, 10), "Ngrid": 5},  # Grid range
        }

        with (
            patch("larvaworld.lib.sim.batch_run.ap.Sample") as mock_sample,
            patch("larvaworld.lib.sim.batch_run.ap.Values") as mock_values,
            patch("larvaworld.lib.sim.batch_run.ap.Range") as mock_range,
            patch("larvaworld.lib.sim.batch_run.np.linspace") as mock_linspace,
        ):
            mock_linspace.return_value = np.array([1, 3, 5, 7, 9])
            mock_values.return_value = Mock()
            mock_range.return_value = Mock()
            result = space_search_sample(space_dict)

            # Check that ap.Sample was called
            mock_sample.assert_called_once()

    def test_space_search_sample_custom_n(self):
        """Test space_search_sample with custom n parameter."""
        space_dict = {"param1": 1.0}

        with patch("larvaworld.lib.sim.batch_run.ap.Sample") as mock_sample:
            result = space_search_sample(space_dict, n=5, custom_kwarg="test")

            # Check that ap.Sample was called with custom n
            mock_sample.assert_called_once_with(
                {"param1": 1.0}, n=5, custom_kwarg="test"
            )
