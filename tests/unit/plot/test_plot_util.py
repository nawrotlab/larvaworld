"""
Unit tests for plot.util module - utility plotting functions.

Tests REAL behavior based on ACTUAL code, not assumptions.
All tests validate actual API signatures, return values, and side effects.
"""

import pytest
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from larvaworld.lib.plot.util import (
    configure_subplot_grid,
    plot_mean_and_range,
    plot_quantiles,
    pvalue_star,
    save_plot,
    process_plot,
    circular_hist,
    confidence_ellipse,
    dataset_legend,
    label_diff,
    dual_half_circle,
    prob_hist,
    define_end_ks,
    color_epochs,
)


@pytest.mark.fast
class TestConfigureSubplotGrid:
    """Test configure_subplot_grid function for grid dimension calculation."""

    def test_configure_subplot_grid_basic(self):
        """Test basic grid configuration with N elements."""
        # Lines 810-833: Basic functionality
        result = configure_subplot_grid(N=6, wh=5)

        assert "nrows" in result
        assert "ncols" in result
        assert "figsize" in result
        assert "sharex" in result
        assert "sharey" in result

        # For N=6: should be 2 rows x 3 cols or 3 rows x 2 cols
        assert result["nrows"] * result["ncols"] >= 6

    def test_configure_subplot_grid_with_ncols(self):
        """Test grid configuration with specified Ncols."""
        # Lines 814-816: if Ncols: Nrows = -(-N // Ncols)
        result = configure_subplot_grid(N=10, Ncols=3, wh=4)

        assert result["ncols"] == 3
        assert result["nrows"] == 4  # ceil(10/3) = 4
        assert result["figsize"] == (12, 16)  # 4*3, 4*4

    def test_configure_subplot_grid_with_nrows(self):
        """Test grid configuration with specified Nrows."""
        # Lines 811-813: if Nrows: Ncols = -(-N // Nrows)
        result = configure_subplot_grid(N=10, Nrows=2, wh=5)

        assert result["nrows"] == 2
        assert result["ncols"] == 5  # ceil(10/2) = 5
        assert result["figsize"] == (25, 10)  # 5*5, 5*2

    def test_configure_subplot_grid_with_nrows_coef(self):
        """Test grid configuration with Nrows coefficient."""
        # Line 821-822: if Nrows is not None: Nrows *= Nrows_coef
        result = configure_subplot_grid(N=6, Nrows=2, Nrows_coef=2, wh=4)

        assert result["nrows"] == 4  # 2 * 2
        assert result["ncols"] == 2  # ceil(6/4) = 2

    def test_configure_subplot_grid_with_wh(self):
        """Test grid configuration with wh parameter."""
        # Line 824: figsize = (wh * Ncols, wh * Nrows) if wh
        result = configure_subplot_grid(N=4, wh=6)

        nrows, ncols = result["nrows"], result["ncols"]
        assert result["figsize"] == (6 * ncols, 6 * nrows)

    def test_configure_subplot_grid_with_w_h(self):
        """Test grid configuration with separate w and h parameters."""
        # Line 824: else (w * Ncols, h * Nrows)
        result = configure_subplot_grid(N=4, w=10, h=8)

        nrows, ncols = result["nrows"], result["ncols"]
        assert result["figsize"] == (10 * ncols, 8 * nrows)

    def test_configure_subplot_grid_sharex_sharey(self):
        """Test grid configuration with axis sharing."""
        # Lines 825-831: sharex, sharey parameters
        result = configure_subplot_grid(N=4, sharex=True, sharey=True)

        assert result["sharex"] is True
        assert result["sharey"] is True

    def test_configure_subplot_grid_explicit_figsize(self):
        """Test grid configuration with explicit figsize."""
        # Line 824: figsize = figsize or ...
        result = configure_subplot_grid(N=4, figsize=(20, 15), wh=5)

        # Explicit figsize should override wh calculation
        assert result["figsize"] == (20, 15)

    def test_configure_subplot_grid_square_root(self):
        """Test grid uses square root for auto-sizing."""
        # Line 818: Nrows, Ncols = (int(N**0.5), -(-N // int(N**0.5)))
        result = configure_subplot_grid(N=9, wh=3)

        # 9 elements: sqrt(9) = 3, so 3x3 grid
        assert result["nrows"] == 3
        assert result["ncols"] == 3


@pytest.mark.fast
class TestPvaluestar:
    """Test pvalue_star function for significance star conversion."""

    def test_pvalue_star_ns(self):
        """Test non-significant p-value returns 'ns'."""
        # Line 437-441: pvalue_star logic with dictionary
        result = pvalue_star(0.1)
        assert result == "ns"

        result = pvalue_star(0.06)
        assert result == "ns"

        result = pvalue_star(1.0)
        assert result == "ns"

    def test_pvalue_star_one_star(self):
        """Test p < 0.05 returns '*'."""
        result = pvalue_star(0.04)
        assert result == "*"

        result = pvalue_star(0.02)
        assert result == "*"

    def test_pvalue_star_two_stars(self):
        """Test p < 0.01 returns '**'."""
        result = pvalue_star(0.009)
        assert result == "**"

        result = pvalue_star(0.005)
        assert result == "**"

    def test_pvalue_star_three_stars(self):
        """Test p < 0.001 returns '***'."""
        result = pvalue_star(0.0002)
        assert result == "***"

        result = pvalue_star(0.0005)
        assert result == "***"

    def test_pvalue_star_four_stars(self):
        """Test p < 0.0001 returns '****'."""
        result = pvalue_star(0.00009)
        assert result == "****"

        result = pvalue_star(0.00001)
        assert result == "****"


@pytest.mark.fast
class TestPlotMeanAndRange:
    """Test plot_mean_and_range function."""

    def test_plot_mean_and_range_basic(self):
        """Test basic plotting of mean and range."""
        fig, ax = plt.subplots()
        x = np.array([0, 1, 2, 3, 4])
        mean = np.array([1, 2, 3, 4, 5])
        lb = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        ub = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        # Lines 91-123: plot_mean_and_range
        plot_mean_and_range(x=x, mean=mean, lb=lb, ub=ub, axis=ax, color="blue")

        # Should create line and fill_between
        assert len(ax.lines) >= 1
        assert len(ax.collections) >= 1

        plt.close(fig)

    def test_plot_mean_and_range_with_label(self):
        """Test plotting with label for legend."""
        fig, ax = plt.subplots()
        x = np.array([0, 1, 2])
        mean = np.array([1, 2, 3])
        lb = np.array([0.5, 1.5, 2.5])
        ub = np.array([1.5, 2.5, 3.5])

        # Lines 91-123: with label parameter
        plot_mean_and_range(
            x=x, mean=mean, lb=lb, ub=ub, axis=ax, color="red", label="Test"
        )

        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_plot_mean_and_range_with_color_mean(self):
        """Test plotting with separate color for mean."""
        fig, ax = plt.subplots()
        x = np.array([0, 1, 2])
        mean = np.array([1, 2, 3])
        lb = np.array([0.5, 1.5, 2.5])
        ub = np.array([1.5, 2.5, 3.5])

        # Lines 91-123: color_mean parameter
        plot_mean_and_range(
            x=x, mean=mean, lb=lb, ub=ub, axis=ax, color="blue", color_mean="green"
        )

        assert len(ax.lines) >= 1
        plt.close(fig)


@pytest.mark.fast
class TestPlotQuantiles:
    """Test plot_quantiles function."""

    def test_plot_quantiles_numpy_array(self):
        """Test plotting quantiles from numpy array."""
        fig, ax = plt.subplots()
        data = np.random.rand(10, 20)  # 10 samples, 20 time points

        # Lines 76-88: numpy array branch
        plot_quantiles(df=data, axis=ax, color="blue")

        # Should create plot elements
        assert len(ax.lines) >= 1 or len(ax.collections) >= 1
        plt.close(fig)

    def test_plot_quantiles_with_x(self):
        """Test plotting quantiles with custom x values."""
        fig, ax = plt.subplots()
        data = np.random.rand(5, 10)
        x = np.linspace(0, 1, 10)

        # Lines 76-88: with x parameter
        plot_quantiles(df=data, x=x, axis=ax, color="red")

        assert len(ax.lines) >= 1 or len(ax.collections) >= 1
        plt.close(fig)


@pytest.mark.fast
class TestSavePlot:
    """Test save_plot function."""

    def test_save_plot_creates_file(self, tmp_path):
        """Test save_plot saves figure to filepath."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        # Lines 568-590: save_plot(fig, filepath, filename)
        # filepath is the FULL path including filename
        filepath = str(tmp_path / "test_plot.png")
        save_plot(fig=fig, filepath=filepath, filename="test_plot.png")

        saved_file = tmp_path / "test_plot.png"
        assert saved_file.exists()

        plt.close(fig)


@pytest.mark.fast
class TestProcessPlot:
    """Test process_plot function."""

    def test_process_plot_with_save_to(self, tmp_path):
        """Test process_plot saves figure when save_to is provided."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        # Lines 592-634: process_plot(fig, save_to, filename, return_fig, show)
        result = process_plot(
            fig=fig, save_to=str(tmp_path), filename="output.png", return_fig=False
        )

        saved_file = tmp_path / "output.png"
        assert saved_file.exists()
        assert isinstance(result, Figure)

        plt.close(fig)

    def test_process_plot_without_save_to(self):
        """Test process_plot does not save when save_to is None."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        # Lines 592-634: save_to=None means no save
        result = process_plot(
            fig=fig, save_to=None, filename="test.png", return_fig=False
        )

        # Should return figure
        assert isinstance(result, Figure)

        plt.close(fig)

    def test_process_plot_return_fig_true(self, tmp_path):
        """Test process_plot returns tuple when return_fig=True."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])

        # Lines 626-627: if return_fig: return fig, save_to, filename
        result = process_plot(
            fig=fig, save_to=str(tmp_path), filename="test.png", return_fig=True
        )

        # Should return tuple
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], Figure)

        plt.close(fig)


@pytest.mark.fast
class TestCircularHist:
    """Test circular_hist function for polar histograms."""

    def test_circular_hist_basic(self):
        """Test basic circular histogram creation."""
        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
        angles = np.random.uniform(-np.pi, np.pi, 100)

        # Lines 148-225: circular_hist function
        n, bins, patches = circular_hist(ax, angles, bins=16, density=True)

        # Should return histogram data
        assert isinstance(n, np.ndarray)
        assert isinstance(bins, np.ndarray)
        assert len(n) > 0
        assert len(bins) == len(n) + 1

        plt.close(fig)

    def test_circular_hist_with_gaps_false(self):
        """Test circular histogram partitions full circle when gaps=False."""
        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
        angles = np.random.uniform(-np.pi, np.pi, 50)

        # Lines 185-186: if not gaps: bins partition entire circle
        n, bins, patches = circular_hist(ax, angles, bins=8, gaps=False)

        # Bins should cover [-pi, pi]
        assert bins[0] == pytest.approx(-np.pi, abs=0.01)
        assert bins[-1] == pytest.approx(np.pi, abs=0.01)

        plt.close(fig)

    def test_circular_hist_density_false(self):
        """Test circular histogram with frequency proportional to radius."""
        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
        angles = np.random.uniform(-np.pi, np.pi, 30)

        # Lines 200-202: density=False uses raw counts
        n, bins, patches = circular_hist(ax, angles, bins=12, density=False)

        assert isinstance(n, np.ndarray)
        plt.close(fig)


@pytest.mark.fast
class TestConfidenceEllipse:
    """Test confidence_ellipse function."""

    def test_confidence_ellipse_basic(self):
        """Test confidence ellipse creation."""
        fig, ax = plt.subplots()
        x = np.random.randn(100)
        y = np.random.randn(100)

        # Lines 268-333: confidence_ellipse function
        ellipse = confidence_ellipse(x, y, ax, n_std=2.0)

        # Should return a patch
        assert ellipse is not None
        plt.close(fig)

    def test_confidence_ellipse_mismatched_size_raises(self):
        """Test confidence_ellipse raises on mismatched input sizes."""
        fig, ax = plt.subplots()
        x = np.array([1, 2, 3])
        y = np.array([1, 2])  # Different size

        # Line 298-299: raises ValueError
        with pytest.raises(ValueError, match="x and y must be the same size"):
            confidence_ellipse(x, y, ax)

        plt.close(fig)


@pytest.mark.fast
class TestDatasetLegend:
    """Test dataset_legend function."""

    def test_dataset_legend_basic(self):
        """Test dataset legend creation."""
        fig, ax = plt.subplots()
        labels = ["Dataset 1", "Dataset 2"]
        colors = ["blue", "red"]

        # Lines 336-384: dataset_legend function
        leg = dataset_legend(labels, colors, ax=ax)

        assert leg is not None
        plt.close(fig)

    def test_dataset_legend_without_ax(self):
        """Test dataset legend uses current axes when ax=None."""
        plt.figure()
        labels = ["A", "B"]
        colors = ["green", "yellow"]

        # Lines 378-380: if ax is None: use plt.legend
        leg = dataset_legend(labels, colors, ax=None)

        assert leg is not None
        plt.close("all")


@pytest.mark.fast
class TestLabelDiff:
    """Test label_diff function for annotation."""

    def test_label_diff_creates_annotation(self):
        """Test label_diff adds annotation to axes."""
        fig, ax = plt.subplots()
        X = [1, 2, 3]
        Y = [5, 8, 6]

        # Lines 387-417: label_diff function
        label_diff(i=0, j=2, text="***", X=X, Y=Y, ax=ax)

        # Should have added annotations
        assert len(ax.texts) > 0
        plt.close(fig)


@pytest.mark.fast
class TestDualHalfCircle:
    """Test dual_half_circle function."""

    def test_dual_half_circle_basic(self):
        """Test dual half circle creation."""
        fig, ax = plt.subplots()

        # Lines 529-565: dual_half_circle function
        # Default colors are ('W', 'k') but matplotlib needs lowercase 'w'
        wedges = dual_half_circle(
            center=(0.5, 0.5), radius=0.05, angle=90, ax=ax, colors=("white", "black")
        )

        # Should return list of 2 wedges
        assert isinstance(wedges, list)
        assert len(wedges) == 2
        plt.close(fig)

    def test_dual_half_circle_without_ax(self):
        """Test dual half circle uses current axes when ax=None."""
        plt.figure()
        plt.subplot()

        # Lines 557-559: if ax is None: ax = plt.gca()
        wedges = dual_half_circle(
            center=(0, 0), radius=0.1, ax=None, colors=("red", "blue")
        )

        assert len(wedges) == 2
        plt.close("all")


@pytest.mark.fast
class TestProbHist:
    """Test prob_hist function."""

    def test_prob_hist_sns_hist(self):
        """Test probability histogram with seaborn."""
        fig, ax = plt.subplots()
        data1 = np.random.randn(100)
        data2 = np.random.randn(100)

        # Lines 636-693: prob_hist with sns.hist
        prob_hist(
            vs=[data1, data2],
            colors=["blue", "red"],
            labels=["Data1", "Data2"],
            bins=20,
            ax=ax,
            hist_type="sns.hist",
        )

        # Seaborn histplot creates collections (PolyCollection), not patches
        assert len(ax.collections) > 0 or len(ax.patches) > 0 or len(ax.lines) > 0
        plt.close(fig)

    def test_prob_hist_plt_hist(self):
        """Test probability histogram with matplotlib."""
        fig, ax = plt.subplots()
        data = np.random.randn(50)

        # Lines 684-692: prob_hist with plt.hist
        prob_hist(
            vs=[data],
            colors=["green"],
            labels=["Data"],
            bins=15,
            ax=ax,
            hist_type="plt.hist",
            plot_fit=False,
        )

        # Should create histogram
        assert len(ax.patches) > 0
        plt.close(fig)


@pytest.mark.fast
class TestDefineEndKs:
    """Test define_end_ks function."""

    def test_define_end_ks_basic_mode(self):
        """Test define_end_ks with 'basic' mode."""
        # Lines 836-956: define_end_ks function
        result = define_end_ks(mode="basic")

        assert isinstance(result, (list, tuple))
        assert len(result) > 0
        assert "l" in result  # l_par is always first

    def test_define_end_ks_minimal_mode(self):
        """Test define_end_ks with 'minimal' mode."""
        result = define_end_ks(mode="minimal")

        assert isinstance(result, (list, tuple))
        assert len(result) > 0

    def test_define_end_ks_custom_ks(self):
        """Test define_end_ks with custom parameter list."""
        custom = ["fsv", "sv_mu", "str_sd_mu"]
        result = define_end_ks(ks=custom)

        # Should return the custom list
        assert result == custom

    def test_define_end_ks_invalid_mode_raises(self):
        """Test define_end_ks raises on invalid mode."""
        # Lines 952-955: raises ValueError for invalid mode
        with pytest.raises(
            ValueError, match="Provide parameter shortcuts or define a mode"
        ):
            define_end_ks(mode="invalid_mode")


@pytest.mark.fast
class TestColorEpochs:
    """Test color_epochs function."""

    def test_color_epochs_with_boundaries(self):
        """Test color_epochs adds vertical lines for boundaries."""
        fig, ax = plt.subplots()
        epochs = [(10, 20), (30, 40)]
        trange = np.arange(100)

        # Lines 1000-1040: color_epochs function
        color_epochs(epochs, ax, trange, epoch_boundaries=True, epoch_area=False)

        # Should have added vertical lines
        assert len(ax.lines) >= 4  # 2 epochs * 2 boundaries
        plt.close(fig)

    def test_color_epochs_with_area(self):
        """Test color_epochs fills epoch regions."""
        fig, ax = plt.subplots()
        epochs = [(5, 15), (25, 35)]
        trange = np.arange(50)

        # Lines 1038-1040: epoch_area fills regions
        color_epochs(epochs, ax, trange, epoch_boundaries=False, epoch_area=True)

        # Should have added axvspan patches
        assert len(ax.patches) >= 2
        plt.close(fig)

    def test_color_epochs_both_features(self):
        """Test color_epochs with both boundaries and areas."""
        fig, ax = plt.subplots()
        epochs = [(10, 20)]
        trange = np.arange(30)

        # Both features enabled
        color_epochs(epochs, ax, trange, epoch_boundaries=True, epoch_area=True)

        assert len(ax.lines) >= 2  # Boundaries
        assert len(ax.patches) >= 1  # Area
        plt.close(fig)
