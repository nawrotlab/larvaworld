"""
Unit tests for larvaworld.lib.plot.grid module.

Tests composite grid-structured figure creation using synthetic data
and mock objects. Only tests calibration_plot which doesn't require datasets.
"""

from unittest.mock import MagicMock, patch
import os

import pytest

from larvaworld.lib.plot.grid import calibration_plot


@pytest.mark.fast
class TestCalibrationPlot:
    """Test calibration_plot function for composite figure creation."""

    def test_calibration_plot_basic(self):
        """Test basic calibration plot creation with provided files."""
        # Lines 31-91: calibration_plot function
        # Lines 48-49: from PIL import Image, from matplotlib import pyplot as plt
        # These imports are INSIDE the function, so we patch them at their source

        with (
            patch("PIL.Image.open") as mock_image_open,
            patch("matplotlib.pyplot.figure") as mock_fig,
        ):
            # Mock image objects
            mock_img = MagicMock()
            mock_image_open.return_value = mock_img

            # Mock figure and subplots
            mock_figure = MagicMock()
            mock_gs = MagicMock()
            mock_figure.add_gridspec.return_value = mock_gs
            mock_ax = MagicMock()
            mock_figure.add_subplot.return_value = mock_ax
            mock_fig.return_value = mock_figure

            files = ["file1.png", "file2.png", "file3.png", "file4.png", "file5.png"]
            save_to = "/test/dir"

            result = calibration_plot(save_to=save_to, files=files)

            # Lines 82-87: Opens each file and creates subplot for each
            assert mock_image_open.call_count == 5

            # Lines 84-87: Creates 5 axes (interference, bouts, orient, angular, bend)
            assert mock_figure.add_subplot.call_count == 5

            # Lines 88-89: Saves figure to filepath
            assert mock_figure.savefig.called

            # Returns the figure
            assert result == mock_figure

    def test_calibration_plot_default_save_to(self):
        """Test calibration plot with default save_to (current dir)."""
        # Lines 71-72: if save_to is None: save_to = "."

        with (
            patch("PIL.Image.open") as mock_image_open,
            patch("matplotlib.pyplot.figure") as mock_fig,
        ):
            mock_img = MagicMock()
            mock_image_open.return_value = mock_img

            mock_figure = MagicMock()
            mock_gs = MagicMock()
            mock_figure.add_gridspec.return_value = mock_gs
            mock_ax = MagicMock()
            mock_figure.add_subplot.return_value = mock_ax
            mock_fig.return_value = mock_figure

            files = ["f1.png", "f2.png", "f3.png", "f4.png", "f5.png"]

            result = calibration_plot(save_to=None, files=files)

            # Lines 88: filepath = os.path.join(save_to, filename)
            # Should use current directory "."
            call_args = mock_figure.savefig.call_args[0]
            assert call_args[0].startswith(".")

    def test_calibration_plot_default_files(self):
        """Test calibration plot with default file paths."""
        # Lines 73-81: If files is None, generates default filenames

        with (
            patch("PIL.Image.open") as mock_image_open,
            patch("matplotlib.pyplot.figure") as mock_fig,
            patch("os.path.join", side_effect=lambda *args: "/".join(args)),
        ):
            mock_img = MagicMock()
            mock_image_open.return_value = mock_img

            mock_figure = MagicMock()
            mock_gs = MagicMock()
            mock_figure.add_gridspec.return_value = mock_gs
            mock_ax = MagicMock()
            mock_figure.add_subplot.return_value = mock_ax
            mock_fig.return_value = mock_figure

            save_to = "/test/save"

            result = calibration_plot(save_to=save_to, files=None)

            # Lines 74-81: Default filenames
            # Should open 5 default files
            assert mock_image_open.call_count == 5

            # Lines 74-81: Default paths include interference, bouts, stride, turn
            calls = [call[0][0] for call in mock_image_open.call_args_list]
            assert any("interference" in str(call) for call in calls)
            assert any("bouts" in str(call) for call in calls)
            assert any("stride" in str(call) for call in calls)
            assert any("turn" in str(call) for call in calls)

    def test_calibration_plot_gridspec_layout(self):
        """Test correct gridspec layout creation."""
        # Lines 64-69: Specific gridspec layout for subplots

        with (
            patch("PIL.Image.open") as mock_image_open,
            patch("matplotlib.pyplot.figure") as mock_fig,
        ):
            mock_img = MagicMock()
            mock_image_open.return_value = mock_img

            mock_figure = MagicMock()
            mock_gs = MagicMock()
            mock_figure.add_gridspec.return_value = mock_gs
            mock_ax = MagicMock()
            mock_figure.add_subplot.return_value = mock_ax
            mock_fig.return_value = mock_figure

            files = ["f1.png", "f2.png", "f3.png", "f4.png", "f5.png"]

            result = calibration_plot(files=files)

            # Lines 64: gs = fig.add_gridspec(2, 6)
            mock_figure.add_gridspec.assert_called_once_with(2, 6)

    def test_calibration_plot_figure_size(self):
        """Test figure size calculation."""
        # Lines 63: figsize=(6 * 5, 2 * 5)

        with (
            patch("PIL.Image.open") as mock_image_open,
            patch("matplotlib.pyplot.figure") as mock_fig,
        ):
            mock_img = MagicMock()
            mock_image_open.return_value = mock_img

            mock_figure = MagicMock()
            mock_gs = MagicMock()
            mock_figure.add_gridspec.return_value = mock_gs
            mock_ax = MagicMock()
            mock_figure.add_subplot.return_value = mock_ax
            mock_fig.return_value = mock_figure

            files = ["f1.png"] * 5

            result = calibration_plot(files=files)

            # Lines 63: figsize should be (30, 10)
            call_kwargs = mock_fig.call_args[1]
            assert call_kwargs["figsize"] == (6 * 5, 2 * 5)
            assert call_kwargs["constrained_layout"] == True

    def test_calibration_plot_axes_configuration(self):
        """Test axes are configured correctly (ticks off, axis off)."""
        # Lines 51-60, 85-87: tick_params and axis off

        with (
            patch("PIL.Image.open") as mock_image_open,
            patch("matplotlib.pyplot.figure") as mock_fig,
        ):
            mock_img = MagicMock()
            mock_image_open.return_value = mock_img

            mock_figure = MagicMock()
            mock_gs = MagicMock()
            mock_figure.add_gridspec.return_value = mock_gs
            mock_ax = MagicMock()
            mock_figure.add_subplot.return_value = mock_ax
            mock_fig.return_value = mock_figure

            files = ["f1.png"] * 5

            result = calibration_plot(files=files)

            # Lines 85-87: Each ax should have tick_params and axis off called
            # 5 axes created
            assert mock_ax.tick_params.call_count == 5
            assert mock_ax.axis.call_count == 5

            # Lines 86: ax.axis("off") called
            assert all(call[0][0] == "off" for call in mock_ax.axis.call_args_list)

    def test_calibration_plot_imshow_calls(self):
        """Test imshow is called for each image."""
        # Lines 87: ax.imshow(im, cmap=None, aspect=None)

        with (
            patch("PIL.Image.open") as mock_image_open,
            patch("matplotlib.pyplot.figure") as mock_fig,
        ):
            mock_img = MagicMock()
            mock_image_open.return_value = mock_img

            mock_figure = MagicMock()
            mock_gs = MagicMock()
            mock_figure.add_gridspec.return_value = mock_gs
            mock_ax = MagicMock()
            mock_figure.add_subplot.return_value = mock_ax
            mock_fig.return_value = mock_figure

            files = ["f1.png", "f2.png", "f3.png", "f4.png", "f5.png"]

            result = calibration_plot(files=files)

            # Lines 87: ax.imshow called for each image
            assert mock_ax.imshow.call_count == 5

            # Lines 87: cmap=None, aspect=None
            for call in mock_ax.imshow.call_args_list:
                call_kwargs = call[1]
                assert call_kwargs["cmap"] is None
                assert call_kwargs["aspect"] is None

    def test_calibration_plot_savefig_params(self):
        """Test savefig is called with correct parameters."""
        # Lines 89: fig.savefig(filepath, dpi=300, facecolor=None)

        with (
            patch("PIL.Image.open") as mock_image_open,
            patch("matplotlib.pyplot.figure") as mock_fig,
        ):
            mock_img = MagicMock()
            mock_image_open.return_value = mock_img

            mock_figure = MagicMock()
            mock_gs = MagicMock()
            mock_figure.add_gridspec.return_value = mock_gs
            mock_ax = MagicMock()
            mock_figure.add_subplot.return_value = mock_ax
            mock_fig.return_value = mock_figure

            files = ["f1.png"] * 5
            save_to = "/test"

            result = calibration_plot(save_to=save_to, files=files)

            # Lines 89: savefig with dpi=300, facecolor=None
            call_kwargs = mock_figure.savefig.call_args[1]
            assert call_kwargs["dpi"] == 300
            assert call_kwargs["facecolor"] is None

    def test_calibration_plot_filename(self):
        """Test output filename is correct."""
        # Lines 62, 88: filename = "calibration.pdf"

        with (
            patch("PIL.Image.open") as mock_image_open,
            patch("matplotlib.pyplot.figure") as mock_fig,
            patch("os.path.join", side_effect=os.path.join),
        ):
            mock_img = MagicMock()
            mock_image_open.return_value = mock_img

            mock_figure = MagicMock()
            mock_gs = MagicMock()
            mock_figure.add_gridspec.return_value = mock_gs
            mock_ax = MagicMock()
            mock_figure.add_subplot.return_value = mock_ax
            mock_fig.return_value = mock_figure

            files = ["f1.png"] * 5
            save_to = "/test/output"

            result = calibration_plot(save_to=save_to, files=files)

            # Lines 62, 88: Should save as "calibration.pdf"
            call_args = mock_figure.savefig.call_args[0]
            assert call_args[0].endswith("calibration.pdf")
