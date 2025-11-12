"""
Unit tests for larvaworld.lib.plot.table module.

Tests table creation, formatting, and DataFrame manipulation functions
using synthetic data and mock objects.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from larvaworld.lib.plot.table import (
    arrange_index_labels,
    mpl_table,
    error_table,
)


@pytest.mark.fast
class TestArrangeIndexLabels:
    """Test arrange_index_labels function for centering group labels."""

    def test_arrange_single_group(self):
        """Test with single group (odd count)."""
        # Lines 26-51: arrange_index_labels function
        # Creates ["", "GROUP1", ""] for count=3
        index = pd.Index(["group1", "group1", "group1"])
        result = arrange_index_labels(index)

        # Should center label with empty strings
        assert len(result) == 3
        assert result[0] == ""
        assert result[1] == "GROUP1"
        assert result[2] == ""

    def test_arrange_single_group_even(self):
        """Test with single group (even count)."""
        # Lines 45-48: merge function logic
        # For Nk=4: Nk1=1, Nk2=2 -> ["", "GROUP1", "", ""]
        index = pd.Index(["group1"] * 4)
        result = arrange_index_labels(index)

        assert len(result) == 4
        assert result[0] == ""
        assert result[1] == "GROUP1"
        assert result[2] == ""
        assert result[3] == ""

    def test_arrange_multiple_groups(self):
        """Test with multiple groups."""
        # Lines 42-50: Processes each unique group
        index = pd.Index(["a", "a", "a", "b", "b", "c"])
        result = arrange_index_labels(index)

        # a: 3 items -> ["", "A", ""]
        # b: 2 items, Nk1=0, Nk2=1 -> ["B", ""]
        # c: 1 item -> ["C"]
        assert len(result) == 6
        assert result[0] == ""  # a group
        assert result[1] == "A"
        assert result[2] == ""
        assert result[3] == "B"  # b group (no leading empty for even count=2)
        assert result[4] == ""
        assert result[5] == "C"  # c group

    def test_arrange_single_item_per_group(self):
        """Test when each group has single item."""
        # Lines 46-47: When Nk=1, Nk1=0, Nk2=0 -> just [k.upper()]
        index = pd.Index(["a", "b", "c"])
        result = arrange_index_labels(index)

        assert result == ["A", "B", "C"]

    def test_arrange_preserves_order(self):
        """Test that group order is preserved."""
        # Lines 42: index.unique() preserves order
        index = pd.Index(["z", "z", "a", "a", "m", "m"])
        result = arrange_index_labels(index)

        # Should have Z, A, M in that order (not alphabetical)
        assert "Z" in result
        assert "A" in result
        assert "M" in result
        z_idx = result.index("Z")
        a_idx = result.index("A")
        m_idx = result.index("M")
        assert z_idx < a_idx < m_idx


@pytest.mark.fast
class TestMplTable:
    """Test mpl_table function for matplotlib table creation."""

    def test_mpl_table_basic(self):
        """Test basic table creation."""
        # Lines 192-347: mpl_table function
        # Creates AutoBasePlot with table
        data = pd.DataFrame(
            {"A": [1, 2, 3], "B": [4, 5, 6]}, index=["row1", "row2", "row3"]
        )

        with patch("larvaworld.lib.plot.table.plot.AutoBasePlot") as mock_plot:
            mock_instance = MagicMock()
            mock_ax = MagicMock()
            mock_table = MagicMock()
            mock_table._cells = {}
            mock_ax.table.return_value = mock_table
            mock_instance.axs = [mock_ax]
            mock_instance.fig = MagicMock()
            mock_instance.get.return_value = "plot_output"
            mock_plot.return_value = mock_instance

            result = mpl_table(data, name="test_table", return_table=False)

            # Lines 287: Creates AutoBasePlot
            mock_plot.assert_called_once()
            # Lines 289: Turns off axis
            mock_ax.axis.assert_called_once_with("off")
            # Lines 291-300: Creates table with data
            mock_ax.table.assert_called_once()
            call_kwargs = mock_ax.table.call_args[1]
            assert np.array_equal(call_kwargs["cellText"], data.values)
            # Lines 347: Returns P.get()
            assert result == "plot_output"

    def test_mpl_table_return_table(self):
        """Test returning table object instead of plot."""
        # Lines 344-345: return_table=True returns (ax, fig, mpl)
        data = pd.DataFrame({"A": [1, 2]})

        with patch("larvaworld.lib.plot.table.plot.AutoBasePlot") as mock_plot:
            mock_instance = MagicMock()
            mock_ax = MagicMock()
            mock_table = MagicMock()
            mock_table._cells = {}
            mock_ax.table.return_value = mock_table
            mock_instance.axs = [mock_ax]
            mock_instance.fig = MagicMock()
            mock_plot.return_value = mock_instance

            ax, fig, mpl = mpl_table(data, return_table=True)

            assert ax == mock_ax
            assert fig == mock_instance.fig
            assert mpl == mock_table

    def test_mpl_table_with_title(self):
        """Test table with title."""
        # Lines 340: ax.set_title(title)
        data = pd.DataFrame({"A": [1]})

        with patch("larvaworld.lib.plot.table.plot.AutoBasePlot") as mock_plot:
            mock_instance = MagicMock()
            mock_ax = MagicMock()
            mock_table = MagicMock()
            mock_table._cells = {}
            mock_ax.table.return_value = mock_table
            mock_instance.axs = [mock_ax]
            mock_instance.fig = MagicMock()
            mock_plot.return_value = mock_instance

            mpl_table(data, title="Test Title", return_table=False)

            mock_ax.set_title.assert_called_once_with("Test Title")

    def test_mpl_table_font_size(self):
        """Test table font size setting."""
        # Lines 303-304: mpl.auto_set_font_size(False), mpl.set_fontsize(font_size)
        data = pd.DataFrame({"A": [1]})

        with patch("larvaworld.lib.plot.table.plot.AutoBasePlot") as mock_plot:
            mock_instance = MagicMock()
            mock_ax = MagicMock()
            mock_table = MagicMock()
            mock_table._cells = {}
            mock_ax.table.return_value = mock_table
            mock_instance.axs = [mock_ax]
            mock_instance.fig = MagicMock()
            mock_plot.return_value = mock_instance

            mpl_table(data, font_size=20, return_table=False)

            mock_table.auto_set_font_size.assert_called_once_with(False)
            mock_table.set_fontsize.assert_called_once_with(20)

    def test_mpl_table_with_adjust_kws(self):
        """Test table with figure adjustment keywords."""
        # Lines 342-343: P.fig.subplots_adjust(**adjust_kws)
        data = pd.DataFrame({"A": [1]})
        adjust_kws = {"left": 0.1, "right": 0.9}

        with patch("larvaworld.lib.plot.table.plot.AutoBasePlot") as mock_plot:
            mock_instance = MagicMock()
            mock_ax = MagicMock()
            mock_table = MagicMock()
            mock_table._cells = {}
            mock_ax.table.return_value = mock_table
            mock_instance.axs = [mock_ax]
            mock_instance.fig = MagicMock()
            mock_plot.return_value = mock_instance

            mpl_table(data, adjust_kws=adjust_kws, return_table=False)

            mock_instance.fig.subplots_adjust.assert_called_once_with(**adjust_kws)

    def test_mpl_table_highlighted_cells_row_min(self):
        """Test highlighting minimum values in rows."""
        # Lines 250-281: get_idx function for row_min
        # Lines 253-259: Finds minimum in each row
        data = pd.DataFrame(
            {"A": [3.0, 1.0, 5.0], "B": [1.0, 2.0, 3.0], "C": [2.0, 3.0, 4.0]}
        )

        with patch("larvaworld.lib.plot.table.plot.AutoBasePlot") as mock_plot:
            mock_instance = MagicMock()
            mock_ax = MagicMock()
            mock_table = MagicMock()
            # Cell indices: (row+1, col) for data cells
            mock_table._cells = {
                (0, 0): MagicMock(),  # header
                (1, 0): MagicMock(),  # row 0, col A - value 3
                (1, 1): MagicMock(),  # row 0, col B - value 1 (MIN)
                (2, 0): MagicMock(),  # row 1, col A - value 1 (MIN)
                (3, 1): MagicMock(),  # row 2, col B - value 3 (MIN)
            }
            mock_ax.table.return_value = mock_table
            mock_instance.axs = [mock_ax]
            mock_instance.fig = MagicMock()
            mock_plot.return_value = mock_instance

            mpl_table(data, highlighted_cells="row_min", return_table=False)

            # Lines 308-309: Highlights cells in highlight_idx with highlight_color
            # Should call set_facecolor on highlighted cells
            assert mock_table._cells[(1, 1)].set_facecolor.called  # min in row 0

    def test_mpl_table_header0(self):
        """Test adding additional header row."""
        # Lines 319-331: Adds header0 cell when header0 is not None
        data = pd.DataFrame({"A": [1]})

        with patch("larvaworld.lib.plot.table.plot.AutoBasePlot") as mock_plot:
            mock_instance = MagicMock()
            mock_ax = MagicMock()
            mock_table = MagicMock()
            mock_table._cells = {(0, -1): MagicMock()}
            mock_table._approx_text_height.return_value = 0.05
            mock_ax.table.return_value = mock_table
            mock_instance.axs = [mock_ax]
            mock_instance.fig = MagicMock()
            mock_plot.return_value = mock_instance

            mpl_table(data, header0="FIELD", header0_color="red", return_table=False)

            # Lines 322-330: Adds cell at (0, -1)
            mock_table.add_cell.assert_called_once()
            call_kwargs = mock_table.add_cell.call_args[1]
            assert call_kwargs["text"] == "FIELD"
            assert call_kwargs["facecolor"] == "red"
            # Lines 311-312, 331: set_text_props called for headers AND for (0,-1)
            # So it gets called twice total (once for header row, once for header0)
            assert mock_table._cells[(0, -1)].set_text_props.call_count >= 1


@pytest.mark.fast
class TestErrorTable:
    """Test error_table function for error metric display."""

    def test_error_table_basic(self):
        """Test basic error table creation."""
        # Lines 402-431: error_table function
        # Lines 421: Transposes and rounds data
        data = np.array([[1.2345, 2.3456], [3.4567, 4.5678]])

        with patch("larvaworld.lib.plot.table.mpl_table") as mock_mpl_table:
            mock_mpl_table.return_value = "table_output"

            result = error_table(data, k="test_metric")

            # Lines 423-430: Calls mpl_table with transposed, rounded data
            mock_mpl_table.assert_called_once()
            call_args = mock_mpl_table.call_args

            # Check data is transposed and rounded to 3 decimals
            passed_data = call_args[0][0]
            expected_data = np.round(data, 3).T
            assert np.array_equal(passed_data, expected_data)

            # Lines 425: highlighted_cells="row_min"
            assert call_args[1]["highlighted_cells"] == "row_min"
            # Lines 428: name includes metric key
            assert call_args[1]["name"] == "error_table_test_metric"

            assert result == "table_output"

    def test_error_table_figsize_calculation(self):
        """Test figsize is calculated from data shape."""
        # Lines 421-422: data = np.round(data, 3).T
        # figsize = ((data.shape[1] + 3) * 4, data.shape[0])
        # AFTER the transpose!
        data = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
        # After .T: 3x2
        # figsize uses the NEW shape: ((new_shape[1] + 3) * 4, new_shape[0])
        # new_shape = (3, 2), so figsize = ((2 + 3) * 4, 3) = (20, 3)

        with patch("larvaworld.lib.plot.table.mpl_table") as mock_mpl_table:
            error_table(data, k="")

            call_args = mock_mpl_table.call_args
            # Lines 422: figsize calculated AFTER transpose
            transposed_shape = data.T.shape  # (3, 2)
            expected_figsize = ((transposed_shape[1] + 3) * 4, transposed_shape[0])
            assert call_args[1]["figsize"] == expected_figsize

    def test_error_table_with_kwargs(self):
        """Test passing additional kwargs to mpl_table."""
        # Lines 429: **kwargs passed to mpl_table
        data = np.array([[1.0, 2.0]])

        with patch("larvaworld.lib.plot.table.mpl_table") as mock_mpl_table:
            error_table(data, k="metric", show=True, save_to="/tmp")

            call_args = mock_mpl_table.call_args
            assert call_args[1]["show"] == True
            assert call_args[1]["save_to"] == "/tmp"

    def test_error_table_rounding(self):
        """Test that error values are rounded to 3 decimals."""
        # Lines 421: np.round(data, 3)
        data = np.array([[1.23456789, 2.98765432]])

        with patch("larvaworld.lib.plot.table.mpl_table") as mock_mpl_table:
            error_table(data)

            passed_data = mock_mpl_table.call_args[0][0]
            # Should be rounded to 3 decimals
            assert passed_data[0, 0] == 1.235
            assert passed_data[1, 0] == 2.988


@pytest.mark.fast
class TestMplTableHighlighting:
    """Test mpl_table highlighting modes and features."""

    def test_mpl_table_highlight_row_max(self):
        """Test highlighting maximum values in rows."""
        # Lines 260-266: row_max highlighting mode
        data = pd.DataFrame([[1, 5, 3], [7, 2, 9]], columns=["A", "B", "C"])

        with patch("larvaworld.lib.plot.table.plot.AutoBasePlot") as mock_plot:
            mock_instance = MagicMock()
            mock_ax = MagicMock()
            mock_table = MagicMock()
            # Mock cells for row 0 (max=5 at idx 1) and row 1 (max=9 at idx 2)
            mock_table._cells = {
                (1, 1): MagicMock(),  # row 0, col 1 (value 5)
                (2, 2): MagicMock(),  # row 1, col 2 (value 9)
            }
            mock_ax.table.return_value = mock_table
            mock_instance.axs = [mock_ax]
            mock_instance.fig = MagicMock()
            mock_plot.return_value = mock_instance

            mpl_table(
                data,
                highlighted_cells="row_max",
                highlight_color="red",
                return_table=False,
            )

            # Lines 308-309: Should highlight max cells
            assert mock_table._cells[(1, 1)].set_facecolor.called
            assert mock_table._cells[(2, 2)].set_facecolor.called

    def test_mpl_table_highlight_col_min(self):
        """Test highlighting minimum values in columns."""
        # Lines 267-273: col_min highlighting mode
        data = pd.DataFrame([[3, 8], [1, 5], [7, 2]], columns=["A", "B"])

        with patch("larvaworld.lib.plot.table.plot.AutoBasePlot") as mock_plot:
            mock_instance = MagicMock()
            mock_ax = MagicMock()
            mock_table = MagicMock()
            # Mock cells for col A (min=1 at row 1) and col B (min=2 at row 2)
            mock_table._cells = {
                (2, 0): MagicMock(),  # row 1, col 0 (value 1)
                (3, 1): MagicMock(),  # row 2, col 1 (value 2)
            }
            mock_ax.table.return_value = mock_table
            mock_instance.axs = [mock_ax]
            mock_instance.fig = MagicMock()
            mock_plot.return_value = mock_instance

            mpl_table(data, highlighted_cells="col_min", return_table=False)

            # Lines 308-309: Should highlight min cells
            assert mock_table._cells[(2, 0)].set_facecolor.called
            assert mock_table._cells[(3, 1)].set_facecolor.called

    def test_mpl_table_highlight_col_max(self):
        """Test highlighting maximum values in columns."""
        # Lines 274-280: col_max highlighting mode
        data = pd.DataFrame([[3, 2], [9, 5], [7, 8]], columns=["A", "B"])

        with patch("larvaworld.lib.plot.table.plot.AutoBasePlot") as mock_plot:
            mock_instance = MagicMock()
            mock_ax = MagicMock()
            mock_table = MagicMock()
            # Mock cells for col A (max=9 at row 1) and col B (max=8 at row 2)
            mock_table._cells = {
                (2, 0): MagicMock(),  # row 1, col 0 (value 9)
                (3, 1): MagicMock(),  # row 2, col 1 (value 8)
            }
            mock_ax.table.return_value = mock_table
            mock_instance.axs = [mock_ax]
            mock_instance.fig = MagicMock()
            mock_plot.return_value = mock_instance

            mpl_table(data, highlighted_cells="col_max", return_table=False)

            # Lines 308-309: Should highlight max cells
            assert mock_table._cells[(2, 0)].set_facecolor.called
            assert mock_table._cells[(3, 1)].set_facecolor.called

    def test_mpl_table_highlighted_celltext_dict(self):
        """Test highlighting cells by text content."""
        # Lines 334-340: highlighted_celltext_dict functionality
        data = pd.DataFrame([["red", "blue"], ["green", "red"]], columns=["A", "B"])

        with patch("larvaworld.lib.plot.table.plot.AutoBasePlot") as mock_plot:
            mock_instance = MagicMock()
            mock_ax = MagicMock()
            mock_table = MagicMock()
            mock_cell1 = MagicMock()
            mock_cell1.get_text.return_value.get_text.return_value = "red"
            mock_cell2 = MagicMock()
            mock_cell2.get_text.return_value.get_text.return_value = "blue"
            mock_table._cells = {
                (1, 0): mock_cell1,
                (1, 1): mock_cell2,
            }
            mock_ax.table.return_value = mock_table
            mock_instance.axs = [mock_ax]
            mock_instance.fig = MagicMock()
            mock_plot.return_value = mock_instance

            # Lines 334-340: Loop through cells and highlight if text matches
            highlight_dict = {"red": ["red"], "blue": ["blue"]}
            mpl_table(
                data, highlighted_celltext_dict=highlight_dict, return_table=False
            )

            # Should call set_facecolor for cells containing matched text
            assert mock_cell1.set_facecolor.called
            assert mock_cell2.set_facecolor.called

    def test_mpl_table_with_colWidths(self):
        """Test custom column widths."""
        # Lines 285-286: Custom colWidths passed to table
        data = pd.DataFrame([[1, 2, 3]], columns=["A", "B", "C"])

        with patch("larvaworld.lib.plot.table.plot.AutoBasePlot") as mock_plot:
            mock_instance = MagicMock()
            mock_ax = MagicMock()
            mock_table = MagicMock()
            mock_ax.table.return_value = mock_table
            mock_instance.axs = [mock_ax]
            mock_instance.fig = MagicMock()
            mock_plot.return_value = mock_instance

            col_widths = [0.2, 0.5, 0.3]
            mpl_table(data, colWidths=col_widths, return_table=False)

            # Lines 285-286, 303-304: colWidths passed to ax.table
            call_kwargs = mock_ax.table.call_args[1]
            assert "colWidths" in call_kwargs
            assert call_kwargs["colWidths"] == col_widths

    def test_mpl_table_with_header_columns(self):
        """Test header columns formatting."""
        # Lines 313-314: header_columns > 0 applies header formatting
        data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])

        with patch("larvaworld.lib.plot.table.plot.AutoBasePlot") as mock_plot:
            mock_instance = MagicMock()
            mock_ax = MagicMock()
            mock_table = MagicMock()
            # Mock cells for first column (header column)
            mock_table._cells = {
                (1, 0): MagicMock(),
                (2, 0): MagicMock(),
            }
            mock_ax.table.return_value = mock_table
            mock_instance.axs = [mock_ax]
            mock_instance.fig = MagicMock()
            mock_plot.return_value = mock_instance

            mpl_table(data, header_columns=1, return_table=False)

            # Lines 313-314: Should set text props for header column cells
            assert mock_table._cells[(1, 0)].set_text_props.called
            assert mock_table._cells[(2, 0)].set_text_props.called
