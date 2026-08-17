"""Unit tests for larvaworld.lib.plot.ga (GA optimization-progress plotting)."""

from __future__ import annotations

import matplotlib.figure
import pandas as pd
import pytest

from larvaworld.lib.plot.ga import ga_progress_plot


@pytest.mark.fast
class TestGaProgressPlot:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "generation": [1, 1, 1, 2, 2, 2],
                "input_noise": [0.1, 0.2, 0.15, 0.3, 0.25, 0.28],
                "output_noise": [0.4, 0.35, 0.5, 0.2, 0.22, 0.19],
                "fitness": [-0.5, -0.4, -0.6, -0.2, -0.25, -0.18],
            }
        )

    def test_returns_a_figure_with_one_panel_per_param_plus_fitness(self) -> None:
        df = self._df()
        fig, _save_to, _filename = ga_progress_plot(
            df, ks=["input_noise", "output_noise"], return_fig=True
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 3

    def test_infers_ks_from_dataframe_columns_when_not_given(self) -> None:
        df = self._df()
        fig, _save_to, _filename = ga_progress_plot(df, return_fig=True)

        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 3

    def test_registered_under_graph_registry(self) -> None:
        from larvaworld.lib.reg.graph import GraphRegistry

        gr = GraphRegistry()
        assert gr.exists("ga progress")
