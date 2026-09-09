"""Unit tests for larvaworld.lib.plot.ga (GA optimization-progress plotting)."""

from __future__ import annotations

import matplotlib.figure
import pandas as pd
import pytest

from larvaworld.lib import reg
from larvaworld.lib.plot.ga import ga_progress_plot


@pytest.mark.fast
class TestTurnerNoiseParamRegistration:
    """
    input_noise/output_noise (Effector-level fields, shared by every
    module built on basic.Effector) had no parameter-database entries
    before the GA-progress plot rewrite needed turner-specific symbols
    for them -- registered in parDB.py's build_sim_pars, following the
    existing turner-parameter subscript convention (nam.tex.sub(x, "T")).
    """

    def test_turner_input_noise_registered(self) -> None:
        assert "In_T" in reg.par.dict
        entry = reg.par.dict["In_T"]
        assert entry.p == "brain.turner.input_noise"

    def test_turner_output_noise_registered(self) -> None:
        assert "On_T" in reg.par.dict
        entry = reg.par.dict["On_T"]
        assert entry.p == "brain.turner.output_noise"

    def test_symbols_use_turner_subscript_convention(self) -> None:
        assert reg.getPar(k="In_T", to_return="symbol") == "${In}_{T}$"
        assert reg.getPar(k="On_T", to_return="symbol") == "${On}_{T}$"

    def test_resolvable_by_full_path(self) -> None:
        assert reg.getPar(p="brain.turner.input_noise", to_return="k") == "In_T"
        assert reg.getPar(p="brain.turner.output_noise", to_return="k") == "On_T"


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

    def test_generation_ticks_are_integer_only(self) -> None:
        from matplotlib.ticker import MaxNLocator

        df = self._df()
        fig, _save_to, _filename = ga_progress_plot(
            df, ks=["input_noise", "output_noise"], return_fig=True
        )
        for ax in fig.axes:
            assert isinstance(ax.xaxis.get_major_locator(), MaxNLocator)

    def test_fitness_panel_is_separate_and_labeled(self) -> None:
        df = self._df()
        fig, _save_to, _filename = ga_progress_plot(
            df, ks=["input_noise", "output_noise"], return_fig=True
        )
        fitness_ax = fig.axes[-1]
        assert fitness_ax.get_ylabel() == "Fitness"
        # generation labeling states population size, not a bare "Generation"
        assert "larvae/generation" in fitness_ax.get_xlabel()

    def test_title_identifies_ga_run_with_counts(self) -> None:
        df = self._df()
        fig, _save_to, _filename = ga_progress_plot(
            df, ks=["input_noise", "output_noise"], return_fig=True
        )
        title = fig._suptitle.get_text()
        assert "Genetic algorithm" in title
        assert "2 generations" in title
        assert "3 larvae/generation" in title

    def test_module_resolves_db_symbol_ylabel(self) -> None:
        # "In_T"/"On_T" are the turner input/output-noise registry entries
        # (parDB.py's build_sim_pars) -- resolved via
        # brain.turner.<column-name> when the bare column name isn't
        # itself a registered key.
        df = self._df()
        fig, _save_to, _filename = ga_progress_plot(
            df, ks=["input_noise", "output_noise"], module="turner", return_fig=True
        )
        ylabels = [ax.get_ylabel() for ax in fig.axes[:-1]]
        assert any("In" in lbl and "T" in lbl for lbl in ylabels)
        assert any("On" in lbl and "T" in lbl for lbl in ylabels)

    def test_unresolved_ylabel_falls_back_to_column_name(self) -> None:
        df = self._df().rename(
            columns={"input_noise": "not_a_real_param", "output_noise": "also_fake"}
        )
        fig, _save_to, _filename = ga_progress_plot(
            df, ks=["not_a_real_param", "also_fake"], return_fig=True
        )
        ylabels = {ax.get_ylabel() for ax in fig.axes[:-1]}
        assert ylabels == {"not_a_real_param", "also_fake"}
