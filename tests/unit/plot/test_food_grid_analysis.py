"""Tests for the food-grid feeding assay: its rendering, plots and analysis.

Each covers a defect that made the assay unusable:
the food grid was never drawn, the food-intake plot pointed at a column no
dataset has, two plots crashed when a graphgroup gave them a name, and no
experiment asked for the intake analysis.
"""

import numpy as np
import pytest

from larvaworld.lib import reg
from larvaworld.lib.model.envs.valuegrid import FoodGrid


@pytest.mark.fast
class TestFoodGridRendering:
    def test_food_grid_is_visible_by_default(self):
        """Unlike the odor/wind/thermal layers, food is a physical feature."""
        assert FoodGrid().visible is True

    def test_full_cells_are_green_and_empty_ones_white(self):
        g = FoodGrid()
        full = g.get_color(g.initial_value)
        empty = g.get_color(0.0)
        assert np.allclose(empty, [255, 255, 255])
        assert full[1] > full[0] and full[1] > full[2]

    def test_colour_fades_towards_white_as_food_is_eaten(self):
        g = FoodGrid()
        levels = [g.initial_value, g.initial_value / 2, 0.0]
        whiteness = [g.get_color(v)[0] for v in levels]
        assert whiteness == sorted(whiteness)


@pytest.mark.fast
class TestFoodIntakeParameters:
    def test_intake_parameters_resolve_to_real_columns(self):
        """The plot used a literal 'ingested_food_volume', which is no column."""
        reg.par.update_kdict(ks=["f_am", "f_am_V"])
        assert reg.getPar("f_am") == "amount_eaten"
        assert reg.getPar("f_am_V") == "ingested_volume"

    def test_the_stale_column_name_is_gone(self):
        from pathlib import Path

        src = Path(reg.__file__).parents[1] / "plot" / "deb.py"
        assert "ingested_food_volume" not in src.read_text(encoding="utf-8")


@pytest.mark.fast
class TestFoodGridExperiment:
    def test_analysis_includes_the_intake_group(self):
        groups = reg.graphs.get_analysis_graphgroups("food_grid", sources={})
        assert "intake" in groups

    def test_intake_group_holds_the_expected_plots(self):
        keys = [e["key"] for e in reg.graphs.graphgroups["intake"]]
        assert "food intake (raw)" in keys
        assert "food intake (barplot)" in keys
        assert "ethogram" in keys

    def test_experiment_requests_bout_annotation(self):
        """The ethogram needs annotated crawl and pause epochs to draw."""
        enr = reg.conf.Exp.getID("food_grid").enrichment
        assert "bout_detection" in enr.anot_keys

    def test_larvae_have_a_feeder_and_no_energetics(self):
        conf = reg.conf.Exp.getID("food_grid")
        gID = conf.larva_groups.keylist[0]
        m = reg.conf.Model.getID(conf.larva_groups[gID].model)
        assert m.brain.feeder is not None
        assert m.energetics is None
