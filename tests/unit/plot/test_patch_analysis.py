"""The registered analysis of the odorous-food-patch assays.

The `patch` graphgroup used to ask for four plots of per-turn quantities that
`get_par` cannot serve, because they live in the epoch dictionaries rather than
in the step or endpoint data, so `ExpRun.analyze()` died on every patch
experiment. The group now holds only what it can draw, and the assays request
the bout annotation those plots are built from.
"""

from __future__ import annotations

import pytest

from larvaworld.lib import reg

PATCH_EXPERIMENTS = ["single_odor_patch", "single_odor_patch_x4", "double_patch"]


@pytest.mark.fast
class TestPatchGraphgroup:
    def test_the_group_holds_only_drawable_plots(self):
        keys = [e["key"] for e in reg.graphs.graphgroups["patch"]]
        assert keys == ["Y position", "navigation index", "turn amplitude"]

    def test_every_entry_names_a_registered_plot(self):
        for e in reg.graphs.graphgroups["patch"]:
            assert reg.graphs.exists(e["plotID"]), e["plotID"]


@pytest.mark.fast
class TestPatchExperimentAnalysis:
    @pytest.mark.parametrize("exp", PATCH_EXPERIMENTS)
    def test_analysis_includes_the_patch_group(self, exp):
        assert "patch" in reg.graphs.get_analysis_graphgroups(exp, sources={})

    @pytest.mark.parametrize("exp", PATCH_EXPERIMENTS)
    def test_bouts_are_annotated_for_the_ethogram_and_the_turn_plots(self, exp):
        anot = reg.conf.Exp.getID(exp).enrichment.anot_keys
        assert "bout_detection" in anot
        assert "bout_distribution" in anot

    def test_a_source_adds_its_own_group(self):
        groups = reg.graphs.get_analysis_graphgroups(
            "single_odor_patch_x4", sources={"Patch": (0.0, 0.0)}
        )
        assert "locomotion_relative_to_source_Patch" in groups


@pytest.mark.fast
class TestModelComparisonGroups:
    """The four groups of `single_odor_patch_x4` are a 2x2 of smelling x eating."""

    EXPECTED = {
        "forager": (True, True),
        "Orco": (False, True),
        "navigator": (True, False),
        "explorer": (False, False),
    }

    def test_the_groups_cross_olfaction_with_feeding(self):
        conf = reg.conf.Exp.getID("single_odor_patch_x4")
        actual = {}
        for gid, g in conf.larva_groups.items():
            b = reg.conf.Model.getID(g.model).brain
            actual[gid] = (b.olfactor is not None, b.feeder is not None)
        assert actual == self.EXPECTED

    def test_no_group_carries_energetics(self):
        """Nothing metabolises what is eaten: the assay is about behavior."""
        conf = reg.conf.Exp.getID("single_odor_patch_x4")
        for g in conf.larva_groups.values():
            assert reg.conf.Model.getID(g.model).energetics is None

    def test_the_two_olfactory_models_are_wired_identically(self):
        """The 2x2 design of the model_comparison tutorial depends on this.

        `navigator` and `max_forager` must differ only in the feeder, so their
        olfactors have to be the same sensor with the same gain *and* the same
        coupling to locomotion. `brute_force` selects that coupling, and it used
        to leak into every forager from the `_brute` navigator built just before
        it, which made the forager's olfactor a stop-on-decrease rule instead of
        a turner modulation.
        """
        nav = reg.conf.Model.getID("navigator").brain.olfactor
        forager = reg.conf.Model.getID("max_forager").brain.olfactor
        assert nav == forager
        assert nav.brute_force is False

    def test_the_two_feeding_models_share_a_feeder(self):
        """The other axis of the 2x2: `Orco` is `forager` minus the olfactor."""
        forager = reg.conf.Model.getID("max_forager").brain
        orco = reg.conf.Model.getID("max_feeder").brain
        assert orco.olfactor is None
        assert forager.feeder == orco.feeder
        assert forager.intermitter == orco.intermitter

    def test_the_brute_force_navigator_keeps_its_wiring(self):
        """The one model that asks for stop-on-decrease still gets it."""
        brute = reg.conf.Model.getID("RE_NEU_PHI_DEF_navigator_brute").brain.olfactor
        assert brute.brute_force is True
