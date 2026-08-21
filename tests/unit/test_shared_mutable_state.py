"""Nothing hands out a shared mutable object for its caller to write into.

Several places used to return, or fill in, an object that outlives the call:
the registry's stored configurations, and the shared default arguments of a few
constructors. Because none of them copied first, one caller's edits silently
became the next caller's starting point - the same defect that gave every stored
forager an olfactor wiring nothing had asked for.
"""

from __future__ import annotations

import pytest

from larvaworld.lib import reg


@pytest.mark.fast
class TestConfExpandIsolation:
    """`ConfType.expand` exists so a caller can adjust a conf before running it."""

    EXP = "food_grid"

    def test_expand_does_not_return_the_stored_object(self):
        assert reg.conf.Exp.expand(self.EXP) is not reg.conf.Exp.getID(self.EXP)

    def test_editing_an_expanded_conf_leaves_the_store_alone(self):
        before = reg.conf.Exp.getID(self.EXP).enrichment.anot_keys
        assert before, "the fixture experiment should request annotation"

        p = reg.conf.Exp.expand(self.EXP)
        p.enrichment = {}

        assert reg.conf.Exp.getID(self.EXP).enrichment.anot_keys == before

    def test_the_substituted_sub_configurations_are_copies_too(self):
        """An expanded Exp used to share its Model and Env with their stores."""
        p = reg.conf.Exp.expand("single_odor_patch_x4")
        g = p.larva_groups["forager"]
        if isinstance(g.model, str):
            pytest.skip("the model was not expanded in place")
        before = reg.conf.Model.getID("max_forager").brain.intermitter.EEB
        g.model.brain.intermitter.EEB = 0.123
        assert reg.conf.Model.getID("max_forager").brain.intermitter.EEB == before

    def test_a_caller_supplied_conf_is_not_written_into(self):
        conf = reg.conf.Exp.getID(self.EXP).get_copy()
        before = conf.get_copy()
        reg.conf.Exp.expand(conf=conf)
        assert conf == before


@pytest.mark.fast
class TestSharedDefaultArguments:
    def test_two_plots_do_not_share_one_build_kws(self):
        from larvaworld.lib.plot.base import BasePlot

        a = BasePlot(name="a", subplot_kw={"projection": "polar"})
        b = BasePlot(name="b")
        assert a.build_kws is not b.build_kws
        assert b.build_kws["subplot_kw"] == {}

    def test_the_plot_build_kws_default_stays_empty(self):
        import inspect

        from larvaworld.lib.plot.base import BasePlot

        BasePlot(name="a", subplot_kw={"projection": "polar"})
        default = inspect.signature(BasePlot.__init__).parameters["build_kws"].default
        assert default == {}
