"""`BrainModule.module_conf` hands out a copy of its per-mode defaults.

The default dict is built once, in `__init__`, and was previously returned and
updated in place, so any argument a caller omitted kept the value the previous
caller had passed. Two consequences were visible in the shipped registry: every
forager inherited `olfactor.brute_force` from the `_brute` navigator built just
before it, and every model built after the Levy block inherited its
`intermitter.run_mode = "exec"`, which erased the very property the Levy
variants exist to express.
"""

from __future__ import annotations

import pytest

from larvaworld.lib import reg
from larvaworld.lib.model.modules.module_modes import moduleDB

OLF = dict(mID="olfactor", mode="default", gain_dict={"Odor": 150.0})


@pytest.mark.fast
class TestModuleConfIsolation:
    def test_an_omitted_argument_does_not_inherit_the_previous_call(self):
        moduleDB.module_conf(**OLF, brute_force=True)
        d = moduleDB.module_conf(**OLF)
        assert d["brain.olfactor"]["brute_force"] is False

    def test_two_calls_do_not_share_one_dict(self):
        a = moduleDB.module_conf(**OLF)["brain.olfactor"]
        b = moduleDB.module_conf(**OLF)["brain.olfactor"]
        assert a is not b

    def test_editing_a_returned_conf_leaves_the_defaults_alone(self):
        d = moduleDB.module_conf(**OLF)["brain.olfactor"]
        d["decay_coef"] = 99.0
        assert moduleDB.module_conf(**OLF)["brain.olfactor"]["decay_coef"] != 99.0


@pytest.mark.fast
class TestLarvaConfIsolation:
    def test_larvaConf_does_not_fill_in_the_caller_mkws(self):
        """The missing module keys used to be written back into the argument,
        which unpassed is the shared default of the method."""
        mkws = {}
        moduleDB.larvaConf(mkws=mkws)
        assert mkws == {}


@pytest.mark.fast
class TestRunModeIsNotInherited:
    """`run_mode` picks how a run epoch is drawn, and only Levy models set it.

    'stridechain' draws a run as a number of strides, 'exec' as a duration in
    seconds - the parametrisation a Levy walk is defined by. Both are fitted
    from the reference dataset, so the distinction is meaningful rather than
    cosmetic, and it was lost while every model read 'exec'.
    """

    LEVY = ["Levy", "NEU_Levy", "Levy_forager", "Levy_max_forager"]
    STANDARD = ["explorer", "navigator", "max_forager", "max_feeder", "rover", "sitter"]

    @pytest.mark.parametrize("mID", LEVY)
    def test_levy_models_parametrise_runs_by_duration(self, mID):
        assert reg.conf.Model.getID(mID).brain.intermitter.run_mode == "exec"

    @pytest.mark.parametrize("mID", STANDARD)
    def test_the_others_parametrise_runs_by_stride_count(self, mID):
        assert reg.conf.Model.getID(mID).brain.intermitter.run_mode == "stridechain"
