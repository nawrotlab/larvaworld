"""`ParamRegistry.compute` on a parameter that has not been realized yet.

`kdict` is filled lazily: a key is instantiated the first time something asks
for it. `get` calls `update_kdict` before indexing, but `compute` indexed
`kdict` directly, so computing a parameter nobody had touched raised a bare
`KeyError`. The chunk-tracked turn parameters hit this in the food-patch
analyses.
"""

from __future__ import annotations

import pytest

from larvaworld.lib import reg
from larvaworld.lib.util import AttrDict

# A chunk-tracked parameter, registered but not instantiated at import time.
LAZY_KEY = "tur_fo0"


class _StubDataset:
    """The whole of the dataset interface `LarvaworldParam.exists` touches."""

    def __init__(self, step_ps=(), end_ps=()):
        self.step_ps = list(step_ps)
        self.end_ps = list(end_ps)


@pytest.mark.fast
class TestLazyKeyRealization:
    def test_the_key_is_registered(self):
        assert LAZY_KEY in reg.par.ks

    def test_compute_realizes_a_key_absent_from_kdict(self):
        column = reg.getPar(LAZY_KEY)
        reg.par.kdict.pop(LAZY_KEY, None)
        assert LAZY_KEY not in reg.par.kdict

        # The parameter is already present in the stub, so `compute` only has to
        # look it up; before the fix it never got that far.
        reg.par.compute(LAZY_KEY, _StubDataset(step_ps=[column]))

        assert LAZY_KEY in reg.par.kdict

    def test_get_param_is_what_makes_it_work(self):
        reg.par.kdict.pop(LAZY_KEY, None)
        p = reg.par.get_param(LAZY_KEY)
        assert p is not None
        assert reg.par.kdict[LAZY_KEY] is p

    def test_exists_reports_a_missing_parameter(self):
        p = reg.par.get_param(LAZY_KEY)
        res = p.exists(_StubDataset())
        assert isinstance(res, (dict, AttrDict))
        assert not any(res.values())
