"""
Integration tests for essay configuration helpers using real registry state.

These tests validate that the stored essay definitions can be instantiated with
real dependencies and provide coherent experiment dictionaries without running
full simulations.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.fast]


@pytest.fixture()
def patched_generators(monkeypatch):
    """Patch heavy generator classes with lightweight stand-ins."""
    from larvaworld.lib import util
    from larvaworld.lib import reg
    from larvaworld.lib.plot import table as plot_table
    from larvaworld.lib import plot as plot_pkg

    class DummyGroup:
        def __init__(self, group_id=None, sample=None, **kwargs):
            self.group_id = group_id or kwargs.get("group_id", "LarvaGroup")
            payload = {"group_id": self.group_id, "sample": sample, **kwargs}
            self.data = util.AttrDict(payload)

        def entry(self, id=None, expand=False, as_entry=True, **_):
            key = id or self.group_id
            value = util.AttrDict({**self.data, "group_id": key})
            return util.AttrDict({key: value})

    class DummyExp:
        def __init__(self, **kwargs):
            from copy import deepcopy

            self.nestedConf = util.AttrDict(deepcopy(kwargs))

    monkeypatch.setattr(reg.gen, "LarvaGroup", DummyGroup)
    monkeypatch.setattr(reg.gen, "Exp", DummyExp)

    def dummy_diff(*_, **__):
        from pandas import DataFrame

        df = DataFrame([{"field": "dummy", "value": 0}])
        return df, ["dummy"]

    monkeypatch.setattr(plot_table, "diff_df", dummy_diff)
    monkeypatch.setattr(plot_pkg, "diff_df", dummy_diff)


def test_essay_dict_contains_expected_types(
    dataset_lock, monkeypatch, patched_generators
):
    """Essay_dict should expose all predefined essay experiment collections."""
    from larvaworld.lib.reg import config as reg_config
    from larvaworld.lib.reg.stored_confs.essay_conf import Essay_dict

    monkeypatch.setattr(reg_config, "next_idx", lambda *_, **__: 1)

    essays = Essay_dict()

    expected = {"RvsS", "DoublePatch", "Chemotaxis"}
    assert expected.issubset(set(essays.keys()))

    for name, exp_collection in essays.items():
        assert exp_collection, f"{name} essay produced no experiments"
        for exp_id, configs in exp_collection.items():
            assert configs, f"{name}/{exp_id} returned empty configuration list"


def test_doublepatch_exp_dict_structure(dataset_lock, patched_generators):
    """DoublePatch essay should build larva groups referencing default dataset."""
    from larvaworld.lib import reg
    from larvaworld.lib.reg.stored_confs.essay_conf import DoublePatch_Essay

    essay = DoublePatch_Essay(
        essay_id="integration_doublepatch",
        substrates=["standard"],
        N=2,
        olfactor=True,
        feeder=True,
    )

    assert set(essay.exp_dict.keys()) == {"standard"}
    configs = essay.exp_dict["standard"]
    assert len(configs) == 1

    conf = configs[0]
    groups = conf.larva_groups
    assert groups, "Larva group configuration missing"

    samples = {g.sample for g in groups.values()}
    assert samples == {reg.default_refID}


def test_chemotaxis_models_provide_variants(dataset_lock, patched_generators):
    """Chemotaxis essay should expose multiple model variants with metadata."""
    from larvaworld.lib.reg.stored_confs.essay_conf import Chemotaxis_Essay

    essay = Chemotaxis_Essay(
        essay_id="integration_chemotaxis",
        gain=150.0,
        mode=1,
    )

    assert essay.models, "Chemotaxis essay models missing"
    assert "controls" in essay.models

    for label, meta in essay.models.items():
        assert "model" in meta
        assert "color" in meta
        flattened_keys = meta.model.flatten().keys()
        assert any(
            key.startswith("brain") for key in flattened_keys
        ), f"{label} model lacks brain configuration entries"

    assert essay.exp_dict, "Chemotaxis experiment configurations missing"
    for configs in essay.exp_dict.values():
        assert configs, "Chemotaxis experiment list empty"
