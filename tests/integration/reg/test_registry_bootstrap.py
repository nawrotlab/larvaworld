"""
Integration tests for registry bootstrap and configuration behaviour.

These tests rely on the ensure_datasets_ready fixture (auto-applied for the
integration suite) so that the Schleyer processed datasets exist before the
registry is exercised.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.fast]


def test_default_reference_dataset_available(dataset_lock):
    """LarvaDataset refID should load and expose persisted artefacts."""
    import importlib

    # Cold start safety: ensure process dataset module is loaded after registry bootstrap
    dataset_module = importlib.import_module("larvaworld.lib.process.dataset")
    LarvaDataset = dataset_module.LarvaDataset

    dataset = LarvaDataset(refID="exploration.30controls", load_data=False)
    data_dir = Path(dataset.config.dir) / "data"

    assert data_dir.exists(), "processed directory missing"
    assert (data_dir / "data.h5").exists(), "processed HDF5 not found"
    assert (data_dir / "conf.txt").exists(), "processed conf.txt not found"


def test_larvagroup_accepts_dict_model(dataset_lock):
    """Regression test for dict-valued model parameter (fixed in PR-3E)."""
    from larvaworld.lib import reg
    from larvaworld.lib.reg.larvagroup import LarvaGroup

    model_id = reg.conf.Model.confIDs[0]
    model_conf = reg.conf.Model.getID(model_id)

    group = LarvaGroup(model=model_conf, group_id="dict_model_test")
    assert isinstance(group.model, dict)
    assert group.group_id == "dict_model_test"
    # Expanded model should equal the provided configuration
    assert group.expanded_model == model_conf


def test_reference_dataset_loads_via_registry(dataset_lock):
    """Reference dataset should load through the registry API and expose files on disk."""
    from larvaworld.lib import reg

    dataset = reg.conf.Ref.loadRef(id="exploration.30controls", load=False)
    dataset_dir = Path(dataset.config.dir)

    assert dataset_dir.exists()
    assert (dataset_dir / "data" / "data.h5").exists()
