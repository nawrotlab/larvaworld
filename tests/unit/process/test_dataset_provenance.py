from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from larvaworld.lib.process.dataset import DatasetConfig, LarvaDataset
from larvaworld.lib.sim.manifest import RunManifestSession, append_dataset_lineage


def test_dataset_config_provenance_roundtrip() -> None:
    provenance = {
        "origin": "simulation",
        "run_manifest": {
            "manifest_id": "manifest-1",
            "workspace_id": "workspace-1",
            "path": "../../run_manifest.json",
        },
        "lineage": [],
    }
    config = DatasetConfig(
        id="dataset", dir="/tmp/dataset", agent_ids=[], provenance=provenance
    )

    restored = DatasetConfig(**config.nestedConf)

    assert restored.provenance == provenance


def test_old_dataset_config_without_provenance_is_supported() -> None:
    config = DatasetConfig(id="legacy", dir="/tmp/legacy", agent_ids=[])

    assert config.provenance is None


def test_derived_lineage_preserves_manifest_and_recalculates_path(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "run" / "run_manifest.json"
    parent_dir = tmp_path / "run" / "data" / "parent"
    derived_dir = tmp_path / "derived" / "child"
    parent_dir.mkdir(parents=True)
    derived_dir.mkdir(parents=True)
    run = SimpleNamespace(
        dir=str(manifest_path.parent),
        id="run",
        runtype="Exp",
        experiment="dish",
        store_data=True,
        parameters={},
        screen_kws={},
    )
    session = RunManifestSession(run=run, seed=1)
    session.finish()
    parent = SimpleNamespace(
        config=SimpleNamespace(
            id="parent",
            dir=str(parent_dir),
            provenance={
                "origin": "simulation",
                "run_manifest": {
                    "manifest_id": session.manifest["run"]["manifest_id"],
                    "workspace_id": session.manifest["run"]["workspace_id"],
                    "path": "../../run_manifest.json",
                },
                "lineage": [],
            },
        )
    )
    derived = SimpleNamespace(
        config=SimpleNamespace(id="child", dir=str(derived_dir), provenance=None)
    )

    append_dataset_lineage(
        parent,
        derived,
        operation="timeseries_slice",
        parameters={"start": 10, "stop": 120},
    )

    assert (
        derived.config.provenance["run_manifest"]["manifest_id"]
        == session.manifest["run"]["manifest_id"]
    )
    assert derived.config.provenance["lineage"] == [
        {
            "operation": "timeseries_slice",
            "parent_dataset_id": "parent",
            "created_at": derived.config.provenance["lineage"][0]["created_at"],
            "parameters": {"start": 10, "stop": 120},
        }
    ]
    # Manifest paths are stored POSIX-style so that provenance written on one
    # platform resolves on another.
    assert (
        derived.config.provenance["run_manifest"]["path"]
        == "../../run/run_manifest.json"
    )


def test_subsample_and_slice_append_lineage_and_keep_manifest_access(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    session = RunManifestSession(
        run=SimpleNamespace(
            dir=str(run_dir),
            id="run",
            runtype="Exp",
            experiment="dish",
            store_data=True,
            parameters={},
            screen_kws={},
        ),
        seed=7,
    )
    session.finish()
    agent_ids = ["a", "b", "c"]
    index = pd.MultiIndex.from_product([range(5), agent_ids], names=["Step", "AgentID"])
    step = pd.DataFrame({"x": range(15), "y": range(15)}, index=index)
    end = pd.DataFrame(
        {"length": [1.0, 2.0, 3.0]},
        index=pd.Index(agent_ids, name="AgentID"),
    )
    parent = LarvaDataset(
        dir=str(run_dir / "data" / "parent"),
        id="parent",
        agent_ids=agent_ids,
        dt=1.0,
        Nsteps=5,
        step=step,
        end=end,
        provenance={
            "origin": "simulation",
            "run_manifest": {
                "manifest_id": session.manifest["run"]["manifest_id"],
                "workspace_id": session.manifest["run"]["workspace_id"],
                "path": "../../run_manifest.json",
            },
            "lineage": [],
        },
        load_data=False,
    )

    subsample = parent.derive_subsample(
        2,
        new_id="subsample",
        new_dir=str(tmp_path / "derived" / "subsample"),
        seed=11,
    )
    sliced = subsample.derive_timeseries_slice(
        (1, 3),
        new_id="slice",
        new_dir=str(tmp_path / "derived" / "slice"),
    )

    assert [entry["operation"] for entry in sliced.config.provenance["lineage"]] == [
        "subsample",
        "timeseries_slice",
    ]
    assert (
        sliced.load_run_manifest()["run"]["manifest_id"]
        == (session.manifest["run"]["manifest_id"])
    )
    reloaded = LarvaDataset(dir=sliced.dir, load_data=True)
    assert reloaded.config.provenance == sliced.config.provenance
    assert reloaded.run_manifest_path == session.manifest_path
