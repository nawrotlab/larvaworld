from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from larvaworld.lib.sim.manifest import (
    MANIFEST_FILENAME,
    RunManifestSession,
    attach_manifest_to_datasets,
    canonical_sha256,
    capture_registry_snapshots,
    discover_run_manifests,
    derive_seed,
    load_run_manifest,
    resolve_manifest_id,
    RunManifestResolutionError,
    rerun_from_manifest,
    registry_snapshot_context,
    resolve_dataset_manifest_path,
    scientific_fingerprints,
    validate_run_manifest,
    _screen_options,
)
from larvaworld.portal.workspace import initialize_workspace


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))


def _fake_run(run_dir: Path, *, store_data: bool = True):
    return SimpleNamespace(
        dir=str(run_dir),
        id=run_dir.name,
        runtype="Exp",
        experiment="dish",
        store_data=store_data,
        parameters={"duration": 1.0},
        screen_kws={},
    )


def test_manifest_lifecycle_and_multi_workspace_discovery(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace", name="Primary")
    run = _fake_run(workspace.experiments_dir / "run-one")

    session = RunManifestSession(run=run, seed=123)
    running = load_run_manifest(session.manifest_path)
    assert running["run"]["status"] == "running"
    assert running["randomness"]["master_seed"] == 123
    assert running["run"]["workspace_id"] == workspace.workspace_id

    session.finish(scientific_result={"answer": 42})
    completed = load_run_manifest(session.manifest_path)
    assert completed["run"]["status"] == "completed"
    assert completed["run"]["completed_at"]

    records = discover_run_manifests(workspaces=[workspace])
    assert len(records) == 1
    assert records[0].valid is True
    assert records[0].manifest_id == completed["run"]["manifest_id"]


def test_manifest_only_run_removes_new_outputs(tmp_path: Path) -> None:
    run = _fake_run(tmp_path / "run", store_data=False)
    (Path(run.dir) / "plots").mkdir(parents=True)
    session = RunManifestSession(run=run, seed=1)
    generated = Path(run.dir) / "data" / "result.txt"
    generated.parent.mkdir(parents=True)
    generated.write_text("temporary output", encoding="utf-8")

    session.finish()

    assert not generated.exists()
    assert sorted(path.name for path in Path(run.dir).iterdir()) == [MANIFEST_FILENAME]


def test_manifest_only_run_preserves_only_explicitly_requested_media(
    tmp_path: Path,
) -> None:
    run = _fake_run(tmp_path / "run", store_data=False)
    run.screen_kws = {
        "save_video": True,
        "video_file": "capture",
        "media_dir": run.dir,
    }
    session = RunManifestSession(run=run, seed=1)
    video = Path(run.dir) / "capture.mp4"
    video.write_bytes(b"video")
    plot = Path(run.dir) / "plots" / "analysis.png"
    plot.parent.mkdir()
    plot.write_bytes(b"plot")

    session.finish()
    payload = load_run_manifest(session.manifest_path)

    assert video.exists()
    assert not plot.exists()
    assert payload["result"]["outputs"] == [
        {
            "path": "capture.mp4",
            "sha256": payload["result"]["outputs"][0]["sha256"],
            "size": 5,
            "media": True,
        }
    ]


def test_rerun_screen_options_never_reuse_source_media_paths(tmp_path: Path) -> None:
    invocation = {
        "runtime_options": {
            "screen_kws": {
                "show_display": True,
                "save_video": True,
                "vis_mode": "video",
                "image_mode": "overlap",
                "video_file": "/source/original.mp4",
                "image_file": "/source/original.png",
                "media_dir": "/source",
            }
        }
    }
    destination = tmp_path / "rerun"

    disabled = _screen_options(invocation, False, destination)
    enabled = _screen_options(invocation, True, destination)

    assert disabled["show_display"] is False
    assert disabled["save_video"] is False
    assert disabled["vis_mode"] is None
    assert disabled["image_mode"] is None
    assert "media_dir" not in disabled
    assert enabled["show_display"] is False
    assert enabled["media_dir"] == str(destination)
    assert enabled["video_file"] == "original"
    assert enabled["image_file"] == "original"


def test_nested_scientific_fingerprint_hashes_values_and_ignores_provenance() -> None:
    first = {
        "metrics": {"pooled": pd.DataFrame({"score": [1.0, 2.0]})},
        "manifest_id": "first",
        "path": "/old/run",
        "created_at": "2025-01-01T00:00:00Z",
    }
    equivalent = {
        "metrics": {"pooled": pd.DataFrame({"score": [1.0, 2.0]})},
        "manifest_id": "second",
        "path": "/new/run",
        "created_at": "2026-01-01T00:00:00Z",
    }
    changed = {"metrics": {"pooled": pd.DataFrame({"score": [1.0, 3.0]})}}

    first_hash = scientific_fingerprints(scientific_result=first)["result"]

    assert scientific_fingerprints(scientific_result=equivalent)["result"] == first_hash
    assert scientific_fingerprints(scientific_result=changed)["result"] != first_hash


def test_manifest_roundtrip_preserves_semantic_mapping_order(tmp_path: Path) -> None:
    run = _fake_run(tmp_path / "run")
    session = RunManifestSession(
        run=run,
        seed=1,
        invocation={
            "resolved_parameters": {
                "eval_metrics": {
                    "spatial displacement": ["v"],
                    "temporal dynamics": ["fsv"],
                    "stride cycle": ["str_d"],
                }
            }
        },
    )

    loaded = load_run_manifest(session.manifest_path)

    assert list(loaded["invocation"]["resolved_parameters"]["eval_metrics"]) == [
        "spatial displacement",
        "temporal dynamics",
        "stride cycle",
    ]


def test_registry_snapshot_context_restores_attrdict_dot_access() -> None:
    from larvaworld.lib import reg

    snapshot = capture_registry_snapshots()
    before_hash = canonical_sha256(reg.conf.Exp.dict)

    with registry_snapshot_context(snapshot):
        assert reg.conf.Exp.getID("dish").env_params is not None

    restored = reg.conf.Exp.getID("dish")
    group = next(iter(restored.larva_groups.values()))
    assert restored.env_params is not None
    assert group.model is not None
    assert canonical_sha256(reg.conf.Exp.dict) == before_hash


def test_strict_version_validation_requires_explicit_override(tmp_path: Path) -> None:
    run = _fake_run(tmp_path / "run")
    session = RunManifestSession(run=run, seed=1)
    session.finish()
    payload = load_run_manifest(session.manifest_path)
    payload["software"]["python"] = "0.0.0"
    session.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    strict = validate_run_manifest(session.manifest_path)
    overridden = validate_run_manifest(
        session.manifest_path, allow_version_mismatch=True
    )

    assert strict.valid is False
    assert any("version mismatch" in error.lower() for error in strict.errors)
    assert overridden.valid is True
    assert overridden.warnings


def test_validation_accepts_the_run_directory(tmp_path: Path) -> None:
    run = _fake_run(tmp_path / "run")
    session = RunManifestSession(run=run, seed=1)
    session.finish()

    report = validate_run_manifest(Path(run.dir))

    assert report.valid is True
    assert report.manifest_path == session.manifest_path


def test_dataset_reference_resolves_relative_path_after_move(tmp_path: Path) -> None:
    original = tmp_path / "run"
    run = _fake_run(original)
    session = RunManifestSession(run=run, seed=7)
    dataset_dir = original / "data" / "group"
    dataset_dir.mkdir(parents=True)
    dataset = SimpleNamespace(
        config=SimpleNamespace(dir=str(dataset_dir), id="group", provenance=None)
    )
    attach_manifest_to_datasets([dataset], session)
    session.finish()

    moved = tmp_path / "moved-run"
    original.rename(moved)
    dataset.config.dir = str(moved / "data" / "group")

    resolved = resolve_dataset_manifest_path(dataset, workspaces=[])
    assert resolved == moved / MANIFEST_FILENAME
    assert (
        dataset.config.provenance["run_manifest"]["path"] == "../../run_manifest.json"
    )


def test_all_run_datasets_share_manifest_and_rerun_uses_a_new_manifest(
    tmp_path: Path,
) -> None:
    source_session = RunManifestSession(run=_fake_run(tmp_path / "source"), seed=13)
    source_datasets = [
        SimpleNamespace(
            config=SimpleNamespace(
                id=dataset_id,
                dir=str(tmp_path / "source" / "data" / dataset_id),
                provenance=None,
            )
        )
        for dataset_id in ("a", "b")
    ]
    attach_manifest_to_datasets(source_datasets, source_session)
    source_session.finish(datasets=source_datasets)
    source_id = source_session.manifest["run"]["manifest_id"]

    rerun_session = RunManifestSession(
        run=_fake_run(tmp_path / "rerun"),
        seed=13,
        source_manifest=source_session.manifest_path,
    )
    rerun_dataset = SimpleNamespace(
        config=SimpleNamespace(
            id="a",
            dir=str(tmp_path / "rerun" / "data" / "a"),
            provenance=source_datasets[0].config.provenance,
        )
    )
    attach_manifest_to_datasets([rerun_dataset], rerun_session)
    rerun_session.finish(datasets=[rerun_dataset])
    rerun_id = rerun_session.manifest["run"]["manifest_id"]

    assert {
        dataset.config.provenance["run_manifest"]["manifest_id"]
        for dataset in source_datasets
    } == {source_id}
    assert rerun_id != source_id
    assert rerun_dataset.config.provenance["run_manifest"]["manifest_id"] == rerun_id


def test_dataset_manifest_uses_catalog_when_relative_path_is_missing(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    session = RunManifestSession(
        run=_fake_run(workspace.experiments_dir / "source-run"), seed=3
    )
    session.finish()
    dataset = SimpleNamespace(
        config=SimpleNamespace(
            dir=str(tmp_path / "detached" / "data"),
            provenance={
                "origin": "simulation",
                "run_manifest": {
                    "manifest_id": session.manifest["run"]["manifest_id"],
                    "workspace_id": workspace.workspace_id,
                    "path": "../../missing/run_manifest.json",
                },
                "lineage": [],
            },
        )
    )

    resolved = resolve_dataset_manifest_path(dataset, workspaces=[workspace])

    assert resolved == session.manifest_path


def test_dataset_manifest_rejects_mismatched_relative_manifest(tmp_path: Path) -> None:
    first = RunManifestSession(run=_fake_run(tmp_path / "first"), seed=1)
    first.finish()
    second = RunManifestSession(run=_fake_run(tmp_path / "second"), seed=2)
    second.finish()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    dataset = SimpleNamespace(
        config=SimpleNamespace(
            dir=str(dataset_dir),
            provenance={
                "origin": "simulation",
                "run_manifest": {
                    "manifest_id": first.manifest["run"]["manifest_id"],
                    "workspace_id": None,
                    "path": str(second.manifest_path),
                },
                "lineage": [],
            },
        )
    )

    with pytest.raises(RunManifestResolutionError, match="id mismatch"):
        resolve_dataset_manifest_path(dataset, workspaces=[])


def test_dataset_manifest_reports_missing_id(tmp_path: Path) -> None:
    dataset = SimpleNamespace(
        config=SimpleNamespace(
            dir=str(tmp_path / "dataset"),
            provenance={
                "origin": "simulation",
                "run_manifest": {
                    "manifest_id": "missing-id",
                    "workspace_id": None,
                    "path": "../../missing/run_manifest.json",
                },
                "lineage": [],
            },
        )
    )

    with pytest.raises(RunManifestResolutionError, match="No run manifest"):
        resolve_dataset_manifest_path(dataset, workspaces=[])


def test_catalog_reports_corrupt_manifest_and_unavailable_workspace(
    tmp_path: Path,
) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    corrupt = workspace.experiments_dir / "broken" / MANIFEST_FILENAME
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{broken", encoding="utf-8")
    missing = {
        "workspace_id": "missing",
        "name": "Missing",
        "path": tmp_path / "does-not-exist",
    }

    records = discover_run_manifests(workspaces=[workspace, missing])

    assert any(
        record.manifest_path == corrupt and not record.valid for record in records
    )
    assert any(
        not record.available and record.workspace_id == "missing" for record in records
    )


def test_input_checksum_is_strict_error_and_parameter_warning(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_file = input_dir / "data.txt"
    input_file.write_text("original", encoding="utf-8")
    from larvaworld.lib.sim.manifest import path_checksum

    checksum, files = path_checksum(input_dir)
    run = _fake_run(tmp_path / "run")
    session = RunManifestSession(
        run=run,
        seed=1,
        inputs=[
            {
                "role": "input_dataset",
                "dataset_id": "source",
                "path": str(input_dir),
                "sha256": checksum,
                "files": files,
            }
        ],
    )
    session.finish()
    input_file.write_text("changed", encoding="utf-8")

    strict = validate_run_manifest(session.manifest_path)
    parameters = validate_run_manifest(
        session.manifest_path, reproducibility="parameters"
    )

    assert strict.valid is False
    assert any("checksum mismatch" in error.lower() for error in strict.errors)
    assert parameters.valid is True
    assert any(
        "checksum mismatch" in warning.lower() for warning in parameters.warnings
    )


def test_child_seed_validation_is_strict_error_and_parameter_warning(
    tmp_path: Path,
) -> None:
    run = _fake_run(tmp_path / "run")
    run.runtype = "Batch"
    session = RunManifestSession(
        run=run,
        seed=1,
        child_seeds={"(None, None)": derive_seed(1, (None, None))},
    )
    session.finish()
    payload = load_run_manifest(session.manifest_path)
    payload["randomness"]["child_seeds"]["(None, None)"] += 1
    session.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    strict = validate_run_manifest(session.manifest_path)
    parameters = validate_run_manifest(
        session.manifest_path, reproducibility="parameters"
    )

    assert strict.valid is False
    assert any("deterministic derivation" in error.lower() for error in strict.errors)
    assert parameters.valid is True
    assert any(
        "deterministic derivation" in warning.lower() for warning in parameters.warnings
    )


def test_duplicate_manifest_ids_are_rejected_by_catalog_resolution(
    tmp_path: Path,
) -> None:
    first_workspace = initialize_workspace(tmp_path / "first")
    second_workspace = initialize_workspace(tmp_path / "second")
    session = RunManifestSession(
        run=_fake_run(first_workspace.experiments_dir / "run"), seed=1
    )
    session.finish()
    duplicate = second_workspace.experiments_dir / "copy" / MANIFEST_FILENAME
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text(
        session.manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    with pytest.raises(RunManifestResolutionError, match="ambiguous"):
        resolve_manifest_id(
            session.manifest["run"]["manifest_id"],
            workspaces=[first_workspace, second_workspace],
        )


@pytest.mark.parametrize("mode", ["Exp", "Replay", "Batch", "Ga", "Eval"])
def test_rerun_dispatches_every_simulation_mode(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from larvaworld.lib import sim as sim_module

    constructor: dict[str, object] = {}
    if mode == "Exp":
        constructor["parameter_dict"] = {}
    elif mode == "Replay":
        constructor["replay_parameters"] = {}
    elif mode == "Batch":
        constructor.update(
            {
                "space_search": {"parameter": 1},
                "space_kws": {},
                "exp": {},
                "exp_kws": {},
                "iterations": 1,
                "model_kwargs": {},
            }
        )
    elif mode == "Eval":
        constructor["eval_kwargs"] = {}
    invocation = {
        "resolved_parameters": {},
        "constructor": constructor,
        "runtime_options": {"store_data": False, "screen_kws": {}},
        "execute": {"method": "simulate", "kwargs": {"seed": 42}},
    }
    source_run = SimpleNamespace(
        dir=str(tmp_path / f"source-{mode.lower()}"),
        id=f"source-{mode.lower()}",
        runtype=mode,
        experiment="dish",
        store_data=False,
        parameters={},
        screen_kws={},
    )
    child_seeds = {
        "Batch": {"(None, None)": derive_seed(42, (None, None))},
        "Ga": {"generation_1": derive_seed(42, ("generation", 1))},
        "Eval": {"evaluation_exp": derive_seed(42, "evaluation_exp")},
    }.get(mode, {})
    source_session = RunManifestSession(
        run=source_run,
        invocation=invocation,
        seed=42,
        child_seeds=child_seeds,
    )
    source_session.finish(scientific_result={"mode": mode})

    class FakeRun:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.runtype = mode
            self.experiment = kwargs.get("experiment", "dish")
            self.id = kwargs["id"]
            self.dir = kwargs["dir"]
            self.store_data = kwargs["store_data"]
            self.screen_kws = kwargs.get("screen_kws", {})

        def _execute(self, seed):
            session = RunManifestSession(
                run=self,
                invocation=invocation,
                seed=seed,
                child_seeds=child_seeds,
                source_manifest=self.kwargs["_source_manifest"],
            )
            session.finish(scientific_result={"mode": mode})
            self.seed = seed
            return mode

        def simulate(self, seed=None, **kwargs):
            return self._execute(seed)

        def run(self, seed=None, **kwargs):
            return self._execute(seed)

    class_name = {
        "Exp": "ExpRun",
        "Replay": "ReplayRun",
        "Batch": "BatchRun",
        "Ga": "GAlauncher",
        "Eval": "EvalRun",
    }[mode]
    monkeypatch.setattr(sim_module, class_name, FakeRun)

    rerun = rerun_from_manifest(source_session.manifest_path)
    payload = load_run_manifest(rerun.manifest_path)

    assert rerun.result == mode
    assert rerun.run.seed == 42
    assert rerun.comparison["matches"] is True
    assert payload["run"]["mode"] == mode
    assert (
        payload["provenance"]["source_manifest"]["manifest_id"]
        == (source_session.manifest["run"]["manifest_id"])
    )
