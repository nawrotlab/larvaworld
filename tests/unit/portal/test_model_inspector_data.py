from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from larvaworld.lib import reg, util
from larvaworld.lib.model import DefaultBrain
from larvaworld.portal.models_architecture import model_inspector_data as data
from larvaworld.portal.models_architecture.model_inspector_data import (
    BASELINE_MODULES,
    compare_model_inspections,
    inspect_model,
    list_model_ids,
    run_model_probe,
)
from larvaworld.portal.workspace import clear_active_workspace_path


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    clear_active_workspace_path()


def _legacy_equivalent_probe(
    model_id: str, *, steps: int = 501, dt: float = 0.1, a_in: float = 0.0
) -> tuple[pd.DataFrame, dict[str, bool]]:
    model = reg.conf.Model.getID(model_id)
    brain = DefaultBrain(conf=model.brain.get_copy(), dt=dt)
    runtime = SimpleNamespace(brain=brain)
    available_from_registry = reg.par.output_reporters(
        ks=["A_T", "A_C"], agents=[runtime]
    )
    available_paths = set(available_from_registry.values())
    reporter_paths: dict[str, str] = {}
    for key in ("A_T", "A_C"):
        try:
            reporter_paths[key] = reg.par.kdict[key].codename
        except Exception:
            reporter_paths[key] = ""
    reporter_available: dict[str, bool] = {}
    for key, path in reporter_paths.items():
        available = bool(path) and path in available_paths
        if not available and path:
            try:
                util.rgetattr(runtime, path)
                available = True
            except Exception:
                available = False
        reporter_available[key] = available

    rows = []
    for tick in range(steps):
        lin, ang, feed_motion = brain.locomotor.step(A_in=a_in)
        row = {
            "time": tick * dt,
            "lin": lin,
            "ang": ang,
            "feed_motion": bool(feed_motion),
        }
        for key, path in reporter_paths.items():
            row[key] = util.rgetattr(runtime, path) if reporter_available[key] else None
        rows.append(row)
    return pd.DataFrame(rows), reporter_available


def test_list_model_ids_returns_registry_ids() -> None:
    ids = list_model_ids()
    assert ids == list(reg.conf.Model.confIDs)
    assert len(ids) > 0


def test_inspect_model_explorer_includes_baseline_modules() -> None:
    if "explorer" not in reg.conf.Model.confIDs:
        pytest.skip('"explorer" model is not available in this environment.')
    inspection = inspect_model("explorer")
    baseline_ids = [module.module_id for module in inspection.baseline_modules]
    assert tuple(baseline_ids) == BASELINE_MODULES
    assert all(module.is_baseline for module in inspection.baseline_modules)


def test_inspect_model_marks_absent_baseline_modules_explicitly() -> None:
    if "explorer" not in reg.conf.Model.confIDs:
        pytest.skip('"explorer" model is not available in this environment.')
    inspection = inspect_model("explorer")
    assert any(module.present for module in inspection.baseline_modules)
    for module in inspection.baseline_modules:
        assert module.present in {True, False}


def test_optional_module_detection_when_configured() -> None:
    for model_id in reg.conf.Model.confIDs:
        inspection = inspect_model(model_id)
        if inspection.optional_modules:
            assert all(not module.is_baseline for module in inspection.optional_modules)
            return
    pytest.skip("No model with configured optional modules was found.")


def test_compare_model_inspections_same_model_is_equal() -> None:
    model_id = list_model_ids()[0]
    a = inspect_model(model_id)
    b = inspect_model(model_id)
    diffs = compare_model_inspections(a, b)
    assert len(diffs) >= len(BASELINE_MODULES)
    assert all(diff.equal for diff in diffs)


def test_compare_model_inspections_detects_changes() -> None:
    ids = list_model_ids()
    if len(ids) < 2:
        pytest.skip("Need at least two models to compare.")
    primary = "explorer" if "explorer" in ids else ids[0]
    comparison = (
        "navigator" if "navigator" in ids and primary != "navigator" else ids[1]
    )
    diffs = compare_model_inspections(inspect_model(primary), inspect_model(comparison))
    assert any(not diff.equal for diff in diffs)


def test_probe_output_schema_and_reporter_metadata() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    result = run_model_probe(model_id, steps=5, dt=0.1, a_in=0.0)
    assert list(result.dataframe.columns)[:4] == ["time", "lin", "ang", "feed_motion"]
    assert "A_T" in result.dataframe.columns
    assert "A_C" in result.dataframe.columns
    assert set(result.reporter_available.keys()) == {"A_T", "A_C"}


def test_probe_parity_with_legacy_equivalent_runner() -> None:
    model_id = (
        "explorer" if "explorer" in reg.conf.Model.confIDs else list_model_ids()[0]
    )
    steps = 25
    dt = 0.1
    a_in = 0.0

    random.seed(1234)
    np.random.seed(1234)
    reference_df, reference_available = _legacy_equivalent_probe(
        model_id, steps=steps, dt=dt, a_in=a_in
    )

    random.seed(1234)
    np.random.seed(1234)
    probe = run_model_probe(model_id, steps=steps, dt=dt, a_in=a_in)

    assert probe.reporter_available == reference_available
    assert probe.dataframe.shape == reference_df.shape
    assert list(probe.dataframe.columns) == list(reference_df.columns)
    assert np.array_equal(
        probe.dataframe["feed_motion"].to_numpy(),
        reference_df["feed_motion"].to_numpy(),
    )
    for column in ["time", "lin", "ang"]:
        left = probe.dataframe[column].to_numpy()
        right = reference_df[column].to_numpy()
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)

    for column in ["A_T", "A_C"]:
        left = probe.dataframe[column].to_numpy()
        right = reference_df[column].to_numpy()
        if reference_available.get(column, False):
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
        else:
            assert pd.isna(left).all()
            assert pd.isna(right).all()


def test_new_data_module_does_not_import_legacy_dashboard_source() -> None:
    source = Path(data.__file__).read_text(encoding="utf-8")
    assert "larvaworld.dashboards.model_inspector" not in source
