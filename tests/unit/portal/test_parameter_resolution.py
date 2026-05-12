from __future__ import annotations

import json
from pathlib import Path

import pytest

from larvaworld.lib import reg, util
from larvaworld.portal.simulation.parameter_resolution import (
    resolve_base_experiment_parameters,
)
from larvaworld.portal.simulation.single_experiment_app import (
    _SingleExperimentController,
)
from larvaworld.portal.workspace import (
    clear_active_workspace_path,
    initialize_workspace,
    set_active_workspace_path,
)


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    clear_active_workspace_path()


def test_parameter_resolution_default_template_matches_controller_base(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.environment_select.value = "__template__"

    resolved = resolve_base_experiment_parameters(
        controller._selected_experiment(),
        controller._load_selected_environment(),
    )
    expected = reg.conf.Exp.getID("dish").get_copy()
    expected["duration"] = float(expected.get("duration", 5.0))

    assert isinstance(resolved, util.AttrDict)
    assert isinstance(resolved.duration, float)
    assert resolved.flatten() == expected.flatten()


def test_parameter_resolution_workspace_environment_override_preserves_behavior(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    (workspace_root / "environments" / "rect_env.json").write_text(
        json.dumps(
            {
                "arena": {"geometry": "rectangular", "dims": [0.2, 0.1]},
                "food_params": {
                    "source_units": {
                        "patch": {
                            "pos": [0.02, 0.0],
                            "radius": 0.005,
                            "amount": 2.0,
                            "odor": {"id": "apple", "intensity": 1.0, "spread": 0.02},
                            "substrate": {"type": "standard", "quality": 1.0},
                            "color": "#44aa55",
                        }
                    },
                    "source_groups": {},
                    "food_grid": {},
                },
                "border_list": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    controller.environment_select.value = controller.environment_select.options[
        "Workspace / rect_env"
    ]
    assert controller.environment_preset_controls.load_selected() is True
    payload = controller._load_selected_environment()

    resolved = resolve_base_experiment_parameters("dish", payload)

    assert resolved.env_params.arena.geometry == "rectangular"
    assert resolved.env_params.arena.dims == (0.2, 0.1)
    assert resolved.env_params.food_params.source_units["patch"]["pos"] == (0.02, 0.0)
    assert isinstance(resolved.duration, float)


def test_parameter_resolution_registry_environment_override_preserves_behavior(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.environment_select.value = controller.environment_select.options[
        "Registry / maze"
    ]
    assert controller.environment_preset_controls.load_selected() is True
    payload = controller._load_selected_environment()

    resolved = resolve_base_experiment_parameters("dish", payload)

    assert resolved.env_params.arena.geometry == "rectangular"
    assert "Maze" in resolved.env_params.border_list
    assert isinstance(resolved.duration, float)
