from __future__ import annotations

import json
from pathlib import Path

import pytest

from larvaworld.gui_v2.apps.models_environments.env_builder_controller import (
    EnvBuilderController,
)
from larvaworld.lib import reg
from larvaworld.portal.workspace import initialize_workspace, set_active_workspace_path


def _sample_payload() -> dict[str, object]:
    return {
        "arena": {"geometry": "rectangular", "dims": [0.2, 0.2], "torus": False},
        "food_params": {
            "source_units": {
                "food_001": {
                    "pos": [0.01, 0.02],
                    "radius": 0.004,
                    "amount": 1.0,
                    "can_be_carried": True,
                    "can_be_displaced": False,
                    "regeneration": False,
                    "odor": {"id": "odor_a", "intensity": 0.5, "spread": 0.3},
                    "substrate": {"type": "standard", "quality": 1.0},
                    "color": "#4caf50",
                }
            },
            "source_groups": {
                "group_001": {
                    "radius": 0.006,
                    "amount": 2.0,
                    "can_be_carried": False,
                    "can_be_displaced": True,
                    "regeneration": True,
                    "distribution": {
                        "N": 5,
                        "loc": [0.0, -0.01],
                        "mode": "uniform",
                        "shape": "circle",
                        "scale": [0.012, 0.012],
                    },
                    "odor": {"id": "odor_b", "intensity": 0.2, "spread": 0.1},
                    "substrate": {"type": "standard", "quality": 0.9},
                    "color": "#6688aa",
                    "distribution_show_shape": True,
                }
            },
            "food_grid": {
                "unique_id": "grid",
                "color": "#c8e6c9",
                "fixed_max": True,
                "grid_dims": [31, 31],
                "initial_value": 0.1,
                "substrate": "standard",
            },
        },
        "border_list": {
            "border_001": {
                "vertices": [[-0.05, -0.05], [0.05, -0.05]],
                "width": 0.001,
                "color": "#333333",
            }
        },
        "odorscape": {"enabled": True, "mode": "Diffusion"},
        "windscape": {"enabled": False},
        "thermoscape": {"enabled": True},
    }


def test_default_controller_builds_canvas_state() -> None:
    controller = EnvBuilderController()

    assert controller.canvas_state.arena.geometry in {"rectangular", "circular"}
    assert isinstance(controller.object_rows, tuple)
    assert controller.export_json().endswith("\n")


def test_controller_preserves_scapes_during_object_edits() -> None:
    controller = EnvBuilderController()
    controller._apply_payload(_sample_payload(), "test")

    controller.add_source_unit(x=0.03, y=0.04)
    exported = json.loads(controller.export_json())

    assert exported["odorscape"] == {"enabled": True, "mode": "Diffusion"}
    assert exported["windscape"] == {"enabled": False}
    assert exported["thermoscape"] == {"enabled": True}
    assert exported["food_params"]["food_grid"]["unique_id"] == "grid"


def test_controller_loads_workspace_json(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    target = workspace_root / "environments" / "sample.json"
    target.write_text(json.dumps(_sample_payload()), encoding="utf-8")

    controller = EnvBuilderController()
    controller.load_workspace_json("sample")

    assert controller.preset_name == "sample"
    assert controller.current_row() is None
    assert controller.object_rows
    assert controller.payload["odorscape"] == {"enabled": True, "mode": "Diffusion"}

    saved = controller.save_workspace("Saved Preset")
    assert saved.name == "Saved_Preset.json"
    assert json.loads(saved.read_text(encoding="utf-8")) == controller.export_payload()


def test_controller_loads_registry_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = EnvBuilderController()
    monkeypatch.setattr(reg.conf.Env, "getID", lambda name: _sample_payload())

    controller.load_registry("custom_env")

    assert controller.preset_name == "custom_env"
    assert controller.payload["food_params"]["source_units"]["food_001"]["pos"] == [
        0.01,
        0.02,
    ]


def test_controller_crud_for_units_groups_and_borders() -> None:
    controller = EnvBuilderController()
    controller._apply_payload(
        {
            "arena": {"geometry": "rectangular", "dims": [0.2, 0.2], "torus": False},
            "food_params": {"source_units": {}, "source_groups": {}, "food_grid": None},
            "border_list": {},
            "odorscape": None,
            "windscape": None,
            "thermoscape": None,
        },
        "test",
    )

    unit = controller.add_source_unit(x=0.01, y=0.02, object_id="food_002")
    group = controller.add_source_group(x=-0.01, y=0.0, object_id="group_002")
    border = controller.add_border_segment(
        start=(-0.03, -0.03), end=(0.03, -0.03), object_id="border_002"
    )

    assert unit.object_id == "food_002"
    assert group.object_id == "group_002"
    assert border.object_id == "border_002"
    assert "food_002" in controller.payload["food_params"]["source_units"]
    assert "group_002" in controller.payload["food_params"]["source_groups"]
    assert "border_002" in controller.payload["border_list"]

    controller.select_object("group_002")
    controller.update_selected_object(color="#445566", amount=3.0)
    assert (
        controller.payload["food_params"]["source_groups"]["group_002"]["color"]
        == "#445566"
    )
    assert (
        controller.payload["food_params"]["source_groups"]["group_002"]["amount"] == 3.0
    )

    controller.delete_object("food_002")
    assert "food_002" not in controller.payload["food_params"]["source_units"]


def test_controller_set_arena_updates_geometry_and_dims() -> None:
    controller = EnvBuilderController()

    controller.set_arena(geometry="circular", width=0.1, torus=True)

    assert controller.payload["arena"]["geometry"] == "circular"
    assert controller.payload["arena"]["dims"] == (0.2, 0.2)
    assert controller.payload["arena"]["torus"] is True
    assert json.loads(controller.export_json())["arena"]["dims"] == [0.2, 0.2]
