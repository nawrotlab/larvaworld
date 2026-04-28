from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from larvaworld.lib import reg, util
from larvaworld.lib.reg import config as reg_config
from larvaworld.portal.models_architecture.environment_builder_app import (
    _EnvironmentBuilderController,
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


@pytest.fixture(autouse=True)
def isolated_env_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(reg_config, "CONF_DIR", str(registry_root))
    original_env_dict = util.AttrDict(reg.conf.Env.dict).get_copy()
    reg.conf.Env.set_dict(util.AttrDict())
    try:
        yield
    finally:
        reg.conf.Env.dict = original_env_dict


def test_environment_builder_saves_preset_to_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller.preset_name.value = "Arena Alpha"
    controller._add_point_object(
        object_type="Source unit",
        x=0.01,
        y=-0.02,
        radius=0.008,
        color="#4caf50",
    )
    controller._add_border_object(
        x0=-0.03,
        y0=-0.03,
        x1=0.04,
        y1=0.04,
        width=0.001,
        color="#111111",
    )

    controller._on_save_preset(None)

    preset_path = workspace_root / "environments" / "Arena_Alpha.json"
    assert preset_path.is_file()
    payload = json.loads(preset_path.read_text(encoding="utf-8"))
    assert payload["arena"]["geometry"] == "rectangular"
    assert payload["arena"]["torus"] is False
    assert "food_001" in payload["food_params"]["source_units"]
    assert "border_002" in payload["border_list"]
    assert "Arena_Alpha" in reg.conf.Env.dict
    registry_payload = reg.conf.Env.dict["Arena_Alpha"]
    assert registry_payload["arena"]["dims"] == (0.2, 0.2)
    assert "food_001" in registry_payload["food_params"]["source_units"]
    assert controller.preset_select.value == "Arena_Alpha.json"
    assert "Saved environment preset" in controller.status.object


def test_environment_builder_loads_preset_from_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    preset_path = workspace_root / "environments" / "demo_env.json"
    preset_path.write_text(
        json.dumps(
            {
                "arena": {"geometry": "circular", "dims": [0.3, 0.3], "torus": True},
                "food_params": {
                    "source_units": {
                        "food_custom": {
                            "pos": [0.02, -0.01],
                            "radius": 0.012,
                            "color": "#88cc44",
                        }
                    },
                    "source_groups": {},
                    "food_grid": {},
                },
                "border_list": {
                    "border_custom": {
                        "vertices": [[-0.05, 0.01], [0.05, 0.01]],
                        "width": 0.002,
                        "color": "#222222",
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    controller = _EnvironmentBuilderController()
    controller.preset_select.value = "demo_env.json"

    controller._on_load_preset(None)

    assert controller.arena_shape.value == "circular"
    assert controller.arena_torus.value is True
    assert controller.arena_width.name == "Arena radius (m)"
    assert controller.arena_height.visible is False
    assert controller.arena_width.value == pytest.approx(0.15)
    assert controller.arena_height.value == pytest.approx(0.15)
    assert {obj.object_id for obj in controller._objects} == {
        "food_custom",
        "border_custom",
    }
    assert controller.food_source.data["id"] == ["food_custom"]
    assert controller.odor_peak_source.data["id"] == []
    assert controller.border_source.data["id"] == ["border_custom"]
    assert controller.preset_name.value == "demo_env"
    assert 'Loaded environment preset "demo_env"' in controller.status.object


def test_environment_builder_loads_environment_from_local_json_file() -> None:
    controller = _EnvironmentBuilderController()
    payload = {
        "arena": {"geometry": "circular", "dims": [0.24, 0.24], "torus": True},
        "food_params": {
            "source_units": {
                "food_file": {
                    "pos": [0.02, -0.01],
                    "radius": 0.012,
                    "color": "#88cc44",
                }
            },
            "source_groups": {},
            "food_grid": {},
        },
        "border_list": {
            "border_file": {
                "vertices": [[-0.05, 0.01], [0.05, 0.01]],
                "width": 0.002,
                "color": "#222222",
            }
        },
    }
    controller.load_file_input.filename = "import_env.json"
    controller.load_file_input.value = (json.dumps(payload) + "\n").encode("utf-8")
    controller._on_load_file(None)  # type: ignore[arg-type]

    assert controller.arena_shape.value == "circular"
    assert controller.arena_torus.value is True
    assert controller.arena_width.value == pytest.approx(0.12)
    assert controller.arena_height.value == pytest.approx(0.12)
    assert "food_file" in {obj.object_id for obj in controller._objects}
    assert "border_file" in {obj.object_id for obj in controller._objects}
    assert controller.preset_name.value == "import_env"
    assert 'Loaded environment file "import_env"' in controller.status.object


def test_environment_builder_loads_registry_environment_without_workspace() -> None:
    reg.conf.Env.set_dict(
        util.AttrDict(
            {
                "registry_env": {
                    "arena": {
                        "geometry": "rectangular",
                        "dims": (0.25, 0.15),
                        "torus": True,
                    },
                    "food_params": {
                        "source_units": {
                            "food_registry": {
                                "pos": (0.02, -0.01),
                                "radius": 0.012,
                                "color": "#88cc44",
                            }
                        },
                        "source_groups": {},
                        "food_grid": {},
                    },
                    "border_list": {
                        "barrier": {
                            "vertices": [
                                (-0.05, 0.01),
                                (0.05, 0.01),
                                (0.05, 0.01),
                                (0.05, 0.05),
                            ],
                            "width": 0.002,
                            "color": "#222222",
                        }
                    },
                }
            }
        )
    )

    controller = _EnvironmentBuilderController()

    assert controller.load_preset_btn.disabled is False
    assert "Registry / registry_env" in controller.preset_select.options
    controller.preset_select.value = "__registry__:registry_env"

    controller._on_load_preset(None)

    assert controller.arena_shape.value == "rectangular"
    assert controller.arena_torus.value is True
    assert controller.arena_width.value == pytest.approx(0.25)
    assert controller.arena_height.value == pytest.approx(0.15)
    assert {obj.object_id for obj in controller._objects} == {
        "food_registry",
        "barrier_001",
        "barrier_002",
    }
    assert controller.food_source.data["id"] == ["food_registry"]
    assert controller.odor_peak_source.data["id"] == []
    assert controller.border_source.data["id"] == ["barrier_001", "barrier_002"]
    assert 'Loaded registry environment "registry_env"' in controller.status.object


def test_environment_builder_deletes_workspace_preset_and_registry_entry(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller.preset_name.value = "Arena Alpha"
    controller._add_point_object(
        object_type="Source unit",
        x=0.01,
        y=-0.02,
        radius=0.008,
        color="#4caf50",
    )
    controller._on_save_preset(None)

    preset_path = workspace_root / "environments" / "Arena_Alpha.json"
    assert preset_path.is_file()
    assert "Arena_Alpha" in reg.conf.Env.dict

    controller.preset_select.value = "Arena_Alpha.json"
    controller._on_delete_preset(None)

    assert not preset_path.exists()
    assert "Arena_Alpha" not in reg.conf.Env.dict
    assert "Workspace / Arena_Alpha" not in controller.preset_select.options
    assert (
        'Deleted preset "Arena_Alpha" from workspace and registry.'
        == controller.status.object
    )


def test_environment_builder_deletes_registry_only_preset() -> None:
    reg.conf.Env.set_dict(
        util.AttrDict(
            {
                "registry_env": {
                    "arena": {
                        "geometry": "rectangular",
                        "dims": (0.25, 0.15),
                        "torus": True,
                    },
                    "food_params": {
                        "source_units": {},
                        "source_groups": {},
                        "food_grid": {},
                    },
                    "border_list": {},
                }
            }
        )
    )

    controller = _EnvironmentBuilderController()
    controller.preset_select.value = "__registry__:registry_env"

    controller._on_delete_preset(None)

    assert "registry_env" not in reg.conf.Env.dict
    assert "Registry / registry_env" not in controller.preset_select.options
    assert controller.status.object == 'Deleted preset "registry_env" from registry.'


def test_environment_builder_reset_configurations_recreates_env_registry() -> None:
    reg.conf.Env.set_dict(
        util.AttrDict(
            {
                "custom_env": {
                    "arena": {
                        "geometry": "rectangular",
                        "dims": (0.18, 0.12),
                        "torus": False,
                    },
                    "food_params": {
                        "source_units": {},
                        "source_groups": {},
                        "food_grid": None,
                    },
                    "border_list": {},
                    "odorscape": None,
                    "windscape": None,
                    "thermoscape": None,
                }
            }
        )
    )
    custom_path = Path(reg.conf.Env.path_to_dict)
    assert custom_path.is_file()

    controller = _EnvironmentBuilderController()
    assert "Registry / custom_env" in controller.preset_select.options

    controller._on_clear_all(None)
    assert controller.reset_confirm_panel.visible is True
    controller._on_confirm_reset_configurations(None)

    assert "custom_env" not in reg.conf.Env.dict
    assert "dish" in reg.conf.Env.dict
    assert custom_path.is_file()
    assert util.load_dict(str(custom_path)) == reg.conf.Env.dict
    assert "Registry / custom_env" not in controller.preset_select.options
    assert "Registry / dish" in controller.preset_select.options
    assert (
        controller.status.object
        == "Cleared all placed objects and recreated the Env registry."
    )


def test_environment_builder_reset_requires_explicit_confirmation() -> None:
    reg.conf.Env.set_dict(
        util.AttrDict(
            {
                "custom_env": {
                    "arena": {
                        "geometry": "rectangular",
                        "dims": (0.18, 0.12),
                        "torus": False,
                    },
                    "food_params": {
                        "source_units": {},
                        "source_groups": {},
                        "food_grid": None,
                    },
                    "border_list": {},
                    "odorscape": None,
                    "windscape": None,
                    "thermoscape": None,
                }
            }
        )
    )
    controller = _EnvironmentBuilderController()
    controller._add_point_object(
        object_type="Source unit",
        x=0.01,
        y=0.02,
        radius=0.008,
        color="#4caf50",
    )

    controller._on_clear_all(None)

    assert "custom_env" in reg.conf.Env.dict
    assert controller._objects
    assert controller.reset_confirm_panel.visible is True
    assert controller._pending_reset_confirmation is True
    assert "Reset requested" in controller.status.object

    controller._on_cancel_reset_configurations(None)
    assert controller.reset_confirm_panel.visible is False
    assert controller._pending_reset_confirmation is False
    assert controller.status.object == "Reset configurations cancelled."


def test_environment_builder_clear_arena_keeps_registry_intact() -> None:
    reg.conf.Env.set_dict(
        util.AttrDict(
            {
                "custom_env": {
                    "arena": {"geometry": "rectangular", "dims": (0.2, 0.2)},
                    "food_params": {
                        "source_units": {},
                        "source_groups": {},
                        "food_grid": None,
                    },
                    "border_list": {},
                    "odorscape": None,
                    "windscape": None,
                    "thermoscape": None,
                }
            }
        )
    )
    controller = _EnvironmentBuilderController()
    controller._add_point_object(
        object_type="Source unit",
        x=0.01,
        y=0.02,
        radius=0.008,
        color="#4caf50",
    )
    controller._add_border_object(
        x0=-0.03,
        y0=-0.03,
        x1=0.04,
        y1=0.04,
        width=0.001,
        color="#111111",
    )

    controller._on_clear_arena(None)

    assert controller._objects == []
    assert controller.food_source.data["id"] == []
    assert controller.border_source.data["id"] == []
    assert "custom_env" in reg.conf.Env.dict
    assert controller.status.object == "Cleared all arena objects."


def test_environment_builder_loads_source_groups_and_food_grid(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    preset_path = workspace_root / "environments" / "group_env.json"
    preset_path.write_text(
        json.dumps(
            {
                "arena": {
                    "geometry": "rectangular",
                    "dims": [0.2, 0.2],
                    "torus": False,
                },
                "food_params": {
                    "source_units": {},
                    "source_groups": {
                        "group_a": {
                            "radius": 0.003,
                            "amount": 0.0,
                            "distribution": {
                                "N": 12,
                                "loc": [0.01, -0.02],
                                "mode": "uniform",
                                "scale": [0.02, 0.03],
                                "shape": "oval",
                            },
                            "odor": {"id": "odor_a", "intensity": 1.2, "spread": 0.04},
                            "substrate": {"type": "cornmeal", "quality": 0.8},
                            "color": "#445566",
                        }
                    },
                    "food_grid": {
                        "unique_id": "FoodGrid",
                        "color": "#66aa33",
                        "fixed_max": True,
                        "grid_dims": [31, 29],
                        "initial_value": 0.002,
                        "substrate": {"type": "standard", "quality": 0.6},
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    controller = _EnvironmentBuilderController()
    controller.preset_select.value = "group_env.json"

    controller._on_load_preset(None)

    assert {obj.object_type for obj in controller._objects} == {"Source group"}
    assert controller.source_group_circle_source.data["id"] == []
    assert controller.source_group_ellipse_source.data["id"] == ["group_a"]
    assert controller.source_group_rect_source.data["id"] == []
    assert controller.odor_layer_source.data["id"]
    assert controller.odor_peak_source.data["id"] == ["group_a"] * 12
    assert controller.food_grid_enabled.value is True
    assert controller.food_grid_dims_x.value == 31
    assert controller.food_grid_dims_y.value == 29
    assert controller.food_grid_initial_value.value == pytest.approx(0.002)


def test_environment_builder_normalizes_group_shapes_and_exports_torus(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    assert controller.group_shape.options == ["circle", "oval", "rect"]
    assert controller.selected_distribution_shape.options == ["circle", "oval", "rect"]

    controller.arena_torus.value = True
    controller.object_type.value = "Source group"
    controller.group_shape.value = "circle"
    controller._add_point_object(
        object_type="Source group",
        x=0.01,
        y=-0.01,
        radius=0.004,
        color="#3355aa",
    )

    assert controller.source_group_circle_source.data["id"] == ["group_001"]
    assert controller.source_group_ellipse_source.data["id"] == []
    assert controller.source_group_rect_source.data["id"] == []

    payload = controller._build_export_config()
    assert payload["arena"]["torus"] is True
    assert payload["arena"]["dims"] == [0.2, 0.2]
    assert (
        payload["food_params"]["source_groups"]["group_001"]["distribution"]["shape"]
        == "circle"
    )


def test_environment_builder_circular_arena_uses_radius_control(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller.arena_shape.value = "circular"

    assert controller.arena_width.name == "Arena radius (m)"
    assert controller.arena_height.visible is False
    assert controller.arena_width.value == pytest.approx(0.1)

    payload = controller._build_export_config()
    assert payload["arena"]["geometry"] == "circular"
    assert payload["arena"]["dims"] == [0.2, 0.2]


def test_environment_builder_circle_groups_use_single_radius_control(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller.object_type.value = "Source group"
    controller.group_shape.value = "circle"
    controller.group_spread_x.value = 18.0

    assert controller.group_spread_x.name == "Group radius (mm)"
    assert controller.group_spread_y.visible is False
    assert controller.group_spread_y.value == pytest.approx(18.0)

    controller._add_point_object(
        object_type="Source group",
        x=0.01,
        y=-0.01,
        radius=0.004,
        color="#3355aa",
    )

    obj = controller._objects[0]
    assert obj.distribution_shape == "circle"
    assert obj.distribution_scale_x == pytest.approx(obj.distribution_scale_y)

    payload = controller._build_export_config()
    distribution = payload["food_params"]["source_groups"]["group_001"]["distribution"]
    assert distribution["shape"] == "circle"
    assert distribution["scale"] == [pytest.approx(0.018), pytest.approx(0.018)]


def test_environment_builder_oval_groups_use_width_and_height_controls(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller.object_type.value = "Source group"
    controller.group_shape.value = "oval"
    controller.group_spread_x.value = 40.0
    controller.group_spread_y.value = 24.0

    assert controller.group_spread_x.name == "Group width (mm)"
    assert controller.group_spread_y.name == "Group height (mm)"
    assert controller.group_spread_y.visible is True

    controller._add_point_object(
        object_type="Source group",
        x=0.01,
        y=-0.01,
        radius=0.004,
        color="#3355aa",
    )

    obj = controller._objects[0]
    assert obj.distribution_shape == "oval"
    assert obj.distribution_scale_x == pytest.approx(0.02)
    assert obj.distribution_scale_y == pytest.approx(0.012)

    controller._set_selected_object("group_001")
    assert controller.selected_distribution_scale_x.name == "Group width (mm)"
    assert controller.selected_distribution_scale_y.name == "Group height (mm)"
    assert controller.selected_distribution_scale_x.value == pytest.approx(40.0)
    assert controller.selected_distribution_scale_y.value == pytest.approx(24.0)

    payload = controller._build_export_config()
    distribution = payload["food_params"]["source_groups"]["group_001"]["distribution"]
    assert distribution["shape"] == "oval"
    assert distribution["scale"] == [pytest.approx(0.02), pytest.approx(0.012)]


def test_environment_builder_can_hide_source_group_shape_preview() -> None:
    controller = _EnvironmentBuilderController()
    controller.object_type.value = "Source group"
    controller._add_point_object(
        object_type="Source group",
        x=0.01,
        y=-0.01,
        radius=0.004,
        color="#3355aa",
    )

    assert controller.source_group_circle_source.data["id"] == ["group_001"]
    assert controller.source_group_member_source.data["parent_id"]

    controller.selected_distribution_show_shape.value = False
    controller._on_apply_selected_object(None)

    assert controller._objects[0].distribution_show_shape is False
    assert controller.source_group_circle_source.data["id"] == []
    assert controller.source_group_member_source.data["parent_id"]


def test_environment_builder_disables_workspace_presets_without_active_workspace() -> (
    None
):
    controller = _EnvironmentBuilderController()

    assert controller.save_preset_btn.disabled is True
    assert controller.load_preset_btn.disabled is True
    assert controller.refresh_presets_btn.disabled is False
    assert (
        "Workspace environments directory unavailable" in controller.preset_meta.object
    )


def test_environment_builder_applies_selected_food_edits_to_export(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller._add_point_object(
        object_type="Source unit",
        x=0.01,
        y=0.02,
        radius=0.008,
        color="#4caf50",
    )

    controller.selected_id.value = "food_custom"
    controller.selected_x.value = 0.03
    controller.selected_y.value = -0.01
    controller.selected_radius.value = 12.0
    controller.selected_color.value = "#123456"
    controller.selected_amount.value = 5.5
    controller.selected_odor_id.value = "banana"
    controller.selected_odor_intensity.value = 2.0
    controller.selected_odor_spread.value = 0.05
    controller.selected_substrate_type.value = "cornmeal"
    controller.selected_substrate_quality.value = 0.8

    controller._on_apply_selected_object(None)

    assert controller._objects[0].object_id == "food_custom"
    assert controller._objects[0].x == pytest.approx(0.03)
    assert controller._objects[0].y == pytest.approx(-0.01)
    assert controller._objects[0].radius == pytest.approx(0.012)
    payload = controller._build_export_config()
    food_entry = payload["food_params"]["source_units"]["food_custom"]
    assert food_entry["pos"] == [0.03, -0.01]
    assert food_entry["radius"] == pytest.approx(0.012)
    assert food_entry["amount"] == pytest.approx(5.5)
    assert food_entry["odor"] == {
        "id": "banana",
        "intensity": pytest.approx(2.0),
        "spread": pytest.approx(0.05),
    }
    assert food_entry["substrate"] == {
        "type": "cornmeal",
        "quality": pytest.approx(0.8),
    }
    assert controller.food_source.data["id"] == ["food_custom"]
    assert controller.odor_peak_source.data["id"] == ["food_custom"]
    assert 'Updated object "food_custom"' in controller.status.object


def test_environment_builder_source_visuals_follow_food_amount_and_odor_layers(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller._add_point_object(
        object_type="Source unit",
        x=0.01,
        y=0.02,
        radius=0.008,
        color="#4caf50",
    )

    initial_fill_alpha = controller.food_source.data["fill_alpha"][0]
    assert controller.odor_layer_source.data["id"] == []
    assert controller.odor_peak_source.data["id"] == []

    controller.selected_amount.value = 4.0
    controller.selected_odor_id.value = "odor_food"
    controller.selected_odor_intensity.value = 2.0
    controller.selected_odor_spread.value = 0.05
    controller._on_apply_selected_object(None)

    food_fill_alpha = controller.food_source.data["fill_alpha"][0]
    assert food_fill_alpha > initial_fill_alpha
    assert controller.odor_layer_source.data["id"]
    assert controller.odor_peak_source.data["id"] == ["food_001"]

    controller.selected_amount.value = 0.0
    controller.selected_odor_id.value = ""
    controller._on_apply_selected_object(None)

    assert controller.food_source.data["fill_alpha"][0] == pytest.approx(
        initial_fill_alpha
    )
    assert controller.food_source.data["line_width"][0] < 2.6
    assert controller.odor_layer_source.data["id"] == []


def test_environment_builder_odor_id_editor_suggests_existing_ids(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller._add_point_object(
        object_type="Source unit",
        x=0.01,
        y=0.02,
        radius=0.008,
        color="#4caf50",
    )

    controller.selected_odor_id.value = "banana"
    controller.selected_odor_intensity.value = 2.0
    controller.selected_odor_spread.value = 0.05
    controller._on_apply_selected_object(None)

    assert "banana" in controller.selected_odor_id.options
    assert controller.selected_odor_id.value == "banana"


def test_environment_builder_editor_family_visibility_tracks_object_type(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller._add_point_object(
        object_type="Source unit",
        x=0.01,
        y=0.02,
        radius=0.008,
        color="#4caf50",
    )

    assert controller._editor_food_family.visible is True
    assert controller._editor_substrate_family.visible is True
    assert controller._editor_odor_family.visible is True
    assert controller._editor_distribution_family.visible is False

    controller._add_point_object(
        object_type="Source group",
        x=-0.02,
        y=0.01,
        radius=0.004,
        color="#3355aa",
    )

    assert controller.selected_object.value == "group_002"
    assert controller._editor_food_family.visible is True
    assert controller._editor_substrate_family.visible is True
    assert controller._editor_odor_family.visible is True
    assert controller._editor_distribution_family.visible is True


def test_environment_builder_exports_source_group_and_food_grid(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller.object_type.value = "Source group"
    controller._add_point_object(
        object_type="Source group",
        x=0.01,
        y=-0.01,
        radius=0.004,
        color="#3355aa",
    )
    controller.food_grid_enabled.value = True
    controller.food_grid_color.value = "#66aa33"
    controller.food_grid_dims_x.value = 41
    controller.food_grid_dims_y.value = 39
    controller.food_grid_initial_value.value = 0.003
    controller.food_grid_substrate_type.value = "cornmeal"
    controller.food_grid_substrate_quality.value = 0.7

    payload = controller._build_export_config()

    assert "group_001" in payload["food_params"]["source_groups"]
    assert payload["food_params"]["food_grid"]["grid_dims"] == [41, 39]
    assert payload["food_params"]["food_grid"]["substrate"] == {
        "type": "cornmeal",
        "quality": pytest.approx(0.7),
    }


def test_environment_builder_rejects_source_group_insert_when_members_leave_arena() -> (
    None
):
    controller = _EnvironmentBuilderController()
    controller.object_type.value = "Source group"
    controller.group_mode.value = "periphery"
    controller.group_shape.value = "circle"
    controller.group_count.value = 6
    controller.group_spread_x.value = 40.0

    controller._add_point_object(
        object_type="Source group",
        x=0.08,
        y=0.0,
        radius=0.004,
        color="#3355aa",
    )

    assert controller._objects == []
    assert (
        controller.status.object
        == 'Source group "group_001" places member units outside the arena. Move it inward or reduce its footprint.'
    )


def test_environment_builder_blocks_save_when_source_group_members_leave_arena(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller.preset_name.value = "Invalid Group Arena"
    controller.object_type.value = "Source group"
    controller.group_mode.value = "periphery"
    controller.group_shape.value = "circle"
    controller.group_count.value = 6
    controller.group_spread_x.value = 20.0
    controller._add_point_object(
        object_type="Source group",
        x=0.0,
        y=0.0,
        radius=0.004,
        color="#3355aa",
    )

    controller.arena_width.value = 0.04
    controller.arena_height.value = 0.04

    controller._on_save_preset(None)

    preset_path = workspace_root / "environments" / "Invalid_Group_Arena.json"
    assert preset_path.exists()
    assert "Invalid_Group_Arena" in reg.conf.Env.dict
    assert (
        controller.status.object
        == 'Saved environment preset "Invalid_Group_Arena" to the workspace and registered it in Env.txt.'
    )


def test_environment_builder_blocks_arena_resize_that_excludes_source_unit() -> None:
    controller = _EnvironmentBuilderController()
    controller._add_point_object(
        object_type="Source unit",
        x=0.085,
        y=0.0,
        radius=0.004,
        color="#4caf50",
    )

    controller.arena_width.value = 0.16

    assert controller.arena_width.value == pytest.approx(0.2)
    assert controller.arena_height.value == pytest.approx(0.2)
    assert (
        controller.status.object
        == 'Object "food_001" has primary coordinates outside the arena. Arena size change was cancelled.'
    )


def test_environment_builder_blocks_arena_resize_that_excludes_source_group_members() -> (
    None
):
    controller = _EnvironmentBuilderController()
    controller.object_type.value = "Source group"
    controller.group_mode.value = "periphery"
    controller.group_shape.value = "circle"
    controller.group_count.value = 6
    controller.group_spread_x.value = 20.0
    controller._add_point_object(
        object_type="Source group",
        x=0.0,
        y=0.0,
        radius=0.004,
        color="#3355aa",
    )

    controller.arena_width.value = 0.04
    controller.arena_height.value = 0.04

    assert controller.arena_width.value == pytest.approx(0.2)
    assert controller.arena_height.value == pytest.approx(0.2)
    assert (
        controller.status.object
        == 'Source group "group_001" places member units outside the arena. Move it inward or reduce its footprint. Arena size change was cancelled.'
    )


def test_environment_builder_locks_arena_controls_when_objects_exist() -> None:
    controller = _EnvironmentBuilderController()

    assert controller.arena_shape.disabled is False
    assert controller.arena_width.disabled is False
    assert controller.arena_height.disabled is False
    assert controller.arena_torus.disabled is False

    controller._add_point_object(
        object_type="Source unit",
        x=0.01,
        y=0.02,
        radius=0.008,
        color="#4caf50",
    )

    assert controller.arena_shape.disabled is True
    assert controller.arena_width.disabled is True
    assert controller.arena_height.disabled is True
    assert controller.arena_torus.disabled is True

    controller._on_clear_arena(None)

    assert controller.arena_shape.disabled is False
    assert controller.arena_width.disabled is False
    assert controller.arena_height.disabled is False
    assert controller.arena_torus.disabled is False


def test_environment_builder_exports_odorscape_windscape_and_thermoscape(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller.odorscape_enabled.value = True
    controller.odorscape_mode.value = "Diffusion"
    controller.odorscape_color.value = "#88aa44"
    controller.odorscape_grid_dims_x.value = 61
    controller.odorscape_grid_dims_y.value = 47
    controller.odorscape_initial_value.value = 0.002
    controller.odorscape_fixed_max.value = True
    controller.odorscape_evap_const.value = 0.93
    controller.odorscape_sigma_x.value = 1.2
    controller.odorscape_sigma_y.value = 0.8

    controller.windscape_enabled.value = True
    controller.windscape_color.value = "#ff5500"
    controller.windscape_direction.value = 90.0
    controller.windscape_speed.value = 12.5
    controller.wind_puffs_table.value = pd.DataFrame(
        [
            {
                "id": "puff_a",
                "duration": 2.0,
                "speed": 18.0,
                "direction": 0.5,
                "start_time": 7.0,
                "N": 3,
                "interval": 11.0,
            }
        ]
    )

    controller.thermoscape_enabled.value = True
    controller.thermoscape_plate_temp.value = 24.5
    controller.thermoscape_spread.value = 0.14
    controller.thermo_sources_table.value = pd.DataFrame(
        [
            {"id": "hotspot", "x": 0.08, "y": -0.04, "dTemp": 6.5},
            {"id": "coldspot", "x": -0.06, "y": 0.03, "dTemp": -4.0},
        ]
    )

    payload = controller._build_export_config()

    assert payload["odorscape"] == {
        "unique_id": "DiffusionValueLayer",
        "odorscape": "Diffusion",
        "color": "#88aa44",
        "grid_dims": [61, 47],
        "initial_value": pytest.approx(0.002),
        "fixed_max": True,
        "evap_const": pytest.approx(0.93),
        "gaussian_sigma": (pytest.approx(1.2), pytest.approx(0.8)),
    }
    assert payload["windscape"]["wind_speed"] == pytest.approx(12.5)
    assert payload["windscape"]["wind_direction"] == pytest.approx(math.pi / 2.0)
    assert payload["windscape"]["puffs"]["puff_a"] == {
        "duration": pytest.approx(2.0),
        "speed": pytest.approx(18.0),
        "direction": pytest.approx(0.5),
        "start_time": pytest.approx(7.0),
        "N": 3,
        "interval": pytest.approx(11.0),
    }
    assert payload["thermoscape"]["plate_temp"] == pytest.approx(24.5)
    assert payload["thermoscape"]["spread"] == pytest.approx(0.14)
    assert payload["thermoscape"]["thermo_sources"] == {
        "hotspot": (pytest.approx(0.08), pytest.approx(-0.04)),
        "coldspot": (pytest.approx(-0.06), pytest.approx(0.03)),
    }
    assert payload["thermoscape"]["thermo_source_dTemps"] == {
        "hotspot": pytest.approx(6.5),
        "coldspot": pytest.approx(-4.0),
    }


def test_environment_builder_loads_scape_controls_from_config(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller._apply_config(
        {
            "arena": {"geometry": "rectangular", "dims": [0.2, 0.2], "torus": False},
            "food_params": {"source_units": {}, "source_groups": {}, "food_grid": None},
            "border_list": {},
            "odorscape": {
                "unique_id": "DiffusionValueLayer",
                "odorscape": "Diffusion",
                "color": "#88aa44",
                "grid_dims": [61, 47],
                "initial_value": 0.002,
                "fixed_max": True,
                "evap_const": 0.93,
                "gaussian_sigma": [1.2, 0.8],
            },
            "windscape": {
                "unique_id": "WindScape",
                "color": "#ff5500",
                "wind_direction": math.pi / 2.0,
                "wind_speed": 12.5,
                "puffs": {
                    "puff_a": {
                        "duration": 2.0,
                        "speed": 18.0,
                        "direction": 0.5,
                        "start_time": 7.0,
                        "N": 3,
                        "interval": 11.0,
                    }
                },
            },
            "thermoscape": {
                "unique_id": "ThermoScape",
                "plate_temp": 24.5,
                "spread": 0.14,
                "thermo_sources": {"hotspot": [0.08, -0.04]},
                "thermo_source_dTemps": {"hotspot": 6.5},
            },
        }
    )

    assert controller.odorscape_enabled.value is True
    assert controller.odorscape_mode.value == "Diffusion"
    assert controller.odorscape_evap_const.visible is True
    assert controller.odorscape_sigma_x.visible is True
    assert controller.windscape_direction.value == pytest.approx(90.0)
    assert controller.odorscape_grid_dims_x.value == 61
    assert controller.odorscape_grid_dims_y.value == 47
    assert controller.windscape_enabled.value is True
    assert controller.wind_puffs_table.value["id"].tolist() == ["puff_a"]
    assert controller.wind_puffs_table.value["N"].tolist() == [3]
    assert controller.thermoscape_enabled.value is True
    assert controller.thermo_sources_table.value["id"].tolist() == ["hotspot"]
    assert controller.thermo_sources_table.value["dTemp"].tolist() == [
        pytest.approx(6.5)
    ]


def test_environment_builder_scape_preview_layers_render_when_enabled(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller._add_point_object(
        object_type="Source unit",
        x=0.01,
        y=0.02,
        radius=0.008,
        color="#4caf50",
    )
    controller.selected_odor_id.value = "odor_preview"
    controller.selected_odor_intensity.value = 2.0
    controller.selected_odor_spread.value = 0.04
    controller._on_apply_selected_object(None)

    controller.odorscape_enabled.value = True
    controller.windscape_enabled.value = True
    controller.windscape_speed.value = 12.0
    controller.thermoscape_enabled.value = True
    controller.thermo_sources_table.value = pd.DataFrame(
        [{"id": "hotspot", "x": 0.08, "y": -0.04, "dTemp": 6.5}]
    )

    controller._sync_scape_preview()

    assert controller.odorscape_contour_source.data["id"]
    assert controller.windscape_segment_source.data["x0"]
    assert controller.windscape_head_source.data["x"]
    assert controller.thermoscape_aura_source.data["id"] == ["hotspot"] * 3
    assert controller.thermoscape_marker_source.data["id"] == ["hotspot"]


def test_environment_builder_scape_preview_layers_clear_when_disabled(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller.odorscape_enabled.value = False
    controller.windscape_enabled.value = False
    controller.thermoscape_enabled.value = False
    controller._sync_scape_preview()

    assert controller.odorscape_contour_source.data["id"] == []
    assert controller.windscape_segment_source.data["x0"] == []
    assert controller.windscape_head_source.data["x"] == []
    assert controller.thermoscape_aura_source.data["id"] == []
    assert controller.thermoscape_marker_source.data["id"] == []


def test_environment_builder_attaches_hover_help_to_native_widgets() -> None:
    controller = _EnvironmentBuilderController()

    assert controller.arena_shape.description
    assert "arena" in controller.arena_shape.description.lower()
    assert "shape" in controller.arena_shape.description.lower()
    assert controller.windscape_direction.description
    assert "degrees" in controller.windscape_direction.description.lower()
    assert controller.preset_name.description
    assert "saved workspace json file" in controller.preset_name.description.lower()


def test_environment_builder_deletes_selected_object(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller._add_point_object(
        object_type="Source unit",
        x=0.01,
        y=0.02,
        radius=0.008,
        color="#4caf50",
    )

    controller._on_delete_selected_object(None)

    assert controller._objects == []
    assert controller.food_source.data["id"] == []
    assert controller.selected_object.disabled is True
    assert 'Deleted object "food_001"' in controller.status.object


def test_environment_builder_canvas_select_mode_syncs_inspector(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller._add_point_object(
        object_type="Source unit",
        x=0.01,
        y=0.02,
        radius=0.008,
        color="#4caf50",
    )
    controller.select_mode.value = True
    controller._on_select_mode_change()

    controller._on_tap(SimpleNamespace(x=0.01, y=0.02))

    assert controller.selected_object.value == "food_001"
    assert controller.selected_id.value == "food_001"
    assert controller.table.selection == [0]
    assert controller.food_highlight_source.data["x"] == [0.01]
    assert 'Selected "food_001" from canvas.' in controller.status.object


def test_environment_builder_table_selection_syncs_editor_and_highlight(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _EnvironmentBuilderController()
    controller._add_point_object(
        object_type="Source unit",
        x=0.01,
        y=0.02,
        radius=0.008,
        color="#4caf50",
    )
    controller._add_border_object(
        x0=-0.03,
        y0=-0.02,
        x1=0.04,
        y1=0.05,
        width=0.0015,
        color="#111111",
    )

    controller.table.selection = [1]
    controller._on_table_selection_change()

    assert controller.selected_object.value == "border_002"
    assert controller.selected_id.value == "border_002"
    assert controller.border_highlight_source.data["x0"] == [-0.03]
    assert controller.border_highlight_source.data["x1"] == [0.04]
