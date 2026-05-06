from __future__ import annotations

from numbers import Number
from pathlib import Path

import pytest

from larvaworld.lib import reg, util
from larvaworld.lib.reg.generators import EnvConf
from larvaworld.portal.canvas_widgets.environment_mapping import (
    env_params_to_canvas_state,
)
from larvaworld.portal.simulation.parameter_resolution import (
    resolve_base_experiment_parameters,
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


def _is_xy_pair(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(coord, Number) for coord in value)
    )


def _all_xy_pairs(points: object) -> bool:
    return (
        isinstance(points, (list, tuple))
        and len(points) > 0
        and all(_is_xy_pair(point) for point in points)
    )


def test_env_params_roundtrip_template_default_preserves_invariants(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    parameters = resolve_base_experiment_parameters("dish")
    typed_env = EnvConf(**parameters.env_params)
    roundtripped = typed_env.nestedConf

    assert roundtripped.arena.geometry == parameters.env_params.arena.geometry
    assert _is_xy_pair(roundtripped.arena.dims)
    assert isinstance(roundtripped.food_params, dict)
    assert isinstance(roundtripped.border_list, dict)

    state = env_params_to_canvas_state(
        roundtripped,
        larva_groups=parameters.larva_groups,
    )
    assert state.arena.geometry == roundtripped.arena.geometry
    assert _is_xy_pair(state.arena.dims)
    assert isinstance(state.objects, tuple)


def test_env_params_roundtrip_workspace_override_preserves_invariants(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    workspace_payload = util.AttrDict(
        {
            "arena": {"geometry": "rectangular", "dims": [0.22, 0.14]},
            "food_params": {
                "source_units": {
                    "patch": {
                        "pos": [0.02, -0.01],
                        "radius": 0.005,
                        "amount": 2.0,
                        "color": "#44aa55",
                        "odor": {"id": "apple", "intensity": 1.0, "spread": 0.02},
                    }
                },
                "source_groups": {
                    "cluster": {
                        "distribution": {
                            "N": 3,
                            "loc": [0.0, 0.02],
                            "mode": "uniform",
                            "shape": "circle",
                            "scale": [0.01, 0.01],
                        }
                    }
                },
                "food_grid": {},
            },
            "border_list": {
                "wall": {
                    "vertices": [[-0.03, 0.0], [0.03, 0.0]],
                    "width": 0.0015,
                    "color": "#333333",
                }
            },
        }
    )

    parameters = resolve_base_experiment_parameters("dish", workspace_payload)
    typed_env = EnvConf(**parameters.env_params)
    roundtripped = typed_env.nestedConf

    assert roundtripped.arena.geometry == "rectangular"
    assert _is_xy_pair(roundtripped.arena.dims)
    assert "patch" in roundtripped.food_params.source_units
    assert _is_xy_pair(roundtripped.food_params.source_units["patch"].pos)
    assert "cluster" in roundtripped.food_params.source_groups
    assert "wall" in roundtripped.border_list
    assert _all_xy_pairs(roundtripped.border_list["wall"].vertices)

    state = env_params_to_canvas_state(
        roundtripped,
        larva_groups=parameters.larva_groups,
    )
    object_ids = {obj.object_id for obj in state.objects}
    assert "patch" in object_ids
    assert any(obj_id.startswith("wall") for obj_id in object_ids)


def test_env_params_roundtrip_registry_override_preserves_invariants(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    registry_payload = util.AttrDict(reg.conf.Env.getID("maze")).get_copy()
    parameters = resolve_base_experiment_parameters("dish", registry_payload)
    typed_env = EnvConf(**parameters.env_params)
    roundtripped = typed_env.nestedConf

    assert roundtripped.arena.geometry == "rectangular"
    assert "Maze" in roundtripped.border_list
    assert _all_xy_pairs(roundtripped.border_list["Maze"].vertices)

    state = env_params_to_canvas_state(
        roundtripped,
        larva_groups=parameters.larva_groups,
    )
    border_objects = [
        obj for obj in state.objects if obj.object_type == "border_segment"
    ]
    assert len(border_objects) > 0
    assert any(obj.object_id.startswith("Maze") for obj in border_objects)
