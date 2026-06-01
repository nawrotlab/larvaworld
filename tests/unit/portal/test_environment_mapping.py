from __future__ import annotations

import pytest

from larvaworld.lib import util
from larvaworld.portal.canvas_widgets.environment_mapping import (
    env_params_to_canvas_state,
)


def test_env_params_to_canvas_state_maps_static_environment_layers() -> None:
    env_params = util.AttrDict(
        {
            "arena": {"geometry": "circular", "dims": [0.24, 0.24], "torus": True},
            "food_params": {
                "source_units": {
                    "patch": {
                        "pos": [0.02, -0.01],
                        "radius": 0.005,
                        "amount": 2.0,
                        "color": "#44aa55",
                        "odor": {
                            "id": "apple",
                            "intensity": 1.0,
                            "spread": 0.03,
                        },
                    }
                },
                "source_groups": {
                    "cluster": {
                        "radius": 0.003,
                        "amount": 1.0,
                        "color": "#6688aa",
                        "distribution": {
                            "N": 4,
                            "loc": [0.0, 0.01],
                            "mode": "uniform",
                            "shape": "rect",
                            "scale": [0.02, 0.01],
                        },
                        "odor": {
                            "id": "yeast",
                            "intensity": 0.8,
                            "spread": 0.02,
                        },
                    }
                },
                "food_grid": {"color": "#77aa44", "grid_dims": [3, 2]},
            },
            "border_list": {
                "wall": {
                    "vertices": [[-0.02, 0.0], [0.02, 0.0]],
                    "width": 0.002,
                    "color": "#333333",
                }
            },
            "odorscape": {"odorscape": "Gaussian", "color": "#99aa33"},
            "windscape": {"wind_speed": 10.0, "wind_direction": 1.0},
            "thermoscape": {
                "spread": 0.04,
                "thermo_sources": {"hot": (0.01, 0.02)},
                "thermo_source_dTemps": {"hot": 3.0},
            },
        }
    )
    larva_groups = {
        "explorer": util.AttrDict(
            {
                "distribution": {
                    "N": 6,
                    "loc": [0.0, -0.02],
                    "shape": "circle",
                    "scale": [0.01, 0.01],
                },
                "color": "#2f4858",
            }
        )
    }

    state = env_params_to_canvas_state(env_params, larva_groups=larva_groups)

    assert state.arena.geometry == "circular"
    assert state.arena.dims == (0.24, 0.24)
    assert state.arena.torus is True
    assert state.food_grid == {"color": "#77aa44", "grid_dims": [3, 2]}
    assert state.odorscape == {"odorscape": "Gaussian", "color": "#99aa33"}
    assert state.windscape["wind_speed"] == pytest.approx(10.0)
    assert state.thermoscape["thermo_sources"]["hot"] == (0.01, 0.02)

    objects = {obj.object_id: obj for obj in state.objects}
    assert objects["patch"].object_type == "source_unit"
    assert objects["patch"].x == pytest.approx(0.02)
    assert objects["patch"].odor_id == "apple"
    assert objects["cluster"].object_type == "source_group"
    assert objects["cluster"].distribution_n == 4
    assert objects["cluster"].distribution_shape == "rect"
    assert objects["wall"].object_type == "border_segment"
    assert objects["wall"].x2 == pytest.approx(0.02)
    assert objects["explorer"].object_type == "larva_group"
    assert objects["explorer"].distribution_n == 6


def test_env_params_to_canvas_state_can_hide_group_shapes() -> None:
    env_params = {
        "arena": {"geometry": "rectangular", "dims": [0.2, 0.1]},
        "food_params": {
            "source_units": {},
            "source_groups": {
                "cluster": {
                    "distribution": {
                        "N": 4,
                        "loc": [0.0, 0.01],
                        "mode": "uniform",
                        "shape": "circle",
                        "scale": [0.02, 0.01],
                    }
                }
            },
        },
    }
    larva_groups = {
        "explorer": util.AttrDict(
            {
                "distribution": {
                    "N": 6,
                    "loc": [0.0, -0.02],
                    "shape": "circle",
                    "scale": [0.01, 0.01],
                },
            }
        )
    }

    state = env_params_to_canvas_state(
        env_params,
        larva_groups=larva_groups,
        show_group_shapes=False,
    )
    objects = {obj.object_id: obj for obj in state.objects}

    assert objects["cluster"].distribution_show_shape is False
    assert objects["explorer"].distribution_show_shape is False


def test_env_params_to_canvas_state_skips_malformed_optional_sections() -> None:
    env_params = {
        "arena": {"geometry": "rectangular", "dims": [0.2, 0.1]},
        "food_params": {
            "source_units": {
                "patch": {"pos": [0.01, 0.02], "radius": 0.004},
                "bad_patch": {"pos": ["bad"]},
            },
            "source_groups": {"bad_group": {"distribution": {"loc": ["bad"]}}},
            "food_grid": "not-a-grid",
        },
        "border_list": {"bad_wall": {"vertices": [[0.0, 0.0]]}},
        "odorscape": "not-a-scape",
    }

    state = env_params_to_canvas_state(env_params)

    assert state.food_grid is None
    assert state.odorscape is None
    assert [obj.object_id for obj in state.objects] == ["patch"]


def test_env_params_to_canvas_state_maps_all_flattened_border_pairs() -> None:
    env_params = {
        "arena": {"geometry": "rectangular", "dims": [0.2, 0.1]},
        "food_params": {"source_units": {}, "source_groups": {}, "food_grid": {}},
        "border_list": {
            "obstacle": {
                "vertices": [
                    [0.0, 0.0],
                    [0.01, 0.0],
                    [0.01, 0.0],
                    [0.01, 0.01],
                    [0.01, 0.01],
                    [0.0, 0.0],
                ],
                "width": 0.002,
            }
        },
    }

    state = env_params_to_canvas_state(env_params)
    borders = [obj for obj in state.objects if obj.object_type == "border_segment"]

    assert [obj.object_id for obj in borders] == [
        "obstacle:0",
        "obstacle:1",
        "obstacle:2",
    ]
    assert borders[0].x == pytest.approx(0.0)
    assert borders[1].y2 == pytest.approx(0.01)
