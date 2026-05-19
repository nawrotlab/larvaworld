from __future__ import annotations

import numpy as np
import pytest

from larvaworld.portal.canvas_widgets.environment_canvas import (
    DEFAULT_LARVA_COLOR,
    EnvironmentCanvas,
    STATIC_LARVA_GROUP_MEMBER_HALF_LENGTH,
)
from larvaworld.portal.canvas_widgets.environment_models import (
    CanvasArena,
    CanvasRingOverlay,
    CanvasObject,
    EnvironmentCanvasState,
    LarvaPreviewFrame,
)


def _state() -> EnvironmentCanvasState:
    return EnvironmentCanvasState(
        arena=CanvasArena("rectangular", (0.2, 0.1)),
        food_grid={"color": "#44aa55", "grid_dims": (4, 2)},
        odorscape={"odorscape": "Gaussian", "color": "#99aa33"},
        windscape={"wind_speed": 20.0, "wind_direction": 0.0, "color": "#ff0000"},
        thermoscape={
            "spread": 0.05,
            "thermo_sources": {"hot": (0.01, 0.02), "cold": (-0.02, -0.01)},
            "thermo_source_dTemps": {"hot": 5.0, "cold": -3.0},
        },
        objects=(
            CanvasObject(
                object_id="patch",
                object_type="source_unit",
                x=0.01,
                y=0.02,
                radius=0.004,
                color="#44aa55",
                amount=1.0,
                odor_id="apple",
                odor_intensity=1.2,
                odor_spread=0.015,
            ),
            CanvasObject(
                object_id="group",
                object_type="source_group",
                x=-0.02,
                y=0.01,
                radius=0.003,
                color="#6688aa",
                amount=1.0,
                odor_id="yeast",
                odor_intensity=1.0,
                odor_spread=0.01,
                distribution_mode="uniform",
                distribution_shape="oval",
                distribution_n=5,
                distribution_scale_x=0.02,
                distribution_scale_y=0.01,
            ),
            CanvasObject(
                object_id="wall",
                object_type="border_segment",
                x=-0.05,
                y=-0.02,
                x2=0.05,
                y2=-0.02,
                width=0.002,
                color="#333333",
            ),
            CanvasObject(
                object_id="larvae",
                object_type="larva_group",
                x=0.0,
                y=-0.01,
                color="#2f4858",
                distribution_shape="circle",
                distribution_scale_x=0.015,
                distribution_scale_y=0.015,
                distribution_n=8,
            ),
        ),
    )


def _legend_item(canvas: EnvironmentCanvas, label: str):
    return next(
        item
        for item in canvas.fig.legend[0].items
        if getattr(item.label, "value", None) == label
    )


def test_environment_canvas_view_is_stable_across_set_state() -> None:
    canvas = EnvironmentCanvas()
    view = canvas.view()

    canvas.set_state(_state())
    canvas.set_state(_state())

    assert canvas.view() is view


def test_environment_canvas_draws_rectangular_and_circular_arena() -> None:
    canvas = EnvironmentCanvas()
    canvas.set_state(
        EnvironmentCanvasState(arena=CanvasArena("rectangular", (0.2, 0.1)))
    )

    assert canvas.arena_source.data["w"] == [0.2]
    assert canvas._arena_rect_renderer.visible is True
    assert canvas._arena_circle_renderer.visible is False

    canvas.set_state(EnvironmentCanvasState(arena=CanvasArena("circular", (0.2, 0.2))))

    assert canvas._arena_rect_renderer.visible is False
    assert canvas._arena_circle_renderer.visible is True


def test_environment_canvas_can_hide_arena_outline_without_changing_viewport() -> None:
    canvas = EnvironmentCanvas(width=760, height=620)
    canvas.set_state(
        EnvironmentCanvasState(
            arena=CanvasArena("rectangular", (0.2, 0.1)),
            show_arena_outline=False,
        )
    )

    assert canvas.arena_source.data["w"] == [0.2]
    assert canvas.arena_source.data["h"] == [0.1]
    assert canvas._arena_rect_renderer.visible is False
    assert canvas._arena_circle_renderer.visible is False
    x_span = canvas.fig.x_range.end - canvas.fig.x_range.start
    y_span = canvas.fig.y_range.end - canvas.fig.y_range.start
    assert x_span > 0
    assert y_span > 0
    assert x_span / y_span == pytest.approx(760 / 620)


def test_environment_canvas_arena_outline_visibility_roundtrips() -> None:
    canvas = EnvironmentCanvas()
    canvas.set_state(
        EnvironmentCanvasState(
            arena=CanvasArena("circular", (0.2, 0.2)),
            show_arena_outline=False,
        )
    )
    assert canvas._arena_circle_renderer.visible is False

    canvas.set_state(
        EnvironmentCanvasState(
            arena=CanvasArena("circular", (0.2, 0.2)),
            show_arena_outline=True,
        )
    )
    assert canvas._arena_circle_renderer.visible is True
    assert canvas._arena_rect_renderer.visible is False


def test_environment_canvas_ranges_preserve_one_to_one_axis_ratio() -> None:
    canvas = EnvironmentCanvas(width=760, height=620)
    canvas.set_state(
        EnvironmentCanvasState(arena=CanvasArena("rectangular", (0.1, 0.1)))
    )

    x_span = canvas.fig.x_range.end - canvas.fig.x_range.start
    y_span = canvas.fig.y_range.end - canvas.fig.y_range.start

    assert x_span / y_span == pytest.approx(760 / 620)


def test_environment_canvas_draws_static_environment_sources() -> None:
    canvas = EnvironmentCanvas()
    canvas.set_state(_state())

    assert len(canvas.food_grid_cell_source.data["x"]) == 8
    assert canvas.food_source.data["id"] == ["patch"]
    assert len(canvas.odor_layer_source.data["x"]) > 0
    assert canvas.odor_peak_source.data["id"]
    assert canvas.source_group_ellipse_source.data["id"] == ["group"]
    assert len(canvas.source_group_member_source.data["x"]) == 5
    assert canvas.border_source.data["id"] == ["wall"]
    assert canvas.larva_group_circle_source.data["id"] == ["larvae"]
    assert len(canvas.larva_group_member_source.data["x0"]) == 8


def test_environment_canvas_source_group_circle_matches_builder_semantics() -> None:
    canvas = EnvironmentCanvas()
    state = EnvironmentCanvasState(
        arena=CanvasArena("rectangular", (0.1, 0.06)),
        objects=(
            CanvasObject(
                object_id="source-group",
                object_type="source_group",
                x=0.0,
                y=0.0,
                color="#4caf50",
                distribution_mode="uniform",
                distribution_shape="circle",
                distribution_n=4,
                distribution_scale_x=0.005,
                distribution_scale_y=0.02,
            ),
        ),
    )

    canvas.set_state(state)

    assert canvas.source_group_circle_source.data["id"] == ["source-group"]
    assert canvas.source_group_circle_source.data["r"] == [0.02]
    assert canvas.source_group_ellipse_source.data["id"] == []


def test_environment_canvas_legend_order_matches_builder_with_larva_extension() -> None:
    canvas = EnvironmentCanvas()
    legend_labels = [
        item.label.value
        for item in canvas.fig.legend[0].items
        if isinstance(getattr(item.label, "value", None), str)
    ]

    assert "Source units" in legend_labels
    assert "Source groups" in legend_labels
    assert "Borders" in legend_labels
    assert "Larva groups" in legend_labels
    assert "Simulated larvae" in legend_labels
    assert "Larva trails" in legend_labels
    assert "Odor aura" in legend_labels
    assert "Odorscape" in legend_labels
    assert "Windscape" in legend_labels
    assert "Thermoscape" in legend_labels


def test_environment_canvas_source_groups_legend_targets_group_renderers() -> None:
    canvas = EnvironmentCanvas()
    source_group_item = _legend_item(canvas, "Source groups")
    data_sources = {renderer.data_source for renderer in source_group_item.renderers}

    assert canvas.source_group_circle_source in data_sources
    assert canvas.source_group_ellipse_source in data_sources
    assert canvas.source_group_rect_source in data_sources
    assert canvas.source_group_member_source in data_sources
    assert canvas.food_highlight_source not in data_sources


def test_environment_canvas_set_larva_frame_populates_dynamic_sources() -> None:
    canvas = EnvironmentCanvas()
    canvas.set_larva_frame(
        LarvaPreviewFrame(
            tick=5,
            centroids=((0.0, 0.0), (0.01, 0.02)),
            heads=((0.0016, 0.0),),
            midlines=(((0.0, 0.0), (0.001, 0.0), (0.002, 0.0)),),
            trails=(((0.0, 0.0), (0.0, 0.003)), ((0.01, 0.02), (0.011, 0.021))),
            colors=("#111111",),
        )
    )

    assert canvas.sim_larva_centroid_source.data["x"] == [0.0, 0.01]
    assert canvas.sim_larva_centroid_source.data["color"] == [
        "#111111",
        DEFAULT_LARVA_COLOR,
    ]
    assert canvas.sim_larva_head_source.data["x"] == [0.0016]
    assert canvas.sim_larva_midline_source.data["xs"] == [[0.0, 0.001, 0.002]]
    assert canvas.sim_larva_trail_source.data["ys"] == [[0.0, 0.003], [0.02, 0.021]]


def test_environment_canvas_set_larva_frame_populates_labels() -> None:
    canvas = EnvironmentCanvas()
    canvas.set_larva_frame(
        LarvaPreviewFrame(
            tick=1,
            centroids=((0.0, 0.0), (0.01, 0.02)),
            colors=("#111111", "#222222"),
            labels=("A0", "A1"),
        )
    )

    assert canvas.sim_larva_label_source.data["label"] == ["A0", "A1"]
    assert canvas.sim_larva_label_source.data["x"] == [0.0, 0.01]


def test_environment_canvas_dynamic_ring_overlays_roundtrip() -> None:
    canvas = EnvironmentCanvas()
    canvas.set_dynamic_overlays(
        rings=(
            CanvasRingOverlay(x=0.0, y=0.0, radius=0.02, color="#ff0000"),
            CanvasRingOverlay(x=0.01, y=0.02, radius=0.01, color="#00ff00"),
        )
    )
    assert canvas.dynamic_ring_source.data["r"] == [0.02, 0.01]
    assert canvas.dynamic_ring_source.data["color"] == ["#ff0000", "#00ff00"]

    canvas.clear_dynamic_overlays()
    assert canvas.dynamic_ring_source.data["r"] == []


def test_environment_canvas_head_snapping_can_be_disabled() -> None:
    canvas = EnvironmentCanvas(snap_heads_to_midline=False)
    canvas.set_larva_frame(
        LarvaPreviewFrame(
            tick=2,
            centroids=((0.0, 0.0),),
            heads=((0.0016, 0.0),),
            midlines=(((0.0, 0.0), (0.001, 0.0), (0.002, 0.0)),),
            colors=("#111111",),
        )
    )

    assert canvas.sim_larva_head_source.data["x"] == [0.0016]
    assert canvas.sim_larva_midline_source.data["xs"] == [[0.0, 0.001, 0.002]]


def test_environment_canvas_head_snapping_can_be_enabled() -> None:
    canvas = EnvironmentCanvas(snap_heads_to_midline=True)
    canvas.set_larva_frame(
        LarvaPreviewFrame(
            tick=2,
            centroids=((0.0, 0.0),),
            heads=((0.0016, 0.0),),
            midlines=(((0.0, 0.0), (0.001, 0.0), (0.002, 0.0)),),
            colors=("#111111",),
        )
    )

    assert canvas.sim_larva_head_source.data["x"] == [0.002]
    assert canvas.sim_larva_midline_source.data["xs"] == [[0.0, 0.001, 0.002]]


def test_environment_canvas_set_larva_frame_skips_invalid_optional_data() -> None:
    canvas = EnvironmentCanvas()
    canvas.set_larva_frame(
        LarvaPreviewFrame(
            tick=0,
            centroids=((0.0, 0.0), ("bad", 1.0), (0.02, 0.03)),
            heads=((0.001, None), (0.002, 0.003), ("bad", "bad")),
            midlines=(
                (),
                ((0.0, 0.0),),
                ((0.02, 0.03), (0.03, 0.03), ("x", 1.0)),
            ),
            trails=(
                ((0.0, 0.0),),
                ("bad",),
                ((0.02, 0.03), (0.02, 0.04)),
            ),
            colors=("", "#ff0000"),
        )
    )

    assert canvas.sim_larva_centroid_source.data["x"] == [0.0, 0.02]
    assert canvas.sim_larva_centroid_source.data["color"] == [
        DEFAULT_LARVA_COLOR,
        DEFAULT_LARVA_COLOR,
    ]
    assert canvas.sim_larva_head_source.data["x"] == []
    assert canvas.sim_larva_midline_source.data["xs"] == [[0.02, 0.03]]
    assert canvas.sim_larva_trail_source.data["xs"] == [[0.02, 0.02]]


def test_environment_canvas_clear_larva_frame_only_clears_dynamic_sources() -> None:
    canvas = EnvironmentCanvas()
    canvas.set_state(_state())
    canvas.set_larva_frame(
        LarvaPreviewFrame(
            tick=1,
            centroids=((0.0, 0.0),),
            trails=(((0.0, 0.0), (0.0, 0.01)),),
        )
    )

    assert canvas.larva_group_circle_source.data["id"] == ["larvae"]
    assert canvas.sim_larva_centroid_source.data["x"] == [0.0]

    canvas.clear_larva_frame()

    assert canvas.sim_larva_centroid_source.data["x"] == []
    assert canvas.sim_larva_head_source.data["x"] == []
    assert canvas.sim_larva_midline_source.data["xs"] == []
    assert canvas.sim_larva_trail_source.data["xs"] == []
    assert canvas.larva_group_circle_source.data["id"] == ["larvae"]


def test_environment_canvas_set_state_clears_stale_simulated_larvae() -> None:
    canvas = EnvironmentCanvas()
    canvas.set_larva_frame(
        LarvaPreviewFrame(
            tick=1,
            centroids=((0.0, 0.0),),
            heads=((0.0, 0.001),),
            trails=(((0.0, 0.0), (0.0, 0.01)),),
        )
    )

    assert canvas.sim_larva_centroid_source.data["x"] == [0.0]
    canvas.set_state(_state())
    assert canvas.sim_larva_centroid_source.data["x"] == []
    assert canvas.sim_larva_head_source.data["x"] == []
    assert canvas.sim_larva_trail_source.data["xs"] == []


def test_environment_canvas_simulated_larvae_legend_membership() -> None:
    canvas = EnvironmentCanvas()
    simulated_item = _legend_item(canvas, "Simulated larvae")
    trail_item = _legend_item(canvas, "Larva trails")
    larva_groups_item = _legend_item(canvas, "Larva groups")

    assert simulated_item.renderers == [
        canvas._sim_larva_centroid_renderer,
        canvas._sim_larva_head_renderer,
        canvas._sim_larva_midline_renderer,
    ]
    assert trail_item.renderers == [canvas._sim_larva_trail_renderer]
    assert canvas._sim_larva_centroid_renderer not in larva_groups_item.renderers
    assert canvas._sim_larva_head_renderer not in larva_groups_item.renderers
    assert canvas._sim_larva_midline_renderer not in larva_groups_item.renderers
    assert canvas._sim_larva_trail_renderer not in larva_groups_item.renderers


def test_environment_canvas_view_is_stable_across_larva_frame_updates() -> None:
    canvas = EnvironmentCanvas()
    view = canvas.view()

    canvas.set_larva_frame(
        LarvaPreviewFrame(tick=0, centroids=((0.0, 0.0),), colors=("#111111",))
    )
    canvas.set_larva_frame(
        LarvaPreviewFrame(
            tick=1,
            centroids=((0.01, 0.01),),
            heads=((0.011, 0.012),),
            trails=(((0.0, 0.0), (0.01, 0.01)),),
        )
    )

    assert canvas.view() is view


def test_environment_canvas_source_group_members_are_deterministic() -> None:
    canvas = EnvironmentCanvas()
    state = _state()
    before = np.random.get_state()

    canvas.set_state(state)
    first = dict(canvas.source_group_member_source.data)
    canvas.set_state(state)
    second = dict(canvas.source_group_member_source.data)
    after = np.random.get_state()

    assert first == second
    assert all(np.array_equal(a, b) for a, b in zip(before[1:], after[1:]))


def test_environment_canvas_draws_scape_layers() -> None:
    canvas = EnvironmentCanvas()
    canvas.set_state(_state())

    assert len(canvas.odorscape_contour_source.data["x"]) > 0
    assert len(canvas.windscape_segment_source.data["x0"]) == 9
    assert len(canvas.windscape_head_source.data["x"]) == 9
    assert len(canvas.thermoscape_aura_source.data["x"]) == 6
    assert canvas.thermoscape_marker_source.data["id"] == ["hot", "cold"]


def test_environment_canvas_clear_and_highlight_are_stable() -> None:
    canvas = EnvironmentCanvas()
    canvas.set_state(_state())

    canvas.set_selected_object("patch")
    assert canvas.food_highlight_source.data["x"] == [0.01]

    canvas.set_selected_object("missing")
    assert canvas.food_highlight_source.data["x"] == []

    view = canvas.view()
    canvas.clear()

    assert canvas.view() is view
    assert canvas.food_source.data["id"] == []
    assert canvas.source_group_member_source.data["x"] == []
    assert canvas.border_source.data["id"] == []


def test_environment_canvas_malformed_optional_scapes_do_not_block_base_render() -> (
    None
):
    canvas = EnvironmentCanvas()
    state = EnvironmentCanvasState(
        arena=CanvasArena("rectangular", (0.2, 0.1)),
        objects=(
            CanvasObject(
                object_id="patch",
                object_type="source_unit",
                x=0.01,
                y=0.02,
                radius=0.004,
            ),
        ),
        odorscape={"grid_dims": object()},
        windscape={"wind_speed": "not-a-number"},
        thermoscape={"thermo_sources": {"bad": object()}},
    )

    canvas.set_state(state)

    assert canvas.arena_source.data["w"] == [0.2]
    assert canvas.food_source.data["id"] == ["patch"]


def test_environment_canvas_larva_group_unequal_circle_scale_draws_ellipse() -> None:
    canvas = EnvironmentCanvas()
    state = EnvironmentCanvasState(
        arena=CanvasArena("rectangular", (0.1, 0.06)),
        objects=(
            CanvasObject(
                object_id="navigator",
                object_type="larva_group",
                x=-0.04,
                y=0.0,
                color="black",
                distribution_mode="uniform",
                distribution_shape="circle",
                distribution_n=8,
                distribution_scale_x=0.005,
                distribution_scale_y=0.02,
            ),
        ),
    )

    canvas.set_state(state)

    assert canvas.larva_group_circle_source.data["id"] == []
    assert canvas.larva_group_ellipse_source.data["id"] == ["navigator"]
    assert canvas.larva_group_ellipse_source.data["w"] == [0.01]
    assert canvas.larva_group_ellipse_source.data["h"] == [0.04]
    assert len(canvas.larva_group_member_source.data["x0"]) == 8


def test_environment_canvas_named_colors_do_not_fallback_to_green() -> None:
    canvas = EnvironmentCanvas()
    state = EnvironmentCanvasState(
        arena=CanvasArena("rectangular", (0.1, 0.06)),
        objects=(
            CanvasObject(
                object_id="target",
                object_type="source_unit",
                x=0.0,
                y=0.0,
                radius=0.003,
                color="blue",
                amount=0.0,
            ),
            CanvasObject(
                object_id="navigator",
                object_type="larva_group",
                x=-0.04,
                y=0.0,
                color="black",
                distribution_mode="uniform",
                distribution_shape="circle",
                distribution_n=5,
                distribution_scale_x=0.0,
                distribution_scale_y=0.0,
            ),
        ),
    )

    canvas.set_state(state)

    assert canvas.food_source.data["fill_color"] == ["#adadff"]
    x0 = canvas.larva_group_member_source.data["x0"]
    x1 = canvas.larva_group_member_source.data["x1"]
    y0 = canvas.larva_group_member_source.data["y0"]
    y1 = canvas.larva_group_member_source.data["y1"]
    assert len(x0) == 5
    assert len(x1) == 5
    assert len(y0) == 5
    assert len(y1) == 5
    for i in range(5):
        cx = 0.5 * (x0[i] + x1[i])
        cy = 0.5 * (y0[i] + y1[i])
        length = ((x1[i] - x0[i]) ** 2 + (y1[i] - y0[i]) ** 2) ** 0.5
        assert cx == pytest.approx(-0.04)
        assert cy == pytest.approx(0.0)
        assert length == pytest.approx(2.0 * STATIC_LARVA_GROUP_MEMBER_HALF_LENGTH)
    assert "r" not in canvas.larva_group_member_source.data
    assert "size" not in canvas.larva_group_member_source.data
    assert canvas.larva_group_member_source.data["fill_color"] == [
        "#2e2e2e",
        "#2e2e2e",
        "#2e2e2e",
        "#2e2e2e",
        "#2e2e2e",
    ]
