from __future__ import annotations

from math import isnan
from types import SimpleNamespace

from larvaworld.portal.canvas_widgets.environment_models import LarvaPreviewFrame
from larvaworld.portal.simulation.preview_frames import (
    capture_larva_frame,
    generate_preview_frames,
)


class DummyAgent:
    def __init__(
        self,
        *,
        pos,
        head,
        midline,
        trajectory,
        color,
        segs=(),
    ) -> None:
        self.pos = pos
        self.head = head
        self.midline_xy = midline
        self.trajectory = trajectory
        self.color = color
        self.segs = segs


class DummySegment:
    def __init__(self, vertices) -> None:
        self.vertices = vertices


class LarvaContoured(DummyAgent):
    pass


class LarvaReplayContoured(DummyAgent):
    pass


class DummyAgents(list):
    def get_position(self):
        return [agent.pos for agent in self]

    @property
    def head(self):
        return SimpleNamespace(front_end=[agent.head for agent in self])


class DummyLauncher:
    def __init__(self, agents, *, tick: int = 0) -> None:
        self.agents = DummyAgents(agents)
        self.t = tick
        self.step_calls = 0

    def sim_step(self) -> None:
        self.t += 1
        self.step_calls += 1
        for agent in self.agents:
            if isinstance(agent.pos, list) and len(agent.pos) >= 2:
                agent.pos[0] += 1.0
                agent.pos[1] += 1.0
            if isinstance(agent.trajectory, list):
                agent.trajectory.append(tuple(agent.pos))


def test_capture_larva_frame_copies_expected_fields() -> None:
    launcher = DummyLauncher(
        [
            DummyAgent(
                pos=[0.0, 0.0],
                head=[0.001, 0.002],
                midline=[(0.0, 0.0), (0.001, 0.002)],
                trajectory=[(0.0, 0.0), (0.001, 0.001), (0.002, 0.002)],
                color="#111111",
                segs=(
                    DummySegment([(0.0, 0.0), (0.001, 0.0), (0.001, 0.001)]),
                    DummySegment([(0.001, 0.0), (0.002, 0.0), (0.002, 0.001)]),
                ),
            ),
            DummyAgent(
                pos=[0.01, 0.02],
                head=[0.011, 0.022],
                midline=[(0.01, 0.02), (0.012, 0.021)],
                trajectory=[(0.01, 0.02), (0.015, 0.025)],
                color="#222222",
            ),
        ],
        tick=7,
    )

    frame = capture_larva_frame(launcher, trail_length=2)

    assert isinstance(frame, LarvaPreviewFrame)
    assert frame.tick == 7
    assert frame.centroids == ((0.0, 0.0), (0.01, 0.02))
    assert frame.heads == ((0.001, 0.002), (0.011, 0.022))
    assert frame.midlines == (
        ((0.0, 0.0), (0.001, 0.002)),
        ((0.01, 0.02), (0.012, 0.021)),
    )
    assert frame.trails == (
        ((0.001, 0.001), (0.002, 0.002)),
        ((0.01, 0.02), (0.015, 0.025)),
    )
    assert frame.segment_polygons == (
        (
            ((0.0, 0.0), (0.001, 0.0), (0.001, 0.001)),
            ((0.001, 0.0), (0.002, 0.0), (0.002, 0.001)),
        ),
        (),
    )
    assert frame.colors == ("#111111", "#222222")


def test_capture_larva_frame_preserves_alignment_for_invalid_optional_data() -> None:
    launcher = DummyLauncher(
        [
            DummyAgent(
                pos=[0.0, 0.0],
                head=[0.001, 0.002],
                midline=[(0.0, 0.0), (0.001, 0.002)],
                trajectory=[(0.0, 0.0), (0.001, 0.001)],
                color="#111111",
            ),
            DummyAgent(
                pos=[0.02, 0.03],
                head=["bad", None],
                midline=[(0.02, 0.03)],
                trajectory=[(0.02, 0.03)],
                color=None,
            ),
            DummyAgent(
                pos=[0.04, 0.05],
                head=[0.041, 0.052],
                midline=[(0.04, 0.05), (0.041, 0.052)],
                trajectory=[(0.04, 0.05), (0.042, 0.054)],
                color="#333333",
                segs=(DummySegment([(0.04, 0.05), (0.041, 0.05), (0.041, 0.051)]),),
            ),
        ]
    )

    frame = capture_larva_frame(launcher, trail_length=5)

    assert len(frame.centroids) == 3
    assert len(frame.heads) == 3
    assert len(frame.midlines) == 3
    assert len(frame.trails) == 3
    assert len(frame.segment_polygons) == 3
    assert len(frame.colors) == 3

    assert frame.heads[0] == (0.001, 0.002)
    assert isnan(frame.heads[1][0]) and isnan(frame.heads[1][1])
    assert frame.heads[2] == (0.041, 0.052)

    assert frame.midlines[0] == ((0.0, 0.0), (0.001, 0.002))
    assert frame.midlines[1] == ((0.02, 0.03),)
    assert frame.midlines[2] == ((0.04, 0.05), (0.041, 0.052))

    assert frame.trails[0] == ((0.0, 0.0), (0.001, 0.001))
    assert frame.trails[1] == ((0.02, 0.03),)
    assert frame.trails[2] == ((0.04, 0.05), (0.042, 0.054))
    assert frame.segment_polygons[0] == ()
    assert frame.segment_polygons[1] == ()
    assert frame.segment_polygons[2] == (((0.04, 0.05), (0.041, 0.05), (0.041, 0.051)),)

    assert frame.colors == ("#111111", "", "#333333")


def test_capture_larva_frame_returns_empty_frame_for_empty_agents() -> None:
    frame = capture_larva_frame(DummyLauncher([], tick=3))
    assert frame == LarvaPreviewFrame(tick=3)


def test_capture_larva_frame_copies_values_not_references() -> None:
    agent = DummyAgent(
        pos=[0.0, 0.0],
        head=[0.001, 0.002],
        midline=[(0.0, 0.0), (0.001, 0.001)],
        trajectory=[(0.0, 0.0), (0.001, 0.001)],
        color="#111111",
        segs=(DummySegment([(0.0, 0.0), (0.001, 0.0), (0.001, 0.001)]),),
    )
    launcher = DummyLauncher([agent], tick=5)

    frame = capture_larva_frame(launcher)

    agent.pos[0] = 9.9
    agent.head[0] = 8.8
    agent.midline_xy[0] = (7.7, 7.7)
    agent.trajectory[0] = (6.6, 6.6)
    agent.segs[0].vertices[0] = (5.5, 5.5)

    assert frame.centroids == ((0.0, 0.0),)
    assert frame.heads == ((0.001, 0.002),)
    assert frame.midlines == (((0.0, 0.0), (0.001, 0.001)),)
    assert frame.trails == (((0.0, 0.0), (0.001, 0.001)),)
    assert frame.segment_polygons == ((((0.0, 0.0), (0.001, 0.0), (0.001, 0.001)),),)


def test_capture_larva_frame_skips_malformed_segment_vertices() -> None:
    launcher = DummyLauncher(
        [
            DummyAgent(
                pos=[0.0, 0.0],
                head=[0.0, 0.0],
                midline=[],
                trajectory=[],
                color="#111111",
                segs=(
                    DummySegment([(0.0, 0.0), (0.001, 0.0), (0.001, 0.001)]),
                    DummySegment([(0.0, 0.0), ("bad", None), (0.001, float("nan"))]),
                ),
            )
        ]
    )

    frame = capture_larva_frame(launcher)

    assert frame.segment_polygons == ((((0.0, 0.0), (0.001, 0.0), (0.001, 0.001)),),)


def test_capture_larva_frame_captures_explicit_contour_xy() -> None:
    agent = DummyAgent(
        pos=[0.0, 0.0],
        head=[0.0, 0.0],
        midline=[],
        trajectory=[],
        color="#111111",
    )
    agent.contour_xy = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001)]
    frame = capture_larva_frame(DummyLauncher([agent]))

    assert frame.body_contours == (((0.0, 0.0), (0.001, 0.0), (0.001, 0.001)),)


def test_capture_larva_frame_captures_vertices_only_for_contoured_agents() -> None:
    contoured = LarvaContoured(
        pos=[0.0, 0.0],
        head=[0.0, 0.0],
        midline=[],
        trajectory=[],
        color="#111111",
    )
    contoured.vertices = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001)]

    replay_contoured = LarvaReplayContoured(
        pos=[0.01, 0.01],
        head=[0.01, 0.01],
        midline=[],
        trajectory=[],
        color="#222222",
    )
    replay_contoured.vertices = [(0.01, 0.01), (0.011, 0.01), (0.011, 0.011)]

    segmented_like = DummyAgent(
        pos=[0.02, 0.02],
        head=[0.02, 0.02],
        midline=[],
        trajectory=[],
        color="#333333",
        segs=(DummySegment([(0.02, 0.02), (0.021, 0.02), (0.021, 0.021)]),),
    )
    segmented_like.vertices = [(0.02, 0.02), (0.021, 0.02), (0.021, 0.021)]

    frame = capture_larva_frame(
        DummyLauncher([contoured, replay_contoured, segmented_like])
    )

    assert frame.body_contours == (
        ((0.0, 0.0), (0.001, 0.0), (0.001, 0.001)),
        ((0.01, 0.01), (0.011, 0.01), (0.011, 0.011)),
        (),
    )


def test_capture_larva_frame_handles_contour_property_errors() -> None:
    class BrokenContourAgent(DummyAgent):
        @property
        def contour_xy(self):
            raise ValueError("broken contour property")

    agent = BrokenContourAgent(
        pos=[0.0, 0.0],
        head=[0.0, 0.0],
        midline=[],
        trajectory=[],
        color="#111111",
    )
    frame = capture_larva_frame(DummyLauncher([agent]))

    assert frame.body_contours == ((),)


def test_capture_larva_frame_skips_malformed_explicit_contours() -> None:
    agent = DummyAgent(
        pos=[0.0, 0.0],
        head=[0.0, 0.0],
        midline=[],
        trajectory=[],
        color="#111111",
    )
    agent.contour_xy = [(0.0, 0.0), ("bad", None), (0.001, float("nan"))]
    frame = capture_larva_frame(DummyLauncher([agent]))

    assert frame.body_contours == ((),)


def test_capture_larva_frame_does_not_call_get_shape() -> None:
    class GetShapeExplodes(DummyAgent):
        def get_shape(self):
            raise AssertionError("get_shape must not be called")

    agent = GetShapeExplodes(
        pos=[0.0, 0.0],
        head=[0.0, 0.0],
        midline=[],
        trajectory=[],
        color="#111111",
        segs=(DummySegment([(0.0, 0.0), (0.001, 0.0), (0.001, 0.001)]),),
    )
    frame = capture_larva_frame(DummyLauncher([agent]))

    assert frame.body_contours == ((),)


def test_capture_larva_frame_copies_contours_not_references() -> None:
    contour = [(0.0, 0.0), (0.001, 0.0), (0.001, 0.001)]
    agent = DummyAgent(
        pos=[0.0, 0.0],
        head=[0.0, 0.0],
        midline=[],
        trajectory=[],
        color="#111111",
    )
    agent.contour_xy = contour
    frame = capture_larva_frame(DummyLauncher([agent]))

    contour[0] = (9.0, 9.0)
    assert frame.body_contours == (((0.0, 0.0), (0.001, 0.0), (0.001, 0.001)),)


def test_capture_larva_frame_tick_override() -> None:
    launcher = DummyLauncher(
        [
            DummyAgent(
                pos=[0.0, 0.0],
                head=[0.0, 0.0],
                midline=[],
                trajectory=[],
                color="black",
            )
        ],
        tick=4,
    )
    frame = capture_larva_frame(launcher, tick=99)
    assert frame.tick == 99


def test_generate_preview_frames_returns_exact_count_and_steps() -> None:
    launcher = DummyLauncher(
        [
            DummyAgent(
                pos=[0.0, 0.0],
                head=[0.001, 0.002],
                midline=[(0.0, 0.0), (0.001, 0.001)],
                trajectory=[(0.0, 0.0), (0.001, 0.001)],
                color="#111111",
            )
        ],
        tick=0,
    )

    frames = generate_preview_frames(launcher, preview_steps=5)

    assert len(frames) == 5
    assert launcher.step_calls == 4
    assert [frame.tick for frame in frames] == [0, 1, 2, 3, 4]


def test_generate_preview_frames_returns_empty_list_for_non_positive_steps() -> None:
    launcher = DummyLauncher([], tick=10)
    assert generate_preview_frames(launcher, preview_steps=0) == []
    assert generate_preview_frames(launcher, preview_steps=-3) == []
    assert launcher.step_calls == 0
