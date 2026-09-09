"""Tests for the shared bounded-run / playback engine."""

from __future__ import annotations

from threading import Event

import pytest
from bokeh.document import Document

from larvaworld.portal.canvas_widgets.environment_models import LarvaPreviewFrame
from larvaworld.portal.simulation.run_playback import (
    DEFAULT_TICK_INTERVAL_MS,
    ChunkedFrameRunner,
)


class _FakeLauncher:
    """Minimal stand-in for an ExpRun: advances a tick and moves one agent."""

    def __init__(self, *, fail_at: int | None = None) -> None:
        self.t = 0
        self.steps = 0
        self.fail_at = fail_at

    def sim_step(self) -> None:
        self.steps += 1
        self.t += 1
        if self.fail_at is not None and self.steps >= self.fail_at:
            raise RuntimeError("simulation exploded")


@pytest.fixture(autouse=True)
def _stub_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    def _capture(launcher, *, trail_length=30, **kwargs):  # noqa: ANN001, ANN003
        return LarvaPreviewFrame(
            tick=int(launcher.t),
            centroids=((float(launcher.t), 0.0),),
            heads=(),
            midlines=(),
            trails=(),
            segment_polygons=(),
            body_contours=(),
            colors=("",),
        )

    monkeypatch.setattr(
        "larvaworld.portal.simulation.run_playback.capture_larva_frame", _capture
    )


def test_runner_produces_exactly_the_capped_frame_count() -> None:
    frames: list[LarvaPreviewFrame] = []
    runner = ChunkedFrameRunner(
        _FakeLauncher(), total_steps=25, on_frame=frames.append, chunk_size=4
    )
    runner.run_to_completion()

    assert len(frames) == 25
    assert runner.finished is True


def test_runner_does_not_overrun_the_cap() -> None:
    launcher = _FakeLauncher()
    runner = ChunkedFrameRunner(
        launcher, total_steps=10, on_frame=lambda _f: None, chunk_size=7
    )
    runner.run_to_completion()

    # One fewer step than frames: the first frame is captured before stepping.
    assert launcher.steps == 9
    assert len(runner.frames) == 10


def test_runner_reports_progress_monotonically() -> None:
    seen: list[tuple[int, int]] = []
    runner = ChunkedFrameRunner(
        _FakeLauncher(),
        total_steps=12,
        on_frame=lambda _f: None,
        on_progress=lambda done, total: seen.append((done, total)),
        chunk_size=3,
    )
    runner.run_to_completion()

    assert seen
    assert all(total == 12 for _done, total in seen)
    assert [done for done, _t in seen] == sorted(done for done, _t in seen)
    assert seen[-1][0] == 12


def test_runner_fires_completion_once_with_all_frames() -> None:
    completed: list[list[LarvaPreviewFrame]] = []
    runner = ChunkedFrameRunner(
        _FakeLauncher(),
        total_steps=8,
        on_frame=lambda _f: None,
        on_complete=completed.append,
        chunk_size=5,
    )
    runner.run_to_completion()

    assert len(completed) == 1
    assert len(completed[0]) == 8


def test_runner_routes_failures_to_the_error_callback() -> None:
    errors: list[Exception] = []
    runner = ChunkedFrameRunner(
        _FakeLauncher(fail_at=3),
        total_steps=50,
        on_frame=lambda _f: None,
        on_error=errors.append,
        chunk_size=10,
    )
    runner.run_to_completion()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert runner.finished is True


def test_runner_raises_when_no_error_callback_is_given() -> None:
    runner = ChunkedFrameRunner(
        _FakeLauncher(fail_at=2), total_steps=50, on_frame=lambda _f: None
    )
    with pytest.raises(RuntimeError, match="simulation exploded"):
        runner.run_to_completion()


def test_stop_is_safe_before_start() -> None:
    runner = ChunkedFrameRunner(
        _FakeLauncher(), total_steps=5, on_frame=lambda _f: None
    )
    runner.stop()
    assert runner.finished is False


def test_runner_uses_explicit_document_without_blocking() -> None:
    runner = ChunkedFrameRunner(
        _FakeLauncher(), total_steps=5, on_frame=lambda _frame: None
    )
    document = Document()

    runner.start(document=document)

    assert runner._callback is not None
    assert runner._callback in document.session_callbacks
    assert runner._callback.period == DEFAULT_TICK_INTERVAL_MS
    assert runner.frames == []
    runner.stop()
    assert document.session_callbacks == []


def test_interactive_runner_computes_steps_outside_the_document_callback() -> None:
    class _BlockingLauncher(_FakeLauncher):
        def __init__(self) -> None:
            super().__init__()
            self.entered_step = Event()
            self.release_step = Event()

        def sim_step(self) -> None:
            self.entered_step.set()
            assert self.release_step.wait(timeout=2)
            super().sim_step()

    launcher = _BlockingLauncher()
    delivered: list[LarvaPreviewFrame] = []
    runner = ChunkedFrameRunner(launcher, total_steps=2, on_frame=delivered.append)
    runner.start(document=Document())

    runner._advance_chunk()
    assert launcher.entered_step.wait(timeout=1)
    assert delivered == []

    launcher.release_step.set()
    assert runner._frame_future is not None
    runner._frame_future.result(timeout=1)
    runner._advance_chunk()

    assert len(delivered) == 1
    runner.stop()


# ---- runtime_parameters / build_bounded_launcher --------------------------


def test_runtime_parameters_strip_collection_and_enrichment_overhead() -> None:
    from larvaworld.lib import reg, util
    from larvaworld.portal.simulation.run_playback import runtime_parameters

    original = util.AttrDict(reg.conf.Exp.expand("dish"))
    stripped = runtime_parameters(original)

    assert stripped["collections"] == []
    assert stripped["enrichment"] is None
    # The source config must not be mutated.
    assert original.get("collections") != [] or original.get("enrichment") is not None


def test_runtime_parameters_preserve_scientific_settings() -> None:
    from larvaworld.lib import reg, util
    from larvaworld.portal.simulation.run_playback import runtime_parameters

    original = util.AttrDict(reg.conf.Exp.expand("dish"))
    stripped = runtime_parameters(original)

    assert dict(stripped.env_params.arena) == dict(original.env_params.arena)
    assert stripped.get("dt") == original.get("dt")
    assert stripped.get("duration") == original.get("duration")


def test_build_bounded_launcher_caps_steps_and_writes_nothing(tmp_path) -> None:
    from larvaworld.lib import reg, util
    from larvaworld.portal.simulation.run_playback import build_bounded_launcher

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parameters = util.AttrDict(reg.conf.Exp.expand("dish"))

    launcher, note = build_bounded_launcher("dish", parameters, run_dir, step_cap=12)

    assert note is None or isinstance(note, str)
    # sim_setup stores the cap as an absolute stop tick, independent of the
    # much larger Nsteps the stored config would otherwise run for.
    assert launcher._steps == 12
    assert launcher.p["Nsteps"] > 12
    assert launcher.p["store_data"] is False
    assert list(run_dir.iterdir()) == []


def test_bounded_launcher_stops_running_at_the_cap(tmp_path) -> None:
    from larvaworld.lib import reg, util
    from larvaworld.portal.simulation.run_playback import build_bounded_launcher

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parameters = util.AttrDict(reg.conf.Exp.expand("dish"))
    for group in parameters.larva_groups.values():
        group.distribution["N"] = 2

    launcher, _note = build_bounded_launcher("dish", parameters, run_dir, step_cap=4)
    assert launcher.running is True
    for _ in range(4):
        launcher.sim_step()

    assert launcher.running is False


def test_bounded_launcher_drives_a_real_simulation_forward(tmp_path) -> None:
    from larvaworld.lib import reg, util
    from larvaworld.portal.simulation.preview_frames import capture_larva_frame
    from larvaworld.portal.simulation.run_playback import build_bounded_launcher

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    parameters = util.AttrDict(reg.conf.Exp.expand("dish"))
    for group in parameters.larva_groups.values():
        group.distribution["N"] = 2

    launcher, _note = build_bounded_launcher("dish", parameters, run_dir, step_cap=8)
    first = capture_larva_frame(launcher)
    for _ in range(5):
        launcher.sim_step()
    later = capture_larva_frame(launcher)

    assert later.tick > first.tick
    assert len(later.centroids) == len(first.centroids) == 2
    assert later.centroids != first.centroids
