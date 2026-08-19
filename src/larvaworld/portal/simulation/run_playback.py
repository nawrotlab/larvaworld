"""Shared engine for bounded, watchable simulation runs in the portal.

The portal needs to show a simulation *while it happens* rather than block the
Bokeh document until it finishes. This module provides the three pieces that
make that possible and keeps them reusable across apps:

* :func:`build_bounded_launcher` - build an ``ExpRun`` that never persists data
  and is capped at a known number of steps.
* :class:`ChunkedFrameRunner` - advance that launcher a few steps per event-loop
  tick, emitting one :class:`LarvaPreviewFrame` per step, so the UI stays live.
* :class:`FramePlayback` - scrub through captured frames afterwards.

Because the step count is bounded up front, total runtime is predictable, which
is what makes a "press once and watch" experience safe on a laptop.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import panel as pn

from larvaworld.lib import sim, util
from larvaworld.portal.canvas_widgets.environment_models import LarvaPreviewFrame
from larvaworld.portal.simulation.preview_frames import capture_larva_frame

#: One simulation step per event-loop tick keeps navigation controls responsive
#: while a live preview is running.
DEFAULT_CHUNK_SIZE = 1

#: Leave time between simulation steps for Bokeh to process canvas interactions
#: such as legend clicks while a live preview is running.
DEFAULT_TICK_INTERVAL_MS = 100


def runtime_parameters(parameters: util.AttrDict) -> util.AttrDict:
    """Strip data-collection and enrichment from a config for display-only runs.

    Args:
        parameters: A resolved experiment configuration.

    Returns:
        A copy with ``collections`` and ``enrichment`` cleared, so the run does
        no bookkeeping it will not use.
    """
    preview_parameters = util.AttrDict(parameters.get_copy())
    preview_parameters["collections"] = []
    preview_parameters["enrichment"] = None
    return preview_parameters


def build_bounded_launcher(
    experiment: str,
    parameters: util.AttrDict,
    run_dir: Path,
    *,
    step_cap: int,
) -> tuple[Any, str | None]:
    """Build an ``ExpRun`` set up to run at most ``step_cap`` steps.

    Mirrors the preview launcher used by the Single Experiment app, including
    its retry when agent overlap elimination fails on a crowded arena.

    Args:
        experiment: A stored ``Exp`` configuration ID.
        parameters: The resolved configuration for that experiment.
        run_dir: Directory handed to the launcher; nothing is written there
            because ``store_data`` is disabled.
        step_cap: Maximum simulation steps.

    Returns:
        The launcher and an optional note describing any fallback applied.
    """
    preview_parameters = runtime_parameters(parameters)
    try:
        launcher = sim.ExpRun(
            experiment=experiment,
            parameters=preview_parameters,
            id=run_dir.name,
            dir=str(run_dir),
            store_data=False,
        )
        launcher.sim_setup(steps=step_cap)
        return launcher, None
    except Exception as exc:
        if "get_polygon" not in str(exc):
            raise
        preview_parameters = preview_parameters.get_copy()
        preview_parameters["larva_collisions"] = True
        launcher = sim.ExpRun(
            experiment=experiment,
            parameters=preview_parameters,
            id=run_dir.name,
            dir=str(run_dir),
            store_data=False,
        )
        launcher.sim_setup(steps=step_cap)
        return (
            launcher,
            "Overlap elimination was disabled for this run so the arena could be built.",
        )


class ChunkedFrameRunner:
    """Advance a launcher in small chunks, emitting frames as they are produced.

    Interactive runs compute a frame and its following simulation step on a
    worker thread. The Panel callback only publishes completed frames, keeping
    the Bokeh document free to process navigation and canvas interactions.
    """

    def __init__(
        self,
        launcher: Any,
        *,
        total_steps: int,
        on_frame: Callable[[LarvaPreviewFrame], None],
        on_progress: Callable[[int, int], None] | None = None,
        on_complete: Callable[[list[LarvaPreviewFrame]], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        trail_length: int = 30,
    ) -> None:
        self.launcher = launcher
        self.total_steps = max(1, int(total_steps))
        self.on_frame = on_frame
        self.on_progress = on_progress
        self.on_complete = on_complete
        self.on_error = on_error
        self.chunk_size = max(1, int(chunk_size))
        self.trail_length = trail_length
        self.frames: list[LarvaPreviewFrame] = []
        self._callback: Any = None
        self._document: Any | None = None
        self._worker: ThreadPoolExecutor | None = None
        self._frame_future: Future[LarvaPreviewFrame] | None = None
        self._submitted_frames = 0
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    def start(
        self,
        *,
        interval_ms: int = DEFAULT_TICK_INTERVAL_MS,
        document: Any | None = None,
    ) -> None:
        """Begin stepping, scheduling on an explicit Bokeh document when supplied.

        The synchronous fallback is reserved for headless callers. Interactive
        controllers should pass their session document so a transient callback
        context cannot turn a live preview into a blocking run.
        """
        document = document or pn.state.curdoc
        if document is None:
            self.run_to_completion()
            return
        self._worker = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="lw-preview",
        )
        self._document = document
        self._callback = document.add_periodic_callback(
            self._advance_chunk,
            interval_ms,
        )

    def stop(self) -> None:
        if self._callback is not None:
            try:
                self._document.remove_periodic_callback(self._callback)
            except (ValueError, RuntimeError):
                pass
            self._callback = None
            self._document = None
        if self._worker is not None:
            self._worker.shutdown(wait=False, cancel_futures=True)
            self._worker = None
            self._frame_future = None

    def run_to_completion(self) -> list[LarvaPreviewFrame]:
        """Step synchronously until the cap is reached. Used headless and in tests."""
        while not self._finished:
            self._advance_synchronously()
        return self.frames

    def _capture(self) -> None:
        frame = capture_larva_frame(self.launcher, trail_length=self.trail_length)
        self.frames.append(frame)
        self.on_frame(frame)

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self.stop()
        if self.on_complete is not None:
            self.on_complete(self.frames)

    def _advance_synchronously(self) -> None:
        if self._finished:
            return
        try:
            for _ in range(self.chunk_size):
                if len(self.frames) >= self.total_steps:
                    self._finish()
                    return
                self._capture()
                if len(self.frames) < self.total_steps:
                    self.launcher.sim_step()
        except Exception as exc:
            self._finished = True
            self.stop()
            if self.on_error is not None:
                self.on_error(exc)
            else:
                raise
            return
        if self.on_progress is not None:
            self.on_progress(len(self.frames), self.total_steps)
        if len(self.frames) >= self.total_steps:
            self._finish()

    def _compute_frame(self, frame_index: int) -> LarvaPreviewFrame:
        """Capture one frame and advance the simulation off the Bokeh thread."""
        frame = capture_larva_frame(self.launcher, trail_length=self.trail_length)
        if frame_index + 1 < self.total_steps:
            self.launcher.sim_step()
        return frame

    def _advance_chunk(self) -> None:
        """Publish one completed worker frame and schedule the next one."""
        if self._finished or self._worker is None:
            return
        if self._frame_future is not None:
            if not self._frame_future.done():
                return
            try:
                frame = self._frame_future.result()
            except Exception as exc:
                self._finished = True
                self.stop()
                if self.on_error is not None:
                    self.on_error(exc)
                else:
                    raise
                return
            self._frame_future = None
            self.frames.append(frame)
            self.on_frame(frame)
            if self.on_progress is not None:
                self.on_progress(len(self.frames), self.total_steps)
            if len(self.frames) >= self.total_steps:
                self._finish()
                return

        self._frame_future = self._worker.submit(
            self._compute_frame,
            self._submitted_frames,
        )
        self._submitted_frames += 1


class FramePlayback:
    """Scrub through captured frames on an :class:`EnvironmentCanvas`."""

    def __init__(
        self,
        *,
        canvas: Any,
        frames: list[LarvaPreviewFrame],
        dt: float,
    ) -> None:
        if not frames:
            raise ValueError("frames must not be empty")
        self.canvas = canvas
        self.frames = list(frames)
        self.dt = max(0.0, float(dt))
        self.frame_player = pn.widgets.Player(
            name="Frame",
            start=0,
            end=len(self.frames) - 1,
            value=0,
            step=1,
            interval=max(50, min(1000, int(self.dt * 1000))),
            loop_policy="once",
            show_loop_controls=False,
            width=420,
        )
        self.metadata = pn.pane.HTML("", sizing_mode="stretch_width")
        self.frame_player.param.watch(self._on_player_change, "value")
        self._show_frame(0)

    def _set_metadata(self, index: int) -> None:
        frame = self.frames[index]
        end_index = len(self.frames) - 1
        timestamp = frame.tick * self.dt
        end_time = self.frames[-1].tick * self.dt
        self.metadata.object = (
            '<div class="lw-single-exp-preview-meta">'
            f"<strong>Frame:</strong> {index}/{end_index}; "
            f"<strong>Tick:</strong> {frame.tick}; "
            f"<strong>Time:</strong> {timestamp:.1f} s; "
            f"<strong>Displayed range:</strong> 0.0-{end_time:.1f} s."
            "</div>"
        )

    def _show_frame(self, index: int) -> None:
        clamped = max(0, min(index, len(self.frames) - 1))
        if int(self.frame_player.value) != clamped:
            self.frame_player.value = clamped
            return
        self.canvas.set_larva_frame(self.frames[clamped])
        self._set_metadata(clamped)

    def _on_player_change(self, event: Any) -> None:
        self._show_frame(int(event.new))

    def view(self) -> pn.viewable.Viewable:
        return pn.Column(
            self.canvas.view(),
            pn.Row(pn.Column("Frame", self.frame_player), sizing_mode="stretch_width"),
            self.metadata,
            sizing_mode="stretch_width",
        )
