"""Background job for non-blocking Explore simulation previews.

The job intentionally has no Panel or Bokeh imports.  It owns all expensive
simulation, dataset, and Matplotlib work while the Explore controller remains
responsible for rendering progress and results on the document thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from queue import SimpleQueue
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any

__all__: list[str] = []


class _PreviewCancelled(Exception):
    """Internal signal used to stop a preview between expensive operations."""


@dataclass(frozen=True)
class PreviewProgress:
    """One thread-safe progress update emitted by an Explore preview job."""

    phase: str
    completed: int | None = None
    total: int | None = None


@dataclass
class ExplorePreviewPayload:
    """In-memory preview result transferred from the worker to the UI thread."""

    frames: list[Any]
    datasets: list[Any]
    analysis: Any
    dt: float
    note: str | None
    temporary_directory: TemporaryDirectory[str]
    _released: bool = field(default=False, init=False, repr=False)

    def release(self) -> None:
        """Release figures and temporary files when a result is discarded."""
        if self._released:
            return
        self._released = True
        try:
            from larvaworld.portal.explore.analysis import close_preview_figures

            close_preview_figures(self.analysis.figures)
        finally:
            self.temporary_directory.cleanup()


class ExplorePreviewJob:
    """Create one Explore preview without touching Panel or Bokeh objects."""

    def __init__(self, *, scenario: Any, parameters: Any) -> None:
        self.scenario = scenario
        self.parameters = parameters
        self.progress_queue: SimpleQueue[PreviewProgress] = SimpleQueue()
        self._cancel_event = Event()

    def cancel(self) -> None:
        """Request cooperative cancellation of the active preview."""
        self._cancel_event.set()

    def _check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise _PreviewCancelled()

    def _publish(
        self,
        phase: str,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        self.progress_queue.put(PreviewProgress(phase, completed, total))

    def _on_frame_progress(self, completed: int, total: int) -> None:
        self._check_cancelled()
        self._publish("frames", completed, total)

    def run(self) -> ExplorePreviewPayload:
        """Run the bounded simulation, enrichment, and plots in one worker."""
        # These imports are deliberately worker-local: their first import can be
        # expensive and must not delay the interactive Explore page.
        from larvaworld.portal.explore.analysis import (
            PreviewAnalysisResult,
            build_preview_analysis,
            close_preview_figures,
            preview_enrichment,
            render_preview_figure_images,
        )
        from larvaworld.portal.simulation.run_playback import (
            ChunkedFrameRunner,
            build_bounded_launcher,
            finalize_preview_datasets,
        )

        temporary_directory = TemporaryDirectory(prefix="lw_explore_")
        launcher = None
        screen_closed = False
        analysis = PreviewAnalysisResult()
        try:
            self._publish("initializing")
            run_dir = Path(temporary_directory.name) / self.scenario.id
            run_dir.mkdir(parents=True, exist_ok=True)
            self._check_cancelled()
            launcher, note = build_bounded_launcher(
                self.scenario.exp_id,
                self.parameters,
                run_dir,
                step_cap=self.scenario.step_cap,
                analysis_enrichment=preview_enrichment(self.scenario.id),
            )
            self._check_cancelled()

            runner = ChunkedFrameRunner(
                launcher,
                total_steps=self.scenario.step_cap,
                on_frame=lambda _frame: None,
                on_progress=self._on_frame_progress,
            )
            frames = runner.run_to_completion()
            self._check_cancelled()
            if not frames:
                raise ValueError("No preview frames were generated.")

            self._publish("dataset")
            try:
                datasets = finalize_preview_datasets(launcher)
            except Exception as exc:
                datasets = []
                analysis.warnings.append(
                    f"Preview datasets were unavailable: {type(exc).__name__}: {exc}"
                )
            else:
                self._check_cancelled()
                self._publish("analysis")
                try:
                    analysis = build_preview_analysis(self.scenario.id, datasets)
                    self._check_cancelled()
                    render_preview_figure_images(analysis)
                except _PreviewCancelled:
                    raise
                except Exception as exc:
                    analysis.warnings.append(
                        f"Preview analysis was unavailable: {type(exc).__name__}: {exc}"
                    )
            self._check_cancelled()
            if launcher is not None and getattr(launcher, "screen_manager", None):
                try:
                    launcher.screen_manager.close()
                except Exception:
                    pass
                finally:
                    screen_closed = True
            self._publish("ready", len(frames), self.scenario.step_cap)
            return ExplorePreviewPayload(
                frames=frames,
                datasets=datasets,
                analysis=analysis,
                dt=float(launcher.dt),
                note=note,
                temporary_directory=temporary_directory,
            )
        except BaseException:
            close_preview_figures(analysis.figures)
            temporary_directory.cleanup()
            raise
        finally:
            try:
                if (
                    not screen_closed
                    and launcher is not None
                    and getattr(launcher, "screen_manager", None)
                ):
                    launcher.screen_manager.close()
            except Exception:
                pass
