"""Focused tests for the non-blocking Explore preview worker."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from larvaworld.portal.canvas_widgets.environment_models import LarvaPreviewFrame
from larvaworld.portal.explore.analysis import (
    PreviewAnalysisResult,
    PreviewFigure,
)
from larvaworld.portal.explore.preview_job import (
    ExplorePreviewJob,
    _PreviewCancelled,
)
from larvaworld.portal.explore.scenarios import scenario_by_id


class _ScreenManager:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Launcher:
    dt = 0.1

    def __init__(self) -> None:
        self.screen_manager = _ScreenManager()


class _Runner:
    def __init__(self, _launcher, *, total_steps, on_frame, on_progress) -> None:
        self.total_steps = total_steps
        self.on_frame = on_frame
        self.on_progress = on_progress
        self.frames: list[LarvaPreviewFrame] = []

    def run_to_completion(self) -> list[LarvaPreviewFrame]:
        for tick in range(self.total_steps):
            frame = LarvaPreviewFrame(tick=tick, centroids=((float(tick), 0.0),))
            self.frames.append(frame)
            self.on_frame(frame)
            self.on_progress(tick + 1, self.total_steps)
        return self.frames


def _patch_worker_dependencies(monkeypatch, launcher, *, datasets, analysis) -> dict:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "larvaworld.portal.simulation.run_playback.build_bounded_launcher",
        lambda *args, **kwargs: captured.update({"args": args, "kwargs": kwargs})
        or (launcher, "fallback note"),
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.run_playback.ChunkedFrameRunner", _Runner
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.run_playback.finalize_preview_datasets",
        lambda received_launcher: captured.setdefault("finalized", received_launcher)
        and datasets,
    )
    monkeypatch.setattr(
        "larvaworld.portal.explore.analysis.preview_enrichment",
        lambda scenario_id: {"scenario": scenario_id},
    )
    monkeypatch.setattr(
        "larvaworld.portal.explore.analysis.build_preview_analysis",
        lambda scenario_id, received_datasets: captured.update(
            {"analysis_scenario": scenario_id, "analysis_datasets": received_datasets}
        )
        or analysis,
    )
    return captured


def _drain_progress(job: ExplorePreviewJob) -> list:
    events = []
    while not job.progress_queue.empty():
        events.append(job.progress_queue.get_nowait())
    return events


def test_job_uses_one_launcher_for_frames_dataset_and_analysis(monkeypatch) -> None:
    scenario = scenario_by_id("dish")
    launcher = _Launcher()
    datasets = [SimpleNamespace(config=SimpleNamespace(id="preview"))]
    analysis = SimpleNamespace(figures=[], warnings=[], metrics=[])
    captured = _patch_worker_dependencies(
        monkeypatch,
        launcher,
        datasets=datasets,
        analysis=analysis,
    )
    job = ExplorePreviewJob(scenario=scenario, parameters={"resolved": True})

    payload = job.run()

    assert len(payload.frames) == scenario.step_cap
    assert payload.datasets is datasets
    assert payload.analysis is analysis
    assert payload.dt == pytest.approx(0.1)
    assert payload.note == "fallback note"
    assert captured["args"][:2] == (scenario.exp_id, {"resolved": True})
    assert captured["kwargs"] == {
        "step_cap": scenario.step_cap,
        "analysis_enrichment": {"scenario": scenario.id},
    }
    assert captured["finalized"] is launcher
    assert captured["analysis_scenario"] == scenario.id
    assert captured["analysis_datasets"] is datasets
    assert launcher.screen_manager.closed is True

    events = _drain_progress(job)
    assert [event.phase for event in events] == [
        "initializing",
        *("frames" for _ in range(scenario.step_cap)),
        "dataset",
        "analysis",
        "ready",
    ]
    frame_counts = [event.completed for event in events if event.phase == "frames"]
    assert frame_counts == list(range(1, scenario.step_cap + 1))
    assert all(
        event.total == scenario.step_cap for event in events if event.phase == "frames"
    )

    temporary_path = Path(payload.temporary_directory.name)
    assert temporary_path.exists()
    payload.release()
    assert not temporary_path.exists()


def test_job_keeps_playback_when_dataset_finalization_fails(monkeypatch) -> None:
    scenario = scenario_by_id("dish")
    launcher = _Launcher()
    analysis_calls: list[object] = []
    _patch_worker_dependencies(
        monkeypatch,
        launcher,
        datasets=[],
        analysis=SimpleNamespace(figures=[], warnings=[], metrics=[]),
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.run_playback.finalize_preview_datasets",
        lambda _launcher: (_ for _ in ()).throw(RuntimeError("dataset boom")),
    )
    monkeypatch.setattr(
        "larvaworld.portal.explore.analysis.build_preview_analysis",
        lambda *_args: analysis_calls.append(object()),
    )
    job = ExplorePreviewJob(scenario=scenario, parameters={})

    payload = job.run()

    assert len(payload.frames) == scenario.step_cap
    assert payload.datasets == []
    assert analysis_calls == []
    assert "dataset boom" in payload.analysis.warnings[0]
    payload.release()


def test_job_hands_png_bytes_to_the_ui_instead_of_live_figures(monkeypatch) -> None:
    from matplotlib.figure import Figure

    scenario = replace(scenario_by_id("dish"), step_cap=1, n_agents=1)
    launcher = _Launcher()
    figure = Figure(figsize=(2, 1))
    figure.subplots().plot([0, 1], [0, 1])
    analysis = PreviewAnalysisResult(
        figures=[PreviewFigure(title="Worker plot", figure=figure)]
    )
    _patch_worker_dependencies(
        monkeypatch,
        launcher,
        datasets=[SimpleNamespace(config=SimpleNamespace(id="preview"))],
        analysis=analysis,
    )

    payload = ExplorePreviewJob(scenario=scenario, parameters={}).run()

    assert len(payload.analysis.figures) == 1
    rendered = payload.analysis.figures[0]
    assert rendered.figure is None
    assert rendered.png is not None
    assert rendered.png.startswith(b"\x89PNG\r\n\x1a\n")
    payload.release()


def test_job_cancels_between_simulation_steps_and_cleans_resources(monkeypatch) -> None:
    scenario = scenario_by_id("dish")
    launcher = _Launcher()
    _patch_worker_dependencies(
        monkeypatch,
        launcher,
        datasets=[],
        analysis=SimpleNamespace(figures=[], warnings=[], metrics=[]),
    )
    job = ExplorePreviewJob(scenario=scenario, parameters={})

    class _CancellingRunner(_Runner):
        def run_to_completion(self) -> list[LarvaPreviewFrame]:
            frame = LarvaPreviewFrame(tick=0, centroids=((0.0, 0.0),))
            self.frames.append(frame)
            self.on_progress(1, self.total_steps)
            job.cancel()
            self.on_progress(2, self.total_steps)
            return self.frames

    monkeypatch.setattr(
        "larvaworld.portal.simulation.run_playback.ChunkedFrameRunner",
        _CancellingRunner,
    )

    with pytest.raises(_PreviewCancelled):
        job.run()

    assert launcher.screen_manager.closed is True
