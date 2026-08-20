"""Tests for the Explore app: the zero-configuration entry point."""

from __future__ import annotations

from concurrent.futures import Future
from html import escape
from queue import SimpleQueue
from types import SimpleNamespace
from unittest.mock import PropertyMock

import panel as pn
import pytest

from larvaworld.lib import reg, util
from larvaworld.portal.explore.explore_app import (
    MAX_AGENTS,
    _ExploreController,
    _preview_dataset_tables_view,
    explore_app,
)
from larvaworld.portal.explore.scenarios import SCENARIOS, scenario_by_id


def _walk(obj, seen=None):
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj
    for child in getattr(obj, "objects", []) or []:
        yield from _walk(child, seen)


def test_explore_app_builds_a_template() -> None:
    view = explore_app()
    assert view.main
    assert view.header


def test_gallery_shows_every_scenario_with_a_watch_button() -> None:
    controller = _ExploreController()
    nodes = list(_walk(controller.view()))
    labels = [getattr(n, "name", None) for n in nodes]

    assert labels.count("Watch") == len(SCENARIOS)


def test_gallery_renders_human_titles_not_registry_ids() -> None:
    controller = _ExploreController()
    text = " ".join(str(getattr(n, "object", "")) for n in _walk(controller.view()))
    for scenario in SCENARIOS:
        assert scenario.title in text


def test_gallery_is_the_initial_state() -> None:
    controller = _ExploreController()
    # No preview has been generated, so nothing is holding a temp directory.
    assert controller._temp_dir is None


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_overrides_shrink_group_sizes_without_touching_other_settings(
    scenario,
) -> None:
    original = util.AttrDict(reg.conf.Exp.expand(scenario.exp_id))
    arena_before = dict(original.env_params.arena)
    duration_before = original.get("duration")

    patched = _ExploreController._apply_scenario_overrides(
        util.AttrDict(reg.conf.Exp.expand(scenario.exp_id)), scenario
    )

    total = sum(
        int(g.distribution.N)
        for g in patched.get("larva_groups", {}).values()
        if g.get("distribution") is not None
    )
    assert 0 < total <= MAX_AGENTS

    # Scientific settings must be untouched: only agent counts change.
    assert dict(patched.env_params.arena) == arena_before
    assert patched.get("duration") == duration_before


def test_overrides_tolerate_configs_without_groups() -> None:
    scenario = SCENARIOS[0]
    parameters = util.AttrDict({"larva_groups": {}})
    assert (
        _ExploreController._apply_scenario_overrides(parameters, scenario) is parameters
    )


def test_error_state_offers_a_way_back() -> None:
    controller = _ExploreController()
    scenario = SCENARIOS[0]
    controller._show_error(scenario, RuntimeError("boom"))

    text = " ".join(str(getattr(n, "object", "")) for n in _walk(controller.view()))
    assert "boom" in text
    assert scenario.title in text
    labels = [getattr(n, "name", None) for n in _walk(controller.view())]
    assert "< All scenarios" in labels


def test_scenario_lookup_is_wired_to_the_catalog() -> None:
    assert scenario_by_id(SCENARIOS[0].id) is SCENARIOS[0]


def test_catalog_includes_the_no_food_odor_choice_experiment() -> None:
    scenario = scenario_by_id("CS_UCS_off")

    assert scenario is not None
    assert scenario.exp_id == "PItest_off"
    assert scenario.category == "chemotaxis"


def test_scenario_stage_offers_an_offline_simulation_preview() -> None:
    controller = _ExploreController()
    controller.start_scenario(scenario_by_id("dish"))

    labels = [getattr(node, "name", None) for node in _walk(controller.view())]

    assert "Generate simulation preview" in labels
    assert "< All scenarios" in labels


def test_scenario_stage_shows_an_experiment_summary_above_the_canvas() -> None:
    controller = _ExploreController()
    scenario = scenario_by_id("dish")

    controller.start_scenario(scenario)

    stage = controller.body.objects
    summary = getattr(stage[2], "object", "")
    assert "Experiment summary" in summary
    assert scenario.teaser in summary
    assert escape(" ".join(scenario.explanation.split())) in summary


def test_scenario_stage_shows_static_larva_groups(monkeypatch) -> None:
    created: list[dict[str, object]] = []

    class _Canvas:
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)

        def set_state(self, _state) -> None:
            pass

        def view(self):
            return "canvas"

    monkeypatch.setattr(
        "larvaworld.portal.canvas_widgets.environment_canvas.EnvironmentCanvas",
        _Canvas,
    )
    controller = _ExploreController()

    controller.start_scenario(scenario_by_id("dish"))

    assert created == [{"editable": False, "show_larva_groups": True}]


def test_preview_request_starts_background_job_and_shows_progress(
    monkeypatch,
) -> None:
    from larvaworld.portal.explore import preview_job

    controller = _ExploreController()
    scenario = scenario_by_id("dish")
    parameters = util.AttrDict(reg.conf.Exp.expand(scenario.exp_id))

    class _Job:
        instances: list["_Job"] = []

        def __init__(self, *, scenario, parameters) -> None:
            self.scenario = scenario
            self.parameters = parameters
            self.progress_queue = SimpleQueue()
            self.ran = False
            self.cancelled = False
            self.instances.append(self)

        def run(self):
            self.ran = True
            raise AssertionError("The worker must not run in the click callback")

        def cancel(self) -> None:
            self.cancelled = True

    class _Future:
        def __init__(self) -> None:
            self.callbacks = []

        def done(self) -> bool:
            return False

        def cancel(self) -> bool:
            return True

        def add_done_callback(self, callback) -> None:
            self.callbacks.append(callback)

    class _Executor:
        def __init__(self) -> None:
            self.submitted = []
            self.future = _Future()

        def submit(self, callback):
            self.submitted.append(callback)
            return self.future

    class _Document:
        def __init__(self) -> None:
            self.callbacks = []
            self.session_context = None

        def add_periodic_callback(self, callback, period):
            self.callbacks.append((callback, period))
            return callback

        def remove_periodic_callback(self, callback) -> None:
            self.callbacks = [
                entry for entry in self.callbacks if entry[0] is not callback
            ]

    document = _Document()
    executor = _Executor()
    monkeypatch.setattr(
        preview_job,
        "ExplorePreviewJob",
        _Job,
    )
    monkeypatch.setattr(
        type(pn.state),
        "curdoc",
        PropertyMock(return_value=document),
    )
    controller._preview_executor = executor
    static_canvas = pn.pane.HTML("static scenario canvas")
    generate_preview = pn.widgets.Button(
        name="Generate simulation preview",
        width=240,
    )
    controls = pn.Row(pn.widgets.Button(name="< All scenarios"), generate_preview)
    controller._scenario_controls = controls
    controller._generate_preview_button = generate_preview
    controller._active_scenario_id = scenario.id
    controller.body[:] = [controls, pn.pane.HTML("title"), static_canvas]

    controller._request_simulation_preview(scenario, parameters)

    text = " ".join(
        str(getattr(node, "object", "")) for node in _walk(controller.view())
    )
    labels = [getattr(node, "name", None) for node in _walk(controller.view())]
    assert "Initializing environment and agents." in text
    assert "Cancel" in labels
    assert static_canvas in controller.body.objects
    assert controller.body.objects[-1] is static_canvas
    assert controller._preview_progress in controls.objects
    assert controls.objects.index(controller._preview_progress) == 2
    assert controller._preview_progress.width == 180
    assert generate_preview.disabled is True
    assert generate_preview.loading is True
    assert len(executor.submitted) == 1
    assert _Job.instances[0].ran is False
    assert document.callbacks[0][1] == 100

    controller._cancel_active_preview()
    assert _Job.instances[0].cancelled is True
    assert document.callbacks == []


def test_completed_worker_payload_is_rendered_on_the_next_document_tick(
    monkeypatch,
) -> None:
    from larvaworld.portal.canvas_widgets.environment_models import LarvaPreviewFrame
    from larvaworld.portal.explore.preview_job import PreviewProgress

    controller = _ExploreController()
    scenario = scenario_by_id("dish")
    parameters = _ExploreController._apply_scenario_overrides(
        util.AttrDict(reg.conf.Exp.expand(scenario.exp_id)), scenario
    )
    frames = [LarvaPreviewFrame(tick=0, centroids=((0.0, 0.0),))]
    captured: dict[str, object] = {}
    datasets = [SimpleNamespace(config=SimpleNamespace(id="preview"))]
    analysis = SimpleNamespace(metrics=[], figures=[], warnings=[])
    payload = SimpleNamespace(
        frames=frames,
        datasets=datasets,
        analysis=analysis,
        dt=0.1,
        note=None,
    )

    class _Job:
        progress_queue = SimpleQueue()

    class _Document:
        def __init__(self) -> None:
            self.next_tick_callbacks = []

        def add_next_tick_callback(self, callback) -> None:
            self.next_tick_callbacks.append(callback)

        def remove_periodic_callback(self, _callback) -> None:
            pass

    future: Future[object] = Future()
    future.set_result(payload)
    _Job.progress_queue.put(PreviewProgress("frames", 1, scenario.step_cap))
    controller._preview_request_id = 4
    controller._preview_job = _Job()
    controller._preview_future = future
    controller._preview_document = _Document()
    controller._preview_poll_callback = object()
    controller._preview_status = pn.pane.HTML("")
    controller._preview_progress = pn.indicators.Progress(
        value=-1, max=scenario.step_cap
    )
    monkeypatch.setattr(
        controller,
        "_show_preview_payload",
        lambda *args: captured.update(
            {
                "scenario": args[0],
                "parameters": args[1],
                "payload": args[2],
            }
        ),
    )

    document = _Document()
    controller._preview_document = document
    controller._schedule_preview_completion(document, 4, scenario, parameters, future)

    assert len(document.next_tick_callbacks) == 1
    document.next_tick_callbacks[0]()

    assert captured["scenario"] is scenario
    assert captured["parameters"] is parameters
    assert captured["payload"] is payload
    assert controller._preview_future is None


def test_preview_progress_updates_frame_count_without_rendering_canvas() -> None:
    from larvaworld.portal.explore.preview_job import PreviewProgress

    controller = _ExploreController()
    job = SimpleNamespace(progress_queue=SimpleQueue())
    controller._preview_status = pn.pane.HTML("")
    controller._preview_progress = pn.indicators.Progress(value=-1, max=1)
    job.progress_queue.put(PreviewProgress("frames", 42, 600))

    controller._update_preview_progress(job)

    assert "42 / 600 frames" in controller._preview_status.object
    assert controller._preview_progress.value == 42
    assert controller._preview_progress.max == 600


def test_all_scenarios_cancels_an_active_preview() -> None:
    controller = _ExploreController()
    cancelled: list[bool] = []
    callbacks: list[object] = []

    class _Job:
        def cancel(self) -> None:
            cancelled.append(True)

    class _Future:
        def cancel(self) -> bool:
            return False

        def add_done_callback(self, callback) -> None:
            callbacks.append(callback)

    class _Document:
        def remove_periodic_callback(self, _callback) -> None:
            pass

    controller._preview_job = _Job()
    controller._preview_future = _Future()
    controller._preview_document = _Document()
    controller._preview_poll_callback = object()

    controller.show_gallery()

    assert cancelled == [True]
    assert len(callbacks) == 1
    assert controller._preview_future is None


def test_close_cancels_preview_and_shuts_down_its_executor() -> None:
    controller = _ExploreController()
    shutdown_calls: list[tuple[bool, bool]] = []

    class _Executor:
        def shutdown(self, *, wait, cancel_futures) -> None:
            shutdown_calls.append((wait, cancel_futures))

    controller._preview_executor = _Executor()

    controller.close()

    assert shutdown_calls == [(False, True)]


def test_preview_analysis_exposes_step_and_endpoint_dataset_actions() -> None:
    dataset = SimpleNamespace(config=SimpleNamespace(id="preview"))
    analysis = SimpleNamespace(metrics=[], figures=[], warnings=[])

    view = _ExploreController._analysis_view(analysis, datasets=[dataset])
    labels = [getattr(node, "name", None) for node in _walk(view)]

    assert "Step" in labels
    assert "Endpoint" in labels


def test_preview_analysis_renders_worker_png_without_a_matplotlib_pane() -> None:
    from larvaworld.portal.explore.analysis import (
        PreviewAnalysisResult,
        PreviewFigure,
    )

    analysis = PreviewAnalysisResult(
        figures=[
            PreviewFigure(
                title="Worker plot",
                png=b"\x89PNG\r\n\x1a\n",
            )
        ]
    )

    view = _ExploreController._analysis_view(analysis, datasets=[])

    assert len(view.select(pn.pane.PNG)) == 1
    assert view.select(pn.pane.Matplotlib) == []


def test_preview_dataset_actions_offer_group_selection_for_multiple_datasets() -> None:
    datasets = [
        SimpleNamespace(config=SimpleNamespace(id="rover")),
        SimpleNamespace(config=SimpleNamespace(id="sitter")),
    ]

    view = _preview_dataset_tables_view(datasets)
    selector = next(iter(view.select(pn.widgets.Select)))

    assert selector.name == "Larva group"
    assert selector.options == {"rover": 0, "sitter": 1}
    selector.value = 1


# ---- result view -----------------------------------------------------------


def _run_scenario_headless(scenario, *, step_cap=6):
    """Build and step a real scenario without a Panel document."""
    import tempfile
    from pathlib import Path

    from larvaworld.portal.canvas_widgets.environment_canvas import EnvironmentCanvas
    from larvaworld.portal.simulation.preview_frames import generate_preview_frames
    from larvaworld.portal.simulation.run_playback import build_bounded_launcher

    parameters = _ExploreController._apply_scenario_overrides(
        util.AttrDict(reg.conf.Exp.expand(scenario.exp_id)), scenario
    )
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td) / scenario.id
        run_dir.mkdir()
        launcher, _note = build_bounded_launcher(
            scenario.exp_id, parameters, run_dir, step_cap=step_cap
        )
        canvas = EnvironmentCanvas(editable=False)
        frames = generate_preview_frames(launcher, preview_steps=step_cap)
    return canvas, frames


def test_result_view_uses_distinct_observation_cues_and_offers_a_way_forward() -> None:
    controller = _ExploreController()
    scenario = scenario_by_id("dish")
    canvas, frames = _run_scenario_headless(scenario)

    controller.show_result(scenario, canvas, frames, dt=0.1, note=None)
    text = " ".join(str(getattr(n, "object", "")) for n in _walk(controller.view()))
    observation = getattr(controller.body.objects[4], "object", "")

    assert "What you just saw" in text
    assert scenario.explanation[:40] in text
    assert scenario.watch_for[0] in text
    assert escape(" ".join(scenario.explanation.split())) not in observation
    # The graduation path into the full editor must be present.
    assert "/wf.run_experiment" in text
    labels = [getattr(n, "name", None) for n in _walk(controller.view())]
    assert "Run it again" in labels
    assert "< All scenarios" in labels


def test_result_view_cites_literature_when_the_scenario_has_it() -> None:
    controller = _ExploreController()
    scenario = next(s for s in SCENARIOS if s.literature)
    canvas, frames = _run_scenario_headless(scenario)

    controller.show_result(scenario, canvas, frames, dt=0.1, note=None)
    text = " ".join(str(getattr(n, "object", "")) for n in _walk(controller.view()))

    assert scenario.literature in text


def test_result_view_surfaces_a_fallback_note() -> None:
    controller = _ExploreController()
    scenario = scenario_by_id("dish")
    canvas, frames = _run_scenario_headless(scenario)

    controller.show_result(
        scenario, canvas, frames, dt=0.1, note="Overlap elimination was disabled."
    )
    text = " ".join(str(getattr(n, "object", "")) for n in _walk(controller.view()))

    assert "Overlap elimination was disabled." in text


def test_returning_to_the_gallery_releases_run_resources() -> None:
    controller = _ExploreController()
    scenario = scenario_by_id("dish")
    canvas, frames = _run_scenario_headless(scenario)
    controller.show_result(scenario, canvas, frames, dt=0.1, note=None)

    controller.show_gallery()

    assert controller._temp_dir is None
    labels = [getattr(n, "name", None) for n in _walk(controller.view())]
    assert labels.count("Watch") == len(SCENARIOS)
