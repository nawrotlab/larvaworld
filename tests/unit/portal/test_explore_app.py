"""Tests for the Explore app: the zero-configuration entry point."""

from __future__ import annotations

from html import escape
from types import SimpleNamespace

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


def test_preview_request_shows_preparation_message_before_generation(
    monkeypatch,
) -> None:
    controller = _ExploreController()
    scenario = scenario_by_id("dish")
    parameters = util.AttrDict(reg.conf.Exp.expand(scenario.exp_id))
    generated: list[tuple[object, object]] = []
    monkeypatch.setattr(
        controller,
        "_generate_simulation_preview",
        lambda requested_scenario, requested_parameters: generated.append(
            (requested_scenario, requested_parameters)
        ),
    )

    controller._request_simulation_preview(scenario, parameters)

    text = " ".join(
        str(getattr(node, "object", "")) for node in _walk(controller.view())
    )
    assert (
        "Generating simulation preview. The environment and agents are being initialized."
        in text
    )
    assert generated == [(scenario, parameters)]


def test_preview_generation_collects_all_frames_before_showing_result(
    monkeypatch,
) -> None:
    from larvaworld.portal.canvas_widgets.environment_models import LarvaPreviewFrame

    controller = _ExploreController()
    scenario = scenario_by_id("dish")
    parameters = _ExploreController._apply_scenario_overrides(
        util.AttrDict(reg.conf.Exp.expand(scenario.exp_id)), scenario
    )
    frames = [LarvaPreviewFrame(tick=0, centroids=((0.0, 0.0),))]
    captured: dict[str, object] = {}

    class _Launcher:
        dt = 0.1
        screen_manager = None

    class _Canvas:
        def __init__(self, **kwargs) -> None:
            captured["canvas_options"] = kwargs

        def set_state(self, _state) -> None:
            pass

    datasets = [SimpleNamespace(config=SimpleNamespace(id="preview"))]
    analysis = SimpleNamespace(metrics=[], figures=[], warnings=[])

    monkeypatch.setattr(
        "larvaworld.portal.simulation.run_playback.build_bounded_launcher",
        lambda *args, **kwargs: (_Launcher(), None),
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.preview_frames.generate_preview_frames",
        lambda launcher, *, preview_steps: captured.setdefault(
            "preview_steps", preview_steps
        )
        and frames,
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.run_playback.finalize_preview_datasets",
        lambda _launcher: datasets,
    )
    monkeypatch.setattr(
        "larvaworld.portal.explore.analysis.build_preview_analysis",
        lambda _scenario_id, received_datasets: analysis,
    )
    monkeypatch.setattr(
        "larvaworld.portal.canvas_widgets.environment_canvas.EnvironmentCanvas",
        _Canvas,
    )
    monkeypatch.setattr(
        controller,
        "show_result",
        lambda *args, **kwargs: captured.update(
            {
                "scenario": args[0],
                "frames": args[2],
                "dt": kwargs["dt"],
                "datasets": kwargs["datasets"],
            }
        ),
    )

    controller._generate_simulation_preview(scenario, parameters)

    assert captured["preview_steps"] == scenario.step_cap
    assert captured["scenario"] is scenario
    assert captured["frames"] == frames
    assert captured["dt"] == pytest.approx(0.1)
    assert captured["datasets"] == datasets
    assert captured["canvas_options"] == {
        "editable": False,
        "show_larva_groups": False,
    }


def test_preview_analysis_exposes_step_and_endpoint_dataset_actions() -> None:
    dataset = SimpleNamespace(config=SimpleNamespace(id="preview"))
    analysis = SimpleNamespace(metrics=[], figures=[], warnings=[])

    view = _ExploreController._analysis_view(analysis, datasets=[dataset])
    labels = [getattr(node, "name", None) for node in _walk(view)]

    assert "Step" in labels
    assert "Endpoint" in labels


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
