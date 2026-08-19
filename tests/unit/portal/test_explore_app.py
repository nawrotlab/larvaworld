"""Tests for the Explore app: the zero-configuration entry point."""

from __future__ import annotations

import pytest

from larvaworld.lib import reg, util
from larvaworld.portal.explore.explore_app import (
    MAX_AGENTS,
    _ExploreController,
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
    # No run has started, so nothing is holding a launcher or temp directory.
    assert controller._runner is None
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


# ---- result view -----------------------------------------------------------


def _run_scenario_headless(scenario, *, step_cap=6):
    """Build and step a real scenario without a Panel document."""
    import tempfile
    from pathlib import Path

    from larvaworld.portal.canvas_widgets.environment_canvas import EnvironmentCanvas
    from larvaworld.portal.simulation.run_playback import (
        ChunkedFrameRunner,
        build_bounded_launcher,
    )

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
        runner = ChunkedFrameRunner(
            launcher, total_steps=step_cap, on_frame=lambda _f: None
        )
        frames = runner.run_to_completion()
    return canvas, frames


def test_result_view_explains_the_behavior_and_offers_a_way_forward() -> None:
    controller = _ExploreController()
    scenario = scenario_by_id("dish")
    canvas, frames = _run_scenario_headless(scenario)

    controller.show_result(scenario, canvas, frames, dt=0.1, note=None)
    text = " ".join(str(getattr(n, "object", "")) for n in _walk(controller.view()))

    assert "What you just saw" in text
    assert scenario.explanation[:40] in text
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

    assert controller._runner is None
    assert controller._temp_dir is None
    labels = [getattr(n, "name", None) for n in _walk(controller.view())]
    assert labels.count("Watch") == len(SCENARIOS)
