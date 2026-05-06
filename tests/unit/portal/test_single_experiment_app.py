from __future__ import annotations

import json
from pathlib import Path

import holoviews as hv
import numpy as np
import panel as pn
import pytest

from larvaworld.lib import reg, util
from larvaworld.lib.reg.larvagroup import LarvaGroup
from larvaworld.portal.canvas_widgets.environment_models import (
    CanvasArena,
    EnvironmentCanvasState,
    LarvaPreviewFrame,
)
from larvaworld.portal.landing_registry import ITEMS
from larvaworld.portal.simulation.single_experiment_app import (
    _ExperimentPreview,
    _FrameSimulationPreview,
    _SingleExperimentController,
    _builder_obstacle_border_vertices,
    _default_run_name,
    _display_label_for_path,
    _family_spec_for_path,
    _safe_slug,
)
from larvaworld.portal.workspace import (
    clear_active_workspace_path,
    initialize_workspace,
    set_active_workspace_path,
)

SINGLE_EXPERIMENT_APP_INCOMPLETE_REASON = (
    "single_experiment_app is incomplete; re-enable after app stabilization"
)

hv.extension("bokeh")


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    clear_active_workspace_path()


def test_single_experiment_lists_workspace_environment_presets(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    (workspace_root / "environments" / "dish_custom.json").write_text(
        json.dumps({"arena": {"geometry": "circular", "dims": [0.12, 0.12]}}) + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()

    assert (
        controller.environment_select.options["Template default environment"]
        == "__template__"
    )
    assert (
        controller.environment_select.options["Registry / dish"] == "__registry__:dish"
    )
    assert controller.environment_select.options["dish_custom"] == "dish_custom.json"


def test_single_experiment_lists_and_loads_registry_environment_presets(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()

    assert "Registry / maze" in controller.environment_select.options
    controller.experiment.value = "dish"
    controller.environment_select.value = controller.environment_select.options[
        "Registry / maze"
    ]

    parameters = controller._build_parameters()

    assert parameters.env_params.arena.geometry == "rectangular"
    assert "Maze" in parameters.env_params.border_list
    assert controller._selected_environment_label() == "registry / maze"


def test_single_experiment_template_change_auto_refreshes_environment_presets(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    current_template = str(controller.experiment.value)
    env_ids = set(str(env_id) for env_id in reg.conf.Env.confIDs)
    candidate = next(
        (
            exp_id
            for exp_id in (str(exp_id) for exp_id in reg.conf.Exp.confIDs)
            if exp_id in env_ids and exp_id != current_template
        ),
        None,
    )
    if candidate is None:
        pytest.skip("No experiment template with matching environment ID available")

    controller.experiment.value = candidate

    assert (
        controller.environment_select.options["Template default environment"]
        == "__template__"
    )
    assert (
        controller.environment_select.options[f"Registry / {candidate}"]
        == f"__registry__:{candidate}"
    )


def test_single_experiment_build_parameters_applies_environment_override(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    (workspace_root / "environments" / "rect_env.json").write_text(
        json.dumps(
            {
                "arena": {"geometry": "rectangular", "dims": [0.2, 0.1]},
                "food_params": {
                    "source_units": {
                        "patch": {
                            "pos": [0.02, 0.0],
                            "radius": 0.005,
                            "amount": 2.0,
                            "odor": {"id": "apple", "intensity": 1.0, "spread": 0.02},
                            "substrate": {"type": "standard", "quality": 1.0},
                            "color": "#44aa55",
                        }
                    },
                    "source_groups": {},
                    "food_grid": {},
                },
                "border_list": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    controller.experiment.value = "dish"
    controller.environment_select.value = "rect_env.json"
    controller._parameter_widgets["duration"][1].value = 1.5

    parameters = controller._build_parameters()

    assert parameters.duration == pytest.approx(1.5)
    assert parameters.env_params.arena.geometry == "rectangular"
    assert parameters.env_params.arena.dims == (0.2, 0.1)
    assert "patch" in parameters.env_params.food_params.source_units
    assert parameters.env_params.food_params.source_units["patch"]["pos"] == (0.02, 0.0)
    assert isinstance(parameters, util.AttrDict)


def test_single_experiment_configuration_preview_does_not_call_exp_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    captured: dict[str, object] = {}
    canvas_view = pn.pane.HTML("canvas-view")

    class DummyCanvas:
        def __init__(self, *, editable=False):
            captured["editable"] = editable

        def set_state(self, state):
            captured["state"] = state

        def view(self):
            return canvas_view

    def fail_exp_run(**kwargs):
        raise AssertionError("configuration preview should not instantiate ExpRun")

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.EnvironmentCanvas",
        DummyCanvas,
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.sim.ExpRun",
        fail_exp_run,
    )

    controller = _SingleExperimentController()
    controller.run_name.value = "dish_config_preview"
    controller._on_prepare_preview()

    assert captured["editable"] is False
    assert "state" in captured
    assert controller.preview[0] is canvas_view
    assert "No simulation has been run" in controller.status.object


def test_single_experiment_configuration_preview_uses_mapped_canvas_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    mapped_state = EnvironmentCanvasState(arena=CanvasArena("rectangular", (0.2, 0.1)))
    captured: dict[str, object] = {}
    canvas_view = pn.pane.HTML("canvas-view")

    class DummyCanvas:
        def __init__(self, *, editable=False):
            pass

        def set_state(self, state):
            captured["state"] = state

        def view(self):
            return canvas_view

    def fake_mapping(env_params, *, larva_groups=None, show_group_shapes=True):
        captured["env_params"] = env_params
        captured["larva_groups"] = larva_groups
        captured["show_group_shapes"] = show_group_shapes
        return mapped_state

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.EnvironmentCanvas",
        DummyCanvas,
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.env_params_to_canvas_state",
        fake_mapping,
    )

    controller = _SingleExperimentController()
    controller._on_prepare_preview()

    assert captured["state"] is mapped_state
    assert captured["env_params"] is not None
    assert captured["larva_groups"] is not None
    assert captured["show_group_shapes"] is False
    assert controller.preview[0] is canvas_view


def test_single_experiment_prepare_preview_uses_resolved_parameters_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    resolved = controller._build_parameters()

    def fail_build_parameters():
        raise AssertionError("_build_parameters should not be called directly")

    class DummyCanvas:
        def __init__(self, *, editable=False):
            pass

        def set_state(self, state):
            self.state = state

        def view(self):
            return pn.pane.HTML("canvas-view")

    monkeypatch.setattr(controller, "_build_parameters", fail_build_parameters)
    monkeypatch.setattr(controller, "_resolve_experiment_parameters", lambda: resolved)
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.EnvironmentCanvas",
        DummyCanvas,
    )

    controller._on_prepare_preview()
    assert isinstance(controller.preview[0], pn.pane.HTML)


def test_single_experiment_configuration_preview_uses_environment_preset_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    (workspace_root / "environments" / "rect_env.json").write_text(
        json.dumps({"arena": {"geometry": "rectangular", "dims": [0.2, 0.1]}}) + "\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    class DummyCanvas:
        def __init__(self, *, editable=False):
            pass

        def set_state(self, state):
            captured["state"] = state

        def view(self):
            return "canvas-view"

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.EnvironmentCanvas",
        DummyCanvas,
    )

    controller = _SingleExperimentController()
    controller.environment_select.value = "rect_env.json"
    controller._on_prepare_preview()

    state = captured["state"]
    assert state.arena.geometry == "rectangular"
    assert state.arena.dims == (0.2, 0.1)


def test_single_experiment_builder_obstacles_are_translated_into_border_entries(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    (workspace_root / "environments" / "builder_scene.json").write_text(
        json.dumps(
            {
                "arena": {"geometry": "circular", "dims": [0.24, 0.24]},
                "food_params": {
                    "source_units": {},
                    "source_groups": {},
                    "food_grid": {},
                },
                "border_list": {},
                "obstacles": {
                    "obstacle_1": {
                        "pos": [0.03, -0.01],
                        "radius": 0.02,
                        "color": "#ff8800",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    controller.environment_select.value = "builder_scene.json"

    parameters = controller._build_parameters()
    border_list = parameters.env_params.border_list

    assert len(_builder_obstacle_border_vertices((0.0, 0.0), 0.01)) == 36
    assert "Obstacle_obstacle_1" in border_list
    assert border_list["Obstacle_obstacle_1"]["color"] == "#ff8800"
    assert len(border_list["Obstacle_obstacle_1"]["vertices"]) == 36
    assert isinstance(border_list["Obstacle_obstacle_1"]["vertices"][0], tuple)


class _DummyPreviewAgents(list):
    head = type("Head", (), {"front_end": []})()

    def get_position(self):
        return []


class _DummyPreviewLauncher:
    def __init__(
        self,
        *,
        agents: _DummyPreviewAgents | None = None,
        sources: list[object] | None = None,
        borders: list[object] | None = None,
        t: int = 0,
        nsteps: int = 8,
        step_hook=None,
    ) -> None:
        self.p = util.AttrDict(
            {
                "steps": nsteps,
                "env_params": util.AttrDict(
                    {
                        "arena": util.AttrDict(
                            {"geometry": "circular", "dims": (0.2, 0.2)}
                        )
                    }
                ),
            }
        )
        self.dt = 0.1
        self.Nsteps = nsteps
        self.t = t
        self.agents = agents if agents is not None else _DummyPreviewAgents()
        self.sources = sources if sources is not None else []
        self.borders = borders if borders is not None else []
        self.step_calls = 0
        self._step_hook = step_hook

    def sim_step(self) -> None:
        self.step_calls += 1
        if self._step_hook is not None:
            self._step_hook()
            return
        self.t += 1


class _ProgressRecorder:
    def __init__(self) -> None:
        self.updates: list[int] = []

    @property
    def value(self) -> int:
        if not self.updates:
            return 0
        return self.updates[-1]

    @value.setter
    def value(self, value: int) -> None:
        self.updates.append(value)


def _disable_dynamic_preview_layers(preview: _ExperimentPreview) -> None:
    preview.draw_ops.draw_centroid = False
    preview.draw_ops.draw_head = False
    preview.draw_ops.draw_midline = False
    preview.draw_ops.visible_trails = False
    preview.draw_ops.draw_segs = False


def test_single_experiment_preview_overlay_draws_sources_odors_and_borders() -> None:
    source = util.AttrDict(
        {
            "pos": (0.02, 0.01),
            "radius": 0.01,
            "color": "#44aa55",
            "odor": util.AttrDict({"id": "apple", "spread": 0.03}),
        }
    )
    border = util.AttrDict(
        {
            "width": 0.002,
            "color": "#cc3344",
            "border_xy": [np.array([[0.0, 0.0], [0.04, 0.0]])],
        }
    )
    launcher = _DummyPreviewLauncher(sources=[source], borders=[border])

    preview = _ExperimentPreview(launcher, launcher_ready=True)
    _disable_dynamic_preview_layers(preview)

    overlay = preview._draw_overlay()

    assert len(overlay) == 4


def test_single_experiment_preview_disabled_layers_skip_expensive_agent_attrs() -> None:
    class ExplodingAgent:
        color = "#111111"

        @property
        def segs(self):
            raise AssertionError("segs should not be accessed")

        @property
        def midline_xy(self):
            raise AssertionError("midline_xy should not be accessed")

        @property
        def trajectory(self):
            raise AssertionError("trajectory should not be accessed")

    launcher = _DummyPreviewLauncher(agents=_DummyPreviewAgents([ExplodingAgent()]))
    preview = _ExperimentPreview(launcher, launcher_ready=True)
    _disable_dynamic_preview_layers(preview)

    preview._draw_overlay()


def test_single_experiment_preview_static_layers_are_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"source": 0, "odor": 0, "border": 0}

    def source_layers(self):
        calls["source"] += 1
        return []

    def odor_layers(self):
        calls["odor"] += 1
        return []

    def border_layers(self):
        calls["border"] += 1
        return []

    monkeypatch.setattr(_ExperimentPreview, "_source_layers", source_layers)
    monkeypatch.setattr(_ExperimentPreview, "_odor_layers", odor_layers)
    monkeypatch.setattr(_ExperimentPreview, "_border_layers", border_layers)

    preview = _ExperimentPreview(_DummyPreviewLauncher(), launcher_ready=True)
    _disable_dynamic_preview_layers(preview)

    preview._draw_overlay()
    preview._draw_overlay()

    assert calls == {"source": 1, "odor": 1, "border": 1}


def test_single_experiment_preview_image_for_tick_batches_progress_update_once() -> (
    None
):
    launcher = _DummyPreviewLauncher(t=0, nsteps=8)
    preview = _ExperimentPreview(launcher, launcher_ready=True)
    _disable_dynamic_preview_layers(preview)
    progress = _ProgressRecorder()
    preview.progress_bar = progress

    preview._image_for_tick(3)

    assert launcher.t == 3
    assert launcher.step_calls == 3
    assert progress.updates == [3]


def test_single_experiment_preview_current_tick_only_redraws() -> None:
    def fail_step() -> None:
        raise AssertionError("sim_step should not be called for the current tick")

    launcher = _DummyPreviewLauncher(t=2, nsteps=8, step_hook=fail_step)
    preview = _ExperimentPreview(launcher, launcher_ready=True)
    _disable_dynamic_preview_layers(preview)
    preview.forward_only_note.object = _ExperimentPreview._FORWARD_ONLY_MESSAGE
    progress = _ProgressRecorder()
    preview.progress_bar = progress

    preview._image_for_tick(2)

    assert launcher.step_calls == 0
    assert preview.forward_only_note.object == ""
    assert progress.updates == []


def test_single_experiment_preview_backward_seek_syncs_to_current_tick() -> None:
    def fail_step() -> None:
        raise AssertionError("sim_step should not be called on backward seek")

    launcher = _DummyPreviewLauncher(t=4, nsteps=8, step_hook=fail_step)
    preview = _ExperimentPreview(launcher, launcher_ready=True)
    _disable_dynamic_preview_layers(preview)
    progress = _ProgressRecorder()
    preview.progress_bar = progress

    preview._image_for_tick(1)

    assert launcher.step_calls == 0
    assert preview.time_slider.value == 4
    assert "forward-only" in preview.forward_only_note.object
    assert progress.updates == [4]


@pytest.mark.skip(reason=SINGLE_EXPERIMENT_APP_INCOMPLETE_REASON)
def test_single_experiment_preview_metadata_summarizes_applied_settings(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.experiment.value = "dish"
    controller._on_experiment_change()
    population_path = next(
        path
        for path in controller._parameter_widgets
        if path.endswith(".distribution.N")
    )
    controller._parameter_widgets["duration"][1].value = 2.0
    controller._parameter_widgets[population_path][1].value = 5

    html = controller._preview_metadata_html(
        controller._build_parameters(), "template default"
    )

    assert "Applied preview config" in html
    assert "duration = 2.00 min" in html
    assert "larvae = 5" in html


def test_single_experiment_resolved_plan_payload_serializes_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    parameters = controller._build_parameters()

    payload = controller._resolved_plan_payload(
        experiment="dish",
        run_name="dish_demo",
        selected_env="template default",
        parameters=parameters,
    )

    assert payload["experiment"] == "dish"
    assert payload["run_name"] == "dish_demo"
    assert payload["selected_environment"] == "template default"
    assert payload["parameters"]["env_params"]["arena"]["geometry"] == "circular"


def test_single_experiment_preview_runtime_parameters_strip_preview_only_overhead(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    parameters = controller._build_parameters()
    preview_parameters = controller._preview_runtime_parameters(parameters)

    assert preview_parameters.collections == []
    assert preview_parameters.enrichment is None


def test_single_experiment_run_experiment_writes_plan_and_reports_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    simulated = {}

    class DummyRun:
        def __init__(self, **kwargs):
            simulated.update(kwargs)

        def simulate(self):
            return ["dataset_a", "dataset_b"]

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.sim.ExpRun",
        DummyRun,
    )

    controller = _SingleExperimentController()
    controller.run_name.value = "dish_demo"
    controller._on_run_experiment()

    run_dir = workspace_root / "experiments" / "dish_demo"
    plan_path = run_dir / "resolved_experiment.json"

    assert simulated["store_data"] is True
    assert simulated["id"] == "dish_demo"
    assert simulated["dir"] == str(run_dir)
    assert plan_path.exists()
    assert "Stored outputs in" in controller.status.object
    assert "resolved_experiment.json" in controller.status.object


@pytest.mark.skip(reason=SINGLE_EXPERIMENT_APP_INCOMPLETE_REASON)
def test_single_experiment_run_experiment_shows_running_state_before_deferred_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    captured: dict[str, object] = {}
    callbacks: list[object] = []

    class DummyDoc:
        def add_next_tick_callback(self, callback):
            callbacks.append(callback)

    class DummyRun:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def simulate(self):
            return ["dataset_a"]

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.sim.ExpRun",
        DummyRun,
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.pn.state",
        "curdoc",
        DummyDoc(),
        raising=False,
    )

    controller = _SingleExperimentController()
    controller.run_name.value = "dish_demo"
    controller._on_run_experiment()

    assert 'Running "dish"' in controller.status.object
    assert (
        "UI will be unresponsive until the simulation finishes"
        in controller.status.object
    )
    assert "writing outputs to" in controller.preview[0].object
    assert controller.prepare_btn.disabled is True
    assert controller.run_btn.disabled is True
    assert len(callbacks) == 1

    callbacks[0]()

    assert captured["store_data"] is True
    assert "Completed run" in controller.status.object
    assert controller.prepare_btn.disabled is False
    assert controller.run_btn.disabled is False


def test_single_experiment_runtime_screen_kws_include_video_when_enabled(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.save_video.value = True
    controller.video_filename.value = "dish_video"
    controller.video_fps.value = 3
    controller.show_display.value = True
    controller.display_every_n_steps.value = 4

    kws = controller._runtime_screen_kws(workspace_root / "experiments" / "dish_demo")

    assert controller.video_fps.name == "Video speed-up"
    assert kws["save_video"] is True
    assert kws["vis_mode"] == "video"
    assert kws["video_file"] == "dish_video"
    assert kws["fps"] == 3
    assert kws["show_display"] is True
    assert kws["display_every_n_steps"] == 4
    assert kws["media_dir"] == str(workspace_root / "experiments" / "dish_demo")


def test_single_experiment_show_display_uses_video_render_mode(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.show_display.value = True
    controller.display_every_n_steps.value = 3

    kws = controller._runtime_screen_kws(workspace_root / "experiments" / "dish_demo")

    assert kws["show_display"] is True
    assert kws["vis_mode"] == "video"
    assert kws["display_every_n_steps"] == 3
    assert "save_video" not in kws


def test_single_experiment_run_experiment_passes_video_screen_kws(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    captured = {}

    class DummyRun:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def simulate(self):
            return ["dataset_a"]

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.sim.ExpRun",
        DummyRun,
    )

    controller = _SingleExperimentController()
    controller.run_name.value = "dish_demo"
    controller.save_video.value = True
    controller.video_filename.value = "dish_capture"
    controller.video_fps.value = 5
    controller.display_every_n_steps.value = 6
    controller._on_run_experiment()

    assert captured["screen_kws"]["save_video"] is True
    assert captured["screen_kws"]["video_file"] == "dish_capture"
    assert captured["screen_kws"]["fps"] == 5
    assert captured["screen_kws"]["display_every_n_steps"] == 6
    assert "dish_capture.mp4" in controller.status.object


def test_single_experiment_video_speed_up_defaults_to_realtime() -> None:
    controller = _SingleExperimentController()

    assert controller.video_fps.name == "Video speed-up"
    assert controller.video_fps.value == 1
    assert "simulated real time" in controller.video_fps.description
    assert "twice as fast" in controller.video_fps.description
    assert "one fifth as long" in controller.video_fps.description


def test_single_experiment_display_every_n_steps_follows_show_display() -> None:
    controller = _SingleExperimentController()

    assert controller.display_every_n_steps.value == 1
    assert controller.display_every_n_steps.disabled is True
    controller.show_display.value = True
    assert controller.display_every_n_steps.disabled is False
    controller.show_display.value = False
    assert controller.display_every_n_steps.disabled is True


def test_single_experiment_run_experiment_explicitly_closes_screen_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    closed = {"count": 0}

    class DummyScreenManager:
        def close(self):
            closed["count"] += 1

    class DummyRun:
        def __init__(self, **kwargs):
            self.screen_manager = DummyScreenManager()

        def simulate(self):
            return ["dataset_a"]

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.sim.ExpRun",
        DummyRun,
    )

    controller = _SingleExperimentController()
    controller.run_name.value = "dish_demo"
    controller._execute_run_experiment(
        parameters=controller._build_parameters(),
        run_dir=workspace_root / "experiments" / "dish_demo",
        selected_env="template default",
    )

    assert closed["count"] == 1
    assert "Completed run" in controller.status.object


def test_single_experiment_run_experiment_uses_resolved_parameters_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    resolved = controller._build_parameters()
    captured: dict[str, object] = {}

    def fail_build_parameters():
        raise AssertionError("_build_parameters should not be called directly")

    def fake_execute(self, *, parameters, run_dir, selected_env):
        captured["parameters"] = parameters
        captured["run_dir"] = run_dir
        captured["selected_env"] = selected_env

    monkeypatch.setattr(controller, "_build_parameters", fail_build_parameters)
    monkeypatch.setattr(controller, "_resolve_experiment_parameters", lambda: resolved)
    monkeypatch.setattr(
        _SingleExperimentController,
        "_execute_run_experiment",
        fake_execute,
    )

    controller.run_name.value = "dish_boundary"
    controller._on_run_experiment()

    assert captured["parameters"] is resolved
    assert str(captured["run_dir"]).endswith("dish_boundary")


def test_single_experiment_preview_status_uses_reserved_run_directory_wording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    class DummyPreviewLauncher:
        def __init__(self):
            self.p = util.AttrDict({"steps": 9})
            self.dt = 0.1

    class DummyCanvas:
        def __init__(self, *, editable=False):
            self.editable = editable
            self.state = None
            self.frames: list[LarvaPreviewFrame] = []

        def set_state(self, state):
            self.state = state

        def set_larva_frame(self, frame):
            self.frames.append(frame)

        def view(self):
            return pn.pane.HTML("canvas")

    frame0 = LarvaPreviewFrame(tick=0, centroids=((0.0, 0.0),))
    frame1 = LarvaPreviewFrame(tick=1, centroids=((0.01, 0.01),))

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app._SingleExperimentController._prepare_preview_launcher",
        lambda self, experiment, parameters, run_dir: (DummyPreviewLauncher(), None),
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.generate_preview_frames",
        lambda launcher, preview_steps, **kwargs: [frame0, frame1],
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.EnvironmentCanvas",
        DummyCanvas,
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app._ExperimentPreview",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "_ExperimentPreview should not be used in default preview path"
            )
        ),
    )

    controller = _SingleExperimentController()
    controller.run_name.value = "dish_preview"
    controller._on_generate_simulation_preview()

    assert "Reserved output directory for a future run" in controller.status.object
    assert "Simulation preview ready: 2 frames generated." in controller.status.object
    assert "Displayed range: 0.0-0.1 s simulated time." in controller.status.object
    assert "Outputs are not stored" in controller.status.object


def test_single_experiment_generate_preview_uses_resolved_parameters_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    resolved = controller._build_parameters()

    def fail_build_parameters():
        raise AssertionError("_build_parameters should not be called directly")

    class DummyPreviewLauncher:
        def __init__(self):
            self.p = util.AttrDict({"steps": 3})
            self.dt = 0.1

    class DummyCanvas:
        def __init__(self, *, editable=False):
            pass

        def set_state(self, state):
            self.state = state

        def set_larva_frame(self, frame):
            self.frame = frame

        def view(self):
            return pn.pane.HTML("canvas")

    monkeypatch.setattr(controller, "_build_parameters", fail_build_parameters)
    monkeypatch.setattr(controller, "_resolve_experiment_parameters", lambda: resolved)
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app._SingleExperimentController._prepare_preview_launcher",
        lambda self, experiment, parameters, run_dir: (DummyPreviewLauncher(), None),
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.generate_preview_frames",
        lambda launcher, preview_steps, **kwargs: [
            LarvaPreviewFrame(tick=0, centroids=((0.0, 0.0),))
        ],
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.EnvironmentCanvas",
        DummyCanvas,
    )

    controller._on_generate_simulation_preview()


def test_frame_simulation_preview_player_is_random_access() -> None:
    class DummyCanvas:
        def __init__(self):
            self.applied_ticks: list[int] = []

        def set_larva_frame(self, frame: LarvaPreviewFrame) -> None:
            self.applied_ticks.append(frame.tick)

        def view(self):
            return pn.pane.HTML("canvas")

    canvas = DummyCanvas()
    preview = _FrameSimulationPreview(
        canvas=canvas,
        frames=[
            LarvaPreviewFrame(tick=0, centroids=((0.0, 0.0),)),
            LarvaPreviewFrame(tick=1, centroids=((0.01, 0.01),)),
            LarvaPreviewFrame(tick=2, centroids=((0.02, 0.02),)),
        ],
        dt=0.1,
    )

    assert canvas.applied_ticks == [0]
    preview.frame_player.value = 2
    preview.frame_player.value = 1

    assert canvas.applied_ticks == [0, 2, 1]
    assert "Frame:</strong> 1/2" in preview.metadata.object
    assert "Tick:</strong> 1" in preview.metadata.object


def test_frame_simulation_preview_show_frame_clamps_indices() -> None:
    class DummyCanvas:
        def __init__(self):
            self.applied_ticks: list[int] = []

        def set_larva_frame(self, frame: LarvaPreviewFrame) -> None:
            self.applied_ticks.append(frame.tick)

        def view(self):
            return pn.pane.HTML("canvas")

    canvas = DummyCanvas()
    preview = _FrameSimulationPreview(
        canvas=canvas,
        frames=[
            LarvaPreviewFrame(tick=0, centroids=((0.0, 0.0),)),
            LarvaPreviewFrame(tick=1, centroids=((0.01, 0.01),)),
            LarvaPreviewFrame(tick=2, centroids=((0.02, 0.02),)),
        ],
        dt=0.1,
    )

    preview._show_frame(-1)
    preview._show_frame(999)

    assert canvas.applied_ticks == [0, 0, 2]
    assert int(preview.frame_player.value) == 2
    assert "Frame:</strong> 2/2" in preview.metadata.object
    assert "Tick:</strong> 2" in preview.metadata.object


def test_generate_preview_uses_requested_preview_frame_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    captured: dict[str, object] = {}

    class DummyPreviewLauncher:
        def __init__(self):
            self.p = util.AttrDict({"steps": 500})
            self.dt = 0.1

    class DummyCanvas:
        def __init__(self, *, editable=False):
            pass

        def set_state(self, state):
            captured["state"] = state

        def set_larva_frame(self, frame):
            captured["frame"] = frame

        def view(self):
            return pn.pane.HTML("canvas")

    def fake_generate_preview_frames(launcher, preview_steps, **kwargs):
        captured["preview_steps"] = preview_steps
        return [LarvaPreviewFrame(tick=0, centroids=((0.0, 0.0),))]

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app._SingleExperimentController._prepare_preview_launcher",
        lambda self, experiment, parameters, run_dir: (DummyPreviewLauncher(), None),
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.generate_preview_frames",
        fake_generate_preview_frames,
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.EnvironmentCanvas",
        DummyCanvas,
    )

    controller = _SingleExperimentController()
    controller.preview_frames_input.value = 120
    controller._on_generate_simulation_preview()

    assert captured["preview_steps"] == 120


def test_generate_preview_caps_requested_frames_by_launcher_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    captured: dict[str, object] = {}

    class DummyPreviewLauncher:
        def __init__(self):
            self.p = util.AttrDict({"steps": 7})
            self.dt = 0.1

    class DummyCanvas:
        def __init__(self, *, editable=False):
            pass

        def set_state(self, state):
            captured["state"] = state

        def set_larva_frame(self, frame):
            captured["frame"] = frame

        def view(self):
            return pn.pane.HTML("canvas")

    def fake_generate_preview_frames(launcher, preview_steps, **kwargs):
        captured["preview_steps"] = preview_steps
        return [LarvaPreviewFrame(tick=0, centroids=((0.0, 0.0),))]

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app._SingleExperimentController._prepare_preview_launcher",
        lambda self, experiment, parameters, run_dir: (DummyPreviewLauncher(), None),
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.generate_preview_frames",
        fake_generate_preview_frames,
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.EnvironmentCanvas",
        DummyCanvas,
    )

    controller = _SingleExperimentController()
    controller.preview_frames_input.value = 120
    controller._on_generate_simulation_preview()

    assert captured["preview_steps"] == 7


def test_single_experiment_action_rows_separate_preview_and_execution() -> None:
    controller = _SingleExperimentController()

    assert controller.prepare_btn in controller.preview_action_row.objects
    assert (
        controller.simulation_preview_btn not in controller.preview_action_row.objects
    )
    assert controller.run_btn not in controller.preview_action_row.objects
    assert controller.preview_frames_input in controller.preview_options_row.objects
    assert controller.simulation_preview_btn in controller.preview_generate_row.objects
    assert controller.run_btn in controller.execution_action_row.objects


def test_generate_preview_uses_show_group_shapes_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    captured: dict[str, object] = {}

    class DummyPreviewLauncher:
        def __init__(self):
            self.p = util.AttrDict({"steps": 1})
            self.dt = 0.1

    class DummyCanvas:
        def __init__(self, *, editable=False):
            pass

        def set_state(self, state):
            captured["state"] = state

        def set_larva_frame(self, frame):
            captured["frame"] = frame

        def view(self):
            return pn.pane.HTML("canvas")

    def fake_mapping(env_params, *, larva_groups=None, show_group_shapes=True):
        captured["larva_groups"] = larva_groups
        captured["show_group_shapes"] = show_group_shapes
        return EnvironmentCanvasState(arena=CanvasArena("rectangular", (0.1, 0.1)))

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app._SingleExperimentController._prepare_preview_launcher",
        lambda self, experiment, parameters, run_dir: (DummyPreviewLauncher(), None),
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.generate_preview_frames",
        lambda launcher, preview_steps, **kwargs: [
            LarvaPreviewFrame(tick=0, centroids=((0.0, 0.0),))
        ],
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.EnvironmentCanvas",
        DummyCanvas,
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.env_params_to_canvas_state",
        fake_mapping,
    )

    controller = _SingleExperimentController()
    controller._on_generate_simulation_preview()

    assert captured["larva_groups"] is None
    assert captured["show_group_shapes"] is False


def test_single_experiment_preview_launcher_falls_back_when_overlap_elimination_breaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    parameters = controller._build_parameters()
    parameters["larva_collisions"] = False
    run_dir = workspace_root / "experiments" / "preview_case"

    calls: list[bool] = []

    class DummyLauncher:
        def __init__(self, parameters):
            self.parameters = parameters

        def sim_setup(self, steps):
            calls.append(bool(self.parameters.larva_collisions))
            if not self.parameters.larva_collisions:
                raise AttributeError("has no attribute 'get_polygon'")

    def fake_exp_run(*, parameters, **kwargs):
        return DummyLauncher(parameters)

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.sim.ExpRun",
        fake_exp_run,
    )

    launcher, fallback_note = controller._prepare_preview_launcher(
        "dish",
        parameters,
        run_dir,
    )

    assert isinstance(launcher, DummyLauncher)
    assert calls == [False, True]
    assert "fallback disabled larva overlap elimination" in fallback_note


def test_single_experiment_parameter_editor_exposes_template_fields(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.experiment.value = "dish"
    controller._on_experiment_change()

    assert "duration" in controller._parameter_widgets
    assert "env_params.arena.geometry" in controller._parameter_widgets
    assert not any(path == "enrichment" for path in controller._parameter_widgets)
    assert not any(
        path.startswith("enrichment.") for path in controller._parameter_widgets
    )
    assert not any(
        path.startswith("larva_groups.") for path in controller._parameter_widgets
    )
    assert "collections" in controller._parameter_widgets
    assert "parameter_dict" not in controller._parameter_widgets
    assert controller.parameter_group.options["Enrichment"] == "enrichment"
    assert controller.parameter_group.options["Larva Groups"] == "larva_groups"
    assert controller.parameter_group.value == "larva_groups"
    assert len(controller.parameters_editor.objects) > 0


def test_single_experiment_enrichment_uses_typed_widget_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    sentinel = pn.pane.Markdown("typed-enrichment")
    captured: dict[str, object] = {}

    def fake_build(owner, *, wrap=True):
        captured["owner"] = owner
        captured["wrap"] = wrap
        return sentinel

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.build_enrichment_widget",
        fake_build,
    )

    controller = _SingleExperimentController()

    assert captured["owner"] is controller._typed_experiment_for_enrichment.enrichment
    assert captured["wrap"] is False
    controller.parameter_group.value = "enrichment"
    assert controller.parameters_editor.objects == [sentinel]


def test_single_experiment_larva_groups_uses_typed_widget_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    sentinel = pn.pane.Markdown("typed-larva-groups")
    captured: dict[str, object] = {}

    def fake_build(owner, *, parameter_name="larva_groups", wrap=True):
        captured["owner"] = owner
        captured["parameter_name"] = parameter_name
        captured["wrap"] = wrap
        return sentinel

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.build_larva_groups_widget",
        fake_build,
    )

    controller = _SingleExperimentController()

    assert captured["owner"] is controller._typed_experiment_for_larva_groups
    assert captured["parameter_name"] == "larva_groups"
    assert captured["wrap"] is True
    assert controller.parameter_group.value == "larva_groups"
    assert controller.parameters_editor.objects == [sentinel]


def test_single_experiment_mixed_flattened_and_typed_edits_survive_build_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.experiment.value = "dish"
    controller._on_experiment_change()

    controller._parameter_widgets["duration"][1].value = 2.0
    typed_owner = controller._typed_experiment_for_larva_groups
    assert typed_owner is not None
    group_id = next(iter(typed_owner.larva_groups.keys()))
    typed_owner.larva_groups[group_id].distribution.N = 11

    parameters = controller._build_parameters()
    flat = parameters.flatten()

    assert parameters.duration == pytest.approx(2.0)
    assert flat[f"larva_groups.{group_id}.distribution.N"] == 11


def test_single_experiment_enrichment_typed_edits_feed_build_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.experiment.value = "dish"
    controller._on_experiment_change()

    typed_owner = controller._typed_experiment_for_enrichment
    assert typed_owner is not None
    typed_owner.enrichment.mode = "full"
    typed_owner.enrichment.recompute = True
    typed_owner.enrichment.pre_kws.rescale_by = 0.002

    parameters = controller._build_parameters()

    assert parameters.enrichment.mode == "full"
    assert parameters.enrichment.recompute is True
    assert parameters.enrichment.pre_kws.rescale_by == pytest.approx(0.002)


def test_single_experiment_mixed_flattened_and_typed_enrichment_edits_survive_build_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.experiment.value = "dish"
    controller._on_experiment_change()

    controller._parameter_widgets["duration"][1].value = 2.0
    typed_owner = controller._typed_experiment_for_enrichment
    assert typed_owner is not None
    typed_owner.enrichment.mode = "full"

    parameters = controller._build_parameters()

    assert parameters.duration == pytest.approx(2.0)
    assert parameters.enrichment.mode == "full"


def test_single_experiment_build_parameters_falls_back_to_base_larva_groups_when_typed_owner_missing(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller._typed_experiment_for_larva_groups = None
    controller._larva_groups_group_view = None

    parameters = controller._build_parameters()

    assert "larva_groups" in parameters
    assert len(parameters.larva_groups) > 0


def test_single_experiment_build_parameters_falls_back_to_base_enrichment_when_typed_owner_missing(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller._typed_experiment_for_enrichment = None
    controller._enrichment_group_view = None

    parameters = controller._build_parameters()

    assert "enrichment" in parameters
    assert parameters.enrichment.mode in {"minimal", "full"}


@pytest.mark.skip(reason=SINGLE_EXPERIMENT_APP_INCOMPLETE_REASON)
def test_single_experiment_parameter_editor_values_feed_build_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.experiment.value = "dish"
    controller._on_experiment_change()
    population_path = next(
        path
        for path in controller._parameter_widgets
        if path.endswith(".distribution.N")
    )

    duration_kind, duration_widget = controller._parameter_widgets["duration"]
    geometry_kind, geometry_widget = controller._parameter_widgets[
        "env_params.arena.geometry"
    ]
    population_kind, population_widget = controller._parameter_widgets[population_path]
    collections_kind, collections_widget = controller._parameter_widgets["collections"]

    assert duration_kind == "float"
    assert geometry_kind == "option"
    assert population_kind == "int"
    assert collections_kind == "multichoice"

    duration_widget.value = 2.5
    geometry_widget.value = "rectangular"
    population_widget.value = 12
    collections_widget.value = ["pose"]

    parameters = controller._build_parameters()

    assert parameters.duration == pytest.approx(2.5)
    assert parameters.env_params.arena.geometry == "rectangular"
    assert parameters.flatten()[population_path] == 12


@pytest.mark.skip(reason=SINGLE_EXPERIMENT_APP_INCOMPLETE_REASON)
def test_single_experiment_distribution_tuple_fields_stay_typed(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.experiment.value = "dish"
    controller._on_experiment_change()
    population_path = next(
        path
        for path in controller._parameter_widgets
        if path.endswith(".distribution.N")
    )

    controller._parameter_widgets[population_path][1].value = 7
    parameters = controller._build_parameters()
    larva_group = LarvaGroup(**parameters.larva_groups.explorer)

    assert isinstance(larva_group.distribution.loc, tuple)
    assert isinstance(larva_group.distribution.scale, tuple)
    assert isinstance(larva_group.distribution.orientation_range, tuple)
    assert larva_group.distribution.N == 7


@pytest.mark.skip(reason=SINGLE_EXPERIMENT_APP_INCOMPLETE_REASON)
def test_single_experiment_uses_typed_widgets_for_model_and_optional_odor_fields(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.experiment.value = "chemotaxis"
    controller._on_experiment_change()

    model_path = next(
        path for path in controller._parameter_widgets if path.endswith(".model")
    )
    sample_path = next(
        path for path in controller._parameter_widgets if path.endswith(".sample")
    )
    odor_id_path = next(
        path for path in controller._parameter_widgets if path.endswith(".odor.id")
    )
    odor_intensity_path = next(
        path
        for path in controller._parameter_widgets
        if path.endswith(".odor.intensity")
    )
    odor_spread_path = next(
        path for path in controller._parameter_widgets if path.endswith(".odor.spread")
    )

    assert controller._parameter_widgets[model_path][0] == "option"
    assert controller._parameter_widgets[sample_path][0] == "option"
    assert controller._parameter_widgets[odor_id_path][0] == "option"
    assert controller._parameter_widgets[odor_intensity_path][0] == "optional_float"
    assert controller._parameter_widgets[odor_spread_path][0] == "optional_float"

    controller._parameter_widgets[odor_id_path][1].value = "Odor"
    controller._parameter_widgets[odor_intensity_path][1]["enabled"].value = True
    controller._parameter_widgets[odor_intensity_path][1]["widget"].value = 1.2
    controller._parameter_widgets[odor_spread_path][1]["enabled"].value = True
    controller._parameter_widgets[odor_spread_path][1]["widget"].value = 0.03

    parameters = controller._build_parameters()

    assert parameters.flatten()[odor_id_path] == "apple"
    assert parameters.flatten()[odor_intensity_path] == pytest.approx(1.2)
    assert parameters.flatten()[odor_spread_path] == pytest.approx(0.03)


def test_single_experiment_parameter_editor_has_no_text_fallback_widgets(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    raw_kinds = set()
    for exp_id in controller.experiment.options:
        controller.experiment.value = exp_id
        controller._on_experiment_change()
        raw_kinds.update(
            kind
            for kind, _control in controller._parameter_widgets.values()
            if kind in {"str", "json", "optional_str"}
        )

    assert raw_kinds == set()


def test_single_experiment_help_text_uses_param_docs_and_runtime_notes(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()

    duration_help = controller._help_text_for_path("duration")
    arena_help = controller._help_text_for_path("env_params.arena.geometry")
    group_help = controller._help_text_for_path("larva_groups.explorer.model")

    assert duration_help is not None and "simulation" in duration_help.lower()
    assert arena_help is not None and "arena" in arena_help.lower()
    assert group_help is not None and "group" in group_help.lower()


def test_single_experiment_nested_help_text_prefers_leaf_doc_over_generic_parent(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()

    odor_id_help = controller._help_text_for_path("larva_groups.explorer.odor.id")
    odor_spread_help = controller._help_text_for_path(
        "larva_groups.explorer.odor.spread"
    )

    assert (
        odor_id_help is not None and "unique id of the odorant" in odor_id_help.lower()
    )
    assert "the odor of the agent" not in odor_id_help.lower()
    assert (
        odor_spread_help is not None
        and "spread of the concentration gradient" in odor_spread_help.lower()
    )
    assert "the odor of the agent" not in odor_spread_help.lower()


def test_single_experiment_optional_family_toggles_seed_disabled_controls(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.experiment.value = "dish"
    controller._on_experiment_change()

    assert controller._parameter_widgets["env_params.odorscape"][0] == "toggle_factory"
    assert (
        controller._parameter_widgets["env_params.food_params.food_grid"][0]
        == "toggle_factory"
    )
    assert (
        controller._parameter_widgets["env_params.odorscape"][1]["enabled"].value
        is False
    )
    assert (
        controller._parameter_widgets["env_params.food_params.food_grid"][1][
            "enabled"
        ].value
        is False
    )
    assert "env_params.odorscape.odorscape" in controller._parameter_widgets
    assert "env_params.food_params.food_grid.unique_id" in controller._parameter_widgets
    assert (
        controller._parameter_widgets["env_params.odorscape.odorscape"][1].disabled
        is True
    )
    assert (
        controller._parameter_widgets["env_params.food_params.food_grid.color"][
            1
        ].disabled
        is True
    )

    controller._parameter_widgets["env_params.odorscape"][1]["enabled"].value = True
    controller._parameter_widgets["env_params.food_params.food_grid"][1][
        "enabled"
    ].value = True

    parameters = controller._build_parameters()
    assert parameters.env_params.odorscape.odorscape == "Gaussian"
    assert parameters.env_params.food_params.food_grid.unique_id == "FoodGrid"


def test_single_experiment_epoch_families_use_toggle_based_activation(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.experiment.value = "dish"
    controller._on_experiment_change()

    assert controller._parameter_widgets["trials.epochs"][0] == "toggle_factory"
    assert controller._parameter_widgets["trials.epochs"][1]["enabled"].value is False

    controller._parameter_widgets["trials.epochs"][1]["enabled"].value = True

    parameters = controller._build_parameters()

    assert len(parameters.trials.epochs) == 1


def test_single_experiment_source_unit_toggle_enables_nested_controls_near_source_family(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.experiment.value = "dish"
    controller._on_experiment_change()

    assert (
        controller._parameter_widgets["env_params.food_params.source_units"][0]
        == "toggle_factory"
    )

    env_paths = controller._parameter_groups["env_params"]
    source_amount_path = "env_params.food_params.source_units.Source.amount"
    source_color_path = "env_params.food_params.source_units.Source.color"

    assert source_amount_path in controller._parameter_widgets
    assert source_color_path in controller._parameter_widgets
    assert controller._parameter_widgets[source_amount_path][1].disabled is True
    assert controller._parameter_widgets[source_color_path][1].disabled is True
    assert env_paths.index(source_amount_path) < env_paths.index("env_params.odorscape")
    assert env_paths.index(source_color_path) < env_paths.index("env_params.odorscape")

    controller._parameter_widgets["env_params.food_params.source_units"][1][
        "enabled"
    ].value = True

    assert controller._parameter_widgets[source_amount_path][1].disabled is False
    assert controller._parameter_widgets[source_color_path][1].disabled is False


def test_single_experiment_uses_color_and_nested_odorscape_widgets(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.experiment.value = "chemotaxis"
    controller._on_experiment_change()

    color_path = next(
        path
        for path in controller._parameter_widgets
        if path.endswith(".source_units.Source.color")
    )
    odorscape_mode_path = next(
        path
        for path in controller._parameter_widgets
        if path.endswith(".odorscape.odorscape")
    )

    assert controller._parameter_widgets[color_path][0] == "color"
    assert controller._parameter_widgets[odorscape_mode_path][0] == "option"

    controller._parameter_widgets[color_path][1].value = "#112233"
    controller._parameter_widgets[odorscape_mode_path][1].value = "Gaussian"

    parameters = controller._build_parameters()

    assert parameters.flatten()[color_path] == "#112233"
    assert parameters.flatten()[odorscape_mode_path] == "Gaussian"


def test_single_experiment_sequence_widgets_use_semantic_component_labels(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.experiment.value = "dish"
    controller._on_experiment_change()

    dims_kind, dims_control = controller._parameter_widgets["env_params.arena.dims"]

    assert dims_kind == "sequence"
    assert [widget.name for widget in dims_control["widgets"]] == ["x", "y"]
    assert "enrichment.tor_durs" not in controller._parameter_widgets


def test_single_experiment_display_labels_hide_redundant_family_prefixes() -> None:
    assert (
        _display_label_for_path("env_params.food_params.source_groups")
        == "Source Groups"
    )
    assert _display_label_for_path("env_params.odorscape") == "Odorscape"
    assert _display_label_for_path("env_params.border_list.Border.group") == "Group"


def test_single_experiment_optional_roots_share_family_with_their_nested_fields() -> (
    None
):
    assert (
        _family_spec_for_path("env_params.food_params.source_units")[0]
        == _family_spec_for_path("env_params.food_params.source_units.Source.amount")[0]
    )
    assert (
        _family_spec_for_path("env_params.odorscape")[0]
        == _family_spec_for_path("env_params.odorscape.odorscape")[0]
    )
    assert (
        _family_spec_for_path("env_params.border_list")[0]
        == _family_spec_for_path("env_params.border_list.Border.width")[0]
    )


def test_single_experiment_preview_run_directory_gets_unique_suffix(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.run_name.value = "dish_demo"
    first = controller._build_run_directory()
    first.mkdir(parents=True, exist_ok=True)

    second = controller._build_run_directory()

    assert first.name == "dish_demo"
    assert second.name == "dish_demo_2"


def test_single_experiment_registry_item_is_now_panel_app() -> None:
    item = ITEMS["wf.run_experiment"]

    assert item.kind == "panel_app"
    assert item.status == "ready"
    assert item.panel_app_id == "wf.run_experiment"


def test_single_experiment_slug_helpers() -> None:
    assert _safe_slug(" Dish Demo / 01 ") == "Dish_Demo_01"
    assert _default_run_name("dish").startswith("dish_")
