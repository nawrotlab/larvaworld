from __future__ import annotations

import copy
import json
from pathlib import Path

import holoviews as hv
import numpy as np
import panel as pn
import pytest

from larvaworld.lib import reg, util
from larvaworld.lib.reg.larvagroup import LarvaGroup
from larvaworld.lib.sim.validation import CompatibilityIssue, CompatibilityReport
from larvaworld.portal.canvas_widgets.environment_models import (
    CanvasArena,
    EnvironmentCanvasState,
    LarvaPreviewFrame,
)
from larvaworld.portal.landing_registry import ITEMS
from larvaworld.portal.simulation.parameter_resolution import (
    resolve_base_experiment_parameters,
)
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


def _find_widget(
    viewable: pn.viewable.Viewable,
    name: str,
    widget_type: type[pn.widgets.Widget]
    | tuple[type[pn.widgets.Widget], ...] = pn.widgets.Widget,
):
    for widget in viewable.select(widget_type):
        if getattr(widget, "name", None) == name:
            return widget
    raise AssertionError(f"Could not find widget {name!r}.")


def _switches(viewable: pn.viewable.Viewable) -> list[pn.widgets.Switch]:
    return [widget for widget in viewable.select(pn.widgets.Switch)]


def _set_registry_experiment(
    controller: _SingleExperimentController, experiment_name: str
) -> None:
    controller.experiment.value = controller.experiment.options[
        f"Registry / {experiment_name}"
    ]


def _load_registry_template(
    controller: _SingleExperimentController, template_name: str
) -> None:
    controller.experiment_template_select.value = (
        controller.experiment_template_select.options[f"Registry / {template_name}"]
    )
    assert controller.experiment_template_preset_controls.load_selected() is True


def _load_workspace_template(
    controller: _SingleExperimentController, template_name: str
) -> None:
    controller.experiment_template_select.value = (
        controller.experiment_template_select.options[f"Workspace / {template_name}"]
    )
    assert controller.experiment_template_preset_controls.load_selected() is True


def _incompatible_parameters() -> util.AttrDict:
    return util.AttrDict(
        {
            "env_params": {
                "arena": {"geometry": "rectangular", "dims": (0.2, 0.2)},
                "food_params": {
                    "food_grid": None,
                    "source_groups": {},
                    "source_units": {},
                },
                "border_list": {},
                "odorscape": None,
                "windscape": None,
                "thermoscape": None,
            },
            "larva_groups": {
                "explorer": {
                    "distribution": {
                        "loc": (0.11, 0.0),
                        "scale": (0.01, 0.01),
                        "mode": "uniform",
                        "shape": "rect",
                    }
                }
            },
        }
    )


def test_single_experiment_lists_workspace_environment_presets(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    (workspace_root / "environments" / "dish_custom.json").write_text(
        json.dumps({"arena": {"geometry": "circular", "dims": [0.12, 0.12]}}) + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()

    assert controller.selection.environment_preset == "__template__"
    assert controller.environment_select.options["Registry / dish"].startswith(
        "registry:Env:"
    )
    assert (
        controller.environment_select.options["Workspace / dish_custom"]
        == "workspace:single-experiment-environments:dish_custom.json"
    )


def test_single_experiment_keeps_registry_and_workspace_same_name_visible(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    (workspace_root / "environments" / "dish.json").write_text(
        json.dumps({"arena": {"geometry": "circular", "dims": [0.12, 0.12]}}) + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()

    assert "Registry / dish" in controller.environment_select.options
    assert "Workspace / dish" in controller.environment_select.options


def test_single_experiment_lists_and_loads_registry_environment_presets(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()

    assert "Registry / maze" in controller.environment_select.options
    _set_registry_experiment(controller, "dish")
    controller.environment_select.value = controller.environment_select.options[
        "Registry / maze"
    ]
    assert controller.environment_preset_controls.load_selected() is True

    parameters = controller._build_parameters()

    assert parameters.env_params.arena.geometry == "rectangular"
    assert "Maze" in parameters.env_params.border_list
    assert controller._selected_environment_label() == "Registry / maze"


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

    _set_registry_experiment(controller, candidate)

    assert controller.selection.environment_preset == "__template__"
    assert controller._selected_environment_label() == "template default"
    assert controller.environment_select.options[f"Registry / {candidate}"].startswith(
        "registry:Env:"
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
    _set_registry_experiment(controller, "dish")
    controller.environment_select.value = controller.environment_select.options[
        "Workspace / rect_env"
    ]
    assert controller.environment_preset_controls.load_selected() is True
    typed_sim_owner = controller._typed_experiment_for_sim_ops
    assert typed_sim_owner is not None
    typed_sim_owner.duration = 1.5

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
    assert isinstance(controller.preview[0], pn.Row)
    assert controller.preview[0][0] is canvas_view
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
    assert isinstance(controller.preview[0], pn.Row)
    assert controller.preview[0][0] is canvas_view


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
    assert isinstance(controller.preview[0], pn.Row)
    assert isinstance(controller.preview[0][0], pn.pane.HTML)


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
    controller.environment_select.value = controller.environment_select.options[
        "Workspace / rect_env"
    ]
    assert controller.environment_preset_controls.load_selected() is True
    controller._on_prepare_preview()

    state = captured["state"]
    assert state.arena.geometry == "rectangular"
    assert state.arena.dims == (0.2, 0.1)


def test_single_experiment_prepare_preview_blocks_on_incompatible_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    monkeypatch.setattr(
        controller,
        "_resolve_experiment_parameters",
        lambda: _incompatible_parameters(),
    )
    monkeypatch.setattr(
        controller,
        "_build_run_directory",
        lambda: (_ for _ in ()).throw(
            AssertionError("run directory should not be created")
        ),
    )

    controller._on_prepare_preview()

    assert "Experiment configuration is incompatible for Arena Preview" in str(
        controller.status.object
    )
    assert "incompatible" in str(getattr(controller.preview[0], "object", "")).lower()


def test_single_experiment_registry_template_uses_legacy_validation_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    captured: dict[str, object] = {}
    warning_report = CompatibilityReport(
        issues=(
            CompatibilityIssue(
                severity="warning",
                path="larva_groups.toucher.distribution",
                message="Distribution envelope extends outside the arena.",
            ),
        )
    )

    def fake_validate(parameters, *, allow_registry_legacy=False, experiment_id=None):
        captured["allow_registry_legacy"] = allow_registry_legacy
        captured["experiment_id"] = experiment_id
        return warning_report

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.validate_experiment_environment_compatibility",
        fake_validate,
    )

    controller = _SingleExperimentController()
    _load_registry_template(controller, "tactile_detection")
    ok, warning_note = controller._validate_resolved_parameters_for_action(
        parameters=controller._resolve_experiment_parameters(),
        action_label="Arena Preview",
        show_preview_failure=False,
    )

    assert ok is True
    assert "Warning:" in warning_note
    assert captured["allow_registry_legacy"] is True
    assert captured["experiment_id"] == "tactile_detection"


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
    controller.environment_select.value = controller.environment_select.options[
        "Workspace / builder_scene"
    ]
    assert controller.environment_preset_controls.load_selected() is True

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
    _set_registry_experiment(controller, "dish")
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

    run_dir = workspace_root / "simulations" / "dish_demo"
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

    kws = controller._runtime_screen_kws(workspace_root / "simulations" / "dish_demo")

    assert controller.video_fps.name == "Video speed-up"
    assert kws["save_video"] is True
    assert kws["vis_mode"] == "video"
    assert kws["video_file"] == "dish_video"
    assert kws["fps"] == 3
    assert kws["show_display"] is True
    assert kws["display_every_n_steps"] == 4
    assert kws["pygame_keys"]["pause"] == "K_SPACE"
    assert kws["media_dir"] == str(workspace_root / "simulations" / "dish_demo")


def test_single_experiment_show_display_uses_video_render_mode(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.show_display.value = True
    controller.display_every_n_steps.value = 3

    kws = controller._runtime_screen_kws(workspace_root / "simulations" / "dish_demo")

    assert kws["show_display"] is True
    assert kws["vis_mode"] == "video"
    assert kws["display_every_n_steps"] == 3
    assert "pygame_keys" in kws
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
        run_dir=workspace_root / "simulations" / "dish_demo",
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


def test_single_experiment_run_experiment_blocks_on_incompatible_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    monkeypatch.setattr(
        controller,
        "_resolve_experiment_parameters",
        lambda: _incompatible_parameters(),
    )
    monkeypatch.setattr(
        controller,
        "_build_run_directory",
        lambda: (_ for _ in ()).throw(
            AssertionError("run directory should not be created")
        ),
    )
    monkeypatch.setattr(
        _SingleExperimentController,
        "_execute_run_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("run execution should not start")
        ),
    )

    controller._on_run_experiment()

    assert "Experiment configuration is incompatible for Run experiment" in str(
        controller.status.object
    )


def test_single_experiment_run_experiment_appends_compatibility_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    warning_report = CompatibilityReport(
        issues=(
            CompatibilityIssue(
                severity="warning",
                path="larva_groups.explorer.distribution",
                message="Normal distribution 3-sigma envelope extends outside the arena.",
            ),
        )
    )
    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.validate_experiment_environment_compatibility",
        lambda _parameters, **_kwargs: warning_report,
    )
    monkeypatch.setattr(
        _SingleExperimentController,
        "_execute_run_experiment",
        lambda *args, **kwargs: None,
    )

    controller.run_name.value = "dish_warning"
    controller._on_run_experiment()

    assert "Warning:" in str(controller.status.object)


def test_single_experiment_run_experiment_blocks_on_invalid_shortcuts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    monkeypatch.setattr(
        controller.display_shortcuts,
        "validate",
        lambda: ['simulation.pause: unsupported key "nope".'],
    )
    monkeypatch.setattr(
        controller,
        "_resolve_experiment_parameters",
        lambda: (_ for _ in ()).throw(
            AssertionError("parameter resolution should not start")
        ),
    )
    monkeypatch.setattr(
        controller,
        "_build_run_directory",
        lambda: (_ for _ in ()).throw(
            AssertionError("run directory should not be created")
        ),
    )
    monkeypatch.setattr(
        _SingleExperimentController,
        "_execute_run_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("run execution should not start")
        ),
    )

    controller._on_run_experiment()

    assert "display shortcuts are invalid" in str(controller.status.object)
    assert "unsupported key" in str(controller.status.object)


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

    assert "Output directory for a future run:" in controller.status.object
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


def test_single_experiment_generate_preview_blocks_on_incompatible_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    monkeypatch.setattr(
        controller,
        "_resolve_experiment_parameters",
        lambda: _incompatible_parameters(),
    )
    monkeypatch.setattr(
        controller,
        "_build_run_directory",
        lambda: (_ for _ in ()).throw(
            AssertionError("run directory should not be created")
        ),
    )
    monkeypatch.setattr(
        _SingleExperimentController,
        "_prepare_preview_launcher",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("preview launcher should not be created")
        ),
    )

    controller._on_generate_simulation_preview()

    assert (
        "Experiment configuration is incompatible for Generate simulation preview"
        in str(controller.status.object)
    )


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


def test_single_experiment_preview_canvas_row_excludes_display_shortcuts_legend() -> (
    None
):
    controller = _SingleExperimentController()

    assert isinstance(controller.preview[0], pn.Row)
    assert len(controller.preview[0].objects) == 1
    assert "Pause / resume" not in str(controller.preview[0])


def test_single_experiment_run_info_contains_display_shortcuts_link() -> None:
    controller = _SingleExperimentController()

    assert controller.display_shortcuts_link in controller.run_info.objects
    assert controller.display_shortcuts_link.name == "Display Shortcuts"
    assert "lw-single-exp-run-info-box" in controller.run_info.css_classes


def test_single_experiment_display_shortcuts_dialog_open_close_and_note() -> None:
    controller = _SingleExperimentController()

    assert controller.display_shortcuts_dialog.visible is False
    controller._on_open_display_shortcuts()
    assert controller.display_shortcuts_dialog.visible is True
    dialog_text = str(controller.display_shortcuts_dialog[0][1].object)
    assert "live pygame display opened by Run experiment" in dialog_text
    assert "They do not control the preview canvas." in dialog_text
    dialog_card = controller.display_shortcuts_dialog[0]
    drag_headers = [
        row
        for row in dialog_card.select(pn.Row)
        if "lw-single-exp-shortcuts-drag-handle" in getattr(row, "css_classes", [])
    ]
    assert len(drag_headers) == 1
    assert controller.display_shortcuts_close_btn in drag_headers[0].objects
    shortcuts_panel = controller.display_shortcuts_dialog[0][2]
    buttons = shortcuts_panel.select(pn.widgets.Button)
    names = [button.name for button in buttons]
    assert "Edit shortcuts" in names
    assert "Save shortcuts" in names
    assert "Reset all to defaults" in names
    markdowns = shortcuts_panel.select(pn.pane.Markdown)
    shortcuts_body = "\n".join(str(pane.object) for pane in markdowns)
    assert "]" not in shortcuts_body
    text_inputs = shortcuts_panel.select(pn.widgets.TextInput)
    assert all(widget.visible is False for widget in text_inputs)

    controller._on_close_display_shortcuts()
    assert controller.display_shortcuts_dialog.visible is False


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
    run_dir = workspace_root / "simulations" / "preview_case"

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
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()

    for key in (
        "duration",
        "Nsteps",
        "fr",
        "dt",
        "constant_framerate",
        "Box2D",
        "larva_collisions",
    ):
        assert key not in controller._parameter_widgets
    assert not any(path == "env_params" for path in controller._parameter_widgets)
    assert not any(
        path.startswith("env_params.") for path in controller._parameter_widgets
    )
    assert not any(path == "enrichment" for path in controller._parameter_widgets)
    assert not any(
        path.startswith("enrichment.") for path in controller._parameter_widgets
    )
    assert not any(
        path.startswith("larva_groups.") for path in controller._parameter_widgets
    )
    assert "collections" not in controller._parameter_widgets
    assert not any(
        path.startswith("collections.") for path in controller._parameter_widgets
    )
    assert "trials" not in controller._parameter_widgets
    assert not any(path.startswith("trials.") for path in controller._parameter_widgets)
    assert "parameter_dict" not in controller._parameter_widgets
    assert "sim_ops" in controller._parameter_groups
    assert "collections" in controller._parameter_groups
    assert "trials" in controller._parameter_groups
    assert "env_params" in controller._parameter_groups
    assert "enrichment" in controller._parameter_groups
    assert "larva_groups" in controller._parameter_groups
    assert len(controller.parameters_editor.objects) > 0


def test_single_experiment_view_hides_parameter_group_dropdown(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    view = controller.view()

    assert not any(
        isinstance(widget, pn.widgets.Select) and widget.name == "Parameter group"
        for widget in view.select(pn.widgets.Select)
    )


def test_single_experiment_parameter_groups_render_in_three_columns(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()

    assert len(controller.environment_parameters_editor.objects) == 1
    assert isinstance(controller.environment_parameters_editor[0], pn.Column)
    assert len(controller.parameters_editor.objects) == 1
    row = controller.parameters_editor[0]
    assert isinstance(row, pn.Row)
    assert len(row.objects) == 2
    assert all(isinstance(column, pn.Column) for column in row.objects)


def test_single_experiment_preview_container_is_stable_before_generation(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()

    assert "lw-single-exp-preview-body" in controller.preview.css_classes
    assert len(controller.preview.objects) >= 1
    placeholders = [
        pane
        for pane in controller.preview.objects
        if isinstance(pane, pn.pane.HTML)
        and "prepare the configuration preview" in str(getattr(pane, "object", ""))
    ]
    assert placeholders


def test_single_experiment_environment_preset_box_is_first_in_environment_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()

    env_section = controller.environment_parameters_editor[0]
    assert isinstance(env_section, pn.Column)
    env_content = env_section[0]
    assert isinstance(env_content, pn.Column)
    preset_box = env_content[0]
    assert isinstance(preset_box, pn.Column)
    assert controller.environment_preset_view in preset_box.objects
    assert controller.environment_template_default_btn in preset_box.select(
        pn.widgets.Button
    )
    assert controller.refresh_environments_btn in preset_box.select(pn.widgets.Button)
    assert controller.environment_preset_controls.reset_button is None


def test_single_experiment_environment_save_controls_dirty_and_name_gating(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    env_view = controller._get_parameter_group_view("env_params")
    assert env_view is not None

    assert controller.environment_save_name.disabled is True
    assert controller.environment_save_btn.disabled is True

    arena_width = _find_widget(env_view, "Arena width", pn.widgets.FloatInput)
    arena_width.value = float(arena_width.value) + 0.001

    assert controller.environment_save_name.disabled is False
    assert controller.environment_save_name.value == "my_environment"
    assert controller.environment_save_btn.disabled is False
    controller.environment_save_name.value = "My nice arena!"
    assert controller.environment_save_btn.disabled is False


def test_single_experiment_environment_save_writes_env_params_only(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    env_view = controller._get_parameter_group_view("env_params")
    assert env_view is not None
    arena_width = _find_widget(env_view, "Arena width", pn.widgets.FloatInput)
    arena_width.value = float(arena_width.value) + 0.002

    controller.environment_save_name.value = "saved_env"
    controller._on_save_environment_preset()

    target = workspace_root / "environments" / "saved_env.json"
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "arena" in payload
    assert "food_params" in payload
    assert "border_list" in payload
    assert "larva_groups" not in payload
    assert "enrichment" not in payload
    assert "trials" not in payload
    assert (
        controller.environment_select.options["Workspace / saved_env"]
        == "workspace:single-experiment-environments:saved_env.json"
    )
    assert (
        controller.selection.environment_preset
        == "workspace:single-experiment-environments:saved_env.json"
    )
    assert controller.environment_save_btn.disabled is True


def test_single_experiment_environment_save_collision_requires_confirmation(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    target = workspace_root / "environments" / "saved_env.json"
    target.write_text(
        json.dumps({"arena": {"geometry": "circular", "dims": [0.3, 0.3]}}) + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    env_view = controller._get_parameter_group_view("env_params")
    assert env_view is not None
    arena_width = _find_widget(env_view, "Arena width", pn.widgets.FloatInput)
    arena_width.value = float(arena_width.value) + 0.003
    controller.environment_save_name.value = "saved_env"

    initial_payload = target.read_text(encoding="utf-8")
    controller._on_save_environment_preset()

    assert controller.environment_preset_controls.confirmation_host.objects
    assert target.read_text(encoding="utf-8") == initial_payload

    controller._on_cancel_overwrite_environment()
    assert not controller.environment_preset_controls.confirmation_host.objects
    assert target.read_text(encoding="utf-8") == initial_payload

    controller._on_save_environment_preset()
    controller._on_confirm_overwrite_environment()
    assert target.read_text(encoding="utf-8") != initial_payload


def test_single_experiment_registry_load_then_workspace_save_same_name_keeps_registry(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    before_registry = copy.deepcopy(reg.conf.Env.getID("dish"))
    controller = _SingleExperimentController()
    controller.environment_select.value = controller.environment_select.options[
        "Registry / dish"
    ]
    assert controller.environment_preset_controls.load_selected() is True

    env_view = controller._get_parameter_group_view("env_params")
    assert env_view is not None
    arena_width = _find_widget(env_view, "Arena width", pn.widgets.FloatInput)
    arena_width.value = float(arena_width.value) + 0.002
    controller.environment_save_name.value = "dish"

    assert controller._on_save_environment_preset() is None
    target = workspace_root / "environments" / "dish.json"
    assert target.exists()
    assert "Registry / dish" in controller.environment_select.options
    assert "Workspace / dish" in controller.environment_select.options
    assert copy.deepcopy(reg.conf.Env.getID("dish")) == before_registry


def test_single_experiment_template_save_box_in_configuration_and_disabled_initially(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    view = controller.view()
    config_card = next(
        card
        for card in view.select(pn.Card)
        if getattr(card, "title", "") == "Configuration"
    )
    config_column = config_card.objects[0]

    assert controller.experiment_template_save_box in config_column.objects
    assert controller.experiment_template_preset_controls.view in (
        controller.experiment_template_save_box.objects
    )
    assert controller.experiment_template_preset_controls.reset_button is None
    assert controller.experiment_template_save_name.disabled is True
    assert controller.experiment_template_save_btn.disabled is True


def test_single_experiment_template_save_dirty_state_from_env_and_experiment_edits(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    assert controller.experiment_template_save_name.disabled is True
    assert controller.experiment_template_save_btn.disabled is True

    env_view = controller._get_parameter_group_view("env_params")
    assert env_view is not None
    arena_width = _find_widget(env_view, "Arena width", pn.widgets.FloatInput)
    arena_width.value = float(arena_width.value) + 0.001
    controller.experiment_template_save_name.value = "env_dirty_template"
    assert controller.experiment_template_save_name.disabled is False
    assert controller.experiment_template_save_btn.disabled is False
    assert "Will save as:" not in str(controller.experiment_template_save_hint.object)

    controller._refresh_parameter_editor()
    assert controller.experiment_template_save_name.disabled is True
    assert controller.experiment_template_save_btn.disabled is True
    experiment_view = controller._get_parameter_group_view("sim_ops")
    assert experiment_view is not None
    larva_collisions = _find_widget(
        experiment_view, "Larva collisions", pn.widgets.Checkbox
    )
    larva_collisions.value = not bool(larva_collisions.value)
    controller.experiment_template_save_name.value = "exp_dirty_template"
    assert controller.experiment_template_save_name.disabled is False
    assert controller.experiment_template_save_btn.disabled is False
    assert "Will save as:" not in str(controller.experiment_template_save_hint.object)


def test_single_experiment_template_save_writes_whitelisted_payload(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    env_view = controller._get_parameter_group_view("env_params")
    assert env_view is not None
    arena_width = _find_widget(env_view, "Arena width", pn.widgets.FloatInput)
    arena_width.value = float(arena_width.value) + 0.004
    controller.experiment_template_save_name.value = "saved_template"
    controller._on_save_experiment_template()

    target = (
        workspace_root / "metadata" / "experiment_templates" / "saved_template.json"
    )
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["experiment"] == "dish"
    for key in (
        "env_params",
        "larva_groups",
        "trials",
        "enrichment",
        "collections",
        "duration",
        "Nsteps",
        "fr",
        "dt",
        "constant_framerate",
        "Box2D",
        "larva_collisions",
    ):
        assert key in payload
    for key in (
        "__workspace__",
        "run_name",
        "save_video",
        "video_filename",
        "video_fps",
        "show_display",
        "display_every_n_steps",
        "preview_frames",
        "status",
        "dir",
        "screen_kws",
    ):
        assert key not in payload


def test_single_experiment_selector_lists_registry_and_workspace_templates(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    templates_dir = workspace_root / "metadata" / "experiment_templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    (templates_dir / "my_template.json").write_text(
        json.dumps({"experiment": "dish"}) + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    controller._refresh_experiment_template_options()

    assert "Registry / dish" in controller.experiment_template_select.options
    assert controller.experiment_template_select.options["Registry / dish"].startswith(
        "registry:Exp:"
    )
    assert "Workspace / my_template" in controller.experiment_template_select.options
    assert (
        controller.experiment_template_select.options["Workspace / my_template"]
        == "workspace:single-experiment-templates:my_template.json"
    )


def test_single_experiment_template_helper_select_does_not_mutate_state(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    before = controller.selection.experiment_template
    controller.experiment_template_select.value = (
        controller.experiment_template_select.options["Registry / chemotaxis"]
    )

    assert controller.selection.experiment_template == before
    assert controller._selected_experiment() == "dish"


def test_single_experiment_template_helper_hides_registry_reset_action(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    assert controller.experiment_template_preset_controls.reset_button is None


def test_single_experiment_template_helper_registry_actions_are_read_only(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    assert (
        controller.experiment_template_preset_controls.policy.can_save_registry is False
    )
    assert (
        controller.experiment_template_preset_controls.policy.can_delete_registry
        is False
    )
    assert (
        controller.experiment_template_preset_controls.policy.can_reset_registry
        is False
    )
    controller.experiment_template_select.value = (
        controller.experiment_template_select.options["Registry / dish"]
    )
    assert controller.experiment_template_preset_controls.delete_selected() is False
    assert "read-only" in str(
        controller.experiment_template_preset_controls.status.object
    )


def test_single_experiment_template_same_name_registry_and_workspace_coexist(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    templates_dir = workspace_root / "metadata" / "experiment_templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    (templates_dir / "dish.json").write_text(
        json.dumps({"experiment": "dish"}) + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    assert "Registry / dish" in controller.experiment_template_select.options
    assert "Workspace / dish" in controller.experiment_template_select.options


def test_single_experiment_registry_template_load_resets_env_preset_to_default(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller.environment_select.value = controller.environment_select.options[
        "Registry / maze"
    ]
    assert controller.environment_preset_controls.load_selected() is True
    assert controller.selection.environment_preset != "__template__"

    _load_registry_template(controller, "dish")
    assert controller.selection.environment_preset == "__template__"


def test_single_experiment_workspace_template_load_uses_payload_experiment_id(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    templates_dir = workspace_root / "metadata" / "experiment_templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    (templates_dir / "my_template.json").write_text(
        json.dumps(
            {
                "experiment": "dish",
                "env_params": {"arena": {"geometry": "circular", "dims": [0.2, 0.2]}},
                "larva_collisions": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    _load_workspace_template(controller, "my_template")
    parameters = controller._build_parameters()

    assert controller._selected_experiment() == "dish"
    assert parameters.env_params.arena.dims == (0.2, 0.2)
    assert bool(parameters.larva_collisions) is False
    assert controller.experiment_template_save_name.value == "my_template"
    assert controller.experiment_template_save_btn.disabled is True


def test_single_experiment_workspace_template_load_refreshes_typed_ui_sections(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    templates_dir = workspace_root / "metadata" / "experiment_templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    (templates_dir / "dish_1.json").write_text(
        json.dumps(
            {
                "experiment": "dish",
                "env_params": {
                    "arena": {"geometry": "rectangular", "dims": [0.21, 0.11]}
                },
                "larva_groups": {
                    "explorer": {
                        "distribution": {"N": 13, "mode": "uniform", "shape": "rect"}
                    }
                },
                "enrichment": {"mode": "full"},
                "duration": 4.0,
                "larva_collisions": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    _load_workspace_template(controller, "dish_1")

    env_owner = controller._typed_experiment_for_env_params
    larva_owner = controller._typed_experiment_for_larva_groups
    enrich_owner = controller._typed_experiment_for_enrichment
    sim_owner = controller._typed_experiment_for_sim_ops
    assert env_owner is not None
    assert larva_owner is not None
    assert enrich_owner is not None
    assert sim_owner is not None

    assert env_owner.env_params.arena.geometry == "rectangular"
    assert tuple(env_owner.env_params.arena.dims) == pytest.approx((0.21, 0.11))
    assert larva_owner.larva_groups["explorer"].distribution.N == 13
    assert larva_owner.larva_groups["explorer"].distribution.mode == "uniform"
    assert larva_owner.larva_groups["explorer"].distribution.shape == "rect"
    assert enrich_owner.enrichment.mode == "full"
    assert sim_owner.duration == pytest.approx(4.0)


def test_single_experiment_template_save_load_preserves_added_larva_group(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    larva_view = controller._get_parameter_group_view("larva_groups")
    assert larva_view is not None
    key_input = _find_widget(larva_view, "New larva group ID", pn.widgets.TextInput)
    add_button = _find_widget(larva_view, "Add larva group", pn.widgets.Button)

    key_input.value = "added_group"
    add_button.clicks += 1
    controller.experiment_template_save_name.value = "dish_with_added_group"
    controller._on_save_experiment_template()

    target = (
        workspace_root
        / "metadata"
        / "experiment_templates"
        / "dish_with_added_group.json"
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["larva_groups"]["added_group"]["group_id"] == "added_group"

    reloaded = _SingleExperimentController()
    _load_workspace_template(reloaded, "dish_with_added_group")
    larva_owner = reloaded._typed_experiment_for_larva_groups
    assert larva_owner is not None
    assert "added_group" in larva_owner.larva_groups
    assert larva_owner.larva_groups["added_group"].group_id == "added_group"


def test_single_experiment_workspace_template_load_materializes_nested_env_items(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    templates_dir = workspace_root / "metadata" / "experiment_templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    (templates_dir / "dish_nested_items.json").write_text(
        json.dumps(
            {
                "experiment": "dish",
                "env_params": {
                    "food_params": {
                        "source_units": {
                            "new_source": {
                                "pos": [0.0, 0.0],
                                "radius": 0.01,
                                "color": "red",
                            }
                        },
                        "source_groups": {
                            "new_group": {
                                "distribution": {
                                    "N": 2,
                                    "loc": [0.0, 0.0],
                                    "scale": [0.0, 0.0],
                                },
                                "radius": 0.008,
                            }
                        },
                    },
                    "border_list": {
                        "new_border": {
                            "vertices": [[-0.01, 0.0], [0.01, 0.0]],
                            "width": 0.002,
                        }
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    _load_workspace_template(controller, "dish_nested_items")

    env_owner = controller._typed_experiment_for_env_params
    assert env_owner is not None
    food_params = env_owner.env_params.food_params
    assert "new_source" in food_params.source_units
    assert food_params.source_units["new_source"].pos == pytest.approx((0.0, 0.0))
    assert food_params.source_units["new_source"].radius == pytest.approx(0.01)
    assert "new_group" in food_params.source_groups
    assert food_params.source_groups["new_group"].distribution.N == 2
    assert "new_border" in env_owner.env_params.border_list
    assert env_owner.env_params.border_list["new_border"].width == pytest.approx(0.002)
    assert env_owner.env_params.border_list["new_border"].vertices == [
        (-0.01, 0.0),
        (0.01, 0.0),
    ]


def test_single_experiment_workspace_template_missing_experiment_is_rejected(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    templates_dir = workspace_root / "metadata" / "experiment_templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    (templates_dir / "bad_template.json").write_text(
        json.dumps({"env_params": {"arena": {"dims": [0.3, 0.3]}}}) + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    previous = controller.selection.experiment_template
    controller.experiment_template_select.value = (
        controller.experiment_template_select.options["Workspace / bad_template"]
    )
    assert controller.experiment_template_preset_controls.load_selected() is False

    assert controller.selection.experiment_template == previous
    assert controller._selected_experiment() == "dish"
    assert previous.startswith("registry:Exp:")
    assert "missing required field: experiment" in str(controller.status.object)


def test_single_experiment_template_save_refreshes_selector_options(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    env_view = controller._get_parameter_group_view("env_params")
    assert env_view is not None
    arena_width = _find_widget(env_view, "Arena width", pn.widgets.FloatInput)
    arena_width.value = float(arena_width.value) + 0.01
    controller.experiment_template_save_name.value = "new_selector_template"
    controller._on_save_experiment_template()

    assert (
        "Workspace / new_selector_template"
        in controller.experiment_template_select.options
    )
    assert (
        controller.experiment_template_select.options[
            "Workspace / new_selector_template"
        ]
        == "workspace:single-experiment-templates:new_selector_template.json"
    )


def test_single_experiment_template_save_collision_requires_confirmation(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    target = (
        workspace_root / "metadata" / "experiment_templates" / "saved_template.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"env_params": {"arena": {"dims": [0.1, 0.1]}}}) + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    env_view = controller._get_parameter_group_view("env_params")
    assert env_view is not None
    arena_width = _find_widget(env_view, "Arena width", pn.widgets.FloatInput)
    arena_width.value = float(arena_width.value) + 0.005
    controller.experiment_template_save_name.value = "saved_template"
    initial_payload = target.read_text(encoding="utf-8")
    controller._on_save_experiment_template()

    assert controller.experiment_template_preset_controls.confirmation_host.objects
    assert target.read_text(encoding="utf-8") == initial_payload


def test_single_experiment_template_save_blocks_before_overwrite_confirmation_when_incompatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    target = (
        workspace_root / "metadata" / "experiment_templates" / "saved_template.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {"experiment": "dish", "env_params": {"arena": {"dims": [0.1, 0.1]}}}
        )
        + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    controller.experiment_template_save_name.value = "saved_template"
    initial_payload = target.read_text(encoding="utf-8")
    monkeypatch.setattr(
        controller,
        "_resolve_experiment_parameters",
        lambda: _incompatible_parameters(),
    )

    controller._on_save_experiment_template()

    assert not controller.experiment_template_preset_controls.confirmation_host.objects
    assert target.read_text(encoding="utf-8") == initial_payload
    assert "incompatible" in str(controller.status.object).lower()


def test_single_experiment_template_save_cancel_and_confirm_overwrite(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    target = (
        workspace_root / "metadata" / "experiment_templates" / "saved_template.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"env_params": {"arena": {"dims": [0.1, 0.1]}}}) + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    env_view = controller._get_parameter_group_view("env_params")
    assert env_view is not None
    arena_width = _find_widget(env_view, "Arena width", pn.widgets.FloatInput)
    arena_width.value = float(arena_width.value) + 0.006
    controller.experiment_template_save_name.value = "saved_template"
    initial_payload = target.read_text(encoding="utf-8")

    controller._on_save_experiment_template()
    controller._on_cancel_overwrite_experiment_template()
    assert target.read_text(encoding="utf-8") == initial_payload
    assert not controller.experiment_template_preset_controls.confirmation_host.objects

    controller._on_save_experiment_template()
    controller._on_confirm_overwrite_experiment_template()
    assert target.read_text(encoding="utf-8") != initial_payload
    assert controller.experiment_template_save_name.disabled is True
    assert controller.experiment_template_save_btn.disabled is True


def test_single_experiment_workspace_load_edit_save_requires_overwrite_confirmation(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    templates_dir = workspace_root / "metadata" / "experiment_templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    target = templates_dir / "my_template.json"
    target.write_text(
        json.dumps({"experiment": "dish", "larva_collisions": True}) + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    _load_workspace_template(controller, "my_template")
    sim_ops_view = controller._get_parameter_group_view("sim_ops")
    assert sim_ops_view is not None
    larva_collisions = _find_widget(
        sim_ops_view, "Larva collisions", pn.widgets.Checkbox
    )
    larva_collisions.value = not bool(larva_collisions.value)

    assert controller.experiment_template_save_name.disabled is False
    assert controller.experiment_template_save_btn.disabled is False
    controller._on_save_experiment_template()

    assert controller.experiment_template_preset_controls.confirmation_host.objects


def test_single_experiment_workspace_load_larva_edit_enables_template_save(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)
    templates_dir = workspace_root / "metadata" / "experiment_templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    target = templates_dir / "dish_1.json"
    target.write_text(
        json.dumps(
            {
                "experiment": "dish",
                "larva_groups": {
                    "explorer": {
                        "distribution": {
                            "N": 35,
                            "mode": "grid",
                            "shape": "rect",
                        }
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    controller = _SingleExperimentController()
    _load_workspace_template(controller, "dish_1")
    assert controller.experiment_template_save_name.disabled is True
    assert controller.experiment_template_save_btn.disabled is True

    larva_owner = controller._typed_experiment_for_larva_groups
    assert larva_owner is not None
    larva_owner.larva_groups["explorer"].distribution.N = 36
    controller._refresh_experiment_template_save_state(reset_baseline=False)

    assert controller.experiment_template_save_name.value == "dish_1"
    assert controller.experiment_template_save_name.disabled is False
    assert controller.experiment_template_save_btn.disabled is False


def test_single_experiment_env_params_uses_typed_widget_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    sentinel = pn.pane.Markdown("typed-env-params")
    captured: dict[str, object] = {}

    def fake_build(owner, *, wrap=True):
        captured["owner"] = owner
        captured["wrap"] = wrap
        return sentinel

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.build_env_params_widget",
        fake_build,
    )

    controller = _SingleExperimentController()

    assert captured["owner"] is controller._typed_experiment_for_env_params.env_params
    assert captured["wrap"] is False
    env_group_view = controller._get_parameter_group_view("env_params")
    assert env_group_view is not None
    if hasattr(env_group_view, "objects"):
        assert sentinel in env_group_view.objects
    else:
        assert env_group_view is sentinel


def test_single_experiment_sim_ops_uses_typed_widget_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    sentinel = pn.pane.Markdown("typed-sim-ops")
    captured: dict[str, object] = {}

    def fake_build(owner, *, wrap=True):
        captured["owner"] = owner
        captured["wrap"] = wrap
        return sentinel

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.build_sim_ops_widget",
        fake_build,
    )

    controller = _SingleExperimentController()

    assert captured["owner"] is controller._typed_experiment_for_sim_ops
    assert captured["wrap"] is False
    assert controller._get_parameter_group_view("sim_ops") is sentinel


def test_single_experiment_collections_uses_typed_widget_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    sentinel = pn.pane.Markdown("typed-collections")
    captured: dict[str, object] = {}

    def fake_build(owner, *, wrap=True):
        captured["owner"] = owner
        captured["wrap"] = wrap
        return sentinel

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.build_collections_widget",
        fake_build,
    )

    controller = _SingleExperimentController()

    assert captured["owner"] is controller._typed_experiment_for_collections
    assert captured["wrap"] is False
    assert controller._get_parameter_group_view("collections") is sentinel


def test_single_experiment_trials_uses_typed_widget_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    sentinel = pn.pane.Markdown("typed-trials")
    captured: dict[str, object] = {}

    def fake_build(owner, *, wrap=True):
        captured["owner"] = owner
        captured["wrap"] = wrap
        return sentinel

    monkeypatch.setattr(
        "larvaworld.portal.simulation.single_experiment_app.build_trials_widget",
        fake_build,
    )

    controller = _SingleExperimentController()

    assert captured["owner"] is controller._typed_experiment_for_trials
    assert captured["wrap"] is False
    assert controller._get_parameter_group_view("trials") is sentinel


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
    assert controller._get_parameter_group_view("enrichment") is sentinel


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
    assert controller._get_parameter_group_view("larva_groups") is sentinel


def test_single_experiment_mixed_flattened_and_typed_edits_survive_build_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()

    typed_collections_owner = controller._typed_experiment_for_collections
    assert typed_collections_owner is not None
    option_values = list(typed_collections_owner.param["collections"].objects)
    selected = option_values[:2] if len(option_values) > 1 else option_values[:1]
    typed_collections_owner.collections = selected
    typed_owner = controller._typed_experiment_for_larva_groups
    assert typed_owner is not None
    group_id = next(iter(typed_owner.larva_groups.keys()))
    typed_owner.larva_groups[group_id].distribution.N = 11

    parameters = controller._build_parameters()
    flat = parameters.flatten()

    assert list(parameters.collections) == selected
    assert flat[f"larva_groups.{group_id}.distribution.N"] == 11


def test_single_experiment_env_params_typed_edits_feed_build_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()

    typed_owner = controller._typed_experiment_for_env_params
    assert typed_owner is not None
    typed_owner.env_params.arena.geometry = "rectangular"
    typed_owner.env_params.arena.dims = (0.2, 0.1)

    parameters = controller._build_parameters()

    assert parameters.env_params.arena.geometry == "rectangular"
    assert parameters.env_params.arena.dims == pytest.approx((0.2, 0.1))


def test_single_experiment_mixed_flattened_and_typed_env_params_edits_survive_build_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()

    typed_trials_owner = controller._typed_experiment_for_trials
    assert typed_trials_owner is not None
    typed_trials_owner.trials["epochs"] = util.ItemList([reg.gen.Epoch().nestedConf])
    typed_owner = controller._typed_experiment_for_env_params
    assert typed_owner is not None
    typed_owner.env_params.arena.geometry = "rectangular"

    parameters = controller._build_parameters()

    assert len(parameters.trials.epochs) == 1
    assert parameters.env_params.arena.geometry == "rectangular"


def test_single_experiment_sim_ops_typed_edits_feed_build_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()

    typed_owner = controller._typed_experiment_for_sim_ops
    assert typed_owner is not None
    typed_owner.duration = 3.0
    typed_owner.dt = 0.2
    typed_owner.Box2D = True
    typed_owner.larva_collisions = False

    parameters = controller._build_parameters()

    assert parameters.duration == pytest.approx(3.0)
    assert parameters.dt == pytest.approx(0.2)
    assert parameters.fr == pytest.approx(1 / parameters.dt)
    assert parameters.Nsteps == int(parameters.duration * 60 / parameters.dt)
    assert parameters.Box2D is True
    assert parameters.larva_collisions is False


def test_single_experiment_mixed_typed_sim_and_flattened_trials_survive_build_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()

    typed_trials_owner = controller._typed_experiment_for_trials
    assert typed_trials_owner is not None
    typed_trials_owner.trials["epochs"] = util.ItemList([reg.gen.Epoch().nestedConf])

    typed_owner = controller._typed_experiment_for_sim_ops
    assert typed_owner is not None
    typed_owner.duration = 2.4

    parameters = controller._build_parameters()

    assert parameters.duration == pytest.approx(2.4)
    assert len(parameters.trials.epochs) == 1


def test_single_experiment_workspace_environment_override_survives_typed_env_ownership(
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
                            "color": "#44aa55",
                            "odor": {"id": "apple", "intensity": 1.0, "spread": 0.02},
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
    _set_registry_experiment(controller, "dish")
    controller.environment_select.value = controller.environment_select.options[
        "Workspace / rect_env"
    ]
    assert controller.environment_preset_controls.load_selected() is True
    typed_owner = controller._typed_experiment_for_env_params
    assert typed_owner is not None
    typed_owner.env_params.arena.torus = True

    parameters = controller._build_parameters()

    assert parameters.env_params.arena.geometry == "rectangular"
    assert parameters.env_params.arena.dims == pytest.approx((0.2, 0.1))
    assert parameters.env_params.arena.torus is True
    assert "patch" in parameters.env_params.food_params.source_units
    assert parameters.env_params.food_params.source_units["patch"].pos == pytest.approx(
        (0.02, 0.0)
    )


def test_single_experiment_registry_environment_override_survives_typed_env_ownership(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
    controller.environment_select.value = controller.environment_select.options[
        "Registry / maze"
    ]
    assert controller.environment_preset_controls.load_selected() is True
    typed_owner = controller._typed_experiment_for_env_params
    assert typed_owner is not None
    typed_owner.env_params.arena.torus = True

    parameters = controller._build_parameters()

    assert parameters.env_params.arena.geometry == "rectangular"
    assert parameters.env_params.arena.torus is True
    assert "Maze" in parameters.env_params.border_list
    assert len(parameters.env_params.border_list.Maze.vertices) > 0


def test_single_experiment_final_typed_env_params_remain_canvas_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    captured: dict[str, object] = {}
    mapped_state = EnvironmentCanvasState(arena=CanvasArena("rectangular", (0.2, 0.1)))
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
    typed_owner = controller._typed_experiment_for_env_params
    assert typed_owner is not None
    typed_owner.env_params.arena.geometry = "rectangular"
    typed_owner.env_params.arena.dims = (0.2, 0.1)

    controller._on_prepare_preview()

    assert captured["state"] is mapped_state
    env_params = util.AttrDict(captured["env_params"])
    assert env_params.arena.geometry == "rectangular"
    assert env_params.arena.dims == pytest.approx((0.2, 0.1))
    assert captured["larva_groups"] is not None
    assert captured["show_group_shapes"] is False


def test_single_experiment_enrichment_typed_edits_feed_build_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
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
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()

    typed_trials_owner = controller._typed_experiment_for_trials
    assert typed_trials_owner is not None
    typed_trials_owner.trials["epochs"] = util.ItemList([reg.gen.Epoch().nestedConf])
    typed_owner = controller._typed_experiment_for_enrichment
    assert typed_owner is not None
    typed_owner.enrichment.mode = "full"

    parameters = controller._build_parameters()

    assert len(parameters.trials.epochs) == 1
    assert parameters.enrichment.mode == "full"


def test_single_experiment_collections_typed_edits_feed_build_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()

    typed_owner = controller._typed_experiment_for_collections
    assert typed_owner is not None
    option_values = list(typed_owner.param["collections"].objects)
    selected = option_values[:2] if len(option_values) > 1 else option_values[:1]
    typed_owner.collections = selected

    parameters = controller._build_parameters()
    assert list(parameters.collections) == selected


def test_single_experiment_mixed_typed_collections_and_flattened_trials_survive_build_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()

    typed_trials_owner = controller._typed_experiment_for_trials
    assert typed_trials_owner is not None
    typed_trials_owner.trials["epochs"] = util.ItemList([reg.gen.Epoch().nestedConf])

    typed_owner = controller._typed_experiment_for_collections
    assert typed_owner is not None
    option_values = list(typed_owner.param["collections"].objects)
    selected = option_values[:2] if len(option_values) > 1 else option_values[:1]
    typed_owner.collections = selected

    parameters = controller._build_parameters()
    assert list(parameters.collections) == selected
    assert len(parameters.trials.epochs) == 1


def test_single_experiment_typed_sim_edits_do_not_reset_other_typed_groups(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()

    sim_owner = controller._typed_experiment_for_sim_ops
    trials_owner = controller._typed_experiment_for_trials
    env_owner = controller._typed_experiment_for_env_params
    collections_owner = controller._typed_experiment_for_collections
    larva_owner = controller._typed_experiment_for_larva_groups
    enrich_owner = controller._typed_experiment_for_enrichment

    assert sim_owner is not None
    assert trials_owner is not None
    assert env_owner is not None
    assert collections_owner is not None
    assert larva_owner is not None
    assert enrich_owner is not None

    group_id = next(iter(larva_owner.larva_groups.keys()))
    option_values = list(collections_owner.param["collections"].objects)
    selected = option_values[:2] if len(option_values) > 1 else option_values[:1]
    sim_owner.duration = 2.2
    trials_owner.trials["epochs"] = util.ItemList([reg.gen.Epoch().nestedConf])
    env_owner.env_params.arena.geometry = "rectangular"
    collections_owner.collections = selected
    larva_owner.larva_groups[group_id].distribution.N = 9
    enrich_owner.enrichment.mode = "full"

    parameters = controller._build_parameters()

    assert parameters.duration == pytest.approx(2.2)
    assert len(parameters.trials.epochs) == 1
    assert parameters.env_params.arena.geometry == "rectangular"
    assert list(parameters.collections) == selected
    assert parameters.flatten()[f"larva_groups.{group_id}.distribution.N"] == 9
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


def test_single_experiment_build_parameters_falls_back_to_base_env_params_when_typed_owner_missing(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller._typed_experiment_for_env_params = None
    controller._env_params_group_view = None

    parameters = controller._build_parameters()

    assert "env_params" in parameters
    assert parameters.env_params.arena.geometry in {"circular", "rectangular"}


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


def test_single_experiment_build_parameters_falls_back_to_base_sim_ops_when_typed_owner_missing(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller._typed_experiment_for_sim_ops = None
    controller._sim_ops_group_view = None

    parameters = controller._build_parameters()

    for key in (
        "duration",
        "Nsteps",
        "fr",
        "dt",
        "constant_framerate",
        "Box2D",
        "larva_collisions",
    ):
        assert key in parameters


def test_single_experiment_build_parameters_falls_back_to_base_collections_when_typed_owner_missing(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller._typed_experiment_for_collections = None
    controller._collections_group_view = None

    parameters = controller._build_parameters()

    assert "collections" in parameters
    assert isinstance(parameters.collections, list)


def test_single_experiment_build_parameters_falls_back_to_base_trials_when_typed_owner_missing(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    controller._typed_experiment_for_trials = None
    controller._trials_group_view = None

    parameters = controller._build_parameters()
    base = resolve_base_experiment_parameters(
        controller._selected_experiment(),
        controller._load_selected_environment(),
    )

    assert util.AttrDict(parameters.trials) == util.AttrDict(base.trials)


def test_single_experiment_trials_preserve_unknown_keys_in_build_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()

    typed_trials_owner = controller._typed_experiment_for_trials
    assert typed_trials_owner is not None
    typed_trials_owner.trials["custom_key"] = "keep-me"
    typed_trials_owner.trials["epochs"] = util.ItemList([reg.gen.Epoch().nestedConf])

    parameters = controller._build_parameters()
    assert parameters.trials.custom_key == "keep-me"
    assert len(parameters.trials.epochs) == 1


def test_single_experiment_trials_edits_survive_group_switching(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()

    typed_trials_owner = controller._typed_experiment_for_trials
    assert typed_trials_owner is not None
    typed_trials_owner.trials["epochs"] = util.ItemList([reg.gen.Epoch().nestedConf])
    typed_trials_owner.trials["epochs"][0]["age_range"] = (1.0, 2.0)

    assert controller._get_parameter_group_view("env_params") is not None
    assert controller._get_parameter_group_view("trials") is not None

    parameters = controller._build_parameters()
    assert tuple(parameters.trials.epochs[0].age_range) == pytest.approx((1.0, 2.0))


@pytest.mark.skip(reason=SINGLE_EXPERIMENT_APP_INCOMPLETE_REASON)
def test_single_experiment_parameter_editor_values_feed_build_parameters(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
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
    _set_registry_experiment(controller, "dish")
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
    _set_registry_experiment(controller, "chemotaxis")
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
    for exp_token in controller.experiment.options.values():
        controller.experiment.value = exp_token
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
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()
    env_view = controller._get_parameter_group_view("env_params")
    assert env_view is not None

    typed_owner = controller._typed_experiment_for_env_params
    assert typed_owner is not None
    assert typed_owner.env_params.odorscape is None
    assert typed_owner.env_params.food_params.food_grid is None

    enable_food_grid = _find_widget(env_view, "Enable Food grid", pn.widgets.Checkbox)
    enable_food_grid.value = True
    enable_odorscape, *_ = _switches(env_view)
    enable_odorscape.value = True

    parameters = controller._build_parameters()
    assert parameters.env_params.odorscape.odorscape == "Gaussian"
    assert parameters.env_params.food_params.food_grid.unique_id == "FoodGrid"


def test_single_experiment_trials_group_uses_typed_owner_for_activation(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()

    assert "trials" not in controller._parameter_widgets
    assert not any(path.startswith("trials.") for path in controller._parameter_widgets)
    assert "trials" in controller._parameter_groups

    typed_trials_owner = controller._typed_experiment_for_trials
    assert typed_trials_owner is not None
    typed_trials_owner.trials["epochs"] = util.ItemList([reg.gen.Epoch().nestedConf])

    parameters = controller._build_parameters()

    assert len(parameters.trials.epochs) == 1


def test_single_experiment_source_unit_toggle_enables_nested_controls_near_source_family(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()
    env_view = controller._get_parameter_group_view("env_params")
    assert env_view is not None

    add_source_unit = _find_widget(env_view, "Add source unit", pn.widgets.Button)
    source_unit_id = _find_widget(env_view, "New source unit ID", pn.widgets.TextInput)
    source_unit_id.value = "Source"
    add_source_unit.clicks += 1
    source_color = _find_widget(env_view, "Color", pn.widgets.ColorPicker)

    typed_owner = controller._typed_experiment_for_env_params
    assert typed_owner is not None
    assert "Source" in typed_owner.env_params.food_params.source_units

    source_color.value = "#112233"
    parameters = controller._build_parameters()

    assert parameters.env_params.food_params.source_units.Source.color == "#112233"


def test_single_experiment_uses_color_and_nested_odorscape_widgets(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "chemotaxis")
    controller._on_experiment_change()
    env_view = controller._get_parameter_group_view("env_params")
    assert env_view is not None

    add_source_unit = _find_widget(env_view, "Add source unit", pn.widgets.Button)
    source_unit_id = _find_widget(env_view, "New source unit ID", pn.widgets.TextInput)
    source_unit_id.value = "Source"
    add_source_unit.clicks += 1
    source_color = _find_widget(env_view, "Color", pn.widgets.ColorPicker)

    enable_odorscape, *_ = _switches(env_view)
    enable_odorscape.value = True
    odorscape_mode = _find_widget(env_view, "Odorscape type", pn.widgets.Select)

    source_color.value = "#112233"
    odorscape_mode.value = "DiffusionValueLayer"

    parameters = controller._build_parameters()

    assert parameters.env_params.food_params.source_units.Source.color == "#112233"
    assert parameters.env_params.odorscape.odorscape == "Diffusion"


def test_single_experiment_sequence_widgets_use_semantic_component_labels(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    initialize_workspace(workspace_root)
    set_active_workspace_path(workspace_root)

    controller = _SingleExperimentController()
    _set_registry_experiment(controller, "dish")
    controller._on_experiment_change()
    env_view = controller._get_parameter_group_view("env_params")
    assert env_view is not None

    width = _find_widget(env_view, "Arena width", pn.widgets.FloatInput)
    height = _find_widget(env_view, "Arena height", pn.widgets.FloatInput)
    assert width.name == "Arena width"
    assert height.name == "Arena height"
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
