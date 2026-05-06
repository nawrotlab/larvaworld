from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import holoviews as hv
import panel as pn
import param

from larvaworld.lib import reg, screen, sim, util
from larvaworld.lib.param.custom import ClassAttr, ClassDict
from larvaworld.portal.canvas_widgets import (
    EnvironmentCanvas,
    LarvaPreviewFrame,
    env_params_to_canvas_state,
)
from larvaworld.portal.landing_registry import (
    DOCS_EXPERIMENT_TYPES,
    DOCS_SINGLE_EXPERIMENTS,
)
from larvaworld.portal.panel_components import PORTAL_RAW_CSS, build_app_header
from larvaworld.portal.config_widgets.env_widget import build_env_params_widget
from larvaworld.portal.config_widgets.enrichment_widget import build_enrichment_widget
from larvaworld.portal.config_widgets.larvagroup_widget import (
    build_larva_groups_widget,
)
from larvaworld.portal.simulation.parameter_resolution import (
    _builder_obstacle_border_vertices,
    _coerce_xy_sequences,
    _normalize_scalar,
    apply_environment_payload,
    resolve_base_experiment_parameters,
)
from larvaworld.portal.simulation.preview_frames import generate_preview_frames
from larvaworld.portal.workspace import WorkspaceError, get_workspace_dir


__all__ = [
    "_SingleExperimentController",
    "_default_run_name",
    "_editor_group_title",
    "_safe_slug",
    "single_experiment_app",
]


SINGLE_EXPERIMENT_RAW_CSS = """
.lw-single-exp-root {
  padding: 14px 12px 20px 12px;
}

.lw-single-exp-intro {
  border-left: 4px solid #7aa6c2;
  background: rgba(122, 166, 194, 0.16);
  border-radius: 10px;
  padding: 10px 12px;
  margin: 0 0 10px 0;
}

.lw-single-exp-intro a {
  color: #284b63;
}

.lw-single-exp-preview-placeholder {
  padding: 22px 20px;
  border: 1px dashed rgba(17, 17, 17, 0.18);
  border-radius: 12px;
  background: rgba(248, 250, 252, 0.9);
  color: rgba(17, 17, 17, 0.72);
  line-height: 1.55;
}

.lw-single-exp-media {
  border-radius: 10px;
  border: 1px solid rgba(17, 17, 17, 0.08);
  background: rgba(226, 232, 240, 0.28);
  padding: 10px 10px 8px 10px;
}

.lw-single-exp-params-group .bk-input,
.lw-single-exp-params-group textarea,
.lw-single-exp-params-group .bk-input-group {
  font-size: 12px;
}

.lw-single-exp-param-summary {
  font-size: 12px;
  line-height: 1.5;
  color: rgba(17, 17, 17, 0.72);
  background: rgba(241, 245, 249, 0.95);
  border: 1px solid rgba(17, 17, 17, 0.1);
  border-radius: 8px;
  padding: 8px 10px;
}

.lw-single-exp-param-family {
  border-radius: 12px;
  border: 1px solid rgba(17, 17, 17, 0.08);
  padding: 10px 10px 8px 10px;
  margin: 0 0 10px 0;
}

.lw-single-exp-param-family-title {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.01em;
  color: rgba(17, 17, 17, 0.8);
  margin: 0;
}

.lw-single-exp-param-family--arena {
  background: rgba(248, 250, 252, 0.96);
}

.lw-single-exp-param-family--food {
  background: rgba(250, 250, 249, 0.96);
}

.lw-single-exp-param-family--scape {
  background: rgba(249, 248, 250, 0.96);
}

.lw-single-exp-param-family--border {
  background: rgba(248, 250, 252, 0.96);
}

.lw-single-exp-param-family--larva {
  background: rgba(248, 251, 248, 0.96);
}

.lw-single-exp-param-family--life {
  background: rgba(250, 249, 247, 0.96);
}

.lw-single-exp-param-family--odor {
  background: rgba(250, 248, 249, 0.96);
}

.lw-single-exp-param-family--runtime {
  background: rgba(248, 250, 252, 0.96);
}

.lw-single-exp-param-family--enrichment {
  background: rgba(248, 249, 251, 0.96);
}

.lw-single-exp-param-family--trials {
  background: rgba(250, 251, 247, 0.96);
}
""".strip()

_EDITOR_EXCLUDED_PATHS = {"experiment", "parameter_dict"}
_ARENA_GEOMETRY_OPTIONS = ["circular", "rectangular"]
_SPATIAL_DISTRO_MODE_OPTIONS = ["uniform", "normal", "periphery", "grid"]
_SPATIAL_DISTRO_SHAPE_OPTIONS = ["circle", "rect", "oval", "rectangular"]
_ENRICHMENT_MODE_OPTIONS = ["minimal", "full"]
_ENRICHMENT_PROC_KEY_OPTIONS = ["angular", "spatial", "source", "PI", "wind"]
_ENRICHMENT_ANOT_KEY_OPTIONS = [
    "bout_detection",
    "bout_distribution",
    "interference",
    "source_attraction",
    "patch_residency",
]
_ODORSCAPE_OPTIONS = ["Analytical", "Gaussian", "Diffusion"]
_PREVIEW_STEP_CAP = 300
_REGISTRY_ENV_PRESET_PREFIX = "__registry__:"
_WORKSPACE_ENV_PRESET_PREFIX = "__workspace__:"
_NONE_OPTION_LABEL = "None"


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned or "single_experiment"


def _default_run_name(experiment_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{_safe_slug(experiment_id)}_{stamp}"


def _default_experiment_template() -> str | None:
    experiment_ids = list(reg.conf.Exp.confIDs)
    if not experiment_ids:
        return None
    return "dish" if "dish" in experiment_ids else experiment_ids[0]


class _SingleExperimentSelection(param.Parameterized):
    experiment_template = reg.conf.Exp.confID_selector(
        default=_default_experiment_template()
    )
    environment_preset = param.Selector(
        default="__template__",
        objects={"Template default environment": "__template__"},
        doc=(
            "Environment preset from the template default, workspace JSON presets, "
            "or registry Env configurations."
        ),
    )

    def __init__(
        self,
        *,
        experiment_ids: list[str],
        default_experiment: str,
        environment_options: dict[str, str] | None = None,
    ) -> None:
        super().__init__(experiment_template=default_experiment)
        self.param["experiment_template"].objects = experiment_ids
        self.param["experiment_template"].label = "Experiment template"
        self.param["environment_preset"].label = "Environment preset"
        if environment_options is not None:
            self.set_environment_options(environment_options)

    def set_environment_options(self, options: dict[str, str]) -> None:
        self.param["environment_preset"].objects = options
        if self.environment_preset not in options.values():
            self.environment_preset = "__template__"


def _editor_group_title(key: str) -> str:
    return key.replace("_", " ").title()


def _editor_field_title(path: str) -> str:
    return path.replace("_", " ").replace(".", " / ").title()


def _title_case_name(value: str) -> str:
    return value.replace("_", " ").title()


def _family_spec_for_path(path: str) -> tuple[str, str, str]:
    if path.startswith("env_params.arena."):
        return "env_arena", "Arena", "arena"
    if path == "env_params.food_params.source_units":
        return "env_food_source_units", "Source Units", "food"
    if path == "env_params.food_params.source_groups":
        return "env_food_source_groups", "Source Groups", "food"
    if path == "env_params.food_params.food_grid":
        return "env_food_grid", "Food Grid", "food"
    if path.startswith("env_params.food_params.source_units."):
        return "env_food_source_units", "Source Units", "food"
    if path.startswith("env_params.food_params.source_groups."):
        return "env_food_source_groups", "Source Groups", "food"
    if path.startswith("env_params.food_params.food_grid"):
        return "env_food_grid", "Food Grid", "food"
    if path.startswith("env_params.food_params."):
        return "env_food", "Food Params", "food"
    if path == "env_params.odorscape":
        return "env_odorscape", "Odorscape", "scape"
    if path == "env_params.windscape":
        return "env_windscape", "Windscape", "scape"
    if path == "env_params.thermoscape":
        return "env_thermoscape", "Thermoscape", "scape"
    if path.startswith("env_params.odorscape."):
        return "env_odorscape", "Odorscape", "scape"
    if path.startswith("env_params.windscape."):
        return "env_windscape", "Windscape", "scape"
    if path.startswith("env_params.thermoscape."):
        return "env_thermoscape", "Thermoscape", "scape"
    if path == "env_params.border_list":
        return "env_borders", "Borders", "border"
    if path.startswith("env_params.border_list."):
        return "env_borders", "Borders", "border"
    if path.startswith("larva_groups."):
        parts = path.split(".")
        group_id = parts[1]
        family = parts[2] if len(parts) > 2 else "core"
        if family == "distribution":
            return (
                f"larva_{group_id}_distribution",
                f"{group_id} / Distribution",
                "larva",
            )
        if family == "life_history":
            return (
                f"larva_{group_id}_life_history",
                f"{group_id} / Life History",
                "life",
            )
        if family == "odor":
            return (f"larva_{group_id}_odor", f"{group_id} / Odor", "odor")
        return (f"larva_{group_id}_core", f"{group_id} / Identity", "larva")
    if path.startswith("enrichment.pre_kws."):
        return "enrichment_preprocessing", "Enrichment / Preprocessing", "enrichment"
    if path.startswith("enrichment."):
        return "enrichment", "Enrichment", "enrichment"
    if path.startswith("trials."):
        return "trials", "Trials", "trials"
    if path in {
        "dt",
        "duration",
        "fr",
        "Nsteps",
        "Box2D",
        "constant_framerate",
        "larva_collisions",
        "collections",
        "parameter_dict",
    }:
        return "runtime", "Runtime / Outputs", "runtime"
    return (
        path.split(".", 1)[0],
        _editor_group_title(path.split(".", 1)[0]),
        "runtime",
    )


def _display_label_for_path(path: str) -> str:
    if path == "env_params.food_params.source_groups":
        return "Source Groups"
    if path == "env_params.food_params.source_units":
        return "Source Units"
    if path == "env_params.food_params.food_grid":
        return "Food Grid"
    if path == "env_params.odorscape":
        return "Odorscape"
    if path == "env_params.windscape":
        return "Windscape"
    if path == "env_params.thermoscape":
        return "Thermoscape"
    if path == "env_params.border_list":
        return "Borders"
    if path.startswith("env_params.arena."):
        leaf = path.split(".")[-1]
        return {
            "geometry": "Geometry",
            "dims": "Dimensions",
            "torus": "Torus",
        }.get(leaf, _title_case_name(leaf))
    if path.startswith("env_params.border_list."):
        return {
            "color": "Color",
            "group": "Group",
            "unique_id": "Unique ID",
            "vertices": "Vertices",
            "width": "Width",
        }.get(path.split(".")[-1], _title_case_name(path.split(".")[-1]))
    if path.startswith("larva_groups."):
        parts = path.split(".")
        if len(parts) >= 4 and parts[2] == "distribution":
            leaf = parts[-1]
            return {
                "N": "Larvae count",
                "loc": "Start position",
                "scale": "Spread",
                "mode": "Placement mode",
                "shape": "Placement shape",
                "orientation_range": "Initial orientation range",
            }.get(leaf, _title_case_name(leaf))
        if len(parts) >= 4 and parts[2] == "life_history":
            leaf = parts[-1]
            return {
                "age": "Initial age",
                "epochs": "Epochs",
                "reach_pupation": "Reach pupation",
            }.get(leaf, _title_case_name(leaf))
        if len(parts) >= 4 and parts[2] == "odor":
            leaf = parts[-1]
            return {
                "id": "Odor ID",
                "intensity": "Odor intensity",
                "spread": "Odor spread",
            }.get(leaf, _title_case_name(leaf))
        if len(parts) >= 3:
            leaf = parts[-1]
            return {
                "group_id": "Group ID",
                "model": "Model",
                "sample": "Sample",
                "color": "Color",
                "imitation": "Imitation",
            }.get(leaf, _title_case_name(leaf))
    if path.startswith("env_params.food_params.source_units.") or path.startswith(
        "env_params.food_params.source_groups."
    ):
        leaf = path.split(".")[-1]
        return {
            "group_id": "Group ID",
            "unique_id": "Unique ID",
            "odor": "Odor",
            "substrate": "Substrate",
        }.get(leaf, _title_case_name(leaf))
    if path.startswith("env_params.food_params.food_grid."):
        return _title_case_name(path.split(".")[-1])
    if (
        path.startswith("env_params.odorscape.")
        or path.startswith("env_params.windscape.")
        or path.startswith("env_params.thermoscape.")
    ):
        return _title_case_name(path.split(".")[-1])
    if path.startswith("enrichment.pre_kws."):
        return _title_case_name(path.split(".")[-1])
    if path.startswith("enrichment."):
        return _title_case_name(path.split(".")[-1])
    if path.startswith("trials."):
        return _title_case_name(path.split(".")[-1])
    return _editor_field_title(path)


def _sequence_component_labels(
    path: str, value: list[Any] | tuple[Any, ...]
) -> list[str]:
    length = len(value)
    if path.endswith((".dims", ".loc", ".scale")) and length == 2:
        return ["x", "y"]
    if path.endswith((".orientation_range", ".age_range")) and length == 2:
        return ["min", "max"]
    if path == "enrichment.dsp_starts":
        return [f"Start {index}" for index in range(1, length + 1)]
    if path == "enrichment.dsp_stops":
        return [f"Stop {index}" for index in range(1, length + 1)]
    if path == "enrichment.tor_durs":
        return [f"Duration {index}" for index in range(1, length + 1)]
    return [f"Value {index}" for index in range(1, length + 1)]


def _attach_scroll_restore(button: pn.widgets.Button) -> None:
    button._lw_scroll_restore_callback = button.jscallback(
        clicks="""
        const scrollY = window.scrollY || window.pageYOffset || 0;
        [40, 140, 320, 700].forEach((delay) => {
          setTimeout(() => {
            window.scrollTo({top: scrollY, left: 0, behavior: 'auto'});
          }, delay);
        });
        """
    )


def _is_absent_optional_value(value: Any, absent_value: Any) -> bool:
    if absent_value == "empty_dict":
        return value == "empty_dict"
    if isinstance(absent_value, list):
        return value == absent_value
    return value is None


def _summary_label(text: str) -> str:
    return f'<div class="lw-single-exp-param-summary">{text}</div>'


def _field_label_html(label: str) -> str:
    escaped_label = html.escape(label)
    return f'<div class="lw-single-exp-param-family-title">{escaped_label}</div>'


def _apply_widget_help(
    widget: pn.viewable.Viewable, label: str, help_text: str | None
) -> pn.viewable.Viewable:
    if hasattr(widget, "name"):
        widget.name = label
    if help_text and hasattr(widget, "description"):
        widget.description = help_text
    return widget


def _optional_family_control(
    path: str, enabled: bool, help_text: str | None
) -> pn.viewable.Viewable:
    widget = pn.widgets.Switch(
        name="",
        value=enabled,
        width=19,
        margin=0,
    )
    if help_text and hasattr(widget, "description"):
        widget.description = help_text
    return widget


def _family_title_row(
    title: str, toggle: pn.viewable.Viewable | None
) -> pn.viewable.Viewable:
    if toggle is None:
        return pn.pane.HTML(
            _field_label_html(title),
            margin=(0, 0, 8, 0),
        )
    grid = pn.GridSpec(
        ncols=12,
        nrows=1,
        sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )
    grid[0, 0:11] = pn.pane.HTML(_field_label_html(title), margin=0)
    grid[0, 11] = pn.Row(toggle, align="end", margin=0)
    return grid


def _json_ready(value: Any) -> Any:
    nested_conf = getattr(type(value), "nestedConf", None)
    if nested_conf is not None:
        return _json_ready(value.nestedConf)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _join_help_parts(*parts: str | None) -> str | None:
    cleaned = []
    seen = set()
    for part in parts:
        if not part or not part.strip():
            continue
        normalized = part.strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    if not cleaned:
        return None
    return "\n\n".join(cleaned)


class _ExperimentPreview:
    _FORWARD_ONLY_MESSAGE = (
        "Preview is forward-only; prepare the preview again to replay from the start."
    )

    def __init__(
        self,
        launcher: sim.ExpRun,
        *,
        size: int = 620,
        preview_step_cap: int = _PREVIEW_STEP_CAP,
        launcher_ready: bool = False,
    ) -> None:
        self.launcher = launcher
        self.size = size
        self.draw_ops = screen.AgentDrawOps(draw_centroid=True, draw_segs=False)
        preview_steps = max(2, min(int(self.launcher.p.steps), preview_step_cap))
        if not launcher_ready:
            self.launcher.sim_setup(steps=preview_steps)
        self.Nfade = max(1, int(self.draw_ops.trail_dt / self.launcher.dt))
        self.env = self.launcher.p.env_params
        xdim, ydim = self.env.arena.dims
        self.image_kws = {
            "title": "Arena preview",
            "xlim": (-xdim / 2, xdim / 2),
            "ylim": (-ydim / 2, ydim / 2),
            "width": self.size,
            "height": int(self.size * ydim / xdim) if xdim else self.size,
            "xlabel": "X (m)",
            "ylabel": "Y (m)",
        }
        self.progress_bar = pn.widgets.Progress(
            name="Simulation timestep",
            bar_color="primary",
            width=int(self.size / 2),
            max=self.launcher.Nsteps - 1,
            value=self.launcher.t,
        )
        self.time_slider = pn.widgets.IntSlider(
            name="Tick",
            width=int(self.size / 2),
            start=0,
            end=preview_steps - 1,
            step=1,
            value=0,
        )
        self.forward_only_note = pn.pane.HTML("", sizing_mode="stretch_width")
        self._syncing_slider = False
        self.tank_plot = self._get_tank_plot()
        self.static_overlay = self._build_static_overlay()

    def _get_tank_plot(self) -> hv.element.Overlay:
        arena = self.env.arena
        if arena.geometry == "circular":
            return hv.Ellipse(0, 0, arena.dims[0]).opts(
                line_width=5,
                bgcolor="lightgrey",
            )
        if arena.geometry == "rectangular":
            return hv.Box(0, 0, spec=arena.dims).opts(
                line_width=5,
                bgcolor="lightgrey",
            )
        raise ValueError(f"Unsupported arena geometry: {arena.geometry}")

    def _source_layers(self) -> list[Any]:
        return [
            hv.Ellipse(source.pos[0], source.pos[1], source.radius * 2).opts(
                line_width=5,
                color=source.color,
                bgcolor=source.color,
            )
            for source in self.launcher.sources
        ]

    def _odor_layers(self) -> list[Any]:
        odor_layers = []
        for source in self.launcher.sources:
            odor = getattr(source, "odor", None)
            spread = getattr(odor, "spread", None)
            odor_id = getattr(odor, "id", None)
            if odor_id in {None, ""} or spread in {None, ""}:
                continue
            try:
                spread_value = float(spread)
            except Exception:
                continue
            if spread_value <= 0:
                continue
            odor_layers.append(
                hv.Ellipse(source.pos[0], source.pos[1], spread_value * 2).opts(
                    line_width=2,
                    color=source.color,
                    alpha=0.22,
                )
            )
        return odor_layers

    def _border_layers(self) -> list[Any]:
        border_layers = []
        for border in getattr(self.launcher, "borders", []):
            line_width = max(1.0, float(getattr(border, "width", 0.001)) * 1800)
            for segment in getattr(border, "border_xy", []):
                border_layers.append(
                    hv.Path(segment).opts(
                        color=getattr(border, "color", "#d94841"),
                        line_width=line_width,
                    )
                )
        return border_layers

    def _build_static_overlay(self) -> hv.Overlay:
        return hv.Overlay(
            [self.tank_plot]
            + self._odor_layers()
            + self._source_layers()
            + self._border_layers()
        )

    def _dynamic_agent_layers(self) -> list[Any]:
        agents = self.launcher.agents
        layers = []
        if self.draw_ops.draw_segs:
            layers.append(
                hv.Overlay(
                    [
                        hv.Polygons([seg.vertices for seg in agent.segs]).opts(
                            color=agent.color
                        )
                        for agent in agents
                    ]
                )
            )
        if self.draw_ops.draw_centroid:
            layers.append(
                hv.Points(agents.get_position()).opts(
                    size=5,
                    color="black",
                )
            )
        if self.draw_ops.draw_head:
            layers.append(
                hv.Points(agents.head.front_end).opts(
                    size=5,
                    color="red",
                )
            )
        if self.draw_ops.draw_midline:
            layers.append(
                hv.Overlay(
                    [
                        hv.Path(agent.midline_xy).opts(color="blue", line_width=2)
                        for agent in agents
                    ]
                )
            )
        if self.draw_ops.visible_trails:
            layers.append(
                hv.Contours([agent.trajectory[-self.Nfade :] for agent in agents]).opts(
                    color="black"
                )
            )
        return layers

    def _draw_overlay(self) -> hv.Overlay:
        agent_layers = self._dynamic_agent_layers()
        overlay = self.static_overlay
        if agent_layers:
            overlay = overlay * hv.Overlay(agent_layers)
        return overlay.opts(
            responsive=False,
            **self.image_kws,
        )

    def _sync_slider_to_current_tick(self) -> None:
        self._syncing_slider = True
        try:
            self.time_slider.value = self.launcher.t
        finally:
            self._syncing_slider = False

    def _image_for_tick(self, i: int) -> hv.Overlay:
        if self._syncing_slider:
            return self._draw_overlay()
        if i < self.launcher.t:
            self.forward_only_note.object = self._FORWARD_ONLY_MESSAGE
            self._sync_slider_to_current_tick()
            self.progress_bar.value = self.launcher.t
            return self._draw_overlay()
        self.forward_only_note.object = ""
        if i == self.launcher.t:
            return self._draw_overlay()
        while i > self.launcher.t:
            self.launcher.sim_step()
        self.progress_bar.value = self.launcher.t
        return self._draw_overlay()

    def view(self) -> pn.viewable.Viewable:
        @pn.depends(i=self.time_slider)
        def _image(i: int) -> hv.Overlay:
            return self._image_for_tick(i)

        preview = hv.DynamicMap(_image)
        return pn.Row(
            preview,
            pn.Column(
                pn.Row(pn.Column("Tick", self.time_slider)),
                pn.Row(pn.Column("Computed timestep", self.progress_bar)),
                self.forward_only_note,
                pn.Param(self.draw_ops),
            ),
            sizing_mode="stretch_width",
        )


class _FrameSimulationPreview:
    def __init__(
        self,
        *,
        canvas: EnvironmentCanvas,
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


class _SingleExperimentController:
    def __init__(self) -> None:
        experiment_ids = list(reg.conf.Exp.confIDs)
        default_experiment = "dish" if "dish" in experiment_ids else experiment_ids[0]
        self.selection = _SingleExperimentSelection(
            experiment_ids=experiment_ids,
            default_experiment=default_experiment,
        )
        self.experiment = pn.widgets.Select.from_param(
            self.selection.param.experiment_template
        )
        self.run_name = pn.widgets.TextInput(
            name="Run name",
            value=_default_run_name(self.selection.experiment_template),
        )
        self.environment_select = pn.widgets.Select.from_param(
            self.selection.param.environment_preset
        )
        self.refresh_environments_btn = pn.widgets.Button(
            name="Refresh environment",
            button_type="default",
        )
        self.prepare_btn = pn.widgets.Button(
            name="Arena Preview",
            button_type="primary",
        )
        self.simulation_preview_btn = pn.widgets.Button(
            name="Generate simulation preview",
            button_type="default",
        )
        self.run_btn = pn.widgets.Button(
            name="Run experiment",
            button_type="success",
        )
        self.preview_frames_input = pn.widgets.IntInput(
            name="Preview frames",
            value=_PREVIEW_STEP_CAP,
            start=1,
            end=1000,
            step=50,
        )
        self.save_video = pn.widgets.Checkbox(name="Save video", value=False)
        self.video_filename = pn.widgets.TextInput(
            name="Video filename",
            value=self.run_name.value,
            disabled=True,
        )
        self.video_fps = pn.widgets.IntInput(
            name="Video speed-up",
            value=1,
            step=1,
            start=1,
            end=120,
            disabled=True,
            description=(
                "Controls video playback speed, not the encoder frame rate. "
                "1x keeps the saved video close to simulated real time. 2x plays "
                "twice as fast and makes the video about half as long; 5x plays "
                "five times as fast and makes it about one fifth as long."
            ),
        )
        self.show_display = pn.widgets.Checkbox(name="Show display", value=False)
        self.display_every_n_steps = pn.widgets.IntInput(
            name="Display every N steps",
            value=1,
            step=1,
            start=1,
            end=20,
            disabled=True,
            description=(
                "Live display redraw cadence. 1 updates every simulation step; "
                "higher values redraw less often to reduce display overhead."
            ),
        )
        self.prepare_btn.width = None
        self.prepare_btn.sizing_mode = "stretch_width"
        self.simulation_preview_btn.width = None
        self.simulation_preview_btn.sizing_mode = "stretch_width"
        self.run_btn.width = None
        self.run_btn.sizing_mode = "stretch_width"
        self.preview_frames_input.width = None
        self.preview_frames_input.sizing_mode = "stretch_width"
        self.preview_action_row = pn.Row(
            self.prepare_btn,
            sizing_mode="stretch_width",
            margin=0,
        )
        self.preview_options_row = pn.Row(
            self.preview_frames_input,
            sizing_mode="stretch_width",
            margin=0,
        )
        self.preview_generate_row = pn.Row(
            self.simulation_preview_btn,
            sizing_mode="stretch_width",
            margin=(4, 0, 0, 0),
        )
        self.execution_action_row = pn.Row(
            self.run_btn,
            sizing_mode="stretch_width",
            margin=(6, 0, 0, 0),
        )
        self.summary = pn.pane.HTML(
            "", sizing_mode="stretch_width", margin=(0, 0, 4, 0)
        )
        self.parameter_group = pn.widgets.Select(
            name="Parameter group",
            options=[],
            value=None,
        )
        self.parameters_editor = pn.Column(
            sizing_mode="stretch_width",
            margin=0,
            styles={
                "font-size": "12px",
                "line-height": "1.45",
            },
        )
        self._parameter_groups: dict[str, list[str]] = {}
        self._parameter_widgets: dict[str, tuple[str, Any]] = {}
        self._parameter_widget_specs: dict[
            str, tuple[str, Any, pn.viewable.Viewable]
        ] = {}
        self._typed_experiment_for_larva_groups: Any | None = None
        self._larva_groups_group_view: pn.viewable.Viewable | None = None
        self._typed_experiment_for_enrichment: Any | None = None
        self._enrichment_group_view: pn.viewable.Viewable | None = None
        self._typed_experiment_for_env_params: Any | None = None
        self._env_params_group_view: pn.viewable.Viewable | None = None
        self._editor_context: dict[str, list[str]] = {"odor_ids": []}
        self._parameter_seed_overrides = util.AttrDict()
        self._optional_family_meta: dict[str, dict[str, Any]] = {}
        self.summary.styles = {
            "font-size": "12px",
            "line-height": "1.55",
            "color": "rgba(17, 17, 17, 0.76)",
            "padding": "0 6px 0 12px",
        }
        self.status = pn.pane.Markdown(
            "",
            styles={
                "font-size": "12px",
                "line-height": "1.55",
            },
        )
        self.preview_meta = pn.pane.HTML(
            "",
            styles={
                "font-size": "12px",
                "line-height": "1.55",
                "color": "rgba(17, 17, 17, 0.78)",
                "background": "rgba(193, 176, 194, 0.12)",
                "border-left": "3px solid #c1b0c2",
                "border-radius": "8px",
                "padding": "8px 10px",
            },
            margin=(0, 0, 8, 0),
        )
        self.preview = pn.Column(
            pn.pane.HTML(
                (
                    '<div class="lw-single-exp-preview-placeholder">'
                    "Choose an experiment template, optionally apply a workspace environment preset, "
                    "and prepare the configuration preview here."
                    "</div>"
                ),
                margin=0,
            ),
            sizing_mode="stretch_width",
        )

        self.selection.param.watch(self._on_experiment_change, "experiment_template")
        self.run_name.param.watch(self._on_run_name_change, "value")
        self.selection.param.watch(
            self._on_parameter_override_change, "environment_preset"
        )
        self.parameter_group.param.watch(self._on_parameter_group_change, "value")
        self.save_video.param.watch(self._on_save_video_change, "value")
        self.show_display.param.watch(self._on_show_display_change, "value")
        self.refresh_environments_btn.on_click(self._on_refresh_environments)
        self.prepare_btn.on_click(self._on_prepare_preview)
        self.simulation_preview_btn.on_click(self._on_generate_simulation_preview)
        self.run_btn.on_click(self._on_run_experiment)

        self._on_show_display_change()
        self._refresh_environment_options()
        self._refresh_summary()
        self._refresh_parameter_editor()
        self.status.object = "Select a template and prepare a single-run preview."

    def _environment_dir(self) -> Path:
        return get_workspace_dir("environments")

    def _experiment_dir(self) -> Path:
        return get_workspace_dir("experiments")

    def _selected_experiment(self) -> str:
        return str(self.selection.experiment_template)

    def _environment_options(self) -> dict[str, str]:
        options = {"Template default environment": "__template__"}
        preset_dir = self._environment_dir()
        preset_dir.mkdir(parents=True, exist_ok=True)
        workspace_labels: set[str] = set()
        for path in sorted(preset_dir.glob("*.json")):
            workspace_labels.add(path.stem)
            options[path.stem] = path.name
        registry_options = {}
        for name in sorted(str(key) for key in reg.conf.Env.dict.keys()):
            if name in workspace_labels:
                continue
            registry_options[f"Registry / {name}"] = (
                f"{_REGISTRY_ENV_PRESET_PREFIX}{name}"
            )
        options.update(registry_options)
        return options

    def _load_selected_environment(self) -> util.AttrDict | None:
        selected = self.selection.environment_preset
        if selected in {None, "", "__template__"}:
            return None
        if str(selected).startswith(_WORKSPACE_ENV_PRESET_PREFIX):
            filename = str(selected)[len(_WORKSPACE_ENV_PRESET_PREFIX) :]
            preset_path = self._environment_dir() / filename
            payload = json.loads(preset_path.read_text(encoding="utf-8"))
            return util.AttrDict(payload)
        if str(selected).startswith(_REGISTRY_ENV_PRESET_PREFIX):
            registry_id = str(selected)[len(_REGISTRY_ENV_PRESET_PREFIX) :]
            return util.AttrDict(reg.conf.Env.getID(registry_id)).get_copy()
        preset_path = self._environment_dir() / str(selected)
        payload = json.loads(preset_path.read_text(encoding="utf-8"))
        return util.AttrDict(payload)

    def _selected_environment_label(self) -> str:
        selected = self.selection.environment_preset
        if selected == "__template__":
            return "template default"
        if selected is not None and str(selected).startswith(
            _WORKSPACE_ENV_PRESET_PREFIX
        ):
            filename = str(selected)[len(_WORKSPACE_ENV_PRESET_PREFIX) :]
            return Path(filename).stem
        if selected is not None and str(selected).startswith(
            _REGISTRY_ENV_PRESET_PREFIX
        ):
            registry_id = str(selected)[len(_REGISTRY_ENV_PRESET_PREFIX) :]
            return f"registry / {registry_id}"
        return str(selected)

    @staticmethod
    def _apply_environment_payload(
        env_params: util.AttrDict,
        environment_payload: util.AttrDict,
    ) -> util.AttrDict:
        return apply_environment_payload(env_params, environment_payload)

    def _build_parameters(self) -> util.AttrDict:
        parameters = resolve_base_experiment_parameters(
            self._selected_experiment(),
            self._load_selected_environment(),
        )
        flat = self._merge_seed_overrides(parameters.flatten())
        for path, (kind, widget) in self._parameter_widgets.items():
            if kind == "toggle_factory":
                continue
            flat[path] = self._parse_widget_value(kind, widget)
        flat = self._apply_optional_family_states(flat)
        resolved = util.AttrDict(flat.unflatten())
        if self._typed_experiment_for_larva_groups is not None:
            nested_conf = util.AttrDict(
                self._typed_experiment_for_larva_groups.nestedConf
            )
            larva_groups = nested_conf.get("larva_groups")
            if larva_groups is not None:
                resolved["larva_groups"] = util.AttrDict(larva_groups)
        if self._typed_experiment_for_enrichment is not None:
            nested_conf = util.AttrDict(
                self._typed_experiment_for_enrichment.nestedConf
            )
            enrichment = nested_conf.get("enrichment")
            if enrichment is not None:
                resolved["enrichment"] = util.AttrDict(enrichment)
        if self._typed_experiment_for_env_params is not None:
            nested_conf = util.AttrDict(
                self._typed_experiment_for_env_params.nestedConf
            )
            env_params = nested_conf.get("env_params")
            if env_params is not None:
                resolved["env_params"] = util.AttrDict(env_params)
        return util.AttrDict(_coerce_xy_sequences(resolved))

    def _resolve_experiment_parameters(self) -> util.AttrDict:
        return self._build_parameters()

    def _refresh_environment_options(self) -> None:
        try:
            options = self._environment_options()
        except WorkspaceError as exc:
            self.selection.set_environment_options(
                {"Workspace unavailable": "__template__"}
            )
            self.selection.environment_preset = "__template__"
            self.environment_select.disabled = True
            self.refresh_environments_btn.disabled = True
            self.status.object = f"Cannot load workspace environment presets: {exc}"
            return
        selected = self.selection.environment_preset
        self.selection.set_environment_options(options)
        self.environment_select.disabled = False
        self.refresh_environments_btn.disabled = False
        self.selection.environment_preset = (
            selected if selected in options.values() else "__template__"
        )

    def _refresh_summary(self) -> None:
        experiment = self._selected_experiment()
        parameters = reg.conf.Exp.getID(experiment).get_copy()
        larva_groups = list(parameters.get("larva_groups", {}).keys())
        env = util.AttrDict(parameters.env_params)
        epochs = parameters.get("trials", {}).get("epochs", {})
        self.summary.object = (
            '<div class="lw-single-exp-summary">'
            f"<strong>Template:</strong> <code>{experiment}</code><br>"
            f"<strong>Default duration:</strong> {float(parameters.get('duration', 0.0)):.2f} min<br>"
            f"<strong>Arena geometry:</strong> {env.arena.geometry}<br>"
            f"<strong>Larva groups:</strong> {', '.join(larva_groups) if larva_groups else 'None'}<br>"
            f"<strong>Epochs:</strong> {len(epochs)}<br>"
            "<strong>Parameter editing:</strong> all resolved experiment parameters are editable below."
            "</div>"
        )

    def _editable_flat_parameters(self) -> util.AttrDict:
        parameters = resolve_base_experiment_parameters(
            self._selected_experiment(),
            self._load_selected_environment(),
        )
        flat = parameters.flatten()
        flat = self._merge_seed_overrides(flat)
        flat = self._augment_optional_family_entries(flat)
        filtered = util.AttrDict()
        for path, value in flat.items():
            if path in _EDITOR_EXCLUDED_PATHS:
                continue
            filtered[path] = _normalize_scalar(value)
        return filtered

    def _merge_seed_overrides(self, flat: util.AttrDict) -> util.AttrDict:
        merged = util.AttrDict(flat)
        seed_keys = list(self._parameter_seed_overrides.keys())
        for seed_key in seed_keys:
            parts = seed_key.split(".")
            for idx in range(1, len(parts)):
                prefix = ".".join(parts[:idx])
                merged.pop(prefix, None)
        merged.update(self._parameter_seed_overrides)
        return merged

    def _optional_family_specs(
        self, flat: util.AttrDict | None = None
    ) -> dict[str, dict[str, Any]]:
        from larvaworld.lib.reg.generators import gen

        specs = {
            "env_params.odorscape": {
                "absent": None,
                "payload": gen.GaussianValueLayer().nestedConf,
            },
            "env_params.windscape": {
                "absent": None,
                "payload": gen.WindScape().nestedConf,
            },
            "env_params.thermoscape": {
                "absent": None,
                "payload": gen.ThermoScape().nestedConf,
            },
            "env_params.food_params.food_grid": {
                "absent": None,
                "payload": gen.FoodGrid().nestedConf,
            },
            "env_params.food_params.source_groups": {
                "absent": "empty_dict",
                "payload": gen.FoodGroup().entry("SourceGroup"),
            },
            "env_params.food_params.source_units": {
                "absent": "empty_dict",
                "payload": gen.Food().entry("Source"),
            },
            "env_params.border_list": {
                "absent": "empty_dict",
                "payload": gen.Border(
                    vertices=[(-0.01, 0.0), (0.01, 0.0)],
                    width=0.001,
                ).entry("Border"),
            },
        }
        if flat is not None:
            windscape_enabled = flat.get("env_params.windscape", None) is not None
            if windscape_enabled or any(
                key.startswith("env_params.windscape.")
                for key in flat.keys()
                if key != "env_params.windscape.puffs"
            ):
                specs["env_params.windscape.puffs"] = {
                    "absent": "empty_dict",
                    "payload": {"Puff": gen.AirPuff().nestedConf},
                }
            epoch_payload = [gen.Epoch().nestedConf]
            if "trials.epochs" in flat:
                specs["trials.epochs"] = {
                    "absent": [],
                    "payload": epoch_payload,
                }
            for path in flat.keys():
                if path.startswith("larva_groups.") and path.endswith(
                    ".life_history.epochs"
                ):
                    specs[path] = {
                        "absent": [],
                        "payload": epoch_payload,
                    }
        return specs

    def _augment_optional_family_entries(self, flat: util.AttrDict) -> util.AttrDict:
        augmented = util.AttrDict(flat)
        self._optional_family_meta = {}
        for root, spec in self._optional_family_specs(augmented).items():
            descendants = {
                key: value
                for key, value in augmented.items()
                if key.startswith(f"{root}.")
            }
            root_value = augmented.get(root, spec["absent"])
            enabled = bool(descendants) or not _is_absent_optional_value(
                root_value, spec["absent"]
            )
            self._optional_family_meta[root] = {
                "absent": spec["absent"],
                "payload": spec["payload"],
                "enabled": enabled,
            }
            augmented[root] = root_value
            if not descendants:
                for key, value in self._flatten_seed_payload(
                    root, spec["payload"]
                ).items():
                    if key not in augmented:
                        augmented[key] = value
        return augmented

    def _optional_root_for_path(self, path: str) -> str | None:
        for root in sorted(self._optional_family_meta.keys(), key=len, reverse=True):
            if path == root:
                return root
            if path.startswith(f"{root}."):
                return root
        return None

    @staticmethod
    def _set_control_disabled(kind: str, control: Any, disabled: bool) -> None:
        if kind in {"readonly", "factory", "toggle_factory"}:
            return
        if kind == "sequence":
            for widget in control["widgets"]:
                widget.disabled = disabled
            return
        if kind == "optional_float":
            control["enabled"].disabled = disabled
            control["widget"].disabled = disabled or not bool(control["enabled"].value)
            return
        if hasattr(control, "disabled"):
            control.disabled = disabled

    @staticmethod
    def _optional_toggle_enabled_value(widget: Any) -> bool:
        return bool(getattr(widget, "value", None))

    def _apply_optional_family_disabled_state(self, root: str, enabled: bool) -> None:
        for path, (kind, control) in self._parameter_widgets.items():
            if path == root or not path.startswith(f"{root}."):
                continue
            self._set_control_disabled(kind, control, not enabled)

    def _wire_optional_family_toggles(self) -> None:
        for root in self._optional_family_meta:
            kind, control = self._parameter_widgets.get(root, (None, None))
            if kind != "toggle_factory":
                continue
            enabled_widget = control["enabled"]
            self._apply_optional_family_disabled_state(
                root, self._optional_toggle_enabled_value(enabled_widget)
            )

            def _sync(event: Any, root: str = root) -> None:
                self._apply_optional_family_disabled_state(root, bool(event.new))

            enabled_widget.param.watch(_sync, "value")

    def _apply_optional_family_states(self, flat: util.AttrDict) -> util.AttrDict:
        updated = util.AttrDict(flat)
        for root, meta in self._optional_family_meta.items():
            kind, control = self._parameter_widgets.get(root, (None, None))
            if kind != "toggle_factory":
                continue
            enabled = self._optional_toggle_enabled_value(control["enabled"])
            descendant_keys = [
                key for key in list(updated.keys()) if key.startswith(f"{root}.")
            ]
            if enabled:
                if descendant_keys:
                    updated.pop(root, None)
                elif _is_absent_optional_value(
                    updated.get(root, meta["absent"]), meta["absent"]
                ):
                    updated[root] = _normalize_scalar(meta["payload"])
                continue
            for key in descendant_keys:
                updated.pop(key, None)
            updated[root] = meta["absent"]
        return updated

    @staticmethod
    def _runtime_note_for_path(path: str) -> str | None:
        if path == "duration":
            return "Runtime: total simulated duration in minutes."
        if path == "dt":
            return "Runtime: simulation timestep in seconds."
        if path.startswith("env_params."):
            return "Runtime: consumed during ExpRun.setup() when build_env(self.p.env_params) constructs the environment."
        if path.startswith("larva_groups."):
            return "Runtime: consumed during ExpRun.build_agents() to generate and place larva groups."
        if path == "collections" or path.startswith("collections."):
            return "Runtime: consumed during ExpRun.setup() when set_collectors(self.p.collections) defines recorded outputs."
        if path.startswith("trials.epochs"):
            return "Runtime: converted in ExpRun.setup() from age ranges to start/stop simulation steps."
        if path == "enrichment" or path.startswith("enrichment."):
            return "Runtime: applied after simulate() completes, during post-simulation dataset enrichment."
        if path == "larva_collisions":
            return "Runtime: when disabled, ExpRun.setup() runs overlap elimination before the simulation starts."
        return None

    @staticmethod
    def _resolve_doc_from_class(cls: type[Any], parts: list[str]) -> str | None:
        if not hasattr(cls, "param") or not parts:
            return None
        objects = cls.param.objects(instance=False)
        name = parts[0]
        if name not in objects:
            return None
        p = objects[name]
        current_doc = getattr(p, "doc", None)
        if len(parts) == 1:
            return current_doc
        rest = parts[1:]
        if isinstance(p, ClassDict):
            if not rest:
                return current_doc
            item_cls = p.item_type
            if item_cls is None or len(rest) < 2:
                return current_doc
            nested_doc = _SingleExperimentController._resolve_doc_from_class(
                item_cls, rest[1:]
            )
            return nested_doc or current_doc
        if isinstance(p, ClassAttr):
            nested_cls = p.class_[0] if isinstance(p.class_, tuple) else p.class_
            nested_doc = _SingleExperimentController._resolve_doc_from_class(
                nested_cls, rest
            )
            return nested_doc or current_doc
        if name == "trials" and rest and rest[0] == "epochs":
            from larvaworld.lib.reg.generators import gen

            epoch_doc = getattr(
                gen.Epoch.param.objects(instance=False)["age_range"], "doc", None
            )
            return _join_help_parts(current_doc, epoch_doc)
        return current_doc

    @staticmethod
    def _param_doc_for_path(path: str) -> str | None:
        from larvaworld.lib.param.enrichment import EnrichConf
        from larvaworld.lib.reg.generators import EnvConf, ExpConf
        from larvaworld.lib.reg.larvagroup import LarvaGroup

        if path.startswith("env_params."):
            return _SingleExperimentController._resolve_doc_from_class(
                EnvConf, path.split(".")[1:]
            )
        if path.startswith("larva_groups."):
            return _SingleExperimentController._resolve_doc_from_class(
                LarvaGroup, path.split(".")[2:]
            )
        if path.startswith("enrichment."):
            return _SingleExperimentController._resolve_doc_from_class(
                EnrichConf, path.split(".")[1:]
            )
        return _SingleExperimentController._resolve_doc_from_class(
            ExpConf, path.split(".")
        )

    @staticmethod
    def _help_text_for_path(path: str) -> str | None:
        return _join_help_parts(
            _SingleExperimentController._param_doc_for_path(path),
            _SingleExperimentController._runtime_note_for_path(path),
        )

    @staticmethod
    def _factory_spec_for_path(path: str, value: Any) -> tuple[str, Any] | None:
        from larvaworld.lib.reg.generators import gen

        if path == "env_params.odorscape" and value is None:
            return "Add odorscape", gen.GaussianValueLayer().nestedConf
        if path == "env_params.windscape" and value is None:
            return "Add windscape", gen.WindScape().nestedConf
        if path == "env_params.thermoscape" and value is None:
            return "Add thermoscape", gen.ThermoScape().nestedConf
        if path == "env_params.food_params.food_grid" and value is None:
            return "Add food grid", gen.FoodGrid().nestedConf
        if path == "env_params.food_params.source_groups" and value == "empty_dict":
            return "Add source group", gen.FoodGroup().entry("SourceGroup")
        if path == "env_params.food_params.source_units" and value == "empty_dict":
            return "Add source unit", gen.Food().entry("Source")
        if path == "env_params.border_list" and value == "empty_dict":
            return (
                "Add border",
                gen.Border(vertices=[(-0.01, 0.0), (0.01, 0.0)], width=0.001).entry(
                    "Border"
                ),
            )
        if path == "env_params.windscape.puffs" and value == "empty_dict":
            return "Add air puff", {"Puff": gen.AirPuff().nestedConf}
        return None

    @staticmethod
    def _flatten_seed_payload(path: str, payload: Any) -> util.AttrDict:
        payload = _normalize_scalar(payload)
        if isinstance(payload, dict) and payload:
            items = util.AttrDict()
            for key, value in payload.items():
                nested = _SingleExperimentController._flatten_seed_payload(
                    f"{path}.{key}", value
                )
                items.update(nested)
            return items
        return util.AttrDict({path: payload})

    def _activate_factory_value(self, path: str, payload: Any, message: str) -> None:
        keys_to_remove = [
            key
            for key in self._parameter_seed_overrides
            if key == path or key.startswith(f"{path}.")
        ]
        for key in keys_to_remove:
            self._parameter_seed_overrides.pop(key, None)
        self._parameter_seed_overrides.update(self._flatten_seed_payload(path, payload))
        self._refresh_parameter_editor()
        self.status.object = message

    def _options_for_path(self, path: str, value: Any) -> tuple[str, list[Any]] | None:
        if path == "collections":
            return "multi", list(reg.parDB.output_keys)
        if path == "enrichment.proc_keys":
            return "multi", list(_ENRICHMENT_PROC_KEY_OPTIONS)
        if path == "enrichment.anot_keys":
            return "multi", list(_ENRICHMENT_ANOT_KEY_OPTIONS)
        if path.endswith("arena.geometry"):
            return "single", list(_ARENA_GEOMETRY_OPTIONS)
        if path.endswith("distribution.mode"):
            return "single", list(_SPATIAL_DISTRO_MODE_OPTIONS)
        if path.endswith("distribution.shape"):
            return "single", list(_SPATIAL_DISTRO_SHAPE_OPTIONS)
        if path.endswith(".model"):
            return "single", list(reg.conf.Model.confIDs)
        if path.endswith(".sample"):
            return "single_optional", [None] + list(reg.conf.Ref.confIDs)
        if path.endswith(".odor.id"):
            return "single_optional", [None] + list(
                self._editor_context.get("odor_ids", [])
            )
        if path == "enrichment.mode":
            return "single", list(_ENRICHMENT_MODE_OPTIONS)
        if path == "env_params.odorscape":
            return "single_optional", [None] + list(_ODORSCAPE_OPTIONS)
        if path.endswith(".odorscape.odorscape"):
            return "single", list(_ODORSCAPE_OPTIONS)
        return None

    @staticmethod
    def _optional_scalar_kind(path: str, value: Any) -> str | None:
        if path.endswith((".odor.intensity", ".odor.spread")) and (
            value is None or isinstance(value, (int, float))
        ):
            return "float"
        return None

    @staticmethod
    def _summarize_value(path: str, value: Any) -> str:
        if value == "empty_dict":
            return "Empty collection. This family should be edited via a dedicated structured editor, not free text."
        if value is None:
            return "Not configured."
        if isinstance(value, list):
            if path.endswith(".epochs"):
                return f"{len(value)} epoch item(s). This should be edited with a dedicated epoch editor."
            if path.endswith(".vertices"):
                return f"{len(value)} vertex points. This should be edited with the canvas/drawing tools."
            return f"List with {len(value)} item(s)."
        if isinstance(value, dict):
            return f"Object with {len(value)} field(s)."
        return str(value)

    def _widget_for_value(
        self, path: str, value: Any
    ) -> tuple[str, Any, pn.viewable.Viewable]:
        label = _display_label_for_path(path)
        help_text = self._help_text_for_path(path)
        optional_meta = self._optional_family_meta.get(path)
        if optional_meta is not None:
            toggle = _optional_family_control(
                path, bool(optional_meta["enabled"]), help_text
            )
            view = pn.Row(
                pn.Column(
                    pn.pane.HTML(_field_label_html(label), margin=0),
                    sizing_mode="stretch_width",
                    width_policy="max",
                    margin=0,
                ),
                pn.Spacer(width=16),
                pn.Row(
                    toggle,
                    align="end",
                    margin=0,
                ),
                sizing_mode="stretch_width",
                width_policy="max",
                margin=(0, 0, 6, 0),
            )
            return "toggle_factory", {"enabled": toggle}, view
        factory = self._factory_spec_for_path(path, value)
        if factory is not None:
            button_label, payload = factory
            button = pn.widgets.Button(
                name=button_label,
                button_type="primary",
                width=None,
                sizing_mode="stretch_width",
            )
            if help_text and hasattr(button, "description"):
                button.description = help_text
            _attach_scroll_restore(button)

            def _create_value(*_: object) -> None:
                self._activate_factory_value(
                    path,
                    payload,
                    f'Initialized "{label}" with a default configuration.',
                )

            button.on_click(_create_value)
            summary = pn.Column(
                pn.pane.HTML(_field_label_html(label), margin=0),
                pn.pane.HTML(
                    _summary_label(self._summarize_value(path, _json_ready(value))),
                    margin=0,
                ),
                button,
                sizing_mode="stretch_width",
                margin=(0, 0, 8, 0),
            )
            return "factory", {"value": value, "button": button}, summary
        options = self._options_for_path(path, value)
        if options is not None:
            mode, option_values = options
            if mode == "multi":
                widget = _apply_widget_help(
                    pn.widgets.MultiChoice(
                        name="",
                        value=list(value or []),
                        options=option_values,
                        sizing_mode="stretch_width",
                    ),
                    label,
                    help_text,
                )
                return "multichoice", widget, widget
            normalized_options = [
                _NONE_OPTION_LABEL if item is None else item for item in option_values
            ]
            selected_value = _NONE_OPTION_LABEL if value is None else value
            widget = _apply_widget_help(
                pn.widgets.Select(
                    name="",
                    value=selected_value
                    if selected_value in normalized_options
                    else normalized_options[0],
                    options=normalized_options,
                    sizing_mode="stretch_width",
                ),
                label,
                help_text,
            )
            return "option", widget, widget
        optional_scalar_kind = _SingleExperimentController._optional_scalar_kind(
            path, value
        )
        if optional_scalar_kind == "float":
            enabled = value is not None
            scalar = _apply_widget_help(
                pn.widgets.FloatInput(
                    name="",
                    value=float(value) if value is not None else 0.0,
                    step=0.1,
                    disabled=not enabled,
                    width=None,
                    sizing_mode="stretch_width",
                ),
                label,
                help_text,
            )
            enabled_toggle = pn.widgets.Checkbox(
                name="Enabled",
                value=enabled,
                margin=(5, 0, 0, 8),
            )

            def _sync_optional_float(event: Any) -> None:
                scalar.disabled = not bool(event.new)

            enabled_toggle.param.watch(_sync_optional_float, "value")
            view = pn.Column(
                pn.Row(scalar, enabled_toggle, sizing_mode="stretch_width", margin=0),
                sizing_mode="stretch_width",
                margin=(0, 0, 8, 0),
            )
            control = {"widget": scalar, "enabled": enabled_toggle}
            return "optional_float", control, view
        if isinstance(value, bool):
            widget = _apply_widget_help(
                pn.widgets.Checkbox(name="", value=value),
                label,
                help_text,
            )
            return "bool", widget, widget
        if isinstance(value, int):
            widget = _apply_widget_help(
                pn.widgets.IntInput(
                    name="",
                    value=value,
                    step=1,
                    sizing_mode="stretch_width",
                ),
                label,
                help_text,
            )
            return "int", widget, widget
        if isinstance(value, float):
            widget = _apply_widget_help(
                pn.widgets.FloatInput(
                    name="",
                    value=value,
                    step=0.1,
                    sizing_mode="stretch_width",
                ),
                label,
                help_text,
            )
            return "float", widget, widget
        if (
            isinstance(value, (list, tuple))
            and 1 <= len(value) <= 4
            and all(isinstance(item, (int, float)) for item in value)
        ):
            scalar_kind = (
                "int" if all(isinstance(item, int) for item in value) else "float"
            )
            component_labels = _sequence_component_labels(path, value)
            subwidgets: list[pn.viewable.Viewable] = []
            for index, item in enumerate(value, start=1):
                component_label = component_labels[index - 1]
                if scalar_kind == "int":
                    subwidgets.append(
                        _apply_widget_help(
                            pn.widgets.IntInput(
                                name=component_label,
                                value=int(item),
                                step=1,
                                width=None,
                                sizing_mode="stretch_width",
                            ),
                            component_label,
                            help_text,
                        )
                    )
                else:
                    subwidgets.append(
                        _apply_widget_help(
                            pn.widgets.FloatInput(
                                name=component_label,
                                value=float(item),
                                step=0.1,
                                width=None,
                                sizing_mode="stretch_width",
                            ),
                            component_label,
                            help_text,
                        )
                    )
            view = pn.Column(
                pn.pane.HTML(_field_label_html(label), margin=0),
                pn.Row(*subwidgets, sizing_mode="stretch_width", margin=0),
                sizing_mode="stretch_width",
                margin=(0, 0, 8, 0),
            )
            control = {
                "widgets": subwidgets,
                "container": tuple if isinstance(value, tuple) else list,
                "scalar_kind": scalar_kind,
            }
            return "sequence", control, view
        if isinstance(value, str):
            if path.endswith(".color"):
                widget = _apply_widget_help(
                    pn.widgets.ColorPicker(
                        name="",
                        value=value,
                        sizing_mode="stretch_width",
                    ),
                    label,
                    help_text,
                )
                return "color", widget, widget
            summary = pn.Column(
                pn.pane.HTML(_field_label_html(label), margin=0),
                pn.pane.HTML(_summary_label(str(value)), margin=0),
                sizing_mode="stretch_width",
                margin=(0, 0, 8, 0),
            )
            return "readonly", {"value": value}, summary
        summary = pn.Column(
            pn.pane.HTML(_field_label_html(label), margin=0),
            pn.pane.HTML(
                _summary_label(self._summarize_value(path, _json_ready(value))),
                margin=0,
            ),
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )
        return "readonly", {"value": _json_ready(value)}, summary

    @staticmethod
    def _parse_widget_value(kind: str, control: Any) -> Any:
        if kind == "toggle_factory":
            return bool(control["enabled"].value)
        if kind == "factory":
            return control["value"]
        if kind == "readonly":
            return control["value"]
        if kind == "sequence":
            scalar_kind = control["scalar_kind"]
            parsed = []
            for widget in control["widgets"]:
                value = getattr(widget, "value")
                parsed.append(int(value) if scalar_kind == "int" else float(value))
            return control["container"](parsed)
        if kind == "optional_float":
            return (
                None if not control["enabled"].value else float(control["widget"].value)
            )
        value = getattr(control, "value")
        if kind == "bool":
            return bool(value)
        if kind == "int":
            return int(value)
        if kind == "float":
            return float(value)
        if kind == "color":
            return str(value)
        if kind == "option":
            return None if value == _NONE_OPTION_LABEL else value
        if kind == "multichoice":
            return list(value)
        raise ValueError(f"Unsupported widget kind: {kind}")

    def _render_parameter_group(self) -> None:
        group_key = self.parameter_group.value
        if group_key == "env_params":
            if self._env_params_group_view is None:
                self.parameters_editor[:] = []
            else:
                self.parameters_editor[:] = [self._env_params_group_view]
            return
        if group_key == "enrichment":
            if self._enrichment_group_view is None:
                self.parameters_editor[:] = []
            else:
                self.parameters_editor[:] = [self._enrichment_group_view]
            return
        if group_key == "larva_groups":
            if self._larva_groups_group_view is None:
                self.parameters_editor[:] = []
            else:
                self.parameters_editor[:] = [self._larva_groups_group_view]
            return
        paths = self._parameter_groups.get(group_key, [])
        if not paths:
            self.parameters_editor[:] = []
            return
        families: dict[str, dict[str, Any]] = {}
        ordered_family_ids: list[str] = []
        for path in paths:
            family_id, family_title, color_key = _family_spec_for_path(path)
            if family_id not in families:
                families[family_id] = {
                    "title": family_title,
                    "color_key": color_key,
                    "views": [],
                    "title_toggle": None,
                }
                ordered_family_ids.append(family_id)
            kind, control, view = self._parameter_widget_specs[path]
            if (
                kind == "toggle_factory"
                and _display_label_for_path(path) == family_title
            ):
                families[family_id]["title_toggle"] = control["enabled"]
                continue
            families[family_id]["views"].append(view)

        self.parameters_editor[:] = [
            pn.Column(
                _family_title_row(
                    families[family_id]["title"],
                    families[family_id]["title_toggle"],
                ),
                *families[family_id]["views"],
                css_classes=[
                    "lw-single-exp-param-family",
                    f'lw-single-exp-param-family--{families[family_id]["color_key"]}',
                ],
                sizing_mode="stretch_width",
                margin=0,
            )
            for family_id in ordered_family_ids
        ]

    def _refresh_parameter_editor(self) -> None:
        flat = self._editable_flat_parameters()
        self._refresh_typed_larva_groups_owner()
        self._refresh_typed_enrichment_owner()
        self._refresh_typed_env_params_owner()
        self._editor_context = {
            "odor_ids": sorted(
                {
                    str(value)
                    for path, value in flat.items()
                    if path.endswith(".odor.id")
                    and isinstance(value, str)
                    and value.strip()
                }
            )
        }
        self._parameter_widget_specs = {}
        grouped: dict[str, list[str]] = {}
        self._parameter_widgets = {}
        # Keep paths in lexical order so newly materialized nested configs
        # (e.g. source units/groups) appear close to their family instead of
        # being appended at the end of the editor after factory activation.
        for path in sorted(flat.keys()):
            if path == "env_params" or path.startswith("env_params."):
                continue
            if path == "enrichment" or path.startswith("enrichment."):
                continue
            if path.startswith("larva_groups."):
                continue
            value = flat[path]
            group_key = path.split(".", 1)[0]
            kind, control, view = self._widget_for_value(path, value)
            grouped.setdefault(group_key, []).append(path)
            self._parameter_widgets[path] = (kind, control)
            self._parameter_widget_specs[path] = (kind, control, view)
        grouped.setdefault("enrichment", [])
        grouped.setdefault("larva_groups", [])
        grouped.setdefault("env_params", [])
        self._parameter_groups = grouped
        options = {_editor_group_title(group): group for group in grouped.keys()}
        current_group = self.parameter_group.value
        self.parameter_group.options = options
        if current_group not in options.values():
            if "larva_groups" in grouped:
                preferred = "larva_groups"
            elif "env_params" in grouped:
                preferred = "env_params"
            else:
                preferred = next(iter(grouped), None)
            self.parameter_group.value = preferred
        self._wire_optional_family_toggles()
        self._render_parameter_group()

    def _refresh_typed_larva_groups_owner(self) -> None:
        from larvaworld.lib.reg.generators import ExpConf

        parameters = resolve_base_experiment_parameters(
            self._selected_experiment(),
            self._load_selected_environment(),
        )
        self._typed_experiment_for_larva_groups = ExpConf(**dict(parameters))
        self._larva_groups_group_view = build_larva_groups_widget(
            self._typed_experiment_for_larva_groups
        )

    def _refresh_typed_enrichment_owner(self) -> None:
        from larvaworld.lib.reg.generators import ExpConf

        parameters = resolve_base_experiment_parameters(
            self._selected_experiment(),
            self._load_selected_environment(),
        )
        self._typed_experiment_for_enrichment = ExpConf(**dict(parameters))
        self._enrichment_group_view = build_enrichment_widget(
            self._typed_experiment_for_enrichment.enrichment,
            wrap=False,
        )

    def _refresh_typed_env_params_owner(self) -> None:
        from larvaworld.lib.reg.generators import ExpConf

        parameters = resolve_base_experiment_parameters(
            self._selected_experiment(),
            self._load_selected_environment(),
        )
        self._typed_experiment_for_env_params = ExpConf(**dict(parameters))
        self._env_params_group_view = build_env_params_widget(
            self._typed_experiment_for_env_params.env_params,
            wrap=False,
        )

    def _on_experiment_change(self, *_: object) -> None:
        experiment = self._selected_experiment()
        self._parameter_seed_overrides = util.AttrDict()
        self.run_name.value = _default_run_name(experiment)
        self._refresh_environment_options()
        self._refresh_summary()
        self._refresh_parameter_editor()
        self.status.object = f'Template "{experiment}" loaded.'

    def _on_run_name_change(self, *_: object) -> None:
        current = (self.video_filename.value or "").strip()
        if not current or current == _safe_slug(current):
            self.video_filename.value = _safe_slug(
                self.run_name.value or self._selected_experiment()
            )

    def _on_save_video_change(self, *_: object) -> None:
        enabled = bool(self.save_video.value)
        self.video_filename.disabled = not enabled
        self.video_fps.disabled = not enabled

    def _on_show_display_change(self, *_: object) -> None:
        self.display_every_n_steps.disabled = not bool(self.show_display.value)

    def _on_parameter_override_change(self, *_: object) -> None:
        self._parameter_seed_overrides = util.AttrDict()
        self._refresh_parameter_editor()

    def _on_parameter_group_change(self, *_: object) -> None:
        self._render_parameter_group()

    def _on_refresh_environments(self, *_: object) -> None:
        self._refresh_environment_options()
        self._refresh_parameter_editor()
        self.status.object = "Refreshed workspace environment presets."

    def _build_run_directory(self) -> Path:
        run_id = _safe_slug(self.run_name.value or self._selected_experiment())
        base_dir = self._experiment_dir()
        candidate = base_dir / run_id
        if not candidate.exists():
            return candidate
        suffix = 2
        while (base_dir / f"{run_id}_{suffix}").exists():
            suffix += 1
        return base_dir / f"{run_id}_{suffix}"

    @staticmethod
    def _resolved_plan_payload(
        *,
        experiment: str,
        run_name: str,
        selected_env: str,
        parameters: util.AttrDict,
    ) -> dict[str, Any]:
        return {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": experiment,
            "run_name": run_name,
            "selected_environment": selected_env,
            "parameters": _json_ready(parameters.get_copy()),
        }

    def _write_resolved_plan(
        self,
        *,
        run_dir: Path,
        parameters: util.AttrDict,
        selected_env: str,
    ) -> Path:
        run_dir.mkdir(parents=True, exist_ok=False)
        plan_path = run_dir / "resolved_experiment.json"
        payload = self._resolved_plan_payload(
            experiment=self._selected_experiment(),
            run_name=self.run_name.value or run_dir.name,
            selected_env=selected_env,
            parameters=parameters,
        )
        plan_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return plan_path

    def _runtime_screen_kws(self, run_dir: Path) -> dict[str, Any]:
        kws: dict[str, Any] = {
            "show_display": bool(self.show_display.value),
            "display_every_n_steps": int(self.display_every_n_steps.value),
        }
        if self.show_display.value:
            kws["vis_mode"] = "video"
        if self.save_video.value:
            video_name = _safe_slug(
                (self.video_filename.value or "").strip() or run_dir.name
            )
            kws.update(
                {
                    "save_video": True,
                    "vis_mode": "video",
                    "video_file": video_name,
                    "media_dir": str(run_dir),
                    "fps": int(self.video_fps.value),
                }
            )
        return kws

    def _set_run_controls_disabled(self, disabled: bool) -> None:
        self.prepare_btn.disabled = disabled
        self.simulation_preview_btn.disabled = disabled
        self.run_btn.disabled = disabled
        self.preview_frames_input.disabled = disabled
        self.display_every_n_steps.disabled = disabled or not bool(
            self.show_display.value
        )
        self.refresh_environments_btn.disabled = disabled

    def _execute_run_experiment(
        self,
        *,
        parameters: util.AttrDict,
        run_dir: Path,
        selected_env: str,
    ) -> None:
        launcher = None
        try:
            plan_path = self._write_resolved_plan(
                run_dir=run_dir,
                parameters=parameters,
                selected_env=selected_env,
            )
            screen_kws = self._runtime_screen_kws(run_dir)
            launcher = sim.ExpRun(
                experiment=self._selected_experiment(),
                parameters=parameters,
                id=run_dir.name,
                dir=str(run_dir),
                store_data=True,
                screen_kws=screen_kws,
            )
            datasets = launcher.simulate()
        except Exception as exc:
            self.status.object = f"Single experiment run failed: {exc}"
            self._set_run_controls_disabled(False)
            return
        finally:
            try:
                if launcher is not None and getattr(launcher, "screen_manager", None):
                    launcher.screen_manager.close()
            except Exception:
                pass

        dataset_count = len(datasets or [])
        self.status.object = (
            f'Completed run "{self._selected_experiment()}". '
            f"Stored outputs in <code>{run_dir}</code>. "
            f"Saved resolved config to <code>{plan_path.name}</code>. "
            f"Datasets produced: {dataset_count}."
            + (
                f' Video target: <code>{run_dir / ((_safe_slug((self.video_filename.value or "").strip() or run_dir.name)) + ".mp4")}</code>.'
                if self.save_video.value
                else ""
            )
        )
        self._set_run_controls_disabled(False)

    @staticmethod
    def _preview_metadata_html(parameters: util.AttrDict, selected_env: str) -> str:
        env = util.AttrDict(parameters.env_params)
        larva_groups = util.AttrDict(parameters.get("larva_groups", {}))
        counts = []
        total = 0
        for group_id, group in larva_groups.items():
            try:
                count = int(group.distribution.N)
            except Exception:
                count = 0
            total += count
            counts.append(f"{group_id}: {count}")
        dims = getattr(env.arena, "dims", ("?", "?"))
        if isinstance(dims, (list, tuple)) and len(dims) >= 2:
            dims_text = f"{float(dims[0]):.3f} x {float(dims[1]):.3f} m"
        else:
            dims_text = str(dims)
        return (
            '<div class="lw-single-exp-preview-meta">'
            f"<strong>Applied preview config:</strong> "
            f"environment = {selected_env}; "
            f"arena = {env.arena.geometry} ({dims_text}); "
            f"duration = {float(parameters.duration):.2f} min; "
            f"larvae = {total}"
            + (f" ({', '.join(counts)})" if counts else "")
            + ".</div>"
        )

    @staticmethod
    def _preview_runtime_parameters(parameters: util.AttrDict) -> util.AttrDict:
        preview_parameters = util.AttrDict(_coerce_xy_sequences(parameters.get_copy()))
        preview_parameters["collections"] = []
        preview_parameters["enrichment"] = None
        return preview_parameters

    @staticmethod
    def _prepare_preview_launcher(
        experiment: str,
        parameters: util.AttrDict,
        run_dir: Path,
    ) -> tuple[sim.ExpRun, str | None]:
        preview_parameters = _SingleExperimentController._preview_runtime_parameters(
            parameters
        )
        fallback_note = None
        try:
            launcher = sim.ExpRun(
                experiment=experiment,
                parameters=preview_parameters,
                id=run_dir.name,
                dir=str(run_dir),
                store_data=False,
            )
            launcher.sim_setup(steps=_PREVIEW_STEP_CAP)
            return launcher, fallback_note
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
            launcher.sim_setup(steps=_PREVIEW_STEP_CAP)
            fallback_note = "Preview fallback disabled larva overlap elimination for this visualization."
            return launcher, fallback_note

    def _on_prepare_preview(self, *_: object) -> None:
        try:
            parameters = self._resolve_experiment_parameters()
            run_dir = self._build_run_directory()
        except (WorkspaceError, OSError, json.JSONDecodeError) as exc:
            self.status.object = f"Cannot prepare the configuration preview: {exc}"
            return

        selected_env = self._selected_environment_label()
        self._set_run_controls_disabled(True)
        try:
            state = env_params_to_canvas_state(
                parameters.env_params,
                larva_groups=parameters.get("larva_groups", {}),
                show_group_shapes=False,
            )
            canvas = EnvironmentCanvas(editable=False)
            canvas.set_state(state)
        except Exception as exc:
            self.preview_meta.object = ""
            self.preview[:] = [
                pn.pane.HTML(
                    (
                        '<div class="lw-single-exp-preview-placeholder">'
                        f"Configuration preview failed: {exc}"
                        "</div>"
                    ),
                    margin=0,
                )
            ]
            self.status.object = f"Cannot prepare the configuration preview: {exc}"
            self._set_run_controls_disabled(False)
            return

        self.preview_meta.object = self._preview_metadata_html(parameters, selected_env)
        self.preview[:] = [canvas.view()]
        self.status.object = (
            f'Prepared configuration preview for "{self._selected_experiment()}" using '
            f"{selected_env}. Reserved output directory for a future run: "
            f"<code>{run_dir}</code>. No simulation has been run."
        )
        self._set_run_controls_disabled(False)

    def _on_generate_simulation_preview(self, *_: object) -> None:
        try:
            parameters = self._resolve_experiment_parameters()
            run_dir = self._build_run_directory()
        except (WorkspaceError, OSError, json.JSONDecodeError) as exc:
            self.status.object = f"Cannot generate the simulation preview: {exc}"
            return

        selected_env = self._selected_environment_label()
        self.preview_meta.object = self._preview_metadata_html(parameters, selected_env)
        self.preview[:] = [
            pn.pane.HTML(
                (
                    '<div class="lw-single-exp-preview-placeholder">'
                    "Generating simulation preview. The environment and agents are being initialized."
                    "</div>"
                ),
                margin=0,
            )
        ]
        experiment = self._selected_experiment()
        self.status.object = (
            f'Generating simulation preview for "{experiment}" using {selected_env}.'
        )
        self._set_run_controls_disabled(True)
        launcher = None
        try:
            launcher, fallback_note = self._prepare_preview_launcher(
                experiment,
                parameters,
                run_dir,
            )
            requested_steps = int(self.preview_frames_input.value)
            preview_steps = max(1, min(requested_steps, int(launcher.p.steps)))
            frames = generate_preview_frames(
                launcher,
                preview_steps=preview_steps,
            )
            if not frames:
                raise ValueError("No preview frames were generated.")
            state = env_params_to_canvas_state(
                parameters.env_params,
                larva_groups=None,
                show_group_shapes=False,
            )
            canvas = EnvironmentCanvas(editable=False)
            canvas.set_state(state)
            preview = _FrameSimulationPreview(
                canvas=canvas,
                frames=frames,
                dt=float(launcher.dt),
            ).view()
        except Exception as exc:
            self.preview_meta.object = ""
            self.preview[:] = [
                pn.pane.HTML(
                    (
                        '<div class="lw-single-exp-preview-placeholder">'
                        f"Preview preparation failed: {exc}"
                        "</div>"
                    ),
                    margin=0,
                )
            ]
            self.status.object = f"Cannot generate the simulation preview: {exc}"
            self._set_run_controls_disabled(False)
            return
        finally:
            try:
                if launcher is not None and getattr(launcher, "screen_manager", None):
                    launcher.screen_manager.close()
            except Exception:
                pass

        self.preview_meta.object = self._preview_metadata_html(parameters, selected_env)
        self.preview[:] = [preview]
        displayed_end = frames[-1].tick * float(launcher.dt)
        status = (
            f'Generated simulation preview for "{experiment}" using {selected_env}. '
            f"Reserved output directory for a future run: <code>{run_dir}</code>. "
            f"Simulation preview ready: {len(frames)} frames generated. "
            f"Displayed range: 0.0-{displayed_end:.1f} s simulated time. "
            "Generated from the actual simulation engine. Outputs are not stored; use Run experiment for the full run."
        )
        if fallback_note:
            status = f"{status} {fallback_note}"
        self.status.object = status
        self._set_run_controls_disabled(False)

    def _on_run_experiment(self, *_: object) -> None:
        try:
            parameters = self._resolve_experiment_parameters()
            run_dir = self._build_run_directory()
        except (WorkspaceError, OSError, json.JSONDecodeError) as exc:
            self.status.object = f"Cannot start the single experiment run: {exc}"
            return

        selected_env = self._selected_environment_label()
        experiment = self._selected_experiment()
        self.status.object = (
            f'Running "{experiment}" using {selected_env}. '
            f"Outputs will be stored under <code>{run_dir}</code>. "
            "The UI will be unresponsive until the simulation finishes."
        )
        self.preview[:] = [
            pn.pane.HTML(
                (
                    '<div class="lw-single-exp-preview-placeholder">'
                    f'Running "{html.escape(experiment)}" and writing outputs to '
                    f"<code>{html.escape(str(run_dir))}</code>. "
                    "This view will update again after the simulation finishes."
                    "</div>"
                ),
                margin=0,
            )
        ]
        self._set_run_controls_disabled(True)
        document = pn.state.curdoc
        if document is not None:
            document.add_next_tick_callback(
                lambda: self._execute_run_experiment(
                    parameters=parameters,
                    run_dir=run_dir,
                    selected_env=selected_env,
                )
            )
            return
        self._execute_run_experiment(
            parameters=parameters,
            run_dir=run_dir,
            selected_env=selected_env,
        )

    def view(self) -> pn.viewable.Viewable:
        intro = pn.pane.Markdown(
            (
                "### Single Experiment\n"
                "Prepare one Larvaworld `Exp` run in the portal: select an experiment template, "
                "optionally override its environment with a workspace preset, and inspect the arena "
                "dynamics in an interactive preview before broader simulation workflows are added. "
                f"References: [Single Experiments]({DOCS_SINGLE_EXPERIMENTS}) and "
                f"[Experiment Types]({DOCS_EXPERIMENT_TYPES})."
            ),
            css_classes=["lw-single-exp-intro"],
            margin=0,
        )
        media_controls = pn.Column(
            pn.pane.Markdown("#### Media / Output", margin=(0, 0, 6, 0)),
            self.save_video,
            self.video_filename,
            self.video_fps,
            self.show_display,
            self.display_every_n_steps,
            css_classes=["lw-single-exp-media"],
            sizing_mode="stretch_width",
            margin=0,
        )
        controls = pn.Card(
            pn.Column(
                self.experiment,
                self.run_name,
                self.environment_select,
                self.refresh_environments_btn,
                self.summary,
                self.preview_action_row,
                self.preview_options_row,
                self.preview_generate_row,
                media_controls,
                self.execution_action_row,
                self.status,
                sizing_mode="stretch_width",
                margin=0,
            ),
            title="Configuration",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        preview = pn.Card(
            self.preview_meta,
            self.preview,
            title="Preview",
            collapsed=False,
            sizing_mode="stretch_width",
        )
        parameters = pn.Card(
            pn.Column(
                self.parameter_group,
                self.parameters_editor,
                sizing_mode="stretch_width",
                margin=0,
            ),
            title="Experiment Parameters",
            collapsed=False,
            sizing_mode="stretch_width",
            css_classes=["lw-single-exp-params-group"],
        )
        left_column = pn.Column(
            controls,
            parameters,
            sizing_mode="stretch_width",
        )
        content = pn.GridSpec(
            ncols=4,
            nrows=1,
            sizing_mode="stretch_width",
        )
        content[0, 0] = left_column
        content[0, 1:4] = preview
        return pn.Column(
            intro,
            content,
            css_classes=["lw-single-exp-root"],
            sizing_mode="stretch_width",
        )


def single_experiment_app() -> pn.viewable.Viewable:
    pn.extension(raw_css=[PORTAL_RAW_CSS, SINGLE_EXPERIMENT_RAW_CSS])
    hv.extension("bokeh")
    controller = _SingleExperimentController()

    template = pn.template.MaterialTemplate(
        title="",
        header_background="#c1b0c2",
        header_color="#111111",
    )
    template.header.append(build_app_header(title="Single Experiment"))
    template.main.append(controller.view())
    return template
