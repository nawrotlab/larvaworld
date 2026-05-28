from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path
import re
from typing import Any

import panel as pn

from larvaworld.lib import reg, sim
from larvaworld.lib.process.dataset import LarvaDataset
from larvaworld.portal.canvas_widgets import EnvironmentCanvas
from larvaworld.portal.datasets.replay_data import (
    build_environment_state_for_member,
    build_render_state,
    build_source_catalog,
    parse_agent_indices,
    prepare_replay_source,
)
from larvaworld.portal.datasets.replay_models import (
    PreparedReplayMember,
    PreparedReplaySource,
    ReplaySource,
)
from larvaworld.portal.panel_components import PORTAL_RAW_CSS, build_app_header
from larvaworld.portal.runtime.display_shortcuts import (
    DISPLAY_SHORTCUTS_RAW_CSS,
    build_display_shortcuts_dialog,
)
from larvaworld.portal.workspace import get_active_workspace, get_workspace_dir


__all__ = ["_DatasetReplayController", "dataset_replay_app"]


_REPLAY_CANVAS_WIDTH = 920
_REPLAY_CANVAS_HEIGHT = 760


DATASET_REPLAY_RAW_CSS = """
.lw-dataset-replay-root { padding: 14px 12px 20px 12px; }
.lw-dataset-replay-intro {
  border-left: 4px solid #7aa6c2;
  background: rgba(122, 166, 194, 0.16);
  border-radius: 10px;
  padding: 10px 12px;
  margin: 0 0 10px 0;
}
.lw-dataset-replay-status {
  font-size: 12px;
  line-height: 1.45;
  border-radius: 10px;
  padding: 10px 12px;
  border: 1px solid rgba(17, 17, 17, 0.1);
  background: rgba(248, 250, 252, 0.94);
}
.lw-dataset-replay-controls {
  gap: 10px;
}
.lw-dataset-replay-control-tile {
  border-radius: 8px;
  border: 1px solid rgba(17, 17, 17, 0.1);
  background: rgba(255, 255, 255, 0.96);
  box-shadow: none;
}
""".strip()


def _status_html(text: str) -> str:
    return f'<div class="lw-dataset-replay-status">{escape(text)}</div>'


def _control_tile(title: str, *children: object) -> pn.Card:
    return pn.Card(
        pn.Column(*children, sizing_mode="stretch_width", margin=0),
        title=title,
        collapsed=False,
        collapsible=False,
        sizing_mode="stretch_width",
        css_classes=["lw-dataset-replay-control-tile"],
    )


def _subcontrol_tile(title: str, *children: object) -> pn.Card:
    return pn.Card(
        pn.Column(*children, sizing_mode="stretch_width", margin=0),
        title=title,
        collapsed=False,
        collapsible=False,
        sizing_mode="stretch_width",
        margin=(4, 0, 4, 0),
        css_classes=["lw-dataset-replay-control-tile"],
    )


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug


class _DatasetReplayController:
    _FIX_SEGMENT_NONE = "__none__"
    _FIX_SEGMENT_FRONT = "front"
    _FIX_SEGMENT_REAR = "rear"
    _BODY_RENDER_SEGMENTS = "segments"
    _BODY_RENDER_CONTOUR = "contour"
    _BODY_RENDER_MIDLINE = "midline"

    def __init__(self) -> None:
        self.workspace = get_active_workspace()
        self.canvas = EnvironmentCanvas(
            width=_REPLAY_CANVAS_WIDTH,
            height=_REPLAY_CANVAS_HEIGHT,
            snap_heads_to_midline=False,
        )

        self._sources = build_source_catalog(self.workspace)
        self._source_by_token = {source.token: source for source in self._sources}
        self._prepared: PreparedReplaySource | None = None
        self._last_static_state_key: tuple[str, str | None, bool, bool] | None = None
        self._native_replay_controls_locked = False

        self.source_select = pn.widgets.Select(name="Source", options={})

        self.member_visibility = pn.widgets.CheckBoxGroup(
            name="Visible members", options=[], value=[]
        )
        self.show_positions = pn.widgets.Checkbox(name="Positions", value=True)
        self.show_ids = pn.widgets.Checkbox(name="IDs", value=False)
        self.show_tracks = pn.widgets.Checkbox(name="Tracks", value=False)
        self.show_heads = pn.widgets.Checkbox(name="Heads", value=False)
        self.show_midlines = pn.widgets.Checkbox(name="Midlines", value=True)
        self.show_segments = pn.widgets.Checkbox(name="Body segments", value=True)
        self.show_body_contours = pn.widgets.Checkbox(name="Body contours", value=False)
        self.show_dispersal = pn.widgets.Checkbox(name="Dispersal circle", value=True)
        self.trail_length = pn.widgets.IntSlider(
            name="Trail length", start=1, end=300, step=1, value=80
        )
        replay_cls = reg.gen.Replay
        transposition_objects = list(replay_cls.param["transposition"].objects or [])
        transposition_options = {"Stored coordinates": None}
        for item in transposition_objects:
            transposition_options[str(item)] = str(item)
        self.transposition = pn.widgets.Select(
            name="Transposition",
            options=transposition_options,
            value="origin" if "origin" in transposition_options.values() else None,
        )
        self.track_point = pn.widgets.IntInput(name="Track point", value=-1, step=1)
        self.fix_point = pn.widgets.Select(
            name="Fix point",
            options={"None": None},
            value=None,
            disabled=True,
        )
        self.close_view = pn.widgets.Checkbox(
            name="Close view", value=False, disabled=True
        )
        self.fix_segment = pn.widgets.Select(
            name="Fix orientation",
            options={"None": self._FIX_SEGMENT_NONE},
            value=self._FIX_SEGMENT_NONE,
            disabled=True,
        )
        self.native_body_rendering = pn.widgets.Select(
            name="Body rendering",
            options={
                "Segments": self._BODY_RENDER_SEGMENTS,
                "Contour": self._BODY_RENDER_CONTOUR,
                "Midline": self._BODY_RENDER_MIDLINE,
            },
            value=self._BODY_RENDER_SEGMENTS,
        )
        self.native_segment_count = pn.widgets.Select(
            name="Segment count",
            options={"2": 2},
            value=2,
        )
        self.agent_indices = pn.widgets.TextInput(
            name="Agent indices",
            placeholder="empty = all; e.g. 0,1,2",
            value="",
        )
        self.tick_player = pn.widgets.Player(
            name="Tick", start=0, end=1, step=1, value=0, interval=100
        )
        self.time_start = pn.widgets.FloatInput(name="Start (s)", value=0.0, step=1.0)
        self.time_end = pn.widgets.FloatInput(name="End (s)", value=0.0, step=1.0)
        self.use_time_range = pn.widgets.Checkbox(
            name="Limit replay time range", value=False
        )
        self.show_display = pn.widgets.Checkbox(name="Show display", value=True)
        self.display_every_n_steps = pn.widgets.IntInput(
            name="Display every N steps", value=2, step=1, start=1, end=20
        )
        self.save_video = pn.widgets.Checkbox(name="Save video", value=False)
        self.video_filename = pn.widgets.TextInput(
            name="Video filename",
            value="",
            disabled=True,
        )
        self.video_fps = pn.widgets.IntInput(
            name="Video speed-up",
            value=1,
            step=1,
            start=1,
            end=120,
            disabled=True,
        )
        self.open_pygame_replay_btn = pn.widgets.Button(
            name="Run replay", button_type="primary"
        )
        self.display_shortcuts_dialog_controller = build_display_shortcuts_dialog(
            note=(
                "These shortcuts apply only to the native pygame display opened by "
                "Dataset Replay when Show display is enabled. "
                "They do not control the browser replay canvas."
            )
        )
        self.display_shortcuts = self.display_shortcuts_dialog_controller.controller
        self.display_shortcuts_link = (
            self.display_shortcuts_dialog_controller.open_button
        )
        self.display_shortcuts_link.button_type = "default"
        self.display_shortcuts_link.css_classes = []
        self.display_shortcuts_close_btn = (
            self.display_shortcuts_dialog_controller.close_button
        )
        self.display_shortcuts_dialog = self.display_shortcuts_dialog_controller.dialog
        self.status_pane = pn.pane.HTML(
            _status_html("Select a replay source to begin."), margin=0
        )

        self.source_select.param.watch(self._on_source_change, "value")
        self.member_visibility.param.watch(self._on_any_control_change, "value")
        self.fix_point.param.watch(self._on_any_control_change, "value")
        self.native_body_rendering.param.watch(self._on_any_control_change, "value")
        self.show_display.param.watch(self._on_show_display_change, "value")
        self.save_video.param.watch(self._on_save_video_change, "value")
        self.open_pygame_replay_btn.on_click(self._on_open_pygame_replay)
        for widget in (
            self.show_positions,
            self.show_ids,
            self.show_tracks,
            self.show_heads,
            self.show_midlines,
            self.show_segments,
            self.show_body_contours,
            self.show_dispersal,
            self.trail_length,
            self.transposition,
            self.track_point,
            self.agent_indices,
            self.tick_player,
            self.time_start,
            self.time_end,
            self.use_time_range,
        ):
            widget.param.watch(self._on_any_control_change, "value")

        self._refresh_native_replay_control_state()
        self._reload_source_options()
        self.canvas._sim_larva_centroid_renderer.visible = True
        self.canvas._sim_larva_head_renderer.visible = True

    def _set_status(self, text: str) -> None:
        self.status_pane.object = _status_html(text)

    def _reload_source_options(self) -> None:
        previous = self.source_select.value
        options = self._source_options()
        self.source_select.options = options
        if previous in options.values():
            self.source_select.value = previous
        elif options:
            self.source_select.value = next(iter(options.values()))
        else:
            self.source_select.value = None
            self._prepared = None
            self._set_status("No replay sources found.")
            self.canvas.clear()
            self.canvas.clear_dynamic_overlays()

    def _source_options(self) -> dict[str, str]:
        by_label: dict[str, list[ReplaySource]] = defaultdict(list)
        for source in self._sources:
            by_label[source.label].append(source)
        options: dict[str, str] = {}
        for label, sources in by_label.items():
            if len(sources) == 1:
                options[label] = sources[0].token
                continue
            disambiguator_counts: dict[str, int] = defaultdict(int)
            for source in sources:
                suffix = self._source_disambiguator(source)
                disambiguator_counts[suffix] += 1
                index = disambiguator_counts[suffix]
                disambiguator = suffix if index == 1 else f"{suffix}#{index}"
                options[f"{label} ({disambiguator})"] = source.token
        return options

    @staticmethod
    def _source_disambiguator(source: ReplaySource) -> str:
        token = str(source.token)
        if token.startswith("workspace:"):
            raw = token.split("workspace:", 1)[1]
            path = Path(raw)
            parent = path.parent.name
            name = path.name
            return f"{parent}/{name}" if parent else name
        if token.startswith("workspace_group:"):
            return token.split("workspace_group:", 1)[1]
        if token.startswith("workspace_simulation_run:"):
            return token.split("workspace_simulation_run:", 1)[1]
        if token.startswith("workspace_simulation:"):
            return token.split("workspace_simulation:", 1)[1]
        if token.startswith("registry_ref:"):
            return token.split("registry_ref:", 1)[1]
        if token.startswith("registry_group:"):
            return token.split("registry_group:", 1)[1]
        return token

    def _on_source_change(self, _event=None) -> None:
        token = self.source_select.value
        source = self._source_by_token.get(str(token))
        if source is None:
            self._prepared = None
            return
        try:
            self._prepared = prepare_replay_source(source)
        except Exception as exc:
            self._prepared = None
            self.member_visibility.options = {}
            self.member_visibility.value = []
            self.canvas.clear()
            self.canvas.clear_dynamic_overlays()
            self._set_status(f"Replay source load failed: {source.label}: {exc}")
            self._refresh_native_replay_control_state()
            return
        member_options = {member.label: member.token for member in source.members}
        self.member_visibility.options = member_options
        self.member_visibility.value = list(member_options.values())
        nticks = max(1, self._prepared.nticks)
        self.tick_player.end = max(nticks - 1, 0)
        dt = max(self._prepared.dt, 0.001)
        self.tick_player.interval = int(max(50, round(dt * 1000)))
        self.time_end.value = float(max(nticks - 1, 0) * dt)
        self.tick_player.value = 0
        self._set_status(f"Loaded source: {source.label}")
        self._last_static_state_key = None
        self._refresh_close_inspection_controls()
        self._refresh_native_segment_count_options()
        self._render()
        self._refresh_native_replay_control_state()
        self._show_native_replay_blocker_status_if_needed()

    def _on_any_control_change(self, _event=None) -> None:
        self._refresh_close_inspection_controls()
        self._refresh_native_segment_count_options()
        self._render()
        self._refresh_native_replay_control_state()
        self._show_native_replay_blocker_status_if_needed()

    def _on_show_display_change(self, _event=None) -> None:
        self._refresh_native_replay_control_state()

    def _on_save_video_change(self, _event=None) -> None:
        self._refresh_native_replay_control_state()

    def _refresh_native_replay_control_state(self) -> None:
        controls_locked = bool(self._native_replay_controls_locked)
        show_display = bool(self.show_display.value)
        save_video = bool(self.save_video.value)
        blocker = self._native_replay_blocker()
        self.display_every_n_steps.disabled = controls_locked or not show_display
        self.video_filename.disabled = controls_locked or not save_video
        self.video_fps.disabled = controls_locked or not save_video
        self.open_pygame_replay_btn.disabled = (
            controls_locked or not (show_display or save_video) or blocker is not None
        )

    def _native_replay_blocker(self) -> str | None:
        if self._prepared is None:
            return "Select a replay source to begin."
        visible_tokens = [str(token) for token in self.member_visibility.value]
        if len(visible_tokens) == 0:
            return "Select one visible member for native replay."
        if len(visible_tokens) > 1:
            return "Native replay supports one visible member in this version."
        prepared_member = self._prepared.members.get(visible_tokens[0])
        if prepared_member is None:
            return "Prepared replay member is unavailable."
        if prepared_member.native_replay_missing_columns:
            missing = ", ".join(prepared_member.native_replay_missing_columns)
            return f"Native replay is unavailable for this member: missing {missing}."
        return None

    def _show_native_replay_blocker_status_if_needed(self) -> None:
        blocker = self._native_replay_blocker()
        if blocker is not None and blocker.startswith("Native replay is unavailable"):
            self._set_status(blocker)

    def _set_native_replay_controls_disabled(self, disabled: bool) -> None:
        self._native_replay_controls_locked = bool(disabled)
        self.show_display.disabled = bool(disabled)
        self.save_video.disabled = bool(disabled)
        self.display_shortcuts_dialog_controller.set_disabled(bool(disabled))
        self._refresh_close_inspection_controls()
        self._refresh_native_replay_control_state()

    def _visible_member_tokens(self) -> list[str]:
        return [str(token) for token in self.member_visibility.value]

    def _single_visible_member_token(self) -> str | None:
        visible_tokens = self._visible_member_tokens()
        if len(visible_tokens) != 1:
            return None
        return visible_tokens[0]

    def _prepared_member_for_token(
        self, token: str | None
    ) -> PreparedReplayMember | None:
        if token is None or self._prepared is None:
            return None
        return self._prepared.members.get(token)

    def _native_fix_points_for_member(
        self, member: PreparedReplayMember
    ) -> list[tuple[str, int]]:
        items: list[tuple[str, int]] = []
        mapping = member.native_track_point_by_ui_track_point or {}
        for ui_idx in sorted(mapping.keys()):
            native_idx = mapping.get(ui_idx)
            if native_idx is None:
                continue
            native_point = int(native_idx)
            if native_point <= 0:
                continue
            items.append((f"Body point {ui_idx + 1}", native_point))
        return items

    @staticmethod
    def _orientation_options_for_native_fix_point(
        native_fix_point: int | None,
        native_fix_points: set[int],
    ) -> dict[str, str]:
        options: dict[str, str] = {"None": _DatasetReplayController._FIX_SEGMENT_NONE}
        if native_fix_point is None:
            return options
        if (native_fix_point - 1) in native_fix_points:
            options["Front segment"] = _DatasetReplayController._FIX_SEGMENT_FRONT
        if (native_fix_point + 1) in native_fix_points:
            options["Rear segment"] = _DatasetReplayController._FIX_SEGMENT_REAR
        return options

    def _selected_native_fix_point(self) -> int | None:
        if self.fix_point.value is None:
            return None
        return int(self.fix_point.value)

    def _selected_fix_segment(self) -> str | None:
        value = self.fix_segment.value
        if value in [None, self._FIX_SEGMENT_NONE]:
            return None
        return str(value)

    def _refresh_close_inspection_controls(self) -> None:
        controls_locked = bool(self._native_replay_controls_locked)
        token = self._single_visible_member_token()
        member = self._prepared_member_for_token(token)

        if controls_locked or member is None:
            self.fix_point.options = {"None": None}
            self.fix_point.value = None
            self.fix_point.disabled = True
            self.close_view.value = False
            self.close_view.disabled = True
            self.fix_segment.options = {"None": self._FIX_SEGMENT_NONE}
            self.fix_segment.value = self._FIX_SEGMENT_NONE
            self.fix_segment.disabled = True
            return

        point_options: dict[str, int | None] = {"None": None}
        point_items = self._native_fix_points_for_member(member)
        for label, native_idx in point_items:
            point_options[label] = native_idx
        self.fix_point.options = point_options
        if self.fix_point.value not in set(point_options.values()):
            self.fix_point.value = None
        self.fix_point.disabled = False

        selected_fix_point = self._selected_native_fix_point()
        if selected_fix_point is None:
            self.close_view.value = False
            self.close_view.disabled = True
            self.fix_segment.options = {"None": self._FIX_SEGMENT_NONE}
            self.fix_segment.value = self._FIX_SEGMENT_NONE
            self.fix_segment.disabled = True
            return

        self.close_view.disabled = False
        native_fix_points = {native_idx for _, native_idx in point_items}
        orientation_options = self._orientation_options_for_native_fix_point(
            selected_fix_point, native_fix_points
        )
        self.fix_segment.options = orientation_options
        if self.fix_segment.value not in set(orientation_options.values()):
            self.fix_segment.value = self._FIX_SEGMENT_NONE
        self.fix_segment.disabled = len(orientation_options) <= 1

    @staticmethod
    def _native_segment_count_options_for_member(
        member: PreparedReplayMember,
    ) -> dict[str, int]:
        options = {"2": 2}
        if len(member.body_xy_by_point) >= 2:
            full_count = max(2, len(member.body_xy_by_point) - 1)
            options[f"All available ({full_count})"] = full_count
        return options

    def _refresh_native_segment_count_options(self) -> None:
        token = self._single_visible_member_token()
        member = self._prepared_member_for_token(token)
        if member is None:
            options = {"2": 2}
        else:
            options = self._native_segment_count_options_for_member(member)
        previous_values = set(self.native_segment_count.options.values())
        self.native_segment_count.options = options
        option_values = set(options.values())
        if (
            previous_values != option_values
            or self.native_segment_count.value not in option_values
        ):
            self.native_segment_count.value = max(options.values())
        self.native_segment_count.disabled = (
            self.native_body_rendering.value != self._BODY_RENDER_SEGMENTS
        )

    def _native_replay_video_output_dir(self) -> Path:
        return (
            get_workspace_dir("analysis", workspace=self.workspace)
            / "dataset_replay_media"
        )

    def _native_replay_video_name(self, selected_member_token: str) -> str:
        raw = (self.video_filename.value or "").strip()
        if raw:
            base = raw[:-4] if raw.lower().endswith(".mp4") else raw
            slug = _safe_slug(base)
            return slug or "dataset_replay"
        member = self._source_member_for_token(selected_member_token)
        label = member.label if member is not None else selected_member_token
        slug = _safe_slug(label)
        return f"dataset_replay_{slug}" if slug else "dataset_replay"

    def _native_replay_screen_kws(
        self, selected_member_token: str
    ) -> tuple[dict[str, Any], Path | None]:
        show_display = bool(self.show_display.value)
        save_video = bool(self.save_video.value)
        screen_kws: dict[str, Any] = {
            "show_display": show_display,
            "display_every_n_steps": int(self.display_every_n_steps.value),
            "pygame_keys": self.display_shortcuts.runtime_pygame_keys(),
        }
        video_target: Path | None = None
        if show_display or save_video:
            screen_kws["vis_mode"] = "video"
        if save_video:
            media_dir = self._native_replay_video_output_dir()
            media_dir.mkdir(parents=True, exist_ok=True)
            video_name = self._native_replay_video_name(selected_member_token)
            video_target = media_dir / f"{video_name}.mp4"
            screen_kws.update(
                {
                    "save_video": True,
                    "video_file": video_name,
                    "media_dir": str(media_dir),
                    "fps": int(self.video_fps.value),
                }
            )
        screen_kws.update(self._native_body_rendering_screen_kws())
        return screen_kws, video_target

    def _native_body_rendering_screen_kws(self) -> dict[str, bool]:
        mode = str(self.native_body_rendering.value)
        if mode == self._BODY_RENDER_MIDLINE:
            return {"draw_contour": False, "draw_midline": True}
        return {}

    def _selected_time_range(self) -> tuple[float, float] | None:
        if not self.use_time_range.value:
            return None
        time_range = (
            float(self.time_start.value or 0.0),
            float(self.time_end.value or 0.0),
        )
        if time_range[1] < time_range[0]:
            time_range = (time_range[1], time_range[0])
        return time_range

    def _selected_replay_member_token(self) -> str | None:
        visible_tokens = [str(token) for token in self.member_visibility.value]
        if len(visible_tokens) == 0:
            self._set_status("Select one visible member for native replay.")
            return None
        if len(visible_tokens) > 1:
            self._set_status(
                "Native replay supports one visible member in this version."
            )
            return None
        return visible_tokens[0]

    def _source_member_for_token(self, token: str):
        if self._prepared is None:
            return None
        source = self._prepared.source
        for member in source.members:
            if str(member.token) == token:
                return member
        return None

    def _build_native_replay_parameters(
        self,
        *,
        selected_member_token: str,
        agent_indices: tuple[int, ...] | None,
        time_range: tuple[float, float] | None,
    ) -> tuple[Any, LarvaDataset | None]:
        assert self._prepared is not None
        source_member = self._source_member_for_token(selected_member_token)
        if source_member is None:
            raise ValueError("Selected replay member is unavailable.")
        prepared_member = self._prepared.members.get(selected_member_token)
        if prepared_member is None:
            raise ValueError("Prepared replay member is unavailable.")
        selected_fix_point = self._selected_native_fix_point()
        selected_fix_segment = self._selected_fix_segment()
        if selected_fix_point is None:
            selected_fix_segment = None
        else:
            native_fix_points = {
                native_idx
                for _label, native_idx in self._native_fix_points_for_member(
                    prepared_member
                )
            }
            if selected_fix_point not in native_fix_points:
                raise ValueError(
                    f"Fix point {selected_fix_point} is unavailable for native replay."
                )
            valid_orientation_values = set(
                self._orientation_options_for_native_fix_point(
                    selected_fix_point, native_fix_points
                ).values()
            )
            if selected_fix_segment is not None and (
                selected_fix_segment not in valid_orientation_values
            ):
                raise ValueError(
                    "Fix orientation is unavailable for the selected fix point."
                )

        workspace_record = source_member.workspace_record
        ref_id = source_member.registry_ref_id
        ref_dir = None
        dataset: LarvaDataset | None = None
        if workspace_record is not None:
            if workspace_record.ref_id in reg.conf.Ref.confIDs:
                ref_id = workspace_record.ref_id
            else:
                ref_id = reg.default_refID
            ref_dir = str(workspace_record.dataset_dir)
            dataset = LarvaDataset(dir=ref_dir, load_data=True)
        if not ref_id:
            raise ValueError("Replay member has no dataset reference id.")

        draw_nsegs = 2
        if self.native_body_rendering.value in [
            self._BODY_RENDER_CONTOUR,
            self._BODY_RENDER_MIDLINE,
        ]:
            draw_nsegs = None
        else:
            valid_segment_counts = set(
                self._native_segment_count_options_for_member(prepared_member).values()
            )
            selected_segment_count = int(self.native_segment_count.value)
            if selected_segment_count not in valid_segment_counts:
                raise ValueError(
                    "Segment count is unavailable for native replay on this member."
                )
            draw_nsegs = selected_segment_count

        if prepared_member.native_replay_missing_columns:
            missing = ", ".join(prepared_member.native_replay_missing_columns)
            raise ValueError(
                "Native replay is unavailable for this member: " f"missing {missing}."
            )

        track_point = int(self.track_point.value)
        if track_point == -1:
            native_track_point = prepared_member.native_default_track_point
            if native_track_point is None:
                raise ValueError(
                    "No native replay track point is available for this member."
                )
        else:
            native_track_point = (
                prepared_member.native_track_point_by_ui_track_point.get(track_point)
            )
            if native_track_point is None:
                raise ValueError(
                    f"Track point {track_point} is unavailable for native replay."
                )

        parameters = reg.gen.Replay(
            refID=ref_id,
            refDir=ref_dir,
            track_point=int(native_track_point),
            transposition=None
            if selected_fix_point is not None
            else self.transposition.value,
            agent_ids=list(agent_indices) if agent_indices is not None else [],
            time_range=time_range,
            overlap_mode=False,
            draw_Nsegs=draw_nsegs,
            close_view=bool(self.close_view.value),
            fix_point=selected_fix_point,
            fix_segment=selected_fix_segment,
        ).nestedConf
        return parameters, dataset

    def _execute_native_pygame_replay(
        self,
        *,
        parameters: Any,
        dataset: LarvaDataset | None,
        screen_kws: dict[str, Any],
        video_target: Path | None,
    ) -> None:
        launcher = None
        try:
            launcher = sim.ReplayRun(
                parameters=parameters,
                dataset=dataset,
                screen_kws=screen_kws,
                store_data=False,
            )
            launcher.run()
        except Exception as exc:
            detail = str(exc) or exc.__class__.__name__
            self._set_status(f"Native replay failed: {detail}")
            self._set_native_replay_controls_disabled(False)
            return
        finally:
            try:
                if launcher is not None and getattr(launcher, "screen_manager", None):
                    launcher.screen_manager.close()
            except Exception:
                pass
        if video_target is not None:
            self._set_status(f"Native replay finished. Video target: {video_target}.")
        else:
            self._set_status("Native pygame replay finished.")
        self._set_native_replay_controls_disabled(False)

    def _on_open_pygame_replay(self, *_: object) -> None:
        if self._prepared is None:
            self._set_status("Select a replay source to begin.")
            return
        if not bool(self.show_display.value) and not bool(self.save_video.value):
            self._set_status("Enable Show display or Save video to run native replay.")
            return
        selected_member_token = self._selected_replay_member_token()
        if selected_member_token is None:
            return
        try:
            agent_indices = parse_agent_indices(self.agent_indices.value)
        except ValueError as exc:
            self._set_status(f"Invalid Agent indices: {exc}")
            return
        time_range = self._selected_time_range()
        shortcut_errors = self.display_shortcuts.validate()
        if shortcut_errors:
            self._set_status("Display shortcut errors: " + "; ".join(shortcut_errors))
            return
        try:
            parameters, dataset = self._build_native_replay_parameters(
                selected_member_token=selected_member_token,
                agent_indices=agent_indices,
                time_range=time_range,
            )
        except Exception as exc:
            self._set_status(f"Native replay failed: {exc}")
            return
        screen_kws, video_target = self._native_replay_screen_kws(selected_member_token)

        if video_target is not None:
            self._set_status(
                "Starting native replay video export. "
                "The portal UI may be unresponsive until export finishes. "
                f"Video target: {video_target}."
            )
        else:
            self._set_status(
                "Starting native pygame replay. "
                "The portal UI may be unresponsive until the replay window closes."
            )
        self._set_native_replay_controls_disabled(True)
        document = pn.state.curdoc
        if document is not None:
            document.add_next_tick_callback(
                lambda: self._execute_native_pygame_replay(
                    parameters=parameters,
                    dataset=dataset,
                    screen_kws=screen_kws,
                    video_target=video_target,
                )
            )
            return
        self._execute_native_pygame_replay(
            parameters=parameters,
            dataset=dataset,
            screen_kws=screen_kws,
            video_target=video_target,
        )

    def _render(self) -> None:
        if self._prepared is None:
            return
        visible_tokens = [str(token) for token in self.member_visibility.value]
        transposition = self.transposition.value
        try:
            agent_indices = parse_agent_indices(self.agent_indices.value)
        except ValueError as exc:
            self._set_status(f"Invalid Agent indices: {exc}")
            return
        try:
            state = build_render_state(
                self._prepared,
                tick=int(self.tick_player.value),
                member_tokens=visible_tokens,
                show_positions=bool(self.show_positions.value),
                show_ids=bool(self.show_ids.value),
                show_tracks=bool(self.show_tracks.value),
                trail_length=int(self.trail_length.value),
                transposition=transposition,
                track_point=int(self.track_point.value),
                agent_indices=agent_indices,
                time_range=None,
                show_dispersal_ring=bool(self.show_dispersal.value),
                show_heads=bool(self.show_heads.value),
                show_midlines=bool(self.show_midlines.value),
                show_segments=bool(self.show_segments.value),
                show_body_contours=bool(self.show_body_contours.value),
            )
        except ValueError as exc:
            self._set_status(f"Replay render error: {exc}")
            return

        allow_static_layers = transposition == "arena"
        if self._prepared.members:
            first_member = next(iter(self._prepared.members.values()))
            show_arena_outline = self._show_arena_outline_for_mode(
                transposition,
                first_member.coordinate_origin,
            )
            static_state_key = (
                self._prepared.source.token,
                transposition,
                allow_static_layers,
                show_arena_outline,
            )
            if static_state_key != self._last_static_state_key:
                self.canvas.set_state(
                    build_environment_state_for_member(
                        first_member,
                        allow_static_layers=allow_static_layers,
                        show_arena_outline=show_arena_outline,
                    )
                )
                self._last_static_state_key = static_state_key
        self.canvas.set_larva_frame(state.frame)
        self.canvas.set_dynamic_overlays(rings=state.rings)

    @staticmethod
    def _show_arena_outline_for_mode(
        transposition: str | None, coordinate_origin: str
    ) -> bool:
        if transposition == "arena":
            return True
        if transposition is None and coordinate_origin == "centered":
            return True
        return False

    def view(self) -> pn.viewable.Viewable:
        intro = pn.pane.HTML(
            (
                '<div class="lw-dataset-replay-intro">'
                "Replay imported workspace datasets, single experiment run outputs, "
                "and registry references with a read-only, frame-based viewer."
                "</div>"
            ),
            margin=0,
        )
        pygame_replay = _control_tile(
            "Pygame replay",
            _subcontrol_tile(
                "Time range",
                self.use_time_range,
                self.time_start,
                self.time_end,
            ),
            _subcontrol_tile(
                "Close inspection",
                self.fix_point,
                self.close_view,
                self.fix_segment,
            ),
            _subcontrol_tile(
                "Body rendering",
                self.native_body_rendering,
                self.native_segment_count,
            ),
            _subcontrol_tile(
                "Output",
                self.show_display,
                self.display_every_n_steps,
                self.save_video,
                self.video_filename,
                self.video_fps,
                pn.Row(
                    self.display_shortcuts_link,
                    self.open_pygame_replay_btn,
                    sizing_mode="stretch_width",
                    margin=(4, 0, 0, 0),
                ),
            ),
        )
        controls = pn.Column(
            _control_tile("Source", self.source_select, self.member_visibility),
            _control_tile(
                "Display",
                self.show_positions,
                self.show_ids,
                self.show_heads,
                self.show_midlines,
                self.show_segments,
                self.show_body_contours,
            ),
            _control_tile(
                "Motion", self.show_tracks, self.trail_length, self.show_dispersal
            ),
            _control_tile(
                "Coordinates", self.transposition, self.track_point, self.agent_indices
            ),
            pygame_replay,
            width=360,
            css_classes=["lw-dataset-replay-controls"],
        )
        main = pn.Column(
            self.tick_player,
            self.canvas.view(),
            self.status_pane,
            sizing_mode="stretch_width",
        )
        return pn.Column(
            intro,
            self.display_shortcuts_dialog,
            pn.Row(controls, main, sizing_mode="stretch_width"),
            css_classes=["lw-dataset-replay-root"],
            sizing_mode="stretch_width",
        )


def dataset_replay_app() -> pn.viewable.Viewable:
    pn.extension(
        raw_css=[PORTAL_RAW_CSS, DATASET_REPLAY_RAW_CSS, DISPLAY_SHORTCUTS_RAW_CSS]
    )
    controller = _DatasetReplayController()
    template = pn.template.MaterialTemplate(
        title="",
        header_background="#b0b4c2",
        header_color="#111111",
    )
    template.header.append(build_app_header(title="Dataset Replay"))
    template.main.append(controller.view())
    return template
