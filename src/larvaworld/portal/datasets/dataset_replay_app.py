from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path

import panel as pn

from larvaworld.lib import reg
from larvaworld.portal.canvas_widgets import EnvironmentCanvas
from larvaworld.portal.datasets.replay_data import (
    build_environment_state_for_member,
    build_render_state,
    build_source_catalog,
    parse_agent_indices,
    prepare_replay_source,
)
from larvaworld.portal.datasets.replay_models import PreparedReplaySource, ReplaySource
from larvaworld.portal.panel_components import PORTAL_RAW_CSS, build_app_header
from larvaworld.portal.workspace import get_active_workspace


__all__ = ["_DatasetReplayController", "dataset_replay_app"]


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
""".strip()


def _status_html(text: str) -> str:
    return f'<div class="lw-dataset-replay-status">{escape(text)}</div>'


class _DatasetReplayController:
    def __init__(self) -> None:
        self.workspace = get_active_workspace()
        self.canvas = EnvironmentCanvas(snap_heads_to_midline=False)

        self._sources = build_source_catalog(self.workspace)
        self._source_by_token = {source.token: source for source in self._sources}
        self._prepared: PreparedReplaySource | None = None
        self._last_static_state_key: tuple[str, str | None, bool, bool] | None = None

        self.source_select = pn.widgets.Select(name="Source", options={})

        self.member_visibility = pn.widgets.CheckBoxGroup(
            name="Visible members", options=[], value=[]
        )
        self.show_positions = pn.widgets.Checkbox(name="Positions", value=True)
        self.show_ids = pn.widgets.Checkbox(name="IDs", value=False)
        self.show_tracks = pn.widgets.Checkbox(name="Tracks", value=False)
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
        self.agent_indices = pn.widgets.TextInput(
            name="Agent indices",
            placeholder="empty = all; e.g. 0,1,2",
            value="",
        )
        self.tick_player = pn.widgets.Player(
            name="Tick", start=0, end=1, step=1, value=0, interval=100
        )
        self.time_start = pn.widgets.FloatInput(
            name="Time start (s)", value=0.0, step=1.0
        )
        self.time_end = pn.widgets.FloatInput(name="Time end (s)", value=0.0, step=1.0)
        self.use_time_range = pn.widgets.Checkbox(name="Apply time range", value=False)
        self.status_pane = pn.pane.HTML(
            _status_html("Select a replay source to begin."), margin=0
        )

        self.source_select.param.watch(self._on_source_change, "value")
        self.member_visibility.param.watch(self._on_any_control_change, "value")
        for widget in (
            self.show_positions,
            self.show_ids,
            self.show_tracks,
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

        self._reload_source_options()

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
        self._prepared = prepare_replay_source(source)
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
        self._render()

    def _on_any_control_change(self, _event=None) -> None:
        self._render()

    def _render(self) -> None:
        if self._prepared is None:
            return
        visible_tokens = [str(token) for token in self.member_visibility.value]
        transposition = self.transposition.value
        time_range = None
        if self.use_time_range.value:
            time_range = (
                float(self.time_start.value or 0.0),
                float(self.time_end.value or 0.0),
            )
            if time_range[1] < time_range[0]:
                time_range = (time_range[1], time_range[0])
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
                time_range=time_range,
                show_dispersal_ring=bool(self.show_dispersal.value),
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
        controls = pn.Column(
            self.source_select,
            self.member_visibility,
            pn.layout.Divider(),
            self.show_positions,
            self.show_ids,
            self.show_tracks,
            self.show_dispersal,
            self.trail_length,
            pn.layout.Divider(),
            self.transposition,
            self.track_point,
            self.agent_indices,
            self.use_time_range,
            self.time_start,
            self.time_end,
            width=360,
            sizing_mode="fixed",
        )
        main = pn.Column(
            self.tick_player,
            self.canvas.view(),
            self.status_pane,
            sizing_mode="stretch_width",
        )
        return pn.Column(
            intro,
            pn.Row(controls, main, sizing_mode="stretch_width"),
            css_classes=["lw-dataset-replay-root"],
            sizing_mode="stretch_width",
        )


def dataset_replay_app() -> pn.viewable.Viewable:
    pn.extension(raw_css=[PORTAL_RAW_CSS, DATASET_REPLAY_RAW_CSS])
    controller = _DatasetReplayController()
    template = pn.template.MaterialTemplate(
        title="",
        header_background="#b0b4c2",
        header_color="#111111",
    )
    template.header.append(build_app_header(title="Dataset Replay"))
    template.main.append(controller.view())
    return template
