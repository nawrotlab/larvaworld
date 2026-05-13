from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import panel as pn
import param

from larvaworld.lib import util
from larvaworld.lib.reg import keymap
from larvaworld.portal.workspace import (
    WorkspaceError,
    get_active_workspace,
    get_workspace_dir,
)

__all__: list[str] = [
    "DisplayShortcutsConfig",
    "DisplayShortcutsController",
]


@dataclass(frozen=True)
class ShortcutFieldSpec:
    field: str
    section: str
    action: str
    label: str


_SHORTCUT_FIELDS: tuple[ShortcutFieldSpec, ...] = (
    ShortcutFieldSpec("pause", "simulation", "pause", "Pause / resume"),
    ShortcutFieldSpec("snapshot", "simulation", "snapshot", "Snapshot"),
    ShortcutFieldSpec(
        "larva_collisions", "simulation", "larva_collisions", "Larva collisions"
    ),
    ShortcutFieldSpec("visible_clock", "aux", "visible_clock", "Clock"),
    ShortcutFieldSpec("visible_scale", "aux", "visible_scale", "Scale"),
    ShortcutFieldSpec("visible_state", "aux", "visible_state", "State overlay"),
    ShortcutFieldSpec("visible_ids", "aux", "visible_ids", "Agent IDs"),
    ShortcutFieldSpec("visible_trails", "draw", "visible_trails", "Toggle trails"),
    ShortcutFieldSpec(
        "trail_duration_up", "draw", "▲ trail duration", "Trail duration +"
    ),
    ShortcutFieldSpec(
        "trail_duration_down", "draw", "▼ trail duration", "Trail duration -"
    ),
    ShortcutFieldSpec("trail_color", "draw", "trail_color", "Trail color"),
    ShortcutFieldSpec(
        "black_background", "color", "black_background", "Black background"
    ),
    ShortcutFieldSpec("random_colors", "color", "random_colors", "Random colors"),
    ShortcutFieldSpec("color_behavior", "color", "color_behavior", "Color by behavior"),
    ShortcutFieldSpec("screen_move_up", "screen", "move up", "Pan up"),
    ShortcutFieldSpec("screen_move_down", "screen", "move down", "Pan down"),
    ShortcutFieldSpec("screen_move_left", "screen", "move left", "Pan left"),
    ShortcutFieldSpec("screen_move_right", "screen", "move right", "Pan right"),
)

_FIELD_TO_SPEC = {spec.field: spec for spec in _SHORTCUT_FIELDS}
_SECTION_ORDER = ("simulation", "aux", "draw", "color", "screen")
_SECTION_LABELS = {
    "simulation": "Simulation",
    "aux": "Display overlays",
    "draw": "Draw",
    "color": "Colors",
    "screen": "Screen",
}

_SECTION_TO_FIELDS: dict[str, list[ShortcutFieldSpec]] = {
    section: [] for section in _SECTION_ORDER
}
for _spec in _SHORTCUT_FIELDS:
    _SECTION_TO_FIELDS[_spec.section].append(_spec)


class DisplayShortcutsConfig(param.Parameterized):
    pause = param.String(default="space", doc="Pause or resume the live display.")
    snapshot = param.String(default="i", doc="Capture a display snapshot.")
    larva_collisions = param.String(default="y", doc="Toggle larva collisions.")
    visible_clock = param.String(default="t", doc="Toggle the clock overlay.")
    visible_scale = param.String(default="n", doc="Toggle the scale overlay.")
    visible_state = param.String(default="s", doc="Toggle the state overlay.")
    visible_ids = param.String(default="tab", doc="Toggle larva IDs.")
    visible_trails = param.String(default="p", doc="Toggle larva trails.")
    trail_duration_up = param.String(default="+", doc="Increase trail duration.")
    trail_duration_down = param.String(default="-", doc="Decrease trail duration.")
    trail_color = param.String(default="x", doc="Cycle trail color mode.")
    black_background = param.String(default="g", doc="Toggle black background.")
    random_colors = param.String(default="r", doc="Toggle random larva colors.")
    color_behavior = param.String(default="b", doc="Color larvae by behavior.")
    screen_move_up = param.String(default="UP", doc="Pan the display up.")
    screen_move_down = param.String(default="DOWN", doc="Pan the display down.")
    screen_move_left = param.String(default="LEFT", doc="Pan the display left.")
    screen_move_right = param.String(default="RIGHT", doc="Pan the display right.")


class DisplayShortcutsController(param.Parameterized):
    config = param.ClassSelector(class_=DisplayShortcutsConfig, constant=True)
    dirty = param.Boolean(default=False)
    status = param.String(default="")

    def __init__(self, **params: Any) -> None:
        super().__init__(config=DisplayShortcutsConfig(), **params)
        self._defaults = keymap.default_controls()
        self._defaults_subset = self._extract_curated_keys(
            self._defaults.get("keys", {})
        )
        self._status_pane = pn.pane.Markdown("", margin=(6, 0, 0, 0))
        self._save_btn = pn.widgets.Button(
            name="Save shortcuts",
            button_type="primary",
            width=132,
        )
        self._reset_btn = pn.widgets.Button(
            name="Reset all to defaults",
            button_type="default",
            width=164,
        )
        self._save_btn.on_click(self._on_save)
        self._reset_btn.on_click(self._on_reset)
        self._watchers = [
            self.config.param.watch(self._on_config_change, spec.field)
            for spec in _SHORTCUT_FIELDS
        ]
        self.load_from_workspace()
        self._update_controls_state()

    @staticmethod
    def _extract_curated_keys(raw_keys: Any) -> util.AttrDict:
        result: dict[str, dict[str, str]] = {}
        for spec in _SHORTCUT_FIELDS:
            section_values = (
                raw_keys.get(spec.section, {}) if isinstance(raw_keys, dict) else {}
            )
            value = (
                section_values.get(spec.action)
                if isinstance(section_values, dict)
                else None
            )
            if not isinstance(value, str):
                continue
            result.setdefault(spec.section, {})[spec.action] = value
        return util.AttrDict(result)

    def _current_keys(self) -> util.AttrDict:
        result: dict[str, dict[str, str]] = {}
        for spec in _SHORTCUT_FIELDS:
            value = str(getattr(self.config, spec.field)).strip()
            result.setdefault(spec.section, {})[spec.action] = value
        return util.AttrDict(result)

    def _apply_keys(self, keys: dict[str, dict[str, str]]) -> None:
        for spec in _SHORTCUT_FIELDS:
            section_values = (
                keys.get(spec.section, {}) if isinstance(keys, dict) else {}
            )
            value = (
                section_values.get(spec.action)
                if isinstance(section_values, dict)
                else None
            )
            if isinstance(value, str) and value.strip():
                setattr(self.config, spec.field, value.strip())
        self.dirty = False
        self.status = ""
        self._status_pane.object = ""
        self._update_controls_state()

    def _shortcuts_file(self) -> Path:
        metadata_dir = get_workspace_dir("metadata")
        return metadata_dir / "display_shortcuts.json"

    def _workspace_available(self) -> bool:
        return get_active_workspace() is not None

    def _update_controls_state(self) -> None:
        workspace_ok = self._workspace_available()
        self._save_btn.disabled = (not self.dirty) or (not workspace_ok)
        self._reset_btn.disabled = not workspace_ok

    def _on_config_change(self, *_: Any) -> None:
        self.dirty = True
        self.status = ""
        self._status_pane.object = ""
        self._update_controls_state()

    def load_from_workspace(self) -> None:
        keys = self._defaults_subset
        try:
            path = self._shortcuts_file()
        except WorkspaceError:
            self._apply_keys(keys)
            return
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                loaded_keys = payload.get("keys", {})
                if isinstance(loaded_keys, dict):
                    merged = keymap.merge_controls(
                        {"keys": keys},
                        {"keys": loaded_keys},
                    )
                    keys = self._extract_curated_keys(merged.get("keys", {}))
            except (OSError, json.JSONDecodeError):
                pass
        self._apply_keys(keys)

    def validate(self) -> list[str]:
        return keymap.validate_shortcut_conf(self._current_keys())

    def runtime_pygame_keys(self) -> dict[str, str]:
        merged = keymap.merge_controls(
            self._defaults,
            {"keys": self._current_keys()},
        )
        return dict(merged.get("pygame_keys", {}))

    def _on_save(self, *_: Any) -> None:
        errors = self.validate()
        if errors:
            msg = "Cannot save shortcuts:\n" + "\n".join(f"- {err}" for err in errors)
            self.status = msg
            self._status_pane.object = msg
            return
        if not self._workspace_available():
            msg = "Cannot save shortcuts: no active workspace."
            self.status = msg
            self._status_pane.object = msg
            self._update_controls_state()
            return
        try:
            path = self._shortcuts_file()
            payload = {
                "version": 1,
                "keys": self._current_keys(),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except (WorkspaceError, OSError) as exc:
            msg = f"Cannot save shortcuts: {exc}"
            self.status = msg
            self._status_pane.object = msg
            return
        self.dirty = False
        self.status = "Shortcuts saved."
        self._status_pane.object = self.status
        self._update_controls_state()

    def _on_reset(self, *_: Any) -> None:
        self._apply_keys(self._defaults_subset)
        self.dirty = True
        self.status = "Defaults restored in memory. Save shortcuts to persist."
        self._status_pane.object = self.status
        self._update_controls_state()

    def view(self) -> pn.viewable.Viewable:
        groups: list[pn.viewable.Viewable] = []
        for section in _SECTION_ORDER:
            rows: list[pn.viewable.Viewable] = []
            for spec in _SECTION_TO_FIELDS[section]:
                widget = pn.widgets.TextInput.from_param(
                    self.config.param[spec.field],
                    name="",
                    width=84,
                    margin=0,
                )
                default_value = self._defaults_subset.get(spec.section, {}).get(
                    spec.action, ""
                )
                rows.append(
                    pn.Row(
                        pn.pane.Markdown(spec.label, width=170, margin=(3, 0, 0, 0)),
                        widget,
                        pn.pane.Markdown(
                            f'<span style="font-size:11px;color:#6b7280;">Default: {default_value}</span>',
                            margin=(3, 0, 0, 0),
                        ),
                        sizing_mode="stretch_width",
                        margin=(0, 0, 4, 0),
                    )
                )
            groups.append(
                pn.Column(
                    pn.pane.Markdown(
                        f"**{_SECTION_LABELS[section]}**", margin=(4, 0, 4, 0)
                    ),
                    *rows,
                    margin=(0, 0, 4, 0),
                    sizing_mode="stretch_width",
                )
            )
        actions = pn.Row(self._save_btn, self._reset_btn, sizing_mode="stretch_width")
        return pn.Column(
            *groups,
            actions,
            self._status_pane,
            sizing_mode="stretch_width",
            margin=0,
        )
