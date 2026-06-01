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
    "DISPLAY_SHORTCUTS_RAW_CSS",
    "DisplayShortcutsConfig",
    "DisplayShortcutsController",
    "DisplayShortcutsDialog",
    "build_display_shortcuts_dialog",
]


DISPLAY_SHORTCUTS_RAW_CSS = """
.lw-inline-help-link .bk-btn {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: #2f4858 !important;
  font-size: 11px !important;
  line-height: 1.3 !important;
  padding: 0 !important;
  min-height: 0 !important;
  text-decoration: underline;
}

.lw-inline-help-link .bk-btn:hover,
.lw-inline-help-link .bk-btn:focus {
  color: #1f3542 !important;
  text-decoration: underline;
}

.lw-display-shortcuts-overlay {
  position: fixed;
  inset: 0;
  z-index: 2000;
  background: rgba(15, 23, 42, 0.58);
  padding: 24px;
}

.lw-display-shortcuts-dialog {
  position: fixed;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  width: min(560px, 95vw);
  max-height: 85vh;
  overflow: auto;
  border-radius: 10px;
  border: 1px solid rgba(60, 60, 60, 0.26);
  background: #dddddd;
  box-shadow: 0 16px 36px rgba(15, 23, 42, 0.34);
  color: rgba(17, 17, 17, 0.86);
  padding: 12px 14px;
}

.lw-display-shortcuts-note {
  font-size: 12px;
  line-height: 1.45;
  margin: 0 0 8px 0;
}

.lw-display-shortcuts-drag-handle {
  display: flex;
  align-items: center;
  justify-content: space-between;
  cursor: default;
  user-select: none;
  margin: -2px -4px 10px -4px;
  padding: 6px 8px;
  border-radius: 8px;
  background: rgba(60, 60, 60, 0.1);
}

.lw-display-shortcuts-dialog-title {
  margin: 0;
  font-size: 13px;
  line-height: 1.3;
  font-weight: 600;
}

.lw-display-shortcuts-table-wrap {
  border-radius: 8px;
  border: 1px solid rgba(17, 17, 17, 0.1);
  background: rgba(181,194,176,0.14);
  padding: 10px 10px 8px 10px;
}

.lw-display-shortcuts-table-wrap h4 {
  margin: 0 0 8px 0;
  font-size: 13px;
  line-height: 1.2;
}

.lw-display-shortcuts-table-wrap table {
  width: 100%;
  border-collapse: collapse;
}

.lw-display-shortcuts-table-wrap td {
  padding: 3px 0;
  vertical-align: top;
}

.lw-display-shortcuts-table-wrap kbd {
  display: inline-block;
  min-width: 28px;
  border-radius: 5px;
  border: 1px solid rgba(17, 17, 17, 0.18);
  background: rgba(255, 255, 255, 0.82);
  padding: 1px 5px;
  text-align: center;
  font: 11px/1.35 monospace;
  color: rgba(17, 17, 17, 0.78);
}
""".strip()


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
    editing = param.Boolean(default=False)
    capturing_field = param.String(default="")

    def __init__(self, **params: Any) -> None:
        super().__init__(config=DisplayShortcutsConfig(), **params)
        self._defaults = keymap.default_controls()
        self._defaults_subset = self._extract_curated_keys(
            self._defaults.get("keys", {})
        )
        self._status_pane = pn.pane.Markdown("", margin=(6, 0, 0, 0))
        self._capture_bridge = pn.widgets.TextInput(value="", visible=False)
        self._editing_bridge = pn.widgets.TextInput(value="0", visible=False)
        self._edit_btn = pn.widgets.Button(
            name="Edit shortcuts",
            button_type="default",
            width=120,
        )
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
        self._key_buttons: dict[str, pn.widgets.Button] = {}
        self._view_obj: pn.viewable.Viewable | None = None
        self._edit_btn.on_click(self._toggle_editing)
        self._save_btn.on_click(self._on_save)
        self._reset_btn.on_click(self._on_reset)
        self._capture_bridge.param.watch(self._on_capture_payload, "value")
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
        self._edit_btn.disabled = not workspace_ok
        self._edit_btn.name = "Done editing" if self.editing else "Edit shortcuts"
        self._editing_bridge.value = "1" if self.editing else "0"
        self._refresh_key_buttons()

    def _on_config_change(self, *_: Any) -> None:
        self.dirty = True
        self.status = ""
        self._status_pane.object = ""
        self._update_controls_state()

    @staticmethod
    def _normalize_browser_key(raw_key: str) -> str | None:
        key_raw = str(raw_key)
        if key_raw in {" ", "Space", "Spacebar"}:
            return "space"
        key = key_raw.strip()
        if not key:
            return None
        if key in {"Escape", "Esc"}:
            return "__cancel__"
        fixed = {
            "Tab": "tab",
            "Delete": "del",
            "ArrowUp": "UP",
            "ArrowDown": "DOWN",
            "ArrowLeft": "LEFT",
            "ArrowRight": "RIGHT",
        }.get(key)
        if fixed is not None:
            return fixed
        if len(key) == 1:
            if key.isalpha():
                return key.lower()
            if key.isdigit() or key in {"+", "-"}:
                return key
        return key

    @staticmethod
    def _format_key_label(key: str) -> str:
        value = str(key)
        if value == "":
            return ""
        value = value.strip()
        mapping = {"space": "Space", "tab": "Tab", "del": "Delete"}
        if value in mapping:
            return mapping[value]
        if len(value) == 1 and value.isalpha():
            return value.upper()
        return value

    def _find_duplicate_assignment(
        self, key: str, excluding_field: str
    ) -> ShortcutFieldSpec | None:
        for spec in _SHORTCUT_FIELDS:
            if spec.field == excluding_field:
                continue
            current = str(getattr(self.config, spec.field))
            if current == key:
                return spec
        return None

    def _refresh_key_buttons(self) -> None:
        for field, button in self._key_buttons.items():
            spec = _FIELD_TO_SPEC[field]
            if self.capturing_field == field:
                button.name = "Press key..."
                button.button_type = "primary"
            else:
                current = getattr(self.config, field)
                button.name = self._format_key_label(str(current))
                button.button_type = "default"
            button.disabled = not self.editing
            button.tooltip = spec.label

    def _cancel_capture(self, *, clear_status: bool = False) -> None:
        self.capturing_field = ""
        self._refresh_key_buttons()
        if clear_status:
            self.status = ""
            self._status_pane.object = ""

    def _toggle_editing(self, *_: Any) -> None:
        if self.editing:
            self._cancel_capture(clear_status=True)
            self.editing = False
        else:
            self.editing = True
            self.status = "Edit mode enabled. Click a shortcut and press a key."
            self._status_pane.object = self.status
        self._update_controls_state()

    def _start_capture(self, field: str) -> None:
        if not self.editing:
            return
        spec = _FIELD_TO_SPEC.get(field)
        if spec is None:
            return
        self.capturing_field = field
        self.status = f'Press a key for "{spec.label}". Press Esc to cancel.'
        self._status_pane.object = self.status
        self._refresh_key_buttons()

    def _on_capture_payload(self, event: param.parameterized.Event) -> None:
        payload = str(event.new).strip()
        if not payload:
            return
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return
        field = str(data.get("field", "")).strip()
        key_value = data.get("key", "")
        key = key_value if isinstance(key_value, str) else str(key_value)
        if not field or key == "":
            return
        self._apply_captured_key(field, key)

    def _apply_captured_key(self, field: str, raw_key: str) -> None:
        if field != self.capturing_field or field not in _FIELD_TO_SPEC:
            return
        normalized = self._normalize_browser_key(raw_key)
        if normalized == "__cancel__":
            self.status = "Shortcut capture cancelled."
            self._status_pane.object = self.status
            self._cancel_capture()
            return
        if not normalized or not keymap.validate_key_name(normalized):
            setattr(self.config, field, "")
            self.status = (
                f'Unsupported key "{raw_key}". Shortcut cleared; '
                "choose a supported key before saving."
            )
            self._status_pane.object = self.status
            self._cancel_capture()
            return
        duplicate_spec = self._find_duplicate_assignment(
            normalized, excluding_field=field
        )
        if duplicate_spec is not None:
            self.status = (
                f"Cannot assign {self._format_key_label(normalized)} to "
                f"{_FIELD_TO_SPEC[field].label}: already used by {duplicate_spec.label}."
            )
            self._status_pane.object = self.status
            self._cancel_capture()
            return
        setattr(self.config, field, normalized)
        self.status = (
            f"{_FIELD_TO_SPEC[field].label} updated to "
            f"{self._format_key_label(normalized)}."
        )
        self._status_pane.object = self.status
        self._cancel_capture()

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
        if self._view_obj is not None:
            return self._view_obj
        groups: list[pn.viewable.Viewable] = []
        for section in _SECTION_ORDER:
            rows: list[pn.viewable.Viewable] = []
            for spec in _SECTION_TO_FIELDS[section]:
                button = pn.widgets.Button(
                    name=self._format_key_label(str(getattr(self.config, spec.field))),
                    button_type="default",
                    width=100,
                    margin=0,
                )
                self._key_buttons[spec.field] = button
                button.on_click(
                    lambda _event, field=spec.field: self._start_capture(field)
                )
                button.jscallback(
                    args={
                        "bridge": self._capture_bridge,
                        "editingBridge": self._editing_bridge,
                        "fieldName": spec.field,
                    },
                    clicks="""
if (editingBridge.value !== "1") {
  return;
}
if (window.__larvaworldShortcutCaptureCleanup) {
  window.__larvaworldShortcutCaptureCleanup();
}
setTimeout(() => {
  const handler = (ev) => {
    ev.preventDefault();
    ev.stopPropagation();
    bridge.value = JSON.stringify({
      field: fieldName,
      key: ev.key,
      nonce: Date.now(),
    });
    bridge.change.emit();
    window.removeEventListener("keydown", handler, true);
    window.__larvaworldShortcutCaptureCleanup = null;
  };
  window.addEventListener("keydown", handler, true);
  window.__larvaworldShortcutCaptureCleanup = () => {
    window.removeEventListener("keydown", handler, true);
    window.__larvaworldShortcutCaptureCleanup = null;
  };
}, 0);
""",
                )
                default_value = self._defaults_subset.get(spec.section, {}).get(
                    spec.action, ""
                )
                rows.append(
                    pn.Row(
                        pn.pane.Markdown(spec.label, width=170, margin=(3, 0, 0, 0)),
                        button,
                        pn.pane.Markdown(
                            f'<span style="font-size:11px;color:#6b7280;">Default: {self._format_key_label(default_value)}</span>',
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
                    css_classes=["lw-display-shortcuts-table-wrap"],
                )
            )
        actions = pn.Row(
            self._edit_btn,
            self._save_btn,
            self._reset_btn,
            sizing_mode="stretch_width",
        )
        self._view_obj = pn.Column(
            *groups,
            actions,
            self._status_pane,
            self._capture_bridge,
            self._editing_bridge,
            sizing_mode="stretch_width",
            margin=0,
        )
        self._refresh_key_buttons()
        self._update_controls_state()
        return self._view_obj


@dataclass
class DisplayShortcutsDialog:
    controller: DisplayShortcutsController
    open_button: pn.widgets.Button
    close_button: pn.widgets.Button
    dialog: pn.Column

    def open(self, *_: object) -> None:
        self.dialog.visible = True

    def close(self, *_: object) -> None:
        self.dialog.visible = False

    def set_disabled(self, disabled: bool) -> None:
        self.open_button.disabled = bool(disabled)
        self.close_button.disabled = bool(disabled)
        if disabled:
            self.dialog.visible = False


def build_display_shortcuts_dialog(*, note: str) -> DisplayShortcutsDialog:
    controller = DisplayShortcutsController()
    open_button = pn.widgets.Button(
        name="Display Shortcuts",
        button_type="light",
        css_classes=["lw-inline-help-link"],
        margin=(4, 0, 0, 0),
        width_policy="min",
    )
    close_button = pn.widgets.Button(
        name="Close",
        button_type="default",
        width=88,
        margin=0,
    )
    dialog = pn.Column(
        pn.Column(
            pn.Row(
                pn.pane.Markdown(
                    "<p class='lw-display-shortcuts-dialog-title'>Display Shortcuts</p>",
                    margin=0,
                ),
                pn.Spacer(sizing_mode="stretch_width"),
                close_button,
                css_classes=["lw-display-shortcuts-drag-handle"],
                sizing_mode="stretch_width",
                margin=0,
            ),
            pn.pane.Markdown(
                note,
                css_classes=["lw-display-shortcuts-note"],
                margin=0,
            ),
            controller.view(),
            css_classes=["lw-display-shortcuts-dialog"],
            sizing_mode="fixed",
        ),
        visible=False,
        css_classes=["lw-display-shortcuts-overlay"],
        sizing_mode="stretch_width",
        margin=0,
    )
    wrapper = DisplayShortcutsDialog(
        controller=controller,
        open_button=open_button,
        close_button=close_button,
        dialog=dialog,
    )
    open_button.on_click(wrapper.open)
    close_button.on_click(wrapper.close)
    return wrapper
