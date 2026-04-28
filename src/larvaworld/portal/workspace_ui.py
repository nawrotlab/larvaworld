from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable
from typing import Literal

import panel as pn

from larvaworld.portal.path_picker import pick_directory
from larvaworld.portal.workspace import (
    WorkspaceState,
    clear_active_workspace_path,
    get_active_workspace,
    get_active_workspace_path,
    initialize_workspace,
    set_active_workspace_path,
    validate_workspace,
)


def _short_path(path: Path, *, keep_parts: int = 3) -> str:
    parts = path.parts
    if len(parts) <= keep_parts:
        return str(path)
    return str(Path("...").joinpath(*parts[-keep_parts:]))


def _workspace_chip_html(workspace: WorkspaceState | None) -> str:
    if workspace is None:
        return (
            '<div class="lw-portal-workspace-chip lw-portal-workspace-chip--missing">'
            '<span class="lw-portal-workspace-chip-label">Workspace</span>'
            '<span class="lw-portal-workspace-chip-value">Not set</span>'
            "</div>"
        )

    return (
        '<div class="lw-portal-workspace-chip" '
        f'title="{escape(str(workspace.root))}">'
        '<span class="lw-portal-workspace-chip-label">Workspace</span>'
        f'<span class="lw-portal-workspace-chip-value">{escape(workspace.name)}</span>'
        f'<span class="lw-portal-workspace-chip-path">{escape(_short_path(workspace.root))}</span>'
        "</div>"
    )


def _workspace_led_html(workspace: WorkspaceState | None) -> str:
    active = workspace is not None
    title = "Workspace configured" if active else "Workspace not configured"
    cls = "lw-portal-workspace-led"
    if active:
        cls += " lw-portal-workspace-led--active"
    else:
        cls += " lw-portal-workspace-led--inactive"
    return f'<div class="{cls}" title="{escape(title)}" aria-label="{escape(title)}"></div>'


def _status_html(
    *,
    text: str,
    tone: str = "neutral",
    detail: str | None = None,
    theme: Literal["light", "dark"] = "light",
) -> str:
    style_map = {
        "light": {
            "neutral": (
                "border:1px solid rgba(0,0,0,0.10);"
                "background:rgba(0,0,0,0.03);"
                "color:rgba(17,17,17,0.84);"
            ),
            "success": (
                "border:1px solid rgba(62,124,67,0.24);"
                "background:rgba(62,124,67,0.10);"
                "color:rgba(17,17,17,0.84);"
            ),
            "warning": (
                "border:1px solid rgba(176,112,33,0.28);"
                "background:rgba(245,161,66,0.12);"
                "color:rgba(17,17,17,0.84);"
            ),
            "danger": (
                "border:1px solid rgba(160,40,40,0.24);"
                "background:rgba(160,40,40,0.10);"
                "color:rgba(17,17,17,0.84);"
            ),
        },
        "dark": {
            "neutral": (
                "border:1px solid rgba(148,163,184,0.28);"
                "background:rgba(255,255,255,0.08);"
                "color:rgba(241,245,249,0.94);"
            ),
            "success": (
                "border:1px solid rgba(134,239,172,0.32);"
                "background:rgba(22,101,52,0.28);"
                "color:rgba(241,245,249,0.94);"
            ),
            "warning": (
                "border:1px solid rgba(245,161,66,0.34);"
                "background:rgba(245,161,66,0.16);"
                "color:rgba(241,245,249,0.94);"
            ),
            "danger": (
                "border:1px solid rgba(248,113,113,0.34);"
                "background:rgba(127,29,29,0.24);"
                "color:rgba(241,245,249,0.94);"
            ),
        },
    }
    theme_styles = style_map.get(theme, style_map["light"])
    tone_style = theme_styles.get(tone, theme_styles["neutral"])
    detail_color = (
        "rgba(241,245,249,0.84)" if theme == "dark" else "rgba(17,17,17,0.72)"
    )
    detail_html = ""
    if detail:
        detail_html = (
            '<div class="lw-portal-workspace-status-detail" '
            f'style="margin-top:4px;font-size:11px;word-break:break-word;color:{detail_color};">'
            f"{escape(detail)}"
            "</div>"
        )
    theme_class = ""
    if theme == "dark":
        theme_class = " lw-portal-workspace-status--theme-dark"
    return (
        "<div "
        f'class="lw-portal-workspace-status lw-portal-workspace-status--{escape(tone)}{theme_class}" '
        f'style="padding:8px 10px;border-radius:10px;font-size:12px;line-height:1.35;{tone_style}">'
        f"{escape(text)}"
        f"{detail_html}"
        "</div>"
    )


def _default_workspace_candidate() -> Path:
    active = get_active_workspace_path()
    if active is not None:
        return active
    return Path.home() / "Documents" / "Larvaworld" / "workspace"


@dataclass
class WorkspaceUiController:
    theme: Literal["light", "dark"] = "light"
    on_workspace_change: Callable[[WorkspaceState | None], None] | None = None

    def __post_init__(self) -> None:
        self.trigger_button = pn.widgets.Button(
            name="",
            button_type="default",
            margin=0,
            width=22,
            height=22,
            css_classes=["lw-portal-workspace-trigger-button"],
        )
        self.trigger_led = pn.pane.HTML(margin=0, width=22, height=22)
        current = get_active_workspace_path()
        self.path_input = pn.widgets.TextInput(
            name="",
            value=str(current)
            if current is not None
            else str(_default_workspace_candidate()),
            placeholder="/path/to/larvaworld-workspace",
            margin=0,
            sizing_mode="stretch_width",
            css_classes=["lw-portal-workspace-input"],
        )
        self.browse_button = pn.widgets.Button(
            name="Browse",
            button_type="primary",
            margin=0,
            sizing_mode="stretch_width",
            css_classes=["lw-portal-workspace-browse-btn"],
        )
        self.init_button = pn.widgets.Button(
            name="Initialize",
            button_type="default",
            margin=0,
            sizing_mode="stretch_width",
        )
        self.clear_button = pn.widgets.Button(
            name="Clear",
            button_type="default",
            margin=0,
            sizing_mode="stretch_width",
        )
        self.current_pane = pn.pane.HTML(margin=0, sizing_mode="stretch_width")
        self.status_pane = pn.pane.HTML(
            margin=(8, 0, 0, 0),
            sizing_mode="stretch_width",
        )
        self.chip_pane = pn.pane.HTML(margin=0)
        self.trigger_view = pn.Column(
            self.trigger_led,
            self.trigger_button,
            margin=0,
            width=22,
            height=22,
            css_classes=["lw-portal-workspace-trigger-shell"],
        )

        self.browse_button.on_click(self._on_browse)
        self.init_button.on_click(self._on_initialize)
        self.clear_button.on_click(self._on_clear)

        self._refresh()

    def _emit(self, workspace: WorkspaceState | None) -> None:
        if self.on_workspace_change is not None:
            self.on_workspace_change(workspace)

    def _refresh(
        self,
        message: str | None = None,
        tone: str = "neutral",
        *,
        preserve_input: bool = False,
    ) -> None:
        workspace = get_active_workspace()
        self.chip_pane.object = _workspace_chip_html(workspace)
        self.trigger_led.object = _workspace_led_html(workspace)

        if workspace is None:
            self.trigger_button.css_classes = ["lw-portal-workspace-trigger-button"]
            current = get_active_workspace_path()
            if (
                not preserve_input
                and current is not None
                and not self.path_input.value.strip()
            ):
                self.path_input.value = str(current)
            self.current_pane.object = _status_html(
                text="No active workspace is configured.",
                tone="warning",
                detail="Select an initialized workspace or initialize a new one.",
                theme=self.theme,
            )
        else:
            self.trigger_button.css_classes = ["lw-portal-workspace-trigger-button"]
            if not preserve_input:
                self.path_input.value = str(workspace.root)
            self.current_pane.object = _status_html(
                text=f'Active workspace: "{workspace.name}"',
                tone="success",
                detail=str(workspace.root),
                theme=self.theme,
            )

        self.status_pane.object = ""
        if message is not None:
            self.status_pane.object = _status_html(
                text=message,
                tone=tone,
                theme=self.theme,
            )

    def _candidate_path(self) -> Path | None:
        raw = self.path_input.value.strip()
        if not raw:
            self._refresh("Enter a workspace folder path first.", tone="warning")
            return None
        return Path(raw).expanduser()

    def _on_initialize(self, _: object) -> None:
        candidate = self._candidate_path()
        if candidate is None:
            return

        validation = validate_workspace(candidate)
        if validation.errors:
            self._refresh("; ".join(validation.errors), tone="danger")
            return

        workspace = initialize_workspace(candidate)
        set_active_workspace_path(workspace.root)
        self._refresh("Workspace initialized and activated.", tone="success")
        self._emit(workspace)

    def _on_browse(self, _: object) -> None:
        current = self.path_input.value.strip()
        initial_dir = (
            Path(current).expanduser() if current else _default_workspace_candidate()
        )
        selected, error = pick_directory(
            initial_dir=initial_dir,
            fallback_dir=_default_workspace_candidate(),
            title="Select Larvaworld workspace folder",
        )
        if selected is not None:
            self.path_input.value = str(selected)
            validation = validate_workspace(selected)
            if validation.errors:
                self._refresh(
                    "; ".join(validation.errors),
                    tone="danger",
                    preserve_input=True,
                )
                return
            if validation.initialized:
                set_active_workspace_path(validation.path)
                workspace = get_active_workspace()
                self._refresh(
                    "Active workspace updated.",
                    tone="success",
                    preserve_input=True,
                )
                self._emit(workspace)
                return
            self.status_pane.object = _status_html(
                text="Selected workspace folder.",
                tone="warning",
                detail="Folder is not initialized yet. Use Initialize to create the workspace layout.",
                theme=self.theme,
            )
            return
        if error is not None:
            self._refresh(error, tone="warning")

    def _on_clear(self, _: object) -> None:
        clear_active_workspace_path()
        self.path_input.value = str(_default_workspace_candidate())
        self._refresh("Active workspace cleared.", tone="warning", preserve_input=True)
        self._emit(None)

    def build_controls(self) -> pn.viewable.Viewable:
        classes = ["lw-portal-workspace-controls"]
        if self.theme == "dark":
            classes.append("lw-portal-workspace-controls--dark")
        title_classes = ["lw-portal-settings-title"]
        field_label_classes = ["lw-portal-field-label"]
        title_style = ""
        field_label_style = ""
        if self.theme == "dark":
            title_classes.append("lw-portal-settings-title--dark")
            field_label_classes.append("lw-portal-field-label--dark")
            title_style = ' style="font-size:13px;font-weight:650;margin:0 0 6px 0;color:rgba(241,245,249,0.96);"'
            field_label_style = (
                ' style="font-size:12px;font-weight:600;color:rgba(241,245,249,0.92);"'
            )
        title_class_attr = " ".join(title_classes)
        field_label_class_attr = " ".join(field_label_classes)
        return pn.Column(
            pn.pane.HTML(
                f'<div class="{title_class_attr}"{title_style}>Workspace</div>',
                margin=0,
            ),
            self.current_pane,
            pn.pane.HTML(
                (
                    f'<div class="{field_label_class_attr}"{field_label_style}>'
                    "Workspace folder"
                    "</div>"
                ),
                margin=(8, 0, 4, 0),
            ),
            pn.Row(
                self.path_input,
                sizing_mode="stretch_width",
                margin=0,
                css_classes=["lw-portal-workspace-path-row"],
            ),
            pn.Row(
                self.browse_button,
                self.init_button,
                self.clear_button,
                sizing_mode="stretch_width",
                margin=(6, 0, 0, 0),
                css_classes=["lw-portal-workspace-actions"],
            ),
            self.status_pane,
            sizing_mode="stretch_width",
            margin=0,
            css_classes=classes,
        )
