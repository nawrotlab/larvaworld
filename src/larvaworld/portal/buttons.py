"""Reusable Panel button helpers for the portal.

No heavy imports -- only `panel`, consistent with the "keep startup
lightweight" portal rule in CLAUDE.md.
"""

from __future__ import annotations

from typing import Any, Callable

import panel as pn

__all__: list[str] = [
    "build_load_file_button",
    "build_export_file_button",
    "save_button",
    "load_button",
    "delete_button",
    "remove_button",
    "reset_button",
    "run_button",
    "draw_button",
    "pause_button",
    "confirm_button",
    "cancel_button",
    "refresh_button",
    "add_button",
    "import_button",
    "export_button",
]

#: Inline styles that fully remove a widget from the visible layout while
#: keeping it mounted in the DOM (so JS can still `.setv()` its value) --
#: used to hide the native FileInput backing `build_load_file_button`,
#: since a native `<input type=file>`'s own button label ("Browse"/"Choose
#: File") is localized by the browser/OS UI language and can't be
#: restyled or overridden from the page.
_HIDDEN_FILE_INPUT_STYLES: dict[str, str] = {
    "position": "absolute",
    "left": "-10000px",
    "top": "auto",
    "width": "1px",
    "height": "1px",
    "opacity": "0",
    "overflow": "hidden",
    "pointer-events": "none",
}


def build_load_file_button(
    name: str = "Load file",
    *,
    accept: str = "",
    multiple: bool = False,
    button_type: str = "default",
    **button_kwargs: Any,
) -> tuple[pn.widgets.Button, pn.widgets.FileInput]:
    """
    Build a `(button, hidden_file_input)` pair for loading a file through a
    custom-labeled button instead of a visible `FileInput`.

    A native `<input type=file>`'s own trigger button is rendered by the
    browser using the browser/OS UI language -- Panel and CSS can't
    override that text. Instead, `button` (fully custom, e.g. "Load
    file") opens a throwaway `<input type=file>` created purely in JS
    (never inserted into the page, so its label is never shown), reads
    the chosen file client-side, and forwards its value/filename/mime
    type onto the returned `FileInput`, which is itself styled off-screen
    and also never shown.

    Wire the actual load logic with:
        file_input.param.watch(on_file_chosen, "value")
    `on_file_chosen` should read `file_input.value` / `.filename`; it
    fires as soon as a file is picked, not on a second click of `button`.
    """
    file_input = pn.widgets.FileInput(name="", accept=accept, multiple=multiple)
    file_input.styles = dict(_HIDDEN_FILE_INPUT_STYLES)

    button = pn.widgets.Button(name=name, button_type=button_type, **button_kwargs)
    button.js_on_click(
        args={"load_input": file_input},
        code=f"""
            const picker = document.createElement('input');
            picker.type = 'file';
            picker.accept = {accept!r};
            picker.multiple = {"true" if multiple else "false"};
            picker.onchange = async () => {{
                const file = picker.files && picker.files[0];
                if (!file) {{
                    return;
                }}
                const dataUrl = await new Promise((resolve, reject) => {{
                    const reader = new FileReader();
                    reader.onload = () => resolve(reader.result || '');
                    reader.onerror = () => reject(reader.error || new Error(`unable to read '${{file.name}}'`));
                    reader.readAsDataURL(file);
                }});
                const [, mime_type = '', , value = ''] = String(dataUrl).split(/[:;,]/, 4);
                load_input.setv({{value, filename: file.name, mime_type}});
            }};
            picker.click();
        """,
    )
    return button, file_input


#: Canonical `button_type` per semantic action -- save/run/add=green,
#: delete=red, load/remove=yellow, draw (secondary preview/comparison
#: actions distinct from the main "Run")/pause=blue, cancel/refresh=white,
#: import/export=light (Panel has no true pastel yellow/green -- `light`
#: is the closest built-in, pale/near-white type; a real tint would need
#: custom CSS). Divergent call sites keep their look by passing an
#: explicit `button_type=` override to the factory below rather than
#: being silently restyled.
_ACTION_BUTTON_TYPE: dict[str, str] = {
    "save": "success",
    "load": "warning",
    "delete": "danger",
    "remove": "warning",
    "reset": "danger",
    "run": "success",
    "draw": "primary",
    "pause": "primary",
    "confirm": "danger",
    "cancel": "default",
    "refresh": "default",
    "add": "success",
    "import": "light",
    "export": "light",
}


#: Default size for standardized buttons that don't specify their own --
#: fills the width of whatever Row/Column contains them, matching the most
#: common sizing already in use across the portal. Only applied when the
#: caller gives no `width`, `height`, or `sizing_mode` of their own, so it
#: never overrides an explicit fixed-width button.
_DEFAULT_BUTTON_SIZING: dict[str, str] = {"sizing_mode": "stretch_width"}


def _apply_default_sizing(kwargs: dict[str, Any]) -> None:
    if not ({"width", "height", "sizing_mode"} & kwargs.keys()):
        kwargs.update(_DEFAULT_BUTTON_SIZING)


def _action_button(action: str, name: str, **kwargs: Any) -> pn.widgets.Button:
    """Build a `pn.widgets.Button` defaulted to `action`'s canonical `button_type`."""
    kwargs.setdefault("button_type", _ACTION_BUTTON_TYPE[action])
    _apply_default_sizing(kwargs)
    return pn.widgets.Button(name=name, **kwargs)


def save_button(name: str = "Save", **kwargs: Any) -> pn.widgets.Button:
    """Standard "Save" button (default `button_type='success'`)."""
    return _action_button("save", name, **kwargs)


def load_button(name: str = "Load", **kwargs: Any) -> pn.widgets.Button:
    """Standard "Load" button (default `button_type='warning'`)."""
    return _action_button("load", name, **kwargs)


def delete_button(name: str = "Delete", **kwargs: Any) -> pn.widgets.Button:
    """Standard destructive "Delete" button (default `button_type='danger'`)."""
    return _action_button("delete", name, **kwargs)


def remove_button(name: str = "Remove", **kwargs: Any) -> pn.widgets.Button:
    """Standard non-destructive row-removal button (default `button_type='warning'`)."""
    return _action_button("remove", name, **kwargs)


def reset_button(name: str = "Reset", **kwargs: Any) -> pn.widgets.Button:
    """Standard "Reset" button (default `button_type='danger'`)."""
    return _action_button("reset", name, **kwargs)


def run_button(name: str = "Run", **kwargs: Any) -> pn.widgets.Button:
    """Standard "Run" button (default `button_type='success'`)."""
    return _action_button("run", name, **kwargs)


def draw_button(name: str = "Draw", **kwargs: Any) -> pn.widgets.Button:
    """Standard secondary preview/comparison-run button (default `button_type='primary'`), distinct from the main green `run_button`."""
    return _action_button("draw", name, **kwargs)


def pause_button(name: str = "Pause", **kwargs: Any) -> pn.widgets.Button:
    """Standard "Pause" button (default `button_type='primary'`)."""
    return _action_button("pause", name, **kwargs)


def confirm_button(name: str = "Confirm", **kwargs: Any) -> pn.widgets.Button:
    """Standard confirm-destructive-action button (default `button_type='danger'`)."""
    return _action_button("confirm", name, **kwargs)


def cancel_button(name: str = "Cancel", **kwargs: Any) -> pn.widgets.Button:
    """Standard "Cancel" button (default `button_type='default'`)."""
    return _action_button("cancel", name, **kwargs)


def refresh_button(name: str = "Refresh", **kwargs: Any) -> pn.widgets.Button:
    """Standard "Refresh" button (default `button_type='default'`)."""
    return _action_button("refresh", name, **kwargs)


def add_button(name: str = "Add", **kwargs: Any) -> pn.widgets.Button:
    """Standard "Add" button (default `button_type='primary'`)."""
    return _action_button("add", name, **kwargs)


def import_button(
    name: str = "Import",
    *,
    accept: str,
    multiple: bool = False,
    button_type: str | None = None,
    **button_kwargs: Any,
) -> tuple[pn.widgets.Button, pn.widgets.FileInput]:
    """
    Standard "Import" button (default `button_type='light'`), built on
    `build_load_file_button`. `accept` has no default since it's always
    caller-specific (e.g. `.json,application/json` for a config dict vs
    `.pkl,.json` for a stored `LarvaDataset`/param config).
    """
    _apply_default_sizing(button_kwargs)
    return build_load_file_button(
        name,
        accept=accept,
        multiple=multiple,
        button_type=button_type or _ACTION_BUTTON_TYPE["import"],
        **button_kwargs,
    )


def build_export_file_button(
    name: str = "Export",
    *,
    callback: Callable[[], Any],
    filename: str,
    button_type: str = "default",
    **button_kwargs: Any,
) -> tuple[pn.widgets.Button, pn.widgets.FileDownload]:
    """
    Build a `(button, hidden_file_download)` pair -- the export-side mirror
    of `build_load_file_button`. `button` carries the visible custom label;
    clicking it increments the hidden `FileDownload`'s `clicks` via JS,
    triggering the real browser download without ever showing Panel's
    native `FileDownload` label/styling.

    `callback` and `filename` are forwarded to the underlying
    `pn.widgets.FileDownload` exactly as with that widget directly.
    """
    file_download = pn.widgets.FileDownload(
        name="",
        label=name,
        button_type=button_type,
        callback=callback,
        filename=filename,
    )
    file_download.styles = dict(_HIDDEN_FILE_INPUT_STYLES)
    _apply_default_sizing(button_kwargs)
    button = pn.widgets.Button(name=name, button_type=button_type, **button_kwargs)
    button.js_on_click(
        args={"download_proxy": file_download},
        code="download_proxy.setv({clicks: download_proxy.clicks + 1});",
    )
    return button, file_download


def export_button(
    name: str = "Export",
    *,
    callback: Callable[[], Any],
    filename: str,
    button_type: str | None = None,
    **button_kwargs: Any,
) -> tuple[pn.widgets.Button, pn.widgets.FileDownload]:
    """Standard "Export" button (default `button_type='light'`), built on `build_export_file_button`."""
    return build_export_file_button(
        name,
        callback=callback,
        filename=filename,
        button_type=button_type or _ACTION_BUTTON_TYPE["export"],
        **button_kwargs,
    )
