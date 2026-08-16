"""Reusable Panel button helpers for the portal.

No heavy imports -- only `panel`, consistent with the "keep startup
lightweight" portal rule in CLAUDE.md.
"""

from __future__ import annotations

from typing import Any

import panel as pn

__all__: list[str] = ["build_load_file_button"]

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
