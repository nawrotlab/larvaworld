from __future__ import annotations

import json
from html import escape

import panel as pn

from larvaworld.portal.notebook_workspace import launch_notebook_for_item


def _card_html(*, title: str, body: str, footer: str = "") -> str:
    footer_html = f'<p style="margin:12px 0 0 0;">{footer}</p>' if footer else ""
    return (
        '<div style="max-width:720px;margin:36px auto;padding:16px 18px;'
        "border:1px solid rgba(0,0,0,0.15);border-radius:12px;"
        'font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;">'
        f'<h3 style="margin:0 0 10px 0;">{escape(title)}</h3>'
        f'<div style="margin:0;font-size:14px;line-height:1.5;">{body}</div>'
        f"{footer_html}"
        "</div>"
    )


def _query_param(name: str) -> str | None:
    values = pn.state.session_args.get(name, [])
    if not values:
        return None
    value = values[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _error_view(message: str) -> pn.viewable.Viewable:
    html = _error_html(message)
    return pn.Column(pn.pane.HTML(html, margin=0), sizing_mode="stretch_width")


def _error_html(message: str) -> str:
    return _card_html(
        title="Notebook launch unavailable",
        body=f'<p style="margin:0;">{escape(message)}</p>',
        footer='<a href="/landing">Back to landing</a>',
    )


def _initializing_html() -> str:
    return _card_html(
        title="Initializing notebooks",
        body=(
            '<p style="margin:0 0 10px 0;">'
            "Preparing notebook files in the active workspace and checking the notebook runtime."
            "</p>"
            '<p style="margin:0;color:rgba(15,23,42,0.72);">'
            "The first notebook launch can take a little longer. Please wait."
            "</p>"
        ),
        footer='<a href="/landing">Back to landing</a>',
    )


def _redirect_html(notebook_url: str) -> str:
    js_url = json.dumps(notebook_url)
    return _card_html(
        title="Opening notebook",
        body=(
            f"<script>window.location.replace({js_url});</script>"
            '<p style="margin:0 0 10px 0;">Opening notebook...</p>'
            f'<p style="margin:0;">If you are not redirected, <a href="{escape(notebook_url)}">open it here</a>.</p>'
        ),
    )


def notebook_launch_app() -> pn.viewable.Viewable:
    pn.extension()

    item_id = (_query_param("id") or "").strip()
    if not item_id:
        return _error_view("Missing notebook id.")

    status_pane = pn.pane.HTML(_initializing_html(), margin=30)

    def _launch_after_render() -> None:
        notebook_url, error = launch_notebook_for_item(item_id)
        if not notebook_url:
            status_pane.object = _error_html(
                error or "Notebook runtime is unavailable."
            )
            return
        status_pane.object = _redirect_html(notebook_url)

    pn.state.onload(_launch_after_render)
    return pn.Column(status_pane, sizing_mode="stretch_width")
