"""Builder for the "About" popup shown in the portal header."""

from __future__ import annotations

from pathlib import Path

import panel as pn

_ABOUT_MD_PATH = Path(__file__).parent / "about.md"


def build_about_content(version: str) -> pn.viewable.Viewable:
    """Build the About popup body: a version header plus the static about.md text."""
    about_text = _ABOUT_MD_PATH.read_text(encoding="utf-8")
    return pn.Column(
        pn.pane.Markdown(f"### Larvaworld v{version}", margin=(0, 0, 12, 0)),
        pn.pane.Markdown(about_text, margin=0),
        margin=0,
    )
