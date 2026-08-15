"""Tests for the About portal popup content and UI component."""

from __future__ import annotations

from larvaworld.portal.about import build_about_content
from larvaworld.portal.panel_components import build_template_header


def _find_by_css_class(root: object, css_class: str) -> object | None:
    """Depth-first search for the first descendant carrying `css_class`."""
    if css_class in (getattr(root, "css_classes", None) or []):
        return root
    for child in getattr(root, "objects", []):
        found = _find_by_css_class(child, css_class)
        if found is not None:
            return found
    return None


def _all_markdown_text(root: object) -> str:
    """Concatenate the `.object` text of every Markdown pane under `root`."""
    parts: list[str] = []
    if hasattr(root, "object") and isinstance(getattr(root, "object"), str):
        parts.append(root.object)
    for child in getattr(root, "objects", []):
        parts.append(_all_markdown_text(child))
    return "\n".join(parts)


def test_about_content_contains_version() -> None:
    """The about popup includes the version number."""
    content = _all_markdown_text(build_about_content("9.9.9"))
    assert "9.9.9" in content
    assert "Larvaworld" in content


def test_about_content_contains_license() -> None:
    """The about popup includes license info."""
    content = _all_markdown_text(build_about_content("1.0.0"))
    assert "MIT" in content


def test_about_content_contains_citation() -> None:
    """The about popup includes the citation DOI."""
    content = _all_markdown_text(build_about_content("1.0.0"))
    assert "10.1101/2025.06.15.659765" in content
    assert "bioRxiv" in content


def test_about_content_contains_credits() -> None:
    """The about popup credits the developers."""
    content = _all_markdown_text(build_about_content("1.0.0"))
    assert "Panagiotis Sakagiannis" in content
    assert "Computational Neuroscience" in content
    assert "University of Cologne" in content


def test_template_header_includes_about_trigger() -> None:
    """The template header contains the about trigger: a visible icon pane
    plus an invisible Button stacked on top, like the workspace trigger."""
    header = build_template_header()
    about_trigger_view = _find_by_css_class(header, "lw-about-trigger-shell")
    assert about_trigger_view is not None, "About trigger shell not found in header"
    about_led, about_button = about_trigger_view.objects
    assert about_led.object.startswith("<img")
    assert about_button.css_classes == ["lw-about-trigger-button"]


def test_about_panel_starts_hidden() -> None:
    """The about dropdown panel starts hidden, like the settings dropdown."""
    header = build_template_header()
    about_panel = _find_by_css_class(header, "lw-about-dropdown-panel")
    assert about_panel is not None
    assert about_panel.visible is False


def test_about_button_toggles_panel() -> None:
    """Clicking the about trigger toggles the panel visibility."""
    header = build_template_header()
    about_trigger_view = _find_by_css_class(header, "lw-about-trigger-shell")
    about_panel = _find_by_css_class(header, "lw-about-dropdown-panel")
    _, about_button = about_trigger_view.objects

    assert about_panel.visible is False

    about_button.clicks += 1
    assert about_panel.visible is True

    about_button.clicks += 1
    assert about_panel.visible is False
