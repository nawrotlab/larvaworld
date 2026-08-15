"""Tests for the Parameter Database popup trigger in the portal header."""

from __future__ import annotations

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


def test_template_header_includes_parameter_db_trigger() -> None:
    """The template header contains the parameter-db trigger: a visible icon
    pane plus an invisible Button stacked on top, like the about trigger."""
    header = build_template_header()
    trigger_view = _find_by_css_class(header, "lw-parameter-db-trigger-shell")
    assert trigger_view is not None, "Parameter DB trigger shell not found in header"
    led, button = trigger_view.objects
    assert led.object.startswith("<img")
    assert button.css_classes == ["lw-parameter-db-trigger-button"]


def test_parameter_db_panel_starts_hidden() -> None:
    """The parameter-db dropdown panel starts hidden."""
    header = build_template_header()
    panel = _find_by_css_class(header, "lw-parameter-db-dropdown-panel")
    assert panel is not None
    assert panel.visible is False


def test_parameter_db_button_toggles_panel() -> None:
    """Clicking the parameter-db trigger toggles the panel visibility."""
    header = build_template_header()
    trigger_view = _find_by_css_class(header, "lw-parameter-db-trigger-shell")
    panel = _find_by_css_class(header, "lw-parameter-db-dropdown-panel")
    _, button = trigger_view.objects

    assert panel.visible is False

    button.clicks += 1
    assert panel.visible is True

    button.clicks += 1
    assert panel.visible is False


def test_old_info_button_code_removed() -> None:
    """The inert workspace-header info-icon link and its loader are gone."""
    import larvaworld.portal.workspace_ui as workspace_ui

    assert not hasattr(workspace_ui, "_workspace_header_icons_html")
    assert not hasattr(workspace_ui, "_load_info_icon_data_uri")
