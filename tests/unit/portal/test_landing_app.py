"""Tests for the portal landing page."""

from __future__ import annotations

from pathlib import Path

import panel as pn
import pytest

from larvaworld.portal.landing_app import landing_app
from larvaworld.portal.landing_registry import ITEMS, LANES, QUICK_START_MODES
from larvaworld.portal.panel_components import PORTAL_RAW_CSS, _header_links_html
from larvaworld.portal.registry_logic import compute_badges
from larvaworld.portal.workspace import (
    clear_active_workspace_path,
    initialize_workspace,
    set_active_workspace_path,
)


def _walk(obj, seen=None):
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj
    for child in getattr(obj, "objects", []) or []:
        yield from _walk(child, seen)


def _rendered_text(app) -> str:
    return " ".join(str(getattr(n, "object", "")) for n in _walk(app.main))


@pytest.fixture(autouse=True)
def workspace_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LARVAWORLD_PORTAL_CONFIG_DIR", str(tmp_path / "config"))
    clear_active_workspace_path()


def test_landing_app_without_workspace_redirects() -> None:
    app = landing_app()

    assert isinstance(app, pn.Column)
    assert "window.location.replace" in app.objects[0].object


def test_landing_app_with_active_workspace_renders_template(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)

    app = landing_app()

    assert isinstance(app, pn.template.MaterialTemplate)
    assert app.main.objects


def test_landing_app_banner_renders_below_quick_start(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)

    app = landing_app()
    root = next(
        child
        for child in app.main
        if "lw-portal-root" in getattr(child, "css_classes", [])
    )
    css_classes = [tuple(getattr(child, "css_classes", ())) for child in root.objects]
    quick_start_index = next(
        i
        for i, classes in enumerate(css_classes)
        if any(c.startswith("lw-portal-quick-start") for c in classes)
    )
    banner_index = next(
        i for i, classes in enumerate(css_classes) if "lw-portal-banner" in classes
    )
    assert quick_start_index < banner_index


def test_quick_start_tabs_stack_below_template_header() -> None:
    quick_start_css = PORTAL_RAW_CSS.split(".lw-portal-quick-start-tabs", 1)[1]

    assert "z-index: 1;" in quick_start_css.split("}", 1)[0]


def test_landing_header_places_tutorials_between_docs_and_github() -> None:
    links = _header_links_html()

    docs_index = links.index('title="Read the Docs"')
    tutorials_index = links.index('title="Tutorial course"')
    github_index = links.index('title="GitHub"')

    assert docs_index < tutorials_index < github_index
    assert (
        "https://larvaworld.readthedocs.io/en/latest/tutorials/" "index.html#tutorials"
    ) in links
    assert 'alt="Education icon"' in links


def test_planned_items_are_hidden_by_default(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)

    text = _rendered_text(landing_app())

    planned = [
        ITEMS[item_id]
        for lane in LANES
        for item_id in lane.item_ids
        if ITEMS[item_id].status == "planned"
    ]
    assert planned, "expected the registry to still contain planned placeholders"
    for item in planned:
        assert item.title not in text
    assert "Under construction" not in text


def test_ready_items_are_shown_by_default(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)

    text = _rendered_text(landing_app())

    ready = [
        ITEMS[item_id]
        for lane in LANES
        for item_id in lane.item_ids
        if ITEMS[item_id].status == "ready"
    ]
    for item in ready:
        assert item.title in text


def test_show_planned_toggle_reveals_placeholders(tmp_path: Path) -> None:
    workspace = initialize_workspace(tmp_path / "workspace")
    set_active_workspace_path(workspace.root)

    app = landing_app()
    toggle = next(
        node for node in _walk(app.main) if isinstance(node, pn.widgets.Checkbox)
    )
    assert toggle.value is False

    toggle.value = True
    text = _rendered_text(app)

    planned = [
        ITEMS[item_id]
        for lane in LANES
        for item_id in lane.item_ids
        if ITEMS[item_id].status == "planned"
    ]
    for item in planned:
        assert item.title in text


def test_default_user_quick_start_entries_are_all_ready() -> None:
    user_mode = next(m for m in QUICK_START_MODES if m.mode_id == "user")

    assert len(user_mode.item_ids) == 3
    for item_id in user_mode.item_ids:
        assert (
            ITEMS[item_id].status == "ready"
        ), f"quick-start entry '{item_id}' is not usable on a fresh install"


def test_core_badge_is_not_rendered() -> None:
    # "Core" applied to nearly every item, so it carried no information.
    for item in ITEMS.values():
        assert "Core" not in compute_badges(item)


def test_student_facing_apps_are_not_badged_developer() -> None:
    for item_id in ("wf.explore", "wf.run_experiment", "wf.environment_builder"):
        assert "Developer" not in compute_badges(ITEMS[item_id])
