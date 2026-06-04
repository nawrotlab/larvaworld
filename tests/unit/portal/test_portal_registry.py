from __future__ import annotations

from larvaworld.portal.landing_registry import ITEMS
from larvaworld.portal.registry_logic import validate_registry
from larvaworld.portal.serve import APP_ID_TO_FACTORY_PATH, SERVED_APP_IDS


def test_validate_registry_strict_passes() -> None:
    validate_registry(strict=True)


def test_ready_panel_apps_are_served() -> None:
    ready_panel_app_ids = {
        item.panel_app_id
        for item in ITEMS.values()
        if item.kind == "panel_app" and item.status == "ready"
    }
    assert None not in ready_panel_app_ids
    assert {x for x in ready_panel_app_ids if x is not None} <= SERVED_APP_IDS


def test_open_dataset_is_a_ready_panel_app() -> None:
    item = ITEMS["wf.open_dataset"]

    assert item.kind == "panel_app"
    assert item.status == "ready"
    assert item.panel_app_id == "wf.open_dataset"
    assert item.panel_app_id in SERVED_APP_IDS


def test_dataset_manager_is_a_ready_panel_app() -> None:
    item = ITEMS["wf.dataset_manager"]

    assert item.kind == "panel_app"
    assert item.status == "ready"
    assert item.panel_app_id == "wf.dataset_manager"
    assert item.panel_app_id in SERVED_APP_IDS


def test_track_viewer_is_a_ready_panel_app() -> None:
    item = ITEMS["track_viewer"]

    assert item.kind == "panel_app"
    assert item.status == "ready"
    assert item.panel_app_id == "track_viewer"
    assert item.panel_app_id in SERVED_APP_IDS


def test_larva_models_is_a_ready_panel_app() -> None:
    item = ITEMS["larva_models"]

    assert item.kind == "panel_app"
    assert item.status == "ready"
    assert item.panel_app_id == "larva_models"
    assert item.panel_app_id in SERVED_APP_IDS


def test_larva_models_route_points_to_portal_model_inspector() -> None:
    assert APP_ID_TO_FACTORY_PATH["larva_models"] == (
        "larvaworld.portal.models_architecture.model_inspector_app:model_inspector_app"
    )


def test_locomotory_modules_route_points_to_portal_module_inspector() -> None:
    assert APP_ID_TO_FACTORY_PATH["locomotory_modules"] == (
        "larvaworld.portal.models_architecture.module_inspector_app:module_inspector_app"
    )
