from __future__ import annotations

import panel as pn
import pytest

from panel.links import Callback

from larvaworld.portal.buttons import (
    add_button,
    build_export_file_button,
    build_load_file_button,
    cancel_button,
    confirm_button,
    delete_button,
    draw_button,
    export_button,
    import_button,
    load_button,
    pause_button,
    refresh_button,
    remove_button,
    reset_button,
    run_button,
    save_button,
)

_PLAIN_FACTORIES = [
    (save_button, "Save", "success"),
    (load_button, "Load", "warning"),
    (delete_button, "Delete", "danger"),
    (remove_button, "Remove", "warning"),
    (reset_button, "Reset", "danger"),
    (run_button, "Run", "success"),
    (draw_button, "Draw", "primary"),
    (pause_button, "Pause", "primary"),
    (confirm_button, "Confirm", "danger"),
    (cancel_button, "Cancel", "default"),
    (refresh_button, "Refresh", "default"),
    (add_button, "Add", "success"),
]


@pytest.mark.parametrize("factory,default_name,default_type", _PLAIN_FACTORIES)
def test_default_name_and_button_type(factory, default_name, default_type):
    button = factory()
    assert isinstance(button, pn.widgets.Button)
    assert button.name == default_name
    assert button.button_type == default_type


@pytest.mark.parametrize("factory,default_name,default_type", _PLAIN_FACTORIES)
def test_button_type_override(factory, default_name, default_type):
    other_type = "light" if default_type != "light" else "warning"
    button = factory(button_type=other_type)
    assert button.button_type == other_type


@pytest.mark.parametrize("factory,default_name,default_type", _PLAIN_FACTORIES)
def test_custom_name(factory, default_name, default_type):
    button = factory("Custom label")
    assert button.name == "Custom label"


@pytest.mark.parametrize("factory,default_name,default_type", _PLAIN_FACTORIES)
def test_on_click_kwarg_wires_callback(factory, default_name, default_type):
    calls = []
    button = factory(on_click=lambda event: calls.append(event))
    button.clicks += 1
    assert len(calls) == 1


@pytest.mark.parametrize("factory,default_name,default_type", _PLAIN_FACTORIES)
def test_default_sizing_fills_width(factory, default_name, default_type):
    button = factory()
    assert button.sizing_mode == "stretch_width"


@pytest.mark.parametrize("factory,default_name,default_type", _PLAIN_FACTORIES)
def test_explicit_width_is_not_overridden(factory, default_name, default_type):
    button = factory(width=130)
    assert button.width == 130
    assert button.sizing_mode != "stretch_width"


@pytest.mark.parametrize("factory,default_name,default_type", _PLAIN_FACTORIES)
def test_explicit_sizing_mode_is_not_overridden(factory, default_name, default_type):
    button = factory(sizing_mode="fixed")
    assert button.sizing_mode == "fixed"


def test_import_button_wraps_build_load_file_button():
    button, file_input = import_button("Import", accept=".json,application/json")
    assert isinstance(button, pn.widgets.Button)
    assert isinstance(file_input, pn.widgets.FileInput)
    assert button.name == "Import"
    assert button.button_type == "light"
    assert file_input.accept == ".json,application/json"


def test_import_button_override_button_type():
    button, _file_input = import_button(
        "Load from file", accept=".pkl,.json", button_type="warning"
    )
    assert button.button_type == "warning"


def test_import_button_default_sizing_fills_width():
    button, _file_input = import_button("Import", accept=".json")
    assert button.sizing_mode == "stretch_width"


def test_import_button_explicit_width_not_overridden():
    button, _file_input = import_button("Import", accept=".json", width=140)
    assert button.width == 140
    assert button.sizing_mode != "stretch_width"


def test_import_button_matches_build_load_file_button_defaults():
    a_button, a_input = import_button("Import", accept=".json")
    b_button, b_input = build_load_file_button(
        "Import", accept=".json", button_type="light"
    )
    assert a_button.button_type == b_button.button_type
    assert a_input.accept == b_input.accept


def test_export_button_returns_button_and_hidden_file_download():
    calls = []
    button, file_download = export_button(
        "Export", callback=lambda: calls.append(1), filename="out.json"
    )
    assert isinstance(button, pn.widgets.Button)
    assert isinstance(file_download, pn.widgets.FileDownload)
    assert button.name == "Export"
    assert button.button_type == "light"
    assert file_download.filename == "out.json"
    # Hidden off-screen, same mechanism as the load-side FileInput.
    assert file_download.styles.get("position") == "absolute"
    assert file_download.styles.get("opacity") == "0"


def test_export_button_default_sizing_fills_width():
    button, _file_download = export_button(
        "Export", callback=lambda: None, filename="out.json"
    )
    assert button.sizing_mode == "stretch_width"


def test_export_button_override_button_type():
    button, file_download = export_button(
        "Export", callback=lambda: None, filename="out.json", button_type="primary"
    )
    assert button.button_type == "primary"
    assert file_download.button_type == "primary"


def test_build_export_file_button_js_on_click_increments_proxy_clicks():
    button, file_download = build_export_file_button(
        "Export", callback=lambda: None, filename="out.json"
    )
    callbacks = Callback.registry[button]
    assert any(
        "download_proxy.clicks" in code
        for callback in callbacks
        for code in callback.code.values()
    )
    assert any(
        callback.args.get("download_proxy") is file_download for callback in callbacks
    )
