"""Tests for the Parameter Database Panel UI layer.

Integration-style behavior (per-row actions, popups, Add flow) is
exercised through `build_standalone_page()` — the same entry point used by
`python -m larvaworld.portal.parameter_database` — rather than by calling
internal builders (`build_parameter_db_content`, `_build_detail_popup`,
`_build_add_parameter_popup`) directly. `build_param_detail_popup` is a
public, independently reusable component and is still tested directly.
"""

from __future__ import annotations

import panel as pn

from larvaworld.lib import reg
from larvaworld.portal.parameter_database.parameter_db_app import (
    _DraggableResizablePopup,
    build_param_detail_popup,
    build_standalone_page,
)
from larvaworld.portal.parameter_database.parameter_funcs import (
    get_param_instance,
    register_new_param,
)


def _standalone() -> (
    tuple[pn.viewable.Viewable, pn.widgets.Tabulator, _DraggableResizablePopup]
):
    """Build a fresh standalone page and locate its table and detail popup."""
    page = build_standalone_page()
    table = page.select(pn.widgets.Tabulator)[0]
    detail_popup = page.select(_DraggableResizablePopup)[0]
    return page, table, detail_popup


def _click(
    page: pn.viewable.Viewable, name: str, button_type: str | None = None
) -> None:
    candidates = [b for b in page.select(pn.widgets.Button) if b.name == name]
    if button_type is not None:
        candidates = [b for b in candidates if b.button_type == button_type]
    candidates[0].clicks += 1


class _FakeCellClickEvent:
    def __init__(self, column: str, row: int) -> None:
        self.column = column
        self.row = row
        self.value = None


def _click_row_action(table: pn.widgets.Tabulator, column: str, row: int) -> None:
    for callback in table._on_click_callbacks[column]:
        callback(_FakeCellClickEvent(column, row))


def _row_for_key(table: pn.widgets.Tabulator, k: str) -> int:
    row = table.value.index[table.value["Key"] == k][0]
    return table.value.index.get_loc(row)


def _register_disposable_clone(k_suffix: str) -> str:
    config = dict(get_param_instance("t").to_config())
    config.update(k=f"x_{k_suffix}_k", p=f"x_{k_suffix}_p", d=f"x_{k_suffix}_d")
    return register_new_param(config)


# --- build_param_detail_popup: a public, independently reusable component ---


def test_build_param_detail_popup_readonly_shows_correct_values() -> None:
    view = build_param_detail_popup("t", editable=False)
    widgets = {w.name: w.value for w in view.select(pn.widgets.Widget)}

    assert widgets.get("Display name") == reg.par.dict["t"].disp
    assert widgets.get("Symbol") == reg.par.dict["t"].sym
    assert widgets.get("Name") == reg.par.dict["t"].p
    assert widgets.get("Key") == reg.par.dict["t"].k
    assert all(getattr(w, "disabled", True) for w in view.select(pn.widgets.Widget))


def test_build_param_detail_popup_editable_widgets_not_disabled() -> None:
    # "Unit" and "Computing function" stay read-only even in editable mode
    # (see _ALWAYS_READONLY_FIELDS): editing a pint Unit via free text is
    # scientifically risky, and "Computing function" is a display-only
    # module.qualname string, not a meaningfully editable field.
    view = build_param_detail_popup("t", editable=True)
    widgets = {w.name: w for w in view.select(pn.widgets.Widget)}
    assert widgets
    always_readonly = {"Unit", "Computing function"}
    for name, widget in widgets.items():
        expected_disabled = name in always_readonly
        assert getattr(widget, "disabled", False) is expected_disabled, name


def test_build_param_detail_popup_editable_wires_save_button() -> None:
    saved = {}

    def _on_save(fields):
        saved.update(fields)

    view = build_param_detail_popup("t", editable=True, on_save=_on_save)

    buttons = list(view.select(pn.widgets.Button))
    assert len(buttons) == 1
    save_button = buttons[0]
    assert save_button.name == "Save"

    save_button.clicks += 1
    assert saved.get("p") == reg.par.dict["t"].p


def test_build_param_detail_popup_shows_all_fields() -> None:
    # Read-only view: a fixed set of labeled fields grouped into four
    # panels (see _DETAIL_FIELD_GROUPS), two panels per column.
    from larvaworld.portal.parameter_database.parameter_popups import (
        _DETAIL_FIELD_GROUPS,
    )

    view = build_param_detail_popup("t", editable=False)
    widgets = list(view.select(pn.widgets.Widget))
    expected_labels = {
        label for _, fields in _DETAIL_FIELD_GROUPS for _, label in fields
    }

    assert len(widgets) == len(expected_labels)
    assert {w.name for w in widgets} == expected_labels

    cards = list(view.select(pn.Card))
    assert {card.title for card in cards} == {
        title for title, _ in _DETAIL_FIELD_GROUPS
    }

    rows = view.select(pn.Row)
    assert rows
    assert len(rows[0]) == 2  # two columns


def test_build_param_detail_popup_has_no_download_button() -> None:
    # Exporting is now a per-row table action, not part of the detail popup.
    view = build_param_detail_popup("t", editable=False)
    assert not list(view.select(pn.widgets.FileDownload))


def test_required_param_keys_is_a_multi_select_dropdown() -> None:
    # Previously a free-text comma-separated field -- now a proper
    # multi-select over the registry's own param keys, so a user picks
    # existing keys instead of retyping them by hand.
    view = build_param_detail_popup("t", editable=True)
    widgets = [
        w
        for w in view.select(pn.widgets.MultiChoice)
        if w.name == "Required param keys"
    ]
    assert len(widgets) == 1
    widget = widgets[0]
    assert isinstance(widget.value, list)
    assert set(widget.value) == set(reg.par.dict["t"].required_ks or [])
    assert reg.par.dict["t"].k in widget.options


# --- standalone page: essential shape ---


def test_build_standalone_page_has_table_and_detail_popup() -> None:
    page, table, detail_popup = _standalone()

    assert len(table.value) == len(reg.par.dict)
    assert detail_popup.visible is False


def test_standalone_table_has_no_fixed_height() -> None:
    # wide=True (standalone): the table grows to its full row count instead
    # of scrolling internally.
    _, table, _ = _standalone()
    assert table.height is None


def test_portal_embedded_table_has_fixed_height() -> None:
    # wide=False (portal popup): limited popup height, so the table keeps
    # its own scrollbar.
    from larvaworld.portal.parameter_database.parameter_db_app import (
        _build_detail_popup,
        build_parameter_db_content,
    )

    content = build_parameter_db_content(_build_detail_popup())
    table = content.select(pn.widgets.Tabulator)[0]
    assert table.height == 500


def test_sidebar_has_only_add_parameter_button() -> None:
    page, _, _ = _standalone()
    buttons = {b.name: b.button_type for b in page.select(pn.widgets.Button)}

    assert buttons == {"Add parameter": "success"}
    # No standalone Inspect/Remove/Clone/Export buttons left visible. A
    # single FileDownload still exists (it drives the per-row Export
    # icon's actual browser download), but it is not shown in the panel.
    downloads = page.select(pn.widgets.FileDownload)
    assert len(downloads) == 1
    assert downloads[0].visible is False


def test_columns_picker_is_not_collapsible() -> None:
    page, _, _ = _standalone()
    assert not list(page.select(pn.Card))
    checkbox_group = page.select(pn.widgets.CheckBoxGroup)[0]
    assert "Category" in checkbox_group.options
    # Action columns aren't part of the toggle-able set.
    for action in ("Inspect", "Remove", "Clone", "Export"):
        assert action not in checkbox_group.options


# --- per-row table actions ---


def test_table_has_grouped_action_columns_with_fixed_widths() -> None:
    _, table, _ = _standalone()

    assert table.groups == {"Actions": ["Inspect", "Remove", "Clone", "Export"]}
    for name in ("Inspect", "Remove", "Clone", "Export"):
        assert table.widths[name] == "100px"
        assert table.formatters[name] == "html"
    assert set(table._on_click_callbacks.keys()) == {
        "Inspect",
        "Remove",
        "Clone",
        "Export",
    }


def test_action_column_cells_have_tooltip_titles() -> None:
    _, table, _ = _standalone()
    for name in ("Inspect", "Remove", "Clone", "Export"):
        assert "title=" in table.value[name].iloc[0]


def test_inspect_icon_click_opens_cyan_popup_titled_with_display_name() -> None:
    _, table, detail_popup = _standalone()

    k = str(table.value.iloc[0]["Key"])
    _click_row_action(table, "Inspect", 0)

    assert detail_popup.visible is True
    assert detail_popup.title == f"Parameter: {reg.par.dict[k].disp}"
    assert detail_popup.theme_class == "lw-parameter-db-panel-surface--cyan"


def test_remove_icon_click_opens_confirmation_popup() -> None:
    k = _register_disposable_clone("remove_icon_confirm_test")
    _, table, detail_popup = _standalone()
    row_index = _row_for_key(table, k)

    _click_row_action(table, "Remove", row_index)

    assert detail_popup.visible is True
    assert detail_popup.theme_class == "lw-parameter-db-panel-surface--red"
    delete_buttons = [
        b for b in detail_popup.body.select(pn.widgets.Button) if b.name == "Delete"
    ]
    assert len(delete_buttons) == 1


def test_remove_icon_confirm_deletes_param_and_closes_popup() -> None:
    k = _register_disposable_clone("remove_icon_delete_test")
    _, table, detail_popup = _standalone()
    row_index = _row_for_key(table, k)

    _click_row_action(table, "Remove", row_index)
    _click(detail_popup.body, "Delete")

    assert k not in reg.par.dict
    assert detail_popup.visible is False


def test_clone_icon_click_opens_green_add_popup_prepopulated() -> None:
    _, table, detail_popup = _standalone()
    k = str(table.value.iloc[0]["Key"])

    _click_row_action(table, "Clone", 0)

    assert detail_popup.visible is True
    assert detail_popup.theme_class == "lw-parameter-db-panel-surface--green"
    p_widget = [
        w for w in detail_popup.body.select(pn.widgets.TextInput) if w.name == "Name"
    ][0]
    assert p_widget.value == reg.par.dict[k].p
    # The header's "Clone by key" input is always present alongside the
    # pre-populated form, as an alternative way to load a different key.
    assert len(detail_popup.body.select(pn.widgets.AutocompleteInput)) == 1


def test_export_icon_click_triggers_hidden_download() -> None:
    page, table, _ = _standalone()
    k = str(table.value.iloc[0]["Key"])
    download = page.select(pn.widgets.FileDownload)[0]
    clicks_before = download._clicks

    _click_row_action(table, "Export", 0)

    assert download.filename == f"{k}_config.pkl"
    assert download._clicks == clicks_before + 1
    data = download.callback()
    assert len(data.getvalue()) > 0


# --- Add flow ---


def test_add_parameter_button_opens_green_popup_without_redundant_heading() -> None:
    page, _, detail_popup = _standalone()

    _click(page, "Add parameter", button_type="success")

    assert detail_popup.visible is True
    assert detail_popup.theme_class == "lw-parameter-db-panel-surface--green"
    markdowns = [m.object for m in detail_popup.body.select(pn.pane.Markdown)]
    assert not any("Add Parameter" in (text or "") for text in markdowns)


def test_add_form_buttons_include_clone_by_key() -> None:
    page, _, detail_popup = _standalone()

    _click(page, "Add parameter", button_type="success")

    # Nothing has been loaded yet, so build_param_detail_popup (and its
    # "Save" button) hasn't been built into the form yet -- only the
    # header row's "Clone by key" and "Load from file" triggers are
    # present at this point.
    buttons = {
        b.name: b.button_type for b in detail_popup.body.select(pn.widgets.Button)
    }
    assert buttons == {"Clone by key": "primary", "Load from file": "warning"}
    key_inputs = detail_popup.body.select(pn.widgets.AutocompleteInput)
    assert len(key_inputs) == 1
    assert reg.par.dict["t"].k in key_inputs[0].options


def test_add_form_clone_by_key_populates_fields() -> None:
    page, _, detail_popup = _standalone()
    _click(page, "Add parameter", button_type="success")

    key_input = detail_popup.body.select(pn.widgets.AutocompleteInput)[0]
    key_input.value = "t"
    clone_button = [
        b
        for b in detail_popup.body.select(pn.widgets.Button)
        if b.name == "Clone by key"
    ][0]
    clone_button.clicks += 1

    source_instance = get_param_instance("t")
    p_widget = [
        w for w in detail_popup.body.select(pn.widgets.TextInput) if w.name == "Name"
    ][0]
    assert p_widget.value == source_instance.p
    save_buttons = [
        b for b in detail_popup.body.select(pn.widgets.Button) if b.name == "Save"
    ]
    assert len(save_buttons) == 1


def test_add_form_clone_by_key_unknown_key_shows_error() -> None:
    page, _, detail_popup = _standalone()
    _click(page, "Add parameter", button_type="success")

    key_input = detail_popup.body.select(pn.widgets.AutocompleteInput)[0]
    key_input.value = "__not_a_real_key__"
    clone_button = [
        b
        for b in detail_popup.body.select(pn.widgets.Button)
        if b.name == "Clone by key"
    ][0]
    clone_button.clicks += 1

    status = [
        m
        for m in detail_popup.body.select(pn.pane.Markdown)
        if "Unknown parameter key" in (m.object or "")
    ]
    assert status


def test_add_form_load_from_file_reconstructs_instance() -> None:
    from larvaworld.portal.parameter_database.parameter_funcs import (
        save_param_config,
    )

    page, _, detail_popup = _standalone()
    _click(page, "Add parameter", button_type="success")

    # config_input is a hidden FileInput (see _build_add_parameter_popup):
    # "Load from file" opens a native file picker purely in JS and
    # forwards the choice here, so setting .value directly -- as a real
    # file pick would via the JS side -- is what actually triggers the
    # load (via config_input.param.watch), not a click on the button.
    config_input = detail_popup.body.select(pn.widgets.FileInput)[0]
    source_instance = get_param_instance("t")
    config_input.filename = "t_config.pkl"
    config_input.value = save_param_config(source_instance)

    p_widget = [
        w for w in detail_popup.body.select(pn.widgets.TextInput) if w.name == "Name"
    ][0]
    assert p_widget.value == source_instance.p
    save_buttons = [
        b for b in detail_popup.body.select(pn.widgets.Button) if b.name == "Save"
    ]
    assert len(save_buttons) == 1


def test_clone_then_add_registers_new_param_with_edited_doc() -> None:
    _, table, detail_popup = _standalone()
    _click_row_action(table, "Clone", 0)

    text_inputs = detail_popup.body.select(pn.widgets.TextInput)
    k_widget = [w for w in text_inputs if w.name == "Key"][0]
    p_widget = [w for w in text_inputs if w.name == "Name"][0]
    d_widget = [w for w in text_inputs if w.name == "Name in dataset"][0]
    k_widget.value = "x_clone_add_test_k"
    p_widget.value = "x_clone_add_test_p"
    d_widget.value = "x_clone_add_test_d"
    doc_widget = [
        w
        for w in detail_popup.body.select(pn.widgets.TextAreaInput)
        if w.name == "Description"
    ][0]
    doc_widget.value = "custom doc text"

    rows_before = len(table.value)
    _click(detail_popup.body, "Save", button_type="primary")

    assert "x_clone_add_test_k" in reg.par.dict
    assert reg.par.kdict["x_clone_add_test_k"].description == "custom doc text"
    assert len(table.value) == rows_before + 1  # table refreshed
