from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from larvaworld.gui_v2.apps.models_environments.env_builder_widget import (
    EnvBuilderWidget,
    build_environment_builder_widget,
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_environment_builder_widget_constructs_headlessly() -> None:
    _app()
    widget = build_environment_builder_widget(None)

    assert isinstance(widget, EnvBuilderWidget)
    assert widget.controller is not None
    assert widget.status_bar is not None
    assert widget.left_panel.save_workspace_button.isEnabled() is False


def test_table_selection_updates_inspector() -> None:
    _app()
    widget = EnvBuilderWidget()
    row = widget.controller.add_source_unit(x=0.01, y=0.02)

    widget.controller.select_object(row.object_id)

    assert widget.inspector_panel.object_id_edit.text() == row.object_id
    selected_items = widget.inspector_panel.table.selectedItems()
    assert selected_items
    assert selected_items[0].text() == row.object_id


def test_apply_and_delete_route_through_controller() -> None:
    _app()
    widget = EnvBuilderWidget()
    row = widget.controller.add_source_group(x=0.0, y=0.0)

    widget.controller.select_object(row.object_id)
    widget.inspector_panel.color_edit.setText("#123456")
    widget.inspector_panel.amount_spin.setValue(7.5)
    widget.inspector_panel.distribution_shape_combo.setCurrentText("oval")
    widget.inspector_panel.distribution_scale_x_spin.setValue(0.014)
    widget.inspector_panel.distribution_scale_y_spin.setValue(0.008)
    widget.inspector_panel._apply_selected()

    group = widget.controller.payload["food_params"]["source_groups"][row.object_id]
    assert group["color"] == "#123456"
    assert group["amount"] == 7.5
    assert group["distribution"]["shape"] == "oval"
    assert group["distribution"]["scale"] == [0.014, 0.008]

    widget.inspector_panel._delete_selected()
    assert (
        row.object_id not in widget.controller.payload["food_params"]["source_groups"]
    )
