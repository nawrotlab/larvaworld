from __future__ import annotations

from dataclasses import asdict

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from larvaworld.gui_v2.apps.models_environments.env_builder_controller import (
    EnvBuilderController,
)
from larvaworld.gui_v2.gui_aux.qt_layout import section_card
from larvaworld.portal.models_architecture.environment_builder_common import (
    BORDER_SEGMENT,
    SOURCE_GROUP,
    SOURCE_UNIT,
    normalize_preset_filename,
)
from larvaworld.portal.workspace import get_active_workspace


class EnvBuilderLeftPanel(QWidget):
    def __init__(self, controller: EnvBuilderController) -> None:
        super().__init__()
        self.controller = controller
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.preset_card, preset_layout = section_card("Presets")
        layout.addWidget(self.preset_card)
        self.preset_name_edit = QLineEdit()
        self.preset_name_edit.setPlaceholderText("environment_builder_config")
        self.preset_name_edit.editingFinished.connect(self._on_preset_name_changed)
        preset_form = QFormLayout()
        preset_form.setContentsMargins(0, 0, 0, 0)
        preset_layout.addLayout(preset_form)
        preset_form.addRow("Preset name", self.preset_name_edit)

        preset_buttons = QHBoxLayout()
        self.load_workspace_button = QPushButton("Load workspace")
        self.load_registry_button = QPushButton("Load registry")
        self.save_workspace_button = QPushButton("Save workspace")
        self.export_button = QPushButton("Export JSON")
        self.reset_button = QPushButton("Reset default")
        self.load_workspace_button.clicked.connect(self._load_workspace)
        self.load_registry_button.clicked.connect(self._load_registry)
        self.save_workspace_button.clicked.connect(self._save_workspace)
        self.export_button.clicked.connect(self._export_json)
        self.reset_button.clicked.connect(self._reset_default)
        for button in (
            self.load_workspace_button,
            self.load_registry_button,
            self.save_workspace_button,
            self.export_button,
            self.reset_button,
        ):
            preset_buttons.addWidget(button)
        preset_layout.addLayout(preset_buttons)

        self.arena_card, arena_layout = section_card("Arena")
        layout.addWidget(self.arena_card)
        self.arena_shape_combo = QComboBox()
        self.arena_shape_combo.addItems(["rectangular", "circular"])
        self.arena_shape_combo.currentTextChanged.connect(self._on_arena_shape_changed)
        self.arena_width_spin = QDoubleSpinBox()
        self.arena_width_spin.setRange(0.01, 10.0)
        self.arena_width_spin.setDecimals(4)
        self.arena_width_spin.setSingleStep(0.01)
        self.arena_width_spin.valueChanged.connect(self._on_arena_width_changed)
        self.arena_height_spin = QDoubleSpinBox()
        self.arena_height_spin.setRange(0.01, 10.0)
        self.arena_height_spin.setDecimals(4)
        self.arena_height_spin.setSingleStep(0.01)
        self.arena_height_spin.valueChanged.connect(self._on_arena_height_changed)
        self.arena_torus_check = QCheckBox("Toroidal")
        self.arena_torus_check.toggled.connect(self._on_arena_torus_changed)
        self.arena_width_label = QLabel("Arena width (m)")
        self.arena_height_label = QLabel("Arena height (m)")
        self.arena_width_label.setText("Arena width (m)")
        self.arena_height_label.setText("Arena height (m)")
        arena_form = QFormLayout()
        arena_form.setContentsMargins(0, 0, 0, 0)
        arena_layout.addLayout(arena_form)
        arena_form.addRow("Geometry", self.arena_shape_combo)
        arena_form.addRow(self.arena_width_label, self.arena_width_spin)
        arena_form.addRow(self.arena_height_label, self.arena_height_spin)
        arena_form.addRow("", self.arena_torus_check)

        self.insert_card, insert_layout = section_card("Insert")
        layout.addWidget(self.insert_card)
        self.insert_object_id_edit = QLineEdit()
        self.insert_object_id_edit.setPlaceholderText("optional object id")
        insert_form = QFormLayout()
        insert_form.setContentsMargins(0, 0, 0, 0)
        insert_layout.addLayout(insert_form)
        insert_form.addRow("Object id", self.insert_object_id_edit)
        insert_buttons = QHBoxLayout()
        self.add_unit_button = QPushButton("Add unit")
        self.add_group_button = QPushButton("Add group")
        self.add_border_button = QPushButton("Border mode")
        self.add_unit_button.clicked.connect(self._add_unit)
        self.add_group_button.clicked.connect(self._add_group)
        self.add_border_button.clicked.connect(self._add_border_mode)
        for button in (
            self.add_unit_button,
            self.add_group_button,
            self.add_border_button,
        ):
            insert_buttons.addWidget(button)
        insert_layout.addLayout(insert_buttons)

        self.mode_card, mode_layout = section_card("Canvas modes")
        layout.addWidget(self.mode_card)
        self.select_mode_button = QPushButton("Select")
        self.move_mode_button = QPushButton("Move")
        self.erase_mode_button = QPushButton("Erase")
        self.select_mode_button.clicked.connect(lambda: self._set_mode("select"))
        self.move_mode_button.clicked.connect(lambda: self._set_mode("move"))
        self.erase_mode_button.clicked.connect(lambda: self._set_mode("erase"))
        mode_buttons = QHBoxLayout()
        for button in (
            self.select_mode_button,
            self.move_mode_button,
            self.erase_mode_button,
        ):
            button.setCheckable(True)
            mode_buttons.addWidget(button)
        mode_layout.addLayout(mode_buttons)

        layout.addStretch(1)
        self.refresh_from_controller()

    def refresh_from_controller(self) -> None:
        with QSignalBlocker(self.preset_name_edit):
            self.preset_name_edit.setText(self.controller.preset_name)
        with QSignalBlocker(self.arena_shape_combo):
            self.arena_shape_combo.setCurrentText(
                self.controller.canvas_state.arena.geometry
            )
        with QSignalBlocker(self.arena_width_spin):
            if self.controller.canvas_state.arena.geometry == "circular":
                self.arena_width_spin.setValue(
                    self.controller.canvas_state.arena.dims[0] / 2.0
                )
            else:
                self.arena_width_spin.setValue(
                    self.controller.canvas_state.arena.dims[0]
                )
        with QSignalBlocker(self.arena_height_spin):
            if self.controller.canvas_state.arena.geometry == "circular":
                self.arena_height_spin.setValue(
                    self.controller.canvas_state.arena.dims[1] / 2.0
                )
            else:
                self.arena_height_spin.setValue(
                    self.controller.canvas_state.arena.dims[1]
                )
        with QSignalBlocker(self.arena_torus_check):
            self.arena_torus_check.setChecked(self.controller.canvas_state.arena.torus)

        geometry = self.controller.canvas_state.arena.geometry
        if geometry == "circular":
            self.arena_width_label.setText("Arena radius (m)")
            self.arena_height_label.setText("Arena height (m)")
            self.arena_height_spin.setVisible(False)
            self.arena_height_label.setVisible(False)
        else:
            self.arena_width_label.setText("Arena width (m)")
            self.arena_height_label.setText("Arena height (m)")
            self.arena_height_spin.setVisible(True)
            self.arena_height_label.setVisible(True)

        self.save_workspace_button.setEnabled(get_active_workspace() is not None)
        self._set_mode_button_state(self.controller.interaction_mode)

    def _set_mode_button_state(self, mode: str) -> None:
        for button, key in (
            (self.select_mode_button, "select"),
            (self.move_mode_button, "move"),
            (self.erase_mode_button, "erase"),
        ):
            button.setChecked(key == mode)

    def _on_preset_name_changed(self) -> None:
        self.controller.set_preset_name(self.preset_name_edit.text())

    def _load_workspace(self) -> None:
        self.controller.load_workspace_json(self.preset_name_edit.text())

    def _load_registry(self) -> None:
        self.controller.load_registry(self.preset_name_edit.text())

    def _save_workspace(self) -> None:
        self.controller.save_workspace(self.preset_name_edit.text())

    def _export_json(self) -> None:
        self.controller.current_export_json = self.controller.export_json()
        self.controller.set_status(
            f"Export JSON updated for {normalize_preset_filename(self.preset_name_edit.text() or self.controller.preset_name)}"
        )

    def _reset_default(self) -> None:
        result = QMessageBox.question(
            self,
            "Reset environment",
            "Reset the environment builder to the default environment?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            self.controller.set_status("Reset cancelled.")
            return
        self.controller.reset_to_default()

    def _set_mode(self, mode: str) -> None:
        self.controller.set_interaction_mode(mode)
        self._set_mode_button_state(mode)

    def _add_unit(self) -> None:
        if self.controller.interaction_mode == "add_unit":
            self.controller.set_interaction_mode("select")
            self._set_mode_button_state("select")
            return
        self.controller.set_interaction_mode("add_unit")
        self._set_mode_button_state("add_unit")

    def _add_group(self) -> None:
        self.controller.add_source_group(
            object_id=self.insert_object_id_edit.text().strip() or None
        )

    def _add_border_mode(self) -> None:
        self.controller.set_interaction_mode("add_border")
        self._set_mode_button_state("add_border")

    def _on_arena_shape_changed(self, value: str) -> None:
        self.controller.set_arena(geometry=value)

    def _on_arena_width_changed(self, value: float) -> None:
        self.controller.set_arena(width=float(value))

    def _on_arena_height_changed(self, value: float) -> None:
        self.controller.set_arena(height=float(value))

    def _on_arena_torus_changed(self, checked: bool) -> None:
        self.controller.set_arena(torus=bool(checked))


class EnvBuilderInspectorPanel(QWidget):
    def __init__(self, controller: EnvBuilderController) -> None:
        super().__init__()
        self.controller = controller
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.table_card, table_layout = section_card("Objects")
        layout.addWidget(self.table_card)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["id", "type", "x", "y", "x2", "y2", "radius", "width", "amount"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        table_layout.addWidget(self.table)

        self.editor_card, editor_layout = section_card("Selected object")
        layout.addWidget(self.editor_card)
        self.editor_form = QFormLayout()
        self.editor_form.setContentsMargins(0, 0, 0, 0)
        editor_layout.addLayout(self.editor_form)

        self.object_id_edit = QLineEdit()
        self.object_type_label = QLineEdit()
        self.object_type_label.setReadOnly(True)
        self.x_spin = self._spin()
        self.y_spin = self._spin()
        self.x2_spin = self._spin()
        self.y2_spin = self._spin()
        self.radius_spin = self._spin()
        self.width_spin = self._spin()
        self.color_edit = QLineEdit()
        self.amount_spin = self._spin()
        self.odor_id_edit = QLineEdit()
        self.odor_intensity_spin = self._spin()
        self.odor_spread_spin = self._spin()
        self.substrate_type_edit = QLineEdit()
        self.substrate_quality_spin = self._spin()
        self.can_be_carried_check = QCheckBox()
        self.can_be_displaced_check = QCheckBox()
        self.regeneration_check = QCheckBox()
        self.distribution_mode_edit = QLineEdit()
        self.distribution_shape_combo = QComboBox()
        self.distribution_shape_combo.addItems(["circle", "oval", "rect"])
        self.distribution_n_spin = QSpinBox()
        self.distribution_n_spin.setRange(0, 1_000_000)
        self.distribution_scale_x_spin = self._spin()
        self.distribution_scale_y_spin = self._spin()
        self.distribution_show_shape_check = QCheckBox()

        for label, widget in (
            ("Object id", self.object_id_edit),
            ("Object type", self.object_type_label),
            ("x", self.x_spin),
            ("y", self.y_spin),
            ("x2", self.x2_spin),
            ("y2", self.y2_spin),
            ("radius", self.radius_spin),
            ("width", self.width_spin),
            ("color", self.color_edit),
            ("amount", self.amount_spin),
            ("odor id", self.odor_id_edit),
            ("odor intensity", self.odor_intensity_spin),
            ("odor spread", self.odor_spread_spin),
            ("substrate type", self.substrate_type_edit),
            ("substrate quality", self.substrate_quality_spin),
            ("can be carried", self.can_be_carried_check),
            ("can be displaced", self.can_be_displaced_check),
            ("regeneration", self.regeneration_check),
            ("distribution mode", self.distribution_mode_edit),
            ("distribution shape", self.distribution_shape_combo),
            ("distribution n", self.distribution_n_spin),
            ("distribution scale x", self.distribution_scale_x_spin),
            ("distribution scale y", self.distribution_scale_y_spin),
            ("show shape", self.distribution_show_shape_check),
        ):
            self.editor_form.addRow(label, widget)

        button_row = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.delete_button = QPushButton("Delete selected")
        self.apply_button.clicked.connect(self._apply_selected)
        self.delete_button.clicked.connect(self._delete_selected)
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.delete_button)
        editor_layout.addLayout(button_row)

        self.distribution_shape_combo.currentTextChanged.connect(
            self._sync_group_shape_controls
        )
        self.distribution_scale_x_spin.valueChanged.connect(self._sync_circle_scale)
        layout.addStretch(1)
        self.refresh_from_controller()

    def refresh_from_controller(self) -> None:
        rows = list(self.controller.object_rows)
        with QSignalBlocker(self.table):
            self.table.setRowCount(len(rows))
            for index, row in enumerate(rows):
                values = [
                    row.object_id,
                    row.object_type,
                    row.x,
                    row.y,
                    row.x2,
                    row.y2,
                    row.radius,
                    row.width,
                    row.amount,
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem("" if value is None else str(value))
                    self.table.setItem(index, col, item)
        self._load_current_row()
        self.apply_button.setEnabled(self.controller.current_row() is not None)
        self.delete_button.setEnabled(self.controller.current_row() is not None)

    def _load_current_row(self) -> None:
        row = self.controller.current_row()
        self._set_visibility_for_row(row.object_type if row else None)
        if row is None:
            for widget in self._all_edit_widgets():
                if isinstance(widget, QLineEdit):
                    widget.clear()
                elif isinstance(widget, QDoubleSpinBox):
                    widget.setValue(0.0)
                elif isinstance(widget, QSpinBox):
                    widget.setValue(0)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(False)
            return
        with QSignalBlocker(self.object_id_edit):
            self.object_id_edit.setText(row.object_id)
        with QSignalBlocker(self.object_type_label):
            self.object_type_label.setText(row.object_type)
        self._set_spin(self.x_spin, row.x)
        self._set_spin(self.y_spin, row.y)
        self._set_spin(self.x2_spin, row.x2)
        self._set_spin(self.y2_spin, row.y2)
        self._set_spin(self.radius_spin, row.radius)
        self._set_spin(self.width_spin, row.width)
        with QSignalBlocker(self.color_edit):
            self.color_edit.setText(row.color or "")
        self._set_spin(self.amount_spin, row.amount)
        with QSignalBlocker(self.odor_id_edit):
            self.odor_id_edit.setText(row.odor_id or "")
        self._set_spin(self.odor_intensity_spin, row.odor_intensity)
        self._set_spin(self.odor_spread_spin, row.odor_spread)
        with QSignalBlocker(self.substrate_type_edit):
            self.substrate_type_edit.setText(row.substrate_type or "")
        self._set_spin(self.substrate_quality_spin, row.substrate_quality)
        with QSignalBlocker(self.can_be_carried_check):
            self.can_be_carried_check.setChecked(bool(row.can_be_carried))
        with QSignalBlocker(self.can_be_displaced_check):
            self.can_be_displaced_check.setChecked(bool(row.can_be_displaced))
        with QSignalBlocker(self.regeneration_check):
            self.regeneration_check.setChecked(bool(row.regeneration))
        with QSignalBlocker(self.distribution_mode_edit):
            self.distribution_mode_edit.setText(row.distribution_mode or "")
        with QSignalBlocker(self.distribution_shape_combo):
            self.distribution_shape_combo.setCurrentText(
                row.distribution_shape or "circle"
            )
        with QSignalBlocker(self.distribution_n_spin):
            self.distribution_n_spin.setValue(int(row.distribution_n or 0))
        self._set_spin(self.distribution_scale_x_spin, row.distribution_scale_x)
        self._set_spin(self.distribution_scale_y_spin, row.distribution_scale_y)
        with QSignalBlocker(self.distribution_show_shape_check):
            self.distribution_show_shape_check.setChecked(
                True
                if row.distribution_show_shape is None
                else bool(row.distribution_show_shape)
            )
        self._sync_group_shape_controls(self.distribution_shape_combo.currentText())
        self._select_table_row(row.object_id)

    def _select_table_row(self, object_id: str | None) -> None:
        if object_id is None:
            return
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == object_id:
                with QSignalBlocker(self.table):
                    self.table.selectRow(row)
                break

    def _on_table_selection_changed(self) -> None:
        selected = self.table.selectedItems()
        if not selected:
            self.controller.select_object(None)
            return
        object_id = selected[0].text()
        self.controller.select_object(object_id)

    def _apply_selected(self) -> None:
        row = self.controller.current_row()
        if row is None:
            return
        changes = self._collect_changes(row.object_type)
        self.controller.update_selected_object(**changes)

    def _delete_selected(self) -> None:
        if self.controller.selected_object_id is not None:
            self.controller.delete_object(self.controller.selected_object_id)

    def _collect_changes(self, object_type: str) -> dict[str, object]:
        changes: dict[str, object] = {
            "object_id": self.object_id_edit.text().strip()
            or self.controller.selected_object_id
        }
        changes["x"] = self.x_spin.value()
        changes["y"] = self.y_spin.value()
        changes["color"] = self.color_edit.text().strip() or None
        if object_type == SOURCE_UNIT:
            changes.update(
                {
                    "radius": self.radius_spin.value(),
                    "amount": self.amount_spin.value(),
                    "odor_id": self.odor_id_edit.text().strip() or None,
                    "odor_intensity": self.odor_intensity_spin.value(),
                    "odor_spread": self.odor_spread_spin.value(),
                    "substrate_type": self.substrate_type_edit.text().strip() or None,
                    "substrate_quality": self.substrate_quality_spin.value(),
                    "can_be_carried": self.can_be_carried_check.isChecked(),
                    "can_be_displaced": self.can_be_displaced_check.isChecked(),
                    "regeneration": self.regeneration_check.isChecked(),
                }
            )
        elif object_type == SOURCE_GROUP:
            changes.update(
                {
                    "radius": self.radius_spin.value(),
                    "amount": self.amount_spin.value(),
                    "odor_id": self.odor_id_edit.text().strip() or None,
                    "odor_intensity": self.odor_intensity_spin.value(),
                    "odor_spread": self.odor_spread_spin.value(),
                    "substrate_type": self.substrate_type_edit.text().strip() or None,
                    "substrate_quality": self.substrate_quality_spin.value(),
                    "can_be_carried": self.can_be_carried_check.isChecked(),
                    "can_be_displaced": self.can_be_displaced_check.isChecked(),
                    "regeneration": self.regeneration_check.isChecked(),
                    "distribution_mode": self.distribution_mode_edit.text().strip()
                    or None,
                    "distribution_shape": self.distribution_shape_combo.currentText(),
                    "distribution_n": self.distribution_n_spin.value(),
                    "distribution_scale_x": self.distribution_scale_x_spin.value(),
                    "distribution_scale_y": self.distribution_scale_y_spin.value(),
                    "distribution_show_shape": self.distribution_show_shape_check.isChecked(),
                }
            )
        else:
            changes.update(
                {
                    "x2": self.x2_spin.value(),
                    "y2": self.y2_spin.value(),
                    "width": self.width_spin.value(),
                }
            )
        return changes

    def _set_visibility_for_row(self, object_type: str | None) -> None:
        unit_fields = {
            self.x2_spin,
            self.y2_spin,
            self.width_spin,
            self.distribution_mode_edit,
            self.distribution_shape_combo,
            self.distribution_n_spin,
            self.distribution_scale_x_spin,
            self.distribution_scale_y_spin,
            self.distribution_show_shape_check,
        }
        group_fields = {self.x2_spin, self.y2_spin}
        border_fields = {
            self.radius_spin,
            self.amount_spin,
            self.odor_id_edit,
            self.odor_intensity_spin,
            self.odor_spread_spin,
            self.substrate_type_edit,
            self.substrate_quality_spin,
            self.can_be_carried_check,
            self.can_be_displaced_check,
            self.regeneration_check,
            self.distribution_mode_edit,
            self.distribution_shape_combo,
            self.distribution_n_spin,
            self.distribution_scale_x_spin,
            self.distribution_scale_y_spin,
            self.distribution_show_shape_check,
        }
        if object_type == SOURCE_UNIT:
            hide = unit_fields
        elif object_type == SOURCE_GROUP:
            hide = group_fields
        elif object_type == BORDER_SEGMENT:
            hide = border_fields
        else:
            hide = set()
        for widget in self._all_edit_widgets():
            label = self.editor_form.labelForField(widget)
            visible = widget not in hide
            widget.setVisible(visible)
            if label is not None:
                label.setVisible(visible)
        self.distribution_scale_y_spin.setEnabled(
            self.distribution_shape_combo.currentText() != "circle"
        )

    def _sync_group_shape_controls(self, shape: str) -> None:
        is_group = (
            self.controller.current_row() is not None
            and self.controller.current_row().object_type == SOURCE_GROUP
        )
        self.distribution_scale_y_spin.setEnabled(shape != "circle")
        if shape == "circle" and is_group:
            with QSignalBlocker(self.distribution_scale_y_spin):
                self.distribution_scale_y_spin.setValue(
                    self.distribution_scale_x_spin.value()
                )

    def _sync_circle_scale(self, value: float) -> None:
        if self.distribution_shape_combo.currentText() != "circle":
            return
        with QSignalBlocker(self.distribution_scale_y_spin):
            self.distribution_scale_y_spin.setValue(value)

    def _all_edit_widgets(self) -> list[QWidget]:
        return [
            self.object_id_edit,
            self.object_type_label,
            self.x_spin,
            self.y_spin,
            self.x2_spin,
            self.y2_spin,
            self.radius_spin,
            self.width_spin,
            self.color_edit,
            self.amount_spin,
            self.odor_id_edit,
            self.odor_intensity_spin,
            self.odor_spread_spin,
            self.substrate_type_edit,
            self.substrate_quality_spin,
            self.can_be_carried_check,
            self.can_be_displaced_check,
            self.regeneration_check,
            self.distribution_mode_edit,
            self.distribution_shape_combo,
            self.distribution_n_spin,
            self.distribution_scale_x_spin,
            self.distribution_scale_y_spin,
            self.distribution_show_shape_check,
        ]

    def _spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1000.0, 1000.0)
        spin.setDecimals(4)
        spin.setSingleStep(0.001)
        return spin

    def _set_spin(self, spin: QDoubleSpinBox, value: float | None) -> None:
        with QSignalBlocker(spin):
            spin.setValue(0.0 if value is None else float(value))
