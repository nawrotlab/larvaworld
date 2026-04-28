from __future__ import annotations

from larvaworld.gui_v2.registry_bridge import GuiEntry
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def build_environment_builder_text(
    entry: GuiEntry,
) -> tuple[str, str, list[tuple[str, str]]]:
    badges = ", ".join(entry.badges) if entry.badges else "None"
    docs = entry.docs_url or "No documentation link registered."

    blocks = [
        (
            "Rough gui_v2 composition",
            "The gui_v2 Environment Builder should likely be assembled from three legacy building blocks: "
            "environment configuration, interactive arena drawing, and body/agent drawing. "
            "This is a rough composition study, not final integration yet.",
        ),
        (
            "Legacy source mapping",
            "1. `src/larvaworld/gui/tabs/env.py`\n"
            "   - structured environment configuration editor\n"
            "   - arena, food groups/units, borders, odorscape, windscape\n\n"
            "2. `src/larvaworld/gui/tabs/env_draw.py`\n"
            "   - interactive arena/environment drawing tab\n"
            "   - add food, larvae, borders\n"
            "   - move / inspect / erase items on canvas\n\n"
            "3. `src/larvaworld/gui/tabs/body_draw.py`\n"
            "   - interactive body/agent drawing tab\n"
            "   - body points, segments, sensor positions\n"
            "   - useful if Environment Builder grows into agent-placement/body-related editing",
        ),
        (
            "Provisional gui_v2 interpretation",
            "Near-term rough mapping inside gui_v2 could be:\n"
            "- Environment setup tab/pane from `env.py`\n"
            "- Arena drawing surface from `env_draw.py`\n"
            "- Optional secondary body editor surface from `body_draw.py`\n\n"
            "This suggests that the current browser-side Environment Builder and the legacy desktop editors "
            "should eventually converge into one desktop-native composite tool.",
        ),
        (
            "Current shared metadata",
            f"Lane: {entry.lane_title}\n"
            f"Kind: {entry.kind}\n"
            f"Status: {entry.status}\n"
            f"Level: {entry.level}\n"
            f"CTA: {entry.cta}\n"
            f"Badges: {badges}\n"
            f"Docs: {docs}",
        ),
    ]

    return entry.title, "Rough legacy-to-gui_v2 mapping view.", blocks


def build_environment_builder_widget(entry: GuiEntry) -> QWidget:
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(14)

    intro = _card(
        "Composition study",
        "Rough gui_v2 Environment Builder composition that maps the legacy desktop editors into "
        "one integrated shell-native tool. The `Arena draw` tab below is now a first interactive "
        "prototype based on the legacy `env_draw.py` interaction model.",
    )
    layout.addWidget(intro)

    tabs = QTabWidget()
    tabs.setDocumentMode(True)
    tabs.setStyleSheet(
        """
        QTabWidget::pane {
            border: 1px solid #d1d5db;
            background: #ffffff;
            border-radius: 10px;
        }
        QTabBar::tab {
            background: #f3f4f6;
            color: #374151;
            border: 1px solid #d1d5db;
            padding: 8px 14px;
            margin-right: 6px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: 600;
        }
        QTabBar::tab:selected {
            background: #ede9f0;
            color: #111827;
            border-color: #c1b0c2;
        }
        """
    )
    tabs.addTab(_environment_setup_tab(), "Environment setup")
    tabs.addTab(_arena_draw_tab(), "Arena draw")
    tabs.addTab(_body_draw_tab(), "Body draw")
    layout.addWidget(tabs)

    meta = _card(
        "Shared metadata",
        f"Lane: {entry.lane_title} · Status: {entry.status} · Level: {entry.level} · "
        f"CTA: {entry.cta} · Docs: {entry.docs_url or 'No documentation link registered.'}",
    )
    layout.addWidget(meta)
    layout.addStretch(1)
    return host


def _environment_setup_tab() -> QWidget:
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(14)

    layout.addWidget(_card("Legacy source", "`src/larvaworld/gui/tabs/env.py`"))

    grid = QWidget()
    grid_layout = QGridLayout(grid)
    grid_layout.setContentsMargins(0, 0, 0, 0)
    grid_layout.setHorizontalSpacing(12)
    grid_layout.setVerticalSpacing(12)
    grid_layout.addWidget(
        _mini_panel("Arena", "shape · dimensions · reset/new arena"), 0, 0
    )
    grid_layout.addWidget(_mini_panel("Food", "groups · units · food grid"), 0, 1)
    grid_layout.addWidget(_mini_panel("Borders", "border list · ids · geometry"), 1, 0)
    grid_layout.addWidget(_mini_panel("Fields", "odorscape · windscape"), 1, 1)
    layout.addWidget(grid)

    layout.addWidget(
        _card(
            "Proposed gui_v2 role",
            "Configuration pane that stays docked next to the drawing surface. It should host the "
            "structured environment metadata while the user edits the arena visually.",
        )
    )
    layout.addStretch(1)
    return host


def _arena_draw_tab() -> QWidget:
    return ArenaDrawPrototype()


def _body_draw_tab() -> QWidget:
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(14)

    layout.addWidget(_card("Legacy source", "`src/larvaworld/gui/tabs/body_draw.py`"))

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(12)
    row.addWidget(
        _mini_panel("Body config", "points · segments · olfaction · touch sensors"), 0
    )
    row.addWidget(_body_mock(), 1)
    layout.addLayout(row)

    layout.addWidget(
        _card(
            "Possible fit inside Environment Builder",
            "Body drawing may stay optional rather than always visible. It is still useful as a related "
            "editor if Environment Builder evolves into a broader arena + agent-placement desktop tool.",
        )
    )
    layout.addStretch(1)
    return host


class ArenaCanvasView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, owner: "ArenaDrawPrototype") -> None:
        super().__init__(scene)
        self.owner = owner
        self.setRenderHints(self.renderHints())
        self.setStyleSheet(
            "background:#ffffff;border:1px solid #cbd5e1;border-radius:8px;"
        )

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        position = self.mapToScene(event.position().toPoint())
        self.owner.handle_canvas_click(position)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        super().mouseReleaseEvent(event)
        self.owner.handle_canvas_release()


class ArenaDrawPrototype(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.scene_width = 620
        self.scene_height = 380
        self.scene = QGraphicsScene(0, 0, self.scene_width, self.scene_height)
        self.arena_rect: QGraphicsRectItem | None = None
        self.arena_circle: QGraphicsEllipseItem | None = None
        self.preview_point: QGraphicsEllipseItem | None = None
        self.pending_border_start: QPointF | None = None
        self.item_records: dict[str, dict[str, object]] = {}
        self.tool_buttons: dict[str, QPushButton] = {}
        self.mode_buttons: dict[str, QPushButton] = {}
        self.current_tool = "food"
        self.current_mode = "add"
        self.info_label = QLabel()
        self.action_label = QLabel()
        self.arena_shape = "rectangular"
        self.arena_width_m = 0.20
        self.arena_height_m = 0.20
        self.current_color = "#4caf50"
        self.border_width_m = 0.001

        self.shape_combo = QComboBox()
        self.width_spin = QDoubleSpinBox()
        self.height_spin = QDoubleSpinBox()
        self.object_id_edit = QLineEdit()
        self.color_combo = QComboBox()
        self.border_width_spin = QDoubleSpinBox()

        self._build()
        self._draw_arena()
        self._refresh_control_visibility()
        self._sync_object_id()
        self._sync_buttons()
        self._set_info(
            "Legacy env_draw-style prototype. Add Food, Larva, or Border items directly on the canvas."
        )
        self._set_action("Ready.")

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        layout.addWidget(
            _card("Legacy source", "`src/larvaworld/gui/tabs/env_draw.py`")
        )

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(12)
        layout.addLayout(row)

        controls = QFrame()
        controls.setFixedWidth(250)
        controls.setStyleSheet(
            "background:#f8fafc;border:1px solid #cbd5e1;border-radius:10px;"
        )
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(14, 14, 14, 14)
        controls_layout.setSpacing(12)

        controls_layout.addWidget(_section_label("Arena"))
        arena_row = QHBoxLayout()
        arena_row.setContentsMargins(0, 0, 0, 0)
        arena_row.setSpacing(8)
        self.shape_combo.addItems(["rectangular", "circular"])
        self.shape_combo.currentTextChanged.connect(self._on_arena_changed)
        arena_row.addWidget(self.shape_combo, 1)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._reset_arena)
        new_button = QPushButton("New")
        new_button.clicked.connect(self._new_arena)
        reset_button.setStyleSheet(_button_style(False, "#b0b4c2"))
        new_button.setStyleSheet(_button_style(False, "#b0b4c2"))
        arena_row.addWidget(reset_button)
        arena_row.addWidget(new_button)
        controls_layout.addLayout(arena_row)

        dim_row = QHBoxLayout()
        dim_row.setContentsMargins(0, 0, 0, 0)
        dim_row.setSpacing(8)
        self.width_spin.setDecimals(2)
        self.width_spin.setRange(0.05, 1.0)
        self.width_spin.setSingleStep(0.05)
        self.width_spin.setValue(self.arena_width_m)
        self.width_spin.setPrefix("W ")
        self.width_spin.valueChanged.connect(self._on_arena_changed)
        self.height_spin.setDecimals(2)
        self.height_spin.setRange(0.05, 1.0)
        self.height_spin.setSingleStep(0.05)
        self.height_spin.setValue(self.arena_height_m)
        self.height_spin.setPrefix("H ")
        self.height_spin.valueChanged.connect(self._on_arena_changed)
        dim_row.addWidget(self.width_spin)
        dim_row.addWidget(self.height_spin)
        controls_layout.addLayout(dim_row)

        controls_layout.addWidget(_section_label("Insert object"))
        id_row = QHBoxLayout()
        id_row.setContentsMargins(0, 0, 0, 0)
        id_row.setSpacing(8)
        self.object_id_edit.setPlaceholderText("Object id")
        id_row.addWidget(self.object_id_edit, 1)
        controls_layout.addLayout(id_row)

        controls_layout.addWidget(_section_label("Insert object"))
        for tool_id, label in (
            ("food", "Food"),
            ("larva", "Larva"),
            ("border", "Border segment"),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, value=tool_id: self._set_tool(value)
            )
            self.tool_buttons[tool_id] = button
            controls_layout.addWidget(button)

        self.color_combo.addItems(["green", "black", "orange", "magenta", "blue"])
        self.color_combo.currentTextChanged.connect(self._on_color_changed)
        controls_layout.addWidget(self.color_combo)
        self.current_color = _named_color(self.color_combo.currentText())

        self.border_width_spin.setDecimals(4)
        self.border_width_spin.setRange(0.0005, 0.01)
        self.border_width_spin.setSingleStep(0.0005)
        self.border_width_spin.setValue(self.border_width_m)
        self.border_width_spin.setPrefix("Border width ")
        self.border_width_spin.setSuffix(" m")
        self.border_width_spin.valueChanged.connect(self._on_border_width_changed)
        controls_layout.addWidget(self.border_width_spin)

        controls_layout.addSpacing(4)
        controls_layout.addWidget(_section_label("Action mode"))
        for mode_id, label in (
            ("add", "Add"),
            ("move", "Move"),
            ("inspect", "Inspect"),
            ("erase", "Erase"),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, value=mode_id: self._set_mode(value)
            )
            self.mode_buttons[mode_id] = button
            controls_layout.addWidget(button)

        controls_layout.addSpacing(4)
        controls_layout.addWidget(_section_label("Quick source note"))
        source_note = QLabel(
            "This prototype mirrors the legacy env_draw logic more closely than the browser-side "
            "Environment Builder: Food, Larva, and Border tools are taken directly from the old desktop tab."
        )
        source_note.setWordWrap(True)
        source_note.setStyleSheet("color:#4b5563;border:none;")
        source_note.setFont(_font(10, False))
        controls_layout.addWidget(source_note)
        controls_layout.addStretch(1)
        row.addWidget(controls, 0)

        canvas_frame = QFrame()
        canvas_frame.setStyleSheet(
            "background:#f8fafc;border:1px solid #cbd5e1;border-radius:10px;"
        )
        canvas_layout = QVBoxLayout(canvas_frame)
        canvas_layout.setContentsMargins(14, 14, 14, 14)
        canvas_layout.setSpacing(10)

        title = QLabel("Arena drawing surface")
        title.setFont(_font(12, True))
        title.setStyleSheet("color:#111827;border:none;")
        canvas_layout.addWidget(title)

        subtitle = QLabel(
            "First interactive gui_v2 prototype derived from env_draw behavior. Border segments require two clicks."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#4b5563;border:none;")
        subtitle.setFont(_font(10, False))
        canvas_layout.addWidget(subtitle)

        view = ArenaCanvasView(self.scene, self)
        view.setMinimumHeight(340)
        canvas_layout.addWidget(view, 1)

        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            "background:#ffffff;color:#374151;border:1px solid #d1d5db;border-radius:8px;padding:10px;"
        )
        self.info_label.setFont(_font(10, False))
        canvas_layout.addWidget(self.info_label)

        self.action_label.setWordWrap(True)
        self.action_label.setStyleSheet(
            "background:#ffffff;color:#374151;border:1px solid #d1d5db;border-radius:8px;padding:10px;"
        )
        self.action_label.setFont(_font(10, False))
        canvas_layout.addWidget(self.action_label)

        row.addWidget(canvas_frame, 1)

        layout.addWidget(
            _card(
                "Proposed gui_v2 role",
                "Primary interactive surface for arena geometry and object placement. This first pass is "
                "native PySide6, but it is explicitly driven by the legacy env_draw interaction pattern.",
            )
        )
        layout.addStretch(1)

    def _draw_arena(self) -> None:
        if self.arena_rect is not None:
            self.scene.removeItem(self.arena_rect)
            self.arena_rect = None
        if self.arena_circle is not None:
            self.scene.removeItem(self.arena_circle)
            self.arena_circle = None
        pen = QPen(QColor("#6b7280"))
        pen.setWidth(2)
        rect = self._arena_scene_rect()
        if self.arena_shape == "circular":
            diameter = min(rect.width(), rect.height())
            x = rect.center().x() - diameter / 2
            y = rect.center().y() - diameter / 2
            self.arena_circle = self.scene.addEllipse(
                x,
                y,
                diameter,
                diameter,
                pen,
                QBrush(QColor("#ffffff")),
            )
        else:
            self.arena_rect = self.scene.addRect(rect, pen, QBrush(QColor("#ffffff")))

    def _set_tool(self, tool_id: str) -> None:
        self.current_tool = tool_id
        self.pending_border_start = None
        self._clear_preview()
        self._refresh_control_visibility()
        self._sync_object_id()
        self._sync_buttons()
        self._set_info(f"Insert tool set to {self.tool_buttons[tool_id].text()}.")

    def _set_mode(self, mode_id: str) -> None:
        self.current_mode = mode_id
        self.pending_border_start = None
        self._clear_preview()
        self._sync_buttons()
        self._set_items_movable(mode_id == "move")
        self._set_action(f"Action mode set to {self.mode_buttons[mode_id].text()}.")

    def _sync_buttons(self) -> None:
        for tool_id, button in self.tool_buttons.items():
            active = tool_id == self.current_tool
            button.setStyleSheet(_button_style(active, "#c1b0c2"))
        for mode_id, button in self.mode_buttons.items():
            active = mode_id == self.current_mode
            button.setStyleSheet(_button_style(active, "#b0b4c2"))

    def _set_info(self, message: str) -> None:
        self.info_label.setText(message)

    def _set_action(self, message: str) -> None:
        self.action_label.setText(message)

    def _set_items_movable(self, movable: bool) -> None:
        for record in self.item_records.values():
            item = record["item"]
            item.setFlag(item.GraphicsItemFlag.ItemIsMovable, movable)

    def _on_arena_changed(self, *_args) -> None:
        self.arena_shape = self.shape_combo.currentText()
        self.arena_width_m = float(self.width_spin.value())
        self.arena_height_m = float(self.height_spin.value())
        self._draw_arena()
        self._set_info(
            f"Arena updated to {self.arena_shape} with dimensions "
            f"{self.arena_width_m:.2f}m × {self.arena_height_m:.2f}m."
        )

    def _on_color_changed(self, color_name: str) -> None:
        self.current_color = _named_color(color_name)

    def _on_border_width_changed(self, value: float) -> None:
        self.border_width_m = float(value)

    def _refresh_control_visibility(self) -> None:
        self.border_width_spin.setVisible(self.current_tool == "border")

    def _reset_arena(self) -> None:
        for record in self.item_records.values():
            self.scene.removeItem(record["item"])
        self.item_records.clear()
        self.pending_border_start = None
        self._clear_preview()
        self._draw_arena()
        self._sync_object_id()
        self._set_action("Arena has been reset.")

    def _new_arena(self) -> None:
        self._reset_arena()
        self._set_action("New arena initialized. All items erased.")

    def _arena_scene_rect(self) -> QRectF:
        margin_x = 70
        margin_y = 40
        available_w = self.scene_width - 2 * margin_x
        available_h = self.scene_height - 2 * margin_y
        if self.arena_shape == "circular":
            diameter = min(available_w, available_h)
            return self.scene.sceneRect().adjusted(
                (self.scene_width - diameter) / 2,
                (self.scene_height - diameter) / 2,
                -(self.scene_width - diameter) / 2,
                -(self.scene_height - diameter) / 2,
            )
        ratio = self.arena_width_m / max(self.arena_height_m, 1e-6)
        draw_w = available_w
        draw_h = draw_w / ratio
        if draw_h > available_h:
            draw_h = available_h
            draw_w = draw_h * ratio
        x = (self.scene_width - draw_w) / 2
        y = (self.scene_height - draw_h) / 2
        return QRectF(x, y, draw_w, draw_h)

    def _is_inside_arena(self, position: QPointF) -> bool:
        if self.arena_circle is not None:
            circle = self.arena_circle.rect()
            center = circle.center()
            radius = circle.width() / 2
            dx = position.x() - center.x()
            dy = position.y() - center.y()
            return (dx * dx + dy * dy) <= radius * radius
        if self.arena_rect is not None:
            return self.arena_rect.rect().contains(position)
        return False

    def _to_logical(self, position: QPointF) -> tuple[float, float]:
        rect = self._arena_scene_rect()
        rel_x = (position.x() - rect.center().x()) / max(rect.width(), 1e-6)
        rel_y = (rect.center().y() - position.y()) / max(rect.height(), 1e-6)
        return rel_x * self.arena_width_m, rel_y * self.arena_height_m

    def handle_canvas_click(self, position: QPointF) -> None:
        if self.arena_rect is None and self.arena_circle is None:
            return
        if not self._is_inside_arena(position):
            return

        clicked_items = [
            item
            for item in self.scene.items(position)
            if item not in {self.arena_rect, self.arena_circle, self.preview_point}
        ]

        if self.current_mode == "inspect":
            self._inspect_item(clicked_items, position)
            return
        if self.current_mode == "erase":
            self._erase_item(clicked_items)
            return
        if self.current_mode == "move":
            self._set_action(
                "Move mode active. Drag existing items directly on the canvas."
            )
            return

        if self.current_tool == "border":
            self._handle_border_click(position)
            return

        self._add_point_item(position, self.current_tool)

    def _add_point_item(self, position: QPointF, item_type: str) -> None:
        color = self.current_color
        radius = 9 if item_type == "food" else 7
        item = self.scene.addEllipse(
            position.x() - radius,
            position.y() - radius,
            radius * 2,
            radius * 2,
            QPen(QColor(color)),
            QBrush(QColor(color)),
        )
        item_id = self.object_id_edit.text().strip() or self._next_object_id(item_type)
        item.setData(0, item_type)
        item.setData(1, item_id)
        self.item_records[item_id] = {
            "type": item_type,
            "item": item,
            "color": color,
            "logical": self._to_logical(position),
        }
        self._set_items_movable(self.current_mode == "move")
        lx, ly = self.item_records[item_id]["logical"]
        self._set_action(
            f"{item_type.title()} {item_id} placed at ({lx:.4f}, {ly:.4f})."
        )
        self._sync_object_id()

    def _handle_border_click(self, position: QPointF) -> None:
        if self.pending_border_start is None:
            self.pending_border_start = position
            self._clear_preview()
            self.preview_point = self.scene.addEllipse(
                position.x() - 4,
                position.y() - 4,
                8,
                8,
                QPen(QColor("#111111")),
                QBrush(QColor("#111111")),
            )
            self._set_info(
                "Border start point placed. Click a second point to finish the segment."
            )
            return

        item_id = self.object_id_edit.text().strip() or self._next_object_id("border")
        line = self.scene.addLine(
            self.pending_border_start.x(),
            self.pending_border_start.y(),
            position.x(),
            position.y(),
            QPen(QColor(self.current_color), max(2, int(self.border_width_m * 2000))),
        )
        line.setData(0, "border")
        line.setData(1, item_id)
        self.item_records[item_id] = {
            "type": "border",
            "item": line,
            "color": self.current_color,
            "width": self.border_width_m,
            "logical_points": (
                self._to_logical(self.pending_border_start),
                self._to_logical(position),
            ),
        }
        self.pending_border_start = None
        self._clear_preview()
        self._set_items_movable(self.current_mode == "move")
        p1, p2 = self.item_records[item_id]["logical_points"]
        self._set_action(
            f"Border {item_id} placed from ({p1[0]:.4f}, {p1[1]:.4f}) to ({p2[0]:.4f}, {p2[1]:.4f})."
        )
        self._sync_object_id()

    def _inspect_item(self, clicked_items: list[object], position: QPointF) -> None:
        if not clicked_items:
            lx, ly = self._to_logical(position)
            self._set_action(f"Inspect mode: no item at ({lx:.4f}, {ly:.4f}).")
            return
        item = clicked_items[0]
        item_type = item.data(0) or "unknown"
        item_id = item.data(1) or "unknown"
        if isinstance(item, QGraphicsLineItem):
            logical = self.item_records.get(item_id, {}).get("logical_points")
            if logical is not None:
                p1, p2 = logical
                self._set_action(
                    f"Inspect border {item_id}: ({p1[0]:.4f}, {p1[1]:.4f}) -> ({p2[0]:.4f}, {p2[1]:.4f})."
                )
                return
            line = item.line()
            self._set_action(
                f"Inspect border segment: ({line.x1():.1f}, {line.y1():.1f}) -> ({line.x2():.1f}, {line.y2():.1f})."
            )
            return
        logical = self.item_records.get(item_id, {}).get("logical")
        if logical is not None:
            self._set_action(
                f"Inspect {item_type} {item_id}: ({logical[0]:.4f}, {logical[1]:.4f})."
            )
            return
        rect = item.rect()
        self._set_action(
            f"Inspect {item_type}: center ≈ ({rect.center().x() + item.pos().x():.1f}, {rect.center().y() + item.pos().y():.1f})."
        )

    def _erase_item(self, clicked_items: list[object]) -> None:
        if not clicked_items:
            self._set_action("Erase mode: no item selected.")
            return
        target = clicked_items[0]
        item_id = target.data(1)
        self.scene.removeItem(target)
        if item_id in self.item_records:
            self.item_records.pop(item_id)
        item_type = target.data(0) or "item"
        self._set_action(f"Item {item_id or item_type} erased.")
        self._sync_object_id()

    def handle_canvas_release(self) -> None:
        if self.current_mode != "move":
            return
        moved = False
        for record in self.item_records.values():
            item = record["item"]
            if isinstance(item, QGraphicsLineItem):
                line = item.line()
                offset = item.pos()
                p1 = QPointF(line.x1() + offset.x(), line.y1() + offset.y())
                p2 = QPointF(line.x2() + offset.x(), line.y2() + offset.y())
                logical_points = (self._to_logical(p1), self._to_logical(p2))
                if record.get("logical_points") != logical_points:
                    record["logical_points"] = logical_points
                    moved = True
            else:
                rect = item.rect()
                center = rect.center() + item.pos()
                logical = self._to_logical(center)
                if record.get("logical") != logical:
                    record["logical"] = logical
                    moved = True
        if moved:
            self._set_action("Item positions updated after drag.")

    def _clear_preview(self) -> None:
        if self.preview_point is not None:
            self.scene.removeItem(self.preview_point)
            self.preview_point = None

    def _next_object_id(self, item_type: str) -> str:
        prefix = {"food": "Food", "larva": "Larva", "border": "Border"}.get(
            item_type, "Item"
        )
        count = 1
        while f"{prefix}_{count}" in self.item_records:
            count += 1
        return f"{prefix}_{count}"

    def _sync_object_id(self) -> None:
        self.object_id_edit.setText(self._next_object_id(self.current_tool))


def _body_mock() -> QWidget:
    frame = QFrame()
    frame.setMinimumHeight(280)
    frame.setStyleSheet(
        "background:#f8fafc;border:1px solid #cbd5e1;border-radius:10px;"
    )
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(10)

    title = QLabel("Body geometry surface")
    title.setFont(_font(12, True))
    title.setStyleSheet("color:#111827;border:none;")
    layout.addWidget(title)

    canvas = QLabel(
        "Body placeholder\n\n- points\n- segments\n- sensors\n- direct drag editing"
    )
    canvas.setAlignment(Qt.AlignCenter)
    canvas.setStyleSheet(
        "background:#ffffff;color:#475569;border:1px dashed #94a3b8;border-radius:8px;padding:24px;"
    )
    canvas.setMinimumHeight(180)
    layout.addWidget(canvas, 1)
    return frame


def _mini_panel(title: str, body: str) -> QWidget:
    frame = QFrame()
    frame.setStyleSheet(
        "background:#ffffff;border:1px solid #d1d5db;border-radius:10px;"
    )
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(8)

    title_label = QLabel(title)
    title_label.setFont(_font(11, True))
    title_label.setStyleSheet("color:#111827;border:none;")
    layout.addWidget(title_label)

    body_label = QLabel(body)
    body_label.setWordWrap(True)
    body_label.setStyleSheet("color:#4b5563;border:none;")
    body_label.setFont(_font(10, False))
    layout.addWidget(body_label)
    return frame


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setFont(_font(10, True))
    label.setStyleSheet("color:#111827;border:none;")
    return label


def _button_style(active: bool, accent: str) -> str:
    if active:
        return (
            f"background:{accent};color:#111827;border:1px solid #9ca3af;"
            "border-radius:8px;padding:8px 12px;font-weight:600;"
        )
    return (
        "background:#ffffff;color:#374151;border:1px solid #d1d5db;"
        "border-radius:8px;padding:8px 12px;"
    )


def _named_color(name: str) -> str:
    return {
        "green": "#4caf50",
        "black": "#111111",
        "orange": "#f59e0b",
        "magenta": "#d946ef",
        "blue": "#2563eb",
    }.get(name, "#4caf50")


def _card(title: str, body: str) -> QWidget:
    frame = QFrame()
    frame.setStyleSheet(
        "background:#ffffff;border:1px solid #d1d5db;border-radius:10px;"
    )
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(8)

    title_label = QLabel(title)
    title_label.setFont(_font(12, True))
    title_label.setStyleSheet("color:#111827;border:none;")
    layout.addWidget(title_label)

    body_label = QLabel(body)
    body_label.setWordWrap(True)
    body_label.setTextFormat(Qt.PlainText)
    body_label.setStyleSheet("color:#374151;border:none;")
    body_label.setFont(_font(10, False))
    layout.addWidget(body_label)
    return frame


def _font(size: int, bold: bool) -> QFont:
    font = QFont("Helvetica", size)
    font.setBold(bold)
    return font
