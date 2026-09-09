from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QTransform
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QVBoxLayout,
    QWidget,
)

from larvaworld.gui_v2.apps.models_environments.env_builder_controller import (
    EnvBuilderController,
)
from larvaworld.portal.canvas_widgets.environment_models import (
    CanvasArena,
    CanvasObject,
    EnvironmentCanvasState,
)


class EnvBuilderCanvasView(QGraphicsView):
    def __init__(self, controller: EnvBuilderController, scene: QGraphicsScene) -> None:
        super().__init__(scene)
        self.controller = controller
        self.setRenderHints(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor("#ffffff")))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMinimumSize(760, 620)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        scene_pos = self.mapToScene(event.position().toPoint())
        object_id = self._object_id_at(scene_pos)
        world_x, world_y = self._scene_to_world(scene_pos)
        self.controller.canvas_click(
            world_x=world_x, world_y=world_y, clicked_object_id=object_id
        )
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        super().mouseReleaseEvent(event)
        if self.controller.interaction_mode == "move":
            self.controller.sync_scene_items(list(self.scene().items()))

    def _object_id_at(self, scene_pos: QPointF) -> str | None:
        item = self.scene().itemAt(scene_pos, QTransform())
        if item is None:
            return None
        try:
            value = item.data(0)
        except Exception:
            return None
        return str(value) if value else None

    def _scene_to_world(self, scene_pos: QPointF) -> tuple[float, float]:
        arena = self.controller.canvas_state.arena
        scene_rect = self.scene().sceneRect()
        arena_rect = self._arena_scene_rect(arena, scene_rect)
        if arena.geometry == "circular":
            radius = arena.dims[0] / 2.0
            scale = arena_rect.width() / max(radius * 2.0, 1e-9)
        else:
            scale = arena_rect.width() / max(arena.dims[0], 1e-9)
        world_x = (scene_pos.x() - arena_rect.center().x()) / scale
        world_y = (arena_rect.center().y() - scene_pos.y()) / scale
        return float(world_x), float(world_y)


class EnvBuilderCanvasWidget(QWidget):
    def __init__(self, controller: EnvBuilderController) -> None:
        super().__init__()
        self.controller = controller
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0.0, 0.0, 760.0, 620.0)
        self.view = EnvBuilderCanvasView(controller, self.scene)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.view, 1)

        self.controller.add_listener(self.refresh_from_controller)
        self.refresh_from_controller()

    def refresh_from_controller(self) -> None:
        state = self.controller.canvas_state
        self.scene.clear()
        self.scene.setSceneRect(0.0, 0.0, 760.0, 620.0)
        arena_rect = self._arena_scene_rect(state.arena)
        self._draw_arena(state.arena, arena_rect, state.show_arena_outline)
        self._draw_food_grid(state.food_grid, arena_rect)
        self._draw_scape_labels(state, arena_rect)
        for obj in state.objects:
            self._draw_object(obj, arena_rect)
        if self.controller.selected_object_id:
            self._highlight_selected(self.controller.selected_object_id)

    def _arena_scene_rect(self, arena: CanvasArena) -> QRectF:
        scene_rect = self.scene.sceneRect()
        margin_x = 70.0
        margin_y = 50.0
        if arena.geometry == "circular":
            diameter = min(
                scene_rect.width() - 2 * margin_x, scene_rect.height() - 2 * margin_y
            )
            left = scene_rect.center().x() - diameter / 2.0
            top = scene_rect.center().y() - diameter / 2.0
            return QRectF(left, top, diameter, diameter)
        arena_w = max(float(arena.dims[0]), 1e-9)
        arena_h = max(float(arena.dims[1]), 1e-9)
        scale = min(
            (scene_rect.width() - 2 * margin_x) / arena_w,
            (scene_rect.height() - 2 * margin_y) / arena_h,
        )
        draw_w = arena_w * scale
        draw_h = arena_h * scale
        left = scene_rect.center().x() - draw_w / 2.0
        top = scene_rect.center().y() - draw_h / 2.0
        return QRectF(left, top, draw_w, draw_h)

    def _world_to_scene(
        self, x: float, y: float, arena_rect: QRectF, arena: CanvasArena
    ) -> QPointF:
        if arena.geometry == "circular":
            scale = arena_rect.width() / max(arena.dims[0], 1e-9)
        else:
            scale = arena_rect.width() / max(arena.dims[0], 1e-9)
        scene_x = arena_rect.center().x() + x * scale
        scene_y = arena_rect.center().y() - y * scale
        return QPointF(scene_x, scene_y)

    def _scale_value(
        self, value: float, arena_rect: QRectF, arena: CanvasArena
    ) -> float:
        if arena.geometry == "circular":
            scale = arena_rect.width() / max(arena.dims[0], 1e-9)
        else:
            scale = arena_rect.width() / max(arena.dims[0], 1e-9)
        return float(value) * scale

    def _draw_arena(
        self, arena: CanvasArena, arena_rect: QRectF, show_outline: bool
    ) -> None:
        pen = QPen(QColor("#6b7280"))
        pen.setWidth(2 if show_outline else 0)
        if arena.geometry == "circular":
            item = self.scene.addEllipse(arena_rect, pen, QBrush(QColor("#ffffff")))
        else:
            item = self.scene.addRect(arena_rect, pen, QBrush(QColor("#ffffff")))
        item.setZValue(-10)

    def _draw_food_grid(
        self, food_grid: dict[str, Any] | None, arena_rect: QRectF
    ) -> None:
        if not isinstance(food_grid, dict):
            return
        grid_dims = food_grid.get("grid_dims")
        if not isinstance(grid_dims, (list, tuple)) or len(grid_dims) < 2:
            return
        color = QColor(str(food_grid.get("color") or "#c8e6c9"))
        overlay = self.scene.addRect(
            arena_rect,
            QPen(Qt.PenStyle.NoPen),
            QBrush(color, Qt.BrushStyle.Dense7Pattern),
        )
        overlay.setOpacity(0.12)
        overlay.setZValue(-5)
        label = self.scene.addSimpleText(
            f"food grid {int(grid_dims[0])}x{int(grid_dims[1])}"
        )
        label.setBrush(QBrush(QColor("#374151")))
        label.setPos(arena_rect.left() + 8, arena_rect.top() + 6)
        label.setZValue(5)

    def _draw_scape_labels(
        self, state: EnvironmentCanvasState, arena_rect: QRectF
    ) -> None:
        labels = []
        if state.odorscape:
            labels.append("odorscape")
        if state.windscape:
            labels.append("windscape")
        if state.thermoscape:
            labels.append("thermoscape")
        if not labels:
            return
        label = self.scene.addSimpleText(" / ".join(labels))
        label.setBrush(QBrush(QColor("#6b7280")))
        label.setPos(arena_rect.left() + 8, arena_rect.top() + 24)
        label.setZValue(5)

    def _draw_object(self, obj: CanvasObject, arena_rect: QRectF) -> None:
        arena = self.controller.canvas_state.arena
        selected = obj.object_id == self.controller.selected_object_id
        highlight_pen = QPen(QColor("#f97316" if selected else "#111827"))
        highlight_pen.setWidth(3 if selected else 1)
        if obj.object_type == "source_unit":
            self._draw_source_unit(obj, arena_rect, arena, highlight_pen)
        elif obj.object_type == "source_group":
            self._draw_source_group(obj, arena_rect, arena, highlight_pen)
        elif obj.object_type == "border_segment":
            self._draw_border(obj, arena_rect, arena, highlight_pen)

    def _draw_source_unit(
        self,
        obj: CanvasObject,
        arena_rect: QRectF,
        arena: CanvasArena,
        pen: QPen,
    ) -> None:
        if obj.x is None or obj.y is None:
            return
        center = self._world_to_scene(obj.x, obj.y, arena_rect, arena)
        radius = self._scale_value(obj.radius or 0.003, arena_rect, arena)
        item = self.scene.addEllipse(
            center.x() - radius,
            center.y() - radius,
            radius * 2.0,
            radius * 2.0,
            pen,
            QBrush(QColor(obj.color or "#4caf50")),
        )
        item.setData(0, obj.object_id)
        item.setData(1, obj.object_type)
        item.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
            self.controller.interaction_mode == "move",
        )
        item.setZValue(2)

    def _draw_source_group(
        self,
        obj: CanvasObject,
        arena_rect: QRectF,
        arena: CanvasArena,
        pen: QPen,
    ) -> None:
        if obj.x is None or obj.y is None:
            return
        center = self._world_to_scene(obj.x, obj.y, arena_rect, arena)
        scale_x = self._scale_value(
            obj.distribution_scale_x or 0.012, arena_rect, arena
        )
        scale_y = self._scale_value(
            obj.distribution_scale_y or 0.012, arena_rect, arena
        )
        shape = (obj.distribution_shape or "circle").strip().lower()
        if shape == "rect":
            rect = QRectF(
                center.x() - scale_x, center.y() - scale_y, scale_x * 2.0, scale_y * 2.0
            )
            item = self.scene.addRect(rect, pen, QBrush(QColor(obj.color or "#6688aa")))
        else:
            if shape == "circle":
                scale_y = scale_x
            rect = QRectF(
                center.x() - scale_x, center.y() - scale_y, scale_x * 2.0, scale_y * 2.0
            )
            item = self.scene.addEllipse(
                rect, pen, QBrush(QColor(obj.color or "#6688aa"))
            )
        item.setData(0, obj.object_id)
        item.setData(1, obj.object_type)
        item.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
            self.controller.interaction_mode == "move",
        )
        item.setZValue(1)
        if obj.distribution_show_shape is False:
            dot = self.scene.addEllipse(
                center.x() - 3,
                center.y() - 3,
                6,
                6,
                QPen(QColor("#111827")),
                QBrush(QColor("#111827")),
            )
            dot.setData(0, obj.object_id)
            dot.setData(1, obj.object_type)
            dot.setZValue(3)

    def _draw_border(
        self,
        obj: CanvasObject,
        arena_rect: QRectF,
        arena: CanvasArena,
        pen: QPen,
    ) -> None:
        if obj.x is None or obj.y is None or obj.x2 is None or obj.y2 is None:
            return
        p1 = self._world_to_scene(obj.x, obj.y, arena_rect, arena)
        p2 = self._world_to_scene(obj.x2, obj.y2, arena_rect, arena)
        pen.setWidthF(
            max(1.0, self._scale_value(obj.width or 0.001, arena_rect, arena) * 18.0)
        )
        line = self.scene.addLine(p1.x(), p1.y(), p2.x(), p2.y(), pen)
        line.setData(0, obj.object_id)
        line.setData(1, obj.object_type)
        line.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable,
            self.controller.interaction_mode == "move",
        )
        line.setZValue(0)

    def _highlight_selected(self, object_id: str) -> None:
        for item in self.scene.items():
            try:
                if item.data(0) != object_id:
                    continue
            except Exception:
                continue
            try:
                pen = item.pen()
                pen.setColor(QColor("#f97316"))
                pen.setWidth(max(3, pen.width()))
                item.setPen(pen)
            except Exception:
                continue
