from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QHBoxLayout, QSplitter, QVBoxLayout, QWidget

from larvaworld.gui_v2.apps.models_environments.env_builder_canvas_qt import (
    EnvBuilderCanvasWidget,
)
from larvaworld.gui_v2.apps.models_environments.env_builder_controller import (
    EnvBuilderController,
)
from larvaworld.gui_v2.apps.models_environments.env_builder_controls_qt import (
    EnvBuilderInspectorPanel,
    EnvBuilderLeftPanel,
)
from larvaworld.gui_v2.gui_aux.qt_status import EnvBuilderStatusBar


class EnvBuilderWidget(QWidget):
    def __init__(self, entry: Any | None = None) -> None:
        super().__init__()
        self.entry = entry
        self.controller = EnvBuilderController()

        self.left_panel = EnvBuilderLeftPanel(self.controller)
        self.canvas_panel = EnvBuilderCanvasWidget(self.controller)
        self.inspector_panel = EnvBuilderInspectorPanel(self.controller)
        self.status_bar = EnvBuilderStatusBar()

        self.controller.add_listener(self.refresh_from_controller)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        self.splitter = QSplitter()
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.canvas_panel)
        self.splitter.addWidget(self.inspector_panel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        root_layout.addWidget(self.splitter, 1)
        root_layout.addWidget(self.status_bar, 0)

        self.left_panel.setMinimumWidth(280)
        self.left_panel.setMaximumWidth(360)
        self.inspector_panel.setMinimumWidth(320)
        self.inspector_panel.setMaximumWidth(420)
        self.canvas_panel.setMinimumWidth(700)

        self.refresh_from_controller()

    def refresh_from_controller(self) -> None:
        self.left_panel.refresh_from_controller()
        self.inspector_panel.refresh_from_controller()
        self.status_bar.set_message(
            self.controller.status_message, dirty=self.controller.dirty
        )


def build_environment_builder_widget(entry: Any | None = None) -> EnvBuilderWidget:
    return EnvBuilderWidget(entry=entry)


__all__ = ["EnvBuilderWidget", "build_environment_builder_widget"]
