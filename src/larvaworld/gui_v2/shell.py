from __future__ import annotations

import importlib.metadata as im

from larvaworld.gui_v2.registry_bridge import (
    GuiEntry,
    GuiNavigationModel,
    build_navigation_model,
)
from larvaworld.gui_v2.tabs.home import build_home_text, build_home_widget
from larvaworld.gui_v2.tabs.placeholder import (
    build_entry_text,
    build_entry_widget,
    build_group_text,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class LarvaworldGuiV2Shell:
    def __init__(self, *, geometry: str = "1360x860") -> None:
        self.model = build_navigation_model()
        self.app = QApplication.instance() or QApplication([])
        self.window = QMainWindow()
        self.window.setWindowTitle("Larvaworld gui_v2")
        width, height = self._parse_geometry(geometry)
        self.window.resize(width, height)
        self.window.setMinimumSize(1100, 760)

        self._tree_payload: dict[str, tuple[str, str | GuiEntry | tuple[str, str]]] = {}
        self._block_widgets: list[QWidget] = []
        self._nav_collapsed = False
        self._nav_expanded_width = 320

        self._build_layout()
        self._populate_tree()
        self._select_default_view()

    def run(self) -> None:
        self.window.show()
        self.app.exec()

    def _build_layout(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root.setStyleSheet("background:#f3f4f6;")
        self.window.setCentralWidget(root)

        topbar = QFrame()
        topbar.setFixedHeight(56)
        topbar.setStyleSheet("background:#1f2937;")
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setContentsMargins(18, 0, 18, 0)
        topbar_layout.setSpacing(10)
        root_layout.addWidget(topbar)

        self.nav_toggle_button = QPushButton("◀")
        self.nav_toggle_button.setFixedSize(34, 34)
        self.nav_toggle_button.setCursor(Qt.PointingHandCursor)
        self.nav_toggle_button.setStyleSheet(
            "background:#111827;color:#f9fafb;border:1px solid #4b5563;border-radius:8px;font-size:16px;font-weight:700;"
        )
        self.nav_toggle_button.clicked.connect(self._toggle_navigation)
        topbar_layout.addWidget(self.nav_toggle_button)

        version = self._resolve_version()
        title = QLabel(f"Larvaworld gui_v2  ·  v{version}")
        title.setStyleSheet("color:#f9fafb;")
        title.setFont(self._font(16, True))
        topbar_layout.addWidget(title)

        subtitle = QLabel("M1 desktop shell scaffold")
        subtitle.setStyleSheet("color:#cbd5e1;")
        subtitle.setFont(self._font(10, False))
        topbar_layout.addWidget(subtitle)
        topbar_layout.addStretch(1)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(6)
        root_layout.addWidget(self.main_splitter, 1)

        self.nav_frame = QFrame()
        self.nav_frame.setMinimumWidth(280)
        self.nav_frame.setMaximumWidth(380)
        self.nav_frame.setStyleSheet("background:#e5e7eb;")
        nav_layout = QVBoxLayout(self.nav_frame)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)

        nav_header = QLabel("Navigation")
        nav_header.setStyleSheet("background:#d1d5db;color:#111827;padding:12px;")
        nav_header.setFont(self._font(13, True))
        nav_layout.addWidget(nav_header)

        self.nav_tree = QTreeWidget()
        self.nav_tree.setHeaderHidden(True)
        self.nav_tree.setIndentation(18)
        self.nav_tree.setStyleSheet(
            """
            QTreeWidget {
                background:#e5e7eb;
                border:none;
                padding:8px;
                color:#111827;
                font-size:11px;
            }
            QTreeWidget::item {
                height:24px;
            }
            QTreeWidget::item:selected {
                background:#d1d5db;
                color:#111827;
            }
            """
        )
        self.nav_tree.itemSelectionChanged.connect(self._on_tree_select)
        nav_layout.addWidget(self.nav_tree, 1)

        content_frame = QWidget()
        content_frame.setStyleSheet("background:#ffffff;")
        content_layout = QVBoxLayout(content_frame)
        content_layout.setContentsMargins(14, 14, 14, 14)
        content_layout.setSpacing(8)

        self.content_header = QFrame()
        self.content_header.setMinimumHeight(112)
        self.content_header.setStyleSheet("background:#ede9f0;border-radius:10px;")
        header_layout = QVBoxLayout(self.content_header)
        header_layout.setContentsMargins(16, 14, 16, 14)
        header_layout.setSpacing(6)
        content_layout.addWidget(self.content_header)

        self.content_title = QLabel("")
        self.content_title.setStyleSheet("color:#111827;")
        self.content_title.setFont(self._font(20, True))
        header_layout.addWidget(self.content_title)

        self.content_subtitle = QLabel("")
        self.content_subtitle.setStyleSheet("color:#374151;")
        self.content_subtitle.setFont(self._font(11, False))
        self.content_subtitle.setWordWrap(True)
        header_layout.addWidget(self.content_subtitle)

        self.content_scroll = QScrollArea()
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setFrameShape(QFrame.NoFrame)
        self.content_scroll.setStyleSheet("background:#ffffff;")
        content_layout.addWidget(self.content_scroll, 1)

        self.content_body = QWidget()
        self.content_body_layout = QVBoxLayout(self.content_body)
        self.content_body_layout.setContentsMargins(0, 0, 0, 0)
        self.content_body_layout.setSpacing(12)
        self.content_body_layout.addStretch(1)
        self.content_scroll.setWidget(self.content_body)

        self.main_splitter.addWidget(self.nav_frame)
        self.main_splitter.addWidget(content_frame)
        self.main_splitter.setSizes([self._nav_expanded_width, 980])

        footer = QLabel(
            "Desktop-only shell · Shared portal metadata bridge · No external browser flow"
        )
        footer.setStyleSheet("background:#111827;color:#e5e7eb;padding:6px 12px;")
        footer.setFont(self._font(9, False))
        root_layout.addWidget(footer)

    def _populate_tree(self) -> None:
        self._tree_payload["home"] = ("home", None)
        home_item = QTreeWidgetItem(["Home"])
        home_item.setData(0, Qt.UserRole, "home")
        self.nav_tree.addTopLevelItem(home_item)

        quick_start_id = "group:quick_start"
        self._tree_payload[quick_start_id] = (
            "group",
            ("Quick Start", "Portal-aligned quick-start modes and priority workflows."),
        )
        quick_start_item = QTreeWidgetItem(["Quick Start"])
        quick_start_item.setData(0, Qt.UserRole, quick_start_id)
        self.nav_tree.addTopLevelItem(quick_start_item)
        quick_start_item.setExpanded(True)

        for mode in self.model.quick_start_modes:
            mode_id = f"mode:{mode.mode_id}"
            self._tree_payload[mode_id] = (
                "group",
                (
                    mode.title,
                    f"Mode color {mode.color} · Contains {len(mode.entries)} quick-start workflows.",
                ),
            )
            mode_item = QTreeWidgetItem([mode.title])
            mode_item.setData(0, Qt.UserRole, mode_id)
            quick_start_item.addChild(mode_item)
            mode_item.setExpanded(True)
            for entry in mode.entries:
                item_id = f"entry:{mode.mode_id}:{entry.entry_id}"
                self._tree_payload[item_id] = ("entry", entry)
                entry_item = QTreeWidgetItem([entry.title])
                entry_item.setData(0, Qt.UserRole, item_id)
                mode_item.addChild(entry_item)

        for lane in self.model.lanes:
            lane_id = f"lane:{lane.lane_id}"
            self._tree_payload[lane_id] = (
                "group",
                (
                    lane.title,
                    f"{len(lane.entries)} workflows in the {lane.title} group.",
                ),
            )
            lane_item = QTreeWidgetItem([lane.title])
            lane_item.setData(0, Qt.UserRole, lane_id)
            self.nav_tree.addTopLevelItem(lane_item)
            lane_item.setExpanded(True)
            for entry in lane.entries:
                item_id = f"entry:{lane.lane_id}:{entry.entry_id}"
                self._tree_payload[item_id] = ("entry", entry)
                entry_item = QTreeWidgetItem([entry.title])
                entry_item.setData(0, Qt.UserRole, item_id)
                lane_item.addChild(entry_item)

    def _select_default_view(self) -> None:
        home_item = self.nav_tree.topLevelItem(0)
        self.nav_tree.setCurrentItem(home_item)
        self._show_home_view()

    def _on_tree_select(self) -> None:
        item = self.nav_tree.currentItem()
        if item is None:
            return

        node_id = item.data(0, Qt.UserRole)
        payload = self._tree_payload.get(node_id)
        if payload is None:
            return

        payload_kind, payload_value = payload
        if payload_kind == "home":
            self._show_home_view()
        elif payload_kind == "group":
            title, body = payload_value
            self._show_group_view(title, body)
        else:
            self._show_entry_view(payload_value)

    def _show_home_view(self) -> None:
        title, subtitle, _blocks = build_home_text(self.model)
        self._set_header(title, subtitle)
        self._clear_body()
        self._add_widget(
            build_home_widget(self.model, on_entry_selected=self._show_entry_by_id)
        )

    def _show_group_view(self, title: str, description: str) -> None:
        block_title, subtitle, blocks = build_group_text(title, description, self.model)
        self._set_header(block_title, subtitle)
        self._clear_body()

        for inner_title, inner_body in blocks:
            self._add_block(inner_title, inner_body)

    def _show_entry_view(self, entry: GuiEntry) -> None:
        title, subtitle, blocks = build_entry_text(entry)
        self._set_header(title, subtitle)
        self._clear_body()

        widget = build_entry_widget(entry)
        if widget is not None:
            self._add_widget(widget)
            return

        for block_title, block_body in blocks:
            self._add_block(block_title, block_body)

    def _set_header(self, title: str, subtitle: str) -> None:
        self.content_title.setText(title)
        self.content_subtitle.setText(subtitle)

    def _clear_body(self) -> None:
        while self.content_body_layout.count():
            item = self.content_body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._block_widgets.clear()
        self.content_body_layout.addStretch(1)

    def _add_block(self, title: str, body: str) -> None:
        stretch = self.content_body_layout.takeAt(self.content_body_layout.count() - 1)

        outer = QFrame()
        outer.setFrameShape(QFrame.StyledPanel)
        outer.setStyleSheet(
            "background:#ffffff;border:1px solid #d1d5db;border-radius:8px;"
        )
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        header = QLabel(title)
        header.setStyleSheet(
            "background:#f3f4f6;color:#111827;padding:8px 12px;border-top-left-radius:8px;border-top-right-radius:8px;"
        )
        header.setFont(self._font(12, True))
        outer_layout.addWidget(header)

        content = QLabel(body)
        content.setStyleSheet("background:#ffffff;color:#374151;padding:12px;")
        content.setFont(self._font(11, False))
        content.setWordWrap(True)
        content.setTextFormat(Qt.PlainText)
        outer_layout.addWidget(content)

        self.content_body_layout.addWidget(outer)
        self._block_widgets.append(outer)
        if stretch is not None:
            self.content_body_layout.addItem(stretch)

    def _add_widget(self, widget: QWidget) -> None:
        stretch = self.content_body_layout.takeAt(self.content_body_layout.count() - 1)
        self.content_body_layout.addWidget(widget)
        self._block_widgets.append(widget)
        if stretch is not None:
            self.content_body_layout.addItem(stretch)

    def _show_entry_by_id(self, entry_id: str) -> None:
        entry = self.model.entry_index.get(entry_id)
        if entry is None:
            return
        self._show_entry_view(entry)

    def _toggle_navigation(self) -> None:
        if self._nav_collapsed:
            self.nav_frame.setMinimumWidth(280)
            self.nav_frame.setMaximumWidth(380)
            self.main_splitter.setSizes([self._nav_expanded_width, 1000])
            self.nav_toggle_button.setText("◀")
            self._nav_collapsed = False
            return

        current_width = self.main_splitter.sizes()[0]
        if current_width > 0:
            self._nav_expanded_width = current_width
        self.nav_frame.setMinimumWidth(0)
        self.nav_frame.setMaximumWidth(0)
        self.main_splitter.setSizes([0, 1280])
        self.nav_toggle_button.setText("▶")
        self._nav_collapsed = True

    @staticmethod
    def _resolve_version() -> str:
        try:
            return im.version("larvaworld")
        except Exception:
            return "dev"

    @staticmethod
    def _parse_geometry(geometry: str) -> tuple[int, int]:
        try:
            width_str, height_str = geometry.lower().split("x", maxsplit=1)
            return max(1100, int(width_str)), max(760, int(height_str))
        except Exception:
            return 1360, 860

    @staticmethod
    def _font(size: int, bold: bool) -> QFont:
        font = QFont("Helvetica", size)
        font.setBold(bold)
        return font
