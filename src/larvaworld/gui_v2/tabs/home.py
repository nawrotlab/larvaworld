from __future__ import annotations

from collections.abc import Callable

from larvaworld.gui_v2.registry_bridge import GuiNavigationModel
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def build_home_text(
    model: GuiNavigationModel,
) -> tuple[str, str, list[tuple[str, str]]]:
    total_entries = sum(len(lane.entries) for lane in model.lanes)
    quick_start_modes = len(model.quick_start_modes)
    ready_entries = sum(
        1 for entry in model.entry_index.values() if entry.status == "ready"
    )
    planned_entries = sum(
        1 for entry in model.entry_index.values() if entry.status == "planned"
    )

    blocks = [
        (
            "M1 shell objective",
            "This gui_v2 scaffold establishes a desktop-native shell aligned with the portal "
            "information architecture while keeping all navigation and content inside the GUI window.",
        ),
        (
            "Shared metadata baseline",
            f"The shell currently reads {total_entries} lane entries and {quick_start_modes} quick-start "
            f"modes from the shared portal registry layer. Ready items: {ready_entries}. "
            f"Under-construction items: {planned_entries}.",
        ),
        (
            "Next implementation pass",
            "Replace metadata-only content pages with embedded local app surfaces, shell-native views, "
            "or compatibility wrappers, depending on the per-app render strategy.",
        ),
    ]

    return (
        "gui_v2 desktop shell",
        "Initial desktop GUI shell aligned with the Larvaworld portal structure.",
        blocks,
    )


LANE_COLORS: dict[str, str] = {
    "quick_start": "#f5a142",
    "simulate": "#b5c2b0",
    "data": "#b0b4c2",
    "models": "#c1b0c2",
}


class GuiHomeView(QWidget):
    def __init__(
        self,
        model: GuiNavigationModel,
        *,
        on_entry_selected: Callable[[str], None],
    ) -> None:
        super().__init__()
        self.model = model
        self.on_entry_selected = on_entry_selected
        self.active_mode_id = model.quick_start_modes[0].mode_id
        self.mode_buttons: dict[str, QPushButton] = {}
        self.quick_start_cards_layout: QGridLayout | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(self._build_quick_start_section())

        for lane in self.model.lanes:
            layout.addWidget(
                self._build_lane_section(lane.title, lane.lane_id, list(lane.entries))
            )

        layout.addStretch(1)

    def _build_quick_start_section(self) -> QWidget:
        section = QFrame()
        section.setStyleSheet(
            "background:#f3f4f6;border:1px solid #d1d5db;border-left:4px solid #f5a142;border-radius:12px;"
        )
        outer = QVBoxLayout(section)
        outer.setContentsMargins(14, 12, 14, 14)
        outer.setSpacing(12)

        title = QLabel("Quick Start")
        title.setStyleSheet(
            "color:#111827; font-size:18px; font-weight:700; border:none;"
        )
        outer.addWidget(title)

        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(10)
        for mode in self.model.quick_start_modes:
            button = QPushButton(mode.title.replace(" mode", ""))
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(
                lambda _checked=False, mode_id=mode.mode_id: self._set_mode(mode_id)
            )
            self.mode_buttons[mode.mode_id] = button
            mode_row.addWidget(button)
        mode_row.addStretch(1)
        outer.addLayout(mode_row)

        cards_host = QWidget()
        self.quick_start_cards_layout = QGridLayout(cards_host)
        self.quick_start_cards_layout.setContentsMargins(0, 0, 0, 0)
        self.quick_start_cards_layout.setHorizontalSpacing(14)
        self.quick_start_cards_layout.setVerticalSpacing(14)
        outer.addWidget(cards_host)

        self._set_mode(self.active_mode_id)
        return section

    def _build_lane_section(self, title: str, lane_key: str, entries: list) -> QWidget:
        section = QFrame()
        color = LANE_COLORS.get(lane_key, "#d1d5db")
        section.setStyleSheet(
            f"background:#ffffff;border:1px solid #d1d5db;border-top:3px solid {color};border-radius:12px;"
        )
        outer = QVBoxLayout(section)
        outer.setContentsMargins(14, 12, 14, 14)
        outer.setSpacing(12)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            "color:#111827; font-size:18px; font-weight:700; border:none;"
        )
        outer.addWidget(title_label)

        grid_host = QWidget()
        grid = QGridLayout(grid_host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)

        for index, entry in enumerate(entries):
            grid.addWidget(
                self._build_entry_card(entry, accent=color), index // 3, index % 3
            )

        outer.addWidget(grid_host)
        return section

    def _set_mode(self, mode_id: str) -> None:
        self.active_mode_id = mode_id
        for current_mode_id, button in self.mode_buttons.items():
            mode = next(
                mode
                for mode in self.model.quick_start_modes
                if mode.mode_id == current_mode_id
            )
            if current_mode_id == mode_id:
                button.setStyleSheet(
                    f"background:{mode.color}; color:#111827; border:1px solid #9ca3af; border-radius:8px; padding:6px 14px; font-weight:600;"
                )
            else:
                button.setStyleSheet(
                    "background:#ffffff; color:#374151; border:1px solid #d1d5db; border-radius:8px; padding:6px 14px;"
                )

        if self.quick_start_cards_layout is None:
            return

        while self.quick_start_cards_layout.count():
            item = self.quick_start_cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        mode = next(
            mode for mode in self.model.quick_start_modes if mode.mode_id == mode_id
        )
        for index, entry in enumerate(mode.entries):
            self.quick_start_cards_layout.addWidget(
                self._build_entry_card(entry, accent=mode.color),
                index // 3,
                index % 3,
            )

    def _build_entry_card(self, entry, *, accent: str) -> QWidget:
        card = QFrame()
        card.setStyleSheet(
            f"background:#ffffff;border:1px solid #d1d5db;border-top:3px solid {accent};border-radius:12px;"
        )
        outer = QVBoxLayout(card)
        outer.setContentsMargins(14, 12, 14, 14)
        outer.setSpacing(10)

        badges_row = QHBoxLayout()
        badges_row.setContentsMargins(0, 0, 0, 0)
        badges_row.setSpacing(6)
        for badge in entry.badges:
            badge_label = QLabel(badge)
            badge_label.setStyleSheet(
                "background:#f3f4f6;color:#374151;border:1px solid #d1d5db;border-radius:10px;padding:2px 8px;font-size:11px;"
            )
            badges_row.addWidget(badge_label)
        badges_row.addStretch(1)
        outer.addLayout(badges_row)

        title = QLabel(entry.title)
        title.setStyleSheet("color:#111827;font-size:16px;font-weight:700;border:none;")
        title.setWordWrap(True)
        outer.addWidget(title)

        subtitle = QLabel(entry.subtitle.replace("\n", " "))
        subtitle.setStyleSheet(
            "color:#4b5563;font-size:11px;line-height:1.3;border:none;"
        )
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle, 1)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)

        open_button = QPushButton("Open view")
        open_button.setCursor(Qt.PointingHandCursor)
        open_button.setStyleSheet(
            f"background:{accent}; color:#111827; border:1px solid #9ca3af; border-radius:8px; padding:6px 12px; font-weight:600;"
        )
        open_button.clicked.connect(
            lambda _checked=False, entry_id=entry.entry_id: self.on_entry_selected(
                entry_id
            )
        )
        action_row.addWidget(open_button)
        action_row.addStretch(1)
        outer.addLayout(action_row)

        return card


def build_home_widget(
    model: GuiNavigationModel,
    *,
    on_entry_selected: Callable[[str], None],
) -> GuiHomeView:
    return GuiHomeView(model, on_entry_selected=on_entry_selected)
