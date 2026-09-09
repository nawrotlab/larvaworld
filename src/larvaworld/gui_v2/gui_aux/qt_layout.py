from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


def section_card(title: str) -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setStyleSheet(
        "background:#ffffff;border:1px solid #d1d5db;border-radius:8px;"
    )
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(8)
    header = QLabel(title)
    header.setStyleSheet("color:#111827;font-weight:700;border:none;")
    layout.addWidget(header)
    return frame, layout


def inset_frame(widget: QWidget) -> QFrame:
    frame = QFrame()
    frame.setStyleSheet(
        "background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;"
    )
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setSpacing(0)
    layout.addWidget(widget)
    return frame
