from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget


class EnvBuilderStatusBar(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._message = QLabel("Ready.")
        self._dirty = QLabel("Clean")
        self._dirty.setStyleSheet(
            "color:#065f46;background:#d1fae5;border-radius:4px;padding:2px 6px;"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(10)
        frame = QFrame()
        frame.setStyleSheet("background:#111827;color:#e5e7eb;border-radius:8px;")
        inner = QHBoxLayout(frame)
        inner.setContentsMargins(10, 8, 10, 8)
        inner.addWidget(self._message, 1)
        inner.addWidget(self._dirty, 0)
        layout.addWidget(frame, 1)

    def set_message(self, message: str, *, dirty: bool | None = None) -> None:
        self._message.setText(message)
        if dirty is None:
            return
        if dirty:
            self._dirty.setText("Dirty")
            self._dirty.setStyleSheet(
                "color:#7f1d1d;background:#fee2e2;border-radius:4px;padding:2px 6px;"
            )
        else:
            self._dirty.setText("Clean")
            self._dirty.setStyleSheet(
                "color:#065f46;background:#d1fae5;border-radius:4px;padding:2px 6px;"
            )
