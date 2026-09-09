from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QWidget,
)


def labeled_line_edit(
    label: str, *, text: str = "", on_change: Callable[[str], None] | None = None
) -> tuple[QWidget, QLineEdit]:
    widget = QLineEdit()
    widget.setText(text)
    if on_change is not None:
        widget.editingFinished.connect(lambda: on_change(widget.text()))
    form = QFormLayout()
    container = QWidget()
    container.setLayout(form)
    form.setContentsMargins(0, 0, 0, 0)
    form.addRow(label, widget)
    return container, widget


def labeled_double_spinbox(
    label: str,
    *,
    value: float = 0.0,
    minimum: float = -1_000_000.0,
    maximum: float = 1_000_000.0,
    step: float = 0.001,
    decimals: int = 4,
    on_change: Callable[[float], None] | None = None,
) -> tuple[QWidget, QDoubleSpinBox]:
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setSingleStep(step)
    widget.setValue(value)
    if on_change is not None:
        widget.valueChanged.connect(on_change)
    form = QFormLayout()
    container = QWidget()
    container.setLayout(form)
    form.setContentsMargins(0, 0, 0, 0)
    form.addRow(label, widget)
    return container, widget


def labeled_spinbox(
    label: str,
    *,
    value: int = 0,
    minimum: int = -1_000_000,
    maximum: int = 1_000_000,
    step: int = 1,
    on_change: Callable[[int], None] | None = None,
) -> tuple[QWidget, QSpinBox]:
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setSingleStep(step)
    widget.setValue(value)
    if on_change is not None:
        widget.valueChanged.connect(on_change)
    form = QFormLayout()
    container = QWidget()
    container.setLayout(form)
    form.setContentsMargins(0, 0, 0, 0)
    form.addRow(label, widget)
    return container, widget


def labeled_checkbox(
    label: str,
    *,
    checked: bool = False,
    on_change: Callable[[bool], None] | None = None,
) -> tuple[QWidget, QCheckBox]:
    widget = QCheckBox()
    widget.setChecked(checked)
    if on_change is not None:
        widget.toggled.connect(on_change)
    form = QFormLayout()
    container = QWidget()
    container.setLayout(form)
    form.setContentsMargins(0, 0, 0, 0)
    form.addRow(label, widget)
    return container, widget


def labeled_combo_box(
    label: str,
    *,
    items: list[str],
    current: str | None = None,
    on_change: Callable[[str], None] | None = None,
) -> tuple[QWidget, QComboBox]:
    widget = QComboBox()
    widget.addItems(items)
    if current in items:
        widget.setCurrentText(current)
    if on_change is not None:
        widget.currentTextChanged.connect(on_change)
    form = QFormLayout()
    container = QWidget()
    container.setLayout(form)
    form.setContentsMargins(0, 0, 0, 0)
    form.addRow(label, widget)
    return container, widget
