from __future__ import annotations

from typing import Any

import panel as pn
import param

from larvaworld.lib import reg, util

from .widget_base import collapsible_family_box, family_box, parameterized_editor

__all__ = ["build_trials_widget"]


def _coerce_epoch_like(value: Any) -> param.Parameterized:
    if isinstance(value, param.Parameterized):
        return value
    if isinstance(value, dict):
        return reg.gen.Epoch(**dict(value))
    return reg.gen.Epoch()


def _coerce_epochs_container(raw_epochs: Any) -> list[param.Parameterized]:
    if raw_epochs is None:
        return []
    if isinstance(raw_epochs, dict):
        values = list(raw_epochs.values())
    else:
        values = list(raw_epochs)
    return [_coerce_epoch_like(value) for value in values]


def _serialize_epochs(raw_epochs: Any, epochs: list[param.Parameterized]) -> Any:
    serialized = [util.AttrDict(epoch.nestedConf) for epoch in epochs]
    if isinstance(raw_epochs, util.ItemList):
        return util.ItemList(serialized)
    if isinstance(raw_epochs, tuple):
        return tuple(serialized)
    return serialized


def _epoch_label(index: int) -> str:
    return f"Epoch {index + 1}"


def _editable_epoch_names(epoch: param.Parameterized) -> list[str]:
    return [name for name in ("age_range", "substrate") if name in epoch.param]


def build_trials_widget(
    owner: param.Parameterized,
    *,
    wrap: bool = True,
) -> object:
    state = {"syncing": False}
    current_trials = util.AttrDict(getattr(owner, "trials", {}) or {})
    raw_epochs = current_trials.get("epochs")
    epochs = _coerce_epochs_container(raw_epochs)

    select = pn.widgets.Select(
        name="Selected epoch",
        options={"No epochs yet": ""},
        value="",
        sizing_mode="stretch_width",
    )
    add_button = pn.widgets.Button(name="Add epoch", button_type="primary", width=130)
    delete_button = pn.widgets.Button(
        name="Delete epoch", button_type="warning", width=130
    )
    editor_host = pn.Column(sizing_mode="stretch_width", margin=0)

    watchers: list[tuple[param.Parameterized, param.parameterized.Watcher]] = []

    def _clear_watchers() -> None:
        for epoch, watcher in watchers:
            try:
                epoch.param.unwatch(watcher)
            except Exception:
                pass
        watchers.clear()

    def _write_trials() -> None:
        updated_trials = util.AttrDict(getattr(owner, "trials", {}) or {})
        updated_raw_epochs = updated_trials.get("epochs")
        updated_trials["epochs"] = _serialize_epochs(updated_raw_epochs, epochs)
        state["syncing"] = True
        try:
            setattr(owner, "trials", updated_trials)
        finally:
            state["syncing"] = False

    def _watch_epoch(epoch: param.Parameterized) -> None:
        for name in _editable_epoch_names(epoch):
            watcher = epoch.param.watch(lambda *_: _write_trials(), name)
            watchers.append((epoch, watcher))

    def _refresh() -> None:
        _clear_watchers()
        if epochs:
            options = {_epoch_label(i): str(i) for i in range(len(epochs))}
            select.options = options
            if select.value not in options.values():
                select.value = "0"
            select.disabled = False
            delete_button.disabled = False
            current_epoch = epochs[int(select.value)]
            _watch_epoch(current_epoch)
            editor_host.objects = [
                family_box(
                    _epoch_label(int(select.value)),
                    parameterized_editor(
                        current_epoch,
                        parameter_order=_editable_epoch_names(current_epoch),
                    ),
                    css_classes=[
                        "lw-import-datasets-config-subfamily-card",
                        "lw-import-datasets-config-compact-card",
                    ],
                )
            ]
        else:
            select.options = {"No epochs yet": ""}
            select.value = ""
            select.disabled = True
            delete_button.disabled = True
            editor_host.objects = [
                pn.pane.HTML(
                    '<div class="lw-import-datasets-config-help">No epochs configured.</div>',
                    margin=0,
                )
            ]

    def _handle_add(_event: Any) -> None:
        epochs.append(reg.gen.Epoch())
        _write_trials()
        select.value = str(len(epochs) - 1)
        _refresh()

    def _handle_delete(_event: Any) -> None:
        if select.value in {"", None}:
            return
        index = int(select.value)
        if index < 0 or index >= len(epochs):
            return
        epochs.pop(index)
        _write_trials()
        _refresh()

    def _handle_owner_trials_change(*_events: Any) -> None:
        if state["syncing"]:
            return
        latest_trials = util.AttrDict(getattr(owner, "trials", {}) or {})
        latest_epochs = _coerce_epochs_container(latest_trials.get("epochs"))
        epochs.clear()
        epochs.extend(latest_epochs)
        _refresh()

    select.param.watch(lambda *_: _refresh(), "value")
    add_button.on_click(_handle_add)
    delete_button.on_click(_handle_delete)
    owner.param.watch(_handle_owner_trials_change, "trials")
    _refresh()

    content = pn.Column(
        select,
        pn.Row(
            add_button,
            delete_button,
            align="end",
            css_classes=["lw-import-datasets-inline-action-row"],
            sizing_mode="stretch_width",
            margin=0,
        ),
        editor_host,
        sizing_mode="stretch_width",
        margin=0,
    )
    if not wrap:
        return content
    return collapsible_family_box("Trials", content)
