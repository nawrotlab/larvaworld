from __future__ import annotations

from collections.abc import Callable
from typing import Any

import panel as pn
import param

from larvaworld.lib import reg

__all__ = ["ConftypeActionsController", "build_conftype_actions"]


def _is_parameterized_class(config_cls: type[Any]) -> bool:
    return isinstance(config_cls, type) and issubclass(config_cls, param.Parameterized)


def _matches_registered_class(
    config_cls: type[param.Parameterized], candidate_cls: type[Any]
) -> bool:
    if not _is_parameterized_class(candidate_cls):
        return False
    if candidate_cls is config_cls:
        return True
    try:
        return issubclass(config_cls, candidate_cls) or issubclass(
            candidate_cls, config_cls
        )
    except TypeError:
        pass
    agent_class = getattr(candidate_cls, "agent_class", None)
    if callable(agent_class):
        try:
            return agent_class() == config_cls.__name__
        except Exception:
            return False
    return False


def _widget_size(
    *,
    width: int | None,
    sizing_mode: str,
) -> dict[str, Any]:
    if width is not None:
        return {"width": width}
    return {"sizing_mode": sizing_mode}


class ConftypeActionsController:
    def __init__(
        self,
        config_cls: type[param.Parameterized],
        *,
        conftype: str,
        build_save_payload: Callable[[str], Any],
        get_selected_id: Callable[[], str | None],
        get_save_id: Callable[[], str | None] | None = None,
        on_load: Callable[[str, Any], None] | None = None,
        on_save: Callable[[str, Any], None] | None = None,
        on_delete: Callable[[str], None] | None = None,
        on_reset: Callable[[str | None], None] | None = None,
        on_status: Callable[..., None] | None = None,
        allow_reset: bool = True,
        confirm_reset: bool = True,
        load_button_name: str = "Load",
        save_button_name: str = "Save",
        delete_button_name: str = "Delete",
        reset_button_name: str = "Reset configurations",
        load_button_type: str = "primary",
        save_button_type: str = "primary",
        delete_button_type: str = "warning",
        reset_button_type: str = "danger",
        button_width: int | None = None,
        sizing_mode: str = "stretch_width",
    ) -> None:
        if not _is_parameterized_class(config_cls):
            raise TypeError("config_cls must be a param.Parameterized subclass.")
        if conftype not in reg.conf:
            raise ValueError(f'Unknown conftype "{conftype}".')

        self.config_cls = config_cls
        self.conf_type = reg.conf[conftype]
        if not _matches_registered_class(config_cls, self.conf_type.conf_class):
            raise ValueError(
                f"{config_cls.__name__} does not match conftype "
                f'"{self.conf_type.conftype}".'
            )

        self.build_save_payload = build_save_payload
        self.get_selected_id = get_selected_id
        self.get_save_id = get_save_id or get_selected_id
        self.on_load = on_load
        self.on_save = on_save
        self.on_delete = on_delete
        self.on_reset = on_reset
        self.on_status = on_status
        self.confirm_reset = confirm_reset

        self.load_button = pn.widgets.Button(
            name=load_button_name,
            button_type=load_button_type,
            **_widget_size(width=button_width, sizing_mode=sizing_mode),
        )
        self.save_button = pn.widgets.Button(
            name=save_button_name,
            button_type=save_button_type,
            **_widget_size(width=button_width, sizing_mode=sizing_mode),
        )
        self.delete_button = pn.widgets.Button(
            name=delete_button_name,
            button_type=delete_button_type,
            **_widget_size(width=button_width, sizing_mode=sizing_mode),
        )
        self.reset_button = pn.widgets.Button(
            name=reset_button_name,
            button_type=reset_button_type,
            visible=allow_reset,
            **_widget_size(width=button_width, sizing_mode=sizing_mode),
        )
        self.reset_confirm_host = pn.Column(sizing_mode="stretch_width")

        self.load_button.on_click(lambda _event: self.load_selected())
        self.save_button.on_click(lambda _event: self.save_current())
        self.delete_button.on_click(lambda _event: self.delete_selected())
        self.reset_button.on_click(lambda _event: self.request_reset())

        self.view = pn.Column(
            pn.Row(
                self.load_button,
                self.save_button,
                self.delete_button,
                sizing_mode="stretch_width",
            ),
            pn.Row(
                self.reset_button,
                sizing_mode="stretch_width",
                visible=allow_reset,
            ),
            self.reset_confirm_host,
            sizing_mode="stretch_width",
        )

    def _selected_id(self) -> str | None:
        value = self.get_selected_id()
        return str(value) if value else None

    def _save_id(self) -> str | None:
        value = self.get_save_id()
        return str(value) if value else None

    def _set_status(self, message: str, *, tone: str = "neutral") -> None:
        if self.on_status is not None:
            self.on_status(message, tone=tone)

    def refresh_registry(self) -> None:
        self.conf_type.load()

    def load_selected(self) -> bool:
        config_id = self._selected_id()
        if not config_id:
            self._set_status("Select a stored configuration to load.", tone="warning")
            return False
        try:
            self.conf_type.load()
            config = self.conf_type.get(config_id)
        except Exception as exc:
            self._set_status(f"Load failed: {exc}", tone="danger")
            return False
        if self.on_load is not None:
            self.on_load(config_id, config)
        self._set_status(
            f'Loaded {self.conf_type.conftype} "{config_id}".',
            tone="success",
        )
        return True

    def save_current(self) -> bool:
        config_id = self._save_id()
        if not config_id:
            self._set_status("Enter a configuration ID before saving.", tone="warning")
            return False
        try:
            payload = self.build_save_payload(config_id)
            self.conf_type.setID(config_id, payload)
        except Exception as exc:
            self._set_status(f"Save failed: {exc}", tone="danger")
            return False
        self.refresh_registry()
        if self.on_save is not None:
            self.on_save(config_id, payload)
        self._set_status(
            f'{self.conf_type.conftype} "{config_id}" saved to the registry.',
            tone="success",
        )
        return True

    def delete_selected(self) -> bool:
        config_id = self._selected_id()
        if not config_id:
            self._set_status("Select a stored configuration to delete.", tone="warning")
            return False
        try:
            self.conf_type.delete(config_id)
        except Exception as exc:
            self._set_status(f"Delete failed: {exc}", tone="danger")
            return False
        self.refresh_registry()
        if self.on_delete is not None:
            self.on_delete(config_id)
        self._set_status(
            f'{self.conf_type.conftype} "{config_id}" deleted from the registry.',
            tone="success",
        )
        return True

    def reset_store(self) -> bool:
        try:
            self.conf_type.reset(recreate=True)
        except Exception as exc:
            self._set_status(f"Reset failed: {exc}", tone="danger")
            return False
        self.refresh_registry()
        if self.on_reset is not None:
            self.on_reset(self._selected_id())
        self._set_status(
            f"{self.conf_type.conftype} registry recreated from the built-in defaults.",
            tone="success",
        )
        return True

    def request_reset(self) -> None:
        if not self.confirm_reset:
            self.reset_store()
            return
        confirm = pn.widgets.Button(
            name="Yes, recreate store",
            button_type="danger",
            sizing_mode="stretch_width",
        )
        cancel = pn.widgets.Button(
            name="No, cancel",
            button_type="default",
            sizing_mode="stretch_width",
        )
        message = pn.pane.HTML(
            (
                '<div style="font-size:12px;color:#9a3412;">'
                f"This recreates the full {self.conf_type.conftype} registry store "
                "from its default definitions."
                "</div>"
            ),
            margin=(0, 0, 8, 0),
        )

        def _confirm(_event: Any = None) -> None:
            self.reset_store()
            self.reset_confirm_host.objects = []

        def _cancel(_event: Any = None) -> None:
            self.reset_confirm_host.objects = []
            self._set_status("Reset cancelled.")

        confirm.on_click(_confirm)
        cancel.on_click(_cancel)
        self.reset_confirm_host.objects = [
            pn.Column(
                message,
                pn.Row(confirm, cancel, sizing_mode="stretch_width"),
                sizing_mode="stretch_width",
                margin=(0, 0, 8, 0),
            )
        ]


def build_conftype_actions(
    config_cls: type[param.Parameterized],
    *,
    conftype: str,
    build_save_payload: Callable[[str], Any],
    get_selected_id: Callable[[], str | None],
    **kwargs: Any,
) -> pn.Column:
    controller = ConftypeActionsController(
        config_cls,
        conftype=conftype,
        build_save_payload=build_save_payload,
        get_selected_id=get_selected_id,
        **kwargs,
    )
    return controller.view
