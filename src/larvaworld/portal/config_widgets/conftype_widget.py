from __future__ import annotations

from typing import Any

import panel as pn
import param

from larvaworld import CONFTYPES
from larvaworld.lib import reg, util
from larvaworld.lib.param.custom import ClassAttr, ClassDict
from larvaworld.lib.param.nested_parameter_group import NestedConf

__all__ = [
    "ConftypeWidgetController",
    "build_conftype_widget",
    "resolve_conftype",
]


def _is_parameterized_class(config_cls: type[Any]) -> bool:
    return isinstance(config_cls, type) and issubclass(config_cls, param.Parameterized)


def _generated_agent_class_name(config_cls: type[Any]) -> str | None:
    agent_class = getattr(config_cls, "agent_class", None)
    if callable(agent_class):
        try:
            value = agent_class()
        except Exception:
            return None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _matches_conftype_class(
    config_cls: type[param.Parameterized], candidate_cls: type[Any]
) -> bool:
    if not _is_parameterized_class(candidate_cls):
        return False
    if candidate_cls is config_cls:
        return True
    try:
        if issubclass(config_cls, candidate_cls) or issubclass(
            candidate_cls, config_cls
        ):
            return True
    except TypeError:
        pass
    generated_name = _generated_agent_class_name(candidate_cls)
    return generated_name == config_cls.__name__


def resolve_conftype(
    config_cls: type[param.Parameterized],
    *,
    conftype: str | None = None,
):
    if not _is_parameterized_class(config_cls):
        raise TypeError(
            "config_cls must be a subclass of param.Parameterized or NestedConf."
        )

    if conftype is not None:
        if conftype not in reg.conf:
            raise ValueError(f'Unknown conftype "{conftype}".')
        conf_type = reg.conf[conftype]
        if not _matches_conftype_class(config_cls, conf_type.conf_class):
            raise ValueError(
                f"{config_cls.__name__} does not match the registered class for "
                f'conftype "{conftype}".'
            )
        return conf_type

    matches = []
    for candidate_name in CONFTYPES:
        if candidate_name not in reg.conf:
            continue
        conf_type = reg.conf[candidate_name]
        if _matches_conftype_class(config_cls, conf_type.conf_class):
            matches.append(conf_type)
    if not matches:
        raise ValueError(
            f"Could not resolve a conftype for {config_cls.__name__}. "
            "Pass conftype= explicitly."
        )
    if len(matches) > 1:
        names = ", ".join(conf_type.conftype for conf_type in matches)
        raise ValueError(
            f"Ambiguous conftype resolution for {config_cls.__name__}: {names}. "
            "Pass conftype= explicitly."
        )
    return matches[0]


def _serialize_value(value: Any) -> Any:
    if isinstance(value, param.Parameterized):
        return _serialize_parameterized(value)
    if isinstance(value, dict):
        return util.AttrDict(
            {key: _serialize_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def _serialize_parameterized(instance: param.Parameterized) -> util.AttrDict:
    if isinstance(instance, NestedConf):
        return instance.nestedConf

    payload = util.AttrDict()
    for name, parameter in instance.param.objects(instance=False).items():
        if name == "name" or parameter.readonly:
            continue
        payload[name] = _serialize_value(getattr(instance, name))
    return payload


def _instantiate_classattr(parameter: ClassAttr) -> param.Parameterized:
    nested_cls = (
        parameter.class_[0] if isinstance(parameter.class_, tuple) else parameter.class_
    )
    return nested_cls()


def _widget_overrides(
    instance: param.Parameterized, parameter_names: list[str]
) -> dict[str, Any]:
    widgets: dict[str, Any] = {}
    for name in parameter_names:
        parameter = instance.param.objects(instance=False)[name]
        if isinstance(parameter, param.ListSelector):
            widgets[name] = {"type": pn.widgets.MultiChoice}
        elif isinstance(parameter, param.Integer):
            widgets[name] = {"type": pn.widgets.IntInput}
        elif isinstance(parameter, param.Number):
            widgets[name] = {"type": pn.widgets.FloatInput}
        elif isinstance(
            parameter, (param.Range, param.NumericTuple, param.Dict, param.List)
        ):
            widgets[name] = {"type": pn.widgets.LiteralInput}
    return widgets


class ConftypeWidgetController:
    def __init__(
        self,
        config_cls: type[param.Parameterized],
        *,
        conftype: str | None = None,
        title: str | None = None,
        allow_reset: bool = True,
    ) -> None:
        self.config_cls = config_cls
        self.conf_type = resolve_conftype(config_cls, conftype=conftype)
        self.allow_reset = allow_reset
        self.title = title or f"{self.conf_type.conftype} Configurations"
        self._editor_host = pn.Column(sizing_mode="stretch_width")
        self._reset_confirm_host = pn.Column(sizing_mode="stretch_width")
        self._current = self._new_instance()

        self.header = pn.pane.Markdown(
            f"### {self.title}",
            margin=(0, 0, 6, 0),
        )
        self.meta = pn.pane.HTML(
            (
                '<div style="font-size:12px;color:#52606d;">'
                f"<strong>Conftype:</strong> {self.conf_type.conftype}"
                f' <span style="color:#8b95a1;">|</span> '
                f"<strong>Class:</strong> {self.config_cls.__name__}"
                "</div>"
            ),
            margin=(0, 0, 10, 0),
        )
        self.config_id_input = pn.widgets.TextInput(
            name="Configuration ID",
            placeholder="Enter an ID to save or overwrite",
            sizing_mode="stretch_width",
        )
        self.preset_select = pn.widgets.Select(
            name="Stored configurations",
            options={},
            sizing_mode="stretch_width",
        )
        self.load_button = pn.widgets.Button(
            name="Load",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self.save_button = pn.widgets.Button(
            name="Save",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        self.delete_button = pn.widgets.Button(
            name="Delete",
            button_type="warning",
            sizing_mode="stretch_width",
        )
        self.reset_button = pn.widgets.Button(
            name="Reset configurations",
            button_type="danger",
            sizing_mode="stretch_width",
            visible=allow_reset,
        )
        self.status = pn.pane.HTML(
            '<div style="font-size:12px;color:#52606d;">Ready.</div>',
            margin=(8, 0, 8, 0),
        )

        self.load_button.on_click(self._on_load)
        self.save_button.on_click(self._on_save)
        self.delete_button.on_click(self._on_delete)
        self.reset_button.on_click(self._on_request_reset)
        self.preset_select.param.watch(self._on_select_change, "value")

        self._refresh_preset_options()
        self._render_editor()

        self.view = pn.Column(
            self.header,
            self.meta,
            self.config_id_input,
            self.preset_select,
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
            self._reset_confirm_host,
            self.status,
            self._editor_host,
            sizing_mode="stretch_width",
        )

    def _new_instance(self) -> param.Parameterized:
        return self.config_cls()

    def _set_status(self, message: str, *, tone: str = "neutral") -> None:
        palette = {
            "neutral": "#52606d",
            "success": "#166534",
            "warning": "#9a3412",
            "danger": "#b91c1c",
        }
        color = palette.get(tone, palette["neutral"])
        self.status.object = (
            f'<div style="font-size:12px;color:{color};">{message}</div>'
        )

    def _refresh_preset_options(self, *, select_id: str | None = None) -> None:
        self.conf_type.load()
        options = {config_id: config_id for config_id in self.conf_type.confIDs}
        self.preset_select.options = options
        if select_id is not None and select_id in options:
            self.preset_select.value = select_id
            return
        if self.preset_select.value not in options:
            self.preset_select.value = next(iter(options.values()), None)

    def _render_editor(self) -> None:
        self._editor_host.objects = [self._build_parameterized_section(self._current)]

    def _on_select_change(self, event: param.parameterized.Event) -> None:
        if event.new:
            self.config_id_input.value = str(event.new)

    def _on_load(self, _event: Any = None) -> None:
        config_id = self.preset_select.value
        if not config_id:
            self._set_status("Select a stored configuration to load.", tone="warning")
            return
        try:
            self.conf_type.load()
            self._current = self.conf_type.get(config_id)
        except Exception as exc:
            self._set_status(f"Load failed: {exc}", tone="danger")
            return
        self.config_id_input.value = str(config_id)
        self._render_editor()
        self._set_status(
            f'Loaded {self.conf_type.conftype} configuration "{config_id}".',
            tone="success",
        )

    def _on_save(self, _event: Any = None) -> None:
        config_id = self.config_id_input.value.strip() or self.preset_select.value
        if not config_id:
            self._set_status("Enter a configuration ID before saving.", tone="warning")
            return
        try:
            payload = _serialize_parameterized(self._current)
            self.conf_type.setID(config_id, payload)
        except Exception as exc:
            self._set_status(f"Save failed: {exc}", tone="danger")
            return
        self._refresh_preset_options(select_id=config_id)
        self._set_status(
            f'Saved {self.conf_type.conftype} configuration "{config_id}".',
            tone="success",
        )

    def _on_delete(self, _event: Any = None) -> None:
        config_id = self.preset_select.value
        if not config_id:
            self._set_status("Select a stored configuration to delete.", tone="warning")
            return
        try:
            self.conf_type.delete(config_id)
        except Exception as exc:
            self._set_status(f"Delete failed: {exc}", tone="danger")
            return
        self._refresh_preset_options()
        if self.config_id_input.value == config_id:
            self.config_id_input.value = ""
        self._set_status(
            f'Deleted {self.conf_type.conftype} configuration "{config_id}".',
            tone="success",
        )

    def _on_request_reset(self, _event: Any = None) -> None:
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

        def _confirm(_click: Any = None) -> None:
            try:
                self.conf_type.reset(recreate=True)
            except Exception as exc:
                self._set_status(f"Reset failed: {exc}", tone="danger")
            else:
                self._refresh_preset_options()
                self._set_status(
                    f"Recreated the {self.conf_type.conftype} registry store.",
                    tone="success",
                )
            self._reset_confirm_host.objects = []

        def _cancel(_click: Any = None) -> None:
            self._reset_confirm_host.objects = []
            self._set_status("Reset configurations cancelled.")

        confirm.on_click(_confirm)
        cancel.on_click(_cancel)
        self._reset_confirm_host.objects = [
            pn.Column(
                message,
                pn.Row(confirm, cancel, sizing_mode="stretch_width"),
                sizing_mode="stretch_width",
                margin=(0, 0, 8, 0),
            )
        ]

    def _build_parameterized_section(
        self,
        instance: param.Parameterized,
        *,
        title: str | None = None,
    ) -> pn.Column:
        section = pn.Column(sizing_mode="stretch_width", margin=(0, 0, 10, 0))
        heading = title or instance.name or type(instance).__name__
        section.append(
            pn.pane.HTML(
                f'<div style="font-size:14px;font-weight:600;color:#1f2933;">{heading}</div>',
                margin=(0, 0, 8, 0),
            )
        )

        simple_params: list[str] = []
        nested_params: list[tuple[str, Any]] = []
        for name, parameter in instance.param.objects(instance=False).items():
            if name == "name" or parameter.readonly:
                continue
            if isinstance(parameter, (ClassAttr, ClassDict)):
                nested_params.append((name, parameter))
            else:
                simple_params.append(name)

        if simple_params:
            section.append(
                pn.Param(
                    instance,
                    parameters=simple_params,
                    show_name=False,
                    widgets=_widget_overrides(instance, simple_params),
                    sizing_mode="stretch_width",
                )
            )

        for name, parameter in nested_params:
            if isinstance(parameter, ClassAttr):
                section.append(self._build_classattr_section(instance, name, parameter))
            elif isinstance(parameter, ClassDict):
                section.append(self._build_classdict_section(instance, name, parameter))
        if not simple_params and not nested_params:
            section.append(
                pn.pane.HTML(
                    '<div style="font-size:12px;color:#7b8794;">No editable parameters.</div>',
                    margin=0,
                )
            )
        return section

    def _build_classattr_section(
        self,
        owner: param.Parameterized,
        name: str,
        parameter: ClassAttr,
    ) -> pn.Card:
        nested_value = getattr(owner, name)
        title = name.replace("_", " ").title()
        content = pn.Column(sizing_mode="stretch_width")

        if nested_value is None:
            create_button = pn.widgets.Button(
                name=f"Initialize {title}",
                button_type="primary",
                sizing_mode="stretch_width",
            )

            def _create(_event: Any = None) -> None:
                try:
                    setattr(owner, name, _instantiate_classattr(parameter))
                except Exception as exc:
                    self._set_status(
                        f'Could not initialize "{title}": {exc}', tone="danger"
                    )
                    return
                self._render_editor()
                self._set_status(f'Initialized "{title}".', tone="success")

            create_button.on_click(_create)
            content.append(create_button)
        else:
            content.append(self._build_parameterized_section(nested_value, title=title))

        return pn.Card(
            content,
            title=title,
            collapsed=False,
            sizing_mode="stretch_width",
            margin=(8, 0, 8, 0),
        )

    def _build_classdict_section(
        self,
        owner: param.Parameterized,
        name: str,
        parameter: ClassDict,
    ) -> pn.Card:
        title = name.replace("_", " ").title()
        items = getattr(owner, name)
        if items is None:
            items = util.AttrDict()
            setattr(owner, name, items)

        select = pn.widgets.Select(
            name="Selected item",
            options={key: key for key in items.keys()},
            sizing_mode="stretch_width",
        )
        key_input = pn.widgets.TextInput(
            name="New item key",
            placeholder="Enter a key and click Add item",
            sizing_mode="stretch_width",
        )
        add_button = pn.widgets.Button(
            name="Add item",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        delete_button = pn.widgets.Button(
            name="Delete item",
            button_type="warning",
            sizing_mode="stretch_width",
        )
        editor_host = pn.Column(sizing_mode="stretch_width")

        def _refresh_nested_editor() -> None:
            select.options = {key: key for key in items.keys()}
            if select.value not in select.options.values():
                select.value = next(iter(select.options.values()), None)
            current_key = select.value
            if current_key is None:
                editor_host.objects = [
                    pn.pane.HTML(
                        '<div style="font-size:12px;color:#7b8794;">No items in this collection.</div>',
                        margin=0,
                    )
                ]
                return
            current_item = items[current_key]
            if isinstance(current_item, param.Parameterized):
                editor_host.objects = [
                    self._build_parameterized_section(
                        current_item, title=str(current_key)
                    )
                ]
            else:
                editor_host.objects = [
                    pn.pane.JSON(current_item, depth=3, sizing_mode="stretch_width")
                ]

        def _add_item(_event: Any = None) -> None:
            item_key = key_input.value.strip()
            if not item_key:
                self._set_status(
                    f'Enter a key before adding to "{title}".', tone="warning"
                )
                return
            if item_key in items:
                self._set_status(
                    f'"{item_key}" already exists in "{title}".', tone="warning"
                )
                return
            item_cls = parameter.item_type
            if not _is_parameterized_class(item_cls):
                self._set_status(
                    f'"{title}" does not expose a parameterized item type.',
                    tone="danger",
                )
                return
            try:
                items[item_key] = item_cls()
            except Exception as exc:
                self._set_status(f'Could not add "{item_key}": {exc}', tone="danger")
                return
            key_input.value = ""
            _refresh_nested_editor()
            select.value = item_key
            self._set_status(f'Added "{item_key}" to "{title}".', tone="success")

        def _delete_item(_event: Any = None) -> None:
            item_key = select.value
            if item_key is None:
                self._set_status(
                    f'Select an item to delete from "{title}".', tone="warning"
                )
                return
            items.pop(item_key, None)
            _refresh_nested_editor()
            self._set_status(f'Deleted "{item_key}" from "{title}".', tone="success")

        add_button.on_click(_add_item)
        delete_button.on_click(_delete_item)
        select.param.watch(lambda _event: _refresh_nested_editor(), "value")

        _refresh_nested_editor()

        return pn.Card(
            pn.Column(
                select,
                key_input,
                pn.Row(add_button, delete_button, sizing_mode="stretch_width"),
                editor_host,
                sizing_mode="stretch_width",
            ),
            title=title,
            collapsed=False,
            sizing_mode="stretch_width",
            margin=(8, 0, 8, 0),
        )


def build_conftype_widget(
    config_cls: type[param.Parameterized],
    *,
    conftype: str | None = None,
    title: str | None = None,
    allow_reset: bool = True,
):
    controller = ConftypeWidgetController(
        config_cls,
        conftype=conftype,
        title=title,
        allow_reset=allow_reset,
    )
    return controller.view
