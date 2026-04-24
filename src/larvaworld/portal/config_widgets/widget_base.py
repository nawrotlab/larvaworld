from __future__ import annotations

from html import escape
from typing import Any, Callable

import panel as pn
import param

from larvaworld.lib import util
from larvaworld.lib.param.custom import ClassAttr, ClassDict

__all__ = [
    "classattr_section",
    "classdict_editor",
    "collapsible_family_box",
    "doc_pane",
    "editable_parameter_names",
    "family_box",
    "instantiate_classattr",
    "numeric_tuple_param_control",
    "param_control",
    "param_controls",
    "parameterized_editor",
    "safe_widget_overrides",
    "widget_block",
]


def family_box(
    title: str,
    *children: object,
    css_classes: list[str] | None = None,
    title_css_classes: list[str] | None = None,
    header_right: object | None = None,
) -> pn.Column:
    title_pane = pn.pane.Markdown(
        f"**{title}**",
        css_classes=title_css_classes or [],
        margin=(0, 0, 4, 0),
    )
    header: object
    if header_right is None:
        header = title_pane
    else:
        header = pn.Row(
            title_pane,
            pn.Spacer(sizing_mode="stretch_width"),
            header_right,
            css_classes=["lw-import-datasets-config-family-header"],
            sizing_mode="stretch_width",
            margin=(0, 0, 4, 0),
        )
    return pn.Column(
        header,
        *children,
        css_classes=css_classes or [],
        sizing_mode="stretch_width",
        margin=0,
    )


def collapsible_family_box(
    title: str,
    *children: object,
    css_classes: list[str] | None = None,
    collapsed: bool = False,
) -> pn.Card:
    return pn.Card(
        *children,
        title=title,
        collapsed=collapsed,
        collapsible=True,
        css_classes=css_classes or [],
        sizing_mode="stretch_width",
        margin=0,
    )


def _widget_has_native_help(widget: object) -> bool:
    description = getattr(widget, "description", None)
    return isinstance(description, str) and description.strip() != ""


def doc_pane(doc: str | None) -> pn.pane.HTML | None:
    if not doc:
        return None
    return pn.pane.HTML(
        f'<div class="lw-import-datasets-config-help">{escape(doc)}</div>',
        margin=0,
    )


def widget_block(widget: object, *, doc: str | None = None) -> pn.Column:
    children = [widget]
    pane = None if _widget_has_native_help(widget) else doc_pane(doc)
    if pane is not None:
        children.append(pane)
    return pn.Column(*children, sizing_mode="stretch_width", margin=0)


def _single_widget_override(parameter: param.Parameter) -> dict[str, object]:
    default = getattr(parameter, "default", None)
    if isinstance(parameter, param.ListSelector):
        return {"type": pn.widgets.MultiChoice}
    if isinstance(parameter, param.Integer):
        if getattr(parameter, "allow_None", False):
            return {"type": pn.widgets.LiteralInput}
        return {"type": pn.widgets.IntInput}
    if isinstance(parameter, param.Number):
        if getattr(parameter, "allow_None", False):
            return {"type": pn.widgets.LiteralInput}
        return {"type": pn.widgets.FloatInput}
    if isinstance(parameter, (param.Range, param.NumericTuple, param.Dict, param.List)):
        return {"type": pn.widgets.LiteralInput}
    if isinstance(default, (dict, list, tuple)):
        return {"type": pn.widgets.LiteralInput}
    return {}


def safe_widget_overrides(
    instance: param.Parameterized, parameter_names: list[str]
) -> dict[str, dict[str, object]]:
    widgets: dict[str, dict[str, object]] = {}
    for name in parameter_names:
        parameter = instance.param.objects(instance=False)[name]
        override = _single_widget_override(parameter)
        if override:
            widgets[name] = override
    return widgets


def editable_parameter_names(
    instance: param.Parameterized,
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    excluded = {"name"}
    if exclude is not None:
        excluded.update(exclude)
    return [
        name
        for name, parameter in instance.param.objects(instance=False).items()
        if name not in excluded and not parameter.readonly
    ]


def param_control(
    obj: param.Parameterized,
    *,
    parameter_name: str,
    widget_overrides: dict[str, dict[str, object]] | None = None,
) -> pn.Column:
    param_pane = pn.Param(
        obj,
        parameters=[parameter_name],
        widgets=widget_overrides or safe_widget_overrides(obj, [parameter_name]),
        sizing_mode="stretch_width",
        show_name=False,
        expand_button=False,
        expand=False,
    )
    widget = param_pane._widgets.get(parameter_name)
    if widget is None:
        return pn.Column(sizing_mode="stretch_width", margin=0)
    container = widget_block(widget, doc=getattr(obj.param[parameter_name], "doc", None))
    container._param_pane = param_pane
    container._widgets = {parameter_name: widget}
    return container


def param_controls(
    obj: param.Parameterized,
    *,
    parameters: list[str],
    widget_overrides: dict[str, dict[str, object]] | None = None,
) -> pn.Column:
    controls = [
        param_control(
            obj,
            parameter_name=name,
            widget_overrides=widget_overrides,
        )
        for name in parameters
    ]
    return pn.Column(*controls, sizing_mode="stretch_width", margin=0)


def _normalize_two_tuple(value: Any) -> tuple[Any, Any]:
    if value is None:
        return (None, None)
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value, value)


def numeric_tuple_param_control(
    obj: Any,
    *,
    parameter_name: str,
    labels: tuple[str, str],
    numeric_type: type = float,
    title: str | None = None,
    doc: str | None = None,
    step: float | int | None = None,
) -> pn.Column:
    input_type = pn.widgets.FloatInput if numeric_type is float else pn.widgets.IntInput
    raw_value = _normalize_two_tuple(getattr(obj, parameter_name))
    state = {"syncing": False}
    widgets = [
        input_type(name=labels[0], value=raw_value[0], step=step, sizing_mode="stretch_width"),
        input_type(name=labels[1], value=raw_value[1], step=step, sizing_mode="stretch_width"),
    ]

    def _coerce_values() -> tuple[Any, Any]:
        values = []
        for widget in widgets:
            value = widget.value
            if value is None:
                values.append(None)
            else:
                values.append(numeric_type(value))
        return tuple(values)

    def _push_to_owner(*_events: Any) -> None:
        if state["syncing"]:
            return
        setattr(obj, parameter_name, _coerce_values())

    for widget in widgets:
        widget.param.watch(_push_to_owner, "value")

    if isinstance(obj, param.Parameterized):
        def _sync_from_owner(*_events: Any) -> None:
            state["syncing"] = True
            try:
                left, right = _normalize_two_tuple(getattr(obj, parameter_name))
                widgets[0].value = left
                widgets[1].value = right
            finally:
                state["syncing"] = False

        obj.param.watch(_sync_from_owner, parameter_name)

    row = pn.Row(*widgets, sizing_mode="stretch_width", margin=0)
    block = widget_block(row, doc=doc)
    block._widgets = {labels[0]: widgets[0], labels[1]: widgets[1]}
    if title:
        return pn.Column(
            pn.pane.Markdown(f"*{title}*", margin=(0, 0, 2, 0)),
            block,
            sizing_mode="stretch_width",
            margin=0,
        )
    return block


def _class_choices(parameter: ClassAttr) -> list[type[Any]]:
    class_ = parameter.class_
    if isinstance(class_, tuple):
        return list(class_)
    return [class_]


def _class_label(cls: type[Any]) -> str:
    agent_class = getattr(cls, "agent_class", None)
    if callable(agent_class):
        try:
            label = agent_class()
        except Exception:
            label = None
        if isinstance(label, str) and label.strip():
            return label.strip()
    return cls.__name__


def instantiate_classattr(
    parameter: ClassAttr,
    *,
    target_class: type[Any] | None = None,
    source_instance: param.Parameterized | None = None,
) -> param.Parameterized:
    nested_cls = target_class or _class_choices(parameter)[0]
    new_instance = nested_cls()
    if source_instance is None or not isinstance(new_instance, param.Parameterized):
        return new_instance

    for name, source_parameter in source_instance.param.objects(instance=False).items():
        if name in {"name", "unique_id", "odorscape"} or source_parameter.readonly:
            continue
        if name not in new_instance.param:
            continue
        target_parameter = new_instance.param[name]
        if target_parameter.readonly or isinstance(target_parameter, (ClassAttr, ClassDict)):
            continue
        try:
            setattr(new_instance, name, getattr(source_instance, name))
        except Exception:
            continue
    return new_instance


def _ensure_attrdict(owner: param.Parameterized, parameter_name: str) -> util.AttrDict:
    current = getattr(owner, parameter_name)
    if isinstance(current, util.AttrDict):
        return current
    current = util.AttrDict(current or {})
    setattr(owner, parameter_name, current)
    return current


def parameterized_editor(
    instance: param.Parameterized,
    *,
    parameter_order: list[str] | None = None,
    exclude: set[str] | None = None,
    custom_builders: dict[str, Callable[[param.Parameterized, str, Any], object]] | None = None,
) -> pn.Column:
    custom = custom_builders or {}
    params = instance.param.objects(instance=False)
    ordered_names = parameter_order or editable_parameter_names(instance, exclude=exclude)
    children: list[object] = []
    for name in ordered_names:
        parameter = params.get(name)
        if parameter is None or parameter.readonly or name == "name":
            continue
        if name in custom:
            children.append(custom[name](instance, name, parameter))
            continue
        if isinstance(parameter, ClassAttr):
            children.append(
                classattr_section(
                    instance,
                    name=name,
                    parameter=parameter,
                )
            )
            continue
        if isinstance(parameter, ClassDict):
            children.append(
                classdict_editor(
                    instance,
                    name=name,
                    parameter=parameter,
                )
            )
            continue
        children.append(param_control(instance, parameter_name=name))

    if not children:
        children.append(
            pn.pane.HTML(
                '<div class="lw-import-datasets-config-help">No editable parameters.</div>',
                margin=0,
            )
        )
    return pn.Column(*children, sizing_mode="stretch_width", margin=0)


def classattr_section(
    owner: param.Parameterized,
    *,
    name: str,
    parameter: ClassAttr,
    title: str | None = None,
    build_editor: Callable[[param.Parameterized], object] | None = None,
    controls_layout: str = "row",
    box_css_classes: list[str] | None = None,
    title_css_classes: list[str] | None = None,
    enable_control: str = "checkbox",
) -> pn.Column:
    section_title = title or name.replace("_", " ").title()
    choices = _class_choices(parameter)
    current_value = getattr(owner, name)
    optional = parameter.default is None
    state = {"syncing": False}
    use_switch_header = optional and enable_control == "switch"

    if enable_control == "switch":
        enabled = pn.widgets.Switch(name="", value=current_value is not None, width=18, margin=0)
    else:
        enabled = pn.widgets.Checkbox(
            name=f"Enable {section_title}",
            value=current_value is not None,
            sizing_mode="stretch_width",
        )
    class_options = {_class_label(cls): cls for cls in choices}
    current_class = type(current_value) if current_value is not None else choices[0]
    class_select = pn.widgets.Select(
        name=f"{section_title} type",
        options=list(class_options),
        value=_class_label(current_class),
        sizing_mode="stretch_width",
    )
    header_controls: list[object] = []
    if optional and enable_control != "switch":
        header_controls.append(enabled)
    if len(choices) > 1 and not optional:
        header_controls.append(class_select)
    editor_host = pn.Column(sizing_mode="stretch_width", margin=0)

    def _selected_class() -> type[Any]:
        return class_options[class_select.value]

    def _render() -> None:
        current = getattr(owner, name)
        if optional:
            state["syncing"] = True
            try:
                enabled.value = current is not None
            finally:
                state["syncing"] = False
        if current is not None:
            state["syncing"] = True
            try:
                class_select.value = _class_label(type(current))
            finally:
                state["syncing"] = False
        if optional and current is None:
            editor_host.objects = [
                pn.pane.HTML(
                    '<div class="lw-import-datasets-config-help">Disabled.</div>',
                    margin=0,
                )
            ]
            return
        if current is None:
            current = instantiate_classattr(parameter, target_class=_selected_class())
            setattr(owner, name, current)
        builder = build_editor or (
            lambda instance: parameterized_editor(instance)
        )
        objects: list[object] = []
        if len(choices) > 1:
            objects.append(class_select)
        objects.append(builder(current))
        editor_host.objects = objects

    def _handle_enabled(event: param.parameterized.Event) -> None:
        if state["syncing"]:
            return
        if event.new:
            source = getattr(owner, name)
            setattr(
                owner,
                name,
                instantiate_classattr(
                    parameter,
                    target_class=_selected_class(),
                    source_instance=source,
                ),
            )
        else:
            setattr(owner, name, None)
        _render()

    def _handle_class_change(event: param.parameterized.Event) -> None:
        if state["syncing"]:
            return
        current = getattr(owner, name)
        if optional and current is None and not enabled.value:
            return
        if current is not None and _class_label(type(current)) == event.new:
            return
        setattr(
            owner,
            name,
            instantiate_classattr(
                parameter,
                target_class=class_options[event.new],
                source_instance=current,
            ),
        )
        _render()

    if optional:
        enabled.param.watch(_handle_enabled, "value")
    if len(choices) > 1:
        class_select.param.watch(_handle_class_change, "value")
    owner.param.watch(lambda *_: _render(), name)
    _render()

    children: list[object] = []
    if parameter.doc and not use_switch_header:
        pane = doc_pane(parameter.doc)
        if pane is not None:
            children.append(pane)
    if header_controls:
        if controls_layout == "column":
            children.append(
                pn.Column(*header_controls, sizing_mode="stretch_width", margin=0)
            )
        else:
            children.append(
                pn.Row(*header_controls, sizing_mode="stretch_width", margin=0)
            )
    header_right = enabled if use_switch_header else None
    children.append(editor_host)
    return family_box(
        section_title,
        *children,
        css_classes=box_css_classes,
        title_css_classes=title_css_classes,
        header_right=header_right,
    )


def classdict_editor(
    owner: param.Parameterized,
    *,
    name: str,
    parameter: ClassDict,
    title: str | None = None,
    build_item_editor: Callable[[param.Parameterized, str], object] | None = None,
    item_label: str | None = None,
    box_css_classes: list[str] | None = None,
    title_css_classes: list[str] | None = None,
    wrap: bool = True,
) -> pn.Column:
    section_title = title or name.replace("_", " ").title()
    label = item_label or section_title.rstrip("s")
    empty_label = f"No {label.lower()}s yet"
    items = _ensure_attrdict(owner, name)
    select = pn.widgets.Select(
        name=f"Selected {label.lower()}",
        options={empty_label: ""},
        value="",
        sizing_mode="stretch_width",
    )
    key_input = pn.widgets.TextInput(
        name=f"New {label.lower()} ID",
        placeholder=f"Enter a {label.lower()} key",
        sizing_mode="stretch_width",
    )
    add_button = pn.widgets.Button(
        name=f"Add {label.lower()}",
        button_type="primary",
        width=150,
    )
    delete_button = pn.widgets.Button(
        name=f"Delete {label.lower()}",
        button_type="warning",
        width=150,
    )
    editor_host = pn.Column(sizing_mode="stretch_width", margin=0)

    def _assign_items(new_items: util.AttrDict) -> None:
        setattr(owner, name, util.AttrDict(new_items))

    def _refresh() -> None:
        current_items = _ensure_attrdict(owner, name)
        if current_items:
            select.options = {key: key for key in current_items}
            if select.value not in current_items:
                select.value = next(iter(current_items))
            select.disabled = False
            current_key = select.value
            current_item = current_items[current_key]
            builder = build_item_editor or (
                lambda item, key: family_box(key, parameterized_editor(item))
            )
            editor_host.objects = [builder(current_item, str(current_key))]
            delete_button.disabled = False
        else:
            select.options = {empty_label: ""}
            select.value = ""
            select.disabled = True
            editor_host.objects = [
                pn.pane.HTML(
                    '<div class="lw-import-datasets-config-help">No items in this collection.</div>',
                    margin=0,
                )
            ]
            delete_button.disabled = True

    def _build_new_item(item_key: str) -> param.Parameterized:
        item = parameter.item_type()
        if hasattr(item, "param") and "unique_id" in item.param:
            try:
                item.unique_id = item_key
            except Exception:
                pass
        return item

    def _handle_add(_event: Any) -> None:
        item_key = key_input.value.strip()
        if not item_key:
            return
        current_items = _ensure_attrdict(owner, name)
        updated = util.AttrDict(current_items)
        updated[item_key] = _build_new_item(item_key)
        _assign_items(updated)
        select.value = item_key
        key_input.value = ""
        _refresh()

    def _handle_delete(_event: Any) -> None:
        current_key = select.value
        if current_key in {None, ""}:
            return
        current_items = _ensure_attrdict(owner, name)
        updated = util.AttrDict(current_items)
        updated.pop(current_key, None)
        _assign_items(updated)
        _refresh()

    select.param.watch(lambda *_: _refresh(), "value")
    add_button.on_click(_handle_add)
    delete_button.on_click(_handle_delete)
    owner.param.watch(lambda *_: _refresh(), name)
    _refresh()

    children: list[object] = []
    if parameter.doc:
        pane = doc_pane(parameter.doc)
        if pane is not None:
            children.append(pane)
    children.extend(
        [
            select,
            key_input,
            pn.Row(
                add_button,
                delete_button,
                align="end",
                css_classes=["lw-import-datasets-inline-action-row"],
                sizing_mode="stretch_width",
                margin=0,
            ),
            editor_host,
        ]
    )
    if not wrap:
        return pn.Column(*children, sizing_mode="stretch_width", margin=0)
    return family_box(
        section_title,
        *children,
        css_classes=box_css_classes,
        title_css_classes=title_css_classes,
    )
