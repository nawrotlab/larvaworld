"""Single-parameter popup content: the Inspect/Add-Parameter view, shared
between read-only display and editable clone-and-save. Standalone from
`parameter_db_app.py`, which owns the popup *shell*
(`_DraggableResizablePopup`) and database-wide actions (Remove
confirmation, the table itself) -- these builders only produce content to
be placed into that shell's body.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import panel as pn

from larvaworld.lib import reg
from larvaworld.lib.reg.data_aux import LarvaworldParam
from larvaworld.portal.buttons import import_button, save_button

from . import parameter_db_data
from .parameter_funcs import get_param_instance, param_from_file, register_new_param

__all__: list[str] = [
    "build_param_detail_popup",
]

#: pn.Card renders as a native Web Component with its own shadow root
#: (Panel's card.css uses `:host(.card)`), so a stylesheet on the
#: *surrounding* popup can never reach `.card-title`/`.card-header` --
#: this must be passed directly to each Card's own `stylesheets` param
#: instead, same reasoning as parameter_db_app._POPUP_STYLESHEET but one
#: shadow root deeper.
_CARD_STYLESHEET = """
.card-header {
  justify-content: center;
}

.card-title {
  text-align: center;
  width: 100%;
}
"""

#: Options for the "Data type" field's Select widget -- the base Python
#: types `resolve_param_class` actually maps to a param.Parameter class
#: (see larvaworld.lib.param.custom.resolve_param_class); typing generics
#: and other exotic dtypes some existing parameters may carry are appended
#: dynamically in `_dtype_select_options` instead of listed here, to keep
#: the common-case dropdown short.
_DTYPE_BASE_OPTIONS: list[type] = [float, int, str, bool, list, dict]


def _dtype_select_options(current: Any) -> dict[str, Any]:
    options = list(_DTYPE_BASE_OPTIONS)
    if current not in options:
        options.append(current)
    return {getattr(t, "__name__", str(t)): t for t in options}


#: Detail-popup layout: four labeled groups of attributes, laid out as two
#: panels per column. `"v"` and `"description"` aren't LarvaworldParam-
#: declared attrs directly ("v" is the dynamically-typed value param added
#: by get_LarvaworldParam; "description" is a derived property reading
#: "v"'s doc) but are included here since the user-facing grouping treats
#: them the same as any other field.
_DETAIL_FIELD_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Identity",
        [
            ("p", "Name"),
            ("k", "Key"),
            ("sym", "Symbol"),
            ("description", "Description"),
        ],
    ),
    ("Value", [("v", "Default value"), ("u", "Unit"), ("dtype", "Data type")]),
    (
        "Naming",
        [
            ("disp", "Display name"),
            ("d", "Name in dataset"),
            ("codename", "Name in code"),
            ("flatname", "Name in config file"),
        ],
    ),
    (
        "Computation",
        [("func", "Computing function"), ("required_ks", "Required param keys")],
    ),
]

#: Fields that stay read-only regardless of `editable`: "u" (a pint Unit --
#: editing units via free text is scientifically risky and wasn't asked
#: for) and "func" (a Python callable, shown as "module.qualname" -- not
#: meaningfully editable as text).
_ALWAYS_READONLY_FIELDS = frozenset({"u", "func"})


def _detail_field_widget(
    instance: LarvaworldParam, attr: str, label: str, *, editable: bool
) -> pn.viewable.Viewable:
    """A single labeled widget for one detail-popup field, disabled unless
    `editable` and the field allows editing (see _ALWAYS_READONLY_FIELDS).
    "v" gets its properly-typed widget (Number/Range/...) via `pn.Param`;
    "dtype" gets a Select over the common base types; the rest are plain
    text."""
    field_editable = editable and attr not in _ALWAYS_READONLY_FIELDS

    if attr == "v":
        return pn.Param(
            instance,
            parameters=["v"],
            show_name=False,
            # description=None suppresses the "?" tooltip pn.Param would
            # otherwise copy in from the "v" param's own `doc` -- that doc
            # describes the parameter's scientific meaning, not this
            # widget, and reads as a stray, unexplained icon here.
            widgets={
                "v": {
                    "name": label,
                    "disabled": not field_editable,
                    "description": None,
                }
            },
            sizing_mode="stretch_width",
        )
    if attr == "dtype":
        return pn.widgets.Select(
            name=label,
            options=_dtype_select_options(instance.dtype),
            value=instance.dtype,
            disabled=not field_editable,
            sizing_mode="stretch_width",
        )
    if attr == "description":
        return pn.widgets.TextAreaInput(
            name=label,
            value=instance.description or "",
            disabled=not field_editable,
            height=70,
            sizing_mode="stretch_width",
        )
    if attr == "required_ks":
        return pn.widgets.MultiChoice(
            name=label,
            options=sorted(reg.par.dict.keys()),
            value=list(instance.required_ks) if instance.required_ks else [],
            disabled=not field_editable,
            sizing_mode="stretch_width",
        )
    if attr == "u":
        value = str(instance.u)
    elif attr == "func":
        func = instance.func
        value = f"{func.__module__}.{func.__qualname__}" if func is not None else "—"
    else:
        value = getattr(instance, attr)
    return pn.widgets.TextInput(
        name=label,
        value=str(value),
        disabled=not field_editable,
        sizing_mode="stretch_width",
    )


def _build_detail_grid(
    instance: LarvaworldParam, *, editable: bool
) -> tuple[pn.viewable.Viewable, dict[str, pn.viewable.Viewable]]:
    """Two-column layout: `_DETAIL_FIELD_GROUPS` rendered as four labeled
    panels, two per column, with a gap between the columns. Returns the
    view plus a {attr: widget} map (only for fields that can be edited)
    so a Save handler can read the current values back out."""
    field_widgets: dict[str, pn.viewable.Viewable] = {}
    panels = []
    for title, fields in _DETAIL_FIELD_GROUPS:
        widgets = []
        for attr, label in fields:
            widget = _detail_field_widget(instance, attr, label, editable=editable)
            widgets.append(widget)
            # "v"'s widget is a pn.Param pane bound live to `instance` (a
            # two-way binding, unlike the plain TextInput/Select widgets
            # built for every other field) -- editing it already updates
            # `instance.v` directly, so `instance.to_config()` picks up
            # the edit on its own; it doesn't need a readback entry here
            # (and a pn.Param pane has no single `.value` to read anyway).
            if editable and attr not in _ALWAYS_READONLY_FIELDS and attr != "v":
                field_widgets[attr] = widget
        panels.append(
            pn.Card(
                *widgets,
                title=title,
                collapsible=False,
                sizing_mode="stretch_width",
                margin=(0, 0, 10, 0),
                stylesheets=[_CARD_STYLESHEET],
            )
        )
    left = pn.Column(panels[0], panels[1], sizing_mode="stretch_width", margin=0)
    right = pn.Column(
        panels[2], panels[3], sizing_mode="stretch_width", margin=(0, 0, 0, 16)
    )
    return pn.Row(left, right, sizing_mode="stretch_width", margin=0), field_widgets


def _field_widget_value(attr: str, widget: pn.viewable.Viewable) -> Any:
    if attr == "required_ks":
        return list(widget.value)
    return widget.value


def build_param_detail_popup(
    k: Optional[str] = None,
    *,
    instance: Optional[LarvaworldParam] = None,
    editable: bool = False,
    on_save: Optional[Callable[[dict[str, Any]], None]] = None,
) -> pn.viewable.Viewable:
    """
    Build a two-column view of a `LarvaworldParam`'s attributes (see
    `_DETAIL_FIELD_GROUPS`), grouped into four labeled panels with
    human-readable field labels rather than raw attribute names.

    The parameter can be given either by registry key (`k`, looked up via
    `parameter_funcs.get_param_instance`) or directly as an already-built
    `instance` -- the latter is how the Add-Parameter flow uses this for an
    in-memory clone that isn't registered under any key yet. Exactly one of
    `k`/`instance` must be given.

    `editable=False` (the Inspect popup): every field disabled.
    `editable=True` (the Add-Parameter popup): most fields editable (see
    `_ALWAYS_READONLY_FIELDS` for the read-only exceptions -- Unit and
    Function); if `on_save` is given, a "Save" button collects the current
    widget values (via `LarvaworldParam.to_config()` as the base, with
    edited fields overlaid -- "description" maps back to the `doc` key)
    and calls `on_save(fields)`, showing `ValueError`/`KeyError` inline
    instead of raising.

    Exporting a parameter's config to file is handled by the sidebar's
    "Export" button (see parameter_db_app.build_parameter_db_content), not
    here.
    """
    if instance is None:
        if k is None:
            raise ValueError("build_param_detail_popup requires either k or instance.")
        instance = get_param_instance(k)

    # No redundant "Parameter configuration" heading here -- the popup's own
    # title ("Parameter: <name>" for Inspect, "Add Parameter" for Add)
    # already says what this is.
    grid, field_widgets = _build_detail_grid(instance, editable=editable)
    children: list[pn.viewable.Viewable] = [grid]

    if editable and on_save is not None:
        status_pane = pn.pane.Markdown("", margin=(4, 0, 0, 0))
        save_btn = save_button(name="Save", margin=(8, 0, 0, 0), sizing_mode=None)

        def _on_click(_: object) -> None:
            fields = dict(instance.to_config())
            for attr, widget in field_widgets.items():
                key = "doc" if attr == "description" else attr
                fields[key] = _field_widget_value(attr, widget)
            try:
                on_save(fields)
                status_pane.object = ""
            except (ValueError, KeyError) as exc:
                status_pane.object = f":red_circle: {exc}"

        save_btn.on_click(_on_click)
        children.extend([save_btn, status_pane])

    return pn.Column(*children, css_classes=["lw-parameter-db-detail"], margin=0)


def _build_add_parameter_popup(
    table: pn.widgets.Tabulator,
    refresh: Callable[[], None],
    *,
    initial_source_k: Optional[str] = None,
) -> pn.viewable.Viewable:
    """
    Clone-and-edit Add-Parameter form. Layout: a "Load from file" button
    row on top, then the editable field grid below it, via
    `build_param_detail_popup(instance=..., editable=True)`.

    If `initial_source_k` is given (from a row's "Clone" icon, which
    pre-loads that row), the form starts already populated with that
    parameter's config instead of empty.
    """
    clone_key_input = pn.widgets.AutocompleteInput(
        name="",
        placeholder="Parameter key to clone",
        options=sorted(reg.par.dict.keys()),
        case_sensitive=False,
        restrict=False,
        width=200,
        margin=(0, 4, 0, 0),
    )
    clone_key_button = pn.widgets.Button(
        name="Clone by key",
        button_type="primary",
        margin=(0, 8, 0, 0),
    )
    load_file_button, config_input = import_button(
        "Load from file",
        accept=".pkl,.json",
        margin=(0, 8, 0, 0),
        sizing_mode=None,
    )
    status_pane = pn.pane.Markdown("", margin=(0, 0, 0, 8))
    detail_panel = pn.Column(
        pn.pane.Markdown("_Load a parameter to edit its fields._"),
        margin=0,
        sizing_mode="stretch_width",
    )

    def _on_save(fields: dict[str, Any]) -> None:
        # ValueError/KeyError propagate to build_param_detail_popup's own
        # Save handler, which displays them inline.
        register_new_param(fields)
        table.value = parameter_db_data.build_parameter_table_df()
        refresh()

    def _show_instance(instance: LarvaworldParam) -> None:
        detail_panel[:] = [
            build_param_detail_popup(instance=instance, editable=True, on_save=_on_save)
        ]

    def _load_key(source_k: Optional[str]) -> None:
        if not source_k:
            status_pane.object = ":red_circle: Pick a parameter key to clone from."
            return
        try:
            source_instance = get_param_instance(source_k)
        except KeyError:
            status_pane.object = f":red_circle: Unknown parameter key {source_k!r}."
            return
        status_pane.object = ""
        _show_instance(LarvaworldParam.from_config(source_instance.to_config()))

    def _load_file(_: object = None) -> None:
        # Fires when config_input.value changes -- populated by
        # load_file_button's js_on_click above, not by a Python click
        # handler on this function.
        if not config_input.value:
            return
        try:
            suffix = Path(config_input.filename or "").suffix or ".pkl"
            instance = param_from_file(config_input.value, suffix=suffix)
        except Exception as exc:
            status_pane.object = f":red_circle: Could not load config: {exc}"
            return
        status_pane.object = ""
        _show_instance(instance)

    config_input.param.watch(_load_file, "value")
    clone_key_button.on_click(lambda _: _load_key(clone_key_input.value))

    header = pn.Row(
        clone_key_input,
        clone_key_button,
        config_input,
        load_file_button,
        status_pane,
        css_classes=["lw-parameter-db-add-header"],
        margin=(0, 0, 12, 0),
    )

    if initial_source_k:
        _load_key(initial_source_k)

    return pn.Column(
        header,
        detail_panel,
        css_classes=["lw-parameter-db-add-form"],
        sizing_mode="stretch_width",
        margin=0,
    )
