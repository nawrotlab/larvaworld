"""Panel UI layer for the Parameter Database popup: the draggable popup
shell, the table + sidebar, database-wide actions (Remove confirmation),
and the top-level dropdown/standalone-page wiring. Single-parameter popup
content (Inspect, Add) lives in `parameter_popups.py`.
"""

from __future__ import annotations

import io
from typing import Any, Callable, Optional

import panel as pn
import param

from . import parameter_db_data
from .parameter_funcs import (
    get_param_instance,
    remove_param,
    save_param_config,
    save_param_config_to_workspace,
)
from .parameter_popups import _build_add_parameter_popup, build_param_detail_popup

__all__: list[str] = [
    "build_parameter_db_content",
    "build_param_detail_popup",
    "build_standalone_page",
]

_LARGE_PANEL_MODIFIER_CLASS = "lw-parameter-db-dropdown-panel--large"

#: CSS for _DraggableResizablePopup's own template, passed via its
#: `stylesheets` param rather than the portal's global raw_css: ReactiveHTML
#: components render into their own scope, and document-level <style> rules
#: are not reliably guaranteed to reach into it.
_POPUP_STYLESHEET = """
.lw-parameter-db-panel-surface {
  position: absolute;
  top: 40px;
  right: 0;
  width: 65vw;
  max-width: 1100px;
  height: 68vh;
  max-height: 760px;
  min-width: 480px;
  min-height: 360px;
  resize: both;
  overflow: auto;
  display: flex;
  flex-direction: column;
  padding: 0;
  border-radius: 12px;
  border: 1px solid rgba(0,0,0,0.14);
  background: #ffffff;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.16);
  box-sizing: border-box;
  z-index: 30;
  color: #111111;
  font-size: 13px;
  line-height: 1.6;
}

.lw-parameter-db-panel-surface h3 {
  font-size: 17px;
  font-weight: 650;
  margin: 0 0 4px 0;
  color: #111111;
}

.lw-parameter-db-panel-surface p {
  margin: 0 0 10px 0;
  color: rgba(17,17,17,0.86);
}

.lw-parameter-db-drag-handle {
  position: relative;
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding: 10px 10px 10px 16px;
  cursor: grab;
  border-bottom: 1px solid rgba(0,0,0,0.1);
  border-radius: 12px 12px 0 0;
  background: rgba(15, 23, 42, 0.03);
  user-select: none;
}

.lw-parameter-db-drag-handle:active {
  cursor: grabbing;
}

.lw-parameter-db-drag-label {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  font-weight: 700;
  font-size: 19px;
  color: #111111;
  white-space: nowrap;
}

.lw-parameter-db-close-button {
  appearance: none;
  -webkit-appearance: none;
  box-sizing: border-box;
  width: 26px;
  height: 26px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: #111111;
  font-size: 20px;
  line-height: 1;
  cursor: pointer;
  padding: 0;
  margin: 0;
  z-index: 1;
}

.lw-parameter-db-close-button:hover {
  background: rgba(0, 0, 0, 0.08);
}

.lw-parameter-db-body-slot {
  flex: 1 1 auto;
  overflow-y: auto;
  padding: 16px;
}

.lw-parameter-db-panel-surface--compact {
  width: 900px;
  max-width: calc(100vw - 48px);
  height: 80vh;
  max-height: 900px;
  position: fixed;
  top: 50%;
  left: 50%;
  right: auto;
  transform: translate(-50%, -50%);
}

.lw-parameter-db-panel-surface--confirm {
  width: 320px;
  max-width: calc(100vw - 48px);
  height: auto;
  max-height: 50vh;
  min-height: 0;
  resize: none;
  position: fixed;
  top: 50%;
  left: 50%;
  right: auto;
  transform: translate(-50%, -50%);
}

.lw-parameter-db-sidebar {
  flex: 0 0 220px;
  width: 220px;
  max-width: 220px;
}

.lw-parameter-db-table-area {
  flex: 1 1 auto;
  min-width: 0;
}

.lw-parameter-db-panel-surface--yellow {
  background: #fffdf3;
  border-color: #e0b400;
}

.lw-parameter-db-panel-surface--yellow .lw-parameter-db-drag-handle {
  background: #ffe9a8;
  border-bottom-color: #e0b400;
}

.lw-parameter-db-panel-surface--cyan {
  background: #f2fdfe;
  border-color: #0e94a8;
}

.lw-parameter-db-panel-surface--cyan .lw-parameter-db-drag-handle {
  background: #a5f3fc;
  border-bottom-color: #0e94a8;
}

.lw-parameter-db-panel-surface--green {
  background: #f3fdf6;
  border-color: #2e9e5b;
}

.lw-parameter-db-panel-surface--green .lw-parameter-db-drag-handle {
  background: #b9eece;
  border-bottom-color: #2e9e5b;
}

.lw-parameter-db-panel-surface--red {
  background: #fff4f3;
  border-color: #cc3b30;
}

.lw-parameter-db-panel-surface--red .lw-parameter-db-drag-handle {
  background: #f6c2bd;
  border-bottom-color: #cc3b30;
}

.lw-parameter-db-clone-input {
  background: #e8fbfd;
  border: 1px solid #17a9b8;
  border-radius: 6px;
  padding: 4px 8px;
}

.lw-parameter-db-add-header {
  align-items: center;
  padding-bottom: 12px;
  border-bottom: 1px solid rgba(0,0,0,0.1);
}
"""


class _DraggableResizablePopup(pn.reactive.ReactiveHTML):
    """A floating popup wrapping arbitrary Panel content that can be
    dragged by its header bar and resized via its bottom-right corner
    (native CSS `resize`). Closing (the header's × button) sets the
    inherited `visible` param to False, same as toggling the trigger.
    """

    body = param.Parameter()
    title = param.String(default="Parameter Database")
    #: Extra CSS class appended to the surface div; pass
    #: "lw-parameter-db-panel-surface--compact" for smaller secondary
    #: popups (e.g. the parameter detail view) nested inside a larger one.
    size_modifier_class = param.String(default="")
    #: Extra CSS class for a two-color theme, e.g.
    #: "lw-parameter-db-panel-surface--yellow"/"--green"/"--red".
    theme_class = param.String(default="")
    stylesheets = param.List(default=[_POPUP_STYLESHEET])

    _template = """
    <div id="popup_surface" class="lw-parameter-db-panel-surface ${size_modifier_class} ${theme_class}">
      <div id="drag_handle" class="lw-parameter-db-drag-handle" onmousedown="${script('start_drag')}">
        <span class="lw-parameter-db-drag-label">${title}</span>
        <button id="close_btn" type="button" class="lw-parameter-db-close-button"
                onclick="${_close_btn_click}">×</button>
      </div>
      <div id="body_slot" class="lw-parameter-db-body-slot">${body}</div>
    </div>
    """

    _scripts = {
        "start_drag": """
          const surface = popup_surface
          const rect = surface.getBoundingClientRect()
          surface.style.position = 'fixed'
          surface.style.margin = '0'
          surface.style.right = 'auto'
          // Centered/compact popups use a CSS `transform: translate(-50%,-50%)`
          // for centering; getBoundingClientRect() already reflects that, so
          // it must be cleared here or the translate would double-apply on
          // top of the explicit left/top set below.
          surface.style.transform = 'none'
          surface.style.left = rect.left + 'px'
          surface.style.top = rect.top + 'px'
          const startX = event.clientX
          const startY = event.clientY
          const baseLeft = rect.left
          const baseTop = rect.top
          function onMove(e) {
            surface.style.left = (baseLeft + e.clientX - startX) + 'px'
            surface.style.top = (baseTop + e.clientY - startY) + 'px'
          }
          function onUp() {
            document.removeEventListener('mousemove', onMove)
            document.removeEventListener('mouseup', onUp)
          }
          document.addEventListener('mousemove', onMove)
          document.addEventListener('mouseup', onUp)
        """,
    }

    def _close_btn_click(self, event: object) -> None:
        self.visible = False


def _build_remove_confirm_popup(
    k: str, table: pn.widgets.Tabulator, hide: Callable[[], None]
) -> pn.viewable.Viewable:
    disp = get_param_instance(k).disp
    status_pane = pn.pane.Markdown("", margin=(8, 0, 0, 0))
    confirm_button = pn.widgets.Button(
        name="Delete", button_type="danger", margin=(0, 8, 0, 0)
    )
    cancel_button = pn.widgets.Button(name="Cancel", margin=0)

    def _confirm(_: object) -> None:
        try:
            remove_param(k)
        except KeyError as exc:
            status_pane.object = f":red_circle: {exc}"
            return
        table.value = parameter_db_data.build_parameter_table_df()
        hide()

    def _cancel(_: object) -> None:
        hide()

    confirm_button.on_click(_confirm)
    cancel_button.on_click(_cancel)

    return pn.Column(
        pn.pane.Markdown(
            f"Remove parameter **{disp}** (`{k}`)?",
            margin=(0, 0, 4, 0),
        ),
        pn.pane.Markdown(
            "*This cannot be undone.*",
            margin=(0, 0, 12, 0),
        ),
        pn.Row(
            confirm_button,
            cancel_button,
            margin=0,
            styles={"justify-content": "center"},
        ),
        status_pane,
        css_classes=["lw-parameter-db-remove-confirm"],
        margin=0,
    )


def _build_detail_popup() -> "_DraggableResizablePopup":
    """Build the (initially empty/hidden) secondary popup used by Show
    details / Add Parameter. Kept as a small helper since both the header
    dropdown and the standalone page need to construct one identically."""
    return _DraggableResizablePopup(
        body=pn.Column(margin=0),
        title="",
        visible=False,
        size_modifier_class="lw-parameter-db-panel-surface--compact",
        css_classes=["lw-parameter-db-detail-popup"],
        margin=0,
    )


#: CSS for the parameter Tabulator itself, passed via its own `stylesheets`
#: param (not the popup's): Tabulator.js renders into its own encapsulated
#: root, separate from _DraggableResizablePopup's, so the popup's
#: stylesheet doesn't reach these selectors — same reasoning as
#: _POPUP_STYLESHEET above, one level deeper.
_TABLE_STYLESHEET = """
.tabulator {
  border: 1px solid #c9d6e8;
  border-radius: 8px;
  overflow: hidden;
}

.tabulator-header {
  background: #2f5d8a !important;
  color: #ffffff;
  border-bottom: 2px solid #1f4266;
}

.tabulator-header .tabulator-col {
  /* Tabulator's own .tabulator-header .tabulator-col rule sets its own
     background, at the same specificity as a plain .tabulator-header
     rule — it must be overridden explicitly here, not just on the
     outer header, or the column cells hide the header color entirely. */
  background: #2f5d8a !important;
  color: #ffffff;
  border-right-color: #1f4266;
}

.tabulator-header .tabulator-col-title {
  color: #ffffff;
  font-weight: 650;
}

.tabulator-header .tabulator-col-sorter {
  color: #dce8f7 !important;
}

.tabulator-header .tabulator-col-sorter .tabulator-arrow {
  border-bottom-color: #dce8f7 !important;
  border-top-color: #dce8f7 !important;
}

.tabulator-header-filter input {
  border-radius: 4px;
  border: 1px solid #9fb6cf;
}

.tabulator-row {
  background: #ffffff !important;
  color: #17222f;
}

.tabulator-row.tabulator-row-even {
  background: #eaf1fa !important;
}

.tabulator-row:hover {
  background: #d3e6fb !important;
}

.tabulator-row.tabulator-selected {
  background: #ffe9a8 !important;
}

/* Action columns (Inspect/Remove/Clone/Export): each icon's <span> in the
   cell's "html"-formatted content carries its own class (see
   _ACTION_COLUMNS) -- styling is tied to :has() of that class rather than
   nth-last-child DOM-position counting, since the latter proved fragile
   once the "Actions" column group changed the surrounding structure (it
   only affects header markup, but position-based guessing was still easy
   to get wrong). No internal grid lines between action cells, a thick
   separator before the first and after the last, themed per-column
   background alternating with row parity, no sort UI, tooltip cursor. */
.tabulator-row .tabulator-cell:has(.lw-pardb-action) {
  cursor: pointer;
  text-align: center;
  border-right: none !important;
}

/* The "Actions" group's sub-columns (Inspect/Remove/Clone/Export) render
   nested one level deeper than plain data columns, under
   .tabulator-col-group-cols -- targeting them this way (rather than
   nth-last-child on .tabulator-col, which counts siblings within
   whichever parent a column happens to have) avoids also matching the
   last few *data* columns' headers now that the group wrapper occupies
   one slot among the top-level columns. */
.tabulator-header .tabulator-col-group-cols .tabulator-col-sorter {
  display: none !important;
}

/* Basic-color theme per action, keyed by the icon's own class (see
   _ACTION_COLUMNS): Inspect=cyan, Remove=red, Clone=green, Export=yellow. */
.tabulator-row .tabulator-cell:has(.lw-pardb-action-inspect) {
  border-left: 4px solid #1f4266 !important;
  background: #a5f3fc !important;
}
.tabulator-row.tabulator-row-even .tabulator-cell:has(.lw-pardb-action-inspect) {
  background: #67e8f9 !important;
}

.tabulator-row .tabulator-cell:has(.lw-pardb-action-remove) {
  background: #fecaca !important;
}
.tabulator-row.tabulator-row-even .tabulator-cell:has(.lw-pardb-action-remove) {
  background: #fca5a5 !important;
}

.tabulator-row .tabulator-cell:has(.lw-pardb-action-clone) {
  background: #bbf7d0 !important;
}
.tabulator-row.tabulator-row-even .tabulator-cell:has(.lw-pardb-action-clone) {
  background: #86efac !important;
}

.tabulator-row .tabulator-cell:has(.lw-pardb-action-export) {
  border-right: 4px solid #1f4266 !important;
  background: #fef08a !important;
}
.tabulator-row.tabulator-row-even .tabulator-cell:has(.lw-pardb-action-export) {
  background: #fde047 !important;
}

/* The header's "Actions" group is a single wrapper element
   (.tabulator-col.tabulator-col-group) spanning all four sub-columns, so
   the thick separator goes on its own left/right edges directly -- not
   on the individual Inspect/Export sub-column headers. */
.tabulator-header .tabulator-col.tabulator-col-group {
  border-left: 4px solid #1f4266 !important;
  border-right: 4px solid #1f4266 !important;
}

/* Hard clamp to the intended 100px: Tabulator's fitData-family layout
   engine treats an explicit column `width` as a minimum and grows a
   column past it to fit its own content (header title text included),
   which is why the action columns kept rendering oversized despite
   `widths={"Inspect": "100px", ...}` below. Tabulator sets its own
   computed width via an inline `style="width:...px"` on both the header
   sub-column and every row cell; a same-specificity class selector with
   `!important` beats an inline style lacking `!important`, so this wins
   regardless of what the layout engine computes -- applied identically
   to header and row cells so they stay aligned with each other. */
.tabulator-header .tabulator-col-group-cols .tabulator-col,
.tabulator-row .tabulator-cell:has(.lw-pardb-action) {
  width: 100px !important;
  min-width: 100px !important;
  max-width: 100px !important;
}
"""

#: The four per-row action columns, in this fixed order (appended after
#: all data columns) — icon HTML (rendered via the Tabulator "html"
#: formatter), each with a native-tooltip `title` attribute and its own
#: "lw-pardb-action-<name>" class (used by _TABLE_STYLESHEET's :has()
#: selectors, rather than guessing DOM sibling position), and their own
#: minimal fixed width.
_ACTION_COLUMNS: list[tuple[str, str, str]] = [
    (
        "Inspect",
        '<span class="lw-pardb-action lw-pardb-action-inspect" title="Inspect this parameter">🔍</span>',
        "100px",
    ),
    (
        "Remove",
        '<span class="lw-pardb-action lw-pardb-action-remove" title="Remove this parameter">🗑️</span>',
        "100px",
    ),
    (
        "Clone",
        '<span class="lw-pardb-action lw-pardb-action-clone" title="Clone parameter to add new one">📋</span>',
        "100px",
    ),
    (
        "Export",
        '<span class="lw-pardb-action lw-pardb-action-export" title="Export parameter configuration to file">📤</span>',
        "100px",
    ),
]


def build_parameter_db_content(
    detail_popup: "_DraggableResizablePopup", *, wide: bool = False
) -> pn.viewable.Viewable:
    """Build the Parameter Database popup body: a sortable/searchable table
    of every registered parameter (with per-row Inspect/Remove/Clone/Export
    actions), plus a fixed-width sidebar (an "Add parameter" button and a
    column-visibility picker).

    `detail_popup` is a second `_DraggableResizablePopup`, owned and placed
    by the caller (as a sibling of the main popup, not nested inside it) —
    this function only points its `title`/`body`/`visible` at the right
    content when a row action or Add parameter is clicked.

    `wide` selects the table's height: True for the standalone page (full
    browser width/height available, table grows to its full length, no
    inner scrollbar), False (default) for the portal-embedded popup
    (capped at ~1100px minus the sidebar, with an inner scrollbar since
    the popup itself has limited height). Data-column widths aren't
    hardcoded in either case -- `layout="fit_data_table"` (the default)
    auto-sizes each one to its actual rendered content, which fits real
    values better than a guessed pixel preset; only the four action
    columns (fixed-size icons, not data) get an explicit width below.
    """
    df = parameter_db_data.build_parameter_table_df()
    data_columns = list(df.columns)
    default_hidden = [
        c for c in parameter_db_data.DEFAULT_HIDDEN_COLUMNS if c in df.columns
    ]

    column_widths: dict[str, str] = {}
    for name, icon_html, width in _ACTION_COLUMNS:
        df[name] = icon_html
        column_widths[name] = width
    action_names = [name for name, _, _ in _ACTION_COLUMNS]

    table_kwargs: dict[str, Any] = dict(
        show_index=False,
        header_filters={c: True for c in data_columns},
        editors={col: None for col in df.columns},
        formatters={name: "html" for name in action_names},
        text_align={name: "center" for name in action_names},
        # Applies to every column's header (not just the action ones) --
        # a plain string, rather than a per-column dict, is the Panel
        # idiom that reliably reaches Tabulator.js's per-column
        # `headerHozAlign`; `configuration.columnDefaults` below is a
        # belt-and-suspenders fallback for anything it doesn't cover.
        header_align="center",
        # Action columns keep their own title text (Inspect/Remove/...)
        # as the second header row, under the merged "Actions" group row
        # -- the 100px width is enforced with a hard CSS clamp in
        # _TABLE_STYLESHEET instead of hiding this text, since Tabulator's
        # fitData-family layout treats `width` as a minimum and grows a
        # column to fit its content regardless of the title being present.
        configuration={
            "movableColumns": True,
            "columnDefaults": {"headerHozAlign": "center"},
        },
        widths={c: column_widths[c] for c in df.columns if c in column_widths},
        groups={"Actions": action_names},
        sizing_mode="stretch_width",
        hidden_columns=list(default_hidden),
        css_classes=["lw-parameter-db-table"],
        stylesheets=[_TABLE_STYLESHEET],
    )
    if not wide:
        # Portal popup: fixed height with its own scrollbar (limited screen
        # space). Standalone page: grow to the full row count instead, so
        # the page scrolls rather than the table having an inner slider.
        table_kwargs["height"] = 500
    table = pn.widgets.Tabulator(df, **table_kwargs)

    column_visibility = pn.widgets.CheckBoxGroup(
        options=data_columns,
        value=[c for c in data_columns if c not in default_hidden],
        margin=(4, 0),
    )

    def _update_hidden_columns(event: object) -> None:
        table.hidden_columns = [
            c for c in data_columns if c not in column_visibility.value
        ]

    column_visibility.param.watch(_update_hidden_columns, "value")

    columns_section = pn.Column(
        pn.pane.Markdown("**Columns**", margin=(0, 0, 4, 0)),
        column_visibility,
        margin=(8, 0, 0, 0),
        sizing_mode="stretch_width",
    )

    def _hide_detail() -> None:
        detail_popup.visible = False

    def _open_inspect(k: str) -> None:
        instance = get_param_instance(k)
        detail_popup.title = f"Parameter: {instance.disp}"
        detail_popup.theme_class = "lw-parameter-db-panel-surface--cyan"
        detail_popup.size_modifier_class = "lw-parameter-db-panel-surface--compact"
        detail_popup.body[:] = [build_param_detail_popup(k, editable=False)]
        detail_popup.visible = True

    def _open_remove_confirm(k: str) -> None:
        detail_popup.title = f"Remove Parameter {k}"
        detail_popup.theme_class = "lw-parameter-db-panel-surface--red"
        detail_popup.size_modifier_class = "lw-parameter-db-panel-surface--confirm"
        detail_popup.body[:] = [_build_remove_confirm_popup(k, table, _hide_detail)]
        detail_popup.visible = True

    def _open_add(*, initial_source_k: Optional[str] = None) -> None:
        detail_popup.title = "Add Parameter"
        detail_popup.theme_class = "lw-parameter-db-panel-surface--green"
        detail_popup.size_modifier_class = "lw-parameter-db-panel-surface--compact"
        detail_popup.body[:] = [
            _build_add_parameter_popup(
                table, _hide_detail, initial_source_k=initial_source_k
            )
        ]
        detail_popup.visible = True

    def _add_parameter(_: object) -> None:
        _open_add()

    def _row_key(row_index: int) -> Optional[str]:
        try:
            return str(table.value.iloc[row_index]["Key"])
        except Exception:
            return None

    def _on_inspect_icon_click(event: object) -> None:
        k = _row_key(event.row)
        if k is not None:
            _open_inspect(k)

    def _on_remove_icon_click(event: object) -> None:
        k = _row_key(event.row)
        if k is not None:
            _open_remove_confirm(k)

    def _on_clone_icon_click(event: object) -> None:
        k = _row_key(event.row)
        if k is not None:
            _open_add(initial_source_k=k)

    # Not placed in the visible sidebar (per-row Export icon replaces the
    # standalone download button) -- kept only as the mechanism that
    # actually produces a browser download: FileDownload._transfer is a
    # param.depends('_clicks', watch=True) watcher, so incrementing
    # _clicks from Python re-fires it exactly as a real click would,
    # pushing fresh base64 file data to the frontend, which auto-triggers
    # the download (FileDownload's default auto=True).
    export_download = pn.widgets.FileDownload(
        callback=lambda: io.BytesIO(
            save_param_config(get_param_instance(export_state["k"]))
        ),
        filename="parameter_config.pkl",
        visible=False,
    )
    export_state: dict[str, Optional[str]] = {"k": None}

    def _on_export_icon_click(event: object) -> None:
        k = _row_key(event.row)
        if k is None:
            return
        export_state["k"] = k
        export_download.filename = f"{k}_config.pkl"
        export_download._clicks += 1
        # Best-effort: also persist a copy into the active workspace's
        # "parameters" folder (the default save location for exports) --
        # a no-op if there's no active workspace (e.g. the standalone
        # launcher), so it never blocks the browser download above.
        save_param_config_to_workspace(get_param_instance(k))

    table.on_click(_on_inspect_icon_click, column="Inspect")
    table.on_click(_on_remove_icon_click, column="Remove")
    table.on_click(_on_clone_icon_click, column="Clone")
    table.on_click(_on_export_icon_click, column="Export")

    add_button = pn.widgets.Button(
        name="Add parameter",
        button_type="success",
        sizing_mode="stretch_width",
        margin=0,
    )
    add_button.on_click(_add_parameter)

    sidebar = pn.Column(
        add_button,
        columns_section,
        export_download,  # visible=False: not shown, only drives the download
        css_classes=["lw-parameter-db-sidebar"],
        width=220,
        margin=0,
    )
    table_area = pn.Column(
        table,
        css_classes=["lw-parameter-db-table-area"],
        margin=0,
        sizing_mode="stretch_width",
    )

    return pn.Row(
        table_area,
        sidebar,
        css_classes=["lw-parameter-db-content"],
        margin=0,
        sizing_mode="stretch_width",
    )


def _build_parameter_db_dropdown() -> pn.viewable.Viewable:
    # Same trigger-shell/toggle wiring as the about/workspace triggers, see
    # larvaworld.portal.panel_components._build_about_dropdown. The icon
    # helpers stay in panel_components (they share its media/icon-loading
    # infrastructure), so they're imported lazily here to avoid a circular
    # import at module load time.
    from larvaworld.portal.panel_components import _parameter_db_button_icon_html

    parameter_db_led = pn.pane.HTML(_parameter_db_button_icon_html(), margin=0)
    parameter_db_button = pn.widgets.Button(
        name="",
        margin=0,
        css_classes=["lw-parameter-db-trigger-button"],
    )
    parameter_db_trigger_view = pn.Column(
        parameter_db_led,
        parameter_db_button,
        margin=0,
        width=22,
        height=22,
        css_classes=["lw-parameter-db-trigger-shell"],
    )
    parameter_db_panel = _DraggableResizablePopup(
        body=pn.Column(margin=0),
        visible=False,
        css_classes=["lw-parameter-db-dropdown-panel", _LARGE_PANEL_MODIFIER_CLASS],
        margin=0,
    )
    # Sibling of parameter_db_panel (not nested inside its body) so that
    # toggling it doesn't rely on a reactive update propagating through two
    # layers of ReactiveHTML embedding.
    detail_popup = _build_detail_popup()
    _content_built = {"done": False}

    def _toggle_parameter_db(_: object) -> None:
        # Build the (Tabulator-backed) content on first open rather than at
        # header-construction time: widgets measured/sized while the popup
        # is still `visible=False` (display:none) can end up with incorrect
        # dimensions once revealed.
        if not _content_built["done"]:
            parameter_db_panel.body = build_parameter_db_content(detail_popup)
            _content_built["done"] = True
        parameter_db_panel.visible = not parameter_db_panel.visible

    parameter_db_button.on_click(_toggle_parameter_db)

    return pn.Column(
        parameter_db_trigger_view,
        parameter_db_panel,
        detail_popup,
        css_classes=["lw-parameter-db-dropdown-wrap"],
        margin=(0, 0, 0, 36),
        sizing_mode="fixed",
        width_policy="min",
    )


def build_standalone_page() -> pn.viewable.Viewable:
    """Build a full-page view of the Parameter Database, laid out directly
    (no header trigger/dropdown chrome) for standalone launch via
    `python -m larvaworld.portal.parameter_database`, independent of the
    rest of the portal."""
    detail_popup = _build_detail_popup()
    title = pn.pane.Markdown(
        "## Parameter Database",
        margin=(16, 16, 4, 16),
        styles={"text-align": "center"},
        sizing_mode="stretch_width",
    )
    return pn.Column(
        title,
        build_parameter_db_content(detail_popup, wide=True),
        detail_popup,
        sizing_mode="stretch_width",
        margin=0,
    )
