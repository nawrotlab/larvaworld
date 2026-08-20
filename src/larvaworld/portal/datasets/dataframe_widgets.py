"""Reusable read-only Panel tables for LarvaDataset step and endpoint data."""

from __future__ import annotations

from html import escape
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
import panel as pn

from larvaworld.lib.param.custom import EndpointDataFrame, StepDataFrame

if TYPE_CHECKING:
    from larvaworld.lib.process.dataset import LarvaDataset

__all__: list[str] = [
    "EndpointDataFrameTable",
    "LarvaDatasetTablesWidget",
    "StepDataFrameTable",
]


_TABLE_STYLESHEET = """
.tabulator {
  border: 1px solid rgba(15, 23, 42, 0.14);
  border-radius: 8px;
  overflow: hidden;
  font-size: 12px;
}

.tabulator .tabulator-col,
.tabulator .tabulator-cell {
  font-size: 12px;
}
"""


def _button_tooltip_stylesheet(text: str) -> str:
    """Return a self-contained hover tooltip stylesheet for a Panel button."""
    return f"""
:host {{
  overflow: visible !important;
}}

.bk-btn {{
  position: relative;
  overflow: visible !important;
}}

.bk-btn:hover::after {{
  content: "{text}";
  position: absolute;
  left: 50%;
  bottom: calc(100% + 7px);
  transform: translateX(-50%);
  z-index: 10020;
  padding: 5px 8px;
  border-radius: 4px;
  background: rgba(15, 23, 42, 0.92);
  color: #ffffff;
  font-size: 11px;
  font-weight: 500;
  line-height: 1.25;
  white-space: nowrap;
  pointer-events: none;
  box-shadow: 0 4px 10px rgba(15, 23, 42, 0.2);
}}
"""


def _index_levels(
    parameter_type: type[StepDataFrame | EndpointDataFrame],
) -> tuple[str, ...]:
    """Read the canonical index-level names from a core dataframe parameter."""
    parameter = parameter_type(allow_None=True)
    return tuple(parameter.levels or ())


def _popup_drag_handle_html(title: str) -> str:
    """Return a plain HTML drag handle without a custom Bokeh model."""
    return f"""
    <div style="height:32px;display:flex;align-items:center;padding:0 4px;
                box-sizing:border-box;font-weight:650;cursor:grab;user-select:none;"
         onmousedown="
           if (event.button !== 0) return;
           let node = this;
           let popup = null;
           while (node &amp;&amp; !popup) {{
             popup = node.closest ? node.closest('.lw-dataset-table-popup') : null;
             if (popup) break;
             const root = node.getRootNode ? node.getRootNode() : null;
             node = root &amp;&amp; root.host ? root.host : null;
           }}
           if (!popup) return;
           event.preventDefault();
           this.style.cursor = 'grabbing';
           window.__larvaworldDatasetTablePopupZ =
             (window.__larvaworldDatasetTablePopupZ || 10000) + 1;
           popup.style.zIndex = window.__larvaworldDatasetTablePopupZ;
           const rect = popup.getBoundingClientRect();
           const startX = event.clientX;
           const startY = event.clientY;
           const baseLeft = rect.left;
           const baseTop = rect.top;
           popup.style.left = rect.left + 'px';
           popup.style.right = 'auto';
           popup.style.top = rect.top + 'px';
           const handle = this;
           const move = function(moveEvent) {{
             const maxLeft = Math.max(0, window.innerWidth - popup.offsetWidth);
             const maxTop = Math.max(0, window.innerHeight - popup.offsetHeight);
             const left = Math.min(
               maxLeft,
               Math.max(0, baseLeft + moveEvent.clientX - startX)
             );
             const top = Math.min(
               maxTop,
               Math.max(0, baseTop + moveEvent.clientY - startY)
             );
             popup.style.left = left + 'px';
             popup.style.top = top + 'px';
           }};
           const stop = function() {{
             handle.style.cursor = 'grab';
             document.removeEventListener('mousemove', move);
             document.removeEventListener('mouseup', stop);
             window.removeEventListener('blur', stop);
           }};
           document.addEventListener('mousemove', move);
           document.addEventListener('mouseup', stop);
           window.addEventListener('blur', stop);
         ">{escape(title)}</div>
    """


class _FloatingTablePopup(pn.Column):
    """Dependency-free floating Panel window for one dataset table."""

    def __init__(
        self,
        *,
        title: str,
        body: pn.viewable.Viewable,
        position_class: str,
        visible: bool = False,
    ) -> None:
        drag_handle = pn.pane.HTML(
            _popup_drag_handle_html(title),
            margin=0,
            sizing_mode="stretch_width",
            height=32,
            sanitize_html=False,
        )
        close_button = pn.widgets.Button(
            name="×",
            button_type="light",
            width=36,
            height=32,
            margin=0,
        )
        body_slot = pn.Column(
            body,
            sizing_mode="stretch_both",
            margin=0,
            styles={"overflow": "auto"},
        )
        position_styles = (
            {"left": "24px"} if position_class.endswith("--step") else {"right": "24px"}
        )
        super().__init__(
            pn.Row(
                drag_handle,
                close_button,
                sizing_mode="stretch_width",
                margin=0,
                styles={
                    "align-items": "center",
                    "border-bottom": "1px solid rgba(15, 23, 42, 0.14)",
                    "padding": "8px 10px",
                },
            ),
            body_slot,
            css_classes=["lw-dataset-table-popup", position_class],
            width=720,
            height=600,
            sizing_mode="fixed",
            margin=0,
            styles={
                "position": "fixed",
                "top": "72px",
                "z-index": "10000",
                "background": "#ffffff",
                "border": "1px solid rgba(15, 23, 42, 0.22)",
                "border-radius": "12px",
                "box-shadow": "0 18px 48px rgba(15, 23, 42, 0.28)",
                "overflow": "hidden",
                "resize": "both",
                "min-width": "360px",
                "min-height": "280px",
                "max-width": "calc(100vw - 48px)",
                "max-height": "calc(100vh - 96px)",
                **position_styles,
            },
            visible=visible,
        )
        self._drag_handle = drag_handle
        self._close_button = close_button
        self._body_slot = body_slot
        self._title = title
        self._close_callback: Callable[[], None] | None = None
        self._close_button.on_click(self._close_button_click)

    @property
    def body(self) -> pn.viewable.Viewable:
        """Return the single table body hosted by the floating panel."""
        return self._body_slot.objects[0]

    @body.setter
    def body(self, value: pn.viewable.Viewable) -> None:
        self._body_slot.objects = [value]

    @property
    def title(self) -> str:
        """Return the title shown in the floating-panel header."""
        return self._title

    @title.setter
    def title(self, value: str) -> None:
        self._title = value
        self._drag_handle.object = _popup_drag_handle_html(value)

    def set_close_callback(self, callback: Callable[[], None]) -> None:
        """Set the host callback that unmounts this popup from its document."""
        self._close_callback = callback

    def _close_button_click(self, _event: object) -> None:
        """Close this window without affecting the other dataset table."""
        if self._close_callback is None:
            self.visible = False
            return
        self._close_callback()


class _DataFrameTable:
    """Shared read-only Tabulator implementation for a validated dataframe."""

    expected_index_levels: ClassVar[tuple[str, ...]]
    label: ClassVar[str]

    def __init__(self, dataframe: pd.DataFrame | None = None, *, page_size: int = 50):
        if page_size < 1:
            raise ValueError("page_size must be greater than zero.")
        self.page_size = int(page_size)
        self._dataframe: pd.DataFrame | None = None
        self.summary = pn.pane.HTML(margin=(0, 0, 8, 0))
        self.empty_state = pn.pane.HTML(margin=0)
        self.table = pn.widgets.Tabulator(
            pd.DataFrame(),
            show_index=True,
            selectable=False,
            editors={},
            header_filters=True,
            pagination="remote",
            page_size=self.page_size,
            height=600,
            layout="fit_data_table",
            frozen_columns=list(self.expected_index_levels),
            sizing_mode="stretch_width",
            stylesheets=[_TABLE_STYLESHEET],
            visible=False,
        )
        self._view = pn.Column(
            self.summary,
            self.empty_state,
            self.table,
            sizing_mode="stretch_width",
            margin=0,
        )
        self.clear()
        if dataframe is not None:
            self.update(dataframe)

    @property
    def dataframe(self) -> pd.DataFrame | None:
        """The original dataframe reference, never copied or mutated by this table."""
        return self._dataframe

    @classmethod
    def validate(cls, dataframe: pd.DataFrame) -> None:
        """Validate the dataframe type and its exact canonical index schema."""
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"{cls.label} must be a pandas DataFrame.")
        actual = tuple(dataframe.index.names)
        if actual != cls.expected_index_levels:
            raise TypeError(
                f"{cls.label} requires index levels {list(cls.expected_index_levels)!r}, "
                f"not {list(actual)!r}."
            )

    def update(self, dataframe: pd.DataFrame) -> None:
        """Display a validated dataframe without altering it or its index."""
        self.validate(dataframe)
        self._dataframe = dataframe
        self.table.value = dataframe
        self.table.editors = {column: None for column in dataframe.columns}
        self.table.frozen_columns = list(self.expected_index_levels)
        self.summary.object = (
            "<strong>"
            f"{escape(self.label)}</strong> · {len(dataframe):,} rows · "
            f"{len(dataframe.columns):,} data columns"
        )
        is_empty = dataframe.empty
        self.empty_state.object = (
            f"<em>No {escape(self.label.lower())} are available.</em>"
            if is_empty
            else ""
        )
        self.empty_state.visible = is_empty
        self.table.visible = not is_empty

    def clear(self, *, message: str | None = None) -> None:
        """Drop references to the current dataframe and show an empty state."""
        self._dataframe = None
        self.table.value = pd.DataFrame()
        self.table.editors = {}
        self.summary.object = ""
        self.empty_state.object = (
            f"<em>{escape(message)}</em>"
            if message
            else f"<em>No {escape(self.label.lower())} are loaded.</em>"
        )
        self.empty_state.visible = True
        self.table.visible = False

    def view(self) -> pn.viewable.Viewable:
        """Return the reusable Panel view for this table."""
        return self._view


class StepDataFrameTable(_DataFrameTable):
    """Read-only, paginated inspection table for ``LarvaDataset.s``."""

    expected_index_levels = _index_levels(StepDataFrame)
    label = "Step data"


class EndpointDataFrameTable(_DataFrameTable):
    """Read-only, paginated inspection table for ``LarvaDataset.e``."""

    expected_index_levels = _index_levels(EndpointDataFrame)
    label = "Endpoint data"


class LarvaDatasetTablesWidget:
    """Open independent Step and Endpoint inspection windows for one dataset."""

    def __init__(
        self,
        dataset: LarvaDataset | None = None,
        *,
        page_size: int = 50,
    ) -> None:
        self._dataset: LarvaDataset | None = None
        self.step_table = StepDataFrameTable(page_size=page_size)
        self.endpoint_table = EndpointDataFrameTable(page_size=page_size)
        self.step_button = pn.widgets.Button(
            name="Step",
            button_type="default",
            width=100,
            disabled=True,
            stylesheets=[
                _button_tooltip_stylesheet("Step-by-step parameter timeseries")
            ],
        )
        self.endpoint_button = pn.widgets.Button(
            name="Endpoint",
            button_type="default",
            width=100,
            disabled=True,
            stylesheets=[_button_tooltip_stylesheet("Endpoint parameter measurements")],
        )
        self._step_error = pn.pane.HTML(margin=(0, 0, 8, 0), visible=False)
        self._endpoint_error = pn.pane.HTML(margin=(0, 0, 8, 0), visible=False)
        self._step_popup_body = pn.Column(
            self._step_error,
            self.step_table.view(),
            sizing_mode="stretch_width",
            margin=0,
        )
        self._endpoint_popup_body = pn.Column(
            self._endpoint_error,
            self.endpoint_table.view(),
            sizing_mode="stretch_width",
            margin=0,
        )
        self.step_popup = _FloatingTablePopup(
            title="Step data",
            body=self._step_popup_body,
            position_class="lw-dataset-table-popup--step",
            visible=False,
        )
        self.endpoint_popup = _FloatingTablePopup(
            title="Endpoint data",
            body=self._endpoint_popup_body,
            position_class="lw-dataset-table-popup--endpoint",
            visible=False,
        )
        self.step_popup.set_close_callback(lambda: self._unmount_popup(self.step_popup))
        self.endpoint_popup.set_close_callback(
            lambda: self._unmount_popup(self.endpoint_popup)
        )
        self._view = pn.Column(
            pn.Row(self.step_button, self.endpoint_button, margin=0),
            sizing_mode="stretch_width",
            margin=0,
        )
        self.step_button.on_click(self._open_step)
        self.endpoint_button.on_click(self._open_endpoint)
        self.set_dataset(dataset)

    @property
    def dataset(self) -> LarvaDataset | None:
        """The dataset currently supplied to the two table actions."""
        return self._dataset

    def set_dataset(self, dataset: LarvaDataset | None) -> None:
        """Replace the dataset and clear stale tables and open windows."""
        self._dataset = dataset
        self.step_button.disabled = dataset is None
        self.endpoint_button.disabled = dataset is None
        self._unmount_popup(self.step_popup)
        self._unmount_popup(self.endpoint_popup)
        self.step_popup.title = self._popup_title("Step data")
        self.endpoint_popup.title = self._popup_title("Endpoint data")
        self._clear_error(self._step_error)
        self._clear_error(self._endpoint_error)
        self.step_table.clear()
        self.endpoint_table.clear()

    def view(self) -> pn.viewable.Viewable:
        """Return buttons and any floating table windows currently open."""
        return self._view

    def _popup_title(self, title: str) -> str:
        if self._dataset is None:
            return title
        dataset_id = getattr(getattr(self._dataset, "config", None), "id", None)
        return f"{title} — {dataset_id}" if dataset_id else title

    @staticmethod
    def _clear_error(error_pane: pn.pane.HTML) -> None:
        error_pane.object = ""
        error_pane.visible = False

    @staticmethod
    def _show_error(error_pane: pn.pane.HTML, exc: Exception) -> None:
        error_pane.object = (
            '<div style="color:#9b1c1c;line-height:1.45;">'
            "<strong>Table unavailable.</strong><br/>"
            f"{escape(f'{type(exc).__name__}: {exc}')}"
            "</div>"
        )
        error_pane.visible = True

    def _open_step(self, _event: object | None = None) -> None:
        self._open_table(
            table=self.step_table,
            popup=self.step_popup,
            error_pane=self._step_error,
            attribute="s",
        )

    def _open_endpoint(self, _event: object | None = None) -> None:
        self._open_table(
            table=self.endpoint_table,
            popup=self.endpoint_popup,
            error_pane=self._endpoint_error,
            attribute="e",
        )

    def _mount_popup(self, popup: _FloatingTablePopup) -> None:
        """Attach the standard Panel popup only while its table is open."""
        if popup not in self._view.objects:
            self._view.append(popup)
        popup.visible = True

    def _unmount_popup(self, popup: _FloatingTablePopup) -> None:
        """Remove a popup and its table models from the current document."""
        if popup in self._view.objects:
            self._view.remove(popup)
        popup.visible = False

    def _open_table(
        self,
        *,
        table: _DataFrameTable,
        popup: _FloatingTablePopup,
        error_pane: pn.pane.HTML,
        attribute: str,
    ) -> None:
        self._clear_error(error_pane)
        if self._dataset is None:
            table.clear(message="Select a dataset before opening this table.")
        else:
            try:
                table.update(getattr(self._dataset, attribute))
            except Exception as exc:
                table.clear(message="The table could not be loaded.")
                self._show_error(error_pane, exc)
        self._mount_popup(popup)
