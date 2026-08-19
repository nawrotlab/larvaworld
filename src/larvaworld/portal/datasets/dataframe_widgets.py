"""Reusable read-only Panel tables for LarvaDataset step and endpoint data."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
import panel as pn
import param

from larvaworld.lib.param.custom import EndpointDataFrame, StepDataFrame

if TYPE_CHECKING:
    from larvaworld.lib.process.dataset import LarvaDataset

__all__: list[str] = [
    "EndpointDataFrameTable",
    "LarvaDatasetTablesWidget",
    "StepDataFrameTable",
]


_POPUP_STYLESHEET = """
.lw-dataset-table-popup {
  position: fixed;
  top: 9vh;
  width: min(720px, calc(50vw - 36px));
  height: min(76vh, 780px);
  min-width: 440px;
  min-height: 360px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  resize: both;
  border: 1px solid rgba(15, 23, 42, 0.24);
  border-radius: 12px;
  background: #ffffff;
  box-shadow: 0 16px 36px rgba(15, 23, 42, 0.28);
  color: #111111;
  z-index: 2100;
  box-sizing: border-box;
}

.lw-dataset-table-popup--step {
  left: 24px;
}

.lw-dataset-table-popup--endpoint {
  right: 24px;
}

.lw-dataset-table-popup-header {
  position: relative;
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  min-height: 42px;
  padding: 0 10px 0 16px;
  border-bottom: 1px solid rgba(15, 23, 42, 0.12);
  background: rgba(15, 23, 42, 0.04);
  cursor: grab;
  user-select: none;
}

.lw-dataset-table-popup-header:active {
  cursor: grabbing;
}

.lw-dataset-table-popup-title {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 15px;
  font-weight: 650;
  white-space: nowrap;
}

.lw-dataset-table-popup-close {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: #111111;
  font-size: 22px;
  line-height: 1;
  cursor: pointer;
  z-index: 1;
}

.lw-dataset-table-popup-close:hover {
  background: rgba(15, 23, 42, 0.10);
}

.lw-dataset-table-popup-body {
  flex: 1 1 auto;
  min-height: 0;
  overflow: hidden;
  padding: 12px;
  box-sizing: border-box;
}

@media (max-width: 900px) {
  .lw-dataset-table-popup {
    top: 6vh;
    width: calc(100vw - 32px);
    max-width: none;
    min-width: 0;
    height: 82vh;
  }

  .lw-dataset-table-popup--step,
  .lw-dataset-table-popup--endpoint {
    right: 16px;
    left: 16px;
  }
}
"""

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


def _index_levels(
    parameter_type: type[StepDataFrame | EndpointDataFrame],
) -> tuple[str, ...]:
    """Read the canonical index-level names from a core dataframe parameter."""
    parameter = parameter_type(allow_None=True)
    return tuple(parameter.levels or ())


class _FloatingTablePopup(pn.reactive.ReactiveHTML):
    """A floating, draggable, independently resizable in-app table window."""

    body = param.Parameter()
    title = param.String(default="Dataset table")
    position_class = param.String(default="")
    stylesheets = param.List(default=[_POPUP_STYLESHEET])

    _template = """
    <div id="popup_surface" class="lw-dataset-table-popup ${position_class}"
         onmousedown="${script('bring_to_front')}">
      <div id="drag_handle" class="lw-dataset-table-popup-header"
           onmousedown="${script('start_drag')}">
        <span class="lw-dataset-table-popup-title">${title}</span>
        <button id="close_button" type="button" class="lw-dataset-table-popup-close"
                onclick="${_close_button_click}">×</button>
      </div>
      <div id="body_slot" class="lw-dataset-table-popup-body">${body}</div>
    </div>
    """

    _scripts = {
        "bring_to_front": """
          window.__larvaworldDatasetTablePopupZ =
            (window.__larvaworldDatasetTablePopupZ || 2100) + 1;
          popup_surface.style.zIndex = window.__larvaworldDatasetTablePopupZ;
        """,
        "start_drag": """
          if (event.target.closest('button')) {
            return;
          }
          const surface = popup_surface;
          const rect = surface.getBoundingClientRect();
          surface.style.position = 'fixed';
          surface.style.margin = '0';
          surface.style.right = 'auto';
          surface.style.transform = 'none';
          surface.style.left = rect.left + 'px';
          surface.style.top = rect.top + 'px';
          const startX = event.clientX;
          const startY = event.clientY;
          const baseLeft = rect.left;
          const baseTop = rect.top;
          function onMove(moveEvent) {
            surface.style.left = (baseLeft + moveEvent.clientX - startX) + 'px';
            surface.style.top = (baseTop + moveEvent.clientY - startY) + 'px';
          }
          function onUp() {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
          }
          document.addEventListener('mousemove', onMove);
          document.addEventListener('mouseup', onUp);
        """,
    }

    def _close_button_click(self, _event: object) -> None:
        self.visible = False


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
        )
        self.endpoint_button = pn.widgets.Button(
            name="Endpoint",
            button_type="default",
            width=100,
            disabled=True,
        )
        self._step_error = pn.pane.HTML(margin=(0, 0, 8, 0), visible=False)
        self._endpoint_error = pn.pane.HTML(margin=(0, 0, 8, 0), visible=False)
        self.step_popup = _FloatingTablePopup(
            title="Step data",
            body=pn.Column(
                self._step_error,
                self.step_table.view(),
                sizing_mode="stretch_width",
                margin=0,
            ),
            position_class="lw-dataset-table-popup--step",
            visible=False,
        )
        self.endpoint_popup = _FloatingTablePopup(
            title="Endpoint data",
            body=pn.Column(
                self._endpoint_error,
                self.endpoint_table.view(),
                sizing_mode="stretch_width",
                margin=0,
            ),
            position_class="lw-dataset-table-popup--endpoint",
            visible=False,
        )
        self._view = pn.Column(
            pn.Row(self.step_button, self.endpoint_button, margin=0),
            self.step_popup,
            self.endpoint_popup,
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
        self.step_popup.visible = False
        self.endpoint_popup.visible = False
        self.step_popup.title = self._popup_title("Step data")
        self.endpoint_popup.title = self._popup_title("Endpoint data")
        self._clear_error(self._step_error)
        self._clear_error(self._endpoint_error)
        self.step_table.clear()
        self.endpoint_table.clear()

    def view(self) -> pn.viewable.Viewable:
        """Return buttons and both floating popup windows as one viewable."""
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

    def _open_table(
        self,
        *,
        table: _DataFrameTable,
        popup: _FloatingTablePopup,
        error_pane: pn.pane.HTML,
        attribute: str,
    ) -> None:
        popup.visible = True
        self._clear_error(error_pane)
        if self._dataset is None:
            table.clear(message="Select a dataset before opening this table.")
            return
        try:
            table.update(getattr(self._dataset, attribute))
        except Exception as exc:
            table.clear(message="The table could not be loaded.")
            self._show_error(error_pane, exc)
