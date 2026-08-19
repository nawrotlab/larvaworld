from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import panel as pn
import pytest

from larvaworld.portal.datasets import (
    EndpointDataFrameTable,
    LarvaDatasetTablesWidget,
    StepDataFrameTable,
)


@pytest.fixture
def step_dataframe() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [[0, 1], ["Larva_1", "Larva_2"]], names=["Step", "AgentID"]
    )
    return pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}, index=index)


@pytest.fixture
def endpoint_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {"cum_d": [1.5, 2.5]},
        index=pd.Index(["Larva_1", "Larva_2"], name="AgentID"),
    )


class _LazyDataset:
    def __init__(self, step: pd.DataFrame, endpoint: pd.DataFrame) -> None:
        self._step = step
        self._endpoint = endpoint
        self.loaded: list[str] = []
        self.config = SimpleNamespace(id="example-dataset")

    @property
    def s(self) -> pd.DataFrame:
        self.loaded.append("s")
        return self._step

    @property
    def e(self) -> pd.DataFrame:
        self.loaded.append("e")
        return self._endpoint


class _FailingStepDataset(_LazyDataset):
    @property
    def s(self) -> pd.DataFrame:
        self.loaded.append("s")
        raise OSError("step data is unavailable")


def test_step_table_accepts_canonical_schema_without_mutation(
    step_dataframe: pd.DataFrame,
) -> None:
    original_index = step_dataframe.index.copy()
    table = StepDataFrameTable(step_dataframe)

    assert table.dataframe is step_dataframe
    assert table.table.value is step_dataframe
    assert table.table.indexes == ["Step", "AgentID"]
    assert table.table.frozen_columns == ["Step", "AgentID"]
    assert table.table.pagination == "remote"
    assert table.table.page_size == 50
    assert table.table.header_filters is True
    assert table.table.selectable is False
    assert table.table.editors == {"x": None}
    assert step_dataframe.index.equals(original_index)
    assert list(step_dataframe.columns) == ["x"]


def test_endpoint_table_accepts_canonical_schema(
    endpoint_dataframe: pd.DataFrame,
) -> None:
    table = EndpointDataFrameTable(endpoint_dataframe)

    assert table.dataframe is endpoint_dataframe
    assert table.table.indexes == ["AgentID"]
    assert table.table.frozen_columns == ["AgentID"]
    assert "2 rows" in table.summary.object


@pytest.mark.parametrize(
    ("table_type", "dataframe"),
    [
        (
            StepDataFrameTable,
            pd.DataFrame(
                {"x": [1.0]},
                index=pd.MultiIndex.from_tuples(
                    [("Larva_1", 0)], names=["AgentID", "Step"]
                ),
            ),
        ),
        (
            EndpointDataFrameTable,
            pd.DataFrame({"cum_d": [1.0]}, index=pd.Index(["Larva_1"])),
        ),
    ],
)
def test_table_rejects_wrong_index_schema(
    table_type: type[StepDataFrameTable] | type[EndpointDataFrameTable],
    dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(TypeError, match="requires index levels"):
        table_type(dataframe)


def test_empty_table_displays_empty_state() -> None:
    empty = pd.DataFrame(
        {"x": pd.Series(dtype=float)},
        index=pd.MultiIndex.from_arrays([[], []], names=["Step", "AgentID"]),
    )
    table = StepDataFrameTable(empty)

    assert table.empty_state.visible is True
    assert table.table.visible is False
    assert "No step data" in table.empty_state.object


def test_dataset_widget_loads_each_dataframe_only_on_its_button(
    step_dataframe: pd.DataFrame,
    endpoint_dataframe: pd.DataFrame,
) -> None:
    dataset = _LazyDataset(step_dataframe, endpoint_dataframe)
    widget = LarvaDatasetTablesWidget(dataset)

    assert dataset.loaded == []
    assert widget.step_button.disabled is False
    assert widget.endpoint_button.disabled is False

    widget.step_button.clicks += 1

    assert dataset.loaded == ["s"]
    assert widget.step_popup.visible is True
    assert widget.endpoint_popup.visible is False
    assert widget.step_table.dataframe is step_dataframe
    assert widget.endpoint_table.dataframe is None

    widget.endpoint_button.clicks += 1

    assert dataset.loaded == ["s", "e"]
    assert widget.step_popup.visible is True
    assert widget.endpoint_popup.visible is True
    assert widget.endpoint_table.dataframe is endpoint_dataframe


def test_dataset_widget_releases_stale_data_when_dataset_changes(
    step_dataframe: pd.DataFrame,
    endpoint_dataframe: pd.DataFrame,
) -> None:
    widget = LarvaDatasetTablesWidget(_LazyDataset(step_dataframe, endpoint_dataframe))
    widget._open_step()
    widget._open_endpoint()

    next_step = step_dataframe.copy()
    next_step["x"] = 99.0
    next_dataset = _LazyDataset(next_step, endpoint_dataframe)
    widget.set_dataset(next_dataset)

    assert next_dataset.loaded == []
    assert widget.step_popup.visible is False
    assert widget.endpoint_popup.visible is False
    assert widget.step_table.dataframe is None
    assert widget.endpoint_table.dataframe is None

    widget._open_step()

    assert widget.step_table.dataframe is next_step

    widget.set_dataset(None)

    assert widget.step_button.disabled is True
    assert widget.endpoint_button.disabled is True
    assert widget.step_popup.visible is False
    assert widget.endpoint_popup.visible is False
    assert widget.step_table.dataframe is None
    assert widget.endpoint_table.dataframe is None


def test_dataset_widget_keeps_popup_open_and_reports_load_error(
    endpoint_dataframe: pd.DataFrame,
) -> None:
    bad_step = pd.DataFrame({"x": [1.0]}, index=pd.Index(["Larva_1"], name="AgentID"))
    widget = LarvaDatasetTablesWidget(_LazyDataset(bad_step, endpoint_dataframe))

    widget._open_step()

    assert widget.step_popup.visible is True
    assert widget.step_table.dataframe is None
    assert widget._step_error.visible is True
    assert "Table unavailable" in widget._step_error.object


def test_dataset_widget_reports_a_lazy_loading_exception(
    step_dataframe: pd.DataFrame,
    endpoint_dataframe: pd.DataFrame,
) -> None:
    widget = LarvaDatasetTablesWidget(
        _FailingStepDataset(step_dataframe, endpoint_dataframe)
    )

    widget._open_step()

    assert widget.step_popup.visible is True
    assert widget._step_error.visible is True
    assert "OSError: step data is unavailable" in widget._step_error.object


def test_dataset_widget_closes_each_popup_independently(
    step_dataframe: pd.DataFrame,
    endpoint_dataframe: pd.DataFrame,
) -> None:
    widget = LarvaDatasetTablesWidget(_LazyDataset(step_dataframe, endpoint_dataframe))
    widget._open_step()
    widget._open_endpoint()

    widget.step_popup._close_button_click(None)

    assert widget.step_popup.visible is False
    assert widget.endpoint_popup.visible is True


def test_widget_view_renders_as_a_panel_viewable(
    step_dataframe: pd.DataFrame,
    endpoint_dataframe: pd.DataFrame,
) -> None:
    pn.extension("tabulator")
    widget = LarvaDatasetTablesWidget(_LazyDataset(step_dataframe, endpoint_dataframe))

    assert pn.Column(widget.view()).get_root() is not None
