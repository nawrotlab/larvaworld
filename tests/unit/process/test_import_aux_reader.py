import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from larvaworld.lib.process import import_aux


@pytest.mark.fast
def test_read_timeseries_from_raw_files_per_parameter(tmp_path):
    """
    Verify that the legacy Jovanic-format reader assembles columns, applies defaults,
    and computes Step values when tracker metadata is provided.
    """
    prefix = tmp_path / "jovanic" / "ProteinDeprivation" / "Fed" / "sample"
    prefix.parent.mkdir(parents=True)

    # Two larvae, two timesteps each.
    larva_ids = pd.DataFrame([1, 1, 2, 2])
    times = pd.DataFrame([0.0, 0.5, 0.0, 0.5])
    # Npoints = 2 -> head/tail coordinates.
    x_spine = pd.DataFrame(
        [
            [0.0, 1.0],
            [0.1, 1.1],
            [0.2, 1.2],
            [0.3, 1.3],
        ]
    )
    y_spine = pd.DataFrame(
        [
            [0.0, -1.0],
            [0.1, -0.9],
            [0.2, -0.8],
            [0.3, -0.7],
        ]
    )
    states = pd.DataFrame(["rest", "crawl", "rest", "crawl"])

    data_map = {
        "_larvaid.txt": larva_ids,
        "_t.txt": times,
        "_x_spine.txt": x_spine,
        "_y_spine.txt": y_spine,
        "_state.txt": states,
    }

    for suffix, df in data_map.items():
        df.to_csv(prefix.as_posix() + suffix, header=False, index=False, sep="\t")

    tracker = SimpleNamespace(Npoints=2, Ncontour=0, dt=0.5)

    result = import_aux.read_timeseries_from_raw_files_per_parameter(
        prefix.as_posix(), tracker=tracker
    )

    # Columns include AgentID-derived coordinates plus the optional state and Step.
    expected_columns = {
        "t",
        "head_x",
        "tail_x",
        "head_y",
        "tail_y",
        "state",
        "Step",
    }
    assert expected_columns.issubset(result.columns)

    # Verify AgentID index and the derived Step column.
    ordered = result.reset_index()
    assert ordered["AgentID"].tolist() == [1, 1, 2, 2]
    np.testing.assert_allclose(ordered["Step"].values, [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(ordered[["head_x", "tail_x"]].iloc[1].values, [0.1, 1.1])
