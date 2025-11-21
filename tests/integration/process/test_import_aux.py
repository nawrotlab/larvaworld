import os

import pandas as pd
import pytest

from larvaworld.lib.process import read_timeseries_from_raw_files_per_parameter
from larvaworld.lib import reg
from larvaworld import DATA_DIR

pytestmark = [pytest.mark.integration, pytest.mark.fast]


@pytest.fixture
def tracker():
    labID = "Jovanic"
    return reg.conf.LabFormat.get(labID).tracker


def xtest_read_timeseries_from_raw_files_per_parameter(tracker):
    pref = f"{DATA_DIR}/JovanicGroup/raw/ProteinDeprivation/Fed"

    # Verify that the directory exists
    # assert os.path.isdir(pref)

    df = read_timeseries_from_raw_files_per_parameter(pref, tracker=tracker)

    # Check if the dataframe has the expected columns
    # expected_columns = ["AgentID", "t"] + [f"x_spine_{i}" for i in range(10)] + [f"y_spine_{i}" for i in range(10)] + ["state"]
    # assert list(df.columns) == expected_columns

    # Check if the dataframe has the expected index
    assert df.index.name == "AgentID"

    # Check if the dataframe has the expected values
    # assert df.loc[1, "t"] == 0.1
    # assert df.loc[2, "t"] == 0.2

    # Check if the Step column is correctly calculated
    # assert df.loc[1, "Step"] == 0.1 / tracker.dt
    # assert df.loc[2, "Step"] == 0.2 / tracker.dt
