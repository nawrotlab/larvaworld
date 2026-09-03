"""Tests for the trajectory-mode dataset built on movingpandas/geopandas.

The module had no coverage, which is how a missing import and a dropped constructor
argument both survived in it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from larvaworld.lib.process.dataset_geo import GeoLarvaDataset

NTICKS = 40
DT = 0.1
IDS = ["Larva_0", "Larva_1"]


def _step_data():
    index = pd.MultiIndex.from_product([range(NTICKS), IDS], names=["Step", "AgentID"])
    x = np.tile(np.linspace(0, 0.01, NTICKS), (len(IDS), 1)).T.ravel()
    y = np.tile(np.linspace(0, 0.005, NTICKS), (len(IDS), 1)).T.ravel()
    return pd.DataFrame({"x": x, "y": y}, index=index)


def _dataset(tmp_path):
    end = pd.DataFrame(
        {"length": np.full(len(IDS), 0.004)}, index=pd.Index(IDS, name="AgentID")
    )
    return GeoLarvaDataset(
        dir=str(tmp_path / "geo"),
        id="geo",
        agent_ids=IDS,
        dt=DT,
        Nsteps=NTICKS,
        step=_step_data(),
        end=end,
        load_data=False,
    )


def test_geo_dataset_builds_one_trajectory_per_agent(tmp_path):
    d = _dataset(tmp_path)

    assert len(list(d)) == len(IDS)


def test_geo_dataset_keeps_the_timestep_in_its_configuration(tmp_path):
    """dt drives the trajectories, but it also defines the dataset's framerate."""
    d = _dataset(tmp_path)

    assert d.config.dt == pytest.approx(DT)
    assert d.config.fr == pytest.approx(1 / DT)


@pytest.mark.parametrize(
    ("prop", "expected"),
    [("spatial_pint_unit", "meter"), ("temporal_pint_unit", "second")],
)
def test_pint_unit_properties_resolve(tmp_path, prop, expected):
    """These reference PintType, which the module has to import at module level."""
    d = _dataset(tmp_path)

    assert expected in str(getattr(d, prop))


def test_spatial_unit_is_metres(tmp_path):
    d = _dataset(tmp_path)

    assert str(d.spatial_unit) == "meter"
