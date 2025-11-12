import numpy as np
import pandas as pd
import pytest

from larvaworld.lib import reg, util
from larvaworld.lib.process.calibration import comp_segmentation, fit_metric_definition


def _make_stride_variability_row(
    point_idx: int = 2, use_component: bool = True
) -> pd.DataFrame:
    """Create a minimal stride variability DataFrame compatible with fit_metric_definition."""
    str_cols = reg.getPar(["str_sd_var", "str_t_var"])
    data = {
        str_cols[0]: [0.15],
        str_cols[1]: [0.1],
        "point_idx": [point_idx],
        "use_component_vel": [use_component],
    }
    return pd.DataFrame(data, index=pd.Index(["velocity_point"], name="VelPar"))


def test_fit_metric_definition_updates_config():
    """fit_metric_definition should update spatial/angular metric configuration on the dataset config."""
    str_var = _make_stride_variability_row()
    df_corr = pd.DataFrame(
        {"pars": [["velocity_angle0"]], "corr": [0.9], "p-value": [0.01]},
        index=pd.MultiIndex.from_tuples([(1, 2)], names=["i0", "i1"]),
    )
    config = util.AttrDict({"Npoints": 4, "metric_definition": util.AttrDict()})

    fit_metric_definition(str_var=str_var, df_corr=df_corr, c=config)

    expected_point = util.nam.midline(config.Npoints, type="point")[1]
    assert config.metric_definition.spatial.point_idx == 2
    assert config.metric_definition.spatial.use_component_vel is True
    assert config.point == expected_point
    assert config.metric_definition.angular.best_combo == str((1, 2))
    assert config.metric_definition.angular.front_body_ratio == pytest.approx(1.0)


def test_comp_segmentation_basic():
    """comp_segmentation should return regression and correlation tables for angular metrics."""
    Npoints = 4
    config = util.AttrDict({"Npoints": Npoints})
    angles = [f"angle{i}" for i in range(Npoints - 2)]
    avels = util.nam.vel(angles)
    hov = util.nam.vel(util.nam.orient("front"))

    t = np.linspace(0, 1, 20)
    base_signal = np.sin(2 * np.pi * t)
    data = {
        hov: base_signal + 0.05,
        avels[0]: base_signal * 0.8 + 0.02,
        avels[1]: base_signal * 0.6 + 0.01,
    }
    step_df = pd.DataFrame(data)

    result = comp_segmentation(step_df, pd.DataFrame(), config)

    assert set(result.keys()) == {"bend2or_regression", "bend2or_correlation"}
    assert not result["bend2or_regression"].empty
    assert not result["bend2or_correlation"].empty
    assert result["bend2or_regression"].index[0] == 1


def test_comp_segmentation_missing_velocities_raises():
    """comp_segmentation should raise when required angular velocity columns are absent."""
    config = util.AttrDict({"Npoints": 4})
    with pytest.raises(ValueError):
        comp_segmentation(pd.DataFrame({"dummy": [0.1, 0.2]}), pd.DataFrame(), config)
