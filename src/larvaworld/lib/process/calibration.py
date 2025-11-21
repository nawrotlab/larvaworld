"""
Methods for model calibration
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from .dataset import LarvaDataset

import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

from ... import vprint
from .. import reg, util
from ..util import nam
from .dataset import DatasetConfig

__all__: list[str] = [
    "vel_definition",
    "comp_stride_variation",
    "fit_metric_definition",
    "comp_segmentation",
]


def comp_linear(d: LarvaDataset, mode: str = "minimal") -> None:
    s, e, c = d.data
    assert isinstance(c, DatasetConfig)
    points = c.midline_points
    if mode == "full":
        vprint(
            f"Computing linear distances, velocities and accelerations for {c.Npoints - 1} points"
        )
        points = points[1:]
        orientations = c.seg_orientations
    elif mode == "minimal":
        if c.point == "centroid" or c.point == points[0]:
            vprint(
                "Defined point is either centroid or head. Orientation of front segment not defined."
            )
            return
        else:
            vprint(
                "Computing linear distances, velocities and accelerations for a single spinepoint"
            )
            points = [c.point]
            orientations = ["rear_orientation"]

    if not util.cols_exist(orientations, s):
        vprint("Required orients not found. Component linear metrics not computed.")
        return

    all_d = [s.xs(id, level="AgentID", drop_level=True) for id in c.agent_ids]
    dsts = nam.lin(nam.dst(points))
    cum_dsts = nam.cum(nam.lin(dsts))
    vels = nam.lin(nam.vel(points))
    accs = nam.lin(nam.acc(points))

    for p, dst, cum_dst, vel, acc, orient in zip(
        points, dsts, cum_dsts, vels, accs, orientations
    ):
        D = np.zeros([c.Nticks, c.N]) * np.nan
        Dcum = np.zeros([c.Nticks, c.N]) * np.nan
        V = np.zeros([c.Nticks, c.N]) * np.nan
        A = np.zeros([c.Nticks, c.N]) * np.nan

        for i, data in enumerate(all_d):
            v, d = util.compute_component_velocity(
                xy=data[nam.xy(p)].values,
                angles=data[orient].values,
                dt=c.dt,
                return_dst=True,
            )
            a = np.diff(v) / c.dt
            cum_d = np.nancumsum(d)
            D[:, i] = d
            Dcum[:, i] = cum_d
            V[:, i] = v
            A[1:, i] = a

        s[dst] = D.flatten()
        s[cum_dst] = Dcum.flatten()
        s[vel] = V.flatten()
        s[acc] = A.flatten()
        e[nam.cum(dst)] = Dcum[-1, :]
    pars = nam.xy(points) + dsts + cum_dsts + vels + accs
    d.scale_to_length(pars=pars)
    vprint("All linear parameters computed")


def vel_definition(d: LarvaDataset) -> Dict[str, Any]:
    """
    Compute velocity-related metrics for model calibration.

    Combines stride variability analysis with bend-orientation correlation
    to determine optimal velocity calculation methods for larva movement.

    Args:
        d: LarvaDataset with computed spatial and angular data.
           Must have midline positions and angular velocities computed.

    Returns:
        Dict containing calibration metrics with keys:
        - 'stride_data': DataFrame with stride analysis
        - 'stride_variability': Variability coefficients
        - 'bend2or_regression': Regression parameters
        - 'bend2or_correlation': Correlation coefficients

    Side Effects:
        Updates d.vel_definition attribute and saves results to disk.

    Example:
        >>> d = LarvaDataset(dir='path/to/data')
        >>> d.comp_spatial()
        >>> results = vel_definition(d)
    """
    s, e, c = d.data
    assert isinstance(c, DatasetConfig)
    res_v = comp_stride_variation(d)
    res_fov = comp_segmentation(s, e, c)
    fit_metric_definition(
        str_var=res_v["stride_variability"], df_corr=res_fov["bend2or_correlation"], c=c
    )
    dic = {**res_v, **res_fov}
    d.vel_definition = dic
    d.save_config()
    util.storeH5(dic, key=None, path=f"{d.data_dir}/vel_definition.h5")
    vprint("Velocity definition dataset stored.")
    return dic


def comp_stride_variation(d: LarvaDataset) -> Dict[str, Any]:
    """
    Compute stride variability metrics for movement analysis.

    Analyzes stride length and frequency variations across different
    movement conditions to characterize locomotor patterns.

    Args:
        d: LarvaDataset with spatial data and computed velocities.

    Returns:
        Dict with keys:
        - 'stride_data': DataFrame with stride analysis
        - 'stride_variability': Variability metrics (mean, std, CV)

    Example:
        >>> d = LarvaDataset(dir='path/to/data')
        >>> d.comp_spatial()
        >>> stride_vars = comp_stride_variation(d)
    """
    s, e, c = d.data
    N = c.Npoints
    points = c.midline_points
    vels = util.nam.vel(points)
    cvel = util.nam.vel("centroid")
    lvels = util.nam.lin(util.nam.vel(points[1:]))

    all_point_idx = np.arange(N).tolist() + [-1] + np.arange(N).tolist()[1:]
    all_points = points + ["centroid"] + points[1:]
    lin_flag = [False] * N + [False] + [True] * (N - 1)
    all_vels0 = vels + [cvel] + lvels
    all_vels = util.nam.scal(all_vels0)

    vel_num_strings = ["{" + str(i + 1) + "}" for i in range(N)]
    lvel_num_strings = ["{" + str(i + 2) + "}" for i in range(N - 1)]
    symbols = (
        [rf"$v_{i}$" for i in vel_num_strings]
        + [r"$v_{cen}$"]
        + [rf'$v^{"c"}_{i}$' for i in lvel_num_strings]
    )

    markers = ["o" for i in range(len(vels))] + ["s"] + ["v" for i in range(len(lvels))]
    cnum = 1 + N
    cmap0 = plt.colormaps["hsv"]
    cmap0 = [cmap0(1.0 * i / cnum) for i in range(cnum)]
    cmap0 = cmap0[1:] + [cmap0[0]] + cmap0[2:]

    dic = {
        all_vels[ii]: {
            "symbol": symbols[ii],
            "marker": markers[ii],
            "color": cmap0[ii],
            "idx": ii,
            "par": all_vels0[ii],
            "point": all_points[ii],
            "point_idx": all_point_idx[ii],
            "use_component_vel": lin_flag[ii],
        }
        for ii in range(len(all_vels))
    }

    if not util.cols_exist(vels + [cvel], s):
        s[c.centroid_xy] = (
            np.sum(s[c.contour_xy].values.reshape([-1, c.Ncontour, 2]), axis=1)
            / c.Ncontour
        )
        d.comp_spatial(mode="full")

    if not util.cols_exist(lvels, s):
        d.comp_orientations(mode="full")
        comp_linear(d, mode="full")

    if not util.cols_exist(all_vels, s):
        d.scale_to_length(pars=all_vels0)

    svels = util.existing_cols(all_vels, s)

    shorts = [
        "fsv",
        "str_N",
        "str_tr",
        "str_t_mu",
        "str_t_std",
        "str_sd_mu",
        "str_sd_std",
        "str_t_var",
        "str_sd_var",
    ]

    my_index = pd.MultiIndex.from_product(
        [svels, c.agent_ids], names=["VelPar", "AgentID"]
    )
    df = pd.DataFrame(index=my_index, columns=reg.getPar(shorts))

    for ii in range(c.N):
        id = c.agent_ids[ii]
        ss = s.xs(id, level="AgentID")
        for vv in svels:
            cum_dur = ss[vv].dropna().values.shape[0] * c.dt
            a = ss[vv].values
            fr = util.fft_max(a, c.dt, fr_range=(1, 2.5))
            strides = d.detect_strides(a, fr=fr, return_extrema=False)
            if len(strides) == 0:
                row = [fr, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            else:
                strides[:, 1] = strides[:, 1] - 1
                durs = d.epoch_durs(strides)
                amps = d.epoch_amps(strides, a)
                row = [
                    fr,
                    strides.shape[0],
                    np.sum(durs) / cum_dur,
                    np.mean(durs),
                    np.std(durs),
                    np.mean(amps),
                    np.std(amps),
                    stats.variation(durs),
                    stats.variation(amps),
                ]

            df.loc[(vv, id)] = row
    str_var = (
        df[reg.getPar(["str_sd_var", "str_t_var", "str_tr"])]
        .astype(float)
        .groupby("VelPar")
        .mean()
    )
    for ii in [
        "symbol",
        "color",
        "marker",
        "par",
        "idx",
        "point",
        "point_idx",
        "use_component_vel",
    ]:
        str_var[ii] = [dic[jj][ii] for jj in str_var.index.values]
    dic = {"stride_data": df, "stride_variability": str_var}

    vprint("Stride variability analysis complete!")
    return dic


def fit_metric_definition(
    str_var: pd.DataFrame, df_corr: pd.DataFrame, c: DatasetConfig
) -> None:
    """
    Fit metric definitions using stride variability and correlation data.

    Determines optimal threshold parameters for movement classification
    based on stride variability patterns and behavioral correlations.

    Args:
        str_var: DataFrame containing stride variability metrics.
        df_corr: DataFrame with correlation coefficients between
                different movement parameters.
        c: Dataset configuration with angular parameters.

    Side Effects:
        Updates c.angular.best_combo and related configuration attributes.

    Example:
        >>> metrics = fit_metric_definition(stride_data, corr_data, config)
    """
    Nangles = 0 if c.Npoints < 3 else c.Npoints - 2
    sNt_cv = str_var[reg.getPar(["str_sd_var", "str_t_var"])].sum(axis=1)
    best_idx = sNt_cv.argmin()

    best_combo = df_corr.index.values[0]
    best_combo_max = np.max(best_combo)

    md = c.metric_definition
    if "spatial" not in md:
        md.spatial = util.AttrDict()
    idx = md.spatial.point_idx = int(str_var["point_idx"].iloc[best_idx])
    md.spatial.use_component_vel = bool(str_var["use_component_vel"].iloc[best_idx])
    try:
        p = util.nam.midline(c.Npoints, type="point")[idx - 1]
    except:
        p = "centroid"
    c.point = p
    if "angular" not in md:
        md.angular = util.AttrDict()
    md.angular.best_combo = str(best_combo)
    md.angular.front_body_ratio = best_combo_max / Nangles
    md.angular.bend = "from_vectors"


def comp_segmentation(
    s: pd.DataFrame, e: pd.DataFrame, c: DatasetConfig
) -> Dict[str, Any]:
    """
    Compute segmentation metrics for behavioral analysis.

    Analyzes movement segments to identify distinct behavioral
    patterns and transitions in larva locomotion using bend-orientation
    correlation analysis.

    Args:
        s: DataFrame with step segment data.
        e: DataFrame with epoch segment data.
        c: Dataset configuration with midline point information.

    Returns:
        Dict with keys:
        - 'bend2or_regression': Regression parameters
        - 'bend2or_correlation': Correlation coefficients

    Example:
        >>> segments = comp_segmentation(step_data, epoch_data, config)
    """
    N = np.clip(c.Npoints - 2, a_min=0, a_max=None)
    angles = [f"angle{i}" for i in range(N)]
    avels = util.nam.vel(angles)
    hov = util.nam.vel(util.nam.orient("front"))

    if not util.cols_exist(avels, s):
        raise ValueError("Spineangle angular velocities do not exist in step")

    ss = s.loc[s[hov].dropna().index.values]
    y = ss[hov].values

    vprint(
        "Computing linear regression of angular velocity based on segmental bending velocities"
    )
    df_reg = []
    for i in range(N):
        p0 = avels[i]
        X0 = ss[[p0]].values
        reg0 = LinearRegression().fit(X0, y)
        sc0 = reg0.score(X0, y)
        c0 = reg0.coef_[0]
        p1 = avels[: i + 1]
        X1 = ss[p1].values
        reg1 = LinearRegression().fit(X1, y)
        sc1 = reg1.score(X1, y)
        c1 = reg1.coef_[0]

        df_reg.append(
            {
                "idx": i + 1,
                "single_par": p0,
                "single_score": sc0,
                "single_coef": c0,
                "cum_pars": p1,
                "cum_score": sc1,
                "cum_coef": c1,
            }
        )
    df_reg = pd.DataFrame(df_reg)
    df_reg.set_index("idx", inplace=True)

    vprint(
        "Computing correlation of angular velocity with combinations of segmental bending velocities"
    )
    df_corr = []
    for i in range(int(N * 4 / 7)):
        for idx in itertools.combinations(np.arange(N), i + 1):
            if i == 0:
                idx = idx[0]
                idx0 = idx + 1
                ps = avels[idx]
                X = ss[ps].values
            else:
                idx = list(idx)
                idx0 = [ii + 1 for ii in idx]
                ps = [avels[ii] for ii in idx]
                X = ss[ps].sum(axis=1).values
            r, p = stats.pearsonr(y, X)

            df_corr.append({"idx": idx0, "pars": ps, "corr": r, "p-value": p})

    df_corr = pd.DataFrame(df_corr)
    df_corr.set_index("idx", inplace=True)
    df_corr.sort_values("corr", ascending=False, inplace=True)
    dic = {"bend2or_regression": df_reg, "bend2or_correlation": df_corr}
    vprint("Angular velocity definition analysis complete!")
    return dic
