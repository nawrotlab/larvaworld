"""
Sensory landscape plotting
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

from .. import plot, reg, util, funcs

__all__: list[str] = [
    "plot_odorscape",
    "plot_2d",
    "plot_3pars",
    "odorscape_isocontours",
    "odorscape_with_sample_tracks",
    "plot_heatmap_PI",
]


def plot_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vars: Sequence[str],
    target: str,
    z0: Optional[float] = None,
    title: Optional[str] = None,
    lims: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]] = None,
    azim: int = 115,
    elev: int = 15,
    **kwargs: Any,
) -> Any:
    P = plot.AutoBasePlot(name="3d_surface", dim3=True, azim=azim, elev=elev, **kwargs)
    P.conf_ax_3d(vars=vars, target=target, lims=lims, title=title)
    from matplotlib import cm

    P.axs[0].plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    if z0 is not None:
        P.axs[0].plot_surface(x, y, np.ones(x.shape) * z0, alpha=0.5)
    return P.get()


@funcs.graph("odorscape", required={"args": ["odor_layers"]})
def plot_odorscape(
    odor_layers: dict, scale: float = 1.0, idx: int = 0, **kwargs: Any
) -> None:
    """
    Plot 3D odorscape surfaces for multiple odor layers.

    Creates 3D surface plots showing spatial distribution of odor
    concentrations across the environment arena.

    Args:
        odor_layers: Dictionary of odor layer objects with grid and meshgrid
        scale: Spatial scale factor. Defaults to 1.0
        idx: Index for file naming. Defaults to 0
        **kwargs: Additional arguments passed to plot_surface

    Example:
        >>> plot_odorscape(odor_layers={'odor1': layer1}, scale=1.0, idx=0)
    """
    for id, layer in odor_layers.items():
        X, Y = layer.meshgrid
        x = X * 1000 / scale
        y = Y * 1000 / scale
        plot_surface(
            x=x,
            y=y,
            z=layer.grid,
            vars=[r"x $(mm)$", r"y $(mm)$"],
            target=r"concentration $(μM)$",
            title=f"{id} odorscape",
            save_as=f"{id}_odorscape_{idx}",
            **kwargs,
        )


def odorscape_isocontours(
    intensity: float = 2, spread: float = 0.0002, radius: float = 0.05
) -> None:
    """
    Plot odorscape as filled contour isocontours.

    Creates 2D contour plot showing odor concentration gradients
    from a point source using multivariate normal distribution.

    Args:
        intensity: Peak odor intensity. Defaults to 2
        spread: Odor spread parameter. Defaults to 0.0002
        radius: Plot radius in meters. Defaults to 0.05

    Example:
        >>> odorscape_isocontours(intensity=2, spread=0.0002, radius=0.05)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import multivariate_normal

    x, y = np.mgrid[-radius:radius:0.001, -radius:radius:0.001]
    rv = multivariate_normal([0, 0], [[spread, 0], [0, spread]])
    p0 = rv.pdf((0, 0))
    data = np.dstack((x, y))
    z = rv.pdf(data) * intensity / p0
    plt.contourf(x, y, z, cmap="coolwarm")
    plt.show()


def odorscape_from_config(
    c: Any,
    mode: str = "2D",
    fig: Any = None,
    axs: Any = None,
    show: bool = True,
    grid_dims: Tuple[int, int] = (201, 201),
    col_max: Tuple[int, int, int] = (0, 0, 0),
    **kwargs: Any,
) -> Any:
    env = c.env_params
    source = list(env.food_params.source_units.values())[0]
    a0, b0 = source.pos
    oP, oS = source.odor.intensity, source.odor.spread
    oD = multivariate_normal([0, 0], [[oS, 0], [0, oS]])
    oM = oP / oD.pdf([0, 0])
    if col_max is None:
        col_max = source.color if source.color is not None else (0, 0, 0)
    if grid_dims is not None:
        X, Y = grid_dims
    else:
        X, Y = [51, 51] if env.odorscape.grid_dims is None else env.odorscape.grid_dims
    Xdim, Ydim = env.arena.dims
    s = 1
    Xmesh, Ymesh = np.meshgrid(
        np.linspace(-Xdim * s / 2, Xdim * s / 2, X),
        np.linspace(-Ydim * s / 2, Ydim * s / 2, Y),
    )

    @np.vectorize
    def func(a: float, b: float) -> float:
        return oD.pdf([a - a0, b - b0]) * oM

    grid = func(Xmesh, Ymesh)

    if mode == "2D":
        from matplotlib import pyplot as plt

        if fig is None and axs is None:
            fig, axs = plt.subplots(1, 1, figsize=(10, 10 * Ydim / Xdim))
        q = grid.flatten() - np.min(grid)
        q /= np.max(q)
        cols = util.col_range(q, low=(255, 255, 255), high=col_max, mul255=False)
        x, y = (
            Xmesh * 1000 / s,
            Ymesh * 1000 / s,
        )
        axs.scatter(x=x, y=y, color=cols)
        axs.set_aspect("equal", adjustable="box")
        axs.set_xlim([np.min(x), np.max(x)])
        axs.set_ylim([np.min(y), np.max(y)])
        axs.set_xlabel(r"X $(mm)$")
        axs.set_ylabel(r"Y $(mm)$")
        if show:
            plt.show()
    elif mode == "3D":
        return plot_surface(
            x=Xmesh * 1000 / s,
            y=Ymesh * 1000 / s,
            z=grid,
            vars=[r"X $(mm)$", r"Y $(mm)$"],
            target=r"concentration $(μM)$",
            save_as="odorscape",
            show=show,
            fig=fig,
            axs=axs,
            azim=0,
            elev=0,
        )


def odorscape_with_sample_tracks(
    datasets: Sequence[Any],
    unit: str = "mm",
    fig: Any = None,
    axs: Any = None,
    show: bool = False,
    save_to: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Plot odorscape with sample trajectory tracks overlaid.

    Creates 2D odorscape visualization with individual trajectory tracks
    from multiple datasets shown as colored paths.

    Args:
        datasets: List of datasets containing trajectory data
        unit: Distance unit ('mm' or 'm'). Defaults to 'mm'
        fig: Matplotlib figure. Creates new if None
        axs: Matplotlib axes. Creates new if None
        show: Whether to display plot. Defaults to False
        save_to: Directory to save plot. Defaults to None
        **kwargs: Additional arguments passed to odorscape_from_config

    Returns:
        Matplotlib figure object

    Example:
        >>> fig = odorscape_with_sample_tracks(datasets=[d1, d2], unit='mm', show=True)
    """
    scale = 1000 if unit == "mm" else 1
    if fig is None and axs is None:
        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
    odorscape_from_config(
        datasets[0].config, mode="2D", fig=fig, axs=axs, show=False, **kwargs
    )
    for d in datasets:
        xy = d.step_data[["x", "y"]].xs(d.agent_ids[0], level="AgentID").values * scale
        axs.plot(xy[:, 0], xy[:, 1], label=d.id, color=d.color)
    axs.legend(loc="upper left", fontsize=15)
    if show:
        plt.show()
    return fig


def plot_3pars(
    df: pd.DataFrame,
    vars: Sequence[str],
    target: str,
    z0: Optional[float] = None,
    **kwargs: Any,
) -> dict:
    """
    Create multiple plots for 3-parameter relationships.

    Generates 3D plot, heatmap, and surface plot showing relationship
    between two independent variables and a target variable.

    Args:
        df: DataFrame containing parameter data
        vars: List of 2 independent variable names
        target: Target variable name
        z0: Reference z-value for horizontal plane. Defaults to None
        **kwargs: Additional arguments passed to plotting functions

    Returns:
        Dictionary of figure objects keyed by plot type

    Example:
        >>> figs = plot_3pars(df, vars=['param1', 'param2'], target='result')
    """
    figs = {}
    pr = f"{vars[0]}VS{vars[1]}"
    figs[f"{pr}_3d"] = plot_3d(df=df, vars=vars, target=target, **kwargs)
    try:
        x, y = np.unique(df[vars[0]].values), np.unique(df[vars[1]].values)
        X, Y = np.meshgrid(x, y)

        z = df[target].values.reshape(X.shape).T

        figs[f"{pr}_heatmap"] = plot_heatmap(
            z,
            ax_kws={
                "xticklabels": x.tolist(),
                "yticklabels": y.tolist(),
                "xlab": vars[0],
                "ylab": vars[1],
            },
            cbar_kws={"label": target},
            **kwargs,
        )
        figs[f"{pr}_surface"] = plot_surface(
            X, Y, z, vars=vars, target=target, z0=z0, **kwargs
        )
    except:
        pass
    return figs


def plot_3d(
    df: pd.DataFrame,
    vars: Sequence[str],
    target: str,
    name: Optional[str] = None,
    lims: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]] = None,
    title: Optional[str] = None,
    surface: bool = True,
    line: bool = False,
    dfID: Optional[str] = None,
    color: str = "black",
    **kwargs: Any,
) -> Any:
    if name is None:
        name = "3d_plot"
    from statsmodels import api as sm

    P = plot.AutoBasePlot(name=name, dim3=True, **kwargs)
    P.conf_ax_3d(vars=vars, target=target, lims=lims, title=title)

    l0, l1 = vars
    X = df[vars]
    y = df[target].values

    X = sm.add_constant(X)
    # plot hyperplane
    if surface:
        est = sm.OLS(y, X).fit()

        xx1, xx2 = np.meshgrid(
            np.linspace(X[l0].min(), X[l0].max(), 100),
            np.linspace(X[l1].min(), X[l1].max(), 100),
        )
        # plot the hyperplane by evaluating the parameters on the grid
        Z = est.params[0] + est.params[1] * xx1 + est.params[2] * xx2
        surf = P.axs[0].plot_surface(
            xx1, xx2, Z, cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0
        )
        # plot data points - points over the HP are white, points below are black
        resid = y - est.predict(X)
        P.axs[0].scatter(
            X[resid >= 0][l0],
            X[resid >= 0][l1],
            y[resid >= 0],
            color="black",
            alpha=0.4,
            facecolor="white",
        )
        P.axs[0].scatter(
            X[resid < 0][l0],
            X[resid < 0][l1],
            y[resid < 0],
            color="black",
            alpha=0.4,
            facecolor=color,
        )
    else:
        P.axs[0].scatter(X[l0].values, X[l1].values, y, color="black", alpha=0.4)

    return P.get()


def plot_3d_multi(
    dfs: Sequence[pd.DataFrame],
    dfIDs: Sequence[str],
    df_colors: Optional[Sequence[str]] = None,
    show: bool = True,
    **kwargs: Any,
) -> None:
    """
    Create multi-dataset 3D plot with regression surfaces.

    Generates 3D scatter plot with regression hyperplanes for multiple
    datasets shown in different colors.

    Args:
        dfs: List of DataFrames with parameter data
        dfIDs: List of dataset identifiers
        df_colors: List of colors for each dataset. Defaults to None
        show: Whether to display plot. Defaults to True
        **kwargs: Additional arguments passed to plot_3d

    Example:
        >>> plot_3d_multi(dfs=[df1, df2], dfIDs=['Control', 'Test'], show=True)
    """
    from mpl_toolkits.mplot3d import Axes3D

    if df_colors is None:
        df_colors = [None] * len(dfs)
    fig = plt.figure(figsize=(18, 12))
    ax = Axes3D(fig, azim=115, elev=15)
    for df, dfID, dfC in zip(dfs, dfIDs, df_colors):
        plot_3d(df, dfID=dfID, color=dfC, axs=ax, fig=fig, show=False, **kwargs)
    if show:
        plt.show()


def plot_heatmap(
    z: Any, heat_kws: dict = {}, ax_kws: dict = {}, cbar_kws: dict = {}, **kwargs: Any
) -> Any:
    base_heat_kws = {"annot": True, "cmap": cm.coolwarm, "vmin": None, "vmax": None}
    base_heat_kws.update(heat_kws)
    base_cbar_kws = {"orientation": "vertical"}
    base_cbar_kws.update(cbar_kws)
    P = plot.AutoBasePlot(name="heatmap", **kwargs)
    sns.heatmap(z, ax=P.axs[0], **base_heat_kws, cbar_kws=base_cbar_kws)
    from matplotlib import pyplot as plt

    cax = plt.gcf().axes[-1]
    cax.tick_params(length=0)
    P.conf_ax(**ax_kws)
    P.adjust((0.15, 0.95), (0.15, 0.95))
    return P.get()


@funcs.graph("PI heatmap")
def plot_heatmap_PI(
    z: Optional[pd.DataFrame] = None,
    csv_filepath: str = "PIs.csv",
    save_as: str = "PI_heatmap.pdf",
    **kwargs: Any,
) -> Any:
    """
    Create heatmap of Preference Index (PI) values.

    Generates color-coded heatmap showing PI values across left and right
    odor gain combinations, with red-yellow-green colormap.

    Args:
        z: DataFrame with PI values. Loads from csv_filepath if None
        csv_filepath: Path to CSV file with PI data. Defaults to 'PIs.csv'
        save_as: Filename for saved plot. Defaults to 'PI_heatmap.pdf'
        **kwargs: Additional arguments passed to plot_heatmap

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_heatmap_PI(z=pi_data, save_as='preference_heatmap.pdf')
    """
    if z is None:
        z = pd.read_csv(csv_filepath, index_col=0)
    Lgains = z.index.values.astype(int)
    Rgains = z.columns.values.astype(int)
    Ngains = len(Lgains)
    r = np.linspace(0.5, Ngains - 0.5, 5)
    ax_kws = {
        "xticklabels": Rgains[r.astype(int)],
        "yticklabels": Lgains[r.astype(int)],
        "xticklabelrotation": 0,
        "yticklabelrotation": 0,
        "xticks": r,
        "yticks": r,
        "xlab": r"Right odor gain, $G_{R}$",
        "ylab": r"Left odor gain, $G_{L}$",
        "xlabelpad": 20,
    }
    heat_kws = {
        "annot": False,
        "vmin": -1,
        "vmax": 1,
        "cmap": "RdYlGn",
    }

    cbar_kws = {"label": "Preference for left odor", "ticks": [1, 0, -1]}

    return plot_heatmap(
        z,
        heat_kws=heat_kws,
        ax_kws=ax_kws,
        cbar_kws=cbar_kws,
        save_as=save_as,
        **kwargs,
    )


def plot_2d(df: pd.DataFrame, labels: Sequence[str], **kwargs: Any) -> Any:
    """
    Create 2D scatter plot of parameter vs result.

    Simple scatter plot showing relationship between a parameter
    and a result variable.

    Args:
        df: DataFrame containing data
        labels: List of 2 labels [parameter, result]
        **kwargs: Additional arguments passed to AutoBasePlot

    Returns:
        Plot output (figure object or None based on return_fig setting)

    Example:
        >>> fig = plot_2d(df, labels=['temperature', 'velocity'])
    """
    P = plot.AutoBasePlot(name="2d_plot", **kwargs)
    par = labels[0]
    res = labels[1]
    p = df[par].values
    r = df[res].values
    P.axs[0].scatter(p, r)
    P.conf_ax(xlab=par, ylab=res)
    return P.get()
