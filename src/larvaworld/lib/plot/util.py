"""
Methods used in plotting
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

import itertools
import os
import warnings

import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend for headless environments

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches, transforms

from scipy.stats import mannwhitneyu

warnings.simplefilter(action="ignore", category=FutureWarning)

from ... import vprint

__all__: list[str] = [
    "plot_quantiles",
    "plot_mean_and_range",
    "circular_hist",
    "circNarrow",
    "confidence_ellipse",
    "dataset_legend",
    "label_diff",
    "annotate_plot",
    "dual_half_circle",
    "annotate_plot",
    "save_plot",
    "process_plot",
    "prob_hist",
    "single_boxplot",
    "configure_subplot_grid",
    "define_end_ks",
    "get_vs",
    "color_epochs",
]


def plot_quantiles(
    df: "np.ndarray | pd.DataFrame | pd.Series",
    x: Optional["np.ndarray"] = None,
    **kwargs: Any,
) -> None:
    """
    Plot quantiles or confidence intervals along with the mean.

    Computes and plots 25th, 50th (median), and 75th percentiles with
    shaded interquartile range. Handles numpy arrays, DataFrames, or Series.

    Args:
        df: Data to compute quantiles from. If numpy.ndarray, quantiles are
            computed along the first axis (columns). If pandas.DataFrame or
            pandas.Series, quantiles are computed grouped by the 'Step' level
        x: X-axis values for the plot. Uses default range if None
        **kwargs: Additional keyword arguments passed to plot_mean_and_range

    Raises:
        Exception: If the input data type is not recognized

    Example:
        >>> plot_quantiles(data_array, x_values, axis=plt.gca(), color='blue', label='Data')
    """
    if isinstance(df, np.ndarray):
        mean = np.nanquantile(df, q=0.5, axis=0)
        ub = np.nanquantile(df, q=0.75, axis=0)
        lb = np.nanquantile(df, q=0.25, axis=0)
        if x is None:
            x = np.arange(len(mean))
    elif isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        mean = df.groupby(level="Step").quantile(q=0.5)
        ub = df.groupby(level="Step").quantile(q=0.75)
        lb = df.groupby(level="Step").quantile(q=0.25)
    else:
        raise Exception("Input data type not recognized.")
    plot_mean_and_range(x=x, mean=mean, lb=lb, ub=ub, **kwargs)


def plot_mean_and_range(
    x: "np.ndarray",
    mean: "np.ndarray | pd.Series",
    lb: "np.ndarray | pd.Series",
    ub: "np.ndarray | pd.Series",
    axis: "Axes",
    color: str,
    color_mean: Optional[str] = None,
    label: Optional[str] = None,
    linestyle: str = "solid",
    linewidth: int = 2,
) -> None:
    """
    Plot the mean and a shaded range (quantiles or confidence intervals).

    Draws mean line with shaded region between lower and upper bounds,
    commonly used for showing uncertainty or variability.

    Args:
        x: X-axis values for the plot
        mean: Mean values to be plotted
        lb: Lower bound of the range
        ub: Upper bound of the range
        axis: Matplotlib axes where the plot will be drawn
        color: Color of the shaded range
        color_mean: Color of the mean line. Uses color if None
        label: Label for the legend. Defaults to None
        linestyle: Line style for the mean line ('solid', 'dashed'). Defaults to 'solid'
        linewidth: Line width for the mean line. Defaults to 2

    Example:
        >>> plot_mean_and_range(x_values, mean_values, lower_bound, upper_bound, plt.gca(), color='blue', label='Mean and Range')
    """
    N = mean.shape[0]
    if x.shape[0] > N:
        xx = x[:N]
    elif x.shape[0] == N:
        xx = x
    else:
        raise Exception("Incompatible input shapes.")
    if color_mean is None:
        color_mean = color
    # Plot the shaded range (confidence intervals)
    axis.fill_between(xx, ub, lb, color=color, alpha=0.2, zorder=0)
    # Plot the mean on top
    axis.plot(
        xx,
        mean,
        color_mean,
        label=label,
        linewidth=linewidth,
        alpha=1.0,
        zorder=10,
        linestyle=linestyle,
    )


def circular_hist(
    ax: "Axes",
    x: "np.ndarray",
    bins: int = 16,
    density: bool = True,
    offset: float = 0,
    gaps: bool = True,
    **kwargs: Any,
) -> Tuple["np.ndarray", "np.ndarray", Any]:
    """
    Produce a circular histogram of angles on polar axes.

    Creates polar histogram showing angular distribution with optional
    density normalization and customizable bin partitioning.

    Args:
        ax: Polar axes instance created with subplot_kw=dict(projection='polar')
        x: Angles to plot in radians
        bins: Number of equal-width bins. Defaults to 16
        density: If True, plot frequency proportional to area. If False, plot
            frequency proportional to radius. Defaults to True
        offset: Offset for the location of 0 direction in radians. Defaults to 0
        gaps: Whether to allow gaps between bins. When False, bins partition
            entire [-pi, pi] range. Defaults to True
        **kwargs: Additional keyword arguments passed to matplotlib bar plot

    Returns:
        Tuple of (n, bins, patches) where n is number of values in each bin,
        bins are bin edges, and patches is BarContainer or list of Polygon

    Example:
        >>> n, bins, patches = circular_hist(polar_ax, angle_data, bins=20, density=True)
    """
    # Wrap angles to [-pi, pi)
    x = (x + np.pi) % (2 * np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins + 1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area / np.pi) ** 0.5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    from matplotlib import pyplot as plt

    patches = plt.bar(
        bins[:-1],
        radius,
        zorder=1,
        align="edge",
        width=widths,
        edgecolor="black",
        fill=True,
        linewidth=2,
        **kwargs,
    )

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


def circNarrow(
    ax: "Axes",
    data: "np.ndarray",
    alpha: float,
    label: str,
    color: str,
    Nbins: int = 16,
) -> None:
    """
    Create a circular histogram with an arrow indicator.

    Combines polar histogram with fancy arrow pointing to mean direction,
    useful for visualizing angular distributions with directional bias.

    Args:
        ax: Polar axes instance created with subplot_kw=dict(projection='polar')
        data: Angles to plot in radians
        alpha: Transparency of the circular histogram and arrow
        label: Label for the circular histogram
        color: Color of the circular histogram and arrow
        Nbins: Number of equal-width bins. Defaults to 16

    Example:
        >>> circNarrow(polar_ax, angle_data, alpha=0.5, label='Orientation', color='blue', Nbins=16)
    """
    # Create the circular histogram
    circular_hist(
        ax, data, bins=Nbins, alpha=alpha, label=label, color=color, offset=np.pi / 2
    )

    # Create an arrow indicator
    arrow = patches.FancyArrowPatch(
        (0, 0),
        (np.mean(data), 0.3),
        zorder=2,
        mutation_scale=30,
        alpha=alpha,
        facecolor=color,
        edgecolor="black",
        fill=True,
        linewidth=0.5,
    )
    ax.add_patch(arrow)


def confidence_ellipse(
    x: "np.ndarray",
    y: "np.ndarray",
    ax: "Axes",
    n_std: float = 3.0,
    facecolor: str = "none",
    **kwargs: Any,
) -> Any:
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object_class to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse'sigma radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse

    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = patches.Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def dataset_legend(
    labels: Sequence[str],
    colors: Sequence[str],
    ax: Optional["Axes"] = None,
    anchor: Optional[Tuple[float, float]] = None,
    handlelength: float = 0.5,
    handleheight: float = 0.5,
    **kwargs: Any,
) -> Any:
    """
    Create a legend for all datasets with their specified labels and colors.

    Generates custom legend with colored patch handles for multiple datasets,
    with customizable positioning and handle dimensions.

    Args:
        labels: List of labels for each dataset
        colors: List of colors corresponding to each dataset
        ax: Axes to which legend should be added. Uses current axes if None
        anchor: Bounding box anchor coordinates for legend. Defaults to None
        handlelength: Length of legend handles. Defaults to 0.5
        handleheight: Height of legend handles. Defaults to 0.5
        **kwargs: Additional keyword arguments passed to legend

    Returns:
        matplotlib.legend.Legend: The created legend

    Example:
        >>> leg = dataset_legend(['Control', 'Test'], ['blue', 'red'], ax=axes, anchor=(1.05, 1))
    """
    kws = {
        "handles": [
            patches.Patch(facecolor=c, label=l, edgecolor="black")
            for c, l in zip(colors, labels)
        ],
        "handlelength": handlelength,
        "handleheight": handleheight,
        "labels": labels,
        "bbox_to_anchor": anchor,
        **kwargs,
    }

    if ax is None:
        from matplotlib import pyplot as plt

        leg = plt.legend(**kws)
    else:
        leg = ax.legend(**kws)
        ax.add_artist(leg)
    return leg


def label_diff(
    i: int, j: int, text: str, X: Sequence[float], Y: Sequence[float], ax: "Axes"
) -> None:
    """
    Label the difference between two data points with an annotation and an arrow.

    Draws horizontal bracket with text annotation between two points,
    commonly used for showing statistical significance.

    Args:
        i: Index of the first data point
        j: Index of the second data point
        text: Text label for the difference annotation
        X: List of x-coordinates for data points
        Y: List of y-coordinates for data points
        ax: Matplotlib axes on which to annotate the difference

    Example:
        >>> label_diff(0, 1, '***', [1, 2], [10, 12], ax)
    """
    x = (X[i] + X[j]) / 2
    y = 1.5 * max(Y[i], Y[j])
    dx = abs(X[i] - X[j])

    props = {
        "connectionstyle": "bar",
        "arrowstyle": "-",
        "shrinkA": 20,
        "shrinkB": 20,
        "linewidth": 2,
    }
    ax.annotate(text, xy=(X[i], y), zorder=10)
    ax.annotate("", xy=(X[i], y), xytext=(X[j], y), arrowprops=props)


def pvalue_star(pv: float) -> str:
    """
    Convert a p-value to a star annotation for significance.

    Maps p-value to standard significance notation: **** (p<0.0001),
    *** (p<0.001), ** (p<0.01), * (p<0.05), ns (not significant).

    Args:
        pv: The p-value to be converted

    Returns:
        Star annotation representing the significance level

    Example:
        >>> pvalue_star(0.001)
        '***'
    """
    a = {1e-4: "****", 1e-3: "***", 1e-2: "**", 0.05: "*", 1: "ns"}
    for k, v in a.items():
        if pv < k:
            return v
    return "ns"


def annotate_plot(
    box: Any,
    data: "pd.DataFrame",
    x: str,
    y: str,
    hue: Optional[str] = None,
    show_ns: bool = True,
    target_only: Any = None,
    **kwargs: Any,
) -> None:
    """
    Annotate a plot with Mann-Whitney U test p-values.

    Performs pairwise Mann-Whitney U tests and adds statistical annotations
    (stars) to boxplots or similar plots using statannot.

    Args:
        box: Seaborn plot object to annotate
        data: DataFrame containing the data
        x: Column name for the x-axis variable
        y: Column name for the y-axis variable
        hue: Column name for grouping data by hue. Defaults to None
        show_ns: Whether to display annotations for non-significant comparisons. Defaults to True
        target_only: Specify a target value for comparisons. Defaults to None
        **kwargs: Additional arguments for annotation customization

    Example:
        >>> annotate_plot(box, data, x='group', y='value', show_ns=False, target_only='control')
    """
    import statannot

    d = data
    ids = np.unique(d[x].values)
    pairs = []
    pvs = []

    def get_data(id: Any, h: Any = None) -> pd.Series:
        dd = d[(d[x] == id) & (d[hue] == h)] if h is not None else d[d[x] == id]
        return dd[y].dropna()

    def get_pv(id1: Any, id2: Any, h1: Any = None, h2: Any = None) -> float:
        return mannwhitneyu(
            get_data(id1, h1), get_data(id2, h2), alternative="two-sided"
        ).pvalue

    def eval_pair(id1: Any, id2: Any, h1: Any = None, h2: Any = None) -> None:
        pv = get_pv(id1, id2, h1, h2)
        pair = (
            ((id1, h1), (id2, h2)) if (h1 is not None or h2 is not None) else (id1, id2)
        )
        if not show_ns and pv >= 0.05:
            pass
        else:
            pairs.append(pair)
            pvs.append(pv)

    if hue is not None:
        h1, h2 = np.unique(d[hue].values)
        for id in ids:
            eval_pair(id, id, h1, h2)

    else:
        if target_only is None:
            for id1, id2 in list(itertools.combinations(ids, 2)):
                eval_pair(id1, id2)

        else:
            for id in ids:
                if id != target_only:
                    eval_pair(target_only, id)

    if len(pairs) > 0:
        statannot.add_stat_annotation(
            box,
            data=d,
            perform_stat_test=False,
            pvalues=pvs,
            box_pairs=pairs,
            text_format="star",
            loc="inside",
            verbose=0,
            **kwargs,
        )


def dual_half_circle(
    center: Tuple[float, float],
    radius: float = 0.04,
    angle: float = 90,
    ax: Optional["Axes"] = None,
    colors: Tuple[str, str] = ("W", "k"),
    **kwargs: Any,
) -> List[Any]:
    """
    Add two half circles to axes with specified face colors rotated at angle.

    Creates two wedge patches forming complete circle with different colored
    halves, rotated to specified angle in degrees.

    Args:
        center: Center coordinates of the half circles
        radius: Radius of the half circles. Defaults to 0.04
        angle: Angle by which the half circles are rotated in degrees. Defaults to 90
        ax: Matplotlib axes to add half circles to. Uses current axes if None
        colors: Face colors of the two half circles (left half, right half). Defaults to ('W', 'k')
        **kwargs: Additional keyword arguments to customize the appearance of the half circles

    Returns:
        List containing the two half circle wedge patches

    Example:
        >>> wedges = dual_half_circle((0.5, 0.5), radius=0.05, angle=45, colors=('red', 'blue'))
    """
    if ax is None:
        from matplotlib import pyplot as plt

        ax = plt.gca()
    theta1, theta2 = angle, angle + 180
    w1 = patches.Wedge(center, radius, theta1, theta2, fc=colors[0], **kwargs)
    w2 = patches.Wedge(center, radius, theta2, theta1, fc=colors[1], **kwargs)
    ax.add_patch(w1)
    ax.add_patch(w2)
    return [w1, w2]


def save_plot(fig: "Figure", filepath: str, filename: str) -> None:
    """
    Save a Matplotlib figure to a specified file path.

    Saves figure at high resolution (300 DPI), closes it, and prints
    confirmation message.

    Args:
        fig: Matplotlib figure to save
        filepath: Full file path where the figure should be saved
        filename: Name of the file to save

    Example:
        >>> save_plot(fig, '/path/to/output.png', 'output.png')
    """
    fig.savefig(filepath, dpi=300, facecolor=None)
    try:
        from matplotlib import pyplot as plt

        plt.close(fig)
    except:
        pass
    vprint(f"Plot {filename} saved as {filepath}", 1)


def process_plot(
    fig: "Figure",
    save_to: Optional[str],
    filename: str,
    return_fig: bool = False,
    show: bool = False,
) -> Any:
    """
    Process and optionally save or show a Matplotlib figure.

    Handles common plot finalization: showing, saving to file, and
    returning figure or metadata based on return_fig flag.

    Args:
        fig: Matplotlib figure to process
        save_to: Directory where the figure should be saved. Figure won't be saved if None
        filename: Name of the file to save
        return_fig: Whether to return the figure in the result. Defaults to False
        show: Whether to display the figure. Defaults to False

    Returns:
        If return_fig=False: The processed figure
        If return_fig=True: Tuple of (fig, save_to, filename)

    Example:
        >>> result = process_plot(fig, save_to='./plots', filename='output.png', return_fig=True, show=False)
    """
    if show:
        from matplotlib import pyplot as plt

        plt.show()

    if hasattr(fig, "patch"):
        fig.patch.set_visible(False)

    if return_fig:
        return fig, save_to, filename
    else:
        if save_to:
            os.makedirs(save_to, exist_ok=True)
            filepath = os.path.join(save_to, filename)
            save_plot(fig, filepath, filename)
        return fig


def prob_hist(
    vs: Sequence["np.ndarray"],
    colors: Sequence[str],
    labels: Sequence[str],
    bins: "int | Sequence[float]",
    ax: "Axes",
    hist_type: str = "sns.hist",
    kde: bool = False,
    sns_kws: Dict[str, Any] = {},
    plot_fit: bool = True,
    **kwargs: Any,
) -> None:
    """
    Create a probability histogram or distribution plot for multiple datasets.

    Generates probability histograms using seaborn or matplotlib with optional
    KDE overlay and polynomial fit smoothing.

    Args:
        vs: List of arrays - datasets to plot
        colors: List of colors for each dataset
        labels: List of labels for the legend
        bins: Number of bins or bin edges for the histogram
        ax: Matplotlib axis object to plot on
        hist_type: Type of histogram to create ('plt.hist' or 'sns.hist'). Defaults to 'sns.hist'
        kde: Whether to overlay a kernel density estimate. Defaults to False
        sns_kws: Additional keyword arguments for Seaborn. Defaults to {}
        plot_fit: Whether to plot a smoothed fit curve. Defaults to True
        **kwargs: Additional keyword arguments for the histogram

    Example:
        >>> fig, ax = plt.subplots()
        >>> prob_hist([data1, data2], ['blue', 'green'], ['Dataset 1', 'Dataset 2'], bins=20, ax=ax)
        >>> plt.show()
    """
    for v, c, l in zip(vs, colors, labels):
        ax_kws = {"label": l, "color": c}
        if hist_type == "sns.hist":
            sns_kws0 = {
                "kde": kde,
                "stat": "probability",
                "element": "step",
                "fill": True,
                "multiple": "layer",
                "shrink": 1,
            }
            sns_kws0.update(sns_kws)
            sns.histplot(v, bins=bins, ax=ax, **ax_kws, **sns_kws0)
        elif hist_type == "plt.hist":
            y, x, patches = ax.hist(
                v, bins=bins, weights=np.ones_like(v) / len(v), **ax_kws, **kwargs
            )
            if plot_fit:
                x = x[:-1] + (x[1] - x[0]) / 2
                y_smooth = np.polyfit(x, y, 5)
                poly_y = np.poly1d(y_smooth)(x)
                ax.plot(x, poly_y, **ax_kws, linewidth=3)


def single_boxplot(
    x: str,
    y: str,
    ax: "Axes",
    data: "pd.DataFrame",
    hue: Optional[str] = None,
    palette: Any = None,
    color: Optional[str] = None,
    annotation: bool = True,
    show_ns: bool = False,
    target_only: Optional[str] = None,
    stripplot: bool = True,
    **kwargs: Any,
) -> None:
    """
    Create a single boxplot with optional annotations and stripplot.

    Generates seaborn boxplot with optional statistical annotations and
    overlaid stripplot showing individual data points.

    Args:
        x: Column name for the x-axis
        y: Column name for the y-axis
        ax: Matplotlib axes where the boxplot will be drawn
        data: DataFrame - the data source
        hue: Grouping variable that will produce boxes with different colors. Defaults to None
        palette: Color palette to use for coloring the boxes. Defaults to None
        color: Color for the boxes. Defaults to None
        annotation: Whether to annotate the plot with statistical information. Defaults to True
        show_ns: Show non-significant comparisons. Defaults to False
        target_only: Filter the data to include only a specific target. Defaults to None
        stripplot: Whether to include a stripplot alongside the boxplot. Defaults to True
        **kwargs: Additional keyword arguments to customize the boxplot

    Example:
        >>> single_boxplot(x='group', y='value', ax=ax, data=df, annotation=True, stripplot=True)
    """
    kws = {
        "x": x,
        "y": y,
        "ax": ax,
        "palette": palette,
        "color": color,
        "hue": hue,
        "data": data,
    }

    box_kws = {"width": 0.8, "fliersize": 3, "whis": 1.5, "linewidth": None}
    box_kws.update(kwargs)

    with sns.plotting_context("notebook", font_scale=1.4):
        g1 = sns.boxplot(**kws, **box_kws)
        g1.set(xlabel=None)
        try:
            g1.get_legend().remove()
        except:
            pass

        if annotation:
            try:
                annotate_plot(show_ns=show_ns, target_only=target_only, **kws)
            except:
                pass

        if stripplot:
            g2 = sns.stripplot(**kws)
            try:
                g2.get_legend().remove()
            except:
                pass
            g2.set(xlabel=None)


def configure_subplot_grid(
    N: Optional[int] = None,
    wh: Optional[float] = None,
    w: float = 8,
    h: float = 8,
    sharex: bool = False,
    sharey: bool = False,
    Ncols: Optional[int] = None,
    Nrows: Optional[int] = None,
    Nrows_coef: int = 1,
    figsize: Optional[Tuple[int, int]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Calculate grid dimensions and configure subplot grid parameters.

    Determines optimal number of rows and columns for arranging N elements
    in a grid, with customizable figure sizing and axis sharing.

    Args:
        N: Total number of elements. Defaults to None
        wh: Width and height for each subplot. Defaults to None
        w: Width for each subplot when wh is not specified. Defaults to 8
        h: Height for each subplot when wh is not specified. Defaults to 8
        sharex: Share the x-axis among subplots. Defaults to False
        sharey: Share the y-axis among subplots. Defaults to False
        Ncols: Number of columns for the subplot grid. Defaults to None
        Nrows: Number of rows for the subplot grid. Defaults to None
        Nrows_coef: Coefficient to adjust the number of rows. Defaults to 1
        figsize: Figure size (width, height). Defaults to None
        **kwargs: Additional keyword arguments passed to subplot creation function

    Returns:
        Dictionary of keyword arguments for configuring subplots including
        nrows, ncols, figsize, sharex, and sharey

    Example:
        >>> kws = configure_subplot_grid(N=6, wh=5, sharex=True, Ncols=3)
    """

    # print(figsize,wh,w,h)
    def calculate_grid_dimensions(
        N: Optional[int], Ncols: Optional[int], Nrows: Optional[int]
    ) -> Tuple[int, int]:
        if N:
            if Nrows:
                if Ncols is None:
                    Ncols = -(-N // Nrows)
            elif Ncols:
                if Nrows is None:
                    Nrows = -(-N // Ncols)
            else:
                Nrows, Ncols = (int(N**0.5), -(-N // int(N**0.5)))
        return Nrows or 1, Ncols or 1

    if Nrows is not None:
        Nrows *= Nrows_coef
    Nrows, Ncols = calculate_grid_dimensions(N, Ncols, Nrows)
    figsize = figsize or (wh * Ncols, wh * Nrows) if wh else (w * Ncols, h * Nrows)
    kws = {
        "sharex": sharex,
        "sharey": sharey,
        "ncols": Ncols,
        "nrows": Nrows,
        "figsize": figsize,
        **kwargs,
    }
    return kws


def define_end_ks(
    ks: Optional[Sequence[str]] = None, mode: str = "basic"
) -> Sequence[str]:
    """
    Define endpoint parameter shortcuts for different analysis modes.

    Returns predefined sets of endpoint parameter keys based on analysis
    mode (basic, minimal, tiny, etc.) or custom list.

    Args:
        ks: Custom parameter shortcuts. Uses mode-specific set if None
        mode: Analysis mode ('basic', 'minimal', 'tiny', 'deb', etc.). Defaults to 'basic'

    Returns:
        List of parameter shortcut keys

    Example:
        >>> ks = define_end_ks(mode='minimal')
        >>> ks = define_end_ks(ks=['fsv', 'sv_mu', 'cum_sd'])
    """
    l_par = "l"
    if ks is None:
        dic = {
            "basic": [
                l_par,
                "fsv",
                "sv_mu",
                "str_sd_mu",
                "str_tr",
                "pau_tr",
                "Ltur_tr",
                "Rtur_tr",
                "tor20_mu",
                "dsp_0_40_fin",
                "b_mu",
                "bv_mu",
            ],
            "minimal": [
                l_par,
                "fsv",
                "sv_mu",
                "str_sd_mu",
                "cum_t",
                "str_tr",
                "pau_tr",
                "tor5_std",
                "tor5_mu",
                "tor20_mu",
                "dsp_0_40_max",
                "dsp_0_40_fin",
                "b_mu",
                "bv_mu",
                "Ltur_tr",
                "Rtur_tr",
            ],
            "tiny": [
                l_par,
                "v_mu",
                "sv_mu",
                "fsv",
                "pau_tr",
                "run_tr",
                "cum_sd",
                "tor2_mu",
                "tor5_mu",
                "tor10_mu",
                "tor20_mu",
                "tor60_mu",
            ],
            "stride_def": [l_par, "fsv", "str_sd_mu", "str_sd_std"],
            "reorientation": ["str_fo_mu", "str_fo_std", "tur_fou_mu", "tur_fou_std"],
            "tortuosity": ["tor2_mu", "tor5_mu", "tor10_mu", "tor20_mu"],
            "result": ["sv_mu", "str_tr", "pau_tr", "pau_t_mu"],
            "limited": [
                l_par,
                "fsv",
                "sv_mu",
                "str_sd_mu",
                "cum_t",
                "str_tr",
                "pau_tr",
                "pau_t_mu",
                "tor5_mu",
                "tor5_std",
                "tor20_mu",
                "tor20_std",
                "tor",
                "sdsp_mu",
                "sdsp_0_40_max",
                "sdsp_0_40_fin",
                "b_mu",
                "b_std",
                "bv_mu",
                "bv_std",
                "Ltur_tr",
                "Rtur_tr",
                "Ltur_fou_mu",
                "Rtur_fou_mu",
            ],
            "deb": [
                "deb_f_mu",
                "hunger",
                "reserve_density",
                "puppation_buffer",
                "cum_d",
                "cum_sd",
                "str_N",
                "fee_N",
                "str_tr",
                "pau_tr",
                "fee_tr",
                "f_am",
                l_par,
                "m",
                # 'tor2_mu', 'tor5_mu', 'tor10_mu', 'tor20_mu',
                # 'v_mu', 'sv_mu',
            ],
        }
        if mode in dic:
            ks = dic[mode]
        else:
            raise ValueError("Provide parameter shortcuts or define a mode")
    return ks


def get_vs(
    datasets: Sequence[Any],
    par: str,
    key: str = "step",
    absolute: bool = False,
    rad2deg: bool = False,
) -> List["np.ndarray"]:
    """
    Extract parameter values from multiple datasets.

    Collects parameter arrays from datasets with optional transformations
    (absolute values, radian to degree conversion).

    Args:
        datasets: List of datasets to extract from
        par: Parameter name to extract
        key: Data key ('step', 'end', etc.). Defaults to 'step'
        absolute: Use absolute values. Defaults to False
        rad2deg: Convert radians to degrees. Defaults to False

    Returns:
        List of numpy arrays containing parameter values

    Example:
        >>> values = get_vs(datasets=[d1, d2], par='v', key='step', absolute=True)
    """
    vs: List["np.ndarray"] = []
    for d in datasets:
        v = d.get_par(par, key=key)
        if v is not None:
            v = v.dropna().values
        else:
            continue
        if absolute:
            v = np.abs(v)
        if rad2deg:
            v = np.rad2deg(v)
        vs.append(v)
    return vs


def color_epochs(
    epochs: Sequence[Tuple[int, int]],
    ax: "Axes",
    trange: Sequence[float],
    edgecolor: str = f"{0.4 * (0 + 1)}",
    facecolor: str = "lightblue",
    epoch_boundaries: bool = True,
    epoch_area: bool = True,
) -> None:
    """
    Color behavioral epochs on time series plot.

    Adds colored background regions and boundary lines for behavioral
    epochs (strides, pauses, turns) on existing axes.

    Args:
        epochs: List of (start, end) index tuples
        ax: Matplotlib axes to annotate
        trange: Time array corresponding to indices
        edgecolor: Color for epoch boundaries. Defaults to gray
        facecolor: Fill color for epoch regions. Defaults to 'lightblue'
        epoch_boundaries: Show vertical lines at boundaries. Defaults to True
        epoch_area: Fill epoch regions with color. Defaults to True

    Example:
        >>> color_epochs([(10, 20), (30, 40)], ax, time_array, facecolor='green')
    """
    if epoch_boundaries:
        for s0, s1 in epochs:
            for s01 in [s0, s1]:
                ax.axvline(
                    trange[s01],
                    color=edgecolor,
                    alpha=0.3,
                    linestyle="dashed",
                    linewidth=1,
                )

    if epoch_area:
        for s0, s1 in epochs:
            ax.axvspan(trange[s0], trange[s1], color=facecolor, alpha=1.0)
