import itertools
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
# matplotlib.use('Agg')
from matplotlib import pyplot as plt, patches, transforms
from scipy.stats import mannwhitneyu

from larvaworld.lib import reg

warnings.simplefilter(action='ignore', category=FutureWarning)


suf = 'pdf'

def plot_quantiles(df, x=None, **kwargs):
    """
    Plot quantiles or confidence intervals along with the mean.

    Parameters:
    - df: numpy.ndarray, pandas.DataFrame, or pandas.Series
        The data to compute quantiles from. If it's a numpy.ndarray,
        quantiles are computed along the first axis (columns).
        If it's a pandas.DataFrame or pandas.Series, quantiles are
        computed grouped by the 'Step' level.
    - x: numpy.ndarray, optional
        The x-axis values for the plot. If None, a default range is used.
    - **kwargs: keyword arguments
        Additional keyword arguments to be passed to the plot_mean_and_range function.

    Raises:
    - Exception if the input data type is not recognized.

    Example usage:
    plot_quantiles(data_array, x_values, axis=plt.gca(), color_shading='blue', label='Data')
    """
    if isinstance(df, np.ndarray):
        mean = np.nanquantile(df, q=0.5, axis=0)
        ub = np.nanquantile(df, q=0.75, axis=0)
        lb = np.nanquantile(df, q=0.25, axis=0)
        if x is None:
            x = np.arange(len(mean))
    elif isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        mean = df.groupby(level='Step').quantile(q=0.5)
        ub = df.groupby(level='Step').quantile(q=0.75)
        lb = df.groupby(level='Step').quantile(q=0.25)
    else:
        raise Exception("Input data type not recognized.")
    plot_mean_and_range(x=x, mean=mean, lb=lb, ub=ub, **kwargs)


def plot_mean_and_range(x, mean, lb, ub, axis, color_shading, color_mean=None, label=None, linestyle='solid', linewidth=2):
    """
    Plot the mean and a shaded range (quantiles or confidence intervals).

    Parameters:
    - x: numpy.ndarray
        The x-axis values for the plot.
    - mean: numpy.ndarray or pandas.Series
        The mean values to be plotted.
    - lb: numpy.ndarray or pandas.Series
        The lower bound of the range.
    - ub: numpy.ndarray or pandas.Series
        The upper bound of the range.
    - axis: matplotlib.axes.Axes
        The axis where the plot will be drawn.
    - color_shading: str
        The color of the shaded range.
    - color_mean: str, optional
        The color of the mean line. If None, it will be set to color_shading.
    - label: str, optional
        The label for the legend.
    - linestyle: str, optional
        The line style for the mean line (e.g., 'solid', 'dashed').
    - linewidth: int, optional
        The line width for the mean line.

    Example usage:
    plot_mean_and_range(x_values, mean_values, lower_bound, upper_bound, plt.gca(), color_shading='blue', label='Mean and Range')
    """
    N = mean.shape[0]
    if x.shape[0] > N:
        xx = x[:N]
    elif x.shape[0] == N:
        xx = x
    else:
        raise Exception("Incompatible input shapes.")
    if color_mean is None:
        color_mean = color_shading
    # Plot the shaded range (confidence intervals)
    axis.fill_between(xx, ub, lb, color=color_shading, alpha=0.2, zorder=0)
    # Plot the mean on top
    axis.plot(xx, mean, color_mean, label=label, linewidth=linewidth, alpha=1.0, zorder=10, linestyle=linestyle)



def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, **kwargs):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
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
        radius = (area / np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    patches = plt.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                      edgecolor='black', fill=True, linewidth=2, **kwargs)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


def circNarrow(ax, data, alpha, label, color, Nbins=16):
    circular_hist(ax, data, bins=Nbins, alpha=alpha, label=label, color=color, offset=np.pi / 2)
    arrow = patches.FancyArrowPatch((0, 0), (np.mean(data), 0.3), zorder=2, mutation_scale=30, alpha=alpha,
                                    facecolor=color, edgecolor='black', fill=True, linewidth=0.5)
    ax.add_patch(arrow)


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
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
    ellipse = patches.Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                              facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def dataset_legend(labels, colors, ax=None, anchor=None, handlelength=0.5, handleheight=0.5, **kwargs):
    kws = {
        'handles': [patches.Patch(facecolor=c, label=l, edgecolor='black') for c, l in zip(colors, labels)],
        'handlelength': handlelength,
        'handleheight': handleheight,
        'labels': labels,
        'bbox_to_anchor': anchor,
        **kwargs
    }

    if ax is None:
        leg = plt.legend(**kws)
    else:
        leg = ax.legend(**kws)
        ax.add_artist(leg)
    return leg


def label_diff(i, j, text, X, Y, ax):
    x = (X[i] + X[j]) / 2
    y = 1.5 * max(Y[i], Y[j])
    dx = abs(X[i] - X[j])

    props = {'connectionstyle': 'bar', 'arrowstyle': '-', \
             'shrinkA': 20, 'shrinkB': 20, 'linewidth': 2}
    ax.annotate(text, xy=(X[i], y), zorder=10)
    # ax.annotate(text, xy=(X[i], y), zorder=10)
    ax.annotate('', xy=(X[i], y), xytext=(X[j], y), arrowprops=props)

def pvalue_star(pv):
    a = {1e-4: "****", 1e-3: "***",
         1e-2: "**", 0.05: "*", 1: "ns"}
    for k, v in a.items():
        if pv < k:
            return v
    return "ns"



def annotate_plot(data, x, y, hue=None, show_ns=True, target_only=None, **kwargs):
    """
    Annotate a plot with Mann-Whitney U test p-values.

    Parameters:
    - data: DataFrame
        The input data.
    - x: str
        The column name for the x-axis variable.
    - y: str
        The column name for the y-axis variable.
    - hue: str or None
        The column name for grouping data by hue (optional).
    - show_ns: bool
        Whether to display annotations for non-significant comparisons (default: True).
    - target_only: Any or None
        Specify a target value for comparisons (optional).
    - **kwargs: keyword arguments
        Additional arguments for annotation customization.

    Returns:
    - None
    """
    from statannotations.Annotator import Annotator
    subIDs0 = np.unique(data[x].values)

    if hue is not None:
        h1, h2 = np.unique(data[hue].values)
        pairs = [((subID, h1), (subID, h2)) for subID in subIDs0]
        pvs = [mannwhitneyu(
            data[(data[x] == subID) & (data[hue] == h1)][y].dropna(),
            data[(data[x] == subID) & (data[hue] == h2)][y].dropna(),
            alternative="two-sided").pvalue for subID in subIDs0]
    else:
        if target_only is None:
            pairs = list(itertools.combinations(subIDs0, 2))
            pvs = [mannwhitneyu(
                data[data[x] == subID0][y].dropna(),
                data[data[x] == subID1][y].dropna(),
                alternative="two-sided").pvalue for subID0, subID1 in pairs]
        else:
            pairs = [(target_only, subID) for subID in subIDs0 if subID != target_only]
            dd0 = data[data[x] == target_only][y].dropna()
            pvs = [mannwhitneyu(
                dd0,
                data[data[x] == subID][y].dropna(),
                alternative="two-sided").pvalue for subID in subIDs0 if subID != target_only]

    f_pvs = [pvalue_star(pv) for pv in pvs]

    if not show_ns:
        valid_idx = [i for i, f_pv in enumerate(f_pvs) if f_pv != 'ns']
        pairs = [pairs[i] for i in valid_idx]
        f_pvs = [f_pvs[i] for i in valid_idx]

    if len(pairs) > 0:
        annotator = Annotator(pairs=pairs, data=data, x=x, y=y, hue=hue, **kwargs)
        annotator.verbose = False
        annotator.annotate_custom_annotations(f_pvs)


def dual_half_circle(center, radius=0.04, angle=90, ax=None, colors=('W', 'k'), **kwargs):
    """
    Add two half circles to the axes 'ax' (or the current axes) with the specified face colors 'colors' rotated at 'angle' (in degrees).

    Parameters:
    - center: tuple
        Center coordinates of the half circles.
    - radius: float, optional (default: 0.04)
        Radius of the half circles.
    - angle: float, optional (default: 90)
        Angle by which the half circles are rotated (in degrees).
    - ax: matplotlib.axes.Axes, optional (default: None)
        The axes to which the half circles will be added. If None, the current axes are used.
    - colors: tuple, optional (default: ('W', 'k'))
        Face colors of the two half circles. The first color is for the left half, and the second color is for the right half.
    - **kwargs: keyword arguments
        Additional keyword arguments to customize the appearance of the half circles.

    Returns:
    - wedge_list: list
        A list containing the two half circle patches.
    """
    if ax is None:
        ax = plt.gca()
    theta1, theta2 = angle, angle + 180
    w1 = patches.Wedge(center, radius, theta1, theta2, fc=colors[0], **kwargs)
    w2 = patches.Wedge(center, radius, theta2, theta1, fc=colors[1], **kwargs)
    ax.add_patch(w1)
    ax.add_patch(w2)
    return [w1, w2]


def save_plot(fig, filepath, filename):
    """
        Save a Matplotlib figure to a specified file path.

        Parameters:
        - fig: matplotlib.figure.Figure
            The figure to save.
        - filepath: str
            The full file path where the figure should be saved.
        - filename: str
            The name of the file to save.

        Returns:
        None
    """
    fig.savefig(filepath, dpi=300, facecolor=None)
    try:
        plt.close(fig)
    except:
        pass
    reg.vprint(f'Plot {filename} saved as {filepath}', 1)


def process_plot(fig, save_to, filename, return_fig=False, show=False):
    """
        Process and optionally save or show a Matplotlib figure.

        Parameters:
        - fig: matplotlib.figure.Figure
            The figure to process.
        - save_to: str or None
            The directory where the figure should be saved. If None, the figure won't be saved.
        - filename: str
            The name of the file to save.
        - return_fig: bool
            Whether to return the figure in the result.
        - show: bool
            Whether to display the figure.

        Returns:
        - fig: matplotlib.figure.Figure (if return_fig=False)
            The processed figure.
        - save_to (if return_fig=True)
        - filename (if return_fig=True)
    """
    if show:
        plt.show()
    if hasattr(fig, 'patch'):
        fig.patch.set_visible(False)

    if return_fig:
        return fig, save_to, filename
    else:
        if save_to is not None:
            os.makedirs(save_to, exist_ok=True)
            filepath = os.path.join(save_to, filename)
            save_plot(fig, filepath, filename)
        return fig

def prob_hist(vs, colors, labels, bins, ax, hist_type='plt.hist', kde=False, sns_kws={}, plot_fit=True, **kwargs):
    """
    Create a probability histogram or distribution plot for multiple datasets.

    Parameters:
    - vs: list of arrays
        List of datasets to plot.
    - colors: list of str
        List of colors for each dataset.
    - labels: list of str
        List of labels for the legend.
    - bins: int or list
        Number of bins or bin edges for the histogram.
    - ax: Matplotlib axis object
        The axis to plot on.
    - hist_type: str ('plt.hist' or 'sns.hist')
        Type of histogram to create (default: 'plt.hist').
    - kde: bool
        Whether to overlay a kernel density estimate (default: False).
    - sns_kws: dict
        Additional keyword arguments for Seaborn (default: {}).
    - plot_fit: bool
        Whether to plot a smoothed fit curve (default: True).
    - **kwargs: keyword arguments
        Additional keyword arguments for the histogram.

    Returns:
    None

    Example usage:
    fig, ax = plt.subplots()
    prob_hist([data1, data2], ['blue', 'green'], ['Dataset 1', 'Dataset 2'], bins=20, ax=ax)
    plt.show()
    """
    for v, c, l in zip(vs, colors, labels):
        ax_kws = {'label': l, 'color': c}
        if hist_type == 'sns.hist':
            sns_kws0 = {'kde': kde, 'stat': 'probability', 'element': 'step', 'fill': True, 'multiple': 'layer', 'shrink': 1}
            sns_kws0.update(sns_kws)
            sns.histplot(v, bins=bins, ax=ax, **ax_kws, **sns_kws0)
        elif hist_type == 'plt.hist':
            y, x, patches = ax.hist(v, bins=bins, weights=np.ones_like(v) / len(v), **ax_kws, **kwargs)
            if plot_fit:
                x = x[:-1] + (x[1] - x[0]) / 2
                y_smooth = np.polyfit(x, y, 5)
                poly_y = np.poly1d(y_smooth)(x)
                ax.plot(x, poly_y, **ax_kws, linewidth=3)




def single_boxplot(x, y, ax, data, hue=None, palette=None, color=None,
                   annotation=True, show_ns=False, target_only=None, stripplot=True, **kwargs):
    """
    Create a single boxplot with optional annotations and stripplot.

    Parameters:
    - x: str
        Column name for the x-axis.
    - y: str
        Column name for the y-axis.
    - ax: matplotlib.axes.Axes
        The axes where the boxplot will be drawn.
    - data: pandas.DataFrame
        The data source.
    - hue: str, optional
        Grouping variable that will produce boxes with different colors.
    - palette: str or dict, optional
        Color palette to use for coloring the boxes.
    - color: str, optional
        Color for the boxes.
    - annotation: bool, optional
        Whether to annotate the plot with additional information.
    - show_ns: bool, optional
        Show notches on the boxes.
    - target_only: str, optional
        Filter the data to include only a specific target.
    - stripplot: bool, optional
        Whether to include a stripplot alongside the boxplot.
    - **kwargs: keyword arguments
        Additional keyword arguments to customize the boxplot.

    Returns:
    - None
    """
    kws = {'x': x, 'y': y, 'ax': ax, 'palette': palette, 'color': color, 'hue': hue, 'data': data}

    box_kws = {'width': 0.8, 'fliersize': 3, 'whis': 1.5, 'linewidth': None}
    box_kws.update(kwargs)

    with sns.plotting_context('notebook', font_scale=1.4):
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




def configure_subplot_grid(N=None, wh=None, w=8, h=8, sharex=False, sharey=False, Ncols=None, Nrows=None, Nrows_coef=1, figsize=None, **kwargs):
    """
    Calculate the number of rows and columns for arranging N elements in a grid and configure subplot grid parameters.

    Parameters:
    - N: int or None
        Total number of elements (optional).
    - wh: float or None
        Width and height for each subplot (optional).
    - w: float
        Width for each subplot when wh is not specified (default: 8).
    - h: float
        Height for each subplot when wh is not specified (default: 8).
    - sharex: bool
        Share the x-axis among subplots (default: False).
    - sharey: bool
        Share the y-axis among subplots (default: False).
    - Ncols: int or None
        Number of columns for the subplot grid (optional).
    - Nrows: int or None
        Number of rows for the subplot grid (optional).
    - Nrows_coef: int
        Coefficient to adjust the number of rows (default: 1).
    - figsize: tuple or None
        Figure size (optional).
    - **kwargs: keyword arguments
        Additional keyword arguments to be passed to the subplot creation function.

    Returns:
    - kws: dict
        A dictionary of keyword arguments for configuring subplots.
    """
    def calculate_grid_dimensions(N, Ncols, Nrows):
        if N:
            Nrows, Ncols = Nrows or -(-N // Ncols), Ncols or -(-N // Nrows) if Nrows else (int(N**0.5), -(-N // int(N**0.5)))
        return Nrows or 1, Ncols or 1

    if Nrows is not None:
        Nrows *= Nrows_coef
    Nrows, Ncols = calculate_grid_dimensions(N, Ncols, Nrows)
    figsize = figsize or (wh * Ncols, wh * Nrows) if wh else (w * Ncols, h * Nrows)

    kws = {
        'sharex': sharex,
        'sharey': sharey,
        'ncols': Ncols,
        'nrows': Nrows,
        'figsize': figsize,
        **kwargs
    }
    return kws



def define_end_ks(ks=None, mode='basic'):
    l_par = 'l'
    if ks is None:
        dic = {
            'basic': [l_par, 'fsv', 'sv_mu', 'str_sd_mu',
                      'str_tr', 'pau_tr', 'Ltur_tr', 'Rtur_tr',
                      'tor20_mu', 'dsp_0_40_fin', 'b_mu', 'bv_mu'],
            'minimal': [l_par, 'fsv', 'sv_mu', 'str_sd_mu',
                        'cum_t', 'str_tr', 'pau_tr', 'tor5_std',
                        'tor5_mu', 'tor20_mu', 'dsp_0_40_max', 'dsp_0_40_fin',
                        'b_mu', 'bv_mu', 'Ltur_tr', 'Rtur_tr'],
            'tiny': [l_par,'v_mu','sv_mu', 'fsv',
                      'pau_tr','run_tr', 'cum_sd',
                     'tor2_mu','tor5_mu', 'tor10_mu', 'tor20_mu','tor60_mu'],
            'stride_def': [l_par, 'fsv', 'str_sd_mu', 'str_sd_std'],
            'reorientation': ['str_fo_mu', 'str_fo_std', 'tur_fou_mu', 'tur_fou_std'],
            'tortuosity': ['tor2_mu', 'tor5_mu', 'tor10_mu', 'tor20_mu'],
            'result': ['sv_mu', 'str_tr', 'pau_tr', 'pau_t_mu'],
            'limited': [l_par, 'fsv', 'sv_mu', 'str_sd_mu',
                        'cum_t', 'str_tr', 'pau_tr', 'pau_t_mu',
                        'tor5_mu', 'tor5_std', 'tor20_mu', 'tor20_std',
                        'tor', 'sdsp_mu', 'sdsp_0_40_max', 'sdsp_0_40_fin',
                        'b_mu', 'b_std', 'bv_mu', 'bv_std',
                        'Ltur_tr', 'Rtur_tr', 'Ltur_fou_mu', 'Rtur_fou_mu'],

            'deb': [
                'deb_f_mu', 'hunger', 'reserve_density', 'puppation_buffer',
                'cum_d', 'cum_sd', 'str_N', 'fee_N',
                'str_tr', 'pau_tr', 'fee_tr', 'f_am',
                l_par, 'm'
                # 'tor2_mu', 'tor5_mu', 'tor10_mu', 'tor20_mu',
                # 'v_mu', 'sv_mu',

            ]
        }
        if mode in dic.keys():
            ks = dic[mode]
        else:
            raise ValueError('Provide parameter shortcuts or define a mode')
    return ks

def get_vs(datasets, par, key='step',absolute=False, rad2deg=False):
    vs = []
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