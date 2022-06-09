import itertools
import os

import numpy as np
from matplotlib import pyplot as plt, patches, transforms, ticker

from lib.aux import dictsNlists as dNl
from lib.aux.colsNstr import N_colors

plt_conf = {'axes.labelsize': 20,
            'axes.titlesize': 25,
            'figure.titlesize': 25,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 20,
            'legend.title_fontsize': 20}
plt.rcParams.update(plt_conf)


def plot_quantiles(df, from_np=False, x=None, **kwargs):
    if from_np:
        df_m = np.nanquantile(df, q=0.5, axis=0)
        df_u = np.nanquantile(df, q=0.75, axis=0)
        df_b = np.nanquantile(df, q=0.25, axis=0)
        if x is None:
            x = np.arange(len(df_m))
    else:
        df_m = df.groupby(level='Step').quantile(q=0.5)
        df_u = df.groupby(level='Step').quantile(q=0.75)
        df_b = df.groupby(level='Step').quantile(q=0.25)
    plot_mean_and_range(x=x, mean=df_m, lb=df_b, ub=df_u, **kwargs)


def plot_mean_and_range(x, mean, lb, ub, axis, color_shading, color_mean=None, label=None, linewidth=2):
    if x.shape[0] > mean.shape[0]:
        xx = x[:mean.shape[0]]
    elif x.shape[0] == mean.shape[0]:
        xx = x
    if color_mean is None:
        color_mean = color_shading
    # plot the shaded range of e.g. the confidence intervals
    axis.fill_between(xx, ub, lb, color=color_shading, alpha=.2, zorder=0)
    # plot the mean on top
    if label is not None:
        axis.plot(xx, mean, color_mean, label=label, linewidth=linewidth, alpha=1.0, zorder=10)
    else:
        axis.plot(xx, mean, color_mean, linewidth=linewidth, alpha=1.0, zorder=10)

    # pass


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


def circNarrow(ax, data, alpha, label, color):
    circular_hist(ax, data, bins=16, alpha=alpha, label=label, color=color, offset=np.pi / 2)
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


def dataset_legend(labels, colors, ax=None, loc=None, anchor=None, fontsize=None, handlelength=0.5, handleheight=0.5,
                   **kwargs):
    if ax is None:
        leg = plt.legend(
            bbox_to_anchor=anchor,
            handles=[patches.Patch(facecolor=c, label=l, edgecolor='black') for c, l in zip(colors, labels)],
            labels=labels, loc=loc, handlelength=handlelength, handleheight=handleheight, fontsize=fontsize, **kwargs)
    else:
        leg = ax.legend(
            bbox_to_anchor=anchor,
            handles=[patches.Patch(facecolor=c, label=l, edgecolor='black') for c, l in zip(colors, labels)],
            labels=labels, loc=loc, handlelength=handlelength, handleheight=handleheight, fontsize=fontsize, **kwargs)
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

def annotate_plot(data, x, y, hue=None, show_ns=True, target_only=None, **kwargs):
    from statannotations.Annotator import Annotator
    from lib.anal.fitting import pvalue_star
    from scipy.stats import mannwhitneyu
    subIDs0 = np.unique(data[x].values)
    # print(subIDs0)
    if hue is not None:
        h1, h2 = np.unique(data[hue].values)

        pairs = [((subID, h1), (subID, h2)) for subID in subIDs0]
        pvs = []
        for subID in subIDs0:
            dd = data[data[x] == subID]
            dd0 = dd[dd[hue] == h1][y].dropna().values.tolist()
            dd1 = dd[dd[hue] == h2][y].dropna().values.tolist()
            pvs.append(mannwhitneyu(dd0, dd1, alternative="two-sided").pvalue)
    else:
        if target_only is None:
            pairs = list(itertools.combinations(subIDs0, 2))
            pvs = []
            for subID0, subID1 in pairs:
                dd0 = data[data[x] == subID0][y].dropna().values.tolist()
                dd1 = data[data[x] == subID1][y].dropna().values.tolist()
                pvs.append(mannwhitneyu(dd0, dd1, alternative="two-sided").pvalue)
        else:
            pairs = []
            pvs = []
            dd0 = data[data[x] == target_only][y].dropna().values.tolist()
            for subID in subIDs0:
                if subID != target_only:
                    pairs.append((target_only, subID))
                    dd1 = data[data[x] == subID][y].dropna().values.tolist()
                    pvs.append(mannwhitneyu(dd0, dd1, alternative="two-sided").pvalue)

    f_pvs = [pvalue_star(pv) for pv in pvs]

    if not show_ns:
        valid_idx = [i for i, f_pv in enumerate(f_pvs) if f_pv != 'ns']
        pairs = [pairs[i] for i in valid_idx]
        f_pvs = [f_pvs[i] for i in valid_idx]

    # Add annotations
    if len(pairs) > 0:
        annotator = Annotator(pairs=pairs, data=data, x=x, y=y, hue=hue, **kwargs)
        annotator.verbose = False
        annotator.annotate_custom_annotations(f_pvs)


def conf_ax_3d(vars, target, ax=None, fig=None, lims=None, title=None, maxN=5, labelpad=30, tickpad=10):
    if fig is None and ax is None:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(15, 10))
        ax = Axes3D(fig, azim=115, elev=15)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(maxN))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(maxN))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(maxN))
    ax.xaxis.set_tick_params(pad=tickpad)
    ax.yaxis.set_tick_params(pad=tickpad)
    ax.zaxis.set_tick_params(pad=tickpad)

    ax.set_xlabel(vars[0], labelpad=labelpad)
    ax.set_ylabel(vars[1], labelpad=labelpad)
    ax.set_zlabel(target, labelpad=labelpad)
    if lims is not None:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
        ax.set_zlim(lims[2])

    if title is not None:
        ax.set_suptitle(title, fontsize=20)

    return fig, ax


def dual_half_circle(center, radius, angle=0, ax=None, colors=('W', 'k'), **kwargs):
    """
    Add two half circles to the axes *ax* (or the current axes) with the
    specified facecolors *colors* rotated at *angle* (in degrees).
    """
    if ax is None:
        ax = plt.gca()
    theta1, theta2 = angle, angle + 180
    w1 = patches.Wedge(center, radius, theta1, theta2, fc=colors[0], **kwargs)
    w2 = patches.Wedge(center, radius, theta2, theta1, fc=colors[1], **kwargs)
    for wedge in [w1, w2]:
        ax.add_artist(wedge)
    return [w1, w2]


def save_plot(fig, filepath, filename=None):
    fig.savefig(filepath, dpi=300, facecolor=None)
    print(f'Plot saved as {filepath}')
    # print(fig.get_size_inches(), filename)
    # fig.clear()
    plt.close(fig)
    if filename is not None:
        pass
        # print(f'Plot saved as {filename}')


def plot_config(datasets, labels, save_to, subfolder=None):
    if labels is None:
        labels = [d.id for d in datasets]
    Ndatasets = len(datasets)
    if Ndatasets != len(labels):
        raise ValueError(f'Number of labels {len(labels)} does not much number of datasets {Ndatasets}')

    def get_colors(datasets):
        try:
            cs = [d.config['color'] for d in datasets]
            u_cs = dNl.unique_list(cs)
            if len(u_cs) == len(cs) and None not in u_cs:
                colors = cs
            elif len(u_cs) == len(cs) - 1 and cs[-1] in cs[:-1] and 'black' not in cs:
                cs[-1] = 'black'
                colors = cs
            else:
                colors = N_colors(Ndatasets)
        except:
            colors = N_colors(Ndatasets)
        return colors

    cols = get_colors(datasets)
    if save_to is not None:
        if subfolder is not None:
            save_to = f'{save_to}/{subfolder}'
        os.makedirs(save_to, exist_ok=True)
    return Ndatasets, cols, save_to, labels


def process_plot(fig, save_to, filename, return_fig=False, show=False):
    if show:
        plt.show()
    fig.patch.set_visible(False)
    if return_fig:
        res = fig, save_to, filename
    else:
        res = fig
        if save_to is not None:
            os.makedirs(save_to, exist_ok=True)
            filepath = os.path.join(save_to, filename)
            save_plot(fig, filepath, filename)
    return res
