import itertools
import os
import numpy as np
from matplotlib import pyplot as plt, patches, transforms, ticker
from scipy.stats import mannwhitneyu
import warnings

from larvaworld.lib import reg, aux

warnings.simplefilter(action='ignore', category=FutureWarning)


suf = 'pdf'


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
    from statannotations.Annotator import Annotator

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
            # print(hue, subID)
            # print(len(dd0),len(dd1))
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


def save_plot(fig, filepath, filename, verbose=1):
    fig.savefig(filepath, dpi=300, facecolor=None)

    plt.close(fig)
    reg.vprint(f'Plot {filename} saved as {filepath}', verbose=verbose)


def plot_config(datasets, labels, save_to, subfolder=None):
    if labels is None:
        labels = [d.id for d in datasets]
    Ndatasets = len(datasets)
    if Ndatasets != len(labels):
        raise ValueError(f'Number of labels {len(labels)} does not much number of datasets {Ndatasets}')

    def get_colors(datasets):
        try:
            cs = [d.config['color'] for d in datasets]
            u_cs = aux.unique_list(cs)
            if len(u_cs) == len(cs) and None not in u_cs:
                colors = cs
            elif len(u_cs) == len(cs) - 1 and cs[-1] in cs[:-1] and 'black' not in cs:
                cs[-1] = 'black'
                colors = cs
            else:
                colors = aux.N_colors(Ndatasets)
        except:
            colors = aux.N_colors(Ndatasets)
        return colors

    cols = get_colors(datasets)
    if save_to is not None:
        if subfolder is not None:
            save_to = f'{save_to}/{subfolder}'
        os.makedirs(save_to, exist_ok=True)
    return Ndatasets, cols, save_to, labels


def process_plot(fig, save_to, filename, return_fig=False, show=False, verbose=1):
    if show:
        # raise
        plt.show()
    fig.patch.set_visible(False)

    if return_fig:
        res = fig, save_to, filename
    else:
        res = fig
        if save_to is not None:
            os.makedirs(save_to, exist_ok=True)
            filepath = os.path.join(save_to, filename)
            save_plot(fig, filepath, filename, verbose=verbose)
    return res


def scatter_hist(xs, ys, labels, colors, Nbins=40, xlabel=None, ylabel=None, cumylabel=None, ylim=None, fig=None,
                 cumy=False):
    ticksize = 15
    labelsize = 15
    labelsize2 = 20
    # definitions for the axes
    left, width = 0.15, 0.6
    bottom, height = 0.12, 0.4
    dh = 0.01
    # dw = 0.01
    h = 0.2
    if not cumy:
        height += h
    h1 = bottom + dh + h
    h2 = h1 + height + dh
    w1 = left + width + dh

    y0, y1 = np.min([np.min(y) for y in ys]), np.max([np.max(y) for y in ys])
    ybins = np.linspace(y0, y1, Nbins)
    if ylim is None:
        ylim = (y0, y1)
    # ymax=0.4
    show_zero = True if ylim is not None and ylim[0] == -ylim[1] else False
    x0, x1 = np.min([np.min(x) for x in xs]), np.max([np.max(x) for x in xs])
    xbins = np.linspace(x0, x1, Nbins)
    dx = xbins[1] - xbins[0]
    xbin_mids = xbins[:-1] + dx / 2

    rect_scatter = [left, h1, width, height]
    rect_cumy = [left, h2, width, 1.1 * h]
    rect_histy = [w1 + dh, h1, h, height]
    rect_histx = [left, bottom, width, h]

    # start with a rectangular Figure
    if fig is None:
        fig = plt.figure(figsize=(10, 8))
    cc = {
        'left': True,
        'top': False,
        'bottom': True,
        'right': False,
        'labelsize': ticksize,
        'direction': 'in',
    }
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(labelbottom=False, **cc)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(**cc)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(labelleft=False, **cc)

    ax_scatter.set_xlim([x0, x1])
    ax_scatter.set_ylim(ylim)
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_histy.set_xlabel('pdf', fontsize=labelsize)
    if xlabel is not None:
        ax_histx.set_xlabel(xlabel, fontsize=labelsize2)
    if ylabel is not None:
        ax_scatter.set_ylabel(ylabel, fontsize=labelsize2)

    if cumy:
        ax_cumy = plt.axes(rect_cumy)
        ax_cumy.tick_params(labelbottom=False, **cc)
        ax_cumy.set_xlim(ax_scatter.get_xlim())
    xmax_ps, ymax_ps = [], []
    for x, y, l, c in zip(xs, ys, labels, colors):
        ax_scatter.scatter(x, y, marker='.', color=c, alpha=1.0, label=l)
        if show_zero:
            ax_scatter.axhline(0.0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)

        yw = np.ones_like(y) / float(len(y))
        y_vs0, y_vs1, y_patches = ax_histy.hist(y, bins=ybins, weights=yw, color=c, alpha=0.5, orientation='horizontal')

        y_vs1 = y_vs1[:-1] + (y_vs1[1] - y_vs1[0]) / 2
        y_smooth = np.polyfit(y_vs1, y_vs0, 5)
        poly_y = np.poly1d(y_smooth)(y_vs1)
        ax_histy.plot(poly_y, y_vs1, color=c, linewidth=2)

        xw = np.ones_like(x) / float(len(x))
        x_vs0, x_vs1, x_patches = ax_histx.hist(x, bins=xbins, weights=xw, color=c, alpha=0.5)
        x_vs1 = x_vs1[:-1] + (x_vs1[1] - x_vs1[0]) / 2
        x_smooth = np.polyfit(x_vs1, x_vs0, 5)
        poly_x = np.poly1d(x_smooth)(x_vs1)
        ax_histx.plot(x_vs1, poly_x, color=c, linewidth=2)

        xmax_ps.append(np.max(x_vs0))
        ymax_ps.append(np.max(y_vs0))
        ax_histx.set_ylabel('pdf', fontsize=labelsize)
        if cumy:
            xbinned_y = [y[(x0 <= x) & (x < x1)] for x0, x1 in zip(xbins[:-1], xbins[1:])]
            cum_y = np.array([np.sum(y) / len(y) for y in xbinned_y])
            ax_cumy.plot(xbin_mids, cum_y, color=c, alpha=0.5)
            if show_zero:
                ax_cumy.axhline(0.0, color='green', alpha=0.5, linestyle='dashed', linewidth=1)
            if cumylabel is not None:
                ax_cumy.set_ylabel(cumylabel, fontsize=labelsize)
    ax_histx.set_ylim([0.0, np.max(xmax_ps) + 0.05])
    ax_histy.set_xlim([0.0, np.max(ymax_ps) + 0.05])
    dataset_legend(labels, colors, ax=ax_scatter, loc='upper left', anchor=(1.0, 1.6) if cumy else None, fontsize=10)

    # plt.show()
    # raise
    return fig


def get_figsize(Ncols, Nrows, wh=None, w=8, h=8):
    if wh is not None:
        w = wh
        h = wh
    figsize = (w * Ncols, h * Nrows)
    return figsize


def getNcolsNrows(N=None, Ncols=None, Nrows=None):
    if N is not None:
        if Nrows is None and Ncols is not None:
            Nrows = int(np.ceil(N / Ncols))
        elif Ncols is None and Nrows is not None:
            Ncols = int(np.ceil(N / Nrows))
        elif Ncols is None and Nrows is None:
            Ncols = int(np.sqrt(N))
            Nrows = int(np.ceil(N / Ncols))
    if Nrows is None:
        Nrows = 1
    if Ncols is None:
        Ncols = 1
    return Nrows, Ncols


def sharexy(mode=None, sharex=False, sharey=False):
    if mode == 'box':
        sharex, sharey = True, False
    elif mode == 'hist':
        sharex, sharey = False, True
    elif mode == 'both':
        sharex, sharey = True, True

    kws2 = {'sharex': sharex, 'sharey': sharey}
    return kws2


def NcolNrows0(N=None, wh=None, w=8, h=8, Ncols=None, Nrows=None, figsize=None):
    Nrows, Ncols = getNcolsNrows(N=N, Ncols=Ncols, Nrows=Nrows)
    # Nplots=Ncols*Nrows

    if figsize is None:
        figsize = get_figsize(Ncols, Nrows, wh=wh, w=w, h=h)
    kws = {
        'ncols': Ncols,
        'nrows': Nrows,
        'figsize': figsize,
        # **kws2, **kwargs
        # 'Ncols' : Ncols,
    }
    return kws


def NcolNrows(N=None, wh=None, w=8, h=8, mode=None, sharex=False, sharey=False, Ncols=None, Nrows=None,Nrows_coef=1, figsize=None,
              **kwargs):
    if Nrows is not None:
        Nrows*=Nrows_coef
    kws1 = NcolNrows0(N=N, Ncols=Ncols, Nrows=Nrows, wh=wh, w=w, h=h, figsize=figsize)

    kws2 = sharexy(mode=mode, sharex=sharex, sharey=sharey)

    kws = {
        **kws1,
        **kws2, **kwargs
        # 'Ncols' : Ncols,
    }
    return kws
    # Ncols = Ncols, Nrows = Nrows, figsize = (8 * Ncols, 8 * Nrows)
