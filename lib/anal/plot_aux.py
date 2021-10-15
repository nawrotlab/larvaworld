import os

import numpy as np
from matplotlib import pyplot as plt, patches, transforms, ticker
from matplotlib.pyplot import bar

from lib.aux.dictsNlists import unique_list
from lib.aux.colsNstr import N_colors
from lib.conf.base.par import getPar

plt_conf = {'axes.labelsize': 20,
            'axes.titlesize': 25,
            'figure.titlesize': 25,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 20,
            'legend.title_fontsize': 20}
plt.rcParams.update(plt_conf)

class Plot :
    def __init__(self, name, datasets, labels=None, subfolder=None,
                       save_fits_as=None, save_as=None, save_to=None, return_fig=False, show=False, **kwargs):
        suf = 'pdf'
        self.datasets=datasets
        self.Ndatasets, self.colors, self.save_to, self.labels = plot_config(datasets, labels, save_to, subfolder=subfolder)
        self.filename = f'{name}.{suf}' if save_as is None else save_as
        ff = f'{name}_fits.csv' if save_fits_as is None else save_fits_as
        self.fit_filename=os.path.join(self.save_to, ff) if ff is not None else None
        self.return_fig=return_fig
        self.show=show
        # self.fig=self.build(**kwargs)

    def build(self, Nrows=1, Ncols=1, figsize=None, **kwargs):
        if figsize is None :
            figsize=(6*Ncols, 5*Nrows)
        self.fig, axs = plt.subplots(Nrows, Ncols, figsize=figsize, **kwargs)
        self.axs = axs.ravel() if Nrows*Ncols > 1 else [axs]


    def conf_ax(self, idx=0, xlab=None, ylab=None, xlim=None, ylim=None, xticks=None, xticklabels=None,
                xMaxN=None, yMaxN=None, leg_loc=None, title=None):
        if ylab is not None:
            self.axs[idx].set_ylabel(ylab)
        if xlab is not None:
            self.axs[idx].set_xlabel(xlab)
        if xlim is not None:
            self.axs[idx].set_xlim(xlim)
        if ylim is not None:
            self.axs[idx].set_ylim(ylim)

        if xticks is not None:
            self.axs[idx].set_xticks(ticks=xticks)
        if xticklabels is not None:
            self.axs[idx].set_xticklabels(labels=xticklabels)
        if xMaxN is not None:
            self.axs[idx].xaxis.set_major_locator(ticker.MaxNLocator(xMaxN))
        if yMaxN is not None:
            self.axs[idx].yaxis.set_major_locator(ticker.MaxNLocator(yMaxN))
        if title is not None:
            self.axs[idx].set_title(title)
        if leg_loc is not None:
            self.axs[idx].legend(loc=leg_loc)

    def set(self, fig):
        self.fig=fig

    def get(self):
        return process_plot(self.fig, self.save_to, self.filename, self.return_fig, self.show)

# class TurnPlot(Plot) :
#     def __init__(self, absolute=True,**kwargs):
#         self.absolute = absolute
#         super().__init__(name='turns', **kwargs)
#
#
#     def build(self):
#         fig, axs = plt.subplots(1, 1, figsize=(6, 5))
#         par, xlabel = getPar('tur_fou', to_return=['d', 'l'])
#
#         ts = [d.get_par(par).dropna().values for d in self.datasets]
#
#         r = 150
#         Nbins = 30
#
#         for data, col, l in zip(ts, self.colors, self.labels):
#             if self.absolute:
#                 data = np.abs(data)
#                 r0, r1 = np.min(data), r
#
#             else:
#                 r0, r1 = -r, r
#                 Nbins *= 2
#
#             x = np.linspace(r0, r1, Nbins)
#             weights = np.ones_like(data) / float(len(data))
#             axs.hist(data, bins=x, weights=weights, label=l, color=col, alpha=1.0, histtype='step')
#
#         axs.set_ylabel('probability, $P$')
#         axs.set_xlabel(xlabel)
#         axs.set_xlim([r0, r1])
#         axs.yaxis.set_major_locator(ticker.MaxNLocator(4))
#         axs.legend(loc='upper right', fontsize=10)
#         fig.subplots_adjust(top=0.92, bottom=0.15, left=0.25, right=0.95, hspace=.005, wspace=0.05)
#         return fig
def plot_quantiles(df,from_np=False,x=None, **kwargs):
    if from_np :
        df_m = np.nanquantile(df, q=0.5, axis=0)
        df_u = np.nanquantile(df, q=0.75, axis=0)
        df_b = np.nanquantile(df, q=0.25, axis=0)
        x = np.arange(len(df_m))
    else :
        df_m = df.groupby(level='Step').quantile(q=0.5)
        df_u = df.groupby(level='Step').quantile(q=0.75)
        df_b = df.groupby(level='Step').quantile(q=0.25)
    plot_mean_and_range(x=x, mean=df_m, lb=df_b, ub=df_u,  **kwargs)

def plot_mean_and_range(x, mean, lb, ub, axis, color_shading,color_mean=None,  label=None):
    if x.shape[0] > mean.shape[0]:
        xx = x[:mean.shape[0]]
    elif x.shape[0] == mean.shape[0]:
        xx = x
    if color_mean is None :
        color_mean=color_shading
    # plot the shaded range of e.g. the confidence intervals
    axis.fill_between(xx, ub, lb, color=color_shading, alpha=.2)
    # plot the mean on top
    if label is not None:
        axis.plot(xx, mean, color_mean, label=label, linewidth=2, alpha=1.0)
    else:
        axis.plot(xx, mean, color_mean, linewidth=2, alpha=1.0)

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
    patches = bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                  edgecolor='black', fill=True, linewidth=2, **kwargs)

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


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


def save_plot(fig, filepath, filename=None):
    fig.savefig(filepath, dpi=300, facecolor=None)
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
    try:
        cs = [d.config['color'] for d in datasets]
        u_cs = unique_list(cs)
        if len(u_cs) == len(cs):
            colors = cs
        elif len(u_cs) == len(cs) - 1 and cs[-1] in cs[:-1]:
            if 'black' not in cs:
                cs[-1] = 'black'
                colors = cs
        else:
            colors = N_colors(Ndatasets)
    except:
        colors = N_colors(Ndatasets)
    if save_to is None:
        save_to = datasets[0].config['parent_plot_dir']
    if subfolder is not None:
        save_to = f'{save_to}/{subfolder}'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    return Ndatasets, colors, save_to, labels


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


def process_plot(fig, save_to, filename, return_fig, show=False):
    if show:
        plt.show()
    fig.patch.set_visible(False)
    if return_fig:
        res= fig, save_to, filename
    else:
        filepath = os.path.join(save_to, filename)
        save_plot(fig, filepath, filename)
        res= fig

    return res


def label_diff(i, j, text, X, Y, ax):
    x = (X[i] + X[j]) / 2
    y = 1.5 * max(Y[i], Y[j])
    dx = abs(X[i] - X[j])

    props = {'connectionstyle': 'bar', 'arrowstyle': '-', \
             'shrinkA': 20, 'shrinkB': 20, 'linewidth': 2}
    ax.annotate(text, xy=(X[i], y), zorder=10)
    # ax.annotate(text, xy=(X[i], y), zorder=10)
    ax.annotate('', xy=(X[i], y), xytext=(X[j], y), arrowprops=props)


def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out