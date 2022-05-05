import itertools
import os

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, patches, transforms, ticker
from matplotlib.pyplot import bar
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import mannwhitneyu, ttest_ind

from lib.anal.fitting import pvalue_star, fit_bout_distros

from lib.conf.base.par import getPar
from lib.conf.stored.conf import loadRef, kConfDict, loadConf
from lib.aux.colsNstr import N_colors
from lib.aux.dictsNlists import unique_list

plt_conf = {'axes.labelsize': 20,
            'axes.titlesize': 25,
            'figure.titlesize': 25,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 20,
            'legend.title_fontsize': 20}
plt.rcParams.update(plt_conf)


class BasePlot:
    def __init__(self, name, save_to='.', save_as=None, return_fig=False, show=False, suf='pdf', **kwargs):
        self.filename = f'{name}.{suf}' if save_as is None else save_as
        self.return_fig = return_fig
        self.show = show
        self.fit_df = None
        self.save_to = save_to

    def build(self, Nrows=1, Ncols=1, figsize=None, fig=None, axs=None, dim3=False, azim=115, elev=15, **kwargs):
        if fig is None and axs is None:
            if figsize is None:
                figsize = (12 * Ncols, 10 * Nrows)
            if not dim3:
                self.fig, axs = plt.subplots(Nrows, Ncols, figsize=figsize, **kwargs)
                self.axs = axs.ravel() if Nrows * Ncols > 1 else [axs]
            else:
                self.fig = plt.figure(figsize=(15, 10))
                ax = Axes3D(self.fig, azim=azim, elev=elev)
                self.axs = [ax]
        else:
            self.fig = fig
            self.axs = axs if type(axs) == list else [axs]

    def conf_ax(self, idx=0, xlab=None, ylab=None, zlab=None, xlim=None, ylim=None, zlim=None, xticks=None,
                xticklabels=None, yticks=None, xticklabelrotation=None, yticklabelrotation=None,
                yticklabels=None, zticks=None, zticklabels=None, xtickpos=None, xtickpad=None, ytickpad=None,
                ztickpad=None,
                xlabelpad=None, ylabelpad=None, zlabelpad=None,
                xMaxN=None, yMaxN=None, zMaxN=None, xMath=None, tickMath=None, ytickMath=None, leg_loc=None,
                leg_handles=None,
                title=None):
        ax = self.axs[idx]
        if ylab is not None:
            ax.set_ylabel(ylab, labelpad=ylabelpad)
        if xlab is not None:
            ax.set_xlabel(xlab, labelpad=xlabelpad)
        if zlab is not None:
            ax.set_zlabel(zlab, labelpad=zlabelpad)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if zlim is not None:
            ax.set_zlim(zlim)
        if xticks is not None:
            ax.set_xticks(ticks=xticks)
        if xticklabels is not None:
            ax.set_xticklabels(labels=xticklabels, rotation=xticklabelrotation)
        if yticks is not None:
            ax.set_yticks(ticks=yticks)
        if yticklabels is not None:
            ax.set_yticklabels(labels=yticklabels, rotation=yticklabelrotation)
        if zticks is not None:
            ax.set_zticks(ticks=zticks)
        if zticklabels is not None:
            ax.set_zticklabels(labels=zticklabels)
        if tickMath is not None:
            ax.ticklabel_format(useMathText=True, scilimits=tickMath)
        if ytickMath is not None:
            ax.ticklabel_format(axis='y', useMathText=True, scilimits=ytickMath, useOffset=True)
        if xMaxN is not None:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(xMaxN))
        if yMaxN is not None:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(yMaxN))
        if zMaxN is not None:
            ax.zaxis.set_major_locator(ticker.MaxNLocator(zMaxN))
        if xMath is not None:
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True, useMathText=True))
        if xtickpos is not None:
            ax.xaxis.set_ticks_position(xtickpos)
        if title is not None:
            ax.set_title(title)
        if xtickpad is not None:
            ax.xaxis.set_tick_params(pad=xtickpad)
        if ytickpad is not None:
            ax.yaxis.set_tick_params(pad=ytickpad)
        if ztickpad is not None:
            ax.zaxis.set_tick_params(pad=ztickpad)

        if leg_loc is not None:
            if leg_handles is not None:
                ax.legend(handles=leg_handles, loc=leg_loc)
            else:
                ax.legend(loc=leg_loc)

    def adjust(self, LR=None, BT=None, W=None, H=None):
        kws = {}
        if LR is not None:
            kws['left'] = LR[0]
            kws['right'] = LR[1]
        if BT is not None:
            kws['bottom'] = BT[0]
            kws['top'] = BT[1]
        if W is not None:
            kws['wspace'] = W
        if H is not None:
            kws['hspace'] = H
        self.fig.subplots_adjust(**kws)

    def set(self, fig):
        self.fig = fig

    def get(self):
        if self.fit_df is not None:
            self.fit_df.to_csv(self.fit_filename, index=True, header=True)
        return process_plot(self.fig, self.save_to, self.filename, self.return_fig, self.show)


class ParPlot(BasePlot):
    def __init__(self, name, pref=None, **kwargs):
        if pref is not None:
            name = f'{pref}_{name}'
        super().__init__(name, **kwargs)

    def conf_ax_3d(self, vars, target, lims=None, title=None, maxN=3, labelpad=15, tickpad=5, idx=0):
        if lims is None:
            xlim, ylim, zlim = None, None, None
        else:
            xlim, ylim, zlim = lims
        self.conf_ax(idx=idx, xlab=vars[0], ylab=vars[1], zlab=target, xlim=xlim, ylim=ylim, zlim=zlim,
                     xtickpad=tickpad, ytickpad=tickpad, ztickpad=tickpad,
                     xlabelpad=labelpad, ylabelpad=labelpad, zlabelpad=labelpad,
                     xMaxN=maxN, yMaxN=maxN, zMaxN=maxN, title=title)


class Plot(BasePlot):
    def __init__(self, name, datasets, labels=None, subfolder=None, save_fits_as=None, save_to=None, add_samples=False,
                 **kwargs):

        if add_samples:
            targetIDs = unique_list([d.config['sample'] for d in datasets])

            targets = [loadRef(id) for id in targetIDs if id in kConfDict('Ref')]
            datasets += targets
            if labels is not None:
                labels += targetIDs
        self.Ndatasets, self.colors, save_to, self.labels = plot_config(datasets, labels, save_to,
                                                                        subfolder=subfolder)
        super().__init__(name, save_to=save_to, **kwargs)
        self.datasets = datasets
        ff = f'{name}_fits.csv' if save_fits_as is None else save_fits_as
        self.fit_filename = os.path.join(self.save_to, ff) if ff is not None else None
        self.fit_ind = None

    def init_fits(self, pars, names=('dataset1', 'dataset2'), multiindex=True):
        if self.Ndatasets > 1:
            if multiindex:
                fit_ind = np.array([np.array([l1, l2]) for l1, l2 in itertools.combinations(self.labels, 2)])
                self.fit_ind = pd.MultiIndex.from_arrays([fit_ind[:, 0], fit_ind[:, 1]], names=names)
                self.fit_df = pd.DataFrame(index=self.fit_ind,
                                           columns=pars + [f'S_{p}' for p in pars] + [f'P_{p}' for p in pars])
            else:
                self.fit_df = pd.DataFrame(index=self.labels,
                                           columns=pars + [f'S_{p}' for p in pars] + [f'P_{p}' for p in pars])

    def comp_pvalues(self, values, p):
        if self.fit_ind is not None:
            for ind, (v1, v2) in zip(self.fit_ind, itertools.combinations(values, 2)):
                self.comp_pvalue(ind, v1, v2, p)

    def comp_pvalue(self, ind, v1, v2, p):
        st, pv = ttest_ind(v1, v2, equal_var=False)
        if not pv <= 0.01:
            self.fit_df[p].loc[ind] = 0
        else:
            self.fit_df[p].loc[ind] = 1 if np.nanmean(v1) < np.nanmean(v2) else -1
        self.fit_df[f'S_{p}'].loc[ind] = st
        self.fit_df[f'P_{p}'].loc[ind] = np.round(pv, 11)

    def plot_half_circles(self, p, i):
        if self.fit_df is not None:
            ax = self.axs[i]
            ii = 0
            for z, (l1, l2) in enumerate(self.fit_df.index.values):
                col1, col2 = self.colors[self.labels.index(l1)], self.colors[self.labels.index(l2)]
                res = self.plot_half_circle(p, ax, col1, col2, v=self.fit_df[p].iloc[z], ind=(l1, l2), coef=z - ii)
                if not res:
                    ii += 1
                    continue

    def plot_half_circle(self, p, ax, col1, col2, v, ind, coef=0):
        res = True
        if v == 1:
            c1, c2 = col1, col2
        elif v == -1:
            c1, c2 = col2, col1
        else:
            res = False

        if res:
            rad = 0.04
            yy = 0.95 - coef * 0.08
            xx = 0.75
            dual_half_circle(center=(xx, yy), radius=rad, angle=90, ax=ax, colors=(c1, c2), transform=ax.transAxes)
            pv = self.fit_df[f'P_{p}'].loc[ind]
            if pv == 0:
                pvi = -9
            else:
                for pvi in np.arange(-1, -10, -1):
                    if np.log10(pv) > pvi:
                        pvi += 1
                        break
            ax.text(xx + 0.05, yy + rad / 1.5, f'p<10$^{{{pvi}}}$', ha='left', va='top', color='k',
                    fontsize=15, transform=ax.transAxes)
        return res

    @property
    def Nticks(self):
        Nticks_list = [len(d.step_data.index.unique('Step')) for d in self.datasets]
        return np.max(unique_list(Nticks_list))

    @property
    def fr(self):
        fr_list = [d.fr for d in self.datasets]
        return np.max(unique_list(fr_list))

    @property
    def dt(self):
        dt_list = unique_list([d.dt for d in self.datasets])
        # print(dt_list)
        return np.max(dt_list)

    @property
    def tlim(self):
        return (0, int(self.Nticks * self.dt))
        # return (0, int(self.Nticks / self.fr))

    def trange(self, unit='min'):
        if unit == 'min':
            T = 60
        elif unit == 'sec':
            T = 1
        t0, t1 = self.tlim
        x = np.linspace(t0 / T, t1 / T, self.Nticks)
        # print(t1, self.fr, self.dt, T, t1/T, self.Nticks)
        # raise
        return x

    def angrange(self, r, absolute=False, nbins=200):
        lim = (r0, r1) = (0, r) if absolute else (-r, r)
        x = np.linspace(r0, r1, nbins)
        return x, lim

    def plot_par(self, par, bins, i=0, labels=None, absolute=False, nbins=None, type='plt.hist',
                 pvalues=False, half_circles=False, **kwargs):
        if labels is None:
            labels = self.labels
        vs = []
        for d in self.datasets:
            v = d.get_par(par).dropna().values
            if absolute:
                v = np.abs(v)
            vs.append(v)
        if bins == 'broad' and nbins is not None:
            bins = np.linspace(np.min([np.min(v) for v in vs]), np.max([np.max(v) for v in vs]), nbins)
        for v, c, l in zip(vs, self.colors, labels):
            if type == 'sns.hist':
                sns.histplot(v, color=c, bins=bins, ax=self.axs[i], label=l, **kwargs)
            elif type == 'plt.hist':
                self.axs[i].hist(v, bins=bins, weights=np.ones_like(v) / float(len(v)), label=l, color=c, **kwargs)
        if pvalues:
            self.comp_pvalues(vs, par)
        if half_circles:
            self.plot_half_circles(par, i)
        return vs


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


def plot_mean_and_range(x, mean, lb, ub, axis, color_shading, color_mean=None, label=None,linewidth=2):
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
        labels = [d.config.id for d in datasets]
    Ndatasets = len(datasets)
    if Ndatasets != len(labels):
        raise ValueError(f'Number of labels {len(labels)} does not much number of datasets {Ndatasets}')

    def get_colors(datasets):
        try:
            cs = [d.config['color'] for d in datasets]
            u_cs = unique_list(cs)
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
    if save_to is None:
        save_to = datasets[0].config['parent_plot_dir']
    if subfolder is not None:
        save_to = f'{save_to}/{subfolder}'
    os.makedirs(save_to, exist_ok=True)
    return Ndatasets, cols, save_to, labels


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


def annotate_plot(data, x, y, hue, **kwargs):
    from statannotations.Annotator import Annotator
    h1, h2 = np.unique(data[hue].values)
    subIDs0 = np.unique(data[x].values)
    pairs = [((subID, h1), (subID, h2)) for subID in subIDs0]
    pvs = []
    for subID in subIDs0:
        dd = data[data[x] == subID]
        dd0 = dd[dd[hue] == h1][y].values
        dd1 = dd[dd[hue] == h2][y].values
        pvs.append(mannwhitneyu(dd0, dd1, alternative="two-sided").pvalue)
    f_pvs = [pvalue_star(pv) for pv in pvs]
    # f_pvs = [f'p={pv:.2e}' for pv in pvs]

    # Add annotations
    annotator = Annotator(pairs=pairs, data=data, x=x, y=y, hue=hue, **kwargs)
    annotator.verbose = False
    annotator.annotate_custom_annotations(f_pvs)


def concat_datasets(ds, key='end', unit='sec'):
    dfs = []
    for d in ds:
        df = d.read(key)
        df['GroupID'] = d.id
        dfs.append(df)
    df0 = pd.concat(dfs)
    if key == 'step':
        df0.reset_index(level='Step', drop=False, inplace=True)
        dts = np.unique([d.config['dt'] for d in ds])
        if len(dts) == 1:
            dt = dts[0]
            dic = {'sec': 1, 'min': 60, 'hour': 60 * 60, 'day': 24 * 60 * 60}
            df0['Step'] *= dt / dic[unit]
    return df0


def conf_ax_3d(vars, target, ax=None, fig=None, lims=None, title=None, maxN=5, labelpad=30, tickpad=10):
    if fig is None and ax is None:
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


def plot_single_bout(x0, discr, bout, i, color, label, axs, fit_dic=None, plot_fits='best',
                     marker='.', legend_outside=False, **kwargs):
    distro_ls = ['powerlaw', 'exponential', 'lognormal', 'lognorm-pow', 'levy', 'normal', 'uniform']
    distro_cs = ['c', 'g', 'm', 'k', 'orange', 'brown', 'purple']
    num_distros = len(distro_ls)
    lws = [2] * num_distros

    if fit_dic is None:
        xmin, xmax = np.min(x0), np.max(x0)
        fit_dic = fit_bout_distros(x0, xmin, xmax, discr, dataset_id='test', bout=bout, **kwargs)
    idx_Kmax = fit_dic['idx_Kmax']
    cdfs = fit_dic['cdfs']
    pdfs = fit_dic['pdfs']
    u2, du2, c2, c2cum = fit_dic['values']
    lws[idx_Kmax] = 4
    ylabel = 'cumulative probability'
    xlabel = 'time (sec)' if not discr else '# strides'
    xrange = u2
    y = c2cum
    ddfs = cdfs
    for ii in ddfs:
        if ii is not None:
            ii /= ii[0]
    axs[i].loglog(xrange, y, marker, color=color, alpha=0.7, label=label)
    axs[i].set_title(bout)
    axs[i].set_xlabel(xlabel)
    axs[i].set_ylim([10 ** -3.5, 10 ** 0.2])
    distro_ls0, distro_cs0 = [], []
    for z, (l, col, lw, ddf) in enumerate(zip(distro_ls, distro_cs, lws, ddfs)):
        if ddf is None:
            continue
        if plot_fits == 'best' and z == idx_Kmax:
            cc = color
        elif plot_fits == 'all' or l in plot_fits:
            distro_ls0.append(l)
            distro_cs0.append(col)
            cc = col
        else:
            continue
        axs[i].loglog(xrange, ddf, color=cc, lw=lw, label=l)
    if len(distro_ls0) > 1:
        if legend_outside:
            dataset_legend(distro_ls0, distro_cs0, ax=axs[1], loc='center left', fontsize=25, anchor=(1.0, 0.5))
        else:
            for ax in axs:
                dataset_legend(distro_ls0, distro_cs0, ax=ax, loc='lower left', fontsize=15)
    # dataset_legend(gIDs, colors, ax=axs[1], loc='center left', fontsize=25, anchor=(1.0, 0.5))
    # fig.subplots_adjust(left=0.1, right=0.95, wspace=0.08, hspace=0.3, bottom=0.05)
    for jj in [0]:
        axs[jj].set_ylabel(ylabel)

def modelConfTable(confID, save_as, columns = ['Parameter', 'Symbol', 'Value', 'Unit'],
                   rows = None,**kwargs
                   ) :
    from lib.aux.combining import render_mpl_table
    from lib.conf.base.dtypes import par
    from lib.conf.base.init_pars import init_pars
    m = loadConf(confID, "Model")
    if rows is None :
        rows=['physics','body']+[k for k,v in m.brain.modules.items() if v]

    rowDicts =[]
    for k in rows :
        try :
            rowDicts.append(m[k])
        except :
            rowDicts.append(m.brain[f'{k}_params'])
    #rowColors0 = N_colors(len(rows))
    rowColors0 = ['lightskyblue', 'lightsteelblue',  'lightcoral', 'indianred','lightsalmon', '#a55af4','palegreen','plum',   'pink'][:len(rows)]
    # rowColors0 = ['lightskyblue', 'lightsteelblue',  'lightcoral', 'indianred','lightsalmon', 'mediumpurple','palegreen','plum',   'pink'][:len(rows)]
    Nrows = {rowLab: 0 for rowLab in rows}

    def register(vs, rowColor):
        data.append(vs)
        rowColors.append(rowColor)
        Nrows[vs[0]] += 1

    rowColors = [None]
    data = []
    for rowLab, rowDic, rowColor in zip(rows, rowDicts, rowColors0):
        d0 = init_pars().get(rowLab, None)
        if rowLab=='interference':
            if rowDic.mode == 'square':
                rowValid = ['crawler_phi_range', 'attenuation','suppression_mode']
            elif rowDic.mode == 'phasic':
                rowValid = ['max_attenuation_phase', 'attenuation', 'attenuation_max','suppression_mode']
            elif rowDic.mode == 'default':
                rowValid = ['attenuation','suppression_mode']
        elif rowLab == 'physics':
            rowValid = ['torque_coef', 'ang_damping', 'body_spring_k', 'bend_correction_coef']
        elif rowLab == 'body':
            rowValid = ['initial_length', 'Nsegs']
        elif rowLab == 'turner':
            if rowDic.mode == 'neural':
                rowValid = ['base_activation', 'activation_range']
            elif rowDic.mode == 'constant':
                rowValid = ['initial_amp']
            elif rowDic.mode == 'sinusoidal':
                rowValid = ['initial_amp', 'initial_freq']
        elif rowLab == 'crawler':
            if rowDic.waveform == 'realistic':
                rowValid = ['initial_freq','max_scaled_vel', 'max_vel_phase',  'stride_dst_mean', 'stride_dst_std']
            elif rowDic.waveform == 'constant':
                rowValid = ['initial_amp']
        elif rowLab == 'intermitter':
            rowValid = ['stridechain_dist', 'pause_dist']
            rowValid = [n for n in rowValid if rowDic[n] is not None and rowDic[n].name is not None]
        elif rowLab == 'olfactor':
            rowValid = ['decay_coef']
        if len(rowValid)==0 :
            Nrows.pop(rowLab, None)
            continue
        for n, vv in d0.items():
            if n not in rowValid:
                continue
            v = rowDic[n]
            if n in ['stridechain_dist', 'pause_dist'] :
                # print(rowLab, n,v)
                if v.name == 'exponential':
                    dist_v = f'Exp(b={v.beta})'
                elif v.name == 'powerlaw':
                    dist_v = f'Powerlaw(a={v.alpha})'
                elif v.name == 'levy':
                    dist_v = f'Levy(m={v.mu}, s={v.sigma})'
                elif v.name == 'uniform':
                    dist_v = f'Uniform()'
                elif v.name == 'lognormal':
                    dist_v = f'Lognormal(m={v.mu}, s={v.sigma})'
                if n == 'stridechain_dist':
                    vs1 = [rowLab, 'run length distribution', '$N_{R}$', dist_v, '-']
                    vs2 = [rowLab, 'run length range', '$[N_{R}^{min},N_{R}^{max}]$', v.range, '# $strides$']
                elif n == 'pause_dist':
                    vs1 = [rowLab, 'pause duration distribution', '$t_{P}$', dist_v, '-']
                    vs2 = [rowLab, 'pause duration range', '$[t_{P}^{min},t_{P}^{max}]$', v.range, '$sec$']
                register(vs1, rowColor)
                register(vs2, rowColor)
            else:
                p = par(n, **vv)

                if n == 'initial_length':
                    v *=1000
                elif n == 'suppression_mode':
                    if v=='both' :
                        v='input & output'
                    elif v=='amplitude' :
                        v='output'
                    elif v=='oscillation' :
                        v='input'

                else:
                    try:
                        v = np.round(v, 2)
                    except:
                        pass
                vs = [rowLab, p[n]['label'], p[n]['symbol'], v, p[n]['unit']]
                register(vs, rowColor)

    cumNrows = dict(zip(list(Nrows.keys()), np.cumsum(list(Nrows.values())).astype(int)))
    df = pd.DataFrame(data, columns=['field'] + columns)
    df.set_index(['field'], inplace=True)

    ax, fig, mpl = render_mpl_table(df, colWidths=[0.35, 0.1, 0.25, 0.15], cellLoc='center', rowLoc='center',
                                    row_colors=rowColors, return_table=True,**kwargs)

    for k, cell in mpl._cells.items():
        if k[1] == -1:
            cell._text._text = ''
            cell._linewidth = 0

    for rowLab, idx in cumNrows.items():
        cell = mpl._cells[(idx-Nrows[rowLab]+1, -1)]
        cell._text._text = rowLab.upper()
    fig.savefig(save_as, dpi=300)
    # return fig,ax,mpl
