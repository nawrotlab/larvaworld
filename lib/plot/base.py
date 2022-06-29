import itertools
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, ticker, patches
from matplotlib.gridspec import GridSpec




from lib.aux import dictsNlists as dNl

from lib.plot.aux import dual_half_circle, plot_config, process_plot


class BasePlot:
    def __init__(self, name, save_to='.', save_as=None, return_fig=False, show=False, suf='pdf', text_xy0=(0.05, 0.98),
                 **kwargs):
        self.filename = f'{name}.{suf}' if save_as is None else f'{save_as}.{suf}'
        self.return_fig = return_fig
        self.show = show
        self.fit_df = None
        self.save_to = save_to

        self.cur_idx = 0
        self.text_x0, self.text_y0 = text_xy0
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        self.letter_dict = {}
        self.x0s, self.y0s = [], []

    def build(self, Nrows=1, Ncols=1, figsize=None, fig=None, axs=None, dim3=False, azim=115, elev=15, **kwargs):
        if fig is None and axs is None:
            if figsize is None:
                figsize = (12 * Ncols, 10 * Nrows)
            if not dim3:
                self.fig, axs = plt.subplots(Nrows, Ncols, figsize=figsize, **kwargs)
                self.axs = axs.ravel() if Nrows * Ncols > 1 else [axs]
            else:
                from mpl_toolkits.mplot3d import Axes3D
                self.fig = plt.figure(figsize=(15, 10))
                ax = Axes3D(self.fig, azim=azim, elev=elev)
                self.axs = [ax]
        else:
            self.fig = fig
            self.axs = axs if type(axs) == list else [axs]

    def conf_ax(self, idx=0, xlab=None, ylab=None, zlab=None, xlim=None, ylim=None, zlim=None, xticks=None,
                xticklabels=None, yticks=None, xticklabelrotation=None, yticklabelrotation=None,
                yticklabels=None, zticks=None, zticklabels=None, xtickpos=None, xtickpad=None, ytickpad=None,
                ztickpad=None, xlabelfontsize=None, xticklabelsize=None, yticklabelsize=None, zticklabelsize=None,
                xlabelpad=None, ylabelpad=None, zlabelpad=None, equal_aspect=None,
                xMaxN=None, yMaxN=None, zMaxN=None, xMath=None, yMath=None, tickMath=None, ytickMath=None, xMaxFix=False,leg_loc=None,
                leg_handles=None, xvis=None, yvis=None, zvis=None,
                title=None, title_y=None, titlefontsize=None):
        ax = self.axs[idx]
        if equal_aspect is not None:
            ax.set_aspect('equal', adjustable='box')
        if xvis is not None:
            ax.xaxis.set_visible(xvis)
        if yvis is not None:
            ax.yaxis.set_visible(yvis)
        if zvis is not None:
            ax.zaxis.set_visible(zvis)
        if ylab is not None:
            ax.set_ylabel(ylab, labelpad=ylabelpad)
        if xlab is not None:
            if xlabelfontsize is not None:
                ax.set_xlabel(xlab, labelpad=xlabelpad, fontsize=xlabelfontsize)
            else:
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
        if xticklabelrotation is not None:
            ax.tick_params(axis='x', which='major', rotation=xticklabelrotation)
        if xticklabelsize is not None:
            ax.tick_params(axis='x', which='major', labelsize=xticklabelsize)
        if yticklabelsize is not None:
            ax.tick_params(axis='y', which='major', labelsize=yticklabelsize)
        if zticklabelsize is not None:
            ax.tick_params(axis='z', which='major', labelsize=zticklabelsize)
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
        if xMaxFix:
            ticks_loc = ax.get_xticks().tolist()
            ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
            # ax.xaxis.set_major_locator(ticker.MaxNLocator(xMaxN))
        if xMaxN is not None:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(xMaxN))
        if yMaxN is not None:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(yMaxN))
        if zMaxN is not None:
            ax.zaxis.set_major_locator(ticker.MaxNLocator(zMaxN))
        if xMath is not None:
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True, useMathText=True))
        if yMath is not None:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=True, useMathText=True))
        if xtickpos is not None:
            ax.xaxis.set_ticks_position(xtickpos)
        if title is not None:
            ax.set_title(title, fontsize=titlefontsize, y=title_y)
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

    def add_letter(self, ax, letter=True, x0=False, y0=False):
        if letter:
            self.letter_dict[ax] = self.letters[self.cur_idx]
            self.cur_idx += 1
            if x0:
                self.x0s.append(ax)
            if y0:
                self.y0s.append(ax)

    def annotate(self, dx=-0.05, dy=0.005, full_dict=False):
        if full_dict:

            for i, ax in enumerate(self.axs):
                self.letter_dict[ax] = self.letters[i]
        for i, (ax, text) in enumerate(self.letter_dict.items()):
            X = self.text_x0 if ax in self.x0s else ax.get_position().x0 + dx
            Y = self.text_y0 if ax in self.y0s else ax.get_position().y1 + dy
            self.fig.text(X, Y, text, size=30, weight='bold')


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
            from lib.conf.stored.conf import loadRef, kConfDict
            targetIDs = dNl.unique_list([d.config['sample'] for d in datasets])

            targets = [loadRef(id) for id in targetIDs if id in kConfDict('Ref')]
            datasets += targets
            if labels is not None:
                labels += targetIDs
        self.Ndatasets, self.colors, save_to, self.labels = plot_config(datasets, labels, save_to,
                                                                        subfolder=subfolder)
        super().__init__(name, save_to=save_to, **kwargs)
        self.datasets = datasets
        ff = f'{name}_fits.csv' if save_fits_as is None else save_fits_as
        self.fit_filename = os.path.join(self.save_to, ff) if ff is not None and self.save_to is not None else None
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
        from scipy.stats import ttest_ind
        st, pv = ttest_ind(v1, v2, equal_var=False)
        if not pv <= 0.01:
            t = 0
        elif np.nanmean(v1) < np.nanmean(v2):
            t = 1
        else :
            t=-1
        self.fit_df.loc[ind, [p,f'S_{p}',f'P_{p}']]=[t,st,np.round(pv, 11)]
        # self.fit_df[f'S_{p}'].loc[ind] = st
        # self.fit_df[f'P_{p}'].loc[ind] = np.round(pv, 11)

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

    def data_leg(self,idx=None, labels=None, colors=None, anchor=None, handlelength=0.5, handleheight=0.5, **kwargs):
        if labels is None :
            labels=self.labels
        if colors is None :
            colors=self.colors
        kws = {
            'handles': [patches.Patch(facecolor=c, label=l, edgecolor='black') for c, l in zip(colors, labels)],
            'handlelength': handlelength,
            'handleheight': handleheight,
            'labels': labels,
            'bbox_to_anchor': anchor,
            **kwargs
        }
        if idx is None:
            leg = plt.legend(**kws)
        else:
            ax = self.axs[idx]
            leg = ax.legend(**kws)
            ax.add_artist(leg)
        return leg

    @property
    def Nticks(self):
        Nticks_list = [d.config.Nticks for d in self.datasets]
        return np.max(dNl.unique_list(Nticks_list))

    @property
    def fr(self):
        fr_list = [d.fr for d in self.datasets]
        return np.max(dNl.unique_list(fr_list))

    @property
    def dt(self):
        dt_list = dNl.unique_list([d.dt for d in self.datasets])
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
        return x

    def angrange(self, r, absolute=False, nbins=200):
        lim = (r0, r1) = (0, r) if absolute else (-r, r)
        x = np.linspace(r0, r1, nbins)
        return x, lim

    def plot_par(self, short=None, par=None, vs=None, bins='broad', i=0, labels=None, absolute=False, nbins=None,
                 type='plt.hist', sns_kws={},
                 pvalues=False, half_circles=False, key='step', **kwargs):
        if labels is None:
            labels = self.labels
        if vs is None:
            vs = []
            for d in self.datasets:
                if key == 'step':
                    try:
                        v = d.step_data[par]
                    except:
                        v = d.get_par(par, key=key)
                elif key == 'end':
                    try:
                        v = d.endpoint_data[par]
                    except:
                        v = d.get_par(par, key=key)
                if v is not None:
                    v = v.dropna().values
                else:
                    continue
                if absolute:
                    v = np.abs(v)
                vs.append(v)
        if bins == 'broad' and nbins is not None:
            bins = np.linspace(np.min([np.min(v) for v in vs]), np.max([np.max(v) for v in vs]), nbins)
        for v, c, l in zip(vs, self.colors, labels):
            if type == 'sns.hist':
                sns.histplot(v, color=c, bins=bins, ax=self.axs[i], label=l, **sns_kws, **kwargs)
            elif type == 'plt.hist':
                self.axs[i].hist(v, bins=bins, weights=np.ones_like(v) / float(len(v)), label=l, color=c, **kwargs)
        if pvalues:
            self.comp_pvalues(vs, par)
        if half_circles:
            self.plot_half_circles(par, i)
        return vs


class AutoPlot(Plot):
    def __init__(self, Nrows=1, Ncols=1, figsize=None, fig=None, axs=None, sharex=False, sharey=False, **kwargs):
        super().__init__(**kwargs)
        self.build(Nrows=Nrows, Ncols=Ncols, figsize=figsize, fig=fig, axs=axs, sharex=sharex, sharey=sharey)


def load_ks(ks, ds,ls,cols, d0):
    dic = {}
    for k in ks:
        dic[k] = {}
        for d,l,col in zip(ds,ls,cols):
            # print(d0.get(k=k, d=d, compute=True))

            vs = d0.get(k=k, d=d, compute=True)
            dic[k][l] = dNl.NestDict({'df':vs, 'col':col})
    return dNl.NestDict(dic)


class AutoLoadPlot(AutoPlot) :
    def __init__(self, ks, **kwargs):
        from lib.registry.pars import preg
        super().__init__(**kwargs)
        d0 = preg
        self.kdict= load_ks(ks, self.datasets,self.labels,self.colors, d0)
        self.pdict=dNl.NestDict({k:d0.dict[k] for k in ks})
        self.kpdict=dNl.NestDict({k:[self.kdict[k],self.pdict[k]] for k in ks})
        self.ks=ks
        self.pars=[self.pdict[k].d for k in ks]


class GridPlot(BasePlot):
    def __init__(self, name, width, height, scale=(1, 1), **kwargs):
        super().__init__(name, **kwargs)
        ws, hs = scale
        self.width, self.height = width, height
        figsize = (int(width * ws), int(height * hs))
        self.fig = plt.figure(constrained_layout=False, figsize=figsize)
        self.grid = GridSpec(height, width, figure=self.fig)
        self.cur_w, self.cur_h = 0, 0


    def add(self, N=1, w=None, h=None, w0=None, h0=None, dw=0, dh=0, share_w=False, share_h=False, letter=True,
            x0=False, y0=False):

        if w0 is None:
            w0 = self.cur_w
        if h0 is None:
            h0 = self.cur_h

        if w is None:
            w = self.width - w0
        if h is None:
            h = self.height - h0

        if N == 1:
            axs = self.fig.add_subplot(self.grid[h0:h0 + h, w0:w0 + w])
            ax_letter = axs
            # if letter:
            #     self.letter_dict[axs]=self.letters[self.cur_idx]
            #     self.cur_idx += 1
            # return axs
        else:
            if share_h:
                ww = int((w - (N - 1) * dw) / N)
                axs = [self.fig.add_subplot(self.grid[h0:h0 + h, w0 + dw * i + ww * i:w0 + dw * i + ww * (i + 1)]) for i
                       in range(N)]
            elif share_w:
                hh = int((h - (N - 1) * dh) / N)
                axs = [self.fig.add_subplot(self.grid[h0 + dh * i + hh * i:h0 + dh * i + hh * (i + 1), w0:w0 + w]) for i
                       in range(N)]
            ax_letter = axs[0]
        self.add_letter(ax_letter, letter, x0=x0, y0=y0)
        return axs

    def plot(self, func, kws, axs=None, **kwargs):
        if axs is None:
            axs = self.add(**kwargs)
        from lib.plot.dict import graph_dict
        func=graph_dict.get(func)
        _ = func(fig=self.fig, axs=axs, **kws)
