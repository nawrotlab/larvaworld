import itertools
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, ticker, patches
from matplotlib.gridspec import GridSpec

from larvaworld.lib import reg, aux, plot

plt_conf = {'axes.labelsize': 20,
            'axes.titlesize': 25,
            'figure.titlesize': 25,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 20,
            'legend.title_fontsize': 20}
plt.rcParams.update(plt_conf)


class BasePlot:
    def __init__(self, name, save_to=None, save_as=None, return_fig=False, show=False, suf='pdf', text_xy0=(0.05, 0.98),verbose=1,
                 subplot_kw={}, build_kws={},
                 **kwargs):
        self.filename = f'{name}.{suf}' if save_as is None else f'{save_as}.{suf}'
        self.return_fig = return_fig
        self.verbose = verbose
        self.show = show
        self.fit_df = None
        self.save_to = save_to
        self.cur_idx = 0
        self.text_x0, self.text_y0 = text_xy0
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        self.letter_dict = {}
        self.x0s, self.y0s = [], []
        self.fig_kws={}
        self.build_kws=self.set_build_kws(subplot_kw=subplot_kw, build_kws=build_kws)

    def set_build_kws(self,subplot_kw, build_kws):
        for k,v in build_kws.items():
            if v=='Ndatasets':
                build_kws[k]=self.Ndatasets
        build_kws['subplot_kw']=subplot_kw
        return build_kws


    def build(self, fig=None, axs=None, dim3=False, azim=115, elev=15):
        if fig is not None and axs is not None:
            self.fig = fig
            self.axs = axs if type(axs) == list else [axs]

        else:
            if dim3:
                from mpl_toolkits.mplot3d import Axes3D
                self.fig = plt.figure(figsize=(15, 10))
                ax = Axes3D(self.fig, azim=azim, elev=elev)
                self.axs = [ax]
            else:
                self.fig_kws = plot.NcolNrows(**self.build_kws)
                self.fig, axs = plt.subplots(**self.fig_kws)
                self.axs = axs.ravel() if self.Ncols*self.Nrows > 1 else [axs]

    @ property
    def Naxs(self):
        return len(self.axs)

    @property
    def Ncols(self):
        # if self.Naxs==1 :
        #     return 1
        if 'ncols' in self.fig_kws.keys() :
            return self.fig_kws['ncols']
        else :
            return 1

    @property
    def Nrows(self):
        # if self.Naxs == 1:
        #     return 1
        if 'nrows' in self.fig_kws.keys():
            return self.fig_kws['nrows']
        else:
            return 1


    def conf_ax(self, idx=0, xlab=None, ylab=None, zlab=None, xlim=None, ylim=None, zlim=None, xticks=None,
                xticklabels=None, yticks=None, xticklabelrotation=None, yticklabelrotation=None,
                yticklabels=None, zticks=None, zticklabels=None, xtickpos=None, xtickpad=None, ytickpad=None,
                ztickpad=None, xlabelfontsize=None, xticklabelsize=None, yticklabelsize=None, zticklabelsize=None,
                xlabelpad=None, ylabelpad=None, zlabelpad=None, equal_aspect=None,
                xMaxN=None, yMaxN=None, zMaxN=None, xMath=None, yMath=None, tickMath=None, ytickMath=None, xMaxFix=False,leg_loc=None,
                leg_handles=None, leg_labels=None,legfontsize=None,xvis=None, yvis=None, zvis=None,
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
            kws={
                'loc' : leg_loc,
                'fontsize' : legfontsize,
            }
            if leg_handles is not None:
                kws['handles']=leg_handles
            if leg_labels is not None:
                kws['labels']=leg_labels
            ax.legend(**kws)


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
        return plot.process_plot(self.fig, self.save_to, self.filename, self.return_fig, self.show, verbose=self.verbose)

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

    def conf_fig(self, idx=0, xlab=None, ylab=None, zlab=None, xlim=None, ylim=None, zlim=None, xticks=None,
                xticklabels=None, yticks=None, xticklabelrotation=None, yticklabelrotation=None,
                yticklabels=None, zticks=None, zticklabels=None, xtickpos=None, xtickpad=None, ytickpad=None,
                ztickpad=None, xlabelfontsize=None, xticklabelsize=None, yticklabelsize=None, zticklabelsize=None,
                xlabelpad=None, ylabelpad=None, zlabelpad=None, equal_aspect=None,
                xMaxN=None, yMaxN=None, zMaxN=None, xMath=None, yMath=None, tickMath=None, ytickMath=None,
                xMaxFix=False, leg_loc=None,
                leg_handles=None, xvis=None, yvis=None, zvis=None,adjust_kws=None,align=None,
                title=None, title_kws={}):
        if title is not None:
            pairs={
                # 't':'t',
                'w':'fontweight',
                's':'fontsize',
                # 't':title_kws.t,
            }
            kws=aux.AttrDict(title_kws).replace_keys(pairs)
            # kws=aux.replace_in_dict(title_kws, pairs, replace_key=True)
            self.fig.suptitle(t=title,**kws)
        if adjust_kws is not None :
            self.adjust(**adjust_kws)
        if align is not None :
            if type(align)==list :
                ax_list=align
            else :
                ax_list=self.axs[:]
            self.fig.align_ylabels(ax_list)


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


class AutoBasePlot(BasePlot):
    def __init__(self, fig=None, axs=None, **kwargs):
        super().__init__(**kwargs)

        self.build(fig=fig, axs=axs)

class Plot(BasePlot):
    def __init__(self, name, datasets, labels=None, subfolder=None, save_fits_as=None, save_to=None, add_samples=False,
                 **kwargs):

        if add_samples:
            targetIDs = aux.unique_list([d.config['sample'] for d in datasets])

            targets = [reg.loadRef(id) for id in targetIDs if id in reg.storedConf('Ref')]
            datasets += targets
            if labels is not None:
                labels += targetIDs
        self.Ndatasets, self.colors, save_to, self.labels = plot.plot_config(datasets, labels, save_to,
                                                                        subfolder=subfolder)

        super().__init__(name, save_to=save_to, **kwargs)
        self.datasets = datasets
        # print([d.id for d in self.datasets], self.labels)
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
            plot.dual_half_circle(center=(xx, yy), radius=rad, angle=90, ax=ax, colors=(c1, c2), transform=ax.transAxes)
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
    def data_dict(self):
        # N_list = [d.config.N for d in self.datasets]
        return dict(zip(self.labels,self.datasets))

    @property
    def data_palette(self):
        # N_list = [d.config.N for d in self.datasets]
        return zip(self.labels, self.datasets, self.colors)

    @property
    def Nticks(self):
        Nticks_list = [d.config.Nticks for d in self.datasets]
        return np.max(aux.unique_list(Nticks_list))

    @property
    def N(self):
        N_list = [d.config.N for d in self.datasets]
        return np.max(aux.unique_list(N_list))

    @property
    def fr(self):
        fr_list = [d.fr for d in self.datasets]
        return np.max(aux.unique_list(fr_list))

    @property
    def dt(self):
        dt_list = aux.unique_list([d.dt for d in self.datasets])
        return np.max(dt_list)

    @property
    def duration(self):
        return int(self.Nticks * self.dt)

    @property
    def tlim(self):
        return (0, self.duration)

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
    def __init__(self, fig=None, axs=None, **kwargs):
        super().__init__(**kwargs)

        self.build(fig=fig, axs=axs)


def load_ks(ks, ds,ls,cols, d0):
    dic = {}
    for k in ks:
        dic[k] = {}
        for d,l,col in zip(ds,ls,cols):
            vs = d0.get(k=k, d=d, compute=True)
            dic[k][l] = aux.AttrDict({'df':vs, 'col':col})
    return aux.AttrDict(dic)


class AutoLoadPlot(AutoPlot) :
    def __init__(self, ks, **kwargs):
        super().__init__(**kwargs)
        self.kdict= load_ks(ks, self.datasets,self.labels,self.colors, reg.par)
        self.pdict=aux.AttrDict({k:reg.par.kdict[k] for k in ks})
        self.kpdict=aux.AttrDict({k:[self.kdict[k], self.pdict[k]] for k in ks})
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
            x0=False, y0=False, cols_first=False):

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
            if share_h and not share_w:
                ww = int((w - (N - 1) * dw) / N)
                axs = [self.fig.add_subplot(self.grid[h0:h0 + h, w0 + dw * i + ww * i:w0 + dw * i + ww * (i + 1)]) for i
                       in range(N)]
            elif share_w and not share_h:
                hh = int((h - (N - 1) * dh) / N)
                axs = [self.fig.add_subplot(self.grid[h0 + dh * i + hh * i:h0 + dh * i + hh * (i + 1), w0:w0 + w]) for i
                       in range(N)]
            elif share_w and share_h:
                Nrows,Ncols=N,N
                hh = int((h - (Nrows - 1) * dh) / Nrows)
                ww = int((w - (Ncols - 1) * dw) / Ncols)
                axs=[]
                if not cols_first :
                    for i in range(Nrows):
                        for j in range(Ncols) :
                            ax=self.fig.add_subplot(self.grid[
                                                    h0 + dh * i + hh * i:h0 + dh * i + hh * (i + 1),
                                                    w0 + dw * j + ww * j:w0 + dw * j + ww * (j + 1)])
                            axs.append(ax)
                else :
                    for j in range(Ncols) :
                        for i in range(Nrows):
                            ax=self.fig.add_subplot(self.grid[
                                                    h0 + dh * i + hh * i:h0 + dh * i + hh * (i + 1),
                                                    w0 + dw * j + ww * j:w0 + dw * j + ww * (j + 1)])
                            axs.append(ax)
            ax_letter = axs[0]
        self.add_letter(ax_letter, letter, x0=x0, y0=y0)
        return axs

    def plot(self, func, kws, axs=None, **kwargs):
        if axs is None:
            axs = self.add(**kwargs)
        func=reg.graphs.get(func)
        _ = func(fig=self.fig, axs=axs, **kws)
