
import itertools
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker, patches
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind

from larvaworld.lib import reg, aux, plot
from larvaworld.lib.process.dataset import LarvaDatasetCollection

plt_conf = {'axes.labelsize': 20,
            'axes.titlesize': 25,
            'figure.titlesize': 25,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 20,
            'legend.title_fontsize': 20}
plt.rcParams.update(plt_conf)


class BasePlot:
    def __init__(self, name='larvaworld_plot', save_as=None,pref=None,suf='pdf',
                 save_to=None, subfolder=None,
                 return_fig=False, show=False,
                 subplot_kw={}, build_kws={}, **kwargs):
        if save_as is None :
            if pref is not None:
                name = f'{pref}_{name}'
            save_as=name
        self.filename = f'{save_as}.{suf}'
        self.fit_filename = f'{save_as}_fits.csv'
        self.fit_ind = None
        self.fit_df = None

        if save_to is not None:
            if subfolder is not None:
                save_to = f'{save_to}/{subfolder}'
            os.makedirs(save_to, exist_ok=True)
        self.save_to = save_to

        self.return_fig = return_fig
        self.show = show

        self.fig_kws=self.set_fig_kws(subplot_kw=subplot_kw, build_kws=build_kws)


    def set_fig_kws(self,subplot_kw={}, build_kws={}):
        for k,v in build_kws.items():
            if v=='Ndatasets' :
                if hasattr(self, 'Ndatasets'):
                    build_kws[k]=self.Ndatasets
                else :
                    build_kws[k] = None
            if v=='Nks' :
                if hasattr(self, 'Nks'):
                    build_kws[k]=self.Nks
                else :
                    build_kws[k] = None
        build_kws['subplot_kw']=subplot_kw
        return plot.configure_subplot_grid(**build_kws)


    def build(self, fig=None, axs=None, dim3=False, azim=115, elev=15):
        '''
        Method that defines the figure and axes on which to draw.
        These can be provided externally as arguments to create a composite figure. Otherwise they are created independently.
        Args:
            fig: The figure of the plot (optional)
            axs: The axes of the figure (optional)
            dim3: Whether the figure will be 3-dimensional. Default : False
            azim: The azimuth of a 3D figure. Default : 115
            elev: The elevation of a 3D figure. Default : 15

        '''
        if fig is not None and axs is not None:
            self.fig = fig
            self.axs = axs if type(axs) in [list, np.ndarray] else [axs]

        else:
            if dim3:
                from mpl_toolkits.mplot3d import Axes3D
                self.fig = plt.figure(figsize=(15, 10))
                ax = Axes3D(self.fig, azim=azim, elev=elev)
                self.axs = [ax]
            else:
                self.fig, axs = plt.subplots(**self.fig_kws)
                self.axs = axs.ravel() if isinstance(axs, np.ndarray) else [axs]

    @ property
    def Naxs(self):
        return len(self.axs)

    @property
    def Ncols(self):
        return self.axs[0].get_gridspec().ncols


    @property
    def Nrows(self):
        return self.axs[0].get_gridspec().nrows


    def conf_ax(self, idx=0,ax=None, xlab=None, ylab=None, zlab=None, xlim=None, ylim=None, zlim=None, xticks=None,
                xticklabels=None, yticks=None, xticklabelrotation=None, yticklabelrotation=None,
                yticklabels=None, zticks=None, zticklabels=None, xtickpos=None, xtickpad=None, ytickpad=None,
                ztickpad=None, xlabelfontsize=None,ylabelfontsize=None, xticklabelsize=None, yticklabelsize=None, zticklabelsize=None,
                major_ticklabelsize=None,minor_ticklabelsize=None,
                xlabelpad=None, ylabelpad=None, zlabelpad=None, equal_aspect=None,
                xMaxN=None, yMaxN=None, zMaxN=None,yStrN=None, xMath=None, yMath=None, tickMath=None, ytickMath=None, xMaxFix=False,leg_loc=None,
                leg_handles=None, leg_labels=None,legfontsize=None,xvis=None, yvis=None, zvis=None,
                title=None, title_y=None, titlefontsize=None):
        '''
        Helper method that configures an axis of the figure

        '''
        if ax is None :
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
            if ylabelfontsize is not None:
                ax.set_ylabel(ylab, labelpad=ylabelpad, fontsize=ylabelfontsize)
            else :
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
        if major_ticklabelsize is not None:
            ax.tick_params(axis='both', which='major', labelsize=major_ticklabelsize)
        if minor_ticklabelsize is not None:
            ax.tick_params(axis='both', which='minor', labelsize=minor_ticklabelsize)

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
        if xMaxN is not None:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(xMaxN))
        if yMaxN is not None:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(yMaxN))
        if zMaxN is not None:
            ax.zaxis.set_major_locator(ticker.MaxNLocator(zMaxN))
        if yStrN is not None:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(f'%.{yStrN}f'))

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


    def conf_ax_3d(self, vars, target, lims=None, title=None, maxN=3, labelpad=15, tickpad=5, idx=0):
        if lims is None:
            xlim, ylim, zlim = None, None, None
        else:
            xlim, ylim, zlim = lims
        self.conf_ax(idx=idx, xlab=vars[0], ylab=vars[1], zlab=target, xlim=xlim, ylim=ylim, zlim=zlim,
                     xtickpad=tickpad, ytickpad=tickpad, ztickpad=tickpad,
                     xlabelpad=labelpad, ylabelpad=labelpad, zlabelpad=labelpad,
                     xMaxN=maxN, yMaxN=maxN, zMaxN=maxN, title=title)

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
        if self.fit_df is not None and self.save_to is not None:
            ff = os.path.join(self.save_to, self.fit_filename)
            self.fit_df.to_csv(ff, index=True, header=True)
        return plot.process_plot(self.fig, self.save_to, self.filename, self.return_fig, self.show)


    def conf_fig(self, adjust_kws=None,align=None,title=None, title_kws={}):
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




class AutoBasePlot(BasePlot):
    def __init__(self, fig=None, axs=None, dim3=False, azim=115, elev=15, **kwargs):
        super().__init__(**kwargs)

        self.build(fig=fig, axs=axs, dim3=dim3, azim=azim, elev=elev)


class AutoPlot(AutoBasePlot,LarvaDatasetCollection):
    def __init__(self, ks=[],key='step',klabels={},datasets=[], labels=None, add_samples=False,
                 ranges=None,absolute=False, rad2deg=False,space_unit='mm',**kwargs):
        '''
        Extension of the basic plotting class that receives datasets of type larvaworld.LarvaDataset
        Args:
            datasets: The datasets to access for plotting
            labels: The labels by which the datasets will be indicated in the plots. If not specified the IDs of the datasets will be used
            add_samples: Whether to also plot the reference datasets of any simulated datasets
            **kwargs:
        '''
        LarvaDatasetCollection.__init__(self, datasets=datasets, labels=labels, add_samples=add_samples)
        self.key = key
        self.ks = []
        self.kkdict = aux.AttrDict()
        # self.kdict = aux.AttrDict()
        self.pdict = aux.AttrDict()
        self.vdict = aux.AttrDict()

        for k in ks :
            try :
                p = reg.par.kdict[k]
                if p.u == reg.units.m and space_unit == 'mm':
                    p.u = reg.units.millimeter
                    coeff = 1000
                else:
                    coeff=1
                if k in klabels.keys():
                    p.disp=klabels[k]
                dfs = aux.AttrDict()
                # dics = aux.AttrDict()
                vs=[]
                for l, d, col in self.data_palette :
                    df = d.get_par(k=k, key=key)*coeff
                    assert df is not None
                    v = df.dropna().values
                    if absolute:
                        v = np.abs(v)
                    if rad2deg:
                        v = np.rad2deg(v)
                    dfs[l]=df
                    # dics[l]=aux.AttrDict({'df': v, 'col': col})
                    vs.append(v)
                self.kkdict[k]=dfs
                # self.kdict[k] = dics
                self.vdict[k] = vs
                self.pdict[k] = p
                self.ks.append(k)
            except :
                reg.vprint(f'Failed to retrieve key {k}', 1)
                pass
        self.dkdict=aux.AttrDict({l:{k:self.kkdict[k][l] for k in self.ks} for l in self.labels})
        self.pars = reg.getPar(self.ks)
        self.Nks=len(self.ks)
        self.ranges = ranges
        self.absolute = absolute
        self.rad2deg = rad2deg
        AutoBasePlot.__init__(self,**kwargs)

    def comp_all_pvalues(self):
        if self.Ndatasets < 2:
            return
        columns = pd.MultiIndex.from_product([self.ks, ['significance', 'stat', 'pvalue']])
        fit_ind = pd.MultiIndex.from_tuples(list(itertools.combinations(self.labels, 2)))
        self.fit_df = pd.DataFrame(index=fit_ind, columns=columns)

        for k in self.ks:
            for ind, (vv1, vv2) in zip(fit_ind, itertools.combinations(self.vdict[k], 2)):
                v1, v2 = list(vv1), list(vv2)
                st, pv = ttest_ind(v1, v2, equal_var=False)
                if not pv <= 0.01:
                    t = 0
                elif np.nanmean(v1) < np.nanmean(v2):
                    t = 1
                else:
                    t = -1
                # print([t, st, np.round(pv, 11)])
                self.fit_df.loc[ind, k] = [t, st, np.round(pv, 11)]

        # print(self.fit_df)

    def plot_all_half_circles(self):
        if self.fit_df is None:
            return
        for i,k in enumerate(self.ks):
            ax=self.axs[i]
            df=self.fit_df[k]
            ii = 0
            for z, (l1, l2) in enumerate(df.index.values):
                col1, col2 = self.colors[self.labels.index(l1)], self.colors[self.labels.index(l2)]
                pv = df['pvalue'].loc[(l1, l2)]
                v = df['significance'].loc[(l1, l2)]
                res = self.plot_half_circle(ax, col1, col2, v=v, pv=pv, coef=z - ii)
                if not res:
                    ii += 1
                    continue


    def plot_half_circle(self, ax, col1, col2, v, pv, coef=0):
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
            plot.dual_half_circle(center=(xx, yy), radius=rad, ax=ax, colors=(c1, c2), transform=ax.transAxes)
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

    def data_leg(self,idx=None, labels=None, colors=None, anchor=None,
                 handlelength=0.5, handleheight=0.5,Nagents_in_label=True, **kwargs):
        if labels is None :
            if not Nagents_in_label:
                labels=self.labels
            else:
                labels=self.labels_with_N
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


    def plot_quantiles(self, k=None, par=None, idx=0, ax=None,xlim=None, ylim=None, ylab=None,
                       unit='sec', leg_loc='upper left',coeff=1,
                       absolute=False, individuals=False,show_first=False,
                       Nagents_in_label=True, **kwargs):
        x=self.trange(unit)
        if ax is None :
            ax = self.axs[idx]

        try :
            if k is None :
                k = reg.getPar(d=par, to_return='k')
            p=reg.par.kdict[k]

            if ylab is None:
                ylab = p.l
            if ylim is None:
                ylim = p.lim
        except :
            pass
        if xlim is None:
            xlim = [x[0], x[-1]]

        if not Nagents_in_label :
            data=self.data_palette
        else:
            data = self.data_palette_with_N

        for l, d, c in data:
            df=d.get_par(k=k, par=par)*coeff
            if absolute:
                df = df.abs()
            if individuals:
                # plot each timeseries individually
                for id in df.index.get_level_values('AgentID'):
                    df_single = df.xs(id, level='AgentID')
                    ax.plot(x, df_single, color=c, linewidth=1)
            else :
                # plot the shaded range between first and third quantile
                df_u = df.groupby(level='Step').quantile(q=0.75)
                df_b = df.groupby(level='Step').quantile(q=0.25)
                # print(df_u.shape,df_b.shape,x.shape,x[:df_u.shape[0]].shape,self.Nticks,d.Nticks)
                # if x.shape[0]!=df_u.shape[0]
                # x=x[:df_u.shape[0]]
                ax.fill_between(x[:df_u.shape[0]], df_u, df_b, color=c, alpha=.2, zorder=0)

                if show_first:
                    df_single = df.xs(df.index.get_level_values('AgentID')[0], level='AgentID')
                    ax.plot(x, df_single, color=c, linestyle='dashed', linewidth=1)

            # plot the mean on top
            df_m = df.groupby(level='Step').quantile(q=0.5)
            ax.plot(x[:df_m.shape[0]], df_m, c, label=l, linewidth=2, alpha=1.0, zorder=10)
        self.conf_ax(ax=ax, xlab=f'time, ${unit}$', ylab=ylab,
                  xlim=xlim, ylim=ylim, xMaxN=5, yMaxN=5, leg_loc=leg_loc, **kwargs)


    def plot_hist(self,half_circles=True, use_title=False,par_legend=False,
                  nbins=30,alpha=0.5,ylim=[0, 0.2],Nagents_in_label=True, **kwargs):
        loc = 'upper left' if half_circles else 'upper right'
        for i, k in enumerate(self.ks):
            p = self.pdict[k]
            vs = self.vdict[k]
            if self.ranges :
                r=self.ranges[i]
                r0, r1 = r,r
            else :
                r0,r1=np.min([np.min(v) for v in vs]), np.max([np.max(v) for v in vs])
            if self.absolute :
                r0=0
            bins = np.linspace(r0, r1, nbins)
            xlim = (r0, r1)
            plot.prob_hist(vs=vs, colors=self.colors, labels=self.labels, ax=self.axs[i], bins=bins, alpha=alpha, **kwargs)
            self.conf_ax(i, ylab='probability', yvis=True if i % self.Ncols == 0 else False,
                         xlab=p.l, xlim=xlim,ylim=ylim,
                      xMaxN=4, yMaxN=4, xMath=True, title=p.disp if use_title else None,
                         leg_loc=loc if par_legend else None)

        self.comp_all_pvalues()
        if half_circles:
            self.plot_all_half_circles()
        self.data_leg(0, loc=loc,Nagents_in_label=Nagents_in_label)

    def boxplots(self,grouped=False,annotation=True, show_ns=False,target_only=None,
                 stripplot=False, ylims=None, **kwargs):

        if not grouped:
            hue = None
            palette = dict(zip(self.labels, self.colors))
        else:
            hue = 'GroupID'
            palette = dict(zip(self.group_ids, aux.N_colors(self.Ngroups)))
        kws0 = {
            'x': "DatasetID",
            'palette': palette,
            'hue': hue,
            'data': aux.concat_datasets(dict(zip(self.labels, self.datasets)), key=self.key),
        }

        for ii, k in enumerate(self.ks):
            p = self.pdict[k]
            kws = {
                'y': p.d,
                'ax': self.axs[ii],
                **kws0
            }
            plot.single_boxplot(stripplot=stripplot,annotation=annotation, show_ns=show_ns, target_only=target_only,**kws)
            self.conf_ax(ii, xticklabelrotation=30, ylab=p.l, yMaxN=4,
                          ylim=ylims[ii] if ylims is not None else None,
                          xvis=False if ii < (self.Nrows - 1) * self.Ncols else True)





class GridPlot(BasePlot):
    def __init__(self, name, width, height, scale=(1, 1), **kwargs):
        '''
        Class for compiling composite plots

        '''
        super().__init__(name, **kwargs)
        ws, hs = scale
        self.width, self.height = width, height
        figsize = (int(width * ws), int(height * hs))
        self.fig = plt.figure(constrained_layout=False, figsize=figsize)
        self.grid = GridSpec(height, width, figure=self.fig)
        self.cur_w, self.cur_h = 0, 0

        self.cur_idx = 0
        self.letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        self.letter_dict = {}
        self.x0s, self.y0s = [], []


    def add(self, N=1, w=None, h=None, w0=None, h0=None, dw=0, dh=0, share_w=False, share_h=False, letter=True,
            x0=False, y0=False, cols_first=False, annotate_all=False):

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
            self.add_letter(axs, letter, x0=x0, y0=y0)
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
            if annotate_all :
                for i,ax in enumerate(axs) :
                    if i==0 :
                        self.add_letter(ax, letter, x0=x0, y0=y0)
                    else:
                        self.add_letter(ax, letter)
            else:
                self.add_letter(axs[0], letter, x0=x0, y0=y0)
            # ax_letter = axs[0]
        return axs

    def add_letter(self, ax, letter=True, x0=False, y0=False):
        if letter:
            self.letter_dict[ax] = self.letters[self.cur_idx]
            self.cur_idx += 1
            if x0:
                self.x0s.append(ax)
            if y0:
                self.y0s.append(ax)

    def annotate(self, dx=-0.05, dy=0.005, full_dict=False):
        text_x0, text_y0 = 0.05, 0.98

        if full_dict:

            for i, ax in enumerate(self.axs):
                self.letter_dict[ax] = self.letters[i]
        for i, (ax, text) in enumerate(self.letter_dict.items()):
            X = text_x0 if ax in self.x0s else ax.get_position().x0 + dx
            Y = text_y0 if ax in self.y0s else ax.get_position().y1 + dy
            self.fig.text(X, Y, text, size=30, weight='bold')

    def plot(self, func, kws, axs=None, **kwargs):
        if axs is None:
            axs = self.add(**kwargs)
        _ = reg.graphs.run(ID = func,fig=self.fig, axs=axs, **kws)
