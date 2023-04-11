import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm, pyplot as plt
from scipy.stats import multivariate_normal

from larvaworld.lib import reg, aux, plot


def plot_surface(x, y, z, vars, target, z0=None, ax=None, fig=None, title=None, lims=None, azim=115, elev=15, **kwargs):
    P = plot.ParPlot(name='3d_surface', **kwargs)
    P.build(fig=fig, axs=ax, dim3=True, azim=azim, elev=elev)
    P.conf_ax_3d(vars, target, lims=lims, title=title)
    P.axs[0].plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    if z0 is not None:
        P.axs[0].plot_surface(x, y, np.ones(x.shape) * z0, alpha=0.5)
    return P.get()

@reg.funcs.graph('odorscape')
def plot_odorscape(odor_layers, scale=1.0, idx=0, **kwargs):
    for id, layer in odor_layers.items():
        X, Y = layer.meshgrid
        x = X * 1000 / scale
        y = Y * 1000 / scale
        plot_surface(x=x, y=y, z=layer.get_grid(), vars=[r'x $(mm)$', r'y $(mm)$'], target=r'concentration $(μM)$',
                     title=f'{id} odorscape', save_as=f'{id}_odorscape_{idx}', **kwargs)


def odorscape_from_config(c, mode='2D', fig=None, axs=None, show=True, grid_dims=(201, 201), col_max=(0, 0, 0),
                          **kwargs):
    env = c.env_params
    source = list(env.food_params.source_units.values())[0]
    a0, b0 = source.pos
    oP, oS = source.odor.odor_intensity, source.odor.odor_spread
    oD = multivariate_normal([0, 0], [[oS, 0], [0, oS]])
    oM = oP / oD.pdf([0, 0])
    if col_max is None:
        col_max = source.default_color if source.default_color is not None else (0, 0, 0)
    if grid_dims is not None:
        X, Y = grid_dims
    else:
        X, Y = [51, 51] if env.odorscape.grid_dims is None else env.odorscape.grid_dims
    Xdim, Ydim = env.arena.dims
    s = 1
    Xmesh, Ymesh = np.meshgrid(np.linspace(-Xdim * s / 2, Xdim * s / 2, X), np.linspace(-Ydim * s / 2, Ydim * s / 2, Y))

    @np.vectorize
    def func(a, b):
        return oD.pdf([a - a0, b - b0]) * oM

    grid = func(Xmesh, Ymesh)

    if mode == '2D':
        if fig is None and axs is None:
            fig, axs = plt.subplots(1, 1, figsize=(10, 10 * Ydim / Xdim))
        q = grid.flatten() - np.min(grid)
        q /= np.max(q)
        cols = aux.col_range(q, low=(255, 255, 255), high=col_max, mul255=False)
        x, y = Xmesh * 1000 / s, Ymesh * 1000 / s,
        axs.scatter(x=x, y=y, color=cols)
        axs.set_aspect('equal', adjustable='box')
        axs.set_xlim([np.min(x), np.max(x)])
        axs.set_ylim([np.min(y), np.max(y)])
        axs.set_xlabel(r'X $(mm)$')
        axs.set_ylabel(r'Y $(mm)$')
        if show:
            plt.show()
    elif mode == '3D':
        return plot_surface(x=Xmesh * 1000 / s, y=Ymesh * 1000 / s, z=grid, vars=[r'X $(mm)$', r'Y $(mm)$'],
                            target=r'concentration $(μM)$', save_as=f'odorscape', show=show, fig=fig, ax=axs, azim=0,
                            elev=0)


def odorscape_with_sample_tracks(datasets, unit='mm', fig=None, axs=None, show=False, save_to=None, **kwargs):
    scale = 1000 if unit == 'mm' else 1
    if fig is None and axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(20, 20))
    odorscape_from_config(datasets[0].config, mode='2D', fig=fig, axs=axs, show=False, **kwargs)
    for d in datasets:
        s, c = d.step_data, d.config
        xy = s[['x', 'y']].xs(c.agent_ids[0], level="AgentID").values * scale
        axs.plot(xy[:, 0], xy[:, 1], label=c.id, color=c.color)
    axs.legend(loc='upper left', fontsize=15)
    if show:
        plt.show()
    return fig


def plot_3pars(df, vars, target, z0=None, **kwargs):
    figs = {}
    pr = f'{vars[0]}VS{vars[1]}'
    figs[f'{pr}_3d'] = plot_3d(df=df, vars=vars, target=target, **kwargs)
    try:
        x, y = np.unique(df[vars[0]].values), np.unique(df[vars[1]].values)
        X, Y = np.meshgrid(x, y)

        z = df[target].values.reshape(X.shape).T

        figs[f'{pr}_heatmap'] = plot_heatmap(z, ax_kws={'xticklabels': x.tolist(), 'yticklabels': y.tolist(),
                                                        'xlab': vars[0], 'ylab': vars[1]},
                                             cbar_kws={'label': target}, **kwargs)
        figs[f'{pr}_surface'] = plot_surface(X, Y, z, vars=vars, target=target, z0=z0, **kwargs)
    except:
        pass
    return figs


def plot_3d(df, vars, target, name=None,lims=None, title=None, surface=True, line=False, ax=None, fig=None, dfID=None,
            color='black', **kwargs):
    if name is None :
        name = '3d_plot'
    from statsmodels import api as sm
    P = plot.ParPlot(name=name, **kwargs)
    P.build(fig=fig, axs=ax, dim3=True)
    P.conf_ax_3d(vars, target, lims=lims, title=title)

    l0, l1 = vars
    X = df[vars]
    y = df[target].values

    X = sm.add_constant(X)
    # print(X[l0], X[l1], y)
    # plot hyperplane
    if surface:
        est = sm.OLS(y, X).fit()

        xx1, xx2 = np.meshgrid(np.linspace(X[l0].min(), X[l0].max(), 100),
                               np.linspace(X[l1].min(), X[l1].max(), 100))
        # plot the hyperplane by evaluating the parameters on the grid
        Z = est.params[0] + est.params[1] * xx1 + est.params[2] * xx2
        surf = P.axs[0].plot_surface(xx1, xx2, Z, cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0)
        # plot data points - points over the HP are white, points below are black
        resid = y - est.predict(X)
        P.axs[0].scatter(X[resid >= 0][l0], X[resid >= 0][l1], y[resid >= 0], color='black', alpha=0.4,
                         facecolor='white')
        P.axs[0].scatter(X[resid < 0][l0], X[resid < 0][l1], y[resid < 0], color='black', alpha=0.4, facecolor=color)
    else:
        P.axs[0].scatter(X[l0].values, X[l1].values, y, color='black', alpha=0.4)

    return P.get()


def plot_3d_multi(dfs, dfIDs, df_colors=None, show=True, **kwargs):
    from mpl_toolkits.mplot3d import Axes3D
    if df_colors is None:
        df_colors = [None] * len(dfs)
    fig = plt.figure(figsize=(18, 12))
    ax = Axes3D(fig, azim=115, elev=15)
    for df, dfID, dfC in zip(dfs, dfIDs, df_colors):
        plot_3d(df, dfID=dfID, color=dfC, ax=ax, fig=fig, show=False, **kwargs)
    if show:
        plt.show()


def plot_heatmap(z, heat_kws={}, ax_kws={}, cbar_kws={}, **kwargs):
    base_heat_kws = {'annot': True, 'cmap': cm.coolwarm, 'vmin': None, 'vmax': None}
    base_heat_kws.update(heat_kws)
    base_cbar_kws = {"orientation": "vertical"}
    base_cbar_kws.update(cbar_kws)
    P = plot.ParPlot(name='heatmap', **kwargs)
    P.build()
    sns.heatmap(z, ax=P.axs[0], **base_heat_kws, cbar_kws=base_cbar_kws)
    cax = plt.gcf().axes[-1]
    cax.tick_params(length=0)
    P.conf_ax(**ax_kws)
    P.adjust((0.15, 0.95), (0.15, 0.95))
    return P.get()

@reg.funcs.graph('PI heatmap')
def plot_heatmap_PI(z=None, csv_filepath='PIs.csv', save_as='PI_heatmap.pdf', **kwargs):
    if z is None :
        z = pd.read_csv(csv_filepath, index_col=0)
    Lgains = z.index.values.astype(int)
    Rgains = z.columns.values.astype(int)
    Ngains = len(Lgains)
    r = np.linspace(0.5, Ngains - 0.5, 5)
    ax_kws = {
        'xticklabels': Rgains[r.astype(int)],
        'yticklabels': Lgains[r.astype(int)],
        'xticklabelrotation': 0,
        'yticklabelrotation': 0,
        'xticks': r,
        'yticks': r,
        'xlab': r'Right odor gain, $G_{R}$',
        'ylab': r'Left odor gain, $G_{L}$',
        'xlabelpad': 20
    }
    heat_kws = {
        'annot': False,
        'vmin': -1,
        'vmax': 1,
        'cmap': 'RdYlGn',
    }

    cbar_kws = {
        'label': 'Preference for left odor',
        'ticks': [1, 0, -1]
    }

    return plot_heatmap(z, heat_kws=heat_kws, ax_kws=ax_kws, cbar_kws=cbar_kws, save_as=save_as, **kwargs)


def plot_2d(df, labels, **kwargs):
    P = plot.ParPlot(name='2d_plot', **kwargs)
    par = labels[0]
    res = labels[1]
    p = df[par].values
    r = df[res].values
    P.build()
    P.axs[0].scatter(p, r)
    P.conf_ax(xlab=par, ylab=res)
    return P.get()


def plot_2pars(shorts, subfolder='step', larva_legend=True, **kwargs):
    ypar, ylab, ylim = reg.getPar(shorts[1], to_return=['d', 'l', 'lim'])
    xpar, xlab, xlim = reg.getPar(shorts[0], to_return=['d', 'l', 'lim'])
    P = plot.Plot(name=f'{ypar}_VS_{xpar}', subfolder=subfolder, **kwargs)
    P.build()
    ax = P.axs[0]
    if P.Ndatasets == 1 and larva_legend:
        d = P.datasets[0]
        Nids = len(d.agent_ids)
        cs = aux.N_colors(Nids)
        s = d.read('step')
        for j, id in enumerate(d.agent_ids):
            ss = s.xs(id, level='AgentID', drop_level=True)
            ax.scatter(ss[xpar], ss[ypar], color=cs[j], marker='.', label=id)
            ax.legend()
    else:
        for d, c in zip(P.datasets, P.colors):
            s = d.read('step')
            ax.scatter(s[xpar], s[ypar], color=c, marker='.')
        P.data_leg(0, loc='upper left')
        # dataset_legend(P.labels, P.colors, ax=ax, loc='upper left')
    P.conf_ax(xlab=xlab, ylab=ylab, xlim=xlim, ylim=ylim, xMaxN=4, yMaxN=4)
    P.adjust((0.15, 0.95), (0.15, 0.92), 0.05, 0.005)
    return P.get()
