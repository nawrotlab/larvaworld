import os

import numpy as np
import pandas as pd
import six
from matplotlib import pyplot as plt

from lib.aux import dictsNlists as dNl
from lib.plot.base import BasePlot


def modelConfTable(mID, save_to=None, save_as=None, columns=['Parameter', 'Symbol', 'Value', 'Unit'], rows=None,
                   figsize=(14, 11), **kwargs):
    from lib.conf.base.dtypes import par
    from lib.conf.base.init_pars import init_pars
    from lib.conf.stored.conf import loadConf
    m = loadConf(mID, "Model")
    if rows is None:
        rows = ['physics', 'body'] + [k for k, v in m.brain.modules.items() if v]

    rowDicts = []
    for k in rows:
        try:
            rowDicts.append(m[k])
        except:
            rowDicts.append(m.brain[f'{k}_params'])
    rowColors0 = ['lightskyblue', 'lightsteelblue', 'lightcoral', 'indianred', 'lightsalmon', '#a55af4', 'palegreen',
                  'plum', 'pink'][:len(rows)]
    Nrows = {rowLab: 0 for rowLab in rows}

    def register(vs, rowColor):
        data.append(vs)
        rowColors.append(rowColor)
        Nrows[vs[0]] += 1

    rowColors = [None]
    data = []

    dvalid = dNl.AttrDict.from_nested_dicts({'interference': {
        'square': ['crawler_phi_range', 'attenuation', 'attenuation_max'],
        'phasic': ['max_attenuation_phase', 'attenuation', 'attenuation_max'],
        'default': ['attenuation']
    },
        'turner': {
            'neural': ['base_activation', 'activation_range', 'n', 'tau'],
            'constant': ['initial_amp'],
            'sinusoidal': ['initial_amp', 'initial_freq']
        },
        'crawler': {
            'realistic': ['initial_freq', 'max_scaled_vel', 'max_vel_phase', 'stride_dst_mean', 'stride_dst_std'],
            'constant': ['initial_amp']
        },
        'physics': ['ang_damping', 'torque_coef', 'body_spring_k', 'bend_correction_coef'],
        'body': ['initial_length', 'Nsegs'],
        'olfactor': ['decay_coef'],
    })

    for l, dd, rowColor in zip(rows, rowDicts, rowColors0):
        d0 = init_pars().get(l, None)
        if l in ['physics', 'body', 'olfactor']:
            rowValid = dvalid[l]
        elif l == 'interference':
            rowValid = dvalid[l][dd.mode]
        elif l == 'turner':
            rowValid = dvalid[l][dd.mode]
        elif l == 'crawler':
            rowValid = dvalid[l][dd.waveform]
        elif l == 'intermitter':
            rowValid = [n for n in ['stridechain_dist', 'pause_dist'] if dd[n] is not None and dd[n].name is not None]

        if len(rowValid) == 0:
            Nrows.pop(l, None)
            continue
        for n, vv in d0.items():
            if n not in rowValid:
                continue
            v = dd[n]
            if n in ['stridechain_dist', 'pause_dist']:
                dist_dic={
                    'exponential' : f'Exp(b={v.beta})',
                    'powerlaw' : f'Powerlaw(a={v.alpha})',
                    'levy' : f'Levy(m={v.mu}, s={v.sigma})',
                    'uniform' : f'Uniform()',
                    'lognormal' : f'Lognormal(m={np.round(v.mu, 2)}, s={np.round(v.sigma, 2)})'
                }
                dist_v = dist_dic[v.name]

                if n == 'stridechain_dist':
                    vs1 = [l, 'run length distribution', '$N_{R}$', dist_v, '-']
                    vs2 = [l, 'run length range', '$[N_{R}^{min},N_{R}^{max}]$', v.range, '# $strides$']
                elif n == 'pause_dist':
                    vs1 = [l, 'pause duration distribution', '$t_{P}$', dist_v, '-']
                    vs2 = [l, 'pause duration range', '$[t_{P}^{min},t_{P}^{max}]$', v.range, '$sec$']
                register(vs1, rowColor)
                register(vs2, rowColor)
            else:
                p = par(n, **vv)

                if n == 'initial_length':
                    v *= 1000
                elif n == 'suppression_mode':
                    if v == 'both':
                        v = '$I_{T}$ & $\omega$'
                    elif v == 'amplitude':
                        v = fr'$\omega$'
                    elif v == 'oscillation':
                        v = '$I_{T}$'

                else:
                    try:
                        v = np.round(v, 2)
                    except:
                        pass
                vs = [l, p[n]['label'], p[n]['symbol'], v, p[n]['unit']]
                register(vs, rowColor)

    cumNrows = dict(zip(list(Nrows.keys()), np.cumsum(list(Nrows.values())).astype(int)))
    df = pd.DataFrame(data, columns=['field'] + columns)
    df.set_index(['field'], inplace=True)

    ax, fig, mpl = render_mpl_table(df, colWidths=[0.35, 0.1, 0.25, 0.15], cellLoc='center', rowLoc='center',
                                    figsize=figsize, adjust_kws={'left': 0.2, 'right': 0.95},
                                    row_colors=rowColors, return_table=True, **kwargs)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    for k, cell in mpl._cells.items():
        if k[1] == -1:
            cell._text._text = ''
            cell._linewidth = 0

    for rowLab, idx in cumNrows.items():
        try:
            cell = mpl._cells[(idx - Nrows[rowLab] + 1, -1)]
            cell._text._text = rowLab.upper()
        except:
            pass
    if save_to is not None:
        os.makedirs(save_to, exist_ok=True)
        if save_as is None:
            save_as = mID
        filename = f'{save_to}/{save_as}.pdf'
        fig.savefig(filename, dpi=300)
    plt.close()
    return fig


def render_mpl_table(data, col_width=4.0, row_height=0.625, font_size=14, title=None,figsize=None,save_to=None,name='mpl_table',
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='black',show=False,adjust_kws=None,
                     bbox=[0, 0, 1, 1], header_columns=0, axs=None,fig=None,  highlighted_cells=None, highlight_color='yellow', return_table=False,
                     **kwargs):
    def get_idx(highlighted_cells):
        d = data.values
        res = []
        if highlighted_cells == 'row_min':
            idx = np.nanargmin(d, axis=1)
            for i in range(d.shape[0]):
                res.append((i + 1, idx[i]))
                for j in range(d.shape[1]):
                    if d[i, j] == d[i, idx[i]] and j != idx[i]:
                        res.append((i + 1, j))
        elif highlighted_cells == 'row_max':
            idx = np.nanargmax(d, axis=1)
            for i in range(d.shape[0]):
                res.append((i + 1, idx[i]))
                for j in range(d.shape[1]):
                    if d[i, j] == d[i, idx[i]] and j != idx[i]:
                        res.append((i + 1, j))
        elif highlighted_cells == 'col_min':
            idx = np.nanargmin(d, axis=0)
            for i in range(d.shape[1]):
                res.append((idx[i] + 1, i))
                for j in range(d.shape[0]):
                    if d[j, i] == d[idx[i], i] and j != idx[i]:
                        res.append((j + 1, i))
        elif highlighted_cells == 'col_max':
            idx = np.nanargmax(d, axis=0)
            for i in range(d.shape[1]):
                res.append((idx[i] + 1, i))
                for j in range(d.shape[0]):
                    if d[j, i] == d[idx[i], i] and j != idx[i]:
                        res.append((j + 1, i))
        # else :
        #     res=  []
        return res

    try:
        highlight_idx = get_idx(highlighted_cells)
    except:
        highlight_idx = []

    P=BasePlot(name=name, save_to=save_to, show=show)
    if figsize is None:
        figsize = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
    P.build(1,1,figsize = figsize, axs=axs, fig=fig)
    ax=P.axs[0]
    ax.axis('off')
    mpl = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns.values,
                   rowLabels=data.index.values, **kwargs)
    mpl.auto_set_font_size(False)
    mpl.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl._cells):
        cell.set_edgecolor(edge_color)
        if k in highlight_idx:
            cell.set_facecolor(highlight_color)
        elif k[0] == 0 :
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        elif k[1] < header_columns:
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    ax.set_title(title)

    if adjust_kws is not None :
        P.fig.subplots_adjust(**adjust_kws)
    if return_table:
        return ax, P.fig, mpl
    else :
        return P.get()


def error_table(data, k='',title = None, **kwargs):
    from lib.plot.table import render_mpl_table
    data = np.round(data, 3).T
    figsize = ((data.shape[1] + 3) * 4, data.shape[0])
    fig = render_mpl_table(data, highlighted_cells='row_min', title=title, figsize=figsize,
                               adjust_kws={'left': 0.3, 'right': 0.95},
                               name=f'error_table_{k}', **kwargs)
    return fig
