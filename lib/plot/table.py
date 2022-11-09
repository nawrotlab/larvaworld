import os

import numpy as np
import six


from lib.plot.base import BasePlot, AutoBasePlot
from lib.aux import dictsNlists as dNl
from lib.registry import reg


def modelConfTable(mID, **kwargs):
    #from lib.registry.pars import preg
    return reg.Dic.MD.mIDtable(mID, **kwargs)


def mtable(k, columns=['symbol', 'value', 'description'], figsize=(14, 11),
           show=False, save_to=None, save_as=None, **kwargs):
    from lib.registry.pars import preg
    from lib.aux.data_aux import mdict2df, init2mdict
    # mdict = preg.init_dict.init2mdict(k)
    mdict = init2mdict(preg.init_dict.dict[k])

    df = mdict2df(mdict, columns=columns)

    # row_colors = [None] + [None for ii in df.index.values]
    ax, fig, mpl = mpl_table(df, header0=columns[0],
                             # colWidths=[0.35, 0.1, 0.25, 0.15],
                             cellLoc='center', rowLoc='center',
                             figsize=figsize, adjust_kws={'left': 0.2, 'right': 0.95},
                             # row_colors=row_colors,
                             return_table=True,
                             **kwargs)
    if save_as is None:
        save_as = k
    P = BasePlot('mtable', save_as=save_as, save_to=save_to, show=show)
    P.set(fig)
    return P.get()


def conf_table(df, row_colors, mID, show=False, save_to=None, save_as=None,
               build_kws={'Nrows': 1, 'Ncols': 1, 'w': 15, 'h': 20}, **kwargs):

    ax, fig, mpl = mpl_table(df, header0='MODULE', header0_color= 'darkred',
                             cellLoc='center', rowLoc='center',build_kws=build_kws,
                             adjust_kws={'left': 0.2, 'right': 0.95},
                             row_colors=row_colors, return_table=True, **kwargs)

    mmID = mID.replace("_", "-")
    # ax.set_title(f'Model ID : ' + rf'$\bf{mmID}$', y=1.05, fontsize=30)
    ax.set_title(f'Model ID : ' + rf'${mmID}$', y=1.05, fontsize=30)

    if save_as is None:
        save_as = mID
    P = BasePlot('conf_table', save_as=save_as, save_to=save_to, show=show)
    P.set(fig)
    return P.get()





def mpl_table(data, cellLoc='center',colLoc='center', rowLoc='center', font_size=14, title=None,
              name='mpl_table', header0=None,header0_color=None,
              header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='black',
              adjust_kws=None, highlighted_celltext_dict=None,highlighted_cells=None,
              bbox=[0, 0, 1, 1], header_columns=0,colWidths=None,
              highlight_color='yellow', return_table=False,
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
    P = AutoBasePlot(name=name, **kwargs)

    ax = P.axs[0]
    ax.axis('off')
    mpl = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns.values,rowLoc=rowLoc,
                   rowLabels=data.index.values, colWidths=colWidths, colLoc=colLoc,cellLoc=cellLoc,
                   )
    # FIXME deleted **kwargs
                   # rowLabels=data.index.values, **kwargs)
    mpl.auto_set_font_size(False)
    mpl.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl._cells):
        cell.set_edgecolor(edge_color)
        if k in highlight_idx:
            cell.set_facecolor(highlight_color)
        elif k[0] == 0:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        elif k[1] < header_columns:
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    if header0 is not None:
        if header0_color is None:
            header0_color=header_color
        mpl.add_cell(0, -1, facecolor=header0_color, loc='center',
                     width=0.5, height=mpl._approx_text_height(),
                     text=header0)
        mpl._cells[(0, -1)].set_text_props(weight='bold', color='w', fontsize=font_size)

    if highlighted_celltext_dict is not None:
        for color, texts in highlighted_celltext_dict.items():
            for (k0, k1), cell in mpl._cells.items():
                if k1 != -1:
                    if any([cell._text._text == text for text in texts]):
                        cell.set_facecolor(color)

    ax.set_title(title)

    if adjust_kws is not None:
        P.fig.subplots_adjust(**adjust_kws)
    if return_table:
        return ax, P.fig, mpl
    else:
        return P.get()


def mdiff_table(mIDs, dIDs,show=False, save_to=None, save_as=None, **kwargs):
    from lib.registry.pars import preg
    data, row_colors = preg.larva_conf_dict.diff_df(mIDs=mIDs, dIDs=dIDs)
    mpl_kws = {
        'name': 'mdiff_table',
        'header0': 'MODULE',
        'header0_color': 'darkred',
        'name': 'mdiff_table',
        'figsize': (24, 14),
        'adjust_kws': {'left': 0.3, 'right': 0.95},
        'font_size': 14,
        'highlighted_celltext_dict': {'green': ['sample'], 'grey': ['nan', '', None, np.nan]},
        'cellLoc' : 'center',
        'rowLoc' : 'center',
        'row_colors': row_colors
    }
    mpl_kws.update(kwargs)

    ax, fig, mpl =mpl_table(data, return_table=True,**mpl_kws)


    mpl._cells[(0, 0)].set_text_props(weight='bold', color='w')
    mpl._cells[(0, 0)].set_facecolor(mpl_kws['header0_color'])


    if save_as is None:
        save_as = 'mdiff_table'
    P = BasePlot('mdiff_table', save_as=save_as, save_to=save_to, show=show)
    P.set(fig)
    return P.get()


def error_table(data, k='', title=None, **kwargs):
    data = np.round(data, 3).T
    figsize = ((data.shape[1] + 3) * 4, data.shape[0])
    fig = mpl_table(data, highlighted_cells='row_min', title=title, figsize=figsize,
                    adjust_kws={'left': 0.3, 'right': 0.95},
                    name=f'error_table_{k}', **kwargs)
    return fig


def store_model_graphs(mIDs=None):
    from lib.registry.pars import preg
    from lib.aux.combining import combine_pdfs
    from lib.plot.grid import model_summary
    f1 = preg.path_dict['model_tables']
    f2 = preg.path_dict['model_summaries']
    if mIDs is None:
        mIDs = preg.storedConf('Model')
    for mID in mIDs:
        try:
            _ = modelConfTable(mID, save_to=f1)
        except:
            print('TABLE FAIL', mID)
        try:
            _ = model_summary(refID='None.150controls', mID=mID, Nids=10, save_to=f2)
        except:
            print('SUMMARY FAIL', mID)

    combine_pdfs(file_dir=f1, save_as="___ALL_MODEL_CONFIGURATIONS___.pdf")
    combine_pdfs(file_dir=f2, save_as="___ALL_MODEL_SUMMARIES___.pdf")


if __name__ == '__main__':
    # for mID in kConfDict('Model'):
    #     print(mID)
    # raise
    # pass
    mID = 'basic_navigator'
    _ = modelConfTable(mID, show=True)
    # store_model_graphs()
