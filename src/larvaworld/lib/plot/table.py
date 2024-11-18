"""
Tables
"""

import numpy as np
import pandas as pd
import param

from ... import ROOT_DIR
from .. import plot, reg, util, funcs

__all__ = [
    "modelConfTable",
    "mpl_table",
    "conf_table",
    "mdiff_table",
    "error_table",
    "store_model_graphs",
    "diff_df",
]


def arrange_index_labels(index):
    ks = index.unique().tolist()
    Nks = index.value_counts(sort=False)

    def merge(k, Nk):
        Nk1 = int((Nk - 1) / 2)
        Nk2 = Nk - 1 - Nk1
        return [""] * Nk1 + [k.upper()] + [""] * Nk2

    new = util.flatten_list([merge(k, Nks[k]) for k in ks])
    return new


def conf_table(
    df,
    row_colors,
    mID,
    show=False,
    save_to=None,
    save_as=None,
    build_kws={"Nrows": 1, "Ncols": 1, "w": 15, "h": 20},
    **kwargs,
):
    ax, fig, mpl = mpl_table(
        df,
        header0="MODULE",
        header0_color="darkred",
        cellLoc="center",
        rowLoc="center",
        build_kws=build_kws,
        adjust_kws={"left": 0.2, "right": 0.95},
        row_colors=row_colors,
        return_table=True,
        **kwargs,
    )

    mmID = mID.replace("_", "-")
    ax.set_title("Model ID : " + rf"${mmID}$", y=1.05, fontsize=30)

    if save_as is None:
        save_as = mID
    P = plot.AutoBasePlot(
        name="conf_table", save_as=save_as, save_to=save_to, show=show, fig=fig, axs=ax
    )
    return P.get()


@funcs.graph("model table")
def modelConfTable(
    mID,
    m=None,
    columns=["parameter", "symbol", "value", "unit"],
    colWidths=[0.35, 0.1, 0.25, 0.15],
    **kwargs,
):
    from ..model import moduleDB as MD

    def mIDtable_data(m, columns):
        def gen_rows2(d, parent, data):
            for k, p in d.items():
                if isinstance(p, param.Parameterized):
                    ddd = [getattr(p, pname) for pname in columns]
                    row = [parent] + ddd
                    data.append(row)

        data = []
        for k in MD.BrainMods:
            d0 = m.brain[k]
            if d0 is not None:
                if k == "intermitter":
                    d = MD.module_conf(mID=k, mode=d0.mode, as_entry=False, **d0)
                    run_mode = d["run_mode"]
                    for p in d.keylist:
                        if p == "run_dist" and run_mode == "stridechain":
                            continue
                        if p == "stridechain_dist" and run_mode == "exec":
                            continue
                        v = d[p]
                        if v is not None:
                            if v.name is not None:
                                vs1, vs2 = reg.get_dist(
                                    k=p, k0=k, v=v, return_tabrows=True
                                )
                                data.append(vs1)
                                data.append(vs2)
                else:
                    gen_rows2(d0, k, data)

        gen_rows2(MD.body_kws(**m.body), "body", data)
        gen_rows2(MD.physics_kws(**m.physics), "physics", data)
        if "sensorimotor" in m and m.sensorimotor is not None:
            gen_rows2(MD.sensorimotor_kws(**m.sensorimotor), "sensorimotor", data)
        if m.energetics is not None:
            d = MD.energetics_kws(DEB_kws=m.energetics.DEB, gut_kws=m.energetics.gut)
            gen_rows2(d.DEB, "DEB", data)
            gen_rows2(d.gut, "gut", data)

        df = pd.DataFrame(data, columns=["field"] + columns)
        df.set_index(["field"], inplace=True)
        return df

    if m is None:
        m = reg.conf.Model.getID(mID)
    df = mIDtable_data(m, columns=columns)
    row_colors = [None] + [MD.ModuleColorDict[ii] for ii in df.index.values]
    df.index = arrange_index_labels(df.index)
    return conf_table(df, row_colors, mID=mID, colWidths=colWidths, **kwargs)


@funcs.graph("mpl")
def mpl_table(
    data,
    cellLoc="center",
    colLoc="center",
    rowLoc="center",
    font_size=14,
    title=None,
    name="mpl_table",
    header0=None,
    header0_color=None,
    header_color="#40466e",
    row_colors=["#f1f1f2", "w"],
    edge_color="black",
    adjust_kws=None,
    highlighted_celltext_dict=None,
    highlighted_cells=None,
    bbox=[0, 0, 1, 1],
    header_columns=0,
    colWidths=None,
    highlight_color="yellow",
    return_table=False,
    **kwargs,
):
    def get_idx(highlighted_cells):
        d = data.values
        res = []
        if highlighted_cells == "row_min":
            idx = np.nanargmin(d, axis=1)
            for i in range(d.shape[0]):
                res.append((i + 1, idx[i]))
                for j in range(d.shape[1]):
                    if d[i, j] == d[i, idx[i]] and j != idx[i]:
                        res.append((i + 1, j))
        elif highlighted_cells == "row_max":
            idx = np.nanargmax(d, axis=1)
            for i in range(d.shape[0]):
                res.append((i + 1, idx[i]))
                for j in range(d.shape[1]):
                    if d[i, j] == d[i, idx[i]] and j != idx[i]:
                        res.append((i + 1, j))
        elif highlighted_cells == "col_min":
            idx = np.nanargmin(d, axis=0)
            for i in range(d.shape[1]):
                res.append((idx[i] + 1, i))
                for j in range(d.shape[0]):
                    if d[j, i] == d[idx[i], i] and j != idx[i]:
                        res.append((j + 1, i))
        elif highlighted_cells == "col_max":
            idx = np.nanargmax(d, axis=0)
            for i in range(d.shape[1]):
                res.append((idx[i] + 1, i))
                for j in range(d.shape[0]):
                    if d[j, i] == d[idx[i], i] and j != idx[i]:
                        res.append((j + 1, i))
        return res

    try:
        highlight_idx = get_idx(highlighted_cells)
    except:
        highlight_idx = []
    P = plot.AutoBasePlot(name=name, **kwargs)

    ax = P.axs[0]
    ax.axis("off")
    mpl = ax.table(
        cellText=data.values,
        bbox=bbox,
        colLabels=data.columns.values,
        rowLoc=rowLoc,
        rowLabels=data.index.values,
        colWidths=colWidths,
        colLoc=colLoc,
        cellLoc=cellLoc,
    )
    # FIXME deleted **kwargs

    mpl.auto_set_font_size(False)
    mpl.set_fontsize(font_size)

    for k, cell in mpl._cells.items():
        cell.set_edgecolor(edge_color)
        if k in highlight_idx:
            cell.set_facecolor(highlight_color)
        elif k[0] == 0:
            cell.set_text_props(weight="bold", color="w")
            cell.set_facecolor(header_color)
        elif k[1] < header_columns:
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    if header0 is not None:
        if header0_color is None:
            header0_color = header_color
        mpl.add_cell(
            0,
            -1,
            facecolor=header0_color,
            loc="center",
            width=0.5,
            height=mpl._approx_text_height(),
            text=header0,
        )
        mpl._cells[(0, -1)].set_text_props(weight="bold", color="w", fontsize=font_size)

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


@funcs.graph("model diff")
def mdiff_table(mIDs, dIDs, show=False, save_to=None, save_as=None, **kwargs):
    data, row_colors = diff_df(mIDs=mIDs, dIDs=dIDs)
    mpl_kws = {
        "name": "mdiff_table",
        "header0": "MODULE",
        "header0_color": "darkred",
        "name": "mdiff_table",
        "figsize": (24, 14),
        "adjust_kws": {"left": 0.3, "right": 0.95},
        "font_size": 14,
        "highlighted_celltext_dict": {
            "green": ["sample"],
            "grey": ["nan", "", None, np.nan],
        },
        "cellLoc": "center",
        "rowLoc": "center",
        "row_colors": row_colors,
    }
    mpl_kws.update(kwargs)

    ax, fig, mpl = mpl_table(data, return_table=True, **mpl_kws)

    mpl._cells[(0, 0)].set_text_props(weight="bold", color="w")
    mpl._cells[(0, 0)].set_facecolor(mpl_kws["header0_color"])

    P = plot.AutoBasePlot(
        "mdiff_table", save_as=save_as, save_to=save_to, show=show, fig=fig, axs=ax
    )
    return P.get()


@funcs.graph("error table")
def error_table(data, k="", **kwargs):
    data = np.round(data, 3).T
    figsize = ((data.shape[1] + 3) * 4, data.shape[0])
    fig = mpl_table(
        data,
        highlighted_cells="row_min",
        figsize=figsize,
        adjust_kws={"left": 0.3, "right": 0.95},
        name=f"error_table_{k}",
        **kwargs,
    )
    return fig


def store_model_graphs(mIDs=None):
    from .grid import model_summary

    f1 = f"{ROOT_DIR}/media/model_tables"
    f2 = f"{ROOT_DIR}/media/model_summaries"
    if mIDs is None:
        mIDs = reg.conf.Model.confIDs
    for mID in mIDs:
        try:
            _ = modelConfTable(mID, save_to=f1)
        except:
            print("TABLE FAIL", mID)
        try:
            _ = model_summary(refID="None.150controls", mID=mID, Nids=10, save_to=f2)
        except:
            print("SUMMARY FAIL", mID)

    util.combine_pdfs(file_dir=f1, save_as="___ALL_MODEL_CONFIGURATIONS___.pdf")
    util.combine_pdfs(file_dir=f2, save_as="___ALL_MODEL_SUMMARIES___.pdf")


def diff_df(mIDs, ms=None, dIDs=None):
    from ..model import moduleDB as MD

    dic = {}
    if dIDs is None:
        dIDs = mIDs
    if ms is None:
        ms = reg.conf.Model.getID(mIDs)
    ms = [m.flatten() for m in ms]
    ks = util.unique_list(util.flatten_list([m.keylist for m in ms]))

    for k in ks:
        entry = {dID: m[k] if k in m else None for dID, m in zip(dIDs, ms)}
        l = list(entry.values())
        if all([a == l[0] for a in l]):
            continue
        else:
            k0 = k.split(".")[-1]
            k00 = k.split(".")[0]
            if k00 == "brain":
                k01 = k.split(".")[1]
                k00 = k01.split("_")[0]
            entry["field"] = k00
            dic[k0] = entry
    df = pd.DataFrame.from_dict(dic).T
    df.index = df.index.set_names(["parameter"])
    df.reset_index(drop=False, inplace=True)
    df.set_index(["field"], inplace=True)
    df.sort_index(inplace=True)

    row_colors = [None] + [MD.ModuleColorDict[ii] for ii in df.index.values]
    df.index = arrange_index_labels(df.index)

    return df, row_colors
