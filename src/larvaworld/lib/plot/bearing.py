"""
Bearing-related plotting
"""

import numpy as np

from .. import plot, funcs
from ..util import nam

__all__ = [
    "plot_turn_Dbearing",
    "plot_turn_Dorient2center",
    "plot_chunk_Dorient2source",
]


@funcs.graph("bearing/turn")
def plot_turn_Dbearing(
    name=None,
    min_angle=30.0,
    max_angle=180.0,
    ref_angle=None,
    source_ID="Source",
    Nplots=4,
    subfolder="turn",
    **kwargs,
):
    if ref_angle is None:
        if name is None:
            name = "turn_Dorient_to_center"
        ang0 = 0
        norm = False
        p = nam.bearing_to(source_ID)
    else:
        ang0 = ref_angle
        norm = True
        if name is None:
            name = f"turn_Dorient_to_{ang0}deg"
        p = nam.unwrap(nam.orient("front"))

    P = plot.AutoPlot(
        name=name,
        subfolder=subfolder,
        subplot_kw=dict(projection="polar"),
        build_kws={"Nrows": "Ndatasets", "Ncols": Nplots, "wh": 5, "sharey": True},
        **kwargs,
    )

    # for k, (chunk, side) in enumerate(zip(['Lturn', 'Rturn'], ['left', 'right'])):
    #     b0_par, b1_par, bd_par = nam.atStartStopChunk(p, chunk)

    for i, (d, c) in enumerate(zip(P.datasets, P.colors)):
        ii = Nplots * i
        for k, (chunk, side) in enumerate(zip(["Lturn", "Rturn"], ["left", "right"])):
            try:
                b0_par, b1_par, bd_par = nam.atStartStopChunk(p, chunk)

                b0 = d.get_par(b0_par).dropna().values.flatten()
                b1 = d.get_par(b1_par).dropna().values.flatten()
                db = d.get_par(bd_par).dropna().values.flatten()
            except:
                b0, b1, db = d.get_chunk_par(chunk=chunk, par=p, mode="extrema")
            b0 -= ang0
            b1 -= ang0
            if norm:
                b0 %= 360
                b1 = b0 + db
                b0[b0 > 180] -= 360
                b1[b0 > 180] -= 360
            B0 = np.deg2rad(b0[(np.abs(db) > min_angle) & (np.abs(db) < max_angle)])
            B1 = np.deg2rad(b1[(np.abs(db) > min_angle) & (np.abs(db) < max_angle)])
            if Nplots == 2:
                for tt, BB, aa in zip(["start", "stop"], [B0, B1], [0.3, 0.6]):
                    plot.circNarrow(P.axs[ii + k], BB, aa, tt, c)
                P.axs[ii + 1].legend(
                    bbox_to_anchor=(-0.7, 0.1), loc="center", fontsize=12
                )
            elif Nplots == 4:
                B00 = B0[B0 < 0]
                B10 = B1[B0 < 0]
                B01 = B0[B0 > 0]
                B11 = B1[B0 > 0]
                for tt, BB, aa in zip(
                    [r"$\theta^{init}_{or}$", r"$\theta^{fin}_{or}$"],
                    [(B01, B00), (B11, B10)],
                    [0.3, 0.6],
                ):
                    for kk, ss, BBB in zip(
                        [0, 1], [r"$L_{sided}$", r"$R_{sided}$"], BB
                    ):
                        plot.circNarrow(
                            P.axs[ii + k + 2 * kk], BBB, aa, f"{ss} {tt}", c
                        )
                        for iii in [ii + 1, ii + 2 + 1]:
                            P.axs[iii].legend(
                                bbox_to_anchor=(-0.3, 0.1), loc="center", fontsize=12
                            )
            if i == P.Ndatasets - 1:
                if Nplots == 2:
                    P.axs[ii + k].set_title(f"Bearing due to {side} turn.", y=-0.4)
                elif Nplots == 4:
                    P.axs[ii + k].set_title(rf"$L_{{sided}}$ {side} turn.", y=-0.4)
                    P.axs[ii + 2 + k].set_title(rf"$R_{{sided}}$ {side} turn.", y=-0.4)
    for ax in P.axs:
        ax.set_xticklabels([0, "", +90, "", 180, "", -90, ""], fontsize=15)
    P.data_leg(
        0, loc="upper center", anchor=(0.5, 0.99), bbox_transform=P.fig.transFigure
    )
    P.adjust((0.0, 1.0), (0.15, 0.9), 0.0, 0.35)
    return P.get()


@funcs.graph("bearing to center/turn")
def plot_turn_Dorient2center(**kwargs):
    return plot_turn_Dbearing(ref_angle=None, **kwargs)


@funcs.graph("bearing to source/epoch")
def plot_chunk_Dorient2source(
    source_ID,
    datasets,
    name=None,
    subfolder="bouts",
    chunk="stride",
    Nbins=16,
    min_dur=0.0,
    plot_merged=False,
    **kwargs,
):
    N = len(datasets)
    if plot_merged:
        N += 1

    Ncols = int(np.ceil(np.sqrt(N)))

    if name is None:
        name = f"{chunk}_Dorient_to_{source_ID}"

    P = plot.AutoPlot(
        name=name,
        subfolder=subfolder,
        datasets=datasets,
        subplot_kw=dict(projection="polar"),
        build_kws={"N": N, "wh": 8, "sharey": True},
        **kwargs,
    )

    if plot_merged:
        P.Ndatasets += 1
        P.colors.insert(0, "black")
        P.labels.insert(0, "merged")

    c_dur = nam.dur(chunk)
    b = nam.bearing_to(source_ID)
    b0s, b1s, dbs = [], [], []
    try:
        b0_par, b1_par, db_par = nam.atStartStopChunk(b, chunk)
        for d in P.datasets:
            dur = d.get_par(c_dur).dropna().values
            b0 = d.get_par(b0_par).dropna().values
            b0 = b0[dur > min_dur]
            b0s.append(b0)
            b1 = d.get_par(b1_par).dropna().values
            b1 = b1[dur > min_dur]
            b1s.append(b1)
            db = d.get_par(db_par).dropna().values
            db = db[dur > min_dur]
            dbs.append(db)
    except:
        for d in P.datasets:
            b0, b1, db = d.get_chunk_par(
                chunk=chunk, par=b, min_dur=min_dur, mode="extrema"
            )
            b0s.append(b0)
            b1s.append(b1)
            dbs.append(db)

    if plot_merged:
        b0s.insert(0, np.vstack(b0s))
        b1s.insert(0, np.vstack(b1s))
        dbs.insert(0, np.vstack(dbs))

    for i, (b0, b1, db, label, c) in enumerate(zip(b0s, b1s, dbs, P.labels, P.colors)):
        ax = P.axs[i]
        dbm = np.round(np.mean(np.deg2rad(db)), 2)
        plot.circNarrow(ax, np.deg2rad(b0), alpha=0.3, label="start", color=c)
        plot.circNarrow(ax, np.deg2rad(b1), alpha=0.6, label="stop", color=c)
        text_x = -0.3
        text_y = 1.2
        for dy, text in zip(
            [0, 0.1, 0.2, 0.3],
            [
                f"Dataset : {label}",
                f"Chunk (#) : {chunk} ({len(b0)})",
                f"Min duration : {min_dur} sec",
                rf'Correction $\Delta\theta_{{{"or"}}} : {dbm}^{{{"o"}}}$',
            ],
        ):
            ax.text(text_x, text_y - dy, text, transform=ax.transAxes)
        P.conf_ax(
            i,
            leg_loc=[0.9, 0.9],
            title=f"Bearing before and after a {chunk}.",
            title_y=-0.2,
            titlefontsize=15,
            xticklabels=[0, "", +90, "", 180, "", -90, ""],
            xMaxFix=True,
        )
    P.adjust((0.05 * Ncols / 2, 0.9), (0.2, 0.8), 0.8, 0.3)
    return P.get()
