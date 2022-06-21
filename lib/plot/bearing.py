import numpy as np
from matplotlib import patches, ticker

from lib.aux import naming as nam,dictsNlists as dNl
from lib.plot.aux import circNarrow, circular_hist
from lib.plot.base import Plot


def plot_turn_Dbearing(min_angle=30.0, max_angle=180.0, ref_angle=None, source_ID='Source',
                       Nplots=4, subfolder='turn', **kwargs):
    if ref_angle is None:
        name = f'turn_Dorient_to_center'
        ang0 = 0
        norm = False
        p = nam.bearing2(source_ID)
    else:
        ang0 = ref_angle
        norm = True
        name = f'turn_Dorient_to_{ang0}deg'
        p = nam.unwrap(nam.orient('front'))
    P = Plot(name=name, subfolder=subfolder, **kwargs)
    P.build(P.Ndatasets, Nplots, figsize=(5 * Nplots, 5 * P.Ndatasets), subplot_kw=dict(projection='polar'),
            sharey=True)



    for i, (d, c) in enumerate(zip(P.datasets, P.colors)):
        ii = Nplots * i
        for k, (chunk, side) in enumerate(zip(['Lturn', 'Rturn'], ['left', 'right'])):
            b0_par = nam.at(p, nam.start(chunk))
            b1_par = nam.at(p, nam.stop(chunk))
            bd_par = nam.chunk_track(chunk, p)
            # print(b0_par)
            b0 = d.get_par(b0_par).dropna().values.flatten() - ang0
            b1 = d.get_par(b1_par).dropna().values.flatten() - ang0
            db = d.get_par(bd_par).dropna().values.flatten()
            if norm:
                b0 %= 360
                b1 = b0 + db
                b0[b0 > 180] -= 360
                b1[b0 > 180] -= 360
            B0 = np.deg2rad(b0[(np.abs(db) > min_angle) & (np.abs(db) < max_angle)])
            B1 = np.deg2rad(b1[(np.abs(db) > min_angle) & (np.abs(db) < max_angle)])
            if Nplots == 2:
                for tt, BB, aa in zip(['start', 'stop'], [B0, B1], [0.3, 0.6]):
                    circNarrow(P.axs[ii + k], BB, aa, tt, c)
                P.axs[ii + 1].legend(bbox_to_anchor=(-0.7, 0.1), loc='center', fontsize=12)
            elif Nplots == 4:
                B00 = B0[B0 < 0]
                B10 = B1[B0 < 0]
                B01 = B0[B0 > 0]
                B11 = B1[B0 > 0]
                for tt, BB, aa in zip([r'$\theta^{init}_{or}$', r'$\theta^{fin}_{or}$'], [(B01, B00), (B11, B10)],
                                      [0.3, 0.6]):
                    for kk, ss, BBB in zip([0, 1], [r'$L_{sided}$', r'$R_{sided}$'], BB):
                        circNarrow(P.axs[ii + k + 2 * kk], BBB, aa, f'{ss} {tt}', c)
                        for iii in [ii + 1, ii + 2 + 1]:
                            P.axs[iii].legend(bbox_to_anchor=(-0.3, 0.1), loc='center', fontsize=12)
            if i == P.Ndatasets - 1:
                if Nplots == 2:
                    P.axs[ii + k].set_title(f'Bearing due to {side} turn.', y=-0.4)
                elif Nplots == 4:
                    P.axs[ii + k].set_title(fr'$L_{{sided}}$ {side} turn.', y=-0.4)
                    P.axs[ii + 2 + k].set_title(fr'$R_{{sided}}$ {side} turn.', y=-0.4)
    for ax in P.axs:
        ax.set_xticklabels([0, '', +90, '', 180, '', -90, ''], fontsize=15)
    P.data_leg(0,loc='upper center', anchor=(0.5, 0.99),bbox_transform=P.fig.transFigure)
    # dataset_legend(P.labels, P.colors, ax=P.axs[0], loc='upper center', anchor=(0.5, 0.99),
    #                bbox_transform=P.fig.transFigure)
    P.adjust((0.0, 1.0), (0.15, 0.9), 0.0, 0.35)
    return P.get()


def plot_turn_Dorient2center(**kwargs):
    return plot_turn_Dbearing(ref_angle=None, **kwargs)


def plot_chunk_Dorient2source(source_ID, subfolder='bouts', chunk='stride', Nbins=16, min_dur=0.0, plot_merged=False,
                              **kwargs):
    P = Plot(name=f'{chunk}_Dorient_to_{source_ID}', subfolder=subfolder, **kwargs)

    if plot_merged:
        P.Ndatasets += 1
        P.colors.insert(0, 'black')
        P.labels.insert(0, 'merged')
    Ncols = int(np.ceil(np.sqrt(P.Ndatasets)))
    Nrows = Ncols - 1 if P.Ndatasets < Ncols ** 2 - Ncols else Ncols
    P.build(Nrows, Ncols, figsize=(8 * Ncols, 8 * Nrows), subplot_kw=dict(projection='polar'), sharey=True)

    durs = [d.get_par(nam.dur(chunk)).dropna().values for d in P.datasets]
    c0 = nam.start(chunk)
    c1 = nam.stop(chunk)
    b = nam.bearing2(source_ID)
    b0_par = nam.at(b, c0)
    b1_par = nam.at(b, c1)
    db_par = nam.chunk_track(chunk, b)
    b0s = [d.get_par(b0_par).dropna().values for d in P.datasets]
    b1s = [d.get_par(b1_par).dropna().values for d in P.datasets]
    dbs = [d.get_par(db_par).dropna().values for d in P.datasets]

    if plot_merged:
        b0s.insert(0, np.vstack(b0s))
        b1s.insert(0, np.vstack(b1s))
        dbs.insert(0, np.vstack(dbs))
        durs.insert(0, np.vstack(durs))

    for i, (b0, b1, db, dur, label, c) in enumerate(zip(b0s, b1s, dbs, durs, P.labels, P.colors)):
        ax = P.axs[i]
        b0 = b0[dur > min_dur]
        b1 = b1[dur > min_dur]
        db = db[dur > min_dur]
        b0m, b1m = np.mean(b0), np.mean(b1)
        dbm = np.round(np.mean(db), 2)
        if np.isnan([dbm, b0m, b1m]).any():
            continue
        circNarrow(ax, np.deg2rad(b0), alpha=0.3, label='start', color=c)
        circNarrow(ax, np.deg2rad(b1), alpha=0.6, label='stop', color=c)
        # circular_hist(ax, b0, bins=Nbins, alpha=0.3, label='start', color=c, offset=np.pi / 2)
        # circular_hist(ax, b1, bins=Nbins, alpha=0.6, label='stop', color=c, offset=np.pi / 2)
        # arrow0 = patches.FancyArrowPatch((0, 0), (np.deg2rad(b0m), 0.3), zorder=2, mutation_scale=30, alpha=0.3,
        #                                  facecolor=c, edgecolor='black', fill=True, linewidth=0.5)
        #
        # ax.add_patch(arrow0)
        # arrow1 = patches.FancyArrowPatch((0, 0), (np.deg2rad(b1m), 0.3), zorder=2, mutation_scale=30, alpha=0.6,
        #                                  facecolor=c, edgecolor='black', fill=True, linewidth=0.5)
        # ax.add_patch(arrow1)

        text_x = -0.3
        text_y = 1.2
        ax.text(text_x, text_y, f'Dataset : {label}', transform=ax.transAxes)
        ax.text(text_x, text_y - 0.1, f'Chunk (#) : {chunk} ({len(b0)})', transform=ax.transAxes)
        ax.text(text_x, text_y - 0.2, f'Min duration : {min_dur} sec', transform=ax.transAxes)
        ax.text(text_x, text_y - 0.3, fr'Correction $\Delta\theta_{{{"or"}}} : {dbm}^{{{"o"}}}$',
                transform=ax.transAxes)
        P.conf_ax(i,leg_loc=[0.9, 0.9], title=f'Bearing before and after a {chunk}.', title_y=-0.2,titlefontsize=15,
                  xticklabels = [0, '', +90, '', 180, '', -90, ''],xMaxFix=True)
        # ax.legend(loc=[0.9, 0.9])
        # ax.set_title(f'Bearing before and after a {chunk}.', fontsize=15, y=-0.2)
    # for ax in P.axs:
    #     ticks_loc = ax.get_xticks().tolist()
    #     ax.xaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
    #     ax.set_xticklabels([0, '', +90, '', 180, '', -90, ''])
    P.adjust((0.05 * Ncols / 2, 0.9), (0.2, 0.8), 0.8, 0.3)
    return P.get()




