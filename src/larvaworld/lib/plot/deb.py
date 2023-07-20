import os
import warnings

import numpy as np
from matplotlib import pyplot as plt, ticker
from scipy import signal


from larvaworld.lib import reg, aux, plot


@reg.funcs.graph('gut')
def plot_gut(**kwargs):
    P = plot.AutoPlot(name='gut', **kwargs)
    P.plot_quantiles(par='gut_occupancy', coeff=100, ylab='% gut occupied',ylim=[0, 100])
    '''
    x = P.trange()
    for l, d, c in P.data_palette:
        df = d.step_data['gut_occupancy'] * 100
        plot.plot_quantiles(df=df, x=x, axis=P.axs[0], color_shading=c, label=l)
    P.conf_ax(xlab='time, $min$', ylab='% gut occupied',
              xlim=P.tlim, ylim=[0, 100], xMaxN=5, yMaxN=5, leg_loc='upper left')
    '''

    P.adjust((0.1, 0.95), (0.15, 0.95), 0.05, 0.005)
    return P.get()



@reg.funcs.graph('food intake (timeplot)')
def plot_food_amount(filt_amount=False, scaled=False, **kwargs):
    name = 'food_intake'
    ylab = r'Cumulative food intake $(mg)$'
    par = 'ingested_food_volume'
    if scaled:
        name = f'scaled_{name}'
        ylab = r'Cumulative food intake as % larval mass'
        par = 'ingested_body_mass_ratio'
    if filt_amount:
        name = f'filtered_{name}'
        ylab = r'Food intake $(mg)$'
    if filt_amount and scaled:
        ylab = 'Food intake as % larval mass'
    P = plot.AutoPlot(name=name, **kwargs)

    for lab, d, c in P.data_palette:
        dst_df = d.step_data[par]
        dst_m = dst_df.groupby(level='Step').quantile(q=0.5)
        dst_u = dst_df.groupby(level='Step').quantile(q=0.75)
        dst_b = dst_df.groupby(level='Step').quantile(q=0.25)
        if filt_amount:
            sos = signal.butter(N=1, Wn=0.1, btype='lowpass', analog=False, fs=1/P.dt, output='sos')
            dst_m = dst_m.diff()
            dst_m.iloc[0] = 0
            dst_m = signal.sosfiltfilt(sos, dst_m)
            dst_u = dst_u.diff()
            dst_u.iloc[0] = 0
            dst_u = signal.sosfiltfilt(sos, dst_u)
            dst_b = dst_b.diff()
            dst_b.iloc[0] = 0
            dst_b = signal.sosfiltfilt(sos, dst_b)
        plot.plot_mean_and_range(x=P.trange(), mean=dst_m, lb=dst_b, ub=dst_u, axis=P.axs[0], color_shading=c, label=lab)
    P.conf_ax(xlab='time, $min$', ylab=ylab, xlim=P.tlim, xMaxN=5, leg_loc='upper left')
    P.adjust((0.1, 0.95), (0.15, 0.95), 0.05, 0.005)
    return P.get()

@reg.funcs.graph('deb')
def plot_debs(deb_dicts=None, name=None, save_to=None, mode='full', roversVSsitters=False, include_egg=True,
              time_unit='hours', sim_only=False, force_ymin=None, color_epoch_quality=True,
              datasets=None, labels=None, label_epochs=True, label_lifestages=True, **kwargs):
    warnings.filterwarnings('ignore')
    if name is None :
        name=f'debs'
    if save_to is None:
        save_to = f'{reg.SIM_DIR}/deb_runs'
    if deb_dicts is None:
        deb_dicts = aux.flatten_list([d.load_dicts('deb') for d in datasets])
    Ndebs = len(deb_dicts)
    ids = [d['id'] for d in deb_dicts]
    if Ndebs == 1:
        cols = [(0, 1, 0.1)]
        leg_ids = ids
        leg_cols = cols
    elif roversVSsitters:
        cols = []
        temp_id = None
        for id in ids:
            if str.startswith(id, 'Rover'):
                cols.append((0, 0, 1))
            elif str.startswith(id, 'Sitter'):
                cols.append((1, 0, 0))
            else:
                cols.append((0, 1, 0.1))
                temp_id = id
        if temp_id is not None:
            leg_cols = [(0, 1, 0.1), (0, 0, 1), (1, 0, 0)]
            leg_ids = [temp_id, 'Rovers', 'Sitters']
        else:
            leg_cols = [(0, 0, 1), (1, 0, 0)]
            leg_ids = ['Rovers', 'Sitters']
    else:
        cols = [(0.9 - i, 0.1 + i, 0.1) for i in np.linspace(0, 0.9, Ndebs)]
        leg_ids = ids
        leg_cols = cols

    labels0 = ['mass', 'length',
               'reserve', 'reserve_density', 'hunger',
               'pupation_buffer',
               'f', 'f_filt',
               'EEB',
               'M_gut', 'mol_ingested', 'M_absorbed', 'M_faeces', 'M_not_digested', 'M_not_absorbed',
               'R_faeces', 'R_absorbed', 'R_not_digested', 'gut_occupancy',
               'deb_p_A', 'sim_p_A', 'gut_p_A', 'gut_f', 'gut_p_A_deviation',
               'M_X', 'M_P', 'M_Pu', 'M_g', 'M_c', 'R_M_c', 'R_M_g', 'R_M_X_M_P', 'R_M_X', 'R_M_P'
               ]
    ylabels0 = ['wet weight $(mg)$', 'body length $(mm)$',
                r'reserve $(J)$', r'reserve density $(-)$', r'hunger drive $(-)$',
                r'pupation buffer $(-)$',
                r'f $^{sim}$ $(-)$', r'f $_{filt}^{sim}$ $(-)$',
                r'exploit VS explore $(-)$',
                'gut content $(mg)$', 'food intake $(C-mmole)$', 'food absorption $(mg)$',
                'faeces $(mg)$', 'food not digested $(mg)$', 'product not absorbed $(mg)$',
                'faeces fraction', 'absorption efficiency', 'fraction not digested', 'gut occupancy',
                r'[p$_{A}^{deb}$] $(microJ/cm^3)$', r'[p$_{A}^{sim}$] $(microJ/cm^3)$',
                r'[p$_{A}^{gut}$] $(microJ/cm^3)$',
                # r'[p$_{A}^{deb}$] $(\mu J/cm^3)$', r'[p$_{A}^{sim}$] $(\mu J/cm^3)$',r'[p$_{A}^{gut}$] $(\mu J/cm^3)$',
                r'f $^{gut}$ $(-)$', r'$\Delta$p$_{A}^{gut}$ $(-)$',
                r'Food in gut $(C-moles)$', r'Product in gut $(C-moles)$', r'Product absorbed $(C-mmoles)$',
                r'Active enzyme amount in gut $(-)$', r'Available carrier amount in gut surface $(-)$',
                r'Available carrier ratio in gut surface $(-)$', r'Active enzyme ratio in gut surface $(-)$',
                r'Food VS Product ratio in gut $(-)$',
                r'Ratio of Food in gut $(-)$', r'Ratio of Product in gut $(-)$'
                # r'(deb) assimilation energy $(J)$', r'(f) assimilation energy $(J)$', r'(gut) assimilation energy $(J)$'
                ]
    sharey = False
    if mode == 'energy':
        idx = [2, 3, 4, 5]
    elif mode == 'growth':
        idx = [0, 1, 5]
    elif mode == 'full':
        idx = [0, 1, 2, 3, 4, 5]
    elif mode == 'feeding':
        idx = [3, 4, 8]
    elif mode in labels0:
        idx = [labels0.index(mode)]
    elif mode == 'food_mass':
        idx = [10, 26]
        # idx = [9, 10, 11, 12, 13, 14]
        sharey = True
    elif mode == 'food_ratio':
        idx = [17, 15, 16, 18]
    elif mode == 'food_mass_1':
        idx = [9, 10, 11]
    elif mode == 'food_mass_2':
        idx = [12, 13, 14]
    elif mode == 'food_ratio_1':
        idx = [16]
        # idx = [18, 16]
    elif mode == 'food_ratio_2':
        idx = [17, 15]
    elif mode == 'assimilation':
        idx = [19, 20, 21]
        sharey = True
    elif mode == 'fs':
        idx = [6, 7, 22]
        sharey = True
    elif mode == 'plug_flow_food':
        idx = [24, 25, 32, 33, 31]
        # sharey = True
    elif mode == 'plug_flow_enzymes':
        idx = [29, 30]
        # sharey = True

    tickstep = 24
    time_unit_dict = {
        'hours': 1,
        'minutes': 60,
        'seconds': 60 * 60,
    }
    t_coef = time_unit_dict[time_unit]

    labels = [labels0[i] for i in idx]
    ylabels = [ylabels0[i] for i in idx]
    Npars = len(labels)

    P = plot.AutoBasePlot(name=name,save_to=save_to, build_kws={'Nrows': Npars, 'sharex': True,'sharey': sharey, 'w': 20, 'h': 6}, **kwargs)

    t0s, t1s, t2s, t3s, max_ages = [], [], [], [], []
    for jj, (d, id, c) in enumerate(zip(deb_dicts, ids, cols)):
        t0_sim, t0, t1, t2, t3, age = d['sim_start'], d['birth'], d['pupation'], d['death'], d['hours_as_larva'] + d[
            'birth'], np.array(d['age'])
        t00 = 0
        epochs = np.array(d['epochs'])
        if 'epoch_qs' in d.keys():
            epoch_qs = np.array(d['epoch_qs'])
        else:
            epoch_qs = np.zeros(len(epochs))
        if sim_only:
            t0_sim -= t3
            t00 -= t3
            t0 -= t3
            t1 -= t3
            t2 -= t3
            age -= t3
            epochs -= t3
            t3 = 0
        elif not include_egg:
            t0_sim -= t0
            t00 -= t0
            t1 -= t0
            t2 -= t0
            t3 -= t0
            age -= t0
            epochs -= t0
            t0 = 0

        t0_sim *= t_coef
        t00 *= t_coef
        t0 *= t_coef
        t1 *= t_coef
        t2 *= t_coef
        t3 *= t_coef
        age *= t_coef
        epochs *= t_coef
        tickstep *= t_coef

        t0s.append(t0)
        t1s.append(t1)
        t2s.append(t2)
        t3s.append(t3)
        max_ages.append(age[-1])

        for j, (l, yl) in enumerate(zip(labels, ylabels)):
            if l == 'f_filt':
                ppp = d['f']
                sos = signal.butter(N=1, Wn=d['fr'] / 1000, btype='lowpass', analog=False, fs=d['fr'], output='sos')
                ppp = signal.sosfiltfilt(sos, ppp)
            else:
                ppp = d[l]
            ax = P.axs[j]
            ax.plot(age, ppp, color=c, label=id, linewidth=2, alpha=1.0)
            for tt in [t0,t1,t2]:
                ax.axvline(tt, color=c, alpha=0.6, linestyle='dashdot', linewidth=3)

            ax.axvspan(t00, t0, color='darkgrey', alpha=0.5)
            ax.axvspan(t0, t0_sim, color='lightgrey', alpha=0.5)



            if d['simulation']:
                ax.axvspan(t0, t3, color='grey', alpha=0.05)
            for (st0, st1), qq in zip(epochs, epoch_qs):
                q_col = aux.col_range(qq, low=(255, 0, 0), high=(255, 255, 255)) if color_epoch_quality else c
                ax.axvspan(st0, st1, color=q_col, alpha=0.2)

            if l in ['pupation_buffer', 'EEB', 'R_faeces', 'R_absorbed', 'R_not_digested',
                     'gut_occupancy']:
                ylim=(0, 1)
            else :
                ylim = (None, None)
            if force_ymin is not None:
                ylim = (force_ymin, ylim[1])
            P.conf_ax(j, ylab=yl, ylabelpad=15, ylabelfontsize=10, yMaxN=3, yStrN=2, yticklabelsize=10,
                      ylim=ylim)

            if l == 'f' or mode == 'fs':
                ax.axhline(np.nanmean(ppp), color=c, alpha=0.6, linestyle='dashed', linewidth=2)
            if mode == 'assimilation':
                ax.axhline(np.nanmean(ppp), color=c, alpha=0.6, linestyle='dashed', linewidth=2)
            if label_lifestages and not sim_only:
                y0, y1 = ax.get_ylim()
                x0, x1 = ax.get_xlim()
                if jj == Ndebs - 1:
                    try:
                        ytext = y0 + 0.5 * (y1 - y0)
                        xtext = t00 + 0.5 * (t0 - t00)
                        ax.annotate('$incubation$', rotation=90, fontsize=15, va='center', ha='center',
                                    xy=(xtext, ytext), xycoords='data',
                                    )
                    except:
                        pass

                try:

                    ytext = y0 + 0.5 * (y1 - y0)
                    if not np.isnan(t1) and x1 > t1:
                        xtext = t1 + 0.5 * (x1 - t1)
                        ax.annotate('$pupation$', rotation=90, fontsize=15, va='center', ha='center',
                                    xy=(xtext, ytext), xycoords='data',
                                    )
                except:
                    pass
            if label_epochs and Ndebs == 1:

                try:
                    y0, y1 = ax.get_ylim()
                    ytext = y0 + 0.8 * (y1 - y0)
                    xpre = t0 + 0.5 * (t0_sim - t0)
                    if t0_sim - t0 > 0.2 * (np.max(age) - t00):
                        ax.annotate('$prediction$', rotation=0, fontsize=15, va='center', ha='center',
                                    xy=(xpre, ytext), xycoords='data',
                                    )
                    xsim = t0_sim + 0.5 * (np.max(age) - t0_sim)
                    if np.max(age) - t0_sim > 0.2 * (np.max(age) - t00):
                        ax.annotate('$simulation$', rotation=0, fontsize=15, va='center', ha='center',
                                    xy=(xsim, ytext), xycoords='data',
                                    )
                except:
                    pass
        for t in [0, t0, t1, t2]:
            if not np.isnan(t):
                try:
                    y0, y1 = ax.get_ylim()
                    ytext = y0 - 0.2 * (y1 - y0)
                    ax.annotate('', xy=(t, y0), xycoords='data',
                                xytext=(t, ytext), textcoords='data',
                                arrowprops=dict(color='black', shrink=0.08, alpha=0.6)
                                )
                except:
                    pass


    T0 = np.nanmean(t0s)
    T1 = np.nanmean(t1s)
    T2 = np.nanmean(t2s)

    fontsize = 20
    y = -0.2
    texts = ['egg', 'hatch', 'pupation', 'death']
    text_xs = [0, T0, T1, T2]
    for text, x in zip(texts, text_xs):
        try:
            y0, y1 = ax.get_ylim()
            ytext = y0 - 0.2 * (y1 - y0)
            ax.annotate(text, xy=(x, y0), xycoords='data', fontsize=fontsize,
                        xytext=(x, ytext), textcoords='data',
                        horizontalalignment='center', verticalalignment='top')
        except:
            pass
    P.conf_ax(j, xlab=f'time $({time_unit})$', xMaxN=5, xlim=(0, np.max(max_ages) if sim_only else None))

    if not sim_only:
        for ax in P.axs:
            ax.set_xticks(ticks=np.arange(0, np.max(max_ages), tickstep))

    plot.dataset_legend(leg_ids, leg_cols, ax=P.axs[0], loc='upper left', fontsize=20, prop={'size': 15})

    P.adjust((0.15, 0.93), (0.15, 0.95), H=0.15)
    return P.get()

@reg.funcs.graph('EEBvsQuality')
def plot_EEB_vs_food_quality(refIDs=None, dt=None,name=None, species_list=['rover', 'sitter', 'default'], **kwargs):
    if refIDs is None:
        raise ('No sample configurations provided')
    from larvaworld.lib.model.modules.intermitter import get_EEB_poly1d
    from larvaworld.lib.model.deb.deb import DEB

    if name is None :
        name=f'EEB_vs_food_quality'
    qs = np.arange(0.01, 1, 0.01)

    P = plot.AutoBasePlot(name=name, build_kws={'Nrows': 3, 'Ncols': len(refIDs), 'w': 10, 'h': 7}, **kwargs)

    for i, refID in enumerate(refIDs):
        kws= reg.getRef(refID)['intermitter']
        z = get_EEB_poly1d(dt=dt, **kws)
        for col, species in zip(aux.N_colors(len(species_list)), species_list):
            ss = []
            EEBs = []
            cc = {'color': col,
                  'label': species,
                  'marker': '.'}
            for q in qs:
                deb = DEB(substrate_quality=q, species=species, **kwargs)
                s = np.round(deb.fr_feed, 2)
                ss.append(s)
                EEBs.append(z(s))

            P.axs[3 * i].scatter(qs, ss, **cc)
            P.axs[3 * i + 1].scatter(qs, EEBs, **cc)
            P.axs[3 * i + 2].scatter(ss, EEBs, **cc)

        P.conf_ax(3 * i + 0, xlab='food quality', ylab=r'estimated feed freq $Hz$')
        P.conf_ax(3 * i + 1, xlab='food quality', ylab='EEB', ylim=[0,1])
        P.conf_ax(3 * i + 2, xlab=r'estimated feed freq $Hz$', ylab='EEB', ylim=[0,1])

    for ax in P.axs:
        ax.legend()
    return P.get()





