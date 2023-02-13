import warnings

import numpy as np
import pandas as pd

from larvaworld.lib.aux import naming as nam

from larvaworld.lib import reg, aux, plot,util



def plot_single_bout(x0, discr, bout, color, label, ax, fit_dic=None, plot_fits='best',print_fits=False,
                     marker='.', legend_outside=False,xlabel = 'time (sec)',xlim=None, **kwargs):
    distro_ls = ['powerlaw', 'exponential', 'lognormal', 'lognorm-pow', 'levy', 'normal', 'uniform']
    distro_cs = ['c', 'g', 'm', 'k', 'orange', 'brown', 'purple']
    num_distros = len(distro_ls)
    lws = [2] * num_distros

    if fit_dic is None:
        xmin, xmax = np.min(x0), np.max(x0)
        fit_dic = util.fit_bout_distros(x0, xmin, xmax, discr, dataset_id='test', bout=bout,print_fits=print_fits, **kwargs)
    idx_Kmax = fit_dic['idx_Kmax']
    xrange, du2, c2, y = fit_dic['values']
    lws[idx_Kmax] = 4
    ddfs = fit_dic['cdfs']
    for ii in ddfs:
        if ii is not None:
            ii /= ii[0]
    ax.loglog(xrange, y, marker, color=color, alpha=0.7, label=label)
    ax.set_title(bout)
    ax.set_xlabel(xlabel)
    ax.set_ylim([10 ** -3.5, 10 ** 0.2])
    if xlim is not None :
        ax.set_xlim(xlim)
    distro_ls0, distro_cs0 = [], []
    for z, (l, col, lw, ddf) in enumerate(zip(distro_ls, distro_cs, lws, ddfs)):
        if ddf is None:
            continue
        if plot_fits == 'best' and z == idx_Kmax:
            cc = color
        elif plot_fits == 'all' or l in plot_fits:
            distro_ls0.append(l)
            distro_cs0.append(col)
            cc = col
        else:
            continue
        ax.loglog(xrange, ddf, color=cc, lw=lw, label=l)
    if len(distro_ls0) > 1:
        if legend_outside:
            plot.dataset_legend(distro_ls0, distro_cs0, ax=ax, loc='center left', fontsize=25, anchor=(1.0, 0.5))
        else:
            plot.dataset_legend(distro_ls0, distro_cs0, ax=ax, loc='lower left', fontsize=15)


@reg.funcs.graph('epochs')
def plot_bouts(name=None, plot_fits='',print_fits=False, turns=False, stridechain_duration=False, legend_outside=False, **kwargs):
    if name is None :
        if not turns:
            name = f'runsNpauses{plot_fits}'
        else:
            name = f'turn_epochs{plot_fits}'
    P = plot.AutoPlot(name=name,build_kws={'Ncols': 2, 'sharey': True, 'wh' : 5}, **kwargs)
    valid_labs = {}
    for j, d in enumerate(P.datasets):
        id = d.id
        v = d.pooled_epochs
        if v is None:
            continue

        kws = {
            'marker': 'o',
            'plot_fits': plot_fits,
            'print_fits': print_fits,
            'label': id,
            'color': d.color,
            'legend_outside': legend_outside,
            # 'axs': P.axs,
            'x0': None
        }

        if not turns:
            if 'pause_dur' in v.keys() and v.pause_dur is not None:
                plot_single_bout(fit_dic=v.pause_dur, discr=False, bout='pauses', ax=P.axs[1], **kws)
                valid_labs[id] = kws['color']
            if stridechain_duration and 'run_dur' in v.keys() and v.run_dur is not None:
                plot_single_bout(fit_dic=v.run_dur, discr=False, bout='runs', ax=P.axs[0], **kws)
                valid_labs[id] = kws['color']
            elif not stridechain_duration and 'run_count' in v.keys() and v.run_count is not None:
                plot_single_bout(fit_dic=v.run_count, discr=True, bout='stridechains', xlabel='# strides', ax=P.axs[0], **kws)
                valid_labs[id] = kws['color']
        else:
            if 'turn_dur' in v.keys() and v.turn_dur is not None:
                plot_single_bout(fit_dic=v.turn_dur, discr=False, bout='turn duration', ax=P.axs[0], **kws)
                valid_labs[id] = kws['color']
            if 'turn_amp' in v.keys() and v.turn_amp is not None:
                plot_single_bout(fit_dic=v.turn_amp, discr=False, bout='turn amplitude', xlabel='angle (deg)',
                                 xlim=(10 ** -0.5, 10 ** 3), ax=P.axs[1], **kws)
                valid_labs[id] = kws['color']
    P.axs[0].set_ylabel('probability')
    P.axs[1].yaxis.set_visible(False)
    if P.Ndatasets > 1:
        P.data_leg(0, labels=valid_labs.keys(), colors=valid_labs.values(), loc='lower left', fontsize=15)
        # dataset_legend(valid_labs.keys(), valid_labs.values(), ax=P.axs[0], loc='lower left', fontsize=15)
        # dataset_legend(valid_labs.keys(), valid_labs.values(), ax=P.axs[0], loc='lower left', fontsize=15)
    P.adjust((0.15, 0.95), (0.15, 0.92), 0.05, 0.005)
    return P.get()

@reg.funcs.graph('runs & pauses')
def plot_stridesNpauses(name=None, stridechain_duration=False, time_unit='sec',
                        plot_fits='all', range='default', print_fits=False, only_fit_one=True, mode='cdf',
                        subfolder='bouts', refit_distros=False, test_detection=False, **kwargs):
    warnings.filterwarnings('ignore')
    if name is None:
        nn = f'stridesNpauses_{mode}_{range}_{plot_fits}'
        name = nn if not only_fit_one else f'{nn}_0'
    P = plot.AutoPlot(name=name, subfolder=subfolder,build_kws={'Ncols': 2, 'sharey': True, 'wh' : 5}, **kwargs)
    pause_par = nam.dur('pause')
    if stridechain_duration:
        chain_par = nam.dur('exec')
        chn_discr = False
        chain_xlabel = f'time $({time_unit})$'
        chn0 = 0.5
        chn1 = 50
        chn_t0, chn_t1 = 0, 10 ** 2
    else:
        chain_par = nam.length(nam.chain('stride'))
        chn_discr = True
        chain_xlabel = '# chained strides'
        chn0 = 1
        chn1 = 100
        chn_t0, chn_t1 = 10 ** 0, 10 ** 2.5

    pau_discr = False
    pau0 = 0.4
    pau1 = 20.0
    pau_t0, pau_t1 = 0, 10 ** 1.4
    pause_xlabel = f'time $({time_unit})$'

    pau_durs = []
    chn_durs = []

    frs = []
    for label, dataset in zip(P.labels, P.datasets):
        frs.append(dataset.fr)

        pau_dur = dataset.get_par(pause_par).dropna().values
        chn_dur = dataset.get_par(chain_par).dropna().values
        if time_unit == 'ms':
            pau_dur *= 1000
            pau0 *= 1000
            pau1 *= 1000
            pau_t0 *= 1000
            pau_t1 *= 1000
            if stridechain_duration:
                chn_dur *= 1000
                chn0 *= 1000
                chn1 *= 1000
                chn_t0 *= 1000
                chn_t1 *= 1000
        pau_durs.append(pau_dur)
        chn_durs.append(chn_dur)

    if test_detection:
        for l, d, col in zip(P.labels, P.datasets, P.colors):
            dic0 = d.load_dicts('bout_dicts')
            dic = {}
            for iid, ddd in dic0.items():
                df = pd.DataFrame.from_dict(ddd)
                df.index.set_names(0, inplace=True)
                dic[iid] = df

            pau_dur = np.array(aux.flatten_list([ddic[pause_par] for ddic in dic.values()]))
            chn_dur = np.array(aux.flatten_list([ddic[chain_par] for ddic in dic.values()]))
            pau_durs.append(pau_dur)
            chn_durs.append(chn_dur)
            P.labels.append(f'{l} truth')
            frs.append(d.fr)
            P.colors.append(f'dark{col}')

    min_pauses, max_pauses = [np.min(dur) for dur in pau_durs], [np.max(dur) for dur in pau_durs]
    min_chains, max_chains = [np.min(dur) for dur in chn_durs], [np.max(dur) for dur in chn_durs]

    if range == 'broad':
        pau0, pau1 = np.min(min_pauses), np.max(max_pauses)
        chn0, chn1 = np.min(min_chains), np.max(max_chains)
    elif range == 'restricted':
        pau0, pau1 = np.max(min_pauses), np.min(max_pauses)
        chn0, chn1 = np.max(min_chains), np.min(max_chains)
    elif range == 'default':
        pass
    fits = {l: {} for l in P.labels}

    distro_ls = ['powerlaw', 'exponential', 'lognormal', 'lognorm-pow', 'levy', 'normal', 'uniform']
    distro_cs = ['c', 'g', 'm', 'k', 'yellow', 'brown', 'purple']
    num_distros = len(distro_ls)

    for j, (pau_dur, chn_dur, c, label, fr) in enumerate(zip(pau_durs, chn_durs, P.colors, P.labels, frs)):
        try:
            ref = reg.loadConf(id=label, conftype='Ref')
        except:
            ref = None
        for i, (x0, discr, xmin, xmax) in enumerate(
                zip([chn_dur, pau_dur], [chn_discr, pau_discr], [chn0, pau0], [chn1, pau1])):
            bout = 'stride' if i == 0 else 'pause'
            lws = [2] * num_distros

            if not refit_distros and ref is not None:
                u2, du2, c2, c2cum = util.compute_density(x0, xmin, xmax)
                b = util.BoutGenerator(**ref[bout]['best'])
                pdfs = [b.get(x=du2, mode='pdf')] * num_distros
                cdfs = [1 - b.get(x=u2, mode='cdf')] * num_distros
                idx_Kmax = 0

            else:
                fit_dic = util.fit_bout_distros(x0, xmin, xmax, discr, dataset_id=label, bout=bout,
                                           print_fits=print_fits, combine=False)
                idx_Kmax = fit_dic['idx_Kmax']
                cdfs = fit_dic['cdfs']
                pdfs = fit_dic['pdfs']
                u2, du2, c2, c2cum = fit_dic['values']
                lws[idx_Kmax] = 4
                fits[label].update(fit_dic['res_dict'])
            if mode == 'cdf':
                ylabel = 'cumulative probability'
                xrange = u2
                y = c2cum
                ddfs = cdfs
                for ii in ddfs:
                    if ii is not None:
                        ii /= ii[0]

            elif mode == 'pdf':
                ylabel = 'probability'
                xrange = du2
                y = c2
                ddfs = pdfs
                for ii in ddfs:
                    if ii is not None:
                        ii /= sum(ii)

            P.axs[i].loglog(xrange, y, '.', color=c, alpha=0.7)
            for z, (l, col, lw, ddf) in enumerate(zip(distro_ls, distro_cs, lws, ddfs)):
                if ddf is None:
                    continue
                if plot_fits == 'best' and z == idx_Kmax:
                    cc = c
                elif plot_fits == 'all':
                    cc = col
                else:
                    continue
                P.axs[i].loglog(xrange, ddf, color=cc, lw=lw, label=l)

    for ii in [0, 1]:
        if plot_fits in ['all']:
            P.data_leg(ii,labels=distro_ls, colors=distro_cs, loc='lower left', fontsize=15)
        P.data_leg(ii,loc='upper right', fontsize=15)
    P.conf_ax(0, xlab=chain_xlabel, ylab=ylabel, xlim=[chn_t0, chn_t1], title=r'$\bf{stridechains}$')
    P.conf_ax(1, xlab=pause_xlabel, xlim=[pau_t0, pau_t1], ylim=[10 ** -3.5, 10 ** 0], title=r'$\bf{pauses}$')
    P.adjust((0.15, 0.95), (0.15, 0.92), 0.05, 0.005)
    fit_df = pd.DataFrame.from_dict(fits, orient="index")
    fit_df.to_csv(P.fit_filename, index=True, header=True)
    return P.get()




