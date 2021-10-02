import os
import warnings
import pandas as pd
import numpy as np
import scipy as sp
from fitter import Fitter
from matplotlib import pyplot as plt
import scipy.stats as st
from scipy import stats as st
from scipy.stats import ks_2samp, stats, levy, norm, uniform

import lib.anal.process.aux
from lib.aux import naming as nam
from lib.aux import colsNstr as fun
from lib.conf.conf import saveConf



def fit_angular_params(d, fit_filepath=None, chunk_only=None, absolute=False,
                       save_to=None, save_as=None):
    if save_to is None:
        save_to = d.plot_dir
    if save_as is None:
        if absolute:
            save_as = 'angular_fit_abs.pdf'
        else:
            save_as = 'angular_fit.pdf'
    filepath = os.path.join(save_to, save_as)
    if chunk_only is not None:
        s = d.step_data.loc[d.step_data[nam.id(chunk_only)].dropna().index]
    else:
        s = d.step_data
    # sigma = d.step
    # TURNER PARAMETERS
    # -----------------------

    # The bend of the front-body (sum of half the angles) is the target for the turner calibration
    # The head reorientation velocity is the result of the turner calibration
    # if mode == 'experiment':
    #     # point = d.critical_spinepoint
    #
    #     ho = d.unwrapped_flag(d.orientation_flag(d.segments[0]))
    #     # act_r = 'non_rest_dur_fraction'
    #     # dst = d.dst_param(d.critical_spinepoint)
    # elif mode == 'simulation':
    #     # point = 'centroid'
    #     ho = d.orientation_flag('head')
    # b = 'front_half_bend'
    b = 'bend'
    bv = nam.vel(b)
    ba = nam.acc(b)
    # hb=d.angles[0]
    # hbv=d.vel_param(hb)
    # hba=d.acc_param(hb)
    # ho = d.orientation_flag('head')
    ho = 'front_orientation'
    hov = nam.vel(ho)
    hoa = nam.acc(ho)
    # hca = f'turn_{ho}'
    hca = f'turn_{nam.unwrap(ho)}'
    pars = [b, bv, ba, hov, hoa, hca]
    ranges = [150, 400, 5000, 400, 5000, 100]

    if fit_filepath is None:
        # These are the fits of a 100 larvae dataset
        fitted_distros = [{'t': (2.36, 0, 13)},
                          {'t': (1.71, 0, 32.1)},
                          {'t': (1.94, 0, 331)},
                          {'t': (1.54, 0, 26.6)},
                          {'t': (1.87, 0, 300)},
                          {'t': (0.965, 0, 5.33)}]
        target_stats = [0.006, 0.01, 0.01, 0.005, 0.005, 0.026]
    else:
        pars, fitted_distros, target_stats = d.load_fits(filepath=fit_filepath, selected_pars=pars)
    nbins = 500
    height = 0.05
    colors = ['g', 'r', 'b', 'r', 'b', 'g']
    labels = [r'$\theta_{b}$', r'$\dot{\theta}_{b}$', r'$\ddot{\theta}_{b}$', r'$\dot{\theta}_{or}$',
              r'$\ddot{\theta}_{or}$', r'$\theta_{turn}$']
    xlabels = ['angle $(deg)$', 'angular velocity $(deg/sec)$', 'angular acceleration, $(deg^2/sec)$',
               'angular velocity $(deg/sec)$', 'angular acceleration, $(deg^2/sec)$', 'angle $(deg)$']
    order = [2, 1, 0, 4, 3, 5]
    fits = []
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharey=True)
    axs = axs.ravel()
    for i, (par, w, r, l, xl, j, col, target) in enumerate(
            zip(pars, fitted_distros, ranges, labels, xlabels, order, colors, target_stats)):
        x = np.linspace(-r, r, nbins)
        data = s[par].dropna().values
        if absolute:
            data = np.abs(data)
        weights = np.ones_like(data) / float(len(data))
        axs[j].hist(data, bins=x, weights=weights, label=l, color=col, alpha=0.8)
        dist_name = list(w.keys())[0]
        dist_args = list(w.values())[0]
        dist = getattr(st, dist_name)

        if dist.shapes is None:
            dist_args_dict = dict(zip(['loc', 'scale'], dist_args))
        else:
            dist_args_dict = dict(zip([dist.shapes] + ['loc', 'scale'], dist_args))
        stat, pvalue = sp.stats.kstest(data, dist_name, args=dist_args)
        fits.append([par, stat, pvalue])
        print(f'Parameter {par} was fitted with stat : {stat} vs target stat : {target}')
        y = dist.rvs(size=100000, **dist_args_dict)
        n_weights = np.ones_like(y) / float(len(y))
        my_n, my_bins, my_patches = axs[j].hist(y, bins=x, weights=n_weights, alpha=0)
        axs[j].scatter(my_bins[:-1] + 0.5 * (my_bins[1:] - my_bins[:-1]), my_n, marker='.', c='k', s=40, alpha=0.6)
        axs[j].plot(my_bins[:-1] + 0.5 * (my_bins[1:] - my_bins[:-1]), my_n, c='k', linewidth=2, alpha=0.6)
        axs[j].legend(loc='upper right', fontsize=15)
        axs[j].set_xlabel(xl, fontsize=15)

        axs[j].set_ylim([0, height])
        if absolute:
            axs[j].set_xlim([0, r])
        else:
            axs[j].set_xlim([-r, r])
    axs[0].set_ylabel('probability, $P$', fontsize=15)
    axs[3].set_ylabel('probability, $P$', fontsize=15)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.3)
    plt.savefig(filepath, dpi=300)
    print(f'Image saved as {filepath} !')
    return fits


def fit_endpoint_params(d, fit_filepath=None, save_to=None, save_as='endpoint_fit.pdf'):
    if save_to is None:
        save_to = d.plot_dir
    filepath = os.path.join(save_to, save_as)
    e = d.endpoint_data
    point = d.point
    dst = 'distance'
    v = 'velocity'
    a = 'acceleration'
    sv = nam.scal(v)
    sa = nam.scal(a)
    cum_dst = nam.cum(dst)
    cum_sdst = nam.cum(nam.scal(dst))

    # Stride related parameters. The scal-distance-per-stride is the result of the crawler calibration
    stride_flag = nam.max(sv)
    stride_d = nam.dst('stride')
    stride_sd = nam.scal(stride_d)
    f_crawler = nam.freq(sv)
    Nstrides = nam.num('stride')
    stride_ratio = nam.dur_ratio('stride')
    l = 'length'
    pars = [l, f_crawler,
            nam.mean(stride_d), nam.mean(stride_sd),
            Nstrides, stride_ratio,
            cum_dst, cum_sdst,
            nam.max('40sec_dispersion'), nam.scal(nam.max('40sec_dispersion')),
            nam.final('40sec_dispersion'), nam.scal(nam.final('40sec_dispersion')),
            'stride_reoccurence_rate', 'stride_reoccurence_rate',
            nam.mean('bend'), nam.mean(nam.vel('bend'))]
    ranges = [(2, 6), (0.6, 2.25),
              (0.4, 1.6), (0.1, 0.35),
              (100, 300), (0.3, 1.0),
              (0, 360), (0, 80),
              (0, 70), (0, 20),
              (0, 70), (0, 20),
              (0.5, 1.0), (0.5, 1.0),
              (-20.0, 20.0), (-8.0, 8.0)]

    if fit_filepath is None:
        # These are the fits of a 100 larvae dataset
        fitted_distros = [{'norm': (4.57, 0.54)},
                          {'norm': (1.44, 0.203)},
                          {'norm': (1.081, 0.128)},
                          {'norm': (0.239, 0.031)},
                          {'norm': (249.8, 36.4)},
                          {'norm': (55.1, 7.9)},
                          {'norm': (43.3, 16.2)},
                          {'norm': (9.65, 3.79)}, ]
        target_stats = [0.074, 0.087, 0.086, 0.084, 0.057, 0.048, 0.068, 0.094]
    else:
        pars, fitted_distros, target_stats = d.load_fits(filepath=fit_filepath, selected_pars=pars)
    fits = []

    labels = ['body length', 'stride frequency',
              'stride displacement', 'scal stride displacement',
              'num strides', 'crawling ratio',
              'displacement in 3 min', 'scal displacement in 3 min',
              'max dispersion in 40 sec', 'max scal dispersion in 40 sec',
              'dispersion in 40 sec', 'scal dispersion in 40 sec',
              'stride reoccurence rate', 'stride reoccurence rate',
              'mean bend angle', 'mean bend velocity']
    xlabels = ['length $(mm)$', 'frequency $(Hz)$',
               'distance $(mm)$', 'scal distance $(-)$',
               'counts $(-)$', 'time ratio $(-)$',
               'distance $(mm)$', 'scal distance $(-)$',
               'distance $(mm)$', 'scal distance $(-)$',
               'distance $(mm)$', 'scal distance $(-)$',
               'rate $(-)$', 'rate $(-)$',
               'angle $(deg)$', 'angular velocity $(deg/sec)$']
    nbins = 20
    height = 0.3
    fig, axs = plt.subplots(int(len(pars) / 2), 2, figsize=(15, int(5 * len(pars) / 2)), sharey=True)
    axs = axs.ravel()
    for i, (par, lab, xl, (rmin, rmax), w, target) in enumerate(
            zip(pars, labels, xlabels, ranges, fitted_distros, target_stats)):
        data = e[par].dropna().values
        x = np.linspace(rmin, rmax, nbins)
        loc, scale = list(w.values())[0]
        stat, pvalue = sp.stats.kstest(data, list(w.keys())[0], args=list(w.values())[0])
        fits.append([par, stat, pvalue])
        print(f'Parameter {par} was fitted with stat : {stat} vs target stat : {target}')
        y = sp.norm.rvs(size=10000, loc=loc, scale=scale)
        n_weights = np.ones_like(y) / float(len(y))
        my_n, my_bins, my_patches = axs[i].hist(y, bins=x, weights=n_weights, alpha=0)
        axs[i].scatter(my_bins[:-1] + 0.5 * (my_bins[1:] - my_bins[:-1]), my_n, marker='o', c='k', s=40, alpha=0.6)
        axs[i].plot(my_bins[:-1] + 0.5 * (my_bins[1:] - my_bins[:-1]), my_n, alpha=0.6, c='k', linewidth=2,
                    label='norm fit')

        weights = np.ones_like(data) / float(len(data))
        axs[i].hist(data, bins=x, weights=weights, label=lab, color='b', alpha=0.6)
        axs[i].legend(loc='upper right', fontsize=12)
        axs[i].set_xlabel(xl, fontsize=12)
        axs[i].set_ylim([0, height])
    axs[0].set_ylabel('probability, $P$', fontsize=15)
    axs[2].set_ylabel('probability, $P$', fontsize=15)
    axs[4].set_ylabel('probability, $P$', fontsize=15)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.3)
    plt.savefig(filepath, dpi=300)
    print(f'Image saved as {filepath} !')
    return fits


def fit_bout_params(d, fit_filepath=None, save_to=None, save_as='bout_fit.pdf'):
    if save_to is None:
        save_to = os.path.join(d.plot_dir, 'plot_bouts')
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    filepath = os.path.join(save_to, save_as)
    e = d.endpoint_data
    s = d.step_data
    pars = [nam.chain_counts_par('stride'), nam.dur(nam.non('stride')),
            nam.dur_ratio('stride'), nam.dur_ratio('non_stride'),
            nam.num('stride'), nam.num('non_stride'),
            nam.dur('rest'), nam.dur('activity'),
            nam.dur_ratio('rest'), nam.dur_ratio('activity'),
            nam.num('rest'), nam.num('activity')]
    ranges = [(1.0, 40.0), (0.0, 10.0),
              (0.0, 1.0), (0, 1.0),
              (0, 300), (0, 120),
              (0, 3), (0, 60),
              (0.0, 1.0), (0.0, 1.0),
              (0, 100), (0, 100)]

    if fit_filepath is None:
        # These are the fits of a 100 larvae dataset
        raise ValueError('Not implemented. Please provide fit file')
    else:
        pars, fitted_distros, target_stats = d.load_fits(filepath=fit_filepath, selected_pars=pars)
    fits = []

    labels = ['stride chains', 'stride-free bouts',
              'stride ratio', 'stride-free ratio',
              'num strides', 'num non-strides',
              'rest bouts', 'activity bouts',
              'rest ratio', 'activity ratio',
              'num rests', 'num activities']
    xlabels = ['length $(-)$', 'time $(sec)$',
               'time fraction $(-)$', 'time fraction $(-)$',
               'counts $(-)$', 'counts $(-)$',
               'time $(sec)$', 'time $(sec)$',
               'time fraction $(-)$', 'time fraction $(-)$',
               'counts $(-)$', 'counts $(-)$']
    nbins = 30
    height = 0.4
    fig, axs = plt.subplots(6, 2, figsize=(15, 30), sharey=True)
    axs = axs.ravel()
    for i, (par, lab, xl, (rmin, rmax), w, target) in enumerate(
            zip(pars, labels, xlabels, ranges, fitted_distros, target_stats)):
        print(par)
        try:
            data = e[par].dropna().values
        except:
            data = s[par].dropna().values
        x = np.linspace(rmin, rmax, nbins)
        args = list(w.values())[0]
        name = list(w.keys())[0]
        stat, pvalue = sp.stats.kstest(data, name, args=args)
        fits.append([par, stat, pvalue])
        print(f'Parameter {par} was fitted with stat : {stat} vs target stat : {target}')
        distr = getattr(sp.stats.distributions, name)
        y = distr.rvs(*args, size=10000)
        n_weights = np.ones_like(y) / float(len(y))
        my_n, my_bins, my_patches = axs[i].hist(y, bins=x, weights=n_weights, alpha=0)
        axs[i].scatter(my_bins[:-1] + 0.5 * (my_bins[1:] - my_bins[:-1]), my_n, marker='o', c='k', s=40, alpha=0.6)
        axs[i].plot(my_bins[:-1] + 0.5 * (my_bins[1:] - my_bins[:-1]), my_n, alpha=0.6, c='k', linewidth=2,
                    label=f'{name} fit')
        weights = np.ones_like(data) / float(len(data))
        axs[i].hist(data, bins=x, weights=weights, label=lab, color='b', alpha=0.6)
        axs[i].legend(loc='upper right', fontsize=12)
        axs[i].set_xlabel(xl, fontsize=12)
        axs[i].set_ylim([0, height])
    axs[0].set_ylabel('probability, $P$', fontsize=15)
    axs[2].set_ylabel('probability, $P$', fontsize=15)
    axs[4].set_ylabel('probability, $P$', fontsize=15)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.01, hspace=0.3)
    plt.savefig(filepath, dpi=300)
    print(f'Image saved as {filepath} !')
    return fits


def fit_crawl_params(d, target_point=None, fit_filepath=None, save_to=None, save_as='crawl_fit.pdf'):
    if save_to is None:
        save_to = d.plot_dir
    filepath = os.path.join(save_to, save_as)
    e = d.endpoint_data
    if target_point is None:
        target_point = d.point
    point = d.point
    exp_dst = 'distance'
    dst = 'distance'
    exp_cum_sdst = nam.cum(nam.scal(exp_dst))
    cum_sdst = nam.cum(nam.scal(dst))
    Nstrides = nam.num('stride')
    stride_ratio = nam.dur_ratio('stride')
    dispersion = nam.scal(nam.final('40sec_dispersion'))

    exp_pars = [Nstrides, stride_ratio, exp_cum_sdst]
    pars = [Nstrides, stride_ratio, cum_sdst]
    ranges = [(100, 300), (0.5, 1.0), (20, 80)]
    exp_pars, fitted_distros, target_stats = d.load_fits(filepath=fit_filepath, selected_pars=exp_pars)
    fits = []
    labels = ['$N_{strides}$', 'crawling ratio',
              '$distance_{scal}$']
    xlabels = ['counts $(-)$', 'time ratio $(-)$',
               'scal distance $(-)$']
    colors = ['r', 'c', 'g']
    nbins = 20
    height = 0.3
    fig, axs = plt.subplots(int(len(pars) / 3), 3, figsize=(15, int(5 * len(pars) / 3)), sharey=True)
    axs = axs.ravel()
    for i, (par, lab, xl, (rmin, rmax), w, target, c) in enumerate(
            zip(pars, labels, xlabels, ranges, fitted_distros, target_stats, colors)):
        data = e[par].dropna().values
        x = np.linspace(rmin, rmax, nbins)
        loc, scale = list(w.values())[0]
        stat, pvalue = sp.stats.kstest(data, list(w.keys())[0], args=list(w.values())[0])
        fits.append([par, stat, pvalue])
        print(f'Parameter {par} was fitted with stat : {stat} vs target stat : {target}')
        y = sp.norm.rvs(size=10000, loc=loc, scale=scale)
        n_weights = np.ones_like(y) / float(len(y))
        my_n, my_bins, my_patches = axs[i].hist(y, bins=x, weights=n_weights, alpha=0)
        axs[i].scatter(my_bins[:-1] + 0.5 * (my_bins[1:] - my_bins[:-1]), my_n, marker='o', c='k', s=40, alpha=0.6)
        axs[i].plot(my_bins[:-1] + 0.5 * (my_bins[1:] - my_bins[:-1]), my_n, alpha=0.6, c='k', linewidth=2)
        weights = np.ones_like(data) / float(len(data))
        axs[i].hist(data, bins=x, weights=weights, label=lab, color=c, alpha=0.6)
        axs[i].legend(loc='upper right', fontsize=12)
        axs[i].set_xlabel(xl, fontsize=12)
        axs[i].set_ylim([0, height])
    axs[0].set_ylabel('probability, $P$', fontsize=15)
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.95, wspace=0.01, hspace=0.3)
    plt.savefig(filepath, dpi=300)
    print(f'Image saved as {filepath} !')
    return fits


def powerlaw_cdf(x, xmin, alpha):
    cum = (x / xmin) ** (1 - alpha)
    # print(cum)
    return 1 - cum


def powerlaw_pdf(x, xmin, alpha):
    res = (alpha - 1) / xmin * (x / xmin) ** (-alpha)
    # res/=sum(res)
    return res


def levy_pdf(x, mu, sigma):
    res = np.sqrt(sigma / (2 * np.pi)) * np.exp(-sigma / (2 * (x - mu))) / (x - mu) ** 1.5
    # print(x,res)
    return res


def levy_cdf(x, mu, sigma):
    res = 1 - sp.special.erf(np.sqrt(sigma / (2 * (x - mu))))
    if np.isnan(res[0]):
        res[0] = 0
    return res


def norm_pdf(x, mu, sigma):
    res = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return res


def norm_cdf(x, mu, sigma):
    res = 0.5 * (1 + sp.special.erf((x - mu) / (sigma * np.sqrt(2))))
    if np.isnan(res[0]):
        res[0] = 0
    return res


def uniform_pdf(x, xmin, xmax):
    return uniform.pdf(x, xmin, xmin + xmax)


def uniform_cdf(x, xmin, xmax):
    return uniform.cdf(x, xmin, xmin + xmax)


def exponential_cdf(x, xmin, beta):
    return 1 - np.exp(-beta * (x - xmin))


def exponential_pdf(x, xmin, beta):
    return beta * np.exp(-beta * (x - xmin))


def lognorm_cdf(x, mu, sigma, xmin=0):
    cdf = 0.5 + 0.5 * sp.special.erf((np.log(x) - mu) / np.sqrt(2) / sigma)
    # print(cdf)
    return cdf


def lognormal_pdf(x, mu, sigma, xmin=0):
    pdf = 1 / (x * sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)
    # pdf/=sum(pdf)
    return pdf


def logNpow_pdf(x, mu, sigma, alpha, xmin, switch, ratio, overlap=0):
    x0 = x[x < switch]
    x1 = x[x >= switch]
    log_pdf = lognormal_pdf(x0, mu, sigma, xmin) * ratio
    # log_pdf = log_pdf/np.sum(log_pdf)*ratio
    pow_pdf = powerlaw_pdf(x1, switch, alpha) * (1 - ratio)
    # pow_pdf = pow_pdf/np.sum(pow_pdf) * (1 - ratio)
    pdf = np.hstack([log_pdf, pow_pdf])
    # pdf/=sum(pdf)
    return pdf


def logNpow_cdf2(x, mu, sigma, alpha, xmin, switch, ratio, overlap=0):
    pdf = logNpow_pdf(x, mu, sigma, alpha, xmin, switch, ratio, overlap)
    cdf = np.cumsum(pdf)
    # print(cdf)

    return cdf


def logNpow_cdf(x, mu, sigma, alpha, xmin, switch, ratio, overlap=0):
    x0 = x[x < switch]
    x1 = x[x >= switch]
    N0 = x0.shape[0]
    log_cdf = 1 - lognorm_cdf(x0, mu, sigma)
    pow_cdf = 1 - powerlaw_cdf(x1, switch, alpha)
    # log_cdf *= ratio
    pow_cdf *= (1 - ratio)
    cdf0 = np.hstack([log_cdf, pow_cdf])
    cdf = 1 - cdf0
    return cdf


def get_distro(name, x, range, mode='cdf', **kwargs):
    ddfs = {
        'powerlaw': {'cdf': powerlaw_cdf, 'pdf': powerlaw_pdf, 'args': ['alpha'], 'rvs': 'trunc_powerlaw'},
        'exponential': {'cdf': exponential_cdf, 'pdf': exponential_pdf, 'args': ['beta'], 'rvs': ''},
        'lognormal': {'cdf': lognorm_cdf, 'pdf': lognormal_pdf, 'args': ['mu', 'sigma'], 'rvs': ''},
        'logNpow': {'cdf': logNpow_cdf, 'pdf': logNpow_pdf,
                    'args': ['alpha', 'mu', 'sigma', 'switch', 'ratio', 'overlap'], 'rvs': 'logNpow_distro'},
        'levy': {'cdf': levy_cdf, 'pdf': levy_pdf, 'args': ['mu', 'sigma'], 'rvs': ''},
        'norm': {'cdf': norm_cdf, 'pdf': norm_pdf, 'args': ['mu', 'sigma'], 'rvs': ''}
    }
    xmin, xmax = range
    func = ddfs[name][mode]
    args = ddfs[name]['args']
    return func(x=x, xmin=xmin, **{a: kwargs[a] for a in args})


def get_logNpow(dur, dur0, dur1, durmid, fr, overlap=0, discrete=False):
    d0 = dur[dur < durmid]
    d1 = dur[dur >= durmid]
    r = len(d0) / len(dur)
    d00 = dur[dur < durmid + overlap * (dur1 - durmid)]
    m, s = get_lognormal(d00, dur0)
    a = get_powerlaw_alpha(d1, durmid, dur1, discrete=discrete)
    return m, s, a, r


def get_powerlaw_alpha(dur, dur0, dur1, discrete=False):
    with lib.anal.process.aux.suppress_stdout_stderr():
        from powerlaw import Fit
        results = Fit(dur, xmin=dur0, xmax=dur1, discrete=discrete)
        alpha = results.power_law.alpha
        return alpha


def get_lognormal(dur, xmin):
    m = np.mean(np.log(dur))
    s = np.std(np.log(dur))
    return m, s


def compute_density(x, xmin, xmax, Nbins=64):
    log_range = np.linspace(np.log2(xmin), np.log2(xmax), Nbins)
    bins = np.unique((2 * 2 ** (log_range)) / 2)
    x_filt = x[x >= xmin]
    x_filt = x_filt[x_filt <= xmax]
    cdf = np.ones(len(bins))
    pdf = np.zeros(len(bins) - 1)
    for i in range(len(bins)):
        cdf[i] = 1 - np.mean(x_filt < bins[i])
        if i >= 1:
            pdf[i - 1] = -(cdf[i] - cdf[i - 1]) / (bins[i] - bins[i - 1])
    bins1 = 0.5 * (bins[:-1] + bins[1:])
    return bins, bins1, pdf, cdf


def KS(a1, a2):
    return np.max(np.abs(a1 - a2))


def MSE(a1, a2, scaled=False):
    # e=10**-5
    # s=a1+1 if scaled else 1
    if scaled:
        s1 = sum(a1)
        s2 = sum(a2)
    else:
        s1, s2 = 1, 1
    return np.sum((a1 / s1 - a2 / s2) ** 2) / a1.shape[0]


def logNpow_switch(x, xmin, xmax, u2, du2, c2cum, c2, fr, discrete=False, fit_by='cdf'):
    xmids = u2[1:-int(len(u2) / 3)][::2]
    overlaps = np.linspace(0, 1, 6)
    # overlaps = [0]
    temp = np.ones([len(xmids), len(overlaps)])
    for i, xmid in enumerate(xmids):
        for j, ov in enumerate(overlaps):
            mm, ss, aa, r = get_logNpow(x, xmin, xmax, xmid, fr, discrete=discrete, overlap=ov)
            lp_cdf = 1 - logNpow_cdf(u2, mm, ss, aa, xmin, xmid, r)
            lp_pdf = logNpow_pdf(du2, mm, ss, aa, xmin, xmid, r)
            if fit_by == 'cdf':
                temp[i, j] = MSE(c2cum, lp_cdf)
            elif fit_by == 'pdf':
                temp[i, j] = MSE(c2, lp_pdf)

    if all(np.isnan(temp.flatten())):
        return np.nan
    else:
        ii, jj = np.unravel_index(np.nanargmin(temp), temp.shape)
        xmid = xmids[ii]
        ov = overlaps[jj]
        return xmid, ov


def fit_bouts(config, dataset=None, s=None, e=None, id=None, store=False, bouts=['stride', 'pause'], **kwargs):
    from lib.model.modules.intermitter import get_EEB_poly1d
    if id is None:
        id = config['id']
    config['bout_distros'] = {}
    dic = {}
    for bout, p, disc, comb, (xmin, xmax) in zip(bouts,
                                                 ['stridechain_length', 'pause_dur'],
                                                 [True, False],
                                                 [False, True],
                                                 [(1, 100), (0.4, 20.0)]):
        if dataset is not None:
            x0 = dataset.get_par(p).values
        elif s is not None:
            x0 = s[p].dropna().values
        dic = fit_bout_distros(x0, xmin, xmax, fr=config['fr'], discrete=disc, print_fits=True,
                               dataset_id=id, bout=bout, combine=comb, store=store)
        config['bout_distros'][bout] = dic['best'][bout]['best']

    config['intermitter'] = {
        nam.freq('crawl'): e[nam.freq(nam.scal(nam.vel('')))].mean(),
        nam.freq('feed'): e[nam.freq('feed')].mean() if nam.freq('feed') in e.columns else 2.0,
        'dt': config['dt'],
        'crawl_bouts': True,
        'feed_bouts': True,
        'stridechain_dist': config['bout_distros']['stride'],
        'pause_dist': config['bout_distros']['pause'],
        'feeder_reoccurence_rate': None,
    }
    config['EEB_poly1d'] = get_EEB_poly1d(**config['intermitter']).c.tolist()

    return dic


def fit_bout_distros(x0, xmin, xmax, fr, discrete=False, xmid=np.nan, overlap=0.0, Nbins=64, print_fits=True,
                     dataset_id='dataset', bout='pause', combine=True, store=False, fit_by='cdf'):
    with lib.anal.process.aux.suppress_stdout(True):
        warnings.filterwarnings('ignore')
        x = x0[x0 >= xmin]
        x = x[x <= xmax]

        u2, du2, c2, c2cum = compute_density(x, xmin, xmax, Nbins=Nbins)
        values = [u2, du2, c2, c2cum]

        a2 = 1 + len(x) / np.sum(np.log(x / xmin))
        a = get_powerlaw_alpha(x, xmin, xmax, discrete=discrete)
        p_cdf = 1 - powerlaw_cdf(u2, xmin, a)
        p_pdf = powerlaw_pdf(du2, xmin, a)

        b = len(x) / np.sum(x - xmin)
        e_cdf = 1 - exponential_cdf(u2, xmin, b)
        e_pdf = exponential_pdf(du2, xmin, b)

        m, s = get_lognormal(x, xmin)
        l_cdf = 1 - lognorm_cdf(u2, m, s, xmin)
        l_pdf = lognormal_pdf(du2, m, s, xmin)

        m_lev, s_lev = levy.fit(x)
        lev_cdf = 1 - levy_cdf(u2, m_lev, s_lev)
        lev_pdf = levy_pdf(du2, m_lev, s_lev)

        m_nor, s_nor = norm.fit(x)
        nor_cdf = 1 - norm_cdf(u2, m_nor, s_nor)
        nor_pdf = norm_pdf(du2, m_nor, s_nor)

        # m_nor, s_nor = norm.fit(x)
        uni_cdf = 1 - uniform_cdf(u2, xmin, xmin + xmax)
        uni_pdf = uniform_pdf(du2, xmin, xmin + xmax)

        if np.isnan(xmid) and combine:
            xmid, overlap = logNpow_switch(x, xmin, xmax, u2, du2, c2cum, c2, fr, discrete, fit_by)

        if not np.isnan(xmid):
            mm, ss, aa, r = get_logNpow(x, xmin, xmax, xmid, fr, discrete=discrete, overlap=overlap)
            lp_cdf = 1 - logNpow_cdf(u2, mm, ss, aa, xmin, xmid, r)
            lp_pdf = logNpow_pdf(du2, mm, ss, aa, xmin, xmid, r)
            # lp_st, lp_pv = ks_2samp(c2cum, lp_cdf)
        else:
            mm, ss, aa, r = np.nan, np.nan, np.nan, np.nan
            lp_cdf, lp_pdf = None, None
            # lp_st, lp_pv = np.nan, np.nan

        if fit_by == 'cdf':
            KS_pow = MSE(c2cum, p_cdf)
            KS_exp = MSE(c2cum, e_cdf)
            KS_logn = MSE(c2cum, l_cdf)
            KS_lognNpow = MSE(c2cum, lp_cdf) if lp_cdf is not None else np.nan
            KS_lev = MSE(c2cum, lev_cdf)
            KS_norm = MSE(c2cum, nor_cdf)
            KS_uni = MSE(c2cum, uni_cdf)
        elif fit_by == 'pdf':
            KS_pow = MSE(c2, p_pdf)
            KS_exp = MSE(c2, e_pdf)
            KS_logn = MSE(c2, l_pdf)
            KS_lognNpow = MSE(c2, lp_pdf) if lp_pdf is not None else np.nan
            KS_lev = MSE(c2, lev_pdf)
            KS_norm = MSE(c2, nor_pdf)
            KS_uni = MSE(c2, uni_pdf)
        # p_st, p_pv = ks_2samp(c2cum , p_cdf)

        #
        # e_st, e_pv = ks_2samp(c2cum, e_cdf)

        #
        # l_st, l_pv = ks_2samp(c2cum, l_cdf)

        Ks = np.array([KS_pow, KS_exp, KS_logn, KS_lognNpow, KS_lev, KS_norm, KS_uni])
        # idx_Kmax = 3
        idx_Kmax = np.nanargmin(Ks)

        res = np.round(
            [a, KS_pow, b, KS_exp, m, s, KS_logn, mm, ss, aa, xmid, r, overlap, KS_lognNpow, m_lev, s_lev, KS_lev,
             m_nor, s_nor, KS_norm, KS_uni, xmin, xmax], 5)
        pdfs = [p_pdf, e_pdf, l_pdf, lp_pdf, lev_pdf, nor_pdf, uni_pdf]
        cdfs = [p_cdf, e_cdf, l_cdf, lp_cdf, lev_cdf, nor_cdf, uni_cdf]
    p = bout

    names = [f'alpha_{p}', f'KS_pow_{p}',
             f'beta_{p}', f'KS_exp_{p}',
             f'mu_log_{p}', f'sigma_log_{p}', f'KS_log_{p}',
             f'mu_logNpow_{p}', f'sigma_logNpow_{p}', f'alpha_logNpow_{p}', f'switch_logNpow_{p}', f'ratio_logNpow_{p}',
             f'overlap_logNpow_{p}', f'KS_logNpow_{p}',
             f'mu_levy_{p}', f'sigma_levy_{p}', f'KS_levy_{p}',
             f'mu_norm_{p}', f'sigma_norm_{p}', f'KS_norm_{p}',
             f'KS_uni_{p}',
             f'min_{p}', f'max_{p}']
    res_dict = dict(zip(names, res))

    names2 = ['alpha', 'KS_pow',
              'beta', 'KS_exp',
              'mu_log', 'sigma_log', 'KS_log',
              'mu_logNpow', 'sigma_logNpow', 'alpha_logNpow', 'switch_logNpow', 'ratio_logNpow',
              'overlap_logNpow', 'KS_logNpow',
              f'mu_levy', f'sigma_levy', f'KS_levy',
              f'mu_norm', f'sigma_norm', f'KS_norm',
              f'KS_uni',
              'xmin', 'xmax']
    res_dict2 = dict(zip(names2, res))
    best = {bout: {'best': get_best_distro(p, res_dict, idx_Kmax=idx_Kmax),
                   'fits': res_dict2}}
    if store:
        saveConf(best, conf_type='Ref', id=dataset_id, mode='update')

    if print_fits:
        print()
        print(f'-----{dataset_id}-{bout}----------')
        print(f'initial range : {np.min(x0)} - {np.max(x0)}, Nbouts : {len(x0)}')
        print(f'accepted range : {xmin} - {xmax}, Nbouts : {len(x)}')
        print("powerlaw exponent MLE:", a2)
        print("powerlaw exponent powerlaw package:", a)
        print("exponential exponent MLE:", b)
        print("lognormal mean,std:", m, s)
        print("lognormal-powerlaw mean,std, alpha, switch, ratio, overlap :", mm, ss, aa, xmid, r, overlap)
        print("levy loc,scale:", m_lev, s_lev)
        print("normal loc,scale:", m_nor, s_nor)
        print('MSE pow', KS_pow)
        print('MSE exp', KS_exp)
        print('MSE logn', KS_logn)
        print('MSE lognNpow', KS_lognNpow)
        print('MSE levy', KS_lev)
        print('MSE normal', KS_norm)
        print('MSE uniform', KS_uni)
        # print('KS2 pow', p_st, p_pv)
        # print('KS2 exp', e_st, e_pv)
        # print('KS2 logn', l_st, l_pv)
        # print('KS2 lognNpow', lp_st, lp_pv)

        print()
        print(f'---{dataset_id}-{bout}-distro')
        print(best)
        print()

    dic = {
        'values': values, 'pdfs': pdfs, 'cdfs': cdfs, 'Ks': Ks, 'idx_Kmax': idx_Kmax, 'res': res, 'res_dict': res_dict,
        'best': best
    }
    return dic


def get_best_distro(bout, f, idx_Kmax=None):
    k = bout
    r = (f[f'min_{k}'], f[f'max_{k}'])
    if idx_Kmax is None:
        idx_Kmax = np.argmin([f[f'KS_{d}_{k}'] for d in ['pow', 'exp', 'log', 'logNpow', 'levy', 'norm', 'uni']])
    # ind = np.argmin(f[[f'KS_pow_{k}', f'KS_exp_{k}', f'KS_log_{k}', f'KS_logNpow_{k}']])
    if idx_Kmax == 0:
        distro = {'range': r,
                  'name': 'powerlaw',
                  'alpha': f[f'alpha_{k}']}
    elif idx_Kmax == 1:
        distro = {'range': r,
                  'name': 'exponential',
                  'beta': f[f'beta_{k}']}
    elif idx_Kmax == 2:
        distro = {'range': r,
                  'name': 'lognormal',
                  'mu': f[f'mu_log_{k}'],
                  'sigma': f[f'sigma_log_{k}']}
    elif idx_Kmax == 3:
        n = 'logNpow'
        distro = {'range': r,
                  'name': n,
                  'mu': f[f'mu_{n}_{k}'],
                  'sigma': f[f'sigma_{n}_{k}'],
                  'alpha': f[f'alpha_{n}_{k}'],
                  'switch': f[f'switch_{n}_{k}'],
                  'ratio': f[f'ratio_{n}_{k}'],
                  'overlap': f[f'overlap_{n}_{k}'],
                  }
    elif idx_Kmax == 4:
        distro = {'range': r,
                  'name': 'levy',
                  'mu': f[f'mu_levy_{k}'],
                  'sigma': f[f'sigma_levy_{k}']}
    elif idx_Kmax == 5:
        distro = {'range': r,
                  'name': 'normal',
                  'mu': f[f'mu_norm_{k}'],
                  'sigma': f[f'sigma_norm_{k}']}
    elif idx_Kmax == 6:
        distro = {'range': r,
                  'name': 'uniform'}
    return distro


def analyse_bouts(dataset, parameter, scale_coef=1, label=None, xlabel=r'time$(sec)$',
                  dur_max_in_std=None, dur_range=None, save_as=None, save_to=None):
    if label is None:
        label = parameter
    if save_as is None:
        save_as = f'{label}_bouts.pdf'
    if save_to is None:
        save_to = os.path.join(dataset.plot_dir, 'plot_bouts')
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    filepath = os.path.join(save_to, save_as)

    s = dataset.step_data
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    axs = axs.ravel()
    dur = s[parameter].dropna().values * scale_coef
    if dur_range is None:
        if dur_max_in_std is None:
            # xmin, durmax = 1, np.max(dur)
            durmin, durmax = np.min(dur), np.max(dur)
            print(durmin, durmax)
        else:
            std = np.std(dur)
            m = np.mean(dur)
            durmin, durmax = np.min(dur), m + dur_max_in_std * std
    else:
        durmin, durmax = dur_range

    dur = dur[dur >= durmin]
    dur = dur[dur <= durmax]
    u2, du2, c2, c2cum = compute_density(dur, durmin, durmax)
    alpha = 1 + len(dur) / np.sum(np.log(dur / durmin))
    print("powerlaw exponent MLE:", alpha)

    beta = len(dur) / np.sum(dur - durmin)
    print("exponential exponent MLE:", beta)

    mean_lognormal = np.mean(np.log(dur))
    std_lognormal = np.std(np.log(dur))
    print("lognormal mean,std:", mean_lognormal, std_lognormal)

    KS_plaw = np.max(np.abs(c2cum - 1 + powerlaw_cdf(u2, durmin, alpha)))
    KS_exp = np.max(np.abs(c2cum - 1 + exponential_cdf(u2, durmin, beta)))
    KS_logn = np.max(np.abs(c2cum - 1 + lognorm_cdf(u2, mean_lognormal, std_lognormal, durmin)))
    print()
    print('MSE plaw', KS_plaw)
    print('MSE exp', KS_exp)
    print('MSE logn', KS_logn)

    idx_max = np.argmin([KS_plaw, KS_exp, KS_logn])
    lws = [2, 2, 2]
    lws[idx_max] = 4

    # axs[i].loglog(u1, c1, 'or', label=name)
    axs[0].loglog(du2, c2, 'or', label=label)
    axs[0].loglog(du2, powerlaw_pdf(du2, durmin, alpha), 'r', lw=lws[0], label='powerlaw MLE')
    axs[0].loglog(du2, exponential_pdf(du2, durmin, beta), 'g', lw=lws[1], label='exponential MLE')
    axs[0].loglog(du2, lognormal_pdf(du2, mean_lognormal, std_lognormal, durmin), 'b', lw=lws[2], label='lognormal MLE')

    axs[0].legend(loc='lower left', fontsize=15)
    axs[0].axis([durmin, durmax, 1E-5, 1E-0])

    axs[1].loglog(u2, c2cum, 'or', label=label)
    axs[1].loglog(u2, 1 - powerlaw_cdf(u2, durmin, alpha), 'r', lw=lws[0], label='powerlaw MLE')
    axs[1].loglog(u2, 1 - exponential_cdf(u2, durmin, beta), 'g', lw=lws[1], label='exponential MLE')
    axs[1].loglog(u2, 1 - lognorm_cdf(u2, mean_lognormal, std_lognormal, durmin), 'b', lw=lws[2], label='lognormal MLE')

    axs[1].legend(loc='lower left', fontsize=15)
    # axs[1].axis([1.1*xmin, 1.1*durmax,1E-2,1.1*1E-0])
    axs[0].set_title('pdf', fontsize=20)
    axs[1].set_title('cdf', fontsize=20)
    axs[0].set_ylabel('probability', fontsize=15)
    axs[0].set_xlabel(xlabel, fontsize=15)
    axs[1].set_xlabel(xlabel, fontsize=15)
    # axs[i].text(25, 10 ** - 1.5, r'$\alpha=' + str(np.round(alpha * 100) / 100) + '$',
    #        {'color': 'k', 'fontsize': 16})
    # fig.text(0.5, 0.04, r'Duration, $d$', ha='center',fontsize=30)
    # fig.text(0.04, 0.5, r'Cumulative density function, $P_\theta(d)$', va='center', rotation='vertical',fontsize=30)
    fig.subplots_adjust(top=0.92, bottom=0.15, left=0.1, right=0.95, hspace=.005, wspace=0.005)
    fig.savefig(filepath, dpi=300)
    print(f'Plot saved as {filepath}.')


def fit_distribution(dataset, parameters, num_sample=None, num_candidate_dist=10, time_to_fit=120,
                     candidate_distributions=None, distributions=None, save_fits=False,
                     chunk_only=None, absolute=False):
    d = dataset
    if d.step_data is None or d.endpoint_data:
        d.load()
    if chunk_only is not None:
        s = d.step_data.loc[d.step_data[nam.id(chunk_only)].dropna().index]
    else:
        s = d.step_data
    all_dists = sorted([k for k in stats._continuous_distns.__all__ if not (
        (k.startswith('rv_') or k.endswith('_gen') or (k == 'levy_stable') or (k == 'weibull_min')))])
    dists = []
    for k in all_dists:
        dist = getattr(stats.distributions, k)
        if dist.shapes is None:
            dists.append(k)
        elif len(dist.shapes) <= 1:
            dists.append(k)
    results = []
    for i, p in enumerate(parameters):
        try:
            dd = d.endpoint_data[p].dropna().values
        except:
            dd = s[p].dropna().values
        if absolute:
            dd = np.abs(dd)
        if distributions is None:
            if candidate_distributions is None:
                if num_sample is None:
                    ids = d.agent_ids
                else:
                    ids = d.agent_ids[:num_sample]
                try:
                    sample = s.loc[(slice(None), ids), p].dropna().values
                except:
                    sample = d.endpoint_data.loc[ids, p].dropna().values
                if absolute:
                    sample = np.abs(sample)
                f = Fitter(sample)
                f.distributions = dists
                f.fit()
                dists = f.summary(Nbest=num_candidate_dist).index.values
            else:
                dists = candidate_distributions
            ff = Fitter(dd)
            ff.distributions = dists
            ff.timeout = time_to_fit
            ff.fit()
            distribution = ff.get_best()
        else:
            distribution = distributions[i]
        name = list(distribution.keys())[0]
        args = list(distribution.values())[0]
        stat, pv = stats.kstest(dd, name, args=args)
        print(
            f'Parameter {p} was fitted best by a {name} of args {args} with statistic {stat} and p-value {pv}')
        results.append((name, args, stat, pv))

    if save_fits:
        fits = [[p, nam, args, st, pv] for p, (nam, args, st, pv)
                in zip(parameters, results)]
        fits_pd = pd.DataFrame(fits, columns=['parameter', 'dist_name', 'dist_args', 'statistic', 'p_value'])
        fits_pd = fits_pd.set_index('parameter')
        try:
            d.fit_data = pd.read_csv(d.dir_dict['conf'], index_col=['parameter'])
            d.fit_data = fits_pd.combine_first(d.fit_data)
            print('Updated fits')
        except:
            d.fit_data = fits_pd
            print('Initialized fits')
        d.fit_data.to_csv(d.dir_dict['conf'], index=True, header=True)
    return results


def fit_dataset(dataset, target_dir, target_point=None, fit_filename=None,
                angular_fit=True, endpoint_fit=True, bout_fit=True, crawl_fit=True,
                absolute=False, save_to=None):
    d = dataset
    if save_to is None:
        save_to = d.dir_dict['comp_plot']
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    from lib.stor.larva_dataset import LarvaDataset
    dd = LarvaDataset(dir=target_dir, load_data=False)
    file = dd.dir_dict['conf'] if fit_filename is None else os.path.join(dd.data_dir, fit_filename)

    if angular_fit:
        ang_fits = fit_angular_params(d=d, fit_filepath=file, absolute=absolute,
                                      save_to=save_to, save_as='angular_fit.pdf')
    if endpoint_fit:
        end_fits = fit_endpoint_params(d=d, fit_filepath=file,
                                       save_to=save_to, save_as='endpoint_fit.pdf')
    if crawl_fit:
        crawl_fits = fit_crawl_params(d=d, target_point=target_point, fit_filepath=file,
                                      save_to=save_to,
                                      save_as='crawl_fit.pdf')
    if bout_fit:
        bout_fits = fit_bout_params(d=d, fit_filepath=file, save_to=save_to,
                                    save_as='bout_fit.pdf')


def fit_distributions_from_file(dataset, filepath, selected_pars=None, save_fits=True):
    d = dataset
    pars, dists, stats = d.load_fits(filepath=filepath, selected_pars=selected_pars)
    results = fit_distribution(dataset=d, parameters=pars, distributions=dists, save_fits=save_fits)
    global_fit = 0
    for s, (dist_name, dist_args, statistic, p_value) in zip(stats, results):
        global_fit += np.clip(statistic - s, a_min=0, a_max=np.inf)
    return global_fit


def fit_geom_to_stridechains(dataset, is_last=True):
    d = dataset
    if d.step_data is None:
        d.load()
    stridechains = d.step_data[nam.length(nam.chain('stride'))]
    # self.end['stride_reoccurence_rate'] = 1 - 1 / stridechains.mean()
    mean, std = stridechains.mean(), stridechains.std()
    print(f'Mean and std of stride reoccurence rate among larvae : {mean}, {std}')
    p, sse = fit_geom_distribution(stridechains.dropna().values)
    print(f'Stride reoccurence rate is {1 - p}')
    d.stride_reoccurence_rate = 1 - p
    d.stride_reoccurence_rate_sse = sse
    d.config['stride_reoccurence_rate'] = d.stride_reoccurence_rate
    d.config['stride_reoccurence_rate_sse'] = d.stride_reoccurence_rate_sse
    d.save_config()
    if is_last:
        d.save()
    print('Geometric distribution fitted to stridechains')


def fit_geom_distribution(data):
    data = pd.Series(data)
    """Model data by finding best fit distribution to data"""
    x = np.arange(np.min(data), np.max(data) + 1, 1).astype(int)
    y = np.zeros(len(x)).astype(int)

    counts = data.value_counts()
    for i, k in enumerate(x):
        if k in counts.index.values.astype(int):
            y[i] = int(counts.loc[k])
    y = y / len(data)
    # print(y)

    mean = data.mean()
    p = 1 / mean

    # Calculate fitted PDF and error with fit in distribution
    pdf = st.geom.pmf(x, p=p)
    sse = np.sum(np.power(y - pdf, 2.0)) / len(y)
    print(f'geom distribution fitted with SSE :{sse}')
    return p, sse


def fit_powerlaw_distribution(data):
    f = Fitter(data)
    f.distributions = ['powerlaw']
    f.fit()
    k = f.get_best()
    alpha = list(k.values())[0][0]
    return alpha