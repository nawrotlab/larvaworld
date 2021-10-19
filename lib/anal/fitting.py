import os
import warnings
import numpy as np
from scipy.stats import levy, norm, uniform, rv_discrete
from scipy.special import erf

from lib.aux import naming as nam
from lib.conf.stored.conf import saveConf
from lib.process.aux import suppress_stdout, suppress_stdout_stderr


def powerlaw_cdf(x, xmin, alpha):
    return 1 - (x / xmin) ** (1 - alpha)


def powerlaw_pdf(x, xmin, alpha):
    return (alpha - 1) / xmin * (x / xmin) ** (-alpha)


def levy_pdf(x, mu, sigma):
    return np.sqrt(sigma / (2 * np.pi)) * np.exp(-sigma / (2 * (x - mu))) / (x - mu) ** 1.5


def levy_cdf(x, mu, sigma):
    res = 1 - erf(np.sqrt(sigma / (2 * (x - mu))))
    if np.isnan(res[0]):
        res[0] = 0
    return res


def norm_pdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def norm_cdf(x, mu, sigma):
    res = 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))
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


def lognorm_cdf(x, mu, sigma):
    return 0.5 + 0.5 * erf((np.log(x) - mu) / np.sqrt(2) / sigma)


def lognormal_pdf(x, mu, sigma):
    return 1 / (x * sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)


def logNpow_pdf(x, mu, sigma, alpha, switch, ratio):
    log_pdf = lognormal_pdf(x[x < switch], mu, sigma) * ratio
    pow_pdf = powerlaw_pdf(x[x >= switch], switch, alpha) * (1 - ratio)
    return np.hstack([log_pdf, pow_pdf])


def logNpow_cdf(x, mu, sigma, alpha, switch, ratio):
    log_cdf = 1 - lognorm_cdf(x[x < switch], mu, sigma)
    pow_cdf = (1 - powerlaw_cdf(x[x >= switch], switch, alpha)) * (1 - ratio)
    return 1 - np.hstack([log_cdf, pow_cdf])


def get_logNpow(x, xmax, xmid, overlap=0, discrete=False):
    r = len(x[x < xmid]) / len(x)
    m, s = get_lognormal(x[x < xmid + overlap * (xmax - xmid)])
    a = get_powerlaw_alpha(x[x >= xmid], xmid, xmax, discrete=discrete)
    return m, s, a, r


def get_powerlaw_alpha(dur, dur0, dur1, discrete=False):
    with suppress_stdout_stderr():
        from powerlaw import Fit
        return Fit(dur, xmin=dur0, xmax=dur1, discrete=discrete).power_law.alpha


def get_lognormal(dur):
    d = np.log(dur)
    return np.mean(d), np.std(d)


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
    if scaled:
        s1 = sum(a1)
        s2 = sum(a2)
    else:
        s1, s2 = 1, 1
    return np.sum((a1 / s1 - a2 / s2) ** 2) / a1.shape[0]


def logNpow_switch(x, xmax, u2, du2, c2cum, c2, discrete=False, fit_by='cdf'):
    xmids = u2[1:-int(len(u2) / 3)][::2]
    overlaps = np.linspace(0, 1, 6)
    temp = np.ones([len(xmids), len(overlaps)])
    for i, xmid in enumerate(xmids):
        for j, ov in enumerate(overlaps):
            mm, ss, aa, r = get_logNpow(x, xmax, xmid, discrete=discrete, overlap=ov)
            lp_cdf = 1 - logNpow_cdf(u2, mm, ss, aa, xmid, r)
            lp_pdf = logNpow_pdf(du2, mm, ss, aa, xmid, r)
            if fit_by == 'cdf':
                temp[i, j] = MSE(c2cum, lp_cdf)
            elif fit_by == 'pdf':
                temp[i, j] = MSE(c2, lp_pdf)

    if all(np.isnan(temp.flatten())):
        return np.nan
    else:
        ii, jj = np.unravel_index(np.nanargmin(temp), temp.shape)
        return xmids[ii], overlaps[jj]


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
        dic = fit_bout_distros(x0, xmin, xmax, discrete=disc, print_fits=True,
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
    # config['EEB_poly1d'] = {config['dt']: get_EEB_poly1d(**config['intermitter']).c.tolist()}

    return dic


def fit_bout_distros(x0, xmin, xmax, discrete=False, xmid=np.nan, overlap=0.0, Nbins=64, print_fits=True,
                     dataset_id='dataset', bout='pause', combine=True, store=False, fit_by='pdf'):
    with suppress_stdout(True):
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

        m, s = get_lognormal(x)
        l_cdf = 1 - lognorm_cdf(u2, m, s)
        l_pdf = lognormal_pdf(du2, m, s)

        m_lev, s_lev = levy.fit(x)
        lev_cdf = 1 - levy_cdf(u2, m_lev, s_lev)
        lev_pdf = levy_pdf(du2, m_lev, s_lev)

        m_nor, s_nor = norm.fit(x)
        nor_cdf = 1 - norm_cdf(u2, m_nor, s_nor)
        nor_pdf = norm_pdf(du2, m_nor, s_nor)

        uni_cdf = 1 - uniform_cdf(u2, xmin, xmin + xmax)
        uni_pdf = uniform_pdf(du2, xmin, xmin + xmax)

        if np.isnan(xmid) and combine:
            xmid, overlap = logNpow_switch(x, xmax, u2, du2, c2cum, c2, discrete, fit_by)

        if not np.isnan(xmid):
            mm, ss, aa, r = get_logNpow(x, xmax, xmid, discrete=discrete, overlap=overlap)
            lp_cdf = 1 - logNpow_cdf(u2, mm, ss, aa, xmid, r)
            lp_pdf = logNpow_pdf(du2, mm, ss, aa, xmid, r)
        else:
            mm, ss, aa, r = np.nan, np.nan, np.nan, np.nan
            lp_cdf, lp_pdf = None, None

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

        Ks = np.array([KS_pow, KS_exp, KS_logn, KS_lognNpow, KS_lev, KS_norm, KS_uni])
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


def pvalue_star(pv):
    a = {1e-4: "****", 1e-3: "***",
         1e-2: "**", 0.05: "*", 1: "ns"}
    for k, v in a.items():
        if pv < k:
            return v
    return "ns"


class BoutGenerator:
    def __init__(self, name, range, dt, **kwargs):
        self.name = name
        self.dt = dt
        self.range = range
        self.ddfs = {
            'powerlaw': {'cdf': powerlaw_cdf, 'pdf': powerlaw_pdf, 'args': ['xmin', 'alpha']},
            'exponential': {'cdf': exponential_cdf, 'pdf': exponential_pdf, 'args': ['xmin', 'beta']},
            'lognormal': {'cdf': lognorm_cdf, 'pdf': lognormal_pdf, 'args': ['mu', 'sigma']},
            'logNpow': {'cdf': logNpow_cdf, 'pdf': logNpow_pdf,
                        'args': ['alpha', 'mu', 'sigma', 'switch', 'ratio']},
            'levy': {'cdf': levy_cdf, 'pdf': levy_pdf, 'args': ['mu', 'sigma']},
            'normal': {'cdf': norm_cdf, 'pdf': norm_pdf, 'args': ['mu', 'sigma']},
            'uniform': {'cdf': uniform_cdf, 'pdf': uniform_pdf, 'args': ['xmin', 'xmax']},
        }
        self.xmin, self.xmax = range
        kwargs.update({'xmin': self.xmin, 'xmax': self.xmax})
        self.args = {a: kwargs[a] for a in self.ddfs[self.name]['args']}

        self.dist = self.build(**self.args)

    def sample(self, size=1):
        vs = self.dist.rvs(size=size) * self.dt
        return vs[0] if size == 1 else vs

    def build(self, **kwargs):
        x0, x1 = int(self.xmin / self.dt), int(self.xmax / self.dt)
        xx = np.arange(x0, x1 + 1)
        pmf = self.ddfs[self.name]['pdf'](xx * self.dt, **kwargs)
        mask = ~np.isnan(pmf)
        pmf = pmf[mask]
        xx = xx[mask]
        pmf /= pmf.sum()
        return rv_discrete(values=(xx, pmf))

    def get(self, x, mode):
        func = self.ddfs[self.name][mode]
        return func(x=x, **self.args)
