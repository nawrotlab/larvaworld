from typing import Tuple

import numpy as np
from scipy.special import erf
from scipy.stats import uniform, levy, norm

import lib.aux.dictsNlists as dNl
from lib.aux.par_aux import sub, subsup
from lib.registry.units import ureg


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


def get_powerlaw_alpha2(x, xmin=None, xmax=None, discrete=False):
    from lib.aux.stdout import suppress_stdout_stderr
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    with suppress_stdout_stderr():
        from powerlaw import Fit
        a = Fit(x, xmin=xmin, xmax=xmax, discrete=discrete).power_law.alpha
        return dNl.NestDict({'xmin': xmin, 'alpha': a})


def get_exp_beta2(x, xmin=None):
    if xmin is None:
        xmin = np.min(x)
    b = len(x) / np.sum(x - xmin)
    return {'xmin': xmin, 'beta': b}


def fit_levy(x):
    m, s = levy.fit(x)
    return {'mu': m, 'sigma': s}


def fit_norm(x):
    m, s = norm.fit(x)
    return {'mu': m, 'sigma': s}


def fit_uni(x, xmin=None, xmax=None):
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.np.max(x)
    return {'xmin': xmin, 'xmax': xmax}


def get_logNpow2(x, xmax, xmid, overlap=0, discrete=False):
    dic = dNl.NestDict()
    dic.ratio = len(x[x < xmid]) / len(x)
    xx = np.log(x[x < xmid + overlap * (xmax - xmid)])
    dic.mu = np.mean(xx)
    dic.sigma = np.std(xx)
    dic.switch = xmid
    from lib.aux.stdout import suppress_stdout_stderr
    with suppress_stdout_stderr():
        from powerlaw import Fit
        dic.alpha = Fit(x=x[x >= xmid], xmin=xmid, xmax=xmax, discrete=discrete).power_law.alpha


        return dic


def build_dist_dict():
    d = dNl.NestDict({
        'powerlaw': {'cdf': powerlaw_cdf, 'pdf': powerlaw_pdf, 'args': ['xmin', 'alpha'],
                     'lab_func': lambda v: f'Powerlaw(a={np.round(v.alpha, 2)})',
                     'func': lambda x, xmin=None, xmax=None, discrete=False: get_powerlaw_alpha2(x, xmin, xmax,
                                                                                                 discrete)},
        'exponential': {'cdf': exponential_cdf, 'pdf': exponential_pdf, 'args': ['xmin', 'beta'],
                        'lab_func': lambda v: f'Exp(b={np.round(v.beta, 2)})',
                        'func': lambda x, xmin=None: get_exp_beta2(x, xmin)},
        'lognormal': {'cdf': lognorm_cdf, 'pdf': lognormal_pdf, 'args': ['mu', 'sigma'],
                      'lab_func': lambda v: f'Lognormal(m={np.round(v.mu, 2)}, s={np.round(v.sigma, 2)})',
                      'func': lambda x: {'mu': np.mean(np.log(x)), 'sigma': np.std(np.log(x))}},
        'logNpow': {'cdf': logNpow_cdf, 'pdf': logNpow_pdf,
                    'args': ['alpha', 'mu', 'sigma', 'switch', 'ratio'], 'lab_func': lambda
                v: f'Lognormal-Powerlaw(a={np.round(v.alpha, 2)}, m={np.round(v.mu, 2)}, s={np.round(v.sigma, 2)})',
                    'func': lambda x, xmax, xmid, overlap=0, discrete=False: get_logNpow2(x, xmax, xmid, overlap,
                                                                                          discrete)},
        'levy': {'cdf': levy_cdf, 'pdf': levy_pdf, 'args': ['mu', 'sigma'],
                 'lab_func': lambda v: f'Levy(m={np.round(v.mu, 2)}, s={np.round(v.sigma, 2)})',
                 'func': lambda x: fit_levy(x)},
        'normal': {'cdf': norm_cdf, 'pdf': norm_pdf, 'args': ['mu', 'sigma'],
                   'lab_func': lambda v: f'N(m={np.round(v.mu, 2)}, s={np.round(v.sigma, 2)})',
                   'func': lambda x: fit_norm(x)},
        'uniform': {'cdf': uniform_cdf, 'pdf': uniform_pdf, 'args': ['xmin', 'xmax'],
                    'lab_func': lambda v: f'Uniform()',
                    'func': lambda x, xmin=None, xmax=None: fit_uni(x, xmin, xmax)},
    })
    return d


def get_dist(k, k0='intermitter', v=None, return_tabrows=False, d0=None, return_all=False):
    if d0 is None:
        d0 = build_dist_dict()
    dict0 = {
        'stridechain_dist': ('run length', ('N', 'R'), ureg.dimensionless, '# $strides$'),
        'pause_dist': ('pause duration', ('t', 'P'), ureg.s, '$sec$'),
        'run_dist': ('run duration', ('t', 'R'), ureg.s, '$sec$')
    }
    disp, (tt0, tt1), u, uname = dict0[k]
    dispD, dispR = f'{disp} distribution', f'{disp} range'
    symD = sub(tt0, tt1)
    kD = f'{tt0}_{tt1}'
    kR = f'{kD}_r'
    sym1, sym2 = subsup(tt0, tt1, 'min'), subsup(tt0, tt1, 'max')
    symR = f'[{sym1},{sym2}]'
    p = {'disp': disp, 'k': kD, 'sym': symD, 'u_name': uname, 'u': u, 'dtype': dict,
         'v0': {'fit': True, 'name': None, 'range': None}}

    if return_tabrows:
        dist_v = d0[v.name].lab_func(v)
        vs1 = [k0, dispD, symD, dist_v, '-']
        vs2 = [k0, dispR, symR, v.range, uname]
        return vs1, vs2
    elif return_all:
        pD = {'disp': dispD, 'k': kD, 'v0': None, 'vs': list(d0.keys()), 'sym': symD, 'dtype': str}
        pR = {'disp': dispR, 'k': kR, 'u_name': uname, 'u': u, 'sym': symR, 'v0': None, 'dtype': Tuple[float]}
        return p, pD, pR
    else:
        return p


class DistDict:
    def __init__(self, load=False):
        from lib.registry import paths
        if not load:
            self.dict = build_dist_dict()
            # dNl.save_dict(self.dict, paths.path_dict['DistDict'])
        else:
            self.dict = dNl.load_dict(paths.path_dict['DistDict'])


    def get_dist(self, **kwargs):
        return get_dist(d0=self.dict, **kwargs)

dist_dict = DistDict()
