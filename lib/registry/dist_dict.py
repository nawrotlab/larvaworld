from typing import Tuple

import numpy as np
from scipy.special import erf
from scipy.stats import uniform

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


def build_dist_dict() :
    d = dNl.NestDict({
            'powerlaw': {'cdf': powerlaw_cdf, 'pdf': powerlaw_pdf, 'args': ['xmin', 'alpha'], 'lab_func' : lambda v : f'Powerlaw(a={np.round(v.alpha, 2)})'},
            'exponential': {'cdf': exponential_cdf, 'pdf': exponential_pdf, 'args': ['xmin', 'beta'], 'lab_func' : lambda v : f'Exp(b={np.round(v.beta, 2)})'},
            'lognormal': {'cdf': lognorm_cdf, 'pdf': lognormal_pdf, 'args': ['mu', 'sigma'], 'lab_func' : lambda v : f'Lognormal(m={np.round(v.mu, 2)}, s={np.round(v.sigma, 2)})'},
            'logNpow': {'cdf': logNpow_cdf, 'pdf': logNpow_pdf,
                        'args': ['alpha', 'mu', 'sigma', 'switch', 'ratio'], 'lab_func' : lambda v : f'Lognormal-Powerlaw(a={np.round(v.alpha, 2)}, m={np.round(v.mu, 2)}, s={np.round(v.sigma, 2)})'},
            'levy': {'cdf': levy_cdf, 'pdf': levy_pdf, 'args': ['mu', 'sigma'], 'lab_func' : lambda v : f'Levy(m={np.round(v.mu, 2)}, s={np.round(v.sigma, 2)})'},
            'normal': {'cdf': norm_cdf, 'pdf': norm_pdf, 'args': ['mu', 'sigma'], 'lab_func' : lambda v : f'N(m={np.round(v.mu, 2)}, s={np.round(v.sigma, 2)})'},
            'uniform': {'cdf': uniform_cdf, 'pdf': uniform_pdf, 'args': ['xmin', 'xmax'], 'lab_func' : lambda v : f'Uniform()'},
        })
    return d

class DistDict :
    def __init__(self):
        self.dict=build_dist_dict()

    def get_dist(self,v, k, k0='intermitter',return_tabrows=False):
        dict0 = {
            'stridechain_dist': ('run length',('N','R'),ureg.dimensionless, '# $strides$'),
            'pause_dist':('pause duration',('t','P'),ureg.s, '$sec$'),
            'run_dist': ('run duration',('t','R'), ureg.s,'$sec$')
        }
        ll0,(tt0,tt1), u, uname=dict0[k]
        dispD,dispR=f'{ll0} distribution',f'{ll0} range'
        symD=sub(tt0,tt1)
        kD=f'{tt0}_{tt1}'
        kR=f'{kD}_r'
        sym1,sym2=subsup(tt0,tt1,'min'),subsup(tt0,tt1,'max')
        symR=f'[{sym1},{sym2}]'

        dist_v = self.dict[v.name].lab_func(v)

        pD={'disp' : dispD, 'k' :kD, 'label' : dist_v, 'sym' : symD}
        pR={'disp' : dispR, 'k' :kR, 'u_name' : uname,'u' : u, 'sym' : symR, 'v0':v.range, 'dtype' : Tuple[float]}

        if return_tabrows:
            vs1 = [k0, dispD, symD, dist_v, '-']
            vs2 = [k0, dispR, symR, v.range,uname]
            return vs1,vs2
        else :
            return pD,pR