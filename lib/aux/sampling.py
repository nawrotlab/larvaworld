import scipy.integrate as integrate
import scipy.stats as st
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import truncnorm

from lib.stor.paths import RefFolder


class PowerLawDist(st.rv_continuous):
    def __init__(self, coef=None, range=None, **kwargs):
        super().__init__(**kwargs)
        if range is None:
            range = (-1, 1)
        if coef is None:
            coef = 1
        self.range = range
        self.w_range = range[1] - range[0]
        self.coef = coef
        self.integral = integrate.quad(lambda x: self.power_func(x), self.range[0], self.range[1])[0]

    def power_func(self, t):
        return self.coef * t ** (self.coef - 1)

    def power_func_pdf(self, t):
        if t < self.range[0]:
            return 0
        elif t > self.range[1]:
            return 0
        else:
            return self.power_func(t) / self.integral

    def _pdf(self, t):
        return self.power_func_pdf(t)  # Normalized over its range

    def sample(self, size=1):
        return self.rvs(size=size, random_state=None)

    def sample_int(self, size=1):
        sample = self.rvs(size=size, random_state=None)
        return [int(x) for x in sample]


class GeometricDist(st.rv_continuous):
    def __init__(self, rate=None, **kwargs):
        super().__init__(**kwargs)
        if rate:
            self.rate = rate
        else:
            self.rate = 0.1

    def geometric_func(self, n):
        return self.rate ** (n - 1) * (1 - self.rate)

    def _pdf(self, n):
        return self.geometric_func(n)  # Normalized over its range, in this case [0,1]

    def sample(self, size=1):
        return self.rvs(size=size, random_state=None)

    def sample_int(self, size=1):
        sample = self.rvs(size=size, random_state=None)
        return [int(x) for x in sample]


def sample_agents(filepath=None, pars=None, N=1, sample_dataset='reference'):
    path_dir = f'{RefFolder}/{sample_dataset}'
    path_data = f'{path_dir}/data/reference.csv'
    # path_fits = f'{path_dir}/data/bout_fits.csv'
    if filepath is None:
        filepath = path_data
    data = pd.read_csv(filepath, index_col=0)
    if pars is None:
        pars = data.columns
    else:
        pars = [p for p in data.columns if p in pars]
    means = [data[p].mean() for p in pars]
    if len(pars)>=2:
        base=data[pars].values.T
        cov = np.cov(base)
        samples = np.random.multivariate_normal(means, cov, N).T

        # print(data[pars].values.T.shape)
        # print(pars)
        # import matplotlib.pyplot  as plt
        # plt.scatter(x=base[2,:], y=base[3,:], color='b')
        # plt.scatter(x=samples[2,:], y=samples[3,:], color='r')
        # plt.show()
        # raise
    elif len(pars)==1:
        std=np.std(data[pars].values)
        samples = np.atleast_2d(np.random.normal(means[0], std, N))
    return pars, samples


def get_ref_bout_distros(mode='stridechain_dist', sample_dataset='reference'):
    path_dir = f'{RefFolder}/{sample_dataset}'
    # path_data = f'{path_dir}/data/reference.csv'
    path_fits = f'{path_dir}/data/bout_fits.csv'

    f = pd.read_csv(path_fits, index_col=0).xs(sample_dataset)
    if mode=='stridechain_dist' :
        str_i = np.argmin(f[['KS_pow_stride', 'KS_exp_stride', 'KS_log_stride']])
        if str_i == 0:
            str_dist = {'range': (f['min_stride'], f['max_stride']),
                        'name': 'powerlaw',
                        'alpha': f['alpha_stride']}
        elif str_i == 1:
            str_dist = {'range': (f['min_stride'], f['max_stride']),
                        'name': 'exponential',
                        'lambda': f['lambda_stride']}
        elif str_i == 2:
            str_dist = {'range': (f['min_stride'], f['max_stride']),
                        'name': 'lognormal',
                        'mu': f['mu_log_stride'],
                        'sigma': f['sigma_log_stride']}
        return str_dist

    elif mode=='pause_dist' :
        pau_i = np.argmin(f[['KS_pow_pause', 'KS_exp_pause', 'KS_log_pause']])
        if pau_i == 0:
            pau_dist = {'range': (f['min_pause'], f['max_pause']),
                        'name': 'powerlaw',
                        'alpha': f['alpha_pause']}
        elif pau_i == 1:
            pau_dist = {'range': (f['min_pause'], f['max_pause']),
                        'name': 'exponential',
                        'lambda': f['lambda_pause']}
        elif pau_i == 2:
            pau_dist = {'range': (f['min_pause'], f['max_pause']),
                        'name': 'lognormal',
                        'mu': f['mu_log_pause'],
                        'sigma': f['sigma_log_pause']}
        return pau_dist




def truncated_power_law(a, xmin, xmax):
    x = np.arange(xmin, xmax + 1, dtype='float')
    pmf = 1 / x ** a
    pmf /= pmf.sum()
    # return stats.rv_continuous(values=(range(xmin, xmax+1), pmf))
    return stats.rv_discrete(values=(range(xmin, xmax + 1), pmf))


def sample_lognormal(mean, sigma, xmin, xmax):
    # print(mean, sigma)
    while True:
        v = np.random.lognormal(mean=mean, sigma=sigma, size=None)
        if v >= xmin and v <= xmax:
            break
    return v


def sample_lognormal_int(mean, sigma, xmin, xmax):
    while True:
        v = np.floor(np.random.lognormal(mean=mean, sigma=sigma, size=None))
        if v >= xmin and v <= xmax:
            break
    # print(np.round(v))
    return v


def lognorm_params(mode, stddev):
    """
    Given the mode and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy's parameterization of the
    distribution.
    """
    p = np.poly1d([1, -1, 0, 0, -(stddev / mode) ** 2])
    r = p.roots
    sol = r[(r.imag == 0) & (r.real > 0)].real
    shape = np.sqrt(np.log(sol))
    scale = mode * sol
    return shape, scale


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
