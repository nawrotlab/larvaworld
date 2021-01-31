import scipy.integrate as integrate
import scipy.stats as st
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import truncnorm

from lib.stor.paths import Ref_path


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


def sample_agents(filepath=None, pars=None, num_agents=1):
    if filepath is None:
        filepath=Ref_path
    data = pd.read_csv(filepath, index_col=0)
    if pars is None:
        pars = data.columns
    else:
        pars = [p for p in data.columns if p in pars]
    cov = np.cov(data[pars].values.T)
    means = [data[p].mean() for p in pars]
    samples = np.random.multivariate_normal(means, cov, num_agents).T
    return pars, samples


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
    p = np.poly1d([1, -1, 0, 0, -(stddev/mode)**2])
    r = p.roots
    sol = r[(r.imag == 0) & (r.real > 0)].real
    shape = np.sqrt(np.log(sol))
    scale = mode * sol
    return shape, scale



def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


