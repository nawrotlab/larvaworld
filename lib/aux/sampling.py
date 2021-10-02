import scipy.integrate as integrate
import scipy.stats as st
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import truncnorm, lognorm, rv_discrete, uniform

from lib.anal.fitting import compute_density, powerlaw_cdf, exponential_cdf, lognorm_cdf, powerlaw_pdf, logNpow_pdf, \
    fit_bout_distros, logNpow_cdf, get_distro, lognormal_pdf, exponential_pdf, levy_cdf, levy_pdf, \
    norm_cdf, norm_pdf, uniform_pdf, uniform_cdf
from lib.conf.conf import loadConf
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


def sample_agents(pars, filepath=None, N=1, sample_dataset='reference'):

    if filepath is None:
        path_dir = f'{RefFolder}/{sample_dataset}'
        path_data = f'{path_dir}/data/reference.csv'
        filepath = path_data
    data = pd.read_csv(filepath, index_col=0)
    pars = [p for p in pars if p in data.columns]
    means = [data[p].mean() for p in pars]

    if len(pars)>=2:
        base=data[pars].values.T
        cov = np.cov(base)
        samples = np.random.multivariate_normal(means, cov, N).T
    elif len(pars)==1:
        std=np.std(data[pars].values)
        samples = np.atleast_2d(np.random.normal(means[0], std, N))
    return pars, samples

class BoutGenerator :
    def __init__(self, name, range,dt,  mode='rvs', **kwargs):
        self.name=name
        self.dt=dt
        ddfs = {
            'powerlaw': {'cdf': powerlaw_cdf, 'pdf': powerlaw_pdf, 'args': ['alpha'], 'rvs': trunc_powerlaw},
            'exponential': {'cdf': exponential_cdf, 'pdf': exponential_pdf, 'args': ['beta'], 'rvs': exponential_discrete},
            'lognormal': {'cdf': lognorm_cdf, 'pdf': lognormal_pdf, 'args': ['mu', 'sigma'], 'rvs': lognormal_discrete},
            'logNpow': {'cdf': logNpow_cdf, 'pdf': logNpow_pdf,
                        'args': ['alpha', 'mu', 'sigma', 'switch', 'ratio', 'overlap'], 'rvs': logNpow_distro},
            'levy': {'cdf': levy_cdf, 'pdf': levy_pdf, 'args': ['mu', 'sigma'], 'rvs': levy_discrete},
            'normal': {'cdf': norm_cdf, 'pdf': norm_pdf, 'args': ['mu', 'sigma'], 'rvs': norm_discrete},
            'uniform': {'cdf': uniform_cdf, 'pdf': uniform_pdf, 'args': [], 'rvs': uniform_discrete},
        }
        self.xmin, self.xmax = range
        self.funct = ddfs[name][mode]
        self.args= {'xmin' : self.xmin,'xmax' : self.xmax,'range' : range,'dt' :self.dt,'name':self.name,  **kwargs}
        # self.args  = {'xmin' : self.xmin, **{a: kwargs[a] for a in ddfs[name]['args']}}

        self.dist = self.funct(**self.args)
    def sample(self, size=1):
        vs = self.dist.rvs(size=size) * self.dt
        return vs[0] if size == 1 else vs
        # if self.name in ['logNpow', 'powerlaw'] :
        #     self.dist = self.funct(**self.args)
        #     vs = self.dist.rvs(size=size) * self.dt
        # else :
        #     vs=np.ones(size)*np.nan
        #     for i in range(size) :
        #         vs[i] = self.funct(**self.args, size=1)
        # return vs[0] if size==1 else vs


def get_sample_bout_distro(bout='stride', sample_dataset='reference'):
    # path_dir = f'{RefFolder}/{sample_dataset}'
    # path_fits = f'{path_dir}/data/bout_fits.csv'
    # f = pd.read_csv(path_fits, index_col=0).xs(sample_dataset)
    # return get_best_distro(bout, f)
    distro=loadConf(sample_dataset, 'Ref')[bout]['best']
    return distro


def logNpow_distro(alpha, range, mu, sigma, switch, ratio, dt, overlap=0, **kwargs) :
    xmin, xmax=range
    x0, x1 = int(xmin/ dt), int(xmax/ dt)
    xx = np.arange(x0,x1)
    x=xx*dt
    pmf=logNpow_pdf(x, mu, sigma, alpha, xmin, switch, ratio, overlap)
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(xx, pmf))


def trunc_powerlaw(alpha, range, dt=1, **kwargs):
    xmin, xmax = range
    x0, x1 = int(xmin / dt), int(xmax / dt)
    xx = np.arange(x0, x1)
    x = xx * dt
    # xk = np.arange(xmin, xmax,dt).astype(int)
    # pk = 1 / xk ** alpha
    # print('ddd')
    pk = (alpha - 1) / x[0] * (x / x[0]) ** (-alpha)
    pk /= pk.sum()

    return stats.rv_discrete(values=(xx, pk))

def exponential_discrete(beta, range, dt=1, **kwargs) :
    xmin, xmax = range
    x0, x1 = int(xmin / dt), int(xmax / dt)
    xx = np.arange(x0, x1 + 1)
    x = xx * dt
    pmf = exponential_pdf(x, xmin, beta)
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(xx, pmf))


def lognormal_discrete2(mu, sigma, range, dt=1, **kwargs):
    xmin,xmax=range
    x0, x1 = np.round(xmin / dt).astype(int), np.ceil(xmax / dt).astype(int)
    N=x1-x0
    Dd = lognorm(s=sigma, loc=0.0, scale=np.exp(mu))
    pk2 = Dd.cdf(np.linspace(xmin + 1*dt, xmax + 2*dt, N+1)) - Dd.cdf(np.linspace(xmin, xmax + 1*dt, N+1))
    pk2 = pk2 / np.sum(pk2)
    xrng = np.arange(x0, x1 + 1, 1)
    # print(dt)
    return stats.rv_discrete(values=(xrng, pk2))

def lognormal_discrete(mu, sigma, range, dt=1, **kwargs):
    xmin, xmax = range
    x0, x1 = int(xmin / dt), int(xmax / dt)
    xx = np.arange(x0, x1+1)
    x = xx * dt
    pmf = lognormal_pdf(x, mu, sigma)
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(xx, pmf))

def levy_discrete(mu, sigma, range, dt=1, **kwargs):
    xmin, xmax = range
    x0, x1 = int(xmin / dt), int(xmax / dt)
    xx = np.arange(x0, x1+1)
    x = xx * dt
    pmf = levy_pdf(x, mu, sigma)
    # print(pmf)
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(xx, pmf))

def norm_discrete(mu, sigma, range, dt=1, **kwargs):
    xmin, xmax = range
    x0, x1 = int(xmin / dt), int(xmax / dt)
    xx = np.arange(x0, x1+1)
    x = xx * dt
    pmf = norm_pdf(x, mu, sigma)
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(xx, pmf))

def uniform_discrete(range, dt=1, **kwargs):
    xmin, xmax = range
    x0, x1 = int(xmin / dt), int(xmax / dt)
    xx = np.arange(x0, x1+1)
    x = xx * dt
    pmf = uniform.pdf(x, xmin, xmin+xmax)
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(xx, pmf))


def sample_lognormal(mu, sigma,range, size=1, **kwargs):
    xmin, xmax = range
    while True:
        v = np.random.lognormal(mean=mu, sigma=sigma, size=size)
        if v >= xmin and v <= xmax:
            break
    return v


def sample_lognormal_int(mu, sigma, range, size=1, **kwargs):
    xmin, xmax = range
    while True:
        v = np.floor(np.random.lognormal(mean=mu, sigma=sigma, size=size))
        if v >= xmin and v <= xmax:
            break
    # print(np.round(v))
    return v


def lognorm_params(mode, stddev):
    """
    Given the mode and std. dev. of the log-normal distribution, this function
    returns the shape and scale parameters for scipy'sigma parameterization of the
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    d_id='Fed'
    bouts=['stride', 'pause']
    # bout='pause'
    # e=pd.read_csv(f'{RefFolder}/Starved/data/end.csv', index_col=0)

    fr=11.27
    dt=1/fr

    # pau_dist1=logNpow_distro(**pause_dist, dt=dt)
    # pp1=pau_dist1.rvs(size=10000)* dt

    # pau_dist2=logNpow_distro(alpha=2.3436,mu=-1.0676, sigma=0.52, switch=0.454, dt=dt)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5),sharex=False, sharey=True)
    axs=axs.ravel()
    for j, (bout, comb,discr, (xmin,xmax), par, dt_bout) in enumerate(zip(bouts, [False, True],[False, False], [(1,100), (0.4,20.0)], ['stridechain_length', 'pause_dur'], [1, dt])) :
        # if j==1 :
        #     continue
        print(f'------{bout}--------')
        pp = pd.read_csv(f'{RefFolder}/{d_id}/aux/par_distros/{par}.csv', index_col=0).values.flatten()
        # dist0 = get_sample_bout_distro(bout=bout, sample_dataset='Starved')
        # a, m, s, xmid, r = [dist0[k] for k in ['alpha', 'mu', 'sigma', 'switch', 'ratio']]
        dist0 =loadConf(d_id,'Ref')[bout]['best']
        print(-1, dist0)
        u2, du2, c2, c2cum = compute_density(pp, xmin, xmax)
        cdf0 = 1 - get_distro(x=u2, mode='cdf', **dist0)
        pdf0 = get_distro(x=du2, mode='pdf', **dist0)
        axs[j].loglog(u2, cdf0, 'g', lw=4, label='stored')

        ls=['exp','sim1', 'sim2', 'sim3', 'sim4', 'sim5', 'sim6']
        cols=['b', 'r','c', 'm', 'orange', 'grey', 'lightgreen']
        for i in range(5) :
        # for i, dur in enumerate([pp0, pp1]) :
            values, pdfs, cdfs, Ks, idx_Kmax, res, res_dict, best = fit_bout_distros(pp, xmin=xmin, xmax=xmax, fr=fr, xmid=np.nan, bout=bout, fit_by='cdf',
                                                                                     print_fits=False, combine=comb, discrete=discr, overlap=0.2)
            u2, du2, c2, c2cum = values
            p_cdf, e_cdf, l_cdf, lp_cdf = cdfs
            # print(sum(pdf0))
            # print(cdf0[0],len(cdf0))
            l=ls[i]
            # axs[j].loglog(du2, c2, '.', color=cols[i], alpha=0.7)
            axs[j].loglog(u2, c2cum, '.', color=cols[i], alpha=0.7)
            axs[j].loglog(u2, cdfs[idx_Kmax], color=cols[i], lw=2, label=f'{l}_{i}')

            print(i,c2cum[0], u2[0], len(u2), best[bout]['best'])
            pau_dist = BoutGenerator(**best[bout]['best'], dt=dt_bout)
            # pau_dist = logNpow_distro(**best[bout], dt=dt)
            pp = pau_dist.sample(size=10000)
        axs[j].legend()
        axs[j].set_ylim([10**-3.5, 10**0])
    # axs[1].loglog(du2, c2, '.', color='red', alpha=0.7)
    # axs[1].loglog(du2, powerlaw_pdf(du2, xmin, aa2), 'c', lw=2, label='powerlaw')

    # axs[2].loglog(u2_t, c2cum_t, '.', color='red', alpha=0.7)
    # axs[2].loglog(u2_t, 1 - powerlaw_cdf(u2_t, dur0_t, aa2_t), 'c', lw=2, label='powerlaw')
    # axs[3].loglog(du2_t, c2_t, '.', color='red', alpha=0.7)
    # axs[3].loglog(du2_t, powerlaw_pdf(du2_t, dur0_t, aa2_t), 'c', lw=2, label='powerlaw')

    # axs[4].set_xticks(bins, ["2^%sigma" % i for i in bins])
    # ys,xs,pol=axs[5].hist(numpy.log10(dur), log=True, bins=np.linspace(np.log10(xmin), np.log10(dur1), Nbins), histtype='step',alpha=0)
    # ys,xs,pol=axs[5].hist(numpy.log10(dur), log=True, bins=np.linspace(np.log10(xmin), np.log10(dur1), Nbins), histtype='step',alpha=0)
    # xs=0.5 * (xs[:-1] + xs[1:])
    # axs[5].plot(xs,ys)
    # cum_ys = 1-np.cumsum(ys)/np.sum(ys)
    # cum_ys = np.array([np.cumsum(ys[:i+1]) for i,(x,y) in enumerate(zip(xs,ys))])
    # print(xs, du2)
    # axs[4].loglog(du2, cum_ys, '.', color='red', alpha=0.7)
    # axs[4].loglog(u2, 1 - powerlaw_cdf(u2, xmin, aa2), 'c', lw=2, label='powerlaw')
    # axs[4].scatter(xs, np.log10(cum_ys))
    # axs[4].set_yticks([])
    # axs[5].hist(numpy.log10(dur), log=True, bins=np.linspace(np.log10(xmin), np.log10(dur1), Nbins), histtype='step')
    # ys,xs,pol = axs[7].hist(numpy.log10(dur_t), log=True, bins=np.linspace(np.log10(dur0_t), np.log10(dur1_t), Nbins), histtype='step',alpha=0)
    # xs = 0.5 * (xs[:-1] + xs[1:])
    # axs[7].plot(xs, ys)
    # cum_ys = 1 - np.cumsum(ys) / np.sum(ys)
    # axs[6].loglog(du2_t, cum_ys, '.', color='red', alpha=0.7)
    # axs[6].loglog(u2_t, 1 - powerlaw_cdf(u2_t, dur0_t, aa2_t), 'c', lw=2, label='powerlaw')
    plt.show()

    raise
    # pp=['brain.crawler_params.step_to_length_std']
    pp=['brain.crawler_params.step_to_length_mu', 'brain.crawler_params.step_to_length_std']
    cs=['Fed', 'Deprived', 'Starved']
    # # import pandas as pd
    # es=[pd.read_csv(f'{RefFolder}/{c}/data/reference.csv') for c in cs]
    import matplotlib.pyplot as plt
    # for c,e in zip(cs,es):
    #     plt.hist(e[pp[3]], bins=20, label=c, histtype='step')
    # #     # print(len(k[j][k[j] < 0]))
    # # # plt.suptitle(p)
    # plt.legend()
    # plt.show()
    # raise
    # kk = [sample_agents(pars=pp, N=2000, sample_dataset=c)[1] for c in cs]
    for c in cs :
        samples=sample_agents(pars=pp, N=2000, sample_dataset=c)[1]
        ms0, ss0 = samples[0], samples[1]
        ms1,ss1=[],[]
        for i in range(2000) :
            m,s=ms0[i],ss0[i]
            z=np.random.normal(loc=m, scale=s, size=2000)
            mm,ss=np.mean(z), np.std(z)
            ms1.append(mm)
            ss1.append(ss)
        plt.hist(ms0, bins=20, label=f'{c}_exp', histtype='step')
        plt.hist(ms1, bins=20, label=f'{c}_sim', histtype='step')
        plt.suptitle(f'{c}_mean')
        plt.legend()
        plt.show()
        plt.hist(ss0, bins=20, label=f'{c}_exp', histtype='step')
        plt.hist(ss1, bins=20, label=f'{c}_sim', histtype='step')
        plt.suptitle(f'{c}_std')
        plt.legend()
        plt.show()
        # break
    # for j,p in enumerate(pp) :
    #     print(p)
    #     for i,(k,c) in enumerate(zip(kk,cs)) :
    #         plt.hist(k[j], bins=20, label=c, histtype='step')
    #         print(len(k[j][k[j]<0]))
    #     plt.suptitle(p)
    #     plt.legend()
    #     plt.show()


