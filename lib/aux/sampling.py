import scipy.integrate as integrate
import scipy.stats as st
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import truncnorm

from lib.anal.fitting import compute_density, power_cdf, exp_cdf, lognorm_cdf, powerlaw_pdf, logNpow_pdf, \
    fit_bout_distros, logNpow_cdf
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
    if filepath is None:
        filepath = path_data
    data = pd.read_csv(filepath, index_col=0)
    pars = data.columns if pars is None else [p for p in data.columns if p in pars]
    means = [data[p].mean() for p in pars]

    if len(pars)>=2:
        base=data[pars].values.T
        cov = np.cov(base)
        samples = np.random.multivariate_normal(means, cov, N).T
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
        k = 'stride'
        str_i = np.argmin(f[['KS_pow_stride', 'KS_exp_stride', 'KS_log_stride', 'KS_logNpow_stride']])
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
        elif str_i == 3:
            str_dist = {'range': (f['min_stride'], f['max_stride']),
                        'name': 'logNpow',
                        'mu': f['mu_logNpow_stride'],
                        'sigma': f['sigma_logNpow_stride'],
                        'alpha': f['alpha_logNpow_stride'],
                        'switch': f['switch_logNpow_stride'],
                        'ratio': f[f'ratio_logNpow_{k}'],
                        'overlap': f[f'overlap_logNpow_{k}'],
                        }
        return str_dist

    elif mode=='pause_dist' :
        k='pause'
        pau_i = np.argmin(f[['KS_pow_pause', 'KS_exp_pause', 'KS_log_pause', 'KS_logNpow_pause']])
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
        elif pau_i == 3:
            pau_dist = {'range': (f[f'min_{k}'], f[f'max_{k}']),
                        'name': 'logNpow',
                        'mu': f[f'mu_logNpow_{k}'],
                        'sigma': f[f'sigma_logNpow_{k}'],
                        'alpha': f[f'alpha_logNpow_{k}'],
                        'switch': f[f'switch_logNpow_{k}'],
                        'ratio': f[f'ratio_logNpow_{k}'],
                        'overlap': f[f'overlap_logNpow_{k}'],
                        }
        # print(pau_dist, sample_dataset)
        # raise
        return pau_dist


def logNpow_distro(a, xmin, xmax, m,s,xmid,r, dt, overlap=0) :
    x0, x1 = int(xmin/ dt), int(xmax/ dt)
    xx = np.arange(x0,x1)
    # xx = (x / dt).astype(int)
    x=xx*dt
    # x=np.arange(xmin,xmax,dt)
    pmf=logNpow_pdf(x,m,s, a,xmin, xmid, r, overlap)
    pmf /= pmf.sum()

    # x0,x1=xx[0],xx[-1]
    # print(len(xx), len(np.unique(xx)))
    # print(len(pmf), len(np.unique(pmf)))
    # raise
    return stats.rv_discrete(values=(range(x0, x1), pmf))


def truncated_power_law(a, xmin, xmax):
    x = np.arange(xmin, xmax + 1, dtype='float')
    pmf = 1 / x ** a
    # pmf =(a - 1) / xmin * (x / xmin) ** (-a)
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

if __name__ == '__main__':
    import numpy
    from matplotlib import pyplot as plt

    import powerlaw as pow
    import matplotlib.pyplot as plt

    # Nbins = 256
    fr=11.27
    dt=1/fr
    a, dur0, dur1=3.194, 0.1,19.43
    m, s, xmid, r = -1.131,0.521, 1.898, 0.985
    # a, dur0, dur1=3.2, 1.5,90.0
    # dur0_t, dur1_t = int(dur0/dt),int(dur1/dt)
    # pau_dist=truncated_power_law(a,dur0_t, dur1_t)
    pau_dist=logNpow_distro(a, dur0, dur1, m,s,xmid,r, dt)
    dur_t=pau_dist.rvs(size=10000)
    dur=dur_t * dt
    values, pdfs,cdfs, Ks, idx_Kmax,res = fit_bout_distros(dur, xmin=0.1, xmax=20, fr=fr)
    u2, du2, c2, c2cum = values
    p_cdf, e_cdf, l_cdf, lp_cdf= cdfs
    # raise
    # aa = 1 + len(dur) / np.sum(np.log(dur / dur0))
    # aa_t = 1 + len(dur_t) / np.sum(np.log(dur_t / dur0_t))
    # adur0,adur1=np.min(dur), np.max(dur)
    # adur0_t,adur1_t=np.min(dur_t), np.max(dur_t)
    # aa2_t = pow.Fit(dur_t, xmin=dur0_t, xmax=dur1_t, discrete=True).power_law.alpha
    # aa2 = pow.Fit(np.array(dur * fr).astype(int), xmin=int(dur0 * fr), xmax=int(dur1 * fr), discrete=True).power_law.alpha
    #
    # u2,du2, c2, c2cum = compute_density(dur, dur0, dur1, Nbins=Nbins)
    # u2_t,du2_t, c2_t, c2cum_t = compute_density(dur_t, dur0_t, dur1_t, Nbins=Nbins)
    # td=load_reference_dataset('Starved')
    # vs=td.get_par('pause_dur')
    lp_cdf0=1-logNpow_cdf(u2,m, s,a, dur0, xmid, r)


    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs=axs.ravel()
    axs[0].loglog(u2, c2cum, '.', color='red', alpha=0.7)
    axs[0].loglog(u2, lp_cdf, 'c', lw=2, label='powerlaw')
    axs[0].loglog(u2, lp_cdf0, 'r', lw=2, label='powerlaw')
    # axs[1].loglog(du2, c2, '.', color='red', alpha=0.7)
    # axs[1].loglog(du2, powerlaw_pdf(du2, dur0, aa2), 'c', lw=2, label='powerlaw')

    # axs[2].loglog(u2_t, c2cum_t, '.', color='red', alpha=0.7)
    # axs[2].loglog(u2_t, 1 - power_cdf(u2_t, dur0_t, aa2_t), 'c', lw=2, label='powerlaw')
    # axs[3].loglog(du2_t, c2_t, '.', color='red', alpha=0.7)
    # axs[3].loglog(du2_t, powerlaw_pdf(du2_t, dur0_t, aa2_t), 'c', lw=2, label='powerlaw')

    # axs[4].set_xticks(bins, ["2^%s" % i for i in bins])
    # ys,xs,pol=axs[5].hist(numpy.log10(dur), log=True, bins=np.linspace(np.log10(dur0), np.log10(dur1), Nbins), histtype='step',alpha=0)
    # ys,xs,pol=axs[5].hist(numpy.log10(dur), log=True, bins=np.linspace(np.log10(dur0), np.log10(dur1), Nbins), histtype='step',alpha=0)
    # xs=0.5 * (xs[:-1] + xs[1:])
    # axs[5].plot(xs,ys)
    # cum_ys = 1-np.cumsum(ys)/np.sum(ys)
    # cum_ys = np.array([np.cumsum(ys[:i+1]) for i,(x,y) in enumerate(zip(xs,ys))])
    # print(xs, du2)
    # axs[4].loglog(du2, cum_ys, '.', color='red', alpha=0.7)
    # axs[4].loglog(u2, 1 - power_cdf(u2, dur0, aa2), 'c', lw=2, label='powerlaw')
    # axs[4].scatter(xs, np.log10(cum_ys))
    # axs[4].set_yticks([])
    # axs[5].hist(numpy.log10(dur), log=True, bins=np.linspace(np.log10(dur0), np.log10(dur1), Nbins), histtype='step')
    # ys,xs,pol = axs[7].hist(numpy.log10(dur_t), log=True, bins=np.linspace(np.log10(dur0_t), np.log10(dur1_t), Nbins), histtype='step',alpha=0)
    # xs = 0.5 * (xs[:-1] + xs[1:])
    # axs[7].plot(xs, ys)
    # cum_ys = 1 - np.cumsum(ys) / np.sum(ys)
    # axs[6].loglog(du2_t, cum_ys, '.', color='red', alpha=0.7)
    # axs[6].loglog(u2_t, 1 - power_cdf(u2_t, dur0_t, aa2_t), 'c', lw=2, label='powerlaw')

    plt.show()
    # raise

    # print(a,dur0,dur1)
    # print(aa,aa2, adur0,adur1)


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