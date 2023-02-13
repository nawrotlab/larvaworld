import warnings
import numpy as np
from scipy.stats import levy, norm, rv_discrete, ks_2samp


from larvaworld.lib import reg, aux



def get_logNpow(x, xmax, xmid, overlap=0, discrete=False):
    r = len(x[x < xmid]) / len(x)
    m, s = get_lognormal(x[x < xmid + overlap * (xmax - xmid)])
    a = get_powerlaw_alpha(x[x >= xmid], xmid, xmax, discrete=discrete)
    return m, s, a, r


def get_powerlaw_alpha(dur, dur0=None, dur1=None, discrete=False):
    from powerlaw import Fit
    if dur0 is None :
        dur0=np.min(dur)
    if dur1 is None :
        dur1=np.max(dur)
    with aux.stdout.suppress_stdout_stderr():

        return Fit(dur, xmin=dur0, xmax=dur1, discrete=discrete).power_law.alpha


def get_lognormal(dur):
    d = np.log(dur)
    return np.mean(d), np.std(d)

def get_exp_beta(x, xmin=None) :
    if xmin is None :
        xmin=np.min(x)
    return len(x) / np.sum(x - xmin)

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

def KS2(a1, a2):
    if len(a1)==0 or len(a2)==0 :
        return np.nan
    else :
        return ks_2samp(a1, a2)[0]


def logNpow_switch(x, xmax, u2, du2, c2cum, c2, discrete=False, fit_by='cdf'):
    D=reg.distro_database['logNpow']
    xmids = u2[1:-int(len(u2) / 3)][::2]
    overlaps = np.linspace(0, 1, 6)
    temp = np.ones([len(xmids), len(overlaps)])
    for i, xmid in enumerate(xmids):
        for j, ov in enumerate(overlaps):
            mm, ss, aa, r = get_logNpow(x, xmax, xmid, discrete=discrete, overlap=ov)
            lp_cdf = 1 - D['cdf'](u2, mm, ss, aa, xmid, r)
            lp_pdf = D['pdf'](du2, mm, ss, aa, xmid, r)
            if fit_by == 'cdf':
                temp[i, j] = MSE(c2cum, lp_cdf)
            elif fit_by == 'pdf':
                temp[i, j] = MSE(c2, lp_pdf)

    if all(np.isnan(temp.flatten())):
        return np.nan, np.nan
    else:
        ii, jj = np.unravel_index(np.nanargmin(temp), temp.shape)
        return xmids[ii], overlaps[jj]





def fit_epochs(grouped_epochs):
    fitted = {}
    for k, v in grouped_epochs.items():
        if k == 'stridechain_length':
            k = 'run_count'
        discr = True if k == 'run_count' else False
        if v is not None and v.shape[0] > 0:
            try:
                fitted[k] = fit_bout_distros(np.abs(v), bout=k, combine=False, discrete=discr)
            except:
                fitted[k] = None
        else:
            fitted[k] = None
    return aux.AttrDict(fitted)


def get_bout_distros(fitted_epochs) :
    d={}
    for k, dic in fitted_epochs.items():
        if isinstance(dic,dict) and 'best' in dic.keys():
            d[k]=dic['best']
        else :
            d[k]=None
    return aux.AttrDict(d)




def fit_bout_distros(x0, xmin=None, xmax=None, discrete=False, xmid=np.nan, overlap=0.0, Nbins=64, print_fits=False,
                     dataset_id='dataset', bout='pause', combine=True, fit_by='pdf', eval_func_id='KS2'):
    eval_func_dic={
        'MSE':MSE,
        'KS':KS,
        'KS2':KS2,
    }
    eval_func=eval_func_dic[eval_func_id]

    if xmin is None :
        xmin=np.nanmin(x0)
    if xmax is None :
        xmax=np.nanmax(x0)
    with aux.suppress_stdout(False):
        warnings.filterwarnings('ignore')
        distros=reg.distro_database
        x = x0[x0 >= xmin]
        x = x[x <= xmax]

        u2, du2, c2, c2cum = compute_density(x, xmin, xmax, Nbins=Nbins)
        values = [u2, du2, c2, c2cum]

        a2 = 1 + len(x) / np.sum(np.log(x / xmin))
        a = get_powerlaw_alpha(x, xmin, xmax, discrete=discrete)
        p_cdf = 1 - distros['powerlaw']['cdf'](u2, xmin, a)
        p_pdf = distros['powerlaw']['pdf'](du2, xmin, a)

        b = get_exp_beta(x, xmin)
        e_cdf = 1 - distros['exponential']['cdf'](u2, xmin, b)
        e_pdf = distros['exponential']['pdf'](du2, xmin, b)

        m, s = get_lognormal(x)
        l_cdf = 1 - distros['lognormal']['cdf'](u2, m, s)
        l_pdf = distros['lognormal']['pdf'](du2, m, s)

        m_lev, s_lev = levy.fit(x)
        lev_cdf = 1 - distros['levy']['cdf'](u2, m_lev, s_lev)
        lev_pdf = distros['levy']['pdf'](du2, m_lev, s_lev)

        m_nor, s_nor = norm.fit(x)
        nor_cdf = 1 - distros['normal']['cdf'](u2, m_nor, s_nor)
        nor_pdf = distros['normal']['pdf'](du2, m_nor, s_nor)

        uni_cdf = 1 - distros['uniform']['cdf'](u2, xmin, xmin + xmax)
        uni_pdf = distros['uniform']['pdf'](du2, xmin, xmin + xmax)

        if np.isnan(xmid) and combine:
            xmid, overlap = logNpow_switch(x, xmax, u2, du2, c2cum, c2, discrete, fit_by)

        if not np.isnan(xmid):
            mm, ss, aa, r = get_logNpow(x, xmax, xmid, discrete=discrete, overlap=overlap)
            lp_cdf = 1 - distros['logNpow']['cdf'](u2, mm, ss, aa, xmid, r)
            lp_pdf = distros['logNpow']['pdf'](du2, mm, ss, aa, xmid, r)
        else:
            mm, ss, aa, r = np.nan, np.nan, np.nan, np.nan
            lp_cdf, lp_pdf = None, None

        if fit_by == 'cdf':
            KS_pow = eval_func(c2cum, p_cdf)
            KS_exp = eval_func(c2cum, e_cdf)
            KS_logn = eval_func(c2cum, l_cdf)
            KS_lognNpow = eval_func(c2cum, lp_cdf) if lp_cdf is not None else np.nan
            KS_lev = eval_func(c2cum, lev_cdf)
            KS_norm = eval_func(c2cum, nor_cdf)
            KS_uni = eval_func(c2cum, uni_cdf)
        elif fit_by == 'pdf':
            KS_pow = eval_func(c2, p_pdf)
            KS_exp = eval_func(c2, e_pdf)
            KS_logn = eval_func(c2, l_pdf)
            KS_lognNpow = eval_func(c2, lp_pdf) if lp_pdf is not None else np.nan
            KS_lev = eval_func(c2, lev_pdf)
            KS_norm = eval_func(c2, nor_pdf)
            KS_uni = eval_func(c2, uni_pdf)

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

    dic = aux.AttrDict({
        'values': values, 'pdfs': pdfs, 'cdfs': cdfs, 'Ks': Ks, 'idx_Kmax': idx_Kmax, 'res': res, 'res_dict': res_dict,
        'best': get_best_distro(p, res_dict, idx_Kmax=idx_Kmax), 'fits': dict(zip(names2, res))
    })




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

        print()
        print(f'---{dataset_id}-{bout}-distro')
        print(dic.best)
        print()


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
    else:
        raise ValueError
    return distro



def critical_bout(c=0.9, sigma=1, N=1000, tmax=1100, tmin=1) :
    t = 0
    S = 1
    S_prev = 0
    while S > 0:
        p = (sigma * S - c * (S - S_prev)) / N
        p = np.clip(p, 0, 1)
        S_prev = S
        S = np.random.binomial(N, p)
        t += 1
        if t > tmax:
            t = 0
            S = 1
            S_prev = 0
        if S<=0 and t<tmin :
            t = 0
            S = 1
            S_prev = 0
    return t

def exp_bout(beta=0.01, tmax=1100, tmin=1) :
    t = 0
    S = 0
    while S <= 0:
        S = int(np.random.rand() < beta)
        t += 1
        if t > tmax:
            t = 0
            S = 0
        if S>0 and t<tmin :
            t = 0
            S = 0
    return t

class BoutGenerator:
    def __init__(self, name, range, dt, **kwargs):
        self.name = name
        self.dt = dt
        self.range = range
        self.ddfs = reg.distro_database
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


def test_boutGens(mID,refID=None,refDataset=None, **kwargs):
    if refDataset is None :
        refDataset=reg.loadRef(refID, load=True)
    c=refDataset.config
    chunk_dicts = refDataset.load_chunk_dicts()
    aux_dic = aux.group_epoch_dicts(chunk_dicts)
    Npau = aux_dic['pause_dur'].shape[0]
    Nrun = aux_dic['run_dur'].shape[0]

    from larvaworld.lib.util.sampling import get_sample_bout_distros
    m=reg.loadConf(id=mID, conftype='Model')
    m=get_sample_bout_distros(m, c)
    dicM=m.brain.intermitter_params
    dic = {}
    for n,n0 in zip(['pause', 'exec', 'stridechain'], ['pause_dur', 'run_dur', 'run_count']) :
        N=Npau if n == 'pause' else Nrun
        discr = True if n == 'stridechain' else False
        dt = 1 if n == 'stridechain' else c.dt

        k=f'{n}_dist'
        kk=dicM[k]
        if kk is not None :
            B = BoutGenerator(**kk, dt=dt)
            vs = B.sample(N)
            dic[n0] = fit_bout_distros(vs, dataset_id=mID, bout=n, combine=False, discrete=discr)
    datasets=[{'id' : 'model', 'pooled_epochs': dic, 'color': 'blue'},
              {'id' : 'experiment', 'pooled_epochs': refDataset.load_pooled_epochs(), 'color': 'red'}]
    datasets = [aux.AttrDict(dd) for dd in datasets]
    return datasets

