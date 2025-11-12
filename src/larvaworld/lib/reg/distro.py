"""
Distribution database, registry and associated methods.
This modules provides classes and methods for managing and generating distributions.
"""

from __future__ import annotations
from typing import Any, Optional
import typing
import os
import warnings

# Deprecation: discourage deep imports from internal registry internals
if os.getenv("LARVAWORLD_STRICT_DEPRECATIONS") == "1":
    raise ImportError(
        "Deep import path deprecated. Access registry via 'from larvaworld.lib import reg'"
    )
else:
    warnings.warn(
        "Deep import path deprecated. Access registry via 'from larvaworld.lib import reg'",
        DeprecationWarning,
        stacklevel=2,
    )

import numpy as np
import powerlaw
import scipy
from scipy.stats import ks_2samp, levy, norm, rv_discrete

from .. import reg, util
from ..util import nam

__all__: list[str] = [
    "distroDB",
    "get_dist",
    "fit_bout_distros",
    "BoutGenerator",
]


def powerlaw_cdf(x, xmin, alpha):
    return 1 - (x / xmin) ** (1 - alpha)


def powerlaw_pdf(x, xmin, alpha):
    return (alpha - 1) / xmin * (x / xmin) ** (-alpha)


def levy_pdf(x, mu, sigma):
    return (
        np.sqrt(sigma / (2 * np.pi)) * np.exp(-sigma / (2 * (x - mu))) / (x - mu) ** 1.5
    )


def levy_cdf(x, mu, sigma):
    res = 1 - scipy.special.erf(np.sqrt(sigma / (2 * (x - mu))))
    if np.isnan(res[0]):
        res[0] = 0
    return res


def norm_pdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def norm_cdf(x, mu, sigma):
    res = 0.5 * (1 + scipy.special.erf((x - mu) / (sigma * np.sqrt(2))))
    if np.isnan(res[0]):
        res[0] = 0
    return res


def uniform_pdf(x, xmin, xmax):
    return scipy.stats.uniform.pdf(x, xmin, xmin + xmax)


def uniform_cdf(x, xmin, xmax):
    return scipy.stats.uniform.cdf(x, xmin, xmin + xmax)


def exponential_cdf(x, xmin, beta):
    return 1 - np.exp(-beta * (x - xmin))


def exponential_pdf(x, xmin, beta):
    return beta * np.exp(-beta * (x - xmin))


def lognorm_cdf(x, mu, sigma):
    return 0.5 + 0.5 * scipy.special.erf((np.log(x) - mu) / np.sqrt(2) / sigma)


def lognormal_pdf(x, mu, sigma):
    return (
        1
        / (x * sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((np.log(x) - mu) / sigma) ** 2)
    )


def logNpow_pdf(x, mu, sigma, alpha, switch, ratio):
    log_pdf = lognormal_pdf(x[x < switch], mu, sigma) * ratio
    pow_pdf = powerlaw_pdf(x[x >= switch], switch, alpha) * (1 - ratio)
    return np.hstack([log_pdf, pow_pdf])


def logNpow_cdf(x, mu, sigma, alpha, switch, ratio):
    log_cdf = 1 - lognorm_cdf(x[x < switch], mu, sigma)
    pow_cdf = (1 - powerlaw_cdf(x[x >= switch], switch, alpha)) * (1 - ratio)
    return 1 - np.hstack([log_cdf, pow_cdf])


def get_powerlaw_alpha2(x, xmin=None, xmax=None, discrete=False):
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    with util.suppress_stdout_stderr():
        a = powerlaw.Fit(x, xmin=xmin, xmax=xmax, discrete=discrete).power_law.alpha
        return util.AttrDict({"xmin": xmin, "alpha": a})


def get_exp_beta2(x, xmin=None):
    if xmin is None:
        xmin = np.min(x)
    b = len(x) / np.sum(x - xmin)
    return {"xmin": xmin, "beta": b}


def fit_levy(x):
    m, s = scipy.stats.levy.fit(x)
    return {"mu": m, "sigma": s}


def fit_norm(x):
    m, s = scipy.stats.norm.fit(x)
    return {"mu": m, "sigma": s}


def fit_uni(x, xmin=None, xmax=None):
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.np.max(x)
    return {"xmin": xmin, "xmax": xmax}


def get_logNpow2(x, xmax, xmid, overlap=0, discrete=False):
    dic = util.AttrDict()
    dic.ratio = len(x[x < xmid]) / len(x)
    xx = np.log(x[x < xmid + overlap * (xmax - xmid)])
    dic.mu = np.mean(xx)
    dic.sigma = np.std(xx)
    dic.switch = xmid
    with util.suppress_stdout_stderr():
        dic.alpha = powerlaw.Fit(
            x=x[x >= xmid], xmin=xmid, xmax=xmax, discrete=discrete
        ).power_law.alpha

        return dic


def generate_distro_database():
    """
    Generates a dictionary of distribution configurations.
    Each key in the dictionary corresponds to a distribution name, and the value is another dictionary containing:
    - 'cdf': The cumulative probability density function for the distribution.
    - 'pdf': The probability density function for the distribution.
    - 'args': A list of argument names required by the distribution.
    - 'lab_func': A lambda function that generates a label for the distribution given its parameters.
    - 'func': A lambda function that fits the distribution to data or computes distribution parameters.

    Returns:
        dict: A dictionary where each key is a distribution name and the value is a dictionary of distribution properties.

    """
    d = util.AttrDict(
        {
            "powerlaw": {
                "cdf": powerlaw_cdf,
                "pdf": powerlaw_pdf,
                "args": ["xmin", "alpha"],
                "lab_func": lambda v: f"Powerlaw(a={np.round(v.alpha, 2)})",
                "func": lambda x,
                xmin=None,
                xmax=None,
                discrete=False: get_powerlaw_alpha2(x, xmin, xmax, discrete),
            },
            "exponential": {
                "cdf": exponential_cdf,
                "pdf": exponential_pdf,
                "args": ["xmin", "beta"],
                "lab_func": lambda v: f"Exp(b={np.round(v.beta, 2)})",
                "func": lambda x, xmin=None: get_exp_beta2(x, xmin),
            },
            "lognormal": {
                "cdf": lognorm_cdf,
                "pdf": lognormal_pdf,
                "args": ["mu", "sigma"],
                "lab_func": lambda v: f"Lognormal(m={np.round(v.mu, 2)}, s={np.round(v.sigma, 2)})",
                "func": lambda x: {
                    "mu": np.mean(np.log(x)),
                    "sigma": np.std(np.log(x)),
                },
            },
            "logNpow": {
                "cdf": logNpow_cdf,
                "pdf": logNpow_pdf,
                "args": ["alpha", "mu", "sigma", "switch", "ratio"],
                "lab_func": lambda v: f"Lognormal-Powerlaw(a={np.round(v.alpha, 2)}, m={np.round(v.mu, 2)}, s={np.round(v.sigma, 2)})",
                "func": lambda x, xmax, xmid, overlap=0, discrete=False: get_logNpow2(
                    x, xmax, xmid, overlap, discrete
                ),
            },
            "levy": {
                "cdf": levy_cdf,
                "pdf": levy_pdf,
                "args": ["mu", "sigma"],
                "lab_func": lambda v: f"Levy(m={np.round(v.mu, 2)}, s={np.round(v.sigma, 2)})",
                "func": lambda x: fit_levy(x),
            },
            "normal": {
                "cdf": norm_cdf,
                "pdf": norm_pdf,
                "args": ["mu", "sigma"],
                "lab_func": lambda v: f"N(m={np.round(v.mu, 2)}, s={np.round(v.sigma, 2)})",
                "func": lambda x: fit_norm(x),
            },
            "uniform": {
                "cdf": uniform_cdf,
                "pdf": uniform_pdf,
                "args": ["xmin", "xmax"],
                "lab_func": lambda v: "Uniform()",
                "func": lambda x, xmin=None, xmax=None: fit_uni(x, xmin, xmax),
            },
        }
    )
    return d


#: Distribution database containing all predefined bout duration distributions.
#:
#: Contains configurations for behavioral bout distributions (pause, run, stridechain)
#: including powerlaw, exponential, lognormal, levy, and combined distributions.
#:
#: Example:
#:     >>> distroDB['powerlaw']['args']
#:     ['alpha']
distroDB: util.AttrDict = generate_distro_database()


def get_dist(
    k: str,
    k0: str = "intermitter",
    v: Any = None,
    return_tabrows: bool = False,
    return_all: bool = False,
) -> dict | tuple:
    """
    Retrieve a distribution from the database.

    Args:
        k: Key to identify the distribution
        k0: Module key for the distribution. Defaults to 'intermitter'
        v: An object containing distribution details. Defaults to None
        return_tabrows: If True, returns table rows. Defaults to False
        return_all: If True, returns all distribution details. Defaults to False

    Returns:
        dict: A dictionary containing distribution details if return_tabrows and return_all are False.
        tuple: Two lists of table rows if return_tabrows is True.
        tuple: Three dictionaries containing distribution details if return_all is True.

    Example:
        >>> dist_info = get_dist('pause_dist', k0='intermitter')
        >>> table_rows = get_dist('pause_dist', return_tabrows=True)
    """
    dict0 = {
        "stridechain_dist": (
            "exec length",
            ("N", "R"),
            reg.units.dimensionless,
            "# $strides$",
        ),
        "pause_dist": ("pause duration", ("t", "P"), reg.units.s, "$sec$"),
        "run_dist": ("exec duration", ("t", "R"), reg.units.s, "$sec$"),
    }
    disp, (tt0, tt1), u, uname = dict0[k]
    dispD, dispR = f"{disp} distribution", f"{disp} range"
    symD = nam.tex.sub(tt0, tt1)
    kD = f"{tt0}_{tt1}"
    kR = f"{kD}_r"
    sym1, sym2 = nam.tex.subsup(tt0, tt1, "min"), nam.tex.subsup(tt0, tt1, "max")
    symR = f"[{sym1},{sym2}]"
    p = {
        "disp": disp,
        "k": kD,
        "sym": symD,
        "u_name": uname,
        "u": u,
        "dtype": dict,
        "v0": {"fit": True, "name": None, "range": None},
    }

    if return_tabrows:
        dist_v = distroDB[v.name].lab_func(v)
        vs1 = [k0, dispD, symD, dist_v, "-"]
        vs2 = [k0, dispR, symR, v.range, uname]
        return vs1, vs2
    elif return_all:
        pD = {
            "disp": dispD,
            "k": kD,
            "v0": None,
            "vs": list(distroDB.keys()),
            "sym": symD,
            "dtype": str,
        }
        pR = {
            "disp": dispR,
            "k": kR,
            "u_name": uname,
            "u": u,
            "sym": symR,
            "v0": None,
            "dtype": typing.Tuple[float],
        }
        return p, pD, pR
    else:
        return p


def fit_bout_distros(
    x0: np.ndarray,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    discrete: bool = False,
    xmid: float = np.nan,
    overlap: float = 0.0,
    Nbins: int = 64,
    print_fits: bool = False,
    bout: str = "pause",
    combine: bool = True,
    fit_by: str = "pdf",
    eval_func_id: str = "KS2",
) -> util.AttrDict:
    """
    Fit various distributions to data and evaluate goodness of fit.

    Fits multiple distribution types (powerlaw, exponential, lognormal, lognormal-powerlaw,
    levy, normal, uniform) to the given data and evaluates their goodness of fit using
    the specified evaluation function.

    Args:
        x0: The data to fit the distributions to
        xmin: Minimum value to consider for fitting. If None, set to minimum of x0
        xmax: Maximum value to consider for fitting. If None, set to maximum of x0
        discrete: Whether the data is discrete. Defaults to False
        xmid: Midpoint for lognormal-powerlaw distribution. If NaN, determined automatically
        overlap: Overlap parameter for lognormal-powerlaw distribution. Defaults to 0.0
        Nbins: Number of bins to use for density computation. Defaults to 64
        print_fits: Whether to print the fit results. Defaults to False
        bout: Label for the bout type. Defaults to 'pause'
        combine: Whether to combine distributions. Defaults to True
        fit_by: Criterion to fit by ('pdf' or 'cdf'). Defaults to 'pdf'
        eval_func_id: Evaluation function identifier ('MSE', 'KS', 'KS2'). Defaults to 'KS2'

    Returns:
        util.AttrDict: Dictionary containing fit results, including:
            - values: Computed density values
            - pdfs: Probability density functions of the fitted distributions
            - cdfs: Cumulative density functions of the fitted distributions
            - Ks: Goodness of fit values
            - idx_Kmax: Index of the best fitting distribution
            - res: Rounded fit parameters
            - best: Dictionary of the best fitting distribution parameters
            - fits: Dictionary of all fit parameters

    Note:
        This function fits several distributions (powerlaw, exponential, lognormal,
        lognormal-powerlaw, levy, normal, uniform) to the given data and evaluates
        their goodness of fit using the specified evaluation function.

    Example:
        >>> data = np.random.exponential(2.0, 1000)
        >>> results = fit_bout_distros(data, xmin=0.1, xmax=10.0, bout='pause')
        >>> best_fit = results['best']
    """

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
        if len(a1) == 0 or len(a2) == 0:
            return np.nan
        else:
            return ks_2samp(a1, a2)[0]

    eval_func_dic = {
        "MSE": MSE,
        "KS": KS,
        "KS2": KS2,
    }
    F = eval_func_dic[eval_func_id]

    if xmin is None:
        xmin = np.nanmin(x0)
    if xmax is None:
        xmax = np.nanmax(x0)
    with util.suppress_stdout(False):
        warnings.filterwarnings("ignore")

        def get_powerlaw_alpha(dur, dur0=None, dur1=None, discrete=False):
            from powerlaw import Fit

            if dur0 is None:
                dur0 = np.min(dur)
            if dur1 is None:
                dur1 = np.max(dur)
            with util.stdout.suppress_stdout_stderr():
                return Fit(dur, xmin=dur0, xmax=dur1, discrete=discrete).power_law.alpha

        def get_lognormal(dur):
            d = np.log(dur)
            return np.mean(d), np.std(d)

        def get_logNpow(x, xmax, xmid, overlap=0, discrete=False):
            r = len(x[x < xmid]) / len(x)
            m, s = get_lognormal(x[x < xmid + overlap * (xmax - xmid)])
            a = get_powerlaw_alpha(x[x >= xmid], xmid, xmax, discrete=discrete)
            return m, s, a, r

        D = distroDB
        x = x0[x0 >= xmin]
        x = x[x <= xmax]

        u2, du2, c2, c2cum = compute_density(x, xmin, xmax, Nbins=Nbins)

        a2 = 1 + len(x) / np.sum(np.log(x / xmin))
        a = get_powerlaw_alpha(x, xmin, xmax, discrete=discrete)
        p_cdf = 1 - D.powerlaw["cdf"](u2, xmin, a)
        p_pdf = D.powerlaw["pdf"](du2, xmin, a)

        b = len(x) / np.sum(x - xmin)
        e_cdf = 1 - D.exponential["cdf"](u2, xmin, b)
        e_pdf = D.exponential["pdf"](du2, xmin, b)

        m, s = get_lognormal(x)
        l_cdf = 1 - D.lognormal["cdf"](u2, m, s)
        l_pdf = D.lognormal["pdf"](du2, m, s)

        m_lev, s_lev = levy.fit(x)
        lev_cdf = 1 - D.levy["cdf"](u2, m_lev, s_lev)
        lev_pdf = D.levy["pdf"](du2, m_lev, s_lev)

        m_nor, s_nor = norm.fit(x)
        nor_cdf = 1 - D.normal["cdf"](u2, m_nor, s_nor)
        nor_pdf = D.normal["pdf"](du2, m_nor, s_nor)

        uni_cdf = 1 - D.uniform["cdf"](u2, xmin, xmin + xmax)
        uni_pdf = D.uniform["pdf"](du2, xmin, xmin + xmax)

        if np.isnan(xmid) and combine:

            def logNpow_switch(
                x, xmax, u2, du2, c2cum, c2, discrete=False, fit_by="cdf"
            ):
                xmids = u2[1 : -int(len(u2) / 3)][::2]
                overlaps = np.linspace(0, 1, 6)
                temp = np.ones([len(xmids), len(overlaps)])
                for i, xmid in enumerate(xmids):
                    for j, ov in enumerate(overlaps):
                        mm, ss, aa, r = get_logNpow(
                            x, xmax, xmid, discrete=discrete, overlap=ov
                        )
                        lp_cdf = 1 - D.logNpow["cdf"](u2, mm, ss, aa, xmid, r)
                        lp_pdf = D.logNpow["pdf"](du2, mm, ss, aa, xmid, r)
                        if fit_by == "cdf":
                            temp[i, j] = MSE(c2cum, lp_cdf)
                        elif fit_by == "pdf":
                            temp[i, j] = MSE(c2, lp_pdf)

                if all(np.isnan(temp.flatten())):
                    return np.nan, np.nan
                else:
                    ii, jj = np.unravel_index(np.nanargmin(temp), temp.shape)
                    return xmids[ii], overlaps[jj]

            xmid, overlap = logNpow_switch(
                x, xmax, u2, du2, c2cum, c2, discrete, fit_by
            )

        if not np.isnan(xmid):
            mm, ss, aa, r = get_logNpow(
                x, xmax, xmid, discrete=discrete, overlap=overlap
            )
            lp_cdf = 1 - D.logNpow["cdf"](u2, mm, ss, aa, xmid, r)
            lp_pdf = D.logNpow["pdf"](du2, mm, ss, aa, xmid, r)
        else:
            mm, ss, aa, r = np.nan, np.nan, np.nan, np.nan
            lp_cdf, lp_pdf = None, None
        pdfs = [p_pdf, e_pdf, l_pdf, lp_pdf, lev_pdf, nor_pdf, uni_pdf]
        cdfs = [p_cdf, e_cdf, l_cdf, lp_cdf, lev_cdf, nor_cdf, uni_cdf]

        if fit_by == "cdf":
            Ks = np.array(
                [F(c2cum, cdf0) if cdf0 is not None else np.nan for cdf0 in cdfs]
            )
        elif fit_by == "pdf":
            Ks = np.array(
                [F(c2, pdf0) if pdf0 is not None else np.nan for pdf0 in pdfs]
            )

        idx_Kmax = np.nanargmin(Ks)
        KS_pow, KS_exp, KS_logn, KS_lognNpow, KS_lev, KS_norm, KS_uni = Ks
        res = np.round(
            [
                a,
                KS_pow,
                b,
                KS_exp,
                m,
                s,
                KS_logn,
                mm,
                ss,
                aa,
                xmid,
                r,
                overlap,
                KS_lognNpow,
                m_lev,
                s_lev,
                KS_lev,
                m_nor,
                s_nor,
                KS_norm,
                KS_uni,
                xmin,
                xmax,
            ],
            5,
        )

    p = bout

    names = [
        f"alpha_{p}",
        f"KS_pow_{p}",
        f"beta_{p}",
        f"KS_exp_{p}",
        f"mu_log_{p}",
        f"sigma_log_{p}",
        f"KS_log_{p}",
        f"mu_logNpow_{p}",
        f"sigma_logNpow_{p}",
        f"alpha_logNpow_{p}",
        f"switch_logNpow_{p}",
        f"ratio_logNpow_{p}",
        f"overlap_logNpow_{p}",
        f"KS_logNpow_{p}",
        f"mu_levy_{p}",
        f"sigma_levy_{p}",
        f"KS_levy_{p}",
        f"mu_norm_{p}",
        f"sigma_norm_{p}",
        f"KS_norm_{p}",
        f"KS_uni_{p}",
        f"min_{p}",
        f"max_{p}",
    ]

    names2 = [
        "alpha",
        "KS_pow",
        "beta",
        "KS_exp",
        "mu_log",
        "sigma_log",
        "KS_log",
        "mu_logNpow",
        "sigma_logNpow",
        "alpha_logNpow",
        "switch_logNpow",
        "ratio_logNpow",
        "overlap_logNpow",
        "KS_logNpow",
        "mu_levy",
        "sigma_levy",
        "KS_levy",
        "mu_norm",
        "sigma_norm",
        "KS_norm",
        "KS_uni",
        "xmin",
        "xmax",
    ]

    def get_best_distro(bout, f, idx_Kmax=None):
        k = bout
        r = (f[f"min_{k}"], f[f"max_{k}"])
        if idx_Kmax is None:
            idx_Kmax = np.argmin(
                [
                    f[f"KS_{d}_{k}"]
                    for d in ["pow", "exp", "log", "logNpow", "levy", "norm", "uni"]
                ]
            )
        if idx_Kmax == 0:
            return {"range": r, "name": "powerlaw", "alpha": f[f"alpha_{k}"]}
        elif idx_Kmax == 1:
            return {"range": r, "name": "exponential", "beta": f[f"beta_{k}"]}
        elif idx_Kmax == 2:
            return {
                "range": r,
                "name": "lognormal",
                "mu": f[f"mu_log_{k}"],
                "sigma": f[f"sigma_log_{k}"],
            }
        elif idx_Kmax == 3:
            n = "logNpow"
            return {
                "range": r,
                "name": n,
                "mu": f[f"mu_{n}_{k}"],
                "sigma": f[f"sigma_{n}_{k}"],
                "alpha": f[f"alpha_{n}_{k}"],
                "switch": f[f"switch_{n}_{k}"],
                "ratio": f[f"ratio_{n}_{k}"],
                "overlap": f[f"overlap_{n}_{k}"],
            }
        elif idx_Kmax == 4:
            return {
                "range": r,
                "name": "levy",
                "mu": f[f"mu_levy_{k}"],
                "sigma": f[f"sigma_levy_{k}"],
            }
        elif idx_Kmax == 5:
            return {
                "range": r,
                "name": "normal",
                "mu": f[f"mu_norm_{k}"],
                "sigma": f[f"sigma_norm_{k}"],
            }
        elif idx_Kmax == 6:
            return {"range": r, "name": "uniform"}
        else:
            raise ValueError

    dic = util.AttrDict(
        {
            "values": [u2, du2, c2, c2cum],
            "pdfs": pdfs,
            "cdfs": cdfs,
            "Ks": Ks,
            "idx_Kmax": idx_Kmax,
            "res": res,
            "best": get_best_distro(p, dict(zip(names, res)), idx_Kmax=idx_Kmax),
            "fits": dict(zip(names2, res)),
        }
    )

    if print_fits:
        print()
        print(f"-----{bout}-epochs---------")
        print(f"initial range : {np.min(x0)} - {np.max(x0)}, Nbouts : {len(x0)}")
        print(f"accepted range : {xmin} - {xmax}, Nbouts : {len(x)}")
        print("powerlaw exponent MLE:", a2)
        print("powerlaw exponent powerlaw package:", a)
        print("exponential exponent MLE:", b)
        print("lognormal mean,std:", m, s)
        print(
            "lognormal-powerlaw mean,std, alpha, switch, ratio, overlap :",
            mm,
            ss,
            aa,
            xmid,
            r,
            overlap,
        )
        print("levy loc,scale:", m_lev, s_lev)
        print("normal loc,scale:", m_nor, s_nor)
        print("MSE pow", KS_pow)
        print("MSE exp", KS_exp)
        print("MSE logn", KS_logn)
        print("MSE lognNpow", KS_lognNpow)
        print("MSE levy", KS_lev)
        print("MSE normal", KS_norm)
        print("MSE uniform", KS_uni)

        print()
        print(f"---{bout} epochs distro---")
        print(dic.best)
        print()

    return dic


class BoutGenerator:
    """
    Generator for behavioral bout durations from statistical distributions.

    Creates random bout durations (pause, run, stridechain) based on specified
    distribution types from the distribution database. Supports powerlaw,
    exponential, lognormal, levy, and other distributions.

    Attributes:
        name: Distribution name from distroDB
        dt: Time step for converting to real time units
        range: Valid range as (xmin, xmax) tuple
        args: Distribution-specific arguments
        dist: Built scipy.stats distribution object

    Example:
        >>> gen = BoutGenerator('exponential', range=(0.1, 10.0), dt=0.1, beta=0.5)
        >>> duration = gen.sample()  # Single bout duration
        >>> durations = gen.sample(size=100)  # 100 bout durations
    """

    def __init__(
        self, name: str, range: tuple[float, float], dt: float, **kwargs: Any
    ) -> None:
        self.name = name
        self.dt = dt
        self.range = range
        self.xmin, self.xmax = range
        kwargs.update({"xmin": self.xmin, "xmax": self.xmax})
        self.args = {a: kwargs[a] for a in distroDB[self.name]["args"]}

        self.dist = self.build(**self.args)

    def sample(self, size: int = 1):
        vs = self.dist.rvs(size=size) * self.dt
        return vs[0] if size == 1 else vs

    def build(self, **kwargs: Any):
        x0, x1 = int(self.xmin / self.dt), int(self.xmax / self.dt)
        xx = np.arange(x0, x1 + 1)
        pmf = distroDB[self.name]["pdf"](xx * self.dt, **kwargs)
        mask = ~np.isnan(pmf)
        pmf = pmf[mask]
        xx = xx[mask]
        pmf /= pmf.sum()
        return rv_discrete(values=(xx, pmf))

    def get(self, x: np.ndarray, mode: str):
        func = distroDB[self.name][mode]
        return func(x=x, **self.args)
