"""
Parameter-computing functions.
Contains functions that compute/derive higher-order parameters from the existing ones in the dataset.
"""

import copy

import numpy as np

from .. import util, funcs
from ..util import nam

__all__ = [
    "track_par_func",
    "chunk_func",
    "dsp_func",
    "tor_func",
    "mean_func",
    "std_func",
    "var_func",
    "min_func",
    "max_func",
    "fin_func",
    "init_func",
    "cum_func",
    "freq_func",
    "tr_func",
    "unwrap_func",
    "dst_func",
    "func_v_spatial",
]


@funcs.param("track_par")
def track_par_func(chunk, par):
    def func(d):
        d.track_par_in_chunk(chunk, par)

    return func


@funcs.param("chunk")
def chunk_func(kc):
    if kc in ["str", "pau", "exec", "str_c", "run"]:

        def func(d):
            d.crawl_annotation()

        required_ks = ["a", "sa", "ba", "foa", "fv"]
    elif kc in ["tur", "Ltur", "Rtur"]:

        def func(d):
            d.turn_annotation()

        required_ks = ["fov"]
    else:
        func = None
        required_ks = []
    return util.AttrDict({"func": func, "required_ks": required_ks})


@funcs.param("dsp")
def dsp_func(range):
    r0, r1 = range

    def func(d):
        d.comp_dispersal(r0, r1)

    return func


@funcs.param("tor")
def tor_func(dur):
    def func(d):
        d.comp_tortuosity(dur)

    return func


@funcs.param("mean")
def mean_func(par):
    def func(d):
        d.e[nam.mean(par)] = d.s[par].dropna().groupby("AgentID").mean()

    return func


@funcs.param("std")
def std_func(par):
    def func(d):
        d.e[nam.std(par)] = d.s[par].dropna().groupby("AgentID").std()

    return func


@funcs.param("var")
def var_func(par):
    def func(d):
        d.e[nam.var(par)] = (
            d.s[par].dropna().groupby("AgentID").mean()
            / d.s[par].dropna().groupby("AgentID").std()
        )

    return func


@funcs.param("min")
def min_func(par):
    def func(d):
        d.e[nam.min(par)] = d.s[par].dropna().groupby("AgentID").min()

    return func


@funcs.param("max")
def max_func(par):
    def func(d):
        d.e[nam.max(par)] = d.s[par].dropna().groupby("AgentID").max()

    return func


@funcs.param("final")
def fin_func(par):
    def func(d):
        d.e[nam.final(par)] = d.s[par].dropna().groupby("AgentID").last()

    return func


@funcs.param("initial")
def init_func(par):
    def func(d):
        d.e[nam.initial(par)] = d.s[par].dropna().groupby("AgentID").first()

    return func


@funcs.param("cum")
def cum_func(par):
    def func(d):
        d.e[nam.cum(par)] = d.s[par].dropna().groupby("AgentID").sum()

    return func


@funcs.param("freq")
def freq_func(par):
    def func(d):
        d.comp_freq(par=par, fr_range=(0.0, +np.inf))

    return func


@funcs.param("tr")
def tr_func(pc):
    def func(d):
        d.e[nam.dur_ratio(pc)] = d.e[nam.cum(nam.dur(pc))] / d.e[nam.cum(nam.dur(""))]

    return func


@funcs.param("unwrap")
def unwrap_func(par, in_deg):
    def func(d):
        s = copy.deepcopy(d.s[par])
        d.s[nam.unwrap(par)] = util.apply_per_level(s, util.unwrap_deg).flatten()

    return func


@funcs.param("dst")
def dst_func(point=""):
    def func(d):
        util.compute_dst(d.s, point)

    return func


@funcs.param("vel")
def func_v_spatial(p_d, p_v):
    def func(d):
        d.s[p_v] = d.s[p_d] / d.c.dt

    return func
