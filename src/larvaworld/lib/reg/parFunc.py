import copy

import numpy as np


from larvaworld.lib.aux import naming as nam
from larvaworld.lib.reg import funcs
from larvaworld.lib import aux

@funcs.param("track_par")
def track_par_func(chunk, par):
    def func(d):
        from larvaworld.lib.process.annotation import track_par_in_chunk
        track_par_in_chunk(d, chunk, par)

    return func

@funcs.param("chunk")
def chunk_func(kc, store=False):
    if kc in ['str', 'pau', 'exec', 'str_c', 'run']:
        def func(d):
            from larvaworld.lib.process.annotation import crawl_annotation
            s, e, c = d.step_data, d.endpoint_data, d.config
            crawl_annotation(s, e, c, strides_enabled=True, store=store)

        required_ks = ['a', 'sa', 'ba', 'foa', 'fv']
    elif kc in ['tur', 'Ltur', 'Rtur']:
        def func(d):
            from larvaworld.lib.process.annotation import turn_annotation
            s, e, c = d.step_data, d.endpoint_data, d.config
            turn_annotation(s, e, c, store=store)

        required_ks = ['fov']
    else:
        func = None
        required_ks = []
    return aux.AttrDict({'func': func, 'required_ks': required_ks})

@funcs.param("dsp")
def dsp_func(range):
    r0, r1 = range

    def func(d):
        from larvaworld.lib.process.spatial import comp_dispersion
        s, e, c = d.step_data, d.endpoint_data, d.config
        comp_dispersion(s, e, c, recompute=True, dsp_starts=[r0], dsp_stops=[r1], store=False)

    return func

@funcs.param("tor")
def tor_func(dur):
    def func(d):
        from larvaworld.lib.process.spatial import comp_straightness_index
        s, e, c = d.step_data, d.endpoint_data, d.config
        comp_straightness_index(s, e=e, c=c, dt=c.dt, tor_durs=[dur], store=False)

    return func

@funcs.param("mean")
def mean_func(par):
    def func(d):
        d.endpoint_data[nam.mean(par)] = d.step_data[par].dropna().groupby('AgentID').mean()

    return func

@funcs.param("std")
def std_func(par):
    def func(d):
        d.endpoint_data[nam.std(par)] = d.step_data[par].dropna().groupby('AgentID').std()

    return func

@funcs.param("var")
def var_func(par):
    def func(d):
        d.endpoint_data[nam.var(par)] = d.step_data[par].dropna().groupby('AgentID').mean()/d.step_data[par].dropna().groupby('AgentID').std()

    return func

@funcs.param("min")
def min_func(par):
    def func(d):
        d.endpoint_data[nam.min(par)] = d.step_data[par].dropna().groupby('AgentID').min()

    return func

@funcs.param("max")
def max_func(par):
    def func(d):
        d.endpoint_data[nam.max(par)] = d.step_data[par].dropna().groupby('AgentID').max()

    return func

@funcs.param("final")
def fin_func(par):
    def func(d):
        d.endpoint_data[nam.final(par)] = d.step_data[par].dropna().groupby('AgentID').last()

    return func

@funcs.param("initial")
def init_func(par):
    def func(d):
        d.endpoint_data[nam.initial(par)] = d.step_data[par].dropna().groupby('AgentID').first()

    return func

@funcs.param("cum")
def cum_func(par):
    def func(d):
        d.endpoint_data[nam.cum(par)] = d.step_data[par].dropna().groupby('AgentID').sum()

    return func

@funcs.param("freq")
def freq_func(par):
    def func(d):
        aux.get_freq(d, par=par, fr_range=(0.0, +np.inf))

    return func

@funcs.param("tr")
def tr_func(pc):
    def func(d):
        e = d.endpoint_data
        e[nam.dur_ratio(pc)] = e[nam.cum(nam.dur(pc))] / e[nam.cum(nam.dur(''))]

    return func

@funcs.param("unwrap")
def unwrap_func(par, in_deg):

    def func(d):
        s = copy.deepcopy(d.step_data[par])
        d.step_data[nam.unwrap(par)]=aux.unwrap_rad(s, in_deg)

    return func

@funcs.param("dst")
def dst_func(point=''):
    def func(d):
        aux.compute_dst(d.step_data, point)
    return func

@funcs.param("vel")
def func_v_spatial(p_d, p_v):
    def func(d):
        s, e, c = d.step_data, d.endpoint_data, d.config
        s[p_v] = s[p_d] / c.dt

    return func