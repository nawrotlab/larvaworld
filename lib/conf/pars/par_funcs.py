import numpy as np

from lib.aux import naming as nam, dictsNlists as dNl



def chunk_func(kc, store=False):
    if kc in ['str', 'pau', 'run', 'str_c']:
        def func(d):
            from lib.process.aux import crawl_annotation
            s, e, c = d.step_data, d.endpoint_data, d.config
            crawl_annotation(s, e, c, strides_enabled=True, store=store)
        required_ks=['a', 'sa','ba','foa', 'fv']
    elif kc in ['tur', 'Ltur', 'Rtur']:
        def func(d):
            from lib.process.aux import turn_annotation
            s, e, c = d.step_data, d.endpoint_data, d.config
            turn_annotation(s, e, c, store=store)

        required_ks = ['fov']
    else:
        func = None
        required_ks = []
    return dNl.NestDict({'func': func, 'required_ks' :required_ks})


def dsp_func(range):
    r0, r1 = range

    def func(d):
        from lib.process.spatial import comp_dispersion
        s, e, c = d.step_data, d.endpoint_data, d.config
        comp_dispersion(s, e, c, recompute=True, dsp_starts=[r0], dsp_stops=[r1], store=False)

    return func

def tor_func(dur):
    def func(d):
        from lib.process.spatial import comp_straightness_index
        s, e, c = d.step_data, d.endpoint_data, d.config
        comp_straightness_index(s, e=e, c=c, dt=c.dt, tor_durs=[dur], store=True)

    return func

def mean_func(par):
    def func(d):
        d.endpoint_data[nam.mean(par)] = d.step_data[par].dropna().groupby('AgentID').mean()
    return func

def std_func(par):
    def func(d):
        d.endpoint_data[nam.std(par)] = d.step_data[par].dropna().groupby('AgentID').std()
    return func

def min_func(par):
    def func(d):
        d.endpoint_data[nam.min(par)] = d.step_data[par].dropna().groupby('AgentID').min()
    return func

def max_func(par):
    def func(d):
        d.endpoint_data[nam.max(par)] = d.step_data[par].dropna().groupby('AgentID').max()
    return func

def fin_func(par):
    def func(d):
        d.endpoint_data[nam.final(par)] = d.step_data[par].dropna().groupby('AgentID').last()
    return func

def init_func(par):
    def func(d):
        d.endpoint_data[nam.initial(par)] = d.step_data[par].dropna().groupby('AgentID').first()
    return func

def cum_func(par):
    def func(d):
        d.endpoint_data[nam.cum(par)] = d.step_data[par].dropna().groupby('AgentID').sum()
    return func

def freq_func(par) :
    from lib.aux.sim_aux import get_freq
    def func(d):
        get_freq(d, par=par, fr_range=(0.0, +np.inf))
    return func

def tr_func(pc):
    def func(d):
        e=d.endpoint_data
        e[nam.dur_ratio(pc)] = e[nam.cum(nam.dur(pc))] / e[nam.cum(nam.dur(''))]
    return func

def unwrap_func(par, in_deg):
    from lib.aux.ang_aux import unwrap_rad

    def func(d):
        s, c = d.step_data, d.config
        unwrap_rad(s,c,par, in_deg)
    return func

def dst_func(point=''):
    from lib.aux.xy_aux import comp_dst

    def func(d):
        s, c = d.step_data, d.config
        comp_dst(s,c,point)
    return func

def func_v_spatial(p_d, p_v):
    def func(d):
        s, e, c = d.step_data, d.endpoint_data, d.config
        s[p_v] = s[p_d] / c.dt

    return func

def build_func_dict() :
    func_dict = dNl.NestDict({
            'chunk': chunk_func,
            'tor': tor_func,
            'dsp': dsp_func,
            'ops': {
                'mean': mean_func,
                'std': std_func,
                'min': min_func,
                'max': max_func,
                'final': fin_func,
                'initial': init_func,
                'cum': cum_func,

            },
            'freq': freq_func,
            'tr': tr_func,
            'dst': dst_func,
            'unwrap': unwrap_func,
            'vel': func_v_spatial,
        })
    return func_dict
