import numpy as np
import pandas as pd

from lib.aux import naming as nam, dictsNlists as dNl
from lib.registry import reg, base

def track_par_func(chunk, par):
    def func(d):
        from lib.process.aux import track_par_in_chunk
        track_par_in_chunk(d, chunk, par)

    return func


def chunk_func(kc, store=False):
    if kc in ['str', 'pau', 'exec', 'str_c', 'run']:
        def func(d):
            from lib.process.aux import crawl_annotation
            s, e, c = d.step_data, d.endpoint_data, d.config
            crawl_annotation(s, e, c, strides_enabled=True, store=store)

        required_ks = ['a', 'sa', 'ba', 'foa', 'fv']
    elif kc in ['tur', 'Ltur', 'Rtur']:
        def func(d):
            from lib.process.aux import turn_annotation
            s, e, c = d.step_data, d.endpoint_data, d.config
            turn_annotation(s, e, c, store=store)

        required_ks = ['fov']
    else:
        func = None
        required_ks = []
    return dNl.NestDict({'func': func, 'required_ks': required_ks})


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
        comp_straightness_index(s, e=e, c=c, dt=c.dt, tor_durs=[dur], store=False)

    return func


def mean_func(par):
    def func(d):
        d.endpoint_data[nam.mean(par)] = d.step_data[par].dropna().groupby('AgentID').mean()

    return func


def std_func(par):
    def func(d):
        d.endpoint_data[nam.std(par)] = d.step_data[par].dropna().groupby('AgentID').std()

    return func

def var_func(par):
    def func(d):
        d.endpoint_data[nam.var(par)] = d.step_data[par].dropna().groupby('AgentID').mean()/d.step_data[par].dropna().groupby('AgentID').std()

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


def freq_func(par):
    from lib.aux.sim_aux import get_freq
    def func(d):
        get_freq(d, par=par, fr_range=(0.0, +np.inf))

    return func


def tr_func(pc):
    def func(d):
        e = d.endpoint_data
        e[nam.dur_ratio(pc)] = e[nam.cum(nam.dur(pc))] / e[nam.cum(nam.dur(''))]

    return func


def unwrap_func(par, in_deg):
    from lib.aux.ang_aux import unwrap_rad

    def func(d):
        s, c = d.step_data, d.config
        unwrap_rad(s, c, par, in_deg)

    return func


def dst_func(point=''):
    from lib.aux.xy_aux import comp_dst

    def func(d):
        s, c = d.step_data, d.config
        comp_dst(s, c, point)

    return func


def func_v_spatial(p_d, p_v):
    def func(d):
        s, e, c = d.step_data, d.endpoint_data, d.config
        s[p_v] = s[p_d] / c.dt

    return func





class ParamComputeFunctionRegistry(base.BaseConfDict):

    def build(self):
        return dNl.NestDict({
            'chunk': chunk_func,
            'track_par': track_par_func,
            'tor': tor_func,
            'dsp': dsp_func,
            'ops': {
                'mean': mean_func,
                'std': std_func,
                'var': var_func,
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


class FuncParHelper:

    def __init__(self) :

        self.func_df=self.inspect_funcs()

    def get_func(self, func):
        module=self.func_df['module'].loc[func]
        return getattr(module, func)

    def apply_func(self,func,s,**kwargs):
        f=self.get_func(func)
        kws={k:kwargs[k] for k in kwargs.keys() if k in self.func_df['args'].loc[func]}
        f(s=s,**kws)
        return s

    def assemble_func_df(self,arg='s'):
        from lib.process import angular, spatial,aux, calibration
        arg_dicts = {}
        for module in [angular, spatial, aux, calibration]:
            dic = self.get_arg_dict(module, arg)
            arg_dicts.update(dic)
        df = pd.DataFrame.from_dict(arg_dicts,orient='index')

        return df

    def get_arg_dict(self, module, arg):
        from inspect import getmembers, isfunction, signature


        # funcnames = []
        arg_dict={}
        funcs = getmembers(module, isfunction)
        for k, f in funcs:
            args = signature(f)
            args = list(args.parameters.keys())
            if arg in args:
                arg_dict[k]= {'args' : args, 'module':module}
        return arg_dict

    def inspect_funcs(self, arg='s'):
        df=self.assemble_func_df(arg)
        new_cols=['requires', 'depends', 'computes']
        for col in new_cols :
            df[col]=np.nan

        df[new_cols]=self.manual_fill(df[new_cols])
        return df

    def manual_fill(self,df):
        df.loc['comp_ang_from_xy'] = ['x', 'y'], ['ang_from_xy'], ['fov', 'foa']
        df.loc['angular_processing'] = [], ['comp_orientations', 'comp_bend', 'comp_ang_from_xy', 'comp_angular',
                                            'comp_extrema', 'compute_LR_bias'], []
        df.loc['comp_angular'] = ['fo', 'ro', 'b'], ['unwrap_orientations'], ['fov', 'foa', 'rov', 'roa', 'bv', 'ba']
        df.loc['unwrap_orientations'] = ['fo', 'ro'], [], ['fou', 'rou']
        df.loc['comp_orientation_1point'] = ['x', 'y'], [], ['fov']
        df.loc['compute_LR_bias'] = ['b', 'bv', 'fov'], [], []
        df.loc['comp_orientations'] = ['xys'], ['comp_orientation_1point'], ['fo', 'ro']
        df.loc['comp_bend'] = ['fo', 'ro'], ['comp_angles'], ['b']
        df.loc['comp_angles'] = ['xys'], [], ['angles']
        return df

    def is_computed_by(self, short):
        return [k for k in self.func_df.index if short in self.func_df['computes'].loc[k]]

    def requires(self, func):
        return self.func_df['requires'].loc[func]

    def depends(self,func):
        return self.func_df['depends'].loc[func]

    def requires_all(self, func):
        import lib.aux.dictsNlists as dNl
        shorts=[]
        shorts.append(self.requires(func))
        for f in self.depends(func) :
            shorts.append(self.requires_all(func))
        shorts=dNl.unique_list(shorts)
        return shorts

    def get_options(self, short):
        options={}
        for func in self.is_computed_by(short):
            options[func]=self.requires(func)
        return options

    def how_to_compute(self, s, par=None, short=None, **kwargs):
        if par is None :
            par = reg.getPar(short)
        elif short is None :
            short=reg.getPar(d=par, to_return='k')
        if par in s.columns :
            return True
        else :
            options=self.get_options(short)
            available= []
            for i,(func, shorts) in enumerate(options.items()) :
                pars = reg.getPar(shorts)
                if all([p in s.columns for p in pars]):

                    available.append(func)
            if len(available)==0 :
                return False
            else :
                return available

    def compute(self,s,**kwargs):
        res=self.how_to_compute(s=s,**kwargs)
        if res in [True, False]:
            return res
        else:
            self.apply_func(res[0],s=s, **kwargs)
            return self.compute(s=s,**kwargs)
