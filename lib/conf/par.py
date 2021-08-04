import copy
import inspect
import os
from typing import Tuple, Type, Union

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from siunits import Composite, DerivedUnit, BaseUnit

from lib.aux import functions as fun
from lib.aux import naming as nam
from lib.model.DEB.gut import Gut
from lib.stor import paths
from lib.conf.par_conf import sup, sub, th, dot, ddot, subsup, Delta, delta, ast, bar, wave, par_dict_lists, paren, \
    brack, odot, circledcirc, circledast, ddot_th, dot_th, get_par_dict
from lib.model.DEB.deb import DEB
from lib.model.agents._agent import LarvaworldAgent, Larva
import siunits as siu

import lib.conf.dtype_dicts as dtypes

class Collection:
    def __init__(self, name, par_dict, keys=None, object_class=None):
        if keys is None:
            keys = collection_dict[name]
        self.name = name

        # par_dict = build_par_dict(save=False)
        pars = [par_dict[k] for k in keys]
        if object_class is not None:
            self.object_class = object_class
        else:
            os = [p.o for p in pars if p.o is not None]
            os = fun.unique_list(os)

            if len(os) == 0:
                self.object_class = None
            elif len(os) == 1:
                self.object_class = os[0]
            else:
                raise ValueError('Not all parameters have the same object_class class')

        self.par_names = [p.d for p in pars]
        self.dict = {p.d: p for p in pars}



    # def get_from(self, object, u=True):
    #     if self.object_class is not None :
    #         if not isinstance(object, self.object_class):
    #             raise ValueError(f'Parameter Group {self.name} collected from {self.object_class} not from {type(object)}')
    #     dic = {n: p.get_from(object, u=u) for n, p in self.dict.items()}
    #     return dic


class AgentCollector:
    def __init__(self, collection_name, object,save_as=None, save_to=None):
        if save_as is None:
            save_as = f'{object.unique_id}.csv'
        self.save_as = save_as
        self.save_to = save_to
        self.object = object

        dic=self.get_constants(self.object)
        par_dict = build_par_dict(save=False, df=dic)
        self.collection = Collection(collection_name, par_dict)

        self.table = {n: [] for n in self.collection.par_names}
        self.tick = 0

        if self.collection.object_class is not None:
            if not isinstance(object, self.collection.object_class):
                raise ValueError(
                    f'Parameter Collection {self.collection.name} collected from {self.collection.object_class} not from {type(object)}')

    def collect(self, u=True, tick=None, df=None):
        if tick is None:
            tick = self.tick
        for n, p in self.collection.dict.items():
            try:
                self.table[n].append(p.get_from(self.object, u=u, tick=tick, df=df))
            except:
                self.table.pop(n, None)
        self.tick += 1

    def save(self, as_df=False):
        if self.save_to is not None:
            os.makedirs(self.save_to, exist_ok=True)
            f = f'{self.save_to}/{self.save_as}'
            if as_df:
                df = pd.DataFrame(self.table)
                df.to_csv(f, index=True, header=True)
            else:
                fun.save_dict(self.table, f)

    def get_constants(self, object):
        dic =build_constants()
        for k,p in dic.items():
            p.const=p.get_from(object, u=False, tick=None, df=None)
        return dic

class GroupCollector:
    def __init__(self, objects, name, save_to=None, save_as=None, common=False,save_units=True, **kwargs):
        if save_as is None:
            save_as = f'{name}.csv'
        self.save_units = save_units
        self.save_as = save_as
        self.save_to = save_to
        self.common = common
        self.name = name
        self.collectors = [AgentCollector(object=o, collection_name=name, save_to=save_to, **kwargs) for o in
                           objects]
        self.tick = 0

    def collect(self, u=None, tick=None, df=None):
        if tick is None:
            tick = self.tick
        if u is None:
            u = self.save_units
        for c in self.collectors:
            c.collect(u=u, tick=tick, df=df)
        self.tick += 1

    def save(self, as_df=False, save_to=None, save_units_dict=False):
        if not self.common:
            for c in self.collectors:
                c.save(as_df)
            return None, None
        else:
            dfs = []
            for c in self.collectors:
                df = pd.DataFrame(c.table)
                df['AgentID'] = c.object.unique_id
                df.index.set_names(['Step'], inplace=True)
                df.reset_index(drop=False, inplace=True)
                df.set_index(['Step', 'AgentID'], inplace=True)
                dfs.append(df)
            ddf = pd.concat(dfs)
            ddf.sort_index(level=['Step', 'AgentID'], inplace=True)
            dddf, u_dict = fun.split_si_composite(ddf)
            if save_to is None:
                save_to = self.save_to
            if save_to is not None:
                os.makedirs(save_to, exist_ok=True)
                f = f'{save_to}/{self.save_as}'

                if save_units_dict:
                    ff = f'{save_to}/units.csv'
                    fun.save_dict(u_dict, ff)

                if as_df:
                    dddf.to_csv(f, index=True, header=True)
                else:
                    fun.save_dict(dddf.to_dict(), f)
            return dddf, u_dict


class CompGroupCollector(GroupCollector):
    def __init__(self, names, save_to=None, save_as=None, save_units=True, **kwargs):
        # self.collectors=collectors
        if save_as is None:
            save_as = 'complete.csv'
        self.save_units = save_units
        self.save_as = save_as
        self.save_to = save_to
        self.collectors = [GroupCollector(name=n, save_to=save_to, save_units=save_units, **kwargs) for n in names]
        self.tick = 0

    def save(self, as_df=False, save_to=None, save_units_dict=False):
        if save_to is None:
            save_to = self.save_to
        dfs = []
        u0_dict = {}
        for c in self.collectors:
            df, u_dict = c.save(as_df=as_df, save_to=save_to, save_units_dict=False)
            if df is not None:
                u0_dict.update(u_dict)
                dfs.append(df)
        if len(dfs) > 0:
            df0 = pd.concat(dfs, axis=1)
            _, i = np.unique(df0.columns, return_index=True)
            df0 = df0.iloc[:, i]
            df0.sort_index(level=['Step', 'AgentID'], inplace=True)
        else:
            df0 = None
        if save_to is not None and df0 is not None:
            os.makedirs(save_to, exist_ok=True)
            f = f'{save_to}/{self.save_as}'
            if as_df:
                df0.to_csv(f, index=True, header=True)
            else:
                fun.save_dict(df0.to_dict(), f)
            if save_units_dict:
                # ff = f'{save_to}/units.csv'
                # fun.save_dict(u0_dict, ff)
                try:
                    uu_dict = fun.load_dicts([paths.UnitDict_path])[0]
                    u0_dict.update(uu_dict)
                except:
                    pass
                fun.save_dict(u0_dict, paths.UnitDict_path)
        return df0


class Parameter:
    def __init__(self, p, u, k=None, s=None, o=None, lim=None,
                 d=None, l=None,exists=True, func=None, const=None, par_dict=None, fraction=False,
                 operator=None, k0=None, k_num=None, k_den=None, dst2source=None, or2source=None, dispersion=False,wrap_mode=None):
        self.wrap_mode = wrap_mode
        self.fraction = fraction
        self.func = func
        self.exists = exists
        self.p = p
        if k is None:
            k = p
        self.k = k
        if s is None:
            s = self.k
        self.s = s
        self.o = o

        if d is None:
            d = p
        self.d = d
        self.const = const
        self.operator = operator
        # self.cum = cum
        self.k0 = k0
        self.k_num = k_num
        self.k_den = k_den
        self.p0 = par_dict[k0] if k0 is not None else None
        self.p_num = par_dict[k_num] if k_num is not None else None
        self.p_den = par_dict[k_den] if k_den is not None else None
        self.tick = np.nan
        self.current = np.nan
        self.previous = np.nan
        if self.p_num is not None and self.p_den is not None:
            u = self.p_num.u / self.p_den.u
        elif u is None:
            u = 1 * siu.I
        self.u = u
        self.dst2source = dst2source
        self.or2source = or2source
        self.dispersion = dispersion
        self.par_dict = par_dict
        if wrap_mode=='positive' :
            if lim is None :
                if self.u.unit==siu.deg :
                    lim=(0.0,360.0)
                    # self.range=360
                elif self.u.unit==siu.rad :
                    lim=(0.0,2*np.pi)
                    # self.range = 2*np.pi

        elif wrap_mode=='zero' :
            if lim is None :
                if self.u.unit==siu.deg :
                    lim=(-180.0,180.0)
                    # self.range = 360
                elif self.u.unit==siu.rad :
                    lim=(-np.pi,np.pi)
                    # self.range = 2 * np.pi

        # else :
            # self.range=None
        self.lim=lim

        self.range=lim[1]-lim[0] if lim is not None else None




        # print(self.k, self.dispersion, self.)
    @property
    def l(self):
        return f'{self.d},  {self.s}$({self.u.unit.abbrev})$'

    def get_from(self, o, u=True, tick=None, df=None):
        # print(o,u,tick,df)
        if self.const is not None:
            v = self.const
        if tick != self.tick:
            if self.func is not None:
                # print(self.k)
                v = getattr(self, self.func[0])(o, **self.func[1])
                # print(v)
            elif self.exists:
                v = getattr(o, self.p)
            elif self.p0 is not None:
                if self.operator in ['diff', 'cum']:
                    v = self.p0.get_from(o, u=False, tick=tick)
                elif self.operator in ['mean', 'std', 'min', 'max']:
                    if df is not None:
                        vs = df[self.p0.d].xs(o.unique_id, level='AgentID').dropna()
                        v = vs.apply(self.operator)
                    else:
                        v = np.nan
                elif self.operator == 'freq':
                    if df is not None:
                        vs = df[self.p0.d].xs(o.unique_id, level='AgentID').dropna()
                        dt=self.par_dict['dt'].get_from(o, u=False)
                        v = fun.freq(vs,dt)
                    else:
                        v = np.nan
                elif self.operator == 'final':
                    if df is not None:
                        vs = df[self.p0.d].xs(o.unique_id, level='AgentID').dropna()
                        v = vs[-1]
                    else:
                        v = np.nan

            elif self.fraction:
                v_n = self.p_num.get_from(o, u=False, tick=tick)
                v_d = self.p_den.get_from(o, u=False, tick=tick)
                v = v_n / v_d
            elif self.dst2source is not None:
                v = euclidean(self.xy(o=o,tick=tick,df=df), self.dst2source)
            elif self.dispersion:
                v = euclidean(self.xy(o=o,tick=tick,df=df), self.xy0(o=o,df=df))
            elif self.or2source is not None:
                v = fun.angle_dif(getattr(o, 'front_orientation'),fun.angle_to_x_axis(self.xy(o=o,tick=tick,df=df), self.or2source))
            v = self.postprocess(v)
            self.tick = tick
        else:
            v = self.current
        if u:
            v *= self.u
        return v


    def xy(self, o, tick, df=None):
        # print(o,type(o),df)
        if type(o) == str and df is not None:
            # print('ssssssssssssssssssssss')
            xys=df[nam.xy('')].xs(o, level='AgentID')
            xy = tuple(xys.values[-1,:])
        else:
            if tick is not None :
                xy = getattr(o, 'trajectory')[tick]
            else :
                xy = getattr(o, 'pos')
        return xy
        # return getattr(o, 'trajectory')[tick]

    def xy0(self, o, df=None):
        if type(o) == str and df is not None:
            xys=df[nam.xy('')].xs(o, level='AgentID')
            xy0 = tuple(xys.values[0,:])
        else:
            xy0 = getattr(o, 'initial_pos')
        return xy0
    # def xy(self, o, u=False, tick=None):
    #     x = self.par_dict['x'].get_from(o, u=u, tick=tick)
    #     y = self.par_dict['y'].get_from(o, u=u, tick=tick)
    #     return x, y

    def d_2D(self,o,t0,t1):
        dt=self.par_dict['dt'].get_from(o, u=False)
        tick0=int(t0/dt)
        tick1=int(t1/dt)
        # xs = df['x'].xs(o.unique_id, level='AgentID').values
        # ys = df['y'].xs(o.unique_id, level='AgentID').values
        # xy0=xs[tick0], ys[tick0]
        # xy1=xs[tick1], ys[tick1]
        d=euclidean(self.xy(o,tick1), self.xy(o,tick0))
        return d



    def postprocess(self, v):
        if self.operator == 'diff':
            vv = v - self.previous
            v0 = v
        elif self.operator == 'cum':
            vv = np.nansum([v, self.previous])
            v0 = vv
        elif self.range is not None and self.wrap_mode is not None:
            vv=v%self.range
            if vv>self.lim[1] :
                vv-=self.range
            v0 = v
        else:
            vv = v
            v0 = v
        self.previous = v0
        self.current = vv
        return vv







def add_par(dic, **kwargs):
    p = dtypes.get_dict('par', **kwargs)
    k = p['k']
    if k in dic.keys():
        raise ValueError(f'Key {k} already exists')
    dic[k] = Parameter(**p, par_dict=dic)
    return dic


def add_diff_par(dic, k0):
    b = dic[k0]
    dic = add_par(dic, p=f'D_{b.p}', k=f'D_{k0}', u=b.u, d=f'{b.d} change', s=Delta(b.s), exists=False, operator='diff',
                  k0=k0)
    return dic


def add_cum_par(dic, k0, d=None,p=None, s=None, k=None):
    b = dic[k0]
    if d is None:
        d = nam.cum(b.d)
    if p is None:
        p = nam.cum(b.p)
    if s is None:
        s = sub(b.s, 'cum')
    if k is None:
        k = nam.cum(k0)
        # d = f'total {b.d}'
    dic = add_par(dic, p=p, k=k, u=b.u, d=d, s=s, exists=False, operator='cum',k0=k0)
    return dic


def add_mean_par(dic, k0, d=None, s=None):
    b = dic[k0]
    if d is None:
        d = nam.mean(b.d)
        # d = f'total {b.d}'
    if s is None :
        s=bar(b.s)
    dic = add_par(dic, p=nam.mean(b.p), k=f'{b.k}_mu', u=b.u, d=d, s=s, exists=False, operator='mean', k0=k0)
    return dic


def add_std_par(dic, k0, d=None, s=None):
    b = dic[k0]
    if d is None:
        d = nam.std(b.d)
        # d = f'total {b.d}'
    if s is None :
        s=wave(b.s)
    dic = add_par(dic, p=nam.std(b.p), k=f'{b.k}_std', u=b.u, d=d, s=s, exists=False, operator='std', k0=k0)
    return dic


def add_min_par(dic, k0, d=None, s=None):
    b = dic[k0]
    if d is None:
        d = nam.min(b.d)
        # d = f'total {b.d}'
    if s is None :
        s=sub(b.s, 'min')
    dic = add_par(dic, p=nam.min(b.p), k=f'{b.k}_min', u=b.u, d=d, s=s, exists=False, operator='min',k0=k0)
    return dic


def add_max_par(dic, k0, d=None, s=None):
    b = dic[k0]
    if d is None:
        d = nam.max(b.d)
        # d = f'total {b.d}'
    if s is None :
        s=sub(b.s, 'max')
    dic = add_par(dic, p=nam.max(b.p), k=f'{b.k}_max', u=b.u, d=d, s=s, exists=False, operator='max',k0=k0)
    return dic

def add_fin_par(dic, k0, d=None, s=None):
    b = dic[k0]
    if d is None:
        d = nam.final(b.d)
        # d = f'total {b.d}'
    if s is None :
        s=sub(b.s, 'fin')
    dic = add_par(dic, p=nam.final(b.p), k=f'{b.k}_fin', u=b.u, d=d, s=s, exists=False, operator='final',k0=k0)
    return dic

def add_freq_par(dic, k0, d=None):
    b = dic[k0]
    if d is None:
        d = nam.freq(b.d)
        # d = f'total {b.d}'
    dic = add_par(dic, p=nam.freq(b.p), k=f'f{b.k}', u=1*siu.hz, d=d, s=sub(b.s, 'freq'), exists=False, operator='freq',
                  k0=k0)
    return dic

def add_dsp_par(dic, range=(0,40)) :
    a='dispersion'
    k0='dsp'
    s0=circledast('d')
    r0,r1=range
    p=f'{a}_{r0}_{r1}'
    k=f'{k0}_{r0}_{r1}'
    dic = add_par(dic, p=p, k=k, d=p, s=subsup(s0,f'{r0}',f'{r1}'), exists=False, func=('d_2D', {'t0':r0, 't1':r1}))
    # dic = add_rate_par(dic, k_den='l', k_num=k)
    dic = add_scaled_par(dic, k0=k, s=subsup(paren(k0),f'{r0}',f'{r1}'))
    # dic = add_rate_par(dic, k_den='l', k_num=k, k=kk)
    #
    for k00 in [k, f's{k}'] :
        dic = add_mean_par(dic, k0=k00)
        dic = add_std_par(dic, k0=k00)
        dic = add_min_par(dic, k0=k00)
        dic = add_max_par(dic, k0=k00)
        dic = add_fin_par(dic, k0=k00)

    return dic


def add_rate_par(dic, k0=None, k_time='t', p=None, k=None, d=None, s=None, k_num=None, k_den=None):
    if k0 is not None:
        b = dic[k0]
        if p is None:
            p = f'd_{k0}'
        if k is None:
            k = f'd_{k0}'
        if d is None:
            d = f'{b.d} rate'
        if s is None:
            s = dot(b.s)
        if k_num is None:
            k_num = f'D_{k0}'
    if k_den is None:
        k_den = f'D_{k_time}'

    dic = add_par(dic, p=p, k=k, d=d, s=s, exists=False, fraction=True, k_num=k_num, k_den=k_den)
    return dic


def add_Vspec_par(dic, k0):
    b = dic[k0]
    dic = add_par(dic, p=f'[{k0}]', k=f'[{k0}]', d=f'volume specific {b.d}', s=f'[{b.s}]', exists=False, fraction=True,
                  k_num=k0, k_den='V')
    return dic

def add_chunk(dic, pc, kc) :
    p0,p1, pt, pid, ptr, pN=nam.start(pc), nam.stop(pc), nam.dur(pc),nam.id(pc),nam.dur_ratio(pc), nam.num(pc)

    dic = add_par(dic, p=pc, k=kc, d=pc, s=f'${kc}$', exists=False)
    b=dic[kc]
    dic = add_par(dic, p=p0, k=f'{kc}0',u=1*siu.s, d=p0, s=subsup('t', kc, 0), exists=False)
    dic = add_par(dic, p=p1, k=f'{kc}1',u=1*siu.s, d=p1, s=subsup('t', kc, 1), exists=False)

    dic = add_par(dic, p=pid, k=f'{kc}_id', d=pid, s=sub('idx', kc), exists=False)
    dic = add_par(dic, p=ptr, k=f'{kc}_tr', d=ptr, s=sub('r', kc), exists=False)
    dic = add_par(dic, p=pN, k=f'{kc}_N', d=pN, s=sub('N', f'{pc}s'), exists=False)

    k00=f'{kc}_t'
    s00 =Delta('t')
    dic = add_par(dic, p=pt, k=k00, u=1 * siu.s, d=pt, s=sub(s00, kc), exists=False)
    dic = add_cum_par(dic, k0=k00)
    dic = add_mean_par(dic, k0=k00, s=sub(bar(s00), kc))
    dic = add_std_par(dic, k0=k00, s=sub(wave(s00), kc))
    dic = add_min_par(dic, k0=k00)
    dic = add_max_par(dic, k0=k00)


    if str.endswith(pc, 'chain') :
        pl=nam.length(pc)
        k00 = f'{kc}_l'
        dic = add_par(dic, p=pl, k=k00, d=pl, s=sub('l', kc), exists=False)
        dic = add_cum_par(dic, k0=k00)
        dic = add_mean_par(dic, k0=k00, s=sub(bar('l'), kc))
        dic = add_std_par(dic, k0=k00, s=sub(wave('l'), kc))
        dic = add_min_par(dic, k0=k00)
        dic = add_max_par(dic, k0=k00)


    return dic

def add_chunk_track(dic,kc, k, extrema=True) :
    bc=dic[kc]
    b=dic[k]
    u = dic[k].u
    b0,b1=dic[f'{kc}0'],dic[f'{kc}1']
    p0,p1=nam.at(b.p,b0.p),nam.at(b.p,b1.p)

    k00=f'{kc}_{k}'
    s00=Delta(b.s)
    dic = add_par(dic, p=nam.chunk_track(bc.p,b.p), k=k00, u=u, d=nam.chunk_track(bc.p,b.p), s=sub(s00, kc),exists=False)
    dic=add_mean_par(dic, k0=k00, s=sub(bar(s00), kc))
    dic=add_std_par(dic, k0=k00, s=sub(wave(s00), kc))
    if extrema :
        dic = add_par(dic, p=p0, k=f'{kc}_{k}0', u=u, d=p0, s=subsup(b.s, kc, 0),exists=False)
        dic = add_par(dic, p=p1, k=f'{kc}_{k}1', u=u, d=p1, s=subsup(b.s, kc, 1),exists=False)
    return dic

def add_scaled_par(dic, k0, s=None) :
    k_den = 'l'
    k_num=k0
    b=dic[k0]
    d=nam.scal(b.d)
    if s is None :
        s=paren(b.s)
    dic = add_par(dic, p=d, k=f's{k0}', d=d, s=s, exists=False, fraction=True, k_num=k_num, k_den=k_den)
    return dic


def build_constants() :
    df={}
    df = add_par(df, p='x0', k='x0', u=1 * siu.m, o=Larva, d='initial x coordinate', s=sub('x', 0))
    df = add_par(df, p='y0', k='y0', u=1 * siu.m, o=Larva, d='initial y coordinate', s=sub('y', 0))
    df = add_par(df, p='dt', k='dt', u=1 * siu.s, o=Larva, d='timestep', s='$dt$')
    df = add_par(df, p='real_length', k='l', u=1 * siu.m, o=Larva, d='length', s='l')
    return df

def build_DEB_par_dict(df=None) :
    if df is None :
        df={}
    df = add_par(df, p='L', k='L', u=1 * siu.cm, o=DEB, d='structural length', s='L')
    df = add_par(df, p='Lw', k='Lw', u=1 * siu.cm, o=DEB, d='physical length', s=sub('L', 'w'))
    df = add_par(df, p='V', k='V', u=1 * siu.cm ** 3, o=DEB, d='structural volume', s='V')
    df = add_par(df, p='Ww', k='Ww', u=1 * siu.g, o=DEB, d='wet weight', s=sub('W', 'w'))
    df = add_par(df, p='age', k='age', u=1 * siu.day, o=DEB, d='age', s='a')
    df = add_par(df, p='hunger', k='H', o=DEB, d='hunger drive', s='H')
    df = add_par(df, p='E', k='E', u=1 * siu.j, o=DEB, d='reserve energy', s='E')
    df = add_par(df, p='E_H', k='E_H', u=1 * siu.j, o=DEB, d='maturity energy', s=sub('E', 'H'))
    df = add_par(df, p='E_R', k='E_R', u=1 * siu.j, o=DEB, d='reproduction buffer', s=sub('E', 'R'))
    df = add_par(df, p='deb_p_A', k='deb_p_A', u=1 * siu.j, o=DEB, d='assimilation energy (model)',
                 s=subsup('p', 'A', 'deb'))
    df = add_par(df, p='sim_p_A', k='sim_p_A', u=1 * siu.j, o=DEB, d='assimilation energy (sim)',
                 s=subsup('p', 'A', 'sim'))
    df = add_par(df, p='gut_p_A', k='gut_p_A', u=1 * siu.j, o=DEB, d='assimilation energy (gut)',
                 s=subsup('p', 'A', 'gut'))
    df = add_par(df, p='e', k='e', o=DEB, d='scaled reserve density', s='e')
    df = add_par(df, p='f', k='f', o=DEB, d='scaled functional response', s='f')
    df = add_par(df, p='base_f', k='f0', o=DEB, d='base scaled functional response', s=sub('f', 0))
    df = add_par(df, p='F', k='[F]', u=siu.hz / (24 * 60 * 60), o=DEB, d='volume specific filtering rate', s=brack(dot('F')))
    df = add_par(df, p='fr_feed', k='fr_f', u=1 * siu.hz, o=DEB, d='feed motion frequency (estimate)',
                 s=sub(dot('fr'), 'feed'))
    df = add_par(df, p='pupation_buffer', k='pupation', o=DEB, d='pupation ratio', s=sub('r', 'pup'))

    df = add_diff_par(df, k0='age')
    for k0 in ['f', 'e', 'H']:
        df = add_diff_par(df, k0=k0)
        df = add_rate_par(df, k0=k0, k_time='age')

    for k0 in ['E', 'Ww', 'E_R', 'E_H']:
        df = add_Vspec_par(df, k0=k0)

    return df

def build_gut_par_dict(df=None) :
    if df is None :
        df={}
    df = add_par(df, p='ingested_volume', k='f_am_V', u=1 * siu.m ** 3, o=Gut, d='ingested_food_volume',
                 s=sub('V', 'in'))
    df = add_par(df, p='ingested_gut_volume_ratio', k='sf_am_Vg', o=Larva, d='ingested_gut_volume_ratio',
                 s=subsup('[V]', 'in', 'gut'))
    df = add_par(df, p='ingested_body_volume_ratio', k='sf_am_V', o=Larva, d='ingested_body_volume_ratio',
                 s=sub('[V]', 'in'))
    df = add_par(df, p='ingested_body_area_ratio', k='sf_am_A', o=Larva, d='ingested_body_area_ratio',
                 s=sub('{V}', 'in'))
    df = add_par(df, p='ingested_body_mass_ratio', k='sf_am_M', o=Larva, d='ingested_body_mass_ratio',
                 s=sub('[M]', 'in'))

    return df


def build_spatial_par_dict(df):
    d=nam.dst('')
    std=nam.straight_dst('')
    df = add_par(df, p=d, k='d', u=1 * siu.m, o=Larva, d=d, s='d')
    df = add_par(df, p=std, k='std', u=1 * siu.m, o=Larva, d=std, s='d')
    df = add_par(df, p='dispersion', k='dsp', u=1 * siu.m, o=Larva, d='dispersion', s=circledast('d'), exists=False,
                 dispersion=True)

    df = add_par(df, p=nam.dst2('center'), k='d_cent', u=1 * siu.m, o=Larva, d=nam.dst2('center'), s=odot('d'),
                 exists=False, dst2source=(0, 0))
    df = add_par(df, p=nam.dst2('source'), k='d_chem', u=1 * siu.m, o=Larva, d=nam.dst2('source'),
                 s=circledcirc('d'), exists=False, dst2source=(0.04, 0.0))

    df = add_par(df, p='x', k='x', u=1 * siu.m, o=Larva, d='x', s='x')
    df = add_par(df, p='y', k='y', u=1 * siu.m, o=Larva, d='y', s='y')

    df = add_diff_par(df, k0='x')
    df = add_diff_par(df, k0='y')

    space_ks = ['d', 'D_x', 'D_y', 'd_chem', 'd_cent', 'dsp', 'std']



    for k0 in space_ks:
        df = add_scaled_par(df, k0=k0)


    for k00 in space_ks :
        for k0 in [k00, f's{k00}'] :
            df = add_cum_par(df, k0=k0)
            df = add_mean_par(df, k0=k0)
            df = add_std_par(df, k0=k0)
            df = add_min_par(df, k0=k0)
            df = add_max_par(df, k0=k0)
            df = add_fin_par(df, k0=k0)
    v = nam.vel('')
    a = nam.acc('')
    sv, sa = nam.scal([v, a])
    df = add_rate_par(df, k_num='d', k_den='dt', k='v', p=v, d=v, s='v')
    df = add_rate_par(df, k_num='v', k_den='dt', k='a', p=a, d=a, s='a')
    df = add_rate_par(df, k_num='sd', k_den='dt', k='sv', p=sv, d=sv, s=paren('v'))
    df = add_rate_par(df, k_num='sv', k_den='dt', k='sa', p=sa, d=sa, s=paren('a'))

    for i in [(0,40), (0,80), (20,80)] :
        df = add_dsp_par(df, range=i)

    for k0 in ['l', 'v', 'sv']:
        df =add_mean_par(df, k0=k0)
        # print(df[f'{df[k0].k}_mu'].d)

    for k0 in ['sv']:
        df =add_freq_par(df, k0=k0)

    for i in ['',2,5,10,20] :
        if i=='' :
            p='tortuosity'
        else :
            p=f'tortuosity_{i}'
        k0='tor'
        k=f'{k0}{i}'
        df = add_par(df, p=p, k=k, d=p,s=sub(k0, i), exists=False)
        df = add_mean_par(df, k0=k, s=sub(bar(k0), i))
        df = add_std_par(df, k0=k, s=sub(wave(k0), i))
    return df


def build_angular_par_dict(df):
    df = add_par(df, p=nam.bearing2('center'), k='o_cent', u=1 * siu.deg, o=Larva, d=nam.bearing2('center'),
                 s=odot(th('or')), exists=False, or2source=(0, 0), wrap_mode='zero')
    df = add_par(df, p=nam.bearing2('source'), k='o_chem', u=1 * siu.deg, o=Larva, d=nam.bearing2('source'),
                 s=circledcirc(th('or')), exists=False, or2source=(0.04, 0.0), wrap_mode='zero')

    df = add_par(df, p='bend', k='b', u=1 * siu.deg, o=Larva, d='bend', s=th('b'), wrap_mode='zero')
    fo = sub('or', 'f')
    # fo='or_f'
    ro = sub('or', 'r')
    # ro='or_r'
    df = add_par(df, p='front_orientation', k='fo', u=1 * siu.deg, o=Larva, d=nam.orient('front'), s=th(fo),
                 wrap_mode='positive')
    df = add_par(df, p='rear_orientation', k='ro', u=1 * siu.deg, o=Larva, d=nam.orient('rear'), s=th(ro),
                 wrap_mode='positive')
    df = add_par(df, p='front_orientation_unwrapped', k='fou', u=1 * siu.deg, o=Larva,
                 d=nam.unwrap(nam.orient('front')),
                 s=th(fo), wrap_mode=None)
    df = add_par(df, p='rear_orientation_unwrapped', k='rou', u=1 * siu.deg, o=Larva, d=nam.unwrap(nam.orient('rear')),
                 s=th(ro), wrap_mode=None)



    for k0, kv, ka, s in zip(['b', 'fou', 'rou'], ['bv', 'fov', 'rov'],
                          ['ba', 'foa', 'roa'], ['b', fo, ro]):
        df = add_diff_par(df, k0=k0)
        df = add_rate_par(df, k0=k0, k_den='dt', k=kv)

        if k0 == 'fou':
            k0 = 'fo'
        elif k0 == 'rou':
            k0 = 'ro'
        df[kv].d = nam.vel(df[k0].d)
        df[kv].s = dot_th(s)
        df = add_diff_par(df, k0=kv)
        df = add_rate_par(df, k0=kv, k_den='dt', k=ka)
        df[ka].d = nam.acc(df[k0].d)
        df[ka].s = ddot_th(s)
        # df[ka].d = f'{df[k0].d} acceleration'

    for k0 in ['b', 'bv'] :
        df=add_mean_par(df,k0=k0)
        df=add_std_par(df,k0=k0)

    return df

def build_chunk_par_dict(df) :
    for kc,pc in chunk_dict.items() :
        df=add_chunk(df, pc=pc, kc=kc)
        for k in ['x', 'y', 'fo', 'fou', 'fov','ro', 'rou', 'rov', 'b', 'bv', 'v', 'sv', 'o_cent', 'o_chem', 'd_cent', 'd_chem', 'sd_cent', 'sd_chem'] :
            df=add_chunk_track(df, kc=kc, k=k)
        if pc=='stride':
            for k in ['d','sd', 'std', 'sstd']:
            # for k in [nam.cum('d'), nam.cum('sd')]:
                df = add_par(df, p=nam.chunk_track(pc, df[k].p), k=f'{kc}_{k}', u=df[k].u, d=nam.chunk_track(pc, df[k].p),
                          s=sub(Delta(df[k].s), kc), exists=False)
                df=add_mean_par(df, k0=f'{kc}_{k}')
                df=add_std_par(df, k0=f'{kc}_{k}')
    return df


def build_par_dict(save=True, df=None):
    siu.day = siu.s * 24 * 60 * 60
    siu.cm = siu.m * 10 ** -2
    siu.mm = siu.m * 10 ** -3
    siu.g = siu.kg * 10 ** -3
    # siu.deg = siu.rad / 180*np.pi
    siu.deg = siu.I.rename("deg", "deg", "plain angle")
    siu.microM = siu.mol * 10 ** -6
    if df is None :
        df =build_constants()

    df=build_DEB_par_dict(df)
    df=build_gut_par_dict(df)

    # df = add_par(df, p='dt', k='dt', u=1 * siu.s, o=Larva, d='dt', s='dt')
    # df = add_par(df, p='cum_dur', k='t', u=1 * siu.s, o=Larva, d=nam.cum('dur'), s='t')
    df = add_cum_par(df, p='cum_dur', k0='dt', d=nam.cum('dur'), k=nam.cum('t'), s=sub('t', 'cum'))
    # df = add_par(df, p='cum_dur', k='t', u=1 * siu.s, o=Larva, d='time', s='t')
    # df = add_diff_par(df, k0='t')


    df=build_spatial_par_dict(df)
    df=build_angular_par_dict(df)
    df=build_chunk_par_dict(df)

    df = add_par(df, p='amount_eaten', k='f_am', u=1 * siu.m ** 3, o=Larva, d='ingested_food_volume',
                 s=sub('V', 'in'))
    df = add_par(df, p='scaled_amount_eaten', k='sf_am', o=Larva, d='ingested_food_volume_ratio',
                 s=sub('[V]', 'in'))
    df = add_par(df, p='lin_activity', k='Act_cr', o=Larva, d='crawler output', s=sub('A', 'crawl'))
    df = add_par(df, p='ang_activity', k='Act_tur', o=Larva, d='turner output', s=subsup('A', 'tur', 'out'))
    df = add_par(df, p='turner_activation', k='A_tur', o=Larva, d='turner input', s=subsup('A', 'tur', 'in'),lim=(10, 40))
    df = add_par(df, p='olfactory_activation', k='A_olf', o=Larva, d='olfactory activation',s=sub('A', 'olf'), lim=(-1, 1))
    df = add_par(df, p='exploitVSexplore_balance', k='EEB', o=Larva, d='exploitVSexplore_balance',s='EEB', lim=(0, 1))




    for i, n in enumerate(['first', 'second', 'third']):
        k = f'c_odor{i + 1}'
        df = add_par(df, p=f'{n}_odor_concentration', k=k, u=1 * siu.microM, o=Larva,d=f'Odor {i+1} Conc', s=sub('C', i+1))
        df = add_par(df, p=f'{n}_odor_concentration_change', k=f'd{k}', u=1 * siu.microM, o=Larva,d=f'Odor {i+1} DConc', s=sub(dot('C'), i+1))
        kk = f'g_odor{i + 1}'
        df = add_par(df, p=f'{n}_odor_best_gain', k=kk, o=Larva, d=f'Odor {i + 1} Gain',
                     s=sub('G', i + 1))
        # df = add_par(df, p=f'{n}_odor_concentration_change', k=f'd{k}', u=1 * siu.microM, o=Larva,
        #              d=f'Odor {i + 1} DConc', s=sub(dot('C'), i + 1))

        # df = add_diff_par(df, k0=k)
        # df = add_rate_par(df, k0=k, k=f'dc_odor{i+1}',k_den='dt', s=sub(dot('C'), i+1))





    #



    for k, p in df.items():
        p.par_dict = df
    if save:
        save_ParDict_frame(df)
    return df

def post_get_par(dataset, par):
    # print(dataset.step_data)
    k=getPar(d=par, to_return=['k'])[0]
    p=ParDict[k]
    vs=[p.get_from(o,u=False,tick=None, df=dataset.step_data) for o in dataset.agent_ids]
    # print(vs)
    return vs
chunk_dict={
    'str' : 'stride',
    'pau' : 'pause',
    'fee' : 'feed',
    'tur' : 'turn',
    'Ltur' : 'Lturn',
    'Rtur' : 'Return',
    'str_c' : nam.chain('stride'),
    'fee_c' : nam.chain('feed')
            }

collection_dict = {
    'bouts': ['x', 'y', 'b', 'fou', 'rou','v', 'sv', 'd', 'fov', 'bv', 'sd', 'o_cent'],
    # 'bouts': ['str_d_mu', 'str_sd_mu'],
    'basic': ['x', 'y', 'b', 'fo'],
    'e_basic': ['l_mu', nam.cum('d'), f'{nam.cum("sd")}', nam.cum('t'), 'x', 'y', 'sv_mu'],
    'e_dispersion': ['dsp', 'sdsp', 'dsp_max', 'sdsp_max', 'dsp_0_40', 'dsp_0_80', 'dsp_20_80','sdsp_0_40', 'sdsp_0_80', 'sdsp_20_80'],
    'spatial': fun.flatten_list([[k, f's{k}'] for k in
                                 ['dsp', 'd', 'v', 'a', 'D_x', 'xv', 'xa', 'D_y', 'yv', 'ya', nam.cum('d'),
                                  nam.cum('D_x'), nam.cum('D_y'), ]]),
    'e_spatial' : [f'tor{i}_mu' for i in ['',2,5,10,20]],
    'angular': ['b', 'bv', 'ba', 'fo', 'fov', 'foa', 'ro', 'rov', 'roa'],

    'chemorbit': ['d_cent', 'sd_cent', 'o_cent'],
    'e_chemorbit': fun.flatten_list([[k, f'{k}_mu', f'{k}_std', f'{k}_max', f'{k}_fin'] for k in ['d_cent', 'sd_cent']]),
    'chemotax': ['d_chem', 'sd_chem', 'o_chem'],
    'e_chemotax': fun.flatten_list([[k, f'{k}_mu', f'{k}_std', f'{k}_max', f'{k}_fin'] for k in ['d_chem', 'sd_chem']]),

    'olfactor': ['Act_tur', 'A_tur', 'A_olf'],
    'odors': ['c_odor1', 'c_odor2', 'c_odor3', 'dc_odor1', 'dc_odor2', 'dc_odor3'],
    # 'constants': ['dt', 'x0', 'y0'],
}

combo_collection_dict = {
    'pose': {'step': ['basic', 'bouts', 'spatial', 'angular'], 'end': ['e_basic', 'e_dispersion']},
    'source_vincinity': {'step': ['chemorbit'], 'end': ['e_chemorbit']},
    'source_approach': {'step': ['chemotax'], 'end': ['e_chemotax']},
    'olfactor': {'step': ['odors', 'olfactor'], 'end': []},
}


def load_ParDict():
    # import lib.aux.functions as fun
    dic = fun.load_dicts([paths.ParDict_path])[0]
    return dic


def save_ParDict_frame(df, save_pdf=False):
    import inspect
    args = list(inspect.signature(Parameter.__init__).parameters.keys())
    args = [a for a in args if a not in ['self', 'par_dict']]
    d = {k: {a: getattr(p, a) for a in args} for k, p in df.items()}
    fun.save_dict(d, paths.ParDict_path)
    if save_pdf :
        dd= [{'symbol': p.s, 'unit': p.u.unit.abbrev, 'codename': k, 'interpretation': p.d} for i,(k, p) in enumerate(df.items()) if 240<i<280]
        ddf = pd.DataFrame.from_records(dd)
        ws = np.array([1, 1, 1, 5])
        ws = (ws / sum(ws))
        ddf.to_csv(paths.ParDf_path)
        fun.df2pdf(ddf, paths.ParPdf_path, colWidths=ws, edges='horizontal')
    return d


def reconstruct_ParDict(test=False):
    frame = fun.load_dicts([paths.ParDict_path])[0]
    dic = {}
    for k, args in frame.items():
        dic[k] = Parameter(**args, par_dict=dic)
    for k, p in dic.items():
        p.par_dict = dic

    # Test
    if test:
        dic0 = build_par_dict()
        for a in args:
            if a != 'self':
                print(a, all([getattr(dic0[k], a) == getattr(dic[k], a) for k in dic.keys()]))
    return dic


ParDict=build_par_dict()
ParFrame = load_ParDict()

runtime_pars=[v['d'] for k,v in ParFrame.items() if v['o']==Larva and not k in build_constants().keys()]

def getPar(k=None,p=None, d=None, to_return=['d', 'l'], new_format=True) :
    if new_format is None :
        new_format=paths.new_format
    dic={
        'd' : 'par',
        'l' : 'unit',
        's' : 'symbol',
        'lim' : 'lim',
         }
    if new_format :
        if k is None :
            if p is not None :
                if type(p) == str:
                    k=[k for k in ParFrame.keys() if ParFrame[k]['p']==p][0]
                elif type(p) == list:
                    k = fun.flatten_list([[k for k in ParFrame.keys() if ParFrame[k]['p'] == p0][0] for p0 in p])
            elif d is not None :
                if type(d) == str:
                    k=[k for k in ParFrame.keys() if ParFrame[k]['d']==d][0]
                elif type(d) == list:
                    k = fun.flatten_list([[k for k in ParFrame.keys() if ParFrame[k]['d'] == d0][0] for d0 in d])
        if type(k) == str:
            return [ParFrame[k][i] for i in to_return]
        elif type(k) == list:
            return [[ParFrame[kk][i] for kk in k] for i in to_return]
    else :
        if type(k)==str :
            res = par_dict_lists(shorts=[k], to_return=[dic[i] for i in to_return])
            return [r[0] for r in res]
        elif type(k)==list :
            res = par_dict_lists(shorts=k, to_return=[dic[i] for i in to_return])
            return res

if __name__ == '__main__':
    # for short in ['f_am', 'sf_am_Vg', 'sf_am_V', 'sf_am_A', 'sf_am_M']:
    #     p = getPar(short, to_return=['d'])[0]
    #     print(p)
    # dic=build_par_dict()
    dic=load_ParDict()
    # print(getPar(d=nam.scal(nam.cum(nam.dst(''))), to_return=['k']))
    # print(dic['D_olf'])
    # print(dic['D_olf'])
    # d,u=getPar('cum_d', to_return=['d', 'u'])
    # print(u.unit==siu.m)

    raise
    for k in ParFrame.keys() :
        try :
            a=getPar(k, ['u'])[0].unit
            b=get_unit(getPar(k, ['d'])[0])
            print(a==b)
        except :
            pass
    # print(a, type(a))
    # print(b, type(b))

    # print(getPar('tur_fo0'))
    # print(getPar('tur_fo1'))
    pass
    # print(ParFrame['sv']['d'])
    # print(ParFrame['sv']['l'])
    # dic=build_par_dict()
    # print(dic['sv'].d)
    # print(dic['v'].d)
    # raise
    #
    #
    # import time
    # s0=time.time()
    # dic=build_par_dict(save=False)
    # s1 = time.time()
    # dic0=reconstruct_ParDict()
    # s2 = time.time()
    #
    # print(s1-s0, s2-s1)
    #
    # raise
    # import time
    # s=time.time()
    # a=build_par_dict(save=False)['sv'].d
    # # a=load_ParDict()['sv'].d
    # print(a)
    # e = time.time()
    # print(e-s)
    #
    # raise
    #
    #
    #
    # deb = DEB(id='test_DEB', steps_per_day=24*60)
    # deb.grow_larva()
    # print(deb.fr_feed)
    # dic=load_ParDict()
    # raise
    # for i in range(5) :
    #     for k,v in dic.items() :
    #         if k in ['fr_f'] :
    #             print(k, v.get_from(deb))
    #     deb.run()
    # raise
    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(10), np.arange(10))
    # plt.xlabel(df['unit'].iloc[1]/1973*u.day)
    # # plt.xlabel([d[k].symbol for k in list(d.keys())])
    # plt.show()
