import copy
import os
from operator import attrgetter
import siunits as siu
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import euclidean

from lib.aux.ang_aux import angle_to_x_axis, angle_dif
from lib.aux.dictsNlists import flatten_list, save_dict, load_dicts, unique_list

from lib.aux import naming as nam
from lib.conf.base.dtypes import null_dict
from lib.conf.base import paths
from lib.aux.par_aux import bar, wave, sub, subsup, th, Delta, dot, circledcirc, circledast, odot, paren, brack, dot_th, ddot_th

from lib.model.agents._larva import Larva





def split_si_composite(df) :
    ddf=copy.deepcopy(df)
    d={}
    for c in df.columns :
        ddf[c] = df[c].apply(attrgetter('coef'))
        d[c] = df[c].iloc[0].unit
    return ddf, d

class AgentCollector:
    def __init__(self, collection_name, object, par_dict,collection_dict,  save_as=None, save_to=None, object_class=None, **kwargs):
        if save_as is None:
            save_as = f'{object.unique_id}.csv'
        self.save_as = save_as
        self.save_to = save_to
        self.object = object
        self.collection_name = collection_name
        par_dict.set_object(self.object)
        pars = [par_dict[k] for k in collection_dict[collection_name]]
        self.dict = {p.d: p for p in pars}
        self.table = {p.d: [] for p in pars}
        self.tick = 0
        self.object_class = self.set_class(object_class, object)

    def collect(self, u=True, tick=None, df=None):
        if tick is None:
            tick = self.tick
        for n, p in self.dict.items():
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
                pd.DataFrame(self.table).to_csv(f, index=True, header=True)
            else:
                save_dict(self.table, f)

    def set_class(self, object_class, object):
        if object_class is not None:
            c= object_class
        else:
            os = [p.o for p in pars if p.o is not None]
            os = unique_list(os)

            if len(os) == 0:
                c = None
            elif len(os) == 1:
                c = os[0]
            else:
                raise ValueError('Not all parameters have the same object_class class')
        if c is not None :
            if not isinstance(object, c):
                raise ValueError(
                    f'Parameter Collection {self.collection_name} collected from {c} not from {type(object)}')


class GroupCollector:
    def __init__(self, objects, name, save_to=None, save_as=None, common=False, save_units=True, **kwargs):
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
            dddf, u_dict = split_si_composite(ddf)
            if save_to is None:
                save_to = self.save_to
            if save_to is not None:
                os.makedirs(save_to, exist_ok=True)
                f = f'{save_to}/{self.save_as}'

                if save_units_dict:
                    ff = f'{save_to}/units.csv'
                    save_dict(u_dict, ff)

                if as_df:
                    dddf.to_csv(f, index=True, header=True)
                else:
                    save_dict(dddf.to_dict(), f)
            return dddf, u_dict


class CompGroupCollector(GroupCollector):
    def __init__(self, names, save_to=None, save_as=None, save_units=True, **kwargs):
        # print('ss')
        # self.par_dict = ParDict(mode='load')
        # print('sss')
        if save_as is None:
            save_as = 'complete.csv'
        self.save_units = save_units
        self.save_as = save_as
        self.save_to = save_to
        self.collectors = [GroupCollector(name=n, save_to=save_to, save_units=save_units, **kwargs) for n in names]
        self.tick = 0
        # print('ssss')

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
                save_dict(df0.to_dict(), f)
            if save_units_dict:
                f0=paths.path('Unit')
                try:
                    uu_dict = load_dicts([f0])[0]
                    u0_dict.update(uu_dict)
                except:
                    pass
                save_dict(u0_dict, f0)
        return df0


class Parameter:
    def __init__(self, p, u, k=None, s=None, o=Larva, lim=None,
                 d=None, lab=None, exists=True, func=None, const=None, par_dict=None, fraction=False,
                 operator=None, k0=None, k_num=None, k_den=None, dst2source=None, or2source=None, dispersion=False,
                 wrap_mode=None,l=None):
        # print(p,k)
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
        self.lab = lab

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
        if wrap_mode == 'positive':
            if lim is None:
                if self.u.unit == siu.deg:
                    lim = (0.0, 360.0)
                    # self.range=360
                elif self.u.unit == siu.rad:
                    lim = (0.0, 2 * np.pi)
                    # self.range = 2*np.pi

        elif wrap_mode == 'zero':
            if lim is None:
                if self.u.unit == siu.deg:
                    lim = (-180.0, 180.0)
                    # self.range = 360
                elif self.u.unit == siu.rad:
                    lim = (-np.pi, np.pi)
                    # self.range = 2 * np.pi

        # else :
        # self.range=None
        self.lim = lim

        self.range = lim[1] - lim[0] if lim is not None else None

        # print(self.k, self.dispersion, self.)

    @property
    def l(self):
        return f'{self.d},  {self.s}$({self.u.unit.abbrev})$' if self.lab is None else self.lab

    def get_from(self, o, u=True, tick=None, df=None):
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
                        from lib.aux.sim_aux import freq
                        vs = df[self.p0.d].xs(o.unique_id, level='AgentID').dropna()
                        dt = self.par_dict['dt'].get_from(o, u=False)
                        v = freq(vs, dt)
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
                v = euclidean(self.xy(o=o, tick=tick, df=df), self.dst2source)
            elif self.dispersion:
                v = euclidean(self.xy(o=o, tick=tick, df=df), self.xy0(o=o, df=df))
            elif self.or2source is not None:
                v = angle_dif(getattr(o, 'front_orientation'),
                                              angle_to_x_axis(self.xy(o=o, tick=tick, df=df), self.or2source))
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
            xys = df[nam.xy('')].xs(o, level='AgentID')
            xy = tuple(xys.values[-1, :])
        else:
            if tick is not None:
                xy = getattr(o, 'trajectory')[tick]
            else:
                xy = getattr(o, 'pos')
        return xy
        # return getattr(o, 'trajectory')[tick]

    def xy0(self, o, df=None):
        if type(o) == str and df is not None:
            xys = df[nam.xy('')].xs(o, level='AgentID')
            xy0 = tuple(xys.values[0, :])
        else:
            xy0 = getattr(o, 'initial_pos')
        return xy0

    def d_2D(self, o, t0, t1):
        dt = self.par_dict['dt'].get_from(o, u=False)
        tick0 = int(t0 / dt)
        tick1 = int(t1 / dt)
        # xs = df['x'].xs(o.unique_id, level='AgentID').values
        # ys = df['y'].xs(o.unique_id, level='AgentID').values
        # xy0=xs[tick0], ys[tick0]
        # xy1=xs[tick1], ys[tick1]
        d = euclidean(self.xy(o, tick1), self.xy(o, tick0))
        return d

    def postprocess(self, v):
        if self.operator == 'diff':
            vv = v - self.previous
            v0 = v
        elif self.operator == 'cum':
            vv = np.nansum([v, self.previous])
            v0 = vv
        elif self.range is not None and self.wrap_mode is not None:
            vv = v % self.range
            if vv > self.lim[1]:
                vv -= self.range
            v0 = v
        else:
            vv = v
            v0 = v
        self.previous = v0
        self.current = vv
        return vv


class ParDict:
    def __init__(self, mode='load', object=None, save=True):
        if mode=='load' :
            self.load()
        elif mode=='reconstruct' :
            self.reconstruct()
        elif mode=='build' :
            self.build(save=save, object=object)

    def set_object(self, object):
        for k in self.build_constants().keys():
            self.dict[k].const = self.dict[k].get_from(object, u=False, tick=None, df=None)

    def getPar(self, k=None, p=None, d=None, to_return=['d', 'l']):
        PF=self.dict
        if k is None:
            if p is not None:
                if type(p) == str:
                    k = [k for k in PF.keys() if PF[k]['p'] == p][0]
                elif type(p) == list:
                    k = flatten_list([[k for k in PF.keys() if PF[k]['p'] == p0][0] for p0 in p])
            elif d is not None:
                if type(d) == str:
                    k = [k for k in PF.keys() if PF[k]['d'] == d][0]
                elif type(d) == list:
                    k = flatten_list([[k for k in PF.keys() if PF[k]['d'] == d0][0] for d0 in d])
        if type(k) == str:
            return [PF[k][i] for i in to_return]
        elif type(k) == list:
            return [[PF[kk][i] for kk in k] for i in to_return]

    def add(self, dic=None, lab=None, **kwargs):
        if dic is None :
            dic=self.dict
        p = null_dict('par', **kwargs)
        k = p['k']
        if k in dic.keys():
            raise ValueError(f'Key {k} already exists')
        dic[k] = Parameter(**p, par_dict=dic, lab=lab)

    def add_diff(self, k0):
        b = self.dict[k0]
        self.add(p=f'D_{b.p}', k=f'D_{k0}', u=b.u, d=f'{b.d} change', s=Delta(b.s), exists=False,
                 operator='diff', k0=k0)

    def add_cum(self, k0, d=None, p=None, s=None, k=None):
        b = self.dict[k0]
        if d is None:
            d = nam.cum(b.d)
        if p is None:
            p = nam.cum(b.p)
        if s is None:
            s = sub(b.s, 'cum')
        if k is None:
            k = nam.cum(k0)
            # d = f'total {b.d}'
        self.add(p=p, k=k, u=b.u, d=d, s=s, exists=False, operator='cum', k0=k0)

    def add_mean(self, k0, d=None, s=None):
        b = self.dict[k0]
        if d is None:
            d = nam.mean(b.d)
        if s is None:
            s = bar(b.s)
        self.add(p=nam.mean(b.p), k=f'{b.k}_mu', u=b.u, d=d, s=s, exists=False, operator='mean', k0=k0)

    def add_std(self, k0, d=None, s=None):
        b = self.dict[k0]
        if d is None:
            d = nam.std(b.d)
        if s is None:
            s = wave(b.s)
        self.add(p=nam.std(b.p), k=f'{b.k}_std', u=b.u, d=d, s=s, exists=False, operator='std', k0=k0)

    def add_min(self, k0, d=None, s=None):
        b = self.dict[k0]
        if d is None:
            d = nam.min(b.d)
        if s is None:
            s = sub(b.s, 'min')
        self.add(p=nam.min(b.p), k=f'{b.k}_min', u=b.u, d=d, s=s, exists=False, operator='min', k0=k0)

    def add_max(self, k0, d=None, s=None):
        b = self.dict[k0]
        if d is None:
            d = nam.max(b.d)
        if s is None:
            s = sub(b.s, 'max')
        self.add(p=nam.max(b.p), k=f'{b.k}_max', u=b.u, d=d, s=s, exists=False, operator='max', k0=k0)

    def add_fin(self, k0, d=None, s=None):
        b = self.dict[k0]
        if d is None:
            d = nam.final(b.d)
        if s is None:
            s = sub(b.s, 'fin')
        self.add(p=nam.final(b.p), k=f'{b.k}_fin', u=b.u, d=d, s=s, exists=False, operator='final', k0=k0)

    def add_freq(self, k0, d=None):
        b = self.dict[k0]
        if d is None:
            d = nam.freq(b.d)
        self.add(p=nam.freq(b.p), k=f'f{b.k}', u=1 * siu.hz, d=d, s=sub(b.s, 'freq'), exists=False,
                 operator='freq',
                 k0=k0)

    def add_dsp(self, range=(0, 40)):
        a = 'dispersion'
        k0 = 'dsp'
        s0 = circledast('d')
        r0, r1 = range
        p = f'{a}_{r0}_{r1}'
        k = f'{k0}_{r0}_{r1}'
        self.add(p=p, k=k, d=p, s=subsup(s0, f'{r0}', f'{r1}'), exists=False,
                 func=('d_2D', {'t0': r0, 't1': r1}))
        self.add_scaled(k0=k, s=subsup(paren(k0), f'{r0}', f'{r1}'))
        for k00 in [k, f's{k}']:
            self.add_mean(k0=k00)
            self.add_std(k0=k00)
            self.add_min(k0=k00)
            self.add_max(k0=k00)
            self.add_fin(k0=k00)


    def add_rate(self, k0=None, k_time='t', p=None, k=None, d=None, s=None, k_num=None, k_den=None, **kwargs):
        if k0 is not None:
            b = self.dict[k0]
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

        self.add(p=p, k=k, d=d, s=s, exists=False, fraction=True, k_num=k_num, k_den=k_den, **kwargs)

    def add_Vspec(self, k0):
        b = self.dict[k0]
        self.add(p=f'[{k0}]', k=f'[{k0}]', d=f'volume specific {b.d}', s=f'[{b.s}]', exists=False,
                 fraction=True,
                 k_num=k0, k_den='V')

    def add_chunk(self, pc, kc):
        p0, p1, pt, pid, ptr, pN = nam.start(pc), nam.stop(pc), nam.dur(pc), nam.id(pc), nam.dur_ratio(pc), nam.num(pc)

        self.add(p=pc, k=kc, d=pc, s=f'${kc}$', exists=False)
        b = self.dict[kc]
        self.add(p=p0, k=f'{kc}0', u=1 * siu.s, d=p0, s=subsup('t', kc, 0), exists=False)
        self.add(p=p1, k=f'{kc}1', u=1 * siu.s, d=p1, s=subsup('t', kc, 1), exists=False)

        self.add(p=pid, k=f'{kc}_id', d=pid, s=sub('idx', kc), exists=False)
        self.add(p=ptr, k=f'{kc}_tr', d=ptr, s=sub('r', kc), exists=False)
        self.add(p=pN, k=f'{kc}_N', d=pN, s=sub('N', f'{pc}s'), exists=False)
        self.add_rate(k_num=f'{kc}_N', k_den=nam.cum('t'), k=f'{kc}_N_mu', p=nam.mean(pN), d=nam.mean(pN), s=bar(f'{kc}_N'))
        self.add(p=f'{nam.mean(pN)}_on_food', k=f'{kc}_N_mu_on_food')
        self.add(p=f'{nam.mean(pN)}_off_food', k=f'{kc}_N_mu_off_food')
        self.add(p=f'{ptr}_on_food', k=f'{kc}_tr_on_food')
        self.add(p=f'{ptr}_off_food', k=f'{kc}_tr_off_food')

        k00 = f'{kc}_t'
        s00 = Delta('t')
        self.add(p=pt, k=k00, u=1 * siu.s, d=pt, s=sub(s00, kc), exists=False)
        self.add_cum(k0=k00)
        self.add_mean(k0=k00, s=sub(bar(s00), kc))
        self.add_std(k0=k00, s=sub(wave(s00), kc))
        self.add_min(k0=k00)
        self.add_max(k0=k00)

        if str.endswith(pc, 'chain'):
            pl = nam.length(pc)
            k00 = f'{kc}_l'
            self.add(p=pl, k=k00, d=pl, s=sub('l', kc), exists=False)
            self.add_cum(k0=k00)
            self.add_mean(k0=k00, s=sub(bar('l'), kc))
            self.add_std(k0=k00, s=sub(wave('l'), kc))
            self.add_min(k0=k00)
            self.add_max(k0=k00)


    def add_chunk_track(self, kc, k, extrema=True):
        bc = self.dict[kc]
        b = self.dict[k]
        u = self.dict[k].u
        b0, b1 = self.dict[f'{kc}0'], self.dict[f'{kc}1']
        p0, p1 = nam.at(b.p, b0.p), nam.at(b.p, b1.p)

        k00 = f'{kc}_{k}'
        s00 = Delta(b.s)
        self.add(p=nam.chunk_track(bc.p, b.p), k=k00, u=u, d=nam.chunk_track(bc.p, b.p), s=sub(s00, kc),
                 exists=False)
        self.add_mean(k0=k00, s=sub(bar(s00), kc))
        self.add_std(k0=k00, s=sub(wave(s00), kc))
        if extrema:
            self.add(p=p0, k=f'{kc}_{k}0', u=u, d=p0, s=subsup(b.s, kc, 0), exists=False)
            self.add(p=p1, k=f'{kc}_{k}1', u=u, d=p1, s=subsup(b.s, kc, 1), exists=False)

    def add_scaled(self, k0, s=None):
        k_den = 'l'
        k_num = k0
        b = self.dict[k0]
        d = nam.scal(b.d)
        lim=tuple(np.array(b.lim)/0.002) if b.lim is not None else None
        if s is None:
            s = paren(b.s)
        self.add(p=d, k=f's{k0}', d=d, s=s, exists=False, fraction=True, k_num=k_num, k_den=k_den, lim=lim)

    def build_constants(self, object=None):
        dic={}
        self.add(dic=dic,p='x0', k='x0', u=1 * siu.m, d='initial x coordinate', s=sub('x', 0))
        self.add(dic=dic,p='y0', k='y0', u=1 * siu.m, d='initial y coordinate', s=sub('y', 0))
        self.add(dic=dic,p='dt', k='dt', u=1 * siu.s, d='timestep', s='$dt$')
        self.add(dic=dic,p='real_length', k='l', u=1 * siu.m, d='length', s='l')
        if object is not None :
            for k, p in dic.items():
                p.const = p.get_from(object, u=False, tick=None, df=None)
        return dic

    def build_DEB(self):
        from lib.model.DEB.deb import DEB
        self.add(p='L', k='L', u=1 * siu.cm, o=DEB, d='structural length', s='L')
        self.add(p='Lw', k='Lw', u=1 * siu.cm, o=DEB, d='physical length', s=sub('L', 'w'))
        self.add(p='V', k='V', u=1 * siu.cm ** 3, o=DEB, d='structural volume', s='V')
        self.add(p='Ww', k='Ww', u=1 * siu.g, o=DEB, d='wet weight', s=sub('W', 'w'))
        self.add(p='age', k='age', u=1 * siu.day, o=DEB, d='age', s='a')
        self.add(p='hunger', k='H', o=DEB, d='hunger drive', s='H')
        self.add(p='E', k='E', u=1 * siu.j, o=DEB, d='reserve energy', s='E')
        self.add(p='E_H', k='E_H', u=1 * siu.j, o=DEB, d='maturity energy', s=sub('E', 'H'))
        self.add(p='E_R', k='E_R', u=1 * siu.j, o=DEB, d='reproduction buffer', s=sub('E', 'R'))
        self.add(p='deb_p_A', k='deb_p_A', u=1 * siu.j, o=DEB, d='assimilation energy (model)',
                 s=subsup('p', 'A', 'deb'))
        self.add(p='sim_p_A', k='sim_p_A', u=1 * siu.j, o=DEB, d='assimilation energy (sim)',
                 s=subsup('p', 'A', 'sim'))
        self.add(p='gut_p_A', k='gut_p_A', u=1 * siu.j, o=DEB, d='assimilation energy (gut)',
                 s=subsup('p', 'A', 'gut'))
        self.add(p='e', k='e', o=DEB, d='scaled reserve density', s='e')
        self.add(p='f', k='f', o=DEB, d='scaled functional response', s='f')
        self.add(p='base_f', k='f0', o=DEB, d='base scaled functional response', s=sub('f', 0))
        self.add(p='F', k='[F]', u=siu.hz / (24 * 60 * 60), o=DEB, d='volume specific filtering rate',
                 s=brack(dot('F')))
        self.add(p='fr_feed', k='fr_f', u=1 * siu.hz, o=DEB, d='feed motion frequency (estimate)',
                 s=sub(dot('fr'), 'feed'))
        self.add(p='pupation_buffer', k='pupation', o=DEB, d='pupation ratio', s=sub('r', 'pup'))

        self.add_diff(k0='age')
        for k0 in ['f', 'e', 'H']:
            self.add_diff(k0=k0)
            self.add_rate(k0=k0, k_time='age')

        for k0 in ['E', 'Ww', 'E_R', 'E_H']:
            self.add_Vspec(k0=k0)


    def build_gut(self):
        self.add(p='ingested_volume', k='f_am_V', u=1 * siu.m ** 3, d='ingested_food_volume',
                 s=sub('V', 'in'), lab='ingested food volume')
        self.add(p='ingested_gut_volume_ratio', k='sf_am_Vg', d='ingested_gut_volume_ratio',
                 s=subsup('[V]', 'in', 'gut'), lab='intake as % larva gut volume')
        self.add(p='ingested_body_volume_ratio', k='sf_am_V', d='ingested_body_volume_ratio',
                 s=sub('[V]', 'in'), lab='intake as % larva volume')
        self.add(p='ingested_body_area_ratio', k='sf_am_A',d='ingested_body_area_ratio',
                 s=sub('{V}', 'in'), lab='intake as % larva area')
        self.add(p='ingested_body_mass_ratio', k='sf_am_M', d='ingested_body_mass_ratio',
                 s=sub('[M]', 'in'), lab='intake as % larva mass')


    def build_spatial(self):
        d = nam.dst('')
        std = nam.straight_dst('')
        self.add(p=d, k='d', u=1 * siu.m,  d=d, s='d')
        self.add(p=std, k='std', u=1 * siu.m,  d=std, s='d')
        self.add(p='dispersion', k='dsp', u=1 * siu.m,  d='dispersion', s=circledast('d'), exists=False,
                 dispersion=True)

        self.add(p=nam.dst2('center'), k='d_cent', u=1 * siu.m,  d=nam.dst2('center'), s=odot('d'),
                 exists=False, dst2source=(0, 0), lim=(0.0,0.02))
        self.add(p=nam.dst2('source'), k='d_chem', u=1 * siu.m, d=nam.dst2('source'),
                 s=circledcirc('d'), exists=False, dst2source=(0.04, 0.0), lim=(0.0,0.02))

        self.add(p='x', k='x', u=1 * siu.m,  d='x', s='x')
        self.add(p='y', k='y', u=1 * siu.m, d='y', s='y')

        self.add_diff(k0='x')
        self.add_diff(k0='y')

        space_ks = ['d', 'D_x', 'D_y', 'd_chem', 'd_cent', 'dsp', 'std']

        for k0 in space_ks:
            self.add_scaled(k0=k0)

        for k00 in space_ks:
            for k0 in [k00, f's{k00}']:
                self.add_cum(k0=k0)
                self.add_mean(k0=k0)
                self.add_std(k0=k0)
                self.add_min(k0=k0)
                self.add_max(k0=k0)
                self.add_fin(k0=k0)
        v = nam.vel('')
        a = nam.acc('')
        sv, sa = nam.scal([v, a])
        self.add_rate(k_num='d', k_den='dt', k='v', p=v, d=v, s='v')
        self.add_rate(k_num='v', k_den='dt', k='a', p=a, d=a, s='a')
        self.add_rate(k_num='sd', k_den='dt', k='sv', p=sv, d=sv, s=paren('v'))
        self.add_rate(k_num='sv', k_den='dt', k='sa', p=sa, d=sa, s=paren('a'))

        for i in [(0, 40), (0, 80), (20, 80)]:
            self.add_dsp(range=i)

        for k0 in ['l', 'v', 'sv']:
            self.add_mean(k0=k0)

        for k0 in ['sv']:
            self.add_freq(k0=k0)

        for i in ['', 2, 5, 10, 20]:
            if i == '':
                p = 'tortuosity'
            else:
                p = f'tortuosity_{i}'
            k0 = 'tor'
            k = f'{k0}{i}'
            self.add(p=p, k=k, d=p, s=sub(k0, i), exists=False)
            self.add_mean(k0=k, s=sub(bar(k0), i))
            self.add_std(k0=k, s=sub(wave(k0), i))

    def build_angular(self):
        self.add(p=nam.bearing2('center'), k='o_cent', u=1 * siu.deg, d=nam.bearing2('center'),
                 s=odot(th('or')), exists=False, or2source=(0, 0), wrap_mode='zero', lim=(-180.0,180.0))
        self.add(p=nam.bearing2('source'), k='o_chem', u=1 * siu.deg, d=nam.bearing2('source'),
                 s=circledcirc(th('or')), exists=False, or2source=(0.04, 0.0), wrap_mode='zero', lim=(-180.0,180.0))

        self.add(p='bend', k='b', u=1 * siu.deg, d='bend', s=th('b'), wrap_mode='zero')
        fo = sub('or', 'f')
        ro = sub('or', 'r')
        self.add(p='front_orientation', k='fo', u=1 * siu.deg, d=nam.orient('front'), s=th(fo),
                 wrap_mode='positive')
        self.add(p='rear_orientation', k='ro', u=1 * siu.deg, d=nam.orient('rear'), s=th(ro),
                 wrap_mode='positive')
        self.add(p='front_orientation_unwrapped', k='fou', u=1 * siu.deg,
                 d=nam.unwrap(nam.orient('front')),
                 s=th(fo), wrap_mode=None)
        self.add(p='rear_orientation_unwrapped', k='rou', u=1 * siu.deg,
                 s=th(ro), wrap_mode=None)

        for k0, kv, ka, s in zip(['b', 'fou', 'rou'], ['bv', 'fov', 'rov'],
                                 ['ba', 'foa', 'roa'], ['b', fo, ro]):
            self.add_diff(k0=k0)
            self.add_rate(k0=k0, k_den='dt', k=kv)

            if k0 == 'fou':
                k0 = 'fo'
            elif k0 == 'rou':
                k0 = 'ro'
            self.dict[kv].d = nam.vel(self.dict[k0].d)
            self.dict[kv].s = dot_th(s)
            self.add_diff(k0=kv)
            self.add_rate(k0=kv, k_den='dt', k=ka)
            self.dict[ka].d = nam.acc(self.dict[k0].d)
            self.dict[ka].s = ddot_th(s)

        for k0 in ['b', 'bv']:
            self.add_mean(k0=k0)
            self.add_std(k0=k0)

    def build_neural(self):
        self.add(p='amount_eaten', k='f_am', u=1 * siu.m ** 3, d='ingested_food_volume', s=sub('V', 'in'),
                 lab='food intake')
        self.add(p='cum_food_detected', k='cum_f_det', d='cum_food_detected', s=subsup('t', 'on food', 'cum'),
                 lab='time on food')
        self.add(p='on_food', k='on_food', d='on_food', s='on_food',lab='Is inside patches')
        self.add(p=f'{nam.mean(nam.vel(""))}_on_food', k='v_mu_on_food')
        self.add(p=f'{nam.mean(nam.vel(""))}_off_food', k='v_mu_off_food')
        self.add(p=nam.dur_ratio('on_food'), k='on_food_tr', d=nam.dur_ratio('on_food'), s=sub('r', 'on_food'),
                 lab='Fraction of time spent inside patches', lim=(0.0,1.0))
        self.add(p='scaled_amount_eaten', k='sf_am',  d='ingested_food_volume_ratio', s=sub('[V]', 'in'))
        self.add(p='lin_activity', k='Act_cr',  d='crawler output', s=sub('A', 'crawl'))
        self.add(p='ang_activity', k='Act_tur',  d='turner output', s=subsup('A', 'tur', 'out'), lim=(-20, 20))
        self.add(p='turner_activation', k='A_tur',  d='turner input', s=subsup('A', 'tur', 'in'), lim=(10, 40))
        self.add(p='olfactory_activation', k='A_olf',  d='olfactory activation', s=sub('A', 'olf'), lim=(-1, 1))
        self.add(p='touch_activation', k='A_touch',  d='tactile activation', s=sub('A', 'touch'), lim=(-1, 1))
        self.add(p='exploitVSexplore_balance', k='EEB',  d='exploitVSexplore_balance', s='EEB', lim=(0, 1))

        for i, n in enumerate(['first', 'second', 'third']):
            k = f'c_odor{i + 1}'
            self.add(p=f'{n}_odor_concentration', k=k, u=1 * siu.microM,  d=f'Odor {i + 1} Conc',
                     s=sub('C', i + 1), lim=(0.0,2.5))
            self.add(p=f'{n}_odor_concentration_change', k=f'd{k}', u=1 * siu.microM,
                     d=f'Odor {i + 1} DConc', s=sub(dot('C'), i + 1))
            kk = f'g_odor{i + 1}'
            self.add(p=f'{n}_odor_best_gain', k=kk,  d=f'Odor {i + 1} Gain',
                     s=sub('G', i + 1))

    def build_chunk(self):
        chunk_dict = {
            'str': 'stride',
            'pau': 'pause',
            'fee': 'feed',
            'tur': 'turn',
            'Ltur': 'Lturn',
            'Rtur': 'Rturn',
            'str_c': nam.chain('stride'),
            'fee_c': nam.chain('feed')
        }
        for kc, pc in chunk_dict.items():
            self.add_chunk(pc=pc, kc=kc)
            for k in ['x', 'y', 'fo', 'fou', 'fov', 'ro', 'rou', 'rov', 'b', 'bv', 'v', 'sv', 'o_cent', 'o_chem',
                      'd_cent',
                      'd_chem', 'sd_cent', 'sd_chem']:
                self.add_chunk_track(kc=kc, k=k)
            if pc == 'stride':
                for k in ['d', 'std']:
                    self.add(p=nam.chunk_track(pc, self.dict[k].p), k=f'{kc}_{k}', u=self.dict[k].u,
                             d=nam.chunk_track(pc, self.dict[k].p),
                             s=sub(Delta(self.dict[k].s), kc), exists=False)
                    self.add_mean(k0=f'{kc}_{k}')
                    self.add_std(k0=f'{kc}_{k}')
                    for k0 in [f'{kc}_{k}', f'{kc}_{k}_mu', f'{kc}_{k}_std']:
                        self.add_scaled(k0=k0)
        self.add_rate(k_num='Ltur_N', k_den='tur_N', k='tur_H', p='handedness_score', d='handedness_score',
                      s=sub('H', 'tur'), lim=(0.0,1.0), lab='Handedness score')
        self.add(p=f'handedness_score_on_food', k='tur_H_on_food')
        self.add(p=f'handedness_score_off_food', k='tur_H_off_food')


    def build(self, save=True, object=None):
        siu.day = siu.s * 24 * 60 * 60
        siu.cm = siu.m * 10 ** -2
        siu.mm = siu.m * 10 ** -3
        siu.g = siu.kg * 10 ** -3
        siu.deg = siu.I.rename("deg", "deg", "plain angle")
        siu.microM = siu.mol * 10 ** -6
        self.dict=self.build_constants(object=object)

        self.build_DEB()
        self.build_gut()

        self.add_cum(p='cum_dur', k0='dt', d=nam.cum('dur'), k=nam.cum('t'), s=sub('t', 'cum'))

        self.build_spatial()
        self.build_angular()
        self.build_chunk()
        self.build_neural()


        for k, p in self.dict.items():
            p.par_dict = self.dict
        if save:
            self.save()

    def save(self, save_pdf=False):
        import inspect
        args = list(inspect.signature(Parameter.__init__).parameters.keys())
        args = [a for a in args if a not in ['self', 'par_dict']]
        d = {k: {a: getattr(p, a) for a in args} for k, p in self.dict.items()}
        save_dict(d, paths.path('ParDict'))
        if save_pdf:
            def df2pdf(df, path, **kwargs):
                # https://stackoverflow.com/questions/32137396/how-do-i-plot-only-a-table-in-matplotlib
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.axis('tight')
                ax.axis('off')
                the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', **kwargs)
                # the_table.set_fontsize(20)
                the_table.scale(1, 2)
                from matplotlib.font_manager import FontProperties

                for (row, col), cell in the_table.get_celld().items():
                    if (row == 0) or (col == -1):
                        cell.set_text_props(fontproperties=FontProperties(weight='bold'))

                # https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
                pp = PdfPages(path)
                pp.savefig(fig, bbox_inches='tight')
                pp.close()
            dd = [{'symbol': p.s, 'unit': p.u.unit.abbrev, 'codename': k, 'interpretation': p.d} for i, (k, p) in
                  enumerate(self.dict.items()) if 240 < i < 280]
            ddf = pd.DataFrame.from_records(dd)
            ws = np.array([1, 1, 1, 5])
            ws = (ws / sum(ws))
            ddf.to_csv(paths.path('ParDf'))
            df2pdf(ddf, paths.path('ParPdf'), colWidths=ws, edges='horizontal')

    def load(self):
        self.dict = load_dicts([paths.path('ParDict')])[0]

    def reconstruct(self):
        frame = load_dicts([paths.path('ParDict')])[0]
        for k, args in frame.items():
            self.dict[k] = Parameter(**args, par_dict=self.dict)
        for k, p in self.dict.items():
            p.par_dict = self.dict

    def runtime_pars(self):
        return [v['d'] for k, v in self.dict.items() if v['o'] == Larva and not k in self.build_constants().keys()]


chunk_dict = {
    'str': 'stride',
    'pau': 'pause',
    'fee': 'feed',
    'tur': 'turn',
    'Ltur': 'Lturn',
    'Rtur': 'Rturn',
    'str_c': nam.chain('stride'),
    'fee_c': nam.chain('feed')
}

def runtime_pars( PF=None) :
    if PF is None :
        PF = ParDict(mode='load').dict
    return [v['d'] for k, v in PF.dict.items() if v['o'] == Larva and not k in PF.build_constants().keys()]


def getPar(k=None, p=None, d=None, to_return=['d', 'l'], PF=None):
    if PF is None :
        PF = ParDict(mode='load').dict
    if k is None:
        if p is not None:
            if type(p) == str:
                k = [k for k in PF.keys() if PF[k]['p'] == p][0]
            elif type(p) == list:
                k = flatten_list([[k for k in PF.keys() if PF[k]['p'] == p0][0] for p0 in p])
        elif d is not None:
            if type(d) == str:
                k = [k for k in PF.keys() if PF[k]['d'] == d][0]
            elif type(d) == list:
                k = flatten_list([[k for k in PF.keys() if PF[k]['d'] == d0][0] for d0 in d])
    if type(k) == str:
        return [PF[k][i] for i in to_return]
    elif type(k) == list:
        return [[PF[kk][i] for kk in k] for i in to_return]



if __name__ == '__main__':
    # o, d = nam.bearing2('n'), nam.dst2('n')
    # fo = getPar(['fo'], to_return=['d'])[0][0]
    # print(o,d)
    # d=ParDict(mode='build').dict
    print(getPar(['D_olf'], to_return=['d', 's', 's', 'l', 'lim']))
    # # d = ParDict(mode='reconstruct').dict
    # # print(d.keys())
    raise
    # for short in ['f_am', 'sf_am_Vg', 'sf_am_V', 'sf_am_A', 'sf_am_M']:
    #     p = getPar(short, to_return=['d'])[0]
    #     print(p)
    # dic = build_par_dict()
    # print(dic.keys())
    # print(runtime_pars)
    # dic=load_ParDict()
    # print(getPar(k='sstr_d_mu'))
    # print(getPar(k='str_sd_mu'))
    # print(dic['b'])
    # print(dic['D_olf'])
    # d,u=getPar('cum_d', to_return=['d', 'u'])
    # print(u.unit==siu.m)

    # raise
    # for k in PF.keys() :
    #     try :
    #         a=getPar(k, ['u'])[0].unit
    #         b=get_unit(getPar(k, ['d'])[0])
    #         print(a==b)
    #     except :
    #         pass
    # # print(a, type(a))
    # # print(b, type(b))
    #
    # # print(getPar('tur_fo0'))
    # # print(getPar('tur_fo1'))
    # pass
    # # print(PF['sv']['d'])
    # # print(PF['sv']['l'])
    # # dic=build_par_dict()
    # # print(dic['sv'].d)
    # # print(dic['v'].d)
    # # raise
    # #
    # #
    # # import time
    # # s0=time.time()
    # # dic=build_par_dict(save=False)
    # # s1 = time.time()
    # # dic0=reconstruct_ParDict()
    # # s2 = time.time()
    # #
    # # print(s1-s0, s2-s1)
    # #
    # # raise
    # # import time
    # # s=time.time()
    # # a=build_par_dict(save=False)['sv'].d
    # # # a=load_ParDict()['sv'].d
    # # print(a)
    # # e = time.time()
    # # print(e-s)
    # #
    # # raise
    # #
    # #
    # #
    # # deb = DEB(id='test_DEB', steps_per_day=24*60)
    # # deb.grow_larva()
    # # print(deb.fr_feed)
    # # dic=load_ParDict()
    # # raise
    # # for i in range(5) :
    # #     for k,v in dic.items() :
    # #         if k in ['fr_f'] :
    # #             print(k, v.get_from(deb))
    # #     deb.run()
    # # raise
    # # import matplotlib.pyplot as plt
    # # plt.plot(np.arange(10), np.arange(10))
    # # plt.xlabel(df['unit'].iloc[1]/1973*u.day)
    # # # plt.xlabel([d[k].symbol for k in list(d.keys())])
    # # plt.show()
    pars, sim_ls, exp_ls, xlabs, xlims = getPar(['str_N', 'str_tr', 'cum_d'], to_return=['d', 's', 's', 'l', 'lim'])
    print(pars, xlims)
