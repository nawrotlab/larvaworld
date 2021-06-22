import copy
import os
from typing import Tuple, Type, Union

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from siunits import Composite, DerivedUnit, BaseUnit

from lib.aux import functions as fun
from lib.aux import naming as nam
from lib.stor import paths
from lib.conf.par_conf import sup, sub, th, dot, ddot, subsup, Delta, delta
from lib.model.DEB.deb import DEB
from lib.model.agents._agent import LarvaworldAgent
import siunits as siu

import lib.conf.dtype_dicts as dtypes
# default_unit_dict = {
#     'distance': 'm',
#     'length': 'm',
#     'time': 'sec',
#     'mass': 'g',
#     'angle': 'deg',
#     'frequency': 'Hz',
#     'energy': 'J',
# }
#
#
# def unit_conversion(u0, u1):
#     if u0 == f'm{u1}':
#         return 10 ** -3
#     elif u1 == f'm{u0}':
#         return 10 ** 3
#     if u0 == f'c{u1}':
#         return 10 ** -2
#     elif u1 == f'c{u0}':
#         return 10 ** 2
#     if u0 == 'mm' and u1 == 'cm':
#         return 10 ** -1
#     elif u1 == 'mm' and u0 == 'cm':
#         return 10 ** 1
#     if u0 == 'deg' and u1 == 'rad':
#         return 1 / (180 * np.pi)
#     elif u1 == 'deg' and u0 == 'rad':
#         return 180 * np.pi
#     elif u0 == 'sec':
#         if u1 == 'min':
#             return 1 / 60
#         elif u1 == 'hours':
#             return 1 / (60 * 60)
#         elif u1 == 'days':
#             return 1 / (60 * 60 * 24)
#     elif u0 == 'min':
#         if u1 == 'hours':
#             return 1 / 60
#         elif u1 == 'days':
#             return 1 / (60 * 24)
#         elif u1 == 'sec':
#             return 60
#     elif u0 == 'hours':
#         if u1 == 'days':
#             return 1 / 24
#         elif u1 == 'min':
#             return 60
#         elif u1 == 'sec':
#             return 60 * 60
#     elif u0 == 'days':
#         if u1 == 'hours':
#             return 24
#         elif u1 == 'min':
#             return 24 * 60
#         elif u1 == 'sec':
#             return 24 * 60 * 60


# new_par_dict = {}


class Parameter:
    def __init__(self, p, u, k=None, s=None, o=None, lim=None,
                 d=None, exists=True, func=None, const=None, par_dict=None, fraction=False,
                 diff=False, cum=False, k0=None, k_num=None, k_den=None):
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

        if o is None:
            o = LarvaworldAgent
        self.o = o
        self.lim = lim
        if d is None:
            d = p
        self.d = d
        self.const = const
        self.diff = diff
        self.cum = cum
        self.k0 = k0
        self.k_num = k_num
        self.k_den = k_den
        self.p0 = par_dict[k0] if k0 is not None else None
        self.p_num = par_dict[k_num] if k_num is not None else None
        self.p_den = par_dict[k_den] if k_den is not None else None
        self.previous = np.nan
        if self.p_num is not None and self.p_den is not None :
            # print(self.par, unit, self.numerator_par.unit, self.denominator_par.unit, self.numerator_par.unit*self.denominator_par.unit)
            u = self.p_num.u/self.p_den.u
            # print(self.par, unit)
        elif u is None :
            u=1*siu.j**0
        self.u = u

    def get_from(self, o, u=True):
        # print(self.p,o)
        if self.const is not None :
            v = self.const
        elif self.func is not None:
            v = self.func(o)
        elif self.exists:
            v = getattr(o, self.p)
        elif self.p0 is not None:
            v = self.p0.get_from(o, u=False)
            # print(self.key, self.base_par.key, v)
        elif self.fraction :
            v_n = self.p_num.get_from(o, u=False)
            v_d = self.p_den.get_from(o, u=False)
            v = v_n / v_d
        # v = self.preprocess(v)
        v = self.postprocess(v)
        if u :
            v*=self.u
        return v

    #
    # def preprocess(self, v):
    #     return v

    def postprocess(self,v):
        v0=self.previous
        if self.diff :
            self.previous=v
            return v - v0
        elif self.cum :
            self.previous=v+v0
            return v + v0
        else :
            self.previous = v
            return v



class Collection:
    def __init__(self, name, par_dict, keys=None, object_class=None):
        if keys is None :
            keys=collection_dict[name]
        # print(name, keys)
        # raise
        self.name = name
        pars=[par_dict[k] for k in keys]
        if object_class is not None:
            for p in pars:
                p.object_class = object_class
            self.object_class = object_class
        else:
            os = [p.object_class for p in pars]
            o = fun.unique_list(os)
            if len(o) != 1:
                raise ValueError('Not all parameters have the same object_class class')
            else:
                self.object_class = o[0]
        self.par_names = [p.disp for p in pars]
        # self.par_names = [p.disp for p in pars]
        self.par_dict = {p.disp: p for p in pars}
        # self.par_dict = {p.disp: p for p in pars}

    def get_from(self, object):
        if not isinstance(object, self.object_class):
            raise ValueError(f'Parameter Group {self.name} collected from {self.object_class} not from {type(object)}')
        dic = {n: p.get_from(object) for n, p in self.par_dict.items()}
        return dic


class AgentCollector:
    def __init__(self, collection, object, save_as=None, save_to=None):
        if save_as is None :
            save_as=f'{object.unique_id}.csv'
        self.save_as=save_as
        self.save_to=save_to
        self.collection = collection
        self.object = object
        self.table = {n: [] for n in self.collection.par_names}

    def collect(self):
        for n, p in self.collection.par_dict.items():
            # self.table[n].append(p.get_from(self.object))
            try:
                self.table[n].append(p.get_from(self.object))
            except:
                self.table[n].append(np.nan)

    def save(self):
        if self.save_to is not None :
            os.makedirs(self.save_to, exist_ok=True)
            f=f'{self.save_to}/{self.save_as}'
            df = pd.DataFrame(self.table)
            df.to_csv(f, index=True, header=True)

class GroupCollector :
    def __init__(self, objects, name, par_dict,save_to,save_as=None, common=False, **kwargs):
        if save_as is None :
            save_as = f'{name}.csv'
        self.save_as=save_as
        self.save_to = save_to
        self.common = common
        self.name = name
        self.collection = Collection(name, par_dict=par_dict)
        self.collectors=[AgentCollector(object=o, collection=self.collection,save_to=save_to, **kwargs) for o in objects]

    def collect(self):
        for c in self.collectors :
            c.collect()

    def save(self):
        if not self.common :
            for c in self.collectors :
                c.save()
        else :
            if self.save_to is not None:
                os.makedirs(self.save_to, exist_ok=True)
                f = f'{self.save_to}/{self.save_as}'
                dfs=[]
                for c in self.collectors:
                    df = pd.DataFrame(c.table)
                    df['AgentID'] = c.object.unique_id
                    df.index.set_names(['Step'], inplace=True)
                    df.reset_index(drop=False, inplace=True)
                    df.set_index(['Step', 'AgentID'], inplace=True)
                    dfs.append(df)
                ddf=pd.concat(dfs)
                ddf.sort_index(level=['Step', 'AgentID'], inplace=True)
                ddf.to_csv(f, index=True, header=True)



#     AnglePar(name='body_bend', key='b', theta_base='b', disp='bend', **cc, **c),
#     AnglePar(name='front_orientation', key='fo', theta_base=sub('or', 'f'),#disp='front orientation',
#              func=lambda o: o.get_head().get_orientation(), unwrapped=False, **cc, **c),
#     AnglePar(name=nam.unwrap('front_orientation'), key='fou', theta_base=sub('or', 'f'),#disp='front orientation unwrapped',
#              func=lambda o: o.get_head().get_orientation(), unwrapped=True, **cc, **c),
#     DiffAngPar(name='d_body_bend', par=d['b'], **c),
#     DiffAngPar(name='d_front_orientation', par=d['fou'], **c),
#     AnglePar(name='rear_orientation', key='ro', theta_base=sub('or', 'r'),#disp='rear orientation',
#              func=lambda o: o.get_tail().get_orientation(), unwrapped=False, **cc, **c),
#     AnglePar(name=nam.unwrap('rear_orientation'), key='rou', theta_base=sub('or', 'r'),#disp='rear orientation unwrapped',
#              func=lambda o: o.get_tail().get_orientation(), unwrapped=True, **cc, **c),
#     Or2DPar(p=(0,0), **cc, **c)
#     TemporalPar(name='dt', constant=dt, unit='sec', **c),
#     TemporalPar(name='dt2', constant=dt ** 2, unit='sec', power=2, **c),
#     SpatialPar(name='length', key='l', func=lambda o: o.get_real_length(), **c),
#     SpatialPar(name='dst', key='d', **c),
#     # SpatialPar(name='dst', key='d', disp='distance', **c),
#     SpatialPar(name='x', func=lambda o: o.pos[0] / o.model.scaling_factor, **c),
#     SpatialPar(name='y', func=lambda o: o.pos[1] / o.model.scaling_factor, **c),
#     for n,k,p in [[None, None, (0, 0)],[None, None, (0.8, 0.0)], ['dispersion', 'dsp', None]] :
#         Dst2DPar(name=n, key=k, p=p, **c),
#     for p_s in ['d', 'd_cen', 'dsp', 'd_(0.8, 0.0)'] :
#         ScaledSpatialPar(p_key=p_s, **c),
#     for k_p,n_pref,k_pref in[['d', '', ''], ['db', 'bend_', 'b'], ['dfou', 'front_orientation_', 'fo'], ['sd', 'scaled_', 's']] :
#         add_moments(k_p=k_p, n_pref=n_pref, k_pref=k_pref, **c)
#     for i in range(4) :
#         ConcentrationPar(i,diff=False,**c)
#         ConcentrationPar(i, diff=True, **c)


collection_dict={
    'stride' : ['x', 'y', 'b','fou','rou', 'sv', 'd', 'fov', 'bv'],
    'pose' : ['x', 'y', 'b','fo'],
    'angular' : ['b', 'bv', 'ba', 'fo', 'fov', 'foa'],
    'dispersion' : ['d_cen','sd_cen','dsp','sdsp', 'd_(0.8, 0.0)', 'sd_(0.8, 0.0)'],
    'odor' : ['C_od0', 'C_od1','C_od2','dC_od0', 'dC_od1','dC_od2'],
}


def load_ParDict() :
    dic=fun.load_dicts([paths.ParDict_path])[0]
    return dic


def add_par(dic, **kwargs) :
    p=dtypes.get_dict('par', **kwargs)
    k=p['k']
    if k in dic.keys() :
        raise ValueError (f'Key {k} already exists')
    dic[k]=Parameter(**p, par_dict=dic)
    return dic

def add_diff_par(dic, k0) :
    b=dic[k0]
    dic=add_par(dic,p=f'D_{b.p}' , k=f'D_{k0}', u=b.u, d=f'{b.d} change', s=Delta(b.s), exists=False, diff=True, k0=k0)
    return dic


def add_rate_par(dic, k0) :
    b = dic[k0]
    dic = add_par(dic, p=f'd_{k0}', k=f'd_{k0}', d=f'{b.d} change',s=dot(b.s), exists=False, fraction=True, k_num=f'D_{k0}',k_den='D_t')
    return dic

def add_Vspec_par(dic, k0) :
    b = dic[k0]
    dic = add_par(dic, p=f'[{k0}]', k=f'[{k0}]', d=f'volume specific {b.d}',s=f'[{b.s}]', exists=False, fraction=True, k_num=k0, k_den='V')
    return dic



def build_par_dict() :
    siu.day = siu.s * 24 * 60 * 60
    siu.cm = siu.m * 10 ** -2
    siu.g = siu.kg * 10 ** -3
    siu.deg=siu.rad/np.pi*180
    df = {}
    # df = pd.DataFrame(columns=list(dtypes.get_dict('par').keys()))
    df = add_par(df, p='L', k='L', u=1*siu.cm, o=DEB, d='structural length', s='L')
    df = add_par(df, p='Lw', k='Lw', u=1*siu.cm, o=DEB, d='physical length', s=sub('L', 'w'))
    df = add_par(df, p='V', k='V', u=1*siu.cm ** 3, o=DEB, d='structural volume', s='V')
    df = add_par(df, p='Ww', k='Ww', u=1*siu.g, o=DEB, d='wet weight', s=sub('W', 'w'))
    df = add_par(df, p='age', k='t', u=1*siu.day, o=DEB, d='age', s='t')
    df = add_par(df, p='hunger', k='H', o=DEB, d='hunger drive', s='H')
    df = add_par(df, p='E', k='E', u=1*siu.j, o=DEB, d='reserve energy', s='E')
    df = add_par(df, p='E_H', k='E_H', u=1*siu.j, o=DEB, d='maturity energy', s=sub('E', 'H'))
    df = add_par(df, p='E_R', k='E_R', u=1*siu.j, o=DEB, d='reproduction buffer', s=sub('E', 'R'))
    df = add_par(df, p='deb_p_A', k='deb_p_A', u=1*siu.j, o=DEB, d='assimilation energy (model)',s=subsup('p', 'A', 'deb'))
    df = add_par(df, p='sim_p_A', k='sim_p_A', u=1*siu.j, o=DEB, d='assimilation energy (sim)',s=subsup('p', 'A', 'sim'))
    df = add_par(df, p='gut_p_A', k='gut_p_A', u=1*siu.j, o=DEB, d='assimilation energy (gut)',s=subsup('p', 'A', 'gut'))
    df = add_par(df, p='e', k='e', o=DEB, d='scaled reserve density', s='e')
    df = add_par(df, p='f', k='f', o=DEB, d='scaled functional response', s='f')
    df = add_par(df, p='base_f', k='f0', o=DEB, d='base scaled functional response',s=sub('f', 0))
    df = add_par(df, p='F', k='[F]',u=siu.s**-1 / (24 * 60 * 60), o=DEB, d='volume specific filtering rate',s=dot('[F]'))
    df = add_par(df, p='fr_feed', k='fr_f',u=siu.s**-1, o=DEB, d='feed motion frequency (estimate)',s=sub(dot('fr'), 'feed'))
    df = add_par(df, p='pupation_buffer', k='pupation', o=DEB, d='pupation ratio',s=sub('r', 'pupation'))

    df = add_diff_par(df, k0='t')
    for k0 in ['f', 'e', 'H'] :
        df = add_diff_par(df, k0=k0)
        df = add_rate_par(df, k0=k0)

    for k0 in ['E', 'Ww', 'E_R', 'E_H'] :
        df = add_Vspec_par(df, k0=k0)

    fun.save_dict(df, paths.ParDict_path)
    return df




if __name__ == '__main__':
    # build_par_dict()
    # raise



    deb = DEB(id='test_DEB', steps_per_day=24*60)
    deb.grow_larva()
    print(deb.fr_feed)

    dic=load_ParDict()


    for i in range(5) :

        for k,v in dic.items() :
            if k in ['fr_f'] :
                print(k, v.get_from(deb))
            # print(k, v.get_from(deb))
        deb.run()
    #
    # print(b.values())
    # df=pd.DataFrame.from_dict(b)

    # k=[dtypes.get_dict('par', par='L', unit=u.cm, key='L', object_class=DEB, disp='structural length'),
    #     dtypes.get_dict('par', par='Lw', unit=u.cm, key='Lw', object_class=DEB, disp='physical length')]
    # df=df.append(k, ignore_index=True)
    # print(b['L'].get_from(deb))
    # df=pd.read_csv(paths.ParDict_path, index_col=0)
    # print(type(DEB.__class__))
    # print(type(df['unit'].values[0]))
    # print(type(df['object_class'].values[0]))
    # print(u.cm.__class__)
    # dic = {k: Parameter(**df.loc[k]) for k in df.index.values}
    # print(dic['L'].get_from(deb))
    raise

    print(df)


    # k1 = VolumePar(name='V', object_class=DEB, unit='cm')
    # # k=SpatialPar(name='L', object_class=DEB, unit='cm')
    # k2 = FractionPar(name='F_m', object_class=DEB, numerator=VolumeRatePar(n_unit='cm', d_unit='day'),
    #                   denominator=AreaPar(unit='cm'))
    # k = FractionPar(name='some', object_class=DEB, numerator=k1, denominator=k2, exists=False)
    # # k=EnergyRatePar(name='deb_p_A', object_class=DEB, denominator=TemporalPar(unit='day'))
    # # k=EnergyPar(name='E', object_class=DEB)
    # # k=TemporalPar(name='age', object_class=DEB, unit='days')
    # # k=TemporalPar(name='cum_dur')
    # deb = DEB(id='test_DEB')
    # deb.grow_larva()
    # print(k.get_from(deb), k1.get_from(deb) / k2.get_from(deb))
    # # print(k.get_from(deb,(('cm','hour'), 'cm')), getattr(deb, k.name)/24)
    # print(k.get_unit(), k.type)
    #
    import matplotlib.pyplot as plt
    #
    plt.plot(np.arange(10), np.arange(10))
    plt.xlabel(df['unit'].iloc[1]/1973*u.day)
    # plt.xlabel([d[k].symbol for k in list(d.keys())])
    plt.show()
