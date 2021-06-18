import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

from lib.aux import functions as fun
from lib.aux import naming as nam
from lib.stor import paths
from lib.conf.par_conf import sup, sub, th, dot, ddot
from lib.model.DEB.deb import DEB
from lib.model.agents._agent import LarvaworldAgent

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
    def __init__(self, unit, name, type, key=None, symbol=None, power=None, object_class=None, lim=None,
                 disp=None, exists=True, func=None, return_unit=None, constant=None, par_dict=None, fraction=False):
        self.fraction = fraction
        self.func = func
        self.exists = exists
        self.name = name
        if key is None:
            key = name
        self.key = key
        if symbol is None:
            symbol = self.key
        self.symbol = symbol
        self.type = type
        self.unit = unit
        if return_unit is None:
            return_unit = self.unit
        self.return_unit = return_unit
        self.power = power
        if object_class is None:
            object_class = LarvaworldAgent
        self.object_class = object_class
        self.lim = lim
        if disp is None:
            disp = name
        self.disp = disp
        self.constant = constant

        par_dict[self.key] = self
        # return {self.key : self}

    def convert_to(self, u):
        raise ValueError('Implemented by subclasses')



    def get_from(self, object, unit=None):
        if unit is None:
            unit = self.return_unit
        if not self.fraction :
            if self.constant is not None:
                v = self.constant
            elif self.func is not None:
                v = self.func(object)
            elif self.exists:
                v = getattr(object, self.name)
            vv = v * self.convert_to(unit)
            vv = self.post_process(vv, unit)
        else:
            u_n, u_v = unit
            v_n = self.numerator.get_from(object, u_n)
            v_d = self.denominator.get_from(object, u_v)
            vv = v_n / v_d
        return vv

    def get_unit(self):
        if self.power == 1 or self.power is None:
            return self.unit
        else:
            return sup(self.unit, self.power)
            # return f'{self.unit}^{self.power}'

    def post_process(self, v, u):
        return v


class SpatialPar(Parameter):
    def __init__(self, unit=None, name='length', type='length', power=1, **kwargs):
        if unit is None:
            unit = 'm'
        super().__init__(unit=unit, name=name, type=type, power=power, **kwargs)

    def convert_to(self, u=None):
        u0 = self.unit
        if u is None:
            u = self.return_unit
        if u == u0:
            c = 1
        elif u0 == 'mm':
            if u == 'cm':
                c = 10 ** -1
            if u == 'm':
                c = 10 ** -3
        elif u0 == 'm':
            if u == 'cm':
                c = 10 ** 2
            if u == 'mm':
                c = 10 ** 3
        elif u0 == 'cm':
            if u == 'm':
                c = 10 ** -2
            if u == 'mm':
                c = 10 ** 1
        return c ** self.power

class Dst2DPar(SpatialPar):
    def __init__(self, p, name=None,key=None,**kwargs):
        if p is None :
            func = lambda o: euclidean(o.pos, o.initial_pos) / o.model.scaling_factor
        else :
            func=lambda o: euclidean(o.pos,p) / o.model.scaling_factor
        if name is None :
            if p==(0,0) :
                name = 'distance to center'
            else :
                name=f'distance to {p}'
        if key is None :
            if p==(0,0) :
                key='d_cen'
            else :
                key=f'd_{p}'
            # key=f'd_{p}'
        # print(key)
        super().__init__(name=name,key=key, exists=False, func=func, **kwargs)




class VolumePar(SpatialPar):
    def __init__(self, name='volume', **kwargs):
        super().__init__(name=name, type='volume', power=3, **kwargs)


class AreaPar(SpatialPar):
    def __init__(self, name='area', **kwargs):
        super().__init__(name=name, type='area', power=2, **kwargs)


class TemporalPar(Parameter):
    def __init__(self, unit=None, name='time', power=1, **kwargs):
        if unit is None:
            unit = 'sec'
        super().__init__(unit=unit, name=name, type='time', power=power, **kwargs)

    def convert_to(self, u=None):
        u0 = self.unit
        if u is None:
            u = self.return_unit
        if u == u0:
            c = 1
        if u0 == 'sec':
            if u == 'min':
                c = 1 / 60
            elif u == 'hour':
                c = 1 / (60 * 60)
            elif u == 'day':
                c = 1 / (60 * 60 * 24)

        elif u0 == 'min':
            if u == 'hour':
                c = 1 / 60
            elif u == 'day':
                c = 1 / (60 * 24)
            elif u == 'sec':
                c = 60

        elif u0 == 'hour':
            if u == 'day':
                c = 1 / 24
            elif u == 'min':
                c = 60
            elif u == 'sec':
                c = 60 * 60
        elif u0 == 'day':
            if u == 'hour':
                c = 24
            elif u == 'min':
                c = 24 * 60
            elif u == 'sec':
                c = 24 * 60 * 60
        return c ** self.power


class AnglePar(Parameter):
    def __init__(self, unit='deg', name='angle', symbol=None, power=1, theta_base=None,
                 unwrapped=True, deg_lim=(0, 360), rad_lim=(0, 2 * np.pi), **kwargs):
        if symbol is None and theta_base is not None:
            symbol = th(theta_base)
        super().__init__(unit=unit, name=name, type='angle', power=power, symbol=symbol, **kwargs)
        self.unwrapped = unwrapped
        self.deg_lim = deg_lim
        self.rad_lim = rad_lim
        self.wrap_dic = {'rad': (2 * np.pi, self.rad_lim), 'deg': (360, self.deg_lim)}

    def convert_to(self, u=None):
        u0 = self.unit
        if u is None:
            u = self.return_unit
        if u == u0:
            c = 1
        if u0 == 'deg' and u == 'rad':
            c = np.pi / 180
        elif u == 'deg' and u0 == 'rad':
            c = 180 / np.pi
        return c

    def post_process(self, v, u):
        if not self.unwrapped:
            c, (l0, l1) = self.wrap_dic[u]
            v %= c
            if v > l1:
                v -= c
        return v

class Or2DPar(AnglePar):
    def __init__(self, p, name=None,key=None,**kwargs):
        func = lambda o: fun.angle_dif(o.get_head().get_normalized_orientation(),fun.angle_to_x_axis(o.pos, p))
        if name is None:
            if p == (0, 0):
                name = 'orientation to center'
            else:
                name = f'orientation to {p}'
        if key is None:
            if p == (0, 0):
                key = 'o_cen'
            else:
                key = f'o_{p}'
        super().__init__(name=name,key=key, exists=False, func=func, **kwargs)


class EnergyPar(Parameter):
    def __init__(self, unit=None, name='energy', power=1, **kwargs):
        if unit is None:
            unit = 'J'
        super().__init__(unit=unit, name=name, type='energy', power=power, **kwargs)

    def convert_to(self, u=None):
        u0 = self.unit
        if u is None:
            u = self.return_unit
        if u == u0:
            c = 1
        if u0 == 'J' and u == 'mJ':
            c = 10 ** 3
        elif u == 'J' and u0 == 'mJ':
            c = 10 ** -3
        return c ** self.power

class ConcentrationPar(Parameter):
    def __init__(self, odor_idx, diff=False, unit=None, name=None, **kwargs):
        if unit is None:
            unit = '$\mu$M'
        if not diff :
            func=lambda o: list(o.brain.olfactor.Con.values())[odor_idx]
            key = f'C_od{odor_idx}'
            if name is None:
                name = f'Odor {odor_idx} concentration'
        else :
            func=lambda o: list(o.brain.olfactor.dCon.values())[odor_idx]
            key = f'dC_od{odor_idx}'
            if name is None :
                name = f'delta Odor {odor_idx} concentration'
        super().__init__(unit=unit, name=name, key=key, type='concentration', exists=False, func=func, **kwargs)

    def convert_to(self, u=None):
        u0 = self.unit
        if u is None:
            u = self.return_unit
        if u == u0:
            c = 1
        if u0 == '$\mu$M' and u == 'mM':
            c = 10 ** -3
        elif u == '$\mu$M' and u0 == 'mM':
            c = 10 ** 3
        return c

class FractionPar(Parameter):
    def __init__(self, numerator, denominator, type=None, unit=None, name=None, return_unit=None, **kwargs):
        if name is None:
            name = f'{numerator.name}/{denominator.name}'
        if type is None:
            type = f'{numerator.type}/{denominator.type}'
        if unit is None:
            unit = (numerator.unit, denominator.unit)
        if return_unit is None:
            return_unit = (numerator.return_unit, denominator.return_unit)
        super().__init__(unit=unit, return_unit=return_unit, type=type, name=name,fraction=True, **kwargs)
        self.numerator = numerator
        self.denominator = denominator

    def convert_to(self, u=None):
        u0 = self.unit
        if u is None:
            u = self.return_unit
        if u == u0:
            c = 1
        else:
            u_n, u_d = u
            if u_n == self.numerator.unit and u_d == self.denominator.unit:
                c = 1
            else:
                c_n = self.numerator.convert_to(u_n)
                c_d = self.denominator.convert_to(u_d)
                c = c_n / c_d
        return c

    def get_unit(self):
        u_n = self.numerator.get_unit()
        u_d = self.denominator.get_unit()
        try:
            u_d0, u_dp = u_d.split('^')
            u_dp = u_dp.replace("{", "")
            u_dp = u_dp.replace("}", "")
            u_dp = u_dp.replace("$", "")
            return f'{u_n}{sup(u_d0, -1 * int(u_dp))}'
        except:
            return f'{u_n}{sup(u_d, -1)}'


class ScaledSpatialPar(FractionPar) :
    def __init__(self, p_key, par_dict, name=None, **kwargs):
        p=par_dict[p_key]
        if name is None :
            name=nam.scal(p.name)
        super().__init__(name=name, key=f's{p.key}', numerator=p,
                         denominator=par_dict['l'], exists=False, par_dict=par_dict, **kwargs),

class RatePar(FractionPar):
    def __init__(self, numerator, denominator=None, type=None, d_unit=None, **kwargs):
        if denominator is None:
            denominator = TemporalPar(unit=d_unit)
        if type is None:
            type = f'{numerator.type} rate'
        super().__init__(numerator=numerator, denominator=denominator, type=type, **kwargs)


class DiffAngPar(AnglePar):
    def __init__(self, par, name=None, disp=None, **kwargs):
        if name is None:
            name = f'd_{par.name}'
        if disp is None:
            disp = f'delta {par.disp}'
        super().__init__(name=name, disp=disp, key=f'd{par.key}', power=par.power,
                         unit=par.unit, return_unit=par.return_unit, **kwargs)

class EnergyRatePar(RatePar):
    def __init__(self, numerator=None, type=None, n_unit=None, **kwargs):
        if numerator is None:
            numerator = EnergyPar(unit=n_unit)
        if type is None:
            type = 'energy rate'
        super().__init__(numerator=numerator, type=type, **kwargs)


class VolumeRatePar(RatePar):
    def __init__(self, numerator=None, type=None, n_unit=None, **kwargs):
        if numerator is None:
            numerator = VolumePar(unit=n_unit)
        if type is None:
            type = 'volume rate'
        super().__init__(numerator=numerator, type=type, **kwargs)


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


def add_moments(k_p, par_dict,n_pref='', k_pref='', n_vel='vel', n_acc='acc', k_vel='v', k_acc='a', r_vel=None, r_acc=None, **kwargs):
    n_vel=f'{n_pref}{n_vel}'
    n_acc=f'{n_pref}{n_acc}'
    k_vel=f'{k_pref}{k_vel}'
    k_acc=f'{k_pref}{k_acc}'
    p=par_dict[k_p]
    v = RatePar(name=n_vel, numerator=p, denominator=par_dict['dt'], key=k_vel,symbol=dot(p.symbol),
                exists=False, return_unit=r_vel, par_dict=par_dict, **kwargs)
    a = RatePar(name=n_acc, numerator=p, denominator=par_dict['dt2'], key=k_acc,symbol=ddot(p.symbol),
                exists=False, return_unit=r_acc, par_dict=par_dict, **kwargs)
    return [v, a]

def build_par_dict(dt=0.1) :
    d={}
    c = {
        'par_dict': d
    }
    cc = {
        'unit': 'rad',
        'return_unit': 'deg',
    }

    AnglePar(name='body_bend', key='b', theta_base='b', disp='bend', **cc, **c),
    AnglePar(name='front_orientation', key='fo', theta_base=sub('or', 'f'),#disp='front orientation',
             func=lambda o: o.get_head().get_orientation(), unwrapped=False, **cc, **c),
    AnglePar(name=nam.unwrap('front_orientation'), key='fou', theta_base=sub('or', 'f'),#disp='front orientation unwrapped',
             func=lambda o: o.get_head().get_orientation(), unwrapped=True, **cc, **c),
    DiffAngPar(name='d_body_bend', par=d['b'], **c),
    DiffAngPar(name='d_front_orientation', par=d['fou'], **c),
    AnglePar(name='rear_orientation', key='ro', theta_base=sub('or', 'r'),#disp='rear orientation',
             func=lambda o: o.get_tail().get_orientation(), unwrapped=False, **cc, **c),
    AnglePar(name=nam.unwrap('rear_orientation'), key='rou', theta_base=sub('or', 'r'),#disp='rear orientation unwrapped',
             func=lambda o: o.get_tail().get_orientation(), unwrapped=True, **cc, **c),
    Or2DPar(p=(0,0), **cc, **c)
    TemporalPar(name='dt', constant=dt, unit='sec', **c),
    TemporalPar(name='dt2', constant=dt ** 2, unit='sec', power=2, **c),
    SpatialPar(name='length', key='l', func=lambda o: o.get_real_length(), **c),
    SpatialPar(name='dst', key='d', **c),
    # SpatialPar(name='dst', key='d', disp='distance', **c),
    SpatialPar(name='x', func=lambda o: o.pos[0] / o.model.scaling_factor, **c),
    SpatialPar(name='y', func=lambda o: o.pos[1] / o.model.scaling_factor, **c),
    for n,k,p in [[None, None, (0, 0)],[None, None, (0.8, 0.0)], ['dispersion', 'dsp', None]] :
        Dst2DPar(name=n, key=k, p=p, **c),
    for p_s in ['d', 'd_cen', 'dsp', 'd_(0.8, 0.0)'] :
        ScaledSpatialPar(p_key=p_s, **c),
    for k_p,n_pref,k_pref in[['d', '', ''], ['db', 'bend_', 'b'], ['dfou', 'front_orientation_', 'fo'], ['sd', 'scaled_', 's']] :
        add_moments(k_p=k_p, n_pref=n_pref, k_pref=k_pref, **c)
    for i in range(4) :
        ConcentrationPar(i,diff=False,**c)
        ConcentrationPar(i, diff=True, **c)
    # fun.save_dict(d, paths.ParDict_path)
    return d

collection_dict={
    'stride' : ['x', 'y', 'b','fou','rou', 'sv', 'd', 'fov', 'bv'],
    'pose' : ['x', 'y', 'b','fo'],
    'angular' : ['b', 'bv', 'ba', 'fo', 'fov', 'foa'],
    'dispersion' : ['d_cen','sd_cen','dsp','sdsp', 'd_(0.8, 0.0)', 'sd_(0.8, 0.0)'],
    'odor' : ['C_od0', 'C_od1','C_od2','dC_od0', 'dC_od1','dC_od2'],
}

if __name__ == '__main__':
    d=build_par_dict(0.1)
    print(list(d.keys()))
    print(d['fov'].symbol)
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
    # import matplotlib.pyplot as plt
    #
    # plt.plot(np.arange(10), np.arange(10))
    # plt.xlabel(k.get_unit())
    # plt.show()
