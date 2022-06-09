import random

import numpy as np
import pandas as pd
import param

from lib.conf.base import paths

from lib.conf.pars.units import ureg




from lib.aux import naming as nam, dictsNlists as dNl

def v_descriptor(vparfunc,v0=None,dv=None, **kws):
    class LarvaworldParNew(param.Parameterized):
        p = param.String(default='', doc='Name of the parameter')
        d = param.String(default='', doc='Dataset name of the parameter')
        disp = param.String(default='', doc='Displayed name of the parameter')
        k = param.String(default='', doc='Key of the parameter')
        sym = param.String(default='', doc='Symbol of the parameter')
        codename = param.String(default='', doc='Name of the parameter in code')
        dtype = param.Parameter(default=float, doc='Data type of the parameter value')
        v = vparfunc
        func = param.Callable(default=None, doc='Function to get the parameter from a dataset', allow_None=True)
        required_ks = param.List(default=[], doc='Keys of prerequired parameters for computation in a dataset')
        u = param.Parameter(default=ureg.dimensionless, doc='Unit of the parameter values')

        @property
        def s(self):
            return self.disp

        @property
        def l(self):
            return self.param.v.label

        @property
        def unit(self):
            return self.param.u

        @property
        def short(self):
            return self.k

        @property
        def v0(self):
            return self.param.v.default

        @property
        def initial_value(self):
            return self.param.v.default

        @property
        def symbol(self):
            return self.sym

        @property
        def label(self):
            return self.param.v.label

        @property
        def lab(self):
            return self.param.v.label

        @property
        def tooltip(self):
            return self.param.v.doc

        @property
        def help(self):
            return self.param.v.doc

        @property
        def parclass(self):
            return type(self.param.v)

        @property
        def min(self):
            try:
                vmin, vmax = self.param.v.bounds
                return vmin
            except:
                return None

        @property
        def max(self):
            try:
                vmin, vmax = self.param.v.bounds
                return vmax
            except:
                return None

        @property
        def lim(self):
            try:
                lim = self.param.v.bounds
                return lim
            except:
                return None

        @property
        def step(self):
            try:
                step = self.param.v.step
                return step
            except:
                return None

        @property
        def get_ParsArg(self):
            from lib.anal.argparsers import build_ParsArg
            return build_ParsArg(name=self.name, k=self.k, h=self.help, t=self.dtype, v=self.initial_value, vs=None)

        def exists(self, dataset):
            par = self.d
            s, e, c = dataset.step_data, dataset.endpoint_data, dataset.config
            dic = {'step': par in s.columns, 'end': par in e.columns}
            if 'aux_pars' in c.keys():
                for k, ps in c.aux_pars.items():
                    dic[k] = par in ps
            return dic

        def get(self, dataset, compute=True):
            res = self.exists(dataset)
            for key, exists in res.items():
                if exists:
                    return dataset.get_par(key=key, par=self.d)

            if compute:
                self.compute(dataset)
                return self.get(dataset, compute=False)
            else:
                print(f'Parameter {self.disp} not found')

        def compute(self, dataset):
            if self.func is not None:
                self.func(dataset)
            else:
                print(f'Function to compute parameter {self.disp} is not defined')

        def randomize(self):
            if self.parclass == param.Number:
                vmin, vmax = self.param.v.bounds
                self.v = random.uniform(vmin, vmax)
            elif self.parclass == param.Integer:
                vmin, vmax = self.param.v.bounds
                self.v = random.randint(vmin, vmax)
            elif self.parclass == param.Magnitude:
                self.v = random.uniform(0.0, 1.0)

            elif self.parclass == param.Selector:
                self.v = random.choice(self.param.v.objects)
            elif self.parclass == param.Boolean:
                self.v = random.choice([True, False])
            elif self.parclass == param.Range:
                vmin, vmax = self.param.v.bounds
                vv0 = random.uniform(vmin, vmax)
                vv1 = random.uniform(vv0, vmax)
                self.v = (vv0, vv1)

        def mutate(self, Pmut, Cmut):
            if random.random() < Pmut:
                if self.parclass == param.Number:

                    vmin, vmax = self.param.v.bounds
                    vr = np.abs(vmax - vmin)
                    v0 = self.v if self.v is not None else vmin + vr / 2
                    vv = random.gauss(v0, Cmut * vr)
                    self.v = self.param.v.crop_to_bounds(vv)
                elif self.parclass == param.Integer:
                    vmin, vmax = self.param.v.bounds
                    vr = np.abs(vmax - vmin)
                    v0 = self.v if self.v is not None else int(vmin + vr / 2)
                    vv = random.gauss(v0, Cmut * vr)
                    self.v = self.param.v.crop_to_bounds(int(vv))
                elif self.parclass == param.Magnitude:
                    v0 = self.v if self.v is not None else 0.5
                    vv = random.gauss(v0, Cmut)
                    self.v = self.param.v.crop_to_bounds(vv)
                elif self.parclass == param.Selector:
                    self.v = random.choice(self.param.v.objects)
                elif self.parclass == param.Boolean:
                    self.v = random.choice([True, False])
                elif self.parclass == param.Range:
                    vmin, vmax = self.param.v.bounds
                    vr = np.abs(vmax - vmin)
                    v0, v1 = self.v if self.v is not None else (vmin, vmax)
                    vv0 = random.gauss(v0, Cmut * vr)
                    vv1 = random.gauss(v1, Cmut * vr)
                    vv0 = np.clip(vv0, a_min=vmin, a_max=vmax)
                    vv1 = np.clip(vv1, a_min=vv0, a_max=vmax)
                    self.v = (vv0, vv1)

    par = LarvaworldParNew(**kws)
    return par



class ParRegistry:
    def __init__(self, mode='build', object=None, save=True, load_funcs=True):
        from lib.conf.pars.par_dict import BaseParDict
        from lib.conf.pars.par_funcs import build_func_dict
        if load_funcs :
            self.func_dict = dNl.load_dict(paths.path('ParFuncDict'))
        else :
            self.func_dict = build_func_dict()

        if mode=='load' :

            self.dict =self.load()
        elif mode=='build' :
            self.dict_entries = BaseParDict(func_dict=self.func_dict).dict_entries

            self.dict = self.finalize_dict(self.dict_entries)

            self.ddict=dNl.AttrDict.from_nested_dicts({p.d:p for k,p in self.dict.items()})
            self.pdict=dNl.AttrDict.from_nested_dicts({p.p:p for k,p in self.dict.items()})
            if save :
                self.save()

    def finalize_dict(self, entries):
        dic = dNl.AttrDict.from_nested_dicts({})
        for prepar in entries:
            p = v_descriptor(**prepar)
            dic[p.k] = p
        return dic

    def save(self):
        dNl.save_dict(self.func_dict, paths.path('ParFuncDict'))
        df = pd.DataFrame.from_records(self.dict_entries, index='k')
        df.to_csv(paths.path('ParDf'))

    def load(self):
        # FIXME Not working
        df = pd.read_csv(paths.path('ParDf'),index_col=0)
        # df = df.where(pd.notnull(df), None)
        entries=df.to_dict(orient='records')
        # print(entries[0]['func'])
        dict = self.finalize_dict(entries)
        return dict


    def get(self, k, d, compute=True):
        p = self.dict[k]
        res = p.exists(d)

        if res['step']:
            if hasattr(d, 'step_data') :
                return d.step_data[p.d]
            else :
                return d.read(key='step')[p.d]
        elif res['end']:
            if hasattr(d, 'endpoint_data') :
                return d.endpoint_data[p.d]
            else :
                return d.read(key='end', file='endpoint_h5')[p.d]
        else :
            for key in res.keys() :
                if key not in ['step', 'end'] and res[key] :
                    return d.read(key=f'{key}.{p.d}', file='aux_h5')


        if compute:
            self.compute(k, d)
            return self.get(k, d, compute=False)
        else:
            print(f'Parameter {p.disp} not found')

    def compute(self, k, d):
        p = self.dict[k]
        res = p.exists(d)
        if not any(list(res.values())):
            k0s = p.required_ks
            for k0 in k0s:
                self.compute(k0,d)
            p.compute(d)



ParDict=ParRegistry()


def getPar(k=None, p=None, d=None, to_return='d', PF=ParDict):
    if k is not None:
        d0=PF.dict
        k0=k
    elif d is not None:
        d0 = PF.ddict
        k0 = d
    elif p is not None:
        d0 = PF.pdict
        k0 = p

    if type(k0) == str:
        par = d0[k0]
        if type(to_return) == list:
            return [getattr(par, i) for i in to_return]
        elif type(to_return) == str:
            return getattr(par, to_return)
    elif type(k0) == list:
        pars = [d0[i] for i in k0]
        if type(to_return) == list:
            return [[getattr(par, i) for par in pars] for i in to_return]
        elif type(to_return) == str:
            return [getattr(par, to_return) for par in pars]

def runtime_pars(PF=ParDict.dict):
    return [v.d for k, v in PF.items()]

if __name__ == '__main__':
    # for d,p in ParDict.dict.items() :
    #     print(d,p.v,type(p.func),type(p.dtype))
    # # p.param.trigger('disp', 'd')
    d=ParDict.dict['ba'].param.d
    # # print(d.name)
    # # print(ParDict.dict['b'].v)
    # # print(p.param.values())
    # # p._internal_name = 'ddd'
    # # ParDict.dict['b'].param.add_parameter('FF', param.String())
    # print(p.param.schema())
    # df = pd.read_csv(paths.path('ParDf'),index_col=0)
