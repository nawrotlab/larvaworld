import random

import numpy as np
import param

from lib.aux import dictsNlists as dNl
from lib.registry.units import ureg


def v_descriptor(vparfunc, v0=None, dv=None, **kws):
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
            return self.u

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
        def value(self):
            return self.v

        @property
        def symbol(self):
            return self.sym

        @property
        def label(self):
            return self.param.v.label

        @property
        def parameter(self):
            return self.disp

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
            if self.parclass in [param.Number, param.Range] :
                return self.param.v.step
            elif self.parclass==param.Magnitude:
                return 0.01
            elif self.dtype==float:
                return 0.01
            else :
                return None

        @property
        def Ndec(self):
            if self.step is not None :
                return str(self.step)[::-1].find('.')
            else :
                return None



        @property
        def get_ParsArg(self):
            from lib.registry.parser_dict import build_ParsArg
            return build_ParsArg(name=self.name, k=self.k, h=self.help, t=self.dtype, v=self.initial_value, vs=None)

        def exists(self, dataset):
            par = self.d
            d=dataset
            dic = dNl.NestDict({'step': False, 'end': False})
            if hasattr(d,'step_data'):
                s=d.step_data
                if par in s.columns :
                    dic.step=True
            if hasattr(d, 'endpoint_data'):
                e = d.endpoint_data
                if par in e.columns:
                    dic.end = True



            c=d.config
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
                self.v = np.round(random.uniform(vmin, vmax), self.Ndec)
            elif self.parclass == param.Integer:
                vmin, vmax = self.param.v.bounds
                self.v = random.randint(vmin, vmax)
            elif self.parclass == param.Magnitude:
                self.v = np.round(random.uniform(0.0, 1.0), self.Ndec)
            elif self.parclass == param.Selector:
                self.v = random.choice(self.param.v.objects)
            elif self.parclass == param.Boolean:
                self.v = random.choice([True, False])
            elif self.parclass == param.Range:
                vmin, vmax = self.param.v.bounds
                vv0 = np.round(random.uniform(vmin, vmax), self.Ndec)
                vv1 = np.round(random.uniform(vv0, vmax), self.Ndec)

                self.v = (vv0, vv1)

        def mutate(self, Pmut, Cmut):
            if random.random() < Pmut:
                if self.parclass == param.Number:

                    vmin, vmax = self.param.v.bounds
                    vr = np.abs(vmax - vmin)
                    v0 = self.v if self.v is not None else vmin + vr / 2
                    vv = random.gauss(v0, Cmut * vr)
                    self.v = np.round(self.param.v.crop_to_bounds(vv), self.Ndec)
                elif self.parclass == param.Integer:
                    vmin, vmax = self.param.v.bounds
                    vr = np.abs(vmax - vmin)
                    v0 = self.v if self.v is not None else int(vmin + vr / 2)
                    vv = random.gauss(v0, Cmut * vr)
                    self.v = self.param.v.crop_to_bounds(int(vv))
                elif self.parclass == param.Magnitude:
                    v0 = self.v if self.v is not None else 0.5
                    vv = random.gauss(v0, Cmut)
                    self.v = np.round(self.param.v.crop_to_bounds(vv), self.Ndec)
                    # self.v = np.round(self.v, self.Ndec)
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
                    vv0 = np.round(np.clip(vv0, a_min=vmin, a_max=vmax), self.Ndec)
                    vv1 = np.round(np.clip(vv1, a_min=vv0, a_max=vmax), self.Ndec)
                    self.v = (vv0, vv1)

    par = LarvaworldParNew(**kws)
    return par
