import pandas as pd
import random
import typing
from types import FunctionType
from typing import Tuple, List
import numpy as np
import param

from lib.aux import dictsNlists as dNl


from lib.registry.units import ureg


def maxNdigits(array, Min=None):
    N = len(max(array.astype(str), key=len))
    if Min is not None:
        N = max([N, Min])
    return N


def boolean_indexing(v, fillval=np.nan):
    lens = np.array([len(item) for item in v])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(v)
    return out


def concat_datasets(ddic, key='end', unit='sec'):
    dfs = []
    for l, d in ddic.items():
        if key == 'end':
            try:
                df = d.endpoint_data
            except:
                df = d.read(key='end')
        elif key == 'step':
            try:
                df = d.step_data
            except:
                df = d.read(key='step')
        df['DatasetID'] = l
        df['GroupID'] = d.group_id
        dfs.append(df)
    df0 = pd.concat(dfs)
    if key == 'step':
        df0.reset_index(level='Step', drop=False, inplace=True)
        dts = np.unique([d.config['dt'] for l, d in ddic.items()])
        if len(dts) == 1:
            dt = dts[0]
            dic = {'sec': 1, 'min': 60, 'hour': 60 * 60, 'day': 24 * 60 * 60}
            df0['Step'] *= dt / dic[unit]
    return df0


def moving_average(a, n=3):
    # ret = np.cumsum(a, dtype=float)
    # ret[n:] = ret[n:] - ret[:-n]
    return np.convolve(a, np.ones((n,)) / n, mode='same')
    # return ret[n - 1:] / n


def arrange_index_labels(index):
    from lib.aux import dictsNlists as dNl
    ks=index.unique().tolist()
    Nks = index.value_counts(sort=False)

    def merge(k, Nk):
        Nk1 = int((Nk - 1) / 2)
        Nk2 = Nk - 1 - Nk1
        return [''] * Nk1 + [k.upper()] + [''] * Nk2

    new = dNl.flatten_list([merge(k, Nks[k]) for k in ks])
    return new


def mdict2df(mdict, columns=['symbol', 'value', 'description']):
    data = []
    for k, p in mdict.items():
        entry = [getattr(p, col) for col in columns]
        data.append(entry)
    df = pd.DataFrame(data, columns=columns)
    df.set_index(columns[0], inplace=True)
    return df


def init2mdict(d0):
    from lib.aux import dictsNlists as dNl
    # d = {}

    def check(D0):
        D = {}
        for kk, vv in D0.items():
            if not isinstance(vv, dict):
                pass
            elif 'dtype' in vv.keys() and vv['dtype'] == dict:
                mdict = check(vv)
                vv0 = {kkk: vvv for kkk, vvv in vv.items() if kkk not in mdict.keys()}
                if 'v0' not in vv0.keys():
                    vv0['v0'] = gConf(mdict)
                prepar = preparePar(p=kk, mdict=mdict, **vv0)
                p = v_descriptor(**prepar)
                D[kk] = p

            elif any([a in vv.keys() for a in ['symbol', 'h', 'label', 'disp', 'k']]):
                prepar = preparePar(p=kk, **vv)
                p = v_descriptor(**prepar)
                D[kk] = p

            else:

                # label = f'{kk} conf'
                # if 'v0' in vv.keys() :
                #     v0=vv['v0']
                # else :
                #     v0=None
                # if 'k' in vv.keys() :
                #     k=vv['k']
                # else :
                #     k=kk
                # vparfunc = vdicpar(mdict, h=f'The {kk} conf', lab=label, v0=v0)
                # kws = {
                #     'name': kk,
                #     'p': kk,
                #     'd': kk,
                #     'k': k,
                #     'disp': label,
                #     'sym': sub(k, 'conf'),
                #     'codename': kk,
                #     'dtype': dict,
                #     # 'func': func,
                #     # 'u': ureg.dimensionless,
                #     # 'u_name': None,
                #     # 'required_ks': [],
                #     'vparfunc': vparfunc,
                #     # 'dv': None,
                #     'v0': v0,
                #
                # }
                # p = v_descriptor(**kws)
                D[kk] = check(vv)
        return D

    d = check(d0)
    return dNl.NestDict(d)


def gConf(mdict, **kwargs):
    from lib.aux import dictsNlists as dNl
    if mdict is None:
        return None


    elif isinstance(mdict, param.Parameterized):
        return mdict.v
    elif isinstance(mdict, dict):

        conf = {}
        for d, p in mdict.items():
            if isinstance(p, param.Parameterized):
                conf[d] = p.v
            else:
                conf[d] = gConf(mdict=p)
            conf = dNl.update_existingdict(conf, kwargs)
        # conf.update(kwargs)
        return dNl.NestDict(conf)
    else:
        return dNl.NestDict(mdict)


def update_mdict(mdict, mmdic):
    if mmdic is None or mdict is None:
        return None
    elif not isinstance(mmdic, dict) or not isinstance(mdict, dict):
        return mdict
    else:
        for d, p in mdict.items():
            new_v = mmdic[d] if d in mmdic.keys() else None
            if isinstance(p, param.Parameterized):
                if type(new_v) == list:
                    if p.parclass in [param.Range, param.NumericTuple, param.Tuple]:
                        new_v = tuple(new_v)
                p.v = new_v
            else:
                mdict[d] = update_mdict(mdict=p, mmdic=new_v)
        return mdict


def update_existing_mdict(mdict, mmdic):
    if mmdic is None:
        return mdict
    else:
        for d, v in mmdic.items():
            p = mdict[d]

            # new_v = mmdic[d] if d in mmdic.keys() else None
            if isinstance(p, param.Parameterized):
                if type(v) == list:
                    if p.parclass in [param.Range, param.NumericTuple, param.Tuple]:
                        v = tuple(v)

                p.v = v
            elif isinstance(p, dict) and isinstance(v, dict):
                mdict[d] = update_existing_mdict(mdict=p, mmdic=v)
        return mdict


def get_ks(d0, k0=None, ks=[]):
    for k, p in d0.items():
        if k0 is not None:
            k = f'{k0}.{k}'
        if isinstance(p, param.Parameterized):

            ks.append(k)
        else:
            ks = get_ks(p, k0=k, ks=ks)
    return ks

def v_descriptor(vparfunc, v0=None, dv=None, u_name=None, **kws):
    class LarvaworldParNew(param.Parameterized):
        p = param.String(default='', doc='Name of the parameter')
        d = param.String(default='', doc='Dataset name of the parameter')
        disp = param.String(default='', doc='Displayed name of the parameter')
        k = param.String(default='', doc='Key of the parameter')
        sym = param.String(default='', doc='Symbol of the parameter')
        codename = param.String(default='', doc='Name of the parameter in code')
        dtype = param.Parameter(default=float, doc='Data type of the parameter value')
        v = vparfunc
        mdict=param.Dict(default=None, doc='The parameter dict in case of a dict header', allow_None=True)
        func = param.Callable(default=None, doc='Function to get the parameter from a dataset', allow_None=True)
        required_ks = param.List(default=[], doc='Keys of prerequired parameters for computation in a dataset')
        u = param.Parameter(default=ureg.dimensionless, doc='Unit of the parameter values', label=u_name)

        @property
        def s(self):
            return self.disp

        @property
        def l(self):
            return self.disp + '  ' + self.ulabel

        @property
        def symunit(self):
            return self.sym + '  ' + self.ulabel

        @property
        def ulabel(self):
            return '(' + self.unit + ')'

        @property
        def unit(self):
            if self.u == ureg.dimensionless:
                return '-'
            else:
                return fr'${self.u}$'

        @property
        def short(self):
            return self.k

        @ property
        def gConf(self):
            if self.mdict is None:
                return None
            else :
                from lib.aux.data_aux import gConf
                return gConf(self.mdict)

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
        def description(self):
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
            if self.parclass in [param.Number, param.Range] and self.param.v.step is not None:
                return self.param.v.step
            elif self.parclass == param.Magnitude:
                return 0.01
            elif self.dtype in [float, List[float], List[Tuple[float]], Tuple[float]]:
                return 0.01
            else:
                return None

        @property
        def Ndec(self):
            if self.step is not None:
                return str(self.step)[::-1].find('.')
            else:
                return None

        @property
        def get_ParsArg(self):
            from lib.registry.parsers import build_ParsArg
            return build_ParsArg(name=self.name, k=self.k, h=self.help, dtype=self.dtype, v=self.initial_value, vs=None)

        def exists(self, dataset):
            par = self.d
            d = dataset
            dic = dNl.NestDict({'step': False, 'end': False})
            if hasattr(d, 'step_data'):
                s = d.step_data
                if par in s.columns:
                    dic.step = True
            if hasattr(d, 'endpoint_data'):
                e = d.endpoint_data
                if par in e.columns:
                    dic.end = True

            c = d.config
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
                self.v = self.param.v.crop_to_bounds(np.round(random.uniform(vmin, vmax), self.Ndec))
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
                    self.v = self.param.v.crop_to_bounds(np.round(vv, self.Ndec))
                elif self.parclass == param.Integer:
                    vmin, vmax = self.param.v.bounds
                    vr = np.abs(vmax - vmin)
                    v0 = self.v if self.v is not None else int(vmin + vr / 2)
                    vv = random.gauss(v0, Cmut * vr)
                    self.v = self.param.v.crop_to_bounds(int(vv))
                elif self.parclass == param.Magnitude:
                    v0 = self.v if self.v is not None else 0.5
                    vv = random.gauss(v0, Cmut)
                    self.v = self.param.v.crop_to_bounds(np.round(vv, self.Ndec))
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


def get_vfunc(dtype, lim, vs):
    func_dic = {
        float: param.Number,
        int: param.Integer,
        str: param.String,
        bool: param.Boolean,
        dict: param.Dict,
        list: param.List,
        type: param.ClassSelector,
        List[int]: param.List,
        List[str]: param.List,
        List[float]: param.List,
        List[Tuple[float]]: param.List,
        FunctionType: param.Callable,
        Tuple[float]: param.Range,
        Tuple[int]: param.NumericTuple,
        typing.TypedDict: param.Dict
    }
    if dtype == float and lim == (0.0, 1.0):
        return param.Magnitude
    if type(vs) == list and dtype in [str, int]:
        return param.Selector
    elif dtype in func_dic.keys():
        return func_dic[dtype]
    else :
        return param.Parameter



def vpar(vfunc, v0, h, lab, lim, dv, vs):
    f_kws = {
        'default': v0,
        'doc': h,
        'label': lab,
        'allow_None': True
    }
    if vfunc in [param.List, param.Number, param.Range]:
        if lim is not None:
            f_kws['bounds'] = lim
    if vfunc in [param.Range, param.Number]:
        if dv is not None:
            f_kws['step'] = dv
    if vfunc in [param.Selector]:
        f_kws['objects'] = vs
    # print(vfunc,v0, h, lab, lim, dv, vs)
    # if vfunc in [param.Dict] :
        # if v0 is not None :
        #     f_kws['class_'] = type(v0)
        # else :
        #     f_kws['class_'] = dict
        # print(f_kws)

    func = vfunc(**f_kws, instantiate=True)
    return func



def preparePar(p, k=None, dtype=float, d=None, disp=None, sym=None, symbol=None, codename=None, lab=None, h=None,
               u_name=None,mdict=None,
               required_ks=[], u=ureg.dimensionless, v0=None, v=None, lim=None, dv=None, vs=None,
               vfunc=None, vparfunc=None, func=None, **kwargs):
    codename = p if codename is None else codename
    d = p if d is None else d
    disp = d if disp is None else disp
    k = k if k is not None else d
    v0 = v if v is not None else v0
    if sym is None:
        if symbol is not None:
            sym = symbol
        else:
            sym = k

    if lab is None:
        if u == ureg.dimensionless:
            lab = f'{disp}'
        else:
            ulab=fr'${u}$'
            lab = fr'{disp} ({ulab})'
    if dv is None and dtype in [float, List[float], List[Tuple[float]], Tuple[float]]:
        dv = 0.01
    h = lab if h is None else h

    if vparfunc is None:
        if vfunc is None:
            vfunc = get_vfunc(dtype=dtype, lim=lim, vs=vs)
        vparfunc = vpar(vfunc, v0, h, lab, lim, dv, vs)
    else:
        vparfunc = vparfunc()

    kws = {
        'name': p,
        'p': p,
        'd': d,
        'k': k,
        'disp': disp,
        'sym': sym,
        'codename': codename,
        'dtype': dtype,
        'func': func,
        'u': u,
        'u_name': u_name,
        'required_ks': required_ks,
        'vparfunc': vparfunc,
        'mdict': mdict,
        'dv': dv,
        'v0': v0,

    }
    return dNl.NestDict(kws)
