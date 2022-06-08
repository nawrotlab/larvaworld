from typing import Tuple

import param

from lib.aux import dictsNlists as dNl
from lib.aux.par_aux import sub
from lib.conf.base.units import ureg

func_dic = {
    float: param.Number,
    int: param.Integer,
    str: param.String,
    bool: param.Boolean,
    dict: param.Dict,
    list: param.List,
    Tuple[float]: param.Range,
    Tuple[int]: param.NumericTuple,
}


def vpar(vfunc, v0, h, lab, lim, dv):
    f_kws = {
        'default': v0,
        'doc': h,
        'label': lab,
    }
    if vfunc in [param.List, param.Range]:
        if lim is not None:
            f_kws['bounds'] = lim
        if dv is not None:
            f_kws['step'] = dv
    func = vfunc(**f_kws, instantiate=True)
    return func


def buildBasePar(p, k, dtype=float, d=None, disp=None, sym=None, codename=None, lab=None, h=None, u_name=None,
                 required_ks=[], u=ureg.dimensionless, v0=None, lim=None, dv=None,
                 vfunc=None, vparfunc=None, func=None, **kwargs):
    codename = p if codename is None else codename
    d = p if d is None else d
    disp = d if disp is None else disp
    sym = k if sym is None else sym
    lab = f'{disp} ({u})' if lab is None else lab
    h = lab if h is None else h
    if vparfunc is None:
        if vfunc is None:
            vfunc = func_dic[dtype]
        vparfunc = vpar(vfunc, v0, h, lab, lim, dv)
    else:
        vparfunc = vparfunc()

    class LarvaworldParNew(param.Parameterized):

        p = param.String('p')
        d = param.String('')
        disp = param.String('')
        k = param.String('k')
        sym = param.String('')
        codename = param.String('')
        dtype = param.Parameter(float)
        v = vparfunc

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

    par = LarvaworldParNew(name=p, p=p, k=k, d=d, dtype=dtype, disp=disp, sym=sym, codename=codename)
    par.param.add_parameter('func', param.Callable(default=func, doc='Function to get the parameter from a dataset',
                                                   constant=True, allow_None=True))
    par.u = u
    par.required_ks = required_ks
    return par


def init2par(d0=None, d=None):
    if d0 is None:
        from lib.conf.base.init_pars import init_pars
        d0 = init_pars()

    if d is None:
        d = {}
    from lib.conf.base.dtypes import par
    for n, v in d0.items():
        depth = dNl.dict_depth(v)
        if depth == 0:
            continue
        if depth == 1:
            try:
                entry = par(name=n, **v, convert2par=True)
                d.update(entry)
            except:
                continue
        elif depth > 1:
            init2par(v, d=d)
    return d


class ModuleConfDict:
    def __init__(self):
        from lib.conf.base.init_pars import init_pars
        from lib.model.modules import crawler, turner, intermitter, crawl_bend_interference, sensor, memory, feeder
        dinit=init_pars()
        self.mfunc = {
            'crawler': crawler.Crawler,
            'turner': turner.Turner,
            'interference': crawl_bend_interference.Coupling,
            'intermitter': intermitter.ChoiceIntermitter,
            'olfactor': sensor.Olfactor,
            'windsensor': sensor.WindSensor,
            'toucher': sensor.Toucher,
            'feeder': feeder.Feeder,
            'memory': memory.RLmemory,
        }
        self.mkeys = list(self.mfunc.keys())
        self.mdicts = dNl.AttrDict.from_nested_dicts({k: init2par(d0=dinit[k]) for k in self.mkeys})

    def conf(self, mkey, **kwargs):
        mdict = self.mdicts[mkey]
        conf0 = dNl.AttrDict.from_nested_dicts({p.d: p.v for k, p in mdict.items()})
        conf0.update(kwargs)
        return conf0

    def module(self, mkey, **kwargs):
        conf0 = self.conf(mkey, **kwargs)
        func= self.mfunc[mkey]
        m=func(**conf0)
        return m


def confID_dict():
    from lib.conf.stored.conf import kConfDict, ConfSelector
    dic = dNl.AttrDict.from_nested_dicts({})
    keys = ['Ref', 'Model', 'Env', 'Exp', 'Ga']
    for K0 in keys:
        k0 = K0.lower()
        k = f'{k0}ID'
        # print(K0,kConfDict(K0))

        # p=buildBasePar(p=k, k=k, dtype=str, d=None, disp=f'{K0} IDs', sym=sub('ID', k0), codename=None, lab=f'{K0} IDs',
        #            h=f'The stored {k0} configurations as a list of IDs',v0=None,vparfunc=ConfSelector(K0))
        vparfunc = ConfSelector(K0, doc=f'The stored {K0} configurations as a list of IDs', label=sub('ID', k0))
        dic[K0] = vparfunc()
    return dic


if __name__ == '__main__':
    pass
    # dd = ModuleConfDict()
    # from lib.conf.stored.conf import loadConf
    # dt=0.1
    # mID='explorer'
    # m = loadConf(mID, 'Model').brain
    #
    # for mkey, exists in m.modules.items() :
    #     if exists :
    #         kws=m[f'{mkey}_params']
    #         M = dd.module(mkey=mkey,dt=dt, **kws)
    #         print(mkey, M)

    # T=dd.module(mkey='turner', noise=0.5)
    # for i in range(1000) :
    #     T.step()
    #     print(T.activity)
    # print(T)
    # from lib.conf.base.init_pars import init_pars
    # dd=init2par(d0=init_pars()['turner'], d=None)
    # print(dd)
    # for k,p in pp.items():
    #     print(k,p.v, p.d)
