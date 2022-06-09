import random
from typing import Tuple

import numpy as np
import param
from unflatten import unflatten

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


def vpar(vfunc, v0, h, lab, lim, dv,vs):
    f_kws = {
        'default': v0,
        'doc': h,
        'label': lab,
    }
    if vfunc in [param.List,param.Number, param.Range]:
        if lim is not None:
            f_kws['bounds'] = lim
    if vfunc in [param.Range,param.Number]:
        if dv is not None:
            f_kws['step'] = dv
    if vfunc in [param.Selector]:
        f_kws['objects'] = vs
    func = vfunc(**f_kws, instantiate=True)
    return func


def buildBasePar(p, k, dtype=float, d=None, disp=None, sym=None, codename=None, lab=None, h=None, u_name=None,
                 required_ks=[], u=ureg.dimensionless, v0=None, lim=None, dv=None,vs=None,
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
        vparfunc = vpar(vfunc, v0, h, lab, lim, dv,vs)
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
            if self.parclass == param.Number :
                vmin, vmax = self.param.v.bounds
                self.v=random.uniform(vmin, vmax)
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
                self.v = (vv0,vv1)

        def mutate(self, Pmut, Cmut):
            if random.random() < Pmut:
                if self.parclass == param.Number :

                    vmin, vmax = self.param.v.bounds
                    vr = np.abs(vmax - vmin)
                    v0 = self.v if self.v is not None else vmin+vr/2
                    vv = random.gauss(v0, Cmut * vr)
                    self.v=self.param.v.crop_to_bounds(vv)
                elif self.parclass == param.Integer:
                    vmin, vmax = self.param.v.bounds
                    vr = np.abs(vmax - vmin)
                    v0 = self.v if self.v is not None else int(vmin + vr / 2)
                    vv = random.gauss(v0, Cmut * vr)
                    self.v=self.param.v.crop_to_bounds(int(vv))
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
                    v0,v1=self.v if self.v is not None else (vmin, vmax)
                    vv0 = random.gauss(v0, Cmut * vr)
                    vv1 = random.gauss(v1, Cmut * vr)
                    vv0=np.clip(vv0, a_min=vmin,a_max=vmax)
                    vv1=np.clip(vv1, a_min=vv0,a_max=vmax)
                    self.v = (vv0,vv1)



    par = LarvaworldParNew(name=p, p=p, k=k, d=d, dtype=dtype, disp=disp, sym=sym, codename=codename)
    par.param.add_parameter('func', param.Callable(default=func, doc='Function to get the parameter from a dataset',
                                                   constant=True, allow_None=True))
    par.u = u
    par.required_ks = required_ks
    return par


def init2par(d0=None, d=None, aux_args={}):
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
                p = par(name=n, **v, convert2par=True)
                if p is not None :
                    for kk,vv in aux_args.items() :
                        setattr(p,kk,vv)
                    d[p.d]=p
            except:
                continue
        elif depth > 1:
            d[n]=init2par(v)
    return d


class ModuleConfDict:
    def __init__(self):
        from lib.conf.base.init_pars import init_pars
        from lib.model.modules import crawler, turner, intermitter, crawl_bend_interference, sensor, memory, feeder, locomotor, brain
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
            # 'locomotor': locomotor.DefaultLocomotor,
        }


        self.mkeys = list(self.mfunc.keys())
        self.mpref = {k: f'brain.{k}_params.' for k in self.mkeys}
        self.mdicts = dNl.AttrDict.from_nested_dicts({})
        self.mfunc['locomotor']=locomotor.DefaultLocomotor
        self.mfunc['brain']=brain.DefaultBrain
        for k in self.mkeys :
            mdic=init2par(d0=dinit[k], aux_args={'pref' : self.mpref[k]})
            # for d,p in mdic.items() :
            #     p.pref=self.mpref[k]
            self.mdicts[k]=mdic


    def conf(self, mdict,prefix=False, **kwargs):
        conf0 = dNl.AttrDict.from_nested_dicts({})
        for d, p in mdict.items() :
            if isinstance(p, param.Parameterized) :
                d0=f'{p.pref}{d}' if prefix else d
                conf0[d0]=p.v
            else :
                conf0[d] = self.conf(mdict=p, prefix=False)

        conf0.update(kwargs)
        return conf0

    def module(self, mkey, **kwargs):
        mdict = self.mdicts[mkey]
        conf0 = self.conf(mdict,prefix=False)
        func= self.mfunc[mkey]
        m=func(**conf0, **kwargs)
        return m

    def update_modelConf(self, mconf, mdict, **kwargs):
        conf0 = self.conf(mdict, prefix=True, **kwargs)
        return dNl.update_nestdict(mconf,conf0)

    def crossover(self, mdict, mdict2):
        # mdict = self.mdicts[mkey]
        for d,p in mdict.items() :
            if random.random() < 0.5 :
                p.v = mdict2[d].v


    def mutate(self, mdict, Pmut, Cmut):
        # mdict = self.mdicts[mkey]
        for d,p in mdict.items() :
            p.mutate(Pmut, Cmut)

    def randomize(self, mdict):
        # mdict = self.mdicts[mkey]
        for d,p in mdict.items() :
            p.randomize()

    def initConf(self,init_mode, mdict, mconf0,**kwargs):
        if init_mode=='model' :
            conf=self.conf(mdict,prefix=True,**mconf0)
            # return mconf0
        elif init_mode=='default' :
            conf = self.conf(mdict, prefix=True)
            # return self.update_modelConf(mconf0, mdict,**kwargs)
        elif init_mode=='random' :
            self.randomize(mdict)
            conf = self.conf(mdict, prefix=True)
        return conf

    def compile_pdict(self, dic):
        pdict = dNl.AttrDict.from_nested_dicts({})
        for mkey, ds in dic.items() :
            mdict = self.mdicts[mkey]
            for d in ds :
                pdict[d]=mdict[d]
        return pdict

    def loco_conf(self,mkeys=None):
        mkeys0 = ['crawler', 'turner', 'interference', 'intermitter', 'feeder']
        if mkeys is None :
            mkeys=mkeys0
        conf=dNl.AttrDict.from_nested_dicts({'modules': {mkey : mkey in mkeys for mkey in mkeys0}})
        for mkey in mkeys0 :
            if mkey in mkeys :
                mdict = self.mdicts[mkey]
                conf[f'{mkey}_params'] = self.conf(mdict, prefix=False)
            else :
                conf[f'{mkey}_params'] = None
        return conf

    def loco_module(self,mkeys=None, **kwargs):
        conf=self.loco_conf(mkeys)
        L=self.mfunc['locomotor'](conf=conf, **kwargs)
        return L

    def brain_conf(self,mkeys=None):
        mkeys0 = self.mkeys
        if mkeys is None :
            mkeys=mkeys0

        conf=dNl.AttrDict.from_nested_dicts({'modules': {mkey : mkey in mkeys for mkey in mkeys0}})
        for mkey in mkeys0 :
            if mkey in mkeys :
                mdict = self.mdicts[mkey]
                conf[f'{mkey}_params'] = self.conf(mdict, prefix=False)
            else :
                conf[f'{mkey}_params'] = None
        return conf

    def brain_module(self,mkeys=None, **kwargs):
        conf=self.brain_conf(mkeys)
        L=self.mfunc['brain'](conf=conf, **kwargs)
        return L




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
    # dic={'crawler' : ['initial_freq', 'freq_range'], 'turner' : ['base_activation', 'tau'], 'interference' : ['attenuation', 'crawler_phi_range']}
    # dd = ModuleConfDict()
    # pdict = dd.compile_pdict(dic)
    # dd.mutate(pdict,0.8,0.3)
    # print(dd.conf(pdict))
    # print()
    # print()
    # print()
    #print(dd.conf(pdict, prefix=True))
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
