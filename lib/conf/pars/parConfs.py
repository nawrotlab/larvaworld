import random

import param

from lib.aux import dictsNlists as dNl
from lib.aux.par_aux import sub


from lib.conf.stored.conf import loadConf


def init2par(d0=None, d=None,k=None, aux_args={}):
    from lib.conf.pars.par_dict import preparePar
    from lib.conf.pars.pars import v_descriptor
    if d0 is None:
        from lib.conf.base.init_pars import init_pars
        if k is None :
            d0 = init_pars()
        else:
            d0 = init_pars()[k]

    if d is None:
        d = {}
    from lib.conf.base.dtypes import par
    for n, v in d0.items():
        depth = dNl.dict_depth(v)
        if depth == 0:
            continue
        if depth == 1:
            try:
                pkws = par(name=n, **v, convert2par=True)
                prepar = preparePar(**pkws)
                p = v_descriptor(**prepar)
                if p is not None:
                    for kk, vv in aux_args.items():
                        setattr(p, kk, vv)
                    d[p.d] = p
            except:
                continue
        elif depth > 1:
            d[n] = init2par(v)
    return d




class LarvaConfDict:
    def __init__(self):
        from lib.model.modules import crawler, turner, intermitter, crawl_bend_interference, sensor, memory, feeder, \
            locomotor, brain
        # dinit = init_pars()
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
        self.mfunc['locomotor'] = locomotor.DefaultLocomotor
        self.mfunc['brain'] = brain.DefaultBrain
        for k in self.mkeys:
            mdic = init2par(k=k, aux_args={'pref': self.mpref[k]})

            # for d,p in mdic.items() :
            #     p.pref=self.mpref[k]
            self.mdicts[k] = mdic

        self.aux_keys = ['body', 'physics', 'energetics', 'Box2D_params']
        self.aux_dicts = dNl.AttrDict.from_nested_dicts({k: init2par(k=k) for k in self.aux_keys})

    def conf(self, mdict, prefix=False, **kwargs):
        conf0 = dNl.AttrDict.from_nested_dicts({})
        for d, p in mdict.items():
            if isinstance(p, param.Parameterized):
                d0 = f'{p.pref}{d}' if prefix else d
                conf0[d0] = p.v
            else:
                conf0[d] = self.conf(mdict=p, prefix=False)

        conf0.update(kwargs)
        return conf0

    def multiconf(self, mConf):
        multiconf = dNl.AttrDict({})
        for mkey, mdict in mConf.items():
            if mkey == 'modules':
                multiconf.modules = mConf.modules
            elif mdict is None:
                multiconf[mkey] = None
            else:
                multiconf[mkey] = self.conf(mdict)
        return multiconf

    def module(self, mkey, **kwargs):
        mdict = self.mdicts[mkey]
        conf0 = self.conf(mdict, prefix=False)
        func = self.mfunc[mkey]
        m = func(**conf0, **kwargs)
        return m

    def update_modelConf(self, mconf, mdict, **kwargs):
        conf0 = self.conf(mdict, prefix=True, **kwargs)
        return dNl.update_nestdict(mconf, conf0)

    def crossover(self, mdict, mdict2):
        # mdict = self.mdicts[mkey]
        for d, p in mdict.items():
            if random.random() < 0.5:
                p.v = mdict2[d].v

    def mutate(self, mdict, Pmut, Cmut):
        # mdict = self.mdicts[mkey]
        for d, p in mdict.items():
            p.mutate(Pmut, Cmut)

    def randomize(self, mdict):
        # mdict = self.mdicts[mkey]
        for d, p in mdict.items():
            p.randomize()

    def initConf(self, init_mode, mdict, mconf0, **kwargs):
        if init_mode == 'model':
            conf = self.conf(mdict, prefix=True, **mconf0)
            # return mconf0
        elif init_mode == 'default':
            conf = self.conf(mdict, prefix=True)
            # return self.update_modelConf(mconf0, mdict,**kwargs)
        elif init_mode == 'random':
            self.randomize(mdict)
            conf = self.conf(mdict, prefix=True)
        return conf

    def compile_pdict(self, dic):
        pdict = dNl.AttrDict.from_nested_dicts({})
        for mkey, ds in dic.items():
            mdict = self.mdicts[mkey]
            for d in ds:
                pdict[d] = mdict[d]
        return pdict

    def loco_conf(self, mkeys=None):
        mkeys0 = ['crawler', 'turner', 'interference', 'intermitter', 'feeder']
        if mkeys is None:
            mkeys = mkeys0
        conf = dNl.AttrDict.from_nested_dicts({'modules': {mkey: mkey in mkeys for mkey in mkeys0}})
        for mkey in mkeys0:
            if mkey in mkeys:
                mdict = self.mdicts[mkey]
                conf[f'{mkey}_params'] = self.conf(mdict, prefix=False)
            else:
                conf[f'{mkey}_params'] = None
        return conf

    def loco_module(self, mkeys=None, **kwargs):
        conf = self.loco_conf(mkeys)
        L = self.mfunc['locomotor'](conf=conf, **kwargs)
        return L

    def brain_conf(self, mkeys=None):
        mkeys0 = self.mkeys
        if mkeys is None:
            mkeys = mkeys0

        conf = dNl.AttrDict.from_nested_dicts({'modules': {mkey: mkey in mkeys for mkey in mkeys0}})
        for mkey in mkeys0:
            if mkey in mkeys:
                mdict = self.mdicts[mkey]
                conf[f'{mkey}_params'] = self.conf(mdict, prefix=False)
            else:
                conf[f'{mkey}_params'] = None
        return conf

    def brain_module(self, mkeys=None, **kwargs):
        conf = self.brain_conf(mkeys)
        L = self.mfunc['brain'](conf=conf, **kwargs)
        return L

    def mIDconf(self, mID=None, m=None):
        if m is None :
            m = loadConf(mID, 'Model').brain
        mIDconf = dNl.AttrDict({})
        mIDconf.modules = m.modules
        for mkey, mdic in self.mdicts.items():
            if m.modules[mkey]:
                mmdic = m[f'{mkey}_params']
                mdic = self.copyID(mdic, mmdic)
                mIDconf[f'{mkey}_params'] = mdic
            else:
                mIDconf[f'{mkey}_params'] = None
        return mIDconf

    def mIDmodule(self,mID,module='brain', **kwargs):
        mConf=self.mIDconf(mID)
        multiconf=self.multiconf(mConf)
        return self.mfunc[module](conf=multiconf, **kwargs)



    def copyID(self, mdic, mmdic):
        for d, p in mdic.items():
            if isinstance(p, param.Parameterized):
                new_v = mmdic[d] if d in mmdic.keys() else None
                if type(new_v) == list:
                    if p.parclass == param.Range:
                        new_v = tuple(new_v)
                # print(d,p.v==new_v,p.v,new_v)
                p.v = new_v
            else:
                self.copyID(mdic=mdic[d], mmdic=mmdic[d])
        return mdic


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
    # print(kConfDict('Model'))
    # mID='forager'
    # m=loadConf(mID, 'Model').brain.turner_params
    # print(m)
    #
    dd = LarvaConfDict()
