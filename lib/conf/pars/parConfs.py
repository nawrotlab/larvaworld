import os
import random

import numpy as np
import pandas as pd
import param
from matplotlib import pyplot as plt
from lib.aux import dictsNlists as dNl
from lib.aux.par_aux import sub


def init2par(d0, d=None,pre_d=None, aux_args={}):
    from lib.conf.pars.par_dict import preparePar
    from lib.conf.pars.pars import v_descriptor
    if d is None and pre_d is None:
        d, pre_d = {},{}
    from lib.conf.base.dtypes import par
    for n, v in d0.items():
        depth = dNl.dict_depth(v)
        if depth == 0:
            continue
        if depth == 1:
            try:
                pkws = par(name=n, **v, convert2par=True)
                prepar = preparePar(**pkws)
                pre_d[prepar.k] = prepar
                p = v_descriptor(**prepar)
                if p is not None:
                    for kk, vv in aux_args.items():
                        setattr(p, kk, vv)
                    d[p.d] = p
            except:
                continue
        elif depth > 1:
            d[n], pre_d[n] = init2par(d0=v)
    return d, pre_d



class LarvaConfDict:
    def __init__(self, init_dict=None, mfunc=None,dist_dict=None):
        if init_dict is None :
            from lib.conf.pars.pars import ParDict
            init_dict=ParDict.init_dict
        if mfunc is None :
            from lib.conf.pars.par_funcs import module_func_dict
            mfunc=module_func_dict()
        if dist_dict is None :
            from lib.conf.pars.pars import ParDict
            dist_dict=ParDict.dist_dict
        self.dist_dict=dist_dict
        self.mfunc=mfunc
        self.mcolor = dNl.NestDict({
            'body' : 'lightskyblue',
            'physics' : 'lightsteelblue',
            'energetics' : 'lightskyblue',
            'Box2D_params': 'lightcoral',
            'crawler': 'lightcoral',
            'turner': 'indianred',
            'interference': 'lightsalmon',
            'intermitter': '#a55af4',
            'olfactor': 'palegreen',
            'windsensor': 'plum',
            'toucher': 'pink',
            'feeder': 'pink',
            'memory': 'pink',
            # 'locomotor': locomotor.DefaultLocomotor,
        })



        self.mbkeys = list(init_dict['modules'].keys())
        # self.mbkeys = list(self.mfunc.keys())
        self.mpref = {k: f'brain.{k}_params.' for k in self.mbkeys}
        self.mbdicts = dNl.NestDict()
        self.mbpredicts = dNl.NestDict()


        # self.mfunc['locomotor'] = locomotor.DefaultLocomotor
        # self.mfunc['brain'] = brain.DefaultBrain
        for k in self.mbkeys:
            self.mbdicts[k],self.mbpredicts[k] = init2par(d0 = init_dict[k], aux_args={'pref': self.mpref[k]})

        self.aux_keys = ['body', 'physics', 'energetics']
        # self.aux_keys = ['body', 'physics', 'energetics', 'Box2D_params']
        self.aux_dicts = dNl.NestDict()
        self.aux_predicts = dNl.NestDict()
        for k in self.aux_keys:
            self.aux_dicts[k],self.aux_predicts[k] = init2par(d0 = init_dict[k])

        self.mkeys=self.aux_keys+self.mbkeys
        self.mdicts = dNl.NestDict({**self.aux_dicts, **self.mbdicts})
        self.mpredicts = dNl.NestDict({**self.aux_predicts, **self.mbpredicts})

        def build_mpredfs(mpredicts):
            mpredfs=dNl.NestDict()
            for k,predict in mpredicts.items() :
                if predict is not None :
                    entries=[]
                    for kk,vv in predict.items() :
                        if 'k' in vv.keys() :
                            entries.append(vv)
                        else :
                            for kkk, vvv in vv.items():
                                if 'k' in vvv.keys():
                                    entries.append(vvv)
                                else :
                                    raise ValueError(kkk,kk,k)
                    mpredfs[k]=pd.DataFrame.from_records(entries, index='k')
                else :
                    mpredfs[k] = None
            return mpredfs







    def conf(self, mdict, prefix=False, **kwargs):
        conf0 = dNl.NestDict()
        for d, p in mdict.items():
            if isinstance(p, param.Parameterized):
                d0 = f'{p.pref}{d}' if prefix else d
                conf0[d0] = p.v
            else:
                conf0[d] = self.conf(mdict=p, prefix=False)

        conf0.update(kwargs)
        return conf0

    def multibconf(self, mbConf):
        multiconf = dNl.NestDict()
        for mkey, mdict in mbConf.items():
            if mkey == 'modules':
                multiconf.modules = mbConf.modules
            elif mdict is None:
                multiconf[mkey] = None
            else:
                multiconf[mkey] = self.conf(mdict)
        return multiconf

    def multiconf(self, mConf):
        mc = dNl.NestDict()
        mc.brain = self.multibconf(mConf['brain'])
        for mkey, mdict in mConf.items():
            if mkey=='brain':
                continue
            if mdict is None:
                mc[mkey] = None
            else:
                mc[mkey] = self.conf(mdict)
        return mc

    def module(self, mkey, **kwargs):
        mdict = self.mbdicts[mkey]
        conf0 = self.conf(mdict, prefix=False)
        func = self.mfunc[mkey]
        m = func(**conf0, **kwargs)
        return m

    def update_modelConf(self, mconf, mdict, **kwargs):
        conf0 = self.conf(mdict, prefix=True, **kwargs)
        return dNl.update_nestdict(mconf, conf0)

    def crossover(self, mdict, mdict2):
        for d, p in mdict.items():
            if random.random() < 0.5:
                p.v = mdict2[d].v

    def mutate(self, mdict, Pmut, Cmut):
        for d, p in mdict.items():
            p.mutate(Pmut, Cmut)

    def randomize(self, mdict):
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
        pdict = dNl.NestDict()
        for mkey, ds in dic.items():
            mdict = self.mbdicts[mkey]
            for d in ds:
                pdict[d] = mdict[d]
        return pdict

    def loco_conf(self, mkeys=None):
        mkeys0 = ['crawler', 'turner', 'interference', 'intermitter', 'feeder']
        if mkeys is None:
            mkeys = mkeys0
        conf = dNl.NestDict({'modules': {mkey: mkey in mkeys for mkey in mkeys0}})
        for mkey in mkeys0:
            if mkey in mkeys:
                mdict = self.mbdicts[mkey]
                conf[f'{mkey}_params'] = self.conf(mdict, prefix=False)
            else:
                conf[f'{mkey}_params'] = None
        return conf

    def loco_module(self, mkeys=None, **kwargs):
        conf = self.loco_conf(mkeys)
        L = self.mfunc['locomotor'](conf=conf, **kwargs)
        return L

    def brain_conf(self, mkeys=None):
        mkeys0 = self.mbkeys
        if mkeys is None:
            mkeys = mkeys0

        conf = dNl.NestDict({'modules': {mkey: mkey in mkeys for mkey in mkeys0}})
        for mkey in mkeys0:
            if mkey in mkeys:
                mdict = self.mbdicts[mkey]
                conf[f'{mkey}_params'] = self.conf(mdict, prefix=False)
            else:
                conf[f'{mkey}_params'] = None
        return conf

    def brain_module(self, mkeys=None, **kwargs):
        conf = self.brain_conf(mkeys)
        L = self.mfunc['brain'](conf=conf, **kwargs)
        return L

    def mIDbconf(self, mID=None, m=None):
        if m is None:
            from lib.conf.stored.conf import loadConf
            m = loadConf(mID, 'Model').brain
        mIDconf = dNl.NestDict()
        mIDconf.modules = m.modules
        for mkey, mdic in self.mbdicts.items():
            if m.modules[mkey]:
                mmdic = m[f'{mkey}_params']
                mdic = self.copyID(mdic, mmdic)
                mIDconf[f'{mkey}_params'] = mdic
            else:
                mIDconf[f'{mkey}_params'] = None
        return mIDconf

    def mIDconf(self, mID=None, m=None):
        if m is None:
            from lib.conf.stored.conf import loadConf
            m = loadConf(mID, 'Model')
        mc = dNl.NestDict()
        mc.brain =self.mIDbconf(self, m=m.brain)
        for mkey, mdic in self.aux_dicts.items():
            mmdic = m[mkey]
            if mmdic:
                mc[mkey] = self.copyID(mdic, mmdic)
            else:
                mc[mkey]= None
        return mc

    def mIDmodule(self, mID, module='brain', **kwargs):
        mbConf = self.mIDbconf(mID)
        multibconf = self.multibconf(mbConf)
        return self.mfunc[module](conf=multibconf, **kwargs)

    def copyID(self, mdic, mmdic):
        for d, p in mdic.items():
            if isinstance(p, param.Parameterized):
                new_v = mmdic[d] if d in mmdic.keys() else None
                if type(new_v) == list:
                    if p.parclass == param.Range:
                        new_v = tuple(new_v)
                p.v = new_v
            else:
                self.copyID(mdic=mdic[d], mmdic=mmdic[d])
        return mdic

    def mIDtable_data(self, mID, columns=['parameter', 'symbol', 'value', 'unit']):
        mConf = self.mIDconf(mID)
        m = self.multiconf(mConf)
        data = []

        def mvalid(k, dic):
            dvalid = dNl.NestDict({
                'interference': {
                    'square': ['crawler_phi_range', 'attenuation', 'attenuation_max'],
                    'phasic': ['max_attenuation_phase', 'attenuation', 'attenuation_max'],
                    'default': ['attenuation']
                },
                'turner': {
                    'neural': ['base_activation', 'activation_range', 'n', 'tau'],
                    'constant': ['initial_amp'],
                    'sinusoidal': ['initial_amp', 'initial_freq']
                },
                'crawler': {
                    'realistic': ['initial_freq', 'max_scaled_vel', 'max_vel_phase', 'stride_dst_mean',
                                  'stride_dst_std'],
                    'constant': ['initial_amp']
                },
                'physics': ['ang_damping', 'torque_coef', 'body_spring_k', 'bend_correction_coef'],
                'body': ['initial_length', 'Nsegs'],
                'energetics': [],
                'Box2D_params': [],
                'olfactor': ['decay_coef'],
                'windsensor': [],
                'toucher': [],
                'feeder': [],
                'memory': []
            })

            if k == 'interference':
                vals = dvalid[k][dic.mode]
            elif k == 'turner':
                vals = dvalid[k][dic.mode]
            elif k == 'crawler':
                vals = dvalid[k][dic.waveform]
            elif k == 'intermitter':
                vals = [n for n in ['stridechain_dist', 'pause_dist'] if
                            dic[n] is not None and dic[n].name is not None]
            else:
                vals = dvalid[k]
            return vals

        for k in self.mkeys:
            if k in self.aux_keys:
                dic = m[k]
                dic0 = mConf[k]
            elif k in self.mbkeys:
                dic = m['brain'][f'{k}_params']
                dic0 = mConf['brain'][f'{k}_params']
            if dic is None:
                valid = []
            else :
                valid = mvalid(k, dic)
            if len(valid) > 0:
                for n in valid:
                    if n in ['stridechain_dist', 'pause_dist']:
                        vv=dic[n]
                        dist_v = self.dist_dict[vv.name].lab_func(vv)
                        if n == 'stridechain_dist':
                            vs1 = [k, 'run length distribution', '$N_{R}$', dist_v, '-']
                            vs2 = [k, 'run length range', '$[N_{R}^{min},N_{R}^{max}]$', vv.range,
                                   '# $strides$']
                        elif n == 'pause_dist':
                            vs1 = [k, 'pause duration distribution', '$t_{P}$', dist_v, '-']
                            vs2 = [k, 'pause duration range', '$[t_{P}^{min},t_{P}^{max}]$', vv.range, '$sec$']
                        data.append(vs1)
                        data.append(vs2)
                    else:
                        ddd=[getattr(dic0[n], pname) for pname in columns]
                        data.append([k]+ddd)

        df = pd.DataFrame(data, columns=['field'] + columns)
        df.set_index(['field'], inplace=True)

        return df

    def mIDtable(self, mID, columns=['parameter', 'symbol', 'value', 'unit'], figsize=(14, 11), **kwargs):
        from lib.plot.table import render_conf_table
        df=self.mIDtable_data(mID, columns=columns)
        row_colors = [None] + [self.mcolor[ii] for ii in df.index.values]
        return render_conf_table(df, row_colors, figsize=figsize, **kwargs)







def confID_dict():
    from lib.conf.stored.conf import kConfDict, ConfSelector
    dic = dNl.NestDict()
    keys = ['Ref', 'Model', 'Env', 'Exp', 'Ga']
    for K0 in keys:
        k0 = K0.lower()
        k = f'{k0}ID'
        vparfunc = ConfSelector(K0, doc=f'The stored {K0} configurations as a list of IDs', label=sub('ID', k0))
        dic[K0] = vparfunc()
    return dic

#
# if __name__ == '__main__':
#     #
#     dd = LarvaConfDict()
#     dd.mIDtable(mID='PHIonNEU', show=True)
#     # print(dd.aux_keys)
#     # raise
