import copy
import random
from typing import Tuple

import pandas as pd
import param
from lib.aux import dictsNlists as dNl
from lib.aux.par_aux import sub

from lib.registry.units import ureg


def init2par(d0, d=None, pre_d=None, aux_args={}):
    def par(name, t=float, v=None, vs=None, lim=None, min=None, max=None, dv=None, disp=None, h='', k=None, symbol='',
            u=ureg.dimensionless, u_name=None, label='', codename=None,
            vfunc=None, vparfunc=None, **kwargs):
        from lib.aux.par_aux import define_range
        if k is None:
            k = name
        dv, lim, vs = define_range(dtype=t, lim=lim, vs=vs, dv=dv, min=min, max=max, u=u, wrap_mode=None)

        p_kws = {
            'p': name,
            'k': k,
            'lim': lim,
            'dv': dv,
            'vs': vs,
            'v0': v,
            'dtype': t,
            'disp': label,
            'h': h,
            'u_name': u_name,
            'u': u,
            'sym': symbol,
            'codename': codename,
            'vfunc': vfunc,
            'vparfunc': vparfunc,
        }
        return p_kws

    from lib.registry.par_dict import preparePar
    from lib.registry.par import v_descriptor
    if d is None and pre_d is None:
        d, pre_d = {}, {}
    for n, v in d0.items():
        depth = dNl.dict_depth(v)

        if depth == 0:
            continue
        if depth == 1:
            try:
                pkws = par(name=n, **v)
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
    def __init__(self, init_dict=None, mfunc=None, dist_dict0=None):



        if dist_dict0 is None:
            from lib.registry.pars import preg
            dist_dict0 = preg.dist_dict0
        self.dist_dict0 = dist_dict0
        self.dist_dict = self.dist_dict0.dict

        self.mcolor = dNl.NestDict({
            'body': 'lightskyblue',
            'physics': 'lightsteelblue',
            'energetics': 'lightskyblue',
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

        if init_dict is None:
            from lib.registry.pars import preg
            init_dict = preg.init_dict

        self.mbkeys = list(init_dict['modules'].keys())
        self.aux_keys = ['body', 'physics', 'energetics']
        self.mkeys = self.mbkeys + self.aux_keys

        self.dict1=self.build_mode1(init_dict=init_dict, mfunc=mfunc)
        self.dict2=self.build_mode2()






    def build_mode2(self):
        from lib.registry.modConfs import build_modConf_dict, build_aux_dict
        init_bdicts2, mbpredicts2, mbdicts2 = build_modConf_dict()
        init_auxdicts2, aux_predicts2, aux_dicts2 = build_aux_dict()

        mdicts2 = dNl.NestDict({**mbdicts2, **aux_dicts2})
        mpredicts2 = dNl.NestDict({**mbpredicts2, **aux_predicts2})
        init_dicts2 = dNl.NestDict({**init_bdicts2, **init_auxdicts2})

        bd={'init' : init_bdicts2, 'pre' : mbpredicts2, 'm' : mbdicts2}
        auxd={'init' : init_auxdicts2, 'pre' : aux_predicts2, 'm' : aux_dicts2}
        d={'init' : init_dicts2, 'pre' : mpredicts2, 'm' : mdicts2}

        dd={'brain' : bd, 'aux' : auxd, 'model' : d}
        return dNl.NestDict(dd)




    def build_mode1(self, init_dict, mfunc=None):


        if mfunc is None:
            from lib.registry.par_funcs import module_func_dict
            mfunc = module_func_dict()
        self.mfunc = mfunc

        mpref = {k: f'brain.{k}_params.' for k in self.mbkeys}
        init_bdicts = dNl.NestDict()
        mbdicts = dNl.NestDict()
        mbpredicts = dNl.NestDict()

        for k in self.mbkeys:
            init_bdicts[k] = init_dict[k]
            mbdicts[k], mbpredicts[k] = init2par(d0=init_bdicts[k], aux_args={'pref': mpref[k]})


        init_auxdicts = dNl.NestDict()
        aux_dicts = dNl.NestDict()
        aux_predicts = dNl.NestDict()
        for k in self.aux_keys:
            init_auxdicts[k] = init_dict[k]
            aux_dicts[k], aux_predicts[k] = init2par(d0=init_auxdicts[k])


        mdicts = dNl.NestDict({**mbdicts, **aux_dicts})
        mpredicts = dNl.NestDict({**mbpredicts, **aux_predicts})
        init_dicts = dNl.NestDict({**init_bdicts, **init_auxdicts})

        def build_mpredfs(mpredicts):
            mpredfs = dNl.NestDict()
            for k, predict in mpredicts.items():
                if predict is not None:
                    entries = []
                    for kk, vv in predict.items():
                        if 'k' in vv.keys():
                            entries.append(vv)
                        else:
                            for kkk, vvv in vv.items():
                                if 'k' in vvv.keys():
                                    entries.append(vvv)
                                else:
                                    raise ValueError(kkk, kk, k)
                    mpredfs[k] = pd.DataFrame.from_records(entries, index='k')
                else:
                    mpredfs[k] = None
            return mpredfs

        bd = {'init': init_bdicts, 'pre': mbpredicts, 'm': mbdicts}
        auxd = {'init': init_auxdicts, 'pre': aux_predicts, 'm': aux_dicts}
        d = {'init': init_dicts, 'pre': mpredicts, 'm': mdicts}

        dd = {'brain': bd, 'aux': auxd, 'model': d}
        return dNl.NestDict(dd)


    def conf2(self, mdict=None, mkey=None, prefix=False, mode=None, refID=None, **kwargs):
        conf0 = dNl.NestDict()
        if mdict is None:
            if mkey is not None:
                if mode is None:
                    mdict = self.dict2.model.m[mkey].args
                else:
                    mdict = self.dict2.model.m[mkey].mode[mode].args
            else:
                raise ValueError('Module dictionary or key must be defined')
        for d, p in mdict.items():
            if isinstance(p, param.Parameterized):
                d0 = f'{p.pref}{d}' if prefix else d
                conf0[d0] = p.v
            else:
                conf0[d] = self.conf2(mdict=p, prefix=False)

        conf0.update(kwargs)
        if refID is not None and mkey == 'intermitter':
            try:
                from lib.aux.sim_aux import get_sample_bout_distros0
                from lib.conf.stored.conf import loadConf
                kkkws = {
                    'Im': conf0,
                    'bout_distros': loadConf(refID, 'Ref').bout_distros,
                }
                conf0 = get_sample_bout_distros0(**kkkws)
            except:
                pass
            # bout_distros = sample.bout_distros
        return dNl.NestDict(conf0)

    def module2(self, mkey, mode=None, refID=None, **kwargs):
        if mode is None:
            mdict = self.dict2.brain.m[mkey]
        else:
            mdict = self.dict2.brain.m[mkey].mode[mode]
        mkws = self.dict2.brain.m[mkey].kwargs
        conf0 = self.conf2(mdict=mdict.args, prefix=False, refID=refID)
        func = mdict.class_func
        mkws.update(kwargs)
        m = func(**conf0, **mkws)
        return m

    def conf(self, mdict=None, mkey=None, prefix=False, **kwargs):
        conf0 = dNl.NestDict()
        if mdict is None:
            if mkey is not None:
                mdict = self.dict1.model.m[mkey]
            else:
                raise ValueError('Module dictionary or key must be defined')
        for d, p in mdict.items():
            if isinstance(p, param.Parameterized):
                d0 = f'{p.pref}{d}' if prefix else d
                conf0[d0] = p.v
            else:
                conf0[d] = self.conf(mdict=p, prefix=False)

        conf0.update(kwargs)
        return conf0

    def multibconf(self, mbConf, mode=1):
        multiconf = dNl.NestDict()
        for mkey, mdict in mbConf.items():
            if mkey == 'modules':
                multiconf.modules = mbConf.modules
            elif mdict is None:
                multiconf[mkey] = None
            else:
                if mode == 1:
                    multiconf[mkey] = self.conf(mdict)
                elif mode == 2:
                    multiconf[mkey] = self.conf2(mdict)
        return multiconf

    def multiconf(self, mConf, mode=1):
        mc = dNl.NestDict()
        mc.brain = self.multibconf(mConf['brain'], mode=mode)
        for mkey, mdict in mConf.items():
            if mkey == 'brain':
                continue
            if mdict is None:
                mc[mkey] = None
            else:
                if mode == 1:
                    mc[mkey] = self.conf(mdict)
                elif mode == 2:
                    mc[mkey] = self.conf2(mdict)
                # mc[mkey] = self.conf(mdict)
        return mc

    def module(self, mkey, **kwargs):
        mdict = self.dict1.brain.m[mkey]
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
            mdict = self.dict1.brain.m[mkey]
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
                mdict = self.dict1.brain.m[mkey]
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
                mdict = self.dict1.brain.m[mkey]
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
        for mkey, mdic in self.dict1.brain.m.items():
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
        mc.brain = self.mIDbconf(self, m=m.brain)
        for mkey, mdic in self.dict1.aux.m.items():
            mmdic = m[mkey]
            mc[mkey] = self.copyID(mdic, mmdic)
            # if mmdic:
            #
            # else:
            #     mc[mkey]= None
        return mc

    def mIDmodule(self, mID, module='brain', **kwargs):
        mbConf = self.mIDbconf(mID)
        multibconf = self.multibconf(mbConf)
        return self.mfunc[module](conf=multibconf, **kwargs)

    def copyID(self, mdic, mmdic):
        if mmdic is None:
            return None
        else:
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

    def mIDtable_data(self, mID, columns=['parameter', 'symbol', 'value', 'unit'], **kwargs):
        mConf = self.mIDconf(mID, **kwargs)
        m = self.multiconf(mConf, **kwargs)
        data = []

        # print(mID,m.energetics)
        def gen_rows(mdic, mConf_dic, parent, data0, suf_keys=False):

            for n in mdic:
                p = mConf_dic[n]
                if isinstance(p, param.Parameterized):
                    ddd = [getattr(p, pname) for pname in columns]
                    row = [parent] + ddd
                    data0.append(row)
                else:
                    if suf_keys:
                        new_parent = f'{parent}.{n}'
                    else:
                        new_parent = parent
                    print(p)
                    print(isinstance(p, dict))
                    print(k, p == mConf_dic[n])
                    for ii in p:
                        iii = mConf_dic[n][ii]
                        print(ii == iii)
                    data0 = gen_rows(p, mConf_dic[n], new_parent, data0, suf_keys=suf_keys)
            return data0

        def mvalid(k, dic, data0):
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
                'energetics': ['DEB'],
                'Box2D_params': [],
                'olfactor': ['decay_coef', 'perception'],
                'windsensor': ['weights'],
                'toucher': ['touch_sensors', 'decay_coef', 'perception', 'initial_gain'],
                'feeder': ['initial_freq', 'feed_radius', 'V_bite'],
                'memory': [],
                # 'intermitter': ['stridechain_dist', 'pause_dist']
            })

            if k == 'interference':
                vals = dvalid[k][dic.mode]
            elif k == 'turner':
                vals = dvalid[k][dic.mode]
            elif k == 'crawler':
                vals = dvalid[k][dic.waveform]
            elif k == 'intermitter':
                for kkk in ['stridechain_dist', 'pause_dist', 'run_dist']:
                    if dic[kkk] is not None:
                        if dic[kkk].name is not None:
                            vs1, vs2 = self.dist_dict0.get_dist(k=kkk, k0=k, v=dic[kkk], return_tabrows=True)
                            data0.append(vs1)
                            data0.append(vs2)

                vals = []
            else:
                vals = dvalid[k]

            return vals, data0

        for k in self.mkeys:
            if k in self.mbkeys:
                dic = m['brain'][f'{k}_params']
                dic0 = mConf['brain'][f'{k}_params']


            elif k in self.aux_keys:
                dic = m[k]
                dic0 = mConf[k]

            if dic is None:
                valid = []
            else:
                valid, data = mvalid(k, dic, data)

            if len(valid) > 0:
                data = gen_rows(valid, dic0, k, data)

        df = pd.DataFrame(data, columns=['field'] + columns)
        df.set_index(['field'], inplace=True)
        return df

    def mIDtable(self, mID, columns=['parameter', 'symbol', 'value', 'unit'], figsize=(14, 11), **kwargs):
        from lib.plot.table import conf_table
        df = self.mIDtable_data(mID, columns=columns)
        row_colors = [None] + [self.mcolor[ii] for ii in df.index.values]
        return conf_table(df, row_colors, mID=mID, figsize=figsize, **kwargs)

    def init_loco(self, conf, L):
        D = self.dict2.model.m
        for k in ['crawler', 'turner', 'interference', 'feeder', 'intermitter']:
            if conf.modules[k]:
                m = conf[f'{k}_params']
                if k == 'crawler' :
                    mode = m.waveform
                elif k == 'feeder' :
                    mode = 'default'
                else :
                    mode = m.mode
                kws = {kw: getattr(L, kw) for kw in D[k].kwargs.keys()}
                M = D[k].mode[mode].class_func(**m, **kws)
                if k == 'intermitter':
                    M.disinhibit_locomotion(L)
                if k == 'crawler':
                    M.waveform = m.waveform
            else:
                M = None
            setattr(L, k, M)
        return L

    def init_brain(self, conf, B):
        D = self.dict2.model.m
        for k in ['olfactor', 'toucher', 'windsensor', 'thermosensor']:
            if conf.modules[k]:
                m = conf[f'{k}_params']
                if k == 'windsensor':
                    m.gain_dict={'windsensor': 1.0}
                mode = 'default'
                kws = {kw: getattr(B, kw) for kw in D[k].kwargs.keys()}
                M = D[k].mode[mode].class_func(**m, **kws)
                if k == 'toucher':
                    M.init_sensors(brain=B)


            else:
                M = None
            setattr(B, k, M)
        B.touch_memory = None
        B.memory = None
        if conf.modules['memory']:
            mm = conf['memory_params']
            mode= mm['modality']
            kws = {kw: getattr(B, kw) for kw in D['memory'].kwargs.keys()}
            if mode=='olfaction' and B.olfactor :
                mm.gain=B.olfactor.gain
                B.memory = D['memory'].mode[mode].class_func(**mm, **kws)
            elif mode=='touch' and B.toucher :
                mm.gain=B.toucher.gain
                B.touch_memory = D['memory'].mode[mode].class_func(**mm, **kws)
        return B

    def init_loco_mID(self, mID):
        from lib.conf.stored.conf import loadConf
        m = loadConf(mID, 'Model')
        from lib.model.modules.locomotor import DefaultLocomotor
        L = DefaultLocomotor(conf=m.brain)
        return L

    def init_brain_mID(self, mID):
        from lib.conf.stored.conf import loadConf
        m = loadConf(mID, 'Model')
        from lib.model.modules.brain import DefaultBrain
        # L = DefaultLocomotor(conf=m.brain)
        B = DefaultBrain(conf=m.brain)
        return B


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
