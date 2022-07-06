import copy
import random
from typing import Tuple

import numpy as np
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
    def __init__(self, init_dict=None, dist_dict0=None):

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

        self.dict1 = self.build_mode1(init_dict=init_dict)
        self.dict2 = self.build_mode2()

    def build_mode2(self):
        from lib.registry.modConfs import build_brain_module_dict, build_aux_module_dict
        init_bdicts2, mbpredicts2, mbdicts2 = build_brain_module_dict()
        init_auxdicts2, aux_predicts2, aux_dicts2 = build_aux_module_dict()

        mdicts2 = dNl.NestDict({**mbdicts2, **aux_dicts2})
        mpredicts2 = dNl.NestDict({**mbpredicts2, **aux_predicts2})
        init_dicts2 = dNl.NestDict({**init_bdicts2, **init_auxdicts2})

        bd = {'init': init_bdicts2, 'pre': mbpredicts2, 'm': mbdicts2}
        auxd = {'init': init_auxdicts2, 'pre': aux_predicts2, 'm': aux_dicts2}
        d = {'init': init_dicts2, 'pre': mpredicts2, 'm': mdicts2}

        dd = {'brain': bd, 'aux': auxd, 'model': d}
        return dNl.NestDict(dd)

    def build_mode1(self, init_dict):

        # if mfunc is None:
        #     from lib.registry.par_funcs import module_func_dict
        #     mfunc = module_func_dict()
        # self.mfunc = mfunc

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

    def get_mdict2(self, mkey, mode=None):
        if mkey is not None:
            if mkey in self.aux_keys:
                mdict = self.dict2.model.m[mkey].args
            elif mkey in self.mbkeys:
                if mode is None:
                    mode = 'default'
                mdict = self.dict2.model.m[mkey].mode[mode].args
            return mdict
        else:
            raise ValueError('Module dictionary or key must be defined')

    def conf2(self, mdict=None, mkey=None, prefix=False, mode=None, refID=None, **kwargs):

        if mdict is None:
            mdict = self.get_mdict2(mkey, mode)
        conf0 = dNl.NestDict()
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

    def module2(self, mkey, mode=None, refID=None, mkwargs={}, **kwargs):
        if mode is None:
            mdict = self.dict2.brain.m[mkey]
        else:
            mdict = self.dict2.brain.m[mkey].mode[mode]
        mkws = self.dict2.brain.m[mkey].kwargs
        conf0 = self.conf2(mdict=mdict.args, prefix=False, refID=refID, **kwargs)
        func = mdict.class_func
        mkws.update(mkwargs)
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
        from lib.model.modules.locomotor import DefaultLocomotor

        conf = self.loco_conf(mkeys)
        L = DefaultLocomotor(conf=conf, **kwargs)
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
        from lib.model.modules.brain import DefaultBrain
        conf = self.brain_conf(mkeys)
        L = DefaultBrain(conf=conf, **kwargs)
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

            if k in ['crawler', 'turner', 'interference']:
                vals = dvalid[k][dic.mode]
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
                continue
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
                if k == 'feeder':
                    mode = 'default'
                else:
                    mode = m.mode
                kws = {kw: getattr(L, kw) for kw in D[k].kwargs.keys()}
                M = D[k].mode[mode].class_func(**m, **kws)
                if k == 'intermitter':
                    M.disinhibit_locomotion(L)
                if k == 'crawler':
                    M.mode = m.mode
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
                    m.gain_dict = {'windsensor': 1.0}
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
            mode = mm['modality']
            kws = {kw: getattr(B, kw) for kw in D['memory'].kwargs.keys()}
            if mode == 'olfaction' and B.olfactor:
                mm.gain = B.olfactor.gain
                B.memory = D['memory'].mode[mode].class_func(**mm, **kws)
            elif mode == 'touch' and B.toucher:
                mm.gain = B.toucher.gain
                B.touch_memory = D['memory'].mode[mode].class_func(**mm, **kws)
        return B


class LarvaConfDict2:
    def __init__(self, dist_dict0=None):
        if dist_dict0 is None:
            from lib.registry.pars import preg
            dist_dict0 = preg.dist_dict0
        self.dist_dict0 = dist_dict0
        self.dist_dict = self.dist_dict0.dict

        self.dict = self.build()
        self.full_dict=self.build_full_dict()

    def build(self):
        from lib.registry.modConfs import build_brain_module_dict, build_aux_module_dict
        init_bdicts2, mbpredicts2, mbdicts2 = build_brain_module_dict()
        init_auxdicts2, aux_predicts2, aux_dicts2 = build_aux_module_dict()

        mdicts2 = dNl.NestDict({**mbdicts2, **aux_dicts2})
        mpredicts2 = dNl.NestDict({**mbpredicts2, **aux_predicts2})
        init_dicts2 = dNl.NestDict({**init_bdicts2, **init_auxdicts2})

        bkeys = list(init_bdicts2.keys())
        auxkeys = list(init_auxdicts2.keys())

        bd = {'init': init_bdicts2, 'pre': mbpredicts2, 'm': mbdicts2, 'keys': bkeys}
        auxd = {'init': init_auxdicts2, 'pre': aux_predicts2, 'm': aux_dicts2, 'keys': auxkeys}
        d = {'init': init_dicts2, 'pre': mpredicts2, 'm': mdicts2, 'keys': bkeys + auxkeys}

        dd = {'brain': bd, 'aux': auxd, 'model': d}
        return dNl.NestDict(dd)

    def get_mdict(self, mkey, mode='default'):
        if mkey is None or mkey not in self.dict.model.keys:
            raise ValueError('Module key must be one of larva-model configuration keys')
        else:
            if mkey in self.dict.brain.keys+['energetics']:
                return self.dict.model.m[mkey].mode[mode].args
            elif mkey in self.dict.aux.keys:
                return self.dict.model.m[mkey].args


    def generate_configuration(self, mdict, **kwargs):
        conf = dNl.NestDict()
        for d, p in mdict.items():
            if isinstance(p, param.Parameterized):
                conf[d] = p.v
            else:
                conf[d] = self.generate_configuration(mdict=p)
        conf=dNl.update_existingdict(conf,kwargs)
        # conf.update(kwargs)
        return conf

    def conf(self, mdict=None, mkey=None, mode=None, refID=None, **kwargs):
        if mdict is None:
            mdict = self.get_mdict(mkey, mode)
        conf0 = self.generate_configuration(mdict, **kwargs)
        if refID is not None and mkey == 'intermitter':
            conf0 = self.adapt_intermitter(refID=refID, mode=mode, conf=conf0)
        return dNl.NestDict(conf0)

    # def fit2refID(self, conf, mkey=None, refID=None):
    #     if refID is not None and mkey == 'intermitter':
    #         try:
    #             from lib.aux.sim_aux import get_sample_bout_distros0
    #             from lib.registry.pars import preg
    #             kkkws = {
    #                 'Im': conf,
    #                 'bout_distros': preg.loadConf(id=refID, conftype='Ref').bout_distros,
    #             }
    #             conf = get_sample_bout_distros0(**kkkws)
    #         except:
    #             pass
    #         # bout_distros = sample.bout_distros
    #     return conf

    def module(self, mkey, mode=None, refID=None, mkwargs={}, **kwargs):
        if mode is None:
            mdict = self.dict.brain.m[mkey]
        else:
            mdict = self.dict.brain.m[mkey].mode[mode]
        mkws = self.dict.brain.m[mkey].kwargs
        conf0 = self.conf(mdict=mdict.args, prefix=False, refID=refID, **kwargs)
        func = mdict.class_func
        mkws.update(mkwargs)
        m = func(**conf0, **mkws)
        return m

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
            if mkey == 'brain':
                continue
            if mdict is None:
                mc[mkey] = None
            else:
                mc[mkey] = self.conf(mdict)
                # mc[mkey] = self.conf(mdict)
        return mc

    def update_modelConf(self, mconf, mdict, **kwargs):
        conf0 = self.conf(mdict, prefix=True, **kwargs)
        return dNl.update_nestdict(mconf, conf0)

    def crossover(self, mdict, mdict2):

        for d, p in mdict.items():
            if random.random() < 0.5:
                p.v = mdict2[d].v
        return mdict



    def mutate(self, mdict, Pmut, Cmut):
        for d, p in mdict.items():
            p.mutate(Pmut, Cmut)
        return mdict

    def randomize(self, mdict):
        for d, p in mdict.items():
            p.randomize()

    # def initConf_old(self, init_mode, mdict, mconf0, **kwargs):
    #     if init_mode == 'model':
    #         conf = self.conf(mdict, prefix=True, **mconf0)
    #         # return mconf0
    #     elif init_mode == 'default':
    #         conf = self.conf(mdict, prefix=True)
    #         # return self.update_modelConf(mconf0, mdict,**kwargs)
    #     elif init_mode == 'random':
    #         self.randomize(mdict)
    #         conf = self.conf(mdict, prefix=True)
    #     return conf

    def initConf(self, init_mode, mdict, mConf0):
        if init_mode == 'model':
            mdict=self.update_mdict(mdict, dNl.flatten_dict(mConf0))
            # conf = self.conf(mdict, **dNl.flatten_dict(mConf0))
            # return mconf0
        elif init_mode == 'default':
            pass
            # conf = self.conf(mdict)
            # return self.update_modelConf(mconf0, mdict,**kwargs)
        elif init_mode == 'random':
            self.randomize(mdict)
        conf = self.conf(mdict)
        return conf,mdict

    def compile_pdict(self, dic):
        pdict = dNl.NestDict()
        for mkey, ds in dic.items():
            mdict = self.dict.brain.m[mkey]
            for d in ds:
                pdict[d] = mdict[d]
        return pdict

    def mIDbconf(self, mID=None, m=None):
        if m is None:
            from lib.conf.stored.conf import loadConf
            m = loadConf(mID, 'Model').brain
        mIDconf = dNl.NestDict()
        mIDconf.modules = m.modules
        for mkey, mdic in self.dict.brain.m.items():
            if m.modules[mkey]:
                mmdic = m[f'{mkey}_params']
                mdic = self.update_mdict(mdic, mmdic)
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
        for mkey, mdic in self.dict.aux.m.items():
            mmdic = m[mkey]
            mc[mkey] = self.update_mdict(mdic, mmdic)
        return mc

    def update_mdict(self, mdict, mmdic):
        if mmdic is None:
            return None
        else:
            for d, p in mdict.items():
                if isinstance(p, param.Parameterized):
                    new_v = mmdic[d] if d in mmdic.keys() else None
                    if type(new_v) == list:
                        if p.parclass == param.Range:
                            new_v = tuple(new_v)
                    p.v = new_v
                else:
                    self.update_mdict(mdict=mdict[d], mmdic=mmdic[d])
            return mdict

    def mIDtable_data(self, mID, columns=['parameter', 'symbol', 'value', 'unit'], **kwargs):
        mConf = self.mIDconf(mID, **kwargs)
        m = self.multiconf(mConf, **kwargs)
        data = []

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
                    # print(p)
                    # print(isinstance(p, dict))
                    # print(k, p == mConf_dic[n])
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

            if k in ['crawler', 'turner', 'interference']:
                vals = dvalid[k][dic.mode]
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

        for k in self.dict.model.keys:
            if k in self.dict.brain.keys:
                dic = m['brain'][f'{k}_params']
                dic0 = mConf['brain'][f'{k}_params']





            elif k in self.dict.aux.keys:
                dic = m[k]
                dic0 = mConf[k]
            if dic is None:
                continue
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
        D = self.dict.model.m
        for k in ['crawler', 'turner', 'interference', 'feeder', 'intermitter']:
            if conf.modules[k]:
                m = conf[f'{k}_params']
                if k == 'feeder':
                    mode = 'default'
                else:
                    mode = m.mode
                kws = {kw: getattr(L, kw) for kw in D[k].kwargs.keys()}
                M = D[k].mode[mode].class_func(**m, **kws)
                if k == 'intermitter':
                    M.disinhibit_locomotion(L)
                if k == 'crawler':
                    M.mode = m.mode
            else:
                M = None
            setattr(L, k, M)
        return L

    def init_brain(self, conf, B):
        D = self.dict.model.m
        for k in ['olfactor', 'toucher', 'windsensor', 'thermosensor']:
            if conf.modules[k]:
                m = conf[f'{k}_params']
                if k == 'windsensor':
                    m.gain_dict = {'windsensor': 1.0}
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
            mode = mm['modality']
            kws = {kw: getattr(B, kw) for kw in D['memory'].kwargs.keys()}
            if mode == 'olfaction' and B.olfactor:
                mm.gain = B.olfactor.gain
                B.memory = D['memory'].mode[mode].class_func(**mm, **kws)
            elif mode == 'touch' and B.toucher:
                mm.gain = B.toucher.gain
                B.touch_memory = D['memory'].mode[mode].class_func(**mm, **kws)
        return B

    def locoConf(self, module_modes=None, modkws={}):

        if module_modes is None:
            module_modes = {'crawler': 'realistic',
                            'turner': 'neural',
                            'interference': 'default',
                            'intermitter': 'default'}

        conf = dNl.NestDict()
        modules = dNl.NestDict()

        for mkey, mode in module_modes.items():
            mlongkey = f'{mkey}_params'
            if mode is None:
                modules[mkey] = False
                conf[mlongkey] = None
            else:
                modules[mkey] = True
                mdict = self.dict.brain.m[mkey].mode[mode].args
                if mkey in modkws.keys():
                    mkws = modkws[mkey]
                else:
                    mkws = {}
                conf[mlongkey] = self.generate_configuration(mdict, **mkws)

        conf.modules = modules
        return conf

    def brainConf(self, modes=None, modkws={}, nengo=False):

        if modes is None:
            modes = {'crawler': 'realistic',
                     'turner': 'neural',
                     'interference': 'phasic',
                     'intermitter': 'default'}

        conf = dNl.NestDict()
        modules = dNl.NestDict()

        for mkey in self.dict.brain.keys:
            mlongkey = f'{mkey}_params'
            if mkey not in modes.keys():
                modules[mkey] = False
                conf[mlongkey] = None
            else:
                mode = modes[mkey]
                modules[mkey] = True
                mdict = self.dict.brain.m[mkey].mode[mode].args
                if mkey in modkws.keys():
                    mkws = modkws[mkey]
                else:
                    mkws = {}
                conf[mlongkey] = self.generate_configuration(mdict, **mkws)
                conf[mlongkey]['mode'] = mode

        conf.modules = modules
        conf.nengo = nengo
        return conf

    def larvaConf(self, modes=None, energetics=None, auxkws={}, modkws={}, nengo=False, mID=None):
        bconf = self.brainConf(modes, modkws, nengo=nengo)

        conf = dNl.NestDict()
        conf.brain = bconf
        # for mkey in self.dict.brain.keys:

        for auxkey in self.dict.aux.keys:
            if auxkey in auxkws.keys():
                mkws = auxkws[auxkey]
            else:
                mkws = {}
            if auxkey == 'energetics':
                if energetics is None:
                    conf[auxkey] = None
                    continue
                else :
                    for m,mdic in self.dict.aux.m[auxkey].mode.items():
                        mdict = mdic.args
                        conf[auxkey][m] = self.generate_configuration(mdict, **mkws[m])
            elif auxkey == 'sensorimotor':
                continue
            else :
                mdict = self.dict.aux.m[auxkey].args
                conf[auxkey] = self.generate_configuration(mdict, **mkws)

        #  TODO thsi
        null_Box2D_params = {
            'joint_types': {
                'friction': {'N': 0, 'args': {}},
                'revolute': {'N': 0, 'args': {}},
                'distance': {'N': 0, 'args': {}}
            }
        }
        conf.Box2D_params = null_Box2D_params

        #
        # T0=

        self.saveConf(conf, mID)

        return conf

    def newConf(self, mID0,mID=None,  kwargs={}):
        T0 = dNl.NestDict(copy.deepcopy(self.loadConf(mID=mID0)))
        T = dNl.update_nestdict(T0, kwargs)
        if mID is not None :
            self.saveConf(conf=T, mID=mID)
        return T


    def baseConfs(self):
        kws= {'modkws': {'interference': {'attenuation': 0.1, 'attenuation_max': 0.6}}}
        self.larvaConf(mID='loco_default',**kws)
        self.larvaConf(mID='RE_NEU_PHI_NENGO', modes = {'crawler': 'realistic','turner': 'neural','interference': 'phasic','intermitter': 'nengo'}, nengo=True,**kws)
        self.larvaConf(mID='RE_NEU_PHI_BR', modes = {'crawler': 'realistic','turner': 'neural','interference': 'phasic','intermitter': 'branch'},**kws)
        self.larvaConf(mID='RE_NEU_PHI_DEF', modes = {'crawler': 'realistic','turner': 'neural','interference': 'phasic','intermitter': 'default'},**kws)
        self.larvaConf(mID='RE_NEU_SQ_DEF', modes = {'crawler': 'realistic','turner': 'neural','interference': 'square','intermitter': 'default'},**kws)
        self.larvaConf(mID='RE_NEU_DEF_DEF', modes = {'crawler': 'realistic','turner': 'neural','interference': 'default','intermitter': 'default'})
        self.larvaConf(mID='SQ_NEU_PHI_DEF', modes = {'crawler': 'square','turner': 'neural','interference': 'phasic','intermitter': 'default'},**kws)
        self.larvaConf(mID='SQ_NEU_SQ_DEF', modes = {'crawler': 'square','turner': 'neural','interference': 'square','intermitter': 'default'},**kws)
        self.larvaConf(mID='SQ_NEU_DEF_DEF', modes = {'crawler': 'square','turner': 'neural','interference': 'default','intermitter': 'default'})
        self.larvaConf(mID='CON_NEU_PHI_DEF', modes = {'crawler': 'constant','turner': 'neural','interference': 'phasic','intermitter': 'default'},**kws)
        self.larvaConf(mID='CON_NEU_SQ_DEF', modes = {'crawler': 'constant','turner': 'neural','interference': 'square','intermitter': 'default'},**kws)
        self.larvaConf(mID='CON_NEU_DEF_DEF', modes = {'crawler': 'constant','turner': 'neural','interference': 'default','intermitter': 'default'})
        self.larvaConf(mID='CON_SIN_PHI_DEF',modes={'crawler': 'constant', 'turner': 'sinusoidal', 'interference': 'phasic','intermitter': 'default'}, **kws)
        self.larvaConf(mID='CON_SIN_SQ_DEF', modes={'crawler': 'constant', 'turner': 'sinusoidal', 'interference': 'square','intermitter': 'default'}, **kws)
        self.larvaConf(mID='CON_SIN_DEF_DEF', modes={'crawler': 'constant', 'turner': 'sinusoidal', 'interference': 'default','intermitter': 'default'})

        self.larvaConf(mID='RE_SIN_PHI_DEF', modes = {'crawler': 'realistic','turner': 'sinusoidal','interference': 'phasic','intermitter': 'default'},**kws)
        self.larvaConf(mID='RE_SIN_SQ_DEF', modes = {'crawler': 'realistic','turner': 'sinusoidal','interference': 'square','intermitter': 'default'},**kws)
        self.larvaConf(mID='RE_SIN_DEF_DEF', modes = {'crawler': 'realistic','turner': 'sinusoidal','interference': 'default','intermitter': 'default'})

        kws2 = {'modkws': {'interference': {'attenuation': 0.0}}}
        self.larvaConf(mID='Levy', modes={'crawler': 'constant', 'turner': 'sinusoidal', 'interference': 'default', 'intermitter': 'default'},**kws2)
        self.larvaConf(mID='NEU_Levy', modes={'crawler': 'constant', 'turner': 'neural', 'interference': 'default', 'intermitter': 'default'},**kws2)
        self.larvaConf(mID='NEU_Levy_continuous', modes={'crawler': 'constant', 'turner': 'neural', 'interference': 'default'},**kws2)

        self.larvaConf(mID='CON_SIN',modes={'crawler': 'constant', 'turner': 'sinusoidal'})

        olf_pars1=self.generate_configuration(self.dict.brain.m['olfactor'].mode['default'].args, odor_dict = {'Odor': {'mean': 150.0, 'std': 0.0}})
        olf_pars2=self.generate_configuration(self.dict.brain.m['olfactor'].mode['default'].args, odor_dict = {'CS': {'mean': 150.0, 'std': 0.0}, 'UCS': {'mean': 0.0, 'std': 0.0}})
        kwargs1 = {'brain.modules.olfactor': True, 'brain.olfactor_params': olf_pars1}
        kwargs2 = {'brain.modules.olfactor': True, 'brain.olfactor_params': olf_pars2}
        for Tmod in ['NEU', 'SIN'] :
            for Ifmod in ['PHI', 'SQ', 'DEF'] :
                mID0=f'RE_{Tmod}_{Ifmod}_DEF'
                mID1 = f'{mID0}_nav'
                self.newConf(mID=mID1,mID0=mID0,kwargs=kwargs1)
                mID1br = f'{mID1}_brute'
                self.newConf(mID=mID1br,mID0=mID1,kwargs={'brain.olfactor_params.brute_force': True})
                mID2 = f'{mID0}_nav_x2'
                self.newConf(mID=mID2, mID0=mID0, kwargs=kwargs2)
                mID2br = f'{mID2}_brute'
                self.newConf(mID=mID2br, mID0=mID2, kwargs={'brain.olfactor_params.brute_force': True})


        for mID0 in ['Levy','NEU_Levy','NEU_Levy_continuous', 'CON_SIN']:
            mID1 = f'{mID0}_nav'
            self.newConf(mID=mID1, mID0=mID0, kwargs=kwargs1)
            mID2 = f'{mID0}_nav_x2'
            self.newConf(mID=mID2, mID0=mID0, kwargs=kwargs2)

        sm_pars=self.generate_configuration(self.dict.aux.m['sensorimotor'].mode['default'].args)
        self.newConf(mID='obstacle_avoider', mID0='RE_NEU_PHI_DEF_nav', kwargs={'sensorimotor' : sm_pars})






    def saveConf(self, conf, mID=None):
        if mID is not None:
            from lib.conf.stored.conf import saveConf
            saveConf(conf, 'Model', mID)

    def loadConf(self, mID=None):
        if mID is not None:
            from lib.conf.stored.conf import loadConf
            return loadConf(mID, 'Model')

    def storedConf(self, **kwargs):
        from lib.conf.stored.conf import kConfDict
        return kConfDict('Model',**kwargs)

    def build_full_dict(self):
        D=self.dict
        def register(dic, k0, full_dic):
            for k, p in dic.items():
                kk = f'{k0}.{k}'
                if isinstance(p, param.Parameterized):
                    full_dic[kk] = p
                else:
                    print(kk)
                    register(p, kk, full_dic)

        full_dic = dNl.NestDict()
        for aux_key in D.aux.keys:
            if aux_key in ['energetics', 'sensorimotor']:
                continue
            aux_dic = D.aux.m[aux_key]
            register(aux_dic.args, aux_key, full_dic)
        for aux_key in ['energetics', 'sensorimotor']:
            for m, mdic in D.aux.m[aux_key].mode.items():
                k0 = f'{aux_key}.{m}'
                register(mdic.args, k0, full_dic)

        for bkey in D.brain.keys:
            bkey0 = f'brain.{bkey}_params'
            bdic = D.brain.m[bkey]
            for mod in bdic.mode.keys():
                mdic = bdic.mode[mod].args
                register(mdic, bkey0, full_dic)

        return full_dic

    def diff_df(self, mIDs, ms=None):
        dic = {}
        if ms is None :
            ms=[self.loadConf(mID) for mID in mIDs]
        ms=[dNl.flatten_dict(m) for m in ms]
        ks=dNl.unique_list(dNl.flatten_list([list(m.keys()) for m in ms]))

        for k in ks :
            entry = {mID: m[k] if k in m.keys() else None for mID, m in zip(mIDs, ms)}
            l=list(entry.values())
            if all([a==l[0] for a in l]) :
                continue
            else :
                if k in self.full_dict.keys():
                    k0=self.full_dict[k].disp
                else :
                    k0=k.split('.')[-1]
                dic[k0] = entry
        df = pd.DataFrame.from_dict(dic).T
        return df

    def adapt_crawler(self,refID=None,e=None,mode='realistic',average = True):
        if e is None :
            from lib.registry.pars import preg
            d = preg.loadRef(refID)
            d.load(step=False)
            e = d.endpoint_data

        mdict = dd.dict.model.m['crawler'].mode[mode].args
        crawler_conf = dNl.NestDict({'mode': mode})
        for d, p in mdict.items():
            if isinstance(p, param.Parameterized):
                try:
                    crawler_conf[d] = epar(e, par=p.codename, average=average)
                except:
                    pass
            else:
                raise
        return crawler_conf

    def adapt_intermitter(self,refID=None,e=None,c=None,mode='default', conf=None):
        if e is None or c is None:
            from lib.registry.pars import preg
            d = preg.loadRef(refID)
            d.load(step=False)
            e,c = d.endpoint_data, d.config

        if conf is None :
            mdict = dd.dict.model.m['intermitter'].mode[mode].args
            conf = self.generate_configuration(mdict)
        conf.stridechain_dist = c.bout_distros.run_count
        try:
            ll1, ll2 = conf.stridechain_dist.range
            conf.stridechain_dist.range = (int(ll1), int(ll2))
        except:
            pass

        conf.run_dist = c.bout_distros.run_dur
        try:
            ll1, ll2 = conf.run_dist.range
            conf.run_dist.range = (np.round(ll1, 2), np.round(ll2, 2))
        except:
            pass
        conf.pause_dist = c.bout_distros.pause_dur
        try:
            ll1, ll2 = conf.pause_dist.range
            conf.pause_dist.range = (np.round(ll1, 2), np.round(ll2, 2))
        except:
            pass
        conf.crawl_freq = epar(e, 'fsv', average=True)
        conf.mode = mode
        return conf

    def adapt_mID(self,refID, mID0,mID=None,e=None,c=None):
        if e is None or c is None:
            from lib.registry.pars import preg
            d = preg.loadRef(refID)
            d.load(step=False)
            e, c = d.endpoint_data, d.config

        m0=self.loadConf(mID0)
        m0.brain.crawler_params=self.adapt_crawler(e=e,mode=m0.brain.crawler_params.mode)
        m0.brain.intermitter_params=self.adapt_intermitter(e=e,c=c,mode=m0.brain.intermitter_params.mode, conf=m0.brain.intermitter_params)
        m0.body.initial_length=epar(e, 'l', average=True,Nround=5)

        from lib.eval.model_fit import Calibration, calibrate_interference
        C=Calibration(refID=refID,turner_mode=m0.brain.turner_params.mode, physics_keys=None)
        C.run()
        m0.brain.turner_params.update(C.best.turner)

        if mID is None :
            mID=f'{mID0}_fitted'

        self.saveConf(conf=m0, mID=mID)
        entry = calibrate_interference(mID=mID, refID=refID)
        # m0=entry[mID]
        # if 'modelConfs' not in c.keys():
        #     c.modelConfs = dNl.NestDict({'average': {}, 'variable': {}, 'individual': {}})
        # c.modelConfs.average[mID] = m
        # d.save_config(add_reference=True)
        return entry

    def adapt_6mIDs(self,refID, e=None,c=None):
        if e is None or c is None:
            from lib.registry.pars import preg
            d = preg.loadRef(refID)
            d.load(step=False)
            e, c = d.endpoint_data, d.config
        entries={}
        mIDs=[]
        for Tmod in ['NEU', 'SIN']:
            for Ifmod in ['PHI', 'SQ', 'DEF']:
                mID0 = f'RE_{Tmod}_{Ifmod}_DEF'
                mID=f'{Ifmod}on{Tmod}'
                entry=self.adapt_mID(refID=refID, mID0=mID0,mID=mID,e=e,c=c)
                entries.update(entry)
                mIDs.append(mID)
        return entries, mIDs
        # if 'modelConfs' not in c.keys():
        #     c.modelConfs = dNl.NestDict({'average': {}, 'variable': {}, 'individual': {}})
        # c.modelConfs.average=entries
        # d.save_config(add_reference=True)
        # d.store_model_graphs(mIDs=mIDs)
        # return entries

    def add_var_mIDs(self, refID, e=None,c=None, mID0s=None, sample_ks=None):
        if e is None or c is None:
            from lib.registry.pars import preg
            d = preg.loadRef(refID)
            d.load(step=False)
            e, c = d.endpoint_data, d.config

        if mID0s is None :
            mID0s=list(c.modelConfs.average.keys())
        if sample_ks is None :
            sample_ks=[
                'brain.crawler_params.stride_dst_mean',
                'brain.crawler_params.stride_dst_std',
                'brain.crawler_params.max_scaled_vel',
                'brain.crawler_params.max_vel_phase',
                'brain.crawler_params.initial_freq',
            ]
        kwargs={k : 'sample' for k in sample_ks}
        mIDs=[]
        entries={}
        for mID0 in mID0s :
            mID=f'{mID0}_var'
            m=self.newConf(mID0=mID0, mID=mID, kwargs=kwargs)
            mIDs.append(mID)
            entries[mID]=m

        return entries, mIDs

    def space_dict(self, mkeys, mID0):
        # M = preg.larva_conf_dict2
        m = self.loadConf(mID0)
        mF = dNl.flatten_dict(m)
        dic = {}
        for mkey in mkeys:
            d0 = self.dict.model.init[mkey]
            d00 = self.dict.model.m[mkey]
            if f'{d0.pref}mode' in mF.keys() :
                mod_v = mF[f'{d0.pref}mode']
            else :
                mod_v = 'default'
            var_ks = d0.mode[mod_v].variable
            for k in var_ks:
                k0 = f'{d0.pref}{k}'
                dic[k0] = d00.mode[mod_v].args[k]
                dic[k0].v = mF[k0]

        return dNl.NestDict(dic)


    def to_string(self,mdict):
        s=''
        for k,p in mdict.items():
            s=s+f'{p.d} : {p.v}'
        return s



def epar(e, k=None, par=None, average=True, Nround=2):
    from lib.registry.pars import preg
    if par is None:
        D = preg.dict
        par = D[k].d
    vs = e[par]
    if average:
        return np.round(vs.median(), Nround)
    else:
        return vs

if __name__ == '__main__':
    from lib.registry.pars import preg
    refID = 'None.150controls'
    d = preg.loadRef(refID)
    d.load(step=False)
    # d.modelConf_analysis()

    dd = LarvaConfDict2()
    # raise
    c=d.config
    # print(c.Nticks * c.dt / 60)
    # raise
    # for mID,m in c.modelConfs.variable.items() :
    #     print(mID, m.brain.crawler_params)


    mIDs0=list(c.modelConfs.average.keys())
    diff_df_avg = dd.diff_df(mIDs=mIDs0)

    _=preg.graph_dict.dict['mpl'](data=diff_df_avg, show=True, font_size=20,bbox=[0.15, 0, 0.8, 1], figsize=(50,8))

    raise
    mIDs1=[f'{a}_var' for a in mIDs0]
    mIDs=mIDs0+mIDs1
    evrun=d.eval_model_graphs(mIDs=mIDs,norm_modes=['raw', 'minmax'], id='3mIDs_meanVSvar', N=50, dur=c.duration/60)

    # raise
    # from lib.eval.evaluation import EvalRun
    # evrun = EvalRun(refID=refID, id='6mIDs', modelIDs=mIDs, dataset_ids=mIDs, N=4,save_to=d.dir_dict.evaluation,
    #                 bout_annotation=True, enrichment=True, show=False, offline=False)
    #
    # evrun.run(video=False)
    # evrun.eval()
    # evrun.plot_models()
    # evrun.plot_results()

    # d.store_model_graphs(mIDs=mIDs)

    # print()
    # raise
    # dd = LarvaConfDict2()
    # for mID in dd.storedConf() :
    #     mm=dd.loadConf(mID)
    #     m=mm.brain.crawler_params
    #     print(m)
    #     if m is not None :
    #         for k,v in m.items() :
    #             if v=='fit' :
    #                 print(k,v)

    # dd.add_var_mIDs(refID)
    # m=dd.loadConf(mID)

    # print(m.brain.intermitter_params)

    # dd.adapt_mID(mID0='RE_NEU_PHI_DEF',mID='RE_NEU_PHI_DEF_fitted',refID = 'None.150controls')

    raise

    from lib.registry.pars import preg
    refID = 'None.150controls'
    d=preg.loadRef(refID)
    d.load(step=False)
    e=d.endpoint_data
    average = True
    mdict0 = dd.dict.model.m['crawler'].mode
    for m,mdic in mdict0.items():


    # [mode].args
        conf0 = dNl.NestDict({'mode': m})
        for d, p in mdic.args.items():
            # print(d, p.codename)
            if isinstance(p, param.Parameterized):
                try :
                    conf0[d] = epar(e, par=p.codename, average=average)
                except :
                    pass
            else:
                raise

        print(conf0)
        print()
    # print(dd.dict.aux.m.energetics.args)
    # raise







    # dd.storeConfs()
    # m = dd.larvaConf(mID='loco_default')

    # print(dd.storedConf())
    # print(dd.dict.aux.keys)
    # d=dd.dict
    # from lib.conf.stored.conf import kConfDict
    # from lib.conf.stored.conf import loadConf,loadRef
    #
    # refID = 'None.150controls'
    # d = loadRef(refID)
