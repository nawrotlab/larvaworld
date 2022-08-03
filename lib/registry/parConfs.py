import copy
import random
from typing import Tuple

import numpy as np
import pandas as pd
import param
from lib.aux import dictsNlists as dNl
from lib.aux.par_aux import sub
from lib.registry.pars import preg

from lib.registry.units import ureg




class LarvaConfDict:
    def __init__(self,load=False, save=False):

        preg.vprint('started LarvaConfDict', 2)
        from lib.registry.modConfs import build_LarvaConfDict, build_confdicts0
        self.dict_path = preg.paths['LarvaConfDict']
        if not load:

            self.dict0 = build_confdicts0()
            if save:
                dNl.save_dict(self.dict0, self.dict_path)
        else:
            self.dict0 = dNl.load_dict(self.dict_path)

        self.dict = build_LarvaConfDict(self.dict0)
        self.full_dict = self.build_full_dict(D = self.dict)

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

        preg.vprint('completed LarvaConfDict',2)

        # print('Completed LarvaConfDict')

    def get_mdict(self, mkey, mode='default'):
        if mkey is None or mkey not in self.dict.model.keys:
            raise ValueError('Module key must be one of larva-model configuration keys')
        else:
            if mkey in self.dict.brain.keys + ['energetics']:
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
        conf = dNl.update_existingdict(conf, kwargs)
        # conf.update(kwargs)
        return conf

    def conf(self, mdict=None, mkey=None, mode=None, refID=None, **kwargs):
        if mdict is None:
            mdict = self.get_mdict(mkey, mode)
        conf0 = self.generate_configuration(mdict, **kwargs)
        if refID is not None and mkey == 'intermitter':
            conf0 = self.adapt_intermitter(refID=refID, mode=mode, conf=conf0)
        return dNl.NestDict(conf0)

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

    def mutate(self, mdict, Pmut, Cmut):
        for d, p in mdict.items():
            p.mutate(Pmut, Cmut)
        # return mdict

    def randomize(self, mdict):
        for d, p in mdict.items():
            p.randomize()

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
            # from lib.conf.stored.conf import loadConf
            m = self.loadConf(mID)

        mc = dNl.NestDict()
        mc.brain = self.mIDbconf(self, m=m.brain)
        for mkey, mdic in self.dict.aux.m.items():
            mmdic = m[mkey]
            mc[mkey] = self.update_mdict(mdic, mmdic)
        return mc

    def mIDtable_data(self, mID, columns):
        def gen_rows2(var_mdict, parent, columns, data):
            for k, p in var_mdict.items():
                if isinstance(p, param.Parameterized):
                    ddd = [getattr(p, pname) for pname in columns]
                    row = [parent] + ddd
                    data.append(row)

        m = self.loadConf(mID)
        mF = dNl.flatten_dict(m)
        data = []
        for mkey in self.dict.brain.keys:
            if m.brain.modules[mkey]:
                d0 = self.dict.model.init[mkey]
                if f'{d0.pref}mode' in mF.keys():
                    mod_v = mF[f'{d0.pref}mode']
                else:
                    mod_v = 'default'

                if mkey == 'intermitter':
                    run_mode = m.brain[f'{mkey}_params']['run_mode']
                    var_ks = d0.mode[mod_v].variable
                    for var_k in var_ks:
                        if var_k == 'run_dist' and run_mode == 'stridechain':
                            continue
                        if var_k == 'stridechain_dist' and run_mode == 'run':
                            continue
                        v = m.brain[f'{mkey}_params'][var_k]
                        if v is not None:
                            if v.name is not None:
                                vs1, vs2 = preg.dist_dict.get_dist(k=var_k, k0=mkey, v=v, return_tabrows=True)
                                data.append(vs1)
                                data.append(vs2)
                else:
                    var_mdict = self.variable_mdict(mkey, mode=mod_v)
                    var_mdict = self.update_mdict(var_mdict, m.brain[f'{mkey}_params'])
                    gen_rows2(var_mdict, mkey, columns, data)
        for aux_key in self.dict.aux.keys:
            if aux_key not in ['energetics', 'sensorimotor']:
                var_ks = self.dict.aux.init[aux_key].variable
                var_mdict = dNl.NestDict({k: self.dict.aux.m[aux_key].args[k] for k in var_ks})
                var_mdict = self.update_mdict(var_mdict, m[aux_key])
                gen_rows2(var_mdict, aux_key, columns, data)
        if m['energetics']:
            for mod, dic in self.dict.aux.init['energetics'].mode.items():
                var_ks = dic.variable
                var_mdict = dNl.NestDict({k: self.dict.aux.m['energetics'].mode[mod].args[k] for k in var_ks})
                var_mdict = self.update_mdict(var_mdict, m['energetics'].mod)
                gen_rows2(var_mdict, f'energetics.{mod}', columns, data)
        if 'sensorimotor' in m.keys():
            for mod, dic in self.dict.aux.init['sensorimotor'].mode.items():
                var_ks = dic.variable
                var_mdict = dNl.NestDict({k: self.dict.aux.m['sensorimotor'].mode[mod].args[k] for k in var_ks})
                var_mdict = self.update_mdict(var_mdict, m['sensorimotor'])
                gen_rows2(var_mdict, 'sensorimotor', columns, data)
        df = pd.DataFrame(data, columns=['field'] + columns)
        df.set_index(['field'], inplace=True)
        return df

    def mIDtable_data2(self, mID, columns):

        mConf = self.mIDconf(mID)
        # m = self.loadConf(mID)
        m = self.multiconf(mConf)
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
                            vs1, vs2 = preg.dist_dict.get_dist(k=kkk, k0=k, v=dic[kkk], return_tabrows=True)
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
        from lib.aux.data_aux import arrange_index_labels
        df.index = arrange_index_labels(df.index)
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
                else:
                    for m, mdic in self.dict.aux.m[auxkey].mode.items():
                        mdict = mdic.args
                        conf[auxkey][m] = self.generate_configuration(mdict, **mkws[m])
            elif auxkey == 'sensorimotor':
                continue
            else:
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

        # self.saveConf(conf, mID)

        return {mID : conf}

    def newConf(self, m0=None,mID0=None, mID=None, kwargs={}):
        if m0 is None :
            m0=self.loadConf(mID=mID0)
        T0 = dNl.copyDict(m0)
        conf = dNl.update_nestdict(T0, kwargs)
        if mID is not None:
            self.saveConf(conf=conf, mID=mID)
        return conf

    def baseConfs(self):
        mod_dict = {'realistic': 'RE', 'square': 'SQ', 'gaussian': 'GAU', 'constant': 'CON',
                    'default': 'DEF', 'neural': 'NEU', 'sinusoidal': 'SIN', 'nengo': 'NENGO', 'phasic': 'PHI',
                    'branch': 'BR'}
        kws = {'modkws': {'interference': {'attenuation': 0.1, 'attenuation_max': 0.6}}}
        entries={}
        for Cmod in ['realistic', 'square', 'gaussian', 'constant']:
            for Tmod in ['neural', 'sinusoidal', 'constant']:
                for Ifmod in ['phasic', 'square', 'default']:
                    for IMmod in ['nengo', 'branch', 'default']:
                        kkws = {
                            'mID': f'{mod_dict[Cmod]}_{mod_dict[Tmod]}_{mod_dict[Ifmod]}_{mod_dict[IMmod]}',
                            'modes': {'crawler': Cmod, 'turner': Tmod, 'interference': Ifmod, 'intermitter': IMmod},
                            'nengo': True if IMmod == 'nengo' else False,

                        }
                        if Ifmod != 'default':
                            kkws.update(**kws)
                        entries.update(self.larvaConf(**kkws))
        e1=self.larvaConf(mID='loco_default', **kws)
        kws2 = {'modkws': {'interference': {'attenuation': 0.0}}}
        e2=self.larvaConf(mID='Levy', modes={'crawler': 'constant', 'turner': 'sinusoidal', 'interference': 'default',
                                          'intermitter': 'default'}, **kws2)
        e3=self.larvaConf(mID='NEU_Levy', modes={'crawler': 'constant', 'turner': 'neural', 'interference': 'default',
                                              'intermitter': 'default'}, **kws2)
        e4=self.larvaConf(mID='NEU_Levy_continuous',
                       modes={'crawler': 'constant', 'turner': 'neural', 'interference': 'default'}, **kws2)

        e5=self.larvaConf(mID='CON_SIN', modes={'crawler': 'constant', 'turner': 'sinusoidal'})
        entries.update(**e1,**e2,**e3,**e4,**e5)
        mID0dic = {}
        # m0s=[]
        for Tmod in ['NEU', 'SIN']:
            for Ifmod in ['PHI', 'SQ', 'DEF']:
                mID0 = f'RE_{Tmod}_{Ifmod}_DEF'
                mID0dic[mID0]=entries[mID0]
                # mID0s.append(mID0)
                # m0s.append(entries[mID0])
                for mm in [f'{mID0}_avg', f'{mID0}_var', f'{mID0}_var2']:
                    if mm in self.storedConf():
                        mID0dic[mm] = self.loadConf(mm)
                        # mID0s.append(mm)
                        # m0s.append(self.loadConf(mm))

        olf_pars1 = self.generate_configuration(self.dict.brain.m['olfactor'].mode['default'].args,
                                                odor_dict={'Odor': {'mean': 150.0, 'std': 0.0}})
        olf_pars2 = self.generate_configuration(self.dict.brain.m['olfactor'].mode['default'].args,
                                                odor_dict={'CS': {'mean': 150.0, 'std': 0.0},
                                                           'UCS': {'mean': 0.0, 'std': 0.0}})
        kwargs1 = {'brain.modules.olfactor': True, 'brain.olfactor_params': olf_pars1}
        kwargs2 = {'brain.modules.olfactor': True, 'brain.olfactor_params': olf_pars2}

        # for m0 in m0s:
        for mID0,m0 in mID0dic.items():
            mID1 = f'{mID0}_nav'
            entries[mID1]=self.newConf(m0=m0, kwargs=kwargs1)
            mID1br = f'{mID1}_brute'
            entries[mID1br] = self.newConf(m0=entries[mID1], kwargs={'brain.olfactor_params.brute_force': True})
            mID2 = f'{mID0}_nav_x2'
            entries[mID2] = self.newConf(m0=m0, kwargs=kwargs2)
            mID2br = f'{mID2}_brute'
            entries[mID2br] = self.newConf(m0=entries[mID2], kwargs={'brain.olfactor_params.brute_force': True})
        entries['explorer'] =self.newConf(m0=entries['loco_default'], kwargs={})
        entries['navigator'] =self.newConf(m0=entries['explorer'], kwargs=kwargs1)
        for mID0 in ['Levy', 'NEU_Levy', 'NEU_Levy_continuous', 'CON_SIN']:
            entries[f'{mID0}_nav']=self.newConf(m0=entries[mID0], kwargs=kwargs1)
            entries[f'{mID0}_nav_x2']=self.newConf(m0=entries[mID0], kwargs=kwargs2)

        sm_pars = self.generate_configuration(self.dict.aux.m['sensorimotor'].mode['default'].args)
        entries['obstacle_avoider']=self.newConf(m0=entries['RE_NEU_PHI_DEF_nav'], kwargs={'sensorimotor': sm_pars})
        return entries


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
        return kConfDict('Model', **kwargs)

    def build_full_dict(self, D):


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

    def diff_df(self, mIDs, ms=None, dIDs=None):
        dic = {}
        if dIDs is None:
            dIDs = mIDs
        if ms is None:
            ms = [self.loadConf(mID) for mID in mIDs]
        ms = [dNl.flatten_dict(m) for m in ms]
        ks = dNl.unique_list(dNl.flatten_list([list(m.keys()) for m in ms]))

        for k in ks:
            entry = {dID: m[k] if k in m.keys() else None for dID, m in zip(dIDs, ms)}
            l = list(entry.values())
            if all([a == l[0] for a in l]):
                continue
            else:
                if k in self.full_dict.keys():
                    k0 = self.full_dict[k].disp
                else:
                    k0 = k.split('.')[-1]
                k00=k.split('.')[0]
                if k00=='brain' :
                    k01=k.split('.')[1]
                    k00=k01.split('_')[0]
                entry['field']=k00
                dic[k0] = entry
        df = pd.DataFrame.from_dict(dic).T
        df.index = df.index.set_names(['parameter'])
        # df=df.reset_index().rename(columns={df.index.name: 'parameter'})
        df.reset_index(drop=False,inplace=True)
        df.set_index(['field'], inplace=True)
        df.sort_index(inplace=True)


        # print(df.index.values)
        # raise
        row_colors = [None] + [self.mcolor[ii] for ii in df.index.values]
        from lib.aux.data_aux import arrange_index_labels
        df.index = arrange_index_labels(df.index)
        # df.index = df.index.set_names(['field'])
        # df.reset_index(drop=False, inplace=True)
        # df.set_index(['field','parameter'], inplace=True)



        return df,row_colors

    def adapt_crawler(self, refID=None, e=None, mode='realistic', average=True):
        if e is None:
            d = preg.loadRef(refID)
            d.load(step=False)
            e = d.endpoint_data

        mdict = self.dict.model.m['crawler'].mode[mode].args
        crawler_conf = dNl.NestDict({'mode': mode})
        for d, p in mdict.items():
            # print(d, p.codename)
            if isinstance(p, param.Parameterized):
                try:
                    crawler_conf[d] = epar(e, par=p.codename, average=average)
                except:
                    pass
            else:
                raise
        return crawler_conf

    def adapt_intermitter(self, refID=None, e=None, c=None, mode='default', conf=None):
        if e is None or c is None:
            d = preg.loadRef(refID)
            d.load(step=False)
            e, c = d.endpoint_data, d.config

        if conf is None:
            mdict = self.dict.model.m['intermitter'].mode[mode].args
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

    def adapt_mID(self, refID, mID0, mID=None, space_mkeys=['turner', 'interference'], save_to=None, e=None, c=None,
                  **kwargs):
        if mID is None:
            mID = f'{mID0}_fitted'
        print(f'Adapting {mID0} on {refID} as {mID} fitting {space_mkeys} modules')
        if e is None or c is None:
            d = preg.loadRef(refID)
            d.load(step=False)
            e, c = d.endpoint_data, d.config
        if save_to is None:
            save_to = c.dir_dict.GAoptimization,
        m0 = self.loadConf(mID0)
        if 'crawler' not in space_mkeys:
            m0.brain.crawler_params = self.adapt_crawler(e=e, mode=m0.brain.crawler_params.mode)
            # print(m0.brain.crawler_params)
        if 'intermitter' not in space_mkeys:
            m0.brain.intermitter_params = self.adapt_intermitter(e=e, c=c, mode=m0.brain.intermitter_params.mode,
                                                                 conf=m0.brain.intermitter_params)
        m0.body.initial_length = epar(e, 'l', average=True, Nround=5)

        self.saveConf(conf=m0, mID=mID)

        from lib.sim.eval.model_fit import optimize_mID
        entry = optimize_mID(mID0=mID, space_mkeys=space_mkeys, dt=c.dt, refID=refID,
                             sim_ID=mID, save_to=save_to, **kwargs)
        return entry

    def adapt_6mIDs(self, refID, e=None, c=None):
        if e is None or c is None:
            d = preg.loadRef(refID)
            d.load(step=False)
            e, c = d.endpoint_data, d.config

        from lib.sim.ga.functions import GA_optimization
        fit_kws = {
            'eval_metrics': {
                'angular kinematics': ['b', 'fov', 'foa'],
                'spatial displacement': ['v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a',
                                         'dsp_0_40_max', 'dsp_0_60_max'],
                'temporal dynamics': ['fsv', 'ffov', 'run_tr', 'pau_tr'],
            },
            'cycle_curves': ['fov', 'foa', 'b']
        }

        fit_dict = GA_optimization(refID, fitness_target_kws=fit_kws)
        entries = {}
        mIDs = []
        for Tmod in ['NEU', 'SIN']:
            for Ifmod in ['PHI', 'SQ', 'DEF']:
                mID0 = f'RE_{Tmod}_{Ifmod}_DEF'
                mID = f'{Ifmod}on{Tmod}'
                entry = self.adapt_mID(refID=refID, mID0=mID0, mID=mID, e=e, c=c,
                                       space_mkeys=['turner', 'interference'],
                                       fit_dict=fit_dict)
                entries.update(entry)
                mIDs.append(mID)
        return entries, mIDs

    def adapt_3modules(self, refID, e=None, c=None):
        if e is None or c is None:
            d = preg.loadRef(refID)
            d.load(step=False)
            e, c = d.endpoint_data, d.config

        from lib.sim.ga.functions import GA_optimization
        fit_kws = {
            'eval_metrics': {
                'angular kinematics': ['b', 'fov', 'foa'],
                'spatial displacement': ['v_mu', 'pau_v_mu', 'run_v_mu', 'v', 'a',
                                         'dsp_0_40_max', 'dsp_0_60_max'],
                'temporal dynamics': ['fsv', 'ffov', 'run_tr', 'pau_tr'],
            },
            'cycle_curves': ['fov', 'foa', 'b']
        }

        fit_dict = GA_optimization(refID, fitness_target_kws=fit_kws)
        entries = {}
        mIDs = []
        # for Cmod in ['GAU', 'CON']:
        for Cmod in ['RE', 'SQ', 'GAU', 'CON']:
            for Tmod in ['NEU', 'SIN', 'CON']:
                for Ifmod in ['PHI', 'SQ', 'DEF']:
                    mID0 = f'{Cmod}_{Tmod}_{Ifmod}_DEF'
                    mID = f'{mID0}_fit'
                    entry = self.adapt_mID(refID=refID, mID0=mID0, mID=mID, e=e, c=c,
                                           space_mkeys=['crawler', 'turner', 'interference'],
                                           fit_dict=fit_dict)
                    entries.update(entry)
                    mIDs.append(mID)
        return entries, mIDs

    def add_var_mIDs(self, refID, e=None, c=None, mID0s=None, mIDs=None, sample_ks=None):
        if e is None or c is None:
            d = preg.loadRef(refID)
            d.load(step=False)
            e, c = d.endpoint_data, d.config

        if mID0s is None:
            mID0s = list(c.modelConfs.average.keys())
        if mIDs is None:
            mIDs = [f'{mID0}_var' for mID0 in mID0s]
        if sample_ks is None:
            sample_ks = [
                'brain.crawler_params.stride_dst_mean',
                'brain.crawler_params.stride_dst_std',
                'brain.crawler_params.max_scaled_vel',
                'brain.crawler_params.max_vel_phase',
                'brain.crawler_params.initial_freq',
            ]
        kwargs = {k: 'sample' for k in sample_ks}
        entries = {}
        for mID0, mID in zip(mID0s, mIDs):
            m0 = dNl.copyDict(self.loadConf(mID0))
            m = dNl.update_existingnestdict(m0, kwargs)
            self.saveConf(conf=m, mID=mID)
            entries[mID] = m
        return entries

    def update_mdict(self, mdict, mmdic):
        if mmdic is None:
            return None
        else:
            for d, p in mdict.items():
                new_v = mmdic[d] if d in mmdic.keys() else None
                if isinstance(p, param.Parameterized):
                    if type(new_v) == list:
                        if p.parclass in [param.Range, param.NumericTuple, param.Tuple]:
                            new_v = tuple(new_v)
                    p.v = new_v
                else:
                    mdict[d]=self.update_mdict(mdict=p, mmdic=new_v)
            return mdict

    def variable_keys(self, mkey, mode='default'):
        d0 = self.dict.model.init[mkey]
        var_ks = d0.mode[mode].variable
        return var_ks

    def variable_mdict(self, mkey, mode='default'):
        var_ks = self.variable_keys(mkey, mode=mode)
        d00 = self.dict.model.m[mkey].mode[mode].args
        mdict = dNl.NestDict({k: d00[k] for k in var_ks})
        return mdict

    def space_dict(self, mkeys, mConf0):
        mF = dNl.flatten_dict(mConf0)
        dic = {}
        for mkey in mkeys:
            d0 = self.dict.model.init[mkey]
            if f'{d0.pref}mode' in mF.keys():
                mod_v = mF[f'{d0.pref}mode']
            else:
                mod_v = 'default'
            var_mdict = self.variable_mdict(mkey, mode=mod_v)
            for k, p in var_mdict.items():

                k0 = f'{d0.pref}{k}'

                if k0 in mF.keys():
                    dic[k0] = p
                    if type(mF[k0]) == list:
                        if dic[k0].parclass == param.Range:
                            mF[k0] = tuple(mF[k0])
                    dic[k0].v = mF[k0]
        return dNl.NestDict(dic)

    def to_string(self, mdict):
        s = ''
        for k, p in mdict.items():
            s = s + f'{p.d} : {p.v}'
        return s


def epar(e, k=None, par=None, average=True, Nround=2):
    if par is None:
        D = preg.dict
        par = D[k].d
    vs = e[par]
    if average:
        # print(k,par, np.round(vs.median(), Nround))
        return np.round(vs.median(), Nround)
    else:
        return vs


# larva_conf_dict=LarvaConfDict()

#
# if __name__ == '__main__':
#     LM=larva_conf_dict
#     mID='RE_NEU_PHI_DEF_nav'
#     m=LM.loadConf(mID)
#     ol=m.brain.olfactor_params
#     print(ol)
#
#
