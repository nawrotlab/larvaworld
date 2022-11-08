import copy
import json
import os
import pickle
import shutil

import numpy as np
import pandas
import param

import lib.aux.dictsNlists as dNl
from lib.aux.data_aux import update_mdict, update_existing_mdict, get_ks
from lib.registry.base import BaseType
from lib.aux import colsNstr as cNs
# from lib.registry.pars import preg
from lib.registry import reg




class GroupType(BaseType):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_dict0(reg.PI.dict[self.k])



    def expand_mdict(self, m0=None):
        if m0 is  None:
            if self.mdict is None:
                return None
            else :
                m0=self.mdict

        def retrieve(p,ct):
            conf = None
            if p in ct.ConfIDs:
                # print(v)
                conf = ct.loadConf(p)
            elif isinstance(p, param.Parameterized):
                if p.v in ct.ConfIDs:
                    conf = ct.loadConf(p.v)
            if conf is not None:
                mm = copy.deepcopy(ct.mdict)
                # print(m, conf)

                mm = update_existing_mdict(mm, conf)

                return mm
            else :
                return ct.mdict


        for subID, subk in self.subks.items():
            if self.parent.dict[subk].mdict is None :
                continue

            if subID == 'larva_groups' and subk == 'Model':

                for k,dic in m0[subID].v.items():
                    if 'model' in dic.keys():
                        p=dic.model
                        mm=retrieve(p,self.parent.dict['Model'])
                        dic.model=mm

            else:
                # ct=CT[subk]
                # mm=copy.deepcopy(ct.mdict)
                # p = m0[subID]
                m0[subID] = retrieve(m0[subID], self.parent.dict[subk])
                # m0[subID]=mm
        return m0








    def ConfID_entry(self, default=None, k=None, symbol=None, single_choice=True):
        from typing import List
        from lib.aux.par_aux import sub
        low = self.k.lower()
        if single_choice:
            t = str
            IDstr = 'ID'
        else:
            t = List[str]
            IDstr = 'IDs'
        if k is None:
            k = f'{low}{IDstr}'
        if symbol is None:
            symbol = sub(IDstr, low)
        d = {'dtype': t, 'vparfunc': self.ConfSelector(default=default, single_choice=single_choice),
             'vs': self.ConfIDs, 'v': default,
             'symbol': symbol, 'k': k, 'h': f'The {self.k} configuration {IDstr}',
             'disp': f'{self.k} {IDstr}'}
        return dNl.NestDict(d)









    def checkDict(self):
        d = self.loadDict()
        eval = {}
        for id, conf in d.items():
            try:
                eval[id] =update_mdict(self.mdict, conf)
            except:
                eval[id] = None
        return eval


    def RvsS_groups(self,N=1, age=72.0, q=1.0, h_starved=0.0, sample='None.150controls', substrate_type='standard',pref='',
                    navigator=False, expand=False, **kwargs):
        if age==0.0 :
            epochs={}
        else :
            if h_starved==0:
                eps={
                    0 : {'start': 0.0, 'stop' : age, 'substate':{'type': substrate_type, 'quality': q}}
                }
            else :
                eps = {
                    0: {'start': 0.0, 'stop': age-h_starved, 'substate': {'type': substrate_type, 'quality': q}},
                    1: {'start': age-h_starved, 'stop': age, 'substate': {'type': substrate_type, 'quality': 0}},
                }
            epochs={}
            for id,kws in eps.items():
                epochs.update(self.GT.dict.epoch.entry(id=id,**kws))




        kws0 = {
            'kwdic': {
                'distribution': {'N': N, 'scale': (0.005, 0.005)},
                'life_history': {'age': age,
                                 'epochs': epochs,
                                 },
            'odor':{}
            },
            'sample': sample,
        }

        mcols = ['blue', 'red']
        mID0s = ['rover', 'sitter']
        lgs = {}
        for mID0, mcol in zip(mID0s, mcols):
            id=f'{pref}{mID0.capitalize()}'



            if navigator :
                mID0=f'navigator_{mID0}'
            if expand:
                mID0=reg.CT.dict.Model.loadConf(mID0)



            kws = {
                'default_color': mcol,
                'model': mID0,
                **kws0
            }

            lgs.update(self.entry(id, **kws))
        return dNl.NestDict(lgs)

    def lg_entry(self, id=None, c='black', N=1, mode='uniform', sh='circle', loc=(0.0, 0.0), ors=(0.0, 360.0),
           s=(0.0, 0.0), mID='explorer',age=0.0, epochs={},  o=None,sample = None, expand=False, **kwargs):
        if id is None :
            id=mID
        m=mID if not expand else reg.CT.dict.Model.loadConf(mID)
        if type(s) == float:
            s = (s, s)
        kws = {'kwdic': {
            'distribution': {'N': N, 'scale': s, 'orientation_range': ors, 'loc': loc, 'shape': sh, 'mode': mode},
            'life_history': {'age': age,'epochs': epochs}
        },
               'default_color': c, 'model': m,'sample':sample,  **kwargs}
        if o is not None:
            kws['odor'] = o

        return self.entry(id=id, **kws)

    def lgs(self, mIDs, ids=None, cs=None,**kwargs):

        if ids is None:
            ids = mIDs
        N = len(mIDs)
        if cs is None :
            cs = cNs.N_colors(N)
        return dNl.NestDict(dNl.merge_dicts([self.lg_entry(id, c=c, mID=mID, **kwargs) for mID, c, id in zip(mIDs, cs, ids)]))



class GroupTypeDict:
    def __init__(self,load=False, save=False):



        reg.vprint('started GroupTypes',0)
        self.grouptypes = ['LarvaGroup', 'SourceGroup', 'epoch']

        self.dict = self.build(self.grouptypes)

        reg.vprint('completed GroupTypes',0)

    def build_subk_dict(self, ks):
        d0 = dNl.NestDict({k: {} for k in ks})
        d1 = dNl.NestDict({
            'LarvaGroup': {'Model'},
            # 'Ga': {'env_params': 'Env'},
            # 'Exp': {'env_params': 'Env',
            #         'trials': 'Trial',
            #         'larva_groups': 'Model',
            #         }
        })
        d0.update(d1)
        return d0



    def build(self, ks):

        self.subk_dict = self.build_subk_dict(ks)

        d = dNl.NestDict({k: GroupType(k=k, subks=subks, parent=self) for k, subks in self.subk_dict.items()})

        # aa = d['Ga'].loadDict()
        # print(aa)
        # # # aa=CTs['Ga'].ConfID_entry(default='realism')
        # # # aa=CTs['Ga'].ConfID_entry(default='exploration')
        # raise
        return d


